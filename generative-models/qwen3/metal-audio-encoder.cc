#include "generative-models/qwen3/metal-audio-encoder.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"

// The make_metal_audio_encoder() adapter (bottom of this file) wraps the
// pure metal encoder in an AudioEncoder whose EncodedAudio holds an
// mlx::core::array; it is MLX-only. The MetalAudioEncoder forward above
// is MLX-free.

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param
// block the vendored steel / NAX flash-attention kernels read. Identical
// layout to the vision tower's copy.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

std::size_t
numel_(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
// conv recurrence for kernel=3, stride=2, padding=1.
inline int
conv_dim_(int L)
{
  return (L - 1) / 2 + 1;
}
}  // namespace

std::unique_ptr<MetalAudioEncoder>
MetalAudioEncoder::load(const std::string& model_dir,
                        metal_compute::MetalCompute* mc, const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) { return nullptr; }

  auto m = std::unique_ptr<MetalAudioEncoder>(new MetalAudioEncoder());
  m->_cfg = cfg;
  m->_mc = mc;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_vis = mc->load_library("qwen3_5_vision");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_lib_conv = mc->load_library("audio_encoder");
  m->_fn_gemm = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_ln = m->_lib_vis.function("layer_norm_bias_f16");
  m->_fn_gelu_erf = m->_lib_vis.function("gelu_erf_f16");
  m->_fn_conv = m->_lib_conv.function("audio_conv2d_3x3_s2p1_f16");
  m->_fn_im2col = m->_lib_conv.function("audio_im2col_3x3_s2p1_f16");
  m->_fn_swap_last2 = m->_lib_conv.function("swap_last2_f16");
  m->_fn_sdpa_window = m->_lib_sdpa.function("sdpa_window_f16");
  m->_fn_head_slice = m->_lib_elt.function("head_slice_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_ln.valid() || !m->_fn_gelu_erf.valid() ||
      !m->_fn_conv.valid() || !m->_fn_sdpa_window.valid() ||
      !m->_fn_head_slice.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_residual.valid()) {
    return nullptr;
  }
  // M5-only matrix-core dense GEMM (matmul2d/NAX) -- the same NAX path the LM
  // prefill and the Gemma ViT/audio encoders use. Gated on
  // supports_matrix_cores() (GPUFamilyApple10+) so M4 / older GPUs never load
  // it and keep the steel dense_gemm_t (byte-identical, performance
  // unchanged). The encoder is GEMM-dominated (24 blocks: qkv/o/fc1/fc2 +
  // conv_out + the im2col conv stem); the windowed attention is a separate
  // scalar kernel (no NAX-attn lever -- it is block-local, not O(n^2)).
  // VPIPE_ASR_NO_MMA2=1 forces the steel path even on M5 (A/B + safety).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_ASR_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() &&
                   m->_fn_dense_mma_deep.valid() && m->_fn_bias_add.valid();
  }
  // M5 NAX steel full attention for the short-clip case (seq <= window). The
  // bd64 entry point matches the ASR head_dim (1024/16 == 64). Gated
  // independently of the matmul2d GEMM above so the two levers A/B
  // orthogonally. The scalar windowed kernel stays for seq > window (and on
  // M4, byte-identical). VPIPE_ASR_NO_NAX_ATTN=1 reverts.
  if (mc->supports_matrix_cores() && m->_cfg.head_dim() == 64 &&
      std::getenv("VPIPE_ASR_NO_NAX_ATTN") == nullptr) {
    m->_lib_attn_nax = mc->load_library("attn_steel_nax");
    m->_use_attn_nax = m->_lib_attn_nax.valid();
    if (m->_use_attn_nax) {
      m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
      m->_attn_params_last = mc->make_shared_buffer(sizeof(SteelAttnParams));
    }
  }

  auto to_f16 = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts->info(name);
    if (info == nullptr) { return {}; }
    if (info->dtype == "F16") { return wts->load(name, mc); }
    SharedBuffer raw = wts->load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<_Float16*>(out.contents());
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)bf16_to_f32_(s[i]); }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)s[i]; }
    } else {
      return {};
    }
    return out;
  };
  // Row-concatenate q|k|v (each [d_model, d_model]) into one [3*d, d]
  // weight and [3*d] bias -- one fused GEMM replaces three.
  auto fuse3 = [&](const std::string& a, const std::string& b,
                   const std::string& c, SharedBuffer& w,
                   SharedBuffer& bias) -> bool {
    SharedBuffer wa = to_f16(a + ".weight"), wb = to_f16(b + ".weight"),
                 wc = to_f16(c + ".weight");
    SharedBuffer ba = to_f16(a + ".bias"), bb = to_f16(b + ".bias"),
                 bc = to_f16(c + ".bias");
    if (wa.empty() || wb.empty() || wc.empty() || ba.empty() || bb.empty() ||
        bc.empty()) {
      return false;
    }
    w = mc->make_shared_buffer(wa.byte_size() + wb.byte_size() + wc.byte_size());
    bias = mc->make_shared_buffer(ba.byte_size() + bb.byte_size() +
                                  bc.byte_size());
    char* wp = static_cast<char*>(w.contents());
    std::memcpy(wp, wa.contents(), wa.byte_size());
    std::memcpy(wp + wa.byte_size(), wb.contents(), wb.byte_size());
    std::memcpy(wp + wa.byte_size() + wb.byte_size(), wc.contents(),
                wc.byte_size());
    char* bp = static_cast<char*>(bias.contents());
    std::memcpy(bp, ba.contents(), ba.byte_size());
    std::memcpy(bp + ba.byte_size(), bb.contents(), bb.byte_size());
    std::memcpy(bp + ba.byte_size() + bb.byte_size(), bc.contents(),
                bc.byte_size());
    return true;
  };

  const std::string r = "audio_tower.";
  m->_c1w = to_f16(r + "conv2d1.weight");
  m->_c1b = to_f16(r + "conv2d1.bias");
  m->_c2w = to_f16(r + "conv2d2.weight");
  m->_c2b = to_f16(r + "conv2d2.bias");
  m->_c3w = to_f16(r + "conv2d3.weight");
  m->_c3b = to_f16(r + "conv2d3.bias");
  m->_convout_w = to_f16(r + "conv_out.weight");
  m->_ln_post_w = to_f16(r + "ln_post.weight");
  m->_ln_post_b = to_f16(r + "ln_post.bias");
  m->_proj1_w = to_f16(r + "proj1.weight");
  m->_proj1_b = to_f16(r + "proj1.bias");
  m->_proj2_w = to_f16(r + "proj2.weight");
  m->_proj2_b = to_f16(r + "proj2.bias");
  bool ok = !m->_c1w.empty() && !m->_c1b.empty() && !m->_c2w.empty() &&
            !m->_c2b.empty() && !m->_c3w.empty() && !m->_c3b.empty() &&
            !m->_convout_w.empty() && !m->_ln_post_w.empty() &&
            !m->_ln_post_b.empty() && !m->_proj1_w.empty() &&
            !m->_proj2_w.empty();

  m->_blocks.resize(cfg.n_layers);
  for (int b = 0; b < cfg.n_layers; ++b) {
    const std::string p = r + "layers." + std::to_string(b) + ".";
    Block& blk = m->_blocks[b];
    blk.n1w = to_f16(p + "self_attn_layer_norm.weight");
    blk.n1b = to_f16(p + "self_attn_layer_norm.bias");
    ok = ok && fuse3(p + "self_attn.q_proj", p + "self_attn.k_proj",
                     p + "self_attn.v_proj", blk.qkvw, blk.qkvb);
    blk.ow = to_f16(p + "self_attn.out_proj.weight");
    blk.ob = to_f16(p + "self_attn.out_proj.bias");
    blk.n2w = to_f16(p + "final_layer_norm.weight");
    blk.n2b = to_f16(p + "final_layer_norm.bias");
    blk.fc1w = to_f16(p + "fc1.weight");
    blk.fc1b = to_f16(p + "fc1.bias");
    blk.fc2w = to_f16(p + "fc2.weight");
    blk.fc2b = to_f16(p + "fc2.bias");
    ok = ok && !blk.n1w.empty() && !blk.ow.empty() && !blk.ob.empty() &&
         !blk.n2w.empty() && !blk.fc1w.empty() && !blk.fc2w.empty();
  }
  if (!ok) { return nullptr; }

  WhisperFeatureExtractor::Params fp;
  fp.n_fft = 400;
  fp.hop_length = 160;
  fp.n_mel_bins = cfg.n_mel;
  fp.sample_rate = cfg.sample_rate;
  fp.n_samples_target = 0;
  fp.f_min = 0.0f;
  fp.f_max = (float)cfg.sample_rate / 2.0f;
  m->_fx = std::make_unique<WhisperFeatureExtractor>(fp);
  return m;
}

MetalAudioEncoder::Result
MetalAudioEncoder::encode(const float* pcm, std::size_t n_samples,
                          int sample_rate)
{
  Result res;
  const Config& c = _cfg;
  if (pcm == nullptr || n_samples == 0 || sample_rate != c.sample_rate) {
    return res;
  }

  // 1. CPU log-mel: [n_mel, T] mel-major (out[m*T + t]).
  std::vector<float> mel;
  const int T = (int)_fx->extract(pcm, n_samples, &mel);
  if (T <= 0) { return res; }
  const int nm = c.n_mel;

  // 2. Chunk geometry (per-chunk conv matches training).
  const int cmel = c.chunk_mel();                 // 100
  const int n_chunks = (T + cmel - 1) / cmel;
  const int last_len = (T % cmel == 0) ? cmel : (T % cmel);
  const int H1 = nm, W1 = cmel;
  const int H1o = conv_dim_(H1), W1o = conv_dim_(W1);     // 64, 50
  const int H2o = conv_dim_(H1o), W2o = conv_dim_(W1o);   // 32, 25
  const int H3o = conv_dim_(H2o), W3o = conv_dim_(W2o);   // 16, 13
  const int Tp = W3o;                                     // tokens / chunk
  const int Fp = H3o;                                     // freq after stem
  const int CH = c.conv_hidden;                           // 480
  const int K_convout = CH * Fp;                          // 7680
  const int d = c.d_model;

  auto vfor = [](int L) {
    int v = conv_dim_(L); v = conv_dim_(v); return conv_dim_(v);
  };
  const int last_valid = std::min(Tp, vfor(last_len));
  const int seq = (n_chunks - 1) * Tp + last_valid;
  if (seq <= 0) { return res; }

  // 3. Host conv input [n_chunks, n_mel, cmel, 1] f16 (zero-padded tail).
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer conv_in = buf((std::size_t)n_chunks * nm * cmel);
  {
    auto* ip = static_cast<_Float16*>(conv_in.contents());
    for (int ch = 0; ch < n_chunks; ++ch) {
      for (int mm = 0; mm < nm; ++mm) {
        for (int t = 0; t < cmel; ++t) {
          const int gt = ch * cmel + t;
          const float v = (gt < T) ? mel[(std::size_t)mm * T + gt] : 0.0f;
          ip[((std::size_t)ch * nm + mm) * cmel + t] = (_Float16)v;
        }
      }
    }
  }

  // Per-chunk sinusoidal position embedding [Tp, d], tiled to [n_chunks*Tp, d].
  SharedBuffer posb = buf((std::size_t)n_chunks * Tp * d);
  {
    std::vector<float> inv_ts(d / 2);
    const float inc = std::log(10000.0f) / (float)(d / 2 - 1);
    for (int i = 0; i < d / 2; ++i) { inv_ts[i] = std::exp(-inc * (float)i); }
    auto* pp = static_cast<_Float16*>(posb.contents());
    for (int ch = 0; ch < n_chunks; ++ch) {
      for (int t = 0; t < Tp; ++t) {
        _Float16* row = pp + ((std::size_t)ch * Tp + t) * d;
        for (int i = 0; i < d / 2; ++i) {
          const float s = (float)t * inv_ts[i];
          row[i] = (_Float16)std::sin(s);
          row[d / 2 + i] = (_Float16)std::cos(s);
        }
      }
    }
  }

  const int rows = n_chunks * Tp;                 // pre-slice token rows
  const int heads = c.n_heads, hd = c.head_dim(), inter = c.ffn;
  const int outd = c.output_dim;
  const float eps = c.ln_eps;
  const float scale = 1.0f / std::sqrt((float)hd);
  const int window = c.window_tokens();

  // Conv intermediates + encoder scratch.
  SharedBuffer c1 = buf((std::size_t)n_chunks * H1o * W1o * CH);
  SharedBuffer c2 = buf((std::size_t)n_chunks * H2o * W2o * CH);
  SharedBuffer c3 = buf((std::size_t)n_chunks * Tp * K_convout);  // transposed
  SharedBuffer x = buf((std::size_t)rows * d);
  SharedBuffer n1 = buf((std::size_t)seq * d);
  SharedBuffer qkv = buf((std::size_t)seq * 3 * d);
  SharedBuffer q3 = buf((std::size_t)seq * d), k3 = buf((std::size_t)seq * d),
               v3 = buf((std::size_t)seq * d);
  SharedBuffer qt = buf((std::size_t)seq * d), kt = buf((std::size_t)seq * d),
               vt = buf((std::size_t)seq * d);
  SharedBuffer atb = buf((std::size_t)seq * d), att = buf((std::size_t)seq * d);
  SharedBuffer proj = buf((std::size_t)seq * d);
  SharedBuffer hbuf = buf((std::size_t)seq * inter);
  SharedBuffer out2 = buf((std::size_t)seq * d);
  SharedBuffer pj1 = buf((std::size_t)seq * d);
  SharedBuffer emb = buf((std::size_t)seq * outd);

  // im2col / transpose scratch for the MMA conv path. Must outlive
  // commit().wait() (the command buffer references them), so they live
  // here, not inside the conv lambda. Reserved so push_back never
  // reallocates (which would move buffers mid-encode). Up to 3: conv2 col,
  // conv3 col, conv3 tmp.
  std::vector<SharedBuffer> conv_scratch;
  conv_scratch.reserve(6);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    const bool use_mma_gemm = _fn_gemm_t.valid();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& bias, const SharedBuffer& y, int M,
                    int N, int Kk, int has_bias) {
      if (_use_mma2) {
        // M5 matrix-core matmul2d GEMM y = x @ w^T (no bias); fold the linear
        // bias over the rows after. The matmul2d tensor extents clamp the M/N
        // tails so M and N need not be tile multiples. 128x256 tile for deep K
        // (K>=6144, e.g. conv_out's K=7680); else the n128 tile.
        const bool deep = (Kk >= 6144);
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);     // bias slot unused (has_bias=0)
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        if (has_bias) {
          enc.set_function(_fn_bias_add);
          enc.set_buffer(0, y);
          enc.set_buffer(1, bias);
          enc.set_constant(2, N);
          enc.set_constant(3, M * N);
          enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
        }
      } else if (use_mma_gemm) {
        // Steel MMA GEMM y = x @ w^T (+bias when has_bias); bias folded in.
        enc.set_function(_fn_gemm_t);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, has_bias);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      } else {
        enc.set_function(_fn_gemm);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, y);
        enc.set_constant(4, M);
        enc.set_constant(5, N);
        enc.set_constant(6, Kk);
        enc.set_constant(7, has_bias);
        const unsigned gx = (unsigned)(((N + 15) / 16) * 16);
        const unsigned gy = (unsigned)(((M + 15) / 16) * 16);
        enc.dispatch({gx, gy, 1}, {16, 16, 1});
      }
    };
    auto gelu = [&](const SharedBuffer& xin, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu_erf);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto ln = [&](const SharedBuffer& xin, const SharedBuffer& w,
                  const SharedBuffer& b, const SharedBuffer& y, int R, int Hd) {
      enc.set_function(_fn_ln);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, Hd);
      enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out, int Hh,
                      int Sd, int Wd, int off) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, Hh);
      enc.set_constant(3, Sd);
      enc.set_constant(4, Wd);
      enc.set_constant(5, off);
      const int zero = 0;
      enc.set_constant(6, zero);
      enc.set_constant(7, zero);
      enc.dispatch({(unsigned)(Hh * Wd), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, hd);
      enc.dispatch({(unsigned)hd, (unsigned)Bd, (unsigned)A}, {(unsigned)hd, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto conv = [&](const SharedBuffer& in, const SharedBuffer& w,
                    const SharedBuffer& bias, const SharedBuffer& out, int Cin,
                    int Cout, int Hh, int Ww, int Ho, int Wo, int tpose) {
      const int M = n_chunks * Ho * Wo;
      const int K9 = 9 * Cin;
      const std::size_t col_elems = (std::size_t)M * (std::size_t)K9;
      // im2col + steel MMA GEMM when worthwhile (large Cin, K%32==0) and
      // the transient col buffer stays within budget; else scalar direct
      // conv (always correct -- e.g. conv1 has Cin=1, and very long clips
      // exceed the col budget).
      const bool use_im2col = use_mma_gemm && _fn_im2col.valid() &&
          _fn_swap_last2.valid() && Cin >= 32 && (K9 % 32) == 0 &&
          col_elems * 2 <= (std::size_t)768 * 1024 * 1024;
      if (use_im2col) {
        conv_scratch.push_back(buf(col_elems));
        const SharedBuffer& col = conv_scratch.back();
        enc.set_function(_fn_im2col);
        enc.set_buffer(0, in);
        enc.set_buffer(1, col);
        enc.set_constant(2, n_chunks);
        enc.set_constant(3, Hh);
        enc.set_constant(4, Ww);
        enc.set_constant(5, Cin);
        enc.set_constant(6, Ho);
        enc.set_constant(7, Wo);
        enc.set_constant(8, tpose ? 0 : 1);   // row_major_hw
        enc.dispatch({(unsigned)(M * 9), 1, 1}, {256, 1, 1});
        if (tpose == 0) {
          gemm(col, w, bias, out, M, Cout, K9, 1);
        } else {
          // GEMM -> [N,Wo,Ho,Cout]; swap (Ho,Cout) -> [N,Wo,Cout,Ho].
          conv_scratch.push_back(buf((std::size_t)M * (std::size_t)Cout));
          const SharedBuffer& tmp = conv_scratch.back();
          gemm(col, w, bias, tmp, M, Cout, K9, 1);
          enc.set_function(_fn_swap_last2);
          enc.set_buffer(0, tmp);
          enc.set_buffer(1, out);
          enc.set_constant(2, n_chunks * Wo);   // batch
          enc.set_constant(3, Ho);              // A
          enc.set_constant(4, Cout);            // B
          enc.dispatch({(unsigned)(M * Cout), 1, 1}, {256, 1, 1});
        }
      } else {
        enc.set_function(_fn_conv);
        enc.set_buffer(0, in);
        enc.set_buffer(1, w);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, out);
        enc.set_constant(4, n_chunks);
        enc.set_constant(5, Hh);
        enc.set_constant(6, Ww);
        enc.set_constant(7, Cin);
        enc.set_constant(8, Cout);
        enc.set_constant(9, Ho);
        enc.set_constant(10, Wo);
        enc.set_constant(11, tpose);
        const unsigned gx = (unsigned)(((Cout + 31) / 32) * 32);
        enc.dispatch({gx, (unsigned)Wo, (unsigned)(n_chunks * Ho)}, {32, 1, 1});
      }
    };

    // Conv stem: 3x (conv + GELU-erf). conv3 writes transposed
    // [n_chunks, Tp, Cout, Fp] so conv_out reads contiguous K=Cout*Fp.
    conv(conv_in, _c1w, _c1b, c1, 1, CH, H1, W1, H1o, W1o, 0);
    gelu(c1, c1, n_chunks * H1o * W1o * CH);
    conv(c1, _c2w, _c2b, c2, CH, CH, H1o, W1o, H2o, W2o, 0);
    gelu(c2, c2, n_chunks * H2o * W2o * CH);
    conv(c2, _c3w, _c3b, c3, CH, CH, H2o, W2o, H3o, W3o, 1);
    gelu(c3, c3, n_chunks * Tp * K_convout);

    // conv_out projection (no bias) + per-chunk sinusoidal pos.
    gemm(c3, _convout_w, _convout_w, x, rows, d, K_convout, 0);
    residual(x, posb, x, rows * d);

    // Block-windowed steel/NAX attention setup -- shared across all blocks
    // (same seq/heads/hd/window). The block-aligned windowing is block-diagonal
    // full attention over each window-block, which maps onto attn_steel_nax by
    // viewing the head-major [heads, seq, hd] qt/kt/vt as [n_blocks, heads, W,
    // hd]: batch (block) stride W*hd, head stride seq*hd. Two dispatches per
    // layer -- the n_full full blocks (B=n_full, qL=kL=W) and the partial last
    // block (B=1, qL=kL=last, bound at a byte offset). n_full==0 (seq<W)
    // reduces to a single full-attention dispatch (the short-clip case).
    const int W = window;
    const int n_full = W > 0 ? seq / W : 0;
    const int last = W > 0 ? seq - n_full * W : seq;
    const bool steel_attn = _use_attn_nax && _lib_attn_nax.valid() &&
        hd == 64 && !_attn_params.empty() && !_attn_params_last.empty();
    auto fill_attn = [&](SharedBuffer& pbuf, int bs) {
      const int a_bq = 64, a_bk = 32;
      auto* p = static_cast<SteelAttnParams*>(pbuf.contents());
      p->B = 1; p->H = heads; p->D = hd; p->qL = bs; p->kL = bs;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (bs + a_bq - 1) / a_bq; p->NK = (bs + a_bk - 1) / a_bk;
      p->NQ_aligned = bs / a_bq; p->NK_aligned = bs / a_bk;
      p->qL_rem = bs - p->NQ_aligned * a_bq;
      p->kL_rem = bs - p->NK_aligned * a_bk;
      p->qL_off = 0;
      // [n_blocks, heads, W, hd]: block-batch stride W*hd, head stride seq*hd.
      p->Q_strides[0] = (std::int64_t)W * hd;
      p->Q_strides[1] = (std::int64_t)seq * hd; p->Q_strides[2] = hd;
      p->K_strides[0] = p->Q_strides[0];
      p->K_strides[1] = p->Q_strides[1]; p->K_strides[2] = hd;
      p->V_strides[0] = p->Q_strides[0];
      p->V_strides[1] = p->Q_strides[1]; p->V_strides[2] = hd;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = hd;
    };
    auto make_fn = [&](int bs) -> metal_compute::ComputeFunction {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (bs % 64) == 0).set_bool(201, (bs % 32) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      return _lib_attn_nax.function("attn_steel_nax_h_bd64", fc);
    };
    metal_compute::ComputeFunction fn_full, fn_last;
    if (steel_attn) {
      if (n_full > 0) { fill_attn(_attn_params, W); fn_full = make_fn(W); }
      if (last > 0) {
        fill_attn(_attn_params_last, last);
        fn_last = make_fn(last);
      }
    }
    const bool use_steel = steel_attn &&
        (n_full == 0 || fn_full.valid()) && (last == 0 || fn_last.valid());

    // Encoder blocks over the first `seq` (valid) rows.
    for (int b = 0; b < c.n_layers; ++b) {
      Block& blk = _blocks[b];
      ln(x, blk.n1w, blk.n1b, n1, seq, d);
      gemm(n1, blk.qkvw, blk.qkvb, qkv, seq, 3 * d, d, 1);
      hslice(qkv, q3, seq, 3 * d, d, 0);
      hslice(qkv, k3, seq, 3 * d, d, d);
      hslice(qkv, v3, seq, 3 * d, d, 2 * d);
      transpose(q3, qt, seq, heads);   // [seq,heads,hd]->[heads,seq,hd]
      transpose(k3, kt, seq, heads);
      transpose(v3, vt, seq, heads);
      if (use_steel) {
        // Block-windowed steel/NAX: the n_full full window-blocks batched
        // (B=n_full, grid.z=n_full) + the partial last block (B=1) bound at a
        // byte offset. Q/K/V/O all [heads, seq, hd]; params set above the loop.
        if (n_full > 0) {
          enc.set_function(fn_full);
          enc.set_buffer(0, qt);
          enc.set_buffer(1, kt);
          enc.set_buffer(2, vt);
          enc.set_buffer(3, atb);
          enc.set_buffer(4, _attn_params);
          const unsigned nqb = (unsigned)((W + 63) / 64);
          enc.dispatch({32 * nqb, 4 * (unsigned)heads, (unsigned)n_full},
                       {32, 4, 1});
        }
        if (last > 0) {
          enc.set_function(fn_last);
          const std::size_t off = (std::size_t)n_full * W * hd * 2;
          enc.set_buffer(0, qt, off);
          enc.set_buffer(1, kt, off);
          enc.set_buffer(2, vt, off);
          enc.set_buffer(3, atb, off);
          enc.set_buffer(4, _attn_params_last);
          const unsigned nqb = (unsigned)((last + 63) / 64);
          enc.dispatch({32 * nqb, 4 * (unsigned)heads, 1}, {32, 4, 1});
        }
      } else {
        enc.set_function(_fn_sdpa_window);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kt);
        enc.set_buffer(2, vt);
        enc.set_buffer(3, atb);
        enc.set_constant(4, scale);
        enc.set_constant(5, seq);     // T_kv
        enc.set_constant(6, hd);
        enc.set_constant(7, heads);
        enc.set_constant(8, heads);
        enc.set_constant(9, seq);     // n_q
        enc.set_constant(10, seq);    // kv_stride
        enc.set_constant(11, window);
        enc.dispatch({32, (unsigned)heads, (unsigned)seq}, {32, 1, 1});
      }
      transpose(atb, att, heads, seq);  // [heads,seq,hd]->[seq,heads,hd]
      gemm(att, blk.ow, blk.ob, proj, seq, d, d, 1);
      residual(x, proj, x, seq * d);
      ln(x, blk.n2w, blk.n2b, n1, seq, d);
      gemm(n1, blk.fc1w, blk.fc1b, hbuf, seq, inter, d, 1);
      gelu(hbuf, hbuf, seq * inter);
      gemm(hbuf, blk.fc2w, blk.fc2b, out2, seq, d, inter, 1);
      residual(x, out2, x, seq * d);
    }

    // ln_post + proj1 + GELU + proj2.
    ln(x, _ln_post_w, _ln_post_b, n1, seq, d);
    gemm(n1, _proj1_w, _proj1_b, pj1, seq, d, d, 1);
    gelu(pj1, pj1, seq * d);
    gemm(pj1, _proj2_w, _proj2_b, emb, seq, outd, d, 1);
  }
  stream.commit().wait();

  res.embeddings.resize((std::size_t)seq * outd);
  const auto* ep = static_cast<const _Float16*>(emb.contents());
  for (std::size_t i = 0; i < res.embeddings.size(); ++i) {
    res.embeddings[i] = (float)ep[i];
  }
  res.n_tokens = seq;
  return res;
}


}  // namespace vpipe::genai
