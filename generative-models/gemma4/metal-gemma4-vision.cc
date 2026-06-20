#include "generative-models/gemma4/metal-gemma4-vision.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// Mirrors the vendored MLX steel `attention` AttnParams (attn_steel.metal).
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

// Bilinear resize + rescale (mean 0 / std 1 -> px/255), U8 [3,H,W] ->
// f32 [3,th,tw]. (Gemma resamples with bicubic; M1/M3 gate the encoder
// against a pre-sized fixture so this is an identity copy.)
void
preprocess_(const std::uint8_t* rgb, int H, int W, int th, int tw,
            float* out)
{
  if (H <= 0 || W <= 0) { return; }
  const double sy = (double)H / th, sx = (double)W / tw;
  const int pin = H * W, pout = th * tw;
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * pin;
    float* dst = out + (std::size_t)c * pout;
    for (int yy = 0; yy < th; ++yy) {
      double y = (yy + 0.5) * sy - 0.5;
      y = std::min(std::max(y, 0.0), (double)(H - 1));
      int y0 = (int)std::floor(y), y1 = std::min(y0 + 1, H - 1);
      float dy = (float)(y - y0);
      for (int xx = 0; xx < tw; ++xx) {
        double x = (xx + 0.5) * sx - 0.5;
        x = std::min(std::max(x, 0.0), (double)(W - 1));
        int x0 = (int)std::floor(x), x1 = std::min(x0 + 1, W - 1);
        float dx = (float)(x - x0);
        float v00 = src[y0 * W + x0], v01 = src[y0 * W + x1];
        float v10 = src[y1 * W + x0], v11 = src[y1 * W + x1];
        float v0 = v00 * (1 - dx) + v01 * dx;
        float v1 = v10 * (1 - dx) + v11 * dx;
        dst[yy * tw + xx] = (v0 * (1 - dy) + v1 * dy) / 255.0f;
      }
    }
  }
}

}  // namespace

std::unique_ptr<MetalGemma4VisionEncoder>
MetalGemma4VisionEncoder::load(const std::string& model_dir,
                               metal_compute::MetalCompute* mc,
                               const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) { return nullptr; }

  auto m =
      std::unique_ptr<MetalGemma4VisionEncoder>(new MetalGemma4VisionEncoder());
  m->_cfg = cfg;
  m->_mc = mc;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_vis = mc->load_library("qwen3_5_vision");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_lib_rms = mc->load_library("rms_norm");
  m->_lib_qmm = mc->load_library("affine_qmm_steel");
  // Vendored MLX steel flash attention (the same kernel the Qwen3-VL ViT
  // uses); ~10x the scalar full SDPA for the ViT's bidirectional O(n^2)
  // attention -- the encode bottleneck. head_dim==64 entry point.
  m->_lib_attn = mc->load_library("attn_steel");
  if (m->_lib_attn.valid()) {
    m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  }
  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_grope = m->_lib_vis.function("gemma_vision_rope_f16");
  m->_fn_sdpa_full = m->_lib_sdpa.function("sdpa_full_f16");
  // Non-causal MMA flash attention (D % 64 == 0) -- ~3x the scalar full SDPA
  // for the ViT's O(n^2) bidirectional attention (the encode bottleneck).
  m->_fn_sdpa_full_mma = m->_lib_sdpa.function("sdpa_full_mma_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_geglu = m->_lib_elt.function("geglu_f16");
  m->_fn_qmm = m->_lib_qmm.function("affine_qmm_steel_w4g64");
  // M5-only matrix-core dense GEMM (matmul2d/NAX) -- the same NAX path the LM
  // prefill and the Qwen ViT use. Gated on supports_matrix_cores()
  // (GPUFamilyApple10+) so M4 / older GPUs never load it and keep the steel
  // dense_gemm_t (byte-identical, performance unchanged).
  // VPIPE_GEMMA_VISION_NO_MMA2=1 forces the steel path even on M5 (A/B).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_GEMMA_VISION_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    // DEFAULT on M5: MLX's matrix-core (NAX) steel attention -- the kernel MLX
    // itself dispatches on M5 (bq=64/bk=32, register-resident softmax). ~1.2-
    // 1.7x the ALU steel and token-correct vs it (verified by the MLX-free
    // metal_vision_nax_matches_steel oracle) -- PROVIDED the NAX kernels are
    // built with deployment target 26.2 (see metal-compute CMakeLists /
    // VPIPE_NAX_MIN_OS; without it the matmul2d miscompiles ~400% off).
    // VPIPE_GEMMA_VISION_NO_NAX_ATTN=1 falls back to the ALU steel.
    if (cfg.head_dim() == 64 &&
        std::getenv("VPIPE_GEMMA_VISION_NO_NAX_ATTN") == nullptr) {
      m->_lib_attn_nax = mc->load_library("attn_steel_nax");
      m->_use_attn_nax = m->_lib_attn_nax.valid();
    }
  }
  if (!m->_fn_gemm_t.valid() || !m->_fn_rms.valid() ||
      !m->_fn_grope.valid() || !m->_fn_sdpa_full.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_residual.valid() ||
      !m->_fn_geglu.valid() || !m->_fn_qmm.valid()) {
    return nullptr;
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
      for (std::size_t i = 0; i < n; ++i) {
        o[i] = (_Float16)bf16_to_f32_(s[i]);
      }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)s[i]; }
    } else {
      return {};
    }
    return out;
  };

  const std::string r = "vision_tower.";
  m->_patch_w = to_f16(r + "patch_embedder.input_proj.weight");
  m->_pos_table = to_f16(r + "patch_embedder.position_embedding_table");
  bool ok = !m->_patch_w.empty() && !m->_pos_table.empty();

  m->_blocks.resize(cfg.depth);
  for (int b = 0; b < cfg.depth; ++b) {
    const std::string p = r + "encoder.layers." + std::to_string(b) + ".";
    Block& blk = m->_blocks[b];
    blk.in_ln       = to_f16(p + "input_layernorm.weight");
    blk.post_attn_ln = to_f16(p + "post_attention_layernorm.weight");
    blk.pre_ffn_ln  = to_f16(p + "pre_feedforward_layernorm.weight");
    blk.post_ffn_ln = to_f16(p + "post_feedforward_layernorm.weight");
    blk.qw = to_f16(p + "self_attn.q_proj.linear.weight");
    blk.kw = to_f16(p + "self_attn.k_proj.linear.weight");
    blk.vw = to_f16(p + "self_attn.v_proj.linear.weight");
    blk.ow = to_f16(p + "self_attn.o_proj.linear.weight");
    blk.qn = to_f16(p + "self_attn.q_norm.weight");
    blk.kn = to_f16(p + "self_attn.k_norm.weight");
    blk.gw = to_f16(p + "mlp.gate_proj.linear.weight");
    blk.uw = to_f16(p + "mlp.up_proj.linear.weight");
    blk.dw = to_f16(p + "mlp.down_proj.linear.weight");
    ok = ok && !blk.in_ln.empty() && !blk.qw.empty() && !blk.kw.empty() &&
         !blk.vw.empty() && !blk.ow.empty() && !blk.qn.empty() &&
         !blk.kn.empty() && !blk.gw.empty() && !blk.uw.empty() &&
         !blk.dw.empty();
  }

  // embed_vision: 4-bit affine projection (weight u32 raw; scales/biases
  // bf16 -> f16).
  m->_ev_w = wts->load("embed_vision.embedding_projection.weight", mc);
  m->_ev_s = to_f16("embed_vision.embedding_projection.scales");
  m->_ev_b = to_f16("embed_vision.embedding_projection.biases");
  ok = ok && !m->_ev_w.empty() && !m->_ev_s.empty() && !m->_ev_b.empty();
  if (!ok) { return nullptr; }

  // Weightless-RMSNorm ones (v_norm over head_dim, embed_vision pre-proj
  // norm over hidden) + a zero bias for the bias-free dense GEMMs.
  const int hd = cfg.head_dim();
  m->_ones_hd = mc->make_shared_buffer((std::size_t)hd * 2);
  m->_ones_hid = mc->make_shared_buffer((std::size_t)cfg.hidden * 2);
  const int zb = std::max({cfg.hidden, cfg.intermediate, cfg.out_hidden});
  m->_zero_bias = mc->make_shared_buffer((std::size_t)zb * 2);
  {
    auto* a = static_cast<_Float16*>(m->_ones_hd.contents());
    for (int i = 0; i < hd; ++i) { a[i] = (_Float16)1.0f; }
    auto* c = static_cast<_Float16*>(m->_ones_hid.contents());
    for (int i = 0; i < cfg.hidden; ++i) { c[i] = (_Float16)1.0f; }
    std::memset(m->_zero_bias.contents(), 0, (std::size_t)zb * 2);
  }
  return m;
}

MetalGemma4VisionEncoder::Config
MetalGemma4VisionEncoder::config_from(const ModelConfig& c)
{
  Config m;
  const auto& v = c.vision;
  m.depth        = v.depth;
  m.hidden       = v.hidden_size;
  m.n_heads      = v.num_heads;
  m.patch_size   = v.patch_size;
  m.pool_kernel  = v.pooling_kernel_size;
  m.out_length   = v.default_output_length;
  m.video_out_length = v.video_default_output_length > 0
      ? v.video_default_output_length : 70;
  m.pos_size     = v.position_embedding_size;
  m.intermediate = v.intermediate_size;
  m.out_hidden   = v.out_hidden_size;
  m.rope_theta   = v.vit_rope_theta;
  m.group_size   = c.quantization.group_size;
  m.bits         = c.quantization.bits;
  return m;
}

MetalGemma4VisionEncoder::Result
MetalGemma4VisionEncoder::encode(const std::uint8_t* rgb, int H, int W,
                                 int max_soft_tokens)
{
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmVision,
                     kPerfLlmVisionBegin, 1);
  using Clock = std::chrono::steady_clock;
  const bool profile = std::getenv("VPIPE_VISION_PROFILE") != nullptr;
  const bool skip_attn = std::getenv("VPIPE_VISION_SKIP_ATTN") != nullptr;
  const auto t_enter = Clock::now();
  const Config& c = _cfg;
  const int P = c.patch_size, K = c.pool_kernel;
  const int hidden = c.hidden, heads = c.n_heads, hd = c.head_dim();
  const int inter = c.intermediate, outh = c.out_hidden;
  const float eps = c.rms_eps;
  Result res;

  // Aspect-preserving resize geometry (<= budget*K^2 patches, edges
  // divisible by K*P). `max_soft_tokens` > 0 overrides the still budget
  // for video frames.
  const int side = K * P;
  const int budget = (max_soft_tokens > 0) ? max_soft_tokens : c.out_length;
  const int max_patches = budget * K * K;
  const double target_px = (double)max_patches * P * P;
  const double f = std::sqrt(target_px / ((double)H * W));
  int th = (int)std::floor(f * H / side) * side;
  int tw = (int)std::floor(f * W / side) * side;
  const int max_side = (max_patches / (K * K)) * side;
  if (th == 0) { th = side; }
  if (tw == 0) { tw = side; }
  th = std::min(th, max_side);
  tw = std::min(tw, max_side);

  const int pH = th / P, pW = tw / P;
  const int n = pH * pW;
  const int mh = pH / K, mw = pW / K;
  const int n_im = mh * mw;
  res.n_tokens = n_im;
  res.grid_h = mh;
  res.grid_w = mw;
  res.out_hidden = outh;

  std::vector<float> px((std::size_t)3 * th * tw);
  preprocess_(rgb, H, W, th, tw, px.data());

  // Raster-order patchify: feature [n, 3*P*P], inner order (p_h,p_w,c),
  // pixels 2*(x-0.5). Patch k = row*pW + col.
  const int feat_dim = 3 * P * P;
  std::vector<_Float16> feat((std::size_t)n * feat_dim);
  const int plane = th * tw;
  for (int i = 0; i < pH; ++i) {
    for (int j = 0; j < pW; ++j) {
      _Float16* fp = feat.data() + (std::size_t)(i * pW + j) * feat_dim;
      int o = 0;
      for (int ph = 0; ph < P; ++ph) {
        for (int pw = 0; pw < P; ++pw) {
          const int yy = i * P + ph, xx = j * P + pw;
          for (int ch = 0; ch < 3; ++ch) {
            const float v = px[(std::size_t)ch * plane + yy * tw + xx];
            fp[o++] = (_Float16)(2.0f * (v - 0.5f));
          }
        }
      }
    }
  }

  // Learned 2-axis position embeds (host gather): pos[k] =
  // table[0,col] + table[1,row]. table is [2, pos_size, hidden].
  const auto* pos_tab = static_cast<const _Float16*>(_pos_table.contents());
  const std::size_t axis_stride = (std::size_t)c.pos_size * hidden;
  std::vector<_Float16> pos((std::size_t)n * hidden);
  // Gemma 2-D RoPE cos/sin tables [n, hd]: first half col freqs, second
  // half row freqs (theta^(-jf/(hd/4))).
  std::vector<_Float16> cosb((std::size_t)n * hd), sinb((std::size_t)n * hd);
  const int half = hd / 2, quad = hd / 4;
  for (int i = 0; i < pH; ++i) {
    for (int j = 0; j < pW; ++j) {
      const int k = i * pW + j;
      _Float16* pp = pos.data() + (std::size_t)k * hidden;
      const _Float16* t0 = pos_tab + (std::size_t)j * hidden;
      const _Float16* t1 = pos_tab + axis_stride + (std::size_t)i * hidden;
      for (int d = 0; d < hidden; ++d) {
        pp[d] = (_Float16)((float)t0[d] + (float)t1[d]);
      }
      _Float16* cp = cosb.data() + (std::size_t)k * hd;
      _Float16* sp = sinb.data() + (std::size_t)k * hd;
      for (int d = 0; d < hd; ++d) {
        const int axis = d / half;            // 0 col, 1 row
        const int jf = (d % half) % quad;
        const float posv = (axis == 0) ? (float)j : (float)i;
        const float inv = std::pow(c.rope_theta, -(float)jf / (float)quad);
        const float ang = posv * inv;
        cp[d] = (_Float16)std::cos(ang);
        sp[d] = (_Float16)std::sin(ang);
      }
    }
  }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer featb = buf((std::size_t)n * feat_dim);
  SharedBuffer posb = buf((std::size_t)n * hidden);
  SharedBuffer cosbuf = buf((std::size_t)n * hd);
  SharedBuffer sinbuf = buf((std::size_t)n * hd);
  std::memcpy(featb.contents(), feat.data(), feat.size() * 2);
  std::memcpy(posb.contents(), pos.data(), pos.size() * 2);
  std::memcpy(cosbuf.contents(), cosb.data(), cosb.size() * 2);
  std::memcpy(sinbuf.contents(), sinb.data(), sinb.size() * 2);

  SharedBuffer x = buf((std::size_t)n * hidden);
  SharedBuffer n1 = buf((std::size_t)n * hidden);
  SharedBuffer q = buf((std::size_t)n * hidden), kk = buf((std::size_t)n * hidden),
               v = buf((std::size_t)n * hidden);
  SharedBuffer qr = buf((std::size_t)n * hidden), kr = buf((std::size_t)n * hidden);
  SharedBuffer qt = buf((std::size_t)n * hidden), kt = buf((std::size_t)n * hidden),
               vt = buf((std::size_t)n * hidden);
  SharedBuffer atb = buf((std::size_t)n * hidden), att = buf((std::size_t)n * hidden);
  SharedBuffer proj = buf((std::size_t)n * hidden);
  SharedBuffer yn = buf((std::size_t)n * hidden);
  SharedBuffer gbuf = buf((std::size_t)n * inter), ubuf = buf((std::size_t)n * inter);
  SharedBuffer ff = buf((std::size_t)n * hidden);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& y, int M, int N, int Kk) {
      if (_use_mma2) {
        // M5 matrix-core matmul2d GEMM y = x @ w^T. The Gemma vision linears
        // are bias-free, so (unlike the Qwen tower) there is no bias-add fold.
        // The matmul2d tensor extents clamp the M/N tails, so M=n and N need
        // not be tile multiples. 128x256 tile for deep K (K>=6144); e4b's
        // widest K is 3072 (down-proj) so it always takes the n128 tile.
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
      } else {
        enc.set_function(_fn_gemm_t);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, _zero_bias);
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);   // no bias (vision linears are bias-free)
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      }
    };
    auto rms = [&](const SharedBuffer& xin, const SharedBuffer& w,
                   const SharedBuffer& y, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y);
      enc.set_constant(3, Hd);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto grope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_grope);
      enc.set_buffer(0, in);
      enc.set_buffer(1, cosbuf);
      enc.set_buffer(2, sinbuf);
      enc.set_buffer(3, out);
      enc.set_constant(4, heads);
      enc.set_constant(5, hd);
      enc.dispatch({(unsigned)(n * heads * hd), 1, 1}, {256, 1, 1});
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
    auto geglu = [&](const SharedBuffer& g, const SharedBuffer& u,
                     const SharedBuffer& out, int nn) {
      enc.set_function(_fn_geglu);
      enc.set_buffer(0, g);
      enc.set_buffer(1, u);
      enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };

    // Patch embed (bias-free) + learned position embed.
    gemm(featb, _patch_w, x, n, hidden, feat_dim);
    residual(x, posb, x, n * hidden);

    // Steel flash-attention (MMA) setup -- shared across all blocks (same
    // n/heads/hd). Non-causal bidirectional ViT attention, scale 1.0 (the
    // q/k norms absorb 1/sqrt(d)). Vendored MLX kernel; ~10x the scalar
    // full SDPA. Prefer MLX's matrix-core (NAX) steel attention on M5
    // (attn_steel_nax, bq=64/bk=32, register-resident softmax) over the ALU
    // steel (attn_steel, bq=32/bk=16). Same AttnParams + func-const contract;
    // only the tiles and the kernel differ. Falls back to the local MMA /
    // scalar when neither steel kernel is available.
    const bool attn_nax = _use_attn_nax && _lib_attn_nax.valid();
    const int a_bq = attn_nax ? 64 : 32;
    const int a_bk = attn_nax ? 32 : 16;
    metal_compute::ComputeFunction fn_steel;
    if ((attn_nax || _lib_attn.valid()) && hd == 64 && !_attn_params.empty()) {
      auto* p = static_cast<SteelAttnParams*>(_attn_params.contents());
      p->B = 1; p->H = heads; p->D = hd; p->qL = n; p->kL = n;
      p->gqa_factor = 1; p->scale = 1.0f;
      p->NQ = (n + a_bq - 1) / a_bq; p->NK = (n + a_bk - 1) / a_bk;
      p->NQ_aligned = n / a_bq; p->NK_aligned = n / a_bk;
      p->qL_rem = n - p->NQ_aligned * a_bq;
      p->kL_rem = n - p->NK_aligned * a_bk;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)heads * n * hd;
      p->Q_strides[1] = (std::int64_t)n * hd; p->Q_strides[2] = hd;
      p->K_strides[0] = p->Q_strides[0];
      p->K_strides[1] = p->Q_strides[1]; p->K_strides[2] = hd;
      p->V_strides[0] = p->Q_strides[0];
      p->V_strides[1] = p->Q_strides[1]; p->V_strides[2] = hd;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = hd;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (n % a_bq) == 0).set_bool(201, (n % a_bk) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      fn_steel = attn_nax
          ? _lib_attn_nax.function("attn_steel_nax_h_bd64", fc)
          : _lib_attn.function("attn_steel_h_bd64", fc);
    }
    const bool use_steel = fn_steel.valid();
    const unsigned nqb = (unsigned)((n + a_bq - 1) / a_bq);

    for (int b = 0; b < c.depth; ++b) {
      Block& blk = _blocks[b];
      rms(x, blk.in_ln, n1, n, hidden);
      gemm(n1, blk.qw, q, n, hidden, hidden);
      gemm(n1, blk.kw, kk, n, hidden, hidden);
      gemm(n1, blk.vw, v, n, hidden, hidden);
      rms(q, blk.qn, q, n * heads, hd);          // per-head q_norm
      rms(kk, blk.kn, kk, n * heads, hd);
      rms(v, _ones_hd, v, n * heads, hd);        // weightless v_norm
      grope(q, qr);
      grope(kk, kr);
      transpose(qr, qt, n, heads);               // [n,H,hd]->[H,n,hd]
      transpose(kr, kt, n, heads);
      transpose(v, vt, n, heads);
      // Bidirectional SDPA, scale 1.0 (q/k norm absorb 1/sqrt(d)). Steel
      // MMA flash (~10x, MLX's kernel) when available; else the local MMA
      // flash (~3x, head_dim % 64 == 0); else the scalar fallback.
      if (!skip_attn) {
      if (use_steel) {
        // Q/K/V/O all [heads, n, hd]; params/constants set above the loop.
        enc.set_function(fn_steel);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kt);
        enc.set_buffer(2, vt);
        enc.set_buffer(3, atb);
        enc.set_buffer(4, _attn_params);
        enc.dispatch({32 * nqb, 4 * (unsigned)heads, 1}, {32, 4, 1});
      } else {
      const bool use_mma =
          _fn_sdpa_full_mma.valid() && (hd % 64 == 0) && (hd <= 512);
      enc.set_function(use_mma ? _fn_sdpa_full_mma : _fn_sdpa_full);
      enc.set_buffer(0, qt);
      enc.set_buffer(1, kt);
      enc.set_buffer(2, vt);
      enc.set_buffer(3, atb);
      enc.set_constant(4, 1.0f);
      enc.set_constant(5, n);            // T_kv
      enc.set_constant(6, hd);
      enc.set_constant(7, heads);
      enc.set_constant(8, heads);
      enc.set_constant(9, n);            // n_q
      enc.set_constant(10, n);           // kv_stride
      if (use_mma) {
        const unsigned nthreads = 4u * (unsigned)(hd / 64) * 32u;  // WM*WD*32
        const unsigned tiles = (unsigned)((n + 31) / 32);          // MMAF_BQ
        enc.dispatch({nthreads, (unsigned)heads, tiles},
                     {nthreads, 1, 1});
      } else {
        enc.dispatch({32, (unsigned)heads, (unsigned)n}, {32, 1, 1});
      }
      }
      }
      transpose(atb, att, heads, n);             // [H,n,hd]->[n,H,hd]
      gemm(att, blk.ow, proj, n, hidden, hidden);
      rms(proj, blk.post_attn_ln, proj, n, hidden);
      residual(x, proj, x, n * hidden);

      rms(x, blk.pre_ffn_ln, yn, n, hidden);
      gemm(yn, blk.gw, gbuf, n, inter, hidden);
      gemm(yn, blk.uw, ubuf, n, inter, hidden);
      geglu(gbuf, ubuf, gbuf, n * inter);
      gemm(gbuf, blk.dw, ff, n, hidden, inter);
      rms(ff, blk.post_ffn_ln, ff, n, hidden);
      residual(x, ff, x, n * hidden);
    }
  }
  const auto t_host = Clock::now();
  stream.commit().wait();

  // 3x3 position avg-pool (host) -> [n_im, hidden] x sqrt(hidden). Block
  // b = (row/K)*mw + (col/K); patch k = row*pW + col.
  const auto* xc = static_cast<const _Float16*>(x.contents());
  SharedBuffer pooledb = buf((std::size_t)n_im * hidden);
  auto* pooled = static_cast<_Float16*>(pooledb.contents());
  std::vector<float> acc((std::size_t)n_im * hidden, 0.0f);
  const float invk = 1.0f / (float)(K * K);
  const float root = std::sqrt((float)hidden);
  for (int i = 0; i < pH; ++i) {
    for (int j = 0; j < pW; ++j) {
      const int blk = (i / K) * mw + (j / K);
      const _Float16* xr = xc + (std::size_t)(i * pW + j) * hidden;
      float* ar = acc.data() + (std::size_t)blk * hidden;
      for (int d = 0; d < hidden; ++d) { ar[d] += (float)xr[d]; }
    }
  }
  for (int t = 0; t < n_im; ++t) {
    for (int d = 0; d < hidden; ++d) {
      pooled[(std::size_t)t * hidden + d] =
          (_Float16)(acc[(std::size_t)t * hidden + d] * invk * root);
    }
  }

  SharedBuffer normed = buf((std::size_t)n_im * hidden);
  SharedBuffer emb = buf((std::size_t)n_im * outh);
  metal_compute::CommandStream s2 = _mc->make_command_stream();
  {
    ComputeEncoder enc = s2.begin_compute();
    // embed_vision: RMSNormNoScale (ones weight) + 4-bit 768->out_hidden.
    enc.set_function(_fn_rms);
    enc.set_buffer(0, pooledb);
    enc.set_buffer(1, _ones_hid);
    enc.set_buffer(2, normed);
    enc.set_constant(3, hidden);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)n_im, 1}, {256, 1, 1});

    enc.set_function(_fn_qmm);
    enc.set_buffer(0, _ev_w);
    enc.set_buffer(1, _ev_s);
    enc.set_buffer(2, _ev_b);
    enc.set_buffer(3, normed);
    enc.set_buffer(4, emb);
    enc.set_constant(5, hidden);     // K
    enc.set_constant(6, outh);       // N
    enc.set_constant(7, n_im);       // M
    enc.dispatch({(unsigned)(((outh + 31) / 32) * 32),
                  (unsigned)(((n_im + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  s2.commit().wait();
  const auto t_gpu = Clock::now();
  if (profile) {
    auto ms = [](Clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    std::printf("MetalGemma4VisionEncoder: %dx%d -> %d tok (grid %dx%d) | "
                "n=%d heads=%d hd=%d depth=%d%s | host %.2f | gpu %.2f | "
                "total %.2f ms\n",
                H, W, res.n_tokens, mh, mw, n, heads, hd, c.depth,
                skip_attn ? " [SKIP-ATTN]" : "", ms(t_host - t_enter),
                ms(t_gpu - t_host), ms(t_gpu - t_enter));
  }

  res.embeddings = std::move(emb);
  return res;
}

}  // namespace vpipe::genai
