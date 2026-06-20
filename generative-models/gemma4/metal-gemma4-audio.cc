#include "generative-models/gemma4/metal-gemma4-audio.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// Coarse profiling (VPIPE_AUDIO_PROFILE): cumulative gemm_ wall (GPU dispatch
// + host f16<->f32 glue) within one encode(), vs the host attention/conv.
double g_audio_gemm_ms = 0.0;

std::size_t numel_(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}
inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Host elementwise helpers (f32).
inline float softplus_(float x)
{
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

}  // namespace

std::unique_ptr<MetalGemma4AudioEncoder>
MetalGemma4AudioEncoder::load(const std::string& model_dir,
                              metal_compute::MetalCompute* mc,
                              const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) { return nullptr; }

  auto m = std::unique_ptr<MetalGemma4AudioEncoder>(
      new MetalGemma4AudioEncoder());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_qmm = mc->load_library("affine_qmm_steel");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_rms = mc->load_library("rms_norm");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_lib_audio = mc->load_library("audio_encoder");
  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_qmm = m->_lib_qmm.function("affine_qmm_steel_w4g64");
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_clamp = m->_lib_elt.function("clamp_f16");
  m->_fn_silu = m->_lib_elt.function("mul_sigmoid_f16");   // silu via a=g
  m->_fn_glu = m->_lib_elt.function("glu_split_f16");
  m->_fn_dwconv = m->_lib_elt.function("depthwise_conv1d_causal_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_scale = m->_lib_elt.function("scale_inplace_f16");
  m->_fn_local_attn = m->_lib_sdpa.function("gemma_audio_local_attn_f16");
  m->_fn_conv2d = m->_lib_audio.function("audio_conv2d_3x3_s2p1_f16");
  m->_fn_ln_relu = m->_lib_rms.function("layer_norm_relu_f16");
  if (!m->_fn_gemm_t.valid() || !m->_fn_qmm.valid() || !m->_fn_rms.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_silu.valid() || !m->_fn_glu.valid() ||
      !m->_fn_dwconv.valid() || !m->_fn_residual.valid() ||
      !m->_fn_scale.valid() || !m->_fn_local_attn.valid() ||
      !m->_fn_conv2d.valid() || !m->_fn_ln_relu.valid()) {
    return nullptr;
  }
  // M5-only matrix-core dense GEMM (matmul2d/NAX) -- the same NAX path the LM
  // prefill and the Gemma ViT use. Gated on supports_matrix_cores()
  // (GPUFamilyApple10+) so M4 / older GPUs never load it and keep the steel
  // dense_gemm_t (byte-identical, performance unchanged). The conformer
  // linears dominate the encode (macaron FFN D<->4096 x2, q/k/v/post, lconv);
  // matmul2d reads each weight once across the T rows. The attention is a
  // cheap custom chunked-local kernel (hd=128), so there is no NAX-attn lever
  // here -- the GEMM is the whole win. VPIPE_GEMMA_AUDIO_NO_MMA2=1 forces the
  // steel path even on M5 (A/B + safety).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_GEMMA_AUDIO_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() &&
                   m->_fn_dense_mma_deep.valid() && m->_fn_bias_add.valid();
  }
  m->_zero_bias = mc->make_shared_buffer((std::size_t)cfg.ffn * 2);
  std::memset(m->_zero_bias.contents(), 0, (std::size_t)cfg.ffn * 2);

  bool ok = true;
  auto host_f32 = [&](const std::string& name) -> std::vector<float> {
    const auto* info = wts->info(name);
    if (!info) {
      ok = false;
      if (std::getenv("VPIPE_GEMMA_AUDIO_DBG")) {
        std::fprintf(stderr, "[metal-audio] missing host %s\n", name.c_str());
      }
      return {};
    }
    SharedBuffer raw = wts->load(name, mc);
    const std::size_t n = numel_(info->shape);
    std::vector<float> out(n);
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(s[i]); }
    } else if (info->dtype == "F32") {
      std::memcpy(out.data(), raw.contents(), n * sizeof(float));
    } else if (info->dtype == "F16") {
      const auto* s = static_cast<const _Float16*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { out[i] = (float)s[i]; }
    } else { ok = false; }
    return out;
  };
  auto f16_gpu = [&](const std::string& name, int* n_out,
                     int* k_out) -> SharedBuffer {
    const auto* info = wts->info(name);
    if (!info || info->shape.size() != 2) {
      ok = false;
      if (std::getenv("VPIPE_GEMMA_AUDIO_DBG")) {
        std::fprintf(stderr, "[metal-audio] missing/non2d gpu %s (ndim=%d)\n",
                     name.c_str(), info ? (int)info->shape.size() : -1);
      }
      return {};
    }
    if (n_out) { *n_out = (int)info->shape[0]; }
    if (k_out) { *k_out = (int)info->shape[1]; }
    SharedBuffer raw = wts->load(name, mc);
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<_Float16*>(out.contents());
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)bf16_to_f32_(s[i]); }
    } else if (info->dtype == "F16") {
      std::memcpy(o, raw.contents(), n * 2);
    } else { ok = false; }
    return out;
  };
  auto clip_of = [&](const std::string& base) -> Clip {
    Clip c;
    auto rd = [&](const char* f, float* v) -> bool {
      const auto* info = wts->info(base + f);
      if (!info) { return false; }
      SharedBuffer raw = wts->load(base + f, mc);
      if (info->dtype == "BF16") {
        *v = bf16_to_f32_(*static_cast<const std::uint16_t*>(raw.contents()));
      } else if (info->dtype == "F32") {
        *v = *static_cast<const float*>(raw.contents());
      } else { return false; }
      return true;
    };
    c.on = rd(".input_min", &c.imin) && rd(".input_max", &c.imax)
        && rd(".output_min", &c.omin) && rd(".output_max", &c.omax);
    return c;
  };
  auto load_lin = [&](const std::string& base, bool clippable,
                      bool nested) -> Lin {
    Lin l;
    const std::string wk = base + (nested ? ".linear.weight" : ".weight");
    l.w = f16_gpu(wk, &l.n, &l.k);
    if (clippable) { l.clip = clip_of(base); }
    return l;
  };

  const std::string r = "audio_tower.";
  const std::string s = r + "subsample_conv_projection.";
  m->_conv0_w = host_f32(s + "layer0.conv.weight");      // [128,3,3,1]
  m->_conv0_norm = host_f32(s + "layer0.norm.weight");   // [128]
  m->_conv1_w = host_f32(s + "layer1.conv.weight");      // [32,3,3,128]
  m->_conv1_norm = host_f32(s + "layer1.norm.weight");   // [32]
  m->_in_proj = load_lin(s + "input_proj_linear", false, false);
  m->_out_proj = load_lin(r + "output_proj", false, false);
  m->_out_proj_bias = host_f32(r + "output_proj.bias");

  m->_ev_w = wts->load("embed_audio.embedding_projection.weight", mc);
  { const auto* info = wts->info("embed_audio.embedding_projection.scales");
    SharedBuffer s2 = wts->load("embed_audio.embedding_projection.scales", mc);
    SharedBuffer b2 = wts->load("embed_audio.embedding_projection.biases", mc);
    auto cast = [&](SharedBuffer& raw, const char* nm) {
      const auto* in2 = wts->info(nm);
      const std::size_t n = in2 ? numel_(in2->shape) : 0;
      SharedBuffer o = mc->make_shared_buffer(n * 2);
      auto* op = static_cast<_Float16*>(o.contents());
      const auto* sp = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { op[i] = (_Float16)bf16_to_f32_(sp[i]); }
      return o;
    };
    (void)info;
    m->_ev_s = cast(s2, "embed_audio.embedding_projection.scales");
    m->_ev_b = cast(b2, "embed_audio.embedding_projection.biases");
  }

  m->_layers.resize((std::size_t)cfg.n_layers);
  for (int i = 0; i < cfg.n_layers; ++i) {
    auto& L = m->_layers[(std::size_t)i];
    const std::string b = r + "layers." + std::to_string(i) + ".";
    L.ff1_w1 = load_lin(b + "feed_forward1.ffw_layer_1", true, true);
    L.ff1_w2 = load_lin(b + "feed_forward1.ffw_layer_2", true, true);
    L.ff2_w1 = load_lin(b + "feed_forward2.ffw_layer_1", true, true);
    L.ff2_w2 = load_lin(b + "feed_forward2.ffw_layer_2", true, true);
    L.ff1_pre  = host_f32(b + "feed_forward1.pre_layer_norm.weight");
    L.ff1_post = host_f32(b + "feed_forward1.post_layer_norm.weight");
    L.ff2_pre  = host_f32(b + "feed_forward2.pre_layer_norm.weight");
    L.ff2_post = host_f32(b + "feed_forward2.post_layer_norm.weight");
    L.q = load_lin(b + "self_attn.q_proj", true, true);
    L.k = load_lin(b + "self_attn.k_proj", true, true);
    L.v = load_lin(b + "self_attn.v_proj", true, true);
    L.post = load_lin(b + "self_attn.post", true, true);
    L.rel_k = load_lin(b + "self_attn.relative_k_proj", false, false);
    L.per_dim_scale = host_f32(b + "self_attn.per_dim_scale");
    L.norm_pre_attn  = host_f32(b + "norm_pre_attn.weight");
    L.norm_post_attn = host_f32(b + "norm_post_attn.weight");
    L.norm_out       = host_f32(b + "norm_out.weight");
    L.lc_start = load_lin(b + "lconv1d.linear_start", true, true);
    L.lc_end   = load_lin(b + "lconv1d.linear_end", true, true);
    L.lc_pre  = host_f32(b + "lconv1d.pre_layer_norm.weight");
    L.lc_norm = host_f32(b + "lconv1d.conv_norm.weight");
    L.lc_dw   = host_f32(b + "lconv1d.depthwise_conv1d.weight");  // [D,k,1]
  }
  if (!ok) { return nullptr; }

  // Sinusoidal rel-pos timing table [span, d_model].
  const int max_back = std::max(0, cfg.ctx_left - 1);
  const int span = max_back + cfg.ctx_right + 1;
  const int D = cfg.d_model, nts = D / 2;
  const float log_inc = std::log(10000.0f) / (float)std::max(nts - 1, 1);
  m->_timing.assign((std::size_t)span * D, 0.0f);
  for (int sp = 0; sp < span; ++sp) {
    const float pos = (float)(max_back - sp);
    for (int z = 0; z < nts; ++z) {
      const float sc = pos * std::exp(-(float)z * log_inc);
      m->_timing[(std::size_t)sp * D + z] = std::sin(sc);
      m->_timing[(std::size_t)sp * D + nts + z] = std::cos(sc);
    }
  }
  return m;
}

MetalGemma4AudioEncoder::Config
MetalGemma4AudioEncoder::config_from(const ModelConfig& c)
{
  Config m;
  const auto& a = c.audio;
  m.d_model     = a.d_model;
  m.n_heads     = a.encoder_attention_heads;
  m.n_layers    = a.encoder_layers;
  m.ffn         = a.encoder_ffn_dim;
  m.output_dim  = a.output_dim;
  m.out_hidden  = c.hidden;
  m.n_mel       = a.num_mel_bins;
  m.conv_kernel = a.conv_kernel_size;
  m.chunk       = a.attention_chunk_size;
  m.ctx_left    = a.attention_context_left;
  m.ctx_right   = a.attention_context_right;
  m.softcap     = a.attention_logit_cap;
  m.grad_clip   = a.gradient_clipping;
  m.residual_w  = a.residual_weight;
  m.rms_eps     = a.audio_rms_eps;
  m.group_size  = c.quantization.group_size;
  m.bits        = c.quantization.bits;
  m.sscp_channels = a.subsampling_conv_channels;
  return m;
}

std::vector<float>
MetalGemma4AudioEncoder::gemm_(const std::vector<float>& x, int M, int K,
                               const Lin& lin) const
{
  const auto _t0 = std::chrono::steady_clock::now();
  const int N = lin.n;
  SharedBuffer xb = _mc->make_shared_buffer((std::size_t)M * K * 2);
  auto* xp = static_cast<_Float16*>(xb.contents());
  if (lin.clip.on) {
    for (std::size_t i = 0; i < (std::size_t)M * K; ++i) {
      xp[i] = (_Float16)std::min(std::max(x[i], lin.clip.imin), lin.clip.imax);
    }
  } else {
    for (std::size_t i = 0; i < (std::size_t)M * K; ++i) { xp[i] = (_Float16)x[i]; }
  }
  SharedBuffer yb = _mc->make_shared_buffer((std::size_t)M * N * 2);
  metal_compute::CommandStream st = _mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    enc.set_function(_fn_gemm_t);
    enc.set_buffer(0, xb);
    enc.set_buffer(1, lin.w);
    enc.set_buffer(2, _zero_bias);
    enc.set_buffer(3, yb);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  st.commit().wait();
  std::vector<float> y((std::size_t)M * N);
  const auto* yp = static_cast<const _Float16*>(yb.contents());
  if (lin.clip.on) {
    for (std::size_t i = 0; i < y.size(); ++i) {
      y[i] = std::min(std::max((float)yp[i], lin.clip.omin), lin.clip.omax);
    }
  } else {
    for (std::size_t i = 0; i < y.size(); ++i) { y[i] = (float)yp[i]; }
  }
  g_audio_gemm_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - _t0).count();
  return y;
}

std::vector<float>
MetalGemma4AudioEncoder::qmm_(const std::vector<float>& x, int M) const
{
  const int K = _cfg.output_dim, N = _cfg.out_hidden;
  SharedBuffer xb = _mc->make_shared_buffer((std::size_t)M * K * 2);
  auto* xp = static_cast<_Float16*>(xb.contents());
  for (std::size_t i = 0; i < (std::size_t)M * K; ++i) { xp[i] = (_Float16)x[i]; }
  SharedBuffer yb = _mc->make_shared_buffer((std::size_t)M * N * 2);
  metal_compute::CommandStream st = _mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    enc.set_function(_fn_qmm);
    enc.set_buffer(0, _ev_w);
    enc.set_buffer(1, _ev_s);
    enc.set_buffer(2, _ev_b);
    enc.set_buffer(3, xb);
    enc.set_buffer(4, yb);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  st.commit().wait();
  std::vector<float> y((std::size_t)M * N);
  const auto* yp = static_cast<const _Float16*>(yb.contents());
  for (std::size_t i = 0; i < y.size(); ++i) { y[i] = (float)yp[i]; }
  return y;
}

void
MetalGemma4AudioEncoder::prepare_gpu_()
{
  if (_gpu_prepared) { return; }
  const Config& c = _cfg;
  const int D = c.d_model, Hd = c.head_dim(), K = c.conv_kernel;
  auto vec_f16 = [&](const std::vector<float>& v) -> SharedBuffer {
    SharedBuffer b = _mc->make_shared_buffer(v.size() * 2);
    auto* p = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { p[i] = (_Float16)v[i]; }
    return b;
  };
  for (auto& L : _layers) {
    L.ff1_pre_b = vec_f16(L.ff1_pre);
    L.ff1_post_b = vec_f16(L.ff1_post);
    L.ff2_pre_b = vec_f16(L.ff2_pre);
    L.ff2_post_b = vec_f16(L.ff2_post);
    L.npa_b = vec_f16(L.norm_pre_attn);
    L.npost_b = vec_f16(L.norm_post_attn);
    L.nout_b = vec_f16(L.norm_out);
    L.lc_pre_b = vec_f16(L.lc_pre);
    L.lc_norm_b = vec_f16(L.lc_norm);
    std::vector<float> pds((std::size_t)Hd);
    for (int d = 0; d < Hd; ++d) { pds[d] = softplus_(L.per_dim_scale[d]); }
    L.pds_b = vec_f16(pds);
    L.lc_dw_b = vec_f16(L.lc_dw);   // [D, K]
    (void)K;
  }
  _out_proj_bias_b = vec_f16(_out_proj_bias);
  _ones_outdim = vec_f16(std::vector<float>((std::size_t)c.output_dim, 1.0f));
  _timing_b = vec_f16(_timing);
  // SSCP conv stem f16 weights + a zero conv bias (Gemma convs are
  // bias-free; the shared conv2d kernel always adds bias[cout]).
  _conv0_w_b = vec_f16(_conv0_w);
  _conv1_w_b = vec_f16(_conv1_w);
  _conv0_norm_b = vec_f16(_conv0_norm);
  _conv1_norm_b = vec_f16(_conv1_norm);
  const int cmax = std::max(c.sscp_channels[0], c.sscp_channels[1]);
  _conv_zero_bias = vec_f16(std::vector<float>((std::size_t)cmax, 0.0f));
  (void)D;
  _gpu_prepared = true;
}

MetalGemma4AudioEncoder::Result
MetalGemma4AudioEncoder::encode(const float* pcm, std::size_t n_samples,
                                int sample_rate)
{
  Result res;
  if (pcm == nullptr || n_samples == 0 || sample_rate != 16000) { return res; }
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudio,
                     kPerfLlmAudioBegin,
                     static_cast<std::uint64_t>(n_samples));
  const bool profile = std::getenv("VPIPE_AUDIO_PROFILE") != nullptr;
  const auto t_enter = std::chrono::steady_clock::now();
  g_audio_gemm_ms = 0.0;
  std::vector<float> mel;
  const int T0 = (int)_fx.extract(pcm, n_samples, &mel);   // [T0, n_mel]
  if (T0 == 0) { return res; }

  const Config& c = _cfg;
  const int D = c.d_model, N = c.n_heads, Hd = c.head_dim();
  const int Fm = c.n_mel;
  const float eps = c.rms_eps;

  // ---- SSCP conv stem dims. conv = 3x3 stride 2 pad 1 -> out =
  //      (in + 2 - 3) / 2 + 1. mel [T0,Fm] -> conv0 [T1,F1,c0] -> conv1
  //      [T2,F2,c1] = flatten [T, FC] -> input_proj [T, D]. The whole
  //      stem runs on the GPU in the stream below (host only does the mel
  //      front-end).
  const int c0 = c.sscp_channels[0], c1 = c.sscp_channels[1];
  const int T1 = (T0 + 2 - 3) / 2 + 1, F1 = (Fm + 2 - 3) / 2 + 1;
  const int T2 = (T1 + 2 - 3) / 2 + 1, F2 = (F1 + 2 - 3) / 2 + 1;
  const int T = T2;
  const int FC = F2 * c1;

  // ---- GPU forward (conv stem + 12 Conformer blocks + output stage) ----
  // Everything after the host mel front-end runs as ONE command stream:
  // the SSCP conv2d stem (reusing the Qwen3-ASR audio_conv2d kernel) +
  // input_proj, then the 12 blocks (steel GEMM linears, the chunked-local
  // rel-pos attention gemma_audio_local_attn_f16, and the glue -- RMSNorm,
  // LayerNorm+ReLU, SiLU, GLU, depthwise Conv1d, clamps, residuals).
  // Default-serial ComputeEncoder hazard-tracks the in-order dispatches,
  // so no explicit barriers are needed.
  prepare_gpu_();
  const int W = c.chunk, mb = std::max(0, c.ctx_left - 1), mf = c.ctx_right;
  const int span = mb + mf + 1;
  const float q_scale = std::pow((float)Hd, -0.5f) / std::log(2.0f);
  const float k_scale = std::log(1.0f + (float)M_E) / std::log(2.0f);

  auto mkb = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer melb = mkb((std::size_t)T0 * Fm);     // [T0, Fm, Cin=1] f16
  {
    auto* mp = static_cast<_Float16*>(melb.contents());
    for (std::size_t i = 0; i < (std::size_t)T0 * Fm; ++i) {
      mp[i] = (_Float16)mel[i];
    }
  }
  SharedBuffer c0b = mkb((std::size_t)T1 * F1 * c0);
  SharedBuffer c1b = mkb((std::size_t)T2 * F2 * c1);   // == [T, FC]
  SharedBuffer xb = mkb((std::size_t)T * D);
  SharedBuffer nrm = mkb((std::size_t)T * D);
  SharedBuffer cbuf = mkb((std::size_t)T * c.ffn);   // clamp scratch
  SharedBuffer t1 = mkb((std::size_t)T * c.ffn);     // ff hidden / [T,2D]
  SharedBuffer t2 = mkb((std::size_t)T * D);
  SharedBuffer qb = mkb((std::size_t)T * D), kb = mkb((std::size_t)T * D),
               vb = mkb((std::size_t)T * D), ctxb = mkb((std::size_t)T * D);
  SharedBuffer gb = mkb((std::size_t)T * D), cvb = mkb((std::size_t)T * D);
  SharedBuffer sinb = mkb((std::size_t)span * D);
  SharedBuffer opb = mkb((std::size_t)T * c.output_dim);
  SharedBuffer embb = mkb((std::size_t)T * c.out_hidden);

  const auto t_host = std::chrono::steady_clock::now();
  metal_compute::CommandStream st = _mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& bias, int has_bias,
                    const SharedBuffer& y, int M, int Kk, int Nn) {
      if (_use_mma2) {
        // M5 matrix-core matmul2d GEMM y = x @ w^T; the matmul2d tensor extents
        // clamp the M/N tails so M=T and N need not be tile multiples. 128x256
        // tile for deep K (K>=6144); the conformer's widest K is 4096 (ffn) so
        // it always takes n128. Bias (output_proj only) is folded over the rows.
        const bool deep = (Kk >= 6144);
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);     // bias slot unused (has_bias=0)
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, Nn);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((Nn + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        if (has_bias) {
          enc.set_function(_fn_bias_add);
          enc.set_buffer(0, y);
          enc.set_buffer(1, bias);
          enc.set_constant(2, Nn);
          enc.set_constant(3, M * Nn);
          enc.dispatch({(unsigned)(M * Nn), 1, 1}, {256, 1, 1});
        }
      } else {
        enc.set_function(_fn_gemm_t);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, Nn);
        enc.set_constant(6, M);
        enc.set_constant(7, has_bias);
        enc.dispatch({(unsigned)(((Nn + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      }
    };
    auto clampb = [&](const SharedBuffer& in, const SharedBuffer& out, int nn,
                      float lo, float hi) {
      enc.set_function(_fn_clamp);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, nn);
      enc.set_constant(3, lo);
      enc.set_constant(4, hi);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    // GEMM with a ClippableLinear's input/output clamps (mirrors gemm_).
    auto gemm_lin = [&](const SharedBuffer& xin, const Lin& lin,
                        const SharedBuffer& y, int M, int Kk) {
      const SharedBuffer* src = &xin;
      if (lin.clip.on) {
        clampb(xin, cbuf, M * Kk, lin.clip.imin, lin.clip.imax);
        src = &cbuf;
      }
      gemm(*src, lin.w, _zero_bias, 0, y, M, Kk, lin.n);
      if (lin.clip.on) {
        clampb(y, y, M * lin.n, lin.clip.omin, lin.clip.omax);
      }
    };
    auto rms = [&](const SharedBuffer& xin, const SharedBuffer& w,
                   const SharedBuffer& y, int R, int Hdim) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y);
      enc.set_constant(3, Hdim);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto silu = [&](const SharedBuffer& xin, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_silu);     // mul_sigmoid(a=g=x) = x*sigmoid(x)
      enc.set_buffer(0, xin);
      enc.set_buffer(1, xin);
      enc.set_buffer(2, y);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& y, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, y);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto scale = [&](const SharedBuffer& xin, int nn, float s) {
      enc.set_function(_fn_scale);
      enc.set_buffer(0, xin);
      enc.set_constant(1, nn);
      enc.set_constant(2, s);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto glu = [&](const SharedBuffer& in, const SharedBuffer& y, int R,
                   int Dd) {
      enc.set_function(_fn_glu);
      enc.set_buffer(0, in);
      enc.set_buffer(1, y);
      enc.set_constant(2, R);
      enc.set_constant(3, Dd);
      enc.dispatch({(unsigned)(R * Dd), 1, 1}, {256, 1, 1});
    };
    auto dwconv = [&](const SharedBuffer& in, const SharedBuffer& w,
                      const SharedBuffer& y, int Tt, int Dd, int Kk) {
      enc.set_function(_fn_dwconv);
      enc.set_buffer(0, in);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y);
      enc.set_constant(3, Tt);
      enc.set_constant(4, Dd);
      enc.set_constant(5, Kk);
      enc.dispatch({(unsigned)(Tt * Dd), 1, 1}, {256, 1, 1});
    };
    auto local_attn = [&](const Layer& L) {
      enc.set_function(_fn_local_attn);
      enc.set_buffer(0, qb);
      enc.set_buffer(1, kb);
      enc.set_buffer(2, vb);
      enc.set_buffer(3, sinb);
      enc.set_buffer(4, L.pds_b);
      enc.set_buffer(5, ctxb);
      enc.set_constant(6, T);
      enc.set_constant(7, N);
      enc.set_constant(8, Hd);
      enc.set_constant(9, W);
      enc.set_constant(10, mb);
      enc.set_constant(11, mf);
      enc.set_constant(12, span);
      enc.set_constant(13, q_scale);
      enc.set_constant(14, k_scale);
      enc.set_constant(15, c.softcap);
      enc.dispatch({32, (unsigned)N, (unsigned)T}, {32, 1, 1});
    };
    // Macaron FFN: rms(pre) -> w1 -> silu -> w2 -> rms(post) -> x += w*ff.
    auto ffn = [&](const SharedBuffer& pre, const SharedBuffer& post,
                   const Lin& w1, const Lin& w2) {
      rms(xb, pre, nrm, T, D);
      gemm_lin(nrm, w1, t1, T, D);            // [T, ffn]
      silu(t1, t1, T * c.ffn);
      gemm_lin(t1, w2, t2, T, c.ffn);         // [T, D]
      rms(t2, post, t2, T, D);
      scale(t2, T * D, c.residual_w);
      residual(xb, t2, xb, T * D);
    };
    auto conv2d = [&](const SharedBuffer& in, const SharedBuffer& w,
                      const SharedBuffer& out, int H, int Wd, int Cin,
                      int Cout, int Hout, int Wout) {
      enc.set_function(_fn_conv2d);
      enc.set_buffer(0, in);
      enc.set_buffer(1, w);
      enc.set_buffer(2, _conv_zero_bias);
      enc.set_buffer(3, out);
      enc.set_constant(4, 1);          // N
      enc.set_constant(5, H);
      enc.set_constant(6, Wd);
      enc.set_constant(7, Cin);
      enc.set_constant(8, Cout);
      enc.set_constant(9, Hout);
      enc.set_constant(10, Wout);
      enc.set_constant(11, 0);         // transpose_out
      enc.dispatch({(unsigned)(((Cout + 31) / 32) * 32), (unsigned)Wout,
                    (unsigned)Hout}, {32, 1, 1});
    };
    auto ln_relu = [&](const SharedBuffer& xin, const SharedBuffer& w,
                       const SharedBuffer& out, int rows, int Cc) {
      enc.set_function(_fn_ln_relu);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, out);
      enc.set_constant(3, Cc);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    };

    // SSCP conv stem -> input_proj -> xb [T, D].
    conv2d(melb, _conv0_w_b, c0b, T0, Fm, 1, c0, T1, F1);
    ln_relu(c0b, _conv0_norm_b, c0b, T1 * F1, c0);
    conv2d(c0b, _conv1_w_b, c1b, T1, F1, c0, c1, T2, F2);
    ln_relu(c1b, _conv1_norm_b, c1b, T2 * F2, c1);
    gemm(c1b, _in_proj.w, _zero_bias, 0, xb, T, FC, D);

    for (const auto& L : _layers) {
      ffn(L.ff1_pre_b, L.ff1_post_b, L.ff1_w1, L.ff1_w2);

      // chunked-local rel-pos attention.
      rms(xb, L.npa_b, nrm, T, D);
      gemm_lin(nrm, L.q, qb, T, D);
      gemm_lin(nrm, L.k, kb, T, D);
      gemm_lin(nrm, L.v, vb, T, D);
      gemm(_timing_b, L.rel_k.w, _zero_bias, 0, sinb, span, D, L.rel_k.n);
      local_attn(L);
      gemm_lin(ctxb, L.post, t2, T, D);
      rms(t2, L.npost_b, t2, T, D);
      residual(xb, t2, xb, T * D);

      // light depthwise Conv1d + GLU.
      rms(xb, L.lc_pre_b, nrm, T, D);
      gemm_lin(nrm, L.lc_start, t1, T, D);    // [T, 2D]
      glu(t1, gb, T, D);
      dwconv(gb, L.lc_dw_b, cvb, T, D, c.conv_kernel);
      rms(cvb, L.lc_norm_b, cvb, T, D);
      silu(cvb, cvb, T * D);
      gemm_lin(cvb, L.lc_end, t2, T, D);
      residual(xb, t2, xb, T * D);

      ffn(L.ff2_pre_b, L.ff2_post_b, L.ff2_w1, L.ff2_w2);
      rms(xb, L.nout_b, xb, T, D);
    }

    // output_proj (+bias) -> embed_audio (RMSNormNoScale + 4-bit qmm).
    gemm(xb, _out_proj.w, _out_proj_bias_b, 1, opb, T, D, c.output_dim);
    rms(opb, _ones_outdim, opb, T, c.output_dim);
    enc.set_function(_fn_qmm);
    enc.set_buffer(0, _ev_w);
    enc.set_buffer(1, _ev_s);
    enc.set_buffer(2, _ev_b);
    enc.set_buffer(3, opb);
    enc.set_buffer(4, embb);
    enc.set_constant(5, c.output_dim);
    enc.set_constant(6, c.out_hidden);
    enc.set_constant(7, T);
    enc.dispatch({(unsigned)(((c.out_hidden + 31) / 32) * 32),
                  (unsigned)(((T + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  st.commit().wait();

  res.embeddings.resize((std::size_t)T * c.out_hidden);
  {
    const auto* ep = static_cast<const _Float16*>(embb.contents());
    for (std::size_t i = 0; i < res.embeddings.size(); ++i) {
      res.embeddings[i] = (float)ep[i];
    }
  }
  res.n_tokens = T;
  if (profile) {
    auto ms = [](std::chrono::steady_clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    const auto t_done = std::chrono::steady_clock::now();
    std::printf("MetalGemma4AudioEncoder: T=%d (mel %d) layers=%d | "
                "host %.2f | gpu %.2f | total %.2f ms\n",
                T, T0, c.n_layers, ms(t_host - t_enter), ms(t_done - t_host),
                ms(t_done - t_enter));
  }
  return res;
}

}  // namespace vpipe::genai
