#include "generative-models/moss/metal-moss-codec-v2.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// Convert a 2-D F32/F16/BF16 tensor to a fresh f16 [r,c] buffer.
SharedBuffer
to_f16_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)s[i]; }
  } else if (info->dtype == "F16") {
    std::memcpy(d, raw.contents(), n * 2);
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f; std::memcpy(&f, &u, 4); d[i] = (_Float16)f;
    }
  } else {
    return {};
  }
  return out;
}

// to_f16 with each output ROW scaled by scale[row] -- folds a per-channel
// LayerScale into a [out,in] projection weight. F32 source (codec checkpoint).
SharedBuffer
to_f16_rowscale_(const MetalLlamaWeights& wts, MetalCompute* mc,
                 const std::string& wname, const std::string& sname)
{
  const auto* wi = wts.info(wname);
  const auto* si = wts.info(sname);
  if (wi == nullptr || si == nullptr || wi->shape.size() != 2) { return {}; }
  const int out = (int)wi->shape[0], in = (int)wi->shape[1];
  SharedBuffer ws = wts.load(wname, mc), ss = wts.load(sname, mc);
  if (ws.empty() || ss.empty()) { return {}; }
  SharedBuffer o = mc->make_shared_buffer((std::size_t)out * in * 2);
  if (o.empty()) { return {}; }
  const auto* w = static_cast<const float*>(ws.contents());
  const auto* sc = static_cast<const float*>(ss.contents());
  auto* d = static_cast<_Float16*>(o.contents());
  for (int r = 0; r < out; ++r) {
    for (int c = 0; c < in; ++c) {
      d[(std::size_t)r * in + c] =
          (_Float16)(sc[r] * w[(std::size_t)r * in + c]);
    }
  }
  return o;
}

// Fold a weight-normalized 1x1 conv into a plain [out,in] f16 matrix:
// w_eff[o,i] = g[o] * v[o,i] / ||v[o,:]||_2 (original0=g [out,1,1],
// original1=v [out,in,1]).
SharedBuffer
fold_wnconv_(const MetalLlamaWeights& wts, MetalCompute* mc,
             const std::string& prefix)
{
  const std::string gn = prefix + ".parametrizations.weight.original0";
  const std::string vn = prefix + ".parametrizations.weight.original1";
  const auto* vi = wts.info(vn);
  const auto* gi = wts.info(gn);
  if (vi == nullptr || gi == nullptr || vi->shape.size() < 2) { return {}; }
  const int out = (int)vi->shape[0], in = (int)vi->shape[1];
  SharedBuffer vs = wts.load(vn, mc), gs = wts.load(gn, mc);
  if (vs.empty() || gs.empty()) { return {}; }
  const auto* v = static_cast<const float*>(vs.contents());
  const auto* g = static_cast<const float*>(gs.contents());
  SharedBuffer o = mc->make_shared_buffer((std::size_t)out * in * 2);
  if (o.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(o.contents());
  for (int r = 0; r < out; ++r) {
    double ss = 0.0;
    for (int c = 0; c < in; ++c) {
      const double x = v[(std::size_t)r * in + c];
      ss += x * x;
    }
    const float inv = (float)(1.0 / std::sqrt(ss));
    for (int c = 0; c < in; ++c) {
      d[(std::size_t)r * in + c] =
          (_Float16)(g[r] * v[(std::size_t)r * in + c] * inv);
    }
  }
  return o;
}

}  // namespace

std::unique_ptr<MetalMossCodecV2>
MetalMossCodecV2::load(const std::string& model_dir, MetalCompute* mc,
                       bool with_encoder)
{
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts.has_value()) { return nullptr; }
  auto self = std::unique_ptr<MetalMossCodecV2>(new MetalMossCodecV2());
  self->_mc = mc;
  self->_session = mc->session();
  self->_with_encoder = with_encoder;
  if (!self->init_(*wts, mc, with_encoder)) { return nullptr; }
  return self;
}

bool
MetalMossCodecV2::init_(const MetalLlamaWeights& wts, MetalCompute* mc,
                        bool with_encoder)
{
  // Decoder structure (config.json decoder_kwargs). 6 ProjectedTransformers,
  // patch upsamplers x2 x2 x2 x2 x2 x240. context = round(frame_rate *
  // context_duration); frame rate doubles per x2 patch: 12.5,25,50,100,200,400.
  static const StageCfg cfgs[6] = {
      {768, 1280, 32, 20, 5120, 1280,   2, 125},  // decoder.0  (12.5Hz, 10s)
      {640,  768, 12, 12, 3072,  768,   2, 250},  // decoder.2  (25Hz,   10s)
      {384,  768, 12, 12, 3072,  768,   2, 400},  // decoder.4  (50Hz,    8s)
      {384,  768, 12, 12, 3072,  768,   2, 400},  // decoder.6  (100Hz,   4s)
      {384,  768, 12, 12, 3072,  768,   2, 400},  // decoder.8  (200Hz,   2s)
      {384,  768, 12, 12, 3072,  240, 240, 400},  // decoder.10 (400Hz,   1s)
  };
  static const int module_idx[6] = {0, 2, 4, 6, 8, 10};

  // RLFQ (the LM emits 12 codes/frame; the codec defines 32 quantizers).
  _codebook.resize((std::size_t)_n_vq);
  _q_outw.resize((std::size_t)_n_vq);
  _q_outb.resize((std::size_t)_n_vq);
  bool ok = true;
  for (int i = 0; i < _n_vq; ++i) {
    const std::string q = "quantizer.quantizers." + std::to_string(i) + ".";
    _codebook[(std::size_t)i] = to_f16_(wts, mc, q + "codebook.weight");
    _q_outw[(std::size_t)i]   = fold_wnconv_(wts, mc, q + "out_proj");
    _q_outb[(std::size_t)i]   = to_f16_(wts, mc, q + "out_proj.bias");
    ok = ok && !_codebook[(std::size_t)i].empty() &&
         !_q_outw[(std::size_t)i].empty() && !_q_outb[(std::size_t)i].empty();
  }
  _rvq_outw = fold_wnconv_(wts, mc, "quantizer.output_proj");
  _rvq_outb = to_f16_(wts, mc, "quantizer.output_proj.bias");
  ok = ok && !_rvq_outw.empty() && !_rvq_outb.empty();
  if (!ok) { return false; }

  // Decoder transformer stages.
  _stages.resize(6);
  for (int s = 0; s < 6; ++s) {
    Stage& st = _stages[(std::size_t)s];
    st.cfg = cfgs[s];
    const StageCfg& c = st.cfg;
    const std::string base = "decoder." + std::to_string(module_idx[s]) + ".";
    // The v2 ProjectedTransformer ALWAYS has both input_proj and output_proj
    // (even when out_dim == d_model -- a real learned [d,d] Linear, not an
    // identity). (void)c so the unused-with-the-old-condition note is gone.
    (void)c;
    st.in_proj  = to_f16_(wts, mc, base + "input_proj.weight");
    st.out_proj = to_f16_(wts, mc, base + "output_proj.weight");
    ok = ok && !st.in_proj.empty() && !st.out_proj.empty();
    st.layers.resize((std::size_t)c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) {
      Layer& L = st.layers[(std::size_t)l];
      const std::string p =
          base + "transformer.layers." + std::to_string(l) + ".";
      L.n1w = to_f16_(wts, mc, p + "norm1.weight");
      L.n1b = to_f16_(wts, mc, p + "norm1.bias");
      L.n2w = to_f16_(wts, mc, p + "norm2.weight");
      L.n2b = to_f16_(wts, mc, p + "norm2.bias");
      L.qkvw = to_f16_(wts, mc, p + "self_attn.in_proj.weight");
      // Fold the per-channel LayerScales into out_proj / ffn.2 rows.
      L.ow = to_f16_rowscale_(wts, mc, p + "self_attn.out_proj.weight",
                              p + "layer_scale_1.scale");
      L.fc1 = to_f16_(wts, mc, p + "ffn.0.weight");
      L.fc2 = to_f16_rowscale_(wts, mc, p + "ffn.2.weight",
                               p + "layer_scale_2.scale");
      ok = ok && !L.n1w.empty() && !L.n1b.empty() && !L.n2w.empty() &&
           !L.n2b.empty() && !L.qkvw.empty() && !L.ow.empty() &&
           !L.fc1.empty() && !L.fc2.empty();
    }
  }
  if (!ok) { return false; }

  // ---- encode path (optional) --------------------------------------------
  // The 6 encoder transformer stages MIRROR the decoder but run in the DOWN
  // direction: each StageCfg.patch is the DOWN patch-reshape applied BEFORE the
  // stage. Contexts = round(frame_rate * context_duration), the reverse of the
  // decoder (frame rate halves per x2 patch on the way down: 400,400,400,400,
  // 250,125). The first stage's patch (x240) reshapes the channel-interleaved
  // waveform; the rest are x2.
  if (with_encoder) {
    static const StageCfg enc_cfgs[6] = {
        { 240,  768, 12, 12, 3072, 384, 240, 400},  // encoder.1  (400Hz)
        { 768,  768, 12, 12, 3072, 384,   2, 400},  // encoder.3  (200Hz)
        { 768,  768, 12, 12, 3072, 384,   2, 400},  // encoder.5  (100Hz)
        { 768,  768, 12, 12, 3072, 384,   2, 400},  // encoder.7  (50Hz)
        { 768,  768, 12, 12, 3072, 640,   2, 250},  // encoder.9  (25Hz)
        {1280, 1280, 32, 20, 5120, 768,   2, 125},  // encoder.11 (12.5Hz)
    };
    static const int enc_module_idx[6] = {1, 3, 5, 7, 9, 11};

    // RLFQ encode projections: global input_proj (768->512) folded wnconv +
    // bias, and each codebook's in_proj (512->8) folded wnconv + bias. The
    // out_proj / codebook / output_proj are already loaded (decode path).
    _q_inw.resize((std::size_t)_n_vq);
    _q_inb.resize((std::size_t)_n_vq);
    _rvq_inw = fold_wnconv_(wts, mc, "quantizer.input_proj");
    _rvq_inb = to_f16_(wts, mc, "quantizer.input_proj.bias");
    ok = ok && !_rvq_inw.empty() && !_rvq_inb.empty();
    for (int i = 0; i < _n_vq; ++i) {
      const std::string q = "quantizer.quantizers." + std::to_string(i);
      _q_inw[(std::size_t)i] = fold_wnconv_(wts, mc, q + ".in_proj");
      _q_inb[(std::size_t)i] = to_f16_(wts, mc, q + ".in_proj.bias");
      ok = ok && !_q_inw[(std::size_t)i].empty() &&
           !_q_inb[(std::size_t)i].empty();
    }

    // Encoder transformer stages (same load as the decoder stages).
    _enc_stages.resize(6);
    for (int s = 0; s < 6; ++s) {
      Stage& st = _enc_stages[(std::size_t)s];
      st.cfg = enc_cfgs[s];
      const StageCfg& c = st.cfg;
      const std::string base =
          "encoder." + std::to_string(enc_module_idx[s]) + ".";
      st.in_proj  = to_f16_(wts, mc, base + "input_proj.weight");
      st.out_proj = to_f16_(wts, mc, base + "output_proj.weight");
      ok = ok && !st.in_proj.empty() && !st.out_proj.empty();
      st.layers.resize((std::size_t)c.n_layers);
      for (int l = 0; l < c.n_layers; ++l) {
        Layer& L = st.layers[(std::size_t)l];
        const std::string p =
            base + "transformer.layers." + std::to_string(l) + ".";
        L.n1w = to_f16_(wts, mc, p + "norm1.weight");
        L.n1b = to_f16_(wts, mc, p + "norm1.bias");
        L.n2w = to_f16_(wts, mc, p + "norm2.weight");
        L.n2b = to_f16_(wts, mc, p + "norm2.bias");
        L.qkvw = to_f16_(wts, mc, p + "self_attn.in_proj.weight");
        L.ow = to_f16_rowscale_(wts, mc, p + "self_attn.out_proj.weight",
                                p + "layer_scale_1.scale");
        L.fc1 = to_f16_(wts, mc, p + "ffn.0.weight");
        L.fc2 = to_f16_rowscale_(wts, mc, p + "ffn.2.weight",
                                 p + "layer_scale_2.scale");
        ok = ok && !L.n1w.empty() && !L.n1b.empty() && !L.n2w.empty() &&
             !L.n2b.empty() && !L.qkvw.empty() && !L.ow.empty() &&
             !L.fc1.empty() && !L.fc2.empty();
      }
    }
    if (!ok) { return false; }

    // L2-normalized codebooks for the cosine-nearest search (F.normalize:
    // v / max(||v||_2, 1e-12)). The RAW codebook (already loaded) is used for
    // the residual subtraction, matching the LFQ decode_latents semantics.
    _codebook_norm.resize((std::size_t)_n_vq);
    const int CD = _codebook_dim, Csz = _codebook_size;
    for (int i = 0; i < _n_vq; ++i) {
      SharedBuffer nb = mc->make_shared_buffer((std::size_t)Csz * CD * 2);
      if (nb.empty()) { return false; }
      const auto* s =
          static_cast<const _Float16*>(_codebook[(std::size_t)i].contents());
      auto* d = static_cast<_Float16*>(nb.contents());
      for (int r = 0; r < Csz; ++r) {
        double sq = 0.0;
        for (int e = 0; e < CD; ++e) {
          const double x = (double)s[(std::size_t)r * CD + e];
          sq += x * x;
        }
        const double inv = 1.0 / std::max(std::sqrt(sq), 1e-12);
        for (int e = 0; e < CD; ++e) {
          d[(std::size_t)r * CD + e] =
              (_Float16)((double)s[(std::size_t)r * CD + e] * inv);
        }
      }
      _codebook_norm[(std::size_t)i] = std::move(nb);
    }
  }

  // inv_freq for the interleaved RoPE (head_dim 64, max_period 10000).
  const int hd = 64, half = hd / 2;
  _inv_freq = mc->make_shared_buffer((std::size_t)half * sizeof(float));
  if (_inv_freq.empty()) { return false; }
  auto* invf = static_cast<float*>(_inv_freq.contents());
  for (int i = 0; i < half; ++i) {
    invf[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)hd);
  }

  _lib_gemm = mc->load_library("dense_gemm");
  _lib_vis  = mc->load_library("qwen3_5_vision");
  _lib_elt  = mc->load_library("llm_elementwise");
  _lib_sdpa = mc->load_library("sdpa");
  _lib_rope = mc->load_library("rope");
  _fn_gemm      = _lib_gemm.function("dense_gemm_t_f16");
  _fn_gemm_bias = _lib_gemm.function("dense_gemm_bias_f16");
  _fn_ln        = _lib_vis.function("layer_norm_bias_f16");
  _fn_gelu      = _lib_vis.function("gelu_erf_f16");
  _fn_hslice    = _lib_elt.function("head_slice_f16");
  _fn_transpose = _lib_elt.function("transpose_abd_f16");
  _fn_residual  = _lib_elt.function("residual_add_f16");
  _fn_sdpa      = _lib_sdpa.function("sdpa_causal_window_f16");
  _fn_rope      = _lib_rope.function("rope_interleaved_f16");
  _fn_ring_append = _lib_elt.function("ring_append_f16");
  _ok = _fn_gemm.valid() && _fn_gemm_bias.valid() && _fn_ln.valid() &&
        _fn_gelu.valid() && _fn_hslice.valid() && _fn_transpose.valid() &&
        _fn_residual.valid() && _fn_sdpa.valid() && _fn_rope.valid();

  // M5 matrix-core (matmul2d/NAX) dense GEMM, gated on GPU support so M4 and
  // older keep the steel dense_gemm_t path byte-identical. VPIPE_MOSS_CODEC_
  // NO_MMA2=1 forces steel on M5 (A/B + safety). The n128 tile fits every
  // codec GEMM (max K = ff = 5120 < the 6144 deep threshold).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_MOSS_CODEC_NO_MMA2") == nullptr) {
    _lib_dense_mma = mc->load_library("dense_gemm_mma");
    _fn_gemm_mma = _lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_gemm_mma_deep =
        _lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    _mma_available = _fn_gemm_mma.valid() && _fn_gemm_mma_deep.valid();
    _use_mma2 = _mma_available;
  }

  // M5 matrix-core windowed-causal flash attention (head_dim 64). Replaces the
  // scalar sdpa_causal_window_f16 (the ~86% decode bottleneck at long seq).
  // Independent lever: VPIPE_MOSS_CODEC_NO_ATTN_MMA=1 forces the scalar path.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_MOSS_CODEC_NO_ATTN_MMA") == nullptr) {
    _lib_sdpa_mma = mc->load_library("sdpa_mma");
    _fn_sdpa_mma = _lib_sdpa_mma.function("sdpa_causal_mma2_d64_f16");
    _attn_mma_available = _fn_sdpa_mma.valid();
    _use_attn_mma = _attn_mma_available;
  }
  return _ok;
}

SharedBuffer
MetalMossCodecV2::decode_latent(
    const std::vector<std::vector<std::int32_t>>& codes, int T)
{
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  const int CD = _codebook_dim, RV = _rvq_dim, OD = _code_dim;

  // Host-gather each codebook's rows for the T frames -> [T, 8].
  std::vector<SharedBuffer> gathered((std::size_t)_n_vq);
  for (int cb = 0; cb < _n_vq; ++cb) {
    gathered[(std::size_t)cb] = buf((std::size_t)T * CD);
    auto* g = static_cast<_Float16*>(gathered[(std::size_t)cb].contents());
    const auto* tbl =
        static_cast<const _Float16*>(_codebook[(std::size_t)cb].contents());
    for (int t = 0; t < T; ++t) {
      int code = codes[(std::size_t)t][(std::size_t)cb];
      if (code < 0) { code = 0; }
      if (code >= _codebook_size) { code = _codebook_size - 1; }
      for (int e = 0; e < CD; ++e) {
        g[(std::size_t)t * CD + e] = tbl[(std::size_t)code * CD + e];
      }
    }
  }
  SharedBuffer emb = buf((std::size_t)T * RV);
  std::memset(emb.contents(), 0, (std::size_t)T * RV * 2);
  SharedBuffer tmp = buf((std::size_t)T * RV);
  SharedBuffer hidden = buf((std::size_t)T * OD);

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm_bias = [&](const SharedBuffer& xin, const SharedBuffer& w,
                         const SharedBuffer& b, const SharedBuffer& y, int M,
                         int N, int K) {
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, xin); enc.set_buffer(1, w); enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    for (int cb = 0; cb < _n_vq; ++cb) {
      gemm_bias(gathered[(std::size_t)cb], _q_outw[(std::size_t)cb],
                _q_outb[(std::size_t)cb], tmp, T, RV, CD);
      residual(emb, tmp, emb, T * RV);
    }
    gemm_bias(emb, _rvq_outw, _rvq_outb, hidden, T, OD, RV);
  }
  stream.commit().wait();
  return hidden;
}

SharedBuffer
MetalMossCodecV2::run_stage_(const Stage& st, int T, const SharedBuffer& in,
                             std::vector<SharedBuffer>* kc,
                             std::vector<SharedBuffer>* vc, int pos,
                             int ring_cap)
{
  const bool streaming = (kc != nullptr && vc != nullptr);
  const StageCfg& c = st.cfg;
  const int d = c.d_model, heads = c.n_heads, hd = d / heads, ff = c.ff;
  const int in_dim = c.in_dim, out_dim = c.out_dim;
  const float scale = 1.0f / std::sqrt((float)hd);
  const float eps = 1e-5f;
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };

  SharedBuffer x = buf((std::size_t)T * d);
  SharedBuffer n1 = buf((std::size_t)T * d), qkv = buf((std::size_t)T * 3 * d);
  SharedBuffer q3 = buf((std::size_t)T * d), k3 = buf((std::size_t)T * d),
               v3 = buf((std::size_t)T * d);
  SharedBuffer qt = buf((std::size_t)T * d), kt = buf((std::size_t)T * d),
               vt = buf((std::size_t)T * d);
  SharedBuffer atb = buf((std::size_t)T * d), att = buf((std::size_t)T * d);
  SharedBuffer o = buf((std::size_t)T * d), o2 = buf((std::size_t)T * d);
  SharedBuffer h = buf((std::size_t)T * ff);
  SharedBuffer out = buf((std::size_t)T * out_dim);

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& y, int M, int N, int K) {
      if (_use_mma2) {
        // M5 matrix-core matmul2d GEMM y = x @ w^T (no bias). The tensor
        // extents clamp the M/N tails, so M/N need not be tile multiples.
        const bool deep = (K >= 6144);
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_gemm_mma_deep : _fn_gemm_mma);
        enc.set_buffer(0, xin); enc.set_buffer(1, w); enc.set_buffer(2, w);
        enc.set_buffer(3, y);
        enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, xin); enc.set_buffer(1, w); enc.set_buffer(2, w);
      enc.set_buffer(3, y);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto ln = [&](const SharedBuffer& xin, const SharedBuffer& w,
                  const SharedBuffer& b, const SharedBuffer& y, int R, int Hd) {
      enc.set_function(_fn_ln);
      enc.set_buffer(0, xin); enc.set_buffer(1, w); enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, Hd); enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& xin, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, xin); enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& inb, const SharedBuffer& outb, int Hh,
                      int Sd, int Wd, int off) {
      enc.set_function(_fn_hslice);
      enc.set_buffer(0, inb); enc.set_buffer(1, outb);
      enc.set_constant(2, Hh); enc.set_constant(3, Sd); enc.set_constant(4, Wd);
      enc.set_constant(5, off);
      const int zero = 0;
      enc.set_constant(6, zero); enc.set_constant(7, zero);
      enc.dispatch({(unsigned)(Hh * Wd), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& inb, const SharedBuffer& outb,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, inb); enc.set_buffer(1, outb);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, hd);
      enc.dispatch({(unsigned)hd, (unsigned)Bd, (unsigned)A},
                   {(unsigned)hd, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& outb, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, outb);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, xb); enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads); enc.set_constant(3, T); enc.set_constant(4, hd);
      const int off = pos;   // absolute position of the first (new) frame
      enc.set_constant(5, off);
      enc.dispatch({(unsigned)(hd / 2), (unsigned)T, (unsigned)heads},
                   {1, 1, 1});
    };
    // Attention geometry: one-shot decodes all T frames from local K/V
    // (linear addressing); streaming decodes the T NEW frames (query rows) at
    // absolute offset `pos` against the windowed K/V RING (ring_cap = the
    // stage's context, kv length = pos+T). The kernels already accept
    // (n_q, q_offset, kv_stride, window, ring_cap) so the same call serves both.
    const int Tkv      = streaming ? (pos + T) : T;
    const int qoff     = streaming ? pos : 0;
    const int kvstride = streaming ? ring_cap : T;
    const int ringcap  = streaming ? ring_cap : T;
    auto attn = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, const SharedBuffer& outb) {
      if (_use_attn_mma && hd == 64) {
        // M5 matrix-core windowed-causal flash attention (BQ=16 query tile,
        // 128 threads). Drop-in: same [heads,*,hd] q/k/v/out layout + causal +
        // window semantics; ring addressing shared with the scalar path.
        enc.set_function(_fn_sdpa_mma);
        enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
        enc.set_buffer(3, outb);
        enc.set_constant(4, scale);
        enc.set_constant(5, Tkv); enc.set_constant(6, hd);
        enc.set_constant(7, heads); enc.set_constant(8, heads);
        enc.set_constant(9, T); enc.set_constant(10, qoff);
        enc.set_constant(11, kvstride); enc.set_constant(12, c.context);
        enc.set_constant(13, streaming ? ringcap : 0);
        enc.dispatch({128, (unsigned)heads, (unsigned)((T + 15) / 16)},
                     {128, 1, 1});
        return;
      }
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, outb);
      enc.set_constant(4, scale);
      enc.set_constant(5, Tkv); enc.set_constant(6, hd); enc.set_constant(7, heads);
      enc.set_constant(8, heads); enc.set_constant(9, T);
      enc.set_constant(10, qoff); enc.set_constant(11, kvstride);
      enc.set_constant(12, c.context); enc.set_constant(13, ringcap);
      enc.dispatch({32, (unsigned)heads, (unsigned)T}, {32, 1, 1});
    };
    // Scatter the T new frames' per-head K (or V) rows into a windowed ring.
    auto ring_append = [&](const SharedBuffer& src, const SharedBuffer& ring) {
      enc.set_function(_fn_ring_append);
      enc.set_buffer(0, src); enc.set_buffer(1, ring);
      enc.set_constant(2, heads); enc.set_constant(3, T); enc.set_constant(4, hd);
      enc.set_constant(5, ring_cap); enc.set_constant(6, pos);
      enc.dispatch({(unsigned)hd, (unsigned)T, (unsigned)heads}, {1, 1, 1});
    };

    gemm(in, st.in_proj, x, T, d, in_dim);                   // input_proj
    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = st.layers[(std::size_t)l];
      ln(x, L.n1w, L.n1b, n1, T, d);
      gemm(n1, L.qkvw, qkv, T, 3 * d, d);
      hslice(qkv, q3, T, 3 * d, d, 0);
      hslice(qkv, k3, T, 3 * d, d, d);
      hslice(qkv, v3, T, 3 * d, d, 2 * d);
      transpose(q3, qt, T, heads);
      transpose(k3, kt, T, heads);
      transpose(v3, vt, T, heads);
      rope(qt);
      rope(kt);
      if (streaming) {
        // Append the new frames' K/V into this layer's window ring, then
        // attend the new queries over the ring (past + new).
        ring_append(kt, (*kc)[(std::size_t)l]);
        ring_append(vt, (*vc)[(std::size_t)l]);
        attn(qt, (*kc)[(std::size_t)l], (*vc)[(std::size_t)l], atb);
      } else {
        attn(qt, kt, vt, atb);
      }
      transpose(atb, att, heads, T);
      gemm(att, L.ow, o, T, d, d);
      residual(x, o, x, T * d);
      ln(x, L.n2w, L.n2b, n1, T, d);
      gemm(n1, L.fc1, h, T, ff, d);
      gelu(h, h, T * ff);
      gemm(h, L.fc2, o2, T, d, ff);
      residual(x, o2, x, T * d);
    }
    gemm(x, st.out_proj, out, T, out_dim, d);               // output_proj
  }
  stream.commit().wait();
  return out;
}

std::vector<float>
MetalMossCodecV2::decode(const std::vector<std::vector<std::int32_t>>& codes,
                         std::vector<std::vector<float>>* stages)
{
  std::vector<float> wave;
  const int T = (int)codes.size();
  if (T <= 0 || !_ok) { return wave; }

  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudioCodec,
                     kPerfLlmAudioCodecBegin, (std::uint64_t)T);

  // Opt-in per-phase profiler (VPIPE_MOSS_CODEC_PROFILE): RLFQ latent, each of
  // the 6 GPU transformer stages, the host patch-upsamples, and the final
  // de-interleave -- so the decode bottleneck is visible.
  const bool prof = std::getenv("VPIPE_MOSS_CODEC_PROFILE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double t_stage[6] = {0}, t_up[6] = {0}, t_latent = 0, t_deint = 0;
  const auto t_all0 = clk::now();

  auto chan_major = [](const SharedBuffer& b, int t, int cc) {
    std::vector<float> out((std::size_t)cc * t);
    const auto* s = static_cast<const _Float16*>(b.contents());
    for (int i = 0; i < t; ++i) {
      for (int c = 0; c < cc; ++c) {
        out[(std::size_t)c * t + i] = (float)s[(std::size_t)i * cc + c];
      }
    }
    return out;
  };

  const auto t_lat0 = clk::now();
  SharedBuffer cur = decode_latent(codes, T);   // [T, 768]
  if (prof) { t_latent = ms(t_lat0, clk::now()); }
  if (stages != nullptr) { stages->push_back(chan_major(cur, T, _code_dim)); }

  int curT = T;
  for (std::size_t si = 0; si < _stages.size(); ++si) {
    const Stage& st = _stages[si];
    const auto ts0 = clk::now();
    SharedBuffer out = run_stage_(st, curT, cur);   // [curT, out_dim]
    if (prof) { t_stage[si] = ms(ts0, clk::now()); }
    if (stages != nullptr) {
      stages->push_back(chan_major(out, curT, st.cfg.out_dim));
    }
    // Patch upsample (host): out[t][c*P+p] -> up[t*P+p][c].
    const auto tu0 = clk::now();
    const int P = st.cfg.patch, Cout = st.cfg.out_dim / P, Tout = curT * P;
    SharedBuffer up = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(out.contents());
    auto* d = static_cast<_Float16*>(up.contents());
    for (int t = 0; t < curT; ++t) {
      for (int co = 0; co < Cout; ++co) {
        for (int p = 0; p < P; ++p) {
          d[(std::size_t)(t * P + p) * Cout + co] =
              s[(std::size_t)t * st.cfg.out_dim + co * P + p];
        }
      }
    }
    if (prof) { t_up[si] = ms(tu0, clk::now()); }
    if (stages != nullptr) { stages->push_back(chan_major(up, Tout, Cout)); }
    cur = std::move(up);
    curT = Tout;
  }

  const auto td0 = clk::now();
  // cur is [curT, 1] = interleaved stereo samples. De-interleave:
  // flat[2k] -> ch0[k], flat[2k+1] -> ch1[k]. Return channel-major [ch0|ch1].
  const auto* w = static_cast<const _Float16*>(cur.contents());
  if (_channels == 2) {
    const int per = curT / 2;
    wave.resize((std::size_t)curT);
    for (int k = 0; k < per; ++k) {
      wave[(std::size_t)k] = (float)w[(std::size_t)(2 * k)];
      wave[(std::size_t)per + k] = (float)w[(std::size_t)(2 * k + 1)];
    }
  } else {
    wave.resize((std::size_t)curT);
    for (int i = 0; i < curT; ++i) { wave[(std::size_t)i] = (float)w[i]; }
  }
  if (prof) {
    t_deint = ms(td0, clk::now());
    const double wall = ms(t_all0, clk::now());
    std::fprintf(stderr,
        "[codec-v2-prof] T=%d %.1f ms (%.2fx RT for %.2fs audio) | latent=%.2f "
        "| stages(gpu)=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f | upsample(host)=%.2f "
        "total | deint=%.2f\n",
        T, wall, (wall / 1000.0) / ((double)(curT / 2) / _sample_rate),
        (double)(curT / 2) / _sample_rate, t_latent,
        t_stage[0], t_stage[1], t_stage[2], t_stage[3], t_stage[4], t_stage[5],
        t_up[0] + t_up[1] + t_up[2] + t_up[3] + t_up[4] + t_up[5], t_deint);
  }
  return wave;
}

std::unique_ptr<MetalMossCodecV2::StreamState>
MetalMossCodecV2::decode_stream_begin(int max_chunk_frames) const
{
  if (!_ok || max_chunk_frames <= 0) { return nullptr; }
  auto st = std::make_unique<StreamState>();
  const std::size_t ns = _stages.size();
  st->kc.resize(ns);
  st->vc.resize(ns);
  st->pos.assign(ns, 0);
  st->cap.assign(ns, 0);
  st->max_chunk = max_chunk_frames;
  // Frames entering stage si = latent frames * product of the earlier stages'
  // patch upsamples. Ring capacity = context + that stage's per-chunk frame
  // count, so a chunk's appends never evict context the chunk still needs.
  int patchprod = 1;
  for (std::size_t si = 0; si < ns; ++si) {
    const StageCfg& c = _stages[si].cfg;
    const int hd = c.d_model / c.n_heads;
    const int cap = c.context + max_chunk_frames * patchprod;
    st->cap[si] = cap;
    const std::size_t ring =
        (std::size_t)c.n_heads * (std::size_t)cap * (std::size_t)hd;
    st->kc[si].reserve((std::size_t)c.n_layers);
    st->vc[si].reserve((std::size_t)c.n_layers);
    for (int l = 0; l < c.n_layers; ++l) {
      st->kc[si].push_back(_mc->make_shared_buffer(ring * 2));
      st->vc[si].push_back(_mc->make_shared_buffer(ring * 2));
    }
    patchprod *= c.patch;
  }
  return st;
}

std::vector<float>
MetalMossCodecV2::decode_stream_chunk(
    StreamState& st, const std::vector<std::vector<std::int32_t>>& codes)
{
  std::vector<float> wave;
  const int Cnew = (int)codes.size();
  if (Cnew <= 0 || !_ok || st.pos.size() != _stages.size()
      || Cnew > st.max_chunk) {
    return wave;
  }

  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudioCodec,
                     kPerfLlmAudioCodecBegin, (std::uint64_t)Cnew);

  SharedBuffer cur = decode_latent(codes, Cnew);   // [Cnew, 768]
  int curT = Cnew;
  for (std::size_t si = 0; si < _stages.size(); ++si) {
    const Stage& stg = _stages[si];
    const int pos = st.pos[si];
    SharedBuffer out =
        run_stage_(stg, curT, cur, &st.kc[si], &st.vc[si], pos, st.cap[si]);
    st.pos[si] = pos + curT;
    // Patch upsample (host, within-frame): out[t][c*P+p] -> up[t*P+p][c].
    const int P = stg.cfg.patch, Cout = stg.cfg.out_dim / P, Tout = curT * P;
    SharedBuffer up = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(out.contents());
    auto* d = static_cast<_Float16*>(up.contents());
    for (int t = 0; t < curT; ++t) {
      for (int co = 0; co < Cout; ++co) {
        for (int p = 0; p < P; ++p) {
          d[(std::size_t)(t * P + p) * Cout + co] =
              s[(std::size_t)t * stg.cfg.out_dim + co * P + p];
        }
      }
    }
    cur = std::move(up);
    curT = Tout;
  }

  // cur is [curT, 1] interleaved stereo -> channel-major [ch0 | ch1]. The
  // chunk's samples are contiguous, so de-interleaving is fully local.
  const auto* w = static_cast<const _Float16*>(cur.contents());
  if (_channels == 2) {
    const int per = curT / 2;
    wave.resize((std::size_t)curT);
    for (int k = 0; k < per; ++k) {
      wave[(std::size_t)k] = (float)w[(std::size_t)(2 * k)];
      wave[(std::size_t)per + k] = (float)w[(std::size_t)(2 * k + 1)];
    }
  } else {
    wave.resize((std::size_t)curT);
    for (int i = 0; i < curT; ++i) { wave[(std::size_t)i] = (float)w[i]; }
  }
  return wave;
}

std::vector<std::vector<std::int32_t>>
MetalMossCodecV2::encode(const std::vector<float>& wave,
                         std::vector<std::vector<float>>* stages)
{
  std::vector<std::vector<std::int32_t>> codes;
  if (!_ok || !_with_encoder || wave.empty()) { return codes; }

  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmAudioCodec,
                     kPerfLlmAudioCodecBegin, (std::uint64_t)wave.size());

  // Opt-in per-phase profiler (VPIPE_MOSS_CODEC_PROFILE): each of the 6 GPU
  // encoder transformer stages, the host DOWN patch-reshapes, and the RLFQ
  // residual encode -- the encode-path twin of decode()'s profiler.
  const bool prof = std::getenv("VPIPE_MOSS_CODEC_PROFILE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  double t_stage[6] = {0}, t_down[6] = {0}, t_rvq = 0;
  const auto t_all0 = clk::now();

  auto chan_major = [](const SharedBuffer& b, int t, int cc) {
    std::vector<float> out((std::size_t)cc * t);
    const auto* s = static_cast<const _Float16*>(b.contents());
    for (int i = 0; i < t; ++i) {
      for (int c = 0; c < cc; ++c) {
        out[(std::size_t)c * t + i] = (float)s[(std::size_t)i * cc + c];
      }
    }
    return out;
  };

  // `wave` is channel-major [ch0(per) | ch1(per)]. Pad per-channel to a whole
  // 3840x frame, then channel-interleave: flat[2k]=ch0[k], flat[2k+1]=ch1[k]
  // (the inverse of decode's de-interleave). `hop` = downsample_rate.
  const int hop = 3840;
  int per = (_channels == 2) ? (int)(wave.size() / 2) : (int)wave.size();
  const int T = (per + hop - 1) / hop;     // codes/frames
  const int per_pad = T * hop;
  const int L = (_channels == 2) ? (2 * per_pad) : per_pad;  // interleaved len

  SharedBuffer cur = _mc->make_shared_buffer((std::size_t)L * 2);
  if (cur.empty()) { return codes; }
  {
    auto* w = static_cast<_Float16*>(cur.contents());
    if (_channels == 2) {
      const float* ch0 = wave.data();
      const float* ch1 = wave.data() + per;
      for (int k = 0; k < per_pad; ++k) {
        w[(std::size_t)(2 * k)] =
            (_Float16)(k < per ? ch0[(std::size_t)k] : 0.0f);
        w[(std::size_t)(2 * k + 1)] =
            (_Float16)(k < per ? ch1[(std::size_t)k] : 0.0f);
      }
    } else {
      for (int k = 0; k < per_pad; ++k) {
        w[(std::size_t)k] = (_Float16)(k < per ? wave[(std::size_t)k] : 0.0f);
      }
    }
  }
  int curT = L, curC = 1;

  // DOWN patch-reshape [Tin, Cin] -> [Tin/P, Cin*P], out[t][c*P+p] =
  // in[t*P+p][c] (the inverse of decode's UP patch).
  auto down_patch = [&](const SharedBuffer& inb, int Tin, int Cin, int P) {
    const int Tout = Tin / P, Cout = Cin * P;
    SharedBuffer outb = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(inb.contents());
    auto* d = static_cast<_Float16*>(outb.contents());
    for (int t = 0; t < Tout; ++t) {
      for (int cc = 0; cc < Cin; ++cc) {
        for (int p = 0; p < P; ++p) {
          d[(std::size_t)t * Cout + cc * P + p] =
              s[(std::size_t)(t * P + p) * Cin + cc];
        }
      }
    }
    return outb;
  };

  for (std::size_t si = 0; si < _enc_stages.size(); ++si) {
    const Stage& st = _enc_stages[si];
    const int P = st.cfg.patch;
    const auto td0 = clk::now();
    SharedBuffer patched = down_patch(cur, curT, curC, P);
    if (prof) { t_down[si] = ms(td0, clk::now()); }
    const int Tst = curT / P;               // frames into this stage
    const auto ts0 = clk::now();
    cur = run_stage_(st, Tst, patched);     // [Tst, out_dim]
    if (prof) { t_stage[si] = ms(ts0, clk::now()); }
    curT = Tst;
    curC = st.cfg.out_dim;
    if (stages != nullptr) { stages->push_back(chan_major(cur, curT, curC)); }
  }
  // cur = [T, code_dim].
  const auto tr0 = clk::now();
  std::vector<std::vector<std::int32_t>> out = encode_rvq_(cur, curT);
  if (prof) {
    t_rvq = ms(tr0, clk::now());
    const double wall = ms(t_all0, clk::now());
    std::fprintf(stderr,
        "[codec-v2-enc-prof] T=%d %.1f ms (%.2fx RT for %.2fs audio) | "
        "stages(gpu)=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f | down(host)=%.2f total | "
        "rvq(host)=%.2f\n",
        curT, wall, (wall / 1000.0) / ((double)per / _sample_rate),
        (double)per / _sample_rate, t_stage[0], t_stage[1], t_stage[2],
        t_stage[3], t_stage[4], t_stage[5],
        t_down[0] + t_down[1] + t_down[2] + t_down[3] + t_down[4] + t_down[5],
        t_rvq);
  }
  return out;
}

std::vector<std::vector<std::int32_t>>
MetalMossCodecV2::encode_rvq_(const SharedBuffer& hidden, int T)
{
  const int CD = _codebook_dim, RV = _rvq_dim, OD = _code_dim;
  const int Csz = _codebook_size, NV = _n_vq;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  if (T <= 0) { return codes; }

  // hid [T,OD], Win [RV,OD], bin [RV] (the global input_proj).
  const auto* hid = static_cast<const _Float16*>(hidden.contents());
  const auto* Win = static_cast<const _Float16*>(_rvq_inw.contents());
  const auto* bin = static_cast<const _Float16*>(_rvq_inb.contents());

  std::vector<float> resid((std::size_t)RV);   // running residual (rvq_dim)
  std::vector<float> z((std::size_t)CD);        // per-codebook in_proj output
  for (int t = 0; t < T; ++t) {
    // Global input_proj: resid = Win @ hidden[t] + bin.
    const _Float16* hr = hid + (std::size_t)t * OD;
    for (int o = 0; o < RV; ++o) {
      const _Float16* wr = Win + (std::size_t)o * OD;
      float acc = (float)bin[o];
      for (int k = 0; k < OD; ++k) { acc += (float)wr[k] * (float)hr[k]; }
      resid[(std::size_t)o] = acc;
    }
    // LFQ residual loop over the n_vq codebooks.
    for (int cb = 0; cb < NV; ++cb) {
      // Wq [CD,RV], bq [CD] (this codebook's in_proj).
      const auto* Wq =
          static_cast<const _Float16*>(_q_inw[(std::size_t)cb].contents());
      const auto* bq =
          static_cast<const _Float16*>(_q_inb[(std::size_t)cb].contents());
      // z = in_proj_cb(resid).
      for (int e = 0; e < CD; ++e) {
        const _Float16* wr = Wq + (std::size_t)e * RV;
        float acc = (float)bq[e];
        for (int k = 0; k < RV; ++k) {
          acc += (float)wr[k] * resid[(std::size_t)k];
        }
        z[(std::size_t)e] = acc;
      }
      // Cosine-nearest over the L2-normalized codebook: argmax(z . cb_norm)
      // == argmin squared distance of the normalized vectors (z's norm is a
      // positive per-step constant, so it does not change the argmax).
      const auto* cbn =
          static_cast<const _Float16*>(
              _codebook_norm[(std::size_t)cb].contents());  // [Csz,CD]
      int best = 0;
      float bestdot = -std::numeric_limits<float>::infinity();
      for (int code = 0; code < Csz; ++code) {
        const _Float16* cr = cbn + (std::size_t)code * CD;
        float dot = 0.0f;
        for (int e = 0; e < CD; ++e) {
          dot += (float)cr[e] * z[(std::size_t)e];
        }
        if (dot > bestdot) { bestdot = dot; best = code; }
      }
      codes[(std::size_t)t][(std::size_t)cb] = best;
      // residual -= out_proj(raw codebook[cb][best]) (== the decode path).
      // Wout [RV,CD], bout [RV], cbr [Csz,CD] (raw codebook + this out_proj).
      const auto* Wout =
          static_cast<const _Float16*>(_q_outw[(std::size_t)cb].contents());
      const auto* bout =
          static_cast<const _Float16*>(_q_outb[(std::size_t)cb].contents());
      const auto* cbr =
          static_cast<const _Float16*>(_codebook[(std::size_t)cb].contents());
      const _Float16* crow = cbr + (std::size_t)best * CD;
      for (int o = 0; o < RV; ++o) {
        const _Float16* wr = Wout + (std::size_t)o * CD;
        float acc = (float)bout[o];
        for (int e = 0; e < CD; ++e) { acc += (float)wr[e] * (float)crow[e]; }
        resid[(std::size_t)o] -= acc;
      }
    }
  }
  return codes;
}

}  // namespace vpipe::genai
