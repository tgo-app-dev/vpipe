#include "generative-models/llama3/metal-llama-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param
// block the vendored steel attention kernel (attn_steel_h_bd128) reads.
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
}  // namespace

std::unique_ptr<MetalLlamaModel>
MetalLlamaModel::load(const std::string& model_dir,
                      metal_compute::MetalCompute* mc, const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) {
    return nullptr;
  }

  auto m = std::unique_ptr<MetalLlamaModel>(new MetalLlamaModel());
  m->_cfg = cfg;
  m->_mc = mc;

  m->_lib_qmv = mc->load_library("affine_qmv");
  m->_lib_qmm = mc->load_library("affine_qmm_steel");
  m->_lib_rms = mc->load_library("rms_norm");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_rope = mc->load_library("rope");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_lib_attn = mc->load_library("attn_steel");
  m->_fn_qmv = m->_lib_qmv.function("affine_qmv_w4g64");
  m->_fn_qmv_add = m->_lib_qmv.function("affine_qmv_w4g64_add");
  m->_fn_qmv_swiglu = m->_lib_qmv.function("affine_qmv_swiglu_w4g64");
  m->_fn_qmm = m->_lib_qmm.function("affine_qmm_steel_w4g64");
  m->_fn_qmm_swiglu = m->_lib_qmm.function("affine_qmm_swiglu_w4g64");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  // simd_sum RMSNorm (dispatched at 256). The gemma-introduced tree
  // rms_norm_f16 (f1ab287) strides by a fixed 512 -> silently wrong at 256;
  // rms_norm_fast_f16 is the threadgroup-size-agnostic simd_sum (== pre-f1ab287
  // behavior, no RMS_PAD cap). See gpu-kernels/metal/ops/rms_norm.metal.
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_embed = m->_lib_elt.function("dequant_embed_gather_f16");
  m->_fn_argmax = m->_lib_elt.function("argmax_f16");
  m->_fn_sample = m->_lib_elt.function("sample_topp_f16");
  // Two-stage parallel argmax + histogram multi-tg sampler (defaults; the
  // single-tg _fn_argmax/_fn_sample are the ARGMAX1/SAMPLE1 fallbacks).
  m->_fn_argmax_partial    = m->_lib_elt.function("argmax_partial_f16");
  m->_fn_argmax_combine    = m->_lib_elt.function("argmax_combine_f16");
  m->_fn_smp_max_partial   = m->_lib_elt.function("sample_max_partial_f16");
  m->_fn_smp_max_combine   = m->_lib_elt.function("sample_max_combine_f16");
  m->_fn_smp_zhist_partial = m->_lib_elt.function("sample_zhist_partial_f16");
  m->_fn_smp_zhist_combine = m->_lib_elt.function("sample_zhist_combine_f16");
  m->_fn_smp_thresh        = m->_lib_elt.function("sample_thresh_f16");
  m->_fn_smp_pick_partial  = m->_lib_elt.function("sample_pick_partial_f16");
  m->_fn_smp_pick_combine  = m->_lib_elt.function("sample_pick_combine_f16");
  m->_fn_kv_write_paged = m->_lib_elt.function("kv_write_paged_f16");
  m->_fn_kv_gather_paged = m->_lib_elt.function("kv_gather_paged_f16");
  m->_fn_rope = m->_lib_rope.function("rope_f16");
  m->_fn_sdpa_paged = m->_lib_sdpa.function("sdpa_paged_causal_f16");
  m->_fn_sdpa_paged_mb = m->_lib_sdpa.function("sdpa_paged_mb_f16");
  m->_fn_sdpa_paged_mma = m->_lib_sdpa.function("sdpa_paged_mma_f16");
  // Flash-decode-GQA serial attention (head_dim <= 256). Optional: decode
  // falls back to sdpa_paged_mb/scalar when either kernel is absent.
  m->_fn_sdpa_gqa = m->_lib_sdpa.function("sdpa_paged_gqa_mb256_f16");
  m->_fn_sdpa_gqa_merge = m->_lib_sdpa.function("sdpa_gqa_merge_f16");
  if (!m->_fn_qmv.valid() || !m->_fn_qmv_add.valid() || !m->_fn_rms.valid() ||
      !m->_fn_qmv_swiglu.valid() || !m->_fn_qmm_swiglu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_embed.valid() ||
      !m->_fn_rope.valid() || !m->_fn_qmm.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_kv_write_paged.valid() ||
      !m->_fn_kv_gather_paged.valid() ||
      !m->_fn_sdpa_paged.valid() || !m->_fn_sdpa_paged_mb.valid() ||
      !m->_fn_sdpa_paged_mma.valid() || !m->_fn_argmax.valid() ||
      !m->_fn_sample.valid() || !m->_lib_attn.valid()) {
    return nullptr;
  }
  if (const char* e = std::getenv("VPIPE_SDPA_MB_MIN")) {
    const int v = std::atoi(e);
    if (v >= 0) { m->_sdpa_mb_min = v; }
  }
  // Flash-decode-GQA serial attention (head_dim <= 256, GQA G=Hq/Hkv <= 4):
  // read each KV head once for all G query heads. Default ON when capable;
  // VPIPE_GQA_ATTN=0/1 overrides, VPIPE_GQA_SPLIT sets the split.
  const int gqa_g =
      (cfg.n_kv_heads > 0) ? cfg.n_heads / cfg.n_kv_heads : 0;
  const bool gqa_capable =
      m->_fn_sdpa_gqa.valid() && m->_fn_sdpa_gqa_merge.valid()
      && cfg.head_dim >= 32 && cfg.head_dim <= 256
      && (cfg.head_dim % 32 == 0) && cfg.n_kv_heads > 0
      && (cfg.n_heads % cfg.n_kv_heads == 0) && gqa_g >= 1 && gqa_g <= 4;
  m->_gqa_attn = gqa_capable;
  if (const char* e = std::getenv("VPIPE_GQA_ATTN")) {
    m->_gqa_attn = gqa_capable && (std::atoi(e) != 0);
  }
  if (const char* e = std::getenv("VPIPE_GQA_SPLIT")) {
    const int v = std::atoi(e);
    if (v >= 1 && v <= 64) { m->_gqa_split = v; }
  }

  auto qt = [&](const std::string& pfx, SharedBuffer& w, SharedBuffer& s,
                SharedBuffer& b) -> bool {
    w = wts->load(pfx + ".weight", mc);
    s = wts->load(pfx + ".scales", mc);
    b = wts->load(pfx + ".biases", mc);
    return !w.empty() && !s.empty() && !b.empty();
  };

  // Build the fused gate/up weight: interleave the two [ffn, hidden]
  // quantized matrices by output feature into one [2*ffn, hidden] matrix
  // (row 2g = gate row g, row 2g+1 = up row g) so the SwiGLU activation
  // fuses into the matmul store (affine_qmm_swiglu / affine_qmv_swiglu).
  // Quantized layout: weight uint32 [rows, K/8], scales/biases f16
  // [rows, K/64]; interleaving is a per-row byte copy.
  const int Kc = cfg.hidden;
  const int Fc = cfg.ffn_inner;
  const std::size_t wrow = Kc / 8;        // uint32 per weight row
  const std::size_t grow = Kc / 64;       // f16 per scale/bias row
  auto interleave = [&](const SharedBuffer& gw, const SharedBuffer& gs,
                        const SharedBuffer& gb, const SharedBuffer& uw,
                        const SharedBuffer& us, const SharedBuffer& ub,
                        SharedBuffer& ow, SharedBuffer& os,
                        SharedBuffer& ob) -> bool {
    ow = mc->make_shared_buffer((std::size_t)2 * Fc * wrow * 4);
    os = mc->make_shared_buffer((std::size_t)2 * Fc * grow * 2);
    ob = mc->make_shared_buffer((std::size_t)2 * Fc * grow * 2);
    if (ow.empty() || os.empty() || ob.empty()) { return false; }
    const auto* gwp = static_cast<const std::uint32_t*>(gw.contents());
    const auto* uwp = static_cast<const std::uint32_t*>(uw.contents());
    auto* owp = static_cast<std::uint32_t*>(ow.contents());
    const auto* gsp = static_cast<const std::uint16_t*>(gs.contents());
    const auto* usp = static_cast<const std::uint16_t*>(us.contents());
    auto* osp = static_cast<std::uint16_t*>(os.contents());
    const auto* gbp = static_cast<const std::uint16_t*>(gb.contents());
    const auto* ubp = static_cast<const std::uint16_t*>(ub.contents());
    auto* obp = static_cast<std::uint16_t*>(ob.contents());
    for (std::size_t g = 0; g < (std::size_t)Fc; ++g) {
      std::memcpy(owp + (2 * g) * wrow, gwp + g * wrow, wrow * 4);
      std::memcpy(owp + (2 * g + 1) * wrow, uwp + g * wrow, wrow * 4);
      std::memcpy(osp + (2 * g) * grow, gsp + g * grow, grow * 2);
      std::memcpy(osp + (2 * g + 1) * grow, usp + g * grow, grow * 2);
      std::memcpy(obp + (2 * g) * grow, gbp + g * grow, grow * 2);
      std::memcpy(obp + (2 * g + 1) * grow, ubp + g * grow, grow * 2);
    }
    return true;
  };

  m->_layers.resize(cfg.n_layers);
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p = "model.layers." + std::to_string(L) + ".";
    Layer& ly = m->_layers[L];
    ly.in_ln = wts->load(p + "input_layernorm.weight", mc);
    ly.post_ln = wts->load(p + "post_attention_layernorm.weight", mc);
    bool ok = !ly.in_ln.empty() && !ly.post_ln.empty();
    ok = ok && qt(p + "self_attn.q_proj", ly.qw, ly.qs, ly.qb);
    ok = ok && qt(p + "self_attn.k_proj", ly.kw, ly.ks, ly.kb);
    ok = ok && qt(p + "self_attn.v_proj", ly.vw, ly.vs, ly.vb);
    ok = ok && qt(p + "self_attn.o_proj", ly.ow, ly.os, ly.ob);
    SharedBuffer gw, gs, gb, uw, us, ub;
    ok = ok && qt(p + "mlp.gate_proj", gw, gs, gb);
    ok = ok && qt(p + "mlp.up_proj", uw, us, ub);
    ok = ok && interleave(gw, gs, gb, uw, us, ub, ly.guw, ly.gus, ly.gub);
    ok = ok && qt(p + "mlp.down_proj", ly.dw, ly.ds, ly.db);
    if (!ok) {
      return nullptr;
    }
  }

  bool ok = qt("model.embed_tokens", m->_embed_w, m->_embed_s, m->_embed_b);
  ok = ok && qt("lm_head", m->_lm_w, m->_lm_s, m->_lm_b);
  m->_final_ln = wts->load("model.norm.weight", mc);
  if (!ok || m->_final_ln.empty()) {
    return nullptr;
  }

  // llama3-scaled inv_freq (the rotation rate).
  const int half = cfg.head_dim / 2;
  m->_inv_freq = mc->make_shared_buffer((std::size_t)half * sizeof(float));
  auto* invf = static_cast<float*>(m->_inv_freq.contents());
  const float kPi = 3.14159265358979323846f;
  for (int i = 0; i < half; ++i) {
    const float freq =
        1.0f / std::pow(cfg.rope_theta, (2.0f * i) / cfg.head_dim);
    const float wavelen = 2.0f * kPi / freq;
    const float low_wl = cfg.rope_orig_ctx / cfg.rope_low_freq;
    const float high_wl = cfg.rope_orig_ctx / cfg.rope_high_freq;
    float v;
    if (wavelen > low_wl) {
      v = freq / cfg.rope_factor;
    } else if (wavelen < high_wl) {
      v = freq;
    } else {
      const float sm = (cfg.rope_orig_ctx / wavelen - cfg.rope_low_freq) /
          (cfg.rope_high_freq - cfg.rope_low_freq);
      v = (1.0f - sm) * freq / cfg.rope_factor + sm * freq;
    }
    invf[i] = v;
  }

  // Backend-native embedding + KV (shared metal LLM abstractions).
  m->_muxer = std::make_unique<MetalTokenMuxer>(
      mc, &m->_embed_w, &m->_embed_s, &m->_embed_b, cfg.hidden);
  if (!m->_muxer->valid()) {
    return nullptr;
  }
  // Shared ContextManager with its metal-compute backend (Spec::metal).
  // Llama is pure attention: no linear/SSM layers, so every layer gets a
  // paged K/V pool.
  ContextManager::Spec cspec;
  cspec.metal = mc;
  cspec.n_layers = cfg.n_layers;
  cspec.n_kv_heads = cfg.n_kv_heads;
  cspec.head_dim = cfg.head_dim;
  cspec.max_seq = cfg.max_seq;
  cspec.page_tokens = cfg.page_tokens;
  cspec.max_pages = cfg.max_pages;
  m->_ctx = std::make_unique<ContextManager>(cspec, nullptr);
  m->_cid = m->_ctx->acquire_root();
  if (!m->_cid.valid()) {
    return nullptr;
  }
  // Page-table scratch: {page_id, n_valid, global_start} per page.
  m->_pgtab = mc->make_shared_buffer(
      (std::size_t)m->_ctx->max_pages() * 3 * sizeof(std::int32_t));
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));

  // Scratch (one decode token, n=1).
  const int H = cfg.hidden;
  const int qd = cfg.n_heads * cfg.head_dim;
  const int kd = cfg.n_kv_heads * cfg.head_dim;
  m->_x = mc->make_shared_buffer((std::size_t)H * 2);
  m->_hn = mc->make_shared_buffer((std::size_t)H * 2);
  m->_q = mc->make_shared_buffer((std::size_t)qd * 2);
  m->_k = mc->make_shared_buffer((std::size_t)kd * 2);
  m->_v = mc->make_shared_buffer((std::size_t)kd * 2);
  m->_attn = mc->make_shared_buffer((std::size_t)qd * 2);
  m->_ao = mc->make_shared_buffer((std::size_t)H * 2);
  // _sg holds the fused gate/up+SwiGLU output [1, ffn] for decode.
  m->_sg = mc->make_shared_buffer((std::size_t)cfg.ffn_inner * 2);
  m->_mo = mc->make_shared_buffer((std::size_t)H * 2);
  m->_logits = mc->make_shared_buffer((std::size_t)cfg.vocab * 2);
  // Two-stage-argmax + histogram-sampler scratch (mirrors Gemma/Qwen).
  // Allocated once at load; reused every decode step (serial encoder).
  m->_d_argmax_part =
      mc->make_shared_buffer((std::size_t)2 * kArgmaxM * sizeof(float));
  m->_d_smp_maxpart = mc->make_shared_buffer((std::size_t)kSampM
                                             * sizeof(float));
  m->_d_smp_hpart = mc->make_shared_buffer(
      (std::size_t)kSampM * (2 * kSampB + 1) * sizeof(float));
  m->_d_smp_hist = mc->make_shared_buffer(
      (std::size_t)(2 * kSampB + 1) * sizeof(float));
  m->_d_smp_maxl = mc->make_shared_buffer(sizeof(float));
  m->_d_smp_wt = mc->make_shared_buffer(sizeof(float));
  m->_d_smp_pickpart =
      mc->make_shared_buffer((std::size_t)2 * kSampM * sizeof(float));
  // Flash-decode-GQA partials (f32): O [Hq,split,D], m/l [Hq,split].
  if (m->_gqa_attn) {
    const std::size_t sp = (std::size_t)m->_gqa_split;
    const std::size_t Hq = (std::size_t)cfg.n_heads;
    const std::size_t Dd = (std::size_t)cfg.head_dim;
    m->_d_gqa_oacc = mc->make_shared_buffer(Hq * sp * Dd * sizeof(float));
    m->_d_gqa_m = mc->make_shared_buffer(Hq * sp * sizeof(float));
    m->_d_gqa_l = mc->make_shared_buffer(Hq * sp * sizeof(float));
  }
  return m;
}

std::vector<float>
MetalLlamaModel::forward(ContextId cid, std::int32_t token_id)
{
  const Config& c = _cfg;
  const int H = c.hidden;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads, D = c.head_dim;
  const int qd = Hq * D, kd = Hkv * D;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);

  // Reserve this token's KV slot in the paged cache. The page+slot are
  // identical across layers (the page list is per-context); the
  // absolute position drives rope. A shared (post-branch) tail page
  // forces a fresh private page here -- the no-copy divergence point.
  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) {
    return {};
  }
  const int pos = slot.position;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int page_tokens = _ctx->page_tokens();

  // Page table (id, n_valid, global_start) for this context's pages,
  // reused across all layers in this forward.
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  // Embedding via the metal TokenMuxer (its own command buffer), then
  // use it as the residual stream (_x) for the layer chain below.
  std::int32_t id1 = token_id;
  metal_compute::SharedBuffer emb =
      _muxer->fetch_text(std::span<const std::int32_t>(&id1, 1));
  if (emb.empty()) {
    return {};
  }
  std::memcpy(_x.contents(), emb.contents(), (std::size_t)H * 2);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();

    auto qmv = [&](const SharedBuffer& w, const SharedBuffer& s,
                   const SharedBuffer& b, const SharedBuffer& x,
                   const SharedBuffer& y, int K, int N) {
      enc.set_function(_fn_qmv);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, x);
      enc.set_buffer(4, y);
      enc.set_constant(5, K);
      enc.set_constant(6, N);
      enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
    };
    auto rms = [&](const SharedBuffer& x, const SharedBuffer& w,
                   const SharedBuffer& y) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y);
      enc.set_constant(3, H);
      enc.set_constant(4, eps);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
    };
    auto rope = [&](const SharedBuffer& x, int heads) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, x);
      enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads);
      const int one = 1;
      enc.set_constant(3, one);
      enc.set_constant(4, D);
      enc.set_constant(5, pos);
      enc.dispatch({(unsigned)(D / 2), 1, (unsigned)heads},
                   {(unsigned)(D / 2), 1, 1});
    };
    // Write the new token's K/V into its page at slot_offset (cnt=1).
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      enc.set_function(_fn_kv_write_paged);
      enc.set_buffer(0, src);
      enc.set_buffer(1, pool, page_off);
      enc.set_constant(2, page_tokens);
      enc.set_constant(3, D);
      const int one = 1;
      enc.set_constant(4, one);            // n_src
      const int zero = 0;
      enc.set_constant(5, zero);           // src_off
      enc.set_constant(6, slot.slot_offset);
      enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
    };
    // y = x @ dequant(w)^T + res, fusing the residual add.
    auto qmv_add = [&](const SharedBuffer& w, const SharedBuffer& s,
                       const SharedBuffer& b, const SharedBuffer& x,
                       const SharedBuffer& y, const SharedBuffer& res,
                       int K, int N) {
      enc.set_function(_fn_qmv_add);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, x);
      enc.set_buffer(4, y);
      enc.set_constant(5, K);
      enc.set_constant(6, N);
      enc.set_buffer(7, res);
      enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
    };

    // _x already holds the embedding (copied from the muxer above).
    for (int L = 0; L < c.n_layers; ++L) {
      Layer& ly = _layers[L];
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);

      rms(_x, ly.in_ln, _hn);
      qmv(ly.qw, ly.qs, ly.qb, _hn, _q, H, qd);
      qmv(ly.kw, ly.ks, ly.kb, _hn, _k, H, kd);
      qmv(ly.vw, ly.vs, ly.vb, _hn, _v, H, kd);
      rope(_q, Hq);
      rope(_k, Hkv);
      kv_write(_k, kp);
      kv_write(_v, vp);

      // Long-context decode: the single-simdgroup kernel scans the KV
      // serially (only 32 lanes), so at high pos it dominates the
      // per-layer critical path. Switch to the multi-simdgroup kernel
      // (32 simdgroups stride the KV + cross-reduce) past the threshold.
      const bool use_mb = (pos + 1) >= _sdpa_mb_min;
      if (_gqa_attn && use_mb && !_d_gqa_oacc.empty()) {
        // Flash-decode-GQA: read each KV head once for all G query heads.
        encode_gqa_attn_(enc, _q, kp, vp, _attn, _pgtab, 0, scale, D, Hq,
                         Hkv, pos, page_tokens, n_pages);
      } else {
      enc.set_function(use_mb ? _fn_sdpa_paged_mb : _fn_sdpa_paged);
      enc.set_buffer(0, _q);
      enc.set_buffer(1, kp);
      enc.set_buffer(2, vp);
      enc.set_buffer(3, _attn);
      enc.set_constant(4, scale);
      enc.set_constant(5, D);
      enc.set_constant(6, Hq);
      enc.set_constant(7, Hkv);
      const int one = 1;
      enc.set_constant(8, one);            // n_q
      enc.set_constant(9, pos);            // q_offset
      enc.set_constant(10, page_tokens);
      enc.set_constant(11, n_pages);
      enc.set_buffer(12, _pgtab);
      const unsigned w = use_mb ? 32u * 32u : 32u;
      enc.dispatch({w, (unsigned)Hq, 1}, {w, 1, 1});
      }

      qmv_add(ly.ow, ly.os, ly.ob, _attn, _x, _x, H, H);

      rms(_x, ly.post_ln, _hn);
      // Fused gate/up GEMV + SwiGLU: one dispatch over the interleaved
      // weight writes silu(gate)*up straight to _sg [1, ffn].
      enc.set_function(_fn_qmv_swiglu);
      enc.set_buffer(0, ly.guw);
      enc.set_buffer(1, ly.gus);
      enc.set_buffer(2, ly.gub);
      enc.set_buffer(3, _hn);
      enc.set_buffer(4, _sg);
      enc.set_constant(5, H);
      enc.set_constant(6, 2 * c.ffn_inner);
      enc.dispatch({32, (unsigned)(c.ffn_inner / 2), 1}, {32, 2, 1});
      qmv_add(ly.dw, ly.ds, ly.db, _sg, _x, _x, c.ffn_inner, H);
    }

    rms(_x, _final_ln, _hn);
    qmv(_lm_w, _lm_s, _lm_b, _hn, _logits, H, c.vocab);
  }
  stream.commit().wait();

  std::vector<float> out((std::size_t)c.vocab);
  const auto* lp = static_cast<const _Float16*>(_logits.contents());
  for (int i = 0; i < c.vocab; ++i) {
    out[i] = (float)lp[i];
  }
  return out;
}

std::vector<float>
MetalLlamaModel::prefill(ContextId cid, const std::vector<std::int32_t>& ids)
{
  const Config& c = _cfg;
  const int n = (int)ids.size();
  const int H = c.hidden;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads, D = c.head_dim;
  const int qd = Hq * D, kd = Hkv * D, ffn = c.ffn_inner;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  if (n <= 0 || n > c.max_seq) {
    return {};
  }

  auto buf = [&](std::size_t elems) {
    return _mc->make_shared_buffer(elems * 2);
  };
  // Embedding via the metal TokenMuxer -> [n, hidden] residual stream.
  SharedBuffer x = _muxer->fetch_text(ids);
  if (x.empty()) {
    return {};
  }
  SharedBuffer hn = buf((std::size_t)n * H);
  SharedBuffer q = buf((std::size_t)n * qd), k = buf((std::size_t)n * kd),
               v = buf((std::size_t)n * kd);
  // qt gets one extra query tile (32*D halves) of slack: sdpa_paged_mma
  // reads 8-row Q frags, so the last partial tile may load up to 31 rows
  // past n (those rows are masked out / never written).
  SharedBuffer qt = buf((std::size_t)n * qd + (std::size_t)32 * D);
  SharedBuffer kt = buf((std::size_t)n * kd), vt = buf((std::size_t)n * kd);
  SharedBuffer at = buf((std::size_t)n * qd), att = buf((std::size_t)n * qd);
  SharedBuffer ao = buf((std::size_t)n * H);
  // Fused gate/up + SwiGLU writes silu(gate)*up straight here [n, ffn];
  // no separate gate/up buffers.
  SharedBuffer sg = buf((std::size_t)n * ffn);
  SharedBuffer logits = buf((std::size_t)c.vocab);

  // Reserve KV slots for all n tokens in the paged cache (host-side;
  // may grow the pools, so fetch kpool/vpool pointers AFTER this). Each
  // chunk lives in ONE page: {page byte offset, dst slot, src token
  // offset, count}. Then build the page table for the attention read.
  struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
  std::vector<Chunk> chunks;
  int q_offset = -1;   // absolute pos of the first prefill token
  for (int written = 0; written < n; ) {
    const int cap = _ctx->next_append_capacity(cid);
    const int cnt = std::min(n - written, cap);
    ContextManager::AppendSlot s = _ctx->append(cid, cnt);
    if (!s.valid()) {
      return {};
    }
    if (q_offset < 0) { q_offset = s.position; }
    chunks.push_back({(std::size_t)s.page_id.v * _ctx->page_stride_bytes(),
                      s.slot_offset, written, cnt});
    written += cnt;
  }
  if (q_offset < 0) { q_offset = 0; }
  const int page_tokens = _ctx->page_tokens();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  // Prefill attention via MLX's vendored steel flash kernel (optimal MMA).
  // Steel reads CONTIGUOUS K/V. Fresh prefill (q_offset==0): the n new
  // tokens in qt/kt/vt ARE the whole context. Extend/branch (q_offset>0):
  // per layer, de-page the full context (shared-prefix + new) into the
  // kfull/vfull scratch and let steel attend it with qL_off=q_offset. The
  // AttnParams are identical across layers (same shapes/scratch) -> fill
  // once. kL == t_kv and qL_off == q_offset cover both cases uniformly.
  const int t_kv = q_offset + n;            // full context length
  SharedBuffer kfull, vfull;
  if (q_offset > 0) {
    kfull = buf((std::size_t)Hkv * t_kv * D);
    vfull = buf((std::size_t)Hkv * t_kv * D);
  }
  metal_compute::ComputeFunction fn_steel;
  {
    auto* p = static_cast<SteelAttnParams*>(_attn_params.contents());
    p->B = 1; p->H = Hq; p->D = D; p->qL = n; p->kL = t_kv;
    p->gqa_factor = Hq / Hkv; p->scale = scale;
    p->NQ = (n + 31) / 32; p->NK = (t_kv + 15) / 16;
    p->NQ_aligned = n / 32; p->NK_aligned = t_kv / 16;
    p->qL_rem = n - p->NQ_aligned * 32; p->kL_rem = t_kv - p->NK_aligned * 16;
    p->qL_off = q_offset;
    p->Q_strides[0] = (std::int64_t)Hq * n * D;
    p->Q_strides[1] = (std::int64_t)n * D; p->Q_strides[2] = D;
    p->K_strides[0] = (std::int64_t)Hkv * t_kv * D;
    p->K_strides[1] = (std::int64_t)t_kv * D; p->K_strides[2] = D;
    p->V_strides[0] = p->K_strides[0];
    p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = D;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = D;
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (n % 32) == 0).set_bool(201, (t_kv % 16) == 0)
        .set_bool(300, false).set_bool(301, true).set_bool(302, false);
    fn_steel = _lib_attn.function("attn_steel_h_bd128", fc);
  }
  const bool use_steel = fn_steel.valid();

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();

    auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                   const SharedBuffer& b, const SharedBuffer& xin,
                   const SharedBuffer& y, int K, int N) {
      enc.set_function(_fn_qmm);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, K);
      enc.set_constant(6, N);
      enc.set_constant(7, n);
      const unsigned gx = (unsigned)(((N + 31) / 32) * 32);
      const unsigned gy = (unsigned)(((n + 31) / 32) * 2);
      enc.dispatch({gx, gy, 2}, {32, 2, 2});
    };
    // Single-row GEMM (M=1) reading row 0 of xin, writing row 0 of y. Used
    // for the last layer's position-wise FFN, which only the last token
    // needs (see the last-layer branch below).
    auto qmm1 = [&](const SharedBuffer& w, const SharedBuffer& s,
                    const SharedBuffer& b, const SharedBuffer& xin,
                    const SharedBuffer& y, int K, int N) {
      enc.set_function(_fn_qmm);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, K);
      enc.set_constant(6, N);
      enc.set_constant(7, 1);
      const unsigned gx = (unsigned)(((N + 31) / 32) * 32);
      enc.dispatch({gx, 2, 2}, {32, 2, 2});
    };
    // Fused gate/up GEMM + SwiGLU: one GEMM over the interleaved
    // [2*ffn, H] weight writes silu(gate)*up directly to y [M, ffn].
    // Nf is the fused width (2*ffn); M is the row count (n, or 1 on the
    // pruned last layer).
    auto qmm_swiglu = [&](const SharedBuffer& w, const SharedBuffer& s,
                          const SharedBuffer& b, const SharedBuffer& xin,
                          const SharedBuffer& y, int K, int Nf, int M) {
      enc.set_function(_fn_qmm_swiglu);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, K);
      enc.set_constant(6, Nf);
      enc.set_constant(7, M);
      const unsigned gx = (unsigned)(((Nf + 31) / 32) * 32);
      const unsigned gy = (unsigned)(((M + 31) / 32) * 2);
      enc.dispatch({gx, gy, 2}, {32, 2, 2});
    };
    auto rms = [&](const SharedBuffer& xin, const SharedBuffer& w,
                   const SharedBuffer& y) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y);
      enc.set_constant(3, H);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xin, int heads) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads);
      enc.set_constant(3, n);
      enc.set_constant(4, D);
      enc.set_constant(5, q_offset);   // global pos of first prefill token
      enc.dispatch({(unsigned)(D / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(D / 2), 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int B) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, B);
      enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)B, (unsigned)A},
                   {(unsigned)D, 1, 1});
    };
    // Write the transposed [Hkv, n, D] K/V into the context's pages,
    // one dispatch per page-chunk (a src token range -> page slots).
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      for (const Chunk& ch : chunks) {
        enc.set_function(_fn_kv_write_paged);
        enc.set_buffer(0, src);
        enc.set_buffer(1, pool, ch.page_off);
        enc.set_constant(2, page_tokens);
        enc.set_constant(3, D);
        enc.set_constant(4, n);              // n_src (source row stride)
        enc.set_constant(5, ch.src_off);
        enc.set_constant(6, ch.slot);
        enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, bb);
      enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    // De-page the whole context K/V into a contiguous [Hkv, t_kv, D] buffer
    // (extend prefill: steel reads contiguous). Loops every page in the
    // table (shared-prefix + new), so it is correct across branch-shared
    // and partial pages.
    const std::int32_t* ptab =
        static_cast<const std::int32_t*>(_pgtab.contents());
    auto gather = [&](const SharedBuffer& pool, const SharedBuffer& dst) {
      for (int pg = 0; pg < n_pages; ++pg) {
        const int pid = ptab[pg * 3 + 0];
        const int nv = ptab[pg * 3 + 1];
        const int gst = ptab[pg * 3 + 2];
        enc.set_function(_fn_kv_gather_paged);
        enc.set_buffer(0, dst);
        enc.set_buffer(1, pool, (std::size_t)pid * _ctx->page_stride_bytes());
        enc.set_constant(2, page_tokens);
        enc.set_constant(3, D);
        enc.set_constant(4, t_kv);
        enc.set_constant(5, gst);
        enc.dispatch({(unsigned)D, (unsigned)nv, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };

    // x already holds the [n, hidden] embedding (from the muxer).
    for (int L = 0; L < c.n_layers; ++L) {
      Layer& ly = _layers[L];
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);
      rms(x, ly.in_ln, hn);
      qmm(ly.qw, ly.qs, ly.qb, hn, q, H, qd);
      qmm(ly.kw, ly.ks, ly.kb, hn, k, H, kd);
      qmm(ly.vw, ly.vs, ly.vb, hn, v, H, kd);
      transpose(q, qt, n, Hq);     // [n,Hq,D] -> [Hq,n,D]
      transpose(k, kt, n, Hkv);
      transpose(v, vt, n, Hkv);
      rope(qt, Hq);
      rope(kt, Hkv);
      kv_write(kt, kp);
      kv_write(vt, vp);

      // Attention. Steel flash kernel on contiguous K/V: fresh prefill
      // reads kt/vt directly; extend (q_offset>0) first de-pages the full
      // context (prefix + new) into kfull/vfull. Fallback to the paged MMA
      // kernel if steel is unavailable.
      if (use_steel) {
        const SharedBuffer* ksb = &kt;
        const SharedBuffer* vsb = &vt;
        if (q_offset > 0) {
          gather(kp, kfull);
          gather(vp, vfull);
          ksb = &kfull;
          vsb = &vfull;
        }
        enc.set_function(fn_steel);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, *ksb);
        enc.set_buffer(2, *vsb);
        enc.set_buffer(3, at);
        enc.set_buffer(4, _attn_params);
        const unsigned nqb = (unsigned)((n + 31) / 32);
        enc.dispatch({32 * nqb, 4 * (unsigned)Hq, 1}, {32, 4, 1});
      } else {
        enc.set_function(_fn_sdpa_paged_mma);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kp);
        enc.set_buffer(2, vp);
        enc.set_buffer(3, at);
        enc.set_constant(4, scale);
        enc.set_constant(5, D);
        enc.set_constant(6, Hq);
        enc.set_constant(7, Hkv);
        enc.set_constant(8, n);          // n_q = n
        enc.set_constant(9, q_offset);   // abs pos of first prefill token
        enc.set_constant(10, page_tokens);
        enc.set_constant(11, n_pages);
        enc.set_buffer(12, _pgtab);
        enc.dispatch({256, (unsigned)Hq, (unsigned)((n + 31) / 32)},
                     {256, 1, 1});
      }

      transpose(at, att, Hq, n);       // [Hq,n,D] -> [n,Hq,D]
      qmm(ly.ow, ly.os, ly.ob, att, ao, H, H);
      residual(x, ao, x, n * H);

      // FFN. The MLP (gate/up/swiglu/down + residual) and the final norm
      // are position-wise, and prefill only consumes the last token's
      // logits -- so on the LAST layer we run the FFN for that one row
      // only. The last row's math (and thus its logits) is bit-identical;
      // rows 0..n-2 simply never get the pruned last-layer FFN, and no one
      // reads them. Attention/K-V/o_proj above still run for all rows
      // (attention mixes tokens), so this touches only the MLP.
      if (L + 1 < c.n_layers) {
        rms(x, ly.post_ln, hn);
        qmm_swiglu(ly.guw, ly.gus, ly.gub, hn, sg, H, 2 * ffn, n);
        qmm(ly.dw, ly.ds, ly.db, sg, ao, ffn, H);
        residual(x, ao, x, n * H);
      } else {
        const std::size_t roff = (std::size_t)(n - 1) * H * 2;  // bytes
        enc.set_function(_fn_rms);            // norm of the last row -> hn[0]
        enc.set_buffer(0, x, roff);
        enc.set_buffer(1, ly.post_ln);
        enc.set_buffer(2, hn);
        enc.set_constant(3, H);
        enc.set_constant(4, eps);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
        qmm_swiglu(ly.guw, ly.gus, ly.gub, hn, sg, H, 2 * ffn, 1);
        qmm1(ly.dw, ly.ds, ly.db, sg, ao, ffn, H);
        enc.set_function(_fn_residual);       // x[n-1] += ao[0]
        enc.set_buffer(0, x, roff);
        enc.set_buffer(1, ao);
        enc.set_buffer(2, x, roff);
        enc.set_constant(3, H);
        enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
      }
    }

    // Final norm on the last row only (rows 0..n-2 never got the pruned
    // last-layer FFN; only this row's logits are consumed) -> hn[0].
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x, (std::size_t)(n - 1) * H * 2);
    enc.set_buffer(1, _final_ln);
    enc.set_buffer(2, hn);
    enc.set_constant(3, H);
    enc.set_constant(4, eps);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
    // Last position: hn row 0 -> lm_head via GEMV.
    enc.set_function(_fn_qmv);
    enc.set_buffer(0, _lm_w);
    enc.set_buffer(1, _lm_s);
    enc.set_buffer(2, _lm_b);
    enc.set_buffer(3, hn);
    enc.set_buffer(4, logits);
    enc.set_constant(5, H);
    enc.set_constant(6, c.vocab);
    enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
  }
  stream.commit().wait();

  std::vector<float> out((std::size_t)c.vocab);
  const auto* lp = static_cast<const _Float16*>(logits.contents());
  for (int i = 0; i < c.vocab; ++i) {
    out[i] = (float)lp[i];
  }
  return out;
}

std::int32_t
MetalLlamaModel::forward_argmax(std::int32_t token_id)
{
  std::vector<float> l = forward(token_id);
  std::int32_t best = 0;
  float bv = l.empty() ? 0.0f : l[0];
  for (std::size_t i = 1; i < l.size(); ++i) {
    if (l[i] > bv) { bv = l[i]; best = (std::int32_t)i; }
  }
  return best;
}

bool
MetalLlamaModel::decode_pipelined(
    ContextId cid, std::int32_t first_token, int n_steps,
    std::vector<std::int32_t>& out_ids, float temperature, float top_p,
    std::uint64_t seed)
{
  out_ids.clear();
  if (n_steps <= 0) { return true; }
  const bool greedy = (temperature <= 0.0f);
  const Config& c = _cfg;   // per-step dims live in encode_decode_step_

  // GPU-resident token chain: gen_ids[0] = seed; each step's argmax writes
  // gen_ids[s+1], which the next step's embed gather reads as its input id.
  // The sampled token never round-trips through the host.
  metal_compute::SharedBuffer gen_ids = _mc->make_shared_buffer(
      (std::size_t)(n_steps + 1) * sizeof(std::int32_t));
  if (gen_ids.empty()) { return false; }
  static_cast<std::int32_t*>(gen_ids.contents())[0] = first_token;

  // One page-table slice per step. The host fills these AHEAD of GPU
  // execution (the encode loop runs ahead), so they must not alias -- a
  // single reused _pgtab would be clobbered before the GPU read it.
  const int pt_stride = _ctx->max_pages() * 3;   // int32 entries / step
  metal_compute::SharedBuffer pgt = _mc->make_shared_buffer(
      (std::size_t)n_steps * pt_stride * sizeof(std::int32_t));
  if (pgt.empty()) { return false; }

  // Top-p sampling scratch: [vocab] cached softmax weights, reused every
  // step (the GPU event chain serializes the steps). Only the sampler
  // touches it; greedy skips the allocation.
  metal_compute::SharedBuffer sample_ws, seen_batch;
  int sample_iters = 16;
  if (!greedy) {
    sample_ws = _mc->make_shared_buffer((std::size_t)c.vocab * 2);
    seen_batch = _mc->make_shared_buffer((std::size_t)c.vocab);
    if (sample_ws.empty() || seen_batch.empty()) { return false; }
    std::memset(seen_batch.contents(), 0, (std::size_t)c.vocab);
    if (const char* e = std::getenv("VPIPE_SAMPLE_ITERS")) {
      sample_iters = std::atoi(e);
      if (sample_iters < 1) { sample_iters = 1; }
    }
  }
  GpuSamplerParams bsp;          // batch path: plain temperature + top-p
  bsp.greedy = greedy;
  bsp.temperature = temperature;
  bsp.top_p = top_p;
  bsp.seed = seed;
  bsp.n_iter = sample_iters;

  metal_compute::CommandStream stream = _mc->make_command_stream();

  // CPU/GPU overlap: one command buffer per step, committed WITHOUT
  // waiting, so the host encodes step s+1 while the GPU runs step s. A
  // monotonic event chains the buffers on the GPU -- step s waits for
  // value s (the prior step's signal) before reading gen_ids[s] / the
  // shared layer scratch, and signals s+1 when done. That GPU-side
  // ordering is what makes the reused scratch (and the argmax->embed
  // token feedback) safe across separate command buffers, which a bare
  // same-queue commit does NOT guarantee. The host never blocks until the
  // final wait. (Page-table slices stay per-step: the host fills them all
  // ahead of execution, so they must not alias.)
  metal_compute::Event ev = _mc->make_event();
  metal_compute::CommandStream::Fence last;

  for (int s = 0; s < n_steps; ++s) {
    ContextManager::AppendSlot slot = _ctx->append(cid, 1);
    if (!slot.valid()) { return false; }
    const int pos = slot.position;
    const std::size_t page_off =
        (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
    std::int32_t* pt_ptr = static_cast<std::int32_t*>(pgt.contents()) +
                           (std::size_t)s * pt_stride;
    const int n_pages = _ctx->fill_page_table(cid, pt_ptr);
    const std::size_t pt_off =
        (std::size_t)s * pt_stride * sizeof(std::int32_t);

    stream.encode_wait(ev, (std::uint64_t)s);
    {
      ComputeEncoder enc = stream.begin_compute();
      encode_decode_step_(enc, cid, pos, page_off, n_pages, slot, gen_ids,
                          (std::size_t)s * sizeof(std::int32_t), pgt, pt_off);

      // Next token -> gen_ids[s+1], entirely on-GPU (no host pull) so the
      // embed gather feeds it straight into the next step. Greedy argmax,
      // or temperature+top-p sampling with a per-step seed.
      const std::uint32_t step_seed =
          (std::uint32_t)(seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
      encode_sample_(enc, _logits, gen_ids,
                     (std::size_t)(s + 1) * sizeof(std::int32_t), bsp,
                     step_seed, sample_ws, seen_batch);
    }
    stream.encode_signal(ev, (std::uint64_t)(s + 1));
    last = stream.commit();
  }
  last.wait();

  out_ids.resize(n_steps);
  const auto* g = static_cast<const std::int32_t*>(gen_ids.contents());
  for (int s = 0; s < n_steps; ++s) { out_ids[s] = g[s + 1]; }
  return true;
}

void
MetalLlamaModel::encode_argmax_(metal_compute::ComputeEncoder& enc,
                                const metal_compute::SharedBuffer& logits,
                                const metal_compute::SharedBuffer& out,
                                std::size_t out_off, int vocab)
{
  static const bool kForce1 = std::getenv("VPIPE_LLAMA_ARGMAX1") != nullptr;
  if (!kForce1 && _fn_argmax_partial.valid() && _fn_argmax_combine.valid()
      && !_d_argmax_part.empty()) {
    enc.set_function(_fn_argmax_partial);
    enc.set_buffer(0, logits);
    enc.set_buffer(1, _d_argmax_part);
    enc.set_constant(2, vocab);
    enc.set_constant(3, kArgmaxM);
    enc.dispatch({(unsigned)(256 * kArgmaxM), 1, 1}, {256, 1, 1});
    enc.set_function(_fn_argmax_combine);
    enc.set_buffer(0, _d_argmax_part);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, kArgmaxM);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  } else {
    enc.set_function(_fn_argmax);
    enc.set_buffer(0, logits);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, vocab);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  }
}

void
MetalLlamaModel::encode_sample_core_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& logits,
    const metal_compute::SharedBuffer& out, std::size_t out_off,
    const metal_compute::SharedBuffer& sample_ws,
    const metal_compute::SharedBuffer& seen, const GpuSamplerParams& sp,
    std::uint32_t step_seed, int vocab)
{
  static const bool kForce1 = std::getenv("VPIPE_LLAMA_SAMPLE1") != nullptr;
  const bool hist_ok =
      _fn_smp_max_partial.valid() && _fn_smp_max_combine.valid() &&
      _fn_smp_zhist_partial.valid() && _fn_smp_zhist_combine.valid() &&
      _fn_smp_thresh.valid() && _fn_smp_pick_partial.valid() &&
      _fn_smp_pick_combine.valid() && !_d_smp_hpart.empty();
  if (kForce1 || !hist_ok) {
    enc.set_function(_fn_sample);
    enc.set_buffer(0, logits);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, vocab);
    enc.set_constant(3, sp.temperature);
    enc.set_constant(4, sp.top_p);
    enc.set_constant(5, step_seed);
    enc.set_buffer(6, sample_ws);
    enc.set_constant(7, sp.n_iter);
    enc.set_constant(8, sp.repetition_penalty);
    enc.set_constant(9, sp.presence_penalty);
    enc.set_constant(10, sp.top_k);
    enc.set_constant(11, sp.min_p);
    enc.set_buffer(12, seen);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
    return;
  }
  const int M = kSampM;
  const float rep = sp.repetition_penalty, pres = sp.presence_penalty;
  // Pass A: max of penalised eff (partial -> combine).
  enc.set_function(_fn_smp_max_partial);
  enc.set_buffer(0, logits);
  enc.set_buffer(1, _d_smp_maxpart);
  enc.set_constant(2, vocab);
  enc.set_constant(3, M);
  enc.set_constant(4, rep);
  enc.set_constant(5, pres);
  enc.set_buffer(6, seen);
  enc.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_smp_max_combine);
  enc.set_buffer(0, _d_smp_maxpart);
  enc.set_buffer(1, _d_smp_maxl);
  enc.set_constant(2, M);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
  // Pass B: ws + partial Z + per-tg histogram, then combine.
  enc.set_function(_fn_smp_zhist_partial);
  enc.set_buffer(0, logits);
  enc.set_buffer(1, sample_ws);
  enc.set_buffer(2, _d_smp_hpart);
  enc.set_constant(3, vocab);
  enc.set_constant(4, M);
  enc.set_constant(5, sp.temperature);
  enc.set_constant(6, rep);
  enc.set_constant(7, pres);
  enc.set_buffer(8, seen);
  enc.set_buffer(9, _d_smp_maxl);
  enc.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_smp_zhist_combine);
  enc.set_buffer(0, _d_smp_hpart);
  enc.set_buffer(1, _d_smp_hist);
  enc.set_constant(2, M);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
  // Threshold scan over the bins (single tg, cheap).
  enc.set_function(_fn_smp_thresh);
  enc.set_buffer(0, _d_smp_hist);
  enc.set_buffer(1, _d_smp_wt);
  enc.set_constant(2, sp.top_p);
  enc.set_constant(3, sp.top_k);
  enc.set_constant(4, sp.min_p);
  enc.set_constant(5, vocab);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
  // Pass C: Gumbel-max pick over the nucleus (partial -> combine, sets seen).
  enc.set_function(_fn_smp_pick_partial);
  enc.set_buffer(0, logits);
  enc.set_buffer(1, sample_ws);
  enc.set_buffer(2, _d_smp_pickpart);
  enc.set_constant(3, vocab);
  enc.set_constant(4, M);
  enc.set_constant(5, sp.temperature);
  enc.set_constant(6, step_seed);
  enc.set_constant(7, rep);
  enc.set_constant(8, pres);
  enc.set_buffer(9, seen);
  enc.set_buffer(10, _d_smp_wt);
  enc.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_smp_pick_combine);
  enc.set_buffer(0, _d_smp_pickpart);
  enc.set_buffer(1, out, out_off);
  enc.set_constant(2, M);
  enc.set_buffer(3, seen);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
}

void
MetalLlamaModel::encode_sample_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& logits,
    const metal_compute::SharedBuffer& out_id, std::size_t out_off,
    const GpuSamplerParams& sp, std::uint32_t step_seed,
    const metal_compute::SharedBuffer& sample_ws,
    const metal_compute::SharedBuffer& seen)
{
  if (sp.greedy) {
    encode_argmax_(enc, logits, out_id, out_off, _cfg.vocab);
  } else {
    encode_sample_core_(enc, logits, out_id, out_off, sample_ws, seen,
                        sp, step_seed, _cfg.vocab);
  }
}

void
MetalLlamaModel::encode_gqa_attn_(
    ComputeEncoder& enc, const SharedBuffer& q, const SharedBuffer& kp,
    const SharedBuffer& vp, const SharedBuffer& out, const SharedBuffer& pgtab,
    std::size_t pgtab_off, float scale, int D, int Hq, int Hkv, int pos,
    int page_tokens, int n_pages)
{
  const int sp = _gqa_split;
  // Phase A: each (kv, split) single-simdgroup worker reads its KV-head
  // position-slice ONCE and applies it to all G=Hq/Hkv query heads.
  enc.set_function(_fn_sdpa_gqa);
  enc.set_buffer(0, q);
  enc.set_buffer(1, kp);
  enc.set_buffer(2, vp);
  enc.set_buffer(3, _d_gqa_oacc);
  enc.set_buffer(4, _d_gqa_m);
  enc.set_buffer(5, _d_gqa_l);
  enc.set_constant(6, scale);
  enc.set_constant(7, D);
  enc.set_constant(8, Hq);
  enc.set_constant(9, Hkv);
  enc.set_constant(10, pos);            // q_offset (n_q==1 decode)
  enc.set_constant(11, page_tokens);
  enc.set_constant(12, n_pages);
  enc.set_buffer(13, pgtab, pgtab_off);
  enc.set_constant(14, sp);
  enc.dispatch({32, (unsigned)Hkv, (unsigned)sp}, {32, 1, 1});
  // Phase B: merge the `sp` partials per query head -> normalized out[Hq*D].
  enc.set_function(_fn_sdpa_gqa_merge);
  enc.set_buffer(0, _d_gqa_oacc);
  enc.set_buffer(1, _d_gqa_m);
  enc.set_buffer(2, _d_gqa_l);
  enc.set_buffer(3, out);
  enc.set_constant(4, D);
  enc.set_constant(5, sp);
  enc.set_constant(6, Hq);
  enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
}

void
MetalLlamaModel::encode_decode_step_(
    metal_compute::ComputeEncoder& enc, ContextId cid, int pos,
    std::size_t page_off, int n_pages,
    const ContextManager::AppendSlot& slot,
    const metal_compute::SharedBuffer& in_ids, std::size_t in_off,
    const metal_compute::SharedBuffer& pgt, std::size_t pt_off)
{
  (void)cid;
  const Config& c = _cfg;
  const int H = c.hidden;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads, D = c.head_dim;
  const int qd = Hq * D, kd = Hkv * D;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const int page_tokens = _ctx->page_tokens();

  auto qmv = [&](const SharedBuffer& w, const SharedBuffer& sc,
                 const SharedBuffer& b, const SharedBuffer& x,
                 const SharedBuffer& y, int K, int N) {
    enc.set_function(_fn_qmv);
    enc.set_buffer(0, w);
    enc.set_buffer(1, sc);
    enc.set_buffer(2, b);
    enc.set_buffer(3, x);
    enc.set_buffer(4, y);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto rms = [&](const SharedBuffer& x, const SharedBuffer& w,
                 const SharedBuffer& y) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y);
    enc.set_constant(3, H);
    enc.set_constant(4, eps);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  };
  auto rope = [&](const SharedBuffer& x, int heads) {
    enc.set_function(_fn_rope);
    enc.set_buffer(0, x);
    enc.set_buffer(1, _inv_freq);
    enc.set_constant(2, heads);
    const int one = 1;
    enc.set_constant(3, one);
    enc.set_constant(4, D);
    enc.set_constant(5, pos);
    enc.dispatch({(unsigned)(D / 2), 1, (unsigned)heads},
                 {(unsigned)(D / 2), 1, 1});
  };
  auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
    enc.set_function(_fn_kv_write_paged);
    enc.set_buffer(0, src);
    enc.set_buffer(1, pool, page_off);
    enc.set_constant(2, page_tokens);
    enc.set_constant(3, D);
    const int one = 1;
    enc.set_constant(4, one);
    const int zero = 0;
    enc.set_constant(5, zero);
    enc.set_constant(6, slot.slot_offset);
    enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
  };
  auto qmv_add = [&](const SharedBuffer& w, const SharedBuffer& sc,
                     const SharedBuffer& b, const SharedBuffer& x,
                     const SharedBuffer& y, const SharedBuffer& res,
                     int K, int N) {
    enc.set_function(_fn_qmv_add);
    enc.set_buffer(0, w);
    enc.set_buffer(1, sc);
    enc.set_buffer(2, b);
    enc.set_buffer(3, x);
    enc.set_buffer(4, y);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_buffer(7, res);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };

  // Zero-copy embedding fetch: gather the (dequantized) table row
  // in_ids[in_off] straight into the residual stream _x.
  enc.set_function(_fn_embed);
  enc.set_buffer(0, in_ids, in_off);
  enc.set_buffer(1, _embed_w);
  enc.set_buffer(2, _embed_s);
  enc.set_buffer(3, _embed_b);
  enc.set_buffer(4, _x);
  enc.set_constant(5, H);
  enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});

  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    const SharedBuffer& kp = *_ctx->kpool(L);
    const SharedBuffer& vp = *_ctx->vpool(L);

    rms(_x, ly.in_ln, _hn);
    qmv(ly.qw, ly.qs, ly.qb, _hn, _q, H, qd);
    qmv(ly.kw, ly.ks, ly.kb, _hn, _k, H, kd);
    qmv(ly.vw, ly.vs, ly.vb, _hn, _v, H, kd);
    rope(_q, Hq);
    rope(_k, Hkv);
    kv_write(_k, kp);
    kv_write(_v, vp);

    const bool use_mb = (pos + 1) >= _sdpa_mb_min;
    if (_gqa_attn && use_mb && !_d_gqa_oacc.empty()) {
      // Flash-decode-GQA: read each KV head once for all G query heads.
      encode_gqa_attn_(enc, _q, kp, vp, _attn, pgt, pt_off, scale, D, Hq,
                       Hkv, pos, page_tokens, n_pages);
    } else {
    enc.set_function(use_mb ? _fn_sdpa_paged_mb : _fn_sdpa_paged);
    enc.set_buffer(0, _q);
    enc.set_buffer(1, kp);
    enc.set_buffer(2, vp);
    enc.set_buffer(3, _attn);
    enc.set_constant(4, scale);
    enc.set_constant(5, D);
    enc.set_constant(6, Hq);
    enc.set_constant(7, Hkv);
    const int one = 1;
    enc.set_constant(8, one);
    enc.set_constant(9, pos);
    enc.set_constant(10, page_tokens);
    enc.set_constant(11, n_pages);
    enc.set_buffer(12, pgt, pt_off);
    const unsigned w = use_mb ? 32u * 32u : 32u;
    enc.dispatch({w, (unsigned)Hq, 1}, {w, 1, 1});
    }

    qmv_add(ly.ow, ly.os, ly.ob, _attn, _x, _x, H, H);

    rms(_x, ly.post_ln, _hn);
    enc.set_function(_fn_qmv_swiglu);
    enc.set_buffer(0, ly.guw);
    enc.set_buffer(1, ly.gus);
    enc.set_buffer(2, ly.gub);
    enc.set_buffer(3, _hn);
    enc.set_buffer(4, _sg);
    enc.set_constant(5, H);
    enc.set_constant(6, 2 * c.ffn_inner);
    enc.dispatch({32, (unsigned)(c.ffn_inner / 2), 1}, {32, 2, 1});
    qmv_add(ly.dw, ly.ds, ly.db, _sg, _x, _x, c.ffn_inner, H);
  }

  rms(_x, _final_ln, _hn);
  qmv(_lm_w, _lm_s, _lm_b, _hn, _logits, H, c.vocab);
}

bool
MetalLlamaModel::pdecode_begin(ContextId cid, std::int32_t first_token,
                               std::span<const std::int32_t> prompt,
                               const GpuSamplerParams& sp, int max_tokens,
                               int rope_first)
{
  (void)rope_first;   // Llama is text-only -> sequential rope (== KV slot)
  if (!_fn_embed.valid() || !_fn_argmax.valid() ||
      (!sp.greedy && !_fn_sample.valid())) {
    return false;
  }
  if (max_tokens < 1) { max_tokens = 1; }

  PDecode& pd = _pdec[cid.v];
  pd = PDecode{};
  pd.cid = cid;
  pd.sp = sp;
  pd.produced = 1;
  pd.cap = max_tokens + 1;

  pd.gen_ids = _mc->make_shared_buffer(
      (std::size_t)pd.cap * sizeof(std::int32_t));
  const int pt_stride = _ctx->max_pages() * 3;
  pd.pgt = _mc->make_shared_buffer(
      (std::size_t)pt_stride * sizeof(std::int32_t));
  if (pd.gen_ids.empty() || pd.pgt.empty()) { _pdec.erase(cid.v); return false; }
  static_cast<std::int32_t*>(pd.gen_ids.contents())[0] = first_token;

  if (!sp.greedy) {
    pd.sample_ws = _mc->make_shared_buffer((std::size_t)_cfg.vocab * 2);
    pd.seen = _mc->make_shared_buffer((std::size_t)_cfg.vocab);
    if (pd.sample_ws.empty() || pd.seen.empty()) {
      _pdec.erase(cid.v); return false;
    }
    std::memset(pd.seen.contents(), 0, (std::size_t)_cfg.vocab);
    auto* sb = static_cast<std::uint8_t*>(pd.seen.contents());
    for (std::int32_t t : prompt) {
      if (t >= 0 && t < _cfg.vocab) { sb[t] = 1; }
    }
    if (first_token >= 0 && first_token < _cfg.vocab) { sb[first_token] = 1; }
  }

  pd.stream = _mc->make_command_stream();
  pd.ev = _mc->make_event();
  return true;
}

bool
MetalLlamaModel::pdecode_commit(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return false; }
  PDecode& pd = it->second;
  if (pd.have_inflight) { return false; }
  const int in_idx = pd.produced - 1;
  const int out_idx = pd.produced;
  if (out_idx >= pd.cap) { return false; }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return false; }
  const int pos = slot.position;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(pd.pgt.contents()));

  const std::uint64_t s = pd.gpu_step;
  pd.stream.encode_wait(pd.ev, s);
  {
    ComputeEncoder enc = pd.stream.begin_compute();
    encode_decode_step_(enc, cid, pos, page_off, n_pages, slot, pd.gen_ids,
                        (std::size_t)in_idx * sizeof(std::int32_t), pd.pgt, 0);
    const std::uint32_t step_seed = (std::uint32_t)(
        pd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
    encode_sample_(enc, _logits, pd.gen_ids,
                   (std::size_t)out_idx * sizeof(std::int32_t), pd.sp,
                   step_seed, pd.sample_ws, pd.seen);
  }
  pd.stream.encode_signal(pd.ev, s + 1);
  pd.inflight = pd.stream.commit();
  pd.gpu_step = s + 1;
  pd.pending = out_idx;
  pd.have_inflight = true;
  return true;
}

std::int32_t
MetalLlamaModel::pdecode_next(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return -1; }
  PDecode& pd = it->second;
  if (!pd.have_inflight) { return -1; }
  pd.inflight.wait();
  const std::int32_t tok =
      static_cast<const std::int32_t*>(pd.gen_ids.contents())[pd.pending];
  pd.produced = pd.pending + 1;
  pd.have_inflight = false;
  pd.pending = -1;
  return tok;
}

void
MetalLlamaModel::pdecode_end(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return; }
  if (it->second.have_inflight) { it->second.inflight.wait(); }
  _pdec.erase(it);
}

}  // namespace vpipe::genai
