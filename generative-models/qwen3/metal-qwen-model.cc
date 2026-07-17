#include "generative-models/qwen3/metal-qwen-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/shared/i8-gemm.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cassert>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
// Q4_K -> affine-g32 decode/prefill repack: ON by default (lossless, faster
// decode + prefill, and net memory-NEGATIVE since the raw kq is freed after
// repack). VPIPE_QWEN_Q4K_AFFINE=0 opts out (A/B + safety hatch).
bool
q4k_affine_enabled_()
{
  const char* e = std::getenv("VPIPE_QWEN_Q4K_AFFINE");
  return e == nullptr || std::atoi(e) != 0;
}
std::size_t
numel_(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}
// Length of the page-id prefix the N branches share (identical leading page
// ids -- they were all branched off one base, so the prefix pages are
// refcount-shared and byte-identical; the first private page differs per
// branch). 0 for N<2. `pgt` is the [N, max_pages, 3] int32 page-table block
// (stride pt_stride = max_pages*3); n_pages_v[i] is branch i's page count.
int
shared_prefix_pages_(const std::int32_t* pgt,
                     const std::vector<int>& n_pages_v, int N,
                     std::size_t pt_stride)
{
  if (N < 2 || pgt == nullptr) { return 0; }
  int minp = n_pages_v[0];
  for (int i = 1; i < N; ++i) {
    if (n_pages_v[(std::size_t)i] < minp) { minp = n_pages_v[(std::size_t)i]; }
  }
  int sp = 0;
  while (sp < minp) {
    const std::int32_t pid0 = pgt[(std::size_t)sp * 3 + 0];
    bool same = true;
    for (int i = 1; i < N && same; ++i) {
      if (pgt[(std::size_t)i * pt_stride + (std::size_t)sp * 3 + 0] != pid0) {
        same = false;
      }
    }
    if (!same) { break; }
    ++sp;
  }
  return sp;
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  // round-to-nearest-even
  b += 0x7fffu + ((b >> 16) & 1u);
  return (std::uint16_t)(b >> 16);
}
// Read a 16-bit compute-element buffer (f16 or bf16) into f32.
inline void
read_elt_(const void* src, float* dst, std::size_t n, bool bf16)
{
  if (bf16) {
    const auto* s = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < n; ++i) { dst[i] = bf16_to_f32_(s[i]); }
  } else {
    const auto* s = static_cast<const _Float16*>(src);
    for (std::size_t i = 0; i < n; ++i) { dst[i] = (float)s[i]; }
  }
}
}  // namespace

std::unique_ptr<MetalQwenModel>
MetalQwenModel::load(const std::string& model_dir,
                     metal_compute::MetalCompute* mc, const Config& cfg_in)
{
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) {
    return nullptr;
  }

  // Working copy: auto-correct the LM weight prefix for naming variants.
  // config_from() guesses "language_model." + "model." (the 4B/9B layout); a
  // newer checkpoint (the 27B) uses the REVERSED "model.language_model.". If
  // the configured prefix doesn't locate layer 0, probe the tensor names
  // (every layer, full-attn or GDN, has input_layernorm).
  Config cfg = cfg_in;
  // Streaming calibration implies a backbone-only load (no lm_head / muxer):
  // the per-layer forward never reaches the output head.
  if (cfg.calib_stream) { cfg.backbone_only = true; }
  {
    // Anchor on the LAST layer's input_layernorm so we lock onto the main LM
    // stack and not a smaller co-resident stack that reuses "layers.N." names
    // (the 27B's 1-layer MTP draft head lives at "mtp.layers.0."). Falls back
    // to layer 0 only if n_layers is unknown.
    const int aL = cfg.n_layers > 0 ? cfg.n_layers - 1 : 0;
    const std::string anchor =
        "layers." + std::to_string(aL) + ".input_layernorm.weight";
    if (!wts->has(cfg.weight_prefix + cfg.model_seg + anchor)) {
      for (const std::string& n : wts->tensor_names()) {
        if (n.size() > anchor.size() &&
            n.compare(n.size() - anchor.size(), anchor.size(), anchor) == 0) {
          cfg.weight_prefix = n.substr(0, n.size() - anchor.size());
          cfg.model_seg     = "";
          break;
        }
      }
    }
  }
  // Untied lm_head can live at "<prefix>lm_head" (4B/VLM convention) or be
  // hoisted to a top-level "lm_head" (the 27B). Resolve whichever exists.
  std::string lm_head_base;   // name minus the ".weight"/".scales"/... suffix
  for (const std::string& cand : {cfg.weight_prefix + "lm_head",
                                  std::string("lm_head"),
                                  cfg.weight_prefix + cfg.model_seg + "lm_head"}) {
    if (wts->has(cand + ".weight") || wts->has(cand + ".scales") ||
        wts->has(cand + ".q6k") || wts->has(cand + ".q4k")) {
      lm_head_base = cand;
      break;
    }
  }

  auto m = std::unique_ptr<MetalQwenModel>(new MetalQwenModel());
  m->_cfg = cfg;
  m->_mc = mc;

  // ---- Mixed-precision affine (OptiQ) detection ----------------------
  // Infer an affine linear's bit width from its packed-weight column count
  // vs its input dim K: weight is [..., N, K*bits/32], so bits = wcols*32/K
  // where wcols is the LAST (packed) dim. 2D linears ([N, K*bits/32]) and 3D
  // batched MoE experts ([E, N, K*bits/32]) both pack K last, so use the
  // trailing dim -- NOT shape[1], which for a 3D expert tensor is N (the
  // per-expert row count, bit-width-independent) and would misreport the
  // width (e.g. a w4 MoE looking like w8). A k-quant weight (raw blocks, no
  // .scales) returns 0. If a checkpoint mixes 4-bit and 8-bit linears
  // (mlx-optiq sensitivity quant), take the de-fused per-tensor path; uniform
  // 4-bit and uniform 8-bit both stay on the fused single-width path.
  auto affine_bits = [&](const std::string& pfx, int K) -> int {
    const auto* wi = wts->info(pfx + ".weight");
    if (wi == nullptr || wi->shape.size() < 2 || K <= 0) { return 0; }
    return (int)((wi->shape.back() * 32) / K);
  };
  // Affine group size from the .scales shape [N, K/group]; 0 if not affine.
  auto affine_group = [&](const std::string& pfx, int K) -> int {
    const auto* si = wts->info(pfx + ".scales");
    if (si == nullptr || si->shape.empty() || si->shape.back() <= 0
        || K <= 0) {
      return 0;
    }
    return (int)(K / si->shape.back());
  };
  {
    bool saw4 = false, saw8 = false;
    // Reject any AFFINE linear that isn't 4/8-bit group-64: every dispatch is a
    // binary `bits==8?w8:w4` and every mixed kernel is g64, so a 2/3/6-bit or
    // non-64-group weight would silently bind the wrong kernel -> garbage. Only
    // affine weights (with .scales) are checked; k-quant / dense / bf16 (no
    // .scales) take their own paths and are skipped. Fail the load loudly.
    std::string quant_err;
    auto note = [&](const std::string& nm, int K) {
      if (!wts->has(nm + ".scales")) { return; }   // non-affine: not our path
      const int b = affine_bits(nm, K);
      const int g = affine_group(nm, K);
      if (b != 4 && b != 8) {
        if (quant_err.empty()) {
          quant_err = nm + " is " + std::to_string(b)
                    + "-bit (only 4/8-bit affine supported)";
        }
        return;
      }
      if (g != 64) {
        if (quant_err.empty()) {
          quant_err = nm + " has group " + std::to_string(g)
                    + " (only group-64 affine supported)";
        }
        return;
      }
      if (b == 4) { saw4 = true; } else { saw8 = true; }
    };
    for (int L = 0; L < cfg.n_layers; ++L) {
      const std::string p =
          cfg.weight_prefix + cfg.model_seg + "layers." + std::to_string(L) + ".";
      if (cfg.layer_is_full(L)) {
        note(p + "self_attn.q_proj", cfg.hidden);
        note(p + "self_attn.k_proj", cfg.hidden);
        note(p + "self_attn.v_proj", cfg.hidden);
        note(p + "self_attn.o_proj", cfg.qd());
      } else {
        note(p + "linear_attn.in_proj_qkv", cfg.hidden);
        note(p + "linear_attn.in_proj_z", cfg.hidden);
        note(p + "linear_attn.out_proj", cfg.value_dim());
      }
      note(p + "mlp.gate_proj", cfg.hidden);
      note(p + "mlp.up_proj", cfg.hidden);
      note(p + "mlp.down_proj", cfg.ffn_inner);
    }
    m->_mixed = saw4 && saw8;
    // TEST HOOK: force the mixed/de-fused paths on a UNIFORM affine model so
    // they can be token-exact-verified against the fused path (a uniform layer
    // then takes qkv_fused / mlp_fused, the GDN in_proj de-fuses, etc. -- all
    // must reproduce the fused output). Only meaningful for affine models.
    if ((saw4 || saw8) && std::getenv("VPIPE_QWEN_FORCE_MIXED") != nullptr) {
      m->_mixed = true;
    }
    // Embed/lm_head bit width (q6k k-quant -> 0 -> fall back to the scalar).
    const std::string ebn =
        cfg.weight_prefix + cfg.model_seg + "embed_tokens";
    const int eb = affine_bits(ebn, cfg.hidden);
    // MOSS-TTS text artifact: a bf16 embed (a plain [vocab,H] .weight, no
    // .scales / k-quant blocks) atop an AFFINE backbone. The embed + its tied
    // lm_head are kept in full precision -- bind them via the dense f16 path
    // (muxer gather + dense GEMV head) while the backbone stays affine, so the
    // output logits are not double-quantized. The on-disk embed stays bf16,
    // so the TTS path's host-side gather is unaffected (one artifact serves
    // both). Detected purely from tensor names, so it never fires for a normal
    // affine LM (embed has .scales) or the raw-HF dense MOSS (no affine
    // linears in the backbone -> saw4/saw8 false).
    const bool embed_bf16 = (eb != 4 && eb != 8) &&
        wts->has(ebn + ".weight") && !wts->has(ebn + ".scales") &&
        !wts->has(ebn + ".q6k") && !wts->has(ebn + ".q4k");
    m->_dense_embed = embed_bf16 && (saw4 || saw8);
    m->_embed_bits = (eb == 4 || eb == 8) ? eb : cfg.quant_bits;
    m->_lm_bits = m->_embed_bits;
    // An AFFINE embed (has .scales; not the bf16/k-quant artifacts above) must
    // also be 4/8-bit group-64 -- the gather picks w8 iff _embed_bits==8.
    if (quant_err.empty() && wts->has(ebn + ".scales")) {
      const int eg = affine_group(ebn, cfg.hidden);
      if (eb != 4 && eb != 8) {
        quant_err = ebn + " is " + std::to_string(eb)
                  + "-bit (only 4/8-bit affine supported)";
      } else if (eg != 64) {
        quant_err = ebn + " has group " + std::to_string(eg)
                  + " (only group-64 affine supported)";
      }
    }
    if (!quant_err.empty()) {
      // Every affine dispatch is a binary bits==8?w8:w4 over g64 kernels, so an
      // unsupported width/group would silently mis-bind -> garbage. Fail loudly
      // via the session log delegate (no direct stderr) instead.
      if (const SessionContextIntf* s = mc->session()) {
        s->warn(fmt("[qwen] unsupported affine quantization: {} -- model not "
                    "loaded (only 4/8-bit, group-64)", quant_err));
      }
      return nullptr;
    }
  }

  const std::string sfx = cfg.use_bf16 ? "_bf16" : "";
  m->_lib_qmv = mc->load_library("affine_qmv" + sfx);
  m->_lib_qmm = mc->load_library("affine_qmm_steel" + sfx);
  m->_lib_rms = mc->load_library("rms_norm" + sfx);
  m->_lib_elt = mc->load_library("llm_elementwise" + sfx);
  m->_lib_rope = mc->load_library("rope" + sfx);
  m->_lib_sdpa = mc->load_library("sdpa" + sfx);
  m->_lib_gdn = mc->load_library("qwen3_5_gated_delta" + sfx);
  // MLX steel register-resident flash (attn_steel) is half-only; load it for
  // f16 models so head_dim-256 fresh prefill can hand attention to it.
  if (!cfg.use_bf16) {
    // MLX steel register-softmax flash, paged-KV variant -- the head_dim-256
    // full-attention prefill kernel (fresh + mid-context). half-only -> f16
    // models; bf16 keeps the paged flash. No function constants -> bind once.
    m->_lib_attn = mc->load_library("attn_steel");
    m->_fn_steel_paged = m->_lib_attn.function("attn_steel_paged_bd256");
  }
  // 4-bit (Qwen3.5/Llama) vs 8-bit (Qwen3-ASR) quantized-linear kernels.
  const std::string g = cfg.quant_bits == 8 ? "w8g64" : "w4g64";
  m->_fn_qmv = m->_lib_qmv.function("affine_qmv_" + g);
  m->_fn_qmv_add = m->_lib_qmv.function("affine_qmv_" + g + "_add");
  m->_fn_qmv_swiglu = m->_lib_qmv.function("affine_qmv_swiglu_" + g);
  m->_fn_qmm = m->_lib_qmm.function("affine_qmm_steel_" + g);
  m->_fn_qmm_swiglu = m->_lib_qmm.function("affine_qmm_swiglu_" + g);
  // Batched-decode GEMV (M-row qmv): weights read once across N branches,
  // at qmv bandwidth (steel GEMM is ~3x slower per weight-read at small M).
  // MAXM=2 (N>2 tiles along grid.z; N==1 stays on plain qmv). 4-bit only
  // -- 8-bit batches fall back to the steel GEMM.
  if (cfg.quant_bits != 8) {
    m->_fn_qmv_batch = m->_lib_qmv.function("affine_qmv_batch_w4g64");
    m->_fn_qmv_batch_swiglu =
        m->_lib_qmv.function("affine_qmv_batch_swiglu_w4g64");
    // MAXM=4 twins for the MTP verify at draft depth>=2 (see _qmv4_* gate).
    // The m=3..4 tier defaults to the grouped-x xp form (weight packs in
    // registers, x resident 2 rows at a time): same ALU as the register-
    // resident batch4 but ~70 regs instead of ~120, so the occupancy that
    // capped batch4 at ~67 GB/s/pass comes back -> ~84 GB/s, ~1.25x on the
    // MTP verify's >SLC lm_head (the depth-2 cliff). BIT-IDENTICAL
    // (qmv_batch_tg_matches_batch); VPIPE_QMV_XP4=0 reverts for A/B.
    m->_fn_qmv_batch4 = m->_lib_qmv.function("affine_qmv_batch4_xp_w4g64");
    if (const char* e = std::getenv("VPIPE_QMV_XP4");
        (e && std::atoi(e) == 0) || !m->_fn_qmv_batch4.valid()) {
      m->_fn_qmv_batch4 = m->_lib_qmv.function("affine_qmv_batch4_w4g64");
    }
    // Fused-SwiGLU twins follow the same tiers (bit-exact epilogue
    // mirror of the register form); the same VPIPE_QMV_XP4 knob reverts.
    m->_fn_qmv_batch4_swiglu =
        m->_lib_qmv.function("affine_qmv_batch4_xp_swiglu_w4g64");
    if (const char* e = std::getenv("VPIPE_QMV_XP4");
        (e && std::atoi(e) == 0) || !m->_fn_qmv_batch4_swiglu.valid()) {
      m->_fn_qmv_batch4_swiglu =
          m->_lib_qmv.function("affine_qmv_batch4_swiglu_w4g64");
    }
    m->_fn_qmv_batch8_xp_swiglu =
        m->_lib_qmv.function("affine_qmv_batch8_xp2_swiglu_w4g64");
    // MAXM=8 tall-tile GEMV for m=7..8: one weight read for all rows.
    // VPIPE_QMV_XP8 picks the tier: 1 (DEFAULT) = xp2, the grouped-x form
    // -- BIT-IDENTICAL to the MAXM=2 kernel, ~43 GB/s/pass, ~1.3x over
    // the MAXM=2 tiling on the FFN/lm_head-size matrices whose grid.z
    // re-reads are NOT cache-served; 2 = xh16, the half-dot hoisted
    // kernel (~52 GB/s/pass, ~1.6x) whose f16 in-quad sums are rel-L2
    // ~4e-4 off the reference -- measured NOT greedy token-exact
    // (qwen_batched_decode_token_exact N=8: 93/96, a near-tie argmax
    // flip), so it stays OPT-IN for tolerance-accepting workloads;
    // 0 = off (MAXM=2 tiling).
    int xp8_mode = 1;
    if (const char* e = std::getenv("VPIPE_QMV_XP8")) {
      xp8_mode = std::atoi(e);
      if (xp8_mode < 0 || xp8_mode > 2) { xp8_mode = 1; }
    }
    if (xp8_mode == 2) {
      m->_fn_qmv_batch8_xp =
          m->_lib_qmv.function("affine_qmv_batch8_xh16_w4g64");
    }
    if (xp8_mode == 1 || (xp8_mode == 2 && !m->_fn_qmv_batch8_xp.valid())) {
      m->_fn_qmv_batch8_xp =
          m->_lib_qmv.function("affine_qmv_batch8_xp2_w4g64");
    }
  }
  // simd_sum RMSNorm (dispatched at 256). The gemma tree rms_norm_f16 (f1ab287)
  // strides by a fixed 512 -> silently wrong at 256; rms_norm_fast_f16 is the
  // threadgroup-size-agnostic simd_sum (== pre-f1ab287, no RMS_PAD cap).
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_f16");
  // Generic f16 copy (used by the k-quant dequant fold AND the Leviathan-Chen
  // MTP logit stash). Load unconditionally so the mixed-affine (OptiQ) path has
  // it too -- it lives in the always-loaded elementwise lib.
  m->_fn_copy = m->_lib_elt.function("copy_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  // Mixture-of-Experts (Qwen3.5-MoE). Gather GEMVs + the shared-expert gate
  // live in affine_qmv.metal (_lib_qmv); router/combine/finalize in
  // llm_elementwise.metal (_lib_elt). Present in every Qwen build; only the
  // MoE forward dispatches them, so loading unconditionally is harmless.
  m->_fn_moe_route = m->_lib_elt.function("moe_route_f16");
  m->_fn_moe_combine = m->_lib_elt.function("moe_combine_f16");
  m->_fn_moe_finalize = m->_lib_elt.function("moe_finalize_f16");
  m->_fn_moe_finalize_combined =
      m->_lib_elt.function("moe_finalize_combined_f16");
  m->_fn_moe_gather_swiglu =
      m->_lib_qmv.function("affine_gather_qmv_swiglu_w4g64");
  m->_fn_moe_gather_down =
      m->_lib_qmv.function("affine_gather_down_qmv_w4g64");
  m->_fn_moe_gate = m->_lib_qmv.function("affine_moe_gate_w8g64");
  // Grouped-prefill kernels (counting sort + segmented expert GEMV).
  m->_fn_moe_grouped_swiglu =
      m->_lib_qmv.function("affine_grouped_swiglu_w4g64");
  m->_fn_moe_grouped_down = m->_lib_qmv.function("affine_grouped_down_w4g64");
  m->_fn_moe_ifill = m->_lib_elt.function("moe_ifill_i32");
  m->_fn_moe_hist = m->_lib_elt.function("moe_hist_i32");
  m->_fn_moe_sort_setup = m->_lib_elt.function("moe_sort_setup_i32");
  m->_fn_moe_scatter = m->_lib_elt.function("moe_scatter_i32");
  m->_fn_moe_scatter_back = m->_lib_elt.function("moe_scatter_back_f16");
  m->_fn_moe_qmm_grouped = m->_lib_qmm.function("affine_qmm_grouped_w4g64");
  m->_fn_moe_qmm_grouped_swiglu =
      m->_lib_qmm.function("affine_qmm_grouped_swiglu_w4g64");
  // 8-bit twins (used when the routed experts are w8-quantized). Validated
  // unconditionally so an 8-bit MoE never silently no-ops on an unbuilt fn.
  m->_fn_moe_gather_swiglu_w8 =
      m->_lib_qmv.function("affine_gather_qmv_swiglu_w8g64");
  m->_fn_moe_gather_down_w8 =
      m->_lib_qmv.function("affine_gather_down_qmv_w8g64");
  m->_fn_moe_grouped_swiglu_w8 =
      m->_lib_qmv.function("affine_grouped_swiglu_w8g64");
  m->_fn_moe_grouped_down_w8 =
      m->_lib_qmv.function("affine_grouped_down_w8g64");
  m->_fn_moe_qmm_grouped_w8 =
      m->_lib_qmm.function("affine_qmm_grouped_w8g64");
  m->_fn_moe_qmm_grouped_swiglu_w8 =
      m->_lib_qmm.function("affine_qmm_grouped_swiglu_w8g64");
  if (!m->_fn_moe_gather_swiglu_w8.valid() ||
      !m->_fn_moe_gather_down_w8.valid() ||
      !m->_fn_moe_grouped_swiglu_w8.valid() ||
      !m->_fn_moe_grouped_down_w8.valid() ||
      !m->_fn_moe_qmm_grouped_w8.valid() ||
      !m->_fn_moe_qmm_grouped_swiglu_w8.valid()) {
    return nullptr;
  }
  m->_fn_head_slice = m->_lib_elt.function("head_slice_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_kv_write_paged = m->_lib_elt.function("kv_write_paged_f16");
  m->_fn_rope_partial = m->_lib_rope.function("rope_partial_f16");
  // Fused per-head RMSNorm + partial RoPE (decode): folds q_norm+rope_q
  // and k_norm+rope_k into one dispatch each. Optional -> falls back to
  // the separate rms + rope kernels.
  m->_fn_rms_rope = m->_lib_rope.function("rms_rope_partial_f16");
  m->_fn_mrope = m->_lib_rope.function("mrope_partial_f16");
  m->_fn_sdpa_paged = m->_lib_sdpa.function("sdpa_paged_causal_f16");
  m->_fn_sdpa_paged_mb256 = m->_lib_sdpa.function("sdpa_paged_mb256_f16");
  // D<=128 multi-simdgroup paged decode (shared with the Llama path);
  // optional -- decode falls back to the scalar kernel if absent.
  m->_fn_sdpa_paged_mb = m->_lib_sdpa.function("sdpa_paged_mb_f16");
  m->_fn_sdpa_paged_qtile = m->_lib_sdpa.function("sdpa_paged_qtile_f16");
  // simdgroup_matrix key-split flash prefill (head_dim 256) -- M4-capable
  // (runs on every Apple GPU; the matmul2d sdpa_mma is M5-only).
  m->_fn_sdpa_paged_flash = m->_lib_sdpa.function("sdpa_paged_flash_f16");
  // Register-resident simdgroup_matrix flash prefill (paged KV, head_dim 256):
  // O stays in registers across the key scan (no per-block tg round-trip) ->
  // ~closes the 2x-roofline gap of the key-split sdpa_paged_flash at long ctx.
  m->_fn_sdpa_paged_mma = m->_lib_sdpa.function("sdpa_paged_mma_d256_f16");
  // Shared-prefix batched decode attention (head_dim 256). Optional: the
  // batched step falls back to the per-branch SDPA when either is absent.
  m->_fn_sdpa_shared_mb256 = m->_lib_sdpa.function("sdpa_shared_mb256_f16");
  m->_fn_sdpa_merge_mb256 = m->_lib_sdpa.function("sdpa_merge_mb256_f16");
  // Flash-decode-GQA serial attention (head_dim 256). Optional: serial decode
  // falls back to mb256 when either is absent or the shape is unsupported.
  m->_fn_sdpa_gqa = m->_lib_sdpa.function("sdpa_paged_gqa_mb256_f16");
  // Latency-optimal decode form (per-head simdgroup, UK key-unroll + vec4):
  // ~2x the all-G mb256 form at decode (head_dim % 128 == 0; D=256/128).
  m->_fn_sdpa_gqa_vec = m->_lib_sdpa.function("sdpa_paged_gqa_vec_f16");
  // Key-across-lanes block decode: each lane owns a key, NL/NE tiling -> no
  // per-key simd_sum (llama flash_attn_ext_vec port). head_dim 256 + 128.
  m->_fn_sdpa_gqa_kbl = m->_lib_sdpa.function("sdpa_paged_gqa_kbl_f16");
  // MLX-sdpa_vector port (head_dim 256): one TG per qhead, 32-simd key stripe
  // (32-lane head-dim), intra-TG merge -> partial; 2-pass split + merge.
  // ~10-13% faster than kbl at decode depth (8k-32k). Same oacc/m/l contract.
  m->_fn_sdpa_gqa_vec1 = m->_lib_sdpa.function("sdpa_paged_gqa_vec1_f16");
  // FAITHFUL MLX sdpa_vector_2pass (vec2 pass-1 = per-kvhead/G-simdgroup, no
  // intra-TG merge, block-strided keys; merge2 = MLX's coalesced 2pass_2).
  // 16-25% faster than kbl @8k-32k -- the production decode attention (D=256).
  m->_fn_sdpa_gqa_vec2  = m->_lib_sdpa.function("sdpa_paged_gqa_vec2_f16");
  m->_fn_sdpa_gqa_merge2 = m->_lib_sdpa.function("sdpa_gqa_merge2_f16");
  m->_fn_sdpa_gqa_kbl128 = m->_lib_sdpa.function("sdpa_paged_gqa_kbl128_f16");
  m->_fn_sdpa_gqa_merge = m->_lib_sdpa.function("sdpa_gqa_merge_f16");
  // The decode GQA attention kernel set loads its own member handles (dtype-
  // correct library); it's tuned + sized later in ensure_decode_scratch_.
  m->_decode_set.load(m->_lib_sdpa, cfg.use_bf16);
  m->_fn_gdn_step = m->_lib_gdn.function("qwen3_5_gated_delta_step_f16");
  // Prefill GDN step variant: 4 dv per simdgroup, k/q loaded once/step ->
  // 1.33x the per-dv kernel at prefill T (token-exact, plain MSL so it runs
  // on every GPU). Used only for prefill (T large); decode (T=1) keeps v1.
  m->_fn_gdn_step_ndv4 =
      m->_lib_gdn.function("qwen3_5_gated_delta_step_ndv4_f16");
  if (const char* e = std::getenv("VPIPE_QWEN_GDN_V1")) {
    m->_gdn_force_v1 = (std::atoi(e) != 0);   // diagnostic A/B
  }
  m->_fn_gdn_conv1d = m->_lib_gdn.function("qwen3_5_gdn_conv1d_silu_f16");
  m->_fn_gdn_g_beta = m->_lib_gdn.function("qwen3_5_gdn_g_beta_f32");
  m->_fn_gdn_qk_norm = m->_lib_gdn.function("qwen3_5_gdn_qk_norm_f16");
  m->_fn_gdn_gated_rms = m->_lib_gdn.function("qwen3_5_gdn_gated_rms_f16");
  // Matrix-core (M5+) prefill GEMM: only when the GPU has hardware matrix
  // units and the model is 4-bit. dequant a projection weight -> dense
  // compute-elt scratch, then run dense matmul2d (matrix units). Gated so
  // older GPUs keep the steel quantized GEMM (byte-identical behaviour).
  // VPIPE_QWEN_NO_MMA=1 forces the steel path even on M5 (A/B + safety).
  if (cfg.quant_bits != 8 && mc->supports_matrix_cores()) {
    m->_lib_dequant = mc->load_library("affine_dequant" + sfx);
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma" + sfx);
    m->_fn_dequant = m->_lib_dequant.function("affine_dequant_w4g64");
    // Tile-adaptive dense matmul2d: 128x128 (8 simdgroups) for K <= 4096,
    // 128x256 for deeper K (down_proj) -- the 128x128 tile degrades past
    // K~4096 while 128x256 stays ~10 TFLOP/s at all depths (M5 tile sweep,
    // gemm_mma.tune). ~1.3-1.7x the prior 64x64 tile at prefill shapes.
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_swiglu_inter = m->_lib_elt.function("swiglu_interleaved_f16");
    m->_use_mma = m->_fn_dequant.valid() && m->_fn_dense_mma.valid() &&
                  m->_fn_dense_mma_deep.valid() &&
                  m->_fn_swiglu_inter.valid();
    // Matrix-core (matmul2d) flash attention: head_dim 256 (Qwen3.5) and 128
    // (Llama-3 / the Krea-2 Qwen3-VL text encoder). Optional: its absence just
    // leaves prefill attention on the key-split simdgroup flash / scalar path.
    if (cfg.head_dim == 256 || cfg.head_dim == 128) {
      m->_lib_sdpa_mma = mc->load_library("sdpa_mma" + sfx);
      m->_fn_sdpa_mma = m->_lib_sdpa_mma.function("sdpa_mma_f16");
      m->_fn_sdpa_mma_d128 = m->_lib_sdpa_mma.function("sdpa_mma_d128_f16");
    }
    if (const char* e = std::getenv("VPIPE_QWEN_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_QWEN_MMA_ATTN_MIN_N")) {
      m->_mma_attn_min_n = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_QWEN_NO_MMA")) {
      if (std::atoi(e) != 0) { m->_use_mma = false; }
    }
    if (const char* e = std::getenv("VPIPE_QWEN_MMA_NODQ")) {
      m->_skip_dequant = (std::atoi(e) != 0);   // diagnostic only
    }
  }
  // The prefill GQA attention set loads its members now that mma/use_mma are
  // resolved (it's tuned later in ensure_decode_scratch_).
  m->_prefill_set.load(m->_lib_sdpa, &m->_lib_attn, &m->_lib_sdpa_mma,
                       m->_use_mma);
  // Pipelined-decode kernels (validated lazily in decode_pipelined, not
  // here, so non-pipelined model loads never depend on them).
  m->_fn_embed = m->_lib_elt.function(
      m->_embed_bits == 8 ? "dequant_embed_gather_w8_f16"
                          : "dequant_embed_gather_f16");
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
  m->_fn_lc_sample = m->_lib_elt.function("lc_sample_f16");
  m->_fn_lc_accept = m->_lib_elt.function("lc_accept_f16");
  m->_fn_lc_sample_batch = m->_lib_elt.function("lc_sample_batch_f16");
  if (!m->_fn_qmv.valid() || !m->_fn_qmv_add.valid() ||
      !m->_fn_qmv_swiglu.valid() || !m->_fn_qmm.valid() ||
      !m->_fn_qmm_swiglu.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_rms.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_residual.valid() ||
      !m->_fn_mul_sigmoid.valid() ||
      !m->_fn_head_slice.valid() ||
      !m->_fn_kv_write_paged.valid() || !m->_fn_rope_partial.valid() ||
      !m->_fn_mrope.valid() || !m->_fn_sdpa_paged_mb256.valid() ||
      !m->_fn_sdpa_paged_qtile.valid() ||
      !m->_fn_sdpa_paged.valid() || !m->_fn_gdn_step.valid() ||
      !m->_fn_gdn_conv1d.valid() || !m->_fn_gdn_g_beta.valid() ||
      !m->_fn_gdn_qk_norm.valid() ||
      !m->_fn_gdn_gated_rms.valid()) {
    return nullptr;
  }
  if (const char* e = std::getenv("VPIPE_SDPA_MB_MIN")) {
    const int v = std::atoi(e);
    if (v >= 0) { m->_sdpa_mb_min = v; }
  }
  // Key-split flash prefill (M4 scalar path): on when the kernel loaded;
  // VPIPE_QWEN_NO_FLASH=1 forces the scalar qtile (A/B + safety).
  m->_flash_attn = m->_fn_sdpa_paged_flash.valid();
  if (const char* e = std::getenv("VPIPE_QWEN_NO_FLASH")) {
    if (std::atoi(e) != 0) { m->_flash_attn = false; }
  }
  // Shared-prefix batched decode attention: default ON, requires both
  // kernels; VPIPE_QWEN_SHARED_ATTN=0 forces the per-branch path (A/B).
  m->_shared_attn = m->_fn_sdpa_shared_mb256.valid()
                 && m->_fn_sdpa_merge_mb256.valid();
  if (const char* e = std::getenv("VPIPE_QWEN_SHARED_ATTN")) {
    if (std::atoi(e) == 0) { m->_shared_attn = false; }
  }
  // Flash-decode-GQA serial attention (head_dim <= 256, GQA G=Hq/Hkv<=4):
  // read each KV head once for all G query heads. Covers Qwen3.5 (D=256) and
  // the dense Qwen3-ASR text decoder (D=128, G=2), which both route here.
  // Default ON when capable; VPIPE_GQA_ATTN=0/1 overrides (legacy alias
  // VPIPE_QWEN_GQA_ATTN), VPIPE_GQA_SPLIT sets the split.
  const int gqa_g =
      (cfg.n_kv_heads > 0) ? cfg.n_heads / cfg.n_kv_heads : 0;
  const bool gqa_capable =
      m->_fn_sdpa_gqa.valid() && m->_fn_sdpa_gqa_merge.valid()
      && cfg.head_dim >= 32 && cfg.head_dim <= 256
      && (cfg.head_dim % 32 == 0) && cfg.n_kv_heads > 0
      && (cfg.n_heads % cfg.n_kv_heads == 0) && gqa_g >= 1 && gqa_g <= 8;
  m->_gqa_attn = gqa_capable;
  const char* gqa_e = std::getenv("VPIPE_GQA_ATTN");
  if (!gqa_e) { gqa_e = std::getenv("VPIPE_QWEN_GQA_ATTN"); }
  if (gqa_e) { m->_gqa_attn = gqa_capable && (std::atoi(gqa_e) != 0); }
  const char* gqa_nv = std::getenv("VPIPE_GQA_NO_VEC");
  if (gqa_nv && std::atoi(gqa_nv) != 0) { m->_gqa_vec = false; }
  // Key-across-lanes block decode: ON by default for head_dim 256/128
  // (token-exact + flatter long-ctx decode: +4.8% @8k on the 27B).
  // VPIPE_GQA_BLK=0 opts out.
  const bool kbl_ok =
      m->_fn_sdpa_gqa_kbl.valid() && m->_fn_sdpa_gqa_kbl128.valid();
  m->_gqa_blk = kbl_ok;
  if (const char* gb = std::getenv("VPIPE_GQA_BLK")) {
    m->_gqa_blk = (std::atoi(gb) != 0) && kbl_ok;
  }
  const char* gqa_s = std::getenv("VPIPE_GQA_SPLIT");
  if (!gqa_s) { gqa_s = std::getenv("VPIPE_QWEN_GQA_SPLIT"); }
  if (gqa_s) {
    const int v = std::atoi(gqa_s);
    if (v >= 1 && v <= 256) { m->_gqa_split = v; m->_gqa_split_fixed = true; }
  }
  // MAXM=4 / grouped-x MAXM=8 batched GEMV: read each weight ONCE per 4/8
  // rows vs the MAXM=2 form's ceil(m/2) grid.z tiles. Selected ADAPTIVELY by
  // row count in qmm_auto_ et al. (m=7..8 -> xp2, m in 3..4 -> MAXM=4, else
  // MAXM=2) for BOTH the MTP verify (n=3..4) and realtime-vqa batched decode.
  // _qmv4_enabled gates the MAXM=4 tier (VPIPE_MTP_QMV4).
  m->_qmv4_enabled = m->_fn_qmv_batch4.valid();
  if (const char* e = std::getenv("VPIPE_MTP_QMV4")) {
    m->_qmv4_enabled = m->_fn_qmv_batch4.valid() && (std::atoi(e) != 0);
  }
  // _qmv8_enabled gates the m=7..8 tall-tile tier (VPIPE_QMV_XP8: 0 = off
  // -> MAXM=2 tiling, 1 = bit-identical xp2 (default), 2 = half-dot xh16
  // opt-in; the function was picked at load above).
  m->_qmv8_enabled = m->_fn_qmv_batch8_xp.valid();
  if (const char* e = std::getenv("VPIPE_QMV_XP8")) {
    m->_qmv8_enabled = m->_fn_qmv_batch8_xp.valid() && (std::atoi(e) != 0);
  }
  // m=5..6 mixed 4+rest tiling (see _qmv_mix56). Worth it only when the
  // head tile is the fast xp4 form, so it shares the MAXM=4 tier gate.
  m->_qmv_mix56 =
      m->_qmv4_enabled && m->_fn_qmv_batch.valid();
  if (const char* e = std::getenv("VPIPE_QMV_MIX56")) {
    m->_qmv_mix56 = m->_qmv_mix56 && (std::atoi(e) != 0);
  }
  // Seed the per-m batched-GEMV plans with the static (M5-measured)
  // ladder; the first decode refines them per machine via the probe.
  m->qmv_ladder_defaults_();
  // Leviathan-Chen MTP sampling (opt-in; default off keeps the exact-match
  // sampler that's token-exact vs serial). VPIPE_MTP_LEVIATHAN=1 enables.
  if (const char* e = std::getenv("VPIPE_MTP_LEVIATHAN")) {
    m->_leviathan = (std::atoi(e) != 0) && m->_fn_copy.valid();
  }
  // Decode category profiler: only when VPIPE_QWEN_CATPROF is set does the
  // decode step read VPIPE_QWEN_DUP_CAT (so production never pays a
  // per-step getenv). See encode_decode_step_ / metal_asr_decode_catprof.
  m->_catprof = std::getenv("VPIPE_QWEN_CATPROF") != nullptr;

  // Native k-quant (GGUF) detection: GgufQwen35Converter tags linear
  // weights with a k-quant family dtype ("Q4K"/"Q5K"/"Q6K") instead of the
  // affine U32+scales+biases triple. Probe a tensor present on every layer.
  {
    const auto* probe =
        wts->info(cfg.weight_prefix + cfg.model_seg + "layers.0.mlp.gate_proj.weight");
    m->_kquant = probe != nullptr &&
        (probe->dtype == "Q4K" || probe->dtype == "Q5K" ||
         probe->dtype == "Q6K");
  }
  // Unquantized dense (raw-HF bf16/f16) detection: a representative linear has
  // a `.weight` but NO `.scales` (the affine triple). Probe the last layer's
  // mlp.down_proj (present on every layer). Dense routes the plain f16 GEMM/
  // GEMV; force the quant/mixed flags off so it never takes those paths.
  {
    const int dL = cfg.n_layers > 0 ? cfg.n_layers - 1 : 0;
    // Dense MLP has mlp.down_proj; a MoE layer has only mlp.switch_mlp.
    // down_proj (the per-expert slab) -- probe whichever exists.
    const std::string base = cfg.weight_prefix + cfg.model_seg + "layers." +
                             std::to_string(dL) + ".";
    std::string dp = base + "mlp.down_proj";
    bool moe_raw = false;
    if (!wts->has(dp + ".weight") && !wts->has(dp + ".scales")) {
      // Raw-HF MoE: fused 3D experts (no .weight suffix, no .scales).
      dp = base + "mlp.experts.down_proj";
      moe_raw = wts->has(dp) && !wts->has(dp + ".scales");
    }
    if (!m->_kquant &&
        ((wts->has(dp + ".weight") && !wts->has(dp + ".scales")) || moe_raw)) {
      m->_dense = true;
      m->_kquant = false;
      m->_mixed = false;
    }
  }
  if (m->_dense) {
    // Dense f16 GEMM/GEMV (the same kernels the k-quant prefill runs post-
    // dequant) + the in-stream embed gather. dense_gemm_ rides the matrix
    // units when present (_use_mma, resolved above); the steel GEMM is the
    // M4 / small-M fallback.
    m->_lib_dense = mc->load_library("dense_gemm" + sfx);
    m->_fn_dense_gemm = m->_lib_dense.function("dense_gemm_t_f16");
    m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
    m->_fn_embed_dense = m->_lib_elt.function("embed_gather_f16");
    if (!m->_fn_dense_gemm.valid() || !m->_fn_dense_gemv.valid() ||
        !m->_fn_embed_dense.valid()) {
      return nullptr;
    }
    if (cfg.is_moe()) {
      // Dense f16 MoE gather GEMVs (raw-HF bf16 MoE); reuse the affine path's
      // weight-free route/combine/finalize kernels (loaded for any is_moe()).
      m->_fn_moe_gather_swiglu_dense =
          m->_lib_dense.function("dense_moe_gather_swiglu_f16");
      m->_fn_moe_gather_down_dense =
          m->_lib_dense.function("dense_moe_gather_down_f16");
      m->_fn_moe_gate_dense = m->_lib_dense.function("dense_moe_gate_f16");
      if (!m->_fn_moe_gather_swiglu_dense.valid() ||
          !m->_fn_moe_gather_down_dense.valid() ||
          !m->_fn_moe_gate_dense.valid()) {
        return nullptr;
      }
      // Single-zero pair_eid for the shared-expert gather (slab 0, top_k=1).
      m->_moe_zero_eid = mc->make_shared_buffer(sizeof(std::int32_t));
      if (m->_moe_zero_eid.empty()) { return nullptr; }
      *static_cast<std::int32_t*>(m->_moe_zero_eid.contents()) = 0;
    }
  } else if (m->_dense_embed) {
    // MOSS-TTS affine backbone with a bf16 embed/tied-head: the backbone runs
    // the affine kernels, but the muxer gather + tied lm_head need the dense
    // f16 GEMV/GEMM (full-precision head). Load just those.
    m->_lib_dense = mc->load_library("dense_gemm" + sfx);
    m->_fn_dense_gemm = m->_lib_dense.function("dense_gemm_t_f16");
    m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
    m->_fn_embed_dense = m->_lib_elt.function("embed_gather_f16");
    if (!m->_fn_dense_gemm.valid() || !m->_fn_dense_gemv.valid() ||
        !m->_fn_embed_dense.valid()) {
      return nullptr;
    }
  }
  if (m->_kquant) {
    m->_fn_qmv_q4k = m->_lib_elt.function("qmv_q4k_f16");
    m->_fn_qmv_q5k = m->_lib_elt.function("qmv_q5k_f16");
    // q6_K decode qmv: r2n8 (RPS=2, NSG=8 -> 16 rows/tg) is the sweep winner
    // across the real Q6_K shapes (best on gdn_qkv, ~tied on ffn_down/lm_head).
    // Fall back to r4n2 (the original v2, 8 rows/tg) then the scalar q6k.
    m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_v2_r2n8_f16");
    m->_q6k_nsg = 8;  m->_q6k_rpt = 16;
    if (!m->_fn_qmv_q6k.valid()) {
      m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_v2_f16");
      m->_q6k_nsg = 2;  m->_q6k_rpt = 8;
    }
    if (!m->_fn_qmv_q6k.valid()) {
      m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_f16");
      m->_q6k_nsg = 2;  m->_q6k_rpt = 8;
    }
    // Batched k-quant GEMV (MTP verify weight-bound matmul, weight read once
    // across the 2-row tile). Optional -- kqmv_batch_ loops single-row qmv when
    // unavailable or when VPIPE_GGUF_MTP_LOOPED_GEMV is set (A/B baseline).
    m->_fn_qmv_q4k_batch = m->_lib_elt.function("qmv_q4k_batch_f16");
    m->_fn_qmv_q5k_batch = m->_lib_elt.function("qmv_q5k_batch_f16");
    m->_fn_qmv_q6k_batch = m->_lib_elt.function("qmv_q6k_batch_f16");
    // MAXM=4 twins: the depth-2 MTP verify (n=3..4) reads the weight ONCE (one
    // tile) vs the MAXM=2 form's 2 grid.z tiles. The _qmv4_enabled gate above
    // keyed off the affine batch4 (absent on the k-quant path), so re-derive it
    // here from the k-quant twins, honoring the same VPIPE_MTP_QMV4 override.
    m->_fn_qmv_q4k_batch4 = m->_lib_elt.function("qmv_q4k_batch4_f16");
    m->_fn_qmv_q5k_batch4 = m->_lib_elt.function("qmv_q5k_batch4_f16");
    m->_fn_qmv_q6k_batch4 = m->_lib_elt.function("qmv_q6k_batch4_f16");
    {
      const bool kq4 = m->_fn_qmv_q4k_batch4.valid()
          && m->_fn_qmv_q5k_batch4.valid() && m->_fn_qmv_q6k_batch4.valid();
      bool qmv4_on = true;
      if (const char* e = std::getenv("VPIPE_MTP_QMV4")) {
        qmv4_on = std::atoi(e) != 0;
      }
      m->_qmv4_enabled = m->_qmv4_enabled || (kq4 && qmv4_on);
    }
    m->_fn_embed_q6k = m->_lib_elt.function("embed_gather_q6k_f16");
    m->_fn_embed_q4k = m->_lib_elt.function("embed_gather_q4k_f16");
    m->_fn_dequant_q4k = m->_lib_elt.function("dequant_q4k_f16");
    m->_fn_dequant_q5k = m->_lib_elt.function("dequant_q5k_f16");
    m->_fn_dequant_q6k = m->_lib_elt.function("dequant_q6k_f16");
    m->_fn_copy = m->_lib_elt.function("copy_f16");
    // Q4_K->affine-g32 repack (opt-in): decode qmv (+fused-add), the GPU
    // repack kernel, and the affine g32 prefill dequant.
    m->_fn_qmv_w4g32 = m->_lib_qmv.function("affine_qmv_w4g32");
    m->_fn_qmv_w4g32_add = m->_lib_qmv.function("affine_qmv_w4g32_add");
    m->_fn_repack_q4k = m->_lib_elt.function("repack_q4k_to_affine_g32");
    m->_lib_dequant = mc->load_library("affine_dequant" + sfx);
    m->_fn_dequant_w4g32 = m->_lib_dequant.function("affine_dequant_w4g32");
    m->_lib_dense = mc->load_library("dense_gemm" + sfx);
    m->_fn_dense_gemm = m->_lib_dense.function("dense_gemm_t_f16");
    m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
    if (!m->_fn_qmv_q4k.valid() || !m->_fn_qmv_q5k.valid() ||
        !m->_fn_qmv_q6k.valid() || !m->_fn_embed_q6k.valid() ||
        !m->_fn_embed_q4k.valid() ||
        !m->_fn_dequant_q4k.valid() || !m->_fn_dequant_q5k.valid() ||
        !m->_fn_dequant_q6k.valid() || !m->_fn_copy.valid() ||
        !m->_fn_dense_gemm.valid() || !m->_fn_dense_gemv.valid()) {
      return nullptr;
    }
  }

  // Mixed-precision affine (OptiQ): the de-fused per-tensor path needs BOTH
  // bit widths' kernels (the 4-bit set is already loaded above as _fn_qmv/
  // _add/_qmm). Add the 8-bit qmv/qmv_add/steel-qmm, the affine dequant
  // (w4+w8) and the dense f16 GEMM/GEMV for the heterogeneous-bit prefill
  // (dequant each part into one f16 scratch -> one dense GEMM, the k-quant
  // strategy). The dequant + dense path is needed on EVERY GPU here, so load
  // them independent of matrix cores (M5 already loaded _fn_dequant +
  // dense_mma in the block above; dense_gemm_ picks the matrix units there).
  // MoE needs the w8 set too (router mlp.gate + shared_expert_gate are w8):
  // qmv8 for decode-router, qmm8 for prefill-router, dequant8/dense for the
  // prefill router dequant->GEMM fallback.
  if (m->_mixed || cfg.is_moe()) {
    m->_fn_qmv8 = m->_lib_qmv.function("affine_qmv_w8g64");
    m->_fn_qmv8_add = m->_lib_qmv.function("affine_qmv_w8g64_add");
    m->_fn_qmv8_batch = m->_lib_qmv.function("affine_qmv_batch_w8g64");
    // w8 tiers mirror the w4 ladder: xp forms by default (bit-identical
    // per row; VPIPE_QMV_XP4=0 reverts the MAXM=4 tier), plus the tall
    // xp8 twin (used by vqmm_ per the probed _qmv_plan).
    m->_fn_qmv8_batch4 = m->_lib_qmv.function("affine_qmv_batch4_xp_w8g64");
    if (const char* e = std::getenv("VPIPE_QMV_XP4");
        (e && std::atoi(e) == 0) || !m->_fn_qmv8_batch4.valid()) {
      m->_fn_qmv8_batch4 = m->_lib_qmv.function("affine_qmv_batch4_w8g64");
    }
    m->_fn_qmv8_batch8_xp =
        m->_lib_qmv.function("affine_qmv_batch8_xp2_w8g64");
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8g64");
    if (!m->_fn_dequant.valid()) {       // not taken the M5 mma path above
      m->_lib_dequant = mc->load_library("affine_dequant" + sfx);
      m->_fn_dequant = m->_lib_dequant.function("affine_dequant_w4g64");
    }
    m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8g64");
    m->_fn_requant_w8w4 =
        m->_lib_dequant.function("affine_requant_w8_to_w4_g64");
    if (!m->_fn_dense_gemm.valid()) {    // not taken the k-quant path
      m->_lib_dense = mc->load_library("dense_gemm" + sfx);
      m->_fn_dense_gemm = m->_lib_dense.function("dense_gemm_t_f16");
      m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
    }
    if (!m->_fn_qmv8.valid() || !m->_fn_qmv8_add.valid() ||
        !m->_fn_qmm8.valid() || !m->_fn_dequant.valid() ||
        !m->_fn_dequant8.valid() || !m->_fn_dense_gemm.valid() ||
        !m->_fn_dense_gemv.valid()) {
      return nullptr;
    }
  }

  // Qwen 4-bit checkpoints store scales/biases and all the non-quantized
  // tensors (norms, conv1d, dt_bias) as BF16; our kernels want F16 (and
  // F32 for the recurrence's A_log/dt_bias). Convert at load.
  // Convert a tensor to the compute element type (f16 or bf16). In bf16
  // mode the Qwen checkpoint's BF16 scales/norms/conv pass through raw
  // (no conversion); only f32 sources are narrowed.
  const bool bf16 = cfg.use_bf16;
  auto to_elt = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts->info(name);
    if (info == nullptr) { return {}; }
    const std::string want = bf16 ? "BF16" : "F16";
    if (info->dtype == want) { return wts->load(name, mc); }
    SharedBuffer raw = wts->load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<std::uint16_t*>(out.contents());
    auto to16 = [&](float f) -> std::uint16_t {
      if (bf16) { return f32_to_bf16_(f); }
      _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
    };
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16(bf16_to_f32_(s[i])); }
    } else if (info->dtype == "F16") {
      const auto* s = static_cast<const _Float16*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16((float)s[i]); }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16(s[i]); }
    } else {
      return {};
    }
    return out;
  };
  auto to_f16 = to_elt;   // alias: all callers want the compute element type
  auto to_f32 = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts->info(name);
    if (info == nullptr) { return {}; }
    if (info->dtype == "F32") { return wts->load(name, mc); }
    SharedBuffer raw = wts->load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 4);
    auto* o = static_cast<float*>(out.contents());
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = bf16_to_f32_(s[i]); }
    } else {
      return {};
    }
    return out;
  };
  // Quantized linear: U32 weight (raw), F16 scales+biases (converted).
  auto qtri = [&](const std::string& pfx, SharedBuffer& w, SharedBuffer& s,
                  SharedBuffer& b) -> bool {
    w = wts->load(pfx + ".weight", mc);
    s = to_f16(pfx + ".scales");
    b = to_f16(pfx + ".biases");
    return !w.empty() && !s.empty() && !b.empty();
  };
  // Fuse several quantized projections that share the input (K = hidden)
  // into ONE [sum(N_i), K] matrix by row-concatenation -- one wide GEMM
  // replaces N separate ones (MLX issues them individually). Rows are
  // contiguous in the packed/scale/bias layout so concat is a memcpy.
  // Kc/8 u32 per weight row, Kc/64 (16-bit) per scale & bias row.
  auto fuse_q = [&](const std::vector<std::string>& names,
                    SharedBuffer& fw, SharedBuffer& fs, SharedBuffer& fb)
      -> bool {
    // u32 words per weight row = K * bits / 32 (4-bit: K/8, 8-bit: K/4).
    const std::size_t wrow =
        (std::size_t)cfg.hidden * cfg.quant_bits / 32;
    const std::size_t grow = (std::size_t)cfg.hidden / 64;  // 16-bit / scale row
    std::vector<SharedBuffer> ws, ss, bs;
    std::size_t Ntot = 0;
    for (const auto& nm : names) {
      SharedBuffer w, s, b;
      if (!qtri(nm, w, s, b)) { return false; }
      const auto* info = wts->info(nm + ".scales");   // [N, Kc/64]
      if (info == nullptr) { return false; }
      Ntot += (std::size_t)info->shape[0];
      ws.push_back(std::move(w)); ss.push_back(std::move(s));
      bs.push_back(std::move(b));
    }
    fw = mc->make_shared_buffer(Ntot * wrow * 4);
    fs = mc->make_shared_buffer(Ntot * grow * 2);
    fb = mc->make_shared_buffer(Ntot * grow * 2);
    if (fw.empty() || fs.empty() || fb.empty()) { return false; }
    std::size_t wo = 0, go = 0;
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const std::size_t wn = ws[i].byte_size();
      const std::size_t gn = ss[i].byte_size();
      std::memcpy((char*)fw.contents() + wo, ws[i].contents(), wn);
      std::memcpy((char*)fs.contents() + go, ss[i].contents(), gn);
      std::memcpy((char*)fb.contents() + go, bs[i].contents(), gn);
      wo += wn; go += gn;
    }
    return true;
  };

  // Fused interleaved gate/up (row 2g=gate g, 2g+1=up g) -- same as the
  // Llama path; scales/biases are already F16 here.
  const int Kc = cfg.hidden, Fc = cfg.ffn_inner;
  const std::size_t wrow = (std::size_t)Kc * cfg.quant_bits / 32, grow = Kc / 64;
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

  // ---- Unquantized dense f16 loaders (raw-HF bf16 checkpoint) -------
  // Each helper narrows the raw bf16/f16 `.weight` to the compute element
  // (2 bytes/elt) and writes the SAME fused/interleaved layout the affine
  // path produces, but as a plain [N,K] f16 weight (no scales/biases). The
  // forward dispatches the dense GEMM/GEMV whenever the scales slot is empty.
  //
  // Row-concatenate several [N_i, K] f16 weights into one [sum(N_i), K]
  // matrix (q|k|v fuse, in_proj qkv|z|a|b fuse): the dense GEMM reads it as
  // one taller weight. Each row is K elements = K*2 bytes.
  auto fuse_f16 = [&](const std::vector<std::string>& names,
                      SharedBuffer& out) -> bool {
    std::vector<SharedBuffer> ws;
    std::size_t total_bytes = 0;
    for (const auto& nm : names) {
      SharedBuffer w = to_f16(nm + ".weight");
      if (w.empty()) { return false; }
      total_bytes += w.byte_size();
      ws.push_back(std::move(w));
    }
    out = mc->make_shared_buffer(total_bytes);
    if (out.empty()) { return false; }
    std::size_t off = 0;
    for (auto& w : ws) {
      std::memcpy((char*)out.contents() + off, w.contents(), w.byte_size());
      off += w.byte_size();
    }
    return true;
  };
  // +1 RMSNorm fold: raw-HF Qwen stores zero-centered norm weights; the model
  // applies (1+weight) while vpipe's rms kernel multiplies by weight directly.
  // Add 1.0 to every element of a loaded f16/bf16 norm buffer. EXCLUDES the
  // gated GDN norm (linear_attn.norm, ones-init -- already absolute).
  // Dense raw-HF norm fold applies only when the checkpoint is zero-centered
  // (Qwen3.5 family). Plain-Qwen3 backbones (MOSS) use standard RMSNorm.
  const bool fold_norm_plus_one = cfg.zero_centered_norm;
  auto add_one_f16 = [&](SharedBuffer& buf) {
    if (buf.empty()) { return; }
    auto* p = static_cast<std::uint16_t*>(buf.contents());
    const std::size_t n = buf.byte_size() / 2;
    for (std::size_t i = 0; i < n; ++i) {
      if (bf16) {
        p[i] = f32_to_bf16_(bf16_to_f32_(p[i]) + 1.0f);
      } else {
        _Float16 h; std::memcpy(&h, &p[i], 2);
        h = (_Float16)((float)h + 1.0f);
        std::memcpy(&p[i], &h, 2);
      }
    }
  };

  // ---- Dense f16 MoE interleave (raw-HF bf16 MoE) ------------------
  // Mirror the affine interleave/interleave_moe layouts but with plain f16
  // rows (2 bytes/elt, no scales/biases). gate|up -> [2*rows, H] interleaved
  // (row 2g=gate g, 2g+1=up g); the per-expert dense_gemv forward reads gate
  // at slab row 2g, up at 2g+1. K = cfg.hidden, each row H elements.
  const std::size_t wrowHf = (std::size_t)cfg.hidden;          // f16 elts/row
  auto interleave_w4_f16 = [&](const SharedBuffer& gw, const SharedBuffer& uw,
                               int rows, SharedBuffer& ow) -> bool {
    ow = mc->make_shared_buffer((std::size_t)2 * rows * wrowHf * 2);
    if (ow.empty() || gw.empty() || uw.empty()) { return false; }
    const auto* gp = static_cast<const std::uint16_t*>(gw.contents());
    const auto* up = static_cast<const std::uint16_t*>(uw.contents());
    auto* op = static_cast<std::uint16_t*>(ow.contents());
    for (std::size_t g = 0; g < (std::size_t)rows; ++g) {
      std::memcpy(op + (2 * g) * wrowHf, gp + g * wrowHf, wrowHf * 2);
      std::memcpy(op + (2 * g + 1) * wrowHf, up + g * wrowHf, wrowHf * 2);
    }
    return true;
  };
  // Batched-expert twin: gate/up are 3D f16 [E, rows, H]; interleave each
  // expert slab -> [E, 2*rows, H] f16. Slab e's gate rows live at
  // e*rows*wrowHf f16 elements.
  auto interleave_moe_f16 = [&](const SharedBuffer& gw, const SharedBuffer& uw,
                                int E, int rows, SharedBuffer& ow) -> bool {
    ow = mc->make_shared_buffer((std::size_t)E * 2 * rows * wrowHf * 2);
    if (ow.empty() || gw.empty() || uw.empty()) { return false; }
    const auto* gp = static_cast<const std::uint16_t*>(gw.contents());
    const auto* up = static_cast<const std::uint16_t*>(uw.contents());
    auto* op = static_cast<std::uint16_t*>(ow.contents());
    for (std::size_t e = 0; e < (std::size_t)E; ++e) {
      const std::size_t ob = e * 2 * rows * wrowHf;
      const std::size_t ib = e * rows * wrowHf;
      for (std::size_t g = 0; g < (std::size_t)rows; ++g) {
        std::memcpy(op + ob + (2 * g) * wrowHf, gp + ib + g * wrowHf,
                    wrowHf * 2);
        std::memcpy(op + ob + (2 * g + 1) * wrowHf, up + ib + g * wrowHf,
                    wrowHf * 2);
      }
    }
    return true;
  };

  // ---- MoE weight interleave (Qwen3.5-MoE) -------------------------
  // Interleave two affine [rows, H] matrices (gate, up) into one
  // [2*rows, H] (row 2g=gate g, 2g+1=up g) -- the layout the SwiGLU-fused
  // matvec/GEMM reads. K = cfg.hidden. Used for the shared expert and (slab
  // by slab) for the batched experts. The packed weight-row width depends on
  // the expert quant width `bw` (w4: H/8 u32, w8: H/4 u32); pass it in so
  // 4-bit and 8-bit experts both interleave correctly (a hardcoded w4 width
  // corrupted + under-allocated w8 slabs).
  const std::size_t growH = (std::size_t)cfg.hidden / 64;      // 32 (g64)
  auto interleave_w4 = [&](int bw, const SharedBuffer& gw,
                           const SharedBuffer& gs,
                           const SharedBuffer& gb, const SharedBuffer& uw,
                           const SharedBuffer& us, const SharedBuffer& ub,
                           int rows, SharedBuffer& ow, SharedBuffer& os,
                           SharedBuffer& ob) -> bool {
    const std::size_t wrowH = (std::size_t)cfg.hidden * bw / 32;
    ow = mc->make_shared_buffer((std::size_t)2 * rows * wrowH * 4);
    os = mc->make_shared_buffer((std::size_t)2 * rows * growH * 2);
    ob = mc->make_shared_buffer((std::size_t)2 * rows * growH * 2);
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
    for (std::size_t g = 0; g < (std::size_t)rows; ++g) {
      std::memcpy(owp + (2 * g) * wrowH, gwp + g * wrowH, wrowH * 4);
      std::memcpy(owp + (2 * g + 1) * wrowH, uwp + g * wrowH, wrowH * 4);
      std::memcpy(osp + (2 * g) * growH, gsp + g * growH, growH * 2);
      std::memcpy(osp + (2 * g + 1) * growH, usp + g * growH, growH * 2);
      std::memcpy(obp + (2 * g) * growH, gbp + g * growH, growH * 2);
      std::memcpy(obp + (2 * g + 1) * growH, ubp + g * growH, growH * 2);
    }
    return true;
  };
  // Batched-expert twin: gate/up are 3D [E, rows, H]; interleave each expert
  // slab -> [E, 2*rows, H]. Slab e's gate rows live at e*rows*wrowH words.
  auto interleave_moe = [&](int bw, const SharedBuffer& gw,
                            const SharedBuffer& gs, const SharedBuffer& gb,
                            const SharedBuffer& uw, const SharedBuffer& us,
                            const SharedBuffer& ub, int E, int rows,
                            SharedBuffer& ow, SharedBuffer& os,
                            SharedBuffer& ob) -> bool {
    const std::size_t wrowH = (std::size_t)cfg.hidden * bw / 32;
    ow = mc->make_shared_buffer((std::size_t)E * 2 * rows * wrowH * 4);
    os = mc->make_shared_buffer((std::size_t)E * 2 * rows * growH * 2);
    ob = mc->make_shared_buffer((std::size_t)E * 2 * rows * growH * 2);
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
    for (std::size_t e = 0; e < (std::size_t)E; ++e) {
      const std::size_t wob = e * 2 * rows * wrowH, gob = e * 2 * rows * growH;
      const std::size_t wib = e * rows * wrowH,     gib = e * rows * growH;
      for (std::size_t g = 0; g < (std::size_t)rows; ++g) {
        std::memcpy(owp + wob + (2 * g) * wrowH,
                    gwp + wib + g * wrowH, wrowH * 4);
        std::memcpy(owp + wob + (2 * g + 1) * wrowH,
                    uwp + wib + g * wrowH, wrowH * 4);
        std::memcpy(osp + gob + (2 * g) * growH,
                    gsp + gib + g * growH, growH * 2);
        std::memcpy(osp + gob + (2 * g + 1) * growH,
                    usp + gib + g * growH, growH * 2);
        std::memcpy(obp + gob + (2 * g) * growH,
                    gbp + gib + g * growH, growH * 2);
        std::memcpy(obp + gob + (2 * g + 1) * growH,
                    ubp + gib + g * growH, growH * 2);
      }
    }
    return true;
  };

  // ---- Native k-quant (GGUF) weight loaders ------------------------
  // The .weight tensors are raw k-quant blocks (no scales/biases); the
  // dtype tag picks the family. Q4_K_M is heterogeneous, so only same-type
  // projections fuse (q+k); v / o / in_proj parts / mlp stay separate.
  auto kq_of = [](const std::string& dt) -> KQ {
    if (dt == "Q4K") { return KQ::kQ4K; }
    if (dt == "Q5K") { return KQ::kQ5K; }
    if (dt == "Q6K") { return KQ::kQ6K; }
    return KQ::kNone;
  };
  auto load_kq = [&](const std::string& name, SharedBuffer& buf,
                     KQ& ty) -> bool {
    const auto* info = wts->info(name);
    if (info == nullptr) {
      // Required k-quant tensor absent. The GGUF converter DROPS tensors of an
      // unsupported quant type (Q2_K/Q3_K/Q4_0/Q8_0-linear/IQ*), so a missing
      // tensor here is most often exactly that. Fail loud with the name (no
      // direct stderr -- via the session, mirroring the affine-quant guard).
      if (const SessionContextIntf* s = mc->session()) {
        s->warn(fmt("[qwen] k-quant load failed: required tensor '{}' is "
                    "absent -- wrong checkpoint, or an unsupported GGUF quant "
                    "type was dropped at conversion (only Q4_K/Q5_K/Q6_K "
                    "linears are supported)", name));
      }
      return false;
    }
    ty = kq_of(info->dtype);
    if (ty == KQ::kNone) {
      if (const SessionContextIntf* s = mc->session()) {
        s->warn(fmt("[qwen] k-quant load failed: tensor '{}' has unsupported "
                    "type '{}' (only Q4_K/Q5_K/Q6_K)", name, info->dtype));
      }
      return false;
    }
    buf = wts->load(name, mc);
    return !buf.empty();
  };
  // Fuse two same-family raw k-quant weights by row (byte) concatenation:
  // [N1+N2, K] -- the qmv reads it as one taller matrix. nrows = N1+N2.
  auto fuse_kq = [&](const std::string& n1, const std::string& n2,
                     SharedBuffer& buf, KQ& ty, int& nrows) -> bool {
    const auto* i1 = wts->info(n1);
    const auto* i2 = wts->info(n2);
    if (i1 == nullptr || i2 == nullptr || i1->dtype != i2->dtype) {
      return false;
    }
    ty = kq_of(i1->dtype);
    if (ty == KQ::kNone) { return false; }
    SharedBuffer b1 = wts->load(n1, mc), b2 = wts->load(n2, mc);
    if (b1.empty() || b2.empty()) { return false; }
    buf = mc->make_shared_buffer(b1.byte_size() + b2.byte_size());
    if (buf.empty()) { return false; }
    std::memcpy(buf.contents(), b1.contents(), b1.byte_size());
    std::memcpy(static_cast<char*>(buf.contents()) + b1.byte_size(),
                b2.contents(), b2.byte_size());
    nrows = static_cast<int>(i1->shape[0] + i2->shape[0]);
    return true;
  };
  // The two tiny Q8_0 alpha/beta projections come from the converter as
  // f32; narrow to f16 and concat into one [2*Hv, H] dense GEMV weight.
  auto load_ab = [&](const std::string& na, const std::string& nb,
                     SharedBuffer& buf) -> bool {
    SharedBuffer a = to_f16(na), b = to_f16(nb);
    if (a.empty() || b.empty()) { return false; }
    buf = mc->make_shared_buffer(a.byte_size() + b.byte_size());
    if (buf.empty()) { return false; }
    std::memcpy(buf.contents(), a.contents(), a.byte_size());
    std::memcpy(static_cast<char*>(buf.contents()) + a.byte_size(),
                b.contents(), b.byte_size());
    return true;
  };
  // Repack a raw Q4_K weight [N rows, K cols] -> affine-4bit-g32 (aw/as/ab)
  // on the GPU, for the faster affine decode matvec. Lossless. Caller keeps
  // the raw buffer (prefill still dequant->GEMMs it).
  auto repack_q4k = [&](const SharedBuffer& raw, int N, int K,
                        SharedBuffer& aw, SharedBuffer& as,
                        SharedBuffer& ab) -> bool {
    if (!m->_fn_repack_q4k.valid() || !m->_fn_qmv_w4g32.valid()) { return false; }
    if (K % 256 != 0 || N % 8 != 0 || raw.empty()) { return false; }
    aw = mc->make_shared_buffer((std::size_t)N * (K / 8) * 4);
    as = mc->make_shared_buffer((std::size_t)N * (K / 32) * 2);
    ab = mc->make_shared_buffer((std::size_t)N * (K / 32) * 2);
    if (aw.empty() || as.empty() || ab.empty()) { return false; }
    const int nsb = K / 256;
    auto st = mc->make_command_stream();
    { auto enc = st.begin_compute();
      enc.set_function(m->_fn_repack_q4k);
      enc.set_buffer(0, raw); enc.set_buffer(1, aw); enc.set_buffer(2, as);
      enc.set_buffer(3, ab); enc.set_constant(4, K); enc.set_constant(5, N);
      enc.dispatch({(unsigned)nsb, (unsigned)N, 1}, {(unsigned)nsb, 1, 1});
    }
    st.commit().wait();
    return true;
  };

  m->_layers.resize(cfg.n_layers);
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p =
        cfg.weight_prefix + cfg.model_seg + "layers." + std::to_string(L) + ".";
    Layer& ly = m->_layers[L];
    ly.is_full = cfg.layer_is_full(L);
    // Streaming calibration: leave the layer's weights EMPTY (loaded one at a
    // time later by calib_build_layer) so the full model never resides.
    if (cfg.calib_stream) { continue; }
    ly.in_ln = to_f16(p + "input_layernorm.weight");
    ly.post_ln = to_f16(p + "post_attention_layernorm.weight");
    bool ok = !ly.in_ln.empty() && !ly.post_ln.empty();
    if (m->_dense && fold_norm_plus_one) {
      add_one_f16(ly.in_ln); add_one_f16(ly.post_ln);
    }
    if (ly.is_full) {
      if (m->_dense) {
        // Dense f16: fuse q|k|v into ONE [2*qd+2*kd, H] weight (q_proj gated,
        // 2*qd), o_proj plain. No scales/biases -> the forward dispatches the
        // dense GEMM/GEMV when the s slot is empty.
        ok = ok && fuse_f16({p + "self_attn.q_proj", p + "self_attn.k_proj",
                             p + "self_attn.v_proj"}, ly.qw);
        ok = ok && !(ly.ow = to_f16(p + "self_attn.o_proj.weight")).empty();
      } else if (m->_kquant) {
        // q+k fuse (both q4_K); v is often q6_K in Q4_K_M -> separate.
        ok = ok && fuse_kq(p + "self_attn.q_proj.weight",
                           p + "self_attn.k_proj.weight",
                           ly.kqk, ly.kqk_t, ly.kqk_n);
        ok = ok && load_kq(p + "self_attn.v_proj.weight", ly.kqv, ly.kqv_t);
        ok = ok && load_kq(p + "self_attn.o_proj.weight", ly.kqo, ly.kqo_t);
        static const bool kAff = q4k_affine_enabled_();
        if (ok && kAff && ly.kqk_t == KQ::kQ4K) {
          ly.qk_q4k_aff = repack_q4k(ly.kqk, ly.kqk_n, cfg.hidden,
                                     ly.qk_aw, ly.qk_as, ly.qk_ab);
          if (ly.qk_q4k_aff) { ly.kqk = {}; }
        }
        if (ok && kAff && ly.kqo_t == KQ::kQ4K) {
          // o_proj: [hidden, qd]; K = qd (attention output width).
          ly.o_q4k_aff = repack_q4k(ly.kqo, cfg.hidden, cfg.qd(),
                                    ly.o_aw, ly.o_as, ly.o_ab);
          if (ly.o_q4k_aff) { ly.kqo = {}; }
        }
      } else if (m->_mixed) {
        const int qb = affine_bits(p + "self_attn.q_proj", cfg.hidden);
        const int kb = affine_bits(p + "self_attn.k_proj", cfg.hidden);
        const int vb = affine_bits(p + "self_attn.v_proj", cfg.hidden);
        ok = ok && qtri(p + "self_attn.o_proj", ly.ow, ly.os, ly.ob);
        ly.o_bits = affine_bits(p + "self_attn.o_proj", cfg.qd());
        if (qb == kb && kb == vb && qb == cfg.quant_bits) {
          // Uniform q/k/v at the BASE width: FUSE into one [2*qd+2*kd, H] GEMM
          // like the non-mixed path -- the forward's fused branch (gated on
          // !ly.qkv_fused) then dispatches it with the base-width qmv/qmm, so
          // no per-bits handling is needed. o_proj stays per-tensor.
          ok = ok && fuse_q({p + "self_attn.q_proj", p + "self_attn.k_proj",
                             p + "self_attn.v_proj"}, ly.qw, ly.qs, ly.qb);
          ly.qkv_fused = true;
          ly.q_bits = ly.k_bits = ly.v_bits = qb;
        } else {
          // Genuinely mixed q/k/v -> de-fused per-tensor: q|k|v each keep their
          // own triple + bits, the decode writes them into the same
          // _d_qfull[q|k|v] offsets the fused path produces, the prefill
          // dequants each into _w_deq for one GEMM.
          ok = ok && qtri(p + "self_attn.q_proj", ly.qw, ly.qs, ly.qb);
          ok = ok && qtri(p + "self_attn.k_proj", ly.kw, ly.ks, ly.kb);
          ok = ok && qtri(p + "self_attn.v_proj", ly.vw, ly.vs, ly.vb);
          ly.q_bits = qb; ly.k_bits = kb; ly.v_bits = vb;
        }
      } else {
        // Fuse q|k|v into ONE [2*qd+2*kd, H] GEMM (q_proj is gated, 2*qd).
        ok = ok && fuse_q({p + "self_attn.q_proj", p + "self_attn.k_proj",
                           p + "self_attn.v_proj"}, ly.qw, ly.qs, ly.qb);
        ok = ok && qtri(p + "self_attn.o_proj", ly.ow, ly.os, ly.ob);
      }
      ly.q_norm = to_f16(p + "self_attn.q_norm.weight");
      ly.k_norm = to_f16(p + "self_attn.k_norm.weight");
      ok = ok && !ly.q_norm.empty() && !ly.k_norm.empty();
      if (m->_dense && fold_norm_plus_one) {
        add_one_f16(ly.q_norm); add_one_f16(ly.k_norm);
      }
    } else {
      if (m->_dense) {
        // Dense f16: fuse the four in_proj projections into ONE
        // [Cd+vald+2Hv, H] weight (qkv|z|a|b order), out_proj plain.
        ok = ok && fuse_f16({p + "linear_attn.in_proj_qkv",
                             p + "linear_attn.in_proj_z",
                             p + "linear_attn.in_proj_a",
                             p + "linear_attn.in_proj_b"}, ly.iqw);
        ly.gow = to_f16(p + "linear_attn.out_proj.weight");
        ok = ok && !ly.gow.empty();
      } else if (m->_kquant) {
        // in_proj is heterogeneous (qkv q5_K | z q4_K | a,b q8_0) -> the
        // decode path runs one qmv per part into the mixqkv offsets.
        ok = ok && load_kq(p + "linear_attn.in_proj_qkv.weight",
                           ly.kqkv, ly.kqkv_t);
        ok = ok && load_kq(p + "linear_attn.in_proj_z.weight",
                           ly.kqz, ly.kqz_t);
        ok = ok && load_ab(p + "linear_attn.in_proj_a.weight",
                           p + "linear_attn.in_proj_b.weight", ly.kqab);
        ok = ok && load_kq(p + "linear_attn.out_proj.weight",
                           ly.kqout, ly.kqout_t);
      } else if (m->_mixed) {
        // De-fused per-tensor in_proj: qkv->iqw, z->izw, a->iaw, b->ibw, each
        // its own triple + bits, written to the same _d_mixqkv[qkv|z|a|b]
        // offsets the fused path produces; out_proj->gow.
        ok = ok && qtri(p + "linear_attn.in_proj_qkv", ly.iqw, ly.iqs, ly.iqb);
        ok = ok && qtri(p + "linear_attn.in_proj_z", ly.izw, ly.izs, ly.izb);
        ok = ok && qtri(p + "linear_attn.in_proj_a", ly.iaw, ly.ias, ly.iab);
        ok = ok && qtri(p + "linear_attn.in_proj_b", ly.ibw, ly.ibs, ly.ibb);
        ok = ok && qtri(p + "linear_attn.out_proj", ly.gow, ly.gos, ly.gob);
        ly.qkv_bits = affine_bits(p + "linear_attn.in_proj_qkv", cfg.hidden);
        ly.z_bits = affine_bits(p + "linear_attn.in_proj_z", cfg.hidden);
        ly.a_bits = affine_bits(p + "linear_attn.in_proj_a", cfg.hidden);
        ly.b_bits = affine_bits(p + "linear_attn.in_proj_b", cfg.hidden);
        ly.gout_bits = affine_bits(p + "linear_attn.out_proj", cfg.value_dim());
      } else {
        // Fuse the four in_proj projections into ONE [Cd+vald+2Hv, H] GEMM.
        ok = ok && fuse_q({p + "linear_attn.in_proj_qkv",
                           p + "linear_attn.in_proj_z",
                           p + "linear_attn.in_proj_a",
                           p + "linear_attn.in_proj_b"},
                          ly.iqw, ly.iqs, ly.iqb);
        ok = ok && qtri(p + "linear_attn.out_proj", ly.gow, ly.gos, ly.gob);
      }
      ly.conv_w = to_f16(p + "linear_attn.conv1d.weight");
      ly.A_log = to_f32(p + "linear_attn.A_log");
      ly.dt_bias = to_f32(p + "linear_attn.dt_bias");
      ly.gdn_norm = to_f16(p + "linear_attn.norm.weight");
      ok = ok && !ly.conv_w.empty() && !ly.A_log.empty() &&
           !ly.dt_bias.empty() && !ly.gdn_norm.empty();
    }
    // MLP (every layer).
    if (m->_dense && cfg.is_moe()) {
      // Dense f16 Mixture-of-Experts (raw-HF bf16 MoE). Router (mlp.gate),
      // batched experts (switch_mlp gate|up interleaved per slab + down), the
      // dense shared expert (gate|up interleaved + down) and the shared sigmoid
      // gate -- ALL plain f16 (no scales). The forward runs a per-expert
      // dense_gemv loop + the existing weight-free routing/combine kernels.
      const int E = cfg.n_experts, I = cfg.moe_inner, S = cfg.moe_shared_inner;
      ok = ok && !(ly.rgw = to_f16(p + "mlp.gate.weight")).empty();  // [E,H]
      {
        SharedBuffer gw = to_f16(p + "mlp.switch_mlp.gate_proj.weight");
        SharedBuffer uw = to_f16(p + "mlp.switch_mlp.up_proj.weight");
        ok = ok && interleave_moe_f16(gw, uw, E, I, ly.eguw);   // [E,2I,H]
      }
      ok = ok && !(ly.edw =
                   to_f16(p + "mlp.switch_mlp.down_proj.weight")).empty();
      {
        SharedBuffer gw = to_f16(p + "mlp.shared_expert.gate_proj.weight");
        SharedBuffer uw = to_f16(p + "mlp.shared_expert.up_proj.weight");
        ok = ok && interleave_w4_f16(gw, uw, S, ly.sguw);        // [2S,H]
      }
      ok = ok && !(ly.sdw =
                   to_f16(p + "mlp.shared_expert.down_proj.weight")).empty();
      ok = ok && !(ly.segw =
                   to_f16(p + "mlp.shared_expert_gate.weight")).empty();
    } else if (m->_dense) {
      // Dense f16: gate->guw, up->uw, down->dw (de-fused, like the mixed
      // path -- two GEMVs + plain SwiGLU, no interleaved-swiglu kernel needed
      // so it runs on every GPU). The forward dispatches dense (s empty).
      ok = ok && !(ly.guw = to_f16(p + "mlp.gate_proj.weight")).empty();
      ok = ok && !(ly.uw = to_f16(p + "mlp.up_proj.weight")).empty();
      ok = ok && !(ly.dw = to_f16(p + "mlp.down_proj.weight")).empty();
    } else if (m->_kquant) {
      // gate/up stay separate raw q4_K (two qmv + swiglu at decode); the
      // affine path's interleaved-swiglu kernel has no k-quant variant.
      ok = ok && load_kq(p + "mlp.gate_proj.weight", ly.kqgate, ly.kqgate_t);
      ok = ok && load_kq(p + "mlp.up_proj.weight", ly.kqup, ly.kqup_t);
      ok = ok && load_kq(p + "mlp.down_proj.weight", ly.kqdown, ly.kqdown_t);
      // Opt-in: repack the Q4_K gate/up to affine-g32 for the ~1.9x faster
      // affine decode matvec (decode is FFN-dominated). Prefill keeps the raw.
      static const bool kAff = q4k_affine_enabled_();
      if (ok && kAff && ly.kqgate_t == KQ::kQ4K && ly.kqup_t == KQ::kQ4K) {
        ly.ffn_q4k_aff =
            repack_q4k(ly.kqgate, cfg.ffn_inner, cfg.hidden,
                       ly.gate_aw, ly.gate_as, ly.gate_ab) &&
            repack_q4k(ly.kqup, cfg.ffn_inner, cfg.hidden,
                       ly.up_aw, ly.up_as, ly.up_ab);
        // Repacked for BOTH decode (amv_g32) and prefill (aqmm_g32_): free the
        // raw Q4_K (the affine triple is the only copy now).
        if (ly.ffn_q4k_aff) { ly.kqgate = {}; ly.kqup = {}; }
      }
    } else if (m->_mixed) {
      // Mixed per-tensor MLP. When gate/up share the (4-bit) width, interleave
      // them into guw and run the FUSED swiglu qmv/qmm (the same kernels the
      // uniform path uses -- recovers the fusion the de-fused path drops on
      // 28/32 OptiQ layers). Otherwise keep gate->guw, up->uw de-fused (the 4
      // genuinely mixed-bit layers). down->dw always per-tensor (own bits).
      ly.gate_bits = affine_bits(p + "mlp.gate_proj", cfg.hidden);
      ly.up_bits = affine_bits(p + "mlp.up_proj", cfg.hidden);
      ly.down_bits = affine_bits(p + "mlp.down_proj", cfg.ffn_inner);
      if (ly.gate_bits == 4 && ly.up_bits == 4) {
        SharedBuffer gw, gs, gb, uw, us, ub;
        ok = ok && qtri(p + "mlp.gate_proj", gw, gs, gb);
        ok = ok && qtri(p + "mlp.up_proj", uw, us, ub);
        ok = ok && interleave(gw, gs, gb, uw, us, ub, ly.guw, ly.gus, ly.gub);
        ly.mlp_fused = true;
      } else {
        ok = ok && qtri(p + "mlp.gate_proj", ly.guw, ly.gus, ly.gub);
        ok = ok && qtri(p + "mlp.up_proj", ly.uw, ly.us, ly.ub);
      }
      ok = ok && qtri(p + "mlp.down_proj", ly.dw, ly.ds, ly.db);
    } else if (cfg.is_moe()) {
      // Mixture-of-Experts MLP (Qwen3.5-MoE). Router (w8) + batched experts
      // (w4, gate|up interleaved per slab + down) + dense shared expert (w4,
      // gate|up interleaved + down) + shared-expert sigmoid gate (w8).
      const int E = cfg.n_experts, I = cfg.moe_inner, S = cfg.moe_shared_inner;
      // Routed-expert quant width (4 or 8) -> w4/w8 expert kernels at dispatch.
      const int egb = affine_bits(p + "mlp.switch_mlp.gate_proj", cfg.hidden);
      ly.eg_bits = (egb == 8) ? 8 : 4;
      ok = ok && qtri(p + "mlp.gate", ly.rgw, ly.rgs, ly.rgb);   // router w8
      {
        SharedBuffer gw, gs, gb, uw, us, ub;
        ok = ok && qtri(p + "mlp.switch_mlp.gate_proj", gw, gs, gb);
        ok = ok && qtri(p + "mlp.switch_mlp.up_proj", uw, us, ub);
        ok = ok && interleave_moe(ly.eg_bits, gw, gs, gb, uw, us, ub, E, I,
                                  ly.eguw, ly.egus, ly.egub);
      }
      ok = ok && qtri(p + "mlp.switch_mlp.down_proj", ly.edw, ly.eds, ly.edb);
      {
        SharedBuffer gw, gs, gb, uw, us, ub;
        ok = ok && qtri(p + "mlp.shared_expert.gate_proj", gw, gs, gb);
        ok = ok && qtri(p + "mlp.shared_expert.up_proj", uw, us, ub);
        // Shared expert gate|up is read by the global-width qmv/qmm swiglu
        // kernels (w8g64 iff cfg.quant_bits==8, else w4g64) -- pack to match.
        ok = ok && interleave_w4(cfg.quant_bits == 8 ? 8 : 4,
                                 gw, gs, gb, uw, us, ub, S,
                                 ly.sguw, ly.sgus, ly.sgub);
      }
      ok = ok && qtri(p + "mlp.shared_expert.down_proj", ly.sdw, ly.sds, ly.sdb);
      ok = ok && qtri(p + "mlp.shared_expert_gate", ly.segw, ly.segs, ly.segb);
    } else {
      SharedBuffer gw, gs, gb, uw, us, ub;
      ok = ok && qtri(p + "mlp.gate_proj", gw, gs, gb);
      ok = ok && qtri(p + "mlp.up_proj", uw, us, ub);
      ok = ok && interleave(gw, gs, gb, uw, us, ub, ly.guw, ly.gus, ly.gub);
      ok = ok && qtri(p + "mlp.down_proj", ly.dw, ly.ds, ly.db);
    }
    if (!ok) {
      return nullptr;
    }
  }

  bool ok = true;
  if (!cfg.backbone_only) {
    if (m->_kquant) {
      // token_embd is a raw k-quant table: gathered + matvec'd natively
      // (embed_gather_q{4,6}k / qmv_q6k), no affine requant. Q6_K on tied
      // checkpoints (lm_head == embed); Q4_K on untied ones (the 27B), whose
      // lm_head is the separate Q6_K output.weight loaded below.
      const std::string ep = cfg.weight_prefix + cfg.model_seg + "embed_tokens.";
      if (wts->has(ep + "q6k")) {
        m->_embed_q6k = wts->load(ep + "q6k", mc);
        m->_embed_kqt = KQ::kQ6K;
      } else if (wts->has(ep + "q4k")) {
        m->_embed_q6k = wts->load(ep + "q4k", mc);
        m->_embed_kqt = KQ::kQ4K;
      }
      m->_embed_is_q6k = !m->_embed_q6k.empty();
      if (!m->_embed_is_q6k) {
        // Only Q4_K / Q6_K token_embd have a native gather kernel. A Q5_K embed
        // (the converter emits a .q5k tag) is intentionally unsupported -- fail
        // loud with the reason instead of an opaque nullptr. (Q5_K works for
        // every LINEAR; only the embedding table lacks a q5k gather.)
        if (const SessionContextIntf* s = mc->session()) {
          s->warn(fmt("[qwen] unsupported k-quant embedding table: {} -- only "
                      "Q4_K / Q6_K token_embd is supported (re-quantize the "
                      "embedding table to Q6_K or Q4_K)",
                      wts->has(ep + "q5k") ? "token_embd is Q5_K"
                          : "token_embd is missing or an unknown k-quant type"));
        }
      }
      ok = m->_embed_is_q6k;
    } else if (m->_dense || m->_dense_embed) {
      // Dense f16 embed table [vocab, H] (scales/biases left empty). The
      // _dense_embed (MOSS-TTS) case keeps the bf16 embed/tied-head in full
      // precision atop an affine backbone -- the muxer + lm_head take the
      // dense path below; the backbone linears stay affine.
      m->_embed_w = to_f16(cfg.weight_prefix + cfg.model_seg +
                           "embed_tokens.weight");
      ok = !m->_embed_w.empty();
    } else {
      ok = qtri(cfg.weight_prefix + cfg.model_seg + "embed_tokens", m->_embed_w,
                m->_embed_s, m->_embed_b);
    }
  }
  m->_final_ln = to_f16(cfg.weight_prefix + cfg.model_seg + "norm.weight");
  if (m->_dense && fold_norm_plus_one) {           // +1 RMSNorm fold (raw-HF)
    add_one_f16(m->_final_ln);
  }
  if (!ok || m->_final_ln.empty()) {
    return nullptr;
  }
  if (!cfg.backbone_only) {
    m->_tied = cfg.tie_embeddings && lm_head_base.empty();
    // A dense (bf16) untied lm_head: has a plain `.weight` but no affine
    // `.scales`. Bind it via the f16 GEMV head when the logits path already
    // runs dense -- either the whole model is dense (_dense) or the bf16
    // embed/head sit atop an affine backbone (_dense_embed, the AWQ 27B: its
    // embed_tokens + lm_head are bf16 while the layers are w4). Without this
    // the untied bf16 head fell into the affine `qtri` branch below and failed
    // to bind (the AWQ-4bit load failure).
    const bool lm_dense = !lm_head_base.empty() &&
        wts->has(lm_head_base + ".weight") &&
        !wts->has(lm_head_base + ".scales");
    if (!m->_tied && lm_dense && (m->_dense || m->_dense_embed)) {
      // Dense f16 untied lm_head [vocab, H] (scales/biases empty).
      m->_lm_w = to_f16(lm_head_base + ".weight");
      if (m->_lm_w.empty()) { return nullptr; }
    } else if (!m->_tied && !m->_kquant) {
      if (m->_mixed) {
        m->_lm_bits = affine_bits(lm_head_base, cfg.hidden);
        if (m->_lm_bits != 4 && m->_lm_bits != 8) { m->_lm_bits = 4; }
      }
      ok = qtri(lm_head_base, m->_lm_w, m->_lm_s, m->_lm_b);
      if (!ok) { return nullptr; }
    }
    if (!m->_tied && m->_kquant) {
      // Untied k-quant lm_head: the GGUF's separate output.weight (raw Q6_K
      // on the 27B). The matvec reads it natively via lm_head_kq_() /
      // lm_head_kqt_(); when _tied these fall back to the embed table.
      if (!load_kq(lm_head_base + ".weight", m->_lm_q6k, m->_lm_kqt)) {
        return nullptr;
      }
    }
  }

  // Standard RoPE inv_freq over rotary_dim (base = rope_theta). Qwen3.5
  // has no rope scaling -- the partial rotary fraction is the only twist.
  const int half = cfg.rotary_dim / 2;
  m->_inv_freq = mc->make_shared_buffer((std::size_t)half * sizeof(float));
  auto* invf = static_cast<float*>(m->_inv_freq.contents());
  for (int i = 0; i < half; ++i) {
    invf[i] = 1.0f / std::pow(cfg.rope_theta,
                              (2.0f * (float)i) / (float)cfg.rotary_dim);
  }

  // Interleaved mROPE axis lookup over the rotary_dim/2 pairs (matches
  // mlx-vlm apply_interleaved_mrope): T everywhere, then H overrides
  // slots 1+3k (mrope_section[1] of them), W overrides 2+3k
  // (mrope_section[2]). Used only by the multimodal prefill.
  m->_mrope_axis.assign((std::size_t)half, 0);
  if (cfg.mrope_section.size() >= 3) {
    for (int k = 0; k < cfg.mrope_section[1]; ++k) {
      const int idx = 1 + 3 * k;
      if (idx < half) { m->_mrope_axis[(std::size_t)idx] = 1; }
    }
    for (int k = 0; k < cfg.mrope_section[2]; ++k) {
      const int idx = 2 + 3 * k;
      if (idx < half) { m->_mrope_axis[(std::size_t)idx] = 2; }
    }
  }

  // GDN q/k RMS-no-weight is rms_norm_f16 with a constant weight vector:
  // q uses inv_scale^2, k uses inv_scale, inv_scale = 1/sqrt(k_head_dim).
  const float inv_scale = 1.0f / std::sqrt((float)cfg.gdn_k_dim);
  m->_gdn_qscale = mc->make_shared_buffer((std::size_t)cfg.gdn_k_dim * 2);
  m->_gdn_kscale = mc->make_shared_buffer((std::size_t)cfg.gdn_k_dim * 2);
  auto* qsc = static_cast<std::uint16_t*>(m->_gdn_qscale.contents());
  auto* ksc = static_cast<std::uint16_t*>(m->_gdn_kscale.contents());
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  for (int i = 0; i < cfg.gdn_k_dim; ++i) {
    qsc[i] = store_elt(inv_scale * inv_scale);
    ksc[i] = store_elt(inv_scale);
  }

  if (!cfg.backbone_only) {
    if (m->_embed_is_q6k) {
      // Raw k-quant embed table: the 5-arg ctor picks the Q4_K / Q6_K gather.
      m->_muxer = std::make_unique<MetalTokenMuxer>(
          mc, &m->_embed_q6k, cfg.hidden, bf16,
          m->_embed_kqt == KQ::kQ4K);
    } else if (m->_dense || m->_dense_embed) {
      // Unquantized dense f16 embed table -> embed_gather_f16 (full precision;
      // _dense_embed keeps the MOSS bf16 embed even though the backbone is
      // affine).
      m->_muxer = std::make_unique<MetalTokenMuxer>(
          MetalTokenMuxer::DenseTag{}, mc, &m->_embed_w, cfg.hidden, bf16);
    } else {
      m->_muxer = std::make_unique<MetalTokenMuxer>(
          mc, &m->_embed_w, &m->_embed_s, &m->_embed_b, cfg.hidden, bf16,
          m->_embed_bits);
    }
    if (!m->_muxer->valid()) {
      return nullptr;
    }
  }

  // Shared ContextManager with its metal-compute backend (Spec::metal).
  // is_linear_layer marks the GDN layers (no paged K/V; conv/ssm state
  // instead); the SSM dims map the Qwen GDN config onto the generic
  // ssm_* spec fields.
  ContextManager::Spec cspec;
  cspec.metal = mc;
  cspec.n_layers = cfg.n_layers;
  cspec.n_kv_heads = cfg.n_kv_heads;
  cspec.head_dim = cfg.head_dim;
  cspec.max_seq = cfg.max_seq;
  cspec.page_tokens = cfg.page_tokens;
  cspec.max_pages = cfg.max_pages;
  cspec.is_linear_layer.assign((std::size_t)cfg.n_layers, false);
  for (int L = 0; L < cfg.n_layers; ++L) {
    if (!cfg.layer_is_full(L)) { cspec.is_linear_layer[(std::size_t)L] = true; }
  }
  cspec.ssm_conv_dim    = cfg.gdn_conv_dim;
  cspec.ssm_conv_kernel = cfg.gdn_conv_kernel;
  cspec.ssm_num_v_heads = cfg.gdn_v_heads;
  cspec.ssm_v_head_dim  = cfg.gdn_v_dim;
  cspec.ssm_k_head_dim  = cfg.gdn_k_dim;
  m->_ctx = std::make_unique<ContextManager>(cspec, nullptr);
  m->_cid = m->_ctx->acquire_root();
  if (!m->_cid.valid()) {
    return nullptr;
  }
  m->_pgtab = mc->make_shared_buffer(
      (std::size_t)m->_ctx->max_pages() * 3 * sizeof(std::int32_t));

  // ---- MTP head (mtp.safetensors), optional --------------------------
  // A bundled Multi-Token-Prediction drafter: the eh-fusion fc + pre-norms,
  // ONE full-attn decoder layer (4-bit affine, same shape as a main full
  // layer), and a final norm before the shared lm_head. Loaded if present;
  // its absence just leaves has_mtp() false. The head is read either from a
  // sibling mtp.safetensors (the OptiQ/MLX convention) or, when that's absent,
  // from the MAIN weights when they carry the mtp.* tensors in-shard (the
  // model-quantize output, mirroring the raw-HF dense layout). The 4-bit
  // helpers below bind to whichever handle `W` is. Embed/lm_head are shared
  // with the main model (already loaded).
  if (!m->_kquant && !m->_dense) {   // affine MTP (sibling file OR in-shard)
    const std::string mtp_path = model_dir + "/mtp.safetensors";
    auto mwts = MetalLlamaWeights::open(mtp_path);
    MetalLlamaWeights* Wp =
        (mwts && mwts->has("mtp.fc.weight")) ? &*mwts
      : (wts->has("mtp.fc.weight"))          ? &*wts
                                             : nullptr;
    if (Wp != nullptr) {
      auto& W = *Wp;
      // The MTP eh-fusion fc is a dense f16 GEMV; ensure the kernel is loaded
      // (it already is when _mixed/_kquant; a uniform-4-bit MTP model needs it
      // here). If unavailable the MTP head stays off (has_mtp() false).
      if (!m->_fn_dense_gemv.valid()) {
        m->_lib_dense = mc->load_library("dense_gemm" + sfx);
        m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
      }
      auto mto_elt = [&](const std::string& name) -> SharedBuffer {
        const auto* info = W.info(name);
        if (info == nullptr) { return {}; }
        const std::string want = bf16 ? "BF16" : "F16";
        if (info->dtype == want) { return W.load(name, mc); }
        SharedBuffer raw = W.load(name, mc);
        if (raw.empty()) { return {}; }
        const std::size_t n = numel_(info->shape);
        SharedBuffer out = mc->make_shared_buffer(n * 2);
        auto* o = static_cast<std::uint16_t*>(out.contents());
        auto to16 = [&](float f) -> std::uint16_t {
          if (bf16) { return f32_to_bf16_(f); }
          _Float16 h = (_Float16)f; std::uint16_t b;
          std::memcpy(&b, &h, 2); return b;
        };
        if (info->dtype == "BF16") {
          const auto* s = static_cast<const std::uint16_t*>(raw.contents());
          for (std::size_t i = 0; i < n; ++i) { o[i] = to16(bf16_to_f32_(s[i])); }
        } else if (info->dtype == "F32") {
          const auto* s = static_cast<const float*>(raw.contents());
          for (std::size_t i = 0; i < n; ++i) { o[i] = to16(s[i]); }
        } else { return {}; }
        return out;
      };
      // RMSNorm weight loaded with the +1 shift. The base Qwen3.5 stores norm
      // weights as (w-1) and adds 1 at runtime; mlx-optiq baked the +1 into the
      // MAIN model's norms during conversion (they read back as true weights),
      // but the MTP head's norms are stripped by mlx-lm so they were never
      // shifted -- they're still raw (w-1) here and need the +1 applied.
      auto mto_norm = [&](const std::string& name) -> SharedBuffer {
        SharedBuffer buf = mto_elt(name);
        if (buf.empty()) { return buf; }
        const std::size_t n = buf.byte_size() / 2;
        auto* o = static_cast<std::uint16_t*>(buf.contents());
        for (std::size_t i = 0; i < n; ++i) {
          if (bf16) {
            o[i] = f32_to_bf16_(bf16_to_f32_(o[i]) + 1.0f);
          } else {
            _Float16 h; std::memcpy(&h, &o[i], 2);
            h = (_Float16)((float)h + 1.0f); std::memcpy(&o[i], &h, 2);
          }
        }
        return buf;
      };
      // 4-bit affine triple (raw u32 weight + f16 scales/biases).
      auto mqtri = [&](const std::string& pfx, SharedBuffer& w,
                       SharedBuffer& s, SharedBuffer& b) -> bool {
        w = W.load(pfx + ".weight", mc);
        s = mto_elt(pfx + ".scales");
        b = mto_elt(pfx + ".biases");
        return !w.empty() && !s.empty() && !b.empty();
      };
      const int Hh = cfg.hidden, Fc = cfg.ffn_inner;
      // MTP layer bit-width: usually matches the backbone, but infer it from
      // the packed q_proj (weight is [N, Hh*bits/32]) so an 8-bit checkpoint's
      // 8-bit MTP head is sized correctly. Hardcoding 4-bit here undersized the
      // fused qkv / gate-up buffers by 2x for a w8 head -> heap overflow +
      // segfault at load (the 27B-8bit crash).
      int mtp_bits = cfg.quant_bits;
      if (const auto* qi = W.info("mtp.layers.0.self_attn.q_proj.weight");
          qi != nullptr && qi->shape.size() >= 2 && Hh > 0) {
        mtp_bits = (int)((qi->shape.back() * 32) / Hh);
      }
      if (mtp_bits != 4 && mtp_bits != 8) { mtp_bits = 4; }
      const std::size_t wrow = (std::size_t)Hh * mtp_bits / 32, grow = Hh / 64;
      // Row-concat q|k|v into one [2*qd+2*kd, H], like the main fuse_q.
      auto mfuse = [&](const std::vector<std::string>& names, SharedBuffer& fw,
                       SharedBuffer& fs, SharedBuffer& fb) -> bool {
        std::vector<SharedBuffer> ws, ss, bs; std::size_t Nt = 0;
        for (const auto& nm : names) {
          SharedBuffer w, s, b;
          if (!mqtri(nm, w, s, b)) { return false; }
          const auto* in = W.info(nm + ".scales");
          if (in == nullptr) { return false; }
          Nt += (std::size_t)in->shape[0];
          ws.push_back(std::move(w)); ss.push_back(std::move(s));
          bs.push_back(std::move(b));
        }
        fw = mc->make_shared_buffer(Nt * wrow * 4);
        fs = mc->make_shared_buffer(Nt * grow * 2);
        fb = mc->make_shared_buffer(Nt * grow * 2);
        if (fw.empty() || fs.empty() || fb.empty()) { return false; }
        std::size_t wo = 0, go = 0;
        for (std::size_t i = 0; i < ws.size(); ++i) {
          std::memcpy((char*)fw.contents() + wo, ws[i].contents(),
                      ws[i].byte_size());
          std::memcpy((char*)fs.contents() + go, ss[i].contents(),
                      ss[i].byte_size());
          std::memcpy((char*)fb.contents() + go, bs[i].contents(),
                      bs[i].byte_size());
          wo += ws[i].byte_size(); go += ss[i].byte_size();
        }
        return true;
      };
      // Interleave gate|up (row 2g=gate, 2g+1=up) -> [2*ffn, H] (4-bit).
      auto minter = [&](const SharedBuffer& gw, const SharedBuffer& gs,
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
      MtpHead& M = m->_mtp;
      Layer& ly = M.lyr;
      ly.is_full = true;
      // The head is uniform mtp_bits (fused qkv / interleaved gate-up / o /
      // down all packed at wrow above); record it so the forward dispatches
      // the matching-width kernel instead of a hardcoded 4-bit.
      ly.q_bits = ly.k_bits = ly.v_bits = ly.o_bits = mtp_bits;
      ly.gate_bits = ly.up_bits = ly.down_bits = mtp_bits;
      bool ok = true;
      // fc is [H, 2H] dense; split each output row into its embedding half
      // (cols 0:H) and hidden half (cols H:2H), both contiguous [H, H].
      {
        SharedBuffer fcw = mto_elt("mtp.fc.weight");
        if (!fcw.empty()) {
          M.fc_e = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
          M.fc_h = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
          const auto* src = static_cast<const std::uint16_t*>(fcw.contents());
          auto* de = static_cast<std::uint16_t*>(M.fc_e.contents());
          auto* dh = static_cast<std::uint16_t*>(M.fc_h.contents());
          for (int i = 0; i < Hh; ++i) {
            std::memcpy(de + (std::size_t)i * Hh,
                        src + (std::size_t)i * 2 * Hh, (std::size_t)Hh * 2);
            std::memcpy(dh + (std::size_t)i * Hh,
                        src + (std::size_t)i * 2 * Hh + Hh, (std::size_t)Hh * 2);
          }
        }
      }
      M.prenorm_e = mto_norm("mtp.pre_fc_norm_embedding.weight");
      M.prenorm_h = mto_norm("mtp.pre_fc_norm_hidden.weight");
      M.final_norm = mto_norm("mtp.norm.weight");
      ly.in_ln = mto_norm("mtp.layers.0.input_layernorm.weight");
      ly.post_ln = mto_norm("mtp.layers.0.post_attention_layernorm.weight");
      ly.q_norm = mto_norm("mtp.layers.0.self_attn.q_norm.weight");
      ly.k_norm = mto_norm("mtp.layers.0.self_attn.k_norm.weight");
      ok = ok && !M.fc_e.empty() && !M.fc_h.empty() && !M.prenorm_e.empty() &&
           !M.prenorm_h.empty() && !M.final_norm.empty() &&
           !ly.in_ln.empty() && !ly.post_ln.empty() &&
           !ly.q_norm.empty() && !ly.k_norm.empty();
      ok = ok && mfuse({"mtp.layers.0.self_attn.q_proj",
                        "mtp.layers.0.self_attn.k_proj",
                        "mtp.layers.0.self_attn.v_proj"},
                       ly.qw, ly.qs, ly.qb);
      ok = ok && mqtri("mtp.layers.0.self_attn.o_proj", ly.ow, ly.os, ly.ob);
      {
        SharedBuffer gw, gs, gb, uw, us, ub;
        ok = ok && mqtri("mtp.layers.0.mlp.gate_proj", gw, gs, gb);
        ok = ok && mqtri("mtp.layers.0.mlp.up_proj", uw, us, ub);
        // Interleave gate|up for the 4-bit fused swiglu; an 8-bit head has no
        // fused w8 swiglu kernel, so keep gate/up de-fused (own w8 GEMVs +
        // plain swiglu in the forward), mirroring the backbone MLP.
        if (mtp_bits == 4) {
          ok = ok && minter(gw, gs, gb, uw, us, ub, ly.guw, ly.gus, ly.gub);
          ly.mlp_fused = true;
        } else {
          ly.guw = std::move(gw); ly.gus = std::move(gs);
          ly.gub = std::move(gb);
          ly.uw = std::move(uw); ly.us = std::move(us); ly.ub = std::move(ub);
          ly.mlp_fused = false;
        }
        ok = ok && mqtri("mtp.layers.0.mlp.down_proj", ly.dw, ly.ds, ly.db);
      }
      ok = ok && m->_fn_dense_gemv.valid();
      if (ok) {
        // The MTP layer's own 1-layer full-attn paged context. ONE large page
        // so the persistent path (mtp_persist) can keep the draft head's KV
        // over the committed decode history without straddling pages (the
        // kv-write/sdpa here are single-page); a long decode that overflows
        // resets the ctx and continues (graceful loss of older history).
        ContextManager::Spec ms;
        ms.metal = mc;
        ms.n_layers = 1;
        ms.n_kv_heads = cfg.n_kv_heads;
        ms.head_dim = cfg.head_dim;
        ms.max_seq = 2048;
        ms.page_tokens = 2048;
        ms.max_pages = 0;
        ms.is_linear_layer.assign(1, false);
        M.ctx = std::make_unique<ContextManager>(ms, nullptr);
        M.cid = M.ctx->acquire_root();
        M.ok = M.cid.valid();
        if (M.ok) {
          m->_mtp_h = mc->make_shared_buffer((std::size_t)cfg.hidden * 2);
          M.pgt = mc->make_shared_buffer(
              (std::size_t)M.ctx->max_pages() * 3 * sizeof(std::int32_t));
        }
      }
    }
  }

  // ---- MTP head (GGUF NextN block), optional -------------------------
  // The k-quant twin of the affine block above. The bundled NextN draft layer
  // lives in the SAME GGUF (exposed by the converter under the mtp.* names) as
  // native k-quant / q8_0 payloads, so it loads with the main k-quant helpers
  // off `wts` (not a sibling file). GGUF norms are TRUE weights (the +1 is
  // baked at conversion), so they load via plain to_f16 -- no (w-1)+1 shift
  // like the stripped mtp.safetensors norms. The eh-fusion fc arrives as f32
  // (q8_0 dequant), narrowed to f16 and split into its embedding/hidden halves.
  if (m->_kquant && wts->has("mtp.fc.weight")) {
    MtpHead& M = m->_mtp;
    Layer& ly = M.lyr;
    ly.is_full = true;
    const int Hh = cfg.hidden;
    bool ok = true;
    {
      SharedBuffer fcw = to_f16("mtp.fc.weight");   // [H, 2H] f16
      if (!fcw.empty()) {
        M.fc_e = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
        M.fc_h = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
        const auto* src = static_cast<const std::uint16_t*>(fcw.contents());
        auto* de = static_cast<std::uint16_t*>(M.fc_e.contents());
        auto* dh = static_cast<std::uint16_t*>(M.fc_h.contents());
        for (int i = 0; i < Hh; ++i) {
          std::memcpy(de + (std::size_t)i * Hh,
                      src + (std::size_t)i * 2 * Hh, (std::size_t)Hh * 2);
          std::memcpy(dh + (std::size_t)i * Hh,
                      src + (std::size_t)i * 2 * Hh + Hh, (std::size_t)Hh * 2);
        }
      }
    }
    M.prenorm_e = to_f16("mtp.pre_fc_norm_embedding.weight");
    M.prenorm_h = to_f16("mtp.pre_fc_norm_hidden.weight");
    M.final_norm = to_f16("mtp.norm.weight");
    ly.in_ln = to_f16("mtp.layers.0.input_layernorm.weight");
    ly.post_ln = to_f16("mtp.layers.0.post_attention_layernorm.weight");
    ly.q_norm = to_f16("mtp.layers.0.self_attn.q_norm.weight");
    ly.k_norm = to_f16("mtp.layers.0.self_attn.k_norm.weight");
    ok = ok && !M.fc_e.empty() && !M.fc_h.empty() && !M.prenorm_e.empty() &&
         !M.prenorm_h.empty() && !M.final_norm.empty() &&
         !ly.in_ln.empty() && !ly.post_ln.empty() &&
         !ly.q_norm.empty() && !ly.k_norm.empty();
    // q+k fuse (both q4_K); v / o / mlp parts stay separate raw k-quant.
    ok = ok && fuse_kq("mtp.layers.0.self_attn.q_proj.weight",
                       "mtp.layers.0.self_attn.k_proj.weight",
                       ly.kqk, ly.kqk_t, ly.kqk_n);
    ok = ok && load_kq("mtp.layers.0.self_attn.v_proj.weight",
                       ly.kqv, ly.kqv_t);
    ok = ok && load_kq("mtp.layers.0.self_attn.o_proj.weight",
                       ly.kqo, ly.kqo_t);
    ok = ok && load_kq("mtp.layers.0.mlp.gate_proj.weight",
                       ly.kqgate, ly.kqgate_t);
    ok = ok && load_kq("mtp.layers.0.mlp.up_proj.weight", ly.kqup, ly.kqup_t);
    ok = ok && load_kq("mtp.layers.0.mlp.down_proj.weight",
                       ly.kqdown, ly.kqdown_t);
    if (ok) {
      // The MTP layer's own 1-layer full-attn paged context. ONE large page so
      // the persistent path keeps the draft head's KV over the committed decode
      // history (single-page kv-write/sdpa); overflow resets + continues.
      ContextManager::Spec ms;
      ms.metal = mc;
      ms.n_layers = 1;
      ms.n_kv_heads = cfg.n_kv_heads;
      ms.head_dim = cfg.head_dim;
      ms.max_seq = 2048;
      ms.page_tokens = 2048;
      ms.max_pages = 0;
      ms.is_linear_layer.assign(1, false);
      M.ctx = std::make_unique<ContextManager>(ms, nullptr);
      M.cid = M.ctx->acquire_root();
      M.ok = M.cid.valid();
      if (M.ok) {
        m->_mtp_h = mc->make_shared_buffer((std::size_t)cfg.hidden * 2);
        M.pgt = mc->make_shared_buffer(
            (std::size_t)M.ctx->max_pages() * 3 * sizeof(std::int32_t));
      }
    }
  }

  // ---- MTP head (raw-HF dense bf16/f16), optional --------------------
  // The dense twin of the two blocks above. The raw-HF Qwen3.6 checkpoint
  // carries the MTP head's mtp.* tensors in the MAIN safetensors (not a sibling
  // file), all bf16/f16 with NO scales, so they load with the main to_f16 off
  // `wts`. Layout mirrors the backbone dense load: q|k|v fused into ly.qw,
  // o_proj plain, gate/up de-fused (guw/uw) + down, empty scales/biases ->
  // the mtp_step forward dispatches the dense GEMV/GEMM on s.empty(). fc is
  // f16 [H,2H], split into embedding/hidden halves. ALL the MTP RMSNorm
  // weights are Qwen3_5RMSNorm zero-centered -> +1 fold (add_one_f16); the
  // MTP layer has no gated norm, so every norm here gets the +1.
  if (m->_dense && wts->has("mtp.fc.weight")) {
    MtpHead& M = m->_mtp;
    Layer& ly = M.lyr;
    ly.is_full = true;
    const int Hh = cfg.hidden;
    bool ok = true;
    {
      SharedBuffer fcw = to_f16("mtp.fc.weight");   // [H, 2H] f16
      if (!fcw.empty()) {
        M.fc_e = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
        M.fc_h = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
        const auto* src = static_cast<const std::uint16_t*>(fcw.contents());
        auto* de = static_cast<std::uint16_t*>(M.fc_e.contents());
        auto* dh = static_cast<std::uint16_t*>(M.fc_h.contents());
        for (int i = 0; i < Hh; ++i) {
          std::memcpy(de + (std::size_t)i * Hh,
                      src + (std::size_t)i * 2 * Hh, (std::size_t)Hh * 2);
          std::memcpy(dh + (std::size_t)i * Hh,
                      src + (std::size_t)i * 2 * Hh + Hh, (std::size_t)Hh * 2);
        }
      }
    }
    // All MTP norms are zero-centered raw-HF -> +1 fold (no gated norm here).
    auto mnorm = [&](const char* name) -> SharedBuffer {
      SharedBuffer b = to_f16(name);
      add_one_f16(b);
      return b;
    };
    M.prenorm_e = mnorm("mtp.pre_fc_norm_embedding.weight");
    M.prenorm_h = mnorm("mtp.pre_fc_norm_hidden.weight");
    M.final_norm = mnorm("mtp.norm.weight");
    ly.in_ln = mnorm("mtp.layers.0.input_layernorm.weight");
    ly.post_ln = mnorm("mtp.layers.0.post_attention_layernorm.weight");
    ly.q_norm = mnorm("mtp.layers.0.self_attn.q_norm.weight");
    ly.k_norm = mnorm("mtp.layers.0.self_attn.k_norm.weight");
    ok = ok && !M.fc_e.empty() && !M.fc_h.empty() && !M.prenorm_e.empty() &&
         !M.prenorm_h.empty() && !M.final_norm.empty() &&
         !ly.in_ln.empty() && !ly.post_ln.empty() &&
         !ly.q_norm.empty() && !ly.k_norm.empty();
    // Dense f16: fuse q|k|v into ONE [2*qd+2*kd, H] weight (q gated), o_proj
    // plain, gate/up de-fused, down -- all empty scales/biases.
    ok = ok && fuse_f16({"mtp.layers.0.self_attn.q_proj",
                         "mtp.layers.0.self_attn.k_proj",
                         "mtp.layers.0.self_attn.v_proj"}, ly.qw);
    ly.ow = to_f16("mtp.layers.0.self_attn.o_proj.weight");
    ok = ok && !ly.ow.empty();
    ok = ok && !(ly.guw = to_f16("mtp.layers.0.mlp.gate_proj.weight")).empty();
    ok = ok && !(ly.uw = to_f16("mtp.layers.0.mlp.up_proj.weight")).empty();
    ok = ok && !(ly.dw = to_f16("mtp.layers.0.mlp.down_proj.weight")).empty();
    ok = ok && m->_fn_dense_gemv.valid();
    if (ok) {
      // The MTP layer's own 1-layer full-attn paged context (ONE large page;
      // same as the affine/GGUF blocks above).
      ContextManager::Spec ms;
      ms.metal = mc;
      ms.n_layers = 1;
      ms.n_kv_heads = cfg.n_kv_heads;
      ms.head_dim = cfg.head_dim;
      ms.max_seq = 2048;
      ms.page_tokens = 2048;
      ms.max_pages = 0;
      ms.is_linear_layer.assign(1, false);
      M.ctx = std::make_unique<ContextManager>(ms, nullptr);
      M.cid = M.ctx->acquire_root();
      M.ok = M.cid.valid();
      if (M.ok) {
        m->_mtp_h = mc->make_shared_buffer((std::size_t)cfg.hidden * 2);
        M.pgt = mc->make_shared_buffer(
            (std::size_t)M.ctx->max_pages() * 3 * sizeof(std::int32_t));
      }
    }
  }

  // Optional 4-bit DRAFT lm_head for the MTP draft (affine path, 8-bit tied
  // embed source): requant the 8-bit head to 4-bit ONCE here so the draft's
  // vocab GEMV reads half the bytes (the verifier keeps the 8-bit head). The
  // draft is verify-corrected -> output stays exact; the lossy 4-bit quant
  // costs a little draft acceptance. VPIPE_MTP_NO_DRAFT_HEAD disables.
  if (m->_mtp.ok && m->_mixed && m->_fn_requant_w8w4.valid() &&
      std::getenv("VPIPE_MTP_NO_DRAFT_HEAD") == nullptr) {
    const bool tied = m->_tied;
    const int src_bits = tied ? m->_embed_bits : m->_lm_bits;
    const SharedBuffer& sw = tied ? m->_embed_w : m->_lm_w;
    const SharedBuffer& ss = tied ? m->_embed_s : m->_lm_s;
    const SharedBuffer& sb = tied ? m->_embed_b : m->_lm_b;
    const int V = cfg.vocab, H = cfg.hidden;
    if (src_bits == 8 && !sw.empty() && (H % 64) == 0) {
      MtpHead& M = m->_mtp;
      M.dlm_w = mc->make_shared_buffer((std::size_t)V * (H / 8) * 4);
      M.dlm_s = mc->make_shared_buffer((std::size_t)V * (H / 64) * 2);
      M.dlm_b = mc->make_shared_buffer((std::size_t)V * (H / 64) * 2);
      if (!M.dlm_w.empty() && !M.dlm_s.empty() && !M.dlm_b.empty()) {
        metal_compute::CommandStream stream = mc->make_command_stream();
        ComputeEncoder enc = stream.begin_compute();
        enc.set_function(m->_fn_requant_w8w4);
        enc.set_buffer(0, sw); enc.set_buffer(1, ss); enc.set_buffer(2, sb);
        enc.set_buffer(3, M.dlm_w); enc.set_buffer(4, M.dlm_s);
        enc.set_buffer(5, M.dlm_b);
        enc.set_constant(6, H); enc.set_constant(7, V);
        enc.dispatch({(unsigned)(H / 64), (unsigned)V, 1}, {8, 32, 1});
        enc.end();
        stream.commit().wait();
        M.draft_head = true;
      }
    }
  }
  // Dynamic-int8 accelerated prefill GEMMs: default OFF; the stage opts
  // in via set_i8_gemm (exec) and VPIPE_I8_GEMM=1 force-enables for
  // benches/A-B (the context handles the env override internally).
  m->set_i8_gemm(false);
  return m;
}

MetalQwenModel::Config
MetalQwenModel::config_from(const ModelConfig& c)
{
  Config m;
  m.n_layers   = c.n_layers;
  m.hidden     = c.hidden;
  m.n_heads    = c.n_heads;
  m.n_kv_heads = c.n_kv_heads;
  m.head_dim   = c.head_dim;
  m.ffn_inner  = c.ffn_inner;
  m.vocab      = c.vocab_size;
  m.rope_theta = c.rope_theta;
  m.rms_eps    = c.rms_eps;
  m.rotary_dim =
      static_cast<int>(c.head_dim * c.partial_rotary_factor);
  m.tie_embeddings = c.tie_word_embeddings;
  if (c.mrope_section.size() >= 3) {
    m.mrope_section = c.mrope_section;
  }
  // Dense = no linear-attention layers (Qwen3-ASR text decoder).
  bool any_linear = false;
  for (bool b : c.is_linear_layer) { any_linear = any_linear || b; }
  m.dense = !any_linear;
  m.attn_output_gate = c.attn_output_gate;
  // Mixture-of-Experts (Qwen3.5-MoE): every layer's MLP becomes a
  // SparseMoeBlock. Backbone (GDN+full-attn) is unchanged.
  m.n_experts        = c.num_experts;
  m.top_k            = c.num_experts_per_tok;
  m.moe_inner        = c.moe_intermediate_size;
  m.moe_shared_inner = c.shared_expert_inter;
  m.moe_norm_topk    = c.norm_topk_prob;
  m.quant_bits = c.quantization.bits > 0 ? c.quantization.bits : 4;
  m.weight_prefix =
      (c.architecture == "Qwen3ASRForConditionalGeneration")
          ? "" : "language_model.";
  m.full_attn_interval =
      c.full_attention_interval > 0 ? c.full_attention_interval : 4;
  if (!m.dense) {
    m.gdn_conv_kernel = c.linear_conv_kernel;
    m.gdn_k_heads = c.linear_num_k_heads;
    m.gdn_v_heads = c.linear_num_v_heads;
    m.gdn_k_dim   = c.linear_k_head_dim;
    m.gdn_v_dim   = c.linear_v_head_dim;
    m.gdn_conv_dim =
        2 * c.linear_num_k_heads * c.linear_k_head_dim +
        c.linear_num_v_heads * c.linear_v_head_dim;
  }
  // Per-context logical length cap. The KV page pool grows lazily up to
  // a cap derived from this (max_pages = ceil(max_seq/page_tokens) * 16),
  // so a generous value costs only page-table bytes -- no KV memory until
  // the tokens are actually appended. 2048 was too small for streaming
  // video QA: realtime-vqa accumulates one ~160-256-token grid per
  // (fused) frame up to its max_frames_per_scene safety cap (default 64),
  // so the multimodal base prefill could need ~10k+ tokens. Overflowing
  // the cap made append() return an invalid slot -> empty logits ->
  // prefill returned -1 (every scene failed). 16384 holds 64 grids at up
  // to 256 tokens each plus the describe decode, so the stage's own frame
  // cap -- not the LM context -- is the binding limit; text / audio
  // decoders never approach it. This mirrors the MLX path, whose context
  // manager grows on demand with no fixed per-context cap.
  m.max_seq    = 16384;
  return m;
}

// Hybrid dispatch (DEFAULT ON; VPIPE_QWEN_NO_HYBRID=1 disables for A/B): keep the
// decode encoder SERIAL (cheap native ordering, no explicit barriers) but wrap
// each de-fused sibling group (unfused q|k|v, GDN in_proj qkv|z|a|b, de-fused
// gate|up -- in the mixed-precision OptiQ and k-quant/GGUF paths) in a short
// CONCURRENT sub-encoder so those independent GEMVs overlap. Cross-group ordering
// rides Metal's automatic hazard tracking across the encoder boundaries (Tracked
// buffers). Token-exact; measured +2-3% on the de-fused decode paths.
static bool decode_hybrid_()
{
  static const bool hybrid = std::getenv("VPIPE_QWEN_NO_HYBRID") == nullptr;
  return hybrid;
}

// Batched-decode per-branch concurrency (DEFAULT ON; VPIPE_QWEN_NO_BATCH_
// CONCURRENT=1 disables for A/B): the N branches' per-stream attention
// (shared-prefix phase-B merge) and GDN recurrence write DISJOINT output slices
// and use their OWN KV / conv / ssm state, so across branches they are
// independent. Dispatch them into concurrent sub-encoders (attention: one scope
// over the N branch merges; GDN: phase-interleaved -- all conv1d+g_beta, then
// all qk_norm, then all ssm, then all gated_rms, each phase a concurrent scope)
// so the N branches overlap instead of serializing. Only helps N>1. Measured
// aggregate batched decode: +2.5-3% @N=4 (typical realtime-vqa), growing to
// ~+8% @N=32 (more branches -> more independent work to overlap). Token-exact.
static bool decode_batch_concurrent_()
{
  static const bool on =
      std::getenv("VPIPE_QWEN_NO_BATCH_CONCURRENT") == nullptr;
  return on;
}


// No-hslice GDN views (DEFAULT ON; VPIPE_QWEN_HSLICE_COPY=1 restores the copy
// path for A/B): skip the 3 GDN hslice COPY dispatches that materialize z/a/b out
// of the contiguous _d_mixqkv [qkv|z|a|b] buffer; instead bind _d_mixqkv directly
// at the right byte offset to g_beta (a,b) and gated_rms (z) -- a free "view" the
// way MLX's mx.split works. Removes 72 launch-bound copies/token; token-exact.
// Applies to every Qwen3.x variant (all share this GDN recurrence block).
static bool decode_no_hslice_()
{
  static const bool on = std::getenv("VPIPE_QWEN_HSLICE_COPY") == nullptr;
  return on;
}

// GDN recurrence overlap (DEFAULT ON; VPIPE_QWEN_NO_GDN_OVERLAP=1 disables for
// A/B): in the GDN recurrence block the g_beta dispatch (reads the a|b slices of
// the in_proj output) is independent of the conv1d -> qk_norm sub-chain, yet the
// serial order conv1d -> qk_norm -> g_beta -> gdn_step puts g_beta's launch
// latency fully on the critical path. Issue {conv1d, g_beta} in one CONCURRENT
// sub-encoder so g_beta overlaps conv1d (and rides the encoder boundary into the
// serial qk_norm -> gdn_step tail). Removes one dispatch from the critical path
// per GDN layer (x24). Token-exact; measured a small consistent win (+0.3-0.4%,
// strongest at short ctx where the context-independent GDN recurrence is a larger
// fraction of the step). Rides decode_hybrid_ (no-op if hybrid dispatch is off).
static bool decode_gdn_overlap_()
{
  static const bool on = std::getenv("VPIPE_QWEN_NO_GDN_OVERLAP") == nullptr;
  return on && decode_hybrid_();
}

// No-hslice full-attention K/V views (DEFAULT ON; VPIPE_QWEN_ATTN_HSLICE_COPY=1
// restores the copy for A/B): the fused qkv matvec writes [q|k|v] (or
// [q|gate|...|k|v]) into _d_qfull; k and v are the CONTIGUOUS tail (byte offsets
// qdo*2, (qdo+kd)*2), so bind _d_qfull directly there for the k-path
// (qk_norm+rope in place) and the K/V-cache write instead of COPYING them out
// first. Removes 2 hslice copies per full-attn layer (x8). q/gate always stay
// copied -- Qwen3.5's output gate interleaves them ([q|gate] per head), a
// strided de-interleave not expressible as an offset view. Token-exact; a small
// win at short ctx, neutral at long ctx (fewer copies, never a regression).
static bool decode_attn_nohs_()
{
  static const bool on = std::getenv("VPIPE_QWEN_ATTN_HSLICE_COPY") == nullptr;
  return on;
}

bool
MetalQwenModel::ensure_decode_scratch_()
{
  if (_dec_ready) { return true; }
  const Config& c = _cfg;
  const int H = c.hidden, qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, vald = c.value_dim(), Hv = c.gdn_v_heads;
  auto b = [&](std::size_t elems) {
    return _mc->make_shared_buffer(elems * 2);
  };
  _d_x = b((std::size_t)H);
  _d_hn = b((std::size_t)H);
  _d_logits = b((std::size_t)c.vocab);
  _d_qfull = b((std::size_t)(2 * qd + 2 * kd));
  _d_kbuf = b((std::size_t)kd);
  _d_vbuf = b((std::size_t)kd);
  _d_q3 = b((std::size_t)qd);
  _d_gate3 = b((std::size_t)qd);
  _d_attn = b((std::size_t)qd);
  _d_mixqkv = b((std::size_t)(Cd + vald + 2 * Hv));
  _d_zbuf = b((std::size_t)vald);
  _d_abuf = b((std::size_t)Hv);
  _d_bbuf = b((std::size_t)Hv);
  _d_convout = b((std::size_t)Cd);
  _d_gbuf = _mc->make_shared_buffer((std::size_t)Hv * 4);
  _d_betabuf = _mc->make_shared_buffer((std::size_t)Hv * 4);
  _d_ygdn = b((std::size_t)vald);
  _d_normout = b((std::size_t)vald);
  _d_sg = b((std::size_t)c.ffn_inner);
  if (_kquant || _mixed || _dense) {
    // k-quant / mixed-affine / dense decode temps: [H] for the GEMV-then-
    // residual-add (the dense/k-quant path has no fused-add GEMV), [ffn]
    // up-proj temp (gate/up run as two GEMVs + SwiGLU, no fused-swiglu).
    _d_radd = b((std::size_t)H);
    _d_up = b((std::size_t)c.ffn_inner);
  }
  if (c.is_moe()) {
    // MoE decode scratch (ffn_inner is 0 for MoE; the shared expert reuses
    // _d_moe_ssg sized to moe_shared_inner). Router/experts run on-GPU.
    const int E = c.n_experts, K = c.top_k, I = c.moe_inner;
    auto i32 = [&](std::size_t e) {
      return _mc->make_shared_buffer(e * sizeof(std::int32_t));
    };
    _d_moe_logits = b((std::size_t)E);
    _d_moe_ids  = i32((std::size_t)K);
    _d_moe_w    = b((std::size_t)K);
    _d_moe_act  = b((std::size_t)K * I);
    _d_moe_part = b((std::size_t)K * H);
    _d_moe_out  = b((std::size_t)H);
    _d_moe_ssg  = b((std::size_t)c.moe_shared_inner);
    _d_moe_sout = b((std::size_t)H);
    _d_moe_gate = b((std::size_t)1);
  }
  _d_tok_in = _mc->make_shared_buffer(sizeof(std::int32_t));
  _d_argmax_id = _mc->make_shared_buffer(sizeof(std::int32_t));
  // Flash-decode-GQA partials (f32): O [Hq,split,D], m/l [Hq,split]. Sized for
  // the MAX split (256) so the context-adaptive split (gqa_split_for) can grow
  // with depth without reallocating.
  if (_gqa_attn) {
    const std::size_t sp = 256;
    const std::size_t Hq = (std::size_t)c.n_heads, Dd = (std::size_t)c.head_dim;
    _d_gqa_oacc = _mc->make_shared_buffer(Hq * sp * Dd * sizeof(float));
    _d_gqa_m = _mc->make_shared_buffer(Hq * sp * sizeof(float));
    _d_gqa_l = _mc->make_shared_buffer(Hq * sp * sizeof(float));
    // Let the decode GQA attention set size its own scratch + autotune
    // kernel+split for this GPU; debug-log the resolved choice + a tuning-time
    // summary (more sets append to the same report).
    TuningReport tuning;
    const DecodeGqaAttnSet::Dims dd{c.head_dim, c.n_heads, c.n_kv_heads};
    _decode_set.prepare(_mc, dd, tuning);
    _prefill_set.prepare(_mc, {c.head_dim, c.n_heads, c.n_kv_heads}, tuning);
    // MoE grouped-GEMM GEMV->steel crossover. Left as a tunable member (default
    // 1024); the per-machine probe is deferred to an M5 session where matrix
    // cores make the win directly measurable. VPIPE_QWEN_MOE_STEEL_MIN overrides.
    if (c.is_moe()) {
      if (const char* e = std::getenv("VPIPE_QWEN_MOE_STEEL_MIN")) {
        _moe_steel_min = std::max(1, std::atoi(e));
      }
    }
    log_decode_attn_choice_();
    const SessionContextIntf* s = _mc ? _mc->session() : nullptr;
    if (s != nullptr && !tuning.empty()) {
      s->log_debug(fmt("[qwen] load-time kernel tuning {}ms: {}",
          (int)(tuning.total_ms() + 0.5), tuning.summary()));
    }
  }
  // Per-machine batched-GEMV ladder probe (one-shot; also triggered by
  // ensure_bscratch_ for callers that batch without a serial decode).
  autotune_qmv_ladder_();
  _dec_ready = !_d_logits.empty();
  return _dec_ready;
}

std::int32_t
MetalQwenModel::decode_step_fast(ContextId cid, std::int32_t token_id,
                                 int rope_pos)
{
  const Config& c = _cfg;
  const int H = c.hidden;
  // Requires the in-stream embed gather + on-GPU argmax kernels. Checked
  // BEFORE any state mutation so the caller's fallback re-runs cleanly.
  // k-quant gathers from the raw Q6_K table (embed_q6k_); affine uses the
  // muxer's dequant gather. Both fold into the decode command buffer so the
  // token never round-trips to host (vs forward()'s separate embed submit).
  const bool embed_ok = _kquant ? _embed_is_q6k
                       : ((_dense || _dense_embed) ? _fn_embed_dense.valid()
                                                   : _fn_embed.valid());
  if (!embed_ok || !_fn_argmax.valid()) {
    return std::numeric_limits<std::int32_t>::min();
  }
  if (!ensure_decode_scratch_() || _d_tok_in.empty() ||
      _d_argmax_id.empty()) {
    return std::numeric_limits<std::int32_t>::min();
  }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return -1; }
  const int pos = slot.position;
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  *static_cast<std::int32_t*>(_d_tok_in.contents()) = token_id;

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // In-stream embed gather (token id -> _d_x), folded into the decode
    // command buffer (vs forward()'s separate muxer command buffer +
    // host memcpy). Same table as the muxer -> identical embedding.
    if (_kquant) {
      embed_q6k_(enc, _d_tok_in, 0, _d_x, 1);
    } else if (_dense || _dense_embed) {
      // Dense f16 embed gather: embed_gather_f16(ids, table, out, H).
      enc.set_function(_fn_embed_dense);
      enc.set_buffer(0, _d_tok_in);
      enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, _d_x);
      enc.set_constant(3, H);
      enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
    } else {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, _d_tok_in);
      enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, _embed_s);
      enc.set_buffer(3, _embed_b);
      enc.set_buffer(4, _d_x);
      enc.set_constant(5, H);
      enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
    }

    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot,
                        _pgtab, 0);

    // On-GPU greedy argmax -> _d_argmax_id (1 int): no full [vocab] host
    // logit pull + host scan. Two-stage parallel argmax (lowest-index tie-
    // break), matching the synchronous host argmax exactly.
    encode_argmax_(enc, _d_logits, 0, _d_argmax_id, 0, c.vocab);
  }
  stream.commit().wait();
  return *static_cast<const std::int32_t*>(_d_argmax_id.contents());
}

// ---------------------------------------------------------------------
// Native k-quant (GGUF) matmul dispatch.
// ---------------------------------------------------------------------
void
MetalQwenModel::kqmv_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                      const SharedBuffer& x, std::size_t xoff,
                      const SharedBuffer& y, std::size_t yoff, int K, int N)
{
  // Defensive: kNone must never reach dispatch (load_kq aborts the load on an
  // unsupported/absent tensor). The kernel-selection else below maps any
  // non-Q6K/Q5K tag to the Q4_K kernel, so a kNone would silently mis-decode.
  assert(type != KQ::kNone && "k-quant dispatch received kNone");
  // y[0:N] = dequant(w[N,K]) @ x[0:K]. q4_K/q5_K: one simdgroup per row
  // (grid (32,N,1)); q6_K: the optimized v2 form (grid (32,ceil(N/8)*2,1),
  // tg (32,2,1)). Offsets are in elements (the helper converts to bytes).
  enc.set_buffer(0, w);
  enc.set_buffer(1, x, xoff * 2);
  enc.set_buffer(2, y, yoff * 2);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  if (type == KQ::kQ6K) {
    // q6k_v2: _q6k_rpt rows/threadgroup (RPS*NSG), _q6k_nsg simdgroups -- the
    // tuned variant chosen at load.
    enc.set_function(_fn_qmv_q6k);
    const unsigned nsg = (unsigned)_q6k_nsg, rpt = (unsigned)_q6k_rpt;
    enc.dispatch({32, ((unsigned)N + rpt - 1) / rpt * nsg, 1}, {32, nsg, 1});
  } else {
    // q4k/q5k full-byte: 1 row/simdgroup x NSG=2 = 2 rows/threadgroup.
    enc.set_function(type == KQ::kQ5K ? _fn_qmv_q5k : _fn_qmv_q4k);
    enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1});
  }
}

void
MetalQwenModel::kqmv_batch_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                            const SharedBuffer& x, const SharedBuffer& y, int K,
                            int N, int M, int ystride, int yoff)
{
  assert(type != KQ::kNone && "k-quant batch dispatch received kNone");
  // y[base+m, yoff + 0:N] = dequant(w[N,K]) @ x[base+m, 0:K] with the raw
  // k-quant weight read ONCE across each tile. MAXM=2 (depth-1 draft window M=2:
  // one tile, weight read once -> ~1 decode step) or, in the MTP verify at
  // draft depth>=2 (M=3..4, _qmv4_active), the MAXM=4 twin (one tile vs the
  // MAXM=2 form's ceil(M/2)=2 tiles = the depth-2 cliff). Per-row math is
  // bit-identical across MAXM (token-exact). Falls back to looped single-row qmv
  // when the batched kernel is unavailable or VPIPE_GGUF_MTP_LOOPED_GEMV is set
  // (the A/B baseline). M==1 -> plain qmv.
  static const bool force_looped =
      std::getenv("VPIPE_GGUF_MTP_LOOPED_GEMV") != nullptr;
  if (M == 1) {
    kqmv_(enc, type, w, x, 0, y, (std::size_t)yoff, K, N);
    return;
  }
  // Adaptive MAXM by row count (mirrors qmm_auto_): m in 3..4 -> MAXM=4 (the
  // MTP verify depth-2 cliff needs one weight read for 3-4 rows), else MAXM=2.
  // m>=5 stays on MAXM=2: larger MAXM blew occupancy (a net loss for the
  // realtime-vqa batched decode). Per-row math is bit-identical (token-exact).
  // Invalid selected fn -> the looped single-row fallback below.
  const bool use4 = _qmv4_enabled && M > 2 && M <= 4;
  const metal_compute::ComputeFunction& fn =
      use4 ? (type == KQ::kQ6K ? _fn_qmv_q6k_batch4
                               : (type == KQ::kQ5K ? _fn_qmv_q5k_batch4
                                                   : _fn_qmv_q4k_batch4))
           : (type == KQ::kQ6K ? _fn_qmv_q6k_batch
                               : (type == KQ::kQ5K ? _fn_qmv_q5k_batch
                                                   : _fn_qmv_q4k_batch));
  if (force_looped || !fn.valid()) {
    for (int m = 0; m < M; ++m) {
      kqmv_(enc, type, w, x, (std::size_t)m * K, y,
            (std::size_t)m * ystride + yoff, K, N);
    }
    return;
  }
  const int maxm = use4 ? 4 : 2;
  enc.set_function(fn);
  enc.set_buffer(0, w);
  enc.set_buffer(1, x);
  enc.set_buffer(2, y);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  enc.set_constant(5, M);
  enc.set_constant(6, ystride);
  enc.set_constant(7, yoff);
  enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2),
               (unsigned)((M + maxm - 1) / maxm)}, {32, 2, 1});
}

void
MetalQwenModel::dense_gemv_(ComputeEncoder& enc, const SharedBuffer& w,
                            const SharedBuffer& x, std::size_t xoff,
                            const SharedBuffer& y, std::size_t yoff,
                            int K, int N)
{
  // y[0:N] = w[N,K] @ x[0:K], all f16 (dense_gemv_t_f16: RPS=4 rows/simd).
  enc.set_function(_fn_dense_gemv);
  enc.set_buffer(0, x, xoff * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y, yoff * 2);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
}

MetalQwenModel::~MetalQwenModel() = default;

void
MetalQwenModel::set_i8_gemm(bool on)
{
  // Dynamic-int8 accelerated prefill GEMMs (LOSSY, opt-in; see
  // shared/i8-gemm.h). The context self-gates on matrix cores + kernels;
  // VPIPE_I8_GEMM overrides the flag either way.
  auto i8 = std::make_unique<I8GemmContext>(_mc, on);
  _i8 = i8->enabled() ? std::move(i8) : nullptr;
}

void
MetalQwenModel::dense_gemm_(ComputeEncoder& enc, const SharedBuffer& w,
                            const SharedBuffer& x, const SharedBuffer& y,
                            int K, int N, int M)
{
  // Dynamic-int8 accelerated mode (opt-in, LOSSY): quantize activations +
  // the f16 weight on the fly, run the int8 matmul. Prefill-only by
  // construction (the M gate); decode M=1 never qualifies.
  if (_i8 && _i8->gemm(enc, x, 0, w, y, 0, M, N, K)) { return; }
  // y[M,N] = x[M,K] @ w[N,K]^T, all f16, no bias. The caller (kqmm_ / the
  // fused-dequant QKV+GDN paths) has already dequantized the k-quant weight
  // into the f16 scratch, so this is the same plain f16 GEMM the affine
  // prefill runs post-dequant -- and on M5 it should ride the same matrix
  // units. Matrix-core (M5+) matmul2d when the units are present AND the
  // prompt is tall enough for the tiled kernel to amortize (M >= _mma_min_m):
  // tile-adaptive 128x128 for K < 6144, 128x256 for deeper K (down_proj),
  // matching the affine dense_mma path. The steel dense_gemm_t_f16 (BM/BN/
  // BK=32 tiles, tails clamped) stays the M4 / small-M fallback. _use_mma
  // already honours VPIPE_QWEN_NO_MMA, so that A/B toggle covers k-quant too.
  // Both kernels take the identical buffer/constant binding; only the
  // selected function + dispatch grid differ.
  const bool mma = _use_mma && M >= _mma_min_m;
  const bool deep = mma && K >= 6144;
  enc.set_function(mma ? (deep ? _fn_dense_mma_deep : _fn_dense_mma)
                       : _fn_dense_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, w);          // bias slot unused (has_bias=0)
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  if (mma) {
    const int BN = deep ? 256 : 128;
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  // BM=BN=32 tiles, WM=WN=2 -> 128 threads/tg ({32,2,2}); tid.x/tid.y index
  // the N/M tiles (grid.y carries M-tiles via tg.y=2, z=2 is the 4th dim).
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalQwenModel::kqmm_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                      const SharedBuffer& x, const SharedBuffer& y,
                      int K, int N, int M)
{
  assert(type != KQ::kNone && "k-quant prefill GEMM received kNone");
  // Prefill (M rows): dequant the k-quant weight into the reusable f16
  // scratch (sized to the largest N*K seen), then a dense f16 GEMM. The
  // M-independent dequant amortizes across the M prompt rows.
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
  }
  enc.set_function(type == KQ::kQ6K ? _fn_dequant_q6k
                  : (type == KQ::kQ5K ? _fn_dequant_q5k : _fn_dequant_q4k));
  enc.set_buffer(0, w);
  enc.set_buffer(1, _w_deq);
  enc.set_constant(2, N * K);
  enc.dispatch({(unsigned)(N * K), 1, 1}, {256, 1, 1});
  dense_gemm_(enc, _w_deq, x, y, K, N, M);
}

void
MetalQwenModel::kdequant_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                          std::size_t dst_off, int count)
{
  assert(type != KQ::kNone && "k-quant dequant received kNone");
  // Dequant `count` k-quant elements of `w` into _w_deq[dst_off:], for a
  // fused prefill GEMM (heterogeneous parts share one f16 scratch).
  enc.set_function(type == KQ::kQ6K ? _fn_dequant_q6k
                  : (type == KQ::kQ5K ? _fn_dequant_q5k : _fn_dequant_q4k));
  enc.set_buffer(0, w);
  enc.set_buffer(1, _w_deq, dst_off * 2);
  enc.set_constant(2, count);
  enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
}

void
MetalQwenModel::kcopy_(ComputeEncoder& enc, const SharedBuffer& src,
                       std::size_t dst_off, int count)
{
  // copy_f16(src, _w_deq, off, count): fold an f16 sub-tensor (a/b) into the
  // fused dequant scratch alongside the k-quant parts.
  const int off = (int)dst_off;
  enc.set_function(_fn_copy);
  enc.set_buffer(0, src);
  enc.set_buffer(1, _w_deq);
  enc.set_constant(2, off);
  enc.set_constant(3, count);
  enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
}

void
MetalQwenModel::adequant_(ComputeEncoder& enc, const SharedBuffer& w,
                          const SharedBuffer& s, const SharedBuffer& b,
                          int bits, std::size_t dst_off, int N, int K)
{
  // Dequant an affine weight [N,K] (bits 4 or 8, group 64) into _w_deq at
  // element offset dst_off. 4-bit: K/8 u32 words/row; 8-bit: K/4. Grid is
  // one thread per word: {K*bits/32, N}. Byte/nibble order matches the
  // steel/qmv loaders, so the post-dequant dense GEMM equals the qmm.
  enc.set_function(bits == 8 ? _fn_dequant8 : _fn_dequant);
  enc.set_buffer(0, w);
  enc.set_buffer(1, s);
  enc.set_buffer(2, b);
  enc.set_buffer(3, _w_deq, dst_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)((std::size_t)K * bits / 32), (unsigned)N, 1},
               {64, 1, 1});
}

void
MetalQwenModel::aqmm_(ComputeEncoder& enc, const SharedBuffer& w,
                      const SharedBuffer& s, const SharedBuffer& b, int bits,
                      const SharedBuffer& x, const SharedBuffer& y, int K,
                      int N, int M)
{
  // Single affine projection prefill: dequant the whole weight into the
  // reusable f16 scratch, then a dense f16 GEMM (matrix-core on M5). The
  // affine twin of kqmm_; the M-independent dequant amortizes over M rows.
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
  }
  adequant_(enc, w, s, b, bits, 0, N, K);
  dense_gemm_(enc, _w_deq, x, y, K, N, M);
}

void
MetalQwenModel::adequant_g32_(ComputeEncoder& enc, const SharedBuffer& w,
                              const SharedBuffer& s, const SharedBuffer& b,
                              std::size_t dst_off, int N, int K)
{
  // affine_dequant_w4g32(w, scales, biases, _w_deq+off, K, N): one thread per
  // u32 word ({K/8, N}). Twin of adequant_ for the repacked Q4_K (group 32).
  enc.set_function(_fn_dequant_w4g32);
  enc.set_buffer(0, w);
  enc.set_buffer(1, s);
  enc.set_buffer(2, b);
  enc.set_buffer(3, _w_deq, dst_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
}

void
MetalQwenModel::aqmm_g32_(ComputeEncoder& enc, const SharedBuffer& w,
                          const SharedBuffer& s, const SharedBuffer& b,
                          const SharedBuffer& x, const SharedBuffer& y,
                          int K, int N, int M)
{
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
  }
  adequant_g32_(enc, w, s, b, 0, N, K);
  dense_gemm_(enc, _w_deq, x, y, K, N, M);
}

void
MetalQwenModel::amv_g32_batch_(ComputeEncoder& enc, const SharedBuffer& w,
                               const SharedBuffer& s, const SharedBuffer& b,
                               const SharedBuffer& x, const SharedBuffer& y,
                               int K, int N, int M, int ystride, int yoff)
{
  // One affine g32 qmv per branch row (M small: the decode-batch / draft
  // window). The weight is re-read per row vs the k-quant batched tile, but
  // the affine kernel is ~1.9x faster per read, so it still wins at these M.
  for (int m = 0; m < M; ++m) {
    enc.set_function(_fn_qmv_w4g32);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, x, (std::size_t)m * K * 2);
    enc.set_buffer(4, y, ((std::size_t)m * ystride + yoff) * 2);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  }
}

void
MetalQwenModel::embed_q6k_(ComputeEncoder& enc, const SharedBuffer& ids,
                           std::size_t ioff, const SharedBuffer& out, int n)
{
  enc.set_function(_embed_kqt == KQ::kQ4K ? _fn_embed_q4k : _fn_embed_q6k);
  enc.set_buffer(0, ids, ioff * 4);
  enc.set_buffer(1, _embed_q6k);
  enc.set_buffer(2, out);
  enc.set_constant(3, _cfg.hidden);
  enc.dispatch({(unsigned)_cfg.hidden, (unsigned)n, 1}, {256, 1, 1});
}

void
MetalQwenModel::encode_decode_step_(
    ComputeEncoder& enc, ContextId cid, int pos, int rpos,
    std::size_t page_off, int n_pages,
    const ContextManager::AppendSlot& slot,
    const SharedBuffer& pgtab, std::size_t pgtab_off, bool return_hidden)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const int page_tokens = _ctx->page_tokens();

  auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                 const SharedBuffer& w, const SharedBuffer& y,
                 std::size_t yoff, int R, int Hd) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin, xoff);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, Hd);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto qmv = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, const SharedBuffer& xin,
                 std::size_t xoff, const SharedBuffer& y, int Kk, int N) {
    if (s.empty()) {   // dense f16: plain GEMV, no scales/biases
      dense_gemv_(enc, w, xin, xoff, y, 0, Kk, N);
      return;
    }
    enc.set_function(_fn_qmv);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin, xoff);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto qmv_add = [&](const SharedBuffer& w, const SharedBuffer& s,
                     const SharedBuffer& b, const SharedBuffer& xin,
                     const SharedBuffer& y, const SharedBuffer& res, int Kk,
                     int N) {
    if (s.empty()) {   // dense f16: y = res + W@xin (GEMV into _d_radd + add)
      dense_gemv_(enc, w, xin, 0, _d_radd, 0, Kk, N);
      enc.set_function(_fn_residual);
      enc.set_buffer(0, res);
      enc.set_buffer(1, _d_radd);
      enc.set_buffer(2, y);
      enc.set_constant(3, N);
      enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_qmv_add);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.set_buffer(7, res);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  // Mixed-precision affine (OptiQ) per-tensor matvec: dequant(w at `bits`) @
  // x -> y[yoff_e : yoff_e+N]. Writes a sub-range of a wider de-fused buffer
  // (the q|k|v or qkv|z|a|b layout the fused qmv produces in one shot), so the
  // downstream hslices are unchanged. Same dispatch as qmv (w4 and w8 share
  // the qmv_fast grid). `bits` 4 -> _fn_qmv, 8 -> _fn_qmv8.
  auto amv = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, int bits, const SharedBuffer& xin,
                 const SharedBuffer& y, std::size_t yoff_e, int Kk, int N) {
    enc.set_function(bits == 8 ? _fn_qmv8 : _fn_qmv);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin, 0);
    enc.set_buffer(4, y, yoff_e * 2);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  // Affine 4-bit g32 matvec (the Q4_K->affine decode repack path).
  auto amv_g32 = [&](const SharedBuffer& w, const SharedBuffer& s,
                     const SharedBuffer& b, const SharedBuffer& xin,
                     const SharedBuffer& y, int Kk, int N) {
    enc.set_function(_fn_qmv_w4g32);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto amv_add = [&](const SharedBuffer& w, const SharedBuffer& s,
                     const SharedBuffer& b, int bits, const SharedBuffer& xin,
                     const SharedBuffer& y, const SharedBuffer& res, int Kk,
                     int N) {
    enc.set_function(bits == 8 ? _fn_qmv8_add : _fn_qmv_add);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.set_buffer(7, res);     // read (residual); y==res is in-place
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto hslice = [&](const SharedBuffer& in, std::size_t ioff,
                    const SharedBuffer& out, std::size_t ooff, int Hh,
                    int S, int W, int off, int block = 0, int gstride = 0) {
    enc.set_function(_fn_head_slice);
    enc.set_buffer(0, in, ioff);
    enc.set_buffer(1, out, ooff);
    enc.set_constant(2, Hh);
    enc.set_constant(3, S);
    enc.set_constant(4, W);
    enc.set_constant(5, off);
    enc.set_constant(6, block);
    enc.set_constant(7, gstride);
    enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
  };
  auto rope = [&](const SharedBuffer& xb, int heads, std::size_t xoff = 0) {
    enc.set_function(_fn_rope_partial);
    enc.set_buffer(0, xb, xoff);  // in-place rotate
    enc.set_buffer(1, _inv_freq);
    enc.set_constant(2, heads);
    const int one = 1;
    enc.set_constant(3, one);
    enc.set_constant(4, D);
    enc.set_constant(5, c.rotary_dim);
    enc.set_constant(6, rpos);
    enc.dispatch({(unsigned)(c.rotary_dim / 2), 1, (unsigned)heads},
                 {(unsigned)(c.rotary_dim / 2), 1, 1});
  };
  // Fused per-head RMSNorm + partial RoPE (one dispatch vs rms + rope).
  auto rms_rope = [&](const SharedBuffer& xb, const SharedBuffer& w,
                      int heads, std::size_t xoff = 0) {
    enc.set_function(_fn_rms_rope);
    enc.set_buffer(0, xb, xoff);  // in-place norm+rotate
    enc.set_buffer(1, w);
    enc.set_buffer(2, _inv_freq);
    enc.set_constant(3, D);
    enc.set_constant(4, c.rotary_dim);
    enc.set_constant(5, rpos);
    enc.set_constant(6, eps);
    enc.dispatch({256, (unsigned)heads, 1}, {256, 1, 1});
  };
  auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool,
                      std::size_t soff = 0) {
    enc.set_function(_fn_kv_write_paged);
    enc.set_buffer(0, src, soff);
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
  // Category profiler: D(cat, fn) runs fn() once, and AGAIN if profiling
  // that category (re-issues the dispatch(es) in the same command buffer).
  // The bench diffs decode time vs the `none` baseline -> delta == that
  // category's GPU cost (compute + its hazard-barrier drains). Off in
  // production (dup<0): fn() runs exactly once. Idempotent on the dense
  // ASR path (no accumulating state). Profiling forces the unfused
  // norm+rope so norm vs rope separate.
  enum { DC_PROJ = 1, DC_FFN = 2, DC_LMHEAD = 3, DC_ATTN = 4, DC_NORM = 5,
         DC_ROPE = 6, DC_MISC = 7, DC_GDN = 8, DC_GDN_REC = 9 };
  // Production: dup<0 -> DUP(cat,fn) runs fn() exactly once (zero cost, no
  // getenv). Only a profiling session (VPIPE_QWEN_CATPROF set at load)
  // reads VPIPE_QWEN_DUP_CAT per step, so a within-process profiler can
  // toggle the duplicated category between passes -- the only reliable way
  // to compare categories at a STEADY GPU clock (cross-process runs start
  // at a low clock and ramp, corrupting cross-run deltas).
  auto parse_cat = [](const char* s) -> int {
    const std::string v = s;
    return (v == "proj") ? 1 : (v == "ffn") ? 2 : (v == "lmhead") ? 3
         : (v == "attn") ? 4 : (v == "norm") ? 5 : (v == "rope") ? 6
         : (v == "misc") ? 7 : (v == "gdn") ? 8 : (v == "gdn_rec") ? 9 : 0;
  };
  int dup = -1, skip = -1;
  if (_catprof) {
    if (const char* e = std::getenv("VPIPE_QWEN_DUP_CAT")) { dup = parse_cat(e); }
    // Ablation: VPIPE_QWEN_SKIP_CAT short-circuits a category (fn() not run) ->
    // the tok/s delta vs baseline = that category's REAL cost, with no
    // cache-reuse inflation (unlike DUP doubling a BW-bound matmul). Output is
    // garbage but the timing is valid.
    if (const char* e = std::getenv("VPIPE_QWEN_SKIP_CAT")) { skip = parse_cat(e); }
  }
  auto DUP = [&](int cat, auto&& fn) {
    if (skip == cat) { return; }
    fn();
    if (dup == cat) { fn(); }
  };
  // Fine GDN-recurrence sub-op AMPLIFICATION: re-dispatch one sub-op N extra
  // times so the measured delta is N x its per-invocation GPU cost (small ops
  // are below the tok/s noise floor at 1x). VPIPE_QWEN_GDN_REP_OP in
  // {conv,qknorm,gbeta,ssm,gatedrms}, VPIPE_QWEN_GDN_REP_N = extra count.
  int grep_op = -1, grep_n = 0;
  if (_catprof) {
    if (const char* e = std::getenv("VPIPE_QWEN_GDN_REP_OP")) {
      const std::string v = e;
      grep_op = (v == "conv") ? 1 : (v == "qknorm") ? 2 : (v == "gbeta") ? 3
              : (v == "ssm") ? 4 : (v == "gatedrms") ? 5 : -1;
    }
    if (const char* e = std::getenv("VPIPE_QWEN_GDN_REP_N")) {
      grep_n = std::atoi(e);
    }
  }
  auto GREP = [&](int op, metal_compute::LaunchDims g,
                  metal_compute::LaunchDims tg) {
    enc.dispatch(g, tg);
    if (op == grep_op) {
      for (int i = 0; i < grep_n; ++i) { enc.dispatch(g, tg); }
    }
  };
  // k-quant residual add: _d_x[0:n] += _d_radd[0:n] (the qmv-then-add the
  // affine qmv_*_add fuses; native k-quant has no fused-add qmv variant).
  auto radd = [&](int n) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, _d_x);
    enc.set_buffer(1, _d_radd);
    enc.set_buffer(2, _d_x);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };

  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    DUP(DC_NORM, [&] { rms(_d_x, 0, ly.in_ln, _d_hn, 0, 1, H); });
    if (ly.is_full) {
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);
      const bool gate = c.attn_output_gate;
      const int qdo = gate ? 2 * qd : qd;
      const int Nfqkv = qdo + 2 * kd;
      DUP(DC_PROJ, [&] {
        if (_kquant) {
          // q+k fused (q4_K) -> _d_qfull[0:kqk_n]; v (q6_K) -> the kd tail
          // [kqk_n : kqk_n+kd], reproducing the [q|k|v] fused layout. The qk and
          // v matvecs are independent (shared _d_hn, disjoint _d_qfull writes) ->
          // overlap them under hybrid (GGUF mixed-quant can't fuse qk with v).
          auto _cg = enc.concurrent_scope(decode_hybrid_());
          if (ly.qk_q4k_aff) {
            amv_g32(ly.qk_aw, ly.qk_as, ly.qk_ab, _d_hn, _d_qfull, H,
                    ly.kqk_n);
          } else {
            kqmv_(enc, ly.kqk_t, ly.kqk, _d_hn, 0, _d_qfull, 0, H, ly.kqk_n);
          }
          kqmv_(enc, ly.kqv_t, ly.kqv, _d_hn, 0, _d_qfull,
                (std::size_t)ly.kqk_n, H, kd);
        } else if (_mixed && !ly.qkv_fused) {
          // q|k|v each its own bit width into the [q(qdo)|k(kd)|v(kd)] layout.
          // Independent (shared _d_hn read, disjoint _d_qfull writes) -> run
          // them concurrently so k/v overlap under q's DRAM read (no-op on a
          // Serial encoder). (A uniform-base-width layer is fused -> the plain
          // qmv else below; gated on !ly.qkv_fused.)
          auto _cg = enc.concurrent_scope(decode_hybrid_());
          amv(ly.qw, ly.qs, ly.qb, ly.q_bits, _d_hn, _d_qfull, 0, H, qdo);
          amv(ly.kw, ly.ks, ly.kb, ly.k_bits, _d_hn, _d_qfull,
              (std::size_t)qdo, H, kd);
          amv(ly.vw, ly.vs, ly.vb, ly.v_bits, _d_hn, _d_qfull,
              (std::size_t)(qdo + kd), H, kd);
        } else {
          qmv(ly.qw, ly.qs, ly.qb, _d_hn, 0, _d_qfull, H, Nfqkv);
        }
      });
      // No-hslice K/V: k and v are the contiguous [q..|k(kd)|v(kd)] tail of
      // _d_qfull -> bind it at the k/v byte offsets to the k-path + kv-write
      // instead of copying them out. q/gate always stay copied (the gated case
      // interleaves them). kb/vb are the (buffer, byte-offset) of k/v.
      const bool attn_nohs = decode_attn_nohs_();
      const SharedBuffer& kb = attn_nohs ? _d_qfull : _d_kbuf;
      const SharedBuffer& vb = attn_nohs ? _d_qfull : _d_vbuf;
      const std::size_t kb_o = attn_nohs ? (std::size_t)qdo * 2 : 0;
      const std::size_t vb_o = attn_nohs ? (std::size_t)(qdo + kd) * 2 : 0;
      DUP(DC_MISC, [&] {
        if (gate) {
          hslice(_d_qfull, 0, _d_q3, 0, Hq, 2 * D, D, 0);
          hslice(_d_qfull, 0, _d_gate3, 0, Hq, 2 * D, D, D);
        } else {
          hslice(_d_qfull, 0, _d_q3, 0, 1, Nfqkv, qd, 0);
        }
        if (!attn_nohs) {
          hslice(_d_qfull, 0, _d_kbuf, 0, 1, Nfqkv, kd, qdo);
          hslice(_d_qfull, 0, _d_vbuf, 0, 1, Nfqkv, kd, qdo + kd);
        }
      });
      if (_fn_rms_rope.valid() && dup < 0) {
        // Fused: q_norm+rope_q and k_norm+rope_k -> one dispatch each. (These
        // two are independent, but overlapping them under hybrid measured ~0 --
        // latency-bound tiny ops, encoder-switch cost cancels the gain.)
        rms_rope(_d_q3, ly.q_norm, Hq);
        rms_rope(kb, ly.k_norm, Hkv, kb_o);
      } else {
        DUP(DC_NORM, [&] {
          rms(_d_q3, 0, ly.q_norm, _d_q3, 0, Hq, D);
          rms(kb, kb_o, ly.k_norm, kb, kb_o, Hkv, D);
        });
        DUP(DC_ROPE, [&] { rope(_d_q3, Hq); rope(kb, Hkv, kb_o); });
      }
      DUP(DC_MISC, [&] { kv_write(kb, kp, kb_o); kv_write(vb, vp, vb_o); });
      // Multi-simdgroup paged attention past the long-context threshold
      // (the scalar kernel scans KV with only 32 lanes -> dominates the
      // per-layer critical path at high pos). D==256 (Qwen3.5) -> mb256
      // (BN=16); D<=128 (Qwen3-ASR) -> mb (BN=32, shared with Llama);
      // else scalar.
      const bool long_ctx = (pos + 1) >= _sdpa_mb_min;
      // Flash-decode-GQA (head_dim 256): read each KV head ONCE for all G
      // query heads (scan) + position-split merge -> G x less KV bandwidth
      // than mb256's per-q-head re-scan. Writes the same _d_attn [Hq*D].
      const bool use_gqa = _gqa_attn && long_ctx && (D <= 256)
          && (D % 32 == 0) && !_d_gqa_oacc.empty();
      if (use_gqa) {
        static const int kVec1 = []() {
          const char* e = std::getenv("VPIPE_QWEN_GQA_VEC1");
          return (e && std::atoi(e) != 0) ? 1 : 0;   // default OFF (legacy A/B)
        }();
        const bool use_vec1 = kVec1 && (D == 256) && _fn_sdpa_gqa_vec1.valid();
        // PRIMARY: the decode GQA attention SET picks vec2/kbl + split for this
        // position's regime (tuned at load) and runs it via its unified
        // entrance. The legacy split kernels below handle D!=256, a not-ready
        // set, or a forced vec1 (the superseded intra-TG-merge variant).
        if (D == 256 && _decode_set.ready() && !use_vec1) {
          DUP(DC_ATTN, [&] {
            DecodeGqaAttnSet::Attn a;
            a.q = &_d_q3; a.kpool = &kp; a.vpool = &vp; a.out = &_d_attn;
            a.page_table = &pgtab; a.pgt_off = pgtab_off;
            a.pos = pos; a.page_tokens = page_tokens; a.n_pages = n_pages;
            a.scale = scale;
            _decode_set.dispatch(enc, a);
          });
        } else {
          const int sp = use_vec1
              ? std::max(4, std::min(64, (pos + 511) / 512))
              : gqa_decode_split(D, pos);
          const int G = Hq / Hkv;
          const bool use_vec =
              _gqa_vec && (D % 128 == 0) && _fn_sdpa_gqa_vec.valid();
          const bool use_kbl = use_vec && _gqa_blk && (D == 256 || D == 128);
          const metal_compute::ComputeFunction& vec_fn =
              use_vec1 ? _fn_sdpa_gqa_vec1
              : (!use_kbl ? _fn_sdpa_gqa_vec
                          : (D == 256 ? _fn_sdpa_gqa_kbl
                                      : _fn_sdpa_gqa_kbl128));
          DUP(DC_ATTN, [&] {
            enc.set_function((use_vec || use_vec1) ? vec_fn : _fn_sdpa_gqa);
            enc.set_buffer(0, _d_q3);
            enc.set_buffer(1, kp);
            enc.set_buffer(2, vp);
            enc.set_buffer(3, _d_gqa_oacc);
            enc.set_buffer(4, _d_gqa_m);
            enc.set_buffer(5, _d_gqa_l);
            enc.set_constant(6, scale);
            enc.set_constant(7, D);
            enc.set_constant(8, Hq);
            enc.set_constant(9, Hkv);
            enc.set_constant(10, pos);
            enc.set_constant(11, page_tokens);
            enc.set_constant(12, n_pages);
            enc.set_buffer(13, pgtab, pgtab_off);
            enc.set_constant(14, sp);
            if (use_vec1) {
              enc.dispatch({1024, (unsigned)Hq, (unsigned)sp}, {1024, 1, 1});
            } else if (use_vec) {
              enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                           {32, (unsigned)G, 1});
            } else {
              enc.dispatch({32, (unsigned)Hkv, (unsigned)sp}, {32, 1, 1});
            }
            enc.set_function(_fn_sdpa_gqa_merge);
            enc.set_buffer(0, _d_gqa_oacc);
            enc.set_buffer(1, _d_gqa_m);
            enc.set_buffer(2, _d_gqa_l);
            enc.set_buffer(3, _d_attn);
            enc.set_constant(4, D);
            enc.set_constant(5, sp);
            enc.set_constant(6, Hq);
            enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
          });
        }
      } else {
      const bool use_mb256 = long_ctx && (D == 256);
      const bool use_mb128 =
          long_ctx && (D <= 128) && _fn_sdpa_paged_mb.valid();
      enc.set_function(use_mb256 ? _fn_sdpa_paged_mb256
                                 : (use_mb128 ? _fn_sdpa_paged_mb
                                              : _fn_sdpa_paged));
      enc.set_buffer(0, _d_q3);
      enc.set_buffer(1, kp);
      enc.set_buffer(2, vp);
      enc.set_buffer(3, _d_attn);
      enc.set_constant(4, scale);
      enc.set_constant(5, D);
      enc.set_constant(6, Hq);
      enc.set_constant(7, Hkv);
      const int one = 1;
      enc.set_constant(8, one);
      enc.set_constant(9, pos);
      enc.set_constant(10, page_tokens);
      enc.set_constant(11, n_pages);
      enc.set_buffer(12, pgtab, pgtab_off);
      const unsigned w =
          use_mb256 ? 32u * 16u : (use_mb128 ? 32u * 32u : 32u);
      DUP(DC_ATTN, [&] { enc.dispatch({w, (unsigned)Hq, 1}, {w, 1, 1}); });
      }
      if (gate) {
        enc.set_function(_fn_mul_sigmoid);
        enc.set_buffer(0, _d_attn);
        enc.set_buffer(1, _d_gate3);
        enc.set_buffer(2, _d_attn);
        enc.set_constant(3, qd);
        enc.dispatch({(unsigned)qd, 1, 1}, {256, 1, 1});
      }
      DUP(DC_PROJ, [&] {
        if (_kquant) {
          if (ly.o_q4k_aff) {
            amv_g32(ly.o_aw, ly.o_as, ly.o_ab, _d_attn, _d_radd, qd, H);
          } else {
            kqmv_(enc, ly.kqo_t, ly.kqo, _d_attn, 0, _d_radd, 0, qd, H);
          }
          radd(H);
        } else if (_mixed) {
          amv_add(ly.ow, ly.os, ly.ob, ly.o_bits, _d_attn, _d_x, _d_x, qd, H);
        } else {
          qmv_add(ly.ow, ly.os, ly.ob, _d_attn, _d_x, _d_x, qd, H);
        }
      });
    } else {
      // Whole GDN (linear-attn) layer as one profiler category (DC_GDN). The
      // ops carry SSM/conv state, so the duplicate-for-timing only advances it
      // an extra step -- valid for the throwaway profiling runs (CATPROF), a
      // no-op in production (dup<0 -> runs once).
      // GDN recurrent state via the run-ahead ring: read the current slot,
      // write the next. The ring is OFF in the sync/decode path (read ==
      // write == canonical -> in-place, identical to before); pdecode depth>1
      // ping-pongs the slots so a speculative advance can be rolled back.
      const SharedBuffer* cs_r = _ctx->conv_read(cid, L);
      const SharedBuffer* cs_w = _ctx->conv_write(cid, L);
      const SharedBuffer* ss_r = _ctx->ssm_read(cid, L);
      const SharedBuffer* ss_w = _ctx->ssm_write(cid, L);
      const int one = 1;
      const int Nf = Cd + vald + 2 * Hv;
      DUP(DC_GDN, [&] {     // GDN in_proj matvecs
      if (_kquant) {
        // in_proj is heterogeneous: qkv (q5_K) + z (q4_K) + a|b (f16) each
        // run their own matmul into the [qkv|z|a|b] mixqkv layout. Independent
        // (shared _d_hn read, disjoint _d_mixqkv writes) -> overlap under hybrid.
        auto _cg = enc.concurrent_scope(decode_hybrid_());
        kqmv_(enc, ly.kqkv_t, ly.kqkv, _d_hn, 0, _d_mixqkv, 0, H, Cd);
        kqmv_(enc, ly.kqz_t, ly.kqz, _d_hn, 0, _d_mixqkv, (std::size_t)Cd,
              H, vald);
        dense_gemv_(enc, ly.kqab, _d_hn, 0, _d_mixqkv,
                    (std::size_t)(Cd + vald), H, 2 * Hv);
      } else if (_mixed) {
        // qkv|z|a|b each its own bit width into the [qkv(Cd)|z(vald)|a(Hv)|
        // b(Hv)] fused layout the hslices below carve up. Independent (shared
        // _d_hn read, disjoint _d_mixqkv writes) -> concurrent so z/a/b overlap
        // under qkv's read (no-op on a Serial encoder).
        auto _cg = enc.concurrent_scope(decode_hybrid_());
        amv(ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, _d_hn, _d_mixqkv, 0, H, Cd);
        amv(ly.izw, ly.izs, ly.izb, ly.z_bits, _d_hn, _d_mixqkv,
            (std::size_t)Cd, H, vald);
        amv(ly.iaw, ly.ias, ly.iab, ly.a_bits, _d_hn, _d_mixqkv,
            (std::size_t)(Cd + vald), H, Hv);
        amv(ly.ibw, ly.ibs, ly.ibb, ly.b_bits, _d_hn, _d_mixqkv,
            (std::size_t)(Cd + vald + Hv), H, Hv);
      } else {
        qmv(ly.iqw, ly.iqs, ly.iqb, _d_hn, 0, _d_mixqkv, H, Nf);
      }
      });
      DUP(DC_GDN_REC, [&] {   // conv1d + qk_norm + g_beta + ssm step + gated_rms
      // z/a/b are contiguous in _d_mixqkv [qkv|z|a|b]; materialize them as
      // separate buffers OR (no-hslice) bind _d_mixqkv directly by offset below.
      const bool noslice = decode_no_hslice_();
      if (!noslice) {
        hslice(_d_mixqkv, 0, _d_zbuf, 0, 1, Nf, vald, Cd);
        hslice(_d_mixqkv, 0, _d_abuf, 0, 1, Nf, Hv, Cd + vald);
        hslice(_d_mixqkv, 0, _d_bbuf, 0, 1, Nf, Hv, Cd + vald + Hv);
      }
      // _d_mixqkv is 2-byte elements (f16/bf16); z/a/b byte offsets:
      const std::size_t z_off = (std::size_t)Cd * 2;
      const std::size_t a_off = (std::size_t)(Cd + vald) * 2;
      const std::size_t b_off = (std::size_t)(Cd + vald + Hv) * 2;
      const float inv_scale = 1.0f / std::sqrt((float)Dk);
      const float s_q = inv_scale * inv_scale, s_k = inv_scale;
      // conv1d and g_beta are independent (conv1d reads the qkv slice of the
      // in_proj output, g_beta reads the a|b slices); issue them into one short
      // CONCURRENT sub-encoder so g_beta overlaps conv1d and leaves the critical
      // path. The scope-end boundary syncs both into the serial qk_norm tail.
      {
        auto _cg = enc.concurrent_scope(decode_gdn_overlap_());
        enc.set_function(_fn_gdn_conv1d);
        enc.set_buffer(0, *cs_r);
        enc.set_buffer(1, _d_mixqkv);
        enc.set_buffer(2, ly.conv_w);
        enc.set_buffer(3, _d_convout);
        enc.set_constant(4, one);
        enc.set_constant(5, Cd);
        enc.set_constant(6, K);
        enc.set_constant(7, Nf);
        enc.set_constant(8, keyd);
        enc.set_buffer(9, *cs_w);
        GREP(1, {(unsigned)Cd, 1, 1}, {256, 1, 1});
        enc.set_function(_fn_gdn_g_beta);
        if (noslice) {
          enc.set_buffer(0, _d_mixqkv, a_off);
          enc.set_buffer(1, _d_mixqkv, b_off);
        } else {
          enc.set_buffer(0, _d_abuf);
          enc.set_buffer(1, _d_bbuf);
        }
        enc.set_buffer(2, ly.A_log);
        enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, _d_gbuf);
        enc.set_buffer(5, _d_betabuf);
        enc.set_constant(6, Hv);
        enc.set_constant(7, one);
        GREP(3, {(unsigned)Hv, 1, 1}, {256, 1, 1});
      }
      enc.set_function(_fn_gdn_qk_norm);
      enc.set_buffer(0, _d_convout);
      enc.set_constant(1, Hk);
      enc.set_constant(2, Dk);
      enc.set_constant(3, s_q);
      enc.set_constant(4, s_k);
      enc.set_constant(5, eps);
      GREP(2, {128, (unsigned)(2 * Hk), 1}, {128, 1, 1});
      enc.set_function(_fn_gdn_step);
      enc.set_buffer(0, _d_convout, 0);
      enc.set_buffer(1, _d_convout, (std::size_t)keyd * 2);
      enc.set_buffer(2, _d_convout, (std::size_t)2 * keyd * 2);
      enc.set_buffer(3, _d_gbuf);
      enc.set_buffer(4, _d_betabuf);
      enc.set_buffer(5, *ss_r);
      enc.set_buffer(6, _d_ygdn);
      enc.set_buffer(7, *ss_w);
      enc.set_constant(8, one);
      // Negative Hk selects strided GQA (hk = hv % Hk): the GGUF (llama.cpp)
      // stores GDN v-heads de-interleaved [0,2,..,30,1,3,..,31], vs the HF/
      // safetensors interleaved order that the contiguous mapping
      // (hk = hv / (Hv/Hk)) assumes. No-op when Hv == Hk (2B).
      enc.set_constant(9, _kquant ? -Hk : Hk);
      enc.set_constant(10, Hv);
      GREP(4, {32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
      enc.set_function(_fn_gdn_gated_rms);
      enc.set_buffer(0, _d_ygdn);
      enc.set_buffer(1, ly.gdn_norm);
      if (noslice) { enc.set_buffer(2, _d_mixqkv, z_off); }
      else { enc.set_buffer(2, _d_zbuf); }
      enc.set_buffer(3, _d_normout);
      enc.set_constant(4, Dv);
      enc.set_constant(5, eps);
      GREP(5, {128, (unsigned)Hv, 1}, {128, 1, 1});
      });   // DUP(DC_GDN_REC)
      DUP(DC_GDN, [&] {     // GDN out_proj
      if (_kquant) {
        kqmv_(enc, ly.kqout_t, ly.kqout, _d_normout, 0, _d_radd, 0, vald, H);
        radd(H);
      } else if (_mixed) {
        amv_add(ly.gow, ly.gos, ly.gob, ly.gout_bits, _d_normout, _d_x, _d_x,
                vald, H);
      } else {
        qmv_add(ly.gow, ly.gos, ly.gob, _d_normout, _d_x, _d_x, vald, H);
      }
      });   // DUP(DC_GDN out_proj)
    }
    DUP(DC_NORM, [&] { rms(_d_x, 0, ly.post_ln, _d_hn, 0, 1, H); });
    if (c.is_moe()) {
      // Mixture-of-Experts MLP (decode, M=1). _d_hn = post-attention normed
      // hidden; the helper routes on-GPU and adds the expert mixture + the
      // sigmoid-gated shared expert into the residual _d_x.
      DUP(DC_FFN, [&] {
        encode_moe_mlp_(enc, ly, 1, _d_x, _d_hn, _d_moe_logits, _d_moe_ids,
                        _d_moe_w, _d_moe_act, _d_moe_part, _d_moe_out,
                        _d_moe_ssg, _d_moe_sout, _d_moe_gate);
      });
    } else if (_kquant) {
      // gate + up as two q4_K qmv into _d_sg / _d_up, then SwiGLU + down.
      DUP(DC_FFN, [&] {
        {
          // gate + up independent (shared _d_hn, disjoint _d_sg/_d_up) -> overlap
          // under hybrid; the swiglu below reads both -> barriers at scope end.
          auto _cg = enc.concurrent_scope(decode_hybrid_());
          if (ly.ffn_q4k_aff) {
            amv_g32(ly.gate_aw, ly.gate_as, ly.gate_ab, _d_hn, _d_sg, H,
                    c.ffn_inner);
            amv_g32(ly.up_aw, ly.up_as, ly.up_ab, _d_hn, _d_up, H, c.ffn_inner);
          } else {
            kqmv_(enc, ly.kqgate_t, ly.kqgate, _d_hn, 0, _d_sg, 0, H, c.ffn_inner);
            kqmv_(enc, ly.kqup_t, ly.kqup, _d_hn, 0, _d_up, 0, H, c.ffn_inner);
          }
        }
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, _d_sg);
        enc.set_buffer(1, _d_up);
        enc.set_buffer(2, _d_sg);
        enc.set_constant(3, c.ffn_inner);
        enc.dispatch({(unsigned)c.ffn_inner, 1, 1}, {256, 1, 1});
      });
      DUP(DC_FFN, [&] {
        kqmv_(enc, ly.kqdown_t, ly.kqdown, _d_sg, 0, _d_radd, 0,
              c.ffn_inner, H);
        radd(H);
      });
    } else if (_mixed) {
      if (ly.mlp_fused) {
        // gate|up fused (both w4) -> ONE swiglu qmv (weights read once, no
        // _d_up round-trip, no standalone swiglu bubble). down keeps its bits.
        enc.set_function(_fn_qmv_swiglu);
        enc.set_buffer(0, ly.guw);
        enc.set_buffer(1, ly.gus);
        enc.set_buffer(2, ly.gub);
        enc.set_buffer(3, _d_hn);
        enc.set_buffer(4, _d_sg);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * c.ffn_inner);
        DUP(DC_FFN, [&] {
          enc.dispatch({32, (unsigned)(c.ffn_inner / 2), 1}, {32, 2, 1});
        });
      } else {
        // De-fused: gate + up as two affine qmv (own bits) into _d_sg / _d_up,
        // then a standalone SwiGLU (the 4 genuinely mixed-bit layers).
        DUP(DC_FFN, [&] {
          {
            // gate/up independent (shared _d_hn read, distinct _d_sg/_d_up
            // writes) -> concurrent. The swiglu below reads both -> barriers
            // after the group. (no-op on a Serial encoder.)
            auto _cg = enc.concurrent_scope(decode_hybrid_());
            amv(ly.guw, ly.gus, ly.gub, ly.gate_bits, _d_hn, _d_sg, 0, H,
                c.ffn_inner);
            amv(ly.uw, ly.us, ly.ub, ly.up_bits, _d_hn, _d_up, 0, H,
                c.ffn_inner);
          }
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, _d_sg);
          enc.set_buffer(1, _d_up);
          enc.set_buffer(2, _d_sg);
          enc.set_constant(3, c.ffn_inner);
          enc.dispatch({(unsigned)c.ffn_inner, 1, 1}, {256, 1, 1});
        });
      }
      DUP(DC_FFN, [&] {
        amv_add(ly.dw, ly.ds, ly.db, ly.down_bits, _d_sg, _d_x, _d_x,
                c.ffn_inner, H);
      });
    } else if (_dense) {
      // Dense f16: gate + up as two GEMVs into _d_sg / _d_up, SwiGLU, then
      // down GEMV into _d_radd + residual add.
      DUP(DC_FFN, [&] {
        dense_gemv_(enc, ly.guw, _d_hn, 0, _d_sg, 0, H, c.ffn_inner);
        dense_gemv_(enc, ly.uw, _d_hn, 0, _d_up, 0, H, c.ffn_inner);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, _d_sg);
        enc.set_buffer(1, _d_up);
        enc.set_buffer(2, _d_sg);
        enc.set_constant(3, c.ffn_inner);
        enc.dispatch({(unsigned)c.ffn_inner, 1, 1}, {256, 1, 1});
      });
      DUP(DC_FFN, [&] {
        dense_gemv_(enc, ly.dw, _d_sg, 0, _d_radd, 0, c.ffn_inner, H);
        radd(H);
      });
    } else {
      enc.set_function(_fn_qmv_swiglu);
      enc.set_buffer(0, ly.guw);
      enc.set_buffer(1, ly.gus);
      enc.set_buffer(2, ly.gub);
      enc.set_buffer(3, _d_hn);
      enc.set_buffer(4, _d_sg);
      enc.set_constant(5, H);
      enc.set_constant(6, 2 * c.ffn_inner);
      DUP(DC_FFN, [&] {
        enc.dispatch({32, (unsigned)(c.ffn_inner / 2), 1}, {32, 2, 1});
      });
      DUP(DC_FFN, [&] {
        qmv_add(ly.dw, ly.ds, ly.db, _d_sg, _d_x, _d_x, c.ffn_inner, H);
      });
    }
  }

  DUP(DC_NORM, [&] { rms(_d_x, 0, _final_ln, _d_hn, 0, 1, H); });
  // MOSS-TTS: the caller (decode_embedding_hidden) consumes _d_hn directly;
  // skip the lm_head (backbone_only models never loaded one).
  if (return_hidden) { return; }
  if (_kquant) {
    DUP(DC_LMHEAD, [&] {
      kqmv_(enc, lm_head_kqt_(), lm_head_kq_(), _d_hn, 0, _d_logits, 0, H,
            c.vocab);
    });
  } else if (_dense || _dense_embed) {
    DUP(DC_LMHEAD, [&] {
      dense_gemv_(enc, _tied ? _embed_w : _lm_w, _d_hn, 0, _d_logits, 0, H,
                  c.vocab);
    });
  } else {
    // Mixed: lm_head is the (tied) 8-bit embed in OptiQ -> pick the w8 qmv.
    const int lb = _tied ? _embed_bits : _lm_bits;
    enc.set_function((_mixed && lb == 8) ? _fn_qmv8 : _fn_qmv);
    enc.set_buffer(0, _tied ? _embed_w : _lm_w);
    enc.set_buffer(1, _tied ? _embed_s : _lm_s);
    enc.set_buffer(2, _tied ? _embed_b : _lm_b);
    enc.set_buffer(3, _d_hn);
    enc.set_buffer(4, _d_logits);
    enc.set_constant(5, H);
    enc.set_constant(6, c.vocab);
    DUP(DC_LMHEAD, [&] {
      enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
    });
  }
}

// ---------------------------------------------------------------------
// Batched (N-branch parallel) decode.
// ---------------------------------------------------------------------
bool
MetalQwenModel::ensure_bscratch_(BScratch& bs, int n)
{
  // One-shot per-machine GEMV ladder probe (no-op after the first call);
  // batched decode is the main _qmv_plan consumer.
  autotune_qmv_ladder_();
  if (bs.n == n && !bs.logits.empty()) { return true; }
  const Config& c = _cfg;
  const int H = c.hidden, qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, vald = c.value_dim(), Hv = c.gdn_v_heads;
  const int ffn = c.ffn_inner;
  auto f16 = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  auto i32 = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(std::int32_t));
  };
  bs.x      = f16((std::size_t)n * H);
  bs.hn     = f16((std::size_t)n * H);
  bs.qfull  = f16((std::size_t)n * (2 * qd + 2 * kd));
  bs.q3     = f16((std::size_t)n * qd);
  bs.gate3  = f16((std::size_t)n * qd);
  bs.kbuf   = f16((std::size_t)n * kd);
  bs.vbuf   = f16((std::size_t)n * kd);
  bs.at     = f16((std::size_t)n * qd);
  bs.ao     = f16((std::size_t)n * H);
  bs.mixqkv = f16((std::size_t)n * (Cd + vald + 2 * Hv));
  bs.zbuf   = f16((std::size_t)n * vald);
  bs.abuf   = f16((std::size_t)n * Hv);
  bs.bbuf   = f16((std::size_t)n * Hv);
  bs.convout = f16((std::size_t)n * Cd);
  bs.gbuf   = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  bs.betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  bs.ygdn   = f16((std::size_t)n * vald);
  bs.normout = f16((std::size_t)n * vald);
  bs.sg     = f16((std::size_t)n * ffn);
  bs.upb    = f16((std::size_t)n * ffn);   // mixed MLP: de-fused up_proj
  bs.logits = f16((std::size_t)n * c.vocab);
  bs.pgt    = i32((std::size_t)n * _ctx->max_pages() * 3);
  bs.tok_in = i32((std::size_t)n);
  bs.argmax_id = i32((std::size_t)n);
  // Shared-prefix attention partials (f32): O_acc [n, Hq, D]=[n*qd], m/l
  // [n, Hq]=[n*n_heads]. Only used by the head_dim-256 shared-attn path.
  auto f32 = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(float));
  };
  bs.shacc = f32((std::size_t)n * qd);
  bs.shm   = f32((std::size_t)n * c.n_heads);
  bs.shl   = f32((std::size_t)n * c.n_heads);
  if (c.is_moe()) {
    // MoE batched scratch (npair = n*top_k). ffn is 0 for MoE so bs.sg/upb
    // above are empty (unused); the helper uses these instead.
    const int E = c.n_experts, K = c.top_k, I = c.moe_inner;
    const std::size_t np = (std::size_t)n * K;
    bs.moe_logits = f16((std::size_t)n * E);
    bs.moe_eid    = i32(np);
    bs.moe_w      = f16(np);
    bs.moe_act    = f16(np * I);
    bs.moe_part   = f16(np * H);
    bs.moe_out    = f16((std::size_t)n * H);
    bs.moe_ssg    = f16((std::size_t)n * c.moe_shared_inner);
    bs.moe_sout   = f16((std::size_t)n * H);
    bs.moe_gate   = f16((std::size_t)n);
  }
  bs.n = n;
  return !bs.logits.empty() && !bs.pgt.empty();
}

// Bench/experiment knob: VPIPE_QMV_BATCH_MAX_ROWS caps the batched-GEMV row
// range (1..kQmvBatchMaxRows=dflt); rows above the cap fall to the steel GEMM.
// Lets qwen_batched_decode_bench measure the GEMV vs GEMM cost at the SAME m
// (the crossover). Unset -> dflt (no behavior change).
static int
qmv_batch_cap_(int dflt)
{
  static const int v = [dflt] {
    if (const char* e = std::getenv("VPIPE_QMV_BATCH_MAX_ROWS")) {
      const int n = std::atoi(e);
      if (n >= 1 && n <= dflt) { return n; }
    }
    return dflt;
  }();
  return v;
}

void
MetalQwenModel::qmv_ladder_defaults_()
{
  // Static per-m plans = the M5-measured ladder, honoring the env gates.
  // The probe below refines them per machine.
  for (int m = 0; m <= kQmvBatchMaxRows; ++m) {
    _qmv_plan[m] = QmvPlan::kTile2;
  }
  if (_qmv4_enabled) {
    _qmv_plan[3] = _qmv_plan[4] = QmvPlan::kXp4;
  }
  if (_qmv_mix56) {
    _qmv_plan[5] = _qmv_plan[6] = QmvPlan::kMix4R;
  }
  if (_qmv8_enabled) {
    _qmv_plan[7] = _qmv_plan[8] = QmvPlan::kXp8;
  }
}

void
MetalQwenModel::autotune_qmv_ladder_()
{
  if (_qmv_plan_probed) { return; }
  _qmv_plan_probed = true;
  // The probe covers the uniform affine-4bit path only (mixed / k-quant /
  // MoE keep the static defaults); VPIPE_QMV_AUTOTUNE=0 opts out.
  if (const char* e = std::getenv("VPIPE_QMV_AUTOTUNE");
      e != nullptr && std::atoi(e) == 0) {
    return;
  }
  if (_mixed || _kquant || _cfg.is_moe() || !_qmv4_enabled
      || !_fn_qmv_batch.valid() || !_fn_qmv_batch4.valid()) {
    return;
  }
  // Time the THREE single-tile per-pass costs -- T2 (MAXM=2), T4 (xp4),
  // T8 (the tall xp tile) -- on REAL gate|up weights cycled across
  // layers, so every dispatch streams a distinct multi-MB slab and stays
  // DRAM-cold whatever this machine's SLC size (the production regime;
  // a single reused matrix would let a big-SLC chip flatter the
  // re-read-heavy plans). Plan costs are additive in tile times (the
  // encoder serializes dispatches; verified within 1% on M5), so these
  // three numbers resolve the whole m=2..8 ladder.
  const int Kk = _cfg.hidden, Nout = 2 * _cfg.ffn_inner;
  if (Kk <= 0 || Nout <= 0) { return; }
  std::vector<const Layer*> pl;
  for (const auto& ly : _layers) {
    if (!ly.guw.empty() && !ly.gus.empty() && !ly.gub.empty()) {
      pl.push_back(&ly);
      if ((int)pl.size() >= 8) { break; }
    }
  }
  if ((int)pl.size() < 2) { return; }
  SharedBuffer px = _mc->make_shared_buffer((std::size_t)8 * Kk * 2);
  SharedBuffer py = _mc->make_shared_buffer((std::size_t)8 * Nout * 2);
  if (px.empty() || py.empty()) { return; }
  std::memset(px.contents(), 0, px.byte_size());

  struct Cand { const metal_compute::ComputeFunction* fn; int m; };
  std::vector<Cand> cands = {{&_fn_qmv_batch, 2}, {&_fn_qmv_batch4, 4}};
  const bool have8 = _qmv8_enabled && _fn_qmv_batch8_xp.valid();
  if (have8) { cands.push_back({&_fn_qmv_batch8_xp, 8}); }

  // 3 passes over the layer cycle per bench call: the one-off command-
  // stream commit+wait latency then costs only a few us per dispatch, so
  // it no longer inflates the SHORT tiles (T2) relative to the tall ones
  // (a ~20% bias that mis-ranked multi-tile plans on the first cut).
  const int reps = (int)pl.size() * 3;
  const auto t0 = std::chrono::steady_clock::now();
  auto bench = [&](int i) -> double {
    int li = 0;
    return autotune_time(_mc, reps,
        [&](metal_compute::ComputeEncoder& enc) {
          const Layer& ly = *pl[(std::size_t)(li++) % pl.size()];
          enc.set_function(*cands[(std::size_t)i].fn);
          enc.set_buffer(0, ly.guw); enc.set_buffer(1, ly.gus);
          enc.set_buffer(2, ly.gub);
          enc.set_buffer(3, px); enc.set_buffer(4, py);
          enc.set_constant(5, Kk); enc.set_constant(6, Nout);
          enc.set_constant(7, cands[(std::size_t)i].m);
          enc.dispatch({32, (unsigned)(Nout / 4), 1}, {32, 2, 1});
        });
  };
  std::vector<double> us;
  autotune_vote((int)cands.size(), 3, reps, bench, &us);
  if (us.size() < 2 || us[0] <= 0.0 || us[1] <= 0.0) { return; }
  const double t2 = us[0], t4 = us[1];
  const double t8 = (have8 && us.size() > 2 && us[2] > 0.0) ? us[2] : 1e30;

  // Resolve each m's cheapest plan from the additive tile costs.
  for (int m = 2; m <= kQmvBatchMaxRows; ++m) {
    double best = ((m + 1) / 2) * t2;
    QmvPlan p = QmvPlan::kTile2;
    const double c4 = ((m + 3) / 4) * t4;
    if (c4 < best) { best = c4; p = QmvPlan::kXp4; }
    if (m > 4) {
      const double cmix = t4 + ((m - 3) / 2) * t2;
      if (_qmv_mix56 && cmix < best) { best = cmix; p = QmvPlan::kMix4R; }
    }
    if (m > 2 && have8 && t8 < best) { best = t8; p = QmvPlan::kXp8; }
    _qmv_plan[m] = p;
  }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

  static const char* kPlanCh = "24" "8M";   // kTile2,kXp4,kXp8,kMix4R
  char plans[kQmvBatchMaxRows + 2] = {};
  for (int m = 2; m <= kQmvBatchMaxRows; ++m) {
    plans[m - 2] = kPlanCh[(int)_qmv_plan[m]];
  }
  const SessionContextIntf* sess = _mc ? _mc->session() : nullptr;
  if (sess != nullptr) {
    sess->log_debug(fmt(
        "[qwen] qmv-ladder probe {}ms: T2/T4/T8 = {:.0f}/{:.0f}/{} us"
        " -> plans m2..8 [{}] (2=MAXM2 4=xp4 8=xp8 M=mix)",
        (int)(ms + 0.5), t2, t4,
        t8 < 1e29 ? fmt("{:.0f}", us[2])() : std::string("-"), plans));
  }
  if (const char* e = std::getenv("VPIPE_QMV_AUTOTUNE");
      e != nullptr && std::atoi(e) == 2) {
    std::printf("[qmv-ladder] probe %.0fms T2=%.0fus T4=%.0fus T8=%s"
                " -> m2..8 [%s]\n", ms, t2, t4,
                t8 < 1e29 ? fmt("{:.0f}us", us[2])().c_str() : "-", plans);
  }
}

void
MetalQwenModel::qmm_auto_(
    ComputeEncoder& enc, int m, const SharedBuffer& w, const SharedBuffer& s,
    const SharedBuffer& b, const SharedBuffer& xin, const SharedBuffer& y,
    int Kk, int Nout)
{
  // 2..kQmvBatchMaxRows: batched GEMV (weights read once per tile row set;
  // MAXM=2 -> grid.z tiles ceil(m/2)). Decode is DRAM-bound on the weight
  // read, so this beats the steel GEMM (compute-tiled for prefill) until m
  // is large enough that the steel matrix cores win. The per-m PLAN
  // (which tile kernels, homogeneous or mixed) comes from _qmv_plan --
  // static defaults = the M5-measured ladder, refined per machine by the
  // load-time probe (autotune_qmv_ladder_). Every tile kernel is
  // bit-identical per row, so the plan is a pure perf choice
  // (token-exact regardless).
  const int cap = qmv_batch_cap_(kQmvBatchMaxRows);
  if (m > 1 && m <= cap && _fn_qmv_batch.valid()) {
    auto tile = [&](const metal_compute::ComputeFunction& fn,
                    std::size_t row0, int rows,
                    unsigned tiles) {
      enc.set_function(fn);
      enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
      enc.set_buffer(3, xin, row0 * (std::size_t)Kk * 2);
      enc.set_buffer(4, y, row0 * (std::size_t)Nout * 2);
      enc.set_constant(5, Kk); enc.set_constant(6, Nout);
      enc.set_constant(7, rows);
      enc.dispatch({32, (unsigned)(Nout / 4), tiles}, {32, 2, 1});
    };
    switch (_qmv_plan[m]) {
      case QmvPlan::kXp8:
        if (_qmv8_enabled && _fn_qmv_batch8_xp.valid()) {
          tile(_fn_qmv_batch8_xp, 0, m, (unsigned)((m + 7) / 8));
          return;
        }
        break;
      case QmvPlan::kMix4R:
        if (_qmv4_enabled && _fn_qmv_batch4.valid() && m > 4) {
          tile(_fn_qmv_batch4, 0, 4, 1);
          tile(_fn_qmv_batch, 4, m - 4, (unsigned)((m - 3) / 2));
          return;
        }
        break;
      case QmvPlan::kXp4:
        if (_qmv4_enabled && _fn_qmv_batch4.valid()) {
          tile(_fn_qmv_batch4, 0, m, (unsigned)((m + 3) / 4));
          return;
        }
        break;
      case QmvPlan::kTile2:
        break;
    }
    tile(_fn_qmv_batch, 0, m, (unsigned)((m + 1) / 2));
    return;
  }
  if (m == 1) {
    enc.set_function(_fn_qmv);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout);
    enc.dispatch({32, (unsigned)(Nout / 4), 1}, {32, 2, 1});
    return;
  }
  // m == 0 shouldn't happen; large m (or 8-bit, no batched-GEMV) -> steel.
  enc.set_function(_fn_qmm);
  enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
  enc.set_buffer(3, xin); enc.set_buffer(4, y);
  enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, m);
  enc.dispatch({(unsigned)(((Nout + 31) / 32) * 32),
               (unsigned)(((m + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalQwenModel::qmm_auto_swiglu_(
    ComputeEncoder& enc, int m, const SharedBuffer& w, const SharedBuffer& s,
    const SharedBuffer& b, const SharedBuffer& xin, const SharedBuffer& y,
    int Kk, int Nout)
{
  const unsigned gy = (unsigned)(Nout / 4);   // == ffn/2 (fused width 2*ffn)
  const int cap = qmv_batch_cap_(kQmvBatchMaxRows);
  if (m > 1 && m <= cap && _fn_qmv_batch_swiglu.valid()) {
    // Same probed per-m plan as qmm_auto_ (identical weight stream +
    // tile costs; the silu epilogue is noise), on the swiglu twins.
    // y rows stride Nout/2 (the halved fused output).
    auto tile = [&](const metal_compute::ComputeFunction& fn,
                    std::size_t row0, int rows, unsigned tiles) {
      enc.set_function(fn);
      enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
      enc.set_buffer(3, xin, row0 * (std::size_t)Kk * 2);
      enc.set_buffer(4, y, row0 * (std::size_t)(Nout / 2) * 2);
      enc.set_constant(5, Kk); enc.set_constant(6, Nout);
      enc.set_constant(7, rows);
      enc.dispatch({32, gy, tiles}, {32, 2, 1});
    };
    switch (_qmv_plan[m]) {
      case QmvPlan::kXp8:
        if (_qmv8_enabled && _fn_qmv_batch8_xp_swiglu.valid()) {
          tile(_fn_qmv_batch8_xp_swiglu, 0, m, (unsigned)((m + 7) / 8));
          return;
        }
        break;
      case QmvPlan::kMix4R:
        if (_qmv4_enabled && _fn_qmv_batch4_swiglu.valid() && m > 4) {
          tile(_fn_qmv_batch4_swiglu, 0, 4, 1);
          tile(_fn_qmv_batch_swiglu, 4, m - 4, (unsigned)((m - 3) / 2));
          return;
        }
        break;
      case QmvPlan::kXp4:
        if (_qmv4_enabled && _fn_qmv_batch4_swiglu.valid()) {
          tile(_fn_qmv_batch4_swiglu, 0, m, (unsigned)((m + 3) / 4));
          return;
        }
        break;
      case QmvPlan::kTile2:
        break;
    }
    tile(_fn_qmv_batch_swiglu, 0, m, (unsigned)((m + 1) / 2));
    return;
  }
  if (m == 1) {
    enc.set_function(_fn_qmv_swiglu);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout);
    enc.dispatch({32, gy, 1}, {32, 2, 1});
    return;
  }
  enc.set_function(_fn_qmm_swiglu);
  enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
  enc.set_buffer(3, xin); enc.set_buffer(4, y);
  enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, m);
  enc.dispatch({(unsigned)(((Nout + 31) / 32) * 32),
               (unsigned)(((m + 31) / 32) * 2), 2}, {32, 2, 2});
}

// Mixture-of-Experts MLP for M rows (Qwen3.5-MoE). Shared by serial decode
// (M=1), prefill / multimodal (M=n), and batched multi-branch decode (M=N).
// hn = post-attention normed hidden [M,H]; x = residual stream [M,H] updated
// in place. Pairs are [row*top_k + slot]; routing runs on-GPU (no readback).
void
MetalQwenModel::encode_moe_mlp_(
    ComputeEncoder& enc, const Layer& ly, int M, const SharedBuffer& x,
    const SharedBuffer& hn, const SharedBuffer& logits, const SharedBuffer& eid,
    const SharedBuffer& w, const SharedBuffer& act, const SharedBuffer& part,
    const SharedBuffer& moe_out, const SharedBuffer& ssg,
    const SharedBuffer& sout, const SharedBuffer& gate, const MoeGrouped* grp)
{
  const Config& c = _cfg;
  const int H = c.hidden, E = c.n_experts, K = c.top_k, I = c.moe_inner;
  const int Sh = c.moe_shared_inner, np = M * K;
  auto ifill = [&](const SharedBuffer& b, int val, int nb) {
    enc.set_function(_fn_moe_ifill);
    enc.set_buffer(0, b); enc.set_constant(1, val); enc.set_constant(2, nb);
    enc.dispatch({(unsigned)nb, 1, 1}, {256, 1, 1});
  };
  // Bind a weight/scales/biases affine triple to buffers 0/1/2.
  auto sb012 = [&](const SharedBuffer& wb, const SharedBuffer& sb,
                   const SharedBuffer& bb) {
    enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
  };
  // router logits[M, E] = hn @ rgw^T  (qmv for M=1, steel GEMM for M>1; dense
  // f16 raw-HF MoE uses dense_gemv/gemm -- the rest of the routing/combine
  // pipeline below is weight-free and identical to the affine path).
  if (_dense) {
    if (M == 1) {
      dense_gemv_(enc, ly.rgw, hn, 0, logits, 0, H, E);
    } else {
      dense_gemm_(enc, ly.rgw, hn, logits, H, E, M);
    }
  } else if (M == 1) {
    enc.set_function(_fn_qmv8);
    sb012(ly.rgw, ly.rgs, ly.rgb);
    enc.set_buffer(3, hn); enc.set_buffer(4, logits);
    enc.set_constant(5, H); enc.set_constant(6, E);
    enc.dispatch({32, (unsigned)(E / 4), 1}, {32, 2, 1});
  } else {
    enc.set_function(_fn_qmm8);
    sb012(ly.rgw, ly.rgs, ly.rgb);
    enc.set_buffer(3, hn); enc.set_buffer(4, logits);
    enc.set_constant(5, H); enc.set_constant(6, E); enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((E + 31) / 32) * 32),
                 (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  // per-row route -> pair_eid[np], pair_w[np]
  enc.set_function(_fn_moe_route);
  enc.set_buffer(0, logits); enc.set_buffer(1, eid); enc.set_buffer(2, w);
  enc.set_constant(3, E); enc.set_constant(4, K);
  enc.set_constant(5, c.moe_norm_topk ? 1 : 0);
  enc.dispatch({32, 1, (unsigned)M}, {32, 1, 1});
  if (grp == nullptr && _dense) {
    // Dense f16 pair-batched gather: gate|up+SwiGLU -> act[np, I], then down
    // -> partials[np, H] (un-weighted). Grids match the f16 RPS=4/NSG=2 gather
    // kernels: 2*I out rows / 8 per tg for swiglu (I/2 features), H/8 for down.
    enc.set_function(_fn_moe_gather_swiglu_dense);
    enc.set_buffer(0, ly.eguw); enc.set_buffer(1, hn); enc.set_buffer(2, act);
    enc.set_constant(3, H); enc.set_constant(4, 2 * I);
    enc.set_buffer(5, eid); enc.set_constant(6, K);
    // grid.y in THREADS: #tg_y(=I/4) * tg.y(=2) = I/2 (8 out rows / 4 feats
    // per tg over the 2*I interleaved gate|up rows).
    enc.dispatch({32, (unsigned)(I / 2), (unsigned)np}, {32, 2, 1});
    enc.set_function(_fn_moe_gather_down_dense);
    enc.set_buffer(0, ly.edw); enc.set_buffer(1, act); enc.set_buffer(2, part);
    enc.set_constant(3, I); enc.set_constant(4, H); enc.set_buffer(5, eid);
    // grid.y THREADS: #tg_y(=H/8) * 2 = H/4 (8 H rows per tg).
    enc.dispatch({32, (unsigned)(H / 4), (unsigned)np}, {32, 2, 1});
  } else if (grp == nullptr) {
    // Pair-batched: one gathered GEMV per (token,expert) pair.
    // gathered gate|up + SwiGLU -> act[np, I]
    enc.set_function(ly.eg_bits == 8 ? _fn_moe_gather_swiglu_w8
                                     : _fn_moe_gather_swiglu);
    sb012(ly.eguw, ly.egus, ly.egub);
    enc.set_buffer(3, hn); enc.set_buffer(4, act);
    enc.set_constant(5, H); enc.set_constant(6, 2 * I);
    enc.set_buffer(7, eid); enc.set_constant(8, K);
    enc.dispatch({32, (unsigned)(I / 2), (unsigned)np}, {32, 2, 1});
    // gathered down -> partials[np, H]
    enc.set_function(ly.eg_bits == 8 ? _fn_moe_gather_down_w8
                                     : _fn_moe_gather_down);
    sb012(ly.edw, ly.eds, ly.edb);
    enc.set_buffer(3, act); enc.set_buffer(4, part);
    enc.set_constant(5, I); enc.set_constant(6, H); enc.set_buffer(7, eid);
    enc.dispatch({32, (unsigned)(H / 4), (unsigned)np}, {32, 2, 1});
  } else {
    // Grouped: counting-sort pairs by expert -> segmented MAXM-batched GEMV
    // (each expert's weight read once per tile). part[np,H] is filled by
    // scatter-back in pair order (so combine below is unchanged).
    const int mt = grp->maxtiles, npad = grp->npad;
    const bool skip_gs = grp->abl_gs || grp->abl_gemm;   // gather/scatter cost
    if (!grp->abl_sort) {
      ifill(*grp->hist, 0, E);
      ifill(*grp->srow, -1, npad);
      ifill(*grp->sdst, -1, npad);
      ifill(*grp->t2e, -1, mt);
      enc.set_function(_fn_moe_hist);            // hist[e] = #pairs to e
      enc.set_buffer(0, eid); enc.set_buffer(1, *grp->hist);
      enc.set_constant(2, np);
      enc.dispatch({(unsigned)np, 1, 1}, {256, 1, 1});
      enc.set_function(_fn_moe_sort_setup);      // boff/t2e/cursor from hist
      enc.set_buffer(0, *grp->hist); enc.set_buffer(1, *grp->boff);
      enc.set_buffer(2, *grp->t2e); enc.set_buffer(3, *grp->curs);
      enc.set_buffer(4, *grp->ntile);
      enc.set_constant(5, E); enc.set_constant(6, grp->maxm);
      enc.set_constant(7, mt);
      enc.dispatch({1, 1, 1}, {1, 1, 1});
      enc.set_function(_fn_moe_scatter);         // pairs -> sorted slots
      enc.set_buffer(0, eid); enc.set_buffer(1, *grp->boff);
      enc.set_buffer(2, *grp->curs); enc.set_buffer(3, *grp->srow);
      enc.set_buffer(4, *grp->sdst);
      enc.set_constant(5, np); enc.set_constant(6, K);
      enc.dispatch({(unsigned)np, 1, 1}, {256, 1, 1});
    }
    if (grp->steel) {
      // Matrix-tiled grouped steel GEMM (BM=32 tiles, one expert per tile).
      // The row-gather (hn via srow) is fused into the gate|up GEMM loader and
      // the scatter (to partials via sdst) into the down GEMM store -- no
      // explicit gather/scatter copies (the ablation's #1 prefill overhead).
      if (!grp->abl_gemm) {
        // gate|up+SwiGLU steel: gact[npad, I] = gather(hn,srow) @ eguw_e^T
        enc.set_function(ly.eg_bits == 8 ? _fn_moe_qmm_grouped_swiglu_w8
                                         : _fn_moe_qmm_grouped_swiglu);
        sb012(ly.eguw, ly.egus, ly.egub);
        enc.set_buffer(3, hn); enc.set_buffer(4, *grp->gact);
        enc.set_constant(5, H); enc.set_constant(6, 2 * I);
        enc.set_constant(7, npad); enc.set_buffer(8, *grp->t2e);
        enc.set_buffer(9, *grp->srow);
        enc.dispatch({(unsigned)(((2 * I + 31) / 32) * 32),
                     (unsigned)(((npad + 31) / 32) * 2), 2}, {32, 2, 2});
        // down steel: partials[np, H] = gact @ edw_e^T, scattered via sdst
        enc.set_function(ly.eg_bits == 8 ? _fn_moe_qmm_grouped_w8
                                         : _fn_moe_qmm_grouped);
        sb012(ly.edw, ly.eds, ly.edb);
        enc.set_buffer(3, *grp->gact); enc.set_buffer(4, part);
        enc.set_constant(5, I); enc.set_constant(6, H);
        enc.set_constant(7, npad); enc.set_buffer(8, *grp->t2e);
        enc.set_buffer(9, *grp->sdst);
        enc.dispatch({(unsigned)(((H + 31) / 32) * 32),
                     (unsigned)(((npad + 31) / 32) * 2), 2}, {32, 2, 2});
      }
    } else {
      // Bandwidth GEMV grouping (MAXM=2 tiles; medium prefill).
      enc.set_function(ly.eg_bits == 8 ? _fn_moe_grouped_swiglu_w8
                                       : _fn_moe_grouped_swiglu);
      sb012(ly.eguw, ly.egus, ly.egub);
      enc.set_buffer(3, hn); enc.set_buffer(4, *grp->gact);
      enc.set_constant(5, H); enc.set_constant(6, 2 * I);
      enc.set_buffer(7, *grp->srow); enc.set_buffer(8, *grp->t2e);
      enc.dispatch({32, (unsigned)(I / 2), (unsigned)mt}, {32, 2, 1});
      enc.set_function(ly.eg_bits == 8 ? _fn_moe_grouped_down_w8
                                       : _fn_moe_grouped_down);
      sb012(ly.edw, ly.eds, ly.edb);
      enc.set_buffer(3, *grp->gact); enc.set_buffer(4, *grp->gdout);
      enc.set_constant(5, I); enc.set_constant(6, H);
      enc.set_buffer(7, *grp->srow); enc.set_buffer(8, *grp->t2e);
      enc.dispatch({32, (unsigned)(H / 4), (unsigned)mt}, {32, 2, 1});
    }
    // GEMV path writes sorted gdout -> scatter back to pair order (the steel
    // path scatters inside the down GEMM's store, so it skips this).
    if (!grp->steel && !skip_gs) {
      enc.set_function(_fn_moe_scatter_back);
      enc.set_buffer(0, *grp->gdout); enc.set_buffer(1, *grp->sdst);
      enc.set_buffer(2, part); enc.set_constant(3, H);
      enc.dispatch({(unsigned)(npad * H), 1, 1}, {256, 1, 1});
    }
  }
  // combine -> moe_out[M, H]. Fused into finalize below (one dispatch + barrier
  // less per layer) unless VPIPE_QWEN_MOE_FUSE_FINAL=0.
  static const int kFuseFinal = []() {
    const char* e = std::getenv("VPIPE_QWEN_MOE_FUSE_FINAL");
    return (e && std::atoi(e) == 0) ? 0 : 1;
  }();
  const bool fuse_final = kFuseFinal && _fn_moe_finalize_combined.valid();
  if (!fuse_final) {
    enc.set_function(_fn_moe_combine);
    enc.set_buffer(0, part); enc.set_buffer(1, w); enc.set_buffer(2, moe_out);
    enc.set_constant(3, H); enc.set_constant(4, K);
    enc.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
  }
  // shared expert: gate|up SwiGLU -> ssg[M, Sh] (ablation: skip for timing)
  const bool abl_shared = grp && grp->abl_shared;
  if (!abl_shared) {
  if (_dense) {
    // Dense f16 shared expert. gate|up are interleaved [2Sh, H] -> the dense
    // gather-swiglu kernel with a single "expert 0" (eid all-zero is implicit:
    // we pass the slab directly, top_k=1, one pair per row). Simplest correct
    // route: reuse the gather-swiglu kernel over M rows treating each row as
    // its own pair into the single shared slab.
    for (int mrow = 0; mrow < M; ++mrow) {
      enc.set_function(_fn_moe_gather_swiglu_dense);
      enc.set_buffer(0, ly.sguw);
      enc.set_buffer(1, hn, (std::size_t)mrow * H * 2);
      enc.set_buffer(2, ssg, (std::size_t)mrow * Sh * 2);
      enc.set_constant(3, H); enc.set_constant(4, 2 * Sh);
      enc.set_buffer(5, _moe_zero_eid); enc.set_constant(6, 1);
      enc.dispatch({32, (unsigned)(Sh / 2), 1}, {32, 2, 1});
    }
    // shared down -> sout[M, H] (dense GEMV per row / GEMM).
    if (M == 1) {
      dense_gemv_(enc, ly.sdw, ssg, 0, sout, 0, Sh, H);
    } else {
      dense_gemm_(enc, ly.sdw, ssg, sout, Sh, H, M);
    }
    // shared-expert sigmoid gate g[M].
    enc.set_function(_fn_moe_gate_dense);
    enc.set_buffer(0, ly.segw); enc.set_buffer(1, hn); enc.set_buffer(2, gate);
    enc.set_constant(3, H);
    enc.dispatch({32, (unsigned)M, 1}, {32, 1, 1});
  } else {
  if (M == 1) {
    enc.set_function(_fn_qmv_swiglu);
    sb012(ly.sguw, ly.sgus, ly.sgub);
    enc.set_buffer(3, hn); enc.set_buffer(4, ssg);
    enc.set_constant(5, H); enc.set_constant(6, 2 * Sh);
    enc.dispatch({32, (unsigned)(Sh / 2), 1}, {32, 2, 1});
  } else {
    enc.set_function(_fn_qmm_swiglu);
    sb012(ly.sguw, ly.sgus, ly.sgub);
    enc.set_buffer(3, hn); enc.set_buffer(4, ssg);
    enc.set_constant(5, H); enc.set_constant(6, 2 * Sh); enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((2 * Sh + 31) / 32) * 32),
                 (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  // shared down -> sout[M, H]  (qmm_auto_ picks qmv/batched/steel by M)
  qmm_auto_(enc, M, ly.sdw, ly.sds, ly.sdb, ssg, sout, Sh, H);
  // shared-expert sigmoid gate g[M]
  enc.set_function(_fn_moe_gate);
  sb012(ly.segw, ly.segs, ly.segb);
  enc.set_buffer(3, hn); enc.set_buffer(4, gate); enc.set_constant(5, H);
  enc.dispatch({32, (unsigned)M, 1}, {32, 1, 1});
  }
  }
  // finalize: x += combine(part,w) + sigmoid(g) * sout. Fused does the combine
  // inline (no separate moe_combine dispatch/barrier/moe_out buffer).
  if (fuse_final) {
    enc.set_function(_fn_moe_finalize_combined);
    enc.set_buffer(0, x); enc.set_buffer(1, part); enc.set_buffer(2, w);
    enc.set_buffer(3, sout); enc.set_buffer(4, gate);
    enc.set_constant(5, H); enc.set_constant(6, K);
    enc.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
  } else {
    enc.set_function(_fn_moe_finalize);
    enc.set_buffer(0, x); enc.set_buffer(1, moe_out); enc.set_buffer(2, sout);
    enc.set_buffer(3, gate); enc.set_constant(4, H);
    enc.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
  }
}

void
MetalQwenModel::encode_batched_step_(
    ComputeEncoder& enc, BScratch& bs,
    std::span<const ContextId> cids,
    const std::vector<ContextManager::AppendSlot>& slots,
    const std::vector<std::size_t>& page_offs,
    const std::vector<int>& n_pages_v,
    const std::vector<int>& rope_pos_v,
    int shared_pages)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel;
  const int ffn = c.ffn_inner;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const int N = bs.n;
  const int page_tokens = _ctx->page_tokens();
  const std::size_t pt_stride = (std::size_t)_ctx->max_pages() * 3;

  // --- batched helpers (M=N steel GEMM / N-row elementwise) -----------
  auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                 const SharedBuffer& w, const SharedBuffer& y,
                 std::size_t yoff, int R, int Hd) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, Hd); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  // Weight-bound projection (q/k/v/o, gdn in/out, lm_head). Prefer the
  // size-based kernel via the shared qmm_auto_ entrance (qmv / batched
  // GEMV / steel by the active row count N) -- same selector the single
  // decode + prefill conceptually use, so the batched path adapts to the
  // CURRENT (possibly shrunk) branch count exactly like MLX's
  // quantized_matmul.
  auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, const SharedBuffer& xin,
                 const SharedBuffer& y, int Kk, int Nout) {
    qmm_auto_(enc, N, w, s, b, xin, y, Kk, Nout);
  };
  auto hslice = [&](const SharedBuffer& in, std::size_t ioff,
                    const SharedBuffer& out, std::size_t ooff, int Hh,
                    int S, int W, int off, int block = 0, int gstride = 0) {
    enc.set_function(_fn_head_slice);
    enc.set_buffer(0, in, ioff); enc.set_buffer(1, out, ooff);
    enc.set_constant(2, Hh); enc.set_constant(3, S); enc.set_constant(4, W);
    enc.set_constant(5, off); enc.set_constant(6, block);
    enc.set_constant(7, gstride);
    enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
  };
  // Partial RoPE on one branch's q or k block (offset `xoff` bytes, `heads`
  // head-rows of D), at that branch's own position `rp` -- so branches need
  // NOT share a seq_len. The norm is done batched beforehand (rms).
  auto rope = [&](const SharedBuffer& xb, std::size_t xoff, int heads,
                  int rp) {
    enc.set_function(_fn_rope_partial);
    enc.set_buffer(0, xb, xoff); enc.set_buffer(1, _inv_freq);
    enc.set_constant(2, heads); const int one = 1; enc.set_constant(3, one);
    enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
    enc.set_constant(6, rp);
    enc.dispatch({(unsigned)(c.rotary_dim / 2), 1, (unsigned)heads},
                 {(unsigned)(c.rotary_dim / 2), 1, 1});
  };
  auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                      const SharedBuffer& out, int nn) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
    enc.set_constant(3, nn);
    enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
  };

  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    rms(bs.x, 0, ly.in_ln, bs.hn, 0, N, H);
    if (ly.is_full) {
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);
      const bool gate = c.attn_output_gate;
      const int qdo = gate ? 2 * qd : qd;
      const int Nfqkv = qdo + 2 * kd;
      if (_mixed && !ly.qkv_fused) {
        // Mixed-precision de-fuses q|k|v into per-tensor-bit projections (a
        // batched-GEMV can't write a column slice of a fused buffer). q (+gate)
        // -> qfull[N,qdo], k/v -> their own buffers; slice q|gate at stride qdo.
        // (A uniform-base-width layer is fused -> the qmm else below.)
        vqmm_(enc, N, ly.qw, ly.qs, ly.qb, ly.q_bits, bs.hn, bs.qfull, H, qdo);
        vqmm_(enc, N, ly.kw, ly.ks, ly.kb, ly.k_bits, bs.hn, bs.kbuf, H, kd);
        vqmm_(enc, N, ly.vw, ly.vs, ly.vb, ly.v_bits, bs.hn, bs.vbuf, H, kd);
        if (gate) {
          hslice(bs.qfull, 0, bs.q3, 0, N * Hq, 2 * D, D, 0, Hq, qdo);
          hslice(bs.qfull, 0, bs.gate3, 0, N * Hq, 2 * D, D, D, Hq, qdo);
        } else {
          hslice(bs.qfull, 0, bs.q3, 0, N, qdo, qd, 0);
        }
      } else {
        qmm(ly.qw, ly.qs, ly.qb, bs.hn, bs.qfull, H, Nfqkv);
        if (gate) {
          hslice(bs.qfull, 0, bs.q3, 0, N * Hq, 2 * D, D, 0, Hq, Nfqkv);
          hslice(bs.qfull, 0, bs.gate3, 0, N * Hq, 2 * D, D, D, Hq, Nfqkv);
        } else {
          hslice(bs.qfull, 0, bs.q3, 0, N, Nfqkv, qd, 0);
        }
        hslice(bs.qfull, 0, bs.kbuf, 0, N, Nfqkv, kd, qdo);
        hslice(bs.qfull, 0, bs.vbuf, 0, N, Nfqkv, kd, qdo + kd);
      }
      // Batched per-head q/k RMSNorm (position-independent). RoPE is applied
      // per branch below at its own position (no transpose: the decode SDPA
      // reads q/k head-major [Hq|Hkv, D]).
      rms(bs.q3, 0, ly.q_norm, bs.q3, 0, N * Hq, D);
      rms(bs.kbuf, 0, ly.k_norm, bs.kbuf, 0, N * Hkv, D);
      // Per-branch RoPE at this branch's position + write K/V to its slot
      // (cheap, not weight-bound). Branches need NOT share a seq_len.
      //
      // Attention: when the N branches share a prefix (shared_pages>0) and
      // head_dim is 256, read that shared K/V ONCE for all branches (phase A,
      // sdpa_shared_mb256) then merge each branch's private pages (phase B,
      // sdpa_merge_mb256) -- avoids the per-branch re-read of the (large)
      // shared prefix. Otherwise each branch runs its own paged SDPA over its
      // whole page table. SDPA_SHARED_MAXN (=4) caps the batched query rows.
      const bool use_shared =
          _shared_attn && N > 1 && D == 256 && shared_pages > 0 && N <= 4;
      for (int i = 0; i < N; ++i) {
        const std::size_t qoff = (std::size_t)i * qd * 2;
        const std::size_t koff = (std::size_t)i * kd * 2;
        const int pos_i = slots[(std::size_t)i].position;
        rope(bs.q3, qoff, Hq, rope_pos_v[(std::size_t)i]);
        rope(bs.kbuf, koff, Hkv, rope_pos_v[(std::size_t)i]);
        // kv_write k_i, v_i.
        auto kvw = [&](const SharedBuffer& src, std::size_t soff,
                       const SharedBuffer& pool) {
          enc.set_function(_fn_kv_write_paged);
          enc.set_buffer(0, src, soff);
          enc.set_buffer(1, pool, page_offs[(std::size_t)i]);
          enc.set_constant(2, page_tokens); enc.set_constant(3, D);
          const int one = 1; enc.set_constant(4, one);
          const int zero = 0; enc.set_constant(5, zero);
          enc.set_constant(6, slots[(std::size_t)i].slot_offset);
          enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
        };
        kvw(bs.kbuf, koff, kp);
        kvw(bs.vbuf, koff, vp);
        if (use_shared) { continue; }   // attention runs in phase A/B below
        const bool long_ctx = (pos_i + 1) >= _sdpa_mb_min;
        // Long-context per-branch decode: the MLX 2-pass (vec2) beats the all-G
        // mb256 kernel. Uses the shared _d_gqa_oacc scratch, so the N branches'
        // attention serializes -- fine since the common realtime-vqa path is
        // the shared-prefix kernels above, not this fallback.
        if (long_ctx && gqa_vec2_on(D, pos_i) && !_d_gqa_oacc.empty()) {
          encode_gqa_vec2_(enc, bs.q3, qoff, kp, vp, bs.at, qoff, pos_i,
              scale, D, Hq, Hkv, page_tokens, n_pages_v[(std::size_t)i],
              bs.pgt, (std::size_t)i * pt_stride * 4);
          continue;
        }
        const bool use_mb256 = long_ctx && (D == 256);
        const bool use_mb128 =
            long_ctx && (D <= 128) && _fn_sdpa_paged_mb.valid();
        enc.set_function(use_mb256 ? _fn_sdpa_paged_mb256
                                   : (use_mb128 ? _fn_sdpa_paged_mb
                                                : _fn_sdpa_paged));
        enc.set_buffer(0, bs.q3, qoff);
        enc.set_buffer(1, kp); enc.set_buffer(2, vp);
        enc.set_buffer(3, bs.at, qoff);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        const int one = 1; enc.set_constant(8, one);
        enc.set_constant(9, pos_i); enc.set_constant(10, page_tokens);
        enc.set_constant(11, n_pages_v[(std::size_t)i]);
        enc.set_buffer(12, bs.pgt, (std::size_t)i * pt_stride * 4);
        const unsigned w =
            use_mb256 ? 32u * 16u : (use_mb128 ? 32u * 32u : 32u);
        enc.dispatch({w, (unsigned)Hq, 1}, {w, 1, 1});
      }
      if (use_shared) {
        // PHASE A: shared prefix over all N branches, read once. Branch 0's
        // page table (offset 0) is the shared prefix (identical across
        // branches). No causal mask -- the whole prefix precedes every
        // branch's decode position. Writes un-normalized O/m/l to shacc/shm/shl.
        enc.set_function(_fn_sdpa_shared_mb256);
        enc.set_buffer(0, bs.q3);
        enc.set_buffer(1, kp); enc.set_buffer(2, vp);
        enc.set_buffer(3, bs.shacc); enc.set_buffer(4, bs.shm);
        enc.set_buffer(5, bs.shl);
        enc.set_constant(6, scale); enc.set_constant(7, D);
        enc.set_constant(8, Hq); enc.set_constant(9, Hkv);
        enc.set_constant(10, N); enc.set_constant(11, page_tokens);
        enc.set_constant(12, shared_pages);
        enc.set_buffer(13, bs.pgt, 0);
        enc.dispatch({32u * 16u, (unsigned)Hq, 1}, {32u * 16u, 1, 1});
        // PHASE B: per branch -- scan its PRIVATE pages (offset past the
        // shared prefix) and merge with its phase-A shared partial. Each branch
        // writes its own bs.at slice -> the N merges are independent, so run
        // them in one concurrent sub-encoder to overlap.
        auto _bcg = enc.concurrent_scope(decode_batch_concurrent_() && N > 1);
        for (int i = 0; i < N; ++i) {
          const std::size_t qoff = (std::size_t)i * qd * 2;
          const int pos_i = slots[(std::size_t)i].position;
          const int n_priv = n_pages_v[(std::size_t)i] - shared_pages;
          enc.set_function(_fn_sdpa_merge_mb256);
          enc.set_buffer(0, bs.q3, qoff);
          enc.set_buffer(1, kp); enc.set_buffer(2, vp);
          enc.set_buffer(3, bs.at, qoff);
          enc.set_constant(4, scale); enc.set_constant(5, D);
          enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
          const int one = 1; enc.set_constant(8, one);
          enc.set_constant(9, pos_i); enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_priv);
          enc.set_buffer(12, bs.pgt,
              ((std::size_t)i * pt_stride + (std::size_t)shared_pages * 3) * 4);
          enc.set_buffer(13, bs.shacc, (std::size_t)i * qd * sizeof(float));
          enc.set_buffer(14, bs.shm, (std::size_t)i * Hq * sizeof(float));
          enc.set_buffer(15, bs.shl, (std::size_t)i * Hq * sizeof(float));
          enc.dispatch({32u * 16u, (unsigned)Hq, 1}, {32u * 16u, 1, 1});
        }
      }
      if (gate) {
        enc.set_function(_fn_mul_sigmoid);
        enc.set_buffer(0, bs.at); enc.set_buffer(1, bs.gate3);
        enc.set_buffer(2, bs.at); enc.set_constant(3, N * qd);
        enc.dispatch({(unsigned)(N * qd), 1, 1}, {256, 1, 1});
      }
      if (_mixed) {
        vqmm_(enc, N, ly.ow, ly.os, ly.ob, ly.o_bits, bs.at, bs.ao, qd, H);
      } else {
        qmm(ly.ow, ly.os, ly.ob, bs.at, bs.ao, qd, H);
      }
      residual(bs.x, bs.ao, bs.x, N * H);
    } else {
      const int Nf = Cd + vald + 2 * Hv;
      // GDN-input layout / conv x-stride: fused (uniform) packs qkv|z|a|b in one
      // [N,Nf] row (conv reads the qkv part at stride Nf); mixed de-fuses qkv ->
      // its own [N,Cd] buffer (conv stride Cd), z/a/b to their own buffers.
      const int gdn_in_stride = _mixed ? Cd : Nf;
      if (_mixed) {
        vqmm_(enc, N, ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, bs.hn, bs.mixqkv,
              H, Cd);
        vqmm_(enc, N, ly.izw, ly.izs, ly.izb, ly.z_bits, bs.hn, bs.zbuf,
              H, vald);
        vqmm_(enc, N, ly.iaw, ly.ias, ly.iab, ly.a_bits, bs.hn, bs.abuf, H, Hv);
        vqmm_(enc, N, ly.ibw, ly.ibs, ly.ibb, ly.b_bits, bs.hn, bs.bbuf, H, Hv);
      } else {
        qmm(ly.iqw, ly.iqs, ly.iqb, bs.hn, bs.mixqkv, H, Nf);
        hslice(bs.mixqkv, 0, bs.zbuf, 0, N, Nf, vald, Cd);
        hslice(bs.mixqkv, 0, bs.abuf, 0, N, Nf, Hv, Cd + vald);
        hslice(bs.mixqkv, 0, bs.bbuf, 0, N, Nf, Hv, Cd + vald + Hv);
      }
      // Per-branch GDN over each branch's OWN conv/ssm state + disjoint output
      // slices -> the N branches are independent. Phase-interleave the chain
      // (all conv1d+g_beta, then all qk_norm, then all ssm, then all gated_rms),
      // each phase a concurrent sub-encoder, so the branches overlap. Off ->
      // the scopes are no-ops and the phases run serially (same result; deps
      // preserved by the phase order conv1d -> qk_norm -> ssm -> gated_rms).
      const bool bconc = decode_batch_concurrent_() && N > 1;
      const int one = 1;
      const float inv_scale = 1.0f / std::sqrt((float)Dk);
      const float s_q = inv_scale * inv_scale, s_k = inv_scale;
      {   // phase A: conv1d + g_beta (mutually independent, per branch)
        auto _g = enc.concurrent_scope(bconc);
        for (int i = 0; i < N; ++i) {
          const SharedBuffer* cs = _ctx->conv_state(cids[(std::size_t)i], L);
          const std::size_t coff = (std::size_t)i * Cd * 2;
          const std::size_t moff = (std::size_t)i * gdn_in_stride * 2;
          enc.set_function(_fn_gdn_conv1d);
          enc.set_buffer(0, *cs); enc.set_buffer(1, bs.mixqkv, moff);
          enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, bs.convout, coff);
          enc.set_constant(4, one); enc.set_constant(5, Cd);
          enc.set_constant(6, K); enc.set_constant(7, gdn_in_stride);
          enc.set_constant(8, keyd);
          enc.set_buffer(9, *cs);          // in-place (no run-ahead ring here)
          enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
          enc.set_function(_fn_gdn_g_beta);
          enc.set_buffer(0, bs.abuf, (std::size_t)i * Hv * 2);
          enc.set_buffer(1, bs.bbuf, (std::size_t)i * Hv * 2);
          enc.set_buffer(2, ly.A_log); enc.set_buffer(3, ly.dt_bias);
          enc.set_buffer(4, bs.gbuf, (std::size_t)i * Hv * 4);
          enc.set_buffer(5, bs.betabuf, (std::size_t)i * Hv * 4);
          enc.set_constant(6, Hv); enc.set_constant(7, one);
          enc.dispatch({(unsigned)Hv, 1, 1}, {256, 1, 1});
        }
      }
      {   // phase B: qk_norm (reads conv output)
        auto _g = enc.concurrent_scope(bconc);
        for (int i = 0; i < N; ++i) {
          const std::size_t coff = (std::size_t)i * Cd * 2;
          enc.set_function(_fn_gdn_qk_norm);
          enc.set_buffer(0, bs.convout, coff);
          enc.set_constant(1, Hk); enc.set_constant(2, Dk);
          enc.set_constant(3, s_q); enc.set_constant(4, s_k);
          enc.set_constant(5, eps);
          enc.dispatch({128, (unsigned)(2 * Hk), 1}, {128, 1, 1});
        }
      }
      {   // phase C: ssm step (reads qk-normed conv output + g/beta + ssm state)
        auto _g = enc.concurrent_scope(bconc);
        for (int i = 0; i < N; ++i) {
          const SharedBuffer* ss = _ctx->ssm_state(cids[(std::size_t)i], L);
          const std::size_t coff = (std::size_t)i * Cd * 2;
          enc.set_function(_fn_gdn_step);
          enc.set_buffer(0, bs.convout, coff);
          enc.set_buffer(1, bs.convout, coff + (std::size_t)keyd * 2);
          enc.set_buffer(2, bs.convout, coff + (std::size_t)2 * keyd * 2);
          enc.set_buffer(3, bs.gbuf, (std::size_t)i * Hv * 4);
          enc.set_buffer(4, bs.betabuf, (std::size_t)i * Hv * 4);
          enc.set_buffer(5, *ss);
          enc.set_buffer(6, bs.ygdn, (std::size_t)i * vald * 2);
          enc.set_buffer(7, *ss);
          enc.set_constant(8, one);
          enc.set_constant(9, _kquant ? -Hk : Hk);   // strided GQA for GGUF
          enc.set_constant(10, Hv);
          enc.dispatch({32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
        }
      }
      {   // phase D: gated RMSNorm
        auto _g = enc.concurrent_scope(bconc);
        for (int i = 0; i < N; ++i) {
          enc.set_function(_fn_gdn_gated_rms);
          enc.set_buffer(0, bs.ygdn, (std::size_t)i * vald * 2);
          enc.set_buffer(1, ly.gdn_norm);
          enc.set_buffer(2, bs.zbuf, (std::size_t)i * vald * 2);
          enc.set_buffer(3, bs.normout, (std::size_t)i * vald * 2);
          enc.set_constant(4, Dv); enc.set_constant(5, eps);
          enc.dispatch({128, (unsigned)Hv, 1}, {128, 1, 1});
        }
      }
      if (_mixed) {
        vqmm_(enc, N, ly.gow, ly.gos, ly.gob, ly.gout_bits, bs.normout, bs.ao,
              vald, H);
      } else {
        qmm(ly.gow, ly.gos, ly.gob, bs.normout, bs.ao, vald, H);
      }
      residual(bs.x, bs.ao, bs.x, N * H);
    }
    // MLP (batched over ALL N rows -- every branch needs its logits). MoE
    // routes each branch independently (shared helper, M=N); uniform fuses
    // gate|up (one swiglu GEMV); mixed de-fuses gate/up/down per-tensor.
    rms(bs.x, 0, ly.post_ln, bs.hn, 0, N, H);
    if (c.is_moe()) {
      encode_moe_mlp_(enc, ly, N, bs.x, bs.hn, bs.moe_logits, bs.moe_eid,
                      bs.moe_w, bs.moe_act, bs.moe_part, bs.moe_out,
                      bs.moe_ssg, bs.moe_sout, bs.moe_gate);
    } else {
      if (_mixed) {
        if (ly.mlp_fused) {
          qmm_auto_swiglu_(enc, N, ly.guw, ly.gus, ly.gub, bs.hn, bs.sg, H,
                           2 * ffn);
        } else {
          vqmm_(enc, N, ly.guw, ly.gus, ly.gub, ly.gate_bits, bs.hn, bs.sg, H,
                ffn);
          vqmm_(enc, N, ly.uw, ly.us, ly.ub, ly.up_bits, bs.hn, bs.upb, H, ffn);
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, bs.sg); enc.set_buffer(1, bs.upb);
          enc.set_buffer(2, bs.sg); enc.set_constant(3, N * ffn);
          enc.dispatch({(unsigned)(N * ffn), 1, 1}, {256, 1, 1});
        }
        vqmm_(enc, N, ly.dw, ly.ds, ly.db, ly.down_bits, bs.sg, bs.ao, ffn, H);
      } else {
        qmm_auto_swiglu_(enc, N, ly.guw, ly.gus, ly.gub, bs.hn, bs.sg, H,
                         2 * ffn);
        qmm(ly.dw, ly.ds, ly.db, bs.sg, bs.ao, ffn, H);
      }
      residual(bs.x, bs.ao, bs.x, N * H);
    }
  }
  // Final norm + lm_head over ALL N rows.
  rms(bs.x, 0, _final_ln, bs.hn, 0, N, H);
  if (_dense_embed) {
    dense_gemm_(enc, _tied ? _embed_w : _lm_w, bs.hn, bs.logits, H, c.vocab, N);
  } else if (_mixed) {
    const int lb = _tied ? _embed_bits : _lm_bits;
    vqmm_(enc, N, _tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
          _tied ? _embed_b : _lm_b, lb, bs.hn, bs.logits, H, c.vocab);
  } else {
    qmm(_tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
        _tied ? _embed_b : _lm_b, bs.hn, bs.logits, H, c.vocab);
  }
}

bool
MetalQwenModel::decode_batched_step(
    std::span<const ContextId>    cids,
    std::span<const std::int32_t> in_tokens,
    std::span<const std::int32_t> rope_pos,
    std::vector<float>&           out_logits)
{
  const int N = (int)cids.size();
  if (N <= 0 || (int)in_tokens.size() != N) { return false; }
  if (!rope_pos.empty() && (int)rope_pos.size() != N) { return false; }
  if (!_fn_embed.valid()) { return false; }
  // Mixed-precision (OptiQ) is supported: encode_batched_step_ de-fuses the
  // projections per-tensor (vqmm_), like the verify/single decode. The affine
  // embed (_embed_w) is built for mixed too. (k-quant batched is a separate
  // path -- its q6_K embed gather isn't wired into this affine-embed step.)
  if (!ensure_bscratch_(_bdec, N)) { return false; }
  BScratch& bs = _bdec;
  const Config& c = _cfg;
  const int H = c.hidden;
  const std::size_t pt_stride = (std::size_t)_ctx->max_pages() * 3;

  std::vector<ContextManager::AppendSlot> slots((std::size_t)N);
  std::vector<std::size_t> page_offs((std::size_t)N);
  std::vector<int> n_pages_v((std::size_t)N);
  std::vector<int> rope_pos_v((std::size_t)N);
  auto* tok = static_cast<std::int32_t*>(bs.tok_in.contents());
  auto* pgt = static_cast<std::int32_t*>(bs.pgt.contents());
  for (int i = 0; i < N; ++i) {
    slots[(std::size_t)i] = _ctx->append(cids[(std::size_t)i], 1);
    if (!slots[(std::size_t)i].valid()) { return false; }
    page_offs[(std::size_t)i] =
        (std::size_t)slots[(std::size_t)i].page_id.v * _ctx->page_stride_bytes();
    n_pages_v[(std::size_t)i] = _ctx->fill_page_table(
        cids[(std::size_t)i], pgt + (std::size_t)i * pt_stride);
    // rope_pos[i] < 0 (or an empty span) => sequential text decode at the KV
    // slot position; >= 0 => the mROPE-advanced position (post-image).
    rope_pos_v[(std::size_t)i] =
        (rope_pos.empty() || rope_pos[(std::size_t)i] < 0)
            ? slots[(std::size_t)i].position
            : rope_pos[(std::size_t)i];
    tok[i] = in_tokens[(std::size_t)i];
  }
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // In-stream embed gather: N tokens -> bs.x[N, H].
    enc.set_function(_fn_embed);
    enc.set_buffer(0, bs.tok_in); enc.set_buffer(1, _embed_w);
    enc.set_buffer(2, _embed_s); enc.set_buffer(3, _embed_b);
    enc.set_buffer(4, bs.x); enc.set_constant(5, H);
    enc.dispatch({(unsigned)H, (unsigned)N, 1}, {256, 1, 1});
    encode_batched_step_(enc, bs, cids, slots, page_offs, n_pages_v,
                         rope_pos_v,
                         shared_prefix_pages_(pgt, n_pages_v, N, pt_stride));
  }
  stream.commit().wait();
  // [N, vocab] f16 -> host f32 (the caller samples each row).
  out_logits.resize((std::size_t)N * c.vocab);
  read_elt_(bs.logits.contents(), out_logits.data(),
            (std::size_t)N * c.vocab, c.use_bf16);
  return true;
}

std::vector<std::vector<std::int32_t>>
MetalQwenModel::decode_batched_argmax(
    std::span<const ContextId> cids,
    std::span<const std::int32_t> first_tokens, int n_steps)
{
  const int N = (int)cids.size();
  std::vector<std::vector<std::int32_t>> out((std::size_t)N);
  if (N == 0 || (int)first_tokens.size() != N || n_steps <= 0) { return out; }
  const Config& c = _cfg;
  std::vector<std::int32_t> cur(first_tokens.begin(), first_tokens.end());
  std::vector<float> logits;
  for (int s = 0; s < n_steps; ++s) {
    if (!decode_batched_step(
            cids, std::span<const std::int32_t>(cur.data(), cur.size()),
            std::span<const std::int32_t>(), logits)) {
      break;
    }
    for (int i = 0; i < N; ++i) {
      const float* row = logits.data() + (std::size_t)i * c.vocab;
      std::int32_t best = 0; float bv = row[0];
      for (int v = 1; v < c.vocab; ++v) {
        if (row[v] > bv) { bv = row[v]; best = v; }   // ties -> lowest index
      }
      out[(std::size_t)i].push_back(best);
      cur[(std::size_t)i] = best;
    }
  }
  return out;
}

// ---- Batched pipelined decode (bdecode_*) ----------------------------

void
MetalQwenModel::encode_sample_batched_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& gen_ids, int out_idx, int n,
    const GpuSamplerParams& sp, std::uint32_t step_seed)
{
  const int V = _cfg.vocab;
  for (int b = 0; b < n; ++b) {
    // gen_ids is [cap][N] step-major: row out_idx, column b.
    const std::size_t out_off =
        (std::size_t)(out_idx * n + b) * sizeof(std::int32_t);
    // Shared partial/histogram scratch is reused across branches: the serial
    // encoder serialises the WAR hazard, so no per-branch scratch is needed.
    if (sp.greedy) {
      encode_argmax_(enc, _bdec.logits, (std::size_t)b * V * 2,   // 2B / elt
                     gen_ids, out_off, V);
    } else {
      // Per-branch seed so branches don't sample in lock-step.
      const std::uint32_t seed_b =
          step_seed ^ (std::uint32_t)(0x9e3779b9u * (std::uint32_t)(b + 1));
      encode_sample_core_(enc, _bdec.logits, (std::size_t)b * V * 2,
                          gen_ids, out_off, _bdec_sess.sample_ws,
                          (std::size_t)b * V * 2, _bdec_sess.seen,
                          (std::size_t)b * V, sp, seed_b, V);
    }
  }
}

bool
MetalQwenModel::bdecode_begin(std::span<const ContextId> cids,
                              std::span<const std::int32_t> first_tokens,
                              const GpuSamplerParams& sp, int max_tokens,
                              std::span<const std::int32_t> rope_pos)
{
  const int N = (int)cids.size();
  if (N <= 0 || (int)first_tokens.size() != N) { return false; }
  if (!rope_pos.empty() && (int)rope_pos.size() != N) { return false; }
  if (!_fn_embed.valid() || !_fn_argmax.valid()
      || (!sp.greedy && !_fn_sample.valid())) {
    return false;
  }
  // Mixed-precision (OptiQ) supported: encode_batched_step_ de-fuses the
  // projections per-tensor (vqmm_). Shares the encoder with the sync path.
  if (!ensure_bscratch_(_bdec, N)) { return false; }
  if (max_tokens < 1) { max_tokens = 1; }

  BDecode& bd = _bdec_sess;
  bd = BDecode{};
  bd.cids.assign(cids.begin(), cids.end());
  bd.sp = sp;
  bd.n = N;
  // Run-ahead depth (default 2): the batched step's CPU encode (per-branch
  // attention/GDN dispatches x N) plus the driver's emit/stop-check gap are
  // hidden under the in-flight GPU step. Depth-1 restores the old lockstep.
  // No rollback cost: see the BDecode comment (constant-N never rolls back).
  bd.depth = 2;
  if (const char* e = std::getenv("VPIPE_QWEN_BDECODE_DEPTH")) {
    bd.depth = std::max(1, std::min(4, std::atoi(e)));
  }
  bd.cap = max_tokens + 1 + bd.depth;
  bd.produced = 1;                 // row 0 = first_tokens
  bd.committed = 1;                // next commit writes row 1
  bd.rope_base.resize((std::size_t)N);
  bd.kv_base.resize((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    bd.rope_base[(std::size_t)i] =
        (rope_pos.empty() || rope_pos[(std::size_t)i] < 0)
            ? -1 : rope_pos[(std::size_t)i];
    bd.kv_base[(std::size_t)i] = _ctx->seq_len_of(cids[(std::size_t)i]);
  }
  bd.gen_ids = _mc->make_shared_buffer(
      (std::size_t)bd.cap * N * sizeof(std::int32_t));
  if (bd.gen_ids.empty()) { bd = BDecode{}; return false; }
  auto* g = static_cast<std::int32_t*>(bd.gen_ids.contents());
  for (int i = 0; i < N; ++i) { g[i] = first_tokens[(std::size_t)i]; }

  if (!sp.greedy) {
    bd.sample_ws = _mc->make_shared_buffer((std::size_t)N * _cfg.vocab * 2);
    bd.seen = _mc->make_shared_buffer((std::size_t)N * _cfg.vocab);   // uint8
    if (bd.sample_ws.empty() || bd.seen.empty()) { bd = BDecode{}; return false; }
    std::memset(bd.seen.contents(), 0, (std::size_t)N * _cfg.vocab);
    auto* sb = static_cast<std::uint8_t*>(bd.seen.contents());
    for (int i = 0; i < N; ++i) {
      const std::int32_t t = first_tokens[(std::size_t)i];
      if (t >= 0 && t < _cfg.vocab) { sb[(std::size_t)i * _cfg.vocab + t] = 1; }
    }
  }
  bd.stream = _mc->make_command_stream();
  bd.ev = _mc->make_event();
  bd.active = true;
  return true;
}

bool
MetalQwenModel::bdecode_commit()
{
  BDecode& bd = _bdec_sess;
  if (!bd.active || (int)bd.ring.size() >= bd.depth) { return false; }
  const int N = bd.n;
  const int in_idx = bd.committed - 1;
  const int out_idx = bd.committed;
  if (out_idx >= bd.cap) { return false; }
  BScratch& bs = _bdec;
  const int H = _cfg.hidden;
  const std::size_t pt_stride = (std::size_t)_ctx->max_pages() * 3;

  std::vector<ContextManager::AppendSlot> slots((std::size_t)N);
  std::vector<std::size_t> page_offs((std::size_t)N);
  std::vector<int> n_pages_v((std::size_t)N), rope_pos_v((std::size_t)N);
  auto* pgt = static_cast<std::int32_t*>(bs.pgt.contents());
  for (int i = 0; i < N; ++i) {
    slots[(std::size_t)i] = _ctx->append(bd.cids[(std::size_t)i], 1);
    if (!slots[(std::size_t)i].valid()) { return false; }
    page_offs[(std::size_t)i] =
        (std::size_t)slots[(std::size_t)i].page_id.v * _ctx->page_stride_bytes();
    n_pages_v[(std::size_t)i] = _ctx->fill_page_table(
        bd.cids[(std::size_t)i], pgt + (std::size_t)i * pt_stride);
    const int pos = slots[(std::size_t)i].position;
    rope_pos_v[(std::size_t)i] =
        (bd.rope_base[(std::size_t)i] < 0)
            ? pos
            : bd.rope_base[(std::size_t)i] + (pos - bd.kv_base[(std::size_t)i]);
  }

  const std::uint64_t s = bd.gpu_step;
  bd.stream.encode_wait(bd.ev, s);
  {
    ComputeEncoder enc = bd.stream.begin_compute();
    // In-stream batched embed gather: gen_ids row in_idx [N] -> bs.x [N,H].
    enc.set_function(_fn_embed);
    enc.set_buffer(0, bd.gen_ids,
                   (std::size_t)in_idx * N * sizeof(std::int32_t));
    enc.set_buffer(1, _embed_w); enc.set_buffer(2, _embed_s);
    enc.set_buffer(3, _embed_b); enc.set_buffer(4, bs.x);
    enc.set_constant(5, H);
    enc.dispatch({(unsigned)H, (unsigned)N, 1}, {256, 1, 1});

    encode_batched_step_(enc, bs,
                         std::span<const ContextId>(bd.cids.data(),
                                                    bd.cids.size()),
                         slots, page_offs, n_pages_v, rope_pos_v,
                         shared_prefix_pages_(pgt, n_pages_v, N, pt_stride));

    const std::uint32_t step_seed = (std::uint32_t)(
        bd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
    encode_sample_batched_(enc, bd.gen_ids, out_idx, N, bd.sp, step_seed);
  }
  bd.stream.encode_signal(bd.ev, s + 1);
  BDecode::InFlight f;
  f.fence = bd.stream.commit();
  f.idx = out_idx;
  bd.ring.push_back(std::move(f));
  bd.gpu_step = s + 1;
  bd.committed = out_idx + 1;
  return true;
}

bool
MetalQwenModel::bdecode_next(std::vector<std::int32_t>& out_tokens)
{
  BDecode& bd = _bdec_sess;
  if (bd.ring.empty()) { return false; }
  BDecode::InFlight f = std::move(bd.ring.front());
  bd.ring.pop_front();
  f.fence.wait();
  const int N = bd.n;
  out_tokens.resize((std::size_t)N);
  const auto* g = static_cast<const std::int32_t*>(bd.gen_ids.contents());
  for (int i = 0; i < N; ++i) {
    out_tokens[(std::size_t)i] = g[(std::size_t)f.idx * N + i];
  }
  bd.produced = f.idx + 1;
  return true;
}

void
MetalQwenModel::bdecode_end()
{
  // Drain any run-ahead tail. No rollback: the speculative steps' KV/GDN
  // advances are discarded with the branch contexts, exactly like the
  // over-advanced KV/GDN of a branch that stopped before the others.
  BDecode& bd = _bdec_sess;
  for (auto& f : bd.ring) { f.fence.wait(); }
  bd = BDecode{};
}


std::vector<float>
MetalQwenModel::forward(ContextId cid, std::int32_t token_id, int rope_pos)
{
  const Config& c = _cfg;
  const int H = c.hidden;
  if (!ensure_decode_scratch_()) {
    return {};
  }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) {
    return {};
  }
  const int pos = slot.position;
  // RoPE position for the full-attn layers: the KV slot position for text,
  // or an explicit override for post-image decode (where mROPE positions
  // diverge from KV slots).
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  // Embed via the muxer (its own command buffer), then copy into the
  // shared residual stream _d_x that the encode helper consumes.
  std::int32_t id1 = token_id;
  SharedBuffer emb = _muxer->fetch_text(std::span<const std::int32_t>(&id1, 1));
  if (emb.empty()) {
    return {};
  }
  std::memcpy(_d_x.contents(), emb.contents(), (std::size_t)H * 2);

  // Decode profiler (VPIPE_QWEN_DECODE_PROFILE): split CPU encode time
  // from commit+GPU time (decode issues ~500 serial dispatches/token).
  static const bool kProf =
      std::getenv("VPIPE_QWEN_DECODE_PROFILE") != nullptr;
  const auto t_enc0 = std::chrono::steady_clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  long n_disp = 0;
  {
    ComputeEncoder enc = stream.begin_compute();
    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot,
                        _pgtab, 0);
    n_disp = enc.dispatch_count();
  }
  const auto t_enc1 = std::chrono::steady_clock::now();
  stream.commit().wait();
  if (kProf) {
    const auto t_done = std::chrono::steady_clock::now();
    static double enc_ms = 0.0, gpu_ms = 0.0;
    static int cnt = 0;
    enc_ms += std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
    gpu_ms += std::chrono::duration<double, std::milli>(t_done - t_enc1).count();
    if (++cnt % 16 == 0) {
      std::printf("[qwen-decode-prof] encode %.3f ms | commit+gpu %.3f ms "
                  "| dispatches/token %ld (%d steps avg)\n",
                  enc_ms / cnt, gpu_ms / cnt, n_disp, cnt);
    }
  }

  std::vector<float> out((std::size_t)c.vocab);
  read_elt_(_d_logits.contents(), out.data(), (std::size_t)c.vocab,
            _cfg.use_bf16);
  return out;
}

std::vector<float>
MetalQwenModel::prefill(ContextId cid, const std::vector<std::int32_t>& ids)
{
  const int n = (int)ids.size();
  if (n <= 0) { return {}; }
  if (n == 1) { return forward(cid, ids[0]); }   // decode path handles n=1
  SharedBuffer emb =
      _muxer->fetch_text(std::span<const std::int32_t>(ids.data(), ids.size()));
  if (emb.empty()) { return {}; }
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * _cfg.hidden * 2);
  std::memcpy(x.contents(), emb.contents(), (std::size_t)n * _cfg.hidden * 2);
  // MTP prefix seed (opt-in: set_mtp_prefix_seed; VPIPE_MTP_NO_SEED hard-off):
  // ask forward_chunk_ for the per-position post-norm hiddens and remember the
  // prefix ids so mtp_decode can populate the drafter's KV with the prompt.
  static const bool seed_off = std::getenv("VPIPE_MTP_NO_SEED") != nullptr;
  const bool seed = _mtp.ok && _mtp_seed_enabled && !seed_off;
  SharedBuffer allh;
  std::vector<float> r =
      forward_chunk_(cid, x, n, nullptr, nullptr, false, nullptr, false,
                     nullptr, seed ? &allh : nullptr);
  if (seed && !allh.empty()) {
    _mtp_prefix_h = std::move(allh);
    _mtp_prefix_ids = ids;
    _mtp_prefix_len = n;
    _mtp_prefix_valid = true;
  }
  return r;
}

std::vector<float>
MetalQwenModel::prefill_multimodal(ContextId cid,
                                   const std::vector<float>& embeddings,
                                   const std::vector<std::int32_t>& position_ids)
{
  const int H = _cfg.hidden;
  const int n = (int)(embeddings.size() / (std::size_t)H);
  if (n <= 0) { return {}; }
  const bool bf16 = _cfg.use_bf16;
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  for (std::size_t i = 0; i < (std::size_t)n * H; ++i) {
    xp[i] = store_elt(embeddings[i]);
  }
  return prefill_multimodal_buf(cid, std::move(x), position_ids, n);
}

std::vector<float>
MetalQwenModel::prefill_multimodal_buf(
    ContextId cid, SharedBuffer&& x,
    const std::vector<std::int32_t>& position_ids, int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || x.byte_size() < (std::size_t)n * H * 2) { return {}; }
  const bool bf16 = _cfg.use_bf16;
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  // mROPE cos/sin tables [n, rotary_dim] from the 3-axis position_ids +
  // per-pair axis lookup + inv_freq (cat([f,f]) layout: [d]==[d+half]).
  const int rd = _cfg.rotary_dim, half = rd / 2;
  const auto* invf = static_cast<const float*>(_inv_freq.contents());
  SharedBuffer cosb = _mc->make_shared_buffer((std::size_t)n * rd * 2);
  SharedBuffer sinb = _mc->make_shared_buffer((std::size_t)n * rd * 2);
  auto* cp = static_cast<std::uint16_t*>(cosb.contents());
  auto* sp = static_cast<std::uint16_t*>(sinb.contents());
  for (int t = 0; t < n; ++t) {
    for (int d = 0; d < half; ++d) {
      const int axis = (d < (int)_mrope_axis.size()) ? _mrope_axis[d] : 0;
      const float pos = (float)position_ids[(std::size_t)axis * n + t];
      const float ang = pos * invf[d];
      const std::uint16_t cc = store_elt(std::cos(ang));
      const std::uint16_t ss = store_elt(std::sin(ang));
      cp[(std::size_t)t * rd + d] = cc;
      cp[(std::size_t)t * rd + half + d] = cc;
      sp[(std::size_t)t * rd + d] = ss;
      sp[(std::size_t)t * rd + half + d] = ss;
    }
  }
  return forward_chunk_(cid, x, n, &cosb, &sinb);
}

std::vector<float>
MetalQwenModel::prefill_embeddings(ContextId cid,
                                   const std::vector<float>& embeddings, int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || embeddings.size() < (std::size_t)n * H) { return {}; }
  const bool bf16 = _cfg.use_bf16;
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  for (std::size_t i = 0; i < (std::size_t)n * H; ++i) {
    xp[i] = store_elt(embeddings[i]);
  }
  return prefill_embeddings_buf(cid, std::move(x), n);
}

std::vector<float>
MetalQwenModel::prefill_embeddings_buf(ContextId cid, SharedBuffer&& x, int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || x.byte_size() < (std::size_t)n * H * 2) { return {}; }
  // Plain 1-D RoPE over sequential positions (nullptr mrope tables).
  return forward_chunk_(cid, x, n, nullptr, nullptr);
}

metal_compute::SharedBuffer
MetalQwenModel::forward_embeddings_hidden(ContextId cid, const SharedBuffer& x,
                                          int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || x.byte_size() < (std::size_t)n * H * 2) { return {}; }
  SharedBuffer hidden;
  // return_hidden mode: appends to cid's paged KV, runs the layer stack +
  // final norm, and moves the last position's normed hidden into `hidden`.
  // While calibrating, force the full MLP on every layer (verify_all) so the
  // deepest layer's gate-up/down taps cover all positions, not just the last.
  forward_chunk_(cid, x, n, nullptr, nullptr,
                 /*verify_all=*/_calib_on, /*preds_out=*/nullptr,
                 /*return_hidden=*/true, &hidden);
  return hidden;
}

metal_compute::SharedBuffer
MetalQwenModel::forward_embeddings_taps(ContextId cid, const SharedBuffer& x,
                                        int n,
                                        const std::vector<int>& tap_layers)
{
  const int H = _cfg.hidden;
  if (n <= 0 || tap_layers.empty()
      || x.byte_size() < (std::size_t)n * H * 2) {
    return {};
  }
  SharedBuffer taps =
      _mc->make_shared_buffer((std::size_t)tap_layers.size() * n * H * 2);
  if (taps.empty()) { return {}; }
  SharedBuffer hidden;   // discarded; return_hidden just skips the lm_head
  forward_chunk_(cid, x, n, nullptr, nullptr,
                 /*verify_all=*/false, /*preds_out=*/nullptr,
                 /*return_hidden=*/true, &hidden, /*allhidden_out=*/nullptr,
                 &tap_layers, &taps);
  return taps;
}

void
MetalQwenModel::calib_begin()
{
  const int nL = _cfg.n_layers, H = _cfg.hidden, F = _cfg.ffn_inner;
  _calib_qkv.assign((std::size_t)nL, std::vector<float>((std::size_t)H, 0.0f));
  _calib_gu.assign((std::size_t)nL, std::vector<float>((std::size_t)H, 0.0f));
  _calib_dn.assign((std::size_t)nL, std::vector<float>((std::size_t)F, 0.0f));
  _calib_on = true;
}

// ---- Streaming per-layer MoE calibration --------------------------------
void
MetalQwenModel::calib_begin_streaming()
{
  const int nL = _cfg.n_layers, H = _cfg.hidden;
  const int E = _cfg.n_experts, I = _cfg.moe_inner;
  _calib_qkv.assign((std::size_t)nL, std::vector<float>((std::size_t)H, 0.0f));
  _calib_gu.assign((std::size_t)nL, std::vector<float>((std::size_t)H, 0.0f));
  _calib_dn.clear();   // MoE: ffn_inner is 0; the down stats are PER-EXPERT
  _calib_eg.assign((std::size_t)nL,
                   std::vector<float>((std::size_t)E * H, 0.0f));
  _calib_ed.assign((std::size_t)nL,
                   std::vector<float>((std::size_t)E * I, 0.0f));
  _calib_on = true;
}

bool
MetalQwenModel::calib_build_layer(const MetalLlamaWeights& wts, int L,
                                  std::uint64_t* bytes_out, std::string* err)
{
  auto fail = [&](const std::string& m) { if (err) { *err = m; }
                                          return false; };
  if (L < 0 || L >= (int)_layers.size()) { return fail("calib-layer: range"); }
  if (!(_dense && _cfg.is_moe())) {
    return fail("calib-layer: streaming forward supports raw-HF dense MoE only");
  }
  const Config& c = _cfg;
  const bool bf16 = c.use_bf16;
  metal_compute::MetalCompute* mc = _mc;
  std::uint64_t bytes = 0;
  // Narrow a raw bf16/f16/f32 `.weight` to the compute element (2 bytes).
  auto to_elt = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts.info(name);
    if (info == nullptr) { return {}; }
    const std::string want = bf16 ? "BF16" : "F16";
    if (info->dtype == want) { return wts.load(name, mc); }
    SharedBuffer raw = wts.load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<std::uint16_t*>(out.contents());
    auto to16 = [&](float f) -> std::uint16_t {
      if (bf16) { return f32_to_bf16_(f); }
      _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2);
      return b;
    };
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16(bf16_to_f32_(s[i])); }
    } else if (info->dtype == "F16") {
      const auto* s = static_cast<const _Float16*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16((float)s[i]); }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16(s[i]); }
    } else { return {}; }
    return out;
  };
  auto to_f32 = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts.info(name);
    if (info == nullptr) { return {}; }
    if (info->dtype == "F32") { return wts.load(name, mc); }
    SharedBuffer raw = wts.load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 4);
    auto* o = static_cast<float*>(out.contents());
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = bf16_to_f32_(s[i]); }
    } else { return {}; }
    return out;
  };
  auto add_one = [&](SharedBuffer& buf) {
    if (buf.empty()) { return; }
    auto* p = static_cast<std::uint16_t*>(buf.contents());
    const std::size_t n = buf.byte_size() / 2;
    for (std::size_t i = 0; i < n; ++i) {
      if (bf16) { p[i] = f32_to_bf16_(bf16_to_f32_(p[i]) + 1.0f); }
      else {
        _Float16 h; std::memcpy(&h, &p[i], 2);
        h = (_Float16)((float)h + 1.0f); std::memcpy(&p[i], &h, 2);
      }
    }
  };
  auto fuse_f16 = [&](const std::vector<std::string>& names,
                      SharedBuffer& out) -> bool {
    std::vector<SharedBuffer> ws;
    std::size_t total = 0;
    for (const auto& nm : names) {
      SharedBuffer w = to_elt(nm + ".weight");
      if (w.empty()) { return false; }
      total += w.byte_size();
      ws.push_back(std::move(w));
    }
    out = mc->make_shared_buffer(total);
    if (out.empty()) { return false; }
    std::size_t off = 0;
    for (auto& w : ws) {
      std::memcpy((char*)out.contents() + off, w.contents(), w.byte_size());
      off += w.byte_size();
    }
    return true;
  };
  const std::size_t wrowHf = (std::size_t)c.hidden;   // f16 elts per row
  auto interleave_w4_f16 = [&](const SharedBuffer& gw, const SharedBuffer& uw,
                               int rows, SharedBuffer& ow) -> bool {
    ow = mc->make_shared_buffer((std::size_t)2 * rows * wrowHf * 2);
    if (ow.empty() || gw.empty() || uw.empty()) { return false; }
    const auto* gp = static_cast<const std::uint16_t*>(gw.contents());
    const auto* up = static_cast<const std::uint16_t*>(uw.contents());
    auto* op = static_cast<std::uint16_t*>(ow.contents());
    for (std::size_t g = 0; g < (std::size_t)rows; ++g) {
      std::memcpy(op + (2 * g) * wrowHf, gp + g * wrowHf, wrowHf * 2);
      std::memcpy(op + (2 * g + 1) * wrowHf, up + g * wrowHf, wrowHf * 2);
    }
    return true;
  };
  // Interleave a fused [E, 2*rows, H] gate_up slab (gate = rows [0:rows], up =
  // rows [rows:2*rows], the HF concatenated layout) into the [E, 2*rows, H]
  // interleaved layout (row 2g = gate g, 2g+1 = up g) the dense MoE gather
  // kernel reads -- mirroring interleave_moe_f16 but from one fused source.
  auto interleave_gateup_fused = [&](const SharedBuffer& gu, int E, int rows,
                                     SharedBuffer& ow) -> bool {
    ow = mc->make_shared_buffer((std::size_t)E * 2 * rows * wrowHf * 2);
    if (ow.empty() || gu.empty()) { return false; }
    const auto* sp = static_cast<const std::uint16_t*>(gu.contents());
    auto* op = static_cast<std::uint16_t*>(ow.contents());
    const std::size_t slab = (std::size_t)2 * rows * wrowHf;
    for (std::size_t e = 0; e < (std::size_t)E; ++e) {
      const std::size_t gb = e * slab;                  // gate rows [0:rows]
      const std::size_t ub = e * slab + (std::size_t)rows * wrowHf;  // up
      for (std::size_t g = 0; g < (std::size_t)rows; ++g) {
        std::memcpy(op + e * slab + (2 * g) * wrowHf, sp + gb + g * wrowHf,
                    wrowHf * 2);
        std::memcpy(op + e * slab + (2 * g + 1) * wrowHf, sp + ub + g * wrowHf,
                    wrowHf * 2);
      }
    }
    return true;
  };

  const std::string p =
      c.weight_prefix + c.model_seg + "layers." + std::to_string(L) + ".";
  Layer& ly = _layers[(std::size_t)L];
  ly = Layer{};
  ly.is_full = c.layer_is_full(L);
  ly.in_ln = to_elt(p + "input_layernorm.weight");
  ly.post_ln = to_elt(p + "post_attention_layernorm.weight");
  bool ok = !ly.in_ln.empty() && !ly.post_ln.empty();
  add_one(ly.in_ln); add_one(ly.post_ln);
  if (ly.is_full) {
    ok = ok && fuse_f16({p + "self_attn.q_proj", p + "self_attn.k_proj",
                         p + "self_attn.v_proj"}, ly.qw);
    ok = ok && !(ly.ow = to_elt(p + "self_attn.o_proj.weight")).empty();
    ly.q_norm = to_elt(p + "self_attn.q_norm.weight");
    ly.k_norm = to_elt(p + "self_attn.k_norm.weight");
    ok = ok && !ly.q_norm.empty() && !ly.k_norm.empty();
    add_one(ly.q_norm); add_one(ly.k_norm);
  } else {
    ok = ok && fuse_f16({p + "linear_attn.in_proj_qkv",
                         p + "linear_attn.in_proj_z",
                         p + "linear_attn.in_proj_a",
                         p + "linear_attn.in_proj_b"}, ly.iqw);
    ok = ok && !(ly.gow = to_elt(p + "linear_attn.out_proj.weight")).empty();
    ly.conv_w = to_elt(p + "linear_attn.conv1d.weight");
    ly.A_log = to_f32(p + "linear_attn.A_log");
    ly.dt_bias = to_f32(p + "linear_attn.dt_bias");
    ly.gdn_norm = to_elt(p + "linear_attn.norm.weight");   // gated: no +1
    ok = ok && !ly.conv_w.empty() && !ly.A_log.empty() &&
         !ly.dt_bias.empty() && !ly.gdn_norm.empty();
  }
  // Dense f16 MoE MLP.
  const int E = c.n_experts, I = c.moe_inner, S = c.moe_shared_inner;
  ok = ok && !(ly.rgw = to_elt(p + "mlp.gate.weight")).empty();
  {
    // Raw-HF fused experts: gate_up_proj [E,2I,H] (concat gate|up) -> interleave;
    // down_proj [E,H,I] -> read straight (the dense gather kernel's layout).
    SharedBuffer gu = to_elt(p + "mlp.experts.gate_up_proj");
    ok = ok && interleave_gateup_fused(gu, E, I, ly.eguw);
  }
  ok = ok && !(ly.edw = to_elt(p + "mlp.experts.down_proj")).empty();
  {
    SharedBuffer gw = to_elt(p + "mlp.shared_expert.gate_proj.weight");
    SharedBuffer uw = to_elt(p + "mlp.shared_expert.up_proj.weight");
    ok = ok && interleave_w4_f16(gw, uw, S, ly.sguw);
  }
  ok = ok && !(ly.sdw = to_elt(p + "mlp.shared_expert.down_proj.weight")).empty();
  ok = ok && !(ly.segw = to_elt(p + "mlp.shared_expert_gate.weight")).empty();
  if (!ok) {
    std::string miss;
    auto chk = [&](const char* nm, const SharedBuffer& b) {
      if (b.empty()) { miss += nm; miss += " "; }
    };
    chk("in_ln", ly.in_ln); chk("post_ln", ly.post_ln);
    if (ly.is_full) { chk("qw", ly.qw); chk("ow", ly.ow);
                      chk("q_norm", ly.q_norm); chk("k_norm", ly.k_norm); }
    else { chk("iqw", ly.iqw); chk("gow", ly.gow); chk("conv_w", ly.conv_w);
           chk("A_log", ly.A_log); chk("dt_bias", ly.dt_bias);
           chk("gdn_norm", ly.gdn_norm); }
    chk("rgw", ly.rgw); chk("eguw", ly.eguw); chk("edw", ly.edw);
    chk("sguw", ly.sguw); chk("sdw", ly.sdw); chk("segw", ly.segw);
    ly = Layer{};
    return fail("calib-layer: load L=" + std::to_string(L) +
                " empty: [" + miss + "]");
  }
  // Account every resident buffer of this layer.
  auto add = [&](const SharedBuffer& b) { bytes += b.byte_size(); };
  add(ly.in_ln); add(ly.post_ln); add(ly.qw); add(ly.ow); add(ly.q_norm);
  add(ly.k_norm); add(ly.iqw); add(ly.gow); add(ly.conv_w); add(ly.A_log);
  add(ly.dt_bias); add(ly.gdn_norm); add(ly.rgw); add(ly.eguw); add(ly.edw);
  add(ly.sguw); add(ly.sdw); add(ly.segw);
  if (bytes_out) { *bytes_out = bytes; }
  return true;
}

void
MetalQwenModel::calib_free_layer(int L)
{
  if (L >= 0 && L < (int)_layers.size()) { _layers[(std::size_t)L] = Layer{}; }
}

bool
MetalQwenModel::calib_run_layer(int L, std::vector<SharedBuffer>& resid,
                                const std::vector<int>& seq_lens,
                                std::string* err)
{
  auto fail = [&](const std::string& m) { if (err) { *err = m; }
                                          return false; };
  if (L < 0 || L >= (int)_layers.size()) { return fail("calib-run: range"); }
  const Config& c = _cfg;
  const Layer& ly = _layers[(std::size_t)L];
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel;
  const int E = c.n_experts, topk = c.top_k, I = c.moe_inner, Sh = c.moe_shared_inner;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const bool gate = c.attn_output_gate;
  const int qdo = gate ? 2 * qd : qd;
  const int Nfqkv = qdo + 2 * kd;
  const int Nf = Cd + vald + 2 * Hv;

  std::vector<float>& cqkv = _calib_qkv[(std::size_t)L];
  std::vector<float>& cgu  = _calib_gu[(std::size_t)L];
  std::vector<float>& ceg  = _calib_eg[(std::size_t)L];
  std::vector<float>& ced  = _calib_ed[(std::size_t)L];

  for (std::size_t si = 0; si < resid.size(); ++si) {
    const int n = seq_lens[si];
    if (n <= 0) { continue; }
    SharedBuffer& x = resid[si];
    auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
    SharedBuffer hn_attn = buf((std::size_t)n * H);
    SharedBuffer hn_moe  = buf((std::size_t)n * H);
    SharedBuffer ao = buf((std::size_t)n * H);
    // Per-layer-type projection scratch.
    SharedBuffer qfull, q3, gate3, kbuf, vbuf, qt, kt, vt, at, att;
    SharedBuffer mixqkv, zbuf, abuf, bbuf, convout, ygdn, normout;
    SharedBuffer gbuf, betabuf;
    // MoE scratch.
    const int np = n * topk;
    auto i32n = [&](std::size_t e) {
      return _mc->make_shared_buffer(e * sizeof(std::int32_t));
    };
    SharedBuffer moe_logits = buf((std::size_t)n * E);
    SharedBuffer moe_eid = i32n((std::size_t)np);
    SharedBuffer moe_w   = buf((std::size_t)np);
    SharedBuffer moe_act = buf((std::size_t)np * I);
    SharedBuffer moe_part = buf((std::size_t)np * H);
    SharedBuffer moe_out = buf((std::size_t)n * H);
    SharedBuffer moe_ssg = buf((std::size_t)n * Sh);
    SharedBuffer moe_sout = buf((std::size_t)n * H);
    SharedBuffer moe_gate = buf((std::size_t)n);

    // Reserve KV slots for the full-attn sequence (fresh context).
    ContextId cid = _ctx->acquire_root();
    if (!cid.valid()) { return fail("calib-run: acquire_root"); }
    struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
    std::vector<Chunk> chunks;
    int q_offset = -1, page_tokens = _ctx->page_tokens(), n_pages = 0;
    if (ly.is_full) {
      for (int written = 0; written < n; ) {
        const int cap = _ctx->next_append_capacity(cid);
        const int cnt = std::min(n - written, cap);
        ContextManager::AppendSlot s = _ctx->append(cid, cnt);
        if (!s.valid()) { _ctx->release(cid); return fail("calib-run: append"); }
        if (q_offset < 0) { q_offset = s.position; }
        chunks.push_back({(std::size_t)s.page_id.v * _ctx->page_stride_bytes(),
                          s.slot_offset, written, cnt});
        written += cnt;
      }
      if (q_offset < 0) { q_offset = 0; }
      n_pages = _ctx->fill_page_table(
          cid, static_cast<std::int32_t*>(_pgtab.contents()));
      qfull = buf((std::size_t)n * Nfqkv);
      q3 = buf((std::size_t)n * qd); gate3 = buf((std::size_t)n * qd);
      kbuf = buf((std::size_t)n * kd); vbuf = buf((std::size_t)n * kd);
      qt = buf((std::size_t)n * qd); kt = buf((std::size_t)n * kd);
      vt = buf((std::size_t)n * kd);
      at = buf((std::size_t)n * qd); att = buf((std::size_t)n * qd);
    } else {
      mixqkv = buf((std::size_t)n * Nf);
      zbuf = buf((std::size_t)n * vald);
      abuf = buf((std::size_t)n * Hv); bbuf = buf((std::size_t)n * Hv);
      convout = buf((std::size_t)n * Cd);
      ygdn = buf((std::size_t)n * vald); normout = buf((std::size_t)n * vald);
      gbuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
      betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
    }

    metal_compute::CommandStream stream = _mc->make_command_stream();
    {
      ComputeEncoder enc = stream.begin_compute();
      auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                     const SharedBuffer& w, const SharedBuffer& y,
                     std::size_t yoff, int R, int Hd) {
        enc.set_function(_fn_rms);
        enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
        enc.set_buffer(2, y, yoff);
        enc.set_constant(3, Hd); enc.set_constant(4, eps);
        enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      };
      auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out,
                        int Hh, int Sd, int W, int off, int block = 0,
                        int gstride = 0) {
        enc.set_function(_fn_head_slice);
        enc.set_buffer(0, in, 0); enc.set_buffer(1, out, 0);
        enc.set_constant(2, Hh); enc.set_constant(3, Sd); enc.set_constant(4, W);
        enc.set_constant(5, off); enc.set_constant(6, block);
        enc.set_constant(7, gstride);
        enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
      };
      auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                           int A, int Bd) {
        enc.set_function(_fn_transpose);
        enc.set_buffer(0, in); enc.set_buffer(1, out);
        enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
        enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A}, {(unsigned)D, 1, 1});
      };
      auto rope = [&](const SharedBuffer& xb, int heads) {
        enc.set_function(_fn_rope_partial);
        enc.set_buffer(0, xb); enc.set_buffer(1, _inv_freq);
        enc.set_constant(2, heads); enc.set_constant(3, n);
        enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
        enc.set_constant(6, q_offset);
        enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n,
                      (unsigned)heads}, {(unsigned)(c.rotary_dim / 2), 1, 1});
      };
      auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                          const SharedBuffer& out, int nn) {
        enc.set_function(_fn_residual);
        enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
        enc.set_constant(3, nn);
        enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
      };
      auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
        for (const Chunk& ch : chunks) {
          enc.set_function(_fn_kv_write_paged);
          enc.set_buffer(0, src); enc.set_buffer(1, pool, ch.page_off);
          enc.set_constant(2, page_tokens); enc.set_constant(3, D);
          enc.set_constant(4, n); enc.set_constant(5, ch.src_off);
          enc.set_constant(6, ch.slot);
          enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                       {(unsigned)D, 1, 1});
        }
      };

      // x is the residual stream [n, H]; advance it in place.
      rms(x, 0, ly.in_ln, hn_attn, 0, n, H);          // attn input
      if (ly.is_full) {
        const SharedBuffer& kp = *_ctx->kpool(L);
        const SharedBuffer& vp = *_ctx->vpool(L);
        dense_gemm_(enc, ly.qw, hn_attn, qfull, H, Nfqkv, n);
        if (gate) {
          hslice(qfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
          hslice(qfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
        } else {
          hslice(qfull, q3, n, Nfqkv, qd, 0);
        }
        hslice(qfull, kbuf, n, Nfqkv, kd, qdo);
        hslice(qfull, vbuf, n, Nfqkv, kd, qdo + kd);
        rms(q3, 0, ly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, ly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        rope(qt, Hq); rope(kt, Hkv);
        kv_write(kt, kp); kv_write(vt, vp);
        enc.set_function(_fn_sdpa_paged);
        enc.set_buffer(0, qt); enc.set_buffer(1, kp); enc.set_buffer(2, vp);
        enc.set_buffer(3, at);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        enc.set_constant(8, n); enc.set_constant(9, q_offset);
        enc.set_constant(10, page_tokens); enc.set_constant(11, n_pages);
        enc.set_buffer(12, _pgtab);
        enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
        transpose(at, att, Hq, n);
        if (gate) {
          enc.set_function(_fn_mul_sigmoid);
          enc.set_buffer(0, att); enc.set_buffer(1, gate3); enc.set_buffer(2, att);
          enc.set_constant(3, n * qd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        }
        dense_gemm_(enc, ly.ow, att, ao, qd, H, n);
        residual(x, ao, x, n * H);
      } else {
        const SharedBuffer* csb = _ctx->conv_state(cid, L);
        const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
        if (csb == nullptr || ssb == nullptr) {
          _ctx->release(cid); return fail("calib-run: gdn state");
        }
        dense_gemm_(enc, ly.iqw, hn_attn, mixqkv, H, Nf, n);
        hslice(mixqkv, zbuf, n, Nf, vald, Cd);
        hslice(mixqkv, abuf, n, Nf, Hv, Cd + vald);
        hslice(mixqkv, bbuf, n, Nf, Hv, Cd + vald + Hv);
        enc.set_function(_fn_gdn_conv1d);
        enc.set_buffer(0, *csb); enc.set_buffer(1, mixqkv);
        enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
        enc.set_constant(4, n); enc.set_constant(5, Cd); enc.set_constant(6, K);
        enc.set_constant(7, Nf); enc.set_constant(8, keyd);
        enc.set_buffer(9, *csb);
        enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
        const std::size_t kb_off = (std::size_t)n * keyd * 2;
        const std::size_t vb_off = 2 * kb_off;
        rms(convout, 0, _gdn_qscale, convout, 0, n * Hk, Dk);
        rms(convout, kb_off, _gdn_kscale, convout, kb_off, n * Hk, Dk);
        enc.set_function(_fn_gdn_g_beta);
        enc.set_buffer(0, abuf); enc.set_buffer(1, bbuf);
        enc.set_buffer(2, ly.A_log); enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, gbuf); enc.set_buffer(5, betabuf);
        enc.set_constant(6, Hv); enc.set_constant(7, n);
        enc.dispatch({(unsigned)(n * Hv), 1, 1}, {256, 1, 1});
        const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                          !_gdn_force_v1;
        enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
        enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
        enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gbuf);
        enc.set_buffer(4, betabuf); enc.set_buffer(5, *ssb);
        enc.set_buffer(6, ygdn); enc.set_buffer(7, *ssb);
        enc.set_constant(8, n); enc.set_constant(9, Hk); enc.set_constant(10, Hv);
        const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
        enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
        rms(ygdn, 0, ly.gdn_norm, normout, 0, n * Hv, Dv);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, zbuf); enc.set_buffer(1, normout);
        enc.set_buffer(2, normout); enc.set_constant(3, n * vald);
        enc.dispatch({(unsigned)(n * vald), 1, 1}, {256, 1, 1});
        dense_gemm_(enc, ly.gow, normout, ao, vald, H, n);
        residual(x, ao, x, n * H);
      }
      // MoE MLP: post_attn_ln -> router/expert/shared input (tapped) -> add.
      rms(x, 0, ly.post_ln, hn_moe, 0, n, H);
      encode_moe_mlp_(enc, ly, n, x, hn_moe, moe_logits, moe_eid, moe_w,
                      moe_act, moe_part, moe_out, moe_ssg, moe_sout, moe_gate,
                      nullptr);
    }
    stream.commit().wait();
    _ctx->release(cid);

    // ---- host-side accumulation ----------------------------------------
    const auto* ha = static_cast<const _Float16*>(hn_attn.contents());
    const auto* hm = static_cast<const _Float16*>(hn_moe.contents());
    const auto* eid = static_cast<const std::int32_t*>(moe_eid.contents());
    const auto* act = static_cast<const _Float16*>(moe_act.contents());
    for (int t = 0; t < n; ++t) {
      const _Float16* ra = ha + (std::size_t)t * H;
      const _Float16* rm = hm + (std::size_t)t * H;
      for (int d = 0; d < H; ++d) {
        const float va = std::fabs((float)ra[d]);
        if (va > cqkv[(std::size_t)d]) { cqkv[(std::size_t)d] = va; }
        const float vm = std::fabs((float)rm[d]);
        if (vm > cgu[(std::size_t)d]) { cgu[(std::size_t)d] = vm; }
      }
      for (int k = 0; k < topk; ++k) {
        const int e = eid[(std::size_t)t * topk + k];
        if (e < 0 || e >= E) { continue; }
        float* eg = &ceg[(std::size_t)e * H];
        for (int d = 0; d < H; ++d) {
          const float vm = std::fabs((float)rm[d]);
          if (vm > eg[(std::size_t)d]) { eg[(std::size_t)d] = vm; }
        }
        float* ed = &ced[(std::size_t)e * I];
        const _Float16* arow = act + (std::size_t)(t * topk + k) * I;
        for (int i = 0; i < I; ++i) {
          const float v = std::fabs((float)arow[i]);
          if (v > ed[(std::size_t)i]) { ed[(std::size_t)i] = v; }
        }
      }
    }
  }
  return true;
}

const metal_compute::SharedBuffer*
MetalQwenModel::decode_embedding_hidden(
    ContextId cid, const SharedBuffer& emb, int rope_pos,
    const std::function<void(ComputeEncoder&, const SharedBuffer&)>* post_hidden)
{
  const int H = _cfg.hidden;
  if (emb.byte_size() < (std::size_t)H * 2) { return nullptr; }
  if (!ensure_decode_scratch_()) { return nullptr; }
  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return nullptr; }
  const int pos = slot.position;
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));
  // Splice the caller's [H] embedding into the decode residual stream.
  std::memcpy(_d_x.contents(), emb.contents(), (std::size_t)H * 2);

  static const bool kProf =
      std::getenv("VPIPE_QWEN_DECODE_PROFILE") != nullptr;
  const auto t_enc0 = std::chrono::steady_clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot, _pgtab,
                        0, /*return_hidden=*/true);
    // Fuse the caller's dispatches (MOSS heads) into the same command buffer,
    // reading _d_hn which the final norm above wrote (GPU-ordered).
    if (post_hidden != nullptr) { (*post_hidden)(enc, _d_hn); }
  }
  const auto t_enc1 = std::chrono::steady_clock::now();
  stream.commit().wait();
  if (kProf) {
    static double enc_ms = 0.0, gpu_ms = 0.0;
    static int cnt = 0;
    enc_ms += std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
    gpu_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_enc1).count();
    if (++cnt % 16 == 0) {
      std::printf("[moss-decode-prof] encode %.2f ms | commit+gpu %.2f ms "
                  "(%d steps avg)\n", enc_ms / cnt, gpu_ms / cnt, cnt);
    }
  }
  return &_d_hn;
}

SharedBuffer&
MetalQwenModel::decode_input_buffer()
{
  ensure_decode_scratch_();
  return _d_x;
}

const SharedBuffer*
MetalQwenModel::encode_decode_prewritten(ComputeEncoder& enc, ContextId cid,
                                         int rope_pos)
{
  if (!ensure_decode_scratch_()) { return nullptr; }
  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return nullptr; }
  const int pos = slot.position;
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));
  // No host copy: the caller already populated _d_x (decode_input_buffer())
  // for this step (host memcpy or a GPU dispatch earlier in `enc`).
  encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot, _pgtab, 0,
                      /*return_hidden=*/true);
  return &_d_hn;
}

metal_compute::SharedBuffer
MetalQwenModel::embed_text_buf(const std::vector<std::int32_t>& ids)
{
  if (ids.empty() || _muxer == nullptr) { return {}; }
  return _muxer->fetch_text(
      std::span<const std::int32_t>(ids.data(), ids.size()));
}

std::vector<float>
MetalQwenModel::forward_chunk_(ContextId cid, const SharedBuffer& x, int n,
                               const SharedBuffer* mrope_cos,
                               const SharedBuffer* mrope_sin, bool verify_all,
                               std::vector<std::int32_t>* preds_out,
                               bool return_hidden, SharedBuffer* hidden_out,
                               SharedBuffer* allhidden_out,
                               const std::vector<int>* tap_layers,
                               SharedBuffer* taps_out)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel, ffn = c.ffn_inner;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  ensure_decode_scratch_();   // also tunes the decode + prefill attention sets

  // Reserve KV slots (chunked into pages), build the page table.
  struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
  std::vector<Chunk> chunks;
  int q_offset = -1;
  for (int written = 0; written < n; ) {
    const int cap = _ctx->next_append_capacity(cid);
    const int cnt = std::min(n - written, cap);
    ContextManager::AppendSlot s = _ctx->append(cid, cnt);
    if (!s.valid()) { return {}; }
    if (q_offset < 0) { q_offset = s.position; }
    chunks.push_back({(std::size_t)s.page_id.v * _ctx->page_stride_bytes(),
                      s.slot_offset, written, cnt});
    written += cnt;
  }
  if (q_offset < 0) { q_offset = 0; }
  const int page_tokens = _ctx->page_tokens();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  static const bool kPProf =
      std::getenv("VPIPE_QWEN_PREFILL_PROFILE") != nullptr;
  const auto t_p0 = std::chrono::steady_clock::now();
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer hn = buf((std::size_t)n * H);
  // MTP prefix seed: all-position post-norm hiddens [n, H], filled in-band.
  SharedBuffer ah;
  // qfull holds the FUSED q|k|v projection [n, 2*qd+2*kd] (q gated = 2*qd).
  SharedBuffer qfull = buf((std::size_t)n * (2 * qd + 2 * kd));
  SharedBuffer kbuf = buf((std::size_t)n * kd), vbuf = buf((std::size_t)n * kd);
  SharedBuffer q3 = buf((std::size_t)n * qd), gate3 = buf((std::size_t)n * qd);
  // qt holds [Hq, n, D]. The MMA flash prefill (sdpa_paged_mma_d256) reads Q in
  // 8-row simdgroup_matrix tiles, so the last head's last (partial) 32-query
  // tile over-reads up to MMAF_BQ-1 rows past the buffer; pad by 32*head_dim
  // (rows are masked on write, so the padded contents don't matter).
  SharedBuffer qt = buf((std::size_t)n * qd + (std::size_t)32 * D);
  SharedBuffer kt = buf((std::size_t)n * kd), vt = buf((std::size_t)n * kd);
  SharedBuffer at = buf((std::size_t)n * qd), att = buf((std::size_t)n * qd),
               ao = buf((std::size_t)n * H);
  // mixqkv holds the FUSED in_proj output [n, Cd+vald+2*Hv] (qkv|z|a|b).
  SharedBuffer mixqkv = buf((std::size_t)n * (Cd + vald + 2 * Hv));
  SharedBuffer zbuf = buf((std::size_t)n * vald);
  SharedBuffer abuf = buf((std::size_t)n * Hv), bbuf = buf((std::size_t)n * Hv);
  SharedBuffer convout = buf((std::size_t)n * Cd);
  SharedBuffer gq = buf((std::size_t)n * keyd), gk = buf((std::size_t)n * keyd),
               gv = buf((std::size_t)n * vald);
  SharedBuffer gbuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer ygdn = buf((std::size_t)n * vald), normout = buf((std::size_t)n * vald);
  SharedBuffer sg = buf((std::size_t)n * ffn);
  SharedBuffer logits = buf((std::size_t)c.vocab);
  // MoE prefill scratch (on-GPU routing, "pair-batched": npair = n*top_k
  // expert evaluations). ffn is 0 for MoE so sg/kqubuf/gu_full above are
  // empty (unused). Allocated only when is_moe().
  const int moe_np = c.is_moe() ? n * c.top_k : 0;
  auto i32n = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(std::int32_t));
  };
  SharedBuffer moe_logits = c.is_moe()
      ? buf((std::size_t)n * c.n_experts) : SharedBuffer{};
  SharedBuffer moe_eid = c.is_moe() ? i32n((std::size_t)moe_np)
                                    : SharedBuffer{};
  SharedBuffer moe_w = c.is_moe() ? buf((std::size_t)moe_np) : SharedBuffer{};
  SharedBuffer moe_act  = c.is_moe()
      ? buf((std::size_t)moe_np * c.moe_inner) : SharedBuffer{};
  SharedBuffer moe_part = c.is_moe()
      ? buf((std::size_t)moe_np * H) : SharedBuffer{};
  SharedBuffer moe_out  = c.is_moe() ? buf((std::size_t)n * H) : SharedBuffer{};
  SharedBuffer moe_ssg  = c.is_moe()
      ? buf((std::size_t)n * c.moe_shared_inner) : SharedBuffer{};
  SharedBuffer moe_sout = c.is_moe() ? buf((std::size_t)n * H) : SharedBuffer{};
  SharedBuffer moe_gate = c.is_moe() ? buf((std::size_t)n) : SharedBuffer{};
  // Grouped-prefill scratch (counting sort by expert -> per-expert expert
  // GEMM, weight read once per tile). Two tiers: the matrix-tiled STEEL GEMM
  // (n >= 1024; closes the prefill gap vs MLX gather_qmm, pads tiles to BM=32)
  // and the bandwidth GEMV grouping (64 <= n < 1024; MAXM=2). Default-on at
  // n >= 64; VPIPE_QWEN_MOE_GROUPED=1/0 forces on/off, VPIPE_QWEN_MOE_STEEL=1/0
  // forces the steel tier. npad pads each expert's block to the tile (maxm);
  // maxtiles bounds the segmented dispatch.
  const bool moe_grouped = c.is_moe() && !_dense && [&] {
    // Dense f16 MoE has no grouped/steel kernel twins -> always the
    // pair-batched gather path (correctness-first; this is for calibration).
    const char* e = std::getenv("VPIPE_QWEN_MOE_GROUPED");
    if (e) { return std::atoi(e) != 0; }
    return n >= 64;
  }();
  const bool moe_steel = moe_grouped && [&] {
    const char* e = std::getenv("VPIPE_QWEN_MOE_STEEL");
    if (e) { return std::atoi(e) != 0; }
    return n >= _moe_steel_min;          // per-machine crossover (tunable)
  }();
  const int kMoeMAXM = moe_steel ? 32 : 2;   // steel BM tile vs GEMV sweet spot
  // npad is a multiple of the tile size and >= every expert block padded to
  // MAXM (= ceil(np/MAXM) real tiles + one pad tile per expert), so the
  // segmented dispatch (ceil(npad/MAXM) tiles) never indexes past tile2e.
  const int moe_maxtiles = moe_grouped
      ? (moe_np + kMoeMAXM - 1) / kMoeMAXM + c.n_experts : 0;
  const int moe_npad = moe_maxtiles * kMoeMAXM;
  const std::size_t nE = (std::size_t)c.n_experts;
  const std::size_t npd = (std::size_t)moe_npad;
  auto gi = [&](std::size_t e) {   // grouped int buffer (or empty when off)
    return moe_grouped ? i32n(e) : SharedBuffer{};
  };
  SharedBuffer moe_hist = gi(nE), moe_boff = gi(nE), moe_curs = gi(nE);
  SharedBuffer moe_t2e = gi((std::size_t)moe_maxtiles), moe_ntile = gi(1);
  SharedBuffer moe_srow = gi(npd), moe_sdst = gi(npd);
  SharedBuffer moe_gact  = moe_grouped
      ? buf((std::size_t)moe_npad * c.moe_inner) : SharedBuffer{};
  // gdout (sorted down output) is only needed by the GEMV tier; the steel tier
  // scatters straight to partials inside the down GEMM's store.
  SharedBuffer moe_gdout = (moe_grouped && !moe_steel)
      ? buf((std::size_t)moe_npad * H) : SharedBuffer{};
  // MTP batched-verify: per-position logits [n, vocab] + per-position argmax.
  SharedBuffer vlogits = verify_all ? buf((std::size_t)n * c.vocab)
                                    : SharedBuffer{};
  SharedBuffer vamax = verify_all
      ? _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t))
      : SharedBuffer{};
  // Native k-quant prefill: a second [n, ffn] MLP temp (gate -> sg, up ->
  // this, then SwiGLU); the dense GEMM weight scratch _w_deq is shared.
  SharedBuffer kqubuf =
      (_kquant || _mixed || _dense) ? buf((std::size_t)n * ffn)
                                    : SharedBuffer{};
  auto ensure_wdeq = [&](std::size_t elems) {
    if (_w_deq.empty() || _w_deq.byte_size() < elems * 2) {
      _w_deq = _mc->make_shared_buffer(elems * 2);
    }
  };
  // Interleaved gate|up output [n, 2*ffn] for the matrix-core MLP path
  // (dense matmul2d then swiglu_interleaved -> sg). Only when used.
  const bool mma_mlp = _use_mma && n >= _mma_min_m;
  SharedBuffer gu_full =
      mma_mlp ? buf((std::size_t)n * 2 * ffn) : SharedBuffer{};

  // On-device calibration taps: persistent per-layer copies of the qkv /
  // gate-up / down Linear inputs (read host-side after the commit). Only
  // allocated while calibrating; the in-stream copies are guarded too.
  std::vector<SharedBuffer> ctq, ctg, ctd;
  if (_calib_on) {
    ctq.resize((std::size_t)c.n_layers); ctg.resize((std::size_t)c.n_layers);
    ctd.resize((std::size_t)c.n_layers);
    for (int L = 0; L < c.n_layers; ++L) {
      ctq[(std::size_t)L] = buf((std::size_t)n * H);
      ctg[(std::size_t)L] = buf((std::size_t)n * H);
      ctd[(std::size_t)L] = buf((std::size_t)n * ffn);
    }
  }

  // Debug-only per-layer residual dump (env VPIPE_QWEN_LAYER_DUMP=<prefix>):
  // snapshot the [n,H] residual after every layer + the final last-position
  // logits row, then write raw f32 to <prefix>_{embed,layer_L,logits}.bin
  // after the commit. Guarded -> zero impact when the env var is unset.
  static const char* const kLayerDump =
      std::getenv("VPIPE_QWEN_LAYER_DUMP");
  std::vector<SharedBuffer> dbgL;
  SharedBuffer dbgEmbed, dbgLogits;
  if (kLayerDump != nullptr) {
    dbgL.resize((std::size_t)c.n_layers);
    for (int L = 0; L < c.n_layers; ++L) {
      dbgL[(std::size_t)L] = buf((std::size_t)n * H);
    }
    dbgEmbed = buf((std::size_t)n * H);
    dbgLogits = buf((std::size_t)c.vocab);
  }

  const auto t_alloc = std::chrono::steady_clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto tap = [&](const SharedBuffer& src, const SharedBuffer& dst,
                   int count) {
      enc.set_function(_fn_copy);
      enc.set_buffer(0, src); enc.set_buffer(1, dst);
      const int zero = 0;
      enc.set_constant(2, zero); enc.set_constant(3, count);
      enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
    };
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    // Tile-adaptive matrix-core dense GEMM: y[n,N] = xin[n,Kk]@wdeq[N,Kk]^T
    // (no bias). 128x128 tile (8 simdgroups) for K <= 4096; 128x256 for
    // deeper K (down_proj) where the square tile is bandwidth-starved on
    // weight streaming. Both clamp M/N tails via the matmul2d tensor
    // extents, so n and N need not be tile multiples.
    auto dense_mma = [&](const SharedBuffer& xin, const SharedBuffer& wdeq,
                         const SharedBuffer& y, int Kk, int Nn) {
      const bool deep = (Kk >= 6144);
      const int BN = deep ? 256 : 128;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, wdeq);
      enc.set_buffer(2, wdeq);      // bias slot unused (has_bias=0)
      enc.set_buffer(3, y);
      enc.set_constant(4, Kk);
      enc.set_constant(5, Nn);
      enc.set_constant(6, n);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((Nn + BN - 1) / BN) * 256),
                    (unsigned)((n + 127) / 128), 1}, {256, 1, 1});
    };
    auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                   const SharedBuffer& b, const SharedBuffer& xin,
                   const SharedBuffer& y, int Kk, int N) {
      if (s.empty()) {   // dense f16: plain GEMM (rides matrix units via mma)
        dense_gemm_(enc, w, xin, y, Kk, N, n);
        return;
      }
      // Matrix-core path (M5+): dequant the 4-bit weight into a reusable
      // dense scratch, then run the dense matmul2d GEMM on the hardware
      // matrix units. ~2-2.5x the steel quantized GEMM at prefill row
      // counts; M-independent dequant cost amortizes over the n rows.
      // Restricted to enough rows that the dequant pass pays off.
      if (_use_mma && n >= _mma_min_m) {
        const std::size_t need = (std::size_t)N * Kk * 2;  // compute-elt bytes
        if (_w_deq.empty() || _w_deq.byte_size() < need) {
          _w_deq = _mc->make_shared_buffer(need);
        }
        // dequant: (w, s, b) -> _w_deq[N,Kk]  (one thread per weight byte)
        if (!_skip_dequant) {
          enc.set_function(_fn_dequant);
          enc.set_buffer(0, w);
          enc.set_buffer(1, s);
          enc.set_buffer(2, b);
          enc.set_buffer(3, _w_deq);
          enc.set_constant(4, Kk);
          enc.set_constant(5, N);
          enc.dispatch({(unsigned)(Kk / 8), (unsigned)N, 1}, {64, 1, 1});
        }
        // dense matmul2d: y[n,N] = xin[n,Kk] @ _w_deq[N,Kk]^T (no bias) --
        // or the dynamic-int8 accelerated GEMM when enabled + qualifying.
        if (!(_i8 && _i8->gemm(enc, xin, 0, _w_deq, y, 0, n, N, Kk))) {
          dense_mma(xin, _w_deq, y, Kk, N);
        }
        return;
      }
      enc.set_function(_fn_qmm);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, Kk);
      enc.set_constant(6, N);
      enc.set_constant(7, n);
      const unsigned gx = (unsigned)(((N + 31) / 32) * 32);
      const unsigned gy = (unsigned)(((n + 31) / 32) * 2);
      enc.dispatch({gx, gy, 2}, {32, 2, 2});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A}, {(unsigned)D, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb, int heads) {
      enc.set_function(_fn_rope_partial);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads);
      enc.set_constant(3, n);
      enc.set_constant(4, D);
      enc.set_constant(5, c.rotary_dim);
      enc.set_constant(6, q_offset);
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    // Multimodal: table-driven mROPE (cos/sin built from 3-axis positions).
    auto mrope = [&](const SharedBuffer& xb, int heads) {
      enc.set_function(_fn_mrope);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, *mrope_cos);
      enc.set_buffer(2, *mrope_sin);
      enc.set_constant(3, heads);
      enc.set_constant(4, n);
      enc.set_constant(5, D);
      enc.set_constant(6, c.rotary_dim);
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, std::size_t ioff,
                      const SharedBuffer& out, std::size_t ooff, int Hh,
                      int S, int W, int off, int block = 0, int gstride = 0) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in, ioff);
      enc.set_buffer(1, out, ooff);
      enc.set_constant(2, Hh);
      enc.set_constant(3, S);
      enc.set_constant(4, W);
      enc.set_constant(5, off);
      enc.set_constant(6, block);
      enc.set_constant(7, gstride);
      enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
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
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      for (const Chunk& ch : chunks) {
        enc.set_function(_fn_kv_write_paged);
        enc.set_buffer(0, src);
        enc.set_buffer(1, pool, ch.page_off);
        enc.set_constant(2, page_tokens);
        enc.set_constant(3, D);
        enc.set_constant(4, n);             // n_src (row stride of kt/vt)
        enc.set_constant(5, ch.src_off);
        enc.set_constant(6, ch.slot);
        enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };

    // A/B probe (VPIPE_QWEN_SKIP_ATTN): skip a single GPU dispatch to measure
    // its share of prefill (logits become garbage -- timing only; every other
    // op still runs, so the with/without delta is pure kernel time). Modes:
    //   1 (or "all") -- skip the full-attention SDPA (the O(n^2) softmax attn
    //                   on the full layers; the key-split flash-port target)
    //   2 (or "gdn") -- skip the gated-delta step (the linear/GDN-layer
    //                   recurrence) to attribute GDN's prefill cost separately
    // Read per call (NOT static) so an ablation harness can flip it between
    // back-to-back prefills via setenv.
    const int kSkipMode = []() {
      const char* e = std::getenv("VPIPE_QWEN_SKIP_ATTN");
      if (e == nullptr || *e == '\0') { return 0; }
      if (std::strcmp(e, "gdn") == 0 || std::atoi(e) == 2) { return 2; }
      return 1;
    }();

    // MLX steel register-resident flash (head_dim 256), PAGED-KV variant: the
    // full-attention prefill kernel. Online softmax on the register MMATile (no
    // threadgroup score round-trip) with K/V staged straight from the paged
    // pool into tg -- so it serves BOTH fresh (q_offset==0) and mid-context
    // (q_offset>0) prefill with no de-paged kfull/vfull scratch. Closes the
    // long-ctx prefill gap vs omlx the hand-rolled key-split sdpa_paged_flash
    // (~2x roofline) left: MoE +7/+18/+26% @ 2k/4k/8k (beats omlx) and +43% on
    // 2k-on-6k mid-context. half-only kernel -> f16 models (_lib_attn invalid
    // for bf16). Crossover ~1.5k -> default min 2048 (attention too small a
    // slice below it; the lighter key-split flash wins). VPIPE_QWEN_STEEL_ATTN=0
    // forces the paged flash for A/B.
    bool steel_paged = false;
    {
      static const int kSteel = []() {
        const char* e = std::getenv("VPIPE_QWEN_STEEL_ATTN");
        return (e && std::atoi(e) == 0) ? 0 : 1;
      }();
      static const int kSteelMin = []() {
        const char* e = std::getenv("VPIPE_QWEN_STEEL_MIN");
        return (e && std::atoi(e) > 0) ? std::atoi(e) : 2048;
      }();
      steel_paged = kSteel && _lib_attn.valid() && _fn_steel_paged.valid() &&
          D == 256 && n >= kSteelMin && kSkipMode != 1;
    }
    // PRIMARY: the prefill GQA attention SET picks steel/flash/qtile per the
    // chunk's n-regime (steel/flash crossover discovered at load). The legacy
    // steel/non-steel blocks below handle D!=256 or a not-ready set.
    const bool use_pset = (D == 256) && _prefill_set.ready();

    if (kLayerDump != nullptr) { tap(x, dbgEmbed, n * H); }
    for (int L = 0; L < c.n_layers; ++L) {
      Layer& ly = _layers[L];
      rms(x, 0, ly.in_ln, hn, 0, n, H);
      if (_calib_on) { tap(hn, ctq[(std::size_t)L], n * H); }  // qkv input
      if (ly.is_full) {
        const SharedBuffer& kp = *_ctx->kpool(L);
        const SharedBuffer& vp = *_ctx->vpool(L);
        // Single fused q|k|v GEMM. q_proj width is 2*qd (gated, Qwen3.5)
        // or qd (Qwen3-ASR); k/v tails follow at qdo, qdo+kd.
        const bool gate = c.attn_output_gate;
        const int qdo = gate ? 2 * qd : qd;
        const int Nfqkv = qdo + 2 * kd;
        if (_kquant) {
          // q+k (q4_K) + v (q6_K) dequant'd into one f16 scratch, one dense
          // -> qfull[n, Nfqkv] (the exact [q|k|v] fused layout downstream).
          ensure_wdeq((std::size_t)Nfqkv * H);
          if (ly.qk_q4k_aff) {
            adequant_g32_(enc, ly.qk_aw, ly.qk_as, ly.qk_ab, 0, ly.kqk_n, H);
          } else {
            kdequant_(enc, ly.kqk_t, ly.kqk, 0, ly.kqk_n * H);
          }
          kdequant_(enc, ly.kqv_t, ly.kqv, (std::size_t)ly.kqk_n * H, kd * H);
          dense_gemm_(enc, _w_deq, hn, qfull, H, Nfqkv, n);
        } else if (_mixed && !ly.qkv_fused) {
          // q|k|v each its own bits -> one f16 scratch [Nfqkv,H], one dense
          // -> qfull[n, Nfqkv] (the same [q|k|v] layout downstream slices).
          // (A uniform-base-width layer is fused -> the qmm else below.)
          ensure_wdeq((std::size_t)Nfqkv * H);
          adequant_(enc, ly.qw, ly.qs, ly.qb, ly.q_bits, 0, qdo, H);
          adequant_(enc, ly.kw, ly.ks, ly.kb, ly.k_bits,
                    (std::size_t)qdo * H, kd, H);
          adequant_(enc, ly.vw, ly.vs, ly.vb, ly.v_bits,
                    (std::size_t)(qdo + kd) * H, kd, H);
          dense_gemm_(enc, _w_deq, hn, qfull, H, Nfqkv, n);
        } else {
          qmm(ly.qw, ly.qs, ly.qb, hn, qfull, H, Nfqkv);
        }
        if (gate) {
          // q is the gated [n_heads, 2*head_dim] front block; block-strided
          // read extracts q / gate straight from the fused buffer.
          hslice(qfull, 0, q3, 0, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
          hslice(qfull, 0, gate3, 0, n * Hq, 2 * D, D, D, Hq, Nfqkv);
        } else {
          // q is contiguous [n, Hq*D]; plain row-strided slice (stride Nfqkv).
          hslice(qfull, 0, q3, 0, n, Nfqkv, qd, 0);
        }
        // k/v are contiguous tails -> plain row-strided slice into kbuf/vbuf.
        hslice(qfull, 0, kbuf, 0, n, Nfqkv, kd, qdo);
        hslice(qfull, 0, vbuf, 0, n, Nfqkv, kd, qdo + kd);
        rms(q3, 0, ly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, ly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);       // [n,Hq,D] -> [Hq,n,D]
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        if (mrope_cos != nullptr) { mrope(qt, Hq); mrope(kt, Hkv); }
        else { rope(qt, Hq); rope(kt, Hkv); }
        kv_write(kt, kp);
        kv_write(vt, vp);
        if (use_pset && kSkipMode != 1) {
          PrefillGqaAttnSet::Attn pa;
          pa.qt = &qt; pa.kpool = &kp; pa.vpool = &vp; pa.out = &at;
          pa.page_table = &_pgtab;
          pa.n = n; pa.q_offset = q_offset; pa.page_tokens = page_tokens;
          pa.n_pages = n_pages; pa.scale = scale;
          _prefill_set.dispatch(enc, pa);
        } else if (steel_paged) {
          // MLX steel register-softmax flash, K/V staged from the paged pool
          // (serves fresh + mid-context; O written to at [Hq,n,D]).
          enc.set_function(_fn_steel_paged);
          enc.set_buffer(0, qt);
          enc.set_buffer(1, kp);
          enc.set_buffer(2, vp);
          enc.set_buffer(3, at);
          enc.set_constant(4, scale);
          enc.set_constant(5, D);
          enc.set_constant(6, Hq);
          enc.set_constant(7, Hkv);
          enc.set_constant(8, n);
          enc.set_constant(9, q_offset);
          enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_pages);
          enc.set_buffer(12, _pgtab);
          const unsigned nqb = (unsigned)((n + 31) / 32);
          enc.dispatch({32 * nqb, 4 * (unsigned)Hq, 1}, {32, 4, 1});
        } else {
        // Query-tiled flash (head_dim 256): stages each K/V block once and
        // reuses it across a 16-query tile -> ~16x less K/V traffic than the
        // per-query scalar kernel, which is what dominates O(n^2) prefill
        // attention at long context. The tg-staging overhead isn't repaid for
        // short prompts (attention is tiny there, prefill is GEMM-bound), so
        // use it only past a crossover. Same buffer contract.
        // Matrix-core flash attention (M5+, head_dim 256): same buffer
        // contract as the scalar query-tiled kernel, QK^T/P*V on the
        // hardware matrix units -- ~1.4-2.3x the scalar qtile past the
        // crossover. Falls back to qtile (scalar, no matrix cores) / the
        // per-query scalar kernel for short prompts where attention is tiny.
        // Matrix-core (matmul2d) flash: head_dim 256 uses sdpa_mma_f16, 128 the
        // sdpa_mma_d128_f16 instantiation (Llama-3 / Qwen3-VL text encoder).
        // M5-only (gated on _use_mma == supports_matrix_cores()); on M4 this is
        // false and the key-split simdgroup flash below serves D=128 instead.
        const bool mma_attn_d =
            (D == 256 && _fn_sdpa_mma.valid())
            || (D == 128 && _fn_sdpa_mma_d128.valid());
        const bool use_mma_attn =
            _use_mma && mma_attn_d && (n >= _mma_attn_min_n);
        const metal_compute::ComputeFunction& fn_mma_attn =
            (D == 128) ? _fn_sdpa_mma_d128 : _fn_sdpa_mma;
        // Register-resident simdgroup_matrix flash (sdpa_paged_mma_d256), the
        // O-in-registers port of sdpa_causal_mma. MEASURED SLOWER than the
        // key-split sdpa_paged_flash for this shape (head_dim 256): staging BK
        // keys x 256 dims into tg forces BK=16 (4x more blocks/barriers than
        // the flash's BK=64) + 512 threads/~27KB tg (low occupancy), and the
        // key-split flash reads K straight from device (no staging) -- the
        // better fit at large head_dim. ~23% slower prefill @8k. DEFAULT OFF;
        // VPIPE_QWEN_SDPA_PMMA=1 opts in (token-exact, kept for M5/redesign
        // reference -- the lever is a no-staging register-resident hybrid).
        static const int kPmma = []() {
          const char* e = std::getenv("VPIPE_QWEN_SDPA_PMMA");
          return (e && std::atoi(e) == 1) ? 1 : 0;
        }();
        static const int kPmmaMin = []() {
          const char* e = std::getenv("VPIPE_QWEN_SDPA_PMMA_MIN");
          return (e && std::atoi(e) > 0) ? std::atoi(e) : 384;
        }();
        const bool use_pmma = !use_mma_attn && kPmma &&
            _fn_sdpa_paged_mma.valid() && (D == 256) && (n >= kPmmaMin);
        // simdgroup_matrix key-split flash (M4 fallback): preferred over the
        // scalar qtile when neither MMA path is taken. The kernel is
        // head_dim-GENERIC (tg arrays sized to FLP_DMAX=256, D-driven QK/PV
        // loops, O-frags = D/8/FL_NSG), so it also covers head_dim 128 --
        // Llama-3 and the Krea-2 Qwen3-VL text encoder, which previously fell to
        // the scalar per-query sdpa_paged (O(n^2), the long-prompt bottleneck).
        // Only the host gate was Qwen3.5-D256-specific. D=128 additionally needs
        // the page span 64-aligned (FL_C, the flash key block) so the partial-
        // block V tail read stays inside the page allocation; page_tokens=256
        // (the encoder + Llama) satisfies it. Short prompts (n<384, attention
        // tiny, staging not repaid) stay on the scalar kernel for both widths.
        const bool flash_d = (D == 256) || (D == 128 && page_tokens % 64 == 0);
        const bool use_flash = !use_mma_attn && !use_pmma && _flash_attn &&
            _fn_sdpa_paged_flash.valid() && flash_d && (n >= 384);
        const bool use_qt = !use_mma_attn && !use_pmma && !use_flash &&
            (D == 256) && (n >= 384);
        enc.set_function(
            use_mma_attn ? fn_mma_attn
            : (use_pmma ? _fn_sdpa_paged_mma
               : (use_flash ? _fn_sdpa_paged_flash
                  : (use_qt ? _fn_sdpa_paged_qtile : _fn_sdpa_paged))));
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kp);
        enc.set_buffer(2, vp);
        enc.set_buffer(3, at);
        enc.set_constant(4, scale);
        enc.set_constant(5, D);
        enc.set_constant(6, Hq);
        enc.set_constant(7, Hkv);
        enc.set_constant(8, n);
        enc.set_constant(9, q_offset);
        enc.set_constant(10, page_tokens);
        enc.set_constant(11, n_pages);
        enc.set_buffer(12, _pgtab);
        if (kSkipMode != 1) {     // VPIPE_QWEN_SKIP_ATTN full-attn SDPA probe
          if (use_pmma) {
            // MMAF_WM*WD*32 = 4*1*32 = 128 threads (WD=1), MMAF_BQ=32 q/tile.
            const unsigned ntg = (unsigned)((n + 31) / 32);
            enc.dispatch({128, (unsigned)Hq, ntg}, {128, 1, 1});
          } else if (use_flash) {
            // FL_NSG*32 = 256 threads, FL_Q = 8 query rows per tile.
            const unsigned ntg = (unsigned)((n + 7) / 8);
            enc.dispatch({256, (unsigned)Hq, ntg}, {256, 1, 1});
          } else if (use_mma_attn || use_qt) {
            const unsigned n_tiles = (unsigned)((n + 15) / 16);   // BQ=16
            const unsigned tgsz = use_mma_attn ? 128u : 512u;
            enc.dispatch({tgsz, (unsigned)Hq, n_tiles}, {tgsz, 1, 1});
          } else {
            enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
          }
        }
        }  // end else (paged / non-steel attention path)
        transpose(at, att, Hq, n);      // [Hq,n,D] -> [n,Hq,D]
        if (gate) {
          enc.set_function(_fn_mul_sigmoid);
          enc.set_buffer(0, att);
          enc.set_buffer(1, gate3);
          enc.set_buffer(2, att);
          enc.set_constant(3, n * qd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        }
        if (_kquant && ly.o_q4k_aff) {
          aqmm_g32_(enc, ly.o_aw, ly.o_as, ly.o_ab, att, ao, qd, H, n);
        } else if (_kquant) { kqmm_(enc, ly.kqo_t, ly.kqo, att, ao, qd, H, n); }
        else if (_mixed) {
          aqmm_(enc, ly.ow, ly.os, ly.ob, ly.o_bits, att, ao, qd, H, n);
        } else { qmm(ly.ow, ly.os, ly.ob, att, ao, qd, H); }
        residual(x, ao, x, n * H);
      } else {
        const SharedBuffer* csb = _ctx->conv_state(cid, L);
        const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
        // Single fused in_proj GEMM -> mixqkv[n, Nf] laid out qkv|z|a|b.
        const int Nf = Cd + vald + 2 * Hv;
        if (_kquant) {
          // qkv (q5_K) + z (q4_K) dequant'd, a|b (f16) copied, into one f16
          // scratch -> one dense GEMM -> mixqkv[n, Nf] (the fused layout).
          ensure_wdeq((std::size_t)Nf * H);
          kdequant_(enc, ly.kqkv_t, ly.kqkv, 0, Cd * H);
          kdequant_(enc, ly.kqz_t, ly.kqz, (std::size_t)Cd * H, vald * H);
          kcopy_(enc, ly.kqab, (std::size_t)(Cd + vald) * H, 2 * Hv * H);
          dense_gemm_(enc, _w_deq, hn, mixqkv, H, Nf, n);
        } else if (_mixed) {
          // qkv|z|a|b each its own bits -> one f16 scratch [Nf,H], one dense
          // GEMM -> mixqkv[n, Nf] (the fused qkv|z|a|b layout).
          ensure_wdeq((std::size_t)Nf * H);
          adequant_(enc, ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, 0, Cd, H);
          adequant_(enc, ly.izw, ly.izs, ly.izb, ly.z_bits,
                    (std::size_t)Cd * H, vald, H);
          adequant_(enc, ly.iaw, ly.ias, ly.iab, ly.a_bits,
                    (std::size_t)(Cd + vald) * H, Hv, H);
          adequant_(enc, ly.ibw, ly.ibs, ly.ibb, ly.b_bits,
                    (std::size_t)(Cd + vald + Hv) * H, Hv, H);
          dense_gemm_(enc, _w_deq, hn, mixqkv, H, Nf, n);
        } else {
          qmm(ly.iqw, ly.iqs, ly.iqb, hn, mixqkv, H, Nf);
        }
        hslice(mixqkv, 0, zbuf, 0, n, Nf, vald, Cd);
        hslice(mixqkv, 0, abuf, 0, n, Nf, Hv, Cd + vald);
        hslice(mixqkv, 0, bbuf, 0, n, Nf, Hv, Cd + vald + Hv);
        // Conv1d reads the qkv segment (offset 0, row stride Nf) of the fused
        // buffer and writes its output DE-INTERLEAVED into three contiguous
        // blocks q|k|v inside convout (keyd arg), so q/k-norm and the
        // delta-rule step read each block directly -- no hslice copy.
        enc.set_function(_fn_gdn_conv1d);
        enc.set_buffer(0, *csb);
        enc.set_buffer(1, mixqkv);
        enc.set_buffer(2, ly.conv_w);
        enc.set_buffer(3, convout);
        enc.set_constant(4, n);
        enc.set_constant(5, Cd);
        enc.set_constant(6, K);
        enc.set_constant(7, Nf);       // x_stride: qkv is first Cd cols of fused
        enc.set_constant(8, keyd);     // de-interleave q|k|v
        enc.set_buffer(9, *csb);       // in-place (prefill bumps state directly)
        enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
        const std::size_t kb_off = (std::size_t)n * keyd * 2;   // bytes
        const std::size_t vb_off = 2 * kb_off;
        rms(convout, 0, _gdn_qscale, convout, 0, n * Hk, Dk);
        rms(convout, kb_off, _gdn_kscale, convout, kb_off, n * Hk, Dk);
        enc.set_function(_fn_gdn_g_beta);
        enc.set_buffer(0, abuf);
        enc.set_buffer(1, bbuf);
        enc.set_buffer(2, ly.A_log);
        enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, gbuf);
        enc.set_buffer(5, betabuf);
        enc.set_constant(6, Hv);
        enc.set_constant(7, n);
        enc.dispatch({(unsigned)(n * Hv), 1, 1}, {256, 1, 1});
        // Prefill: prefer the ndv4 step (1.33x; 4 dv/simdgroup, k/q read
        // once/step). Token-exact with v1; falls back to v1 if Dv isn't a
        // multiple of 4 or the variant is unavailable.
        const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                          !_gdn_force_v1;
        enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
        enc.set_buffer(0, convout, 0);        // q block
        enc.set_buffer(1, convout, kb_off);   // k block
        enc.set_buffer(2, convout, vb_off);   // v block
        enc.set_buffer(3, gbuf);
        enc.set_buffer(4, betabuf);
        enc.set_buffer(5, *ssb);
        enc.set_buffer(6, ygdn);
        enc.set_buffer(7, *ssb);
        enc.set_constant(8, n);                       // T = n
        enc.set_constant(9, _kquant ? -Hk : Hk);      // strided GQA for GGUF
        enc.set_constant(10, Hv);
        const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
        if (kSkipMode != 2) {     // VPIPE_QWEN_SKIP_ATTN=gdn step probe
          enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
        }
        rms(ygdn, 0, ly.gdn_norm, normout, 0, n * Hv, Dv);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, zbuf);
        enc.set_buffer(1, normout);
        enc.set_buffer(2, normout);
        enc.set_constant(3, n * vald);
        enc.dispatch({(unsigned)(n * vald), 1, 1}, {256, 1, 1});
        if (_kquant) {
          kqmm_(enc, ly.kqout_t, ly.kqout, normout, ao, vald, H, n);
        } else if (_mixed) {
          aqmm_(enc, ly.gow, ly.gos, ly.gob, ly.gout_bits, normout, ao, vald,
                H, n);
        } else {
          qmm(ly.gow, ly.gos, ly.gob, normout, ao, vald, H);
        }
        residual(x, ao, x, n * H);
      }
      // MLP (every layer): fused interleaved gate/up + SwiGLU, then down.
      // The MLP and final norm are position-wise and prefill only consumes
      // the LAST token's logits, so on the final layer we run the MLP for
      // that one row (GEMV path) -- rows 0..n-2 never get the pruned
      // last-layer FFN and no one reads them. Bit-identical for row n-1.
      if (c.is_moe()) {
        // Mixture-of-Experts MLP over all n rows (on-GPU routed). Default is
        // the pair-batched path; VPIPE_QWEN_MOE_GROUPED switches to the sorted
        // grouped path (each expert's weight read once per MAXM-row tile).
        rms(x, 0, ly.post_ln, hn, 0, n, H);            // hn[n, H]
        MoeGrouped grp;
        if (moe_grouped) {
          grp.hist = &moe_hist; grp.boff = &moe_boff; grp.curs = &moe_curs;
          grp.t2e = &moe_t2e; grp.ntile = &moe_ntile; grp.srow = &moe_srow;
          grp.sdst = &moe_sdst; grp.gact = &moe_gact; grp.gdout = &moe_gdout;
          grp.npad = moe_npad; grp.maxtiles = moe_maxtiles; grp.maxm = kMoeMAXM;
          grp.steel = moe_steel;
          if (const char* a = std::getenv("VPIPE_MOE_ABL")) {
            const std::string s(a);
            grp.abl_sort   = s.find("sort")   != std::string::npos;
            grp.abl_gs     = s.find("gs")     != std::string::npos;
            grp.abl_gemm   = s.find("gemm")   != std::string::npos;
            grp.abl_shared = s.find("shared") != std::string::npos;
          }
        }
        encode_moe_mlp_(enc, ly, n, x, hn, moe_logits, moe_eid, moe_w,
                        moe_act, moe_part, moe_out, moe_ssg, moe_sout,
                        moe_gate, moe_grouped ? &grp : nullptr);
      } else if (_kquant) {
        // Native k-quant MLP over all n rows (no last-layer prune; lm_head
        // still consumes only the last position). gate/up are two q4_K
        // dequant+dense into sg / kqubuf, SwiGLU, then down (q6_K).
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        if (ly.ffn_q4k_aff) {
          aqmm_g32_(enc, ly.gate_aw, ly.gate_as, ly.gate_ab, hn, sg, H, ffn, n);
          aqmm_g32_(enc, ly.up_aw, ly.up_as, ly.up_ab, hn, kqubuf, H, ffn, n);
        } else {
          kqmm_(enc, ly.kqgate_t, ly.kqgate, hn, sg, H, ffn, n);
          kqmm_(enc, ly.kqup_t, ly.kqup, hn, kqubuf, H, ffn, n);
        }
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, sg);
        enc.set_buffer(1, kqubuf);
        enc.set_buffer(2, sg);
        enc.set_constant(3, n * ffn);
        enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        kqmm_(enc, ly.kqdown_t, ly.kqdown, sg, ao, ffn, H, n);
        residual(x, ao, x, n * H);
      } else if (_mixed) {
        // Mixed MLP over all n rows (no last-layer prune). Fused (gate|up both
        // w4): the fused swiglu qmm (interleaved guw). De-fused (mixed bits):
        // gate/up each its own bits into sg / kqubuf, then standalone SwiGLU.
        // down always per-tensor (own bits).
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        if (ly.mlp_fused && mma_mlp) {
          // Fused gate|up are both 4-bit (interleaved in guw), so take the
          // SAME matrix-core MLP the uniform path uses: dequant guw -> dense
          // matmul2d -> gu_full[n, 2*ffn], then SwiGLU-combine -> sg. This is
          // what closes the OptiQ/mixed prefill gap vs uniform-4bit on M5 --
          // the FFN is the bulk of the prefill GEMM work, and the steel
          // _fn_qmm_swiglu below (the M4 / small-M fallback) leaves it off the
          // matrix units. Byte-identical weight/layout to the uniform branch,
          // so it stays token-exact (verified vs the steel path).
          const int N2 = 2 * ffn;
          const std::size_t need = (std::size_t)N2 * H * 2;
          if (_w_deq.empty() || _w_deq.byte_size() < need) {
            _w_deq = _mc->make_shared_buffer(need);
          }
          if (!_skip_dequant) {
            enc.set_function(_fn_dequant);
            enc.set_buffer(0, ly.guw); enc.set_buffer(1, ly.gus);
            enc.set_buffer(2, ly.gub); enc.set_buffer(3, _w_deq);
            enc.set_constant(4, H); enc.set_constant(5, N2);
            enc.dispatch({(unsigned)(H / 8), (unsigned)N2, 1}, {64, 1, 1});
          }
          dense_mma(hn, _w_deq, gu_full, H, N2);   // K=H (<=4096) -> 128x128
          enc.set_function(_fn_swiglu_inter);
          enc.set_buffer(0, gu_full); enc.set_buffer(1, sg);
          enc.set_constant(2, n); enc.set_constant(3, ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        } else if (ly.mlp_fused) {
          enc.set_function(_fn_qmm_swiglu);
          enc.set_buffer(0, ly.guw);
          enc.set_buffer(1, ly.gus);
          enc.set_buffer(2, ly.gub);
          enc.set_buffer(3, hn);
          enc.set_buffer(4, sg);
          enc.set_constant(5, H);
          enc.set_constant(6, 2 * ffn);
          enc.set_constant(7, n);
          const unsigned gx = (unsigned)(((2 * ffn + 31) / 32) * 32);
          const unsigned gy = (unsigned)(((n + 31) / 32) * 2);
          enc.dispatch({gx, gy, 2}, {32, 2, 2});
        } else {
          aqmm_(enc, ly.guw, ly.gus, ly.gub, ly.gate_bits, hn, sg, H, ffn, n);
          aqmm_(enc, ly.uw, ly.us, ly.ub, ly.up_bits, hn, kqubuf, H, ffn, n);
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, sg);
          enc.set_buffer(1, kqubuf);
          enc.set_buffer(2, sg);
          enc.set_constant(3, n * ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        }
        aqmm_(enc, ly.dw, ly.ds, ly.db, ly.down_bits, sg, ao, ffn, H, n);
        residual(x, ao, x, n * H);
      } else if (_dense) {
        // Dense f16 MLP over all n rows: gate/up two dense GEMMs into
        // sg / kqubuf, SwiGLU, then down GEMM + residual. (No last-layer
        // prune -- lm_head still reads only the last row.)
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        qmm(ly.guw, ly.gus, ly.gub, hn, sg, H, ffn);
        qmm(ly.uw, ly.us, ly.ub, hn, kqubuf, H, ffn);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, sg);
        enc.set_buffer(1, kqubuf);
        enc.set_buffer(2, sg);
        enc.set_constant(3, n * ffn);
        enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        qmm(ly.dw, ly.ds, ly.db, sg, ao, ffn, H);
        residual(x, ao, x, n * H);
      } else if (L + 1 < c.n_layers || verify_all) {
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        if (_calib_on) { tap(hn, ctg[(std::size_t)L], n * H); }  // gate/up in
        if (mma_mlp) {
          // Matrix-core MLP: dequant interleaved gate|up weight -> dense
          // matmul2d -> gu_full[n, 2*ffn], then SwiGLU-combine -> sg[n,ffn].
          const int N2 = 2 * ffn;
          const std::size_t need = (std::size_t)N2 * H * 2;
          if (_w_deq.empty() || _w_deq.byte_size() < need) {
            _w_deq = _mc->make_shared_buffer(need);
          }
          if (!_skip_dequant) {
            enc.set_function(_fn_dequant);
            enc.set_buffer(0, ly.guw); enc.set_buffer(1, ly.gus);
            enc.set_buffer(2, ly.gub); enc.set_buffer(3, _w_deq);
            enc.set_constant(4, H); enc.set_constant(5, N2);
            enc.dispatch({(unsigned)(H / 8), (unsigned)N2, 1}, {64, 1, 1});
          }
          dense_mma(hn, _w_deq, gu_full, H, N2);   // K=H (<=4096) -> 128x128
          enc.set_function(_fn_swiglu_inter);
          enc.set_buffer(0, gu_full); enc.set_buffer(1, sg);
          enc.set_constant(2, n); enc.set_constant(3, ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        } else {
          enc.set_function(_fn_qmm_swiglu);
          enc.set_buffer(0, ly.guw);
          enc.set_buffer(1, ly.gus);
          enc.set_buffer(2, ly.gub);
          enc.set_buffer(3, hn);
          enc.set_buffer(4, sg);
          enc.set_constant(5, H);
          enc.set_constant(6, 2 * ffn);
          enc.set_constant(7, n);
          const unsigned gx = (unsigned)(((2 * ffn + 31) / 32) * 32);
          const unsigned gy = (unsigned)(((n + 31) / 32) * 2);
          enc.dispatch({gx, gy, 2}, {32, 2, 2});
        }
        if (_calib_on) { tap(sg, ctd[(std::size_t)L], n * ffn); }  // down in
        qmm(ly.dw, ly.ds, ly.db, sg, ao, ffn, H);
        residual(x, ao, x, n * H);
      } else {
        const std::size_t roff = (std::size_t)(n - 1) * H * 2;  // bytes
        rms(x, roff, ly.post_ln, hn, 0, 1, H);   // hn[0] = norm(x[n-1])
        enc.set_function(_fn_qmv_swiglu);
        enc.set_buffer(0, ly.guw);
        enc.set_buffer(1, ly.gus);
        enc.set_buffer(2, ly.gub);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, sg);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * ffn);
        enc.dispatch({32, (unsigned)(ffn / 2), 1}, {32, 2, 1});
        enc.set_function(_fn_qmv_add);          // x[n-1] += down(sg)
        enc.set_buffer(0, ly.dw);
        enc.set_buffer(1, ly.ds);
        enc.set_buffer(2, ly.db);
        enc.set_buffer(3, sg);
        enc.set_buffer(4, x, roff);
        enc.set_constant(5, ffn);
        enc.set_constant(6, H);
        enc.set_buffer(7, x, roff);
        enc.dispatch({32, (unsigned)(H / 4), 1}, {32, 2, 1});
      }
      if (kLayerDump != nullptr) { tap(x, dbgL[(std::size_t)L], n * H); }
      // Krea-2 encoder: snapshot the residual after this layer into the
      // requested tap slots (un-normed [n,H] == HF hidden_states[L+1]).
      if (tap_layers != nullptr && taps_out != nullptr) {
        for (std::size_t j = 0; j < tap_layers->size(); ++j) {
          if ((*tap_layers)[j] != L) { continue; }
          enc.set_function(_fn_copy);
          enc.set_buffer(0, x);
          enc.set_buffer(1, *taps_out, (std::size_t)j * n * H * 2);
          const int zero = 0, count = n * H;
          enc.set_constant(2, zero);
          enc.set_constant(3, count);
          enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
        }
      }
    }

    if (verify_all && preds_out) {
      // MTP batched verify: final-norm ALL n rows, lm_head over the whole
      // [n, H] stack (steel GEMM M=n -> weights read ONCE for all drafts),
      // then a per-row argmax -> the per-position greedy predictions.
      rms(x, 0, _final_ln, hn, 0, n, H);
      if (_dense || _dense_embed) {
        dense_gemm_(enc, _tied ? _embed_w : _lm_w, hn, vlogits, H,
                    c.vocab, n);
      } else {
      const int lb = _tied ? _embed_bits : _lm_bits;
      enc.set_function((_mixed && lb == 8) ? _fn_qmm8 : _fn_qmm);
      enc.set_buffer(0, _tied ? _embed_w : _lm_w);
      enc.set_buffer(1, _tied ? _embed_s : _lm_s);
      enc.set_buffer(2, _tied ? _embed_b : _lm_b);
      enc.set_buffer(3, hn);
      enc.set_buffer(4, vlogits);
      enc.set_constant(5, H);
      enc.set_constant(6, c.vocab);
      enc.set_constant(7, n);
      enc.dispatch({(unsigned)(((c.vocab + 31) / 32) * 32),
                   (unsigned)(((n + 31) / 32) * 2), 2}, {32, 2, 2});
      }
      for (int k = 0; k < n; ++k) {
        encode_argmax_(enc, vlogits, (std::size_t)k * c.vocab * 2, vamax,
                       (std::size_t)k * sizeof(std::int32_t), c.vocab);
      }
    } else if (return_hidden) {
      // MOSS-TTS: final-norm the last position into hn[0:H]; the caller
      // applies its own output heads. No lm_head, no logit pull.
      rms(x, (std::size_t)(n - 1) * H * 2, _final_ln, hn, 0, 1, H);
    } else {
      // Final norm + lm_head on the last position only.
      rms(x, (std::size_t)(n - 1) * H * 2, _final_ln, hn, 0, 1, H);
      if (_kquant) {
        kqmv_(enc, lm_head_kqt_(), lm_head_kq_(), hn, 0, logits, 0, H, c.vocab);
      } else if (_dense || _dense_embed) {
        dense_gemv_(enc, _tied ? _embed_w : _lm_w, hn, 0, logits, 0, H,
                    c.vocab);
      } else {
        const int lb = _tied ? _embed_bits : _lm_bits;
        enc.set_function((_mixed && lb == 8) ? _fn_qmv8 : _fn_qmv);
        enc.set_buffer(0, _tied ? _embed_w : _lm_w);
        enc.set_buffer(1, _tied ? _embed_s : _lm_s);
        enc.set_buffer(2, _tied ? _embed_b : _lm_b);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, logits);
        enc.set_constant(5, H);
        enc.set_constant(6, c.vocab);
        enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
      }
      if (kLayerDump != nullptr) { tap(logits, dbgLogits, c.vocab); }
    }
    // MTP prefix seed: final-norm ALL n positions into `ah` (the per-position
    // post-norm hiddens). Encoded in this same buffer so no extra round-trip.
    if (allhidden_out) {
      ah = buf((std::size_t)n * H);
      rms(x, 0, _final_ln, ah, 0, n, H);
    }
  }
  const auto t_enc = std::chrono::steady_clock::now();
  stream.commit().wait();
  if (kLayerDump != nullptr) {
    const std::string pre = kLayerDump;
    auto wr = [&](const std::string& nm, const SharedBuffer& b, int cnt) {
      std::ofstream f(pre + "_" + nm + ".bin", std::ios::binary);
      std::vector<float> tmp((std::size_t)cnt);
      read_elt_(b.contents(), tmp.data(), (std::size_t)cnt, _cfg.use_bf16);
      f.write(reinterpret_cast<const char*>(tmp.data()),
              (std::size_t)cnt * sizeof(float));
    };
    wr("embed", dbgEmbed, n * H);
    for (int L = 0; L < c.n_layers; ++L) {
      wr("layer_" + std::to_string(L), dbgL[(std::size_t)L], n * H);
    }
    wr("logits", dbgLogits, c.vocab);
  }
  if (allhidden_out) { *allhidden_out = std::move(ah); }

  // On-device AWQ calibration: fold this chunk's per-channel |x| into the
  // running abs-max accumulators (dense full-attn taps captured above).
  if (_calib_on) {
    auto acc = [&](const std::vector<SharedBuffer>& taps,
                   std::vector<std::vector<float>>& out, int dim) {
      for (int L = 0; L < c.n_layers; ++L) {
        const auto* s =
            static_cast<const _Float16*>(taps[(std::size_t)L].contents());
        auto& a = out[(std::size_t)L];
        for (int t = 0; t < n; ++t) {
          const _Float16* r = s + (std::size_t)t * dim;
          for (int d = 0; d < dim; ++d) {
            const float v = std::fabs((float)r[d]);
            if (v > a[(std::size_t)d]) { a[(std::size_t)d] = v; }
          }
        }
      }
    };
    acc(ctq, _calib_qkv, H);
    acc(ctg, _calib_gu, H);
    acc(ctd, _calib_dn, ffn);
  }

  // MTP batched verify: return the per-position greedy predictions; the
  // per-position pre-final-norm hidden stays in `x` for the caller. No
  // [vocab] logit pull (the empty return signals verify mode).
  if (verify_all && preds_out) {
    preds_out->resize((std::size_t)n);
    const auto* a = static_cast<const std::int32_t*>(vamax.contents());
    for (int k = 0; k < n; ++k) { (*preds_out)[(std::size_t)k] = a[k]; }
    return {};
  }

  // MOSS-TTS: hand the last position's final-normed hidden (hn[0:H]) to the
  // caller's own output heads. No [vocab] logit pull (the empty return).
  if (return_hidden && hidden_out) {
    *hidden_out = std::move(hn);
    return {};
  }

  // MTP drafter input: snapshot the last position's pre-final-norm hidden
  // (x holds the post-last-layer residual stream; the final norm read it but
  // did not modify it). Cheap host copy; only when an MTP head is present.
  if (_mtp.ok && !_mtp_h.empty()) {
    std::memcpy(_mtp_h.contents(),
                static_cast<const char*>(x.contents()) +
                    (std::size_t)(n - 1) * H * 2,
                (std::size_t)H * 2);
  }

  if (kPProf) {
    const auto t_gpu = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("[qwen-prefill-prof n=%4d] alloc %.3f ms | encode %.3f ms | "
                "commit+gpu %.3f ms\n", n, ms(t_p0, t_alloc),
                ms(t_alloc, t_enc), ms(t_enc, t_gpu));
  }

  std::vector<float> out((std::size_t)c.vocab);
  read_elt_(logits.contents(), out.data(), (std::size_t)c.vocab, _cfg.use_bf16);
  return out;
}

void
MetalQwenModel::vqmm_(ComputeEncoder& enc, int m, const SharedBuffer& w,
                      const SharedBuffer& s, const SharedBuffer& b, int bits,
                      const SharedBuffer& xin, const SharedBuffer& y, int K,
                      int N)
{
  if (s.empty()) {   // dense f16: plain GEMM (M=m rows), no scales/biases
    dense_gemm_(enc, w, xin, y, K, N, m);
    return;
  }
  if (bits != 8) {
    qmm_auto_(enc, m, w, s, b, xin, y, K, N);   // 4-bit: qmv_batch/qmv/steel
    return;
  }
  // 8-bit, 2..kQmvBatchMaxRows: batched-GEMV (weight read once per tile
  // row set) -- qmv bandwidth, so a small-M verify costs ~1 decode step
  // instead of steel's ~3x. Follows the same probed per-m plan as the
  // 4-bit ladder (relative tile costs are ~bit-width independent: the
  // xp wins are register/occupancy effects, not dequant effects), on
  // the w8 twins.
  const int cap = qmv_batch_cap_(kQmvBatchMaxRows);
  if (m > 1 && m <= cap && _fn_qmv8_batch.valid()) {
    auto tile = [&](const metal_compute::ComputeFunction& fn,
                    std::size_t row0, int rows, unsigned tiles) {
      enc.set_function(fn);
      enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
      enc.set_buffer(3, xin, row0 * (std::size_t)K * 2);
      enc.set_buffer(4, y, row0 * (std::size_t)N * 2);
      enc.set_constant(5, K); enc.set_constant(6, N);
      enc.set_constant(7, rows);
      enc.dispatch({32, (unsigned)(N / 4), tiles}, {32, 2, 1});
    };
    switch (_qmv_plan[m]) {
      case QmvPlan::kXp8:
        if (_qmv8_enabled && _fn_qmv8_batch8_xp.valid()) {
          tile(_fn_qmv8_batch8_xp, 0, m, (unsigned)((m + 7) / 8));
          return;
        }
        break;
      case QmvPlan::kMix4R:
        if (_qmv4_enabled && _fn_qmv8_batch4.valid() && m > 4) {
          tile(_fn_qmv8_batch4, 0, 4, 1);
          tile(_fn_qmv8_batch, 4, m - 4, (unsigned)((m - 3) / 2));
          return;
        }
        break;
      case QmvPlan::kXp4:
        if (_qmv4_enabled && _fn_qmv8_batch4.valid()) {
          tile(_fn_qmv8_batch4, 0, m, (unsigned)((m + 3) / 4));
          return;
        }
        break;
      case QmvPlan::kTile2:
        break;
    }
    tile(_fn_qmv8_batch, 0, m, (unsigned)((m + 1) / 2));
    return;
  }
  if (m == 1) {
    enc.set_function(_fn_qmv8);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, K); enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
    return;
  }
  // 8-bit, large m: no batched-GEMV -> steel (reads the weight once across m).
  enc.set_function(_fn_qmm8);
  enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
  enc.set_buffer(3, xin); enc.set_buffer(4, y);
  enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, m);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
               (unsigned)(((m + 31) / 32) * 2), 2}, {32, 2, 2});
}

// Verify sub-profile accumulators (VPIPE_MTP_VPROFILE): the verify command
// buffer is split into main-forward / verifier-head (lm_head+decision) / MTP-
// head segments, each commit()+wait() timed. Reset per mtp_decode, printed in
// its profile block. Off by default -> one command buffer, no perturbation.
static double g_vp_main = 0.0, g_vp_vhead = 0.0, g_vp_mtp = 0.0;
static long g_vp_n = 0;

// Decode-attn kernel selection now lives in the DecodeGqaAttnSet; these forward
// to it (for the few non-dispatch callers + the legacy fallback paths).
bool
MetalQwenModel::gqa_vec2_on(int D, int pos) const {
  return D == 256 && _decode_set.ready() && _decode_set.uses_vec2(pos);
}

int
MetalQwenModel::gqa_decode_split(int D, int pos) const {
  if (D == 256 && _decode_set.ready()) { return _decode_set.split_for(pos); }
  return gqa_split_for(pos);
}

// Decode GQA attention via the kernel set's unified entrance. The MTP per-draft
// verify loop and the batched per-branch fallback tap here; the set picks the
// member kernel (+ split) for `pos`'s regime and uses its own scratch.
void
MetalQwenModel::encode_gqa_vec2_(
    ComputeEncoder& enc, const SharedBuffer& q, std::size_t qoff,
    const SharedBuffer& kp, const SharedBuffer& vp,
    const SharedBuffer& out, std::size_t outoff, int pos, float scale, int D,
    int Hq, int Hkv, int page_tokens, int n_pages,
    const SharedBuffer& pgtab, std::size_t pgoff) {
  (void)D; (void)Hq; (void)Hkv;        // the set carries the attention dims
  DecodeGqaAttnSet::Attn a;
  a.q = &q; a.q_off = qoff;
  a.kpool = &kp; a.vpool = &vp;
  a.out = &out; a.out_off = outoff;
  a.page_table = &pgtab; a.pgt_off = pgoff;
  a.pos = pos; a.page_tokens = page_tokens; a.n_pages = n_pages;
  a.scale = scale;
  _decode_set.dispatch(enc, a);
}

void
MetalQwenModel::log_decode_attn_choice_() const {
  const SessionContextIntf* s = _mc ? _mc->session() : nullptr;
  if (s == nullptr || !_decode_set.ready()) { return; }
  const char* v2 = std::getenv("VPIPE_QWEN_GQA_VEC2");
  const char* at = std::getenv("VPIPE_QWEN_GQA_AUTOTUNE");
  const char* how = v2 ? "env-forced"
      : ((at && std::atoi(at) == 0) ? "default" : "autotuned");
  s->log_debug(fmt(
      "[qwen] decode-attn kernel ({}): short(<8k)={}@{} mid(8-24k)={}@{} "
      "long(>=24k)={}@{}", how,
      _decode_set.kernel_name(4096), _decode_set.split_for(4096),
      _decode_set.kernel_name(16384), _decode_set.split_for(16384),
      _decode_set.kernel_name(32768), _decode_set.split_for(32768)));
}

bool
MetalQwenModel::mtp_verify_chunk_(ContextId cid, const SharedBuffer& x, int n,
                                  std::vector<std::int32_t>* preds,
                                  std::vector<std::int32_t>* mtp_preds,
                                  std::vector<std::int32_t>* mtp_preds2,
                                  int rope_delta, const GpuSamplerParams& sp,
                                  int seed_slot0, GdnVerifyCache* gcache,
                                  bool lc_mode, bool gdn_ring,
                                  const SharedBuffer* mtp_cond,
                                  const std::function<void(ComputeEncoder&)>&
                                      pre_commit)
{
  // Leviathan-Chen mode: expose the full verifier + MTP-head logits so
  // mtp_decode can do the ratio test + residual/bonus sampling on the host.
  // The verifier's per-position decision uses ARGMAX here (it only feeds the
  // MTP head's drafting context; the committed token is chosen by mtp_decode).
  if (lc_mode) {
    const std::size_t need = (std::size_t)n * _cfg.vocab * 2;
    if (_lc_vlogits.byte_size() < need) {
      _lc_vlogits = _mc->make_shared_buffer(need);
    }
    if (_lc_mlogits.byte_size() < need) {
      _lc_mlogits = _mc->make_shared_buffer(need);
    }
    if (_lc_vlogits.empty() || _lc_mlogits.empty()) { lc_mode = false; }
    if (lc_mode && mtp_preds2 != nullptr &&     // depth-2: the chained q2
        _lc_mlogits2.byte_size() < need) {
      _lc_mlogits2 = _mc->make_shared_buffer(need);
      if (_lc_mlogits2.empty()) { lc_mode = false; }
    }
  }
  // Direct-quantized batched verify. One forward over the n drafts, a per-
  // position decision (argmax or speculative sample), and the fused MTP head.
  // Two weight layouts:
  //  * mixed-affine (OptiQ): the matmuls are direct quantized batched-GEMV
  //    (vqmm_: weight read ONCE per group across the n rows, NO f16 dequant),
  //    DE-FUSED (a batched-GEMV can't write a column slice of a fused buffer),
  //    so q|k|v and the in_proj parts each land in their own buffer.
  //  * native k-quant (GGUF NextN): the matmuls dequant the raw k-quant block
  //    into the shared f16 scratch then a dense GEMM (kqmm_), exactly as the
  //    main k-quant prefill -- the FUSED q|k|v layout (qfull) is reused here,
  //    sliced into q|gate|k|v just like encode_decode_step_.
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, Kc = c.gdn_conv_kernel, ffn = c.ffn_inner;
  const float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)D);
  const int gate = c.attn_output_gate ? 1 : 0;
  const int qdo = gate ? 2 * qd : qd;

  struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
  std::vector<Chunk> chunks;
  int q_offset = -1;
  for (int written = 0; written < n;) {
    const int cap = _ctx->next_append_capacity(cid);
    const int cnt = std::min(n - written, cap);
    ContextManager::AppendSlot s = _ctx->append(cid, cnt);
    if (!s.valid()) { return false; }
    if (q_offset < 0) { q_offset = s.position; }
    chunks.push_back({(std::size_t)s.page_id.v * _ctx->page_stride_bytes(),
                      s.slot_offset, written, cnt});
    written += cnt;
  }
  if (q_offset < 0) { q_offset = 0; }
  const int page_tokens = _ctx->page_tokens();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer hn = buf((std::size_t)n * H);
  SharedBuffer qraw = buf((std::size_t)n * qdo);
  // k-quant fused scratch: q|k|v dequant into one [n, qdo+2kd] GEMM output
  // (qfull), and the a|b in_proj split [n, 2*Hv] (the affine path GEMVs each
  // part straight into qkv/zbuf/abuf/bbuf instead).
  SharedBuffer qfull =
      (_kquant || _dense) ? buf((std::size_t)n * (qdo + 2 * kd))
                          : SharedBuffer{};
  SharedBuffer abtmp =
      _kquant ? buf((std::size_t)n * 2 * Hv) : SharedBuffer{};
  // Dense fused GDN in_proj scratch [n, Nf] (qkv|z|a|b), hsliced into the
  // de-fused qkv/zbuf/abuf/bbuf the recurrent path consumes.
  const int Nf_gdn = Cd + vald + 2 * Hv;
  SharedBuffer mixf =
      _dense ? buf((std::size_t)n * Nf_gdn) : SharedBuffer{};
  SharedBuffer q3 = buf((std::size_t)n * qd), gate3 = buf((std::size_t)n * qd);
  SharedBuffer kbuf = buf((std::size_t)n * kd), vbuf = buf((std::size_t)n * kd);
  SharedBuffer qt = buf((std::size_t)n * qd), kt = buf((std::size_t)n * kd),
               vt = buf((std::size_t)n * kd);
  SharedBuffer at = buf((std::size_t)n * qd), att = buf((std::size_t)n * qd),
               ao = buf((std::size_t)n * H);
  SharedBuffer qkv = buf((std::size_t)n * Cd);
  SharedBuffer zbuf = buf((std::size_t)n * vald);
  SharedBuffer abuf = buf((std::size_t)n * Hv), bbuf = buf((std::size_t)n * Hv);
  SharedBuffer convout = buf((std::size_t)n * Cd);
  SharedBuffer gbuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer ygdn = buf((std::size_t)n * vald);
  SharedBuffer normout = buf((std::size_t)n * vald);
  SharedBuffer sg = buf((std::size_t)n * ffn), upb = buf((std::size_t)n * ffn);
  SharedBuffer vlogits = buf((std::size_t)n * c.vocab);
  SharedBuffer vamax =
      _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
  // Speculative-sampling scratch (only when the verify samples): per-position
  // softmax-weight workspace [n,vocab] (the sample kernel can't alias one ws
  // across the n in-flight dispatches) + a zeroed seen-set the kernel still
  // binds (penalties are off on this path).
  SharedBuffer vsample_ws, vseen;
  if (!sp.greedy && _fn_sample.valid()) {
    vsample_ws = buf((std::size_t)n * c.vocab);
    vseen = _mc->make_shared_buffer((std::size_t)c.vocab);
    if (!vsample_ws.empty() && !vseen.empty()) {
      std::memset(vseen.contents(), 0, (std::size_t)c.vocab);
    }
  }

  // Fused MTP head: append the verify positions to the MTP layer's own
  // (fresh-per-verify) KV so the MTP drafts in the SAME command buffer. For
  // depth-2 a SECOND window of n positions is appended (the chained second MTP
  // application), so the ctx holds 2n positions and depth-2 attends causally
  // over both windows (a small leak to rejected drafts -- verification keeps
  // the output exact regardless).
  const bool run_mtp = (mtp_preds != nullptr) && _mtp.ok;
  const bool run_mtp2 = run_mtp && (mtp_preds2 != nullptr);
  SharedBuffer emb_P, mnorm_e, mnorm_h, mcomb, mcomb2, mctmp, mqfull, mvamax,
      mvamax2, mhs;
  ContextManager::AppendSlot mslot;
  std::size_t m_page_off = 0;
  int m_npages = 0;
  if (run_mtp) {
    // Persistent MTP KV: do NOT wipe each round -- append this round's draft
    // window onto the retained committed history so the head attends over the
    // decode so far. mtp_decode rolls back the rejected window after each round
    // and resets once at decode start. Non-persistent (fallback): wipe per
    // round (the old stateless behavior). Overflow of the single MTP page
    // (long decode) -> reset + retry, gracefully dropping older history.
    const int mtp_app = run_mtp2 ? 2 * n : n;
    if (!_mtp_persist) { mtp_ctx_reset_(); }
    mslot = _mtp.ctx->append(_mtp.cid, mtp_app);
    if (!mslot.valid() && _mtp_persist) {
      mtp_ctx_reset_();
      mslot = _mtp.ctx->append(_mtp.cid, mtp_app);
    }
    if (!mslot.valid()) { return false; }
    m_page_off = (std::size_t)mslot.page_id.v * _mtp.ctx->page_stride_bytes();
    m_npages = _mtp.ctx->fill_page_table(
        _mtp.cid, static_cast<std::int32_t*>(_mtp.pgt.contents()));
    emb_P = buf((std::size_t)n * H);
    mnorm_e = buf((std::size_t)n * H);
    mnorm_h = buf((std::size_t)n * H);
    mcomb = buf((std::size_t)n * H);
    mctmp = buf((std::size_t)n * H);
    mhs = buf((std::size_t)n * H);   // base post-final-norm hidden for the MTP
    mqfull = buf((std::size_t)n * (qdo + 2 * kd));
    mvamax = _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
    if (run_mtp2) {
      mcomb2 = buf((std::size_t)n * H);
      mvamax2 = _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
    }
  }

  // MAXM=4 batched GEMV (read each weight once per 4 rows) is now selected
  // adaptively by row count in qmm_auto_/kqmv_batch_ (m>2 -> MAXM=4), so the
  // MTP verify (n=3-4) and realtime-vqa batched decode (>=3 branches) both get
  // it with no per-call scoping needed.
  static const bool vprof =
      (std::getenv("VPIPE_MTP_VPROFILE") != nullptr);
  std::chrono::steady_clock::time_point t_phase;
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // Verify sub-profile: close the current segment (end+commit+wait), credit
    // its wall time to `acc`, open a fresh command buffer. The lambdas below +
    // the vqmm_/kqmv_batch_ helpers re-read `enc` each call, so the re-seat is
    // transparent. No-op when vprof is off (one command buffer, no perturb).
    auto vp_split = [&](double& acc) {
      if (!vprof) { return; }
      enc.end();
      stream.commit().wait();
      const auto t = std::chrono::steady_clock::now();
      acc += std::chrono::duration<double, std::milli>(t - t_phase).count();
      t_phase = t;
      enc = stream.begin_compute();
    };
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out, int Hh,
                      int S, int W, int off, int block, int gstride) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in, 0); enc.set_buffer(1, out, 0);
      enc.set_constant(2, Hh); enc.set_constant(3, S); enc.set_constant(4, W);
      enc.set_constant(5, off); enc.set_constant(6, block);
      enc.set_constant(7, gstride);
      enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                   {(unsigned)D, 1, 1});
    };
    // Main-model RoPE base = KV slot + rope_delta: sequential text passes
    // rope_delta 0 (rope == slot); a post-multimodal decode passes the mROPE
    // offset so the verify uses the same rotary position the serial decode
    // would. Attention masking + KV writes still use the true slot (q_offset).
    const int rope_base = q_offset + rope_delta;
    auto rope = [&](const SharedBuffer& xb, int heads) {
      enc.set_function(_fn_rope_partial);
      enc.set_buffer(0, xb); enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads); enc.set_constant(3, n);
      enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
      enc.set_constant(6, rope_base);
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      for (const Chunk& ch : chunks) {
        enc.set_function(_fn_kv_write_paged);
        enc.set_buffer(0, src); enc.set_buffer(1, pool, ch.page_off);
        enc.set_constant(2, page_tokens); enc.set_constant(3, D);
        enc.set_constant(4, n); enc.set_constant(5, ch.src_off);
        enc.set_constant(6, ch.slot);
        enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };

    if (vprof) { t_phase = std::chrono::steady_clock::now(); }
    int gci = 0;   // linear-layer index into gcache (when capturing GDN inputs)
    for (int L = 0; L < c.n_layers; ++L) {
      Layer& ly = _layers[L];
      rms(x, 0, ly.in_ln, hn, 0, n, H);
      if (ly.is_full) {
        const SharedBuffer& kp = *_ctx->kpool(L);
        const SharedBuffer& vp = *_ctx->vpool(L);
        if (_kquant) {
          // Batched k-quant GEMV (weight read once across the n rows): q|k
          // (kqk) into qfull cols [0, kqk_n), v into [kqk_n, Nfqkv), giving the
          // exact encode_decode_step_ fused layout, then slice q|gate|k|v.
          const int Nfqkv = qdo + 2 * kd;
          if (ly.qk_q4k_aff) {
            amv_g32_batch_(enc, ly.qk_aw, ly.qk_as, ly.qk_ab, hn, qfull, H,
                           ly.kqk_n, n, Nfqkv, 0);
          } else {
            kqmv_batch_(enc, ly.kqk_t, ly.kqk, hn, qfull, H, ly.kqk_n, n,
                        Nfqkv, 0);
          }
          kqmv_batch_(enc, ly.kqv_t, ly.kqv, hn, qfull, H, kd, n, Nfqkv,
                      ly.kqk_n);
          if (gate) {
            hslice(qfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
            hslice(qfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
          } else {
            hslice(qfull, q3, n, Nfqkv, qd, 0, 0, 0);
          }
          hslice(qfull, kbuf, n, Nfqkv, kd, qdo, 0, 0);
          hslice(qfull, vbuf, n, Nfqkv, kd, qdo + kd, 0, 0);
        } else if (_dense) {
          // Dense f16: fused q|k|v GEMM (M=n) -> qfull[n, Nfqkv], then slice
          // q|gate|k|v exactly like the k-quant fused layout above.
          const int Nfqkv = qdo + 2 * kd;
          dense_gemm_(enc, ly.qw, hn, qfull, H, Nfqkv, n);
          if (gate) {
            hslice(qfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
            hslice(qfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
          } else {
            hslice(qfull, q3, n, Nfqkv, qd, 0, 0, 0);
          }
          hslice(qfull, kbuf, n, Nfqkv, kd, qdo, 0, 0);
          hslice(qfull, vbuf, n, Nfqkv, kd, qdo + kd, 0, 0);
        } else if (ly.qkv_fused) {
          // Uniform-base-width fused q|k|v -> ONE affine GEMV into qfull, then
          // slice k/v from the tail (like the dense/kquant fused branches).
          const int Nfqkv = qdo + 2 * kd;
          vqmm_(enc, n, ly.qw, ly.qs, ly.qb, ly.q_bits, hn, qfull, H, Nfqkv);
          if (gate) {
            hslice(qfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
            hslice(qfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
          } else {
            hslice(qfull, q3, n, Nfqkv, qd, 0, 0, 0);
          }
          hslice(qfull, kbuf, n, Nfqkv, kd, qdo, 0, 0);
          hslice(qfull, vbuf, n, Nfqkv, kd, qdo + kd, 0, 0);
        } else {
          // De-fused q|k|v, each a direct quantized batched-GEMV.
          vqmm_(enc, n, ly.qw, ly.qs, ly.qb, ly.q_bits, hn, qraw, H, qdo);
          vqmm_(enc, n, ly.kw, ly.ks, ly.kb, ly.k_bits, hn, kbuf, H, kd);
          vqmm_(enc, n, ly.vw, ly.vs, ly.vb, ly.v_bits, hn, vbuf, H, kd);
          if (gate) {
            hslice(qraw, q3, n * Hq, 2 * D, D, 0, Hq, qdo);
            hslice(qraw, gate3, n * Hq, 2 * D, D, D, Hq, qdo);
          } else {
            hslice(qraw, q3, n, qdo, qd, 0, 0, 0);
          }
        }
        rms(q3, 0, ly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, ly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        rope(qt, Hq); rope(kt, Hkv);
        kv_write(kt, kp); kv_write(vt, vp);
        // Attention. The scalar paged kernel runs one simdgroup per (head,
        // query) scanning the WHOLE KV with 32 lanes -- fine at short KV, but
        // it dominates the verify (and erodes MTP) at long context. Past the
        // mb threshold switch to the same flash-decode-GQA the single-token
        // decode uses (each KV head read ONCE for all G query heads, KV split
        // across simdgroups + merge), run once PER query token (n is tiny).
        // Query i attends causally to KV 0..q_offset+i (pos = q_offset+i), so
        // the result equals the scalar multi-query kernel -- and the serial
        // decode it's verified against ALSO uses this kernel, so token-exact.
        const bool long_ctx = (q_offset + n) >= _sdpa_mb_min;
        const bool use_gqa = _gqa_attn && long_ctx && (D <= 256)
            && (D % 32 == 0) && !_d_gqa_oacc.empty();
        if (use_gqa) {
          const int sp = gqa_split_for(q_offset + n), G = Hq / Hkv;
          // Prefer the MLX 2-pass decode (vec2): each draft query is a single-
          // token decode at pos q_offset+i, so it drops straight in.
          const bool use_vec2 = gqa_vec2_on(D, q_offset + n);
          // vec is latency-optimal even at G=6 -- see decode_step_fast.
          const bool use_vec =
              _gqa_vec && (D % 128 == 0) && _fn_sdpa_gqa_vec.valid();
          const bool use_kbl = use_vec && _gqa_blk && (D == 256 || D == 128);
          const metal_compute::ComputeFunction& vec_fn =
              !use_kbl ? _fn_sdpa_gqa_vec
                       : (D == 256 ? _fn_sdpa_gqa_kbl : _fn_sdpa_gqa_kbl128);
          // Per-query roped q must be contiguous [Hq, D]: transpose qt
          // [Hq,n,D] -> at [n,Hq,D] so query i = at + i*qd. The merges write
          // att [n,Hq,D] directly (no post-attention transpose needed).
          transpose(qt, at, Hq, n);
          for (int i = 0; i < n; ++i) {
            if (use_vec2) {
              encode_gqa_vec2_(enc, at, (std::size_t)i * qd * 2, kp, vp,
                  att, (std::size_t)i * qd * 2, q_offset + i, scale, D, Hq,
                  Hkv, page_tokens, n_pages, _pgtab, 0);
              continue;
            }
            enc.set_function(use_vec ? vec_fn : _fn_sdpa_gqa);
            enc.set_buffer(0, at, (std::size_t)i * qd * 2);
            enc.set_buffer(1, kp); enc.set_buffer(2, vp);
            enc.set_buffer(3, _d_gqa_oacc); enc.set_buffer(4, _d_gqa_m);
            enc.set_buffer(5, _d_gqa_l);
            enc.set_constant(6, scale); enc.set_constant(7, D);
            enc.set_constant(8, Hq); enc.set_constant(9, Hkv);
            enc.set_constant(10, q_offset + i);
            enc.set_constant(11, page_tokens); enc.set_constant(12, n_pages);
            enc.set_buffer(13, _pgtab); enc.set_constant(14, sp);
            if (use_vec) {
              enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                           {32, (unsigned)G, 1});
            } else {
              enc.dispatch({32, (unsigned)Hkv, (unsigned)sp}, {32, 1, 1});
            }
            enc.set_function(_fn_sdpa_gqa_merge);
            enc.set_buffer(0, _d_gqa_oacc); enc.set_buffer(1, _d_gqa_m);
            enc.set_buffer(2, _d_gqa_l);
            enc.set_buffer(3, att, (std::size_t)i * qd * 2);
            enc.set_constant(4, D); enc.set_constant(5, sp);
            enc.set_constant(6, Hq);
            enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
          }
        } else {
          // n is tiny (<= draft window): the scalar paged kernel is cheapest
          // at short KV.
          enc.set_function(_fn_sdpa_paged);
          enc.set_buffer(0, qt); enc.set_buffer(1, kp); enc.set_buffer(2, vp);
          enc.set_buffer(3, at);
          enc.set_constant(4, scale); enc.set_constant(5, D);
          enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
          enc.set_constant(8, n); enc.set_constant(9, q_offset);
          enc.set_constant(10, page_tokens); enc.set_constant(11, n_pages);
          enc.set_buffer(12, _pgtab);
          enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
          transpose(at, att, Hq, n);
        }
        if (gate) {
          enc.set_function(_fn_mul_sigmoid);
          enc.set_buffer(0, att); enc.set_buffer(1, gate3);
          enc.set_buffer(2, att); enc.set_constant(3, n * qd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        }
        if (_kquant) {
          if (ly.o_q4k_aff) {
            amv_g32_batch_(enc, ly.o_aw, ly.o_as, ly.o_ab, att, ao, qd, H, n,
                           H, 0);
          } else {
            kqmv_batch_(enc, ly.kqo_t, ly.kqo, att, ao, qd, H, n, H, 0);
          }
        } else {
          vqmm_(enc, n, ly.ow, ly.os, ly.ob, ly.o_bits, att, ao, qd, H);
        }
        residual(x, ao, x, n * H);
      } else {
        // GDN recurrent-step inputs (conv input qkv, gdn_step g/beta) land in
        // this layer's verify cache when capturing (non-ring path, for
        // gdn_replay_); else shared scratch. The non-recurrent parts
        // (zbuf/abuf/bbuf/convout/ygdn) stay shared.
        const SharedBuffer& qkv_c =
            (gcache && gci < (int)gcache->qkv.size()) ? gcache->qkv[gci] : qkv;
        const SharedBuffer& gbuf_c =
            (gcache && gci < (int)gcache->gbuf.size()) ? gcache->gbuf[gci]
                                                       : gbuf;
        const SharedBuffer& betabuf_c =
            (gcache && gci < (int)gcache->betabuf.size())
                ? gcache->betabuf[gci] : betabuf;
        ++gci;
        // De-fused in_proj parts. qkv lands in its own [n, Cd] buffer, so
        // conv1d reads it with x_stride = Cd (vs Nf for the fused layout).
        if (_kquant) {
          // qkv (q5_K) + z (q4_K) batched-GEMV into their own buffers; a|b are
          // f16 (kqab [2Hv,H]) -> one dense GEMM -> abtmp[n,2Hv] split a|b.
          kqmv_batch_(enc, ly.kqkv_t, ly.kqkv, hn, qkv_c, H, Cd, n, Cd, 0);
          kqmv_batch_(enc, ly.kqz_t, ly.kqz, hn, zbuf, H, vald, n, vald, 0);
          dense_gemm_(enc, ly.kqab, hn, abtmp, H, 2 * Hv, n);
          hslice(abtmp, abuf, n, 2 * Hv, Hv, 0, 0, 0);
          hslice(abtmp, bbuf, n, 2 * Hv, Hv, Hv, 0, 0);
        } else if (_dense) {
          // Dense f16: fused in_proj GEMM (M=n) -> mixf[n, Nf] (qkv|z|a|b),
          // then slice qkv/z/a/b into the de-fused buffers the recurrent path
          // consumes (qkv_c with x_stride Cd, zbuf, abuf, bbuf).
          dense_gemm_(enc, ly.iqw, hn, mixf, H, Nf_gdn, n);
          hslice(mixf, qkv_c, n, Nf_gdn, Cd, 0, 0, 0);
          hslice(mixf, zbuf, n, Nf_gdn, vald, Cd, 0, 0);
          hslice(mixf, abuf, n, Nf_gdn, Hv, Cd + vald, 0, 0);
          hslice(mixf, bbuf, n, Nf_gdn, Hv, Cd + vald + Hv, 0, 0);
        } else {
          vqmm_(enc, n, ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, hn, qkv_c, H, Cd);
          vqmm_(enc, n, ly.izw, ly.izs, ly.izb, ly.z_bits, hn, zbuf, H, vald);
          vqmm_(enc, n, ly.iaw, ly.ias, ly.iab, ly.a_bits, hn, abuf, H, Hv);
          vqmm_(enc, n, ly.ibw, ly.ibs, ly.ibb, ly.b_bits, hn, bbuf, H, Hv);
        }
        const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                          !_gdn_force_v1;
        const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
        // g_beta is position-independent (reads abuf/bbuf, not convout) -> batch
        // it for both the ring and the in-place path.
        enc.set_function(_fn_gdn_g_beta);
        enc.set_buffer(0, abuf); enc.set_buffer(1, bbuf);
        enc.set_buffer(2, ly.A_log); enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, gbuf_c); enc.set_buffer(5, betabuf_c);
        enc.set_constant(6, Hv); enc.set_constant(7, n);
        enc.dispatch({(unsigned)(n * Hv), 1, 1}, {256, 1, 1});
        if (gdn_ring) {
          // ZERO-COPY rollback path: advance the recurrent state ONE position
          // at a time through the GDN ring, binding each token's state-IN
          // (slot cur+p) and state-OUT (slot cur+p+1) to a DISTINCT ring slot.
          // Every intermediate S_{p+1..p+K} survives on-device, so a partial
          // accept rolls back by a pure cursor move in mtp_decode -- no host
          // snapshot, no gdn_replay_. Projections + g_beta stay batched above;
          // only the cheap conv1d + gdn_step recurrence goes per-position. The
          // cursor is fixed across this whole verify (advanced once, by `keep`,
          // afterward), so all layers stay batched per-position -- the batching
          // is NOT defeated.
          const std::size_t kb1 = (std::size_t)keyd * 2;   // n=1 section stride
          const std::size_t vb1 = 2 * kb1;
          for (int p = 0; p < n; ++p) {
            const SharedBuffer* cr = _ctx->conv_slot(cid, L, p);
            const SharedBuffer* cw = _ctx->conv_slot(cid, L, p + 1);
            const SharedBuffer* sr = _ctx->ssm_slot(cid, L, p);
            const SharedBuffer* sw = _ctx->ssm_slot(cid, L, p + 1);
            enc.set_function(_fn_gdn_conv1d);
            enc.set_buffer(0, *cr);
            enc.set_buffer(1, qkv_c, (std::size_t)p * Cd * 2);
            enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
            enc.set_constant(4, 1); enc.set_constant(5, Cd);
            enc.set_constant(6, Kc); enc.set_constant(7, Cd);
            enc.set_constant(8, keyd); enc.set_buffer(9, *cw);
            enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
            rms(convout, 0, _gdn_qscale, convout, 0, Hk, Dk);
            rms(convout, kb1, _gdn_kscale, convout, kb1, Hk, Dk);
            enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
            enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb1);
            enc.set_buffer(2, convout, vb1);
            enc.set_buffer(3, gbuf_c, (std::size_t)p * Hv * 4);
            enc.set_buffer(4, betabuf_c, (std::size_t)p * Hv * 4);
            enc.set_buffer(5, *sr);
            enc.set_buffer(6, ygdn, (std::size_t)p * vald * 2);
            enc.set_buffer(7, *sw);
            enc.set_constant(8, 1);
            enc.set_constant(9, _kquant ? -Hk : Hk);
            enc.set_constant(10, Hv);
            enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
          }
        } else {
          // In-place batched path (host snapshot + gdn_replay_ on reject).
          const SharedBuffer* csb = _ctx->conv_state(cid, L);
          const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
          enc.set_function(_fn_gdn_conv1d);
          enc.set_buffer(0, *csb); enc.set_buffer(1, qkv_c);
          enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
          enc.set_constant(4, n); enc.set_constant(5, Cd);
          enc.set_constant(6, Kc); enc.set_constant(7, Cd);   // x_stride = Cd
          enc.set_constant(8, keyd); enc.set_buffer(9, *csb);
          enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
          const std::size_t kb_off = (std::size_t)n * keyd * 2;
          const std::size_t vb_off = 2 * kb_off;
          rms(convout, 0, _gdn_qscale, convout, 0, n * Hk, Dk);
          rms(convout, kb_off, _gdn_kscale, convout, kb_off, n * Hk, Dk);
          enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
          enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
          enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gbuf_c);
          enc.set_buffer(4, betabuf_c); enc.set_buffer(5, *ssb);
          enc.set_buffer(6, ygdn); enc.set_buffer(7, *ssb);
          enc.set_constant(8, n);
          enc.set_constant(9, _kquant ? -Hk : Hk);   // strided GQA for GGUF
          enc.set_constant(10, Hv);
          enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
        }
        rms(ygdn, 0, ly.gdn_norm, normout, 0, n * Hv, Dv);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, zbuf); enc.set_buffer(1, normout);
        enc.set_buffer(2, normout); enc.set_constant(3, n * vald);
        enc.dispatch({(unsigned)(n * vald), 1, 1}, {256, 1, 1});
        if (_kquant) {
          kqmv_batch_(enc, ly.kqout_t, ly.kqout, normout, ao, vald, H, n, H, 0);
        } else {
          vqmm_(enc, n, ly.gow, ly.gos, ly.gob, ly.gout_bits, normout, ao,
                vald, H);
        }
        residual(x, ao, x, n * H);
      }
      // MLP: gate/up GEMM + SwiGLU + down (all rows). k-quant batched-GEMV
      // (weight read once) into sg/upb; affine direct batched-GEMV (vqmm_).
      rms(x, 0, ly.post_ln, hn, 0, n, H);
      if (!_kquant && ly.mlp_fused) {
        // Mixed-fused (gate|up both w4): fused swiglu qmm straight into sg
        // (weights read once, no standalone swiglu pass).
        qmm_auto_swiglu_(enc, n, ly.guw, ly.gus, ly.gub, hn, sg, H, 2 * ffn);
      } else {
        if (_kquant) {
          if (ly.ffn_q4k_aff) {
            amv_g32_batch_(enc, ly.gate_aw, ly.gate_as, ly.gate_ab, hn, sg, H,
                           ffn, n, ffn, 0);
            amv_g32_batch_(enc, ly.up_aw, ly.up_as, ly.up_ab, hn, upb, H, ffn,
                           n, ffn, 0);
          } else {
            kqmv_batch_(enc, ly.kqgate_t, ly.kqgate, hn, sg, H, ffn, n, ffn, 0);
            kqmv_batch_(enc, ly.kqup_t, ly.kqup, hn, upb, H, ffn, n, ffn, 0);
          }
        } else {
          vqmm_(enc, n, ly.guw, ly.gus, ly.gub, ly.gate_bits, hn, sg, H, ffn);
          vqmm_(enc, n, ly.uw, ly.us, ly.ub, ly.up_bits, hn, upb, H, ffn);
        }
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, sg); enc.set_buffer(1, upb); enc.set_buffer(2, sg);
        enc.set_constant(3, n * ffn);
        enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
      }
      if (_kquant) {
        kqmv_batch_(enc, ly.kqdown_t, ly.kqdown, sg, ao, ffn, H, n, H, 0);
      } else {
        vqmm_(enc, n, ly.dw, ly.ds, ly.db, ly.down_bits, sg, ao, ffn, H);
      }
      residual(x, ao, x, n * H);
    }
    vp_split(g_vp_main);   // main 36-layer forward over the n drafts
    // Final norm + per-position lm_head (direct quantized, M=n) + the verifier's
    // per-position DECISION -> vamax[k] (the token after drafts[0..k]). Greedy:
    // argmax. Speculative sampling: a token sampled from P at slot q_offset+k+1,
    // seeded by that ABSOLUTE slot (q_offset+k+1 - seed_slot0) so it is
    // byte-identical to decode_pipelined's per-step seed -- every committed
    // token is then the verifier's sample from the right conditional P, i.e.
    // exact autoregressive sampling; the drafts only gate how many land/round.
    rms(x, 0, _final_ln, hn, 0, n, H);
    if (_kquant) {
      // Tied q6_K lm_head (the embed table doubles as the head): ONE batched
      // q6_K GEMV reads the [vocab, H] table once for all n positions (vs n
      // re-reads of the huge table, or a ~1 GB fused dequant of it).
      kqmv_batch_(enc, lm_head_kqt_(), lm_head_kq_(), hn, vlogits, H,
                  c.vocab, n, c.vocab, 0);
    } else if (_dense_embed) {
      dense_gemm_(enc, _tied ? _embed_w : _lm_w, hn, vlogits, H, c.vocab, n);
    } else {
      const int lb = _tied ? _embed_bits : _lm_bits;
      vqmm_(enc, n, _tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
            _tied ? _embed_b : _lm_b, lb, hn, vlogits, H, c.vocab);
    }
    // lc_mode keeps vamax = ARGMAX (the MTP drafting context); the committed
    // token is chosen by mtp_decode's Leviathan-Chen step from _lc_vlogits.
    const bool vsample = !sp.greedy && !vsample_ws.empty() && !lc_mode;
    if (lc_mode) {   // stash the verifier logits before the MTP head reuses them
      enc.set_function(_fn_copy);
      enc.set_buffer(0, vlogits); enc.set_buffer(1, _lc_vlogits);
      const int zero = 0, cnt = n * c.vocab;
      enc.set_constant(2, zero); enc.set_constant(3, cnt);
      enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
    }
    for (int k = 0; k < n; ++k) {
      const std::size_t vamax_off = (std::size_t)k * sizeof(std::int32_t);
      const std::size_t vlog_off = (std::size_t)k * c.vocab * 2;
      if (!vsample) {
        encode_argmax_(enc, vlogits, vlog_off, vamax, vamax_off, c.vocab);
      } else {
        const std::uint32_t step_seed = (std::uint32_t)(
            sp.seed + 0x9e3779b9ull *
                          (std::uint64_t)(q_offset + k + 1 - seed_slot0));
        encode_sample_core_(enc, vlogits, vlog_off, vamax, vamax_off,
                            vsample_ws, (std::size_t)k * c.vocab * 2,
                            vseen, 0, sp, step_seed, c.vocab);
      }
    }

    vp_split(g_vp_vhead);   // final norm + verifier lm_head + per-pos decision
    // ---- Fused MTP head (the "verify is a decode step that drafts") ----
    // At each position i: combined = fc_e@norm_e(embed(P_i)) + fc_h@norm_h(H_i),
    // one full-attn MTP layer (its own KV), then argmax(lm_head(mtp.norm(.))) =
    // the next-next-token draft. Runs in THIS command buffer on the on-GPU main
    // preds (vamax) + hiddens (x) -- no separate draft forward.
    if (run_mtp) {
      // The MTP head conditions on the POST-final-norm hidden, matching the
      // reference contract (base_hidden_variant=post_norm / mtp_hidden_variant=
      // post_norm): depth-1 from model.norm(main hidden), depth-2 from
      // mtp.norm(prev MTP hidden) -- NOT the pre-norm residual stream. Feeding
      // the pre-norm hidden mis-conditions the trained head and tanks draft
      // acceptance (~0.58 vs ~0.92 measured) while staying invisible to the
      // token-exact tests (the verifier-corrected output is independent of the
      // draft). VPIPE_MTP_PRENORM_HIDDEN reverts to the old pre-norm feed for
      // an A/B.
      const bool mtp_prenorm =
          std::getenv("VPIPE_MTP_PRENORM_HIDDEN") != nullptr;
      Layer& mly = _mtp.lyr;
      const int Nfqkv = qdo + 2 * kd;
      const SharedBuffer& mkp = *_mtp.ctx->kpool(0);
      const SharedBuffer& mvp = *_mtp.ctx->vpool(0);
      const int mpt = _mtp.ctx->page_tokens();
      // One MTP application over the n positions: combined = fc_e@norm_e(emb(
      // ids)) + fc_h@norm_h(hsrc); the full-attn MTP layer (MTP-ctx window at
      // slot_off / RoPE qpos); mtp.norm + shared lm_head + argmax -> the draft
      // (argmax_out). hidden_out gets the pre-final-norm MTP hidden, which the
      // depth-2 application chains from.
      auto mtp_step = [&](const SharedBuffer& ids, const SharedBuffer& hsrc,
                          int slot_off, int qpos, const SharedBuffer& hidden_out,
                          const SharedBuffer& argmax_out) {
        if (_kquant) {
          embed_q6k_(enc, ids, 0, emb_P, n);    // tied q6_K table
        } else if (_dense || _dense_embed) {
          // Dense f16 embed gather: embed_gather_f16(ids, table, out, H).
          enc.set_function(_fn_embed_dense);
          enc.set_buffer(0, ids); enc.set_buffer(1, _embed_w);
          enc.set_buffer(2, emb_P); enc.set_constant(3, H);
          enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
        } else {
          enc.set_function(_fn_embed);
          enc.set_buffer(0, ids);
          enc.set_buffer(1, _embed_w); enc.set_buffer(2, _embed_s);
          enc.set_buffer(3, _embed_b); enc.set_buffer(4, emb_P);
          enc.set_constant(5, H);
          enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
        }
        rms(emb_P, 0, _mtp.prenorm_e, mnorm_e, 0, n, H);
        rms(hsrc, 0, _mtp.prenorm_h, mnorm_h, 0, n, H);
        dense_gemm_(enc, _mtp.fc_e, mnorm_e, hidden_out, H, H, n);
        dense_gemm_(enc, _mtp.fc_h, mnorm_h, mctmp, H, H, n);
        residual(hidden_out, mctmp, hidden_out, n * H);
        rms(hidden_out, 0, mly.in_ln, hn, 0, n, H);
        if (_kquant) {
          // q|k (kqk) + v (kqv) batched-GEMV into mqfull[n, Nfqkv].
          kqmv_batch_(enc, mly.kqk_t, mly.kqk, hn, mqfull, H, mly.kqk_n, n,
                      Nfqkv, 0);
          kqmv_batch_(enc, mly.kqv_t, mly.kqv, hn, mqfull, H, kd, n, Nfqkv,
                      mly.kqk_n);
        } else if (mly.qs.empty()) {
          // Dense f16: fused q|k|v GEMM (M=n) -> mqfull[n, Nfqkv].
          dense_gemm_(enc, mly.qw, hn, mqfull, H, Nfqkv, n);
        } else {
          vqmm_(enc, n, mly.qw, mly.qs, mly.qb, mly.q_bits, hn, mqfull, H,
                Nfqkv);
        }
        hslice(mqfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
        hslice(mqfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
        hslice(mqfull, kbuf, n, Nfqkv, kd, qdo, 0, 0);
        hslice(mqfull, vbuf, n, Nfqkv, kd, qdo + kd, 0, 0);
        rms(q3, 0, mly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, mly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        for (int two = 0; two < 2; ++two) {     // RoPE q, then k, at qpos
          enc.set_function(_fn_rope_partial);
          enc.set_buffer(0, two == 0 ? qt : kt);
          enc.set_buffer(1, _inv_freq);
          enc.set_constant(2, two == 0 ? Hq : Hkv); enc.set_constant(3, n);
          enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
          enc.set_constant(6, qpos);
          enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n,
                        (unsigned)(two == 0 ? Hq : Hkv)},
                       {(unsigned)(c.rotary_dim / 2), 1, 1});
        }
        for (int two = 0; two < 2; ++two) {     // KV write k, then v, at slot
          enc.set_function(_fn_kv_write_paged);
          enc.set_buffer(0, two == 0 ? kt : vt);
          enc.set_buffer(1, two == 0 ? mkp : mvp, m_page_off);
          enc.set_constant(2, mpt); enc.set_constant(3, D);
          enc.set_constant(4, n); enc.set_constant(5, 0);
          enc.set_constant(6, mslot.slot_offset + slot_off);
          enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)Hkv},
                       {(unsigned)D, 1, 1});
        }
        enc.set_function(_fn_sdpa_paged);
        enc.set_buffer(0, qt); enc.set_buffer(1, mkp); enc.set_buffer(2, mvp);
        enc.set_buffer(3, at);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        enc.set_constant(8, n); enc.set_constant(9, qpos);
        enc.set_constant(10, mpt); enc.set_constant(11, m_npages);
        enc.set_buffer(12, _mtp.pgt);
        enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
        transpose(at, att, Hq, n);
        enc.set_function(_fn_mul_sigmoid);
        enc.set_buffer(0, att); enc.set_buffer(1, gate3);
        enc.set_buffer(2, att); enc.set_constant(3, n * qd);
        enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        if (_kquant) {
          kqmv_batch_(enc, mly.kqo_t, mly.kqo, att, ao, qd, H, n, H, 0);
        } else if (mly.os.empty()) {
          dense_gemm_(enc, mly.ow, att, ao, qd, H, n);   // dense f16 o_proj
        } else {
          vqmm_(enc, n, mly.ow, mly.os, mly.ob, mly.o_bits, att, ao, qd, H);
        }
        residual(hidden_out, ao, hidden_out, n * H);
        rms(hidden_out, 0, mly.post_ln, hn, 0, n, H);
        if (_kquant) {
          // gate/up batched-GEMV into sg/upb, SwiGLU, then down.
          kqmv_batch_(enc, mly.kqgate_t, mly.kqgate, hn, sg, H, ffn, n, ffn, 0);
          kqmv_batch_(enc, mly.kqup_t, mly.kqup, hn, upb, H, ffn, n, ffn, 0);
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, sg); enc.set_buffer(1, upb); enc.set_buffer(2, sg);
          enc.set_constant(3, n * ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
          kqmv_batch_(enc, mly.kqdown_t, mly.kqdown, sg, ao, ffn, H, n, H, 0);
        } else if (mly.gus.empty()) {
          // Dense f16: gate/up are DE-FUSED (guw/uw) -> two GEMMs + plain
          // SwiGLU, then down GEMM. Mirrors the backbone dense MLP.
          dense_gemm_(enc, mly.guw, hn, sg, H, ffn, n);
          dense_gemm_(enc, mly.uw, hn, upb, H, ffn, n);
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, sg); enc.set_buffer(1, upb); enc.set_buffer(2, sg);
          enc.set_constant(3, n * ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
          dense_gemm_(enc, mly.dw, sg, ao, ffn, H, n);
        } else if (mly.mlp_fused) {
          // 4-bit interleaved gate/up SwiGLU via the size-adaptive path: at the
          // small M of a verify it picks the batched-GEMV swiglu (weight read
          // once across rows) instead of steel (~3x slower; run 1-2x/round).
          qmm_auto_swiglu_(enc, n, mly.guw, mly.gus, mly.gub, hn, sg, H,
                           2 * ffn);
          vqmm_(enc, n, mly.dw, mly.ds, mly.db, mly.down_bits, sg, ao, ffn, H);
        } else {
          // 8-bit head (no fused w8 swiglu): de-fused gate/up GEMVs + plain
          // SwiGLU, then down -- each at its own bits. Mirrors the kquant path.
          vqmm_(enc, n, mly.guw, mly.gus, mly.gub, mly.gate_bits, hn, sg, H,
                ffn);
          vqmm_(enc, n, mly.uw, mly.us, mly.ub, mly.up_bits, hn, upb, H, ffn);
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, sg); enc.set_buffer(1, upb); enc.set_buffer(2, sg);
          enc.set_constant(3, n * ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
          vqmm_(enc, n, mly.dw, mly.ds, mly.db, mly.down_bits, sg, ao, ffn, H);
        }
        residual(hidden_out, ao, hidden_out, n * H);
        rms(hidden_out, 0, _mtp.final_norm, hn, 0, n, H);
        if (_kquant) {
          kqmv_batch_(enc, lm_head_kqt_(), lm_head_kq_(), hn, vlogits, H,
                      c.vocab, n, c.vocab, 0);
        } else if (_mtp.draft_head) {
          // 4-bit DRAFT head: half the vocab-matrix bandwidth of the 8-bit
          // verifier head. Draft-only -> verify-corrected, output stays exact.
          vqmm_(enc, n, _mtp.dlm_w, _mtp.dlm_s, _mtp.dlm_b, 4, hn, vlogits, H,
                c.vocab);
        } else if (_dense || _dense_embed) {
          // Dense f16 shared lm_head (tied embed or separate lm_w), M=n GEMM.
          dense_gemm_(enc, _tied ? _embed_w : _lm_w, hn, vlogits, H,
                    c.vocab, n);
        } else {
          const int mlb = _tied ? _embed_bits : _lm_bits;
          vqmm_(enc, n, _tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
                _tied ? _embed_b : _lm_b, mlb, hn, vlogits, H, c.vocab);
        }
        for (int k = 0; k < n; ++k) {
          encode_argmax_(enc, vlogits, (std::size_t)k * c.vocab * 2,
                         argmax_out, (std::size_t)k * sizeof(std::int32_t),
                         c.vocab);
        }
        // Chain the POST-final-norm MTP hidden (mtp.norm output, now in hn) so
        // a depth-2 step conditions on post_norm, not the pre-norm residual.
        if (!mtp_prenorm) {
          enc.set_function(_fn_copy);
          enc.set_buffer(0, hn); enc.set_buffer(1, hidden_out);
          const int zc = 0, cc = n * H;
          enc.set_constant(2, zc); enc.set_constant(3, cc);
          enc.dispatch({(unsigned)cc, 1, 1}, {256, 1, 1});
        }
      };
      // depth-1: from the main hidden + emb(main pred). depth-2: chain from the
      // depth-1 MTP hidden + emb(depth-1 draft), into the SECOND ctx window.
      // Conditioning on the chained ARGMAX (not the candidate token) is what
      // gives good drafts here: the boundary's next token is the main model's
      // prediction, which lies past the candidate window -- conditioning on a
      // candidate instead collapsed acceptance to ~0 (measured). This argmax
      // dependency is also why the MTP output heads can't be batched.
      auto lc_stash = [&](const SharedBuffer& dst) {   // vlogits -> dst (q)
        enc.set_function(_fn_copy);
        enc.set_buffer(0, vlogits); enc.set_buffer(1, dst);
        const int zero = 0, cnt = n * c.vocab;
        enc.set_constant(2, zero); enc.set_constant(3, cnt);
        enc.dispatch({(unsigned)cnt, 1, 1}, {256, 1, 1});
      };
      // depth-1 hidden = model.norm(main hidden) (post_norm), not the pre-norm
      // residual x. mhs holds it; the verifier's hn (= _final_ln(x)) is reused
      // as scratch inside mtp_step, so compute a dedicated copy here.
      if (!mtp_prenorm) { rms(x, 0, _final_ln, mhs, 0, n, H); }
      // mtp_cond (teacher-forcing) overrides the conditioning token: condition
      // the head on a caller-supplied token instead of the verifier's argmax.
      mtp_step(mtp_cond ? *mtp_cond : vamax, mtp_prenorm ? x : mhs, 0,
               mslot.position, mcomb, mvamax);
      if (lc_mode) { lc_stash(_lc_mlogits); }   // q1 (1st MTP application)
      if (run_mtp2) {
        mtp_step(mvamax, mcomb, n, mslot.position + n, mcomb2, mvamax2);
        if (lc_mode) { lc_stash(_lc_mlogits2); }   // q2 (chained 2nd)
      }
      // Lever (b): fold the L-C step into THIS command buffer (one commit per
      // round, not a second stream + sync). The verifier p (_lc_vlogits) and
      // drafter q1/q2 (_lc_mlogits/_lc_mlogits2) are now stashed; the caller's
      // pre_commit encodes lc_accept/lc_sample reading them, so the commit below
      // covers the whole round's GPU work and the host reads only token ids.
      if (pre_commit) { pre_commit(enc); }
    }
  }
  stream.commit().wait();
  if (vprof) {   // last segment: the fused MTP head (1x depth-1, 2x depth-2)
    g_vp_mtp += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_phase).count();
    ++g_vp_n;
  }
  preds->resize((std::size_t)n);
  const auto* a = static_cast<const std::int32_t*>(vamax.contents());
  for (int k = 0; k < n; ++k) { (*preds)[(std::size_t)k] = a[k]; }
  if (run_mtp) {
    mtp_preds->resize((std::size_t)n);
    const auto* mp = static_cast<const std::int32_t*>(mvamax.contents());
    for (int k = 0; k < n; ++k) { (*mtp_preds)[(std::size_t)k] = mp[k]; }
  }
  if (run_mtp2) {
    mtp_preds2->resize((std::size_t)n);
    const auto* mp = static_cast<const std::int32_t*>(mvamax2.contents());
    for (int k = 0; k < n; ++k) { (*mtp_preds2)[(std::size_t)k] = mp[k]; }
  }
  return true;
}

void
MetalQwenModel::gdn_replay_(ContextId cid, int keep, const GdnVerifyCache& gc)
{
  // Re-advance every linear layer's GDN recurrent state to S_keep using the
  // verify's cached conv/step inputs (the caller has already restored each
  // state to the round's S0 snapshot). Same kernels + same inputs as the
  // verify's first `keep` tokens -> bit-identical state. No attention /
  // projections / lm_head.
  if (keep <= 0 || gc.layers.empty()) { return; }
  const Config& c = _cfg;
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, Kc = c.gdn_conv_kernel;
  const float eps = c.rms_eps;
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer convout = buf((std::size_t)keep * Cd);
  SharedBuffer ygdn = buf((std::size_t)keep * vald);
  const std::size_t kb_off = (std::size_t)keep * keyd * 2;
  const std::size_t vb_off = 2 * kb_off;
  const bool gdn4 =
      _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) && !_gdn_force_v1;
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    for (std::size_t gi = 0; gi < gc.layers.size(); ++gi) {
      const int L = gc.layers[gi];
      const Layer& ly = _layers[(std::size_t)L];
      const SharedBuffer* csb = _ctx->conv_state(cid, L);
      const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
      if (csb == nullptr || ssb == nullptr) { continue; }
      enc.set_function(_fn_gdn_conv1d);
      enc.set_buffer(0, *csb); enc.set_buffer(1, gc.qkv[gi]);
      enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
      enc.set_constant(4, keep); enc.set_constant(5, Cd);
      enc.set_constant(6, Kc); enc.set_constant(7, Cd);   // x_stride = Cd
      enc.set_constant(8, keyd); enc.set_buffer(9, *csb);
      enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
      rms(convout, 0, _gdn_qscale, convout, 0, keep * Hk, Dk);
      rms(convout, kb_off, _gdn_kscale, convout, kb_off, keep * Hk, Dk);
      enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
      enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
      enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gc.gbuf[gi]);
      enc.set_buffer(4, gc.betabuf[gi]); enc.set_buffer(5, *ssb);
      enc.set_buffer(6, ygdn); enc.set_buffer(7, *ssb);
      enc.set_constant(8, keep);
      enc.set_constant(9, _kquant ? -Hk : Hk);   // strided GQA for GGUF
      enc.set_constant(10, Hv);
      const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
      enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
    }
  }
  stream.commit().wait();
}

std::vector<float>
MetalQwenModel::embed_text(const std::vector<std::int32_t>& ids)
{
  std::vector<float> out;
  if (ids.empty()) { return out; }
  SharedBuffer e =
      _muxer->fetch_text(std::span<const std::int32_t>(ids.data(), ids.size()));
  if (e.empty()) { return out; }
  out.resize(ids.size() * (std::size_t)_cfg.hidden);
  read_elt_(e.contents(), out.data(), out.size(), _cfg.use_bf16);
  return out;
}

std::int32_t
MetalQwenModel::forward_argmax(std::int32_t token_id)
{
  // Prefer the in-stream fast path (embed gather + on-GPU argmax folded into
  // the decode command buffer -- no separate embed submit, no [vocab] host
  // pull); fall back to forward()+host argmax when unavailable.
  const std::int32_t r = decode_step_fast(_cid, token_id);
  if (r != std::numeric_limits<std::int32_t>::min()) { return r; }
  std::vector<float> l = forward(token_id);
  std::int32_t best = 0;
  float bv = l.empty() ? 0.0f : l[0];
  for (std::size_t i = 1; i < l.size(); ++i) {
    if (l[i] > bv) { bv = l[i]; best = (std::int32_t)i; }
  }
  return best;
}

bool
MetalQwenModel::decode_pipelined(
    ContextId cid, std::int32_t first_token, int n_steps,
    std::vector<std::int32_t>& out_ids, float temperature, float top_p,
    std::uint64_t seed)
{
  out_ids.clear();
  if (n_steps <= 0) { return true; }
  const bool embed_ok = _kquant ? _embed_is_q6k : _fn_embed.valid();
  if (!embed_ok || !_fn_argmax.valid() || !_fn_sample.valid()) {
    return false;
  }
  if (!ensure_decode_scratch_()) { return false; }
  const Config& c = _cfg;
  const int H = c.hidden;
  const bool greedy = (temperature <= 0.0f);

  // GPU-resident token chain (gen_ids[0]=seed; argmax/sample writes [s+1],
  // embed reads [s] -- the token never round-trips to the host).
  metal_compute::SharedBuffer gen_ids = _mc->make_shared_buffer(
      (std::size_t)(n_steps + 1) * sizeof(std::int32_t));
  if (gen_ids.empty()) { return false; }
  static_cast<std::int32_t*>(gen_ids.contents())[0] = first_token;

  // Per-step page-table slices for the full-attn layers (host fills all
  // ahead of GPU execution, so they must not alias). GDN conv/ssm state
  // is updated IN PLACE each step -- the event chain orders those writes.
  const int pt_stride = _ctx->max_pages() * 3;
  metal_compute::SharedBuffer pgt = _mc->make_shared_buffer(
      (std::size_t)n_steps * pt_stride * sizeof(std::int32_t));
  if (pgt.empty()) { return false; }

  // Top-p sampling scratch: [vocab] cached softmax weights + a (here
  // unused, rep=1/pres=0) seen-set the shared sampler kernel still binds.
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
      // Zero-copy embed: gather row gen_ids[s] -> _d_x in this command
      // buffer (no muxer command buffer + host memcpy).
      if (_kquant) {
        embed_q6k_(enc, gen_ids, (std::size_t)s, _d_x, 1);
      } else {
        enc.set_function(_fn_embed);
        enc.set_buffer(0, gen_ids, (std::size_t)s * sizeof(std::int32_t));
        enc.set_buffer(1, _embed_w);
        enc.set_buffer(2, _embed_s);
        enc.set_buffer(3, _embed_b);
        enc.set_buffer(4, _d_x);
        enc.set_constant(5, H);
        enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
      }

      // Text decode: rope position == KV slot position.
      encode_decode_step_(enc, cid, pos, pos, page_off, n_pages, slot, pgt,
                          pt_off);

      // Next token -> gen_ids[s+1], on-GPU (greedy argmax or temp+top-p).
      const std::uint32_t step_seed =
          (std::uint32_t)(seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
      encode_sample_(enc, _d_logits, gen_ids,
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

bool
MetalQwenModel::ensure_sampler_scratch_()
{
  if (_smp_scratch_ready) { return true; }
  // [2*kArgmaxM] f32 (val,idx) partials for the two-stage argmax.
  _d_argmax_part =
      _mc->make_shared_buffer((std::size_t)2 * kArgmaxM * sizeof(float));
  // Histogram-sampler scratch (multi-tg sample_*_f16). Sizes derive from
  // kSampM (stage-1 tgs) and kSampB (bins); kSampB MUST match the .metal value.
  _d_smp_maxpart =
      _mc->make_shared_buffer((std::size_t)kSampM * sizeof(float));
  _d_smp_hpart = _mc->make_shared_buffer(
      (std::size_t)kSampM * (2 * kSampB + 1) * sizeof(float));
  _d_smp_hist =
      _mc->make_shared_buffer((std::size_t)(2 * kSampB + 1) * sizeof(float));
  _d_smp_maxl = _mc->make_shared_buffer(sizeof(float));
  _d_smp_wt   = _mc->make_shared_buffer(sizeof(float));
  _d_smp_pickpart =
      _mc->make_shared_buffer((std::size_t)2 * kSampM * sizeof(float));
  _smp_scratch_ready = !_d_argmax_part.empty() && !_d_smp_hpart.empty();
  return _smp_scratch_ready;
}

// Two-stage parallel argmax over logits[logits_off..] -> out[out_off].
// VPIPE_QWEN_ARGMAX1=1 forces the single-tg argmax (A/B). Token/bit-exact.
void
MetalQwenModel::encode_argmax_(metal_compute::ComputeEncoder& enc,
                               const metal_compute::SharedBuffer& logits,
                               std::size_t logits_off,
                               const metal_compute::SharedBuffer& out,
                               std::size_t out_off, int vocab)
{
  static const bool kForce1 = std::getenv("VPIPE_QWEN_ARGMAX1") != nullptr;
  if (!kForce1 && _fn_argmax_partial.valid() && _fn_argmax_combine.valid()
      && ensure_sampler_scratch_()) {
    enc.set_function(_fn_argmax_partial);
    enc.set_buffer(0, logits, logits_off);
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
    enc.set_buffer(0, logits, logits_off);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, vocab);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  }
}

// Histogram multi-tg sampler over logits[logits_off..] -> out[out_off],
// updating seen[seen_off..]. VPIPE_QWEN_SAMPLE1=1 (or missing kernels/scratch)
// falls back to the single-tg _fn_sample. Deterministic (integer-atomic hist).
void
MetalQwenModel::encode_sample_core_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& logits, std::size_t logits_off,
    const metal_compute::SharedBuffer& out, std::size_t out_off,
    const metal_compute::SharedBuffer& sample_ws, std::size_t ws_off,
    const metal_compute::SharedBuffer& seen, std::size_t seen_off,
    const GpuSamplerParams& sp, std::uint32_t step_seed, int vocab)
{
  static const bool kForce1 = std::getenv("VPIPE_QWEN_SAMPLE1") != nullptr;
  const bool hist_ok =
      _fn_smp_max_partial.valid() && _fn_smp_max_combine.valid() &&
      _fn_smp_zhist_partial.valid() && _fn_smp_zhist_combine.valid() &&
      _fn_smp_thresh.valid() && _fn_smp_pick_partial.valid() &&
      _fn_smp_pick_combine.valid() && ensure_sampler_scratch_();
  if (kForce1 || !hist_ok) {
    enc.set_function(_fn_sample);
    enc.set_buffer(0, logits, logits_off);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, vocab);
    enc.set_constant(3, sp.temperature);
    enc.set_constant(4, sp.top_p);
    enc.set_constant(5, step_seed);
    enc.set_buffer(6, sample_ws, ws_off);
    enc.set_constant(7, sp.n_iter);
    enc.set_constant(8, sp.repetition_penalty);
    enc.set_constant(9, sp.presence_penalty);
    enc.set_constant(10, sp.top_k);
    enc.set_constant(11, sp.min_p);
    enc.set_buffer(12, seen, seen_off);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
    return;
  }
  const int M = kSampM;
  const float rep = sp.repetition_penalty, pres = sp.presence_penalty;
  // Pass A: max of penalised eff (partial -> combine).
  enc.set_function(_fn_smp_max_partial);
  enc.set_buffer(0, logits, logits_off);
  enc.set_buffer(1, _d_smp_maxpart);
  enc.set_constant(2, vocab);
  enc.set_constant(3, M);
  enc.set_constant(4, rep);
  enc.set_constant(5, pres);
  enc.set_buffer(6, seen, seen_off);
  enc.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_smp_max_combine);
  enc.set_buffer(0, _d_smp_maxpart);
  enc.set_buffer(1, _d_smp_maxl);
  enc.set_constant(2, M);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
  // Pass B: ws + partial Z + per-tg histogram, then combine.
  enc.set_function(_fn_smp_zhist_partial);
  enc.set_buffer(0, logits, logits_off);
  enc.set_buffer(1, sample_ws, ws_off);
  enc.set_buffer(2, _d_smp_hpart);
  enc.set_constant(3, vocab);
  enc.set_constant(4, M);
  enc.set_constant(5, sp.temperature);
  enc.set_constant(6, rep);
  enc.set_constant(7, pres);
  enc.set_buffer(8, seen, seen_off);
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
  enc.set_buffer(0, logits, logits_off);
  enc.set_buffer(1, sample_ws, ws_off);
  enc.set_buffer(2, _d_smp_pickpart);
  enc.set_constant(3, vocab);
  enc.set_constant(4, M);
  enc.set_constant(5, sp.temperature);
  enc.set_constant(6, step_seed);
  enc.set_constant(7, rep);
  enc.set_constant(8, pres);
  enc.set_buffer(9, seen, seen_off);
  enc.set_buffer(10, _d_smp_wt);
  enc.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
  enc.set_function(_fn_smp_pick_combine);
  enc.set_buffer(0, _d_smp_pickpart);
  enc.set_buffer(1, out, out_off);
  enc.set_constant(2, M);
  enc.set_buffer(3, seen, seen_off);
  enc.dispatch({256, 1, 1}, {256, 1, 1});
}

void
MetalQwenModel::encode_sample_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& logits,
    const metal_compute::SharedBuffer& out_id, std::size_t out_off,
    const GpuSamplerParams& sp, std::uint32_t step_seed,
    const metal_compute::SharedBuffer& sample_ws,
    const metal_compute::SharedBuffer& seen)
{
  if (sp.greedy) {
    encode_argmax_(enc, logits, 0, out_id, out_off, _cfg.vocab);
  } else {
    encode_sample_core_(enc, logits, 0, out_id, out_off, sample_ws, 0,
                        seen, 0, sp, step_seed, _cfg.vocab);
  }
}

bool
MetalQwenModel::pdecode_begin(ContextId cid, std::int32_t first_token,
                              std::span<const std::int32_t> prompt,
                              const GpuSamplerParams& sp, int max_tokens,
                              int rope_first)
{
  const bool embed_ok = _kquant ? _embed_is_q6k : _fn_embed.valid();
  if (!embed_ok || !_fn_argmax.valid() ||
      (!sp.greedy && !_fn_sample.valid())) {
    return false;
  }
  if (!ensure_decode_scratch_()) { return false; }
  if (max_tokens < 1) { max_tokens = 1; }

  PDecode& pd = _pdec[cid.v];
  pd = PDecode{};                 // reset any stale session for this cid
  pd.cid = cid;
  pd.sp = sp;
  pd.produced = 1;                // gen_ids[0] = first_token
  pd.committed = 1;               // next forward writes gen_ids[1]
  // Default depth-1 (no run-ahead). The GDN ssm/conv ring makes depth>=2
  // CORRECT (token-exact + rollback-safe), but measured depth-2 ~= depth-1 for
  // Qwen3.5-4B: its CPU-encode bubble (~0.26ms) is tiny vs the ~21ms/token GPU
  // decode, so run-ahead has almost nothing to hide (unlike gemma-e4b's
  // +3.6%). Keeping depth-1 avoids the ~96MB ssm shadow alloc for no gain.
  // VPIPE_QWEN_PDECODE_DEPTH=2 opts in (lower variance; MTP foundation).
  pd.depth = 1;
  if (const char* e = std::getenv("VPIPE_QWEN_PDECODE_DEPTH")) {
    pd.depth = std::max(1, std::min(4, std::atoi(e)));
  }
  pd.cap = max_tokens + 1 + pd.depth;
  // mROPE anchor: the first decode token lands at KV slot seq_len with
  // rotary position `rope_first` (-1 => sequential, rope == KV slot).
  pd.rope_base = rope_first;
  pd.kv_base = _ctx->seq_len_of(cid);

  pd.gen_ids = _mc->make_shared_buffer(
      (std::size_t)pd.cap * sizeof(std::int32_t));
  const int pt_stride = _ctx->max_pages() * 3;
  pd.pgt = _mc->make_shared_buffer(
      (std::size_t)pt_stride * sizeof(std::int32_t));
  if (pd.gen_ids.empty() || pd.pgt.empty()) { _pdec.erase(cid.v); return false; }
  static_cast<std::int32_t*>(pd.gen_ids.contents())[0] = first_token;

  if (!sp.greedy) {
    pd.sample_ws = _mc->make_shared_buffer((std::size_t)_cfg.vocab * 2);
    pd.seen = _mc->make_shared_buffer((std::size_t)_cfg.vocab);   // uint8
    if (pd.sample_ws.empty() || pd.seen.empty()) {
      _pdec.erase(cid.v); return false;
    }
    // Prime the penalty seen-set with the prompt tokens + first id.
    std::memset(pd.seen.contents(), 0, (std::size_t)_cfg.vocab);
    auto* sb = static_cast<std::uint8_t*>(pd.seen.contents());
    for (std::int32_t t : prompt) {
      if (t >= 0 && t < _cfg.vocab) { sb[t] = 1; }
    }
    if (first_token >= 0 && first_token < _cfg.vocab) { sb[first_token] = 1; }
  }

  pd.stream = _mc->make_command_stream();
  pd.ev = _mc->make_event();

  // Arm the GDN recurrent-state run-ahead ring for depth>1 (no-op on a dense
  // model with no GDN layers, or when depth==1). If the shadow alloc fails,
  // fall back to depth-1 (no run-ahead) rather than abort decode. Done LAST so
  // there is no failure path after it that could leave the ring half-armed.
  if (!_ctx->gdn_ring_begin(cid, pd.depth)) { pd.depth = 1; }
  return true;
}

bool
MetalQwenModel::pdecode_commit(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return false; }
  PDecode& pd = it->second;
  if ((int)pd.ring.size() >= pd.depth) { return false; }  // pipeline full
  const int in_idx = pd.committed - 1;
  const int out_idx = pd.committed;
  if (out_idx >= pd.cap) { return false; }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return false; }
  const int pos = slot.position;
  // Rotary position: sequential text decode (rope_base<0) uses the KV
  // slot; after a multimodal prefill rope advances from the mROPE anchor.
  const int rpos = (pd.rope_base < 0) ? pos
                                      : pd.rope_base + (pos - pd.kv_base);
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(pd.pgt.contents()));

  const std::uint64_t s = pd.gpu_step;
  pd.stream.encode_wait(pd.ev, s);
  {
    ComputeEncoder enc = pd.stream.begin_compute();
    // In-stream embed gather of gen_ids[in_idx] -> _d_x.
    if (_kquant) {
      embed_q6k_(enc, pd.gen_ids, (std::size_t)in_idx, _d_x, 1);
    } else {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, pd.gen_ids,
                     (std::size_t)in_idx * sizeof(std::int32_t));
      enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, _embed_s);
      enc.set_buffer(3, _embed_b);
      enc.set_buffer(4, _d_x);
      enc.set_constant(5, _cfg.hidden);
      enc.dispatch({(unsigned)_cfg.hidden, 1, 1}, {256, 1, 1});
    }

    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot, pd.pgt, 0);

    const std::uint32_t step_seed = (std::uint32_t)(
        pd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
    encode_sample_(enc, _d_logits, pd.gen_ids,
                   (std::size_t)out_idx * sizeof(std::int32_t), pd.sp,
                   step_seed, pd.sample_ws, pd.seen);
  }
  pd.stream.encode_signal(pd.ev, s + 1);
  PDecode::InFlight f;
  f.fence = pd.stream.commit();
  f.idx = out_idx;
  pd.ring.push_back(std::move(f));
  pd.gpu_step = s + 1;
  pd.committed = out_idx + 1;
  // One GDN advance per committed forward: the slot this step WROTE becomes
  // the next commit's read slot. No-op when the ring is off (depth-1).
  _ctx->gdn_ring_advance(cid);
  return true;
}

std::int32_t
MetalQwenModel::pdecode_next(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return -1; }
  PDecode& pd = it->second;
  if (pd.ring.empty()) { return -1; }
  PDecode::InFlight f = std::move(pd.ring.front());
  pd.ring.pop_front();
  f.fence.wait();
  const std::int32_t tok =
      static_cast<const std::int32_t*>(pd.gen_ids.contents())[f.idx];
  pd.produced = f.idx + 1;
  return tok;
}

void
MetalQwenModel::pdecode_end(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return; }
  PDecode& pd = it->second;
  for (auto& f : pd.ring) { f.fence.wait(); }
  pd.ring.clear();
  // Discard a run-ahead speculative tail: roll the paged KV AND the GDN
  // recurrent (ssm/conv) ring back to the last drained token, matching the
  // synchronous loop (a stop token's forward is never committed). depth-1
  // has committed == produced -> spec 0 (no behavior change); gdn_ring_end
  // then restores the canonical GDN buffers and disarms the ring. All ring
  // fences are already waited above, so the end-time handle swap is safe.
  const int spec = pd.committed - pd.produced;
  if (spec > 0) { _ctx->kv_rollback(pd.cid, spec); }
  _ctx->gdn_ring_rollback(pd.cid, spec);
  _ctx->gdn_ring_end(pd.cid);
  _pdec.erase(it);
}

// ---------------------------------------------------------------------
// MTP speculative decode (MTP head FUSED into the verify + rollback).
// ---------------------------------------------------------------------
void
MetalQwenModel::mtp_ctx_reset_()
{
  if (!_mtp.ok) { return; }
  const int seq = _mtp.ctx->seq_len_of(_mtp.cid);
  if (seq > 0) { _mtp.ctx->kv_rollback(_mtp.cid, seq); }
}

int
MetalQwenModel::mtp_seed_prefix_(std::int32_t first_token)
{
  if (!_mtp.ok || !_mtp_persist || !_mtp_seed_enabled) { return 0; }
  if (!_mtp_prefix_valid || _mtp_prefix_len <= 0 || _mtp_prefix_h.empty()) {
    return 0;
  }
  // The prenorm-hidden diagnostic conditions the head on raw x, but the seed
  // captured POST-norm hiddens -> skip seeding on that (non-default) path.
  if (std::getenv("VPIPE_MTP_PRENORM_HIDDEN") != nullptr) { return 0; }
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int gate = c.attn_output_gate ? 1 : 0;
  const int qdo = gate ? 2 * qd : qd;
  const int Nfqkv = qdo + 2 * kd;
  const float eps = c.rms_eps;
  (void)Hq;
  // Seed the most-recent M prefix positions that fit the MTP page after leaving
  // headroom for the first round's append (a partial seed keeps the tail; RoPE
  // attention is shift-invariant so a tail at slots 0.. preserves all relative
  // distances). Single page (page_tokens == max_seq), so one append + dispatch.
  const int cap = _mtp.ctx->page_tokens();
  const int reserve = 16;
  const int s = _mtp_prefix_len;
  int M = (s > cap - reserve) ? (cap - reserve) : s;
  if (M <= 0) { return 0; }
  const int tail0 = s - M;            // first prefix position seeded
  ContextManager::AppendSlot mslot = _mtp.ctx->append(_mtp.cid, M);
  if (!mslot.valid()) {
    const int seq = _mtp.ctx->seq_len_of(_mtp.cid);
    if (seq > 0) { _mtp.ctx->kv_rollback(_mtp.cid, seq); }
    mslot = _mtp.ctx->append(_mtp.cid, M);
    if (!mslot.valid()) { return 0; }
  }
  const std::size_t m_page_off =
      (std::size_t)mslot.page_id.v * _mtp.ctx->page_stride_bytes();
  const int mpt = _mtp.ctx->page_tokens();
  const SharedBuffer& mkp = *_mtp.ctx->kpool(0);
  const SharedBuffer& mvp = *_mtp.ctx->vpool(0);
  Layer& mly = _mtp.lyr;

  // Conditioning tokens: position (tail0 + i) conditions on the token at
  // (tail0 + i + 1); the last seeded position uses the decode's first token.
  SharedBuffer ids =
      _mc->make_shared_buffer((std::size_t)M * sizeof(std::int32_t));
  if (ids.empty()) { return 0; }
  auto* idp = static_cast<std::int32_t*>(ids.contents());
  for (int i = 0; i < M; ++i) {
    const int t = tail0 + i;
    idp[i] = (t + 1 < s) ? _mtp_prefix_ids[(std::size_t)(t + 1)] : first_token;
  }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer emb_P = buf((std::size_t)M * H);
  SharedBuffer mnorm_e = buf((std::size_t)M * H);
  SharedBuffer mnorm_h = buf((std::size_t)M * H);
  SharedBuffer mcomb = buf((std::size_t)M * H);
  SharedBuffer mctmp = buf((std::size_t)M * H);
  SharedBuffer hn = buf((std::size_t)M * H);
  SharedBuffer mqfull = buf((std::size_t)M * Nfqkv);
  SharedBuffer kbuf = buf((std::size_t)M * kd);
  SharedBuffer vbuf = buf((std::size_t)M * kd);
  SharedBuffer kt = buf((std::size_t)M * kd);
  SharedBuffer vt = buf((std::size_t)M * kd);
  if (emb_P.empty() || hn.empty() || mqfull.empty() || kt.empty()
      || vt.empty()) {
    return 0;
  }
  const std::size_t hsrc_off = (std::size_t)tail0 * H * 2;   // tail of _mtp_h

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out, int Hh,
                      int S, int W, int off, int block, int gstride) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in, 0); enc.set_buffer(1, out, 0);
      enc.set_constant(2, Hh); enc.set_constant(3, S); enc.set_constant(4, W);
      enc.set_constant(5, off); enc.set_constant(6, block);
      enc.set_constant(7, gstride);
      enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                   {(unsigned)D, 1, 1});
    };
    // combined = fc_e@norm_e(emb(ids)) + fc_h@norm_h(hsrc); MTP-layer in_ln.
    if (_kquant) {
      embed_q6k_(enc, ids, 0, emb_P, M);
    } else if (_dense || _dense_embed) {
      enc.set_function(_fn_embed_dense);
      enc.set_buffer(0, ids); enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, emb_P); enc.set_constant(3, H);
      enc.dispatch({(unsigned)H, (unsigned)M, 1}, {256, 1, 1});
    } else {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, ids);
      enc.set_buffer(1, _embed_w); enc.set_buffer(2, _embed_s);
      enc.set_buffer(3, _embed_b); enc.set_buffer(4, emb_P);
      enc.set_constant(5, H);
      enc.dispatch({(unsigned)H, (unsigned)M, 1}, {256, 1, 1});
    }
    rms(emb_P, 0, _mtp.prenorm_e, mnorm_e, 0, M, H);
    rms(_mtp_prefix_h, hsrc_off, _mtp.prenorm_h, mnorm_h, 0, M, H);
    dense_gemm_(enc, _mtp.fc_e, mnorm_e, mcomb, H, H, M);
    dense_gemm_(enc, _mtp.fc_h, mnorm_h, mctmp, H, H, M);
    residual(mcomb, mctmp, mcomb, M * H);
    rms(mcomb, 0, mly.in_ln, hn, 0, M, H);
    // q|k|v projection -> slice k,v only (q/attention not needed to seed KV).
    if (_kquant) {
      kqmv_batch_(enc, mly.kqk_t, mly.kqk, hn, mqfull, H, mly.kqk_n, M,
                  Nfqkv, 0);
      kqmv_batch_(enc, mly.kqv_t, mly.kqv, hn, mqfull, H, kd, M, Nfqkv,
                  mly.kqk_n);
    } else {
      vqmm_(enc, M, mly.qw, mly.qs, mly.qb, 4, hn, mqfull, H, Nfqkv);
    }
    hslice(mqfull, kbuf, M, Nfqkv, kd, qdo, 0, 0);
    hslice(mqfull, vbuf, M, Nfqkv, kd, qdo + kd, 0, 0);
    rms(kbuf, 0, mly.k_norm, kbuf, 0, M * Hkv, D);
    transpose(kbuf, kt, M, Hkv);
    transpose(vbuf, vt, M, Hkv);
    // RoPE k at positions mslot.position .. +M-1 (rope_partial adds the row).
    enc.set_function(_fn_rope_partial);
    enc.set_buffer(0, kt); enc.set_buffer(1, _inv_freq);
    enc.set_constant(2, Hkv); enc.set_constant(3, M);
    enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
    enc.set_constant(6, mslot.position);
    enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)M, (unsigned)Hkv},
                 {(unsigned)(c.rotary_dim / 2), 1, 1});
    // Write K/V to the MTP ctx at slots [mslot.slot_offset, +M).
    for (int two = 0; two < 2; ++two) {
      enc.set_function(_fn_kv_write_paged);
      enc.set_buffer(0, two == 0 ? kt : vt);
      enc.set_buffer(1, two == 0 ? mkp : mvp, m_page_off);
      enc.set_constant(2, mpt); enc.set_constant(3, D);
      enc.set_constant(4, M); enc.set_constant(5, 0);
      enc.set_constant(6, mslot.slot_offset);
      enc.dispatch({(unsigned)D, (unsigned)M, (unsigned)Hkv},
                   {(unsigned)D, 1, 1});
    }
  }
  stream.commit().wait();
  return M;
}

bool
MetalQwenModel::mtp_decode(ContextId cid, std::int32_t first_token,
                           int n_steps, std::vector<std::int32_t>& out_ids,
                           int draft_len, long* accepted_out, long* rounds_out,
                           const MtpDecodeCtl& ctl)
{
  out_ids.clear();
  if (ctl.hit_stop) { *ctl.hit_stop = false; }
  // The MTP head is fused into mtp_verify_chunk_, supported on the de-fused
  // affine (mixed) path, the native k-quant (GGUF NextN) path, and the dense
  // (raw-HF bf16/f16) path.
  if (!_mtp.ok || n_steps <= 0 || !(_mixed || _kquant || _dense)) {
    return false;
  }
  if (!ensure_decode_scratch_()) { return false; }
  const Config& c = _cfg;
  // Persistent MTP self-attention KV over this decode's committed positions
  // (default ON; VPIPE_MTP_NO_PERSIST reverts to the per-round wipe). Reset the
  // MTP ctx ONCE here so the head starts fresh and accumulates as the decode
  // commits tokens; mtp_verify_chunk_ then appends without wiping and the loop
  // below rolls back each round's rejected draft window.
  _mtp_persist = (std::getenv("VPIPE_MTP_NO_PERSIST") == nullptr);
  if (_mtp_persist) {
    mtp_ctx_reset_();
    // Prefix-seed the drafter's KV with the captured prompt (auto when an MTP
    // head + a text prefill preceded this; no-op otherwise). The head then
    // attends over the whole prompt from round 1, lifting draft acceptance.
    mtp_seed_prefix_(first_token);
  }
  // Perf-investigation knobs (default off): VPIPE_MTP_DRAFT_LEN overrides the
  // speculative depth (1 = two drafts/round, 2 = three); VPIPE_MTP_PROFILE
  // prints a per-round wall-clock breakdown so the speedup ceiling can be
  // attributed to the verify forward vs the partial-accept re-run vs the GDN
  // snapshot. mtp_verify_chunk_ commits+waits internally, so these wraps time
  // the GPU+sync of each piece.
  if (const char* e = std::getenv("VPIPE_MTP_DRAFT_LEN")) {
    const int d = std::atoi(e);
    if (d >= 1) { draft_len = d; }
  }
  static const bool mtp_prof = (std::getenv("VPIPE_MTP_PROFILE") != nullptr);
  // Verify sub-profile (VPIPE_MTP_VPROFILE): reset the per-segment accumulators
  // mtp_verify_chunk_ fills so this decode's breakdown stands alone.
  static const bool vprof_d = (std::getenv("VPIPE_MTP_VPROFILE") != nullptr);
  if (vprof_d) { g_vp_main = g_vp_vhead = g_vp_mtp = 0.0; g_vp_n = 0; }
  using mtp_clock = std::chrono::steady_clock;
  double prof_verify_ms = 0.0, prof_rerun_ms = 0.0, prof_snap_ms = 0.0;
  // L-C step wall-time (host post()/sample() grind vs the GPU lc_* kernels).
  static const bool lc_prof = (std::getenv("VPIPE_MTP_LC_PROFILE") != nullptr);
  double prof_lc_ms = 0.0;
  long prof_reruns = 0;
  auto prof_ms = [](mtp_clock::time_point a, mtp_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  // k-quant gathers the MTP drafter's embed via embed_q6k (tied q6_K table),
  // not the affine _fn_embed; both paths still need the per-position argmax.
  const bool kembed_ok = _kquant ? (_embed_is_q6k && _fn_embed_q6k.valid())
                       : ((_dense || _dense_embed) ? _fn_embed_dense.valid()
                                                   : _fn_embed.valid());
  if (!kembed_ok || !_fn_argmax.valid()) { return false; }
  // Speculative SAMPLING needs the sample kernel.
  const bool sampling = !ctl.sampler.greedy;
  if (sampling && !_fn_sample.valid()) { return false; }

  // KV slot of the first decoded token (= seq_len now). Anchors both the RoPE
  // offset and (for sampling) the per-slot seed so the verify reproduces the
  // serial decode path's positions AND its per-slot randomness.
  const int seed_slot0 = _ctx->seq_len_of(cid);

  // RoPE offset: rope_first (the first decoded token's rotary position) minus
  // its KV slot (the current seq length). 0 for sequential text; the mROPE
  // gap after a multimodal prefill otherwise. Constant across the whole decode
  // (every token's rope advances in lockstep with its KV slot), so the verify
  // just adds it to each token's slot -- matching the serial decode path.
  const int rope_delta =
      (ctl.rope_first < 0) ? 0 : (ctl.rope_first - seed_slot0);

  // GDN recurrent-state snapshot (one conv + ssm buffer per linear layer) --
  // the FALLBACK path (VPIPE_MTP_NO_RING / ring alloc failure). The batched
  // verify advances the canonical GDN state in place by K; a partial accept
  // restores this snapshot and re-runs the accepted tokens (gdn_replay_). The
  // default path instead drives the GDN ring (gdn_ring_begin below): the verify
  // writes each token's state to a DISTINCT ring slot (state-in cur+p, state-out
  // cur+p+1) -- the cursor is fixed across the verify (advanced once, by `keep`,
  // afterward), so all layers stay batched per-position and rollback is a pure
  // cursor move, no host copy / replay.
  std::vector<int> glayers;
  std::vector<SharedBuffer> gconv, gssm;
  for (int L = 0; L < c.n_layers; ++L) {
    if (c.layer_is_full(L)) { continue; }
    const SharedBuffer* cs = _ctx->conv_state(cid, L);
    const SharedBuffer* ss = _ctx->ssm_state(cid, L);
    if (cs == nullptr || ss == nullptr) { continue; }
    glayers.push_back(L);
    gconv.push_back(_mc->make_shared_buffer(cs->byte_size()));
    gssm.push_back(_mc->make_shared_buffer(ss->byte_size()));
  }
  auto snapshot_gdn = [&](bool save) {
    for (std::size_t i = 0; i < glayers.size(); ++i) {
      const SharedBuffer* cs = _ctx->conv_state(cid, glayers[i]);
      const SharedBuffer* ss = _ctx->ssm_state(cid, glayers[i]);
      if (save) {
        std::memcpy(gconv[i].contents(), cs->contents(), cs->byte_size());
        std::memcpy(gssm[i].contents(), ss->contents(), ss->byte_size());
      } else {
        std::memcpy(cs->contents(), gconv[i].contents(), cs->byte_size());
        std::memcpy(ss->contents(), gssm[i].contents(), ss->byte_size());
      }
    }
  };

  // GDN run-ahead ring: size R = K_max+1 = (draft_len+1)+1 slots so the verify
  // can write every speculative token's recurrent state into a DISTINCT slot
  // (state-in cur+p, state-out cur+p+1) and a partial accept rolls back by a
  // pure cursor move (advance by `keep`) -- no host snapshot, no gdn_replay_.
  // Falls back to the in-place snapshot path on alloc failure or VPIPE_MTP_NO_RING.
  bool use_ring = (std::getenv("VPIPE_MTP_NO_RING") == nullptr);
  if (use_ring && !_ctx->gdn_ring_begin(cid, draft_len + 1)) {
    use_ring = false;
  }

  // Depth: draft_len>=2 chains a 2nd MTP application (3 drafts/round); else
  // depth-1 (2 drafts/round). Both are FULLY fused -- the verify emits all the
  // next round's drafts, no separate draft forward.
  const bool d2 = (draft_len >= 2);

  // Per-(linear layer) GDN-input cache for the partial-accept GDN replay: sized
  // for the max drafts/round (depth-1: 2, depth-2: 3). Each verify writes its
  // recurrent-step inputs here; gdn_replay_ reads them on a partial accept.
  // Allocated once, reused across rounds.
  GdnVerifyCache gcache;
  gcache.layers = glayers;
  {
    const int maxK = d2 ? 3 : 2;
    const int Cd = c.gdn_conv_dim, Hv = c.gdn_v_heads;
    for (std::size_t i = 0; i < glayers.size(); ++i) {
      gcache.qkv.push_back(
          _mc->make_shared_buffer((std::size_t)maxK * Cd * 2));
      gcache.gbuf.push_back(
          _mc->make_shared_buffer((std::size_t)maxK * Hv * 4));
      gcache.betabuf.push_back(
          _mc->make_shared_buffer((std::size_t)maxK * Hv * 4));
    }
  }

  // Round state: the next round's drafts, produced by THIS round's fused
  // verify at the accepted boundary. drafts[0] is always the verified next
  // token; d1/d2v are the depth-1/depth-2 MTP drafts (absent for round 1).
  std::int32_t d0 = first_token, d1 = -1, d2v = -1;
  bool have_d1 = false, have_d2 = false;
  long accepted = 0, rounds = 0;

  // Leviathan-Chen rejection sampling (opt-in). Covers depth-1 and depth-2 and
  // temperature + top_k/top_p/min_p (the CPU post-processing replicates the GPU
  // sampler's nucleus). Repetition/presence penalties are NOT modelled -> fall
  // back to the exact-match sampler. qcarry/qcarry2 hold the drafter dists q for
  // the pending drafts d1/d2 (carried from the round that produced them -- the
  // ratio p(d)/q(d) and the residual norm(max(0,p-q)) need them).
  const GpuSamplerParams& sp = ctl.sampler;
  const bool lc_mode =
      _leviathan && !sp.greedy && sp.repetition_penalty == 1.0f
      && sp.presence_penalty == 0.0f && c.vocab > 0;
  // qcarry/qcarry2 are valid whenever a draft is verified: drafts[1]/drafts[2]
  // exist only when have_d1/have_d2 were set last round, which (in lc_mode) set
  // qcarry/qcarry2 in the same block -- so they stay in sync with the drafts.
  std::vector<float> qcarry, qcarry2;
  std::uint64_t lc_rng =
      sp.seed * 0x2545F4914F6CDD1Dull + 0x9E3779B97F4A7C15ull;

  // On-GPU L-C: do the nucleus + accept/residual in lc_accept_f16/lc_sample_f16
  // instead of the host post()/sample() full-vocab grind. Opt-in via set_lc_gpu
  // or VPIPE_MTP_GPU_LC; falls back to the host path on alloc/function failure.
  static const bool gpu_lc_env = (std::getenv("VPIPE_MTP_GPU_LC") != nullptr);
  bool gpu_lc = lc_mode && (_lc_gpu || gpu_lc_env)
                && _fn_lc_sample.valid() && _fn_lc_accept.valid()
                && _fn_lc_sample_batch.valid();
  // Max concurrent sample rows in one batched dispatch (depth-2: bonus is 1,
  // q1/q2 are K=3 rows each); _lc_ws_p must hold that many [vocab] scratches.
  const int lc_kmax = (draft_len >= 2) ? 3 : 2;
  if (gpu_lc) {
    const std::size_t vb = (std::size_t)c.vocab * 2;
    const std::size_t wpb = (std::size_t)lc_kmax * vb;
    if (_lc_ws_p.byte_size() < wpb) { _lc_ws_p = _mc->make_shared_buffer(wpb); }
    if (_lc_ws_q.byte_size() < vb) { _lc_ws_q = _mc->make_shared_buffer(vb); }
    if (_lc_qcarry.byte_size() < vb) {
      _lc_qcarry = _mc->make_shared_buffer(vb);
    }
    if (_lc_qcarry2.byte_size() < vb) {
      _lc_qcarry2 = _mc->make_shared_buffer(vb);
    }
    const std::size_t ob = 16 * sizeof(std::int32_t);
    if (_lc_out.byte_size() < ob) { _lc_out = _mc->make_shared_buffer(ob); }
    if (_lc_seed_in.byte_size() < ob) {
      _lc_seed_in = _mc->make_shared_buffer(ob);
    }
    if (_lc_ws_p.empty() || _lc_ws_q.empty() || _lc_qcarry.empty()
        || _lc_qcarry2.empty() || _lc_out.empty() || _lc_seed_in.empty()) {
      gpu_lc = false;
    }
  }
  // Per-(round,purpose,position) seed for the GPU L-C draws. Counter-based (vs
  // the host's sequential lc_rng) so each draw is an independent deterministic
  // sub-stream -- L-C is lossless either way; the scheme only fixes the seed.
  auto lc_seed32 = [&](long round, int purpose, int i) -> std::uint32_t {
    std::uint64_t s = sp.seed * 0x2545F4914F6CDD1Dull + 0x9E3779B97F4A7C15ull;
    s ^= (std::uint64_t)round * 0x9E3779B97F4A7C15ull;
    s ^= (std::uint64_t)purpose * 0xD1B54A32D192ED03ull;
    s ^= (std::uint64_t)(i + 1) * 0xBF58476D1CE4E5B9ull;
    s ^= s >> 31;
    return (std::uint32_t)s;
  };
  // GPU L-C step config (read once). lc_niter: bisection iters (diagnostic
  // override). lc_hist: histogram nucleus (default) vs the n_iter bisection
  // (VPIPE_MTP_LC_NO_HIST). lc_nofold: run the L-C dispatches in a separate
  // command buffer + sync instead of folding them into the verify's buffer
  // (default folded -> one commit/round); VPIPE_MTP_LC_NOFOLD opts out (A/B).
  int lc_niter = sp.n_iter;
  if (const char* e = std::getenv("VPIPE_MTP_LC_NITER")) {
    const int v = std::atoi(e);
    if (v > 0) { lc_niter = v; }
  }
  const int lc_hist =
      (std::getenv("VPIPE_MTP_LC_NO_HIST") != nullptr) ? 0 : 1;
  const bool lc_nofold = (std::getenv("VPIPE_MTP_LC_NOFOLD") != nullptr);

  bool terminate = false;   // stop token or caller abort
  while (!terminate && (int)out_ids.size() < n_steps) {
    ++rounds;
    std::vector<std::int32_t> drafts;
    drafts.push_back(d0);
    if (have_d1) { drafts.push_back(d1); }
    if (d2 && have_d2) { drafts.push_back(d2v); }
    const int K = (int)drafts.size();

    // ONE fused forward: the main model verifies the K drafts AND the MTP head
    // (run 1x for depth-1, 2x chained for depth-2, in the SAME command buffer
    // on the main preds + hiddens) emits the next round's drafts. The paged KV
    // grows by K, GDN advances by K.
    SharedBuffer xK = _muxer->fetch_text(
        std::span<const std::int32_t>(drafts.data(), drafts.size()));
    if (xK.empty()) { break; }
    auto t_snap0 = mtp_clock::now();
    if (!use_ring) { snapshot_gdn(/*save=*/true); }
    auto t_ver0 = mtp_clock::now();
    prof_snap_ms += prof_ms(t_snap0, t_ver0);
    std::vector<std::int32_t> preds, mpreds, mpreds2;
    std::vector<std::vector<float>> qnew, qnew2;   // L-C: per-pos drafter dists
    // L-C encoder closure for this round: encodes lc_accept (verifier p vs the
    // carried q) + batched lc_sample (bonus + new q1/q2 drafts) reading the
    // verify's stashed logits, writing token ids to _lc_out. Folded into the
    // verify's command buffer via pre_commit (one commit/round); or run in a
    // separate stream below when lc_nofold. Called synchronously this round, so
    // the by-ref captures stay valid.
    std::function<void(ComputeEncoder&)> encode_lc;
    if (gpu_lc) {
      const int V = c.vocab;
      encode_lc = [&, V](ComputeEncoder& enc) {
        auto* sd = static_cast<std::uint32_t*>(_lc_seed_in.contents());
        // Batched nucleus sample: `rows` rows from `src` (starting at logit row
        // `row0`) run as `rows` CONCURRENT threadgroups; out -> _lc_out[out0..],
        // per-row seed lc_seed32(round, purpose, pos0+r) -> _lc_seed_in[sd0..].
        auto sample_batch = [&](const SharedBuffer& src, int row0, int rows,
                                int out0, int sd0, int purpose, int pos0) {
          for (int r = 0; r < rows; ++r) {
            sd[sd0 + r] = lc_seed32(rounds, purpose, pos0 + r);
          }
          enc.set_function(_fn_lc_sample_batch);
          enc.set_buffer(0, src, (std::size_t)row0 * V * 2);
          enc.set_buffer(1, _lc_out, (std::size_t)out0 * sizeof(std::int32_t));
          enc.set_constant(2, V);
          enc.set_constant(3, sp.temperature);
          enc.set_constant(4, sp.top_p);
          enc.set_buffer(5, _lc_seed_in,
                         (std::size_t)sd0 * sizeof(std::uint32_t));
          enc.set_buffer(6, _lc_ws_p);
          enc.set_constant(7, lc_niter);
          enc.set_constant(8, sp.top_k);
          enc.set_constant(9, sp.min_p);
          enc.set_constant(10, lc_hist);
          enc.dispatch({256 * (unsigned)rows, 1, 1}, {256, 1, 1});
        };
        // Accept positions 0..K-2: verify drafts[i+1] against the carried q.
        for (int i = 0; i + 1 < K; ++i) {
          const SharedBuffer& q = (i == 0) ? _lc_qcarry : _lc_qcarry2;
          enc.set_function(_fn_lc_accept);
          enc.set_buffer(0, _lc_vlogits, (std::size_t)i * V * 2);
          enc.set_buffer(1, q);
          enc.set_buffer(2, _lc_out, (std::size_t)i * sizeof(std::int32_t));
          enc.set_constant(3, V);
          enc.set_constant(4, sp.temperature);
          enc.set_constant(5, sp.top_p);
          const std::uint32_t aseed = lc_seed32(rounds, 0, i);
          enc.set_constant(6, aseed);
          enc.set_buffer(7, _lc_ws_p);
          enc.set_buffer(8, _lc_ws_q);
          enc.set_constant(9, lc_niter);
          enc.set_constant(10, sp.top_k);
          enc.set_constant(11, sp.min_p);
          enc.set_constant(12, drafts[(std::size_t)(i + 1)]);
          enc.set_constant(13, lc_hist);
          enc.dispatch({256, 1, 1}, {256, 1, 1});
        }
        // Bonus at position K-1 (sample from the verifier dist p). seeds[0].
        sample_batch(_lc_vlogits, K - 1, 1, K - 1, 0, 1, K - 1);
        // New depth-1 drafts q1 (K concurrent rows). seeds[1..K].
        sample_batch(_lc_mlogits, 0, K, K, 1, 2, 0);
        if (d2) {           // new depth-2 drafts q2 (K concurrent rows).
          sample_batch(_lc_mlogits2, 0, K, 2 * K, 1 + K, 3, 0);
        }
      };
    }
    const bool vok = mtp_verify_chunk_(cid, xK, K, &preds, &mpreds,
                                       d2 ? &mpreds2 : nullptr, rope_delta,
                                       ctl.sampler, seed_slot0,
                                       use_ring ? nullptr : &gcache, lc_mode,
                                       use_ring, nullptr,
                                       (gpu_lc && !lc_nofold)
                                           ? encode_lc
                                           : std::function<void(
                                                 ComputeEncoder&)>{});
    prof_verify_ms += prof_ms(t_ver0, mtp_clock::now());
    if (!vok ||
        (int)preds.size() != K || (int)mpreds.size() != K ||
        (d2 && (int)mpreds2.size() != K)) {
      // Verify aborted: nothing committed this round. The ring's cursor never
      // advanced, so slot `cur` still holds S0 -- finalize it back to canonical.
      if (use_ring) { _ctx->gdn_ring_end(cid); }
      else { snapshot_gdn(/*save=*/false); }
      break;
    }
    auto t_lc0 = mtp_clock::now();
    // Leviathan-Chen step: the GPU kernels already ran -- folded into the
    // verify's command buffer (pre_commit = encode_lc, default) or, when
    // lc_nofold, in a separate stream encoded here. Either way the corrected
    // token ids are now in _lc_out; read them into preds/mpreds(/mpreds2).
    if (lc_mode && gpu_lc) {
      if (lc_nofold) {
        metal_compute::CommandStream stream = _mc->make_command_stream();
        { ComputeEncoder enc = stream.begin_compute(); encode_lc(enc); }
        stream.commit().wait();
      }
      const auto* o = static_cast<const std::int32_t*>(_lc_out.contents());
      for (int i = 0; i < K; ++i) {
        preds[(std::size_t)i] = o[i];
        mpreds[(std::size_t)i] = o[K + i];
        if (d2) { mpreds2[(std::size_t)i] = o[2 * K + i]; }
      }
    }
    // Leviathan-Chen step ON HOST: from the full verifier/MTP logits the verify
    // stashed (_lc_vlogits / _lc_mlogits / _lc_mlogits2), post-process each row
    // into the SAME nucleus distribution the GPU sampler draws from, then run
    // the ratio test + residual/bonus sampling. The accept loop below is
    // unchanged (the L-C decision is encoded into preds so `drafts[i]==
    // preds[i-1]` reproduces accept/reject). New drafts (mpreds) are SAMPLED
    // from the MTP dist q and that dist is carried (qnew) for next round.
    if (lc_mode && !gpu_lc) {
      const int V = c.vocab;
      const float temp = (sp.temperature > 1e-3f) ? sp.temperature : 1e-3f;
      const auto* vl =
          static_cast<const std::uint16_t*>(_lc_vlogits.contents());
      const auto* ml =
          static_cast<const std::uint16_t*>(_lc_mlogits.contents());
      auto h2f = [](std::uint16_t h) -> float {
        const std::uint32_t s = (std::uint32_t)(h & 0x8000u) << 16;
        const std::uint32_t e = (h >> 10) & 0x1Fu;
        const std::uint32_t m = h & 0x3FFu;
        std::uint32_t bits;
        if (e == 0u) {
          if (m == 0u) { bits = s; }
          else {
            std::uint32_t ee = 127u - 15u + 1u, mm = m;
            while ((mm & 0x400u) == 0u) { mm <<= 1; --ee; }
            mm &= 0x3FFu;
            bits = s | (ee << 23) | (mm << 13);
          }
        } else if (e == 0x1Fu) {
          bits = s | 0x7F800000u | (m << 13);
        } else {
          bits = s | ((e - 15u + 127u) << 23) | (m << 13);
        }
        float f; std::memcpy(&f, &bits, 4); return f;
      };
      auto next_u = [&]() -> float {
        lc_rng += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = lc_rng;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= (z >> 31);
        return (float)((double)(z >> 11) * (1.0 / 9007199254740992.0));
      };
      auto sample = [&](const std::vector<float>& d, float u) -> std::int32_t {
        float cdf = 0.0f;
        for (int v = 0; v < V; ++v) {
          cdf += d[(std::size_t)v];
          if (u < cdf) { return v; }
        }
        return V - 1;
      };
      // Post-process logits into the SAME distribution the GPU sampler draws
      // from: softmax weights w=exp((logit-max)/temp), then the top_k weight
      // threshold + min_p floor + top_p nucleus (binary search, matching
      // sample_topp_f16), renormalized over the kept set. Pure temperature
      // skips the nucleus search.
      const int n_iter = (sp.n_iter > 0) ? sp.n_iter : 24;
      const bool nucleus = (sp.top_k > 0 && sp.top_k < V) || sp.top_p < 1.0f
                           || sp.min_p > 0.0f;
      auto post = [&](const std::uint16_t* row, std::vector<float>& out) {
        out.resize((std::size_t)V);
        float maxl = -1e30f;
        for (int v = 0; v < V; ++v) {
          const float l = h2f(row[v]);
          out[(std::size_t)v] = l; if (l > maxl) { maxl = l; }
        }
        const float inv_t = 1.0f / temp;
        float Z = 0.0f;
        for (int v = 0; v < V; ++v) {
          const float w = std::exp((out[(std::size_t)v] - maxl) * inv_t);
          out[(std::size_t)v] = w; Z += w;
        }
        if (!nucleus) {
          const float inv = (Z > 0.0f) ? 1.0f / Z : 0.0f;
          for (int v = 0; v < V; ++v) { out[(std::size_t)v] *= inv; }
          return;
        }
        float t_k = 0.0f;
        if (sp.top_k > 0 && sp.top_k < V) {
          float lo = 0.0f, hi = 1.0f;
          for (int it = 0; it < n_iter; ++it) {
            const float mid = 0.5f * (lo + hi);
            int cnt = 0;
            for (int v = 0; v < V; ++v) {
              if (out[(std::size_t)v] >= mid) { ++cnt; }
            }
            if (cnt >= sp.top_k) { lo = mid; } else { hi = mid; }
          }
          t_k = lo;
        }
        const float t_floor = std::max(t_k, std::max(sp.min_p, 0.0f));
        const float target = sp.top_p * Z;
        float lo = t_floor, hi = 1.0f;
        for (int it = 0; it < n_iter; ++it) {
          const float mid = 0.5f * (lo + hi);
          float mass = 0.0f;
          for (int v = 0; v < V; ++v) {
            const float w = out[(std::size_t)v];
            if (w >= mid) { mass += w; }
          }
          if (mass >= target) { lo = mid; } else { hi = mid; }
        }
        const float wt = std::max(lo, t_floor);
        float kept = 0.0f;
        for (int v = 0; v < V; ++v) {
          if (out[(std::size_t)v] < wt) { out[(std::size_t)v] = 0.0f; }
          else { kept += out[(std::size_t)v]; }
        }
        const float inv = (kept > 0.0f) ? 1.0f / kept : 0.0f;
        for (int v = 0; v < V; ++v) { out[(std::size_t)v] *= inv; }
      };
      // L-C accept of draft `dft` against target `p` using carried draft dist
      // `q`: accept w/ min(1,p(d)/q(d)) -> dft; else a residual sample from
      // norm(max(0,p-q)). Encodes accept/reject into preds for the loop below.
      std::vector<float> res;
      auto lc_accept = [&](std::int32_t dft, const std::vector<float>& p,
                           const std::vector<float>& q) -> std::int32_t {
        const float pd = p[(std::size_t)dft];
        const float qd = q[(std::size_t)dft];
        const float acc = (qd > 0.0f) ? std::min(1.0f, pd / qd) : 1.0f;
        if (next_u() < acc) { return dft; }            // accept
        float s = 0.0f; res.assign((std::size_t)V, 0.0f);
        for (int v = 0; v < V; ++v) {
          float r = p[(std::size_t)v] - q[(std::size_t)v];
          r = (r > 0.0f) ? r : 0.0f;
          res[(std::size_t)v] = r; s += r;
        }
        if (s > 0.0f) {
          const float inv = 1.0f / s;
          for (int v = 0; v < V; ++v) { res[(std::size_t)v] *= inv; }
          return sample(res, next_u());
        }
        return sample(p, next_u());
      };
      // Verifier preds: position i verifies drafts[i+1] (the i-th chained draft,
      // carried dist q1=qcarry / q2=qcarry2); the last position is the bonus.
      std::vector<float> p;
      preds.assign((std::size_t)K, 0);
      for (int i = 0; i < K; ++i) {
        post(vl + (std::size_t)i * V, p);
        if (i + 1 < K) {
          const std::vector<float>& q = (i == 0) ? qcarry : qcarry2;
          preds[(std::size_t)i] = lc_accept(drafts[(std::size_t)(i + 1)], p, q);
        } else {
          preds[(std::size_t)i] = sample(p, next_u());   // bonus / anchor
        }
      }
      // New drafts: SAMPLE from the (post-processed) MTP dists, carry the dists.
      mpreds.assign((std::size_t)K, 0);
      qnew.assign((std::size_t)K, std::vector<float>());
      for (int i = 0; i < K; ++i) {
        post(ml + (std::size_t)i * V, qnew[(std::size_t)i]);
        mpreds[(std::size_t)i] = sample(qnew[(std::size_t)i], next_u());
      }
      if (d2) {
        const auto* ml2 =
            static_cast<const std::uint16_t*>(_lc_mlogits2.contents());
        mpreds2.assign((std::size_t)K, 0);
        qnew2.assign((std::size_t)K, std::vector<float>());
        for (int i = 0; i < K; ++i) {
          post(ml2 + (std::size_t)i * V, qnew2[(std::size_t)i]);
          mpreds2[(std::size_t)i] = sample(qnew2[(std::size_t)i], next_u());
        }
      }
    }
    if (lc_prof && lc_mode) { prof_lc_ms += prof_ms(t_lc0, mtp_clock::now()); }
    // Accept the longest prefix matching the verifier's greedy preds (drafts[0]
    // is always correct -- it IS the prior round's verified next token).
    int j = 1;
    for (int i = 1; i < K; ++i) {
      if (drafts[(std::size_t)i] == preds[(std::size_t)(i - 1)]) { ++j; }
      else { break; }
    }
    // Stop-token cap: the serial loop never appends a stop token's KV nor emits
    // it, so accept only up to (excluding) the first stop within [0, j) and end
    // the decode. j becomes the stop's index, so the rollback below truncates
    // the stop (and any tail) out of the KV/GDN, leaving the context at the
    // last kept token -- ready for the caller's assistant_close commit.
    if (ctl.is_stop) {
      for (int i = 0; i < j; ++i) {
        if (ctl.is_stop(drafts[(std::size_t)i])) {
          j = i;
          terminate = true;
          if (ctl.hit_stop) { *ctl.hit_stop = true; }
          break;
        }
      }
    }
    // Keep at most the remaining budget. `keep` (<= j) drives both out_ids and
    // the KV/GDN rollback boundary, unifying partial-accept, stop, and budget.
    int keep = 0;
    for (int i = 0; i < j && (int)out_ids.size() < n_steps; ++i) {
      out_ids.push_back(drafts[(std::size_t)i]);
      ++keep;
    }
    if (keep > 0) { accepted += (keep - 1); }
    // GDN ring commit: advance the cursor by `keep`, so slot cur now holds the
    // committed S_{p+keep}. The (K-keep) rejected states sit in cur+1.. and are
    // overwritten next round -- a pure cursor move, NO host copy / replay.
    if (use_ring) {
      for (int a = 0; a < keep; ++a) { _ctx->gdn_ring_advance(cid); }
    }
    // Persistent MTP KV commit: the verify appended (d2?2:1)*K draft positions
    // onto the MTP ctx; retain the `keep` committed ones and roll back the rest
    // (rejected depth-1 drafts + the whole depth-2 lookahead window) so next
    // round's drafts attend over exactly the committed decode history.
    if (_mtp_persist) {
      const int mtp_rb = (d2 ? 2 : 1) * K - keep;
      if (mtp_rb > 0) { _mtp.ctx->kv_rollback(_mtp.cid, mtp_rb); }
    }
    const bool budget_done = ((int)out_ids.size() >= n_steps);
    // Next round's drafts come straight out of THIS verify at the boundary --
    // only when we will actually continue (no stop, abort, or budget end).
    if (!terminate && !budget_done && keep > 0) {
      d0 = preds[(std::size_t)(keep - 1)];
      d1 = mpreds[(std::size_t)(keep - 1)];
      have_d1 = true;
      // Carry the drafter dists q1/q2 for the new d1/d2 (next round's ratios).
      // GPU L-C carries the logit ROWS (copied out before next round's verify
      // overwrites _lc_mlogits); the host path carries the post()'d dists.
      if (lc_mode && gpu_lc) {
        const std::size_t vb = (std::size_t)c.vocab * 2;
        const std::size_t off = (std::size_t)(keep - 1) * vb;
        std::memcpy(_lc_qcarry.contents(),
                    static_cast<const std::uint8_t*>(_lc_mlogits.contents())
                        + off, vb);
        if (d2) {
          std::memcpy(_lc_qcarry2.contents(),
                      static_cast<const std::uint8_t*>(_lc_mlogits2.contents())
                          + off, vb);
        }
      } else if (lc_mode) {
        qcarry = std::move(qnew[(std::size_t)(keep - 1)]);
        if (d2) { qcarry2 = std::move(qnew2[(std::size_t)(keep - 1)]); }
      }
      if (d2) { d2v = mpreds2[(std::size_t)(keep - 1)]; have_d2 = true; }
    }
    // Over-append rollback: the main KV + GDN advanced by K but we keep only
    // `keep`. Drop ONLY the rejected tokens' KV (kv_rollback K-keep) -- the kept
    // tokens' KV is valid (each was written with the same causal prefix a
    // keep-token forward sees). The GDN recurrent state has no positional
    // rewind, so restore the round's S0 snapshot and replay conv1d->qk_norm->
    // gdn_step for the kept tokens from the verify's cached inputs: bit-
    // identical to S_{p+keep}, GDN-only (no attention / projections / lm_head),
    // so far cheaper than the full main re-forward this used to do.
    if (keep < K) {
      auto t_rb0 = mtp_clock::now();
      _ctx->kv_rollback(cid, K - keep);
      if (!use_ring) {
        // In-place GDN: restore the round's S0 then replay the kept tokens to
        // S_{p+keep}. (The ring path already committed S_{p+keep} via the cursor
        // advance above -- the rejected states are abandoned, zero copy.)
        snapshot_gdn(/*save=*/false);
        auto t_rr0 = mtp_clock::now();
        prof_snap_ms += prof_ms(t_rb0, t_rr0);
        if (keep > 0) {
          gdn_replay_(cid, keep, gcache);
          prof_rerun_ms += prof_ms(t_rr0, mtp_clock::now());
          ++prof_reruns;
        }
      }
    }
    // Stream this round's kept tokens; an on_round false return aborts.
    if (ctl.on_round && keep > 0) {
      std::span<const std::int32_t> fresh(
          out_ids.data() + (out_ids.size() - (std::size_t)keep),
          (std::size_t)keep);
      if (!ctl.on_round(fresh)) { terminate = true; }
    }
  }
  // Finalize the GDN ring: O(1) handle-swap the cursor's (final committed) state
  // into the canonical slot so the caller's assistant_close / next sync decode
  // read the correct recurrent state. No-op when the ring was off.
  if (use_ring) { _ctx->gdn_ring_end(cid); }
  if (accepted_out) { *accepted_out = accepted; }
  if (rounds_out) { *rounds_out = rounds; }
  if (mtp_prof) {
    const double tot = prof_verify_ms + prof_rerun_ms + prof_snap_ms;
    std::printf(
        "[mtp_prof] depth=%d tok=%d rounds=%ld reruns=%ld (%.0f%% of rounds) "
        "| verify=%.1fms (%.2f/round) rerun=%.1fms (%.2f/rerun) snap=%.1fms "
        "| sum=%.1fms\n",
        (draft_len >= 2 ? 2 : 1), (int)out_ids.size(), rounds, prof_reruns,
        rounds > 0 ? 100.0 * (double)prof_reruns / (double)rounds : 0.0,
        prof_verify_ms, rounds > 0 ? prof_verify_ms / (double)rounds : 0.0,
        prof_rerun_ms,
        prof_reruns > 0 ? prof_rerun_ms / (double)prof_reruns : 0.0,
        prof_snap_ms, tot);
  }
  if (lc_prof) {
    std::printf(
        "[mtp_lc] %s | tok=%d rounds=%ld | lc-step=%.1fms (%.3f/round)\n",
        gpu_lc ? "GPU" : "host", (int)out_ids.size(), rounds, prof_lc_ms,
        rounds > 0 ? prof_lc_ms / (double)rounds : 0.0);
  }
  if (vprof_d && g_vp_n > 0) {
    const double v = (double)g_vp_n;
    const double sum = g_vp_main + g_vp_vhead + g_vp_mtp;
    std::printf(
        "[mtp_vprof] depth=%d verifies=%ld | main=%.2f/v vhead=%.2f/v "
        "mtp=%.2f/v (%.0fx head) | sum/v=%.2fms (main %.0f%% vhead %.0f%% "
        "mtp %.0f%%)\n",
        (draft_len >= 2 ? 2 : 1), g_vp_n, g_vp_main / v, g_vp_vhead / v,
        g_vp_mtp / v, g_vp_vhead > 0 ? g_vp_mtp / g_vp_vhead : 0.0, sum / v,
        100.0 * g_vp_main / sum, 100.0 * g_vp_vhead / sum,
        100.0 * g_vp_mtp / sum);
  }
  return true;
}

bool
MetalQwenModel::mtp_teacher_force(ContextId cid,
                                  const std::vector<std::int32_t>& cont,
                                  int chunk, long* d1_hits, long* d1_total)
{
  if (!_mtp.ok || !(_mixed || _kquant)) { return false; }
  if (!ensure_decode_scratch_()) { return false; }
  const Config& c = _cfg;
  if (chunk < 1) { chunk = 1; }
  const int N = (int)cont.size();
  // Honor the persistent-MTP-KV flag so this hook measures the SAME head the
  // decode uses: persistent accumulates the walked history (no per-chunk wipe),
  // non-persistent (VPIPE_MTP_NO_PERSIST) wipes each chunk. Reset once here so
  // the walk starts from a clean MTP ctx either way.
  _mtp_persist = (std::getenv("VPIPE_MTP_NO_PERSIST") == nullptr);
  mtp_ctx_reset_();
  // Seed the MTP KV with the prefill's prompt (the "full history" mode -- this
  // is what the production mtp_decode now does), so the measured draft accuracy
  // matches the decode. cont[0] is the decode's first token (the last seeded
  // position's conditioning). VPIPE_MTP_NO_SEED disables the capture upstream,
  // leaving the decode-only ("no prefix") baseline for an A/B.
  if (_mtp_persist && !cont.empty()) { mtp_seed_prefix_(cont[0]); }

  // GDN snapshot cache for the use_ring=false verify path. Teacher-forcing keeps
  // every token (keep==W each chunk), so the GDN advances in place and the cache
  // is filled but never replayed.
  std::vector<int> glayers;
  for (int L = 0; L < c.n_layers; ++L) {
    if (c.layer_is_full(L)) { continue; }
    if (_ctx->conv_state(cid, L) != nullptr &&
        _ctx->ssm_state(cid, L) != nullptr) {
      glayers.push_back(L);
    }
  }
  GdnVerifyCache gcache;
  gcache.layers = glayers;
  {
    const int Cd = c.gdn_conv_dim, Hv = c.gdn_v_heads;
    for (std::size_t i = 0; i < glayers.size(); ++i) {
      gcache.qkv.push_back(
          _mc->make_shared_buffer((std::size_t)chunk * Cd * 2));
      gcache.gbuf.push_back(
          _mc->make_shared_buffer((std::size_t)chunk * Hv * 4));
      gcache.betabuf.push_back(
          _mc->make_shared_buffer((std::size_t)chunk * Hv * 4));
    }
  }
  const int seed_slot0 = _ctx->seq_len_of(cid);
  const GpuSamplerParams sp;   // greedy
  SharedBuffer condbuf =
      _mc->make_shared_buffer((std::size_t)chunk * sizeof(std::int32_t));

  long hits = 0, total = 0, aligned = 0, atot = 0;
  int j = 0;
  while (j + 1 < N) {
    const int W = std::min(chunk, N - 1 - j);   // need cont[j+W] to condition
    if (W < 1) { break; }
    SharedBuffer xK = _muxer->fetch_text(
        std::span<const std::int32_t>(cont.data() + j, (std::size_t)W));
    if (xK.empty()) { break; }
    // Position k conditions the MTP head on the TRUE next token cont[j+k+1].
    auto* cb = static_cast<std::int32_t*>(condbuf.contents());
    for (int k = 0; k < W; ++k) { cb[k] = cont[(std::size_t)(j + k + 1)]; }
    std::vector<std::int32_t> preds, mpreds;
    const bool ok =
        mtp_verify_chunk_(cid, xK, W, &preds, &mpreds, nullptr, 0, sp,
                          seed_slot0, &gcache, false, false, &condbuf);
    if (!ok || (int)mpreds.size() != W) { return false; }
    for (int k = 0; k < W; ++k) {
      if ((int)preds.size() == W) {
        ++atot;
        if (preds[(std::size_t)k] == cont[(std::size_t)(j + k + 1)]) {
          ++aligned;
        }
      }
      const int tgt = j + k + 2;   // MTP draft predicts cont[j+k+2]
      if (tgt < N) {
        ++total;
        if (mpreds[(std::size_t)k] == cont[(std::size_t)tgt]) { ++hits; }
      }
    }
    j += W;
  }
  if (d1_hits) { *d1_hits = hits; }
  if (d1_total) { *d1_total = total; }
  std::printf("[mtp_tf] depth-1 draft acc %ld/%ld = %.3f | greedy-align "
              "%ld/%ld = %.3f (chunk=%d)\n",
              hits, total, total ? (double)hits / (double)total : 0.0,
              aligned, atot, atot ? (double)aligned / (double)atot : 0.0,
              chunk);
  return true;
}

}  // namespace vpipe::genai
