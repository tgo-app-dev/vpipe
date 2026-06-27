#include "generative-models/gemma4/metal-gemma-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/shared/kernel-autotune.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
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
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  b += 0x7fffu + ((b >> 16) & 1u);
  return (std::uint16_t)(b >> 16);
}
inline std::uint16_t
f32_to_elt_(float f, bool bf16)
{
  if (bf16) { return f32_to_bf16_(f); }
  _Float16 h = (_Float16)f;
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
inline float
elt_to_f32_(std::uint16_t e, bool bf16)
{
  if (bf16) { return bf16_to_f32_(e); }
  _Float16 h;
  std::memcpy(&h, &e, 2);
  return (float)h;
}
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

std::unique_ptr<MetalGemmaModel>
MetalGemmaModel::load(const std::string& model_dir,
                      metal_compute::MetalCompute* mc, const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) { return nullptr; }
  if ((int)cfg.is_full_layer.size() != cfg.n_layers) { return nullptr; }

  auto m = std::unique_ptr<MetalGemmaModel>(new MetalGemmaModel());
  m->_cfg = cfg;
  m->_mc = mc;
  const bool bf16 = cfg.use_bf16;

  // Mixed quant: derive the MLP bit width from the gate_proj packing
  // (out=ffn, in_packed = hidden*bits/32 u32 columns). gemma4_unified is
  // 8-bit MLP / 4-bit elsewhere; e4b is uniform. Falls back to quant_bits.
  int mlp_bits = cfg.quant_bits;
  {
    const auto* gi = wts->info(cfg.weight_prefix + "layers.0.mlp.gate_proj.weight");
    if (gi != nullptr && cfg.hidden > 0 && gi->shape.size() == 2) {
      const long wcols = gi->shape[1];
      const int derived = static_cast<int>((32 * wcols) / cfg.hidden);
      if (derived == 4 || derived == 8) { mlp_bits = derived; }
    }
  }
  m->_cfg.mlp_bits = mlp_bits;

  // Embedding / tied lm_head bit width, derived from the embed weight
  // packing (gemma4_unified mlx checkpoints are 4-bit; the GGUF path keeps
  // it 8-bit over a 4-bit body). bits = 32 * w_cols / hidden.
  int embed_bits = cfg.quant_bits;
  {
    const auto* ei = wts->info(cfg.weight_prefix + "embed_tokens.weight");
    if (ei != nullptr && cfg.hidden > 0 && ei->shape.size() == 2) {
      const long wcols = ei->shape[1];
      const int derived = static_cast<int>((32 * wcols) / cfg.hidden);
      if (derived == 4 || derived == 8) { embed_bits = derived; }
    }
  }
  m->_cfg.embed_bits = embed_bits;

  // Affine group size (64 default, 32 for GGUF q4_0). Selects the kernel
  // variant suffix w<bits>g<group>.
  const int qg = cfg.quant_group > 0 ? cfg.quant_group : 64;
  m->_cfg.quant_group = qg;
  const std::string gtok = "g" + std::to_string(qg);

  const std::string sfx = bf16 ? "_bf16" : "";
  m->_lib_qmv = mc->load_library("affine_qmv" + sfx);
  m->_lib_qmm = mc->load_library("affine_qmm_steel" + sfx);
  m->_lib_rms = mc->load_library("rms_norm" + sfx);
  m->_lib_elt = mc->load_library("llm_elementwise" + sfx);
  m->_lib_rope = mc->load_library("rope" + sfx);
  m->_lib_sdpa = mc->load_library("sdpa" + sfx);
  m->_lib_dense = mc->load_library("dense_gemm" + sfx);
  // Base bits drive attn / lm_head / embed; mlp bits drive gate/up (geglu)
  // and down. They coincide for e4b (uniform 4-bit).
  const std::string g =
      (cfg.quant_bits == 8 ? "w8" : "w4") + gtok;
  const std::string gmlp = (mlp_bits == 8 ? "w8" : "w4") + gtok;
  const std::string gemb = (embed_bits == 8 ? "w8" : "w4") + gtok;
  m->_fn_qmv = m->_lib_qmv.function("affine_qmv_" + g);
  m->_fn_qmv_add = m->_lib_qmv.function("affine_qmv_" + g + "_add");
  // Tied lm_head GEMV runs at embed_bits (decode + prefill last token).
  m->_fn_qmv_embed = m->_lib_qmv.function("affine_qmv_" + gemb);
  m->_fn_qmv_geglu = m->_lib_qmv.function("affine_qmv_geglu_" + gmlp);
  m->_fn_qmv_mlp = m->_lib_qmv.function("affine_qmv_" + gmlp);
  // Fused per-layer-input gate GEMV (e4b PLE; base quant). Folds the geglu.
  m->_fn_qmv_gelu_mul = m->_lib_qmv.function("affine_qmv_gelu_mul_" + g);
  // QKV-fused decode GEMV (one full-occupancy matvec over the q|k|v row-concat;
  // the 512-row k/v matvecs alone under-occupy the GPU). ~0.9% decode GPU-time
  // win (gpu_active 15.93 -> 15.79 ms/tok on M4 Pro, 4-sample medians),
  // token-exact. Memory-neutral: prefill's qw/kw/vw are re-pointed as subviews
  // of the concat (build_qkv) so there is no weight duplication. On by default
  // for uniform-4-bit checkpoints (e4b + realtime-vqa); VPIPE_GEMMA_NO_QKV_FUSE
  // disables. 8-bit/mixed (gemma4_unified, OptiQ) fall back per layer.
  m->_qkv_fuse = std::getenv("VPIPE_GEMMA_NO_QKV_FUSE") == nullptr
              && cfg.quant_bits != 8;
  if (m->_qkv_fuse) {
    m->_fn_qmv_qkv = m->_lib_qmv.function("affine_qmv_qkv_" + g);
    if (!m->_fn_qmv_qkv.valid()) { m->_qkv_fuse = false; }
    // input_layernorm fused into the QKV GEMV (decode). OPT-IN only: it is
    // token-exact but ~0.4 ms/tok SLOWER -- the QKV matvec fans out into ~384
    // threadgroups that each redundantly redo the O(H) RMS reduction (+ read x
    // twice), and the staging threadgroup memory cuts the bandwidth-bound
    // GEMV's occupancy; both outweigh the one saved rms dispatch. The standalone
    // rms (one threadgroup, reduces once) wins. Parked behind the flag.
    m->_qkv_rms_fuse = m->_qkv_fuse
        && std::getenv("VPIPE_GEMMA_QKV_RMS_FUSE") != nullptr;
    if (m->_qkv_rms_fuse) {
      m->_fn_qmv_qkv_rms = m->_lib_qmv.function("affine_qmv_qkv_rms_" + g);
      if (!m->_fn_qmv_qkv_rms.valid()) { m->_qkv_rms_fuse = false; }
    }
  }
  m->_fn_qmm = m->_lib_qmm.function("affine_qmm_steel_" + g);
  m->_fn_qmm_geglu = m->_lib_qmm.function("affine_qmm_geglu_" + gmlp);
  m->_fn_qmm_mlp = m->_lib_qmm.function("affine_qmm_steel_" + gmlp);
  // Batched-decode GEMV (qmv bandwidth, weights read once across the N
  // branches). Enabled only for uniformly-4-bit checkpoints (e4b and the
  // other realtime-vqa models): that's the verified, deterministic path
  // (e4b token-exact across runs incl. grid.z tiling). The 10GB 12B
  // (gemma4_unified, 8-bit MLP) stays on the steel GEMM -- its batched
  // GEMV corrupts non-deterministically under memory pressure on a 16GB
  // box (correct kernel, but the model doesn't stay resident during the
  // longer batched execution); steel's single coalesced read tolerates it.
  if (cfg.quant_bits != 8 && mlp_bits != 8) {
    m->_fn_qmv_batch = m->_lib_qmv.function("affine_qmv_batch_" + g);
    m->_fn_qmv_batch_geglu =
        m->_lib_qmv.function("affine_qmv_batch_geglu_" + gmlp);
    m->_fn_qmv_batch_mlp = m->_lib_qmv.function("affine_qmv_batch_" + g);
  }
  // GGUF Q4_0 symmetric: the per-group affine bias is exactly -8*scale, so the
  // 4-bit g32 decode GEMVs skip the bias buffer (scale-only) -- bit-identical,
  // ~10% fewer weight-read bytes (the dominant FFN/proj decode cost; matches
  // llama.cpp's native scale-only Q4_0 byte count). Swap in the _q40 variants;
  // the 8-bit / q6_k (embed/lm_head) and asymmetric MLX-affine paths keep
  // reading the bias. Decode-only kernels; prefill steel GEMM is unchanged.
  if (m->_cfg.quant_symmetric && qg == 32) {
    auto swap = [&](metal_compute::ComputeFunction& dst, const char* name) {
      auto f = m->_lib_qmv.function(name);
      if (f.valid()) { dst = std::move(f); }
    };
    if (cfg.quant_bits != 8) {                       // attn proj (w4g32)
      swap(m->_fn_qmv, "affine_qmv_w4g32_q40");
      swap(m->_fn_qmv_add, "affine_qmv_w4g32_add_q40");
    }
    if (mlp_bits != 8) {                             // FFN gate/up + down
      swap(m->_fn_qmv_geglu, "affine_qmv_geglu_w4g32_q40");
      swap(m->_fn_qmv_mlp, "affine_qmv_w4g32_q40");
    }
  }
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_rms_add = m->_lib_rms.function("rms_add_f16");
  // Fast simd_sum full-hidden norms: swap the order-invariant RMS_PAD tree (12
  // barriers, reduces 4096) for MLX's rms_single_row simd_sum reduction (2
  // barriers, reduces only H). ~33% off the norm category, +2% decode; matches
  // the omlx/MLX reduction structure. DEFAULT ON (the tree's cross-config bit-
  // exactness was already moot -- vpipe diverges from omlx at exact ties anyway,
  // and all gemma token-exact regression tests pass with this on; the only
  // change vs the tree is benign near-tie greedy reorders, §27-30/§53).
  // VPIPE_GEMMA_RMS_FAST=0 reverts to the deterministic tree for A/B.
  m->_fn_rms_fast = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_rms_add_fast = m->_lib_rms.function("rms_add_fast_f16");
  m->_rms_fast = m->_fn_rms_fast.valid() && m->_fn_rms_add_fast.valid();
  if (const char* e = std::getenv("VPIPE_GEMMA_RMS_FAST")) {
    m->_rms_fast = m->_rms_fast && (std::atoi(e) != 0);
  }
  m->_fn_rope = m->_lib_rope.function("rope_f16");
  m->_fn_rms_rope = m->_lib_rope.function("rms_rope_f16");
  m->_fn_rms_rope2 = m->_lib_rope.function("rms_rope2_f16");
  m->_fn_rms_rope3 = m->_lib_rope.function("rms_rope3_f16");
  // Fused rope3 + ring KV-write (sliding decode). Default ON;
  // VPIPE_GEMMA_NO_ROPE_KV_FUSE opts out for A/B. Falls back if unavailable.
  m->_fn_rms_rope3_kvwrite =
      m->_lib_rope.function("rms_rope3_kvwrite_f16");
  m->_rope_kv_fuse = m->_fn_rms_rope3_kvwrite.valid()
      && std::getenv("VPIPE_GEMMA_NO_ROPE_KV_FUSE") == nullptr;
  m->_fn_geglu = m->_lib_elt.function("geglu_f16");
  m->_fn_softcap = m->_lib_elt.function("softcap_f16");
  m->_fn_suppress = m->_lib_elt.function("suppress_logits_f16");
  m->_fn_scale = m->_lib_elt.function("scale_inplace_f16");
  m->_fn_dummy_disp = m->_lib_elt.function("dummy_disp_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_copy = m->_lib_elt.function("copy_f16");
  // Embedding gather at embed_bits. Only the w8 g32 variant is new (the
  // GGUF path); the 4-bit / g64 names are the existing mlx-checkpoint ones.
  std::string embed_fn;
  if (embed_bits == 8) {
    embed_fn = (qg == 32) ? "dequant_embed_gather_w8_g32_f16"
                          : "dequant_embed_gather_w8_f16";
  } else {
    embed_fn = "dequant_embed_gather_f16";
  }
  m->_fn_embed = m->_lib_elt.function(embed_fn);
  // Native Q6_K tied embed/lm_head (GGUF path). Present iff the converter
  // emitted a raw `embed_tokens.q6k` table (lossless, ~25% smaller than the
  // affine8 requant). When present, embed-gather + lm_head GEMV read it
  // directly and the affine8 _embed_w/s/b are never loaded.
  m->_embed_is_q6k =
      wts->info(cfg.weight_prefix + "embed_tokens.q6k") != nullptr;
  // A/B + safety fallback: force the affine8 embed/lm_head even when the raw
  // Q6_K table is present (loads the requant, ~25% more memory).
  if (const char* e = std::getenv("VPIPE_GEMMA_NO_Q6K")) {
    if (std::atoi(e) != 0) { m->_embed_is_q6k = false; }
  }
  if (m->_embed_is_q6k) {
    m->_fn_embed_q6k = m->_lib_elt.function("embed_gather_q6k_f16");
    // The tied lm_head GEMV (262144 x 3840 = ~825 MB raw Q6_K) is BANDWIDTH-
    // BOUND at the M5 ceiling: both qmv_q6k_v2_f16 (llama.cpp-style: each lane
    // reads every ql/qh byte once, extracts all nibbles/fields in-thread) and
    // the per-nibble qmv_q6k_f16 measure ~130 GB/s == saturation (microbench
    // metal_compute_qmv.q6k_lmhead_bandwidth, VPIPE_QMV_Q6K). v2's win over v1
    // was an instruction-bound regime that no longer applies once saturated;
    // it stays the default (no downside). No M5 kernel headroom here -- M=1
    // GEMV (no matrix-core reuse), already at peak BW; only fewer BYTES (lower
    // precision) would help, which breaks the lossless/token-exact bar.
    m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_v2_f16");
    if (!m->_fn_qmv_q6k.valid()) {
      m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_f16");
    }
  }
  m->_fn_kv_write = m->_lib_elt.function("kv_write_f16");
  m->_fn_kv_write2 = m->_lib_elt.function("kv_write2_f16");
  m->_fn_kv_write_sub = m->_lib_elt.function("kv_write_sub_f16");
  // PAGED K/V write + scalar paged attention for the FULL (global) layers when
  // full_layers_paged (branch shares prefix pages, no full-KV copy). The scalar
  // paged-causal kernel is register-based (D up to 512) and runs on every GPU.
  m->_fn_kv_write_paged = m->_lib_elt.function("kv_write_paged_f16");
  m->_fn_sdpa_paged_causal = m->_lib_sdpa.function("sdpa_paged_causal_f16");
  // M4 / non-matrix-core PAGED prefill for the GLOBAL (head_dim 512) layers:
  // the page-walked sibling of sdpa_causal_flash_f16 (simdgroup_matrix flash --
  // runs on every Apple GPU). Fills the tier between the M5-only matmul2d
  // sdpa_paged_mma2_d512 and the scalar sdpa_paged_causal; without it the M4
  // bulk global paged prefill fell to the O(ctx)/32-lane scalar kernel and
  // regressed ~30-40% vs the old contiguous flash. VPIPE_GEMMA_NO_PFLASH=1
  // reverts to scalar (A/B + safety).
  m->_fn_sdpa_pflash = m->_lib_sdpa.function("sdpa_paged_flash_d512_f16");
  m->_pflash_attn = m->_fn_sdpa_pflash.valid()
      && std::getenv("VPIPE_GEMMA_NO_PFLASH") == nullptr;
  // MATERIALIZED GLOBAL (head_dim 512) prefill: per-head Q.K^T (steel dense) ->
  // causal softmax (this kernel) -> P.V (steel dense), mirroring MLX's
  // head_dim-512 fallback (no fused flash). Replaces the ~1 TFLOP/s fused
  // sdpa_paged_flash_d512 for single-chunk global prefill: +17.7% pp@8k,
  // +24.1% pp@16k, token-exact, decode unchanged (report §34). DEFAULT ON when
  // the kernel validated; VPIPE_GEMMA_MATERIALIZED_GLOBAL=0 forces it off (A/B).
  m->_fn_causal_softmax = m->_lib_sdpa.function("causal_softmax_rows_f16");
  m->_materialized_global = m->_fn_causal_softmax.valid();
  if (const char* e = std::getenv("VPIPE_GEMMA_MATERIALIZED_GLOBAL")) {
    m->_materialized_global = std::atoi(e) != 0 && m->_fn_causal_softmax.valid();
  }
  // KV-split GQA paged flash-decode for the full layers + its merge. The vec
  // kernel (one q-head/simdgroup, UK-unrolled vec4 loads, latency-optimal) is
  // already D-flexible (per4 = D/128; D=512 ok) -- the paged sibling of the
  // contiguous gqa_vec the full-layer decode used before paging.
  m->_fn_sdpa_pgqa = m->_lib_sdpa.function("sdpa_paged_gqa_vec_f16");
  // Materialized paged decode (global D=512): QK GEMV -> rowstat -> PV GEMV
  // -> merge (the omlx/MLX head_dim-512 fallback structure; no online softmax).
  m->_fn_dec_qk      = m->_lib_sdpa.function("sdpa_paged_dec_qk_f16");
  m->_fn_dec_rowstat = m->_lib_sdpa.function("sdpa_dec_rowstat_f16");
  m->_fn_dec_pv      = m->_lib_sdpa.function("sdpa_paged_dec_pv_f16");
  m->_fn_dec_merge   = m->_lib_sdpa.function("sdpa_dec_pv_merge_f16");
  m->_fn_argmax = m->_lib_elt.function("argmax_f16");
  // Two-stage parallel argmax (optional; single-tg _fn_argmax is the fallback).
  m->_fn_argmax_partial = m->_lib_elt.function("argmax_partial_f16");
  m->_fn_argmax_combine = m->_lib_elt.function("argmax_combine_f16");
  // GPU sampler for the pipelined-decode non-greedy path (same kernel as
  // Qwen; dtype-correct via _lib_elt's suffix). Optional: greedy pdecode
  // only needs _fn_argmax.
  m->_fn_sample = m->_lib_elt.function("sample_topp_f16");
  // Histogram multi-tg sampler (default; _fn_sample is the SAMPLE1 fallback).
  m->_fn_smp_max_partial   = m->_lib_elt.function("sample_max_partial_f16");
  m->_fn_smp_max_combine   = m->_lib_elt.function("sample_max_combine_f16");
  m->_fn_smp_zhist_partial = m->_lib_elt.function("sample_zhist_partial_f16");
  m->_fn_smp_zhist_combine = m->_lib_elt.function("sample_zhist_combine_f16");
  m->_fn_smp_thresh        = m->_lib_elt.function("sample_thresh_f16");
  m->_fn_smp_pick_partial  = m->_lib_elt.function("sample_pick_partial_f16");
  m->_fn_smp_pick_combine  = m->_lib_elt.function("sample_pick_combine_f16");
  m->_fn_row_scatter = m->_lib_elt.function("row_scatter_f16");
  m->_fn_sdpa_causal = m->_lib_sdpa.function("sdpa_causal_f16");
  m->_fn_sdpa_window = m->_lib_sdpa.function("sdpa_causal_window_f16");
  m->_fn_sdpa_mb = m->_lib_sdpa.function("sdpa_causal_mb_f16");
  // MMA flash-attention prefill kernel. Present in BOTH the f16 and bf16
  // metallibs (the QK^T/PV operands are VPIPE_ELT; accumulation stays fp32).
  m->_fn_sdpa_mma = m->_lib_sdpa.function("sdpa_causal_mma_f16");
  // Device-direct contiguous prefill kernel (global/full layers).
  m->_fn_sdpa_mma_dev = m->_lib_sdpa.function("sdpa_causal_mma_dev_f16");
  // llama.cpp-style flash kernel (default global-layer prefill SDPA).
  m->_fn_sdpa_flash = m->_lib_sdpa.function("sdpa_causal_flash_f16");
  // Flash-decode-GQA serial attention (contiguous KV). Optional: decode
  // falls back to sdpa_mb when either kernel is absent or the shape is
  // unsupported (head_dim > 256 or G > 4).
  m->_fn_sdpa_gqa = m->_lib_sdpa.function("sdpa_causal_gqa_f16");
  m->_fn_sdpa_gqa_merge = m->_lib_sdpa.function("sdpa_gqa_merge_f16");
  m->_fn_sdpa_gqa_tile = m->_lib_sdpa.function("sdpa_causal_gqa_tile_f16");
  m->_fn_sdpa_gqa_direct = m->_lib_sdpa.function("sdpa_causal_gqa_direct_f16");
  m->_fn_sdpa_gqa_vec = m->_lib_sdpa.function("sdpa_causal_gqa_vec_f16");
  m->_fn_sdpa_gqa_vec_lin =
      m->_lib_sdpa.function("sdpa_causal_gqa_vec_lin_f16");
  m->_fn_sdpa_mma_qhead = m->_lib_sdpa.function("sdpa_causal_mma_qhead_f16");
  // Matrix-core (M5+) matmul2d attention for GLOBAL (head_dim 512) prefill: the
  // sdpa_mma metallib needs -std=metal4.0, so load it (and enable the path)
  // only on a matrix-core GPU. ~1.7-1.9x the simdgroup_matrix flash kernel.
  if (mc->supports_matrix_cores()) {
    m->_lib_sdpa_mma = mc->load_library("sdpa_mma" + sfx);
    m->_fn_sdpa_mma2 = m->_lib_sdpa_mma.function("sdpa_causal_mma2_d512_f16");
    m->_fn_sdpa_mma2_d256 =
        m->_lib_sdpa_mma.function("sdpa_causal_mma2_d256_f16");
    m->_mma2_attn = m->_fn_sdpa_mma2.valid()
        && std::getenv("VPIPE_GEMMA_NO_MMA2_ATTN") == nullptr;
    // Matrix-core PAGED prefill attention for the full (D=512) layers (the
    // paged sibling of sdpa_causal_mma2_d512 -- same fast body, page-walked).
    m->_fn_sdpa_pmma2 = m->_lib_sdpa_mma.function("sdpa_paged_mma2_d512_f16");
    m->_pmma2_attn = m->_fn_sdpa_pmma2.valid()
        && std::getenv("VPIPE_GEMMA_NO_MMA2_ATTN") == nullptr;
  }
  // Lazy single-pass prefill: grow a fresh context's sliding ring to fit a
  // one-shot prompt that would otherwise wrap (-> the slow staged sliding
  // attention). Default ON; VPIPE_GEMMA_NO_SLIDING_GROW=1 keeps the bounded
  // ring (forces the wrap/chunk path -- exercised by the ringwrap test, and
  // an e2e A/B knob).
  m->_sliding_grow = std::getenv("VPIPE_GEMMA_NO_SLIDING_GROW") == nullptr;
  // Bounded-ring single-pass prefill: when the ring is bounded (no grow) and a
  // one-shot prompt would wrap, run ONE forward over the whole prompt instead
  // of chunking the entire stack to <= page tokens. The sliding ATTENTION reads
  // the full-batch K/V scratch (materialized banded path, ring-independent), so
  // only the sliding KV-WRITE touches the bounded ring -- and a single full
  // write leaves the correct trailing window resident (the last `ring` logical
  // positions are never clobbered). This decouples the proj/FFN/global GEMM
  // batch from the page-sized ring sub-block, recovering the grown one-pass
  // prefill speed while the ring stays bounded. Default ON; opt out with
  // VPIPE_GEMMA_PREFILL_SUBBLOCK=0. Only fires when the materialized sliding
  // path is available (it is the ring-independent attention primitive).
  m->_prefill_subblock = [] {
    const char* e = std::getenv("VPIPE_GEMMA_PREFILL_SUBBLOCK");
    return !(e && std::atoi(e) == 0);
  }();
  m->_mat_sliding = [] {
    const char* e = std::getenv("VPIPE_GEMMA_MAT_SLIDING");
    return !(e && std::atoi(e) == 0);
  }();
  m->_fn_dense_t = m->_lib_dense.function("dense_gemm_t_f16");
  m->_fn_dense_t_qkcausal =
      m->_lib_dense.function("dense_gemm_t_qkcausal_f16");
  m->_fn_dense_t_pvcausal =
      m->_lib_dense.function("dense_gemm_t_pvcausal_f16");
  // M=1 decode GEMV for the e4b PLE f16 projections (default ON when the
  // entry point is present; VPIPE_GEMMA_NO_PLE_GEMV=1 reverts to dense_t).
  m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
  m->_ple_gemv = m->_fn_dense_gemv.valid() &&
                 std::getenv("VPIPE_GEMMA_NO_PLE_GEMV") == nullptr;
  // 4-bit per_layer_projection (plp) by default: the checkpoint ships plp
  // 4-bit, but vpipe dequantized it to f16 -> read 4x the bytes (55MB vs 14MB
  // /tok across 42 layers). Native qmv/qmm matches omlx. (plm is BF16 in the
  // checkpoint -> stays f16, no opportunity.) VPIPE_GEMMA_PLE_F16=1 reverts.
  m->_ple_quant = std::getenv("VPIPE_GEMMA_PLE_F16") == nullptr;
  if (!m->_fn_qmv.valid() || !m->_fn_qmv_add.valid() || !m->_fn_rms.valid() ||
      !m->_fn_qmv_embed.valid() ||
      !m->_fn_qmv_geglu.valid() || !m->_fn_qmm_geglu.valid() ||
      !m->_fn_qmv_mlp.valid() || !m->_fn_qmm_mlp.valid() ||
      !m->_fn_rms_add.valid() ||
      !m->_fn_geglu.valid() || !m->_fn_softcap.valid() ||
      !m->_fn_scale.valid() || !m->_fn_residual.valid() ||
      !m->_fn_embed.valid() || !m->_fn_kv_write.valid() ||
      !m->_fn_sdpa_causal.valid() || !m->_fn_sdpa_window.valid() ||
      !m->_fn_sdpa_mb.valid() || !m->_fn_sdpa_mma.valid() ||
      !m->_fn_kv_write_paged.valid() || !m->_fn_sdpa_paged_causal.valid() ||
      !m->_fn_dense_t.valid() || !m->_fn_rope.valid() ||
      !m->_fn_rms_rope.valid() ||
      !m->_fn_argmax.valid() || !m->_fn_qmm.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_row_scatter.valid()) {
    return nullptr;
  }
  // Flash-decode-GQA serial attention (contiguous KV) -- OFF by default for
  // Gemma. Unlike Qwen/Llama (full attention, where GQA cuts the dominant KV
  // bandwidth), Gemma-4 is sliding-window-dominated: per-q-head sdpa_mb
  // already scans only `window` (512) keys, so GQA's merge overhead makes it
  // net-neutral-to-negative even restricted to full layers (measured ~0-4%
  // slower on e4b). Kept token-exact + opt-in (VPIPE_GQA_ATTN=1) for future
  // tuning / full-attention-heavy variants. VPIPE_GQA_SPLIT sets the split.
  const bool gqa_ok =
      m->_fn_sdpa_gqa.valid() && m->_fn_sdpa_gqa_merge.valid();
  m->_gqa_attn = false;
  if (const char* e = std::getenv("VPIPE_GQA_ATTN")) {
    m->_gqa_attn = gqa_ok && (std::atoi(e) != 0);
  }
  m->_catprof = std::getenv("VPIPE_GEMMA_CATPROF") != nullptr;
  if (const char* e = std::getenv("VPIPE_GQA_SPLIT")) {
    const int v = std::atoi(e);
    if (v >= 1 && v <= 64) { m->_gqa_split = v; }
  }
  // Direct-read flash-decode for the GLOBAL layers -- DEFAULT ON. The 7 global
  // D=512 full-context layers are the whole metal-vs-omlx decode gap. The
  // direct kernel reads K/V straight from device memory (no threadgroup
  // staging/barriers, fast::exp) -- a microbench (sdpa_mma.gqa_decode_micro)
  // measured it ~20-24% faster than the older staged tile kernel across 1k-4k
  // (staging's read-once loses to direct+L2+ILP, MLX's sdpa_vector approach).
  // Global-layer decode kernel selection at the dispatch sites. DEFAULT is the
  // vec kernel (UK-unrolled + vec4 loads, ~2x the direct kernel, bit-identical;
  // the global decode attn is latency-bound). VPIPE_GEMMA_ATTN_STAGED=1 reverts
  // to the staged tile, VPIPE_GEMMA_ATTN_DIRECT=1 to the scalar direct kernel
  // (both for A/B). Sliding layers keep sdpa_mb (cheap, window-capped).
  // VPIPE_GEMMA_GTILE_ATTN=0 disables (-> sdpa_mb); VPIPE_GEMMA_GTILE_SPLIT
  // sets the split (default 64).
  m->_gtile_staged = std::getenv("VPIPE_GEMMA_ATTN_STAGED") != nullptr
      && m->_fn_sdpa_gqa_tile.valid();
  m->_gtile_direct = std::getenv("VPIPE_GEMMA_ATTN_DIRECT") != nullptr;
  const bool decode_ok = m->_gtile_staged
      ? m->_fn_sdpa_gqa_tile.valid()
      : (m->_fn_sdpa_gqa_vec.valid() || m->_fn_sdpa_gqa_direct.valid());
  m->_gtile_attn = decode_ok && m->_fn_sdpa_gqa_merge.valid();
  if (const char* e = std::getenv("VPIPE_GEMMA_GTILE_ATTN")) {
    m->_gtile_attn = m->_gtile_attn && (std::atoi(e) != 0);
  }
  // Linearized sliding ring read (mirror tail): read the bounded window as ONE
  // contiguous span (no per-key % ring_cap), token-exact and free/faster than
  // the modulo read. Default ON when the lin kernel is loaded; opt out with
  // VPIPE_GEMMA_RING_LINEAR=0. Alloc + mirror writes are always-on regardless.
  m->_ring_linear = m->_fn_sdpa_gqa_vec_lin.valid();
  if (const char* e = std::getenv("VPIPE_GEMMA_RING_LINEAR")) {
    m->_ring_linear = m->_ring_linear && (std::atoi(e) != 0);
  }
  // KV-split GQA paged flash-decode for the (paged) full layers: the fast
  // long-context decode path (the scalar paged kernel has no split). Default
  // ON when present; VPIPE_GEMMA_NO_PGQA=1 reverts to scalar sdpa_paged_causal.
  m->_pgqa_attn = m->_fn_sdpa_pgqa.valid() && m->_fn_sdpa_gqa_merge.valid()
      && std::getenv("VPIPE_GEMMA_NO_PGQA") == nullptr;
  // Materialized global decode: the omlx/MLX head_dim-512 fallback structure
  // (QK GEMV -> parallel softmax -> PV GEMV, no online-softmax serial chain).
  // Token-exact but M4-NEUTRAL -- the scores DRAM round-trip cancels the gain
  // the fused flash gets for free by keeping scores in registers (the KV-read
  // latency, identical for both, is the real bottleneck). KEPT default-off as
  // the M5 matrix-core substrate: the QK/PV are GEMM-shaped, so an MMA variant
  // flips the scratch-vs-compute economics that lose on M4. VPIPE_GEMMA_MAT_DECODE=1.
  m->_mat_decode = m->_pgqa_attn && m->_fn_dec_qk.valid()
      && m->_fn_dec_rowstat.valid() && m->_fn_dec_pv.valid()
      && m->_fn_dec_merge.valid()
      && std::getenv("VPIPE_GEMMA_MAT_DECODE") != nullptr;
  if (const char* e = std::getenv("VPIPE_GEMMA_GTILE_SPLIT")) {
    const int v = std::atoi(e);
    if (v >= 1 && v <= 512) { m->_gtile_split = v; }
  }
  if (const char* e = std::getenv("VPIPE_GEMMA_GTILE_KPS")) {
    const int v = std::atoi(e);
    if (v >= 0) { m->_gtile_kps = v; }   // 0 => fixed split (no adaptation)
  }
  // MMA q-head flash-decode (global layers, 8|G): default ON; the split scales
  // with depth so partials are sized for _mma_qhead_cap = ceil(max_seq/64).
  m->_mma_qhead = m->_fn_sdpa_mma_qhead.valid()
      && std::getenv("VPIPE_GEMMA_NO_MMA_QHEAD") == nullptr;
  {
    const int max_seq = cfg.page_tokens * std::max(1, cfg.max_pages);
    int cap = (max_seq + 63) / 64;
    if (cap < 1) { cap = 1; }
    if (cap > 512) { cap = 512; }
    m->_mma_qhead_cap = cap;
  }
  if (m->_embed_is_q6k &&
      (!m->_fn_embed_q6k.valid() || !m->_fn_qmv_q6k.valid())) {
    return nullptr;
  }

  // Matrix-core (M5+) prefill GEMM: only when the GPU has hardware matrix
  // units and the checkpoint is uniformly 4-bit (e4b). dequant a projection
  // weight -> dense f16 scratch, then dense matmul2d on the matrix units;
  // ~2-2.5x the steel quantized GEMM at prefill row counts. The 8-bit-MLP
  // 12B (gemma4_unified) stays on steel. Same source keeps the steel path on
  // non-matrix-core GPUs. VPIPE_GEMMA_NO_MMA=1 forces steel (A/B + safety).
  if (cfg.quant_bits != 8 && mlp_bits != 8 && mc->supports_matrix_cores()) {
    m->_lib_dequant = mc->load_library("affine_dequant" + sfx);
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma" + sfx);
    m->_fn_dequant = m->_lib_dequant.function("affine_dequant_w4" + gtok);
    // Tile-adaptive dense matmul2d: 128x128 for K <= 4096 (q/k/v/o, gate/up),
    // 128x256 for deeper K (down_proj, K=ffn) where the square tile is
    // bandwidth-starved on weight streaming (M5 gemm_mma.tune).
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    // Causal-tiled matmul2d QK for the MATERIALIZED attention core (M5): the
    // diagonal-grid tile-skip on the matrix units. Only QK moves to matmul2d;
    // softmax stays banded and PV stays steel causal-K-capped (a full-K matmul2d
    // PV loses at 16k where PV dominates, and the matmul2d K-cap needs a tiny BM
    // that starves the matrix units).
    m->_fn_dense_mma_qkcausal =
        m->_lib_dense_mma.function("dense_gemm_mma_t_qkcausal_n128_f16");
    m->_fn_geglu_inter = m->_lib_elt.function("geglu_interleaved_f16");
    m->_use_mma = m->_fn_dequant.valid() && m->_fn_dense_mma.valid() &&
                  m->_fn_dense_mma_deep.valid() &&
                  m->_fn_geglu_inter.valid();
    if (const char* e = std::getenv("VPIPE_GEMMA_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_GEMMA_NO_MMA")) {
      if (std::atoi(e) != 0) { m->_use_mma = false; }
    }
    if (const char* e = std::getenv("VPIPE_GEMMA_MMA_NODQ")) {
      m->_skip_dequant = (std::atoi(e) != 0);   // diagnostic only
    }
    // Materialized-attention matmul2d core (M5): causal QK on the matrix units
    // + full-K matmul2d PV, behind the materialized path. Default ON when the
    // matmul2d GEMM path is on (e4b 4-bit + matrix cores); VPIPE_GEMMA_MAT_NO_MMA2
    // forces the materialized GEMMs back to steel (A/B the matrix-core attn).
    m->_mat_mma = m->_use_mma && m->_fn_dense_mma_qkcausal.valid();
    if (const char* e = std::getenv("VPIPE_GEMMA_MAT_NO_MMA2")) {
      if (std::atoi(e) != 0) { m->_mat_mma = false; }
    }
  }

  // ---- conversion helpers (mirror MetalQwenModel) -----------------
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
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = f32_to_elt_(bf16_to_f32_(s[i]), bf16); }
    } else if (info->dtype == "F16") {
      const auto* s = static_cast<const _Float16*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = f32_to_elt_((float)s[i], bf16); }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = f32_to_elt_(s[i], bf16); }
    } else {
      return {};
    }
    return out;
  };
  auto to_f16 = to_elt;
  auto qtri = [&](const std::string& pfx, SharedBuffer& w, SharedBuffer& s,
                  SharedBuffer& b) -> bool {
    w = wts->load(pfx + ".weight", mc);
    s = to_f16(pfx + ".scales");
    b = to_f16(pfx + ".biases");
    return !w.empty() && !s.empty() && !b.empty();
  };
  // Dequantize a 4-bit affine [out, in] weight into a DENSE elt [out, in]
  // for the dense_gemm_t path. Used for per_layer_projection whose in dim
  // (hpli=256) is < the qmv block_size (512). MLX g64 packing: 8 nibbles
  // per u32 (low-nibble first), scales/biases per group of 64.
  auto dequant_dense = [&](const std::string& pfx, int out_dim,
                           int in_dim) -> SharedBuffer {
    SharedBuffer wq = wts->load(pfx + ".weight", mc);
    const auto* sinfo = wts->info(pfx + ".scales");
    const auto* binfo = wts->info(pfx + ".biases");
    if (wq.empty() || sinfo == nullptr || binfo == nullptr) { return {}; }
    SharedBuffer sraw = wts->load(pfx + ".scales", mc);
    SharedBuffer braw = wts->load(pfx + ".biases", mc);
    if (sraw.empty() || braw.empty()) { return {}; }
    const bool sbf = sinfo->dtype == "BF16";
    const int gs = qg;
    const int ng = in_dim / gs;
    const int words = in_dim / 8;
    const auto* wptr = static_cast<const std::uint32_t*>(wq.contents());
    const auto* sptr = static_cast<const std::uint16_t*>(sraw.contents());
    const auto* bptr = static_cast<const std::uint16_t*>(braw.contents());
    SharedBuffer out = mc->make_shared_buffer((std::size_t)out_dim * in_dim * 2);
    auto* o = static_cast<std::uint16_t*>(out.contents());
    for (int r = 0; r < out_dim; ++r) {
      for (int c = 0; c < in_dim; ++c) {
        const std::uint32_t word = wptr[(std::size_t)r * words + c / 8];
        const int nib = (int)((word >> (4 * (c % 8))) & 0xF);
        const int grp = c / gs;
        const float sc = elt_to_f32_(sptr[(std::size_t)r * ng + grp], sbf);
        const float bi = elt_to_f32_(bptr[(std::size_t)r * ng + grp], sbf);
        o[(std::size_t)r * in_dim + c] = f32_to_elt_((float)nib * sc + bi, bf16);
      }
    }
    return out;
  };

  const std::string& P = cfg.weight_prefix;   // "language_model.model."

  // ---- model-level weights ----------------------------------------
  // Per-Layer-Input (PLE) embeddings exist only when hpli > 0 (e4b). The
  // gemma4_unified 12B ships none of the PLE weights.
  const bool has_ple = cfg.hpli > 0;
  if (m->_embed_is_q6k) {
    // Lossless raw Q6_K table: load it once; skip the affine8 requant
    // entirely (ensure_embed_ never runs -> the ~25% memory saving is real).
    m->_embed_q6k = wts->load(P + "embed_tokens.q6k", mc);
    if (m->_embed_q6k.empty()) { return nullptr; }
  } else if (!qtri(P + "embed_tokens", m->_embed_w, m->_embed_s,
                   m->_embed_b)) {
    return nullptr;
  }
  if (has_ple) {
    if (!qtri(P + "embed_tokens_per_layer", m->_ple_w, m->_ple_s, m->_ple_b)) {
      return nullptr;
    }
    // per_layer_model_projection ships BF16 (not quantized) -> f16 dense.
    m->_plm_proj_w = to_f16(P + "per_layer_model_projection.weight");
    m->_ple_proj_norm = to_f16(P + "per_layer_projection_norm.weight");
    if (m->_plm_proj_w.empty() || m->_ple_proj_norm.empty()) {
      return nullptr;
    }
  }
  m->_final_ln = to_f16(P + "norm.weight");
  if (m->_final_ln.empty()) {
    return nullptr;
  }

  // ---- per-layer-type RoPE inv_freq -------------------------------
  // Sliding: full rotation over head_dim_sliding, base theta_sliding.
  {
    const int hd = cfg.head_dim_sliding, half = hd / 2;
    m->_inv_freq_sliding = mc->make_shared_buffer((std::size_t)half * 4);
    auto* f = static_cast<float*>(m->_inv_freq_sliding.contents());
    for (int i = 0; i < half; ++i) {
      f[i] = 1.0f / std::pow(cfg.rope_theta_sliding,
                             (2.0f * (float)i) / (float)hd);
    }
  }
  // Full: proportional partial rotary -- the first rotated_dims/2 pairs
  // rotate (base theta_full, normalised by the FULL head_dim_full), the
  // rest pass through (inv_freq 0). rope_f16 pairs (j, j+head_dim_full/2).
  {
    const int hd = cfg.head_dim_full, half = hd / 2;
    const int rot = (int)(hd * cfg.full_partial_rotary);
    const int rhalf = rot / 2;
    m->_inv_freq_full = mc->make_shared_buffer((std::size_t)half * 4);
    auto* f = static_cast<float*>(m->_inv_freq_full.contents());
    for (int i = 0; i < half; ++i) {
      f[i] = (i < rhalf)
          ? 1.0f / std::pow(cfg.rope_theta_full,
                            (2.0f * (float)i) / (float)hd)
          : 0.0f;
    }
  }
  // v_norm weight = all ones [head_dim_full] (RMSNorm with no learnable
  // scale; reused for sliding head_dim by reading the first 256 entries).
  m->_ones_vnorm = mc->make_shared_buffer((std::size_t)cfg.head_dim_full * 2);
  {
    auto* o = static_cast<std::uint16_t*>(m->_ones_vnorm.contents());
    for (int i = 0; i < cfg.head_dim_full; ++i) { o[i] = f32_to_elt_(1.0f, bf16); }
  }

  // Interleave gate/up [ffn, hidden] quantized weights into one [2*ffn,
  // hidden] (row 2g=gate g, 2g+1=up g) so the GeGLU activation fuses into
  // the matmul store (affine_qmv_geglu / affine_qmm_geglu) in BOTH decode
  // and prefill -- so only the interleaved weight is stored (same bytes as
  // gate+up, no extra memory) and the separate gate/up scratch is dropped.
  // u32 columns per row: hidden*mlp_bits/32 (4-bit -> hidden/8, 8-bit ->
  // hidden/4). The interleave is byte-level so this is the only bit-width
  // dependency. scales/biases are per group-of-64 regardless of bits.
  const std::size_t gu_wrow =
      (std::size_t)cfg.hidden * (std::size_t)mlp_bits / 32;    // u32/row
  const std::size_t gu_grow =
      (std::size_t)cfg.hidden / (std::size_t)qg;               // f16/row
  auto interleave_gu = [&](const SharedBuffer& gw, const SharedBuffer& gs,
                           const SharedBuffer& gb, const SharedBuffer& uw,
                           const SharedBuffer& us, const SharedBuffer& ub,
                           SharedBuffer& ow, SharedBuffer& os,
                           SharedBuffer& ob) -> bool {
    const std::size_t Fc = (std::size_t)cfg.ffn_inner;
    ow = mc->make_shared_buffer(2 * Fc * gu_wrow * 4);
    os = mc->make_shared_buffer(2 * Fc * gu_grow * 2);
    ob = mc->make_shared_buffer(2 * Fc * gu_grow * 2);
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
    for (std::size_t g = 0; g < Fc; ++g) {
      std::memcpy(owp + (2 * g) * gu_wrow, gwp + g * gu_wrow, gu_wrow * 4);
      std::memcpy(owp + (2 * g + 1) * gu_wrow, uwp + g * gu_wrow, gu_wrow * 4);
      std::memcpy(osp + (2 * g) * gu_grow, gsp + g * gu_grow, gu_grow * 2);
      std::memcpy(osp + (2 * g + 1) * gu_grow, usp + g * gu_grow, gu_grow * 2);
      std::memcpy(obp + (2 * g) * gu_grow, gbp + g * gu_grow, gu_grow * 2);
      std::memcpy(obp + (2 * g + 1) * gu_grow, ubp + g * gu_grow, gu_grow * 2);
    }
    return true;
  };

  // Row-concatenate q|k|(v) quantized weights into one [q+k+v rows, hidden]
  // buffer for the fused decode GEMV (affine_qmv_qkv). All three share the base
  // bits/group so the concat is a byte-level row append; v empty => q|k only
  // (k_eq_v). Skips (per-layer fallback) if any byte size is not a clean
  // multiple of the base-bits row stride (mixed-precision safety).
  const std::size_t qkv_wrow_b =
      (std::size_t)cfg.hidden * (std::size_t)cfg.quant_bits / 8;   // bytes/row
  const std::size_t qkv_grow_b = (std::size_t)(cfg.hidden / qg) * 2;  // f16/row
  auto build_qkv = [&](Layer& ly) -> void {
    if (qkv_wrow_b == 0 || ly.qw.empty() || ly.kw.empty()) { return; }
    const bool vok = !ly.vw.empty();
    if ((ly.qw.byte_size() % qkv_wrow_b) || (ly.kw.byte_size() % qkv_wrow_b) ||
        (vok && (ly.vw.byte_size() % qkv_wrow_b))) {
      return;                                    // not uniform base bits
    }
    const int qr = (int)(ly.qw.byte_size() / qkv_wrow_b);
    const int kr = (int)(ly.kw.byte_size() / qkv_wrow_b);
    const int vr = vok ? (int)(ly.vw.byte_size() / qkv_wrow_b) : 0;
    if ((qr % 8) || (kr % 8) || (vr % 8)) { return; }   // 8-row TG boundary
    // Reject mixed precision (e.g. OptiQ 8-bit q/k/v): the derived row counts
    // must match the expected output dims, else an 8-bit tensor would parse as
    // 2x the rows at the base 4-bit stride and corrupt the concat.
    const int qd_exp = cfg.n_heads * ly.head_dim;
    const int kd_exp = ly.n_kv * ly.head_dim;
    if (qr != qd_exp || kr != kd_exp || (vr && vr != kd_exp)) { return; }
    const int rows = qr + kr + vr;
    ly.qkvw = mc->make_shared_buffer((std::size_t)rows * qkv_wrow_b);
    ly.qkvs = mc->make_shared_buffer((std::size_t)rows * qkv_grow_b);
    ly.qkvb = mc->make_shared_buffer((std::size_t)rows * qkv_grow_b);
    if (ly.qkvw.empty() || ly.qkvs.empty() || ly.qkvb.empty()) { return; }
    auto cat = [](const SharedBuffer& dst, std::size_t& off,
                  const SharedBuffer& src) {
      std::memcpy(static_cast<std::uint8_t*>(dst.contents()) + off,
                  src.contents(), src.byte_size());
      off += src.byte_size();
    };
    std::size_t wo = 0, so = 0, bo = 0;
    cat(ly.qkvw, wo, ly.qw); cat(ly.qkvw, wo, ly.kw);
    cat(ly.qkvs, so, ly.qs); cat(ly.qkvs, so, ly.ks);
    cat(ly.qkvb, bo, ly.qb); cat(ly.qkvb, bo, ly.kb);
    if (vr) {
      cat(ly.qkvw, wo, ly.vw); cat(ly.qkvs, so, ly.vs); cat(ly.qkvb, bo, ly.vb);
    }
    ly.qkv_qrows = qr; ly.qkv_krows = kr; ly.qkv_vrows = vr;
    ly.qkv_fused = true;
    // Re-point the per-proj weights at slices of the concat so PREFILL (which
    // still calls qmm(ly.qw/kw/vw)) reads the SAME storage as decode's fused
    // GEMV -- no duplication. Reassignment frees the standalone q/k/v buffers
    // copied above (subview shares qkvw's allocation by refcount).
    const std::size_t qwb = (std::size_t)qr * qkv_wrow_b;
    const std::size_t kwb = (std::size_t)kr * qkv_wrow_b;
    const std::size_t qgb = (std::size_t)qr * qkv_grow_b;
    const std::size_t kgb = (std::size_t)kr * qkv_grow_b;
    ly.qw = ly.qkvw.subview(0, qwb);
    ly.qs = ly.qkvs.subview(0, qgb);
    ly.qb = ly.qkvb.subview(0, qgb);
    ly.kw = ly.qkvw.subview(qwb, kwb);
    ly.ks = ly.qkvs.subview(qgb, kgb);
    ly.kb = ly.qkvb.subview(qgb, kgb);
    if (vr) {
      ly.vw = ly.qkvw.subview(qwb + kwb, (std::size_t)vr * qkv_wrow_b);
      ly.vs = ly.qkvs.subview(qgb + kgb, (std::size_t)vr * qkv_grow_b);
      ly.vb = ly.qkvb.subview(qgb + kgb, (std::size_t)vr * qkv_grow_b);
    }
  };

  // ---- per-layer bind ---------------------------------------------
  m->_layers.resize(cfg.n_layers);   // Layer is move-only (SharedBuffer)
  const int first_shared = cfg.first_shared();
  for (int L = 0; L < cfg.n_layers; ++L) {
    Layer& ly = m->_layers[L];
    ly.is_full = cfg.layer_is_full(L);
    ly.head_dim = cfg.head_dim(L);
    ly.n_kv = cfg.n_kv(L);
    ly.k_eq_v = cfg.k_eq_v(L);
    const bool shared = (cfg.num_kv_shared > 0) && (L >= first_shared);
    if (shared) {
      int src = -1;
      for (int j = first_shared - 1; j >= 0; --j) {
        if (cfg.layer_is_full(j) == ly.is_full) { src = j; break; }
      }
      if (src < 0) { return nullptr; }
      ly.kv_source = src;
    }
    const std::string p = P + "layers." + std::to_string(L) + ".";
    ly.in_ln       = to_f16(p + "input_layernorm.weight");
    ly.post_attn_ln = to_f16(p + "post_attention_layernorm.weight");
    ly.pre_ffn_ln  = to_f16(p + "pre_feedforward_layernorm.weight");
    ly.post_ffn_ln = to_f16(p + "post_feedforward_layernorm.weight");
    if (has_ple) {
      ly.post_pli_ln = to_f16(p + "post_per_layer_input_norm.weight");
    }
    ly.q_norm = to_f16(p + "self_attn.q_norm.weight");
    SharedBuffer gw, gs, gb, uw, us, ub;   // freed after interleave
    bool ok = qtri(p + "self_attn.q_proj", ly.qw, ly.qs, ly.qb)
           && qtri(p + "self_attn.o_proj", ly.ow, ly.os, ly.ob)
           && qtri(p + "mlp.gate_proj", gw, gs, gb)
           && qtri(p + "mlp.up_proj", uw, us, ub)
           && interleave_gu(gw, gs, gb, uw, us, ub,
                            ly.guw, ly.gus, ly.gub)
           && qtri(p + "mlp.down_proj", ly.dw, ly.ds, ly.db);
    if (has_ple) {
      ok = ok
        && qtri(p + "per_layer_input_gate", ly.plg_w, ly.plg_s, ly.plg_b);
      // per_layer_projection: native 4-bit qmv/qmm (qmv handles K=hpli=256 <
      // block_size via the partial-block safe load) or f16 dense (A/B).
      if (m->_ple_quant) {
        ok = ok && qtri(p + "per_layer_projection", ly.plp_w, ly.plp_s,
                        ly.plp_b);
      } else {
        ly.plp_w = dequant_dense(p + "per_layer_projection", cfg.hidden,
                                 cfg.hpli);
      }
    }
    if (ly.kv_source < 0) {
      // k_eq_v full layers have no v_proj (values reuse the k_proj output).
      ok = ok && qtri(p + "self_attn.k_proj", ly.kw, ly.ks, ly.kb)
              && (ly.k_eq_v
                  || qtri(p + "self_attn.v_proj", ly.vw, ly.vs, ly.vb));
      ly.k_norm = to_f16(p + "self_attn.k_norm.weight");
      if (ly.k_norm.empty()) { return nullptr; }
      if (m->_qkv_fuse) { build_qkv(ly); }   // concat for the fused decode GEMV
    }
    // layer_scalar [1] -> host float.
    {
      SharedBuffer ls = to_f16(p + "layer_scalar");
      if (ls.empty()) { return nullptr; }
      ly.layer_scalar =
          elt_to_f32_(*static_cast<const std::uint16_t*>(ls.contents()), bf16);
    }
    if (!ok || ly.in_ln.empty() || ly.post_attn_ln.empty() ||
        ly.pre_ffn_ln.empty() || ly.post_ffn_ln.empty() ||
        (has_ple && (ly.post_pli_ln.empty() || ly.plp_w.empty())) ||
        ly.q_norm.empty()) {
      return nullptr;
    }
  }

  // Contiguous KV via the shared ContextManager: per-layer head_dim
  // (sliding 256 / full 512), cross-layer KV sharing (kv_source), and
  // sliding-window layers -- the topology that doesn't fit the paged
  // pool. The per-layer is_full/head_dim/kv_source were just computed in
  // the bind loop above.
  {
    // Ring chunk B: a smaller B bounds sliding layers more aggressively
    // (and is exercised by the long-context wrap test). Default cfg value;
    // VPIPE_GEMMA_SLIDING_CHUNK overrides (0 disables the ring).
    int sliding_chunk = cfg.sliding_chunk;
    if (const char* e = std::getenv("VPIPE_GEMMA_SLIDING_CHUNK")) {
      const int v = std::atoi(e);
      if (v >= 0) { sliding_chunk = v; }
    }
    m->_sliding_chunk = sliding_chunk;

    ContextManager::Spec sp;
    sp.metal          = mc;
    sp.kv_layout      = ContextManager::KvLayout::Contiguous;
    sp.n_layers       = cfg.n_layers;
    sp.n_kv_heads     = cfg.n_kv_heads;
    sp.head_dim       = cfg.head_dim_full;     // fallback; per-layer below
    // FULL (global) layers use the shared PAGED pool (page_tokens-token pages,
    // a multiple of the mma2 kernel's BK=16 over-read block) so branch children
    // share the parent's frozen prefix pages by refcount -- no per-branch
    // full-KV copy. Sliding layers keep the bounded contiguous ring.
    sp.page_tokens    = 256;
    // A/B knob: scan the paged-pool page size (page-walk granularity in the
    // pflash/pgqa kernels). Must stay a multiple of 16 (mma2 BK over-read).
    if (const char* e = std::getenv("VPIPE_GEMMA_PAGE_TOKENS")) {
      const int v = std::atoi(e);
      if (v >= 16 && v <= 8192 && (v % 16) == 0) { sp.page_tokens = v; }
    }
    sp.max_seq        = cfg.max_seq;
    sp.sliding_window = cfg.sliding_window;
    sp.sliding_chunk  = sliding_chunk;
    sp.layer_kind.resize((std::size_t)cfg.n_layers);
    sp.layer_head_dim.resize((std::size_t)cfg.n_layers);
    sp.layer_n_kv_heads = cfg.layer_n_kv_heads;   // empty for e4b (uniform)
    sp.kv_source.resize((std::size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; ++L) {
      sp.layer_kind[(std::size_t)L] = m->_layers[L].is_full
          ? ContextManager::LayerKind::Full
          : ContextManager::LayerKind::Sliding;
      sp.layer_head_dim[(std::size_t)L] = m->_layers[L].head_dim;
      sp.kv_source[(std::size_t)L]      = m->_layers[L].kv_source;
    }
    m->_ctx = std::make_unique<ContextManager>(sp, nullptr);
    // Full (global) layers are paged when the manager could pool them (uniform
    // full-layer dims -- always true for Gemma e4b/12B). The exec routes full
    // layers through kv_write_paged + sdpa_paged_* and sliding layers through
    // the contiguous ring. If pooling failed, fall back is unsupported (the
    // contiguous full path was removed), so require it.
    m->_full_paged = m->_ctx->full_layers_paged();
    if (!m->_full_paged) {
      return nullptr;
    }
  }
  return m;
}

MetalGemmaModel::Config
MetalGemmaModel::config_from(const ModelConfig& c)
{
  Config m;
  m.n_layers   = c.n_layers;
  m.hidden     = c.hidden;
  m.n_heads    = c.n_heads;
  m.n_kv_heads = c.n_kv_heads;
  m.head_dim_sliding = c.gemma4.head_dim_sliding;
  m.head_dim_full    = c.gemma4.head_dim_full;
  m.ffn_inner  = c.ffn_inner;
  m.vocab      = c.vocab_size;
  m.num_kv_shared = c.gemma4.num_kv_shared_layers;
  m.hpli       = c.gemma4.hidden_per_layer_input;
  m.sliding_window = c.gemma4.sliding_window;
  m.final_softcap  = c.gemma4.final_logit_softcapping;
  m.rope_theta_sliding = c.gemma4.rope_theta_sliding;
  m.rope_theta_full    = c.gemma4.rope_theta_full;
  m.full_partial_rotary = c.gemma4.full_partial_rotary_factor;
  m.rms_eps    = c.rms_eps;
  m.quant_bits = c.quantization.bits > 0 ? c.quantization.bits : 4;
  m.quant_group = c.quantization.group_size > 0 ? c.quantization.group_size
                                                : 64;
  m.quant_symmetric = c.quantization.symmetric;
  m.attention_k_eq_v   = c.gemma4.attention_k_eq_v;
  m.layer_n_kv_heads   = c.gemma4.layer_n_kv_heads;
  m.is_full_layer = c.gemma4.is_full_layer;
  // Per-context KV is preallocated contiguous [Hkv, max_seq, head_dim]
  // per owned layer (no lazy growth), so a generous cap costs memory.
  // This is only the standalone/test default: the LM load path overrides
  // max_seq with the configured budget (page_tokens * max_pages) so the
  // user's knobs actually control the conversation length.
  m.max_seq    = 2048;
  return m;
}

// Per-machine decode-attention SPLIT autotune (gtile-vec flash-decode). The
// split count sp trades parallelism against the merge cost; the sweet spot is a
// GPU property. Two regimes: the GLOBAL (head_dim_full, long-context) layers --
// where sp pins to the _gtile_split CAP (the flagged D=512 long-context gap) --
// and the window-bounded SLIDING layers (head_dim_sliding, ~window keys, set by
// _gtile_kps). Greedy-token-exact safe: the split never changes the output, only
// how the scan is parallelized. VPIPE_GEMMA_ATTN_AUTOTUNE=0 skips; an explicit
// VPIPE_GEMMA_GTILE_SPLIT / _KPS (read in load()) is respected (no probe).
void
MetalGemmaModel::tune_decode_attn_(TuningReport& rep)
{
  if (!_gtile_attn || !_fn_sdpa_gqa_vec.valid()
      || !_fn_sdpa_gqa_merge.valid()) {
    return;
  }
  if (const char* e = std::getenv("VPIPE_GEMMA_ATTN_AUTOTUNE")) {
    if (std::atoi(e) == 0) { return; }
  }
  const bool split_locked = std::getenv("VPIPE_GEMMA_GTILE_SPLIT") != nullptr;
  const bool kps_locked   = std::getenv("VPIPE_GEMMA_GTILE_KPS") != nullptr;
  if (split_locked && kps_locked) { return; }
  const bool log = std::getenv("VPIPE_GEMMA_AUTOTUNE_LOG") != nullptr;

  const Config& c = _cfg;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads;
  if (Hkv <= 0 || Hq % Hkv != 0) { return; }
  const int G = Hq / Hkv;
  const int Dg = c.head_dim_full, Ds = c.head_dim_sliding;
  const bool global_ok  = (Dg % 128 == 0);     // gtile-vec needs D % 128 == 0
  const bool sliding_ok = (Ds % 128 == 0);
  if (!global_ok && !sliding_ok) { return; }

  // Synthetic q[Hq,D] attends a contiguous KV pool [Hkv,cap,D]. Buffers sized
  // for the largest regime (global D, long T_kv, sp cap 256 -- matching the
  // decode partials below); uninitialised values are fine (timing only).
  const int kProbeTg = 16384;                  // long-context global probe
  const int win = (c.sliding_window > 0) ? c.sliding_window : 512;
  const int Dmax = (Dg > Ds) ? Dg : Ds;
  const int sp_max = 256;
  const std::size_t qb = (std::size_t)Hq * Dmax * 2;
  const std::size_t kvg = (std::size_t)Hkv * kProbeTg * Dg * 2;
  metal_compute::SharedBuffer q  = _mc->make_shared_buffer(qb);
  metal_compute::SharedBuffer kp = _mc->make_shared_buffer(kvg);
  metal_compute::SharedBuffer vp = _mc->make_shared_buffer(kvg);
  metal_compute::SharedBuffer oacc =
      _mc->make_shared_buffer((std::size_t)Hq * sp_max * Dmax * sizeof(float));
  metal_compute::SharedBuffer mm =
      _mc->make_shared_buffer((std::size_t)Hq * sp_max * sizeof(float));
  metal_compute::SharedBuffer ll =
      _mc->make_shared_buffer((std::size_t)Hq * sp_max * sizeof(float));
  metal_compute::SharedBuffer at = _mc->make_shared_buffer(qb);
  if (q.empty() || kp.empty() || vp.empty() || oacc.empty() || mm.empty()
      || ll.empty() || at.empty()) {
    return;
  }

  // One gtile-vec flash-decode + merge at (D, T_kv, window, cap, sp).
  auto encode = [&](metal_compute::ComputeEncoder& enc, int D, int T_kv,
                    int window, int cap, int sp) {
    const float scale = 1.0f / std::sqrt((float)D);
    enc.set_function(_fn_sdpa_gqa_vec);
    enc.set_buffer(0, q);
    enc.set_buffer(1, kp);
    enc.set_buffer(2, vp);
    enc.set_buffer(3, oacc);
    enc.set_buffer(4, mm);
    enc.set_buffer(5, ll);
    enc.set_constant(6, scale);
    enc.set_constant(7, T_kv);
    enc.set_constant(8, D);
    enc.set_constant(9, Hq);
    enc.set_constant(10, Hkv);
    enc.set_constant(11, T_kv - 1);            // q_offset (kv_off)
    enc.set_constant(12, cap);                 // kv_stride (head stride)
    enc.set_constant(13, window);
    enc.set_constant(14, 0);                   // ring_cap (0 = linear)
    enc.set_constant(15, sp);
    enc.dispatch({32, (unsigned)Hq, (unsigned)sp}, {32, (unsigned)G, 1});
    enc.set_function(_fn_sdpa_gqa_merge);
    enc.set_buffer(0, oacc);
    enc.set_buffer(1, mm);
    enc.set_buffer(2, ll);
    enc.set_buffer(3, at);
    enc.set_constant(4, D);
    enc.set_constant(5, sp);
    enc.set_constant(6, Hq);
    enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::string detail;

  // GLOBAL long-context: pick the split CAP (_gtile_split).
  if (global_ok && !split_locked) {
    const int cand[] = {64, 128, 192, 256};
    auto bench = [&](int i) -> double {
      return autotune_time(_mc, 2, [&](metal_compute::ComputeEncoder& enc) {
        encode(enc, Dg, kProbeTg, 0, kProbeTg, cand[i]);
      });
    };
    std::vector<double> us;
    const int w = autotune_vote(4, 3, 2, bench, &us);
    // On a GPU that already saturates at the default split the candidates are
    // near-equivalent (~noise), so the per-round winner flips run-to-run. Move
    // the cap only on a LARGE (>15%) clear win -- a genuine per-machine signal
    // (e.g. more cores / matrix units favouring a different split), never noise.
    int def_idx = 1;
    for (int i = 0; i < 4; ++i) { if (cand[i] == _gtile_split) { def_idx = i; } }
    const bool moved = (cand[w] != _gtile_split) && (us[w] < 0.85 * us[def_idx]);
    if (moved) { _gtile_split = cand[w]; }
    detail += "split=" + std::to_string(_gtile_split)
        + (moved ? "" : "(def)");
    if (log) {
      std::string line = "[gemma] decode-attn global split:";
      for (int i = 0; i < 4; ++i) {
        char bb[40];
        std::snprintf(bb, sizeof(bb), " sp%d=%.0fus", cand[i], us[i]);
        line += bb;
      }
      std::fprintf(stderr, "%s -> %d%s\n", line.c_str(), _gtile_split,
                   moved ? "" : " (kept default)");
    }
  }

  // SLIDING window: the window-bounded layers scan only ~`win` keys, so the
  // split is usually in the noise floor (the kernel-launch + merge cost
  // dominates). Probe it, but only move _gtile_kps off its robust default when a
  // candidate CLEARLY wins -- noise must not drive a mid-context regression
  // (_gtile_kps also sets sp for moderate global scans). _gtile_kps = round(win
  // / sp) reproduces the chosen split via the model's sp = ceil(scan/_gtile_kps).
  if (sliding_ok && !kps_locked) {
    const int cand[] = {8, 16, 32, 64};
    auto bench = [&](int i) -> double {
      return autotune_time(_mc, 3, [&](metal_compute::ComputeEncoder& enc) {
        encode(enc, Ds, win, win, win, cand[i]);
      });
    };
    std::vector<double> us;
    const int w = autotune_vote(4, 3, 3, bench, &us);
    const int best_sp = cand[w];
    // The current default's split (sp = ceil(win/_gtile_kps)); keep it unless
    // the winner beats that candidate's time by a clear (>8%) margin.
    const int def_sp = (win + _gtile_kps - 1) / _gtile_kps;
    int def_idx = 1;
    for (int i = 0; i < 4; ++i) { if (cand[i] == def_sp) { def_idx = i; } }
    // ~512-key scan sits in the launch/merge noise floor (run-to-run spread seen
    // up to ~11%); require a LARGE (>20%) margin so only a genuine per-machine
    // signal -- never noise -- moves kps off its robust default.
    bool changed = false;
    if (best_sp != def_sp && us[w] < 0.80 * us[def_idx]) {
      int kps = (win + best_sp / 2) / best_sp;
      _gtile_kps = (kps < 1) ? 1 : (kps > 64 ? 64 : kps);
      changed = true;
    }
    if (!detail.empty()) { detail += "/"; }
    detail += "kps=" + std::to_string(_gtile_kps)
        + (changed ? "" : "(def)");
    if (log) {
      std::string line = "[gemma] decode-attn sliding split:";
      for (int i = 0; i < 4; ++i) {
        char bb[40];
        std::snprintf(bb, sizeof(bb), " sp%d=%.0fus", cand[i], us[i]);
        line += bb;
      }
      std::fprintf(stderr, "%s -> sp=%d kps=%d%s\n", line.c_str(), best_sp,
                   _gtile_kps, changed ? "" : " (kept default; within noise)");
    }
  }

  const double ms = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count() * 1e3;
  rep.add("gemma-decode-attn", ms, detail);
  if (log) {
    std::fprintf(stderr, "[gemma] decode-attn tuning %.0fms -> %s\n", ms,
                 detail.c_str());
  }
}

bool
MetalGemmaModel::ensure_scratch_()
{
  if (_scratch_ready) { return true; }
  const Config& c = _cfg;
  const int H = c.hidden;
  const int qd = c.n_heads * c.head_dim_full;       // max (full layers)
  const int kd = c.n_kv_heads * c.head_dim_full;
  const int ple = c.n_layers * c.hpli;
  auto b = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  _d_x = b(H);
  _d_hn = b(H);
  _d_q = b(qd);
  _d_k = b(kd);
  _d_v = b(kd);
  _d_attn = b(qd);
  // Materialized GLOBAL-prefill scratch (VPIPE_GEMMA_MATERIALIZED_GLOBAL): the
  // per-head score matrix [CAP,CAP] + V transpose [Hkv,D,CAP]. CAP bounds the
  // preallocation so we never reserve max_seq^2 (e.g. 64k^2 -> 8 GB); beyond
  // CAP the path falls back to pflash. Default 16384 (scores 16384^2*2 =
  // 512 MB at 16k; scales DOWN with max_seq for shorter contexts).
  // VPIPE_GEMMA_MATERIALIZED_CAP overrides (multiple of 16).
  if (_materialized_global) {
    int cap = 16384;
    if (const char* e = std::getenv("VPIPE_GEMMA_MATERIALIZED_CAP")) {
      const int v = std::atoi(e);
      if (v >= 512 && (v % 16) == 0) { cap = v; }
    }
    _scores_cap = std::min(c.max_seq, cap);
    _d_scores = b((std::size_t)_scores_cap * _scores_cap);
    _d_vT = b((std::size_t)c.n_kv_heads * c.head_dim_full * _scores_cap);
    if (_d_scores.empty() || _d_vT.empty()) {
      _materialized_global = false;     // alloc failed -> fall back to pflash
      _scores_cap = 0;
    }
  }
  _d_o = b(H);
  _d_act = b(c.ffn_inner);   // fused gate/up+geglu output (no gate/up bufs)
  _d_mlp = b(H);
  _d_ple = b(ple);
  _d_pleproj = b(ple);
  _d_pli = b(ple);
  _d_plg = b(c.hpli);
  _d_plp = b(H);
  // Pipelined PLE (Design A): per-chunk pli/pleproj buffers + private _d_x copy.
  _ple_chunk_k = 0;
  if (c.hpli > 0) {
    if (const char* e = std::getenv("VPIPE_GEMMA_PLE_CHUNK_K")) {
      _ple_chunk_k = std::max(0, std::atoi(e));
    }
    if (_ple_chunk_k > 0) {
      const int nch = (c.n_layers + _ple_chunk_k - 1) / _ple_chunk_k;
      _d_pli_ch.resize(nch);
      _d_pleproj_ch.resize(nch);
      for (int ci = 0; ci < nch; ++ci) {
        const int a = ci * _ple_chunk_k;
        const int nlc = std::min(a + _ple_chunk_k, c.n_layers) - a;
        _d_pli_ch[ci] = b((std::size_t)nlc * c.hpli);
        _d_pleproj_ch[ci] = b((std::size_t)nlc * c.hpli);
      }
      _d_x_ple = b(H);
    }
  }
  _d_logits = b(c.vocab);
  _d_tok = _mc->make_shared_buffer(sizeof(std::int32_t));
  _d_argmax_id = _mc->make_shared_buffer(sizeof(std::int32_t));
  // [2*kArgmaxM] f32 (val,idx) partials for the two-stage argmax.
  _d_argmax_part =
      _mc->make_shared_buffer((std::size_t)2 * kArgmaxM * sizeof(float));
  // Histogram-sampler scratch (multi-tg sample_*_f16). Shared across pdecode
  // steps; the per-context event serialises them on the GPU. Sizes derive from
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
  _d_dummy = _mc->make_shared_buffer(256 * 2);   // DUMMY_DISP scratch (f16)
  // Page table for the FULL-layer paged attention: max_pages triplets per
  // slot. kPgtabSlots slots form a ring so run-ahead pdecode (depth<=4) gives
  // each in-flight step its own table (the slot 0 path is used by every
  // synchronous caller, which waits before reusing it).
  if (_full_paged) {
    _d_pgtab = _mc->make_shared_buffer(
        (std::size_t)kPgtabSlots * _ctx->max_pages() * 3 * sizeof(std::int32_t));
  }
  // Autotune the decode-attention split for THIS GPU first -- the partials
  // below derive from _gtile_split, so the probe must set it beforehand.
  if (_gtile_attn || _gqa_attn || _pgqa_attn) {
    TuningReport tuning;
    tune_decode_attn_(tuning);
    if (std::getenv("VPIPE_GEMMA_AUTOTUNE_LOG") && !tuning.empty()) {
      std::fprintf(stderr, "[gemma] load-time kernel tuning %dms: %s\n",
                   (int)(tuning.total_ms() + 0.5), tuning.summary().c_str());
    }
  }
  // Split-decode partials (f32): O [Hq,split,D], m/l [Hq,split]. Shared by the
  // opt-in sliding GQA (D<=256) and the global threadgroup-staged flash-decode
  // (D == head_dim_full, up to 512); sized to the larger split + head_dim.
  if (_gqa_attn || _gtile_attn || _pgqa_attn) {
    // The MMA q-head decode kernel scales its split with depth up to
    // _mma_qhead_cap, so size the partials for that when it's active.
    const int gtile_sp = (_gtile_attn || _pgqa_attn)
        ? std::max(_gtile_split, _mma_qhead ? _mma_qhead_cap : 0) : 1;
    const std::size_t sp = (std::size_t)std::max(
        _gqa_attn ? _gqa_split : 1, gtile_sp);
    const std::size_t Hq = (std::size_t)c.n_heads;
    const std::size_t Dd = (_gtile_attn || _pgqa_attn)
        ? (std::size_t)c.head_dim_full
        : (std::size_t)(c.head_dim_full < 256 ? c.head_dim_full : 256);
    _d_gqa_oacc = _mc->make_shared_buffer(Hq * sp * Dd * sizeof(float));
    _d_gqa_m = _mc->make_shared_buffer(Hq * sp * sizeof(float));
    _d_gqa_l = _mc->make_shared_buffer(Hq * sp * sizeof(float));
    // Materialized global decode scores scratch: [Hq, Tstride] f32, Tstride =
    // full KV capacity (max_pages * page_tokens). PV partials/m/l reuse the
    // buffers above. _d_gqa_m/_l hold ONE stat per head (sp >= 1 fits).
    if (_mat_decode) {
      // f16 scratch (2 bytes/elt) -- halves the scores DRAM round-trip vs f32.
      _dec_tstride = _ctx->max_pages() * c.page_tokens;
      _d_dec_scores = _mc->make_shared_buffer(
          Hq * (std::size_t)_dec_tstride * 2u);
    }
  }
  _scratch_ready = !_d_logits.empty();
  return _scratch_ready;
}

ContextId
MetalGemmaModel::cm_for_(ContextId lm_cid)
{
  auto it = _ctxmap.find(lm_cid.v);
  if (it != _ctxmap.end()) { return it->second; }
  ContextId cm = _ctx->acquire_root();
  if (!cm.valid()) { return ContextId{}; }
  _ctxmap.emplace(lm_cid.v, cm);
  return cm;
}

void
MetalGemmaModel::encode_step_(ComputeEncoder& enc, ContextId cid, int kv_off,
                              const SharedBuffer* tok_src, std::size_t tok_off,
                              std::size_t pgtab_off)
{
  const Config& c = _cfg;
  const int H = c.hidden, Hq = c.n_heads;
  const bool has_ple = c.hpli > 0;
  const int hpli = c.hpli, ple = c.n_layers * hpli;
  const float eps = c.rms_eps;
  // `cid` is this model's ContextManager context (resolved by the caller
  // via cm_for_). kv_k/kv_v follow per-layer kv_source automatically.
  // Input-token source for the embedding gather: `_d_tok` (host-filled) by
  // default, or a GPU-resident token buffer at `tok_off` (pdecode chain).
  const SharedBuffer& tokbuf = tok_src ? *tok_src : _d_tok;
  const std::size_t   toff   = tok_src ? tok_off : 0;

  auto rms = [&](const SharedBuffer& xin, const SharedBuffer& w,
                 const SharedBuffer& y, int R, int Hd) {
    enc.set_function((_rms_fast || Hd > 4096) ? _fn_rms_fast : _fn_rms);
    enc.set_buffer(0, xin);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y);
    enc.set_constant(3, Hd);
    enc.set_constant(4, eps);
    enc.dispatch({512, (unsigned)R, 1}, {512, 1, 1});
  };
  // Fused sublayer-out norm + residual add (+ post-scale): res = (res +
  // rms(xin, w)) * post_scale. One dispatch for the Gemma sandwich
  // `rms(out); h += out;` (and the layer_scalar at the PLE tail).
  auto rms_add = [&](const SharedBuffer& xin, const SharedBuffer& w,
                     const SharedBuffer& res, int R, int Hd,
                     float post_scale = 1.0f) {
    enc.set_function((_rms_fast || Hd > 4096) ? _fn_rms_add_fast : _fn_rms_add);
    enc.set_buffer(0, xin);
    enc.set_buffer(1, w);
    enc.set_buffer(2, res);
    enc.set_buffer(3, res);
    enc.set_constant(4, Hd);
    enc.set_constant(5, eps);
    enc.set_constant(6, post_scale);
    enc.dispatch({512, (unsigned)R, 1}, {512, 1, 1});
  };
  // qmv_fn selects the kernel (base bits for attn / lm_head; the _mlp
  // variant -- 8-bit for gemma4_unified -- for down_proj). `qmv` defaults
  // to the base kernel.
  auto qmv_fn = [&](const auto& fn, const SharedBuffer& w,
                    const SharedBuffer& s, const SharedBuffer& b,
                    const SharedBuffer& xin, const SharedBuffer& y,
                    int Kk, int N) {
    enc.set_function(fn);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto qmv = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, const SharedBuffer& xin,
                 const SharedBuffer& y, int Kk, int N) {
    qmv_fn(_fn_qmv, w, s, b, xin, y, Kk, N);
  };
  // QKV-fused decode GEMV: one dispatch over the q|k|v row-concat (full GPU
  // occupancy), writing q/k/v into their separate buffers. v_out is bound to
  // _d_k as a never-written dummy when vrows==0 (k_eq_v q|k fusion).
  auto qmv_qkv = [&](const Layer& ly, const SharedBuffer& xin) {
    enc.set_function(_fn_qmv_qkv);
    enc.set_buffer(0, ly.qkvw);
    enc.set_buffer(1, ly.qkvs);
    enc.set_buffer(2, ly.qkvb);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, _d_q);
    enc.set_buffer(5, _d_k);
    enc.set_buffer(6, ly.qkv_vrows ? _d_v : _d_k);
    enc.set_constant(7, H);
    enc.set_constant(8, ly.qkv_qrows);
    enc.set_constant(9, ly.qkv_krows);
    enc.set_constant(10, ly.qkv_vrows);
    const int N = ly.qkv_qrows + ly.qkv_krows + ly.qkv_vrows;
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  // Fused input_layernorm + QKV GEMV: reads the residual stream _d_x directly,
  // RMS-norms it in threadgroup memory (norm_w = in_ln, (1+w) already baked at
  // load), then projects -- saving the standalone rms dispatch + _d_hn round-
  // trip. tg_x sized to the block-aligned hidden dim. block_size = 16*32 = 512.
  auto qmv_qkv_rms = [&](const Layer& ly, const SharedBuffer& xres) {
    enc.set_function(_fn_qmv_qkv_rms);
    enc.set_buffer(0, ly.qkvw);
    enc.set_buffer(1, ly.qkvs);
    enc.set_buffer(2, ly.qkvb);
    enc.set_buffer(3, xres);
    enc.set_buffer(4, ly.in_ln);
    enc.set_buffer(5, _d_q);
    enc.set_buffer(6, _d_k);
    enc.set_buffer(7, ly.qkv_vrows ? _d_v : _d_k);
    enc.set_constant(8, H);
    enc.set_constant(9, ly.qkv_qrows);
    enc.set_constant(10, ly.qkv_krows);
    enc.set_constant(11, ly.qkv_vrows);
    enc.set_constant(12, eps);
    const int blk = 512;
    const int aligned = ((H + blk - 1) / blk) * blk;
    enc.set_threadgroup_memory_length(0, (unsigned)aligned * 2);  // f16/bf16
    const int N = ly.qkv_qrows + ly.qkv_krows + ly.qkv_vrows;
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  // Dense f16 GEMV (M=1): y[1,N] = x[1,K] @ W[N,K]^T. The PLE's two f16
  // projections; same {32,N/4,1}/{32,2,1} grid as qmv (RPS=4/NSG=2).
  auto dense_gemv = [&](const SharedBuffer& xin, const SharedBuffer& w,
                        const SharedBuffer& y, int Kk, int N) {
    enc.set_function(_fn_dense_gemv);
    enc.set_buffer(0, xin);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y);
    enc.set_constant(3, Kk);
    enc.set_constant(4, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  // Fused per-head q/k norm + rope (one dispatch). v_norm has no rope so
  // keeps the plain rms.
  auto rms_rope = [&](const SharedBuffer& xb, const SharedBuffer& w,
                      const SharedBuffer& invf, int heads, int D) {
    enc.set_function(_fn_rms_rope);
    enc.set_buffer(0, xb);
    enc.set_buffer(1, w);
    enc.set_buffer(2, invf);
    enc.set_constant(3, heads);
    enc.set_constant(4, D);
    enc.set_constant(5, eps);
    enc.set_constant(6, kv_off);
    enc.dispatch({256, (unsigned)heads, 1}, {256, 1, 1});
  };
  // Fused Q+K norm+rope (one dispatch instead of two). Hq q-heads then Hkv
  // k-heads, same D/invf/offset.
  auto rms_rope2 = [&](const SharedBuffer& q, const SharedBuffer& qw,
                       const SharedBuffer& k, const SharedBuffer& kw,
                       const SharedBuffer& invf, int Hq_, int Hkv_, int D) {
    enc.set_function(_fn_rms_rope2);
    enc.set_buffer(0, q);
    enc.set_buffer(1, qw);
    enc.set_buffer(2, k);
    enc.set_buffer(3, kw);
    enc.set_buffer(4, invf);
    enc.set_constant(5, Hq_);
    enc.set_constant(6, D);
    enc.set_constant(7, eps);
    enc.set_constant(8, kv_off);
    enc.dispatch({256, (unsigned)(Hq_ + Hkv_), 1}, {256, 1, 1});
  };
  // Fused Q+K+V norm(+rope): also folds the V rms-norm (no rope). Only for
  // NON-k_eq_v layers (where V is independent of K).
  auto rms_rope3 = [&](const SharedBuffer& q, const SharedBuffer& qw,
                       const SharedBuffer& k, const SharedBuffer& kw,
                       const SharedBuffer& v, const SharedBuffer& vw,
                       const SharedBuffer& invf, int Hq_, int Hkv_, int D) {
    enc.set_function(_fn_rms_rope3);
    enc.set_buffer(0, q);
    enc.set_buffer(1, qw);
    enc.set_buffer(2, k);
    enc.set_buffer(3, kw);
    enc.set_buffer(4, v);
    enc.set_buffer(5, vw);
    enc.set_buffer(6, invf);
    enc.set_constant(7, Hq_);
    enc.set_constant(8, Hkv_);
    enc.set_constant(9, D);
    enc.set_constant(10, eps);
    enc.set_constant(11, kv_off);
    enc.dispatch({256, (unsigned)(Hq_ + 2 * Hkv_), 1}, {256, 1, 1});
  };
  // Fused rms_rope3 + ring KV-write: q normed+roped in place, k/v written
  // straight into the contiguous ring cache (ck/cv) at slot kv_off%cap --
  // folds the kv_write2 dispatch into the norm kernel. Ring (sliding) only.
  auto rms_rope3_kvwrite =
      [&](const SharedBuffer& q, const SharedBuffer& qw,
          const SharedBuffer& k, const SharedBuffer& kw,
          const SharedBuffer& v, const SharedBuffer& vw,
          const SharedBuffer& invf, int Hq_, int Hkv_, int D,
          const SharedBuffer& ck, const SharedBuffer& cv, int cap_,
          int ring_cap_, int window_) {
    enc.set_function(_fn_rms_rope3_kvwrite);
    enc.set_buffer(0, q);
    enc.set_buffer(1, qw);
    enc.set_buffer(2, k);
    enc.set_buffer(3, kw);
    enc.set_buffer(4, v);
    enc.set_buffer(5, vw);
    enc.set_buffer(6, invf);
    enc.set_constant(7, Hq_);
    enc.set_constant(8, Hkv_);
    enc.set_constant(9, D);
    enc.set_constant(10, eps);
    enc.set_constant(11, kv_off);
    enc.set_buffer(12, ck);
    enc.set_buffer(13, cv);
    enc.set_constant(14, cap_);
    enc.set_constant(15, kv_off);
    enc.set_constant(16, ring_cap_);     // ring modulo (0 = linear, no mirror)
    enc.set_constant(17, window_);       // trailing window (mirror tail size+1)
    enc.dispatch({256, (unsigned)(Hq_ + 2 * Hkv_), 1}, {256, 1, 1});
  };
  auto geglu = [&](const SharedBuffer& gate, const SharedBuffer& up,
                   const SharedBuffer& out, int n, std::size_t up_off = 0) {
    enc.set_function(_fn_geglu);
    enc.set_buffer(0, gate);
    enc.set_buffer(1, up, up_off);
    enc.set_buffer(2, out);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                      const SharedBuffer& out, int n) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a);
    enc.set_buffer(1, bb);
    enc.set_buffer(2, out);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto scale = [&](const SharedBuffer& x, int n, float s) {
    enc.set_function(_fn_scale);
    enc.set_buffer(0, x);
    enc.set_constant(1, n);
    enc.set_constant(2, s);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto copy = [&](const SharedBuffer& src, const SharedBuffer& dst, int n) {
    enc.set_function(_fn_copy);
    enc.set_buffer(0, src);
    enc.set_buffer(1, dst);
    const int zoff = 0;
    enc.set_constant(2, zoff);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto embed_gather = [&](const SharedBuffer& w, const SharedBuffer& s,
                          const SharedBuffer& b, const SharedBuffer& out,
                          int Hd) {
    enc.set_function(_fn_embed);
    enc.set_buffer(0, tokbuf, toff);
    enc.set_buffer(1, w);
    enc.set_buffer(2, s);
    enc.set_buffer(3, b);
    enc.set_buffer(4, out);
    enc.set_constant(5, Hd);
    enc.dispatch({(unsigned)Hd, 1, 1}, {256, 1, 1});
  };
  // Native Q6_K gather/lm_head (GGUF path). `ids`[ntok] -> out[ntok, Hd];
  // lm_head GEMV reads the raw Q6_K table directly (no DRAM dequant).
  auto embed_q6k = [&](const SharedBuffer& ids, std::size_t ioff,
                       const SharedBuffer& out, int Hd, int ntok) {
    enc.set_function(_fn_embed_q6k);
    enc.set_buffer(0, ids, ioff);
    enc.set_buffer(1, _embed_q6k);
    enc.set_buffer(2, out);
    enc.set_constant(3, Hd);
    enc.dispatch({(unsigned)Hd, (unsigned)ntok, 1}, {256, 1, 1});
  };
  auto lm_head_q6k = [&](const SharedBuffer& xin, const SharedBuffer& y,
                         std::size_t xoff, int Kk, int Nv) {
    enc.set_function(_fn_qmv_q6k);
    enc.set_buffer(0, _embed_q6k);
    enc.set_buffer(1, xin, xoff);
    enc.set_buffer(2, y);
    enc.set_constant(3, Kk);
    enc.set_constant(4, Nv);
    enc.dispatch({32, (unsigned)(((Nv + 7) / 8) * 2), 1}, {32, 2, 1});
  };

  // Decode category profiler: DUP(cat, fn) runs fn() once in production
  // (dup<0, zero cost, no getenv), twice when VPIPE_GEMMA_DUP_CAT names that
  // category. The bench diffs decode time vs the `none` baseline -> the delta
  // is that category's GPU cost. Duplicating an accumulating op (rms_add,
  // kvw) corrupts the logits but not the per-op work/shapes, so the TIMING is
  // unaffected (profiling never asserts token-exact). Only a profiling
  // session (VPIPE_GEMMA_CATPROF set at load) reads the per-step env.
  enum { DC_PROJ = 1, DC_FFN = 2, DC_LMHEAD = 3, DC_ATTN = 4, DC_NORM = 5,
         DC_PLE = 6, DC_MISC = 7, DC_ATTN_GLOBAL = 8, DC_ATTN_SLIDE = 9 };
  int dup = -1;
  if (_catprof) {
    if (const char* e = std::getenv("VPIPE_GEMMA_DUP_CAT")) {
      const std::string s = e;
      dup = (s == "proj") ? 1 : (s == "ffn") ? 2 : (s == "lmhead") ? 3
          : (s == "attn") ? 4 : (s == "norm") ? 5 : (s == "ple") ? 6
          : (s == "misc") ? 7 : (s == "attn_global") ? 8
          : (s == "attn_slide") ? 9 : 0;
    }
  }
  auto DUP = [&](int cat, auto&& fn) { fn(); if (dup == cat) { fn(); } };

  // Per-layer-TYPE whole-layer doubling (gpu_active profiling): unlike DUP
  // (per category), this repeats a matching layer's ENTIRE body, so the
  // gpu_active delta vs the `none` baseline = that type's full GPU residency
  // INCLUDING its uncategorized within-layer ops + inter-dispatch bubbles.
  // Buckets by (is_full, owns-own-KV); 5 = lm_head. Timing only (corrupts the
  // running hidden state, same as DUP). VPIPE_GEMMA_DUP_LAYER, _catprof-gated.
  int dup_layer = 0;
  if (_catprof) {
    if (const char* e = std::getenv("VPIPE_GEMMA_DUP_LAYER")) {
      const std::string s = e;
      dup_layer = (s == "slide_own") ? 1 : (s == "global_own") ? 2
                : (s == "slide_skip") ? 3 : (s == "global_skip") ? 4
                : (s == "lmhead") ? 5 : 0;
    }
  }
  // Whole-category STRIP (ablation): skip a layer category's entire body so the
  // baseline-minus-stripped gpu_active delta = that category's GPU cost. Mirror
  // of dup_layer but lreps=0 (remove instead of double). Timing only (corrupts
  // the hidden state). Individual buckets + the user-facing combos:
  //   sliding|global (by attn type), shared_kv|own_kv (by KV ownership).
  // VPIPE_GEMMA_SKIP_LAYER, _catprof-gated.
  int skip_layer = 0;
  if (_catprof) {
    if (const char* e = std::getenv("VPIPE_GEMMA_SKIP_LAYER")) {
      const std::string s = e;
      skip_layer = (s == "slide_own") ? 1 : (s == "global_own") ? 2
                 : (s == "slide_skip") ? 3 : (s == "global_skip") ? 4
                 : (s == "sliding") ? 5 : (s == "global") ? 6
                 : (s == "shared_kv") ? 7 : (s == "own_kv") ? 8
                 : (s == "all") ? 9 : 0;
    }
  }
  // PLE strip (composable with SKIP_LAYER): drop the per-layer-embedding
  // production AND every layer's per_layer_input gate (the layer tail falls
  // back to just *= layer_scalar). Isolates PLE's wall cost, e.g. in an
  // all-global run (SKIP_LAYER=sliding + SKIP_PLE). VPIPE_GEMMA_SKIP_PLE.
  const bool do_ple =
      has_ple && !(_catprof && std::getenv("VPIPE_GEMMA_SKIP_PLE") != nullptr);
  // FFN strip (composable): drop each layer's gate/up + down GEMVs (the MLP
  // matmuls), keeping the pre/post norms. Isolates the attention path once PLE
  // and FFN -- both vpipe-favourable -- are removed. VPIPE_GEMMA_SKIP_FFN.
  const bool skip_ffn =
      _catprof && std::getenv("VPIPE_GEMMA_SKIP_FFN") != nullptr;
  // Attention-vs-projection split inside a layer (composable). SKIP_ATTN drops
  // the SDPA flash-decode core (keeps QKV/O proj, rope, KV-write); SKIP_PROJ
  // drops the QKV + O projection GEMVs (keeps SDPA). Isolates which carries the
  // sliding-layer deficit. VPIPE_GEMMA_SKIP_ATTN / VPIPE_GEMMA_SKIP_PROJ.
  const bool skip_attn =
      _catprof && std::getenv("VPIPE_GEMMA_SKIP_ATTN") != nullptr;
  const bool skip_proj =
      _catprof && std::getenv("VPIPE_GEMMA_SKIP_PROJ") != nullptr;
  // Fixed-tail split (composable): SKIP_LMHEAD drops the final lm_head GEMV
  // (262144-vocab, the bandwidth giant); SKIP_EMBED drops the input embed
  // gather. Isolate the lm_head vs embed vs argmax shares of the fixed bucket.
  // VPIPE_GEMMA_SKIP_LMHEAD / VPIPE_GEMMA_SKIP_EMBED.
  const bool skip_lmhead =
      _catprof && std::getenv("VPIPE_GEMMA_SKIP_LMHEAD") != nullptr;
  const bool skip_embed =
      _catprof && std::getenv("VPIPE_GEMMA_SKIP_EMBED") != nullptr;

  // ---- embeddings + per-layer inputs (once per token) -------------
  if (skip_embed) {
    // embed gather stripped (timing-only): _d_x left stale
  } else if (_embed_is_q6k) { embed_q6k(tokbuf, toff, _d_x, H, 1); }
  else { embed_gather(_embed_w, _embed_s, _embed_b, _d_x, H); }
  scale(_d_x, H, std::sqrt((float)H));                     // embed_scale
  // Diagnostic: fire N extra no-op dispatches/token to measure the per-launch
  // GPU idle (decode is a dependent-dispatch chain; this isolates the idle
  // cost of dispatch COUNT). VPIPE_GEMMA_DUMMY_DISP=N. Off in production.
  {
    static const int kDummy = std::getenv("VPIPE_GEMMA_DUMMY_DISP")
        ? std::atoi(std::getenv("VPIPE_GEMMA_DUMMY_DISP")) : 0;
    const int one = 1;
    for (int i = 0; i < kDummy; ++i) {
      // Realistic per-dispatch load: 8 read buffers + 4 constants + 1 write.
      // b0 == out == _d_dummy makes the dummies a RAW dependency chain (they
      // serialize like the real decode chain instead of overlapping). Reads
      // _d_x (live hidden state) for the filler args; never touches real state.
      enc.set_function(_fn_dummy_disp);
      enc.set_buffer(0, _d_dummy);                 // chain in (== out)
      enc.set_buffer(1, _d_x); enc.set_buffer(2, _d_x); enc.set_buffer(3, _d_x);
      enc.set_buffer(4, _d_x); enc.set_buffer(5, _d_x); enc.set_buffer(6, _d_x);
      enc.set_buffer(7, _d_x);
      enc.set_buffer(8, _d_dummy);                 // chain out
      enc.set_constant(9, one);  enc.set_constant(10, one);
      enc.set_constant(11, one); enc.set_constant(12, one);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
    }
  }
  // Per-Layer-Input projection (e4b PLE only; gemma4_unified has no PLE).
  if (do_ple && _ple_chunk_k > 0 && _ple_gemv) {
    // Pipelined PLE (Design A): produce pli in per-chunk buffers so chunks
    // 1..N-1 overlap the layer chain (only chunk 0 gates layer 0). The chunk
    // GEMVs read a PRIVATE copy of _d_x (no WAR with the per-layer residual
    // writes), and each chunk writes its OWN buffer (no cross-chunk hazard,
    // no whole-pli barrier at layer 0's gate). Token-identical to the block.
    DUP(DC_PLE, [&] {
    copy(_d_x, _d_x_ple, H);
    embed_gather(_ple_w, _ple_s, _ple_b, _d_ple, ple);
    scale(_d_ple, ple, std::sqrt((float)hpli));
    const int nch = (int)_d_pli_ch.size();
    for (int ci = 0; ci < nch; ++ci) {
      const int a = ci * _ple_chunk_k;
      const int nlc = std::min(a + _ple_chunk_k, c.n_layers) - a;
      const int rows = nlc * hpli;
      const std::size_t woff = (std::size_t)a * hpli * (std::size_t)H * 2;
      dense_gemv(_d_x_ple,
                 _plm_proj_w.subview(woff, (std::size_t)rows * (std::size_t)H * 2),
                 _d_pleproj_ch[ci], H, rows);
      scale(_d_pleproj_ch[ci], rows, std::pow((float)H, -0.5f));
      rms(_d_pleproj_ch[ci], _ple_proj_norm, _d_pleproj_ch[ci], nlc, hpli);
      residual(_d_pleproj_ch[ci],
               _d_ple.subview((std::size_t)a * hpli * 2, (std::size_t)rows * 2),
               _d_pli_ch[ci], rows);
      scale(_d_pli_ch[ci], rows, 0.70710678f);
    }
    });
  } else if (do_ple) {
    DUP(DC_PLE, [&] {
    embed_gather(_ple_w, _ple_s, _ple_b, _d_ple, ple);
    scale(_d_ple, ple, std::sqrt((float)hpli));            // ple scale
    // per_layer_model_projection proj[1,ple]=x[1,H]@W[ple,H]^T (BF16 weight).
    // M=1 f16 GEMV (full-bandwidth) by default; dense_t GEMM for A/B.
    if (_ple_gemv) {
      dense_gemv(_d_x, _plm_proj_w, _d_pleproj, H, ple);
    } else {
      enc.set_function(_fn_dense_t);
      enc.set_buffer(0, _d_x);
      enc.set_buffer(1, _plm_proj_w);
      enc.set_buffer(2, _plm_proj_w);   // bias unused (has_bias=0)
      enc.set_buffer(3, _d_pleproj);
      enc.set_constant(4, H);            // K
      enc.set_constant(5, ple);          // N
      const int M = 1;
      enc.set_constant(6, M);
      const int has_bias = 0;
      enc.set_constant(7, has_bias);
      enc.dispatch({(unsigned)(((ple + 31) / 32) * 32), 2, 2}, {32, 2, 2});
    }
    scale(_d_pleproj, ple, std::pow((float)H, -0.5f));      // proj scale
    rms(_d_pleproj, _ple_proj_norm, _d_pleproj, c.n_layers, hpli);
    residual(_d_pleproj, _d_ple, _d_pli, ple);
    scale(_d_pli, ple, 0.70710678f);                       // (proj+ple)*2^-0.5
    });
  }

  // FULL (global) layers are PAGED: claim this token's page slot ONCE (shared
  // by every full layer -- they advance in lockstep) and build the page table
  // for the paged attention. append() advances seq_len, so the caller must NOT
  // kv_append. Sliding layers keep the contiguous ring (positions from kv_off).
  const int page_tokens = _ctx->page_tokens();
  ContextManager::AppendSlot fslot;
  std::size_t fpoff = 0;
  int n_pages = 0;
  if (_full_paged) {
    fslot = _ctx->append(cid, 1);
    if (!fslot.valid()) {
      return;                       // page pool exhausted (soft fail)
    }
    fpoff = (std::size_t)fslot.page_id.v * _ctx->page_stride_bytes();
    n_pages = _ctx->fill_page_table(cid, reinterpret_cast<std::int32_t*>(
        static_cast<std::uint8_t*>(_d_pgtab.contents()) + pgtab_off));
  } else {
    _ctx->kv_append(cid, 1);
  }

  // ---- layer loop -------------------------------------------------
  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    // Whole-layer doubling for per-type gpu_active profiling (see dup_layer).
    const bool owns_kv = ly.kv_source < 0;
    int lreps = 1;
    if      (dup_layer == 1 && !ly.is_full &&  owns_kv) { lreps = 2; }
    else if (dup_layer == 2 &&  ly.is_full &&  owns_kv) { lreps = 2; }
    else if (dup_layer == 3 && !ly.is_full && !owns_kv) { lreps = 2; }
    else if (dup_layer == 4 &&  ly.is_full && !owns_kv) { lreps = 2; }
    if      (skip_layer == 1 && !ly.is_full &&  owns_kv) { lreps = 0; }
    else if (skip_layer == 2 &&  ly.is_full &&  owns_kv) { lreps = 0; }
    else if (skip_layer == 3 && !ly.is_full && !owns_kv) { lreps = 0; }
    else if (skip_layer == 4 &&  ly.is_full && !owns_kv) { lreps = 0; }
    else if (skip_layer == 5 && !ly.is_full)             { lreps = 0; }
    else if (skip_layer == 6 &&  ly.is_full)             { lreps = 0; }
    else if (skip_layer == 7 && !owns_kv)                { lreps = 0; }
    else if (skip_layer == 8 &&  owns_kv)                { lreps = 0; }
    else if (skip_layer == 9)                            { lreps = 0; }
    for (int lrep = 0; lrep < lreps; ++lrep) {
    const int D = ly.head_dim;
    const int Hkv = ly.n_kv;
    const int qd = Hq * D, kd = Hkv * D;
    const SharedBuffer& invf = ly.is_full ? _inv_freq_full : _inv_freq_sliding;
    const bool paged_full = ly.is_full && _full_paged;

    // Attention (input_norm -> attn -> post_attn_norm -> residual). The
    // input_norm RMS is fused into the QKV GEMV (qmv_qkv_rms reads _d_x
    // directly) when QKV is fused; the standalone rms stays for the fallback
    // and the skip_proj ablation (where QKV is removed but the norm is not).
    const bool fuse_rms = _qkv_rms_fuse && ly.qkv_fused && !skip_proj;
    if (!fuse_rms) {
      DUP(DC_NORM, [&] { rms(_d_x, ly.in_ln, _d_hn, 1, H); });
    }
    // QKV-fused decode: one full-occupancy GEMV over the q|k|v row-concat
    // writes q/k/v into their separate buffers (the k/v halves alone
    // under-occupy the GPU). Falls back to the per-proj GEMV (q here, k/v
    // below) when not fused (shared layers, non-4-bit, or disabled).
    if (skip_proj) {
      // projection stripped: leave _d_q/_d_k/_d_v stale (timing-only)
    } else if (fuse_rms) {
      DUP(DC_PROJ, [&] { qmv_qkv_rms(ly, _d_x); });
    } else if (ly.qkv_fused) {
      DUP(DC_PROJ, [&] { qmv_qkv(ly, _d_hn); });
    } else {
      DUP(DC_PROJ, [&] { qmv(ly.qw, ly.qs, ly.qb, _d_hn, _d_q, H, qd); });
    }
    // q_norm+rope is fused with k_norm+rope below (rms_rope2) for non-reuse
    // layers; reuse layers (no k_proj) do q alone in the else branch.

    // Full layers: Kuse/Vuse are the SHARED page pool (kpool/vpool follow
    // kv_source). Sliding layers: the per-context contiguous ring (kv_k/kv_v).
    const SharedBuffer* Kuse =
        paged_full ? _ctx->kpool(L) : _ctx->kv_k(cid, L);
    const SharedBuffer* Vuse =
        paged_full ? _ctx->vpool(L) : _ctx->kv_v(cid, L);
    // Physical K/V capacity (== max_seq for full layers; the bounded ring
    // size for sliding layers, or this context's GROWN ring) and the ring
    // modulo (0 = linear). Per-context: honors ensure_sliding_capacity.
    // Unused for paged-full layers (they address by page table).
    const int cap = paged_full ? 0 : _ctx->kv_capacity(cid, L);
    const int ring_cap = paged_full ? 0 : _ctx->kv_ring_cap(cid, L);
    // One-shot per-layer structural dump (VPIPE_GEMMA_LAYER_DUMP): verify the
    // right type/head_dim/window/KV-source/scan-length is applied per layer.
    if (_catprof || std::getenv("VPIPE_GEMMA_LAYER_DUMP")) {
      static int dumped = 0;
      // Fire deep in the decode (kv_off large) so the dump reflects the real
      // long-context scan, not an early warmup step.
      if (std::getenv("VPIPE_GEMMA_LAYER_DUMP") && dumped < c.n_layers
          && kv_off >= 1024) {
        const int win = ly.is_full ? 0 : c.sliding_window;
        const int q_pos = kv_off;
        const int scan = (win > 0) ? std::min(win, q_pos + 1) : (q_pos + 1);
        std::fprintf(stderr,
            "[gemma-layer %2d] %s D=%d Hkv=%d k_eq_v=%d window=%d ring_cap=%d "
            "cap=%d kv_source=%d T_kv=%d scan=%d\n",
            L, ly.is_full ? "GLOBAL " : "sliding", ly.head_dim, ly.n_kv,
            (int)ly.k_eq_v, win, ring_cap, cap, ly.kv_source, kv_off + 1, scan);
        ++dumped;
      }
    }
    if (ly.kv_source < 0) {
      if (!ly.qkv_fused && !skip_proj) {         // k done in the fused GEMV
        DUP(DC_PROJ, [&] { qmv(ly.kw, ly.ks, ly.kb, _d_hn, _d_k, H, kd); });
      }
      // values: k_eq_v full layers reuse the k_proj output (no v_proj), taking
      // v_norm of it BEFORE k is normed -> can't fold V into the q+k+v kernel
      // (it norms k in place); use rms_rope2(q,k) + a separate v_norm. The
      // common non-k_eq_v case projects v independently, so v folds into one
      // rms_rope3(q,k,v) -- q/k normed+roped, v normed (no rope).
      if (ly.k_eq_v) {
        DUP(DC_NORM, [&] { rms(_d_k, _ones_vnorm, _d_v, Hkv, D); });  // v=v_norm(k)
        DUP(DC_NORM, [&] {
          rms_rope2(_d_q, ly.q_norm, _d_k, ly.k_norm, invf, Hq, Hkv, D);
        });
      }
      // Fuse the rope3 + ring KV-write into ONE dispatch for the common
      // sliding/ring non-k_eq_v case (k/v normed+roped straight into the
      // cache). Paged-global and k_eq_v keep the split path below.
      const bool fuse_rope_kv =
          _rope_kv_fuse && !ly.k_eq_v && !paged_full;
      if (!ly.k_eq_v && !fuse_rope_kv) {
        if (!ly.qkv_fused && !skip_proj) {       // v done in the fused GEMV
          DUP(DC_PROJ, [&] { qmv(ly.vw, ly.vs, ly.vb, _d_hn, _d_v, H, kd); });
        }
        DUP(DC_NORM, [&] {
          rms_rope3(_d_q, ly.q_norm, _d_k, ly.k_norm, _d_v, _ones_vnorm,
                    invf, Hq, Hkv, D);
        });
      } else if (fuse_rope_kv) {
        if (!ly.qkv_fused && !skip_proj) {       // v done in the fused GEMV
          DUP(DC_PROJ, [&] { qmv(ly.vw, ly.vs, ly.vb, _d_hn, _d_v, H, kd); });
        }
        DUP(DC_NORM, [&] {
          const int kvw_win = ly.is_full ? 0 : c.sliding_window;
          rms_rope3_kvwrite(_d_q, ly.q_norm, _d_k, ly.k_norm, _d_v,
                            _ones_vnorm, invf, Hq, Hkv, D, *Kuse, *Vuse, cap,
                            ring_cap, kvw_win);
        });
      }
      if (fuse_rope_kv) {
        // KV-write folded into rms_rope3_kvwrite above; nothing to do.
      } else if (paged_full) {
        // Paged write: this token's K and V into the page slot (one dispatch
        // each -- the fused kv_write2 is contiguous-only). Kuse/Vuse are the
        // pool; bind at the page byte base + slot.
        DUP(DC_MISC, [&] {
          auto wp = [&](const SharedBuffer& src, const SharedBuffer& pool) {
            enc.set_function(_fn_kv_write_paged);
            enc.set_buffer(0, src);
            enc.set_buffer(1, pool, fpoff);
            enc.set_constant(2, page_tokens);
            enc.set_constant(3, D);
            const int one = 1, zero = 0;
            enc.set_constant(4, one);                  // n_src
            enc.set_constant(5, zero);                 // src_off
            enc.set_constant(6, fslot.slot_offset);    // dst_slot
            enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
          };
          wp(_d_k, *Kuse);
          wp(_d_v, *Vuse);
        });
      } else {
      // Fused K+V write into the [Hkv, cap, D] caches at slot kv_off % cap --
      // one dispatch instead of two (cuts the dependent-dispatch idle).
      DUP(DC_MISC, [&] {
        enc.set_function(_fn_kv_write2);
        enc.set_buffer(0, _d_k);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, _d_v);
        enc.set_buffer(3, *Vuse);
        enc.set_constant(4, cap);
        enc.set_constant(5, D);
        const int one = 1;
        enc.set_constant(6, one);
        enc.set_constant(7, kv_off);
        const int kvw_win = ly.is_full ? 0 : c.sliding_window;
        enc.set_constant(8, ring_cap);     // ring modulo (0 = linear)
        enc.set_constant(9, kvw_win);      // trailing window (mirror)
        enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
      });
      }
    } else {
      // Reuse layers reuse a prior layer's K/V, so only Q is normed+roped.
      DUP(DC_NORM, [&] { rms_rope(_d_q, ly.q_norm, invf, Hq, D); });
    }

    // SDPA, scale 1.0 (q_norm/k_norm absorb 1/sqrt(D)).
    const float one_scale = 1.0f;
    const int T_kv = kv_off + 1;
    const int nq = 1;
    // Multi-simdgroup decode attention: BN = 4096/D simdgroups (16@256,
    // 8@512) split the KV-key scan (vs the 1-simdgroup scalar kernel),
    // recovering ~BN x parallelism -- the scalar SDPA was ~4.6x MLX at
    // decode (4.58->0.52 ms/tok). window<=0 == full causal.
    // Profiler: attn_global / attn_slide isolate the two layer types; attn
    // doubles both. (DUP can only match one category, so this is manual.)
    const int attn_cat = ly.is_full ? DC_ATTN_GLOBAL : DC_ATTN_SLIDE;
    const int attn_runs = skip_attn ? 0
                        : (dup == attn_cat || dup == DC_ATTN) ? 2 : 1;
    for (int ar = 0; ar < attn_runs; ++ar) {
      if (paged_full) {
        // Paged decode attention (n_q=1) over the shared pool + page table.
        // Fast path: KV-split GQA flash-decode (read each KV head once for all
        // G q-heads, split the scan across grid.z) + merge -- the long-context
        // full-layer decode. Fallback: the scalar paged kernel (no split).
        const int G = (Hkv > 0) ? Hq / Hkv : 0;
        const bool use_mat = _mat_decode && Hkv > 0 && (Hq % Hkv == 0)
            && G >= 1 && (D % 128 == 0) && !_d_dec_scores.empty()
            && (kv_off + 1) <= _dec_tstride;
        const bool use_pgqa = !use_mat && _pgqa_attn && Hkv > 0
            && (Hq % Hkv == 0) && G >= 1 && G <= 16 && (D % 128 == 0)
            && !_d_gqa_oacc.empty();
        if (use_mat) {
          // Materialized decode (omlx/MLX D=512 fallback): QK GEMV -> rowstat
          // -> PV GEMV -> merge. Split the QK/PV key scan for occupancy.
          const int scan = kv_off + 1;
          int sp = _gtile_split;
          if (_gtile_kps > 0) {
            sp = (scan + _gtile_kps - 1) / _gtile_kps;
            sp = std::max(1, std::min(sp, _gtile_split));
          }
          // K1: QK GEMV -> scores[Hq, Tstride].
          enc.set_function(_fn_dec_qk);
          enc.set_buffer(0, _d_q);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, _d_dec_scores);
          enc.set_constant(3, one_scale);
          enc.set_constant(4, D);
          enc.set_constant(5, Hq);
          enc.set_constant(6, Hkv);
          enc.set_constant(7, kv_off);        // q_offset
          enc.set_constant(8, page_tokens);
          enc.set_constant(9, n_pages);
          enc.set_buffer(10, _d_pgtab, pgtab_off);
          enc.set_constant(11, _dec_tstride);
          enc.set_constant(12, sp);
          enc.dispatch({32, (unsigned)Hq, (unsigned)sp}, {32, 1, 1});
          // K2: per-head max + sum exp.
          enc.set_function(_fn_dec_rowstat);
          enc.set_buffer(0, _d_dec_scores);
          enc.set_buffer(1, _d_gqa_m);
          enc.set_buffer(2, _d_gqa_l);
          enc.set_constant(3, T_kv);
          enc.set_constant(4, _dec_tstride);
          enc.set_constant(5, Hq);
          enc.dispatch({256, (unsigned)Hq, 1}, {256, 1, 1});
          // K3: PV GEMV (reads K2's exp weights) -> partials[Hq, sp, D].
          enc.set_function(_fn_dec_pv);
          enc.set_buffer(0, _d_dec_scores);
          enc.set_buffer(1, *Vuse);
          enc.set_buffer(2, _d_gqa_oacc);
          enc.set_constant(3, D);
          enc.set_constant(4, Hq);
          enc.set_constant(5, Hkv);
          enc.set_constant(6, kv_off);        // q_offset
          enc.set_constant(7, page_tokens);
          enc.set_constant(8, n_pages);
          enc.set_buffer(9, _d_pgtab, pgtab_off);
          enc.set_constant(10, _dec_tstride);
          enc.set_constant(11, sp);
          enc.dispatch({32, (unsigned)Hq, (unsigned)sp}, {32, 1, 1});
          // K4: sum partials / row sum -> _d_attn.
          enc.set_function(_fn_dec_merge);
          enc.set_buffer(0, _d_gqa_oacc);
          enc.set_buffer(1, _d_gqa_l);
          enc.set_buffer(2, _d_attn);
          enc.set_constant(3, D);
          enc.set_constant(4, sp);
          enc.set_constant(5, Hq);
          enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
        } else if (use_pgqa) {
          // Size the split to the scan length so a short context isn't over-
          // split (mirrors the contiguous gtile adaptation).
          const int scan = kv_off + 1;
          int sp = _gtile_split;
          if (_gtile_kps > 0) {
            sp = (scan + _gtile_kps - 1) / _gtile_kps;
            sp = std::max(1, std::min(sp, _gtile_split));
          }
          enc.set_function(_fn_sdpa_pgqa);
          enc.set_buffer(0, _d_q);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, _d_gqa_oacc);
          enc.set_buffer(4, _d_gqa_m);
          enc.set_buffer(5, _d_gqa_l);
          enc.set_constant(6, one_scale);
          enc.set_constant(7, D);
          enc.set_constant(8, Hq);
          enc.set_constant(9, Hkv);
          enc.set_constant(10, kv_off);       // q_offset
          enc.set_constant(11, page_tokens);
          enc.set_constant(12, n_pages);
          enc.set_buffer(13, _d_pgtab, pgtab_off);
          enc.set_constant(14, sp);
          // vec kernel: grid (32, Hq, split), tg (32, G, 1) -- G simdgroups,
          // one per q-head, Hq/G = Hkv threadgroups in y.
          enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                       {32, (unsigned)(Hq / Hkv), 1});
          enc.set_function(_fn_sdpa_gqa_merge);
          enc.set_buffer(0, _d_gqa_oacc);
          enc.set_buffer(1, _d_gqa_m);
          enc.set_buffer(2, _d_gqa_l);
          enc.set_buffer(3, _d_attn);
          enc.set_constant(4, D);
          enc.set_constant(5, sp);
          enc.set_constant(6, Hq);
          enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
        } else {
          enc.set_function(_fn_sdpa_paged_causal);
          enc.set_buffer(0, _d_q);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, _d_attn);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, D);
          enc.set_constant(6, Hq);
          enc.set_constant(7, Hkv);
          enc.set_constant(8, nq);
          enc.set_constant(9, kv_off);        // q_offset
          enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_pages);
          enc.set_buffer(12, _d_pgtab, pgtab_off);
          enc.dispatch({32, (unsigned)Hq, (unsigned)nq}, {32, 1, 1});
        }
        continue;
      }
      const int window = ly.is_full ? 0 : c.sliding_window;
      // Flash-decode-GQA: read each KV head once for all G=Hq/Hkv query
      // heads (contiguous KV + window + ring). head_dim<=256, G<=4 layers
      // only; others stay on the per-q-head sdpa_mb below.
      const int G = (Hkv > 0) ? Hq / Hkv : 0;
      // Full layers only: sliding layers already scan just `window` keys, so
      // GQA's merge overhead outweighs its bandwidth saving there (net
      // regression); the win is on the full-context layers.
      const bool use_gqa = _gqa_attn && (window <= 0) && (D <= 256)
          && (D % 32 == 0) && Hkv > 0 && (Hq % Hkv == 0) && G >= 1 && G <= 4
          && !_d_gqa_oacc.empty();
      // Position-split flash-decode + merge (gtile family) for the GLOBAL
      // (full-context) layers, and -- when the fast vec kernel is active -- the
      // SLIDING layers too. G up to 16: the 12B gemma4_unified globals are
      // Hkv=1/Hq=16 (G=16), i.e. G simdgroups (16*32=512 threads) per
      // threadgroup, under the 1024-thread tg limit. Without this they fell to
      // sdpa_mb, which re-reads the lone KV head once PER q-head and only splits
      // the scan across 8 in-tg simdgroups (no grid.z split -> grows w/ depth).
      // The vec kernel (UK-unrolled + vec4 loads, ~2x direct, bit-identical)
      // ALSO beats sdpa_mb on the window-capped sliding layers (~1.3-1.8x); the
      // older direct/staged were net-neutral there, so sliding stays on sdpa_mb
      // unless vec is the active kernel.
      const bool gtile_vec = !_gtile_staged && !_gtile_direct
          && _fn_sdpa_gqa_vec.valid() && (D % 128 == 0);
      const bool use_gtile = _gtile_attn && (D <= 512) && (D % 32 == 0)
          && Hkv > 0 && (Hq % Hkv == 0) && G >= 1 && G <= 16
          && !_d_gqa_oacc.empty() && (window <= 0 || gtile_vec);
      // MMA q-head flash-decode for the GLOBAL layers (8 q-heads/tile share one
      // kv head -> matrix-core QK/PV, no per-key simd_sum; ~1.7-2x the vec
      // kernel + flatter with depth). Needs 8|G, D%64==0, window<=0, and the
      // C-block over-read in-bounds (kv_off+64<=cap). split scales with depth so
      // each split is ~one 64-key block.
      const bool use_mma_qh = use_gtile && _mma_qhead && (window <= 0)
          && (G % 8 == 0) && (D % 64 == 0) && (D <= 512)
          && (kv_off + 64 <= cap);
      if (use_mma_qh) {
        int sp = (kv_off + 1 + 63) / 64;
        sp = std::max(1, std::min(sp, _mma_qhead_cap));
        enc.set_function(_fn_sdpa_mma_qhead);
        enc.set_buffer(0, _d_q);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, _d_gqa_oacc);
        enc.set_buffer(4, _d_gqa_m);
        enc.set_buffer(5, _d_gqa_l);
        enc.set_constant(6, one_scale);
        enc.set_constant(7, T_kv);
        enc.set_constant(8, D);
        enc.set_constant(9, Hq);
        enc.set_constant(10, Hkv);
        enc.set_constant(11, kv_off);      // q_offset
        enc.set_constant(12, cap);         // kv_stride (head stride == cap)
        enc.set_constant(13, window);
        enc.set_constant(14, ring_cap);
        enc.set_constant(15, sp);
        // grid TOTAL threads: (256, Hq/8, sp), tg (256,1,1) = 8 simdgroups.
        enc.dispatch({256, (unsigned)(Hq / 8), (unsigned)sp}, {256, 1, 1});
        enc.set_function(_fn_sdpa_gqa_merge);
        enc.set_buffer(0, _d_gqa_oacc);
        enc.set_buffer(1, _d_gqa_m);
        enc.set_buffer(2, _d_gqa_l);
        enc.set_buffer(3, _d_attn);
        enc.set_constant(4, D);
        enc.set_constant(5, sp);
        enc.set_constant(6, Hq);
        enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
      } else if (use_gtile) {
        // Adaptive split: size sp to the actual scan length so the tiny
        // sliding window isn't over-split (a 64-way merge over 512 keys reads
        // more than the scan). sp = clamp(ceil(scan/_gtile_kps),1,_gtile_split).
        const int scan = (window > 0) ? std::min(window, T_kv) : T_kv;
        int sp = _gtile_split;
        if (_gtile_kps > 0) {
          sp = (scan + _gtile_kps - 1) / _gtile_kps;
          sp = std::max(1, std::min(sp, _gtile_split));
        }
        const bool use_vec = gtile_vec;
        // One-shot dump (VPIPE_GEMMA_ATTN_DUMP): confirm which kernel + sp +
        // threadgroup count each layer type uses at decode.
        if (std::getenv("VPIPE_GEMMA_ATTN_DUMP") && kv_off >= 256) {
          static int dumped_gt = 0;
          if (dumped_gt < 4) {
            std::fprintf(stderr,
              "[attn-dump L=%2d %s] use_gtile kernel=%s sp=%d scan=%d "
              "grid{32,%d,%d} tg{32,%d,1} -> %d threadgroups\n",
              L, ly.is_full ? "GLOBAL " : "sliding",
              _gtile_staged ? "gqa_tile" : use_vec ? "gqa_vec" : "gqa_direct",
              sp, scan, Hq, sp, Hq / Hkv, (Hq / (Hq / Hkv)) * sp);
            ++dumped_gt;
          }
        }
        // Linearized-ring read (VPIPE_GEMMA_RING_LINEAR): for the BOUNDED
        // sliding window (window>0, ring_cap>0, vec kernel), read the trailing
        // window as ONE contiguous physical span [phys_first, phys_first+scan).
        // The mirror tail (window-1 slots at [ring_cap, ...)) covers the wrap, so
        // the kernel scans linearly -- no `% ring_cap`, no wrap branch. cap (the
        // physical head stride) already includes the mirror tail, keeping the
        // span in-bounds (phys_first+scan <= ring_cap-1+window == cap).
        const bool ring_lin =
            _ring_linear && use_vec && window > 0 && ring_cap > 0;
        if (ring_lin) {
          const int first = std::max(0, kv_off - window + 1);
          const int phys_first = first % ring_cap;     // span start (physical)
          const int scan_len = scan;                   // == min(window, T_kv)
          enc.set_function(_fn_sdpa_gqa_vec_lin);
          enc.set_buffer(0, _d_q);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, _d_gqa_oacc);
          enc.set_buffer(4, _d_gqa_m);
          enc.set_buffer(5, _d_gqa_l);
          enc.set_constant(6, one_scale);
          enc.set_constant(7, T_kv);
          enc.set_constant(8, D);
          enc.set_constant(9, Hq);
          enc.set_constant(10, Hkv);
          enc.set_constant(11, kv_off);    // q_offset (ABI parity)
          enc.set_constant(12, cap);       // kv_stride (physical head stride)
          enc.set_constant(13, phys_first);
          enc.set_constant(14, scan_len);
          enc.set_constant(15, sp);
          enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                       {32, (unsigned)(Hq / Hkv), 1});
        } else {
        enc.set_function(_gtile_staged ? _fn_sdpa_gqa_tile
                         : use_vec      ? _fn_sdpa_gqa_vec
                                        : _fn_sdpa_gqa_direct);
        enc.set_buffer(0, _d_q);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, _d_gqa_oacc);
        enc.set_buffer(4, _d_gqa_m);
        enc.set_buffer(5, _d_gqa_l);
        enc.set_constant(6, one_scale);
        enc.set_constant(7, T_kv);
        enc.set_constant(8, D);
        enc.set_constant(9, Hq);
        enc.set_constant(10, Hkv);
        enc.set_constant(11, kv_off);      // q_offset
        enc.set_constant(12, cap);         // kv_stride (head stride == cap)
        enc.set_constant(13, window);
        enc.set_constant(14, ring_cap);    // ring modulo (0 = linear)
        enc.set_constant(15, sp);
        // Grid is TOTAL threads: grid.y = Hq, tg.y = G=Hq/Hkv -> Hq/G = Hkv
        // threadgroups in y, each with G simdgroups (one per q-head).
        enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                     {32, (unsigned)(Hq / Hkv), 1});
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
      } else if (use_gqa) {
        const int sp = _gqa_split;
        enc.set_function(_fn_sdpa_gqa);
        enc.set_buffer(0, _d_q);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, _d_gqa_oacc);
        enc.set_buffer(4, _d_gqa_m);
        enc.set_buffer(5, _d_gqa_l);
        enc.set_constant(6, one_scale);
        enc.set_constant(7, T_kv);
        enc.set_constant(8, D);
        enc.set_constant(9, Hq);
        enc.set_constant(10, Hkv);
        enc.set_constant(11, kv_off);      // q_offset
        enc.set_constant(12, cap);         // kv_stride (head stride == cap)
        enc.set_constant(13, window);
        enc.set_constant(14, ring_cap);    // ring modulo (0 = linear)
        enc.set_constant(15, sp);
        enc.dispatch({32, (unsigned)Hkv, (unsigned)sp}, {32, 1, 1});
        enc.set_function(_fn_sdpa_gqa_merge);
        enc.set_buffer(0, _d_gqa_oacc);
        enc.set_buffer(1, _d_gqa_m);
        enc.set_buffer(2, _d_gqa_l);
        enc.set_buffer(3, _d_attn);
        enc.set_constant(4, D);
        enc.set_constant(5, sp);
        enc.set_constant(6, Hq);
        enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
      } else {
      if (std::getenv("VPIPE_GEMMA_ATTN_DUMP") && kv_off >= 256) {
        static int dumped_mb = 0;
        if (dumped_mb < 4) {
          std::fprintf(stderr, "[attn-dump L=%2d %s] sdpa_mb (NO split) "
              "grid{%d,%d,1} -> %d threadgroups\n", L,
              ly.is_full ? "GLOBAL " : "sliding", (int)(4096 / D), Hq, Hq);
          ++dumped_mb;
        }
      }
      const unsigned bn = (unsigned)(4096 / D);   // 16@256, 8@512
      enc.set_function(_fn_sdpa_mb);
      enc.set_buffer(0, _d_q);
      enc.set_buffer(1, *Kuse);
      enc.set_buffer(2, *Vuse);
      enc.set_buffer(3, _d_attn);
      enc.set_constant(4, one_scale);
      enc.set_constant(5, T_kv);
      enc.set_constant(6, D);
      enc.set_constant(7, Hq);
      enc.set_constant(8, Hkv);
      enc.set_constant(9, nq);
      enc.set_constant(10, kv_off);        // q_offset
      enc.set_constant(11, cap);           // kv_stride (head stride = cap)
      enc.set_constant(12, window);
      enc.set_constant(13, ring_cap);      // ring modulo (0 = linear)
      enc.dispatch({bn * 32, (unsigned)Hq, 1}, {bn * 32, 1, 1});
      }
    }

    if (!skip_proj) {
      DUP(DC_PROJ, [&] { qmv(ly.ow, ly.os, ly.ob, _d_attn, _d_o, qd, H); });
    }
    DUP(DC_NORM, [&] { rms_add(_d_o, ly.post_attn_ln, _d_x, 1, H); });  // += rms

    // MLP (geglu sandwich).
    DUP(DC_NORM, [&] { rms(_d_x, ly.pre_ffn_ln, _d_hn, 1, H); });
    // Fused gate/up GEMV + GeGLU: one dispatch over the interleaved weight
    // writes gelu(gate)*up straight to _d_act [1, ffn] (no gate/up buffers).
    if (!skip_ffn) {
    DUP(DC_FFN, [&] {
    enc.set_function(_fn_qmv_geglu);
    enc.set_buffer(0, ly.guw);
    enc.set_buffer(1, ly.gus);
    enc.set_buffer(2, ly.gub);
    enc.set_buffer(3, _d_hn);
    enc.set_buffer(4, _d_act);
    enc.set_constant(5, H);
    enc.set_constant(6, 2 * c.ffn_inner);
    enc.dispatch({32, (unsigned)(c.ffn_inner / 2), 1}, {32, 2, 1});
    });
    // down_proj runs at mlp_bits (8-bit for gemma4_unified).
    DUP(DC_FFN, [&] {
      qmv_fn(_fn_qmv_mlp, ly.dw, ly.ds, ly.db, _d_act, _d_mlp, c.ffn_inner, H);
    });
    }
    DUP(DC_NORM, [&] { rms_add(_d_mlp, ly.post_ffn_ln, _d_x, 1, H); });  // += rms

    if (do_ple) {
      DUP(DC_PLE, [&] {
      // Per-layer-input gate: gelu(gate(x)) * pli[L] -> proj -> norm -> add.
      // The gate GEMV + geglu are FUSED (gelu*pli[L] in the GEMV write); falls
      // back to the 2-dispatch path if the fused kernel is unavailable.
      // pli[L] lives in a per-chunk buffer when pipelined (Design A), else the
      // single _d_pli at L*hpli.
      const SharedBuffer& pli_buf =
          _ple_chunk_k > 0 ? _d_pli_ch[L / _ple_chunk_k] : _d_pli;
      const std::size_t pli_off = _ple_chunk_k > 0
          ? (std::size_t)(L % _ple_chunk_k) * hpli * 2
          : (std::size_t)L * hpli * 2;
      if (_fn_qmv_gelu_mul.valid()) {
        enc.set_function(_fn_qmv_gelu_mul);
        enc.set_buffer(0, ly.plg_w);
        enc.set_buffer(1, ly.plg_s);
        enc.set_buffer(2, ly.plg_b);
        enc.set_buffer(3, _d_x);
        enc.set_buffer(4, _d_plg);
        enc.set_constant(5, H);
        enc.set_constant(6, hpli);
        enc.set_buffer(7, pli_buf, pli_off);                  // pli[L] vec
        enc.dispatch({32, (unsigned)(hpli / 4), 1}, {32, 2, 1});
      } else {
        qmv(ly.plg_w, ly.plg_s, ly.plg_b, _d_x, _d_plg, H, hpli);
        geglu(_d_plg, pli_buf, _d_plg, hpli, pli_off);
      }
      // per_layer_projection plp[1,H]=plg[1,hpli]@W[H,hpli]^T. Native 4-bit
      // qmv (K=hpli=256 -> one partial block) by default; f16 path for A/B.
      if (_ple_quant) {
        qmv(ly.plp_w, ly.plp_s, ly.plp_b, _d_plg, _d_plp, hpli, H);
      } else if (_ple_gemv) {
        dense_gemv(_d_plg, ly.plp_w, _d_plp, hpli, H);
      } else {
        enc.set_function(_fn_dense_t);
        enc.set_buffer(0, _d_plg);
        enc.set_buffer(1, ly.plp_w);
        enc.set_buffer(2, ly.plp_w);
        enc.set_buffer(3, _d_plp);
        enc.set_constant(4, hpli);          // K
        enc.set_constant(5, H);             // N
        const int M1 = 1;
        enc.set_constant(6, M1);
        const int hb0 = 0;
        enc.set_constant(7, hb0);
        enc.dispatch({(unsigned)(((H + 31) / 32) * 32), 2, 2}, {32, 2, 2});
      }
      // _d_x = (_d_x + rms(_d_plp)) * layer_scalar  (fused norm+add+scale)
      rms_add(_d_plp, ly.post_pli_ln, _d_x, 1, H, ly.layer_scalar);
      });
    } else {
      // No PLE: the layer tail is just h *= layer_scalar.
      DUP(DC_MISC, [&] { scale(_d_x, H, ly.layer_scalar); });
    }
    }   // lrep (whole-layer doubling for per-type gpu_active profiling)
  }

  // ---- final norm + tied quantized lm_head + softcap --------------
  // lm_head GEMV runs at embed_bits (8-bit on the GGUF path).
  DUP(DC_NORM, [&] { rms(_d_x, _final_ln, _d_hn, 1, H); });
  const int lm_reps = skip_lmhead ? 0 : (dup_layer == 5) ? 2 : 1;
  for (int lr = 0; lr < lm_reps; ++lr) {
  DUP(DC_LMHEAD, [&] {
    if (_embed_is_q6k) { lm_head_q6k(_d_hn, _d_logits, 0, H, c.vocab); }
    else {
      qmv_fn(_fn_qmv_embed, _embed_w, _embed_s, _embed_b, _d_hn, _d_logits, H,
             c.vocab);
    }
  });
  }
  if (c.final_softcap > 0.0f) {
    enc.set_function(_fn_softcap);
    enc.set_buffer(0, _d_logits);
    enc.set_buffer(1, _d_logits);
    enc.set_constant(2, c.vocab);
    enc.set_constant(3, c.final_softcap);
    enc.dispatch({(unsigned)c.vocab, 1, 1}, {256, 1, 1});
  }
  // Ban suppressed tokens (realtime thinking-channel suppression) -- AFTER
  // the softcap so the cap can't pull them back. Covers single-token decode
  // + pdecode (which run through encode_step_); the prefill chunk masks its
  // own local logits buffer separately.
  encode_suppress_(enc, _d_logits, c.vocab);
}

void
MetalGemmaModel::encode_suppress_(ComputeEncoder& enc,
                                  const SharedBuffer& logits, int vocab)
{
  // Mask the banned ids to the far-negative sentinel so neither the host
  // read-back, the GPU argmax, nor the GPU sampler can pick them (the sampler
  // does exp(logit - max), so a far-negative logit -> ~0 weight). Tiny
  // dispatch (one thread per banned id); skipped when nothing is suppressed.
  if (_n_suppress <= 0 || !_fn_suppress.valid() || _d_suppress_ids.empty()) {
    return;
  }
  enc.set_function(_fn_suppress);
  enc.set_buffer(0, logits);
  enc.set_buffer(1, _d_suppress_ids);
  enc.set_constant(2, _n_suppress);
  enc.set_constant(3, vocab);
  enc.dispatch({(unsigned)_n_suppress, 1, 1},
               {(unsigned)std::min(_n_suppress, 64), 1, 1});
}

// ---------------------------------------------------------------------
// Batched (N-branch parallel) decode.
// ---------------------------------------------------------------------
bool
MetalGemmaModel::ensure_bscratch_(BScratch& bs, int n)
{
  if (bs.n == n && !bs.logits.empty()) { return true; }
  const Config& c = _cfg;
  const int H = c.hidden, ffn = c.ffn_inner, hpli = c.hpli, nl = c.n_layers;
  const int qd_max = c.n_heads * c.head_dim_full;
  const int kd_max = c.n_kv_heads * c.head_dim_full;
  auto f16 = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  auto i32 = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(std::int32_t));
  };
  bs.x       = f16((std::size_t)n * H);
  bs.hn      = f16((std::size_t)n * H);
  bs.q3      = f16((std::size_t)n * qd_max);
  bs.kbuf    = f16((std::size_t)n * kd_max);
  bs.vbuf    = f16((std::size_t)n * kd_max);
  bs.attn    = f16((std::size_t)n * qd_max);
  bs.o       = f16((std::size_t)n * H);
  bs.act     = f16((std::size_t)n * ffn);
  bs.mlp     = f16((std::size_t)n * H);
  if (hpli > 0) {
    bs.ple     = f16((std::size_t)n * nl * hpli);
    bs.pleproj = f16((std::size_t)n * nl * hpli);
    bs.pli     = f16((std::size_t)n * nl * hpli);
    bs.plg     = f16((std::size_t)n * hpli);
    bs.plp     = f16((std::size_t)n * H);
  }
  bs.logits    = f16((std::size_t)n * c.vocab);
  bs.tok_in    = i32((std::size_t)n);
  bs.argmax_id = i32((std::size_t)n);
  // Per-branch FULL-layer page tables (full_layers_paged).
  if (_full_paged) {
    bs.pgtab = i32((std::size_t)n * _ctx->max_pages() * 3);
  }
  // Per-branch gtile / paged-GQA partials (f32) for the global layers
  // (head_dim_full); branch i owns its slice.
  if (_gtile_attn || _pgqa_attn) {
    const std::size_t sp = (std::size_t)_gtile_split;
    const std::size_t Hq = (std::size_t)c.n_heads;
    const std::size_t Dd = (std::size_t)c.head_dim_full;
    auto f32 = [&](std::size_t e) {
      return _mc->make_shared_buffer(e * sizeof(float));
    };
    bs.gqa_oacc = f32((std::size_t)n * Hq * sp * Dd);
    bs.gqa_m = f32((std::size_t)n * Hq * sp);
    bs.gqa_l = f32((std::size_t)n * Hq * sp);
  }
  bs.n = n;
  return !bs.logits.empty();
}

void
MetalGemmaModel::encode_batched_step_(
    ComputeEncoder& enc, BScratch& bs,
    std::span<const ContextId> cids,
    const std::vector<int>& kv_off_v,
    const std::vector<int>& rope_pos_v)
{
  const Config& c = _cfg;
  const int H = c.hidden, Hq = c.n_heads, ffn = c.ffn_inner;
  const int hpli = c.hpli, nl = c.n_layers;
  const bool has_ple = hpli > 0;
  const float eps = c.rms_eps;
  const int N = bs.n;

  // FULL (global) layers paged: claim each branch's page slot ONCE (shared by
  // every full layer of that branch) and build its page table into bs.pgtab
  // slot i. append() advances each branch's seq_len, so the caller must NOT
  // kv_append. The branches share the parent's frozen prefix pages by refcount
  // (no per-branch full-KV copy) and own only their divergent decode pages.
  const int page_tokens = _ctx->page_tokens();
  const std::size_t pgstride = (std::size_t)_ctx->max_pages() * 3;
  std::vector<ContextManager::AppendSlot> fslots;
  std::vector<int> fnpages;
  if (_full_paged) {
    fslots.resize((std::size_t)N);
    fnpages.assign((std::size_t)N, 0);
    auto* pgbase = static_cast<std::int32_t*>(bs.pgtab.contents());
    for (int i = 0; i < N; ++i) {
      fslots[(std::size_t)i] = _ctx->append(cids[(std::size_t)i], 1);
      fnpages[(std::size_t)i] = _ctx->fill_page_table(
          cids[(std::size_t)i], pgbase + (std::size_t)i * pgstride);
    }
  }

  auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                 const SharedBuffer& w, const SharedBuffer& y,
                 std::size_t yoff, int R, int Hd) {
    enc.set_function((_rms_fast || Hd > 4096) ? _fn_rms_fast : _fn_rms);
    enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, Hd); enc.set_constant(4, eps);
    enc.dispatch({512, (unsigned)R, 1}, {512, 1, 1});
  };
  auto rms_add = [&](const SharedBuffer& xin, const SharedBuffer& w,
                     const SharedBuffer& res, int R, int Hd,
                     float post_scale = 1.0f) {
    enc.set_function((_rms_fast || Hd > 4096) ? _fn_rms_add_fast : _fn_rms_add);
    enc.set_buffer(0, xin); enc.set_buffer(1, w);
    enc.set_buffer(2, res); enc.set_buffer(3, res);
    enc.set_constant(4, Hd); enc.set_constant(5, eps);
    enc.set_constant(6, post_scale);
    enc.dispatch({512, (unsigned)R, 1}, {512, 1, 1});
  };
  // fn_batch is the MAXM=2 batched GEMV for this weight (qmv bandwidth,
  // weights read once across the rows); invalid when the weight is 8-bit
  // (12B MLP) -> steel GEMM. N==1 -> plain matvec; N>2 tiles ceil(N/2)
  // rows along grid.z. Same geometry/constraints as the qmv path.
  auto qmm_fn = [&](const auto& fn_mv, const auto& fn_mm,
                    const auto& fn_batch,
                    const SharedBuffer& w, const SharedBuffer& s,
                    const SharedBuffer& b, const SharedBuffer& xin,
                    const SharedBuffer& y, int Kk, int Nout) {
    if (N == 1) {
      enc.set_function(fn_mv);
      enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
      enc.set_buffer(3, xin); enc.set_buffer(4, y);
      enc.set_constant(5, Kk); enc.set_constant(6, Nout);
      enc.dispatch({32, (unsigned)(Nout / 4), 1}, {32, 2, 1});
      return;
    }
    if (fn_batch.valid()) {
      enc.set_function(fn_batch);
      enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
      enc.set_buffer(3, xin); enc.set_buffer(4, y);
      enc.set_constant(5, Kk); enc.set_constant(6, Nout);
      enc.set_constant(7, N);
      enc.dispatch({32, (unsigned)(Nout / 4), (unsigned)((N + 1) / 2)},
                   {32, 2, 1});
      return;
    }
    enc.set_function(fn_mm);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, N);
    enc.dispatch({(unsigned)(((Nout + 31) / 32) * 32),
                 (unsigned)(((N + 31) / 32) * 2), 2}, {32, 2, 2});
  };
  auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, const SharedBuffer& xin,
                 const SharedBuffer& y, int Kk, int Nout) {
    qmm_fn(_fn_qmv, _fn_qmm, _fn_qmv_batch, w, s, b, xin, y, Kk, Nout);
  };
  // Per-branch single-position RoPE at that branch's rope position.
  auto rope1 = [&](const SharedBuffer& xb, std::size_t xoff,
                   const SharedBuffer& invf, int heads, int D, int rp) {
    enc.set_function(_fn_rope);
    enc.set_buffer(0, xb, xoff); enc.set_buffer(1, invf);
    enc.set_constant(2, heads); const int one = 1; enc.set_constant(3, one);
    enc.set_constant(4, D); enc.set_constant(5, rp);
    enc.dispatch({(unsigned)(D / 2), 1, (unsigned)heads},
                 {(unsigned)(D / 2 < 256 ? D / 2 : 256), 1, 1});
  };
  auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                      const SharedBuffer& out, int n) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto scale = [&](const SharedBuffer& x, int n, float s) {
    enc.set_function(_fn_scale);
    enc.set_buffer(0, x); enc.set_constant(1, n); enc.set_constant(2, s);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  auto geglu1 = [&](const SharedBuffer& g, std::size_t goff,
                    const SharedBuffer& u, std::size_t uoff, int nn) {
    enc.set_function(_fn_geglu);
    enc.set_buffer(0, g, goff); enc.set_buffer(1, u, uoff);
    enc.set_buffer(2, g, goff); enc.set_constant(3, nn);
    enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
  };

  // Batched embed gather (token ids -> [N, Hd]); the kernel uses grid.y as
  // the row index, so N tokens gather in one dispatch.
  auto embed = [&](const SharedBuffer& w, const SharedBuffer& s,
                   const SharedBuffer& b, const SharedBuffer& out, int Hd) {
    enc.set_function(_fn_embed);
    enc.set_buffer(0, bs.tok_in); enc.set_buffer(1, w); enc.set_buffer(2, s);
    enc.set_buffer(3, b); enc.set_buffer(4, out); enc.set_constant(5, Hd);
    enc.dispatch({(unsigned)Hd, (unsigned)N, 1}, {256, 1, 1});
  };

  // ---- PLE projection (e4b only): batched over the N rows ----------
  if (has_ple) {
    const int ple = nl * hpli;
    embed(_ple_w, _ple_s, _ple_b, bs.ple, ple);
    scale(bs.ple, N * ple, std::sqrt((float)hpli));
    // per_layer_model_projection [N, ple] = x[N,H] @ W[ple,H]^T (BF16 dense).
    enc.set_function(_fn_dense_t);
    enc.set_buffer(0, bs.x); enc.set_buffer(1, _plm_proj_w);
    enc.set_buffer(2, _plm_proj_w); enc.set_buffer(3, bs.pleproj);
    enc.set_constant(4, H); enc.set_constant(5, ple);
    enc.set_constant(6, N); const int hb0 = 0; enc.set_constant(7, hb0);
    enc.dispatch({(unsigned)(((ple + 31) / 32) * 32),
                 (unsigned)(((N + 31) / 32) * 2), 2}, {32, 2, 2});
    scale(bs.pleproj, N * ple, std::pow((float)H, -0.5f));
    rms(bs.pleproj, 0, _ple_proj_norm, bs.pleproj, 0, N * nl, hpli);
    residual(bs.pleproj, bs.ple, bs.pli, N * ple);
    scale(bs.pli, N * ple, 0.70710678f);
  }

  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    const int D = ly.head_dim, Hkv = ly.n_kv;
    const int qd = Hq * D, kd = Hkv * D;
    const SharedBuffer& invf = ly.is_full ? _inv_freq_full : _inv_freq_sliding;
    rms(bs.x, 0, ly.in_ln, bs.hn, 0, N, H);
    qmm(ly.qw, ly.qs, ly.qb, bs.hn, bs.q3, H, qd);
    rms(bs.q3, 0, ly.q_norm, bs.q3, 0, N * Hq, D);
    if (ly.kv_source < 0) {
      qmm(ly.kw, ly.ks, ly.kb, bs.hn, bs.kbuf, H, kd);
      if (ly.k_eq_v) {
        rms(bs.kbuf, 0, _ones_vnorm, bs.vbuf, 0, N * Hkv, D);   // v=vnorm(k)
      } else {
        qmm(ly.vw, ly.vs, ly.vb, bs.hn, bs.vbuf, H, kd);
        rms(bs.vbuf, 0, _ones_vnorm, bs.vbuf, 0, N * Hkv, D);
      }
      rms(bs.kbuf, 0, ly.k_norm, bs.kbuf, 0, N * Hkv, D);
    }
    // Per-branch: RoPE at the branch position, write K/V to the branch's own
    // Contiguous KV at its kv_off, and decode SDPA over its own K/V.
    for (int i = 0; i < N; ++i) {
      const std::size_t qoff = (std::size_t)i * qd * 2;
      const std::size_t koff = (std::size_t)i * kd * 2;
      const int kv_off = kv_off_v[(std::size_t)i];
      rope1(bs.q3, qoff, invf, Hq, D, rope_pos_v[(std::size_t)i]);
      const bool paged_full = ly.is_full && _full_paged;
      const SharedBuffer* Kuse =
          paged_full ? _ctx->kpool(L) : _ctx->kv_k(cids[(std::size_t)i], L);
      const SharedBuffer* Vuse =
          paged_full ? _ctx->vpool(L) : _ctx->kv_v(cids[(std::size_t)i], L);
      const int cap = paged_full ? 0 : _ctx->kv_capacity(cids[(std::size_t)i], L);
      const int ring_cap =
          paged_full ? 0 : _ctx->kv_ring_cap(cids[(std::size_t)i], L);
      if (ly.kv_source < 0) {
        rope1(bs.kbuf, koff, invf, Hkv, D, rope_pos_v[(std::size_t)i]);
        if (paged_full) {
          // Paged write of this branch's one token into its page slot.
          const std::size_t poff = (std::size_t)fslots[(std::size_t)i]
              .page_id.v * _ctx->page_stride_bytes();
          auto wp = [&](const SharedBuffer& src, const SharedBuffer& pool) {
            enc.set_function(_fn_kv_write_paged);
            enc.set_buffer(0, src, koff);
            enc.set_buffer(1, pool, poff);
            enc.set_constant(2, page_tokens);
            enc.set_constant(3, D);
            const int one = 1, zero = 0;
            enc.set_constant(4, one);                          // n_src
            enc.set_constant(5, zero);                         // src_off
            enc.set_constant(6, fslots[(std::size_t)i].slot_offset);
            enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
          };
          wp(bs.kbuf, *Kuse);
          wp(bs.vbuf, *Vuse);
        } else {
          const int kvw_win = ly.is_full ? 0 : c.sliding_window;
          auto kvw = [&](const SharedBuffer& src, std::size_t soff,
                         const SharedBuffer& cache) {
            enc.set_function(_fn_kv_write);
            enc.set_buffer(0, src, soff); enc.set_buffer(1, cache);
            enc.set_constant(2, cap); enc.set_constant(3, D);
            const int one = 1; enc.set_constant(4, one);
            enc.set_constant(5, kv_off);
            enc.set_constant(6, ring_cap);      // ring modulo (0 = linear)
            enc.set_constant(7, kvw_win);       // trailing window (mirror)
            enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
          };
          kvw(bs.kbuf, koff, *Kuse);
          kvw(bs.vbuf, koff, *Vuse);
        }
      }
      const float one_scale = 1.0f;
      const int T_kv = kv_off + 1;
      const int window = ly.is_full ? 0 : c.sliding_window;
      const int nq = 1;
      if (paged_full) {
        // Paged decode attention over this branch's pool + page table. Fast
        // path: KV-split GQA vec flash-decode into this branch's own partial
        // slice (no cross-branch hazard) + merge; fallback: scalar paged.
        const int Gb = (Hkv > 0) ? Hq / Hkv : 0;
        const std::size_t pgoff =
            (std::size_t)i * pgstride * sizeof(std::int32_t);
        const bool use_pgqa = _pgqa_attn && Hkv > 0 && (Hq % Hkv == 0)
            && Gb >= 1 && Gb <= 16 && (D % 128 == 0) && !bs.gqa_oacc.empty();
        if (use_pgqa) {
          const int sp = _gtile_split;
          const std::size_t Df = (std::size_t)c.head_dim_full;
          const std::size_t ooff =
              (std::size_t)i * Hq * (std::size_t)sp * Df * sizeof(float);
          const std::size_t moff =
              (std::size_t)i * Hq * (std::size_t)sp * sizeof(float);
          enc.set_function(_fn_sdpa_pgqa);
          enc.set_buffer(0, bs.q3, qoff);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, bs.gqa_oacc, ooff);
          enc.set_buffer(4, bs.gqa_m, moff);
          enc.set_buffer(5, bs.gqa_l, moff);
          enc.set_constant(6, one_scale);
          enc.set_constant(7, D);
          enc.set_constant(8, Hq);
          enc.set_constant(9, Hkv);
          enc.set_constant(10, kv_off);        // q_offset
          enc.set_constant(11, page_tokens);
          enc.set_constant(12, fnpages[(std::size_t)i]);
          enc.set_buffer(13, bs.pgtab, pgoff);
          enc.set_constant(14, sp);
          enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                       {32, (unsigned)(Hq / Hkv), 1});
          enc.set_function(_fn_sdpa_gqa_merge);
          enc.set_buffer(0, bs.gqa_oacc, ooff);
          enc.set_buffer(1, bs.gqa_m, moff);
          enc.set_buffer(2, bs.gqa_l, moff);
          enc.set_buffer(3, bs.attn, qoff);
          enc.set_constant(4, D);
          enc.set_constant(5, sp);
          enc.set_constant(6, Hq);
          enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
        } else {
          enc.set_function(_fn_sdpa_paged_causal);
          enc.set_buffer(0, bs.q3, qoff);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, bs.attn, qoff);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, D);
          enc.set_constant(6, Hq);
          enc.set_constant(7, Hkv);
          enc.set_constant(8, nq);
          enc.set_constant(9, kv_off);          // q_offset
          enc.set_constant(10, page_tokens);
          enc.set_constant(11, fnpages[(std::size_t)i]);
          enc.set_buffer(12, bs.pgtab, pgoff);
          enc.dispatch({32, (unsigned)Hq, 1}, {32, 1, 1});
        }
        continue;
      }
      // Global layers: threadgroup-staged flash-decode into this branch's own
      // partial slice (no cross-branch scratch hazard), then merge. Sliding
      // layers stay on sdpa_mb. Mirrors the single-decode path.
      const int Gg = (Hkv > 0) ? Hq / Hkv : 0;
      const bool use_gtile = _gtile_attn && (window <= 0) && (D <= 512)
          && (D % 32 == 0) && Hkv > 0 && (Hq % Hkv == 0) && Gg >= 1 && Gg <= 16
          && !bs.gqa_oacc.empty();
      if (use_gtile) {
        const int sp = _gtile_split;
        const std::size_t Df = (std::size_t)c.head_dim_full;   // slice stride
        const std::size_t ooff =
            (std::size_t)i * Hq * (std::size_t)sp * Df * sizeof(float);
        const std::size_t moff =
            (std::size_t)i * Hq * (std::size_t)sp * sizeof(float);
        const bool use_vec = !_gtile_staged && !_gtile_direct
            && _fn_sdpa_gqa_vec.valid() && (D % 128 == 0);
        enc.set_function(_gtile_staged ? _fn_sdpa_gqa_tile
                         : use_vec      ? _fn_sdpa_gqa_vec
                                        : _fn_sdpa_gqa_direct);
        enc.set_buffer(0, bs.q3, qoff);
        enc.set_buffer(1, *Kuse); enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, bs.gqa_oacc, ooff);
        enc.set_buffer(4, bs.gqa_m, moff);
        enc.set_buffer(5, bs.gqa_l, moff);
        enc.set_constant(6, one_scale); enc.set_constant(7, T_kv);
        enc.set_constant(8, D); enc.set_constant(9, Hq);
        enc.set_constant(10, Hkv); enc.set_constant(11, kv_off);
        enc.set_constant(12, cap); enc.set_constant(13, window);
        enc.set_constant(14, ring_cap); enc.set_constant(15, sp);
        enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                     {32, (unsigned)(Hq / Hkv), 1});
        enc.set_function(_fn_sdpa_gqa_merge);
        enc.set_buffer(0, bs.gqa_oacc, ooff);
        enc.set_buffer(1, bs.gqa_m, moff);
        enc.set_buffer(2, bs.gqa_l, moff);
        enc.set_buffer(3, bs.attn, qoff);
        enc.set_constant(4, D); enc.set_constant(5, sp);
        enc.set_constant(6, Hq);
        enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
      } else {
        const unsigned bn = (unsigned)(4096 / D);
        enc.set_function(_fn_sdpa_mb);
        enc.set_buffer(0, bs.q3, qoff);
        enc.set_buffer(1, *Kuse); enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, bs.attn, qoff);
        enc.set_constant(4, one_scale); enc.set_constant(5, T_kv);
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, nq);
        enc.set_constant(10, kv_off); enc.set_constant(11, cap);
        enc.set_constant(12, window); enc.set_constant(13, ring_cap);
        enc.dispatch({bn * 32, (unsigned)Hq, 1}, {bn * 32, 1, 1});
      }
    }
    qmm(ly.ow, ly.os, ly.ob, bs.attn, bs.o, qd, H);
    rms_add(bs.o, ly.post_attn_ln, bs.x, N, H);

    // MLP (geglu): fused gate/up + down, batched.
    rms(bs.x, 0, ly.pre_ffn_ln, bs.hn, 0, N, H);
    qmm_fn(_fn_qmv_geglu, _fn_qmm_geglu, _fn_qmv_batch_geglu, ly.guw, ly.gus,
           ly.gub, bs.hn, bs.act, H, 2 * ffn);
    qmm_fn(_fn_qmv_mlp, _fn_qmm_mlp, _fn_qmv_batch_mlp, ly.dw, ly.ds, ly.db,
           bs.act, bs.mlp, ffn, H);
    rms_add(bs.mlp, ly.post_ffn_ln, bs.x, N, H);

    if (has_ple) {
      qmm(ly.plg_w, ly.plg_s, ly.plg_b, bs.x, bs.plg, H, hpli);
      for (int i = 0; i < N; ++i) {
        // pli[i, L, :] at ((i*nl)+L)*hpli; plg[i] at i*hpli.
        geglu1(bs.plg, (std::size_t)i * hpli * 2, bs.pli,
               ((std::size_t)i * nl + L) * hpli * 2, hpli);
      }
      if (_ple_quant) {
        qmm(ly.plp_w, ly.plp_s, ly.plp_b, bs.plg, bs.plp, hpli, H);
      } else {
        enc.set_function(_fn_dense_t);
        enc.set_buffer(0, bs.plg); enc.set_buffer(1, _layers[L].plp_w);
        enc.set_buffer(2, _layers[L].plp_w); enc.set_buffer(3, bs.plp);
        enc.set_constant(4, hpli); enc.set_constant(5, H);
        enc.set_constant(6, N); const int hb0 = 0; enc.set_constant(7, hb0);
        enc.dispatch({(unsigned)(((H + 31) / 32) * 32),
                     (unsigned)(((N + 31) / 32) * 2), 2}, {32, 2, 2});
      }
      rms_add(bs.plp, ly.post_pli_ln, bs.x, N, H, ly.layer_scalar);
    } else {
      scale(bs.x, N * H, ly.layer_scalar);
    }
  }
  // Final norm + tied lm_head + softcap (softcap matters for sampling;
  // monotonic so argmax is unaffected).
  rms(bs.x, 0, _final_ln, bs.hn, 0, N, H);
  if (_embed_is_q6k) {
    // Native Q6_K lm_head per branch (the GEMV is M=1); N is small.
    for (int i = 0; i < N; ++i) {
      enc.set_function(_fn_qmv_q6k);
      enc.set_buffer(0, _embed_q6k);
      enc.set_buffer(1, bs.hn, (std::size_t)i * H * 2);
      enc.set_buffer(2, bs.logits, (std::size_t)i * c.vocab * 2);
      enc.set_constant(3, H);
      enc.set_constant(4, c.vocab);
      enc.dispatch({32, (unsigned)(((c.vocab + 7) / 8) * 2), 1}, {32, 2, 1});
    }
  } else {
    qmm(_embed_w, _embed_s, _embed_b, bs.hn, bs.logits, H, c.vocab);
  }
  if (c.final_softcap > 0.0f) {
    enc.set_function(_fn_softcap);
    enc.set_buffer(0, bs.logits); enc.set_buffer(1, bs.logits);
    enc.set_constant(2, N * c.vocab); enc.set_constant(3, c.final_softcap);
    enc.dispatch({(unsigned)(N * c.vocab), 1, 1}, {256, 1, 1});
  }
}

bool
MetalGemmaModel::decode_batched_step(
    std::span<const ContextId>    cids,
    std::span<const std::int32_t> in_tokens,
    std::span<const std::int32_t> rope_pos,
    std::vector<float>&           out_logits)
{
  const int N = (int)cids.size();
  if (N <= 0 || (int)in_tokens.size() != N) { return false; }
  if (!rope_pos.empty() && (int)rope_pos.size() != N) { return false; }
  if (!_fn_embed.valid()) { return false; }
  if (!ensure_bscratch_(_bdec, N)) { return false; }
  BScratch& bs = _bdec;
  const Config& c = _cfg;
  const int H = c.hidden;

  std::vector<ContextId> cms((std::size_t)N);
  std::vector<int> kv_off_v((std::size_t)N), rope_pos_v((std::size_t)N);
  auto* tok = static_cast<std::int32_t*>(bs.tok_in.contents());
  for (int i = 0; i < N; ++i) {
    cms[(std::size_t)i] = cm_for_(cids[(std::size_t)i]);
    if (!cms[(std::size_t)i].valid()) { return false; }
    const int kv_off = _ctx->kv_seq_len(cms[(std::size_t)i]);
    if (kv_off >= c.max_seq) { return false; }
    kv_off_v[(std::size_t)i] = kv_off;
    rope_pos_v[(std::size_t)i] =
        (rope_pos.empty() || rope_pos[(std::size_t)i] < 0)
            ? kv_off
            : rope_pos[(std::size_t)i];
    tok[i] = in_tokens[(std::size_t)i];
  }
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // Batched embed gather (N token ids -> bs.x[N, H]) then embed_scale.
    if (_embed_is_q6k) {
      enc.set_function(_fn_embed_q6k);
      enc.set_buffer(0, bs.tok_in); enc.set_buffer(1, _embed_q6k);
      enc.set_buffer(2, bs.x); enc.set_constant(3, H);
    } else {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, bs.tok_in); enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, _embed_s); enc.set_buffer(3, _embed_b);
      enc.set_buffer(4, bs.x); enc.set_constant(5, H);
    }
    enc.dispatch({(unsigned)H, (unsigned)N, 1}, {256, 1, 1});
    enc.set_function(_fn_scale);   // embed_scale over [N, H]
    enc.set_buffer(0, bs.x);
    enc.set_constant(1, N * H);
    const float es = std::sqrt((float)H);
    enc.set_constant(2, es);
    enc.dispatch({(unsigned)(N * H), 1, 1}, {256, 1, 1});
    encode_batched_step_(
        enc, bs,
        std::span<const ContextId>(cms.data(), cms.size()),
        kv_off_v, rope_pos_v);
  }
  stream.commit().wait();
  if (!_full_paged) {
    // Paged path advanced each branch's seq_len via append() in
    // encode_batched_step_; the contiguous fallback bumps here.
    for (int i = 0; i < N; ++i) { _ctx->kv_append(cms[(std::size_t)i], 1); }
  }
  out_logits.resize((std::size_t)N * c.vocab);
  read_elt_(bs.logits.contents(), out_logits.data(),
            (std::size_t)N * c.vocab, c.use_bf16);
  return true;
}

std::vector<float>
MetalGemmaModel::forward(ContextId cid, std::int32_t token_id, int rope_pos)
{
  (void)rope_pos;   // text-only Phase 3: rope == KV position
  const Config& c = _cfg;
  if (!ensure_scratch_()) { return {}; }
  ContextId cm = cm_for_(cid);
  if (!cm.valid()) { return {}; }
  const int kv_off = _ctx->kv_seq_len(cm);
  if (kv_off >= c.max_seq) { return {}; }
  *static_cast<std::int32_t*>(_d_tok.contents()) = token_id;

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    encode_step_(enc, cm, kv_off);   // advances seq_len (append/kv_append)
  }
  stream.commit().wait();

  std::vector<float> out((std::size_t)c.vocab);
  read_elt_(_d_logits.contents(), out.data(), (std::size_t)c.vocab,
            c.use_bf16);
  return out;
}

// Encode a greedy argmax of _d_logits into out[out_off]. Two-stage when the
// argmax_partial/combine kernels + partials scratch are present (the 262k read
// parallelised across kArgmaxM cores; ~17x the single-tg scan in isolation,
// token-exact), else the single-tg _fn_argmax. See header.
void
MetalGemmaModel::encode_argmax_(metal_compute::ComputeEncoder& enc,
                                const metal_compute::SharedBuffer& out,
                                std::size_t out_off, int vocab)
{
  // VPIPE_GEMMA_ARGMAX1=1 forces the single-tg argmax (A/B against two-stage).
  static const bool kForce1 = std::getenv("VPIPE_GEMMA_ARGMAX1") != nullptr;
  if (!kForce1 && _fn_argmax_partial.valid() && _fn_argmax_combine.valid()
      && !_d_argmax_part.empty()) {
    enc.set_function(_fn_argmax_partial);
    enc.set_buffer(0, _d_logits);
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
    enc.set_buffer(0, _d_logits);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, vocab);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  }
}

// Encode the GPU sampler over _d_logits into out[out_off]. Histogram multi-tg
// path (sample_*_f16) by default; VPIPE_GEMMA_SAMPLE1=1 (or missing kernels /
// scratch) falls back to the single-tg _fn_sample. See header.
void
MetalGemmaModel::encode_sample_(metal_compute::ComputeEncoder& enc,
                                const metal_compute::SharedBuffer& out,
                                std::size_t out_off,
                                const metal_compute::SharedBuffer& ws,
                                const metal_compute::SharedBuffer& seen,
                                const GpuSamplerParams& sp,
                                std::uint32_t step_seed, int vocab)
{
  static const bool kForce1 = std::getenv("VPIPE_GEMMA_SAMPLE1") != nullptr;
  const bool hist_ok =
      _fn_smp_max_partial.valid() && _fn_smp_max_combine.valid() &&
      _fn_smp_zhist_partial.valid() && _fn_smp_zhist_combine.valid() &&
      _fn_smp_thresh.valid() && _fn_smp_pick_partial.valid() &&
      _fn_smp_pick_combine.valid() && !_d_smp_hpart.empty();
  if (kForce1 || !hist_ok) {
    enc.set_function(_fn_sample);
    enc.set_buffer(0, _d_logits);
    enc.set_buffer(1, out, out_off);
    enc.set_constant(2, vocab);
    enc.set_constant(3, sp.temperature);
    enc.set_constant(4, sp.top_p);
    enc.set_constant(5, step_seed);
    enc.set_buffer(6, ws);
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
  enc.set_buffer(0, _d_logits);
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
  enc.set_buffer(0, _d_logits);
  enc.set_buffer(1, ws);
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
  enc.set_buffer(0, _d_logits);
  enc.set_buffer(1, ws);
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

std::int32_t
MetalGemmaModel::decode_step_fast(ContextId cid, std::int32_t token_id,
                                  int rope_pos)
{
  (void)rope_pos;
  const Config& c = _cfg;
  if (!_fn_argmax.valid()) {
    return std::numeric_limits<std::int32_t>::min();
  }
  if (!ensure_scratch_()) {
    return std::numeric_limits<std::int32_t>::min();
  }
  ContextId cm = cm_for_(cid);
  if (!cm.valid()) { return -1; }
  const int kv_off = _ctx->kv_seq_len(cm);
  if (kv_off >= c.max_seq) { return -1; }
  *static_cast<std::int32_t*>(_d_tok.contents()) = token_id;

  static const bool kProf = std::getenv("VPIPE_GEMMA_PROF") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  long n_disp = 0;
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    encode_step_(enc, cm, kv_off);   // writes _d_logits
    // On-GPU greedy argmax -> _d_argmax_id (no full [vocab] host pull).
    // softcap is monotonic, so argmax is unaffected by it. Two-stage parallel
    // argmax (encode_argmax_): the 262k-vocab read is split across kArgmaxM
    // cores -- ~17x the single-tg scan in isolation, token-exact. (An EARLIER
    // two-pass attempt without simd reduction regressed; the simd_max/min
    // partials win on the M5 matrix-core GPU -- re-measured here.)
    encode_argmax_(enc, _d_argmax_id, 0, c.vocab);
    n_disp = enc.dispatch_count();
  }
  // FEASIBILITY PROBE (VPIPE_GEMMA_PLE_PROBE): fire ~1.1 ms of INDEPENDENT
  // bandwidth-bound work (the PLE-projection GEMV x5) on a SECOND command
  // stream, concurrently with the main decode. If the main's gpu_active is
  // unchanged, a concurrent queue overlaps into the bubbles (Design B viable);
  // if it rises ~1 ms, both compete for memory bandwidth (B is dead). Timing
  // only -- the throwaway writes to _d_pleproj race the real PLE (tokens wrong).
  static const bool kProbe = std::getenv("VPIPE_GEMMA_PLE_PROBE") != nullptr;
  metal_compute::CommandStream::Fence probeFence;
  if (kProbe && _fn_dense_gemv.valid() && !_plm_proj_w.empty()) {
    const int ple = c.n_layers * c.hpli;
    if (_d_probe_in.empty()) {                  // private, no sharing with main
      _d_probe_in = _mc->make_shared_buffer((std::size_t)c.hidden * 2);
      _d_probe_out = _mc->make_shared_buffer((std::size_t)ple * 2);
    }
    metal_compute::CommandStream sB = _mc->make_command_stream();
    {
      ComputeEncoder eb = sB.begin_compute();
      for (int it = 0; it < 5; ++it) {
        eb.set_function(_fn_dense_gemv);
        eb.set_buffer(0, _d_probe_in);
        eb.set_buffer(1, _plm_proj_w);          // read-only shared (no hazard)
        eb.set_buffer(2, _d_probe_out);
        eb.set_constant(3, c.hidden);
        eb.set_constant(4, ple);
        eb.dispatch({32, (unsigned)(ple / 4), 1}, {32, 2, 1});
      }
    }
    probeFence = sB.commit();                  // fire concurrently
  }
  const auto t1 = std::chrono::steady_clock::now();
  auto fence = stream.commit();
  fence.wait();
  if (probeFence.valid()) { probeFence.wait(); }
  if (kProf) {
    const auto t2 = std::chrono::steady_clock::now();
    const auto gt = fence.gpu_times();         // GPU/kernel active spans (s)
    static double enc_ms = 0, gpu_ms = 0, gpuact_ms = 0, kern_ms = 0;
    static int cnt = 0;
    enc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    gpu_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    gpuact_ms += gt.gpu_s * 1000.0;
    kern_ms += gt.kernel_s * 1000.0;
    if (++cnt % 32 == 0) {
      // commit+gpu = CPU wall from commit() to wait() return.
      // gpu_active = GPUStartTime..GPUEndTime (the buffer's GPU residency,
      // INCLUDING inter-dispatch idle bubbles). kernel = kernel exec window.
      // (commit+gpu - gpu_active) = CPU commit/dispatch + wait-wakeup latency.
      std::fprintf(stderr, "[gemma-metal-prof] encode %.2f | commit+gpu %.2f "
                   "| gpu_active %.2f | kernel %.2f ms | %ld disp/tok (%d)\n",
                   enc_ms / cnt, gpu_ms / cnt, gpuact_ms / cnt, kern_ms / cnt,
                   n_disp, cnt);
    }
  }
  // seq_len already advanced inside encode_step_ (append / kv_append).
  return *static_cast<const std::int32_t*>(_d_argmax_id.contents());
}

// ---- GPU-resident pipelined decode (pdecode_*) ---------------------
// The streaming counterpart of decode_step_fast: same encode_step_ at the
// same kv_off, but the input token is read from the GPU-resident gen_ids
// chain (no host re-upload) and the argmax/sample writes the next id back
// into gen_ids (no host logit pull). One reused command stream + a GPU
// event chains the per-token forwards. Greedy is token-identical to
// decode_step_fast (same kernel, same buffer, same position).
bool
MetalGemmaModel::pdecode_begin(ContextId cid, std::int32_t first_token,
                               std::span<const std::int32_t> prompt,
                               const GpuSamplerParams& sp, int max_tokens,
                               int rope_first)
{
  // Gemma uses 1-D rope == KV slot position (encode_step_ ignores a rope
  // override; image tokens consume real KV slots), so no mROPE anchor.
  (void)rope_first;
  if (!_fn_argmax.valid() || (!sp.greedy && !_fn_sample.valid())) {
    return false;
  }
  if (!ensure_scratch_()) { return false; }
  ContextId cm = cm_for_(cid);
  if (!cm.valid()) { return false; }
  if (max_tokens < 1) { max_tokens = 1; }

  PDecode& pd = _pdec[cid.v];
  pd = PDecode{};                 // reset any stale session for this cid
  pd.sp = sp;
  pd.cm = cm;
  pd.produced = 1;                // gen_ids[0] = first_token (already decided)
  pd.committed = 1;               // next forward writes gen_ids[1]
  pd.depth = 2;                   // run-ahead: 2 forwards in flight (lever #2)
  if (const char* e = std::getenv("VPIPE_GEMMA_PDECODE_DEPTH")) {
    pd.depth = std::max(1, std::min(4, std::atoi(e)));
  }
  // +depth headroom so a run-ahead speculative tail near max_tokens still has
  // gen_ids slots (the host stops emitting at max_tokens; the extra slots only
  // hold speculative forwards that pdecode_end rolls back).
  pd.cap = max_tokens + 1 + pd.depth;
  pd.gen_ids = _mc->make_shared_buffer(
      (std::size_t)pd.cap * sizeof(std::int32_t));
  if (pd.gen_ids.empty()) { _pdec.erase(cid.v); return false; }
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
  return true;
}

bool
MetalGemmaModel::pdecode_commit(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return false; }
  PDecode& pd = it->second;
  if ((int)pd.ring.size() >= pd.depth) { return false; }  // pipeline full
  const int in_idx = pd.committed - 1;
  const int out_idx = pd.committed;
  if (out_idx >= pd.cap) { return false; }
  const int kv_off = _ctx->kv_seq_len(pd.cm);
  if (kv_off >= _cfg.max_seq) { return false; }

  const Config& c = _cfg;
  const std::uint64_t s = pd.gpu_step;
  pd.stream.encode_wait(pd.ev, s);
  {
    ComputeEncoder enc = pd.stream.begin_compute();
    // In-stream embed of gen_ids[in_idx] -> _d_x, full decode step -> _d_logits.
    // Each in-flight step writes a distinct page-table ring slot so a depth>1
    // run-ahead doesn't clobber a still-executing step's table.
    const std::size_t pgoff = _full_paged
        ? (std::size_t)(out_idx % kPgtabSlots)
          * (std::size_t)_ctx->max_pages() * 3 * sizeof(std::int32_t)
        : 0;
    encode_step_(enc, pd.cm, kv_off, &pd.gen_ids,
                 (std::size_t)in_idx * sizeof(std::int32_t), pgoff);
    // Sample/argmax the (softcapped) logits into gen_ids[out_idx], on-GPU.
    const std::size_t out_off = (std::size_t)out_idx * sizeof(std::int32_t);
    if (pd.sp.greedy) {
      encode_argmax_(enc, pd.gen_ids, out_off, c.vocab);
    } else {
      const std::uint32_t step_seed = (std::uint32_t)(
          pd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
      encode_sample_(enc, pd.gen_ids, out_off, pd.sample_ws, pd.seen,
                     pd.sp, step_seed, c.vocab);
    }
  }
  pd.stream.encode_signal(pd.ev, s + 1);
  PDecode::InFlight f;
  f.fence = pd.stream.commit();
  f.idx = out_idx;
  pd.ring.push_back(std::move(f));
  // seq_len already advanced inside encode_step_ (append/kv_append); the GPU
  // K/V writes are ordered by the event chain.
  pd.gpu_step = s + 1;
  pd.committed = out_idx + 1;
  return true;
}

std::int32_t
MetalGemmaModel::pdecode_next(ContextId cid)
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
MetalGemmaModel::pdecode_end(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return; }
  PDecode& pd = it->second;
  for (auto& f : pd.ring) { f.fence.wait(); }
  pd.ring.clear();
  // Discard a run-ahead speculative tail: depth>1 may have committed (and
  // KV-appended) forwards past the last token the host drained, so roll the
  // KV back to the last produced token -- matching the synchronous loop,
  // where a stop token's KV is never appended. depth-1 always has
  // committed == produced -> spec 0 (no behavior change).
  const int spec = pd.committed - pd.produced;
  if (spec > 0) { _ctx->kv_rollback(pd.cm, spec); }
  _pdec.erase(it);
}

// Batched prefill: one command buffer of steel GEMMs (M=n) + batched
// attention over the n tokens. Amortises the weight reads across n
// tokens (the per-token forward() loop re-reads every weight per token).
std::vector<float>
MetalGemmaModel::forward_chunk_(ContextId cid,
                               const std::vector<std::int32_t>& ids,
                               const metal_compute::SharedBuffer* mm_emb,
                               const std::vector<int>* mm_pos,
                               bool want_logits)
{
  const Config& c = _cfg;
  const int n = (int)ids.size();
  const int H = c.hidden, Hq = c.n_heads;
  const bool has_ple = c.hpli > 0;
  const int hpli = c.hpli, nl = c.n_layers, ple_w = nl * hpli;
  const int ffn = c.ffn_inner;
  // Scratch K/V is sized for the worst-case head count (n_kv_heads) over the
  // full head_dim -- an upper bound that covers every per-layer (Hkv, D).
  const int qd_max = Hq * c.head_dim_full;
  const int kd_max = c.n_kv_heads * c.head_dim_full;
  const float eps = c.rms_eps;

  ContextId cm = cm_for_(cid);
  if (!cm.valid()) { return {}; }
  const int kv_off = _ctx->kv_seq_len(cm);
  if (kv_off + n > c.max_seq) { return {}; }

  // One-time layer-type census (only while the SDPA probe is active).
  if (std::getenv("VPIPE_GEMMA_SKIP_ATTN") != nullptr) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      int nf = 0, ns = 0;
      for (const auto& ly : _layers) { if (ly.is_full) { ++nf; } else { ++ns; } }
      std::fprintf(stderr, "[gemma-attn-probe] %d full/global (head_dim %d) + "
                   "%d sliding (head_dim %d, window %d) layers\n",
                   nf, c.head_dim_full, ns, c.head_dim_sliding,
                   c.sliding_window);
    }
  }

  // Prefill SDPA kernel selector:
  //   0 = flash (llama.cpp-style Q8/C64, ~2.3x kernel / +6-11% prefill) DEFAULT
  //       GLOBAL (contiguous) + SLIDING (no-wrap, window-skipped) layers
  //   1 = dev   (device-direct, BIT-IDENTICAL to staged, ~1.3x / +4%)
  //       GLOBAL only; sliding stays on staged at this mode
  //   2 = staged (sdpa_causal_mma, the original) GLOBAL + SLIDING
  // Default is `flash`: it's the fastest AND the MOST faithful to the fp32
  // serial reference -- on the qat-q4_0 12B flash is token-IDENTICAL to the
  // scalar reference (gguf_gemma_sdpa_kernels: flash-vs-scalar=0) while staged
  // diverges (staged-vs-scalar=32), because flash keeps the O accumulator + QK
  // scores in fp32 (staged stores the partial-combine + P in half). flash is
  // NOT bit-identical to staged, so the hardcoded metal token-exact baselines
  // on the SAFETENSORS models (kGemma12bGreedyIds, gemma4 token_exact -- 64 GB
  // box only, skip on 16 GB) must be re-confirmed against the Python/MLX oracle
  // with this default:
  //   VPIPE_GEMMA12B_TEST_MODEL_PATH=<dir> vpipe_test --filter '*token_exact*'
  // If a token flips there, flash is the more-correct kernel -> re-dump the
  // oracle (/tmp/gemma12b_text_oracle.py) and re-baseline the table, do NOT
  // revert. `dev` (bit-identical to staged) + `staged` stay available via
  // VPIPE_GEMMA_SDPA for reproducibility/A-B; VPIPE_GEMMA_NO_SDPA_DEV forces
  // staged. See gemma4-12b-bench-results.md S4/S5.
  int sdpa_mode = 0;   // 0=flash(default) 1=dev 2=staged
  if (const char* e = std::getenv("VPIPE_GEMMA_SDPA")) {
    if (std::strcmp(e, "flash") == 0) { sdpa_mode = 0; }
    else if (std::strcmp(e, "dev") == 0) { sdpa_mode = 1; }
    else if (std::strcmp(e, "staged") == 0) { sdpa_mode = 2; }
  }
  if (std::getenv("VPIPE_GEMMA_NO_SDPA_DEV") != nullptr) { sdpa_mode = 2; }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer x = buf((std::size_t)n * H), hn = buf((std::size_t)n * H);
  SharedBuffer q3 = buf((std::size_t)n * qd_max);
  SharedBuffer kb = buf((std::size_t)n * kd_max), vb = buf((std::size_t)n * kd_max);
  // qt padded by MMA_BQ(32) full-head rows: sdpa_causal_mma_f16 loads
  // 8-row Q frags, so the last query tile over-reads past row n.
  SharedBuffer qt = buf((std::size_t)n * qd_max + 32 * qd_max);
  SharedBuffer kt = buf((std::size_t)n * kd_max), vt = buf((std::size_t)n * kd_max);
  SharedBuffer at = buf((std::size_t)n * qd_max), att = buf((std::size_t)n * qd_max);
  SharedBuffer ao = buf((std::size_t)n * H);
  SharedBuffer act = buf((std::size_t)n * ffn), mlp = buf((std::size_t)n * H);
  // Matrix-core geglu intermediate: dequant gate|up -> dense gu_full[rows,
  // 2*ffn], then GeGLU-combine -> act. Only on the matrix-core path.
  SharedBuffer gu_full = _use_mma ? buf((std::size_t)n * 2 * ffn)
                                  : SharedBuffer{};
  SharedBuffer ple = buf((std::size_t)n * ple_w);
  SharedBuffer plepj = buf((std::size_t)n * ple_w), pli = buf((std::size_t)n * ple_w);
  SharedBuffer pli_lm = buf((std::size_t)nl * n * hpli);
  SharedBuffer plg = buf((std::size_t)n * hpli), plp = buf((std::size_t)n * H);
  SharedBuffer toks = _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
  SharedBuffer logits = buf((std::size_t)c.vocab);
  {
    auto* t = static_cast<std::int32_t*>(toks.contents());
    for (int i = 0; i < n; ++i) { t[i] = ids[(std::size_t)i]; }
  }
  // Multimodal: zero the token id at the splice positions so BOTH the
  // main embed (overwritten below) and the PLE gather use id 0 there
  // (matching mlx_vlm gemma4 get_input_embeddings). Build the int32
  // positions buffer the row_scatter reads.
  const int n_mm = (mm_emb != nullptr && mm_pos != nullptr)
      ? (int)mm_pos->size() : 0;
  SharedBuffer mm_pos_buf;
  if (n_mm > 0) {
    auto* t = static_cast<std::int32_t*>(toks.contents());
    mm_pos_buf =
        _mc->make_shared_buffer((std::size_t)n_mm * sizeof(std::int32_t));
    auto* pb = static_cast<std::int32_t*>(mm_pos_buf.contents());
    for (int k = 0; k < n_mm; ++k) {
      const int p = (*mm_pos)[(std::size_t)k];
      if (p >= 0 && p < n) { t[p] = 0; }
      pb[k] = p;
    }
  }

  // Shared-KV tail collapse (prefill n>1): layers [first_shared, nl) borrow
  // an earlier layer's K/V and feed nothing for positions 0..n-2 -- only the
  // last token's logits are kept. Run those layers on the final position
  // alone (42*n -> first_shared*n + tail*1). xs holds that single row, seeded
  // from x[n-1] via an offset residual against the zeroed `zero`. `rows`/`qpos`
  // track the active row count and the final query's global position; they
  // stay n/kv_off until the tail (so decode/no-share are unchanged).
  const int first_shared = c.first_shared();
  SharedBuffer xs   = buf((std::size_t)H);
  SharedBuffer zero = buf((std::size_t)H);
  std::memset(zero.contents(), 0, (std::size_t)H * 2);
  int rows = n;
  int qpos = kv_off;

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function((_rms_fast || Hd > 4096) ? _fn_rms_fast : _fn_rms);
      enc.set_buffer(0, xin, xoff);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd);
      enc.set_constant(4, eps);
      enc.dispatch({512, (unsigned)R, 1}, {512, 1, 1});
    };
    // Fused sandwich-out norm + residual (+ post-scale): res = (res +
    // rms(xin,w)) * post_scale. Tail-only (single row), mirrors decode.
    auto rms_add = [&](const SharedBuffer& xin, const SharedBuffer& w,
                       const SharedBuffer& res, int R, int Hd,
                       float post_scale = 1.0f) {
      enc.set_function((_rms_fast || Hd > 4096) ? _fn_rms_add_fast
                                                : _fn_rms_add);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, res);
      enc.set_buffer(3, res);
      enc.set_constant(4, Hd);
      enc.set_constant(5, eps);
      enc.set_constant(6, post_scale);
      enc.dispatch({512, (unsigned)R, 1}, {512, 1, 1});
    };
    // Fused per-head q-norm + rope (tail single row), at global pos qpos.
    auto rms_rope = [&](const SharedBuffer& xb, const SharedBuffer& w,
                        const SharedBuffer& invf, int heads, int D) {
      enc.set_function(_fn_rms_rope);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, w);
      enc.set_buffer(2, invf);
      enc.set_constant(3, heads);
      enc.set_constant(4, D);
      enc.set_constant(5, eps);
      enc.set_constant(6, qpos);
      enc.dispatch({256, (unsigned)heads, 1}, {256, 1, 1});
    };
    // Matrix-core (M5+) dense GEMM: dequant the 4-bit weight into the reused
    // f16 scratch, then run the tile-adaptive dense matmul2d on the matrix
    // units. y[rows,Nn] = xin[rows,Kk] @ dequant(w)[Nn,Kk]^T (no bias). The
    // M-independent dequant amortises over `rows`; ~2-2.5x steel at prefill.
    auto dense_mma_qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                             const SharedBuffer& b, const SharedBuffer& xin,
                             const SharedBuffer& y, int Kk, int Nn) {
      const std::size_t need = (std::size_t)Nn * Kk * 2;
      if (_w_deq.empty() || _w_deq.byte_size() < need) {
        _w_deq = _mc->make_shared_buffer(need);
      }
      if (!_skip_dequant) {
        enc.set_function(_fn_dequant);
        enc.set_buffer(0, w);
        enc.set_buffer(1, s);
        enc.set_buffer(2, b);
        enc.set_buffer(3, _w_deq);
        enc.set_constant(4, Kk);
        enc.set_constant(5, Nn);
        enc.dispatch({(unsigned)(Kk / 8), (unsigned)Nn, 1}, {64, 1, 1});
      }
      const bool deep = (Kk >= 6144);
      const int BN = deep ? 256 : 128;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, _w_deq);
      enc.set_buffer(2, _w_deq);          // bias slot unused (has_bias=0)
      enc.set_buffer(3, y);
      enc.set_constant(4, Kk);
      enc.set_constant(5, Nn);
      enc.set_constant(6, rows);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((Nn + BN - 1) / BN) * 256),
                    (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
    };
    // qmm_fn selects the matvec (rows==1) / matmul kernel pair. `qmm`
    // defaults to the base bits (attn / lm_head); down_proj passes the _mlp
    // pair (8-bit for gemma4_unified).
    auto qmm_fn = [&](const auto& fn_mv, const auto& fn_mm,
                      const SharedBuffer& w, const SharedBuffer& s,
                      const SharedBuffer& b, const SharedBuffer& xin,
                      const SharedBuffer& y, int Kk, int N) {
      if (rows == 1) {
        // Single row (shared-KV tail): the matvec kernel beats the matmul,
        // which computes a 32-row tile for 1 valid row.
        enc.set_function(fn_mv);
        enc.set_buffer(0, w);
        enc.set_buffer(1, s);
        enc.set_buffer(2, b);
        enc.set_buffer(3, xin);
        enc.set_buffer(4, y);
        enc.set_constant(5, Kk);
        enc.set_constant(6, N);
        enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
        return;
      }
      if (_use_mma && rows >= _mma_min_m) {
        dense_mma_qmm(w, s, b, xin, y, Kk, N);
        return;
      }
      enc.set_function(fn_mm);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, Kk);
      enc.set_constant(6, N);
      enc.set_constant(7, rows);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((rows + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                   const SharedBuffer& b, const SharedBuffer& xin,
                   const SharedBuffer& y, int Kk, int N) {
      qmm_fn(_fn_qmv, _fn_qmm, w, s, b, xin, y, Kk, N);
    };
    auto tr = [&](const SharedBuffer& in, const SharedBuffer& out, int A,
                  int Bd, int D) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A}, {(unsigned)D, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb, const SharedBuffer& invf,
                    int heads, int D) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, invf);
      enc.set_constant(2, heads);
      enc.set_constant(3, rows);       // T = active rows
      enc.set_constant(4, D);
      enc.set_constant(5, qpos);       // global pos of first active row
      enc.dispatch({(unsigned)(D / 2), (unsigned)rows, (unsigned)heads},
                   {(unsigned)(D / 2 < 256 ? D / 2 : 256), 1, 1});
    };
    auto geglu = [&](const SharedBuffer& g, const SharedBuffer& u,
                     std::size_t uoff, const SharedBuffer& o, int nn) {
      enc.set_function(_fn_geglu);
      enc.set_buffer(0, g);
      enc.set_buffer(1, u, uoff);
      enc.set_buffer(2, o);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, std::size_t aoff,
                        const SharedBuffer& bb, const SharedBuffer& o,
                        std::size_t ooff, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a, aoff);
      enc.set_buffer(1, bb);
      enc.set_buffer(2, o, ooff);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto scale = [&](const SharedBuffer& xb, int nn, float s) {
      enc.set_function(_fn_scale);
      enc.set_buffer(0, xb);
      enc.set_constant(1, nn);
      enc.set_constant(2, s);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto dense = [&](const SharedBuffer& xin, const SharedBuffer& w,
                     const SharedBuffer& y, int Kk, int N) {
      // The only dense f16 GEMM in prefill (per_layer_model_projection,
      // [rows,H] @ [n_layers*hpli,H]^T). Route it onto the matrix units on M5
      // (K=H=2560 <= 4096 -> n128) like the quantized proj/FFN GEMMs; the steel
      // path stays for M4 / non-matrix-core. Same NT contract, f32 accumulate.
      static const bool kPleSteel =
          std::getenv("VPIPE_GEMMA_PLE_STEEL") != nullptr;
      if (_use_mma && rows >= _mma_min_m && !kPleSteel) {
        enc.set_function(_fn_dense_mma);          // 128x128, SG=8
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);                     // bias slot unused
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, rows);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + 127) / 128) * 256),
                      (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
        return;
      }
      enc.set_function(_fn_dense_t);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);
      enc.set_buffer(3, y);
      enc.set_constant(4, Kk);
      enc.set_constant(5, N);
      enc.set_constant(6, rows);
      const int hb = 0;
      enc.set_constant(7, hb);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((rows + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto embed_gather = [&](const SharedBuffer& w, const SharedBuffer& s,
                            const SharedBuffer& b, const SharedBuffer& o,
                            int Hd) {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, toks);
      enc.set_buffer(1, w);
      enc.set_buffer(2, s);
      enc.set_buffer(3, b);
      enc.set_buffer(4, o);
      enc.set_constant(5, Hd);
      enc.dispatch({(unsigned)Hd, (unsigned)n, 1}, {256, 1, 1});
    };
    auto kvw = [&](const SharedBuffer& src, const SharedBuffer& cache,
                   int D, int cap, int Hkv, int ring_cap, int window) {
      enc.set_function(_fn_kv_write);
      enc.set_buffer(0, src);
      enc.set_buffer(1, cache);
      enc.set_constant(2, cap);              // physical head stride
      enc.set_constant(3, D);
      enc.set_constant(4, n);
      enc.set_constant(5, kv_off);
      enc.set_constant(6, ring_cap);         // ring modulo (0 = linear)
      enc.set_constant(7, window);           // trailing window (mirror)
      enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)Hkv},
                   {(unsigned)D, 1, 1});
    };
    // ---- embeddings + per-layer inputs (batched) -------------------
    if (_embed_is_q6k) {
      enc.set_function(_fn_embed_q6k);
      enc.set_buffer(0, toks);
      enc.set_buffer(1, _embed_q6k);
      enc.set_buffer(2, x);
      enc.set_constant(3, H);
      enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
    } else {
      embed_gather(_embed_w, _embed_s, _embed_b, x, H);
    }
    scale(x, n * H, std::sqrt((float)H));
    // Multimodal splice: overlay the (un-scaled) encoder rows into x at
    // the mm positions. PLE ids were zeroed above, so the PLE gather
    // below picks up id 0 there.
    if (n_mm > 0) {
      enc.set_function(_fn_row_scatter);
      enc.set_buffer(0, x);
      enc.set_buffer(1, *mm_emb);
      enc.set_buffer(2, mm_pos_buf);
      enc.set_constant(3, H);
      enc.set_constant(4, n_mm);
      enc.dispatch({(unsigned)(n_mm * H), 1, 1}, {256, 1, 1});
    }
    if (has_ple) {
      embed_gather(_ple_w, _ple_s, _ple_b, ple, ple_w);
      scale(ple, n * ple_w, std::sqrt((float)hpli));
      dense(x, _plm_proj_w, plepj, H, ple_w);          // BF16 [n, nl*hpli]
      scale(plepj, n * ple_w, std::pow((float)H, -0.5f));
      rms(plepj, 0, _ple_proj_norm, plepj, 0, n * nl, hpli);
      residual(plepj, 0, ple, pli, 0, n * ple_w);
      scale(pli, n * ple_w, 0.70710678f);
      tr(pli, pli_lm, n, nl, hpli);                    // [n,nl,hpli]->[nl,n,hpli]
    }

    // FULL (global) layers are PAGED: reserve this chunk's n tokens of pool
    // pages ONCE (shared by every full layer), recording the per-page write
    // segments {page,slot,src_off,cnt}, and build the page table. append()
    // advances seq_len, so the tail kv_append is skipped for the paged path.
    const int page_tokens = _ctx->page_tokens();
    struct FSeg { PageId pid; int slot; int src_off; int cnt; };
    std::vector<FSeg> fsegs;
    int n_pages = 0;
    if (_full_paged) {
      int rem = n, src = 0;
      while (rem > 0) {
        const int capp = _ctx->next_append_capacity(cm);
        const int cnt = std::min(rem, capp);
        ContextManager::AppendSlot s = _ctx->append(cm, cnt);
        if (!s.valid()) { return {}; }
        fsegs.push_back({s.page_id, s.slot_offset, src, cnt});
        src += cnt;
        rem -= cnt;
      }
      n_pages = _ctx->fill_page_table(
          cm, static_cast<std::int32_t*>(_d_pgtab.contents()));
    }

    // ---- layer loop ------------------------------------------------
    const SharedBuffer* xcur = &x;     // current hidden buffer
    for (int L = 0; L < nl; ++L) {
      // KV-only intermediate chunk: the shared-KV tail (L >= first_shared)
      // writes no K/V and its residual/logits are discarded, so stop after
      // the own-KV bulk layers. The bulk layers (incl. their PLE gate) must
      // still run -- they populate the cache the later chunks' tail reads.
      if (!want_logits && L >= first_shared) { break; }
      Layer& ly = _layers[L];
      const int Hkv = ly.n_kv;
      const int D = ly.head_dim, qd = Hq * D, kd = Hkv * D;
      const SharedBuffer& invf = ly.is_full ? _inv_freq_full : _inv_freq_sliding;

      if (L == first_shared && n > 1) {
        // Enter the shared-KV tail: seed xs from x's last row, then run the
        // rest of the stack on that single query.
        residual(x, (std::size_t)(n - 1) * H * 2, zero, xs, 0, H);
        xcur = &xs;
        rows = 1;
        qpos = kv_off + n - 1;
      }

      rms(*xcur, 0, ly.in_ln, hn, 0, rows, H);
      qmm(ly.qw, ly.qs, ly.qb, hn, q3, H, qd);
      // q-norm + rope. Tail (1 row): fused rms_rope in place on q3, already
      // [Hq,D] head-major (== decode q layout); attention reads q3. Bulk:
      // separate norm, transpose to [Hq,rows,D], rope; attention reads qt.
      const SharedBuffer* qsrc;
      if (rows == 1) {
        rms_rope(q3, ly.q_norm, invf, Hq, D);
        qsrc = &q3;
      } else {
        rms(q3, 0, ly.q_norm, q3, 0, rows * Hq, D);
        tr(q3, qt, rows, Hq, D);
        rope(qt, invf, Hq, D);
        qsrc = &qt;
      }

      const bool paged_full = ly.is_full && _full_paged;
      // Full layers: the SHARED page pool (kpool/vpool follow kv_source).
      // Sliding layers: the per-context contiguous ring.
      const SharedBuffer* Kuse =
          paged_full ? _ctx->kpool(L) : _ctx->kv_k(cm, L);
      const SharedBuffer* Vuse =
          paged_full ? _ctx->vpool(L) : _ctx->kv_v(cm, L);
      // Physical K/V capacity (max_seq for full, bounded ring for sliding, or
      // this context's GROWN ring) + ring modulo (0 = linear addressing).
      // Unused for paged-full layers (page-table addressed).
      const int cap = paged_full ? 0 : _ctx->kv_capacity(cm, L);
      const int ring_cap = paged_full ? 0 : _ctx->kv_ring_cap(cm, L);
      if (ly.kv_source < 0) {
        qmm(ly.kw, ly.ks, ly.kb, hn, kb, H, kd);
        // values: k_eq_v full layers reuse the k_proj output (no v_proj),
        // v_norm'd BEFORE k is normed in place. Else project v separately.
        if (ly.k_eq_v) {
          rms(kb, 0, _ones_vnorm, vb, 0, n * Hkv, D);   // vb = v_norm(k)
        } else {
          qmm(ly.vw, ly.vs, ly.vb, hn, vb, H, kd);
          rms(vb, 0, _ones_vnorm, vb, 0, n * Hkv, D);   // v_norm (no wt)
        }
        rms(kb, 0, ly.k_norm, kb, 0, n * Hkv, D);
        tr(kb, kt, n, Hkv, D);
        tr(vb, vt, n, Hkv, D);
        rope(kt, invf, Hkv, D);
        if (paged_full) {
          // Paged write: scatter the chunk's [Hkv,n,D] K/V into the reserved
          // pool pages (one kv_write_paged dispatch per page segment).
          auto wpp = [&](const SharedBuffer& src, const SharedBuffer& pool) {
            for (const FSeg& sg : fsegs) {
              const std::size_t poff =
                  (std::size_t)sg.pid.v * _ctx->page_stride_bytes();
              enc.set_function(_fn_kv_write_paged);
              enc.set_buffer(0, src);
              enc.set_buffer(1, pool, poff);
              enc.set_constant(2, page_tokens);
              enc.set_constant(3, D);
              enc.set_constant(4, n);            // n_src (source seq stride)
              enc.set_constant(5, sg.src_off);
              enc.set_constant(6, sg.slot);      // dst_slot
              enc.dispatch({(unsigned)D, (unsigned)sg.cnt, (unsigned)Hkv},
                           {(unsigned)D, 1, 1});
            }
          };
          wpp(kt, *Kuse);
          wpp(vt, *Vuse);
        } else {
          const int kvw_win = ly.is_full ? 0 : c.sliding_window;
          // Bounded-ring single-pass prefill: when the write spans MORE rows
          // than the ring holds (kv_off+n > ring_cap), a single kv_write
          // dispatch would map multiple source rows to the same physical slot
          // (pos % ring_cap collisions). Those parallel writes race -> the ring
          // contents are non-deterministic. Sub-block the write to <= page
          // (= ring_cap - window) rows so each dispatch touches each slot at
          // most once; the last sub-block leaves the correct trailing window.
          const bool wrap_write = ring_cap > 0
              && (kv_off + n > ring_cap) && _fn_kv_write_sub.valid();
          if (wrap_write) {
            const int page = std::max(1, ring_cap - kvw_win);
            auto kvw_sub = [&](const SharedBuffer& src,
                               const SharedBuffer& cache) {
              for (int s0 = 0; s0 < n; s0 += page) {
                const int cnt = std::min(page, n - s0);
                enc.set_function(_fn_kv_write_sub);
                enc.set_buffer(0, src);
                enc.set_buffer(1, cache);
                enc.set_constant(2, cap);            // physical head stride
                enc.set_constant(3, D);
                enc.set_constant(4, n);              // source head stride
                enc.set_constant(5, kv_off);         // base position
                enc.set_constant(6, ring_cap);       // ring modulo
                enc.set_constant(7, kvw_win);        // window (mirror)
                enc.set_constant(8, s0);             // source start row
                enc.dispatch({(unsigned)D, (unsigned)cnt, (unsigned)Hkv},
                             {(unsigned)D, 1, 1});
              }
            };
            kvw_sub(kt, *Kuse);
            kvw_sub(vt, *Vuse);
          } else {
            kvw(kt, *Kuse, D, cap, Hkv, ring_cap, kvw_win);
            kvw(vt, *Vuse, D, cap, Hkv, ring_cap, kvw_win);
          }
        }
      }

      const float one_scale = 1.0f;
      const int T_kv = kv_off + n;
      // Attention. Default = the simdgroup_matrix (MMA) flash kernel
      // (sdpa_causal_mma_f16) -- QK^T/PV as 8x8x8 matmuls, the only part of
      // prefill that was slower than MLX (now beats/matches it). Runs for
      // BOTH f16 and bf16 (bfloat MMA operands, fp32 accumulation -- same as
      // the steel GEMM). Falls back to the per-query scalar kernel only when
      // D%64!=0; VPIPE_GEMMA_SCALAR_ATTN forces the scalar path (A/B).
      const bool force_scalar =
          std::getenv("VPIPE_GEMMA_SCALAR_ATTN") != nullptr;
      const bool use_mma = !force_scalar && (D % 64 == 0);
      const int window = ly.is_full ? 0 : c.sliding_window;
      // A/B probe (VPIPE_GEMMA_SKIP_ATTN): skip the SDPA dispatch to measure
      // attention's share of prefill. The attn-output buffer is left stale
      // (logits become garbage -- timing only); every matmul still runs, so
      // the with/without delta is pure SDPA kernel time. Modes:
      //   1 (or "all")     -- skip every layer's SDPA
      //   2 (or "global")  -- skip only the full/global layers (head_dim 512,
      //                       O(n^2)); leaves sliding-window SDPA running
      //   3 (or "sliding") -- skip only the sliding-window layers (O(n*win))
      // Modes 2/3 attribute attention time per layer type (gemma-4 has ~8
      // global + ~40 sliding layers, so the two costs scale very differently).
      static const int kSkipMode = []() {
        const char* e = std::getenv("VPIPE_GEMMA_SKIP_ATTN");
        if (e == nullptr || *e == '\0') { return 0; }
        if (std::strcmp(e, "global") == 0 || std::atoi(e) == 2) { return 2; }
        if (std::strcmp(e, "sliding") == 0 || std::atoi(e) == 3) { return 3; }
        return 1;
      }();
      const bool skip_attn =
          kSkipMode == 1 || (kSkipMode == 2 && ly.is_full)
          || (kSkipMode == 3 && !ly.is_full);

      // Causal-tiled dense GEMM available (skip strictly-upper-triangle work,
      // and the below-window region for sliding). Default ON when both entry
      // points are loaded; opt out with VPIPE_GEMMA_MAT_CAUSAL=0.
      static const bool kMatCausal = [] {
        const char* e = std::getenv("VPIPE_GEMMA_MAT_CAUSAL");
        return !(e && std::atoi(e) == 0);
      }();
      // MATERIALIZED attention (steel GEMM Q.K^T -> windowed causal softmax ->
      // P.V), the MLX head_dim-512 fallback shape, extended to BANDED for the
      // sliding layers. window_mat=0 -> pure causal (global); window_mat>0 ->
      // the Gemma-4 trailing band. Single-chunk only (qpos==0): kt/vt hold the
      // full [Hkv,T_kv,D] chunk so they ARE the whole prefix (T_kv==rows).
      // Reads qt/kt/vt, writes at. Token-exact with the flash/pflash path.
      auto materialized_attn = [&](int window_mat) {
        auto off = [](std::size_t e) { return e * sizeof(uint16_t); };
        const bool use_causal = kMatCausal && _fn_dense_t_qkcausal.valid()
            && _fn_dense_t_pvcausal.valid();
        // Steel dense_t(x[M,K], W[N,K]) = x @ W^T -> y[M,N]; M == rows.
        auto dense_g = [&](const metal_compute::ComputeFunction& fn,
                           const SharedBuffer& xin, std::size_t xoff,
                           const SharedBuffer& w, std::size_t woff,
                           const SharedBuffer& y, std::size_t yoff,
                           int Kk, int Nn) {
          enc.set_function(fn);
          enc.set_buffer(0, xin, off(xoff));
          enc.set_buffer(1, w, off(woff));
          enc.set_buffer(2, w, off(woff));
          enc.set_buffer(3, y, off(yoff));
          enc.set_constant(4, Kk);
          enc.set_constant(5, Nn);
          enc.set_constant(6, rows);
          const int hb = 0;
          enc.set_constant(7, hb);
          enc.set_constant(8, qpos);          // q_offset (causal variants)
          enc.set_constant(9, window_mat);    // trailing window (banded)
          enc.dispatch({(unsigned)(((Nn + 31) / 32) * 32),
                        (unsigned)(((rows + 31) / 32) * 2), 2}, {32, 2, 2});
        };
        // Matrix-core (M5) materialized QK: causal-tiled matmul2d (the
        // diagonal-grid skip on the matrix units). Only QK moves to the matrix
        // units; the softmax stays banded and PV stays steel-causal (K-capped) --
        // a full-K matmul2d PV loses at 16k (2x FLOPs) where PV dominates, and
        // the matmul2d causal K-cap would force a tiny BM that starves the matrix
        // units. The QK tile-skip (128-wide) writes a SUPERSET of the softmax/PV
        // 32-row band, so the banded consumers read only written keys.
        const bool use_mat_mma = _mat_mma;
        auto dense_mma_qk = [&](const SharedBuffer& xin, std::size_t xoff,
                                const SharedBuffer& w, std::size_t woff,
                                const SharedBuffer& y, int Kk, int Nn) {
          enc.set_function(_fn_dense_mma_qkcausal);    // 128x128, SG=8
          enc.set_buffer(0, xin, off(xoff));
          enc.set_buffer(1, w, off(woff));
          enc.set_buffer(2, w, off(woff));             // bias slot unused
          enc.set_buffer(3, y);
          enc.set_constant(4, Kk);                     // K = D
          enc.set_constant(5, Nn);                     // N = T_kv
          enc.set_constant(6, rows);                   // M
          enc.set_constant(7, 0);                      // has_bias
          enc.set_constant(8, qpos);                   // q_offset
          enc.set_constant(9, window_mat);             // trailing window
          enc.dispatch({(unsigned)(((Nn + 127) / 128) * 256),
                        (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
        };
        // V transpose per kv-head: vt[kvh] [T_kv,D] -> _d_vT[kvh] [D,T_kv].
        for (int kvh = 0; kvh < Hkv; ++kvh) {
          enc.set_function(_fn_transpose);
          enc.set_buffer(0, vt, off((std::size_t)kvh * T_kv * D));
          enc.set_buffer(1, _d_vT, off((std::size_t)kvh * D * T_kv));
          enc.set_constant(2, T_kv);          // A
          enc.set_constant(3, D);             // Bd
          const int one = 1;
          enc.set_constant(4, one);           // D (last dim)
          // 256-thread groups (32 along Bd for coalesced in[a*D+b] reads) --
          // the old {1,1,1} launched D*T_kv single-thread groups (8.4M @16k).
          enc.dispatch({1u, (unsigned)D, (unsigned)T_kv}, {1u, 32u, 8u});
        }
        const int Gq = Hq / Hkv;
        for (int h = 0; h < Hq; ++h) {
          const int kvh = h / Gq;
          // QK^T: scores[rows,T_kv] = Qh[rows,D] @ Kkvh[T_kv,D]^T. Causal/
          // banded variant skips the tiles softmax would mask away.
          if (use_mat_mma) {
            dense_mma_qk(qt, (std::size_t)h * rows * D,
                         kt, (std::size_t)kvh * T_kv * D,
                         _d_scores, /*Kk=*/D, /*Nn=*/T_kv);
          } else {
            dense_g(use_causal ? _fn_dense_t_qkcausal : _fn_dense_t,
                    qt, (std::size_t)h * rows * D,
                    kt, (std::size_t)kvh * T_kv * D,
                    _d_scores, 0, /*Kk=*/D, /*Nn=*/T_kv);
          }
          // windowed causal softmax in place on scores[rows,T_kv].
          enc.set_function(_fn_causal_softmax);
          enc.set_buffer(0, _d_scores);
          enc.set_constant(1, rows);          // M
          enc.set_constant(2, T_kv);          // N
          enc.set_constant(3, qpos);          // q_offset
          enc.set_constant(4, one_scale);
          enc.set_constant(5, window_mat);    // trailing window
          const int banded = use_causal ? 1 : 0;
          enc.set_constant(6, banded);        // band-limit to PV's [k0,Kc)
          enc.dispatch({256u, (unsigned)rows, 1u}, {256u, 1, 1});
          // PV: O_h[rows,D] = P[rows,T_kv] @ V_T[D,T_kv]^T. Steel banded variant
          // contracts only the in-band keys.
          dense_g(use_causal ? _fn_dense_t_pvcausal : _fn_dense_t,
                  _d_scores, 0,
                  _d_vT, (std::size_t)kvh * D * T_kv,
                  at, (std::size_t)h * rows * D, /*Kk=*/T_kv, /*Nn=*/D);
        }
      };
      // Eligibility: single-chunk prefill, scores fit the scratch cap.
      const bool mat_ok = rows > 1 && _materialized_global && qpos == 0
          && T_kv <= _scores_cap && _fn_causal_softmax.valid();

      if (skip_attn) {
        // intentionally no SDPA dispatch (probe)
      } else if (paged_full) {
        // PAGED full-layer attention over the shared pool + page table. The
        // bulk (n_q>1) uses the matrix-core paged mma2 (M5) or, on a non-
        // matrix-core GPU (M4), the simdgroup_matrix paged flash; only the
        // shared-KV tail (n_q=1) falls to the scalar paged kernel (q3/qt match
        // the q[Hq,n_q,D] layout all three read).
        if (mat_ok && ly.is_full && D == 512) {
          // MATERIALIZED global prefill (window=0: pure causal). _full_paged
          // still wrote K/V to the pool above (for later decode); we attend
          // over the chunk-local kt/vt, which ARE the whole prefix at qpos==0.
          materialized_attn(/*window=*/0);
        } else if (rows > 1 && _pmma2_attn) {
          enc.set_function(_fn_sdpa_pmma2);
          enc.set_buffer(0, *qsrc);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, D);
          enc.set_constant(6, Hq);
          enc.set_constant(7, Hkv);
          enc.set_constant(8, rows);           // n_q
          enc.set_constant(9, qpos);           // q_offset
          enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_pages);
          enc.set_buffer(12, _d_pgtab);
          enc.dispatch({128, (unsigned)Hq, (unsigned)((rows + 7) / 8)},
                       {128, 1, 1});
        } else if (rows > 1 && _pflash_attn && D == 512) {
          // M4 / non-matrix-core bulk prefill: simdgroup_matrix paged flash
          // (D=512), same buffer contract as pmma2. FL_NSG*32=256 threads,
          // FL_Q=8 query rows/threadgroup. Without this tier the M4 bulk
          // global paged prefill fell to the scalar sdpa_paged_causal below.
          enc.set_function(_fn_sdpa_pflash);
          enc.set_buffer(0, *qsrc);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, D);
          enc.set_constant(6, Hq);
          enc.set_constant(7, Hkv);
          enc.set_constant(8, rows);           // n_q
          enc.set_constant(9, qpos);           // q_offset
          enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_pages);
          enc.set_buffer(12, _d_pgtab);
          enc.dispatch({256, (unsigned)Hq, (unsigned)((rows + 7) / 8)},
                       {256, 1, 1});
        } else {
          enc.set_function(_fn_sdpa_paged_causal);
          enc.set_buffer(0, *qsrc);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, D);
          enc.set_constant(6, Hq);
          enc.set_constant(7, Hkv);
          enc.set_constant(8, rows);           // n_q
          enc.set_constant(9, qpos);           // q_offset
          enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_pages);
          enc.set_buffer(12, _d_pgtab);
          enc.dispatch({32, (unsigned)Hq, (unsigned)rows}, {32, 1, 1});
        }
      } else if (rows == 1) {
        // Shared-KV tail: a single query. The MMA/scalar prefill kernels
        // process a full 32-row Q-tile (31 wasted), so use the decode
        // multi-simdgroup kernel (sdpa_mb): BN simdgroups split the key scan.
        // qt at rows==1 is [Hq, D] head-major == the decode q layout.
        const unsigned bn = (unsigned)(4096 / D);
        enc.set_function(_fn_sdpa_mb);
        enc.set_buffer(0, *qsrc);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, at);
        enc.set_constant(4, one_scale);
        enc.set_constant(5, T_kv);
        enc.set_constant(6, D);
        enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv);
        enc.set_constant(9, rows);           // n_q == 1
        enc.set_constant(10, qpos);          // q_offset
        enc.set_constant(11, cap);
        enc.set_constant(12, window);
        enc.set_constant(13, ring_cap);
        enc.dispatch({bn * 32, (unsigned)Hq, 1}, {bn * 32, 1, 1});
      } else if (use_mma) {
        // Fast-path selection: flash > device-direct > staged. Flash covers
        // BOTH global (contiguous, ring_cap==0) AND sliding layers in the
        // no-wrap regime -- for bounded sliding layers cap == ring_cap, so
        // kv_off+n+64 <= cap already implies the ring has not wrapped
        // (logical key == physical slot, the contiguous addressing flash
        // uses) and the C=64 over-read lands in the in-bounds cache tail.
        // Once the ring wraps (kv_off+n+64 > cap), sliding falls to staged;
        // device-direct stays global-only (it lacks window-base skip).
        const bool contig = (ring_cap == 0);
        // mma2 reads K/V in BK=16 blocks; the last block of an n_q tile reaches
        // physical slot roundup16(kv_off+n). The kernel is safe (and the ring
        // has not wrapped) as long as that lands within cap. cap is 16-aligned
        // (max_seq / ring_cap), so this fires mma2 right up to n == max_seq --
        // the exact 4k boundary the old "+16 slack" guard needlessly excluded
        // (where the 40 sliding layers fell back to ALU scalar SDPA).
        const int kv_end16 = ((kv_off + n + 15) / 16) * 16;
        // Matrix-core matmul2d attention for the GLOBAL (head_dim 512, full-
        // causal) layers: QK^T/PV on the M5 matrix units, ~1.7-1.9x the
        // simdgroup_matrix flash kernel (the prefill attention is the 12B gap).
        // VPIPE_GEMMA_NO_MMA2_ATTN=1 reverts to flash.
        const bool mma2_ok = _mma2_attn && contig && (window <= 0)
            && (D == 512) && (kv_end16 <= cap);
        // matmul2d for the SLIDING (head_dim 256, windowed) layers in the
        // no-wrap regime (kv_end16 <= cap means the ring has not wrapped, so
        // linear addressing is valid -- same condition flash relies on).
        const bool mma2s_ok = _mma2_attn && (window > 0) && (D == 256)
            && _fn_sdpa_mma2_d256.valid() && (kv_end16 <= cap);
        const bool flash_ok = sdpa_mode == 0
            && _fn_sdpa_flash.valid() && (D % 64 == 0) && (D <= 512)
            && (kv_off + n + 64 <= cap);     // no-wrap + C=64 over-read slack
        const bool dev_ok = contig && sdpa_mode != 2
            && _fn_sdpa_mma_dev.valid() && (kv_off + n + 32 <= cap);
        // Materialized BANDED sliding prefill (head_dim 256): steel GEMM runs
        // far above the simdgroup flash on M4 (no matrix cores). Reads the
        // chunk-local kt/vt (qpos==0 single chunk), so it is independent of the
        // ring cap/wrap. Default ON; VPIPE_GEMMA_MAT_SLIDING=0 reverts to flash.
        // _mat_sliding is the construction-time member (prefill()'s bounded
        // single-pass path keys off the SAME value so it never routes sliding
        // to a wrapped-ring read when the materialized path is off).
        const bool kMatSliding = _mat_sliding;
        const bool mat_sliding = kMatSliding && mat_ok && window > 0
            && D == 256 && _fn_dense_t_qkcausal.valid()
            && _fn_dense_t_pvcausal.valid();
        if (mat_sliding) {
          materialized_attn(window);
        } else if (mma2_ok) {
          enc.set_function(_fn_sdpa_mma2);
          enc.set_buffer(0, qt);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, T_kv);
          enc.set_constant(6, D);
          enc.set_constant(7, Hq);
          enc.set_constant(8, Hkv);
          enc.set_constant(9, rows);           // n_q
          enc.set_constant(10, qpos);          // q_offset
          enc.set_constant(11, cap);           // kv_stride (head stride = cap)
          enc.set_constant(12, window);
          enc.set_constant(13, ring_cap);
          // grid {SA2_SG*32=128, Hq, ceil(rows/SA2_BQ=8)}, tg {128,1,1}.
          enc.dispatch({128, (unsigned)Hq, (unsigned)((rows + 7) / 8)},
                       {128, 1, 1});
        } else if (mma2s_ok) {
          enc.set_function(_fn_sdpa_mma2_d256);
          enc.set_buffer(0, qt);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, T_kv);
          enc.set_constant(6, D);
          enc.set_constant(7, Hq);
          enc.set_constant(8, Hkv);
          enc.set_constant(9, rows);           // n_q
          enc.set_constant(10, qpos);          // q_offset
          enc.set_constant(11, cap);           // kv_stride (head stride = cap)
          enc.set_constant(12, window);
          enc.set_constant(13, ring_cap);
          // grid {SA3_SG*32=128, Hq, ceil(rows/SA3_BQ=16)}, tg {128,1,1}.
          enc.dispatch({128, (unsigned)Hq, (unsigned)((rows + 15) / 16)},
                       {128, 1, 1});
        } else if (flash_ok) {
          // Flash kernel: Q=8 rows/tg, NSG=8 simdgroups (256 threads).
          enc.set_function(_fn_sdpa_flash);
          enc.set_buffer(0, qt);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, T_kv);
          enc.set_constant(6, D);
          enc.set_constant(7, Hq);
          enc.set_constant(8, Hkv);
          enc.set_constant(9, rows);
          enc.set_constant(10, qpos);
          enc.set_constant(11, cap);
          enc.set_constant(12, window);
          enc.set_constant(13, ring_cap);
          enc.dispatch({256, (unsigned)Hq, (unsigned)((rows + 7) / 8)},
                       {256, 1, 1});
        } else {
          const unsigned tg = (unsigned)(4 * (D / 64) * 32);   // WM*WD*32
          enc.set_function(dev_ok ? _fn_sdpa_mma_dev : _fn_sdpa_mma);
          enc.set_buffer(0, qt);
          enc.set_buffer(1, *Kuse);
          enc.set_buffer(2, *Vuse);
          enc.set_buffer(3, at);
          enc.set_constant(4, one_scale);
          enc.set_constant(5, T_kv);
          enc.set_constant(6, D);
          enc.set_constant(7, Hq);
          enc.set_constant(8, Hkv);
          enc.set_constant(9, rows);           // n_q
          enc.set_constant(10, qpos);          // q_offset
          enc.set_constant(11, cap);           // kv_stride (head stride = cap)
          enc.set_constant(12, window);
          enc.set_constant(13, ring_cap);      // ring modulo (0 = linear)
          enc.dispatch({tg, (unsigned)Hq, (unsigned)((rows + 31) / 32)},
                       {tg, 1, 1});
        }
      } else if (ly.is_full) {
        // Full layers never ring (cap == max_seq); sdpa_causal_f16 has no
        // ring param.
        enc.set_function(_fn_sdpa_causal);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, at);
        enc.set_constant(4, one_scale);
        enc.set_constant(5, T_kv);
        enc.set_constant(6, D);
        enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv);
        enc.set_constant(9, rows);         // n_q
        enc.set_constant(10, qpos);        // q_offset
        enc.set_constant(11, cap);
        enc.dispatch({32, (unsigned)Hq, (unsigned)rows}, {32, 1, 1});
      } else {
        enc.set_function(_fn_sdpa_window);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, *Kuse);
        enc.set_buffer(2, *Vuse);
        enc.set_buffer(3, at);
        enc.set_constant(4, one_scale);
        enc.set_constant(5, T_kv);
        enc.set_constant(6, D);
        enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv);
        enc.set_constant(9, rows);
        enc.set_constant(10, qpos);
        enc.set_constant(11, cap);
        enc.set_constant(12, c.sliding_window);
        enc.set_constant(13, ring_cap);      // ring modulo (0 = linear)
        enc.dispatch({32, (unsigned)Hq, (unsigned)rows}, {32, 1, 1});
      }
      // o-proj input: tail sdpa_mb wrote `at` as [Hq,D] == [1,qd] (feed it
      // straight in); bulk transposes [Hq,rows,D] -> [rows,Hq,D].
      const SharedBuffer* attn_out;
      if (rows == 1) {
        attn_out = &at;
      } else {
        tr(at, att, Hq, rows, D);
        attn_out = &att;
      }
      qmm(ly.ow, ly.os, ly.ob, *attn_out, ao, qd, H);
      if (rows == 1) {
        rms_add(ao, ly.post_attn_ln, *xcur, rows, H);
      } else {
        rms(ao, 0, ly.post_attn_ln, ao, 0, rows, H);
        residual(*xcur, 0, ao, *xcur, 0, rows * H);
      }

      // MLP (geglu). Fused gate/up GEMM + GeGLU: one GEMM over the
      // interleaved [2*ffn, H] weight writes gelu(gate)*up straight to act
      // [n, ffn] (no separate gate/up buffers or geglu pass).
      rms(*xcur, 0, ly.pre_ffn_ln, hn, 0, rows, H);
      if (rows == 1) {
        enc.set_function(_fn_qmv_geglu);
        enc.set_buffer(0, ly.guw);
        enc.set_buffer(1, ly.gus);
        enc.set_buffer(2, ly.gub);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, act);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * ffn);
        enc.dispatch({32, (unsigned)(ffn / 2), 1}, {32, 2, 1});
      } else if (_use_mma && rows >= _mma_min_m) {
        // Matrix-core geglu: dequant the interleaved gate|up weight -> dense
        // matmul2d -> gu_full[rows, 2*ffn], then GeGLU-combine (gelu(gate)*up
        // over the interleaved columns) -> act[rows, ffn]. K=H (<=4096) so
        // dense_mma_qmm picks the 128x128 tile.
        dense_mma_qmm(ly.guw, ly.gus, ly.gub, hn, gu_full, H, 2 * ffn);
        enc.set_function(_fn_geglu_inter);
        enc.set_buffer(0, gu_full);
        enc.set_buffer(1, act);
        enc.set_constant(2, rows);
        enc.set_constant(3, ffn);
        enc.dispatch({(unsigned)(rows * ffn), 1, 1}, {256, 1, 1});
      } else {
        enc.set_function(_fn_qmm_geglu);
        enc.set_buffer(0, ly.guw);
        enc.set_buffer(1, ly.gus);
        enc.set_buffer(2, ly.gub);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, act);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * ffn);
        enc.set_constant(7, rows);
        enc.dispatch({(unsigned)(((2 * ffn + 31) / 32) * 32),
                      (unsigned)(((rows + 31) / 32) * 2), 2}, {32, 2, 2});
      }
      // down_proj at mlp_bits (8-bit for gemma4_unified).
      qmm_fn(_fn_qmv_mlp, _fn_qmm_mlp, ly.dw, ly.ds, ly.db, act, mlp, ffn, H);
      if (rows == 1) {
        rms_add(mlp, ly.post_ffn_ln, *xcur, rows, H);
      } else {
        rms(mlp, 0, ly.post_ffn_ln, mlp, 0, rows, H);
        residual(*xcur, 0, mlp, *xcur, 0, rows * H);
      }

      if (has_ple) {
        // Per-layer-input gate: gelu(gate(x)) * pli[L] -> proj -> norm -> add.
        // When collapsed to the tail, pli for layer L is the last row of its
        // [n, hpli] block: offset (L*n + (n-rows))*hpli.
        qmm(ly.plg_w, ly.plg_s, ly.plg_b, *xcur, plg, H, hpli);
        geglu(plg, pli_lm,
              ((std::size_t)L * n + (n - rows)) * hpli * 2, plg, rows * hpli);
        if (_ple_quant) {                              // 4-bit qmm
          qmm(ly.plp_w, ly.plp_s, ly.plp_b, plg, plp, hpli, H);
        } else {
          dense(plg, ly.plp_w, plp, hpli, H);
        }
        if (rows == 1) {
          rms_add(plp, ly.post_pli_ln, *xcur, rows, H, ly.layer_scalar);
        } else {
          rms(plp, 0, ly.post_pli_ln, plp, 0, rows, H);
          residual(*xcur, 0, plp, *xcur, 0, rows * H);
          scale(*xcur, rows * H, ly.layer_scalar);
        }
      } else {
        // No PLE: the layer tail is just h *= layer_scalar.
        scale(*xcur, rows * H, ly.layer_scalar);
      }
    }

    // ---- final norm (last token) + lm_head + softcap ---------------
    // Skipped on KV-only intermediate chunks (logits discarded there).
    if (want_logits) {
      rms(*xcur, (std::size_t)(rows - 1) * H * 2, _final_ln, hn, 0, 1, H);
      if (_embed_is_q6k) {
        enc.set_function(_fn_qmv_q6k);   // native Q6_K lm_head GEMV
        enc.set_buffer(0, _embed_q6k);
        enc.set_buffer(1, hn);
        enc.set_buffer(2, logits);
        enc.set_constant(3, H);
        enc.set_constant(4, c.vocab);
        enc.dispatch({32, (unsigned)(((c.vocab + 7) / 8) * 2), 1},
                     {32, 2, 1});
      } else {
        enc.set_function(_fn_qmv_embed);   // lm_head at embed_bits
        enc.set_buffer(0, _embed_w);
        enc.set_buffer(1, _embed_s);
        enc.set_buffer(2, _embed_b);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, logits);
        enc.set_constant(5, H);
        enc.set_constant(6, c.vocab);
        enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
      }
      if (c.final_softcap > 0.0f) {
        enc.set_function(_fn_softcap);
        enc.set_buffer(0, logits);
        enc.set_buffer(1, logits);
        enc.set_constant(2, c.vocab);
        enc.set_constant(3, c.final_softcap);
        enc.dispatch({(unsigned)c.vocab, 1, 1}, {256, 1, 1});
      }
      // Ban suppressed tokens on the prefill's last-token logits too, so the
      // FIRST predicted token (this chunk's argmax / the realtime describe's
      // opener) can't be a reasoning-channel token. Same mask as decode.
      encode_suppress_(enc, logits, c.vocab);
    }
  }
  stream.commit().wait();
  if (!_full_paged) {
    _ctx->kv_append(cm, n);   // paged path advanced seq_len via append()
  }

  // Intermediate chunk: KV is populated; no logits to read. Non-empty
  // sentinel so the caller's empty()==failure check passes.
  if (!want_logits) { return std::vector<float>(1, 0.0f); }

  std::vector<float> out((std::size_t)c.vocab);
  read_elt_(logits.contents(), out.data(), (std::size_t)c.vocab, c.use_bf16);
  return out;
}

std::vector<float>
MetalGemmaModel::prefill(ContextId cid, const std::vector<std::int32_t>& ids)
{
  const int n = (int)ids.size();
  if (n <= 0) { return {}; }
  const int B = _sliding_chunk;
  // The sliding-layer K/V ring holds min(max_seq, window+B) slots. A single
  // kv_write wraps the ring (and could clobber a still-in-window key) only
  // when it crosses the ring end, i.e. kv_off+n > ring_cap. If the whole
  // prompt lands before the wrap, run ONE forward regardless of B -- this
  // avoids paying a per-chunk shared-KV tail pass + command-stream commit
  // for every B tokens (MLX runs the prompt in one shot, so this matches it).
  ContextId cm = cm_for_(cid);
  int kv_off = 0;
  if (B > 0 && cm.valid()) { kv_off = _ctx->kv_seq_len(cm); }
  const int ring_cap = std::min(_cfg.max_seq, _cfg.sliding_window + B);
  bool wraps = (B > 0) && (kv_off + n > ring_cap);
  // Lazy single-pass: a FRESH one-shot prefill longer than the bounded sliding
  // ring would wrap, dropping the 40 sliding layers off the matrix-core mma2s
  // path onto the slow staged ALU SDPA (the 4k prefill cliff). Grow THIS
  // context's sliding ring to hold the whole prompt -> the forward runs in one
  // pass, sliding stays on mma2s. Scoped to this context (kv_off==0; the
  // realloc discards KV) and bounded by max_seq. Short prefixes (realtime-vqa
  // scenes) never reach here, so branch contexts keep their small ring -- the
  // grown ring's extra memory is only paid by the long-prompt context itself.
  if (_sliding_grow && wraps && kv_off == 0 && n <= _cfg.max_seq && cm.valid()
      && _ctx->ensure_sliding_capacity(cm, n)) {
    wraps = false;
  }
  // Bounded-ring single-pass prefill: when the ring stays bounded but the
  // materialized (kt/vt-reading) sliding attention can cover the whole prompt,
  // run ONE forward over the full batch. The sliding ATTENTION then reads the
  // full-batch K/V scratch (ring-independent, mat_sliding), so the wrap that
  // would corrupt a ring read never happens; only the sliding KV-WRITE touches
  // the bounded ring, and a single full write leaves the correct trailing
  // window resident (the last `ring` logical positions are never clobbered).
  // Eligible only on a fresh context (qpos==0, so mat_ok holds) and when the
  // prompt fits the materialized score cap (T_kv == n <= _scores_cap).
  if (wraps && _prefill_subblock && _mat_sliding && _materialized_global
      && kv_off == 0 && n <= _scores_cap && _fn_dense_t_qkcausal.valid()
      && _fn_dense_t_pvcausal.valid()) {
    wraps = false;
  }
  if (!wraps) {
    if (n == 1) { return forward(cid, ids[0]); }
    return forward_chunk_(cid, ids);
  }
  // Ring wrap: each kv_write must span <= B positions so a wrap never
  // clobbers an in-window key. Process the prompt in <= B-token chunks; KV
  // accumulates across chunks (kv_append), and the LAST chunk's logits are
  // the prefill output.
  std::vector<float> out;
  for (int off = 0; off < n; off += B) {
    const int m = (n - off < B) ? (n - off) : B;
    const bool last = (off + m >= n);
    std::vector<std::int32_t> chunk(ids.begin() + off, ids.begin() + off + m);
    // Only the final chunk's logits are kept; intermediate chunks run
    // KV-only (skip the shared-KV tail + lm_head) to halve their weight read.
    out = (m == 1) ? forward(cid, chunk[0])
                   : forward_chunk_(cid, chunk, nullptr, nullptr, last);
    if (out.empty()) { return {}; }
  }
  return out;
}

void
MetalGemmaModel::set_suppressed_tokens(std::span<const std::int32_t> ids)
{
  // A handful at most (the reasoning-channel open tokens). Uploaded once to a
  // small device buffer; encode_step_ masks _d_logits at these ids after the
  // softcap (suppress_logits_f16).
  constexpr int kMaxSuppress = 16;
  int n = (int)ids.size();
  if (n > kMaxSuppress) { n = kMaxSuppress; }
  if (n <= 0 || _mc == nullptr) { _n_suppress = 0; return; }
  _d_suppress_ids =
      _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
  if (_d_suppress_ids.empty()) { _n_suppress = 0; return; }
  std::memcpy(_d_suppress_ids.contents(), ids.data(),
              (std::size_t)n * sizeof(std::int32_t));
  _n_suppress = n;
}

std::vector<float>
MetalGemmaModel::prefill_mm(ContextId cid,
                           const std::vector<std::int32_t>& ids,
                           const std::vector<float>& mm_rows, int n_mm,
                           const std::vector<int>& positions)
{
  const int n = (int)ids.size();
  const int H = _cfg.hidden;
  if (n <= 1 || n_mm <= 0 || (int)positions.size() != n_mm ||
      (int)mm_rows.size() < (std::size_t)n_mm * H) {
    return {};
  }
  if (n > _cfg.max_seq) { return {}; }
  // The single forward below writes all n positions into each SLIDING
  // layer's bounded ring (window+B slots) in one pass; for a multimodal
  // prefix longer than the ring -- every multi-frame video scene -- that
  // wraps the ring and clobbers in-window keys BEFORE they are attended,
  // corrupting the sliding/local-attention layers (the global layers are
  // paged and grow on demand, so output stays fluent but loses visual
  // grounding). Grow this fresh context's sliding ring to hold the whole
  // prefix so the forward never wraps -- mirrors prefill()'s _sliding_grow
  // path. Bounded by max_seq; the per-question branches deep-copy the grown
  // ring. (Short single-image prompts that fit the ring never grow.)
  {
    const ContextId cm = cm_for_(cid);
    const int B = _sliding_chunk;
    const int ring_cap = std::min(_cfg.max_seq, _cfg.sliding_window + B);
    if (_sliding_grow && B > 0 && cm.valid() && _ctx->kv_seq_len(cm) == 0
        && n > ring_cap && !_ctx->ensure_sliding_capacity(cm, n)) {
      return {};   // grow failed -> fail rather than silently corrupt
    }
  }
  // Upload the (un-scaled) encoder rows to an [n_mm, H] buffer in the
  // model's COMPUTE dtype. The vision/audio rows arrive as fp32 (mm_rows);
  // row_scatter (forward_chunk_) overlays them into the embedding stream
  // using the _lib_elt metallib, which is dtype-suffixed (VPIPE_ELT=bfloat
  // in bf16 builds). Writing f16 here while the stream is bf16 scattered f16
  // bit-patterns reinterpreted as bf16 -> garbage image/audio tokens (the
  // model "saw no image" and refused; text was unaffected because its rows
  // come from the bf16 embed gather). Match the stream dtype.
  SharedBuffer mm_emb = _mc->make_shared_buffer((std::size_t)n_mm * H * 2);
  auto* d = static_cast<std::uint16_t*>(mm_emb.contents());
  const bool bf16 = _cfg.use_bf16;
  for (std::size_t i = 0; i < (std::size_t)n_mm * H; ++i) {
    d[i] = f32_to_elt_(mm_rows[i], bf16);
  }
  // forward_chunk_ does the splice + PLE zeroing.
  return forward_chunk_(cid, ids, &mm_emb, &positions);
}

bool
MetalGemmaModel::branch_kv(ContextId parent, ContextId child)
{
  auto it = _ctxmap.find(parent.v);
  if (it == _ctxmap.end()) {
    // Parent never used a context yet -> the child simply starts fresh
    // (cm_for_ acquires a root on its first use). Nothing to copy.
    return true;
  }
  // ContextManager deep-copies the parent's contiguous KV prefix into the
  // child so the two diverge independently (it owns the buffers now).
  ContextId cm_child = _ctx->branch(it->second);
  if (!cm_child.valid()) { return false; }
  _ctxmap[child.v] = cm_child;
  return true;
}

void
MetalGemmaModel::release_kv(ContextId cid)
{
  auto it = _ctxmap.find(cid.v);
  if (it == _ctxmap.end()) { return; }
  _ctx->release(it->second);
  _ctxmap.erase(it);
}

}  // namespace vpipe::genai
