#include "generative-models/gemma4/metal-gemma-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
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
  m->_fn_rope = m->_lib_rope.function("rope_f16");
  m->_fn_rms_rope = m->_lib_rope.function("rms_rope_f16");
  m->_fn_rms_rope2 = m->_lib_rope.function("rms_rope2_f16");
  m->_fn_rms_rope3 = m->_lib_rope.function("rms_rope3_f16");
  m->_fn_geglu = m->_lib_elt.function("geglu_f16");
  m->_fn_softcap = m->_lib_elt.function("softcap_f16");
  m->_fn_scale = m->_lib_elt.function("scale_inplace_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
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
  // KV-split GQA paged flash-decode for the full layers + its merge. The vec
  // kernel (one q-head/simdgroup, UK-unrolled vec4 loads, latency-optimal) is
  // already D-flexible (per4 = D/128; D=512 ok) -- the paged sibling of the
  // contiguous gqa_vec the full-layer decode used before paging.
  m->_fn_sdpa_pgqa = m->_lib_sdpa.function("sdpa_paged_gqa_vec_f16");
  m->_fn_argmax = m->_lib_elt.function("argmax_f16");
  // GPU sampler for the pipelined-decode non-greedy path (same kernel as
  // Qwen; dtype-correct via _lib_elt's suffix). Optional: greedy pdecode
  // only needs _fn_argmax.
  m->_fn_sample = m->_lib_elt.function("sample_topp_f16");
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
  m->_fn_dense_t = m->_lib_dense.function("dense_gemm_t_f16");
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
  // KV-split GQA paged flash-decode for the (paged) full layers: the fast
  // long-context decode path (the scalar paged kernel has no split). Default
  // ON when present; VPIPE_GEMMA_NO_PGQA=1 reverts to scalar sdpa_paged_causal.
  m->_pgqa_attn = m->_fn_sdpa_pgqa.valid() && m->_fn_sdpa_gqa_merge.valid()
      && std::getenv("VPIPE_GEMMA_NO_PGQA") == nullptr;
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
  _d_o = b(H);
  _d_act = b(c.ffn_inner);   // fused gate/up+geglu output (no gate/up bufs)
  _d_mlp = b(H);
  _d_ple = b(ple);
  _d_pleproj = b(ple);
  _d_pli = b(ple);
  _d_plg = b(c.hpli);
  _d_plp = b(H);
  _d_logits = b(c.vocab);
  _d_tok = _mc->make_shared_buffer(sizeof(std::int32_t));
  _d_argmax_id = _mc->make_shared_buffer(sizeof(std::int32_t));
  // Page table for the FULL-layer paged attention: max_pages triplets per
  // slot. kPgtabSlots slots form a ring so run-ahead pdecode (depth<=4) gives
  // each in-flight step its own table (the slot 0 path is used by every
  // synchronous caller, which waits before reusing it).
  if (_full_paged) {
    _d_pgtab = _mc->make_shared_buffer(
        (std::size_t)kPgtabSlots * _ctx->max_pages() * 3 * sizeof(std::int32_t));
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
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y);
    enc.set_constant(3, Hd);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  // Fused sublayer-out norm + residual add (+ post-scale): res = (res +
  // rms(xin, w)) * post_scale. One dispatch for the Gemma sandwich
  // `rms(out); h += out;` (and the layer_scalar at the PLE tail).
  auto rms_add = [&](const SharedBuffer& xin, const SharedBuffer& w,
                     const SharedBuffer& res, int R, int Hd,
                     float post_scale = 1.0f) {
    enc.set_function(_fn_rms_add);
    enc.set_buffer(0, xin);
    enc.set_buffer(1, w);
    enc.set_buffer(2, res);
    enc.set_buffer(3, res);
    enc.set_constant(4, Hd);
    enc.set_constant(5, eps);
    enc.set_constant(6, post_scale);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
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

  // ---- embeddings + per-layer inputs (once per token) -------------
  if (_embed_is_q6k) { embed_q6k(tokbuf, toff, _d_x, H, 1); }
  else { embed_gather(_embed_w, _embed_s, _embed_b, _d_x, H); }
  scale(_d_x, H, std::sqrt((float)H));                     // embed_scale
  // Diagnostic: fire N extra no-op dispatches/token to measure the per-launch
  // GPU idle (decode is a dependent-dispatch chain; this isolates the idle
  // cost of dispatch COUNT). VPIPE_GEMMA_DUMMY_DISP=N. Off in production.
  {
    static const int kDummy = std::getenv("VPIPE_GEMMA_DUMMY_DISP")
        ? std::atoi(std::getenv("VPIPE_GEMMA_DUMMY_DISP")) : 0;
    for (int i = 0; i < kDummy; ++i) { scale(_d_argmax_id, 1, 1.0f); }
  }
  // Per-Layer-Input projection (e4b PLE only; gemma4_unified has no PLE).
  if (has_ple) {
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
    const int D = ly.head_dim;
    const int Hkv = ly.n_kv;
    const int qd = Hq * D, kd = Hkv * D;
    const SharedBuffer& invf = ly.is_full ? _inv_freq_full : _inv_freq_sliding;
    const bool paged_full = ly.is_full && _full_paged;

    // Attention (input_norm -> attn -> post_attn_norm -> residual).
    DUP(DC_NORM, [&] { rms(_d_x, ly.in_ln, _d_hn, 1, H); });
    DUP(DC_PROJ, [&] { qmv(ly.qw, ly.qs, ly.qb, _d_hn, _d_q, H, qd); });
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
      DUP(DC_PROJ, [&] { qmv(ly.kw, ly.ks, ly.kb, _d_hn, _d_k, H, kd); });
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
      } else {
        DUP(DC_PROJ, [&] { qmv(ly.vw, ly.vs, ly.vb, _d_hn, _d_v, H, kd); });
        DUP(DC_NORM, [&] {
          rms_rope3(_d_q, ly.q_norm, _d_k, ly.k_norm, _d_v, _ones_vnorm,
                    invf, Hq, Hkv, D);
        });
      }
      if (paged_full) {
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
    const int attn_runs = (dup == attn_cat || dup == DC_ATTN) ? 2 : 1;
    for (int ar = 0; ar < attn_runs; ++ar) {
      if (paged_full) {
        // Paged decode attention (n_q=1) over the shared pool + page table.
        // Fast path: KV-split GQA flash-decode (read each KV head once for all
        // G q-heads, split the scan across grid.z) + merge -- the long-context
        // full-layer decode. Fallback: the scalar paged kernel (no split).
        const int G = (Hkv > 0) ? Hq / Hkv : 0;
        const bool use_pgqa = _pgqa_attn && Hkv > 0 && (Hq % Hkv == 0)
            && G >= 1 && G <= 16 && (D % 128 == 0) && !_d_gqa_oacc.empty();
        if (use_pgqa) {
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

    DUP(DC_PROJ, [&] { qmv(ly.ow, ly.os, ly.ob, _d_attn, _d_o, qd, H); });
    DUP(DC_NORM, [&] { rms_add(_d_o, ly.post_attn_ln, _d_x, 1, H); });  // += rms

    // MLP (geglu sandwich).
    DUP(DC_NORM, [&] { rms(_d_x, ly.pre_ffn_ln, _d_hn, 1, H); });
    // Fused gate/up GEMV + GeGLU: one dispatch over the interleaved weight
    // writes gelu(gate)*up straight to _d_act [1, ffn] (no gate/up buffers).
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
    DUP(DC_NORM, [&] { rms_add(_d_mlp, ly.post_ffn_ln, _d_x, 1, H); });  // += rms

    if (has_ple) {
      DUP(DC_PLE, [&] {
      // Per-layer-input gate: gelu(gate(x)) * pli[L] -> proj -> norm -> add.
      // The gate GEMV + geglu are FUSED (gelu*pli[L] in the GEMV write); falls
      // back to the 2-dispatch path if the fused kernel is unavailable.
      if (_fn_qmv_gelu_mul.valid()) {
        enc.set_function(_fn_qmv_gelu_mul);
        enc.set_buffer(0, ly.plg_w);
        enc.set_buffer(1, ly.plg_s);
        enc.set_buffer(2, ly.plg_b);
        enc.set_buffer(3, _d_x);
        enc.set_buffer(4, _d_plg);
        enc.set_constant(5, H);
        enc.set_constant(6, hpli);
        enc.set_buffer(7, _d_pli, (std::size_t)L * hpli * 2);  // pli[L] vec
        enc.dispatch({32, (unsigned)(hpli / 4), 1}, {32, 2, 1});
      } else {
        qmv(ly.plg_w, ly.plg_s, ly.plg_b, _d_x, _d_plg, H, hpli);
        geglu(_d_plg, _d_pli, _d_plg, hpli, (std::size_t)L * hpli * 2);
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
  }

  // ---- final norm + tied quantized lm_head + softcap --------------
  // lm_head GEMV runs at embed_bits (8-bit on the GGUF path).
  DUP(DC_NORM, [&] { rms(_d_x, _final_ln, _d_hn, 1, H); });
  DUP(DC_LMHEAD, [&] {
    if (_embed_is_q6k) { lm_head_q6k(_d_hn, _d_logits, 0, H, c.vocab); }
    else {
      qmv_fn(_fn_qmv_embed, _embed_w, _embed_s, _embed_b, _d_hn, _d_logits, H,
             c.vocab);
    }
  });
  if (c.final_softcap > 0.0f) {
    enc.set_function(_fn_softcap);
    enc.set_buffer(0, _d_logits);
    enc.set_buffer(1, _d_logits);
    enc.set_constant(2, c.vocab);
    enc.set_constant(3, c.final_softcap);
    enc.dispatch({(unsigned)c.vocab, 1, 1}, {256, 1, 1});
  }
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
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, Hd); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto rms_add = [&](const SharedBuffer& xin, const SharedBuffer& w,
                     const SharedBuffer& res, int R, int Hd,
                     float post_scale = 1.0f) {
    enc.set_function(_fn_rms_add);
    enc.set_buffer(0, xin); enc.set_buffer(1, w);
    enc.set_buffer(2, res); enc.set_buffer(3, res);
    enc.set_constant(4, Hd); enc.set_constant(5, eps);
    enc.set_constant(6, post_scale);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
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
          auto kvw = [&](const SharedBuffer& src, std::size_t soff,
                         const SharedBuffer& cache) {
            enc.set_function(_fn_kv_write);
            enc.set_buffer(0, src, soff); enc.set_buffer(1, cache);
            enc.set_constant(2, cap); enc.set_constant(3, D);
            const int one = 1; enc.set_constant(4, one);
            enc.set_constant(5, kv_off);
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
    // softcap is monotonic, so argmax is unaffected by it. Single-TG: the
    // 262k-vocab scan is small (~25us) and a two-pass parallel argmax
    // regressed (dispatch overhead > scan savings) -- measured, reverted.
    enc.set_function(_fn_argmax);
    enc.set_buffer(0, _d_logits);
    enc.set_buffer(1, _d_argmax_id);
    enc.set_constant(2, c.vocab);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
    n_disp = enc.dispatch_count();
  }
  const auto t1 = std::chrono::steady_clock::now();
  stream.commit().wait();
  if (kProf) {
    const auto t2 = std::chrono::steady_clock::now();
    static double enc_ms = 0, gpu_ms = 0; static int cnt = 0;
    enc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    gpu_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (++cnt % 32 == 0) {
      std::fprintf(stderr, "[gemma-metal-prof] encode %.2f ms | commit+gpu "
                   "%.2f ms | %ld dispatches/tok (%d steps)\n",
                   enc_ms / cnt, gpu_ms / cnt, n_disp, cnt);
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
      enc.set_function(_fn_argmax);
      enc.set_buffer(0, _d_logits);
      enc.set_buffer(1, pd.gen_ids, out_off);
      enc.set_constant(2, c.vocab);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
    } else {
      const std::uint32_t step_seed = (std::uint32_t)(
          pd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
      enc.set_function(_fn_sample);
      enc.set_buffer(0, _d_logits);
      enc.set_buffer(1, pd.gen_ids, out_off);
      enc.set_constant(2, c.vocab);
      enc.set_constant(3, pd.sp.temperature);
      enc.set_constant(4, pd.sp.top_p);
      enc.set_constant(5, step_seed);
      enc.set_buffer(6, pd.sample_ws);
      enc.set_constant(7, pd.sp.n_iter);
      enc.set_constant(8, pd.sp.repetition_penalty);
      enc.set_constant(9, pd.sp.presence_penalty);
      enc.set_constant(10, pd.sp.top_k);
      enc.set_constant(11, pd.sp.min_p);
      enc.set_buffer(12, pd.seen);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
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
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    // Fused sandwich-out norm + residual (+ post-scale): res = (res +
    // rms(xin,w)) * post_scale. Tail-only (single row), mirrors decode.
    auto rms_add = [&](const SharedBuffer& xin, const SharedBuffer& w,
                       const SharedBuffer& res, int R, int Hd,
                       float post_scale = 1.0f) {
      enc.set_function(_fn_rms_add);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, res);
      enc.set_buffer(3, res);
      enc.set_constant(4, Hd);
      enc.set_constant(5, eps);
      enc.set_constant(6, post_scale);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
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
                   int D, int cap, int Hkv) {
      enc.set_function(_fn_kv_write);
      enc.set_buffer(0, src);
      enc.set_buffer(1, cache);
      enc.set_constant(2, cap);              // cache stride == ring cap
      enc.set_constant(3, D);
      enc.set_constant(4, n);
      enc.set_constant(5, kv_off);
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
          kvw(kt, *Kuse, D, cap, Hkv);
          kvw(vt, *Vuse, D, cap, Hkv);
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
      if (skip_attn) {
        // intentionally no SDPA dispatch (probe)
      } else if (paged_full) {
        // PAGED full-layer attention over the shared pool + page table. The
        // bulk (n_q>1) uses the matrix-core paged mma2 (M5) or, on a non-
        // matrix-core GPU (M4), the simdgroup_matrix paged flash; only the
        // shared-KV tail (n_q=1) falls to the scalar paged kernel (q3/qt match
        // the q[Hq,n_q,D] layout all three read).
        if (rows > 1 && _pmma2_attn) {
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
        if (mma2_ok) {
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
