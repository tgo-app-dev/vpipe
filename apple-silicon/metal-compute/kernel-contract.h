#ifndef VPIPE_METAL_COMPUTE_KERNEL_CONTRACT_H
#define VPIPE_METAL_COMPUTE_KERNEL_CONTRACT_H

// THE KERNEL CONTRACT: the host Metal kernels a plugin may dispatch by
// name, and everything a dispatch depends on besides the name -- the
// parameter block the steel attention kernel reads and the function
// constants that specialise it. Feature VPIPE_FEATURE_KERNEL_CONTRACT
// ("kernel-contract/1") for revision 1, and
// VPIPE_FEATURE_KERNEL_CONTRACT_2 ("kernel-contract/2") for the entries
// revision 2 added; a host with the second still serves the first.
//
// Resolving a host kernel by name is an ABI like any other, only a
// string-level one: a renamed entry point resolves to an INVALID
// function, which dispatches nothing -- or worse, re-runs the previous
// dispatch's pipeline over this one's buffers. So:
//
//   * Every entry listed here resolves, in the library listed with it,
//     for as long as this revision is supported, with the same argument
//     order, buffer bindings and function-constant meanings. The test
//     kernel_contract.* resolves each one in the built metallibs.
//   * The ELEMENT TYPE is chosen by the LIBRARY ("..._bf16"), never by
//     the entry's own name -- an entry spelled "..._f16" is the same entry
//     in both twins. Only the libraries below are contracted.
//   * A change to any of it is a new revision (kernel-contract/2); the
//     old one stays for as long as the ABI that shipped it is in the
//     host's support window. Entry points NOT listed here are the host's
//     own and may change at any time: copy the kernel into the plugin
//     (plugins embed their own metallibs) rather than depend on it.

#include <cstdint>
#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {
namespace metal_compute {
namespace contract {

// The newest revision this header describes. kKernels lists revision 1
// first and then what revision 2 added; a plugin that resolves a
// revision-2 entry requires "kernel-contract/2".
inline constexpr int kRevision = 2;

struct Kernel {
  std::string_view library;
  std::string_view entry;
};

// ---- named entry points ---------------------------------------------------

inline constexpr Kernel kKernels[] = {
  // Dense GEMM, tiled (the path a box without matrix cores takes).
  {"dense_gemm_bf16", "dense_gemm_t_f16"},
  {"dense_gemm_bf16", "dense_gemm_t_bm64_f16"},
  {"dense_gemm_bf16", "dense_gemm_t_bm64bn64_f16"},
  {"dense_gemm_bf16", "dense_gemm_bias_f16"},
  {"dense_gemm_bf16", "dense_gemm_swiglu_bm64_f16"},
  {"dense_gemm_bf16", "dense_gemv_t_f16"},
  // Dense GEMM on the matrix cores (M5 matmul2d).
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_n128_f16"},
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_n128_acc_f16"},
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_n128_scaled_f16"},
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_n128x256_f16"},
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_n128x256_acc_f16"},
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_n128x256_tn2_f16"},
  {"dense_gemm_mma_bf16", "dense_gemm_mma_t_scaled_f16"},
  // Elementwise and small ops a DiT block is made of.
  {"llm_elementwise_bf16", "adaln_modulate_f16"},
  {"llm_elementwise_bf16", "adaln_modulate_v4_f16"},
  {"llm_elementwise_bf16", "gated_residual_f16"},
  {"llm_elementwise_bf16", "gated_residual_v4_f16"},
  {"llm_elementwise_bf16", "gelu_tanh_ff_f16"},
  {"llm_elementwise_bf16", "mul_sigmoid_f16"},
  {"llm_elementwise_bf16", "swiglu_f16"},
  {"llm_elementwise_bf16", "layer_norm_plain_f16"},
  {"llm_elementwise_bf16", "bias_add_rows_f16"},
  {"llm_elementwise_bf16", "im2col_hwc_3x3_f16"},
  {"llm_elementwise_bf16", "transpose_abd_f16"},
  {"rms_norm_bf16", "rms_norm_f16"},
  {"rms_norm_bf16", "rms_norm_fast_f16"},
  {"rope_bf16", "rope_pair_table_ftab_f16"},
  {"rope_bf16", "transpose_rope_pair_ftab_f16"},
  // Attention: the scalar SDPA, and the steel flash kernels -- which
  // need the function constants below to resolve at all.
  {"sdpa_bf16", "sdpa_full_f16"},
  {"sdpa_bf16", "sdpa_causal_f16"},
  {"attn_steel", "attn_steel_h_bd64_bf16"},
  {"attn_steel", "attn_steel_h_bd128_bf16"},
  {"attn_steel_nax", "attn_steel_nax_h_bd64_bf16"},
  {"attn_steel_nax", "attn_steel_nax_h_bd128_bf16"},
  // ...and their f16 twins, which the toolkit's Sol resolves for an f16
  // forward (MetalSolAttention::_lib_for_width_). Every ABI 8 host has
  // carried them; they were simply never listed.
  {"attn_steel", "attn_steel_h_bd64"},
  {"attn_steel", "attn_steel_h_bd128"},
  {"attn_steel_nax", "attn_steel_nax_h_bd64"},
  {"attn_steel_nax", "attn_steel_nax_h_bd128"},

  // ---- revision 2 ----
  // y += scale * (x @ W^T): the accumulate twin runtime LoRA adds an
  // adapter's up projection with on a box without matrix cores.
  {"dense_gemm_bf16", "dense_gemm_t_bm64_acc_f16"},
  // The f16 libraries, for a model that runs in f16 (a codec whose
  // reference is f16): the GEMMs, the scalar attention and the norm
  // that are not worth copying into a plugin.
  {"dense_gemm", "dense_gemm_bias_f16"},
  {"dense_gemm_mma", "dense_gemm_mma_t_n128_f16"},
  {"dense_gemm_mma", "dense_gemm_mma_t_n128x256_f16"},
  {"sdpa", "sdpa_full_f16"},
  {"rms_norm", "rms_norm_fast_f16"},
};

// ---- the toolkit's kernels ------------------------------------------------
//
// The toolkit (generative-models/shared, libvpipe_toolkit.a) is compiled
// INTO each plugin, so the kernels it dispatches are resolved by a copy of
// the toolkit as old as the plugin -- they are contracted exactly like the
// ones above. `twins` means the toolkit loads the f16 library under
// `library` AND the bf16 one under `library` + "_bf16", as its model's
// element type decides, so both must carry the entry.
struct ToolkitKernel {
  std::string_view library;
  std::string_view entry;
  bool             twins = true;
};

inline constexpr ToolkitKernel kToolkitKernels[] = {
  // i8_gemm (shared/i8-gemm.h): activation quantization + int8 GEMM.
  {"affine_dequant", "quant_f16_i8_row_g32"},
  {"affine_dequant", "quant_f16_i8_row_g32_pad"},
  {"affine_dequant", "quant_f16_i8_row_g64"},
  {"affine_dequant", "quant_f16_i8_row_g64_pad"},
  {"affine_dequant", "quant_f16_i8_row_g128"},
  {"affine_dequant", "quant_f16_i8_row_g128_pad"},
  {"affine_dequant", "quant_f16_i8_row_g512"},
  {"affine_dequant", "quant_f16_i8_row_g512_pad"},
  {"affine_dequant", "quant_f16_u8_row_g64"},
  {"affine_dequant", "affine_group_sums_w4g64"},
  {"affine_dequant", "affine_group_sums_w8g64"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g32rs"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g32rs_sk"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g64rs"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g64rs_sk"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g128rs"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g128rs_sk"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g512"},
  {"dense_gemm_mma", "gemm_i8i8_sc_f16_n64_g512_sk"},
  {"dense_gemm_mma", "gemm_u8q_w4g64"},
  {"dense_gemm_mma", "gemm_u8q_w4g64_sk"},
  {"dense_gemm_mma", "gemm_u8q_w4g64_cm"},
  {"dense_gemm_mma", "gemm_u8q_w4g64_cm_sk"},
  {"dense_gemm_mma", "gemm_u8q_w8g64"},
  {"dense_gemm_mma", "gemm_u8q_w8g64_sk"},
  {"dense_gemm_mma", "gemm_u8q_w8g64_cm"},
  {"dense_gemm_mma", "gemm_u8q_w8g64_cm_sk"},
  {"llm_elementwise", "splitk_fold_f32_f16"},
  {"llm_elementwise", "splitk_fold_bias_f32_f16"},
  // The signed-code layout (I8GemmContext::gemm_signed): its activation
  // quantizer, the conversion, its dequant and its GEMM. A toolkit on a
  // host older than these finds them invalid and keeps the unsigned
  // route (signed_available() is false), so nothing requires them.
  {"affine_dequant", "quant_f16_i8s_row_g64"},
  {"affine_dequant", "affine_to_signed_w4g64"},
  {"affine_dequant", "affine_to_signed_w8g64"},
  {"affine_dequant", "affine_dequant_w4g64s"},
  {"affine_dequant", "affine_dequant_w8g64s"},
  {"dense_gemm_mma", "gemm_i8s_w4g64"},
  {"dense_gemm_mma", "gemm_i8s_w4g64_sk"},
  {"dense_gemm_mma", "gemm_i8s_w8g64"},
  {"dense_gemm_mma", "gemm_i8s_w8g64_sk"},
  // ...and the down projection's quantizer straight from a SwiGLU's
  // halves (gemm_signed_swiglu). Optional the same way.
  {"affine_dequant", "swiglu_quant_f16_i8s_row_g64"},
  // THE FAST TIER (I8GemmContext::Tier::kFast): the rotated coarse-group
  // quantizers and GEMMs, per group. Optional: without them fast_group()
  // answers 0 and every weight stays on the accurate tier.
  {"affine_dequant", "quant_i8_row_gc256_r"},
  {"affine_dequant", "quant_i8_row_gc512_r"},
  {"affine_dequant", "quant_i8_row_gc1024_r"},
  {"affine_dequant", "swiglu_quant_i8_row_gc256_r"},
  {"affine_dequant", "swiglu_quant_i8_row_gc512_r"},
  {"affine_dequant", "swiglu_quant_i8_row_gc1024_r"},
  {"affine_dequant", "affine_to_i8c_w8_gc256_r"},
  {"affine_dequant", "affine_to_i8c_w8_gc512_r"},
  {"affine_dequant", "affine_to_i8c_w8_gc1024_r"},
  {"affine_dequant", "i8c_dequant_gc256_r"},
  {"affine_dequant", "i8c_dequant_gc512_r"},
  {"affine_dequant", "i8c_dequant_gc1024_r"},
  {"dense_gemm_mma", "gemm_i8c_g256_t64x128"},
  {"dense_gemm_mma", "gemm_i8c_g512_t64x128"},
  {"dense_gemm_mma", "gemm_i8c_g1024_t64x128"},
  {"dense_gemm_mma", "gemm_i8c_g256_t128x64"},
  {"dense_gemm_mma", "gemm_i8c_g512_t128x64"},
  {"dense_gemm_mma", "gemm_i8c_g1024_t128x64"},
  // ...a w4 g64 pack's conversion into it, out of place. Optional apart
  // from the tier: without it a w4 weight stays on the accurate tier.
  {"affine_dequant", "affine_to_i8c_w4_gc256_r"},
  {"affine_dequant", "affine_to_i8c_w4_gc512_r"},
  {"affine_dequant", "affine_to_i8c_w4_gc1024_r"},
  // THE TOKEN TIER (I8GemmContext::Tier::kToken): the per-row
  // quantizers, plain and Hadamard-256 rotated, and the register-
  // fragment GEMM at its two tiles and two K loops (set_token_kernel:
  // m = blocked, d = double-buffered). Optional the same way.
  {"affine_dequant", "quant_i8_tok"},
  {"affine_dequant", "swiglu_quant_i8_tok"},
  {"affine_dequant", "affine_to_i8pt_w8"},
  {"affine_dequant", "i8pt_dequant"},
  {"affine_dequant", "quant_i8_tok_r"},
  // The token tier's ConvRot activation quantizer, for ComfyUI's int8
  // ConvRot weights run as stored (I8GemmContext::quantize_token_convrot).
  // Optional: on an older host convrot_available() is false and such a
  // weight is widened per GEMM as before.
  {"affine_dequant", "quant_i8_tok_cr", false},
  {"affine_dequant", "swiglu_quant_i8_tok_r"},
  {"affine_dequant", "affine_to_i8pt_w8_r"},
  {"affine_dequant", "i8pt_dequant_r"},
  {"dense_gemm_mma", "gemm_i8pt_m128x64"},
  {"dense_gemm_mma", "gemm_i8pt_m64x128"},
  {"dense_gemm_mma", "gemm_i8pt_d128x64"},
  {"dense_gemm_mma", "gemm_i8pt_d64x128"},
  // ...and a w4 pack's conversion, as the fast tier's.
  {"affine_dequant", "affine_to_i8pt_w4"},
  {"affine_dequant", "affine_to_i8pt_w4_r"},
  // SageAttention's quantization (shared/metal-sage-attention.h).
  {"sage_quant", "sage_k_sum_chunk"},
  {"sage_quant", "sage_k_mean_reduce"},
  {"sage_quant", "sage_quant_block"},
  // ...and its int8 P*V's prologue: Q per token, V's absmax over the
  // sequence (chunks, then a reduce), V as uint8. Optional: a toolkit on
  // a host older than these finds them invalid and runs the int8 QK
  // alone (MetalSageAttention::pv8_available() false). The flash
  // kernel's constants kAttnPvInt8 / kAttnQTokenScale arrived with them,
  // so a toolkit sets those only where these resolve.
  {"sage_quant", "sage_quant_token"},
  {"sage_quant", "sage_v_absmax_chunk"},
  {"sage_quant", "sage_v_absmax_reduce"},
  {"sage_quant", "sage_quant_v_channel_u8"},
  // Sol-Attn's routing (shared/metal-sol-attention.h); its exact blocks
  // run on the steel kernels above.
  {"sol_attn_mma", "sol_summaries_mma"},
  {"sol_attn_mma", "sol_kc_stats_mma"},
  {"sol_attn_mma", "sol_route_mma"},
  {"sol_attn_mma", "sol_route_p_mma"},
  {"sol_attn_mma", "sol_scan_mma"},
  {"sol_attn_mma", "sol_emit_mma"},
  {"sol_attn_mma", "sol_approx_mma"},
  {"sol_attn_mma", "sol_merge_mma"},
  {"sol_attn_mma", "sol_merge_ml_mma"},
  // ...and its "exact" threshold: the key centroids' covariance, the
  // thresholds, and the two routes that read them. Optional: a toolkit
  // on a host older than these finds them invalid and refuses "exact" by
  // name (MetalSolAttention::exact_available() false), "diag" unchanged.
  {"sol_attn_mma", "sol_kc_cov_mma"},
  {"sol_attn_mma", "sol_thr_exact_mma"},
  {"sol_attn_mma", "sol_route_t_mma"},
  {"sol_attn_mma", "sol_route_pt_mma"},
  {"sol_proxy_mma", "sol_proxy_mma"},
  // FP8 expansion (shared/fp8-expand.h): the bf16 library only.
  {"affine_dequant_bf16", "fp8_e4m3_dequant", false},
  {"affine_dequant_bf16", "fp8_e5m2_dequant", false},
  {"affine_dequant_bf16", "fp8_e4m3_dequant_scaled", false},
  {"affine_dequant_bf16", "fp8_e5m2_dequant_scaled", false},
  // ...and ComfyUI's int8, plain and ConvRot-rotated. Optional: a toolkit
  // on a host older than these finds them invalid, keeps FP8, and an
  // int8 checkpoint is refused by name (Expander::int8_valid() false).
  {"affine_dequant_bf16", "i8_dequant_scaled", false},
  {"affine_dequant_bf16", "i8_convrot_dequant", false},
  // ...and its W4A8 (four-bit codes over the same int8 grid). Optional
  // alike: an older host refuses a W4A8 checkpoint by name
  // (Expander::w4a8_valid() false).
  {"affine_dequant_bf16", "w4a8_convrot_dequant", false},
  // The accurate tier's quantizers rotated (I8GemmContext::
  // set_g64_rotation). Optional: a toolkit on a host older than these
  // finds them invalid and keeps the plain pair (g64_rotated() false).
  {"affine_dequant", "quant_f16_i8_row_g64_r"},
  {"affine_dequant", "quant_f16_i8_row_g64_r_pad"},
};

// ---- quantized families ---------------------------------------------------
//
// "<prefix>w<bits>g<group><suffix>": every combination of `bits` and
// `groups` resolves, and `wide` (when non-empty) only at `wide_group` --
// the 64-row tile exists at group 64 and nowhere else.
struct QuantFamily {
  std::string_view library;
  std::string_view prefix;
  int              bits[2];
  int              groups[2];
  std::string_view wide;         // optional suffix, e.g. "_bm64"
  int              wide_group;   // the one group `wide` exists at
};

inline constexpr QuantFamily kQuantFamilies[] = {
  {"affine_qmm_steel_bf16", "affine_qmm_steel_", {4, 8}, {32, 64},
   "_bm64", 64},
  {"affine_dequant_bf16", "affine_dequant_", {4, 8}, {32, 64}, "", 0},
  {"affine_qmv_bf16", "affine_qmv_", {4, 8}, {32, 64}, "", 0},
};

// ---- steel attention ------------------------------------------------------
//
// The parameter block attn_steel* read from their params buffer. Layout
// is the contract (the host checks it field for field against the
// vendored kernel's own definition).
struct SteelAttnParams {
  int          B;            // batch
  int          H;            // heads
  int          D;            // head dim
  int          qL;           // query length
  int          kL;           // key length
  int          gqa_factor;   // query heads per kv head
  float        scale;        // softmax scale
  int          NQ;           // query blocks
  int          NK;           // key/value blocks
  int          NQ_aligned;   // full query blocks
  int          NK_aligned;   // full key/value blocks
  int          qL_rem;       // rows in the last query block
  int          kL_rem;       // rows in the last key block
  int          qL_off;       // offset of the query window in the keys
  std::int64_t Q_strides[3]; // (B, H, L) in elements; D is contiguous
  std::int64_t K_strides[3];
  std::int64_t V_strides[3];
  std::int64_t O_strides[3];
};

// Function-constant indices the steel attention kernels are specialised
// on. All are bool.
inline constexpr int kAttnAlignQ  = 200;   // qL is a multiple of BQ
inline constexpr int kAttnAlignK  = 201;   // kL is a multiple of BK
inline constexpr int kAttnHasMask = 300;   // a mask buffer is bound
inline constexpr int kAttnCausal  = 301;   // causal masking
inline constexpr int kAttnSinks   = 302;   // attention sinks are bound
inline constexpr int kAttnQkInt8  = 306;   // SageAttention int8 QK path
// ...and, on the attn_steel_nax entries with kAttnQkInt8 set, its P*V in
// int8 too, held in int32 across key blocks (both set together; see
// MetalSageAttention::pv8_constants). kAttnQTokenScale reads Q's scales
// per TOKEN, [B, H, qL] at buffer 17, where kAttnQkInt8 alone reads one
// per query block. kAttnPvInt8 takes three operands more: V as uint8
// (V minus its mean, quantized, plus 128) [B, KVH, kL, D] at 19, its
// per-channel scale and mean [B, KVH, D] fp32 at 20 and 21. Optional: a
// host that predates them does not declare them, and a function built
// with them set there is the int8 QK alone -- which is why the toolkit
// sets them only where the sage_quant_token family resolves.
inline constexpr int kAttnPvInt8      = 308;
inline constexpr int kAttnQTokenScale = 310;

}  // namespace contract
}  // namespace metal_compute
}  // namespace vpipe

VPIPE_API_END

#endif
