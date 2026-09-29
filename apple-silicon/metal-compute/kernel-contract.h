#ifndef VPIPE_METAL_COMPUTE_KERNEL_CONTRACT_H
#define VPIPE_METAL_COMPUTE_KERNEL_CONTRACT_H

// THE KERNEL CONTRACT: the host Metal kernels a plugin may dispatch by
// name, and everything a dispatch depends on besides the name -- the
// parameter block the steel attention kernel reads and the function
// constants that specialise it. Feature VPIPE_FEATURE_KERNEL_CONTRACT
// ("kernel-contract/1").
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

inline constexpr int kRevision = 1;

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
  // SageAttention's quantization (shared/metal-sage-attention.h).
  {"sage_quant", "sage_k_sum_chunk"},
  {"sage_quant", "sage_k_mean_reduce"},
  {"sage_quant", "sage_quant_block"},
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
  {"sol_proxy_mma", "sol_proxy_mma"},
  // FP8 expansion (shared/fp8-expand.h): the bf16 library only.
  {"affine_dequant_bf16", "fp8_e4m3_dequant", false},
  {"affine_dequant_bf16", "fp8_e5m2_dequant", false},
  {"affine_dequant_bf16", "fp8_e4m3_dequant_scaled", false},
  {"affine_dequant_bf16", "fp8_e5m2_dequant_scaled", false},
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

}  // namespace contract
}  // namespace metal_compute
}  // namespace vpipe

VPIPE_API_END

#endif
