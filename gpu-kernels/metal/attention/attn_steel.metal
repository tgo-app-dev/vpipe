// attn_steel.metal -- MLX's steel flash-attention kernel, vendored and
// instantiated for vpipe's metal-compute prefill path (T=half,
// head_dim=128, GQA). This is the register-resident MMA flash kernel MLX
// itself ships (steel/attn): QK^T + PV as simdgroup_matrix matmuls, the
// online softmax done entirely on the register MMATile via the
// fragment-aware row_reduce / row_bin_op (no threadgroup round-trip),
// with double-staged Q/K/V BlockLoaders. It is the "optimal" reference;
// vpipe's own sdpa_paged_mma_f16 already overtakes MLX end-to-end, but in
// ISOLATION trails this kernel ~1.8x, so for fresh prefill (contiguous
// K/V, q_offset == 0) we hand attention straight to it.
//
// The attn/* steel headers are SELF-CONTAINED (define their own
// BlockLoader/BlockMMA/BaseMMAFrag/MMATile) -- do NOT also pull in the
// gemm/* mma+loader (they would redefine those symbols). Only gemm/params.h
// (GEMMParams, referenced by a leftover GEMMKernel in attn.h) is shared.
//
// Buffer/param contract (steel `attention`):
//   0:Q 1:K 2:V 3:O (half)  4:AttnParams*  (5:AttnMaskParams 6:mask
//   7:sinks are function-constant-gated off: has_mask=has_sinks=false).
//   Function constants: align_Q(200), align_K(201), do_causal(301).
//   scale in AttnParams is the plain 1/sqrt(D) (the kernel folds in
//   M_LOG2E_F and uses exp2). Grid: threadgroups (NQ, H, B), tg (32,wm,wn)
//   = (32,4,1) for BD=128.
// Config for head_dim 128: BQ=32, BK=16, BD=128, WM=4, WN=1 (MLX default).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

using namespace metal;

// steel_attention.h needs Limits<> and M_LOG2E_F, normally supplied by the
// heavy mlx/backend/metal/kernels/utils.h. The attn/* headers themselves do
// NOT use them, so provide just these two here and skip vendoring utils.h.
#ifndef M_LOG2E_F
#define M_LOG2E_F 1.44269504088896340736f
#endif

template <typename U>
struct Limits {
  static constexpr constant U max = metal::numeric_limits<U>::max();
  static constexpr constant U min = metal::numeric_limits<U>::min();
  static constexpr constant U finite_max = metal::numeric_limits<U>::max();
  static constexpr constant U finite_min = metal::numeric_limits<U>::min();
};
template <>
struct Limits<float> {
  static constexpr constant float max =
      metal::numeric_limits<float>::infinity();
  static constexpr constant float min =
      -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max =
      metal::numeric_limits<float>::max();
  static constexpr constant float finite_min =
      -metal::numeric_limits<float>::max();
};
template <>
struct Limits<half> {
  static constexpr constant half max = metal::numeric_limits<half>::infinity();
  static constexpr constant half min =
      -metal::numeric_limits<half>::infinity();
  static constexpr constant half finite_max =
      metal::numeric_limits<half>::max();
  static constexpr constant half finite_min =
      -metal::numeric_limits<half>::max();
};

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

// Concrete kernel entry point (the MLX instantiate_kernel expansion).
template [[host_name("attn_steel_h_bd128")]] [[kernel]] decltype(attention<
                                                                 half,
                                                                 32,
                                                                 16,
                                                                 128,
                                                                 4,
                                                                 1,
                                                                 half,
                                                                 float>)
attention<half, 32, 16, 128, 4, 1, half, float>;

// head_dim=64 variant for the Qwen3-VL vision tower / Qwen3-ASR audio
// encoder (both d_model/n_heads == 64). Same BQ/BK/WM/WN as bd128; only
// BD changes. Driven non-causal (do_causal function constant = false) for
// the bidirectional ViT attention, but the entry point itself is
// causal-agnostic (the constant is bound at pipeline creation).
template [[host_name("attn_steel_h_bd64")]] [[kernel]] decltype(attention<
                                                                half,
                                                                32,
                                                                16,
                                                                64,
                                                                4,
                                                                1,
                                                                half,
                                                                float>)
attention<half, 32, 16, 64, 4, 1, half, float>;
