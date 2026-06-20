// attn_steel_nax.metal -- MLX's MATRIX-CORE (NAX) steel flash-attention,
// vendored for vpipe's metal-compute path (T=half, head_dim 64). This is the
// kernel MLX itself dispatches on M5 (sdpa_full_self_attention_nax, bq=64,
// bk=32, wm=4): QK^T and P*V run on the hardware matrix units via MPP
// matmul2d (nax.h), with the online softmax done entirely on the register-
// resident MMATile (row_reduce / row_bin_op -- no threadgroup round-trip).
// It is the tuned register-blocked flash; vpipe's hand-rolled tg-staged
// sdpa_full_mma2_d64 lost to the (ALU) steel kernel, whereas THIS uses the
// matrix cores stock-mlx can't, so it should beat the no-NAX reference.
//
// Buffer/param contract (identical to attn_steel, i.e. MLX steel `attention`):
//   0:Q 1:K 2:V 3:O (half)  4:AttnParams*  (5:AttnMaskParams 6:mask 7:sinks
//   function-constant-gated off). Func constants: align_Q(200), align_K(201),
//   has_mask(300)=0, do_causal(301)=0, has_sinks(302)=0. scale in AttnParams
//   is plain 1/sqrt(D). Grid: threadgroups (NQ=ceil(qL/64), H, B), tg
//   (32, wm=4, wn=1).  M5-only (matmul2d): #if __HAVE_TENSOR__, stub else.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

using namespace metal;

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

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

// head_dim 64 (Qwen3-VL vision tower / Qwen3-ASR audio encoder). bq=64,
// bk=32, wm=4, wn=1 -- the MLX M5 default for the NAX attention.
template [[host_name("attn_steel_nax_h_bd64")]] [[kernel]] decltype(attention_nax<
                                                                    half,
                                                                    64,
                                                                    32,
                                                                    64,
                                                                    4,
                                                                    1,
                                                                    half,
                                                                    float>)
attention_nax<half, 64, 32, 64, 4, 1, half, float>;

#else
// Tensor ops unavailable for this target: a stub so the metallib still builds.
// The loader never binds this on a non-tensor (pre-M5) GPU.
kernel void attn_steel_nax_h_bd64(device half* O [[buffer(3)]],
                                  uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (half)0; } }
#endif
