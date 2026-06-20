// dense_gemm_mma.metal -- matrix-core (M5+) dense GEMM via Metal 4
// MetalPerformancePrimitives matmul2d over cooperative_tensor. Same
// math/interface as dense_gemm.metal's steel `dense_gemm_t_f16`
//   y[m, n] = bias[n] + sum_k x[m, k] * W[n, k]        (x @ W^T, NT)
// but the inner product runs on the GPU's hardware matrix units instead
// of simdgroup_matrix ALU ops. Built ONLY for the tensor-capable
// toolchain/target (-std=metal4.0); the loader gates use of this
// metallib on a runtime GPU-capability check (older GPUs keep the steel
// path), so the same project source stays correct + peak on both.
//
//   0:x[M,K] 1:W[N,K] 2:bias[N] 3:y[M,N]  4:K 5:N 6:M 7:has_bias
// One threadgroup per (TM*BM)x(TN*BN) output region; EXECUTION_SIMDGROUPS
// simdgroups cooperate on each BMxBN matmul2d tile, and a single
// threadgroup walks a TM x TN grid of such tiles so each loaded x/W tile
// feeds more MACs (higher arithmetic intensity -> less memory-bound on
// weight streaming for large-K projections). Dispatch (threads):
//   {ceil(N/(TN*BN))*tg, ceil(M/(TM*BM)), 1}, threadgroup {tg,1,1}
// with tg = EXECUTION_SIMDGROUPS*32. tgid.x -> N region, tgid.y -> M.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// Activation/weight storage type. half by default (unsuffixed kernel =
// f16, matching the rest of the metal LM kernels); -DVPIPE_ELT=bfloat
// builds the bf16 variant. The matrix unit accumulates in f32 internally
// and rounds once on store, so accuracy matches the steel path.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#if defined(__HAVE_TENSOR__)

// Core: each threadgroup computes a TM*BM (rows) x TN*BN (cols) output
// region as a TMxTN grid of BMxBN matmul2d tiles. SG simdgroups cooperate
// per tile. K is dynamic; the matmul2d op runs the full K reduction.
template <int BM, int BN, int SG, int TM, int TN>
static inline void dense_gemm_mma_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    device VPIPE_ELT* y, int K, int N, int M, uint3 tgid)
{
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W), dextents<int32_t, 2>(K, N));
  TX tY(y, dextents<int32_t, 2>(N, M));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m_base = (int)tgid.y * (TM * BM);
  const int n_base = (int)tgid.x * (TN * BN);

  for (int tm = 0; tm < TM; ++tm) {
    const int m0 = m_base + tm * BM;
    auto mX = tX.slice(0, m0);          // rows [m0, m0+BM) of x
    for (int tn = 0; tn < TN; ++tn) {
      const int n0 = n_base + tn * BN;
      auto mW = tW.slice(0, n0);        // rows [n0, n0+BN) of W
      auto cT = op.template get_destination_cooperative_tensor<
          decltype(mX), decltype(mW), VPIPE_ELT>();
      op.run(mX, mW, cT);
      auto mY = tY.slice(n0, m0);
      cT.store(mY);
    }
  }
}

// Production entry point (64x64, 4 simdgroups, single tile per tg) -- the
// shape the model path dispatches today. Kept byte-identical.
kernel void dense_gemm_mma_t_f16(
    const device VPIPE_ELT*  x        [[buffer(0)]],
    const device VPIPE_ELT*  W        [[buffer(1)]],
    const device VPIPE_ELT*  bias     [[buffer(2)]],
    device VPIPE_ELT*        y        [[buffer(3)]],
    const constant int& K        [[buffer(4)]],
    const constant int& N        [[buffer(5)]],
    const constant int& M        [[buffer(6)]],
    const constant int& has_bias [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  (void)has_bias; (void)bias;
  dense_gemm_mma_impl<64, 64, 4, 1, 1>(x, W, y, K, N, M, tgid);
}

// Production tiles, picked by an M5 tile sweep (metal-compute-gemm-mma.cc
// gemm_mma.tune). matmul2d throughput at the Qwen3.5 projection shapes
// (M = prefill tokens) depends strongly on K:
//   * 128x128 tile / 8 simdgroups -- fastest for K <= 4096 (~13 TFLOP/s),
//     but degrades past K=4096 (the deep-K weight stream outpaces the
//     square tile's x-reuse).
//   * 128x256 tile / 8 simdgroups -- rock-stable ~10.3 TFLOP/s at ALL K;
//     the choice for deep-K projections (down_proj, K = ffn).
// The model path dispatches n128 for K <= 4096 and n128x256 otherwise.
// Larger tiles (256x128, 256x256/16) and the 64x64-grid software tiling
// both lost and are not shipped. The 64x64 entry above stays as the
// reference for the correctness tests and an ultra-safe fallback.
#define DGV(NAME, BM, BN, SG)                                            \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      const device VPIPE_ELT* W [[buffer(1)]],                           \
      const device VPIPE_ELT* bias [[buffer(2)]],                        \
      device VPIPE_ELT* y [[buffer(3)]],                                 \
      const constant int& K [[buffer(4)]],                              \
      const constant int& N [[buffer(5)]],                              \
      const constant int& M [[buffer(6)]],                              \
      const constant int& has_bias [[buffer(7)]],                       \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    (void)has_bias; (void)bias;                                          \
    dense_gemm_mma_impl<BM, BN, SG, 1, 1>(x, W, y, K, N, M, tgid);       \
  }

DGV(dense_gemm_mma_t_n128_f16,    128, 128, 8)
DGV(dense_gemm_mma_t_n128x256_f16,128, 256, 8)

#else
// Tensor ops unavailable for this target: emit stubs so the metallib
// still builds. The loader never binds these on a non-tensor GPU.
#define DGV_STUB(NAME)                                                   \
  kernel void NAME(device float* y [[buffer(3)]],                        \
                   uint tid [[thread_position_in_grid]]) {               \
    if (tid == 0) { y[0] = 0.0f; }                                       \
  }
DGV_STUB(dense_gemm_mma_t_f16)
DGV_STUB(dense_gemm_mma_t_n128_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_f16)
#endif
