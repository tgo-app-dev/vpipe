// dense_gemm_steel.metal -- DENSE f16/bf16 GEMM on MLX's steel BlockMMA, i.e.
// simdgroup_multiply_accumulate over 8x8 fragments instead of scalar FMA.
//
// The hypothesis this kernel exists to test, and the ANSWER:
//
// A register-resident probe says simdgroup MMA beats scalar FMA on a
// pre-matrix-core Apple GPU by ~29% -- 10.14 vs 7.88 TFLOP/s on an M4 Pro
// (gemm_mma.alu_rate_f16_vs_f32) -- while the tree's dense GEMMs sit at
// 6.8-7.2, i.e. ~90% of the SCALAR roofline but only ~70% of the MMA one. The
// tree already ran simdgroup MMA for QUANTIZED weights (affine_qmm_steel:
// BlockMMA + a dequantizing loader) and for attention (attn_steel), but dense
// weights -- what every bf16 DiT runs -- had no MMA path outside the M5-only
// matmul2d kernels. So: does a memory-fed dense MMA GEMM collect that 29%?
//
// **NO.** MEASURED at the Boogu shapes (boogu_perf.steel_mma_dense_gemm), eight
// tile shapes over 4 and 8 simdgroups, against dense_gemm_t_f16:
//   64x64x16   1.010x / 0.957x / 1.015x / 1.003x  (q-o / kv / ff-up / ff-down)
//   64x64x32   0.991x   32x32x32 0.977x   64x32x32 0.978x   32x64x32 0.999x
//   128x64x16  0.935x   64x128x16 0.932x  128x128x16 0.891x
// Everything lands at 4.9-7.2 TFLOP/s with the best config 1.5% ahead at best,
// so the multiply-accumulate ISSUE RATE is not what caps either kernel once the
// operands come through threadgroup memory -- the register-resident 10.14
// figure is unreachable in a real GEMM. Wider tiles are worse, not better, so
// it is not a tiling artifact either. Output is BIT-IDENTICAL to the scalar
// kernel (rel-L2 exactly 0 at every shape and tile).
//
// NOT WIRED into any model path: nothing would gain. Kept as the reproducible
// probe behind that conclusion -- one bench run re-answers the question on a
// new GPU, and it is the dense counterpart to affine_qmm_steel if a future
// kernel (async copies, split-K, a deeper software pipeline) changes the
// balance. This is the SAME BlockMMA and BlockLoader the quantized kernel uses,
// with a plain weight loader in place of the dequantizing one.
//
// Shape contract, matching dense_gemm_t_f16 so it is a drop-in:
//   y[M, N] = x[M, K] @ W[N, K]^T           (W row-major [N, K], "transposed B")
//   0:x 1:W 2:(unused, mirrors the bias slot) 3:y  4:K 5:N 6:M 7:(unused)
// Grid: threadgroups (ceil(N/BN), ceil(M/BM), 1), threadgroup = WM*WN*32.
//
// The gemm/* steel headers define BlockLoader / BlockMMA / MMATile; do NOT also
// include the attn/* ones in this translation unit (they redefine the symbols).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#ifndef STEEL_CONST
#define STEEL_CONST static constant constexpr const
#endif
#ifndef STEEL_PRAGMA_UNROLL
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

using namespace metal;

// Element (storage) type: half by default, -DVPIPE_ELT=bfloat for the bf16
// metallib. Accumulation is always f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define SIMD_SIZE 32

// mma.h has a complex64_t BlockMMA specialization (unused here but it must
// parse), so complex64_t has to be in scope first; MLX's complex.h in turn
// names their bfloat16_t wrapper in an unused ctor. Same two lines
// affine_qmm_steel.metal needs.
typedef bfloat16 bfloat16_t;
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"

// One BMxBN output tile. Both operands are read K-major (x row-major [M,K], W
// row-major [N,K]), so both loaders are the plain BlockLoader with a BK-wide
// inner dimension and BlockMMA runs transpose_b = true -- exactly the
// arrangement affine_qmm_steel uses, minus the dequantization.
template <typename T, int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void dense_steel_impl(
    const device T* x,
    const device T* w,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    uint3 tid,
    uint simd_gid,
    uint simd_lid)
{
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int tgp_size = WM * WN * SIMD_SIZE;

  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, /*transpose_a=*/false, /*transpose_b=*/true,
      BK_padded, BK_padded, float>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, tgp_size>;
  using loader_w_t =
      mlx::steel::BlockLoader<T, BN, BK, BK_padded, 1, tgp_size>;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  if (y_row >= M || y_col >= N) { return; }

  x += (int64_t)y_row * K;
  w += (int64_t)y_col * K;
  y += (int64_t)y_row * N + y_col;

  const short num_els  = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(w, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  // K is a runtime value; the tail iteration is bound-checked on BK. The four
  // arms specialize the common case (full tile, aligned K) so its inner loop
  // carries no predication -- the same shape the quantized kernel uses.
  const int K_full = (K / BK) * BK;
  const bool full_m = (num_els  == BM);
  const bool full_n = (num_outs == BN);

  if (full_m && full_n) {
    for (int k = 0; k < K_full; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_unsafe();
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  } else {
    for (int k = 0; k < K_full; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_safe(short2(BK, num_outs));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
      loader_x.next();
      loader_w.next();
    }
  }
  if (K_full < K) {                      // K tail (K % BK != 0)
    const short kr = (short)(K - K_full);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_safe(short2(kr, num_els));
    loader_w.load_safe(short2(kr, num_outs));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!full_m || !full_n) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

#define VPIPE_DENSE_STEEL(NAME, BM_, BN_, BK_, WM_, WN_)                      \
  kernel void NAME(                                                          \
      const device VPIPE_ELT* x [[buffer(0)]],                               \
      const device VPIPE_ELT* w [[buffer(1)]],                               \
      const device VPIPE_ELT* unused [[buffer(2)]],                          \
      device VPIPE_ELT* y [[buffer(3)]],                                     \
      const constant int& K [[buffer(4)]],                                   \
      const constant int& N [[buffer(5)]],                                   \
      const constant int& M [[buffer(6)]],                                   \
      const constant int& unused2 [[buffer(7)]],                             \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]])                           \
  {                                                                          \
    (void)unused; (void)unused2;                                             \
    constexpr int BKp = BK_ + 16 / sizeof(VPIPE_ELT);                        \
    threadgroup VPIPE_ELT Xs[BM_ * BKp];                                     \
    threadgroup VPIPE_ELT Ws[BN_ * BKp];                                     \
    dense_steel_impl<VPIPE_ELT, BM_, BN_, BK_, WM_, WN_>(                    \
        x, w, y, Xs, Ws, K, N, M, tid, simd_gid, simd_lid);                  \
  }

// Tile shapes to pick from at dispatch. The name carries BMxBNxBK so the host
// side reads unambiguously; all use 4 simdgroups (128 threads).
VPIPE_DENSE_STEEL(dense_gemm_steel_64x64x16_f16, 64, 64, 16, 2, 2)
VPIPE_DENSE_STEEL(dense_gemm_steel_64x64x32_f16, 64, 64, 32, 2, 2)
VPIPE_DENSE_STEEL(dense_gemm_steel_32x32x32_f16, 32, 32, 32, 4, 1)
VPIPE_DENSE_STEEL(dense_gemm_steel_64x32x32_f16, 64, 32, 32, 2, 2)
VPIPE_DENSE_STEEL(dense_gemm_steel_32x64x32_f16, 32, 64, 32, 2, 2)
// Wider tiles on 8 simdgroups (256 threads): more output per operand byte
// staged through threadgroup memory, in case the barrier cadence rather than
// the MMA issue rate is what caps the 4-simdgroup tiles above.
VPIPE_DENSE_STEEL(dense_gemm_steel_128x64x16_f16, 128, 64, 16, 4, 2)
VPIPE_DENSE_STEEL(dense_gemm_steel_64x128x16_f16, 64, 128, 16, 2, 4)
VPIPE_DENSE_STEEL(dense_gemm_steel_128x128x16_f16, 128, 128, 16, 4, 2)
