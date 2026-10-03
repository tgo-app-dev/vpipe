// train_gemm.metal -- the GEMM shapes LoRA training needs that the
// forward's kernels do not have, on MLX's steel BlockMMA.
//
// Every dense GEMM elsewhere in the tree is NT: y = x W^T with W stored
// [N, K]. The backward's LARGE products reuse those unchanged by being
// handed W^T (train_transpose_2d), so they keep every route -- steel
// tiles, matrix cores, int8, the ANE. What is left is small and has no
// NT form:
//
//   TN, reduced over the ROWS, f32 out, accumulated:
//     C[P, Q] += alpha * A[M, P]^T B[M, Q]
//     a LoRA factor's gradient -- dB = dY^T t, dA = u^T x -- where the
//     reduction is thousands of token rows long and the output is a
//     rank-wide strip.
//   NN, storage type out, optionally accumulated:
//     C[M, N] (+)= alpha * A[M, K] B[K, N]
//     u = dY B and the LoRA term of dX, s * u A: B and A are the adapter's
//     factors, too small to be worth transposing.
//
// One output tile per threadgroup, the whole reduction inside it, a
// fixed order: deterministic, no atomics.
//
// Buffers: 0:A 1:B 2:C (storage type) 3:C (f32) 4:TrainGemmParams.
// Grid: threadgroups (ceil(N/BN), ceil(M/BM)), WM*WN*32 threads.
//
// The gemm/* steel headers define BlockLoader / BlockMMA / MMATile; do
// NOT also include the attn/* ones in this translation unit.

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

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define SIMD_SIZE 32

typedef bfloat bfloat16_t;
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"

struct TrainGemmParams {
  int   M;
  int   N;
  int   K;
  int   lda;
  int   ldb;
  int   ldc;
  float alpha;
  int   accumulate;   // add into C rather than overwrite it
  // A batch of independent products (the attention's heads), one per
  // grid z: element offsets between consecutive A, B and C.
  long  a_bstride;
  long  b_bstride;
  long  c_bstride;
};

// C = alpha op(A) op(B) (+ C). op(A) is A [M, K] (lda) or, with TA, A
// stored [K, M]; op(B) is B [K, N] (ldb) or, with TB, B stored [N, K].
// OUT_F32 selects which C is written.
template <typename T, int BM, int BN, int BK, int WM, int WN, bool TA,
          bool TB, bool OUT_F32>
METAL_FUNC void train_gemm_impl(
    const device T* A,
    const device T* B,
    device T* Ct,
    device float* Cf,
    const constant TrainGemmParams& p,
    threadgroup T* As,
    threadgroup T* Bs,
    uint3 tid,
    uint simd_gid,
    uint simd_lid)
{
  constexpr short pad = 16 / sizeof(T);
  constexpr short lda_tgp = TA ? BM + pad : BK + pad;
  constexpr short ldb_tgp = TB ? BK + pad : BN + pad;
  constexpr short tgp_size = WM * WN * SIMD_SIZE;

  using loader_a_t = mlx::steel::BlockLoader<
      T, TA ? BK : BM, TA ? BM : BK, lda_tgp, !TA, tgp_size>;
  using loader_b_t = mlx::steel::BlockLoader<
      T, TB ? BN : BK, TB ? BK : BN, ldb_tgp, TB, tgp_size>;
  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, TA, TB, lda_tgp, ldb_tgp, float>;

  const int c_row = (int)tid.y * BM;
  const int c_col = (int)tid.x * BN;
  if (c_row >= p.M || c_col >= p.N) { return; }
  A += (long)tid.z * p.a_bstride;
  B += (long)tid.z * p.b_bstride;
  Ct += (long)tid.z * p.c_bstride;
  Cf += (long)tid.z * p.c_bstride;

  A += TA ? (long)c_row : (long)c_row * p.lda;
  B += TB ? (long)c_col * p.ldb : (long)c_col;

  const short tbm = (short)min(BM, p.M - c_row);
  const short tbn = (short)min(BN, p.N - c_col);

  loader_a_t loader_a(A, p.lda, As, simd_gid, simd_lid);
  loader_b_t loader_b(B, p.ldb, Bs, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  const short2 dims_a = TA ? short2(tbm, BK) : short2(BK, tbm);
  const short2 dims_b = TB ? short2(BK, tbn) : short2(tbn, BK);
  const bool full = tbm == BM && tbn == BN;
  const int k_full = (p.K / BK) * BK;
  for (int k = 0; k < k_full; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full) {
      loader_a.load_unsafe();
      loader_b.load_unsafe();
    } else {
      loader_a.load_safe(dims_a);
      loader_b.load_safe(dims_b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
  if (k_full < p.K) {
    const short kr = (short)(p.K - k_full);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_safe(TA ? short2(tbm, kr) : short2(kr, tbm));
    loader_b.load_safe(TB ? short2(kr, tbn) : short2(tbn, kr));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }

  // The epilogue, element by element: alpha, the optional accumulate,
  // and the bounds, which the library's store has no accumulate for.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < mma_t::TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < mma_t::TN; j++) {
      const int r = c_row + mma_op.sm + i * mma_t::TM_stride;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 2; e++) {
        const int c = c_col + mma_op.sn + j * mma_t::TN_stride + e;
        if (r < p.M && c < p.N) {
          const float v = p.alpha * mma_op.Ctile.frag_at(i, j)[e];
          const long o = (long)r * p.ldc + c;
          if (OUT_F32) {
            Cf[o] = p.accumulate != 0 ? Cf[o] + v : v;
          } else {
            Ct[o] = T(p.accumulate != 0 ? float(Ct[o]) + v : v);
          }
        }
      }
    }
  }
}

#define VPIPE_TRAIN_GEMM(NAME, BM_, BN_, BK_, WM_, WN_, TA_, TB_, F32_)       \
  kernel void NAME(                                                          \
      const device VPIPE_ELT* A [[buffer(0)]],                               \
      const device VPIPE_ELT* B [[buffer(1)]],                               \
      device VPIPE_ELT* Ct [[buffer(2)]],                                    \
      device float* Cf [[buffer(3)]],                                        \
      const constant TrainGemmParams& p [[buffer(4)]],                       \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]])                           \
  {                                                                          \
    constexpr short pad = 16 / sizeof(VPIPE_ELT);                            \
    threadgroup VPIPE_ELT As[TA_ ? BK_ * (BM_ + pad) : BM_ * (BK_ + pad)];   \
    threadgroup VPIPE_ELT Bs[TB_ ? BN_ * (BK_ + pad) : BK_ * (BN_ + pad)];   \
    train_gemm_impl<VPIPE_ELT, BM_, BN_, BK_, WM_, WN_, TA_, TB_, F32_>(     \
        A, B, Ct, Cf, p, As, Bs, tid, simd_gid, simd_lid);                   \
  }

// The LoRA gradients: a rank-wide strip against a hidden-wide one. 32x64
// for dA [r, K] (few rows, many columns), 64x32 for dB [N, r].
VPIPE_TRAIN_GEMM(train_gemm_tn_f32_32x64, 32, 64, 16, 2, 2, true, false, true)
VPIPE_TRAIN_GEMM(train_gemm_tn_f32_64x32, 64, 32, 16, 2, 2, true, false, true)
// u = dY B and dX += s u A: NN into the storage type.
VPIPE_TRAIN_GEMM(train_gemm_nn_64x32, 64, 32, 16, 2, 2, false, false, false)
VPIPE_TRAIN_GEMM(train_gemm_nn_64x64, 64, 64, 16, 2, 2, false, false, false)
// NT into the storage type, for the tests' reference and the odd small
// product that has no forward route.
VPIPE_TRAIN_GEMM(train_gemm_nt_64x64, 64, 64, 16, 2, 2, false, true, false)
// The attention backward's GEMM form, batched over heads: dV += P^T dO and
// dK += dS^T Q into the f32 accumulators (NN), and dQ = dS K from the
// stored dS^T (TN).
VPIPE_TRAIN_GEMM(train_gemm_nn_f32_64x64, 64, 64, 16, 2, 2, false, false,
                 true)
VPIPE_TRAIN_GEMM(train_gemm_tn_64x64, 64, 64, 16, 2, 2, true, false, false)
