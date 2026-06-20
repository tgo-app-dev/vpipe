// dense_gemm.metal -- plain (non-quantized) f16 GEMM with optional bias,
// for the Qwen3-VL vision tower whose Linear weights are dense bf16/f16
// (the text path's affine_qmm/qmv handle the 4-bit quantized case). HF
// nn.Linear stores weight as [N, K], so we compute
//   y[m, n] = bias[n] + sum_k x[m, k] * W[n, k]
// i.e. x @ W^T. Classic shared-memory tiled GEMM (TILE x TILE output
// block, TILE-deep K steps), f16 storage with f32 accumulation. Tails in
// M/N/K handled by zero-padding the staged tiles. Correctness-first (not
// steel-tuned); the vision tower is not yet the perf-critical path.
//
//   0:x[M,K] half  1:W[N,K] half  2:bias[N] half  3:y[M,N] half(out)
//   4:M  5:N  6:K  7:has_bias(int, 0/1)
// grid (N, M, 1) padded to TILE; threadgroup (TILE, TILE, 1).

#include <metal_stdlib>
using namespace metal;

// Element type: half by default; -DVPIPE_ELT=bfloat builds the bf16
// variant metallib (dense_gemm_bf16). Accumulation stays f32 (the steel
// BlockMMA accumulates in fp32 regardless), so the bf16 path is as
// accurate as f16. Function names keep the "_f16" label (a legacy tag,
// matching rms_norm/sdpa/etc.); the loader picks the right metallib.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define DG_TILE 16

kernel void dense_gemm_bias_f16(
    const device VPIPE_ELT* x    [[buffer(0)]],
    const device VPIPE_ELT* W    [[buffer(1)]],
    const device VPIPE_ELT* bias [[buffer(2)]],
    device VPIPE_ELT*       y    [[buffer(3)]],
    constant int&      M    [[buffer(4)]],
    constant int&      N    [[buffer(5)]],
    constant int&      Kd   [[buffer(6)]],
    constant int&      has_bias [[buffer(7)]],
    uint3 tpit [[thread_position_in_threadgroup]],
    uint3 tgig [[threadgroup_position_in_grid]])
{
  threadgroup VPIPE_ELT As[DG_TILE][DG_TILE];
  threadgroup VPIPE_ELT Bs[DG_TILE][DG_TILE];
  const int tx = (int)tpit.x;
  const int ty = (int)tpit.y;
  const int m = (int)tgig.y * DG_TILE + ty;   // output row
  const int n = (int)tgig.x * DG_TILE + tx;   // output col

  float acc = 0.0f;
  for (int k0 = 0; k0 < Kd; k0 += DG_TILE) {
    // As[ty][tx] = x[m, k0+tx]; Bs[ty][tx] = W[n_of_tx, k0+ty].
    const int ax = k0 + tx;
    As[ty][tx] = (m < M && ax < Kd) ? x[(uint)m * Kd + ax] : (VPIPE_ELT)0;
    const int bn = (int)tgig.x * DG_TILE + ty;   // the W-row this slot feeds
    const int bk = k0 + tx;
    Bs[ty][tx] = (bn < N && bk < Kd) ? W[(uint)bn * Kd + bk] : (VPIPE_ELT)0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < DG_TILE; ++t) {
      // x[m, k0+t] = As[ty][t]; W[n, k0+t] = Bs[tx][t].
      acc += (float)As[ty][t] * (float)Bs[tx][t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (m < M && n < N) {
    if (has_bias) { acc += (float)bias[n]; }
    y[(uint)m * N + n] = (VPIPE_ELT)acc;
  }
}

// ===================================================================
// Steel MMA dense GEMM -- the fast path. y[M,N] = x[M,K] @ W[N,K]^T + bias
// (HF nn.Linear stores W as [N,K], so this is x @ W^T). Reuses MLX's
// vendored steel BlockMMA + BlockLoader, the same machinery
// affine_qmm_steel uses for the 4/8-bit path, minus the dequant (a plain
// BlockLoader for W instead of QuantizedBlockLoader). Far faster than the
// scalar dense_gemm_bias_f16 above (kept as a fallback). Bias is folded
// into the f32 accumulator via apply_epilogue (a [1,N] row broadcast,
// ldc=0/fdc=1) BEFORE the single store-rounding to half -- so the result
// is as accurate as the scalar path, no extra pass.
//
//   0:x[M,K] 1:W[N,K] 2:bias[N] 3:y[M,N] (half)  4:K 5:N 6:M 7:has_bias
// threadgroup (32, WN, WM) = (32,2,2); grid threadgroups
// (ceil(N/32), ceil(M/32), 1). Assumes K % 32 == 0 (all vision/audio
// Linear K are multiples of 32); M/N tails are handled in-kernel.
// ===================================================================

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#define MLX_MTL_CONST static constant constexpr const
#define SIMD_SIZE 32

// steel mma.h has a complex64_t BlockMMA specialization (unused here but
// must parse); complex.h references their bfloat16_t wrapper for an unused
// ctor -- alias it to the native type (we never touch bf16 or complex).
typedef bfloat16 bfloat16_t;
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"

// Binary epilogue: add a (broadcast) bias element to the f32 accumulator.
struct BiasAddEpilogue {
  template <typename AccumT, typename CT>
  METAL_FUNC AccumT apply(AccumT x, CT c) const {
    return x + static_cast<AccumT>(c);
  }
};

template <
    typename T,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void dense_gemm_t_impl(
    const device T* x,
    const device T* W,
    const device T* bias,
    device T*       y,
    threadgroup T*  Xs,
    threadgroup T*  Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& has_bias,
    uint3 tid,
    uint  simd_gid,
    uint  simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      mlx::steel::BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  W += y_col * static_cast<int64_t>(K);
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(W, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Fold bias into the f32 accumulator as a [1, N] row broadcast (ldc=0,
  // fdc=1, base bias + y_col) before the single round-to-half on store.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    if (has_bias) {
      BiasAddEpilogue op;
      mma_op.apply_epilogue_safe(bias + y_col, 0, 1, short2(num_outs, num_els),
                                 op);
    }
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    if (has_bias) {
      BiasAddEpilogue op;
      mma_op.apply_epilogue(bias + y_col, 0, 1, op);
    }
    mma_op.store_result(y, N);
  }
}

kernel void dense_gemm_t_f16(
    const device VPIPE_ELT*  x        [[buffer(0)]],
    const device VPIPE_ELT*  W        [[buffer(1)]],
    const device VPIPE_ELT*  bias     [[buffer(2)]],
    device VPIPE_ELT*        y        [[buffer(3)]],
    const constant int& K        [[buffer(4)]],
    const constant int& N        [[buffer(5)]],
    const constant int& M        [[buffer(6)]],
    const constant int& has_bias [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_impl<VPIPE_ELT, /*aligned_N=*/false, BM, BK, BN>(
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, tid, simd_gid, simd_lid);
}

// ----------------------------------------------------------------------
// Dense f16 GEMV (M=1): y[1, N] = x[1, K] @ W[N, K]^T, W row-major [N, K].
// The steel BlockMMA dense_gemm_t above is tiled for large M (32-row BM);
// at decode M=1 it computes a 32-row output tile for one real row (31
// wasted) with K/32 threadgroup barriers per tile -- ~half DRAM bandwidth.
// This GEMV mirrors qmv_fast's bandwidth-optimal structure (each simdgroup
// owns RPS output rows; the 32 lanes split K into VPT-wide strips; W read
// exactly once, no staging, no barriers) for the e4b PLE's two f16
// projections (per_layer_model_projection + per_layer_projection). Dispatch
// matches the qmv grid: {32, N/RPS, 1} / threadgroup {32, NSG, 1}.
#ifndef DG_SIMD_SIZE
#define DG_SIMD_SIZE 32
#endif

template <typename T, int RPS, int VPT, int NSG>
METAL_FUNC void dense_gemv_t_impl(
    const device T*     x,
    const device T*     W,
    device T*           y,
    const constant int& K,
    const constant int& N,
    uint3 tid,
    uint  simd_gid,
    uint  simd_lid) {
  typedef float U;
  constexpr int block_size = VPT * DG_SIMD_SIZE;
  thread U x_thread[VPT];
  U result[RPS] = {0};

  const int out_row = tid.y * (NSG * RPS) + simd_gid * RPS;
  W += (int64_t)out_row * K + simd_lid * VPT;
  x += simd_lid * VPT;
  y += out_row;

  for (int k = 0; k < K; k += block_size) {
    const int v = K - (k + (int)simd_lid * VPT);   // valid els this lane
    if (v >= VPT) {
#pragma clang loop unroll(full)
      for (int i = 0; i < VPT; i++) { x_thread[i] = (U)x[i]; }
      for (int row = 0; row < RPS; row++) {
        const device T* wl = W + (int64_t)row * K;
        U acc = 0;
#pragma clang loop unroll(full)
        for (int i = 0; i < VPT; i++) { acc += x_thread[i] * (U)wl[i]; }
        result[row] += acc;
      }
    } else if (v > 0) {
      for (int i = 0; i < VPT; i++) {
        x_thread[i] = (i < v) ? (U)x[i] : (U)0;
      }
      for (int row = 0; row < RPS; row++) {
        const device T* wl = W + (int64_t)row * K;
        U acc = 0;
        for (int i = 0; i < v; i++) { acc += x_thread[i] * (U)wl[i]; }
        result[row] += acc;
      }
    }
    W += block_size;
    x += block_size;
  }
  for (int row = 0; row < RPS; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && (out_row + row) < N) { y[row] = (T)r; }
  }
}

kernel void dense_gemv_t_f16(
    const device VPIPE_ELT* x        [[buffer(0)]],
    const device VPIPE_ELT* W        [[buffer(1)]],
    device VPIPE_ELT*       y        [[buffer(2)]],
    const constant int&     K        [[buffer(3)]],
    const constant int&     N        [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  dense_gemv_t_impl<VPIPE_ELT, /*RPS=*/4, /*VPT=*/8, /*NSG=*/2>(
      x, W, y, K, N, tid, simd_gid, simd_lid);
}
