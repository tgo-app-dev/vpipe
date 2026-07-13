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

// The _acc16 entry points run the simdgroup MMA in half (FP16 pipe). In the
// bf16-flavor build (VPIPE_ELT=bfloat) they degrade to f32 accumulate --
// bfloat and half fragments don't interconvert in MSL, and nothing routes a
// bf16 _acc16 -- so the entries still compile in both metallibs.
typedef metal::conditional<
    metal::is_same<VPIPE_ELT, bfloat>::value, float, half>::type
    VPIPE_ACC16_T;

// Binary epilogue: add a (broadcast) bias element to the f32 accumulator.
struct BiasAddEpilogue {
  template <typename AccumT, typename CT>
  METAL_FUNC AccumT apply(AccumT x, CT c) const {
    return x + static_cast<AccumT>(c);
  }
};

// CAUSAL: 0 = dense (default). 1 = QK (scores[query,key]): skip a whole tile
// when its smallest key column (y_col) is above the block's largest query
// (q_offset+y_row+BM-1) -- those columns are masked away by the downstream
// causal softmax, so the never-written output is never read. 2 = PV
// (out[query,D], contraction over keys): P[r,j]=0 for keys j>q_offset+r, so
// cap the K-loop at the block's last query (rounded up to BK). Both halve the
// strictly-upper-triangle work the materialized global path otherwise wastes.
//
// window > 0 adds the TRAILING-window lower bound (Gemma-4 sliding layers):
// query q attends keys [q_offset+q-window+1, q_offset+q]. QK also skips tiles
// fully BELOW the window; PV also starts the contraction at the band's lowest
// key (rounded down to BK). With window <= 0 the band has no lower bound (pure
// causal == the global path). Net: O(n*window) banded work, not O(n^2).
// AccumT: the simdgroup-MMA element type. float (default) widens every
// fragment and runs on the FP32 pipe; half keeps the multiply-accumulate on
// the FP16 pipe, which only pays where the f16 matrix rate exceeds f32.
// MEASURED on M4 Pro (dg_mma_rate probes below): f16 == f32 rate (~10.1 vs
// 10.4 TFLOP/s), so _acc16 entries are perf-neutral there and stay behind
// an accuracy-verified opt-in.
template <
    typename T,
    const bool aligned_N,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    const int CAUSAL = 0,
    typename AccumT = float>
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
    const int q_offset,
    const int window,
    uint3 tid,
    uint  simd_gid,
    uint  simd_lid) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded, AccumT>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      mlx::steel::BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  if (CAUSAL == 1) {
    // Above the causal diagonal: smallest key (y_col) past largest query.
    if (y_col > q_offset + y_row + BM - 1) { return; }
    // Below the trailing window: largest key (y_col+BN-1) before the smallest
    // row's window start (q_offset+y_row-window+1).
    if (window > 0 && y_col + BN - 1 < q_offset + y_row - window + 1) {
      return;
    }
  }
  // PV (CAUSAL==2): only keys in [k0, Kc) can be nonzero for this row block.
  int Kc = K;
  int k0 = 0;
  if (CAUSAL == 2) {
    const int kmax = q_offset + y_row + BM;       // exclusive last key + 1
    Kc = (kmax < K) ? ((kmax + BK - 1) / BK) * BK : K;
    if (window > 0) {
      // Smallest valid key over the block's rows (row y_row), rounded DOWN to
      // a BK tile boundary. Keys below contribute 0 (softmax masked them).
      const int lo = q_offset + y_row - window + 1;
      k0 = (lo > 0) ? (lo / BK) * BK : 0;
    }
  }
  x += y_row * static_cast<int64_t>(K);
  W += y_col * static_cast<int64_t>(K);
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x + k0, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(W + k0, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  // Single loop with per-iteration K-tail masking. The original four-branch
  // form passed BK as the load's K-extent unconditionally -- correct ONLY when
  // K % BK == 0 (the comment above documents that assumption). For K not a
  // multiple of BK (e.g. the materialized-attention PV, K = T_kv) the final
  // block's tail columns [K%BK, BK) are invalid; load_unsafe / load_safe(BK,..)
  // read them anyway, spilling into the next packed row (next score row / next
  // kv-head's V^T) -> garbage. Clamp the K-extent to (K - k) on the tail block
  // so the loader zero-pads it. For K % BK == 0, kt == BK every block, so this
  // is byte-identical to the old fast path (load_unsafe on full tiles).
  const bool n_tail = !aligned_N && num_outs < BN;
  for (int k = k0; k < Kc; k += BK) {
    const short kt = (K - k) < BK ? (short)(K - k) : (short)BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (num_els < BM || kt < BK) {
      loader_x.load_safe(short2(kt, num_els));
    } else {
      loader_x.load_unsafe();
    }
    if (n_tail || kt < BK) {
      loader_w.load_safe(short2(kt, num_outs));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
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
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, /*q_offset=*/0, /*window=*/0,
      tid, simd_gid, simd_lid);
}

// Larger-tile variants for tall-N/deep-K prefill GEMMs (e.g. the Krea-2 DiT
// blocks: M~=seq, N/K in {1536,6144,16384}). A bigger BM computes more output
// rows per threadgroup with the SAME 128 threads, so each [BN,BK] weight tile is
// re-read from memory M/BM times instead of M/32 -- amortizing the f16 weight
// bandwidth that the BM=32 default leaves on the table. Same math/result as
// dense_gemm_t_f16; pick by shape at dispatch. Grid: x=ceil(N/BN)*32,
// y=ceil(M/BM)*2, z=2 (threadgroup {32,2,2}).
kernel void dense_gemm_t_bm64_f16(
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
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_impl<VPIPE_ELT, /*aligned_N=*/false, BM, BK, BN>(
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, /*q_offset=*/0, /*window=*/0,
      tid, simd_gid, simd_lid);
}

// FP16-pipe (AccumT=half) twin of dense_gemm_t_bm64_f16 (see the AccumT
// comment on dense_gemm_t_impl).
kernel void dense_gemm_t_bm64_acc16_f16(
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
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_impl<VPIPE_ELT, /*aligned_N=*/false, BM, BK, BN, /*CAUSAL=*/0,
                    VPIPE_ACC16_T>(
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, /*q_offset=*/0, /*window=*/0,
      tid, simd_gid, simd_lid);
}

kernel void dense_gemm_t_bm64bn64_f16(
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
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_impl<VPIPE_ELT, /*aligned_N=*/false, BM, BK, BN>(
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, /*q_offset=*/0, /*window=*/0,
      tid, simd_gid, simd_lid);
}

// Fused SwiGLU dense GEMM: y[m, n/2] = silu(gate)*up, where W is the
// INTERLEAVED [gate|up] projection (output col 2g = gate_g, 2g+1 = up_g) and
// N = 2*ffn is the fused width. Mirrors dense_gemm_t_impl's BlockMMA loop; only
// the store epilogue differs (register-local silu(gate)*up, output width N/2)
// -- kills the [M, 2*ffn] intermediate + the separate slice/swiglu passes. N is
// assumed BN-aligned (2*ffn % BN == 0); only the M tail is guarded.
template <typename T, const int BM = 64, const int BK = 32, const int BN = 32,
          typename AccumT = float>
METAL_FUNC void dense_gemm_t_swiglu_impl(
    const device T* x,
    const device T* W,
    device T*       y,
    threadgroup T*  Xs,
    threadgroup T*  Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int out_stride,   // output row stride (0 -> contiguous N/2)
    const int out_off,      // output column offset (write into y[:, off:])
    uint3 tid,
    uint  simd_gid,
    uint  simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded, AccumT>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      mlx::steel::BlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * static_cast<int64_t>(K);
  W += y_col * static_cast<int64_t>(K);

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(W, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < K; k += BK) {
    const short kt = (K - k) < BK ? (short)(K - k) : (short)BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (num_els < BM || kt < BK) {
      loader_x.load_safe(short2(kt, num_els));
    } else {
      loader_x.load_unsafe();
    }
    if (num_outs < BN || kt < BK) {
      loader_w.load_safe(short2(kt, num_outs));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Fused SwiGLU store (mirrors qmm_t_swiglu_impl): each accumulator frag holds
  // (gate, up) in elems [0] (even col) / [1] (odd col); emit silu(gate)*up to
  // y[row, col/2]. Output width is N/2.
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = out_stride > 0 ? out_stride : N / 2;   // row stride
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;   // even fused column
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];               // gate (even col)
      const float up = (float)fr[1];               // up   (odd col)
      const float s = ga / (1.0f + metal::exp(-ga));
      y[(int64_t)row * outN + out_off + (col >> 1)] = (T)(s * up);
    }
  }
}

kernel void dense_gemm_swiglu_bm64_f16(
    const device VPIPE_ELT* x [[buffer(0)]],
    const device VPIPE_ELT* W [[buffer(1)]],
    device VPIPE_ELT*       y [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    const constant int& out_stride [[buffer(6)]],
    const constant int& out_off    [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_swiglu_impl<VPIPE_ELT, BM, BK, BN>(
      x, W, y, Xs, Ws, K, N, M, out_stride, out_off, tid, simd_gid, simd_lid);
}

kernel void dense_gemm_swiglu_bm64_acc16_f16(
    const device VPIPE_ELT* x [[buffer(0)]],
    const device VPIPE_ELT* W [[buffer(1)]],
    device VPIPE_ELT*       y [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    const constant int& out_stride [[buffer(6)]],
    const constant int& out_off    [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_swiglu_impl<VPIPE_ELT, BM, BK, BN, VPIPE_ACC16_T>(
      x, W, y, Xs, Ws, K, N, M, out_stride, out_off, tid, simd_gid, simd_lid);
}

// Causal/banded QK (scores = Q @ K^T): skips score tiles above the causal
// diagonal and (window>0) below the trailing window. q_offset is the query's
// absolute start (kv_off); has_bias unused (pass 0). window<=0 = pure causal
// (global); window>0 = Gemma-4 sliding band. Materialized-attention prefill.
kernel void dense_gemm_t_qkcausal_f16(
    const device VPIPE_ELT*  x        [[buffer(0)]],
    const device VPIPE_ELT*  W        [[buffer(1)]],
    const device VPIPE_ELT*  bias     [[buffer(2)]],
    device VPIPE_ELT*        y        [[buffer(3)]],
    const constant int& K        [[buffer(4)]],
    const constant int& N        [[buffer(5)]],
    const constant int& M        [[buffer(6)]],
    const constant int& has_bias [[buffer(7)]],
    const constant int& q_offset [[buffer(8)]],
    const constant int& window   [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_impl<VPIPE_ELT, /*aligned_N=*/false, BM, BK, BN, /*CAUSAL=*/1>(
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, q_offset, window, tid, simd_gid,
      simd_lid);
}

// Causal/banded PV (out = P @ V^T, V_T row-major [D, T_kv]): contracts only
// keys in [k0, Kc) -- caps at the row block's last query, and (window>0)
// starts at the band's lowest key, since softmax zeroed P outside the band.
kernel void dense_gemm_t_pvcausal_f16(
    const device VPIPE_ELT*  x        [[buffer(0)]],
    const device VPIPE_ELT*  W        [[buffer(1)]],
    const device VPIPE_ELT*  bias     [[buffer(2)]],
    device VPIPE_ELT*        y        [[buffer(3)]],
    const constant int& K        [[buffer(4)]],
    const constant int& N        [[buffer(5)]],
    const constant int& M        [[buffer(6)]],
    const constant int& has_bias [[buffer(7)]],
    const constant int& q_offset [[buffer(8)]],
    const constant int& window   [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  dense_gemm_t_impl<VPIPE_ELT, /*aligned_N=*/false, BM, BK, BN, /*CAUSAL=*/2>(
      x, W, bias, y, Xs, Ws, K, N, M, has_bias, q_offset, window, tid, simd_gid,
      simd_lid);
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

// ---- Dense f16 Mixture-of-Experts gather GEMVs (raw-HF bf16 MoE) ---------
// The dense twins of affine_gather_qmv_swiglu_w4g64 / affine_gather_down /
// affine_moe_gate: same on-GPU routing contract (pair_eid selects the expert
// slab), plain f16 weights (no scales/biases). They plug into the identical
// moe_route / moe_combine / moe_finalize flow as the affine path, so a dense
// MoE forward is the affine forward with these GEMVs swapped in.

// Gather gate|up + fused SwiGLU: per pair p, expert e = pair_eid[p]; read the
// expert slab w[e] [2*inner, K] (interleaved row 2g=gate, 2g+1=up), x row =
// p/top_k, write act[p, inner] = silu(gate)*up. grid {32, (2*inner)/8, npair},
// tg {32, 2, 1}; mirrors qmv_swiglu_impl's (g0,u0,g1,u1) RPS=4 layout.
kernel void dense_moe_gather_swiglu_f16(
    const device VPIPE_ELT* w        [[buffer(0)]],   // [E, 2*inner, K]
    const device VPIPE_ELT* x        [[buffer(1)]],   // [M, K] input rows
    device VPIPE_ELT*       y         [[buffer(2)]],   // [npair, inner]
    const constant int&     K         [[buffer(3)]],   // in_vec_size
    const constant int&     N         [[buffer(4)]],   // 2*inner
    const device int*       pair_eid  [[buffer(5)]],   // [npair]
    const constant int&     top_k     [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int VPT = 8, NSG = 2, RPS = 4;
  constexpr int block_size = VPT * DG_SIMD_SIZE;
  typedef float U;
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const device VPIPE_ELT* we = w + (int64_t)e * (int64_t)N * (int64_t)K;
  const device VPIPE_ELT* xr = x + (int64_t)(p / top_k) * (int64_t)K;
  const int out_row = (int)tid.y * (NSG * RPS) + (int)simd_gid * RPS;
  const device VPIPE_ELT* wl = we + (int64_t)out_row * K + simd_lid * VPT;
  const device VPIPE_ELT* xl = xr + simd_lid * VPT;
  thread U x_thread[VPT];
  U result[RPS] = {0};
  for (int k = 0; k < K; k += block_size) {
    const int v = K - (k + (int)simd_lid * VPT);
    if (v >= VPT) {
#pragma clang loop unroll(full)
      for (int i = 0; i < VPT; i++) { x_thread[i] = (U)xl[i]; }
      for (int row = 0; row < RPS; row++) {
        const device VPIPE_ELT* wr = wl + (int64_t)row * K;
        U acc = 0;
#pragma clang loop unroll(full)
        for (int i = 0; i < VPT; i++) { acc += x_thread[i] * (U)wr[i]; }
        result[row] += acc;
      }
    } else if (v > 0) {
      for (int i = 0; i < VPT; i++) {
        x_thread[i] = (i < v) ? (U)xl[i] : (U)0;
      }
      for (int row = 0; row < RPS; row++) {
        const device VPIPE_ELT* wr = wl + (int64_t)row * K;
        U acc = 0;
        for (int i = 0; i < v; i++) { acc += x_thread[i] * (U)wr[i]; }
        result[row] += acc;
      }
    }
    wl += block_size;
    xl += block_size;
  }
  for (int row = 0; row < RPS; row++) { result[row] = simd_sum(result[row]); }
  if (simd_lid == 0) {
    // out_row even -> (g0,u0,g1,u1) for inner features out_row/2 and +1.
    device VPIPE_ELT* yo = y + (int64_t)p * (int64_t)(N / 2) + (out_row >> 1);
    const U g0 = result[0], u0 = result[1], g1 = result[2], u1 = result[3];
    yo[0] = (VPIPE_ELT)((g0 / (1.0f + metal::exp(-g0))) * u0);
    yo[1] = (VPIPE_ELT)((g1 / (1.0f + metal::exp(-g1))) * u1);
  }
}

// Gemma-4 MoE twin of dense_moe_gather_swiglu_f16: identical gather + RPS=4
// (g0,u0,g1,u1) layout over the interleaved [E, 2*inner, K] slab, but the
// activation is gelu_pytorch_tanh(gate)*up (matches geglu_f16 / qmv_geglu_impl)
// instead of silu(gate)*up. Used by the raw-HF bf16 Gemma MoE expert gather.
kernel void dense_moe_gather_geglu_f16(
    const device VPIPE_ELT* w        [[buffer(0)]],   // [E, 2*inner, K]
    const device VPIPE_ELT* x        [[buffer(1)]],   // [M, K] input rows
    device VPIPE_ELT*       y         [[buffer(2)]],   // [npair, inner]
    const constant int&     K         [[buffer(3)]],   // in_vec_size
    const constant int&     N         [[buffer(4)]],   // 2*inner
    const device int*       pair_eid  [[buffer(5)]],   // [npair]
    const constant int&     top_k     [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int VPT = 8, NSG = 2, RPS = 4;
  constexpr int block_size = VPT * DG_SIMD_SIZE;
  typedef float U;
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const device VPIPE_ELT* we = w + (int64_t)e * (int64_t)N * (int64_t)K;
  const device VPIPE_ELT* xr = x + (int64_t)(p / top_k) * (int64_t)K;
  const int out_row = (int)tid.y * (NSG * RPS) + (int)simd_gid * RPS;
  const device VPIPE_ELT* wl = we + (int64_t)out_row * K + simd_lid * VPT;
  const device VPIPE_ELT* xl = xr + simd_lid * VPT;
  thread U x_thread[VPT];
  U result[RPS] = {0};
  for (int k = 0; k < K; k += block_size) {
    const int v = K - (k + (int)simd_lid * VPT);
    if (v >= VPT) {
#pragma clang loop unroll(full)
      for (int i = 0; i < VPT; i++) { x_thread[i] = (U)xl[i]; }
      for (int row = 0; row < RPS; row++) {
        const device VPIPE_ELT* wr = wl + (int64_t)row * K;
        U acc = 0;
#pragma clang loop unroll(full)
        for (int i = 0; i < VPT; i++) { acc += x_thread[i] * (U)wr[i]; }
        result[row] += acc;
      }
    } else if (v > 0) {
      for (int i = 0; i < VPT; i++) {
        x_thread[i] = (i < v) ? (U)xl[i] : (U)0;
      }
      for (int row = 0; row < RPS; row++) {
        const device VPIPE_ELT* wr = wl + (int64_t)row * K;
        U acc = 0;
        for (int i = 0; i < v; i++) { acc += x_thread[i] * (U)wr[i]; }
        result[row] += acc;
      }
    }
    wl += block_size;
    xl += block_size;
  }
  for (int row = 0; row < RPS; row++) { result[row] = simd_sum(result[row]); }
  if (simd_lid == 0) {
    // out_row even -> (g0,u0,g1,u1) for inner features out_row/2 and +1.
    device VPIPE_ELT* yo = y + (int64_t)p * (int64_t)(N / 2) + (out_row >> 1);
    const U g0 = result[0], u0 = result[1], g1 = result[2], u1 = result[3];
    const U k0 = 0.7978845608028654f;                // sqrt(2/pi)
    const U t0 = metal::precise::tanh(k0 * (g0 + 0.044715f * g0 * g0 * g0));
    const U t1 = metal::precise::tanh(k0 * (g1 + 0.044715f * g1 * g1 * g1));
    yo[0] = (VPIPE_ELT)(0.5f * g0 * (1.0f + t0) * u0);
    yo[1] = (VPIPE_ELT)(0.5f * g1 * (1.0f + t1) * u1);
  }
}

// Gather down: per pair p, expert e = pair_eid[p]; read down slab w[e] [H,
// inner], x = act[p, inner] -> partials[p, H] (UN-weighted; moe_combine
// applies the routing weight). grid {32, H/8, npair}, tg {32, 2, 1}.
kernel void dense_moe_gather_down_f16(
    const device VPIPE_ELT* w        [[buffer(0)]],   // [E, H, inner]
    const device VPIPE_ELT* x        [[buffer(1)]],   // [npair, inner]
    device VPIPE_ELT*       y         [[buffer(2)]],   // [npair, H]
    const constant int&     K         [[buffer(3)]],   // inner
    const constant int&     N         [[buffer(4)]],   // H
    const device int*       pair_eid  [[buffer(5)]],   // [npair]
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const int p = (int)tid.z;
  const uint e = (uint)pair_eid[p];
  const device VPIPE_ELT* we = w + (int64_t)e * (int64_t)N * (int64_t)K;
  const device VPIPE_ELT* xr = x + (int64_t)p * (int64_t)K;
  dense_gemv_t_impl<VPIPE_ELT, /*RPS=*/4, /*VPT=*/8, /*NSG=*/2>(
      xr, we, y + (int64_t)p * (int64_t)N, K, N, tid, simd_gid, simd_lid);
}

// ---- Raw simdgroup-MMA ALU rate probes ------------------------------
// Register-resident simdgroup_multiply_accumulate loops with NO memory
// traffic in the timed loop -- they measure the pure matrix-FMA pipe rate
// (f32 fragments vs f16 fragments) to test whether the GPU has a
// double-rate FP16 pipe for simdgroup MMA. 4 independent accumulators
// hide the FMA latency chain. FLOPs = tgs * simdgroups/tg * iters * 4 *
// 2*8*8*8. Grid: threads {32*SG*TGS,1,1}, tg {32*SG,1,1}.
kernel void dg_mma_rate_acc32(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid      [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(
      0.001f * (float)(simd_lid + 1));
  simdgroup_float8x8 b = make_filled_simdgroup_matrix<float, 8, 8>(1.0009f);
  simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 c1 = c0, c2 = c0, c3 = c0;
  for (int i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0);
    simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2);
    simdgroup_multiply_accumulate(c3, a, b, c3);
  }
  simdgroup_multiply_accumulate(c0, c1, c2, c3);   // fold: defeat DCE
  threadgroup float sink[64];
  simdgroup_store(c0, sink, 8);
  if (gid == 0) { out[0] = sink[simd_lid & 63]; }
}

kernel void dg_mma_rate_acc16(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid      [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8, 8>(
      (half)(0.001f * (float)(simd_lid + 1)));
  simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8, 8>(
      (half)1.0009f);
  simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8, 8>((half)0.0f);
  simdgroup_half8x8 c1 = c0, c2 = c0, c3 = c0;
  for (int i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0);
    simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2);
    simdgroup_multiply_accumulate(c3, a, b, c3);
  }
  simdgroup_multiply_accumulate(c0, c1, c2, c3);   // fold: defeat DCE
  threadgroup half sink[64];
  simdgroup_store(c0, sink, 8);
  if (gid == 0) { out[0] = (float)sink[simd_lid & 63]; }
}

// Mixed-pipe probe: 2 f32 + 2 f16 accumulators per iteration. M3/M4-class
// shader cores advertise PARALLEL F32 and F16 pipes; if they dual-issue,
// this exceeds the single-type rate (up to 2x). If it lands at the
// single-type rate, the pipes share issue slots and a mixed-precision GEMM
// has no extra ALU headroom.
kernel void dg_mma_rate_mixed(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid      [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  simdgroup_float8x8 af = make_filled_simdgroup_matrix<float, 8, 8>(
      0.001f * (float)(simd_lid + 1));
  simdgroup_float8x8 bf = make_filled_simdgroup_matrix<float, 8, 8>(1.0009f);
  simdgroup_float8x8 f0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 f1 = f0;
  simdgroup_half8x8 ah = make_filled_simdgroup_matrix<half, 8, 8>(
      (half)(0.001f * (float)(simd_lid + 1)));
  simdgroup_half8x8 bh = make_filled_simdgroup_matrix<half, 8, 8>(
      (half)1.0009f);
  simdgroup_half8x8 h0 = make_filled_simdgroup_matrix<half, 8, 8>((half)0.0f);
  simdgroup_half8x8 h1 = h0;
  for (int i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(f0, af, bf, f0);
    simdgroup_multiply_accumulate(h0, ah, bh, h0);
    simdgroup_multiply_accumulate(f1, af, bf, f1);
    simdgroup_multiply_accumulate(h1, ah, bh, h1);
  }
  simdgroup_multiply_accumulate(f0, f1, f1, f0);   // fold: defeat DCE
  simdgroup_multiply_accumulate(h0, h1, h1, h0);
  threadgroup float sinkf[64];
  threadgroup half sinkh[64];
  simdgroup_store(f0, sinkf, 8);
  simdgroup_store(h0, sinkh, 8);
  if (gid == 0) { out[0] = sinkf[simd_lid & 63] + (float)sinkh[simd_lid & 63]; }
}

// Scalar-FMA twins: the same probe with plain fused multiply-adds (4
// independent chains of vectorized fma) instead of simdgroup matrices --
// separates "matrix op decomposition" from the raw scalar pipe rate.
// FLOPs = threads * iters * 4 * 2 * 4 (float4 lanes).
kernel void dg_fma_rate_f32(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  float4 a = float4(0.999f + 1e-7f * (float)(gid & 31));
  float4 c0 = float4(0.0f), c1 = float4(0.1f), c2 = float4(0.2f),
         c3 = float4(0.3f);
  for (int i = 0; i < iters; ++i) {
    c0 = fma(a, c0, a);
    c1 = fma(a, c1, a);
    c2 = fma(a, c2, a);
    c3 = fma(a, c3, a);
  }
  const float4 r = c0 + c1 + c2 + c3;
  if (gid == 0) { out[0] = r.x + r.y + r.z + r.w; }
}

kernel void dg_fma_rate_f16(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  half4 a = half4((half)(0.999f + 1e-4f * (float)(gid & 31)));
  half4 c0 = half4(0.0h), c1 = half4(0.1h), c2 = half4(0.2h),
        c3 = half4(0.3h);
  for (int i = 0; i < iters; ++i) {
    c0 = fma(a, c0, a);
    c1 = fma(a, c1, a);
    c2 = fma(a, c2, a);
    c3 = fma(a, c3, a);
  }
  const half4 r = c0 + c1 + c2 + c3;
  if (gid == 0) { out[0] = (float)(r.x + r.y + r.z + r.w); }
}

// ---- High-ILP scalar FMA rate probes (v2) ---------------------------
// The v1 dg_fma_rate probes (4 chains) landed ~30% below the simdgroup-MMA
// rate -- latency/ILP-bound, not pipe-bound -- so they can't distinguish
// pipe widths. v2: 8 independent vec4 chains, inner loop unrolled 4x
// (256 flops per thread-iter), one variant per element type plus a MIXED
// f32+f16 variant (family-9 GPUs advertise parallel F32/F16 pipes; if they
// dual-issue, mixed exceeds the single-type rate). bf16 scalar math is
// documented-emulated (widen to f32 per op) -- the probe quantifies it.
// FLOPs = threads * iters * 8 chains * 4 lanes * 2 * 4 unroll.
#define DG_FMA3(a, c) fma(a, c, a)
#define DG_MAD2(a, c) (a * c + a)   // bfloat: no fma() overload
#define DG_RATE2_BODY(VT, ST, INIT_A, MADD)                                    \
  VT a = INIT_A;                                                         \
  VT c0 = a * ST(0.5f), c1 = a * ST(0.6f), c2 = a * ST(0.7f),            \
     c3 = a * ST(0.8f), c4 = a * ST(0.9f), c5 = a * ST(1.1f),            \
     c6 = a * ST(1.2f), c7 = a * ST(1.3f);                               \
  for (int i = 0; i < iters; ++i) {                                      \
    _Pragma("clang loop unroll(full)")                                   \
    for (int u = 0; u < 4; ++u) {                                        \
      c0 = MADD(a, c0); c1 = MADD(a, c1);                                \
      c2 = MADD(a, c2); c3 = MADD(a, c3);                                \
      c4 = MADD(a, c4); c5 = MADD(a, c5);                                \
      c6 = MADD(a, c6); c7 = MADD(a, c7);                                \
    }                                                                    \
  }                                                                      \
  const VT r = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));        \
  if (gid == 0) { out[0] = (float)r.x + (float)r.y + (float)r.z          \
                           + (float)r.w; }

kernel void dg_fma_rate2_f32(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  DG_RATE2_BODY(float4, float, float4(0.999f + 1e-7f * (float)(gid & 31)), DG_FMA3)
}

kernel void dg_fma_rate2_f16(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  DG_RATE2_BODY(half4, half, half4((half)(0.99f + 1e-4f * (float)(gid & 31))), DG_FMA3)
}

kernel void dg_fma_rate2_bf16(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  typedef vec<bfloat, 4> bf4;
  DG_RATE2_BODY(bf4, bfloat, bf4((bfloat)(0.99f + 1e-3f * (float)(gid & 31))), DG_MAD2)
}

// Mixed pipes: 4 f32 chains + 4 f16 chains per unrolled step. If the F32
// and F16 pipes dual-issue, this exceeds both single-type rates (up to 2x).
// FLOPs = threads * iters * 8 chains * 4 lanes * 2 * 4 unroll (same count).
kernel void dg_fma_rate2_mixed(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  float4 a = float4(0.999f + 1e-7f * (float)(gid & 31));
  half4 ah = half4((half)(0.99f + 1e-4f * (float)(gid & 31)));
  float4 c0 = a * 0.5f, c1 = a * 0.6f, c2 = a * 0.7f, c3 = a * 0.8f;
  half4 h0 = ah * (half)0.5f, h1 = ah * (half)0.6f, h2 = ah * (half)0.7f,
        h3 = ah * (half)0.8f;
  for (int i = 0; i < iters; ++i) {
#pragma clang loop unroll(full)
    for (int u = 0; u < 4; ++u) {
      c0 = fma(a, c0, a); h0 = fma(ah, h0, ah);
      c1 = fma(a, c1, a); h1 = fma(ah, h1, ah);
      c2 = fma(a, c2, a); h2 = fma(ah, h2, ah);
      c3 = fma(a, c3, a); h3 = fma(ah, h3, ah);
    }
  }
  const float4 r = (c0 + c1) + (c2 + c3);
  const half4 rh = (h0 + h1) + (h2 + h3);
  if (gid == 0) {
    out[0] = r.x + r.y + r.z + r.w
             + (float)rh.x + (float)rh.y + (float)rh.z + (float)rh.w;
  }
}

// ---- Integer multiply-add rate probes ------------------------------
// MSL exposes no int8 dot-product or integer matrix ops on Apple GPUs, so
// the int8 ALU ceiling is the scalar integer mad rate on packed char4
// (values wrap mod 256 -- harmless for a rate probe). Same 8-chain / 4x
// unroll shape as the float v2 probes; OPS = threads * iters * 8 chains *
// 4 lanes * 2 * 4 unroll (a mul-add counted as 2 ops, like an FMA).
kernel void dg_imad_rate_i8(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  DG_RATE2_BODY(char4, char, char4((char)(1 + (gid & 31))), DG_MAD2)
}

kernel void dg_imad_rate_i16(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  DG_RATE2_BODY(short4, short, short4((short)(1 + (gid & 31))), DG_MAD2)
}

kernel void dg_imad_rate_i32(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  DG_RATE2_BODY(int4, int, int4((int)(1 + (gid & 31))), DG_MAD2)
}

// int8 + f16 co-issue: 4 char4 mad chains + 4 half4 FMA chains -- the
// instruction mix a quantized kernel resembles (integer dequant math
// alongside float accumulation). Family-9 cores run integer and float on
// parallel pipes; this measures the combined ceiling. OPS = threads *
// iters * 8 chains * 4 lanes * 2 * 4 unroll.
kernel void dg_imad_rate_mix_i8f16(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  char4 a = char4((char)(1 + (gid & 31)));
  half4 ah = half4((half)(0.99f + 1e-4f * (float)(gid & 31)));
  char4 c0 = a, c1 = a + (char)1, c2 = a + (char)2, c3 = a + (char)3;
  half4 h0 = ah * (half)0.5f, h1 = ah * (half)0.6f, h2 = ah * (half)0.7f,
        h3 = ah * (half)0.8f;
  for (int i = 0; i < iters; ++i) {
#pragma clang loop unroll(full)
    for (int u = 0; u < 4; ++u) {
      c0 = a * c0 + a; h0 = fma(ah, h0, ah);
      c1 = a * c1 + a; h1 = fma(ah, h1, ah);
      c2 = a * c2 + a; h2 = fma(ah, h2, ah);
      c3 = a * c3 + a; h3 = fma(ah, h3, ah);
    }
  }
  const char4 r = ((c0 + c1) + (c2 + c3));
  const half4 rh = ((h0 + h1) + (h2 + h3));
  if (gid == 0) {
    out[0] = (float)r.x + (float)r.y + (float)r.z + (float)r.w
             + (float)rh.x + (float)rh.y + (float)rh.z + (float)rh.w;
  }
}

// Deeper mixed probe: 8 f32 + 8 f16 chains (the 4+4 mix reached 10.8 vs
// 7.9 pure -- is the mixed ceiling ILP-limited?). FLOPs = threads * iters *
// 16 chains * 4 lanes * 2 * 2 unroll.
kernel void dg_fma_rate2_mixed88(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  float4 a = float4(0.999f + 1e-7f * (float)(gid & 31));
  half4 ah = half4((half)(0.99f + 1e-4f * (float)(gid & 31)));
  float4 c0 = a * 0.5f, c1 = a * 0.6f, c2 = a * 0.7f, c3 = a * 0.8f,
         c4 = a * 0.9f, c5 = a * 1.1f, c6 = a * 1.2f, c7 = a * 1.3f;
  half4 h0 = ah * (half)0.5f, h1 = ah * (half)0.6f, h2 = ah * (half)0.7f,
        h3 = ah * (half)0.8f, h4 = ah * (half)0.9f, h5 = ah * (half)1.1f,
        h6 = ah * (half)1.2f, h7 = ah * (half)1.3f;
  for (int i = 0; i < iters; ++i) {
#pragma clang loop unroll(full)
    for (int u = 0; u < 2; ++u) {
      c0 = fma(a, c0, a); h0 = fma(ah, h0, ah);
      c1 = fma(a, c1, a); h1 = fma(ah, h1, ah);
      c2 = fma(a, c2, a); h2 = fma(ah, h2, ah);
      c3 = fma(a, c3, a); h3 = fma(ah, h3, ah);
      c4 = fma(a, c4, a); h4 = fma(ah, h4, ah);
      c5 = fma(a, c5, a); h5 = fma(ah, h5, ah);
      c6 = fma(a, c6, a); h6 = fma(ah, h6, ah);
      c7 = fma(a, c7, a); h7 = fma(ah, h7, ah);
    }
  }
  const float4 r = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));
  const half4 rh = ((h0 + h1) + (h2 + h3)) + ((h4 + h5) + (h6 + h7));
  if (gid == 0) {
    out[0] = r.x + r.y + r.z + r.w
             + (float)rh.x + (float)rh.y + (float)rh.z + (float)rh.w;
  }
}

// MMA + scalar-f16 co-issue probe: the f32 simdgroup-MMA loop of
// dg_mma_rate_acc32 with 4 scalar half4 FMA chains interleaved. If the F16
// pipe idles during matrix work, the combined rate exceeds the pure-MMA
// 10.1 -- the ceiling a hybrid (matrix f32 + scalar f16) GEMM could chase.
// FLOPs per simdgroup per iter = 4*1024 (mma) + 32 threads * 4 chains *
// 4 lanes * 2 (scalar f16).
kernel void dg_mma_scalar_mix(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid      [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(
      0.001f * (float)(simd_lid + 1));
  simdgroup_float8x8 b = make_filled_simdgroup_matrix<float, 8, 8>(1.0009f);
  simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 c1 = c0, c2 = c0, c3 = c0;
  half4 ah = half4((half)(0.99f + 1e-4f * (float)(simd_lid + 1)));
  half4 h0 = ah * (half)0.5f, h1 = ah * (half)0.6f, h2 = ah * (half)0.7f,
        h3 = ah * (half)0.8f;
  for (int i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0);
    h0 = fma(ah, h0, ah);
    simdgroup_multiply_accumulate(c1, a, b, c1);
    h1 = fma(ah, h1, ah);
    simdgroup_multiply_accumulate(c2, a, b, c2);
    h2 = fma(ah, h2, ah);
    simdgroup_multiply_accumulate(c3, a, b, c3);
    h3 = fma(ah, h3, ah);
  }
  simdgroup_multiply_accumulate(c0, c1, c2, c3);   // fold: defeat DCE
  threadgroup float sink[64];
  simdgroup_store(c0, sink, 8);
  const half4 rh = (h0 + h1) + (h2 + h3);
  if (gid == 0) {
    out[0] = sink[simd_lid & 63]
             + (float)rh.x + (float)rh.y + (float)rh.z + (float)rh.w;
  }
}

// bf16 simdgroup-MMA rate: simdgroup_matrix<bfloat> (family 9 / MSL 3.1+).
// Same shape as dg_mma_rate_acc16.
kernel void dg_mma_rate_accbf(
    device float*       out   [[buffer(0)]],
    const constant int& iters [[buffer(1)]],
    uint gid      [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  simdgroup_matrix<bfloat, 8, 8> a =
      make_filled_simdgroup_matrix<bfloat, 8, 8>(
          (bfloat)(0.001f * (float)(simd_lid + 1)));
  simdgroup_matrix<bfloat, 8, 8> b =
      make_filled_simdgroup_matrix<bfloat, 8, 8>((bfloat)1.0009f);
  simdgroup_matrix<bfloat, 8, 8> c0 =
      make_filled_simdgroup_matrix<bfloat, 8, 8>((bfloat)0.0f);
  simdgroup_matrix<bfloat, 8, 8> c1 = c0, c2 = c0, c3 = c0;
  for (int i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0);
    simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2);
    simdgroup_multiply_accumulate(c3, a, b, c3);
  }
  simdgroup_multiply_accumulate(c0, c1, c2, c3);   // fold: defeat DCE
  threadgroup bfloat sink[64];
  simdgroup_store(c0, sink, 8);
  if (gid == 0) { out[0] = (float)sink[simd_lid & 63]; }
}

// Shared-expert sigmoid gate (dense f16): g[t] = sigmoid(w[1,K] . x[t]).
// One simdgroup per input row (tid.y = row). grid {32, M, 1}, tg {32, 1, 1}.
kernel void dense_moe_gate_f16(
    const device VPIPE_ELT* w        [[buffer(0)]],   // [1, K]
    const device VPIPE_ELT* x        [[buffer(1)]],   // [M, K]
    device VPIPE_ELT*       outg      [[buffer(2)]],   // [M]
    const constant int&     K         [[buffer(3)]],
    uint3 tid     [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  const device VPIPE_ELT* xr = x + (int64_t)tid.y * (int64_t)K;
  float acc = 0.0f;
  for (int k = (int)simd_lid; k < K; k += DG_SIMD_SIZE) {
    acc += (float)w[k] * (float)xr[k];
  }
  acc = simd_sum(acc);
  if (simd_lid == 0) {
    outg[tid.y] = (VPIPE_ELT)(1.0f / (1.0f + metal::exp(-acc)));
  }
}
