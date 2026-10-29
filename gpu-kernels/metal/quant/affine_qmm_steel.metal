// affine_qmm_steel.metal -- 4-bit affine quantized GEMM built on
// MLX's vendored steel BlockMMA + BlockLoader machinery, to reach
// MLX-parity prefill throughput. Our hand-rolled affine_qmm plateaued
// at ~1.17x MLX because the remaining gap is steel's hand-tuned
// register-tiled MMA / load scheduling at an identical tile shape;
// this kernel uses that machinery directly.
//
// Vendored under msl/mlx/backend/metal/kernels/steel/ (resolved via
// the kernel build's -I msl): defines.h, utils.h, gemm/transforms.h,
// gemm/mma.h, gemm/loader.h, utils/{type_traits,integral_constant}.h.
// The quantized pieces (dequantize, QuantizedBlockLoader, qmm_t_impl)
// are copied verbatim from MLX's quantized.h so the math + packed
// layout match exactly.
//
//   T = half     bits = 4     group_size = 64     tile 32x32, WM=WN=2
//
// Non-batched only. Launch (dispatch_threadgroups-equivalent under
// metal-compute's dispatchThreads): threadgroup (32, WN, WM), grid
// threadgroups (ceil(N/BN), ceil(M/BM), 1). M/N tails handled inside
// qmm_t_impl, so y needs no padding.
//
//   0:w(uint32) 1:scales(VPIPE_ELT) 2:biases(VPIPE_ELT) 3:x(VPIPE_ELT) 4:y(VPIPE_ELT)
//   5:K 6:N 7:M

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#define MLX_MTL_CONST static constant constexpr const
#define SIMD_SIZE 32

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// steel mma.h has a complex64_t BlockMMA specialization (unused here
// but must parse), so complex64_t needs to be in scope first. MLX's
// complex.h references their bfloat16_t wrapper for an unused ctor;
// alias it to the native type (we never touch bf16 or complex).
typedef bfloat bfloat16_t;
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

// The dequantizing tile loaders (QuantizedBlockLoader, the MoE row
// gather) live in a header the matrix-core twins share.
#include "quant/affine_qmm_loaders.h"

// ===================================================================
// qmm_t_impl -- verbatim from MLX quantized.h.
// ===================================================================

// AccumT: the simdgroup-MMA element type. float (default) widens every
// A/B fragment and accumulates on the FP32 pipe (MLX-identical). half keeps
// the whole multiply-accumulate on the FP16 pipe (2x ALU rate on M3/M4-class
// GPUs) at the cost of f16 accumulation error over K -- callers must verify
// against their accuracy bar before routing to an _acc16 entry point.
// NOTE ON THE REMOVED `aligned_N` PARAMETER: this template used to take a
// compile-time "N is a multiple of BN" promise and, when given it, let the
// WEIGHT loader run load_unsafe() on the ragged last N-tile -- reading a full
// BN=32 rows regardless. Every entry point passed true, which was fine while
// every model's projection widths happened to be multiples of 32. Boogu-Image
// is not: its GQA k/v projections are N = 7*120 = 840 = 26.25 tiles, so the
// last tile read ~50 KB past the end of the codes + scales + biases (results
// were still correct -- the STORE is bound-checked -- so it was a latent fault,
// visible only if a ragged tensor landed at the end of a shard mapping). The
// promise is gone: the ragged tile is now decided at RUNTIME off num_outs, the
// same way dense_gemm.metal has always done it (it instantiates aligned_N=false
// everywhere) and the same way the store already did. The predicate is
// evaluated once per threadgroup to select a loop, NOT inside the K loop, so an
// aligned N pays a single comparison -- measured no regression. That alone
// did not stop the over-read: load_safe() then tested the row against the
// tile's column count and zeroed nothing, so every row was still read until
// its check was corrected too (see QuantizedBlockLoader::load_safe).
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    typename AccumT = float,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_t_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& K_eff,
    const device int* tile2e,   // MoE grouped: per-tile expert (or nullptr)
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded, AccumT>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  // MoE grouped GEMM: each BM-row M-tile is one expert's (sorted, padded)
  // rows -> offset the weight slab to that expert; skip empty padding tiles.
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
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
    if (num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
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

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

// ===================================================================
// Kernel entry: non-batched, half / 4-bit / group 64, 32x32 tile.
// ===================================================================

kernel void affine_qmm_steel_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// A BM=16 arm (WM=2, WN=2 -> TM=1) was built and measured on M4 Pro and is
// NOT here: it was the slowest tile at every (M, K) tried on both FF GEMMs --
// 15-22% behind BM=64 on the down projection, 20-30% behind on the fused
// gate|up -- with no crossover at either end of the M or K ladder. Smaller
// than 32 does not pay on this GPU; do not re-add it without a machine whose
// occupancy story differs.
//
// BM=64 tile twin of affine_qmm_steel_w4g64 for tall-M GEMMs (M ~= seq,
// e.g. the Krea-2 DiT blocks). Same 128 threads compute a 64-row output
// tile, so each dequantized [BN,BK] weight tile serves twice the output
// rows -- halving both the weight-code re-reads (M/BM passes over the
// codes) and the per-FLOP dequant ALU work vs the 32x32 tile. Mirrors
// the dense dense_gemm_t_bm64_f16 tuning. Grid: x=ceil(N/32)*32,
// y=ceil(M/64)*2, z=2 (threadgroup {32,2,2}).
kernel void affine_qmm_steel_w4g64_bm64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// BM=128 tile (WM=4, WN=2 -> 256 threads, 8 simdgroups; each simdgroup owns a
// 32x16 output tile = 8 accumulator frags, no register spill). Halves the
// weight-code re-reads + dequant ALU again vs the BM=64 tile (M/128 vs M/64
// passes over the codes). MEASURED ~7-8% SLOWER than BM=64 on M4 Pro at
// M=4106/1024px (see the host _qmm_tile note): these GEMMs are compute/
// occupancy-bound, not weight-bandwidth-bound, so fewer weight re-reads
// doesn't pay while the bigger 128-row/256-thread tile halves the threadgroup
// count and cuts latency-hiding. Re-measured on the H3 shapes: still behind
// BM=64 everywhere on M4 (down projection 6.1-6.4 vs 6.5-6.9 TF/s), and the
// gap NARROWS as M grows -- 0.97x of BM=64 at M=602, 0.96x at 9382, but only
// 1.5% behind at 19008 on the fused gate|up.
//
// KEPT, and not as a formality. Halving the threadgroup count is a loss on a
// GPU whose cores this already saturates and a win on one with more of them
// to fill, so the tile that loses here is the one that plausibly wins on a
// larger part -- which is exactly the case a laptop measurement cannot
// settle. Opt-in (host gates on M >= 1024, tile 2), so nothing takes it by
// accident. Grid: x=ceil(N/32)*32, y=ceil(M/128)*2, z=4 (tg {32,2,4}).
kernel void affine_qmm_steel_w4g64_bm128(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 128, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 4, BM, BK, BN, float,
             /*WM=*/4, /*WN=*/2>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// FP16-pipe (AccumT=half) twin of affine_qmm_steel_w4g64_bm64: the whole
// simdgroup MMA runs on half8x8 fragments instead of widening to float.
// Only pays on a GPU whose f16 matrix rate exceeds f32. MEASURED (M4 Pro
// 20c, dg_mma_rate probes): f16 10.4 vs f32 10.1 TFLOP/s (+3%) and mixed
// f16+f32 SLOWER (7.6) -- no double-rate/dual-issue f16 pipe, so this is
// perf-neutral there and stays opt-in (VPIPE_KREA2_ACC16; costs accuracy:
// f16 accumulation drift over K). Re-probe before enabling on new GPUs.
kernel void affine_qmm_steel_w4g64_bm64_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 4, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

kernel void affine_qmm_steel_w8g64_bm64_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 8, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmm_steel_w4g64 (GGUF q4_0). Only the
// template group_size differs.
kernel void affine_qmm_steel_w4g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 32, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// ===================================================================
// Fused SwiGLU MLP GEMM: y = silu(gate) * up in one pass.
//
// The gate and up projection weights are pre-INTERLEAVED by output
// feature into a single [N=2*ffn, K] quantized matrix: fused row 2g is
// gate feature g, row 2g+1 is up feature g. So the GEMM result column
// 2g is x.gate_g and column 2g+1 is x.up_g. With BN even and the 8x8
// simdgroup-matrix frag layout, those two adjacent columns are exactly
// the two accumulator elements THIS thread already holds per frag (the
// frag's column base fn is always even -> elem 0 = gate, elem 1 = up),
// so the activation+multiply is purely register-local -- no cross-lane
// shuffle, and the written output is HALVED to [M, ffn] (no separate
// gate/up buffers, no separate swiglu pass).
//
//   0:w(uint32) 1:scales 2:biases 3:x 4:y([M, N/2]) 5:K 6:N(=2*ffn) 7:M
//
// qmm_t_swiglu_rs_impl is the GENERIC (strided-output) SwiGLU GEMM: it writes
// silu(gate)*up into y[row*out_rs + out_off + g], where out_rs = out_stride or
// (out_stride<=0) the contiguous N/2. The non-strided qmm_t_swiglu_impl below is
// a thin wrapper (out_stride=out_off=0) so its callers (Krea-2 etc.) are
// unchanged. Mirrors qmm_t_impl's MMA loop; N even, matmul N tail aligned
// (2*ffn % BN == 0), only the M tail guarded.
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    typename AccumT = float,
    const int WM = 2,
    const int WN = 2>
// N-ALIGNMENT: unlike qmm_t_impl above, this and the other fused-epilogue
// variants below (swiglu / geglu / grouped-MoE) bound-check NEITHER the weight
// load NOR the store against N -- their epilogues write a fixed pattern of
// simdgroup fragments. They therefore REQUIRE N % BN == 0, which every caller
// satisfies structurally (N = 2*ff_inner, and ff_inner is a multiple of
// `multiple_of` = 256 in every model here). A future caller with a ragged N
// must NOT be routed here: it would over-read the weight AND write past the row
// (the latter is why making these runtime-safe is more than a load fix).
METAL_FUNC void qmm_t_swiglu_rs_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int out_stride,
    const int out_off,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded, AccumT>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
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
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Fused SwiGLU store. Each accumulator frag holds, for this thread, a
  // (gate, up) pair in elems [0] (even column) and [1] (odd column);
  // emit silu(gate)*up to y[row, col/2]. Output width is N/2.
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);   // simdgroup tiles along M (2)
  constexpr int TN = BN / (FS * WN);   // simdgroup tiles along N (2)
  const int outN = out_stride > 0 ? out_stride : N / 2;   // output row stride
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

// Non-strided wrapper: the original qmm_t_swiglu_impl signature (contiguous
// output, N/2 wide). Existing entry points (Krea-2's fused FF, etc.) call this
// unchanged; it forwards to the generic with out_stride = out_off = 0.
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32,
    typename AccumT = float,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_t_swiglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  qmm_t_swiglu_rs_impl<T, group_size, bits, BM, BK, BN, AccumT, WM, WN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, 0, 0, tid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// BM=128 (WM=4, WN=2 -> 256 threads) twin of the fused SwiGLU GEMM. This is
// the biggest DiT GEMM (FF gate+up, N = 2*FF = 32768) and the ONLY quantized
// tile that had stayed at BM=32, so at high res (M = seq = 4106) it re-read
// the fused gate|up codes M/32 = 128x; BM=128 cuts that to M/128 = 33x (4x
// fewer weight re-reads + dequant). Host-gated on M >= 1024. Grid:
// x=ceil(2*FF/32)*32, y=ceil(M/128)*2, z=4 (threadgroup {32,2,4}).
kernel void affine_qmm_swiglu_w4g64_bm128(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 128, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN, float, /*WM=*/4, /*WN=*/2>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// FP16-pipe (AccumT=half) twin of affine_qmm_swiglu_w4g64 (see the qmm
// _acc16 comment); the fused silu(gate)*up epilogue math stays f32.
kernel void affine_qmm_swiglu_w4g64_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// BM64 fused-SwiGLU (Krea-2's tile choice for the FF): a 64-row tile between the
// BM32 base and BM128, float or FP16-pipe (acc16) accumulate.
kernel void affine_qmm_swiglu_w4g64_bm64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w4g64_bm64_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// Strided-output BM64 fused-SwiGLU (writes silu(gate)*up into y[:, off:] of a
// wider buffer, row stride out_stride) -- lets the FLUX.2 single block write the
// mlp straight into scat[:, H:] with no concat. buffer(8)=out_stride,
// buffer(9)=out_off.
kernel void affine_qmm_swiglu_w4g64_bm64_rs(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    const constant int&    out_stride [[buffer(8)]],
    const constant int&    out_off    [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_rs_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, out_stride, out_off,
      tid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w4g64_bm64_rs_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    const constant int&    out_stride [[buffer(8)]],
    const constant int&    out_off    [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_rs_impl<VPIPE_ELT, 64, 4, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, out_stride, out_off,
      tid, simd_gid, simd_lid);
}

// ===================================================================
// MoE GROUPED steel GEMM (Qwen3.5-MoE prefill -- the matrix-tiled expert
// GEMM that closes the prefill gap vs MLX gather_qmm). The (token,expert)
// pairs are counting-sorted into per-expert blocks padded to BM rows; each
// M-tile (tid.y) belongs to ONE expert (tile2e[tid.y]) and the weight slab is
// offset to it -- so the steel MMA runs per expert reading the weight once and
// at full matrix-core/SIMD efficiency. Token-exact with the per-pair/GEMV
// paths. The gate|up GEMM gathers its rows from the hidden via srow (fused into
// the load); the down GEMM scatters its rows to partials via sdst (fused into
// the store) -- no explicit gather/scatter copies.
// ===================================================================
// Forward decls (impls below): the down GEMM's scatter store + gate|up gather.
template <typename T, const int group_size, const int bits,
          const int BM, const int BK, const int BN>
METAL_FUNC void qmm_t_grouped_down_impl(
    const device uint32_t*, const device T*, const device T*, const device T*,
    device T*, threadgroup T*, threadgroup T*, const constant int&,
    const constant int&, const constant int&, const device int*,
    const device int*, uint3, uint, uint);

kernel void affine_qmm_grouped_w4g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // sorted acts [npad,K]
    device VPIPE_ELT*       y      [[buffer(4)]],   // partials [np,N] scattered
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* sdst         [[buffer(9)]],   // sorted-slot -> pair index
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_down_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, sdst,
      tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmm_grouped_w4g64 (prefill steel down GEMM). The impl
// (QuantizedBlockLoader + K_w = K*bytes_per_pack/pack_factor) is fully bit-
// parameterized, so bits=8 just instantiates the w8 slab stride + loader.
kernel void affine_qmm_grouped_w8g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // sorted acts [npad,K]
    device VPIPE_ELT*       y      [[buffer(4)]],   // partials [np,N] scattered
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* sdst         [[buffer(9)]],   // sorted-slot -> pair index
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_down_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, sdst,
      tid, simd_gid, simd_lid);
}

// 5-bit twin (MLX affine w5 g64): the impl's slab offset is in bytes and
// 64-bit, and QuantizedBlockLoader reads one 5-byte pack per thread at
// these tiles, so bits=5 instantiates it as it stands.
kernel void affine_qmm_grouped_w5g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // sorted acts [npad,K]
    device VPIPE_ELT*       y      [[buffer(4)]],   // partials [np,N] scattered
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* sdst         [[buffer(9)]],   // sorted-slot -> pair index
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_down_impl<VPIPE_ELT, 64, 5, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, sdst,
      tid, simd_gid, simd_lid);
}

// Grouped gate|up + SwiGLU GEMM (copy of qmm_t_swiglu_impl + the per-M-tile
// expert weight offset). Interleaved [E, 2*inner, K] weight; output halved to
// sorted [npad, inner] (silu(gate)*up).
template <typename T, const int group_size, const int bits,
          const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void qmm_t_grouped_swiglu_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* srow,
    uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  // x is GATHERED from the hidden via srow (the gather fused into the load).
  using loader_x_t =
      GatherBlockLoader<T, BM, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, srow, y_row, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = N / 2;
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];
      const float up = (float)fr[1];
      const float s = ga / (1.0f + metal::exp(-ga));
      y[(int64_t)row * outN + (col >> 1)] = (T)(s * up);
    }
  }
}

// GeGLU twin of qmm_t_grouped_swiglu_impl (Gemma-4 MoE prefill): identical
// gathered-x / per-tile-expert weight loading, only the store activation is
// gelu_pytorch_tanh instead of silu -- the same relationship
// qmm_t_geglu_impl has to qmm_t_swiglu_impl. Gemma's routed experts use
// GeGLU, so without this the MoE prefill had no grouped path at all and fell
// back to one gathered GEMV per (token,expert) pair, re-reading each expert
// slab once per routed token.
template <typename T, const int group_size, const int bits,
          const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void qmm_t_grouped_geglu_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* srow,
    uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      GatherBlockLoader<T, BM, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, srow, y_row, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = N / 2;
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];               // gate (even col)
      const float up = (float)fr[1];               // up   (odd col)
      const float k0 = 0.7978845608028654f;
      const float t =
          metal::precise::tanh(k0 * (ga + 0.044715f * ga * ga * ga));
      y[(int64_t)row * outN + (col >> 1)] = (T)(0.5f * ga * (1.0f + t) * up);
    }
  }
}

kernel void affine_qmm_grouped_geglu_w4g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // hidden [M, K] (gathered)
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],   // fused width = 2*inner
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],   // sorted-slot -> hidden row
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_geglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

kernel void affine_qmm_grouped_geglu_w8g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_geglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

kernel void affine_qmm_grouped_swiglu_w4g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // hidden [M, K] (gathered)
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],   // fused width = 2*inner
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],   // sorted-slot -> hidden row
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_swiglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

// 8-bit twin of affine_qmm_grouped_swiglu_w4g64 (prefill steel gate|up GEMM).
// Fully bit-parameterized impl; bits=8 instantiates the w8 slab stride +
// QuantizedBlockLoader.
kernel void affine_qmm_grouped_swiglu_w8g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // hidden [M, K] (gathered)
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],   // fused width = 2*inner
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],   // sorted-slot -> hidden row
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

// 5-bit twin of affine_qmm_grouped_swiglu_w8g64 (see affine_qmm_grouped_w5g64).
kernel void affine_qmm_grouped_swiglu_w5g64(
    const device uint32_t* w       [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],   // hidden [M, K] (gathered)
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K          [[buffer(5)]],
    const constant int& N          [[buffer(6)]],   // fused width = 2*inner
    const constant int& M          [[buffer(7)]],
    const device int* tile2e       [[buffer(8)]],
    const device int* srow         [[buffer(9)]],   // sorted-slot -> hidden row
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_grouped_swiglu_impl<VPIPE_ELT, 64, 5, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,
      tid, simd_gid, simd_lid);
}

// Grouped down GEMM with a SCATTER store: each sorted output row is written to
// partials[sdst[slot], :] (fuses moe_scatter_back into the store). x = sorted
// activations (contiguous); w = per-tile expert down slab. Mirrors qmm_t_impl's
// MMA loop; only the store differs (manual per-row scatter, no store_result).
template <typename T, const int group_size, const int bits,
          const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void qmm_t_grouped_down_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* sdst,
    uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  if (tile2e) {
    const int e = tile2e[tid.y];
    if (e < 0) { return; }
    wl += static_cast<int64_t>(e) * N * K_w;
    scales += static_cast<int64_t>(e) * N * K_g;
    biases += static_cast<int64_t>(e) * N * K_g;
  }
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    const int dr = sdst[row];              // pair-ordered destination (or -1)
    if (dr < 0) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      y[(int64_t)dr * N + col] = (T)fr[0];
      y[(int64_t)dr * N + col + 1] = (T)fr[1];
    }
  }
}

// Group-32 and 6-bit twins of the grouped expert GEMMs (Qwen3.8-Flash-Next's
// MLX 4-bit pack holds its experts at group 32; the oQ6e pack at 6 bits).
// The impls take the group and the width as template arguments; BK 32 fits
// a group of 32 (the loader's BCOLS <= group_size).
#define VPIPE_QMM_GROUPED_ANY(NAME_SW, NAME_DN, BITS, GS)                     \
  kernel void NAME_SW(                                                        \
      const device uint32_t* w       [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                           \
      const device VPIPE_ELT* biases [[buffer(2)]],                           \
      const device VPIPE_ELT* x      [[buffer(3)]],                           \
      device VPIPE_ELT*       y      [[buffer(4)]],                           \
      const constant int& K          [[buffer(5)]],                           \
      const constant int& N          [[buffer(6)]],                           \
      const constant int& M          [[buffer(7)]],                           \
      const device int* tile2e       [[buffer(8)]],                           \
      const device int* srow         [[buffer(9)]],                           \
      uint3 tid      [[threadgroup_position_in_grid]],                        \
      uint  simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint  simd_lid [[thread_index_in_simdgroup]]) {                         \
    constexpr int BM = 32, BK = 32, BN = 32;                                  \
    constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));                  \
    threadgroup VPIPE_ELT Xs[BM * BK_padded];                                 \
    threadgroup VPIPE_ELT Ws[BN * BK_padded];                                 \
    qmm_t_grouped_swiglu_impl<VPIPE_ELT, GS, BITS, BM, BK, BN>(               \
        w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, srow,               \
        tid, simd_gid, simd_lid);                                             \
  }                                                                           \
  kernel void NAME_DN(                                                        \
      const device uint32_t* w       [[buffer(0)]],                           \
      const device VPIPE_ELT* scales [[buffer(1)]],                           \
      const device VPIPE_ELT* biases [[buffer(2)]],                           \
      const device VPIPE_ELT* x      [[buffer(3)]],                           \
      device VPIPE_ELT*       y      [[buffer(4)]],                           \
      const constant int& K          [[buffer(5)]],                           \
      const constant int& N          [[buffer(6)]],                           \
      const constant int& M          [[buffer(7)]],                           \
      const device int* tile2e       [[buffer(8)]],                           \
      const device int* sdst         [[buffer(9)]],                           \
      uint3 tid      [[threadgroup_position_in_grid]],                        \
      uint  simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint  simd_lid [[thread_index_in_simdgroup]]) {                         \
    constexpr int BM = 32, BK = 32, BN = 32;                                  \
    constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));                  \
    threadgroup VPIPE_ELT Xs[BM * BK_padded];                                 \
    threadgroup VPIPE_ELT Ws[BN * BK_padded];                                 \
    qmm_t_grouped_down_impl<VPIPE_ELT, GS, BITS, BM, BK, BN>(                 \
        w, scales, biases, x, y, Xs, Ws, K, N, M, tile2e, sdst,               \
        tid, simd_gid, simd_lid);                                             \
  }
VPIPE_QMM_GROUPED_ANY(affine_qmm_grouped_swiglu_w4g32,
                      affine_qmm_grouped_w4g32, 4, 32)
VPIPE_QMM_GROUPED_ANY(affine_qmm_grouped_swiglu_w5g32,
                      affine_qmm_grouped_w5g32, 5, 32)
VPIPE_QMM_GROUPED_ANY(affine_qmm_grouped_swiglu_w6g32,
                      affine_qmm_grouped_w6g32, 6, 32)
VPIPE_QMM_GROUPED_ANY(affine_qmm_grouped_swiglu_w8g32,
                      affine_qmm_grouped_w8g32, 8, 32)
VPIPE_QMM_GROUPED_ANY(affine_qmm_grouped_swiglu_w6g64,
                      affine_qmm_grouped_w6g64, 6, 64)

// Fused GeGLU MLP GEMM (Gemma-4 prefill): mirrors qmm_t_swiglu_impl
// verbatim; only the activation in the store is gelu_pytorch_tanh (matches
// geglu_f16) instead of silu. Interleaved [N=2*ffn, K] weight, output y
// halved to [M, ffn].  0:w 1:scales 2:biases 3:x 4:y 5:K 6:N(=2*ffn) 7:M
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void qmm_t_geglu_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;

  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(BK, num_els));
      loader_w.load_unsafe();
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
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Fused GeGLU store: frag elem[0]=gate (even col), elem[1]=up (odd col);
  // emit gelu_tanh(gate)*up to y[row, col/2]. Output width N/2.
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = N / 2;
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
      const float k0 = 0.7978845608028654f;
      const float t = metal::precise::tanh(k0 * (ga + 0.044715f * ga * ga * ga));
      y[(int64_t)row * outN + (col >> 1)] = (T)(0.5f * ga * (1.0f + t) * up);
    }
  }
}

kernel void affine_qmm_geglu_w4g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_geglu_impl<VPIPE_ELT, 64, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmm_geglu_w4g64 (GGUF q4_0). Only the
// template group_size differs.
kernel void affine_qmm_geglu_w4g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_geglu_impl<VPIPE_ELT, 32, 4, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// ===================================================================
// 8-bit (group 64) steel GEMM entry points. The MMA loop, dequantize,
// and QuantizedBlockLoader are all `bits`-templated (dequantize handles
// bits==8), so these mirror the 4-bit kernels with bits=8. One byte per
// weight value (pack_factor=bytes_per_pack=1 for bits=8).
// ===================================================================
kernel void affine_qmm_steel_w8g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// BM=64 tile twin of affine_qmm_steel_w8g64 (see the w4 bm64 comment).
kernel void affine_qmm_steel_w8g64_bm64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// BM=128 (WM=4, WN=2) twin of affine_qmm_steel_w8g64_bm64 -- see the w4 bm128
// note. Host-gated on M >= 1024.
kernel void affine_qmm_steel_w8g64_bm128(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 128, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 64, 8, BM, BK, BN, float,
             /*WM=*/4, /*WN=*/2>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

// group_size=32 twin of affine_qmm_steel_w8g64 (the MOSS codec int8 path):
// only the template group_size differs. Any N: the ragged last tile is
// bound-checked at runtime (see the note on qmm_t_impl).
kernel void affine_qmm_steel_w8g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 32, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w8g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// Group-32 twin (a w8g32 shared expert).
kernel void affine_qmm_swiglu_w8g32(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 32, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// BM=128 (WM=4, WN=2) twin of affine_qmm_swiglu_w8g64 -- see the w4 bm128
// note. Host-gated on M >= 1024.
kernel void affine_qmm_swiglu_w8g64_bm128(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 128, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN, float, /*WM=*/4, /*WN=*/2>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// FP16-pipe (AccumT=half) twin of affine_qmm_swiglu_w8g64.
kernel void affine_qmm_swiglu_w8g64_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// BM64 fused-SwiGLU, 8-bit (Krea-2's FF tile choice); float or acc16 accumulate.
kernel void affine_qmm_swiglu_w8g64_bm64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w8g64_bm64_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// Strided-output BM64 fused-SwiGLU, 8-bit (FLUX.2 single-block mlp -> scat[:,H:]).
kernel void affine_qmm_swiglu_w8g64_bm64_rs(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    const constant int&    out_stride [[buffer(8)]],
    const constant int&    out_off    [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_rs_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, out_stride, out_off,
      tid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w8g64_bm64_rs_acc16(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],
    const constant int&    M      [[buffer(7)]],
    const constant int&    out_stride [[buffer(8)]],
    const constant int&    out_off    [[buffer(9)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_rs_impl<VPIPE_ELT, 64, 8, BM, BK, BN, VPIPE_ACC16_T>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, out_stride, out_off,
      tid, simd_gid, simd_lid);
}

// Fused GeGLU MLP steel GEMM, 8-bit weights (Gemma-4 12B gate/up). Mirrors
// affine_qmm_geglu_w4g64 with bits=8.
kernel void affine_qmm_geglu_w8g64(
    const device uint32_t* w      [[buffer(0)]],
    const device VPIPE_ELT*     scales [[buffer(1)]],
    const device VPIPE_ELT*     biases [[buffer(2)]],
    const device VPIPE_ELT*     x      [[buffer(3)]],
    device VPIPE_ELT*           y      [[buffer(4)]],
    const constant int&    K      [[buffer(5)]],
    const constant int&    N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&    M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_geglu_impl<VPIPE_ELT, 64, 8, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}

// ===================================================================
// BLOCK-FP8 grouped MoE GEMMs (Fish Audio Drama 3's routed experts: E4M3
// codes, one f32 scale per 128 x 128 block). The affine grouped GEMMs above
// with the weight loader swapped: each expert's weight is read ONCE per
// tile, where the pair-batched gather (dense_moe_gather_*_fp8) reads it once
// per routed row -- the prefill's whole cost. Codes are widened to half by
// bit shift ((b & 0x80) << 8 | (b & 0x7F) << 7 is the E4M3 value / 2^8, NaN
// codes excepted, which no checkpoint holds); the 2^8 is folded into the
// scales at load, as for the gather kernels.
// ===================================================================

inline half fp8_e4m3_half_div256(uchar b) {
  return as_type<half>((ushort)(((ushort)(b & 0x80) << 8) |
                                ((ushort)(b & 0x7F) << 7)));
}

// The weight tile: BROWS output rows x BCOLS of the reduction, each thread
// n_reads consecutive codes of one row scaled by that row's block. Rows of
// an INTERLEAVED slab are gate|up pairs (row 2g gate g, 2g+1 up g) whose
// scales are stored [2][N/2/128][K/128]; a plain slab's [N/128][K/128].
// BCOLS divides 128, so a tile's columns share one scale block.
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size,
          bool Interleaved>
struct Fp8BlockLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads == 8, "a lane reads 8 codes (one 8-byte load)");
  static_assert(128 % BCOLS == 0, "a tile stays inside one scale block");
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device uint8_t* src;
  const device float* srow;
  int k_off;
  METAL_FUNC Fp8BlockLoader(
      const device uint8_t* codes, const device float* scales, const int K,
      const int N, const int y_col, threadgroup T* dst_,
      ushort simd_group_id, ushort simd_lane_id)
      : bi((short)((simd_group_id * 32 + simd_lane_id) / TCOLS)),
        bj((short)(n_reads * ((simd_group_id * 32 + simd_lane_id) % TCOLS))),
        dst(dst_ + bi * dst_ld + bj),
        src(codes + (int64_t)(y_col + bi) * K + bj),
        srow(scales + (int64_t)(Interleaved
                                    ? ((y_col + bi) & 1) * (N / 256) +
                                          ((y_col + bi) >> 1) / 128
                                    : (y_col + bi) / 128) *
                          (K / 128)),
        k_off(0) {}
  METAL_FUNC void load_unsafe() const {
    const float s = srow[(k_off + bj) / 128];
    const uint2 v = *(const device uint2*)(src + k_off);
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 8; ++j) {
      const uint word = j < 4 ? v.x : v.y;
      const uchar b = (uchar)((word >> (8 * (j & 3))) & 0xFF);
      dst[j] = (T)((float)fp8_e4m3_half_div256(b) * s);
    }
  }
  METAL_FUNC void load_safe(short2) const { load_unsafe(); }
  METAL_FUNC void next() { k_off += BCOLS; }
};

// The activation tile in F32 from S storage (bf16 / f16 widen exactly):
// BROWS rows of BCOLS, a row either gathered (srow[base + r], < 0 a zero
// pad row) or contiguous (srow null). The FP8 GEMMs multiply in F32 so the
// dequantized weight is never rounded to bf16 -- the gather kernels' (and
// the reference's) arithmetic, which the bf16 tile measurably was not.
template <typename S, typename T, short BROWS, short BCOLS, short dst_ld,
          short tgp_size>
struct RowConvLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;
  const int src_ld;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device S* src;
  const device int* srow;
  const int base;
  int k_off;
  METAL_FUNC RowConvLoader(
      const device S* src_, const int src_ld_, const device int* srow_,
      const int base_, threadgroup T* dst_, ushort simd_group_id,
      ushort simd_lane_id)
      : src_ld(src_ld_),
        bi((short)((simd_group_id * 32 + simd_lane_id) / TCOLS)),
        bj((short)(n_reads * ((simd_group_id * 32 + simd_lane_id) % TCOLS))),
        dst(dst_ + bi * dst_ld + bj), src(src_), srow(srow_), base(base_),
        k_off(0) {}
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      const int r = srow != nullptr ? srow[base + bi + i] : base + bi + i;
      if (r >= 0) {
        const device S* p = src + (int64_t)r * src_ld + bj + k_off;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_reads; ++j) {
          dst[i * dst_ld + j] = (T)p[j];
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_reads; ++j) { dst[i * dst_ld + j] = T(0); }
      }
    }
  }
  METAL_FUNC void next() { k_off += BCOLS; }
};

// Grouped gate|up + SwiGLU over block-FP8 codes [E, N = 2*inner, K] (rows
// interleaved) -> sorted act [npad, inner]; x gathered from the hidden via
// srow (qmm_t_grouped_swiglu_impl with the FP8 loader, tiles in F32).
template <typename S, const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void fp8_grouped_swiglu_impl(
    const device uint8_t* w, const device float* scales, const device S* x,
    device S* y, threadgroup float* Xs, threadgroup float* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* srow, uint3 tid,
    uint simd_gid, uint simd_lid) {
  using T = float;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      RowConvLoader<S, T, BM, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      Fp8BlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, true>;
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const device uint8_t* we = w + (int64_t)e * N * K;
  const device float* se = scales + (int64_t)e * (N / 128) * (K / 128);
  loader_x_t loader_x(x, K, srow, y_row, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(we, se, K, N, y_col, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const int outN = N / 2;
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      const float ga = (float)fr[0];
      const float up = (float)fr[1];
      const float s = ga / (1.0f + metal::exp(-ga));
      y[(int64_t)row * outN + (col >> 1)] = (S)(s * up);
    }
  }
}

// Grouped down over block-FP8 codes [E, N = hidden, K = inner] with the
// scatter store (qmm_t_grouped_down_impl with the FP8 loader, tiles in F32):
// partials [np, N], row sdst[slot] (or nothing for a pad slot). Every
// sorted slot is a row of x (pad slots hold the gate|up GEMM's zeros).
template <typename S, const int BM = 32, const int BK = 32, const int BN = 32>
METAL_FUNC void fp8_grouped_down_impl(
    const device uint8_t* w, const device float* scales, const device S* x,
    device S* y, threadgroup float* Xs, threadgroup float* Ws,
    const constant int& K, const constant int& N, const constant int& M,
    const device int* tile2e, const device int* sdst, uint3 tid,
    uint simd_gid, uint simd_lid) {
  using T = float;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      RowConvLoader<S, T, BM, BK, BK_padded, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      Fp8BlockLoader<T, BN, BK, BK_padded, WM * WN * SIMD_SIZE, false>;
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const device uint8_t* we = w + (int64_t)e * N * K;
  const device float* se = scales + (int64_t)e * (N / 128) * (K / 128);
  loader_x_t loader_x(x, K, nullptr, y_row, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(we, se, K, N, y_col, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int FS = 8;
  constexpr int TM = BM / (FS * WM);
  constexpr int TN = BN / (FS * WN);
  const short sm = mma_op.sm;
  const short sn = mma_op.sn;
  for (short ti = 0; ti < TM; ti++) {
    const int row = y_row + sm + ti * FS * WM;
    if (row >= M) { continue; }
    const int dr = sdst[row];
    if (dr < 0) { continue; }
    for (short tj = 0; tj < TN; tj++) {
      const int col = y_col + sn + tj * FS * WN;
      const thread auto& fr = mma_op.Ctile.frag_at(ti, tj);
      y[(int64_t)dr * N + col] = (S)fr[0];
      y[(int64_t)dr * N + col + 1] = (S)fr[1];
    }
  }
}

//   0:codes(u8) [E,N,K] 1:scales(f32) 2:x 3:y 4:K 5:N 6:M(npad) 7:tile2e
//   8:srow (swiglu) / sdst (down).  threadgroup (32, 2, 2); grid
//   (ceil(N/32)*32, ceil(npad/32)*2, 2) threads.
kernel void fp8_qmm_grouped_swiglu_b128(
    const device uint8_t* w        [[buffer(0)]],
    const device float* scales     [[buffer(1)]],
    const device VPIPE_ELT* x      [[buffer(2)]],   // hidden [M_in, K]
    device VPIPE_ELT*       y      [[buffer(3)]],   // sorted act [npad, N/2]
    const constant int& K          [[buffer(4)]],
    const constant int& N          [[buffer(5)]],
    const constant int& M          [[buffer(6)]],
    const device int* tile2e       [[buffer(7)]],
    const device int* srow         [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(float));
  threadgroup float Xs[BM * BK_padded];
  threadgroup float Ws[BN * BK_padded];
  fp8_grouped_swiglu_impl<VPIPE_ELT, BM, BK, BN>(
      w, scales, x, y, Xs, Ws, K, N, M, tile2e, srow, tid, simd_gid,
      simd_lid);
}

kernel void fp8_qmm_grouped_down_b128(
    const device uint8_t* w        [[buffer(0)]],
    const device float* scales     [[buffer(1)]],
    const device VPIPE_ELT* x      [[buffer(2)]],   // sorted act [npad, K]
    device VPIPE_ELT*       y      [[buffer(3)]],   // partials [np, N]
    const constant int& K          [[buffer(4)]],
    const constant int& N          [[buffer(5)]],
    const constant int& M          [[buffer(6)]],
    const device int* tile2e       [[buffer(7)]],
    const device int* sdst         [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(float));
  threadgroup float Xs[BM * BK_padded];
  threadgroup float Ws[BN * BK_padded];
  fp8_grouped_down_impl<VPIPE_ELT, BM, BK, BN>(
      w, scales, x, y, Xs, Ws, K, N, M, tile2e, sdst, tid, simd_gid,
      simd_lid);
}

// ===================================================================
// 2-bit, group 128 (the ternary packs): the steel tile at bits 2 -- the
// quantized block loader carries MLX's 2-bit dequant -- and its fused
// SwiGLU twin. Same buffers and dispatch as the w4g64 32x32 entries.
// ===================================================================
kernel void affine_qmm_steel_w2g128(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int&     K      [[buffer(5)]],
    const constant int&     N      [[buffer(6)]],
    const constant int&     M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  lid      [[thread_index_in_threadgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_impl<VPIPE_ELT, 128, 2, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, K,
      /*tile2e=*/nullptr, tid, lid, simd_gid, simd_lid);
}

kernel void affine_qmm_swiglu_w2g128(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int&     K      [[buffer(5)]],
    const constant int&     N      [[buffer(6)]],   // fused width = 2*ffn
    const constant int&     M      [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(VPIPE_ELT));
  threadgroup VPIPE_ELT Xs[BM * BK_padded];
  threadgroup VPIPE_ELT Ws[BN * BK_padded];
  qmm_t_swiglu_impl<VPIPE_ELT, 128, 2, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M,
      tid, simd_gid, simd_lid);
}
