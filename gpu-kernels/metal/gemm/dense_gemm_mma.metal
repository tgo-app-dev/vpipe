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
// RELAX = the descriptor's relaxed_precision flag: true lets the matrix
// unit keep the accumulator at operand precision (f16/bf16) instead of
// widening to f32 -- the matmul2d analogue of the steel acc16 twins.
// Production entries pass false; the _rp twins exist to MEASURE it.
template <int BM, int BN, int SG, int TM, int TN, bool RELAX = false>
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
      /*relaxed_precision=*/RELAX);
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

// relaxed_precision=true twins of the production tiles, swept by
// gemm_mma.tune. MEASURED on M5 (bf16, all tune shapes, K=2560..16384):
// rate identical to the strict kernels (+-1%, noise) AND drift unchanged
// (~1e-3 even at K=16384 -- a true bf16 accumulator would be orders
// worse), i.e. the M5 matrix unit keeps its f32 accumulate pipeline
// regardless; the flag is a permission this implementation does not
// exercise. So there is NOTHING to win here today -- these twins exist
// as the measurement record + a cheap re-probe for future GPUs. NOT
// dispatched by any production path.
#define DGVR(NAME, BM, BN, SG)                                           \
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
    dense_gemm_mma_impl<BM, BN, SG, 1, 1, /*RELAX=*/true>(               \
        x, W, y, K, N, M, tgid);                                         \
  }

DGVR(dense_gemm_mma_t_n128_rp_f16,     128, 128, 8)
DGVR(dense_gemm_mma_t_n128x256_rp_f16, 128, 256, 8)

// Software TM x TN grid of BMxBN tiles per threadgroup (each tg computes a
// (TM*BM) x (TN*BN) region), raising x/W reuse per loaded tile. Dispatch grid
// divides by the effective region (TM*BM, TN*BN).
#define DGVT(NAME, BM, BN, SG, TM, TN)                                   \
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
    dense_gemm_mma_impl<BM, BN, SG, TM, TN>(x, W, y, K, N, M, tgid);     \
  }

// 128x256 with TN=2 software tiling (each tg computes two adjacent 128x256
// N-tiles = a 128x512 region, reusing the loaded x-slice across both). At the
// Krea2 K=6144 block-projection shapes (M~=seq 4106) this beats the plain
// 128x256 ~1.14-1.19x (x-streaming is the bind; x-reuse doubles) -- the tile the
// DiT block projections (qkv/o/ff-up/ff-gate) dispatch. Loses badly at the very
// deep K=16384 ff-down (that goes through split-K instead), so the model routes
// K in [6144, 12288) here and keeps plain 128x256 for deeper unsplit K.
DGVT(dense_gemm_mma_t_n128x256_tn2_f16, 128, 256, 8, 1, 2)

// Split-K deep-reduction tile. A single-op full-K reduction underutilizes the
// matrix units once K gets very deep (Krea2 ff-down, K=16384 runs ~0.7x the
// K<=9728 rate): too few threadgroups (N/BN of them) each walking a long serial
// contraction, so the matrix pipeline stalls with nothing else in flight. This
// variant slices the contraction into n_splits chunks of KC and gives each its
// OWN threadgroup (grid.z = split), so the tg count multiplies by n_splits and
// each reduction shortens back into the fast regime. tgid.z picks the K-chunk
// [tgid.z*KC, tgid.z*KC+KC); the tg writes its BMxBN PARTIAL into plane tgid.z
// of yp[n_splits, M, N]. A cheap residual_add pass then folds the planes into
// the final y (one extra f16 rounding per fold -- fine for the rel-L2-verified
// DiT, not token-exact). KC is a COMPILE-TIME constant so the matmul2d
// contraction extent is static and bounds the reduce to KC from the sliced
// origin; the caller must satisfy K == n_splits * KC. x/W keep their full
// (K,M)/(K,N) extents so the per-row stride stays K after the slice(k0,..).
template <int BM, int BN, int SG, int KC>
static inline void dense_gemm_mma_splitk_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    device VPIPE_ELT* yp, int K, int N, int M, uint3 tgid)
{
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W), dextents<int32_t, 2>(K, N));
  const int kz = (int)tgid.z;
  TX tY(yp + (int64_t)kz * (int64_t)M * (int64_t)N,
        dextents<int32_t, 2>(N, M));            // partial plane kz, [M,N]

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int k0 = kz * KC;
  auto mX = tX.slice(k0, m0);        // origin (K=k0, M=m0); reduce KC along K
  auto mW = tW.slice(k0, n0);        // origin (K=k0, N=n0)
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX), decltype(mW), VPIPE_ELT>();
  op.run(mX, mW, cT);
  auto mY = tY.slice(n0, m0);
  cT.store(mY);
}

#define DGVK(NAME, BM, BN, SG, KC)                                       \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      const device VPIPE_ELT* W [[buffer(1)]],                           \
      device VPIPE_ELT* yp [[buffer(2)]],                                \
      const constant int& K [[buffer(3)]],                              \
      const constant int& N [[buffer(4)]],                              \
      const constant int& M [[buffer(5)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    dense_gemm_mma_splitk_impl<BM, BN, SG, KC>(x, W, yp, K, N, M, tgid); \
  }

// KC=8192 -> Krea2 ff-down (K=16384) splits into 2 planes. 128x256 tile: a
// single-GEMM tune of the K=8192 chunk favored a 64x256 tile (1.2x), but in the
// model split-K runs both planes CONCURRENTLY (grid.z=2), so 64x256's doubled
// threadgroup count just contends -- it LOST ~1.2x on the ff-down section and is
// not shipped. 128x256 (fewer, larger tgs) is the split tile.
DGVK(dense_gemm_mma_splitk_n128x256_k8192_f16, 128, 256, 8, 8192)

// Causal/windowed QK for the MATERIALIZED attention path (M5 matrix-core core
// on top of the M4 diagonal-grid exploit). y[m=query, n=key] = Q[m,:].K[n,:]
// (x @ W^T, contraction K=D, full). A threadgroup whose key-column region is
// entirely above the causal diagonal -- smallest key (n0) past the block's
// largest query (q_offset + m0 + BM - 1) -- early-returns, leaving y unwritten;
// for window>0 a block entirely BELOW the trailing window does too. The
// downstream causal_softmax_rows (banded=0) rewrites the whole row [0,N), so the
// skipped region is never read (same contract as steel dense_gemm_t_qkcausal,
// just with the matmul2d core). M/N tails ride the tensor-extent clamp like the
// dense entry above. Buffers add q_offset(8)/window(9); has_bias is unused.
template <int BM, int BN, int SG>
static inline void dense_gemm_mma_qkcausal_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    device VPIPE_ELT* y, int K, int N, int M,
    int q_offset, int window, uint3 tgid)
{
  const int m0 = (int)tgid.y * BM;          // query-row base
  const int n0 = (int)tgid.x * BN;          // key-col base
  if (n0 > q_offset + m0 + BM - 1) { return; }              // above diagonal
  if (window > 0 && n0 + BN - 1 < q_offset + m0 - window + 1) { return; }
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W), dextents<int32_t, 2>(K, N));
  TX tY(y, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mX = tX.slice(0, m0);
  auto mW = tW.slice(0, n0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX), decltype(mW), VPIPE_ELT>();
  op.run(mX, mW, cT);
  auto mY = tY.slice(n0, m0);
  cT.store(mY);
}

#define DGVC(NAME, BM, BN, SG)                                           \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      const device VPIPE_ELT* W [[buffer(1)]],                           \
      const device VPIPE_ELT* bias [[buffer(2)]],                        \
      device VPIPE_ELT* y [[buffer(3)]],                                 \
      const constant int& K [[buffer(4)]],                              \
      const constant int& N [[buffer(5)]],                              \
      const constant int& M [[buffer(6)]],                              \
      const constant int& has_bias [[buffer(7)]],                       \
      const constant int& q_offset [[buffer(8)]],                       \
      const constant int& window [[buffer(9)]],                         \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    (void)has_bias; (void)bias;                                          \
    dense_gemm_mma_qkcausal_impl<BM, BN, SG>(                            \
        x, W, y, K, N, M, q_offset, window, tgid);                      \
  }

DGVC(dense_gemm_mma_t_qkcausal_n128_f16, 128, 128, 8)

// Full int8 GEMM with fused dequant epilogue (the int8-FFN prototype):
//   y[M,N] (f16) = (xq[M,K] i8 @ wq[N,K]^T i8) * as[m] * ws[n]
// xq is the per-ROW (token) quantized activation (quant_f16_i8_row -- one
// scale per whole K row, since the hw op accumulates int32 over the full
// contraction), wq the OFFLINE per-out-channel quantized weight. The
// matmul2d runs i8 x i8 -> i32 into a THREADGROUP tile (the only
// overflow-safe destination: full-range i8 sums over deep K reach ~1e7,
// far past f16), then the epilogue scales in tgmem and stores f16 --
// no [M,N] i32 DRAM round-trip. BM=BN=64 (i32 tile = 16 KB tgmem), 4
// simdgroups. M/N tails: source extents clamp the reads; the epilogue
// guards the stores.
//   0:xq 1:wq 2:as[M](f16) 3:ws[N](f16) 4:y(f16) 5:K 6:N 7:M
//   dispatch (threads): {ceil(N/64)*128, ceil(M/64), 1}, tg {128,1,1}
kernel void gemm_i8i8_sc_f16_n64(
    const device int8_t* xq [[buffer(0)]],
    const device int8_t* wq [[buffer(1)]],
    const device half*   as [[buffer(2)]],
    const device half*   ws [[buffer(3)]],
    device half*         y  [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  constexpr int BM = 64, BN = 64, SG = 4;
  threadgroup int Ys[BM * BN];

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  using TY = tensor<threadgroup int, dextents<int32_t, 2>, tensor_inline>;
  TY tY(Ys, dextents<int32_t, 2>(BN, BM));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  auto mX = tX.slice(0, m0);
  auto mW = tW.slice(0, n0);
  op.run(mX, mW, tY);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Epilogue: dequant-scale the i32 tile and store f16.
  for (int e = (int)lid; e < BM * BN; e += SG * 32) {
    const int i = e / BN, j = e % BN;
    const int gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) {
      const float v = (float)Ys[e] * (float)as[gm] * (float)ws[gn];
      y[(int64_t)gm * N + gn] = (half)v;
    }
  }
}

// K=512-CHUNKED twins of the int8 GEMM, decomposing the contraction into
// KC-sized ops (static contraction extent, sliced origins -- the split-K
// pattern, but SERIAL within one threadgroup):
//
// _kacc: mode::multiply_accumulate accumulates the chunks' RAW int32
// partials in the tgmem tile (the op does the +=). Scales stay per-row/
// per-channel (int accumulation cannot mix per-chunk scales), so this
// measures the pure cost of chunking the op.
//
// _g512: per-chunk mode::multiply + the accuracy point of chunking: the
// i32 partial is scaled by PER-GROUP quant scales (as[m,g] * ws[n,g],
// g = k/512) and accumulated in per-thread FLOAT registers, so each
// 512-deep group carries its own activation/weight scale (finer
// quantization; needs quant_f16_i8_row_g512 + [N, K/512] weight scales).
// Same tile/dispatch contract as gemm_i8i8_sc_f16_n64; K % 512 == 0.
#define GI8_KC 512

kernel void gemm_i8i8_sc_f16_n64_kacc(
    const device int8_t* xq [[buffer(0)]],
    const device int8_t* wq [[buffer(1)]],
    const device half*   as [[buffer(2)]],   // [M] per-token
    const device half*   ws [[buffer(3)]],   // [N] per-channel
    device half*         y  [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  constexpr int BM = 64, BN = 64, SG = 4;
  threadgroup int Ys[BM * BN];
  for (int e = (int)lid; e < BM * BN; e += SG * 32) { Ys[e] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  using TY = tensor<threadgroup int, dextents<int32_t, 2>, tensor_inline>;
  TY tY(Ys, dextents<int32_t, 2>(BN, BM));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GI8_KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  for (int k0 = 0; k0 < K; k0 += GI8_KC) {
    auto mX = tX.slice(k0, m0);
    auto mW = tW.slice(k0, n0);
    op.run(mX, mW, tY);                       // Ys += chunk partial
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int e = (int)lid; e < BM * BN; e += SG * 32) {
    const int i = e / BN, j = e % BN;
    const int gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) {
      const float v = (float)Ys[e] * (float)as[gm] * (float)ws[gn];
      y[(int64_t)gm * N + gn] = (half)v;
    }
  }
}

kernel void gemm_i8i8_sc_f16_n64_g512(
    const device int8_t* xq [[buffer(0)]],
    const device int8_t* wq [[buffer(1)]],
    const device half*   as [[buffer(2)]],   // [M, K/512] per-token-group
    const device half*   ws [[buffer(3)]],   // [N, K/512] per-chan-group
    device half*         y  [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  constexpr int BM = 64, BN = 64, SG = 4;
  constexpr int EPT = (BM * BN) / (SG * 32);  // elements per thread (32)
  threadgroup int Ys[BM * BN];

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  using TY = tensor<threadgroup int, dextents<int32_t, 2>, tensor_inline>;
  TY tY(Ys, dextents<int32_t, 2>(BN, BM));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GI8_KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / GI8_KC;

  float facc[EPT] = {0.0f};
  for (int g = 0; g < G; ++g) {
    auto mX = tX.slice(g * GI8_KC, m0);
    auto mW = tW.slice(g * GI8_KC, n0);
    op.run(mX, mW, tY);                       // chunk partial (overwrite)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < EPT; ++t) {
      const int e = (int)lid + t * (SG * 32);
      const int i = e / BN, j = e % BN;
      const int gm = m0 + i, gn = n0 + j;
      if (gm < M && gn < N) {
        facc[t] += (float)Ys[e] * (float)as[(int64_t)gm * G + g] *
                   (float)ws[(int64_t)gn * G + g];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // before overwrite
  }
  for (int t = 0; t < EPT; ++t) {
    const int e = (int)lid + t * (SG * 32);
    const int i = e / BN, j = e % BN;
    const int gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) { y[(int64_t)gm * N + gn] = (half)facc[t]; }
  }
}

// SHIFT-ALIGNED integer accumulation twin of _g512 (block-floating-point
// accumulate): group quant scales are POWERS OF TWO (quant_f16_i8_row_
// g512_bfp activations + pow2 offline weights, exponents as int8), so
// the per-chunk partial carries a per-element total exponent eg =
// ea[m,g] + ew[n,g] and the accumulator stays PURE i32 with a tracked
// exponent: incoming coarser (eg > ae) -> right-shift the accumulator to
// match (round-to-nearest); incoming finer -> right-shift the incoming
// partial. One ldexp((float)acc, ae) converts to f16 at the end -- NO
// float scale multiplies anywhere, and the 31-bit integer accumulator
// carries more mantissa than the f32 accumulate it replaces. Headroom:
// |partial| <= 512*127*127 ~ 2^23, up to 48 aligned chunks ~ 2^28.6 <
// i32. Same contract as _g512 but buffers 2/3 are int8 exponents.
//
// MEASURED (gemm_i8.k512_shift_acc): bit-EXACT vs the CPU replica of the
// algorithm (oracle rel-L2 0.0) -- but on M5 it buys NOTHING over the
// float-accumulate _g512: same speed (0.75-0.79x of the single op; both
// epilogues are tgmem-drain-bound, and integer shifts run no faster than
// the float pipe's fma) and WORSE quality on gaussian data (1.46e-2 vs
// 1.05e-2 float-g512 / 1.11e-2 single-op) because pow2 scales on BOTH
// operands each give up ~0.6 bit vs amax. Kept as the measurement record
// + for hardware without fast float pipes or for cross-device bit-exact
// reproducibility requirements.
kernel void gemm_i8i8_sc_f16_n64_g512i(
    const device int8_t* xq [[buffer(0)]],
    const device int8_t* wq [[buffer(1)]],
    const device char*   ea [[buffer(2)]],   // [M, K/512] act exponents
    const device char*   ew [[buffer(3)]],   // [N, K/512] wt exponents
    device half*         y  [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  constexpr int BM = 64, BN = 64, SG = 4;
  constexpr int EPT = (BM * BN) / (SG * 32);
  threadgroup int Ys[BM * BN];

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  using TY = tensor<threadgroup int, dextents<int32_t, 2>, tensor_inline>;
  TY tY(Ys, dextents<int32_t, 2>(BN, BM));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, GI8_KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / GI8_KC;

  int   acc[EPT];
  short ae[EPT];
  for (int t = 0; t < EPT; ++t) { acc[t] = 0; ae[t] = -1000; }

  for (int g = 0; g < G; ++g) {
    auto mX = tX.slice(g * GI8_KC, m0);
    auto mW = tW.slice(g * GI8_KC, n0);
    op.run(mX, mW, tY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < EPT; ++t) {
      const int e = (int)lid + t * (SG * 32);
      const int i = e / BN, j = e % BN;
      const int gm = m0 + i, gn = n0 + j;
      if (gm < M && gn < N) {
        const int eg = (int)ea[(int64_t)gm * G + g] +
                       (int)ew[(int64_t)gn * G + g];
        int p = Ys[e];
        const int d = eg - (int)ae[t];
        if (d > 0) {
          // Incoming coarser: align the accumulator down (rounded).
          acc[t] = (d >= 31) ? 0
                             : ((acc[t] + (1 << (d - 1))) >> d);
          acc[t] += p;
          ae[t] = (short)eg;
        } else if (d < 0) {
          // Incoming finer: align the partial down (rounded).
          const int dd = -d;
          p = (dd >= 31) ? 0 : ((p + (1 << (dd - 1))) >> dd);
          acc[t] += p;
        } else {
          acc[t] += p;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int t = 0; t < EPT; ++t) {
    const int e = (int)lid + t * (SG * 32);
    const int i = e / BN, j = e % BN;
    const int gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) {
      y[(int64_t)gm * N + gn] = (half)ldexp((float)acc[t], (int)ae[t]);
    }
  }
}

#else
// Tensor ops unavailable for this target: emit stubs so the metallib
// still builds. The loader never binds these on a non-tensor GPU.
#define DGV_STUB(NAME)                                                   \
  kernel void NAME(device float* y [[buffer(3)]],                        \
                   uint tid [[thread_position_in_grid]]) {               \
    if (tid == 0) { y[0] = 0.0f; }                                       \
  }
DGV_STUB(dense_gemm_mma_t_f16)
DGV_STUB(gemm_i8i8_sc_f16_n64)
DGV_STUB(gemm_i8i8_sc_f16_n64_kacc)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512i)
DGV_STUB(dense_gemm_mma_t_n128_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_f16)
DGV_STUB(dense_gemm_mma_t_n128_rp_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_rp_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_tn2_f16)
DGV_STUB(dense_gemm_mma_splitk_n128x256_k8192_f16)
DGV_STUB(dense_gemm_mma_t_qkcausal_n128_f16)
#endif
