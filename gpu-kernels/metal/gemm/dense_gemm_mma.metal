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

  // Skip a sub-tile whose ORIGIN is already past the extent. The matmul2d
  // tensors clamp a PARTIALLY out-of-range tile (that is what makes ragged M/N
  // tails safe), but a tile that starts at or beyond N/M is a slice from an
  // out-of-contract origin, and it does not stay in its lane: at TN=2 the
  // dispatch rounds N up to TN*BN, so N=13568 with BN=256 leaves the tail
  // threadgroup's SECOND tile beginning exactly at 13568 -- and its store
  // landed on the following rows. Measured as rel-L2 0.19 against the same
  // GEMM at TN=1 (boogu_perf.mma_tile_sweep, ff-gate/up), while ragged-but-
  // overlapping cases were bit-exact -- which is why only the fully-past tile
  // needs this. The bounds are uniform across the threadgroup (they depend on
  // tgid and compile-time constants only), so this costs no divergence.
  // No shipped output was ever wrong: TN=2 is the only tile that can produce a
  // fully-past sub-tile, and the only model routing to it before now was
  // Krea-2, whose tn2 widths (2560 / 6144 / 32768) are all exact multiples of
  // TN*BN = 512. Boogu's 13568 is the first that is not.
  for (int tm = 0; tm < TM; ++tm) {
    const int m0 = m_base + tm * BM;
    if (m0 >= M) { break; }
    auto mX = tX.slice(0, m0);          // rows [m0, m0+BM) of x
    for (int tn = 0; tn < TN; ++tn) {
      const int n0 = n_base + tn * BN;
      if (n0 >= N) { break; }
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

// ---- the two tiles a runtime LoRA needs --------------------------------
//
// An adapted projection computes y = W x + s * B (A x), which the model
// encodes as t = x A^T followed by y += s * t B^T. Both halves are dense
// bf16 GEMMs that the DENSE tiles above can already serve -- except for
// the two things that make them a LoRA rather than a projection: the
// second one ACCUMULATES onto a y the base GEMM has already written, and
// the strength `s` is a per-forward constant that must not be baked into
// either factor (it is the request's knob, not the adapter's property).
//
// Both ride the register tile, so neither costs a pass over y:
//
//   SCALED   multiplies the accumulator by s before the store. It goes on
//            the FIRST GEMM -- t is [M, rank], the smallest buffer in the
//            pair, so scaling there touches ~rank/N as many elements as
//            scaling the result would, and s*(x A^T) B^T is the same
//            product either way. That is what leaves the second tile free
//            to be a plain accumulate, which is the only form matmul2d
//            offers: mode::multiply_accumulate adds into the cooperative
//            tensor, and there is nowhere in it to put a coefficient.
//   ACC      seeds the cooperative tensor from y instead of starting at
//            zero, so the existing value is folded in the register file
//            and the tile stores once. The steel twin
//            (dense_gemm_t_bm64_acc_f16) does the same thing through its
//            ScaleAddEpilogue, and carries the scale itself because a
//            scalar epilogue has room for one.
//
// Precision matches the overwriting tiles: relaxed_precision stays false,
// so the contraction accumulates in f32 whatever the cooperative tensor's
// storage type is, and y's bf16 round-trip is the one it already had.
template <int BM, int BN, int SG, bool ACC>
static inline void dense_gemm_mma_lora_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    device VPIPE_ELT* y, int K, int N, int M, float scale, uint3 tgid)
{
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W), dextents<int32_t, 2>(K, N));
  TX tY(y, dextents<int32_t, 2>(N, M));

  // Each half takes the mode that MEANS what it is doing: the scaled
  // tile overwrites, the accumulating one folds y in and needs the
  // cooperative tensor seeded. A single multiply_accumulate descriptor
  // with the tile zeroed first computes the same thing and MEASURED the
  // same -- so this split is about the contract, not about a number.
  // Worth stating because the reverse was assumed here for a while: the
  // accumulate mode does NOT demote the contraction to the cooperative
  // tensor's element type, since relaxed_precision is what governs that
  // and it is false on both.
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      ACC ? matmul2d_descriptor::mode::multiply_accumulate
          : matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  // Same rule as dense_gemm_mma_impl: a partially out-of-range tile is
  // clamped by the tensor extents, but one whose ORIGIN is past M/N is a
  // slice from outside the contract and its store does not stay in lane.
  if (m0 >= M || n0 >= N) { return; }
  auto mX = tX.slice(0, m0);
  auto mW = tW.slice(0, n0);
  auto mY = tY.slice(n0, m0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX), decltype(mW), VPIPE_ELT>();
  if (ACC) { cT.load(mY); }
  op.run(mX, mW, cT);
  if (!ACC) {
    for (auto it = cT.begin(); it != cT.end(); ++it) {
      *it = (VPIPE_ELT)((float)*it * scale);
    }
  }
  cT.store(mY);
}

// The base projection and the adapter's second factor in ONE tile:
//
//   y = x W^T + t B^T          K for the first, RANK for the second
//
// This is the form that makes a runtime LoRA nearly free, and the reason
// it is possible at all is that matmul2d accumulates into a REGISTER
// tile. Two op.run() calls against one cooperative tensor contract over
// different depths and land in the same accumulator, so the delta costs
// its own MACs and nothing else.
//
// What it removes is not arithmetic but traffic. Run separately, the
// second GEMM has to READ y, add, and WRITE y -- 2*M*N*2 bytes against
// only 2*M*r*N flops, which at rank 64 is 32 flops/byte and lands it
// squarely on the memory roofline (measured ~107 GB/s of this machine's
// ~150, i.e. already near the limit, so no faster multiplier helps).
// Folded here it reads t[BM, r] and B[BN, r] per tile instead, and y is
// written exactly once by the projection that was writing it anyway.
//
// t carries the STRENGTH: the caller's first GEMM scales it. There is
// nowhere to put a coefficient in a multiply_accumulate, and this tile
// deliberately does not grow one -- an adapter whose strength did not
// reach t must not take this path, or it would silently run at 1.0.
template <int BM, int BN, int SG>
static inline void dense_gemm_mma_lora_fused_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    const device VPIPE_ELT* t, const device VPIPE_ELT* Bf,
    device VPIPE_ELT* y, int K, int N, int M, int R, uint3 tgid)
{
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x),  dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W),  dextents<int32_t, 2>(K, N));
  TX tT(const_cast<device VPIPE_ELT*>(t),  dextents<int32_t, 2>(R, M));
  TX tB(const_cast<device VPIPE_ELT*>(Bf), dextents<int32_t, 2>(R, N));
  TX tY(y, dextents<int32_t, 2>(N, M));

  // TWO descriptors, differing only in mode, and WHICH run gets which is
  // the whole performance of this kernel. multiply_accumulate is not free
  // on a deep contraction: running the base under it cost fc2 (K=14336,
  // 128x256 tile) 252 ms against the plain tile's 163 -- a 55% loss on
  // the projection to save 12 ms on the delta, while the same arrangement
  // was a clear win at K=5376 and K=7168. Whatever it costs scales with
  // the contraction, so the BASE keeps the plain mode (its tile is
  // written first, unconditioned, exactly as dense_gemm_mma_impl does it)
  // and only the rank-deep delta accumulates on top. That also removes
  // the zero-init the other order needs.
  constexpr auto desc_mul = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  constexpr auto desc_acc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_mul, execution_simdgroups<SG>> op;
  matmul2d<desc_acc, execution_simdgroups<SG>> op_acc;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  if (m0 >= M || n0 >= N) { return; }
  auto mX = tX.slice(0, m0);
  auto mW = tW.slice(0, n0);
  auto mT = tT.slice(0, m0);
  auto mB = tB.slice(0, n0);
  auto mY = tY.slice(n0, m0);
  // Both pairs slice to the same types, so ONE cooperative tensor serves
  // both contractions; only the depth differs, and the descriptors take
  // that from the tensors because K is dynamic_extent.
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX), decltype(mW), VPIPE_ELT>();
  op.run(mX, mW, cT);
  op_acc.run(mT, mB, cT);
  cT.store(mY);
}

#define DGVF(NAME, BM, BN, SG)                                           \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      const device VPIPE_ELT* W [[buffer(1)]],                           \
      const device VPIPE_ELT* bias [[buffer(2)]],                        \
      device VPIPE_ELT* y [[buffer(3)]],                                 \
      const constant int& K [[buffer(4)]],                              \
      const constant int& N [[buffer(5)]],                              \
      const constant int& M [[buffer(6)]],                              \
      const constant int& has_bias [[buffer(7)]],                       \
      const device VPIPE_ELT* t [[buffer(8)]],                           \
      const device VPIPE_ELT* Bf [[buffer(9)]],                          \
      const constant int& R [[buffer(10)]],                             \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    (void)has_bias; (void)bias;                                          \
    dense_gemm_mma_lora_fused_impl<BM, BN, SG>(                          \
        x, W, t, Bf, y, K, N, M, R, tgid);                               \
  }

// Buffers 0-7 match the plain tiles exactly, so the host binds the same
// projection and only adds the adapter on 8-10.
DGVF(dense_gemm_mma_t_n128_lora_f16,     128, 128, 8)
DGVF(dense_gemm_mma_t_n128x256_lora_f16, 128, 256, 8)

#define DGVL(NAME, BM, BN, SG, ACC)                                      \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      const device VPIPE_ELT* W [[buffer(1)]],                           \
      const device VPIPE_ELT* bias [[buffer(2)]],                        \
      device VPIPE_ELT* y [[buffer(3)]],                                 \
      const constant int& K [[buffer(4)]],                              \
      const constant int& N [[buffer(5)]],                              \
      const constant int& M [[buffer(6)]],                              \
      const constant int& has_bias [[buffer(7)]],                       \
      const constant float& scale [[buffer(8)]],                        \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    (void)has_bias; (void)bias;                                          \
    dense_gemm_mma_lora_impl<BM, BN, SG, ACC>(                           \
        x, W, y, K, N, M, scale, tgid);                                  \
  }

// The A half (t = s * x A^T). Its N is the RANK, so the 64-wide tile is
// the one that fits rank 64 without half of it hanging past N; rank 128
// and the stacked rank-384 qkv take the 128-wide tile.
DGVL(dense_gemm_mma_t_scaled_f16,      64,  64, 4, false)
DGVL(dense_gemm_mma_t_n128_scaled_f16, 128, 128, 8, false)
// The B half (y += t B^T). Its K is the rank -- a very shallow
// contraction -- and its N is the projection's full width.
DGVL(dense_gemm_mma_t_n128_acc_f16,    128, 128, 8, true)
DGVL(dense_gemm_mma_t_n128x256_acc_f16, 128, 256, 8, true)

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

// f32-PLANE twin of dense_gemm_mma_splitk_impl. Identical tiling and identical
// contraction; the only change is that the partial plane is float, so the fold
// (splitk_fold_f32_f16) can sum in f32 and round to the compute elt exactly
// once -- matching the single-op kernel's ONE rounding instead of paying one
// per plane. The accumulator was always f32; this just stops throwing that
// away between the split and the fold.
//
// That matters here and did not for the diffusion callers: an LM is held to
// greedy token-exact, and the f16-plane fold perturbs ~37% of down_proj
// outputs by up to ~5 ulp, which is exactly the kind of drift that flips a
// near-tie argmax. With f32 planes the split output is bit-identical to the
// single-op output at these shapes (optiq_blocks.splitk_down_proj).
template <int BM, int BN, int SG, int KC>
static inline void dense_gemm_mma_splitk_f32p_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    device float* yp, int K, int N, int M, uint3 tgid)
{
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  using TF = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W), dextents<int32_t, 2>(K, N));
  const int kz = (int)tgid.z;
  TF tY(yp + (int64_t)kz * (int64_t)M * (int64_t)N,
        dextents<int32_t, 2>(N, M));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int k0 = kz * KC;
  auto mX = tX.slice(k0, m0);
  auto mW = tW.slice(k0, n0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX), decltype(mW), float>();
  op.run(mX, mW, cT);
  auto mY = tY.slice(n0, m0);
  cT.store(mY);
}

#define DGVKF(NAME, BM, BN, SG, KC)                                      \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* x [[buffer(0)]],                           \
      const device VPIPE_ELT* W [[buffer(1)]],                           \
      device float* yp [[buffer(2)]],                                    \
      const constant int& K [[buffer(3)]],                              \
      const constant int& N [[buffer(4)]],                              \
      const constant int& M [[buffer(5)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    dense_gemm_mma_splitk_f32p_impl<BM, BN, SG, KC>(                     \
        x, W, yp, K, N, M, tgid);                                        \
  }

// KC=8192 -> Krea2 ff-down (K=16384) splits into 2 planes. 128x256 tile: a
// single-GEMM tune of the K=8192 chunk favored a 64x256 tile (1.2x), but in the
// model split-K runs both planes CONCURRENTLY (grid.z=2), so 64x256's doubled
// threadgroup count just contends -- it LOST ~1.2x on the ff-down section and is
// not shipped. 128x256 (fewer, larger tgs) is the split tile.
DGVK(dense_gemm_mma_splitk_n128x256_k8192_f16, 128, 256, 8, 8192)

// Boogu's ff-down is K=13568 (ff_inner = 4*3360 rounded up to 256), which no
// multiple of 8192 divides -- so the KC=8192 tile above cannot serve it and the
// shape fell through to the unsplit 128x256 path. 13568 = 2*6784 = 4*3392, so
// both a 2-way and a 4-way split are exact. 2-way wins at every M measured
// (11.76 vs 11.40 TFLOP/s at M=2271, 11.60 vs 11.26 at M=4104): more planes
// shortens each reduction but multiplies the threadgroups competing for the
// matrix units, and by 4 the second effect has taken over. The 4-way twin is
// kept only so boogu_perf.mma_tile_sweep can re-establish that on a new GPU.
DGVK(dense_gemm_mma_splitk_n128x256_k6784_f16, 128, 256, 8, 6784)
DGVK(dense_gemm_mma_splitk_n128x256_k3392_f16, 128, 256, 8, 3392)

// The LM down_proj shapes. Exactly the same wall the Krea2/Boogu ff-down hit,
// reached from the language side: the two large dense OptiQ checkpoints have
// ffn = 17408 (Qwen3.6-27B) and 21504 (gemma-4-31B), and at those depths the
// single-op reduction runs ~4.4-5.2 TFLOP/s where the same weights in the
// gate|up direction reach 10-11. A transpose control (N and K swapped,
// identical FLOPs and identical weight bytes) runs 10.9-11.6, which is what
// establishes it as contraction DEPTH and not the narrow N.
//
// Neither depth is a multiple of any KC above, so both need their own.
// 17408 = 2*8704 = 4*4352 and 21504 = 2*10752 = 4*5376.
//
// f32-plane, the variant the LM path dispatches (the f16-plane twins above
// stay for the diffusion callers, which fold with residual_add).
//
// These two widths are no longer the choice -- mma-splitk.h picks a split by
// CHUNK WIDTH, aiming near 2048, so at these depths it takes kc2176 (S=8)
// and kc1792 (S=12) from the ladder below instead. Re-measured on M5 with
// the clock held at ~1.4 GHz, kc4352 reaches 5933 GF/s at M=4096 where
// kc2176 reaches 10566 against a 12176 gate|up reference: the 4-way widths
// recover about a quarter of the deficit and the narrow ones nearly all of
// it. They stay instantiated because the chooser can still land on them
// when a plane budget rules the deeper split out.
DGVKF(dense_gemm_mma_splitk32_n128x256_k8704_f16, 128, 256, 8, 8704)
DGVKF(dense_gemm_mma_splitk32_n128x256_k10752_f16, 128, 256, 8, 10752)
// 4-way twins (17408 = 4*4352, 21504 = 4*5376). Boogu picked 2-way over 4-way
// at M=2271-4104, but a 2-way split of these shapes leaves only ~320
// threadgroups at M=1024 -- under-occupied, and it shows as a much smaller win
// there than at M=4096. Which one the model dispatches is decided by
// measurement per (shape, M); see optiq_blocks.splitk_down_proj.
DGVKF(dense_gemm_mma_splitk32_n128x256_k4352_f16, 128, 256, 8, 4352)
DGVKF(dense_gemm_mma_splitk32_n128x256_k5376_f16, 128, 256, 8, 5376)

// The rest of the ladder. mma-splitk.h resolves a chunk kernel BY NAME from
// the width it wants, so this list is the menu the chooser picks from -- a
// depth whose target width is missing here silently gets a worse split, or
// none. These cover K/S for S = 2..16 over the ffn depths in this tree
// (9216, 12288, 13568, 14336, 16384, 17408, 21504).
DGVKF(dense_gemm_mma_splitk32_n128x256_k7168_f16, 128, 256, 8, 7168)
DGVKF(dense_gemm_mma_splitk32_n128x256_k3584_f16, 128, 256, 8, 3584)
DGVKF(dense_gemm_mma_splitk32_n128x256_k6144_f16, 128, 256, 8, 6144)
DGVKF(dense_gemm_mma_splitk32_n128x256_k3072_f16, 128, 256, 8, 3072)

// The band the chooser actually lands in. Measured best width per depth
// (M5, M=4096-9382): 12288 -> 4096-2048, 14336 -> 2048-1792, 17408 -> 2176,
// 21504 -> 1792-1536. Below ~1300 the extra weight re-reads start costing
// more than the shortened reduction buys (kc1024 loses to kc2048 at every
// depth measured), which is what bounds the ladder at the narrow end.
DGVKF(dense_gemm_mma_splitk32_n128x256_k2176_f16, 128, 256, 8, 2176)
DGVKF(dense_gemm_mma_splitk32_n128x256_k2688_f16, 128, 256, 8, 2688)
DGVKF(dense_gemm_mma_splitk32_n128x256_k1792_f16, 128, 256, 8, 1792)
DGVKF(dense_gemm_mma_splitk32_n128x256_k1536_f16, 128, 256, 8, 1536)

// The rest of the chunk-width ladder. The host picks a split by CHUNK WIDTH
// (mma-splitk.h): given K it walks the split counts whose chunk is exact and
// takes the width nearest the target, so which widths exist here is what
// bounds the choice. These fill the band a real ffn depth can land in --
// every ffn in this tree is a multiple of 256, and K/S for S = 2..16 over
// {9216, 12288, 13568, 14336, 16384, 17408, 21504} lands on these.
DGVKF(dense_gemm_mma_splitk32_n128x256_k1024_f16, 128, 256, 8, 1024)
DGVKF(dense_gemm_mma_splitk32_n128x256_k1088_f16, 128, 256, 8, 1088)
DGVKF(dense_gemm_mma_splitk32_n128x256_k1344_f16, 128, 256, 8, 1344)
DGVKF(dense_gemm_mma_splitk32_n128x256_k1696_f16, 128, 256, 8, 1696)
DGVKF(dense_gemm_mma_splitk32_n128x256_k2048_f16, 128, 256, 8, 2048)
DGVKF(dense_gemm_mma_splitk32_n128x256_k4096_f16, 128, 256, 8, 4096)
// (kc 2304 is instantiated below, for the 9216 verification depth.)

// VERIFICATION ONLY, never dispatched by default: 9216 = 4*2304 is
// Qwen3.5-4B's ffn, which is the deepest down_proj on a checkpoint that fits
// in 16 GB. The two depths this path actually ships for belong to models that
// do not fit, so without this there is no machine on which the split-K wiring
// -- planes, fold, and the token stream that comes out the other end -- can be
// run against a real decoder at all. VPIPE_LM_SPLITK_TEST=1 turns it on.
// 9216 is nowhere near deep enough to WANT splitting; this buys correctness
// coverage, not rate.
DGVKF(dense_gemm_mma_splitk32_n128x256_k2304_f16, 128, 256, 8, 2304)

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

// Element type of the scales/output is VPIPE_ELT (half or bfloat): the FLUX.2
// bf16 DiT loads this from the _bf16 metallib so it reads its bf16 scales and
// stores a bf16 result (the int8 tensors are format-independent). The int32
// tile accumulation + f32 per-group scaling are unchanged.
kernel void gemm_i8i8_sc_f16_n64_g512(
    const device int8_t*    xq [[buffer(0)]],
    const device int8_t*    wq [[buffer(1)]],
    const device VPIPE_ELT* as [[buffer(2)]],   // [M, K/512] per-token-group
    const device VPIPE_ELT* ws [[buffer(3)]],   // [N, K/512] per-chan-group
    device VPIPE_ELT*       y  [[buffer(4)]],
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
    if (gm < M && gn < N) { y[(int64_t)gm * N + gn] = (VPIPE_ELT)facc[t]; }
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
DGV_STUB(dense_gemm_mma_t_scaled_f16)
DGV_STUB(dense_gemm_mma_t_n128_scaled_f16)
DGV_STUB(dense_gemm_mma_t_n128_acc_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_acc_f16)
DGV_STUB(dense_gemm_mma_t_n128_lora_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_lora_f16)
DGV_STUB(dense_gemm_mma_splitk_n128x256_k8192_f16)
DGV_STUB(dense_gemm_mma_splitk_n128x256_k6784_f16)
DGV_STUB(dense_gemm_mma_splitk_n128x256_k3392_f16)
DGV_STUB(dense_gemm_mma_splitk32_n128x256_k8704_f16)
DGV_STUB(dense_gemm_mma_splitk32_n128x256_k10752_f16)
DGV_STUB(dense_gemm_mma_splitk32_n128x256_k4352_f16)
DGV_STUB(dense_gemm_mma_splitk32_n128x256_k5376_f16)
DGV_STUB(dense_gemm_mma_splitk32_n128x256_k2304_f16)
DGV_STUB(dense_gemm_mma_t_qkcausal_n128_f16)
#endif
