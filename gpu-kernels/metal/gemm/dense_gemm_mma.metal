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

// BANDED twins of the two kernels above, for a runtime LoRA that fuses
// several separate adapters into ONE projection's output -- q, k and v
// into MiniMax-H3's qkv_proj. Each output column belongs to exactly one
// part and contracts only over that part's rank, so B is stored [N, R]
// with no zeros (not the [N, parts*R] block-diagonal it would otherwise
// be) and the tile reads the matching R-wide WINDOW of t, which is
// [M, ldt] with the parts' ranks side by side:
//
//   part = (n0 / group) % parts,   t_part = t[:, part*R : part*R + R]
//
// `group` is how many consecutive output columns one part owns before
// the next takes over -- a head's worth (128) where q/k/v interleave per
// head, the whole inner width where they are three flat blocks. The tile
// must not straddle a group (group % BN == 0), which the host checks; a
// tile that did would contract half its columns against the wrong part.
// The window is a strided view of t, so nothing is copied.
template <int BM, int BN, int SG>
static inline void dense_gemm_mma_lora_band_impl(
    const device VPIPE_ELT* t, const device VPIPE_ELT* Bf,
    device VPIPE_ELT* y, int R, int N, int M, int ldt, int group,
    int parts, uint3 tgid)
{
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  if (m0 >= M || n0 >= N) { return; }
  const int part = (n0 / group) % parts;
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tT(const_cast<device VPIPE_ELT*>(t) + part * R,
        dextents<int32_t, 2>(R, M), metal::array<int32_t, 2>{1, ldt});
  TX tB(const_cast<device VPIPE_ELT*>(Bf), dextents<int32_t, 2>(R, N));
  TX tY(y, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mT = tT.slice(0, m0);
  auto mB = tB.slice(0, n0);
  auto mY = tY.slice(n0, m0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mT), decltype(mB), VPIPE_ELT>();
  cT.load(mY);
  op.run(mT, mB, cT);
  cT.store(mY);
}

template <int BM, int BN, int SG>
static inline void dense_gemm_mma_lora_fused_band_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    const device VPIPE_ELT* t, const device VPIPE_ELT* Bf,
    device VPIPE_ELT* y, int K, int N, int M, int R, int ldt, int group,
    int parts, uint3 tgid)
{
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  if (m0 >= M || n0 >= N) { return; }
  const int part = (n0 / group) % parts;
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x),  dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W),  dextents<int32_t, 2>(K, N));
  TX tT(const_cast<device VPIPE_ELT*>(t) + part * R,
        dextents<int32_t, 2>(R, M), metal::array<int32_t, 2>{1, ldt});
  TX tB(const_cast<device VPIPE_ELT*>(Bf), dextents<int32_t, 2>(R, N));
  TX tY(y, dextents<int32_t, 2>(N, M));
  // The same two descriptors, in the same order, as the unbanded fused
  // tile -- see dense_gemm_mma_lora_fused_impl for why the BASE keeps the
  // plain mode and only the delta accumulates.
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
  auto mX = tX.slice(0, m0);
  auto mW = tW.slice(0, n0);
  auto mT = tT.slice(0, m0);
  auto mB = tB.slice(0, n0);
  auto mY = tY.slice(n0, m0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX), decltype(mW), VPIPE_ELT>();
  op.run(mX, mW, cT);
  op_acc.run(mT, mB, cT);
  cT.store(mY);
}

// Buffers 0-10 as DGVF (R is the PER-PART rank); 11: ldt 12: group
// 13: parts.
#define DGVFB(NAME, BM, BN, SG)                                          \
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
      const constant int& ldt [[buffer(11)]],                           \
      const constant int& group [[buffer(12)]],                         \
      const constant int& parts [[buffer(13)]],                         \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    (void)has_bias; (void)bias;                                          \
    dense_gemm_mma_lora_fused_band_impl<BM, BN, SG>(                     \
        x, W, t, Bf, y, K, N, M, R, ldt, group, parts, tgid);            \
  }
DGVFB(dense_gemm_mma_t_n128_lora_band_f16,     128, 128, 8)
DGVFB(dense_gemm_mma_t_n128x256_lora_band_f16, 128, 256, 8)

// The separate second GEMM, banded: 0: t, 1: B [N, R], 3: y, 4: R (the
// PER-PART rank), 5: N, 6: M, 9: ldt, 10: group, 11: parts. Slots 2, 7
// and 8 are unread, kept so the host binds it like the acc tiles.
#define DGVLB(NAME, BM, BN, SG)                                          \
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
      const constant int& ldt [[buffer(9)]],                            \
      const constant int& group [[buffer(10)]],                         \
      const constant int& parts [[buffer(11)]],                         \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    (void)has_bias; (void)bias; (void)scale;                             \
    dense_gemm_mma_lora_band_impl<BM, BN, SG>(                           \
        x, W, y, K, N, M, ldt, group, parts, tgid);                      \
  }
DGVLB(dense_gemm_mma_t_n128_acc_band_f16,     128, 128, 8)
DGVLB(dense_gemm_mma_t_n128x256_acc_band_f16, 128, 256, 8)

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

// THE CHUNK DEPTH AS A PARAMETER, which is the question attention asks.
//
// A flash attention's QK product contracts over head_dim -- 128, and not
// a number a kernel chooses -- and one threadgroup issues one such
// product per key block, back to back into a live accumulator. That is
// this loop's shape exactly, with KC = 128. Whether the matrix pipe
// sustains its int8 rate at that depth, or needs a longer reduction per
// call, is not something a standalone K=128 GEMM can answer: that one
// pays a threadgroup's setup and store for every 128 of contraction,
// where both this and attention pay it once for the whole loop.
template <int KC>
static inline void gemm_i8i8_kacc_impl(
    const device int8_t* xq, const device int8_t* wq,
    const device half* as, const device half* ws, device half* y,
    threadgroup int* Ys, int K, int N, int M, uint3 tgid, uint lid)
{
  constexpr int BM = 64, BN = 64, SG = 4;
  for (int e = (int)lid; e < BM * BN; e += SG * 32) { Ys[e] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  using TY = tensor<threadgroup int, dextents<int32_t, 2>, tensor_inline>;
  TY tY(Ys, dextents<int32_t, 2>(BN, BM));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  for (int k0 = 0; k0 < K; k0 += KC) {
    auto mX = tX.slice(k0, m0);
    auto mW = tW.slice(k0, n0);
    op.run(mX, mW, tY);
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

#define GI8_KACC(NAME, KC)                                               \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device int8_t* wq [[buffer(1)]],                             \
      const device half* as [[buffer(2)]],                               \
      const device half* ws [[buffer(3)]],                               \
      device half* y [[buffer(4)]],                                      \
      const constant int& K [[buffer(5)]],                              \
      const constant int& N [[buffer(6)]],                              \
      const constant int& M [[buffer(7)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ys[64 * 64];                                         \
    gemm_i8i8_kacc_impl<KC>(xq, wq, as, ws, y, Ys, K, N, M, tgid, lid);  \
  }

GI8_KACC(gemm_i8i8_sc_f16_n64_kacc_k128, 128)
GI8_KACC(gemm_i8i8_sc_f16_n64_kacc_k256, 256)
// 64-deep, to carry the chunk-depth curve down to where a per-64-group
// scale would have to live. FREE accumulation still, so what this
// isolates is the DEPTH: the group tax is the _g64 twin below.
GI8_KACC(gemm_i8i8_sc_f16_n64_kacc_k64, 64)
// ...and 32, one more step down the depth curve, for the g32 question.
GI8_KACC(gemm_i8i8_sc_f16_n64_kacc_k32, 32)

// THE f16 TWIN OF THE ABOVE, so the chunk-depth sweep measures the DTYPE
// and not the structure. Same 64x64 tile, same K chunking, same
// multiply_accumulate mode (which carries its own per-call tax -- see
// conv2d_mma's bias-fused note -- and both arms pay it), one store for
// the whole loop.
template <int KC>
static inline void dense_gemm_mma_kacc_impl(
    const device VPIPE_ELT* x, const device VPIPE_ELT* W,
    device VPIPE_ELT* y, int K, int N, int M, uint3 tgid)
{
  constexpr int BM = 64, BN = 64, SG = 4;
  using TX = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device VPIPE_ELT*>(x), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device VPIPE_ELT*>(W), dextents<int32_t, 2>(K, N));
  TX tY(y, dextents<int32_t, 2>(N, M));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), VPIPE_ELT>();
  for (auto it = cT.begin(); it != cT.end(); ++it) { *it = (VPIPE_ELT)0; }
  for (int k0 = 0; k0 < K; k0 += KC) {
    auto mX = tX.slice(k0, m0);
    auto mW = tW.slice(k0, n0);
    op.run(mX, mW, cT);
  }
  auto mY = tY.slice(n0, m0);
  cT.store(mY);
}

#define DGKACC(NAME, KC)                                                 \
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
    dense_gemm_mma_kacc_impl<KC>(x, W, y, K, N, M, tgid);                \
  }

DGKACC(dense_gemm_mma_kacc_k128_f16, 128)
DGKACC(dense_gemm_mma_kacc_k512_f16, 512)

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
// PER-GROUP SCALING, TEMPLATED ON THE GROUP WIDTH.
//
// The width is the chunk depth: one matmul2d per group, drained and
// scaled before the next. So "group size" and "how deep each matmul2d
// call is" are the same number here, and making the groups finer makes
// the calls shallower -- which is why the two costs cannot be separated
// on this kernel and why the kacc twins above exist to separate them.
template <int KC>
static inline void gemm_i8i8_gk_impl(
    const device int8_t* xq, const device int8_t* wq,
    const device VPIPE_ELT* as, const device VPIPE_ELT* ws,
    device VPIPE_ELT* y, threadgroup int* Ys, int K, int N, int M,
    const device VPIPE_ELT* bias, int has_bias,
    uint3 tgid, uint lid)
{
  constexpr int BM = 64, BN = 64, SG = 4;
  constexpr int EPT = (BM * BN) / (SG * 32);

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  using TY = tensor<threadgroup int, dextents<int32_t, 2>, tensor_inline>;
  TY tY(Ys, dextents<int32_t, 2>(BN, BM));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / KC;

  float facc[EPT] = {0.0f};
  for (int g = 0; g < G; ++g) {
    auto mX = tX.slice(g * KC, m0);
    auto mW = tW.slice(g * KC, n0);
    op.run(mX, mW, tY);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int t = 0; t < EPT; ++t) {
    const int e = (int)lid + t * (SG * 32);
    const int i = e / BN, j = e % BN;
    const int gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) {
      const float bv = has_bias != 0 ? (float)bias[gn] : 0.0f;
      y[(int64_t)gm * N + gn] = (VPIPE_ELT)(facc[t] + bv);
    }
  }
}

#define GI8_GK(NAME, KC)                                                 \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device int8_t* wq [[buffer(1)]],                             \
      const device VPIPE_ELT* as [[buffer(2)]],                          \
      const device VPIPE_ELT* ws [[buffer(3)]],                          \
      device VPIPE_ELT* y [[buffer(4)]],                                 \
      const constant int& K [[buffer(5)]],                              \
      const constant int& N [[buffer(6)]],                              \
      const constant int& M [[buffer(7)]],                              \
      const device VPIPE_ELT* bias [[buffer(8)]],                        \
      const constant int& has_bias [[buffer(9)]],                       \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ys[64 * 64];                                         \
    gemm_i8i8_gk_impl<KC>(xq, wq, as, ws, y, Ys, K, N, M, bias,          \
                          has_bias, tgid, lid);                          \
  }

GI8_GK(gemm_i8i8_sc_f16_n64_g512, 512)
// Finer groups, for the group-size sweep (gemm_i8.group_size). Not
// wired: what they are for is measuring what a finer group COSTS.
GI8_GK(gemm_i8i8_sc_f16_n64_g128, 128)
GI8_GK(gemm_i8i8_sc_f16_n64_g64, 64)
GI8_GK(gemm_i8i8_sc_f16_n64_g32, 32)

// SPLIT-K twin of _g512: the same per-512-group accumulation, cut across
// grid.z planes and left in f32 for splitk_fold_f32_f16 to sum.
//
// WHY IT IS THIS EASY, and why the deep-K case wanted it. _g512 already
// walks K as a sequence of independent 512-groups, each scaled by its own
// pair before it joins a float accumulator -- so a plane is nothing but a
// contiguous RANGE of that loop, and the fold is the rest of the sum. The
// float term for group g is computed identically whichever plane owns it,
// so the only difference from the single-op kernel is the ORDER of the G
// float adds: a reassociation, exactly the argument mma-splitk.h makes for
// the dense f32 planes, and a weaker requirement than that one because
// here the summands themselves are bit-identical either way.
//
// The split is by GROUP INDEX, not by K, which is what keeps every plane
// boundary on a 512 multiple for free -- a chunk width in elements would
// have had to be constrained to one.
//
// What it buys is what split-K always buys: a single-op reduction over a
// deep K leaves only (M/BM)*(N/BN) threadgroups walking one long serial
// contraction, and the matrix units stall with nothing else in flight.
// The int8 pipe is not exempt -- the prototype measured the ff-down at
// K=12288 running 18.5 TOPS against 22.5-23.6 at K=4096, the same deep-K
// cliff the f16 path has and the reason this twin was flagged as the
// obvious follow-up when i8 shipped.
//
// NO BIAS HERE, deliberately: a split writes S partial planes and the
// bias must be added ONCE. splitk_fold_bias_f32_f16 does it in the fold.
//   0:xq 1:wq 2:as 3:ws 4:planes (float [S, M, N]) 5:K 6:N 7:M 8:gpp
// grid (N/BN, M/BM, S) THREADGROUPS. `gpp` is groups per plane; the last
// plane takes the remainder, so S need not divide G.
kernel void gemm_i8i8_sc_f16_n64_g512_sk(
    const device int8_t*    xq [[buffer(0)]],
    const device int8_t*    wq [[buffer(1)]],
    const device VPIPE_ELT* as [[buffer(2)]],
    const device VPIPE_ELT* ws [[buffer(3)]],
    device float*           planes [[buffer(4)]],
    const constant int& K   [[buffer(5)]],
    const constant int& N   [[buffer(6)]],
    const constant int& M   [[buffer(7)]],
    const constant int& gpp [[buffer(8)]],
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
  const int G  = K / GI8_KC;
  const int p  = (int)tgid.z;
  const int g0 = p * gpp;
  const int g1 = min(G, g0 + gpp);

  float facc[EPT] = {0.0f};
  for (int g = g0; g < g1; ++g) {
    auto mX = tX.slice(g * GI8_KC, m0);
    auto mW = tW.slice(g * GI8_KC, n0);
    op.run(mX, mW, tY);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // A plane whose range is empty still writes: the fold sums S planes
  // unconditionally, and leaving one uninitialised is the kind of bug
  // that only shows up when S stops dividing G.
  for (int t = 0; t < EPT; ++t) {
    const int e = (int)lid + t * (SG * 32);
    const int i = e / BN, j = e % BN;
    const int gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) {
      planes[((int64_t)p * M + gm) * N + gn] = facc[t];
    }
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
// reproducibility requirements. The register-resident twins below
// (gemm_i8i8_gkrs_impl) are where a shift accumulate DOES pay: with the
// drain gone and the scales staged, the fixed-exponent form runs at the
// float twin's rate.
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


// REGISTER-RESIDENT PER-GROUP ACCUMULATION.
//
// The _g twins above pay their group tax in the DRAIN, not in the
// arithmetic: matmul2d writes each group's i32 partial to a threadgroup
// tile, a barrier, every thread reads its elements back and scales
// them, and a second barrier frees the tile for the next group. At
// KC=64 that round-trip runs eight times per 512 of contraction, and
// MEASURED (gemm_i8.group_size) g64 at 6.4 TOP/s against kacc-64's 18.3
// for the same matrix work -- below the f16 dense kernel.
//
// This twin never drains. The partial lands in a COOPERATIVE tensor --
// the i32 result of each element in the registers of the thread that
// owns it -- and is folded into a per-thread register accumulator
// right there, so between groups there is no threadgroup traffic and
// no barrier. The element -> (row, col) map is the layout's own
// (get_multidimensional_index), resolved once before the loop.
//
// SHIFT=true is the block-floating-point accumulate of _g512i: pow2
// scales carried as int8 EXPONENTS, an i32 accumulator with a tracked
// exponent, rounded right-shifts to align, one ldexp at the end.
// SHIFT=false is the float accumulate of _g512: f16 amax scales, one
// fma per element per group. Buffers as _g512i / _g512 respectively
// (0:xq 1:wq 2:sa[M,K/KC] 3:sw[N,K/KC] 4:y 5:K 6:N 7:M), no bias.
//
// MEASURED (gemm_i8.group_size, M5, TOP/s at 4096x12288x4096 /
// 4096x4096x12288): g64r 7.8 / 7.4 and g64ri 5.2 / 5.4 against the
// drained g64's 6.4 / 6.1 -- removing the drain bought almost nothing,
// and g512r (17.4) is SLOWER than the drained g512 (19.3). The cost
// had moved, not gone: two DEVICE scale loads per element per group
// are 64 loads per thread per group, and with ~160 live registers
// there is no occupancy to hide them behind. The staged twins below
// (gemm_i8i8_gkrs_impl) are the fix; this one is the measurement
// record of why they stage.
template <int KC, bool SHIFT, typename ST>
static inline void gemm_i8i8_gkr_impl(
    const device int8_t* xq, const device int8_t* wq,
    const device ST* sa, const device ST* sw,
    device VPIPE_ELT* y, int K, int N, int M, uint3 tgid)
{
  constexpr int BM = 64, BN = 64, SG = 4;
  // Elements per thread of a 64x64 i32 destination over 4 simdgroups.
  // The layout is the runtime's; gemm_i8.g64_register_acc probes that
  // it is exactly this (32 valid elements per thread, covering the
  // tile once) before anything trusts the kernel.
  constexpr int CAP = 32;

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / KC;
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();

  // Per-element scale rows, resolved once: sa[oa + g], sw[ow + g].
  // Coordinates come back (col, row) -- x is the N extent everywhere
  // in this file.
  int oa[CAP], ow[CAP];
  uint okm = 0u;
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    const int gn = n0 + (int)ids[0], gm = m0 + (int)ids[1];
    const bool ok = cP.is_valid_element((uint16_t)i) && gm < M && gn < N;
    okm |= ok ? (1u << i) : 0u;
    oa[i] = gm * G;
    ow[i] = gn * G;
  }

  int acc[CAP];
  short ae[CAP];
  float facc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { acc[i] = 0; ae[i] = -1000; facc[i] = 0.0f; }

  for (int g = 0; g < G; ++g) {
    auto mX = tX.slice(g * KC, m0);
    auto mW = tW.slice(g * KC, n0);
    op.run(mX, mW, cP);
#pragma clang loop unroll(full)
    for (int i = 0; i < CAP; ++i) {
      if ((okm >> i) & 1u) {
        int p = cP[(uint16_t)i];
        if constexpr (SHIFT) {
          const int eg = (int)sa[oa[i] + g] + (int)sw[ow[i] + g];
          const int d = eg - (int)ae[i];
          if (d > 0) {
            acc[i] = (d >= 31) ? 0 : ((acc[i] + (1 << (d - 1))) >> d);
            acc[i] += p;
            ae[i] = (short)eg;
          } else if (d < 0) {
            const int dd = -d;
            p = (dd >= 31) ? 0 : ((p + (1 << (dd - 1))) >> dd);
            acc[i] += p;
          } else {
            acc[i] += p;
          }
        } else {
          facc[i] += (float)p * (float)sa[oa[i] + g] * (float)sw[ow[i] + g];
        }
      }
    }
  }

#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    if ((okm >> i) & 1u) {
      const auto ids = cP.get_multidimensional_index((uint16_t)i);
      const int gn = n0 + (int)ids[0], gm = m0 + (int)ids[1];
      float v;
      if constexpr (SHIFT) {
        v = ldexp((float)acc[i], (int)ae[i]);
      } else {
        v = facc[i];
      }
      y[(int64_t)gm * N + gn] = (VPIPE_ELT)v;
    }
  }
}

#define GI8_GKR(NAME, KC, SHIFT, ST)                                     \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device int8_t* wq [[buffer(1)]],                             \
      const device ST* sa [[buffer(2)]],                                 \
      const device ST* sw [[buffer(3)]],                                 \
      device VPIPE_ELT* y [[buffer(4)]],                                 \
      const constant int& K [[buffer(5)]],                              \
      const constant int& N [[buffer(6)]],                              \
      const constant int& M [[buffer(7)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    gemm_i8i8_gkr_impl<KC, SHIFT, ST>(xq, wq, sa, sw, y, K, N, M, tgid); \
  }

// _r: float accumulate in registers (f16 scales); _ri: shift-aligned
// i32 accumulate in registers (int8 exponents).
GI8_GKR(gemm_i8i8_sc_f16_n64_g64r, 64, false, VPIPE_ELT)
GI8_GKR(gemm_i8i8_sc_f16_n64_g64ri, 64, true, char)
GI8_GKR(gemm_i8i8_sc_f16_n64_g128r, 128, false, VPIPE_ELT)
GI8_GKR(gemm_i8i8_sc_f16_n64_g128ri, 128, true, char)
GI8_GKR(gemm_i8i8_sc_f16_n64_g512r, 512, false, VPIPE_ELT)
GI8_GKR(gemm_i8i8_sc_f16_n64_g512ri, 512, true, char)

// THE COOPERATIVE-DESTINATION CEILING: the kacc loop (raw i32
// multiply_accumulate, one scale at the end) with the accumulator in a
// cooperative tensor instead of the threadgroup tile. No per-group work
// at all, so the gap to kacc at the same KC is what the register
// destination itself costs, and whatever a register twin below adds on
// top of THIS is its epilogue.
//
// MEASURED (gemm_i8.group_size, M5): kaccr-64 20.4 / 18.5 TOP/s against
// kacc-64's 18.4 / 17.6, kaccr-512 20.8 / 18.8 against 22.9 / 19.4. The
// register destination is not a cost at all at 64 -- it is CHEAPER than
// the tile, which pays a tgmem store per op -- and within 3-9% at 512.
template <int KC>
static inline void gemm_i8i8_kaccr_impl(
    const device int8_t* xq, const device int8_t* wq,
    const device half* as, const device half* ws, device half* y,
    int K, int N, int M, uint3 tgid)
{
  constexpr int BM = 64, BN = 64, SG = 4, CAP = 32;
  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { cP[(uint16_t)i] = 0; }
  for (int k0 = 0; k0 < K; k0 += KC) {
    auto mX = tX.slice(k0, m0);
    auto mW = tW.slice(k0, n0);
    op.run(mX, mW, cP);
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    const int gn = n0 + (int)ids[0], gm = m0 + (int)ids[1];
    if (cP.is_valid_element((uint16_t)i) && gm < M && gn < N) {
      const float v = (float)cP[(uint16_t)i] * (float)as[gm] * (float)ws[gn];
      y[(int64_t)gm * N + gn] = (half)v;
    }
  }
}

#define GI8_KACCR(NAME, KC)                                              \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device int8_t* wq [[buffer(1)]],                             \
      const device half* as [[buffer(2)]],                               \
      const device half* ws [[buffer(3)]],                               \
      device half* y [[buffer(4)]],                                      \
      const constant int& K [[buffer(5)]],                              \
      const constant int& N [[buffer(6)]],                              \
      const constant int& M [[buffer(7)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    gemm_i8i8_kaccr_impl<KC>(xq, wq, as, ws, y, K, N, M, tgid);          \
  }

GI8_KACCR(gemm_i8i8_sc_f16_n64_kaccr_k64, 64)
GI8_KACCR(gemm_i8i8_sc_f16_n64_kaccr_k512, 512)
GI8_KACCR(gemm_i8i8_sc_f16_n64_kaccr_k32, 32)
GI8_KACCR(gemm_i8i8_sc_f16_n64_kaccr_k128, 128)

// STAGED-SCALE register twins. The naive register twin above reads two
// device scales per element per group -- 64 loads per thread per group
// -- and MEASURED no better than the drain it replaced. This one stages
// a STRIP of groups' scales into threadgroup memory first (8 groups x
// (64 rows + 64 cols): one load per thread per group, one barrier per
// strip) and indexes them by the element's tile-local (row, col), which
// the probe shows is all 32 elements valid, so the per-element guard
// goes too: out-of-range rows and cols stage a zero scale and are never
// stored.
//
// AROW=true is the second half of the design: the ACTIVATION quantized
// per ROW (one scale per token over all of K, quant_f16_i8_row) and only
// the WEIGHT per (channel, group) -- which is what "int8-g64 weights"
// asks for, and halves the per-element work again: one staged scale, one
// fma. The row scale is applied once, at the store. Buffer 2 is then
// as[M] (or int8 exponents ea[M]).
//
// MEASURED (gemm_i8.group_size, M5, TOP/s at 4096x12288x4096 /
// 4096x4096x12288; the drained g64 is 6.4 / 6.1, kacc-64 18.4 / 17.6,
// and the f16 kacc twin at the same tile 11.2 / 4.4):
//
//   g64rs   17.4 / 16.9   float, both grouped      (2.7x the drain)
//   g64rsi   9.4 /  9.5   running shift, both grouped
//   w64rs   19.0 / 17.6   float, activation per row
//   w64rsi  10.1 / 10.2   running shift, activation per row
//   w64rsf  17.7 / 17.6   FIXED-exponent shift, activation per row
//   g512rs  21.3 / 19.3   float, both grouped -- above the drained
//                         g512's 19.3 / 14.7 that ships today
//   g32rs   14.0 / 13.6   one step finer: the depth-32 matmul is free
//                         (kaccr-32 19.9 / 18.5) and the epilogue,
//                         run twice as often, is the whole 20%
//   g128rs  19.3 / 18.5   one step coarser, 4% under its ceiling
//
// So a 64-wide group is viable once nothing drains: the float twins
// sit at 0.95-1.03x of the chunk-depth ceiling. The RUNNING shift is
// the slow half of the design -- its branchless align is ~12 integer
// ops per element where the float path is a convert and an fma -- and
// the fixed-exponent form (one pre-staged shift per element) is what
// closes that gap. All four shift twins are bit-exact against a CPU
// replica (gemm_i8.g64_register_acc); f32 quality at 256x512x1024 is
// 8.4e-3 for the amax float twins and 1.29-1.35e-2 for the pow2 shift
// twins, the ~0.6 bit per pow2 operand that _g512i's note predicts.
//
// FIXED=true (SHIFT && AROW only) is the cheaper alignment that the
// per-row activation makes possible: a column's exponent schedule is a
// pure function of ITS weight exponents, all known before the loop, so
// the accumulator sits at the column's FINAL exponent from the start
// and every partial takes one right-shift by a pre-staged amount --
// no tracked exponent, no accumulator shift, no max. Numerically it is
// the running scheme's end state reached directly: the running scheme
// shifts the accumulator down to the same exponent as soon as the
// largest group arrives, and the fixed one rounds each partial there
// instead of the running sum.
//
// SPLIT=true is the split-K twin: grid.z planes, each owning `gpp`
// consecutive groups (the last takes the remainder), left in f32 for
// splitk_fold_f32_f16 -- the same by-group-index split the drained
// _g512_sk makes, for the same reason (a plane is a contiguous range of
// this loop). No bias on that path: the fold adds it once.
template <int KC, bool SHIFT, bool AROW, bool FIXED, bool SPLIT,
          typename ST>
static inline void gemm_i8i8_gkrs_impl(
    const device int8_t* xq, const device int8_t* wq,
    const device ST* sa, const device ST* sw,
    device VPIPE_ELT* y, const device VPIPE_ELT* bias, int has_bias,
    device float* planes, int gpp,
    threadgroup int* Ss, threadgroup int* Ecol,
    int K, int N, int M, uint3 tgid, uint lid)
{
  static_assert(!FIXED || (SHIFT && AROW),
                "FIXED is the per-row-activation shift scheme");
  static_assert(!SPLIT || !SHIFT, "the split twin is the float one");
  constexpr int BM = 64, BN = 64, SG = 4, CAP = 32;
  constexpr int STRIP = 8;
  // Per-group staged values: the rows' scales, then the cols'; AROW
  // stages only the cols'.
  constexpr int SR = AROW ? 0 : BM;
  constexpr int SW = SR + BN;

  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TX tW(const_cast<device int8_t*>(wq), dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / KC;
  int gbeg = 0, gend = G;
  if constexpr (SPLIT) {
    gbeg = (int)tgid.z * gpp;
    gend = min(G, gbeg + gpp);
  }
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();

  uchar lr[CAP], lc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    lc[i] = (uchar)ids[0];
    lr[i] = (uchar)ids[1];
  }
  int acc[CAP];
  int ae[CAP];
  float facc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { acc[i] = 0; ae[i] = -1000; facc[i] = 0.0f; }

  if constexpr (FIXED) {
    // Each column's max exponent over ALL its groups: two threads per
    // column, one per group parity, then the pair's max.
    const int c = (int)lid & 63, par = (int)lid >> 6;
    const int gn = n0 + c;
    int em = -1000;
    if (gn < N) {
      for (int g = par; g < G; g += 2) {
        em = max(em, (int)sw[(int64_t)gn * G + g]);
      }
    }
    Ecol[(int)lid] = em;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 64) { Ecol[lid] = max(Ecol[lid], Ecol[lid + 64]); }
  }

  for (int g0 = gbeg; g0 < gend; g0 += STRIP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lid; e < STRIP * SW; e += SG * 32) {
      const int s = e / SW, r = e - s * SW;
      const int g = g0 + s;
      float fv = 0.0f;
      int iv = 0;
      if (g < gend) {
        if (r < SR) {
          const int gm = m0 + r;
          if (gm < M) {
            if constexpr (SHIFT) { iv = (int)sa[(int64_t)gm * G + g]; }
            else { fv = (float)sa[(int64_t)gm * G + g]; }
          }
        } else {
          const int gn = n0 + r - SR;
          if (gn < N) {
            if constexpr (FIXED) {
              // The pre-staged shift: down from this group's exponent
              // to the column's final one.
              iv = clamp(Ecol[r - SR] - (int)sw[(int64_t)gn * G + g],
                         0, 31);
            } else if constexpr (SHIFT) {
              iv = (int)sw[(int64_t)gn * G + g];
            } else {
              fv = (float)sw[(int64_t)gn * G + g];
            }
          }
        }
      }
      if constexpr (SHIFT) { Ss[e] = iv; } else { Ss[e] = as_type<int>(fv); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 0; s < STRIP; ++s) {
      const int g = g0 + s;
      if (g >= gend) { break; }
      auto mX = tX.slice(g * KC, m0);
      auto mW = tW.slice(g * KC, n0);
      op.run(mX, mW, cP);
      const threadgroup int* sr = Ss + s * SW;
      const threadgroup int* sc = sr + SR;
#pragma clang loop unroll(full)
      for (int i = 0; i < CAP; ++i) {
        const int p = cP[(uint16_t)i];
        if constexpr (FIXED) {
          const int s2 = sc[lc[i]];
          acc[i] += (p + (int)((1u << s2) >> 1)) >> s2;
        } else if constexpr (SHIFT) {
          // Branchless align: shift whichever side is finer down to the
          // coarser exponent, rounding half up. |acc| < 2^30 and |p| <
          // 2^21, so a clamped 31-bit shift is the "drop it" case.
          int eg = sc[lc[i]];
          if constexpr (!AROW) { eg += sr[lr[i]]; }
          const int d = eg - ae[i];
          const int s1 = clamp(d, 0, 31), s2 = clamp(-d, 0, 31);
          acc[i] = ((acc[i] + (int)((1u << s1) >> 1)) >> s1) +
                   ((p + (int)((1u << s2) >> 1)) >> s2);
          ae[i] = max(ae[i], eg);
        } else {
          float sv = as_type<float>(sc[lc[i]]);
          if constexpr (!AROW) { sv *= as_type<float>(sr[lr[i]]); }
          facc[i] += (float)p * sv;
        }
      }
    }
  }

#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const int gn = n0 + (int)lc[i], gm = m0 + (int)lr[i];
    if (gm < M && gn < N) {
      float v;
      if constexpr (FIXED) {
        v = ldexp((float)acc[i], Ecol[lc[i]] + (int)sa[gm]);
      } else if constexpr (SHIFT) {
        int e = ae[i];
        if constexpr (AROW) { e += (int)sa[gm]; }
        v = ldexp((float)acc[i], e);
      } else {
        v = facc[i];
        if constexpr (AROW) { v *= (float)sa[gm]; }
      }
      if constexpr (SPLIT) {
        // A plane whose range is empty still writes: the fold sums S
        // planes unconditionally.
        planes[((int64_t)tgid.z * M + gm) * N + gn] = v;
      } else {
        if (has_bias != 0) { v += (float)bias[gn]; }
        y[(int64_t)gm * N + gn] = (VPIPE_ELT)v;
      }
    }
  }
}

// Same buffer contract as _g512 (bias at 8/9, zero = not added), so
// I8GemmContext binds either family the same way.
#define GI8_GKRS(NAME, KC, SHIFT, AROW, FIXED, ST)                       \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device int8_t* wq [[buffer(1)]],                             \
      const device ST* sa [[buffer(2)]],                                 \
      const device ST* sw [[buffer(3)]],                                 \
      device VPIPE_ELT* y [[buffer(4)]],                                 \
      const constant int& K [[buffer(5)]],                              \
      const constant int& N [[buffer(6)]],                              \
      const constant int& M [[buffer(7)]],                              \
      const device VPIPE_ELT* bias [[buffer(8)]],                        \
      const constant int& has_bias [[buffer(9)]],                       \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ss[8 * 128];                                         \
    threadgroup int Ecol[128];                                           \
    gemm_i8i8_gkrs_impl<KC, SHIFT, AROW, FIXED, false, ST>(              \
        xq, wq, sa, sw, y, bias, has_bias, nullptr, 0, Ss, Ecol, K, N,   \
        M, tgid, lid);                                                   \
  }

// Same contract as _g512_sk: 4:planes (float [S, M, N]), 8:gpp, grid
// (N/64, M/64, S) threadgroups.
#define GI8_GKRS_SK(NAME, KC)                                            \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device int8_t* wq [[buffer(1)]],                             \
      const device VPIPE_ELT* sa [[buffer(2)]],                          \
      const device VPIPE_ELT* sw [[buffer(3)]],                          \
      device float* planes [[buffer(4)]],                                \
      const constant int& K [[buffer(5)]],                              \
      const constant int& N [[buffer(6)]],                              \
      const constant int& M [[buffer(7)]],                              \
      const constant int& gpp [[buffer(8)]],                            \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ss[8 * 128];                                         \
    threadgroup int Ecol[128];                                           \
    gemm_i8i8_gkrs_impl<KC, false, false, false, true, VPIPE_ELT>(       \
        xq, wq, sa, sw, nullptr, nullptr, 0, planes, gpp, Ss, Ecol, K,   \
        N, M, tgid, lid);                                                \
  }

// _rs: float, both operands grouped; _rsi: shift-aligned, both grouped;
// _w64rs / _w64rsi: weights grouped, activation per row; _w64rsf: the
// same with the fixed column exponent.
GI8_GKRS(gemm_i8i8_sc_f16_n64_g64rs, 64, false, false, false, VPIPE_ELT)
GI8_GKRS(gemm_i8i8_sc_f16_n64_g64rsi, 64, true, false, false, char)
GI8_GKRS(gemm_i8i8_sc_f16_n64_g512rs, 512, false, false, false, VPIPE_ELT)
GI8_GKRS(gemm_i8i8_sc_f16_n64_g512rsi, 512, true, false, false, char)
GI8_GKRS(gemm_i8i8_sc_f16_n64_w64rs, 64, false, true, false, VPIPE_ELT)
GI8_GKRS(gemm_i8i8_sc_f16_n64_w64rsi, 64, true, true, false, char)
GI8_GKRS(gemm_i8i8_sc_f16_n64_w64rsf, 64, true, true, true, char)
// THE SHIPPED PAIR: g64rs and its split-K twin are what I8GemmContext
// runs at its default group of 64.
GI8_GKRS_SK(gemm_i8i8_sc_f16_n64_g64rs_sk, 64)
// GROUP 32, the same design one step finer (VPIPE_I8_GROUP=32): each
// matmul2d is 32 deep and the epilogue runs twice as often per K.
GI8_GKRS(gemm_i8i8_sc_f16_n64_g32rs, 32, false, false, false, VPIPE_ELT)
GI8_GKRS_SK(gemm_i8i8_sc_f16_n64_g32rs_sk, 32)
// GROUP 128, one step coarser (VPIPE_I8_GROUP=128), completing the set
// 32 / 64 / 128 / 512 on one design.
GI8_GKRS(gemm_i8i8_sc_f16_n64_g128rs, 128, false, false, false, VPIPE_ELT)
GI8_GKRS_SK(gemm_i8i8_sc_f16_n64_g128rs_sk, 128)

// NATIVE AFFINE-WEIGHT INT8 GEMM: the activation's int8 code against the
// checkpoint's OWN codes -- w8 bytes or w4 nibbles, group 64, the MLX
// affine form w = s*q + b with q unsigned -- with the scale and the zero
// point folded into the register epilogue EXACTLY. No dequant scratch,
// no requant pass, and no second quantization of the weight.
//
// The matrix units have no int8 x uint8 form (uint8 x uint8 and uint8 x
// uint4 are what exist for unsigned codes), so the activation goes in
// as xq + 128 and the epilogue takes the offset back out:
//
//   P'[m,n,g] = sum_k (xq + 128) * q            the MMA, exact in i32
//   P         = P' - 128 * Q[n,g],  Q = sum_k q  affine_group_sums_*
//   y[m,n]    = sum_g  a[m,g] * s[n,g] * P  +  u[m,g] * b[n,g]
//               u = a * X,  X = sum_k xq       the quantizer's row sums
//
// P' and 128*Q are both under 2^23, so the subtraction is exact in
// int32 and what reaches the fma is the true code product. The u*b term
// is the zero point's contribution and it is summed per group in f32
// with the rest; deferring it to an f16 accumulate after the store
// would round a term as large as the output itself.
//
// MEASURED on the M5 at 4096x12288x4096, TOP/s, kernel only
// (gemm_i8.native_affine_kernel_arms):
//
//   ceilings, no epilogue:  i8xi8 22.2   u8xu8 19.8   u8xu4 17.6   i8xi4 18.8
//   g64rs (the requant kernel)         17.3
//   native w8:  per-group form (v1) 12.6, matmul form cm0 15.3, cm1 16.2
//   native w4:  v1 10.6, cm0 12.5, cm1 13.1
//   a signed code format would give: w8 16.9, w4 15.0
//
// Three things moved it from v1 to cm1: the zero-point term as one
// matmul instead of two loads and an fma per element per group;
// staging only what the mode reads (the unread u and b columns cost
// 15% on their own); and adding that matmul in registers on the
// strength of the layout probe, no tile. What remains is the matrix
// pipe: unsigned costs 11% against int8 x int8 and the 4-bit operand
// another 11%, and no epilogue gets that back.
//   0:xu[M,K] u8  1:codes  2:as[M,G]  3:xs[M,G] i16  4:ws[N,G]  5:wb[N,G]
//   6:wsum[N,G] i16  7:y (f32 planes for _sk)  8:K 9:N 10:M
//   11:bias 12:has_bias  (_sk: 11:gpp)  13:u[M,G]  14:c_hi 15:c_lo
//   grid as the g64rs kernels.
//
// UB_MMA=true moves the zero-point term out of the group loop. It is
// sum_g u[m,g] * b[n,g] -- a rank-G product, i.e. one 64x64xG matmul
// per threadgroup over U[M,G] (the quantizer's a*X, in the element
// type) and the checkpoint's OWN qbias tensor [N,G], run once after the
// loop into an f32 cooperative tensor and folded into facc. That leaves
// the per-group epilogue with three loads, an int add, a convert, a
// multiply and one fma. The -128*Q term cannot go the same way: it
// cancels the +128*Q that rides inside P', and an f16/bf16 operand
// would carry that cancellation at 11 (8) bits.
//
// CMODE=1 folds the -128*Q term into that matmul too, as sum_g a[m,g] *
// c[n,g] with c = -128*s*Q carried as an f16 hi/lo PAIR (c_hi = f16(c),
// c_lo = f16(c - c_hi), ~22 bits between them) so the cancellation
// against the +128*Q inside P' survives; the per-group epilogue is then
// exactly g64rs's: two loads, a multiply, a convert and an fma.
// Staged words per group for a mode: [a | u] rows, [s | b | c] cols in
// the per-group form; the matmul forms read a, s (and c under CMODE 0).
template <bool UB, int CM>
constexpr int gemm_u8q_sw()
{
  return (UB ? 1 : 2) * 64 + (UB ? (CM == 1 ? 1 : 2) : 3) * 64;
}

template <int BITS, bool SPLIT, int STRIP = 8, bool UB_MMA = true,
          int CMODE = 0>
static inline void gemm_u8q_impl(
    const device uint8_t* xu, const device uchar* codes,
    const device VPIPE_ELT* as, const device short* xs,
    const device VPIPE_ELT* ws, const device VPIPE_ELT* wb,
    const device short* wsum, const device VPIPE_ELT* uf,
    const device VPIPE_ELT* c_hi, const device VPIPE_ELT* c_lo,
    device VPIPE_ELT* y, const device VPIPE_ELT* bias, int has_bias,
    device float* planes, int gpp,
    threadgroup int* Ss, int K, int N, int M, uint3 tgid, uint lid)
{
  static_assert(CMODE == 0 || UB_MMA, "CMODE 1 rides the post-loop matmul");
  constexpr int BM = 64, BN = 64, SG = 4, CAP = 32, KC = 64;
  // Staged per group: [a | u] rows then [s | b | c] cols in the per-group
  // form; the matmul forms read only a, s (and c under CMODE 0).
  constexpr int NR = UB_MMA ? 1 : 2;
  constexpr int NC = UB_MMA ? (CMODE == 1 ? 1 : 2) : 3;
  constexpr int SW = gemm_u8q_sw<UB_MMA, CMODE>();
  static_assert(SW == NR * BM + NC * BN, "staging layout");
  using TX = tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline>;
  using WE = conditional_t<BITS == 4, uint4b_format, uint8_t>;
  using TW = tensor<device WE, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device uint8_t*>(xu), dextents<int32_t, 2>(K, M));
  // A 4-bit tensor's handle is a byte pointer and its extents count
  // ELEMENTS, so the packed row of K nibbles is K wide here.
  TW tW(const_cast<device uchar*>(codes), dextents<int32_t, 2>(K, N));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / KC;
  int gbeg = 0, gend = G;
  if constexpr (SPLIT) {
    gbeg = (int)tgid.z * gpp;
    gend = min(G, gbeg + gpp);
  }
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();

  uchar lr[CAP], lc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    lc[i] = (uchar)ids[0];
    lr[i] = (uchar)ids[1];
  }
  float facc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { facc[i] = 0.0f; }

  for (int g0 = gbeg; g0 < gend; g0 += STRIP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lid; e < STRIP * SW; e += SG * 32) {
      const int sidx = e / SW, r = e - sidx * SW;
      const int g = g0 + sidx;
      int v = 0;
      if (g < gend) {
        if (r < NR * BM) {
          const int gm = m0 + (r < BM ? r : r - BM);
          if (gm < M) {
            const float a = (float)as[(int64_t)gm * G + g];
            v = as_type<int>(r < BM
                ? a : a * (float)xs[(int64_t)gm * G + g]);
          }
        } else {
          const int rc = r - NR * BM;
          const int which = rc / BN;          // 0: s, 1: b or c, 2: c
          const int gn = n0 + (rc - which * BN);
          if (gn < N) {
            if (which == 0) {
              v = as_type<int>((float)ws[(int64_t)gn * G + g]);
            } else if (which == 1 && !UB_MMA) {
              v = as_type<int>((float)wb[(int64_t)gn * G + g]);
            } else {
              v = -128 * (int)wsum[(int64_t)gn * G + g];
            }
          }
        }
      }
      Ss[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int sidx = 0; sidx < STRIP; ++sidx) {
      const int g = g0 + sidx;
      if (g >= gend) { break; }
      auto mX = tX.slice(g * KC, m0);
      auto mW = tW.slice(g * KC, n0);
      op.run(mX, mW, cP);
      const threadgroup int* sa = Ss + sidx * SW;
      const threadgroup int* su = sa + BM;             // per-group form
      const threadgroup int* ss = sa + NR * BM;
      const threadgroup int* sb = ss + BN;             // per-group form
      const threadgroup int* sc = ss + (NC - 1) * BN;  // last column set
#pragma clang loop unroll(full)
      for (int i = 0; i < CAP; ++i) {
        const int p = CMODE == 1 ? cP[(uint16_t)i]
                                 : cP[(uint16_t)i] + sc[lc[i]];
        const float a = as_type<float>(sa[lr[i]]);
        if constexpr (UB_MMA) {
          facc[i] += a * as_type<float>(ss[lc[i]]) * (float)p;
        } else {
          const float u = as_type<float>(su[lr[i]]);
          facc[i] += a * as_type<float>(ss[lc[i]]) * (float)p +
                     u * as_type<float>(sb[lc[i]]);
        }
      }
    }
  }

  if constexpr (UB_MMA) {
    // The zero-point term as one matmul over ALL G groups, accumulated
    // into a cooperative f32 tensor and added in REGISTERS: the f16 ->
    // f32 destination of this tile has the int8 destination's element
    // layout (probed by gemm_i8.g64_register_acc, which fails if that
    // ever changes), so element i of both is the same (row, col). Under
    // a split the whole term belongs to plane 0 -- the fold is a plain
    // sum of the planes, and a slice from gbeg would contract to the
    // tensor's END, not to gend, which is how the first version
    // double-counted.
    if (!SPLIT || tgid.z == 0) {
      using TU = tensor<device VPIPE_ELT, dextents<int32_t, 2>,
                        tensor_inline>;
      TU tU(const_cast<device VPIPE_ELT*>(uf), dextents<int32_t, 2>(G, M));
      TU tB(const_cast<device VPIPE_ELT*>(wb), dextents<int32_t, 2>(G, N));
      constexpr auto descA = matmul2d_descriptor(
          BM, BN, static_cast<int>(dynamic_extent),
          /*transpose_left=*/false, /*transpose_right=*/true,
          /*relaxed_precision=*/false,
          matmul2d_descriptor::mode::multiply_accumulate);
      matmul2d<descA, execution_simdgroups<SG>> opA;
      auto mU = tU.slice(0, m0);
      auto mB = tB.slice(0, n0);
      auto cF = opA.template get_destination_cooperative_tensor<
          decltype(mU), decltype(mB), float>();
#pragma clang loop unroll(full)
      for (int i = 0; i < CAP; ++i) { cF[(uint16_t)i] = 0.0f; }
      opA.run(mU, mB, cF);
      if constexpr (CMODE == 1) {
        TU tA(const_cast<device VPIPE_ELT*>(as), dextents<int32_t, 2>(G, M));
        TU tH(const_cast<device VPIPE_ELT*>(c_hi),
              dextents<int32_t, 2>(G, N));
        TU tL(const_cast<device VPIPE_ELT*>(c_lo),
              dextents<int32_t, 2>(G, N));
        auto mA = tA.slice(0, m0);
        auto mH = tH.slice(0, n0);
        auto mL = tL.slice(0, n0);
        opA.run(mA, mH, cF);
        opA.run(mA, mL, cF);
      }
#pragma clang loop unroll(full)
      for (int i = 0; i < CAP; ++i) { facc[i] += cF[(uint16_t)i]; }
    }
  }

#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const int gn = n0 + (int)lc[i], gm = m0 + (int)lr[i];
    if (gm < M && gn < N) {
      float v = facc[i];
      if constexpr (SPLIT) {
        planes[((int64_t)tgid.z * M + gm) * N + gn] = v;
      } else {
        if (has_bias != 0) { v += (float)bias[gn]; }
        y[(int64_t)gm * N + gn] = (VPIPE_ELT)v;
      }
    }
  }
}

#define GI8_U8Q_ENTRY(NAME, BITS, STRIP, UB, CM)                         \
  kernel void NAME(                                                      \
      const device uint8_t* xu [[buffer(0)]],                            \
      const device uchar* codes [[buffer(1)]],                           \
      const device VPIPE_ELT* as [[buffer(2)]],                          \
      const device short* xs [[buffer(3)]],                              \
      const device VPIPE_ELT* ws [[buffer(4)]],                          \
      const device VPIPE_ELT* wb [[buffer(5)]],                          \
      const device short* wsum [[buffer(6)]],                            \
      device VPIPE_ELT* y [[buffer(7)]],                                 \
      const constant int& K [[buffer(8)]],                              \
      const constant int& N [[buffer(9)]],                              \
      const constant int& M [[buffer(10)]],                             \
      const device VPIPE_ELT* bias [[buffer(11)]],                       \
      const constant int& has_bias [[buffer(12)]],                      \
      const device VPIPE_ELT* uf [[buffer(13)]],                         \
      const device VPIPE_ELT* c_hi [[buffer(14)]],                       \
      const device VPIPE_ELT* c_lo [[buffer(15)]],                       \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ss[STRIP * gemm_u8q_sw<UB, CM>()];                   \
    gemm_u8q_impl<BITS, false, STRIP, UB, CM>(                           \
        xu, codes, as, xs, ws, wb, wsum, uf, c_hi, c_lo, y, bias,        \
        has_bias, nullptr, 0, Ss, K, N, M, tgid, lid);                   \
  }

#define GI8_U8Q_SK(NAME, BITS, CM)                                       \
  kernel void NAME(                                                      \
      const device uint8_t* xu [[buffer(0)]],                            \
      const device uchar* codes [[buffer(1)]],                           \
      const device VPIPE_ELT* as [[buffer(2)]],                          \
      const device short* xs [[buffer(3)]],                              \
      const device VPIPE_ELT* ws [[buffer(4)]],                          \
      const device VPIPE_ELT* wb [[buffer(5)]],                          \
      const device short* wsum [[buffer(6)]],                            \
      device float* planes [[buffer(7)]],                                \
      const constant int& K [[buffer(8)]],                              \
      const constant int& N [[buffer(9)]],                              \
      const constant int& M [[buffer(10)]],                             \
      const constant int& gpp [[buffer(11)]],                           \
      const device VPIPE_ELT* uf [[buffer(13)]],                         \
      const device VPIPE_ELT* c_hi [[buffer(14)]],                       \
      const device VPIPE_ELT* c_lo [[buffer(15)]],                       \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ss[8 * gemm_u8q_sw<true, CM>()];                     \
    gemm_u8q_impl<BITS, true, 8, true, CM>(                              \
        xu, codes, as, xs, ws, wb, wsum, uf, c_hi, c_lo, nullptr,        \
        nullptr, 0, planes, gpp, Ss, K, N, M, tgid, lid);                \
  }

// The two forms the context can select (VPIPE_I8_NATIVE_CMODE): the
// -128*Q term per group in integer (cm0), or folded into the post-loop
// matmul as an f16 hi/lo pair (cm1); each with its split-K twin.
GI8_U8Q_ENTRY(gemm_u8q_w8g64, 8, 8, true, 0)
GI8_U8Q_ENTRY(gemm_u8q_w4g64, 4, 8, true, 0)
GI8_U8Q_SK(gemm_u8q_w8g64_sk, 8, 0)
GI8_U8Q_SK(gemm_u8q_w4g64_sk, 4, 0)
GI8_U8Q_ENTRY(gemm_u8q_w8g64_cm, 8, 8, true, 1)
GI8_U8Q_ENTRY(gemm_u8q_w4g64_cm, 4, 8, true, 1)
GI8_U8Q_SK(gemm_u8q_w8g64_cm_sk, 8, 1)
GI8_U8Q_SK(gemm_u8q_w4g64_cm_sk, 4, 1)
// The per-group form of the zero-point term, and the smaller strips,
// for the A/B record.
GI8_U8Q_ENTRY(gemm_u8q_w8g64_v1, 8, 8, false, 0)
GI8_U8Q_ENTRY(gemm_u8q_w4g64_v1, 4, 8, false, 0)
GI8_U8Q_ENTRY(gemm_u8q_w4g64_s4, 4, 4, false, 0)
GI8_U8Q_ENTRY(gemm_u8q_w4g64_s2, 4, 2, false, 0)
GI8_U8Q_ENTRY(gemm_u8q_w8g64_s4, 8, 4, false, 0)
GI8_U8Q_ENTRY(gemm_u8q_w8g64_s2, 8, 2, false, 0)

// (c) WHAT A SIGNED CODE FORMAT WOULD GIVE: the same kernel over int8 x
// int8 / int8 x int4b with NO offset -- codes stored as q ^ 0x80 (bytes)
// or q ^ 0x8 (nibbles), which two's complement reads as q - 128 / q - 8,
// and the bias adjusted to b' = b + 128*s / 8*s. No -128*Q term, no u8
// offset, the fastest MMA. Measured here as a kernel; the format change
// it needs in the loaders is a separate decision.
template <int BITS, bool UB_MMA>
static inline void gemm_i8q_impl(
    const device int8_t* xq, const device uchar* codes,
    const device VPIPE_ELT* as, const device VPIPE_ELT* ws,
    const device VPIPE_ELT* wb, const device VPIPE_ELT* uf,
    device VPIPE_ELT* y, threadgroup int* Ss, int K, int N, int M,
    uint3 tgid, uint lid)
{
  constexpr int BM = 64, BN = 64, SG = 4, CAP = 32, KC = 64, STRIP = 8;
  constexpr int NR = UB_MMA ? 1 : 2, NC = UB_MMA ? 1 : 2;
  constexpr int SW = NR * BM + NC * BN;
  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  using WE = conditional_t<BITS == 4, int4b_format, int8_t>;
  // A packed format's handle is a byte pointer; int8's is its own.
  using WH = conditional_t<BITS == 4, uchar, int8_t>;
  using TW = tensor<device WE, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TW tW(reinterpret_cast<device WH*>(const_cast<device uchar*>(codes)),
        dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  const int G = K / KC;
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();
  uchar lr[CAP], lc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    lc[i] = (uchar)ids[0];
    lr[i] = (uchar)ids[1];
  }
  float facc[CAP];
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { facc[i] = 0.0f; }
  for (int g0 = 0; g0 < G; g0 += STRIP) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = (int)lid; e < STRIP * SW; e += SG * 32) {
      const int sidx = e / SW, r = e - sidx * SW;
      const int g = g0 + sidx;
      float v = 0.0f;
      if (g < G) {
        if (r < NR * BM) {
          const int gm = m0 + (r < BM ? r : r - BM);
          if (gm < M) {
            v = r < BM ? (float)as[(int64_t)gm * G + g]
                       : (float)uf[(int64_t)gm * G + g];
          }
        } else {
          const int rc = r - NR * BM;
          const int gn = n0 + (rc < BN ? rc : rc - BN);
          if (gn < N) {
            v = rc < BN ? (float)ws[(int64_t)gn * G + g]
                        : (float)wb[(int64_t)gn * G + g];
          }
        }
      }
      Ss[e] = as_type<int>(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int sidx = 0; sidx < STRIP; ++sidx) {
      const int g = g0 + sidx;
      if (g >= G) { break; }
      auto mX = tX.slice(g * KC, m0);
      auto mW = tW.slice(g * KC, n0);
      op.run(mX, mW, cP);
      const threadgroup int* sa = Ss + sidx * SW;
      const threadgroup int* su = sa + BM;
      const threadgroup int* ss = sa + NR * BM;
      const threadgroup int* sb = ss + BN;
#pragma clang loop unroll(full)
      for (int i = 0; i < CAP; ++i) {
        const float a = as_type<float>(sa[lr[i]]);
        facc[i] += a * as_type<float>(ss[lc[i]]) * (float)cP[(uint16_t)i];
        if constexpr (!UB_MMA) {
          facc[i] += as_type<float>(su[lr[i]]) * as_type<float>(sb[lc[i]]);
        }
      }
    }
  }
  if constexpr (UB_MMA) {
    using TU = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
    TU tU(const_cast<device VPIPE_ELT*>(uf), dextents<int32_t, 2>(G, M));
    TU tB(const_cast<device VPIPE_ELT*>(wb), dextents<int32_t, 2>(G, N));
    constexpr auto descA = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descA, execution_simdgroups<SG>> opA;
    auto mU = tU.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto cF = opA.template get_destination_cooperative_tensor<
        decltype(mU), decltype(mB), float>();
#pragma clang loop unroll(full)
    for (int i = 0; i < CAP; ++i) { cF[(uint16_t)i] = 0.0f; }
    opA.run(mU, mB, cF);
#pragma clang loop unroll(full)
    for (int i = 0; i < CAP; ++i) { facc[i] += cF[(uint16_t)i]; }
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const int gn = n0 + (int)lc[i], gm = m0 + (int)lr[i];
    if (gm < M && gn < N) {
      y[(int64_t)gm * N + gn] = (VPIPE_ELT)facc[i];
    }
  }
}

#define GI8_I8Q_ENTRY(NAME, BITS, UB)                                    \
  kernel void NAME(                                                      \
      const device int8_t* xq [[buffer(0)]],                             \
      const device uchar* codes [[buffer(1)]],                           \
      const device VPIPE_ELT* as [[buffer(2)]],                          \
      const device VPIPE_ELT* ws [[buffer(3)]],                          \
      const device VPIPE_ELT* wb [[buffer(4)]],                          \
      const device VPIPE_ELT* uf [[buffer(5)]],                          \
      device VPIPE_ELT* y [[buffer(6)]],                                 \
      const constant int& K [[buffer(7)]],                              \
      const constant int& N [[buffer(8)]],                              \
      const constant int& M [[buffer(9)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]],                       \
      uint lid [[thread_index_in_threadgroup]]) {                        \
    threadgroup int Ss[8 * (UB ? 128 : 256)];                            \
    gemm_i8q_impl<BITS, UB>(xq, codes, as, ws, wb, uf, y, Ss, K, N, M,   \
                            tgid, lid);                                  \
  }
GI8_I8Q_ENTRY(gemm_i8q_w8g64s, 8, true)
GI8_I8Q_ENTRY(gemm_i8q_w4g64s, 4, true)

// THE UNSIGNED MMA CEILINGS: the kaccr loop over uint8 x uint8 and
// uint8 x uint4, raw multiply_accumulate into a cooperative i32 tensor
// with no scales at all, so the gap between them is what the 4-bit
// operand costs the matrix pipe and the gap to kaccr-64 (int8 x int8)
// is what unsigned costs, if anything.
template <int BITS>
static inline void gemm_u8q_kaccr_impl(
    const device uint8_t* xu, const device uchar* codes, device half* y,
    int K, int N, int M, uint3 tgid)
{
  constexpr int BM = 64, BN = 64, SG = 4, CAP = 32, KC = 64;
  using TX = tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline>;
  using WE = conditional_t<BITS == 4, uint4b_format, uint8_t>;
  using TW = tensor<device WE, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device uint8_t*>(xu), dextents<int32_t, 2>(K, M));
  TW tW(const_cast<device uchar*>(codes), dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { cP[(uint16_t)i] = 0; }
  for (int k0 = 0; k0 < K; k0 += KC) {
    auto mX = tX.slice(k0, m0);
    auto mW = tW.slice(k0, n0);
    op.run(mX, mW, cP);
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    const int gn = n0 + (int)ids[0], gm = m0 + (int)ids[1];
    if (cP.is_valid_element((uint16_t)i) && gm < M && gn < N) {
      y[(int64_t)gm * N + gn] = (half)((float)cP[(uint16_t)i] * 1e-6f);
    }
  }
}

#define GI8_U8Q_KACCR(NAME, BITS)                                        \
  kernel void NAME(                                                      \
      const device uint8_t* xu [[buffer(0)]],                            \
      const device uchar* codes [[buffer(1)]],                           \
      device half* y [[buffer(2)]],                                      \
      const constant int& K [[buffer(3)]],                              \
      const constant int& N [[buffer(4)]],                              \
      const constant int& M [[buffer(5)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]]) {                     \
    gemm_u8q_kaccr_impl<BITS>(xu, codes, y, K, N, M, tgid);              \
  }
GI8_U8Q_KACCR(gemm_u8u8_kaccr_k64, 8)
GI8_U8Q_KACCR(gemm_u8u4_kaccr_k64, 4)

// ...and int8 x int4b, the signed 4-bit form's ceiling.
kernel void gemm_i8i4_kaccr_k64(
    const device int8_t* xq [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    device half* y [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  constexpr int BM = 64, BN = 64, SG = 4, CAP = 32, KC = 64;
  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  using TW = tensor<device int4b_format, dextents<int32_t, 2>, tensor_inline>;
  TX tX(const_cast<device int8_t*>(xq), dextents<int32_t, 2>(K, M));
  TW tW(const_cast<device uchar*>(codes), dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  auto mX0 = tX.slice(0, m0);
  auto mW0 = tW.slice(0, n0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mW0), int32_t>();
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) { cP[(uint16_t)i] = 0; }
  for (int k0 = 0; k0 < K; k0 += KC) {
    auto mX = tX.slice(k0, m0);
    auto mW = tW.slice(k0, n0);
    op.run(mX, mW, cP);
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cP.get_multidimensional_index((uint16_t)i);
    const int gn = n0 + (int)ids[0], gm = m0 + (int)ids[1];
    if (cP.is_valid_element((uint16_t)i) && gm < M && gn < N) {
      y[(int64_t)gm * N + gn] = (half)((float)cP[(uint16_t)i] * 1e-6f);
    }
  }
}

// The layout probe behind the register twins' CAP: one threadgroup of
// the same op writes its cooperative destination's capacity and, per
// thread and element, (valid, col, row). Never runs the matmul.
//   0:out[1 + 128*64*3] int; dispatch {128,1,1}, tg {128,1,1}
kernel void gemm_i8i8_coop_probe(
    device int* out [[buffer(0)]],
    uint lid [[thread_index_in_threadgroup]])
{
  constexpr int BM = 64, BN = 64, SG = 4, KC = 64;
  using TX = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;
  TX tX(reinterpret_cast<device int8_t*>(out), dextents<int32_t, 2>(KC, BM));
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, KC, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mX0 = tX.slice(0, 0);
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mX0), decltype(mX0), int32_t>();
  const int cap = (int)cP.get_capacity();
  if (lid == 0) { out[0] = cap; }
  for (int i = 0; i < 64; ++i) {
    int v = -1, c0 = -1, c1 = -1;
    if (i < cap) {
      v = cP.is_valid_element((uint16_t)i) ? 1 : 0;
      const auto ids = cP.get_multidimensional_index((uint16_t)i);
      c0 = (int)ids[0];
      c1 = (int)ids[1];
    }
    device int* o = out + 1 + ((int)lid * 64 + i) * 3;
    o[0] = v; o[1] = c0; o[2] = c1;
  }
  // And the layout of a half x half -> float destination of the same
  // tile, dynamic K: if it is the int8 one's, a second product can be
  // added in registers without a tile. Region 2 starts at 1 + 128*64*3.
  using TH = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
  TH tH(reinterpret_cast<device half*>(out), dextents<int32_t, 2>(64, BM));
  constexpr auto descF = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<descF, execution_simdgroups<SG>> opF;
  auto mH0 = tH.slice(0, 0);
  auto cF = opF.template get_destination_cooperative_tensor<
      decltype(mH0), decltype(mH0), float>();
  const int capf = (int)cF.get_capacity();
  device int* out2 = out + 1 + 128 * 64 * 3;
  if (lid == 0) { out2[0] = capf; }
  for (int i = 0; i < 64; ++i) {
    int v = -1, c0 = -1, c1 = -1;
    if (i < capf) {
      v = cF.is_valid_element((uint16_t)i) ? 1 : 0;
      const auto ids = cF.get_multidimensional_index((uint16_t)i);
      c0 = (int)ids[0];
      c1 = (int)ids[1];
    }
    device int* o = out2 + 1 + ((int)lid * 64 + i) * 3;
    o[0] = v; o[1] = c0; o[2] = c1;
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
DGV_STUB(gemm_i8i8_sc_f16_n64_g64r)
DGV_STUB(gemm_i8i8_sc_f16_n64_g64ri)
DGV_STUB(gemm_i8i8_sc_f16_n64_g128r)
DGV_STUB(gemm_i8i8_sc_f16_n64_g128ri)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512r)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512ri)
DGV_STUB(gemm_i8i8_coop_probe)
DGV_STUB(gemm_i8i8_sc_f16_n64_kaccr_k64)
DGV_STUB(gemm_i8i8_sc_f16_n64_kaccr_k512)
DGV_STUB(gemm_i8i8_sc_f16_n64_g64rs)
DGV_STUB(gemm_i8i8_sc_f16_n64_g64rsi)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512rs)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512rsi)
DGV_STUB(gemm_i8i8_sc_f16_n64_w64rs)
DGV_STUB(gemm_i8i8_sc_f16_n64_w64rsi)
DGV_STUB(gemm_i8i8_sc_f16_n64_w64rsf)
DGV_STUB(gemm_i8i8_sc_f16_n64_g64rs_sk)
DGV_STUB(gemm_i8i8_sc_f16_n64_g32rs)
DGV_STUB(gemm_i8i8_sc_f16_n64_g32rs_sk)
DGV_STUB(gemm_i8i8_sc_f16_n64_g128rs)
DGV_STUB(gemm_i8i8_sc_f16_n64_g128rs_sk)
DGV_STUB(gemm_i8i8_sc_f16_n64_kaccr_k128)
DGV_STUB(gemm_i8i8_sc_f16_n64_kacc_k32)
DGV_STUB(gemm_i8i8_sc_f16_n64_kaccr_k32)
DGV_STUB(gemm_i8i8_sc_f16_n64_g32)
DGV_STUB(gemm_i8i8_sc_f16_n64_g512_sk)
DGV_STUB(gemm_u8q_w8g64)
DGV_STUB(gemm_u8q_w4g64)
DGV_STUB(gemm_u8q_w8g64_sk)
DGV_STUB(gemm_u8q_w4g64_sk)
DGV_STUB(gemm_u8q_w8g64_v1)
DGV_STUB(gemm_u8q_w4g64_v1)
DGV_STUB(gemm_u8q_w8g64_cm)
DGV_STUB(gemm_u8q_w4g64_cm)
DGV_STUB(gemm_u8q_w8g64_cm_sk)
DGV_STUB(gemm_u8q_w4g64_cm_sk)
DGV_STUB(gemm_i8q_w8g64s)
DGV_STUB(gemm_i8q_w4g64s)
DGV_STUB(gemm_u8q_w4g64_s4)
DGV_STUB(gemm_u8q_w4g64_s2)
DGV_STUB(gemm_u8q_w8g64_s4)
DGV_STUB(gemm_u8q_w8g64_s2)
DGV_STUB(gemm_u8u8_kaccr_k64)
DGV_STUB(gemm_u8u4_kaccr_k64)
DGV_STUB(gemm_i8i4_kaccr_k64)
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
DGV_STUB(dense_gemm_mma_t_n128_lora_band_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_lora_band_f16)
DGV_STUB(dense_gemm_mma_t_n128_acc_band_f16)
DGV_STUB(dense_gemm_mma_t_n128x256_acc_band_f16)
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
