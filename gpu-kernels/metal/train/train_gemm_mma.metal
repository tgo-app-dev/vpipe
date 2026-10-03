// train_gemm_mma.metal -- the matrix-core (M5+) twins of train_gemm's
// batched products, via MetalPerformancePrimitives matmul2d.
//
// The attention backward's GEMM form (attn_bwd.metal, attn_bwd_pds; and
// attn_bwd_pds_mma below, the same pass on the matrix cores) leaves three
// products per head once P^T and dS^T are stored:
//
//   dV[kL, D] += P^T[kL, qL] dO[qL, D]      NN, into the f32 accumulator
//   dK[kL, D] += dS^T[kL, qL] Q[qL, D]      NN, likewise
//   dQ[qL, D]  = (dS^T)^T K[kL, D]          TN, into the storage type
//
// Every operand is a whole row-major plane here (lda == its inner extent),
// so the tensors need no strides: the pitch IS the extent, and matmul2d
// reads its contraction depth from the tensors themselves. The same
// TrainGemmParams as the steel kernels; alpha must be 1 (the kernel that
// wrote dS^T already scaled it), and accumulate is the entry's, not the
// parameter's.
//
//   0:A 1:B 2:C (storage type) 3:C (f32) 4:TrainGemmParams
//   grid (threads) {ceil(N/BN) * SG*32, ceil(M/BM), batch}, tg {SG*32,1,1}

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

struct TrainGemmParams {
  int   M;
  int   N;
  int   K;
  int   lda;
  int   ldb;
  int   ldc;
  float alpha;
  int   accumulate;
  long  a_bstride;
  long  b_bstride;
  long  c_bstride;
};

#if defined(__HAVE_TENSOR__)

// TA: A stored [K][M] (else [M][K]); TB: B stored [N][K] (else [K][N]).
// F32: C is the f32 accumulator, and the tile is ADDED to it.
template <int BM, int BN, int SG, bool TA, bool TB, bool F32>
static inline void train_gemm_mma_impl(
    const device VPIPE_ELT* A, const device VPIPE_ELT* B,
    device VPIPE_ELT* Ct, device float* Cf,
    const constant TrainGemmParams& p, uint3 tgid)
{
  const int M = p.M, N = p.N, K = p.K;
  const int m0 = (int)tgid.y * BM;
  const int n0 = (int)tgid.x * BN;
  // A tile whose ORIGIN is past the extent is outside the contract (see
  // dense_gemm_mma_impl); a partial one is clamped by the extents.
  if (m0 >= M || n0 >= N) { return; }
  const long z = (long)tgid.z;
  A += z * p.a_bstride;
  B += z * p.b_bstride;

  using TE = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  // Extents are (inner, outer).
  TE tA(const_cast<device VPIPE_ELT*>(A),
        TA ? dextents<int32_t, 2>(M, K) : dextents<int32_t, 2>(K, M));
  TE tB(const_cast<device VPIPE_ELT*>(B),
        TB ? dextents<int32_t, 2>(K, N) : dextents<int32_t, 2>(N, K));

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent),
      /*transpose_left=*/TA, /*transpose_right=*/TB,
      /*relaxed_precision=*/false,
      F32 ? matmul2d_descriptor::mode::multiply_accumulate
          : matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  auto mA = TA ? tA.slice(m0, 0) : tA.slice(0, m0);
  auto mB = TB ? tB.slice(0, n0) : tB.slice(n0, 0);
  if constexpr (F32) {
    using TF = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    TF tC(Cf + z * p.c_bstride, dextents<int32_t, 2>(N, M));
    auto mC = tC.slice(n0, m0);
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), float>();
    cT.load(mC);
    op.run(mA, mB, cT);
    cT.store(mC);
  } else {
    TE tC(Ct + z * p.c_bstride, dextents<int32_t, 2>(N, M));
    auto mC = tC.slice(n0, m0);
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), VPIPE_ELT>();
    op.run(mA, mB, cT);
    cT.store(mC);
  }
}

#define VPIPE_TRAIN_GEMM_MMA(NAME, BM, BN, SG, TA, TB, F32)              \
  kernel void NAME(                                                      \
      const device VPIPE_ELT* A [[buffer(0)]],                           \
      const device VPIPE_ELT* B [[buffer(1)]],                           \
      device VPIPE_ELT* Ct [[buffer(2)]],                                \
      device float* Cf [[buffer(3)]],                                    \
      const constant TrainGemmParams& p [[buffer(4)]],                   \
      uint3 tgid [[threadgroup_position_in_grid]])                       \
  {                                                                      \
    train_gemm_mma_impl<BM, BN, SG, TA, TB, F32>(A, B, Ct, Cf, p, tgid); \
  }

// N is the head dim (128) for all three, so one 128-wide tile spans it
// and the grid runs down M.
VPIPE_TRAIN_GEMM_MMA(train_gemm_mma_nn_f32_64x128, 64, 128, 4, false,
                     false, true)
VPIPE_TRAIN_GEMM_MMA(train_gemm_mma_tn_64x128, 64, 128, 4, true, false,
                     false)

// ---- the attention backward's P^T / dS^T pass, on the matrix cores ------

// attn_bwd.metal's AttnBwdParams, verbatim.
struct AttnBwdParams {
  int   H;
  int   qL;
  int   kL;
  float scale;
  int   qL_off;
  int   causal;
  int   T;
  int   q_row0;
  long  q_hstride;
  long  k_hstride;
  long  acc_hstride;
  int   h0;
  int   ldp;
  long  p_hstride;
};

// attn_bwd_pds, one [BM keys x BN queries] tile per threadgroup instead of
// a key block walking every query block: S^T = K Q^T and dP^T = V dO^T
// are two matmul2d runs over the head dim, and P^T / scale * dS^T are
// formed in the accumulators and stored. The grid covers every tile of
// the plane, so masked entries are written as zero, as the GEMMs need.
//
//   0:Q (at the segment's row 0) 1:K 2:V 3:dO (as Q) 4:P^T 5:dS^T
//   6:LSE2 7:Dsum 8:AttnBwdParams
//   grid (threads) {ceil(qL/BN) * SG*32, ceil(kL/BM), heads}, tg {SG*32}
template <int BM, int BN, int SG>
static inline void attn_bwd_pds_mma_impl(
    const device VPIPE_ELT* Q, const device VPIPE_ELT* K,
    const device VPIPE_ELT* V, const device VPIPE_ELT* dO,
    device VPIPE_ELT* Pt, device VPIPE_ELT* dSt,
    const device float* LSE2, const device float* Dsum,
    const constant AttnBwdParams& p, uint3 tgid)
{
  constexpr int D = 128;
  constexpr int CAP = BM * BN / (SG * 32);
  const int hl = (int)tgid.z;
  const int h = p.h0 + hl;
  const int k0 = (int)tgid.y * BM;
  const int q0 = (int)tgid.x * BN;
  if (k0 >= p.kL || q0 >= p.qL) { return; }

  using TE = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TE tK(const_cast<device VPIPE_ELT*>(K) + (long)h * p.k_hstride,
        dextents<int32_t, 2>(D, p.kL));
  TE tV(const_cast<device VPIPE_ELT*>(V) + (long)h * p.k_hstride,
        dextents<int32_t, 2>(D, p.kL));
  TE tQ(const_cast<device VPIPE_ELT*>(Q) + (long)h * p.q_hstride,
        dextents<int32_t, 2>(D, p.qL));
  TE tO(const_cast<device VPIPE_ELT*>(dO) + (long)h * p.q_hstride,
        dextents<int32_t, 2>(D, p.qL));

  // Keys are the rows (left, [kL][D]); queries the columns (right, stored
  // [qL][D], so transposed).
  constexpr auto desc = matmul2d_descriptor(
      BM, BN, D, /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SG>> op;

  auto mK = tK.slice(0, k0);
  auto mV = tV.slice(0, k0);
  auto mQ = tQ.slice(0, q0);
  auto mO = tO.slice(0, q0);
  auto cS = op.template get_destination_cooperative_tensor<
      decltype(mK), decltype(mQ), float>();
  auto cP = op.template get_destination_cooperative_tensor<
      decltype(mV), decltype(mO), float>();
  op.run(mK, mQ, cS);
  op.run(mV, mO, cP);

  const float scale2 = p.scale * 1.44269504088896340736f;
  const long rix0 = (long)h * p.T + p.q_row0;
  device VPIPE_ELT* P0 = Pt + (long)hl * p.p_hstride;
  device VPIPE_ELT* S0 = dSt + (long)hl * p.p_hstride;
#pragma clang loop unroll(full)
  for (int i = 0; i < CAP; ++i) {
    const auto ids = cS.get_multidimensional_index((uint16_t)i);
    const int q = q0 + (int)ids[0];
    const int kr = k0 + (int)ids[1];
    if (q >= p.qL || kr >= p.kL) { continue; }
    const bool ok = p.causal == 0 || q + p.qL_off >= kr;
    const float pv =
        ok ? fast::exp2(cS[(uint16_t)i] * scale2 - LSE2[rix0 + q]) : 0.0f;
    const float ds = pv * (cP[(uint16_t)i] - Dsum[rix0 + q]) * p.scale;
    const long o = (long)kr * p.ldp + q;
    P0[o] = (VPIPE_ELT)pv;
    S0[o] = (VPIPE_ELT)ds;
  }
}

kernel void attn_bwd_pds_mma_64x64(
    const device VPIPE_ELT* Q [[buffer(0)]],
    const device VPIPE_ELT* K [[buffer(1)]],
    const device VPIPE_ELT* V [[buffer(2)]],
    const device VPIPE_ELT* dO [[buffer(3)]],
    device VPIPE_ELT* Pt [[buffer(4)]],
    device VPIPE_ELT* dSt [[buffer(5)]],
    const device float* LSE2 [[buffer(6)]],
    const device float* Dsum [[buffer(7)]],
    const constant AttnBwdParams& p [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  attn_bwd_pds_mma_impl<64, 64, 4>(Q, K, V, dO, Pt, dSt, LSE2, Dsum, p,
                                   tgid);
}

#endif  // __HAVE_TENSOR__
