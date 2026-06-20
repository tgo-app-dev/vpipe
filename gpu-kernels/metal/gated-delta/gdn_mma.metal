// gdn_mma.metal -- batched matrix-core (M5+) matmul2d primitives + a fast
// triangular solve for a chunkwise Qwen3.5 gated-DeltaNet prefill path.
//
// PARKED (de-risk only, NOT wired into the model). The chunkwise gated-delta
// rule reduces to six small per-(head, chunk) matmuls (kkT, qkT, w@S^T,
// q@S^T, P@delta, delta^T@k) + a per-chunk triangular solve. Measurement
// (gdn_mma.bmm_correctness_and_throughput) showed the matmuls are fast
// (~51 ms/24L) but the solve dominates (~88-108 ms even blocked) because
// the per-head tiles are too small (64x64) for M5 matrix cores (~1.75
// TFLOP/s), so chunkwise total ~= the recurrent step (~158 ms) -- NOT a
// win. The shipping GDN prefill speedup is instead the recurrent ndv4
// kernel (qwen3_5_gated_delta.metal, 1.33x). These primitives are kept,
// validated, as reusable batched-matmul2d / fast-solve building blocks.
// Built only for the tensor-capable target (-std=metal4.0).
//
// One threadgroup per (BMxBN output tile, batch); SG simdgroups cooperate.
// Dispatch (threads): {ceil(N/BN)*tg, ceil(M/BM), Batch}, tg = SG*32.
// tgid.x -> N tile, tgid.y -> M tile, tgid.z -> batch. Per-batch base
// pointers are base + batch*stride{A,B,C}. Tails clamp via tensor extents.
//
// Three transpose modes (output[M,N] = opA(A)[M,K] @ opB(B)[K,N]):
//   NT  A[M,K] stored [M,K],  B[K,N] stored [N,K]   (kkT,qkT,w@S^T,q@S^T)
//   NN  A[M,K] stored [M,K],  B[K,N] stored [K,N]   (P@delta)
//   TN  A[M,K] stored [K,M],  B[K,N] stored [K,N]   (delta^T@k)
//
//   0:A 1:B 2:C  3:M 4:N 5:K  6:strideA 7:strideB 8:strideC (elements)

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#ifndef GDN_BM
#define GDN_BM 64
#endif
#ifndef GDN_BN
#define GDN_BN 64
#endif
#ifndef GDN_SG
#define GDN_SG 4
#endif

#if defined(__HAVE_TENSOR__)

// Batched matmul2d. TL/TR select the stored layout of A/B (see header).
template <bool TL, bool TR>
static inline void gdn_bmm_impl(
    const device VPIPE_ELT* A, const device VPIPE_ELT* B, device VPIPE_ELT* C,
    int M, int N, int K, int sA, int sB, int sC, uint3 tgid)
{
  using T2 = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  const int b = (int)tgid.z;
  const device VPIPE_ELT* Ab = A + (int64_t)b * sA;
  const device VPIPE_ELT* Bb = B + (int64_t)b * sB;
  device VPIPE_ELT* Cb = C + (int64_t)b * sC;

  // tensor_inline extents are (contiguous_extent, outer_extent).
  //   A: TL=false stored [M,K] -> (K,M); TL=true stored [K,M] -> (M,K)
  //   B: TR=false stored [K,N] -> (N,K); TR=true stored [N,K] -> (K,N)
  T2 tA(const_cast<device VPIPE_ELT*>(Ab),
        TL ? dextents<int32_t, 2>(M, K) : dextents<int32_t, 2>(K, M));
  T2 tB(const_cast<device VPIPE_ELT*>(Bb),
        TR ? dextents<int32_t, 2>(K, N) : dextents<int32_t, 2>(N, K));
  T2 tC(Cb, dextents<int32_t, 2>(N, M));

  constexpr auto desc = matmul2d_descriptor(
      GDN_BM, GDN_BN, static_cast<int>(dynamic_extent), TL, TR, false);
  matmul2d<desc, execution_simdgroups<GDN_SG>> op;

  const int m0 = (int)tgid.y * GDN_BM;
  const int n0 = (int)tgid.x * GDN_BN;
  // Slice to the (m0,n0) tile. The M index lives in dim1 of A when TL=false
  // (stored [M,K]) and dim0 when TL=true (stored [K,M]); symmetric for N/B.
  auto mA = TL ? tA.slice(m0, 0) : tA.slice(0, m0);
  auto mB = TR ? tB.slice(0, n0) : tB.slice(n0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mA), decltype(mB), VPIPE_ELT>();
  op.run(mA, mB, cT);
  auto mC = tC.slice(n0, m0);
  cT.store(mC);
}

#define GDN_BMM(NAME, TL, TR)                                              \
  kernel void NAME(                                                        \
      const device VPIPE_ELT* A [[buffer(0)]],                            \
      const device VPIPE_ELT* B [[buffer(1)]],                            \
      device VPIPE_ELT* C [[buffer(2)]],                                  \
      const constant int& M [[buffer(3)]],                                \
      const constant int& N [[buffer(4)]],                                \
      const constant int& K [[buffer(5)]],                                \
      const constant int& sA [[buffer(6)]],                               \
      const constant int& sB [[buffer(7)]],                               \
      const constant int& sC [[buffer(8)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]]) {                       \
    gdn_bmm_impl<TL, TR>(A, B, C, M, N, K, sA, sB, sC, tgid);             \
  }

GDN_BMM(gdn_bmm_nt_f16, false, true)
GDN_BMM(gdn_bmm_nn_f16, false, false)
GDN_BMM(gdn_bmm_tn_f16, true,  false)

// Fast per-(head,chunk) triangular solve (I+L)X=B, L[i,j] (j<i) =
// beta[i]*(G[i]/G[j])*kkT[i,j] (unit diagonal). One threadgroup per batch
// builds the strictly-lower L ONCE into threadgroup memory, then each
// thread forward-substitutes ONE rhs column entirely in registers -- so
// the whole substitution has a single barrier (after the L build) instead
// of the parked build_L_solve's per-row barrier + per-col-block redundant
// L rebuild. bt<=64 so L fits 64*64 f32 = 16KB tg; the X column lives in
// GDN_BT_MAX registers. kkT is the compute-elt matmul output; G/beta/B/X
// are f32 for the recurrence's numerical stability.
//   0:kkT[batch,bt,bt] 1:G_cum[batch,bt] 2:beta[batch,bt]
//   3:B[batch,bt,rhs_w] 4:X(out)[batch,bt,rhs_w]  5:bt 6:rhs_w
// Dispatch: grid {rhs_w_rounded, 1, batch}, threadgroup {rhs_w_rounded,1,1}.
#define GDN_BT_MAX 64

kernel void gdn_solve_f16(
    const device VPIPE_ELT* kkT   [[buffer(0)]],
    const device float*     G_cum [[buffer(1)]],
    const device float*     beta  [[buffer(2)]],
    const device float*     B     [[buffer(3)]],
    device float*           X     [[buffer(4)]],
    const constant int&     bt    [[buffer(5)]],
    const constant int&     rhs_w [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpit [[thread_position_in_threadgroup]])
{
  const int b = (int)tgid.z;
  const device VPIPE_ELT* kkT_b = kkT + (int64_t)b * bt * bt;
  const device float* G_b = G_cum + (int64_t)b * bt;
  const device float* beta_b = beta + (int64_t)b * bt;
  const device float* B_b = B + (int64_t)b * bt * rhs_w;
  device float* X_b = X + (int64_t)b * bt * rhs_w;

  // Threadgroup size == rhs_w (one thread per rhs column); use it as the
  // cooperative stride for the L build.
  const int lid = (int)tpit.x;
  threadgroup float Lt[GDN_BT_MAX * GDN_BT_MAX];
  // Build strictly-lower L cooperatively (unit diagonal handled by the
  // substitution seeding X[i]=B[i]).
  for (int idx = lid; idx < bt * bt; idx += rhs_w) {
    const int i = idx / bt, j = idx % bt;
    Lt[idx] = (j < i)
        ? beta_b[i] * (G_b[i] / G_b[j]) * (float)kkT_b[i * bt + j]
        : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int c = lid;
  if (c >= rhs_w) { return; }
  float xr[GDN_BT_MAX];
  for (int i = 0; i < bt; ++i) {
    float acc = B_b[i * rhs_w + c];
    for (int j = 0; j < i; ++j) { acc -= Lt[i * bt + j] * xr[j]; }
    xr[i] = acc;
  }
  for (int i = 0; i < bt; ++i) { X_b[i * rhs_w + c] = xr[i]; }
}

#else
#define GDN_BMM_STUB(NAME)                                                 \
  kernel void NAME(device VPIPE_ELT* C [[buffer(2)]],                      \
                   uint tid [[thread_position_in_grid]]) {                 \
    if (tid == 0) { C[0] = (VPIPE_ELT)0; }                                 \
  }
GDN_BMM_STUB(gdn_bmm_nt_f16)
GDN_BMM_STUB(gdn_bmm_nn_f16)
GDN_BMM_STUB(gdn_bmm_tn_f16)
#endif
