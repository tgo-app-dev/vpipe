// affine_qmm_mma.metal -- matrix-core (M5+) 4-bit affine quantized GEMM
// via Metal 4 MetalPerformancePrimitives matmul2d. Drop-in replacement
// for affine_qmm_steel's affine_qmm_steel_w4g64 (same buffer layout +
// semantics): y[m,n] = sum_k x[m,k] * (scale[n,g]*q[n,k] + bias[n,g]),
// i.e. x @ dequant(W)^T, W row-major [N,K] 4-bit packed (group 64).
//
// Each threadgroup computes one BMxBN output tile cooperatively with
// DG_SG simdgroups. The K loop dequantizes a [BN x BK] weight tile + an
// [BM x BK] activation tile into threadgroup memory, then accumulates the
// tile product on the hardware matrix units (matmul2d multiply_accumulate
// into an f32 cooperative_tensor -- matching the steel BlockMMA's f32
// accumulation, so greedy output stays token-exact with the steel path).
// The dequant formula is byte-identical to the steel QuantizedBlockLoader.
//
// Built ONLY for the tensor-capable target (-std=metal4.0); the loader
// binds it only when a runtime GPU-capability check reports matrix-core
// support, so older GPUs keep the steel path (same source, peak on both).
//
//   0:w(uint32) 1:scales 2:biases 3:x 4:y  5:K 6:N 7:M
// dispatch (threads): {ceil(N/BN)*tg, ceil(M/BM), 1}, tg {DG_SG*32,1,1}.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT half          // unsuffixed = f16; _bf16 build sets bfloat
#endif

#ifndef DG_BM
#define DG_BM 64
#endif
#ifndef DG_BN
#define DG_BN 64
#endif
#ifndef DG_BK
#define DG_BK 32          // K step; group_size(64) is a multiple of it
#endif
#ifndef DG_SG
#define DG_SG 4
#endif
#define DG_GROUP 64
#define DG_THREADS (DG_SG * 32)

#if defined(__HAVE_TENSOR__)

kernel void affine_qmm_mma_w4g64(
    const device uint32_t*  w      [[buffer(0)]],
    const device VPIPE_ELT* scales [[buffer(1)]],
    const device VPIPE_ELT* biases [[buffer(2)]],
    const device VPIPE_ELT* x      [[buffer(3)]],
    device VPIPE_ELT*       y      [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  threadgroup VPIPE_ELT Xs[DG_BM * DG_BK];   // activation tile [BM,BK]
  threadgroup VPIPE_ELT Ws[DG_BN * DG_BK];   // dequant weight tile [BN,BK]
  threadgroup float     Ysf[DG_BM * DG_BN];  // f32 output tile [BM,BN]

  const int m0 = (int)tgid.y * DG_BM;
  const int n0 = (int)tgid.x * DG_BN;
  const int Kw = K >> 1;          // bytes per weight row (4-bit: K/2)
  const int Kg = K / DG_GROUP;    // scale/bias entries per row
  const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);

  // NT matmul over the staged threadgroup tiles: y[BM,BN] = Xtile @ Wtile^T,
  // accumulating in f32 across the K loop.
  constexpr auto desc = matmul2d_descriptor(
      DG_BM, DG_BN, DG_BK,
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<DG_SG>> op;
  using TT = tensor<threadgroup VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  auto cT = op.get_destination_cooperative_tensor<TT, TT, float>();
  for (auto it = cT.begin(); it != cT.end(); ++it) { *it = 0.0f; }

  for (int k0 = 0; k0 < K; k0 += DG_BK) {
    // Stage activation tile: Xs[i,j] = x[m0+i, k0+j] (0 for M tail rows).
    for (int e = (int)lid; e < DG_BM * DG_BK; e += DG_THREADS) {
      const int i = e / DG_BK, j = e % DG_BK;
      const int mm = m0 + i;
      Xs[e] = (mm < M) ? x[(int64_t)mm * K + (k0 + j)] : (VPIPE_ELT)0;
    }
    // Stage dequantized weight tile: Ws[i,j] = scale*q + bias for
    // (n0+i, k0+j) (0 for N tail rows).
    for (int e = (int)lid; e < DG_BN * DG_BK; e += DG_THREADS) {
      const int i = e / DG_BK, j = e % DG_BK;
      const int nn = n0 + i;
      if (nn < N) {
        const int k = k0 + j;
        const uint8_t byte = wb[(int64_t)nn * Kw + (k >> 1)];
        const uint qv = (k & 1) ? (uint)(byte >> 4) : (uint)(byte & 0x0f);
        const int g = nn * Kg + (k / DG_GROUP);
        const float s = (float)scales[g];
        const float b = (float)biases[g];
        Ws[e] = (VPIPE_ELT)(s * (float)qv + b);
      } else {
        Ws[e] = (VPIPE_ELT)0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TT tX(Xs, dextents<int32_t, 2>(DG_BK, DG_BM));
    TT tW(Ws, dextents<int32_t, 2>(DG_BK, DG_BN));
    op.run(tX, tW, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store the f32 tile to threadgroup scratch (cT.store maps the
  // cooperative layout), then cast to VPIPE_ELT and write the in-bounds
  // rows/cols to y. One round of f32->elt rounding, matching the steel
  // path's single store-rounding off its f32 accumulator.
  using TS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
  TS tYs(Ysf, dextents<int32_t, 2>(DG_BN, DG_BM));
  cT.store(tYs);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = (int)lid; e < DG_BM * DG_BN; e += DG_THREADS) {
    const int i = e / DG_BN, j = e % DG_BN;   // Ysf row-major [BM,BN]
    const int mm = m0 + i, nn = n0 + j;
    if (mm < M && nn < N) { y[(int64_t)mm * N + nn] = (VPIPE_ELT)Ysf[e]; }
  }
}

// ---------------------------------------------------------------------------
// PROTOTYPE (perf de-risk only, not wired into production): isolate whether
// reading the weight as int8 (half the bytes of f16) can beat f16 on the NAX
// matmul2d path at the MOSS codec's M. Both kernels share IDENTICAL tiling +
// matmul (only the weight staging differs), so proto_i8 - proto_f16 is the pure
// int8-read+dequant cost. proto_i8 uses a symmetric per-group scale (no bias,
// one multiply) -- the cheap MXINT8-style dequant. Not correctness-verified
// (perf shape only). 0:w 1:x 2:y 3:K 4:N 5:M (f16); i8 inserts scales at 1.
kernel void proto_gemm_mma_f16(
    const device VPIPE_ELT* w [[buffer(0)]],   // f16 weight [N,K]
    const device VPIPE_ELT* x [[buffer(1)]],
    device VPIPE_ELT*       y [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& N [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  threadgroup VPIPE_ELT Xs[DG_BM * DG_BK];
  threadgroup VPIPE_ELT Ws[DG_BN * DG_BK];
  threadgroup float     Ysf[DG_BM * DG_BN];
  const int m0 = (int)tgid.y * DG_BM, n0 = (int)tgid.x * DG_BN;
  constexpr auto desc = matmul2d_descriptor(
      DG_BM, DG_BN, DG_BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<DG_SG>> op;
  using TT = tensor<threadgroup VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  auto cT = op.get_destination_cooperative_tensor<TT, TT, float>();
  for (auto it = cT.begin(); it != cT.end(); ++it) { *it = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += DG_BK) {
    for (int e = (int)lid; e < DG_BM * DG_BK; e += DG_THREADS) {
      const int i = e / DG_BK, j = e % DG_BK, mm = m0 + i;
      Xs[e] = (mm < M) ? x[(int64_t)mm * K + (k0 + j)] : (VPIPE_ELT)0;
    }
    for (int e = (int)lid; e < DG_BN * DG_BK; e += DG_THREADS) {
      const int i = e / DG_BK, j = e % DG_BK, nn = n0 + i;
      Ws[e] = (nn < N) ? w[(int64_t)nn * K + (k0 + j)] : (VPIPE_ELT)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TT tX(Xs, dextents<int32_t, 2>(DG_BK, DG_BM));
    TT tW(Ws, dextents<int32_t, 2>(DG_BK, DG_BN));
    op.run(tX, tW, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  using TS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
  TS tYs(Ysf, dextents<int32_t, 2>(DG_BN, DG_BM));
  cT.store(tYs);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = (int)lid; e < DG_BM * DG_BN; e += DG_THREADS) {
    const int i = e / DG_BN, j = e % DG_BN, mm = m0 + i, nn = n0 + j;
    if (mm < M && nn < N) { y[(int64_t)mm * N + nn] = (VPIPE_ELT)Ysf[e]; }
  }
}

kernel void proto_gemm_mma_i8(
    const device uint32_t*  w      [[buffer(0)]],   // int8 weight [N,K]
    const device VPIPE_ELT* scales [[buffer(1)]],   // per-group [N,K/32]
    const device VPIPE_ELT* x      [[buffer(2)]],
    device VPIPE_ELT*       y      [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  constexpr int GROUP8 = 32;
  threadgroup VPIPE_ELT Xs[DG_BM * DG_BK];
  threadgroup VPIPE_ELT Ws[DG_BN * DG_BK];
  threadgroup float     Ysf[DG_BM * DG_BN];
  const int m0 = (int)tgid.y * DG_BM, n0 = (int)tgid.x * DG_BN;
  const int Kg = K / GROUP8;
  const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
  constexpr auto desc = matmul2d_descriptor(
      DG_BM, DG_BN, DG_BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<DG_SG>> op;
  using TT = tensor<threadgroup VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  auto cT = op.get_destination_cooperative_tensor<TT, TT, float>();
  for (auto it = cT.begin(); it != cT.end(); ++it) { *it = 0.0f; }
  for (int k0 = 0; k0 < K; k0 += DG_BK) {
    for (int e = (int)lid; e < DG_BM * DG_BK; e += DG_THREADS) {
      const int i = e / DG_BK, j = e % DG_BK, mm = m0 + i;
      Xs[e] = (mm < M) ? x[(int64_t)mm * K + (k0 + j)] : (VPIPE_ELT)0;
    }
    for (int e = (int)lid; e < DG_BN * DG_BK; e += DG_THREADS) {
      const int i = e / DG_BK, j = e % DG_BK, nn = n0 + i;
      if (nn < N) {
        const int k = k0 + j;
        const uint qv = (uint)wb[(int64_t)nn * K + k];   // 1 byte
        const float s = (float)scales[nn * Kg + (k / GROUP8)];
        Ws[e] = (VPIPE_ELT)(s * (float)qv);              // symmetric, no bias
      } else {
        Ws[e] = (VPIPE_ELT)0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TT tX(Xs, dextents<int32_t, 2>(DG_BK, DG_BM));
    TT tW(Ws, dextents<int32_t, 2>(DG_BK, DG_BN));
    op.run(tX, tW, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  using TS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
  TS tYs(Ysf, dextents<int32_t, 2>(DG_BN, DG_BM));
  cT.store(tYs);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = (int)lid; e < DG_BM * DG_BN; e += DG_THREADS) {
    const int i = e / DG_BN, j = e % DG_BN, mm = m0 + i, nn = n0 + j;
    if (mm < M && nn < N) { y[(int64_t)mm * N + nn] = (VPIPE_ELT)Ysf[e]; }
  }
}

#else
kernel void proto_gemm_mma_f16(device VPIPE_ELT* y [[buffer(2)]],
                               uint tid [[thread_position_in_grid]])
{ if (tid == 0) { y[0] = (VPIPE_ELT)0; } }
kernel void proto_gemm_mma_i8(device VPIPE_ELT* y [[buffer(3)]],
                              uint tid [[thread_position_in_grid]])
{ if (tid == 0) { y[0] = (VPIPE_ELT)0; } }
kernel void affine_qmm_mma_w4g64(device VPIPE_ELT* y [[buffer(4)]],
                                 uint tid [[thread_position_in_grid]])
{
  if (tid == 0) { y[0] = (VPIPE_ELT)0; }
}
#endif
