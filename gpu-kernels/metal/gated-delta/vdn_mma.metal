// vdn_mma.metal -- the VDN branch's per-frame STATISTICS on the matrix
// cores (M5+), via Metal 4 MetalPerformancePrimitives matmul2d.
//
// The stage this replaces is the branch's largest arithmetic by some
// way: two [d, S] x [S, d] products per (frame, head), 69.2 GFLOP a
// block at generation geometry (37 frames of 510 tokens, 56 heads,
// d = 128). MEASURED on an M5, the fp32 tiled kernel it stands beside
// runs that at 0.90 TFLOP/s and this batched primitive at 6.87
// (metal_vdn_branch.batched_matmul2d_probe, which is the probe that
// decided this was worth writing rather than assumed).
//
// The inputs are NOT the operands the maths names, and that is what
// makes this affordable:
//
//   A[f,h] = sum_s beta_s k_si k_sj      B[f,h] = sum_s beta_s v_si k_sj
//
// A matmul2d contracts two tensors and has nowhere to put a per-row
// weight, so beta is folded into the FEATURES as its square root -- half
// on each side, by the kernel that was writing them anyway
// (vdn_temporal_act_f32). Then A = Ks^T Ks and B = Vs^T Ks with
// Ks = sqrt(beta) k, Vs = sqrt(beta) v: no third tensor, no extra pass,
// and A exactly symmetric for having one operand on both sides.
//
// STRIDED, AND THAT IS THE WHOLE PORT. The features are [F*S, H, d], so
// one (frame, head)'s slab is S rows of d at a pitch of H*d -- and a
// tensor_inline's row pitch IS its contiguous extent. Declaring that
// extent as the PITCH and slicing to the head's columns gives exactly
// the strided view, with the tile covering d = BM columns from the slice
// origin. Nothing is copied and no layout changes; the alternative was a
// head-major features tensor, which every other stage would have had to
// be re-indexed for.
//
//   0:k_scaled [F*S,H,d] 1:v_scaled 2:A out [F*H,d,d] 3:B out
//   4:S 5:H 6:d
// Dispatch: {SG*32, 1, F*H}, threadgroup {SG*32, 1, 1}. One threadgroup
// per (frame, head): d = 128 is exactly one BMxBN tile, so there is no
// tile grid to walk and the batch is the whole of the parallelism.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT bfloat
#endif

// The branch's head_dim, which is the tile: VDN's linear_head_dim is 128
// on the released checkpoint and the host refuses the route otherwise,
// so the tile is not a tuning knob here -- it is the shape.
#ifndef VDN_MM_TILE
#define VDN_MM_TILE 128
#endif
#ifndef VDN_MM_SG
#define VDN_MM_SG 8
#endif

#if defined(__HAVE_TENSOR__)

kernel void vdn_frame_stats_mma(
    const device VPIPE_ELT* ks [[buffer(0)]],
    const device VPIPE_ELT* vs [[buffer(1)]],
    device float*           A  [[buffer(2)]],
    device float*           B  [[buffer(3)]],
    constant int&           S  [[buffer(4)]],
    constant int&           H  [[buffer(5)]],
    constant int&           d  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int fh = (int)tgid.z;
  const int f = fh / H, h = fh - (fh / H) * H;
  const int pitch = H * d;
  const int64_t base = (int64_t)f * (int64_t)S * (int64_t)pitch;

  using TE = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  using TF = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
  // (contiguous, outer) = (pitch, S): S rows of the WHOLE per-token
  // width, which is what makes the slice below a strided view of one
  // head rather than a copy of it.
  TE tK(const_cast<device VPIPE_ELT*>(ks) + base,
        dextents<int32_t, 2>(pitch, S));
  TE tV(const_cast<device VPIPE_ELT*>(vs) + base,
        dextents<int32_t, 2>(pitch, S));
  TF tA(A + (int64_t)fh * d * d, dextents<int32_t, 2>(d, d));
  TF tB(B + (int64_t)fh * d * d, dextents<int32_t, 2>(d, d));

  // TN: the left operand is stored contraction-major ([S, d], so [K, M])
  // and the right one likewise ([S, d] = [K, N]). K is the tensors' own
  // outer extent, which is S.
  constexpr auto desc = matmul2d_descriptor(
      VDN_MM_TILE, VDN_MM_TILE, static_cast<int>(dynamic_extent),
      /*transpose_left=*/true, /*transpose_right=*/false,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<VDN_MM_SG>> op;

  auto mK = tK.slice(h * d, 0);
  auto mV = tV.slice(h * d, 0);
  // A and B are computed one after the other rather than side by side:
  // two live [128, 128] f32 cooperative tensors are 128 KB across the
  // threadgroup, and the second one's registers are not wanted while the
  // first is still accumulating.
  {
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(mK), decltype(mK), float>();
    op.run(mK, mK, cT);
    auto mA = tA.slice(0, 0);
    cT.store(mA);
  }
  {
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(mV), decltype(mK), float>();
    op.run(mV, mK, cT);
    auto mB = tB.slice(0, 0);
    cT.store(mB);
  }
}

// The READOUT's contraction: out[s, v] = sum_k q[s, h, k] state[f, h, v, k],
// a [S, d] x [d, d] per (frame, head).
//
// A GEMM the fp32 kernel was already running as one -- a threadgroup per
// eight tokens, so a 64 KB state came back once for every eight rows
// that share it. Here a tile owns 128 rows, which reads it 16x less, and
// the arithmetic lands on the matrix units on the way.
//
// THE CONTRACTION IS THE STRIDED AXIS HERE, which the statistics' was
// not, and that is the one thing to get right: q's head channels are a
// 128-wide window inside a row of H*d, so the descriptor takes a STATIC
// K rather than reading the tensor's own inner extent (which is the
// pitch). Same arrangement as the split-K dense GEMM, for the same
// reason -- the extent describes the memory, the descriptor the maths.
//
// The RMSNorm, the norm weight and the gate are NOT here: they need a
// whole 128-wide row and a reduction across it, which a cooperative
// tensor does not hand over, so they follow in vdn_readout_norm_f32
// over a tensor still in cache.
//
//   0:q [F*S,H,d] 1:state [F,H,d,d] 2:out [rows,H,d] 3:S 4:H 5:d
//   6:out_base (the tile's first FRAME, since q is addressed globally
//   and the destination tile-locally)
// Dispatch: {SG*32 * 1, ceil(S/TILE), F*H}, threadgroup {SG*32, 1, 1}.
kernel void vdn_readout_mma(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* state [[buffer(1)]],
    device VPIPE_ELT*       out   [[buffer(2)]],
    constant int&           S     [[buffer(3)]],
    constant int&           H     [[buffer(4)]],
    constant int&           d     [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int fh = (int)tgid.z;
  const int f = fh / H, h = fh - (fh / H) * H;
  const int m0 = (int)tgid.y * VDN_MM_TILE;
  if (m0 >= S) { return; }
  const int pitch = H * d;
  const int64_t qbase = (int64_t)f * (int64_t)S * (int64_t)pitch;

  using TE = tensor<device VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  TE tQ(const_cast<device VPIPE_ELT*>(q) + qbase,
        dextents<int32_t, 2>(pitch, S));
  TE tS(const_cast<device VPIPE_ELT*>(state) + (int64_t)fh * d * d,
        dextents<int32_t, 2>(d, d));
  TE tO(out + qbase, dextents<int32_t, 2>(pitch, S));

  constexpr auto desc = matmul2d_descriptor(
      VDN_MM_TILE, VDN_MM_TILE, VDN_MM_TILE,
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false);
  matmul2d<desc, execution_simdgroups<VDN_MM_SG>> op;

  auto mQ = tQ.slice(h * d, m0);
  auto mS = tS.slice(0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mQ), decltype(mS), VPIPE_ELT>();
  op.run(mQ, mS, cT);
  auto mO = tO.slice(h * d, m0);
  cT.store(mO);
}

#else
// No tensor ops for this target: a stub, so the metallib still builds.
// The host gates the route on supports_matrix_cores() and never binds
// this on a pre-M5 GPU -- but an unvalidated function is a silent no-op,
// so the stub writes nothing rather than pretending.
kernel void vdn_frame_stats_mma(device float* A [[buffer(2)]],
                                uint t [[thread_position_in_grid]])
{ if (t == 0) { A[0] = A[0]; } }
kernel void vdn_readout_mma(device VPIPE_ELT* out [[buffer(2)]],
                            uint t [[thread_position_in_grid]])
{ if (t == 0) { out[0] = out[0]; } }
#endif
