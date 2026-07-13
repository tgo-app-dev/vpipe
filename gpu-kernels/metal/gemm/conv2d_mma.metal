// conv2d_mma.metal -- matrix-core (M5+) 3x3 conv2d that runs the 9*Cin
// reduction on the Metal 4 matmul2d units WITHOUT materializing an im2col
// column matrix in DRAM. Serves the Krea-2 (Qwen-Image) VAE, whose single-frame
// 3x3 convs are otherwise im2col (a [H*W, 9*Cin] DRAM scratch) -> dense_gemm.
// Here each threadgroup gathers its output tile's 3x3 neighbourhood straight
// into threadgroup memory (the im2col stays on-chip) and feeds the hardware
// matmul2d -- saving the ~9x im2col DRAM round-trip at the VAE's high
// resolutions.
//
//   out[oy, ox, oc] = sum_{ky,kx,ci} in[oy*S + ky - 1, ox*S + kx - 1, ci]
//                                    * W[oc, (ky*3+kx)*Cin + ci]          (pad 1)
// Activation NHWC [H,W,Cin]; weights the SAME [Cout, 9*Cin] (ky,kx,ci)-flattened
// layout the im2col path already builds (no relayout); dest NHWO [OH,OW,Cout].
// A threadgroup owns a BM-output-pixel x BN-output-channel tile; the K loop
// walks the 9*Cin contraction in BK chunks, staging a dequant-free activation
// tile (with explicit zero-padding at the image border) + a weight tile into
// threadgroup memory and accumulating on the matrix units in f32 (one store
// rounding, matching the im2col+steel/matmul2d path).
//
//   0:in[H,W,Cin] 1:W[Cout,9*Cin] 2:out[OH,OW,Cout] 3:H 4:W 5:Cin 6:Cout 7:OH
//   8:OW.  dispatch (threads): {ceil(Cout/BN)*tg, ceil(OH*OW/BM), 1},
//   threadgroup {tg,1,1}, tg = SG*32. tgid.x -> channel tile, .y -> pixel tile.
//
// Built ONLY for the tensor-capable target (-std=metal4.0). STATUS: correct
// (bit-identical to im2col + matmul2d -- see conv2d_mma.matches_im2col and
// krea2_vae.decode_conv2d_vs_im2col) but ~1.7x SLOWER than im2col + dense_gemm_
// mma on M5: its high UMA bandwidth makes the im2col DRAM round-trip cheap while
// this kernel re-gathers each tile's 3x3 halo per K-chunk (redundant reads +
// index math + barriers). So the VAE keeps im2col by default and only binds this
// under VPIPE_KREA2_CONV2D=1.
//
// The MPP convolution2d HARDWARE op (below) SUPERSEDES both: its multi-tile
// halo/offset semantics -- once undocumented and unverified -- are now probe-
// established (conv2d_mma.hw_op_* tests): set_offsets = dest-tile origin in
// dest pixel coords, the op gathers the source window itself from the full
// device NHWC activation INCLUDING the zero-filled "same" border halo, output
// channels tile via weight/dest slices, and the descriptor's compile-time
// source_dimensions can be RUNTIME-patched through the detail __run entry so
// one kernel serves every image size. Bit-identical to im2col + matmul2d and
// ~1.9-3.0x FASTER at the VAE decoder shapes (skips the [H*W, 9*Cin] im2col
// DRAM round-trip; ~12-13 TFLOP/s). See conv2d_hw_3x3_s1_f16.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#ifndef VPIPE_ELT
#define VPIPE_ELT half          // unsuffixed = f16; _bf16 build sets bfloat
#endif

#ifndef CV_BM
#define CV_BM 64
#endif
#ifndef CV_BN
#define CV_BN 64
#endif
#ifndef CV_BK
#define CV_BK 32
#endif
#ifndef CV_SG
#define CV_SG 4
#endif
#define CV_THREADS (CV_SG * 32)

#if defined(__HAVE_TENSOR__)

template <int STRIDE>
static inline void conv2d_mma_impl(
    const device VPIPE_ELT* inp, const device VPIPE_ELT* wt,
    device VPIPE_ELT* out, int H, int W, int Cin, int Cout, int OH, int OW,
    threadgroup VPIPE_ELT* Xs, threadgroup VPIPE_ELT* Ws,
    threadgroup float* Ysf, uint3 tgid, uint lid)
{
  const int K = 9 * Cin;                     // contraction (taps * channels)
  const int M = OH * OW;                      // flattened output pixels
  const int m0 = (int)tgid.y * CV_BM;         // pixel-tile base
  const int n0 = (int)tgid.x * CV_BN;         // channel-tile base

  constexpr auto desc = matmul2d_descriptor(
      CV_BM, CV_BN, CV_BK,
      /*transpose_left=*/false, /*transpose_right=*/true,
      /*relaxed_precision=*/false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<CV_SG>> op;
  using TT = tensor<threadgroup VPIPE_ELT, dextents<int32_t, 2>, tensor_inline>;
  auto cT = op.get_destination_cooperative_tensor<TT, TT, float>();
  for (auto it = cT.begin(); it != cT.end(); ++it) { *it = 0.0f; }

  for (int k0 = 0; k0 < K; k0 += CV_BK) {
    // Stage activation tile: Xs[i,j] = im2col row for output pixel (m0+i),
    // column (k0+j) -> the (ky,kx,ci) sample, zero when the 3x3 neighbour is
    // outside the image (standard pad-1 zero padding).
    for (int e = (int)lid; e < CV_BM * CV_BK; e += CV_THREADS) {
      const int i = e / CV_BK, j = e % CV_BK;
      const int p = m0 + i, k = k0 + j;
      VPIPE_ELT v = (VPIPE_ELT)0;
      // Guard k < K: K = 9*Cin need not be a multiple of BK, so the last chunk
      // over-reads -- those columns stage 0 (and their weights below too).
      if (p < M && k < K) {
        const int oy = p / OW, ox = p % OW;
        const int tap = k / Cin, ci = k - tap * Cin;   // (ky,kx,ci)
        const int ky = tap / 3, kx = tap - ky * 3;
        const int iy = oy * STRIDE + ky - 1;
        const int ix = ox * STRIDE + kx - 1;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
          v = inp[((int64_t)iy * W + ix) * Cin + ci];
        }
      }
      Xs[e] = v;
    }
    // Stage weight tile: Ws[i,j] = W[n0+i, k0+j] ([Cout, 9*Cin] row-major).
    for (int e = (int)lid; e < CV_BN * CV_BK; e += CV_THREADS) {
      const int i = e / CV_BK, j = e % CV_BK;
      const int nn = n0 + i, k = k0 + j;
      Ws[e] = (nn < Cout && k < K) ? wt[(int64_t)nn * K + k] : (VPIPE_ELT)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TT tX(Xs, dextents<int32_t, 2>(CV_BK, CV_BM));
    TT tW(Ws, dextents<int32_t, 2>(CV_BK, CV_BN));
    op.run(tX, tW, cT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  using TS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
  TS tYs(Ysf, dextents<int32_t, 2>(CV_BN, CV_BM));
  cT.store(tYs);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = (int)lid; e < CV_BM * CV_BN; e += CV_THREADS) {
    const int i = e / CV_BN, j = e % CV_BN;   // Ysf row-major [BM,BN]
    const int p = m0 + i, nn = n0 + j;
    if (p < M && nn < Cout) { out[(int64_t)p * Cout + nn] = (VPIPE_ELT)Ysf[e]; }
  }
}

kernel void conv2d_mma_3x3_s1_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& H    [[buffer(3)]],
    const constant int& W    [[buffer(4)]],
    const constant int& Cin  [[buffer(5)]],
    const constant int& Cout [[buffer(6)]],
    const constant int& OH   [[buffer(7)]],
    const constant int& OW   [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  threadgroup VPIPE_ELT Xs[CV_BM * CV_BK];
  threadgroup VPIPE_ELT Ws[CV_BN * CV_BK];
  threadgroup float     Ysf[CV_BM * CV_BN];
  conv2d_mma_impl<1>(inp, wt, out, H, W, Cin, Cout, OH, OW, Xs, Ws, Ysf, tgid,
                     lid);
}

kernel void conv2d_mma_3x3_s2_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& H    [[buffer(3)]],
    const constant int& W    [[buffer(4)]],
    const constant int& Cin  [[buffer(5)]],
    const constant int& Cout [[buffer(6)]],
    const constant int& OH   [[buffer(7)]],
    const constant int& OW   [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
  threadgroup VPIPE_ELT Xs[CV_BM * CV_BK];
  threadgroup VPIPE_ELT Ws[CV_BN * CV_BK];
  threadgroup float     Ysf[CV_BM * CV_BN];
  conv2d_mma_impl<2>(inp, wt, out, H, W, Cin, Cout, OH, OW, Xs, Ws, Ysf, tgid,
                     lid);
}

// ---------------------------------------------------------------------
// MPP convolution2d HARDWARE-OP PROBE (conv2d_hw.probe). The mpp
// tensor_ops convolution2d op takes the FULL activation as a device
// NHWC tensor and positions each threadgroup's work via set_offsets --
// i.e. the op itself gathers the input window INCLUDING the halo
// (out-of-bounds rows/cols against the activation extents), which is
// exactly the gather-free mode the tgmem-staged kernel above lacks.
// The descriptor has NO padding field and the offset's coordinate
// space is undocumented, so this probe makes the interpretation a
// RUNTIME knob (off_mode) and an oracle test reports which verifies:
//   off_mode 0: set_offsets((ox0, oy0))          -- dest-tile origin
//   off_mode 1: set_offsets((ox0-1, oy0-1))      -- source origin w/ pad
//   off_mode 2/3: the (y, x)-swapped twins of 0/1
// Fixed probe geometry (compile-time: the descriptor is a template
// parameter): 16x16xCIN -> 16x16xCOUT, 3x3, stride 1, pad 1, tile
// CV_TW x CV_TH per threadgroup, cooperative destination stored by US
// at the tile's true position (so ONLY the source read depends on the
// offset). Weights HWIO [3,3,CIN,COUT] (O fastest), activation NHWC.
//   0:in[16,16,CIN] 1:W[3,3,CIN,COUT] 2:out[16,16,COUT] 3:off_mode
//   dispatch {2*CV_THREADS, 2, 1}, tg {CV_THREADS,1,1}
#define CV_HW  16
#define CV_TH  8
#define CV_TW  8
#define CV_CIN 32
#define CV_COUT 64

kernel void conv2d_hw_probe_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& off_mode [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  // NHWC: extents fastest-first (C, W, H, N).
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(CV_CIN, CV_HW, CV_HW, 1));
  // HWIO: extents fastest-first (O, I, W, H).
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(CV_COUT, CV_CIN, 3, 3));
  T4 tD(out, dextents<int32_t, 4>(CV_COUT, CV_HW, CV_HW, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;      // dest-tile origin
  const int oy0 = (int)tgid.y * CV_TH;
  int2 off = int2(ox0, oy0);
  if (off_mode == 1) { off = int2(ox0 - 1, oy0 - 1); }
  if (off_mode == 2) { off = int2(oy0, ox0); }
  if (off_mode == 3) { off = int2(oy0 - 1, ox0 - 1); }
  op.set_offsets(off);

  auto cT = op.get_destination_cooperative_tensor<
      decltype(tA), decltype(tW), VPIPE_ELT>();
  op.run(tA, tW, cT);
  // Store the tile at its TRUE destination position ourselves, so the
  // offset above can only be steering the SOURCE reads.
  auto mD = tD.slice(0, ox0, oy0, 0);
  cT.store(mD);
}

// Probe 2 family (conv2d_hw.probe2*): decompose the two open production
// questions with one variable per entry point.
//   DESC_HW  -- the descriptor's source_dimensions (may be STALE vs image)
//   IMG_HW   -- the real bound image size (tensor extents)
//   COUT_ALL -- total output channels; > CV_COUT tiles channels via
//               tW.slice(oc0,...) + tD.slice(oc0, ox0, oy0, 0) (the int2
//               offset is spatial-only, so channel tiling MUST come from
//               operand slicing if it works at all)
//   2a: DESC 16, IMG 32, COUT 64  -- do runtime extents govern bounds/halo?
//   2b: DESC 16, IMG 16, COUT 128 -- does weight-slice channel tiling work?
//   2c: DESC 32, IMG 32, COUT 64  -- sanity: matching 32x32 descriptor.
// Buffers as probe 1 (off_mode ignored); dispatch
// {(IMG/TW)*CV_THREADS, IMG/TH, COUT_ALL/CV_COUT}.
template <int DESC_HW, int IMG_HW, int COUT_ALL>
static inline void conv2d_hw_probe2_impl(
    const device VPIPE_ELT* inp, const device VPIPE_ELT* wt,
    device VPIPE_ELT* out, uint3 tgid)
{
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(CV_CIN, IMG_HW, IMG_HW, 1));
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(COUT_ALL, CV_CIN, 3, 3));
  T4 tD(out, dextents<int32_t, 4>(COUT_ALL, IMG_HW, IMG_HW, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, DESC_HW, DESC_HW, 1),
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;      // output-channel tile
  op.set_offsets(int2(ox0, oy0));

  auto sW = tW.slice(oc0, 0, 0, 0);           // this tile's CV_COUT filters
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), VPIPE_ELT>();
  op.run(tA, sW, cT);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

#define CV_PROBE2(NAME, DESC_HW, IMG_HW, COUT_ALL)                       \
  kernel void NAME(const device VPIPE_ELT* inp [[buffer(0)]],           \
                   const device VPIPE_ELT* wt [[buffer(1)]],            \
                   device VPIPE_ELT* out [[buffer(2)]],                 \
                   uint3 tgid [[threadgroup_position_in_grid]]) {        \
    conv2d_hw_probe2_impl<DESC_HW, IMG_HW, COUT_ALL>(inp, wt, out, tgid);\
  }

CV_PROBE2(conv2d_hw_probe2a_f16, 16, 32, 64)
CV_PROBE2(conv2d_hw_probe2b_f16, 16, 16, 128)
CV_PROBE2(conv2d_hw_probe2c_f16, 32, 32, 64)

// Probe 2d (RUNTIME source dims): probe 2a proved the driver blob READS the
// descriptor's source_dimensions (stale values -> wrong bounds), and the
// detail entry (__convolution2d_detail::__run) takes the descriptor as a
// RUNTIME struct by reference -- the public run() merely copies the template
// Descriptor into it. So patch source_dimensions (+ real H/W from buffer
// constants) at runtime and call the detail entry directly: if this verifies
// on a 32x32 image with the 16x16-templated descriptor, ONE compiled kernel
// (fixed tile + kernel/stride) serves EVERY image size. The template
// descriptor still fixes the register-sized dest tile (its job).
//   3:img_w 4:img_h added vs the probe-1 buffer contract.
kernel void conv2d_hw_probe2d_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(CV_CIN, img_w, img_h, 1));
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(CV_COUT, CV_CIN, 3, 3));
  T4 tD(out, dextents<int32_t, 4>(CV_COUT, img_w, img_h, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // STALE 16x16
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;

  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(tW), VPIPE_ELT>();
  // Runtime-patched descriptor -> the detail entry the public run() wraps.
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(CV_CIN, img_w, img_h, 1);
  int2 off = int2(ox0, oy0);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(tW), decltype(cT)>(
      tA, tW, cT, rd, off);
  auto mD = tD.slice(0, ox0, oy0, 0);
  cT.store(mD);
}

// GENERAL MPP hardware conv2d, 3x3 stride-1 "same" (the probe-verified
// contract): activation NHWC [H,W,Cin] read as a device tensor -- the op
// gathers the input window INCLUDING the zero-filled border halo itself
// (no im2col, no tgmem staging) -- weights HWIO [3,3,Cin,Cout], dest NHWO.
// The template descriptor fixes only the register-sized dest tile (CV_TW x
// CV_TH pixels x CV_COUT channels); the REAL image/channel dims ride the
// runtime-patched descriptor (probe 2d) + runtime tensor extents, so this
// one kernel serves every shape. Output channels tile via weight/dest
// slices (probe 2b). Requires W%CV_TW == 0, H%CV_TH == 0, Cout%CV_COUT ==
// 0, ANY Cin (runtime).
//   0:in[H,W,Cin] 1:W[3,3,Cin,Cout] 2:out[H,W,Cout] 3:W 4:H 5:Cin
//   dispatch {(W/CV_TW)*CV_THREADS, H/CV_TH, Cout/CV_COUT}, tg
//   {CV_THREADS,1,1}
kernel void conv2d_hw_3x3_s1_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    const constant int& cin   [[buffer(5)]],
    const constant int& cout  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(cin, img_w, img_h, 1));
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(cout, cin, 3, 3));
  T4 tD(out, dextents<int32_t, 4>(cout, img_w, img_h, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // patched below
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;

  auto sW = tW.slice(oc0, 0, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), VPIPE_ELT>();
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(cin, img_w, img_h, 1);
  int2 off = int2(ox0, oy0);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(sW), decltype(cT)>(
      tA, sW, cT, rd, off);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

// STRIDE-2 twin of the general hw conv (the VAE encoder's downsample
// convs): dest [H/2, W/2, Cout], symmetric pad-1 (iy = oy*2 + ky - 1,
// matching im2col_hwc_3x3_s2). The op's stride-2 offset/padding
// convention is probe-established by conv2d_mma.hw_op_s2_probe --
// off_mode selects the interpretation (0 = dest-tile origin in dest
// coords like s1; 1 = source origin incl pad; 2 = source origin no pad);
// production passes the probe-verified mode. Same contract as the s1
// kernel + 7:off_mode; W, H even; dispatch over the DEST grid
// {(W/2/CV_TW)*CV_THREADS, H/2/CV_TH, Cout/CV_COUT}.
kernel void conv2d_hw_3x3_s2_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    const constant int& cin   [[buffer(5)]],
    const constant int& cout  [[buffer(6)]],
    const constant int& off_mode [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const int ow = img_w / 2, oh = img_h / 2;
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(cin, img_w, img_h, 1));
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(cout, cin, 3, 3));
  T4 tD(out, dextents<int32_t, 4>(cout, ow, oh, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // patched below
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(2, 2), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;      // dest-tile origin
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;

  // The op reads src = off + k - 1 (pad-1 relative to the offset), so the
  // offset encodes the padding convention:
  //   mode 2: (2*ox0, 2*oy0)      -> iy = 2*oy + ky - 1 (SYMMETRIC pad 1)
  //   mode 3: (2*ox0+1, 2*oy0+1)  -> iy = 2*oy + ky     (ASYMMETRIC pad
  //           (0,1,0,1), the diffusers Downsample2D / im2col_hwc_3x3_s2
  //           convention the VAE encoders use)
  int2 off = int2(ox0, oy0);
  if (off_mode == 1) { off = int2(ox0 * 2 - 1, oy0 * 2 - 1); }
  if (off_mode == 2) { off = int2(ox0 * 2, oy0 * 2); }
  if (off_mode == 3) { off = int2(ox0 * 2 + 1, oy0 * 2 + 1); }

  auto sW = tW.slice(oc0, 0, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), VPIPE_ELT>();
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(cin, img_w, img_h, 1);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(sW), decltype(cT)>(
      tA, sW, cT, rd, off);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

// Bias-FUSED twin of the general hw conv: the cooperative destination is
// pre-initialized with bias[oc] and the descriptor runs
// mode::multiply_accumulate, so the matrix units accumulate the conv ON
// TOP of the bias -- out = bias + conv with no separate bias_add_rows
// pass. Same contract as conv2d_hw_3x3_s1_f16 + 7:bias[Cout].
//
// MEASURED (M5, conv2d_mma.hw_op_bias_fused): CORRECT (matches conv +
// bias_add_rows to 3e-4) but NOT a win -- 0.99x at 512x512x128, 0.57x at
// 64x64x128. The cost is the multiply_accumulate run path itself, not the
// init (a zero-stride broadcast cT.load(bias) and the per-element
// get_multidimensional_index loop measure IDENTICALLY): in multiply mode
// the destination is write-only to the matrix pipeline, while mul_acc
// feeds the pre-set accumulator through it, costing ~the bias pass it
// replaces (a fixed per-tile tax that dominates small shapes). Kept as
// the measurement record; production should ship conv (multiply) +
// bias_add_rows.
kernel void conv2d_hw_3x3_s1_bias_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    const constant int& cin   [[buffer(5)]],
    const constant int& cout  [[buffer(6)]],
    const device VPIPE_ELT* bias [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(cin, img_w, img_h, 1));
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(cout, cin, 3, 3));
  T4 tD(out, dextents<int32_t, 4>(cout, img_w, img_h, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // patched below
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply_accumulate);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;

  auto sW = tW.slice(oc0, 0, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), VPIPE_ELT>();
  // Broadcast-load the bias into the cooperative layout: a zero-stride
  // rank-4 view repeats bias[c] across x/y/n, and load() vectorizes the
  // coordinate mapping (the per-element get_multidimensional_index loop
  // measured ~as expensive as the bias_add_rows pass it replaces).
  tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline> tB(
      const_cast<device VPIPE_ELT*>(bias) + oc0,
      dextents<int32_t, 4>(CV_COUT, CV_TW, CV_TH, 1),
      metal::array<int32_t, 4>{1, 0, 0, 0});
  cT.load(tB);
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(cin, img_w, img_h, 1);
  int2 off = int2(ox0, oy0);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(sW), decltype(cT)>(
      tA, sW, cT, rd, off);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

// int8 x int8 -> f16 twin of the general hw conv (the impl exposes a
// dv_i8_dv_i8_dv_f16 variant): integer activation/weights halve the
// operand bandwidth vs f16; the accumulator is integer and the raw sum
// converts to f16 on store (NO scale slot in the op -- a real quant conv
// folds scales in a separate epilogue, like dequant elsewhere).
//
// MEASURED (M5, conv2d_mma.hw_op_i8_vs_f16_perf): i8 conv runs ABOVE the
// f16 rate -- 1.83x @512x512x128 (24.1 vs 13.1 TOPS/TFLOPS), 1.88x
// @256x256x256, ~1.45x at smaller shapes -- approaching the 2x int8 peak
// at compute-bound sizes. (Contrast: matmul2d measured i8 AT the f16 rate
// on the codec shapes; whether that was shape-specific is an open
// question.) Production notes: use the f32/i32 destination for real i8
// data (full-range i8 sums reach ~1.9e7, far past f16), and fold the
// quant scales in the epilogue. ELT-independent (explicit int8/half
// types); same contract as conv2d_hw_3x3_s1_f16.
kernel void conv2d_hw_3x3_s1_i8f16(
    const device int8_t* inp  [[buffer(0)]],
    const device int8_t* wt   [[buffer(1)]],
    device half*         out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    const constant int& cin   [[buffer(5)]],
    const constant int& cout  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4I = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>;
  using T4H = tensor<device half, dextents<int32_t, 4>, tensor_inline>;
  T4I tA(const_cast<device int8_t*>(inp),
         dextents<int32_t, 4>(cin, img_w, img_h, 1));
  T4I tW(const_cast<device int8_t*>(wt),
         dextents<int32_t, 4>(cout, cin, 3, 3));
  T4H tD(out, dextents<int32_t, 4>(cout, img_w, img_h, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // patched below
      /*kernel_dimensions=*/int2(3, 3),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;

  auto sW = tW.slice(oc0, 0, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), half>();
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(cin, img_w, img_h, 1);
  int2 off = int2(ox0, oy0);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(sW), decltype(cT)>(
      tA, sW, cT, rd, off);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

// 1x1 twins of the general hw conv (kernel_dimensions (1,1): no halo, a
// pure GEMM with M = H*W, N = Cout, K = Cin -- the VAE's shortcut/attn
// projections). Contracts identical to the 3x3 entries.
//
// MEASURED (M5, conv2d_mma.hw_op_1x1_i8_vs_f16_perf): the int8 win is
// K-DEPENDENT. K=128: i8 LOSES (0.76-0.92x vs f16 conv; 0.52-0.79x vs the
// plain dense_gemm_mma the VAE 1x1 path uses today -- the skinny-K regime
// is bandwidth/store-bound and the f16 destination write doesn't shrink,
// while the conv op adds overhead over plain matmul). K=256: 1.72x.
// K=512: 1.93x (~20 TOPS). Matches the 3x3 results (K=9*Cin >= 1152 ->
// 1.8-1.9x): int8 doubles the MATRIX-PIPE rate, so it pays only once the
// shape is compute-bound -- and explains matmul2d's "i8 at f16 rate"
// finding (measured on skinny codec shapes). Routing rule: i8 for
// K >= ~256, f16/GEMM below.
kernel void conv2d_hw_1x1_s1_f16(
    const device VPIPE_ELT* inp  [[buffer(0)]],
    const device VPIPE_ELT* wt   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    const constant int& cin   [[buffer(5)]],
    const constant int& cout  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4 = tensor<device VPIPE_ELT, dextents<int32_t, 4>, tensor_inline>;
  T4 tA(const_cast<device VPIPE_ELT*>(inp),
        dextents<int32_t, 4>(cin, img_w, img_h, 1));
  T4 tW(const_cast<device VPIPE_ELT*>(wt),
        dextents<int32_t, 4>(cout, cin, 1, 1));
  T4 tD(out, dextents<int32_t, 4>(cout, img_w, img_h, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // patched below
      /*kernel_dimensions=*/int2(1, 1),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;

  auto sW = tW.slice(oc0, 0, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), VPIPE_ELT>();
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(cin, img_w, img_h, 1);
  int2 off = int2(ox0, oy0);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(sW), decltype(cT)>(
      tA, sW, cT, rd, off);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

kernel void conv2d_hw_1x1_s1_i8f16(
    const device int8_t* inp  [[buffer(0)]],
    const device int8_t* wt   [[buffer(1)]],
    device half*         out  [[buffer(2)]],
    const constant int& img_w [[buffer(3)]],
    const constant int& img_h [[buffer(4)]],
    const constant int& cin   [[buffer(5)]],
    const constant int& cout  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  using T4I = tensor<device int8_t, dextents<int32_t, 4>, tensor_inline>;
  using T4H = tensor<device half, dextents<int32_t, 4>, tensor_inline>;
  T4I tA(const_cast<device int8_t*>(inp),
         dextents<int32_t, 4>(cin, img_w, img_h, 1));
  T4I tW(const_cast<device int8_t*>(wt),
         dextents<int32_t, 4>(cout, cin, 1, 1));
  T4H tD(out, dextents<int32_t, 4>(cout, img_w, img_h, 1));

  constexpr auto desc = convolution2d_descriptor(
      /*destination_dimensions=*/int4(CV_COUT, CV_TW, CV_TH, 1),
      /*source_dimensions=*/int4(CV_CIN, CV_HW, CV_HW, 1),   // patched below
      /*kernel_dimensions=*/int2(1, 1),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      /*strides=*/int2(1, 1), /*dilations=*/int2(1, 1), /*groups=*/1,
      /*relaxed_precision=*/false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<CV_SG>> op;

  const int ox0 = (int)tgid.x * CV_TW;
  const int oy0 = (int)tgid.y * CV_TH;
  const int oc0 = (int)tgid.z * CV_COUT;

  auto sW = tW.slice(oc0, 0, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<
      decltype(tA), decltype(sW), half>();
  convolution2d_descriptor rd = desc;
  rd.source_dimensions = int4(cin, img_w, img_h, 1);
  int2 off = int2(ox0, oy0);
  __convolution2d_detail::__run<execution_simdgroups<CV_SG>,
                                decltype(tA), decltype(sW), decltype(cT)>(
      tA, sW, cT, rd, off);
  auto mD = tD.slice(oc0, ox0, oy0, 0);
  cT.store(mD);
}

#else
// Tensor ops unavailable for this target: emit stubs so the metallib still
// builds. The loader never binds these on a non-tensor GPU.
kernel void conv2d_mma_3x3_s1_f16(device VPIPE_ELT* out [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_mma_3x3_s2_f16(device VPIPE_ELT* out [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_probe_f16(device VPIPE_ELT* out [[buffer(2)]],
                                uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_probe2a_f16(device VPIPE_ELT* out [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_probe2b_f16(device VPIPE_ELT* out [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_probe2c_f16(device VPIPE_ELT* out [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_probe2d_f16(device VPIPE_ELT* out [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_3x3_s1_f16(device VPIPE_ELT* out [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_3x3_s1_bias_f16(device VPIPE_ELT* out [[buffer(2)]],
                                      uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_3x3_s1_i8f16(device half* out [[buffer(2)]],
                                   uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (half)0; } }
kernel void conv2d_hw_1x1_s1_f16(device VPIPE_ELT* out [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
kernel void conv2d_hw_1x1_s1_i8f16(device half* out [[buffer(2)]],
                                   uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (half)0; } }
kernel void conv2d_hw_3x3_s2_f16(device VPIPE_ELT* out [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]])
{ if (tid == 0) { out[0] = (VPIPE_ELT)0; } }
#endif
