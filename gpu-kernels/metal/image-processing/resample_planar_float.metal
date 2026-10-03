// resample_planar_float.metal -- resample of a planar FLOAT picture
// [C, in_h, in_w] -> [C, out_h, out_w] of the same element type, C 3 (RGB)
// or 4 (RGBA). The f16 and f32 twins of the u8 letterbox and Lanczos
// kernels, for pictures that must not pass through 8 bits on the way to a
// model or a file.
//
// Samples are read and written as they are: 0..1 by the picture
// convention, but never clamped -- a linear picture exceeds 1, and
// Lanczos and bicubic ring below 0 and above 1 at a hard edge. The
// arithmetic is float whatever the storage, so the f16 twin differs from
// the f32 one only by the rounding of what it stores.
//
// RGBA RESAMPLES PREMULTIPLIED, inside the kernel: every tap's colour is
// weighted by its alpha, and the sum divided back out by the resampled
// alpha. Averaging colour and alpha independently pulls the colour of
// fully transparent pixels into the edge -- the dark halo a generated
// RGBA picture, transparent over black, would otherwise get. A pixel
// whose alpha comes out below 1e-4 keeps its premultiplied colour, as the
// CPU path does.
//
// The mapping is the u8 kernels': output pixels in the destination
// rectangle [pad_x, pad_x+new_w) x [pad_y, pad_y+new_h) are resampled
// content, everything outside is the solid `pad` colour (its alpha is
// the pad's own: 0, transparent, from image-resample). That one rectangle
// expresses every fit -- pad, crop, stretch, manual (ResampleGeom).
//
// Two filter shapes, each instantiated for half and float:
//
//   resample_bilinear_planar_{f16,f32}
//     src = (dst - pad + 0.5) * inv - 0.5 + src0, per axis
//     params  : (in_w, in_h, out_w, out_h)
//     params2 : (new_w, new_h, pad_x, pad_y)
//     params3 : (inv_x, inv_y, src_x0, src_y0)
//     params4 : (channels, _, _, _)
//     pad     : (r, g, b, a) in the picture's units
//
//   resample_separable_planar_{f16,f32}
//     Lanczos-3 or Pillow's bicubic -- the FILTER is the host's
//     coefficient tables (build_lanczos_coeffs / build_cubic_coeffs, the
//     same ones the CPU path uses), so one kernel is both. A single 2D
//     gather, mathematically Pillow's two passes with a full-precision
//     intermediate.
//     wx[new_w * ksx] / bx[new_w] : per-output-x weights + first column
//     wy[new_h * ksy] / by[new_h] : per-output-y weights + first row
//     params  : (in_w, in_h, out_w, out_h)
//     params2 : (new_w, new_h, pad_x, pad_y)
//     params3 : (ksx, ksy, channels, _)
//     pad     : (r, g, b, a) in the picture's units

#include <metal_stdlib>
using namespace metal;

namespace {

template <typename T>
inline void
write_pad_(device T* dst, uint plane, uint idx, uint channels, float4 pad)
{
  for (uint c = 0; c < channels; ++c) {
    dst[c * plane + idx] = T(pad[c]);
  }
}

}  // namespace

template <typename T>
kernel void
resample_bilinear_planar(
    device const T*  src     [[buffer(0)]],
    device T*        dst     [[buffer(1)]],
    constant uint4&  params  [[buffer(2)]],
    constant uint4&  params2 [[buffer(3)]],
    constant float4& params3 [[buffer(4)]],
    constant uint4&  params4 [[buffer(5)]],
    constant float4& pad     [[buffer(6)]],
    uint2            gid     [[thread_position_in_grid]])
{
  const uint in_w  = params.x,  in_h  = params.y;
  const uint out_w = params.z,  out_h = params.w;
  const uint new_w = params2.x, new_h = params2.y;
  const uint pad_x = params2.z, pad_y = params2.w;
  const uint channels = params4.x;

  if (gid.x >= out_w || gid.y >= out_h) { return; }

  const uint plane_dst = out_w * out_h;
  const uint dst_idx   = gid.y * out_w + gid.x;
  const bool outside =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (outside) {
    write_pad_(dst, plane_dst, dst_idx, channels, pad);
    return;
  }

  const float sxf =
      (float(gid.x - pad_x) + 0.5f) * params3.x - 0.5f + params3.z;
  const float syf =
      (float(gid.y - pad_y) + 0.5f) * params3.y - 0.5f + params3.w;
  int ix0 = int(floor(sxf));
  int iy0 = int(floor(syf));
  const float fx = sxf - float(ix0);
  const float fy = syf - float(iy0);
  int ix1 = clamp(ix0 + 1, 0, int(in_w) - 1);
  int iy1 = clamp(iy0 + 1, 0, int(in_h) - 1);
  ix0 = clamp(ix0, 0, int(in_w) - 1);
  iy0 = clamp(iy0, 0, int(in_h) - 1);

  const uint plane_src = in_w * in_h;
  const uint i00 = uint(iy0) * in_w + uint(ix0);
  const uint i01 = uint(iy0) * in_w + uint(ix1);
  const uint i10 = uint(iy1) * in_w + uint(ix0);
  const uint i11 = uint(iy1) * in_w + uint(ix1);

  if (channels == 4) {
    const uint ab = 3 * plane_src;
    const float a00 = float(src[ab + i00]), a01 = float(src[ab + i01]);
    const float a10 = float(src[ab + i10]), a11 = float(src[ab + i11]);
    const float a = mix(mix(a00, a01, fx), mix(a10, a11, fx), fy);
    for (uint c = 0; c < 3; ++c) {
      const uint b = c * plane_src;
      const float p0 = mix(float(src[b + i00]) * a00,
                           float(src[b + i01]) * a01, fx);
      const float p1 = mix(float(src[b + i10]) * a10,
                           float(src[b + i11]) * a11, fx);
      const float p = mix(p0, p1, fy);
      dst[c * plane_dst + dst_idx] = T(a > 1e-4f ? p / a : p);
    }
    dst[3 * plane_dst + dst_idx] = T(a);
    return;
  }
  for (uint c = 0; c < channels; ++c) {
    const uint b = c * plane_src;
    const float v0 = mix(float(src[b + i00]), float(src[b + i01]), fx);
    const float v1 = mix(float(src[b + i10]), float(src[b + i11]), fx);
    dst[c * plane_dst + dst_idx] = T(mix(v0, v1, fy));
  }
}

template <typename T>
kernel void
resample_separable_planar(
    device const T*     src     [[buffer(0)]],
    device T*           dst     [[buffer(1)]],
    device const float* wx      [[buffer(2)]],
    device const int*   bx      [[buffer(3)]],
    device const float* wy      [[buffer(4)]],
    device const int*   by      [[buffer(5)]],
    constant uint4&     params  [[buffer(6)]],
    constant uint4&     params2 [[buffer(7)]],
    constant uint4&     params3 [[buffer(8)]],
    constant float4&    pad     [[buffer(9)]],
    uint2               gid     [[thread_position_in_grid]])
{
  const uint in_w  = params.x,  in_h  = params.y;
  const uint out_w = params.z,  out_h = params.w;
  const uint new_w = params2.x, new_h = params2.y;
  const uint pad_x = params2.z, pad_y = params2.w;
  const uint ksx   = params3.x, ksy   = params3.y;
  const uint channels = params3.z;

  if (gid.x >= out_w || gid.y >= out_h) { return; }

  const uint plane_dst = out_w * out_h;
  const uint dst_idx   = gid.y * out_w + gid.x;
  const bool outside =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (outside) {
    write_pad_(dst, plane_dst, dst_idx, channels, pad);
    return;
  }

  const uint cx = gid.x - pad_x;          // content-rect coords
  const uint cy = gid.y - pad_y;
  const int  xmin = bx[cx];
  const int  ymin = by[cy];
  const int  iw = int(in_w), ih = int(in_h);
  const uint plane_src = in_w * in_h;
  const bool rgba = channels == 4;
  const uint colour = rgba ? 3 : channels;

  // Colour planes (premultiplied for RGBA) and alpha, summed in one walk
  // over the taps so the alpha weighting needs no second read.
  float acc[3] = {0.0f, 0.0f, 0.0f};
  float acc_a = 0.0f;
  for (uint ty = 0; ty < ksy; ++ty) {
    const float wyt = wy[cy * ksy + ty];
    if (wyt == 0.0f) { continue; }        // zero-padded edge tap
    const int  sy  = clamp(ymin + int(ty), 0, ih - 1);
    const uint row = uint(sy) * in_w;
    float racc[3] = {0.0f, 0.0f, 0.0f};
    float racc_a = 0.0f;
    for (uint tx = 0; tx < ksx; ++tx) {
      const float wxt = wx[cx * ksx + tx];
      const int   sx  = clamp(xmin + int(tx), 0, iw - 1);
      const uint  i   = row + uint(sx);
      const float a   = rgba ? float(src[3 * plane_src + i]) : 1.0f;
      for (uint c = 0; c < colour; ++c) {
        racc[c] += wxt * float(src[c * plane_src + i]) * a;
      }
      racc_a += wxt * a;
    }
    for (uint c = 0; c < colour; ++c) { acc[c] += wyt * racc[c]; }
    acc_a += wyt * racc_a;
  }
  for (uint c = 0; c < colour; ++c) {
    const float v = rgba && acc_a > 1e-4f ? acc[c] / acc_a : acc[c];
    dst[c * plane_dst + dst_idx] = T(v);
  }
  if (rgba) { dst[3 * plane_dst + dst_idx] = T(acc_a); }
}

#define VPIPE_RESAMPLE_INSTANTIATE(T, SUFFIX)                                 \
  template [[host_name("resample_bilinear_planar_" #SUFFIX)]] kernel void     \
  resample_bilinear_planar<T>(                                                \
      device const T*  src     [[buffer(0)]],                                 \
      device T*        dst     [[buffer(1)]],                                 \
      constant uint4&  params  [[buffer(2)]],                                 \
      constant uint4&  params2 [[buffer(3)]],                                 \
      constant float4& params3 [[buffer(4)]],                                 \
      constant uint4&  params4 [[buffer(5)]],                                 \
      constant float4& pad     [[buffer(6)]],                                 \
      uint2            gid     [[thread_position_in_grid]]);                  \
  template [[host_name("resample_separable_planar_" #SUFFIX)]] kernel void    \
  resample_separable_planar<T>(                                               \
      device const T*     src     [[buffer(0)]],                              \
      device T*           dst     [[buffer(1)]],                              \
      device const float* wx      [[buffer(2)]],                              \
      device const int*   bx      [[buffer(3)]],                              \
      device const float* wy      [[buffer(4)]],                              \
      device const int*   by      [[buffer(5)]],                              \
      constant uint4&     params  [[buffer(6)]],                              \
      constant uint4&     params2 [[buffer(7)]],                              \
      constant uint4&     params3 [[buffer(8)]],                              \
      constant float4&    pad     [[buffer(9)]],                              \
      uint2               gid     [[thread_position_in_grid]]);

VPIPE_RESAMPLE_INSTANTIATE(half,  f16)
VPIPE_RESAMPLE_INSTANTIATE(float, f32)
