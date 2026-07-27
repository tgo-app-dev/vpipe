// resample_lanczos_planar_u8.metal -- Lanczos-3 resample of planar U8 RGB
// [3, in_h, in_w] -> planar U8 [3, out_h, out_w], matching PIL's LANCZOS
// (ImagingResample). The horizontal + vertical filter coefficients are
// precomputed on the host EXACTLY as Pillow's precompute_coeffs (support 3.0,
// filterscale = max(1, in/out) so a downscale widens the kernel -> anti-
// aliased), normalized per output pixel, and uploaded as (weights, first-source-
// index) tables. This kernel does the separable convolution as a single 2D
// gather (mathematically identical to Pillow's two-pass horizontal-then-vertical
// with a full-precision intermediate).
//
// Like the letterbox kernel the mapping is parameterised: output pixels in the
// destination rectangle [pad_x, pad_x+new_w) x [pad_y, pad_y+new_h) are the
// resampled content; everything OUTSIDE is the solid `pad` colour.
//
//   wx[new_w * ksx] / bx[new_w] : per-output-x weights + first source column
//   wy[new_h * ksy] / by[new_h] : per-output-y weights + first source row
//   params  : (in_w, in_h, out_w, out_h)
//   params2 : (new_w, new_h, pad_x, pad_y)   -- dest rectangle
//   params3 : (ksx, ksy, _, _)               -- per-axis kernel tap count
//   params4 : (pad_r, pad_g, pad_b, _)        -- pad colour, 0..255

#include <metal_stdlib>
using namespace metal;

kernel void resample_lanczos_planar_u8(
    device const uchar* src     [[buffer(0)]],
    device uchar*       dst     [[buffer(1)]],
    device const float* wx      [[buffer(2)]],
    device const int*   bx      [[buffer(3)]],
    device const float* wy      [[buffer(4)]],
    device const int*   by      [[buffer(5)]],
    constant uint4&     params  [[buffer(6)]],
    constant uint4&     params2 [[buffer(7)]],
    constant uint4&     params3 [[buffer(8)]],
    constant float4&    params4 [[buffer(9)]],
    uint2               gid     [[thread_position_in_grid]])
{
  const uint in_w  = params.x, in_h  = params.y;
  const uint out_w = params.z, out_h = params.w;
  const uint new_w = params2.x, new_h = params2.y;
  const uint pad_x = params2.z, pad_y = params2.w;
  const uint ksx   = params3.x, ksy   = params3.y;

  if (gid.x >= out_w || gid.y >= out_h) { return; }

  const uint plane_dst = out_w * out_h;
  const uint dst_idx   = gid.y * out_w + gid.x;

  const bool outside =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (outside) {
    dst[              dst_idx] = uchar(clamp(params4.x + 0.5f, 0.0f, 255.0f));
    dst[plane_dst   + dst_idx] = uchar(clamp(params4.y + 0.5f, 0.0f, 255.0f));
    dst[2*plane_dst + dst_idx] = uchar(clamp(params4.z + 0.5f, 0.0f, 255.0f));
    return;
  }

  const uint cx = gid.x - pad_x;          // content-rect coords
  const uint cy = gid.y - pad_y;
  const int  xmin = bx[cx];
  const int  ymin = by[cy];
  const int  iw = int(in_w), ih = int(in_h);
  const uint plane_src = in_w * in_h;

  for (uint c = 0; c < 3; ++c) {
    const uint base = c * plane_src;
    float acc = 0.0f;
    for (uint ty = 0; ty < ksy; ++ty) {
      const float wyt = wy[cy * ksy + ty];
      if (wyt == 0.0f) { continue; }        // zero-padded edge tap
      const int sy = clamp(ymin + int(ty), 0, ih - 1);
      const uint row = base + uint(sy) * in_w;
      float racc = 0.0f;
      for (uint tx = 0; tx < ksx; ++tx) {
        const int sx = clamp(xmin + int(tx), 0, iw - 1);
        racc += wx[cx * ksx + tx] * float(src[row + uint(sx)]);
      }
      acc += wyt * racc;
    }
    dst[c * plane_dst + dst_idx] = uchar(clamp(acc + 0.5f, 0.0f, 255.0f));
  }
}
