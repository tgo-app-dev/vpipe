// letterbox_planar_u8_to_u8_chw.metal -- bilinear resample of planar U8
// RGB [3, in_h, in_w] -> planar U8 [3, out_h, out_w]. The default use is
// an aspect-preserving letterbox (match the long side, pad the rest), but
// the mapping is fully parameterised so a caller can also crop, stretch,
// or place a sub-region: output pixels in the destination rectangle
// [pad_x, pad_x+new_w) x [pad_y, pad_y+new_h) sample the source at
//   src = (dst - pad) * inv  +  src0        (per axis, inv = 1/scale)
// and everything OUTSIDE that rectangle is filled with the solid `pad`
// colour. Output stays 8-bit (no /255) so it can feed a vision encoder
// that itself takes planar u8 RGB (the GPU VLM-input resampler).
//
//   params  : (in_w, in_h, out_w, out_h)
//   params2 : (new_w, new_h, pad_x, pad_y)   -- dest rectangle
//   params3 : (inv_x, inv_y, src_x0, src_y0) -- source mapping
//   params4 : (pad_r, pad_g, pad_b, _)       -- pad colour, 0..255

#include <metal_stdlib>
using namespace metal;

kernel void letterbox_planar_u8_to_u8_chw(
    device const uchar* src       [[buffer(0)]],
    device uchar*       dst       [[buffer(1)]],
    constant uint4&     params    [[buffer(2)]],
    constant uint4&     params2   [[buffer(3)]],
    constant float4&    params3   [[buffer(4)]],
    constant float4&    params4   [[buffer(5)]],
    uint2               gid       [[thread_position_in_grid]])
{
  uint in_w     = params.x;
  uint in_h     = params.y;
  uint out_w    = params.z;
  uint out_h    = params.w;
  uint new_w    = params2.x;
  uint new_h    = params2.y;
  uint pad_x    = params2.z;
  uint pad_y    = params2.w;
  float inv_x   = params3.x;
  float inv_y   = params3.y;
  float src_x0  = params3.z;
  float src_y0  = params3.w;

  if (gid.x >= out_w || gid.y >= out_h) { return; }

  uint plane_dst = out_w * out_h;
  uint dst_idx   = gid.y * out_w + gid.x;
  bool out =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (out) {
    dst[              dst_idx] = uchar(clamp(params4.x + 0.5f, 0.0f, 255.0f));
    dst[plane_dst   + dst_idx] = uchar(clamp(params4.y + 0.5f, 0.0f, 255.0f));
    dst[2*plane_dst + dst_idx] = uchar(clamp(params4.z + 0.5f, 0.0f, 255.0f));
    return;
  }

  float sxf = (float(gid.x - pad_x) + 0.5) * inv_x - 0.5 + src_x0;
  float syf = (float(gid.y - pad_y) + 0.5) * inv_y - 0.5 + src_y0;
  int ix0 = int(floor(sxf));
  int iy0 = int(floor(syf));
  int ix1 = ix0 + 1;
  int iy1 = iy0 + 1;
  float fx = sxf - float(ix0);
  float fy = syf - float(iy0);
  ix0 = clamp(ix0, 0, int(in_w) - 1);
  ix1 = clamp(ix1, 0, int(in_w) - 1);
  iy0 = clamp(iy0, 0, int(in_h) - 1);
  iy1 = clamp(iy1, 0, int(in_h) - 1);

  uint plane_src = in_w * in_h;
  for (uint c = 0; c < 3; ++c) {
    uint base = c * plane_src;
    float v00 = float(src[base + uint(iy0) * in_w + uint(ix0)]);
    float v01 = float(src[base + uint(iy0) * in_w + uint(ix1)]);
    float v10 = float(src[base + uint(iy1) * in_w + uint(ix0)]);
    float v11 = float(src[base + uint(iy1) * in_w + uint(ix1)]);
    float v0  = mix(v00, v01, fx);
    float v1  = mix(v10, v11, fx);
    float v   = mix(v0,  v1,  fy);
    dst[c * plane_dst + dst_idx] =
        uchar(clamp(v + 0.5f, 0.0f, 255.0f));
  }
}
