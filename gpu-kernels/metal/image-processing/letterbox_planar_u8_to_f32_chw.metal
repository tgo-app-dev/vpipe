// letterbox_planar_u8_to_f32_chw.metal -- ported from
// metal-runtime.cc. Bilinear letterbox resample for inference
// preprocessing: planar U8 [3, in_h, in_w] -> planar F32
// [3, out_h, out_w] with aspect-preserving scale + 114/255 grey
// padding outside the letterbox region.

#include <metal_stdlib>
using namespace metal;

kernel void letterbox_planar_u8_to_f32_chw(
    device const uchar* src       [[buffer(0)]],
    device float*       dst       [[buffer(1)]],
    constant uint4&     params    [[buffer(2)]],
    constant uint4&     params2   [[buffer(3)]],
    constant float4&    params3   [[buffer(4)]],
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
  float inv     = params3.x;

  if (gid.x >= out_w || gid.y >= out_h) { return; }

  uint plane_dst = out_w * out_h;
  uint dst_idx   = gid.y * out_w + gid.x;
  bool out =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (out) {
    float g = 114.0 / 255.0;
    dst[              dst_idx] = g;
    dst[plane_dst   + dst_idx] = g;
    dst[2*plane_dst + dst_idx] = g;
    return;
  }

  float sxf = (float(gid.x - pad_x) + 0.5) * inv - 0.5;
  float syf = (float(gid.y - pad_y) + 0.5) * inv - 0.5;
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
    float v00 = float(src[base + uint(iy0) * in_w + uint(ix0)]) * (1.0 / 255.0);
    float v01 = float(src[base + uint(iy0) * in_w + uint(ix1)]) * (1.0 / 255.0);
    float v10 = float(src[base + uint(iy1) * in_w + uint(ix0)]) * (1.0 / 255.0);
    float v11 = float(src[base + uint(iy1) * in_w + uint(ix1)]) * (1.0 / 255.0);
    float v0  = mix(v00, v01, fx);
    float v1  = mix(v10, v11, fx);
    float v   = mix(v0,  v1,  fy);
    dst[c * plane_dst + dst_idx] = v;
  }
}
