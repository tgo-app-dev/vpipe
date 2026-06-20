// letterbox_planar_u8_to_bgra_u8.metal -- ported from
// metal-runtime.cc. Twin of letterbox_planar_u8_to_f32_chw but
// emits interleaved BGRA8888 with row stride dst_bpr; used by the
// CoreML CVPixelBuffer (ImageType) inference path.

#include <metal_stdlib>
using namespace metal;

kernel void letterbox_planar_u8_to_bgra_u8(
    device const uchar* src       [[buffer(0)]],
    device uchar*       dst       [[buffer(1)]],
    constant uint4&     params    [[buffer(2)]],
    constant uint4&     params2   [[buffer(3)]],
    constant float4&    params3   [[buffer(4)]],
    constant uint&      dst_bpr   [[buffer(5)]],
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

  uint dst_off = gid.y * dst_bpr + gid.x * 4u;
  bool outside =
      (gid.x < pad_x) || (gid.x >= pad_x + new_w) ||
      (gid.y < pad_y) || (gid.y >= pad_y + new_h);
  if (outside) {
    dst[dst_off + 0] = 114;
    dst[dst_off + 1] = 114;
    dst[dst_off + 2] = 114;
    dst[dst_off + 3] = 255;
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

  uint plane = in_w * in_h;
  uchar bgr[3];
  for (uint c = 0; c < 3u; ++c) {
    uint base = c * plane;
    float v00 = float(src[base + uint(iy0) * in_w + uint(ix0)]);
    float v01 = float(src[base + uint(iy0) * in_w + uint(ix1)]);
    float v10 = float(src[base + uint(iy1) * in_w + uint(ix0)]);
    float v11 = float(src[base + uint(iy1) * in_w + uint(ix1)]);
    float v0  = mix(v00, v01, fx);
    float v1  = mix(v10, v11, fx);
    float v   = mix(v0, v1, fy);
    bgr[c]    = uchar(clamp(v + 0.5, 0.0, 255.0));
  }
  dst[dst_off + 0] = bgr[2];
  dst[dst_off + 1] = bgr[1];
  dst[dst_off + 2] = bgr[0];
  dst[dst_off + 3] = 255;
}
