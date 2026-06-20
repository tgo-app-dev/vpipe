// motion_signature_u8.metal -- ported from metal-runtime.cc.
// Averages planar-u8 RGB tiles into a small u8 luma signature
// suitable for cheap frame-to-frame motion scoring. BT.601 luma
// fixed-point: (77 R + 150 G + 29 B) >> 8 over the tile average.

#include <metal_stdlib>
using namespace metal;

kernel void motion_signature_u8(
    device const uchar* src     [[buffer(0)]],
    device uchar*       dst     [[buffer(1)]],
    constant uint4&     params  [[buffer(2)]],
    uint2               gid     [[thread_position_in_grid]])
{
  uint in_w = params.x;
  uint in_h = params.y;
  uint tw   = params.z;
  uint th   = params.w;
  if (gid.x >= tw || gid.y >= th) { return; }

  uint x0 = (gid.x * in_w) / tw;
  uint x1 = ((gid.x + 1u) * in_w) / tw;
  uint y0 = (gid.y * in_h) / th;
  uint y1 = ((gid.y + 1u) * in_h) / th;
  if (x1 <= x0 || y1 <= y0) {
    dst[gid.y * tw + gid.x] = 0;
    return;
  }
  uint plane = in_w * in_h;
  uint sum_r = 0u;
  uint sum_g = 0u;
  uint sum_b = 0u;
  uint count = 0u;
  for (uint y = y0; y < y1; ++y) {
    uint row = y * in_w;
    for (uint x = x0; x < x1; ++x) {
      uint i = row + x;
      sum_r += uint(src[i]);
      sum_g += uint(src[plane + i]);
      sum_b += uint(src[2u * plane + i]);
      ++count;
    }
  }
  uint avg_r = sum_r / count;
  uint avg_g = sum_g / count;
  uint avg_b = sum_b / count;
  uint y_val = (77u * avg_r + 150u * avg_g + 29u * avg_b) >> 8;
  if (y_val > 255u) { y_val = 255u; }
  dst[gid.y * tw + gid.x] = uchar(y_val);
}
