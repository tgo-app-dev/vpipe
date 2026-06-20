// motion_diff_u8.metal -- ported from metal-runtime.cc.
// Per-pixel absolute difference between two previously-computed
// u8 luma signatures.

#include <metal_stdlib>
using namespace metal;

kernel void motion_diff_u8(
    device const uchar* sig_a   [[buffer(0)]],
    device const uchar* sig_b   [[buffer(1)]],
    device uchar*       dst     [[buffer(2)]],
    constant uint2&     params  [[buffer(3)]],
    uint2               gid     [[thread_position_in_grid]])
{
  uint tw = params.x;
  uint th = params.y;
  if (gid.x >= tw || gid.y >= th) { return; }
  uint i = gid.y * tw + gid.x;
  int a = int(sig_a[i]);
  int b = int(sig_b[i]);
  int d = a - b;
  if (d < 0) { d = -d; }
  dst[i] = uchar(d);
}
