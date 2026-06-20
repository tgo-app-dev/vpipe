// nv12_to_planar_rgb_u8.metal -- ported from metal-runtime.cc.
//
// NV12 -> centered-crop -> bilinear/bicubic rescale -> planar RGB
// (buffer-based, no textures). One kernel handles both pass-through
// (out == in, no crop) and arbitrary crop+rescale. BT.709 full-range
// YCbCr->RGB. The kernel switches to Catmull-Rom bicubic when either
// axis is downsampled by more than 2x; bilinear otherwise.

#include <metal_stdlib>
using namespace metal;

inline float4 cubic_w_(float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  return float4(
      (-0.5 * t3 +       t2 - 0.5 * t),
      ( 1.5 * t3 - 2.5 * t2             + 1.0),
      (-1.5 * t3 + 2.0 * t2 + 0.5 * t),
      ( 0.5 * t3 - 0.5 * t2)
  );
}

inline float bicubic_u8_(device const uchar* buf,
                         uint stride,
                         uint chans,
                         uint chan_off,
                         int  w,
                         int  h,
                         float sx, float sy)
{
  int ix = int(floor(sx));
  int iy = int(floor(sy));
  float fx = sx - float(ix);
  float fy = sy - float(iy);
  float4 wx = cubic_w_(fx);
  float4 wy = cubic_w_(fy);
  float acc = 0.0;
  for (int dy = 0; dy < 4; ++dy) {
    int yy = clamp(iy - 1 + dy, 0, h - 1);
    float row = 0.0;
    for (int dx = 0; dx < 4; ++dx) {
      int xx = clamp(ix - 1 + dx, 0, w - 1);
      row += wx[dx] *
          float(buf[uint(yy) * stride + uint(xx) * chans + chan_off]);
    }
    acc += wy[dy] * row;
  }
  return acc;
}

kernel void nv12_to_planar_rgb_u8(
    device const uchar* y_buf       [[buffer(0)]],
    device const uchar* cbcr_buf    [[buffer(1)]],
    device uchar*       out_buf     [[buffer(2)]],
    constant uint4&     params      [[buffer(3)]],
    constant uint4&     params2     [[buffer(4)]],
    constant float4&    params3     [[buffer(5)]],
    constant uint&      use_bicubic [[buffer(6)]],
    uint2               gid         [[thread_position_in_grid]])
{
  uint in_w        = params.x;
  uint in_h        = params.y;
  uint y_stride    = params.z;
  uint cbcr_stride = params.w;
  uint out_w       = params2.x;
  uint out_h       = params2.y;
  float crop_left  = params3.x;
  float crop_top   = params3.y;
  float scale_x    = params3.z;
  float scale_y    = params3.w;
  if (gid.x >= out_w || gid.y >= out_h) { return; }

  float sx = crop_left + (float(gid.x) + 0.5) * scale_x - 0.5;
  float sy = crop_top  + (float(gid.y) + 0.5) * scale_y - 0.5;

  float y_v;
  float2 cbcr;
  float cfx = sx * 0.5;
  float cfy = sy * 0.5;
  uint Wc = in_w >> 1;
  uint Hc = in_h >> 1;

  if (use_bicubic != 0u) {
    y_v = bicubic_u8_(y_buf, y_stride, 1u, 0u,
                      int(in_w), int(in_h), sx, sy) * (1.0 / 255.0);
    float cb_v = bicubic_u8_(cbcr_buf, cbcr_stride, 2u, 0u,
                             int(Wc), int(Hc), cfx, cfy)
                 * (1.0 / 255.0);
    float cr_v = bicubic_u8_(cbcr_buf, cbcr_stride, 2u, 1u,
                             int(Wc), int(Hc), cfx, cfy)
                 * (1.0 / 255.0);
    cbcr = float2(cb_v, cr_v);
  } else {
    int ix0 = int(floor(sx));
    int iy0 = int(floor(sy));
    float fx = sx - float(ix0);
    float fy = sy - float(iy0);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    ix0 = clamp(ix0, 0, int(in_w) - 1);
    ix1 = clamp(ix1, 0, int(in_w) - 1);
    iy0 = clamp(iy0, 0, int(in_h) - 1);
    iy1 = clamp(iy1, 0, int(in_h) - 1);
    float y00 = float(y_buf[uint(iy0) * y_stride + uint(ix0)]);
    float y01 = float(y_buf[uint(iy0) * y_stride + uint(ix1)]);
    float y10 = float(y_buf[uint(iy1) * y_stride + uint(ix0)]);
    float y11 = float(y_buf[uint(iy1) * y_stride + uint(ix1)]);
    y_v = mix(mix(y00, y01, fx), mix(y10, y11, fx), fy)
          * (1.0 / 255.0);

    int cx0 = int(floor(cfx));
    int cy0 = int(floor(cfy));
    float cfx_frac = cfx - float(cx0);
    float cfy_frac = cfy - float(cy0);
    int cx1 = cx0 + 1;
    int cy1 = cy0 + 1;
    cx0 = clamp(cx0, 0, int(Wc) - 1);
    cx1 = clamp(cx1, 0, int(Wc) - 1);
    cy0 = clamp(cy0, 0, int(Hc) - 1);
    cy1 = clamp(cy1, 0, int(Hc) - 1);
    float2 c00 = float2(
        float(cbcr_buf[uint(cy0) * cbcr_stride + uint(cx0) * 2 + 0]),
        float(cbcr_buf[uint(cy0) * cbcr_stride + uint(cx0) * 2 + 1]))
        * (1.0 / 255.0);
    float2 c10 = float2(
        float(cbcr_buf[uint(cy0) * cbcr_stride + uint(cx1) * 2 + 0]),
        float(cbcr_buf[uint(cy0) * cbcr_stride + uint(cx1) * 2 + 1]))
        * (1.0 / 255.0);
    float2 c01 = float2(
        float(cbcr_buf[uint(cy1) * cbcr_stride + uint(cx0) * 2 + 0]),
        float(cbcr_buf[uint(cy1) * cbcr_stride + uint(cx0) * 2 + 1]))
        * (1.0 / 255.0);
    float2 c11 = float2(
        float(cbcr_buf[uint(cy1) * cbcr_stride + uint(cx1) * 2 + 0]),
        float(cbcr_buf[uint(cy1) * cbcr_stride + uint(cx1) * 2 + 1]))
        * (1.0 / 255.0);
    cbcr =
        mix(mix(c00, c10, cfx_frac), mix(c01, c11, cfx_frac), cfy_frac);
  }

  float cb = cbcr.x - 0.5;
  float cr = cbcr.y - 0.5;
  float r = y_v               + 1.5748 * cr;
  float g = y_v - 0.1873 * cb - 0.4681 * cr;
  float b = y_v + 1.8556 * cb;

  uint plane = out_w * out_h;
  uint idx   = gid.y * out_w + gid.x;
  out_buf[idx]              = uchar(clamp(r, 0.0, 1.0) * 255.0 + 0.5);
  out_buf[plane     + idx]  = uchar(clamp(g, 0.0, 1.0) * 255.0 + 0.5);
  out_buf[2 * plane + idx]  = uchar(clamp(b, 0.0, 1.0) * 255.0 + 0.5);
}
