// rife.metal -- the non-conv ops of RIFE's IFNet (Practical-RIFE 4.22-4.26;
// generative-models/rife/metal-rife.h). The convolutions run on the shared
// gather conv (dense_gemm.metal, conv3x3_gemm_*); everything around them is
// here, fused so a full-resolution intermediate is written only where the
// network needs one kept.
//
// LAYOUTS, all channel-LAST at the PADDED size H x W (the driver pads a
// frame with zeros on the right and bottom to a multiple of 128 / scale):
//   img   float4 [H*W]       a frame, rgb + 0, in [0, 1]
//   imgh  half4  [H*W]       the same, the Head's input
//   feat  half4  [H*W*F/4]   the Head's output, F = 4 or 8 channels
//   flow  float4 [H*W]       the running flow, (dx0, dy0, dx1, dy1) px
//   aux   half   [H*W*12]    the last block's mask logit, then its 8 feat
//                            channels, then 3 unused
//
// RESAMPLING IS TORCH'S. F.interpolate(bilinear, align_corners=False) at a
// scale factor: source = max((dst + 0.5) * r - 0.5, 0) with r = 1 / factor,
// the upper tap clamped to the last row -- no antialias, so a 16x
// downscale reads 2 x 2 pixels. grid_sample(bilinear, border,
// align_corners=True) after the warp's normalisation is a sample at
// (x + dx, y + dy) clamped into [0, W-1] x [0, H-1]; a tap past the last
// pixel weighs 0.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------
// A planar [3, h, w] frame into the padded float4 / half4 layouts, zeros
// beyond (h, w). u8 reads as v / 255, f32 as is.
//   0:src 1:img 2:imgh 3:h 4:w 5:H 6:W   grid {W, H}
template <typename S>
static inline void rife_frame_in_impl(const device S* src,
                                      device float4* img,
                                      device half4* imgh, int h, int w,
                                      int H, int W, uint2 gid, float k)
{
  if ((int)gid.x >= W || (int)gid.y >= H) { return; }
  const int x = (int)gid.x, y = (int)gid.y;
  float4 v = float4(0.0f);
  if (x < w && y < h) {
    const long plane = (long)h * w;
    const long o = (long)y * w + x;
    v = float4((float)src[o] * k, (float)src[plane + o] * k,
               (float)src[2 * plane + o] * k, 0.0f);
  }
  const long d = (long)y * W + x;
  img[d] = v;
  imgh[d] = half4(v);
}

kernel void rife_frame_in_u8(const device uchar* src [[buffer(0)]],
                             device float4* img      [[buffer(1)]],
                             device half4* imgh      [[buffer(2)]],
                             const constant int& h   [[buffer(3)]],
                             const constant int& w   [[buffer(4)]],
                             const constant int& H   [[buffer(5)]],
                             const constant int& W   [[buffer(6)]],
                             uint2 gid [[thread_position_in_grid]])
{
  rife_frame_in_impl<uchar>(src, img, imgh, h, w, H, W, gid, 1.0f / 255.0f);
}

kernel void rife_frame_in_f32(const device float* src [[buffer(0)]],
                              device float4* img      [[buffer(1)]],
                              device half4* imgh      [[buffer(2)]],
                              const constant int& h   [[buffer(3)]],
                              const constant int& w   [[buffer(4)]],
                              const constant int& H   [[buffer(5)]],
                              const constant int& W   [[buffer(6)]],
                              uint2 gid [[thread_position_in_grid]])
{
  rife_frame_in_impl<float>(src, img, imgh, h, w, H, W, gid, 1.0f);
}

// ---------------------------------------------------------------------
// The Head's ConvTranspose2d(k4, s2, p1), run as a 3x3 conv whose output
// channel p * F + c is output parity p = (Y % 2) * 2 + (X % 2) of channel
// c: scatter it to the full-resolution map.
//   0:par [(H/2)*(W/2), 4F] 1:feat [H*W*F] 2:H 3:W 4:F   grid {W, H}
kernel void rife_head_d2s(const device half* par [[buffer(0)]],
                          device half* feat      [[buffer(1)]],
                          const constant int& H  [[buffer(2)]],
                          const constant int& W  [[buffer(3)]],
                          const constant int& F  [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= W || (int)gid.y >= H) { return; }
  const int X = (int)gid.x, Y = (int)gid.y;
  const int p = (Y & 1) * 2 + (X & 1);
  const device half* s =
      par + ((long)(Y >> 1) * (W >> 1) + (X >> 1)) * 4 * F + p * F;
  device half* d = feat + ((long)Y * W + X) * F;
  for (int c = 0; c < F; ++c) { d[c] = s[c]; }
}

// ---------------------------------------------------------------------
// Bilinear helpers.

// torch's source index for align_corners=False, clamped below at 0.
static inline void rife_src_(int dst, float r, int n, thread int& i0,
                             thread int& i1, thread float& l1)
{
  float s = ((float)dst + 0.5f) * r - 0.5f;
  s = max(s, 0.0f);
  i0 = (int)s;
  i1 = i0 < n - 1 ? i0 + 1 : i0;
  l1 = s - (float)i0;
}

// A warp tap: bilinear at (x, y), border-clamped, in pixel units.
struct RifeTap {
  int   o00, o01, o10, o11;
  float w00, w01, w10, w11;
};

static inline RifeTap rife_warp_tap_(float x, float y, int H, int W)
{
  x = clamp(x, 0.0f, (float)(W - 1));
  y = clamp(y, 0.0f, (float)(H - 1));
  const int x0 = (int)floor(x), y0 = (int)floor(y);
  const float ax = x - (float)x0, ay = y - (float)y0;
  // A tap one past the last pixel weighs exactly 0 here (x == W - 1);
  // point it at the last pixel so the read stays inside the image.
  const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
  RifeTap t;
  t.o00 = y0 * W + x0;  t.o01 = y0 * W + x1;
  t.o10 = y1 * W + x0;  t.o11 = y1 * W + x1;
  t.w00 = (1.0f - ax) * (1.0f - ay);
  t.w01 = ax * (1.0f - ay);
  t.w10 = (1.0f - ax) * ay;
  t.w11 = ax * ay;
  return t;
}

static inline float3 rife_warp_img_(const device float4* img, RifeTap t)
{
  return img[t.o00].xyz * t.w00 + img[t.o01].xyz * t.w01 +
         img[t.o10].xyz * t.w10 + img[t.o11].xyz * t.w11;
}

// Four feature channels (quad q of F4) of a warp tap, as one vector a
// corner.
static inline float4 rife_warp_feat4_(const device half4* f, int F4, int q,
                                      RifeTap t)
{
  return float4(f[(long)t.o00 * F4 + q]) * t.w00 +
         float4(f[(long)t.o01 * F4 + q]) * t.w01 +
         float4(f[(long)t.o10 * F4 + q]) * t.w10 +
         float4(f[(long)t.o11 * F4 + q]) * t.w11;
}

// The channels an IFBlock reads at one FULL-resolution pixel, in the
// order its first conv was trained on:
//   block 0:  img0 3, img1 3, f0 F, f1 F, t
//   later:    warp(img0) 3, warp(img1) 3, warp(f0) F, warp(f1) F, t,
//             mask, feat 8, flow / s 4
// `v` holds 7 + 2F (+ 13) values; at most 40.
constexpr constant int kRifeMaxIn = 40;

static inline void rife_put4_(thread float* v, thread int& k, float4 q)
{
  v[k++] = q.x; v[k++] = q.y; v[k++] = q.z; v[k++] = q.w;
}

static inline void rife_block_px_(const device float4* img0,
                                  const device float4* img1,
                                  const device half4* f0,
                                  const device half4* f1,
                                  const device float4* flow,
                                  const device half4* aux, int H, int W,
                                  int F4, float t, int first, int y, int x,
                                  thread float* v)
{
  const long o = (long)y * W + x;
  int k = 0;
  if (first) {
    const float3 a = img0[o].xyz, b = img1[o].xyz;
    v[k++] = a.x; v[k++] = a.y; v[k++] = a.z;
    v[k++] = b.x; v[k++] = b.y; v[k++] = b.z;
    for (int q = 0; q < F4; ++q) { rife_put4_(v, k, float4(f0[o * F4 + q])); }
    for (int q = 0; q < F4; ++q) { rife_put4_(v, k, float4(f1[o * F4 + q])); }
    v[k++] = t;
    return;
  }
  const float4 fl = flow[o];
  const RifeTap t0 = rife_warp_tap_((float)x + fl.x, (float)y + fl.y, H, W);
  const RifeTap t1 = rife_warp_tap_((float)x + fl.z, (float)y + fl.w, H, W);
  const float3 a = rife_warp_img_(img0, t0), b = rife_warp_img_(img1, t1);
  v[k++] = a.x; v[k++] = a.y; v[k++] = a.z;
  v[k++] = b.x; v[k++] = b.y; v[k++] = b.z;
  for (int q = 0; q < F4; ++q) {
    rife_put4_(v, k, rife_warp_feat4_(f0, F4, q, t0));
  }
  for (int q = 0; q < F4; ++q) {
    rife_put4_(v, k, rife_warp_feat4_(f1, F4, q, t1));
  }
  v[k++] = t;
  // aux: mask, feat 0-2 | feat 3-6 | feat 7, 3 unused.
  const float4 m0 = float4(aux[o * 3]), m1 = float4(aux[o * 3 + 1]);
  const float m2 = (float)aux[o * 3 + 2].x;
  rife_put4_(v, k, m0);
  rife_put4_(v, k, m1);
  v[k++] = m2;
  rife_put4_(v, k, fl);
}

// ---------------------------------------------------------------------
// An IFBlock's input: its channels at full resolution (warped by the flow
// so far), resized by 1 / s, concatenated and zero-padded to Cp -- the
// [Hs * Ws, Cp] map its first conv gathers from. Neither the warped
// frames nor the concat exist at full resolution.
//   0:img0 1:img1 2:f0 3:f1 4:flow 5:aux 6:out 7:H 8:W 9:Hs 10:Ws 11:F
//   12:Cp 13:s 14:t 15:first   grid {Ws, Hs}
kernel void rife_pack(const device float4* img0 [[buffer(0)]],
                      const device float4* img1 [[buffer(1)]],
                      const device half4* f0    [[buffer(2)]],
                      const device half4* f1    [[buffer(3)]],
                      const device float4* flow [[buffer(4)]],
                      const device half4* aux   [[buffer(5)]],
                      device half4* out         [[buffer(6)]],
                      const constant int& H     [[buffer(7)]],
                      const constant int& W     [[buffer(8)]],
                      const constant int& Hs    [[buffer(9)]],
                      const constant int& Ws    [[buffer(10)]],
                      const constant int& F     [[buffer(11)]],
                      const constant int& Cp    [[buffer(12)]],
                      const constant float& s   [[buffer(13)]],
                      const constant float& t   [[buffer(14)]],
                      const constant int& first [[buffer(15)]],
                      uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= Ws || (int)gid.y >= Hs) { return; }
  const int lx = (int)gid.x, ly = (int)gid.y;
  int y0, y1, x0, x1;
  float ly1, lx1;
  // F.interpolate(scale_factor = 1 / s): r = 1 / (1 / s) = s.
  rife_src_(ly, s, H, y0, y1, ly1);
  rife_src_(lx, s, W, x0, x1, lx1);
  const float ly0 = 1.0f - ly1, lx0 = 1.0f - lx1;
  const int n = first ? 7 + 2 * F : 20 + 2 * F;
  // Tap by tap into one accumulator: four whole channel vectors live at
  // once would spill.
  float acc[kRifeMaxIn], v[kRifeMaxIn];
  for (int c = 0; c < n; ++c) { acc[c] = 0.0f; }
  const int ys[2] = {y0, y1}, xs[2] = {x0, x1};
  const float wy[2] = {ly0, ly1}, wx[2] = {lx0, lx1};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const float wt = wy[a] * wx[b];
      if (wt == 0.0f) { continue; }
      rife_block_px_(img0, img1, f0, f1, flow, aux, H, W, F / 4, t,
                     first, ys[a], xs[b], v);
      for (int c = 0; c < n; ++c) { acc[c] += wt * v[c]; }
    }
  }
  // torch: interpolate(flow) * 1. / scale -- divided after the resize.
  if (!first) {
    for (int c = n - 4; c < n; ++c) { acc[c] = acc[c] / s; }
  }
  for (int c = n; c < kRifeMaxIn; ++c) { acc[c] = 0.0f; }
  device half4* d = out + ((long)ly * Ws + lx) * (Cp / 4);
  for (int q = 0; q < Cp / 4; ++q) {
    const int c = 4 * q;
    d[q] = c < n ? half4(acc[c], acc[c + 1], acc[c + 2], acc[c + 3])
                 : half4(0.0h);
  }
}

// ---------------------------------------------------------------------
// An IFBlock's output. Its last layer, ConvTranspose2d(c, 52, k4, s2, p1)
// then PixelShuffle(2), ran as a 3x3 conv over the block's (Hs/4, Ws/4)
// map: each pixel (m, n) there becomes the 4 x 4 pixels (4m + a, 4n + b)
// of the 13-channel result at the block's input size, and the conv's
// output rows are laid out SUB-POSITION MAJOR -- row sub * KS + c is
// channel c at sub-position sub = a * 4 + b (with a = 2 py + sy: py the
// transposed conv's output parity, sy the pixel shuffle's), so the 13
// values a pixel reads are contiguous. KS is 13, or 16 (rows 13-15 zero)
// where the hardware conv wants a width of 64 and the read is 4 half4s.
// Upsampled by s to full resolution:
//   flow  = (first ? 0 : flow) + t[0:4] * s
//   mask  = t[4], feat = t[5:13]   -> aux
//   0:conv [(Hs/4)*(Ws/4), 16 * KS] 1:flow 2:aux 3:H 4:W 5:Hs 6:Ws 7:s
//   8:first 9:KS   grid {W, H}
static inline void rife_tmp_(const device half* conv, int Wq, int KS, int u,
                             int v, float w, thread float* tv)
{
  const int sub = (u & 3) * 4 + (v & 3);
  const long base = ((long)(u >> 2) * Wq + (v >> 2)) * 16 * KS + sub * KS;
  if (KS == 16) {
    const device half4* p = (const device half4*)(conv + base);
    const float4 a = float4(p[0]), b = float4(p[1]), c = float4(p[2]);
    const float d = (float)p[3].x;
    tv[0] += w * a.x;  tv[1] += w * a.y;  tv[2] += w * a.z;
    tv[3] += w * a.w;  tv[4] += w * b.x;  tv[5] += w * b.y;
    tv[6] += w * b.z;  tv[7] += w * b.w;  tv[8] += w * c.x;
    tv[9] += w * c.y;  tv[10] += w * c.z; tv[11] += w * c.w;
    tv[12] += w * d;
    return;
  }
  for (int c = 0; c < 13; ++c) { tv[c] += w * (float)conv[base + c]; }
}

kernel void rife_unpack(const device half* conv  [[buffer(0)]],
                        device float4* flow      [[buffer(1)]],
                        device half4* aux        [[buffer(2)]],
                        const constant int& H    [[buffer(3)]],
                        const constant int& W    [[buffer(4)]],
                        const constant int& Hs   [[buffer(5)]],
                        const constant int& Ws   [[buffer(6)]],
                        const constant float& s  [[buffer(7)]],
                        const constant int& first [[buffer(8)]],
                        const constant int& KS   [[buffer(9)]],
                        uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= W || (int)gid.y >= H) { return; }
  const int X = (int)gid.x, Y = (int)gid.y;
  // F.interpolate(scale_factor = s): r = 1 / s.
  const float r = 1.0f / s;
  int u0, u1, v0, v1;
  float lu1, lv1;
  rife_src_(Y, r, Hs, u0, u1, lu1);
  rife_src_(X, r, Ws, v0, v1, lv1);
  const float lu0 = 1.0f - lu1, lv0 = 1.0f - lv1;
  const int Wq = Ws >> 2;
  float tv[13];
  for (int c = 0; c < 13; ++c) { tv[c] = 0.0f; }
  rife_tmp_(conv, Wq, KS, u0, v0, lu0 * lv0, tv);
  if (lv1 != 0.0f) { rife_tmp_(conv, Wq, KS, u0, v1, lu0 * lv1, tv); }
  if (lu1 != 0.0f) {
    rife_tmp_(conv, Wq, KS, u1, v0, lu1 * lv0, tv);
    if (lv1 != 0.0f) { rife_tmp_(conv, Wq, KS, u1, v1, lu1 * lv1, tv); }
  }
  const long o = (long)Y * W + X;
  const float4 fd = float4(tv[0], tv[1], tv[2], tv[3]) * s;
  flow[o] = first ? fd : flow[o] + fd;
  aux[o * 3]     = half4(tv[4], tv[5], tv[6], tv[7]);
  aux[o * 3 + 1] = half4(tv[8], tv[9], tv[10], tv[11]);
  aux[o * 3 + 2] = half4(tv[12], 0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------
// The result: both frames warped by the final flow, blended by the
// sigmoid of the mask, cut back to (h, w) and written planar [3, h, w] --
// f32 as computed, u8 rounded to nearest and clamped.
//   0:img0 1:img1 2:flow 3:aux 4:out 5:h 6:w 7:H 8:W   grid {w, h}
static inline float3 rife_merge_px_(const device float4* img0,
                                    const device float4* img1,
                                    const device float4* flow,
                                    const device half* aux, int H, int W,
                                    int x, int y)
{
  const long o = (long)y * W + x;
  const float4 fl = flow[o];
  const float3 a = rife_warp_img_(
      img0, rife_warp_tap_((float)x + fl.x, (float)y + fl.y, H, W));
  const float3 b = rife_warp_img_(
      img1, rife_warp_tap_((float)x + fl.z, (float)y + fl.w, H, W));
  const float m = 1.0f / (1.0f + exp(-(float)aux[o * 12]));
  return a * m + b * (1.0f - m);
}

kernel void rife_merge_f32(const device float4* img0 [[buffer(0)]],
                           const device float4* img1 [[buffer(1)]],
                           const device float4* flow [[buffer(2)]],
                           const device half* aux    [[buffer(3)]],
                           device float* out         [[buffer(4)]],
                           const constant int& h     [[buffer(5)]],
                           const constant int& w     [[buffer(6)]],
                           const constant int& H     [[buffer(7)]],
                           const constant int& W     [[buffer(8)]],
                           uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= w || (int)gid.y >= h) { return; }
  const int x = (int)gid.x, y = (int)gid.y;
  const float3 v = rife_merge_px_(img0, img1, flow, aux, H, W, x, y);
  const long plane = (long)h * w, o = (long)y * w + x;
  out[o] = v.x;
  out[plane + o] = v.y;
  out[2 * plane + o] = v.z;
}

kernel void rife_merge_u8(const device float4* img0 [[buffer(0)]],
                          const device float4* img1 [[buffer(1)]],
                          const device float4* flow [[buffer(2)]],
                          const device half* aux    [[buffer(3)]],
                          device uchar* out         [[buffer(4)]],
                          const constant int& h     [[buffer(5)]],
                          const constant int& w     [[buffer(6)]],
                          const constant int& H     [[buffer(7)]],
                          const constant int& W     [[buffer(8)]],
                          uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= w || (int)gid.y >= h) { return; }
  const int x = (int)gid.x, y = (int)gid.y;
  const float3 v =
      clamp(rife_merge_px_(img0, img1, flow, aux, H, W, x, y) * 255.0f +
                0.5f, 0.0f, 255.0f);
  const long plane = (long)h * w, o = (long)y * w + x;
  out[o] = (uchar)v.x;
  out[plane + o] = (uchar)v.y;
  out[2 * plane + o] = (uchar)v.z;
}
