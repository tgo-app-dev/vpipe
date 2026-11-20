// relayvsr.metal -- RelayVSR's own small ops (generative-models/relayvsr/):
// the frame upload, the keyframe DiT's source rows, patch rows, f32
// modulation, RoPE and one-step latent, and the decoder's condition rows,
// per-head norms, RoPE, SwiGLU and the frame it writes. The GEMMs, the
// gather convs and the windowed attention are the shared kernels.
//
// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the bf16
// library. Math is f32 throughout; where the reference rounds an
// intermediate to its dtype (a tensor it materializes), so does this.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
typedef VPIPE_ELT elt;

// ---- the frame ----------------------------------------------------------
//
// A planar [3, h, w] frame onto the padded grid as CHANNEL-LAST
// [ph * pw, 3] in [-1, 1] -- the layout the decoder's gather conv reads,
// and the one the bilinear and bicubic samplers below index. Padding
// REPLICATES the last row and column (prepare_lr's F.pad(mode=
// "replicate")).
//
// The reference converts a u8 frame as bf16(bf16(u) / 127.5) - 1, in its
// dtype; a u8 frame here takes the same two roundings in this library's.
// A float frame (video-to-rgb's 0..1) is 2 v - 1, rounded once.
//   0:src 1:out 2:h 3:w 4:ph 5:pw 6:is_float (0 u8, 1 f32, 2 f16)
//   grid (pw, ph)
kernel void rvsr_upload(
    const device uchar* src [[buffer(0)]],
    device elt*         out [[buffer(1)]],
    constant int&       h   [[buffer(2)]],
    constant int&       w   [[buffer(3)]],
    constant int&       ph  [[buffer(4)]],
    constant int&       pw  [[buffer(5)]],
    constant int&       fmt [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int x = (int)gid.x, y = (int)gid.y;
  if (x >= pw || y >= ph) { return; }
  const int sy = min(y, h - 1), sx = min(x, w - 1);
  for (int c = 0; c < 3; ++c) {
    const uint i = ((uint)c * (uint)h + (uint)sy) * (uint)w + (uint)sx;
    float v;
    if (fmt == 0) {
      const elt t = elt((float)src[i] / 127.5f);
      v = (float)elt((float)t - 1.0f);
    } else if (fmt == 1) {
      v = 2.0f * ((const device float*)src)[i] - 1.0f;
    } else {
      v = 2.0f * (float)((const device half*)src)[i] - 1.0f;
    }
    out[((uint)y * (uint)pw + (uint)x) * 3u + (uint)c] = elt(v);
  }
}

// ---- the keyframe DiT ---------------------------------------------------
//
// THE SOURCE ROWS: lq_proj's first two steps in one, on the latent grid
// (1/4 of the frame at 4x, 1/8 at 2x). At 4x the frame is upsampled 4x
// bilinearly (torch's align_corners=False rule: source max(0, (d + 0.5) /
// 4 - 0.5), the far neighbour clamped) and pixel-unshuffled by 16, so row
// (ty, tx) of the [ph/4 * pw/4, 768] result holds the 16 x 16 upsampled
// block at (16 ty, 16 tx), channel c * 256 + 16 dy + dx -- the 4ph x 4pw
// upsample itself is never written; each value is rounded to the element
// type once, as the reference's upsampled tensor is. At 2x the frame
// itself is pixel-unshuffled by 8 (LightX2V's native_x2_input): row (ty,
// tx) of [ph/8 * pw/8, 192] holds the 8 x 8 block at (8 ty, 8 tx),
// channel c * 64 + 8 dy + dx, copied.
//   0:lr[ph*pw,3] 1:out[rows,C] 2:ph 3:pw 4:scale (4 | 2)
//   grid (C, rows), C = 768 at 4x, 192 at 2x
kernel void rvsr_lq_rows(
    const device elt* lr    [[buffer(0)]],
    device elt*       out   [[buffer(1)]],
    constant int&     ph    [[buffer(2)]],
    constant int&     pw    [[buffer(3)]],
    constant int&     scale [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ch = (int)gid.x, row = (int)gid.y;
  if (scale == 2) {
    const int tw = pw / 8;
    if (ch >= 192 || row >= (ph / 8) * tw) { return; }
    const int c = ch / 64, dy = (ch / 8) % 8, dx = ch % 8;
    const int Y = (row / tw) * 8 + dy, X = (row % tw) * 8 + dx;
    out[(uint)row * 192u + (uint)ch] =
        lr[((uint)Y * (uint)pw + (uint)X) * 3u + (uint)c];
    return;
  }
  const int tw = pw / 4;
  if (ch >= 768 || row >= (ph / 4) * tw) { return; }
  const int c = ch / 256, dy = (ch / 16) % 16, dx = ch % 16;
  const int Y = (row / tw) * 16 + dy, X = (row % tw) * 16 + dx;
  const float sy = max(0.25f * ((float)Y + 0.5f) - 0.5f, 0.0f);
  const float sx = max(0.25f * ((float)X + 0.5f) - 0.5f, 0.0f);
  const int y0 = (int)sy, x0 = (int)sx;
  const int y1 = y0 + (y0 < ph - 1 ? 1 : 0);
  const int x1 = x0 + (x0 < pw - 1 ? 1 : 0);
  const float ly = sy - (float)y0, lx = sx - (float)x0;
  auto at = [&](int yy, int xx) {
    return (float)lr[((uint)yy * (uint)pw + (uint)xx) * 3u + (uint)c];
  };
  const float v = (1.0f - ly) * ((1.0f - lx) * at(y0, x0) + lx * at(y0, x1)) +
                  ly * ((1.0f - lx) * at(y1, x0) + lx * at(y1, x1));
  out[(uint)row * 768u + (uint)ch] = elt(v);
}

// lq_proj's RMSNorm2d over the channels of a channel-last row, its weight,
// then SiLU: y = elt(x rsqrt(mean x^2 + eps) w), out = elt(silu(y)) -- the
// norm's f32 result is cast back before the activation, as RMSNorm2d
// returns it. One threadgroup (256) per row. In place is fine.
//   0:x[rows,C] 1:w[C] 2:out 3:C 4:eps   grid (256, rows)
kernel void rvsr_rms_silu(
    const device elt* x   [[buffer(0)]],
    const device elt* g   [[buffer(1)]],
    device elt*       out [[buffer(2)]],
    constant int&     C   [[buffer(3)]],
    constant float&   eps [[buffer(4)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const device elt* xr = x + (ulong)tid.y * (ulong)C;
  device elt* outr = out + (ulong)tid.y * (ulong)C;
  float s = 0.0f;
  for (int i = (int)ltid.x; i < C; i += 256) {
    const float v = (float)xr[i];
    s += v * v;
  }
  s = simd_sum(s);
  threadgroup float part[8];
  if (sl == 0) { part[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = sl < 8 ? part[sl] : 0.0f;
  t = simd_sum(t);
  const float inv = rsqrt(t / (float)C + eps);
  for (int i = (int)ltid.x; i < C; i += 256) {
    const float y = (float)elt((float)xr[i] * inv * (float)g[i]);
    outr[i] = elt(y / (1.0f + exp(-y)));
  }
}

// The noise as patch rows: row (y, x) of [(H4/2) * (W4/2), C * 4] holds
// noise[c, 2y + a, 2x + b] at column 4c + 2a + b -- the Conv3d (1, 2, 2)
// patch embedding's weight, flattened, is in that order. The noise is
// rounded to the element type, as the reference feeds the DiT
// `latents.to(compute_dtype)`.
//   0:noise f32 [C,H4,W4] 1:out 2:C 3:H4 4:W4   grid (4C, rows)
kernel void rvsr_patch_rows(
    const device float* noise [[buffer(0)]],
    device elt*         out   [[buffer(1)]],
    constant int&       C     [[buffer(2)]],
    constant int&       H4    [[buffer(3)]],
    constant int&       W4    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int col = (int)gid.x, row = (int)gid.y;
  const int tw = W4 / 2;
  if (col >= 4 * C || row >= (H4 / 2) * tw) { return; }
  const int c = col / 4, a = (col / 2) % 2, b = col % 2;
  const int Y = (row / tw) * 2 + a, X = (row % tw) * 2 + b;
  out[(uint)row * (uint)(4 * C) + (uint)col] =
      elt(noise[((uint)c * (uint)H4 + (uint)Y) * (uint)W4 + (uint)X]);
}

// LayerNorm (no affine), then the block's modulation in F32 -- the
// reference adds its modulation table to the timestep projection in f32
// and keeps it there: out = elt(LN(x) (1 + scale) + shift). Two passes
// over the row for the variance, as torch's layer_norm computes it. One
// threadgroup (256) per row.
//   0:x[rows,D] 1:scale f32[D] 2:shift f32[D] 3:out 4:D 5:eps
//   grid (256, rows)
kernel void rvsr_ln_mod(
    const device elt*   x     [[buffer(0)]],
    const device float* scale [[buffer(1)]],
    const device float* shift [[buffer(2)]],
    device elt*         out   [[buffer(3)]],
    constant int&       D     [[buffer(4)]],
    constant float&     eps   [[buffer(5)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const device elt* xr = x + (ulong)tid.y * (ulong)D;
  device elt* outr = out + (ulong)tid.y * (ulong)D;
  threadgroup float part[8];
  float s = 0.0f;
  for (int i = (int)ltid.x; i < D; i += 256) { s += (float)xr[i]; }
  s = simd_sum(s);
  if (sl == 0) { part[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = sl < 8 ? part[sl] : 0.0f;
  const float mean = simd_sum(t) / (float)D;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (int i = (int)ltid.x; i < D; i += 256) {
    const float d = (float)xr[i] - mean;
    q += d * d;
  }
  q = simd_sum(q);
  if (sl == 0) { part[sg] = q; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  t = sl < 8 ? part[sl] : 0.0f;
  const float inv = rsqrt(simd_sum(t) / (float)D + eps);
  for (int i = (int)ltid.x; i < D; i += 256) {
    const float ln = ((float)xr[i] - mean) * inv;
    outr[i] = elt(ln * (1.0f + scale[i]) + shift[i]);
  }
}

// LayerNorm WITH affine, f32 math (FP32LayerNorm: the block's norm2),
// two-pass. 0:x 1:w 2:b 3:out 4:D 5:eps   grid (256, rows)
kernel void rvsr_ln_affine(
    const device elt* x   [[buffer(0)]],
    const device elt* gw  [[buffer(1)]],
    const device elt* gb  [[buffer(2)]],
    device elt*       out [[buffer(3)]],
    constant int&     D   [[buffer(4)]],
    constant float&   eps [[buffer(5)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const device elt* xr = x + (ulong)tid.y * (ulong)D;
  device elt* outr = out + (ulong)tid.y * (ulong)D;
  threadgroup float part[8];
  float s = 0.0f;
  for (int i = (int)ltid.x; i < D; i += 256) { s += (float)xr[i]; }
  s = simd_sum(s);
  if (sl == 0) { part[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = sl < 8 ? part[sl] : 0.0f;
  const float mean = simd_sum(t) / (float)D;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = 0.0f;
  for (int i = (int)ltid.x; i < D; i += 256) {
    const float d = (float)xr[i] - mean;
    q += d * d;
  }
  q = simd_sum(q);
  if (sl == 0) { part[sg] = q; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  t = sl < 8 ? part[sl] : 0.0f;
  const float inv = rsqrt(simd_sum(t) / (float)D + eps);
  for (int i = (int)ltid.x; i < D; i += 256) {
    outr[i] = elt(((float)xr[i] - mean) * inv * (float)gw[i] + (float)gb[i]);
  }
}

// x += gate * sub with an F32 gate [D] broadcast over the rows, in place:
// x = elt(float(x) + gate float(sub)).   0:x 1:gate 2:sub 3:D 4:total
kernel void rvsr_gated_res(
    device elt*         x    [[buffer(0)]],
    const device float* gate [[buffer(1)]],
    const device elt*   sub  [[buffer(2)]],
    constant int&       D    [[buffer(3)]],
    constant uint&      n    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  x[gid] = elt((float)x[gid] + gate[gid % (uint)D] * (float)sub[gid]);
}

// THE KEYFRAME DiT's ROTARY EMBEDDING over one key/query plane, diffusers'
// Wan rule: head dim 128 split t 44 | h 42 | w 42, adjacent pairs, the
// tables f32 [1024, 128] (angles in f64, cos/sin cast to f32 -- never to the
// model dtype), a pair (2i, 2i + 1) rotated by cos[.][2i] and sin[.][2i+1]
// at its axis's position: the plane's temporal position `t`, the token's
// row, its column. Computed in f32 and rounded once into `out`, as the
// reference writes the rotated tensor in its dtype.
//   0:x 1:out 2:cos 3:sin 4:ld_x 5:ld_out 6:W (token columns) 7:t
//   8:heads   grid (64, tokens, heads)
kernel void rvsr_rope_wan(
    const device elt*   x    [[buffer(0)]],
    device elt*         out  [[buffer(1)]],
    const device float* tcos [[buffer(2)]],
    const device float* tsin [[buffer(3)]],
    constant int&       ldx  [[buffer(4)]],
    constant int&       ldo  [[buffer(5)]],
    constant int&       W    [[buffer(6)]],
    constant int&       t    [[buffer(7)]],
    constant int&       NH   [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x, tok = (int)gid.y, hh = (int)gid.z;
  if (i >= 64 || hh >= NH) { return; }
  const int j = 2 * i;
  const int pos = j < 44 ? t : (j < 86 ? tok / W : tok % W);
  const float c = tcos[pos * 128 + j];
  const float s = tsin[pos * 128 + j + 1];
  const ulong xb = (ulong)tok * (ulong)ldx + (ulong)hh * 128u + (ulong)j;
  const ulong ob = (ulong)tok * (ulong)ldo + (ulong)hh * 128u + (ulong)j;
  const float x1 = (float)x[xb], x2 = (float)x[xb + 1];
  out[ob]     = elt(x1 * c - x2 * s);
  out[ob + 1] = elt(x1 * s + x2 * c);
}

// THE ONE STEP. The head's patch rows [N, 4C] un-patchified onto the
// latent grid and taken from the noise: x0 = noise - v (FlowMatch Euler
// from sigma 1 to 0, in f32 -- the scheduler steps an f32 sample).
//   0:v 1:noise f32[C,H4,W4] 2:out f32[C,H4,W4] 3:C 4:H4 5:W4
//   grid (W4, H4, C)
kernel void rvsr_x0(
    const device elt*   v     [[buffer(0)]],
    const device float* noise [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant int&       C     [[buffer(3)]],
    constant int&       H4    [[buffer(4)]],
    constant int&       W4    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int X = (int)gid.x, Y = (int)gid.y, c = (int)gid.z;
  if (X >= W4 || Y >= H4 || c >= C) { return; }
  const int row = (Y / 2) * (W4 / 2) + X / 2;
  const int col = ((Y % 2) * 2 + (X % 2)) * C + c;
  const uint li = ((uint)c * (uint)H4 + (uint)Y) * (uint)W4 + (uint)X;
  out[li] = noise[li] - (float)v[(uint)row * (uint)(4 * C) + (uint)col];
}

// ---- the decoder --------------------------------------------------------
//
// An f32 latent [C, h, w] as channel-last rows [h * w, C] in the element
// type (the reference hands the decoder `latent.to(dtype)`).
//   0:lat 1:out 2:C 3:n   grid (C, n)
kernel void rvsr_cond_rows(
    const device float* lat [[buffer(0)]],
    device elt*         out [[buffer(1)]],
    constant int&       C   [[buffer(2)]],
    constant int&       n   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, p = (int)gid.y;
  if (c >= C || p >= n) { return; }
  out[(uint)p * (uint)C + (uint)c] = elt(lat[(uint)c * (uint)n + (uint)p]);
}

// RMSNorm over a row (its weight), rounded, then a type embedding added
// and rounded again: elt(elt(rms(x) w) + e) -- the encoder's norm, then
// `tokens + video_type_embedding` on the normed tensor. One threadgroup
// (256) per row; out may alias x.
//   0:x 1:w 2:e 3:out 4:D 5:eps   grid (256, rows)
kernel void rvsr_rms_embed(
    const device elt* x   [[buffer(0)]],
    const device elt* g   [[buffer(1)]],
    const device elt* e   [[buffer(2)]],
    device elt*       out [[buffer(3)]],
    constant int&     D   [[buffer(4)]],
    constant float&   eps [[buffer(5)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const device elt* xr = x + (ulong)tid.y * (ulong)D;
  device elt* outr = out + (ulong)tid.y * (ulong)D;
  float s = 0.0f;
  for (int i = (int)ltid.x; i < D; i += 256) {
    const float v = (float)xr[i];
    s += v * v;
  }
  s = simd_sum(s);
  threadgroup float part[8];
  if (sl == 0) { part[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = sl < 8 ? part[sl] : 0.0f;
  t = simd_sum(t);
  const float inv = rsqrt(t / (float)D + eps);
  for (int i = (int)ltid.x; i < D; i += 256) {
    const float y = (float)elt((float)xr[i] * inv * (float)g[i]);
    outr[i] = elt(y + (float)e[i]);
  }
}

// RMSNorm over each 64-wide HEAD of a row (the decoder's key_norm and
// value_norm, one weight [64] shared by the heads), in place. One
// simdgroup per (row, head).
//   0:x 1:w 2:ld 3:heads 4:eps   grid (32, rows, heads)
kernel void rvsr_rms_heads64(
    device elt*       x   [[buffer(0)]],
    const device elt* g   [[buffer(1)]],
    constant int&     ld  [[buffer(2)]],
    constant int&     NH  [[buffer(3)]],
    constant float&   eps [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int l = (int)gid.x, row = (int)gid.y, hh = (int)gid.z;
  if (hh >= NH) { return; }
  device elt* p = x + (ulong)row * (ulong)ld + (ulong)hh * 64u;
  const float a = (float)p[l], b = (float)p[l + 32];
  const float inv = rsqrt(simd_sum(a * a + b * b) / 64.0f + eps);
  p[l]      = elt(a * inv * (float)g[l]);
  p[l + 32] = elt(b * inv * (float)g[l + 32]);
}

// THE DECODER's ROTARY EMBEDDING: head dim 64 split t 16 | h 24 | w 24,
// adjacent pairs. The caller's tables carry one (cos, sin) per pair per
// position of each axis, already rounded to the element type (the
// reference builds its tables in the model dtype): `tab_t` [8] at this
// plane's temporal position, `tab_h` [rows, 12], `tab_w` [cols, 12], each
// entry a float2 (cos, sin). f32 math, one rounding into `out`.
//   0:x 1:out 2:tab_t 3:tab_h 4:tab_w 5:ld_x 6:ld_out 7:W 8:heads
//   grid (32, tokens, heads)
kernel void rvsr_rope3d(
    const device elt*    x     [[buffer(0)]],
    device elt*          out   [[buffer(1)]],
    const device float2* tab_t [[buffer(2)]],
    const device float2* tab_h [[buffer(3)]],
    const device float2* tab_w [[buffer(4)]],
    constant int&        ldx   [[buffer(5)]],
    constant int&        ldo   [[buffer(6)]],
    constant int&        W     [[buffer(7)]],
    constant int&        NH    [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x, tok = (int)gid.y, hh = (int)gid.z;
  if (i >= 32 || hh >= NH) { return; }
  float2 cs;
  if (i < 8) {
    cs = tab_t[i];
  } else if (i < 20) {
    cs = tab_h[(tok / W) * 12 + (i - 8)];
  } else {
    cs = tab_w[(tok % W) * 12 + (i - 20)];
  }
  const ulong xb = (ulong)tok * (ulong)ldx + (ulong)hh * 64u + (ulong)(2 * i);
  const ulong ob = (ulong)tok * (ulong)ldo + (ulong)hh * 64u + (ulong)(2 * i);
  const float e = (float)x[xb], o = (float)x[xb + 1];
  out[ob]     = elt(e * cs.x - o * cs.y);
  out[ob + 1] = elt(e * cs.y + o * cs.x);
}

// SwiGLU over a fused [rows, 2F] gate|up: out[r, j] = silu(g) * u, gate the
// FIRST half (the reference chunks gate_up_proj's output gate, up).
//   0:gu 1:out 2:F 3:total (rows * F)
kernel void rvsr_swiglu(
    const device elt* gu  [[buffer(0)]],
    device elt*       out [[buffer(1)]],
    constant int&     F   [[buffer(2)]],
    constant uint&    n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  const uint r = gid / (uint)F, j = gid % (uint)F;
  const float g = (float)gu[r * (uint)(2 * F) + j];
  const float u = (float)gu[r * (uint)(2 * F) + (uint)F + j];
  out[gid] = elt(g / (1.0f + exp(-g)) * u);
}

// SiLU in place: x = elt(x sigmoid(x)) (the decoder's spatial head).
//   0:x 1:n
kernel void rvsr_silu(
    device elt*    x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  const float v = (float)x[gid];
  x[gid] = elt(v / (1.0f + exp(-v)));
}

// One bicubic tap row, torch's upsample_bicubic2d (A = -0.75): x clamped
// to the frame, coefficients from the fractional offset.
static inline float rvsr_cubic(float x0, float x1, float x2, float x3,
                               float t)
{
  constexpr float A = -0.75f;
  const float a = t + 1.0f;
  const float c0 = ((A * a - 5.0f * A) * a + 8.0f * A) * a - 4.0f * A;
  const float c1 = ((A + 2.0f) * t - (A + 3.0f)) * t * t + 1.0f;
  const float b = 1.0f - t;
  const float c2 = ((A + 2.0f) * b - (A + 3.0f)) * b * b + 1.0f;
  const float d = b + 1.0f;
  const float c3 = ((A * d - 5.0f * A) * d + 8.0f * A) * d - 4.0f * A;
  return x0 * c0 + x1 * c1 + x2 * c2 + x3 * c3;
}

// THE FRAME OUT: the residual rows [n, 3 * 256] pixel-shuffled by 16 onto
// a `scale` x (4 or 2) bicubic upsample of the padded frame (both in the
// element type and summed in it, as the reference adds two tensors of its
// dtype), cut to the frame's own sh x sw and written planar: u8 as the
// release writes it -- (v + 1) 127.5 clamped to [0, 255] and TRUNCATED
// (`.to(torch.uint8)`) -- or f32 in [0, 1] as (v + 1) / 2, unclamped. The
// residual's grid is 1/16 of the output at either scale.
//   0:res 1:lr[ph*pw,3] 2:out 3:ph 4:pw 5:H (=sh) 6:W (=sw) 7:f32 out
//   8:scale (4 | 2)
//   grid (W, H, 3)
kernel void rvsr_frame_out(
    const device elt* res [[buffer(0)]],
    const device elt* lr  [[buffer(1)]],
    device uchar*     out [[buffer(2)]],
    constant int&     ph  [[buffer(3)]],
    constant int&     pw  [[buffer(4)]],
    constant int&     H   [[buffer(5)]],
    constant int&     W   [[buffer(6)]],
    constant int&     f32 [[buffer(7)]],
    constant int&     scale [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int X = (int)gid.x, Y = (int)gid.y, c = (int)gid.z;
  if (X >= W || Y >= H || c >= 3) { return; }
  // torch's source index for a given scale_factor: (d + 0.5) / s - 0.5
  // (1/4 and 1/2 are exact).
  const float inv = scale == 2 ? 0.5f : 0.25f;
  const float ry = inv * ((float)Y + 0.5f) - 0.5f;
  const float rx = inv * ((float)X + 0.5f) - 0.5f;
  const int iy = (int)floor(ry), ix = (int)floor(rx);
  const float ty = ry - (float)iy, tx = rx - (float)ix;
  float rows[4];
  for (int k = 0; k < 4; ++k) {
    const int yy = clamp(iy - 1 + k, 0, ph - 1);
    float v[4];
    for (int m = 0; m < 4; ++m) {
      const int xx = clamp(ix - 1 + m, 0, pw - 1);
      v[m] = (float)lr[((uint)yy * (uint)pw + (uint)xx) * 3u + (uint)c];
    }
    rows[k] = rvsr_cubic(v[0], v[1], v[2], v[3], tx);
  }
  const float bic = (float)elt(rvsr_cubic(rows[0], rows[1], rows[2], rows[3],
                                          ty));
  const int tw = (scale == 2 ? 2 : 4) * pw / 16;
  const uint row = (uint)(Y / 16) * (uint)tw + (uint)(X / 16);
  const uint ch = (uint)c * 256u + (uint)(Y % 16) * 16u + (uint)(X % 16);
  const float v = (float)elt(bic + (float)res[row * 768u + ch]);
  const uint o = ((uint)c * (uint)H + (uint)Y) * (uint)W + (uint)X;
  if (f32 != 0) {
    ((device float*)out)[o] = 0.5f * (v + 1.0f);
  } else {
    out[o] = (uchar)clamp((v + 1.0f) * 127.5f, 0.0f, 255.0f);
  }
}
