// minimax_h3_regenerate.metal -- the kernels Open-H3-Regenerate-2K needs
// beside MiniMax-H3's own: the latent upscaler's (generative-models/
// minimax-h3/metal-minimax-h3-latent-upscaler.h) and the LMS Sharp curve
// branch on the DiT's AdaLN modulation.
//
// The upscaler's activations are channel-last per latent FRAME, frames
// contiguous: [T, rows (= H * W), C]. Its 3x3x3 convs are the Wan VAE's
// 27-tap gather + GEMM (llm_elementwise's im2col_hwc_3x3x3_tiled_f16), its
// GroupNorm statistics the two-pass group_norm_chan_stats_f16 /
// group_norm_group_stats_f32 over the whole clip; what is here is what no
// other net in the tree has.
//
// Element (storage) type: half by default; -DVPIPE_ELT=bfloat builds the
// bf16 variant (minimax_h3_regenerate_bf16), which the DiT's curve add
// needs. Every reduction and every parameter is f32.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// GroupNorm APPLY with a per-channel affine and an optional SiLU:
//
//   out = act((x - mean_g) * inv_g * a[c] + b[c])
//
// `stats` is group_norm_group_stats_f32's [G][mean, inv]. The affine is
// the norm's own gamma / beta, or -- the upscaler's modulated norm --
// gamma (1 + s) and beta (1 + s) + shift folded on the host, so a
// modulated norm and its SiLU are one pass.
//   0:x[rows,C] 1:stats[2G] 2:a[C] 3:b[C] 4:out 5:C 6:G 7:silu
//   grid {C, rows}
kernel void h3up_group_norm_apply_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device float*     stats [[buffer(1)]],
    const device float*     a     [[buffer(2)]],
    const device float*     b     [[buffer(3)]],
    device VPIPE_ELT*       out   [[buffer(4)]],
    constant int&           C     [[buffer(5)]],
    constant int&           G     [[buffer(6)]],
    constant int&           silu  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  if (c >= C) { return; }
  const long idx = (long)gid.y * (long)C + (long)c;
  const int  g   = c / (C / G);
  float y = (float(x[idx]) - stats[2 * g]) * stats[2 * g + 1] * a[c] + b[c];
  if (silu != 0) { y = y / (1.0f + exp(-y)); }
  out[idx] = VPIPE_ELT(y);
}

// Depthwise convolution along TIME, kernel K (odd), zero padding K/2 at
// both ends -- the upscaler's TemporalConv (a Conv3d of kernel (K, 1, 1)
// with groups = C):
//
//   out[t, r, c] = b[c] + sum_k w[c][k] * x[t + k - K/2, r, c]
//
//   0:x[T,R,C] 1:w[C,K] 2:b[C] 3:out 4:T 5:R 6:C 7:K
//   grid {C, R, T}
kernel void h3up_dwconv_t_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    const device float*     w   [[buffer(1)]],
    const device float*     b   [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&           T   [[buffer(4)]],
    constant int&           R   [[buffer(5)]],
    constant int&           C   [[buffer(6)]],
    constant int&           K   [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, r = (int)gid.y, t = (int)gid.z;
  if (c >= C || r >= R || t >= T) { return; }
  const int h = K / 2;
  const long plane = (long)R * (long)C;
  float acc = b[c];
  for (int k = 0; k < K; ++k) {
    const int s = t + k - h;
    if (s < 0 || s >= T) { continue; }
    acc += w[c * K + k] * float(x[(long)s * plane + (long)r * C + c]);
  }
  out[(long)t * plane + (long)r * C + c] = VPIPE_ELT(acc);
}

// Bilinear resize of every frame, channel-last -- torch's trilinear
// interpolate with align_corners=False when the frame count is unchanged
// (the temporal coordinate then lands on its own frame with weight 1).
// A source coordinate is (dst + 0.5) * scale - 0.5, clamped at 0, with
// scale = in / out as torch computes it for a `size=` call; the upper
// neighbour clamps to the last row / column.
//   0:x[T,h*w,C] 1:out[T,H*W,C] 2:h 3:w 4:H 5:W 6:C 7:sy 8:sx
//   grid {C, H*W, T}
kernel void h3up_resize_bilinear_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           h   [[buffer(2)]],
    constant int&           w   [[buffer(3)]],
    constant int&           H   [[buffer(4)]],
    constant int&           W   [[buffer(5)]],
    constant int&           C   [[buffer(6)]],
    constant float&         sy  [[buffer(7)]],
    constant float&         sx  [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int p = (int)gid.y;
  const int t = (int)gid.z;
  if (c >= C || p >= H * W) { return; }
  const int oy = p / W, ox = p % W;
  float fy = sy * ((float)oy + 0.5f) - 0.5f;
  float fx = sx * ((float)ox + 0.5f) - 0.5f;
  fy = max(fy, 0.0f);
  fx = max(fx, 0.0f);
  const int y0 = (int)fy, x0 = (int)fx;
  const int y1 = y0 + (y0 < h - 1 ? 1 : 0);
  const int x1 = x0 + (x0 < w - 1 ? 1 : 0);
  const float ly = fy - (float)y0, lx = fx - (float)x0;
  const float hy = 1.0f - ly, hx = 1.0f - lx;
  const device VPIPE_ELT* f = x + (long)t * (long)h * w * C;
  const float v00 = float(f[((long)y0 * w + x0) * C + c]);
  const float v01 = float(f[((long)y0 * w + x1) * C + c]);
  const float v10 = float(f[((long)y1 * w + x0) * C + c]);
  const float v11 = float(f[((long)y1 * w + x1) * C + c]);
  const float v = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
  out[((long)t * H * W + p) * (long)C + c] = VPIPE_ELT(v);
}

// The LMS Sharp curve branch on one block's AdaLN modulation, for every
// distinct timestep of a forward:
//
//   mod[t, j] += scale * sum_k wc[j, k] * coords[t, k]
//
// `wc` is the branch's B A, [D, 8] f32 (B [D, r] times A [r, 8], formed
// once at load), and `coords` the 8-D curve point each timestep reads off
// the branch's table. The delta is summed in f32 and rounded once into
// the modulation, which is in the DiT's element type.
//   0:mod[n_t, D] 1:wc[D, 8] 2:coords[n_t, 8] 3:D 4:n_t 5:scale
//   grid {D, n_t}
kernel void h3_adaln_curve_add_f16(
    device VPIPE_ELT*   mod    [[buffer(0)]],
    const device float* wc     [[buffer(1)]],
    const device float* coords [[buffer(2)]],
    constant int&       D      [[buffer(3)]],
    constant int&       n_t    [[buffer(4)]],
    constant float&     scale  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int j = (int)gid.x, t = (int)gid.y;
  if (j >= D || t >= n_t) { return; }
  const device float4* w4 =
      reinterpret_cast<const device float4*>(wc + (long)j * 8);
  const device float4* c4 =
      reinterpret_cast<const device float4*>(coords + (long)t * 8);
  const float d = dot(w4[0], c4[0]) + dot(w4[1], c4[1]);
  const long idx = (long)t * D + j;
  mod[idx] = VPIPE_ELT(float(mod[idx]) + scale * d);
}

// ---- the FAST decoder (vae_fast/; generative-models/minimax-h3/
// metal-minimax-h3-fast-vae.h) ------------------------------------------
//
// Its residual stream is f32 [rows, D] -- token rows (t, y, x) first, then
// five global rows per native block -- and every GEMM operand the element
// type. These are the pieces around the GEMMs and the attention.

// The latent's whitening, post_quant_conv and x_embedder folded into one
// affine map on the host (A [D, Z], c [D], f32), over the channel-first
// whitened latent z [Z, rows]:
//
//   x[r, d] = c[d] + sum_k A[d, k] * z[k, r]
//
//   0:z 1:A 2:c 3:x(f32) 4:rows 5:D 6:Z    grid {D, rows}
kernel void h3fv_embed_f32(
    const device float* z    [[buffer(0)]],
    const device float* A    [[buffer(1)]],
    const device float* c    [[buffer(2)]],
    device float*       x    [[buffer(3)]],
    constant int&       rows [[buffer(4)]],
    constant int&       D    [[buffer(5)]],
    constant int&       Z    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, r = (int)gid.y;
  if (d >= D || r >= rows) { return; }
  float acc = c[d];
  for (int k = 0; k < Z; ++k) {
    acc += A[d * Z + k] * z[(long)k * rows + r];
  }
  x[(long)r * D + d] = acc;
}

// One threadgroup (256 threads) a row of the f32 stream, into the element
// type: RMSNorm with a weight (`ln` 0), or LayerNorm with a weight and a
// bias (`ln` 1). D a multiple of 256.
//   0:x(f32) 1:w(f32) 2:b(f32, ln only) 3:out 4:D 5:eps 6:ln
//   grid {256, rows}, threadgroup {256, 1}
kernel void h3fv_norm_f32(
    const device float* x   [[buffer(0)]],
    const device float* w   [[buffer(1)]],
    const device float* b   [[buffer(2)]],
    device VPIPE_ELT*   out [[buffer(3)]],
    constant int&       D   [[buffer(4)]],
    constant float&     eps [[buffer(5)]],
    constant int&       ln  [[buffer(6)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  threadgroup float part[8];
  const device float* xr = x + (long)tgid.y * D;
  device VPIPE_ELT*   orow = out + (long)tgid.y * D;
  const int n = D / 256;
  float v[16];
  float s = 0.0f;
  for (int i = 0; i < n && i < 16; ++i) {
    v[i] = xr[i * 256 + (int)tid.x];
    s += v[i];
  }
  float mean = 0.0f;
  if (ln != 0) {
    s = simd_sum(s);
    if (sl == 0) { part[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) { t += part[i]; }
    mean = t / (float)D;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float q = 0.0f;
  for (int i = 0; i < n && i < 16; ++i) {
    const float c = v[i] - mean;
    q += c * c;
  }
  q = simd_sum(q);
  if (sl == 0) { part[sg] = q; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (int i = 0; i < 8; ++i) { t += part[i]; }
  const float inv = rsqrt(t / (float)D + eps);
  for (int i = 0; i < n && i < 16; ++i) {
    const int c = i * 256 + (int)tid.x;
    float y = (v[i] - mean) * inv * w[c];
    if (ln != 0) { y += b[c]; }
    orow[c] = VPIPE_ELT(y);
  }
}

// The residual add, in f32: x[i] += s[i % D] * y[i] over n elements.
//   0:x(f32) 1:s(f32) 2:y 3:D 4:n    grid {n}
kernel void h3fv_gated_f32(
    device float*           x [[buffer(0)]],
    const device float*     s [[buffer(1)]],
    const device VPIPE_ELT* y [[buffer(2)]],
    constant int&           D [[buffer(3)]],
    constant uint&          n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  x[gid] += s[gid % (uint)D] * float(y[gid]);
}

// What a fused to_qkv row becomes: per head, its q and k RMS-normed (no
// affine, f32), the first `rd` channels of both rotated (rotate-half over
// two halves of rd/2) by the token's WINDOW-LOCAL ids, then q, k and v
// scattered to where the attention reads them -- q to row qrow[r] of qp,
// k and v to row krow[r] of kp / vp -- and, with `ring` set, k and v also
// to the token's row of its native block in the two-block ring.
//
// The fused projection groups PER HEAD: a head's q, k and v sit 64 apart
// inside a 192-wide stretch. One simdgroup a (row, head), a lane holding
// channels lane and lane + 32.
//
// The rotary table is [3][16][8] cos then sin (axis, window-local index,
// frequency); the 24 angles of a token are the three axes' 8 frequencies
// in that order, duplicated across the two halves. `rope` 0 skips it (the
// global rows), and `tok0` < 0 means the rows carry no token geometry.
//
//   0:qkv 1:qp 2:kp 3:vp 4:qrow(int) 5:krow(int) 6:cs(f32) 7:ringk 8:ringv
//   9:p {tok0, T, H, W, s, st, tb, ws, ring_rows, rope, ring, heads}
//   10:eps
//   grid {32, heads, rows}, threadgroup {32, 4, 1}
struct H3fvPost {
  int tok0, T, H, W, s, st, tb, ws, ring_rows, rope, ring, heads;
};
kernel void h3fv_qkv_post_f16(
    const device VPIPE_ELT* qkv  [[buffer(0)]],
    device VPIPE_ELT*       qp   [[buffer(1)]],
    device VPIPE_ELT*       kp   [[buffer(2)]],
    device VPIPE_ELT*       vp   [[buffer(3)]],
    const device int*       qrow [[buffer(4)]],
    const device int*       krow [[buffer(5)]],
    const device float*     cs   [[buffer(6)]],
    device VPIPE_ELT*       ringk [[buffer(7)]],
    device VPIPE_ELT*       ringv [[buffer(8)]],
    constant H3fvPost&      p    [[buffer(9)]],
    constant float&         eps  [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int lane = (int)gid.x, h = (int)gid.y, r = (int)gid.z;
  if (h >= p.heads) { return; }
  const int D = p.heads * 64;
  const device VPIPE_ELT* src = qkv + (long)r * 3 * D + h * 192;
  float q0 = float(src[lane]), q1 = float(src[lane + 32]);
  float k0 = float(src[64 + lane]), k1 = float(src[64 + lane + 32]);
  const VPIPE_ELT v0 = src[128 + lane], v1 = src[128 + lane + 32];
  const float rq = rsqrt(simd_sum(q0 * q0 + q1 * q1) / 64.0f + eps);
  const float rk = rsqrt(simd_sum(k0 * k0 + k1 * k1) / 64.0f + eps);
  q0 *= rq; q1 *= rq; k0 *= rk; k1 *= rk;
  int t = 0, rem = 0;
  if (p.tok0 >= 0) {
    const int tok = p.tok0 + r;
    const int hw = p.H * p.W;
    t = tok / hw;
    rem = tok - t * hw;
  }
  if (p.rope != 0) {
    const int y = rem / p.W, x = rem - (rem / p.W) * p.W;
    const int li[3] = {(t + p.st) % p.tb, (y + p.s) % p.ws,
                       (x + p.s) % p.ws};
    // Channel d < 48 rotates with angle a = d % 24 and partner d +- 24.
    const int a0 = lane % 24;
    const float c0 = cs[(a0 / 8) * 128 + li[a0 / 8] * 8 + a0 % 8];
    const float s0 = cs[384 + (a0 / 8) * 128 + li[a0 / 8] * 8 + a0 % 8];
    const int a1 = (lane + 32) % 24;
    const float c1 = cs[(a1 / 8) * 128 + li[a1 / 8] * 8 + a1 % 8];
    const float s1 = cs[384 + (a1 / 8) * 128 + li[a1 / 8] * 8 + a1 % 8];
    const int p0 = lane < 24 ? lane + 24 : lane - 24;   // in [0, 48)
    const int p1 = (lane + 32) - 24;                    // lanes < 16
    const float qa = simd_shuffle(q0, (ushort)(p0 & 31));
    const float qb = simd_shuffle(q1, (ushort)(p0 & 31));
    const float ka = simd_shuffle(k0, (ushort)(p0 & 31));
    const float kb = simd_shuffle(k1, (ushort)(p0 & 31));
    const float qc = simd_shuffle(q0, (ushort)(p1 & 31));
    const float kc = simd_shuffle(k0, (ushort)(p1 & 31));
    const float qp0 = p0 < 32 ? qa : qb, kp0 = p0 < 32 ? ka : kb;
    const float sg0 = lane < 24 ? -1.0f : 1.0f;
    const float nq0 = q0 * c0 + sg0 * qp0 * s0;
    const float nk0 = k0 * c0 + sg0 * kp0 * s0;
    if (lane < 16) {
      q1 = q1 * c1 + qc * s1;
      k1 = k1 * c1 + kc * s1;
    }
    q0 = nq0;
    k0 = nk0;
  }
  const long qo = (long)qrow[r] * D + h * 64;
  const long ko = (long)krow[r] * D + h * 64;
  qp[qo + lane] = VPIPE_ELT(q0);
  qp[qo + lane + 32] = VPIPE_ELT(q1);
  kp[ko + lane] = VPIPE_ELT(k0);
  kp[ko + lane + 32] = VPIPE_ELT(k1);
  vp[ko + lane] = v0;
  vp[ko + lane + 32] = v1;
  if (p.ring != 0) {
    const int b = t / p.tb;
    const long ro = ((long)(b & 1) * p.ring_rows +
                     (long)(t - b * p.tb) * p.H * p.W + rem) * D + h * 64;
    ringk[ro + lane] = VPIPE_ELT(k0);
    ringk[ro + lane + 32] = VPIPE_ELT(k1);
    ringv[ro + lane] = v0;
    ringv[ro + lane + 32] = v1;
  }
}

// Rows of D elements: dst[drow(i)] = src[smap[i]] for i < n, where drow is
// dmap[i], or d0 + i when `has_dmap` is 0. A second pair (src2 -> dst2)
// moves the same rows when `pair` is 1 -- a key and its value.
//   0:src 1:dst 2:src2 3:dst2 4:smap 5:dmap 6:D 7:n 8:d0 9:has_dmap
//   10:pair    grid {D/4, n}
kernel void h3fv_copy_rows_f16(
    const device VPIPE_ELT* src  [[buffer(0)]],
    device VPIPE_ELT*       dst  [[buffer(1)]],
    const device VPIPE_ELT* src2 [[buffer(2)]],
    device VPIPE_ELT*       dst2 [[buffer(3)]],
    const device int*       smap [[buffer(4)]],
    const device int*       dmap [[buffer(5)]],
    constant int&           D    [[buffer(6)]],
    constant int&           n    [[buffer(7)]],
    constant int&           d0   [[buffer(8)]],
    constant int&           has_dmap [[buffer(9)]],
    constant int&           pair [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x * 4, i = (int)gid.y;
  if (i >= n || c >= D) { return; }
  const long so = (long)smap[i] * D + c;
  const long dr = has_dmap != 0 ? (long)dmap[i] : (long)(d0 + i);
  const long dof = dr * D + c;
  for (int k = 0; k < 4; ++k) { dst[dof + k] = src[so + k]; }
  if (pair != 0) {
    for (int k = 0; k < 4; ++k) { dst2[dof + k] = src2[so + k]; }
  }
}

// The head's output into frames: a band of patch rows g [n, 3 * pt * 16 *
// 16] (channel slowest, then pt, py, px) plus the head's bias, written to
// out [3, F, 16 H, 16 W] -- the first `trim` of the pt frames of every
// native block's first latent dropped, as the released decoder does, so 5
// latents give 4 * 5 - 3 = 17 frames.
//   0:g 1:bias(f32) 2:out 3:tok0 4:n 5:H 6:W 7:F 8:tb 9:trim 10:pt
//   grid {16 * pt * 3 * 16, n}  (x = (oc, kt, py) * 16 + px)
kernel void h3fv_unpatch_f16(
    const device VPIPE_ELT* g    [[buffer(0)]],
    const device float*     bias [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&           tok0 [[buffer(3)]],
    constant int&           n    [[buffer(4)]],
    constant int&           H    [[buffer(5)]],
    constant int&           W    [[buffer(6)]],
    constant int&           F    [[buffer(7)]],
    constant int&           tb   [[buffer(8)]],
    constant int&           trim [[buffer(9)]],
    constant int&           pt   [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int e = (int)gid.x, r = (int)gid.y;
  const int PE = 3 * pt * 256;
  if (r >= n || e >= PE) { return; }
  const int tok = tok0 + r;
  const int t = tok / (H * W);
  const int rem = tok - t * H * W;
  const int y = rem / W, x = rem - (rem / W) * W;
  const int oc = e / (pt * 256);
  const int kt = (e / 256) % pt;
  const int py = (e / 16) % 16, px = e % 16;
  if (t % tb == 0 && kt < trim) { return; }
  const int fo = t * pt + kt - trim * (t / tb + 1);
  const long W16 = (long)W * 16, H16 = (long)H * 16;
  const long o = (((long)oc * F + fo) * H16 + (long)y * 16 + py) * W16 +
                 (long)x * 16 + px;
  out[o] = VPIPE_ELT(float(g[(long)r * PE + e]) + bias[e]);
}
