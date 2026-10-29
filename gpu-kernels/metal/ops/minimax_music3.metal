// minimax_music3.metal -- the kernels of MiniMax-Music3's depth decoder
// that are its own (generative-models/minimax-music3/): the four-layer
// transformer that picks a frame's residual codes c1..c7 one after the
// other, over a sequence of at most 16 positions per guidance row.
//
// Its attention is causal over those positions with no rotary and no
// mask beyond causality, so the K/V of a frame is a tiny per-layer cache
// [rows, positions, dim]; a step appends its positions and attends them.
//
// A step's input ROWS are its new positions, row-major per guidance row:
// row r is guidance row b = r / np at position p = p0 + r % np.
//
// Element (storage) type: half by default; -DVPIPE_ELT=bfloat builds the
// bf16 variant (minimax_music3_bf16), the one the model runs. Every
// reduction is f32.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// K and V of the step's rows into the caches. `qkv` is the fused [R, 3 D]
// projection ([q | k | v]); the caches are [B, P, D].
//   0:qkv 1:kc 2:vc 3:D 4:R 5:np 6:p0 7:P.   grid (D, R).
kernel void music3_depth_kv_append(
    const device VPIPE_ELT* qkv [[buffer(0)]],
    device VPIPE_ELT*       kc  [[buffer(1)]],
    device VPIPE_ELT*       vc  [[buffer(2)]],
    constant int&      D   [[buffer(3)]],
    constant int&      R   [[buffer(4)]],
    constant int&      np  [[buffer(5)]],
    constant int&      p0  [[buffer(6)]],
    constant int&      P   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int r = (int)gid.y;
  if (d >= D || r >= R) { return; }
  const int b = r / np;
  const int p = p0 + r % np;
  const ulong src = (ulong)r * (ulong)(3 * D);
  const ulong dst = ((ulong)b * (ulong)P + (ulong)p) * (ulong)D + (ulong)d;
  kc[dst] = qkv[src + (ulong)D + (ulong)d];
  vc[dst] = qkv[src + (ulong)(2 * D) + (ulong)d];
}

// Causal attention of the step's rows over the cache: row r (guidance row
// b, position p), head h, attends positions 0..p of row b.
//   out[r, h, :] = sum_j softmax_j(q . k_j * scale) v_j
// One threadgroup per (row, head), Dh threads (one channel each, Dh a
// multiple of 32, at most 1024); at most 16 positions.
//   0:qkv 1:kc 2:vc 3:out[R, D] 4:D 5:Dh 6:np 7:p0 8:P 9:scale.
//   grid (Dh * R, H) threads, tg (Dh, 1).
#define MUSIC3_MAX_POS 16
kernel void music3_depth_attn(
    const device VPIPE_ELT* qkv [[buffer(0)]],
    const device VPIPE_ELT* kc  [[buffer(1)]],
    const device VPIPE_ELT* vc  [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&      D     [[buffer(4)]],
    constant int&      Dh    [[buffer(5)]],
    constant int&      np    [[buffer(6)]],
    constant int&      p0    [[buffer(7)]],
    constant int&      P     [[buffer(8)]],
    constant float&    scale [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const int r = (int)tg.x;
  const int h = (int)tg.y;
  const int b = r / np;
  const int p = p0 + r % np;
  const int d = (int)lid;
  const int nsg = Dh / 32;
  threadgroup float part[MUSIC3_MAX_POS][32];
  threadgroup float sc[MUSIC3_MAX_POS];
  const float q = (float)qkv[(ulong)r * (ulong)(3 * D) + (ulong)(h * Dh + d)];
  const ulong base = (ulong)b * (ulong)P * (ulong)D + (ulong)(h * Dh + d);
  const int nk = p + 1;
  for (int j = 0; j < nk; ++j) {
    const float s = simd_sum(q * (float)kc[base + (ulong)j * (ulong)D]);
    if (sl == 0) { part[j][sg] = s; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < (uint)nk) {
    float s = 0.0f;
    for (int g = 0; g < nsg; ++g) { s += part[lid][g]; }
    sc[lid] = s * scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (int j = 0; j < nk; ++j) { m = max(m, sc[j]); }
  float den = 0.0f, acc = 0.0f;
  for (int j = 0; j < nk; ++j) {
    const float e = exp(sc[j] - m);
    den += e;
    acc += e * (float)vc[base + (ulong)j * (ulong)D];
  }
  out[(ulong)r * (ulong)D + (ulong)(h * Dh + d)] = (VPIPE_ELT)(acc / den);
}

// The depth decoder's input rows: proj(e) + pos[p] -- `proj` the GEMV's
// output [R, D], `pos` the learned table [P, D].
//   0:x[R, D] (in place) 1:pos 2:D 3:R 4:np 5:p0.   grid (D, R).
kernel void music3_depth_add_pos(
    device VPIPE_ELT*       x   [[buffer(0)]],
    const device VPIPE_ELT* pos [[buffer(1)]],
    constant int&      D   [[buffer(2)]],
    constant int&      R   [[buffer(3)]],
    constant int&      np  [[buffer(4)]],
    constant int&      p0  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int r = (int)gid.y;
  if (d >= D || r >= R) { return; }
  const int p = p0 + r % np;
  const ulong i = (ulong)r * (ulong)D + (ulong)d;
  x[i] = (VPIPE_ELT)((float)x[i] + (float)pos[(ulong)p * (ulong)D + d]);
}
