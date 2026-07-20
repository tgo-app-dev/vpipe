// rope.metal -- rotary position embedding (NEOX / non-traditional,
// half-split) applied in place to x[H, T, D] using a precomputed
// per-pair frequency table inv_freq[D/2]. The host bakes any rope
// scaling (e.g. llama3) into inv_freq, so the kernel is generic:
//   angle      = (offset + t) * inv_freq[i]
//   out[i]     = x[i] * cos - x[i + D/2] * sin
//   out[i+D/2] = x[i] * sin + x[i + D/2] * cos
// Matches mlx::fast::rope(x, D, traditional=false, base=none, scale=1,
// offset, freqs=inv_freq).
//
//   0: x        (device VPIPE_ELT*)        in/out [H, T, D]
//   1: inv_freq (device const float*) [D/2]
//   2: H  3: T  4: D  5: offset       (constant int&)
// grid (D/2, T, H).

#include <metal_stdlib>
using namespace metal;

// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the
// bf16 variant metallib. Math stays f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// rms_rope_f16 -- fused per-head RMSNorm + NeoX RoPE for the decode (T=1)
// q/k path: normalize each head's D dims (weight applied), then rotate the
// half-split pairs. One dispatch replaces the separate rms_norm + rope_f16
// -- cuts decode per-dispatch barrier bubbles. One threadgroup per head
// (grid (RR_TG, H, 1)); reduce sum(x^2) via simd_sum across 8 simdgroups.
// Math is rms_norm then rope_f16 with the normed value kept in f32 through
// the rotation (vs the separate path's intermediate f16 round).
//   0:x[H,1,D] (in/out) 1:weight[D] 2:inv_freq[D/2] 3:H 4:D 5:eps 6:offset
#define RR_TG 256
kernel void rms_rope_f16(
    device VPIPE_ELT*        x        [[buffer(0)]],
    const device VPIPE_ELT*  weight   [[buffer(1)]],
    const device float*      inv_freq [[buffer(2)]],
    constant int&       H        [[buffer(3)]],
    constant int&       D        [[buffer(4)]],
    constant float&     eps      [[buffer(5)]],
    constant int&       offset   [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  (void)H;
  const uint h = tid.y;
  const uint lid = ltid.x;
  const int half_d = D / 2;
  device VPIPE_ELT* xr = x + (uint)h * D;

  float local = 0.0f;
  for (int i = (int)lid; i < D; i += RR_TG) {
    const float v = float(xr[i]);
    local += v * v;
  }
  local = simd_sum(local);
  threadgroup float partial[RR_TG / 32];
  if (simd_lid == 0) { partial[simd_gid] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < RR_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);

  // Each thread owns disjoint pairs (i, i+half_d): normalize then rotate.
  for (int i = (int)lid; i < half_d; i += RR_TG) {
    const float a = float(xr[i]) * inv * float(weight[i]);
    const float b = float(xr[i + half_d]) * inv * float(weight[i + half_d]);
    const float angle = float(offset) * inv_freq[i];
    const float c = cos(angle);
    const float s = sin(angle);
    xr[i]          = VPIPE_ELT(a * c - b * s);
    xr[i + half_d] = VPIPE_ELT(a * s + b * c);
  }
}

// Fused Q+K rms-norm + rope: decode issues rms_rope twice (q then k) -> 2
// launches/layer; this does both in ONE (heads 0..Hq-1 norm Q with q_weight,
// Hq..Hq+Hkv-1 norm K with k_weight; same D/offset/inv_freq). Bit-identical
// to two rms_rope calls. grid (RR_TG, Hq+Hkv, 1).
//   0:q 1:q_weight 2:k 3:k_weight 4:inv_freq 5:Hq 6:D 7:eps 8:offset
kernel void rms_rope2_f16(
    device VPIPE_ELT*        q        [[buffer(0)]],
    const device VPIPE_ELT*  q_weight [[buffer(1)]],
    device VPIPE_ELT*        k        [[buffer(2)]],
    const device VPIPE_ELT*  k_weight [[buffer(3)]],
    const device float*      inv_freq [[buffer(4)]],
    constant int&       Hq       [[buffer(5)]],
    constant int&       D        [[buffer(6)]],
    constant float&     eps      [[buffer(7)]],
    constant int&       offset   [[buffer(8)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint hh = tid.y;
  // Route to Q (head hh) or K (head hh-Hq) by the global head index.
  device VPIPE_ELT* xbuf;
  const device VPIPE_ELT* weight;
  if ((int)hh < Hq) { xbuf = q + hh * (uint)D; weight = q_weight; }
  else { xbuf = k + (hh - (uint)Hq) * (uint)D; weight = k_weight; }

  const uint lid = ltid.x;
  const int half_d = D / 2;
  float local = 0.0f;
  for (int i = (int)lid; i < D; i += RR_TG) {
    const float v = float(xbuf[i]);
    local += v * v;
  }
  local = simd_sum(local);
  threadgroup float partial[RR_TG / 32];
  if (simd_lid == 0) { partial[simd_gid] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < RR_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);

  for (int i = (int)lid; i < half_d; i += RR_TG) {
    const float a = float(xbuf[i]) * inv * float(weight[i]);
    const float b = float(xbuf[i + half_d]) * inv * float(weight[i + half_d]);
    const float angle = float(offset) * inv_freq[i];
    const float c = cos(angle);
    const float s = sin(angle);
    xbuf[i]          = VPIPE_ELT(a * c - b * s);
    xbuf[i + half_d] = VPIPE_ELT(a * s + b * c);
  }
}

// Fused Q+K+V norm(+rope): like rms_rope2 but also folds the V rms-norm (which
// has NO rope -- v_weight is the weightless _ones row). Heads 0..Hq-1 = Q
// (q_weight, rope), Hq..Hq+Hkv-1 = K (k_weight, rope), Hq+Hkv..Hq+2Hkv-1 = V
// (v_weight, NO rope). Only valid when V is independent of K (NOT k_eq_v,
// where V must read raw K before K is normed). grid (RR_TG, Hq+2Hkv, 1).
//   0:q 1:q_w 2:k 3:k_w 4:v 5:v_w 6:inv_freq 7:Hq 8:Hkv 9:D 10:eps 11:offset
kernel void rms_rope3_f16(
    device VPIPE_ELT*        q        [[buffer(0)]],
    const device VPIPE_ELT*  q_weight [[buffer(1)]],
    device VPIPE_ELT*        k        [[buffer(2)]],
    const device VPIPE_ELT*  k_weight [[buffer(3)]],
    device VPIPE_ELT*        v        [[buffer(4)]],
    const device VPIPE_ELT*  v_weight [[buffer(5)]],
    const device float*      inv_freq [[buffer(6)]],
    constant int&       Hq       [[buffer(7)]],
    constant int&       Hkv      [[buffer(8)]],
    constant int&       D        [[buffer(9)]],
    constant float&     eps      [[buffer(10)]],
    constant int&       offset   [[buffer(11)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint hh = tid.y;
  device VPIPE_ELT* xbuf;
  const device VPIPE_ELT* weight;
  bool do_rope;
  if ((int)hh < Hq) {
    xbuf = q + hh * (uint)D; weight = q_weight; do_rope = true;
  } else if ((int)hh < Hq + Hkv) {
    xbuf = k + (hh - (uint)Hq) * (uint)D; weight = k_weight; do_rope = true;
  } else {
    xbuf = v + (hh - (uint)(Hq + Hkv)) * (uint)D; weight = v_weight;
    do_rope = false;
  }

  const uint lid = ltid.x;
  const int half_d = D / 2;
  float local = 0.0f;
  for (int i = (int)lid; i < D; i += RR_TG) {
    const float vv = float(xbuf[i]);
    local += vv * vv;
  }
  local = simd_sum(local);
  threadgroup float partial[RR_TG / 32];
  if (simd_lid == 0) { partial[simd_gid] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < RR_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);

  for (int i = (int)lid; i < half_d; i += RR_TG) {
    const float a = float(xbuf[i]) * inv * float(weight[i]);
    const float b = float(xbuf[i + half_d]) * inv * float(weight[i + half_d]);
    if (do_rope) {
      const float angle = float(offset) * inv_freq[i];
      const float c = cos(angle);
      const float s = sin(angle);
      xbuf[i]          = VPIPE_ELT(a * c - b * s);
      xbuf[i + half_d] = VPIPE_ELT(a * s + b * c);
    } else {
      xbuf[i]          = VPIPE_ELT(a);
      xbuf[i + half_d] = VPIPE_ELT(b);
    }
  }
}

// Fused rms_rope3 + ring KV-write (Gemma-4 sliding decode). Same per-head
// norm+rope as rms_rope3_f16, but the K and V heads write their result STRAIGHT
// into the contiguous ring cache slot (cache[h*cap + slot, :]) instead of the
// _d_k/_d_v scratch -- folding the kv_write2 dispatch into the norm kernel (one
// fewer dependent dispatch per sliding layer). Q is normed+roped in place (read
// by SDPA). One threadgroup per head (no fan-out), so the write redirect adds
// zero compute. Sliding/ring layout only; paged-global keeps the split path.
//   0:q 1:q_w 2:k(in) 3:k_w 4:v(in) 5:v_w 6:inv_freq 7:Hq 8:Hkv 9:D 10:eps
//   11:offset 12:cache_k 13:cache_v 14:cap 15:pos. grid (RR_TG, Hq+2*Hkv, 1).
kernel void rms_rope3_kvwrite_f16(
    device VPIPE_ELT*        q        [[buffer(0)]],
    const device VPIPE_ELT*  q_weight [[buffer(1)]],
    const device VPIPE_ELT*  k        [[buffer(2)]],
    const device VPIPE_ELT*  k_weight [[buffer(3)]],
    const device VPIPE_ELT*  v        [[buffer(4)]],
    const device VPIPE_ELT*  v_weight [[buffer(5)]],
    const device float*      inv_freq [[buffer(6)]],
    constant int&       Hq       [[buffer(7)]],
    constant int&       Hkv      [[buffer(8)]],
    constant int&       D        [[buffer(9)]],
    constant float&     eps      [[buffer(10)]],
    constant int&       offset   [[buffer(11)]],
    device VPIPE_ELT*        cache_k  [[buffer(12)]],
    device VPIPE_ELT*        cache_v  [[buffer(13)]],
    constant int&       cap      [[buffer(14)]],
    constant int&       pos      [[buffer(15)]],
    constant int&       ring_cap [[buffer(16)]],
    constant int&       window   [[buffer(17)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  // cap == PHYSICAL head stride (ring modulo + mirror tail). The slot wraps by
  // the ring modulo; a head slot (slot < window-1) is mirrored into the tail
  // (slot+ring_cap) so the trailing window reads linearly. ring_cap<=0 ->
  // linear (slot = pos), no mirror.
  const uint hh = tid.y;
  const int mod  = (ring_cap > 0) ? ring_cap : cap;
  const int slot = pos % mod;
  const bool mirror = (ring_cap > 0) && (slot < window - 1);
  const device VPIPE_ELT* xin;
  device VPIPE_ELT* xout;
  device VPIPE_ELT* xout_m = nullptr;          // mirror tail dst (K/V only)
  const device VPIPE_ELT* weight;
  bool do_rope;
  if ((int)hh < Hq) {                                  // Q: in place (-> SDPA)
    xin = q + hh * (uint)D; xout = q + hh * (uint)D;
    weight = q_weight; do_rope = true;
  } else if ((int)hh < Hq + Hkv) {                     // K: -> ring cache slot
    const uint kh = hh - (uint)Hq;
    xin = k + kh * (uint)D;
    xout = cache_k + ((uint)kh * (uint)cap + (uint)slot) * (uint)D;
    if (mirror) {
      xout_m = cache_k
          + ((uint)kh * (uint)cap + (uint)(slot + ring_cap)) * (uint)D;
    }
    weight = k_weight; do_rope = true;
  } else {                                             // V: -> ring cache slot
    const uint vh = hh - (uint)(Hq + Hkv);
    xin = v + vh * (uint)D;
    xout = cache_v + ((uint)vh * (uint)cap + (uint)slot) * (uint)D;
    if (mirror) {
      xout_m = cache_v
          + ((uint)vh * (uint)cap + (uint)(slot + ring_cap)) * (uint)D;
    }
    weight = v_weight; do_rope = false;
  }

  const uint lid = ltid.x;
  const int half_d = D / 2;
  float local = 0.0f;
  for (int i = (int)lid; i < D; i += RR_TG) {
    const float vv = float(xin[i]);
    local += vv * vv;
  }
  local = simd_sum(local);
  threadgroup float partial[RR_TG / 32];
  if (simd_lid == 0) { partial[simd_gid] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < RR_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);

  for (int i = (int)lid; i < half_d; i += RR_TG) {
    const float a = float(xin[i]) * inv * float(weight[i]);
    const float b = float(xin[i + half_d]) * inv * float(weight[i + half_d]);
    if (do_rope) {
      const float angle = float(offset) * inv_freq[i];
      const float c = cos(angle);
      const float s = sin(angle);
      const VPIPE_ELT lo = VPIPE_ELT(a * c - b * s);
      const VPIPE_ELT hi = VPIPE_ELT(a * s + b * c);
      xout[i]          = lo;
      xout[i + half_d] = hi;
      if (xout_m != nullptr) { xout_m[i] = lo; xout_m[i + half_d] = hi; }
    } else {
      const VPIPE_ELT lo = VPIPE_ELT(a);
      const VPIPE_ELT hi = VPIPE_ELT(b);
      xout[i]          = lo;
      xout[i + half_d] = hi;
      if (xout_m != nullptr) { xout_m[i] = lo; xout_m[i + half_d] = hi; }
    }
  }
}

kernel void rope_f16(
    device VPIPE_ELT*        x        [[buffer(0)]],
    const device float* inv_freq [[buffer(1)]],
    constant int&       H        [[buffer(2)]],
    constant int&       T        [[buffer(3)]],
    constant int&       D        [[buffer(4)]],
    constant int&       offset   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;  // grid z-dim bounds the head index; H kept for clarity
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;

  const float angle = float(offset + t) * inv_freq[i];
  const float c = cos(angle);
  const float s = sin(angle);

  const uint base = ((uint)h * T + t) * D;
  const float x1 = float(x[base + i]);
  const float x2 = float(x[base + i + half_d]);
  x[base + i]          = VPIPE_ELT(x1 * c - x2 * s);
  x[base + i + half_d] = VPIPE_ELT(x1 * s + x2 * c);
}

// Interleaved-pair RoPE (GPT-J / MOSS-Audio-Tokenizer convention): rotates
// ADJACENT pairs (x[2i], x[2i+1]) -- the codec reshapes the head dim to
// (D/2, 2) and rotates within each pair -- vs rope_f16's half-split pairs
// (x[i], x[i+D/2]). inv_freq[i] = max_period^(-2i/D), host-baked. offset is
// the absolute start position (0 for a full non-cached forward).
//   0:x[H,T,D] (in/out) 1:inv_freq[D/2] 2:H 3:T 4:D 5:offset.
//   grid (D/2, T, H).
kernel void rope_interleaved_f16(
    device VPIPE_ELT*        x        [[buffer(0)]],
    const device float* inv_freq [[buffer(1)]],
    constant int&       H        [[buffer(2)]],
    constant int&       T        [[buffer(3)]],
    constant int&       D        [[buffer(4)]],
    constant int&       offset   [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;

  const float angle = float(offset + t) * inv_freq[i];
  const float c = cos(angle);
  const float s = sin(angle);

  const uint base = ((uint)h * T + t) * D;
  const float x1 = float(x[base + 2 * i]);
  const float x2 = float(x[base + 2 * i + 1]);
  x[base + 2 * i]     = VPIPE_ELT(x1 * c - x2 * s);
  x[base + 2 * i + 1] = VPIPE_ELT(x1 * s + x2 * c);
}

// Table-driven adjacent-pair RoPE (Krea-2 DiT / Flux 3-axis): apply a
// precomputed per-position cos/sin table [T, D] to x[H, T, D] in place, using
// the adjacent-pair ("repeat_interleave_real") convention -- the host builds
// cos/sin by concatenating the (t,h,w) rotary axes, so the kernel is generic.
//   out[2i]   = x[2i] * cos[2i] - x[2i+1] * sin[2i]
//   out[2i+1] = x[2i] * sin[2i] + x[2i+1] * cos[2i]     (cos/sin repeat-interl.)
//   0:x[H,T,D] (in/out) 1:cos[T,D] 2:sin[T,D] 3:H 4:T 5:D.  grid (D/2, T, H).
kernel void rope_pair_table_f16(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device VPIPE_ELT* cosb [[buffer(1)]],
    const device VPIPE_ELT* sinb [[buffer(2)]],
    constant int&       H    [[buffer(3)]],
    constant int&       T    [[buffer(4)]],
    constant int&       D    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;

  const uint cb = (uint)t * D + 2 * i;
  const float c = float(cosb[cb]);
  const float s = float(sinb[cb]);
  const uint base = ((uint)h * T + t) * D + 2 * i;
  const float x1 = float(x[base]);
  const float x2 = float(x[base + 1]);
  x[base]     = VPIPE_ELT(x1 * c - x2 * s);
  x[base + 1] = VPIPE_ELT(x1 * s + x2 * c);
}

// Same as rope_pair_table_f16 but the cos/sin tables are FLOAT32 (host-built
// at full precision). The QwenImage DiT reference applies RoPE in f32; keeping
// the tables f32 (only x is bf16) avoids the ~4e-3 bf16-table rounding that
// otherwise compounds over the 60 blocks.
kernel void rope_pair_table_ftab_f16(
    device VPIPE_ELT*    x    [[buffer(0)]],
    const device float*  cosb [[buffer(1)]],
    const device float*  sinb [[buffer(2)]],
    constant int&        H    [[buffer(3)]],
    constant int&        T    [[buffer(4)]],
    constant int&        D    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const uint cb = (uint)t * D + 2 * i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const uint base = ((uint)h * T + t) * D + 2 * i;
  const float x1 = float(x[base]);
  const float x2 = float(x[base + 1]);
  x[base]     = VPIPE_ELT(x1 * c - x2 * s);
  x[base + 1] = VPIPE_ELT(x1 * s + x2 * c);
}

// Fused transpose [T,H,D] -> [H,T,D] + pair RoPE (f32 cos/sin tables), for the
// q/k path: reads TOKEN-major in[(t*H+h)*D + 2i(,+1)], applies token t's rope
// (interleaved pairs, same convention as rope_pair_table_ftab_f16), and writes
// HEAD-major out[(h*T+t)*D + 2i(,+1)]. One pass replaces a transpose_abd_f16
// followed by an in-place rope_pair over the transposed buffer (saves the
// separate rope's full read+write of the head-major tensor). `in` and `out` are
// distinct buffers. grid (D/2, T, H).
kernel void transpose_rope_pair_ftab_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    device VPIPE_ELT*       out  [[buffer(1)]],
    const device float*     cosb [[buffer(2)]],
    const device float*     sinb [[buffer(3)]],
    constant int&           H    [[buffer(4)]],
    constant int&           T    [[buffer(5)]],
    constant int&           D    [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const uint cb = (uint)t * D + 2 * i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const uint ib = ((uint)t * H + h) * D + 2 * i;   // token-major input
  const uint ob = ((uint)h * T + t) * D + 2 * i;   // head-major output
  const float x1 = float(in[ib]);
  const float x2 = float(in[ib + 1]);
  out[ob]     = VPIPE_ELT(x1 * c - x2 * s);
  out[ob + 1] = VPIPE_ELT(x1 * s + x2 * c);
}

// HALF-SPLIT (NEOX) RoPE from f32 cos/sin tables [T, D/2] (Qwen2.5-VL vision:
// rotate_half convention, out = x*cos + rotate_half(x)*sin, cos/sin duplicated
// over the two halves so only D/2 table entries are needed). Pairs dim i with
// i+D/2. x is VPIPE_ELT [H,T,D] in place; tables are host-built f32.
kernel void rope_half_table_ftab_f16(
    device VPIPE_ELT*    x    [[buffer(0)]],
    const device float*  cosb [[buffer(1)]],
    const device float*  sinb [[buffer(2)]],
    constant int&        H    [[buffer(3)]],
    constant int&        T    [[buffer(4)]],
    constant int&        D    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const uint cb = (uint)t * half_d + i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const uint base = ((uint)h * T + t) * D + i;
  const float x1 = float(x[base]);
  const float x2 = float(x[base + half_d]);
  x[base]          = VPIPE_ELT(x1 * c - x2 * s);
  x[base + half_d] = VPIPE_ELT(x2 * c + x1 * s);
}

// Partial RoPE (Qwen3.5): rotate only the FIRST `rotary_dim` of each
// head's `D` dims, leaving [rotary_dim, D) untouched (pass-through). The
// rotated block uses the half-split convention over rotary_dim/2 pairs;
// inv_freq has rotary_dim/2 entries (host-baked, base = rope_theta).
// Qwen3.5: head_dim D=256, rotary_dim=64, rope_theta=1e7. For text the 3
// mROPE axes collapse to one scalar position, so this 1D form matches;
// multimodal prefill needs the separate mROPE kernel.
//   0:x[H,T,D] (in/out) 1:inv_freq[rotary_dim/2] 2:H 3:T 4:D
//   5:rotary_dim 6:offset.   grid (rotary_dim/2, T, H).
kernel void rope_partial_f16(
    device VPIPE_ELT*        x          [[buffer(0)]],
    const device float* inv_freq   [[buffer(1)]],
    constant int&       H          [[buffer(2)]],
    constant int&       T          [[buffer(3)]],
    constant int&       D          [[buffer(4)]],
    constant int&       rotary_dim [[buffer(5)]],
    constant int&       offset     [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;
  const int half_r = rotary_dim / 2;
  const int i = (int)gid.x;
  if (i >= half_r) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;

  const float angle = float(offset + t) * inv_freq[i];
  const float c = cos(angle);
  const float s = sin(angle);

  const uint base = ((uint)h * T + t) * D;
  const float x1 = float(x[base + i]);
  const float x2 = float(x[base + i + half_r]);
  x[base + i]          = VPIPE_ELT(x1 * c - x2 * s);
  x[base + i + half_r] = VPIPE_ELT(x1 * s + x2 * c);
}

// Multimodal partial RoPE (Qwen3-VL prefill): like rope_partial_f16, but
// the per-token cos/sin come from precomputed tables [n, rotary_dim]
// (built host-side from the 3-axis position_ids + interleaved axis
// lookup) rather than a scalar offset. cos[p,i] == cos[p,i+half] (the
// cat([f,f]) layout), so one thread per (i, token, head) rotates the
// pair (i, i+half) in place; the tail [rotary_dim, D) is pass-through.
//   0:q[Hq,n,D] (in/out) 1:cos[n,rd] 2:sin[n,rd] 3:Hq 4:n 5:D 6:rotary_dim
// grid (rotary_dim/2, n, Hq); threadgroup (rotary_dim/2, 1, 1).
kernel void mrope_partial_f16(
    device VPIPE_ELT*        x          [[buffer(0)]],
    const device VPIPE_ELT*  cos_t      [[buffer(1)]],
    const device VPIPE_ELT*  sin_t      [[buffer(2)]],
    constant int&       Hq         [[buffer(3)]],
    constant int&       n          [[buffer(4)]],
    constant int&       D          [[buffer(5)]],
    constant int&       rotary_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)Hq;
  const int half_r = rotary_dim / 2;
  const int i = (int)gid.x;
  if (i >= half_r) { return; }
  const int p = (int)gid.y;
  const int h = (int)gid.z;
  const float c = (float)cos_t[(uint)p * rotary_dim + i];
  const float s = (float)sin_t[(uint)p * rotary_dim + i];
  const uint base = ((uint)h * n + p) * D;
  const float x1 = (float)x[base + i];
  const float x2 = (float)x[base + i + half_r];
  x[base + i]          = (VPIPE_ELT)(x1 * c - x2 * s);
  x[base + i + half_r] = (VPIPE_ELT)(x2 * c + x1 * s);
}

// Fused per-head RMSNorm + partial RoPE for the DECODE path (T==1), to
// drop two dispatches per attention layer (q_norm+rope_q, k_norm+rope_k)
// into one each -- in MTLDispatchTypeSerial mode each dispatch is a GPU
// barrier/drain, so the small q/k norm+rope ops are launch-bound. One
// threadgroup per head: reduce sum(x^2) over D (simd_sum across 8
// simdgroups, f32 accum), normalize in place (x*inv*weight, f16 store),
// barrier, then rotate the first `rotary_dim` dims (half-split pairs).
// Byte-identical to rms_norm_f16 followed by rope_partial_f16: same f16
// intermediate after the norm, same rope math. Tail [rotary_dim, D) is
// normalized but not rotated (pass-through), matching rope_partial_f16.
//   0:x[heads,D] (in/out) 1:weight[D] 2:inv_freq[rotary_dim/2]
//   3:D 4:rotary_dim 5:offset 6:eps
// grid (256, heads, 1); threadgroup (256, 1, 1).
#define RMS_ROPE_TG 256
kernel void rms_rope_partial_f16(
    device VPIPE_ELT*        x          [[buffer(0)]],
    const device VPIPE_ELT*  weight     [[buffer(1)]],
    const device float*      inv_freq   [[buffer(2)]],
    constant int&       D          [[buffer(3)]],
    constant int&       rotary_dim [[buffer(4)]],
    constant int&       offset     [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint head = tid.y;
  const int  lid  = (int)ltid.x;
  device VPIPE_ELT* xr = x + (uint)head * (uint)D;

  // RMSNorm reduction over D.
  float local = 0.0f;
  for (int i = lid; i < D; i += RMS_ROPE_TG) {
    const float v = float(xr[i]);
    local += v * v;
  }
  local = simd_sum(local);
  threadgroup float partial[RMS_ROPE_TG / 32];
  if (simd_lid == 0) { partial[simd_gid] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < RMS_ROPE_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(D) + eps);

  // Normalize in place (same f16 intermediate as rms_norm_f16).
  for (int i = lid; i < D; i += RMS_ROPE_TG) {
    xr[i] = VPIPE_ELT(float(xr[i]) * inv * float(weight[i]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Partial RoPE over the first rotary_dim dims (half-split pairs); each
  // thread owns a unique pair (i, i+half_r), so no write conflicts.
  const int half_r = rotary_dim / 2;
  for (int i = lid; i < half_r; i += RMS_ROPE_TG) {
    const float angle = float(offset) * inv_freq[i];
    const float c = cos(angle);
    const float s = sin(angle);
    const float x1 = float(xr[i]);
    const float x2 = float(xr[i + half_r]);
    xr[i]          = VPIPE_ELT(x1 * c - x2 * s);
    xr[i + half_r] = VPIPE_ELT(x1 * s + x2 * c);
  }
}
