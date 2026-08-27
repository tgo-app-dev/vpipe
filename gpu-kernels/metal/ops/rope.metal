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

// transpose_rope_pair_ftab with a PADDED output head-dim: reads TOKEN-major
// in[(t*H+h)*D + 2i(,+1)], applies token t's interleaved-pair rope, and writes
// HEAD-major out[(h*T+t)*Dp + 2i(,+1)] with the [D, Dp) tail ZEROED. The pad
// lets a head_dim the flash kernels do not support (the Boogu DiT's 120) run on
// the steel bd128 path: zero-padded q/k leave every dot product unchanged.
// See transpose_abd_pad_f16 / transpose_abd_unpad_f16 for the v/output twins.
// grid (Dp/2, T, H).
kernel void transpose_rope_pair_ftab_pad_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    device VPIPE_ELT*       out  [[buffer(1)]],
    const device float*     cosb [[buffer(2)]],
    const device float*     sinb [[buffer(3)]],
    constant int&           H    [[buffer(4)]],
    constant int&           T    [[buffer(5)]],
    constant int&           D    [[buffer(6)]],
    constant int&           Dp   [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x;
  if (2 * i >= Dp) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const uint ob = ((uint)h * T + t) * Dp + 2 * i;
  if (2 * i >= D) {                      // padding tail
    out[ob] = VPIPE_ELT(0.0f);
    out[ob + 1] = VPIPE_ELT(0.0f);
    return;
  }
  const uint cb = (uint)t * D + 2 * i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const uint ib = ((uint)t * H + h) * D + 2 * i;   // token-major input
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

// MiniMax-H3: fused transpose + PARTIAL rotate-half RoPE, reading out of
// a FUSED qkv projection.
//
//   in  [seq, in_stride]  starting at column `in_off`
//   out [H, seq, D]       (what the steel flash-attention kernel reads)
//
// `in_stride` / `in_off` are what let q, k and v come straight out of
// the one [seq, 3*H*D] qkv_proj output with no split pass -- at video
// sequence lengths that buffer is over a gigabyte, so materializing
// three copies of it per block would cost more traffic than the
// attention it feeds.
//
// Three things separate this from transpose_rope_pair_ftab_f16, which
// the Wan DiT uses:
//
//   * the convention is ROTATE-HALF (GPT-NeoX), not adjacent-pair: the
//     partner of channel i is i + rot/2, not i ^ 1.
//   * only the leading `rot` channels rotate. MiniMax-H3 builds its
//     angles from 3 axes x rope_freq_dim frequencies and then
//     concatenates the block with itself, giving rot = 2 * 3 * 16 = 96
//     of the 128-wide head. The remaining 32 channels are copied
//     THROUGH unchanged -- dropping them, or rotating them with
//     wrapped-around angles, silently changes the model.
//   * rot = 0 is legal and makes this a pure strided transpose, which is
//     how the V half of the same projection is laid out for attention.
//
// The cos/sin tables are f32 and hold rot/2 entries per row (the second
// half of the concatenated angle block repeats the first, so storing it
// would be redundant). f32 for the reason the sibling DiTs use it: RoPE
// error is structured, so a bf16 table compounds across 50 blocks and
// every denoise step instead of averaging out.
// `in_head_stride` is how far apart two HEADS sit in the source row. It
// is NOT always D: a fused qkv projection can be grouped either way, and
// which one a checkpoint uses is not visible from the shapes.
//
//   * [q0..qH | k0..kH | v0..vH] -- head stride D, part offset i*H*D.
//   * [h0(q,k,v) | h1(q,k,v) | ...] -- head stride 3*D, part offset i*D.
//     MiniMax-H3's video VAE is this one; reading it as the first layout
//     hands every head a mixture of its own q, k and v, which destroys
//     attention while still producing plausible per-token output.
//
//   0:in 1:out 2:cos[seq,rot/2] 3:sin[seq,rot/2] 4:H 5:seq 6:D 7:rot
//   8:in_stride 9:in_off 10:in_head_stride.
//   grid {D, seq, H}, tg {D, 1, 1}.
kernel void transpose_rope_half_part_ftab_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    device VPIPE_ELT*       out  [[buffer(1)]],
    const device float*     cosb [[buffer(2)]],
    const device float*     sinb [[buffer(3)]],
    constant int&      H         [[buffer(4)]],
    constant int&      S         [[buffer(5)]],
    constant int&      D         [[buffer(6)]],
    constant int&      rot       [[buffer(7)]],
    constant int&      in_stride [[buffer(8)]],
    constant int&      in_off    [[buffer(9)]],
    constant int&      in_head_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  // Bound the head axis for the same reason `d` is bounded: both are
  // exact for today's dispatch ({D, S, H}), and a caller that ever pads
  // the grid up to a threadgroup multiple would otherwise write past
  // `out` rather than be clamped.
  if (h >= H) { return; }
  const uint src = (uint)t * (uint)in_stride + (uint)in_off +
                   (uint)h * (uint)in_head_stride;
  const uint dst = ((uint)h * (uint)S + (uint)t) * (uint)D;
  if (d >= rot) {
    out[dst + (uint)d] = in[src + (uint)d];   // pass-through tail
    return;
  }
  const int half_r = rot / 2;
  // Angles repeat across the two halves, so both halves index [0, half_r).
  const int i = (d < half_r) ? d : (d - half_r);
  const uint cb = (uint)t * (uint)half_r + (uint)i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const float x1 = float(in[src + (uint)(d < half_r ? d : d - half_r)]);
  const float x2 = float(in[src + (uint)(d < half_r ? d + half_r : d)]);
  // rotate_half: out[:half] = x1*c - x2*s, out[half:] = x2*c + x1*s.
  out[dst + (uint)d] =
      VPIPE_ELT((d < half_r) ? (x1 * c - x2 * s) : (x2 * c + x1 * s));
}

// The same partial rotate-half RoPE as above, IN PLACE on the fused qkv
// projection and with NO transpose.
//
// The transposing version exists because the steel flash-attention
// kernel wants [H, seq, D]. It does not have to: steel takes per-axis
// STRIDES for Q, K, V and O, so it can read the heads straight out of
// the [seq, 3*H*D] projection where the qkv GEMM left them. Once it
// does, the only thing the transpose pass was still carrying is the
// rotation -- and that is a rewrite of the same bytes, which is what
// this kernel is.
//
// The saving is four full passes over a [seq, H*D] activation per block
// (q, k, v out, o back), each of them a read plus a write of the whole
// thing. At video sequence lengths that is gigabytes of traffic per
// block against an arithmetic-free permutation.
//
// V never rotates and so needs NOTHING here: the strided read replaces
// its transpose outright.
//
// IN-PLACE MEANS A HAZARD. Channel d is built from d and its partner
// d +/- rot/2, so a thread that writes before its partner has read hands
// the partner a rotated value. Both halves live in the SAME threadgroup
// (the dispatch is one threadgroup per (row, head), D threads wide), so
// one barrier between the reads and the writes settles it -- and every
// thread must reach that barrier, including the `d >= rot` tail that has
// nothing to do, which is why the tail is skipped with a flag rather
// than an early return.
//
//   0:x 1:cos[seq,rot/2] 2:sin[seq,rot/2] 3:H 4:seq 5:D 6:rot 7:stride
//   8:off 9:head_stride.
//   grid {D, seq, H}, tg {D, 1, 1}.
kernel void rope_half_part_ftab_inplace_f16(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device float*     cosb [[buffer(1)]],
    const device float*     sinb [[buffer(2)]],
    constant int&      H         [[buffer(3)]],
    constant int&      S        [[buffer(4)]],
    constant int&      D         [[buffer(5)]],
    constant int&      rot       [[buffer(6)]],
    constant int&      stride    [[buffer(7)]],
    constant int&      off       [[buffer(8)]],
    constant int&      head_stride [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  // Both halves of the pair are read before the barrier and the result
  // written after it, so `live` gates the work and never the barrier.
  const bool live = d < D && t < S && h < H && d < rot;
  const int half_r = rot / 2;
  const uint src = (uint)t * (uint)stride + (uint)off +
                   (uint)h * (uint)head_stride;
  float x1 = 0.0f, x2 = 0.0f, c = 0.0f, s = 0.0f;
  if (live) {
    const int i = (d < half_r) ? d : (d - half_r);
    const uint cb = (uint)t * (uint)half_r + (uint)i;
    c = cosb[cb];
    s = sinb[cb];
    x1 = float(x[src + (uint)(d < half_r ? d : d - half_r)]);
    x2 = float(x[src + (uint)(d < half_r ? d + half_r : d)]);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (live) {
    x[src + (uint)d] =
        VPIPE_ELT((d < half_r) ? (x1 * c - x2 * s) : (x2 * c + x1 * s));
  }
}

// Per-HEAD RMS norm in place over a strided fused projection: the row
// (t, h) lives at `x[t*stride + off + h*D]` and is normalized over its
// own D channels against a shared [D] gamma.
//
// MiniMax-H3's q/k normalization is per head over head_dim, which is the
// usual convention but NOT Wan's -- Wan norms once across the whole
// projection before the head split. In place, and on the fused buffer,
// so the subsequent rope pass reads the normalized values without a
// second copy of a gigabyte-scale activation.
// `head_stride` is the distance between two HEADS in the row -- see the
// note on transpose_rope_half_part_ftab_f16 above for why that is not
// always D.
//   0:x 1:gamma[D] 2:S 3:H 4:D 5:stride 6:off 7:eps 8:head_stride
//   grid {32, S*H, 1}, tg {32, 1, 1}.
kernel void rms_norm_heads_strided_f16(
    device VPIPE_ELT*       x     [[buffer(0)]],
    const device VPIPE_ELT* gamma [[buffer(1)]],
    constant int&      S      [[buffer(2)]],
    constant int&      H      [[buffer(3)]],
    constant int&      D      [[buffer(4)]],
    constant int&      stride [[buffer(5)]],
    constant int&      off    [[buffer(6)]],
    constant float&    eps    [[buffer(7)]],
    constant int&      head_stride [[buffer(8)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
  const int row = (int)tid.y;
  if (row >= S * H) { return; }
  const int t = row / H;
  const int h = row % H;
  device VPIPE_ELT* r = x + (uint)t * (uint)stride + (uint)off +
                        (uint)h * (uint)head_stride;
  const uint lid = ltid.x;
  float acc = 0.0f;
  for (int i = (int)lid; i < D; i += 32) {
    const float v = float(r[i]);
    acc += v * v;
  }
  acc = simd_sum(acc);
  const float inv = rsqrt(acc / (float)D + eps);
  for (int i = (int)lid; i < D; i += 32) {
    r[i] = VPIPE_ELT(float(r[i]) * inv * float(gamma[i]));
  }
}
