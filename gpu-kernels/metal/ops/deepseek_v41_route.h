// deepseek_v41_route.h -- what DeepSeek-V4.1's route and its handshakes
// share: the mailbox checksum's mix and the route's body by a group. Both
// deepseek_v41.metal and host_handshake.metal (the system-scope
// handshake, a library of its own) include it, so the route folded into
// either handshake is the route kernel's own, bit for bit.
//
// The includer brings <metal_stdlib> and `using namespace metal;` first.

#pragma once

// dsv41_handshake_io's checksum term for word `k` holding `v` (the host's
// is hs_sum_, metal-deepseek-v41-model.cc).
inline uint dsv41_hs_mix_(uint k, uint v)
{
  return v * (2u * k + 1u) + (v >> 16) * 0x9E3779B9u;
}

// dsv41_route's body (deepseek_v41.metal) for row `row`, by a group of
// 256: the route kernel's own and, folded, each handshake's.
inline void dsv41_route_tg_(
    const device float* sc, const device float* bias,
    const device float* bias_vl, const device uchar* vl, device int* eid,
    device float* wt, const int E, const int topk, const float rscale,
    const int norm, const int has_vl, const uint row, const uint tid,
    const uint sg, const uint lane, threadgroup float* val,
    threadgroup float* sel, threadgroup float* rv, threadgroup int* ri,
    threadgroup int* pick)
{
  const bool use_vl = has_vl != 0 && vl[row] != 0;
  for (int e = (int)tid; e < E; e += 256) {
    const float x = sc[(long)row * E + e];
    const float sp = x > 20.0f ? x : log(1.0f + exp(x));
    const float v = sqrt(sp);
    val[e] = v;
    sel[e] = v + (use_vl ? bias_vl[e] : bias[e]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k = 0; k < topk; ++k) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int e = (int)tid; e < E; e += 256) {
      const float v = sel[e];
      if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, (ushort)off);
      const int oi = simd_shuffle_down(bi, (ushort)off);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { rv[sg] = bv; ri[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float b = rv[0];
      int bi2 = ri[0];
      for (int g = 1; g < 8; ++g) {
        if (rv[g] > b || (rv[g] == b && ri[g] < bi2)) {
          b = rv[g];
          bi2 = ri[g];
        }
      }
      pick[k] = bi2;
      sel[bi2] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) { sum += val[pick[k]]; }
    const float dn = (norm != 0 && topk > 1) ? sum + 1e-20f : 1.0f;
    for (int k = 0; k < topk; ++k) {
      eid[row * topk + k] = pick[k];
      wt[row * topk + k] = val[pick[k]] / dn * rscale;
    }
  }
}
