// rms_norm.metal -- RMSNorm for LLM inference on metal-compute.
//   out[r,i] = weight[i] * x[r,i] * rsqrt(mean_i(x[r,i]^2) + eps)
// f16/bf16 in/out, float32 accumulation, one threadgroup per row. The sum(x^2)
// reduction is an ORDER-INVARIANT fixed binary tree over an FP32 threadgroup
// buffer (RMS_PAD-padded): the tree order depends only on RMS_PAD, NOT the
// thread count, so the result is BIT-IDENTICAL for any RMS_TREE_TG and across
// GPUs -- a faster (more-thread) kernel can't perturb a near-tie. (The earlier
// simd_sum+partials reduction's grouping changed with thread count, flipping
// greedy tokens at exact f32 logit ties; see GEMMA-E4B-DECODE-GAP §27-§30.)
//
//   0: x      (device const half*)  [R, H]
//   1: weight (device const half*)  [H]
//   2: out    (device half*)        [R, H]
//   3: H      (constant int&)       hidden dim
//   4: eps    (constant float&)

#include <metal_stdlib>
using namespace metal;

// Element (storage) type. Default half; compiled a second time with
// -DVPIPE_ELT=bfloat for the bf16 variant metallib. Math stays f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// RMS_TG: threadgroup size for the (legacy simd-reduction) layer_norm_relu_f16
// audio kernel, dispatched at 256. RMS_TREE_TG: threadgroup size for the tree
// rms_norm_f16 / rms_add_f16 (dispatched at 512 -- more threads pay for the
// extra tree barriers and net faster). RMS_PAD: power-of-2 pad >= max H.
#define RMS_TG 256
#define RMS_TREE_TG 512
#define RMS_PAD 4096

kernel void rms_norm_f16(
    const device VPIPE_ELT*  x      [[buffer(0)]],
    const device VPIPE_ELT*  weight [[buffer(1)]],
    device VPIPE_ELT*        out    [[buffer(2)]],
    constant int&       H      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint3 tptg     [[threads_per_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const uint tg = tptg.x;                 // ACTUAL tg size (256 or 512), not a
                                          // hardcoded stride -> correct for any
                                          // launch site (qwen/llama at 256).
  const device VPIPE_ELT* xr = x + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;

  // Order-invariant fixed-tree sum(x^2) over an FP32 threadgroup buffer. The
  // tree order depends only on RMS_PAD (the s-halving), NOT tg -> bit-identical
  // for any threadgroup size. Caps H <= RMS_PAD (host selects the simd_sum
  // rms_norm_fast_f16 for larger H; see the helper guard).
  threadgroup float sbuf[RMS_PAD];
  for (uint i = lid; i < RMS_PAD; i += tg) {
    const float v = (i < (uint)H) ? float(xr[i]) : 0.0f;
    sbuf[i] = v * v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = RMS_PAD / 2; s >= 1; s >>= 1) {
    for (uint i = lid; i < (uint)s; i += tg) { sbuf[i] += sbuf[i + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(sbuf[0] / float(H) + eps);
  for (uint i = lid; i < (uint)H; i += tg) {
    outr[i] = VPIPE_ELT(float(xr[i]) * inv * float(weight[i]));
  }
}

// layer_norm_relu_f16 -- LayerNorm (weight only, no bias) + ReLU, used by
// the Gemma-4 audio SSCP conv stem: out[r,i] = max(0, (x[r,i] - mean) *
// rsqrt(var + eps) * weight[i]) with population var = E[x^2] - mean^2.
// Same per-row reduction as rms_norm_f16 (one threadgroup per row).
//   0:x[R,H] 1:weight[H] 2:out[R,H] 3:H 4:eps.
kernel void layer_norm_relu_f16(
    const device VPIPE_ELT*  x      [[buffer(0)]],
    const device VPIPE_ELT*  weight [[buffer(1)]],
    device VPIPE_ELT*        out    [[buffer(2)]],
    constant int&       H      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* xr = x + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;

  float s1 = 0.0f, s2 = 0.0f;
  for (int i = (int)lid; i < H; i += RMS_TG) {
    const float v = float(xr[i]);
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  threadgroup float p1[RMS_TG / 32], p2[RMS_TG / 32];
  if (simd_lid == 0) { p1[simd_gid] = s1; p2[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < RMS_TG / 32) ? p1[simd_lid] : 0.0f;
    float b = (simd_lid < RMS_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); b = simd_sum(b);
    if (simd_lid == 0) { p1[0] = a; p2[0] = b; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = p1[0] / float(H);
  const float var = p2[0] / float(H) - mean * mean;
  const float inv = rsqrt(var + eps);
  for (int i = (int)lid; i < H; i += RMS_TG) {
    const float v = (float(xr[i]) - mean) * inv * float(weight[i]);
    outr[i] = VPIPE_ELT(max(v, 0.0f));
  }
}

// rms_add_f16 -- fused sublayer-output norm + residual add (+ optional
// post-scale): out[r,i] = (residual[r,i] + weight[i]*x[r,i]*inv) * post_scale.
// Collapses the Gemma sandwich `rms(sublayer_out); residual(h, sublayer_out)`
// (and the per-layer `* layer_scalar`) into ONE dispatch -- cutting decode
// per-dispatch barrier bubbles. Same per-row reduction as rms_norm_f16.
//   0:x 1:weight 2:residual 3:out (VPIPE_ELT)  4:H 5:eps 6:post_scale
kernel void rms_add_f16(
    const device VPIPE_ELT*  x        [[buffer(0)]],
    const device VPIPE_ELT*  weight   [[buffer(1)]],
    const device VPIPE_ELT*  residual [[buffer(2)]],
    device VPIPE_ELT*        out      [[buffer(3)]],
    constant int&       H          [[buffer(4)]],
    constant float&     eps        [[buffer(5)]],
    constant float&     post_scale [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint3 tptg     [[threads_per_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const uint tg = tptg.x;                 // ACTUAL tg size, not a fixed stride.
  const device VPIPE_ELT* xr = x + (uint)row * H;
  const device VPIPE_ELT* rr = residual + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;

  // Order-invariant fixed-tree sum(x^2) (bit-identical for any tg; H <= RMS_PAD).
  threadgroup float sbuf[RMS_PAD];
  for (uint i = lid; i < RMS_PAD; i += tg) {
    const float v = (i < (uint)H) ? float(xr[i]) : 0.0f;
    sbuf[i] = v * v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = RMS_PAD / 2; s >= 1; s >>= 1) {
    for (uint i = lid; i < (uint)s; i += tg) { sbuf[i] += sbuf[i + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(sbuf[0] / float(H) + eps);
  for (uint i = lid; i < (uint)H; i += tg) {
    const float normed = float(xr[i]) * inv * float(weight[i]);
    outr[i] = VPIPE_ELT((float(rr[i]) + normed) * post_scale);
  }
}

// FAST variants of rms_norm_f16 / rms_add_f16: simd_sum reduction (2 barriers,
// reduces only the actual H) instead of the order-invariant RMS_PAD tree (12
// barriers, reduces 4096). This is MLX's rms_single_row structure (and what
// vpipe's own per-head rms_rope3 norm already uses) -- it is what rms_norm_f16
// WAS before f1ab287, which qwen/llama still launch at 256 threads. The tree
// replaced it for CROSS-CONFIG bit-exactness (RMS_TG 256 vs 512 flipped greedy
// ties, §27-30); the simd_sum grouping is deterministic for a FIXED launch and
// matches MLX's reduction order more closely. THREADGROUP-SIZE-AGNOSTIC: strides
// by the actual [[threads_per_threadgroup]] and bounds the cross-simd reduce by
// the real simdgroup count -> correct at 256 (qwen/llama) AND 512 (gemma), with
// NO RMS_PAD cap (future large H is safe). DEFAULT path for all models (§53);
// VPIPE_GEMMA_RMS_FAST=0 reverts gemma to the tree for A/B.
kernel void rms_norm_fast_f16(
    const device VPIPE_ELT*  x      [[buffer(0)]],
    const device VPIPE_ELT*  weight [[buffer(1)]],
    device VPIPE_ELT*        out    [[buffer(2)]],
    constant int&       H      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint3 tptg     [[threads_per_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const uint tg = tptg.x;
  const uint nsg = tg / 32;               // actual simdgroups (8@256, 16@512)
  const device VPIPE_ELT* xr = x + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;
  float acc = 0.0f;
  for (uint i = lid; i < (uint)H; i += tg) {
    const float v = float(xr[i]); acc += v * v;
  }
  acc = simd_sum(acc);
  threadgroup float partial[32];          // max tg 1024 -> 32 simdgroups
  if (simd_lid == 0) { partial[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < nsg) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(H) + eps);
  for (uint i = lid; i < (uint)H; i += tg) {
    outr[i] = VPIPE_ELT(float(xr[i]) * inv * float(weight[i]));
  }
}

kernel void rms_add_fast_f16(
    const device VPIPE_ELT*  x        [[buffer(0)]],
    const device VPIPE_ELT*  weight   [[buffer(1)]],
    const device VPIPE_ELT*  residual [[buffer(2)]],
    device VPIPE_ELT*        out      [[buffer(3)]],
    constant int&       H          [[buffer(4)]],
    constant float&     eps        [[buffer(5)]],
    constant float&     post_scale [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint3 tptg     [[threads_per_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const uint tg = tptg.x;
  const uint nsg = tg / 32;
  const device VPIPE_ELT* xr = x + (uint)row * H;
  const device VPIPE_ELT* rr = residual + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;
  float acc = 0.0f;
  for (uint i = lid; i < (uint)H; i += tg) {
    const float v = float(xr[i]); acc += v * v;
  }
  acc = simd_sum(acc);
  threadgroup float partial[32];
  if (simd_lid == 0) { partial[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < nsg) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(H) + eps);
  for (uint i = lid; i < (uint)H; i += tg) {
    const float normed = float(xr[i]) * inv * float(weight[i]);
    outr[i] = VPIPE_ELT((float(rr[i]) + normed) * post_scale);
  }
}
