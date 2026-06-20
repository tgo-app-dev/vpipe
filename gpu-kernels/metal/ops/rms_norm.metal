// rms_norm.metal -- RMSNorm for LLM inference on metal-compute.
//   out[r,i] = weight[i] * x[r,i] * rsqrt(mean_i(x[r,i]^2) + eps)
// Matches mlx::fast::rms_norm: per-row reduction over the hidden dim,
// f16 in/out with float32 accumulation. One threadgroup per row;
// reduce sum(x^2) via simd_sum across 8 simdgroups (256 threads).
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

#define RMS_TG 256

kernel void rms_norm_f16(
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

  float local = 0.0f;
  for (int i = (int)lid; i < H; i += RMS_TG) {
    const float v = float(xr[i]);
    local += v * v;
  }
  local = simd_sum(local);

  threadgroup float partial[RMS_TG / 32];
  if (simd_lid == 0) {
    partial[simd_gid] = local;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0) {
    float p = (simd_lid < RMS_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) {
      partial[0] = p;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = rsqrt(partial[0] / float(H) + eps);
  for (int i = (int)lid; i < H; i += RMS_TG) {
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
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* xr = x + (uint)row * H;
  const device VPIPE_ELT* rr = residual + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;

  float local = 0.0f;
  for (int i = (int)lid; i < H; i += RMS_TG) {
    const float v = float(xr[i]);
    local += v * v;
  }
  local = simd_sum(local);

  threadgroup float partial[RMS_TG / 32];
  if (simd_lid == 0) { partial[simd_gid] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float p = (simd_lid < RMS_TG / 32) ? partial[simd_lid] : 0.0f;
    p = simd_sum(p);
    if (simd_lid == 0) { partial[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = rsqrt(partial[0] / float(H) + eps);
  for (int i = (int)lid; i < H; i += RMS_TG) {
    const float normed = float(xr[i]) * inv * float(weight[i]);
    outr[i] = VPIPE_ELT((float(rr[i]) + normed) * post_scale);
  }
}
