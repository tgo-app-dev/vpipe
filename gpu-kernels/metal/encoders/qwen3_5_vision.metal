// qwen3_5_vision.metal -- Qwen3-VL vision-tower pointwise / norm / rope
// kernels (the ViT is dense, not quantized). LayerNorm has a bias and
// subtracts the mean (unlike the LM's RMSNorm); GELU comes in both the
// tanh-approx (ViT MLP) and exact-erf (patch merger / DeepStack) forms;
// 2D-RoPE uses precomputed per-patch cos/sin tables and full-head
// rotate-half. f16 storage, f32 math.

#include <metal_stdlib>
using namespace metal;

// LayerNorm with weight+bias over the last dim H, one threadgroup/row.
//   out[r,i] = (x[r,i]-mean_r)*rsqrt(var_r+eps)*weight[i] + bias[i]
//   0:x[R,H] 1:weight[H] 2:bias[H] 3:out[R,H] 4:H 5:eps
#define LN_TG 256
kernel void layer_norm_bias_f16(
    const device half*  x      [[buffer(0)]],
    const device half*  weight [[buffer(1)]],
    const device half*  bias   [[buffer(2)]],
    device half*        out    [[buffer(3)]],
    constant int&       H      [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device half* xr = x + (uint)row * H;
  device half* outr = out + (uint)row * H;

  float s1 = 0.0f, s2 = 0.0f;
  for (int i = (int)lid; i < H; i += LN_TG) {
    const float v = (float)xr[i];
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  threadgroup float p1[LN_TG / 32], p2[LN_TG / 32];
  if (simd_lid == 0) { p1[simd_gid] = s1; p2[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < LN_TG / 32) ? p1[simd_lid] : 0.0f;
    float b = (simd_lid < LN_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); b = simd_sum(b);
    if (simd_lid == 0) { p1[0] = a; p2[0] = b; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = p1[0] / (float)H;
  const float var = p2[0] / (float)H - mean * mean;
  const float inv = rsqrt(var + eps);
  for (int i = (int)lid; i < H; i += LN_TG) {
    outr[i] = (half)(((float)xr[i] - mean) * inv * (float)weight[i] +
                     (float)bias[i]);
  }
}

// GELU (tanh approx): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
//   0:x 1:out 2:n
kernel void gelu_tanh_f16(
    const device half* x   [[buffer(0)]],
    device half*       out [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float v = (float)x[gid];
  const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
  const float inner = k0 * (v + 0.044715f * v * v * v);
  out[gid] = (half)(0.5f * v * (1.0f + metal::precise::tanh(inner)));
}

// Metal has no erf intrinsic; Abramowitz & Stegun 7.1.26 (max abs error
// ~1.5e-7, well under f16 precision).
inline float erf_approx_(float x) {
  const float p = 0.3275911f;
  const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f,
              a4 = -1.453152027f, a5 = 1.061405429f;
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  const float ax = metal::fabs(x);
  const float t = 1.0f / (1.0f + p * ax);
  const float y = 1.0f -
      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * metal::exp(-ax * ax);
  return s * y;
}

// GELU (exact, erf): 0.5*x*(1+erf(x/sqrt(2))).   0:x 1:out 2:n
kernel void gelu_erf_f16(
    const device half* x   [[buffer(0)]],
    device half*       out [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float v = (float)x[gid];
  out[gid] = (half)(0.5f * v * (1.0f + erf_approx_(v * 0.7071067811865476f)));
}

// Vision 2D-RoPE: q'[p,h,:] = q*cos[p] + rotate_half(q)*sin[p], where
// rotate_half([a|b]) = [-b|a] over the full head_dim. cos/sin tables are
// [n_patches, head_dim] (broadcast over heads). Separate out buffer
// (rotate-half couples d and d +/- D/2, so in-place would race).
//   0:q_in[n,Hh,D] 1:cos[n,D] 2:sin[n,D] 3:q_out[n,Hh,D] 4:Hh 5:D
kernel void vision_rope_f16(
    const device half*  q_in  [[buffer(0)]],
    const device half*  cos_t [[buffer(1)]],
    const device half*  sin_t [[buffer(2)]],
    device half*        q_out [[buffer(3)]],
    constant int&       Hh    [[buffer(4)]],
    constant int&       D     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const uint total = (uint)D;   // grid is flattened below
  (void)total;
  const int d = (int)(gid % (uint)D);
  const int rest = (int)(gid / (uint)D);
  const int h = rest % Hh;
  const int p = rest / Hh;
  const int hd2 = D / 2;
  const uint base = ((uint)p * Hh + h) * D;
  const float qv = (float)q_in[base + d];
  const float rot = (d < hd2) ? -(float)q_in[base + d + hd2]
                              : (float)q_in[base + d - hd2];
  q_out[base + d] = (half)(qv * (float)cos_t[(uint)p * D + d] +
                           rot * (float)sin_t[(uint)p * D + d]);
}

// Gemma-4 vision 2-D RoPE. The head dim splits into a col-half [0,D/2)
// and a row-half [D/2,D); rotate-half is applied INDEPENDENTLY within
// each half (couples d and d +/- D/4 inside its own half), NOT across
// the full head like vision_rope_f16. cos/sin tables are [n,D]
// (broadcast over heads), prebuilt host-side with col freqs in the
// first half and row freqs in the second.
//   0:q_in[n,Hh,D] 1:cos[n,D] 2:sin[n,D] 3:q_out[n,Hh,D] 4:Hh 5:D
kernel void gemma_vision_rope_f16(
    const device half*  q_in  [[buffer(0)]],
    const device half*  cos_t [[buffer(1)]],
    const device half*  sin_t [[buffer(2)]],
    device half*        q_out [[buffer(3)]],
    constant int&       Hh    [[buffer(4)]],
    constant int&       D     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const int d    = (int)(gid % (uint)D);
  const int rest = (int)(gid / (uint)D);
  const int h    = rest % Hh;
  const int p    = rest / Hh;
  const int span = D / 2;          // per-axis span
  const int roff = D / 4;          // within-half rotate offset
  const int blk  = d / span;       // 0 col-half, 1 row-half
  const int sub  = d % span;       // 0..span-1
  const int partner = blk * span + ((sub < roff) ? sub + roff : sub - roff);
  const float sign  = (sub < roff) ? -1.0f : 1.0f;
  const uint base   = ((uint)p * Hh + h) * D;
  const float qv = (float)q_in[base + d];
  const float qp = (float)q_in[base + partner];
  q_out[base + d] = (half)(qv * (float)cos_t[(uint)p * D + d] +
                           sign * qp * (float)sin_t[(uint)p * D + d]);
}
