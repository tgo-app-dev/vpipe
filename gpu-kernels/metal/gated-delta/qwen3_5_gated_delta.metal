// qwen3_5_gated_delta.metal -- Qwen3.5 gated-DeltaNet (linear-attention)
// recurrent step on the metal-compute framework. Faithful port of the
// MLX fast::metal_kernel "qwen3_5_gated_delta_step" in
// generative-models/qwen3/qwen3-5-model-exec.cc (itself a port of mlx-lm's
// gated_delta.py _make_gated_delta_kernel, vectorized=False).
//
// One threadgroup of 32 (Dk lanes) x 4 (Dv lanes) processes one
// (batch, hv-head) pair. Each simdgroup lane holds Dk/32 fp32 state
// elements; the delta-rule recurrence runs in registers and writes the
// state back to global memory at the end. simd_sum reduces the per-lane
// partials across the 32 Dk lanes of a simdgroup.
//
// Dims are fixed to the Qwen3.5 GDN config, which is identical for the
// 4B and 9B checkpoints: Hk=16, Hv=32, Dk=Dv=128 (so Dk/32 = 4 state
// elems per lane). The GQA pattern feeds Hv/Hk = 2 v-heads per k-head.
//
// Buffers (q/k/v half, g/beta/state float; mlx-lm layout, not 4D KV):
//   0: q         [B, T, Hk, Dk]  half
//   1: k         [B, T, Hk, Dk]  half
//   2: v         [B, T, Hv, Dv]  half
//   3: g         [B, T, Hv]      float   (per-step, per-v-head decay)
//   4: beta      [B, T, Hv]      float
//   5: state_in  [B, Hv, Dv, Dk] float
//   6: y         [B, T, Hv, Dv]  half    (out)
//   7: state_out [B, Hv, Dv, Dk] float   (out)
//   8: T         int (constant)
// Dispatch: grid {32, Dv, B*Hv}, threadgroup {32, 4, 1}.

#include <metal_stdlib>
using namespace metal;

// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the
// bf16 variant metallib. Math stays f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// GDN head COUNTS (Hk, Hv) are passed as runtime constants -- they vary
// across the Qwen3.5 family (2B: 16/16, 4B: 16/32, ...). The per-HEAD
// dims (key/value head_dim) are a stable 128 throughout the family, so
// GDN_DK/GDN_DV/GDN_NPT stay compile-time (GDN_NPT also sizes the
// register state array, which must be a compile-time bound).
#define GDN_DK 128
#define GDN_DV 128
#define GDN_NPT (GDN_DK / 32)   // state elements per simdgroup lane (4)

kernel void qwen3_5_gated_delta_step_f16(
    const device VPIPE_ELT*  q         [[buffer(0)]],
    const device VPIPE_ELT*  k         [[buffer(1)]],
    const device VPIPE_ELT*  v         [[buffer(2)]],
    const device float* g         [[buffer(3)]],
    const device float* beta      [[buffer(4)]],
    const device float* state_in  [[buffer(5)]],
    device VPIPE_ELT*        y         [[buffer(6)]],
    device float*       state_out [[buffer(7)]],
    constant int&       T         [[buffer(8)]],
    constant int&       Hk        [[buffer(9)]],
    constant int&       Hv        [[buffer(10)]],
    uint3 tpig  [[thread_position_in_grid]],
    uint3 tpit  [[thread_position_in_threadgroup]],
    uint  silid [[thread_index_in_simdgroup]])
{
  const uint Hku = (uint)(Hk < 0 ? -Hk : Hk), Hvu = (uint)Hv;
  const uint n = tpig.z;                       // b * Hv + hv
  const uint b_idx = n / Hvu;
  const uint hv_idx = n % Hvu;
  const uint hk_idx =
      (Hk < 0) ? (hv_idx % Hku) : (hv_idx / (Hvu / Hku));

  const uint Tu = (uint)T;
  const device VPIPE_ELT* q_ = q + (b_idx * Tu * Hku + hk_idx) * GDN_DK;
  const device VPIPE_ELT* k_ = k + (b_idx * Tu * Hku + hk_idx) * GDN_DK;
  const device VPIPE_ELT* v_ = v + (b_idx * Tu * Hvu + hv_idx) * GDN_DV;
  device VPIPE_ELT* y_ = y + (b_idx * Tu * Hvu + hv_idx) * GDN_DV;

  const uint dk_idx = tpit.x;                   // 0..31  (lane within simdgroup)
  const uint dv_idx = tpig.y;                   // 0..Dv-1

  const device float* i_state = state_in + (n * GDN_DV + dv_idx) * GDN_DK;
  device float* o_state = state_out + (n * GDN_DV + dv_idx) * GDN_DK;

  float state[GDN_NPT];
  for (int i = 0; i < GDN_NPT; ++i) {
    state[i] = i_state[GDN_NPT * dk_idx + i];
  }

  const device float* g_ = g + b_idx * Tu * Hvu;
  const device float* beta_ = beta + b_idx * Tu * Hvu;

  for (int t = 0; t < T; ++t) {
    float kv_mem = 0.0f;
    for (int i = 0; i < GDN_NPT; ++i) {
      const uint s_idx = GDN_NPT * dk_idx + i;
      state[i] = state[i] * g_[hv_idx];
      kv_mem += state[i] * (float)k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    const float delta = ((float)v_[dv_idx] - kv_mem) * beta_[hv_idx];

    float out = 0.0f;
    for (int i = 0; i < GDN_NPT; ++i) {
      const uint s_idx = GDN_NPT * dk_idx + i;
      state[i] = state[i] + (float)k_[s_idx] * delta;
      out += state[i] * (float)q_[s_idx];
    }
    out = simd_sum(out);
    if (silid == 0) {
      y_[dv_idx] = (VPIPE_ELT)out;
    }

    q_ += Hku * GDN_DK;
    k_ += Hku * GDN_DK;
    v_ += Hvu * GDN_DV;
    y_ += Hvu * GDN_DV;
    g_ += Hvu;
    beta_ += Hvu;
  }

  for (int i = 0; i < GDN_NPT; ++i) {
    o_state[GDN_NPT * dk_idx + i] = state[i];
  }
}

// Recurrent step v2: each simdgroup handles NDV consecutive dv, loading the
// per-step k/q ONCE into registers and reusing across all NDV dv (the
// original re-reads k/q from every dv-simdgroup of a head each step). NDV is
// COMPILE-TIME so the state array is exactly NDV*GDN_NPT registers -- a
// runtime NDV forces a max-sized array and the register bloat alone makes it
// ~2x slower, masking any amortization. Math per dv is byte-identical to v1
// (same accumulation + simd_sum) -> token-exact.
//   ...buffers 0-10 as v1...   Dispatch: grid {32, Dv/NDV, Hv}, tg {32,4,1}.
template <int NDV>
static inline void gdn_step_ndv_impl(
    const device VPIPE_ELT* q, const device VPIPE_ELT* k,
    const device VPIPE_ELT* v, const device float* g, const device float* beta,
    const device float* state_in, device VPIPE_ELT* y, device float* state_out,
    int T, int Hk, int Hv, uint3 tpig, uint3 tpit, uint silid)
{
  const uint Hku = (uint)(Hk < 0 ? -Hk : Hk), Hvu = (uint)Hv;
  const uint n = tpig.z;
  const uint b_idx = n / Hvu;
  const uint hv_idx = n % Hvu;
  const uint hk_idx =
      (Hk < 0) ? (hv_idx % Hku) : (hv_idx / (Hvu / Hku));

  const uint Tu = (uint)T;
  const device VPIPE_ELT* q_ = q + (b_idx * Tu * Hku + hk_idx) * GDN_DK;
  const device VPIPE_ELT* k_ = k + (b_idx * Tu * Hku + hk_idx) * GDN_DK;
  const device VPIPE_ELT* v_ = v + (b_idx * Tu * Hvu + hv_idx) * GDN_DV;
  device VPIPE_ELT* y_ = y + (b_idx * Tu * Hvu + hv_idx) * GDN_DV;

  const uint dk_idx = tpit.x;
  const uint dv_base = tpig.y * (uint)NDV;

  float state[NDV][GDN_NPT];
  for (int d = 0; d < NDV; ++d) {
    const device float* is = state_in + (n * GDN_DV + dv_base + d) * GDN_DK;
    for (int i = 0; i < GDN_NPT; ++i) {
      state[d][i] = is[GDN_NPT * dk_idx + i];
    }
  }

  const device float* g_ = g + b_idx * Tu * Hvu;
  const device float* beta_ = beta + b_idx * Tu * Hvu;

  for (int t = 0; t < T; ++t) {
    const float g_t = g_[hv_idx];
    const float beta_t = beta_[hv_idx];
    float kreg[GDN_NPT], qreg[GDN_NPT];
    for (int i = 0; i < GDN_NPT; ++i) {
      kreg[i] = (float)k_[GDN_NPT * dk_idx + i];
      qreg[i] = (float)q_[GDN_NPT * dk_idx + i];
    }
    for (int d = 0; d < NDV; ++d) {
      float kv_mem = 0.0f;
      for (int i = 0; i < GDN_NPT; ++i) {
        state[d][i] = state[d][i] * g_t;
        kv_mem += state[d][i] * kreg[i];
      }
      kv_mem = simd_sum(kv_mem);
      const float delta = ((float)v_[dv_base + d] - kv_mem) * beta_t;
      float out = 0.0f;
      for (int i = 0; i < GDN_NPT; ++i) {
        state[d][i] = state[d][i] + kreg[i] * delta;
        out += state[d][i] * qreg[i];
      }
      out = simd_sum(out);
      if (silid == 0) { y_[dv_base + d] = (VPIPE_ELT)out; }
    }
    q_ += Hku * GDN_DK; k_ += Hku * GDN_DK;
    v_ += Hvu * GDN_DV; y_ += Hvu * GDN_DV;
    g_ += Hvu; beta_ += Hvu;
  }

  for (int d = 0; d < NDV; ++d) {
    device float* os = state_out + (n * GDN_DV + dv_base + d) * GDN_DK;
    for (int i = 0; i < GDN_NPT; ++i) {
      os[GDN_NPT * dk_idx + i] = state[d][i];
    }
  }
}

#define GDN_STEP_NDV(NAME, NDV)                                            \
  kernel void NAME(                                                        \
      const device VPIPE_ELT* q [[buffer(0)]],                            \
      const device VPIPE_ELT* k [[buffer(1)]],                            \
      const device VPIPE_ELT* v [[buffer(2)]],                            \
      const device float* g [[buffer(3)]],                                \
      const device float* beta [[buffer(4)]],                             \
      const device float* state_in [[buffer(5)]],                         \
      device VPIPE_ELT* y [[buffer(6)]],                                  \
      device float* state_out [[buffer(7)]],                              \
      constant int& T [[buffer(8)]], constant int& Hk [[buffer(9)]],      \
      constant int& Hv [[buffer(10)]],                                    \
      uint3 tpig [[thread_position_in_grid]],                             \
      uint3 tpit [[thread_position_in_threadgroup]],                      \
      uint silid [[thread_index_in_simdgroup]]) {                         \
    gdn_step_ndv_impl<NDV>(q, k, v, g, beta, state_in, y, state_out,      \
                           T, Hk, Hv, tpig, tpit, silid);                 \
  }

GDN_STEP_NDV(qwen3_5_gated_delta_step_ndv2_f16, 2)
GDN_STEP_NDV(qwen3_5_gated_delta_step_ndv4_f16, 4)
GDN_STEP_NDV(qwen3_5_gated_delta_step_ndv8_f16, 8)

// Depthwise causal conv1d + SiLU for the GDN block, with the conv_state
// concat AND the state shift folded in -- no inwin materialization, no
// transpose. One thread per channel c; the input window for output t is
//   window(r, c) = r<k-1 ? conv_state[r, c] : x[r-(k-1), c]   (r = t..t+k-1)
//   out[t, c]    = silu( sum_{j=0..k-1} window(t+j, c) * w[c, j] )
// matching mlx::core::conv1d(stride=1, padding=0, groups=C) over
// [conv_state || x] + SiLU. conv_state is then overwritten with the last
// k-1 window rows (the new tail for the next chunk). Handles decode (n=1)
// and prefill (n>1) uniformly. w is the checkpoint conv1d.weight [C,k,1]
// flattened to [C,k] (channel-major). All window reads precede the
// conv_state writes, so the in-place update is race-free per channel.
// x_stride is the row stride of the INPUT x (>= C): lets x be the qkv slice
// of a wider fused in_proj buffer (x_stride = fused N) without a copy.
// keyd (>0) de-INTERLEAVES the output: instead of the [q|k|v]-per-token
// layout (stride C), out is written as three contiguous blocks
// [q(n*keyd) | k(n*keyd) | v(n*vald)] so the downstream q/k-norm and
// delta-rule step read each as a contiguous [n, *] tensor with NO hslice
// copy. keyd==0 keeps the legacy interleaved [n,C] output.
//
// The updated conv tail is written to a SEPARATE output buffer conv_state_out
// (buffer 9) rather than back into conv_state (buffer 0). Callers that bump
// the recurrent state in place bind buffer 9 == buffer 0 (the legacy
// behavior, byte-identical: all window/tail reads precede the writes). The
// pdecode run-ahead ring binds buffer 9 to the NEXT ping-pong slot so the
// previous state survives for rollback -- copy-free GDN snapshot.
//   0:conv_state[(k-1)*C] (in) 1:x[n*x_stride] 2:w[C*k] 3:out
//   4:n 5:C 6:k 7:x_stride 8:keyd 9:conv_state_out[(k-1)*C] (out)
kernel void qwen3_5_gdn_conv1d_silu_f16(
    device VPIPE_ELT*        conv_state [[buffer(0)]],
    const device VPIPE_ELT*  x          [[buffer(1)]],
    const device VPIPE_ELT*  w          [[buffer(2)]],
    device VPIPE_ELT*        out        [[buffer(3)]],
    constant int&       n          [[buffer(4)]],
    constant int&       C          [[buffer(5)]],
    constant int&       k          [[buffer(6)]],
    constant int&       x_stride   [[buffer(7)]],
    constant int&       keyd       [[buffer(8)]],
    device VPIPE_ELT*        conv_state_out [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)C) { return; }
  const uint c = gid;
  const int vald = C - 2 * keyd;
  // Per-token output offset for channel c under the de-interleaved layout.
  // o_base(c) + t*o_stride(c) gives out[t,c].
  uint o_base, o_stride;
  if (keyd <= 0) {                         // legacy interleaved [n,C]
    o_base = c; o_stride = (uint)C;
  } else if ((int)c < keyd) {              // q block
    o_base = c; o_stride = (uint)keyd;
  } else if ((int)c < 2 * keyd) {          // k block
    o_base = (uint)n * keyd + (c - keyd); o_stride = (uint)keyd;
  } else {                                  // v block
    o_base = 2u * (uint)n * keyd + (c - 2 * keyd); o_stride = (uint)vald;
  }
  for (int t = 0; t < n; ++t) {
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
      const int r = t + j;
      const float v = (r < k - 1) ? (float)conv_state[(uint)r * C + c]
                                  : (float)x[(uint)(r - (k - 1)) * x_stride + c];
      acc += v * (float)w[c * (uint)k + j];
    }
    out[o_base + (uint)t * o_stride] = (VPIPE_ELT)(acc / (1.0f + metal::exp(-acc)));
  }
  // New conv_state = last k-1 window rows window(n .. n+k-2). Read into
  // registers before writing (for decode n=1 the read/write indices
  // overlap, and conv_state_out may alias conv_state for in-place callers).
  float ns[8];
  for (int i = 0; i < k - 1; ++i) {
    const int r = n + i;
    ns[i] = (r < k - 1) ? (float)conv_state[(uint)r * C + c]
                        : (float)x[(uint)(r - (k - 1)) * x_stride + c];
  }
  for (int i = 0; i < k - 1; ++i) {
    conv_state_out[(uint)i * C + c] = (VPIPE_ELT)ns[i];
  }
}

// Fused GDN q/k L2-norm: RMS-no-weight over Dk per head, scaled by a
// scalar (s_q for the first Hk rows = q, s_k for the next Hk = k). Acts
// in place on the contiguous q|k region of conv_out [2*Hk, Dk]. Replaces
// the two rms_norm_f16 calls.   0:x[2*Hk,Dk] (in/out) 1:Hk 2:Dk
// 3:s_q 4:s_k 5:eps.  one threadgroup per row, 128 threads.
kernel void qwen3_5_gdn_qk_norm_f16(
    device VPIPE_ELT*    x   [[buffer(0)]],
    constant int&   Hk  [[buffer(1)]],
    constant int&   Dk  [[buffer(2)]],
    constant float& s_q [[buffer(3)]],
    constant float& s_k [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  device VPIPE_ELT* xr = x + row * (uint)Dk;
  const float scale = ((int)row < Hk) ? s_q : s_k;
  float local = 0.0f;
  for (int i = (int)lid; i < Dk; i += 128) {
    const float v = (float)xr[i];
    local += v * v;
  }
  local = simd_sum(local);
  threadgroup float part[4];
  if (sl == 0) { part[sg] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float p = (sl < 4) ? part[sl] : 0.0f;
    p = simd_sum(p);
    if (sl == 0) { part[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(part[0] / (float)Dk + eps) * scale;
  for (int i = (int)lid; i < Dk; i += 128) {
    xr[i] = (VPIPE_ELT)((float)xr[i] * inv);
  }
}

// Fused GDN gated-RMSNorm output: out = rms_norm(y, weight) * silu(z),
// over Dv per v-head (R rows). Replaces rms_norm_f16 + swiglu_f16.
//   0:y[R,Dv] 1:weight[Dv] 2:z[R,Dv] 3:out[R,Dv] 4:Dv 5:eps
kernel void qwen3_5_gdn_gated_rms_f16(
    const device VPIPE_ELT* y      [[buffer(0)]],
    const device VPIPE_ELT* weight [[buffer(1)]],
    const device VPIPE_ELT* z      [[buffer(2)]],
    device VPIPE_ELT*       out    [[buffer(3)]],
    constant int&      Dv     [[buffer(4)]],
    constant float&    eps    [[buffer(5)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  sl   [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* yr = y + row * (uint)Dv;
  const device VPIPE_ELT* zr = z + row * (uint)Dv;
  device VPIPE_ELT* outr = out + row * (uint)Dv;
  float local = 0.0f;
  for (int i = (int)lid; i < Dv; i += 128) {
    const float v = (float)yr[i];
    local += v * v;
  }
  local = simd_sum(local);
  threadgroup float part[4];
  if (sl == 0) { part[sg] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float p = (sl < 4) ? part[sl] : 0.0f;
    p = simd_sum(p);
    if (sl == 0) { part[0] = p; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(part[0] / (float)Dv + eps);
  for (int i = (int)lid; i < Dv; i += 128) {
    const float rn = (float)yr[i] * inv * (float)weight[i];
    const float zv = (float)zr[i];
    outr[i] = (VPIPE_ELT)(rn * (zv / (1.0f + metal::exp(-zv))));   // * silu(z)
  }
}

// GDN gating: per (step, v-head)
//   g    = exp(-exp(A_log) * softplus(a + dt_bias))
//   beta = sigmoid(b)
// a,b are [n, Hv] half (the in_proj_a/b matmul outputs); A_log, dt_bias
// are [Hv] f32 (broadcast over n); g, beta are [n, Hv] f32 (the
// recurrence kernel reads them as f32). softplus computed stably as
// max(x,0) + log1p(exp(-|x|)).
//   0:a[n,Hv] 1:b[n,Hv] 2:A_log[Hv] 3:dt_bias[Hv]
//   4:g[n,Hv](out) 5:beta[n,Hv](out)  6:Hv  7:n
kernel void qwen3_5_gdn_g_beta_f32(
    const device VPIPE_ELT*  a       [[buffer(0)]],
    const device VPIPE_ELT*  b       [[buffer(1)]],
    const device float* A_log   [[buffer(2)]],
    const device float* dt_bias [[buffer(3)]],
    device float*       g       [[buffer(4)]],
    device float*       beta    [[buffer(5)]],
    constant int&       Hv      [[buffer(6)]],
    constant int&       n       [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  const uint total = (uint)n * (uint)Hv;
  if (gid >= total) { return; }
  const uint hv = gid % (uint)Hv;
  const float x = (float)a[gid] + dt_bias[hv];
  const float sp = fmax(x, 0.0f) + metal::log(1.0f + metal::exp(-fabs(x)));
  g[gid] = metal::exp(-metal::exp(A_log[hv]) * sp);
  beta[gid] = 1.0f / (1.0f + metal::exp(-(float)b[gid]));
}
