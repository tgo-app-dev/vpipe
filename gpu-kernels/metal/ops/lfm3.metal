// lfm3.metal -- LFM3 / LFM3-VL's own ops (generative-models/lfm3/): the
// language model's gated short convolution (its causal depthwise conv of
// width K over B*x, gated by C, with a K-1 row state per context), the
// vision tower's gated 3x3 depthwise convolution over a frame's patch
// grid, and the attention's per-head sigmoid output gate.
//
// Rounding follows the reference (modeling_lfm3.py, modeling_lfm3_vl.py),
// which runs the model in bf16: B*x is stored to the element type, the
// conv accumulates in f32 and is stored, C times it is stored -- so the
// element type is rounded where the reference rounds.

#include <metal_stdlib>
using namespace metal;

// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the
// bf16 twin (same entry-point names, different library). Math in f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// The widest state a context keeps (K - 1 rows): LFM3's K is 7.
#define LFM3_MAX_STATE 8

// B*x of row j of a fused [B | C | x] projection, stored as the reference
// stores it.
inline float
lfm3_bx_(const device VPIPE_ELT* bcx, int64_t stride, int j, int H, int c)
{
  const device VPIPE_ELT* r = bcx + (int64_t)j * stride;
  return float(VPIPE_ELT(float(r[c]) * float(r[2 * H + c])));
}

// The gated short convolution over n rows of one context:
//   bx(j) = B[j] * x[j] for j >= 0, state[K - 1 + j] for j < 0
//   y[t]  = C[t] * sum_k w[k] * bx(t - (K - 1) + k)
// bcx [n, 3H] (row stride in_stride), state [K - 1, H] (oldest row first:
// the B*x of the K - 1 positions before this forward), w [H, K]; y [n, H]
// (row stride out_stride). Reads the state, never writes it -- the
// update is lfm3_shortconv_state_f16, after.
//   0:bcx 1:state 2:w 3:y 4:H 5:n 6:K 7:in_stride 8:out_stride
// grid (H, n), threadgroup (<= 256, 1).
kernel void lfm3_shortconv_f16(
    const device VPIPE_ELT* bcx        [[buffer(0)]],
    const device VPIPE_ELT* state      [[buffer(1)]],
    const device VPIPE_ELT* w          [[buffer(2)]],
    device VPIPE_ELT*       y          [[buffer(3)]],
    constant int&           H          [[buffer(4)]],
    constant int&           n          [[buffer(5)]],
    constant int&           K          [[buffer(6)]],
    constant int&           in_stride  [[buffer(7)]],
    constant int&           out_stride [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int t = (int)gid.y;
  if (c >= H || t >= n) { return; }
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const int j = t - (K - 1) + k;
    const float v = j >= 0 ? lfm3_bx_(bcx, in_stride, j, H, c)
                           : float(state[(K - 1 + j) * H + c]);
    acc += float(w[c * K + k]) * v;
  }
  const float g = float(bcx[(int64_t)t * in_stride + H + c]);
  y[(int64_t)t * out_stride + c] = VPIPE_ELT(g * float(VPIPE_ELT(acc)));
}

// The state after n rows: the last K - 1 values of B*x of the sequence
// (the old state followed by this forward's rows). One thread a channel,
// the old state read before any of it is written.
//   0:bcx 1:state 2:H 3:n 4:K 5:in_stride.  grid (H), tg (<= 256).
kernel void lfm3_shortconv_state_f16(
    const device VPIPE_ELT* bcx       [[buffer(0)]],
    device VPIPE_ELT*       state     [[buffer(1)]],
    constant int&           H         [[buffer(2)]],
    constant int&           n         [[buffer(3)]],
    constant int&           K         [[buffer(4)]],
    constant int&           in_stride [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const int c = (int)gid;
  if (c >= H) { return; }
  const int S = K - 1;
  VPIPE_ELT old[LFM3_MAX_STATE];
  for (int i = 0; i < S; ++i) { old[i] = state[i * H + c]; }
  for (int i = 0; i < S; ++i) {
    const int s = n + i;        // index into old ++ rows
    state[i * H + c] = s < S
        ? old[s] : VPIPE_ELT(lfm3_bx_(bcx, in_stride, s - S, H, c));
  }
}

// One decode row: the conv and the state's shift in one dispatch, each
// channel's state read into registers before it is written. y, state and
// the arithmetic are lfm3_shortconv_f16 + lfm3_shortconv_state_f16's at
// n = 1, bit for bit.
//   0:bcx 1:state 2:w 3:y 4:H 5:K.  grid (H), tg (<= 256).
kernel void lfm3_shortconv_step_f16(
    const device VPIPE_ELT* bcx   [[buffer(0)]],
    device VPIPE_ELT*       state [[buffer(1)]],
    const device VPIPE_ELT* w     [[buffer(2)]],
    device VPIPE_ELT*       y     [[buffer(3)]],
    constant int&           H     [[buffer(4)]],
    constant int&           K     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const int c = (int)gid;
  if (c >= H) { return; }
  const int S = K - 1;
  float v[LFM3_MAX_STATE + 1];
  for (int i = 0; i < S; ++i) { v[i] = float(state[i * H + c]); }
  v[S] = lfm3_bx_(bcx, 0, 0, H, c);
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) { acc += float(w[c * K + k]) * v[k]; }
  for (int i = 0; i < S; ++i) { state[i * H + c] = VPIPE_ELT(v[i + 1]); }
  y[c] = VPIPE_ELT(float(bcx[H + c]) * float(VPIPE_ELT(acc)));
}

// The attention's output gate: a head-major attention output [Hq, n, D]
// times sigmoid of the head's gate logit, written token-major
//   out[t, h*D + d] = a[h, t, d] * sigmoid(g[t * g_stride + h])
// the sigmoid stored to the element type before the product, as the
// reference's bf16 sigmoid is. At n == 1 head-major is token-major.
//   0:a 1:g 2:out 3:Hq 4:n 5:D 6:g_stride 7:out_stride
// grid (D, n, Hq), threadgroup (D, 1, 1).
kernel void lfm3_gate_heads_f16(
    const device VPIPE_ELT* a          [[buffer(0)]],
    const device VPIPE_ELT* g          [[buffer(1)]],
    device VPIPE_ELT*       out        [[buffer(2)]],
    constant int&           Hq         [[buffer(3)]],
    constant int&           n          [[buffer(4)]],
    constant int&           D          [[buffer(5)]],
    constant int&           g_stride   [[buffer(6)]],
    constant int&           out_stride [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (d >= D || t >= n || h >= Hq) { return; }
  const float gl = float(g[(int64_t)t * g_stride + h]);
  const float s = float(VPIPE_ELT(1.0f / (1.0f + metal::exp(-gl))));
  const float av = float(a[((int64_t)h * n + t) * D + d]);
  out[(int64_t)t * out_stride + h * D + d] = VPIPE_ELT(av * s);
}

// The vision tower's gated depthwise 3x3 convolution over one frame's
// patch grid (gh x gw patches), its rows in MERGE order -- block-major
// over M x M blocks, row r = ((I * gw/M + J) * M + ki) * M + kj for patch
// (I*M + ki, J*M + kj):
//   out[r] = C[r] * sum_{dy,dx} w[c, dy, dx] * (B*x)[patch + (dy-1, dx-1)]
// zero past the grid's edge. bcx [n, 3C] (B | C | x, row stride
// in_stride), w [C, 3, 3]; out [n, C].
//   0:bcx 1:w 2:out 3:C 4:gh 5:gw 6:M 7:in_stride
// grid (C, gh * gw), threadgroup (<= 256, 1).
kernel void lfm3_vis_conv3x3_f16(
    const device VPIPE_ELT* bcx       [[buffer(0)]],
    const device VPIPE_ELT* w         [[buffer(1)]],
    device VPIPE_ELT*       out       [[buffer(2)]],
    constant int&           C         [[buffer(3)]],
    constant int&           gh        [[buffer(4)]],
    constant int&           gw        [[buffer(5)]],
    constant int&           M         [[buffer(6)]],
    constant int&           in_stride [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int r = (int)gid.y;
  if (c >= C || r >= gh * gw) { return; }
  const int MM = M * M, bw = gw / M;
  const int blk = r / MM, k = r % MM;
  const int pi = (blk / bw) * M + k / M;
  const int pj = (blk % bw) * M + k % M;
  float acc = 0.0f;
  for (int dy = 0; dy < 3; ++dy) {
    const int yi = pi + dy - 1;
    if (yi < 0 || yi >= gh) { continue; }
    for (int dx = 0; dx < 3; ++dx) {
      const int xj = pj + dx - 1;
      if (xj < 0 || xj >= gw) { continue; }
      const int nb = ((yi / M) * bw + xj / M) * MM + (yi % M) * M + xj % M;
      acc += float(w[(c * 3 + dy) * 3 + dx]) *
             lfm3_bx_(bcx, in_stride, nb, C, c);
    }
  }
  const float g = float(bcx[(int64_t)r * in_stride + C + c]);
  out[(int64_t)r * C + c] = VPIPE_ELT(g * float(VPIPE_ELT(acc)));
}

// A few rows through one weight: y[m, :] = W @ x[m, :] for m < M <= 8,
// each simdgroup reading its RPS output rows of W ONCE for every row --
// the batched decode's weights read once a step, not once a branch. A
// row's arithmetic does not depend on M (dense_gemv_t_impl's order: a
// lane's VPT products summed in order per 256-wide block, the blocks
// summed, then simd_sum), so a row of a batch IS that row alone.
//   0:x[M, K] 1:W[N, K] 2:y[M, N] 3:K 4:N 5:M
// grid {32, ceil(N / 8) * 2, 1}, threadgroup {32, 2, 1}.
#define LFM3_GB_RPS  4
#define LFM3_GB_VPT  8
#define LFM3_GB_MAXM 8
kernel void lfm3_gemv_batch_f16(
    const device VPIPE_ELT* x        [[buffer(0)]],
    const device VPIPE_ELT* W        [[buffer(1)]],
    device VPIPE_ELT*       y        [[buffer(2)]],
    constant int&           K        [[buffer(3)]],
    constant int&           N        [[buffer(4)]],
    constant int&           M        [[buffer(5)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int RPS = LFM3_GB_RPS, VPT = LFM3_GB_VPT;
  constexpr int block = VPT * 32;
  const int out_row = (int)tid.y * (2 * RPS) + (int)simd_gid * RPS;
  float res[LFM3_GB_MAXM][RPS];
  for (int m = 0; m < LFM3_GB_MAXM; ++m) {
    for (int r = 0; r < RPS; ++r) { res[m][r] = 0.0f; }
  }
  for (int k = 0; k < K; k += block) {
    const int kb = k + (int)simd_lid * VPT;
    const int v = min(VPT, K - kb);           // valid elements, this lane
    if (v <= 0) { continue; }
    float wv[RPS][VPT];
    for (int r = 0; r < RPS; ++r) {
      const bool live = out_row + r < N;
      const device VPIPE_ELT* wl = W + (int64_t)(out_row + r) * K + kb;
      for (int i = 0; i < VPT; ++i) {
        wv[r][i] = (live && i < v) ? float(wl[i]) : 0.0f;
      }
    }
    for (int m = 0; m < M && m < LFM3_GB_MAXM; ++m) {
      const device VPIPE_ELT* xm = x + (int64_t)m * K + kb;
      float xt[VPT];
      for (int i = 0; i < VPT; ++i) {
        xt[i] = i < v ? float(xm[i]) : 0.0f;
      }
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < VPT; ++i) { acc += xt[i] * wv[r][i]; }
        res[m][r] += acc;
      }
    }
  }
  for (int m = 0; m < M && m < LFM3_GB_MAXM; ++m) {
    for (int r = 0; r < RPS; ++r) {
      const float s = simd_sum(res[m][r]);
      if (simd_lid == 0 && out_row + r < N) {
        y[(int64_t)m * N + out_row + r] = VPIPE_ELT(s);
      }
    }
  }
}
