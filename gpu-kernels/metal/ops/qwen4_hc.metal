// qwen4_hc.metal -- Qwen3.8-Flash-Next (model_type qwen4_exp) kernels:
// the hyper-connections that wrap every sublayer, the per-layer n-gram
// embedding (PLE) block, and the MoE output written rather than added.
//
// HYPER-CONNECTIONS. The residual is S = hc_count streams of H, laid out
// [rows, S*H]. Around each sublayer (Qwen4ExpTextGatedResidual):
//   n      = grouped RMS(X) * (1 + w)            (each H-wide stream alone)
//   d      = down(n)        [R]   -> silu(d / S)
//   mix    = sigmoid(up(silu(d / S)))           [S*H]
//   x      = mean_s(mix[s] * n[s])               [H]  -> the sublayer
//   inj    = 2 sigmoid(inject(n) / S)            [S]
//   X[s]  += inj[s] * sublayer(x)
// down and inject read the same n, so the loader fuses them into one
// [R + S, S*H] matrix; hc_act_ splits its output. The final mixer is the
// same block without inject.
//
// The zero-centred norm weights arrive as f32 (1 + w), as the reference
// forms them: rounding 1 + w to bf16 would cost up to half a bf16 ulp of
// the scale on every element.

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

namespace {

// Sum over the threadgroup (<= 1024 threads, 32-wide simdgroups).
inline float tg_sum_(float v, threadgroup float* red, uint tid, uint sid,
                     uint lane, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) { red[sid] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? red[lane] : 0.0f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

inline float sigmoid_(float v) { return 1.0f / (1.0f + metal::exp(-v)); }

// Exclusive prefix sum of v over the threadgroup's threads (whole
// simdgroups), and the total; sgs holds a value a simdgroup.
inline uint tg_excl_sum_(uint v, threadgroup uint* sgs, uint sid, uint lane,
                         uint nsg, thread uint& total) {
  const uint ex = simd_prefix_exclusive_sum(v);
  if (lane == 31) { sgs[sid] = ex + v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0, tot = 0;
  for (uint g = 0; g < nsg; ++g) {
    const uint t = sgs[g];
    base += g < sid ? t : 0u;
    tot += t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = tot;
  return base + ex;
}

}  // namespace

// y[m, s*H + h] = x[m, s*H + h] * rsqrt(mean_h x^2 + eps) * w[s*H + h].
// One threadgroup per (stream s, row m): grid {256, S, M}, tg {256, 1, 1}.
// x and y may alias.
//
// A decode runs it 100 times a token on four threadgroups, so its latency
// is the cost: a row of up to kRmsRegs values a thread is read ONCE, all
// its loads issued before the first is summed, and kept in registers for
// the second pass -- the same values summed in the same order (a thread's
// h = tid, tid + ntg, ...; then the simdgroups'), so the same bits as the
// two-pass loop it replaced (10.2 -> 4.6 us at 4 x 2560 on an M5 Ultra).
// A longer row takes that loop.
constant constexpr int kRmsRegs = 16;

kernel void hc_group_rms_f16(
    const device VPIPE_ELT* x  [[buffer(0)]],
    const device float*     w  [[buffer(1)]],   // [S*H] = 1 + weight
    device VPIPE_ELT*       y  [[buffer(2)]],
    constant int&           H  [[buffer(3)]],
    constant int&           S  [[buffer(4)]],
    constant float&         eps [[buffer(5)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 tp   [[thread_position_in_threadgroup]],
    uint3 tsz  [[threads_per_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const uint tid = tp.x, ntg = tsz.x;
  threadgroup float red[32];
  const ulong base = ((ulong)tg.z * (ulong)S + tg.y) * (ulong)H;
  const device float* ws = w + (ulong)tg.y * (ulong)H;
  if (H <= kRmsRegs * (int)ntg) {
    float v[kRmsRegs];
    for (int j = 0; j < kRmsRegs; ++j) {
      const int h = (int)tid + j * (int)ntg;
      v[j] = h < H ? (float)x[base + h] : 0.0f;
    }
    float ss = 0.0f;
    for (int j = 0; j < kRmsRegs; ++j) {
      if ((int)tid + j * (int)ntg < H) { ss += v[j] * v[j]; }
    }
    const float tot = tg_sum_(ss, red, tid, sid, lane, (ntg + 31) / 32);
    const float r = metal::rsqrt(tot / (float)H + eps);
    for (int j = 0; j < kRmsRegs; ++j) {
      const int h = (int)tid + j * (int)ntg;
      if (h < H) { y[base + h] = (VPIPE_ELT)(v[j] * r * ws[h]); }
    }
    return;
  }
  float ss = 0.0f;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    const float v = (float)x[base + h];
    ss += v * v;
  }
  const float tot = tg_sum_(ss, red, tid, sid, lane, (ntg + 31) / 32);
  const float r = metal::rsqrt(tot / (float)H + eps);
  for (int h = (int)tid; h < H; h += (int)ntg) {
    y[base + h] = (VPIPE_ELT)((float)x[base + h] * r * ws[h]);
  }
}

// hc_group_rms_f16 for a PREFILL's rows: a thread 8 values a load (the
// kernel above reads one element a lane-step, ~340 GB/s at 2048 x 4 x
// 2560 on an M5 Ultra), the row held in registers for the second pass.
// Its sums run in another order, so a decode step and a verify keep the
// kernel above. H % 8 == 0, H <= 64 * 8 * kRms8. grid {64, S, M}, tg
// {64, 1, 1}.
constant constexpr int kRms8 = 8;   // 8-value vectors a thread

kernel void hc_group_rms8_f16(
    const device VPIPE_ELT* x  [[buffer(0)]],
    const device float*     w  [[buffer(1)]],   // [S*H] = 1 + weight
    device VPIPE_ELT*       y  [[buffer(2)]],
    constant int&           H  [[buffer(3)]],
    constant int&           S  [[buffer(4)]],
    constant float&         eps [[buffer(5)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 tp   [[thread_position_in_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int NT = 64;
  threadgroup float red[32];
  const uint tid = tp.x;
  const ulong base = ((ulong)tg.z * (ulong)S + tg.y) * (ulong)H;
  const device float* ws = w + (ulong)tg.y * (ulong)H;
  const int nv = H / 8;
  vec<VPIPE_ELT, 8> v[kRms8];
  float ss = 0.0f;
  for (int j = 0; j < kRms8; ++j) {
    const int i = (int)tid + j * NT;
    if (i < nv) {
      v[j] = *(const device vec<VPIPE_ELT, 8>*)(x + base + (ulong)i * 8);
      for (int c = 0; c < 8; ++c) {
        const float f = (float)v[j][c];
        ss += f * f;
      }
    }
  }
  const float tot = tg_sum_(ss, red, tid, sid, lane, NT / 32);
  const float r = metal::rsqrt(tot / (float)H + eps);
  for (int j = 0; j < kRms8; ++j) {
    const int i = (int)tid + j * NT;
    if (i < nv) {
      vec<VPIPE_ELT, 8> o;
      for (int c = 0; c < 8; ++c) {
        o[c] = (VPIPE_ELT)((float)v[j][c] * r * ws[i * 8 + c]);
      }
      *(device vec<VPIPE_ELT, 8>*)(y + base + (ulong)i * 8) = o;
    }
  }
}

// The down output d [M, R] <- silu(d / S) in place (the up projection's
// input), and the injection weights inj[m, s] = 2 sigmoid(ir[m, s] / S) in
// f32 from the inject output ir [M, S] (empty on the final mixer: S rows of
// nothing to do). grid {M * (R + S)}.
kernel void hc_act_f16(
    device VPIPE_ELT*       d    [[buffer(0)]],   // [M, R]
    const device VPIPE_ELT* ir   [[buffer(1)]],   // [M, S]
    device float*           inj  [[buffer(2)]],   // [M, S]
    constant int&           R    [[buffer(3)]],
    constant int&           S    [[buffer(4)]],
    constant int&           M    [[buffer(5)]],
    constant int&           with_inject [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  const int W = R + S;
  if (gid >= (uint)(M * W)) { return; }
  const int m = (int)gid / W, j = (int)gid % W;
  if (j < R) {
    // The reference rounds the down output, then divides and applies silu
    // in the element type; the division by a power of two is exact.
    const ulong i = (ulong)m * R + j;
    const float v = (float)d[i] / (float)S;
    d[i] = (VPIPE_ELT)(v * sigmoid_(v));
  } else if (with_inject != 0) {
    const int s = j - R;
    const float v = (float)ir[m * S + s] / (float)S;
    inj[m * S + s] = (float)(VPIPE_ELT)(2.0f * sigmoid_(v));
  }
}

// hc_act_f16 for a block that injects, from its one down | inject
// projection di [M, R + S]: silu(di[:, :R] / S) into d [M, R], 2
// sigmoid(di[:, R:] / S) into inj [M, S] -- the same arithmetic. grid
// {M * (R + S)}.
kernel void hc_act_fused_f16(
    const device VPIPE_ELT* di   [[buffer(0)]],   // [M, R + S]
    device VPIPE_ELT*       d    [[buffer(1)]],   // [M, R]
    device float*           inj  [[buffer(2)]],   // [M, S]
    constant int&           R    [[buffer(3)]],
    constant int&           S    [[buffer(4)]],
    constant int&           M    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const int W = R + S;
  if (gid >= (uint)(M * W)) { return; }
  const int m = (int)gid / W, j = (int)gid % W;
  const float v = (float)di[(ulong)m * W + j] / (float)S;
  if (j < R) {
    d[(ulong)m * R + j] = (VPIPE_ELT)(v * sigmoid_(v));
  } else {
    inj[m * S + (j - R)] = (float)(VPIPE_ELT)(2.0f * sigmoid_(v));
  }
}

// out[m, h] = (1/S) sum_s sigmoid(u[m, s*H + h]) * n[m, s*H + h]: the
// mixed sublayer input. grid {H, M}.
//
// A thread 8 values of h (the same sums for each: a prefill's rows moved
// at ~360 GB/s an element a thread). H % 8 == 0. grid {H / 8, M}.
kernel void hc_mix_f16(
    const device VPIPE_ELT* u   [[buffer(0)]],   // up output [M, S*H]
    const device VPIPE_ELT* n   [[buffer(1)]],   // normed [M, S*H]
    device VPIPE_ELT*       out [[buffer(2)]],   // [M, H]
    constant int&           H   [[buffer(3)]],
    constant int&           S   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H / 8) { return; }
  const ulong row = (ulong)gid.y * (ulong)S * (ulong)H;
  const ulong h0 = (ulong)gid.x * 8;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < S; ++s) {
    const ulong i = row + (ulong)s * (ulong)H + h0;
    const vec<VPIPE_ELT, 8> uv = *(const device vec<VPIPE_ELT, 8>*)(u + i);
    const vec<VPIPE_ELT, 8> nv = *(const device vec<VPIPE_ELT, 8>*)(n + i);
    for (int c = 0; c < 8; ++c) {
      // sigmoid(up) is a bf16 / f16 tensor in the reference, as is its
      // product with the normed stream.
      const float mixw = (float)(VPIPE_ELT)sigmoid_((float)uv[c]);
      acc[c] += (float)(VPIPE_ELT)(mixw * (float)nv[c]);
    }
  }
  vec<VPIPE_ELT, 8> o;
  for (int c = 0; c < 8; ++c) { o[c] = (VPIPE_ELT)(acc[c] / (float)S); }
  *(device vec<VPIPE_ELT, 8>*)(out + (ulong)gid.y * (ulong)H + h0) = o;
}

// X[m, s*H + h] += inj[m, s] * y[m, h]: the sublayer's output into every
// stream at its injection weight -- a thread 8 values of h, each the same
// arithmetic. H % 8 == 0. grid {H / 8, S, M}.
kernel void hc_inject_f16(
    device VPIPE_ELT*       X   [[buffer(0)]],   // [M, S*H]
    const device VPIPE_ELT* y   [[buffer(1)]],   // [M, H]
    const device float*     inj [[buffer(2)]],   // [M, S]
    constant int&           H   [[buffer(3)]],
    constant int&           S   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H / 8) { return; }
  const ulong h0 = (ulong)gid.x * 8;
  const ulong i = ((ulong)gid.z * (ulong)S + gid.y) * (ulong)H + h0;
  const float wj = inj[gid.z * (uint)S + gid.y];
  const vec<VPIPE_ELT, 8> yv =
      *(const device vec<VPIPE_ELT, 8>*)(y + (ulong)gid.z * H + h0);
  vec<VPIPE_ELT, 8> xv = *(device vec<VPIPE_ELT, 8>*)(X + i);
  for (int c = 0; c < 8; ++c) {
    const float add = (float)(VPIPE_ELT)(wj * (float)yv[c]);
    xv[c] = (VPIPE_ELT)((float)xv[c] + add);
  }
  *(device vec<VPIPE_ELT, 8>*)(X + i) = xv;
}

// ---------------------------------------------------------------------
// A DECODE step's and a VERIFY's hyper-connection block in TWO launches.
// It was six -- the norm, down | inject, the activations, up, the mix and
// the sublayer's injection -- each a dependent dispatch on a few
// threadgroups: ~33 us a block, 97 blocks a token (M5 Ultra). The
// arithmetic is the six kernels', value for value: the stream norm as
// hc_group_rms_f16 sums it, the down rows as affine_qmv_splitk8_w8g64
// splits K and adds its runs, the up rows as affine_qmv_w8g64 and its
// batched twin lay their lanes, the activations, the mix and the
// injection as hc_act_fused_f16, hc_mix_f16 and hc_inject_f16 round them.
// A decode keeps its tokens, and a verify's rows are a decode's.
//
// The sublayer's injection is PENDING: X[s] += inj[s] * y is added to
// the stream as the next block's norm reads it (hc_pre_down_f16) and
// written back by that block's second launch (hc_up_mix_f16), in which
// nothing reads X. S == 4 and H % 512 == 0, so the down's 8 runs of K
// are two a stream; R <= kHcfMaxR, R % 64 == 0; H == 512 * kHcfBR.
constant constexpr int kHcfBR = 5;     // H = 512 kHcfBR (2560)
constant constexpr int kHcfRuns = 8;
constant constexpr int kHcfMaxR = 512;

// hc_pre_down_f16: threadgroup (g, s, z) -- down rows g * 4 * RPS ..,
// stream s, rows z * MAXM .. -- norms stream s (every threadgroup of the
// stream alike) into threadgroup memory, then simdgroup j runs K run
// 2 s + (j & 1) of rows g * 4 * RPS + (j >> 1) * RPS ..: each run's sum
// into parts[row][run][N]. Its weights are read FIRST, into registers --
// they wait on nothing, and the norm hides their latency (11.4 -> ? us a
// block on an M5 Ultra). Threadgroups g < 2 BR write the normed stream's
// strip g (a thread's g-th value) to hcn, for the mix. H = 512 BR.
// grid {256 G, 4, ceil(M / MAXM)}, tg {256, 1, 1}, G = ceil(N / (4 RPS)).
template <int RPS, int MAXM, int BR, int GS = 64>
inline void hcf_pre_down_(
    const device VPIPE_ELT* X, const device VPIPE_ELT* py,
    const device float* pinj, const device float* nw,
    device VPIPE_ELT* hcn, const device uint8_t* dw,
    const device VPIPE_ELT* ds, const device VPIPE_ELT* db,
    device float* parts, int N, int M, float eps, int pending,
    threadgroup VPIPE_ELT* xs, threadgroup float* red,
    uint3 tg, uint3 ntg, uint tid, uint sid, uint lane)
{
  constexpr int S = 4;
  constexpr int H = 512 * BR;
  constexpr int NV = 2 * BR;
  constexpr int K = S * H;
  constexpr int Kg = K / GS;     // a lane's 8 values lie in one group
  const int g = (int)tg.x, s = (int)tg.y;
  const int row0 = (int)tg.z * MAXM;
  const int m_rows = min(MAXM, M - row0);

  const int rh = (int)sid & 1, rs = (int)sid >> 1;
  const int run = s * 2 + rh;
  const int o0 = g * 4 * RPS + rs * RPS;
  const int rows_here = clamp(N - o0, 0, RPS);
  const int kb0 = s * H + rh * (H / 2);
  uint2 wq[RPS][BR];
  float wsc[RPS][BR], wbi[RPS][BR];
  for (int r = 0; r < RPS; ++r) {
    const int o = min(o0 + r, N - 1);
    const device uint8_t* wp = dw + (ulong)o * K + kb0 + lane * 8;
    const ulong gi = (ulong)o * Kg + kb0 / GS + lane / (GS / 8);
    for (int b = 0; b < BR; ++b) {
      wq[r][b] = *(const device uint2*)(wp + b * 256);
      wsc[r][b] = ds[gi + b * (256 / GS)];
      wbi[r][b] = db[gi + b * (256 / GS)];
    }
  }

  const device float* ws = nw + (ulong)s * (ulong)H;
  float v[MAXM][NV];
  for (int m = 0; m < MAXM; ++m) {
    const int r = row0 + min(m, m_rows - 1);
    const ulong base = ((ulong)r * S + s) * (ulong)H;
    const float wj = pending != 0 ? pinj[r * S + s] : 0.0f;
    for (int j = 0; j < NV; ++j) {
      const int h = (int)tid + j * 256;
      VPIPE_ELT xv = X[base + h];
      if (pending != 0) {
        const float add =
            (float)(VPIPE_ELT)(wj * (float)py[(ulong)r * H + h]);
        xv = (VPIPE_ELT)((float)xv + add);
      }
      v[m][j] = (float)xv;
    }
    float ss = 0.0f;
    for (int j = 0; j < NV; ++j) { ss += v[m][j] * v[m][j]; }
    ss = simd_sum(ss);
    if (lane == 0) { red[m * 8 + sid] = ss; }
  }
  float wv[NV];
  for (int j = 0; j < NV; ++j) { wv[j] = ws[(int)tid + j * 256]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int m = 0; m < MAXM; ++m) {
    float t = lane < 8 ? red[m * 8 + lane] : 0.0f;
    t = simd_sum(t);
    const float rr = metal::rsqrt(t / (float)H + eps);
    const int r = row0 + min(m, m_rows - 1);
    const ulong base = ((ulong)r * S + s) * (ulong)H;
    for (int j = 0; j < NV; ++j) {
      const int h = (int)tid + j * 256;
      const VPIPE_ELT y = (VPIPE_ELT)(v[m][j] * rr * wv[j]);
      xs[m * H + h] = y;
      if (m < m_rows && j % (int)ntg.x == g) { hcn[base + h] = y; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const threadgroup VPIPE_ELT* xp = xs + rh * (H / 2) + lane * 8;
  float res[RPS][MAXM];
  for (int r = 0; r < RPS; ++r) {
    for (int m = 0; m < MAXM; ++m) { res[r][m] = 0.0f; }
  }
  for (int b = 0; b < BR; ++b) {
    float xt[MAXM][8];
    float sum[MAXM];
    for (int m = 0; m < MAXM; ++m) {
      sum[m] = 0.0f;
      for (int i = 0; i < 8; ++i) {
        const VPIPE_ELT xv = xp[m * H + b * 256 + i];
        sum[m] += xv;
        xt[m][i] = xv;
      }
    }
    for (int r = 0; r < rows_here; ++r) {
      const uint2 q = wq[r][b];
      float acc[MAXM];
      for (int m = 0; m < MAXM; ++m) { acc[m] = 0.0f; }
      for (int i = 0; i < 8; ++i) {
        const uint word = i < 4 ? q.x : q.y;
        const float wv8 = (float)((word >> (8 * (i & 3))) & 0xffu);
        for (int m = 0; m < MAXM; ++m) { acc[m] += xt[m][i] * wv8; }
      }
      for (int m = 0; m < MAXM; ++m) {
        res[r][m] += wsc[r][b] * acc[m] + sum[m] * wbi[r][b];
      }
    }
  }
  for (int r = 0; r < RPS; ++r) {
    for (int m = 0; m < MAXM; ++m) {
      const float t = simd_sum(res[r][m]);
      if (lane == 0 && r < rows_here && m < m_rows) {
        parts[((ulong)(row0 + m) * kHcfRuns + run) * N + o0 + r] = t;
      }
    }
  }
}

#define VPIPE_HCF_PRE(NAME, RPS, MAXM, GS)                                   \
  kernel void NAME(                                                          \
      const device VPIPE_ELT* X     [[buffer(0)]],                           \
      const device VPIPE_ELT* py    [[buffer(1)]],                           \
      const device float*     pinj  [[buffer(2)]],                           \
      const device float*     nw    [[buffer(3)]],                           \
      device VPIPE_ELT*       hcn   [[buffer(4)]],                           \
      const device uint8_t*   dw    [[buffer(5)]],                           \
      const device VPIPE_ELT* ds    [[buffer(6)]],                           \
      const device VPIPE_ELT* db    [[buffer(7)]],                           \
      device float*           parts [[buffer(8)]],                           \
      constant int&           N     [[buffer(10)]],                          \
      constant int&           M     [[buffer(11)]],                          \
      constant float&         eps   [[buffer(12)]],                          \
      constant int&           pend  [[buffer(13)]],                          \
      uint3 tg   [[threadgroup_position_in_grid]],                           \
      uint3 ntg  [[threadgroups_per_grid]],                                  \
      uint  tid  [[thread_index_in_threadgroup]],                            \
      uint  sid  [[simdgroup_index_in_threadgroup]],                         \
      uint  lane [[thread_index_in_simdgroup]]) {                            \
    threadgroup VPIPE_ELT xs[MAXM * 512 * kHcfBR];                           \
    threadgroup float red[MAXM * 8];                                         \
    hcf_pre_down_<RPS, MAXM, kHcfBR, GS>(X, py, pinj, nw, hcn, dw, ds, db,   \
                                     parts, N, M, eps, pend, xs, red, tg,    \
                                     ntg, tid, sid, lane);                   \
  }
// A row a threadgroup (grid z = the row): four rows one took 37 us a
// launch at a verify's four against 14 (M5 Ultra) -- every threadgroup
// norms its rows' streams, 4x the work, in 4x the threadgroup memory.
VPIPE_HCF_PRE(hc_pre_down_r1_f16, 1, 1, 64)
VPIPE_HCF_PRE(hc_pre_down_r2_f16, 2, 1, 64)
VPIPE_HCF_PRE(hc_pre_down_r4_f16, 4, 1, 64)
// The projections at group 32 (re-encoded to w8g32).
VPIPE_HCF_PRE(hc_pre_down_r1_g32_f16, 1, 1, 32)
VPIPE_HCF_PRE(hc_pre_down_r2_g32_f16, 2, 1, 32)
VPIPE_HCF_PRE(hc_pre_down_r4_g32_f16, 4, 1, 32)

// hc_up_mix_f16: threadgroup (x, z) -- h x * 8 * HPS .., rows z * MAXM ..
// -- adds the down's runs in order into the activations (threadgroup x ==
// 0 the injection weights too, into inj), writes the pending injection
// into X for its h, then simdgroup j runs the up rows s * H + h of its
// HPS h and the four streams, and a lane mixes each h. The up weights and
// the normed values the mix reads are fetched FIRST. R <= 512. grid {256
// H / (8 HPS), 1, ceil(M / MAXM)}, tg {256, 1, 1}.
template <int HPS, int MAXM, int GS = 64>
inline void hcf_up_mix_(
    const device float* parts, const device uint8_t* uw,
    const device VPIPE_ELT* us, const device VPIPE_ELT* ub,
    const device VPIPE_ELT* hcn, device VPIPE_ELT* xmix, device float* inj,
    device VPIPE_ELT* X, const device VPIPE_ELT* py,
    const device float* pinj, int H, int R, int N, int M, int pending,
    threadgroup VPIPE_ELT* acts, uint3 tg, uint tid, uint sid, uint lane)
{
  constexpr int S = 4;
  constexpr int HT = 8 * HPS;
  constexpr int Q = HPS * S;
  const int row0 = (int)tg.z * MAXM;
  const int m_rows = min(MAXM, M - row0);
  const int h0 = (int)tg.x * HT;
  const int Rg = R / GS;      // a lane's 8 values lie in one group

  // The up weights: two blocks of 256 at most (R <= 512), a lane's part.
  bool live[2];
  uint2 uq[Q][2];
  float usc[Q][2], ubi[Q][2];
  for (int c = 0; c < 2; ++c) {
    const int kb = c * 256;
    live[c] = kb < R && (R - kb >= 256 || R - (kb + (int)lane * 8) > 0);
  }
  for (int q = 0; q < Q; ++q) {
    const int nr = (q % S) * H + h0 + (int)sid * HPS + q / S;
    for (int c = 0; c < 2; ++c) {
      if (live[c]) {
        const int kb = c * 256;
        uq[q][c] = *(const device uint2*)(uw + (ulong)nr * R + kb + lane * 8);
        usc[q][c] = us[(ulong)nr * Rg + kb / GS + lane / (GS / 8)];
        ubi[q][c] = ub[(ulong)nr * Rg + kb / GS + lane / (GS / 8)];
      } else {
        uq[q][c] = uint2(0u);
        usc[q][c] = 0.0f;
        ubi[q][c] = 0.0f;
      }
    }
  }
  // The normed values this lane mixes (lane k * MAXM + m: h's k, row m).
  float nmix[S];
  const int mk = (int)lane / MAXM, mm = (int)lane % MAXM;
  const bool mixer = (int)lane < HPS * MAXM && mm < m_rows;
  for (int s = 0; s < S; ++s) {
    nmix[s] = mixer ? (float)hcn[((ulong)(row0 + mm) * S + s) * (ulong)H +
                                 h0 + (int)sid * HPS + mk]
                    : 0.0f;
  }

  const int lim = tg.x == 0 ? N : R;
  for (int m = 0; m < MAXM; ++m) {
    const int r = row0 + min(m, m_rows - 1);
    const device float* pp = parts + (ulong)r * kHcfRuns * N;
    for (int i = (int)tid; i < lim; i += 256) {
      float t = 0.0f;
      for (int k = 0; k < kHcfRuns; ++k) { t += pp[k * N + i]; }
      const float vv = (float)(VPIPE_ELT)t / (float)S;
      if (i < R) {
        acts[m * kHcfMaxR + i] = (VPIPE_ELT)(vv * sigmoid_(vv));
      } else if (m < m_rows) {
        inj[r * S + (i - R)] = (float)(VPIPE_ELT)(2.0f * sigmoid_(vv));
      }
    }
  }
  if (pending != 0) {
    for (int e = (int)tid; e < m_rows * S * HT; e += 256) {
      const int m = e / (S * HT), s = (e / HT) % S, h = h0 + e % HT;
      const int r = row0 + m;
      const ulong i = ((ulong)r * S + s) * (ulong)H + h;
      const float add =
          (float)(VPIPE_ELT)(pinj[r * S + s] * (float)py[(ulong)r * H + h]);
      X[i] = (VPIPE_ELT)((float)X[i] + add);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float res[Q][MAXM];
  for (int q = 0; q < Q; ++q) {
    for (int m = 0; m < MAXM; ++m) { res[q][m] = 0.0f; }
  }
  for (int c = 0; c < 2; ++c) {
    if (!live[c]) { continue; }
    const int kb = c * 256;
    const int krem = R - kb;
    const int kvalid = R - (kb + (int)lane * 8);
    float xt[MAXM][8];
    float sum[MAXM];
    for (int m = 0; m < MAXM; ++m) {
      sum[m] = 0.0f;
      for (int i = 0; i < 8; ++i) {
        const float xv = (krem >= 256 || i < kvalid)
            ? (float)acts[m * kHcfMaxR + kb + (int)lane * 8 + i] : 0.0f;
        sum[m] += xv;
        xt[m][i] = xv;
      }
    }
    for (int q = 0; q < Q; ++q) {
      const uint2 w2 = uq[q][c];
      float acc[MAXM];
      for (int m = 0; m < MAXM; ++m) { acc[m] = 0.0f; }
      for (int i = 0; i < 8; ++i) {
        const uint word = i < 4 ? w2.x : w2.y;
        const float wv = (float)((word >> (8 * (i & 3))) & 0xffu);
        for (int m = 0; m < MAXM; ++m) { acc[m] += xt[m][i] * wv; }
      }
      for (int m = 0; m < MAXM; ++m) {
        res[q][m] += usc[q][c] * acc[m] + sum[m] * ubi[q][c];
      }
    }
  }
  VPIPE_ELT u[Q][MAXM];
  for (int q = 0; q < Q; ++q) {
    for (int m = 0; m < MAXM; ++m) {
      u[q][m] = (VPIPE_ELT)simd_sum(res[q][m]);
    }
  }
  for (int k = 0; k < HPS; ++k) {
    for (int m = 0; m < MAXM; ++m) {
      if ((int)lane != k * MAXM + m || m >= m_rows) { continue; }
      const int r = row0 + m, h = h0 + (int)sid * HPS + k;
      float acc = 0.0f;
      for (int s = 0; s < S; ++s) {
        const float mixw = (float)(VPIPE_ELT)sigmoid_((float)u[k * S + s][m]);
        acc += (float)(VPIPE_ELT)(mixw * nmix[s]);
      }
      xmix[(ulong)r * H + h] = (VPIPE_ELT)(acc / (float)S);
    }
  }
}

#define VPIPE_HCF_UP(NAME, HPS, MAXM, GS)                                    \
  kernel void NAME(                                                          \
      const device float*     parts [[buffer(0)]],                           \
      const device uint8_t*   uw    [[buffer(1)]],                           \
      const device VPIPE_ELT* us    [[buffer(2)]],                           \
      const device VPIPE_ELT* ub    [[buffer(3)]],                           \
      const device VPIPE_ELT* hcn   [[buffer(4)]],                           \
      device VPIPE_ELT*       xmix  [[buffer(5)]],                           \
      device float*           inj   [[buffer(6)]],                           \
      device VPIPE_ELT*       X     [[buffer(7)]],                           \
      const device VPIPE_ELT* py    [[buffer(8)]],                           \
      const device float*     pinj  [[buffer(9)]],                           \
      constant int&           H     [[buffer(10)]],                          \
      constant int&           R     [[buffer(11)]],                          \
      constant int&           N     [[buffer(12)]],                          \
      constant int&           M     [[buffer(13)]],                          \
      constant int&           pend  [[buffer(14)]],                          \
      uint3 tg   [[threadgroup_position_in_grid]],                           \
      uint  tid  [[thread_index_in_threadgroup]],                            \
      uint  sid  [[simdgroup_index_in_threadgroup]],                         \
      uint  lane [[thread_index_in_simdgroup]]) {                            \
    threadgroup VPIPE_ELT acts[MAXM * kHcfMaxR];                             \
    hcf_up_mix_<HPS, MAXM, GS>(parts, uw, us, ub, hcn, xmix, inj, X, py,     \
                               pinj, H, R, N, M, pend, acts, tg, tid, sid,   \
                               lane);                                        \
  }
// A row a threadgroup, as the norm's launch.
VPIPE_HCF_UP(hc_up_mix_h1_f16, 1, 1, 64)
VPIPE_HCF_UP(hc_up_mix_h2_f16, 2, 1, 64)
VPIPE_HCF_UP(hc_up_mix_h4_f16, 4, 1, 64)
VPIPE_HCF_UP(hc_up_mix_h1_g32_f16, 1, 1, 32)
VPIPE_HCF_UP(hc_up_mix_h2_g32_f16, 2, 1, 32)
VPIPE_HCF_UP(hc_up_mix_h4_g32_f16, 4, 1, 32)

// hc_attn_prework_f16: a full-attention layer's projection rows into what
// its attention reads, in ONE launch for a decode step's and a verify's
// rows (it was thirteen: four slices, the q and k norms, three transposes,
// two ropes, the K and V writes). Threadgroup (j, t): j < Hq q head j, Hq
// + i k head i, Hq + Hkv + i v head i of row t; thread d its element. q
// and k are normed as hc_group_rms_f16 norms them and turned as
// rope_partial_f16 turns them (q into qt [Hq, n, D]); k and v go into the
// paged pools at kv.rb[t] + head * page_tokens * D. D == 256. grid {256,
// Hq + 2 Hkv, n}, tg {256, 1, 1}.
struct HcKvRows { uint rb[8]; };

kernel void hc_attn_prework_f16(
    const device VPIPE_ELT* qf       [[buffer(0)]],   // [n, Nf]
    const device float*     qnw      [[buffer(1)]],   // [D] 1 + w
    const device float*     knw      [[buffer(2)]],
    const device float*     inv_freq [[buffer(3)]],
    device VPIPE_ELT*       qt       [[buffer(4)]],   // [Hq, n, D]
    device VPIPE_ELT*       kp       [[buffer(5)]],   // the layer's pools
    device VPIPE_ELT*       vp       [[buffer(6)]],
    constant int&           n        [[buffer(7)]],
    constant int&           Nf       [[buffer(8)]],
    constant int&           Hq       [[buffer(9)]],
    constant int&           Hkv      [[buffer(10)]],
    constant int&           rotary_dim [[buffer(11)]],
    constant int&           offset   [[buffer(12)]],
    constant int&           page_tokens [[buffer(13)]],
    constant float&         eps      [[buffer(14)]],
    constant HcKvRows&      kv       [[buffer(15)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int D = 256;
  threadgroup float red[32];
  threadgroup VPIPE_ELT ys[D];
  const int j = (int)tg.y, t = (int)tg.z, d = (int)tid;
  const int qdo = Hq * 2 * D, kd = Hkv * D;
  const device VPIPE_ELT* row = qf + (ulong)t * (ulong)Nf;
  if (j >= Hq + Hkv) {
    const int i = j - Hq - Hkv;
    vp[kv.rb[t] + (uint)i * (uint)page_tokens * D + d] =
        row[qdo + kd + i * D + d];
    return;
  }
  const bool isq = j < Hq;
  const float v = (float)(isq ? row[j * 2 * D + d]
                              : row[qdo + (j - Hq) * D + d]);
  const float wv = (isq ? qnw : knw)[d];
  const float tot = tg_sum_(v * v, red, tid, sid, lane, 8);
  const float r = metal::rsqrt(tot / (float)D + eps);
  const VPIPE_ELT y = (VPIPE_ELT)(v * r * wv);
  ys[d] = y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int half_r = rotary_dim / 2;
  VPIPE_ELT o = y;
  if (d < rotary_dim) {
    const int i = d < half_r ? d : d - half_r;
    const float angle = float(offset + t) * inv_freq[i];
    const float c = cos(angle);
    const float s = sin(angle);
    const float x1 = float(ys[i]);
    const float x2 = float(ys[i + half_r]);
    o = d < half_r ? VPIPE_ELT(x1 * c - x2 * s) : VPIPE_ELT(x1 * s + x2 * c);
  }
  if (isq) {
    qt[((uint)j * (uint)n + (uint)t) * D + d] = o;
  } else {
    kp[kv.rb[t] + (uint)(j - Hq) * (uint)page_tokens * D + d] = o;
  }
}

// hc_attn_post_f16: the attention's rows [Hq, n, D] -- a part's dense
// rows [Hq, r_s, D] apart -- back to [n, Hq D], gated by sigmoid(gate) as
// mul_sigmoid_f16 gates them: one launch where it was two (three with a
// dense part). grid {n Hq D}.
kernel void hc_attn_post_f16(
    const device VPIPE_ELT* at   [[buffer(0)]],   // [Hq, n, D]
    const device VPIPE_ELT* atd  [[buffer(1)]],   // [Hq, r_s, D]
    const device VPIPE_ELT* qf   [[buffer(2)]],   // the gates: j 2D + D
    device VPIPE_ELT*       att  [[buffer(3)]],   // [n, Hq D]
    constant int&           n    [[buffer(4)]],
    constant int&           Hq   [[buffer(5)]],
    constant int&           D    [[buffer(6)]],
    constant int&           Nf   [[buffer(7)]],
    constant int&           r_s  [[buffer(8)]],   // dense rows apart (0)
    uint gid [[thread_position_in_grid]])
{
  const uint total = (uint)(n * Hq * D);
  if (gid >= total) { return; }
  const int d = (int)(gid % (uint)D);
  const int j = (int)((gid / (uint)D) % (uint)Hq);
  const int t = (int)(gid / (uint)(Hq * D));
  const VPIPE_ELT a = t < r_s ? atd[((uint)j * r_s + t) * D + d]
                              : at[((uint)j * n + t) * D + d];
  const float gv = float(qf[(ulong)t * Nf + j * 2 * D + D + d]);
  att[gid] = VPIPE_ELT(float(a) * (1.0f / (1.0f + metal::exp(-gv))));
}

// The embedding into every stream: X[m, s*H + h] = x[m, h]. grid {H, S, M}.
kernel void hc_tile_f16(
    const device VPIPE_ELT* x [[buffer(0)]],   // [M, H]
    device VPIPE_ELT*       X [[buffer(1)]],   // [M, S*H]
    constant int&           H [[buffer(2)]],
    constant int&           S [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H) { return; }
  X[((ulong)gid.z * (ulong)S + gid.y) * (ulong)H + gid.x] =
      x[(ulong)gid.z * (ulong)H + gid.x];
}

// A DFlash drafter's tap: a completed layer's output averaged over the S
// streams (DFlash 2 for Qwen3.8-Flash-Next was trained on it), out[m, h]
// = mean_s X[row0 + m, s*H + h] -- summed in f32 and rounded once, as
// torch's mean of a bf16 tensor. grid {H, M, 1}.
kernel void hc_stream_mean_f16(
    const device VPIPE_ELT* X    [[buffer(0)]],   // [rows, S*H]
    device VPIPE_ELT*       out  [[buffer(1)]],   // [M, H]
    constant int&           H    [[buffer(2)]],
    constant int&           S    [[buffer(3)]],
    constant int&           row0 [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H) { return; }
  const ulong base = (ulong)(row0 + (int)gid.y) * (ulong)S * (ulong)H + gid.x;
  float acc = 0.0f;
  for (int s = 0; s < S; ++s) { acc += (float)X[base + (ulong)s * H]; }
  out[(ulong)gid.y * (ulong)H + gid.x] = (VPIPE_ELT)(acc / (float)S);
}

// moe_finalize_combined_f16's arithmetic WRITTEN, not added: y = combine(
// partials, w) + g * shared. A hyper-connection injects the MoE output
// into its streams, so there is no residual for the MoE block to add to.
kernel void moe_finalize_set_f16(
    device VPIPE_ELT*       y          [[buffer(0)]],  // [M, H]
    const device VPIPE_ELT* partials   [[buffer(1)]],  // [M*top_k, H]
    const device VPIPE_ELT* w          [[buffer(2)]],  // [M*top_k]
    const device VPIPE_ELT* shared_out [[buffer(3)]],  // [M, H]
    const device VPIPE_ELT* gbuf       [[buffer(4)]],  // [M] sigmoid gate
    constant int&           H          [[buffer(5)]],
    constant int&           top_k      [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  const int h = (int)gid % H;
  const int t = (int)gid / H;
  const int base = t * top_k;
  float acc = 0.0f;
  for (int s = 0; s < top_k; ++s) {
    acc += (float)w[base + s] * (float)partials[(long)(base + s) * H + h];
  }
  const float moe = (float)(VPIPE_ELT)acc;
  const float g = (float)gbuf[t];
  y[gid] = (VPIPE_ELT)(moe + (float)(VPIPE_ELT)(g * (float)shared_out[gid]));
}

// ---------------------------------------------------------------------
// PLE (Qwen4ExpTextPLELayer): the hashed n-gram embedding e [H_e] of each
// token is projected to one key per stream (key_proj, [S*H]) and one
// shared value (value_proj, [H]) -- fused here as kv [M, S*H + H]. Each
// stream's normed activations gate the value:
//   kn = grouped RMS(key) (1 + w_k), qn = grouped RMS(X) (1 + w_q)
//   g  = sum_h kn qn / sqrt(H);  g = sign(g) sqrt(max(|g|, 1e-6))
//   gv[s] = sigmoid(g) * value;  gvn = grouped RMS(gv) (1 + w_c)
// and a dilated depthwise conv over time adds local context:
//   X += gv + silu(conv(gvn))
// ---------------------------------------------------------------------

// One threadgroup per (stream s, row m): grid {256, S, M}.
kernel void ple_gate_f16(
    const device VPIPE_ELT* kv  [[buffer(0)]],   // [M, S*H + H]
    const device VPIPE_ELT* X   [[buffer(1)]],   // [M, S*H]
    const device float*     wk  [[buffer(2)]],   // [S*H] 1 + norm_key
    const device float*     wq  [[buffer(3)]],   // [S*H] 1 + norm_query
    const device float*     wc  [[buffer(4)]],   // [S*H] 1 + norm_conv
    device VPIPE_ELT*       gv  [[buffer(5)]],   // [M, S*H]
    device VPIPE_ELT*       gvn [[buffer(6)]],   // [M, S*H]
    constant int&           H   [[buffer(7)]],
    constant int&           S   [[buffer(8)]],
    constant float&         eps [[buffer(9)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 tp   [[thread_position_in_threadgroup]],
    uint3 tsz  [[threads_per_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const uint tid = tp.x, ntg = tsz.x;
  threadgroup float red[32];
  const uint nsg = (ntg + 31) / 32;
  const int s = (int)tg.y;
  const ulong m = tg.z;
  const ulong krow = m * (ulong)(S * H + H);
  const ulong key = krow + (ulong)s * (ulong)H;
  const ulong val = krow + (ulong)S * (ulong)H;
  const ulong xr = (m * (ulong)S + (ulong)s) * (ulong)H;
  const device float* wks = wk + (ulong)s * H;
  const device float* wqs = wq + (ulong)s * H;
  const device float* wcs = wc + (ulong)s * H;
  float sk = 0.0f, sq = 0.0f, sv = 0.0f;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    const float k = (float)kv[key + h], q = (float)X[xr + h];
    const float v = (float)kv[val + h];
    sk += k * k;
    sq += q * q;
    sv += v * v;
  }
  sk = tg_sum_(sk, red, tid, sid, lane, nsg);
  sq = tg_sum_(sq, red, tid, sid, lane, nsg);
  sv = tg_sum_(sv, red, tid, sid, lane, nsg);
  const float rk = metal::rsqrt(sk / (float)H + eps);
  const float rq = metal::rsqrt(sq / (float)H + eps);
  float dot = 0.0f;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    // kn and qn are element-typed tensors in the reference.
    const float kn = (float)(VPIPE_ELT)((float)kv[key + h] * rk * wks[h]);
    const float qn = (float)(VPIPE_ELT)((float)X[xr + h] * rq * wqs[h]);
    dot += kn * qn;
  }
  dot = tg_sum_(dot, red, tid, sid, lane, nsg);
  float g = dot / metal::sqrt((float)H);
  const float sg = g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
  g = sg * metal::sqrt(metal::max(metal::fabs(g), 1e-6f));
  const float sig = (float)(VPIPE_ELT)sigmoid_(g);
  // gv = sig * value (rounded), and its own RMS: the sum of the ROUNDED
  // gv squared, as the reference normalises the stored tensor.
  float sg2 = 0.0f;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    const float v = (float)(VPIPE_ELT)(sig * (float)kv[val + h]);
    sg2 += v * v;
  }
  sg2 = tg_sum_(sg2, red, tid, sid, lane, nsg);
  const float rg = metal::rsqrt(sg2 / (float)H + eps);
  (void)sv;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    const VPIPE_ELT v = (VPIPE_ELT)(sig * (float)kv[val + h]);
    gv[xr + h] = v;
    gvn[xr + h] = (VPIPE_ELT)((float)v * rg * wcs[h]);
  }
}

// X[m, c] += gv[m, c] + silu(sum_j w[c, j] gvn[m - (K-1-j) * dil, c]), rows
// before the chunk read from `state` (the last (K-1)*dil rows of gvn the
// context saw, oldest first). grid {C, M}.
kernel void ple_conv_f16(
    device VPIPE_ELT*       X     [[buffer(0)]],   // [M, C]
    const device VPIPE_ELT* gv    [[buffer(1)]],   // [M, C]
    const device VPIPE_ELT* gvn   [[buffer(2)]],   // [M, C]
    const device VPIPE_ELT* state [[buffer(3)]],   // [(K-1)*dil, C]
    const device VPIPE_ELT* w     [[buffer(4)]],   // [C, K]
    constant int&           C     [[buffer(5)]],
    constant int&           M     [[buffer(6)]],
    constant int&           K     [[buffer(7)]],
    constant int&           dil   [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, m = (int)gid.y;
  if (c >= C || m >= M) { return; }
  const int hist = (K - 1) * dil;
  float acc = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int r = m - (K - 1 - j) * dil;
    const float v = r >= 0 ? (float)gvn[(ulong)r * C + c]
                           : (float)state[(ulong)(hist + r) * C + c];
    acc += (float)w[(ulong)c * K + j] * v;
  }
  const float conv = (float)(VPIPE_ELT)acc;
  const float act = (float)(VPIPE_ELT)(conv * sigmoid_(conv));
  const ulong i = (ulong)m * C + c;
  const float out = (float)(VPIPE_ELT)((float)gv[i] + act);
  X[i] = (VPIPE_ELT)((float)X[i] + out);
}

// The conv history after the chunk: the last (K-1)*dil rows of
// [state_in ; gvn], oldest first. state_out may BE state_in (each thread
// walks its channel oldest first, reading a row no earlier than the one it
// writes). grid {C}.
kernel void ple_state_f16(
    const device VPIPE_ELT* state_in  [[buffer(0)]],
    const device VPIPE_ELT* gvn       [[buffer(1)]],   // [M, C]
    device VPIPE_ELT*       state_out [[buffer(2)]],
    constant int&           C         [[buffer(3)]],
    constant int&           M         [[buffer(4)]],
    constant int&           hist      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const int c = (int)gid;
  if (c >= C) { return; }
  for (int i = 0; i < hist; ++i) {
    const int r = M + i;           // row in [state ; gvn]
    const float v = r < hist ? (float)state_in[(ulong)r * C + c]
                             : (float)gvn[(ulong)(r - hist) * C + c];
    state_out[(ulong)i * C + c] = (VPIPE_ELT)v;
  }
}

// ---------------------------------------------------------------------
// QSA, the token indexer on the full-attention layers
// (Qwen4ExpTextQSAIndexer). Its keys live in the paged K/V pool of three
// PSEUDO layers past the model's own (four full layers each): full layer f
// keeps token t's raw index key (d values) in the K pool of pseudo layer
// f / 4, kv head (f % 4) / 2, values ((f % 4) % 2) * d ..; the POOLED key of
// the block ending at t (mean of 4 raw keys, k_layernorm, RoPE at the
// block's first token) at the same place in the V pool. The block's FIRST
// token keeps its RoPE position there -- three axes (mROPE), int32 as six
// 16-bit halves at the start of its V-pool row, which no pooled key uses
// (ratio > 1) -- so a block an image starts ropes at the image's position.
// A page-table entry is (page, valid tokens, global start); after a branch
// pages can be partly filled, so a token's slot is found by search.
// ---------------------------------------------------------------------

namespace {

struct PagedLoc { int page; int slot; };

inline PagedLoc locate_(const device int* pt, int n_pages, int pos) {
  int lo = 0, hi = n_pages - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (pt[mid * 3 + 2] <= pos) { lo = mid; } else { hi = mid - 1; }
  }
  return {pt[lo * 3 + 0], pos - pt[lo * 3 + 2]};
}

// Element offset of (token pos, kv head, value 0) in a paged pool laid out
// [page][kv head][page_tokens][D].
inline ulong pool_at_(const device int* pt, int n_pages, int pos,
                      int page_tokens, int hkv, int D, int head) {
  const PagedLoc l = locate_(pt, n_pages, pos);
  return (((ulong)l.page * hkv + head) * page_tokens + l.slot) * (ulong)D;
}

}  // namespace

// Raw index keys into the pseudo K pool: row r of `ix` [n, W] (its key at
// column kcol) is token q_offset + r; a block's first token's RoPE position
// (qpos [n, 3]) into the V pool. grid {d, n}.
kernel void qsa_store_f16(
    const device VPIPE_ELT* ix   [[buffer(0)]],
    device VPIPE_ELT*       pool [[buffer(1)]],
    const device int*       pt   [[buffer(2)]],
    constant int& n_pages        [[buffer(3)]],
    constant int& page_tokens    [[buffer(4)]],
    constant int& hkv            [[buffer(5)]],
    constant int& D              [[buffer(6)]],
    constant int& q_offset       [[buffer(7)]],
    constant int& W              [[buffer(8)]],
    constant int& kcol           [[buffer(9)]],
    constant int& head           [[buffer(10)]],
    constant int& voff           [[buffer(11)]],
    constant int& d              [[buffer(12)]],
    constant int& n              [[buffer(13)]],
    device VPIPE_ELT*       vpool [[buffer(14)]],
    const device int*       qpos  [[buffer(15)]],   // [n, 3]
    constant int& ratio          [[buffer(16)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= d || (int)gid.y >= n) { return; }
  const int t = q_offset + (int)gid.y;
  const ulong at = pool_at_(pt, n_pages, t, page_tokens, hkv, D, head);
  pool[at + voff + gid.x] = ix[(ulong)gid.y * W + kcol + gid.x];
  if (gid.x < 6 && ratio > 1 && t % ratio == 0) {
    const uint p = (uint)qpos[(ulong)gid.y * 3 + gid.x / 2];
    device ushort* v16 = (device ushort*)(vpool + at + voff);
    v16[gid.x] = (ushort)((gid.x & 1) ? (p >> 16) : (p & 0xffffu));
  }
}

// The pooled keys of blocks [b0, b0 + nb): one threadgroup (d threads) a
// block. mean (f32) of the block's raw keys, rounded; RMS-normed with
// (1 + w) in f32 and rounded; RoPE (rotate-half over the first `rot`
// values) at the block's first token's position (kept by qsa_store; pair p
// takes axis axis[p] of it), rounded -- each rounding where the
// reference's tensor is stored.
kernel void qsa_pool_f16(
    const device VPIPE_ELT* kpool [[buffer(0)]],
    device VPIPE_ELT*       vpool [[buffer(1)]],
    const device int*       pt    [[buffer(2)]],
    const device float*     w     [[buffer(3)]],   // [d] 1 + k_layernorm
    const device float*     inv_freq [[buffer(4)]],  // [rot / 2]
    constant int& n_pages        [[buffer(5)]],
    constant int& page_tokens    [[buffer(6)]],
    constant int& hkv            [[buffer(7)]],
    constant int& D              [[buffer(8)]],
    constant int& head           [[buffer(9)]],
    constant int& voff           [[buffer(10)]],
    constant int& d              [[buffer(11)]],
    constant int& ratio          [[buffer(12)]],
    constant int& b0             [[buffer(13)]],
    constant int& rot            [[buffer(14)]],
    const device int*       axis [[buffer(15)]],     // [rot / 2]
    constant float& eps          [[buffer(16)]],
    uint  tgb  [[threadgroup_position_in_grid]],
    uint  i    [[thread_position_in_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float red[32];
  threadgroup float val[256];
  const int b = b0 + (int)tgb;
  const int t0 = b * ratio;
  float acc = 0.0f;
  for (int j = 0; j < ratio; ++j) {
    const ulong at = pool_at_(pt, n_pages, t0 + j, page_tokens, hkv, D, head);
    acc += (float)kpool[at + voff + i];
  }
  const float mean = (float)(VPIPE_ELT)(acc / (float)ratio);
  const uint nsg = ((uint)d + 31) / 32;
  float ss = simd_sum(mean * mean);
  if (lane == 0) { red[sid] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = lane < nsg ? red[lane] : 0.0f;
  tot = simd_sum(tot);
  const float r = metal::rsqrt(tot / (float)d + eps);
  val[i] = (float)(VPIPE_ELT)(mean * r * w[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = val[i];
  const int half_r = rot / 2;
  if ((int)i < rot) {
    const int p = (int)i < half_r ? (int)i : (int)i - half_r;
    const ulong a0 = pool_at_(pt, n_pages, t0, page_tokens, hkv, D, head);
    const device ushort* v16 = (const device ushort*)(vpool + a0 + voff);
    const int ax = axis[p];
    const uint pos = (uint)v16[2 * ax] | ((uint)v16[2 * ax + 1] << 16);
    const float ang = (float)(int)pos * inv_freq[p];
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    out = (int)i < half_r ? val[i] * cs - val[i + half_r] * sn
                          : val[i] * cs + val[i - half_r] * sn;
  }
  const ulong at = pool_at_(pt, n_pages, t0 + ratio - 1, page_tokens, hkv, D,
                            head);
  vpool[at + voff + i] = (VPIPE_ELT)out;
}

// The index queries: ix's first `heads` heads of d, RMS-normed with (1 +
// w) and RoPE'd at their row's position (qpos [n, 3]; pair p its axis
// axis[p]), rounded at each step, into q [n, heads, d]. One threadgroup (d
// threads) per (head, row): grid {d, heads, n}.
kernel void qsa_query_f16(
    const device VPIPE_ELT* ix   [[buffer(0)]],
    device VPIPE_ELT*       q    [[buffer(1)]],
    const device float*     w    [[buffer(2)]],
    const device float*     inv_freq [[buffer(3)]],
    constant int& W              [[buffer(4)]],
    constant int& d              [[buffer(5)]],
    constant int& heads          [[buffer(6)]],
    constant int& rot            [[buffer(7)]],
    const device int*       qpos [[buffer(8)]],      // [n, 3]
    constant float& eps          [[buffer(9)]],
    const device int*       axis [[buffer(10)]],     // [rot / 2]
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 tp   [[thread_position_in_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float red[32];
  threadgroup float val[256];
  const uint i = tp.x;
  const int h = (int)tg.y, r = (int)tg.z;
  const float x = (float)ix[(ulong)r * W + (ulong)h * d + i];
  const uint nsg = ((uint)d + 31) / 32;
  float ss = simd_sum(x * x);
  if (lane == 0) { red[sid] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = lane < nsg ? red[lane] : 0.0f;
  tot = simd_sum(tot);
  val[i] = (float)(VPIPE_ELT)(x * metal::rsqrt(tot / (float)d + eps) * w[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = val[i];
  const int half_r = rot / 2;
  if ((int)i < rot) {
    const int p = (int)i < half_r ? (int)i : (int)i - half_r;
    const float ang = (float)qpos[(ulong)r * 3 + axis[p]] * inv_freq[p];
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    out = (int)i < half_r ? val[i] * cs - val[i + half_r] * sn
                          : val[i] * cs + val[i - half_r] * sn;
  }
  q[((ulong)r * heads + h) * d + i] = (VPIPE_ELT)out;
}

// Block scores for the sparse rows: s[r, b] = sum_h relu(q[r0 + r, h] .
// pooled[b]) / sqrt(d) over the blocks that row sees whole (b < (p + 1) /
// ratio, p = q_offset + r0 + r); the rest -1 (never selected). grid
// {nb_max, rows}.
kernel void qsa_score_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],   // [n, heads, d]
    const device VPIPE_ELT* vpool [[buffer(1)]],
    const device int*       pt    [[buffer(2)]],
    device float*           s     [[buffer(3)]],   // [rows, nb_max]
    constant int& n_pages        [[buffer(4)]],
    constant int& page_tokens    [[buffer(5)]],
    constant int& hkv            [[buffer(6)]],
    constant int& D              [[buffer(7)]],
    constant int& head           [[buffer(8)]],
    constant int& voff           [[buffer(9)]],
    constant int& d              [[buffer(10)]],
    constant int& heads          [[buffer(11)]],
    constant int& ratio          [[buffer(12)]],
    constant int& q_offset       [[buffer(13)]],
    constant int& r0             [[buffer(14)]],
    constant int& nb_max         [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int b = (int)gid.x, r = (int)gid.y;
  if (b >= nb_max) { return; }
  const int p = q_offset + r0 + r;
  const int nb = (p + 1) / ratio;
  device float* srow = s + (ulong)r * nb_max;
  if (b >= nb) { srow[b] = -1.0f; return; }
  const ulong at = pool_at_(pt, n_pages, b * ratio + ratio - 1, page_tokens,
                            hkv, D, head) + voff;
  const device VPIPE_ELT* qr = q + (ulong)(r0 + r) * heads * d;
  float tot = 0.0f;
  for (int h = 0; h < heads; ++h) {
    float dot = 0.0f;
    for (int i = 0; i < d; ++i) {
      dot += (float)qr[h * d + i] * (float)vpool[at + i];
    }
    tot += metal::max(dot, 0.0f);
  }
  srow[b] = tot / metal::sqrt((float)d);
}

// qsa_score_f16 for a FEW rows (a decode step, a verify), a simdgroup a
// (block, row): its lanes split each head's d, the head's dot summed
// across them. A thread a (block, row) walked its 512 products alone --
// one thread's chain the whole kernel's time, ~35 us a layer, a decode
// step's 1800 blocks at 7k tokens filling seven threadgroups (M5 Ultra).
// A row's sums are its own, so a verify's row is a decode step's. d <=
// 128, d % 32 == 0. grid {32 nb_max, rows}, tg {256, 1}.
kernel void qsa_score_simd_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],   // [n, heads, d]
    const device VPIPE_ELT* vpool [[buffer(1)]],
    const device int*       pt    [[buffer(2)]],
    device float*           s     [[buffer(3)]],   // [rows, nb_max]
    constant int& n_pages        [[buffer(4)]],
    constant int& page_tokens    [[buffer(5)]],
    constant int& hkv            [[buffer(6)]],
    constant int& D              [[buffer(7)]],
    constant int& head           [[buffer(8)]],
    constant int& voff           [[buffer(9)]],
    constant int& d              [[buffer(10)]],
    constant int& heads          [[buffer(11)]],
    constant int& ratio          [[buffer(12)]],
    constant int& q_offset       [[buffer(13)]],
    constant int& r0             [[buffer(14)]],
    constant int& nb_max         [[buffer(15)]],
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int b = (int)gid.x / 32, r = (int)gid.y;
  if (b >= nb_max) { return; }
  const int p = q_offset + r0 + r;
  const int nb = (p + 1) / ratio;
  device float* srow = s + (ulong)r * nb_max;
  if (b >= nb) {
    if (lane == 0) { srow[b] = -1.0f; }
    return;
  }
  const ulong at = pool_at_(pt, n_pages, b * ratio + ratio - 1, page_tokens,
                            hkv, D, head) + voff;
  const device VPIPE_ELT* qr = q + (ulong)(r0 + r) * heads * d;
  const int per = d / 32;                  // <= 4
  float kv[4];
  for (int j = 0; j < per; ++j) {
    kv[j] = (float)vpool[at + (ulong)((int)lane * per + j)];
  }
  float tot = 0.0f;
  for (int h = 0; h < heads; ++h) {
    float dot = 0.0f;
    for (int j = 0; j < per; ++j) {
      dot += (float)qr[h * d + (int)lane * per + j] * kv[j];
    }
    dot = simd_sum(dot);
    tot += metal::max(dot, 0.0f);
  }
  if (lane == 0) { srow[b] = tot / metal::sqrt((float)d); }
}

// qsa_score_f16 for a PREFILL's sparse rows, as matrix tiles: a
// threadgroup a tile of 32 rows x 32 blocks, the blocks' pooled keys
// staged once for the tile and its four simdgroups (8 rows each), each
// head's 8 x 8 dot tiles summed over d by simdgroup MMA, relu'd and added
// up. (A thread a (block, row) walked 512 products alone: ~30 ms of a
// 2048-row piece on an M5 Ultra.) Its sums run in another order than the
// scalar kernel's, so a decode step and a verify keep that one. q is read
// past the batch's last row to its tile's end: the caller pads it. d <=
// 128, d % 8 == 0. grid {128, ceil(nb_max / 32), ceil(rows / 32)}, tg
// {128, 1, 1}.
kernel void qsa_score_mma_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],   // [n, heads, d]
    const device VPIPE_ELT* vpool [[buffer(1)]],
    const device int*       pt    [[buffer(2)]],
    device float*           s     [[buffer(3)]],   // [rows, nb_max]
    constant int& n_pages        [[buffer(4)]],
    constant int& page_tokens    [[buffer(5)]],
    constant int& hkv            [[buffer(6)]],
    constant int& D              [[buffer(7)]],
    constant int& head           [[buffer(8)]],
    constant int& voff           [[buffer(9)]],
    constant int& d              [[buffer(10)]],
    constant int& heads          [[buffer(11)]],
    constant int& ratio          [[buffer(12)]],
    constant int& q_offset       [[buffer(13)]],
    constant int& r0             [[buffer(14)]],
    constant int& nb_max         [[buffer(15)]],
    constant int& rows           [[buffer(16)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgi  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int TB = 32;
  threadgroup VPIPE_ELT kb[TB * 128];
  threadgroup ulong boff[TB];
  const int b0 = (int)tg.y * TB, t0 = (int)tg.z * 32;
  if (lid < (uint)TB) {
    const int b = b0 + (int)lid;
    boff[lid] = b < nb_max
        ? pool_at_(pt, n_pages, b * ratio + ratio - 1, page_tokens, hkv, D,
                   head) + voff
        : ~0ul;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v8 = d / 8;   // 8-value pieces a key
  for (int e = (int)lid; e < TB * v8; e += 128) {
    const int bb = e / v8, i = (e % v8) * 8;
    uint4 v = uint4(0);
    if (boff[bb] != ~0ul) {
      v = *(const device uint4*)(vpool + boff[bb] + i);
    }
    *(threadgroup uint4*)(kb + bb * d + i) = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int rr = t0 + (int)sgi * 8;   // this simdgroup's first row
  const int qid = (int)lane / 4;
  const int fm = (qid & 4) + (((int)lane / 2) % 4);
  const int fn = (qid & 2) * 2 + ((int)lane % 2) * 2;
  float tot[4][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
  for (int h = 0; h < heads; ++h) {
    simdgroup_matrix<float, 8, 8> acc[4];
    for (int j = 0; j < 4; ++j) {
      acc[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const device VPIPE_ELT* qb = q + ((ulong)(r0 + rr) * heads + h) * d;
    for (int kk = 0; kk < v8; ++kk) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> a;
      simdgroup_load(a, qb + kk * 8, (ulong)heads * d);
      for (int j = 0; j < 4; ++j) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> b;
        simdgroup_load(b, kb + (j * 8) * d + kk * 8, (ulong)d, ulong2(0),
                       true);
        simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
      }
    }
    for (int j = 0; j < 4; ++j) {
      tot[j][0] += metal::max(acc[j].thread_elements()[0], 0.0f);
      tot[j][1] += metal::max(acc[j].thread_elements()[1], 0.0f);
    }
  }
  const int row = rr + fm;
  if (row >= rows) { return; }
  const int nb = (q_offset + r0 + row + 1) / ratio;
  const float inv = 1.0f / metal::sqrt((float)d);
  device float* srow = s + (ulong)row * nb_max;
  for (int j = 0; j < 4; ++j) {
    for (int e = 0; e < 2; ++e) {
      const int b = b0 + j * 8 + fn + e;
      if (b < nb_max) { srow[b] = b < nb ? tot[j][e] * inv : -1.0f; }
    }
  }
}

// Top-k blocks of each sparse row (k = budget / ratio), by radix select on
// the scores' bits (all >= 0 or the -1 of an unseen block), ties at the
// threshold to the EARLIEST blocks; then the row's tokens: every selected
// block's ratio tokens in block order, then the tail the incomplete block
// holds (tokens nb * ratio .. p). sel [rows, k * ratio + ratio - 1],
// nsel [rows]. One threadgroup of 256 per row: each pass's digit found and
// the emit's offsets summed by prefix sums across it (one thread walked
// 256 bins and two 1024-long sums, ~30 ms of a 2048-row piece). The
// selection is the definition's, whatever the threads.
kernel void qsa_select_f16(
    const device float* s     [[buffer(0)]],
    device int*         sel   [[buffer(1)]],
    device int*         nsel  [[buffer(2)]],
    constant int& nb_max      [[buffer(3)]],
    constant int& k           [[buffer(4)]],
    constant int& ratio       [[buffer(5)]],
    constant int& q_offset    [[buffer(6)]],
    constant int& r0          [[buffer(7)]],
    uint  r   [[threadgroup_position_in_grid]],
    uint  tid [[thread_position_in_threadgroup]],
    uint  ntg [[threads_per_threadgroup]],
    uint  sid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup atomic_uint hist[256];
  threadgroup uint shared_u[4];
  threadgroup uint sgs[32];
  const uint nsg = ntg / 32;
  const device float* srow = s + (ulong)r * nb_max;
  const int p = q_offset + r0 + (int)r;
  const int nb = (p + 1) / ratio;
  const int width = k * ratio + ratio - 1;
  device int* out = sel + (ulong)r * width;
  // Radix select: the k-th largest key, MSB first, 8 bits a pass. A
  // non-negative float's bits order as the float does; -1 (unseen) maps
  // below every score.
  auto key = [&](int b) -> uint {
    const float v = srow[b];
    return v < 0.0f ? 0u : as_type<uint>(v) + 1u;
  };
  uint prefix = 0, mask = 0, want = (uint)k;
  for (int pass = 3; pass >= 0; --pass) {
    for (uint i = tid; i < 256; i += ntg) {
      atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint shift = (uint)pass * 8;
    for (int b = (int)tid; b < nb; b += (int)ntg) {
      const uint kk = key(b);
      if ((kk & mask) == prefix) {
        atomic_fetch_add_explicit(&hist[(kk >> shift) & 255u], 1u,
                                  memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Thread t holds digit 255 - t: the keys above it are its exclusive
    // sum; the digit is the one whose bin takes the count to `want`.
    const uint dg = 255u - tid;
    const uint hv = tid < 256
        ? atomic_load_explicit(&hist[dg], memory_order_relaxed) : 0u;
    uint total = 0;
    const uint above = tg_excl_sum_(hv, sgs, sid, lane, nsg, total);
    if (tid < 256 && above < want && above + hv >= want) {
      shared_u[0] = dg;
      shared_u[1] = want - above;   // still wanted inside this digit
    }
    if (tid == 255 && above + hv < want) {   // fewer keys than k
      shared_u[0] = 0;
      shared_u[1] = want - (above + hv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= shared_u[0] << shift;
    mask |= 255u << shift;
    want = shared_u[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint thresh = prefix;   // the k-th largest key
  const uint ties = want;       // how many keys == thresh to take
  // Emit in block order: each thread a contiguous run of blocks.
  const int per = (nb + (int)ntg - 1) / (int)ntg;
  const int lo = min(nb, (int)tid * per), hi = min(nb, lo + per);
  uint ngt = 0, neq = 0;
  for (int b = lo; b < hi; ++b) {
    const uint kk = key(b);
    ngt += kk > thresh;
    neq += kk == thresh;
  }
  uint tot_eq = 0;
  const uint eq_before = tg_excl_sum_(neq, sgs, sid, lane, nsg, tot_eq);
  const uint take_eq = eq_before >= ties ? 0u : min(neq, ties - eq_before);
  uint nblk = 0;
  uint o = tg_excl_sum_(ngt + take_eq, sgs, sid, lane, nsg, nblk);
  uint eq_seen = eq_before;
  for (int b = lo; b < hi; ++b) {
    const uint kk = key(b);
    bool pick = kk > thresh;
    if (kk == thresh) { pick = eq_seen < ties; ++eq_seen; }
    if (pick) {
      for (int j = 0; j < ratio; ++j) { out[o * ratio + j] = b * ratio + j; }
      ++o;
    }
  }
  if (tid == 0) {
    int c = (int)nblk * ratio;
    for (int t = nb * ratio; t <= p; ++t) { out[c++] = t; }
    nsel[r] = c;
  }
}

// qsa_attn_f16 for a FEW rows (a decode step, a speculative verify): the
// same softmax over a row's selected tokens, a threadgroup of 16
// simdgroups to a (head, row) where one simdgroup did it all -- a lane's
// whole 256-wide dot a selected key, then its 8 outputs summed over every
// selected key, ~32 000 dependent steps a lane past 2051 tokens. Here the
// scores are a simdgroup a key (its lanes splitting D), the max and the
// sum reduced across the threadgroup, and the values summed by 8
// simdgroups over contiguous runs of the keys, the runs added in order.
// A row's arithmetic is its own threadgroup's, so a verify's row is the
// decode step's at its position. grid {32, 16 * Hq, rows}, tg {32, 16, 1}.
constant constexpr int kQsaSg = 16;     // simdgroups a (head, row)
constant constexpr int kQsaPvSg = 8;    // of them summing the values

kernel void qsa_attn_split_f16(
    const device VPIPE_ELT* qt    [[buffer(0)]],   // [Hq, n, D]
    const device VPIPE_ELT* kp    [[buffer(1)]],
    const device VPIPE_ELT* vp    [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],   // [Hq, n, D]
    const device int*       pt    [[buffer(4)]],
    const device int*       sel   [[buffer(5)]],
    const device int*       nsel  [[buffer(6)]],
    constant float& scale        [[buffer(7)]],
    constant int& D              [[buffer(8)]],
    constant int& Hq             [[buffer(9)]],
    constant int& Hkv            [[buffer(10)]],
    constant int& n              [[buffer(11)]],
    constant int& page_tokens    [[buffer(12)]],
    constant int& n_pages        [[buffer(13)]],
    constant int& width          [[buffer(14)]],
    constant int& r0             [[buffer(15)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 tp   [[thread_position_in_threadgroup]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float qv[256];
  threadgroup float sc[2112];
  threadgroup uint koff[2112];
  threadgroup float red[kQsaPvSg * 256];
  threadgroup float rs[kQsaSg];
  const int tid = (int)(sid * 32 + lane), nth = kQsaSg * 32;
  const int h = (int)tg.y, r = (int)tg.z;
  const int row = r0 + r;
  const int kvh = h / (Hq / Hkv);
  const device VPIPE_ELT* qrow = qt + ((ulong)h * n + row) * D;
  for (int i = tid; i < D; i += nth) { qv[i] = (float)qrow[i]; }
  const device int* srow = sel + (ulong)r * width;
  const int m = nsel[r];
  for (int j = tid; j < m; j += nth) {
    koff[j] = (uint)pool_at_(pt, n_pages, srow[j], page_tokens, Hkv, D, kvh);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Scores: a simdgroup a key, its lanes over D.
  const int per = D / 32;
  for (int j = (int)sid; j < m; j += kQsaSg) {
    const device VPIPE_ELT* kr = kp + koff[j];
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      const int i = (int)lane * per + p;
      dot += qv[i] * (float)kr[i];
    }
    dot = simd_sum(dot);
    if (lane == 0) { sc[j] = dot * scale; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The max, then exp and the sum, across the threadgroup.
  float mx = -INFINITY;
  for (int j = tid; j < m; j += nth) { mx = metal::max(mx, sc[j]); }
  mx = simd_max(mx);
  if (lane == 0) { rs[sid] = mx; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = -INFINITY;
  for (int g = 0; g < kQsaSg; ++g) { mx = metal::max(mx, rs[g]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  for (int j = tid; j < m; j += nth) {
    const float e = metal::exp(sc[j] - mx);
    sc[j] = e;
    sum += e;
  }
  sum = simd_sum(sum);
  if (lane == 0) { rs[sid] = sum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < kQsaSg; ++g) { tot += rs[g]; }
  const float inv = 1.0f / tot;
  // The values: kQsaPvSg simdgroups, each a contiguous run of the keys,
  // a lane its D / 32 outputs.
  if ((int)sid < kQsaPvSg) {
    const int j0 = (int)sid * m / kQsaPvSg;
    const int j1 = ((int)sid + 1) * m / kQsaPvSg;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = j0; j < j1; ++j) {
      const float w = sc[j];
      const device VPIPE_ELT* vr = vp + koff[j];
      for (int p = 0; p < per; ++p) {
        acc[p] += w * (float)vr[(int)lane * per + p];
      }
    }
    for (int p = 0; p < per; ++p) {
      red[(int)sid * 256 + (int)lane * per + p] = acc[p];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device VPIPE_ELT* orow = out + ((ulong)h * n + row) * D;
  for (int i = tid; i < D; i += nth) {
    float t = 0.0f;
    for (int g = 0; g < kQsaPvSg; ++g) { t += red[g * 256 + i]; }
    orow[i] = (VPIPE_ELT)(t * inv);
  }
}

// qsa_attn_f16 for a FEW rows in TWO launches, flash-decoding's: a row's
// selected keys in chunks of kQsaChunk, a threadgroup a (head, chunk, row)
// -- the chunk's scores, their max, the exps' sum and the values' sums
// under them, into part -- then a threadgroup a (head, row) adding the
// chunks in order, each brought to the row's max. qsa_attn_split_f16's
// threadgroup a (head, row) left a decode step 24 threadgroups on an M5
// Ultra's 80 cores: 330 us a layer past 2051 tokens. A row's arithmetic is
// its own chunks', so a verify's row is the decode step's at its
// position. part: [rows][Hq][nch][2 + D] f32 -- the max, the sum, the
// sums. grid {128, Hq, rows * nch}, tg {128, 1, 1}, nch = ceil(width /
// kQsaChunk); D <= 256.
constant constexpr int kQsaChunk = 64;
constant constexpr int kQsaPartSg = 4;

kernel void qsa_attn_part_f16(
    const device VPIPE_ELT* qt    [[buffer(0)]],   // [Hq, n, D]
    const device VPIPE_ELT* kp    [[buffer(1)]],
    const device VPIPE_ELT* vp    [[buffer(2)]],
    device float*           part  [[buffer(3)]],
    const device int*       pt    [[buffer(4)]],
    const device int*       sel   [[buffer(5)]],
    const device int*       nsel  [[buffer(6)]],
    constant float& scale        [[buffer(7)]],
    constant int& D              [[buffer(8)]],
    constant int& Hq             [[buffer(9)]],
    constant int& Hkv            [[buffer(10)]],
    constant int& n              [[buffer(11)]],
    constant int& page_tokens    [[buffer(12)]],
    constant int& n_pages        [[buffer(13)]],
    constant int& width          [[buffer(14)]],
    constant int& r0             [[buffer(15)]],
    constant int& nch            [[buffer(16)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sid  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float qv[256];
  threadgroup float sc[kQsaChunk];
  threadgroup uint koff[kQsaChunk];
  threadgroup float red[kQsaPartSg * 256];
  const int tid = (int)(sid * 32 + lane), nth = kQsaPartSg * 32;
  const int h = (int)tg.y;
  const int r = (int)tg.z / nch, c = (int)tg.z % nch;
  const int m = nsel[r];
  const int j0 = c * kQsaChunk, cnt = min(m - j0, kQsaChunk);
  if (cnt <= 0) { return; }                // past the row's keys: unread
  const int row = r0 + r;
  const int kvh = h / (Hq / Hkv);
  device float* pr = part + (((ulong)r * Hq + h) * nch + c) * (ulong)(2 + D);
  const device VPIPE_ELT* qrow = qt + ((ulong)h * n + row) * D;
  for (int i = tid; i < D; i += nth) { qv[i] = (float)qrow[i]; }
  const device int* srow = sel + (ulong)r * width + j0;
  for (int j = tid; j < cnt; j += nth) {
    koff[j] = (uint)pool_at_(pt, n_pages, srow[j], page_tokens, Hkv, D, kvh);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Scores: a simdgroup a key, its lanes over D.
  const int per = D / 32;
  for (int j = (int)sid; j < cnt; j += kQsaPartSg) {
    const device VPIPE_ELT* kr = kp + koff[j];
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      const int i = (int)lane * per + p;
      dot += qv[i] * (float)kr[i];
    }
    dot = simd_sum(dot);
    if (lane == 0) { sc[j] = dot * scale; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The chunk's max (every simdgroup alike), then simdgroup 0 the exps.
  float mx = -INFINITY;
  for (int j = (int)lane; j < cnt; j += 32) { mx = metal::max(mx, sc[j]); }
  mx = simd_max(mx);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sid == 0) {
    float s = 0.0f;
    for (int j = (int)lane; j < cnt; j += 32) {
      const float e = metal::exp(sc[j] - mx);
      sc[j] = e;
      s += e;
    }
    s = simd_sum(s);
    if (lane == 0) { pr[0] = mx; pr[1] = s; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The values: a simdgroup a contiguous run of the keys, a lane D / 32
  // outputs; the runs added in order.
  {
    const int a0 = (int)sid * cnt / kQsaPartSg;
    const int a1 = ((int)sid + 1) * cnt / kQsaPartSg;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = a0; j < a1; ++j) {
      const float w = sc[j];
      const device VPIPE_ELT* vr = vp + koff[j];
      for (int p = 0; p < per; ++p) {
        acc[p] += w * (float)vr[(int)lane * per + p];
      }
    }
    for (int p = 0; p < per; ++p) {
      red[(int)sid * 256 + (int)lane * per + p] = acc[p];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < D; i += nth) {
    float t = 0.0f;
    for (int g = 0; g < kQsaPartSg; ++g) { t += red[g * 256 + i]; }
    pr[2 + i] = t;
  }
}

// The chunks of a (head, row), in order: each one's sums and exps' sum
// brought to the row's max, added, divided. grid {256, Hq, rows}, tg
// {256, 1, 1}.
kernel void qsa_attn_comb_f16(
    const device float*     part  [[buffer(0)]],
    device VPIPE_ELT*       out   [[buffer(1)]],   // [Hq, n, D]
    const device int*       nsel  [[buffer(2)]],
    constant int& D              [[buffer(3)]],
    constant int& Hq             [[buffer(4)]],
    constant int& n              [[buffer(5)]],
    constant int& r0             [[buffer(6)]],
    constant int& nch            [[buffer(7)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
  threadgroup float wc[64];                // nch <= 64
  threadgroup float tot_s;
  const int h = (int)tg.y, r = (int)tg.z;
  const int nc = min(nch, (nsel[r] + kQsaChunk - 1) / kQsaChunk);
  const device float* pr = part + ((ulong)r * Hq + h) * nch * (ulong)(2 + D);
  if (tid == 0) {
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) { M = metal::max(M, pr[c * (2 + D)]); }
    float tot = 0.0f;
    for (int c = 0; c < nc; ++c) {
      const float w = metal::exp(pr[c * (2 + D)] - M);
      wc[c] = w;
      tot += w * pr[c * (2 + D) + 1];
    }
    tot_s = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / tot_s;
  device VPIPE_ELT* orow = out + ((ulong)h * n + (r0 + r)) * D;
  for (int i = (int)tid; i < D; i += 256) {
    float t = 0.0f;
    for (int c = 0; c < nc; ++c) { t += wc[c] * pr[c * (2 + D) + 2 + i]; }
    orow[i] = (VPIPE_ELT)(t * inv);
  }
}

// qsa_attn_f16 for a PREFILL's sparse rows: the q-heads that share a kv
// head attend that row's selected keys together, as matrix-core tiles --
// a threadgroup a (kv head, row), the group's heads the MMA rows, so each
// gathered key is read once for all of them (the per-head kernels read it
// once a head: 12 times at 24 / 2, and on an M5 Ultra 0.75 s of a 2048-row
// piece). Two 8-row bands cover a group of 8 < G <= 16 heads (the second
// band's first rows the first's again, written by the first), four
// simdgroups each a 64-wide slice of D for the scores' partial sums and
// for the values; the keys staged BK at a time, the next BK fetched into
// registers while these are used; the softmax online in f32, P rounded to
// the element type for the values' MMA. Not a decode row's arithmetic:
// a few rows keep qsa_attn_split_f16. D == 256. grid {256, Hkv, rows},
// tg {256, 1, 1}.
constant constexpr int kQsaBk = 16;      // keys a stage

kernel void qsa_attn_mma_f16(
    const device VPIPE_ELT* qt    [[buffer(0)]],   // [Hq, n, D]
    const device VPIPE_ELT* kp    [[buffer(1)]],
    const device VPIPE_ELT* vp    [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],   // [Hq, n, D]
    const device int*       pt    [[buffer(4)]],
    const device int*       sel   [[buffer(5)]],
    const device int*       nsel  [[buffer(6)]],
    constant float& scale        [[buffer(7)]],
    constant int& D              [[buffer(8)]],
    constant int& Hq             [[buffer(9)]],
    constant int& Hkv            [[buffer(10)]],
    constant int& n              [[buffer(11)]],
    constant int& page_tokens    [[buffer(12)]],
    constant int& n_pages        [[buffer(13)]],
    constant int& width          [[buffer(14)]],
    constant int& r0             [[buffer(15)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgi  [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int BK = kQsaBk;
  threadgroup VPIPE_ELT ks[BK * 256];
  threadgroup VPIPE_ELT vs[BK * 256];
  threadgroup float sp[4 * 16 * BK];        // [slice][16 rows][BK]
  threadgroup VPIPE_ELT ps[16 * BK];
  threadgroup float corr[16];
  threadgroup float den[16];
  const int kvh = (int)tg.y, r = (int)tg.z, row = r0 + r;
  const int G = Hq / Hkv, h0 = kvh * G;
  const int wm = (int)sgi / 4, wd = (int)sgi % 4;
  const int band = wm == 0 ? 0 : G - 8;     // the band's first head
  const int dc0 = wd * 64;
  // Apple's 8x8 fragment layout: this lane holds row fm, columns fn and
  // fn + 1 of every tile.
  const int qid = (int)lane / 4;
  const int fm = (qid & 4) + (((int)lane / 2) % 4);
  const int fn = (qid & 2) * 2 + ((int)lane % 2) * 2;
  // The band's queries, this slice's 64 dims: 8 tiles held.
  simdgroup_matrix<VPIPE_ELT, 8, 8> qf[8];
  const device VPIPE_ELT* qb =
      qt + ((ulong)(h0 + band) * n + row) * D + dc0;
  for (int i = 0; i < 8; ++i) {
    simdgroup_load(qf[i], qb + i * 8, (ulong)n * D);
  }
  simdgroup_matrix<float, 8, 8> of[8];
  for (int i = 0; i < 8; ++i) {
    of[i] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  const device int* srow = sel + (ulong)r * width;
  const int m = nsel[r];
  const int nch = (m + BK - 1) / BK;
  // Staging: thread lid copies 16 values of key lid / 16.
  const int st = (int)lid / 16, so = ((int)lid % 16) * 16;
  uint4 k0 = uint4(0), k1 = uint4(0), v0 = uint4(0), v1 = uint4(0);
  auto fetch = [&](int c) {
    const int j = c * BK + st;
    if (j < m) {
      const ulong at =
          pool_at_(pt, n_pages, srow[j], page_tokens, Hkv, D, kvh) + so;
      const device uint4* kq = (const device uint4*)(kp + at);
      const device uint4* vq = (const device uint4*)(vp + at);
      k0 = kq[0]; k1 = kq[1]; v0 = vq[0]; v1 = vq[1];
    } else {
      k0 = uint4(0); k1 = uint4(0); v0 = uint4(0); v1 = uint4(0);
    }
  };
  // The softmax: thread lid one (row, key) of the stage, its row's
  // running max and sum held by the row's 16 lanes alike.
  const int sr = (int)lid / 16, sc = (int)lid % 16;
  float mrow = -INFINITY, lrow = 0.0f;
  fetch(0);
  for (int c = 0; c < nch; ++c) {
    threadgroup uint4* kd = (threadgroup uint4*)(ks + st * 256 + so);
    threadgroup uint4* vd = (threadgroup uint4*)(vs + st * 256 + so);
    kd[0] = k0; kd[1] = k1; vd[0] = v0; vd[1] = v1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c + 1 < nch) { fetch(c + 1); }
    // Scores: this slice's partial sums, the band's 8 rows x 16 keys.
    simdgroup_matrix<float, 8, 8> s0 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> s1 = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int i = 0; i < 8; ++i) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> b;
      simdgroup_load(b, ks, 256, ulong2(dc0 + i * 8, 0), true);
      simdgroup_multiply_accumulate(s0, qf[i], b, s0);
      simdgroup_load(b, ks, 256, ulong2(dc0 + i * 8, 8), true);
      simdgroup_multiply_accumulate(s1, qf[i], b, s1);
    }
    threadgroup float* spr = sp + (wd * 16 + wm * 8) * BK;
    simdgroup_store(s0, spr, BK);
    simdgroup_store(s1, spr + 8, BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = sp[(0 * 16 + sr) * BK + sc] + sp[(1 * 16 + sr) * BK + sc] +
              sp[(2 * 16 + sr) * BK + sc] + sp[(3 * 16 + sr) * BK + sc];
    s = c * BK + sc < m ? s * scale : -INFINITY;
    float cm = s;
    for (ushort o = 8; o > 0; o >>= 1) {
      cm = metal::max(cm, simd_shuffle_xor(cm, o));
    }
    const float mnew = metal::max(mrow, cm);
    const float cr = metal::exp(mrow - mnew);
    // P enters the values' MMA in the element type, so the row's sum is
    // taken over THOSE values: the weights the output is built from then
    // sum to what it is divided by. (Summing the f32 p left each row's
    // scale off by its weights' rounding.)
    const VPIPE_ELT pe = (VPIPE_ELT)metal::exp(s - mnew);
    float psum = (float)pe;
    for (ushort o = 8; o > 0; o >>= 1) {
      psum += simd_shuffle_xor(psum, o);
    }
    lrow = lrow * cr + psum;
    mrow = mnew;
    ps[sr * BK + sc] = pe;
    if (sc == 0) { corr[sr] = cr; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Values: this slice's 64 output columns of the band's rows.
    const float crr = corr[wm * 8 + fm];
    for (int i = 0; i < 8; ++i) {
      of[i].thread_elements()[0] *= crr;
      of[i].thread_elements()[1] *= crr;
    }
    for (int kk = 0; kk < 2; ++kk) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> a;
      simdgroup_load(a, ps + (wm * 8) * BK + kk * 8, BK);
      for (int i = 0; i < 8; ++i) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> b;
        simdgroup_load(b, vs, 256, ulong2(dc0 + i * 8, kk * 8), false);
        simdgroup_multiply_accumulate(of[i], a, b, of[i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sc == 0) { den[sr] = lrow; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (wm == 0 || band + fm >= 8) {
    const float dv = den[wm * 8 + fm];
    const float inv = dv > 0.0f ? 1.0f / dv : 0.0f;
    device VPIPE_ELT* orow =
        out + ((ulong)(h0 + band + fm) * n + row) * D + dc0;
    for (int i = 0; i < 8; ++i) {
      orow[i * 8 + fn] = (VPIPE_ELT)(of[i].thread_elements()[0] * inv);
      orow[i * 8 + fn + 1] = (VPIPE_ELT)(of[i].thread_elements()[1] * inv);
    }
  }
}

// Attention over each sparse row's selected tokens: out[h, r0 + r, :] =
// softmax(scale q . k_j) v over j in sel[r]. One simdgroup per (head, row);
// the scores in threadgroup memory (<= 2048 + ratio keys), then each lane
// its D / 32 output values. grid {32, Hq, rows}.
kernel void qsa_attn_f16(
    const device VPIPE_ELT* qt    [[buffer(0)]],   // [Hq, n, D]
    const device VPIPE_ELT* kp    [[buffer(1)]],
    const device VPIPE_ELT* vp    [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],   // [Hq, n, D]
    const device int*       pt    [[buffer(4)]],
    const device int*       sel   [[buffer(5)]],
    const device int*       nsel  [[buffer(6)]],
    constant float& scale        [[buffer(7)]],
    constant int& D              [[buffer(8)]],
    constant int& Hq             [[buffer(9)]],
    constant int& Hkv            [[buffer(10)]],
    constant int& n              [[buffer(11)]],
    constant int& page_tokens    [[buffer(12)]],
    constant int& n_pages        [[buffer(13)]],
    constant int& width          [[buffer(14)]],
    constant int& r0             [[buffer(15)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float qv[256];
  threadgroup float sc[2112];
  threadgroup uint koff[2112];   // each key's pool offset (K and V alike)
  const int h = (int)tg.y, r = (int)tg.z;
  const int row = r0 + r;
  const int kvh = h / (Hq / Hkv);
  const device VPIPE_ELT* qrow = qt + ((ulong)h * n + row) * D;
  for (int i = (int)lane; i < D; i += 32) { qv[i] = (float)qrow[i]; }
  const device int* srow = sel + (ulong)r * width;
  const int m = nsel[r];
  for (int j = (int)lane; j < m; j += 32) {
    koff[j] = (uint)pool_at_(pt, n_pages, srow[j], page_tokens, Hkv, D, kvh);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mx = -INFINITY;
  for (int j = (int)lane; j < m; j += 32) {
    const device VPIPE_ELT* kr = kp + koff[j];
    float dot = 0.0f;
    for (int i = 0; i < D; ++i) { dot += qv[i] * (float)kr[i]; }
    dot *= scale;
    sc[j] = dot;
    mx = metal::max(mx, dot);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (int j = (int)lane; j < m; j += 32) {
    const float e = metal::exp(sc[j] - mx);
    sc[j] = e;
    sum += e;
  }
  sum = simd_sum(sum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / sum;
  device VPIPE_ELT* orow = out + ((ulong)h * n + row) * D;
  for (int i = (int)lane; i < D; i += 32) {
    float acc = 0.0f;
    for (int j = 0; j < m; ++j) { acc += sc[j] * (float)vp[koff[j] + i]; }
    orow[i] = (VPIPE_ELT)(acc * inv);
  }
}
