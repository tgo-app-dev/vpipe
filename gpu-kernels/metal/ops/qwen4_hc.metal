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

}  // namespace

// y[m, s*H + h] = x[m, s*H + h] * rsqrt(mean_h x^2 + eps) * w[s*H + h].
// One threadgroup per (stream s, row m): grid {256, S, M}, tg {256, 1, 1}.
// x and y may alias.
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
  float ss = 0.0f;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    const float v = (float)x[base + h];
    ss += v * v;
  }
  const float tot = tg_sum_(ss, red, tid, sid, lane, (ntg + 31) / 32);
  const float r = metal::rsqrt(tot / (float)H + eps);
  const device float* ws = w + (ulong)tg.y * (ulong)H;
  for (int h = (int)tid; h < H; h += (int)ntg) {
    y[base + h] = (VPIPE_ELT)((float)x[base + h] * r * ws[h]);
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

// out[m, h] = (1/S) sum_s sigmoid(u[m, s*H + h]) * n[m, s*H + h]: the
// mixed sublayer input. grid {H, M}.
kernel void hc_mix_f16(
    const device VPIPE_ELT* u   [[buffer(0)]],   // up output [M, S*H]
    const device VPIPE_ELT* n   [[buffer(1)]],   // normed [M, S*H]
    device VPIPE_ELT*       out [[buffer(2)]],   // [M, H]
    constant int&           H   [[buffer(3)]],
    constant int&           S   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H) { return; }
  const ulong row = (ulong)gid.y * (ulong)S * (ulong)H;
  float acc = 0.0f;
  for (int s = 0; s < S; ++s) {
    const ulong i = row + (ulong)s * (ulong)H + gid.x;
    // sigmoid(up) is a bf16 / f16 tensor in the reference, as is its
    // product with the normed stream.
    const float mixw = (float)(VPIPE_ELT)sigmoid_((float)u[i]);
    acc += (float)(VPIPE_ELT)(mixw * (float)n[i]);
  }
  out[(ulong)gid.y * (ulong)H + gid.x] = (VPIPE_ELT)(acc / (float)S);
}

// X[m, s*H + h] += inj[m, s] * y[m, h]: the sublayer's output into every
// stream at its injection weight. grid {H, S, M}.
kernel void hc_inject_f16(
    device VPIPE_ELT*       X   [[buffer(0)]],   // [M, S*H]
    const device VPIPE_ELT* y   [[buffer(1)]],   // [M, H]
    const device float*     inj [[buffer(2)]],   // [M, S]
    constant int&           H   [[buffer(3)]],
    constant int&           S   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H) { return; }
  const ulong i = ((ulong)gid.z * (ulong)S + gid.y) * (ulong)H + gid.x;
  const float add = (float)(VPIPE_ELT)(inj[gid.z * (uint)S + gid.y] *
                                      (float)y[(ulong)gid.z * H + gid.x]);
  X[i] = (VPIPE_ELT)((float)X[i] + add);
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

// Top-k blocks of each sparse row (k = budget / ratio), by radix select on
// the scores' bits (all >= 0 or the -1 of an unseen block), ties at the
// threshold to the EARLIEST blocks; then the row's tokens: every selected
// block's ratio tokens in block order, then the tail the incomplete block
// holds (tokens nb * ratio .. p). sel [rows, k * ratio + ratio - 1],
// nsel [rows]. One threadgroup of 1024 per row.
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
    uint  ntg [[threads_per_threadgroup]])
{
  threadgroup atomic_uint hist[256];
  threadgroup uint shared_u[4];
  threadgroup uint pref[1024];
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
    if (tid == 0) {
      uint cum = 0, digit = 0;
      for (int dgt = 255; dgt >= 0; --dgt) {
        const uint hv = atomic_load_explicit(&hist[dgt], memory_order_relaxed);
        if (cum + hv >= want) { digit = (uint)dgt; break; }
        cum += hv;
      }
      shared_u[0] = digit;
      shared_u[1] = want - cum;   // still wanted inside this digit
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
  // Exclusive prefix sums of (ties seen) and (selected) over the threads.
  pref[tid] = neq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0;
    for (uint t = 0; t < ntg; ++t) {
      const uint v = pref[t];
      pref[t] = acc;
      acc += v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint eq_before = pref[tid];
  const uint take_eq = eq_before >= ties ? 0u : min(neq, ties - eq_before);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pref[tid] = ngt + take_eq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0;
    for (uint t = 0; t < ntg; ++t) {
      const uint v = pref[t];
      pref[t] = acc;
      acc += v;
    }
    shared_u[2] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint o = pref[tid];
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
    const int nblk = (int)shared_u[2];
    int c = nblk * ratio;
    for (int t = nb * ratio; t <= p; ++t) { out[c++] = t; }
    nsel[r] = c;
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
