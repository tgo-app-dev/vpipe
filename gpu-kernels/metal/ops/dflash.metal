// dflash.metal -- the DFlash / DFlash 2 block-diffusion drafter
// (speculative decoding; generative-models/shared/dflash-drafter.h).
//
// The drafter's GEMMs ride the shared affine / dense kernels and its
// norms the shared rms_norm; what lives here is what is DFlash's own:
//
//   dflash_conv_f16      the grouped dynamic causal convolution DFlash 2
//                        puts before and after every sublayer
//   dflash_qk_prep_f16   per-head RMSNorm + rotate-half RoPE, in place
//   dflash_kv_store_f16  context K/V into a layer's position-tagged ring
//   dflash_attn_f16      the block's queries over [ring context ; block]
//   dflash_topk_*        top-16 of a logit row, two stages
//   dflash_select_f16    DFlash 2's candidate-path selector walk
//   dflash_copy_rows_f16 a strided row-block copy (taps -> concat)
//
// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the
// bf16 metallib. Math is f32 throughout.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// ---------------------------------------------------------------------
// Grouped dynamic causal convolution along the block (DFlash 2):
//   out[t, c] = sum_{o < ks} (base[o, c] + dyn[t, o * G + c / gsz])
//                            * x[t - o, c]
// with x[t - o] = 0 before the block's first row. `dyn` is the
// kernel_projection output [L, 2 * ks * G] read at row stride
// dyn_stride; dyn_off selects which half (0 before the sublayer,
// ks * G after it). `base` is that half's [ks, C] slice of base_kernel.
// out must not alias x (row t reads row t - 1).
//   0:x[L,C] 1:dyn 2:base[ks,C] 3:out[L,C]
//   4:C 5:L 6:ks 7:gsz 8:dyn_stride 9:dyn_off      grid {C, L}
kernel void dflash_conv_f16(
    const device VPIPE_ELT* x          [[buffer(0)]],
    const device VPIPE_ELT* dyn        [[buffer(1)]],
    const device VPIPE_ELT* base       [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant int&           C          [[buffer(4)]],
    constant int&           L          [[buffer(5)]],
    constant int&           ks         [[buffer(6)]],
    constant int&           gsz        [[buffer(7)]],
    constant int&           dyn_stride [[buffer(8)]],
    constant int&           dyn_off    [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int t = (int)gid.y;
  if (c >= C || t >= L) { return; }
  const int G = C / gsz;
  const int g = c / gsz;
  const device VPIPE_ELT* d = dyn + (size_t)t * dyn_stride + dyn_off;
  float acc = 0.0f;
  for (int o = 0; o < ks && o <= t; ++o) {
    const float kw = (float)base[(size_t)o * C + c] + (float)d[o * G + g];
    acc += kw * (float)x[(size_t)(t - o) * C + c];
  }
  out[(size_t)t * C + c] = (VPIPE_ELT)acc;
}

// ---------------------------------------------------------------------
// Per-head RMSNorm (weighted) then rotate-half RoPE over the FULL head,
// in place, on rows [T] of heads [H] at row stride `ld` (elements):
//   head (t, h) = x[t * ld + h * D .. + D), position pos0 + t.
// Mirrors HF Qwen3: the normalized head is rounded to the element type
// before the weight multiply, and cos/sin are rounded to it too (the
// reference casts both to the activation dtype).
//   0:x 1:w[D] 2:inv_freq[D/2] (f32) 3:H 4:D 5:eps 6:pos0 7:ld
//   threadgroups {1, T*H}, tg {32, 1}   (D % 64 == 0, D <= 256)
kernel void dflash_qk_prep_f16(
    device VPIPE_ELT*       x        [[buffer(0)]],
    const device VPIPE_ELT* w        [[buffer(1)]],
    const device float*     inv_freq [[buffer(2)]],
    constant int&           H        [[buffer(3)]],
    constant int&           D        [[buffer(4)]],
    constant float&         eps      [[buffer(5)]],
    constant int&           pos0     [[buffer(6)]],
    constant int&           ld       [[buffer(7)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int row = (int)tg.y;
  const int t = row / H;
  const int h = row % H;
  device VPIPE_ELT* xr = x + (size_t)t * ld + (size_t)h * D;
  const int half_d = D / 2;
  const int np = half_d / 32;          // pairs per lane (<= 4)
  float a[4], b[4];
  float ss = 0.0f;
  for (int j = 0; j < np; ++j) {
    const int i = (int)lane + 32 * j;
    a[j] = (float)xr[i];
    b[j] = (float)xr[i + half_d];
    ss += a[j] * a[j] + b[j] * b[j];
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / (float)D + eps);
  const float pos = (float)(pos0 + t);
  for (int j = 0; j < np; ++j) {
    const int i = (int)lane + 32 * j;
    const float x1 = (float)(VPIPE_ELT)((float)(VPIPE_ELT)(a[j] * inv) *
                                        (float)w[i]);
    const float x2 = (float)(VPIPE_ELT)((float)(VPIPE_ELT)(b[j] * inv) *
                                        (float)w[i + half_d]);
    const float ang = pos * inv_freq[i];
    const float c = (float)(VPIPE_ELT)precise::cos(ang);
    const float s = (float)(VPIPE_ELT)precise::sin(ang);
    xr[i]          = (VPIPE_ELT)(x1 * c - x2 * s);
    xr[i + half_d] = (VPIPE_ELT)(x2 * c + x1 * s);
  }
}

// ---------------------------------------------------------------------
// Store m context rows' K (already normed + roped) and V into one layer's
// cache [Hkv, cap, D] at slot (pos % cap), tagging the slot with its
// absolute position. k/v rows are read at row stride `ld`.
//   0:k 1:v 2:kc 3:vc 4:tags[cap] (int)
//   5:Hkv 6:D 7:cap 8:pos0 9:ld        grid {D, m, Hkv}
kernel void dflash_kv_store_f16(
    const device VPIPE_ELT* k    [[buffer(0)]],
    const device VPIPE_ELT* v    [[buffer(1)]],
    device VPIPE_ELT*       kc   [[buffer(2)]],
    device VPIPE_ELT*       vc   [[buffer(3)]],
    device int*             tags [[buffer(4)]],
    constant int&           Hkv  [[buffer(5)]],
    constant int&           D    [[buffer(6)]],
    constant int&           cap  [[buffer(7)]],
    constant int&           pos0 [[buffer(8)]],
    constant int&           ld   [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (d >= D || h >= Hkv) { return; }
  const int pos = pos0 + t;
  const int slot = pos % cap;
  const size_t src = (size_t)t * ld + (size_t)h * D + d;
  const size_t dst = ((size_t)h * cap + slot) * D + d;
  kc[dst] = k[src];
  vc[dst] = v[src];
  if (d == 0 && h == 0) { tags[slot] = pos; }
}

// ---------------------------------------------------------------------
// The block's attention: query row i (absolute position p0 + i) over
//   * the context cache: slot s is visible iff its tag is a position
//     before the block (0 <= tag < p0) and, for a sliding layer, within
//     the window of the query (p0 + i - tag < window);
//   * the block's own L keys: all of them (DFlash's non-causal block), or
//     j <= i when `causal` (DFlash v1's sliding layers).
// One threadgroup per (query row, kv head); one simdgroup per query head
// of that group, each lane holding D/32 dims. Online softmax in f32.
// q / kb / vb rows are read at row stride `ld` (the fused q|k|v of the
// block); out is [L, Hq * D].
//   0:q 1:kb 2:vb 3:kc[Hkv,cap,D] 4:vc 5:tags[cap] 6:out
//   7:L 8:Hq 9:Hkv 10:D 11:cap 12:p0 13:window 14:causal 15:scale 16:ld
//   threadgroups {L, Hkv}, tg {32, Hq/Hkv}          (D % 32 == 0, <= 256)
kernel void dflash_attn_f16(
    const device VPIPE_ELT* q      [[buffer(0)]],
    const device VPIPE_ELT* kb     [[buffer(1)]],
    const device VPIPE_ELT* vb     [[buffer(2)]],
    const device VPIPE_ELT* kc     [[buffer(3)]],
    const device VPIPE_ELT* vc     [[buffer(4)]],
    const device int*       tags   [[buffer(5)]],
    device VPIPE_ELT*       out    [[buffer(6)]],
    constant int&           L      [[buffer(7)]],
    constant int&           Hq     [[buffer(8)]],
    constant int&           Hkv    [[buffer(9)]],
    constant int&           D      [[buffer(10)]],
    constant int&           cap    [[buffer(11)]],
    constant int&           p0     [[buffer(12)]],
    constant int&           window [[buffer(13)]],
    constant int&           causal [[buffer(14)]],
    constant float&         scale  [[buffer(15)]],
    constant int&           ld     [[buffer(16)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int i = (int)tg.x;
  const int hk = (int)tg.y;
  const int G = Hq / Hkv;
  const int h = hk * G + (int)sg;
  const int npl = D / 32;              // dims per lane (<= 8)
  const int d0 = (int)lane * npl;
  float qv[8], acc[8];
  const device VPIPE_ELT* qr = q + (size_t)i * ld + (size_t)h * D + d0;
  for (int j = 0; j < npl; ++j) {
    qv[j] = (float)qr[j] * scale;
    acc[j] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const int qpos = p0 + i;
  for (int s = 0; s < cap; ++s) {
    const int tag = tags[s];
    if (tag < 0 || tag >= p0) { continue; }
    if (window > 0 && qpos - tag >= window) { continue; }
    const size_t off = ((size_t)hk * cap + s) * D + d0;
    float dot = 0.0f;
    for (int j = 0; j < npl; ++j) { dot += qv[j] * (float)kc[off + j]; }
    dot = simd_sum(dot);
    const float mn = max(m, dot);
    const float corr = exp(m - mn);
    const float p = exp(dot - mn);
    l = l * corr + p;
    for (int j = 0; j < npl; ++j) {
      acc[j] = acc[j] * corr + p * (float)vc[off + j];
    }
    m = mn;
  }
  for (int jb = 0; jb < L; ++jb) {
    if (causal != 0 && jb > i) { break; }
    const size_t off = (size_t)jb * ld + (size_t)hk * D + d0;
    float dot = 0.0f;
    for (int j = 0; j < npl; ++j) { dot += qv[j] * (float)kb[off + j]; }
    dot = simd_sum(dot);
    const float mn = max(m, dot);
    const float corr = exp(m - mn);
    const float p = exp(dot - mn);
    l = l * corr + p;
    for (int j = 0; j < npl; ++j) {
      acc[j] = acc[j] * corr + p * (float)vb[off + j];
    }
    m = mn;
  }
  const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
  device VPIPE_ELT* o = out + (size_t)i * Hq * D + (size_t)h * D + d0;
  for (int j = 0; j < npl; ++j) { o[j] = (VPIPE_ELT)(acc[j] * inv); }
}

// ---------------------------------------------------------------------
// Top-16 of each logit row, two stages. Order: value descending, then
// index ascending, so equal logits resolve the same way every run.
#define DFLASH_TOPK 16
#define DFLASH_TOPK_TG 256

// True iff (va, ia) ranks before (vb, ib).
static inline bool dflash_before(float va, int ia, float vb, int ib)
{
  return va > vb || (va == vb && ia >= 0 && (ib < 0 || ia < ib));
}

// Insert (v, i) into a thread's sorted list, dropping the last.
static inline void dflash_insert(thread float* val, thread int* idx,
                                 float v, int i)
{
  if (!dflash_before(v, i, val[DFLASH_TOPK - 1], idx[DFLASH_TOPK - 1])) {
    return;
  }
  int p = DFLASH_TOPK - 1;
  while (p > 0 && dflash_before(v, i, val[p - 1], idx[p - 1])) {
    val[p] = val[p - 1];
    idx[p] = idx[p - 1];
    --p;
  }
  val[p] = v;
  idx[p] = i;
}

// Pop the threadgroup's best 16 out of every thread's sorted list.
// `head` is this thread's cursor into its list.
static inline void dflash_tg_merge(thread float* val, thread int* idx,
                                   device float* out_v, device int* out_i,
                                   threadgroup float* sv,
                                   threadgroup int* si,
                                   threadgroup int* swin,
                                   uint tid, uint sg, uint lane)
{
  int head = 0;
  const uint nsg = DFLASH_TOPK_TG / 32;
  for (int k = 0; k < DFLASH_TOPK; ++k) {
    const float v = head < DFLASH_TOPK ? val[head] : -INFINITY;
    const int i = head < DFLASH_TOPK ? idx[head] : -1;
    // Simdgroup best: max value, then the lowest index among equals.
    const float bv = simd_max(v);
    const int bi = simd_min((v == bv && i >= 0) ? i : INT_MAX);
    if (lane == 0) {
      sv[sg] = bv;
      si[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float wv = sv[0];
      int wi = si[0];
      for (uint g = 1; g < nsg; ++g) {
        const int gi = si[g] == INT_MAX ? -1 : si[g];
        const int ci = wi == INT_MAX ? -1 : wi;
        if (dflash_before(sv[g], gi, wv, ci)) {
          wv = sv[g];
          wi = si[g];
        }
      }
      out_v[k] = wv;
      out_i[k] = wi == INT_MAX ? -1 : wi;
      swin[0] = wi;
      sv[0] = wv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The owner of the winning entry advances (indices are unique).
    if (head < DFLASH_TOPK && i == swin[0] && v == sv[0]) { ++head; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Stage 1: the top-16 of logits[row, part * chunk ..) into the part's
// slot of pval/pidx [R, P, 16].
//   0:logits[R,V] 1:pval 2:pidx 3:V 4:chunk 5:P
//   threadgroups {P, R}, tg {256}
kernel void dflash_topk_part_f16(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device float*           pval   [[buffer(1)]],
    device int*             pidx   [[buffer(2)]],
    constant int&           V      [[buffer(3)]],
    constant int&           chunk  [[buffer(4)]],
    constant int&           P      [[buffer(5)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float sv[DFLASH_TOPK_TG / 32];
  threadgroup int si[DFLASH_TOPK_TG / 32];
  threadgroup int swin[1];
  const int part = (int)tg.x;
  const int row = (int)tg.y;
  float val[DFLASH_TOPK];
  int idx[DFLASH_TOPK];
  for (int k = 0; k < DFLASH_TOPK; ++k) { val[k] = -INFINITY; idx[k] = -1; }
  const int b = part * chunk;
  const int e = min(V, b + chunk);
  const device VPIPE_ELT* lr = logits + (size_t)row * V;
  for (int j = b + (int)tid; j < e; j += DFLASH_TOPK_TG) {
    dflash_insert(val, idx, (float)lr[j], j);
  }
  const size_t o = ((size_t)row * P + part) * DFLASH_TOPK;
  dflash_tg_merge(val, idx, pval + o, pidx + o, sv, si, swin, tid, sg, lane);
}

// Stage 2: the row's top-16 out of its P partial lists.
//   0:pval 1:pidx 2:val[R,16] 3:idx[R,16] 4:P   threadgroups {R}, tg {256}
kernel void dflash_topk_merge_f16(
    const device float* pval [[buffer(0)]],
    const device int*   pidx [[buffer(1)]],
    device float*       oval [[buffer(2)]],
    device int*         oidx [[buffer(3)]],
    constant int&       P    [[buffer(4)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  threadgroup float sv[DFLASH_TOPK_TG / 32];
  threadgroup int si[DFLASH_TOPK_TG / 32];
  threadgroup int swin[1];
  float val[DFLASH_TOPK];
  int idx[DFLASH_TOPK];
  for (int k = 0; k < DFLASH_TOPK; ++k) { val[k] = -INFINITY; idx[k] = -1; }
  const int n = P * DFLASH_TOPK;
  const device float* pv = pval + (size_t)row * n;
  const device int* pi = pidx + (size_t)row * n;
  for (int j = (int)tid; j < n; j += DFLASH_TOPK_TG) {
    if (pi[j] >= 0) { dflash_insert(val, idx, pv[j], pi[j]); }
  }
  dflash_tg_merge(val, idx, oval + (size_t)row * DFLASH_TOPK,
                  oidx + (size_t)row * DFLASH_TOPK, sv, si, swin, tid, sg,
                  lane);
}

// ---------------------------------------------------------------------
// DFlash 2's candidate-path selector, greedy:
//   score_p(b) = U_p(b) + < pred_cb[a] * hp[p], succ_cb[b] >
// over the 16 candidates b of position p, a the token chosen at p - 1
// (the anchor at p = 0); the chosen b becomes the next a. One
// threadgroup of `rank` threads, one rank dim each; the R positions are
// walked in order. Ids land at out[out_off ..), the verify's input row.
//   0:cval[R,16] (f32) 1:cidx[R,16] 2:hp[R,rank] 3:pred_cb[V,rank]
//   4:succ_cb[V,rank] 5:anchor (int, [0]) 6:out (int)
//   7:R 8:rank 9:out_off            threadgroups {1}, tg {rank}
kernel void dflash_select_f16(
    const device float*     cval    [[buffer(0)]],
    const device int*       cidx    [[buffer(1)]],
    const device VPIPE_ELT* hp      [[buffer(2)]],
    const device VPIPE_ELT* pred_cb [[buffer(3)]],
    const device VPIPE_ELT* succ_cb [[buffer(4)]],
    const device int*       anchor  [[buffer(5)]],
    device int*             out     [[buffer(6)]],
    constant int&           R       [[buffer(7)]],
    constant int&           rank    [[buffer(8)]],
    constant int&           out_off [[buffer(9)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float part[32][DFLASH_TOPK];
  threadgroup float score[DFLASH_TOPK];
  threadgroup int pred_sh[1];
  const uint nsg = ((uint)rank + 31) / 32;
  if (tid == 0) { pred_sh[0] = anchor[0]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = 0; p < R; ++p) {
    const int a = pred_sh[0];
    const float wr = (float)pred_cb[(size_t)a * rank + tid] *
                     (float)hp[(size_t)p * rank + tid];
    for (int k = 0; k < DFLASH_TOPK; ++k) {
      const int b = cidx[p * DFLASH_TOPK + k];
      float v = b >= 0 ? wr * (float)succ_cb[(size_t)b * rank + tid] : 0.0f;
      v = simd_sum(v);
      if (lane == 0) { part[sg][k] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < DFLASH_TOPK) {
      float s = 0.0f;
      for (uint g = 0; g < nsg; ++g) { s += part[g][tid]; }
      const int b = cidx[p * DFLASH_TOPK + tid];
      score[tid] = b >= 0 ? cval[p * DFLASH_TOPK + tid] + s : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      int best = 0;
      for (int k = 1; k < DFLASH_TOPK; ++k) {
        if (score[k] > score[best]) { best = k; }
      }
      const int b = cidx[p * DFLASH_TOPK + best];
      pred_sh[0] = b;
      out[out_off + p] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---------------------------------------------------------------------
// dst[r * dst_ld + c] = src[r * src_ld + c] for c < cols: a block of
// rows from one row-major buffer into another of a different width.
// Column offsets ride the buffer binding offsets.
//   0:src 1:dst 2:cols 3:src_ld 4:dst_ld        grid {cols, rows}
kernel void dflash_copy_rows_f16(
    const device VPIPE_ELT* src    [[buffer(0)]],
    device VPIPE_ELT*       dst    [[buffer(1)]],
    constant int&           cols   [[buffer(2)]],
    constant int&           src_ld [[buffer(3)]],
    constant int&           dst_ld [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= cols) { return; }
  dst[(size_t)gid.y * dst_ld + gid.x] = src[(size_t)gid.y * src_ld + gid.x];
}

// ---------------------------------------------------------------------
// Greedy argmax of each of R logit rows into out[out_off + r] (DFlash v1:
// the drafts are the drafter's own per-position argmax). Lowest index on
// a tie, matching torch.argmax.
//   0:logits[R,V] 1:out (int) 2:V 3:out_off      threadgroups {R}, tg {256}
kernel void dflash_argmax_f16(
    const device VPIPE_ELT* logits  [[buffer(0)]],
    device int*             out     [[buffer(1)]],
    constant int&           V       [[buffer(2)]],
    constant int&           out_off [[buffer(3)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float sv[8];
  threadgroup int si[8];
  const device VPIPE_ELT* lr = logits + (size_t)row * V;
  float bv = -INFINITY;
  int bi = INT_MAX;
  for (int j = (int)tid; j < V; j += 256) {
    const float v = (float)lr[j];
    if (v > bv) { bv = v; bi = j; }
  }
  const float m = simd_max(bv);
  const int mi = simd_min(bv == m ? bi : INT_MAX);
  if (lane == 0) { sv[sg] = m; si[sg] = mi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float wv = sv[0];
    int wi = si[0];
    for (int g = 1; g < 8; ++g) {
      if (sv[g] > wv || (sv[g] == wv && si[g] < wi)) { wv = sv[g]; wi = si[g]; }
    }
    out[out_off + (int)row] = wi;
  }
}

// ---------------------------------------------------------------------
// The drafter computes in bf16 whatever the target's dtype (a drafter's
// residual stream can pass f16's range: DFlash 2's for Qwen3.8-27B runs
// to tens of thousands). These cross the boundary to an f16 target, with
// explicit element types, so either metallib's copy serves.
//   dflash_h2b / dflash_b2h   0:x 1:y 2:n            grid {n}
//   dflash_copy_rows_h2b      as dflash_copy_rows, half in, bfloat out
kernel void dflash_h2b(
    const device half* x [[buffer(0)]],
    device bfloat*     y [[buffer(1)]],
    constant int&      n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid < n) { y[gid] = bfloat(float(x[gid])); }
}

kernel void dflash_b2h(
    const device bfloat* x [[buffer(0)]],
    device half*         y [[buffer(1)]],
    constant int&        n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  // Saturate rather than overflow: the head's input is a normed hidden,
  // well inside half's range, but an inf would poison every logit.
  if ((int)gid < n) {
    y[gid] = half(clamp(float(x[gid]), -65504.0f, 65504.0f));
  }
}

kernel void dflash_copy_rows_h2b(
    const device half* src    [[buffer(0)]],
    device bfloat*     dst    [[buffer(1)]],
    constant int&      cols   [[buffer(2)]],
    constant int&      src_ld [[buffer(3)]],
    constant int&      dst_ld [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= cols) { return; }
  dst[(size_t)gid.y * dst_ld + gid.x] =
      bfloat(float(src[(size_t)gid.y * src_ld + gid.x]));
}
