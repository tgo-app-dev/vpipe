// deepseek_v41.metal -- DeepSeek-V4.1-Flash's own operations
// (generative-models/deepseek-v41/). The reference is the release's
// inference/model.py and kernel.py; each kernel says what it restates.
//
// Element type: half by default, -DVPIPE_ELT=bfloat for the bf16 library.
// Arithmetic is f32 throughout; values are rounded to the element type
// where the reference stores them in its own (bf16) dtype.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
typedef VPIPE_ELT T;

// ---- hyper-connections ------------------------------------------------

// The embedding row of each id, copied into each of the HC residual copies.
// grid {H, n}.
kernel void dsv41_embed_hc(
    const device T*   table [[buffer(0)]],    // [V, H]
    const device int* ids   [[buffer(1)]],    // [n]
    device T*         X     [[buffer(2)]],    // [n, HC, H]
    constant int&     H     [[buffer(3)]],
    constant int&     HC    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, i = (int)gid.y;
  if (d >= H) { return; }
  const T v = table[(long)ids[i] * H + d];
  for (int c = 0; c < HC; ++c) { X[((long)i * HC + c) * H + d] = v; }
}

// Rows already embedded (an image's), copied into each HC copy.
// grid {H, n}.
kernel void dsv41_rows_hc(
    const device T* rows [[buffer(0)]],       // [n, H]
    device T*       X    [[buffer(1)]],       // [n, HC, H]
    constant int&   H    [[buffer(2)]],
    constant int&   HC   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, i = (int)gid.y;
  if (d >= H) { return; }
  const T v = rows[(long)i * H + d];
  for (int c = 0; c < HC; ++c) { X[((long)i * HC + c) * H + d] = v; }
}

// Block.hc_mixes + kernel.hc_split_sinkhorn: the flattened HC*H stream
// normalized by its rms (one statistic a row), its 24 projections onto
// hc_fn, then pre / post / comb -- pre = sigmoid + eps, post = 2 sigmoid,
// comb = row softmax + eps, column-normalized, then iters - 1 rounds of
// row and column normalization (Sinkhorn). Two kernels: the sums over S
// slices of the stream, each a threadgroup (so a row is not one core's
// work), then the rest a row at a time, the slices added in order.
constant constexpr int kMix = 24;             // (2 + hc) * hc, hc = 4

// The slices: part[row][s] = [24 projections | sum of squares] over
// stream elements [s * SK, (s + 1) * SK). A slice's sums come back in
// threads 0..24 (the rest return 0).
inline float dsv41_hc_mix_slice_(
    const device T* X, const device float* fnT, const int K, const int S,
    const int sl, const int row, const uint tid, const uint sg,
    const uint lane, threadgroup float (*red)[kMix + 1])
{
  const int SK = (K + S - 1) / S;
  const int k0 = sl * SK, k1 = min(K, k0 + SK);
  // six float4 accumulators, every index a constant: a loop-indexed
  // register array spills to thread memory
  float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f;
  float ss = 0.0f;
  const device T* x = X + (long)row * K;
  for (int k = k0 + (int)tid; k < k1; k += 256) {
    const float v = (float)x[k];
    ss += v * v;
    const device float4* w =
        reinterpret_cast<const device float4*>(fnT + (long)k * kMix);
    a0 += v * w[0];
    a1 += v * w[1];
    a2 += v * w[2];
    a3 += v * w[3];
    a4 += v * w[4];
    a5 += v * w[5];
  }
  a0 = float4(simd_sum(a0.x), simd_sum(a0.y), simd_sum(a0.z), simd_sum(a0.w));
  a1 = float4(simd_sum(a1.x), simd_sum(a1.y), simd_sum(a1.z), simd_sum(a1.w));
  a2 = float4(simd_sum(a2.x), simd_sum(a2.y), simd_sum(a2.z), simd_sum(a2.w));
  a3 = float4(simd_sum(a3.x), simd_sum(a3.y), simd_sum(a3.z), simd_sum(a3.w));
  a4 = float4(simd_sum(a4.x), simd_sum(a4.y), simd_sum(a4.z), simd_sum(a4.w));
  a5 = float4(simd_sum(a5.x), simd_sum(a5.y), simd_sum(a5.z), simd_sum(a5.w));
  ss = simd_sum(ss);
  if (lane == 0) {
    threadgroup float4* r4 = reinterpret_cast<threadgroup float4*>(red[sg]);
    r4[0] = a0;
    r4[1] = a1;
    r4[2] = a2;
    r4[3] = a3;
    r4[4] = a4;
    r4[5] = a5;
    red[sg][kMix] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  if (tid <= (uint)kMix) {
    for (int g = 0; g < 8; ++g) { t += red[g][tid]; }
  }
  return t;
}

// grid {256 * S, n}, tg {256}.
kernel void dsv41_hc_mix_part(
    const device T*     X    [[buffer(0)]],   // [n, K]
    const device float* fnT  [[buffer(1)]],   // [K, 24]
    device float*       part [[buffer(2)]],   // [n, S, 25]
    constant int&       K    [[buffer(3)]],
    constant int&       S    [[buffer(4)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const uint tid = tid2.x;
  const int sl = (int)tg.x, row = (int)tg.y;
  threadgroup float red[8][kMix + 1];
  const float t =
      dsv41_hc_mix_slice_(X, fnT, K, S, sl, row, tid, sg, lane, red);
  if (tid <= (uint)kMix) {
    part[((long)row * S + sl) * (kMix + 1) + tid] = t;
  }
}

// The rest, one simdgroup a row: lane j sums projection j over the slices
// (lane 24 the squares; `part(g, j)` slice g's sum j), lanes 0..7 make pre
// / post, and lanes 0..15 hold comb[j][k] (lane 4j + k) through the
// Sinkhorn rounds -- a row's sums across a quad (xor 1, 2), a column's
// across the quads (xor 4, 8).
template <typename Part>
METAL_FUNC void dsv41_hc_mix_finish_(
    Part part, const device float* hscale, const device float* hbase,
    device float* pre, device float* post, device float* comb, const int K,
    const float eps, const float hc_eps, const int iters, const int S,
    const uint row, const uint lane)
{
  float t = 0.0f;
  if (lane <= (uint)kMix) {
    for (int g = 0; g < S; ++g) { t += part(g, (int)lane); }
  }
  const float s2 = simd_shuffle(t, (ushort)kMix);
  const float r = 1.0f / sqrt(s2 / (float)K + eps);
  const float m = t * r;                      // lane j: projection j
  if (lane < 4) {
    pre[row * 4 + lane] =
        1.0f / (1.0f + exp(-(m * hscale[0] + hbase[lane]))) + hc_eps;
  } else if (lane < 8) {
    post[row * 4 + lane - 4] =
        2.0f / (1.0f + exp(-(m * hscale[1] + hbase[lane])));
  }
  // comb's 16 to lanes 0..15 (the lanes past them carry what they carry:
  // the shuffles below never mix them in)
  float c = simd_shuffle(m, (ushort)((lane + 8) & 31u));
  c = c * hscale[2] + hbase[8 + (lane & 15u)];
  // a row's softmax over its quad, + eps
  float mx = max(c, simd_shuffle_xor(c, (ushort)1));
  mx = max(mx, simd_shuffle_xor(mx, (ushort)2));
  float e = exp(c - mx);
  float sm = e + simd_shuffle_xor(e, (ushort)1);
  sm = sm + simd_shuffle_xor(sm, (ushort)2);
  c = e / sm + hc_eps;
  // then a column normalization, then iters - 1 rounds of row and column
  float cs = c + simd_shuffle_xor(c, (ushort)4);
  cs = cs + simd_shuffle_xor(cs, (ushort)8);
  c = c / (cs + hc_eps);
  for (int it = 0; it < iters - 1; ++it) {
    float rs = c + simd_shuffle_xor(c, (ushort)1);
    rs = rs + simd_shuffle_xor(rs, (ushort)2);
    c = c / (rs + hc_eps);
    float cc = c + simd_shuffle_xor(c, (ushort)4);
    cc = cc + simd_shuffle_xor(cc, (ushort)8);
    c = c / (cc + hc_eps);
  }
  if (lane < 16) { comb[row * 16 + lane] = c; }
}

// grid {32 * n}, tg {32}.
kernel void dsv41_hc_mix(
    const device float* part   [[buffer(0)]],   // [n, S, 25]
    const device float* hscale [[buffer(1)]],   // [3]
    const device float* hbase  [[buffer(2)]],   // [24]
    device float*       pre    [[buffer(3)]],   // [n, 4]
    device float*       post   [[buffer(4)]],   // [n, 4]
    device float*       comb   [[buffer(5)]],   // [n, 16], [j][k]
    constant int&       K      [[buffer(6)]],
    constant float&     eps    [[buffer(7)]],
    constant float&     hc_eps [[buffer(8)]],
    constant int&       iters  [[buffer(9)]],
    constant int&       S      [[buffer(10)]],
    constant int&       n      [[buffer(11)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
  if ((int)row >= n) { return; }
  const device float* pr = part + (long)row * S * (kMix + 1);
  dsv41_hc_mix_finish_(
      [&](int g, int j) { return pr[g * (kMix + 1) + j]; }, hscale, hbase,
      pre, post, comb, K, eps, hc_eps, iters, S, row, lane);
}

// Block.hc_pre then RMSNorm: y = norm(round(sum_c pre[c] * X[c])), the
// collapse rounded to the element type as the reference stores it, the
// norm in f32 (weight * x * rsqrt(mean x^2 + eps)). With has_w 0 the
// collapse alone. One row a threadgroup of 1024, a thread's (at most 8)
// values kept in registers between the two passes; H <= 8192.
// grid {1024 * n}, tg {1024}.
kernel void dsv41_hc_pre_norm(
    const device T*     X     [[buffer(0)]],    // [n, HC, H]
    const device float* mix   [[buffer(1)]],    // [n, HC]
    const device T*     w     [[buffer(2)]],    // [H]
    device T*           y     [[buffer(3)]],    // [n, H]
    constant int&       H     [[buffer(4)]],
    constant float&     eps   [[buffer(5)]],
    constant int&       has_w [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  const device T* x = X + (long)row * 4 * H;
  const float m0 = mix[row * 4], m1 = mix[row * 4 + 1];
  const float m2 = mix[row * 4 + 2], m3 = mix[row * 4 + 3];
  device T* yr = y + (long)row * H;
  float vv[8];
  float ss = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const int d = (int)tid + i * 1024;
    vv[i] = 0.0f;
    if (d < H) {
      const float s = m0 * (float)x[d] + m1 * (float)x[H + d] +
                      m2 * (float)x[2 * H + d] + m3 * (float)x[3 * H + d];
      const T v = (T)s;
      vv[i] = (float)v;
      ss += vv[i] * vv[i];
      if (has_w == 0) { yr[d] = v; }
    }
  }
  if (has_w == 0) { return; }
  threadgroup float red[32];
  ss = simd_sum(ss);
  if (lane == 0) { red[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < 32; ++g) { tot += red[g]; }
  const float r = 1.0f / sqrt(tot / (float)H + eps);
  for (int i = 0; i < 8; ++i) {
    const int d = (int)tid + i * 1024;
    if (d < H) { yr[d] = (T)((float)w[d] * (vv[i] * r)); }
  }
}

// Block.hc_post, in place: X'[k] = post[k] * a + sum_j comb[j][k] * X[j].
// grid {H, n}.
kernel void dsv41_hc_post(
    const device T*     a    [[buffer(0)]],     // [n, H]
    device T*           X    [[buffer(1)]],     // [n, 4, H]
    const device float* post [[buffer(2)]],     // [n, 4]
    const device float* comb [[buffer(3)]],     // [n, 16]
    constant int&       H    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, i = (int)gid.y;
  if (d >= H) { return; }
  const float av = (float)a[(long)i * H + d];
  device T* xr = X + (long)i * 4 * H + d;
  const float r0 = (float)xr[0], r1 = (float)xr[H];
  const float r2 = (float)xr[2 * H], r3 = (float)xr[3 * H];
  const device float* cm = comb + i * 16;
  for (int k = 0; k < 4; ++k) {
    const float s = cm[k] * r0 + cm[4 + k] * r1 + cm[8 + k] * r2 +
                    cm[12 + k] * r3;
    xr[k * H] = (T)(post[i * 4 + k] * av + s);
  }
}

// dsv41_hc_post then dsv41_hc_pre_norm in one pass: a sublayer's branch `a`
// into the copies with its mix's post and comb (X written back, as
// dsv41_hc_post stores it), and those stored copies collapsed with the
// same mix's pre and normed -- the next sublayer's input. Bit-identical to
// the two. `K` > 0: the branch is the MoE's, summed here from its K
// routed parts and the shared expert's output as dsv41_moe_combine sums
// it (f32, rounded once) -- the combine's dispatch folded in.
// grid {1024 * n}, tg {1024}.
inline void dsv41_hc_post_pre_impl_(
    const device T* a, const device T* part, const device T* sh,
    const int K, device T* X, const device float* pre,
    const device float* post, const device float* comb,
    const device T* w, device T* y, const int H, const float eps,
    const uint row, const uint tid, const uint sg, const uint lane,
    threadgroup float* red)
{
  device T* x = X + (long)row * 4 * H;
  const device float* cm = comb + row * 16;
  const float m0 = pre[row * 4], m1 = pre[row * 4 + 1];
  const float m2 = pre[row * 4 + 2], m3 = pre[row * 4 + 3];
  device T* yr = y + (long)row * H;
  float vv[8];
  float ss = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const int d = (int)tid + i * 1024;
    vv[i] = 0.0f;
    if (d < H) {
      float av;
      if (K > 0) {
        float s = 0.0f;
        for (int j = 0; j < K; ++j) {
          s += (float)part[((long)row * K + j) * H + d];
        }
        s += (float)sh[(long)row * H + d];
        av = (float)(T)s;
      } else {
        av = (float)a[(long)row * H + d];
      }
      const float r0 = (float)x[d], r1 = (float)x[H + d];
      const float r2 = (float)x[2 * H + d], r3 = (float)x[3 * H + d];
      float nx[4];
      for (int k = 0; k < 4; ++k) {
        const float s = cm[k] * r0 + cm[4 + k] * r1 + cm[8 + k] * r2 +
                        cm[12 + k] * r3;
        const T v = (T)(post[row * 4 + k] * av + s);
        x[k * H + d] = v;
        nx[k] = (float)v;
      }
      const float s = m0 * nx[0] + m1 * nx[1] + m2 * nx[2] + m3 * nx[3];
      vv[i] = (float)(T)s;
      ss += vv[i] * vv[i];
    }
  }
  ss = simd_sum(ss);
  if (lane == 0) { red[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < 32; ++g) { tot += red[g]; }
  const float r = 1.0f / sqrt(tot / (float)H + eps);
  for (int i = 0; i < 8; ++i) {
    const int d = (int)tid + i * 1024;
    if (d < H) { yr[d] = (T)((float)w[d] * (vv[i] * r)); }
  }
}

kernel void dsv41_hc_post_pre(
    const device T*     a     [[buffer(0)]],    // [n, H]
    device T*           X     [[buffer(1)]],    // [n, 4, H]
    const device float* pre   [[buffer(2)]],    // [n, 4]
    const device float* post  [[buffer(3)]],    // [n, 4]
    const device float* comb  [[buffer(4)]],    // [n, 16]
    const device T*     w     [[buffer(5)]],    // [H]
    device T*           y     [[buffer(6)]],    // [n, H]
    constant int&       H     [[buffer(7)]],
    constant float&     eps   [[buffer(8)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float red[32];
  dsv41_hc_post_pre_impl_(a, a, a, 0, X, pre, post, comb, w, y, H, eps, row,
                          tid, sg, lane, red);
}

kernel void dsv41_hc_post_pre_moe(
    const device T*     part  [[buffer(0)]],    // [n * K, H]
    device T*           X     [[buffer(1)]],    // [n, 4, H]
    const device float* pre   [[buffer(2)]],    // [n, 4]
    const device float* post  [[buffer(3)]],    // [n, 4]
    const device float* comb  [[buffer(4)]],    // [n, 16]
    const device T*     w     [[buffer(5)]],    // [H]
    device T*           y     [[buffer(6)]],    // [n, H]
    constant int&       H     [[buffer(7)]],
    constant float&     eps   [[buffer(8)]],
    const device T*     sh    [[buffer(9)]],    // [n, H]
    constant int&       K     [[buffer(10)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float red[32];
  dsv41_hc_post_pre_impl_(part, part, sh, K, X, pre, post, comb, w, y, H,
                          eps, row, tid, sg, lane, red);
}

// The collapse over ceil(H / 1024) threadgroups a row, in place of the one
// of dsv41_hc_pre_norm / dsv41_hc_post_pre(_moe): a row's thousand-odd
// values are one load a thread rather than eight in turn on one core.
// Each group takes its 1024 of the row -- the post first when asked (the
// copies written back), the collapse rounded to T into y -- and its sum of
// squares; the LAST of the row's groups to finish (an atomic count) sums
// the groups' figures in group order, the same total whichever finishes
// last, and norms the row in place. What it reads of the others' work
// (their figures, y) it reads by atomic loads, after their device-scope
// fences; it sets the count back to 0 (`cnt` is zero between uses).
// K < 0: the collapse alone; K == 0: the branch `a`'s post first; K > 0:
// the branch is the MoE's K routed parts and the shared expert's output
// summed (f32, rounded once) as dsv41_moe_combine sums it. The post and
// collapse are dsv41_hc_post's and dsv41_hc_pre_norm's arithmetic, so a
// post here or before it gives the same bits. has_w 0: no norm. H even,
// <= 8192. grid {1024 * G, n}, tg {1024}.
template <int TPG>
METAL_FUNC void dsv41_hc_collapse_impl_(
    const device T* a, device T* X, const device float* pre,
    const device float* post, const device float* comb, const device T* w,
    device T* y, const int H, const float eps, const device T* sh,
    const int K, const int has_w, device atomic_uint* ps,
    device atomic_uint* cnt, const uint2 tg, const uint2 ng, const uint tid,
    const uint sg, const uint lane, threadgroup float* red,
    threadgroup uint& last, threadgroup float& rr)
{
  const int g = (int)tg.x, G = (int)ng.x, row = (int)tg.y;
  const int d = g * TPG + (int)tid;
  device T* x = X + (long)row * 4 * H;
  device T* yr = y + (long)row * H;
  const float m0 = pre[row * 4], m1 = pre[row * 4 + 1];
  const float m2 = pre[row * 4 + 2], m3 = pre[row * 4 + 3];
  float vv = 0.0f;
  if (d < H) {
    float nx0, nx1, nx2, nx3;
    if (K >= 0) {
      float av;
      if (K > 0) {
        // the parts' loads issued together (K <= 8), then summed in order
        const device T* ap = a + (long)row * K * H + d;
        float pv[8];
        for (int j = 0; j < 8; ++j) {
          pv[j] = j < K ? (float)ap[(long)j * H] : 0.0f;
        }
        const float shv = (float)sh[(long)row * H + d];
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) {
          if (j < K) { s += pv[j]; }
        }
        for (int j = 8; j < K; ++j) { s += (float)ap[(long)j * H]; }
        s += shv;
        av = (float)(T)s;
      } else {
        av = (float)a[(long)row * H + d];
      }
      const device float* cm = comb + row * 16;
      const float r0 = (float)x[d], r1 = (float)x[H + d];
      const float r2 = (float)x[2 * H + d], r3 = (float)x[3 * H + d];
      float nx[4];
      for (int k = 0; k < 4; ++k) {
        const float s = cm[k] * r0 + cm[4 + k] * r1 + cm[8 + k] * r2 +
                        cm[12 + k] * r3;
        const T v = (T)(post[row * 4 + k] * av + s);
        x[k * H + d] = v;
        nx[k] = (float)v;
      }
      nx0 = nx[0];
      nx1 = nx[1];
      nx2 = nx[2];
      nx3 = nx[3];
    } else {
      nx0 = (float)x[d];
      nx1 = (float)x[H + d];
      nx2 = (float)x[2 * H + d];
      nx3 = (float)x[3 * H + d];
    }
    const float s = m0 * nx0 + m1 * nx1 + m2 * nx2 + m3 * nx3;
    const T v = (T)s;
    vv = (float)v;
    yr[d] = v;
  }
  if (has_w == 0) { return; }
  constexpr int NS = TPG / 32;
  const float ss = simd_sum(vv * vv);
  if (lane == 0) { red[sg] = ss; }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid == 0) {
    float t = 0.0f;
    for (int q = 0; q < NS; ++q) { t += red[q]; }
    atomic_store_explicit(&ps[(long)row * G + g], as_type<uint>(t),
                          memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    const uint prev =
        atomic_fetch_add_explicit(&cnt[row], 1u, memory_order_relaxed);
    last = prev == (uint)(G - 1) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last == 0u) { return; }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  if (tid == 0) {
    float tot = 0.0f;
    for (int q = 0; q < G; ++q) {
      tot += as_type<float>(atomic_load_explicit(&ps[(long)row * G + q],
                                                 memory_order_relaxed));
    }
    rr = 1.0f / sqrt(tot / (float)H + eps);
    atomic_store_explicit(&cnt[row], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = rr;
  // the row's pairs, read as words (H even, so a row starts on one)
  device atomic_uint* yw = reinterpret_cast<device atomic_uint*>(yr);
  device vec<T, 2>* y2 = reinterpret_cast<device vec<T, 2>*>(yr);
  const int P = H / 2;
  for (int p = (int)tid; p < P; p += TPG) {
    const vec<T, 2> v =
        as_type<vec<T, 2>>(atomic_load_explicit(&yw[p],
                                                memory_order_relaxed));
    const float w0 = (float)w[2 * p], w1 = (float)w[2 * p + 1];
    y2[p] = vec<T, 2>((T)(w0 * ((float)v.x * r)),
                      (T)(w1 * ((float)v.y * r)));
  }
}

#define DSV41_HC_COLLAPSE(NAME, TPG)                                         \
  kernel void NAME(                                                          \
      const device T* a [[buffer(0)]], device T* X [[buffer(1)]],            \
      const device float* pre [[buffer(2)]],                                 \
      const device float* post [[buffer(3)]],                                \
      const device float* comb [[buffer(4)]], const device T* w [[buffer(5)]],\
      device T* y [[buffer(6)]], constant int& H [[buffer(7)]],              \
      constant float& eps [[buffer(8)]], const device T* sh [[buffer(9)]],   \
      constant int& K [[buffer(10)]], constant int& has_w [[buffer(11)]],    \
      device atomic_uint* ps [[buffer(12)]],                                 \
      device atomic_uint* cnt [[buffer(13)]],                                \
      uint2 tg [[threadgroup_position_in_grid]],                             \
      uint2 ng [[threadgroups_per_grid]],                                    \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float red[TPG / 32];                                         \
    threadgroup uint last;                                                   \
    threadgroup float rr;                                                    \
    dsv41_hc_collapse_impl_<TPG>(a, X, pre, post, comb, w, y, H, eps, sh, K, \
                                 has_w, ps, cnt, tg, ng, tid, sg, lane, red, \
                                 last, rr);                                  \
  }

// 1024 of a row a threadgroup. (MEASURED on the M5 Ultra, a decode token:
// groups of 256 cost 0.5 ms of GPU time more.)
DSV41_HC_COLLAPSE(dsv41_hc_collapse, 1024)

// ---- norms and rotary -------------------------------------------------

// RMSNorm over rows of C (stride ldx in, ldy out): y = w * x * rsqrt(mean
// x^2 + eps), f32 inside. x_f32 1 reads f32 rows and rounds each value to
// the element type first (the reference norms kv.to(dtype)). One row a
// threadgroup of 256. grid {256 * n}, tg {256}.
kernel void dsv41_rms_norm(
    const device void*  xv    [[buffer(0)]],
    const device T*     w     [[buffer(1)]],
    device T*           y     [[buffer(2)]],
    constant int&       C     [[buffer(3)]],
    constant int&       ldx   [[buffer(4)]],
    constant int&       ldy   [[buffer(5)]],
    constant float&     eps   [[buffer(6)]],
    constant int&       x_f32 [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  const device T* xe = static_cast<const device T*>(xv) + (long)row * ldx;
  const device float* xf =
      static_cast<const device float*>(xv) + (long)row * ldx;
  float ss = 0.0f;
  for (int d = (int)tid; d < C; d += 256) {
    const float v = x_f32 != 0 ? (float)(T)xf[d] : (float)xe[d];
    ss += v * v;
  }
  threadgroup float red[8];
  ss = simd_sum(ss);
  if (lane == 0) { red[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < 8; ++g) { tot += red[g]; }
  const float r = 1.0f / sqrt(tot / (float)C + eps);
  device T* yr = y + (long)row * ldy;
  for (int d = (int)tid; d < C; d += 256) {
    const float v = x_f32 != 0 ? (float)(T)xf[d] : (float)xe[d];
    yr[d] = (T)((float)w[d] * (v * r));
  }
}

// apply_rotary_emb on the LAST rd dims of each head, adjacent pairs as
// complex numbers, in place. Row i is at position pos0 + i * step;
// `inverse` rotates back (the conjugate). tab [positions, rd/2] (cos, sin).
// grid {rd/2, heads, n}.
kernel void dsv41_rope(
    device T*            x       [[buffer(0)]],
    const device float2* tab     [[buffer(1)]],
    constant int&        ld_row  [[buffer(2)]],
    constant int&        ld_head [[buffer(3)]],
    constant int&        D       [[buffer(4)]],
    constant int&        rd      [[buffer(5)]],
    constant int&        pos0    [[buffer(6)]],
    constant int&        step    [[buffer(7)]],
    constant int&        inverse [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int j = (int)gid.x, h = (int)gid.y, i = (int)gid.z;
  if (j >= rd / 2) { return; }
  const int pos = pos0 + i * step;
  const float2 cs = tab[(long)pos * (rd / 2) + j];
  const float sn = inverse != 0 ? -cs.y : cs.y;
  device T* p = x + (long)i * ld_row + (long)h * ld_head + (D - rd) + 2 * j;
  const float re = (float)p[0], im = (float)p[1];
  p[0] = (T)(re * cs.x - im * sn);
  p[1] = (T)(re * sn + im * cs.x);
}

// The window ring: row i of this chunk is position p0 + i, kept in slot
// (p0 + i) % R; only the last R rows are kept. R is the window plus the
// rows a speculative verify may write past what it keeps, so a rejected
// row never takes a slot a later query still reads. grid {D, n}.
kernel void dsv41_ring_write(
    const device T* kv   [[buffer(0)]],         // [n, D]
    device T*       ring [[buffer(1)]],         // [R, D]
    constant int&   n    [[buffer(2)]],
    constant int&   D    [[buffer(3)]],
    constant int&   R    [[buffer(4)]],
    constant int&   p0   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, i = (int)gid.y;
  if (d >= D || i < n - R) { return; }
  ring[(long)((p0 + i) % R) * D + d] = kv[(long)i * D + d];
}

// The latent's tail of a decode or verify step in one dispatch:
// dsv41_rms_norm (its arithmetic and reduction order, 256 threads a row),
// then dsv41_rope on the last rd dims (the normed values rounded first,
// as the two kernels store them), and -- `ring` 1 -- the row into the
// window ring as dsv41_ring_write would, ahead of the attention (safe for
// a step of at most the ring's slack rows: no query of the step reads the
// slots they take). Bit-identical to the three. grid {256 * n}, tg {256}.
kernel void dsv41_kv_norm_rope(
    const device T*      x     [[buffer(0)]],   // [n, D]
    const device T*      w     [[buffer(1)]],   // [D]
    device T*            y     [[buffer(2)]],   // [n, D]
    const device float2* tab   [[buffer(3)]],
    device T*            rg    [[buffer(4)]],   // [R, D]
    constant int&        D     [[buffer(5)]],
    constant float&      eps   [[buffer(6)]],
    constant int&        rd    [[buffer(7)]],
    constant int&        p0    [[buffer(8)]],
    constant int&        R     [[buffer(9)]],
    constant int&        ring  [[buffer(10)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  const device T* xe = x + (long)row * D;
  float ss = 0.0f;
  for (int d = (int)tid; d < D; d += 256) {
    const float v = (float)xe[d];
    ss += v * v;
  }
  threadgroup float red[8];
  ss = simd_sum(ss);
  if (lane == 0) { red[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < 8; ++g) { tot += red[g]; }
  const float r = 1.0f / sqrt(tot / (float)D + eps);
  device T* yr = y + (long)row * D;
  device T* rr = rg + (long)((p0 + (int)row) % max(R, 1)) * D;
  const int pos = p0 + (int)row;
  for (int d = (int)tid; d < D; d += 256) {
    if (d < D - rd) {
      const T v = (T)((float)w[d] * ((float)xe[d] * r));
      yr[d] = v;
      if (ring != 0) { rr[d] = v; }
    } else if ((d & 1) == 0) {
      // a rotary pair (d, d + 1), each normed and rounded as stored, then
      // turned as dsv41_rope turns it
      const float re = (float)(T)((float)w[d] * ((float)xe[d] * r));
      const float im = (float)(T)((float)w[d + 1] * ((float)xe[d + 1] * r));
      const float2 cs = tab[(long)pos * (rd / 2) + (d - (D - rd)) / 2];
      const T o0 = (T)(re * cs.x - im * cs.y);
      const T o1 = (T)(re * cs.y + im * cs.x);
      yr[d] = o0;
      yr[d + 1] = o1;
      if (ring != 0) {
        rr[d] = o0;
        rr[d + 1] = o1;
      }
    }
  }
}

// ---- attention ----------------------------------------------------------

// kernel.sparse_attn over the two key sets Attention.forward concatenates:
// the W positions up to the query's (this chunk's latents for positions
// >= p0, the ring before), then the compressed latents the index source
// picked (idx, compressed positions, -1 none; a pick must be visible:
// t < (pos + 1) / ratio). The latent is the key AND the value. A sink per
// head joins the softmax's denominator. The output's rotary tail is
// rotated back by the query's position (Attention.forward's inverse
// apply_rotary_emb) and stored GROUP-major, [G][n][NH/G * 512], the
// layout wo_a's per-group GEMMs read.
// One simdgroup a (row, head), a lane 16 of the 512 dims; 8 heads a
// threadgroup. grid {32 * 8 * ceil(NH/8), n}, tg {256}.
kernel void dsv41_attn(
    const device T*      q     [[buffer(0)]],   // [n, NH, 512]
    const device T*      kvn   [[buffer(1)]],   // [n, 512]
    const device T*      ring  [[buffer(2)]],   // [W, 512]
    const device T*      comp  [[buffer(3)]],   // [C, 512]
    const device int*    idx   [[buffer(4)]],   // [n, KC]
    const device float*  sink  [[buffer(5)]],   // [NH]
    device T*            o     [[buffer(6)]],
    const device float2* tab   [[buffer(7)]],
    constant int&        n     [[buffer(8)]],
    constant int&        NH    [[buffer(9)]],
    constant int&        W     [[buffer(10)]],
    constant int&        p0    [[buffer(11)]],
    constant int&        KC    [[buffer(12)]],
    constant int&        ratio [[buffer(13)]],
    constant float&      scale [[buffer(14)]],
    constant int&        G     [[buffer(15)]],
    constant int&        rd    [[buffer(16)]],
    constant int&        R     [[buffer(17)]],  // the ring's slots
    constant int&        lo    [[buffer(18)]],  // the window's first
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int D = 512, E = 16;
  const int i = (int)tg.y;
  const int h = (int)tg.x * 8 + (int)sg;
  if (h >= NH || i >= n) { return; }
  const int pos = p0 + i;
  const int c0 = (int)lane * E;
  float qv[E];
  const device T* qr = q + ((long)i * NH + h) * D + c0;
  for (int e = 0; e < E; ++e) { qv[e] = (float)qr[e]; }
  float m = -1e30f, l = 0.0f;
  float acc[E];
  for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
  auto take = [&](const device T* k) {
    float kv[E];
    float s = 0.0f;
    for (int e = 0; e < E; ++e) {
      kv[e] = (float)k[c0 + e];
      s += qv[e] * kv[e];
    }
    s = simd_sum(s) * scale;
    const float mn = max(m, s);
    const float cr = exp(m - mn), pe = exp(s - mn);
    l = l * cr + pe;
    for (int e = 0; e < E; ++e) { acc[e] = acc[e] * cr + pe * kv[e]; }
    m = mn;
  };
  // `lo`: the first position the window may read -- a CED prefill's
  // decoder layers start it at the tail they ran (nothing before it was
  // computed there)
  const int wlo = max(max(0, lo), pos - W + 1);
  for (int p = wlo; p <= pos; ++p) {
    take(p >= p0 ? kvn + (long)(p - p0) * D : ring + (long)(p % R) * D);
  }
  if (KC > 0) {
    const int vis = (pos + 1) / ratio;
    const device int* ir = idx + (long)i * KC;
    for (int j = 0; j < KC; ++j) {
      const int t = ir[j];
      if (t < 0 || t >= vis) { continue; }
      take(comp + (long)t * D);
    }
  }
  {
    const float sk = sink[h];
    const float mn = max(m, sk);
    const float cr = exp(m - mn);
    l = l * cr + exp(sk - mn);
    for (int e = 0; e < E; ++e) { acc[e] *= cr; }
  }
  const float inv = 1.0f / l;
  float out[E];
  for (int e = 0; e < E; ++e) { out[e] = acc[e] * inv; }
  if (c0 >= D - rd) {
    for (int e = 0; e < E; e += 2) {
      const int j = (c0 + e - (D - rd)) / 2;
      const float2 cs = tab[(long)pos * (rd / 2) + j];
      const float re = out[e], im = out[e + 1];
      out[e] = re * cs.x + im * cs.y;
      out[e + 1] = -re * cs.y + im * cs.x;
    }
  }
  const int hg = NH / G;
  const int g = h / hg, hh = h % hg;
  device T* orow = o + ((long)g * n + i) * (long)(hg * D) + hh * D + c0;
  for (int e = 0; e < E; ++e) { orow[e] = (T)out[e]; }
}

// dsv41_attn split by keys, for a decode or verify step's few rows (one
// simdgroup a head walking all 640 keys kept 8 threadgroups of an 80-core
// GPU busy). A row's keys are W window slots -- slot j the position
// pos - W + 1 + j, none below 0 -- then its KC picks, in chunks of 16: a
// threadgroup takes one chunk into threadgroup memory once and each of its
// 16 simdgroups (a head each) its scores, max, sum and weighted values (V
// is K), a partial a (row, head, chunk): part [n, NH, CH, 512] f32, ml
// [n, NH, CH] (max, sum). A chunk with no key leaves (-1e30, 0) and zeros.
// grid {NH / 16 threadgroups, CH, n}, tg {512}.
kernel void dsv41_attn_split(
    const device T*      q     [[buffer(0)]],   // [n, NH, 512]
    const device T*      kvn   [[buffer(1)]],   // [n, 512]
    const device T*      ring  [[buffer(2)]],   // [R, 512]
    const device T*      comp  [[buffer(3)]],   // [C, 512]
    const device int*    idx   [[buffer(4)]],   // [n, KC]
    device float*        part  [[buffer(5)]],
    device float2*       ml    [[buffer(6)]],
    constant int&        NH    [[buffer(7)]],
    constant int&        W     [[buffer(8)]],
    constant int&        p0    [[buffer(9)]],
    constant int&        KC    [[buffer(10)]],
    constant int&        ratio [[buffer(11)]],
    constant float&      scale [[buffer(12)]],
    constant int&        R     [[buffer(13)]],
    constant int&        CH    [[buffer(14)]],
    const device float2* tab   [[buffer(15)]],
    constant int&        rope  [[buffer(16)]],  // rd: q's tail rotated here
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int D = 512, E = 16, CK = 16, HG = 16;
  threadgroup T kt[CK * D];
  threadgroup int ok[CK];
  const int ch = (int)tg.y, i = (int)tg.z;
  const int pos = p0 + i;
  {
    const int kk = (int)t / 32, c0 = ((int)t % 32) * E;
    const int j = ch * CK + kk;
    const device T* src = nullptr;
    if (j < W) {
      const int p = pos - W + 1 + j;
      if (p >= 0) {
        src = p >= p0 ? kvn + (long)(p - p0) * D : ring + (long)(p % R) * D;
      }
    } else if (j < W + KC) {
      const int tt = idx[(long)i * KC + (j - W)];
      if (tt >= 0 && tt < (pos + 1) / ratio) { src = comp + (long)tt * D; }
    }
    if ((int)t % 32 == 0) { ok[kk] = src != nullptr ? 1 : 0; }
    for (int e = 0; e < E; ++e) {
      kt[kk * D + c0 + e] = src != nullptr ? src[c0 + e] : T(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int h = (int)tg.x * HG + (int)sg;
  if (h >= NH) { return; }
  const int c0 = (int)lane * E;
  float qv[E];
  const device T* qr = q + ((long)i * NH + h) * D + c0;
  for (int e = 0; e < E; ++e) { qv[e] = (float)qr[e]; }
  // `rope` > 0: q comes unrotated, and its last `rope` dims are turned
  // here as dsv41_rope turns them (adjacent pairs, rounded as it stores
  // them) -- the rotary's own dispatch folded in
  if (rope > 0 && c0 + E > D - rope) {
    for (int e = 0; e < E; e += 2) {
      const int d = c0 + e;
      if (d < D - rope) { continue; }
      const float2 cs = tab[(long)pos * (rope / 2) + (d - (D - rope)) / 2];
      const float re = qv[e], im = qv[e + 1];
      qv[e] = (float)(T)(re * cs.x - im * cs.y);
      qv[e + 1] = (float)(T)(re * cs.y + im * cs.x);
    }
  }
  float s[CK];
  float m = -1e30f;
  for (int k = 0; k < CK; ++k) {
    s[k] = -1e30f;
    if (ok[k] == 0) { continue; }
    const threadgroup T* kr = kt + k * D + c0;
    float d = 0.0f;
    for (int e = 0; e < E; ++e) { d += qv[e] * (float)kr[e]; }
    s[k] = simd_sum(d) * scale;
    m = max(m, s[k]);
  }
  float l = 0.0f, acc[E];
  for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
  for (int k = 0; k < CK; ++k) {
    if (ok[k] == 0) { continue; }
    const float p = exp(s[k] - m);
    l += p;
    const threadgroup T* kr = kt + k * D + c0;
    for (int e = 0; e < E; ++e) { acc[e] += p * (float)kr[e]; }
  }
  const long at = ((long)i * NH + h) * CH + ch;
  device float* pr = part + at * D + c0;
  for (int e = 0; e < E; ++e) { pr[e] = acc[e]; }
  if (lane == 0) { ml[at] = float2(m, l); }
}

// dsv41_attn_split's partials merged with the head's sink, the output's
// rotary tail turned back at the row's position, group-major as dsv41_attn
// stores it. grid {NH, n} threadgroups of 256 (two features a thread).
kernel void dsv41_attn_merge(
    const device float*  part  [[buffer(0)]],
    const device float2* ml    [[buffer(1)]],
    const device float*  sink  [[buffer(2)]],   // [NH]
    device T*            o     [[buffer(3)]],
    const device float2* tab   [[buffer(4)]],
    constant int&        n     [[buffer(5)]],
    constant int&        NH    [[buffer(6)]],
    constant int&        p0    [[buffer(7)]],
    constant int&        CH    [[buffer(8)]],
    constant int&        G     [[buffer(9)]],
    constant int&        rd    [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint  t  [[thread_index_in_threadgroup]])
{
  constexpr int D = 512, kMaxCh = 64;
  threadgroup float wc[kMaxCh];
  threadgroup float tot[2];
  const int h = (int)tg.x, i = (int)tg.y;
  const int pos = p0 + i;
  const long at = ((long)i * NH + h) * CH;
  if (t == 0) {
    float M = sink[h];
    for (int c = 0; c < CH; ++c) { M = max(M, ml[at + c].x); }
    float L = exp(sink[h] - M);
    for (int c = 0; c < CH; ++c) {
      const float w = exp(ml[at + c].x - M);
      wc[c] = w;
      L += w * ml[at + c].y;
    }
    tot[0] = 1.0f / L;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int d = (int)t * 2;
  float a0 = 0.0f, a1 = 0.0f;
  for (int c = 0; c < CH; ++c) {
    const device float* pr = part + (at + c) * D + d;
    a0 += wc[c] * pr[0];
    a1 += wc[c] * pr[1];
  }
  float o0 = a0 * tot[0], o1 = a1 * tot[0];
  if (d >= D - rd) {
    const float2 cs = tab[(long)pos * (rd / 2) + (d - (D - rd)) / 2];
    const float re = o0, im = o1;
    o0 = re * cs.x + im * cs.y;
    o1 = -re * cs.y + im * cs.x;
  }
  const int hg = NH / G;
  const int g = h / hg, hh = h % hg;
  device T* orow = o + ((long)g * n + i) * (long)(hg * D) + hh * D + d;
  orow[0] = (T)o0;
  orow[1] = (T)o1;
}

// dst[i] = src[i], 32-bit words. grid {n}.
kernel void dsv41_copy32(
    const device uint* src [[buffer(0)]],
    device uint*       dst [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
  if ((int)i < n) { dst[i] = src[i]; }
}

// ---- DSpark (the draft head) ---------------------------------------------

// Transformer.forward's main hidden: a target layer's input, its copies'
// mean (rounded once to the element type), into columns [off, off + H) of
// a row of `ld`. grid {H, n}.
kernel void dsv41_hc_mean(
    const device T* X   [[buffer(0)]],          // [n, HC, H]
    device T*       out [[buffer(1)]],
    constant int&   H   [[buffer(2)]],
    constant int&   ld  [[buffer(3)]],
    constant int&   off [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, i = (int)gid.y;
  if (d >= H) { return; }
  const device T* x = X + (long)i * 4 * H + d;
  const float s = (float)x[0] + (float)x[H] + (float)x[2 * H] +
                  (float)x[3 * H];
  out[(long)i * ld + off + d] = (T)(s * 0.25f);
}

// DSparkAttention over a draft block: every row of the block (position
// p + 1 + i) attends the window of committed latents up to p (the main
// model's, kept in the stage's ring, slot % R) and ALL of the block's own
// rows -- no causal mask inside the block -- with the head's sink; the
// output's rotary tail turned back at the row's position, group-major as
// dsv41_attn stores it. grid {32 * 8 * ceil(NH/8), n}, tg {256}.
kernel void dsv41_spark_attn(
    const device T*      q     [[buffer(0)]],   // [n, NH, 512]
    const device T*      kvb   [[buffer(1)]],   // [n, 512]
    const device T*      ring  [[buffer(2)]],   // [R, 512]
    const device float*  sink  [[buffer(3)]],   // [NH]
    device T*            o     [[buffer(4)]],
    const device float2* tab   [[buffer(5)]],
    constant int&        n     [[buffer(6)]],
    constant int&        NH    [[buffer(7)]],
    constant int&        W     [[buffer(8)]],
    constant int&        R     [[buffer(9)]],
    constant int&        p     [[buffer(10)]],
    constant float&      scale [[buffer(11)]],
    constant int&        G     [[buffer(12)]],
    constant int&        rd    [[buffer(13)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int D = 512, E = 16;
  const int i = (int)tg.y;
  const int h = (int)tg.x * 8 + (int)sg;
  if (h >= NH || i >= n) { return; }
  const int pos = p + 1 + i;
  const int c0 = (int)lane * E;
  float qv[E];
  const device T* qr = q + ((long)i * NH + h) * D + c0;
  for (int e = 0; e < E; ++e) { qv[e] = (float)qr[e]; }
  float m = -1e30f, l = 0.0f;
  float acc[E];
  for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
  auto take = [&](const device T* k) {
    float kv[E];
    float s = 0.0f;
    for (int e = 0; e < E; ++e) {
      kv[e] = (float)k[c0 + e];
      s += qv[e] * kv[e];
    }
    s = simd_sum(s) * scale;
    const float mn = max(m, s);
    const float cr = exp(m - mn), pe = exp(s - mn);
    l = l * cr + pe;
    for (int e = 0; e < E; ++e) { acc[e] = acc[e] * cr + pe * kv[e]; }
    m = mn;
  };
  for (int t = max(0, p - W + 1); t <= p; ++t) {
    take(ring + (long)(t % R) * D);
  }
  for (int j = 0; j < n; ++j) { take(kvb + (long)j * D); }
  {
    const float sk = sink[h];
    const float mn = max(m, sk);
    const float cr = exp(m - mn);
    l = l * cr + exp(sk - mn);
    for (int e = 0; e < E; ++e) { acc[e] *= cr; }
  }
  const float inv = 1.0f / l;
  float out[E];
  for (int e = 0; e < E; ++e) { out[e] = acc[e] * inv; }
  if (c0 >= D - rd) {
    for (int e = 0; e < E; e += 2) {
      const int j = (c0 + e - (D - rd)) / 2;
      const float2 cs = tab[(long)pos * (rd / 2) + j];
      const float re = out[e], im = out[e + 1];
      out[e] = re * cs.x + im * cs.y;
      out[e + 1] = -re * cs.y + im * cs.x;
    }
  }
  const int hg = NH / G;
  const int g = h / hg, hh = h % hg;
  device T* orow = o + ((long)g * n + i) * (long)(hg * D) + hh * D + c0;
  for (int e = 0; e < E; ++e) { orow[e] = (T)out[e]; }
}

// The Markov head's embedding of the block's i-th token (toks[i], read on
// the GPU: the token a previous step picked) into row i of out.
// grid {Dm}.
kernel void dsv41_gather_tok(
    const device T*   table [[buffer(0)]],    // [V, Dm]
    const device int* toks  [[buffer(1)]],
    device T*         out   [[buffer(2)]],    // [n, Dm]
    constant int&     Dm    [[buffer(3)]],
    constant int&     i     [[buffer(4)]],
    uint d [[thread_position_in_grid]])
{
  if ((int)d >= Dm) { return; }
  out[(long)i * Dm + d] = table[(long)toks[i] * Dm + d];
}

// forward_head's step i: logits[i] += bias, then its argmax (the lowest id
// of a tie, as torch's) is the block's next token, toks[i + 1].
// One threadgroup of 1024.
kernel void dsv41_add_argmax(
    device float*       lg   [[buffer(0)]],   // [n, V]
    const device float* bias [[buffer(1)]],   // [V]
    device int*         toks [[buffer(2)]],
    constant int&       V    [[buffer(3)]],
    constant int&       i    [[buffer(4)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  device float* row = lg + (long)i * V;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int v = (int)tid; v < V; v += 1024) {
    const float x = row[v] + bias[v];
    row[v] = x;
    if (x > bv || (x == bv && v < bi)) { bv = x; bi = v; }
  }
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(bv, (ushort)off);
    const int oi = simd_shuffle_down(bi, (ushort)off);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  threadgroup float rv[32];
  threadgroup int ri[32];
  if (lane == 0) { rv[sg] = bv; ri[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float b = rv[0];
    int b2 = ri[0];
    for (int g = 1; g < 32; ++g) {
      if (rv[g] > b || (rv[g] == b && ri[g] < b2)) { b = rv[g]; b2 = ri[g]; }
    }
    toks[i + 1] = b2;
  }
}

// A buffer's first `n` 32-bit words set to v. grid {n}.
kernel void dsv41_fill32(
    device int*   b [[buffer(0)]],
    constant int& v [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
  if ((int)i < n) { b[i] = v; }
}

// A handshake with the host inside a running command buffer: word 0 set to
// `post`, then wait for word 1 to reach `want` (the host's answer), at most
// `spins` loads; word 2 counts the waits that gave up. grid {1}.
kernel void dsv41_handshake(
    device atomic_uint* f     [[buffer(0)]],
    constant uint&      post  [[buffer(1)]],
    constant uint&      want  [[buffer(2)]],
    constant uint&      spins [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
  if (i != 0) { return; }
  atomic_store_explicit(&f[0], post, memory_order_relaxed);
  for (uint k = 0; k < spins; ++k) {
    if (atomic_load_explicit(&f[1], memory_order_relaxed) >= want) {
      return;
    }
  }
  atomic_fetch_add_explicit(&f[2], 1u, memory_order_relaxed);
}

// The host's turn inside a running buffer, with data both ways. The words
// of s0..s2 (n0..n2 each; what the GPU wrote before) go to the mailbox
// `bo` by atomic stores, then their checksum (dsv41_hs_sum_), then word 0
// of `f` is set to `seq` and thread 0 waits for word 1 to reach `seq`;
// then `bi` (the host's answer, m0..m2 words and its checksum) is read by
// atomic loads into d0..d2, again until the checksum holds. Atomics are
// what both sides were seen to observe while a buffer runs (on an M5
// Ultra; the model's probe keeps this off where they are not), plain
// stores are not relied on, and the checksums make a late or partial
// mailbox a give-up rather than a wrong answer.
// A wait gives up when word 3 (the host's heartbeat) has not moved in
// `spins` loads, counting it in word 2; once one has given up, later ones
// do not wait (a host that is gone costs one wait, not one a layer).
// grid {256}, tg {256}.
inline uint dsv41_hs_mix_(uint k, uint v)
{
  return v * (2u * k + 1u) + (v >> 16) * 0x9E3779B9u;
}

// dsv41_route's body (see it, below) for row `row`, by a group of 256:
// the route kernel's own and, folded, the handshake's.
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

inline void dsv41_handshake_body_(
    device atomic_uint* f, const device uint* s0, const device uint* s1,
    const device uint* s2, device atomic_uint* bo, device atomic_uint* bi,
    device uint* d0, device uint* d1, device uint* d2, const uint4 n,
    const uint4 m, const uint t, const uint nt, const uint sg,
    const uint lane, threadgroup uint* part, threadgroup uint& verdict)
{
  const uint no = n.x + n.y + n.z, ni = m.x + m.y + m.z, seq = n.w;
  // the group's sum of what each thread holds
  auto total = [&](uint mine) {
    const uint s = simd_sum(mine);
    if (lane == 0) { part[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint all = 0;
    for (uint g = 0; g < (nt + 31) / 32; ++g) { all += part[g]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return all;
  };
  uint mine = 0;
  for (uint k = t; k < no; k += nt) {
    const uint v = k < n.x ? s0[k] : k < n.x + n.y ? s1[k - n.x]
                                                    : s2[k - n.x - n.y];
    atomic_store_explicit(&bo[k], v, memory_order_relaxed);
    mine += dsv41_hs_mix_(k, v);
  }
  const uint so = total(mine) + seq * 2654435761u;
  if (t == 0) { atomic_store_explicit(&bo[no], so, memory_order_relaxed); }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) {
    atomic_store_explicit(&f[0], seq, memory_order_relaxed);
    uint k = 0;
    uint hb = atomic_load_explicit(&f[3], memory_order_relaxed);
    while (atomic_load_explicit(&f[2], memory_order_relaxed) == 0 &&
           atomic_load_explicit(&f[1], memory_order_relaxed) < seq) {
      if (++k < m.w) { continue; }
      const uint h = atomic_load_explicit(&f[3], memory_order_relaxed);
      if (h == hb) {
        atomic_fetch_add_explicit(&f[2], 1u, memory_order_relaxed);
        break;
      }
      hb = h;
      k = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  // the answer, read until its checksum holds (or the wait gave up)
  for (uint tries = 0;; ++tries) {
    uint got = 0;
    for (uint k = t; k < ni; k += nt) {
      const uint v = atomic_load_explicit(&bi[k], memory_order_relaxed);
      got += dsv41_hs_mix_(k, v);
      if (k < m.x) {
        d0[k] = v;
      } else if (k < m.x + m.y) {
        d1[k - m.x] = v;
      } else {
        d2[k - m.x - m.y] = v;
      }
    }
    const uint sum = total(got) + seq * 2654435761u;
    if (t == 0) {
      const bool gave_up =
          atomic_load_explicit(&f[2], memory_order_relaxed) != 0;
      const bool ok =
          atomic_load_explicit(&bi[ni], memory_order_relaxed) == sum;
      verdict = ok || gave_up ? 1u : tries + 1 >= 4096u ? 2u : 0u;
      if (verdict == 2u) {
        atomic_fetch_add_explicit(&f[2], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (verdict != 0u) { break; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void dsv41_handshake_io(
    device atomic_uint* f     [[buffer(0)]],
    const device uint*  s0    [[buffer(1)]],
    const device uint*  s1    [[buffer(2)]],
    const device uint*  s2    [[buffer(3)]],
    device atomic_uint* bo    [[buffer(4)]],
    device atomic_uint* bi    [[buffer(5)]],
    device uint*        d0    [[buffer(6)]],
    device uint*        d1    [[buffer(7)]],
    device uint*        d2    [[buffer(8)]],
    constant uint4&     n     [[buffer(9)]],    // n0, n1, n2, seq
    constant uint4&     m     [[buffer(10)]],   // m0, m1, m2, spins
    uint t    [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup uint part[32];
  threadgroup uint verdict;
  dsv41_handshake_body_(f, s0, s1, s2, bo, bi, d0, d1, d2, n, m, t, nt, sg,
                        lane, part, verdict);
}

// dsv41_route for a decode step's one row (dsv41_route_tg_, by the whole
// group), then dsv41_handshake_io with its picks as s0: the route's
// dispatch folded into the handshake's. eid's words are read through the
// pointer they were written through, after a device barrier.
// grid {256}, tg {256}.
kernel void dsv41_handshake_route(
    device atomic_uint* f       [[buffer(0)]],
    const device uint*  s1      [[buffer(2)]],
    const device uint*  s2      [[buffer(3)]],
    device atomic_uint* bo      [[buffer(4)]],
    device atomic_uint* bi      [[buffer(5)]],
    device uint*        d0      [[buffer(6)]],
    device uint*        d1      [[buffer(7)]],
    device uint*        d2      [[buffer(8)]],
    constant uint4&     n       [[buffer(9)]],    // n0, n1, n2, seq
    constant uint4&     m       [[buffer(10)]],   // m0, m1, m2, spins
    const device float* sc      [[buffer(11)]],   // [rows, E]
    const device float* bias    [[buffer(12)]],   // [E]
    const device float* bias_vl [[buffer(13)]],   // [E]
    const device uchar* vl      [[buffer(14)]],   // [rows] (has_vl)
    device int*         eid     [[buffer(15)]],   // [rows, k]
    device float*       wt      [[buffer(16)]],   // [rows, k]
    constant int&       E       [[buffer(17)]],
    constant int&       topk    [[buffer(18)]],
    constant float&     rscale  [[buffer(19)]],
    constant int&       norm    [[buffer(20)]],
    constant int&       has_vl  [[buffer(21)]],
    uint t    [[thread_index_in_threadgroup]],
    uint nt   [[threads_per_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup uint part[32];
  threadgroup uint verdict;
  threadgroup float val[1024], sel[1024];
  threadgroup float rv[8];
  threadgroup int ri[8];
  threadgroup int pick[16];
  dsv41_route_tg_(sc, bias, bias_vl, vl, eid, wt, E, topk, rscale, norm,
                  has_vl, 0, t, sg, lane, val, sel, rv, ri, pick);
  threadgroup_barrier(mem_flags::mem_device);
  dsv41_handshake_body_(f, reinterpret_cast<const device uint*>(eid), s1,
                        s2, bo, bi, d0, d1, d2, n, m, t, nt, sg, lane, part,
                        verdict);
}

// The handshake probe's GPU side: `w` set to seq * 4096 + k (before a
// handshake); after it, `d` (the answer, copied) and `raw` (the host's
// plain stores) checked against seq * 7 + k, mismatches counted in words
// 5 (copied) and 6 (plain) of `f`. grid {n}.
kernel void dsv41_probe_write(
    device uint*        w   [[buffer(0)]],
    constant uint&      seq [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint k [[thread_position_in_grid]])
{
  if (k < n) { w[k] = seq * 4096u + k; }
}

kernel void dsv41_probe_check(
    const device uint*  d   [[buffer(0)]],
    const device uint*  raw [[buffer(1)]],
    device atomic_uint* f   [[buffer(2)]],
    constant uint&      seq [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
  if (k >= n) { return; }
  if (d[k] != seq * 7u + k) {
    atomic_fetch_add_explicit(&f[5], 1u, memory_order_relaxed);
  }
  if (raw[k] != seq * 7u + k) {
    atomic_fetch_add_explicit(&f[6], 1u, memory_order_relaxed);
  }
}

// A visibility probe: `rounds` handshakes in one dispatch, posting at word
// 0 and reading the answer at word `at`, by a load (mode 0) or an atomic
// add of 0 (mode 1). grid {1}.
kernel void dsv41_handshake_probe(
    device atomic_uint* f      [[buffer(0)]],
    constant uint&      rounds [[buffer(1)]],
    constant uint&      spins  [[buffer(2)]],
    constant uint&      at     [[buffer(3)]],
    constant uint&      mode   [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
  if (i != 0) { return; }
  for (uint r = 1; r <= rounds; ++r) {
    atomic_store_explicit(&f[0], r, memory_order_relaxed);
    uint k = 0;
    while (++k < spins) {
      const uint v =
          mode == 0 ? atomic_load_explicit(&f[at], memory_order_relaxed)
                    : atomic_fetch_add_explicit(&f[at], 0u,
                                                memory_order_relaxed);
      if (v >= r) { break; }
    }
    if (k >= spins) {
      atomic_fetch_add_explicit(&f[2], 1u, memory_order_relaxed);
      return;
    }
  }
}

// `rounds` handshakes in one dispatch (a latency probe). grid {1}.
kernel void dsv41_handshake_rounds(
    device atomic_uint* f      [[buffer(0)]],
    constant uint&      rounds [[buffer(1)]],
    constant uint&      spins  [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
  if (i != 0) { return; }
  for (uint r = 1; r <= rounds; ++r) {
    atomic_store_explicit(&f[0], r, memory_order_relaxed);
    uint k = 0;
    while (atomic_load_explicit(&f[1], memory_order_relaxed) < r &&
           ++k < spins) {}
    if (k >= spins) {
      atomic_fetch_add_explicit(&f[2], 1u, memory_order_relaxed);
      return;
    }
  }
}

// [G][n][R] -> [n][G * R]. grid {R, n, G}.
kernel void dsv41_ungroup(
    const device T* x [[buffer(0)]],
    device T*       y [[buffer(1)]],
    constant int&   n [[buffer(2)]],
    constant int&   R [[buffer(3)]],
    constant int&   G [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int r = (int)gid.x, i = (int)gid.y, g = (int)gid.z;
  if (r >= R) { return; }
  y[((long)i * G + g) * R + r] = x[((long)g * n + i) * R + r];
}

// ---- the compressor -----------------------------------------------------

// Compressor.forward above ratio 1: each group of `ratio` positions pooled
// into one latent, sum_j kv_j * softmax_j(score) per channel (f32). Group
// g0 + k pools positions (g0 + k) * ratio + j: a position before p0 is in
// the pending rows (slot pos % ratio), the rest are this chunk's. ks rows
// are [kv | score] (2D). grid {D, ng}.
kernel void dsv41_compress_pool(
    const device float* ks    [[buffer(0)]],    // [n, 2D]
    const device float* pend  [[buffer(1)]],    // [ratio, 2D]
    device float*       out   [[buffer(2)]],    // [ng, D]
    constant int&       D     [[buffer(3)]],
    constant int&       ratio [[buffer(4)]],
    constant int&       p0    [[buffer(5)]],
    constant int&       g0    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, k = (int)gid.y;
  if (d >= D) { return; }
  const int g = g0 + k;
  float mx = -INFINITY;
  for (int j = 0; j < ratio; ++j) {
    const int pos = g * ratio + j;
    const device float* r = pos < p0 ? pend + (long)(pos % ratio) * 2 * D
                                     : ks + (long)(pos - p0) * 2 * D;
    mx = max(mx, r[D + d]);
  }
  float num = 0.0f, den = 0.0f;
  for (int j = 0; j < ratio; ++j) {
    const int pos = g * ratio + j;
    const device float* r = pos < p0 ? pend + (long)(pos % ratio) * 2 * D
                                     : ks + (long)(pos - p0) * 2 * D;
    const float e = exp(r[D + d] - mx);
    den += e;
    num += e * r[d];
  }
  out[(long)k * D + d] = num / den;
}

// The open group's rows carried to the next chunk: positions [q0, p0 + n)
// into their slots. grid {2D, rows}.
kernel void dsv41_compress_keep(
    const device float* ks    [[buffer(0)]],
    device float*       pend  [[buffer(1)]],
    constant int&       D2    [[buffer(2)]],
    constant int&       ratio [[buffer(3)]],
    constant int&       p0    [[buffer(4)]],
    constant int&       q0    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, k = (int)gid.y;
  if (d >= D2) { return; }
  const int pos = q0 + k;
  pend[(long)(pos % ratio) * D2 + d] = ks[(long)(pos - p0) * D2 + d];
}

// ---- the indexer --------------------------------------------------------

// Indexer.forward's scores: S[i, t] = sum_h w[i, h] * relu(q[i, h] . k[t])
// over this chunk's queries and the compressed positions, -inf where a
// query cannot see t (t >= (p0 + i + 1) / ratio) or, under a candidate
// source, where t's block was not kept. w is weights_proj's output; wscale
// the reference's softmax_scale * heads^-0.5. One row's q in threadgroup
// memory, a thread a key. grid {256 * ceil(C/256), n}, tg {256}.
kernel void dsv41_index_score(
    const device T*     q        [[buffer(0)]],   // [n, IH, 128]
    const device T*     keys     [[buffer(1)]],   // [C, 128]
    const device T*     w        [[buffer(2)]],   // [n, IH]
    const device uchar* cand     [[buffer(3)]],   // [n, nblk]
    device float*       S        [[buffer(4)]],   // [n, C]
    constant int&       IH       [[buffer(5)]],
    constant int&       C        [[buffer(6)]],
    constant int&       p0       [[buffer(7)]],
    constant int&       ratio    [[buffer(8)]],
    constant float&     wscale   [[buffer(9)]],
    constant int&       use_cand [[buffer(10)]],
    constant int&       blk      [[buffer(11)]],
    constant int&       nblk     [[buffer(12)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]])
{
  const uint tid = tid2.x;
  constexpr int DI = 128;
  const int i = (int)tg.y;
  threadgroup float qs[32 * DI];
  threadgroup float ws[32];
  const device T* qr = q + (long)i * IH * DI;
  for (int k = (int)tid; k < IH * DI; k += 256) { qs[k] = (float)qr[k]; }
  for (int k = (int)tid; k < IH; k += 256) {
    ws[k] = (float)w[(long)i * IH + k] * wscale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = (int)tg.x * 256 + (int)tid;
  if (t >= C) { return; }
  const int vis = (p0 + i + 1) / ratio;
  float s = -INFINITY;
  const bool ok = t < vis &&
                  (use_cand == 0 || cand[(long)i * nblk + t / blk] != 0);
  if (ok) {
    float acc[32];
    for (int h = 0; h < IH; ++h) { acc[h] = 0.0f; }
    const device T* kr = keys + (long)t * DI;
    for (int d = 0; d < DI; ++d) {
      const float kd = (float)kr[d];
      for (int h = 0; h < IH; ++h) { acc[h] += qs[h * DI + d] * kd; }
    }
    s = 0.0f;
    for (int h = 0; h < IH; ++h) { s += ws[h] * max(acc[h], 0.0f); }
  }
  S[(long)i * C + t] = s;
}

// dsv41_index_score on simdgroup matrices: a row's q [32 heads, 128] times
// a tile of keys, 8x8x8 products in f32 -- the scalar kernel had a thread
// walk its key two bytes at a time (16K keys 250 us on an M5 Ultra, a
// third of a TFLOP/s). A threadgroup takes 64 keys (their positions in
// `pos`, -1 for none), a simdgroup 16 of them, staged a quarter of the
// head dimension at a time; each key's score is then sum_h w[h] relu(q_h .
// k), the heads in order, into out[j] for j < nout -- -inf where the key
// is not to be scored. IH <= 32 (padded to 32). tg {128}.
template <typename Ok>
METAL_FUNC void dsv41_iscore_tile_(
    const device T* qr, const device T* keys, const device T* wr,
    device float* out, int nout, threadgroup const int* pos, int IH,
    float wscale, Ok ok, threadgroup float* qs, threadgroup float* ws,
    threadgroup float (*kst)[512], uint tid, uint sg, uint lane)
{
  constexpr int DI = 128, HP = 32, SK = 16, DC = 32;
  for (int k = (int)tid; k < HP * DI; k += 128) {
    qs[k] = k / DI < IH ? (float)qr[k] : 0.0f;
  }
  for (int k = (int)tid; k < HP; k += 128) {
    ws[k] = k < IH ? (float)wr[k] * wscale : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int j0 = (int)sg * SK;            // this simdgroup's keys
  threadgroup float* ks = kst[sg];
  simdgroup_float8x8 acc[4][2];
  for (int hb = 0; hb < 4; ++hb) {
    for (int kb = 0; kb < 2; ++kb) {
      acc[hb][kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int c0 = 0; c0 < DI; c0 += DC) {
    // 16 keys x 32 dims: a lane takes half a key's 32
    {
      const int kk = (int)lane / 2, d0 = ((int)lane % 2) * 16;
      const int t = pos[j0 + kk];
      const device T* kr = keys + (long)max(t, 0) * DI + c0 + d0;
      for (int e = 0; e < 16; ++e) {
        ks[kk * DC + d0 + e] = t >= 0 ? (float)kr[e] : 0.0f;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < DC; kk += 8) {
      simdgroup_float8x8 km[2];
      for (int kb = 0; kb < 2; ++kb) {
        simdgroup_load(km[kb], ks + kb * 8 * DC + kk, DC, ulong2(0, 0), true);
      }
      for (int hb = 0; hb < 4; ++hb) {
        simdgroup_float8x8 qm;
        simdgroup_load(qm, qs + hb * 8 * DI + c0 + kk, DI);
        for (int kb = 0; kb < 2; ++kb) {
          simdgroup_multiply_accumulate(acc[hb][kb], qm, km[kb],
                                        acc[hb][kb]);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  // the 32 x 16 dots into the simdgroup's tile, then a key a lane
  for (int hb = 0; hb < 4; ++hb) {
    for (int kb = 0; kb < 2; ++kb) {
      simdgroup_store(acc[hb][kb], ks + hb * 8 * SK + kb * 8, SK);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < (uint)SK && j0 + (int)lane < nout) {
    const int j = j0 + (int)lane;
    const int t = pos[j];
    float s = -INFINITY;
    if (t >= 0 && ok(t)) {
      s = 0.0f;
      for (int h = 0; h < IH; ++h) {
        s += ws[h] * max(ks[h * SK + (int)lane], 0.0f);
      }
    }
    out[j] = s;
  }
}

// Every compressed position: a threadgroup the 64 from tg.x * 64 (a tile
// none of whose keys the query sees, or whose blocks were all dropped,
// only writes -inf). grid {128 * ceil(C/64), n}.
kernel void dsv41_index_score_mma(
    const device T*     q        [[buffer(0)]],   // [n, IH, 128]
    const device T*     keys     [[buffer(1)]],   // [C, 128]
    const device T*     w        [[buffer(2)]],   // [n, IH]
    const device uchar* cand     [[buffer(3)]],   // [n, nblk]
    device float*       S        [[buffer(4)]],   // [n, C]
    constant int&       IH       [[buffer(5)]],
    constant int&       C        [[buffer(6)]],
    constant int&       p0       [[buffer(7)]],
    constant int&       ratio    [[buffer(8)]],
    constant float&     wscale   [[buffer(9)]],
    constant int&       use_cand [[buffer(10)]],
    constant int&       blk      [[buffer(11)]],
    constant int&       nblk     [[buffer(12)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int DI = 128, TK = 64;
  threadgroup float qs[32 * DI];
  threadgroup float ws[32];
  threadgroup float kst[4][512];          // a simdgroup's keys, then sums
  threadgroup int pos[TK];
  threadgroup int live;
  const int i = (int)tg.y;
  const int t0 = (int)tg.x * TK;
  const int vis = (p0 + i + 1) / ratio;
  auto ok = [&](int t) {
    return t < C && t < vis &&
           (use_cand == 0 || cand[(long)i * nblk + t / blk] != 0);
  };
  if (tid < (uint)TK) {
    const int t = t0 + (int)tid;
    pos[tid] = t < C ? t : -1;
  }
  if (tid == 0) {
    // a block's keys share its fate, and the visible ones are a prefix
    int any = 0;
    for (int t = t0; t < min(C, t0 + TK) && any == 0; t += blk) {
      any = ok(t) ? 1 : 0;
    }
    live = any;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* out = S + (long)i * C + t0;
  if (live == 0) {
    for (int j = (int)tid; j < TK && t0 + j < C; j += 128) {
      out[j] = -INFINITY;
    }
    return;
  }
  dsv41_iscore_tile_(q + (long)i * IH * DI, keys, w + (long)i * IH, out,
                     C - t0, pos, IH, wscale, ok, qs, ws, kst, tid, sg,
                     lane);
}

// The positions a candidate source kept, alone: blocks `bpick` [n, KB]
// (ascending, -1 past the last) of `blk` keys each, into Sc [n, KB * blk]
// in that order -- what dsv41_index_score_mma would have scored, compacted
// in position order, so a top-k over Sc ties as one over S does. 64 slots
// a threadgroup. grid {128 * ceil(KB * blk / 64), n}.
kernel void dsv41_index_score_kept(
    const device T*     q        [[buffer(0)]],   // [n, IH, 128]
    const device T*     keys     [[buffer(1)]],   // [C, 128]
    const device T*     w        [[buffer(2)]],   // [n, IH]
    const device int*   bpick    [[buffer(3)]],   // [n, KB]
    device float*       Sc       [[buffer(4)]],   // [n, KB * blk]
    constant int&       IH       [[buffer(5)]],
    constant int&       C        [[buffer(6)]],
    constant int&       p0       [[buffer(7)]],
    constant int&       ratio    [[buffer(8)]],
    constant float&     wscale   [[buffer(9)]],
    constant int&       KB       [[buffer(10)]],
    constant int&       blk      [[buffer(11)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int DI = 128, TK = 64;
  threadgroup float qs[32 * DI];
  threadgroup float ws[32];
  threadgroup float kst[4][512];
  threadgroup int pos[TK];
  threadgroup int live;
  const int i = (int)tg.y;
  const int j0 = (int)tg.x * TK;          // the tile's first kept slot
  const int R = KB * blk;
  const int vis = (p0 + i + 1) / ratio;
  auto ok = [&](int t) { return t < C && t < vis; };
  if (tid < (uint)TK) {
    const int j = j0 + (int)tid;
    const int b = j < R ? bpick[(long)i * KB + j / blk] : -1;
    const int t = b >= 0 ? b * blk + j % blk : -1;
    pos[tid] = t >= 0 && t < C ? t : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    int any = 0;
    for (int j = 0; j < TK && any == 0; ++j) {
      any = pos[j] >= 0 && ok(pos[j]) ? 1 : 0;
    }
    live = any;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* out = Sc + (long)i * R + j0;
  if (live == 0) {
    for (int j = (int)tid; j < TK && j0 + j < R; j += 128) {
      out[j] = -INFINITY;
    }
    return;
  }
  dsv41_iscore_tile_(q + (long)i * IH * DI, keys, w + (long)i * IH, out,
                     R - j0, pos, IH, wscale, ok, qs, ws, kst, tid, sg,
                     lane);
}

// A top-k over dsv41_index_score_kept's slots back to positions, in place:
// slot c is block bpick[c / blk]'s key c % blk. The slots run in position
// order, so the picks stay ascending. grid {K, n}.
kernel void dsv41_index_unpack(
    device int*         idx   [[buffer(0)]],    // [n, K]
    const device int*   bpick [[buffer(1)]],    // [n, KB]
    constant int&       K     [[buffer(2)]],
    constant int&       KB    [[buffer(3)]],
    constant int&       blk   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int k = (int)gid.x, i = (int)gid.y;
  if (k >= K) { return; }
  device int* o = idx + (long)i * K + k;
  const int c = *o;
  if (c >= 0) { *o = bpick[(long)i * KB + c / blk] * blk + c % blk; }
}

// select_candidate_blocks' block scores: the best position of each block,
// the block holding the query's newest visible position pinned to +inf
// (a partly filled block must not lose to an older full one); -inf for a
// block the query cannot reach. grid {nblk, n}.
kernel void dsv41_block_scores(
    const device float* S     [[buffer(0)]],    // [n, C]
    device float*       B     [[buffer(1)]],    // [n, nblk]
    constant int&       C     [[buffer(2)]],
    constant int&       blk   [[buffer(3)]],
    constant int&       nblk  [[buffer(4)]],
    constant int&       p0    [[buffer(5)]],
    constant int&       ratio [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int b = (int)gid.x, i = (int)gid.y;
  if (b >= nblk) { return; }
  const int vis = (p0 + i + 1) / ratio;
  const int last = (vis - 1) / blk;
  float mx = -INFINITY;
  for (int t = b * blk; t < min(C, (b + 1) * blk); ++t) {
    mx = max(mx, S[(long)i * C + t]);
  }
  if (vis > 0 && b == last) { mx = INFINITY; }
  B[(long)i * nblk + b] = mx;
}

// The kept blocks as a mask: picks (top-k of B, -1 padded) with a finite
// score. grid {K, n}; the mask cleared first by the caller.
kernel void dsv41_block_mask(
    const device int*   picks [[buffer(0)]],    // [n, K]
    const device float* B     [[buffer(1)]],    // [n, nblk]
    device uchar*       mask  [[buffer(2)]],    // [n, nblk]
    constant int&       K     [[buffer(3)]],
    constant int&       nblk  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int k = (int)gid.x, i = (int)gid.y;
  if (k >= K) { return; }
  const int b = picks[(long)i * K + k];
  if (b < 0 || B[(long)i * nblk + b] == -INFINITY) { return; }
  mask[(long)i * nblk + b] = 1;
}

// Top-K of each row of S [n, C]: the indices of the K largest finite
// values (all of them when fewer are finite), ascending, -1 padded to K
// -- Indexer.forward's topk followed by its sort and its `< compress_lens`
// filter. The K-th value by an MSB-first radix select over the floats'
// order-preserving bits (4 passes of 8 bits); ties at it taken lowest
// index first. One row a threadgroup of 1024. grid {1024 * n}, tg {1024}.
inline uint dsv41_key_(float f)
{
  const uint b = as_type<uint>(f);
  return (b & 0x80000000u) != 0 ? ~b : (b | 0x80000000u);
}

kernel void dsv41_topk(
    const device float* S    [[buffer(0)]],     // [n, C]
    device int*         out  [[buffer(1)]],     // [n, K]
    constant int&       C    [[buffer(2)]],
    constant int&       K    [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
  constexpr int NT = 1024;
  const device float* s = S + (long)row * C;
  device int* o = out + (long)row * K;
  threadgroup atomic_uint hist[256];
  threadgroup uint sh_prefix, sh_need, sh_cnt;
  threadgroup uint part[NT], tpart[NT];
  const uint kNegInf = dsv41_key_(-INFINITY);
  // Count the finite entries.
  uint fin = 0;
  for (int t = (int)tid; t < C; t += NT) {
    fin += dsv41_key_(s[t]) > kNegInf ? 1u : 0u;
  }
  part[tid] = fin;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint tot = 0;
    for (int k = 0; k < NT; ++k) { tot += part[k]; }
    sh_cnt = tot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint finite = sh_cnt;
  uint thr = kNegInf + 1;            // keep keys >= thr (all finite)
  uint ties = 0;                     // how many == thr to take
  bool exact = false;
  if (finite > (uint)K) {
    uint prefix = 0, need = (uint)K;
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      const uint hi_mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (int b = (int)tid; b < 256; b += NT) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int t = (int)tid; t < C; t += NT) {
        const uint k = dsv41_key_(s[t]);
        if (k > kNegInf && (k & hi_mask) == prefix) {
          atomic_fetch_add_explicit(&hist[(k >> shift) & 0xFFu], 1u,
                                    memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) {
        uint acc = 0;
        int b = 255;
        for (; b > 0; --b) {
          const uint c = atomic_load_explicit(&hist[b],
                                              memory_order_relaxed);
          if (acc + c >= need) { break; }
          acc += c;
        }
        sh_prefix = prefix | ((uint)b << shift);
        sh_need = need - acc;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = sh_prefix;
      need = sh_need;
    }
    thr = prefix;                    // the K-th largest key exactly
    ties = need;                     // how many of the == thr to keep
    exact = true;
  }
  // Compaction in index order: a contiguous range of t per thread. A
  // thread's first output slot is the elements taken before its range:
  // every greater one, and the ties up to `ties`, lowest index first.
  const int per = (C + NT - 1) / NT;
  const int lo = min(C, (int)tid * per), hi = min(C, lo + per);
  uint gt = 0, eq = 0;
  for (int t = lo; t < hi; ++t) {
    const uint k = dsv41_key_(s[t]);
    gt += k > thr || (!exact && k == thr) ? 1u : 0u;
    eq += exact && k == thr ? 1u : 0u;
  }
  part[tid] = gt;
  tpart[tid] = eq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint a = 0, b = 0;
    for (int k = 0; k < NT; ++k) {
      const uint x = part[k], y = tpart[k];
      part[k] = a;
      tpart[k] = b;
      a += x;
      b += y;
    }
    sh_cnt = a + min(b, ties);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint erank = tpart[tid];
  uint at = part[tid] + min(erank, ties);
  for (int t = lo; t < hi; ++t) {
    const uint k = dsv41_key_(s[t]);
    bool take = k > thr || (!exact && k == thr);
    if (exact && k == thr) {
      take = erank < ties;
      ++erank;
    }
    if (take && at < (uint)K) { o[at++] = t; }
  }
  for (int k = (int)sh_cnt + (int)tid; k < K; k += NT) { o[k] = -1; }
}

// dsv41_topk with its serial steps made parallel, the same picks to the
// index: the finite count and the compaction's offsets by simdgroup prefix
// sums (one thread had summed 1024 partials, twice), each radix pass's
// bucket found by a parallel suffix sum over the 256 bins (one thread had
// scanned them), and the histogram split eight ways (a pass's top byte is
// mostly the exponent, so every thread had hit the same few counters).
// grid {1024 * n}, tg {1024}.
kernel void dsv41_topk_par(
    const device float* S    [[buffer(0)]],     // [n, C]
    device int*         out  [[buffer(1)]],     // [n, K]
    constant int&       C    [[buffer(2)]],
    constant int&       K    [[buffer(3)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  constexpr int NT = 1024, NH = 8;
  const device float* s = S + (long)row * C;
  device int* o = out + (long)row * K;
  threadgroup atomic_uint hist[NH * 256];
  threadgroup uint cnt[256];
  threadgroup uint sums[32], sums2[32];
  threadgroup uint sh_prefix, sh_need;
  const uint kNegInf = dsv41_key_(-INFINITY);
  // the threadgroup's total of one value a thread, and each thread's
  // exclusive prefix (in thread order)
  auto scan = [&](uint v, threadgroup uint* part, thread uint& before) {
    const uint x = simd_prefix_exclusive_sum(v);
    if (lane == 31) { part[sg] = x + v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lo = 0, tot = 0;
    for (uint j = 0; j < 32; ++j) {
      lo += j < sg ? part[j] : 0u;
      tot += part[j];
    }
    before = lo + x;
    return tot;
  };
  uint fin = 0;
  for (int t = (int)tid; t < C; t += NT) {
    fin += dsv41_key_(s[t]) > kNegInf ? 1u : 0u;
  }
  uint ignore = 0;
  const uint finite = scan(fin, sums, ignore);
  uint thr = kNegInf + 1;
  uint ties = 0;
  bool exact = false;
  if (finite > (uint)K) {
    uint prefix = 0, need = (uint)K;
    threadgroup atomic_uint* hp = hist + (sg % NH) * 256;
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      const uint hi_mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (int b = (int)tid; b < NH * 256; b += NT) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int t = (int)tid; t < C; t += NT) {
        const uint k = dsv41_key_(s[t]);
        if (k > kNegInf && (k & hi_mask) == prefix) {
          atomic_fetch_add_explicit(&hp[(k >> shift) & 0xFFu], 1u,
                                    memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 256) {
        uint c = 0;
        for (int j = 0; j < NH; ++j) {
          c += atomic_load_explicit(&hist[j * 256 + (int)tid],
                                    memory_order_relaxed);
        }
        cnt[tid] = c;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // bins from the highest down: a thread's bin 255 - tid, and the
      // count above it
      const uint mine = tid < 256 ? cnt[255 - tid] : 0u;
      uint above = 0;
      (void)scan(mine, sums2, above);
      if (tid < 256 && above < need && need <= above + mine) {
        sh_prefix = prefix | ((255u - tid) << shift);
        sh_need = need - above;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = sh_prefix;
      need = sh_need;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    thr = prefix;
    ties = need;
    exact = true;
  }
  // compaction in index order, a contiguous range a thread, as dsv41_topk
  const int per = (C + NT - 1) / NT;
  const int lo = min(C, (int)tid * per), hi = min(C, lo + per);
  uint gt = 0, eq = 0;
  for (int t = lo; t < hi; ++t) {
    const uint k = dsv41_key_(s[t]);
    gt += k > thr || (!exact && k == thr) ? 1u : 0u;
    eq += exact && k == thr ? 1u : 0u;
  }
  uint gbefore = 0, ebefore = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);   // `sums` read above
  const uint gtot = scan(gt, sums, gbefore);
  (void)scan(eq, sums2, ebefore);
  uint erank = ebefore;
  uint at = gbefore + min(erank, ties);
  for (int t = lo; t < hi; ++t) {
    const uint k = dsv41_key_(s[t]);
    bool take = k > thr || (!exact && k == thr);
    if (exact && k == thr) {
      take = erank < ties;
      ++erank;
    }
    if (take && at < (uint)K) { o[at++] = t; }
  }
  // what was kept: every greater one, and the ties taken
  uint etot = 0;
  for (uint j = 0; j < 32; ++j) { etot += sums2[j]; }
  const uint kept = gtot + min(etot, ties);
  for (int k = (int)kept + (int)tid; k < K; k += NT) { o[k] = -1; }
}

// ---- the two-phase top-k (a decode or verify step's few rows) -----------
//
// dsv41_topk_par gives a row one threadgroup, which a prefill's thousands
// of rows fill the GPU with and a decode step's one row does not: at 128K
// positions it is 150 us on one core of an M5 Ultra. Here a row's chunks of
// DSV41_TOPK_CHUNK positions each take their own top K on their own core
// (every pick of the row's top K is one of its chunk's: anything that beats
// it in the chunk beats it in the row), then one threadgroup takes the top
// K of their union -- of the candidates at or above the largest chunk's
// K-th key, a bound the row's K-th key cannot be under, and few above it.
// The same picks as dsv41_topk_par, ties included.

#define DSV41_TOPK_CHUNK 4096

// A threadgroup of NT's total of one value a thread, and each thread's
// exclusive prefix in thread order (`part`, NT / 32 words, is read after
// the barrier: a caller does not write it again before another one).
template <int NT>
METAL_FUNC uint dsv41_tg_scan_(uint v, threadgroup uint* part,
                               thread uint& before, uint sg, uint lane)
{
  const uint x = simd_prefix_exclusive_sum(v);
  if (lane == 31) { part[sg] = x + v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint lo = 0, tot = 0;
  for (uint j = 0; j < (uint)NT / 32; ++j) {
    lo += j < sg ? part[j] : 0u;
    tot += part[j];
  }
  before = lo + x;
  return tot;
}

// The top K of the entries j < L whose key (keyf(j)) is at least lo, ties
// at the K-th key taken lowest j first, each handed to emit(at, j) with
// `at` ascending in j -- every such entry when there are K or fewer.
// Returns how many, and the K-th key in kth (0 when every one was taken).
// dsv41_topk_par's radix select, its histograms NH deep. Every thread of a
// threadgroup of NT (256 or more) calls it. (256 threads measured
// quicker than 1024 here: 61 against 69 us for 512 of 64K.)
template <int NT, int NH, typename KeyF, typename Emit>
METAL_FUNC uint dsv41_select_(
    int L, KeyF keyf, uint lo, int K, Emit emit, threadgroup atomic_uint* hist,
    threadgroup uint* cnt, threadgroup uint* sums, threadgroup uint* sums2,
    threadgroup uint* shv, thread uint& kth, uint tid, uint sg, uint lane)
{
  uint have = 0;
  for (int j = (int)tid; j < L; j += NT) { have += keyf(j) >= lo ? 1u : 0u; }
  uint ignore = 0;
  const uint total = dsv41_tg_scan_<NT>(have, sums, ignore, sg, lane);
  uint thr = lo, ties = 0;
  bool exact = false;
  if (total > (uint)K) {
    uint prefix = 0, need = (uint)K;
    threadgroup atomic_uint* hp = hist + (sg % NH) * 256;
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      const uint hi_mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (int b = (int)tid; b < NH * 256; b += NT) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int j = (int)tid; j < L; j += NT) {
        const uint k = keyf(j);
        if (k >= lo && (k & hi_mask) == prefix) {
          atomic_fetch_add_explicit(&hp[(k >> shift) & 0xFFu], 1u,
                                    memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int t = (int)tid; t < 256; t += NT) {
        uint c = 0;
        for (int j = 0; j < NH; ++j) {
          c += atomic_load_explicit(&hist[j * 256 + t],
                                    memory_order_relaxed);
        }
        cnt[t] = c;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // bins from the highest down: a thread's bin 255 - tid, and the
      // count above it
      const uint mine = tid < 256 ? cnt[255 - tid] : 0u;
      uint above = 0;
      (void)dsv41_tg_scan_<NT>(mine, sums2, above, sg, lane);
      if (tid < 256 && above < need && need <= above + mine) {
        shv[0] = prefix | ((255u - tid) << shift);
        shv[1] = need - above;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix = shv[0];
      need = shv[1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    thr = prefix;
    ties = need;
    exact = true;
  }
  // compaction in index order, a contiguous range a thread
  const int per = (L + NT - 1) / NT;
  const int a = min(L, (int)tid * per), b = min(L, a + per);
  uint gt = 0, eq = 0;
  for (int j = a; j < b; ++j) {
    const uint k = keyf(j);
    gt += (exact ? k > thr : k >= lo) ? 1u : 0u;
    eq += exact && k == thr ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // `sums` read above
  uint gbefore = 0, ebefore = 0;
  const uint gtot = dsv41_tg_scan_<NT>(gt, sums, gbefore, sg, lane);
  (void)dsv41_tg_scan_<NT>(eq, sums2, ebefore, sg, lane);
  uint erank = ebefore;
  uint at = gbefore + min(erank, ties);
  for (int j = a; j < b; ++j) {
    const uint k = keyf(j);
    bool take = exact ? k > thr : k >= lo;
    if (exact && k == thr) {
      take = erank < ties;
      ++erank;
    }
    if (take && at < (uint)K) { emit(at++, j); }
  }
  kth = exact ? thr : 0u;
  return exact ? (uint)K : min(gtot, (uint)K);
}

// Phase one: chunk g of row i its own top K of the finite entries, in
// index order, keys into ck and positions into cp [n, G, K] (key 0 and -1
// past its last), its K-th key into tk [n, G] (0 when it holds K or
// fewer), and the keys of its top R, in no order, into tr [n, G, R] (0
// past them). grid {NT * G, n}, tg {NT}.
template <int NT>
kernel void dsv41_topk_chunk_(
    const device float* S  [[buffer(0)]],     // [n, C]
    device uint*        ck [[buffer(1)]],     // [n, G, K]
    device int*         cp [[buffer(2)]],     // [n, G, K]
    device uint*        tk [[buffer(3)]],     // [n, G]
    device uint*        tr [[buffer(4)]],     // [n, G, R]
    constant int&       C  [[buffer(5)]],
    constant int&       K  [[buffer(6)]],
    constant int&       R  [[buffer(7)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  constexpr int CH = DSV41_TOPK_CHUNK, NH = 8;
  threadgroup uint keys[CH];
  threadgroup atomic_uint hist[NH * 256];
  threadgroup uint cnt[256], sums[32], sums2[32], shv[2];
  const int G = (C + CH - 1) / CH;
  const int g = (int)tg.x, i = (int)tg.y;
  const int t0 = g * CH, L = min(CH, C - t0);
  const device float* s = S + (long)i * C + t0;
  for (int j = (int)tid; j < L; j += NT) { keys[j] = dsv41_key_(s[j]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup uint* kp = keys;
  const uint fin = dsv41_key_(-INFINITY) + 1u;
  device uint* ok = ck + ((long)i * G + g) * K;
  device int* op = cp + ((long)i * G + g) * K;
  device uint* orr = tr + ((long)i * G + g) * R;
  uint kth = 0;
  const uint kept = dsv41_select_<NT, NH>(
      L, [kp](int j) { return kp[j]; }, fin, K,
      [kp, ok, op, t0](uint at, int j) {
        ok[at] = kp[j];
        op[at] = t0 + j;
      },
      hist, cnt, sums, sums2, shv, kth, tid, sg, lane);
  for (int k = (int)kept + (int)tid; k < K; k += NT) {
    ok[k] = 0u;
    op[k] = -1;
  }
  if (tid == 0) { tk[(long)i * G + g] = kth; }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // `sums` read above
  uint rth = 0;
  const uint rk = dsv41_select_<NT, NH>(
      L, [kp](int j) { return kp[j]; }, fin, R,
      [kp, orr](uint at, int j) { orr[at] = kp[j]; }, hist, cnt, sums,
      sums2, shv, rth, tid, sg, lane);
  for (int k = (int)rk + (int)tid; k < R; k += NT) { orr[k] = 0u; }
}

// Phase two: a row's top K of its chunks' candidates -- those at or above
// a bound the row's K-th key cannot be under (the largest chunk K-th key,
// and the K-th largest of the chunks' top R: a subset's), gathered in
// order into threadgroup memory when they fit (else read where they lie)
// -- into out [n, K], ascending, -1 padded as dsv41_topk_par.
// G * R <= 3072. grid {NT * n}, tg {NT}.
template <int NT>
kernel void dsv41_topk_merge_(
    const device uint* ck  [[buffer(0)]],     // [n, G, K]
    const device int*  cp  [[buffer(1)]],     // [n, G, K]
    const device uint* tk  [[buffer(2)]],     // [n, G]
    const device uint* tr  [[buffer(3)]],     // [n, G, R]
    device int*        out [[buffer(4)]],     // [n, K]
    constant int&      C   [[buffer(5)]],
    constant int&      K   [[buffer(6)]],
    constant int&      R   [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  constexpr int CH = DSV41_TOPK_CHUNK, NH = 4, CAP = 3072;
  threadgroup uint lk[CAP];
  threadgroup int lp[CAP];
  threadgroup atomic_uint hist[NH * 256];
  threadgroup uint cnt[256], sums[32], sums2[32], shv[2], tmax[32];
  const int G = (C + CH - 1) / CH, M = G * K, GR = G * R;
  const device uint* rk = ck + (long)row * M;
  const device int* rp = cp + (long)row * M;
  device int* o = out + (long)row * K;
  const uint fin = dsv41_key_(-INFINITY) + 1u;
  // the bounds: the largest chunk K-th key ...
  uint t = 0;
  for (int g = (int)tid; g < G; g += NT) {
    t = max(t, tk[(long)row * G + g]);
  }
  t = simd_max(t);
  if (lane == 0) { tmax[sg] = t; }
  // ... and the K-th largest of every chunk's top R
  for (int j = (int)tid; j < GR; j += NT) { lk[j] = tr[(long)row * GR + j]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint lo = fin;
  for (int j = 0; j < NT / 32; ++j) { lo = max(lo, tmax[j]); }
  {
    threadgroup uint* kp = lk;
    uint uk = 0;
    (void)dsv41_select_<NT, NH>(GR, [kp](int j) { return kp[j]; }, fin, K,
                            [](uint, int) {}, hist, cnt, sums, sums2, shv,
                            uk, tid, sg, lane);
    lo = max(lo, uk);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // lk, `sums` reused
  const int per = (M + NT - 1) / NT;
  const int a = min(M, (int)tid * per), b = min(M, a + per);
  uint mine = 0;
  for (int j = a; j < b; ++j) { mine += rk[j] >= lo ? 1u : 0u; }
  uint before = 0;
  const uint total = dsv41_tg_scan_<NT>(mine, sums, before, sg, lane);
  uint kept = 0, kth = 0;
  if (total <= (uint)CAP) {
    uint at = before;
    for (int j = a; j < b; ++j) {
      const uint k = rk[j];
      if (k >= lo) {
        lk[at] = k;
        lp[at] = rp[j];
        ++at;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* kp = lk;
    threadgroup int* pp = lp;
    kept = dsv41_select_<NT, NH>(
        (int)total, [kp](int j) { return kp[j]; }, lo, K,
        [o, pp](uint at2, int j) { o[at2] = pp[j]; }, hist, cnt, sums,
        sums2, shv, kth, tid, sg, lane);
  } else {
    threadgroup_barrier(mem_flags::mem_threadgroup);  // `sums` read above
    kept = dsv41_select_<NT, NH>(
        M, [rk](int j) { return rk[j]; }, lo, K,
        [o, rp](uint at2, int j) { o[at2] = rp[j]; }, hist, cnt, sums,
        sums2, shv, kth, tid, sg, lane);
  }
  for (int k = (int)kept + (int)tid; k < K; k += NT) { o[k] = -1; }
}

#define DSV41_TOPK_TWO(NT, SUF)                                             \
  template [[host_name("dsv41_topk_chunk" SUF)]] kernel void                \
  dsv41_topk_chunk_<NT>(const device float*, device uint*, device int*,    \
                        device uint*, device uint*, constant int&,         \
                        constant int&, constant int&, uint2, uint, uint,   \
                        uint);                                             \
  template [[host_name("dsv41_topk_merge" SUF)]] kernel void                \
  dsv41_topk_merge_<NT>(const device uint*, const device int*,             \
                        const device uint*, const device uint*,            \
                        device int*, constant int&, constant int&,         \
                        constant int&, uint, uint, uint, uint);
DSV41_TOPK_TWO(256, "")

// ---- the router -----------------------------------------------------------

// Gate.forward with score_func sqrtsoftplus: s = sqrt(softplus(score)),
// the top-k of s + bias (the bias picks, it does not weigh), weights the
// picked s normalized (+1e-20, as training) and scaled. `vl` rows (an
// image's) select with bias_vl. One row a threadgroup of 256.
// grid {256 * n}, tg {256}.
kernel void dsv41_route(
    const device float* sc      [[buffer(0)]],  // [n, E]
    const device float* bias    [[buffer(1)]],  // [E]
    const device float* bias_vl [[buffer(2)]],  // [E]
    const device uchar* vl      [[buffer(3)]],  // [n] (has_vl)
    device int*         eid     [[buffer(4)]],  // [n, k]
    device float*       wt      [[buffer(5)]],  // [n, k]
    constant int&       E       [[buffer(6)]],
    constant int&       topk    [[buffer(7)]],
    constant float&     rscale  [[buffer(8)]],
    constant int&       norm    [[buffer(9)]],
    constant int&       has_vl  [[buffer(10)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  threadgroup float val[1024], sel[1024];
  threadgroup float rv[8];
  threadgroup int ri[8];
  threadgroup int pick[16];
  dsv41_route_tg_(sc, bias, bias_vl, vl, eid, wt, E, topk, rscale, norm,
                  has_vl, row, tid, sg, lane, val, sel, rv, ri, pick);
}

// ---- the experts (E2M1 codes, E8M0 scales per 32 along K) --------------

constant constexpr float kFp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float dsv41_e8m0_(uchar e)
{
  return as_type<float>((uint)e << 23);     // 2^(e - 127); e < 255
}

// Eight codes of one uint (low nibble first) dotted with x[0..8).
// Two E2M1 nibbles (bits 0-3, 4-7 of v) as a half2. E2M1 is IEEE-like --
// 1 sign, 2 exponent bits with bias 1, 1 mantissa bit, subnormals -- so a
// nibble's bits placed where a half keeps them (sign at 15, exponent and
// mantissa at 11..9) are its value times 2^-14, exactly, zero and the
// subnormal 0.5 included.
inline half2 dsv41_fp4x2_(uint v)
{
  return as_type<half2>(ushort2((ushort)(((v & 0x8u) << 12) |
                                         ((v & 0x7u) << 9)),
                                (ushort)(((v & 0x80u) << 8) |
                                         ((v & 0x70u) << 5))));
}

// Eight codes of one uint (low nibble first) dotted with x[0..8), in the
// codes' order. The values come in at 2^-14 of themselves (dsv41_fp4x2_),
// which scales every product and partial sum exactly, and the sum is
// scaled back at the end: the table lookup's arithmetic, without its
// eight divergent constant reads.
inline float dsv41_fp4_dot8_(uint u, thread const float* x)
{
  const half2 h0 = dsv41_fp4x2_(u), h1 = dsv41_fp4x2_(u >> 8);
  const half2 h2 = dsv41_fp4x2_(u >> 16), h3 = dsv41_fp4x2_(u >> 24);
  float s = 0.0f;
  s += (float)h0.x * x[0];
  s += (float)h0.y * x[1];
  s += (float)h1.x * x[2];
  s += (float)h1.y * x[3];
  s += (float)h2.x * x[4];
  s += (float)h2.y * x[5];
  s += (float)h3.x * x[6];
  s += (float)h3.y * x[7];
  return s * 16384.0f;
}

// Expert.forward's first half for (token, expert) pairs: gate = w1 . x,
// up = w3 . x, the training clamps (up to [-lim, lim], gate from above),
// silu(gate) * up times the pair's routing weight, rounded to the element
// type (the reference weights before w2). The pool holds slot s's w1 codes
// [I, K/2] at s * I * K/2 (part 0), its scales [I, K/32] (part 1), w3 the
// same (parts 2, 3). Pair p reads x row p / topk and slot slot[p]. A lane
// 8 codes a step (256 a simdgroup), 2 features (4 rows) a simdgroup.
// grid {32, I / 4, np} threads, tg {32, 2, 1}.
kernel void dsv41_moe_swiglu_fp4(
    const device uchar* w1    [[buffer(0)]],
    const device uchar* s1    [[buffer(1)]],
    const device uchar* w3    [[buffer(2)]],
    const device uchar* s3    [[buffer(3)]],
    const device T*     x     [[buffer(4)]],    // [rows, K]
    device T*           act   [[buffer(5)]],    // [np, I]
    const device int*   slot  [[buffer(6)]],    // [np]
    const device float* wt    [[buffer(7)]],    // [np]
    constant int&       K     [[buffer(8)]],
    constant int&       I     [[buffer(9)]],
    constant int&       topk  [[buffer(10)]],
    constant float&     lim   [[buffer(11)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int p = (int)tg.z;
  const long e = slot[p];
  const int f0 = ((int)tg.y * 2 + (int)sg) * 2;   // two features
  if (f0 >= I) { return; }
  const int KB = K / 2, KS = K / 32;
  const device T* xr = x + (long)(p / topk) * K;
  float g[2] = {0.0f, 0.0f}, u[2] = {0.0f, 0.0f};
  for (int k = (int)lane * 8; k < K; k += 256) {
    float xv[8];
    for (int j = 0; j < 8; ++j) { xv[j] = (float)xr[k + j]; }
    for (int r = 0; r < 2; ++r) {
      const long row = e * I + f0 + r;
      const uint c1 = *reinterpret_cast<const device uint*>(
          w1 + row * KB + k / 2);
      const uint c3 = *reinterpret_cast<const device uint*>(
          w3 + row * KB + k / 2);
      g[r] += dsv41_e8m0_(s1[row * KS + k / 32]) * dsv41_fp4_dot8_(c1, xv);
      u[r] += dsv41_e8m0_(s3[row * KS + k / 32]) * dsv41_fp4_dot8_(c3, xv);
    }
  }
  for (int r = 0; r < 2; ++r) {
    g[r] = simd_sum(g[r]);
    u[r] = simd_sum(u[r]);
  }
  if (lane == 0) {
    const float w = wt[p];
    for (int r = 0; r < 2; ++r) {
      float gg = g[r], uu = u[r];
      if (lim > 0.0f) {
        uu = clamp(uu, -lim, lim);
        gg = min(gg, lim);
      }
      const float a = gg / (1.0f + exp(-gg)) * uu;
      act[(long)p * I + f0 + r] = (T)(w * a);
    }
  }
}

// Expert.forward's second half: out[p] = w2 . act[p] (part 4 codes [H,
// I/2], part 5 scales [H, I/32]), rounded to the element type (the
// reference's w2 output). 4 rows a simdgroup. grid {32, H / 8, np}
// threads, tg {32, 2, 1}.
kernel void dsv41_moe_down_fp4(
    const device uchar* w2   [[buffer(0)]],
    const device uchar* s2   [[buffer(1)]],
    const device T*     act  [[buffer(2)]],     // [np, I]
    device T*           out  [[buffer(3)]],     // [np, H]
    const device int*   slot [[buffer(4)]],
    constant int&       I    [[buffer(5)]],
    constant int&       H    [[buffer(6)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int p = (int)tg.z;
  const long e = slot[p];
  const int r0 = ((int)tg.y * 2 + (int)sg) * 4;
  if (r0 >= H) { return; }
  const int KB = I / 2, KS = I / 32;
  const device T* ar = act + (long)p * I;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = (int)lane * 8; k < I; k += 256) {
    float xv[8];
    for (int j = 0; j < 8; ++j) { xv[j] = (float)ar[k + j]; }
    for (int r = 0; r < 4; ++r) {
      const long row = e * H + r0 + r;
      const uint c = *reinterpret_cast<const device uint*>(
          w2 + row * KB + k / 2);
      acc[r] += dsv41_e8m0_(s2[row * KS + k / 32]) * dsv41_fp4_dot8_(c, xv);
    }
  }
  for (int r = 0; r < 4; ++r) { acc[r] = simd_sum(acc[r]); }
  if (lane == 0) {
    for (int r = 0; r < 4; ++r) {
      out[(long)p * H + r0 + r] = (T)acc[r];
    }
  }
}

// The two kernels above for a few rows (a speculative verify), each
// expert's weights read ONCE for every pair that picked it: group g is
// gtab[g * 10 ..] = {slot, count, pair 0, ..., pair count-1} (count <= MM;
// a larger group is split). MM 2, 4 and 8.
// Every pair's sums are formed exactly as the one-pair kernels form them --
// the same steps, the same lane order, the same reduction -- so a verify's
// rows are the decode's to the bit. grid {32, I / 4, groups} and {32,
// H / 8, groups} threads, tg {32, 2, 1}.
template <int MM>
inline void dsv41_moe_swiglu_fp4_rows_impl_(
    const device uchar* w1,
    const device uchar* s1,
    const device uchar* w3,
    const device uchar* s3,
    const device T*     x,      // [rows, K]
    device T*           act,    // [np, I]
    const device int*   gtab,
    const device float* wt,     // [np]
    constant int&       K,
    constant int&       I,
    constant int&       topk,
    constant float&     lim,
    uint3 tg, uint sg, uint lane)
{
  const device int* gr = gtab + (long)tg.z * 10;
  const long e = gr[0];
  const int cnt = gr[1];
  const int f0 = ((int)tg.y * 2 + (int)sg) * 2;   // two features
  if (f0 >= I) { return; }
  const int KB = K / 2, KS = K / 32;
  float g[MM][2], u[MM][2];
  for (int m = 0; m < MM; ++m) {
    g[m][0] = g[m][1] = u[m][0] = u[m][1] = 0.0f;
  }
  for (int k = (int)lane * 8; k < K; k += 256) {
    uint c1[2], c3[2];
    float a1[2], a3[2];
    for (int r = 0; r < 2; ++r) {
      const long row = e * I + f0 + r;
      c1[r] = *reinterpret_cast<const device uint*>(w1 + row * KB + k / 2);
      c3[r] = *reinterpret_cast<const device uint*>(w3 + row * KB + k / 2);
      a1[r] = dsv41_e8m0_(s1[row * KS + k / 32]);
      a3[r] = dsv41_e8m0_(s3[row * KS + k / 32]);
    }
    for (int m = 0; m < MM; ++m) {
      if (m >= cnt) { break; }
      const device T* xr = x + (long)(gr[2 + m] / topk) * K;
      float xv[8];
      for (int j = 0; j < 8; ++j) { xv[j] = (float)xr[k + j]; }
      for (int r = 0; r < 2; ++r) {
        g[m][r] += a1[r] * dsv41_fp4_dot8_(c1[r], xv);
        u[m][r] += a3[r] * dsv41_fp4_dot8_(c3[r], xv);
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    if (m >= cnt) { break; }
    const int p = gr[2 + m];
    for (int r = 0; r < 2; ++r) {
      g[m][r] = simd_sum(g[m][r]);
      u[m][r] = simd_sum(u[m][r]);
    }
    if (lane == 0) {
      const float w = wt[p];
      for (int r = 0; r < 2; ++r) {
        float gg = g[m][r], uu = u[m][r];
        if (lim > 0.0f) {
          uu = clamp(uu, -lim, lim);
          gg = min(gg, lim);
        }
        const float a = gg / (1.0f + exp(-gg)) * uu;
        act[(long)p * I + f0 + r] = (T)(w * a);
      }
    }
  }
}

template <int MM>
inline void dsv41_moe_down_fp4_rows_impl_(
    const device uchar* w2,
    const device uchar* s2,
    const device T*     act,    // [np, I]
    device T*           out,    // [np, H]
    const device int*   gtab,
    constant int&       I,
    constant int&       H,
    uint3 tg, uint sg, uint lane)
{
  const device int* gr = gtab + (long)tg.z * 10;
  const long e = gr[0];
  const int cnt = gr[1];
  const int r0 = ((int)tg.y * 2 + (int)sg) * 4;
  if (r0 >= H) { return; }
  const int KB = I / 2, KS = I / 32;
  float acc[MM][4];
  for (int m = 0; m < MM; ++m) {
    for (int r = 0; r < 4; ++r) { acc[m][r] = 0.0f; }
  }
  for (int k = (int)lane * 8; k < I; k += 256) {
    uint c[4];
    float a[4];
    for (int r = 0; r < 4; ++r) {
      const long row = e * H + r0 + r;
      c[r] = *reinterpret_cast<const device uint*>(w2 + row * KB + k / 2);
      a[r] = dsv41_e8m0_(s2[row * KS + k / 32]);
    }
    for (int m = 0; m < MM; ++m) {
      if (m >= cnt) { break; }
      const device T* ar = act + (long)gr[2 + m] * I;
      float xv[8];
      for (int j = 0; j < 8; ++j) { xv[j] = (float)ar[k + j]; }
      for (int r = 0; r < 4; ++r) {
        acc[m][r] += a[r] * dsv41_fp4_dot8_(c[r], xv);
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    if (m >= cnt) { break; }
    for (int r = 0; r < 4; ++r) { acc[m][r] = simd_sum(acc[m][r]); }
    if (lane == 0) {
      const long p = gr[2 + m];
      for (int r = 0; r < 4; ++r) { out[p * H + r0 + r] = (T)acc[m][r]; }
    }
  }
}

kernel void dsv41_moe_swiglu_fp4_rows2(
    const device uchar* w1 [[buffer(0)]], const device uchar* s1 [[buffer(1)]],
    const device uchar* w3 [[buffer(2)]], const device uchar* s3 [[buffer(3)]],
    const device T* x [[buffer(4)]], device T* act [[buffer(5)]],
    const device int* gtab [[buffer(6)]], const device float* wt [[buffer(7)]],
    constant int& K [[buffer(8)]], constant int& I [[buffer(9)]],
    constant int& topk [[buffer(10)]], constant float& lim [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  dsv41_moe_swiglu_fp4_rows_impl_<2>(w1, s1, w3, s3, x, act, gtab, wt, K, I,
                                      topk, lim, tg, sg, lane);
}

kernel void dsv41_moe_down_fp4_rows2(
    const device uchar* w2 [[buffer(0)]], const device uchar* s2 [[buffer(1)]],
    const device T* act [[buffer(2)]], device T* out [[buffer(3)]],
    const device int* gtab [[buffer(4)]], constant int& I [[buffer(5)]],
    constant int& H [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  dsv41_moe_down_fp4_rows_impl_<2>(w2, s2, act, out, gtab, I, H, tg, sg,
                                    lane);
}

kernel void dsv41_moe_swiglu_fp4_rows4(
    const device uchar* w1 [[buffer(0)]], const device uchar* s1 [[buffer(1)]],
    const device uchar* w3 [[buffer(2)]], const device uchar* s3 [[buffer(3)]],
    const device T* x [[buffer(4)]], device T* act [[buffer(5)]],
    const device int* gtab [[buffer(6)]], const device float* wt [[buffer(7)]],
    constant int& K [[buffer(8)]], constant int& I [[buffer(9)]],
    constant int& topk [[buffer(10)]], constant float& lim [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  dsv41_moe_swiglu_fp4_rows_impl_<4>(w1, s1, w3, s3, x, act, gtab, wt, K, I,
                                      topk, lim, tg, sg, lane);
}

kernel void dsv41_moe_down_fp4_rows4(
    const device uchar* w2 [[buffer(0)]], const device uchar* s2 [[buffer(1)]],
    const device T* act [[buffer(2)]], device T* out [[buffer(3)]],
    const device int* gtab [[buffer(4)]], constant int& I [[buffer(5)]],
    constant int& H [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  dsv41_moe_down_fp4_rows_impl_<4>(w2, s2, act, out, gtab, I, H, tg, sg,
                                    lane);
}

kernel void dsv41_moe_swiglu_fp4_rows8(
    const device uchar* w1 [[buffer(0)]], const device uchar* s1 [[buffer(1)]],
    const device uchar* w3 [[buffer(2)]], const device uchar* s3 [[buffer(3)]],
    const device T* x [[buffer(4)]], device T* act [[buffer(5)]],
    const device int* gtab [[buffer(6)]], const device float* wt [[buffer(7)]],
    constant int& K [[buffer(8)]], constant int& I [[buffer(9)]],
    constant int& topk [[buffer(10)]], constant float& lim [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  dsv41_moe_swiglu_fp4_rows_impl_<8>(w1, s1, w3, s3, x, act, gtab, wt, K, I,
                                      topk, lim, tg, sg, lane);
}

kernel void dsv41_moe_down_fp4_rows8(
    const device uchar* w2 [[buffer(0)]], const device uchar* s2 [[buffer(1)]],
    const device T* act [[buffer(2)]], device T* out [[buffer(3)]],
    const device int* gtab [[buffer(4)]], constant int& I [[buffer(5)]],
    constant int& H [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  dsv41_moe_down_fp4_rows_impl_<8>(w2, s2, act, out, gtab, I, H, tg, sg,
                                    lane);
}

// ---- MLX's affine codes (an oMLX pack's requantized modules) ------------
// oQ packs below 4 bits store the routed experts and the projections the
// release keeps in FP8 as MLX's affine codes: a row's B-bit codes (B 2..8)
// one little-endian bit stream in uint32 words, and a bf16 scale and bias
// per gs of a row (gs a multiple of 32): w = s * q + b. A lane takes CHUNKS
// of 32 values -- B words, inside one group -- and a row's sum over one is
// s * (q . x) + b * sum(x), MLX's own GEMV arithmetic, as fmas in a fixed
// order (the codes' order, then the bias term, then the scale's).

// The 32 codes of a chunk (B words at p), as floats, in order.
template <int B>
inline void dsv41_q32_(const device uint* p, thread float* q)
{
  uint w[B];
  for (int i = 0; i < B; ++i) { w[i] = p[i]; }
  for (int e = 0; e < 32; ++e) {
    const int bit = e * B, wd = bit / 32, sh = bit % 32;
    uint v = w[wd] >> sh;
    if (sh + B > 32) { v |= w[wd + 1] << (32 - sh); }
    q[e] = (float)(v & ((1u << B) - 1u));
  }
}

// One chunk of row sums: acc += s * (q . x) + b * sum(x) over x[0..32).
inline float dsv41_q_chunk_(thread const float* q, const device T* x,
                            const float s, const float b, const float acc)
{
  float d = 0.0f, xs = 0.0f;
  for (int e = 0; e < 32; ++e) {
    const float xv = (float)x[e];
    d = fma(q[e], xv, d);
    xs += xv;
  }
  return fma(s, d, fma(b, xs, acc));
}

// The routed experts' two halves (dsv41_moe_swiglu_fp4 / _down_fp4's
// arithmetic -- the clamps, the routing weight, the roundings) over affine
// codes: w1 and w3 at one width, w2 at its own. The cache's pools hold a
// slot's codes at slot * cw bytes and its scales and biases at slot * cs
// elements -- a slot as wide as the widest layer's expert, so the stride is
// the pool's, not this layer's. 8 lanes a row (feature or output), 4 rows
// a simdgroup, each lane every 8th chunk. ONE implementation for one pair
// and for a few rows' groups (gtab: {slot, count, pair...}, count <= MM),
// so a verify's rows are its decode's to the bit.
// grid {32, ceil(rows / 8) * 2, pairs or groups}, tg {32, 2, 1}.
template <int B, int MM>
METAL_FUNC void dsv41_moe_swiglu_q_impl_(
    const device uchar* w1, const device bfloat* s1,
    const device bfloat* b1, const device uchar* w3,
    const device bfloat* s3, const device bfloat* b3, const device T* x,
    device T* act, const long e, const device int* gr, const int p1,
    const int cnt, const device float* wt, const int K, const int I,
    const int topk, const float lim, const int gs, const long cw,
    const long cs, const uint3 tg, const uint sg, const uint lane)
{
  const int r = (int)lane / 8, j = (int)lane % 8;
  const int f = ((int)tg.y * 2 + (int)sg) * 4 + r;
  const bool live = f < I;
  const int G = K / gs;
  const long rw = (long)(K / 32) * B;            // a row's words
  float g[MM], u[MM];
  for (int m = 0; m < MM; ++m) { g[m] = u[m] = 0.0f; }
  if (live) {
    const device uint* c1 =
        reinterpret_cast<const device uint*>(w1 + e * cw) + (long)f * rw;
    const device uint* c3 =
        reinterpret_cast<const device uint*>(w3 + e * cw) + (long)f * rw;
    const long so = e * cs + (long)f * G;
    for (int k = j * 32; k < K; k += 256) {
      float q1[32], q3[32];
      dsv41_q32_<B>(c1 + (k / 32) * B, q1);
      dsv41_q32_<B>(c3 + (k / 32) * B, q3);
      const int gi = k / gs;
      const float a1 = (float)s1[so + gi], z1 = (float)b1[so + gi];
      const float a3 = (float)s3[so + gi], z3 = (float)b3[so + gi];
      for (int m = 0; m < MM; ++m) {
        if (m >= cnt) { break; }
        const int p = gr != nullptr ? gr[2 + m] : p1;
        const device T* xr = x + (long)(p / topk) * K + k;
        g[m] = dsv41_q_chunk_(q1, xr, a1, z1, g[m]);
        u[m] = dsv41_q_chunk_(q3, xr, a3, z3, u[m]);
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    g[m] += simd_shuffle_down(g[m], 4);
    g[m] += simd_shuffle_down(g[m], 2);
    g[m] += simd_shuffle_down(g[m], 1);
    u[m] += simd_shuffle_down(u[m], 4);
    u[m] += simd_shuffle_down(u[m], 2);
    u[m] += simd_shuffle_down(u[m], 1);
  }
  if (j != 0 || !live) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= cnt) { break; }
    const int p = gr != nullptr ? gr[2 + m] : p1;
    float gg = g[m], uu = u[m];
    if (lim > 0.0f) {
      uu = clamp(uu, -lim, lim);
      gg = min(gg, lim);
    }
    const float a = gg / (1.0f + exp(-gg)) * uu;
    act[(long)p * I + f] = (T)(wt[p] * a);
  }
}

template <int B, int MM>
METAL_FUNC void dsv41_moe_down_q_impl_(
    const device uchar* w2, const device bfloat* s2,
    const device bfloat* b2, const device T* act, device T* out,
    const long e, const device int* gr, const int p1, const int cnt,
    const int I, const int H, const int gs, const long cw, const long cs,
    const uint3 tg, const uint sg, const uint lane)
{
  const int r = (int)lane / 8, j = (int)lane % 8;
  const int o = ((int)tg.y * 2 + (int)sg) * 4 + r;
  const bool live = o < H;
  const int G = I / gs;
  const long rw = (long)(I / 32) * B;
  float acc[MM];
  for (int m = 0; m < MM; ++m) { acc[m] = 0.0f; }
  if (live) {
    const device uint* c2 =
        reinterpret_cast<const device uint*>(w2 + e * cw) + (long)o * rw;
    const long so = e * cs + (long)o * G;
    for (int k = j * 32; k < I; k += 256) {
      float q[32];
      dsv41_q32_<B>(c2 + (k / 32) * B, q);
      const int gi = k / gs;
      const float a = (float)s2[so + gi], z = (float)b2[so + gi];
      for (int m = 0; m < MM; ++m) {
        if (m >= cnt) { break; }
        const int p = gr != nullptr ? gr[2 + m] : p1;
        acc[m] = dsv41_q_chunk_(q, act + (long)p * I + k, a, z, acc[m]);
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    acc[m] += simd_shuffle_down(acc[m], 4);
    acc[m] += simd_shuffle_down(acc[m], 2);
    acc[m] += simd_shuffle_down(acc[m], 1);
  }
  if (j != 0 || !live) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= cnt) { break; }
    const long p = gr != nullptr ? gr[2 + m] : p1;
    out[p * H + o] = (T)acc[m];
  }
}

// One pair a threadgroup column (grid.z): slot[p]. Then the few rows'
// groups (MM 2, 4, 8): gtab[z * 10 ..] = {slot, count, pairs...}.
#define DSV41_MOE_Q(B)                                                       \
  kernel void dsv41_moe_swiglu_q##B(                                         \
      const device uchar* w1 [[buffer(0)]],                                  \
      const device bfloat* s1 [[buffer(1)]],                                 \
      const device bfloat* b1 [[buffer(2)]],                                 \
      const device uchar* w3 [[buffer(3)]],                                  \
      const device bfloat* s3 [[buffer(4)]],                                 \
      const device bfloat* b3 [[buffer(5)]],                                 \
      const device T* x [[buffer(6)]], device T* act [[buffer(7)]],          \
      const device int* slot [[buffer(8)]],                                  \
      const device float* wt [[buffer(9)]], constant int& K [[buffer(10)]],  \
      constant int& I [[buffer(11)]], constant int& topk [[buffer(12)]],     \
      constant float& lim [[buffer(13)]], constant int& gs [[buffer(14)]],   \
      constant long& cw [[buffer(15)]], constant long& cs [[buffer(16)]],    \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    const int p = (int)tg.z;                                                 \
    dsv41_moe_swiglu_q_impl_<B, 1>(w1, s1, b1, w3, s3, b3, x, act, slot[p],  \
                                   nullptr, p, 1, wt, K, I, topk, lim, gs,   \
                                   cw, cs, tg, sg, lane);                    \
  }                                                                          \
  kernel void dsv41_moe_down_q##B(                                           \
      const device uchar* w2 [[buffer(0)]],                                  \
      const device bfloat* s2 [[buffer(1)]],                                 \
      const device bfloat* b2 [[buffer(2)]],                                 \
      const device T* act [[buffer(3)]], device T* out [[buffer(4)]],        \
      const device int* slot [[buffer(5)]], constant int& I [[buffer(6)]],   \
      constant int& H [[buffer(7)]], constant int& gs [[buffer(8)]],         \
      constant long& cw [[buffer(9)]], constant long& cs [[buffer(10)]],     \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    const int p = (int)tg.z;                                                 \
    dsv41_moe_down_q_impl_<B, 1>(w2, s2, b2, act, out, slot[p], nullptr, p,  \
                                 1, I, H, gs, cw, cs, tg, sg, lane);         \
  }                                                                          \
  DSV41_MOE_Q_ROWS(B, 2)                                                     \
  DSV41_MOE_Q_ROWS(B, 4)                                                     \
  DSV41_MOE_Q_ROWS(B, 8)

#define DSV41_MOE_Q_ROWS(B, MM)                                              \
  kernel void dsv41_moe_swiglu_q##B##_rows##MM(                              \
      const device uchar* w1 [[buffer(0)]],                                  \
      const device bfloat* s1 [[buffer(1)]],                                 \
      const device bfloat* b1 [[buffer(2)]],                                 \
      const device uchar* w3 [[buffer(3)]],                                  \
      const device bfloat* s3 [[buffer(4)]],                                 \
      const device bfloat* b3 [[buffer(5)]],                                 \
      const device T* x [[buffer(6)]], device T* act [[buffer(7)]],          \
      const device int* gtab [[buffer(8)]],                                  \
      const device float* wt [[buffer(9)]], constant int& K [[buffer(10)]],  \
      constant int& I [[buffer(11)]], constant int& topk [[buffer(12)]],     \
      constant float& lim [[buffer(13)]], constant int& gs [[buffer(14)]],   \
      constant long& cw [[buffer(15)]], constant long& cs [[buffer(16)]],    \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    const device int* gr = gtab + (long)tg.z * 10;                           \
    dsv41_moe_swiglu_q_impl_<B, MM>(w1, s1, b1, w3, s3, b3, x, act, gr[0],   \
                                    gr, 0, gr[1], wt, K, I, topk, lim, gs,   \
                                    cw, cs, tg, sg, lane);                   \
  }                                                                          \
  kernel void dsv41_moe_down_q##B##_rows##MM(                                \
      const device uchar* w2 [[buffer(0)]],                                  \
      const device bfloat* s2 [[buffer(1)]],                                 \
      const device bfloat* b2 [[buffer(2)]],                                 \
      const device T* act [[buffer(3)]], device T* out [[buffer(4)]],        \
      const device int* gtab [[buffer(5)]], constant int& I [[buffer(6)]],   \
      constant int& H [[buffer(7)]], constant int& gs [[buffer(8)]],         \
      constant long& cw [[buffer(9)]], constant long& cs [[buffer(10)]],     \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    const device int* gr = gtab + (long)tg.z * 10;                           \
    dsv41_moe_down_q_impl_<B, MM>(w2, s2, b2, act, out, gr[0], gr, 0, gr[1], \
                                  I, H, gs, cw, cs, tg, sg, lane);           \
  }

DSV41_MOE_Q(2)
DSV41_MOE_Q(3)
DSV41_MOE_Q(4)
DSV41_MOE_Q(5)
DSV41_MOE_Q(6)
DSV41_MOE_Q(8)

// A projection in affine codes, y[m] = x[m] @ W^T for every m < M (M <= MM
// <= 8): dsv41_gemv_fp8_r8_impl_'s mapping -- 8 lanes a row, 4 rows a
// simdgroup, KS simdgroups splitting K and summed in order -- with each
// lane every 8th chunk of 32. The codes read once for all the rows, each
// row's sum the same chain whatever MM is. K % 256 == 0.
// grid {32, ceil(N/8) * 2 * KS, 1}, tg {32, 2 * KS, 1}.
template <int B, int MM, int KS>
METAL_FUNC void dsv41_gemv_q_impl_(
    const device uint* w, const device bfloat* s, const device bfloat* bs,
    const device T* x, device T* y, const int K, const int N, const int M,
    const int gs, const uint3 tg, const uint sgi, const uint lane,
    threadgroup float* red)
{
  const int rg = (int)sgi % 2, ks = (int)sgi / 2;
  const int r = (int)lane / 8, j = (int)lane % 8;
  const int row = ((int)tg.y * 2 + rg) * 4 + r;
  const bool live = row < N;
  const int G = K / gs;
  float acc[MM];
  for (int m = 0; m < MM; ++m) { acc[m] = 0.0f; }
  if (live) {
    const device uint* wr = w + (long)row * (K / 32) * B;
    const long so = (long)row * G;
    for (int k = (ks * 8 + j) * 32; k < K; k += 256 * KS) {
      float q[32];
      dsv41_q32_<B>(wr + (k / 32) * B, q);
      const float a = (float)s[so + k / gs], z = (float)bs[so + k / gs];
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        acc[m] = dsv41_q_chunk_(q, x + (long)m * K + k, a, z, acc[m]);
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    acc[m] += simd_shuffle_down(acc[m], 4);
    acc[m] += simd_shuffle_down(acc[m], 2);
    acc[m] += simd_shuffle_down(acc[m], 1);
  }
  if (KS == 1) {
    if (j == 0 && live) {
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        y[(long)m * N + row] = (T)acc[m];
      }
    }
    return;
  }
  if (j == 0) {
    for (int m = 0; m < MM; ++m) {
      red[((ks * 2 + rg) * MM + m) * 4 + r] = acc[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks != 0 || j != 0 || !live) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= M) { break; }
    float t = 0.0f;
    for (int q = 0; q < KS; ++q) {
      t += red[((q * 2 + rg) * MM + m) * 4 + r];
    }
    y[(long)m * N + row] = (T)t;
  }
}

// The codes widened into the compute dtype (s * q + b, rounded once), for
// a GEMM of many rows. grid {K / 32, N}.
template <int B>
METAL_FUNC void dsv41_dequant_q_impl_(
    const device uint* w, const device bfloat* s, const device bfloat* bs,
    device T* out, const int K, const int N, const int gs, const uint2 gid)
{
  const int k = (int)gid.x * 32, r = (int)gid.y;
  if (k >= K || r >= N) { return; }
  float q[32];
  dsv41_q32_<B>(w + (long)r * (K / 32) * B + (k / 32) * B, q);
  const long g = (long)r * (K / gs) + k / gs;
  const float a = (float)s[g], z = (float)bs[g];
  device T* o = out + (long)r * K + k;
  for (int e = 0; e < 32; ++e) { o[e] = (T)fma(a, q[e], z); }
}

#define DSV41_GEMVQ(name, B, MM, KS)                                         \
  kernel void name(                                                          \
      const device uint* w [[buffer(0)]],                                    \
      const device bfloat* s [[buffer(1)]],                                  \
      const device bfloat* bs [[buffer(2)]],                                 \
      const device T* x [[buffer(3)]], device T* y [[buffer(4)]],            \
      constant int& K [[buffer(5)]], constant int& N [[buffer(6)]],          \
      constant int& M [[buffer(7)]], constant int& gs [[buffer(8)]],         \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float red[2 * KS * MM * 4];                                  \
    dsv41_gemv_q_impl_<B, MM, KS>(w, s, bs, x, y, K, N, M, gs, tg, sg,       \
                                  lane, red);                                \
  }
#define DSV41_GEMVQ_ALL(B)                                                   \
  DSV41_GEMVQ(dsv41_gemv_q##B, B, 1, 1)                                      \
  DSV41_GEMVQ(dsv41_gemv_q##B##_rows, B, 8, 1)                               \
  DSV41_GEMVQ(dsv41_gemv_q##B##_k4, B, 1, 4)                                 \
  DSV41_GEMVQ(dsv41_gemv_q##B##_rows_k4, B, 8, 4)                            \
  kernel void dsv41_dequant_q##B(                                            \
      const device uint* w [[buffer(0)]],                                    \
      const device bfloat* s [[buffer(1)]],                                  \
      const device bfloat* bs [[buffer(2)]], device T* out [[buffer(3)]],    \
      constant int& K [[buffer(4)]], constant int& N [[buffer(5)]],          \
      constant int& gs [[buffer(6)]],                                        \
      uint2 gid [[thread_position_in_grid]])                                 \
  {                                                                          \
    dsv41_dequant_q_impl_<B>(w, s, bs, out, K, N, gs, gid);                  \
  }
DSV41_GEMVQ_ALL(2)
DSV41_GEMVQ_ALL(3)
DSV41_GEMVQ_ALL(4)
DSV41_GEMVQ_ALL(5)
DSV41_GEMVQ_ALL(6)
DSV41_GEMVQ_ALL(8)

// MoE.forward's sum: y = sum over a token's k pairs + the shared expert,
// f32, rounded once. grid {H, rows}.
kernel void dsv41_moe_combine(
    const device T* part [[buffer(0)]],         // [rows * k, H]
    const device T* sh   [[buffer(1)]],         // [rows, H]
    device T*       y    [[buffer(2)]],         // [rows, H]
    constant int&   H    [[buffer(3)]],
    constant int&   topk [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, t = (int)gid.y;
  if (d >= H) { return; }
  float s = 0.0f;
  for (int j = 0; j < topk; ++j) {
    s += (float)part[((long)t * topk + j) * H + d];
  }
  s += (float)sh[(long)t * H + d];
  y[(long)t * H + d] = (T)s;
}

// The shared expert's activation from one [gate | up] GEMM: [n, 2I] ->
// [n, I], the training clamps, rounded. grid {I, n}.
kernel void dsv41_swiglu_clamp(
    const device T* gu  [[buffer(0)]],
    device T*       act [[buffer(1)]],
    constant int&   I   [[buffer(2)]],
    constant float& lim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int f = (int)gid.x, i = (int)gid.y;
  if (f >= I) { return; }
  float g = (float)gu[(long)i * 2 * I + f];
  float u = (float)gu[(long)i * 2 * I + I + f];
  if (lim > 0.0f) {
    u = clamp(u, -lim, lim);
    g = min(g, lim);
  }
  act[(long)i * I + f] = (T)(g / (1.0f + exp(-g)) * u);
}

// ---- engram ---------------------------------------------------------------

// Engram.forward after wkv: kv [n, (HC + 1) H] = HC keys then the shared
// value. Per (row, copy c): the gate is sigmoid of the signed square root
// of the copy's normalized dot with its key (h * q_weight * k_weight .
// key, each side normalized by its own rms, / sqrt(H)); the copy gains
// gate * value. `mask` 0 leaves the row as it is (an image position).
// One row a threadgroup of 256. grid {256 * n}, tg {256}.
kernel void dsv41_engram_gate(
    device T*           X    [[buffer(0)]],     // [n, 4, H]
    const device T*     kv   [[buffer(1)]],     // [n, 5H]
    const device float* qk   [[buffer(2)]],     // [4, H]
    const device uchar* mask [[buffer(3)]],     // [n] (has_mask)
    constant int&       H    [[buffer(4)]],
    constant float&     eps  [[buffer(5)]],
    constant int&       has_mask [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
  if (has_mask != 0 && mask[row] == 0) { return; }
  device T* x = X + (long)row * 4 * H;
  const device T* kr = kv + (long)row * 5 * H;
  float hh[4] = {0, 0, 0, 0}, kk[4] = {0, 0, 0, 0}, dt[4] = {0, 0, 0, 0};
  for (int d = (int)tid; d < H; d += 256) {
    for (int c = 0; c < 4; ++c) {
      const float hv = (float)x[c * H + d];
      const float kv2 = (float)kr[c * H + d];
      hh[c] += hv * hv;
      kk[c] += kv2 * kv2;
      dt[c] += hv * qk[c * H + d] * kv2;
    }
  }
  threadgroup float red[8][12];
  threadgroup float gate[4];
  for (int c = 0; c < 4; ++c) {
    hh[c] = simd_sum(hh[c]);
    kk[c] = simd_sum(kk[c]);
    dt[c] = simd_sum(dt[c]);
  }
  if (lane == 0) {
    for (int c = 0; c < 4; ++c) {
      red[sg][c] = hh[c];
      red[sg][4 + c] = kk[c];
      red[sg][8 + c] = dt[c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4) {
    const int c = (int)tid;
    float a = 0, b = 0, d0 = 0;
    for (int g = 0; g < 8; ++g) {
      a += red[g][c];
      b += red[g][4 + c];
      d0 += red[g][8 + c];
    }
    const float rstd = (1.0f / sqrt(a / (float)H + eps)) *
                       (1.0f / sqrt(b / (float)H + eps));
    const float dot = d0 * rstd / sqrt((float)H);
    const float m = sqrt(max(abs(dot), 1e-6f));
    const float sg2 = dot < 0.0f ? -m : m;
    gate[c] = 1.0f / (1.0f + exp(-sg2));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = (int)tid; d < H; d += 256) {
    const float v = (float)kr[4 * H + d];
    for (int c = 0; c < 4; ++c) {
      x[c * H + d] = (T)((float)x[c * H + d] + gate[c] * v);
    }
  }
}

// ---- the vision tower (vision.py) ------------------------------------------

// ViT apply_rotary on q and k of the fused qkv rows [n, 3 * C] (q at 0, k
// at C, heads of `hd`): each head's halves x1 = [0, hd/2), x2 = [hd/2, hd)
// rotate pairwise, (x1 cos - x2 sin, x2 cos + x1 sin), the angle of pair j
// from tab [n, hd/2] (cos, sin) -- the row's grid row for the first half of
// the pairs, its column for the second. In place. grid {hd/2, 2 * heads,
// n}.
kernel void dsv41_vit_rope(
    device T*            qkv   [[buffer(0)]],
    const device float2* tab   [[buffer(1)]],
    constant int&        C     [[buffer(2)]],
    constant int&        hd    [[buffer(3)]],
    constant int&        heads [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int j = (int)gid.x, hk = (int)gid.y, i = (int)gid.z;
  const int hf = hd / 2;
  if (j >= hf) { return; }
  const int which = hk / heads, h = hk % heads;      // 0 q, 1 k
  device T* x = qkv + (long)i * 3 * C + (long)which * C + (long)h * hd;
  const float2 cs = tab[(long)i * hf + j];
  const float x1 = (float)x[j], x2 = (float)x[j + hf];
  x[j] = (T)(x1 * cs.x - x2 * cs.y);
  x[j + hf] = (T)(x2 * cs.x + x1 * cs.y);
}

// x += y, element-wise. grid {n}.
kernel void dsv41_add(
    device T*       x [[buffer(0)]],
    const device T* y [[buffer(1)]],
    constant int&   n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
  if ((int)i < n) { x[i] = (T)((float)x[i] + (float)y[i]); }
}

// Aligner's F.unfold(kernel r, stride r) of the patch grid [gh, gw, C]
// padded with zeros to multiples of r: out [L, C * r * r], L the cells
// row-major, column c * r * r + ky * r + kx. grid {C * r * r, L}.
kernel void dsv41_unfold(
    const device T* x   [[buffer(0)]],
    device T*       out [[buffer(1)]],
    constant int&   gh  [[buffer(2)]],
    constant int&   gw  [[buffer(3)]],
    constant int&   C   [[buffer(4)]],
    constant int&   r   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int k = (int)gid.x, l = (int)gid.y;
  const int rr = r * r;
  if (k >= C * rr) { return; }
  const int lw = (gw + r - 1) / r;
  const int by = l / lw, bx = l % lw;
  const int c = k / rr, ky = (k % rr) / r, kx = k % r;
  const int y = by * r + ky, xx = bx * r + kx;
  out[(long)l * C * rr + k] =
      (y < gh && xx < gw) ? x[((long)y * gw + xx) * C + c] : (T)0.0f;
}

// erf: Metal has none; Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
inline float dsv41_erf_(float x)
{
  const float s = x < 0.0f ? -1.0f : 1.0f;
  const float a = abs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) +
                            1.421413741f) * t - 0.284496736f) * t +
                          0.254829592f) * t * exp(-a * a);
  return s * y;
}

// The exact (erf) GELU, in place. grid {n}.
kernel void dsv41_gelu(
    device T*     x [[buffer(0)]],
    constant int& n [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
  if ((int)i >= n) { return; }
  const float v = (float)x[i];
  x[i] = (T)(0.5f * v * (1.0f + dsv41_erf_(v * 0.70710678118654752f)));
}

// Rows of an image span written into the residual copies at their
// positions: X[pos[k]][c] = rows[k] for every copy. grid {H, k}.
kernel void dsv41_rows_at_hc(
    const device T*   rows [[buffer(0)]],     // [k, H]
    const device int* pos  [[buffer(1)]],     // [k] rows of this chunk
    device T*         X    [[buffer(2)]],     // [n, 4, H]
    constant int&     H    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, k = (int)gid.y;
  if (d >= H) { return; }
  const T v = rows[(long)k * H + d];
  device T* x = X + (long)pos[k] * 4 * H + d;
  for (int c = 0; c < 4; ++c) { x[c * H] = v; }
}

// ---- f32-output GEMV / GEMM (x @ W^T, element-type operands) -----------

// y[m] = x[m] @ W^T in f32 for every m < M (M <= MM <= 8; the head's
// logits, the routers' scores; the element-type twin below for a dense
// projection's few rows, the MLX pack's bf16 wo_a): 4 output rows a
// simdgroup, 8 of K a lane a step, KS simdgroups splitting K over the same
// rows (a narrow matrix: few rows give too few threadgroups to fill the
// GPU) and their sums added in order. The weights are read ONCE for all
// the rows, and a row's sum is the same chain of fmas whatever MM is --
// so a one-row step (MM 1) and a verify (MM 8) give each row the same value
// to the bit: a speculative verify's rows are its decode's (KS is the
// matrix's, never the rows'). grid.z batches: x, W and y advance by their
// strides (elements) a batch -- wo_a's groups in one dispatch.
// grid {32, ceil(N/8) * 2 * KS, batches}, tg {32, 2 * KS, 1}.
template <int MM, int KS, typename O>
METAL_FUNC void dsv41_gemv_impl_(
    const device T* x, const device T* W, device O* y, const int K,
    const int N, const int M, const uint3 tg, const uint sgi,
    const uint lane, threadgroup float* red)
{
  const int rg = (int)sgi % 2, ks = (int)sgi / 2;
  const int r0 = ((int)tg.y * 2 + rg) * 4;
  const int nr = r0 < N ? min(4, N - r0) : 0;
  float acc[MM][4];
  for (int m = 0; m < MM; ++m) {
    for (int r = 0; r < 4; ++r) { acc[m][r] = 0.0f; }
  }
  for (int k = (ks * 32 + (int)lane) * 8; k < K; k += 256 * KS) {
    float w[4][8];
    for (int r = 0; r < 4; ++r) {
      if (r >= nr) { break; }
      const device T* wr = W + (long)(r0 + r) * K + k;
      for (int j = 0; j < 8; ++j) { w[r][j] = (float)wr[j]; }
    }
    for (int m = 0; m < MM; ++m) {
      if (m >= M) { break; }
      const device T* xr = x + (long)m * K + k;
      float xv[8];
      for (int j = 0; j < 8; ++j) { xv[j] = (float)xr[j]; }
      for (int r = 0; r < 4; ++r) {
        if (r >= nr) { break; }
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) { s = fma(xv[j], w[r][j], s); }
        acc[m][r] += s;
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    for (int r = 0; r < 4; ++r) { acc[m][r] = simd_sum(acc[m][r]); }
  }
  if (KS == 1) {
    if (lane == 0) {
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        for (int r = 0; r < nr; ++r) {
          y[(long)m * N + r0 + r] = (O)acc[m][r];
        }
      }
    }
    return;
  }
  // the K slices' sums, added in slice order
  if (lane == 0) {
    for (int m = 0; m < MM; ++m) {
      for (int r = 0; r < 4; ++r) {
        red[((ks * 2 + rg) * MM + m) * 4 + r] = acc[m][r];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks != 0 || lane != 0) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= M) { break; }
    for (int r = 0; r < nr; ++r) {
      float t = 0.0f;
      for (int q = 0; q < KS; ++q) {
        t += red[((q * 2 + rg) * MM + m) * 4 + r];
      }
      y[(long)m * N + r0 + r] = (O)t;
    }
  }
}

#define DSV41_GEMV(name, MM, KS, O)                                          \
  kernel void name(                                                          \
      const device T* x [[buffer(0)]], const device T* W [[buffer(1)]],      \
      device O* y [[buffer(2)]], constant int& K [[buffer(3)]],              \
      constant int& N [[buffer(4)]], constant int& M [[buffer(5)]],          \
      constant long& xb [[buffer(6)]], constant long& wb [[buffer(7)]],      \
      constant long& yb [[buffer(8)]],                                       \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float red[2 * KS * MM * 4];                                  \
    dsv41_gemv_impl_<MM, KS, O>(x + tg.z * xb, W + tg.z * wb, y + tg.z * yb, \
                                K, N, M, tg, sg, lane, red);                 \
  }
DSV41_GEMV(dsv41_gemv_f32, 1, 1, float)
DSV41_GEMV(dsv41_gemv_f32_rows, 8, 1, float)
DSV41_GEMV(dsv41_gemv_f32_k4, 1, 4, float)
DSV41_GEMV(dsv41_gemv_f32_rows_k4, 8, 4, float)
DSV41_GEMV(dsv41_gemv_one, 1, 1, T)
DSV41_GEMV(dsv41_gemv_rows, 8, 1, T)
DSV41_GEMV(dsv41_gemv_one_k4, 1, 4, T)
DSV41_GEMV(dsv41_gemv_rows_k4, 8, 4, T)

// dsv41_gemv_impl_ with 8 lanes a row (4 rows a simdgroup): each lane takes
// every 8th 8-element chunk of its row, K / 64 of them; a row's sum is each
// lane's chunks in order, then the 8 lanes' sums in a fixed tree, whatever
// MM is. K % 64 == 0. grid {32, ceil(N/8) * 2 * KS, batches},
// tg {32, 2 * KS, 1}.
template <int MM, int KS, typename O>
METAL_FUNC void dsv41_gemv_r8_impl_(
    const device T* x, const device T* W, device O* y, const int K,
    const int N, const int M, const uint3 tg, const uint sgi,
    const uint lane, threadgroup float* red)
{
  const int rg = (int)sgi % 2, ks = (int)sgi / 2;
  const int r = (int)lane / 8, j = (int)lane % 8;
  const int row = ((int)tg.y * 2 + rg) * 4 + r;
  const bool live = row < N;
  float acc[MM];
  for (int m = 0; m < MM; ++m) { acc[m] = 0.0f; }
  if (live) {
    const device T* wr = W + (long)row * K;
    for (int k = (ks * 8 + j) * 8; k < K; k += 64 * KS) {
      float w[8];
      for (int e = 0; e < 8; ++e) { w[e] = (float)wr[k + e]; }
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        const device T* xr = x + (long)m * K + k;
        float s = 0.0f;
        for (int e = 0; e < 8; ++e) { s = fma((float)xr[e], w[e], s); }
        acc[m] += s;
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    acc[m] += simd_shuffle_down(acc[m], 4);
    acc[m] += simd_shuffle_down(acc[m], 2);
    acc[m] += simd_shuffle_down(acc[m], 1);
  }
  if (KS == 1) {
    if (j == 0 && live) {
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        y[(long)m * N + row] = (O)acc[m];
      }
    }
    return;
  }
  if (j == 0) {
    for (int m = 0; m < MM; ++m) {
      red[((ks * 2 + rg) * MM + m) * 4 + r] = acc[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks != 0 || j != 0 || !live) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= M) { break; }
    float t = 0.0f;
    for (int q = 0; q < KS; ++q) {
      t += red[((q * 2 + rg) * MM + m) * 4 + r];
    }
    y[(long)m * N + row] = (O)t;
  }
}

#define DSV41_GEMV_R8(name, MM, KS, O)                                       \
  kernel void name(                                                          \
      const device T* x [[buffer(0)]], const device T* W [[buffer(1)]],      \
      device O* y [[buffer(2)]], constant int& K [[buffer(3)]],              \
      constant int& N [[buffer(4)]], constant int& M [[buffer(5)]],          \
      constant long& xb [[buffer(6)]], constant long& wb [[buffer(7)]],      \
      constant long& yb [[buffer(8)]],                                       \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float red[2 * KS * MM * 4];                                  \
    dsv41_gemv_r8_impl_<MM, KS, O>(x + tg.z * xb, W + tg.z * wb,             \
                                   y + tg.z * yb, K, N, M, tg, sg, lane,     \
                                   red);                                     \
  }
DSV41_GEMV_R8(dsv41_gemv_f32_r8, 1, 1, float)
DSV41_GEMV_R8(dsv41_gemv_f32_r8_rows, 8, 1, float)
DSV41_GEMV_R8(dsv41_gemv_f32_r8_k4, 1, 4, float)
DSV41_GEMV_R8(dsv41_gemv_f32_r8_rows_k4, 8, 4, float)
DSV41_GEMV_R8(dsv41_gemv_one_r8, 1, 1, T)
DSV41_GEMV_R8(dsv41_gemv_rows_r8, 8, 1, T)
DSV41_GEMV_R8(dsv41_gemv_one_r8_k4, 1, 4, T)
DSV41_GEMV_R8(dsv41_gemv_rows_r8_k4, 8, 4, T)

// ---- FP8 projections kept as their codes -------------------------------

// Four E4M3 bytes as floats, each 1/256 of its value: an E4M3 byte (s eeee
// mmm, bias 7) is exactly the half with bits s << 15 | eeeemmm << 7 times
// 2^8 (dense_gemm.metal's fp8x4_); the 256 goes into the scale.
inline float4 dsv41_fp8x4_(uint u)
{
  const uint a = ((u & 0x80u) << 8) | ((u & 0x7Fu) << 7) |
                 ((u & 0x8000u) << 16) | ((u & 0x7F00u) << 15);
  const uint b = ((u & 0x800000u) >> 8) | ((u & 0x7F0000u) >> 9) |
                 (u & 0x80000000u) | ((u & 0x7F000000u) >> 1);
  const half2 ha = as_type<half2>(a);
  const half2 hb = as_type<half2>(b);
  return float4((float)ha.x, (float)ha.y, (float)hb.x, (float)hb.y);
}

// y[m] = x[m] @ W^T over E4M3 codes [N, K] with an E8M0 scale per 32 of a
// row (the release's 32x32 block scales are spread to its rows at load),
// for every m < M (M <= MM <= 8). K % 256 == 0. 4 output rows a
// simdgroup, 16 codes a lane a step, KS simdgroups splitting K (a narrow
// matrix) and summed in order; the codes read once for all the rows, each
// row's sum the same chain of fmas whatever MM is (a verify's rows are its
// decode's). grid.z batches with element strides, as dsv41_gemv_impl_.
// grid {32, ceil(N/8) * 2 * KS, batches}, tg {32, 2 * KS, 1}.
template <int MM, int KS>
METAL_FUNC void dsv41_gemv_fp8_impl_(
    const device uchar* w, const device uchar* s, const device T* x,
    device T* y, const int K, const int N, const int M, const uint3 tg,
    const uint sgi, const uint lane, threadgroup float* red)
{
  const int rg = (int)sgi % 2, ks = (int)sgi / 2;
  const int r0 = ((int)tg.y * 2 + rg) * 4;
  const int nr = r0 < N ? min(4, N - r0) : 0;
  const int KSC = K / 32;
  float acc[MM][4];
  for (int m = 0; m < MM; ++m) {
    for (int r = 0; r < 4; ++r) { acc[m][r] = 0.0f; }
  }
  // 16 codes a lane a step (one 16-byte load, half a scale group)
  for (int k = (ks * 32 + (int)lane) * 16; k < K; k += 512 * KS) {
    float4 wq[4][4];
    float sc[4];
    for (int r = 0; r < 4; ++r) {
      if (r >= nr) { break; }
      const long row = r0 + r;
      const uint4 q = *reinterpret_cast<const device uint4*>(w + row * K + k);
      wq[r][0] = dsv41_fp8x4_(q.x);
      wq[r][1] = dsv41_fp8x4_(q.y);
      wq[r][2] = dsv41_fp8x4_(q.z);
      wq[r][3] = dsv41_fp8x4_(q.w);
      sc[r] = as_type<float>((uint)s[row * KSC + k / 32] << 23);
    }
    for (int m = 0; m < MM; ++m) {
      if (m >= M) { break; }
      const device T* xr = x + (long)m * K + k;
      float4 xv[4];
      for (int j = 0; j < 4; ++j) {
        xv[j] = float4((float)xr[4 * j], (float)xr[4 * j + 1],
                       (float)xr[4 * j + 2], (float)xr[4 * j + 3]);
      }
      for (int r = 0; r < 4; ++r) {
        if (r >= nr) { break; }
        float d = 0.0f;
        for (int j = 0; j < 4; ++j) {
          d = fma(wq[r][j].x, xv[j].x, d);
          d = fma(wq[r][j].y, xv[j].y, d);
          d = fma(wq[r][j].z, xv[j].z, d);
          d = fma(wq[r][j].w, xv[j].w, d);
        }
        acc[m][r] = fma(sc[r], d, acc[m][r]);
      }
    }
  }
  for (int m = 0; m < MM; ++m) {
    for (int r = 0; r < 4; ++r) { acc[m][r] = simd_sum(acc[m][r]); }
  }
  if (KS == 1) {
    if (lane == 0) {
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        for (int r = 0; r < nr; ++r) {
          y[(long)m * N + r0 + r] = (T)(acc[m][r] * 256.0f);
        }
      }
    }
    return;
  }
  if (lane == 0) {
    for (int m = 0; m < MM; ++m) {
      for (int r = 0; r < 4; ++r) {
        red[((ks * 2 + rg) * MM + m) * 4 + r] = acc[m][r];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks != 0 || lane != 0) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= M) { break; }
    for (int r = 0; r < nr; ++r) {
      float t = 0.0f;
      for (int q = 0; q < KS; ++q) {
        t += red[((q * 2 + rg) * MM + m) * 4 + r];
      }
      y[(long)m * N + r0 + r] = (T)(t * 256.0f);
    }
  }
}

// dsv41_gemv_fp8_impl_ with 8 lanes a row (4 rows a simdgroup): each lane
// takes every 8th 16-code chunk of its row -- K / 128 of them, the same
// for every lane -- where the one-row mapping gave a short row (wq_b's K
// of 1280) 2.5 steps a lane with half the lanes idle in the last. A row's
// sum: each lane's chunks in order, then the 8 lanes' sums in a fixed
// tree, whatever MM is. K % 128 == 0.
// grid {32, ceil(N/8) * 2 * KS, batches}, tg {32, 2 * KS, 1}.
template <int MM, int KS>
METAL_FUNC void dsv41_gemv_fp8_r8_impl_(
    const device uchar* w, const device uchar* s, const device T* x,
    device T* y, const int K, const int N, const int M, const uint3 tg,
    const uint sgi, const uint lane, threadgroup float* red)
{
  const int rg = (int)sgi % 2, ks = (int)sgi / 2;
  const int r = (int)lane / 8, j = (int)lane % 8;
  const int row = ((int)tg.y * 2 + rg) * 4 + r;
  const bool live = row < N;
  const int KSC = K / 32;
  float acc[MM];
  for (int m = 0; m < MM; ++m) { acc[m] = 0.0f; }
  if (live) {
    const device uchar* wr = w + (long)row * K;
    const device uchar* sr = s + (long)row * KSC;
    for (int k = (ks * 8 + j) * 16; k < K; k += 128 * KS) {
      const uint4 q = *reinterpret_cast<const device uint4*>(wr + k);
      const float4 wq[4] = {dsv41_fp8x4_(q.x), dsv41_fp8x4_(q.y),
                            dsv41_fp8x4_(q.z), dsv41_fp8x4_(q.w)};
      const float sc = as_type<float>((uint)sr[k / 32] << 23);
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        const device T* xr = x + (long)m * K + k;
        float d = 0.0f;
        for (int jj = 0; jj < 4; ++jj) {
          d = fma(wq[jj].x, (float)xr[4 * jj], d);
          d = fma(wq[jj].y, (float)xr[4 * jj + 1], d);
          d = fma(wq[jj].z, (float)xr[4 * jj + 2], d);
          d = fma(wq[jj].w, (float)xr[4 * jj + 3], d);
        }
        acc[m] = fma(sc, d, acc[m]);
      }
    }
  }
  // the row's 8 lanes, in a fixed tree
  for (int m = 0; m < MM; ++m) {
    acc[m] += simd_shuffle_down(acc[m], 4);
    acc[m] += simd_shuffle_down(acc[m], 2);
    acc[m] += simd_shuffle_down(acc[m], 1);
  }
  if (KS == 1) {
    if (j == 0 && live) {
      for (int m = 0; m < MM; ++m) {
        if (m >= M) { break; }
        y[(long)m * N + row] = (T)(acc[m] * 256.0f);
      }
    }
    return;
  }
  if (j == 0) {
    for (int m = 0; m < MM; ++m) {
      red[((ks * 2 + rg) * MM + m) * 4 + r] = acc[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ks != 0 || j != 0 || !live) { return; }
  for (int m = 0; m < MM; ++m) {
    if (m >= M) { break; }
    float t = 0.0f;
    for (int q = 0; q < KS; ++q) {
      t += red[((q * 2 + rg) * MM + m) * 4 + r];
    }
    y[(long)m * N + row] = (T)(t * 256.0f);
  }
}

#define DSV41_GEMV8R8(name, MM, KS)                                          \
  kernel void name(                                                          \
      const device uchar* w [[buffer(0)]],                                   \
      const device uchar* s [[buffer(1)]], const device T* x [[buffer(2)]],  \
      device T* y [[buffer(3)]], constant int& K [[buffer(4)]],              \
      constant int& N [[buffer(5)]], constant int& M [[buffer(6)]],          \
      constant long& xb [[buffer(7)]], constant long& wb [[buffer(8)]],      \
      constant long& yb [[buffer(9)]],                                       \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float red[2 * KS * MM * 4];                                  \
    dsv41_gemv_fp8_r8_impl_<MM, KS>(w + tg.z * wb, s + tg.z * (wb / 32),     \
                                    x + tg.z * xb, y + tg.z * yb, K, N, M,   \
                                    tg, sg, lane, red);                      \
  }
DSV41_GEMV8R8(dsv41_gemv_fp8_r8, 1, 1)
DSV41_GEMV8R8(dsv41_gemv_fp8_r8_rows, 8, 1)
DSV41_GEMV8R8(dsv41_gemv_fp8_r8_k4, 1, 4)
DSV41_GEMV8R8(dsv41_gemv_fp8_r8_rows_k4, 8, 4)

#define DSV41_GEMV8(name, MM, KS)                                            \
  kernel void name(                                                          \
      const device uchar* w [[buffer(0)]],                                   \
      const device uchar* s [[buffer(1)]], const device T* x [[buffer(2)]],  \
      device T* y [[buffer(3)]], constant int& K [[buffer(4)]],              \
      constant int& N [[buffer(5)]], constant int& M [[buffer(6)]],          \
      constant long& xb [[buffer(7)]], constant long& wb [[buffer(8)]],      \
      constant long& yb [[buffer(9)]],                                       \
      uint3 tg [[threadgroup_position_in_grid]],                             \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float red[2 * KS * MM * 4];                                  \
    dsv41_gemv_fp8_impl_<MM, KS>(w + tg.z * wb, s + tg.z * (wb / 32),        \
                                 x + tg.z * xb, y + tg.z * yb, K, N, M, tg,  \
                                 sg, lane, red);                             \
  }
DSV41_GEMV8(dsv41_gemv_fp8, 1, 1)
DSV41_GEMV8(dsv41_gemv_fp8_rows, 8, 1)
DSV41_GEMV8(dsv41_gemv_fp8_k4, 1, 4)
DSV41_GEMV8(dsv41_gemv_fp8_rows_k4, 8, 4)

// The codes widened into the compute dtype (exact in bf16), for a GEMM of
// many rows. grid {K / 8, N}.
kernel void dsv41_dequant_fp8(
    const device uchar* w   [[buffer(0)]],
    const device uchar* s   [[buffer(1)]],
    device T*           out [[buffer(2)]],
    constant int&       K   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int k = (int)gid.x * 8, r = (int)gid.y;
  if (k >= K || r >= N) { return; }
  const uint2 q = *reinterpret_cast<const device uint2*>(w + (long)r * K + k);
  const float sc =
      as_type<float>((uint)s[(long)r * (K / 32) + k / 32] << 23) * 256.0f;
  const float4 a = dsv41_fp8x4_(q.x) * sc, b = dsv41_fp8x4_(q.y) * sc;
  device T* o = out + (long)r * K + k;
  o[0] = (T)a.x; o[1] = (T)a.y; o[2] = (T)a.z; o[3] = (T)a.w;
  o[4] = (T)b.x; o[5] = (T)b.y; o[6] = (T)b.z; o[7] = (T)b.w;
}

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
#define MLX_MTL_CONST static constant constexpr const
#define SIMD_SIZE 32
typedef bfloat bfloat16_t;
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"

// The steel tile (32x32x32, 2x2 simdgroups, f32 accumulation) storing f32.
// K % 32 == 0. grid threadgroups {ceil(N/32), ceil(M/32)}, tg {32, 2, 2}.
kernel void dsv41_gemm_f32(
    const device T* x [[buffer(0)]],            // [M, K]
    const device T* W [[buffer(1)]],            // [N, K]
    device float*   y [[buffer(2)]],            // [M, N]
    constant int&   K [[buffer(3)]],
    constant int&   N [[buffer(4)]],
    constant int&   M [[buffer(5)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int BM = 32, BK = 32, BN = 32, WM = 2, WN = 2;
  constexpr int BKP = BK + 16 / sizeof(T);
  threadgroup T Xs[BM * BKP];
  threadgroup T Ws[BN * BKP];
  using mma_t = mlx::steel::BlockMMA<T, float, BM, BN, BK, WM, WN, false,
                                     true, BKP, BKP, float>;
  using loader_t =
      mlx::steel::BlockLoader<T, BM, BK, BKP, 1, WM * WN * SIMD_SIZE>;
  const int y_row = tid.y * BM, y_col = tid.x * BN;
  x += y_row * (long)K;
  W += y_col * (long)K;
  y += y_row * (long)N + y_col;
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_t lx(x, K, Xs, simd_gid, simd_lid);
  loader_t lw(W, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (num_els < BM) {
      lx.load_safe(short2(BK, num_els));
    } else {
      lx.load_unsafe();
    }
    if (num_outs < BN) {
      lw.load_safe(short2(BK, num_outs));
    } else {
      lw.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    lx.next();
    lw.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

// ---- the routed experts as grouped GEMMs (a prefill) --------------------
// The pairs sorted by expert, each expert's run padded to a 32-row tile, so
// a tile reads its expert's weights once where the pair kernels read them
// once per routed row. A tile's rows are independent, so the order the
// sort puts pairs in within a run changes nothing.

// One threadgroup of 1024: srow / sdst / swt [cap] (a pad row -1 / -1 / 0)
// and tile2e [maxtiles] (the slot a tile reads, -1 past the last).
constant constexpr int kDsv41SortE = 512;
kernel void dsv41_moe_sort(
    const device int*   eid    [[buffer(0)]],    // [np] expert ids
    const device int*   slot   [[buffer(1)]],    // [np] their slots
    const device float* wt     [[buffer(2)]],    // [np]
    device int*         srow   [[buffer(3)]],
    device int*         sdst   [[buffer(4)]],
    device float*       swt    [[buffer(5)]],
    device int*         tile2e [[buffer(6)]],
    constant int&       np     [[buffer(7)]],
    constant int&       topk   [[buffer(8)]],
    constant int&       E      [[buffer(9)]],
    constant int&       cap    [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]])
{
  threadgroup atomic_uint cnt[kDsv41SortE], fill[kDsv41SortE];
  threadgroup uint off[kDsv41SortE];
  threadgroup int smap[kDsv41SortE];
  for (int e = (int)tid; e < E; e += 1024) {
    atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
    atomic_store_explicit(&fill[e], 0u, memory_order_relaxed);
  }
  for (int i = (int)tid; i < cap; i += 1024) {
    srow[i] = -1;
    sdst[i] = -1;
    swt[i] = 0.0f;
  }
  for (int t = (int)tid; t < cap / 32; t += 1024) { tile2e[t] = -1; }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for (int p = (int)tid; p < np; p += 1024) {
    atomic_fetch_add_explicit(&cnt[eid[p]], 1u, memory_order_relaxed);
    smap[eid[p]] = slot[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (tid == 0) {
    uint acc = 0;
    for (int e = 0; e < E; ++e) {
      off[e] = acc;
      const uint nt =
          (atomic_load_explicit(&cnt[e], memory_order_relaxed) + 31) / 32;
      for (uint t = 0; t < nt; ++t) { tile2e[acc / 32 + t] = smap[e]; }
      acc += nt * 32;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for (int p = (int)tid; p < np; p += 1024) {
    const int e = eid[p];
    const uint pos =
        off[e] + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed);
    srow[pos] = p / topk;
    sdst[pos] = p;
    swt[pos] = wt[p];
  }
}

// A 32 x 32 weight tile of an expert's E2M1 codes [N, K / 2] with an E8M0
// scale per 32 of a row, widened exactly to f32: 128 threads, 8 codes (one
// uint) each, a tile's columns inside one scale.
struct Dsv41Fp4Loader {
  const short bi, bj;
  threadgroup float* dst;
  const device uchar* src;
  const device uchar* sc;
  int k_off;
  METAL_FUNC Dsv41Fp4Loader(const device uchar* codes,
                            const device uchar* scales, const int K,
                            const int y_col, threadgroup float* dst_,
                            ushort sg, ushort lane)
      : bi((short)((sg * 32 + lane) / 4)),
        bj((short)(8 * ((sg * 32 + lane) % 4))),
        dst(dst_ + bi * 36 + bj),
        src(codes + (long)(y_col + bi) * (K / 2) + bj / 2),
        sc(scales + (long)(y_col + bi) * (K / 32)), k_off(0) {}
  METAL_FUNC void load() const {
    const float s = dsv41_e8m0_(sc[k_off / 32]);
    const uint u = *(const device uint*)(src + k_off / 2);
    for (short j = 0; j < 8; ++j) {
      dst[j] = kFp4[(u >> (4 * j)) & 0xFu] * s;
    }
  }
  METAL_FUNC void next() { k_off += 32; }
};

// A 32 x 32 activation tile in f32 from T rows: gathered (srow[base + r],
// < 0 a zero row) or contiguous (srow null).
template <typename S>
struct Dsv41RowLoader {
  const short bi, bj;
  threadgroup float* dst;
  const device S* src;
  const int ld;
  int r;
  int k_off;
  METAL_FUNC Dsv41RowLoader(const device S* src_, const int ld_,
                            const device int* srow, const int base,
                            threadgroup float* dst_, ushort sg, ushort lane)
      : bi((short)((sg * 32 + lane) / 4)),
        bj((short)(8 * ((sg * 32 + lane) % 4))),
        dst(dst_ + bi * 36 + bj), src(src_), ld(ld_), k_off(0) {
    r = srow != nullptr ? srow[base + bi] : base + bi;
  }
  METAL_FUNC void load() const {
    if (r < 0) {
      for (short j = 0; j < 8; ++j) { dst[j] = 0.0f; }
      return;
    }
    const device S* p = src + (long)r * ld + k_off + bj;
    for (short j = 0; j < 8; ++j) { dst[j] = (float)p[j]; }
  }
  METAL_FUNC void next() { k_off += 32; }
};

using dsv41_mma_f32_t = mlx::steel::BlockMMA<float, float, 32, 32, 32, 2, 2,
                                             false, true, 36, 36, float>;

// act_s[row] = swt[row] * silu(min(g, lim)) * clamp(u, +-lim) with g = w1 .
// x, u = w3 . x over the tile's expert (parts 0-3), x row srow[row] of the
// FFN input. threadgroups {I / 32, maxtiles}, tg {32, 2, 2}.
kernel void dsv41_moe_gu_fp4_grouped(
    const device uchar* w1     [[buffer(0)]],
    const device uchar* s1     [[buffer(1)]],
    const device uchar* w3     [[buffer(2)]],
    const device uchar* s3     [[buffer(3)]],
    const device T*     x      [[buffer(4)]],    // [rows, K]
    device T*           act    [[buffer(5)]],    // [cap, I]
    const device int*   srow   [[buffer(6)]],
    const device float* swt    [[buffer(7)]],
    const device int*   tile2e [[buffer(8)]],
    constant int&       K      [[buffer(9)]],
    constant int&       I      [[buffer(10)]],
    constant float&     lim    [[buffer(11)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  threadgroup float Xs[32 * 36], Gs[32 * 36], Us[32 * 36];
  const int y_row = (int)tid.y * 32, y_col = (int)tid.x * 32;
  const long wo = (long)e * I * (K / 2), so = (long)e * I * (K / 32);
  Dsv41RowLoader<T> lx(x, K, srow, y_row, Xs, sg, lane);
  Dsv41Fp4Loader lg(w1 + wo, s1 + so, K, y_col, Gs, sg, lane);
  Dsv41Fp4Loader lu(w3 + wo, s3 + so, K, y_col, Us, sg, lane);
  dsv41_mma_f32_t mg(sg, lane), mu(sg, lane);
  for (int k = 0; k < K; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lx.load();
    lg.load();
    lu.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mg.mma(Xs, Gs);
    mu.mma(Xs, Us);
    lx.next();
    lg.next();
    lu.next();
  }
  for (short ti = 0; ti < 2; ++ti) {
    const int row = y_row + mg.sm + ti * 16;
    const float w = swt[row];
    for (short tj = 0; tj < 2; ++tj) {
      const int col = y_col + mg.sn + tj * 16;
      const thread auto& fg = mg.Ctile.frag_at(ti, tj);
      const thread auto& fu = mu.Ctile.frag_at(ti, tj);
      for (short c = 0; c < 2; ++c) {
        float gg = fg[c], uu = fu[c];
        if (lim > 0.0f) {
          uu = clamp(uu, -lim, lim);
          gg = min(gg, lim);
        }
        act[(long)row * I + col + c] = (T)(w * (gg / (1.0f + exp(-gg)) * uu));
      }
    }
  }
}

// part[sdst[row]] = w2 . act_s[row] over the tile's expert (parts 4, 5),
// rounded to T; pad rows store nothing. threadgroups {H / 32, maxtiles},
// tg {32, 2, 2}.
kernel void dsv41_moe_down_fp4_grouped(
    const device uchar* w2     [[buffer(0)]],
    const device uchar* s2     [[buffer(1)]],
    const device T*     act    [[buffer(2)]],    // [cap, I]
    device T*           part   [[buffer(3)]],    // [np, H]
    const device int*   sdst   [[buffer(4)]],
    const device int*   tile2e [[buffer(5)]],
    constant int&       I      [[buffer(6)]],
    constant int&       H      [[buffer(7)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  threadgroup float Xs[32 * 36], Ws[32 * 36];
  const int y_row = (int)tid.y * 32, y_col = (int)tid.x * 32;
  Dsv41RowLoader<T> lx(act, I, nullptr, y_row, Xs, sg, lane);
  Dsv41Fp4Loader lw(w2 + (long)e * H * (I / 2), s2 + (long)e * H * (I / 32),
                    I, y_col, Ws, sg, lane);
  dsv41_mma_f32_t mm(sg, lane);
  for (int k = 0; k < I; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lx.load();
    lw.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mm.mma(Xs, Ws);
    lx.next();
    lw.next();
  }
  for (short ti = 0; ti < 2; ++ti) {
    const int row = y_row + mm.sm + ti * 16;
    const int dr = sdst[row];
    if (dr < 0) { continue; }
    for (short tj = 0; tj < 2; ++tj) {
      const int col = y_col + mm.sn + tj * 16;
      const thread auto& f = mm.Ctile.frag_at(ti, tj);
      part[(long)dr * H + col] = (T)f[0];
      part[(long)dr * H + col + 1] = (T)f[1];
    }
  }
}

// A 32 x 32 weight tile of an expert's affine codes (B bits, a bf16 scale
// and bias per gs of a row; dsv41_q32_'s layout) widened to f32 as
// s * q + b: 128 threads, 8 codes (B bytes) each, inside one group.
template <int B>
struct Dsv41AffLoader {
  const short bi, bj;
  threadgroup float* dst;
  const device uchar* src;
  const device bfloat* sc;
  const device bfloat* bs;
  const int gs;
  int k_off;
  METAL_FUNC Dsv41AffLoader(const device uchar* codes,
                            const device bfloat* scales,
                            const device bfloat* biases, const int K,
                            const int gs_, const int y_col,
                            threadgroup float* dst_, ushort sg, ushort lane)
      : bi((short)((sg * 32 + lane) / 4)),
        bj((short)(8 * ((sg * 32 + lane) % 4))),
        dst(dst_ + bi * 36 + bj),
        src(codes + (long)(y_col + bi) * (K / 8) * B),
        sc(scales + (long)(y_col + bi) * (K / gs_)),
        bs(biases + (long)(y_col + bi) * (K / gs_)), gs(gs_), k_off(0) {}
  METAL_FUNC void load() const {
    const int k = k_off + bj;
    const device uchar* p = src + (k / 8) * B;
    ulong v = 0;
    for (int i = 0; i < B; ++i) { v |= (ulong)p[i] << (8 * i); }
    const float a = (float)sc[k / gs], z = (float)bs[k / gs];
    for (short j = 0; j < 8; ++j) {
      dst[j] = fma(a, (float)((uint)(v >> (B * j)) & ((1u << B) - 1u)), z);
    }
  }
  METAL_FUNC void next() { k_off += 32; }
};

// dsv41_moe_gu_fp4_grouped / _down_fp4_grouped over affine codes (the
// slot's parts at slot * cw bytes and slot * cs elements).
template <int B>
METAL_FUNC void dsv41_moe_gu_q_grouped_impl_(
    const device uchar* w1, const device bfloat* s1,
    const device bfloat* b1, const device uchar* w3,
    const device bfloat* s3, const device bfloat* b3, const device T* x,
    device T* act, const device int* srow, const device float* swt,
    const device int* tile2e, const int K, const int I, const float lim,
    const int gs, const long cw, const long cs, const uint3 tid,
    const uint sg, const uint lane, threadgroup float* Xs,
    threadgroup float* Gs, threadgroup float* Us)
{
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  const int y_row = (int)tid.y * 32, y_col = (int)tid.x * 32;
  Dsv41RowLoader<T> lx(x, K, srow, y_row, Xs, sg, lane);
  Dsv41AffLoader<B> lg(w1 + (long)e * cw, s1 + (long)e * cs,
                       b1 + (long)e * cs, K, gs, y_col, Gs, sg, lane);
  Dsv41AffLoader<B> lu(w3 + (long)e * cw, s3 + (long)e * cs,
                       b3 + (long)e * cs, K, gs, y_col, Us, sg, lane);
  dsv41_mma_f32_t mg(sg, lane), mu(sg, lane);
  for (int k = 0; k < K; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lx.load();
    lg.load();
    lu.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mg.mma(Xs, Gs);
    mu.mma(Xs, Us);
    lx.next();
    lg.next();
    lu.next();
  }
  for (short ti = 0; ti < 2; ++ti) {
    const int row = y_row + mg.sm + ti * 16;
    const float w = swt[row];
    for (short tj = 0; tj < 2; ++tj) {
      const int col = y_col + mg.sn + tj * 16;
      const thread auto& fg = mg.Ctile.frag_at(ti, tj);
      const thread auto& fu = mu.Ctile.frag_at(ti, tj);
      for (short c = 0; c < 2; ++c) {
        float gg = fg[c], uu = fu[c];
        if (lim > 0.0f) {
          uu = clamp(uu, -lim, lim);
          gg = min(gg, lim);
        }
        act[(long)row * I + col + c] = (T)(w * (gg / (1.0f + exp(-gg)) * uu));
      }
    }
  }
}

template <int B>
METAL_FUNC void dsv41_moe_down_q_grouped_impl_(
    const device uchar* w2, const device bfloat* s2,
    const device bfloat* b2, const device T* act, device T* part,
    const device int* sdst, const device int* tile2e, const int I,
    const int H, const int gs, const long cw, const long cs,
    const uint3 tid, const uint sg, const uint lane, threadgroup float* Xs,
    threadgroup float* Ws)
{
  const int e = tile2e[tid.y];
  if (e < 0) { return; }
  const int y_row = (int)tid.y * 32, y_col = (int)tid.x * 32;
  Dsv41RowLoader<T> lx(act, I, nullptr, y_row, Xs, sg, lane);
  Dsv41AffLoader<B> lw(w2 + (long)e * cw, s2 + (long)e * cs,
                       b2 + (long)e * cs, I, gs, y_col, Ws, sg, lane);
  dsv41_mma_f32_t mm(sg, lane);
  for (int k = 0; k < I; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lx.load();
    lw.load();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mm.mma(Xs, Ws);
    lx.next();
    lw.next();
  }
  for (short ti = 0; ti < 2; ++ti) {
    const int row = y_row + mm.sm + ti * 16;
    const int dr = sdst[row];
    if (dr < 0) { continue; }
    for (short tj = 0; tj < 2; ++tj) {
      const int col = y_col + mm.sn + tj * 16;
      const thread auto& f = mm.Ctile.frag_at(ti, tj);
      part[(long)dr * H + col] = (T)f[0];
      part[(long)dr * H + col + 1] = (T)f[1];
    }
  }
}

#define DSV41_MOE_Q_GROUPED(B)                                               \
  kernel void dsv41_moe_gu_q##B##_grouped(                                   \
      const device uchar* w1 [[buffer(0)]],                                  \
      const device bfloat* s1 [[buffer(1)]],                                 \
      const device bfloat* b1 [[buffer(2)]],                                 \
      const device uchar* w3 [[buffer(3)]],                                  \
      const device bfloat* s3 [[buffer(4)]],                                 \
      const device bfloat* b3 [[buffer(5)]],                                 \
      const device T* x [[buffer(6)]], device T* act [[buffer(7)]],          \
      const device int* srow [[buffer(8)]],                                  \
      const device float* swt [[buffer(9)]],                                 \
      const device int* tile2e [[buffer(10)]],                               \
      constant int& K [[buffer(11)]], constant int& I [[buffer(12)]],        \
      constant float& lim [[buffer(13)]], constant int& gs [[buffer(14)]],   \
      constant long& cw [[buffer(15)]], constant long& cs [[buffer(16)]],    \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float Xs[32 * 36], Gs[32 * 36], Us[32 * 36];                 \
    dsv41_moe_gu_q_grouped_impl_<B>(w1, s1, b1, w3, s3, b3, x, act, srow,    \
                                    swt, tile2e, K, I, lim, gs, cw, cs, tid, \
                                    sg, lane, Xs, Gs, Us);                   \
  }                                                                          \
  kernel void dsv41_moe_down_q##B##_grouped(                                 \
      const device uchar* w2 [[buffer(0)]],                                  \
      const device bfloat* s2 [[buffer(1)]],                                 \
      const device bfloat* b2 [[buffer(2)]],                                 \
      const device T* act [[buffer(3)]], device T* part [[buffer(4)]],       \
      const device int* sdst [[buffer(5)]],                                  \
      const device int* tile2e [[buffer(6)]],                                \
      constant int& I [[buffer(7)]], constant int& H [[buffer(8)]],          \
      constant int& gs [[buffer(9)]], constant long& cw [[buffer(10)]],      \
      constant long& cs [[buffer(11)]],                                      \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]])                               \
  {                                                                          \
    threadgroup float Xs[32 * 36], Ws[32 * 36];                              \
    dsv41_moe_down_q_grouped_impl_<B>(w2, s2, b2, act, part, sdst, tile2e,   \
                                      I, H, gs, cw, cs, tid, sg, lane, Xs,   \
                                      Ws);                                   \
  }

DSV41_MOE_Q_GROUPED(2)
DSV41_MOE_Q_GROUPED(3)
DSV41_MOE_Q_GROUPED(4)
DSV41_MOE_Q_GROUPED(5)
DSV41_MOE_Q_GROUPED(6)
DSV41_MOE_Q_GROUPED(8)
