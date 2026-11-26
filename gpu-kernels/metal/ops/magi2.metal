// magi2.metal -- MAGI-2-preview's own ops (generative-models/magi2/): the
// f32 GEMM its adapters, router and hyper-connections run in, the
// hyper-connection's pre and post steps, per-modality RMSNorm, q/k norm
// with the element-wise Fourier rope, the attention's sigmoid gate,
// swiglu7, and the multi-head MoE (route, fused grouped experts, combine).
// The bf16 linears and the attention are the shared kernels.
//
// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the bf16
// library (the one MAGI-2 runs). Math is f32 throughout; where the
// reference materializes a tensor in its dtype, so does this.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif
typedef VPIPE_ELT elt;

// ---- f32 GEMM ------------------------------------------------------------
//
// C[m, n] = rs[m] * sum_k A[m, k] B[n, k] + bias[n], all in f32 (A may be
// the element type, read and widened). Strided rows; a batch over z with
// per-z element offsets. rs (has_rs) and bias (has_bias) optional.
// Tiles 32 x 32, K in steps of 16, 4 simdgroups each a 16 x 16 quarter.
//   0:A 1:B 2:bias 3:rs 4:C 5:params
//   grid (ceil(N / 32) * 128, ceil(M / 32), Z), threadgroup (128, 1, 1)

struct Magi2G32 {
  int M, N, K;
  int lda, ldb, ldc;
  long az, bz, cz;      // element offsets per z
  int has_bias, has_rs;
};

template <typename TA>
static inline void
magi2_gemm32_impl(const device TA* A, const device float* B,
                  const device float* bias, const device float* rs,
                  device float* C, constant Magi2G32& p,
                  threadgroup float* As, threadgroup float* Bs,
                  threadgroup float* Cs, uint3 tg, uint lid, uint sg)
{
  constexpr int BM = 32, BN = 32, BK = 16;
  const int n0 = (int)tg.x * BN, m0 = (int)tg.y * BM;
  A += (long)tg.z * p.az;
  B += (long)tg.z * p.bz;
  C += (long)tg.z * p.cz;
  const int sm = (int)(sg >> 1) * 16, sn = (int)(sg & 1) * 16;
  simdgroup_matrix<float, 8, 8> acc[2][2];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int k0 = 0; k0 < p.K; k0 += BK) {
    // 32 x 16 of A and of B: 512 values each over 128 threads.
    for (int i = (int)lid; i < BM * BK; i += 128) {
      const int r = i / BK, c = i % BK;
      const int m = m0 + r, k = k0 + c;
      As[r * BK + c] = (m < p.M && k < p.K)
                           ? (float)A[(long)m * p.lda + k] : 0.0f;
      const int n = n0 + r;
      Bs[r * BK + c] = (n < p.N && k < p.K)
                           ? B[(long)n * p.ldb + k] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<float, 8, 8> a[2], b[2];
      for (int i = 0; i < 2; ++i) {
        simdgroup_load(a[i], As + (sm + 8 * i) * BK + kk, BK);
        // B^T: the tile's rows are n, its columns k.
        simdgroup_load(b[i], Bs + (sn + 8 * i) * BK + kk, BK, ulong2(0, 0),
                       true);
      }
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      simdgroup_store(acc[i][j], Cs + (sm + 8 * i) * BN + sn + 8 * j, BN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = (int)lid; i < BM * BN; i += 128) {
    const int r = i / BN, c = i % BN;
    const int m = m0 + r, n = n0 + c;
    if (m >= p.M || n >= p.N) { continue; }
    float v = Cs[i];
    if (p.has_rs != 0) { v *= rs[m]; }
    if (p.has_bias != 0) { v += bias[n]; }
    C[(long)m * p.ldc + n] = v;
  }
}

kernel void magi2_gemm32_a16(
    const device elt*     A    [[buffer(0)]],
    const device float*   B    [[buffer(1)]],
    const device float*   bias [[buffer(2)]],
    const device float*   rs   [[buffer(3)]],
    device float*         C    [[buffer(4)]],
    constant Magi2G32&    p    [[buffer(5)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float As[32 * 16], Bs[32 * 16], Cs[32 * 32];
  magi2_gemm32_impl<elt>(A, B, bias, rs, C, p, As, Bs, Cs, tg, lid, sg);
}

kernel void magi2_gemm32_a32(
    const device float*   A    [[buffer(0)]],
    const device float*   B    [[buffer(1)]],
    const device float*   bias [[buffer(2)]],
    const device float*   rs   [[buffer(3)]],
    device float*         C    [[buffer(4)]],
    constant Magi2G32&    p    [[buffer(5)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float As[32 * 16], Bs[32 * 16], Cs[32 * 32];
  magi2_gemm32_impl<float>(A, B, bias, rs, C, p, As, Bs, Cs, tg, lid, sg);
}

// ---- the pre-adapter's rows --------------------------------------------
//
// A span of the embedded sequence: f32 rows [rows, W] into the element-type
// residual, the first `tdim` channels REPLACED by the span's time embedding
// (the host's f32 vector, [tdim]: t * 1000, cos | sin).
//   0:X (f32) 1:S (elt) 2:temb 3:W 4:tdim
//   grid (W, rows)
kernel void magi2_embed_finish(
    const device float* X    [[buffer(0)]],
    device elt*         S    [[buffer(1)]],
    const device float* temb [[buffer(2)]],
    constant int&       W    [[buffer(3)]],
    constant int&       tdim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, r = (int)gid.y;
  if (c >= W) { return; }
  const long i = (long)r * W + c;
  S[i] = (elt)(c < tdim ? temb[c] : X[i]);
}

// ---- hyper-connections ---------------------------------------------------
//
// The pre step of one sublayer, per token (a threadgroup a row): the row's
// RMS over all n x C of the streams, the raw phi projection D (the f32
// GEMM's, the norm weight folded into phi) scaled by it, the coefficients
//   pre  = sigmoid(a_pre * s * hf[0:n] + b_pre)
//   post = 2 sigmoid(a_post * s * hf[n:2n] + b_post)
//   res  = Sinkhorn(a_res * s * hf[2n:] + b_res), `iters` alternations of
//          column then row normalization (each sum + eps),
// s = 1 / sqrt(n C), written as post | res [rows, n + n n] f32, and the
// sublayer's input  y[c] = sum_i pre[i] x[i, c]  in the element type.
//   0:S [rows, n C] 1:D [rows, ldd] 2:coef [rows, n + n n] 3:Y [rows, C]
//   4:ab (f32: a_pre a_post a_res, b_pre[n], b_post[n], b_res[n n])
//   5:C 6:n 7:ldd 8:eps 9:iters 10:sk_eps
//   grid (256 * rows), threadgroup 256
kernel void magi2_mhc_pre(
    const device elt*   S     [[buffer(0)]],
    const device float* D     [[buffer(1)]],
    device float*       coef  [[buffer(2)]],
    device elt*         Y     [[buffer(3)]],
    const device float* ab    [[buffer(4)]],
    constant int&       C     [[buffer(5)]],
    constant int&       n     [[buffer(6)]],
    constant int&       ldd   [[buffer(7)]],
    constant float&     eps   [[buffer(8)]],
    constant int&       iters [[buffer(9)]],
    constant float&     skeps [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl  [[thread_index_in_simdgroup]],
    uint sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float part[8];
  threadgroup float pre[4];
  const int W = n * C;
  const device elt* x = S + (long)row * W;
  float ss = 0.0f;
  for (int i = (int)lid; i < W; i += 256) {
    const float v = (float)x[i];
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (sl == 0) { part[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) { t += part[i]; }
    const float r = 1.0f / sqrt(t / (float)W + eps);
    const float sc = 1.0f / sqrt((float)W);
    const device float* d = D + (long)row * ldd;
    const float a_pre = ab[0], a_post = ab[1], a_res = ab[2];
    const device float* b_pre = ab + 3;
    const device float* b_post = ab + 3 + n;
    const device float* b_res = ab + 3 + 2 * n;
    device float* cf = coef + (long)row * (n + n * n);
    for (int i = 0; i < n; ++i) {
      pre[i] = 1.0f / (1.0f + exp(-(a_pre * sc * (r * d[i]) + b_pre[i])));
      cf[i] = 2.0f /
              (1.0f + exp(-(a_post * sc * (r * d[n + i]) + b_post[i])));
    }
    float m[16];
    float mx = -INFINITY;
    for (int i = 0; i < n * n; ++i) {
      m[i] = a_res * sc * (r * d[2 * n + i]) + b_res[i];
      mx = max(mx, m[i]);
    }
    for (int i = 0; i < n * n; ++i) { m[i] = exp(m[i] - mx); }
    for (int it = 0; it < iters; ++it) {
      for (int j = 0; j < n; ++j) {          // columns: sum over i
        float s = 0.0f;
        for (int i = 0; i < n; ++i) { s += m[i * n + j]; }
        s += skeps;
        for (int i = 0; i < n; ++i) { m[i * n + j] /= s; }
      }
      for (int i = 0; i < n; ++i) {          // rows: sum over j
        float s = 0.0f;
        for (int j = 0; j < n; ++j) { s += m[i * n + j]; }
        s += skeps;
        for (int j = 0; j < n; ++j) { m[i * n + j] /= s; }
      }
    }
    for (int i = 0; i < n * n; ++i) { cf[n + i] = m[i]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device elt* y = Y + (long)row * C;
  for (int c = (int)lid; c < C; c += 256) {
    float v = 0.0f;
    for (int i = 0; i < n; ++i) { v += pre[i] * (float)x[(long)i * C + c]; }
    y[c] = (elt)v;
  }
}

// The post step: x'[i, c] = sum_j res[i, j] x[j, c] + post[i] out[c], in
// place on the streams (a thread a channel of a row, all n streams).
//   0:S [rows, n C] 1:O [rows, C] 2:coef 3:C 4:n
//   grid (C, rows)
kernel void magi2_mhc_post(
    device elt*         S    [[buffer(0)]],
    const device elt*   O    [[buffer(1)]],
    const device float* coef [[buffer(2)]],
    constant int&       C    [[buffer(3)]],
    constant int&       n    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, row = (int)gid.y;
  if (c >= C) { return; }
  device elt* x = S + (long)row * n * C;
  const device float* cf = coef + (long)row * (n + n * n);
  const float o = (float)O[(long)row * C + c];
  float in[4];
  for (int j = 0; j < n; ++j) { in[j] = (float)x[(long)j * C + c]; }
  for (int i = 0; i < n; ++i) {
    float v = cf[i] * o;
    for (int j = 0; j < n; ++j) { v += cf[n + i * n + j] * in[j]; }
    x[(long)i * C + c] = (elt)v;
  }
}

// ---- RMSNorm, weight + 1 -------------------------------------------------
//
// A span of rows normalized over D in f32 and scaled by (w + 1), w the
// span's modality's f32 weight. Out in the element type.
//   0:X 1:Y 2:w 3:D 4:eps
//   grid (256 * rows), threadgroup 256
kernel void magi2_rms1(
    const device elt*   X   [[buffer(0)]],
    device elt*         Y   [[buffer(1)]],
    const device float* w   [[buffer(2)]],
    constant int&       D   [[buffer(3)]],
    constant float&     eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl  [[thread_index_in_simdgroup]],
    uint sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float part[8];
  threadgroup float rr;
  const device elt* x = X + (long)row * D;
  float ss = 0.0f;
  for (int i = (int)lid; i < D; i += 256) {
    const float v = (float)x[i];
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (sl == 0) { part[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) { t += part[i]; }
    rr = 1.0f / sqrt(t / (float)D + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device elt* y = Y + (long)row * D;
  for (int i = (int)lid; i < D; i += 256) {
    y[i] = (elt)((float)x[i] * rr * (w[i] + 1.0f));
  }
}

// Row sum of squares' inverse RMS: rs[row] = 1 / sqrt(mean(x^2) + eps).
//   0:X 1:rs 2:D 3:eps   grid (256 * rows), threadgroup 256
kernel void magi2_inv_rms(
    const device elt* X  [[buffer(0)]],
    device float*     rs [[buffer(1)]],
    constant int&     D  [[buffer(2)]],
    constant float&   eps [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl  [[thread_index_in_simdgroup]],
    uint sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float part[8];
  const device elt* x = X + (long)row * D;
  float ss = 0.0f;
  for (int i = (int)lid; i < D; i += 256) {
    const float v = (float)x[i];
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (sl == 0) { part[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) { t += part[i]; }
    rs[row] = 1.0f / sqrt(t / (float)D + eps);
  }
}

// ---- q / k norm and the element-wise rope --------------------------------
//
// In place on the q|k|v rows of a span: each of the n_q query and n_kv key
// heads (128 wide, at column h * 128 of the row) RMS-normalized in f32,
// scaled by (w + 1) (the span's modality's [128] weight), then dims [0, 2R)
// rotated in halves by the row's f32 sin / cos [R] (R = 48), the rest
// passed; rounded once to the element type. A simdgroup a (row, head).
//   0:QKV 1:sin 2:cos 3:wq 4:wk 5:ld 6:nq 7:nkv 8:R 9:eps 10:row0
//   grid (32 * (nq + nkv), rows), threadgroup (32, 1)
kernel void magi2_qk_rope(
    device elt*         QKV [[buffer(0)]],
    const device float* sn  [[buffer(1)]],
    const device float* cs  [[buffer(2)]],
    const device float* wq  [[buffer(3)]],
    const device float* wk  [[buffer(4)]],
    constant int&       ld  [[buffer(5)]],
    constant int&       nq  [[buffer(6)]],
    constant int&       nkv [[buffer(7)]],
    constant int&       R   [[buffer(8)]],
    constant float&     eps [[buffer(9)]],
    constant int&       row0 [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint  sl  [[thread_index_in_simdgroup]])
{
  threadgroup float hb[128];
  const int head = (int)gid.x / 32;
  const int row = (int)gid.y;
  const int grow = row0 + row;
  if (head >= nq + nkv) { return; }
  device elt* x = QKV + (long)grow * ld + head * 128;
  const device float* w = head < nq ? wq : wk;
  float v[4];
  float ss = 0.0f;
  for (int i = 0; i < 4; ++i) {
    v[i] = (float)x[sl + 32 * i];
    ss += v[i] * v[i];
  }
  ss = simd_sum(ss);
  const float r = 1.0f / sqrt(ss / 128.0f + eps);
  for (int i = 0; i < 4; ++i) {
    const int d = (int)sl + 32 * i;
    v[i] = v[i] * r * (w[d] + 1.0f);
  }
  // Pairs (d, d + R) for d < R: lanes hold d = sl + 32 i; exchange through
  // threadgroup-free shuffles is awkward across i, so go through memory
  // the simdgroup owns: write the normed head, then rotate from it.
  for (int i = 0; i < 4; ++i) { hb[sl + 32 * i] = v[i]; }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const device float* sr = sn + (long)grow * R;
  const device float* cr = cs + (long)grow * R;
  for (int i = 0; i < 4; ++i) {
    const int d = (int)sl + 32 * i;
    float o;
    if (d < R) {
      o = hb[d] * cr[d] - hb[d + R] * sr[d];
    } else if (d < 2 * R) {
      o = hb[d] * cr[d - R] + hb[d - R] * sr[d - R];
    } else {
      o = hb[d];
    }
    x[d] = (elt)o;
  }
}

// ---- attention's gate ----------------------------------------------------
//
// O[r, h D + d] *= sigmoid(g[r, h]), g read where the projection wrote it
// (column goff of a row of ldg).
//   0:O 1:G 2:heads 3:D 4:ldg 5:goff
//   grid (heads * D, rows)
kernel void magi2_gate(
    device elt*       O    [[buffer(0)]],
    const device elt* G    [[buffer(1)]],
    constant int&     heads [[buffer(2)]],
    constant int&     D    [[buffer(3)]],
    constant int&     ldg  [[buffer(4)]],
    constant int&     goff [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, r = (int)gid.y;
  if (c >= heads * D) { return; }
  const float g = (float)G[(long)r * ldg + goff + c / D];
  const long i = (long)r * heads * D + c;
  O[i] = (elt)((float)O[i] / (1.0f + exp(-g)));
}

// ---- swiglu7 ---------------------------------------------------------------
//
// [rows, 2 I] -> [rows, I]: gate = x[2 j] clamped above at 7, up =
// x[2 j + 1] clamped to [-7, 7]; gate sigmoid(1.702 gate) (up + 1).
//   0:X 1:Y 2:I   grid (I, rows)
static inline float
magi2_swiglu7(float g, float u)
{
  g = min(g, 7.0f);
  u = clamp(u, -7.0f, 7.0f);
  return g / (1.0f + exp(-1.702f * g)) * (u + 1.0f);
}

kernel void magi2_swiglu7(
    const device elt* X [[buffer(0)]],
    device elt*       Y [[buffer(1)]],
    constant int&     I [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int j = (int)gid.x, r = (int)gid.y;
  if (j >= I) { return; }
  const device elt* x = X + (long)r * 2 * I;
  Y[(long)r * I + j] = (elt)magi2_swiglu7((float)x[2 * j], (float)x[2 * j + 1]);
}

// ---- the multi-head MoE ----------------------------------------------------
//
// ROUTE: per (token, head) row of logits [E] (f32, the router GEMM's, at
// column head * E of a token's row of ldl), sigmoid scores; the top k of
// score + bias[head, e] (the EMA bias the release routes with); probs the
// chosen scores L1-normalized, times `scale`. Writes per pair p = (token *
// heads + head) * k + j the GLOBAL expert head * E + e and its prob. A
// simdgroup a row; E <= 32 * kMagi2MaxEpt.
//   0:L 1:bias 2:eid 3:prob 4:heads 5:E 6:topk 7:scale 8:ldl
//   grid (32 * heads, tokens), threadgroup (32, 1)
constant constexpr int kMagi2MaxEpt = 16;

kernel void magi2_moe_topk(
    const device float* L     [[buffer(0)]],
    const device float* bias  [[buffer(1)]],
    device int*         eid   [[buffer(2)]],
    device float*       prob  [[buffer(3)]],
    constant int&       heads [[buffer(4)]],
    constant int&       E     [[buffer(5)]],
    constant int&       topk  [[buffer(6)]],
    constant float&     scale [[buffer(7)]],
    constant int&       ldl   [[buffer(8)]],
    // Tests: picks given, [tokens, heads, topk] head-local experts (the
    // probs still from this row's scores); has_forced 0 ignores it.
    const device int*   forced [[buffer(9)]],
    constant int&       has_forced [[buffer(10)]],
    // A RANK's experts (tensor parallel): [e0, e1) of each head's E. A pick
    // in it is written as this rank's expert, head * (e1 - e0) + e - e0; one
    // outside, as the dropped bin heads * (e1 - e0), which the rank does
    // not compute (magi2_moe_drop_bin, magi2_moe_combine). [0, E): the
    // whole model, every pick head * E + e as before.
    constant int&       e0    [[buffer(11)]],
    constant int&       e1    [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]],
    uint  sl  [[thread_index_in_simdgroup]])
{
  const int head = (int)gid.x / 32;
  const int tok = (int)gid.y;
  if (head >= heads) { return; }
  const int ept = (E + 31) / 32;
  const device float* l = L + (long)tok * ldl + head * E;
  const device float* b = bias + head * E;
  float sc[kMagi2MaxEpt], key[kMagi2MaxEpt];
  for (int i = 0; i < ept; ++i) {
    const int e = (int)sl * ept + i;
    if (e < E) {
      sc[i] = 1.0f / (1.0f + precise::exp(-l[e]));
      key[i] = sc[i] + b[e];
    } else {
      sc[i] = 0.0f;
      key[i] = -INFINITY;
    }
  }
  float sum = 0.0f;
  float chosen_sc[8];
  int chosen_e[8];
  for (int j = 0; j < topk; ++j) {
    // This lane's best; then the simdgroup's, lowest expert on a tie.
    float best = -INFINITY;
    int bi = -1;
    for (int i = 0; i < ept; ++i) {
      if (key[i] > best) { best = key[i]; bi = i; }
    }
    const float mx = simd_max(best);
    const int cand = (best == mx && bi >= 0) ? (int)sl * ept + bi : 1 << 30;
    int win = simd_min(cand);
    if (has_forced != 0) {
      win = forced[((long)tok * heads + head) * topk + j];
    }
    const int lane = win / ept;
    float wsc = 0.0f;
    if ((int)sl == lane) {
      wsc = sc[win % ept];
      key[win % ept] = -INFINITY;
    }
    wsc = simd_broadcast(wsc, (ushort)lane);
    chosen_sc[j] = wsc;
    chosen_e[j] = win;
    sum += wsc;
  }
  if (sl == 0) {
    const float den = max(sum, 1e-12f);
    const long p0 = ((long)tok * heads + head) * topk;
    const int el = e1 - e0;
    for (int j = 0; j < topk; ++j) {
      const int e = chosen_e[j];
      eid[p0 + j] = e >= e0 && e < e1 ? head * el + (e - e0) : heads * el;
      prob[p0 + j] = chosen_sc[j] / den * scale;
    }
  }
}

// A RANK's dropped bin (magi2_moe_topk): its tiles taken off the expert
// kernels' grid -- a tile of -1 is skipped -- so the picks of the other
// ranks' experts are sorted but never computed.
kernel void magi2_moe_drop_bin(
    device int*   tile2e   [[buffer(0)]],
    constant int& maxtiles [[buffer(1)]],
    constant int& bin      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)maxtiles) { return; }
  if (tile2e[gid] == bin) { tile2e[gid] = -1; }
}

// pos[sdst[slot]] = slot -- each pair's slot in the sorted order.
//   0:sdst 1:pos 2:npad   grid (npad)
kernel void magi2_moe_invert(
    const device int* sdst [[buffer(0)]],
    device int*       pos  [[buffer(1)]],
    constant int&     npad [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)npad) { return; }
  const int p = sdst[gid];
  if (p >= 0) { pos[p] = (int)gid; }
}

// THE EXPERTS, fused: a tile of TM sorted slots of ONE expert (its rows of
// 256 gathered from X [rows, 256] by srow), through
//   g = x Wg[e], u = x Wu[e]         ([256, DE] each, f32 accumulation)
//   h = swiglu7(g, u)                (rounded to the element type)
//   y += h Wd[e]                     ([DE, 256])
// over the expert's width in chunks of DE, never materializing h beyond a
// chunk; y times the pair's prob, written f32 [npad, 256]. Weights as the
// release stores them: Wg / Wu [E, 256, I] and Wd [E, I, 256], row-major.
// 4 simdgroups; each owns 8 rows of g / u and 64 columns of y.
//   0:X 1:srow 2:sdst 3:prob 4:tile2e 5:Wg 6:Wu 7:Wd 8:Y 9:I
//   grid (128 * tiles), threadgroup 128
constant constexpr int kMagi2Tm = 32;
constant constexpr int kMagi2De = 32;
constant constexpr int kMagi2Dh = 256;

kernel void magi2_moe_expert(
    const device elt*   X      [[buffer(0)]],
    const device int*   srow   [[buffer(1)]],
    const device int*   sdst   [[buffer(2)]],
    const device float* prob   [[buffer(3)]],
    const device int*   tile2e [[buffer(4)]],
    const device elt*   Wg     [[buffer(5)]],
    const device elt*   Wu     [[buffer(6)]],
    const device elt*   Wd     [[buffer(7)]],
    device float*       Y      [[buffer(8)]],
    constant int&       I      [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
  constexpr int TM = kMagi2Tm, DE = kMagi2De, DH = kMagi2Dh;
  const int e = tile2e[tile];
  if (e < 0) { return; }
  threadgroup elt Xs[TM * DH];
  threadgroup elt Hs[TM * DE];
  // Per simdgroup: g / u staging for the activation, then y's row bands.
  threadgroup float Ws[4][8 * 64];
  threadgroup float* ws = Ws[sg];
  const int s0 = (int)tile * TM;
  for (int i = (int)lid; i < TM * DH; i += 128) {
    const int r = i / DH, c = i % DH;
    const int row = srow[s0 + r];
    Xs[i] = row >= 0 ? X[(long)row * DH + c] : (elt)0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device elt* wg = Wg + (long)e * DH * I;
  const device elt* wu = Wu + (long)e * DH * I;
  const device elt* wd = Wd + (long)e * I * DH;
  simdgroup_matrix<float, 8, 8> y[4][8];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
      y[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  const int gr = (int)sg * 8;          // this simdgroup's rows of g / u
  const int yc = (int)sg * 64;         // and its columns of y
  for (int c0 = 0; c0 < I; c0 += DE) {
    simdgroup_matrix<float, 8, 8> g[4], u[4];
    for (int j = 0; j < 4; ++j) {
      g[j] = simdgroup_matrix<float, 8, 8>(0.0f);
      u[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (int k = 0; k < DH; k += 8) {
      simdgroup_matrix<elt, 8, 8> a;
      simdgroup_load(a, Xs + gr * DH + k, DH);
      for (int j = 0; j < 4; ++j) {
        simdgroup_matrix<elt, 8, 8> bg, bu;
        simdgroup_load(bg, wg + (long)k * I + c0 + 8 * j, I);
        simdgroup_load(bu, wu + (long)k * I + c0 + 8 * j, I);
        simdgroup_multiply_accumulate(g[j], a, bg, g[j]);
        simdgroup_multiply_accumulate(u[j], a, bu, u[j]);
      }
    }
    // swiglu7 over this simdgroup's 8 x DE, through its own scratch.
    for (int j = 0; j < 4; ++j) {
      simdgroup_store(g[j], ws, 8);
      simdgroup_store(u[j], ws + 64, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int q = (int)(lid & 31); q < 64; q += 32) {
        const int r = q / 8, c = q % 8;
        Hs[(gr + r) * DE + 8 * j + c] =
            (elt)magi2_swiglu7(ws[q], ws[64 + q]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < DE; k += 8) {
      simdgroup_matrix<elt, 8, 8> a[4];
      for (int i = 0; i < 4; ++i) {
        simdgroup_load(a[i], Hs + (8 * i) * DE + k, DE);
      }
      for (int j = 0; j < 8; ++j) {
        simdgroup_matrix<elt, 8, 8> b;
        simdgroup_load(b, wd + (long)(c0 + k) * DH + yc + 8 * j, DH);
        for (int i = 0; i < 4; ++i) {
          simdgroup_multiply_accumulate(y[i][j], a[i], b, y[i][j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Out: each simdgroup its 32 x 64, a row band of 8 at a time.
  threadgroup float* ys = ws;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
      simdgroup_store(y[i][j], ys + 8 * j, 64);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = (int)(lid & 31); q < 8 * 64; q += 32) {
      const int r = q / 64, c = q % 64;
      const int slot = s0 + 8 * i + r;
      const int p = sdst[slot];
      if (p >= 0) {
        Y[(long)slot * DH + yc + c] = ys[q] * prob[p];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// THE EXPERTS, STAGED: magi2_moe_expert's tile, buffers and products, for
// a GPU without matrix cores -- the same 8-deep products accumulated in
// the same order, so its outputs are the kernel above's BIT FOR BIT. What
// the one above does not:
//   * every weight slice is STAGED in threadgroup memory by all 256
//     threads in coalesced 16-byte loads and shared by the 8 simdgroups
//     (above, each simdgroup loads its own 8 x 8 tiles straight from
//     device memory, 16 bytes a row at the expert's whole width as the
//     stride, and the four load the SAME gate / up tiles);
//   * 8 SIMDGROUPS: y's [32, 256] is 32 x 32 a simdgroup -- 32 floats a
//     thread, g / u's [32, 64] chunk 16 x 16 (16 more) -- where the one
//     above holds 64 + 16 on four; the gathered rows staged a K slice at a
//     time beside the weights' (16 KB of threadgroup memory a group);
//   * swiglu7 on the accumulators' registers (an 8 x 8 accumulator holds a
//     thread's row fm, columns fn and fn + 1: MLX's steel layout), written
//     straight into h.
// Per chunk of DE = 64 of the expert's width:
//   g, u [32, 64] = x [32, 256] Wg / Wu [256, 64]   K staged 32 at a time
//   h = swiglu7(g, u) (the element type)            into Hs
//   y [32, 256] += h Wd [64, 256]                   K staged 16 at a time
// MEASURED on the M4 Pro (magi2.preview_bench at 16 x 32 x 56, one MoE
// layer at the release's widths, a batch of two -- 8.2 TFLOP of experts):
// the kernel above 5.87 s; this 1.45-1.57 s (3.8-4.0x, ~5.5 TFLOP/s).
// Tried and dropped: the same staging on four simdgroups (y 64 floats a
// thread, the whole gathered rows staged: 31 KB) 4.93 s -- occupancy, not
// the loads, was the limit; 64-row work items on 16 simdgroups (the
// weights staged once for 64 rows) 1.97-2.05 s; K staged 16 / 8 deep
// (11 KB) a tie.
// Requires I % 64 == 0 (else the kernel above).
//   0:X 1:srow 2:sdst 3:prob 4:tile2e 5:Wg 6:Wu 7:Wd 8:Y 9:I
//   grid (256 * tiles), threadgroup 256
constant constexpr int kMagi2Se = 64;     // the hidden chunk
constant constexpr int kMagi2Ska = 32;    // g / u's K staging depth
constant constexpr int kMagi2Skd = 16;    // y's K staging depth

kernel void magi2_moe_expert_tg(
    const device elt*   X      [[buffer(0)]],
    const device int*   srow   [[buffer(1)]],
    const device int*   sdst   [[buffer(2)]],
    const device float* prob   [[buffer(3)]],
    const device int*   tile2e [[buffer(4)]],
    const device elt*   Wg     [[buffer(5)]],
    const device elt*   Wu     [[buffer(6)]],
    const device elt*   Wd     [[buffer(7)]],
    device float*       Y      [[buffer(8)]],
    constant int&       I      [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sl   [[thread_index_in_simdgroup]])
{
  constexpr int TM = kMagi2Tm, DH = kMagi2Dh, DE = kMagi2Se;
  constexpr int SKA = kMagi2Ska, SKD = kMagi2Skd;
  constexpr int ALD = SKA + 8, HLD = DE + 8, BLA = DE + 8, BLD = DH + 8;
  constexpr int NT = 256;
  const int e = tile2e[tile];
  if (e < 0) { return; }
  threadgroup elt As[TM * ALD];
  threadgroup elt Hs[TM * HLD];
  // g / u's [32 x BLA] slices of Wg and Wu, or y's [16 x BLD] of Wd.
  threadgroup elt Bs[2 * SKA * BLA > SKD * BLD ? 2 * SKA * BLA : SKD * BLD];
  const int s0 = (int)tile * TM;
  // This thread's share of the gathered rows' K slices: row ar, vector av.
  const device elt* xa = nullptr;
  int ar = 0, av = 0;
  if ((int)lid < TM * (SKA / 8)) {
    ar = (int)lid / (SKA / 8);
    av = (int)lid % (SKA / 8);
    const int row = srow[s0 + ar];
    xa = row >= 0 ? X + (long)row * DH + 8 * av : nullptr;
  }
  const device elt* wg = Wg + (long)e * DH * I;
  const device elt* wu = Wu + (long)e * DH * I;
  const device elt* wd = Wd + (long)e * I * DH;
  // The accumulators' element layout: (row fm, columns fn, fn + 1).
  const short qid = (short)(sl / 4);
  const short fm = (qid & 4) + (short)((sl / 2) % 4);
  const short fn = (qid & 2) * 2 + (short)(sl % 2) * 2;
  const int gr = ((int)sg / 4) * 16, gc = ((int)sg % 4) * 16;
  const int yc = (int)sg * 32;
  simdgroup_matrix<float, 8, 8> y[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      y[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  for (int c0 = 0; c0 < I; c0 += DE) {
    simdgroup_matrix<float, 8, 8> g[2][2], u[2][2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        g[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        u[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
      }
    }
    for (int k0 = 0; k0 < DH; k0 += SKA) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if ((int)lid < TM * (SKA / 8)) {
        *(threadgroup uint4*)(As + ar * ALD + 8 * av) =
            xa != nullptr ? *(const device uint4*)(xa + k0) : uint4(0);
      }
      // Wg's and Wu's [32 x 64] slices: 2 x 32 rows of 8 vectors.
      for (int i = (int)lid; i < 2 * SKA * (DE / 8); i += NT) {
        const int m = i / (SKA * (DE / 8));
        const int q = i % (SKA * (DE / 8));
        const int r = q / (DE / 8), v = q % (DE / 8);
        const device elt* src = (m == 0 ? wg : wu) +
                                (long)(k0 + r) * I + c0 + 8 * v;
        *(threadgroup uint4*)(Bs + (m * SKA + r) * BLA + 8 * v) =
            *(const device uint4*)src;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kk = 0; kk < SKA; kk += 8) {
        simdgroup_matrix<elt, 8, 8> a[2], bg[2], bu[2];
        for (int i = 0; i < 2; ++i) {
          simdgroup_load(a[i], As + (gr + 8 * i) * ALD + kk, ALD);
        }
        for (int j = 0; j < 2; ++j) {
          simdgroup_load(bg[j], Bs + kk * BLA + gc + 8 * j, BLA);
          simdgroup_load(bu[j], Bs + (SKA + kk) * BLA + gc + 8 * j, BLA);
        }
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            simdgroup_multiply_accumulate(g[i][j], a[i], bg[j], g[i][j]);
            simdgroup_multiply_accumulate(u[i][j], a[i], bu[j], u[i][j]);
          }
        }
      }
    }
    // swiglu7 on the registers, into this simdgroup's 16 x 16 of h.
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        const thread auto& gv = g[i][j].thread_elements();
        const thread auto& uv = u[i][j].thread_elements();
        threadgroup elt* hr =
            Hs + (gr + 8 * i + fm) * HLD + gc + 8 * j + fn;
        hr[0] = (elt)magi2_swiglu7(gv[0], uv[0]);
        hr[1] = (elt)magi2_swiglu7(gv[1], uv[1]);
      }
    }
    for (int kd0 = 0; kd0 < DE; kd0 += SKD) {
      // (The first barrier also orders every simdgroup's h before reads.)
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = (int)lid; i < SKD * (DH / 8); i += NT) {
        const int r = i / (DH / 8), v = i % (DH / 8);
        *(threadgroup uint4*)(Bs + r * BLD + 8 * v) =
            *(const device uint4*)(wd + (long)(c0 + kd0 + r) * DH + 8 * v);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kk = 0; kk < SKD; kk += 8) {
        simdgroup_matrix<elt, 8, 8> a[4], b[4];
        for (int i = 0; i < 4; ++i) {
          simdgroup_load(a[i], Hs + (8 * i) * HLD + kd0 + kk, HLD);
        }
        for (int j = 0; j < 4; ++j) {
          simdgroup_load(b[j], Bs + kk * BLD + yc + 8 * j, BLD);
        }
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            simdgroup_multiply_accumulate(y[i][j], a[i], b[j], y[i][j]);
          }
        }
      }
    }
  }
  // Out: y times the pair's prob at the slot's row, f32, from the
  // registers.
  for (int i = 0; i < 4; ++i) {
    const int slot = s0 + 8 * i + fm;
    const int p = sdst[slot];
    if (p < 0) { continue; }
    const float pr = prob[p];
    device float* o = Y + (long)slot * DH + yc + fn;
    for (int j = 0; j < 4; ++j) {
      const thread auto& yv = y[i][j].thread_elements();
      o[8 * j] = yv[0] * pr;
      o[8 * j + 1] = yv[1] * pr;
    }
  }
}

// COMBINE: out[row, d] = sum_j Y[pos[row * k + j], d] in pick order, f32,
// rounded once. Rows are (token, head) of 256.
//   0:Y 1:pos 2:out 3:topk   grid (256, rows)
kernel void magi2_moe_combine(
    const device float* Y    [[buffer(0)]],
    const device int*   pos  [[buffer(1)]],
    device elt*         out  [[buffer(2)]],
    constant int&       topk [[buffer(3)]],
    // A rank's picks (magi2_moe_topk), and its dropped bin: those picks'
    // rows hold nothing and are left out. -1: none is (the whole model).
    const device int*   eid  [[buffer(4)]],
    constant int&       drop [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, row = (int)gid.y;
  if (d >= kMagi2Dh) { return; }
  float v = 0.0f;
  for (int j = 0; j < topk; ++j) {
    const long p = (long)row * topk + j;
    if (drop >= 0 && eid[p] == drop) { continue; }
    v += Y[(long)pos[p] * kMagi2Dh + d];
  }
  out[(long)row * kMagi2Dh + d] = (elt)v;
}

// ---- element-wise ----------------------------------------------------------
//
// y = a + b, rounded once.   0:A 1:B 2:Y 3:n   grid (n)
kernel void magi2_add(
    const device elt* A [[buffer(0)]],
    const device elt* B [[buffer(1)]],
    device elt*       Y [[buffer(2)]],
    constant uint&    n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  Y[gid] = (elt)((float)A[gid] + (float)B[gid]);
}

// ---- the turbo decoder (TurboV3-Wan22-TinyShallow) ------------------------
//
// Activations CHANNEL-LAST [T, H, W, C]. A conv is an im2col gather and the
// shared GEMM: the gather writes, for output voxels [v0, v0 + nv) of the
// window (voxel v = (t * H + h) * W + w), their taps' inputs tap-major --
// (dt, dh, dw), then the channel -- reading time CLAMPED to the window
// (the release pads time by repeating its edge frames) and space ZERO
// outside the picture.
//   0:X [T, H, W, C] 1:Y [nv, taps * C] 2:T 3:H 4:W 5:C 6:kt 7:kh 8:kw
//   9:v0 10:nv
//   grid (C, kt * kh * kw, nv)
kernel void magi2_im2col(
    const device elt* X  [[buffer(0)]],
    device elt*       Y  [[buffer(1)]],
    constant int&     T  [[buffer(2)]],
    constant int&     H  [[buffer(3)]],
    constant int&     W  [[buffer(4)]],
    constant int&     C  [[buffer(5)]],
    constant int&     kt [[buffer(6)]],
    constant int&     kh [[buffer(7)]],
    constant int&     kw [[buffer(8)]],
    constant long&    v0 [[buffer(9)]],
    constant int&     nv [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, tap = (int)gid.y, r = (int)gid.z;
  if (c >= C || r >= nv) { return; }
  const int taps = kt * kh * kw;
  const long v = v0 + r;
  const int w = (int)(v % W);
  const int h = (int)((v / W) % H);
  const int t = (int)(v / ((long)W * H));
  const int dt = tap / (kh * kw) - (kt - 1) / 2;
  const int dh = (tap / kw) % kh - (kh - 1) / 2;
  const int dw = tap % kw - (kw - 1) / 2;
  const int tt = clamp(t + dt, 0, T - 1);
  const int hh = h + dh, ww = w + dw;
  elt val = (elt)0.0f;
  if (hh >= 0 && hh < H && ww >= 0 && ww < W) {
    val = X[(((long)tt * H + hh) * W + ww) * C + c];
  }
  Y[(long)r * taps * C + (long)tap * C + c] = val;
}

// The same gather four channels a thread (8 bytes), on a 2-D grid with
// 32-bit index math: the 3-D grid above launched a threadgroup of C
// threads for every (tap, voxel), ~16 of a decode's 19 s (MEASURED, M4
// Pro, 25 frames at 384 x 480). C a multiple of 4; a band's voxels and
// a window's elements under 2^32.
//   0:X 1:Y 2:T 3:H 4:W 5:C 6:kt 7:kh 8:kw 9:v0 10:nv
//   grid (taps * C / 4, nv), threadgroup (256, 1, 1)
kernel void magi2_im2col4(
    const device uint2* X  [[buffer(0)]],
    device uint2*       Y  [[buffer(1)]],
    constant int&       T  [[buffer(2)]],
    constant int&       H  [[buffer(3)]],
    constant int&       W  [[buffer(4)]],
    constant int&       C  [[buffer(5)]],
    constant int&       kt [[buffer(6)]],
    constant int&       kh [[buffer(7)]],
    constant int&       kw [[buffer(8)]],
    constant long&      v0 [[buffer(9)]],
    constant int&       nv [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
  const uint C4 = (uint)C / 4u;
  const uint taps = (uint)(kt * kh * kw);
  if (gid.x >= taps * C4 || gid.y >= (uint)nv) { return; }
  const uint c4 = gid.x % C4, tap = gid.x / C4;
  const uint v = (uint)v0 + gid.y;
  const uint w = v % (uint)W;
  const uint h = (v / (uint)W) % (uint)H;
  const uint t = v / ((uint)W * (uint)H);
  const int dt = (int)(tap / (uint)(kh * kw)) - (kt - 1) / 2;
  const int dh = (int)((tap / (uint)kw) % (uint)kh) - (kh - 1) / 2;
  const int dw = (int)(tap % (uint)kw) - (kw - 1) / 2;
  const int tt = clamp((int)t + dt, 0, T - 1);
  const int hh = (int)h + dh, ww = (int)w + dw;
  uint2 val = uint2(0u, 0u);
  if (hh >= 0 && hh < H && ww >= 0 && ww < W) {
    val = X[(((uint)tt * (uint)H + (uint)hh) * (uint)W + (uint)ww) * C4 + c4];
  }
  Y[gid.y * taps * C4 + gid.x] = val;
}

// RMSNorm over the channels of each voxel, no affine (eps), then an
// activation: 0 none, 1 SiLU, 2 ReLU; f32 inside, rounded once. A
// simdgroup a voxel.   0:X 1:Y 2:C 3:eps 4:act   grid (32, voxels)
kernel void magi2_rms_act(
    const device elt* X   [[buffer(0)]],
    device elt*       Y   [[buffer(1)]],
    constant int&     C   [[buffer(2)]],
    constant float&   eps [[buffer(3)]],
    constant int&     act [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint  sl  [[thread_index_in_simdgroup]])
{
  const long v = gid.y;
  const device elt* x = X + v * C;
  float ss = 0.0f;
  for (int c = (int)sl; c < C; c += 32) {
    const float a = (float)x[c];
    ss += a * a;
  }
  ss = simd_sum(ss);
  const float r = 1.0f / sqrt(ss / (float)C + eps);
  device elt* y = Y + v * C;
  for (int c = (int)sl; c < C; c += 32) {
    float a = (float)x[c] * r;
    if (act == 1) { a = a / (1.0f + exp(-a)); }
    else if (act == 2) { a = max(a, 0.0f); }
    y[c] = (elt)a;
  }
}

// Nearest x2 in space, channel-last: [T, H, W, C] -> [T, 2H, 2W, C].
//   0:X 1:Y 2:H 3:W 4:C   grid (C, 2W * 2H, T)
kernel void magi2_up2(
    const device elt* X [[buffer(0)]],
    device elt*       Y [[buffer(1)]],
    constant int&     H [[buffer(2)]],
    constant int&     W [[buffer(3)]],
    constant int&     C [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, p = (int)gid.y, t = (int)gid.z;
  if (c >= C || p >= 4 * H * W) { return; }
  const int oh = p / (2 * W), ow = p % (2 * W);
  Y[((long)t * 4 * H * W + p) * C + c] =
      X[(((long)t * H + oh / 2) * W + ow / 2) * C + c];
}

// The time upsampler's split: [t, HW, 2C] (channel n * C + c is frame
// 2 t + n's channel c) -> [2t - skip, HW, C], the first frame dropped when
// `skip` (the release's first window).
//   0:X 1:Y 2:HW 3:C 4:skip   grid (C, HW, 2t - skip)
kernel void magi2_time_split(
    const device elt* X    [[buffer(0)]],
    device elt*       Y    [[buffer(1)]],
    constant int&     HW   [[buffer(2)]],
    constant int&     C    [[buffer(3)]],
    constant int&     skip [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, p = (int)gid.y, f = (int)gid.z;
  if (c >= C || p >= HW) { return; }
  const int src = f + skip;               // frame in the doubled stream
  const int t = src / 2, n = src % 2;
  Y[((long)f * HW + p) * C + c] =
      X[((long)t * HW + p) * 2 * C + n * C + c];
}

// The 2x2 unpatchify into planar f32 frames: channel c * 4 + b * 2 + a of
// [F, h, w, 4 C] is pixel (2 y + a, 2 x + b) of channel c.
//   0:X 1:Y [C, F, 2h, 2w] 2:h 3:w 4:C 5:F 6:f0 (first frame written)
//   7:Ftot   grid (2w, 2h, F)
kernel void magi2_unpatch(
    const device elt* X    [[buffer(0)]],
    device float*     Y    [[buffer(1)]],
    constant int&     h    [[buffer(2)]],
    constant int&     w    [[buffer(3)]],
    constant int&     C    [[buffer(4)]],
    constant int&     F    [[buffer(5)]],
    constant int&     f0   [[buffer(6)]],
    constant int&     Ftot [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int ox = (int)gid.x, oy = (int)gid.y, f = (int)gid.z;
  if (ox >= 2 * w || oy >= 2 * h || f >= F) { return; }
  const int x = ox / 2, b = ox % 2, y = oy / 2, a = oy % 2;
  const device elt* px = X + (((long)f * h + y) * w + x) * 4 * C;
  for (int c = 0; c < C; ++c) {
    Y[(((long)c * Ftot + f0 + f) * 2 * h + oy) * 2 * w + ox] =
        (float)px[c * 4 + b * 2 + a];
  }
}

// The latent into the window: [48, T, h, w] f32 planar -> channel-last
// element type, un-whitened z * std + mean (rounded to the element type,
// as the release's z.to(z_dtype)), frames [t0, t0 + n) of the latent.
//   0:Z 1:Y 2:mean 3:std 4:T 5:HW 6:C 7:t0   grid (C, HW, n)
kernel void magi2_latent_in(
    const device float* Z    [[buffer(0)]],
    device elt*         Y    [[buffer(1)]],
    const device float* mean [[buffer(2)]],
    const device float* sd   [[buffer(3)]],
    constant int&       T    [[buffer(4)]],
    constant int&       HW   [[buffer(5)]],
    constant int&       C    [[buffer(6)]],
    constant int&       t0   [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, p = (int)gid.y, f = (int)gid.z;
  if (c >= C || p >= HW) { return; }
  const int t = min(t0 + f, T - 1);       // padding repeats the last
  const float z = (float)(elt)Z[((long)c * T + t) * HW + p];
  Y[((long)f * HW + p) * C + c] = (elt)(z / (1.0f / sd[c]) + mean[c]);
}

// ---- the refiner ----------------------------------------------------------
//
// Its residual stream is f32 (the release's bf16 stream turns f32 at
// layer 0's MLP add, bf16 + f32), so the norms read f32 rows and the
// sublayer outputs are added into f32.

// RMSNorm (weight + 1) of f32 rows, rounded once to the element type.
//   0:X f32 [rows, D] 1:Y elt 2:w f32 [D] 3:D 4:eps
//   grid (256 * rows), threadgroup 256
kernel void magi2_rms1_f32(
    const device float* X   [[buffer(0)]],
    device elt*         Y   [[buffer(1)]],
    const device float* w   [[buffer(2)]],
    constant int&       D   [[buffer(3)]],
    constant float&     eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl  [[thread_index_in_simdgroup]],
    uint sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float part[8];
  threadgroup float rr;
  const device float* x = X + (long)row * D;
  float ss = 0.0f;
  for (int i = (int)lid; i < D; i += 256) { ss += x[i] * x[i]; }
  ss = simd_sum(ss);
  if (sl == 0) { part[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) { t += part[i]; }
    rr = 1.0f / sqrt(t / (float)D + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device elt* y = Y + (long)row * D;
  for (int i = (int)lid; i < D; i += 256) {
    y[i] = (elt)(x[i] * rr * (w[i] + 1.0f));
  }
}

// 1 / RMS of f32 rows (the head's norm, its weight folded into the head).
//   0:X f32 [rows, D] 1:rs f32 [rows] 2:D 3:eps
kernel void magi2_inv_rms_f32(
    const device float* X  [[buffer(0)]],
    device float*       rs [[buffer(1)]],
    constant int&       D  [[buffer(2)]],
    constant float&     eps [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sl  [[thread_index_in_simdgroup]],
    uint sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup float part[8];
  const device float* x = X + (long)row * D;
  float ss = 0.0f;
  for (int i = (int)lid; i < D; i += 256) { ss += x[i] * x[i]; }
  ss = simd_sum(ss);
  if (sl == 0) { part[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) { t += part[i]; }
    rs[row] = 1.0f / sqrt(t / (float)D + eps);
  }
}

// X (f32) += Y (elt); with `rnd`, the sum rounded to the element type --
// the release's bf16 + bf16 at layer 0's attention add. Rows of D, Y's
// rows ldy apart.
//   0:X f32 [rows, D] 1:Y elt 2:D 3:ldy 4:rnd
//   grid (D, rows)
kernel void magi2_resid_add(
    device float*     X   [[buffer(0)]],
    const device elt* Y   [[buffer(1)]],
    constant int&     D   [[buffer(2)]],
    constant int&     ldy [[buffer(3)]],
    constant int&     rnd [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  if (c >= D) { return; }
  const long i = (long)gid.y * D + c;
  const float s = X[i] + (float)Y[(long)gid.y * ldy + c];
  X[i] = rnd != 0 ? (float)(elt)s : s;
}

// In place: X = (float)(elt)X (the embedding into params_dtype).
//   0:X f32 [rows, D] 1:D   grid (D, rows)
kernel void magi2_round_f32(
    device float* X [[buffer(0)]],
    constant int& D [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= D) { return; }
  const long i = (long)gid.y * D + gid.x;
  X[i] = (float)(elt)X[i];
}
