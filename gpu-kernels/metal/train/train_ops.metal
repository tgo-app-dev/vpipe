// train_ops.metal -- the elementwise half of LoRA training: the backward
// of every non-GEMM op a DiT block runs, the flow-matching loss, the
// optimizer updates and the reductions they need.
//
// FROZEN BASE. Nothing here produces a gradient for a base weight: a
// LoRA trains only its own factors, so the backward needs the gradient
// with respect to each op's INPUT and nothing else. That is what keeps
// these kernels simple -- no column reductions over the rows for a
// norm's weight or a gate, which a full fine-tune would need.
//
// Precision. Activations and their gradients are the storage type
// (VPIPE_ELT: bf16 for every trainable family today). Every reduction
// accumulates in f32, and the RESIDUAL-STREAM gradient is f32 end to
// end, because it is summed through every block of the stack.
//
// DETERMINISTIC. No atomics anywhere: a reduction writes one partial
// per threadgroup and the host folds them in index order, so the same
// inputs give the same bits run after run -- which is what lets a
// resumed training run match one that was never stopped.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define TRAIN_TG 256

// ---- tiled 2-D transpose -------------------------------------------------
//
// in [R, C] (row stride ld_in) -> out[c * ld_out + col_off + r]. The
// backward multiplies by a weight's NON-transposed form, and every GEMM
// route in the tree (steel, matrix cores, int8, the ANE) is NT; handing
// them W^T instead reuses all of them. `col_off` lands the transpose in
// a column band of a wider matrix, which is how q/k/v (and gate/up)
// become ONE stacked backward GEMM.
//
// Grid: threadgroups (ceil(C/32), ceil(R/32)), 32x8 threads, each thread
// four rows of the tile.
kernel void train_transpose_2d(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int& R       [[buffer(2)]],
    constant int& C       [[buffer(3)]],
    constant int& ld_in   [[buffer(4)]],
    constant int& ld_out  [[buffer(5)]],
    constant int& col_off [[buffer(6)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
  threadgroup VPIPE_ELT tile[32][33];
  const int c0 = (int)tg.x * 32;
  const int r0 = (int)tg.y * 32;
  for (int k = 0; k < 4; ++k) {
    const int r = r0 + (int)lid.y + k * 8;
    const int c = c0 + (int)lid.x;
    if (r < R && c < C) {
      tile[lid.y + k * 8][lid.x] = in[(long)r * ld_in + c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k = 0; k < 4; ++k) {
    const int c = c0 + (int)lid.y + k * 8;     // output row
    const int r = r0 + (int)lid.x;             // output column
    if (r < R && c < C) {
      out[(long)c * ld_out + col_off + r] = tile[lid.x][lid.y + k * 8];
    }
  }
}

// ---- LayerNorm (no affine) * (1 + scale), backward -------------------------
//
// Forward (layernorm_modulate_f16 with a zero shift):
//   out = (1 + s) * (x - mean) * rstd,   rstd = rsqrt(E[x^2] - mean^2 + eps)
// Backward, per row, with g = dout * (1 + s) and xh the normalized x:
//   dx = rstd * (g - mean(g) - xh * mean(g * xh))
// ADDED into dx (f32): the residual gradient already sits there, and
// this is the branch's contribution to it. The statistics are
// recomputed with the forward's own one-pass formula, so xh here is the
// forward's xh.
//
// One threadgroup of TRAIN_TG threads per row. `dout` is the storage
// type or f32 (dout_f32 != 0), since a GEMM may hand either.
kernel void train_ln_mod_bwd(
    const device VPIPE_ELT* x      [[buffer(0)]],
    const device VPIPE_ELT* dout   [[buffer(1)]],
    const device float*     dout32 [[buffer(2)]],
    const device VPIPE_ELT* scale  [[buffer(3)]],
    device float*           dx     [[buffer(4)]],
    constant int&   H        [[buffer(5)]],
    constant float& eps      [[buffer(6)]],
    constant int&   dout_f32 [[buffer(7)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* xr = x + (ulong)row * H;
  threadgroup float p1[TRAIN_TG / 32], p2[TRAIN_TG / 32];

  float s1 = 0.0f, s2 = 0.0f;
  for (int i = (int)lid; i < H; i += TRAIN_TG) {
    const float v = (float)xr[i];
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  if (simd_lid == 0) { p1[simd_gid] = s1; p2[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? p1[simd_lid] : 0.0f;
    float b = (simd_lid < TRAIN_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); b = simd_sum(b);
    if (simd_lid == 0) { p1[0] = a; p2[0] = b; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = p1[0] / (float)H;
  const float var = p2[0] / (float)H - mean * mean;
  const float rstd = rsqrt(var + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float a1 = 0.0f, a2 = 0.0f;
  for (int i = (int)lid; i < H; i += TRAIN_TG) {
    const float d = dout_f32 != 0 ? dout32[(ulong)row * H + i]
                                  : (float)dout[(ulong)row * H + i];
    const float g = d * (1.0f + (float)scale[i]);
    const float xh = ((float)xr[i] - mean) * rstd;
    a1 += g; a2 += g * xh;
  }
  a1 = simd_sum(a1); a2 = simd_sum(a2);
  if (simd_lid == 0) { p1[simd_gid] = a1; p2[simd_gid] = a2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? p1[simd_lid] : 0.0f;
    float b = (simd_lid < TRAIN_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); b = simd_sum(b);
    if (simd_lid == 0) { p1[0] = a; p2[0] = b; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mg = p1[0] / (float)H;
  const float mgx = p2[0] / (float)H;
  device float* dxr = dx + (ulong)row * H;
  for (int i = (int)lid; i < H; i += TRAIN_TG) {
    const float d = dout_f32 != 0 ? dout32[(ulong)row * H + i]
                                  : (float)dout[(ulong)row * H + i];
    const float g = d * (1.0f + (float)scale[i]);
    const float xh = ((float)xr[i] - mean) * rstd;
    dxr[i] += rstd * (g - mg - xh * mgx);
  }
}

// ---- gated residual h += tanh(gate) * sub, backward ------------------------
//
// d(sub) = tanh(gate[n]) * dh, into the storage type for the GEMM that
// consumes it. The residual gradient itself passes through untouched.
// Grid (N, rows).
kernel void train_gated_tanh_bwd(
    const device float* dh   [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    device VPIPE_ELT* dsub   [[buffer(2)]],
    constant int& N          [[buffer(3)]],
    constant int& M          [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)N || gid.y >= (uint)M) { return; }
  const ulong o = (ulong)gid.y * N + gid.x;
  dsub[o] = VPIPE_ELT(precise::tanh(float(gate[gid.x])) * dh[o]);
}

// ---- SwiGLU s = silu(g) * u, backward --------------------------------------
//
//   du = ds * silu(g),   dg = ds * u * sig(g) * (1 + g * (1 - sig(g)))
// silu written as the forward writes it, g / (1 + exp(-g)).
//
// g, u and ds are [rows, width] contiguous; dg and du are written with a
// row stride of `ld`, which lets them land as the two column halves of
// one [rows, 2 * width] matrix -- the stacked input of the gate/up
// backward GEMM -- by binding du `width` elements after dg. Grid n =
// rows * width threads.
kernel void train_swiglu_bwd(
    const device VPIPE_ELT* g  [[buffer(0)]],
    const device VPIPE_ELT* u  [[buffer(1)]],
    const device VPIPE_ELT* ds [[buffer(2)]],
    device VPIPE_ELT*       dg [[buffer(3)]],
    device VPIPE_ELT*       du [[buffer(4)]],
    constant int& n     [[buffer(5)]],
    constant int& width [[buffer(6)]],
    constant int& ld    [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float gv = float(g[gid]);
  const float uv = float(u[gid]);
  const float d = float(ds[gid]);
  const float sg = 1.0f / (1.0f + exp(-gv));
  const float silu = gv / (1.0f + exp(-gv));
  const ulong o = (ulong)(gid / (uint)width) * ld + gid % (uint)width;
  du[o] = VPIPE_ELT(d * silu);
  dg[o] = VPIPE_ELT(d * uv * sg * (1.0f + gv * (1.0f - sg)));
}

// ---- RMSNorm with a FROZEN weight, backward -------------------------------
//
// Forward (rms_norm_fast_f16): y = x * inv * w, inv = rsqrt(E[x^2] + eps).
// Backward, per row, with gw = dy * w and xh = x * inv:
//   dx = inv * (gw - xh * mean(gw * xh))
// One SIMDGROUP per row: the rows here are q/k heads (D = 128), four
// elements a lane. Grid (32, rows) threads in groups of (32, 8).
//
// Row r is head (r % heads) of token (r / heads), and its dx lands at
// token * ld + (r % heads) * D -- so the q, k (and v) gradients can be
// written straight into their column bands of one stacked [T, 3 * H]
// matrix. heads = 1 and ld = D is the plain contiguous layout.
kernel void train_rms_bwd(
    const device VPIPE_ELT* x  [[buffer(0)]],
    const device VPIPE_ELT* dy [[buffer(1)]],
    const device VPIPE_ELT* w  [[buffer(2)]],
    device VPIPE_ELT*       dx [[buffer(3)]],
    constant int&   D     [[buffer(4)]],
    constant float& eps   [[buffer(5)]],
    constant int&   rows  [[buffer(6)]],
    constant int&   heads [[buffer(7)]],
    constant int&   ld    [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const uint row = gid.y;
  if (row >= (uint)rows) { return; }
  const device VPIPE_ELT* xr = x + (ulong)row * D;
  const device VPIPE_ELT* dr = dy + (ulong)row * D;
  float ss = 0.0f;
  for (int i = (int)simd_lid; i < D; i += 32) {
    const float v = float(xr[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(D) + eps);
  float dot = 0.0f;
  for (int i = (int)simd_lid; i < D; i += 32) {
    dot += float(dr[i]) * float(w[i]) * float(xr[i]) * inv;
  }
  dot = simd_sum(dot) / float(D);
  device VPIPE_ELT* o =
      dx + (ulong)(row / (uint)heads) * ld + (ulong)(row % (uint)heads) * D;
  for (int i = (int)simd_lid; i < D; i += 32) {
    const float xh = float(xr[i]) * inv;
    o[i] = VPIPE_ELT(inv * (float(dr[i]) * float(w[i]) - xh * dot));
  }
}

// ---- interleaved-pair RoPE + [T,H,D] -> [H,T,D], backward -----------------
//
// The forward (transpose_rope_pair_ftab_f16) rotates token t's pair i by
// theta and writes it head-major. The backward rotates by -theta and
// writes token-major, which is the layout the projection's backward GEMM
// reads. `in` is head-major [H][T][D]; the storage type, or f32 when
// in_f32 != 0 (dK, which accumulates in f32 across segments).
// Grid (D/2, T, H).
kernel void train_rope_bwd(
    const device VPIPE_ELT* in   [[buffer(0)]],
    const device float*     in32 [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const device float*     cosb [[buffer(3)]],
    const device float*     sinb [[buffer(4)]],
    constant int& H      [[buffer(5)]],
    constant int& T      [[buffer(6)]],
    constant int& D      [[buffer(7)]],
    constant int& in_f32 [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const ulong cb = (ulong)t * D + 2 * i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const ulong ib = ((ulong)h * T + t) * D + 2 * i;   // head-major input
  const ulong ob = ((ulong)t * H + h) * D + 2 * i;   // token-major output
  const float y1 = in_f32 != 0 ? in32[ib] : float(in[ib]);
  const float y2 = in_f32 != 0 ? in32[ib + 1] : float(in[ib + 1]);
  out[ob]     = VPIPE_ELT(y1 * c + y2 * s);
  out[ob + 1] = VPIPE_ELT(-y1 * s + y2 * c);
}

// f32 head-major [H][T][D] -> storage-type token-major [T][H*D]: dV, which
// is accumulated in f32 across the attention's segments and read by the
// v projection's backward GEMM token-major. Grid (D, T, H).
// `ld` is the output's row stride (H * D for a plain [T, H * D]), so dV
// can land as a column band of the stacked q/k/v gradient.
kernel void train_hd_f32_to_td(
    const device float* in  [[buffer(0)]],
    device VPIPE_ELT*   out [[buffer(1)]],
    constant int& H  [[buffer(2)]],
    constant int& T  [[buffer(3)]],
    constant int& D  [[buffer(4)]],
    constant int& ld [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  (void)H;
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  out[(ulong)t * ld + (ulong)h * D + d] =
      VPIPE_ELT(in[((ulong)h * T + t) * D + d]);
}

// ---- attention backward's two per-row inputs -------------------------------
//
// rowdot: Dsum[h][t] = sum_d dO[h][t][d] * O[h][t][d], f32, for every row
// of the joint sequence. One simdgroup per (h, t). Grid (32, T, H).
kernel void train_attn_rowdot(
    const device VPIPE_ELT* dO   [[buffer(0)]],
    const device VPIPE_ELT* O    [[buffer(1)]],
    device float*           Dsum [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& D [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const ulong r = (ulong)gid.z * T + gid.y;
  const device VPIPE_ELT* a = dO + r * D;
  const device VPIPE_ELT* b = O + r * D;
  float acc = 0.0f;
  for (int i = (int)simd_lid; i < D; i += 32) {
    acc += float(a[i]) * float(b[i]);
  }
  acc = simd_sum(acc);
  if (simd_lid == 0) { Dsum[r] = acc; }
}

// lse: the forward's exported statistics for ONE segment -- [H][qL] each,
// the steel kernel's layout when its Q is bound at the segment's first
// row -- into the log-2 LSE of every row, LSE2[h][row0 + i] =
// ml_max + log2(ml_sum). ml_max is already in the scaled log-2 domain
// (the kernel folds scale * log2(e) into S), so P = exp2(S2 - LSE2).
// Grid (qL, H).
kernel void train_attn_lse(
    const device float* ml_max [[buffer(0)]],
    const device float* ml_sum [[buffer(1)]],
    device float*       lse2   [[buffer(2)]],
    constant int& qL   [[buffer(3)]],
    constant int& T    [[buffer(4)]],
    constant int& row0 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)qL) { return; }
  const ulong s = (ulong)gid.y * qL + gid.x;
  lse2[(ulong)gid.y * T + row0 + gid.x] = ml_max[s] + log2(ml_sum[s]);
}

// ---- flow-matching loss ---------------------------------------------------
//
// The model predicts the velocity v = eps - x0 of x_t = (1 - s) x0 + s eps.
//   loss  = w * mean((pred - v)^2)
//   dpred = 2 * w / n * (pred - v)
// Each threadgroup writes the sum of its squared errors (unweighted) to
// partial[tg]; the host folds them in order. Grid n threads, TRAIN_TG a
// group.
kernel void train_flow_loss(
    const device VPIPE_ELT* pred    [[buffer(0)]],
    const device VPIPE_ELT* x0      [[buffer(1)]],
    const device VPIPE_ELT* noise   [[buffer(2)]],
    device VPIPE_ELT*       dpred   [[buffer(3)]],
    device float*           partial [[buffer(4)]],
    constant int&   n    [[buffer(5)]],
    constant float& coef [[buffer(6)]],
    uint gid  [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
  float e = 0.0f;
  if (gid < (uint)n) {
    const float v = float(noise[gid]) - float(x0[gid]);
    e = float(pred[gid]) - v;
    dpred[gid] = VPIPE_ELT(coef * e);
  }
  float sq = simd_sum(e * e);
  threadgroup float p[TRAIN_TG / 32];
  if (simd_lid == 0) { p[simd_gid] = sq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? p[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { partial[tgid] = a; }
  }
}

// ---- reductions and fills --------------------------------------------------

kernel void train_fill_f32(
    device float* out    [[buffer(0)]],
    constant float& v    [[buffer(1)]],
    constant int& n      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid < (uint)n) { out[gid] = v; }
}

// partial[tg] = sum of x^2 over the group's elements (TRAIN_TG threads,
// each striding over `per` elements), so n elements take
// ceil(n / (TRAIN_TG * per)) groups.
kernel void train_sqnorm_partial(
    const device float* x       [[buffer(0)]],
    device float*       partial [[buffer(1)]],
    constant int& n   [[buffer(2)]],
    constant int& per [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
  const long base = (long)tgid * TRAIN_TG * per;
  float acc = 0.0f;
  for (int k = 0; k < per; ++k) {
    const long i = base + (long)k * TRAIN_TG + lid;
    if (i < n) { const float v = x[i]; acc += v * v; }
  }
  acc = simd_sum(acc);
  threadgroup float p[TRAIN_TG / 32];
  if (simd_lid == 0) { p[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? p[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { partial[tgid] = a; }
  }
}

// partial[tg] = sum of a * b over the group's elements.
kernel void train_dot_partial(
    const device float* a       [[buffer(0)]],
    const device float* b       [[buffer(1)]],
    device float*       partial [[buffer(2)]],
    constant int& n   [[buffer(3)]],
    constant int& per [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
  const long base = (long)tgid * TRAIN_TG * per;
  float acc = 0.0f;
  for (int k = 0; k < per; ++k) {
    const long i = base + (long)k * TRAIN_TG + lid;
    if (i < n) { acc += a[i] * b[i]; }
  }
  acc = simd_sum(acc);
  threadgroup float p[TRAIN_TG / 32];
  if (simd_lid == 0) { p[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float s = (simd_lid < TRAIN_TG / 32) ? p[simd_lid] : 0.0f;
    s = simd_sum(s);
    if (simd_lid == 0) { partial[tgid] = s; }
  }
}

// ---- optimizer updates -----------------------------------------------------
//
// Every update runs over one flat f32 arena (all A factors, or all B)
// and ends the same way: the master is rounded to the storage type and
// multiplied by `mul` -- alpha / rank for an A factor, 1 for B -- which
// is EXACTLY what the runtime-LoRA loader does to a file's factors when
// it binds them. So the bf16 factors the forward reads during training
// are bit-for-bit the ones a saved and reloaded adapter produces.
//
// And every update zeroes the gradient it consumed, so the next step's
// backward accumulates into a clean arena without a separate pass.

struct TrainOptParams {
  float lr;
  float beta1;
  float beta2;
  float beta3;        // Prodigy's EMA for s and the numerator
  float eps;
  float weight_decay;
  float bc1;          // 1 - beta1^t (1 where bias correction is off)
  float bc2;          // 1 - beta2^t
  float grad_scale;   // the clip factor (and 1 / grad_accum)
  float mul;          // shadow multiplier: alpha / rank for A, 1 for B
  float d;            // Prodigy: the step-size estimate
  float d0;           // Prodigy: its initial value
  float dlr;          // Prodigy: d * lr (times the bias correction)
  int   n;
};

inline void train_write_shadow_(device VPIPE_ELT* shadow, uint i, float p,
                                float mul)
{
  shadow[i] = VPIPE_ELT(float(VPIPE_ELT(p)) * mul);
}

// AdamW, decoupled weight decay.
kernel void train_adamw(
    device float*     p      [[buffer(0)]],
    device float*     g      [[buffer(1)]],
    device float*     m      [[buffer(2)]],
    device float*     v      [[buffer(3)]],
    device VPIPE_ELT* shadow [[buffer(4)]],
    constant TrainOptParams& o [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
  if (i >= (uint)o.n) { return; }
  const float gi = g[i] * o.grad_scale;
  const float mi = o.beta1 * m[i] + (1.0f - o.beta1) * gi;
  const float vi = o.beta2 * v[i] + (1.0f - o.beta2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  float pi = p[i];
  pi -= o.lr * o.weight_decay * pi;
  pi -= o.lr * (mi / o.bc1) / (sqrt(vi / o.bc2) + o.eps);
  p[i] = pi;
  g[i] = 0.0f;
  train_write_shadow_(shadow, i, pi, o.mul);
}

// 8-BIT AdamW: the two moments held as one-byte codes, a block of 256 a
// threadgroup, each block scaled by its own absmax (Dettmers et al.,
// "8-bit Optimizers via Block-wise Quantization": the dynamic map, signed
// for m and unsigned for v, `codes` [512] sorted ascending, built on the
// host). A step DEQUANTIZES the block's moments, runs AdamW's update in
// f32 exactly as train_adamw, then requantizes the new moments to the
// nearest code at the block's new absmax. `amax` [2 * blocks]: m's, v's.
inline uchar train_nearest_code_(threadgroup const float* c, float x)
{
  // The last code <= x, then the nearer of it and the next.
  int lo = 0;
  for (int step = 128; step > 0; step >>= 1) {
    if (lo + step < 256 && c[lo + step] <= x) { lo += step; }
  }
  if (lo < 255 && fabs(c[lo + 1] - x) < fabs(x - c[lo])) { ++lo; }
  return (uchar)lo;
}

kernel void train_adamw8(
    device float*       p      [[buffer(0)]],
    device float*       g      [[buffer(1)]],
    device uchar*       qm     [[buffer(2)]],
    device uchar*       qv     [[buffer(3)]],
    device float*       amax   [[buffer(4)]],
    device VPIPE_ELT*   shadow [[buffer(5)]],
    constant TrainOptParams& o [[buffer(6)]],
    constant float*     codes  [[buffer(7)]],
    uint i  [[thread_position_in_grid]],
    uint t  [[thread_index_in_threadgroup]],
    uint b  [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
  threadgroup float cs[256];
  threadgroup float cu[256];
  threadgroup float rm[8];
  threadgroup float rv[8];
  threadgroup float am_new[2];
  cs[t] = codes[t];
  cu[t] = codes[256 + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool live = i < (uint)o.n;
  float mi = 0.0f, vi = 0.0f;
  if (live) {
    const float m0 = cs[qm[i]] * amax[2 * b];
    const float v0 = cu[qv[i]] * amax[2 * b + 1];
    const float gi = g[i] * o.grad_scale;
    mi = o.beta1 * m0 + (1.0f - o.beta1) * gi;
    vi = o.beta2 * v0 + (1.0f - o.beta2) * gi * gi;
    float pi = p[i];
    pi -= o.lr * o.weight_decay * pi;
    pi -= o.lr * (mi / o.bc1) / (sqrt(vi / o.bc2) + o.eps);
    p[i] = pi;
    g[i] = 0.0f;
    train_write_shadow_(shadow, i, pi, o.mul);
  }
  const float a = simd_max(fabs(mi));
  const float c = simd_max(vi);
  if (ln == 0) { rm[sg] = a; rv[sg] = c; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float x = 0.0f, y = 0.0f;
    for (int k = 0; k < 8; ++k) { x = max(x, rm[k]); y = max(y, rv[k]); }
    am_new[0] = x;
    am_new[1] = y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float am1 = am_new[0], av1 = am_new[1];
  if (live) {
    qm[i] = train_nearest_code_(cs, am1 > 0.0f ? mi / am1 : 0.0f);
    qv[i] = train_nearest_code_(cu, av1 > 0.0f ? vi / av1 : 0.0f);
  }
  if (t == 0) {
    amax[2 * b] = am1;
    amax[2 * b + 1] = av1;
  }
}

// ADAFACTOR (Shazeer & Stern; transformers' Adafactor with
// relative_step, scale_parameter and the first moment off): every factor
// is a matrix [rows, cols] of the arena, its squared gradient's second
// moment kept as a row vector and a column vector. A matrix table entry:
// its element offset, rows, cols, its row and column states' offsets in
// the state arena, and the rows and columns of the matrices before it
// (rcum, ccum), which is how a thread finds its matrix by binary search.
struct TrainAfMat {
  uint off;
  uint rows;
  uint cols;
  uint row_off;
  uint col_off;
  uint rcum;
  uint ccum;
};

struct TrainAfParams {
  float grad_scale;
  float beta2t;       // 1 - t^decay_rate
  float eps1;         // added to every squared gradient (1e-30)
  float lr;
  float weight_decay;
  float clip;         // the update's RMS threshold (1)
  float mul;          // the shadow's multiplier
  int   nmat;
  int   total;        // rows (or columns) of every matrix together
};

inline uint train_af_find_(device const TrainAfMat* m, int n, uint s,
                           bool by_rows)
{
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    const uint c = by_rows ? m[mid].rcum : m[mid].ccum;
    if (c <= s) { lo = mid; } else { hi = mid - 1; }
  }
  return (uint)lo;
}

// One simdgroup a row (by_rows) or a column: the mean of (g s)^2 + eps1
// over it, into the state by the EMA.
kernel void train_adafactor_stats(
    device const float*      g     [[buffer(0)]],
    device float*            st    [[buffer(1)]],
    device const TrainAfMat* mats  [[buffer(2)]],
    constant TrainAfParams&  o     [[buffer(3)]],
    constant int&            rows  [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
  const uint s = tg * 8 + sg;
  if (s >= (uint)o.total) { return; }
  const bool by_rows = rows != 0;
  const uint j = train_af_find_(mats, o.nmat, s, by_rows);
  const TrainAfMat m = mats[j];
  const uint i = s - (by_rows ? m.rcum : m.ccum);
  const uint count = by_rows ? m.cols : m.rows;
  const uint base = m.off + (by_rows ? i * m.cols : i);
  const uint step = by_rows ? 1u : m.cols;
  float acc = 0.0f;
  for (uint k = ln; k < count; k += 32) {
    const float v = g[base + k * step] * o.grad_scale;
    acc += v * v + o.eps1;
  }
  acc = simd_sum(acc);
  if (ln == 0) {
    const float mean = acc / (float)count;
    device float& x = st[(by_rows ? m.row_off : m.col_off) + i];
    x = o.beta2t * x + (1.0f - o.beta2t) * mean;
  }
}

// The update U = g s * rsqrt(row / mean(rows)) * rsqrt(col), clipped to
// RMS `clip` per matrix, applied: one threadgroup (256) a matrix. Pass 0
// sums U^2 into usq[j]; pass 1 applies -lr wd p and -lr U / max(1,
// RMS(U) / clip), zeroes the gradient and writes the shadow.
kernel void train_adafactor_update(
    device float*            p     [[buffer(0)]],
    device float*            g     [[buffer(1)]],
    device const float*      st    [[buffer(2)]],
    device const TrainAfMat* mats  [[buffer(3)]],
    device float*            usq   [[buffer(4)]],
    device VPIPE_ELT*        shadow [[buffer(5)]],
    constant TrainAfParams&  o     [[buffer(6)]],
    constant int&            pass  [[buffer(7)]],
    uint j  [[threadgroup_position_in_grid]],
    uint t  [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
  threadgroup float part[8];
  const TrainAfMat m = mats[j];
  const uint n = m.rows * m.cols;
  // The row states' mean, in a fixed order (every thread folds the
  // partials alike).
  float rm = 0.0f;
  {
    float a = 0.0f;
    for (uint r = t; r < m.rows; r += 256) { a += st[m.row_off + r]; }
    a = simd_sum(a);
    if (ln == 0) { part[sg] = a; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < 8; ++k) { rm += part[k]; }
    rm /= (float)m.rows;
  }
  if (pass == 0) {
    float acc = 0.0f;
    for (uint e = t; e < n; e += 256) {
      const uint r = e / m.cols, c = e - r * m.cols;
      const float u = g[m.off + e] * o.grad_scale *
                      rsqrt(st[m.row_off + r] / rm) * rsqrt(st[m.col_off + c]);
      acc += u * u;
    }
    acc = simd_sum(acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ln == 0) { part[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float x = 0.0f;
      for (int k = 0; k < 8; ++k) { x += part[k]; }
      usq[j] = x;
    }
    return;
  }
  const float rms = sqrt(usq[j] / (float)n);
  const float div = max(rms / o.clip, 1.0f);
  for (uint e = t; e < n; e += 256) {
    const uint r = e / m.cols, c = e - r * m.cols;
    const uint at = m.off + e;
    const float u = g[at] * o.grad_scale *
                    rsqrt(st[m.row_off + r] / rm) * rsqrt(st[m.col_off + c]);
    float pi = p[at];
    pi -= o.lr * o.weight_decay * pi;
    pi -= o.lr * (u / div);
    p[at] = pi;
    g[at] = 0.0f;
    train_write_shadow_(shadow, at, pi, o.mul);
  }
}

// Lion: sign of an interpolated momentum, one state.
kernel void train_lion(
    device float*     p      [[buffer(0)]],
    device float*     g      [[buffer(1)]],
    device float*     m      [[buffer(2)]],
    device VPIPE_ELT* shadow [[buffer(3)]],
    constant TrainOptParams& o [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
  if (i >= (uint)o.n) { return; }
  const float gi = g[i] * o.grad_scale;
  const float c = o.beta1 * m[i] + (1.0f - o.beta1) * gi;
  float pi = p[i];
  pi -= o.lr * o.weight_decay * pi;
  pi -= o.lr * (c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f));
  m[i] = o.beta2 * m[i] + (1.0f - o.beta2) * gi;
  p[i] = pi;
  g[i] = 0.0f;
  train_write_shadow_(shadow, i, pi, o.mul);
}

// SGD with momentum (beta1; 0 is plain SGD) and decoupled weight decay.
kernel void train_sgd(
    device float*     p      [[buffer(0)]],
    device float*     g      [[buffer(1)]],
    device float*     m      [[buffer(2)]],
    device VPIPE_ELT* shadow [[buffer(3)]],
    constant TrainOptParams& o [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
  if (i >= (uint)o.n) { return; }
  const float gi = g[i] * o.grad_scale;
  const float mi = o.beta1 * m[i] + gi;
  m[i] = mi;
  float pi = p[i];
  pi -= o.lr * o.weight_decay * pi;
  pi -= o.lr * mi;
  p[i] = pi;
  g[i] = 0.0f;
  train_write_shadow_(shadow, i, pi, o.mul);
}

// Prodigy (Mishchenko & Defazio), the Adam form, as prodigyopt runs it.
// Two passes a step, because the step size d is a GLOBAL quantity:
//
//  1. train_prodigy_accum, with the current d: the moments and s move,
//     and num[i] = (d / d0) * dlr * g * (x0 - p) is left for the host to
//     fold. The host also folds |s| (train_abs_partial), then sets
//       d_num = beta3 * d_num + sum(num),  d_hat = d_coef * d_num / |s|_1,
//       d = min(max(d, d_hat) grown at most by growth_rate).
//  2. train_prodigy_step: p -= dlr * (wd * p + m / (sqrt(v) + d * eps)).
//
// `x0` is the master at step 0.
kernel void train_prodigy_accum(
    const device float* p   [[buffer(0)]],
    const device float* g   [[buffer(1)]],
    const device float* x0  [[buffer(2)]],
    device float*       m   [[buffer(3)]],
    device float*       v   [[buffer(4)]],
    device float*       s   [[buffer(5)]],
    device float*       num [[buffer(6)]],
    constant TrainOptParams& o [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
  if (i >= (uint)o.n) { return; }
  const float gi = g[i] * o.grad_scale;
  const float dd = o.d / o.d0;
  num[i] = dd * o.dlr * gi * (x0[i] - p[i]);
  m[i] = o.beta1 * m[i] + o.d * (1.0f - o.beta1) * gi;
  v[i] = o.beta2 * v[i] + o.d * o.d * (1.0f - o.beta2) * gi * gi;
  s[i] = o.beta3 * s[i] + dd * o.dlr * gi;
}

kernel void train_prodigy_step(
    device float*       p      [[buffer(0)]],
    device float*       g      [[buffer(1)]],
    const device float* m      [[buffer(2)]],
    const device float* v      [[buffer(3)]],
    device VPIPE_ELT*   shadow [[buffer(4)]],
    constant TrainOptParams& o [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
  if (i >= (uint)o.n) { return; }
  float pi = p[i];
  pi -= o.dlr * o.weight_decay * pi;
  pi -= o.dlr * m[i] / (sqrt(v[i]) + o.d * o.eps);
  p[i] = pi;
  g[i] = 0.0f;
  train_write_shadow_(shadow, i, pi, o.mul);
}

// partial[tg] = sum of |x| over the group's elements (Prodigy's |s|_1).
kernel void train_abs_partial(
    const device float* x       [[buffer(0)]],
    device float*       partial [[buffer(1)]],
    constant int& n   [[buffer(2)]],
    constant int& per [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
  const long base = (long)tgid * TRAIN_TG * per;
  float acc = 0.0f;
  for (int k = 0; k < per; ++k) {
    const long i = base + (long)k * TRAIN_TG + lid;
    if (i < n) { acc += fabs(x[i]); }
  }
  acc = simd_sum(acc);
  threadgroup float pp[TRAIN_TG / 32];
  if (simd_lid == 0) { pp[simd_gid] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? pp[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { partial[tgid] = a; }
  }
}

// EMA of the master weights, for the averaged adapter a checkpoint also
// saves: e = decay * e + (1 - decay) * p.
kernel void train_ema(
    const device float* p [[buffer(0)]],
    device float*       e [[buffer(1)]],
    constant float& decay [[buffer(2)]],
    constant int& n       [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
  if (i < (uint)n) { e[i] = decay * e[i] + (1.0f - decay) * p[i]; }
}

// The shadow of an arbitrary master (initialization, resume, and the EMA
// adapter a checkpoint writes): shadow = ELT(ELT(p) * mul).
kernel void train_shadow(
    const device float* p      [[buffer(0)]],
    device VPIPE_ELT*   shadow [[buffer(1)]],
    constant float& mul        [[buffer(2)]],
    constant int& n            [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
  if (i < (uint)n) { train_write_shadow_(shadow, i, p[i], mul); }
}

// ==== MiniMax-H3's block, backward ========================================
//
// H3 modulates PER ROW: each row reads its six AdaLN vectors from a table
// row picked by idx[r] (timestep x 3 + modality), the norms are RMS with a
// frozen gamma, the gates are plain (no tanh), q / k rotate only their
// first `rot` dims (rotate-half), and the fused q|k|v groups its output
// either per head or flat. The kernels below are the backward of exactly
// those ops; the ones above stay Qwen-Image-2.1's.

// ---- RMSNorm (frozen w) then a per-row modulation, backward --------------
//
// Forward (rms_norm_fast_f16, then adaln_modulate_idx_f16):
//   n = x * inv * w,  inv = rsqrt(E[x^2] + eps)
//   y = (1 + mod[idx[r] * stride + scale_off + i]) * n + shift
// Backward, per row, with g = dy * (1 + sc) * w and xh = x * inv:
//   dx = inv * (g - xh * mean(g * xh))
// ADDED into dx (f32), the residual gradient's running sum. has_mod 0 is
// the plain norm (the token refiner's, its final norm). `dout` is the
// storage type or f32 (dout_f32 != 0). One threadgroup of TRAIN_TG
// threads per row; 64-bit row offsets.
kernel void train_rms_mod_bwd(
    const device VPIPE_ELT* x      [[buffer(0)]],
    const device VPIPE_ELT* dout   [[buffer(1)]],
    const device float*     dout32 [[buffer(2)]],
    const device VPIPE_ELT* w      [[buffer(3)]],
    const device VPIPE_ELT* mod    [[buffer(4)]],
    const device int*       idx    [[buffer(5)]],
    device float*           dx     [[buffer(6)]],
    constant int&   H         [[buffer(7)]],
    constant float& eps       [[buffer(8)]],
    constant int&   dout_f32  [[buffer(9)]],
    constant int&   stride    [[buffer(10)]],
    constant int&   scale_off [[buffer(11)]],
    constant int&   has_mod   [[buffer(12)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const ulong ro = (ulong)row * (ulong)H;
  const device VPIPE_ELT* xr = x + ro;
  const device VPIPE_ELT* sc =
      has_mod != 0 ? mod + (long)idx[row] * (long)stride + scale_off : mod;
  threadgroup float p1[TRAIN_TG / 32];

  float s2 = 0.0f;
  for (int i = (int)lid; i < H; i += TRAIN_TG) {
    const float v = (float)xr[i];
    s2 += v * v;
  }
  s2 = simd_sum(s2);
  if (simd_lid == 0) { p1[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? p1[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { p1[0] = a; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(p1[0] / (float)H + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float a2 = 0.0f;
  for (int i = (int)lid; i < H; i += TRAIN_TG) {
    const float d = dout_f32 != 0 ? dout32[ro + i] : (float)dout[ro + i];
    const float m = has_mod != 0 ? 1.0f + (float)sc[i] : 1.0f;
    const float g = d * m * (float)w[i];
    a2 += g * ((float)xr[i] * inv);
  }
  a2 = simd_sum(a2);
  if (simd_lid == 0) { p1[simd_gid] = a2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < TRAIN_TG / 32) ? p1[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { p1[0] = a; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mgx = p1[0] / (float)H;
  device float* dxr = dx + ro;
  for (int i = (int)lid; i < H; i += TRAIN_TG) {
    const float d = dout_f32 != 0 ? dout32[ro + i] : (float)dout[ro + i];
    const float m = has_mod != 0 ? 1.0f + (float)sc[i] : 1.0f;
    const float g = d * m * (float)w[i];
    const float xh = (float)xr[i] * inv;
    dxr[i] += inv * (g - xh * mgx);
  }
}

// ---- gated residual h += gate[idx[m]] * sub, backward ---------------------
//
// d(sub) = mod[idx[m] * stride + gate_off + n] * dh (f32 in, the storage
// type out), for the GEMM that consumes it; has_gate 0 is the plain
// residual (the refiner's): d(sub) = dh. The residual gradient passes
// through untouched. Grid (N, rows).
kernel void train_gated_idx_bwd(
    const device float*     dh   [[buffer(0)]],
    const device VPIPE_ELT* mod  [[buffer(1)]],
    const device int*       idx  [[buffer(2)]],
    device VPIPE_ELT*       dsub [[buffer(3)]],
    constant int& N        [[buffer(4)]],
    constant int& M        [[buffer(5)]],
    constant int& stride   [[buffer(6)]],
    constant int& gate_off [[buffer(7)]],
    constant int& has_gate [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)N || gid.y >= (uint)M) { return; }
  const ulong o = (ulong)gid.y * (ulong)N + gid.x;
  const float g = has_gate != 0
                      ? float(mod[(long)idx[gid.y] * (long)stride +
                                  gate_off + gid.x])
                      : 1.0f;
  dsub[o] = VPIPE_ELT(g * dh[o]);
}

// ---- SwiGLU off a fused [gate | up] projection, backward ------------------
//
// Forward (swiglu_split_gate_first_f16): s = up * silu(gate), the gate the
// FIRST half of each [rows, 2W] row. Backward:
//   d(up) = ds * silu(g),  d(gate) = ds * up * sig(g) * (1 + g (1 - sig(g)))
// into the same [gate | up] row layout. Grid n = rows * W threads.
kernel void train_swiglu_gf_bwd(
    const device VPIPE_ELT* gu  [[buffer(0)]],
    const device VPIPE_ELT* ds  [[buffer(1)]],
    device VPIPE_ELT*       dgu [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& W    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if ((ulong)gid >= (ulong)rows * (ulong)W) { return; }
  const uint r = gid / (uint)W;
  const uint d = gid % (uint)W;
  const ulong base = (ulong)r * (ulong)(2 * W);
  const float g = float(gu[base + d]);
  const float u = float(gu[base + W + d]);
  const float dv = float(ds[gid]);
  const float sg = 1.0f / (1.0f + exp(-g));
  const float silu = g / (1.0f + exp(-g));
  dgu[base + W + d] = VPIPE_ELT(dv * silu);
  dgu[base + d] = VPIPE_ELT(dv * u * sg * (1.0f + g * (1.0f - sg)));
}

// ---- partial rotate-half RoPE + [H,T,D] -> strided rows, backward ---------
//
// The forward (transpose_rope_half_part_ftab_f16) rotates the first `rot`
// dims of token t's head h -- pairs (i, i + rot/2), angle table [T, rot/2]
// -- passes the rest through, and writes head-major. The backward rotates
// by -theta and writes back where the head came FROM in the fused
// projection: token t's row at t * ld + off + h * hs. `in` is head-major
// [H][T][D], the storage type or f32 (in_f32 != 0: dK and dV, which
// accumulate in f32). rot 0 is a pure strided copy (dV). Grid (D, T, H).
kernel void train_rope_half_bwd(
    const device VPIPE_ELT* in   [[buffer(0)]],
    const device float*     in32 [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    const device float*     cosb [[buffer(3)]],
    const device float*     sinb [[buffer(4)]],
    constant int& H      [[buffer(5)]],
    constant int& T      [[buffer(6)]],
    constant int& D      [[buffer(7)]],
    constant int& rot    [[buffer(8)]],
    constant int& ld     [[buffer(9)]],
    constant int& off    [[buffer(10)]],
    constant int& hs     [[buffer(11)]],
    constant int& in_f32 [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (h >= H || t >= T) { return; }
  const ulong ib = ((ulong)h * (ulong)T + (ulong)t) * (ulong)D;
  const ulong ob = (ulong)t * (ulong)ld + (ulong)off + (ulong)h * (ulong)hs;
  if (d >= rot) {
    const float v = in_f32 != 0 ? in32[ib + d] : float(in[ib + d]);
    out[ob + d] = VPIPE_ELT(v);
    return;
  }
  const int half_r = rot / 2;
  const int i = d < half_r ? d : d - half_r;
  const ulong cb = (ulong)t * (ulong)half_r + (ulong)i;
  const float c = cosb[cb];
  const float s = sinb[cb];
  const float y1 = in_f32 != 0 ? in32[ib + i] : float(in[ib + i]);
  const float y2 = in_f32 != 0 ? in32[ib + i + half_r]
                               : float(in[ib + i + half_r]);
  out[ob + d] =
      VPIPE_ELT(d < half_r ? (y1 * c + y2 * s) : (y2 * c - y1 * s));
}

// ---- per-head RMSNorm (frozen w) on a strided projection, backward --------
//
// Forward (rms_norm_heads_strided_f16, in place on the fused q|k|v): head
// h of token t, D values at t * ld + off + h * hs, y = x * inv * w.
// Backward IN PLACE on `dq` -- the gradient sits where the head lies and
// is replaced by the input's -- with the forward's input `x` read at the
// same place of its own buffer. Each lane reads and writes only its own
// elements, so in place is safe. One simdgroup per (t, h); grid (32,
// T * H) in groups of (32, 8).
kernel void train_rms_heads_bwd(
    const device VPIPE_ELT* x  [[buffer(0)]],
    device VPIPE_ELT*       dq [[buffer(1)]],
    const device VPIPE_ELT* w  [[buffer(2)]],
    constant int&   T   [[buffer(3)]],
    constant int&   H   [[buffer(4)]],
    constant int&   D   [[buffer(5)]],
    constant int&   ld  [[buffer(6)]],
    constant int&   off [[buffer(7)]],
    constant int&   hs  [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  const uint row = gid.y;
  if (row >= (uint)(T * H)) { return; }
  const uint t = row / (uint)H;
  const uint h = row % (uint)H;
  const ulong b = (ulong)t * (ulong)ld + (ulong)off + (ulong)h * (ulong)hs;
  const device VPIPE_ELT* xr = x + b;
  device VPIPE_ELT* dr = dq + b;
  float ss = 0.0f;
  for (int i = (int)simd_lid; i < D; i += 32) {
    const float v = float(xr[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / float(D) + eps);
  float dot = 0.0f;
  for (int i = (int)simd_lid; i < D; i += 32) {
    dot += float(dr[i]) * float(w[i]) * float(xr[i]) * inv;
  }
  dot = simd_sum(dot) / float(D);
  for (int i = (int)simd_lid; i < D; i += 32) {
    const float xh = float(xr[i]) * inv;
    dr[i] = VPIPE_ELT(inv * (float(dr[i]) * float(w[i]) - xh * dot));
  }
}

// ---- a BANDED adapter B (lora::Factors::banded) as block-diagonal ---------
//
// B [n, r] of a fused projection whose output row j belongs to part
// (j / group) % parts and contracts only over that part's r columns of
// t = x A^T ([M, parts * r]). The backward's GEMMs want the plain
// [n, parts * r] block-diagonal form; this writes it, zeros off the band.
// Grid (parts * r, n).
kernel void train_band_expand(
    const device VPIPE_ELT* b   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int& n     [[buffer(2)]],
    constant int& r     [[buffer(3)]],
    constant int& parts [[buffer(4)]],
    constant int& group [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int j = (int)gid.y;
  if (c >= parts * r || j >= n) { return; }
  const int p = (j / group) % parts;
  const int pc = c / r;
  out[(ulong)j * (ulong)(parts * r) + c] =
      pc == p ? b[(ulong)j * (ulong)r + (c - pc * r)] : VPIPE_ELT(0.0f);
}

// ...and the gradient back: dst [n, r] (f32) += src [n, parts * r] on
// row j's own band. Grid (r, n).
kernel void train_band_extract_acc(
    const device float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant int&   n     [[buffer(2)]],
    constant int&   r     [[buffer(3)]],
    constant int&   parts [[buffer(4)]],
    constant int&   group [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int j = (int)gid.y;
  if (c >= r || j >= n) { return; }
  const int p = (j / group) % parts;
  dst[(ulong)j * (ulong)r + c] +=
      alpha * src[(ulong)j * (ulong)(parts * r) + p * r + c];
}
