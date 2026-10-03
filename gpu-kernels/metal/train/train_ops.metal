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
