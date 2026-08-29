// llm_elementwise.metal -- small pointwise/gather kernels for the
// Llama forward pass on metal-compute. f16 in/out, float32 math.
//
//   swiglu_f16:       out = silu(gate) * up,  silu(x)=x*sigmoid(x)
//   residual_add_f16: out = a + b
//   embed_gather_f16: out[t,:] = table[ids[t], :]   (row gather)

#include <metal_stdlib>
using namespace metal;

// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the
// bf16 variant metallib. Math stays f32.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// ---- opt-in int8 group-32 affine weight quant (MOSS codec) -------------
// Quantize an f16 weight [N,K] to uint8 with a per-group-of-32 (along K)
// affine scale+bias: q = round((v-min)/scale), v ~= q*scale + min. Halves
// the codec's resident weight footprint; the codec dequants per stage at
// decode. Output is explicit half (codec runs f16 in both metallib variants).
//   quant:   0:in(f16)[N,K] 1:out(u8)[N,K] 2:scale(f16)[N,K/32]
//            3:bias(f16)[N,K/32] 4:N 5:K.  grid (N*K/32), one group/thread.
kernel void quant_f16_to_u8g32(
    const device half* in    [[buffer(0)]],
    device uchar*      out   [[buffer(1)]],
    device half*       scale [[buffer(2)]],
    device half*       bias  [[buffer(3)]],
    constant int&      N     [[buffer(4)]],
    constant int&      K     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const int g = K / 32;
  if ((int)gid >= N * g) { return; }
  const int n = (int)gid / g, gg = (int)gid % g, base = n * K + gg * 32;
  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < 32; ++i) {
    const float v = (float)in[base + i];
    mn = min(mn, v); mx = max(mx, v);
  }
  const float s = (mx - mn) / 255.0f;
  const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
  scale[(int)gid] = (half)s;
  bias[(int)gid]  = (half)mn;
  for (int i = 0; i < 32; ++i) {
    int q = (int)round(((float)in[base + i] - mn) * inv);
    out[base + i] = (uchar)clamp(q, 0, 255);
  }
}

// dequant uint8 g32 -> f16 [N,K]: out[n,k] = q*scale[n,k/32] + bias[n,k/32].
//   0:in(u8)[N,K] 1:scale(f16) 2:bias(f16) 3:out(f16)[N,K] 4:N 5:K.  grid(N*K)
kernel void dequant_u8g32_to_f16(
    const device uchar* in    [[buffer(0)]],
    const device half*  scale [[buffer(1)]],
    const device half*  bias  [[buffer(2)]],
    device half*        out   [[buffer(3)]],
    constant int&       N     [[buffer(4)]],
    constant int&       K     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid >= N * K) { return; }
  const int n = (int)gid / K, k = (int)gid % K;
  const int gi = n * (K / 32) + k / 32;
  out[gid] = (half)((float)in[gid] * (float)scale[gi] + (float)bias[gi]);
}

// out[i] = silu(gate[i]) * up[i].   0:gate 1:up 2:out 3:n
// Generic SwiGLU: v = silu(gate[gid]) * up[gid], gate/up contiguous [total]. If
// out_stride <= 0 the output is contiguous (out[gid]); else it is a sub-view --
// row = gid/width, col = gid%width, out[row*out_stride + out_off + col] -- so a
// producer can write straight into a wider buffer (no separate copy).
inline void swiglu_generic_(const device VPIPE_ELT* gate,
                            const device VPIPE_ELT* up, device VPIPE_ELT* out,
                            int total, int width, int out_stride, int out_off,
                            uint gid)
{
  if (gid >= (uint)total) { return; }
  const float g = float(gate[gid]);
  const float silu = g / (1.0f + exp(-g));
  const VPIPE_ELT v = VPIPE_ELT(silu * float(up[gid]));
  if (out_stride <= 0) {
    out[gid] = v;
  } else {
    const int row = (int)gid / width;
    const int col = (int)gid % width;
    out[(long)row * out_stride + out_off + col] = v;
  }
}

// Non-strided wrapper (contiguous output) -- unchanged signature, so existing
// callers (Qwen3.5, Krea-2, ...) are byte-for-byte the same.
kernel void swiglu_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out   [[buffer(2)]],
    constant int&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  swiglu_generic_(gate, up, out, n, 0, 0, 0, gid);
}

// Strided SwiGLU: gate/up [rows, width] contiguous -> out[row*out_stride +
// out_off + col]. Lets the FLUX.2 single block's UNFUSED mlp write straight into
// scat[:, H:] (no concat). 0:gate 1:up 2:out 3:rows 4:width 5:out_stride 6:out_off
kernel void swiglu_rs_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      rows [[buffer(3)]],
    constant int&      width [[buffer(4)]],
    constant int&      out_stride [[buffer(5)]],
    constant int&      out_off    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  swiglu_generic_(gate, up, out, rows * width, width, out_stride, out_off, gid);
}

// SwiGLU over an INTERLEAVED gate|up matrix [rows, 2*ffn] (column 2g =
// gate feature g, 2g+1 = up feature g -- the layout produced by the dense
// matmul2d over the interleaved gate/up weight, mirroring the steel
// affine_qmm_swiglu fused epilogue). Writes out[rows, ffn] = silu(gate)*up.
//   0:in[rows,2*ffn] 1:out[rows,ffn] 2:rows 3:ffn.  grid (rows*ffn).
kernel void swiglu_interleaved_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& ffn  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(rows * ffn)) { return; }
  const int r = (int)gid / ffn;
  const int g = (int)gid % ffn;
  const int64_t base = (int64_t)r * (2 * ffn) + 2 * g;
  const float ga = float(in[base]);
  const float up = float(in[base + 1]);
  const float silu = ga / (1.0f + exp(-ga));
  out[gid] = VPIPE_ELT(silu * up);
}

// ====================================================================
// Qwen3.5-MoE router + combine + finalize (decode, M=1).
// ====================================================================

// MoE router: precise softmax over n_experts logits, then top_k by
// probability (argpartition-equivalent), renormalized when norm_topk != 0.
// One simdgroup (32 lanes) per token (tid.z = token; grid {32,1,M}, decode
// M=1); each lane owns ceil(n_experts/32) experts. Matches MLX qwen3_5_moe:
//   p = softmax(logits, precise); inds = top_k(p); w = p[inds]; w /= w.sum().
// (The softmax denominator cancels in the renorm, but is kept to mirror the
// reference's compute order.) Writes the token's selected ids + weights into
// the contiguous [token*top_k, ..] block.
kernel void moe_route_f16(
    const device VPIPE_ELT* logits [[buffer(0)]],   // [M, n_experts]
    device int*       out_ids       [[buffer(1)]],   // [M*top_k]
    device VPIPE_ELT* out_w         [[buffer(2)]],   // [M*top_k]
    constant int& n_experts         [[buffer(3)]],
    constant int& top_k             [[buffer(4)]],
    constant int& norm_topk         [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_simdgroup]])
{
  const int E = n_experts;
  logits  += (uint)tid.z * (uint)E;
  out_ids += (uint)tid.z * (uint)top_k;
  out_w   += (uint)tid.z * (uint)top_k;
  const int PER = (E + 31) / 32;            // experts per lane (8 for E=256)
  float lg[16];                             // PER <= 16 (E <= 512)
  float lmax = -INFINITY;
  for (int j = 0; j < PER; ++j) {
    const int idx = (int)lid + j * 32;
    const float v = (idx < E) ? (float)logits[idx] : -INFINITY;
    lg[j] = v;
    lmax = max(lmax, v);
  }
  const float gmax = simd_max(lmax);
  float ex[16];                             // exp(logit - max); selection key
  float lsum = 0.0f;
  for (int j = 0; j < PER; ++j) {
    const int idx = (int)lid + j * 32;
    const float e = (idx < E) ? metal::exp(lg[j] - gmax) : 0.0f;
    lsum += e;
    ex[j] = (idx < E) ? e : -1.0f;          // mask padding out of the argmax
  }
  const float S = simd_sum(lsum);           // softmax denominator over all E
  float wsum = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    // local best among this lane's still-available experts
    float bv = -1.0f; int bi = -1;
    for (int j = 0; j < PER; ++j) {
      if (ex[j] > bv) { bv = ex[j]; bi = (int)lid + j * 32; }
    }
    // simdgroup argmax reduction (lane 0 ends with the global best)
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, (uint)off);
      const int   oi = simd_shuffle_down(bi, (uint)off);
      if (ov > bv) { bv = ov; bi = oi; }
    }
    bv = simd_shuffle(bv, 0);
    bi = simd_shuffle(bi, 0);
    for (int j = 0; j < PER; ++j) {         // mask the taken expert
      if ((int)lid + j * 32 == bi) { ex[j] = -1.0f; }
    }
    const float p = bv / S;                 // softmax prob of the selected expert
    wsum += p;
    if (lid == 0) { out_ids[k] = bi; out_w[k] = (VPIPE_ELT)p; }
  }
  if (lid == 0 && norm_topk != 0 && wsum > 0.0f) {
    for (int k = 0; k < top_k; ++k) {
      out_w[k] = (VPIPE_ELT)((float)out_w[k] / wsum);
    }
  }
}

// MoE expert combine: out[t, h] = sum_s w[t*top_k+s] * partials[t*top_k+s, h].
// The pairs for token t are the contiguous block [t*top_k, (t+1)*top_k).
// grid(M*H); decode is the M=1 case.
kernel void moe_combine_f16(
    const device VPIPE_ELT* partials [[buffer(0)]],  // [M*top_k, H]
    const device VPIPE_ELT* w         [[buffer(1)]],  // [M*top_k]
    device VPIPE_ELT*       out        [[buffer(2)]],  // [M, H]
    constant int& H                    [[buffer(3)]],
    constant int& top_k                [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  const int h = (int)gid % H;
  const int t = (int)gid / H;
  float acc = 0.0f;
  const int base = t * top_k;
  for (int s = 0; s < top_k; ++s) {
    acc += (float)w[base + s] * (float)partials[(int64_t)(base + s) * H + h];
  }
  out[gid] = (VPIPE_ELT)acc;
}

// Gemma-4 MoE per-expert router weight scale: w[i] *= per_expert_scale[ids[i]]
// over the M*top_k routed pairs (applied AFTER moe_route's softmax-topk-renorm,
// BEFORE moe_combine). ids[i] is the selected expert of pair i. One thread per
// pair. n == M*top_k.
kernel void moe_expert_scale_f16(
    device VPIPE_ELT*       w                [[buffer(0)]],  // [n]
    const device VPIPE_ELT* per_expert_scale [[buffer(1)]],  // [E]
    const device int*       ids              [[buffer(2)]],  // [n]
    constant int&           n                [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid >= n) { return; }
  w[gid] = (VPIPE_ELT)((float)w[gid] *
                       (float)per_expert_scale[(uint)ids[gid]]);
}

// ---- MoE grouped-prefill counting sort (sorts (token,expert) pairs into
// per-expert blocks padded to MAXM, so the grouped GEMV reads each expert's
// weight once per tile instead of once per token) ----------------------

// Fill an int buffer with a constant (hist->0, srow/sdst/tile2e->-1). grid(n).
kernel void moe_ifill_i32(
    device int* buf   [[buffer(0)]],
    constant int& val [[buffer(1)]],
    constant int& n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid < (uint)n) { buf[gid] = val; }
}

// Histogram: hist[e] = # pairs routed to expert e. grid(np). hist pre-zeroed.
kernel void moe_hist_i32(
    const device int* eid [[buffer(0)]],   // [np] expert per pair
    device atomic_int* hist [[buffer(1)]], // [E]
    constant int& np [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)np) { return; }
  atomic_fetch_add_explicit(&hist[eid[gid]], 1, memory_order_relaxed);
}

// Build MAXM-aligned per-expert offsets + tile->expert map from the histogram.
// Single thread (E=256 is tiny). boff[e] = expert e's first slot (MAXM-aligned);
// tile2e[t] = expert owning tile t; cursor[e] = 0; ntiles_out[0] = tiles used.
kernel void moe_sort_setup_i32(
    const device int* hist [[buffer(0)]],  // [E]
    device int* boff       [[buffer(1)]],  // [E]
    device int* tile2e     [[buffer(2)]],  // [maxtiles] (pre-filled -1)
    device int* cursor     [[buffer(3)]],  // [E]
    device int* ntiles_out [[buffer(4)]],  // [1]
    constant int& E        [[buffer(5)]],
    constant int& MAXM     [[buffer(6)]],
    constant int& maxtiles [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid != 0) { return; }
  int off = 0;
  for (int e = 0; e < E; ++e) {
    boff[e] = off;
    cursor[e] = 0;
    const int ntile_e = (hist[e] + MAXM - 1) / MAXM;
    const int t0 = off / MAXM;
    for (int t = 0; t < ntile_e; ++t) {
      if (t0 + t < maxtiles) { tile2e[t0 + t] = e; }
    }
    off += ntile_e * MAXM;
  }
  ntiles_out[0] = off / MAXM;
}

// Scatter pairs into sorted per-expert slots. grid(np). srow[slot]=token row,
// sdst[slot]=orig pair index (pad slots keep -1). Order within a bucket is
// arbitrary but irrelevant -- combine sums per token in slot order via sdst.
kernel void moe_scatter_i32(
    const device int* eid    [[buffer(0)]],  // [np]
    const device int* boff   [[buffer(1)]],  // [E]
    device atomic_int* cursor [[buffer(2)]], // [E]
    device int* srow         [[buffer(3)]],  // [npad]
    device int* sdst         [[buffer(4)]],  // [npad]
    constant int& np         [[buffer(5)]],
    constant int& top_k      [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)np) { return; }
  const int p = (int)gid;
  const int e = eid[p];
  const int slot = boff[e]
      + atomic_fetch_add_explicit(&cursor[e], 1, memory_order_relaxed);
  srow[slot] = p / top_k;   // input (token) row
  sdst[slot] = p;           // pair-ordered destination
}

// Scatter grouped-down outputs back to pair order: partials[sdst[slot]] =
// dsorted[slot]. grid(npad*H); pad slots (sdst<0) skipped.
kernel void moe_scatter_back_f16(
    const device VPIPE_ELT* dsorted [[buffer(0)]],  // [npad, H]
    const device int* sdst          [[buffer(1)]],  // [npad]
    device VPIPE_ELT* partials      [[buffer(2)]],  // [np, H]
    constant int& H                 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  const int slot = (int)gid / H;
  const int h = (int)gid % H;
  const int p = sdst[slot];
  if (p < 0) { return; }
  partials[(int64_t)p * H + h] = dsorted[(int64_t)slot * H + h];
}

// MoE block finalize: x[t,h] += moe_out[t,h] + g[t] * shared_out[t,h], where
// g[t] = sigmoid(shared_expert_gate . x_t) precomputed in gbuf[t]. In-place on
// x (the post-attention residual stream). grid(M*H); decode is the M=1 case.
kernel void moe_finalize_f16(
    device VPIPE_ELT* x                [[buffer(0)]],  // [M, H] residual, in-place
    const device VPIPE_ELT* moe_out    [[buffer(1)]],  // [M, H]
    const device VPIPE_ELT* shared_out [[buffer(2)]],  // [M, H]
    const device VPIPE_ELT* gbuf       [[buffer(3)]],  // [M] sigmoid gate
    constant int& H                    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  const int t = (int)gid / H;
  const float g = (float)gbuf[t];
  x[gid] = (VPIPE_ELT)((float)x[gid] + (float)moe_out[gid]
                       + g * (float)shared_out[gid]);
}

// Fused combine + finalize: does moe_combine's weighted partial-sum INLINE
// (saves the separate moe_combine dispatch + its hazard barrier + the moe_out
// buffer, x40 decode layers). Bit-exact with moe_combine->moe_finalize: the
// partial-sum is rounded to VPIPE_ELT before the add, matching the f16 moe_out
// the two-kernel path materializes.  grid (M*H).
kernel void moe_finalize_combined_f16(
    device VPIPE_ELT* x                [[buffer(0)]],  // [M, H] residual in-place
    const device VPIPE_ELT* partials   [[buffer(1)]],  // [M*top_k, H]
    const device VPIPE_ELT* w          [[buffer(2)]],  // [M*top_k] routing wts
    const device VPIPE_ELT* shared_out [[buffer(3)]],  // [M, H]
    const device VPIPE_ELT* gbuf       [[buffer(4)]],  // [M] sigmoid gate
    constant int& H                    [[buffer(5)]],
    constant int& top_k                [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  const int h = (int)gid % H;
  const int t = (int)gid / H;
  const int base = t * top_k;
  float acc = 0.0f;
  for (int s = 0; s < top_k; ++s) {
    acc += (float)w[base + s] * (float)partials[(int64_t)(base + s) * H + h];
  }
  const float moe = (float)(VPIPE_ELT)acc;   // match moe_out's f16 round
  const float g = (float)gbuf[t];
  x[gid] = (VPIPE_ELT)((float)x[gid] + moe + g * (float)shared_out[gid]);
}

// out[i] = gelu_approx(gate[i]) * up[i], where gelu_approx is the
// gelu_pytorch_tanh used by Gemma-4 (both the geglu MLP and the
// per-layer-input gate, where `up` carries the per-layer-input vector).
//   gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
//   0:gate 1:up 2:out 3:n.  grid (n). In-place safe (out may alias gate).
kernel void geglu_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out   [[buffer(2)]],
    constant int&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float x = float(gate[gid]);
  const float k0 = 0.7978845608028654f;            // sqrt(2/pi)
  const float t = precise::tanh(k0 * (x + 0.044715f * x * x * x));
  const float gelu = 0.5f * x * (1.0f + t);
  out[gid] = VPIPE_ELT(gelu * float(up[gid]));
}

// GeGLU over an INTERLEAVED gate|up matrix [rows, 2*ffn] (column 2g =
// gate feature g, 2g+1 = up feature g -- the layout produced by the dense
// matmul2d over the interleaved gate/up weight, mirroring the steel
// affine_qmm_geglu fused epilogue). Writes out[rows, ffn] =
// gelu_pytorch_tanh(gate)*up (the gelu variant Gemma-4 uses, matching
// geglu_f16 above so the matrix-core MLP stays token-exact with steel).
//   0:in[rows,2*ffn] 1:out[rows,ffn] 2:rows 3:ffn.  grid (rows*ffn).
kernel void geglu_interleaved_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& ffn  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(rows * ffn)) { return; }
  const int r = (int)gid / ffn;
  const int g = (int)gid % ffn;
  const int64_t base = (int64_t)r * (2 * ffn) + 2 * g;
  const float x = float(in[base]);
  const float up = float(in[base + 1]);
  const float k0 = 0.7978845608028654f;            // sqrt(2/pi)
  const float t = precise::tanh(k0 * (x + 0.044715f * x * x * x));
  const float gelu = 0.5f * x * (1.0f + t);
  out[gid] = VPIPE_ELT(gelu * up);
}

// out[i] = tanh(x[i]/cap) * cap  (Gemma-4 final-logit softcapping).
//   0:x 1:out 2:n 3:cap(float).  grid (n). In-place safe.
kernel void softcap_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    constant float&    cap [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  out[gid] = VPIPE_ELT(precise::tanh(float(x[gid]) / cap) * cap);
}

// Forbid specific tokens: overwrite logits[ids[i]] with a large negative
// sentinel so they are never argmaxed and get ~0 weight in the sampler
// (which does exp(logit - maxlogit), so a far-negative logit -> weight 0).
// Used to keep realtime decode short by banning Gemma's reasoning-channel
// open tokens so the model can't spend the budget on a <|channel>thought
// ...<channel|> block. MUST run AFTER the final softcap (else the cap pulls
// it back). The sentinel is finite in both f16 and bf16 and far below
// Gemma's capped logit range. n_ids is small.
//   0:logits 1:ids(int32) 2:n_ids 3:V.  grid (n_ids).
kernel void suppress_logits_f16(
    device VPIPE_ELT*   logits [[buffer(0)]],
    const device int*   ids    [[buffer(1)]],
    constant int&       n_ids  [[buffer(2)]],
    constant int&       V      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n_ids) { return; }
  const int id = ids[gid];
  if (id >= 0 && id < V) { logits[id] = (VPIPE_ELT)(-6.0e4f); }
}

// x[i] *= s  (Gemma-4 embed_scale, per-layer scalar, PLE scales). In
// place.   0:x 1:n 2:s(float).  grid (n).
kernel void scale_inplace_f16(
    device VPIPE_ELT*  x [[buffer(0)]],
    constant int&      n [[buffer(1)]],
    constant float&    s [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  x[gid] = VPIPE_ELT(float(x[gid]) * s);
}

// out[i] = a[i] + b[i].   0:a 1:b 2:out 3:n
// NON-FINITE TRIPWIRE: scan a tensor on the GPU and record the FIRST
// offender, without draining anything.
//
// Reading a buffer back on the CPU to look for a NaN costs a full GPU
// drain per check, which changes the timing of the very thing being
// measured -- on MiniMax-H3 at video geometry, draining between ops made
// a reproducible corruption disappear. This runs in the stream like any
// other dispatch and writes a fixed 8-word report, so the check is one
// extra pass over the tensor and the report is read ONCE per forward.
//
// `tag` identifies the call site (block * 16 + op). The first thread to
// find a non-finite element claims the report with a compare-exchange
// and writes what it saw; every other one only bumps the count. So the
// report names the earliest DETECTED offender rather than a race between
// reporters, and the count says whether it was one element or a region.
//
// report[0] claimed flag   report[1] tag        report[2] flat index
// report[3] raw bits       report[4] total non-finite found
//   0:x 1:report(atomic_uint[8]) 2:n 3:tag.  grid (n).
kernel void nan_tripwire_f16(
    const device VPIPE_ELT* x      [[buffer(0)]],
    device atomic_uint*     report [[buffer(1)]],
    constant uint&          n      [[buffer(2)]],
    constant uint&          tag    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= n) { return; }
  const float v = (float)x[gid];
  if (isfinite(v)) { return; }
  atomic_fetch_add_explicit(&report[4], 1u, memory_order_relaxed);
  // PER-CALL-SITE count as well as the running total. Whether a tensor
  // holds five bad elements or four hundred million is the difference
  // between a localised fault and a wholesale one, and a single shared
  // counter cannot tell them apart -- it reported 2.8e9 for a tensor of
  // 4.0e8, which is only a total over every check in the pass.
  //
  // Slot 16 + tag, so a 50-block stack at 16 ops fits in the 1024-word
  // report. Tags past that are dropped rather than aliased onto another
  // site's count: a wrong attribution is worse than a missing one.
  if (tag < 1008u) {
    atomic_fetch_add_explicit(&report[16u + tag], 1u, memory_order_relaxed);
  }
  uint expected = 0u;
  if (!atomic_compare_exchange_weak_explicit(&report[0], &expected, 1u,
                                             memory_order_relaxed,
                                             memory_order_relaxed)) {
    return;                                  // someone else claimed it
  }
  atomic_store_explicit(&report[1], tag, memory_order_relaxed);
  atomic_store_explicit(&report[2], gid, memory_order_relaxed);
  atomic_store_explicit(&report[3],
                        (uint)as_type<ushort>(x[gid]), memory_order_relaxed);
}

kernel void residual_add_f16(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  out[gid] = VPIPE_ELT(float(a[gid]) + float(b[gid]));
}

// Fold split-K's f32 partial planes into the final output, summing in f32 and
// rounding to the compute elt ONCE.
//   out[i] = (ELT)(planes[0][i] + planes[1][i] + ... + planes[S-1][i])
//   0:planes(f32, S*n) 1:out 2:n 3:S.  grid (n).
//
// The point is the single rounding. Folding f16 planes with residual_add
// instead costs S roundings against the single-op GEMM's one, which measured
// as 37.5% of outputs differing by up to ~5 ulp at the LM down_proj shapes --
// harmless for a rel-L2-verified diffusion model, not something to spend on a
// decoder held to greedy token-exact. With f32 planes the only difference left
// against the single-op path is where the f32 contraction was split, i.e. a
// reassociation whose error lands ~3 orders below the f16 ulp it rounds into.
kernel void splitk_fold_f32_f16(
    const device float* planes [[buffer(0)]],
    device VPIPE_ELT*   out    [[buffer(1)]],
    constant int&       n      [[buffer(2)]],
    constant int&       splits [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  float acc = 0.0f;
  for (int s = 0; s < splits; ++s) {
    acc += planes[(uint)s * (uint)n + gid];
  }
  out[gid] = VPIPE_ELT(acc);
}

// GELU (tanh approximation), the QwenImage FeedForward activation. VPIPE_ELT
// storage so a bf16 metallib exists (the vision gelu_tanh_f16 is half-only).
//   0:x 1:out 2:n.  grid (n).
kernel void gelu_tanh_ff_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float v = (float)x[gid];
  const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  out[gid] = VPIPE_ELT(0.5f * v * (1.0f + metal::precise::tanh(inner)));
}

// Plain LayerNorm, NO affine (elementwise_affine=False): out = (x-mean)/
// sqrt(var+eps), normalized over the last H dims. VPIPE_ELT storage (a bf16
// metallib exists; the vision layer_norm_bias_f16 is half-only + needs w/b).
// One threadgroup of LN_FF_TG threads per row (row = tid.y).
#define LN_FF_TG 256
kernel void layer_norm_plain_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant float&    eps [[buffer(3)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* xr = x + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;
  float s1 = 0.0f, s2 = 0.0f;
  for (int i = (int)lid; i < H; i += LN_FF_TG) {
    const float v = (float)xr[i];
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  threadgroup float p1[LN_FF_TG / 32], p2[LN_FF_TG / 32];
  if (simd_lid == 0) { p1[simd_gid] = s1; p2[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < LN_FF_TG / 32) ? p1[simd_lid] : 0.0f;
    float b = (simd_lid < LN_FF_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); b = simd_sum(b);
    if (simd_lid == 0) { p1[0] = a; p2[0] = b; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = p1[0] / (float)H;
  const float var = p2[0] / (float)H - mean * mean;
  const float inv = rsqrt(var + eps);
  for (int i = (int)lid; i < H; i += LN_FF_TG) {
    outr[i] = VPIPE_ELT(((float)xr[i] - mean) * inv);
  }
}

// Fused (no-affine) LayerNorm + adaLN modulate: out[r,h] = (1+scale[h]) *
// LN(x[r])[h] + shift[h]. Folds layer_norm_plain_f16 + adaln_modulate_f16 into
// one pass over x -- drops the intermediate normed-row write+read and one
// dispatch per DiT norm site. scale/shift are [H] vectors (offset slices of a
// shared modulation buffer) broadcast over the R rows. One threadgroup per row.
kernel void layernorm_modulate_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* scale [[buffer(1)]],
    const device VPIPE_ELT* shift [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&      H   [[buffer(4)]],
    constant float&    eps [[buffer(5)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* xr = x + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;
  float s1 = 0.0f, s2 = 0.0f;
  for (int i = (int)lid; i < H; i += LN_FF_TG) {
    const float v = (float)xr[i];
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  threadgroup float p1[LN_FF_TG / 32], p2[LN_FF_TG / 32];
  if (simd_lid == 0) { p1[simd_gid] = s1; p2[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < LN_FF_TG / 32) ? p1[simd_lid] : 0.0f;
    float b = (simd_lid < LN_FF_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); b = simd_sum(b);
    if (simd_lid == 0) { p1[0] = a; p2[0] = b; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = p1[0] / (float)H;
  const float var = p2[0] / (float)H - mean * mean;
  const float inv = rsqrt(var + eps);
  for (int i = (int)lid; i < H; i += LN_FF_TG) {
    const float ln = ((float)xr[i] - mean) * inv;
    outr[i] = VPIPE_ELT((1.0f + (float)scale[i]) * ln + (float)shift[i]);
  }
}

// adaLN modulation (Krea-2 DiT): out[m,n] = (1 + scale[n]) * x[m,n] + shift[n],
// with scale/shift [N] broadcast over the M rows (total = M*N). scale/shift may
// be offset slices of a shared modulation buffer.
kernel void adaln_modulate_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* scale [[buffer(1)]],
    const device VPIPE_ELT* shift [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&      N     [[buffer(4)]],
    constant int&      total [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const uint n = gid % (uint)N;
  out[gid] =
      VPIPE_ELT((1.0f + float(scale[n])) * float(x[gid]) + float(shift[n]));
}

// Gated residual (Krea-2 DiT): h[m,n] += gate[n] * sub[m,n], gate [N] broadcast
// over the M rows (total = M*N). In place on h.
kernel void gated_residual_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    const device VPIPE_ELT* sub  [[buffer(2)]],
    constant int&      N     [[buffer(3)]],
    constant int&      total [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const uint n = gid % (uint)N;
  h[gid] = VPIPE_ELT(float(h[gid]) + float(gate[n]) * float(sub[gid]));
}

// Gated residual with a TANH-squashed gate (Lumina / NextDiT "RMSNormZero",
// the Boogu DiT): h[m,n] += tanh(gate[n]) * sub[m,n], gate [N] broadcast over
// the M rows (total = M*N). In place on h. The tanh is part of the model, not
// a numerical guard -- the gate Linear is zero-initialised and trained through
// tanh, so dropping it changes the result.
kernel void gated_residual_tanh_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    const device VPIPE_ELT* sub  [[buffer(2)]],
    constant int&      N     [[buffer(3)]],
    constant int&      total [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const uint n = gid % (uint)N;
  h[gid] = VPIPE_ELT(float(h[gid]) +
                     precise::tanh(float(gate[n])) * float(sub[gid]));
}

// ---- row-major vec4 twins of the three adaLN/gate kernels ----------------
// One-element-per-thread costs ~4x: at a DiT's [2271, 3360] the scalar kernels
// above reach only 37-54 GB/s (measured, boogu_perf.elementwise_shapes) where
// rms_norm_fast_f16 over the same bytes reaches 154 -- the limit is threads
// retired, not bandwidth. These take a 2-D grid (N/4 columns, M rows), so the
// row index is free (no per-element integer modulo, which is emulated on Apple
// GPUs) and every access is an 8-byte vector. Require N % 4 == 0; the callers
// keep the scalar kernels for the odd widths.
#define VPIPE_ELT4 vec<VPIPE_ELT, 4>

kernel void adaln_modulate_v4_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* scale [[buffer(1)]],
    const device VPIPE_ELT* shift [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&      N4    [[buffer(4)]],    // N / 4
    constant int&      M     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)N4 || gid.y >= (uint)M) { return; }
  const uint o = gid.y * (uint)N4 + gid.x;
  const float4 sc = float4(
      reinterpret_cast<const device VPIPE_ELT4*>(scale)[gid.x]);
  const float4 sh = float4(
      reinterpret_cast<const device VPIPE_ELT4*>(shift)[gid.x]);
  const float4 v = float4(reinterpret_cast<const device VPIPE_ELT4*>(x)[o]);
  reinterpret_cast<device VPIPE_ELT4*>(out)[o] =
      VPIPE_ELT4((1.0f + sc) * v + sh);
}

kernel void gated_residual_v4_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    const device VPIPE_ELT* sub  [[buffer(2)]],
    constant int&      N4   [[buffer(3)]],
    constant int&      M    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)N4 || gid.y >= (uint)M) { return; }
  const uint o = gid.y * (uint)N4 + gid.x;
  const float4 g = float4(
      reinterpret_cast<const device VPIPE_ELT4*>(gate)[gid.x]);
  const float4 s = float4(reinterpret_cast<const device VPIPE_ELT4*>(sub)[o]);
  device VPIPE_ELT4* hv = reinterpret_cast<device VPIPE_ELT4*>(h);
  hv[o] = VPIPE_ELT4(float4(hv[o]) + g * s);
}

kernel void gated_residual_tanh_v4_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* gate [[buffer(1)]],
    const device VPIPE_ELT* sub  [[buffer(2)]],
    constant int&      N4   [[buffer(3)]],
    constant int&      M    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)N4 || gid.y >= (uint)M) { return; }
  const uint o = gid.y * (uint)N4 + gid.x;
  const float4 gr = float4(
      reinterpret_cast<const device VPIPE_ELT4*>(gate)[gid.x]);
  const float4 g = float4(precise::tanh(gr.x), precise::tanh(gr.y),
                          precise::tanh(gr.z), precise::tanh(gr.w));
  const float4 s = float4(reinterpret_cast<const device VPIPE_ELT4*>(sub)[o]);
  device VPIPE_ELT4* hv = reinterpret_cast<device VPIPE_ELT4*>(h);
  hv[o] = VPIPE_ELT4(float4(hv[o]) + g * s);
}

// out = a + b over 4 elements per thread, 1-D grid of n/4 threads (n % 4 == 0).
kernel void residual_add_v4_f16(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      n4  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n4) { return; }
  const float4 x = float4(reinterpret_cast<const device VPIPE_ELT4*>(a)[gid]);
  const float4 y = float4(reinterpret_cast<const device VPIPE_ELT4*>(b)[gid]);
  reinterpret_cast<device VPIPE_ELT4*>(out)[gid] = VPIPE_ELT4(x + y);
}

// Broadcast a per-column bias over the rows of a row-major [M, N] matrix,
// in place: y[m, n] += bias[n]. Used to fold the linear bias back after the
// matmul2d dense GEMM (dense_gemm_mma), which omits bias.
//   0:y[M*N] (inout) 1:bias[N] 2:N 3:total(=M*N).  grid (total) or (N, M).
// total + gid are UINT: a >=~2K-px VAE bias fold has M*N > 2^31, which would
// overflow a signed int. The flat index is gid = tpig.y*N + tpig.x -- a 1D
// {M*N} grid passes it directly (tpig.y == 0) and a 2D {N, M} grid keeps each
// dimension small; both dispatch shapes are equivalent.
kernel void bias_add_rows_f16(
    device VPIPE_ELT*       y    [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    constant int&      N     [[buffer(2)]],
    constant uint&     total [[buffer(3)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const uint gid = tpig.y * (uint)N + tpig.x;
  if (gid >= total) { return; }
  y[gid] = VPIPE_ELT(float(y[gid]) + float(bias[gid % (uint)N]));
}

// out[i] = clamp(in[i], lo, hi). Serves the Gemma-4 audio ClippableLinear
// input/output bounds and ReLU (lo=0, hi=+big). In-place safe.
//   0:in 1:out 2:n 3:lo(float) 4:hi(float).  grid (n).
kernel void clamp_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    constant float&    lo  [[buffer(3)]],
    constant float&    hi  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  out[gid] = VPIPE_ELT(min(max(float(in[gid]), lo), hi));
}

// im2col for a channel-last [H*W, C] feature map -> [H*W, 9*C]: the 3x3
// neighborhood (pad 1, stride 1) flattened as (ky,kx,c) so that
// out[r, (ky*3+kx)*C + c] = in[(y+ky-1)*W + (x+kx-1), c] (0 outside the
// border), with r = y*W + x. A dense_gemm over W[Cout, 9*C] then realizes a
// 3x3 conv2d. Serves the Qwen-Image VAE decoder (single-frame causal-conv3d
// collapses to conv2d using only the kt=2 weight slice).
//   0:in[H*W,C] 1:out[H*W,9*C] 2:H 3:W 4:C.  grid (H*W*9*C) or (9*C, H*W).
// The output element index is the flat position `gid` = row*(9*C) + (ky*3+kx)*C
// + c. A 1D dispatch {H*W*9*C,1,1} passes it directly (tpig.y == 0); callers
// with a large H*W*9*C (a 1024px VAE-decode conv is > 2^31) may instead pass a
// 2D grid {9*C, H*W, 1} to keep each grid dimension small, and gid is
// reconstructed as tpig.y*(9*C) + tpig.x -- which reduces to tpig.x for the 1D
// form, so both dispatch shapes are equivalent.
kernel void im2col_hwc_3x3_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant int&      W   [[buffer(3)]],
    constant int&      C   [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]],
    uint2 tpg  [[threads_per_grid]])
{
  // gid + total are ULONG (64-bit): a VAE-decode conv's flat element count
  // H*W*9*C exceeds 2^32 once the OUTPUT area passes ~1.86M px at C=256 (the top
  // up-block's 256-ch full-res resnet: e.g. 2048*1536*9*256 = 7.26G) -- both a
  // signed int (>2^31 at 1024px, 1024*1024*9*256 = 2.4G) and a uint (>2^32 at 2K)
  // wrap, leaving the im2col partially written -> a tiled/gapped decode. 64-bit
  // holds it; r fits uint (H*W <= 2^32 for any real resolution).
  //
  // But a 64-bit div/mod is SOFTWARE-emulated on Apple GPUs (~100+ cycles), and
  // this kernel does three of them per thread over H*W*9*C threads -- which made
  // the VAE's im2col run at ~3 GB/s, ~1% of the box's bandwidth and 7.7x the cost
  // of the GEMM it feeds. Under the documented 2D grid {9*C, rows} the same
  // indices are pure 32-bit: tpig.x IS the within-row column (< 9*C, so c/j
  // divide a small uint) and tpig.y IS the row. Take that path when the grid
  // says so, and keep the 64-bit flat form for the 1D dispatch shape. Both
  // produce identical indices; only the arithmetic width differs.
  const uint cols = (uint)(9 * C);
  const ulong total = (ulong)H * (uint)W * 9u * (uint)C;
  uint c, j, r;
  ulong gid;
  if (tpg.x == cols) {                       // 2D {9*C, H*W}
    if ((uint)tpig.y >= (uint)(H * W)) { return; }
    c = (uint)tpig.x % (uint)C;
    j = (uint)tpig.x / (uint)C;              // ky*3 + kx
    r = (uint)tpig.y;
    gid = (ulong)tpig.y * cols + tpig.x;
  } else {                                   // 1D flat
    gid = (ulong)tpig.y * cols + tpig.x;
    if (gid >= total) { return; }
    c = (uint)(gid % (uint)C);
    j = (uint)((gid / (uint)C) % 9u);
    r = (uint)(gid / (ulong)cols);
  }
  const int x = (int)(r % (uint)W);
  const int y = (int)(r / (uint)W);
  const int sy = y + (int)(j / 3u) - 1;
  const int sx = x + (int)(j % 3u) - 1;
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
    val = in[((ulong)sy * W + sx) * (uint)C + c];
  }
  out[gid] = val;
}

// Row-tiled im2col: same 3x3 channel-last expansion as im2col_hwc_3x3_f16, but
// materializes only OUTPUT rows [row_off, row_off+row_cnt) into a TILE-LOCAL
// out[0 .. row_cnt*9*C). Lets a memory-pressured caller stream a conv in row
// bands so the col scratch is one band, not the full [H*W, 9*C] (a 1024^2 conv
// at C=256 is ~4.6 GB). The input `in` is the WHOLE [H*W, C] map (the 3x3 halo
// reads span the band edges), so the band boundary needs no overlap handling.
// row_off=0, row_cnt=H*W reproduces im2col_hwc_3x3_f16 byte-for-byte.
//   0:in[H*W,C] 1:out[row_cnt,9*C] 2:H 3:W 4:C 5:row_off 6:row_cnt.
//   grid {9*C, row_cnt}.
kernel void im2col_hwc_3x3_tiled_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H       [[buffer(2)]],
    constant int&      W       [[buffer(3)]],
    constant int&      C       [[buffer(4)]],
    constant int&      row_off [[buffer(5)]],
    constant int&      row_cnt [[buffer(6)]],
    uint2 tpig [[thread_position_in_grid]],
    uint2 tpg  [[threads_per_grid]])
{
  // 32-bit index math on the documented 2D grid -- see the note on
  // im2col_hwc_3x3_f16: the 64-bit div/mod triple is software-emulated and was
  // the VAE im2col's actual cost. tpig.x < 9*C here, so c/j divide a small uint.
  const uint cols = (uint)(9 * C);
  ulong gid;
  uint c, j, r;
  if (tpg.x == cols) {                       // 2D {9*C, row_cnt}
    if ((uint)tpig.y >= (uint)row_cnt) { return; }
    c = (uint)tpig.x % (uint)C;
    j = (uint)tpig.x / (uint)C;              // ky*3 + kx
    r = (uint)row_off + (uint)tpig.y;        // global out row
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
  } else {
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
    if (gid >= (ulong)(uint)row_cnt * cols) { return; }
    c = (uint)(gid % (uint)C);
    j = (uint)((gid / (uint)C) % 9u);
    r = (uint)row_off + (uint)(gid / (ulong)cols);
  }
  const int x = (int)(r % (uint)W);
  const int y = (int)(r / (uint)W);
  const int sy = y + (int)(j / 3u) - 1;
  const int sx = x + (int)(j % 3u) - 1;
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
    val = in[((ulong)sy * W + sx) * (uint)C + c];
  }
  out[gid] = val;
}

// Per-column running abs-max accumulate: out[n] = max(out[n], max_m |in[m,n]|)
// over the M rows of in[M,N]. One thread per column (no write races); `out` is
// read+written so it ACCUMULATES across calls (zero it before the first) -- the
// DiT's on-device AWQ activation calibration taps each Linear's input this way
// across prompts x denoising timesteps. Reduces in float, stores f16.
//   0:in[M,N] 1:out[N] 2:M 3:N.  grid (N).
kernel void col_absmax_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      M   [[buffer(2)]],
    constant int&      N   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if ((int)gid >= N) { return; }
  float mx = float(out[gid]);
  for (int m = 0; m < M; ++m) {
    const float v = metal::fabs(float(in[(uint)m * N + gid]));
    if (v > mx) { mx = v; }
  }
  out[gid] = VPIPE_ELT(mx);
}

// Strided im2col for a downsample conv: channel-last [H*W, C] -> the strided
// 3x3 neighborhood at stride 2 with pad (left 0, right 1, top 0, bottom 1) --
// the QwenImage VAE encoder's QwenImageResample (ZeroPad2d((0,1,0,1)) +
// Conv2d stride 2). Output [(H/2)*(W/2), 9*C]: out[r, (ky*3+kx)*C + c] =
// in[(oy*2+ky)*W + (ox*2+kx), c] (0 when the source row/col reaches the padded
// H/W), r = oy*(W/2) + ox. Pairs with W[Cout, 9*C].
//   grid ((H/2)*(W/2)*9*C) or (9*C, (H/2)*(W/2)).
// gid + total are UINT: at >=~2K px an encoder-downsample im2col has
// (H/2)*(W/2)*9*C > 2^31, which overflows a signed int -- `total` goes
// negative, every thread early-returns, and the im2col is left all-zero (the
// grey-decode failure mode). The flat index gid = tpig.y*(9*C) + tpig.x lets a
// caller pass a 2D {9*C, (H/2)*(W/2)} grid so no single dimension is oversized;
// it reduces to tpig.x for the 1D form, so both shapes are equivalent.
kernel void im2col_hwc_3x3_s2_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant int&      W   [[buffer(3)]],
    constant int&      C   [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]],
    uint2 tpg  [[threads_per_grid]])
{
  const int OH = H / 2, OW = W / 2;
  // 32-bit index math on the 2D grid -- see im2col_hwc_3x3_f16.
  const uint cols = (uint)(9 * C);
  ulong gid;
  uint c, j, r;
  if (tpg.x == cols) {                       // 2D {9*C, OH*OW}
    if ((uint)tpig.y >= (uint)(OH * OW)) { return; }
    c = (uint)tpig.x % (uint)C;
    j = (uint)tpig.x / (uint)C;              // ky*3 + kx
    r = (uint)tpig.y;
    gid = (ulong)tpig.y * cols + tpig.x;
  } else {
    gid = (ulong)tpig.y * cols + tpig.x;
    if (gid >= (ulong)OH * (uint)OW * 9u * (uint)C) { return; }
    c = (uint)(gid % (uint)C);
    j = (uint)((gid / (uint)C) % 9u);
    r = (uint)(gid / (ulong)cols);
  }
  const int ox = (int)(r % (uint)OW);
  const int oy = (int)(r / (uint)OW);
  const int iy = oy * 2 + (int)(j / 3u);    // no -1: pad is (0,1,0,1)
  const int ix = ox * 2 + (int)(j % 3u);
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (iy < H && ix < W) { val = in[((ulong)iy * W + ix) * (uint)C + c]; }
  out[gid] = val;
}

// Row-tiled strided (stride-2, pad (0,1,0,1)) im2col -- the tiled twin of
// im2col_hwc_3x3_s2_f16, for the VAE-ENCODE downsample convs. Materializes only
// downsampled OUTPUT rows [row_off, row_off+row_cnt) of the [(H/2)*(W/2), 9*C]
// im2col into a TILE-LOCAL out. `in` is the whole [H*W, C] input (the strided
// 3x3 halo spans band edges). row_off=0, row_cnt=(H/2)*(W/2) == the untiled
// kernel byte-for-byte.
//   0:in[H*W,C] 1:out[row_cnt,9*C] 2:H 3:W 4:C 5:row_off 6:row_cnt.
//   grid {9*C, row_cnt}.
kernel void im2col_hwc_3x3_s2_tiled_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H       [[buffer(2)]],
    constant int&      W       [[buffer(3)]],
    constant int&      C       [[buffer(4)]],
    constant int&      row_off [[buffer(5)]],
    constant int&      row_cnt [[buffer(6)]],
    uint2 tpig [[thread_position_in_grid]],
    uint2 tpg  [[threads_per_grid]])
{
  const int OW = W / 2;
  // 32-bit index math on the 2D grid -- see im2col_hwc_3x3_f16.
  const uint cols = (uint)(9 * C);
  ulong gid;
  uint c, j, r;
  if (tpg.x == cols) {                       // 2D {9*C, row_cnt}
    if ((uint)tpig.y >= (uint)row_cnt) { return; }
    c = (uint)tpig.x % (uint)C;
    j = (uint)tpig.x / (uint)C;              // ky*3 + kx
    r = (uint)row_off + (uint)tpig.y;        // global out row
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
  } else {
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
    if (gid >= (ulong)(uint)row_cnt * cols) { return; }
    c = (uint)(gid % (uint)C);
    j = (uint)((gid / (uint)C) % 9u);
    r = (uint)row_off + (uint)(gid / (ulong)cols);
  }
  const int ox = (int)(r % (uint)OW);
  const int oy = (int)(r / (uint)OW);
  const int iy = oy * 2 + (int)(j / 3u);    // no -1: pad is (0,1,0,1)
  const int ix = ox * 2 + (int)(j % 3u);
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (iy < H && ix < W) { val = in[((ulong)iy * W + ix) * (uint)C + c]; }
  out[gid] = val;
}

// Nearest spatial upsample (== nearest-exact for integer 2x, since both map
// out index o -> o/2) of a channel-last [H*W, C] map to [(2H)*(2W), C]:
// out[(oy,ox), c] = in[(oy/2, ox/2), c]. Qwen-Image VAE decoder resample.
//   0:in[H*W,C] 1:out[4*H*W,C] 2:H 3:W 4:C.  grid (4*H*W*C) or (C, 4*H*W).
// gid + total are UINT (a >=~3K-px decode upsample has 4*H*W*C > 2^31, which
// would overflow a signed int -- the grey-decode all-zero failure mode). The
// flat index gid = tpig.y*C + tpig.x lets a caller pass a 2D {C, 4*H*W} grid;
// it reduces to tpig.x for the 1D form, so both shapes are equivalent.
kernel void upsample_nearest2x_hwc_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant int&      W   [[buffer(3)]],
    constant int&      C   [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const int OW = 2 * W, OH = 2 * H;
  const ulong gid = (ulong)tpig.y * (uint)C + tpig.x;
  const ulong total = (ulong)OH * (uint)OW * (uint)C;
  if (gid >= total) { return; }
  const uint c = (uint)(gid % (uint)C);
  const uint p = (uint)(gid / (uint)C);
  const int ox = (int)(p % (uint)OW);
  const int oy = (int)(p / (uint)OW);
  out[gid] = in[((ulong)(oy / 2) * W + (ox / 2)) * (uint)C + c];
}

// GLU over a [rows, 2*D] matrix split into halves: each row is [a(D) |
// g(D)]; out[r,d] = a[r,d] * sigmoid(g[r,d]). Gemma-4 audio lconv1d GLU.
//   0:in[rows,2*D] 1:out[rows,D] 2:rows 3:D.  grid (rows*D).
kernel void glu_split_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      rows [[buffer(2)]],
    constant int&      D    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(rows * D)) { return; }
  const int r = (int)gid / D;
  const int d = (int)gid % D;
  const int64_t base = (int64_t)r * (2 * D);
  const float a = float(in[base + d]);
  const float g = float(in[base + D + d]);
  out[gid] = VPIPE_ELT(a * (1.0f / (1.0f + exp(-g))));
}

// GroupNorm over a channel-last activation [rows(=H*W), C] with G groups and
// per-channel affine (gamma/beta [C]). Each group's statistics reduce over ALL
// rows x (C/G) channels (the diffusers AutoencoderKL/UNet norm). One threadgroup
// per group (grid.y = G): a grid-stride reduction of sum/sumsq, then a second
// grid-stride pass writes out = (x-mean)/sqrt(var+eps)*gamma + beta.
//   0:x[rows,C] 1:gamma[C] 2:beta[C] 3:out[rows,C] 4:rows 5:C 6:G 7:eps
//   grid {256, G, 1}, tg {256,1,1}.
kernel void group_norm_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* gamma [[buffer(1)]],
    const device VPIPE_ELT* beta  [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&      rows [[buffer(4)]],
    constant int&      C    [[buffer(5)]],
    constant int&      G    [[buffer(6)]],
    constant float&    eps  [[buffer(7)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 ltid  [[thread_position_in_threadgroup]],
    uint3 tptg  [[threads_per_threadgroup]])
{
  const uint lid  = ltid.x;
  const uint tgsz = tptg.x;
  const int g  = (int)tgid.y;
  const int Cg = C / G;
  const int c0 = g * Cg;
  const long total = (long)rows * (long)Cg;
  threadgroup float ssum[256];
  threadgroup float ssq[256];
  float s = 0.0f, sq = 0.0f;
  for (long i = (long)lid; i < total; i += (long)tgsz) {
    const int r  = (int)(i / Cg);
    const int cc = (int)(i % Cg);
    const float v = float(x[(long)r * C + c0 + cc]);
    s += v; sq += v * v;
  }
  ssum[lid] = s; ssq[lid] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = tgsz / 2; st > 0; st >>= 1) {
    if (lid < st) { ssum[lid] += ssum[lid + st]; ssq[lid] += ssq[lid + st]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = ssum[0] / (float)total;
  const float var  = ssq[0] / (float)total - mean * mean;
  const float inv  = 1.0f / sqrt(var + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (long i = (long)lid; i < total; i += (long)tgsz) {
    const int r  = (int)(i / Cg);
    const int cc = (int)(i % Cg);
    const long idx = (long)r * C + c0 + cc;
    const float v = float(x[idx]);
    out[idx] = VPIPE_ELT((v - mean) * inv * float(gamma[c0 + cc])
                         + float(beta[c0 + cc]));
  }
}

// GroupNorm over a channel-last VIDEO activation [T, rows(=H*W), C] in which
// every FRAME normalizes on its own -- the MiniMax-H3 video encoder's
// `use_t_isolated_gn`, where the reference folds the temporal axis into the
// batch axis so statistics never mix across frames. Running the plain
// group_norm_f16 over the whole clip instead is a different function: it
// couples frames that the encoder deliberately keeps independent, and it
// fails silently because a single frame's own statistics are a good enough
// approximation for the output to look plausible.
//
// `do_silu` folds the SiLU that follows every one of these norms in the
// encoder into the write, halving the passes over an activation that is the
// largest thing in the encode (~285 MB for a 17-frame 256x256 tile at 128
// channels).
//   0:x[T,rows,C] 1:gamma[C] 2:beta[C] 3:out 4:rows 5:C 6:G 7:eps 8:do_silu
//   grid {256, G, T}, tg {256,1,1}: one threadgroup per (frame, group).
kernel void group_norm_frames_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* gamma [[buffer(1)]],
    const device VPIPE_ELT* beta  [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&      rows    [[buffer(4)]],
    constant int&      C       [[buffer(5)]],
    constant int&      G       [[buffer(6)]],
    constant float&    eps     [[buffer(7)]],
    constant int&      do_silu [[buffer(8)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 ltid  [[thread_position_in_threadgroup]],
    uint3 tptg  [[threads_per_threadgroup]])
{
  const uint lid  = ltid.x;
  const uint tgsz = tptg.x;
  const int  g    = (int)tgid.y;
  const int  Cg   = C / G;
  const int  c0   = g * Cg;
  const long frame = (long)tgid.z * (long)rows * (long)C;
  const long total = (long)rows * (long)Cg;
  threadgroup float ssum[256];
  threadgroup float ssq[256];
  float s = 0.0f, sq = 0.0f;
  for (long i = (long)lid; i < total; i += (long)tgsz) {
    const int r  = (int)(i / Cg);
    const int cc = (int)(i % Cg);
    const float v = float(x[frame + (long)r * C + c0 + cc]);
    s += v; sq += v * v;
  }
  ssum[lid] = s; ssq[lid] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = tgsz / 2; st > 0; st >>= 1) {
    if (lid < st) { ssum[lid] += ssum[lid + st]; ssq[lid] += ssq[lid + st]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = ssum[0] / (float)total;
  const float var  = max(ssq[0] / (float)total - mean * mean, 0.0f);
  const float inv  = 1.0f / sqrt(var + eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (long i = (long)lid; i < total; i += (long)tgsz) {
    const int r  = (int)(i / Cg);
    const int cc = (int)(i % Cg);
    const long idx = frame + (long)r * C + c0 + cc;
    float v = (float(x[idx]) - mean) * inv * float(gamma[c0 + cc])
              + float(beta[c0 + cc]);
    if (do_silu != 0) { v = v * (1.0f / (1.0f + exp(-v))); }
    out[idx] = VPIPE_ELT(v);
  }
}

// Causal depthwise Conv1d, kernel K, per-channel weight w[D, K] (tap kk
// pairs with input position t+kk-(K-1); left-padded with zeros). One
// thread per (t, d).   0:in[T,D] 1:w[D,K] 2:out[T,D] 3:T 4:D 5:K.
// grid (T*D).
kernel void depthwise_conv1d_causal_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    const device VPIPE_ELT* w   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      T   [[buffer(3)]],
    constant int&      D   [[buffer(4)]],
    constant int&      K   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(T * D)) { return; }
  const int t = (int)gid / D;
  const int d = (int)gid % D;
  float acc = 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    const int ti = t + kk - (K - 1);
    if (ti < 0) { continue; }
    acc += float(in[(uint)ti * D + d]) * float(w[(uint)d * K + kk]);
  }
  out[gid] = VPIPE_ELT(acc);
}

// ---- MageVAE (DiCo conv codec) ops --------------------------------------
// Depthwise 3x3 conv, stride 1, pad 1, CHANNEL-LAST [H*W, C] (the VAE
// layout): out[p, c] = sum_ky,kx in[(y+ky-1)*W + (x+kx-1), c] * w[c, ky, kx]
// + b[c], zero outside the image. Each channel has its own 3x3 tap set
// (groups == C), so unlike the dense 3x3 there is no im2col/GEMM to fold it
// into -- one thread per (pixel, channel) reads 9 neighbours directly.
//   0:in[H*W,C] 1:w[C,9] 2:b[C] 3:out[H*W,C] 4:H 5:W 6:C 7:has_bias.
// grid (C, H*W) -- 2D so neither dimension overflows at >=1K px.
kernel void depthwise_conv2d_3x3_hwc_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    const device VPIPE_ELT* w   [[buffer(1)]],
    const device VPIPE_ELT* b   [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&      H   [[buffer(4)]],
    constant int&      W   [[buffer(5)]],
    constant int&      C   [[buffer(6)]],
    constant int&      has_bias [[buffer(7)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const int c = (int)tpig.x;
  const uint p = tpig.y;
  if (c >= C || p >= (uint)(H * W)) { return; }
  const int y = (int)(p / (uint)W);
  const int x = (int)(p % (uint)W);
  float acc = has_bias ? float(b[c]) : 0.0f;
  for (int ky = 0; ky < 3; ++ky) {
    const int yy = y + ky - 1;
    if (yy < 0 || yy >= H) { continue; }
    for (int kx = 0; kx < 3; ++kx) {
      const int xx = x + kx - 1;
      if (xx < 0 || xx >= W) { continue; }
      acc += float(in[((uint)yy * (uint)W + (uint)xx) * (uint)C + (uint)c])
             * float(w[(uint)c * 9u + (uint)(ky * 3 + kx)]);
    }
  }
  out[p * (uint)C + (uint)c] = VPIPE_ELT(acc);
}

// Column mean of a row-major [M, N] matrix: out[n] = (1/M) sum_m x[m, n].
// The spatial global-average-pool of the DiCo channel-attention branch
// (AdaptiveAvgPool2d(1) over a channel-last [H*W, C] activation). One
// threadgroup per column, strided over rows, f32 accumulation.
//   0:x[M,N] 1:out[N] 2:M 3:N.  grid (N * CM_TG) / tg (CM_TG).
#define CM_TG 256
kernel void col_mean_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      M   [[buffer(2)]],
    constant int&      N   [[buffer(3)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint n = tid.x;
  if (n >= (uint)N) { return; }
  float s = 0.0f;
  for (uint m = ltid.x; m < (uint)M; m += CM_TG) {
    s += float(x[m * (uint)N + n]);
  }
  s = simd_sum(s);
  threadgroup float part[CM_TG / 32];
  if (simd_lid == 0) { part[simd_gid] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < CM_TG / 32) ? part[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { out[n] = VPIPE_ELT(a / (float)M); }
  }
}

// Per-column gated scale, in place: y[m, n] *= sigmoid(v[n]). Fuses the
// Sigmoid tail of the DiCo channel-attention with its broadcast multiply
// (the pooled 1x1-conv logits arrive unactivated), so the [C] vector is
// read once and no separate sigmoid pass is needed.
//   0:y[M*N] (inout) 1:v[N] 2:N 3:total(=M*N).  grid (N, M).
kernel void mul_rows_sigmoid_f16(
    device VPIPE_ELT*       y [[buffer(0)]],
    const device VPIPE_ELT* v [[buffer(1)]],
    constant int&      N     [[buffer(2)]],
    constant uint&     total [[buffer(3)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const uint gid = tpig.y * (uint)N + tpig.x;
  if (gid >= total) { return; }
  const float g = 1.0f / (1.0f + metal::precise::exp(-float(v[gid % (uint)N])));
  y[gid] = VPIPE_ELT(float(y[gid]) * g);
}

// Metal has no erf intrinsic; Abramowitz & Stegun 7.1.26 (max abs error
// ~1.5e-7, well under f16 precision). Same formulation as the vision
// metallib's helper -- each .metal is its own metallib, so it is redefined
// here rather than shared.
inline float elt_erf_approx_(float x) {
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

// EXACT (erf) GELU: out = x * 0.5 * (1 + erf(x / sqrt(2))). MageVAE calls
// F.gelu() with the default approximate='none', so the tanh approximation
// used by the DiT feed-forwards (gelu_tanh_ff_f16) is NOT interchangeable
// here -- they differ by ~1e-3 relative near |x| ~ 2.
//   0:x 1:out 2:n.  grid (n).
kernel void gelu_erf_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float v = (float)x[gid];
  out[gid] =
      VPIPE_ELT(0.5f * v * (1.0f + elt_erf_approx_(v * 0.7071067811865476f)));
}

// LayerNorm WITH affine (weight + bias), normalized over the last H dims:
// out = (x-mean)/sqrt(var+eps) * w + b. The plain twin above has no affine
// and the vision metallib's layer_norm_bias_f16 is half-only; MageVAE's
// encoder LayerNorm2d layers carry affine params and run in both element
// types, so it gets a VPIPE_ELT version here.
//   0:x 1:w[H] 2:b[H] 3:out 4:H 5:eps.  grid (1, rows) / tg (LN_FF_TG).
kernel void layer_norm_affine_f16(
    const device VPIPE_ELT* x   [[buffer(0)]],
    const device VPIPE_ELT* w   [[buffer(1)]],
    const device VPIPE_ELT* b   [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&      H   [[buffer(4)]],
    constant float&    eps [[buffer(5)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  const uint lid = ltid.x;
  const device VPIPE_ELT* xr = x + (uint)row * H;
  device VPIPE_ELT* outr = out + (uint)row * H;
  float s1 = 0.0f, s2 = 0.0f;
  for (int i = (int)lid; i < H; i += LN_FF_TG) {
    const float v = (float)xr[i];
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  threadgroup float p1[LN_FF_TG / 32], p2[LN_FF_TG / 32];
  if (simd_lid == 0) { p1[simd_gid] = s1; p2[simd_gid] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < LN_FF_TG / 32) ? p1[simd_lid] : 0.0f;
    float bb = (simd_lid < LN_FF_TG / 32) ? p2[simd_lid] : 0.0f;
    a = simd_sum(a); bb = simd_sum(bb);
    if (simd_lid == 0) { p1[0] = a; p2[0] = bb; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = p1[0] / (float)H;
  const float var = p2[0] / (float)H - mean * mean;
  const float inv = rsqrt(var + eps);
  for (int i = (int)lid; i < H; i += LN_FF_TG) {
    outr[i] = VPIPE_ELT(((float)xr[i] - mean) * inv * (float)w[i]
                        + (float)b[i]);
  }
}

// Gather non-overlapping dxd tiles out of a channel-last [H*W, C] image
// into a tile-major [nt*d*d, C] buffer, REPLICATE-clamping coordinates past
// the edge. This is the MageVAE CoD decoder's patched attention: the
// reference F.pad(..., mode="replicate") to a whole number of dxd tiles is
// exactly a clamp on the gather, so no separate pad pass is needed.
//   0:in[H*W,C] 1:out[nt*d*d,C] 2:H 3:W 4:C 5:d 6:npw.
// grid (C, d*d, nt) with nt = nph*npw.
kernel void tile_gather_clamp_hwc_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant int&      W   [[buffer(3)]],
    constant int&      C   [[buffer(4)]],
    constant int&      d   [[buffer(5)]],
    constant int&      npw [[buffer(6)]],
    uint3 tpig [[thread_position_in_grid]])
{
  const int c = (int)tpig.x;
  const int uv = (int)tpig.y;
  const int t = (int)tpig.z;
  if (c >= C || uv >= d * d) { return; }
  const int u = uv / d, v = uv % d;
  int y = (t / npw) * d + u;
  int x = (t % npw) * d + v;
  y = metal::min(y, H - 1);
  x = metal::min(x, W - 1);
  out[((uint)t * (uint)(d * d) + (uint)uv) * (uint)C + (uint)c] =
      in[((uint)y * (uint)W + (uint)x) * (uint)C + (uint)c];
}

// Scatter tiles back into a channel-last [H*W, C] image, dropping the
// replicate-padded positions that fall outside the image (the reference
// crops h_[:, :, :H, :W] after the tiled attention).
//   0:src[nt*d*d,C] 1:out[H*W,C] 2:H 3:W 4:C 5:d 6:npw.
// grid (C, d*d, nt).
kernel void tile_scatter_hwc_f16(
    const device VPIPE_ELT* src [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant int&      W   [[buffer(3)]],
    constant int&      C   [[buffer(4)]],
    constant int&      d   [[buffer(5)]],
    constant int&      npw [[buffer(6)]],
    uint3 tpig [[thread_position_in_grid]])
{
  const int c = (int)tpig.x;
  const int uv = (int)tpig.y;
  const int t = (int)tpig.z;
  if (c >= C || uv >= d * d) { return; }
  const int y = (t / npw) * d + uv / d;
  const int x = (t % npw) * d + uv % d;
  if (y >= H || x >= W) { return; }
  out[((uint)y * (uint)W + (uint)x) * (uint)C + (uint)c] =
      src[((uint)t * (uint)(d * d) + (uint)uv) * (uint)C + (uint)c];
}

// Add a row-cyclic constant table: y[r, n] += tbl[(r % P), n]. The MageVAE
// decoder's NerfEmbedder contributes a fixed per-intra-patch-pixel DCT
// position term (folded with the linear bias at load into a [P, N] table,
// P = patch*patch), which repeats for every patch.
//   0:y[M,N] (inout) 1:tbl[P,N] 2:N 3:P 4:total(=M*N).  grid (N, M).
kernel void add_rows_mod_f16(
    device VPIPE_ELT*       y   [[buffer(0)]],
    const device VPIPE_ELT* tbl [[buffer(1)]],
    constant int&      N     [[buffer(2)]],
    constant int&      P     [[buffer(3)]],
    constant uint&     total [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const uint gid = tpig.y * (uint)N + tpig.x;
  if (gid >= total) { return; }
  const uint n = gid % (uint)N;
  const uint r = gid / (uint)N;
  y[gid] = VPIPE_ELT(float(y[gid]) + float(tbl[(r % (uint)P) * (uint)N + n]));
}

// ---- MageVAE per-pixel MLP head: f32 residual stream -------------------
// The decoder's per-pixel MLP is a chain of GATED residual adds with no
// normalization until the very end, so its residual grows enormously: on a
// real photo the row entering the 3 blocks peaks at |x| ~ 39, and the blocks
// take it to ~2.7e4, ~5.7e4, then ~1.5e5. f16 saturates at 65504, so the
// brightest pixels (blown-out highlights) overflow to inf mid-chain; the
// final RMS norm then computes inf * rsqrt(inf) = NaN, and the u8 conversion
// turns NaN into 0 -- a whole latent cell's 16x16 patch comes out BLACK.
// Keeping just this residual in f32 removes the range problem outright (the
// reference runs the head in fp32/bf16, both of which have the exponent room);
// every other tensor here stays f16, and no GEMM touches the f32 buffer.
//
// Widen [n] f16 -> f32: seeds the residual from the input projection.
//   0:in[n] 1:out[n] 2:H 3:M.  grid (M), one row per thread.
kernel void widen_rows_f16_to_f32(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device float*           out [[buffer(1)]],
    constant int&      H [[buffer(2)]],
    constant uint&     M [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= M) { return; }
  const ulong o = (ulong)gid * (uint)H;
  for (int i = 0; i < H; ++i) { out[o + i] = float(in[o + i]); }
}

// Per-ROW adaLN LayerNorm, reading the row from the f32 residual:
//   out[r,i] = (LN(x[r,:])[i] * w[i] + b[i]) * (1 + mod[r, H+i]) + mod[r, i]
// Shift/scale are PER ROW (every pixel gets its own, from that patch's
// latent) and the rows are short (x_dim = 32), so one thread owns a whole
// row. `mod` is the [M, 3H] chunk(shift, scale, gate) buffer; gate is used
// by the twin below.
// Two-pass variance (mean, then mean of squared deviations) rather than
// E[x^2]-E[x]^2: the residual rows are large and near-constant, exactly where
// the one-pass form cancels catastrophically.
//   0:x(f32)[M,H] 1:w[H] 2:b[H] 3:mod[M,3H] 4:out[M,H] 5:H 6:eps 7:M. grid (M).
kernel void layer_norm_mod_rows_x32_f16(
    const device float*     x   [[buffer(0)]],
    const device VPIPE_ELT* w   [[buffer(1)]],
    const device VPIPE_ELT* b   [[buffer(2)]],
    const device VPIPE_ELT* mod [[buffer(3)]],
    device VPIPE_ELT*       out [[buffer(4)]],
    constant int&      H   [[buffer(5)]],
    constant float&    eps [[buffer(6)]],
    constant uint&     M   [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= M) { return; }
  const device float* xr = x + (ulong)gid * (uint)H;
  const device VPIPE_ELT* mr = mod + (ulong)gid * (uint)(3 * H);
  device VPIPE_ELT* orow = out + (ulong)gid * (uint)H;
  float s = 0.0f;
  for (int i = 0; i < H; ++i) { s += xr[i]; }
  const float mean = s / (float)H;
  float v = 0.0f;
  for (int i = 0; i < H; ++i) {
    const float d = xr[i] - mean;
    v += d * d;
  }
  const float inv = rsqrt(v / (float)H + eps);
  for (int i = 0; i < H; ++i) {
    const float ln = (xr[i] - mean) * inv * (float)w[i] + (float)b[i];
    orow[i] = VPIPE_ELT(ln * (1.0f + (float)mr[H + i]) + (float)mr[i]);
  }
}

// Per-ROW gated residual twin, accumulating into the f32 residual:
//   x[r,i] += mod[r, 2H+i] * sub[r,i]
//   0:x(f32)[M,H] (inout) 1:mod[M,3H] 2:sub[M,H] 3:H 4:M.  grid (M).
kernel void gated_residual_rows_x32_f16(
    device float*           x   [[buffer(0)]],
    const device VPIPE_ELT* mod [[buffer(1)]],
    const device VPIPE_ELT* sub [[buffer(2)]],
    constant int&      H [[buffer(3)]],
    constant uint&     M [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= M) { return; }
  const ulong xo = (ulong)gid * (uint)H;
  const device VPIPE_ELT* mr = mod + (ulong)gid * (uint)(3 * H) + (uint)(2 * H);
  for (int i = 0; i < H; ++i) {
    x[xo + i] = x[xo + i] + float(mr[i]) * float(sub[xo + i]);
  }
}

// rms_norm_fast_f16 reading the f32 residual and emitting f16:
//   out[r,i] = x[r,i] * rsqrt(mean(x[r,:]^2) + eps) * w[i]
//   0:x(f32)[M,H] 1:w[H] 2:out[M,H] 3:H 4:eps 5:M.  grid (M).
kernel void rms_norm_rows_x32_f16(
    const device float*     x   [[buffer(0)]],
    const device VPIPE_ELT* w   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      H   [[buffer(3)]],
    constant float&    eps [[buffer(4)]],
    constant uint&     M   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= M) { return; }
  const device float* xr = x + (ulong)gid * (uint)H;
  device VPIPE_ELT* orow = out + (ulong)gid * (uint)H;
  float acc = 0.0f;
  for (int i = 0; i < H; ++i) { acc += xr[i] * xr[i]; }
  const float inv = rsqrt(acc / (float)H + eps);
  for (int i = 0; i < H; ++i) {
    orow[i] = VPIPE_ELT(xr[i] * inv * (float)w[i]);
  }
}

// Row scatter: out[pos[r], :] = src[r, :] for r in 0..R-1. Overlays the
// multimodal soft-token rows into the main embedding stream (the metal
// Gemma owns_kv multimodal splice).
//   0:out[*,H] 1:src[R,H] 2:pos[R] (int32) 3:H 4:R
kernel void row_scatter_f16(
    device VPIPE_ELT*       out [[buffer(0)]],
    const device VPIPE_ELT* src [[buffer(1)]],
    const device int*       pos [[buffer(2)]],
    constant int&      H   [[buffer(3)]],
    constant int&      R   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(R * H)) { return; }
  const int r = (int)gid / H;
  const int d = (int)gid % H;
  out[(uint)pos[r] * H + d] = src[gid];
}

// Extract a width-W sub-block (starting at column `off`) from each row
// of a [H, S] tensor into a contiguous [H, W] tensor:
//   out[h, w] = in[h*S + off + w]
// Qwen3.5 splits the double-width q_proj output [n_heads, 2*head_dim]
// into q [n_heads, head_dim] (off=0) and the gate (off=head_dim).
//
// block>0 enables a BLOCK-strided source: the H rows are grouped into
// blocks of `block` rows whose group base advances by `gstride` instead
// of block*S. This lets the [n_heads, 2*head_dim]-per-token q live inside
// a WIDER fused q|k|v buffer (gstride = fused row width) and still be
// sliced with no copy: in_off = (h/block)*gstride + (h%block)*S + off + w.
//   0:in 1:out 2:H 3:S 4:W 5:off 6:block 7:gstride
kernel void head_slice_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H   [[buffer(2)]],
    constant int&      S   [[buffer(3)]],
    constant int&      W   [[buffer(4)]],
    constant int&      off [[buffer(5)]],
    constant int&      block   [[buffer(6)]],
    constant int&      gstride [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  const uint total = (uint)H * (uint)W;
  if (gid >= total) { return; }
  const uint w = gid % (uint)W;
  const uint h = gid / (uint)W;
  uint in_off;
  if (block > 0) {
    in_off = (h / (uint)block) * (uint)gstride + (h % (uint)block) * (uint)S
             + (uint)off + w;
  } else {
    in_off = h * (uint)S + (uint)off + w;
  }
  out[gid] = in[in_off];
}

// Row-wise column concat: dst[r, 0:wa] = a[r, 0:wa]; dst[r, wa:wa+wb] = b[r,
// 0:wb]. One thread per dst element -- the GPU twin of a host-memcpy concat so
// the [att | mlp] assembly stays in the SAME command stream as its producers +
// consumer (no CPU round-trip / intermediate commit().wait()).
//   0:a[rows,wa] 1:b[rows,wb] 2:dst[rows,wa+wb] 3:rows 4:wa 5:wb
kernel void concat_cols_f16(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       dst [[buffer(2)]],
    constant int&      rows [[buffer(3)]],
    constant int&      wa   [[buffer(4)]],
    constant int&      wb   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  const uint W = (uint)(wa + wb);
  const uint total = (uint)rows * W;
  if (gid >= total) { return; }
  const uint c = gid % W;
  const uint r = gid / W;
  if (c < (uint)wa) {
    dst[gid] = a[r * (uint)wa + c];
  } else {
    dst[gid] = b[r * (uint)wb + (c - (uint)wa)];
  }
}

// out[i] = a[i] * sigmoid(g[i]). Qwen3.5 full-attention output gate:
// post_attn = attn * sigmoid(gate).   0:a 1:g 2:out 3:n
kernel void mul_sigmoid_f16(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* g   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float gv = float(g[gid]);
  out[gid] = VPIPE_ELT(float(a[gid]) * (1.0f / (1.0f + metal::exp(-gv))));
}

// out[t, h] = table[ids[t], h].   0:ids(int32) 1:table 2:out 3:H
// grid (H, T, 1).
kernel void embed_gather_f16(
    const device int*  ids   [[buffer(0)]],
    const device VPIPE_ELT* table [[buffer(1)]],
    device VPIPE_ELT*       out   [[buffer(2)]],
    constant int&      H     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)H) { return; }
  const int id = ids[gid.y];
  out[(uint)gid.y * H + gid.x] = table[(uint)id * H + gid.x];
}

// Dequantizing embedding gather: out[t,h] = dequant(table[ids[t], h])
// for a 4-bit affine table (weight U32 [vocab, H/8], scales/biases
// F16 [vocab, H/64], group 64). Avoids materializing the full dense
// table. 0:ids(int32) 1:w(uint32) 2:scales 3:biases 4:out 5:H.
// grid (H, T, 1).
kernel void dequant_embed_gather_f16(
    const device int*      ids    [[buffer(0)]],
    const device uint32_t* w      [[buffer(1)]],
    const device VPIPE_ELT*     scales [[buffer(2)]],
    const device VPIPE_ELT*     biases [[buffer(3)]],
    device VPIPE_ELT*           out    [[buffer(4)]],
    constant int&          H      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int h = (int)gid.x;
  if (h >= H) { return; }
  const int t = (int)gid.y;
  const int id = ids[t];
  const int H_w = H / 8;
  const int H_g = H / 64;
  const uint32_t packed = w[(uint)id * H_w + h / 8];
  const int nib = (packed >> (4 * (h % 8))) & 0xf;
  const float s = float(scales[(uint)id * H_g + h / 64]);
  const float b = float(biases[(uint)id * H_g + h / 64]);
  out[(uint)t * H + h] = VPIPE_ELT(s * float(nib) + b);
}

// 8-bit variant: weight U32 [vocab, H/4] (4 bytes per word, one byte per
// value), scales/biases [vocab, H/64] group 64. Same affine dequant as
// the 4-bit gather, byte-extracted instead of nibble. Used by the 8-bit
// checkpoints (e.g. Qwen3-ASR).
kernel void dequant_embed_gather_w8_f16(
    const device int*      ids    [[buffer(0)]],
    const device uint32_t* w      [[buffer(1)]],
    const device VPIPE_ELT*     scales [[buffer(2)]],
    const device VPIPE_ELT*     biases [[buffer(3)]],
    device VPIPE_ELT*           out    [[buffer(4)]],
    constant int&          H      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int h = (int)gid.x;
  if (h >= H) { return; }
  const int t = (int)gid.y;
  const int id = ids[t];
  const int H_w = H / 4;
  const int H_g = H / 64;
  const uint32_t packed = w[(uint)id * H_w + h / 4];
  const uint byteval = (packed >> (8 * (h % 4))) & 0xff;
  const float s = float(scales[(uint)id * H_g + h / 64]);
  const float b = float(biases[(uint)id * H_g + h / 64]);
  out[(uint)t * H + h] = VPIPE_ELT(s * float(byteval) + b);
}

// group_size=32 twin of dequant_embed_gather_w8_f16 (8-bit affine embed
// gather, GGUF q8 group 32): scales/biases stride per row is H/32 (the
// only change vs the g64 kernel). Weight U32 [vocab, H/4], one byte per
// value; scales/biases [vocab, H/32]. 0:ids 1:w 2:scales 3:biases 4:out 5:H.
kernel void dequant_embed_gather_w8_g32_f16(
    const device int*      ids    [[buffer(0)]],
    const device uint32_t* w      [[buffer(1)]],
    const device VPIPE_ELT*     scales [[buffer(2)]],
    const device VPIPE_ELT*     biases [[buffer(3)]],
    device VPIPE_ELT*           out    [[buffer(4)]],
    constant int&          H      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int h = (int)gid.x;
  if (h >= H) { return; }
  const int t = (int)gid.y;
  const int id = ids[t];
  const int H_w = H / 4;
  const int H_g = H / 32;
  const uint32_t packed = w[(uint)id * H_w + h / 4];
  const uint byteval = (packed >> (8 * (h % 4))) & 0xff;
  const float s = float(scales[(uint)id * H_g + h / 32]);
  const float b = float(biases[(uint)id * H_g + h / 32]);
  out[(uint)t * H + h] = VPIPE_ELT(s * float(byteval) + b);
}

// ---- Q6_K (llama.cpp k-quant) native unpack ----------------------------
// A Q6_K super-block is 256 weights in 210 bytes: ql[128] (low 4 bits),
// qh[64] (high 2 bits), int8 scales[16], f16 d. Dequant of weight `pos`
// (0..255):  d * scales[sci] * (q - 32). Layout mirrors dequant_row_q6_K
// (see gguf-file.cc kQ6_K) exactly so this is LOSSLESS -- no affine requant.
// Storing the embed/lm_head as raw Q6_K (6.5625 bits/wt) instead of an 8-bit
// affine requant saves ~25% of the (huge, tied) token table and is exact.
inline float q6k_value(const device uchar* sb, int pos) {
  const device uchar* ql = sb;
  const device uchar* qh = sb + 128;
  const device char*  sc = (const device char*)(sb + 192);
  const half d = *(const device half*)(sb + 208);
  const int hf = pos >> 7;            // half (0/1): positions 0..127 / 128..255
  const int p  = pos & 127;
  const int which = p >> 5;           // 0->q1 1->q2 2->q3 3->q4
  const int l = p & 31;
  const int is = l >> 4;
  const int qlo = hf * 64, qho = hf * 32, sco = hf * 8;
  const int hi = qh[qho + l];
  int q, sci;
  if (which == 0)      { q = (ql[qlo + l]      & 0x0F) | (((hi >> 0) & 3) << 4);
                         sci = sco + is + 0; }
  else if (which == 1) { q = (ql[qlo + l + 32] & 0x0F) | (((hi >> 2) & 3) << 4);
                         sci = sco + is + 2; }
  else if (which == 2) { q = (ql[qlo + l]      >> 4 ) | (((hi >> 4) & 3) << 4);
                         sci = sco + is + 4; }
  else                 { q = (ql[qlo + l + 32] >> 4 ) | (((hi >> 6) & 3) << 4);
                         sci = sco + is + 6; }
  return float(d) * float(sc[sci]) * float(q - 32);
}

// Dequantize N Q6_K weights -> f16 (row-major, contiguous super-blocks).
// 0:src(raw Q6_K bytes) 1:out 2:N.  grid (N).
kernel void dequant_q6k_f16(
    const device uchar* src [[buffer(0)]],
    device VPIPE_ELT*   out [[buffer(1)]],
    constant int&       N   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)N) { return; }
  const int i = (int)gid;
  const device uchar* sb = src + (i / 256) * 210;
  out[i] = VPIPE_ELT(q6k_value(sb, i & 255));
}

// Embed gather from a raw Q6_K table: out[t,:] = dequant(table[ids[t], :]).
// Each row is (H/256) super-blocks = (H/256)*210 bytes. 0:ids 1:table(Q6_K)
// 2:out 3:H.  grid (H, n_tokens).
kernel void embed_gather_q6k_f16(
    const device int*   ids   [[buffer(0)]],
    const device uchar* table [[buffer(1)]],
    device VPIPE_ELT*   out   [[buffer(2)]],
    constant int&       H     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int h = (int)gid.x;
  if (h >= H) { return; }
  const int t = (int)gid.y;
  const int id = ids[t];
  const int row_bytes = (H / 256) * 210;
  const device uchar* sb = table + (uint)id * row_bytes + (h / 256) * 210;
  out[(uint)t * H + h] = VPIPE_ELT(q6k_value(sb, h & 255));
}

// lm_head GEMV over a raw Q6_K weight [N, H]:  y[o] = sum_h x[h]*dequant(W[o,h]).
// One simdgroup computes RPS=4 output rows; the 32 lanes split H and simd_sum.
// Q6_K is read directly (no dequant-to-DRAM): ~6.56 bits/wt vs the 8-bit affine
// requant, so this is both smaller and lossless. 0:w(Q6_K) 1:x 2:y 3:H 4:N.
// grid (32, ceil(N/8)*2, 1); threadgroup (32, 2, 1). Requires H % 256 == 0.
//
// Vectorized: each lane owns the contiguous 8-weight block [l*8, l*8+8) of
// every 256-wide super-block. Such a block never crosses a 32-group (`which`)
// or 16-group (`is`) boundary, so d, the int8 scale, and the whole `which`
// branch HOIST out of the inner 8-weight loop (computed once per block, the
// geometry once per lane), x is read 8-wide and reused across the RPS rows,
// and the inner loop is pure FMA. 32 lanes * 8 = 256 still tile a super-block
// so simd_sum over the lanes yields the full row dot -- ~exactly the original
// math, far fewer instructions (the kernel was instruction-bound, not BW).
kernel void qmv_q6k_f16(
    const device uchar*     w [[buffer(0)]],
    const device VPIPE_ELT* x [[buffer(1)]],
    device VPIPE_ELT*       y [[buffer(2)]],
    constant int&           H [[buffer(3)]],
    constant int&           N [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int RPS = 4, NSG = 2;
  const int nsb = H / 256;                 // super-blocks per row
  const int row_bytes = nsb * 210;
  const int out_row = (int)tid.y * (NSG * RPS) + (int)simd_gid * RPS;
  // Per-lane 8-block geometry (constant across super-blocks AND rows).
  const int l    = (int)simd_lid;          // 0..31 -> 8-block index
  const int p0   = l * 8;                  // position base 0..248
  const int hf   = p0 >> 7;                // half 0/1
  const int pm   = p0 & 127;
  const int which= pm >> 5;                // 0..3
  const int l32  = pm & 31;                // 0,8,16,24 (block start in group)
  const int is   = l32 >> 4;               // 0/1
  const int sci  = hf * 8 + is + 2 * which;        // int8-scale index
  const int qlo  = hf * 64 + l32 + ((which & 1) ? 32 : 0);  // ql byte base
  const int qho  = 128 + hf * 32 + l32;            // qh byte base (sb-relative)
  const bool hinib = which >= 2;           // high nibble of the ql byte
  const int hsh  = 2 * which;              // qh bit shift

  float acc[RPS] = {0};
  for (int s = 0; s < nsb; ++s) {
    const int xbase = s * 256 + p0;
    float xv[8];
    for (int j = 0; j < 8; ++j) { xv[j] = float(x[xbase + j]); }
    const int sb_off = s * 210;
    for (int r = 0; r < RPS; ++r) {
      const device uchar* sb = w + (uint)(out_row + r) * row_bytes + sb_off;
      const float scale = float(*(const device half*)(sb + 208))
                          * float(((const device char*)(sb + 192))[sci]);
      const device uchar* qlp = sb + qlo;
      const device uchar* qhp = sb + qho;
      float a = 0.0f;
      for (int j = 0; j < 8; ++j) {
        const uint lo = hinib ? (uint)(qlp[j] >> 4) : (uint)(qlp[j] & 0x0F);
        const uint hi = ((uint)qhp[j] >> hsh) & 3u;
        a += xv[j] * float((int)(lo | (hi << 4)) - 32);
      }
      acc[r] += scale * a;
    }
  }
  for (int r = 0; r < RPS; ++r) {
    const float v = simd_sum(acc[r]);
    if (simd_lid == 0 && out_row + r < N) {
      y[out_row + r] = VPIPE_ELT(v);
    }
  }
}

// qmv_q6k_v2_f16 -- llama.cpp-style Q6_K lm_head GEMV. qmv_q6k_f16 has each lane
// take a SINGLE nibble per ql byte (its `which` group), so each ql byte is
// re-read by the which+2 lane and each qh byte by 4 lanes -- ~2.4x the byte
// loads, which made it L2/load-bound at ~59 GB/s (well under the ~100 GB/s
// ceiling) despite reading the same DRAM. Here each lane reads every q1/q2/qh
// byte ONCE and extracts ALL its nibbles + 2-bit high fields in-thread (16
// weights/lane/super-block from 12 byte-reads, each byte fully consumed), the
// trick from llama.cpp kernel_mul_mv_q6_K. Same [N,H] row-major raw-Q6_K weight
// + f16 x/y contract and the SAME (32, ceil(N/8)*2, 1)/(32, 2, 1) dispatch.
// NOT bit-identical to qmv_q6k_f16 (different fp grouping) but token-exact.
// Requires H % 256 == 0.
template <int RPS, int NSG>
static inline void qmv_q6k_v2_impl(
    const device uchar*     w,
    const device VPIPE_ELT* x,
    device VPIPE_ELT*       y,
    int H, int N, uint3 tid, uint simd_gid, uint simd_lid)
{
  const int nsb = H / 256;                  // super-blocks per row
  const int row_bytes = nsb * 210;
  const int out_row = (int)tid.y * (NSG * RPS) + (int)simd_gid * RPS;
  // 32 lanes = 16 `t` positions x 2 `ix` (even/odd super-block split). Each
  // `t` owns 16 weights of a super-block (ip half x il), read from 4 q1 + 4 q2
  // + 4 qh bytes, every byte fully used (both nibbles / all four 2-bit fields).
  const int lane = (int)simd_lid;
  const int t   = lane >> 1;                 // 0..15
  const int ix  = lane & 1;                  // even/odd super-block
  const int ip  = t >> 3;                    // 0/1 (which 128-half)
  const int il  = t & 7;                     // 0..7
  const int l0  = 4 * il;
  const int is  = 8 * ip + (l0 >> 4);        // int8-scale base index
  const int y_off = 128 * ip + l0;
  const int qlo = 64 * ip + l0;              // q1 base; q2 = q1 + 32
  const int qho = 32 * ip + l0;              // qh base

  float acc[RPS] = {0};
  for (int s = ix; s < nsb; s += 2) {
    float yl[16];
    const int xb = s * 256 + y_off;
    for (int l = 0; l < 4; ++l) {
      yl[4 * l + 0] = float(x[xb + l +  0]);
      yl[4 * l + 1] = float(x[xb + l + 32]);
      yl[4 * l + 2] = float(x[xb + l + 64]);
      yl[4 * l + 3] = float(x[xb + l + 96]);
    }
    for (int r = 0; r < RPS; ++r) {
      const device uchar* sb = w + (uint)(out_row + r) * row_bytes + s * 210;
      const device uchar* q1 = sb + qlo;
      const device uchar* q2 = q1 + 32;
      const device uchar* qh = sb + 128 + qho;
      const device char*  sc = (const device char*)(sb + 192) + is;
      const float d = float(*(const device half*)(sb + 208));
      float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int l = 0; l < 4; ++l) {
        const uint h = qh[l];
        sums[0] += yl[4*l+0] *
            float((int)((q1[l] & 0x0F) | ((h & 0x03) << 4)) - 32);
        sums[1] += yl[4*l+1] *
            float((int)((q2[l] & 0x0F) | ((h & 0x0C) << 2)) - 32);
        sums[2] += yl[4*l+2] *
            float((int)((q1[l] >> 4)   | ((h & 0x30) << 0)) - 32);
        sums[3] += yl[4*l+3] *
            float((int)((q2[l] >> 4)   | ((h & 0xC0) >> 2)) - 32);
      }
      acc[r] += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                   + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    }
  }
  for (int r = 0; r < RPS; ++r) {
    const float v = simd_sum(acc[r]);
    if (simd_lid == 0 && out_row + r < N) {
      y[out_row + r] = VPIPE_ELT(v);
    }
  }
}

// RPS = rows/simdgroup, NSG = simdgroups/threadgroup (= threadgroup.y at
// dispatch). Default = <4,2>; the _rNnM variants exist to sweep the tuning
// (llama.cpp uses nr0=2 + a device-tuned nsg via function constant).
#define VPIPE_Q6K_V2(NAME, RPS, NSG)                                          \
kernel void qmv_q6k_v2##NAME##_f16(                                           \
    const device uchar*     w [[buffer(0)]],                                  \
    const device VPIPE_ELT* x [[buffer(1)]],                                  \
    device VPIPE_ELT*       y [[buffer(2)]],                                  \
    constant int&           H [[buffer(3)]],                                  \
    constant int&           N [[buffer(4)]],                                  \
    uint3 tid      [[threadgroup_position_in_grid]],                          \
    uint  simd_gid [[simdgroup_index_in_threadgroup]],                        \
    uint  simd_lid [[thread_index_in_simdgroup]])                            \
{                                                                            \
  qmv_q6k_v2_impl<RPS, NSG>(w, x, y, H, N, tid, simd_gid, simd_lid);          \
}
VPIPE_Q6K_V2(, 4, 2)        // qmv_q6k_v2_f16 (the production default)
VPIPE_Q6K_V2(_r2n2, 2, 2)
VPIPE_Q6K_V2(_r8n2, 8, 2)
VPIPE_Q6K_V2(_r2n4, 2, 4)
VPIPE_Q6K_V2(_r4n4, 4, 4)
VPIPE_Q6K_V2(_r1n4, 1, 4)
VPIPE_Q6K_V2(_r1n8, 1, 8)
VPIPE_Q6K_V2(_r2n8, 2, 8)
VPIPE_Q6K_V2(_r4n8, 4, 8)

// ---- Q4_K / Q5_K (llama.cpp k-quant) native unpack ---------------------
// Both are 256-weight super-blocks of eight 32-weight sub-blocks, each with
// a 6-bit scale + 6-bit min packed into a shared 12-byte `scales` field
// (unpacked by gsmk4_, bit-identical to llama.cpp get_scale_min_k4). The
// dequant of weight `pos` (0..255) is `d*sc*q - dmin*m` (Q4_K: q is a 4-bit
// nibble; Q5_K: a 5th bit comes from qh). Layouts mirror gguf-file.cc
// kQ4_K/kQ5_K exactly, so these are LOSSLESS (no affine requant) -- the
// metal Qwen GGUF path reads the raw k-quant blocks directly.
//   Q4_K block (144 B): d(f16) dmin(f16) scales[12] qs[128]
//   Q5_K block (176 B): d(f16) dmin(f16) scales[12] qh[32] qs[128]
inline void gsmk4_(int j, const device uchar* q,
                   thread uchar& d, thread uchar& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
  }
}

inline float q4k_value(const device uchar* sb, int pos) {
  const half d    = *(const device half*)(sb + 0);
  const half dmin = *(const device half*)(sb + 2);
  const device uchar* scales = sb + 4;
  const device uchar* qs = sb + 16;
  const int chunk  = pos >> 6;                 // 0..3 (64-weight chunk)
  const int within = pos & 63;
  const int is = chunk * 2 + (within >> 5);    // sub-block 0..7
  const int l  = within & 31;
  const uint qb  = qs[chunk * 32 + l];
  const uint nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
  uchar sc, mn;
  gsmk4_(is, scales, sc, mn);
  return float(d) * float(sc) * float(nib) - float(dmin) * float(mn);
}

inline float q5k_value(const device uchar* sb, int pos) {
  const half d    = *(const device half*)(sb + 0);
  const half dmin = *(const device half*)(sb + 2);
  const device uchar* scales = sb + 4;
  const device uchar* qh = sb + 16;
  const device uchar* qs = sb + 48;
  const int chunk  = pos >> 6;
  const int within = pos & 63;
  const int is = chunk * 2 + (within >> 5);
  const int l  = within & 31;
  const uint qb  = qs[chunk * 32 + l];
  uint nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
  const int bit = 2 * chunk + ((within < 32) ? 0 : 1);  // 5th-bit position
  nib += ((uint(qh[l]) >> bit) & 1u) * 16u;
  uchar sc, mn;
  gsmk4_(is, scales, sc, mn);
  return float(d) * float(sc) * float(nib) - float(dmin) * float(mn);
}

// Dequantize N Q4_K / Q5_K weights -> f16 (row-major, contiguous blocks).
// 0:src(raw k-quant) 1:out 2:N.  grid (N).
kernel void dequant_q4k_f16(
    const device uchar* src [[buffer(0)]],
    device VPIPE_ELT*   out [[buffer(1)]],
    constant int&       N   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)N) { return; }
  const int i = (int)gid;
  out[i] = VPIPE_ELT(q4k_value(src + (i / 256) * 144, i & 255));
}

// Repack a raw Q4_K weight [N, H] into the MLX affine-4bit-g32 form the fast
// affine_qmv_w4g32 kernel consumes (decode runs ~2x faster than the native
// k-quant qmv_q4k -- the k-quant scale unpack is the bottleneck, the affine
// kernel's uint32 loads + separate scale arrays are not). LOSSLESS: a Q4_K
// 32-weight sub-block is exactly scale*q + bias with scale = d*subscale,
// bias = -dmin*submin, which is the affine dot the kernel computes.
//   wq    [N, H/8]  uint32 -- 8 nibbles/word, NATURAL order (weight h -> word
//                   h/8, nibble h%8), matching qdot's masked-bit reconstruction
//   scale [N, H/32] elt = d * subscale
//   bias  [N, H/32] elt = -dmin * submin
// One thread per (row, super-block) owns 32 words + 8 scales/biases (no races).
// 0:src(Q4_K) 1:wq 2:scale 3:bias 4:H 5:N.  grid (H/256, N, 1).
kernel void repack_q4k_to_affine_g32(
    const device uchar* src   [[buffer(0)]],
    device uint*        wq     [[buffer(1)]],
    device VPIPE_ELT*   scale  [[buffer(2)]],
    device VPIPE_ELT*   bias   [[buffer(3)]],
    constant int&       H      [[buffer(4)]],
    constant int&       N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int nsb = H / 256;
  const int s = (int)gid.x;          // super-block within the row
  const int r = (int)gid.y;          // output row
  if (r >= N || s >= nsb) { return; }
  const device uchar* sb = src + ((uint)r * nsb + s) * 144;
  const float d    = float(*(const device half*)(sb + 0));
  const float dmin = float(*(const device half*)(sb + 2));
  const int sc_base = r * (H / 32) + s * 8;
  for (int is = 0; is < 8; ++is) {
    uchar sc, m;
    gsmk4_(is, sb + 4, sc, m);
    scale[sc_base + is] = VPIPE_ELT(d * float(sc));
    bias[sc_base + is]  = VPIPE_ELT(-dmin * float(m));
  }
  const device uchar* qs = sb + 16;
  const int w_base = r * (H / 8) + s * 32;
  for (int u = 0; u < 32; ++u) {       // 32 u32 = 256 weights / super-block
    uint packed = 0;
    for (int n = 0; n < 8; ++n) {
      const int pos = u * 8 + n;       // 0..255 (natural order)
      const int chunk = pos >> 6;
      const int l = pos & 31;
      const uint qb = qs[chunk * 32 + l];
      const uint nib = ((pos & 63) < 32) ? (qb & 0x0F) : (qb >> 4);
      packed |= nib << (n * 4);
    }
    wq[w_base + u] = packed;
  }
}

// Repack a raw Q5_K weight [N, K] into MLX affine-8bit-g32 (Q5_K is 5-bit, so
// it needs the 8-bit affine kernel -- the q value 0..31 stored one byte each,
// natural order; scale = d*subscale, bias = -dmin*submin). Lossless. Consumed
// by affine_qmv_w8g32. EXPERIMENT RESULT (not wired into the model): the 8-bit
// affine reads +64% bytes (9 vs 5.5 bits/wt), which exactly cancels its higher
// bandwidth -> 0.99x vs qmv_q5k (wash) for +64% memory. NOT worth it; only
// Q4_K (same-bit-width 4-bit affine) wins. Kept for the experiment record.
//   wq    [N, K]    uint8 (raw q, natural order, 1 byte/weight)
//   scale [N, K/32] elt, bias [N, K/32] elt
// One thread per (row, super-block). 0:src(Q5_K) 1:wq 2:scale 3:bias 4:K 5:N.
kernel void repack_q5k_to_affine_g32(
    const device uchar* src   [[buffer(0)]],
    device uchar*       wq     [[buffer(1)]],
    device VPIPE_ELT*   scale  [[buffer(2)]],
    device VPIPE_ELT*   bias   [[buffer(3)]],
    constant int&       K      [[buffer(4)]],
    constant int&       N      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int nsb = K / 256;
  const int s = (int)gid.x;
  const int r = (int)gid.y;
  if (r >= N || s >= nsb) { return; }
  const device uchar* sb = src + ((uint)r * nsb + s) * 176;
  const float d    = float(*(const device half*)(sb + 0));
  const float dmin = float(*(const device half*)(sb + 2));
  const device uchar* scales = sb + 4;
  const device uchar* qh = sb + 16;
  const device uchar* qs = sb + 48;
  const int sc_base = r * (K / 32) + s * 8;
  for (int is = 0; is < 8; ++is) {
    uchar sc, m;
    gsmk4_(is, scales, sc, m);
    scale[sc_base + is] = VPIPE_ELT(d * float(sc));
    bias[sc_base + is]  = VPIPE_ELT(-dmin * float(m));
  }
  const int w_base = r * K + s * 256;
  for (int pos = 0; pos < 256; ++pos) {
    const int chunk = pos >> 6;
    const int l = pos & 31;
    const uint qb = qs[chunk * 32 + l];
    uint nib = ((pos & 63) < 32) ? (qb & 0x0F) : (qb >> 4);
    const int bit = 2 * chunk + (((pos & 63) < 32) ? 0 : 1);
    nib += ((uint(qh[l]) >> bit) & 1u) * 16u;
    wq[w_base + pos] = (uchar)nib;
  }
}

// Embed gather from a raw Q4_K table: out[t,:] = dequant(table[ids[t], :]).
// Mirrors embed_gather_q6k_f16 but for Q4_K super-blocks (144 bytes / 256
// weights, see q4k_value). Used by untied checkpoints whose token_embd is
// Q4_K (the separate Q6_K output.weight is then the lm_head). 0:ids
// 1:table(Q4_K) 2:out 3:H.  grid (H, n_tokens).
kernel void embed_gather_q4k_f16(
    const device int*   ids   [[buffer(0)]],
    const device uchar* table [[buffer(1)]],
    device VPIPE_ELT*   out   [[buffer(2)]],
    constant int&       H     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int h = (int)gid.x;
  if (h >= H) { return; }
  const int t = (int)gid.y;
  const int id = ids[t];
  const int row_bytes = (H / 256) * 144;
  const device uchar* sb = table + (uint)id * row_bytes + (h / 256) * 144;
  out[(uint)t * H + h] = VPIPE_ELT(q4k_value(sb, h & 255));
}

// Plain f16 strided copy: out[off + gid] = src[gid] for gid in [0, N).
// Folds an already-f16 sub-tensor (the Q8_0 a/b projections, dequant'd to
// f16 at load) into the fused dequant scratch for the k-quant prefill GEMM,
// alongside the q4_K/q5_K parts the dequant kernels write. 0:src 1:out
// 2:off (elements) 3:N.
kernel void copy_f16(
    const device VPIPE_ELT* src [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           off [[buffer(2)]],
    constant int&           N   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)N) { return; }
  out[off + (int)gid] = src[gid];
}

// Diagnostic-only no-op carrying a REALISTIC dispatch arg load (8 read buffers
// + 4 constants + 1 write), so VPIPE_GEMMA_DUMMY_DISP measures true per-launch
// cost (arg-binding + command-processor setup + dependent-chain bubble) rather
// than a 3-arg toy. Thread 0 reads one element from every bound buffer and the
// constants and writes their sum to out[0] -- a live read->write the compiler
// cannot elide. Binding out == b0 (same buffer) makes successive dummies a RAW
// chain that serializes like the real decode dispatch chain.
kernel void dummy_disp_f16(
    const device VPIPE_ELT* b0  [[buffer(0)]],
    const device VPIPE_ELT* b1  [[buffer(1)]],
    const device VPIPE_ELT* b2  [[buffer(2)]],
    const device VPIPE_ELT* b3  [[buffer(3)]],
    const device VPIPE_ELT* b4  [[buffer(4)]],
    const device VPIPE_ELT* b5  [[buffer(5)]],
    const device VPIPE_ELT* b6  [[buffer(6)]],
    const device VPIPE_ELT* b7  [[buffer(7)]],
    device VPIPE_ELT*       out [[buffer(8)]],
    constant int&           c0  [[buffer(9)]],
    constant int&           c1  [[buffer(10)]],
    constant int&           c2  [[buffer(11)]],
    constant int&           c3  [[buffer(12)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid != 0u) { return; }
  float s = float(b0[0]) + float(b1[0]) + float(b2[0]) + float(b3[0])
          + float(b4[0]) + float(b5[0]) + float(b6[0]) + float(b7[0])
          + float(c0 + c1 + c2 + c3);
  out[0] = VPIPE_ELT(s);
}

kernel void dequant_q5k_f16(
    const device uchar* src [[buffer(0)]],
    device VPIPE_ELT*   out [[buffer(1)]],
    constant int&       N   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)N) { return; }
  const int i = (int)gid;
  out[i] = VPIPE_ELT(q5k_value(src + (i / 256) * 176, i & 255));
}

// GEMV over a raw Q4_K / Q5_K weight [N, H]:
//   y[o] = sum_h x[h] * dequant(W[o,h]).
// One simdgroup per output row; the 32 lanes split H and simd_sum.
// 0:w(k-quant) 1:x 2:y 3:H 4:N.  grid (32, N, 1); threadgroup (32, 1, 1).
// Requires H % 256 == 0.
// Each lane owns 4 contiguous qs bytes of every 256-super-block and uses
// BOTH nibbles -> 8 weights from 4 byte-reads (the one-nibble form had two
// lanes read each byte = 2x the loads). The 4 low nibbles are sub-block
// chunk*2, the 4 high nibbles sub-block chunk*2+1 (so two gsmk4_ scales,
// both hoisted). 32 lanes x 4 bytes tile the 128-byte qs exactly; simd_sum
// gives the row dot. NSG=2 rows per threadgroup for occupancy. Bit-identical
// to q4k_value. grid (32, ceil(N/2)*2, 1); threadgroup (32, 2, 1).
kernel void qmv_q4k_f16(
    const device uchar*     w [[buffer(0)]],
    const device VPIPE_ELT* x [[buffer(1)]],
    device VPIPE_ELT*       y [[buffer(2)]],
    constant int&           H [[buffer(3)]],
    constant int&           N [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int NSG = 2;
  const int row = (int)tid.y * NSG + (int)simd_gid;
  if (row >= N) { return; }
  const int nsb = H / 256;
  const device uchar* rowp = w + (uint)row * (nsb * 144);
  const int lane  = (int)simd_lid;
  const int bi    = lane * 4;                 // first qs byte (0..124)
  const int chunk = bi >> 5;                  // 0..3
  const int l     = bi & 31;                  // 0..28
  const int is_lo = chunk * 2;                // low-nibble sub-block
  const int is_hi = chunk * 2 + 1;            // high-nibble sub-block
  const int xlo   = chunk * 64 + l;           // x base for low nibbles
  const int xhi   = xlo + 32;                 // x base for high nibbles
  float acc = 0.0f;
  for (int s = 0; s < nsb; ++s) {
    const device uchar* sb = rowp + s * 144;
    const float d    = float(*(const device half*)(sb + 0));
    const float dmin = float(*(const device half*)(sb + 2));
    uchar slo, mlo, shi, mhi;
    gsmk4_(is_lo, sb + 4, slo, mlo);
    gsmk4_(is_hi, sb + 4, shi, mhi);
    const float dlo = d * float(slo), blo = dmin * float(mlo);
    const float dhi = d * float(shi), bhi = dmin * float(mhi);
    const device uchar* qs = sb + 16 + chunk * 32 + l;
    const int xb = s * 256;
    for (int k = 0; k < 4; ++k) {
      const uint qb = qs[k];
      acc += float(x[xb + xlo + k]) * (dlo * float(qb & 0x0F) - blo);
      acc += float(x[xb + xhi + k]) * (dhi * float(qb >> 4)   - bhi);
    }
  }
  const float v = simd_sum(acc);
  if (lane == 0) { y[row] = VPIPE_ELT(v); }
}

// Q5_K: ported from llama.cpp kernel_mul_mv_q5_K_f32 (the native q5k qmv was
// the slow single-row scalar form). 1 row/simdgroup (NSG=2), but the 32 lanes
// split FOUR ways over super-blocks (ix = lane%4 -> 4 blocks in flight = more
// memory-level parallelism), with the x slice loaded once, the affine bias
// FACTORED (d*sc*Sum(x*q) - dmin*m*Sum(x)), and the power-of-16 nibble trick
// (high nibble kept in place, scale divided by 16). Token-exact (different fp
// grouping). grid (32, ceil(N/2)*2, 1); threadgroup (32, 2, 1).
kernel void qmv_q5k_f16(
    const device uchar*     w [[buffer(0)]],
    const device VPIPE_ELT* x [[buffer(1)]],
    device VPIPE_ELT*       y [[buffer(2)]],
    constant int&           H [[buffer(3)]],
    constant int&           N [[buffer(4)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  constexpr int NSG = 2;
  const int row = (int)tid.y * NSG + (int)simd_gid;
  if (row >= N) { return; }
  const int nsb = H / 256;
  const device uchar* rowp = w + (uint)row * (nsb * 176);
  const int lane = (int)simd_lid;
  const int tt = lane >> 2;            // 0..7
  const int ix = lane & 3;             // 0..3 (4-way super-block split)
  const int iq = tt >> 2;              // 0..1
  const int ir = tt & 3;               // 0..3
  const int l0 = 8 * ir;
  const int q_offset = 32 * iq + l0;
  const int y_offset = 64 * iq + l0;
  const uint hm1 = 1u << (2 * iq);
  const uint hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
  constexpr ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  float sumf = 0.0f;
  for (int i = ix; i < nsb; i += 4) {
    const device uchar* sb = rowp + i * 176;
    const device half*   dh = (const device half*)(sb + 0);          // d, dmin
    const device ushort* a  = (const device ushort*)(sb + 4) + iq;   // scales
    const device uchar*  qh = sb + 16 + l0;
    const device uchar*  q1 = sb + 48 + q_offset;
    const device uchar*  q2 = q1 + 64;
    const int xb = i * 256 + y_offset;
    float yl[16], yh[16];
    float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int l = 0; l < 8; ++l) {
      yl[l + 0] = float(x[xb + l +   0]); sumy[0] += yl[l + 0];
      yl[l + 8] = float(x[xb + l +  32]); sumy[1] += yl[l + 8];
      yh[l + 0] = float(x[xb + l + 128]); sumy[2] += yh[l + 0];
      yh[l + 8] = float(x[xb + l + 160]); sumy[3] += yh[l + 8];
    }
    ushort sc16[4];
    sc16[0] = a[0] & kmask1;
    sc16[1] = a[2] & kmask1;
    sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
    sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
    const thread uchar* sc8 = (const thread uchar*)sc16;
    float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int l = 0; l < 8; ++l) {
      const uint h = qh[l];
      acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
      acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
      acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
      acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
      acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
      acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
      acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
      acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
    }
    const float d = float(dh[0]), dmin = float(dh[1]);
    sumf += d * (float(sc8[0]) * (acc1[0]         + 16.0f * acc2[0]) +
                 float(sc8[1]) * (acc1[1] / 16.0f + 16.0f * acc2[1]) +
                 float(sc8[4]) * (acc1[2]         + 16.0f * acc2[2]) +
                 float(sc8[5]) * (acc1[3] / 16.0f + 16.0f * acc2[3])) -
          dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                  sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
  }
  const float v = simd_sum(sumf);
  if (lane == 0) { y[row] = VPIPE_ELT(v); }
}

// ---- Batched k-quant GEMV (weight read ONCE across MAXM rows) ----------
// The MTP verify's weight-bound matmul on the GGUF path. y[base+m, yoff+row] =
// dequant(W[row]) . x[base+m] for the MAXM-row tile starting at base = tid.z*
// MAXM. MAXM is a COMPILE-TIME TEMPLATE arg: 2 (qmv_q*k_batch_f16) is the
// depth-1 draft window (acc[] unrolls into registers; the per-row k-quant
// unpack/FMA hides behind the once-read weight load = bandwidth-bound, so a
// 2-token verify costs ~1 qmv step); 4 (qmv_q*k_batch4_f16) serves the depth-2
// MTP verify -- n=3..4 rows in ONE tile, reading the weight ONCE vs the MAXM=2
// form's 2 grid.z tiles (the depth-2 cliff, mirroring the affine batch4). m>MAXM
// tiles along grid.z (ceil(m/MAXM) tiles); pad rows alias the last valid row (no
// OOB) and are never written. Output rows are `ystride` apart at base `yoff`
// (ystride==N writes a contiguous [m,N]; a wider stride writes a column slice
// of a fused buffer). Per-row arithmetic == the single-row qmv kernels
// (acc += lo; acc += hi order) -> token-exact. grid {32, ceil(N/2)*2,
// ceil(m/MAXM)}, tg {32, 2, 1}. x is [m_total, K] tightly packed (xstride==K==H).

template <int MAXM>
inline void qmv_q4k_batch_impl(
    const device uchar*     w,
    const device VPIPE_ELT* x,
    device VPIPE_ELT*       y,
    const constant int&     H,
    const constant int&     N,
    const constant int&     m_total,
    const constant int&     ystride,
    const constant int&     yoff,
    uint3 tid, uint simd_gid, uint simd_lid)
{
  constexpr int NSG = 2;
  const int row = (int)tid.y * NSG + (int)simd_gid;
  if (row >= N) { return; }
  const int base = (int)tid.z * MAXM;
  const int m_rows = metal::min(MAXM, m_total - base);
  if (m_rows <= 0) { return; }
  const int nsb = H / 256;
  const device uchar* rowp = w + (uint)row * (nsb * 144);
  const int lane  = (int)simd_lid;
  const int bi    = lane * 4;
  const int chunk = bi >> 5;
  const int l     = bi & 31;
  const int is_lo = chunk * 2;
  const int is_hi = chunk * 2 + 1;
  const int xlo   = chunk * 64 + l;
  const int xhi   = xlo + 32;
  const device VPIPE_ELT* xptr[MAXM];
  for (int m = 0; m < MAXM; ++m) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);
    xptr[m] = x + (uint)(base + mm) * H;
  }
  float acc[MAXM];
  for (int m = 0; m < MAXM; ++m) { acc[m] = 0.0f; }
  for (int s = 0; s < nsb; ++s) {
    const device uchar* sb = rowp + s * 144;
    const float d    = float(*(const device half*)(sb + 0));
    const float dmin = float(*(const device half*)(sb + 2));
    uchar slo, mlo, shi, mhi;
    gsmk4_(is_lo, sb + 4, slo, mlo);
    gsmk4_(is_hi, sb + 4, shi, mhi);
    const float dlo = d * float(slo), blo = dmin * float(mlo);
    const float dhi = d * float(shi), bhi = dmin * float(mhi);
    const device uchar* qs = sb + 16 + chunk * 32 + l;
    const int xb = s * 256;
    for (int k = 0; k < 4; ++k) {
      const uint qb = qs[k];
      const float wlo = dlo * float(qb & 0x0F) - blo;
      const float whi = dhi * float(qb >> 4)   - bhi;
      for (int m = 0; m < MAXM; ++m) {
        acc[m] += float(xptr[m][xb + xlo + k]) * wlo;
        acc[m] += float(xptr[m][xb + xhi + k]) * whi;
      }
    }
  }
  for (int m = 0; m < MAXM; ++m) {
    const float v = simd_sum(acc[m]);
    if (lane == 0 && m < m_rows) {
      y[(uint)(base + m) * ystride + yoff + row] = VPIPE_ELT(v);
    }
  }
}

template <int MAXM>
inline void qmv_q5k_batch_impl(
    const device uchar*     w,
    const device VPIPE_ELT* x,
    device VPIPE_ELT*       y,
    const constant int&     H,
    const constant int&     N,
    const constant int&     m_total,
    const constant int&     ystride,
    const constant int&     yoff,
    uint3 tid, uint simd_gid, uint simd_lid)
{
  constexpr int NSG = 2;
  const int row = (int)tid.y * NSG + (int)simd_gid;
  if (row >= N) { return; }
  const int base = (int)tid.z * MAXM;
  const int m_rows = metal::min(MAXM, m_total - base);
  if (m_rows <= 0) { return; }
  const int nsb = H / 256;
  const device uchar* rowp = w + (uint)row * (nsb * 176);
  const int lane  = (int)simd_lid;
  const int bi    = lane * 4;
  const int chunk = bi >> 5;
  const int l     = bi & 31;
  const int is_lo = chunk * 2, is_hi = chunk * 2 + 1;
  const int blo_bit = 2 * chunk, bhi_bit = 2 * chunk + 1;
  const int xlo   = chunk * 64 + l;
  const int xhi   = xlo + 32;
  const device VPIPE_ELT* xptr[MAXM];
  for (int m = 0; m < MAXM; ++m) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);
    xptr[m] = x + (uint)(base + mm) * H;
  }
  float acc[MAXM];
  for (int m = 0; m < MAXM; ++m) { acc[m] = 0.0f; }
  for (int s = 0; s < nsb; ++s) {
    const device uchar* sb = rowp + s * 176;
    const float d    = float(*(const device half*)(sb + 0));
    const float dmin = float(*(const device half*)(sb + 2));
    uchar slo, mlo, shi, mhi;
    gsmk4_(is_lo, sb + 4, slo, mlo);
    gsmk4_(is_hi, sb + 4, shi, mhi);
    const float dlo = d * float(slo), blo = dmin * float(mlo);
    const float dhi = d * float(shi), bhi = dmin * float(mhi);
    const device uchar* qh = sb + 16 + l;
    const device uchar* qs = sb + 48 + chunk * 32 + l;
    const int xb = s * 256;
    for (int k = 0; k < 4; ++k) {
      const uint qb  = qs[k];
      const uint qhb = qh[k];
      const uint nlo = (qb & 0x0F) + (((qhb >> blo_bit) & 1u) * 16u);
      const uint nhi = (qb >> 4)   + (((qhb >> bhi_bit) & 1u) * 16u);
      const float wlo = dlo * float(nlo) - blo;
      const float whi = dhi * float(nhi) - bhi;
      for (int m = 0; m < MAXM; ++m) {
        acc[m] += float(xptr[m][xb + xlo + k]) * wlo;
        acc[m] += float(xptr[m][xb + xhi + k]) * whi;
      }
    }
  }
  for (int m = 0; m < MAXM; ++m) {
    const float v = simd_sum(acc[m]);
    if (lane == 0 && m < m_rows) {
      y[(uint)(base + m) * ystride + yoff + row] = VPIPE_ELT(v);
    }
  }
}

// Q6_K batched: the qmv_q6k_v2 algorithm (llama.cpp grouping) per row, batched
// over MAXM inputs -- bit-identical to the serial decode lm_head -> token-exact.
template <int MAXM>
inline void qmv_q6k_batch_impl(
    const device uchar*     w,
    const device VPIPE_ELT* x,
    device VPIPE_ELT*       y,
    const constant int&     H,
    const constant int&     N,
    const constant int&     m_total,
    const constant int&     ystride,
    const constant int&     yoff,
    uint3 tid, uint simd_gid, uint simd_lid)
{
  constexpr int NSG = 2;
  const int nsb = H / 256;
  const int row_bytes = nsb * 210;
  const int out_row = (int)tid.y * NSG + (int)simd_gid;
  if (out_row >= N) { return; }
  const int base = (int)tid.z * MAXM;
  const int m_rows = metal::min(MAXM, m_total - base);
  if (m_rows <= 0) { return; }
  const int lane = (int)simd_lid;
  const int t   = lane >> 1;
  const int ix  = lane & 1;
  const int ip  = t >> 3;
  const int il  = t & 7;
  const int l0  = 4 * il;
  const int is  = 8 * ip + (l0 >> 4);
  const int y_off = 128 * ip + l0;
  const int qlo = 64 * ip + l0;
  const int qho = 32 * ip + l0;
  const device VPIPE_ELT* xptr[MAXM];
  for (int m = 0; m < MAXM; ++m) {
    const int mm = (m < m_rows) ? m : (m_rows - 1);
    xptr[m] = x + (uint)(base + mm) * H;
  }
  float acc[MAXM];
  for (int m = 0; m < MAXM; ++m) { acc[m] = 0.0f; }
  for (int s = ix; s < nsb; s += 2) {
    const device uchar* sb = w + (uint)out_row * row_bytes + s * 210;
    const device uchar* q1 = sb + qlo;
    const device uchar* q2 = q1 + 32;
    const device uchar* qh = sb + 128 + qho;
    const device char*  sc = (const device char*)(sb + 192) + is;
    const float d = float(*(const device half*)(sb + 208));
    int wv0[4], wv1[4], wv2[4], wv3[4];
    for (int l = 0; l < 4; ++l) {
      const uint h = qh[l];
      wv0[l] = (int)((q1[l] & 0x0F) | ((h & 0x03) << 4)) - 32;
      wv1[l] = (int)((q2[l] & 0x0F) | ((h & 0x0C) << 2)) - 32;
      wv2[l] = (int)((q1[l] >> 4)   | ((h & 0x30) << 0)) - 32;
      wv3[l] = (int)((q2[l] >> 4)   | ((h & 0xC0) >> 2)) - 32;
    }
    const int xb = s * 256 + y_off;
    for (int m = 0; m < MAXM; ++m) {
      const device VPIPE_ELT* xm = xptr[m] + xb;
      float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int l = 0; l < 4; ++l) {
        sums[0] += float(xm[l +  0]) * float(wv0[l]);
        sums[1] += float(xm[l + 32]) * float(wv1[l]);
        sums[2] += float(xm[l + 64]) * float(wv2[l]);
        sums[3] += float(xm[l + 96]) * float(wv3[l]);
      }
      acc[m] += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                   + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
    }
  }
  for (int m = 0; m < MAXM; ++m) {
    const float v = simd_sum(acc[m]);
    if (lane == 0 && m < m_rows) {
      y[(uint)(base + m) * ystride + yoff + out_row] = VPIPE_ELT(v);
    }
  }
}

// Entry points: MAXM=2 (qmv_q*k_batch_f16, depth-1 window, grid.z=ceil(m/2))
// and MAXM=4 (qmv_q*k_batch4_f16, depth-2 verify n=3..4 in one tile, grid.z=1).
#define KQ_BATCH_ENTRY(NAME, IMPL, MAXM)                                     \
  kernel void NAME(                                                          \
      const device uchar*     w       [[buffer(0)]],                         \
      const device VPIPE_ELT* x       [[buffer(1)]],                         \
      device VPIPE_ELT*       y       [[buffer(2)]],                         \
      constant int&           H       [[buffer(3)]],                         \
      constant int&           N       [[buffer(4)]],                         \
      constant int&           m_total [[buffer(5)]],                         \
      constant int&           ystride [[buffer(6)]],                         \
      constant int&           yoff    [[buffer(7)]],                         \
      uint3 tid      [[threadgroup_position_in_grid]],                       \
      uint  simd_gid [[simdgroup_index_in_threadgroup]],                     \
      uint  simd_lid [[thread_index_in_simdgroup]]) {                        \
    IMPL<MAXM>(w, x, y, H, N, m_total, ystride, yoff, tid, simd_gid,         \
               simd_lid);                                                    \
  }
KQ_BATCH_ENTRY(qmv_q4k_batch_f16,  qmv_q4k_batch_impl, 2)
KQ_BATCH_ENTRY(qmv_q4k_batch4_f16, qmv_q4k_batch_impl, 4)
KQ_BATCH_ENTRY(qmv_q5k_batch_f16,  qmv_q5k_batch_impl, 2)
KQ_BATCH_ENTRY(qmv_q5k_batch4_f16, qmv_q5k_batch_impl, 4)
KQ_BATCH_ENTRY(qmv_q6k_batch_f16,  qmv_q6k_batch_impl, 2)
KQ_BATCH_ENTRY(qmv_q6k_batch4_f16, qmv_q6k_batch_impl, 4)

// Transpose [A, B, D] -> [B, A, D] (D-vectors kept intact). Used by
// batched prefill to swap token-major <-> head-major:
//   q [n, Hq, D] -> [Hq, n, D]   (A=n, B=Hq)
//   attn [Hq, n, D] -> [n, Hq, D] (A=Hq, B=n)
// 0:in 1:out 2:A 3:B 4:D.  grid (D, B, A).
kernel void transpose_abd_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      A   [[buffer(2)]],
    constant int&      B   [[buffer(3)]],
    constant int&      D   [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int b = (int)gid.y;
  const int a = (int)gid.z;
  out[((uint)b * A + a) * D + d] = in[((uint)a * B + b) * D + d];
}

// transpose_abd with an output ROW-STRIDE: writes out[b*out_rs + a*D + d] so the
// [B, A*D] result lands as columns [0 : A*D] of a WIDER buffer (out_rs >= A*D),
// e.g. attention -> scat[:, :H]. out_rs = 0 falls back to the contiguous A*D.
kernel void transpose_abd_rs_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      A   [[buffer(2)]],
    constant int&      B   [[buffer(3)]],
    constant int&      D   [[buffer(4)]],
    constant int&      out_rs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int b = (int)gid.y;
  const int a = (int)gid.z;
  const int rs = out_rs > 0 ? out_rs : A * D;
  out[(uint)b * rs + a * D + d] = in[((uint)a * B + b) * D + d];
}

// transpose_abd with a PADDED output head-dim: [A, B, D] -> [B, A, Dp] where
// each D-vector is zero-extended to Dp (Dp >= D). Lets a model whose head_dim
// is not one of the flash-attention kernels' supported widths (the Boogu DiT's
// 120) run on the steel bd128 path: padding K/Q with zeros leaves every dot
// product unchanged and padding V with zeros contributes 0 to the output, so
// the padded attention is EXACT. Pair with transpose_abd_unpad_f16 to drop the
// padding off the result.
// 0:in 1:out 2:A 3:B 4:D 5:Dp.  grid (Dp, B, A).
kernel void transpose_abd_pad_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      A   [[buffer(2)]],
    constant int&      B   [[buffer(3)]],
    constant int&      D   [[buffer(4)]],
    constant int&      Dp  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= Dp) { return; }
  const int b = (int)gid.y;
  const int a = (int)gid.z;
  out[((uint)b * A + a) * Dp + d] =
      d < D ? in[((uint)a * B + b) * D + d] : VPIPE_ELT(0.0f);
}

// Inverse of transpose_abd_pad_f16: [A, B, Dp] -> [B, A, D], dropping each
// Dp-vector's [D, Dp) padding tail.
// 0:in 1:out 2:A 3:B 4:D 5:Dp.  grid (D, B, A).
kernel void transpose_abd_unpad_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      A   [[buffer(2)]],
    constant int&      B   [[buffer(3)]],
    constant int&      D   [[buffer(4)]],
    constant int&      Dp  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int b = (int)gid.y;
  const int a = (int)gid.z;
  out[((uint)b * A + a) * D + d] = in[((uint)a * B + b) * Dp + d];
}

// Write src[Hkv, n, D] into a [Hkv, MAX_SEQ, D] KV cache at sequence
// slot `pos` (rows pos..pos+n-1). MAX_SEQ doubles as the RING capacity:
// the write slot is (pos+t) % MAX_SEQ, so a sliding-window layer whose
// buffer is sized to its bounded capacity wraps in place. For a full
// layer MAX_SEQ == max_seq and pos+t < max_seq, so the modulo is a
// no-op. 0:src 1:cache 2:MAX_SEQ 3:D 4:n 5:pos. grid (D, n, Hkv).
// MAX_SEQ here is the PHYSICAL head stride (== ring modulo + mirror tail for a
// bounded sliding ring; == ring modulo otherwise). ring_cap/window are optional
// trailing constants: ring_cap>0 selects the bounded ring (slot = pos % ring_cap)
// and mirrors a head-slot (slot < window-1) into slot+ring_cap so the trailing
// window reads as one linear span. ring_cap<=0 -> plain linear (slot = pos),
// no mirror -- byte-identical to the pre-mirror behavior.
kernel void kv_write_f16(
    const device VPIPE_ELT* src     [[buffer(0)]],
    device VPIPE_ELT*       cache   [[buffer(1)]],
    constant int&      MAX_SEQ [[buffer(2)]],
    constant int&      D       [[buffer(3)]],
    constant int&      n       [[buffer(4)]],
    constant int&      pos     [[buffer(5)]],
    constant int&      ring_cap [[buffer(6)]],
    constant int&      window   [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const int mod  = (ring_cap > 0) ? ring_cap : MAX_SEQ;
  const int slot = (pos + t) % mod;
  const VPIPE_ELT val = src[((uint)h * n + t) * D + d];
  cache[((uint)h * MAX_SEQ + slot) * D + d] = val;
  // Mirror a head slot into the tail so a window-length scan past ring end
  // reads the mirror linearly instead of wrapping.
  if (ring_cap > 0 && slot < window - 1) {
    cache[((uint)h * MAX_SEQ + (uint)(slot + ring_cap)) * D + d] = val;
  }
}

// Fused K+V write: same geometry as kv_write_f16, but writes both the K and
// the V token range in ONE dispatch (decode issues kv_write per K and per V
// -> 2 launches/layer; fusing halves that, cutting the dependent-dispatch
// idle). K and V share the layer's MAX_SEQ/D/n/pos. grid (D, n, Hkv).
//   0:src_k 1:cache_k 2:src_v 3:cache_v 4:MAX_SEQ 5:D 6:n 7:pos
//   8:ring_cap 9:window  (MAX_SEQ == physical head stride; see kv_write_f16)
kernel void kv_write2_f16(
    const device VPIPE_ELT* src_k   [[buffer(0)]],
    device VPIPE_ELT*       cache_k [[buffer(1)]],
    const device VPIPE_ELT* src_v   [[buffer(2)]],
    device VPIPE_ELT*       cache_v [[buffer(3)]],
    constant int&      MAX_SEQ [[buffer(4)]],
    constant int&      D       [[buffer(5)]],
    constant int&      n       [[buffer(6)]],
    constant int&      pos     [[buffer(7)]],
    constant int&      ring_cap [[buffer(8)]],
    constant int&      window   [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const int mod  = (ring_cap > 0) ? ring_cap : MAX_SEQ;
  const int slot = (pos + t) % mod;
  const uint ci = ((uint)h * MAX_SEQ + slot) * D + d;
  const uint si = ((uint)h * n + t) * D + d;
  const VPIPE_ELT kval = src_k[si];
  const VPIPE_ELT vval = src_v[si];
  cache_k[ci] = kval;
  cache_v[ci] = vval;
  if (ring_cap > 0 && slot < window - 1) {
    const uint mi = ((uint)h * MAX_SEQ + (uint)(slot + ring_cap)) * D + d;
    cache_k[mi] = kval;
    cache_v[mi] = vval;
  }
}

// Sub-blocked ring write: write the token sub-range [src_off, src_off+cnt) of a
// FULL source src[Hkv, n_src, D] into the bounded ring cache[Hkv, MAX_SEQ, D].
// Identical addressing to kv_write_f16 but the source row is (src_off + t) with
// a per-head stride of n_src, and the logical position is (pos + src_off + t).
// The bounded-ring single-pass prefill computes Q/K/V over the WHOLE prompt in
// one batch but must sub-block the RING write to <= page (ring_cap - window)
// rows per dispatch: a single dispatch wider than the ring would have multiple
// source rows mapping to the same physical slot (pos % ring_cap collisions),
// and those parallel writes RACE -- non-deterministic ring contents. Each
// sub-block of <= page distinct slots writes each slot at most once.
//   0:src 1:cache 2:MAX_SEQ 3:D 4:n_src 5:pos 6:ring_cap 7:window 8:src_off
// grid (D, cnt, Hkv).
kernel void kv_write_sub_f16(
    const device VPIPE_ELT* src     [[buffer(0)]],
    device VPIPE_ELT*       cache   [[buffer(1)]],
    constant int&      MAX_SEQ [[buffer(2)]],
    constant int&      D       [[buffer(3)]],
    constant int&      n_src   [[buffer(4)]],
    constant int&      pos     [[buffer(5)]],
    constant int&      ring_cap [[buffer(6)]],
    constant int&      window   [[buffer(7)]],
    constant int&      src_off  [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  const int sr = src_off + t;             // source row in [0, n_src)
  const int mod  = (ring_cap > 0) ? ring_cap : MAX_SEQ;
  const int slot = (pos + sr) % mod;
  const VPIPE_ELT val = src[((uint)h * n_src + sr) * D + d];
  cache[((uint)h * MAX_SEQ + slot) * D + d] = val;
  if (ring_cap > 0 && slot < window - 1) {
    cache[((uint)h * MAX_SEQ + (uint)(slot + ring_cap)) * D + d] = val;
  }
}

// Write a token chunk of src[Hkv, n_src, D] -- the token range
// [src_off, src_off+cnt) -- into ONE page of a paged KV pool. The page
// has layout [Hkv, page_tokens, D]; bind `page` at the page's byte base
// (page_id * Hkv*page_tokens*D*2) and tokens land at slots
// [dst_slot, dst_slot+cnt). Used by both decode (cnt=1, src_off=0) and
// chunked prefill (one dispatch per page the prefill spans), since a
// byte offset alone can't express the per-head source stride n_src.
//   0:src 1:page 2:page_tokens 3:D 4:n_src 5:src_off 6:dst_slot
// grid (D, cnt, Hkv).
kernel void kv_write_paged_f16(
    const device VPIPE_ELT* src         [[buffer(0)]],
    device VPIPE_ELT*       page        [[buffer(1)]],
    constant int&      page_tokens [[buffer(2)]],
    constant int&      D           [[buffer(3)]],
    constant int&      n_src       [[buffer(4)]],
    constant int&      src_off     [[buffer(5)]],
    constant int&      dst_slot    [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  page[((uint)h * page_tokens + dst_slot + t) * D + d] =
      src[((uint)h * n_src + src_off + t) * D + d];
}

// kv_gather_paged_f16 -- inverse of kv_write_paged: copy one page's tokens
// into a CONTIGUOUS [Hkv, dst_rows, D] buffer at the page's global
// position. Driven page-by-page from the page table, this assembles a
// context's full K/V (shared-prefix + new, across branch-shared/partial
// pages) into one contiguous tensor so the steel attention kernel -- which
// reads strided contiguous K/V -- can serve EXTEND prefill (q_offset>0).
//   0:dst (VPIPE_ELT) [Hkv, dst_rows, D]   1:page (= pool + page_id*page_stride)
//   2:page_tokens 3:D 4:dst_rows (dst seq stride) 5:dst_off (global_start)
// grid (D, cnt, Hkv); page tokens [0,cnt) -> dst rows [dst_off, dst_off+cnt).
kernel void kv_gather_paged_f16(
    device VPIPE_ELT*       dst         [[buffer(0)]],
    const device VPIPE_ELT* page        [[buffer(1)]],
    constant int&      page_tokens [[buffer(2)]],
    constant int&      D           [[buffer(3)]],
    constant int&      dst_rows    [[buffer(4)]],
    constant int&      dst_off     [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  dst[((uint)h * dst_rows + dst_off + t) * D + d] =
      page[((uint)h * page_tokens + t) * D + d];
}

// Greedy argmax over a [V] logits row -> out_id[0] (int32). Single
// threadgroup (256 lanes): each lane scans a strided slice of the
// vocab, then a threadgroup tree-reduce picks the max (ties -> lowest
// index, matching a host std left-to-right argmax). Used by the
// pipelined decode loop so the sampled token never round-trips to the
// host -- the embed gather reads out_id straight back as its `ids`.
//   0:logits(VPIPE_ELT) 1:out_id(int) 2:V
// In-place repetition penalty over one codebook's recent history: for each of
// the first `n_hist` token ids in `hist`, scale its logit by 1/penalty (if
// positive) or *penalty (if negative) -- the reference MossTTSRealtime rule.
// Single-threaded: the history window is tiny (<= 50) and this avoids a
// read/write race when a token repeats in the window.
kernel void rep_penalty_f16(
    device VPIPE_ELT*    logits  [[buffer(0)]],
    const device int*    hist    [[buffer(1)]],
    constant int&        n_hist  [[buffer(2)]],
    constant float&      penalty [[buffer(3)]],
    constant int&        V       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
  if (tid != 0 || penalty == 1.0f) { return; }
  for (int i = 0; i < n_hist; ++i) {
    const int t = hist[i];
    if (t < 0 || t >= V) { continue; }
    float v = float(logits[t]);
    v = v < 0.0f ? v * penalty : v / penalty;
    logits[t] = (VPIPE_ELT)v;
  }
}

kernel void argmax_f16(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device int*             out_id [[buffer(1)]],
    constant int&           V      [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg_val[256];
  threadgroup int   tg_idx[256];
  float best = -INFINITY;
  int   bi   = 0;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float v = float(logits[i]);
    if (v > best) { best = v; bi = (int)i; }
  }
  tg_val[tid] = best;
  tg_idx[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      // Break ties by LOWEST vocab index, matching a host left-to-right
      // argmax. Reducing by lane id would keep an arbitrary index on an
      // exact (f16) tie, diverging from the synchronous decode path.
      const bool take = tg_val[tid + s] > tg_val[tid] ||
          (tg_val[tid + s] == tg_val[tid] && tg_idx[tid + s] < tg_idx[tid]);
      if (take) {
        tg_val[tid] = tg_val[tid + s];
        tg_idx[tid] = tg_idx[tid + s];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { out_id[0] = tg_idx[0]; }
}

// ---------------------------------------------------------------------
// Two-stage argmax: parallelise the 512KB vocab read across GPU cores.
// argmax_f16 above runs in a SINGLE threadgroup -> one core does the whole
// 262k-element scan. Here stage-1 (argmax_partial_f16) fires M threadgroups,
// each owning a CONTIGUOUS vocab slab, and stage-2 (argmax_combine_f16)
// reduces the M (val,idx) partials in one small threadgroup.
//
// TIE-BREAK (must match argmax_f16 / the host left-to-right argmax exactly):
// the winner on an f16 tie is the LOWEST global vocab index. Because each
// partial carries the GLOBAL index of its slab-best (itself tie-broken to the
// lowest index inside the slab), and the combine breaks ties by lower index
// too, the global lowest-index winner is preserved bit-for-bit. Contiguous
// slabs are not required for correctness (the index tie-break is global), but
// they keep each threadgroup's reads coalesced.
//
// Intra-threadgroup reduction uses a simd_max on the value to find the slab
// max, then a second simd reduction to pick the lowest index AMONG lanes
// holding that max -- carrying the index through simd_shuffle so we never
// need the full 8-step barrier tree. A tiny cross-simdgroup tree (<=8 entries)
// finishes the threadgroup.

// (val,idx) argmax reduction across one simdgroup (32 lanes): returns the max
// value and, among lanes tied at that value, the lowest index. Branchless on
// the value via simd_max; the index is resolved by masking non-winners to INT
// MAX and taking simd_min.
inline void simd_argmax_lowidx(float v, int idx, thread float& out_v,
                               thread int& out_i)
{
  const float mx = simd_max(v);
  // Lanes not holding the max can't win; push their idx to +inf for the min.
  const int cand = (v == mx) ? idx : 0x7fffffff;
  out_v = mx;
  out_i = simd_min(cand);
}

//   0:logits 1:partials(float2-as-[val,idx] packed: 2*M floats) 2:V 3:M
kernel void argmax_partial_f16(
    const device VPIPE_ELT* logits   [[buffer(0)]],
    device float*           partials [[buffer(1)]],   // [2*M]: val,idx pairs
    constant int&           V        [[buffer(2)]],
    constant int&           M        [[buffer(3)]],
    uint  tid      [[thread_position_in_threadgroup]],
    uint  nthg     [[threads_per_threadgroup]],
    uint  gid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  // Contiguous slab for this threadgroup: [lo, hi).
  const uint slab = ((uint)V + (uint)M - 1u) / (uint)M;
  const uint lo = gid * slab;
  uint hi = lo + slab;
  if (hi > (uint)V) { hi = (uint)V; }

  float best = -INFINITY;
  int   bi   = 0x7fffffff;
  for (uint i = lo + tid; i < hi; i += nthg) {
    const float val = float(logits[i]);
    // Lowest-index tie-break within the lane's own stream.
    if (val > best || (val == best && (int)i < bi)) {
      best = val; bi = (int)i;
    }
  }
  // If this lane touched nothing (slab smaller than nthg), bi stays INT_MAX;
  // best stays -inf so it loses the value compare anyway.

  // Reduce within the simdgroup (32 lanes) -> (val,idx).
  float sv; int si;
  simd_argmax_lowidx(best, bi, sv, si);

  threadgroup float tg_val[32];   // <= 256/32 = 8 simdgroups, sized to 32.
  threadgroup int   tg_idx[32];
  if (simd_lid == 0) { tg_val[simd_gid] = sv; tg_idx[simd_gid] = si; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cross-simdgroup reduction in simdgroup 0 (<=8 live entries).
  if (simd_gid == 0) {
    const uint nsg = (nthg + 31u) / 32u;
    float lv = (simd_lid < nsg) ? tg_val[simd_lid] : -INFINITY;
    int   li = (simd_lid < nsg) ? tg_idx[simd_lid] : 0x7fffffff;
    float fv; int fi;
    simd_argmax_lowidx(lv, li, fv, fi);
    if (simd_lid == 0) {
      partials[2u * gid + 0u] = fv;
      partials[2u * gid + 1u] = float(fi);   // idx fits exactly in f32 < 2^24? V=262144 < 2^24 -> exact.
    }
  }
}

//   0:partials([2*M]) 1:out_id(int) 2:M
kernel void argmax_combine_f16(
    const device float* partials [[buffer(0)]],
    device int*         out_id   [[buffer(1)]],
    constant int&       M        [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg_val[256];
  threadgroup int   tg_idx[256];
  float best = -INFINITY;
  int   bi   = 0x7fffffff;
  for (uint p = tid; p < (uint)M; p += nthg) {
    const float val = partials[2u * p + 0u];
    const int   idx = (int)partials[2u * p + 1u];
    if (val > best || (val == best && idx < bi)) { best = val; bi = idx; }
  }
  tg_val[tid] = best;
  tg_idx[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      const bool take = tg_val[tid + s] > tg_val[tid] ||
          (tg_val[tid + s] == tg_val[tid] && tg_idx[tid + s] < tg_idx[tid]);
      if (take) {
        tg_val[tid] = tg_val[tid + s];
        tg_idx[tid] = tg_idx[tid + s];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { out_id[0] = tg_idx[0]; }
}

// Full sampler over a [V] logits row -> out_id[0] (int32), fully on-GPU
// so the sampled token never leaves the device and the pipelined decode
// loop runs non-greedy at greedy speed. Single threadgroup (256 lanes).
// Implements the SAME pipeline as the host llm::Sampler, in the same
// order:  repetition penalty -> presence penalty -> temperature ->
// softmax -> top-k -> min-p -> top-p (nucleus) -> renormalise ->
// multinomial draw.
//
// The three filters (top-k / min-p / top-p) all reduce to a weight
// THRESHOLD test, so we never materialise a full-vocab sort:
//   * after subtracting the max (penalised) logit, the max softmax
//     weight is exactly 1, so min-p's prob floor min_p*max_prob becomes a
//     weight floor t_mp = min_p;
//   * top-k's cutoff is the k-th largest weight, found by binary search
//     on the threshold whose count{w_i >= t} == k (count monotone in t);
//   * top-p's nucleus threshold is the largest t whose tail mass{w_i>=t}
//     >= top_p*Z, searched in [t_floor, 1] where t_floor = max(t_k, t_mp)
//     so the nucleus is taken over the post-top-k/min-p survivors (exactly
//     the host's apply-in-sequence semantics).
// The final pick is the Gumbel-max trick over {w_i >= wt}, equivalent to
// a multinomial draw in proportion to the kept weights (seed per step).
//
// Penalties fold into an "effective logit": for already-seen tokens
// (seen[i] != 0) eff = (v>=0 ? v/rep : v*rep) - presence, matching the
// host. seen[] is a per-context [V] uint8 flag buffer primed from the
// prompt and updated with each drawn token (this kernel sets it).
//
// KEY PERF: the softmax weights w_i = exp((eff_i - max)/temp) are computed
// ONCE into the `ws` scratch (Z pass); the threshold searches then just
// READ ws (no per-iteration exp over the whole vocab).
//   0:logits 1:out_id 2:V 3:temp 4:top_p 5:seed(uint)
//   6:ws (device VPIPE_ELT [V] scratch)  7:n_iter (binary-search steps)
//   8:rep_penalty 9:presence_penalty 10:top_k 11:min_p
//   12:seen (device uchar [V], in/out)
inline float vpipe_hash_u01(uint x)
{
  // bias-reduced integer hash (Wang/PCG-style) -> uniform (0,1).
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16;
  return (float(x) + 0.5f) * (1.0f / 4294967296.0f);
}

// Effective logit after repetition + presence penalty (host order).
inline float vpipe_eff_logit(float v, bool seen, float rep, float pres)
{
  if (seen) {
    v = (v >= 0.0f) ? (v / rep) : (v * rep);
    v -= pres;
  }
  return v;
}

kernel void sample_topp_f16(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device int*             out_id [[buffer(1)]],
    constant int&           V      [[buffer(2)]],
    constant float&         temp   [[buffer(3)]],
    constant float&         top_p  [[buffer(4)]],
    constant uint&          seed   [[buffer(5)]],
    device VPIPE_ELT*       ws     [[buffer(6)]],
    constant int&           n_iter [[buffer(7)]],
    constant float&         rep    [[buffer(8)]],
    constant float&         pres   [[buffer(9)]],
    constant int&           top_k  [[buffer(10)]],
    constant float&         min_p  [[buffer(11)]],
    device uchar*           seen   [[buffer(12)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg[256];
  threadgroup int   tgi[256];
  const float inv_t = 1.0f / max(temp, 1e-6f);

  // 1) max penalised logit (softmax stability).
  float m = -INFINITY;
  for (uint i = tid; i < (uint)V; i += nthg) {
    m = max(m, vpipe_eff_logit(float(logits[i]), seen[i] != 0, rep, pres));
  }
  tg[tid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] = max(tg[tid], tg[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float maxl = tg[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2) Z = sum w_i, caching w_i = exp((eff - maxl)/temp) into ws (the
  // ONE exp pass; everything below reads ws instead of recomputing).
  float z = 0.0f;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float e = vpipe_eff_logit(float(logits[i]), seen[i] != 0, rep, pres);
    const float w = exp((e - maxl) * inv_t);
    ws[i] = (VPIPE_ELT)w;
    z += w;
  }
  tg[tid] = z;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] += tg[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float Z = tg[0];
  const float target = top_p * Z;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3a) top-k weight threshold t_k: largest t with count{w_i >= t} >= k
  // (count monotone-decreasing in t). Skipped (t_k=0) when top_k<=0 or
  // top_k>=V. Integer count reduction over the cached ws.
  float t_k = 0.0f;
  if (top_k > 0 && top_k < V) {
    float lo = 0.0f, hi = 1.0f;
    for (int it = 0; it < n_iter; ++it) {
      const float mid = 0.5f * (lo + hi);
      int cnt = 0;
      for (uint i = tid; i < (uint)V; i += nthg) {
        if (float(ws[i]) >= mid) { ++cnt; }
      }
      tgi[tid] = cnt;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint s = nthg >> 1; s > 0; s >>= 1) {
        if (tid < s) { tgi[tid] += tgi[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const int C = tgi[0];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (C >= top_k) { lo = mid; } else { hi = mid; }
    }
    t_k = lo;
  }
  // 3b) min-p weight floor: prob floor min_p*max_prob == weight floor
  // min_p (max weight is 1 after the max-subtraction).
  const float t_mp = max(min_p, 0.0f);
  const float t_floor = max(t_k, t_mp);

  // 3c) top-p nucleus threshold: largest t in [t_floor,1] whose tail
  // mass{w_i >= t} >= top_p*Z. Searching from t_floor keeps the nucleus
  // over the post-top-k/min-p survivors.
  float lo = t_floor, hi = 1.0f;
  for (int it = 0; it < n_iter; ++it) {
    const float mid = 0.5f * (lo + hi);
    float mass = 0.0f;
    for (uint i = tid; i < (uint)V; i += nthg) {
      const float w = float(ws[i]);
      if (w >= mid) { mass += w; }
    }
    tg[tid] = mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthg >> 1; s > 0; s >>= 1) {
      if (tid < s) { tg[tid] += tg[tid + s]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float M = tg[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (M >= target) { lo = mid; } else { hi = mid; }
  }
  const float wt = max(lo, t_floor);

  // 4) Gumbel-max over the nucleus {w_i >= wt}: argmax of eff_i/temp +
  // Gumbel noise samples i in proportion to exp(eff_i/temp) over the kept
  // set. Ties broken by lowest index (deterministic given seed).
  float best = -INFINITY;
  int   bi   = -1;
  for (uint i = tid; i < (uint)V; i += nthg) {
    if (float(ws[i]) >= wt) {
      const float e =
          vpipe_eff_logit(float(logits[i]), seen[i] != 0, rep, pres);
      const float u = vpipe_hash_u01(seed ^ (i * 2654435761u + 1u));
      const float g = -log(-log(u + 1e-20f) + 1e-20f);
      const float score = e * inv_t + g;
      if (score > best) { best = score; bi = (int)i; }
    }
  }
  tg[tid]  = best;
  tgi[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      const bool take = tg[tid + s] > tg[tid] ||
          (tg[tid + s] == tg[tid] && tgi[tid + s] >= 0 &&
           (tgi[tid] < 0 || tgi[tid + s] < tgi[tid]));
      if (take) { tg[tid] = tg[tid + s]; tgi[tid] = tgi[tid + s]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Fallback to argmax of the (degenerate) row if nothing was kept; mark
  // the drawn token seen so penalties see it on the next step.
  if (tid == 0) {
    const int pick = tgi[0] >= 0 ? tgi[0] : 0;
    out_id[0] = pick;
    seen[pick] = 1;
  }
}

// =====================================================================
// HISTOGRAM-BASED MULTI-THREADGROUP SAMPLER (sample_topp drop-in).
//
// sample_topp_f16 above is a SINGLE threadgroup that rescans the full 262k
// vocab ~18x (the top-k / top-p binary searches). These kernels keep its
// EXACT semantics (penalty formula+order, temp/softmax, the top-k/min-p/top-p
// MAX-of-thresholds combination, the Gumbel hash + seed, the seen[] update)
// but reduce the work to ~3 multi-tg vocab passes + a tiny per-bin pass:
//
//   Pass A  (sample_max_partial / _combine): max of the penalised eff over
//           vocab, split across slabs (mirrors argmax_partial/_combine).
//   Pass B  (sample_zhist_partial / _combine): ONE vocab pass writes
//           ws[i] = exp((eff_i - max)/temp), accumulates a partial Z, AND a
//           per-tg log-spaced HISTOGRAM of ws into B bins; the combine sums
//           the partial Z's and histograms into a global (count,mass)[B].
//   Thresh  (sample_thresh): single tg over the B bins -- scan high-weight
//           bin -> low, find the top-k count cutoff, the top-p mass cutoff,
//           and the min-p floor; wt = max of the three bin-edge thresholds.
//           NO vocab rescan (B is tiny).
//   Pass C  (sample_pick_partial / _combine): Gumbel-max over {ws_i >= wt},
//           lowest-index tie-break, write out_id, set seen[out_id].
//
// BIN LAYOUT: after subtracting the max penalised logit, max weight == 1, so
// log-weight lw = -log(w) >= 0. Bins are uniform in lw over [0, kSampLwMax):
// bin = floor(lw / (kSampLwMax/B)), clamped to [0, B-1] (bin 0 = highest
// weight). The threshold a bin represents is its LOW-weight edge
// exp(-(bin+1)*binw) -- including the whole bin makes the nucleus/top-k
// slightly LOOSER than the exact binary search (distributionally negligible
// at B=1024). Bins are reduced as float pairs (count, mass) so the combine is
// a plain element-wise add.
//
// The histogram is laid out [2*B] floats per partial: [0..B) counts then
// [B..2B) masses, so a tg writes its bins contiguously. The global histogram
// after combine is also [2*B]: counts then masses.

constant constexpr int   kSampB     = 1024;   // histogram bins
constant constexpr float kSampLwMax = 40.0f;  // lw range [0,40): w in [e^-40,1]
// Threadgroup atomic_float is unsupported in this MSL toolchain, so the per-tg
// histogram counts go through atomic_uint (exact) and the per-tg histogram MASS
// goes through atomic_uint fixed-point (weight w in (0,1] scaled by 2^16). A
// single tg's slab has <= V/M elements (each mass <= 1), so the max per-bin
// fixed-point sum is (V/M)*2^16; at M>=64, V=262144 that is < 2.7e8 << 2^32, so
// no overflow. The 2^-16 quantum is ~1.5e-5 per add -- far finer than the ~4%
// (one-bin) granularity the threshold scan needs, so it is distributionally
// invisible.
constant constexpr float kSampMassScale = 65536.0f;   // 2^16 fixed-point

inline int vpipe_samp_bin(float w)
{
  // w in (0,1]; lw = -log(w) >= 0. Clamp lw into [0, kSampLwMax) -> bin.
  const float lw = -log(max(w, 1e-30f));
  const float binw = kSampLwMax / float(kSampB);
  int b = int(lw / binw);
  if (b < 0) { b = 0; }
  if (b >= kSampB) { b = kSampB - 1; }
  return b;
}

// Pass A stage 1: partial max of penalised eff over a contiguous slab.
//   0:logits 1:partials([M] f32 maxes) 2:V 3:M 4:rep 5:pres 6:seen
kernel void sample_max_partial_f16(
    const device VPIPE_ELT* logits   [[buffer(0)]],
    device float*           partials [[buffer(1)]],
    constant int&           V        [[buffer(2)]],
    constant int&           M        [[buffer(3)]],
    constant float&         rep      [[buffer(4)]],
    constant float&         pres     [[buffer(5)]],
    const device uchar*     seen     [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
  const uint slab = ((uint)V + (uint)M - 1u) / (uint)M;
  const uint lo = gid * slab;
  uint hi = lo + slab;
  if (hi > (uint)V) { hi = (uint)V; }

  threadgroup float tg[256];
  float m = -INFINITY;
  for (uint i = lo + tid; i < hi; i += nthg) {
    m = max(m, vpipe_eff_logit(float(logits[i]), seen[i] != 0, rep, pres));
  }
  tg[tid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] = max(tg[tid], tg[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { partials[gid] = tg[0]; }
}

// Pass A stage 2: combine the M partial maxes -> partials_out[0].
//   0:partials([M]) 1:maxl_out([1] f32) 2:M
kernel void sample_max_combine_f16(
    const device float* partials [[buffer(0)]],
    device float*       maxl_out [[buffer(1)]],
    constant int&       M        [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg[256];
  float m = -INFINITY;
  for (uint p = tid; p < (uint)M; p += nthg) { m = max(m, partials[p]); }
  tg[tid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] = max(tg[tid], tg[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { maxl_out[0] = tg[0]; }
}

// Pass B stage 1: over a contiguous slab, cache ws[i]=exp((eff-maxl)/temp),
// accumulate partial Z and a per-tg log-spaced histogram (count,mass) of ws.
// Each partial occupies [2*kSampB+1] floats: [0..B) counts, [B..2B) masses,
// [2B] = partial Z.
//   0:logits 1:ws 2:partials 3:V 4:M 5:temp 6:rep 7:pres 8:seen 9:maxl_in([1])
kernel void sample_zhist_partial_f16(
    const device VPIPE_ELT* logits   [[buffer(0)]],
    device VPIPE_ELT*       ws       [[buffer(1)]],
    device float*           partials [[buffer(2)]],
    constant int&           V        [[buffer(3)]],
    constant int&           M        [[buffer(4)]],
    constant float&         temp     [[buffer(5)]],
    constant float&         rep      [[buffer(6)]],
    constant float&         pres     [[buffer(7)]],
    const device uchar*     seen     [[buffer(8)]],
    const device float*     maxl_in  [[buffer(9)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
  const float maxl  = maxl_in[0];
  const float inv_t = 1.0f / max(temp, 1e-6f);
  const uint  slab  = ((uint)V + (uint)M - 1u) / (uint)M;
  const uint  lo    = gid * slab;
  uint hi = lo + slab;
  if (hi > (uint)V) { hi = (uint)V; }

  // Per-tg histogram in threadgroup memory: counts[B] (exact uint) and masses
  // [B] (uint fixed-point, scale 2^16). atomic_float is unsupported here.
  threadgroup atomic_uint h_cnt[kSampB];
  threadgroup atomic_uint h_mass[kSampB];
  for (uint b = tid; b < (uint)kSampB; b += nthg) {
    atomic_store_explicit(&h_cnt[b], 0u, memory_order_relaxed);
    atomic_store_explicit(&h_mass[b], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float z = 0.0f;
  for (uint i = lo + tid; i < hi; i += nthg) {
    const float e = vpipe_eff_logit(float(logits[i]), seen[i] != 0, rep, pres);
    const float w = exp((e - maxl) * inv_t);
    ws[i] = (VPIPE_ELT)w;
    z += w;
    const int b = vpipe_samp_bin(w);
    atomic_fetch_add_explicit(&h_cnt[b], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&h_mass[b],
                              (uint)(w * kSampMassScale), memory_order_relaxed);
  }
  // Partial Z reduction.
  threadgroup float tg[256];
  tg[tid] = z;
  threadgroup_barrier(mem_flags::mem_threadgroup |
                      mem_flags::mem_device);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] += tg[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Flush this tg's histogram + Z to its partial slab (count, mass-as-float).
  device float* P = partials + (size_t)gid * (size_t)(2 * kSampB + 1);
  for (uint b = tid; b < (uint)kSampB; b += nthg) {
    P[b]          = (float)atomic_load_explicit(&h_cnt[b], memory_order_relaxed);
    P[kSampB + b] = (float)atomic_load_explicit(&h_mass[b], memory_order_relaxed)
                    * (1.0f / kSampMassScale);
  }
  if (tid == 0) { P[2 * kSampB] = tg[0]; }
}

// Pass B stage 2: sum the M partial histograms + Z into the global histogram.
// Global layout: hist[0..B)=counts, hist[B..2B)=masses, hist[2B]=Z.
//   0:partials 1:hist_out([2B+1]) 2:M
kernel void sample_zhist_combine_f16(
    const device float* partials [[buffer(0)]],
    device float*       hist_out [[buffer(1)]],
    constant int&       M        [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  const uint stride = (uint)(2 * kSampB + 1);
  // Each thread owns a strided set of the 2B histogram cells; sum over M.
  for (uint c = tid; c < (uint)(2 * kSampB); c += nthg) {
    float acc = 0.0f;
    for (uint p = 0; p < (uint)M; ++p) { acc += partials[p * stride + c]; }
    hist_out[c] = acc;
  }
  if (tid == 0) {
    float zsum = 0.0f;
    for (uint p = 0; p < (uint)M; ++p) {
      zsum += partials[p * stride + (uint)(2 * kSampB)];
    }
    hist_out[2 * kSampB] = zsum;
  }
}

// Threshold stage: single tg over the B bins. Scan high-weight bin (0) -> low,
// accumulating count and mass; the top-k threshold is the LOW edge of the bin
// where cumulative count first reaches top_k, the top-p threshold the LOW edge
// of the bin where cumulative mass first reaches top_p*Z. min-p floor is the
// raw min_p weight. wt = max(t_k, t_mp, t_p). Writes wt to wt_out[0].
//   0:hist([2B+1]) 1:wt_out([1]) 2:top_p 3:top_k 4:min_p 5:V
kernel void sample_thresh_f16(
    const device float* hist   [[buffer(0)]],
    device float*       wt_out [[buffer(1)]],
    constant float&     top_p  [[buffer(2)]],
    constant int&       top_k  [[buffer(3)]],
    constant float&     min_p  [[buffer(4)]],
    constant int&       V      [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]])
{
  if (tid != 0) { return; }
  const float Z      = hist[2 * kSampB];
  const float target = top_p * Z;
  const float binw   = kSampLwMax / float(kSampB);

  // bin b spans weights (exp(-(b+1)*binw), exp(-b*binw)]; its LOW edge (the
  // threshold that INCLUDES the whole bin) is edge(b) = exp(-(b+1)*binw).
  float t_k = 0.0f, t_p = 0.0f;
  float cum_cnt = 0.0f, cum_mass = 0.0f;
  bool have_k = false, have_p = false;
  const bool do_k = (top_k > 0 && top_k < V);
  for (int b = 0; b < kSampB; ++b) {
    cum_cnt  += hist[b];
    cum_mass += hist[kSampB + b];
    const float edge = exp(-float(b + 1) * binw);
    if (do_k && !have_k && cum_cnt >= float(top_k)) {
      t_k = edge; have_k = true;
    }
    if (!have_p && cum_mass >= target) {
      t_p = edge; have_p = true;
    }
    if ((!do_k || have_k) && have_p) { break; }
  }
  // If a target was never reached (e.g. all mass in the tail bin), the
  // threshold stays 0 -> keeps everything (matches the binary search's lo=0).
  const float t_mp = max(min_p, 0.0f);
  wt_out[0] = max(max(t_k, t_mp), t_p);
}

// Pass C stage 1: Gumbel-max partial over a contiguous slab of {ws_i >= wt}.
// partials: [2*M] (score,idx) pairs (idx as float, exact for V<2^24).
//   0:logits 1:ws 2:partials 3:V 4:M 5:temp 6:seed 7:rep 8:pres 9:seen
//   10:wt_in([1])
kernel void sample_pick_partial_f16(
    const device VPIPE_ELT* logits   [[buffer(0)]],
    const device VPIPE_ELT* ws       [[buffer(1)]],
    device float*           partials [[buffer(2)]],
    constant int&           V        [[buffer(3)]],
    constant int&           M        [[buffer(4)]],
    constant float&         temp     [[buffer(5)]],
    constant uint&          seed     [[buffer(6)]],
    constant float&         rep      [[buffer(7)]],
    constant float&         pres     [[buffer(8)]],
    const device uchar*     seen     [[buffer(9)]],
    const device float*     wt_in    [[buffer(10)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
  const float inv_t = 1.0f / max(temp, 1e-6f);
  const float wt    = wt_in[0];
  const uint  slab  = ((uint)V + (uint)M - 1u) / (uint)M;
  const uint  lo    = gid * slab;
  uint hi = lo + slab;
  if (hi > (uint)V) { hi = (uint)V; }

  threadgroup float tg[256];
  threadgroup int   tgi[256];
  float best = -INFINITY;
  int   bi   = -1;
  for (uint i = lo + tid; i < hi; i += nthg) {
    if (float(ws[i]) >= wt) {
      const float e =
          vpipe_eff_logit(float(logits[i]), seen[i] != 0, rep, pres);
      const float u = vpipe_hash_u01(seed ^ (i * 2654435761u + 1u));
      const float g = -log(-log(u + 1e-20f) + 1e-20f);
      const float score = e * inv_t + g;
      // Lowest-index tie-break (deterministic given seed).
      if (score > best || (score == best && (int)i < bi)) {
        best = score; bi = (int)i;
      }
    }
  }
  tg[tid]  = best;
  tgi[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      const bool take = tg[tid + s] > tg[tid] ||
          (tg[tid + s] == tg[tid] && tgi[tid + s] >= 0 &&
           (tgi[tid] < 0 || tgi[tid + s] < tgi[tid]));
      if (take) { tg[tid] = tg[tid + s]; tgi[tid] = tgi[tid + s]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    partials[2u * gid + 0u] = tg[0];
    partials[2u * gid + 1u] = float(tgi[0]);
  }
}

// Pass C stage 2: combine the M (score,idx) partials -> out_id, set seen.
// Same tie-break as sample_topp_f16's final reduction (lowest index among
// equal scores; -1 means "nothing kept in this partial").
//   0:partials([2M]) 1:out_id 2:M 3:seen
kernel void sample_pick_combine_f16(
    const device float* partials [[buffer(0)]],
    device int*         out_id   [[buffer(1)]],
    constant int&       M        [[buffer(2)]],
    device uchar*       seen     [[buffer(3)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg[256];
  threadgroup int   tgi[256];
  float best = -INFINITY;
  int   bi   = -1;
  for (uint p = tid; p < (uint)M; p += nthg) {
    const float sc = partials[2u * p + 0u];
    const int   id = (int)partials[2u * p + 1u];
    if (id < 0) { continue; }
    if (sc > best || (sc == best && (bi < 0 || id < bi))) {
      best = sc; bi = id;
    }
  }
  tg[tid]  = best;
  tgi[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      const bool take = tg[tid + s] > tg[tid] ||
          (tg[tid + s] == tg[tid] && tgi[tid + s] >= 0 &&
           (tgi[tid] < 0 || tgi[tid + s] < tgi[tid]));
      if (take) { tg[tid] = tg[tid + s]; tgi[tid] = tgi[tid + s]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    const int pick = tgi[0] >= 0 ? tgi[0] : 0;
    out_id[0] = pick;
    seen[pick] = 1;
  }
}

// ---------------------------------------------------------------------
// Leviathan-Chen speculative-sampling kernels (on-GPU MTP correction).
//
// The host L-C path post-processed the full [V] verifier / MTP logits on the
// CPU -- a per-row softmax + 2x threshold bisection + residual + CDF walk, run
// ~7x per round single-threaded over V (~152K). These kernels do the SAME
// nucleus math + accept/residual ON the GPU (one threadgroup per row), so a
// round only moves a handful of token ids back to the host. The nucleus stage
// mirrors sample_topp_f16 exactly (max -> single exp-cache -> top_k/min_p/top_p
// threshold bisection); sampling is Gumbel-max over the kept set (no CDF, no
// sort), matching the GPU decode sampler. L-C is a SAMPLING path (repetition /
// presence penalties are gated out before we reach it), so there is no
// `seen`/penalty handling here.

// Shared nucleus stage: cache ws[i] = exp((logit_i - max)/temp) over the row and
// return the nucleus weight cutoff `wt` plus the kept-weight sum `kept`, so a
// kept token's normalized prob is ws[i]/kept (when ws[i] >= wt) else 0.
// Threadgroup-UNIFORM: every thread in the group must call it (it barriers
// internally), and `tg`/`tgi` are the caller's shared scratch.
inline void lc_nucleus_stage(
    const device VPIPE_ELT* logits, device VPIPE_ELT* ws,
    int V, float inv_t, float top_p, int top_k, float min_p, int n_iter,
    threadgroup float* tg, threadgroup int* tgi, uint tid, uint nthg,
    thread float& wt_out, thread float& kept_out)
{
  // 1) max logit (softmax stability).
  float m = -INFINITY;
  for (uint i = tid; i < (uint)V; i += nthg) { m = max(m, float(logits[i])); }
  tg[tid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] = max(tg[tid], tg[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float maxl = tg[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2) Z = sum w_i, caching w_i into ws (the ONE exp pass).
  float z = 0.0f;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float w = exp((float(logits[i]) - maxl) * inv_t);
    ws[i] = (VPIPE_ELT)w;
    z += w;
  }
  tg[tid] = z;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] += tg[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float Z = tg[0];
  const float target = top_p * Z;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3a) top-k weight threshold: largest t with count{w_i >= t} >= k.
  float t_k = 0.0f;
  if (top_k > 0 && top_k < V) {
    float lo = 0.0f, hi = 1.0f;
    for (int it = 0; it < n_iter; ++it) {
      const float mid = 0.5f * (lo + hi);
      int cnt = 0;
      for (uint i = tid; i < (uint)V; i += nthg) {
        if (float(ws[i]) >= mid) { ++cnt; }
      }
      tgi[tid] = cnt;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint s = nthg >> 1; s > 0; s >>= 1) {
        if (tid < s) { tgi[tid] += tgi[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const int C = tgi[0];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (C >= top_k) { lo = mid; } else { hi = mid; }
    }
    t_k = lo;
  }
  const float t_floor = max(t_k, max(min_p, 0.0f));

  // 3b) top-p nucleus threshold: largest t in [t_floor,1] whose tail mass
  // {w_i >= t} >= top_p*Z.
  float lo = t_floor, hi = 1.0f;
  for (int it = 0; it < n_iter; ++it) {
    const float mid = 0.5f * (lo + hi);
    float mass = 0.0f;
    for (uint i = tid; i < (uint)V; i += nthg) {
      const float w = float(ws[i]);
      if (w >= mid) { mass += w; }
    }
    tg[tid] = mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthg >> 1; s > 0; s >>= 1) {
      if (tid < s) { tg[tid] += tg[tid + s]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float M = tg[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (M >= target) { lo = mid; } else { hi = mid; }
  }
  const float wt = max(lo, t_floor);

  // 4) kept = sum of kept weights, so a kept token's prob is ws[i]/kept.
  float kept = 0.0f;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float w = float(ws[i]);
    if (w >= wt) { kept += w; }
  }
  tg[tid] = kept;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] += tg[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  kept_out = tg[0];
  wt_out = wt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Histogram twin of lc_nucleus_stage: replaces the 2*n_iter full-V threshold
// bisections with a SINGLE-PASS count histogram over log(weight), then a tiny
// B-bin scan that finds the top_p/top_k/min_p cutoff -- so the per-nucleus
// full-V passes drop from ~(2 + 2*n_iter) to ~3 (max, expZ+histogram, exact
// kept). Weights w_i = exp((l_i - max)/temp) in (0,1]; log(w_i) in [LC_LOGMIN,
// 0]; bin = (log(w) - LOGMIN)/(-LOGMIN)*B. The scan reconstructs each bin's mass
// as count*midpoint-weight (a <=1 bin-width approximation, fine for a lossless
// sampling path); the KEPT sum for the chosen threshold is then computed EXACTLY
// in one pass, so p(d)=w/kept downstream stays exact. The cutoff weight is a bin
// lower edge (B=1024 over 30 log-units => ~3% weight granularity).
#define LC_HIST_B 1024
constant float LC_LOGMIN = -30.0f;

inline void lc_nucleus_hist(
    const device VPIPE_ELT* logits, device VPIPE_ELT* ws,
    int V, float inv_t, float top_p, int top_k, float min_p,
    threadgroup float* tg, threadgroup int* tgi,
    threadgroup atomic_uint* hist, threadgroup float* sh,
    uint tid, uint nthg, thread float& wt_out, thread float& kept_out)
{
  (void)tgi;                 // bisection-only scratch; the histogram uses `hist`
  // 1) max logit.
  float m = -INFINITY;
  for (uint i = tid; i < (uint)V; i += nthg) { m = max(m, float(logits[i])); }
  tg[tid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] = max(tg[tid], tg[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float maxl = tg[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // zero the histogram.
  for (uint b = tid; b < (uint)LC_HIST_B; b += nthg) {
    atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2) Z = sum w_i, cache w_i into ws, AND bin it (skip negligible weights so
  // both the work and the bin-0 contention collapse at low temperature).
  const float inv_range = 1.0f / (-LC_LOGMIN);
  const float wmin = exp(LC_LOGMIN);
  float z = 0.0f;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float w = exp((float(logits[i]) - maxl) * inv_t);
    ws[i] = (VPIPE_ELT)w;
    z += w;
    if (w > wmin) {
      const float lw = (float(logits[i]) - maxl) * inv_t;   // == log(w)
      int b = (int)((lw - LC_LOGMIN) * inv_range * (float)LC_HIST_B);
      b = clamp(b, 0, LC_HIST_B - 1);
      atomic_fetch_add_explicit(&hist[b], 1u, memory_order_relaxed);
    }
  }
  tg[tid] = z;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] += tg[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float Z = tg[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3) scan the bins high->low on one thread; keep bins until the cumulative
  // (reconstructed) mass >= top_p*Z and/or the count >= top_k; wt = the most
  // restrictive cutoff's bin lower edge, floored by min_p.
  if (tid == 0) {
    const float targetp = top_p * Z;
    const float binw = (-LC_LOGMIN) / (float)LC_HIST_B;
    float cmass = 0.0f; int ccnt = 0;
    float wt_p = 0.0f, wt_k = 0.0f;
    bool got_p = (top_p >= 1.0f - 1e-9f);
    bool got_k = (top_k <= 0 || top_k >= V);
    for (int b = LC_HIST_B - 1; b >= 0; --b) {
      const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
      const float wlo  = exp(LC_LOGMIN + (float)b * binw);
      const float wmid = exp(LC_LOGMIN + ((float)b + 0.5f) * binw);
      cmass += (float)c * wmid;
      ccnt  += (int)c;
      if (!got_p && cmass >= targetp) { wt_p = wlo; got_p = true; }
      if (!got_k && ccnt >= top_k)    { wt_k = wlo; got_k = true; }
      if (got_p && got_k) { break; }
    }
    sh[0] = max(max(wt_p, wt_k), max(min_p, 0.0f));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float wt = sh[0];

  // 4) exact kept = sum of weights at or above the chosen threshold.
  float kept = 0.0f;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float w = float(ws[i]);
    if (w >= wt) { kept += w; }
  }
  tg[tid] = kept;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nthg >> 1; s > 0; s >>= 1) {
    if (tid < s) { tg[tid] += tg[tid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  kept_out = tg[0];
  wt_out = wt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Gumbel-max argmax reduction over (tg=score, tgi=index), lowest-index tie
// break -- the exact reduction sample_topp_f16 uses for its final draw.
#define LC_ARGMAX_REDUCE(tg, tgi, tid, nthg)                                 \
  do {                                                                       \
    for (uint s = (nthg) >> 1; s > 0; s >>= 1) {                             \
      if ((tid) < s) {                                                       \
        const bool take = (tg)[(tid) + s] > (tg)[(tid)] ||                   \
            ((tg)[(tid) + s] == (tg)[(tid)] && (tgi)[(tid) + s] >= 0 &&      \
             ((tgi)[(tid)] < 0 || (tgi)[(tid) + s] < (tgi)[(tid)]));         \
        if (take) { (tg)[(tid)] = (tg)[(tid) + s];                          \
                    (tgi)[(tid)] = (tgi)[(tid) + s]; }                       \
      }                                                                      \
      threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                        \
  } while (0)

// Pure Gumbel-max nucleus sample of ONE logit row -> out_id[0]. Used for the
// verifier bonus and for the new MTP drafts (q1 / depth-2 q2).
//   0:logits 1:out_id 2:V 3:temp 4:top_p 5:seed 6:ws 7:n_iter 8:top_k 9:min_p
kernel void lc_sample_f16(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device int*             out_id [[buffer(1)]],
    constant int&           V      [[buffer(2)]],
    constant float&         temp   [[buffer(3)]],
    constant float&         top_p  [[buffer(4)]],
    constant uint&          seed   [[buffer(5)]],
    device VPIPE_ELT*       ws     [[buffer(6)]],
    constant int&           n_iter [[buffer(7)]],
    constant int&           top_k  [[buffer(8)]],
    constant float&         min_p  [[buffer(9)]],
    constant int&           use_hist [[buffer(10)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg[256];
  threadgroup int   tgi[256];
  threadgroup atomic_uint hist[LC_HIST_B];
  threadgroup float sh[1];
  const float inv_t = 1.0f / max(temp, 1e-6f);
  float wt = 0.0f, kept = 0.0f;
  if (use_hist) {
    lc_nucleus_hist(logits, ws, V, inv_t, top_p, top_k, min_p,
                    tg, tgi, hist, sh, tid, nthg, wt, kept);
  } else {
    lc_nucleus_stage(logits, ws, V, inv_t, top_p, top_k, min_p, n_iter,
                     tg, tgi, tid, nthg, wt, kept);
  }
  (void)kept;
  float best = -INFINITY; int bi = -1;
  for (uint i = tid; i < (uint)V; i += nthg) {
    if (float(ws[i]) >= wt) {
      const float e = float(logits[i]);
      const float u = vpipe_hash_u01(seed ^ (i * 2654435761u + 1u));
      const float g = -log(-log(u + 1e-20f) + 1e-20f);
      const float score = e * inv_t + g;
      if (score > best) { best = score; bi = (int)i; }
    }
  }
  tg[tid] = best; tgi[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  LC_ARGMAX_REDUCE(tg, tgi, tid, nthg);
  if (tid == 0) { out_id[0] = tgi[0] >= 0 ? tgi[0] : 0; }
}

// Leviathan-Chen accept-or-residual for ONE draft position. Builds the nucleus
// distributions of the verifier (p) and the carried drafter (q), then accepts
// the draft `d` with prob min(1, p(d)/q(d)); on reject it Gumbel-max samples the
// residual norm(max(0, p - q)), falling back to a draw from p when the residual
// is empty (matches the host's sample(p) fallback). Output: out_id[0].
//   0:plogits 1:qlogits 2:out_id 3:V 4:temp 5:top_p 6:seed
//   7:wsp 8:wsq 9:n_iter 10:top_k 11:min_p 12:draft
kernel void lc_accept_f16(
    const device VPIPE_ELT* plogits [[buffer(0)]],
    const device VPIPE_ELT* qlogits [[buffer(1)]],
    device int*             out_id  [[buffer(2)]],
    constant int&           V       [[buffer(3)]],
    constant float&         temp    [[buffer(4)]],
    constant float&         top_p   [[buffer(5)]],
    constant uint&          seed    [[buffer(6)]],
    device VPIPE_ELT*       wsp     [[buffer(7)]],
    device VPIPE_ELT*       wsq     [[buffer(8)]],
    constant int&           n_iter  [[buffer(9)]],
    constant int&           top_k   [[buffer(10)]],
    constant float&         min_p   [[buffer(11)]],
    constant int&           draft   [[buffer(12)]],
    constant int&           use_hist [[buffer(13)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint nthg [[threads_per_threadgroup]])
{
  threadgroup float tg[256];
  threadgroup int   tgi[256];
  threadgroup atomic_uint hist[LC_HIST_B];
  threadgroup float sh[1];
  const float inv_t = 1.0f / max(temp, 1e-6f);
  float wt_p = 0.0f, kept_p = 0.0f, wt_q = 0.0f, kept_q = 0.0f;
  if (use_hist) {
    lc_nucleus_hist(plogits, wsp, V, inv_t, top_p, top_k, min_p,
                    tg, tgi, hist, sh, tid, nthg, wt_p, kept_p);
    lc_nucleus_hist(qlogits, wsq, V, inv_t, top_p, top_k, min_p,
                    tg, tgi, hist, sh, tid, nthg, wt_q, kept_q);
  } else {
    lc_nucleus_stage(plogits, wsp, V, inv_t, top_p, top_k, min_p, n_iter,
                     tg, tgi, tid, nthg, wt_p, kept_p);
    lc_nucleus_stage(qlogits, wsq, V, inv_t, top_p, top_k, min_p, n_iter,
                     tg, tgi, tid, nthg, wt_q, kept_q);
  }

  // p(d), q(d) -> accept probability min(1, p/q). All threads read the same
  // device weights + seed so the accept branch is threadgroup-uniform.
  const float wpd = float(wsp[draft]);
  const float wqd = float(wsq[draft]);
  const float pd = (wpd >= wt_p && kept_p > 0.0f) ? wpd / kept_p : 0.0f;
  const float qd = (wqd >= wt_q && kept_q > 0.0f) ? wqd / kept_q : 0.0f;
  const float acc = (qd > 0.0f) ? min(1.0f, pd / qd) : 1.0f;
  const float u = vpipe_hash_u01(seed);
  if (u < acc) {                 // accept the draft (uniform branch)
    if (tid == 0) { out_id[0] = draft; }
    return;
  }
  // reject -> residual Gumbel-max over r_i = max(0, p_i - q_i); track the
  // p-only argmax in the SAME pass as the empty-residual fallback.
  const uint rseed = (seed * 2654435761u + 1u);
  float bR = -INFINITY; int iR = -1;
  float bP = -INFINITY; int iP = -1;
  for (uint i = tid; i < (uint)V; i += nthg) {
    const float wp = float(wsp[i]);
    const float wq = float(wsq[i]);
    const float pi = (wp >= wt_p && kept_p > 0.0f) ? wp / kept_p : 0.0f;
    const float qi = (wq >= wt_q && kept_q > 0.0f) ? wq / kept_q : 0.0f;
    const float uu = vpipe_hash_u01(rseed ^ (i * 2654435761u + 1u));
    const float g = -log(-log(uu + 1e-20f) + 1e-20f);
    const float r = pi - qi;
    if (r > 0.0f) {
      const float sc = log(r) + g;
      if (sc > bR) { bR = sc; iR = (int)i; }
    }
    if (pi > 0.0f) {
      const float sc = log(pi) + g;
      if (sc > bP) { bP = sc; iP = (int)i; }
    }
  }
  tg[tid] = bR; tgi[tid] = iR;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  LC_ARGMAX_REDUCE(tg, tgi, tid, nthg);
  const int   resid_id   = tgi[0];
  const bool  have_resid = tgi[0] >= 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (have_resid) {
    if (tid == 0) { out_id[0] = resid_id; }
    return;
  }
  // residual empty -> draw from p (its argmax under the same Gumbel noise).
  tg[tid] = bP; tgi[tid] = iP;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  LC_ARGMAX_REDUCE(tg, tgi, tid, nthg);
  if (tid == 0) { out_id[0] = tgi[0] >= 0 ? tgi[0] : 0; }
}

// Batched twin of lc_sample_f16: ROWS independent rows in ONE dispatch (grid.y
// = rows), each row its own threadgroup -> the rows run CONCURRENTLY across GPU
// cores (vs serial single-row dispatches that each leave the memory system
// under-fed). Per-row logits/ws base = row*V; per-row seed = seeds[row];
// out_id[row]. ws MUST be sized [rows*V] (disjoint per-row scratch).
//   0:logits[rows*V] 1:out_id[rows] 2:V 3:temp 4:top_p 5:seeds[rows]
//   6:ws[rows*V] 7:n_iter 8:top_k 9:min_p
kernel void lc_sample_batch_f16(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device int*             out_id [[buffer(1)]],
    constant int&           V      [[buffer(2)]],
    constant float&         temp   [[buffer(3)]],
    constant float&         top_p  [[buffer(4)]],
    const device uint*      seeds  [[buffer(5)]],
    device VPIPE_ELT*       ws     [[buffer(6)]],
    constant int&           n_iter [[buffer(7)]],
    constant int&           top_k  [[buffer(8)]],
    constant float&         min_p  [[buffer(9)]],
    constant int&           use_hist [[buffer(10)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint nthg  [[threads_per_threadgroup]],
    uint tgrow [[threadgroup_position_in_grid]])
{
  threadgroup float tg[256];
  threadgroup int   tgi[256];
  threadgroup atomic_uint hist[LC_HIST_B];
  threadgroup float sh[1];
  const device VPIPE_ELT* lrow = logits + (ulong)tgrow * (ulong)V;
  device VPIPE_ELT*       wrow = ws + (ulong)tgrow * (ulong)V;
  const uint seed = seeds[tgrow];
  const float inv_t = 1.0f / max(temp, 1e-6f);
  float wt = 0.0f, kept = 0.0f;
  if (use_hist) {
    lc_nucleus_hist(lrow, wrow, V, inv_t, top_p, top_k, min_p,
                    tg, tgi, hist, sh, tid, nthg, wt, kept);
  } else {
    lc_nucleus_stage(lrow, wrow, V, inv_t, top_p, top_k, min_p, n_iter,
                     tg, tgi, tid, nthg, wt, kept);
  }
  (void)kept;
  float best = -INFINITY; int bi = -1;
  for (uint i = tid; i < (uint)V; i += nthg) {
    if (float(wrow[i]) >= wt) {
      const float e = float(lrow[i]);
      const float u = vpipe_hash_u01(seed ^ (i * 2654435761u + 1u));
      const float g = -log(-log(u + 1e-20f) + 1e-20f);
      const float score = e * inv_t + g;
      if (score > best) { best = score; bi = (int)i; }
    }
  }
  tg[tid] = best; tgi[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  LC_ARGMAX_REDUCE(tg, tgi, tid, nthg);
  if (tid == 0) { out_id[tgrow] = tgi[0] >= 0 ? tgi[0] : 0; }
}

// MOSS local/depth-transformer single-query causal attention step. The
// current token's q/k/v (post-RoPE) are [n_head*head_dim]; the K/V caches are
// [n_head, pmax, head_dim]. Writes k/v into slot `pos`, then attends over keys
// 0..pos (causal), scale 1/sqrt(head_dim), and writes out[n_head*head_dim].
// One threadgroup per head, head_dim threads/group (head_dim<=128, pos<=15).
//   0:q 1:k_cur 2:v_cur 3:Kc 4:Vc 5:out 6:n_head 7:head_dim 8:pmax 9:pos
//   grid {head_dim, n_head, 1}, tg {head_dim, 1, 1}.
kernel void local_attn_step_f16(
    const device VPIPE_ELT* q     [[buffer(0)]],
    const device VPIPE_ELT* k_cur [[buffer(1)]],
    const device VPIPE_ELT* v_cur [[buffer(2)]],
    device VPIPE_ELT*       Kc    [[buffer(3)]],
    device VPIPE_ELT*       Vc    [[buffer(4)]],
    device VPIPE_ELT*       out   [[buffer(5)]],
    constant int& n_head   [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& pmax     [[buffer(8)]],
    constant int& pos      [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
  const int h = (int)tgid.y;
  const int d = (int)ltid.x;
  const int D = head_dim;
  if (h >= n_head || d >= D) { return; }

  // Store the current k/v into the per-head cache slot `pos`.
  const int slot = (h * pmax + pos) * D;
  Kc[slot + d] = k_cur[h * D + d];
  Vc[slot + d] = v_cur[h * D + d];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float qd = (float)q[h * D + d];
  const float scale = rsqrt((float)D);
  threadgroup float red[128];
  threadgroup float sc[16];

  for (int j = 0; j <= pos; ++j) {
    red[d] = qd * (float)Kc[(h * pmax + j) * D + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d == 0) {
      float s = 0.0f;
      for (int i = 0; i < D; ++i) { s += red[i]; }
      sc[j] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (d == 0) {
    float m = -INFINITY;
    for (int j = 0; j <= pos; ++j) { m = max(m, sc[j]); }
    float den = 0.0f;
    for (int j = 0; j <= pos; ++j) { sc[j] = exp(sc[j] - m); den += sc[j]; }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    for (int j = 0; j <= pos; ++j) { sc[j] *= inv; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc = 0.0f;
  for (int j = 0; j <= pos; ++j) {
    acc += sc[j] * (float)Vc[(h * pmax + j) * D + d];
  }
  out[h * D + d] = (VPIPE_ELT)acc;
}

// ring_append_f16 -- scatter T new per-head rows into a windowed KV ring.
// Streaming codec decode keeps each layer's K/V in a ring of `ring_cap`
// (== the attention window); logical frame (pos+i) lands at physical slot
// (pos+i) % ring_cap, matching sdpa_causal_window_f16's ring addressing.
// src [heads, T, hd] contiguous; dst [heads, ring_cap, hd].
//   0:src 1:dst 2:heads 3:T 4:hd 5:ring_cap 6:pos
// grid (hd, T, heads); one thread per element.
kernel void ring_append_f16(
    const device VPIPE_ELT* src      [[buffer(0)]],
    device VPIPE_ELT*       dst      [[buffer(1)]],
    constant int&      heads    [[buffer(2)]],
    constant int&      T        [[buffer(3)]],
    constant int&      hd       [[buffer(4)]],
    constant int&      ring_cap [[buffer(5)]],
    constant int&      pos      [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]])
{
  const int e = (int)tid.x, i = (int)tid.y, h = (int)tid.z;
  if (e >= hd || i >= T || h >= heads) { return; }
  const int slot = (pos + i) % ring_cap;
  dst[((uint)h * ring_cap + slot) * hd + e] =
      src[((uint)h * T + i) * hd + e];
}

// ---- two-pass group norm (large activations) ---------------------------
// group_norm_f16 above runs ONE threadgroup per group: on a VAE activation
// of ~100M elements that is ~8K threads and a few GB/s, and it reads a
// group's channel slice (Cg of every C) so the reads are scattered as well
// as serial. These three split it -- per-channel partial sums over row
// blocks, a tiny reduce to per-group mean/inv, then a fully parallel apply
// -- and read CONSECUTIVE channels of a row per lane, so both passes are
// coalesced. Same result, same [rows, C] channel-last layout.
//
//   stats: 0:x 1:part(f32)[NB][2*C] 2:rows 3:C 4:NB
//   grid {256*NB}, tg {256}: block b reduces rows [b*chunk, (b+1)*chunk).
kernel void group_norm_chan_stats_f16(
    const device VPIPE_ELT* x    [[buffer(0)]],
    device float*           part [[buffer(1)]],
    constant int&           rows [[buffer(2)]],
    constant int&           C    [[buffer(3)]],
    constant int&           NB   [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]])
{
  const int  b    = (int)tgid.x;
  const uint lid  = ltid.x;
  const uint tgsz = tptg.x;
  const int  chunk = (rows + NB - 1) / NB;
  const int  rlo = b * chunk;
  const int  rhi = min(rlo + chunk, rows);
  for (int c0 = 0; c0 < C; c0 += (int)tgsz) {
    const int c = c0 + (int)lid;
    if (c >= C) { break; }
    float s = 0.0f, sq = 0.0f;
    for (int r = rlo; r < rhi; ++r) {
      const float v = float(x[(long)r * (long)C + (long)c]);
      s += v; sq += v * v;
    }
    part[(long)b * 2 * (long)C + (long)c]              = s;
    part[(long)b * 2 * (long)C + (long)C + (long)c]    = sq;
  }
}

//   reduce: 0:part 1:stats(f32)[2*G] 2:rows 3:C 4:G 5:NB 6:eps
//   grid {256*G}, tg {256}: one threadgroup per group. NB*Cg is small.
kernel void group_norm_group_stats_f32(
    const device float* part  [[buffer(0)]],
    device float*       stats [[buffer(1)]],
    constant int&       rows  [[buffer(2)]],
    constant int&       C     [[buffer(3)]],
    constant int&       G     [[buffer(4)]],
    constant int&       NB    [[buffer(5)]],
    constant float&     eps   [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]])
{
  const int  g    = (int)tgid.x;
  const uint lid  = ltid.x;
  const uint tgsz = tptg.x;
  const int  Cg   = C / G;
  const int  c0   = g * Cg;
  threadgroup float ssum[256];
  threadgroup float ssq[256];
  float s = 0.0f, sq = 0.0f;
  for (int i = (int)lid; i < NB * Cg; i += (int)tgsz) {
    const int b  = i / Cg;
    const int cc = i % Cg;
    s  += part[(long)b * 2 * (long)C + (long)(c0 + cc)];
    sq += part[(long)b * 2 * (long)C + (long)C + (long)(c0 + cc)];
  }
  ssum[lid] = s; ssq[lid] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = tgsz / 2; st > 0; st >>= 1) {
    if (lid < st) { ssum[lid] += ssum[lid + st]; ssq[lid] += ssq[lid + st]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    const float total = (float)rows * (float)Cg;
    const float mean  = ssum[0] / total;
    const float var   = ssq[0] / total - mean * mean;
    stats[2 * g]     = mean;
    stats[2 * g + 1] = 1.0f / sqrt(var + eps);
  }
}

//   apply: 0:x 1:gamma 2:beta 3:out 4:stats 5:C 6:G
//   2D grid {C, rows}: gid = row*C + col (a 1D grid overflows past ~3K px).
kernel void group_norm_apply_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* gamma [[buffer(1)]],
    const device VPIPE_ELT* beta  [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    const device float*     stats [[buffer(4)]],
    constant int&           C     [[buffer(5)]],
    constant int&           G     [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  if (c >= C) { return; }
  const long idx = (long)gid.y * (long)C + (long)c;
  const int  g   = c / (C / G);
  const float m  = stats[2 * g];
  const float iv = stats[2 * g + 1];
  out[idx] = VPIPE_ELT((float(x[idx]) - m) * iv * float(gamma[c])
                       + float(beta[c]));
}

// ---- direct 3x3 convolution for a SMALL output-channel count -----------
// Channel-last [H, W, cin] -> [H, W, cout], stride 1, pad 1.
//
// Why this exists: the NAX hardware conv needs cout % 64 == 0, and the
// im2col+GEMM fallback materializes a [H*W, 9*cin] buffer. Every VAE ends
// in a conv to 3 (or 2*latent) channels AT FULL RESOLUTION, where that
// buffer is ~1.8 GB for a ~5 GFLOP convolution -- pure materialization
// cost. MEASURED 764 ms for FLUX.2's 128->3 at 1024x768.
//
// Here each thread owns one output pixel and keeps its whole (small) output
// vector in registers, so the activation is read once per tap instead of
// being written out and read back. The 3x3 window overlaps its neighbours by
// 2/3, so the redundant taps come from cache rather than DRAM; the weight is
// a few KB and stays resident. Reads are the dominant term, which makes this
// bandwidth-bound rather than materialization-bound.
//
// Output channels are processed COUT_CHUNK at a time so the accumulators
// stay in registers; `nt` is uniform across the threadgroup, so the tail
// chunk costs no divergence. `whwio` is the same [9][cin][cout] layout the
// hardware-conv path uses (out-channel fastest), so no extra weight twin.
//   0:x 1:whwio 2:bias 3:out 4:Wd 5:Hd 6:cin 7:cout 8:has_bias
//   grid {Wd, Hd}: one thread per output pixel.
kernel void conv3x3_hwc_small_cout_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* whwio [[buffer(1)]],
    const device VPIPE_ELT* bias  [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&           Wd       [[buffer(4)]],
    constant int&           Hd       [[buffer(5)]],
    constant int&           cin      [[buffer(6)]],
    constant int&           cout_n   [[buffer(7)]],
    constant int&           has_bias [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int px = (int)gid.x;
  const int py = (int)gid.y;
  if (px >= Wd || py >= Hd) { return; }
  constexpr int COUT_CHUNK = 4;
  for (int o0 = 0; o0 < cout_n; o0 += COUT_CHUNK) {
    const int nt = min(COUT_CHUNK, cout_n - o0);
    float acc[COUT_CHUNK] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int ky = 0; ky < 3; ++ky) {
      const int sy = py + ky - 1;
      if (sy < 0 || sy >= Hd) { continue; }        // zero pad
      for (int kx = 0; kx < 3; ++kx) {
        const int sx = px + kx - 1;
        if (sx < 0 || sx >= Wd) { continue; }
        const long xb = ((long)sy * (long)Wd + (long)sx) * (long)cin;
        const long wb = (long)(ky * 3 + kx) * (long)cin * (long)cout_n
                        + (long)o0;
        for (int ci = 0; ci < cin; ++ci) {
          const float v = float(x[xb + (long)ci]);
          const long wo = wb + (long)ci * (long)cout_n;
          for (int t = 0; t < nt; ++t) { acc[t] += v * float(whwio[wo + t]); }
        }
      }
    }
    const long ob = ((long)py * (long)Wd + (long)px) * (long)cout_n + (long)o0;
    for (int t = 0; t < nt; ++t) {
      const float r = acc[t] + (has_bias != 0 ? float(bias[o0 + t]) : 0.0f);
      out[ob + (long)t] = VPIPE_ELT(r);
    }
  }
}

// ---- Wan VAE: causal 3D convolution ---------------------------------
//
// The Wan video VAE (AutoencoderKLWan) is the net the Qwen-Image VAE above
// is the single-frame specialization of: same channel-last [H*W, C]
// interior, same 3x3 spatial kernels, but every conv is a CAUSAL conv3d --
// output frame t reads input frames t-2, t-1, t, with zero frames before
// the start of the sequence.
//
// Rather than materialize the 3-frame window (which would mean copying the
// two carried-over frames in front of every chunk, for every conv), the
// three taps are bound as three SEPARATE frame views: t0/t1/t2 each point
// at one [H*W, C] frame -- a byte offset into the activation for a tap that
// lives in the current chunk, and into the layer's 2-frame carry for one
// that does not. The caller therefore pays no copy at all, and the "which
// frames exist yet" logic stays on the host where the chunking lives.
//
// `tap_valid` is a 3-bit mask of which taps exist: a tap before the start
// of the sequence reads as zero. It is a mask rather than a bound zero
// frame because that frame would have to be as large as the widest
// activation -- ~177 MB at 720p -- to serve every layer, which is more
// memory than the rest of the decode's working set.
//
// out[r, ((kt*3+ky)*3+kx)*C + c] = t{kt}[(y+ky-1)*W + (x+kx-1), c], zero
// outside the spatial border, with r = y*W + x. A dense_gemm over
// W[Cout, 27*C] then realizes the causal conv3d. Row-tiled like its 2D
// twin (im2col_hwc_3x3_tiled_f16): only output rows [row_off,
// row_off+row_cnt) are written, into a TILE-LOCAL out, so the col scratch
// is one band rather than the full [H*W, 27*C] (three times the 2D
// scratch, so the banding matters more here, not less).
//   0:t0 1:t1 2:t2 (each [H*W,C]) 3:out[row_cnt,27*C]
//   4:H 5:W 6:C 7:row_off 8:row_cnt 9:tap_valid.
//   grid {27*C, row_cnt}.
kernel void im2col_hwc_3x3x3_tiled_f16(
    const device VPIPE_ELT* t0  [[buffer(0)]],
    const device VPIPE_ELT* t1  [[buffer(1)]],
    const device VPIPE_ELT* t2  [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&      H       [[buffer(4)]],
    constant int&      W       [[buffer(5)]],
    constant int&      C       [[buffer(6)]],
    constant int&      row_off [[buffer(7)]],
    constant int&      row_cnt [[buffer(8)]],
    constant int&      tap_valid [[buffer(9)]],
    uint2 tpig [[thread_position_in_grid]],
    uint2 tpg  [[threads_per_grid]])
{
  // 32-bit index math on the documented 2D grid -- see the note on
  // im2col_hwc_3x3_f16 for why the 64-bit div/mod form is the slow path.
  const uint cols = (uint)(27 * C);
  ulong gid;
  uint c, j, r;
  if (tpg.x == cols) {                       // 2D {27*C, row_cnt}
    if ((uint)tpig.y >= (uint)row_cnt) { return; }
    c = (uint)tpig.x % (uint)C;
    j = (uint)tpig.x / (uint)C;              // (kt*3 + ky)*3 + kx
    r = (uint)row_off + (uint)tpig.y;        // global out row
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
  } else {
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
    if (gid >= (ulong)(uint)row_cnt * cols) { return; }
    c = (uint)(gid % (uint)C);
    j = (uint)((gid / (uint)C) % 27u);
    r = (uint)row_off + (uint)(gid / (ulong)cols);
  }
  const uint kt = j / 9u;
  const uint ks = j % 9u;
  const int x = (int)(r % (uint)W);
  const int y = (int)(r / (uint)W);
  const int sy = y + (int)(ks / 3u) - 1;
  const int sx = x + (int)(ks % 3u) - 1;
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (sy >= 0 && sy < H && sx >= 0 && sx < W &&
      (((uint)tap_valid >> kt) & 1u) != 0u) {
    const ulong si = ((ulong)sy * W + sx) * (uint)C + c;
    val = (kt == 0u) ? t0[si] : (kt == 1u) ? t1[si] : t2[si];
  }
  out[gid] = val;
}

// Same causal 3D im2col, but with REFLECT spatial padding and an output
// grid that may be strided -- the MiniMax-H3 video encoder's convolution.
//
// Two things separate it from im2col_hwc_3x3x3_tiled_f16 above, and both are
// silent if got wrong: H3 pads spatially by REFLECTION rather than with
// zeros (a zero-padded border darkens the frame edge into the latent, which
// survives the decode as a vignette), and its stride-2 downsample carries no
// symmetric padding at all -- the reference pads one row/column onto the
// BOTTOM and RIGHT only and then runs a valid convolution, so the output is
// ceil(size/2) and the sampling grid is offset by half a tap from what a
// symmetric pad would give.
//
// Both cases are the one expression `sy = y*stride + ky - pad_lo` reflected
// into [0, H): stride 1 with pad_lo 1 is the symmetric pad, stride 2 with
// pad_lo 0 is the asymmetric one (`sy` then only ever runs past the END of
// the axis, which is exactly the row the bottom/right pad holds).
//
// The temporal axis is unchanged from the zero-padded twin: causal padding
// stays a `tap_valid` mask over three separately-bound frame views, and the
// host picks tap kt as input frame `t*temporal_stride + kt - 2`.
//   0:t0 1:t1 2:t2 (each [H*W,C]) 3:out[row_cnt,27*C]
//   4:H 5:W 6:C 7:row_off 8:row_cnt 9:tap_valid 10:Wo 11:stride 12:pad_lo
//   grid {27*C, row_cnt}; rows index the OUTPUT grid, r = y*Wo + x.
kernel void im2col_hwc_3x3x3_reflect_tiled_f16(
    const device VPIPE_ELT* t0  [[buffer(0)]],
    const device VPIPE_ELT* t1  [[buffer(1)]],
    const device VPIPE_ELT* t2  [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&      H       [[buffer(4)]],
    constant int&      W       [[buffer(5)]],
    constant int&      C       [[buffer(6)]],
    constant int&      row_off [[buffer(7)]],
    constant int&      row_cnt [[buffer(8)]],
    constant int&      tap_valid [[buffer(9)]],
    constant int&      Wo      [[buffer(10)]],
    constant int&      stride  [[buffer(11)]],
    constant int&      pad_lo  [[buffer(12)]],
    uint2 tpig [[thread_position_in_grid]],
    uint2 tpg  [[threads_per_grid]])
{
  const uint cols = (uint)(27 * C);
  ulong gid;
  uint c, j, r;
  if (tpg.x == cols) {                       // 2D {27*C, row_cnt}
    if ((uint)tpig.y >= (uint)row_cnt) { return; }
    c = (uint)tpig.x % (uint)C;
    j = (uint)tpig.x / (uint)C;              // (kt*3 + ky)*3 + kx
    r = (uint)row_off + (uint)tpig.y;        // global out row
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
  } else {
    gid = (ulong)tpig.y * cols + tpig.x;     // tile-local
    if (gid >= (ulong)(uint)row_cnt * cols) { return; }
    c = (uint)(gid % (uint)C);
    j = (uint)((gid / (uint)C) % 27u);
    r = (uint)row_off + (uint)(gid / (ulong)cols);
  }
  const uint kt = j / 9u;
  const uint ks = j % 9u;
  const int x = (int)(r % (uint)Wo);
  const int y = (int)(r / (uint)Wo);
  int sy = y * stride + (int)(ks / 3u) - pad_lo;
  int sx = x * stride + (int)(ks % 3u) - pad_lo;
  // Reflection WITHOUT repeating the edge sample, matching torch's
  // `F.pad(mode="reflect")`: index -1 is row 1, index H is row H-2.
  if (sy < 0) { sy = -sy; }
  if (sy >= H) { sy = 2 * H - 2 - sy; }
  if (sx < 0) { sx = -sx; }
  if (sx >= W) { sx = 2 * W - 2 - sx; }
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (sy >= 0 && sy < H && sx >= 0 && sx < W &&
      (((uint)tap_valid >> kt) & 1u) != 0u) {
    const ulong si = ((ulong)sy * W + sx) * (uint)C + c;
    val = (kt == 0u) ? t0[si] : (kt == 1u) ? t1[si] : t2[si];
  }
  out[gid] = val;
}

// Channel-wise concatenation of three frame views, out[r, kt*C + c] =
// t{kt}[r, c] over `rows` rows. Pairs with a dense_gemm over W[Cout, 3*C]
// to realize the Wan VAE's TEMPORAL-only convolutions -- the (3,1,1)
// `time_conv` of each upsample3d / downsample3d, which mixes three frames
// at a single pixel and so needs no spatial halo at all. Same three-view
// binding as im2col_hwc_3x3x3_tiled_f16 (see there), and the same reason:
// the taps live in different buffers and copying them together would cost
// more than the convolution. `tap_valid` masks taps before the start of
// the sequence, as in im2col_hwc_3x3x3_tiled_f16.
//   0:t0 1:t1 2:t2 (each [rows,C]) 3:out[rows,3*C] 4:C 5:rows 6:tap_valid.
//   grid {3*C, rows}.
kernel void concat3_frames_f16(
    const device VPIPE_ELT* t0  [[buffer(0)]],
    const device VPIPE_ELT* t1  [[buffer(1)]],
    const device VPIPE_ELT* t2  [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&      C    [[buffer(4)]],
    constant int&      rows [[buffer(5)]],
    constant int&      tap_valid [[buffer(6)]],
    uint2 tpig [[thread_position_in_grid]])
{
  if ((uint)tpig.y >= (uint)rows || (uint)tpig.x >= (uint)(3 * C)) { return; }
  const uint c  = (uint)tpig.x % (uint)C;
  const uint kt = (uint)tpig.x / (uint)C;
  const ulong si = (ulong)tpig.y * (uint)C + c;
  VPIPE_ELT val = (VPIPE_ELT)0;
  if ((((uint)tap_valid >> kt) & 1u) != 0u) {
    val = (kt == 0u) ? t0[si] : (kt == 1u) ? t1[si] : t2[si];
  }
  out[(ulong)tpig.y * (uint)(3 * C) + tpig.x] = val;
}

// Interleave the two halves of a doubled-channel map into twice as many
// FRAMES: given in[t, r, 2*C] laid out as the Wan upsample3d time_conv
// writes it (the first C channels are the even output frame, the last C
// the odd one), produce out[2t + h, r, c] = in[t, r, h*C + c]. This is the
// reshape-and-stack the reference does after the temporal upsample, and it
// is the one place frame COUNT changes inside a chunk.
//   0:in[T*rows,2*C] 1:out[2*T*rows,C] 2:C 3:rows 4:T.  grid {C, rows, 2*T}.
kernel void wan_time_unshuffle_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      C    [[buffer(2)]],
    constant int&      rows [[buffer(3)]],
    constant int&      T    [[buffer(4)]],
    uint3 tpig [[thread_position_in_grid]])
{
  if ((uint)tpig.x >= (uint)C || (uint)tpig.y >= (uint)rows ||
      (uint)tpig.z >= (uint)(2 * T)) {
    return;
  }
  const uint t = (uint)tpig.z / 2u;          // source frame
  const uint h = (uint)tpig.z % 2u;          // which half
  const ulong si = ((ulong)t * (uint)rows + tpig.y) * (uint)(2 * C)
                   + h * (uint)C + tpig.x;
  const ulong di = ((ulong)tpig.z * (uint)rows + tpig.y) * (uint)C + tpig.x;
  out[di] = in[si];
}

// ---- umT5 encoder self-attention ------------------------------------
//
// Full bidirectional attention with T5's additive RELATIVE-POSITION bias,
// which is why it cannot ride any of the sdpa kernels: those take no bias
// term, and here the bias is what carries all the position information --
// there is no rotary embedding in T5 at all.
//
// Two more T5 peculiarities are baked in. There is NO 1/sqrt(d) scaling of
// the scores (T5 folds that into initialization), and the bias is looked
// up per (head, query, key) from a bucketed distance rather than being a
// dense table: bias[h][i][j] = rel[bucket(j - i)][h] over 32 buckets. The
// bucket for each (i, j) is precomputed on the HOST and passed in
// `buckets` -- both because it is loop-invariant across heads and layers,
// and because the reference derives it through a float32 log whose
// truncation lands on a bucket boundary often enough that reproducing it
// exactly matters more than saving 256 KB.
//
// Keys at or past `n_valid` are masked out: the Wan pipeline pads the
// prompt to a fixed 512 tokens and passes the attention mask, so the pad
// must not participate.
//
// One threadgroup per (query row, head). Scores live in threadgroup
// memory, so the sequence is capped at UMT5_MAX_L -- the host checks it.
//   0:q 1:k 2:v (each [L, H*D]) 3:rel[32,H] 4:buckets[L*L] (uchar)
//   5:out[L,H*D] 6:L 7:H 8:D 9:n_valid.  grid {UMT5_TG, L, H}.
#define UMT5_TG    256
#define UMT5_MAX_L 1024
kernel void umt5_attn_f16(
    const device VPIPE_ELT* q       [[buffer(0)]],
    const device VPIPE_ELT* k       [[buffer(1)]],
    const device VPIPE_ELT* v       [[buffer(2)]],
    const device VPIPE_ELT* rel     [[buffer(3)]],
    const device uchar*     buckets [[buffer(4)]],
    device VPIPE_ELT*       out     [[buffer(5)]],
    constant int&      L       [[buffer(6)]],
    constant int&      H       [[buffer(7)]],
    constant int&      D       [[buffer(8)]],
    constant int&      n_valid [[buffer(9)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 ltid  [[thread_position_in_threadgroup]])
{
  const uint i = tgid.y;                 // query row
  const uint h = tgid.z;                 // head
  const uint t = ltid.x;
  if ((int)i >= L || (int)h >= H || L > UMT5_MAX_L) { return; }

  threadgroup float scores[UMT5_MAX_L];
  threadgroup float red[UMT5_TG];
  threadgroup float qv[128];             // D <= 128
  threadgroup float part[4][64];         // pass-3 partials, D <= 64 lanes

  const uint hd   = (uint)H * (uint)D;
  const uint qoff = i * hd + h * (uint)D;
  for (uint d = t; d < (uint)D && d < 128u; d += UMT5_TG) {
    qv[d] = (float)q[qoff + d];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // 1. scores, with the relative-position bias and the pad mask.
  float local_max = -INFINITY;
  for (uint j = t; j < (uint)L; j += UMT5_TG) {
    float s;
    if ((int)j >= n_valid) {
      s = -INFINITY;
    } else {
      const uint koff = j * hd + h * (uint)D;
      float acc = 0.0f;
      for (int d = 0; d < D; ++d) { acc += qv[d] * (float)k[koff + d]; }
      const uint b = (uint)buckets[i * (uint)L + j];
      s = acc + (float)rel[b * (uint)H + h];
    }
    scores[j] = s;
    local_max = metal::max(local_max, s);
  }
  red[t] = local_max;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint s = UMT5_TG / 2; s > 0; s >>= 1) {
    if (t < s) { red[t] = metal::max(red[t], red[t + s]); }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  const float mx = red[0];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // 2. exponentiate + normalize (in float, as the reference softmax does).
  float local_sum = 0.0f;
  for (uint j = t; j < (uint)L; j += UMT5_TG) {
    const float e = (scores[j] == -INFINITY) ? 0.0f
                                             : metal::precise::exp(scores[j] - mx);
    scores[j] = e;
    local_sum += e;
  }
  red[t] = local_sum;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint s = UMT5_TG / 2; s > 0; s >>= 1) {
    if (t < s) { red[t] += red[t + s]; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  const float inv = (red[0] > 0.0f) ? (1.0f / red[0]) : 0.0f;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // 3. weighted sum of V. Threads split as (j-stripe, channel) so every
  // lane works and the V reads stay contiguous across the channel axis.
  const uint dd = t % 64u;
  const uint jg = t / 64u;
  if ((int)dd < D && jg < 4u) {
    float acc = 0.0f;
    for (uint j = jg; j < (uint)L; j += 4u) {
      const float p = scores[j];
      if (p != 0.0f) { acc += p * (float)v[j * hd + h * (uint)D + dd]; }
    }
    part[jg][dd] = acc;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (t < (uint)D && t < 64u) {
    const float r = part[0][t] + part[1][t] + part[2][t] + part[3][t];
    out[i * hd + h * (uint)D + t] = VPIPE_ELT(r * inv);
  }
}

// ---------------------------------------------------------------------
// MiniMax-H3: per-ROW modulation.
//
// Every other DiT here modulates a whole activation with ONE (shift,
// scale, gate) triple, because one forward carries one timestep and one
// modality. MiniMax-H3 carries several of both in a single packed
// sequence -- generated video rows, generated audio rows and pinned
// conditioning rows all sit at different noise levels -- so the
// modulation is a TABLE and each row of the activation picks its own
// entry. `idx` is the caller's precomputed `timestep_index * 3 + tag`
// (0 video, 1 text, 2 audio), or the bare timestep index for the final
// norm, which is modality-independent.
//
// `mod` is the block's modulation table [rows, stride]; one table row
// holds all of that block's parameters back to back, so `scale_off` /
// `shift_off` / `gate_off` are column offsets within it. That matches
// the reference's `linear(silu(temb)).view(-1, 6*H).chunk(6, -1)`:
// chunking the last axis of a (rows, 6*H) matrix leaves the six
// parameters of one row contiguous, in the order shift_msa, scale_msa,
// gate_msa, shift_mlp, scale_mlp, gate_mlp.

// out[m,n] = (1 + scale[idx[m], n]) * x[m,n] + shift[idx[m], n]
//   0:x[M,N] 1:mod[*,stride] 2:idx[M] (int32) 3:out[M,N] 4:N 5:stride
//   6:scale_off 7:shift_off 8:total(=M*N).  grid (total).
kernel void adaln_modulate_idx_f16(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* mod   [[buffer(1)]],
    const device int*       idx   [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&      N          [[buffer(4)]],
    constant int&      stride     [[buffer(5)]],
    constant int&      scale_off  [[buffer(6)]],
    constant int&      shift_off  [[buffer(7)]],
    constant int&      total      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const int m = (int)gid / N;
  const int n = (int)gid % N;
  const int64_t base = (int64_t)idx[m] * stride;
  const float sc = float(mod[base + scale_off + n]);
  const float sh = float(mod[base + shift_off + n]);
  out[gid] = VPIPE_ELT((1.0f + sc) * float(x[gid]) + sh);
}

// h[m,n] += gate[idx[m], n] * sub[m,n]. In place on h.
//   0:h[M,N] 1:mod[*,stride] 2:idx[M] (int32) 3:sub[M,N] 4:N 5:stride
//   6:gate_off 7:total(=M*N).  grid (total).
kernel void gated_residual_idx_f16(
    device VPIPE_ELT*       h    [[buffer(0)]],
    const device VPIPE_ELT* mod  [[buffer(1)]],
    const device int*       idx  [[buffer(2)]],
    const device VPIPE_ELT* sub  [[buffer(3)]],
    constant int&      N         [[buffer(4)]],
    constant int&      stride    [[buffer(5)]],
    constant int&      gate_off  [[buffer(6)]],
    constant int&      total     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const int m = (int)gid / N;
  const int n = (int)gid % N;
  const int64_t base = (int64_t)idx[m] * stride;
  h[gid] = VPIPE_ELT(float(h[gid]) +
                     float(mod[base + gate_off + n]) * float(sub[gid]));
}

// SwiGLU over a FUSED [rows, 2*D] projection split into halves, VALUE
// first: out[r,d] = in[r,d] * silu(in[r,D+d]).
//
// The half order is the trap. diffusers' `SwiGLU` reads
// `hidden, gate = proj(x).chunk(2, -1)` and returns `hidden *
// silu(gate)`, so the FIRST half is the value and the SECOND is the
// gate -- the opposite of the llama/Qwen convention every other SwiGLU
// in this tree follows, where the gate leads. Feeding this kernel a
// gate-first matrix produces a plausible, wrong activation rather than
// anything that fails loudly, which is why it is a separate entry point
// instead of a flag on swiglu_f16.
//   0:in[rows,2*D] 1:out[rows,D] 2:rows 3:D.  grid (rows*D).
kernel void swiglu_split_value_first_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      rows [[buffer(2)]],
    constant int&      D    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(rows * D)) { return; }
  const int r = (int)gid / D;
  const int d = (int)gid % D;
  const int64_t base = (int64_t)r * (2 * D);
  const float v = float(in[base + d]);
  const float g = float(in[base + D + d]);
  out[gid] = VPIPE_ELT(v * (g / (1.0f + exp(-g))));
}

// The same split with the halves the OTHER way round -- GATE first:
// out[r,d] = silu(in[r,d]) * in[r,D+d].
//
// Which of the two is right for a checkpoint is not a matter of taste,
// and the two references disagree: diffusers' `SwiGLU` is value-first
// (above), while ComfyUI's `_swiglu_eager` does
// `gate, up = x.chunk(2); silu(gate) * up`. Since a wrong half order is
// silent -- it degrades every block by a fixed amount instead of
// failing -- both entry points exist so the question can be settled by
// running the model rather than by reading either reference.
//   0:in[rows,2*D] 1:out[rows,D] 2:rows 3:D.  grid (rows*D).
kernel void swiglu_split_gate_first_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      rows [[buffer(2)]],
    constant int&      D    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(rows * D)) { return; }
  const int r = (int)gid / D;
  const int d = (int)gid % D;
  const int64_t base = (int64_t)r * (2 * D);
  const float g = float(in[base + d]);
  const float v = float(in[base + D + d]);
  out[gid] = VPIPE_ELT(v * (g / (1.0f + exp(-g))));
}

