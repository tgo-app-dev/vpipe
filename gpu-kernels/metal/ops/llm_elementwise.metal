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

// out[i] = silu(gate[i]) * up[i].   0:gate 1:up 2:out 3:n
kernel void swiglu_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out   [[buffer(2)]],
    constant int&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float g = float(gate[gid]);
  const float silu = g / (1.0f + exp(-g));
  out[gid] = VPIPE_ELT(silu * float(up[gid]));
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

// Broadcast a per-column bias over the rows of a row-major [M, N] matrix,
// in place: y[m, n] += bias[n]. Used to fold the linear bias back after the
// matmul2d dense GEMM (dense_gemm_mma), which omits bias.
//   0:y[M*N] (inout) 1:bias[N] 2:N 3:total(=M*N).  grid (total).
kernel void bias_add_rows_f16(
    device VPIPE_ELT*       y    [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    constant int&      N     [[buffer(2)]],
    constant int&      total [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
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
kernel void qmv_q6k_v2_f16(
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

// Q5_K: same full-byte tiling, plus the 5th bit from qh[l+k] (low nibble at
// bit 2*chunk, high at 2*chunk+1).
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
  const int lane  = (int)simd_lid;
  const int bi    = lane * 4;
  const int chunk = bi >> 5;
  const int l     = bi & 31;
  const int is_lo = chunk * 2, is_hi = chunk * 2 + 1;
  const int blo_bit = 2 * chunk, bhi_bit = 2 * chunk + 1;
  const int xlo   = chunk * 64 + l;
  const int xhi   = xlo + 32;
  float acc = 0.0f;
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
      acc += float(x[xb + xlo + k]) * (dlo * float(nlo) - blo);
      acc += float(x[xb + xhi + k]) * (dhi * float(nhi) - bhi);
    }
  }
  const float v = simd_sum(acc);
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
