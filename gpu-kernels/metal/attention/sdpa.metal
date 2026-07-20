// sdpa.metal -- scaled dot-product attention (causal) for Llama
// inference, online-softmax, GQA-aware. One simdgroup computes one
// (query head, query position) output vector [D] via a single pass
// over the keys it may attend to. Handles BOTH regimes:
//   decode  : n_q=1, q_offset=T_kv-1  -> attends all T_kv keys
//   prefill : n_q=n,  q_offset=0      -> query qi attends keys 0..qi
// Matches mlx::fast::scaled_dot_product_attention(q,k,v,scale,
// mask_mode="" for decode / "causal" for prefill).
//
// Layout (B=1 folded): q [H_q, n_q, D], k/v [H_kv, T_kv, D],
// out [H_q, n_q, D]. GQA: kv head = h / (H_q / H_kv). D % 32 == 0.
//
//   0:q 1:k 2:v 3:out (VPIPE_ELT)
//   4:scale(float) 5:T_kv 6:D 7:Hq 8:Hkv 9:n_q 10:q_offset (int)
//   11:kv_stride(int) -- k/v allocated seq dim (>= T_kv); lets the
//   kernel read straight from a [Hkv, MAX_SEQ, D] KV cache where the
//   valid length T_kv < MAX_SEQ. Pass kv_stride == T_kv for packed.
// grid (32, H_q, n_q); threadgroup (32,1,1).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define SDPA_MAX_PER 16  // max D/32 (supports D up to 512: Gemma-4 full-attn)
#define SDPA_MAX_G   4   // max GQA group (e4b global Hq/Hkv = 8/2 = 4); used by
                         // the kv_read_probe M=G mode (qreg/dd = per*G regs).

// Full (non-causal) bidirectional attention over contiguous K/V, for the
// Qwen3-VL vision tower (every patch attends every patch). Same layout
// and buffer contract as sdpa_causal_f16 minus the causal horizon: query
// qi attends ALL keys 0..T_kv-1. No q_offset.
//   0:q[Hq,n_q,D] 1:k/2:v[Hkv,T_kv,D] 3:out 4:scale 5:T_kv 6:D 7:Hq
//   8:Hkv 9:n_q 10:kv_stride.  grid (32, Hq, n_q); threadgroup (32,1,1).
kernel void sdpa_full_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      kv_stride [[buffer(10)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  // Each lane owns the contiguous block [lane*per, lane*per+per) of the
  // head dim. per = ceil(D/32) so D need NOT be a multiple of 32 (the
  // Qwen3.5-9B ViT has head_dim 72); lanes whose block runs past D guard
  // every access with idx < D. For D divisible by 32 (head_dim 64/256
  // etc.) per is unchanged and every guard is trivially true, so this is
  // behaviour-identical to the old D/32 form.
  const int per = (D + 31) / 32;
  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;

  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    const int idx = lane * per + p;
    qreg[p] = idx < D ? float(qh[idx]) * scale : 0.0f;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int j = 0; j < T_kv; ++j) {
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      const int idx = lane * per + p;
      if (idx < D) { dot += qreg[p] * float(kkv[(uint)j * D + idx]); }
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      const int idx = lane * per + p;
      if (idx < D) {
        acc[p] = acc[p] * corr + pj * float(vkv[(uint)j * D + idx]);
      }
    }
    m = m_new;
  }
  const float inv_l = 1.0f / l;
  for (int p = 0; p < per; ++p) {
    const int idx = lane * per + p;
    if (idx < D) {
      out[((uint)h * n_q + qi) * D + idx] = VPIPE_ELT(acc[p] * inv_l);
    }
  }
}

// Block-windowed (non-causal) attention: identical to sdpa_full_f16 but
// each query attends only to keys in its window-aligned block of `window`
// tokens -- query qi sees keys [floor(qi/window)*window,
// min(start+window, T_kv)). With window >= T_kv this is exactly full
// attention. Reproduces the Qwen3-ASR encoder's block-diagonal mask
// (window = 104 audio tokens) without materializing a [T,T] mask.
//   ... same buffers as sdpa_full_f16 ... 11:window
kernel void sdpa_window_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      kv_stride [[buffer(10)]],
    constant int&      window   [[buffer(11)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int ws = (window > 0) ? (qi / window) * window : 0;
  const int we = (window > 0) ? min(ws + window, T_kv) : T_kv;
  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;

  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int j = ws; j < we; ++j) {
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      dot += qreg[p] * float(kkv[(uint)j * D + lane * per + p]);
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      acc[p] = acc[p] * corr + pj * float(vkv[(uint)j * D + lane * per + p]);
    }
    m = m_new;
  }
  const float inv_l = 1.0f / l;
  for (int p = 0; p < per; ++p) {
    out[((uint)h * n_q + qi) * D + lane * per + p] = VPIPE_ELT(acc[p] * inv_l);
  }
}

// Variable-window (non-causal) attention: like sdpa_window_f16 but each query
// reads its window bounds [win_start[qi], win_end[qi]) from per-query buffers
// instead of a single fixed window size, so windows may have DIFFERENT lengths
// (the Qwen2.5-VL vision tower tiles a grid into 4x4-merged windows whose edge
// tiles are smaller). Uses the GUARDED per = ceil(D/32) like sdpa_full_f16, so
// head_dim need not be a multiple of 32 (the Qwen2.5-VL ViT has head_dim 80).
// Buffers mirror sdpa_full_f16; 11:win_start[n_q] 12:win_end[n_q].
kernel void sdpa_varwindow_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      kv_stride [[buffer(10)]],
    const device int*  win_start [[buffer(11)]],
    const device int*  win_end   [[buffer(12)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  (void)T_kv;   // kept for sdpa_full_f16 buffer-layout parity; the per-query
                // win_start/win_end bound the loop instead of the total KV len.
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = (D + 31) / 32;
  const int ws = win_start[qi];
  const int we = win_end[qi];
  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;

  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    const int idx = lane * per + p;
    qreg[p] = idx < D ? float(qh[idx]) * scale : 0.0f;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int j = ws; j < we; ++j) {
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      const int idx = lane * per + p;
      if (idx < D) { dot += qreg[p] * float(kkv[(uint)j * D + idx]); }
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      const int idx = lane * per + p;
      if (idx < D) {
        acc[p] = acc[p] * corr + pj * float(vkv[(uint)j * D + idx]);
      }
    }
    m = m_new;
  }
  const float inv_l = 1.0f / l;
  for (int p = 0; p < per; ++p) {
    const int idx = lane * per + p;
    if (idx < D) {
      out[((uint)h * n_q + qi) * D + idx] = VPIPE_ELT(acc[p] * inv_l);
    }
  }
}

// Gemma-4 USM Conformer chunked-local attention with relative-position
// bias + logit softcap. Token-major layout q/k/v/out [T, N, Hd] (N heads,
// Hd = head_dim, row stride D = N*Hd). Each query position attends only
// within its chunk + left/right context, with a Transformer-XL relative
// shift over a sinusoidal position embedding (sin_emb [span, N, Hd]).
// Mirrors MetalGemma4AudioEncoder::encode()'s host attention exactly:
//   u = qi/W; w = qi%W; C = W + mb + mf; key ki = u*W - mb + cc, cc in
//   [0,C); causal mask (cc>=w && cc<=w+mb+mf) && 0<=ki<T.
//   ac = (q_scale*pds .* q[qi]) . (k_scale .* k[ki])
//   bd : j=w*C+cc; rr=j/(C+1); col=j%(C+1); valid when col<span && rr<W
//        && u*W+rr<T; bd = (q_scale*pds .* q[u*W+rr]) . sin_emb[col]
//   logit = softcap*tanh((ac+bd)/softcap); masked online softmax over v.
// One simdgroup per (head n=tid.y, query qi=tid.z); 32 lanes, per=Hd/32.
//   0:q 1:k 2:v 3:sin_emb 4:pds[Hd] 5:out  6:T 7:N 8:Hd 9:W 10:mb 11:mf
//   12:span 13:q_scale(float) 14:k_scale(float) 15:softcap(float)
// grid (32, N, T); threadgroup (32,1,1).
kernel void gemma_audio_local_attn_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    const device VPIPE_ELT* sin_emb  [[buffer(3)]],
    const device VPIPE_ELT* pds      [[buffer(4)]],
    device VPIPE_ELT*       out      [[buffer(5)]],
    constant int&      T        [[buffer(6)]],
    constant int&      N        [[buffer(7)]],
    constant int&      Hd       [[buffer(8)]],
    constant int&      W        [[buffer(9)]],
    constant int&      mb       [[buffer(10)]],
    constant int&      mf       [[buffer(11)]],
    constant int&      span     [[buffer(12)]],
    constant float&    q_scale  [[buffer(13)]],
    constant float&    k_scale  [[buffer(14)]],
    constant float&    softcap  [[buffer(15)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int n = (int)tid.y;
  const int qi = (int)tid.z;
  const int D = N * Hd;
  const int per = Hd / 32;
  const int u = qi / W;
  const int w = qi % W;
  const int C = W + mb + mf;

  // Scaled query [qi]: q_scale * pds[d] .* q.
  float qreg[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    const int d = lane * per + p;
    qreg[p] = float(q[(uint)qi * D + n * Hd + d]) * q_scale * float(pds[d]);
  }

  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) { acc[p] = 0.0f; }
  float m = -INFINITY, l = 0.0f;

  for (int cc = 0; cc < C; ++cc) {
    const int ki = u * W - mb + cc;
    const bool causal = (cc >= w) && (cc <= w + mb + mf);
    if (!causal || ki < 0 || ki >= T) { continue; }

    // ac = scaled_q[qi] . (k_scale .* k[ki]).
    float ac = 0.0f;
    for (int p = 0; p < per; ++p) {
      const int d = lane * per + p;
      ac += qreg[p] * (float(k[(uint)ki * D + n * Hd + d]) * k_scale);
    }
    ac = simd_sum(ac);

    // bd via relative shift: q[u*W+rr] (scaled) . sin_emb[col].
    const int j = w * C + cc;
    const int rr = j / (C + 1);
    const int col = j % (C + 1);
    float bd = 0.0f;
    if (col < span && rr < W && (u * W + rr) < T) {
      const int qrow = u * W + rr;
      float b = 0.0f;
      for (int p = 0; p < per; ++p) {
        const int d = lane * per + p;
        const float qs =
            float(q[(uint)qrow * D + n * Hd + d]) * q_scale * float(pds[d]);
        b += qs * float(sin_emb[(uint)col * D + n * Hd + d]);
      }
      bd = simd_sum(b);
    }

    float lg = ac + bd;
    lg = precise::tanh(lg / softcap) * softcap;

    const float m_new = max(m, lg);
    const float corr = exp(m - m_new);
    const float pj = exp(lg - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      const int d = lane * per + p;
      acc[p] = acc[p] * corr + pj * float(v[(uint)ki * D + n * Hd + d]);
    }
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int p = 0; p < per; ++p) {
    const int d = lane * per + p;
    out[(uint)qi * D + n * Hd + d] = VPIPE_ELT(acc[p] * inv_l);
  }
}

kernel void sdpa_causal_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      q_offset [[buffer(10)]],
    constant int&      kv_stride [[buffer(11)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int q_pos = q_offset + qi;
  const int last = min(T_kv - 1, q_pos);   // causal horizon (inclusive)

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;

  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }

  float m = -INFINITY;   // running max
  float l = 0.0f;        // running denom
  for (int j = 0; j <= last; ++j) {
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      dot += qreg[p] * float(kkv[(uint)j * D + lane * per + p]);
    }
    dot = simd_sum(dot);

    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      acc[p] = acc[p] * corr + pj * float(vkv[(uint)j * D + lane * per + p]);
    }
    m = m_new;
  }

  const float inv_l = 1.0f / l;
  for (int p = 0; p < per; ++p) {
    out[((uint)h * n_q + qi) * D + lane * per + p] = VPIPE_ELT(acc[p] * inv_l);
  }
}

// sdpa_causal_window_f16 -- causal attention with a TRAILING sliding
// window (Gemma-4 sliding-attention layers). Identical to sdpa_causal_f16
// but query at absolute position q_pos = q_offset + qi only attends keys
// j in [max(0, q_pos - window + 1), q_pos] (causal AND within `window` of
// the query). window <= 0 disables the lower bound (== plain causal).
// This is a TRAILING window keyed on the absolute position -- NOT the
// window-aligned-block semantics of sdpa_window_f16.
//   0:q 1:k 2:v 3:out 4:scale 5:T_kv 6:D 7:Hq 8:Hkv 9:n_q 10:q_offset
//   11:kv_stride 12:window.  grid (32, Hq, n_q); threadgroup (32,1,1).
kernel void sdpa_causal_window_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      q_offset [[buffer(10)]],
    constant int&      kv_stride [[buffer(11)]],
    constant int&      window   [[buffer(12)]],
    constant int&      ring_cap [[buffer(13)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int q_pos = q_offset + qi;
  const int last = min(T_kv - 1, q_pos);            // causal horizon
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;

  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int j = first; j <= last; ++j) {
    // Ring layout: logical key j at physical slot j % ring_cap (sliding
    // layers); ring_cap == 0 == full layer (slot == j).
    const uint slot = (ring_cap > 0) ? (uint)(j % ring_cap) : (uint)j;
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      dot += qreg[p] * float(kkv[slot * D + lane * per + p]);
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      acc[p] = acc[p] * corr + pj * float(vkv[slot * D + lane * per + p]);
    }
    m = m_new;
  }
  const float inv_l = 1.0f / l;
  for (int p = 0; p < per; ++p) {
    out[((uint)h * n_q + qi) * D + lane * per + p] = VPIPE_ELT(acc[p] * inv_l);
  }
}

// sdpa_causal_mma_f16 -- simdgroup_matrix (MMA) flash attention over a
// CONTIGUOUS [Hkv, kv_stride, D] KV cache, for Gemma-4 PREFILL at head_dim
// 256 / 512. The per-query scalar sdpa_causal_f16 is COMPUTE-bound (one
// 32-lane simd_sum per key) and is the only part of metal prefill slower
// than MLX; this does QK^T and PV as dense simdgroup_matrix matmuls (the
// same 8x8x8 MMA the steel GEMM uses) with online softmax in fp32, causal +
// optional trailing window. The QK^T/PV operands are VPIPE_ELT (half or
// bfloat); the simdgroup_matrix accumulators stay fp32 -- the SAME fp32
// accumulation the steel GEMM does for both dtypes (BlockMMA<bfloat> also
// accumulates fp32), so the bf16 build introduces no extra fp32 work and
// matches the f16 numerics exactly.
//
// Tiling: one threadgroup per (head, MMA_BQ-query block). The head dim is
// split into WD = D/DSLICE slices (DSLICE=64 -> NDF=8 register-resident O
// col-frags per simdgroup, independent of D); WM=4 query bands of 8 rows.
// simdgroups = WM*WD (16 @ D256, 32 @ D512). To dodge the key-frag-split
// breakage at large WD / small BK, QK^T for a band is computed by its lead
// slice (wd==0) over the FULL D contraction (the other WD-1 slices idle
// during QK^T -- wall-time-free, they were spare), then softmax by 32
// threads (one per query row, WD-independent), then PV with every slice
// doing its own DSLICE output cols. K/V staged once per BK-key block into
// tg, BK = 4096/D (16 @ D256, 8 @ D512), so Ks+Vs <= 16 KB.
//
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out 4:scale 5:T_kv 6:D 7:Hq
//   8:Hkv 9:n_q 10:q_offset 11:kv_stride 12:window
// Precondition: D % 64 == 0, D <= 512; q padded by >= MMA_BQ*D halves.
// grid (WM*WD*32, Hq, ceil(n_q/MMA_BQ)); threadgroup (WM*WD*32,1,1).
#define MMAF_BQ      32
#define MMAF_WM      4
#define MMAF_DSLICE  64
#define MMAF_NDF     (MMAF_DSLICE / 8)   // 8 register-resident O col-frags
#define MMAF_KVELT   4096                // BK*D staged halves; BK = KVELT/D
#define MMAF_BKMAX   16                  // max BK (= KVELT/256)
kernel void sdpa_causal_mma_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      q_offset [[buffer(10)]],
    constant int&      kv_stride[[buffer(11)]],
    constant int&      window   [[buffer(12)]],
    constant int&      ring_cap [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h   = (int)tid.y;
  const int qb  = (int)tid.z;
  const int kvh = h / (Hq / Hkv);
  const int q_row0 = qb * MMAF_BQ;
  if (q_row0 >= n_q) { return; }
  const int WD  = D / MMAF_DSLICE;        // 4 @256, 8 @512
  const int BK  = MMAF_KVELT / D;         // 16 @256, 8 @512
  const int per = D / 8;                  // QK contraction frags
  const int nkf = BK / 8;                 // key-frags per block (1 or 2)
  const int nthreads = MMAF_WM * WD * 32;

  const int wm  = (int)simd_gid / WD;     // query band 0..WM-1
  const int wd  = (int)simd_gid % WD;     // head-dim slice 0..WD-1
  const int srow0 = wm * 8;               // band's first tile row
  const int grow0 = q_row0 + srow0;       // band's first global query row
  const int dc0   = wd * MMAF_DSLICE;     // simdgroup's first O col

  threadgroup VPIPE_ELT Ks[MMAF_KVELT];
  threadgroup VPIPE_ELT Vs[MMAF_KVELT];
  threadgroup float     Ss[MMAF_BQ * MMAF_BKMAX];
  // Per-slice QK^T partials: every slice contracts per/WD(==8) of the head
  // dim, the softmax sums the WD partials. Stride WD*BK == 64 (invariant),
  // so MMAF_BQ*64 elements cover all (slice,row,key) regardless of D.
  threadgroup float     Sp[MMAF_BQ * 64];
  threadgroup VPIPE_ELT Ps[MMAF_BQ * MMAF_BKMAX];
  threadgroup float m_[MMAF_BQ];
  threadgroup float l_[MMAF_BQ];
  threadgroup float corr_[MMAF_BQ];

  // Apple simdgroup_matrix<T,8,8> fragment layout (matches steel BaseMMAFrag):
  // this lane carries row `fm` (cols fn, fn+1) of every 8x8 frag.
  const int qid = (int)lane / 4;
  const int fm  = (qid & 4) + (((int)lane / 2) % 4);    // frag row 0..7
  const int fn  = (qid & 2) * 2 + ((int)lane % 2) * 2;  // frag col 0,2,4,6

  simdgroup_matrix<float, 8, 8> O[MMAF_NDF];
  for (int c = 0; c < MMAF_NDF; ++c) {
    O[c] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  if (lid < MMAF_BQ) { m_[lid] = -INFINITY; l_[lid] = 0.0f; }

  const device VPIPE_ELT* kkv = k + (uint)kvh * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kvh * kv_stride * D;

  // Tile-uniform key-scan bounds.
  const int tile_q0   = q_offset + q_row0;
  const int tile_qN   = q_offset + min(q_row0 + MMAF_BQ - 1, n_q - 1);
  const int scan_last = min(T_kv - 1, tile_qN);
  const int tile_first = (window > 0) ? max(0, tile_q0 - window + 1) : 0;
  const int bs0 = (tile_first / BK) * BK;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int bs = bs0; bs <= scan_last; bs += BK) {
    const int bk = min(BK, T_kv - bs);
    // Stage K/V block [bk, D] into tg (zero-pad rows >= bk to keep QK^T/PV
    // over the full BK well-defined; padded keys are masked in softmax).
    for (uint e = lid; e < (uint)(BK * D); e += (uint)nthreads) {
      const int t = (int)e / D, d = (int)e % D;
      if (t < bk) {
        // Ring layout (sliding layers): logical key bs+t lives at physical
        // slot (bs+t) % ring_cap. ring_cap == 0 == full layer (no modulo).
        const uint slot = (ring_cap > 0) ? (uint)((bs + t) % ring_cap)
                                          : (uint)(bs + t);
        Ks[e] = kkv[slot * D + d];
        Vs[e] = vkv[slot * D + d];
      } else {
        Ks[e] = (VPIPE_ELT)0;
        Vs[e] = (VPIPE_ELT)0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // QK^T: every slice contracts its per/WD(==8) frags of the head dim
    // into a partial S[8,BK], stored to its Sp region; the softmax sums the
    // WD partials. (All simdgroups busy, vs the lead-slice-only variant.)
    {
      const int pf0 = wd * (per / WD);                   // this slice's frags
      simdgroup_matrix<float, 8, 8> S[MMAF_BKMAX / 8];
      for (int j = 0; j < nkf; ++j) { S[j] = simdgroup_matrix<float, 8, 8>(0.0f); }
      for (int e = 0; e < per / WD; ++e) {
        const int dd = pf0 + e;
        simdgroup_matrix<VPIPE_ELT, 8, 8> A;             // Q[row, d]
        simdgroup_load(A, q + ((uint)h * n_q + grow0) * D + dd * 8, D);
        for (int j = 0; j < nkf; ++j) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> B;        // B[d,key]=K[key,d]
          simdgroup_load(B, Ks, D, ulong2(dd * 8, j * 8), true);
          simdgroup_multiply_accumulate(S[j], A, B, S[j]);
        }
      }
      for (int j = 0; j < nkf; ++j) {
        simdgroup_store(S[j], Sp + wd * MMAF_BQ * BK + srow0 * BK + j * 8, BK);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax + causal/window mask: 32 threads, one per query row.
    // Each first sums the WD per-slice QK^T partials into the full score.
    if (lid < (uint)MMAF_BQ) {
      const int r = (int)lid;
      const int qpos = q_offset + q_row0 + r;
      const int first = (window > 0) ? max(0, qpos - window + 1) : 0;
      float lmax = -INFINITY;
      for (int j = 0; j < bk; ++j) {
        float sj = 0.0f;
        for (int w = 0; w < WD; ++w) {
          sj += Sp[w * MMAF_BQ * BK + r * BK + j];
        }
        float s = sj * scale;
        const int gkey = bs + j;
        if (gkey > qpos || gkey < first) { s = -INFINITY; }
        Ss[r * BK + j] = s;
        lmax = max(lmax, s);
      }
      const float mold = m_[r];
      const float mnew = max(mold, lmax);
      // Guard the trailing-window case where this row's entire scanned
      // block is masked (mnew stays -inf): exp(-inf - -inf) = NaN would
      // poison O. corr=1 here is a no-op (O and l_ are still 0).
      const float corr = (mnew == -INFINITY) ? 1.0f : exp(mold - mnew);
      float lsum = 0.0f;
      for (int j = 0; j < bk; ++j) {
        const float s = Ss[r * BK + j];
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mnew);
        Ps[r * BK + j] = (VPIPE_ELT)p;
        lsum += p;
      }
      l_[r] = l_[r] * corr + lsum;
      m_[r] = mnew;
      corr_[r] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rescale this simdgroup's O frags by the per-row correction, then PV.
    const float cr = corr_[srow0 + fm];
    for (int c = 0; c < MMAF_NDF; ++c) {
      O[c].thread_elements()[0] *= cr;
      O[c].thread_elements()[1] *= cr;
    }
    for (int kk = 0; kk < nkf; ++kk) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> A2;              // P[row, key]
      simdgroup_load(A2, Ps + srow0 * BK + kk * 8, BK);
      for (int c = 0; c < MMAF_NDF; ++c) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> B2;            // V[key, dcol]
        simdgroup_load(B2, Vs, D, ulong2(dc0 + c * 8, kk * 8), false);
        simdgroup_multiply_accumulate(O[c], A2, B2, O[c]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Normalize (divide by row denom) and write O frags to out, guarding rows
  // past n_q. Each lane writes its 2 cols (fn, fn+1) of each O col-frag.
  const int row = grow0 + fm;
  if (row < n_q) {
    const float denom = l_[srow0 + fm];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    const uint orow = ((uint)h * n_q + row) * D;
    for (int c = 0; c < MMAF_NDF; ++c) {
      const int col = dc0 + c * 8 + fn;
      out[orow + col]     = VPIPE_ELT(O[c].thread_elements()[0] * inv);
      out[orow + col + 1] = VPIPE_ELT(O[c].thread_elements()[1] * inv);
    }
  }
}

// sdpa_full_mma_f16 -- NON-CAUSAL (bidirectional) flash attention, the fast
// path for the ViT / Conformer ENCODERS (full attention over the whole
// patch/frame sequence -- the O(n^2) term that made the scalar sdpa_full_f16
// 5-11x slower than MLX). Identical fragment math to sdpa_causal_mma_f16
// (32-row Q tile, WD head-dim slices, contraction-split QK^T, online softmax,
// MMA PV) but scans ALL keys [0, T_kv) with NO causal/window mask -- every
// query attends every key. Handles small head_dim (ViT hd=64): BK is clamped
// to MMAF_BKMAX so the staged score buffers never overflow when KVELT/D >
// BKMAX. Partial-block tail keys are zero-staged in Vs -> their PV is 0, so no
// explicit padding mask is needed (same trick as the causal kernel).
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out 4:scale 5:T_kv 6:D 7:Hq
//   8:Hkv 9:n_q 10:kv_stride.  D % 64 == 0, D <= 512.
// grid (WM*WD*32, Hq, ceil(n_q/MMA_BQ)); threadgroup (WM*WD*32,1,1).
kernel void sdpa_full_mma_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      kv_stride[[buffer(10)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h   = (int)tid.y;
  const int qb  = (int)tid.z;
  const int kvh = h / (Hq / Hkv);
  const int q_row0 = qb * MMAF_BQ;
  if (q_row0 >= n_q) { return; }
  const int WD  = D / MMAF_DSLICE;
  const int BK  = min(MMAF_KVELT / D, MMAF_BKMAX);  // clamp for small D
  const int per = D / 8;
  const int nkf = BK / 8;
  const int nthreads = MMAF_WM * WD * 32;

  const int wm  = (int)simd_gid / WD;
  const int wd  = (int)simd_gid % WD;
  const int srow0 = wm * 8;
  const int grow0 = q_row0 + srow0;
  const int dc0   = wd * MMAF_DSLICE;

  threadgroup VPIPE_ELT Ks[MMAF_KVELT];
  threadgroup VPIPE_ELT Vs[MMAF_KVELT];
  threadgroup float     Ss[MMAF_BQ * MMAF_BKMAX];
  threadgroup float     Sp[MMAF_BQ * 64];
  threadgroup VPIPE_ELT Ps[MMAF_BQ * MMAF_BKMAX];
  threadgroup float m_[MMAF_BQ];
  threadgroup float l_[MMAF_BQ];
  threadgroup float corr_[MMAF_BQ];

  const int qid = (int)lane / 4;
  const int fm  = (qid & 4) + (((int)lane / 2) % 4);
  const int fn  = (qid & 2) * 2 + ((int)lane % 2) * 2;

  simdgroup_matrix<float, 8, 8> O[MMAF_NDF];
  for (int c = 0; c < MMAF_NDF; ++c) {
    O[c] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  if (lid < MMAF_BQ) { m_[lid] = -INFINITY; l_[lid] = 0.0f; }

  const device VPIPE_ELT* kkv = k + (uint)kvh * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kvh * kv_stride * D;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int bs = 0; bs < T_kv; bs += BK) {
    const int bk = min(BK, T_kv - bs);
    for (uint e = lid; e < (uint)(BK * D); e += (uint)nthreads) {
      const int t = (int)e / D, d = (int)e % D;
      if (t < bk) {
        Ks[e] = kkv[(uint)(bs + t) * D + d];
        Vs[e] = vkv[(uint)(bs + t) * D + d];
      } else {
        Ks[e] = (VPIPE_ELT)0;
        Vs[e] = (VPIPE_ELT)0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
      const int pf0 = wd * (per / WD);
      simdgroup_matrix<float, 8, 8> S[MMAF_BKMAX / 8];
      for (int j = 0; j < nkf; ++j) {
        S[j] = simdgroup_matrix<float, 8, 8>(0.0f);
      }
      for (int e = 0; e < per / WD; ++e) {
        const int dd = pf0 + e;
        simdgroup_matrix<VPIPE_ELT, 8, 8> A;
        simdgroup_load(A, q + ((uint)h * n_q + grow0) * D + dd * 8, D);
        for (int j = 0; j < nkf; ++j) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> B;
          simdgroup_load(B, Ks, D, ulong2(dd * 8, j * 8), true);
          simdgroup_multiply_accumulate(S[j], A, B, S[j]);
        }
      }
      for (int j = 0; j < nkf; ++j) {
        simdgroup_store(S[j], Sp + wd * MMAF_BQ * BK + srow0 * BK + j * 8, BK);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, NO causal/window mask (every key in [0,bk) is valid).
    if (lid < (uint)MMAF_BQ) {
      const int r = (int)lid;
      float lmax = -INFINITY;
      for (int j = 0; j < bk; ++j) {
        float sj = 0.0f;
        for (int w = 0; w < WD; ++w) {
          sj += Sp[w * MMAF_BQ * BK + r * BK + j];
        }
        const float s = sj * scale;
        Ss[r * BK + j] = s;
        lmax = max(lmax, s);
      }
      const float mold = m_[r];
      const float mnew = max(mold, lmax);
      const float corr = (mnew == -INFINITY) ? 1.0f : exp(mold - mnew);
      float lsum = 0.0f;
      for (int j = 0; j < bk; ++j) {
        const float p = exp(Ss[r * BK + j] - mnew);
        Ps[r * BK + j] = (VPIPE_ELT)p;
        lsum += p;
      }
      l_[r] = l_[r] * corr + lsum;
      m_[r] = mnew;
      corr_[r] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cr = corr_[srow0 + fm];
    for (int c = 0; c < MMAF_NDF; ++c) {
      O[c].thread_elements()[0] *= cr;
      O[c].thread_elements()[1] *= cr;
    }
    for (int kk = 0; kk < nkf; ++kk) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> A2;
      simdgroup_load(A2, Ps + srow0 * BK + kk * 8, BK);
      for (int c = 0; c < MMAF_NDF; ++c) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> B2;
        simdgroup_load(B2, Vs, D, ulong2(dc0 + c * 8, kk * 8), false);
        simdgroup_multiply_accumulate(O[c], A2, B2, O[c]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const int row = grow0 + fm;
  if (row < n_q) {
    const float denom = l_[srow0 + fm];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    const uint orow = ((uint)h * n_q + row) * D;
    for (int c = 0; c < MMAF_NDF; ++c) {
      const int col = dc0 + c * 8 + fn;
      out[orow + col]     = VPIPE_ELT(O[c].thread_elements()[0] * inv);
      out[orow + col + 1] = VPIPE_ELT(O[c].thread_elements()[1] * inv);
    }
  }
}

// sdpa_causal_mma_dev_f16 -- the CONTIGUOUS (non-ring) prefill flash kernel,
// the fast path for Gemma-4's GLOBAL/full-attention layers (head_dim 512, the
// O(n^2) term that dominates long-context prefill). Identical fragment math to
// sdpa_causal_mma_f16 (contraction-split QK^T, WD head-dim slices, online
// softmax, 32-row Q tile), but the K/V block is read DIRECTLY from device
// memory via simdgroup_load instead of being staged into threadgroup. Dropping
// the Ks/Vs staging (a) removes one barrier + the staging traffic per block and
// (b) frees ~16 KB of threadgroup memory, which is spent on a 3x larger key
// block (MMAF_BK_DEV=24 vs 8 @ D512) -> ~3x fewer loop iterations and barriers.
//
// PRECONDITION (caller-enforced): the K/V buffer is CONTIGUOUS (no ring) and
// allocated with >= MMAF_BK_DEV rows of slack past T_kv (i.e. kv_stride >=
// round_up(T_kv, MMAF_BK_DEV)), so the final block's device read of keys in
// [T_kv, bs+BK) stays inside the allocation. Those over-read keys are masked in
// softmax (gkey >= T_kv > qpos => -inf => P=0), and the KV tail is finite
// (zero/stale K/V), so the masked PV contributes 0. ring_cap is ignored.
// Same buffer/grid layout as sdpa_causal_mma_f16.
// MMAF_BK_DEV = key-block for the device-direct kernel. Kept at 8 (== the
// staged kernel's BK @ D512) so the online-softmax m/l update sequence is
// IDENTICAL -> output is bit-for-bit equal to sdpa_causal_mma_f16 (token-exact;
// the win here is purely dropping the staging traffic + one barrier/block,
// ~1.3-1.4x). A larger block (e.g. 24) gives ~1.5x but changes the softmax
// blocking -> not bit-identical (fp differences propagate into upper-layer KV
// and flip a greedy token after ~16 steps); that path would need re-validating
// the whole metal Gemma against MLX/omlx before it could ship.
#define MMAF_BK_DEV 8
kernel void sdpa_causal_mma_dev_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      q_offset [[buffer(10)]],
    constant int&      kv_stride[[buffer(11)]],
    constant int&      window   [[buffer(12)]],
    constant int&      ring_cap [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  (void)ring_cap;                         // contiguous only
  const int h   = (int)tid.y;
  const int qb  = (int)tid.z;
  const int kvh = h / (Hq / Hkv);
  const int q_row0 = qb * MMAF_BQ;
  if (q_row0 >= n_q) { return; }
  const int WD  = D / MMAF_DSLICE;        // 4 @256, 8 @512
  const int BK  = MMAF_BK_DEV;            // 24 (vs 4096/D in the staged kernel)
  const int per = D / 8;                  // QK contraction frags
  const int nkf = BK / 8;                 // key-frags per block (3)
  const int nthreads = MMAF_WM * WD * 32;
  (void)nthreads;

  const int wm  = (int)simd_gid / WD;     // query band 0..WM-1
  const int wd  = (int)simd_gid % WD;     // head-dim slice 0..WD-1
  const int srow0 = wm * 8;               // band's first tile row
  const int grow0 = q_row0 + srow0;       // band's first global query row
  const int dc0   = wd * MMAF_DSLICE;     // simdgroup's first O col

  // No Ks/Vs staging (device-direct). Sp holds the per-slice QK^T partials:
  // span = WD*MMAF_BQ*BK; sized for WD<=8 (head_dim<=512).
  threadgroup float     Sp[MMAF_BQ * 8 * MMAF_BK_DEV];
  threadgroup float     Ss[MMAF_BQ * MMAF_BK_DEV];
  threadgroup VPIPE_ELT Ps[MMAF_BQ * MMAF_BK_DEV];
  threadgroup float m_[MMAF_BQ];
  threadgroup float l_[MMAF_BQ];
  threadgroup float corr_[MMAF_BQ];

  const int qid = (int)lane / 4;
  const int fm  = (qid & 4) + (((int)lane / 2) % 4);
  const int fn  = (qid & 2) * 2 + ((int)lane % 2) * 2;

  simdgroup_matrix<float, 8, 8> O[MMAF_NDF];
  for (int c = 0; c < MMAF_NDF; ++c) {
    O[c] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  if (lid < MMAF_BQ) { m_[lid] = -INFINITY; l_[lid] = 0.0f; }

  const device VPIPE_ELT* kkv = k + (uint)kvh * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kvh * kv_stride * D;

  const int tile_q0   = q_offset + q_row0;
  const int tile_qN   = q_offset + min(q_row0 + MMAF_BQ - 1, n_q - 1);
  const int scan_last = min(T_kv - 1, tile_qN);
  const int tile_first = (window > 0) ? max(0, tile_q0 - window + 1) : 0;
  const int bs0 = (tile_first / BK) * BK;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int bs = bs0; bs <= scan_last; bs += BK) {
    const int bk = min(BK, T_kv - bs);
    // QK^T: each slice contracts per/WD frags of the head dim into S[8,BK]
    // partials (device-direct K read). bk<BK over-reads finite tail keys that
    // are masked below.
    {
      const int pf0 = wd * (per / WD);
      simdgroup_matrix<float, 8, 8> S[MMAF_BK_DEV / 8];
      for (int j = 0; j < nkf; ++j) {
        S[j] = simdgroup_matrix<float, 8, 8>(0.0f);
      }
      for (int e = 0; e < per / WD; ++e) {
        const int dd = pf0 + e;
        simdgroup_matrix<VPIPE_ELT, 8, 8> A;
        simdgroup_load(A, q + ((uint)h * n_q + grow0) * D + dd * 8, D);
        for (int j = 0; j < nkf; ++j) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> B;
          simdgroup_load(B, kkv + (uint)bs * D, D,
                         ulong2(dd * 8, j * 8), true);
          simdgroup_multiply_accumulate(S[j], A, B, S[j]);
        }
      }
      for (int j = 0; j < nkf; ++j) {
        simdgroup_store(S[j], Sp + wd * MMAF_BQ * BK + srow0 * BK + j * 8, BK);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax + causal/window mask: 32 threads, one per query row.
    if (lid < (uint)MMAF_BQ) {
      const int r = (int)lid;
      const int qpos = q_offset + q_row0 + r;
      const int first = (window > 0) ? max(0, qpos - window + 1) : 0;
      float lmax = -INFINITY;
      for (int j = 0; j < bk; ++j) {
        float sj = 0.0f;
        for (int w = 0; w < WD; ++w) {
          sj += Sp[w * MMAF_BQ * BK + r * BK + j];
        }
        float s = sj * scale;
        const int gkey = bs + j;
        if (gkey > qpos || gkey < first) { s = -INFINITY; }
        Ss[r * BK + j] = s;
        lmax = max(lmax, s);
      }
      const float mold = m_[r];
      const float mnew = max(mold, lmax);
      const float corr = (mnew == -INFINITY) ? 1.0f : exp(mold - mnew);
      float lsum = 0.0f;
      for (int j = 0; j < bk; ++j) {
        const float s = Ss[r * BK + j];
        const float p = (s == -INFINITY) ? 0.0f : exp(s - mnew);
        Ps[r * BK + j] = (VPIPE_ELT)p;
        lsum += p;
      }
      // Zero the padded key columns [bk,BK) so PV's frag MMA over the full BK
      // multiplies the (finite) over-read V by an exact 0 (no 0*inf risk).
      for (int j = bk; j < BK; ++j) { Ps[r * BK + j] = (VPIPE_ELT)0; }
      l_[r] = l_[r] * corr + lsum;
      m_[r] = mnew;
      corr_[r] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cr = corr_[srow0 + fm];
    for (int c = 0; c < MMAF_NDF; ++c) {
      O[c].thread_elements()[0] *= cr;
      O[c].thread_elements()[1] *= cr;
    }
    for (int kk = 0; kk < nkf; ++kk) {
      simdgroup_matrix<VPIPE_ELT, 8, 8> A2;              // P[row, key]
      simdgroup_load(A2, Ps + srow0 * BK + kk * 8, BK);
      for (int c = 0; c < MMAF_NDF; ++c) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> B2;            // V[key, dcol]
        simdgroup_load(B2, vkv + (uint)bs * D, D,
                       ulong2(dc0 + c * 8, kk * 8), false);
        simdgroup_multiply_accumulate(O[c], A2, B2, O[c]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const int row = grow0 + fm;
  if (row < n_q) {
    const float denom = l_[srow0 + fm];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    const uint orow = ((uint)h * n_q + row) * D;
    for (int c = 0; c < MMAF_NDF; ++c) {
      const int col = dc0 + c * 8 + fn;
      out[orow + col]     = VPIPE_ELT(O[c].thread_elements()[0] * inv);
      out[orow + col + 1] = VPIPE_ELT(O[c].thread_elements()[1] * inv);
    }
  }
}

// sdpa_causal_flash_f16 -- llama.cpp-style flash attention for the CONTIGUOUS
// (global/full-attention) prefill path, head_dim <= 512. The vpipe
// sdpa_causal_mma_f16 kernel scans keys in tiny BK=8 blocks with a SERIAL
// 32-thread softmax + a contraction-split partial-combine; this kernel adopts
// llama.cpp's tiling instead:
//   * Q = 8 query rows per threadgroup, C = 64 KEYS per block (8x bigger ->
//     8x fewer loop iterations + barriers).
//   * NSG = 8 simdgroups (256 threads). QK is KEY-SPLIT: each simdgroup does
//     the FULL head-dim contraction for its own 8 keys -> one 8x8 score tile,
//     no partials to merge. PV is DV-SPLIT: each simdgroup owns DV/NSG output
//     columns (NO = DV/8/NSG register frags).
//   * Online softmax is FULLY PARALLEL: simd_max/simd_sum across the 32 lanes
//     (lane <-> 2 of the C=64 score columns), running m/l in registers.
//   * K/V read DIRECTLY from device memory (no threadgroup staging); only the
//     Q block (Q*D halves), the fp32 O accumulator (Q*DV floats) and the score
//     scratch (Q*C floats) live in threadgroup (~26 KB @ D=512).
// Covers the global/full layers (ring_cap==0) AND sliding-window layers in
// the NO-WRAP regime: for bounded sliding layers cap == ring_cap, so the
// caller's kv_off+n+C <= cap precondition guarantees the ring has not wrapped
// (logical key index == physical slot, exactly the contiguous addressing this
// kernel uses). window>0 enables the trailing-window mask AND a window-base
// scan skip (FL_C-blocks entirely below the lowest query's window are fully
// masked -> the scan starts there, matching the staged kernel's bs0). The
// final C-block may over-read keys in [T_kv, ic+C); they are causal-masked to
// P=0, and the KV tail must be finite + inside the allocation (the
// kv_off+n+C<=cap precondition keeps the over-read rows inside the cache).
// Online-softmax => same math as the scalar/staged kernels, NOT bit-identical.
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out 4:scale 5:T_kv 6:D 7:Hq
//   8:Hkv 9:n_q 10:q_offset 11:kv_stride 12:window 13:ring_cap(ignored)
// grid (FL_NSG*32, Hq, ceil(n_q/FL_Q)); threadgroup (FL_NSG*32,1,1).
#define FL_Q    8
#define FL_C    64
#define FL_NSG  8
#define FL_DMAX 512
kernel void sdpa_causal_flash_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      q_offset [[buffer(10)]],
    constant int&      kv_stride[[buffer(11)]],
    constant int&      window   [[buffer(12)]],
    constant int&      ring_cap [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  (void)ring_cap;
  const int h   = (int)tid.y;
  const int q0  = (int)tid.z * FL_Q;
  if (q0 >= n_q) { return; }
  const int kvh = h / (Hq / Hkv);
  const int sg  = (int)simd_gid;          // 0..7
  const int DV  = D;                       // square head dim
  const int nthreads = FL_NSG * 32;
  const int NO  = (DV / 8) / FL_NSG;        // O frags per simdgroup

  threadgroup VPIPE_ELT sq[FL_Q * FL_DMAX];     // Q block (halves)
  threadgroup float     so[FL_Q * FL_DMAX];     // O accumulator (fp32)
  threadgroup float     ssf[FL_Q * FL_C];       // QK scores (fp32 -> softmax)
  threadgroup VPIPE_ELT ssp[FL_Q * FL_C];       // softmax P (-> PV, same dtype as V)

  // Stage the Q block + zero the O accumulator (cooperative, all threads).
  for (uint e = lid; e < (uint)(FL_Q * D); e += (uint)nthreads) {
    const int j = (int)e / D, d = (int)e % D;
    const int row = q0 + j;
    sq[e] = (row < n_q) ? q[((uint)h * n_q + row) * D + d] : (VPIPE_ELT)0;
  }
  for (uint e = lid; e < (uint)(FL_Q * DV); e += (uint)nthreads) {
    so[e] = 0.0f;
  }
  float M = -INFINITY, S = 0.0f;          // per-row (this SG owns row `sg`)
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* kkv = k + (uint)kvh * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kvh * kv_stride * D;

  // Causal horizon for this Q-block: the highest query position is the last
  // valid row; keys beyond it are masked, so stop scanning there.
  const int qmax = q_offset + min(q0 + FL_Q - 1, n_q - 1);
  const int scan_last = min(T_kv - 1, qmax);
  // Window-base scan skip: the lowest query in this block is at q_offset+q0;
  // FL_C-blocks entirely below its trailing window are fully masked (P=0), so
  // start the scan at that window base (rounded down to FL_C). window<=0
  // (full causal) starts at 0. Matches the staged kernel's bs0 skip so the
  // sliding-layer scan touches ~window keys, not all of T_kv.
  int ic0 = 0;
  if (window > 0) {
    const int tile_first = max(0, (q_offset + q0) - window + 1);
    ic0 = (tile_first / FL_C) * FL_C;
  }

  for (int ic = ic0; ic <= scan_last; ic += FL_C) {
    // ---- QK^T: this SG contracts the full head dim for its 8 keys ----
    {
      simdgroup_matrix<float, 8, 8> mqk = simdgroup_matrix<float, 8, 8>(0.0f);
      const device VPIPE_ELT* pk = kkv + (uint)(ic + sg * 8) * D;
      for (int i = 0; i < D / 8; ++i) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> mq;   // Q[8 rows, 8 dims]
        simdgroup_load(mq, sq + i * 8, D);
        simdgroup_matrix<VPIPE_ELT, 8, 8> mk;   // K^T[8 dims, 8 keys]
        simdgroup_load(mk, pk + i * 8, D, ulong2(0), true);
        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
      }
      simdgroup_store(mqk, ssf + sg * 8, FL_C);  // tile at col 8*sg, stride C
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- online softmax: SG `sg` owns query row j=sg; 32 lanes split C ----
    {
      const int j = sg;
      const int qpos = q_offset + q0 + j;
      const int first = (window > 0) ? max(0, qpos - window + 1) : 0;
      float2 s2;
      s2[0] = ssf[j * FL_C + 2 * (int)lane];
      s2[1] = ssf[j * FL_C + 2 * (int)lane + 1];
      s2 *= scale;
      const int g0 = ic + 2 * (int)lane, g1 = g0 + 1;
      if (g0 > qpos || g0 < first) { s2[0] = -INFINITY; }
      if (g1 > qpos || g1 < first) { s2[1] = -INFINITY; }
      const float lmax = simd_max(max(s2[0], s2[1]));
      const float m_new = max(M, lmax);
      const float ms = (m_new == -INFINITY) ? 1.0f : exp(M - m_new);
      float2 p2;
      p2[0] = (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - m_new);
      p2[1] = (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - m_new);
      S = S * ms + simd_sum(p2[0] + p2[1]);
      M = m_new;
      ssp[j * FL_C + 2 * (int)lane]     = (VPIPE_ELT)p2[0];
      ssp[j * FL_C + 2 * (int)lane + 1] = (VPIPE_ELT)p2[1];
      // rescale this row's O accumulator (in tg) by the correction factor.
      for (int d = (int)lane; d < DV; d += 32) { so[j * DV + d] *= ms; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- PV: SG `sg` owns NO output-column frags (8*sg, stride 8*NSG) ----
    {
      simdgroup_matrix<float, 8, 8> lo[FL_DMAX / 8 / FL_NSG];
      for (int ii = 0; ii < NO; ++ii) {
        simdgroup_load(lo[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
      }
      for (int cc = 0; cc < FL_C / 8; ++cc) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> vs;     // P[8q, 8k]
        simdgroup_load(vs, ssp + cc * 8, FL_C);
        for (int ii = 0; ii < NO; ++ii) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> mv;   // V[8k, 8 dvcol]
          simdgroup_load(mv, vkv + (uint)(ic + cc * 8) * D + 8 * sg
                             + ii * 8 * FL_NSG, D);
          simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
        }
      }
      for (int ii = 0; ii < NO; ++ii) {
        simdgroup_store(lo[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- normalize + write: SG `sg` writes query row j=sg ----
  const int j = sg;
  if (q0 + j < n_q) {
    const float inv = S > 0.0f ? 1.0f / S : 0.0f;
    const uint orow = ((uint)h * n_q + q0 + j) * D;
    for (int d = (int)lane; d < DV; d += 32) {
      out[orow + d] = (VPIPE_ELT)(so[j * DV + d] * inv);
    }
  }
}

// sdpa_causal_mma_qhead_f16 -- MMA flash DECODE for the GLOBAL (full-context,
// head_dim<=512) layers, the matrix-core counterpart of the scalar gtile/vec
// decode kernels and vpipe's answer to llama.cpp's kernel_flash_attn_ext (the
// only attn it has for head_dim 512). The decode trick: the 8 MMA rows are 8
// Q-HEADS at the single decode position q_offset, all sharing ONE kv head
// (requires G=Hq/Hkv divisible by 8) -- so the 8x8 QK^T/PV simdgroup_matrix
// tiles are FULLY used (no 7/8 decode waste), the kv head is read ONCE for 8
// heads, and there is NO per-key simd_sum (the scalar kernels' bottleneck that
// grows with depth). KV is split across grid.z=`split` threadgroups
// (flash-decoding); with split = ceil(scan/FL_C) each split is one FL_C=64 key
// block -> constant per-split work => FLAT with KV depth. Writes UN-normalized
// per-(head,split) partials that sdpa_gqa_merge_f16 folds. Online softmax fp32
// (same approx as sdpa_causal_flash_f16, token-exact not bit-exact). Global/full
// layers only: contiguous no-wrap KV (ring_cap ignored); the caller guarantees
// q_offset+FL_C <= cap so the C-block over-read stays in-bounds.
//   0:q[Hq,D] 1:k 2:v[Hkv,kv_stride,D] 3:o_acc(f32) 4:m(f32) 5:l(f32) 6:scale
//   7:T_kv 8:D 9:Hq 10:Hkv 11:q_offset 12:kv_stride 13:window 14:ring_cap 15:split
// grid (FL_NSG*32, Hq/8, split); threadgroup (FL_NSG*32,1,1).
kernel void sdpa_causal_mma_qhead_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device float*           o_acc    [[buffer(3)]],
    device float*           m_out    [[buffer(4)]],
    device float*           l_out    [[buffer(5)]],
    constant float&    scale    [[buffer(6)]],
    constant int&      T_kv     [[buffer(7)]],
    constant int&      D        [[buffer(8)]],
    constant int&      Hq       [[buffer(9)]],
    constant int&      Hkv      [[buffer(10)]],
    constant int&      q_offset [[buffer(11)]],
    constant int&      kv_stride[[buffer(12)]],
    constant int&      window   [[buffer(13)]],
    constant int&      ring_cap [[buffer(14)]],
    constant int&      split    [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  (void)ring_cap;
  const int ht    = (int)tid.y;            // head-tile of 8 q-heads
  const int s     = (int)tid.z;            // KV split index
  const int G     = Hq / Hkv;
  const int head0 = ht * 8;                // first head of this tile
  const int kvh   = head0 / G;             // 8 heads share this kv head (8|G)
  const int sg    = (int)simd_gid;         // 0..7 -> head head0+sg / key octet
  const int DV    = D;
  const int nthreads = FL_NSG * 32;
  const int NO    = (DV / 8) / FL_NSG;

  threadgroup VPIPE_ELT sq[FL_Q * FL_DMAX];
  threadgroup float     so[FL_Q * FL_DMAX];
  threadgroup float     ssf[FL_Q * FL_C];
  threadgroup VPIPE_ELT ssp[FL_Q * FL_C];

  // Stage Q: the 8 rows are heads head0..head0+7 at the decode position.
  for (uint e = lid; e < (uint)(FL_Q * D); e += (uint)nthreads) {
    const int j = (int)e / D, d = (int)e % D;
    sq[e] = q[(uint)(head0 + j) * D + d];
  }
  for (uint e = lid; e < (uint)(FL_Q * DV); e += (uint)nthreads) { so[e] = 0.0f; }
  float M = -INFINITY, S = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device VPIPE_ELT* kkv = k + (uint)kvh * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kvh * kv_stride * D;

  const int q_pos = q_offset;
  const int last  = min(T_kv - 1, q_pos);
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;
  const int total = last - first + 1;
  const int chunk = (total > 0) ? (total + split - 1) / split : 0;
  const int lo = first + s * chunk;
  const int hi = min(first + (s + 1) * chunk, last + 1);

  // Scan FL_C-aligned blocks covering [lo, hi); keys outside [lo,hi) (and
  // causal / window) are masked so each key is counted in exactly one split.
  for (int ic = (lo / FL_C) * FL_C; ic < hi; ic += FL_C) {
    // ---- QK^T: SG sg contracts the full head dim for its 8 keys ----
    {
      simdgroup_matrix<float, 8, 8> mqk = simdgroup_matrix<float, 8, 8>(0.0f);
      const device VPIPE_ELT* pk = kkv + (uint)(ic + sg * 8) * D;
      for (int i = 0; i < D / 8; ++i) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> mq;
        simdgroup_load(mq, sq + i * 8, D);
        simdgroup_matrix<VPIPE_ELT, 8, 8> mk;
        simdgroup_load(mk, pk + i * 8, D, ulong2(0), true);
        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
      }
      simdgroup_store(mqk, ssf + sg * 8, FL_C);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- online softmax: SG sg owns head row j=sg; 32 lanes split C ----
    {
      const int jj = sg;
      float2 s2;
      s2[0] = ssf[jj * FL_C + 2 * (int)lane];
      s2[1] = ssf[jj * FL_C + 2 * (int)lane + 1];
      s2 *= scale;
      const int g0 = ic + 2 * (int)lane, g1 = g0 + 1;
      if (g0 < lo || g0 >= hi || g0 > q_pos || g0 < first) { s2[0] = -INFINITY; }
      if (g1 < lo || g1 >= hi || g1 > q_pos || g1 < first) { s2[1] = -INFINITY; }
      const float lmax = simd_max(max(s2[0], s2[1]));
      const float m_new = max(M, lmax);
      const float ms = (m_new == -INFINITY) ? 1.0f : exp(M - m_new);
      float2 p2;
      p2[0] = (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - m_new);
      p2[1] = (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - m_new);
      S = S * ms + simd_sum(p2[0] + p2[1]);
      M = m_new;
      ssp[jj * FL_C + 2 * (int)lane]     = (VPIPE_ELT)p2[0];
      ssp[jj * FL_C + 2 * (int)lane + 1] = (VPIPE_ELT)p2[1];
      for (int d = (int)lane; d < DV; d += 32) { so[jj * DV + d] *= ms; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- PV: SG sg owns NO output-column frags ----
    {
      simdgroup_matrix<float, 8, 8> lacc[FL_DMAX / 8 / FL_NSG];
      for (int ii = 0; ii < NO; ++ii) {
        simdgroup_load(lacc[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
      }
      for (int cc = 0; cc < FL_C / 8; ++cc) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> vs;
        simdgroup_load(vs, ssp + cc * 8, FL_C);
        for (int ii = 0; ii < NO; ++ii) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> mv;
          simdgroup_load(mv, vkv + (uint)(ic + cc * 8) * D + 8 * sg
                             + ii * 8 * FL_NSG, D);
          simdgroup_multiply_accumulate(lacc[ii], vs, mv, lacc[ii]);
        }
      }
      for (int ii = 0; ii < NO; ++ii) {
        simdgroup_store(lacc[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- write UN-normalized partials: SG sg -> head head0+sg, split s ----
  const int head = head0 + sg;
  const uint base = ((uint)head * (uint)split + (uint)s) * (uint)DV;
  for (int d = (int)lane; d < DV; d += 32) { o_acc[base + d] = so[sg * DV + d]; }
  if (lane == 0) {
    m_out[(uint)head * split + s] = M;
    l_out[(uint)head * split + s] = S;
  }
}

// sdpa_paged_flash_f16 -- the PAGED-KV sibling of sdpa_causal_flash_f16: the
// llama.cpp-style KEY-SPLIT flash attention (Q=8 rows, C=64 keys/block, NSG=8
// simdgroups, each SG doing the FULL head-dim contraction for its own 8 keys
// -> one complete 8x8 score tile, no partial-combine; parallel
// simd_max/simd_sum softmax; fp32 O accumulator in tg) for Qwen3.5 full-
// attention PREFILL at head_dim 256. It replaces the scalar query-tiled
// sdpa_paged_qtile_f16 on M4 (no matrix cores) -- simdgroup_matrix runs on
// every Apple GPU, unlike the matmul2d sdpa_mma_f16 (M5-only; a no-op stub on
// M4). K/V live in the PAGED pool [n_pages, Hkv, page_tokens, D]; within a page
// a C-block is contiguous, so each SG reads its 8 keys straight from device (no
// tg staging, keeping the D=256 budget ~15 KB). The last block of a page may be
// partial (bk<C): over-reading [bk,C) hits in-bounds page memory (finite), and
// the `key<bk` mask zeroes those P so PV adds 0 -- requires page_tokens%C==0
// (true for 256/512/1024, all multiples of 64) so the C-block over-read stays
// inside the page. Online-softmax, NOT bit-identical to the scalar/qtile
// kernels (same fp-approximation as the Gemma flash, and equally faithful).
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out 4:scale 5:D 6:Hq 7:Hkv 8:n_q
//   9:q_offset 10:page_tokens 11:n_pages 12:page_table({pid,nvalid,gstart})
// grid (FL_NSG*32, Hq, ceil(n_q/FL_Q)); threadgroup (FL_NSG*32,1,1).
#define FLP_DMAX 256
kernel void sdpa_paged_flash_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h   = (int)tid.y;
  const int q0  = (int)tid.z * FL_Q;
  if (q0 >= n_q) { return; }
  const int kvh = h / (Hq / Hkv);
  const int sg  = (int)simd_gid;          // 0..7
  const int DV  = D;                       // square head dim (256)
  const int nthreads = FL_NSG * 32;
  const int NO  = (DV / 8) / FL_NSG;        // O frags per simdgroup (4 @256)
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  threadgroup VPIPE_ELT sq[FL_Q * FLP_DMAX];    // Q block (halves)
  threadgroup float     so[FL_Q * FLP_DMAX];    // O accumulator (fp32)
  threadgroup float     ssf[FL_Q * FL_C];       // QK scores (fp32 -> softmax)
  threadgroup VPIPE_ELT ssp[FL_Q * FL_C];       // softmax P (-> PV)

  // Stage the Q block + zero the O accumulator (cooperative, all threads).
  for (uint e = lid; e < (uint)(FL_Q * D); e += (uint)nthreads) {
    const int j = (int)e / D, d = (int)e % D;
    const int row = q0 + j;
    sq[e] = (row < n_q) ? q[((uint)h * n_q + row) * D + d] : (VPIPE_ELT)0;
  }
  for (uint e = lid; e < (uint)(FL_Q * DV); e += (uint)nthreads) {
    so[e] = 0.0f;
  }
  float M = -INFINITY, S = 0.0f;          // per-row (this SG owns row `sg`)
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Highest query position in this tile bounds the page/block scan (uniform);
  // this SG owns query row `sg` for the per-row softmax + output.
  const int qmax = q_offset + min(q0 + FL_Q - 1, n_q - 1);
  const int qpos = q_offset + q0 + sg;

  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > qmax) { break; }            // whole tile done (uniform)
    const device VPIPE_ELT* kbase =
        kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase =
        vpool + (uint)pid * page_stride + head_off;
    for (int bs = 0; bs < nvalid; bs += FL_C) {
      if (gstart + bs > qmax) { break; }     // block past tile (uniform)
      const int bk = min(FL_C, nvalid - bs);
      // ---- QK^T: this SG contracts the full head dim for its 8 keys ----
      {
        simdgroup_matrix<float, 8, 8> mqk =
            simdgroup_matrix<float, 8, 8>(0.0f);
        const device VPIPE_ELT* pk = kbase + (uint)(bs + sg * 8) * D;
        for (int i = 0; i < D / 8; ++i) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> mq;   // Q[8 rows, 8 dims]
          simdgroup_load(mq, sq + i * 8, D);
          simdgroup_matrix<VPIPE_ELT, 8, 8> mk;   // K^T[8 dims, 8 keys]
          simdgroup_load(mk, pk + i * 8, D, ulong2(0), true);
          simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, ssf + sg * 8, FL_C);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // ---- online softmax: SG `sg` owns query row j=sg; lanes split C.
      // A key is valid iff it is within the page's valid count (k<bk, masks the
      // partial-block over-read) AND causal (global pos <= qpos). ----
      {
        const int j = sg;
        float2 s2;
        s2[0] = ssf[j * FL_C + 2 * (int)lane];
        s2[1] = ssf[j * FL_C + 2 * (int)lane + 1];
        s2 *= scale;
        const int k0 = 2 * (int)lane, k1 = k0 + 1;   // within-block key index
        const int g0 = gstart + bs + k0, g1 = gstart + bs + k1;
        if (k0 >= bk || g0 > qpos) { s2[0] = -INFINITY; }
        if (k1 >= bk || g1 > qpos) { s2[1] = -INFINITY; }
        const float lmax = simd_max(max(s2[0], s2[1]));
        const float m_new = max(M, lmax);
        const float ms = (m_new == -INFINITY) ? 1.0f : exp(M - m_new);
        float2 p2;
        p2[0] = (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - m_new);
        p2[1] = (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - m_new);
        S = S * ms + simd_sum(p2[0] + p2[1]);
        M = m_new;
        ssp[j * FL_C + 2 * (int)lane]     = (VPIPE_ELT)p2[0];
        ssp[j * FL_C + 2 * (int)lane + 1] = (VPIPE_ELT)p2[1];
        for (int d = (int)lane; d < DV; d += 32) { so[j * DV + d] *= ms; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // ---- PV: SG `sg` owns NO output-column frags (8*sg, stride 8*NSG) ----
      {
        simdgroup_matrix<float, 8, 8> lo[FLP_DMAX / 8 / FL_NSG];
        for (int ii = 0; ii < NO; ++ii) {
          simdgroup_load(lo[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
        }
        for (int cc = 0; cc < FL_C / 8; ++cc) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> vs;     // P[8q, 8k]
          simdgroup_load(vs, ssp + cc * 8, FL_C);
          for (int ii = 0; ii < NO; ++ii) {
            simdgroup_matrix<VPIPE_ELT, 8, 8> mv;   // V[8k, 8 dvcol]
            simdgroup_load(mv, vbase + (uint)(bs + cc * 8) * D + 8 * sg
                               + ii * 8 * FL_NSG, D);
            simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
          }
        }
        for (int ii = 0; ii < NO; ++ii) {
          simdgroup_store(lo[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // ---- normalize + write: SG `sg` writes query row j=sg ----
  const int j = sg;
  if (q0 + j < n_q) {
    const float inv = S > 0.0f ? 1.0f / S : 0.0f;
    const uint orow = ((uint)h * n_q + q0 + j) * D;
    for (int d = (int)lane; d < DV; d += 32) {
      out[orow + d] = (VPIPE_ELT)(so[j * DV + d] * inv);
    }
  }
}

// sdpa_paged_mma_d256_f16 -- the head_dim-256 PAGED-KV register-resident
// simdgroup_matrix flash (the existing sdpa_paged_mma_f16 is MMA_DMAX=128,
// Llama-only). Port of sdpa_causal_mma_f16 (WM=4 x WD=D/64=4 -> 16 simdgroups,
// 512 threads) to the paged pool. The key-split sdpa_paged_flash_
// f16 round-trips the O accumulator through threadgroup memory every key block
// (load lo<-so, accumulate, store so, plus a 256-wide per-block rescale so*=ms)
// + 3 barriers/block + split SG ownership -- measured ~2x the compute roofline
// at 8k (the dominant long-ctx prefill cost). This kernel instead keeps O in
// REGISTERS (MMAF_NDF 8x8 frags/simdgroup) across the whole key scan, stages
// K/V once per BK-block into tg, and does QK^T / softmax / PV exactly like the
// contiguous MMA kernel -- the only changes are the page-walked K/V load and
// the per-page causal bound. Full-causal only (Qwen3.5 full-attn, no window).
// Online-softmax (NOT bit-identical to the scalar/qtile kernels; rel-L2 within
// the flash tolerance, same fp32 accumulation as the steel GEMM). Precondition
// page_tokens % BK == 0 (BK = 4096/D = 16 @ D256; 256/512/1024 all OK).
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out 4:scale 5:D 6:Hq 7:Hkv 8:n_q
//   9:q_offset 10:page_tokens 11:n_pages 12:page_table({pid,nvalid,gstart})
// grid (MMAF_WM*WD*32, Hq, ceil(n_q/MMAF_BQ)); threadgroup (MMAF_WM*WD*32,1,1).
// NO head-dim split (WD=1, like MLX wn=1): each of the WM=4 simdgroups owns a
// query band's FULL-D O in registers (32 frags) -> 128 threads (high occupancy)
// and no QK contraction-split tg recombination. (sdpa_causal_mma's WD>1 split
// is for D=512 Gemma register pressure; D=256 holds 32 O frags fine.)
#define MMA2_DSLICE 256
#define MMA2_NDF    32          // MMA2_DSLICE / 8 -- full-D O col-frags / SG
kernel void sdpa_paged_mma_d256_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h   = (int)tid.y;
  const int qb  = (int)tid.z;
  const int kvh = h / (Hq / Hkv);
  const int q_row0 = qb * MMAF_BQ;
  if (q_row0 >= n_q) { return; }
  const int WD  = D / MMA2_DSLICE;        // 1 @256 (no head-dim split)
  const int BK  = MMAF_KVELT / D;         // 16 @256
  const int per = D / 8;                  // QK contraction frags
  const int nkf = BK / 8;                 // key-frags per block (2 @256)
  const int nthreads = MMAF_WM * WD * 32;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  const int wm  = (int)simd_gid / WD;     // query band 0..WM-1
  const int wd  = (int)simd_gid % WD;     // head-dim slice 0 (WD=1)
  const int srow0 = wm * 8;               // band's first tile row
  const int grow0 = q_row0 + srow0;       // band's first global query row
  const int dc0   = wd * MMA2_DSLICE;     // simdgroup's first O col (0 @WD=1)

  threadgroup VPIPE_ELT Ks[MMAF_KVELT];
  threadgroup VPIPE_ELT Vs[MMAF_KVELT];
  threadgroup float     Ss[MMAF_BQ * MMAF_BKMAX];
  threadgroup float     Sp[MMAF_BQ * 64];
  threadgroup VPIPE_ELT Ps[MMAF_BQ * MMAF_BKMAX];
  threadgroup float m_[MMAF_BQ];
  threadgroup float l_[MMAF_BQ];
  threadgroup float corr_[MMAF_BQ];

  const int qid = (int)lane / 4;
  const int fm  = (qid & 4) + (((int)lane / 2) % 4);    // frag row 0..7
  const int fn  = (qid & 2) * 2 + ((int)lane % 2) * 2;  // frag col 0,2,4,6

  simdgroup_matrix<float, 8, 8> O[MMA2_NDF];
  for (int c = 0; c < MMA2_NDF; ++c) {
    O[c] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  if (lid < MMAF_BQ) { m_[lid] = -INFINITY; l_[lid] = 0.0f; }

  // Highest query pos in this tile bounds the causal key scan.
  const int tile_qN   = q_offset + min(q_row0 + MMAF_BQ - 1, n_q - 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > tile_qN) { break; }                    // past tile (uniform)
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int local = 0; local < nvalid; local += BK) {
      const int gbs = gstart + local;                   // global block start
      if (gbs > tile_qN) { break; }                     // past tile (uniform)
      const int bk = min(BK, nvalid - local);
      // Stage K/V block [bk, D] into tg (zero-pad rows >= bk; masked later).
      for (uint e = lid; e < (uint)(BK * D); e += (uint)nthreads) {
        const int t = (int)e / D, d = (int)e % D;
        if (t < bk) {
          Ks[e] = kbase[(uint)(local + t) * D + d];
          Vs[e] = vbase[(uint)(local + t) * D + d];
        } else {
          Ks[e] = (VPIPE_ELT)0;
          Vs[e] = (VPIPE_ELT)0;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // QK^T: each slice contracts its per/WD frags into a partial S[8,BK].
      {
        const int pf0 = wd * (per / WD);
        simdgroup_matrix<float, 8, 8> S[MMAF_BKMAX / 8];
        for (int j = 0; j < nkf; ++j) { S[j] = simdgroup_matrix<float, 8, 8>(0.0f); }
        for (int e = 0; e < per / WD; ++e) {
          const int dd = pf0 + e;
          simdgroup_matrix<VPIPE_ELT, 8, 8> A;
          simdgroup_load(A, q + ((uint)h * n_q + grow0) * D + dd * 8, D);
          for (int j = 0; j < nkf; ++j) {
            simdgroup_matrix<VPIPE_ELT, 8, 8> B;
            simdgroup_load(B, Ks, D, ulong2(dd * 8, j * 8), true);
            simdgroup_multiply_accumulate(S[j], A, B, S[j]);
          }
        }
        for (int j = 0; j < nkf; ++j) {
          simdgroup_store(S[j], Sp + wd * MMAF_BQ * BK + srow0 * BK + j * 8, BK);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Online softmax + causal mask: 32 threads, one per query row.
      if (lid < (uint)MMAF_BQ) {
        const int r = (int)lid;
        const int qpos = q_offset + q_row0 + r;
        float lmax = -INFINITY;
        for (int j = 0; j < bk; ++j) {
          float sj = 0.0f;
          for (int w = 0; w < WD; ++w) {
            sj += Sp[w * MMAF_BQ * BK + r * BK + j];
          }
          float s = sj * scale;
          const int gkey = gbs + j;
          if (gkey > qpos) { s = -INFINITY; }
          Ss[r * BK + j] = s;
          lmax = max(lmax, s);
        }
        const float mold = m_[r];
        const float mnew = max(mold, lmax);
        const float corr = (mnew == -INFINITY) ? 1.0f : exp(mold - mnew);
        float lsum = 0.0f;
        for (int j = 0; j < bk; ++j) {
          const float s = Ss[r * BK + j];
          const float p = (s == -INFINITY) ? 0.0f : exp(s - mnew);
          Ps[r * BK + j] = (VPIPE_ELT)p;
          lsum += p;
        }
        l_[r] = l_[r] * corr + lsum;
        m_[r] = mnew;
        corr_[r] = corr;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Rescale this simdgroup's O frags by the per-row correction, then PV.
      const float cr = corr_[srow0 + fm];
      for (int c = 0; c < MMA2_NDF; ++c) {
        O[c].thread_elements()[0] *= cr;
        O[c].thread_elements()[1] *= cr;
      }
      for (int kk = 0; kk < nkf; ++kk) {
        simdgroup_matrix<VPIPE_ELT, 8, 8> A2;
        simdgroup_load(A2, Ps + srow0 * BK + kk * 8, BK);
        for (int c = 0; c < MMA2_NDF; ++c) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> B2;
          simdgroup_load(B2, Vs, D, ulong2(dc0 + c * 8, kk * 8), false);
          simdgroup_multiply_accumulate(O[c], A2, B2, O[c]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Normalize + write O frags (guard rows past n_q).
  const int row = grow0 + fm;
  if (row < n_q) {
    const float denom = l_[srow0 + fm];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    const uint orow = ((uint)h * n_q + row) * D;
    for (int c = 0; c < MMA2_NDF; ++c) {
      const int col = dc0 + c * 8 + fn;
      out[orow + col]     = VPIPE_ELT(O[c].thread_elements()[0] * inv);
      out[orow + col + 1] = VPIPE_ELT(O[c].thread_elements()[1] * inv);
    }
  }
}

// sdpa_paged_flash_d512_f16 -- D=512 sibling of sdpa_paged_flash_f16 (which is
// budgeted FLP_DMAX=256 for Qwen3.5's head_dim 256). Identical KEY-SPLIT
// simdgroup_matrix flash over the PAGED pool, but the threadgroup Q/O tiles and
// PV register frags are sized to FL_DMAX (512) -- the contiguous
// sdpa_causal_flash_f16 already runs this budget (~27 KB tg) on M4, so this is
// the page-walked sibling of THAT kernel for the GLOBAL (full-attention,
// head_dim 512) Gemma-4 layers. It fills the M4 tier the paged prefill lacked:
// matmul2d sdpa_paged_mma2_d512 is M5-only, and the scalar sdpa_paged_causal is
// the O(ctx)/32-lane fallback that tanked M4 prefill. simdgroup_matrix runs on
// every Apple GPU. Full-causal only (global layers, no window). Same buffer
// contract + page_tokens%FL_C==0 precondition as sdpa_paged_flash_f16.
//   0:q[Hq,n_q,D] 1:kpool 2:vpool 3:out 4:scale 5:D 6:Hq 7:Hkv 8:n_q
//   9:q_offset 10:page_tokens 11:n_pages 12:page_table({pid,nvalid,gstart})
// grid (FL_NSG*32, Hq, ceil(n_q/FL_Q)); threadgroup (FL_NSG*32,1,1).
kernel void sdpa_paged_flash_d512_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h   = (int)tid.y;
  const int q0  = (int)tid.z * FL_Q;
  if (q0 >= n_q) { return; }
  const int kvh = h / (Hq / Hkv);
  const int sg  = (int)simd_gid;          // 0..7
  const int DV  = D;                       // square head dim (512)
  const int nthreads = FL_NSG * 32;
  const int NO  = (DV / 8) / FL_NSG;        // O frags per simdgroup (8 @512)
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  threadgroup VPIPE_ELT sq[FL_Q * FL_DMAX];     // Q block (halves)
  threadgroup float     so[FL_Q * FL_DMAX];     // O accumulator (fp32)
  threadgroup float     ssf[FL_Q * FL_C];       // QK scores (fp32 -> softmax)
  threadgroup VPIPE_ELT ssp[FL_Q * FL_C];       // softmax P (-> PV)

  // Stage the Q block + zero the O accumulator (cooperative, all threads).
  for (uint e = lid; e < (uint)(FL_Q * D); e += (uint)nthreads) {
    const int j = (int)e / D, d = (int)e % D;
    const int row = q0 + j;
    sq[e] = (row < n_q) ? q[((uint)h * n_q + row) * D + d] : (VPIPE_ELT)0;
  }
  for (uint e = lid; e < (uint)(FL_Q * DV); e += (uint)nthreads) {
    so[e] = 0.0f;
  }
  float M = -INFINITY, S = 0.0f;          // per-row (this SG owns row `sg`)
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int qmax = q_offset + min(q0 + FL_Q - 1, n_q - 1);
  const int qpos = q_offset + q0 + sg;

  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > qmax) { break; }            // whole tile done (uniform)
    const device VPIPE_ELT* kbase =
        kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase =
        vpool + (uint)pid * page_stride + head_off;
    for (int bs = 0; bs < nvalid; bs += FL_C) {
      if (gstart + bs > qmax) { break; }     // block past tile (uniform)
      const int bk = min(FL_C, nvalid - bs);
      // ---- QK^T: this SG contracts the full head dim for its 8 keys ----
      {
        simdgroup_matrix<float, 8, 8> mqk =
            simdgroup_matrix<float, 8, 8>(0.0f);
        const device VPIPE_ELT* pk = kbase + (uint)(bs + sg * 8) * D;
        for (int i = 0; i < D / 8; ++i) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> mq;   // Q[8 rows, 8 dims]
          simdgroup_load(mq, sq + i * 8, D);
          simdgroup_matrix<VPIPE_ELT, 8, 8> mk;   // K^T[8 dims, 8 keys]
          simdgroup_load(mk, pk + i * 8, D, ulong2(0), true);
          simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
        simdgroup_store(mqk, ssf + sg * 8, FL_C);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // ---- online softmax: SG `sg` owns query row j=sg; lanes split C. A key
      // is valid iff within the page's valid count (k<bk) AND causal. ----
      {
        const int j = sg;
        float2 s2;
        s2[0] = ssf[j * FL_C + 2 * (int)lane];
        s2[1] = ssf[j * FL_C + 2 * (int)lane + 1];
        s2 *= scale;
        const int k0 = 2 * (int)lane, k1 = k0 + 1;   // within-block key index
        const int g0 = gstart + bs + k0, g1 = gstart + bs + k1;
        if (k0 >= bk || g0 > qpos) { s2[0] = -INFINITY; }
        if (k1 >= bk || g1 > qpos) { s2[1] = -INFINITY; }
        const float lmax = simd_max(max(s2[0], s2[1]));
        const float m_new = max(M, lmax);
        const float ms = (m_new == -INFINITY) ? 1.0f : exp(M - m_new);
        float2 p2;
        p2[0] = (s2[0] == -INFINITY) ? 0.0f : exp(s2[0] - m_new);
        p2[1] = (s2[1] == -INFINITY) ? 0.0f : exp(s2[1] - m_new);
        S = S * ms + simd_sum(p2[0] + p2[1]);
        M = m_new;
        ssp[j * FL_C + 2 * (int)lane]     = (VPIPE_ELT)p2[0];
        ssp[j * FL_C + 2 * (int)lane + 1] = (VPIPE_ELT)p2[1];
        for (int d = (int)lane; d < DV; d += 32) { so[j * DV + d] *= ms; }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // ---- PV: SG `sg` owns NO output-column frags (8*sg, stride 8*NSG) ----
      {
        simdgroup_matrix<float, 8, 8> lo[FL_DMAX / 8 / FL_NSG];
        for (int ii = 0; ii < NO; ++ii) {
          simdgroup_load(lo[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
        }
        for (int cc = 0; cc < FL_C / 8; ++cc) {
          simdgroup_matrix<VPIPE_ELT, 8, 8> vs;     // P[8q, 8k]
          simdgroup_load(vs, ssp + cc * 8, FL_C);
          for (int ii = 0; ii < NO; ++ii) {
            simdgroup_matrix<VPIPE_ELT, 8, 8> mv;   // V[8k, 8 dvcol]
            simdgroup_load(mv, vbase + (uint)(bs + cc * 8) * D + 8 * sg
                               + ii * 8 * FL_NSG, D);
            simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
          }
        }
        for (int ii = 0; ii < NO; ++ii) {
          simdgroup_store(lo[ii], so + 8 * sg + ii * 8 * FL_NSG, DV);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // ---- normalize + write: SG `sg` writes query row j=sg ----
  const int j = sg;
  if (q0 + j < n_q) {
    const float inv = S > 0.0f ? 1.0f / S : 0.0f;
    const uint orow = ((uint)h * n_q + q0 + j) * D;
    for (int d = (int)lane; d < DV; d += 32) {
      out[orow + d] = (VPIPE_ELT)(so[j * DV + d] * inv);
    }
  }
}

// causal_softmax_rows_f16 -- in-place causal-masked softmax over a
// MATERIALIZED score matrix scores[M, N] (M = n_q query rows, N = T_kv keys),
// for the Gemma-4 GLOBAL (head_dim 512) prefill MATERIALIZED path: per-head
// Q.K^T (steel GEMM) -> THIS softmax -> P.V (steel GEMM), mirroring MLX's
// fallback for head_dim 512 (no fused flash). One threadgroup per query row i;
// row i is query position (q_offset + i). Key j is VALID iff j <= q_offset + i
// (causal); invalid keys -> prob 0. Math matches the online-softmax flash
// (sdpa_paged_flash_d512): scaled = scores * scale, p = exp(scaled - row_max)
// over valid j, P = p / sum, f32 accumulation, PLAIN exp (no exp2/M_LOG2E).
//   0: scores (device VPIPE_ELT*)  [M, N]  (in-place: raw Q.K^T in, P out)
//   1: M        (constant int&)    n_q
//   2: N        (constant int&)    T_kv
//   3: q_offset (constant int&)
//   4: scale    (constant float&)
// grid (CSM_TG, M, 1); threadgroup (CSM_TG, 1, 1).
#define CSM_TG 256
kernel void causal_softmax_rows_f16(
    device VPIPE_ELT*  scores   [[buffer(0)]],
    constant int&      M        [[buffer(1)]],
    constant int&      N        [[buffer(2)]],
    constant int&      q_offset [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      window   [[buffer(5)]],
    constant int&      banded   [[buffer(6)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 ltid     [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]])
{
  const uint row = tid.y;
  if ((int)row >= M) { return; }
  const uint lid = ltid.x;
  device VPIPE_ELT* sr = scores + (uint)row * N;
  const int kmax = q_offset + (int)row;   // last valid key (inclusive)
  // Trailing-window lower bound (window>0); else no lower bound (causal).
  const int kmin = (window > 0) ? max(0, kmax - window + 1) : 0;

  // Scan only the PV band [k0, kc) -- the EXACT key range the materialized PV
  // (dense_gemm_t_pvcausal, BM=BK=32) reads for this row's 32-row block. The
  // PV never reads outside it, so writing 0/p there is unnecessary; matching
  // the band turns this from O(N) to O(window) per row (and halves the global
  // causal scan). MUST mirror pvcausal's k0/Kc formula or PV reads stale data.
  // banded==0 (the plain dense_t PV A/B path reads all [0,N)) -> full scan.
  const int BLK = 32;
  int k0 = 0;
  int kc = N;
  if (banded != 0) {
    const int by = ((int)row / BLK) * BLK;        // PV block start
    kc = q_offset + by + BLK;                      // Kc (exclusive)
    kc = (kc < N) ? ((kc + BLK - 1) / BLK) * BLK : N;
    if (window > 0) {
      const int lo = q_offset + by - window + 1;
      k0 = (lo > 0) ? (lo / BLK) * BLK : 0;
    }
  }

  threadgroup float pmax[CSM_TG / 32];
  threadgroup float psum[CSM_TG / 32];

  // --- pass 1: row max over valid (scaled) scores ---
  float m = -INFINITY;
  for (int j = k0 + (int)lid; j < kc; j += CSM_TG) {
    if (j >= kmin && j <= kmax) {
      const float s = float(sr[j]) * scale;
      m = max(m, s);
    }
  }
  m = simd_max(m);
  if (simd_lid == 0) { pmax[simd_gid] = m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < CSM_TG / 32) ? pmax[simd_lid] : -INFINITY;
    a = simd_max(a);
    if (simd_lid == 0) { pmax[0] = a; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_max = pmax[0];

  // --- pass 2: exp(scaled - max) over valid j, accumulate sum ---
  float ssum = 0.0f;
  for (int j = k0 + (int)lid; j < kc; j += CSM_TG) {
    float p = 0.0f;
    if (j >= kmin && j <= kmax) {
      const float s = float(sr[j]) * scale;
      p = (row_max == -INFINITY) ? 0.0f : exp(s - row_max);
    }
    sr[j] = (VPIPE_ELT)p;   // store unnormalized p; normalize after sum known
    ssum += p;
  }
  ssum = simd_sum(ssum);
  if (simd_lid == 0) { psum[simd_gid] = ssum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float a = (simd_lid < CSM_TG / 32) ? psum[simd_lid] : 0.0f;
    a = simd_sum(a);
    if (simd_lid == 0) { psum[0] = a; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = psum[0] > 0.0f ? 1.0f / psum[0] : 0.0f;

  // --- pass 3: normalize (over the same band) ---
  for (int j = k0 + (int)lid; j < kc; j += CSM_TG) {
    sr[j] = (VPIPE_ELT)(float(sr[j]) * inv);
  }
}

// sdpa_causal_mb_f16 -- MULTI-simdgroup DECODE attention over a CONTIGUOUS
// [Hkv, kv_stride, D] KV cache, for Gemma-4 (head_dim 256 / 512). The scalar
// sdpa_causal_f16 scans all T_kv keys with ONE simdgroup (32 lanes), so its
// cost grows ~O(ctx) on the per-layer critical path -- at decode it was ~4.6x
// MLX's attention. Here BN = 4096/D simdgroups (16 @ D256, 8 @ D512) split the
// key range: simdgroup g scans keys first+g, first+g+BN, ... <= last with its
// own private (m,l,O) online-softmax accumulator, then a threadgroup cross-
// reduce combines the BN partials. ~BN x KV-scan parallelism. window > 0 adds a
// trailing sliding window (Gemma sliding layers); window <= 0 == full causal.
// Bit-equivalent (online-softmax) to sdpa_causal_f16 / sdpa_causal_window_f16.
//   0:q[Hq,n_q,D] 1:k 2:v[Hkv,kv_stride,D] 3:out 4:scale 5:T_kv 6:D 7:Hq
//   8:Hkv 9:n_q 10:q_offset 11:kv_stride 12:window
// grid (BN*32, Hq, n_q); threadgroup (BN*32,1,1). sh_o = BN*D = 4096 f32 (16KB).
#define SDPA_MBC_ELT 4096   // BN*D invariant; BN = 4096/D (16@256, 8@512)
kernel void sdpa_causal_mb_f16(
    const device VPIPE_ELT* q        [[buffer(0)]],
    const device VPIPE_ELT* k        [[buffer(1)]],
    const device VPIPE_ELT* v        [[buffer(2)]],
    device VPIPE_ELT*       out      [[buffer(3)]],
    constant float&    scale    [[buffer(4)]],
    constant int&      T_kv     [[buffer(5)]],
    constant int&      D        [[buffer(6)]],
    constant int&      Hq       [[buffer(7)]],
    constant int&      Hkv      [[buffer(8)]],
    constant int&      n_q      [[buffer(9)]],
    constant int&      q_offset [[buffer(10)]],
    constant int&      kv_stride[[buffer(11)]],
    constant int&      window   [[buffer(12)]],
    constant int&      ring_cap [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int BN = SDPA_MBC_ELT / D;        // 16 @256, 8 @512
  const int q_pos = q_offset + qi;
  const int last = min(T_kv - 1, q_pos);
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;

  threadgroup float sh_m[16];             // BN <= 16
  threadgroup float sh_l[16];
  threadgroup float sh_o[SDPA_MBC_ELT];   // BN*D

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  // This simdgroup's strided slice of [first, last]. Loop bound is uniform
  // across the simdgroup's 32 lanes (same simd_gid) -> simd_sum well-formed.
  float m = -INFINITY, l = 0.0f;
  for (int t = first + (int)simd_gid; t <= last; t += BN) {
    // Ring layout (sliding-window layers): the key/value buffer is sized
    // to a bounded capacity, so logical position t reads physical slot
    // t % ring_cap. ring_cap == 0 == full layer (slot == t, no modulo).
    const uint slot = (ring_cap > 0) ? (uint)(t % ring_cap) : (uint)t;
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      dot += qreg[p] * float(kkv[slot * D + lane * per + p]);
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = exp(m - m_new);
    const float pj = exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      acc[p] = acc[p] * corr + pj * float(vkv[slot * D + lane * per + p]);
    }
    m = m_new;
  }
  if (lane == 0) { sh_m[simd_gid] = m; sh_l[simd_gid] = l; }
  for (int p = 0; p < per; ++p) {
    sh_o[(int)simd_gid * D + lane * per + p] = acc[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cross-simdgroup online-softmax reduce. Each thread writes ceil(D/(BN*32))
  // output elements (D=512,BN=8 -> 2/thread; D=256,BN=16 -> 1/thread).
  const uint ntg = (uint)BN * 32;
  const uint ltid = simd_gid * 32 + lane;
  float new_max = -INFINITY;
  for (int g = 0; g < BN; ++g) { new_max = max(new_max, sh_m[g]); }
  float new_sum = 0.0f;
  if (new_max > -INFINITY) {
    for (int g = 0; g < BN; ++g) { new_sum += sh_l[g] * exp(sh_m[g] - new_max); }
  }
  const float inv = new_sum > 0.0f ? 1.0f / new_sum : 0.0f;
  for (uint d = ltid; d < (uint)D; d += ntg) {
    float ov = 0.0f;
    if (new_max > -INFINITY) {
      for (int g = 0; g < BN; ++g) {
        ov += sh_o[(uint)g * D + d] * exp(sh_m[g] - new_max);
      }
    }
    out[((uint)h * n_q + qi) * D + d] = VPIPE_ELT(ov * inv);
  }
}

// sdpa_paged_mb_f16 -- MULTI-simdgroup variant of sdpa_paged_causal_f16
// for long contexts. The single-simdgroup kernels above scan the KV
// serially with only 32-lane parallelism, so attention cost grows ~O(
// ctx) on the per-layer critical path and dominates at long context.
// Here BN=32 simdgroups per (head, query-pos) split the KV: simdgroup g
// processes tokens g, g+BN, g+2BN, ... of EACH page with its own private
// (m,l,O) accumulator (carry stays in registers across pages -- one
// dispatch, no host carry threading), then a threadgroup cross-reduce
// combines the 32 partials via online-softmax rescale. Recovers ~32x
// attention parallelism. Cross-reduce copied from the in-tree
// qwen3_5_paged_attn_block kernel. Same buffer/layout contract as
// sdpa_paged_causal_f16.
// grid (32*BN, H_q, n_q); threadgroup (32*BN,1,1) == 1024 threads.
#define SDPA_MB_BN 32   // simdgroups per threadgroup (KV-stride width)

kernel void sdpa_paged_mb_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int q_pos = q_offset + qi;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;

  // Cross-simdgroup reduce scratch. sh_o holds each simdgroup's full
  // D-vector O-partial (distributed across its lanes). Sized for D<=128
  // (Llama head_dim); larger heads should use the single-simdgroup
  // sdpa_paged_causal_f16.
  threadgroup float sh_m[SDPA_MB_BN];
  threadgroup float sh_l[SDPA_MB_BN];
  threadgroup float sh_o[SDPA_MB_BN * 128];

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }

  // Per-simdgroup running (max, denom) over its strided KV subset.
  float m = -INFINITY;
  float l = 0.0f;
  bool done = false;
  for (int pg = 0; pg < n_pages && !done; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int t = (int)simd_gid; t < nvalid; t += SDPA_MB_BN) {
      if (gstart + t > q_pos) { done = true; break; }  // positions monotonic
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) {
        dot += qreg[p] * float(kbase[(uint)t * D + lane * per + p]);
      }
      dot = simd_sum(dot);
      const float m_new = max(m, dot);
      const float corr = exp(m - m_new);
      const float pj = exp(dot - m_new);
      l = l * corr + pj;
      for (int p = 0; p < per; ++p) {
        acc[p] = acc[p] * corr + pj * float(vbase[(uint)t * D + lane * per + p]);
      }
      m = m_new;
    }
  }

  // Cross-simdgroup online-softmax reduce. Stash each simdgroup's (m,l)
  // + its full O-partial (lane L holds D-indices [L*per, L*per+per)),
  // then have the first D threads each combine ONE output element across
  // all 32 simdgroups: out[d] = (sum_g O_g[d]*exp(m_g-M)) / (sum_g
  // l_g*exp(m_g-M)), M = max_g m_g. Idle simdgroups carry m=-inf so
  // exp(m_g-M)=0 and drop out cleanly.
  if (lane == 0) {
    sh_m[simd_gid] = m;
    sh_l[simd_gid] = l;
  }
  for (int p = 0; p < per; ++p) {
    sh_o[(int)simd_gid * D + lane * per + p] = acc[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int ltid = (int)simd_gid * 32 + (int)lane;   // 0..1023
  if (ltid < D) {
    float new_max = -INFINITY;
    for (int g = 0; g < SDPA_MB_BN; ++g) { new_max = max(new_max, sh_m[g]); }
    float new_sum = 0.0f;
    float ov = 0.0f;
    for (int g = 0; g < SDPA_MB_BN; ++g) {
      const float f = exp(sh_m[g] - new_max);
      new_sum += sh_l[g] * f;
      ov += sh_o[g * D + ltid] * f;
    }
    out[((uint)h * n_q + qi) * D + ltid] = VPIPE_ELT(ov / new_sum);
  }
}

// sdpa_paged_mb256_f16 -- multi-simdgroup paged causal attention for
// head_dim 256 (Qwen3.5 full-attention). Same algorithm as
// sdpa_paged_mb_f16 but BN=16 simdgroups (so the per-simdgroup O-partial
// table sh_o[16*256] f32 = 16 KB fits the 32 KB threadgroup limit, where
// BN=32 would need 32 KB and overflow). 16x KV-scan parallelism vs the
// single-simdgroup sdpa_paged_causal_f16 -- recovers flat decode latency
// at long context. Same buffer contract as sdpa_paged_causal_f16.
// grid (32*16, Hq, n_q); threadgroup (512, 1, 1).
#define SDPA_MB256_BN 16
kernel void sdpa_paged_mb256_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int q_pos = q_offset + qi;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;

  threadgroup float sh_m[SDPA_MB256_BN];
  threadgroup float sh_l[SDPA_MB256_BN];
  threadgroup float sh_o[SDPA_MB256_BN * 256];

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  bool done = false;
  for (int pg = 0; pg < n_pages && !done; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int t = (int)simd_gid; t < nvalid; t += SDPA_MB256_BN) {
      if (gstart + t > q_pos) { done = true; break; }
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) {
        dot += qreg[p] * float(kbase[(uint)t * D + lane * per + p]);
      }
      dot = simd_sum(dot);
      const float m_new = max(m, dot);
      const float corr = exp(m - m_new);
      const float pj = exp(dot - m_new);
      l = l * corr + pj;
      for (int p = 0; p < per; ++p) {
        acc[p] = acc[p] * corr + pj * float(vbase[(uint)t * D + lane * per + p]);
      }
      m = m_new;
    }
  }
  if (lane == 0) { sh_m[simd_gid] = m; sh_l[simd_gid] = l; }
  for (int p = 0; p < per; ++p) {
    sh_o[(int)simd_gid * D + lane * per + p] = acc[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ltid = (int)simd_gid * 32 + (int)lane;   // 0..511
  if (ltid < D) {
    float new_max = -INFINITY;
    for (int g = 0; g < SDPA_MB256_BN; ++g) { new_max = max(new_max, sh_m[g]); }
    float new_sum = 0.0f, ov = 0.0f;
    for (int g = 0; g < SDPA_MB256_BN; ++g) {
      const float f = exp(sh_m[g] - new_max);
      new_sum += sh_l[g] * f;
      ov += sh_o[g * D + ltid] * f;
    }
    out[((uint)h * n_q + qi) * D + ltid] = VPIPE_ELT(ov / new_sum);
  }
}

// ---- Shared-prefix batched decode attention (head_dim 256) --------------
// The N branches of a VQA fanout share a common prefix (image + describe);
// their shared-prefix K/V is byte-identical (refcount-shared pages), so a
// per-branch decode SDPA re-reads it N times. These two kernels split decode
// attention into a flash-attention SPLIT (the same online-softmax merge the
// mb256 kernel already does across its simdgroups, just split shared-vs-
// private instead of strided):
//   A: sdpa_shared_mb256_f16 -- read the shared pages ONCE, apply each key to
//      ALL N branch query rows -> UN-normalized partials (O_acc, m, l).
//   B: sdpa_merge_mb256_f16  -- per branch: scan its PRIVATE pages (causal),
//      fold in the branch's shared partial, normalize -> final output.
// head_dim is fixed at 256 (Qwen3.5 full-attention); other dims use the
// per-branch mb256 / scalar paths.

#define SDPA_SHARED_MAXN     4   // batched query rows / threadgroup (regs)
#define SDPA_SHARED_D256_PER 8   // D/32 for the fixed head_dim 256

// PHASE A. q [N, Hq, D]; partials O_acc [N, Hq, D] f32 (un-normalized), m
// [N, Hq] f32, l [N, Hq] f32. The shared prefix lies entirely before every
// branch's decode position -> NO causal mask. Only the first
// SDPA_SHARED_MAXN branches are batched here; the caller keeps N <= MAXN.
//   0:q 1:kpool 2:vpool 3:o_acc(f32) 4:m(f32) 5:l(f32) 6:scale 7:D 8:Hq
//   9:Hkv 10:N 11:page_tokens 12:n_shared_pages 13:shared_page_table
// grid (32*16, Hq, 1); threadgroup (512,1,1).
kernel void sdpa_shared_mb256_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      D          [[buffer(7)]],
    constant int&      Hq         [[buffer(8)]],
    constant int&      Hkv        [[buffer(9)]],
    constant int&      N          [[buffer(10)]],
    constant int&      page_tokens[[buffer(11)]],
    constant int&      n_pages    [[buffer(12)]],
    const device int*  page_table [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;
  const int Nb = min(N, SDPA_SHARED_MAXN);

  threadgroup float sh_m[SDPA_MB256_BN];
  threadgroup float sh_l[SDPA_MB256_BN];
  threadgroup float sh_o[SDPA_MB256_BN * 256];

  // Per-branch q + accumulator in registers (per is fixed 8 for D=256).
  float qreg[SDPA_SHARED_MAXN][SDPA_SHARED_D256_PER];
  float acc[SDPA_SHARED_MAXN][SDPA_SHARED_D256_PER];
  float m[SDPA_SHARED_MAXN], l[SDPA_SHARED_MAXN];
  for (int n = 0; n < Nb; ++n) {
    const device VPIPE_ELT* qh = q + ((uint)n * Hq + h) * D;
    for (int p = 0; p < per; ++p) {
      qreg[n][p] = float(qh[lane * per + p]) * scale;
      acc[n][p] = 0.0f;
    }
    m[n] = -INFINITY; l[n] = 0.0f;
  }

  // Scan the shared pages ONCE; each loaded key/value updates all Nb rows.
  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int t = (int)simd_gid; t < nvalid; t += SDPA_MB256_BN) {
      float kreg[SDPA_SHARED_D256_PER], vreg[SDPA_SHARED_D256_PER];
      for (int p = 0; p < per; ++p) {
        kreg[p] = float(kbase[(uint)t * D + lane * per + p]);
        vreg[p] = float(vbase[(uint)t * D + lane * per + p]);
      }
      for (int n = 0; n < Nb; ++n) {
        float dot = 0.0f;
        for (int p = 0; p < per; ++p) { dot += qreg[n][p] * kreg[p]; }
        dot = simd_sum(dot);
        const float m_new = max(m[n], dot);
        const float corr = exp(m[n] - m_new);
        const float pj = exp(dot - m_new);
        l[n] = l[n] * corr + pj;
        for (int p = 0; p < per; ++p) { acc[n][p] = acc[n][p] * corr + pj * vreg[p]; }
        m[n] = m_new;
      }
    }
  }

  // Cross-simdgroup online-softmax reduce, per branch (reuse sh_*). Writes
  // UN-normalized O_acc + m + l so the merge kernel can fold the private
  // partial in before dividing.
  for (int n = 0; n < Nb; ++n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { sh_m[simd_gid] = m[n]; sh_l[simd_gid] = l[n]; }
    for (int p = 0; p < per; ++p) {
      sh_o[(int)simd_gid * D + lane * per + p] = acc[n][p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ltid = (int)simd_gid * 32 + (int)lane;   // 0..511
    if (ltid < D) {
      float new_max = -INFINITY;
      for (int g = 0; g < SDPA_MB256_BN; ++g) { new_max = max(new_max, sh_m[g]); }
      float new_sum = 0.0f, ov = 0.0f;
      if (new_max > -INFINITY) {
        for (int g = 0; g < SDPA_MB256_BN; ++g) {
          const float f = (sh_m[g] > -INFINITY) ? exp(sh_m[g] - new_max) : 0.0f;
          new_sum += sh_l[g] * f;
          ov += sh_o[g * D + ltid] * f;
        }
      }
      o_acc[((uint)n * Hq + h) * D + ltid] = ov;     // UN-normalized
      if (ltid == 0) {
        m_out[(uint)n * Hq + h] = new_max;
        l_out[(uint)n * Hq + h] = new_sum;
      }
    }
  }
}

// PHASE B. One branch: online-softmax over its PRIVATE pages (causal by
// global KV position), then MERGE the branch's phase-A shared partial
// (o_acc_s/m_s/l_s for this head, caller-offset to the branch) and normalize.
// n_pages == 0 (no private pages yet) -> output is the normalized shared
// partial. Same scan as sdpa_paged_mb256_f16 + one extra reduce component.
//   0:q[Hq,1,D] 1:kpool 2:vpool 3:out 4:scale 5:D 6:Hq 7:Hkv 8:n_q
//   9:q_offset 10:page_tokens 11:n_priv_pages 12:priv_page_table
//   13:o_acc_s[Hq,D](f32) 14:m_s[Hq](f32) 15:l_s[Hq](f32)
// grid (32*16, Hq, 1); threadgroup (512,1,1).
kernel void sdpa_merge_mb256_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    const device float* o_acc_s   [[buffer(13)]],
    const device float* m_s_in    [[buffer(14)]],
    const device float* l_s_in    [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int q_pos = q_offset + qi;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;

  threadgroup float sh_m[SDPA_MB256_BN];
  threadgroup float sh_l[SDPA_MB256_BN];
  threadgroup float sh_o[SDPA_MB256_BN * 256];

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  bool done = false;
  for (int pg = 0; pg < n_pages && !done; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int t = (int)simd_gid; t < nvalid; t += SDPA_MB256_BN) {
      if (gstart + t > q_pos) { done = true; break; }
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) {
        dot += qreg[p] * float(kbase[(uint)t * D + lane * per + p]);
      }
      dot = simd_sum(dot);
      const float m_new = max(m, dot);
      const float corr = exp(m - m_new);
      const float pj = exp(dot - m_new);
      l = l * corr + pj;
      for (int p = 0; p < per; ++p) {
        acc[p] = acc[p] * corr + pj * float(vbase[(uint)t * D + lane * per + p]);
      }
      m = m_new;
    }
  }
  if (lane == 0) { sh_m[simd_gid] = m; sh_l[simd_gid] = l; }
  for (int p = 0; p < per; ++p) {
    sh_o[(int)simd_gid * D + lane * per + p] = acc[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ltid = (int)simd_gid * 32 + (int)lane;   // 0..511
  if (ltid < D) {
    // Combine the 16 private-page simdgroup partials AND the shared partial.
    const float ms = m_s_in[h];
    float new_max = ms;
    for (int g = 0; g < SDPA_MB256_BN; ++g) { new_max = max(new_max, sh_m[g]); }
    float new_sum = 0.0f, ov = 0.0f;
    if (new_max > -INFINITY) {
      for (int g = 0; g < SDPA_MB256_BN; ++g) {
        const float f = (sh_m[g] > -INFINITY) ? exp(sh_m[g] - new_max) : 0.0f;
        new_sum += sh_l[g] * f;
        ov += sh_o[g * D + ltid] * f;
      }
      const float fs = (ms > -INFINITY) ? exp(ms - new_max) : 0.0f;
      new_sum += l_s_in[h] * fs;
      ov += o_acc_s[(uint)h * D + ltid] * fs;
    }
    out[((uint)h * n_q + qi) * D + ltid] =
        VPIPE_ELT(new_sum > 0.0f ? ov / new_sum : 0.0f);
  }
}

// ---- Flash-decode-GQA paged attention (head_dim 256, decode n_q=1) -------
// The mb256 decode kernel launches one threadgroup per Q-HEAD, so the
// Hq/Hkv query heads that share a KV head each re-scan the SAME K/V --
// GQA-redundant: 4x the KV bandwidth for Qwen3.5's 16/4, which is the
// dominant long-context decode cost (attn ~75 GB/s vs FFN's ~133). These two
// kernels group by KV HEAD instead:
//   A: sdpa_paged_gqa_mb256_f16 -- each (kv, split) threadgroup is a SINGLE
//      simdgroup that reads its KV-head position-slice [lo,hi) ONCE and
//      applies every key/value to ALL G = Hq/Hkv query heads of the group ->
//      KV read exactly once per head. SPLIT threadgroups per KV head (along
//      the position axis) restore the parallelism the single-simdgroup form
//      would lose, with no threadgroup-memory O-table (the per-q-head merge
//      across 16 simdgroups is what caps naive 4-head grouping at the 32 KB
//      limit). Writes UN-normalized per-(qhead,split) partials.
//   B: sdpa_gqa_merge_f16 -- folds the SPLIT partials per query head and
//      normalizes. Online softmax in fp32; same math as the mb256
//      cross-simdgroup reduce, just split along position instead of stride.
// Gated to G <= SDPA_GQA_MAXG (register budget) and head_dim 256; other
// shapes use the mb256 / scalar paths. Qwen3.5-4B is G=4; Qwen3.6-27B is
// G=6 (Hq=24/Hkv=4); Qwen3.5-MoE (35B-A3B) is G=8 (Hq=16/Hkv=2). MAXG sizes
// the all-G kernel's per-head reg arrays so the VPIPE_GQA_NO_VEC A/B path is
// safe up to that G -- but production uses the VEC kernel even at high G:
// decode attention is LATENCY-bound, not KV-byte-bound, so vec (per-head
// simdgroup, one simd_sum/key) beats the all-G 1x-read form (measured +24%
// @8k on the 27B). The vec/kbl kernels don't use MAXG (one head per
// simdgroup), so they have no G ceiling here.
#define SDPA_GQA_MAXG     8   // max Hq/Hkv for the all-G kernel (reg budget)
#define SDPA_GQA_D256_PER 8   // D/32 for the fixed head_dim 256
#define SDPA_GQA_UK       4   // decode key-unroll (latency overlap; see vec)

// PHASE A. q [Hq, 1, D]; partials o_acc [Hq, SPLIT, D] f32 (un-normalized),
// m [Hq, SPLIT] f32, l [Hq, SPLIT] f32. The caller clamps total = q_pos+1 so
// every key this kernel scans is causal (<= q_pos) -- no per-key mask.
//   0:q 1:kpool 2:vpool 3:o_acc(f32) 4:m(f32) 5:l(f32) 6:scale 7:D 8:Hq
//   9:Hkv 10:q_offset 11:page_tokens 12:n_pages 13:page_table 14:split
// grid (32, Hkv, split); threadgroup (32, 1, 1).
kernel void sdpa_paged_gqa_mb256_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      D          [[buffer(7)]],
    constant int&      Hq         [[buffer(8)]],
    constant int&      Hkv        [[buffer(9)]],
    constant int&      q_offset   [[buffer(10)]],
    constant int&      page_tokens[[buffer(11)]],
    constant int&      n_pages    [[buffer(12)]],
    const device int*  page_table [[buffer(13)]],
    constant int&      split      [[buffer(14)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int per = D / 32;
  const int q_pos = q_offset;
  const int total = q_pos + 1;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;

  // This split's contiguous global-position range [lo, hi) (all <= q_pos).
  const int chunk = (total + split - 1) / split;
  const int lo = s * chunk;
  const int hi = min((s + 1) * chunk, total);

  float qreg[SDPA_GQA_MAXG][SDPA_GQA_D256_PER];
  float acc[SDPA_GQA_MAXG][SDPA_GQA_D256_PER];
  float m[SDPA_GQA_MAXG], l[SDPA_GQA_MAXG];
  for (int g = 0; g < G; ++g) {
    const device VPIPE_ELT* qh = q + (uint)(kv * G + g) * D;
    for (int p = 0; p < per; ++p) {
      qreg[g][p] = float(qh[lane * per + p]) * scale;
      acc[g][p] = 0.0f;
    }
    m[g] = -INFINITY; l[g] = 0.0f;
  }

  if (lo < hi) {
    for (int pg = 0; pg < n_pages; ++pg) {
      const int pid    = page_table[pg * 3 + 0];
      const int nvalid = page_table[pg * 3 + 1];
      const int gstart = page_table[pg * 3 + 2];
      const int tlo = max(0, lo - gstart);
      const int thi = min(nvalid, hi - gstart);
      if (tlo >= thi) { continue; }
      const device VPIPE_ELT* kbase =
          kpool + (uint)pid * page_stride + head_off;
      const device VPIPE_ELT* vbase =
          vpool + (uint)pid * page_stride + head_off;
      for (int t = tlo; t < thi; ++t) {
        float kreg[SDPA_GQA_D256_PER], vreg[SDPA_GQA_D256_PER];
        for (int p = 0; p < per; ++p) {
          kreg[p] = float(kbase[(uint)t * D + lane * per + p]);
          vreg[p] = float(vbase[(uint)t * D + lane * per + p]);
        }
        for (int g = 0; g < G; ++g) {
          float dot = 0.0f;
          for (int p = 0; p < per; ++p) { dot += qreg[g][p] * kreg[p]; }
          dot = simd_sum(dot);
          const float m_new = max(m[g], dot);
          const float corr = exp(m[g] - m_new);
          const float pj = exp(dot - m_new);
          l[g] = l[g] * corr + pj;
          for (int p = 0; p < per; ++p) {
            acc[g][p] = acc[g][p] * corr + pj * vreg[p];
          }
          m[g] = m_new;
        }
      }
    }
  }

  // UN-normalized partials per (q-head, split). Empty split -> m=-inf, l=0,
  // acc=0 so the merge drops it cleanly.
  for (int g = 0; g < G; ++g) {
    const uint base = ((uint)(kv * G + g) * (uint)split + (uint)s) * (uint)D;
    for (int p = 0; p < per; ++p) {
      o_acc[base + (uint)(lane * per + p)] = acc[g][p];
    }
    if (lane == 0) {
      m_out[(uint)(kv * G + g) * split + s] = m[g];
      l_out[(uint)(kv * G + g) * split + s] = l[g];
    }
  }
}

// PHASE B. Merge the SPLIT un-normalized partials per query head into the
// normalized decode output. One thread per (q-head, dim).
//   0:o_acc[Hq,SPLIT,D](f32) 1:m[Hq,SPLIT] 2:l[Hq,SPLIT] 3:out[Hq,D]
//   4:D 5:split 6:Hq.  grid (Hq*D); threadgroup (256).
kernel void sdpa_gqa_merge_f16(
    const device float* o_acc [[buffer(0)]],
    const device float* m_in  [[buffer(1)]],
    const device float* l_in  [[buffer(2)]],
    device VPIPE_ELT*   out   [[buffer(3)]],
    constant int& D     [[buffer(4)]],
    constant int& split [[buffer(5)]],
    constant int& Hq    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(Hq * D)) { return; }
  const int h = (int)gid / D;
  const int d = (int)gid % D;
  const device float* mh = m_in + (uint)h * split;
  const device float* lh = l_in + (uint)h * split;
  float new_max = -INFINITY;
  for (int s = 0; s < split; ++s) { new_max = max(new_max, mh[s]); }
  float new_sum = 0.0f, ov = 0.0f;
  if (new_max > -INFINITY) {
    for (int s = 0; s < split; ++s) {
      const float f = (mh[s] > -INFINITY) ? exp(mh[s] - new_max) : 0.0f;
      new_sum += lh[s] * f;
      ov += o_acc[((uint)h * split + s) * D + d] * f;
    }
  }
  out[(uint)h * D + d] = VPIPE_ELT(new_sum > 0.0f ? ov / new_sum : 0.0f);
}

// sdpa_paged_gqa_vec_f16 -- PAGED sibling of sdpa_causal_gqa_vec_f16, the
// LATENCY-optimal decode form of sdpa_paged_gqa_mb256_f16. The mb256 form
// reads each KV head ONCE for all G query heads in a SINGLE simdgroup
// (bandwidth-optimal), but its serial single-key/all-G inner loop exposes
// per-key DRAM + simd_sum + exp latency on the critical path. Here ONE
// simdgroup per Q-HEAD (G simdgroups/threadgroup) reads its KV head G times
// but unrolls UK independent keys' K loads together (compiler overlaps their
// latency) with vec4 loads, applying the online softmax in STRICT key order
// -> bit-identical to the per-key scan, just better-scheduled. Qwen3.5
// full-attention decode is latency-bound (not bandwidth-bound) even at 4k:
// the micro (gqa_decode_micro qwen(G4)) measures this form ~1.8-2.2x the
// mb256 form. KV-split (grid.z) restores parallelism; writes UN-normalized
// per-(qhead,split) partials that sdpa_gqa_merge_f16 folds. head_dim % 128
// == 0 (per4 = D/128; D=256/128); other shapes stay on mb256.
//   0:q 1:kpool 2:vpool 3:o_acc(f32) 4:m(f32) 5:l(f32) 6:scale 7:D 8:Hq
//   9:Hkv 10:q_offset 11:page_tokens 12:n_pages 13:page_table 14:split
// grid (32, Hq, split); threadgroup (32, G, 1).
kernel void sdpa_paged_gqa_vec_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      D          [[buffer(7)]],
    constant int&      Hq         [[buffer(8)]],
    constant int&      Hkv        [[buffer(9)]],
    constant int&      q_offset   [[buffer(10)]],
    constant int&      page_tokens[[buffer(11)]],
    constant int&      n_pages    [[buffer(12)]],
    const device int*  page_table [[buffer(13)]],
    constant int&      split      [[buffer(14)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int g  = (int)simd_gid;
  const int qhead = kv * G + g;
  const int per = D / 32;
  const int per4 = per / 4;
  const int q_pos = q_offset;
  const int total = q_pos + 1;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;

  // This split's contiguous global-position range [lo, hi) (all <= q_pos).
  const int chunk = (total + split - 1) / split;
  const int lo = s * chunk;
  const int hi = min((s + 1) * chunk, total);

  const device VPIPE_ELT* qh = q + (uint)qhead * D;
  typedef vec<VPIPE_ELT, 4> elt4;
  const uint base_off = (uint)lane * (uint)per;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[base_off + (uint)p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;

  if (lo < hi) {
    for (int pg = 0; pg < n_pages; ++pg) {
      const int pid    = page_table[pg * 3 + 0];
      const int nvalid = page_table[pg * 3 + 1];
      const int gstart = page_table[pg * 3 + 2];
      const int tlo = max(0, lo - gstart);
      const int thi = min(nvalid, hi - gstart);
      if (tlo >= thi) { continue; }
      const device VPIPE_ELT* kbase =
          kpool + (uint)pid * page_stride + head_off + base_off;
      const device VPIPE_ELT* vbase =
          vpool + (uint)pid * page_stride + head_off + base_off;
      for (int t = tlo; t < thi; t += SDPA_GQA_UK) {
        const int ub = min((int)SDPA_GQA_UK, thi - t);
        // Hoist the ub keys' K loads (independent -> latency overlap) + dots.
        float dots[SDPA_GQA_UK];
        for (int u = 0; u < ub; ++u) {
          const device elt4* k4 =
              (const device elt4*)(kbase + (uint)(t + u) * (uint)D);
          float dd = 0.0f;
          for (int c = 0; c < per4; ++c) {
            const elt4 kv4 = k4[c];          // vectorized load
            dd += qreg[c * 4 + 0] * float(kv4.x);   // strict L-to-R order ==
            dd += qreg[c * 4 + 1] * float(kv4.y);   // per-key scan ->
            dd += qreg[c * 4 + 2] * float(kv4.z);   // bit-identical result
            dd += qreg[c * 4 + 3] * float(kv4.w);
          }
          dots[u] = simd_sum(dd);
        }
        // Online softmax in strict key order; V loads vectorized per key.
        for (int u = 0; u < ub; ++u) {
          const float dot = dots[u];
          const float m_new = max(m, dot);
          const float corr = fast::exp(m - m_new);
          const float pj = fast::exp(dot - m_new);
          l = l * corr + pj;
          const device elt4* v4 =
              (const device elt4*)(vbase + (uint)(t + u) * (uint)D);
          for (int c = 0; c < per4; ++c) {
            const elt4 vv4 = v4[c];
            acc[c * 4 + 0] = acc[c * 4 + 0] * corr + pj * float(vv4.x);
            acc[c * 4 + 1] = acc[c * 4 + 1] * corr + pj * float(vv4.y);
            acc[c * 4 + 2] = acc[c * 4 + 2] * corr + pj * float(vv4.z);
            acc[c * 4 + 3] = acc[c * 4 + 3] * corr + pj * float(vv4.w);
          }
          m = m_new;
        }
      }
    }
  }

  // UN-normalized partials per (q-head, split). Empty split -> m=-inf, l=0,
  // acc=0 so the merge drops it cleanly.
  const uint obase = ((uint)qhead * (uint)split + (uint)s) * (uint)D;
  for (int p = 0; p < per; ++p) {
    o_acc[obase + base_off + (uint)p] = acc[p];
  }
  if (lane == 0) {
    m_out[(uint)qhead * split + s] = m;
    l_out[(uint)qhead * split + s] = l;
  }
}

// sdpa_gqa_kbl_impl -- KEY-ACROSS-LANES block decode, templated on DK4
// (head_dim/4) and LD (DK4/NL). A port of llama.cpp's kernel_flash_attn_ext_vec
// NL/NE tiling onto vpipe's paged KV. The vec kernel spreads head_dim across the
// 32 lanes and pays a cross-lane simd_sum PER KEY -- the binding long-context
// decode cost (decode attention is latency-bound on that reduction, NOT on KV
// bytes). Here each lane owns its OWN key: NL=8 lanes cooperate on one key's dot
// via COALESCED sub-row K loads, NE=NW/NL=4 keys run in parallel, and the
// BC=32-key block's scores land one-per-lane in shared memory -- so the whole
// block needs just ONE simd_max + ONE simd_sum (not 32). Q is staged once in
// threadgroup memory; P*V is accumulated per lane then tree-reduced across the
// NE groups. Writes the SAME un-normalized per-(qhead,split) partials
// sdpa_gqa_merge_f16 folds. Greedy token-exact vs the vec kernel. The NL/NE
// reductions are independent of head_dim -- only DK4/LD change -- so the 256
// (kbl) and 128 (kbl128) entry points share this body. The vec layout already
// covers head_dim 128, so kbl128 is the matching key-across-lanes form for the
// dense / head_dim-128 Qwen (ASR decoder, classic dense). grid (32, Hq, split);
// threadgroup (32, G, 1).
template <int DK4, int LD>
static inline void sdpa_gqa_kbl_impl(
    const device VPIPE_ELT* q,
    const device VPIPE_ELT* kpool,
    const device VPIPE_ELT* vpool,
    device float*           o_acc,
    device float*           m_out,
    device float*           l_out,
    float    scale,
    int      D,
    int      Hq,
    int      Hkv,
    int      q_offset,
    int      page_tokens,
    int      n_pages,
    const device int*  page_table,
    int      split,
    threadgroup float* sq_all,
    threadgroup float* ss_all,
    uint3 tid, uint simd_gid, uint lane)
{
  constexpr int NW = 32, NL = 8, NE = 4, BC = 32;   // BC keys/block (= NW)
  constexpr int DV = DK4 * 4;                        // head_dim
  typedef vec<VPIPE_ELT, 4> elt4;

  const int s    = (int)tid.z;
  const int G    = Hq / Hkv;
  const int g    = (int)simd_gid;
  const int kvix = (int)tid.y;                      // 0..Hkv-1
  const int qhead = kvix * G + g;
  const int total = q_offset + 1;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvix * page_tokens * D;

  const int chunk = (total + split - 1) / split;
  const int lo = s * chunk;
  const int hi = min((s + 1) * chunk, total);

  const int tx = (int)lane % NL;                    // 0..7 head-dim sub-chunk
  const int ty = (int)lane / NL;                    // 0..3 key in the NE group

  threadgroup float* sq = sq_all + g * DV;
  threadgroup float* ss = ss_all + g * BC;
  threadgroup const float4* sq4 = (threadgroup const float4*)sq;

  // Stage Q (scaled) into shared once.
  const device VPIPE_ELT* qh = q + (uint)qhead * D;
  for (int i = (int)lane; i < DV; i += NW) {
    sq[i] = float(qh[i]) * scale;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  float M = -INFINITY, S = 0.0f;
  float4 oa[LD];
  for (int ii = 0; ii < LD; ++ii) { oa[ii] = float4(0.0f); }

  if (lo < hi) {
    for (int pg = 0; pg < n_pages; ++pg) {
      const int pid    = page_table[pg * 3 + 0];
      const int nvalid = page_table[pg * 3 + 1];
      const int gstart = page_table[pg * 3 + 2];
      const int tlo = max(0, lo - gstart);
      const int thi = min(nvalid, hi - gstart);
      if (tlo >= thi) { continue; }
      const device elt4* kbase4 =
          (const device elt4*)(kpool + (uint)pid * page_stride + head_off);
      const device elt4* vbase4 =
          (const device elt4*)(vpool + (uint)pid * page_stride + head_off);
      for (int bt = tlo; bt < thi; bt += BC) {
        const int bsz = min((int)BC, thi - bt);
        // ---- Q*K^T: lane (tx,ty) accumulates key (NE*cc+ty) over its NL chunk.
        float mqk[BC / NE];
        for (int cc = 0; cc < BC / NE; ++cc) {
          const int key = bt + NE * cc + ty;
          float dd = 0.0f;
          if (key < thi) {
            for (int ii = 0; ii < LD; ++ii) {
              const int d4 = ii * NL + tx;          // coalesced across tx
              const elt4 k4 = kbase4[(uint)key * DK4 + (uint)d4];
              const float4 qq = sq4[d4];
              dd += float(k4.x) * qq.x + float(k4.y) * qq.y
                  + float(k4.z) * qq.z + float(k4.w) * qq.w;
            }
          }
          // Reduce dd over the NL=8 lanes of this ty-group; tx==0 holds the
          // full dot, then broadcast it across the group.
          dd += simd_shuffle_down(dd, 4);
          dd += simd_shuffle_down(dd, 2);
          dd += simd_shuffle_down(dd, 1);
          dd = simd_shuffle(dd, (uint)(NL * ty));
          mqk[cc] = (key < thi) ? dd : -INFINITY;
        }
        // Transpose lane-local mqk -> shared: ss[NE*tx+ty] = dot(key NE*tx+ty).
        ss[NE * tx + ty] = mqk[tx];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // ---- Block online softmax: one score per lane ----
        const float sc = ss[lane];
        const float M_new = simd_max(max(M, sc));
        const float ms = fast::exp(M - M_new);
        const float vs = ((int)lane < bsz) ? fast::exp(sc - M_new) : 0.0f;
        S = S * ms + simd_sum(vs);
        ss[lane] = vs;
        for (int ii = 0; ii < LD; ++ii) { oa[ii] *= ms; }
        M = M_new;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // ---- O += sum_key p_key * V[key], per lane head-dim chunk ----
        for (int cc = 0; cc < BC / NE; ++cc) {
          const int key = bt + NE * cc + ty;
          if (key < thi) {
            const float p = ss[NE * cc + ty];
            for (int ii = 0; ii < LD; ++ii) {
              const int d4 = ii * NL + tx;
              const elt4 vv = vbase4[(uint)key * DK4 + (uint)d4];
              oa[ii] += float4(float(vv.x), float(vv.y),
                               float(vv.z), float(vv.w)) * p;
            }
          }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }

  // Reduce O over the NE=4 ty-groups (lanes tx, tx+8, tx+16, tx+24 -> tx).
  for (int ii = 0; ii < LD; ++ii) {
    oa[ii].x += simd_shuffle_down(oa[ii].x, 8);
    oa[ii].y += simd_shuffle_down(oa[ii].y, 8);
    oa[ii].z += simd_shuffle_down(oa[ii].z, 8);
    oa[ii].w += simd_shuffle_down(oa[ii].w, 8);
    oa[ii].x += simd_shuffle_down(oa[ii].x, 16);
    oa[ii].y += simd_shuffle_down(oa[ii].y, 16);
    oa[ii].z += simd_shuffle_down(oa[ii].z, 16);
    oa[ii].w += simd_shuffle_down(oa[ii].w, 16);
  }
  // Lanes with ty==0 (0..7) now hold the full O. Write float4 d4 = ii*NL+tx.
  if (ty == 0) {
    const uint obase4 = ((uint)qhead * (uint)split + (uint)s) * (uint)DK4;
    device float4* o4 = (device float4*)o_acc;
    for (int ii = 0; ii < LD; ++ii) {
      o4[obase4 + (uint)(ii * NL + tx)] = oa[ii];
    }
  }
  if (lane == 0) {
    m_out[(uint)qhead * split + s] = M;
    l_out[(uint)qhead * split + s] = S;
  }
}

// Entry points: head_dim 256 (DK4=64, LD=8) and head_dim 128 (DK4=32, LD=4).
#define VPIPE_SDPA_KBL(NAME, DK4V, LDV, DVV)                                   \
kernel void NAME(                                                             \
    const device VPIPE_ELT* q          [[buffer(0)]],                         \
    const device VPIPE_ELT* kpool      [[buffer(1)]],                         \
    const device VPIPE_ELT* vpool      [[buffer(2)]],                         \
    device float*           o_acc      [[buffer(3)]],                         \
    device float*           m_out      [[buffer(4)]],                         \
    device float*           l_out      [[buffer(5)]],                         \
    constant float&    scale      [[buffer(6)]],                              \
    constant int&      D          [[buffer(7)]],                              \
    constant int&      Hq         [[buffer(8)]],                              \
    constant int&      Hkv        [[buffer(9)]],                              \
    constant int&      q_offset   [[buffer(10)]],                             \
    constant int&      page_tokens[[buffer(11)]],                             \
    constant int&      n_pages    [[buffer(12)]],                             \
    const device int*  page_table [[buffer(13)]],                             \
    constant int&      split      [[buffer(14)]],                             \
    uint3 tid       [[threadgroup_position_in_grid]],                         \
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],                       \
    uint  lane      [[thread_index_in_simdgroup]])                            \
{                                                                            \
  threadgroup float sq_all[SDPA_GQA_MAXG * (DVV)];                            \
  threadgroup float ss_all[SDPA_GQA_MAXG * 32];                               \
  sdpa_gqa_kbl_impl<DK4V, LDV>(q, kpool, vpool, o_acc, m_out, l_out, scale,    \
      D, Hq, Hkv, q_offset, page_tokens, n_pages, page_table, split,          \
      sq_all, ss_all, tid, simd_gid, lane);                                   \
}
VPIPE_SDPA_KBL(sdpa_paged_gqa_kbl_f16,    64, 8, 256)
VPIPE_SDPA_KBL(sdpa_paged_gqa_kbl128_f16, 32, 4, 128)

// sdpa_paged_gqa_vec1_impl -- PAGED-KV port of MLX's sdpa_vector decode kernel.
// ONE threadgroup per QUERY head: BN=32 simdgroups STRIPE the keys, BD=32 lanes
// split the head dim (a single key's QK dot is fully 32-way parallel), then an
// intra-threadgroup transpose+flash-merge over the BN stripes writes attn[Hq,D]
// directly (no split across threadgroups, no merge kernel). PER (= D/BD) is a
// COMPILE-TIME template arg so the per-lane qk/o loops UNROLL and stay in
// registers (a runtime PER spills them to local memory -> ~8x slower). Decode
// only (M=1): query at q_offset attends [0, q_offset].
// grid (1024, Hq, 1); threadgroup (1024,1,1) = BN*BD simdgroups.
template <int PER>
METAL_FUNC void sdpa_paged_gqa_vec1_impl(
    const device VPIPE_ELT* q, const device VPIPE_ELT* kpool,
    const device VPIPE_ELT* vpool, device float* o_acc, device float* m_out,
    device float* l_out, const constant float& scale, const constant int& D,
    const constant int& Hq, const constant int& Hkv,
    const constant int& q_offset, const constant int& page_tokens,
    const constant int& n_pages, const device int* page_table,
    const constant int& split,
    threadgroup float* outputs, threadgroup float* max_scores,
    threadgroup float* sum_scores,
    uint3 tid, uint simd_gid, uint simd_lid)
{
  constexpr int BN = 32, BD = 32;
  const int qh  = (int)tid.y;
  const int sp  = (int)tid.z;
  const int kvh = qh / (Hq / Hkv);
  const int N   = q_offset + 1;            // keys 0..q_offset
  (void)n_pages;                           // walked via page_table; kept for ABI
  const int blk = (N + split - 1) / split; // keys per split
  const int klo = sp * blk;
  const int khi = min(N, klo + blk);
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  thread float qv[PER], ov[PER];
  const device VPIPE_ELT* qp = q + (uint)qh * D + (uint)simd_lid * PER;
  for (int j = 0; j < PER; ++j) { qv[j] = (float)qp[j]; ov[j] = 0.0f; }
  float m = -INFINITY, l = 0.0f;

  // Each simdgroup strides this split's keys by BN; lanes hold the head dim.
  for (int i = klo + (int)simd_gid; i < khi; i += BN) {
    const int pg    = i / page_tokens;
    const int pid   = page_table[pg * 3 + 0];
    const int local = i - pg * page_tokens;
    const uint koff = (uint)pid * page_stride + head_off + (uint)local * D
                      + (uint)simd_lid * PER;
    const device VPIPE_ELT* kp = kpool + koff;
    const device VPIPE_ELT* vp = vpool + koff;
    float s = 0.0f;
    for (int j = 0; j < PER; ++j) { s += qv[j] * (float)kp[j]; }
    s = simd_sum(s) * scale;
    const float mnew = max(m, s);
    const float f = fast::exp(m - mnew);
    const float e = fast::exp(s - mnew);
    m = mnew;
    l = l * f + e;
    for (int j = 0; j < PER; ++j) { ov[j] = ov[j] * f + e * (float)vp[j]; }
  }

  // Intra-threadgroup flash merge across the BN key-stripes -> this split's
  // UN-normalized partial (O + m + l) in the kbl/sdpa_gqa_merge contract.
  if (simd_lid == 0) { max_scores[simd_gid] = m; sum_scores[simd_gid] = l; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float lm = max_scores[simd_lid];
  const float gm = simd_max(lm);
  const float mf = fast::exp(lm - gm);
  const float gl = simd_sum(sum_scores[simd_lid] * mf);
  for (int j = 0; j < PER; ++j) {
    outputs[simd_lid * BD + simd_gid] = ov[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ov[j] = simd_sum(outputs[simd_gid * BD + simd_lid] * mf);  // un-normalized
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint pbase = (uint)qh * (uint)split + (uint)sp;
  if (simd_lid == 0) {
    device float* op = o_acc + pbase * (uint)D + (uint)simd_gid * PER;
    for (int j = 0; j < PER; ++j) { op[j] = ov[j]; }
  }
  if (simd_gid == 0 && simd_lid == 0) { m_out[pbase] = gm; l_out[pbase] = gl; }
}
kernel void sdpa_paged_gqa_vec1_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float& scale       [[buffer(6)]],
    constant int&   D           [[buffer(7)]],
    constant int&   Hq          [[buffer(8)]],
    constant int&   Hkv         [[buffer(9)]],
    constant int&   q_offset    [[buffer(10)]],
    constant int&   page_tokens [[buffer(11)]],
    constant int&   n_pages     [[buffer(12)]],
    const device int* page_table[[buffer(13)]],
    constant int&   split       [[buffer(14)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  threadgroup float outputs[32 * 32];
  threadgroup float max_scores[32];
  threadgroup float sum_scores[32];
  sdpa_paged_gqa_vec1_impl<8>(q, kpool, vpool, o_acc, m_out, l_out, scale, D,
      Hq, Hkv, q_offset, page_tokens, n_pages, page_table, split,
      outputs, max_scores, sum_scores, tid, simd_gid, simd_lid);
}

// sdpa_paged_gqa_vec2_impl -- the FAITHFUL MLX sdpa_vector_2pass_1 port (vs
// vec1's intra-tg-merge variant). Per (kvhead, block) threadgroup, G simdgroups
// (one per QUERY head, NO intra-tg merge), 32 lanes split the head dim. Block b
// scans keys b, b+blocks, ... (strided), q PRE-scaled, writes its per-qhead
// per-block partial [Hq, blocks, D] for the COOPERATIVE merge sdpa_gqa_merge2.
// PER (= D/32) compile-time (unroll into registers). grid (32, Hq, blocks);
// threadgroup (32, G, 1).
template <int PER>
METAL_FUNC void sdpa_paged_gqa_vec2_impl(
    const device VPIPE_ELT* q, const device VPIPE_ELT* kpool,
    const device VPIPE_ELT* vpool, device float* o_acc, device float* m_out,
    device float* l_out, const constant float& scale, const constant int& D,
    const constant int& Hq, const constant int& Hkv,
    const constant int& q_offset, const constant int& page_tokens,
    const constant int& n_pages, const device int* page_table,
    const constant int& blocks,
    uint3 tid, uint simd_gid, uint simd_lid)
{
  const int kvh   = (int)tid.y;            // threadgroups in y == Hkv
  const int block = (int)tid.z;
  const int G     = Hq / Hkv;
  const int qh    = kvh * G + (int)simd_gid;
  const int N     = q_offset + 1;          // keys 0..q_offset
  (void)n_pages;                           // walked via page_table; kept for ABI
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;

  thread float qv[PER], ov[PER];
  const device VPIPE_ELT* qp = q + (uint)qh * D + (uint)simd_lid * PER;
  for (int j = 0; j < PER; ++j) { qv[j] = scale * (float)qp[j]; ov[j] = 0.0f; }
  float m = -INFINITY, l = 0.0f;

  for (int i = block; i < N; i += blocks) {
    const int pg    = i / page_tokens;
    const int pid   = page_table[pg * 3 + 0];
    const int local = i - pg * page_tokens;
    const uint koff = (uint)pid * page_stride + head_off + (uint)local * D
                      + (uint)simd_lid * PER;
    const device VPIPE_ELT* kp = kpool + koff;
    const device VPIPE_ELT* vp = vpool + koff;
    float s = 0.0f;
    for (int j = 0; j < PER; ++j) { s += qv[j] * (float)kp[j]; }
    s = simd_sum(s);
    const float mnew = max(m, s);
    const float f = fast::exp(m - mnew);
    const float e = fast::exp(s - mnew);
    m = mnew;
    l = l * f + e;
    for (int j = 0; j < PER; ++j) { ov[j] = ov[j] * f + e * (float)vp[j]; }
  }

  const uint pbase = (uint)qh * (uint)blocks + (uint)block;
  device float* op = o_acc + pbase * (uint)D + (uint)simd_lid * PER;
  for (int j = 0; j < PER; ++j) { op[j] = ov[j]; }
  if (simd_lid == 0) { m_out[pbase] = m; l_out[pbase] = l; }
}
kernel void sdpa_paged_gqa_vec2_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float& scale       [[buffer(6)]],
    constant int&   D           [[buffer(7)]],
    constant int&   Hq          [[buffer(8)]],
    constant int&   Hkv         [[buffer(9)]],
    constant int&   q_offset    [[buffer(10)]],
    constant int&   page_tokens [[buffer(11)]],
    constant int&   n_pages     [[buffer(12)]],
    const device int* page_table[[buffer(13)]],
    constant int&   blocks      [[buffer(14)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  sdpa_paged_gqa_vec2_impl<8>(q, kpool, vpool, o_acc, m_out, l_out, scale, D,
      Hq, Hkv, q_offset, page_tokens, n_pages, page_table, blocks,
      tid, simd_gid, simd_lid);
}

// sdpa_gqa_merge2_f16 -- COOPERATIVE merge (MLX sdpa_vector_2pass_2 port) for the
// vec2 partials [Hq, blocks, D]. ONE threadgroup per query head, BN=32 simdgroups
// each owning a block-stripe (block b, b+32, ...). maxs/sums read LANE-strided
// (coalesced) + reduced via simd_max/simd_sum; partials read [block][head-dim]
// with 32 lanes contiguous over D (coalesced), then a transpose+reduce over the
// BN simdgroups. Far cheaper than sdpa_gqa_merge's one-thread-per-d stride-D
// (non-coalesced) loop at high `blocks`. Requires blocks % 32 == 0.
// grid (1024, Hq, 1); threadgroup (1024,1,1) = BN*BD simdgroups.
template <int PER>
METAL_FUNC void sdpa_gqa_merge2_impl(
    const device float* partials, const device float* sums,
    const device float* maxs, device VPIPE_ELT* out,
    const constant int& D, const constant int& blocks,
    threadgroup float* outputs, uint3 tid, uint simd_gid, uint simd_lid)
{
  constexpr int BN = 32, BD = 32;
  const int qh = (int)tid.y;
  const device float* pp = partials + (uint)qh * blocks * D
                           + (uint)simd_gid * D + (uint)simd_lid * PER;
  const device float* sp = sums + (uint)qh * blocks;
  const device float* mp = maxs + (uint)qh * blocks;

  thread float o[PER];
  for (int j = 0; j < PER; ++j) { o[j] = 0.0f; }
  float sum_exp = 0.0f, max_score = -INFINITY;

  const int nb = blocks / BN;
  for (int b = 0; b < nb; ++b) {
    max_score = max(max_score, mp[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);
  for (int b = 0; b < nb; ++b) {
    const float f = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp += f * sp[simd_lid + BN * b];
  }
  sum_exp = simd_sum(sum_exp);

  for (int b = 0; b < nb; ++b) {
    const float f = fast::exp(mp[simd_gid] - max_score);
    for (int j = 0; j < PER; ++j) { o[j] += f * pp[j]; }
    mp += BN; sp += BN; pp += (uint)BN * D;
  }

  // Transpose+reduce over the BN simdgroups (MLX sdpa_vector_2pass_2 tail).
  for (int j = 0; j < PER; ++j) {
    outputs[simd_lid * BD + simd_gid] = o[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float oj = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[j] = (sum_exp == 0.0f) ? oj : (oj / sum_exp);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    device VPIPE_ELT* op = out + (uint)qh * D + (uint)simd_gid * PER;
    for (int j = 0; j < PER; ++j) { op[j] = (VPIPE_ELT)o[j]; }
  }
}
kernel void sdpa_gqa_merge2_f16(
    const device float* partials [[buffer(0)]],
    const device float* sums     [[buffer(1)]],
    const device float* maxs     [[buffer(2)]],
    device VPIPE_ELT*   out       [[buffer(3)]],
    constant int& D      [[buffer(4)]],
    constant int& blocks [[buffer(5)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
  threadgroup float outputs[32 * 32];
  sdpa_gqa_merge2_impl<8>(partials, sums, maxs, out, D, blocks,
      outputs, tid, simd_gid, simd_lid);
}

// ============================================================================
// MATERIALIZED paged DECODE for the Gemma global (head_dim 512) layers.
//
// What omlx/MLX does for head_dim 512 (decode, M=1): head_dim 512 is in
// NEITHER MLX's sdpa_vector {64,96,128,256} nor sdpa_full {64,80,128} support
// set, so mx.fast.scaled_dot_product_attention FALLS BACK to the unfused form
// (mlx/fast.cpp: "O = softmax(Q @ K.T) @ V") -- a materialized GEMV: scores =
// Q.K^T, a single parallel softmax over the whole row, then scores.V. This
// has NO online-softmax serial recurrence -- the m/l carry + per-key acc
// rescale that bottlenecks our fused flash decode at depth (the attn_global
// latency chain). We replicate it in 4 paged passes (the decode analogue of
// the materialized PREFILL path):
//   K1 sdpa_paged_dec_qk    : scores[h,t] = scale * (Q_h . K_t)   (GEMV)
//   K2 sdpa_dec_rowstat     : per head, m[h]=max_t, l[h]=sum_t exp(s-m)
//   K3 sdpa_paged_dec_pv    : partial[h,sp,d] = sum_t exp(s-m[h]) * V_t,d
//   K4 sdpa_dec_pv_merge    : out[h,d] = (sum_sp partial) / l[h]
// K1/K3 split the key range over grid.z (sp) for occupancy; scores live in a
// [Hq, Tstride] f32 scratch. K2 is per-head (Hq threadgroups, tiny). Greedy
// token-exact with the flash path (same softmax, fp reassociation only).
// head_dim % 128 == 0 (per4 = D/128; D=512). Paged-pool contract matches
// sdpa_paged_gqa_vec_f16.

// K1: QK GEMV. grid (32, Hq, sp); tg (32,1,1). Each simdgroup owns one
// (qhead, key-split): page-walk its global-position range [lo,hi), dot Q_h
// with each K_t (32 lanes x per4 vec4), write scale*dot to scores[h*Tstride+t].
//   0:q 1:kpool 2:scores(f32) 3:scale 4:D 5:Hq 6:Hkv 7:q_offset 8:page_tokens
//   9:n_pages 10:page_table 11:Tstride 12:split
#define SDPA_DEC_UK 8     // decode GEMV key-unroll (latency overlap)
kernel void sdpa_paged_dec_qk_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    device VPIPE_ELT*       scores     [[buffer(2)]],   // f16 scratch
    constant float&    scale      [[buffer(3)]],
    constant int&      D          [[buffer(4)]],
    constant int&      Hq         [[buffer(5)]],
    constant int&      Hkv        [[buffer(6)]],
    constant int&      q_offset   [[buffer(7)]],
    constant int&      page_tokens[[buffer(8)]],
    constant int&      n_pages    [[buffer(9)]],
    const device int*  page_table [[buffer(10)]],
    constant int&      Tstride    [[buffer(11)]],
    constant int&      split      [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int qhead = (int)tid.y;
  const int s     = (int)tid.z;
  const int G     = Hq / Hkv;
  const int kv    = qhead / G;
  const int per   = D / 32;
  const int per4  = per / 4;
  const int total = q_offset + 1;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;
  const uint base_off = (uint)lane * (uint)per;

  const int chunk = (total + split - 1) / split;
  const int lo = s * chunk;
  const int hi = min((s + 1) * chunk, total);
  if (lo >= hi) { return; }

  const device VPIPE_ELT* qh = q + (uint)qhead * D;
  typedef vec<VPIPE_ELT, 4> elt4;
  float qreg[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) { qreg[p] = float(qh[base_off + (uint)p]); }

  device VPIPE_ELT* srow = scores + (uint)qhead * (uint)Tstride;
  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    const int tlo = max(0, lo - gstart);
    const int thi = min(nvalid, hi - gstart);
    if (tlo >= thi) { continue; }
    const device VPIPE_ELT* kbase =
        kpool + (uint)pid * page_stride + head_off + base_off;
    // UK-unroll: issue UK independent K-row dot-products together so the
    // compiler overlaps their DRAM latency (memory-level parallelism), then
    // reduce + write. Keys within a page are contiguous, so the unroll never
    // straddles a page boundary.
    for (int t = tlo; t < thi; t += SDPA_DEC_UK) {
      const int ub = min((int)SDPA_DEC_UK, thi - t);
      float dd[SDPA_DEC_UK];
      for (int u = 0; u < ub; ++u) {
        const device elt4* k4 =
            (const device elt4*)(kbase + (uint)(t + u) * (uint)D);
        float a = 0.0f;
        for (int c = 0; c < per4; ++c) {
          const elt4 kv4 = k4[c];
          a += qreg[c * 4 + 0] * float(kv4.x);
          a += qreg[c * 4 + 1] * float(kv4.y);
          a += qreg[c * 4 + 2] * float(kv4.z);
          a += qreg[c * 4 + 3] * float(kv4.w);
        }
        dd[u] = a;
      }
      for (int u = 0; u < ub; ++u) {
        const float r = simd_sum(dd[u]);
        if (lane == 0) { srow[gstart + t + u] = VPIPE_ELT(r * scale); }
      }
    }
  }
}

// K2: per-head softmax stats AND weight write-back. grid (256, Hq, 1);
// tg (256,1,1). Pass 1: m = max_t scores. Pass 2: OVERWRITE scores[t] with the
// un-normalized weight exp(scores[t]-m) and accumulate l = sum_t weight. K3
// then reads the weights directly (no exp on the V-read hot loop) and K4
// divides the merged output by l. scores is read-WRITE here.
//   0:scores(f16, rw) 1:m_out(f32) 2:l_out(f32) 3:T_kv 4:Tstride 5:Hq
kernel void sdpa_dec_rowstat_f16(
    device VPIPE_ELT* scores  [[buffer(0)]],   // f16 scratch (rw)
    device float*     m_out   [[buffer(1)]],
    device float*     l_out   [[buffer(2)]],
    constant int&     T_kv    [[buffer(3)]],
    constant int&     Tstride [[buffer(4)]],
    constant int&     Hq      [[buffer(5)]],
    uint3 tid   [[threadgroup_position_in_grid]],
    uint3 ltid3 [[thread_position_in_threadgroup]],
    uint3 ntg3  [[threads_per_threadgroup]])
{
  (void)Hq;                  // unused: head = tid.y; kept for the host buffer ABI
  const int h = (int)tid.y;
  const uint ltid = ltid3.x;
  const uint ntg = ntg3.x;
  device VPIPE_ELT* srow = scores + (uint)h * (uint)Tstride;
  threadgroup float red[256];

  float lm = -INFINITY;
  for (uint t = ltid; t < (uint)T_kv; t += ntg) {
    lm = max(lm, float(srow[t]));
  }
  red[ltid] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint w = ntg / 2; w > 0; w >>= 1) {
    if (ltid < w) { red[ltid] = max(red[ltid], red[ltid + w]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float m = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float ls = 0.0f;
  for (uint t = ltid; t < (uint)T_kv; t += ntg) {
    const float e = fast::exp(float(srow[t]) - m);
    srow[t] = VPIPE_ELT(e);       // overwrite raw score with its weight
    ls += e;
  }
  red[ltid] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint w = ntg / 2; w > 0; w >>= 1) {
    if (ltid < w) { red[ltid] += red[ltid + w]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (ltid == 0) { m_out[h] = m; l_out[h] = red[0]; }
}

// K3: PV GEMV. grid (32, Hq, sp); tg (32,1,1). Each (qhead, key-split)
// page-walks [lo,hi) accumulating partial[h,sp,:] += weight[t] * V_t, where
// weight[t] = K2's pre-computed exp(scores[t]-m) -- PURE FMA, no transcendental
// on this (V-bandwidth-heavy) hot loop. UK-unroll prefetches the weights and
// overlaps the independent V-row loads (memory-level parallelism). K4 divides
// the merged output by the row sum l.  0:weights(f32) 1:vpool 2:partial(f32)
//   3:D 4:Hq 5:Hkv 6:q_offset 7:page_tokens 8:n_pages 9:page_table 10:Tstride
//   11:split
kernel void sdpa_paged_dec_pv_f16(
    const device VPIPE_ELT* weights    [[buffer(0)]],   // f16 scratch
    const device VPIPE_ELT* vpool      [[buffer(1)]],
    device float*           partial    [[buffer(2)]],
    constant int&      D          [[buffer(3)]],
    constant int&      Hq         [[buffer(4)]],
    constant int&      Hkv        [[buffer(5)]],
    constant int&      q_offset   [[buffer(6)]],
    constant int&      page_tokens[[buffer(7)]],
    constant int&      n_pages    [[buffer(8)]],
    const device int*  page_table [[buffer(9)]],
    constant int&      Tstride    [[buffer(10)]],
    constant int&      split      [[buffer(11)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int qhead = (int)tid.y;
  const int s     = (int)tid.z;
  const int G     = Hq / Hkv;
  const int kv    = qhead / G;
  const int per   = D / 32;
  const int per4  = per / 4;
  const int total = q_offset + 1;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;
  const uint base_off = (uint)lane * (uint)per;

  const int chunk = (total + split - 1) / split;
  const int lo = s * chunk;
  const int hi = min((s + 1) * chunk, total);

  typedef vec<VPIPE_ELT, 4> elt4;
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) { acc[p] = 0.0f; }
  const device VPIPE_ELT* wrow = weights + (uint)qhead * (uint)Tstride;

  if (lo < hi) {
    for (int pg = 0; pg < n_pages; ++pg) {
      const int pid    = page_table[pg * 3 + 0];
      const int nvalid = page_table[pg * 3 + 1];
      const int gstart = page_table[pg * 3 + 2];
      const int tlo = max(0, lo - gstart);
      const int thi = min(nvalid, hi - gstart);
      if (tlo >= thi) { continue; }
      const device VPIPE_ELT* vbase =
          vpool + (uint)pid * page_stride + head_off + base_off;
      for (int t = tlo; t < thi; t += SDPA_DEC_UK) {
        const int ub = min((int)SDPA_DEC_UK, thi - t);
        float w[SDPA_DEC_UK];
        for (int u = 0; u < ub; ++u) { w[u] = float(wrow[gstart + t + u]); }
        for (int u = 0; u < ub; ++u) {
          const device elt4* v4 =
              (const device elt4*)(vbase + (uint)(t + u) * (uint)D);
          const float wu = w[u];
          for (int c = 0; c < per4; ++c) {
            const elt4 vv4 = v4[c];
            acc[c * 4 + 0] += wu * float(vv4.x);
            acc[c * 4 + 1] += wu * float(vv4.y);
            acc[c * 4 + 2] += wu * float(vv4.z);
            acc[c * 4 + 3] += wu * float(vv4.w);
          }
        }
      }
    }
  }
  const uint obase = ((uint)qhead * (uint)split + (uint)s) * (uint)D;
  for (int p = 0; p < per; ++p) {
    partial[obase + base_off + (uint)p] = acc[p];
  }
}

// K4: sum the sp partials and divide by the row sum. grid (Hq*D,1,1);
// tg (256,1,1). out[h,d] = (sum_sp partial[h,sp,d]) / l[h].
//   0:partial(f32) 1:l_in(f32) 2:out 3:D 4:split 5:Hq
kernel void sdpa_dec_pv_merge_f16(
    const device float* partial [[buffer(0)]],
    const device float* l_in    [[buffer(1)]],
    device VPIPE_ELT*   out      [[buffer(2)]],
    constant int& D     [[buffer(3)]],
    constant int& split [[buffer(4)]],
    constant int& Hq    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)(Hq * D)) { return; }
  const int h = (int)gid / D;
  const int d = (int)gid % D;
  float ov = 0.0f;
  for (int s = 0; s < split; ++s) {
    ov += partial[((uint)h * (uint)split + (uint)s) * (uint)D + (uint)d];
  }
  const float l = l_in[h];
  out[(uint)h * (uint)D + (uint)d] = VPIPE_ELT(l > 0.0f ? ov / l : 0.0f);
}

// sdpa_causal_gqa_f16 -- flash-decode-GQA over a CONTIGUOUS [Hkv, kv_stride,
// D] KV cache (Gemma-4 decode), the contiguous analogue of
// sdpa_paged_gqa_mb256_f16. Same idea: each (kv, split) single-simdgroup
// worker reads its KV-head key range [first,last] slice ONCE and applies
// every key/value to all G=Hq/Hkv query heads, writing UN-normalized
// per-(qhead,split) partials that sdpa_gqa_merge_f16 then folds. Adds the
// contiguous-cache features the paged kernel lacks: a trailing sliding
// `window` (window<=0 == full causal) and a `ring_cap` modulo (sliding-layer
// ring buffer; 0 == linear). head_dim <= 256 (per <= 8); D=512 Gemma layers
// stay on sdpa_causal_mb. Online softmax in fp32, bit-compatible split of the
// same scan sdpa_causal_mb_f16 does.
//   0:q[Hq,1,D] 1:k 2:v[Hkv,kv_stride,D] 3:o_acc(f32) 4:m(f32) 5:l(f32)
//   6:scale 7:T_kv 8:D 9:Hq 10:Hkv 11:q_offset 12:kv_stride 13:window
//   14:ring_cap 15:split.  grid (32, Hkv, split); threadgroup (32,1,1).
kernel void sdpa_causal_gqa_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* k          [[buffer(1)]],
    const device VPIPE_ELT* v          [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      T_kv       [[buffer(7)]],
    constant int&      D          [[buffer(8)]],
    constant int&      Hq         [[buffer(9)]],
    constant int&      Hkv        [[buffer(10)]],
    constant int&      q_offset   [[buffer(11)]],
    constant int&      kv_stride  [[buffer(12)]],
    constant int&      window     [[buffer(13)]],
    constant int&      ring_cap   [[buffer(14)]],
    constant int&      split      [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int per = D / 32;
  const int q_pos = q_offset;
  const int last = min(T_kv - 1, q_pos);
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;

  // This split's contiguous sub-range [lo, hi) of [first, last].
  const int total = last - first + 1;
  const int chunk = (total + split - 1) / split;
  const int lo = first + s * chunk;
  const int hi = min(first + (s + 1) * chunk, last + 1);

  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;
  float qreg[SDPA_GQA_MAXG][SDPA_GQA_D256_PER];
  float acc[SDPA_GQA_MAXG][SDPA_GQA_D256_PER];
  float m[SDPA_GQA_MAXG], l[SDPA_GQA_MAXG];
  for (int g = 0; g < G; ++g) {
    const device VPIPE_ELT* qh = q + (uint)(kv * G + g) * D;
    for (int p = 0; p < per; ++p) {
      qreg[g][p] = float(qh[lane * per + p]) * scale;
      acc[g][p] = 0.0f;
    }
    m[g] = -INFINITY; l[g] = 0.0f;
  }

  for (int t = lo; t < hi; ++t) {
    const uint slot = (ring_cap > 0) ? (uint)(t % ring_cap) : (uint)t;
    float kreg[SDPA_GQA_D256_PER], vreg[SDPA_GQA_D256_PER];
    for (int p = 0; p < per; ++p) {
      kreg[p] = float(kkv[slot * D + lane * per + p]);
      vreg[p] = float(vkv[slot * D + lane * per + p]);
    }
    for (int g = 0; g < G; ++g) {
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) { dot += qreg[g][p] * kreg[p]; }
      dot = simd_sum(dot);
      const float m_new = max(m[g], dot);
      const float corr = exp(m[g] - m_new);
      const float pj = exp(dot - m_new);
      l[g] = l[g] * corr + pj;
      for (int p = 0; p < per; ++p) {
        acc[g][p] = acc[g][p] * corr + pj * vreg[p];
      }
      m[g] = m_new;
    }
  }

  for (int g = 0; g < G; ++g) {
    const uint base = ((uint)(kv * G + g) * (uint)split + (uint)s) * (uint)D;
    for (int p = 0; p < per; ++p) {
      o_acc[base + (uint)(lane * per + p)] = acc[g][p];
    }
    if (lane == 0) {
      m_out[(uint)(kv * G + g) * split + s] = m[g];
      l_out[(uint)(kv * G + g) * split + s] = l[g];
    }
  }
}

// sdpa_causal_gqa_tile_f16 -- threadgroup-STAGED flash-decode for the Gemma-4
// GLOBAL layers (head_dim 512, full-context), the decode bottleneck. The
// scalar/register kernels above are bound by per-key DRAM LATENCY (each
// simdgroup loads K[t], stalls, dots, stalls on V[t]) and the per-q-head 4x KV
// re-read (sdpa_mb) -- ~50 GB/s, far under peak. Here ONE threadgroup per
// (kv_head, split-chunk) stages SDPA_TILE_BK keys' K AND V into threadgroup
// memory cooperatively (all G*32 threads issue coalesced loads -> high
// memory-level parallelism, hiding latency), then the G=Hq/Hkv simdgroups
// (one per q-head of the group) each fold the staged block into their own
// online softmax -- so the KV block is read from DRAM ONCE and reused by all G
// heads (vs 4x in sdpa_mb). Registers stay tiny (qreg/acc only; KV is in
// threadgroup memory) -> high occupancy. Writes un-normalized per-(qhead,s)
// partials that sdpa_gqa_merge_f16 folds. head_dim <= SDPA_TILE_DMAX(512).
//   0:q[Hq,1,D] 1:k 2:v[Hkv,kv_stride,D] 3:o_acc(f32) 4:m(f32) 5:l(f32)
//   6:scale 7:T_kv 8:D 9:Hq 10:Hkv 11:q_offset 12:kv_stride 13:window
//   14:ring_cap 15:split.  grid (32, Hkv, split); threadgroup (32, G, 1).
#define SDPA_TILE_BK   8     // keys staged per block
#define SDPA_TILE_DMAX 512   // max head_dim (tg-array sizing)
kernel void sdpa_causal_gqa_tile_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* k          [[buffer(1)]],
    const device VPIPE_ELT* v          [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      T_kv       [[buffer(7)]],
    constant int&      D          [[buffer(8)]],
    constant int&      Hq         [[buffer(9)]],
    constant int&      Hkv        [[buffer(10)]],
    constant int&      q_offset   [[buffer(11)]],
    constant int&      kv_stride  [[buffer(12)]],
    constant int&      window     [[buffer(13)]],
    constant int&      ring_cap   [[buffer(14)]],
    constant int&      split      [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int g  = (int)simd_gid;             // this simdgroup's q-head in group
  const int qhead = kv * G + g;
  const int per = D / 32;
  const int q_pos = q_offset;
  const int last = min(T_kv - 1, q_pos);
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;

  // This split's contiguous sub-range [lo, hi) of [first, last].
  const int total = last - first + 1;
  const int chunk = (total > 0) ? (total + split - 1) / split : 0;
  const int lo = first + s * chunk;
  const int hi = min(first + (s + 1) * chunk, last + 1);

  threadgroup VPIPE_ELT Ks[SDPA_TILE_BK * SDPA_TILE_DMAX];   // 8KB @ D512
  threadgroup VPIPE_ELT Vs[SDPA_TILE_BK * SDPA_TILE_DMAX];   // 8KB -> 16KB

  const device VPIPE_ELT* qh  = q + (uint)qhead * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;

  const uint ntg = (uint)G * 32;            // threads in the threadgroup
  const uint ltid = (uint)g * 32 + lane;    // this thread's tg-flat index
  // Block loop bound is threadgroup-uniform (lo/hi depend only on kv,s), so
  // the barriers below are matched across all G simdgroups -- no deadlock.
  for (int blk = lo; blk < hi; blk += SDPA_TILE_BK) {
    const int bn = min(SDPA_TILE_BK, hi - blk);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // prev block consumed
    // Cooperative coalesced load of bn keys' K and V into threadgroup memory
    // (read from DRAM ONCE, reused by all G simdgroups).
    for (uint i = ltid; i < (uint)(bn * D); i += ntg) {
      const int kk = (int)i / D;
      const int dd = (int)i % D;
      const int t = blk + kk;
      const uint slot = (ring_cap > 0) ? (uint)(t % ring_cap) : (uint)t;
      Ks[kk * D + dd] = kkv[slot * D + dd];
      Vs[kk * D + dd] = vkv[slot * D + dd];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // load visible
    // Each simdgroup folds the staged block into its q-head's online softmax.
    for (int kk = 0; kk < bn; ++kk) {
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) {
        dot += qreg[p] * float(Ks[kk * D + lane * per + p]);
      }
      dot = simd_sum(dot);
      const float m_new = max(m, dot);
      const float corr = exp(m - m_new);
      const float pj = exp(dot - m_new);
      l = l * corr + pj;
      for (int p = 0; p < per; ++p) {
        acc[p] = acc[p] * corr + pj * float(Vs[kk * D + lane * per + p]);
      }
      m = m_new;
    }
  }
  const uint base = ((uint)qhead * (uint)split + (uint)s) * (uint)D;
  for (int p = 0; p < per; ++p) {
    o_acc[base + (uint)(lane * per + p)] = acc[p];
  }
  if (lane == 0) {
    m_out[(uint)qhead * split + s] = m;
    l_out[(uint)qhead * split + s] = l;
  }
}

// sdpa_causal_gqa_direct_f16 -- DIRECT-READ flash-decode for the D=512
// global layers. Drops the threadgroup staging + 2 barriers/block of the
// _tile kernels entirely and reads K/V straight from device memory, exactly
// like MLX sdpa_vector_2pass_1: the G simdgroups of one kv-head re-read its
// keys via the L2 cache, and per-key DRAM latency is hidden by hardware ILP
// across loop iterations (the compiler keeps several K/V loads in flight).
// fast::exp. Registers tiny (qreg/acc only) -> high occupancy. The decode
// gap measurement is whether staging's read-once beats direct+cache here.
// Same buffer/grid contract as sdpa_causal_gqa_tile_f16.
kernel void sdpa_causal_gqa_direct_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* k          [[buffer(1)]],
    const device VPIPE_ELT* v          [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      T_kv       [[buffer(7)]],
    constant int&      D          [[buffer(8)]],
    constant int&      Hq         [[buffer(9)]],
    constant int&      Hkv        [[buffer(10)]],
    constant int&      q_offset   [[buffer(11)]],
    constant int&      kv_stride  [[buffer(12)]],
    constant int&      window     [[buffer(13)]],
    constant int&      ring_cap   [[buffer(14)]],
    constant int&      split      [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int g  = (int)simd_gid;
  const int qhead = kv * G + g;
  const int per = D / 32;
  const int q_pos = q_offset;
  const int last = min(T_kv - 1, q_pos);
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;
  const int total = last - first + 1;
  const int chunk = (total > 0) ? (total + split - 1) / split : 0;
  const int lo = first + s * chunk;
  const int hi = min(first + (s + 1) * chunk, last + 1);

  const device VPIPE_ELT* qh  = q + (uint)qhead * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = lo; t < hi; ++t) {
    const uint slot = (ring_cap > 0) ? (uint)(t % ring_cap) : (uint)t;
    float dot = 0.0f;
    for (int p = 0; p < per; ++p) {
      dot += qreg[p] * float(kkv[slot * D + lane * per + p]);
    }
    dot = simd_sum(dot);
    const float m_new = max(m, dot);
    const float corr = fast::exp(m - m_new);
    const float pj = fast::exp(dot - m_new);
    l = l * corr + pj;
    for (int p = 0; p < per; ++p) {
      acc[p] = acc[p] * corr + pj * float(vkv[slot * D + lane * per + p]);
    }
    m = m_new;
  }
  const uint base = ((uint)qhead * (uint)split + (uint)s) * (uint)D;
  for (int p = 0; p < per; ++p) {
    o_acc[base + (uint)(lane * per + p)] = acc[p];
  }
  if (lane == 0) {
    m_out[(uint)qhead * split + s] = m;
    l_out[(uint)qhead * split + s] = l;
  }
}

// sdpa_causal_gqa_vec_f16 -- DIRECT-READ flash-decode (the D=512 global layers)
// with UK-key unrolling + vectorized (vec4) K/V loads. Decode global attention
// is LATENCY-bound, not bandwidth- or compute-bound (~5 GB/s effective, ~85
// GFLOP/s; staging the KV or adding more position-splits does NOT help -- see
// gqa_decode_micro). The single-key loop in sdpa_causal_gqa_direct_f16 exposes
// per-key DRAM + simd_sum + exp latency on the critical path (m carries across
// iters). Here we issue UK independent keys' K loads together (the compiler
// overlaps their latency), batch their simd_sum reductions, then apply the
// online softmax in STRICT key order with the SAME scalar accumulation order as
// direct -> BIT-IDENTICAL output, just better-scheduled. vec4 loads cut the
// load-instruction count 4x. Requires per = D/32 divisible by 4 (D % 128 == 0;
// true for D=256/512). Same buffer/grid contract as sdpa_causal_gqa_direct_f16.
kernel void sdpa_causal_gqa_vec_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* k          [[buffer(1)]],
    const device VPIPE_ELT* v          [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      T_kv       [[buffer(7)]],
    constant int&      D          [[buffer(8)]],
    constant int&      Hq         [[buffer(9)]],
    constant int&      Hkv        [[buffer(10)]],
    constant int&      q_offset   [[buffer(11)]],
    constant int&      kv_stride  [[buffer(12)]],
    constant int&      window     [[buffer(13)]],
    constant int&      ring_cap   [[buffer(14)]],
    constant int&      split      [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int g  = (int)simd_gid;
  const int qhead = kv * G + g;
  const int per = D / 32;
  const int per4 = per / 4;
  const int q_pos = q_offset;
  const int last = min(T_kv - 1, q_pos);
  const int first = (window > 0) ? max(0, q_pos - window + 1) : 0;
  const int total = last - first + 1;
  const int chunk = (total > 0) ? (total + split - 1) / split : 0;
  const int lo = first + s * chunk;
  const int hi = min(first + (s + 1) * chunk, last + 1);

  const device VPIPE_ELT* qh  = q + (uint)qhead * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;
  typedef vec<VPIPE_ELT, 4> elt4;
  const int base_off = (int)lane * per;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = lo; t < hi; t += SDPA_GQA_UK) {
    const int ub = min((int)SDPA_GQA_UK, hi - t);
    // Hoist the ub keys' K loads (independent -> latency overlap) + dots.
    float dots[SDPA_GQA_UK];
    for (int u = 0; u < ub; ++u) {
      const uint slot =
          (ring_cap > 0) ? (uint)((t + u) % ring_cap) : (uint)(t + u);
      const device elt4* k4 =
          (const device elt4*)(kkv + slot * (uint)D + base_off);
      float d = 0.0f;
      for (int c = 0; c < per4; ++c) {
        const elt4 kv4 = k4[c];          // 8-byte vectorized load
        d += qreg[c * 4 + 0] * float(kv4.x);   // strict L-to-R order ==
        d += qreg[c * 4 + 1] * float(kv4.y);   // direct's scalar dot ->
        d += qreg[c * 4 + 2] * float(kv4.z);   // bit-identical result
        d += qreg[c * 4 + 3] * float(kv4.w);
      }
      dots[u] = simd_sum(d);
    }
    // Online softmax in strict key order; V loads vectorized per key.
    for (int u = 0; u < ub; ++u) {
      const uint slot =
          (ring_cap > 0) ? (uint)((t + u) % ring_cap) : (uint)(t + u);
      const float dot = dots[u];
      const float m_new = max(m, dot);
      const float corr = fast::exp(m - m_new);
      const float pj = fast::exp(dot - m_new);
      l = l * corr + pj;
      const device elt4* v4 =
          (const device elt4*)(vkv + slot * (uint)D + base_off);
      for (int c = 0; c < per4; ++c) {
        const elt4 vv4 = v4[c];
        acc[c * 4 + 0] = acc[c * 4 + 0] * corr + pj * float(vv4.x);
        acc[c * 4 + 1] = acc[c * 4 + 1] * corr + pj * float(vv4.y);
        acc[c * 4 + 2] = acc[c * 4 + 2] * corr + pj * float(vv4.z);
        acc[c * 4 + 3] = acc[c * 4 + 3] * corr + pj * float(vv4.w);
      }
      m = m_new;
    }
  }
  const uint obase = ((uint)qhead * (uint)split + (uint)s) * (uint)D;
  for (int p = 0; p < per; ++p) {
    o_acc[obase + (uint)(lane * per + p)] = acc[p];
  }
  if (lane == 0) {
    m_out[(uint)qhead * split + s] = m;
    l_out[(uint)qhead * split + s] = l;
  }
}

// sdpa_causal_gqa_vec_lin_f16 -- LINEARIZED-RING sibling of
// sdpa_causal_gqa_vec_f16 for the BOUNDED sliding-window decode layers. The
// host has guaranteed that the query's trailing window is one CONTIGUOUS
// physical span: the K/V ring carries a mirror tail (the first window-1 head
// slots duplicated at [ring_cap, ring_cap+window-1)), so a window-length scan
// that crosses the ring end reads straight into the mirror. This kernel scans
// physical slots [phys_first, phys_first+scan_len) LINEARLY -- no `% ring_cap`,
// no wrap branch, and no per-key window mask (every scanned slot is in-window).
// Softmax is order-invariant, so the physical scan order (which differs from the
// logical key order only across the wrap) yields the same result. vec4 loads +
// GQA are IDENTICAL to sdpa_causal_gqa_vec_f16; the split+merge structure is
// unchanged (this only makes the K/V READ contiguous).
//   0:q 1:k 2:v 3:o_acc 4:m_out 5:l_out 6:scale 7:T_kv 8:D 9:Hq 10:Hkv
//   11:q_offset 12:kv_stride 13:phys_first 14:scan_len 15:split
// grid (32, Hq, split); tg (32, G, 1).  (T_kv/q_offset bound for ABI parity.)
kernel void sdpa_causal_gqa_vec_lin_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* k          [[buffer(1)]],
    const device VPIPE_ELT* v          [[buffer(2)]],
    device float*           o_acc      [[buffer(3)]],
    device float*           m_out      [[buffer(4)]],
    device float*           l_out      [[buffer(5)]],
    constant float&    scale      [[buffer(6)]],
    constant int&      T_kv       [[buffer(7)]],
    constant int&      D          [[buffer(8)]],
    constant int&      Hq         [[buffer(9)]],
    constant int&      Hkv        [[buffer(10)]],
    constant int&      q_offset   [[buffer(11)]],
    constant int&      kv_stride  [[buffer(12)]],
    constant int&      phys_first [[buffer(13)]],
    constant int&      scan_len   [[buffer(14)]],
    constant int&      split      [[buffer(15)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  (void)T_kv; (void)q_offset;
  const int kv = (int)tid.y;
  const int s  = (int)tid.z;
  const int G  = Hq / Hkv;
  const int g  = (int)simd_gid;
  const int qhead = kv * G + g;
  const int per = D / 32;
  const int per4 = per / 4;
  // Linear physical span [phys_first, phys_first+scan_len); split across z.
  const int chunk = (scan_len > 0) ? (scan_len + split - 1) / split : 0;
  const int lo = phys_first + s * chunk;
  const int hi = min(phys_first + (s + 1) * chunk, phys_first + scan_len);

  const device VPIPE_ELT* qh  = q + (uint)qhead * D;
  const device VPIPE_ELT* kkv = k + (uint)kv * kv_stride * D;
  const device VPIPE_ELT* vkv = v + (uint)kv * kv_stride * D;
  typedef vec<VPIPE_ELT, 4> elt4;
  const int base_off = (int)lane * per;
  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = lo; t < hi; t += SDPA_GQA_UK) {
    const int ub = min((int)SDPA_GQA_UK, hi - t);
    float dots[SDPA_GQA_UK];
    for (int u = 0; u < ub; ++u) {
      const uint slot = (uint)(t + u);          // LINEAR -- no modulo
      const device elt4* k4 =
          (const device elt4*)(kkv + slot * (uint)D + base_off);
      float d = 0.0f;
      for (int c = 0; c < per4; ++c) {
        const elt4 kv4 = k4[c];
        d += qreg[c * 4 + 0] * float(kv4.x);
        d += qreg[c * 4 + 1] * float(kv4.y);
        d += qreg[c * 4 + 2] * float(kv4.z);
        d += qreg[c * 4 + 3] * float(kv4.w);
      }
      dots[u] = simd_sum(d);
    }
    for (int u = 0; u < ub; ++u) {
      const uint slot = (uint)(t + u);          // LINEAR -- no modulo
      const float dot = dots[u];
      const float m_new = max(m, dot);
      const float corr = fast::exp(m - m_new);
      const float pj = fast::exp(dot - m_new);
      l = l * corr + pj;
      const device elt4* v4 =
          (const device elt4*)(vkv + slot * (uint)D + base_off);
      for (int c = 0; c < per4; ++c) {
        const elt4 vv4 = v4[c];
        acc[c * 4 + 0] = acc[c * 4 + 0] * corr + pj * float(vv4.x);
        acc[c * 4 + 1] = acc[c * 4 + 1] * corr + pj * float(vv4.y);
        acc[c * 4 + 2] = acc[c * 4 + 2] * corr + pj * float(vv4.z);
        acc[c * 4 + 3] = acc[c * 4 + 3] * corr + pj * float(vv4.w);
      }
      m = m_new;
    }
  }
  const uint obase = ((uint)qhead * (uint)split + (uint)s) * (uint)D;
  for (int p = 0; p < per; ++p) {
    o_acc[obase + (uint)(lane * per + p)] = acc[p];
  }
  if (lane == 0) {
    m_out[(uint)qhead * split + s] = m;
    l_out[(uint)qhead * split + s] = l;
  }
}
// sdpa_paged_qtile_f16 -- query-TILED paged causal flash attention for
// head_dim 256 (Qwen3.5 full-attention PREFILL). The scalar and mb256 paged
// kernels are PER-QUERY: each query streams the whole causal K/V, so the K/V
// traffic is O(n_q^2) and dominates long-context prefill wall time. This
// kernel processes a TILE of SDPA_QT_BQ queries per threadgroup (one
// simdgroup per query) and stages each K/V block into threadgroup memory
// ONCE, reused by all BQ queries in the tile -> ~BQ x less K/V traffic, and
// all BQ simdgroups stay busy (no per-query 15/16-idle waste of mb256).
// Online softmax in fp32 registers, causal by global KV position. No
// simdgroup_matrix, so the bf16 variant compiles too. head_dim is fixed at
// 256 (Qwen full-attn); other dims use the scalar/mb256 paths.
// Same buffer contract as sdpa_paged_causal_f16.
// grid (BQ*32, Hq, ceil(n_q/BQ)); threadgroup (BQ*32, 1, 1).
#define SDPA_QT_BQ 16
#define SDPA_QT_BK 16
#define SDPA_QT_D  256
kernel void sdpa_paged_qtile_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  (void)D;   // bound for ABI parity; this kernel specializes on SDPA_QT_D
  const int h = (int)tid.y;
  const int tile = (int)tid.z;
  const int qi = tile * SDPA_QT_BQ + (int)simd_gid;
  const int kv = h / (Hq / Hkv);
  const int per = SDPA_QT_D / 32;   // 8
  const int q_pos = q_offset + qi;
  const uint page_stride = (uint)Hkv * page_tokens * SDPA_QT_D;
  const uint head_off = (uint)kv * page_tokens * SDPA_QT_D;

  threadgroup VPIPE_ELT Ks[SDPA_QT_BK * SDPA_QT_D];   // 16*256*2 = 8 KB
  threadgroup VPIPE_ELT Vs[SDPA_QT_BK * SDPA_QT_D];   // 8 KB -> 16 KB total

  const bool active = qi < n_q;
  float qreg[8], acc[8];
  if (active) {
    const device VPIPE_ELT* qh = q + ((uint)h * n_q + (uint)qi) * SDPA_QT_D;
    for (int p = 0; p < per; ++p) {
      qreg[p] = float(qh[lane * per + p]) * scale;
      acc[p] = 0.0f;
    }
  } else {
    for (int p = 0; p < per; ++p) { qreg[p] = 0.0f; acc[p] = 0.0f; }
  }
  float m = -INFINITY, l = 0.0f;
  bool done = false;

  // Highest query position in this tile (threadgroup-uniform): bounds the scan.
  const int tile_qpos_max =
      q_offset + min(tile * SDPA_QT_BQ + SDPA_QT_BQ - 1, n_q - 1);
  const uint ntg = (uint)SDPA_QT_BQ * 32;
  const uint ltid = simd_gid * 32 + lane;

  for (int pg = 0; pg < n_pages; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > tile_qpos_max) { break; }     // whole tile done (uniform)
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int bs = 0; bs < nvalid; bs += SDPA_QT_BK) {
      if (gstart + bs > tile_qpos_max) { break; }   // block past tile (uniform)
      const int bk = min(SDPA_QT_BK, nvalid - bs);
      // Cooperatively stage the K/V block into tg (all BQ*32 threads).
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = ltid; i < (uint)bk * SDPA_QT_D; i += ntg) {
        const uint key = i / SDPA_QT_D, dim = i % SDPA_QT_D;
        Ks[i] = kbase[(uint)(bs + (int)key) * SDPA_QT_D + dim];
        Vs[i] = vbase[(uint)(bs + (int)key) * SDPA_QT_D + dim];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active && !done) {
        for (int kk = 0; kk < bk; ++kk) {
          if (gstart + bs + kk > q_pos) { done = true; break; }  // causal
          float dot = 0.0f;
          for (int p = 0; p < per; ++p) {
            dot += qreg[p] * float(Ks[(uint)kk * SDPA_QT_D + lane * per + p]);
          }
          dot = simd_sum(dot);
          const float m_new = max(m, dot);
          const float corr = exp(m - m_new);
          const float pj = exp(dot - m_new);
          l = l * corr + pj;
          for (int p = 0; p < per; ++p) {
            acc[p] = acc[p] * corr
                     + pj * float(Vs[(uint)kk * SDPA_QT_D + lane * per + p]);
          }
          m = m_new;
        }
      }
    }
  }
  if (active) {
    const float inv_l = 1.0f / l;
    for (int p = 0; p < per; ++p) {
      out[((uint)h * n_q + (uint)qi) * SDPA_QT_D + lane * per + p] =
          VPIPE_ELT(acc[p] * inv_l);
    }
  }
}

// sdpa_paged_causal_f16 -- same online-softmax GQA causal attention as
// sdpa_causal_f16, but K/V live in a PAGED pool rather than one
// contiguous [Hkv, kv_stride, D] cache. A context's KV is an ordered
// list of pages; the page_table buffer carries, per page, the triplet
//   { physical_page_id, n_valid, global_start }
// (3 ints/entry). The pool layout is [max_pages, Hkv, page_tokens, D];
// page p's head `kv` starts at (p*Hkv + kv)*page_tokens*D. Pages are in
// ascending global-position order and within a page tokens are dense
// 0..n_valid, so the running KV position is monotonic across the whole
// walk -- letting one simdgroup per (head, query-pos) stream all pages
// with the carry kept in registers (no host-side carry threading).
//
// This is what makes no-copy branch work: children SHARE the parent's
// frozen prefix pages (by refcount) and only own their divergent tail
// pages; attention reads straight across the shared + private pages
// with zero gather/copy.
//
//   0:q 1:kpool 2:vpool 3:out (VPIPE_ELT)
//   4:scale(float) 5:D 6:Hq 7:Hkv 8:n_q 9:q_offset 10:page_tokens
//   11:n_pages (int)  12:page_table (int[n_pages*3])
// grid (32, H_q, n_q); threadgroup (32,1,1).  (Mirrors the contiguous
// kernel; only the K/V addressing differs.)
kernel void sdpa_paged_causal_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int h = (int)tid.y;
  const int qi = (int)tid.z;
  const int kv = h / (Hq / Hkv);
  const int per = D / 32;
  const int q_pos = q_offset + qi;

  const device VPIPE_ELT* qh = q + ((uint)h * n_q + qi) * D;
  // Per-page elem stride and the head's offset within a page.
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off = (uint)kv * page_tokens * D;

  float qreg[SDPA_MAX_PER];
  float acc[SDPA_MAX_PER];
  for (int p = 0; p < per; ++p) {
    qreg[p] = float(qh[lane * per + p]) * scale;
    acc[p] = 0.0f;
  }

  float m = -INFINITY;   // running max
  float l = 0.0f;        // running denom
  // The causal break is UNIFORM across the simdgroup (the condition is
  // lane-independent), so simd_sum stays well-formed; we break out of
  // both loops and still fall through to the output write below.
  bool done = false;
  for (int pg = 0; pg < n_pages && !done; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    const device VPIPE_ELT* kbase = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vbase = vpool + (uint)pid * page_stride + head_off;
    for (int t = 0; t < nvalid; ++t) {
      if (gstart + t > q_pos) { done = true; break; }   // positions monotonic
      float dot = 0.0f;
      for (int p = 0; p < per; ++p) {
        dot += qreg[p] * float(kbase[(uint)t * D + lane * per + p]);
      }
      dot = simd_sum(dot);

      const float m_new = max(m, dot);
      const float corr = exp(m - m_new);
      const float pj = exp(dot - m_new);
      l = l * corr + pj;
      for (int p = 0; p < per; ++p) {
        acc[p] = acc[p] * corr + pj * float(vbase[(uint)t * D + lane * per + p]);
      }
      m = m_new;
    }
  }

  const float inv_l = 1.0f / l;
  for (int p = 0; p < per; ++p) {
    out[((uint)h * n_q + qi) * D + lane * per + p] = VPIPE_ELT(acc[p] * inv_l);
  }
}

// sdpa_paged_mma_f16 -- MMA (simdgroup_matrix) flash attention over the
// paged KV pool, for PREFILL (n_q large). The scalar paged kernels above
// do per-key online-softmax dot products with only 32-lane parallelism;
// at long context that O(ctx) serial scan dominates prefill wall time.
// This kernel does QK^T and PV as dense simdgroup_matrix matmuls (8x8x8
// frags, half-in/float-accumulate, the same MMA the steel GEMM uses) with
// K/V staged once per block into threadgroup memory and shared across a
// 32-query tile. Synthesis of the two in-tree references: steel attn's
// register-resident MMA QK^T/softmax/PV flash loop (extern/mlx steel/attn)
// + the paged-pool walk and causal-by-global-position masking of the
// qwen3_5_paged_attn_block / sdpa_paged kernels.
//
// Tiling: one threadgroup per (head, 32-query block); WM*WD = 4*2 = 8
// simdgroups, 256 threads. The 8 simdgroups form WM=4 query bands (8 rows
// each) x WD=2 head-dim slices (D/2 cols each). The running flash output
// O lives in REGISTERS per simdgroup (8 rows x D/2 cols = D/(2*8) frags,
// f32 accumulate -- like MLX, no per-block tg round-trip for O); m_/l_
// per-row max/denom and the score/prob tiles live in tg. Per block: QK^T
// MMA with the two WD simdgroups of each band splitting the BK/8 key-frags
// (all 8 simdgroups busy) -> Ss; online softmax + causal mask done in
// parallel by ALL 256 threads (each simdgroup owns 4 rows, 8 lanes/row
// reduce max/sum via simd_shuffle_xor) -> Ps + per-row correction; each
// simdgroup rescales its O frags IN PLACE using the Apple 8x8 fragment row
// layout (frag row fm from the lane id -- no tg traffic) and accumulates PV
// into O. K/V walked page-by-page in BK=32 blocks (never straddles a page
// since page_tokens % BK == 0). q is read straight from device and `out` is
// written straight from O frags: the caller pads the q buffer by >=
// MMA_BQ*D halves so the last partial tile's 8-row frag loads stay in
// bounds (those rows are masked on read); the out write is bounds-guarded
// (row < n_q) so out needs no padding.
//
// Same buffer/layout contract as sdpa_paged_causal_f16:
//   0:q [Hq,n_q,D] 1:kpool 2:vpool 3:out [Hq,n_q,D]
//   4:scale 5:D 6:Hq 7:Hkv 8:n_q 9:q_offset 10:page_tokens 11:n_pages
//   12:page_table (int[n_pages*3] = {page_id, n_valid, global_start})
// Precondition: D % 16 == 0, D <= MMA_DMAX, page_tokens % MMA_BK == 0.
// grid (256, Hq, ceil(n_q/32)); threadgroup (256,1,1).

// KV-stride DRAM bank-conflict probe (user hypothesis): read a synthetic
// [Hkv, T, Dstride] K array with the EXACT access pattern of sdpa_paged_gqa_vec
// (32 lanes coalesced over the D=512 data dim, G simdgroups per kv-head each
// re-reading it, sp splits over T via grid.z), but with the per-key STORAGE
// stride Dstride decoupled from the data dim D. Dstride=D (512, power-of-2) vs
// D+pad tests whether the 512-element key stride aliases DRAM banks. Pure read
// (no softmax) to isolate read bandwidth; dummy simd_sum -> out[qhead*split+s]
// so the loads aren't dead-code-eliminated. grid (32,Hq,sp); tg (32,G,1).
kernel void kv_read_probe_f16(
    const device VPIPE_ELT* k    [[buffer(0)]],
    device float*           out  [[buffer(1)]],
    constant int& D        [[buffer(2)]],
    constant int& Dstride  [[buffer(3)]],
    constant int& T        [[buffer(4)]],
    constant int& Hq       [[buffer(5)]],
    constant int& Hkv      [[buffer(6)]],
    constant int& split    [[buffer(7)]],
    constant int& softmax  [[buffer(8)]],   // 0 = read-only, 1 = flash chain
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int kv   = (int)tid.y;            // kv-head (grid.y=Hq, tg.y=G -> Hkv tg)
  const int s    = (int)tid.z;            // split
  const int g    = (int)simd_gid;         // q-head within the GQA group
  const int G    = Hq / Hkv;
  const int qhead = kv * G + g;
  const int per  = D / 32;
  const int per4 = per / 4;
  const int chunk = (T + split - 1) / split;
  const int lo = s * chunk;
  const int hi = min((s + 1) * chunk, T);
  typedef vec<VPIPE_ELT, 4> elt4;
  // Per-kv-head plane base + this lane's D-strip; key t at + t*Dstride.
  const device VPIPE_ELT* kbase =
      k + (uint)kv * (uint)T * (uint)Dstride + (uint)lane * (uint)per;
  if (softmax == 0) {
    // READ-ONLY: pure coalesced stream (isolates raw KV-read bandwidth).
    float acc = 0.0f;
    for (int t = lo; t < hi; ++t) {
      const device elt4* k4 =
          (const device elt4*)(kbase + (uint)t * (uint)Dstride);
      for (int c = 0; c < per4; ++c) {
        const elt4 v4 = k4[c];
        acc += float(v4.x) + float(v4.y) + float(v4.z) + float(v4.w);
      }
    }
    acc = simd_sum(acc);
    if (lane == 0) { out[(uint)qhead * (uint)split + (uint)s] = acc; }
  } else if (softmax == 2) {
    // M=G MATERIALIZED read (§58): read K[t] ONCE, compute G PARALLEL dots (no
    // online-softmax serial carry) -> the QK pass is MEMORY-bound, so the
    // bank conflict should be EXPOSED here (unlike flash). qreg[G*per].
    float qreg[SDPA_MAX_PER * SDPA_MAX_G];
    for (int gg = 0; gg < G; ++gg) {
      for (int p = 0; p < per; ++p) { qreg[gg * per + p] = float(kbase[p]); }
    }
    float dd[SDPA_MAX_G];
    for (int gg = 0; gg < G; ++gg) { dd[gg] = 0.0f; }
    for (int t = lo; t < hi; ++t) {
      const device elt4* k4 =
          (const device elt4*)(kbase + (uint)t * (uint)Dstride);
      for (int c = 0; c < per4; ++c) {
        const elt4 v4 = k4[c];                // K[t] loaded ONCE -> G dots
        for (int gg = 0; gg < G; ++gg) {
          dd[gg] += qreg[gg * per + c * 4 + 0] * float(v4.x)
                  + qreg[gg * per + c * 4 + 1] * float(v4.y)
                  + qreg[gg * per + c * 4 + 2] * float(v4.z)
                  + qreg[gg * per + c * 4 + 3] * float(v4.w);
        }
      }
    }
    float o = 0.0f;
    for (int gg = 0; gg < G; ++gg) { o += simd_sum(dd[gg]); }
    if (lane == 0) { out[(uint)qhead * (uint)split + (uint)s] = o; }
  } else {
    // FAITHFUL FLASH: per key -> QK dot (simd_sum) -> online-softmax m/l/acc[per]
    // serial carry -> PV (reuse K as V). The same latency chain as the real
    // sdpa_paged_gqa_vec, so this predicts whether the read win survives.
    float qreg[SDPA_MAX_PER], acc[SDPA_MAX_PER];
    for (int p = 0; p < per; ++p) { qreg[p] = float(kbase[p]); acc[p] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int t = lo; t < hi; ++t) {
      const device elt4* k4 =
          (const device elt4*)(kbase + (uint)t * (uint)Dstride);
      float dot = 0.0f;
      for (int c = 0; c < per4; ++c) {
        const elt4 v4 = k4[c];
        dot += qreg[c * 4 + 0] * float(v4.x) + qreg[c * 4 + 1] * float(v4.y)
             + qreg[c * 4 + 2] * float(v4.z) + qreg[c * 4 + 3] * float(v4.w);
      }
      dot = simd_sum(dot);
      const float m_new = max(m, dot);
      const float corr = fast::exp(m - m_new);
      const float pj = fast::exp(dot - m_new);
      l = l * corr + pj;
      for (int c = 0; c < per4; ++c) {
        const elt4 v4 = k4[c];          // reuse K as V
        acc[c * 4 + 0] = acc[c * 4 + 0] * corr + pj * float(v4.x);
        acc[c * 4 + 1] = acc[c * 4 + 1] * corr + pj * float(v4.y);
        acc[c * 4 + 2] = acc[c * 4 + 2] * corr + pj * float(v4.z);
        acc[c * 4 + 3] = acc[c * 4 + 3] * corr + pj * float(v4.w);
      }
      m = m_new;
    }
    float s0 = 0.0f;
    for (int p = 0; p < per; ++p) { s0 += acc[p]; }
    s0 = simd_sum(s0) / (l > 0.0f ? l : 1.0f);
    if (lane == 0) { out[(uint)qhead * (uint)split + (uint)s] = s0; }
  }
}

// The simdgroup_matrix MMA flash kernel below is half-only (Llama/vision
// prefill); Qwen3.5's head_dim 256 uses the scalar/multi-simdgroup paged
// kernels instead, so the bf16 variant metallib skips this kernel.
#ifndef VPIPE_BF16
#define MMA_BQ     32    // query rows per threadgroup
#define MMA_BK     32    // key rows per block (must divide page_tokens)
#define MMA_WM     4     // query bands (8 rows each) = MMA_BQ / 8
#define MMA_WD     2     // head-dim slices
#define MMA_DMAX   128   // max head_dim supported (Llama = 128)
#define MMA_NDF    (MMA_DMAX / (MMA_WD * 8))   // max O col-frags / simdgroup

kernel void sdpa_paged_mma_f16(
    const device VPIPE_ELT* q          [[buffer(0)]],
    const device VPIPE_ELT* kpool      [[buffer(1)]],
    const device VPIPE_ELT* vpool      [[buffer(2)]],
    device VPIPE_ELT*       out        [[buffer(3)]],
    constant float&    scale      [[buffer(4)]],
    constant int&      D          [[buffer(5)]],
    constant int&      Hq         [[buffer(6)]],
    constant int&      Hkv        [[buffer(7)]],
    constant int&      n_q        [[buffer(8)]],
    constant int&      q_offset   [[buffer(9)]],
    constant int&      page_tokens[[buffer(10)]],
    constant int&      n_pages    [[buffer(11)]],
    const device int*  page_table [[buffer(12)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  lid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]])
{
  const int h    = (int)tid.y;
  const int qb   = (int)tid.z;
  const int kvh  = h / (Hq / Hkv);
  const int q_row0 = qb * MMA_BQ;               // first query row of tile
  if (q_row0 >= n_q) { return; }
  const int per    = D / 8;                     // head-dim frags
  const int dslice = D / MMA_WD;                // O cols per simdgroup
  const int ndf    = dslice / 8;                // O col-frags / simdgroup

  threadgroup half  Ks[MMA_BK * MMA_DMAX];      // staged K block
  threadgroup half  Vs[MMA_BK * MMA_DMAX];      // staged V block
  threadgroup float Ss[MMA_BQ * MMA_BK];        // scores (f32)
  threadgroup half  Ps[MMA_BQ * MMA_BK];        // softmax probs (f16)
  threadgroup float m_[MMA_BQ];                 // running row max
  threadgroup float l_[MMA_BQ];                 // running row denom
  threadgroup float corr_[MMA_BQ];              // per-row rescale this block

  if (lid < MMA_BQ) { m_[lid] = -INFINITY; l_[lid] = 0.0f; }

  const int wm    = (int)simd_gid / MMA_WD;     // query band 0..WM-1
  const int wd    = (int)simd_gid % MMA_WD;     // head-dim slice 0..WD-1
  const int srow0 = wm * 8;                     // simdgroup's tile rows
  const int grow0 = q_row0 + srow0;             // global query rows start
  const int dc0   = wd * dslice;                // simdgroup's first O col
  // Apple simdgroup_matrix<T,8,8> fragment layout: this lane carries one
  // row `fm` (cols fn, fn+1) of every 8x8 frag (matches steel BaseMMAFrag).
  const int qid = (int)lane / 4;
  const int fm  = (qid & 4) + (((int)lane / 2) % 4);     // frag row 0..7
  const int fn  = (qid & 2) * 2 + ((int)lane % 2) * 2;   // frag col 0,2,4,6

  // Register-resident flash output (8 rows x dslice cols), f32.
  simdgroup_matrix<float, 8, 8> O[MMA_NDF];
  for (int c = 0; c < ndf; ++c) { O[c] = simdgroup_matrix<float, 8, 8>(0.0f); }

  // Last query position in this whole tile -> causal early-out horizon.
  const int tile_max_qpos = q_offset + q_row0 + MMA_BQ - 1;
  const uint page_stride = (uint)Hkv * page_tokens * D;
  const uint head_off    = (uint)kvh * page_tokens * D;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bool done = false;
  for (int pg = 0; pg < n_pages && !done; ++pg) {
    const int pid    = page_table[pg * 3 + 0];
    const int nvalid = page_table[pg * 3 + 1];
    const int gstart = page_table[pg * 3 + 2];
    if (gstart > tile_max_qpos) { break; }      // page entirely in future
    const device VPIPE_ELT* kpage = kpool + (uint)pid * page_stride + head_off;
    const device VPIPE_ELT* vpage = vpool + (uint)pid * page_stride + head_off;

    for (int s0 = 0; s0 < nvalid; s0 += MMA_BK) {
      const int blockbase = gstart + s0;
      if (blockbase > tile_max_qpos) { done = true; break; }
      const int bkv = min(MMA_BK, nvalid - s0);

      // Stage K/V block into tg (cooperative, zero-pad rows >= bkv).
      for (uint e = lid; e < (uint)(MMA_BK * D); e += 256) {
        const int t = (int)e / D, d = (int)e % D;
        if (t < bkv) {
          Ks[e] = kpage[(uint)(s0 + t) * D + d];
          Vs[e] = vpage[(uint)(s0 + t) * D + d];
        } else {
          Ks[e] = (VPIPE_ELT)0; Vs[e] = (VPIPE_ELT)0;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // QK^T: S[8, BK] = Q[band rows,:] @ K[block,:]^T. The two wd
      // simdgroups of a band split the BK/8 key-frags between them (each
      // does the full-D contraction for its half), then store to Ss --
      // so all 8 simdgroups work the QK^T (no idle half).
      {
        const int nkf = MMA_BK / 8 / MMA_WD;       // key-frags per wd
        const int kf0 = wd * nkf;
        simdgroup_matrix<float, 8, 8> S[MMA_BK / 8 / MMA_WD];
        for (int j = 0; j < nkf; ++j) {
          S[j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (int dd = 0; dd < per; ++dd) {
          simdgroup_matrix<half, 8, 8> A;          // A[row, d]
          simdgroup_load(A, q + ((uint)h * n_q + grow0) * D + dd * 8, D);
          for (int j = 0; j < nkf; ++j) {
            simdgroup_matrix<half, 8, 8> B;        // B[d, jcol] = K[jcol, d]
            simdgroup_load(B, Ks, D, ulong2(dd * 8, (kf0 + j) * 8), true);
            simdgroup_multiply_accumulate(S[j], A, B, S[j]);
          }
        }
        for (int j = 0; j < nkf; ++j) {
          simdgroup_store(S[j], Ss + srow0 * MMA_BK + (kf0 + j) * 8, MMA_BK);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Online softmax + causal mask, parallel across all 256 threads:
      // simdgroup g owns rows [g*4, g*4+4); within it the 8 lanes sharing
      // a row each handle MMA_BK/8 cols and reduce max/sum over the aligned
      // 8-lane group via simd_shuffle_xor. Only the lead lane (cg==0) of
      // each row commits m_/l_/corr_ (the others compute identical values).
      {
        const int cpb = MMA_BK / 8;               // cols per lane
        const int r   = (int)simd_gid * 4 + (int)lane / 8;
        const int cg  = (int)lane % 8;            // 0..7 within the row
        const int qpos = q_offset + q_row0 + r;
        float lmax = -INFINITY;
        for (int jj = 0; jj < cpb; ++jj) {
          const int j = cg * cpb + jj;
          float s = Ss[r * MMA_BK + j] * scale;
          const int gkey = blockbase + j;
          if (j >= bkv || gkey > qpos) { s = -INFINITY; }
          Ss[r * MMA_BK + j] = s;
          lmax = max(lmax, s);
        }
        lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
        lmax = max(lmax, simd_shuffle_xor(lmax, 2u));
        lmax = max(lmax, simd_shuffle_xor(lmax, 4u));   // row max
        const float mold = m_[r];
        const float mnew = max(mold, lmax);
        const float corr = exp(mold - mnew);      // 0 on first block (-inf)
        float lsum = 0.0f;
        for (int jj = 0; jj < cpb; ++jj) {
          const int j = cg * cpb + jj;
          const float s = Ss[r * MMA_BK + j];
          const float p = (s == -INFINITY) ? 0.0f : exp(s - mnew);
          Ps[r * MMA_BK + j] = (VPIPE_ELT)p;
          lsum += p;
        }
        lsum += simd_shuffle_xor(lsum, 1u);
        lsum += simd_shuffle_xor(lsum, 2u);
        lsum += simd_shuffle_xor(lsum, 4u);             // row sum
        if (cg == 0) {
          l_[r] = l_[r] * corr + lsum;
          m_[r] = mnew;
          corr_[r] = corr;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Rescale this simdgroup's O frags by the per-row correction (each
      // lane owns row fm of every frag), then accumulate PV into them.
      const float cr = corr_[srow0 + fm];
      for (int c = 0; c < ndf; ++c) {
        O[c].thread_elements()[0] *= cr;
        O[c].thread_elements()[1] *= cr;
      }
      for (int kk = 0; kk < MMA_BK / 8; ++kk) {
        simdgroup_matrix<half, 8, 8> A2;          // P[row, k]
        simdgroup_load(A2, Ps + srow0 * MMA_BK + kk * 8, MMA_BK);
        for (int c = 0; c < ndf; ++c) {
          simdgroup_matrix<half, 8, 8> B2;        // V[k, dcol]
          simdgroup_load(B2, Vs, D, ulong2(dc0 + c * 8, kk * 8), false);
          simdgroup_multiply_accumulate(O[c], A2, B2, O[c]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Normalize (divide by row denom) and write O frags straight to out,
  // guarding rows past n_q. Each lane writes its 2 cols (fn, fn+1) of each
  // of the ndf col-frags for its frag row fm.
  const int row = grow0 + fm;
  if (row < n_q) {
    const float denom = l_[srow0 + fm];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    const uint orow = ((uint)h * n_q + row) * D;
    for (int c = 0; c < ndf; ++c) {
      const int col = dc0 + c * 8 + fn;
      out[orow + col]     = VPIPE_ELT(O[c].thread_elements()[0] * inv);
      out[orow + col + 1] = VPIPE_ELT(O[c].thread_elements()[1] * inv);
    }
  }
}
#endif  // !VPIPE_BF16
