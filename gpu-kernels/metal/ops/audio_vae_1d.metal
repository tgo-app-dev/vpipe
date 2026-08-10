// audio_vae_1d.metal -- the 1D ops of the MiniMax-H3 audio VAE's BigVGAN
// decoder: im2col / col2im for the (transposed) convolutions, the SnakeBeta
// activation, and the anti-aliased 2x resamplers that wrap it.
//
// Everything here is TIME-MAJOR [B, T, C] -- channel-contiguous, the
// transpose of PyTorch's [B, C, T]. That is what lets a convolution be an
// im2col plus a dense_gemm over W[Cout, K*Cin] instead of a bespoke kernel,
// and it is why the loaders transpose the checkpoint's weights once at load.
//
// B is the STEREO pair. The autoencoder is mono and MiniMax-H3 carries the
// two channels as two batch items, so every kernel decomposes its row index
// as (b, t) and clamps or skips at the per-item boundary. Running them as
// one longer sequence instead would let the left channel's tail convolve
// into the right channel's head -- inaudible on most material, which is
// exactly what makes it a bad bug to have.
//
//   im2col_1d_tc:      [B,Tin,C]      -> [rows, K*C]      (dilated, zero-pad)
//   col2im_1d_tc:      [B,Tin,K*Cout] -> [B,Tout,Cout]    (strided fold)
//   snake_beta:        x + rbeta * sin(ealpha * x)^2
//   resample_up2:      [B,L,C]        -> [B,2L,C]         (replicate, 12-tap)
//   resample_down2:    [B,L,C]        -> [B,Lo,C]         (replicate, 12-tap)

#include <metal_stdlib>
using namespace metal;

// Element (storage) type: half by default; -DVPIPE_ELT=bfloat for the bf16
// variant metallib. Math stays f32 -- the DAC/BigVGAN stack degrades audibly
// when the ACTIVATIONS are narrowed, and sin() of a learned frequency is the
// most phase-sensitive thing in the model.
#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// im2col for a time-major [B, Tin, C] signal -> [row_cnt, K*C], flattened as
// (k, c) so that
//   out[r, k*C + c] = in[b, t*stride + k*dilation - pad, c]
// (zero outside the sequence), with the GLOBAL output row
// r_global = row_off + r decomposing as b = r_global / Tout, t = r_global %
// Tout. A dense_gemm over W[Cout, K*C] then realizes the conv1d.
//
// `row_off` / `row_cnt` band the output over time so a long clip does not
// need the whole column matrix resident at once. Reads go to `in` directly
// rather than to a band of it, so a band needs no halo and row_off=0,
// row_cnt=B*Tout is the un-banded case.
//   0:in[B,Tin,C] 1:out[row_cnt,K*C] 2:Tin 3:C 4:K 5:dilation 6:pad
//   7:stride 8:Tout 9:row_off 10:row_cnt.   grid {K*C, row_cnt}
kernel void im2col_1d_tc_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      Tin      [[buffer(2)]],
    constant int&      C        [[buffer(3)]],
    constant int&      K        [[buffer(4)]],
    constant int&      dilation [[buffer(5)]],
    constant int&      pad      [[buffer(6)]],
    constant int&      stride   [[buffer(7)]],
    constant int&      Tout     [[buffer(8)]],
    constant int&      row_off  [[buffer(9)]],
    constant int&      row_cnt  [[buffer(10)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const uint cols = (uint)(K * C);
  if (tpig.x >= cols || (uint)tpig.y >= (uint)row_cnt) { return; }
  const uint c = tpig.x % (uint)C;
  const uint k = tpig.x / (uint)C;
  const uint r = (uint)row_off + tpig.y;          // global output row
  const int  b = (int)(r / (uint)Tout);
  const int  t = (int)(r % (uint)Tout);
  const int  s = t * stride + (int)k * dilation - pad;
  VPIPE_ELT val = (VPIPE_ELT)0;
  if (s >= 0 && s < Tin) {
    val = in[((ulong)b * Tin + s) * (uint)C + c];
  }
  out[(ulong)tpig.y * cols + tpig.x] = val;
}

// The fold half of a transposed convolution. A dense_gemm has already
// produced, for every INPUT frame i, the K separate contributions
// in[b, i, k*Cout + co] = sum_ci x[b,i,ci] * W[ci, co, k]; this scatters them
// to where stride and padding put them and sums the overlaps:
//   out[b, n, co] = bias[co] + sum_{i,k : i*stride + k - pad == n}
//                              in[b, i, k*Cout + co]
// Written as a GATHER (one thread owns one output) rather than a scatter, so
// there are no atomics: for a given n only the taps k congruent to (n + pad)
// mod stride can land there, and each fixes i.
//   0:in[B,Tin,K*Cout] 1:bias[Cout] 2:out[B,Tout,Cout] 3:Tin 4:Cout 5:K
//   6:stride 7:pad 8:Tout 9:has_bias.   grid {Cout, B*Tout}
kernel void col2im_1d_tc_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      Tin      [[buffer(3)]],
    constant int&      Cout     [[buffer(4)]],
    constant int&      K        [[buffer(5)]],
    constant int&      stride   [[buffer(6)]],
    constant int&      pad      [[buffer(7)]],
    constant int&      Tout     [[buffer(8)]],
    constant int&      has_bias [[buffer(9)]],
    uint2 tpig [[thread_position_in_grid]])
{
  if (tpig.x >= (uint)Cout) { return; }
  const uint co = tpig.x;
  const int  b  = (int)((uint)tpig.y / (uint)Tout);
  const int  n  = (int)((uint)tpig.y % (uint)Tout);
  float acc = (has_bias != 0) ? float(bias[co]) : 0.0f;
  for (int k = (n + pad) % stride; k < K; k += stride) {
    const int i = (n + pad - k) / stride;
    if (i < 0 || i >= Tin) { continue; }
    acc += float(in[((ulong)b * Tin + i) * (uint)(K * Cout) +
                    (uint)(k * Cout) + co]);
  }
  out[(ulong)tpig.y * (uint)Cout + co] = VPIPE_ELT(acc);
}

// SnakeBeta over a time-major [B, T, C] signal:
//   out = x + rbeta[c] * sin(ealpha[c] * x)^2
// The checkpoint stores alpha and beta in LOG space; the host passes
// ealpha = exp(alpha) and rbeta = 1/(exp(beta) + 1e-9) as f32, so the two
// exponentials are evaluated once per channel per load rather than once per
// sample.
//   0:in[B,T,C] 1:ealpha(f32)[C] 2:rbeta(f32)[C] 3:out[B,T,C] 4:C 5:n.
//   grid {C, n/C}
kernel void snake_beta_f16(
    const device VPIPE_ELT* in     [[buffer(0)]],
    const device float*     ealpha [[buffer(1)]],
    const device float*     rbeta  [[buffer(2)]],
    device VPIPE_ELT*       out    [[buffer(3)]],
    constant int&      C [[buffer(4)]],
    constant int&      n [[buffer(5)]],
    uint2 tpig [[thread_position_in_grid]])
{
  if (tpig.x >= (uint)C) { return; }
  const ulong gid = (ulong)tpig.y * (uint)C + tpig.x;
  if (gid >= (ulong)(uint)n) { return; }
  const float x = float(in[gid]);
  const float s = sin(ealpha[tpig.x] * x);
  out[gid] = VPIPE_ELT(x + rbeta[tpig.x] * s * s);
}

// The anti-aliased 2x upsampler: replicate-pad by 5, transposed depthwise
// convolution with the 12-tap Kaiser-sinc `filter` the checkpoint stores,
// scale by the ratio, then crop 15 from each end -- which comes out to
// exactly 2L samples. Only the 6 taps whose parity matches the output index
// contribute, since the other 6 land on the odd (zero-stuffed) positions.
//   0:in[B,L,C] 1:filt(f32)[12] 2:out[B,2L,C] 3:L 4:C.   grid {C, B*2L}
kernel void resample_up2_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    const device float*     filt [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      L [[buffer(3)]],
    constant int&      C [[buffer(4)]],
    uint2 tpig [[thread_position_in_grid]])
{
  if (tpig.x >= (uint)C) { return; }
  const int Lo = 2 * L;
  const int b  = (int)((uint)tpig.y / (uint)Lo);
  const int n  = (int)((uint)tpig.y % (uint)Lo);
  float acc = 0.0f;
  for (int k = (n + 15) & 1; k < 12; k += 2) {
    const int p = (n + 15 - k) / 2 - 5;          // undo the replicate pad
    const int s = clamp(p, 0, L - 1);
    acc += filt[k] * float(in[((ulong)b * L + s) * (uint)C + tpig.x]);
  }
  out[(ulong)tpig.y * (uint)C + tpig.x] = VPIPE_ELT(2.0f * acc);
}

// The matching 2x downsampler: replicate-pad (5, 6), 12-tap depthwise
// low-pass, stride 2. Output length is (L - 1) / 2 + 1, which the host
// passes as `Lo` rather than recomputing it here.
//   0:in[B,L,C] 1:filt(f32)[12] 2:out[B,Lo,C] 3:L 4:C 5:Lo.  grid {C, B*Lo}
kernel void resample_down2_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    const device float*     filt [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      L  [[buffer(3)]],
    constant int&      C  [[buffer(4)]],
    constant int&      Lo [[buffer(5)]],
    uint2 tpig [[thread_position_in_grid]])
{
  if (tpig.x >= (uint)C) { return; }
  const int b = (int)((uint)tpig.y / (uint)Lo);
  const int n = (int)((uint)tpig.y % (uint)Lo);
  float acc = 0.0f;
  for (int k = 0; k < 12; ++k) {
    const int s = clamp(2 * n + k - 5, 0, L - 1);
    acc += filt[k] * float(in[((ulong)b * L + s) * (uint)C + tpig.x]);
  }
  out[(ulong)tpig.y * (uint)C + tpig.x] = VPIPE_ELT(acc);
}

// The audio encoder's causal-attention projection, output side.
//
// `AttnProjection` narrows the 2048-wide encoder trunk to the 32-wide
// latent, and it does NOT do it the usual way: the heads are not
// concatenated back together but MEAN-POOLED away, and the surviving
// head width (in_dim / num_heads = 256) is then adaptively average-
// pooled down to out_dim. Reading it as a concatenation loads and runs
// perfectly and produces a different latent, which is exactly the kind
// of thing this kernel exists to make explicit.
//
// in is head-major [H, T, D] as the SDPA leaves it; out is [T, O] with
// O dividing D, so each output bin averages D/O neighbouring channels.
//   0:in[H,T,D] 1:out[T,O] 2:H 3:T 4:D 5:O.   grid {O, T}
kernel void attn_head_mean_pool_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&      H [[buffer(2)]],
    constant int&      T [[buffer(3)]],
    constant int&      D [[buffer(4)]],
    constant int&      O [[buffer(5)]],
    uint2 tpig [[thread_position_in_grid]])
{
  if (tpig.x >= (uint)O || tpig.y >= (uint)T) { return; }
  const int o    = (int)tpig.x;
  const int t    = (int)tpig.y;
  const int span = D / O;
  float acc = 0.0f;
  for (int h = 0; h < H; ++h) {
    const device VPIPE_ELT* row = in + ((ulong)h * (uint)T + t) * (uint)D;
    for (int j = 0; j < span; ++j) { acc += float(row[o * span + j]); }
  }
  out[(ulong)t * (uint)O + o] = VPIPE_ELT(acc / (float)(H * span));
}

// The posterior head's GeGLU: out = gelu_tanh(gate) * up.
//
// Fused rather than reusing swiglu_f16, whose activation is SiLU -- the
// two differ by a few percent per element, which is exactly the size of
// error that survives a round trip looking like a plausible latent.
//   0:gate 1:up 2:out 3:n.   grid {n}
kernel void geglu_tanh_f16(
    const device VPIPE_ELT* gate [[buffer(0)]],
    const device VPIPE_ELT* up   [[buffer(1)]],
    device VPIPE_ELT*       out  [[buffer(2)]],
    constant int&      n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const float g = float(gate[gid]);
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  out[gid] = VPIPE_ELT(0.5f * g * (1.0f + metal::precise::tanh(inner)) *
                       float(up[gid]));
}
