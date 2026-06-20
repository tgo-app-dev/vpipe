// audio_encoder.metal -- Qwen3-ASR audio-tower conv stem kernel. The
// encoder body (LayerNorm, dense GEMM, GELU-erf, windowed SDPA) reuses
// the vision/LM kernels (layer_norm_bias_f16, dense_gemm_bias_f16,
// gelu_erf_f16, sdpa_window_f16, transpose/head_slice/residual); only the
// strided Conv2d stem is audio-specific. f16 storage, f32 accumulation.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// Conv2d 3x3, stride 2, padding 1, NHWC input, weight [Cout, 3, 3, Cin]
// (MLX conv2d layout), + per-output-channel bias. NO activation (the
// caller applies gelu_erf_f16 as a separate pass, sharing the vision
// tower's erf approximation). One thread per output element; accumulate
// over the 3x3xCin receptive field in f32.
//
// The Qwen3-ASR stem runs batched over N audio chunks (each chunk gets
// its own zero-padding, matching the training-time per-chunk conv). The
// third conv writes its output already permuted to [N, Wout, Cout, Hout]
// (transpose_out != 0) so the following conv_out GEMM reads contiguous
// [N*Wout, Cout*Hout] rows -- the equivalent of MLX's
// transpose({0,2,3,1}) + reshape.
//
//   0:in[N,H,W,Cin] 1:w[Cout,3,3,Cin] 2:bias[Cout] 3:out
//   4:N 5:H 6:W 7:Cin 8:Cout 9:Hout 10:Wout 11:transpose_out
// grid (Cout, Wout, N*Hout).
kernel void audio_conv2d_3x3_s2p1_f16(
    const device VPIPE_ELT* in   [[buffer(0)]],
    const device VPIPE_ELT* w    [[buffer(1)]],
    const device VPIPE_ELT* bias [[buffer(2)]],
    device VPIPE_ELT*       out  [[buffer(3)]],
    constant int& N             [[buffer(4)]],
    constant int& H             [[buffer(5)]],
    constant int& W             [[buffer(6)]],
    constant int& Cin           [[buffer(7)]],
    constant int& Cout          [[buffer(8)]],
    constant int& Hout          [[buffer(9)]],
    constant int& Wout          [[buffer(10)]],
    constant int& transpose_out [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int cout = (int)gid.x;
  const int wout = (int)gid.y;
  if (cout >= Cout || wout >= Wout) { return; }
  const int z = (int)gid.z;
  const int n = z / Hout;
  const int hout = z % Hout;
  if (n >= N) { return; }

  float acc = 0.0f;
  for (int kh = 0; kh < 3; ++kh) {
    const int ih = hout * 2 - 1 + kh;        // stride 2, pad 1
    if (ih < 0 || ih >= H) { continue; }
    for (int kw = 0; kw < 3; ++kw) {
      const int iw = wout * 2 - 1 + kw;
      if (iw < 0 || iw >= W) { continue; }
      const device VPIPE_ELT* inp =
          in + (((uint)n * H + ih) * W + iw) * Cin;
      const device VPIPE_ELT* wp =
          w + (((uint)cout * 3 + kh) * 3 + kw) * Cin;
      for (int ci = 0; ci < Cin; ++ci) {
        acc += float(inp[ci]) * float(wp[ci]);
      }
    }
  }
  acc += float(bias[cout]);

  uint oidx;
  if (transpose_out != 0) {
    oidx = (((uint)n * Wout + wout) * Cout + cout) * Hout + hout;
  } else {
    oidx = (((uint)n * Hout + hout) * Wout + wout) * Cout + cout;
  }
  out[oidx] = VPIPE_ELT(acc);
}

// im2col for the 3x3/stride-2/pad-1 conv stem: gather each output
// position's 3x3xCin receptive field into a contiguous K=9*Cin row, so
// the conv becomes a dense GEMM  out[M,Cout] = col[M,9*Cin] @ W[Cout,9*Cin]^T
// (W is the [Cout,3,3,Cin] weight reinterpreted as [Cout,9*Cin]; the K
// order (kh*3+kw)*Cin+ci matches the weight exactly). One thread per
// (output position m, kernel tap k9), copying Cin contiguous channels.
//
// row_major_hw=1 -> m = (n*Ho+ho)*Wo+wo   (natural [N,Ho,Wo,*] output)
// row_major_hw=0 -> m = (n*Wo+wo)*Ho+ho   (for the transpose_out conv:
//   the GEMM then yields [N,Wo,Ho,Cout]; swap_last2 -> [N,Wo,Cout,Ho]).
//
//   0:in[N,H,W,Cin] 1:col[M,9*Cin] 2:N 3:H 4:W 5:Cin 6:Ho 7:Wo 8:row_major_hw
// grid (M*9, 1, 1).
kernel void audio_im2col_3x3_s2p1_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       col [[buffer(1)]],
    constant int& N            [[buffer(2)]],
    constant int& H            [[buffer(3)]],
    constant int& W            [[buffer(4)]],
    constant int& Cin          [[buffer(5)]],
    constant int& Ho           [[buffer(6)]],
    constant int& Wo           [[buffer(7)]],
    constant int& row_major_hw [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
  const uint M = (uint)N * Ho * Wo;
  if (gid >= M * 9u) { return; }
  const int k9 = (int)(gid % 9u);
  const uint m = gid / 9u;
  const int kh = k9 / 3, kw = k9 % 3;

  int n, ho, wo;
  if (row_major_hw != 0) {
    wo = (int)(m % (uint)Wo);
    const uint t = m / (uint)Wo;
    ho = (int)(t % (uint)Ho);
    n  = (int)(t / (uint)Ho);
  } else {
    ho = (int)(m % (uint)Ho);
    const uint t = m / (uint)Ho;
    wo = (int)(t % (uint)Wo);
    n  = (int)(t / (uint)Wo);
  }

  device VPIPE_ELT* dst = col + (m * 9u + (uint)k9) * (uint)Cin;
  const int ih = ho * 2 - 1 + kh;          // stride 2, pad 1
  const int iw = wo * 2 - 1 + kw;
  if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
    for (int ci = 0; ci < Cin; ++ci) { dst[ci] = VPIPE_ELT(0); }
  } else {
    const device VPIPE_ELT* src =
        in + (((uint)n * H + ih) * W + iw) * Cin;
    for (int ci = 0; ci < Cin; ++ci) { dst[ci] = src[ci]; }
  }
}

// Swap the last two axes of a [batch, A, B] tensor -> [batch, B, A].
// Used to turn the transpose_out conv's GEMM result [N,Wo,Ho,Cout]
// (batch=N*Wo, A=Ho, B=Cout) into [N,Wo,Cout,Ho]. One thread per element.
//   0:in[batch,A,B] 1:out[batch,B,A] 2:batch 3:A 4:B
kernel void swap_last2_f16(
    const device VPIPE_ELT* in  [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int& batch        [[buffer(2)]],
    constant int& A            [[buffer(3)]],
    constant int& B            [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  const uint total = (uint)batch * (uint)A * (uint)B;
  if (gid >= total) { return; }
  const int b = (int)(gid / ((uint)A * (uint)B));
  const uint r = gid % ((uint)A * (uint)B);
  const int i = (int)(r / (uint)B);   // in A
  const int j = (int)(r % (uint)B);   // in B
  out[((uint)b * B + j) * A + i] = in[gid];
}
