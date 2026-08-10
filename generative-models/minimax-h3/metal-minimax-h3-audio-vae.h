#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_AUDIO_VAE_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_AUDIO_VAE_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;

// The MiniMax-H3 audio VAE decoder (`MiniMaxH3AudioVAE`) -- BigVGAN, run
// on metal-compute.
//
// Waveform out, with no mel front-end and no separate vocoder: the 32-wide
// latents the DiT emits at 40 Hz are upsampled by 800 straight to 32 kHz
// samples, through seven transposed-convolution stages whose rates
// (5,5,2,2,2,2,2) multiply out to the hop length. Each stage is followed
// by three parallel anti-aliased multi-periodicity blocks (kernels 3, 7,
// 11 at dilations 1, 3, 5) whose outputs are AVERAGED, not concatenated.
//
// Three things about this decoder shape the implementation:
//
//   * IT IS MONO. MiniMax-H3's stereo pair is two BATCH items, and the
//     packed sequence carries them channel-major -- all of the left
//     channel's latents, then all of the right's. Everything below is
//     batched over that pair, and every kernel clamps at the per-item
//     boundary rather than at the buffer's, because a convolution that
//     reached across would blend the two channels together at the seam.
//   * THE ACTIVATION IS THE EXPENSIVE PART. Every SnakeBeta
//     (`x + sin(alpha*x)^2 / beta`) is wrapped in a 2x anti-aliasing
//     resample: upsample, activate, downsample, so its harmonics land
//     below Nyquist. There are 127 of them, and the last stages run at
//     the full sample rate.
//   * IT IS SENSITIVE TO PRECISION. The reference keeps the whole stack
//     in fp32 even under a bf16 pipeline. We store activations f16 (10
//     mantissa bits, and the signal is bounded) but every kernel does its
//     arithmetic in f32 -- bf16's 8 bits are not enough for the phase of
//     sin() at a learned frequency.
//
// Both directions. The ENCODER is a different shape from the decoder and
// is only reachable from `ref2va`, whose references carry soundtracks:
// a DAC convolution trunk (1 -> 64 channels, then five stride blocks
// doubling the width to 2048) and then a causal-ATTENTION projection
// that narrows 2048 -> 32 before the posterior heads. Its activation is
// the plain Snake1d (`x + sin(a*x)^2 / a`, a stored linearly) rather
// than the decoder's log-space SnakeBeta, and it is NOT wrapped in the
// anti-aliasing resamplers -- so the encoder is much cheaper per sample
// than the decoder despite the same lineage.
//
// `t2va` / `fl2va` never reach it: `build_packed_sequence` has no
// conditioning audio rows at all.
class MetalMiniMaxH3AudioVae {
 public:
  struct Config {
    int latent_channels = 32;    // width of one audio latent row
    int latent_dim      = 2048;  // the decoder trunk's input width
    int decoder_dim     = 1024;  // channels before the first upsample
    int stereo_channels = 2;     // `output_channel`: the soundtrack's
    int sample_rate     = 32000;

    // The upsample stages. `up_kernels` is read from the checkpoint's
    // own weight shapes rather than from config, because the released
    // rates pair with kernels that are neither 2*rate nor rate+1 (9 for
    // 5, 4 for 2) and a guess would load and produce the wrong length.
    std::vector<int> up_rates   = {5, 5, 2, 2, 2, 2, 2};
    std::vector<int> up_kernels = {9, 9, 4, 4, 4, 4, 4};

    // The three parallel AMP blocks per stage. Dilations cannot be read
    // back from the weights -- a dilated convolution has the same shape
    // as an undilated one -- so these are the reference's defaults.
    std::vector<int> res_kernels = {3, 7, 11};
    std::vector<std::vector<int>> res_dilations = {
        {1, 3, 5}, {1, 3, 5}, {1, 3, 5}};

    // Per-channel latent whitening, as for the video VAE: the DiT works
    // in normalized space, so a decode has to undo `(z - mean) / std`
    // first. Applied by the CALLER -- these are here so the stage has
    // one place to read them from.
    std::vector<float> latents_mean, latents_std;

    // The encoder's downsampling strides, mirroring `up_rates` (their
    // products must agree -- both are the 800-sample hop). Its kernels
    // are 2*stride and its channel widths double per stage from
    // `encoder_dim`, so neither needs storing.
    std::vector<int> down_rates = {2, 4, 4, 5, 5};
    int encoder_dim = 64;
    // The three residual units per encoder stage. As with the decoder's
    // dilations these cannot be read back from the weights -- a dilated
    // convolution has the same shape as an undilated one.
    std::vector<int> enc_dilations = {1, 3, 9};
    // The posterior head's causal attention. Heads are `latent_dim /
    // enc_attn_heads` wide and are mean-pooled away, NOT concatenated.
    int enc_attn_heads = 8;

    // Samples one latent frame becomes: the product of `up_rates`, 800
    // for the released checkpoint.
    int hop() const;
  };

  // Read a Config from the audio VAE's `config.json` + `metadata.json`.
  // `vae_dir` may be the `audio_vae` directory, the partition root, or
  // the repository root.
  static bool config_from_json(const std::string& vae_dir, Config& out,
                               std::string* err = nullptr);

  static std::string resolve_vae_dir(const std::string& path);

  static std::unique_ptr<MetalMiniMaxH3AudioVae>
  load(const std::string& vae_dir, metal_compute::MetalCompute* mc,
       const Config& cfg);

  static std::unique_ptr<MetalMiniMaxH3AudioVae>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg);

  ~MetalMiniMaxH3AudioVae();

  // Decode `[stereo, latent_channels, frames]` DENORMALIZED latents into
  // `[stereo, frames * hop()]` PLANAR f32 samples, clamped to [-1, 1].
  //
  // The caller un-whitens: `decode()` takes the latents the VAE itself
  // was trained on, not the DiT's normalized ones.
  //
  // When `taps` is non-null it is filled with the per-stage activations
  // in time-major `[stereo * length, channels]` order -- conv_pre's
  // output, then one entry per upsample stage -- which is what makes a
  // rel-L2 mismatch bisectable by depth. Test-only; it holds every
  // stage's activation at once.
  bool decode(const float* z, int stereo, int frames,
              std::vector<float>* pcm, std::string* err = nullptr,
              std::vector<std::vector<float>>* taps = nullptr);

  // Samples `frames` latent frames decode to, per channel.
  int decoded_samples(int frames) const;

  // Encode `[stereo, samples]` PLANAR f32 into the posterior MEAN,
  // `[stereo * frames, latent_channels]` row-major -- which is already
  // the CHANNEL-MAJOR row order the packed sequence wants, since the
  // stereo pair is the outer axis.
  //
  // The waveform is right-padded to a multiple of hop() first, so
  // `frames` is ceil(samples / hop) and a caller that needs an exact
  // count should trim its input rather than expect truncation.
  //
  // MiniMax-H3 conditions on the posterior MEAN and never samples it,
  // so the `logs_proj` head is not evaluated (nor loaded). The result is
  // UN-whitened, matching decode()'s input convention: the caller
  // applies `(z - latents_mean) / latents_std` to get DiT-space rows.
  //
  // `taps`, when non-null, collects the per-stage trunk activations in
  // time-major `[stereo * length, channels]` order (the input conv,
  // then one entry per stride block, then the trunk output and the
  // attention projection) -- which is what makes a rel-L2 mismatch
  // bisectable by depth. Test-only.
  bool encode(const float* pcm, int stereo, int samples,
              std::vector<float>* latents, int* frames = nullptr,
              std::string* err = nullptr,
              std::vector<std::vector<float>>* taps = nullptr);

  // Latent frames `samples` samples encode to, per channel: the
  // right-pad makes this a CEILING, not a truncation.
  int encoded_frames(int samples) const;

  // Whether the encoder half is loaded. It is built on first encode()
  // and released with the model; a decode-only pipeline never pays for
  // it. False when the checkpoint carries no `encoder.*` tensors at all
  // (nothing released does, but a decoder-only export could).
  bool has_encoder() const { return _enc_loaded; }

  const Config& config() const { return _cfg; }

 private:
  MetalMiniMaxH3AudioVae() = default;

  // A weight-normed conv1d, already folded (`w = v * g / ||v||`),
  // transposed to the GEMM layout and narrowed to f16. `w` is
  // [cout, k*cin] laid out (k, cin) to pair with im2col_1d_tc; a
  // transposed convolution instead stores [k*cout, cin] to pair with
  // col2im_1d_tc, and keeps its bias for the fold rather than the GEMM.
  struct Conv1d {
    metal_compute::SharedBuffer w, b;
    int cin = 0, cout = 0, k = 1, dilation = 1, pad = 0;
    // Downsampling stride. Only the encoder's stage convolutions use it;
    // everything else is 1, and a transposed convolution carries its
    // stride in `rate` instead.
    int stride = 1;
    bool empty() const { return w.empty(); }
  };

  // One anti-aliased SnakeBeta, with both exponentials evaluated at load.
  struct Snake {
    metal_compute::SharedBuffer ealpha, rbeta;   // f32 [C]
    int c = 0;
    bool empty() const { return ealpha.empty(); }
  };

  // BigVGAN's AMPBlock1: three (dilated conv, plain conv) pairs, each
  // convolution preceded by its own activation.
  struct AmpBlock {
    Conv1d convs1[3], convs2[3];
    Snake  acts[6];
  };

  struct UpStage {
    Conv1d   up;                 // the transposed convolution
    AmpBlock amp[3];             // one per resblock kernel
    int rate = 1, cout = 0;
  };

  // The encoder's residual unit: Snake, dilated k=7, Snake, k=1, added
  // back. Unlike the decoder's AMP blocks the activation is bare -- no
  // anti-aliasing resample around it.
  struct EncResUnit {
    Snake  act1, act2;
    Conv1d conv1, conv2;
  };

  // One encoder stage: three residual units at increasing dilation, an
  // activation, then the strided convolution that halves the time axis
  // and doubles the width.
  struct EncStage {
    EncResUnit units[3];
    Snake      act;
    Conv1d     down;
    int rate = 1, cin = 0, cout = 0;
  };

  // `AttnProjection`: a causal attention that narrows the trunk to the
  // latent width, in parallel with a plain linear over its own norm,
  // then a GeGLU MLP. `attn_k` has no bias -- the checkpoint stores a
  // frozen zero buffer for it, which is why only q and v carry one.
  struct PreBlock {
    metal_compute::SharedBuffer n1w, n1b, n3w, n3b, n2w, n2b;
    metal_compute::SharedBuffer qw, qb, kw, vw, vb;
    metal_compute::SharedBuffer aow, aob;       // attn.proj
    metal_compute::SharedBuffer pw, pb;         // the residual arm
    metal_compute::SharedBuffer mnw, mnb;       // mlp.norm
    metal_compute::SharedBuffer w0, b0, w1, b1, w2, b2;
    int hidden = 0;                             // GeGLU inner width
  };

  metal_compute::SharedBuffer f16_(WeightSet& ws, const std::string& nm);
  // Fold weight-norm and transpose in one pass. `transposed` selects the
  // ConvTranspose1d layout (weight [cin, cout, k], norm over dim 0), where
  // `rate` is the upsampling stride; an ordinary convolution ignores it
  // and pads to preserve length.
  // `stride` downsamples (encoder stage convolutions); `pad_override`
  // takes the padding from the caller instead of deriving it, which the
  // encoder's stride convolutions need -- the reference pads those by
  // ceil(stride/2), which is not the "same" rule every other
  // convolution here follows.
  Conv1d conv1d_(WeightSet& ws, const std::string& nm, bool transposed,
                 int dilation, int rate, int stride = 1,
                 int pad_override = -1);
  Snake  snake_(WeightSet& ws, const std::string& nm);
  // The ENCODER's plain Snake1d: one parameter, stored LINEARLY, and it
  // plays both roles -- `x + sin(a*x)^2 / a` where the decoder's
  // SnakeBeta has a separate beta and keeps both in log space. Same
  // kernel, so it is folded into the same (ealpha, rbeta) pair here.
  Snake  snake1d_(WeightSet& ws, const std::string& nm);

  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, const Conv1d& c,
             const metal_compute::SharedBuffer& y, std::size_t y_off, int M,
             int N, int K, bool bias);
  // M5 matrix-core (matmul2d) route for gemm_. Returns false -- with
  // nothing encoded -- when the shape does not qualify or the GPU has no
  // matrix units, and gemm_ then dispatches its steel kernel unchanged.
  // The bias is NOT applied here (the mma kernel has no bias slot); the
  // caller folds it with a bias_add_rows pass.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x, const Conv1d& c,
                 const metal_compute::SharedBuffer& y, std::size_t y_off,
                 int M, int N, int K);
  // im2col + GEMM, banded over time. `in` is [B*Tin, cin] and `out` is
  // [B*Tout, cout], where Tout is Tin for every stride-1 convolution and
  // the encoder's own output length for a strided one.
  void conv_(metal_compute::ComputeEncoder& enc, const Conv1d& c,
             const metal_compute::SharedBuffer& in,
             const metal_compute::SharedBuffer& out, int B, int Tin,
             int Tout);
  // A plain row-wise linear, `y[M, N] = x[M, K] * w^T + b`. The GEMM
  // wants its weight in the same [N, K] order a Conv1d holds, so this is
  // conv_ with k = 1 and none of the im2col.
  void linear_(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& x,
               const metal_compute::SharedBuffer& w,
               const metal_compute::SharedBuffer& b,
               const metal_compute::SharedBuffer& y, std::size_t y_off, int M,
               int N, int K);
  void layer_norm_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer& b,
                   const metal_compute::SharedBuffer& y, int rows, int C);
  // The alias-free activation: 2x up, SnakeBeta, 2x down. `scratch` holds
  // the doubled signal.
  void activate_(metal_compute::ComputeEncoder& enc, const Snake& s,
                 const metal_compute::SharedBuffer& in,
                 const metal_compute::SharedBuffer& out,
                 const metal_compute::SharedBuffer& scratch, int B, int T,
                 int C);

  struct Scratch {
    int b = -1, t = -1;
    // x0 holds the stage input across all three AMP blocks; acc sums
    // them; w is the running block state; t1/t2 are the convolution
    // temporaries; up is the doubled resample buffer.
    metal_compute::SharedBuffer x0, acc, w, t1, t2, up, col;
    std::size_t col_cap = 0;     // elements
  };
  bool ensure_scratch_(int B, int T);
  Scratch _s;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;

  Conv1d _dec_in, _conv_pre, _conv_post;
  std::vector<UpStage> _stages;
  Snake _act_post;
  metal_compute::SharedBuffer _filt_up, _filt_down;   // f32 [12]

  // The encoder half, built on first encode(). Kept separate from the
  // decoder's members so a pipeline that only decodes never allocates
  // it -- which is every `t2va` / `fl2va` run.
  bool build_encoder_(std::string* err);
  bool _enc_loaded = false;
  Conv1d _enc_in, _enc_out, _mean_proj;
  std::vector<EncStage> _enc_stages;
  Snake    _enc_act;
  PreBlock _pre;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_avae, _lib_sdpa;
  metal_compute::ComputeFunction _fn_gemm, _fn_im2col, _fn_col2im, _fn_snake,
      _fn_up2, _fn_down2, _fn_residual, _fn_scale, _fn_clamp;
  // ---- M5 matrix cores -------------------------------------------------
  // Loaded ONLY when the GPU has matrix units. The MPP matmul2d op
  // EMULATES on pre-M5 GPUs (see metal-compute.h's supports_matrix_cores),
  // so an ungated load would trade this tower's simdgroup-MMA steel kernel
  // -- ~10 TFLOP/s on M4 Pro -- for an emulation. `_use_mma2` false leaves
  // gemm_ dispatching exactly the kernel it dispatched before.
  metal_compute::ComputeLibrary _lib_dense_mma;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_bias_add;
  bool _use_mma2 = false;
  // Shape gate. The win regime measured on this decoder is the trunk's
  // resblock convolutions (M 3k-9k, N 128-256, K 0.9k-2.8k), which carry
  // ~75% of the decode's GEMM FLOPs. The rest of the stack is a very
  // different shape -- huge M against N of 8 to 64 -- and a 128-wide
  // output tile there would leave most of its columns idle, so a narrow
  // N stays on steel. Both overridable for a re-probe.
  // MEASURED on M5 (5 s stereo, arms interleaved, three rounds): steel
  // 370 ms; m1024/n128 305; m256/n128 296; m1024/n64 300; m256/n64 293.
  // Below n=64 the curve is FLAT (292-297 ms at n32/n16, within noise),
  // so the gate stops where the gain does rather than routing every
  // skinny GEMM through a tile it cannot fill.
  int _mma_min_m = 256;
  int _mma_min_n = 64;
  // Encoder-only: the posterior head's attention and its reductions.
  metal_compute::ComputeFunction _fn_ln, _fn_sdpa_causal, _fn_transpose,
      _fn_head_pool, _fn_geglu;
};

}  // namespace genai
}  // namespace vpipe

#endif
