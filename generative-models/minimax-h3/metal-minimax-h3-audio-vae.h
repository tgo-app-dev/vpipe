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
// Decode only. Encoding audio needs the DAC-lineage convolutional encoder
// and the causal-attention posterior head, and nothing in the FL2VA task
// conditions on a reference soundtrack -- `build_packed_sequence` has no
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

  metal_compute::SharedBuffer f16_(WeightSet& ws, const std::string& nm);
  // Fold weight-norm and transpose in one pass. `transposed` selects the
  // ConvTranspose1d layout (weight [cin, cout, k], norm over dim 0), where
  // `rate` is the upsampling stride; an ordinary convolution ignores it
  // and pads to preserve length.
  Conv1d conv1d_(WeightSet& ws, const std::string& nm, bool transposed,
                 int dilation, int rate);
  Snake  snake_(WeightSet& ws, const std::string& nm);

  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, const Conv1d& c,
             const metal_compute::SharedBuffer& y, std::size_t y_off, int M,
             int N, int K, bool bias);
  // im2col + GEMM, banded over time. `in`/`out` are [B*T, cin]/[B*T, cout].
  void conv_(metal_compute::ComputeEncoder& enc, const Conv1d& c,
             const metal_compute::SharedBuffer& in,
             const metal_compute::SharedBuffer& out, int B, int T);
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

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_avae;
  metal_compute::ComputeFunction _fn_gemm, _fn_im2col, _fn_col2im, _fn_snake,
      _fn_up2, _fn_down2, _fn_residual, _fn_scale, _fn_clamp;
};

}  // namespace genai
}  // namespace vpipe

#endif
