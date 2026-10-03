#ifndef VPIPE_GENERATIVE_MODELS_YUE2_METAL_OOBLECK_DECODER_H
#define VPIPE_GENERATIVE_MODELS_YUE2_METAL_OOBLECK_DECODER_H

// MetalOobleckDecoder -- the Oobleck audio VAE decoder (stable-audio-
// tools), as YuE2 ships it (m-a-p/YuE2-Vae, YuE2-Vae-legacy): 64-wide
// latents at 25 Hz to 48 kHz STEREO, through six transposed-convolution
// upsamples (6, 5, 4, 4, 2, 2 -> a 1920-sample hop), each followed by
// three residual units at dilations 1, 3, 9.
//
// The same family of ops as MiniMax-H3's BigVGAN and on the same kernels
// (audio_vae_1d.metal), with three differences that each change the
// waveform if missed:
//
//   * NO anti-aliasing around the activations: a bare SnakeBeta, whose
//     alpha and beta are stored in LOG space.
//   * Stereo is two OUTPUT CHANNELS of one sequence, not a batch pair.
//   * The transposed convolutions pad ceil(stride / 2), not
//     (k - stride) / 2 -- for the stride-5 stage that is 3 against 2, and
//     the decode comes out 1920 * frames - 64 samples long, not
//     1920 * frames.
//
// Activations are f16 with f32 arithmetic in every kernel, as for the H3
// decoder; the reference runs the whole decoder in fp32.
//
// TILED, exactly as the reference tiles: a core of frames is decoded with
// a halo of context on each side as an INDEPENDENT sequence (zero-padded
// at the tile edges) and only its core kept. With a halo at least the
// decoder's receptive field (16 frames for the released config) the
// result does not depend on the tile size at all -- the reference's own
// full and tiled decodes agree bit for bit -- so the tile is a memory
// knob, not a quality one.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class WeightSet;

class MetalOobleckDecoder {
public:
  struct Config {
    int latent_dim   = 64;
    int channels     = 64;
    std::vector<int> c_mults = {1, 2, 4, 8, 16, 32};
    std::vector<int> strides = {2, 2, 4, 4, 5, 6};
    int out_channels = 2;
    int sample_rate  = 48000;
    bool final_tanh  = false;
    int core_frames  = 1024;     // the reference's tile; ours is smaller
    int halo_frames  = 16;
    int hop() const;
  };

  // config.json of a YuE2 VAE export (model_type "yue2_vae"): its
  // decoder_config, sample_rate and decode halo.
  static bool config_from_dir(const std::string& dir, Config* out,
                              std::string* err = nullptr);
  static bool is_oobleck_dir(const std::string& dir);

  static std::unique_ptr<MetalOobleckDecoder> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      const Config& cfg, std::string* err = nullptr);
  static std::unique_ptr<MetalOobleckDecoder> load(
      const std::string& dir, metal_compute::MetalCompute* mc,
      std::string* err = nullptr);
  ~MetalOobleckDecoder();

  const Config& config() const { return _cfg; }
  // Whether the M5 matrix-core GEMM route was selected.
  bool uses_mma() const { return _use_mma; }

  // Latent frames per decoded tile (the halo comes on top). A memory knob
  // only: with the config's halo every tile size gives the same waveform.
  void set_core_frames(int n) { _core = std::max(n, 1); }
  int  core_frames() const { return _core; }

  // Samples `frames` latent frames decode to, per channel: 1920 * frames
  // - 64 for the released config.
  int natural_length(int frames) const;

  // What a decode of `frames` costs: `pcm` is the planar f32 output this
  // hands downstream, `arena` the scratch a tile allocates.
  void decode_cost(int frames, std::size_t* pcm, std::size_t* arena) const;

  // Called once per tile with (done, total). Return false to stop.
  using ProgressFn = std::function<bool(int done, int total)>;

  // `z` is CHANNEL-major [latent_dim, frames] f32 (the reference's
  // [1, 64, T] without its batch axis). `pcm` comes back PLANAR
  // [out_channels, natural_length(frames)] f32, clamped to [-1, 1] when
  // `clamp` (the reference pipeline clamps after decoding; the VAE itself
  // does not).
  bool decode(const float* z, int frames, std::vector<float>* pcm,
              bool clamp = true, const ProgressFn& progress = {},
              std::string* err = nullptr);

  std::size_t resident_bytes() const;

private:
  MetalOobleckDecoder() = default;

  // A weight-normed convolution, folded (w = g * v / ||v||) into the GEMM
  // layout in f16: [cout, k * cin] in (k, cin) order for an ordinary one,
  // [k * cout, cin] for a transposed one (its bias goes to the fold).
  struct Conv {
    metal_compute::SharedBuffer w, b;
    int cin = 0, cout = 0, k = 1, dilation = 1, pad = 0, stride = 1;
    bool empty() const { return w.empty(); }
  };
  struct Snake {
    metal_compute::SharedBuffer ealpha, rbeta;   // f32 [C]
    int c = 0;
    bool empty() const { return ealpha.empty(); }
  };
  struct ResUnit {
    Snake a1, a2;
    Conv  c1, c2;
  };
  struct Block {
    Snake   act;
    Conv    up;                 // transposed, stride `rate`
    ResUnit res[3];
    int     rate = 1;
  };

  bool init_(const std::shared_ptr<WeightSet>& ws,
             metal_compute::MetalCompute* mc, const Config& cfg,
             std::string* err);
  Conv  conv_(WeightSet& ws, const std::string& nm, bool transposed,
              int dilation, int stride);
  Snake snake_(WeightSet& ws, const std::string& nm);
  metal_compute::SharedBuffer f16_(WeightSet& ws, const std::string& nm);

  bool ensure_scratch_(int frames);
  bool decode_tile_(const float* z, int frames, int f0, int f1,
                    std::vector<float>* tile, std::string* err);

  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x,
                 const metal_compute::SharedBuffer& w,
                 const metal_compute::SharedBuffer* bias,
                 const metal_compute::SharedBuffer& y, std::size_t ye,
                 int M, int N, int K);
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer* bias,
             const metal_compute::SharedBuffer& y, std::size_t ye,
             int M, int N, int K);
  void conv_run_(metal_compute::ComputeEncoder& enc, const Conv& c,
                 const metal_compute::SharedBuffer& in,
                 const metal_compute::SharedBuffer& out, int T);
  void snake_run_(metal_compute::ComputeEncoder& enc, const Snake& s,
                  const metal_compute::SharedBuffer& in,
                  const metal_compute::SharedBuffer& out, int T);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;

  Conv _conv_in, _conv_out;
  std::vector<Block> _blocks;
  Snake _act_out;

  // Scratch for one tile: three activations at the widest stage, the
  // transposed convolutions' pre-fold GEMM output, and a banded im2col.
  int _core = 256;
  int _tile_frames = 0;
  metal_compute::SharedBuffer _a, _b, _c, _pre, _col;
  std::size_t _col_cap = 0;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_avae, _lib_mma;
  metal_compute::ComputeFunction _fn_gemm, _fn_im2col, _fn_col2im,
      _fn_snake, _fn_add;
  // M5 matrix cores: matmul2d for the GEMMs wide and tall enough to fill
  // its 128-row tile. The upsampling tail (tens of thousands of rows
  // against 64-wide outputs) stays on steel. VPIPE_YUE2_VAE_NO_MMA2 is
  // the A/B.
  metal_compute::ComputeFunction _fn_mma, _fn_mma_deep, _fn_bias_rows;
  bool _use_mma = false;
  int _mma_min_m = 256;
  int _mma_min_n = 64;
};

}  // namespace vpipe::genai

#endif
