#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_TURBO_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_TURBO_H

// MAGI-2-preview's video decoder: the distilled "TurboV3-Wan22-TinyShallow"
// (turbo_vae/: its config JSON and checkpoint.ckpt, a torch.save of the
// training state whose `ema_state_dict` is the model), written from the
// release's inference/model/turbo_vaed.py -- read, not run. The release
// decodes with it by default.
//
// The 48-channel Wan2.2 latent, un-whitened (z * std + mean, the release's
// constants), through a decoder of non-causal 3x3x3 convs -- time padded by
// REPLICATING a window's edge frames -- parameter-free RMSNorm over the
// channels (eps 1e-8) with SiLU (ReLU before the first conv of a resnet
// that follows an upsampler), two time-and-space x2 upsamplers (the time
// by a (3, 1, 1) conv doubling the channels, split into two frames; the
// first frame of the first window dropped), one space-only, and a 2 x 2
// unpatchify of 12 channels into RGB in [-1, 1]. 4 x 16 x 16 in all:
// T latent frames give 4 (T - 1) + 1.
//
// Decoded over SLIDING WINDOWS of latent frames: the first `first_chunk`
// (7) plus one of context, then `step` (7) at a time with one of context
// on each side, the 4 pixel frames of each context dropped; the latent is
// padded with its last frame to fit and the padding's frames cut.
//
// bf16 weights and activations (the release runs it in bf16), f32 sums;
// channel-last activations, each conv an im2col gather and the shared GEMM.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/dit-flops.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;

class MetalMagi2Turbo {
 public:
  // `dir`: the release's turbo_vae/ (the config JSON and checkpoint.ckpt);
  // `ckpt` the checkpoint's set, opened through the manager (null: a
  // private one).
  static std::unique_ptr<MetalMagi2Turbo>
  load(std::shared_ptr<WeightSet> ckpt, const std::string& dir,
       metal_compute::MetalCompute* mc, std::string* err);
  static std::unique_ptr<MetalMagi2Turbo>
  load(const std::string& dir, metal_compute::MetalCompute* mc,
       std::string* err);

  // The plan, before a load: the weights a load holds and a [48, T, h, w]
  // decode's scratch, off the checkpoint's shapes (no weights read).
  static bool plan(const std::string& dir, int T, int h, int w,
                   std::size_t* weights, std::size_t* scratch);
  ~MetalMagi2Turbo();
  MetalMagi2Turbo(const MetalMagi2Turbo&) = delete;
  MetalMagi2Turbo& operator=(const MetalMagi2Turbo&) = delete;

  // z: [48, T, h, w] f32. Out: [3, 4 (T - 1) + 1, 16 h, 16 w] f32 in the
  // decoder's range ([-1, 1] nominally, not clamped).
  bool decode(const float* z, int T, int h, int w, std::vector<float>* out,
              std::string* err);

  // The frames T latent frames decode to.
  static int frames_of(int T) { return 4 * (T - 1) + 1; }

  // The weights held, in bytes.
  std::uint64_t weight_bytes() const noexcept { return _weight_bytes; }

  // What a decode of a [48, T, h, w] latent allocates: three work
  // buffers at its widest window's (or tile's) largest activation, the
  // im2col band, the latent, the f32 clip on the GPU and the one it hands
  // back -- tiled, a tile's latent and clip twice, beside the whole clip.
  std::size_t scratch_bytes(int T, int h, int w) const;

  // SPATIAL TILES. Every op is local -- 3x3x3 convs, per-voxel norms, the
  // nearest upsamplers, the unpatchify -- so a tile decoded from its
  // latent cells plus a halo each side (clipped at the picture's edges)
  // is the whole decode's there, bit for bit, once the halo covers the
  // receptive field: a k x k conv at an upsampling scale s reaches
  // (k - 1) / 2 / s latent cells, summed over the network (halo_cells()).
  // A plane whose widest window's three work buffers would pass
  // kTileBudget is decoded in the fewest tiles that keep them under it.
  static constexpr std::size_t kTileBudget = std::size_t(4) << 30;
  double receptive_cells() const;
  int halo_cells() const;
  struct Tiling {
    int ny = 1, nx = 1;          // tiles down and across
    int core_h = 0, core_w = 0;  // latent cells a tile keeps
    int halo = 0;
  };
  Tiling tiling(int T, int h, int w) const;
  // TESTS: tiles of this core and halo (latent cells); 0 the rule.
  void set_tiles(int core_h, int core_w, int halo)
  {
    _tile_h = core_h;
    _tile_w = core_w;
    _tile_halo = halo;
  }

  // Gives back the work buffers between decodes (the clip is the
  // caller's once returned).
  void release_scratch();

  // The GPU arithmetic decode() credits to the achieved-FLOP/s meter
  // (common/perf-flops.h): every conv's GEMM, 2 x voxels x cout x taps x
  // cin, over the voxels it runs -- a window's context frames and a
  // tile's halo included, which it decodes again -- credited as each
  // window's command buffer retires. The norms, upsamplers and unpatchify
  // move data and are not counted.
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

 private:
  MetalMagi2Turbo() = default;
  struct Conv {
    metal_compute::SharedBuffer w, b;   // [Cout, taps * Cin] bf16, [Cout]
    int cin = 0, cout = 0, kt = 1, kh = 1, kw = 1;
  };
  struct Resnet {
    Conv c1, c2, sc;
    bool shortcut = false, relu_first = false;
  };
  struct Up {
    bool has_in = false;
    Resnet in;
    int mode = 0;           // 0 none, 1 upsample2d, 2 upsample3d
    Conv time, resample;
    std::vector<Resnet> res;
  };
  bool load_(const std::string& dir, std::string* err);
  bool conv_(const std::string& name, Conv* c, std::string* err);
  bool window_(int t0, int n, bool first, int keep0, int keep_n, int f_out,
               int T, int h, int w, std::string* err);
  // The whole plane, windows over time; `out` [3, F, 16 h, 16 w].
  bool decode_whole_(const float* z, int T, int h, int w,
                     std::vector<float>* out, std::string* err);
  std::size_t whole_scratch_(int T, int h, int w) const;
  std::size_t act_elems_(int n, int h, int w, bool first) const;

  std::shared_ptr<WeightSet> _ws;
  metal_compute::MetalCompute* _mc = nullptr;
  Conv _conv_in, _conv_out;
  std::vector<Resnet> _mid;
  std::vector<Up> _up;
  int _first = 7, _step = 7;
  int _tile_h = 0, _tile_w = 0, _tile_halo = -1;
  std::uint64_t _weight_bytes = 0;
  struct Kernels;
  std::unique_ptr<Kernels> _k;
  metal_compute::CommandStream _stream;
  GpuFlopsTally _flops;   // see gpu_flops()
  // Per decode: the latent on the GPU, the output, the work buffers.
  metal_compute::SharedBuffer _z, _out, _a, _b, _c, _col, _mean, _std;
  std::size_t _col_cap = 0;
};

}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_TURBO_H
