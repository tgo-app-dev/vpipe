#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_LATENT_UPSCALER_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_LATENT_UPSCALER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/shared/dit-flops.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class I8GemmContext;
class WeightSet;

// MiniMax-H3's LATENT UPSCALER: a 3D-conv network that enlarges an H3
// video latent [24, T, h, w] to [24, T, H, W] without decoding it -- the
// "H3-Latent-Upscaler" (LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler,
// MIT) that Open-H3-Regenerate-2K runs before its refine, and ships in its
// `latent_upsampler/`.
//
// The network (the node's LatentResizer3D, attention off as the node
// forces it):
//
//   conv_in   3x3x3, 24 -> C
//   in stack  12 ResBlocks at the INPUT size, a TemporalConv after every
//             second one (after blocks 0, 2, 4, ...)
//   resize    trilinear to (T, H, W), align_corners=False -- the frame
//             count is unchanged, so it is bilinear per frame
//   out stack the same at the OUTPUT size
//   head      GroupNorm -> SiLU -> 3x3x3, C -> 24
//
// A ResBlock is GroupNorm(32) -> SiLU -> 3x3x3 conv -> GroupNorm scaled
// and shifted by an embedding of (scale - 1) -> SiLU -> 3x3x3 conv, plus
// its input; a TemporalConv is GroupNorm -> SiLU -> a depthwise conv of
// kernel 5 along time -> a 1x1 conv, plus its input. Every conv pads with
// zeros, in time too, and every GroupNorm takes its statistics over the
// WHOLE clip -- which is why the clip is upscaled in one pass rather than
// in pieces. C = 512.
//
// NORMALIZATION. The node normalizes its input with MiniMax-H3's latent
// statistics and denormalizes its output, because ComfyUI hands it RAW
// VAE latents. Open-H3-Regenerate-2K calls it on a latent that is ALREADY
// normalized -- the whitened latent every H3 stage here passes -- so the
// release normalizes twice. That is what its published results ran, so
// it is the default (Normalize::kRelease); kOnce feeds the whitened
// latent straight in and takes the output as whitened, the node's own
// intent.
//
// Activations are f16, channel-last per frame ([T, rows, C], frames
// contiguous), as the Wan VAE's are; the 3x3x3 convs run its 27-tap
// gather + GEMM in row bands (matmul2d on matrix cores, steel's 64x64
// tile elsewhere). A 768P draft to 2K (37 x 48 x 84 -> 37 x 90 x 160) is
// ~232 TFLOP, 78% of it in the output stack.
//
// INT8 (matrix cores only; `i8_tier` at load, the stages' i8_gemm_tier by
// number): the conv GEMMs on I8GemmContext -- accurate quantizes the
// weight per call; fast, token and token_plain HOLD the 3x3x3 convs'
// weights in the int8 layout (half the bytes), the rest per call.
class MetalMiniMaxH3LatentUpscaler {
 public:
  struct Config {
    int   in_channels = 24;
    int   channels    = 512;
    int   groups      = 32;
    int   embed_dim   = 64;
    float eps         = 1e-5f;
    // TemporalConv kernel; 0 when the file has none.
    int   temporal_kernel = 5;
    // The two stacks, in order: a ResBlock (false) or a TemporalConv
    // (true), by the index the file numbers them with.
    std::vector<bool> in_entries, out_entries;
  };

  // The node's _detect_arch, off the tensor headers: false (with a
  // reason) for a file that is not this network.
  static bool config_from_weights(const WeightSet& ws, Config* out,
                                  std::string* err = nullptr);

  // The file to open, from what a stage was given: a .safetensors path;
  // a directory holding `latent_upsampler/<one file>` (the
  // Open-H3-Regenerate-2K layout); or a directory with exactly one
  // .safetensors (the upscaler's own repo). Empty, with a reason, else.
  static std::string resolve_weights(const std::string& path,
                                     std::string* err = nullptr);

  // `i8_tier`: -1 off, 0 accurate, 1 fast, 2 token, 3 token_plain;
  // declined without matrix cores (and VPIPE_I8_GEMM / VPIPE_I8_TIER
  // override it, as everywhere).
  static std::unique_ptr<MetalMiniMaxH3LatentUpscaler>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       std::string* err = nullptr, int i8_tier = -1);

  // The tier a load with `i8_tier` runs at on a box with (or without)
  // matrix cores -- the int8 context's own reading -- for a plan.
  static int effective_i8_tier(int i8_tier, bool matrix_cores);

  // The node's TARGET_DIMENSIONS rule: pixel targets rounded to `align`
  // (Python's round, half to even), then to whole latent cells (16 px);
  // the embedding's scale is the mean of the two pixel ratios. A target
  // that is not whole patches comes out larger -- 1080 rows compute at
  // 1088 -- and the caller crops.
  struct Target {
    int    h = 0, w = 0;     // latent cells
    double scale = 1.0;
  };
  static Target target_for(int h, int w, int width_px, int height_px,
                           int align = 32);
  // The SCALE_BY rule: the latent's own pixel size times `scale`.
  static Target target_for_scale(int h, int w, double scale,
                                 int align = 32);

  enum class Normalize { kRelease, kOnce };

  // Enlarge `z`, [24, T, h, w] f32 in the whitened space, to [24, T, H,
  // W] f32 in the same space, `scale` the embedding's (Target::scale).
  // `progress(done, total)` after each block, from the calling thread.
  // False, with a reason, on failure.
  bool upscale(const float* z, int T, int h, int w, int H, int W,
               double scale, Normalize norm, std::vector<float>* out,
               std::string* err = nullptr,
               const std::function<void(int, int)>& progress = {});

  // Device bytes one pass allocates beside the weights, at its high
  // water: the larger of the input stack's three activations beside the
  // resize's output and the output stack's three beside the result,
  // plus the gather band and the statistics. MEASURED 768P -> 2K x 37
  // latents (M4 Pro): 1641 MB sampled at the peak, this 1719.
  static std::size_t scratch_bytes(const Config& cfg, int T, int h, int w,
                                   int H, int W, int i8_tier = -1);
  // The same for the default network, from the shape alone -- what a
  // plan can know before anything is opened.
  static std::size_t scratch_bytes(int T, int h, int w, int H, int W,
                                   int i8_tier = -1);
  // The weights as a load at `i8_tier` (effective) holds them, from the
  // file's headers.
  static std::size_t weight_bytes_for(const std::string& file,
                                      int i8_tier);

  // Every weight as held (f16 convs and 1x1s, f32 norms and embeddings).
  std::size_t weight_bytes() const noexcept { return _weight_bytes; }
  const Config& config() const noexcept { return _cfg; }
  bool matrix_core_gemm() const noexcept { return _use_mma2; }
  // The int8 tier that engaged (-1 none), and the GEMMs it ran in the
  // last upscale().
  int i8_tier() const noexcept { return _i8_tier; }
  long long i8_gemms() const noexcept { return _i8_gemms; }
  // FLOPs of the last upscale(), as counted at encode.
  std::uint64_t last_flops() const noexcept { return _last_flops; }

  ~MetalMiniMaxH3LatentUpscaler();

 private:
  MetalMiniMaxH3LatentUpscaler() = default;

  struct Conv {
    metal_compute::SharedBuffer w, b;   // [cout, taps * cin] f16, [cout]
    // ...or, under a tier that holds it, w's int8 layout at group gc.
    metal_compute::SharedBuffer codes, scales;
    int gc = 0;
    int cin = 0, cout = 0, taps = 0;    // taps: 27 (3x3x3) or 1 (1x1x1)
  };
  struct Norm { std::vector<float> gamma, beta; };
  struct ResBlock {
    Norm in_norm, out_norm;
    Conv conv1, conv2;
    std::vector<float> emb_w, emb_b;    // [2C, embed_dim], [2C]
  };
  struct Temporal {
    Norm norm;
    metal_compute::SharedBuffer dw_w, dw_b;   // f32 [C, K], [C]
    Conv pw;
  };
  struct Entry {
    bool temporal = false;
    ResBlock res;
    Temporal tc;
  };

  // Every weight, in the element type _bf16 names (the f16 -> bf16
  // fallback reloads them).
  bool load_all_(std::string* err);
  bool bind_kernels_();
  bool load_conv_(WeightSet& ws, const std::string& name, Conv* c,
                  std::string* err);
  bool load_norm_(WeightSet& ws, const std::string& name, Norm* n,
                  std::string* err);
  bool load_entry_(WeightSet& ws, const std::string& name, bool temporal,
                   Entry* e, std::string* err);

  struct Pass;
  // y = x W^T + bias, uncounted (the caller counts what is live).
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const Conv& c, const metal_compute::SharedBuffer& y,
             std::size_t ye, int M, int K);
  void dense_(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& x, std::size_t xe,
              const metal_compute::SharedBuffer& w,
              const metal_compute::SharedBuffer& bias, int N,
              const metal_compute::SharedBuffer& y, std::size_t ye, int M,
              int K);
  void count_(std::uint64_t flops) noexcept
  {
    _flops.add(flops);
    _counted += flops;
  }
  void conv3_(Pass& p, const Conv& c, const metal_compute::SharedBuffer& x,
              const metal_compute::SharedBuffer& y, int T, int rows_h,
              int rows_w);
  void norm_(Pass& p, const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& y, std::size_t rows,
             int slot, bool silu);
  void entry_(Pass& p, const Entry& e, int T, int hh, int ww,
              const metal_compute::SharedBuffer& x,
              const metal_compute::SharedBuffer& a,
              const metal_compute::SharedBuffer& b, int* slot);

  std::shared_ptr<WeightSet>    _ws;
  metal_compute::MetalCompute*  _mc = nullptr;
  Config                        _cfg;

  Conv conv_in_, conv_out_;
  Norm norm_out_;
  std::vector<Entry> in_, out_;
  std::vector<float> embed0_w_, embed0_b_, embed2_w_, embed2_b_;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_regen, _lib_mma;
  metal_compute::ComputeFunction _fn_gemm_64, _fn_gemm_32, _fn_mma,
                                 _fn_mma_wide, _fn_bias_add, _fn_im2col3d,
                                 _fn_gn_stats, _fn_gn_group, _fn_gn_apply,
                                 _fn_dwconv, _fn_resize, _fn_residual;
  std::unique_ptr<I8GemmContext> _i8;
  int _i8_want  = -1;          // the tier load() was asked for
  int _i8_tier  = -1;          // ...and the one that engaged
  long long _i8_gemms = 0;
  metal_compute::SharedBuffer _wdeq;   // a held layout widened
  bool _bf16       = false;   // element type; see load()
  bool _use_mma2   = false;
  int  _mma_min_m  = 64;
  std::size_t   _weight_bytes = 0;
  std::uint64_t _last_flops   = 0;
  std::uint64_t _counted      = 0;
  GpuFlopsTally _flops;
};

}  // namespace genai
}  // namespace vpipe

#endif
