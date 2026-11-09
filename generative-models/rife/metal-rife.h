#ifndef VPIPE_GENERATIVE_MODELS_RIFE_METAL_RIFE_H
#define VPIPE_GENERATIVE_MODELS_RIFE_METAL_RIFE_H

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/dit-flops.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

// RIFE -- Real-time Intermediate Flow Estimation (Huang et al., ECCV 2022),
// as Practical-RIFE (github.com/hzwer/Practical-RIFE, MIT) publishes it:
// a frame at any time t in (0, 1) between two frames, from a flow the
// network estimates coarse to fine. What temporal-resample's `rife`
// method runs.
//
// WHICH CHECKPOINTS. Practical-RIFE ships each model as a train_log/
// directory: `flownet.pkl` (a torch state_dict, keys `module.`-prefixed)
// beside the python that defines it. 4.22, 4.22.lite, 4.25, 4.25.lite and
// 4.26 are one network shape, read here off the TENSOR NAMES AND SHAPES:
//
//   encode.cnn{0,1,2}   3x3 convs, s2 then s1, s1 (LeakyReLU 0.2), width
//                       16 (4.25) or 32 (4.22)
//   encode.cnn3         ConvTranspose2d(k4, s2, p1) to F = 4 or 8
//                       channels: each frame's features, full size
//   block<i>            an IFBlock of width c (192 / 128 / 96 / 64 / 32 on
//                       4.25): conv s2, conv s2 (LeakyReLU), eight ResConv
//                       (LeakyReLU(conv(x) * beta + x)), ConvTranspose2d
//                       (k4, s2, p1) to 52 channels then PixelShuffle(2) --
//                       a flow update (4), a mask logit and 8 features
//
// Block i runs at 1 / scale_list[i] of the frame on [both frames warped
// by the flow so far, their features warped, t, the last mask and
// features, the flow], and the result is both frames warped by the final
// flow, blended by sigmoid(mask). The SCALE LIST is not in the tensors:
// it is the driver's (`scale_list = [16/scale, 8/scale, ...]` in
// RIFE_HDv3.py), read from the file beside the checkpoint when it is
// there, else 2^(n-1) .. 1 for n blocks -- right for every release above
// but 4.25.lite, whose driver says [32, 16, 8, 4, 1]. Older releases (no
// Head, no block features: 4.6 and before) do not have this shape and
// are refused by name.
//
// GEOMETRY is the driver's too (inference_video.py): a frame is padded
// with ZEROS on the right and bottom to a multiple of max(128, 128 /
// scale) and the result cut back, `scale` being its --scale (0.5 is what
// it recommends for 4K; 0.25, 0.5, 1, 2, 4 are allowed).
//
// THE CONVS run on the GPU's matrix cores where it has them (M5): the
// MPP convolution2d hardware op with the activation fused into its
// epilogue (conv2d_mma.metal, conv2d_hw_3x3_*_epi_*), over HWIO twins of
// the weights. Elsewhere, and for a conv no hardware entry takes, the
// shared gather conv (dense_gemm.metal, conv3x3_gemm_*).
// VPIPE_RIFE_NO_HWCONV=1 keeps every conv on the gather conv (the A/B).
//
// PRECISION. Weights and activations f16, accumulation f32 (both conv
// routes); the frames, the flow and the blend f32. The release runs
// fp32, or fp16 with --fp16; held against an fp32 restatement of the
// network (tools/dump_rife_golden.py), this port reads as close as the
// release's own fp16 mode does.
class MetalRife {
public:
  struct Info {
    int head_width = 0;            // encode.cnn0's output channels
    int features   = 0;            // F: encode.cnn3's output channels
    std::vector<int> widths;       // each IFBlock's c
    std::vector<int> inputs;       // each IFBlock's first conv's inputs
    std::vector<double> scales;    // the scale list at --scale 1
    bool scales_from_driver = false;
    std::size_t params = 0;
    std::string prefix;            // "module." on the release's files
    int blocks() const noexcept { return (int)widths.size(); }
  };

  // The checkpoint FILE for a path: `flownet.pkl` in a directory (a
  // train_log/), or the path itself when it is a file. Empty + *err when
  // neither holds one.
  static std::string resolve_weights(const std::string& path,
                                     std::string* err = nullptr);

  // True when `path` (a directory or the file) holds a checkpoint this
  // class runs -- model detection's probe; reads tensor headers only.
  static bool looks_like_rife(const std::string& path);

  // Read the network off `file`'s tensor headers (and the driver beside
  // it) without reading a weight. False + *err on a checkpoint that is
  // not this network.
  static bool read_info(const std::string& file, Info* out,
                        std::string* err);

  // The driver's --scale values, and the padding multiple one implies.
  static bool valid_scale(double scale) noexcept;
  static int pad_multiple(double scale) noexcept;
  // The scale the driver recommends for a frame: 0.5 for 4K-class frames
  // (a side past 2560), else 1.
  static double auto_scale(int h, int w) noexcept;

  // Whether a load on `mc` runs its convs on the hardware op: matrix
  // cores, and VPIPE_RIFE_NO_HWCONV unset. Asked by the plan as by load.
  static bool hw_conv_wanted(const metal_compute::MetalCompute* mc);

  // What a load holds (f16, in the convs' own layout, plus the HWIO twins
  // the hardware op reads when `hw`) and what running it at a geometry
  // allocates, both from the Info alone -- the figures the stage books
  // before anything loads, and the ones the class allocates from.
  static std::size_t weight_bytes(const Info& info, bool hw);
  // One prepared frame (see Frame).
  static std::size_t frame_bytes(const Info& info, int h, int w,
                                 double scale);
  // The scratch prepare() and interpolate() share, plus the staging of
  // one frame in and one out (`f32` the element type; `hw` the route, whose
  // flow head is 256 wide where the gather conv's is 208).
  static std::size_t scratch_bytes(const Info& info, int h, int w,
                                   double scale, bool f32, bool hw);

  static std::unique_ptr<MetalRife>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const std::string& file, std::string* err);

  ~MetalRife();

  const Info& info() const noexcept { return _info; }

  // A frame made ready for interpolation: padded, and its features
  // encoded. Prepared ONCE per source frame -- a frame is the second of
  // one pair and the first of the next, and of every output time
  // between them.
  struct Frame {
    metal_compute::SharedBuffer img;    // float4 [ph * pw]
    metal_compute::SharedBuffer feat;   // half   [ph * pw * F]
    int h = 0, w = 0, ph = 0, pw = 0;
    double scale = 0.0;
    bool ready() const noexcept { return !img.empty(); }
  };

  // Prepare a planar [3, h, w] frame, u8 or f32 in [0, 1] (`f32`), at
  // `scale`. Reuses `out`'s buffers when they fit.
  bool prepare(const void* planar, bool f32, int h, int w, double scale,
               Frame* out, std::string* err);

  // The frame at time `t` in (0, 1) between `a` and `b` (prepared at one
  // geometry and scale), written planar [3, h, w] into `out` -- f32 as
  // computed, or u8 rounded.
  bool interpolate(const Frame& a, const Frame& b, float t, bool f32,
                   void* out, std::string* err);

  // What the model holds now: its weights plus the scratch of the last
  // geometry it ran (Frames are the caller's).
  std::size_t resident_bytes() const noexcept;
  std::size_t held_weight_bytes() const noexcept { return _weight_bytes; }

  // Whether the convs run on the hardware op, and how many dispatches
  // took it -- the ENGAGEMENT evidence: the route is a silent fallback
  // otherwise, and a run that stayed on the gather conv matches every
  // golden and is merely slower.
  bool hw_conv() const noexcept { return _use_hw; }
  std::uint64_t hw_dispatches() const noexcept { return _hw_dispatches; }
  std::uint64_t gather_dispatches() const noexcept
  {
    return _gather_dispatches;
  }

  // With VPIPE_RIFE_PROFILE set, GPU time per op category of the last
  // prepare + interpolate ("-hw" marks the hardware op). A profiled run
  // commits after every dispatch: read the SPLIT, never the total.
  struct ProfileRow {
    double ms = 0.0;
    int dispatches = 0;
  };
  const std::map<std::string, ProfileRow>& profile() const
  {
    return _profile;
  }
  void clear_profile() { _profile.clear(); }

  // The release's scene test: SSIM (its `ssim_matlab`, an 11-tap Gaussian
  // over the colour channels as a third axis) of the two frames resized
  // to 32 x 32 -- PADDED as the network sees them. The driver duplicates
  // the first frame across a pair under 0.2 rather than interpolate a
  // cut. Host-side and model-free.
  static double scene_ssim(const void* a, const void* b, bool f32, int h,
                           int w, double scale);

  // The arithmetic the convs did, credited to the FLOP/s meter as each
  // command buffer retires: the model's own convs at their own widths
  // (the transposed ones at their 2 x 2 live taps), not the padded and
  // folded forms the kernels run.
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

  // Test-only: the last interpolation's flow (float4 a padded pixel) and
  // its aux map (half [mask logit, 8 features, 3 unused] a pixel).
  const metal_compute::SharedBuffer& last_flow() const noexcept
  {
    return _flow;
  }
  const metal_compute::SharedBuffer& last_aux() const noexcept
  {
    return _aux;
  }

private:
  struct Conv {
    metal_compute::SharedBuffer w;   // [cout, 9 * cin] f16, (ky, kx, ci)
    metal_compute::SharedBuffer b;   // [cout] f16
    // The same weights as HWIO [3, 3, cin, cout], the only layout the
    // hardware op takes; built only where it runs.
    metal_compute::SharedBuffer whwio;
    int cin = 0, cout = 0;           // as the kernel runs them (padded)
  };
  struct Block {
    Conv c0a, c0b, res[8], last;
    int width = 0, in = 0, in_pad = 0;
    double scale = 1.0;              // at --scale 1
    int ks = 13;                     // the flow head's row group, 13 / 16
  };
  // Buffer sizes for one geometry, in bytes.
  struct Sizes {
    std::size_t frame_img = 0, frame_feat = 0;
    std::size_t imgh = 0, head_a = 0, head_b = 0, head_par = 0;
    std::size_t flow = 0, aux = 0, pack = 0, c0a = 0, res = 0, last = 0;
    std::size_t stage_in = 0, stage_out = 0;
    std::size_t scratch() const;
  };
  static Sizes sizes_for_(const Info& info, int h, int w, double scale,
                          bool f32, bool hw);

  MetalRife() = default;
  bool build_(std::string* err);
  bool ensure_scratch_(int h, int w, double scale, bool f32);
  void conv_(metal_compute::ComputeEncoder& enc, const Conv& c,
             const metal_compute::SharedBuffer& in,
             const metal_compute::SharedBuffer& out, int H, int W,
             int stride, int epi, const char* cat,
             const metal_compute::SharedBuffer* res = nullptr);
  // Every dispatch goes through here, so VPIPE_RIFE_PROFILE can time it.
  void dispatch_(metal_compute::ComputeEncoder& enc, const char* cat,
                 metal_compute::LaunchDims grid,
                 metal_compute::LaunchDims tg, bool hw = false);

  std::shared_ptr<WeightSet>   _ws;
  metal_compute::MetalCompute* _mc = nullptr;
  Info _info;
  Conv _head[4];
  std::vector<Block> _blocks;
  std::size_t _weight_bytes = 0;

  metal_compute::ComputeLibrary _lib_gemm, _lib_rife;
  // [form][bn 32, 64]: 0 plain s1, 1 s1 + lrelu, 2 s1 + residual + lrelu,
  // 3 s2 (pad 1) + lrelu.
  metal_compute::ComputeFunction _fn_conv[4][2];
  metal_compute::ComputeFunction _fn_in_u8, _fn_in_f32, _fn_d2s, _fn_pack,
      _fn_unpack, _fn_merge_u8, _fn_merge_f32;
  // The hardware op, where the GPU has matrix cores. Stride 1:
  // [epi 0 bias, 1 + LeakyReLU, 2 + residual + LeakyReLU][tile 64, 32,
  // 16][tail] -- tile 16 has the bias form alone, an entry that does not
  // exist is an invalid handle and that conv takes the gather conv.
  // Stride 2 (pad 1, + LeakyReLU): [tile 64, 32][tail]. `tail` guards a
  // grid that is not a multiple of the op's 8 x 8 tile.
  metal_compute::ComputeLibrary _lib_hw;
  metal_compute::ComputeFunction _fn_hw_s1[3][3][2], _fn_hw_s2[2][2];
  bool _use_hw = false;
  std::uint64_t _hw_dispatches = 0, _gather_dispatches = 0;
  metal_compute::CommandStream* _prof_cs = nullptr;
  std::map<std::string, ProfileRow> _profile;
  // ONE stream for the model's life: a new one is a new queue, and a
  // queue's first commit costs ~1-2 ms.
  metal_compute::CommandStream _cs;

  // Scratch for the geometry last run.
  int _sh = 0, _sw = 0;
  double _sscale = 0.0;
  bool _sf32 = false;
  metal_compute::SharedBuffer _imgh, _head_a, _head_b, _head_par, _flow,
      _aux, _pack, _c0a, _res_a, _res_b, _last, _stage_in, _stage_out;
  std::size_t _scratch_bytes = 0;
  GpuFlopsTally _flops;
};

}  // namespace genai
}  // namespace vpipe

#endif
