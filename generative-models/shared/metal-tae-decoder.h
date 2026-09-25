#ifndef VPIPE_GENERATIVE_MODELS_SHARED_METAL_TAE_DECODER_H
#define VPIPE_GENERATIVE_MODELS_SHARED_METAL_TAE_DECODER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // generative-models/llama3/metal-llama-weights.h
class WeightSet;           // generative-models/weight-set.h

// A TINY AUTOENCODER decoder, run on metal-compute as a cheap LATENT
// PREVIEW: the model's clean estimate at some denoise step, turned into
// pixels in a small fraction of what the real VAE costs.
//
// WHICH CHECKPOINTS. madebyollin's TAEHV family (taeh3, taew2_1,
// taew2_2, taehv, taehv1_5, taeltx_2, ...) and the 2D decoders built from
// the same blocks (Kijai's MiniMax-H3 TAE; TAESD's decoder has the same
// shape). None of them ships a config, so the network is READ OFF THE
// TENSOR NAMES: `decoder.<i>` in a TAEHV file, a bare `<i>` in Kijai's,
// `decoder.layers.<i>` in diffusers' AutoencoderTiny (taef1, taef2 --
// the same Sequential with its Clamp lifted into forward(), so every
// index is one lower), and at each index one of
//
//   <i>.weight [o, c, 3, 3] (+ .bias)   3x3 conv, padding 1
//   <i>.conv.{0,2,4}.weight (+ .skip)   a Block, or a MemBlock when the
//                                       first conv reads TWICE the
//                                       channels: [x, past]
//   <i>.conv.weight [s*c, c, 1, 1]      TGrow: 1x1 conv, then the channel
//                                       blocks become s frames in time
//   (nothing)                           index 0 is the input Clamp
//                                       tanh(x/3)*3; after a plain conv a
//                                       ReLU; after a block a nearest 2x
//                                       Upsample
//
// That last rule is the one the KJNodes loader states, and it holds for
// every published file. What the rule cannot read is a Super decoder
// (`*_super`), whose blocks are depthwise 7x7; those are REFUSED at
// load, by shape, rather than half-run.
//
// THE LATENT SPACE is the diffusion model's own: a TAE decodes exactly
// what the DiT produces, with no shift or scale. For MiniMax-H3 that is
// the WHITENED space -- (z - latents_mean) / latents_std -- so the DiT's
// output goes straight in, and un-whitening it first (which the real VAE
// decode does) would be wrong.
//
// TIME. A MemBlock sees its own previous INPUT (zeros before the first
// frame), counted at its own rate, and TGrow turns one frame into
// `stride` sub-frames. Decoding therefore runs latent frame by latent
// frame, depth first, with one remembered frame per MemBlock -- O(1)
// state, so a long clip costs time and not memory. The raw decoder emits
// `time_upscale` frames per latent frame; which of them are real is the
// FAMILY's temporal rule, not the decoder's, hence Trim.
//
// Runs in f16. All the kernels are existing ones (the on-chip-gather
// 3x3 conv, the simdgroup GEMM for the 1x1s, the VAE elementwise ops),
// so no kernel is specific to this class.
class MetalTaeDecoder {
public:
  // How the raw frames map onto the clip the real VAE would decode.
  enum class Trim {
    // Drop the first `time_upscale - 1` raw frames: the encoder padded
    // the clip's front so the first latent frame covers one real frame.
    // Every TAEHV but H3. A 2D decoder (time_upscale 1) drops nothing.
    kLeading,
    // MiniMax-H3's chunked rule. Its VAE encodes every 17-frame clip on
    // its own, with 3 zero frames in front, so EACH group of five latent
    // frames opens with one that covers a single real frame. Raw frame j
    // (of 4 per latent frame) is real iff j % 20 >= 3: 5n + 2 latent
    // frames give 17n + 5 frames, exactly what the real VAE decodes.
    kH3Chunks,
  };

  struct Info {
    int latent_channels = 0;
    int patch           = 1;  // pixel-shuffle factor of the head
    int spatial         = 1;  // output pixels per latent cell, per axis
    int time_upscale    = 1;  // raw frames per latent frame
    int image_channels  = 3;  // 3, or 4 for an RGBA TAE (taeqi2_1)
    bool temporal       = false;  // has MemBlocks: a video decoder
    std::size_t params  = 0;
    // "taehv" or "2d" -- what the layout read as, for the load line.
    std::string layout;
  };

  // Build from a checkpoint. The WeightSet is KEPT for the decoder's
  // lifetime (the ownership contract); the tensors are read uncached and
  // converted into the decoder's own f16 layout. Null + *err on a file
  // whose names do not describe a decoder this class can run.
  static std::unique_ptr<MetalTaeDecoder>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       std::string* err);

  ~MetalTaeDecoder();

  const Info& info() const noexcept { return _info; }

  // Frames decode() makes out of `latent_frames` under `trim`, before any
  // `max_frames` cap. The static form is the rule itself, for a decoder
  // with `time_upscale` raw frames per latent frame.
  int frames_for(int latent_frames, Trim trim) const;
  static int frames_for(int latent_frames, int time_upscale, Trim trim);

  struct Frames {
    // [frames, channels, height, width] planar U8: RGB, or RGBA with
    // STRAIGHT alpha from an RGBA TAE.
    std::vector<std::uint8_t> rgb;
    int channels = 3;
    int frames = 0;
    int height = 0;
    int width  = 0;
  };

  // Decode `latent` [C, T, h, w] f32 into planar U8 RGB.
  //
  // `max_frames` > 0 stops once that many frames exist. That is a PREFIX
  // of the clip, and it is exact: the memory chains forward only, so the
  // first frames never depend on the later ones.
  //
  // `shrink` > 1 box-filters the picture by that factor on the way out
  // (and rounds each side down to even, which a 4:2:0 encoder needs). It
  // is applied to the PIXELS, never the latent: pooling the latent 2x
  // before decoding MEASURED 20.5 dB against the decoded-then-downscaled
  // clip on a real H3 latent -- blurred, and with colour shifts the model
  // never produced -- so a smaller preview costs the full decode.
  //
  // `cancel` is polled between latent frames; returning true abandons the
  // decode, which then returns false with an EMPTY `*err`.
  bool decode(const float* latent, int C, int T, int h, int w, Trim trim,
              int max_frames, int shrink, Frames* out, std::string* err,
              const std::function<bool()>& cancel = {});

  // What the decoder holds: weights plus the scratch of the last geometry
  // it decoded.
  std::size_t resident_bytes() const noexcept;

  // What decoding `T` latent frames of `h` x `w` under `trim` would cost
  // (the trim decides how many output frames a commit holds), read off the
  // checkpoint's tensor HEADERS -- no weight is read and nothing is
  // allocated -- so a stage can book it before anything loads. The same
  // sizing the decoder allocates from, so the two cannot drift; exact on
  // the GPU side (the host's own frame buffers are the caller's).
  struct Plan {
    Info        info;
    std::size_t weight_bytes  = 0;
    std::size_t scratch_bytes = 0;
  };
  static bool plan(const std::string& file, int T, int h, int w, Trim trim,
                   Plan* out, std::string* err);

  // The last decode's GPU kernel time per op category, when it ran with
  // VPIPE_TAE_PROFILE set (empty otherwise). A profiled decode commits
  // after every dispatch, so read the SPLIT from it, never the total.
  struct ProfileRow {
    double ms = 0.0;        // GPU time (GPUStartTime..GPUEndTime)
    double wall_ms = 0.0;   // commit..wait, the cross-check
    int    dispatches = 0;
  };
  const std::map<std::string, ProfileRow>& profile() const
  {
    return _profile;
  }

private:
  struct Conv {
    // 3x3: [cout, 9 * cin] with K ordered (ky, kx, ci) -- the layout the
    // gather conv reads. 1x1: [cout, cin]. f16.
    metal_compute::SharedBuffer w;
    metal_compute::SharedBuffer b;      // [cout] f16, or empty
    // The SAME weights as HWIO [3, 3, cin, cout], which is the only
    // layout the MPP convolution2d op accepts -- so it is a twin, not a
    // relayout, and it is built whenever cout % 64 == 0 (what the op
    // takes) regardless of the machine. Unconditionally, because plan()
    // is static and has no GPU to ask: booking it on one box and not
    // the other is how the plan and the allocation come to disagree.
    // The price is a second copy of the 3x3 weights -- ~19 MB on a 20 MB
    // TAE -- which an M4 pays for nothing. Small beside the scratch a
    // preview allocates (377 MB at 960x576), and the alternative is a
    // plan that lies on one of the two machines.
    metal_compute::SharedBuffer whwio;
    // The twin holds TWO [3, 3, cin/2, cout] halves rather than one
    // whole: a MemBlock's first conv reads [x, past] and runs as two
    // hardware convs, one per half.
    bool split_hwio = false;
    int  cin  = 0;
    int  cout = 0;
    bool k3   = true;
  };
  struct Op {
    enum Kind { kConv, kRelu, kUpsample, kTGrow, kBlock, kMemBlock } kind;
    Conv c;                  // kConv; kTGrow's 1x1 [stride * C, C]
    Conv c0, c1, c2;         // blocks
    Conv skip;               // blocks, when n_in != n_out
    bool has_skip = false;
    // TAESD's mid-block pool (taef2): 1x1 up, GroupNorm, ReLU, 1x1 down,
    // added to the block's input before the block reads it.
    Conv pool_in, pool_out;
    metal_compute::SharedBuffer gn_gamma, gn_beta;   // f16 [pool_in.cout]
    bool has_pool = false;
    int  stride   = 1;       // kTGrow
    int  mem      = -1;      // kMemBlock: its memory slot
    int  in_ch    = 0;
    int  out_ch   = 0;
    int  scale    = 1;       // spatial scale of its INPUT vs the latent
    // Fusions (see the end of parse_): `fused` on a ReLU or Upsample that
    // the conv beside it performs, `relu` / `up` on that conv.
    bool fused    = false;
    bool relu     = false;
    int  up       = 1;
  };
  // A gather conv's fused epilogue.
  enum class Epi { kNone, kRelu, kAddRelu };

  MetalTaeDecoder() = default;

  // Element counts of the per-geometry scratch; see ensure_scratch_.
  struct Scratch {
    std::size_t in = 0, tmp = 1, skip = 1, head_frame = 0;
    std::size_t pool = 1, pooled = 1;              // f16 elements
    std::size_t gn_part_bytes = 0, gn_stats_bytes = 0;   // f32
    std::vector<std::size_t> op_out, mem;
    std::size_t fixed_bytes() const;   // everything but head/in_all
  };
  Scratch scratch_for_(int h, int w) const;
  // Head slots one commit holds for a clip of `want` frames: every frame
  // when they fit kHeadBudget, else as many as do (never fewer than one
  // latent frame's worth).
  int slots_per_commit_(std::size_t head_frame, int want) const;
  static constexpr std::size_t kHeadBudget = 256u << 20;
  // The pool's GroupNorm: TAESD's GroupNorm(4, n) at torch's default eps,
  // over row blocks for the stats pass (the FLUX.2 VAE's split).
  static constexpr int   kPoolGroups = 4;
  static constexpr float kGnEps      = 1e-5f;
  static constexpr int   kGnBlocks   = 512;

  // Read the network off `src`'s names and shapes. `read` false stops at
  // the shapes (plan()); true also reads and converts every weight, which
  // needs `_ws` and `_mc`.
  bool parse_(const MetalLlamaWeights& src, bool read,
              std::string* err);
  bool build_kernels_(std::string* err);
  bool ensure_scratch_(int h, int w);

  // Encode op `i` onward for one frame held in `x` at `H` x `W`. `raw` is
  // the raw frame index this path will land on once every TGrow below
  // has split it; `emit` says whether that raw frame is kept.
  void run_(metal_compute::ComputeEncoder& enc, std::size_t i,
            const metal_compute::SharedBuffer& x, int H, int W, long raw,
            const std::function<int(long)>& slot_of);

  // A 3x3 conv of an `H` x `W` input into `out` at byte `out_off`, the
  // input nearest-upsampled `up` (1 or 2) on the fly, with `epi` on the
  // accumulator (`res` the residual for kAddRelu). `cat` names it for
  // VPIPE_TAE_PROFILE.
  void conv3_(metal_compute::ComputeEncoder& enc, const Conv& c,
              const metal_compute::SharedBuffer& in,
              const metal_compute::SharedBuffer& out, std::size_t out_off,
              int H, int W, int up, Epi epi,
              const metal_compute::SharedBuffer* res, const char* cat,
              const metal_compute::SharedBuffer* in2 = nullptr);
  void conv1_(metal_compute::ComputeEncoder& enc, const Conv& c,
              std::size_t w_row0, int rows,
              const metal_compute::SharedBuffer& in,
              const metal_compute::SharedBuffer& out, int M);
  void relu_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t n);
  // `hw` splits the profile row: a conv that took the hardware op is
  // counted as "<cat>-hw". That is the ENGAGEMENT evidence -- the route
  // is a silent fallback otherwise, and a decode that quietly stayed on
  // the gather conv matches every golden and is merely slower.
  void dispatch_(metal_compute::ComputeEncoder& enc, const char* cat,
                 metal_compute::LaunchDims grid,
                 metal_compute::LaunchDims tg, bool hw = false);

  std::shared_ptr<WeightSet>    _ws;
  metal_compute::MetalCompute*  _mc = nullptr;
  std::string                   _prefix;
  bool                          _clamp_input = false;
  std::vector<Op>               _ops;
  int                           _n_mem = 0;
  // Index of the last op that carries state. A raw frame nobody keeps
  // still has to run up to here (it feeds the memory); past it, it is
  // skipped.
  std::size_t                   _last_state_op = 0;
  Info                          _info;

  // Whether this conv's geometry and width let the hardware op take it:
  // whole 8x8 destination tiles and 64-channel slices (see conv3_).
  // `whole_tiles` is the stricter rule the concat pair needs: those two
  // entries store through the device destination tensor, so an edge
  // tile would write past the image. The fused-epilogue entries write
  // the destination themselves and drop the overhang, so they take any
  // grid.
  bool hw_ok_(const Conv& c, int H, int W, int up, bool cat,
              bool whole_tiles) const;

  metal_compute::ComputeLibrary  _lib_gemm, _lib_elt, _lib_convhw;
  // [form][BN 32, 64]; form 0 plain, 1 +relu, 2 +residual+relu, 3 up2,
  // 4 up2+relu, 5 concat-input+relu -- see conv3_.
  static constexpr int kConvForms = 6;
  metal_compute::ComputeFunction _fn_conv[kConvForms][2];
  metal_compute::ComputeFunction _fn_gemm_bm64, _fn_gemm_bm64bn64, _fn_relu,
      _fn_add, _fn_upsample, _fn_copy, _fn_gn_stats,
      _fn_gn_reduce, _fn_gn_apply;
  // The MPP convolution2d op with the decoder's own epilogue fused into
  // its cooperative tensor, indexed by Epi. ~3x the gather conv on an
  // M5 at every shape it accepts, epilogue included. Off without matrix
  // cores or under VPIPE_TAE_NO_HWCONV.
  // [tail][epi]: tail 0 assumes whole 8x8 destination tiles, tail 1
  // guards the store for a grid that is not a multiple of 8. Two
  // entries and not one runtime branch -- the guard costs 1-10% when it
  // can never fire.
  metal_compute::ComputeFunction _fn_hw_epi[2][3];
  // The concat pair (see conv3_): plain, then accumulate + bias + ReLU.
  metal_compute::ComputeFunction _fn_hw_plain, _fn_hw_acc_relu;
  bool _use_hwconv = false;
  // A zero vector the hardware conv binds when a conv has no bias: its
  // epilogue reads the bias unconditionally, and a branch per element
  // to avoid one small buffer is the worse trade.
  metal_compute::SharedBuffer _zero_bias;

  // Scratch for the geometry last decoded. `_op_out[i]` is op i's output
  // for one frame; `_mem[k]` MemBlock k's remembered input; `_tmp*` the
  // blocks' internals, shared by every block (dispatches serialize);
  // `_head` one output slot per frame of a commit.
  int _sh = 0, _sw = 0;
  std::vector<metal_compute::SharedBuffer> _op_out;
  std::vector<metal_compute::SharedBuffer> _mem;
  metal_compute::SharedBuffer _in, _tmp0, _tmp1, _skip;
  // The mid-block pool's scratch, when a block has one.
  metal_compute::SharedBuffer _pool_a, _pool_b, _pooled, _gn_part,
      _gn_stats;
  // The whole clip's latent, uploaded once; frame t is copied into `_in`
  // on the GPU when its turn comes.
  metal_compute::SharedBuffer _in_all;
  metal_compute::SharedBuffer _head;
  std::size_t _head_frame = 0;   // elements per head slot
  std::size_t _scratch_bytes = 0;
  std::size_t _weight_bytes  = 0;
  // VPIPE_TAE_PROFILE: the decode's stream while it runs, else null.
  metal_compute::CommandStream* _prof_cs = nullptr;
  std::map<std::string, ProfileRow> _profile;
};

}  // namespace genai
}  // namespace vpipe

#endif
