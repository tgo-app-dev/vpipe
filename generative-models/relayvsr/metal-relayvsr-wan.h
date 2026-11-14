#ifndef VPIPE_GENERATIVE_MODELS_RELAYVSR_METAL_RELAYVSR_WAN_H
#define VPIPE_GENERATIVE_MODELS_RELAYVSR_METAL_RELAYVSR_WAN_H

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class AneFeedForward;   // generative-models/shared/ane-ffn.h
class WeightSet;   // generative-models/weight-set.h

namespace relayvsr {
class Ops;
}

// RelayVSR's LARGE model: the keyframe generator (kopper/RelayVSR,
// `wan.safetensors`, format refvsr-wan-v1; Apache-2.0).
//
// A Wan2.2-TI2V-5B-shaped DiT -- 30 blocks at 3072 (24 heads x 128), an
// ungated gelu-tanh 14336 feed-forward, patch (1, 2, 2), 48 latent
// channels -- retrained into a ONE-STEP generator of the keyframe's latent
// on the 1/16 grid of the 4x output: t = 1000, latent = noise - v. The
// low-resolution frame enters through `lq_proj` (its 4x bilinear upsample,
// pixel-unshuffled by 16, a 1x1 conv, RMSNorm2d, SiLU, a 3x3 stride-2
// conv, RMSNorm2d, SiLU, a linear) ADDED to the noise's patch embedding.
// The text context is a constant that ships in the file (`prompt_embeds`
// [1, 512, 4096], the reference's precomputed T5 encoding), and the
// timestep is one constant, so the load FOLDS the timestep embedding,
// every block's modulation and every block's cross-attention K and V, and
// the forward never touches the text again.
//
// STREAMING, one keyframe per call. Every block keeps the last
// `context_frames` (4) keyframes' self-attention K (normed, UNROTATED) and
// V in a ring, and RoPE places them as the reference's `fixed_end` policy
// does: the query at f - origin, a kept key at its own keyframe - origin,
// origin = max(0, f - rope_frames + 1) -- so past 8 keyframes every kept
// key is re-rotated at a position one lower each call. Self-attention
// reads each query's 22 x 40 token window, shifted inward at the borders,
// in every kept keyframe.
//
// The reference runs bf16; so does this. What differs from it is in the
// rounding: the modulation is kept in f32 where the reference keeps it
// there, RoPE is applied in f32 from f32 tables and rounded once.
class MetalRelayVsrWan {
 public:
  struct Config {
    int hidden      = 3072;
    int heads       = 24;
    int head_dim    = 128;
    int ffn         = 14336;
    int layers      = 30;
    int in_channels = 48;
    int text_dim    = 4096;
    int text_tokens = 512;
    int freq_dim    = 256;
    int lq_hidden1  = 768;
    int lq_hidden2  = 3072;
    int window_h    = 22;    // spatial_window_size, in tokens
    int window_w    = 40;
    int context_frames = 4;  // temporal_context_frames
    int rope_frames    = 8;  // temporal_rope.training_frames
    float eps       = 1e-6f;
    float timestep  = 1000.0f;

    // ACCELERATION, the stage's to set (none is in the file).
    //
    // The int8 tier of the 30 blocks' eight projections (q, k, v, o, the
    // cross-attention's q and o, the feed-forward's two), as
    // relayvsr::effective_i8_tier answers it: -1 off, 0 accurate, 1 fast,
    // 2 token, 3 token_plain. From fast up the projections are HELD in the
    // int8 layout -- half their bf16 bytes. The source rows, the patch
    // embedding and the head stay bf16. load() records the tier it runs.
    int   i8_tier   = -1;
    bool  autotune  = true;     // the accurate tier's split-K tuner
    // The feed-forward split by rows between the GPU and the Neural Engine
    // (ane-ffn.h: one GELU module with biases for every block, staged per
    // block under its attention). A block whose feed-forward an int8 tier
    // holds keeps the GPU. `ane_rows` is the ANE's share, <= 0 balanced
    // from the two engines' measured times.
    bool  ane_ffn   = false;
    float ane_rows  = 0.0f;
    // SageAttention on the matrix cores (relayvsr::Ops::window_attention
    // and cross_attention): the self- and cross-attention's QK in int8,
    // and with `sage_pv8` their P*V too. The stage sets them only where
    // there are matrix cores.
    bool  sage_attn = false;
    bool  sage_pv8  = false;
    // The int8 mode window_attention takes: 0, 1 (QK) or 2 (QK and P*V).
    int sage_mode() const noexcept
    {
      return sage_attn ? (sage_pv8 ? 2 : 1) : 0;
    }
  };

  // The architecture off the file's metadata (`config`), checked against
  // what this port implements -- a refvsr-wan file whose config says
  // anything else is refused by name rather than run.
  static bool config_from_checkpoint(const std::string& wan_file,
                                     Config* out, std::string* err);

  static std::unique_ptr<MetalRelayVsrWan>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err);

  ~MetalRelayVsrWan();

  // A new stream: the ring forgets every keyframe.
  void reset();
  // Keyframes generated in the current stream.
  int keyframes() const noexcept { return _frame; }

  // ONE KEYFRAME. `lr` is the padded frame, channel-last [ph * pw, 3] bf16
  // in [-1, 1] (Ops::upload in the bf16 library); ph and pw multiples of 8.
  // `noise` is f32 [48, ph/4, pw/4]. Writes the latent, f32 [48, ph/4,
  // pw/4], into `latent` (allocated when empty or too small). `stop`, when
  // set, is asked between blocks; true abandons the keyframe (false
  // returned with no error).
  bool keyframe(const metal_compute::SharedBuffer& lr, int ph, int pw,
                const std::vector<float>& noise,
                metal_compute::SharedBuffer* latent, std::string* err,
                const std::function<bool()>& stop = {});

  // What a stream of this geometry holds: every block's ring of K and V.
  static std::uint64_t ring_bytes(const Config& cfg, int ph, int pw);
  // ...and one keyframe's working set beside it.
  static std::uint64_t scratch_bytes(const Config& cfg, int ph, int pw);
  // What a load holds of the checkpoint, off the tensor headers: every
  // block's weights but the cross-attention K/V projections the load folds
  // away -- the eight projections at cfg.i8_tier's layout -- plus what it
  // folds (the context's K and V, the modulation).
  static std::uint64_t weight_bytes(const Config& cfg);
  // ...and what THIS load holds: weight_bytes' figure with each projection
  // at the layout it took -- the same wherever the int8 context ran the
  // tier the rule predicts, more where it declined one.
  std::uint64_t held_bytes() const;
  // Whether the ANE tier can take a feed-forward at cfg.i8_tier: not where
  // the tier holds it int8 (fast and up), which the staging cannot read.
  static bool ane_stages(const Config& cfg);
  // What the ANE tier holds (CoreML's module and its slots, one block's
  // feed-forward in fp16) at the largest chunk it builds -- a CoreML
  // claim's unit -- and what the module built holds (0 without one).
  static std::uint64_t ane_runtime_bytes(const Config& cfg);
  std::uint64_t ane_held_bytes() const;

  const Config& config() const noexcept { return _cfg; }
  bool uses_matrix_cores() const noexcept;
  // Whether its windowed attention, and its cross-attention, run on the
  // matrix cores.
  bool window_on_matrix_cores() const noexcept;
  bool cross_on_matrix_cores() const noexcept;
  // What the last keyframe ran: GEMMs on the int8 route and densely, and
  // the feed-forward rows the Neural Engine took (summed over its blocks).
  long i8_gemms() const noexcept { return _i8_gemms; }
  long dense_gemms() const noexcept { return _dense_gemms; }
  long ane_rows() const noexcept { return _ane_rows; }
  // Whether the ANE tier was tried (at the first keyframe with it asked) and
  // whether its module was built: what the CoreML booking follows.
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  // TEST SEAM: its kernels and their scratch (the int8 prologue's).
  const relayvsr::Ops& ops() const noexcept;

  // TEST SEAMS, from the last keyframe(): its source rows [n, 3072] and its
  // residual stream after block 0, as f32.
  void keep_debug(bool on) noexcept { _keep_debug = on; }
  const std::vector<float>& debug_lq() const noexcept { return _dbg_lq; }
  const std::vector<float>& debug_block0() const noexcept
  {
    return _dbg_blk0;
  }

 private:
  MetalRelayVsrWan() = default;
  struct Block;
  struct Scratch;
  bool fold_(std::string* err);
  bool ensure_scratch_(int ph, int pw, std::string* err);
  // The ANE tier, built at the first keyframe (its rows fix the chunk).
  bool ane_setup_(int rows);
  bool ane_eligible_(int block) const;
  void ane_stage_(int block);

  Config _cfg{};
  metal_compute::MetalCompute* _mc = nullptr;
  std::shared_ptr<WeightSet> _ws;
  std::unique_ptr<relayvsr::Ops> _ops;
  std::vector<Block> _blocks;
  std::unique_ptr<Scratch> _sc;
  metal_compute::CommandStream _stream;
  bool _have_stream = false;

  // lq_proj, the patch embedding and the head.
  metal_compute::SharedBuffer _lq_c1w, _lq_c1b, _lq_n1, _lq_c2w, _lq_c2b,
      _lq_n2, _lq_ow, _lq_ob, _patch_w, _patch_b, _head_w, _head_b;
  // The head's shift and (1 + scale) - 1, f32 [hidden] each.
  metal_compute::SharedBuffer _head_shift, _head_scale;
  // RoPE tables, f32 [1024, head_dim].
  metal_compute::SharedBuffer _rope_cos, _rope_sin;

  int _frame = 0;   // keyframes in this stream
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  long _i8_gemms = 0, _dense_gemms = 0, _ane_rows = 0;
  bool _keep_debug = false;
  std::vector<float> _dbg_lq, _dbg_blk0;
};

}  // namespace genai
}  // namespace vpipe

#endif
