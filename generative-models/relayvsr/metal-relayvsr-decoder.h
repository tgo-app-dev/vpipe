#ifndef VPIPE_GENERATIVE_MODELS_RELAYVSR_METAL_RELAYVSR_DECODER_H
#define VPIPE_GENERATIVE_MODELS_RELAYVSR_METAL_RELAYVSR_DECODER_H

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
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

// RelayVSR's SMALL model: the per-frame conditional decoder
// (kopper/RelayVSR, `flash.safetensors`, format refvsr-flash-v1; the
// reference's FlashVSRConditionNet, variant S, EMA weights) -- and its
// native 2x sibling (lightx2v/RelayVSR_X2, relayvsr-flash-x2-v1; ComfyUI
// ships both as its "flash VAE").
//
// Every low-resolution frame becomes its 4x (or 2x) frame here, directly
// in RGB: no VAE anywhere in the model. The frame is encoded to tokens on
// its LATENT grid -- 1/4 of it at 4x, 1/8 at 2x, 1/16 of the output either
// way -- (3x3 s2, 3x3 s2, 3x3 s1 convs -- s2 at 2x -- with SiLU, a 1x1 to
// 512, RMSNorm);
// 10 blocks at 512 (8 query heads over 2 key/value heads, head dim 64,
// SwiGLU 1408) attend
//
//   * the two ENDPOINT keyframes -- the keyframe DiT's latents, on the
//     same 1/4 grid -- each PRIMED once into every block's K and V by
//     running it through the stack alone (prepare()),
//   * the previous frame of the window, and the frame itself,
//
// every plane through a query-centred, half-open window truncated at the
// borders: 48 x 48 tokens for the keyframes, 24 x 24 for the video. A
// linear head (512 -> 1024 + SiLU -> 768) is pixel-shuffled by 16 onto a
// 4x (2x) bicubic upsample of the frame. RoPE is 3-D (t 16 | h 24 | w 24) with
// the current frame at t = 0, the previous at -1 and both keyframes at 0
// ("current_relative").
//
// PRECISION. The reference runs bf16; this runs f16, whose range the
// network stays well inside (the residual stream peaks below 20 on real
// frames), and lands closer to the fp32 truth than the reference's own
// bf16 does.
class MetalRelayVsrDecoder {
 public:
  struct Config {
    int dim       = 512;
    int depth     = 10;
    int heads     = 8;
    int kv_heads  = 2;
    int head_dim  = 64;
    int ffn       = 1408;     // ceil(dim * 8/3 / 64) * 64
    int enc[3]    = {64, 128, 128};
    int cond_channels = 48;
    int head_width = 1024;    // spatial_head_dim
    int win_video[2] = {24, 24};
    int win_start[2] = {48, 48};
    int win_end[2]   = {48, 48};
    int win_self[2]  = {48, 48};
    int history   = 2;        // window_size: the frame and one before it
    float eps     = 1e-5f;
    int upscale   = 4;        // upscale_factor: 4, or 2 (the x2 file)
    // The latent grid's stride in frame pixels, and the alignment a frame
    // is padded to (the keyframe DiT's token stride).
    int latent_stride() const noexcept { return 16 / upscale; }
    int pad_stride() const noexcept { return 32 / upscale; }

    // ACCELERATION, the stage's to set. The int8 tier of the 10 blocks'
    // projections (q, k, v, the keyframes' k and v, o, the feed-forward's
    // two), as relayvsr::effective_i8_tier answers it -- -1 off. A
    // projection whose depth a holding tier's layout cannot take (the
    // feed-forward's 1408-deep down on fast and token) runs the accurate
    // route. The frame encoder and the head stay f16: they read the frame
    // and write the pixels. load() records the tier it runs.
    int   i8_tier  = -1;
    bool  autotune = true;
    // A frame's feed-forward split by rows with the Neural Engine (one
    // SwiGLU module in fp16, the decoder's own type, staged per block);
    // the keyframes' priming keeps the GPU. A block whose feed-forward an
    // int8 tier holds keeps the GPU too.
    bool  ane_ffn  = false;
    float ane_rows = 0.0f;
    // SageAttention on the matrix cores: the windowed attention's QK in
    // int8 (frames and the keyframes' priming), with `sage_pv8` its P*V.
    bool  sage_attn = false;
    bool  sage_pv8  = false;
    int sage_mode() const noexcept
    {
      return sage_attn ? (sage_pv8 ? 2 : 1) : 0;
    }
  };

  // The architecture off the file's metadata, checked against what this
  // port implements (endpoint conditions, current-relative RoPE, a 4x
  // decoder, query-centred truncated windows); anything else is refused
  // by name.
  static bool config_from_checkpoint(const std::string& flash_file,
                                     Config* out, std::string* err);

  static std::unique_ptr<MetalRelayVsrDecoder>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err);

  ~MetalRelayVsrDecoder();

  // A keyframe PRIMED for the decoder: every block's K (rotated at t = 0)
  // and V over its tokens, f16 [depth][n, kv_heads * head_dim] each.
  struct Condition {
    metal_compute::SharedBuffer k, v;
    int h = 0, w = 0;   // its token grid
    bool valid() const noexcept { return !k.empty() && !v.empty(); }
  };
  // `latent` is the keyframe DiT's, f32 [cond_channels, h4, w4] -- its
  // latent grid (the frame / latent_stride).
  bool prepare(const metal_compute::SharedBuffer& latent, int h4, int w4,
               Condition* out, std::string* err);

  // A new window between two keyframes: no video history.
  void begin_window() noexcept { _in_window = 0; }

  // ONE FRAME. `lr` is the padded frame, channel-last [ph * pw, 3] f16 in
  // [-1, 1] (Ops::upload in the f16 library), ph and pw multiples of
  // pad_stride(), and `start` / `end` the window's two keyframes at its
  // latent grid. Writes the frame cut to out_h x out_w (at most s ph x
  // s pw, s the scale), planar [3, out_h,
  // out_w] -- u8 as the release writes it, or f32 in [0, 1] -- into `out`
  // (allocated when empty or too small). The frame's own K and V become the
  // next frame's history.
  bool decode(const metal_compute::SharedBuffer& lr, int ph, int pw,
              const Condition& start, const Condition& end, int out_h,
              int out_w, bool f32, metal_compute::SharedBuffer* out,
              std::string* err);

  // What one primed keyframe holds at this padded frame size.
  static std::uint64_t condition_bytes(const Config& cfg, int ph, int pw);
  // A frame's working set and the history the decoder keeps beside it.
  static std::uint64_t scratch_bytes(const Config& cfg, int ph, int pw);
  // The weights as held (f16, the projections at cfg.i8_tier's layout).
  static std::uint64_t weight_bytes(const Config& cfg);
  // ...and what THIS load holds: each projection at the layout it took.
  std::uint64_t held_bytes() const;
  // Whether the ANE tier can take a feed-forward at cfg.i8_tier: not where
  // the tier holds its gate|up int8 (fast and up).
  static bool ane_stages(const Config& cfg);
  // What the ANE tier holds at the largest chunk it builds (a CoreML
  // claim's unit), and what the module built holds (0 without one).
  static std::uint64_t ane_runtime_bytes(const Config& cfg);
  std::uint64_t ane_held_bytes() const;

  const Config& config() const noexcept { return _cfg; }
  bool uses_matrix_cores() const noexcept;
  // Whether its windowed attention runs on the matrix cores.
  bool window_on_matrix_cores() const noexcept;
  // What the last decode() ran: the blocks' GEMMs on the int8 route and
  // densely, and the feed-forward rows the Neural Engine took.
  long i8_gemms() const noexcept { return _i8_gemms; }
  long dense_gemms() const noexcept { return _dense_gemms; }
  long ane_rows() const noexcept { return _ane_rows; }
  // Whether the ANE tier was tried (at the first decode with it asked) and
  // whether its module was built: what the CoreML booking follows.
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  // TEST SEAM: its kernels and their scratch (the int8 prologue's).
  const relayvsr::Ops& ops() const noexcept;

 private:
  MetalRelayVsrDecoder() = default;
  struct Block;
  struct Scratch;
  bool ensure_scratch_(int ph, int pw, std::string* err);
  // The RoPE tables of a geometry (see rvsr_rope3d).
  void build_rope_(int h, int w);
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
  int _in_window = 0;   // frames decoded since begin_window()
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  long _i8_gemms = 0, _dense_gemms = 0, _ane_rows = 0;

  metal_compute::SharedBuffer _e0w, _e0b, _e1w, _e1b, _e2w, _e2b, _e3w,
      _e3b, _enorm, _vtype, _cpw, _cpb, _cnorm, _ctype, _hw, _hb, _rw, _rb;
};

}  // namespace genai
}  // namespace vpipe

#endif
