#ifndef VPIPE_GENERATIVE_MODELS_VIDEO_HOST_EXT_H
#define VPIPE_GENERATIVE_MODELS_VIDEO_HOST_EXT_H

// HOST-ONLY extensions of a VideoModelFamily (not installed; an in-tree
// family answers them through query_extension, and a plugin never sees
// them). Asked by generate-video's planning, which otherwise sizes a
// family from hooks that see neither the graph's model config nor the
// clip a request actually produces.

#include "generative-models/tensor-parallel.h"
#include "pipeline/resource-plan.h"
#include "pipeline/memory-plan.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class FlexData;

namespace genai {

class VideoGenerator;
struct VideoModelCreateArgs;

// "vpipe.video-config-plan/1": a family whose WEIGHTS, SCRATCH or OUTPUT
// CLIP depend on its model config or its acceleration -- MAGI-2's
// refiner, a second checkpoint that turns the preview's clip into one at
// twice the rate on a larger grid, and its int8 tiers, which hold the
// projections in another layout. `model_config` is the family's
// model-config beat when the graph wired it as a constant, null otherwise
// (the family's defaults).
inline constexpr std::string_view kVideoConfigPlanExt =
    "vpipe.video-config-plan/1";

// What the stage knows beside the root when it plans: the model config,
// its acceleration bag (accel-settings.h: a tier that holds weights in
// another layout or adds scratch changes the figures) and whether this
// box's GPU has matrix cores (the int8 and SageAttention tiers' kernels).
struct VideoPlanContext {
  const FlexData* model_config = nullptr;
  const FlexData* accel = nullptr;
  bool matrix_cores = false;
  // A RANK's share (generate-video's `tensor_parallel`; VideoTensorParallel
  // below): the figures are the share's. Null: the whole model.
  const TpShare* tp = nullptr;
};

class VideoConfigPlan {
 public:
  virtual ~VideoConfigPlan() = default;

  // In place of VideoModelFamily::declare_resources / declare_holdings /
  // denoise_scratch_bytes.
  virtual std::vector<ResourceClaim>
  config_resources(const std::string& root,
                   const VideoPlanContext& ctx) const = 0;
  virtual std::vector<StageHolding>
  config_holdings(const std::string& root,
                  const VideoPlanContext& ctx) const = 0;
  virtual std::size_t
  config_denoise_scratch_bytes(const std::string& root, int width,
                               int height, int frames,
                               const VideoPlanContext& ctx) const = 0;

  // The clip a request of (width, height, frames) produces -- pixels,
  // pixel frames and its rate. False when it is the request's own.
  virtual bool output_clip(const std::string& root,
                           const FlexData* model_config, int width,
                           int height, int frames, int* out_width,
                           int* out_height, int* out_frames,
                           double* out_fps) const = 0;

  // What the VAE holds decoding that clip beside the clip itself (the
  // decode arena's model term). 0: cannot say.
  virtual std::size_t decode_working_bytes(const std::string& root,
                                           int width, int height,
                                           int frames) const = 0;
};

// "vpipe.video-held-by-source/1", asked of a VideoGenerator through its
// query_extension: what it holds of EACH checkpoint its family declared a
// holding for. resident_bytes() is the generator's total, which corrects
// one holding and cannot be apportioned among several (MAGI-2's preview
// DiT and its refiner); a generator answering this has every holding
// corrected, in the report ledger and in the manager's declarations, from
// its own figures.
inline constexpr std::string_view kVideoHeldBySourceExt =
    "vpipe.video-held-by-source/1";

class VideoHeldBySource {
 public:
  virtual ~VideoHeldBySource() = default;
  // Bytes held of `source` (a StageHolding::source the family declared):
  // weights, held experts and slots -- what a WeightSet cannot see
  // included. 0 when nothing of it is loaded yet: the declared figure
  // stands.
  virtual std::size_t held_bytes(const std::string& source) const = 0;
};

// "vpipe.video-tensor-parallel/1", asked of a VideoModelFamily: a family
// whose DiT SPLITS ACROSS HOSTS (tensor-parallel.h) as MiniMax H3's does,
// so generate-video's `tensor_parallel` runs it -- MAGI-2's preview DiT,
// its heads and its experts. A family that does not answer is refused by
// name when the stage is given a share.
inline constexpr std::string_view kVideoTensorParallelExt =
    "vpipe.video-tensor-parallel/1";

class VideoTensorParallel {
 public:
  virtual ~VideoTensorParallel() = default;
  // The widths a share of `root`'s DiT counts: TpShare::from_flex's
  // n_heads and ffn (an even split of each where the stage's bag gives
  // none). False, with `err`, where this checkpoint cannot be split.
  virtual bool tp_widths(const std::string& root, int* n_heads, int* n_ffn,
                         std::string* err) const = 0;
  // A RANK's generator: the share's weights alone, booked as the plan
  // booked them (VideoPlanContext::tp). Null on failure, said.
  virtual std::unique_ptr<VideoGenerator>
  load_rank(const VideoModelCreateArgs& args, const TpShare& share) = 0;
};

// "vpipe.video-tp-clip/1", asked of a rank's VideoGenerator: the clip's
// CHANNEL to the other ranks, opened by the stage before generate() and
// let go after it, however it ends -- one channel a clip on every rank,
// so the ranks stay paired clip for clip. `stop` is the run's, which the
// channel polls while it waits on a peer.
inline constexpr std::string_view kVideoTpClipExt = "vpipe.video-tp-clip/1";

class VideoTpClip {
 public:
  virtual ~VideoTpClip() = default;
  virtual bool begin_tp_generation(std::shared_ptr<TpChannel> ch,
                                   std::function<bool()> stop,
                                   std::string* err) = 0;
  virtual void end_tp_generation() = 0;
};

}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_VIDEO_HOST_EXT_H
