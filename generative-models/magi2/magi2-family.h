#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_FAMILY_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_FAMILY_H

// MAGI-2-preview as a generate-video family (registered in-tree, as
// FlashVSR's is): the family owns the whole generation -- the preview
// DiT's guided UniPC loop over its video and audio latents
// (magi2-preview.h) -- and the stage the ports. The plan's figures are
// here so the tests read the same numbers the family books.
//
// THE CLIP. 12.5 frames a second, 4 to a latent frame after the first
// (4 (T - 1) + 1 frames), on a 16-pixel grid; the soundtrack 25 audio
// latents a second -- 2 a frame. The model was trained on 10 s clips
// (T 32, 125 frames, 250 audio latents) and the release supports no
// other length; other lengths are generated, and said to be off its
// training.
//
// THE MEMORY. The preview DiT is 9.9 GB of trunk and 202.5 GB of routed
// experts (5.6 GB a MoE layer, 36 layers). The trunk is held; of the
// experts, as many MoE layers as the box has room for are held and the
// rest STREAM through two slots, each forward reading them once for the
// conditional and unconditional passes together.

#include "generative-models/magi2/metal-magi2-dit.h"
#include "generative-models/magi2/metal-magi2-refiner.h"
#include "generative-models/shared/sage-attention.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
class FlexData;
class SessionContextIntf;
namespace genai {
namespace magi2 {

inline constexpr int kGrid = 16;                 // pixels a latent cell
inline constexpr int kTrainedFrames = 125;       // 10 s at 12.5 fps
// The sideband key the conditioner states image-to-video's special
// tokens by: the conditioning beat's LAST that many rows are the
// pictures' special tokens, the rows before them the prompt's.
inline constexpr const char* kFigureRows = "magi2_figure_rows";

// THE ACCELERATION both DiTs run, settled from the stage's bag
// (accel-settings.h) and the box: the int8 tier as the engine runs it
// (magi2::effective_i8_tier), its tuner, and SageAttention where it runs
// (magi2::sage_runs; `sage.enabled` false where it does not).
struct Accel {
  int i8_tier = -1;
  bool autotune = true;
  sage::Config sage;
};
Accel accel_of(const FlexData* accel, bool matrix_cores);

// Frames rounded up to 4k + 1; latent frames of an aligned count; audio
// latents of a clip of `frames`.
int align_frames(int frames);
int latent_frames(int frames);
int audio_latents_of(int frames);

// The preview DiT's figures off its headers (memoized per directory):
// its config, the trunk as held at each int8 tier (trunk_at[tier + 1]:
// off, accurate, fast, token, token_plain) and one MoE layer's experts.
struct DitPlan {
  MetalMagi2Dit::Config cfg;
  std::size_t trunk_at[5] = {0, 0, 0, 0, 0};
  std::size_t expert_layer = 0;
  int moe_layers = 0;
  std::size_t trunk(int tier = -1) const
  {
    return trunk_at[tier < -1 ? 0 : tier > 3 ? 4 : tier + 1];
  }
};
bool dit_plan(const std::string& preview_dir, DitPlan* out);
// ...for a RANK's share (tensor parallel; tensor-parallel.h): its config
// carries the share, its trunk and one MoE layer's experts are the
// share's (MetalMagi2Dit::Share). The whole model's where `share` is
// inactive. False, with `err`, for a share the model cannot take.
bool dit_plan(const std::string& preview_dir, const TpShare& share,
              DitPlan* out, std::string* err = nullptr);

// The DiT's least holding (the trunk and two expert slots) and its whole,
// the trunk at `tier`.
std::size_t dit_floor_bytes(const DitPlan& p, int tier = -1);
std::size_t dit_preload_bytes(const DitPlan& p, int tier = -1);

// A generation's tokens and the forward's scratch for its batch of two,
// the prompt booked at its longest (the conditioner's 7000-token cap)
// and one picture of the clip's size -- with `a`'s int8 and Sage scratch.
std::size_t denoise_scratch_bytes(const DitPlan& p, int width, int height,
                                  int frames, const Accel& a = {});

// MoE layers to hold: what `room` takes at a layer's experts, the slots
// dropped when every layer fits. -1 = all.
int resident_moe_for(const DitPlan& p, std::size_t room);

// THE RETAINED EXPERTS' ROOM, the `room` above, at load: `ram` less the
// streaming headroom (model_memory::kStreamHeadroom), the trunk at `tier`,
// and what the rest of the run holds beside the experts at its WIDEST --
// the held experts outlive every phase of a clip (the conditioning, the
// preview, the refine with its refiner, the decodes). That is each
// phase's plan figure (`phase_peaks`, GenerativeModelManager::phase_peak:
// every claim at its floor, its scratch in) less this DiT's own floor,
// which the plan counts in every phase; and never less than `own_beside`
// -- this request's forward, the preview's or the refine's whichever is
// larger, with the refiner where it refines -- for a plan that lacks it.
std::size_t retention_room(
    const DitPlan& p, int tier, std::size_t ram,
    const std::vector<std::pair<std::string, std::size_t>>& phase_peaks,
    std::size_t own_beside);

// THE REFINE (the release's 1080p preset: a 896 x 512 preview refined to
// 1920 x 1088), asked for by the model config's `refine_width` and
// `refine_height` (pixels, both > 0); `refine_steps`, `refine_shift`,
// `refine_guidance` over the release's 5, 5 and 2. The grid is the
// pixels floored to whole cells, as the release floors them.
struct RefineAsk {
  int width = 0, height = 0;     // pixels as asked
  int steps = 0;
  double shift = 0.0, guidance = 0.0;
  bool on() const { return width >= kGrid && height >= kGrid; }
  int grid_h() const { return height / kGrid; }
  int grid_w() const { return width / kGrid; }
};
RefineAsk refine_ask(const FlexData* model_config);

// The refined clip of a preview of `frames` pixel frames: its pixel
// frames (2 T - 1 latents, at 25 fps).
int refined_frames(int frames);

struct RefinerPlan {
  MetalMagi2Refiner::Config cfg;
  // What a load holds at each int8 tier (weights_at[tier + 1]).
  std::size_t weights_at[5] = {0, 0, 0, 0, 0};
  std::size_t weights(int tier = -1) const
  {
    return weights_at[tier < -1 ? 0 : tier > 3 ? 4 : tier + 1];
  }
};
// Off the refiner's headers, memoized per directory.
bool refiner_plan(const std::string& refiner_dir, RefinerPlan* out);
// ...for a RANK: the refiner's share at the fractions of `preview_share`
// (the preview DiT's, which a stage splits by: MetalMagi2Refiner::
// share_at), its config carrying it and its weights the share's. The
// whole refiner's where the share is inactive.
bool refiner_plan(const std::string& refiner_dir,
                  const TpShare& preview_share, RefinerPlan* out);
// What a refine of `frames` preview frames to `ask` allocates: the
// refiner's forward (with `a`'s int8 and Sage scratch) and the loop's
// host state.
std::size_t refine_scratch_bytes(const RefinerPlan& p, const RefineAsk& ask,
                                 int frames, const Accel& a = {});

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_FAMILY_H
