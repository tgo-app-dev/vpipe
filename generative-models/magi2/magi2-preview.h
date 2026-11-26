#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_PREVIEW_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_PREVIEW_H

// MAGI-2-preview's base generation (the release's Magi2PreviewSampler.
// sample, read, not run): the video and audio latents drawn from noise --
// [48, T, h, w] then [L, 64], one generator -- and denoised together by
// the preview DiT under classifier-free guidance, each step ONE batched
// forward of the conditional and unconditional passes (MetalMagi2Dit::
// forward over a batch: a streamed layer's experts read once for both),
// the two velocities combined as uncond + g (cond - uncond) per modality
// (video 5, audio 7) and stepped by a FlowSampler each (UniPC over
// magi2::sigma_grid). Every video and audio token is told the step's
// timestep; the text and the images ride at 0.
//
// The unconditional pass takes the NEGATIVE prompt's condition in place
// of the prompt's and keeps everything else, image-to-video's picture
// and its special token included (the release's use_ref_for_uncond,
// which its configs turn on: guidance then steers the text alone).
//
// The noise is torch's CPU generator at the seed (TorchCpuRandn); the
// release draws on CUDA, whose stream is not reproducible here, so a
// seed does not give the release's clip.

#include "generative-models/magi2/magi2-schedule.h"
#include "generative-models/magi2/metal-magi2-dit.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace magi2 {

struct PreviewRequest {
  int T = 0, h = 0, w = 0;       // latent frames and grid
  int audio_len = 0;             // audio latents
  int steps = kPreviewSteps;
  double shift = kPreviewShift;
  double video_guidance = kVideoGuidance;
  double audio_guidance = kAudioGuidance;
  std::uint64_t seed = 0;
  // The conditioner's rows, f32 [rows, text_c]; the negative may be
  // absent, which with guidance > 1 is refused.
  const float* cond = nullptr;
  int cond_rows = 0;
  const float* neg = nullptr;
  int neg_rows = 0;
  // Image-to-video: each picture's latent and special token, on both
  // passes.
  std::vector<MetalMagi2Dit::Image> images;
  // Called after each step with (step, steps), one-based; false stops
  // the run (returns false with an empty error).
  std::function<bool(int step, int total)> progress;
  // TESTS: the initial latents given instead of drawn ([48, T, h, w],
  // [L, 64]).
  const float* video_noise = nullptr;
  const float* audio_noise = nullptr;
};

// Denoises; `video` [48, T, h, w] and `audio` [L, 64] f32 on success.
bool run_preview(MetalMagi2Dit& dit, const PreviewRequest& r,
                 std::vector<float>* video, std::vector<float>* audio,
                 std::string* err);

// The same loop over any velocity: `forward` is MetalMagi2Dit::forward's
// batch shape. What the tests drive with a toy field, so the loop's
// plumbing is held exactly apart from the DiT's arithmetic.
using BatchForward = std::function<bool(
    const std::vector<const MetalMagi2Dit::Sample*>& batch,
    std::vector<std::vector<float>>* vel_video,
    std::vector<std::vector<float>>* vel_audio, std::string* err)>;
bool run_preview(const BatchForward& forward, int video_c, int audio_c,
                 const PreviewRequest& r, std::vector<float>* video,
                 std::vector<float>* audio, std::string* err);

// The CFG combination as the release computes it in float32: uncond +
// g * (cond - uncond), no fused multiply-add. In place on `cond`.
void guide(std::vector<float>& cond, const std::vector<float>& uncond,
           float g);

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_PREVIEW_H
