#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_SCHEDULE_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_SCHEDULE_H

// MAGI-2-preview's sampling, as the release's inference/pipeline/
// sampler.py and inference_engine.py run it (read, not run):
//
//   * the clip: seconds * 12.5 * 2 frames at 25 fps, 8 to a latent frame
//     (32 latent frames for the 10 s the model is trained on -- the only
//     length it supports), and 25 audio latents a second;
//   * the grid: Wan's FlowUniPCMultistepScheduler (UniPC, bh2, order 2,
//     x0 prediction, lower order at the last step -- vpipe's FlowSampler
//     "unipc") over np.linspace(0.999, 0, steps + 1)[:-1] (its sigma_max
//     is float32(0.999)), pushed through the static flow shift (7 for the
//     preview), a terminal 0 appended; video and audio each have their
//     own scheduler over the same grid;
//   * the DiT is told the step's INTEGER timestep int64(sigma * 1000) --
//     the float64 sigma's -- as float32(k) / 1000;
//   * CFG against the negative prompt's condition: video 5, audio 7.
//
// The grid is computed here as numpy computes it (no fused multiply-add:
// a node an ulp off can move int(sigma * 1000) across an integer) and
// handed to the FlowSampler as base_sigmas at shift 1, which runs them as
// written.

#include "generative-models/krea2/flow-sampler.h"

#include <vector>

namespace vpipe {
namespace genai {
namespace magi2 {

// float32(0.999) -- the release scheduler's sigma_max -- as a double.
inline constexpr double kSigmaMax = 0.9990000128746033;

inline constexpr int kPreviewSteps = 100;
inline constexpr double kPreviewShift = 7.0;
inline constexpr double kVideoGuidance = 5.0;
inline constexpr double kAudioGuidance = 7.0;
inline constexpr double kFps = 12.5;
inline constexpr int kTemporalStride = 8;      // frames at 2 x fps
inline constexpr int kSpatialStride = 16;
inline constexpr int kVideoChannels = 48;
inline constexpr int kAudioChannels = 64;

// The release's _resolve_lengths: latent frames and audio latents.
int video_latents(double seconds);
int audio_latents(double seconds);

// The shifted grid, `steps` points, no terminal: shift * s / (1 +
// (shift - 1) * s) of np.linspace(kSigmaMax, 0, steps + 1)[:-1].
std::vector<double> sigma_grid(int steps, double shift);

// The DiT's timestep at a sigma of the grid: int64(sigma * 1000) / 1000
// in float32.
int model_timestep(double sigma);
float model_t(double sigma);

// The FlowSampler that runs it: UniPC bh2 order 2 over sigma_grid.
FlowSamplerSpec sampler_spec();
FlowSchedulerSpec scheduler_spec(int steps, double shift);

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_SCHEDULE_H
