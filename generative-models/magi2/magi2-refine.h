#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_REFINE_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_REFINE_H

// MAGI-2's second stage, the refiner's sampling (the release's
// evaluate_magi2_refiner_with_latent, read not run): the preview's video
// latent upsampled trilinearly (align_corners) to 2T - 1 frames on the
// target grid, re-noised to x s + n sqrt(1 - s^2) with s the ZeroSNR DDPM
// table's entry 220, then denoised by UniPC over 5 steps at shift 5 with
// classifier-free guidance 2 against the negative prompt. No audio (the
// preview's soundtrack is the clip's) and no time input (the refiner
// reads none). Its frames are 4 (2T - 2) + 1 at 25 fps -- the preview's
// clip at twice its rate.

#include "generative-models/magi2/metal-magi2-refiner.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace magi2 {

inline constexpr int kRefineSteps = 5;
inline constexpr double kRefineShift = 5.0;
inline constexpr double kRefineGuidance = 2.0;
inline constexpr int kRefineNoiseIndex = 220;
inline constexpr double kRefineFps = 25.0;

// The refined clip's latent frames for a preview of `T`.
inline int refine_latent_frames(int T) { return T <= 0 ? 0 : 2 * T - 1; }

// ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)[i]:
// the signal scale the refine re-noises the upsampled latent to.
float refine_signal_scale(int index = kRefineNoiseIndex);

// torch's trilinear interpolation with align_corners=True, f32:
// [C, T, H, W] to [C, To, Ho, Wo].
std::vector<float> upsample_trilinear(const float* x, int C, int T, int H,
                                      int W, int To, int Ho, int Wo);

struct RefineRequest {
  int T = 0, h = 0, w = 0;       // the preview's latent
  const float* latent = nullptr; // [48, T, h, w]
  int H = 0, W = 0;              // the target grid (latent cells)
  int steps = kRefineSteps;
  double shift = kRefineShift;
  double guidance = kRefineGuidance;
  std::uint64_t seed = 0;
  // The conditioner's TEXT rows, f32 [rows, text_c] (no picture token);
  // the negative may be absent, which with guidance != 1 is refused.
  const float* cond = nullptr;
  int cond_rows = 0;
  const float* neg = nullptr;
  int neg_rows = 0;
  std::function<bool(int step, int total)> progress;
  // TESTS: the re-noising noise given instead of drawn ([48, 2T-1, H, W]).
  const float* noise = nullptr;
};

// The refiner's velocity for one sample (its forward's shape).
using RefineForward =
    std::function<bool(const MetalMagi2Refiner::Sample& s,
                       std::vector<float>* vel, std::string* err)>;

// The refined latent [48, 2T - 1, H, W] f32 on success.
bool run_refine(MetalMagi2Refiner& refiner, const RefineRequest& r,
                std::vector<float>* video, std::string* err);
bool run_refine(const RefineForward& forward, int video_c,
                const RefineRequest& r, std::vector<float>* video,
                std::string* err);

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_REFINE_H
