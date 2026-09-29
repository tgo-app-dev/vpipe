#ifndef VPIPE_GENERATIVE_MODELS_SHARED_MOTION_CACHE_H
#define VPIPE_GENERATIVE_MODELS_SHARED_MOTION_CACHE_H

// MotionCache -- skip whole denoiser forwards whose output a cached
// residual already predicts, judged by a MOTION-WEIGHTED estimate of how
// far the output has drifted since the last one that ran.
//
// From "Motion-Aware Caching for Efficient Autoregressive Video
// Generation" (Xu et al., arXiv:2605.01725; github.com/MAC-AutoML/
// MotionCache), in the STEP-LEVEL form the MiniMax-H3 adaptation takes
// (github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache). The paper's
// own implementations also route individual TOKENS through a partial
// forward; that half needs a gathered attention per family and is not
// here. What is here decides per forward, which every family can take
// without touching its transformer.
//
// ---------------------------------------------------------------------
// THE IDEA, in three moves
//
//   * A computed forward leaves behind a RESIDUAL -- its output minus
//     its input, in the reference's convention -- and a reused step
//     answers "the input moved, the residual did not". For a flow model
//     that is the Euler velocity damped by the step, not the raw
//     velocity repeated, which is the difference between a skipped step
//     that drifts and one that tracks.
//   * Whether to trust it comes from the last two COMPUTED forwards: the
//     ratio of how far the output moved to how far the input moved is a
//     local gain k, and k * |input change| / |output| is the predicted
//     relative change of the output now. Those scores ACCUMULATE across
//     consecutive skips, so a run of small ones still forces a forward.
//   * Every change is weighted by MOTION: per latent position, the
//     frame-to-frame difference of the model's own clean estimate x0,
//     normalised to mean 1. A drift where the clip is moving costs more
//     than the same drift over a static background, which is what lets
//     a threshold that keeps faces and hands intact still skip.
//
// Warm-up steps always run (the coarse structure is decided there), the
// cache is consulted only inside a sigma window, and a cap on
// CONSECUTIVE skips bounds how stale a residual can get.
//
// ---------------------------------------------------------------------
// CONVENTIONS a family must get right
//
// The cache works on HOST f32 state, the latent the sampler steps and the
// velocity it steps with, for the GENERATED rows only -- conditioning
// rows that are never stepped do not belong in any of its means.
//
//   * VELOCITY SIGN. `x0_sign` is +1 when x0 = x + sigma * v (MiniMax-H3)
//     and -1 when x0 = x - sigma * v. The residual, the reuse and the
//     motion weights all depend on it; the change and norm measures do
//     not.
//   * THE VIDEO GATHER is the SUBSAMPLED latent grid the estimator reads,
//     as element offsets into the caller's flat state, in [C][T][H'][W']
//     order: channel, latent frame, then a raster over the subsampled
//     plane. The reference keeps every `subsample`-th latent row and
//     column (not patch), so a family whose state is patch-packed maps
//     through its own packing here -- and only the family can.
//   * THE AUDIO GATHER (optional) is the same for a second stream with no
//     frame axis; the reference keeps every `subsample`-th time step.
//     Leave it empty for a video-only model and the score is video's.
//
// ---------------------------------------------------------------------
// ABI SHAPE, on purpose
//
// One pointer with every method out of line, configured FROM THE BAG
// rather than from a struct: a plugin family can hold one without its
// layout being compiled into the plugin, and a knob added later is a new
// key in accel-settings.h rather than a field an old plugin would pass
// short. `Config` below is the typed view of the same keys for the
// stage and for logging; it never crosses into the out-of-line code.
// These method signatures are FROZEN once a plugin calls them.

#include "common/flex-data.h"
#include "generative-models/shared/accel-settings.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace vpipe {
namespace genai {
namespace motion_cache {

// The settings, typed. Defaults are the reference node's.
struct Config {
  bool   enabled         = false;
  // Accumulated predicted relative change under which a step reuses.
  double threshold       = 0.15;
  // How much frame-to-frame motion weighs in every change measure. 0
  // weighs every position equally.
  double motion_strength = 1.0;
  // Leading forwards always computed.
  int    warmup_steps    = 4;
  // Most reused forwards in a row.
  int    max_skips       = 2;
  // The window the cache is consulted in, as fractions of the sampling
  // range: 0 is pure noise, 1 is clean. Mapped to sigma through the
  // family's own shift (percent_to_sigma).
  double start           = 0.15;
  double end             = 0.95;
  // Latent stride of the change estimator, both spatial axes (and audio
  // time).
  int    subsample       = 8;
};

// Read out of the acceleration bag. Header-only, so it compiles into the
// caller and an absent key is this file's default.
inline Config
config_from_flex(const FlexData* bag)
{
  Config c;
  c.enabled = accel::flag(bag, accel::kMotionCache, c.enabled);
  c.threshold =
      accel::real(bag, accel::kMotionCacheThreshold, c.threshold);
  c.motion_strength =
      accel::real(bag, accel::kMotionCacheStrength, c.motion_strength);
  c.warmup_steps = (int)accel::integer(bag, accel::kMotionCacheWarmup,
                                       c.warmup_steps);
  c.max_skips = (int)accel::integer(bag, accel::kMotionCacheMaxSkips,
                                    c.max_skips);
  c.start = accel::real(bag, accel::kMotionCacheStart, c.start);
  c.end   = accel::real(bag, accel::kMotionCacheEnd, c.end);
  c.subsample = (int)accel::integer(bag, accel::kMotionCacheSubsample,
                                    c.subsample);
  return c;
}

// A fraction of the sampling range to a sigma, for a flow schedule
// shifted by `shift` -- ComfyUI's ModelSamplingDiscreteFlow::
// percent_to_sigma, which is what the reference's window is defined by.
// 0 is sigma 1 (noise), 1 is sigma 0.
inline double
percent_to_sigma(double percent, double shift)
{
  if (percent <= 0.0) { return 1.0; }
  if (percent >= 1.0) { return 0.0; }
  const double t = 1.0 - percent;
  if (shift == 1.0) { return t; }
  return shift * t / (1.0 + (shift - 1.0) * t);
}

class Cache {
 public:
  Cache();
  ~Cache();
  Cache(Cache&&) noexcept;
  Cache& operator=(Cache&&) noexcept;
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  // Arm for ONE generation from the bag. False (and inert) when the bag
  // does not turn the cache on. The window is given in SIGMA, because
  // only the family knows its shift: see percent_to_sigma. Every earlier
  // generation's state is dropped.
  bool begin(const FlexData* accel, double start_sigma, double end_sigma,
             float x0_sign);
  bool armed() const;
  // The stride the gathers below must be built at.
  int subsample() const;

  // The subsampled grids, as offsets into the caller's flat buffers (see
  // the header). `video_elems` / `audio_elems` are the FULL generated
  // sizes, which is what the residual covers. Copied.
  void set_video(const std::uint32_t* gather, int channels, int frames,
                 int plane, std::size_t video_elems);
  void set_audio(const std::uint32_t* gather, std::size_t count,
                 std::size_t audio_elems);

  // Once per model call, BEFORE it runs, with the state it would run on.
  // True means skip the forward and take predict()'s velocity instead.
  bool decide(double sigma, const float* video_x, const float* audio_x);
  // The reused velocity: the cached residual carried to the current
  // input. Only after decide() said true.
  void predict(const float* video_x, float* video_v, const float* audio_x,
               float* audio_v) const;
  // After a forward that RAN, with the input it ran on (not yet stepped)
  // and the velocity it returned.
  void update(double sigma, const float* video_x, const float* video_v,
              const float* audio_x, const float* audio_v);

  // What happened. calls() counts decide(); skipped() the ones it
  // answered true. last_score() is the score behind the latest decision,
  // negative when none was taken (warm-up, outside the window, no gain
  // yet); accumulated() the running sum it was compared with.
  int    calls() const;
  int    skipped() const;
  double last_score() const;
  double accumulated() const;
  // One line for a log: the settings and the window.
  std::string describe() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace motion_cache
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_MOTION_CACHE_H
