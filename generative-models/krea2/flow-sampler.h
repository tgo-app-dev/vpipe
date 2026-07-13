#ifndef VPIPE_GENERATIVE_MODELS_KREA2_FLOW_SAMPLER_H
#define VPIPE_GENERATIVE_MODELS_KREA2_FLOW_SAMPLER_H

#include "common/flex-data.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

// Interchangeable diffusion sampling for the Krea-2 DiT loop, split into the two
// decoupled choices a diffusion pipeline actually makes:
//
//   * the SCHEDULER (FlowSchedulerSpec) -- the discrete sigma schedule: how the
//     steps are spaced (simple / karras / exponential) plus the flow-matching
//     time-shift (mu). It owns `steps`.
//   * the SAMPLER (FlowSamplerSpec) -- the INTEGRATOR that turns per-step model
//     predictions into latent updates (euler / heun / dpmpp_2m / dpmpp_sde),
//     with the stochastic knobs (eta, s_noise, seed) for the SDE variant.
//
// Each serializes to/from FlexData, so a `sampler-select` and a
// `scheduler-select` stage each emit one on a port and the text-to-image stage
// latches both -- neither the sampler nor the schedule is baked into the loop.
//
// The distilled Krea-2-Turbo default (sampler "euler" + scheduler "simple",
// steps 8, shift 1.15 exponential) reproduces the reference token-exactly.

// ---- scheduler: the sigma schedule -------------------------------------
struct FlowSchedulerSpec {
  std::string type       = "simple";        // "simple" | "karras" | "exponential"
  int         steps      = 8;               // denoising steps
  double      shift      = 1.15;            // mu (flow-matching time-shift)
  std::string shift_type = "exponential";   // "exponential" | "linear"
  double      rho        = 7.0;             // karras curvature

  FlexData to_flex() const;   // {scheduler, type, steps, shift, shift_type, rho}
  static FlowSchedulerSpec from_flex(const FlexData& fd,
                                     std::string* err = nullptr);

  // The [steps+1] schedule: base sigmas (per `type`) over (1/steps .. 1] then
  // time-shifted by mu, decreasing, with a terminal 0 appended.
  std::vector<double> sigmas() const;

  bool operator==(const FlowSchedulerSpec& o) const noexcept
  {
    return type == o.type && steps == o.steps && shift == o.shift &&
           shift_type == o.shift_type && rho == o.rho;
  }
};

// ---- sampler: the integrator -------------------------------------------
struct FlowSamplerSpec {
  std::string   method  = "euler";   // "euler"|"heun"|"dpmpp_2m"|"dpmpp_sde"
  double        eta     = 1.0;       // dpmpp_sde stochasticity (0 => deterministic)
  double        s_noise = 1.0;       // dpmpp_sde added-noise scale
  std::uint64_t seed    = 0;         // dpmpp_sde noise seed

  FlexData to_flex() const;   // {sampler, method, eta, s_noise, seed}
  static FlowSamplerSpec from_flex(const FlexData& fd, std::string* err = nullptr);

  // Canonicalize aliases ("dpm++_2m" -> "dpmpp_2m"); unknown -> "euler".
  static std::string canon_method(const std::string& m, bool* ok = nullptr);

  bool operator==(const FlowSamplerSpec& o) const noexcept
  {
    return method == o.method && eta == o.eta && s_noise == o.s_noise &&
           seed == o.seed;
  }
};

// The runtime sampler: precomputes the scheduler's sigma schedule and steps the
// DiT loop with the selected integrator. Stateful (multistep dpmpp_2m carries
// the previous prediction; dpmpp_sde carries an RNG) -- call reset() before a
// run, then step() for each i.
class FlowSampler {
public:
  // Maps a candidate latent + its sigma to the DiT VELOCITY (dx/dsigma), same
  // length as the latent. Euler/dpmpp_2m evaluate once per step; heun/dpmpp_sde
  // evaluate twice (except the terminal step). The x0 prediction the exponential
  // integrators use is derived internally as `denoised = x - sigma*velocity`.
  using DenoiseFn =
      std::function<std::vector<float>(const std::vector<float>&, double)>;

  FlowSampler(FlowSamplerSpec sampler, FlowSchedulerSpec scheduler);

  const FlowSamplerSpec&     sampler()   const noexcept { return _sampler; }
  const FlowSchedulerSpec&   scheduler() const noexcept { return _scheduler; }
  int                        steps()     const noexcept { return _scheduler.steps; }
  const std::vector<double>& sigmas()    const noexcept { return _sigmas; }

  // Clear per-run state (multistep history + reseed the SDE RNG). Call before
  // the loop; for img2img call it once before the tail loop from `start`.
  void reset();

  // Advance `packed` in place across step i in [0, steps).
  void step(int i, std::vector<float>& packed, const DenoiseFn& denoise);

private:
  std::vector<float> velocity_(const std::vector<float>& x, double sigma,
                               const DenoiseFn& denoise) const;
  std::vector<float> denoised_(const std::vector<float>& x, double sigma,
                               const DenoiseFn& denoise) const;
  void gaussian_(std::vector<float>& out);

  FlowSamplerSpec     _sampler;
  FlowSchedulerSpec   _scheduler;
  std::vector<double> _sigmas;

  // per-run state.
  std::vector<float> _old_denoised;   // dpmpp_2m multistep history
  double             _t_prev = 0.0;   // dpmpp_2m previous -log(sigma)
  bool               _have_prev = false;
  std::uint64_t      _rng = 0;        // dpmpp_sde xorshift state
};

}  // namespace genai
}  // namespace vpipe

#endif
