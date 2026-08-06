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
// `scheduler-select` stage each emit one on a port and the generate-image stage
// latches both -- neither the sampler nor the schedule is baked into the loop.
//
// The distilled Krea-2-Turbo default (sampler "euler" + scheduler "simple",
// steps 8, shift 1.15 exponential) reproduces the reference token-exactly.

// ---- scheduler: the sigma schedule -------------------------------------
struct FlowSchedulerSpec {
  // "simple" | "karras" | "exponential" | "boogu_v1" | "unipc_flow".
  //
  // "unipc_flow" is the flow-sigma schedule of the diffusers
  // UniPCMultistepScheduler, which is what every Wan video model ships
  // with. Its base grid is linspace(1, 1/num_train, steps+1)[:-1] -- note
  // the [:-1] of a steps+1 grid, which is NOT the linspace(1, 1/num_train,
  // steps) the dynamic_shift mode below uses, and the two disagree at every
  // point but the first. Each base sigma is then pushed through the STATIC
  // flow shift
  //     s' = shift*s / (1 + (shift-1)*s)
  // (`shift`, 3.0 for Wan2.2-I2V; `shift_type` does not apply -- this curve
  // is the scheduler's own, not the exponential/linear pair), a terminal 0
  // is appended, and sigma[0] is nudged down by 1e-6 when it lands exactly
  // on 1 so that log(alpha) = log(1 - sigma) stays finite in the first
  // UniPC update.
  //
  // "boogu_v1" is Boogu-Image's FlowMatchEulerDiscreteScheduler time-shifting:
  // its sigma convention is INVERTED relative to every other schedule here --
  // t runs 0 (pure noise) -> 1 (clean) and the Euler update is
  // x += (t_next - t) * v, so `sigmas()` returns an ASCENDING [steps+1] array
  // ending at 1 rather than a descending one ending at 0. The base grid is
  // linspace(0,1,steps+1)[:-1] pushed through the logistic shift
  //   t' = 1 - e^mu / (e^mu + (1/(1-t) - 1)^sigma)
  // with mu from the base_shift/max_shift linear map evaluated at `seq_len`
  // (the checkpoint's static token count -- 4096 for 1K).
  std::string type       = "simple";
  int         steps      = 8;               // denoising steps
  double      shift      = 1.15;            // mu (flow-matching time-shift)
  std::string shift_type = "exponential";   // "exponential" | "linear"
  double      rho        = 7.0;             // karras curvature
  // boogu_v1 only: the STATIC token count the shift's mu is read off (the
  // scheduler_config `seq_len`; 0 => use the runtime img_seq_len).
  int         seq_len    = 4096;

  // ---- FlowMatchEuler dynamic shifting (Qwen-Image / SD3-style) --------
  // When `dynamic_shift`, the flow-matching mu (used in place of `shift`)
  // is computed PER-IMAGE from the packed image-token count (`img_seq_len`)
  // via calculate_shift(base_seq,max_seq,base_shift,max_shift); the base
  // sigma grid is the diffusers linspace(1, 1/num_train, steps) (not the
  // linspace(1, 1/steps) of the static `type`s); and a nonzero
  // `shift_terminal` stretches the shifted schedule so its last nonzero
  // sigma lands exactly on `shift_terminal`. `type`/`shift`/`rho` are
  // ignored in this mode; `shift_type` still selects the time-shift curve.
  bool        dynamic_shift  = false;
  double      base_shift     = 0.5;
  double      max_shift      = 0.9;
  double      shift_terminal = 0.0;         // 0 = no terminal stretch
  int         base_seq       = 256;
  int         max_seq        = 8192;
  int         num_train      = 1000;        // num_train_timesteps
  // Per-image RUNTIME binding (packed grid_h*grid_w) -- NOT a config
  // choice: excluded from operator== and (de)serialization. The caller
  // sets it before constructing the FlowSampler when dynamic_shift is on.
  int         img_seq_len    = 0;

  FlexData to_flex() const;   // {scheduler, type, steps, shift, shift_type, rho, +dyn}
  static FlowSchedulerSpec from_flex(const FlexData& fd,
                                     std::string* err = nullptr);

  // The [steps+1] schedule: base sigmas over (1/steps .. 1] (or the
  // diffusers linspace(1, 1/num_train, steps) when dynamic_shift) then
  // time-shifted by mu (static `shift`, or per-image calculate_shift),
  // decreasing, with a terminal 0 appended. In dynamic_shift mode mu comes
  // from `img_seq_len`; the explicit overload lets a caller pass it inline.
  std::vector<double> sigmas() const;
  std::vector<double> sigmas(int img_seq_len_override) const;

  bool operator==(const FlowSchedulerSpec& o) const noexcept
  {
    return type == o.type && steps == o.steps && shift == o.shift &&
           shift_type == o.shift_type && rho == o.rho &&
           dynamic_shift == o.dynamic_shift && base_shift == o.base_shift &&
           max_shift == o.max_shift && shift_terminal == o.shift_terminal &&
           base_seq == o.base_seq && max_seq == o.max_seq &&
           num_train == o.num_train && seq_len == o.seq_len;
  }
};

// ---- sampler: the integrator -------------------------------------------
struct FlowSamplerSpec {
  // "euler"|"heun"|"dpmpp_2m"|"dpmpp_sde"|"dmd"|"unipc".
  //
  // "unipc" is the UniPC multistep predictor-corrector (diffusers
  // UniPCMultistepScheduler), the sampler every Wan video model ships
  // with. Unlike the others here it is a PREDICTOR-CORRECTOR: each step
  // first CORRECTS the previous step's landing point using the model
  // evaluation just taken, then predicts the next one -- so it gets
  // second-order accuracy out of ONE model evaluation per step, which
  // matters when a step is a 14B DiT over a video-sized latent. It carries
  // the previous x0 prediction and the previous sample as state, so
  // reset() before a run is not optional.
  //
  // `order` is the solver order (2 in every shipped Wan config) and
  // `solver_bh2` selects the B(h) variant: bh2 uses expm1(h) where bh1
  // uses h. Both are the reference's, and the checkpoints say bh2.
  //
  // "dmd" is the Boogu-Image Turbo student's few-step integrator, and it is NOT
  // an ODE solver at all: at each ASCENDING sigma it jumps straight to the x0
  // prediction (x <- x + (1 - sigma) * v) and then RE-NOISES back down to the
  // next sigma (x <- (1 - sigma') * noise + sigma' * x). Four such steps replace
  // a 25-50-step CFG trajectory. It is only meaningful on a DMD-distilled
  // checkpoint (Turbo / Edit-Turbo) and only with guidance 1.
  std::string   method  = "euler";
  double        eta     = 1.0;       // dpmpp_sde stochasticity (0 => deterministic)
  double        s_noise = 1.0;       // dpmpp_sde added-noise scale
  std::uint64_t seed    = 0;         // dpmpp_sde / dmd renoise seed
  // dmd only: the sigma the schedule STARTS at (linspace(conditioning_sigma, 1,
  // steps+1)[:-1]). The reference inference script passes 0.0 for editing and
  // 0.001 for text-to-image.
  double        conditioning_sigma = 0.0;
  // unipc only: the multistep solver order, and the B(h) variant.
  int           order      = 2;
  bool          solver_bh2 = true;

  FlexData to_flex() const;   // {sampler, method, eta, s_noise, seed, +dmd}
  static FlowSamplerSpec from_flex(const FlexData& fd, std::string* err = nullptr);

  // Canonicalize aliases ("dpm++_2m" -> "dpmpp_2m"); unknown -> "euler".
  static std::string canon_method(const std::string& m, bool* ok = nullptr);

  bool operator==(const FlowSamplerSpec& o) const noexcept
  {
    return method == o.method && eta == o.eta && s_noise == o.s_noise &&
           seed == o.seed && conditioning_sigma == o.conditioning_sigma &&
           order == o.order && solver_bh2 == o.solver_bh2;
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

  // The UniPC predictor and corrector share everything but which sigma
  // pair they sit between and where the extra D1 term comes from, so both
  // go through one routine. `rks`/`d1s` are the multistep history's
  // relative step ratios and x0 differences; `corrector` switches to the
  // (step_index-1, step_index) interval and folds in `d1_t`.
  void unipc_update_(std::vector<float>& out, const std::vector<float>& x,
                     const std::vector<float>& m0, double sigma_from,
                     double sigma_to, int order, bool corrector,
                     const std::vector<float>* d1_t) const;

  // per-run state.
  std::vector<float> _old_denoised;   // dpmpp_2m multistep history
  double             _t_prev = 0.0;   // dpmpp_2m previous -log(sigma)
  bool               _have_prev = false;
  std::uint64_t      _rng = 0;        // dpmpp_sde xorshift state

  // ---- UniPC multistep state -----------------------------------------
  // The last `order` x0 predictions (newest last) and the sigmas they were
  // taken at, plus the sample the previous step STARTED from -- the
  // corrector re-derives that step's landing point, so it needs both.
  std::vector<std::vector<float>> _uni_m;
  std::vector<double>             _uni_sigma;
  std::vector<float>              _uni_last_sample;
  bool                            _uni_have_last = false;
  int                             _uni_order     = 0;   // warmed-up order
};

}  // namespace genai
}  // namespace vpipe

#endif
