#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_SCHEDULER_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_SCHEDULER_H

#include <cstddef>
#include <vector>

namespace vpipe {
namespace genai {

// The MiniMax-H3 sampler: rectified-flow Euler (eta = 0) with an
// exponential sigma shift.
//
// This is deliberately NOT a new `type` on the shared FlowSchedulerSpec,
// which every other diffusion model here goes through. Two of this
// model's conventions are inverted relative to that abstraction, and a
// mode that flips a sign inside a shared integrator is how a wrong
// sampler survives review:
//
//   * the velocity is DATA-ward, so the denoised estimate is
//     `x0 = x_t + sigma * v` -- a PLUS where diffusers' flow-match
//     schedulers have a minus.
//   * timesteps are `t = 1 - sigma` in [0, 1] with t = 1 meaning CLEAN.
//     Flow-match schedulers expose `sigma * num_train_timesteps`, i.e.
//     the opposite direction on a 1000x scale, and the transformer's
//     AdaLN consumes `t` unscaled.
//
// A request runs TWO of these in lockstep -- shift 12.0 over the video
// rows and 3.0 over the audio rows, from model_index.json's
// `sigma_shift_scales`. They are stepped independently on their own
// slices of the packed sequence but share a step count, so step `i` of
// one always pairs with step `i` of the other.
class MiniMaxH3Scheduler {
 public:
  // Defaults to the video shift; audio is constructed with 3.0.
  explicit MiniMaxH3Scheduler(double shift = 12.0);

  // Rebuild the schedule. `num_steps` is the number of SIGMA GRID POINTS
  // including the terminal zero, so it drives `num_steps - 1` model
  // evaluations -- the terminal sigma has none. False when `num_steps`
  // is under 2 or the shift is not positive.
  //
  // The grid is linspace(1, 0, num_steps) pushed through
  //     sigma' = s*sigma / (1 + (s-1)*sigma)
  // with consecutive duplicates collapsed: the shift compresses the grid
  // near sigma = 1 hard enough to create float32 collisions at 12.0, and
  // leaving them in would make the Euler ratio below divide two equal
  // sigmas. Collapsing them is why `timesteps().size()` can come back
  // SHORTER than the requested step count.
  bool set_timesteps(int num_steps);

  // Sigma grid, descending, terminating at exactly 0. Size is one more
  // than timesteps().
  const std::vector<float>& sigmas() const { return _sigmas; }
  // The timestep the transformer is conditioned on at each step,
  // `1 - sigma`, ASCENDING towards 1 (clean).
  const std::vector<float>& timesteps() const { return _timesteps; }
  int num_inference_steps() const { return (int)_timesteps.size(); }
  double shift() const { return _shift; }

  // One Euler step, in place on `x` (float32 -- the reference steps in
  // float32 even for half-precision samples, and this loop runs 20+
  // times over a latent that starts as unit noise).
  //
  //   x0     = x + (1 - t[i]) * v
  //   x_next = r * x + (1 - r) * x0,  r = sigma[i+1] / sigma[i]
  //
  // The sigma multiplying `v` is recovered from the TIMESTEP the
  // transformer was conditioned on rather than read from the grid. The
  // two are the same number in exact arithmetic, but `1 - (1 - sigma)`
  // does not round-trip in float32 below sigma = 0.5, and the reference
  // keeps the round trip -- so this does too.
  //
  // False when `step_index` is out of range or the sizes disagree.
  bool step(const float* velocity, int step_index, float* x,
            std::size_t n) const;

  // One `res_multistep` step -- the sampler the ComfyUI t2v template
  // actually ships (`KSamplerSelect: res_multistep`, eta = 0), a
  // second-order exponential multistep in log-sigma time.
  //
  // Writing t = -log(sigma), h = t_next - t and c2 = (t_prev - t)/h,
  //
  //   phi1 = expm1(-h) / -h,   phi2 = (phi1 - 1) / -h
  //   b1   = phi1 - phi2/c2,   b2   = phi2/c2
  //   x_next = exp(-h) * x + h * (b1 * x0 + b2 * x0_prev)
  //
  // exp(-h) is exactly the Euler ratio sigma_next/sigma, so this is a
  // strict generalization: Euler is b1 = (1 - r)/h with b2 = 0.
  //
  // `prev_x0` is the caller's state across steps -- it carries the
  // PREVIOUS step's x0 estimate in and is overwritten with this step's
  // on the way out. Pass the same vector every step and clear it to
  // restart. It is resized to `n` on first use.
  //
  // Falls back to the Euler step, exactly as the reference does, on the
  // first step (no history), on the terminal step (sigma_next = 0 makes
  // t_next infinite), and if c2 is degenerate.
  bool step_res(const float* velocity, int step_index, float* x,
                std::size_t n, std::vector<float>* prev_x0) const;

 private:
  double             _shift;
  std::vector<float> _sigmas;
  std::vector<float> _timesteps;
};

}  // namespace genai
}  // namespace vpipe

#endif
