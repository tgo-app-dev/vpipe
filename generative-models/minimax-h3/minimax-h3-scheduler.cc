#include "generative-models/minimax-h3/minimax-h3-scheduler.h"

#include <cmath>

namespace vpipe {
namespace genai {

MiniMaxH3Scheduler::MiniMaxH3Scheduler(double shift) : _shift(shift) {}

bool
MiniMaxH3Scheduler::set_timesteps(int num_steps)
{
  _sigmas.clear();
  _timesteps.clear();
  if (num_steps < 2 || !(_shift > 0.0)) { return false; }

  // torch.linspace(1, 0, n) in float32, then the exponential shift.
  //
  // Both are reproduced at torch's PRECISION, not at the best precision
  // available, and that is the point. The reference builds the grid as a
  // float32 tensor and applies the shift with float32 tensor ops, so
  // every operation rounds; computing the same expression in double and
  // rounding once lands one float32 ulp away on most points. That ulp
  // cannot reach the model -- the transformer consumes the timestep as
  // bf16, which has an 8-bit mantissa -- but reproducing it costs
  // nothing and keeps the schedule comparable to the reference at the
  // bit level rather than only "close".
  //
  // linspace's own rounding is the halfway split its CPU kernel uses:
  // points below the midpoint are built up from `start`, points at or
  // above it are built DOWN from `end`, both with a float32 step. That
  // is also what makes the terminal point exactly zero rather than a
  // rounded near-zero, which the Euler ratio below depends on.
  const float stepf = (float)(-1.0 / (double)(num_steps - 1));
  const float sh    = (float)_shift;
  const float sh1   = (float)(_shift - 1.0);
  std::vector<float> shifted;
  shifted.reserve((std::size_t)num_steps);
  for (int i = 0; i < num_steps; ++i) {
    const float base = (i < num_steps / 2)
                           ? (1.0f + stepf * (float)i)
                           : (0.0f - stepf * (float)(num_steps - 1 - i));
    shifted.push_back(sh * base / (1.0f + sh1 * base));
  }
  // unique_consecutive: the shift maps distinct base points onto the same
  // float32 near sigma = 1.
  for (float s : shifted) {
    if (_sigmas.empty() || _sigmas.back() != s) { _sigmas.push_back(s); }
  }
  if (_sigmas.size() < 2 || _sigmas.back() != 0.0f) { return false; }

  _timesteps.reserve(_sigmas.size() - 1);
  for (std::size_t i = 0; i + 1 < _sigmas.size(); ++i) {
    _timesteps.push_back(1.0f - _sigmas[i]);
  }
  return true;
}

bool
MiniMaxH3Scheduler::step(const float* velocity, int step_index, float* x,
                         std::size_t n) const
{
  if (velocity == nullptr || x == nullptr) { return false; }
  if (step_index < 0 || step_index >= (int)_timesteps.size()) { return false; }
  const float sigma_from_t = 1.0f - _timesteps[(std::size_t)step_index];
  const float sigma = _sigmas[(std::size_t)step_index];
  const float sigma_next = _sigmas[(std::size_t)step_index + 1];
  if (!(sigma > 0.0f)) { return false; }
  const float ratio = sigma_next / sigma;
  for (std::size_t i = 0; i < n; ++i) {
    // x0 = x + sigma * v, then blend -- written out rather than folded to
    // x + (sigma_next - sigma) * v so the arithmetic matches the
    // reference term for term.
    const float x0 = x[i] + sigma_from_t * velocity[i];
    x[i] = ratio * x[i] + (1.0f - ratio) * x0;
  }
  return true;
}

bool
MiniMaxH3Scheduler::step_res(const float* velocity, int step_index, float* x,
                             std::size_t n,
                             std::vector<float>* prev_x0) const
{
  if (velocity == nullptr || x == nullptr || prev_x0 == nullptr) {
    return false;
  }
  if (step_index < 0 || step_index >= (int)_timesteps.size()) { return false; }
  const std::size_t si = (std::size_t)step_index;
  const float sigma_from_t = 1.0f - _timesteps[si];
  const float sigma = _sigmas[si];
  const float sigma_next = _sigmas[si + 1];
  if (!(sigma > 0.0f)) { return false; }

  // The history is only usable when the previous step ran over the same
  // buffer; a size change means the caller restarted.
  const bool have_prev = step_index > 0 && prev_x0->size() == n;
  const bool second_order = have_prev && sigma_next > 0.0f;

  double b1 = 0.0, b2 = 0.0, decay = 0.0;
  if (second_order) {
    const double t      = -std::log((double)sigma);
    const double t_next = -std::log((double)sigma_next);
    const double t_prev = -std::log((double)_sigmas[si - 1]);
    const double h      = t_next - t;
    // `old_sigma_down` is the previous step's sigma_down, which at
    // eta = 0 is this step's sigma -- so t_old == t and c2 reduces to
    // (t_prev - t)/h. It is NEGATIVE, since t rises as sigma falls.
    const double c2 = (t_prev - t) / h;
    if (h > 0.0 && std::isfinite(c2) && c2 != 0.0) {
      const double phi1 = std::expm1(-h) / -h;
      const double phi2 = (phi1 - 1.0) / -h;
      b1 = h * (phi1 - phi2 / c2);
      b2 = h * (phi2 / c2);
      decay = std::exp(-h);
    }
  }

  if (prev_x0->size() != n) { prev_x0->assign(n, 0.0f); }
  if (b1 == 0.0 && b2 == 0.0) {
    // Euler, and still record x0 so the NEXT step has its history.
    const float ratio = sigma_next / sigma;
    for (std::size_t i = 0; i < n; ++i) {
      const float x0 = x[i] + sigma_from_t * velocity[i];
      (*prev_x0)[i] = x0;
      x[i] = ratio * x[i] + (1.0f - ratio) * x0;
    }
    return true;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const float x0 = x[i] + sigma_from_t * velocity[i];
    x[i] = (float)(decay * (double)x[i] + b1 * (double)x0
                   + b2 * (double)(*prev_x0)[i]);
    (*prev_x0)[i] = x0;
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
