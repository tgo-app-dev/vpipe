#include "generative-models/magi2/magi2-schedule.h"

#include <cmath>

namespace vpipe {
namespace genai {
namespace magi2 {

#pragma clang fp contract(off)

int
video_latents(double seconds)
{
  // int(round(seconds * fps * 2)) frames, (frames - 1) // 8 + 1 latents.
  const long frames = std::lround(std::nearbyint(seconds * kFps * 2.0));
  if (frames <= 0) { return 0; }
  return (int)((frames - 1) / kTemporalStride + 1);
}

int
audio_latents(double seconds)
{
  return (int)std::nearbyint(seconds * 25.0);
}

std::vector<double>
sigma_grid(int steps, double shift)
{
  std::vector<double> g;
  if (steps <= 0) { return g; }
  // np.linspace(start, stop, steps + 1): step = (stop - start) / steps,
  // y[i] = i * step + start; the last point (the stop) is dropped.
  const double start = kSigmaMax, stop = 0.0;
  const double step = (stop - start) / (double)steps;
  g.reserve((std::size_t)steps);
  for (int i = 0; i < steps; ++i) {
    const double prod = (double)i * step;
    const double s = prod + start;
    const double num = shift * s;
    const double den = 1.0 + (shift - 1.0) * s;
    g.push_back(num / den);
  }
  return g;
}

int
model_timestep(double sigma)
{
  return (int)(sigma * 1000.0);
}

float
model_t(double sigma)
{
  return (float)model_timestep(sigma) / 1000.0f;
}

FlowSamplerSpec
sampler_spec()
{
  FlowSamplerSpec s;
  s.method = "unipc";
  s.order = 2;
  s.solver_bh2 = true;
  return s;
}

FlowSchedulerSpec
scheduler_spec(int steps, double shift)
{
  FlowSchedulerSpec s;
  s.type = "unipc_flow";
  s.steps = steps;
  s.shift = 1.0;
  s.base_sigmas = sigma_grid(steps, shift);
  return s;
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe
