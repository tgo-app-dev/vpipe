#ifndef GENERATIVE_MODELS_SHARED_TORCH_RANDN_H
#define GENERATIVE_MODELS_SHARED_TORCH_RANDN_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

namespace vpipe {
namespace genai {

// torch's CPU normal sampler, bit for bit: what
//
//   g = torch.Generator("cpu").manual_seed(seed)
//   torch.randn(n, generator=g)        # float32, n >= 16
//
// draws -- mt19937 uniforms, then the sixteen-wide Box-Muller of ATen's
// normal_fill, with a tail that is not a multiple of 16 redrawing the
// last 16.
//
// STATEFUL, because references draw more than one tensor from one
// generator and the second starts where the first left the stream:
// MiniMax-H3 draws the full video noise and then the audio noise from
// the same seed, and a pipeline that wants the reference's AUDIO has to
// burn the video draw first, at the request's own geometry.
//
// Below 16 elements torch takes a scalar path this does not reproduce;
// randn() refuses such a size (returns empty) rather than drawing
// something else under the same name.
class TorchCpuRandn {
public:
  // at::mt19937 seeds from the low 32 bits exactly as std::mt19937 does.
  explicit TorchCpuRandn(std::uint64_t seed)
      : _gen((std::uint32_t)(seed & 0xffffffffu))
  {
  }

  std::vector<float> randn(std::size_t n)
  {
    std::vector<float> d;
    if (n < 16) { return d; }
    d.resize(n);
    for (std::size_t i = 0; i < n; ++i) { d[i] = uniform_(); }
    for (std::size_t i = 0; i + 16 <= n; i += 16) { fill16_(d.data() + i); }
    if (n % 16 != 0) {
      float* t = d.data() + n - 16;
      for (int i = 0; i < 16; ++i) { t[i] = uniform_(); }
      fill16_(t);
    }
    return d;
  }

private:
  // uniform_real_distribution<float>: 24 random bits over 2^24.
  float uniform_()
  {
    const double x = (double)(_gen() & 0xffffffu) * (1.0 / 16777216.0);
    return (float)x;
  }

  static void fill16_(float* p)
  {
    for (int j = 0; j < 8; ++j) {
      const float u1 = 1.0f - p[j];
      const float u2 = p[j + 8];
      const float radius = std::sqrt(-2.0f * std::log(u1));
      // ATen spells theta `2.0f * pi<double> * u2` into a FLOAT: the
      // product is double, the angle and its cos/sin are not. Keeping it
      // double is off by an ulp in about half the draws.
      const float theta = (float)(2.0 * std::numbers::pi * (double)u2);
      p[j]     = radius * std::cos(theta);
      p[j + 8] = radius * std::sin(theta);
    }
  }

  std::mt19937 _gen;
};

}  // namespace genai
}  // namespace vpipe

#endif
