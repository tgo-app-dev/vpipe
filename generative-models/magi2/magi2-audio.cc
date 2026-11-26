#include "generative-models/magi2/magi2-audio.h"

#include "3rd-party/pocketfft.h"

#include <algorithm>
#include <complex>

namespace vpipe {
namespace genai {
namespace magi2 {

std::vector<float>
fft_resample(const float* planar, int channels, long n, long num)
{
  std::vector<float> out((std::size_t)channels * (std::size_t)num, 0.0f);
  if (n <= 0 || num <= 0 || channels <= 0) { return out; }
  const std::size_t nx = (std::size_t)n, nn = (std::size_t)num;
  std::vector<double> x(nx), y(nn);
  std::vector<std::complex<double>> X(nx / 2 + 1), Y(nn / 2 + 1);
  const pocketfft::shape_t sx{nx}, sy{nn};
  const pocketfft::stride_t rs{(std::ptrdiff_t)sizeof(double)};
  const pocketfft::stride_t cs{
      (std::ptrdiff_t)sizeof(std::complex<double>)};
  const std::size_t N = std::min(nx, nn);
  for (int c = 0; c < channels; ++c) {
    const float* src = planar + (std::size_t)c * nx;
    for (std::size_t i = 0; i < nx; ++i) { x[i] = src[i]; }
    pocketfft::r2c(sx, rs, cs, 0, true, x.data(), X.data(), 1.0);
    std::fill(Y.begin(), Y.end(), std::complex<double>(0.0, 0.0));
    for (std::size_t k = 0; k < N / 2 + 1; ++k) { Y[k] = X[k]; }
    // The Nyquist bin of an even-length truncation holds both halves of
    // the original's +-N/2 pair; of an even-length padding, half of it.
    if (N % 2 == 0) {
      if (nn < nx) {
        Y[N / 2] *= 2.0;
      } else if (nx < nn) {
        Y[N / 2] *= 0.5;
      }
    }
    // numpy's irfft: 1/num normalization, then scipy's num / n.
    pocketfft::c2r(sy, cs, rs, 0, false, Y.data(), y.data(),
                   1.0 / (double)nn);
    float* dst = out.data() + (std::size_t)c * nn;
    const double scale = (double)nn / (double)nx;
    for (std::size_t i = 0; i < nn; ++i) { dst[i] = (float)(y[i] * scale); }
  }
  return out;
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe
