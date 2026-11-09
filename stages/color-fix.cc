#include "stages/color-fix.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace vpipe::color_fix {

namespace {

// OpenCV's borderInterpolate for BORDER_REFLECT_101 (gfedcb|abcdefgh|
// gfedcba), folding as often as it takes: a 41-tap blur reaches 20 past
// the edge of a picture that may be smaller than that.
inline int
reflect101_(int p, int len)
{
  if (len == 1) { return 0; }
  while ((unsigned)p >= (unsigned)len) {
    p = p < 0 ? -p : 2 * (len - 1) - p;
  }
  return p;
}

// Rows [0, H) in about as many pieces as there are cores, each piece run
// by `fn(y0, y1)`.
template <typename Fn>
void
rows_(int H, const Fn& fn)
{
  const int pieces = std::max(1, std::min(H, 64));
  const int per = (H + pieces - 1) / pieces;
  const Fn* f = &fn;
  dispatch_apply((std::size_t)pieces,
                 dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                 ^(std::size_t i) {
    const int y0 = (int)i * per;
    const int y1 = std::min(H, y0 + per);
    if (y0 < y1) { (*f)(y0, y1); }
  });
}

// The row pass over rows [y0, y1): t = row(src) blurred along x, the taps
// summed in order as OpenCV's row filter does. `row(y, pad)` fills `pad`
// (W + 2r) with row y's samples at x = -r .. W + r - 1, edges reflected.
template <typename Row>
void
row_pass_(int y0, int y1, int W, const float* taps, int K, float* tmp,
          const Row& row)
{
  const int r = K / 2;
  std::vector<float> pad((std::size_t)W + 2 * r);
  for (int y = y0; y < y1; ++y) {
    row(y, pad.data());
    float* t = tmp + (std::size_t)y * W;
    std::fill(t, t + W, 0.0f);
    for (int k = 0; k < K; ++k) {
      const float f = taps[k];
      const float* p = pad.data() + k;
      for (int x = 0; x < W; ++x) { t[x] += f * p[x]; }
    }
  }
}

// The column pass for output row y: the centre tap, then each pair of
// rows k above and below together, as OpenCV's symmetric column filter.
inline void
column_(const float* tmp, int y, int H, int W, const float* taps, int K,
        float* o)
{
  const int r = K / 2;
  const float* c = tmp + (std::size_t)y * W;
  const float f0 = taps[r];
  for (int x = 0; x < W; ++x) { o[x] = f0 * c[x]; }
  for (int k = 1; k <= r; ++k) {
    const float f = taps[r + k];
    const float* a = tmp + (std::size_t)reflect101_(y - k, H) * W;
    const float* b = tmp + (std::size_t)reflect101_(y + k, H) * W;
    for (int x = 0; x < W; ++x) { o[x] += f * (a[x] + b[x]); }
  }
}

}  // namespace

std::vector<float>
gaussian_taps(double sigma)
{
  if (!(sigma > 0.0)) { return {1.0f}; }
  // GaussianBlur's size for a float image: cvRound(sigma * 4 * 2 + 1),
  // made odd. cvRound rounds half to even.
  const int n = (int)std::nearbyint(sigma * 8.0 + 1.0) | 1;
  // getGaussianKernelBitExact's sum, in double: x = 2 d runs over the
  // half kernel, so exp(x^2 * -0.125 / sigma^2) is exp(-d^2 / 2 sigma^2).
  const int half = (n - 1) / 2;
  const double scale = -0.125 / (sigma * sigma);
  std::vector<double> v((std::size_t)half);
  double sum = 0.0;
  for (int i = 0, x = 1 - n; i < half; ++i, x += 2) {
    v[(std::size_t)i] = std::exp((double)x * x * scale);
    sum += v[(std::size_t)i];
  }
  sum = sum * 2.0 + 1.0;
  const double mul = 1.0 / sum;
  std::vector<float> taps((std::size_t)n);
  for (int i = 0; i < half; ++i) {
    const float t = (float)(v[(std::size_t)i] * mul);
    taps[(std::size_t)i] = t;
    taps[(std::size_t)(n - 1 - i)] = t;
  }
  taps[(std::size_t)half] = (float)mul;
  return taps;
}

void
gaussian_blur(const float* src, float* dst, float* tmp, int H, int W,
              const std::vector<float>& taps)
{
  if (H <= 0 || W <= 0 || taps.empty()) { return; }
  const float* tp = taps.data();
  const int K = (int)taps.size();
  const int r = K / 2;
  rows_(H, [&](int y0, int y1) {
    row_pass_(y0, y1, W, tp, K, tmp, [&](int y, float* pad) {
      const float* s = src + (std::size_t)y * W;
      for (int x = -r; x < W + r; ++x) { pad[x + r] = s[reflect101_(x, W)]; }
    });
  });
  rows_(H, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      column_(tmp, y, H, W, tp, K, dst + (std::size_t)y * W);
    }
  });
}

void
gaussian(float* x, const float* ref, float* tmp, int H, int W,
         const std::vector<float>& taps)
{
  if (H <= 0 || W <= 0 || taps.empty()) { return; }
  const float* tp = taps.data();
  const int K = (int)taps.size();
  const int r = K / 2;
  // Rows of low(ref - x) along x, for EVERY row before any is written:
  // the column pass below writes x, which this one reads.
  rows_(H, [&](int y0, int y1) {
    row_pass_(y0, y1, W, tp, K, tmp, [&](int y, float* pad) {
      const float* a = ref + (std::size_t)y * W;
      const float* b = x + (std::size_t)y * W;
      for (int i = -r; i < W + r; ++i) {
        const int j = reflect101_(i, W);
        pad[i + r] = a[j] - b[j];
      }
    });
  });
  rows_(H, [&](int y0, int y1) {
    std::vector<float> o((std::size_t)W);
    for (int y = y0; y < y1; ++y) {
      column_(tmp, y, H, W, tp, K, o.data());
      float* d = x + (std::size_t)y * W;
      for (int i = 0; i < W; ++i) {
        d[i] = std::clamp(d[i] + o[(std::size_t)i], 0.0f, 1.0f);
      }
    }
  });
}

void
adain(float* x, const float* ref, std::size_t n)
{
  if (n < 2) { return; }
  // torch.std: the unbiased spread, about a mean taken first.
  auto stats = [n](const float* p, double* m, double* s) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) { sum += p[i]; }
    *m = sum / (double)n;
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = p[i] - *m;
      ss += d * d;
    }
    *s = std::sqrt(ss / (double)(n - 1)) + 1e-5;
  };
  double mx = 0, sx = 0, mr = 0, sr = 0;
  stats(x, &mx, &sx);
  stats(ref, &mr, &sr);
  const double g = sr / sx;
  for (std::size_t i = 0; i < n; ++i) {
    const double v = (x[i] - mx) * g + mr;
    x[i] = (float)std::clamp(v, 0.0, 1.0);
  }
}

}  // namespace vpipe::color_fix
