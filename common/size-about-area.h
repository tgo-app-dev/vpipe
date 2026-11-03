#ifndef COMMON_SIZE_ABOUT_AREA_H
#define COMMON_SIZE_ABOUT_AREA_H

// The picture size a model trained on a bucket of sizes about one AREA
// expects for an aspect ratio -- Bria's rule for Fibo 2.0
// (BriaFibo2ImageProcessor.get_image_size), restated exactly so a crop
// made here is the reference's: of the (width, height) on a `multiple`
// grid, at least 4 of it a side and within 10% of `area`, the one that
// crops least from a picture of `aspect` (width / height), then the one
// closest to `area`, then the wider. {0, 0} when no size qualifies.
//
// A brute-force walk, as the reference's: ~4M candidates at 4 MP, a few
// milliseconds -- a caller sizing every frame of a stream keeps the
// answer per source size.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace vpipe {

inline std::pair<int, int>
size_about_area(double aspect, long long area, int multiple = 16)
{
  std::pair<int, int> best{0, 0};
  if (!(aspect > 0.0) || area <= 0 || multiple <= 0) { return best; }
  const long long min_side = 4LL * multiple;
  double best_crop = 0.0;
  long long best_dist = 0;
  bool have = false;
  for (long long w = min_side; w < 2 * area / min_side; w += multiple) {
    for (long long h = min_side; h < 2 * area / w; h += multiple) {
      const long long a = w * h;
      if (!(0.9 * (double)area <= (double)a &&
            (double)a <= 1.1 * (double)area)) {
        continue;
      }
      const double ratio = (double)w / (double)h;
      // round(crop, 9): the reference compares crops at 9 decimals.
      const double crop = std::nearbyint(
          (1.0 - std::min(ratio / aspect, aspect / ratio)) * 1e9) / 1e9;
      const long long dist = std::llabs(a - area);
      // min over (crop, |a - area|, -w): the wider wins the last tie.
      const bool better =
          !have || crop < best_crop ||
          (crop == best_crop &&
           (dist < best_dist ||
            (dist == best_dist && w > (long long)best.first)));
      if (better) {
        best = {(int)w, (int)h};
        best_crop = crop;
        best_dist = dist;
        have = true;
      }
    }
  }
  return best;
}

}  // namespace vpipe

#endif
