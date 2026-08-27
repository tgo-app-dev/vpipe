#ifndef VPIPE_GENERATIVE_MODELS_SHARED_RIFFLE_ROWS_H
#define VPIPE_GENERATIVE_MODELS_SHARED_RIFFLE_ROWS_H

#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vpipe {
namespace genai {

// The gate|up RIFFLE, in place:
//
//   [g0 .. g_{m-1} | u0 .. u_{m-1}]  ->  [g0, u0, g1, u1, ...]
//
// Row i moves to 2i mod (n-1) with the last row fixed, which is the
// standard riffle. Following the permutation's CYCLES needs one row of
// scratch and a visited byte per row, against a whole second copy of
// the tensor for the out-of-place version -- and on a weight measured
// in hundreds of MB that copy is the difference between a promotion
// that fits and one that fails on a tight box.
//
// Shared because two models want the identical permutation over the
// identical layout (FLUX.2's ff.linear_in, MiniMax-H3's fc1) and the
// cycle walk is not a thing to keep two copies of.
//
// SPLIT IN TWO on purpose. A quantized weight is three buffers that
// must permute together, and the walk cannot fail once started, so a
// caller checks every buffer with riffle_rows_ok() FIRST and only then
// permutes. Interleaving the checks with the walks would leave a weight
// half-permuted on the first refusal -- three buffers that individually
// look fine and together mean nothing.

// True when `b` can be riffled as `n` equal rows. An EMPTY buffer is
// fine and permutes to nothing; a caller with an optional buffer need
// not special-case it.
inline bool
riffle_rows_ok(const metal_compute::SharedBuffer& b, std::size_t n) noexcept
{
  if (b.empty() || n <= 1) { return true; }
  if ((n & 1) != 0) { return false; }        // no gate|up split
  const std::size_t rb = b.byte_size() / n;
  return rb != 0 && rb * n == b.byte_size();
}

// Permute `b` in place. The caller MUST have taken riffle_rows_ok()
// first; this cannot report a failure because a half-finished walk has
// no meaning to report it with.
inline void
riffle_rows(metal_compute::SharedBuffer& b, std::size_t n)
{
  if (b.empty() || n <= 1) { return; }
  const std::size_t rb = b.byte_size() / n;
  auto* base = static_cast<std::uint8_t*>(b.contents());
  if (base == nullptr || rb == 0) { return; }
  std::vector<std::uint8_t> seen(n, 0);
  std::vector<std::uint8_t> hold(rb), next(rb);
  for (std::size_t s = 0; s + 1 < n; ++s) {
    if (seen[s] != 0) { continue; }
    std::memcpy(hold.data(), base + s * rb, rb);
    std::size_t i = s;
    for (;;) {
      const std::size_t j = (2 * i) % (n - 1);
      std::memcpy(next.data(), base + j * rb, rb);
      std::memcpy(base + j * rb, hold.data(), rb);
      hold.swap(next);
      seen[j] = 1;
      i = j;
      if (j == s) { break; }
    }
  }
}

}  // namespace genai
}  // namespace vpipe

#endif
