#pragma once

// First-use picker for a VAE's 3x3 convolution FALLBACK -- the route a conv
// takes once the hardware convolution and the small-cout kernel have both
// declined it. Two ways to run the same contraction:
//
//   kIm2col  materialize [H*W, 9*cin] and hand it to the dense GEMM
//   kOnChip   gather the 3x3 neighbourhood in threadgroup memory and feed the
//             GEMM directly -- no scratch, no DRAM round trip
//
// and which wins turns out to depend on BOTH the GPU and the shape. On an M4
// Pro the on-chip gather is free and beats the round trip outright. On an M5
// the im2col arm feeds matmul2d (the hardware matrix units), so skipping the
// round trip often costs more than it saves -- but not always: a wide `cout`
// makes the gather re-read the activation once per output tile, while a narrow
// one keeps it cheap. MEASURED on an M5 (FLUX.2 VAE, 256x256 probe), im2col vs
// on-chip: 128->128 6.43/5.84 and 256->128 12.15/11.07 (gather wins 1.10x),
// 256->256 15.37/21.74, 512->256 30.19/43.22, 512->512 44.01/85.63 (im2col
// wins up to 1.95x). One global answer loses either way, so the pick is per
// SHAPE.
//
// That is worth measuring rather than branching on supports_matrix_cores(),
// because the case it decides is exactly the one a capability check gets least
// right. Two models reach it from opposite directions:
//
//   FLUX.2  conv3x3_hw_ takes every 3x3 at ordinary resolutions and declines
//           only once cin*W*H passes 2^31 (4096x4096 at cin=128) -- so the
//           fallback is what a very large decode runs, and nothing else.
//   Krea-2  base_dim 96 is not a multiple of 64, so the hardware conv declines
//           the top-resolution convs at EVERY resolution -- the fallback runs
//           six 96->96 convs at full size on every decode.
//
// The Krea-2 answer on M5 came back im2col at EVERY shape -- which is what it
// already shipped, so the tune changes nothing on this box. The margin is the
// part worth keeping: it tracks the contraction depth rather than cout alone
// (its on-chip member re-stages the halo once per K-chunk, so a deeper 9*cin
// costs it more re-reads). MEASURED, im2col/on-chip at scaled probes: 3->96
// 0.77/0.76 (a tie), 96->96 4.77/7.98, 192->96 9.03/15.90, 192->192 3.14/6.74,
// 384->384 2.19/7.88 -- 1.0x to 3.6x. So the shipped default is closest to
// wrong at the SHALLOW convs, and those include the six 96->96 at full
// resolution that dominate this VAE's fallback.
//
// Their on-chip members are different kernels (FLUX.2's register-accumulator
// gather, Krea-2's threadgroup-staged conv2d_mma), which is why the memo is
// namespaced by a caller `tag`.
//
// The caller owns the weights, the scratch and the dispatch; it passes a
// `bench` that times one conv with a given member through its REAL routing, so
// the tuner measures the shipping path rather than a re-derivation of it.

#include "generative-models/shared/kernel-autotune.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace vpipe::genai::vae_conv3x3 {

enum class Kind {
  kIm2col = 0,   // materialized [H*W, 9*cin] + dense GEMM
  kOnChip,       // threadgroup gather straight into the GEMM
};

inline const char* name(Kind k)
{
  return (k == Kind::kOnChip) ? "on-chip-gather" : "im2col+gemm";
}

// Pick between the members by timing them, interleaved, on the caller's probe.
// `fallback` comes back when the tune is skipped or fewer than two members are
// available. Keyed by (tag, cin, cout, stride): per SHAPE because cout swings
// the answer, per STRIDE because a stride-2 conv reads four times the
// activation for the same output, and per TAG because two models' kOnChip are
// two different kernels sharing this one process-wide memo. Memoized so that
// repeated loads in one binary stay consistent -- two arms of an A/B must not
// disagree about which kernel they ran -- and so the tune is paid once.
template <class Bench>
inline Kind autotune(const char* tag, int cin, int cout, int stride,
                     const std::vector<Kind>& cands, Kind fallback,
                     Bench&& bench, std::string* detail = nullptr)
{
  if (std::getenv("VPIPE_VAE_CONV_NO_AUTOTUNE") != nullptr) { return fallback; }
  if (cands.size() < 2) { return fallback; }
  using Key = std::tuple<std::string, int, int, int>;
  const Key key(tag != nullptr ? tag : "", cin, cout, stride);
  static std::mutex memo_mu;
  static std::map<Key, Kind> memo;
  {
    std::lock_guard<std::mutex> lk(memo_mu);
    const auto it = memo.find(key);
    if (it != memo.end()) { return it->second; }
  }
  std::vector<double> us;
  const int w = autotune_vote((int)cands.size(), 5, 1, bench, &us);
  if (w < 0 || (std::size_t)w >= cands.size()) { return fallback; }
  const Kind won = cands[(std::size_t)w];
  if (detail != nullptr) {
    for (std::size_t i = 0; i < cands.size(); ++i) {
      if (i != 0) { *detail += " "; }
      *detail += std::string(name(cands[i]));
      if (i < us.size()) {
        *detail += " " + std::to_string(us[i] / 1000.0).substr(0, 5) + "ms";
      }
    }
  }
  {
    std::lock_guard<std::mutex> lk(memo_mu);
    memo[key] = won;
  }
  return won;
}

}  // namespace vpipe::genai::vae_conv3x3
