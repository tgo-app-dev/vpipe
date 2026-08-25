#pragma once

// Load-time picker for a VAE mid-block attention. The mid block is a SINGLE
// head at the model's widest channel count (D = 384/512), every token attending
// every other, and there are several interchangeable ways to run it. Which is
// fastest is a property of the GPU, not of the shape: on an M4 Pro the
// materialized banded GEMM beat the matrix-core flash 9.1x, and on an M5 the
// flash beats it at every size measured (1.46x at 16k mid tokens narrowing to
// 1.17x at 66k). The query tile of the flash kernel is the same story -- a
// sweep on one machine picked 32, and nothing says the next machine agrees.
//
// So the members are TIMED rather than predicted from a capability bit. The
// voting itself is kernel-autotune's: candidates run interleaved each round and
// the per-round winner votes, which cancels the absolute clock (this box gates
// its GPU clock on the SoC power budget, so a sequential arm-at-a-time layout
// reads back whichever arm ran coldest).
//
// AND THEY ARE VERIFIED FIRST. Timing alone picks a member that is fast and
// WRONG and reports nothing -- which is exactly what happened: see the
// correctness gate below. The members are not numerically interchangeable at
// this call site, so a candidate that cannot reproduce the reference at
// realistic score magnitudes never reaches the stopwatch.
//
// The caller owns the kernels and the dispatch -- it passes an `encode` that
// runs one attention with a given member, the same entry point its decode uses,
// so the tuner measures the shipping path rather than a re-derivation of it.
// FLUX.2 offers all seven members; the Krea-2 VAE has no materialized path and
// simply does not list kMat.

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-autotune.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}  // namespace vpipe::metal_compute

namespace vpipe::genai::vae_mid_attn {

enum class Kind {
  kScalar = 0,   // sdpa_full_f16, per-query O(N^2) -- last resort
  kSmm,          // sdpa_full_mma_f16, simdgroup_matrix flash
  kMma8,         // sdpa_full_mma2_dN_f16, matmul2d flash, BQ=8
  kWide16, kWide32, kWide64,   // ... register-accumulator, wider query tile
  kMat,          // materialized banded GEMM (QK^T -> softmax -> PV)
};

inline const char* name(Kind k)
{
  switch (k) {
    case Kind::kScalar: return "scalar";
    case Kind::kSmm:    return "smm";
    case Kind::kMma8:   return "mma-q8";
    case Kind::kWide16: return "mma-q16";
    case Kind::kWide32: return "mma-q32";
    case Kind::kWide64: return "mma-q64";
    case Kind::kMat:    return "materialized";
  }
  return "?";
}

using Alloc = std::function<metal_compute::SharedBuffer&(std::size_t)>;
using Release = std::function<void(const metal_compute::SharedBuffer&)>;
using EncodeFn = std::function<void(
    metal_compute::ComputeEncoder&, Kind,
    const metal_compute::SharedBuffer& q, const metal_compute::SharedBuffer& k,
    const metal_compute::SharedBuffer& v, const metal_compute::SharedBuffer& att,
    std::size_t hw, int C, float scale, const Alloc&, const Release&)>;

// The probe size. Attention is O(N^2), so this IS the cost of the tune; it
// stands in for a real mid block (4k..66k tokens) and rests on the RANKING
// being stable in N even though the margin is not, which is what the M5 sweep
// shows. MEASURED floor: 1024 cannot separate the query tiles (it picked a
// member that then ran 149 ms against the winner's 112 at a 1024x1024 decode);
// 2048 picks the optimum for ~0.1 s, the same order as the LM autotuners.
inline constexpr int kProbeN = 2048;

// Time every candidate and return the winner. `fallback` comes back unchanged
// when the tune cannot run (allocation refused, fewer than two candidates, or
// VPIPE_VAE_ATTN_NO_AUTOTUNE). `detail` (optional) receives the per-candidate
// timings for a load-time log line.
template <class Mc, class Enc>
inline Kind autotune(Mc* mc, int C, const std::vector<Kind>& cands,
                     Kind fallback, const EncodeFn& encode,
                     std::string* detail = nullptr, bool verify = true)
{
  if (std::getenv("VPIPE_VAE_ATTN_NO_AUTOTUNE") != nullptr) { return fallback; }
  if (cands.size() < 2) { return fallback; }
  // PROCESS-LEVEL MEMO, keyed by the shape and the candidate set. Two loads in
  // one process MUST land on the same member: the members agree only to ~1e-3,
  // so an A/B that builds the model twice would otherwise compare outputs from
  // two different kernels and read the difference as a numerical regression.
  // MEASURED before this existed, krea2_vae.encode_tiled_im2col_matches_hwconv
  // over four consecutive runs: rel-L2 0, 6.5e-4, 6.5e-4, 1.4e-4. It also means
  // a process that builds the VAE repeatedly pays for the tune once.
  std::uint64_t mask = 0;
  for (Kind k : cands) { mask |= 1ull << (int)k; }
  const auto key = std::make_pair((std::uint64_t)C, mask);
  static std::mutex memo_mu;
  static std::map<std::pair<std::uint64_t, std::uint64_t>, Kind> memo;
  {
    std::lock_guard<std::mutex> lk(memo_mu);
    const auto it = memo.find(key);
    if (it != memo.end()) { return it->second; }
  }
  int N = kProbeN;
  if (const char* e = std::getenv("VPIPE_VAE_ATTN_TUNE_N")) {
    N = std::max(256, std::atoi(e));
  }
  const std::size_t hw = (std::size_t)N;
  const std::size_t n = hw * (std::size_t)C;
  metal_compute::SharedBuffer q = mc->make_shared_buffer(n * 2);
  metal_compute::SharedBuffer k = mc->make_shared_buffer(n * 2);
  metal_compute::SharedBuffer v = mc->make_shared_buffer(n * 2);
  metal_compute::SharedBuffer att = mc->make_shared_buffer(n * 2);
  if (q.empty() || k.empty() || v.empty() || att.empty()) { return fallback; }
  // Values are written by fill_probe below -- non-degenerate, because a
  // constant input makes the softmax uniform and every member agrees.
  // Scratch is REUSED across bench calls, not reallocated: these buffers are
  // UMA and mlock-wired, and the materialized member asks for tens of MB of
  // score band per call. Wiring that per dispatch cost more than the kernels
  // being measured (MEASURED 1.21 s of tune against ~0.2 s of GPU work). A
  // DEQUE, not a vector: alloc hands back a reference and the materialized
  // path holds the first while asking for the second, which a reallocating
  // vector would leave dangling.
  std::deque<metal_compute::SharedBuffer> scratch;
  std::size_t next = 0;
  bool alloc_ok = true;
  Alloc alloc = [&](std::size_t cnt) -> metal_compute::SharedBuffer& {
    if (next < scratch.size() && scratch[next].byte_size() >= cnt * 2) {
      return scratch[next++];
    }
    if (next < scratch.size()) { scratch.erase(scratch.begin() + (long)next); }
    scratch.insert(scratch.begin() + (long)next,
                   mc->make_shared_buffer(cnt * 2));
    return scratch[next++];
  };
  Release rel = [](const metal_compute::SharedBuffer&) {};
  const float scale = 1.0f / std::sqrt((float)C);

  // ---- CORRECTNESS GATE, before anything is timed --------------------
  //
  // A tuner that ranks by time alone will pick a member that is fast and
  // WRONG, and say nothing. MEASURED: on an M4 Pro the materialized
  // member won this tune at 3 ms against the flash members' 4-8 ms and
  // took the Boogu-Image VAE round-trip from 37.8 dB to 5.79 -- because
  // it MATERIALIZES QK^T in f16, and this call site's Q/K are raw
  // conv1x1 outputs, not normalized projections. In that VAE's DECODER
  // max|q| = 975 and max|k| = 428 at D = 512, so the scores reach
  // 1.24e8: every one of them is outside f16's 65504 and the softmax
  // collapses to an average. The ENCODER of the same model peaks at
  // 5885 and is fine, which is why the failure looked intermittent.
  //
  // So each candidate is CHECKED against a reference before it is
  // allowed to compete. The probe deliberately drives the scores past
  // f16 -- a positive-mean Q/K, so the dot is ~D*A^2 rather than a
  // random walk -- because that is the property that separates the
  // members. The timing probe below stays small and cheap; this one runs
  // once per candidate and is O(N^2) at the same N.
  //
  // The reference is whatever the caller can encode most simply and is
  // known good at any magnitude: the scalar O(N^2) member accumulates in
  // f32 per query row. If the caller has no scalar member, the gate is
  // skipped rather than guessed at.
  auto fill_probe = [&](float amp) {
    auto* pq = static_cast<_Float16*>(q.contents());
    auto* pk = static_cast<_Float16*>(k.contents());
    auto* pv = static_cast<_Float16*>(v.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float t = (float)(i % 61) * 0.01f - 0.3f;
      pq[i] = (_Float16)(amp + t);
      pk[i] = (_Float16)(amp - t);
      pv[i] = (_Float16)(t * 0.5f);
    }
  };
  std::vector<Kind> live = cands;
  if (verify) {
    // AMPLITUDE. The dot is ~D*A^2, so A=20 at D=512 puts the scores near
    // 2.0e5 -- 3x past f16's 65504 -- while every INPUT stays comfortably
    // inside it. A member is therefore judged on where it keeps the
    // SCORES and on nothing else. MEASURED at A=10 (dot ~5.1e4, just
    // UNDER the limit) the materialized member still failed, but only at
    // rel-L2 0.14 against a 0.05 bar; A=20 makes it unambiguous without
    // driving the softmax so peaked that correct members disagree on
    // which key wins.
    fill_probe(20.0f);
    std::vector<float> ref;
    double worst_kept = 0.0;
    auto capture = [&](Kind kind, std::vector<float>* out) {
      next = 0;
      std::memset(att.contents(), 0, n * 2);
      autotune_time(mc, 1, [&](Enc& enc) {
        encode(enc, kind, q, k, v, att, hw, C, scale, alloc, rel);
      });
      const auto* pa = static_cast<const _Float16*>(att.contents());
      out->resize(n);
      for (std::size_t i = 0; i < n; ++i) { (*out)[i] = (float)pa[i]; }
    };
    capture(Kind::kScalar, &ref);
    double den = 0.0;
    for (float f : ref) { den += (double)f * (double)f; }
    if (den > 0.0) {
      std::vector<Kind> ok;
      std::vector<float> got;
      for (Kind kind : live) {
        capture(kind, &got);
        double num = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < n; ++i) {
          const double d = (double)got[i] - (double)ref[i];
          if (!std::isfinite(got[i])) { finite = false; break; }
          num += d * d;
        }
        const double r = finite ? std::sqrt(num / den) : 1e9;
        if (r < 0.05) {
          ok.push_back(kind);
          if (r > worst_kept) { worst_kept = r; }
          continue;
        }
        // Named, always. A member silently dropped is a performance
        // change nobody can account for later, and the number is the
        // only thing that says whether the bar was close.
        if (detail != nullptr) {
          *detail += std::string(name(kind)) + " REJECTED(rel-L2 " +
                     std::to_string(r) + ") ";
        }
      }
      // Never reject everything: a gate that empties the candidate list
      // has told us the REFERENCE is what disagrees, and falling back is
      // the only answer that cannot make things worse.
      if (!ok.empty()) { live = ok; }
      if (detail != nullptr && !ok.empty()) {
        *detail += "verified(worst rel-L2 " + std::to_string(worst_kept) +
                   ") ";
      }
    }
    fill_probe(0.0f);            // back to the timing probe's values
    if (live.size() < 2) {
      const Kind won = live.empty() ? fallback : live[0];
      std::lock_guard<std::mutex> lk(memo_mu);
      memo[key] = won;
      return won;
    }
  }
  auto bench = [&](int i) -> double {
    if (!alloc_ok) { return 0.0; }
    next = 0;                                  // rewind the scratch ring
    const double t = autotune_time(mc, 1, [&](Enc& enc) {
      encode(enc, live[(std::size_t)i], q, k, v, att, hw, C, scale, alloc, rel);
    });
    for (const metal_compute::SharedBuffer& b : scratch) {
      if (b.empty()) { alloc_ok = false; }
    }
    return alloc_ok ? t : 0.0;
  };
  // PRUNE after one timed pass, keeping what is within 3x of the best. The
  // members differ by more than an order of magnitude on any given GPU (the
  // simdgroup-matrix flash is ~10x the matrix-core one on an M5), so voting
  // over the full set spends its time re-timing kernels that already lost by
  // 10x. Which members survive is still the machine's call: this only drops
  // what it has SEEN lose.
  std::vector<double> first(live.size(), 0.0);
  double best_first = 0.0;
  for (std::size_t i = 0; i < live.size(); ++i) {
    first[i] = bench((int)i);
    if (first[i] <= 0.0) { return fallback; }
    if (best_first == 0.0 || first[i] < best_first) { best_first = first[i]; }
  }
  std::vector<Kind> kept;
  for (std::size_t i = 0; i < live.size(); ++i) {
    if (first[i] <= best_first * 3.0) { kept.push_back(live[i]); }
  }
  std::vector<double> us;
  int w = 0;
  if (kept.size() > 1) {
    live = kept;                               // bench() indexes `live`
    w = autotune_vote((int)live.size(), 5, 1, bench, &us);
    if (!alloc_ok) { return fallback; }
  } else {
    live = kept;
  }
  if (detail != nullptr) {
    for (std::size_t i = 0; i < live.size(); ++i) {
      if (i != 0) { *detail += " "; }
      *detail += std::string(name(live[i]));
      if (i < us.size()) {
        *detail += " " + std::to_string((long long)(us[i] / 1000.0)) + "ms";
      }
    }
  }
  const Kind won = live[(std::size_t)w];
  {
    std::lock_guard<std::mutex> lk(memo_mu);
    memo[key] = won;
  }
  return won;
}

}  // namespace vpipe::genai::vae_mid_attn
