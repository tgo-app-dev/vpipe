#pragma once

// Reusable load-time kernel autotuning: a thermo-agnostic A/B/.../N picker plus
// a per-model tuning-time report. Header-only so any metal model (Qwen, Gemma,
// ...) can include it without a build-graph change. The picker only does the
// voting + timing bookkeeping; the caller supplies a `bench(i)` that runs
// candidate i's fixed GPU work and returns wall-seconds (it owns the buffers /
// command stream). See MetalQwenModel::autotune_decode_attn_ for a user.

#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

// Pick the fastest of `n_cands` candidates. bench(i) runs candidate i's fixed
// work and returns wall-seconds; it is called INTERLEAVED across candidates
// each round so they share the clock/thermal state -- the per-round WINNER (min
// time) votes, which cancels the absolute clock (robust to DVFS / throttling;
// the same property that lets A/B ratios hold at -O0). Returns the index with
// the most round-wins (ties -> lowest index). Optionally reports the per-
// candidate mean us/call (pass the reps each bench() call ran) and win counts.
template <class Bench>
inline int
autotune_vote(int n_cands, int rounds, int reps_for_us, Bench&& bench,
              std::vector<double>* us_out = nullptr,
              std::vector<int>* wins_out = nullptr) {
  if (n_cands <= 0) { return 0; }
  const std::size_t n = (std::size_t)n_cands;
  std::vector<int> wins(n, 0);
  std::vector<double> sum(n, 0.0);
  int valid_rounds = 0;
  for (int i = 0; i < n_cands; ++i) { bench(i); }   // warm (bind + clock)
  std::vector<double> t(n);
  for (int r = 0; r < rounds; ++r) {
    int best = -1;
    double bt = 0.0;
    bool ok = true;
    for (int i = 0; i < n_cands; ++i) {
      t[(std::size_t)i] = bench(i);
      if (t[(std::size_t)i] <= 0.0) { ok = false; break; }
      if (best < 0 || t[(std::size_t)i] < bt) {
        bt = t[(std::size_t)i];
        best = i;
      }
    }
    if (!ok) { continue; }
    ++wins[(std::size_t)best];
    ++valid_rounds;
    for (int i = 0; i < n_cands; ++i) { sum[(std::size_t)i] += t[(std::size_t)i]; }
  }
  int w = 0;
  for (int i = 1; i < n_cands; ++i) {
    if (wins[(std::size_t)i] > wins[(std::size_t)w]) { w = i; }
  }
  if (us_out != nullptr) {
    us_out->assign(n, 0.0);
    if (valid_rounds > 0 && reps_for_us > 0) {
      for (int i = 0; i < n_cands; ++i) {
        (*us_out)[(std::size_t)i] =
            sum[(std::size_t)i] / valid_rounds / reps_for_us * 1e6;
      }
    }
  }
  if (wins_out != nullptr) { *wins_out = wins; }
  return w;
}

// Convenience: time `reps` invocations of one dispatch thunk in a fresh command
// stream and return wall-seconds. `Mc` is metal_compute::MetalCompute, `Enc` the
// ComputeEncoder; `disp(enc)` encodes one unit of work.
template <class Mc, class Disp>
inline double
autotune_time(Mc* mc, int reps, Disp&& disp) {
  const auto t0 = std::chrono::steady_clock::now();
  {
    auto st = mc->make_command_stream();
    {
      auto enc = st.begin_compute();
      for (int r = 0; r < reps; ++r) { disp(enc); }
    }
    st.commit().wait();
  }
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// Accumulates each tuner's wall-time + a one-line detail for a load-time
// summary (logged once via the session log delegate).
class TuningReport {
 public:
  void add(std::string name, double ms, std::string detail) {
    _total_ms += ms;
    _entries.push_back({std::move(name), ms, std::move(detail)});
  }
  bool empty() const { return _entries.empty(); }
  double total_ms() const { return _total_ms; }
  // e.g. "decode-attn 86ms (short/mid/long=vec2), prefill-attn 56ms (>=2k steel)"
  std::string summary() const {
    std::string s;
    for (std::size_t i = 0; i < _entries.size(); ++i) {
      if (i != 0) { s += ", "; }
      s += _entries[i].name + " " + fmt_ms(_entries[i].ms);
      if (!_entries[i].detail.empty()) {
        s += " (" + _entries[i].detail + ")";
      }
    }
    return s;
  }

 private:
  struct Entry {
    std::string name;
    double ms;
    std::string detail;
  };
  static std::string fmt_ms(double ms) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0fms", ms);
    return buf;
  }
  std::vector<Entry> _entries;
  double _total_ms = 0.0;
};

}  // namespace vpipe::genai
