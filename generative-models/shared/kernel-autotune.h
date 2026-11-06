#pragma once

// Reusable load-time kernel autotuning: a thermo-agnostic A/B/.../N picker plus
// a per-model tuning-time report. Header-only so any metal model (Qwen, Gemma,
// ...) can include it without a build-graph change. The picker only does the
// voting + timing bookkeeping; the caller supplies a `bench(i)` that runs
// candidate i's fixed GPU work and returns wall-seconds (it owns the buffers /
// command stream). See MetalQwenModel::autotune_decode_attn_ for a user.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe::genai {

// HOW MANY TIMED SAMPLES this binary's tuners have taken: every
// autotune_time / autotune_time_on call is one, and a tuner that times
// some other way (a forward's own command buffers) counts its samples
// with note_autotune_sample(). It is the evidence that a run asked to time
// nothing (the accel key `autotune` false, accel-keys.h) timed nothing: a
// test reads it before and after a load. Per binary -- a plugin, which
// compiles this header into itself, counts its own.
inline std::atomic<std::uint64_t>&
autotune_sample_counter_()
{
  static std::atomic<std::uint64_t> n{0};
  return n;
}
inline std::uint64_t
autotune_samples()
{
  return autotune_sample_counter_().load(std::memory_order_relaxed);
}
inline void
note_autotune_sample()
{
  autotune_sample_counter_().fetch_add(1, std::memory_order_relaxed);
}

// Pick the fastest of `n_cands` candidates. bench(i) runs candidate i's fixed
// work and returns wall-seconds; it is called INTERLEAVED across candidates
// each round so they share the clock/thermal state -- the per-round WINNER (min
// time) votes, which cancels the absolute clock (robust to DVFS / throttling;
// the same property that lets A/B ratios hold at -O0). Returns the index with
// the most round-wins; a TIE goes to the one with the least total time over
// the rounds, not to the lowest index -- with two candidates and two rounds a
// split vote is common (a timing here includes a fresh queue's first commit,
// 1-2 ms on a 5 ms sample), and the index rule handed it to whichever came
// first: MEASURED on Drama 3's prefill attention, qtile in 2 of 6 loads
// against a flash 8-10% faster on the mean. Optionally reports the per-
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
    const std::size_t a = (std::size_t)i, b = (std::size_t)w;
    if (wins[a] > wins[b] || (wins[a] == wins[b] && sum[a] < sum[b])) {
      w = i;
    }
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
  note_autotune_sample();
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

// The same on a stream the caller keeps for the whole tune. A fresh stream
// is a new queue, and its first commit costs 1-2 ms -- as much as the gap
// between two attention kernels at 768 rows -- so a tuner whose samples are
// a few ms reads that noise instead of the kernels: MEASURED on Drama 3's
// prefill attention, the slower kernel won 5 of 24 loads, each time moving
// the model's output (two kernels, two roundings).
template <class Stream, class Disp>
inline double
autotune_time_on(Stream& st, int reps, Disp&& disp) {
  note_autotune_sample();
  const auto t0 = std::chrono::steady_clock::now();
  {
    auto enc = st.begin_compute();
    for (int r = 0; r < reps; ++r) { disp(enc); }
  }
  st.commit().wait();
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// Run `reps` invocations of `disp` in a fresh stream and wait, timing
// nothing: for a tuner's CORRECTNESS gate, which compares the candidates'
// outputs and is not a sample (an untuned load still runs it).
template <class Mc, class Disp>
inline void
autotune_run(Mc* mc, int reps, Disp&& disp) {
  auto st = mc->make_command_stream();
  {
    auto enc = st.begin_compute();
    for (int r = 0; r < reps; ++r) { disp(enc); }
  }
  st.commit().wait();
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
