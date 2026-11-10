// Beat tracking (stages/beat-tracker.h, stages/beat-track-stage.h)
// against librosa.beat.beat_track, restated.
//
// Goldens come from tools/dump_beat_track_golden.py, which restates
// librosa 0.11's beat tracker in numpy (read, not run): its numpy steps
// are librosa's own calls, so those values are librosa's to the bit, and
// its numba kernels are written out in float64 as librosa runs them.
//
//   VPIPE_BEAT_TRACK_GOLDEN   the dumper's --out root: one directory a
//       case (pcm.f32, env.f32, tg.f64, localscore.f64, cumscore.f64,
//       meta.json). Every case found is checked.
//
// The tests that need no golden (click tracks, the stage) always run.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/beat-track-stage.h"
#include "stages/beat-tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
namespace fs = std::filesystem;
using beat::Mode;
using beat::Params;
using beat::Tracker;

namespace {

const char*
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr && *v != '\0' ? v : nullptr;
}

template <typename T>
bool
read_all_(const fs::path& p, std::vector<T>* out)
{
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { return false; }
  const auto bytes = (std::size_t)f.tellg();
  f.seekg(0);
  out->resize(bytes / sizeof(T));
  f.read(reinterpret_cast<char*>(out->data()), (std::streamsize)bytes);
  return (bool)f;
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream f(p);
  if (!f) { return FlexData{}; }
  return FlexData::from_json(f);
}

double
num_(const FlexData& o, const char* k)
{
  if (!o.is_object() || !o.as_object().contains(k)) { return 0.0; }
  const FlexData v = o.as_object().at(k);
  if (v.is_real()) { return v.as_real(0.0); }
  if (v.is_int()) { return (double)v.as_int(0); }
  if (v.is_uint()) { return (double)v.as_uint(0); }
  return 0.0;
}

std::vector<std::int64_t>
ints_(const FlexData& o, const char* k)
{
  std::vector<std::int64_t> out;
  if (!o.is_object() || !o.as_object().contains(k)) { return out; }
  const FlexData a = o.as_object().at(k);
  if (!a.is_array()) { return out; }
  for (const FlexData& v : a.as_array()) { out.push_back(v.as_int(0)); }
  return out;
}

struct Case {
  std::string name;
  FlexData meta;
  std::vector<float> pcm, env;
  std::vector<double> tg, ls, cs;
};

std::vector<Case>
cases_()
{
  std::vector<Case> out;
  const char* root = env_("VPIPE_BEAT_TRACK_GOLDEN");
  if (root == nullptr) { return out; }
  std::vector<fs::path> dirs;
  for (const auto& e : fs::directory_iterator(root)) {
    if (fs::exists(e.path() / "meta.json")) { dirs.push_back(e.path()); }
  }
  std::sort(dirs.begin(), dirs.end());
  for (const fs::path& d : dirs) {
    Case c;
    c.name = d.filename().string();
    c.meta = read_json_(d / "meta.json");
    if (!read_all_(d / "pcm.f32", &c.pcm) ||
        !read_all_(d / "env.f32", &c.env) ||
        !read_all_(d / "tg.f64", &c.tg) ||
        !read_all_(d / "localscore.f64", &c.ls) ||
        !read_all_(d / "cumscore.f64", &c.cs)) {
      std::printf("  %s: unreadable\n", c.name.c_str());
      continue;
    }
    out.push_back(std::move(c));
  }
  return out;
}

double
max_rel_(const std::vector<double>& a, const std::vector<double>& b)
{
  if (a.size() != b.size()) { return 1e30; }
  double d = 0.0, m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::fabs(a[i] - b[i]));
    m = std::max(m, std::fabs(b[i]));
  }
  return m > 0.0 ? d / m : d;
}

std::vector<std::int64_t>
frames_(const std::vector<beat::Beat>& v)
{
  std::vector<std::int64_t> f;
  for (const auto& b : v) { f.push_back(b.frame); }
  return f;
}

// Push in chunks of `chunk` samples (0: all at once) and collect.
struct Run {
  std::vector<beat::Beat> beats;
  std::vector<std::int64_t> heard;   // frames heard when each came out
  double tempo = 0.0;
};

Run
run_(const std::vector<float>& pcm, Mode mode, std::size_t chunk,
     const Params& p = Params{})
{
  Tracker t(p, mode);
  Run r;
  auto take = [&] {
    for (const auto& b : t.take()) {
      r.beats.push_back(b);
      r.heard.push_back(t.frames());
    }
  };
  const std::size_t step = chunk == 0 ? pcm.size() : chunk;
  for (std::size_t i = 0; i < pcm.size(); i += step) {
    t.push(pcm.data() + i, std::min(step, pcm.size() - i));
    take();
  }
  t.finish();
  take();
  r.tempo = t.tempo();
  return r;
}

// Of `ref`'s beats, the share some beat of `got` is within `tol` frames
// of.
double
matched_(const std::vector<std::int64_t>& ref,
         const std::vector<std::int64_t>& got, std::int64_t tol)
{
  if (ref.empty()) { return got.empty() ? 1.0 : 0.0; }
  std::size_t hit = 0, j = 0;
  for (std::int64_t r : ref) {
    while (j < got.size() && got[j] < r - tol) { ++j; }
    if (j < got.size() && std::llabs(got[j] - r) <= tol) { ++hit; }
  }
  return (double)hit / (double)ref.size();
}

// The frames of `got` no beat of `ref` is within `tol` frames of.
std::vector<std::int64_t>
unmatched_(const std::vector<std::int64_t>& got,
           const std::vector<std::int64_t>& ref, std::int64_t tol)
{
  std::vector<std::int64_t> out;
  for (std::int64_t g : got) {
    bool hit = false;
    for (std::int64_t r : ref) { hit = hit || std::llabs(g - r) <= tol; }
    if (!hit) { out.push_back(g); }
  }
  return out;
}

// Decaying 1 kHz bursts at `bpm`, from `lead` seconds: the dumper's.
std::vector<float>
clicks_(double bpm, int sr, double seconds, double lead,
        std::vector<double>* times)
{
  const auto n = (std::size_t)(seconds * sr);
  std::vector<float> y(n, 0.0f);
  const int burst = (int)(0.03 * sr);
  for (double t = lead; t < seconds - 0.05; t += 60.0 / bpm) {
    const auto s = (std::size_t)std::llround(t * sr);
    for (int k = 0; k < burst && s + (std::size_t)k < n; ++k) {
      y[s + (std::size_t)k] += 0.5f * (float)(
          std::exp(-k / (0.006 * sr)) * std::sin(2.0 * M_PI * 1000.0 * k /
                                                 sr));
    }
    if (times != nullptr) { times->push_back(t); }
  }
  return y;
}

}  // namespace

// The back end alone: fed librosa's own onset envelope (beat_track's
// onset_envelope=), the tempo, local and cumulative scores and the beats
// are librosa's -- the same float64 sums in the same order.
TEST(beat_track, tracks_as_librosa_from_its_envelope)
{
  const auto cs = cases_();
  if (cs.empty()) {
    std::printf("  SKIP: VPIPE_BEAT_TRACK_GOLDEN unset or empty\n");
    return;
  }
  for (const Case& c : cs) {
    Tracker t(Params{}, Mode::kWhole);
    t.push_envelope(c.env.data(), c.env.size());
    t.finish();
    const auto got = frames_(t.take());
    const auto want = ints_(c.meta, "beats");
    const double rl = max_rel_(t.local_score(), c.ls);
    const double rc = max_rel_(t.cumulative_score(), c.cs);
    std::printf("  %-12s tempo %.4f / %.4f, bin %d / %d, std %.9g / %.9g, "
                "local %.2e, cumulative %.2e, beats %zu / %zu %s\n",
                c.name.c_str(), t.tempo(), num_(c.meta, "tempo"),
                t.tempo_bin(), (int)num_(c.meta, "bin"),
                (double)t.onset_std(), num_(c.meta, "std"), rl, rc,
                got.size(), want.size(), got == want ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(t.tempo() == num_(c.meta, "tempo"));
    EXPECT_TRUE(t.tempo_bin() == (int)num_(c.meta, "bin"));
    EXPECT_TRUE((double)t.onset_std() == num_(c.meta, "std"));
    EXPECT_TRUE(rl == 0.0);
    EXPECT_TRUE(rc == 0.0);
    EXPECT_TRUE(got == want);
  }
}

// From the samples: the envelope is librosa's to float32 rounding (its
// mel product is a BLAS sgemm, whose order is not reproducible); the
// tempo and the beats are librosa's.
TEST(beat_track, tracks_as_librosa_from_the_samples)
{
  const auto cs = cases_();
  if (cs.empty()) {
    std::printf("  SKIP: VPIPE_BEAT_TRACK_GOLDEN unset or empty\n");
    return;
  }
  for (const Case& c : cs) {
    Tracker t(Params{}, Mode::kWhole);
    t.push(c.pcm.data(), c.pcm.size());
    t.finish();
    const auto got = frames_(t.take());
    const auto want = ints_(c.meta, "beats");
    const auto& e = t.onset_envelope();
    double d = 0.0, m = 0.0, num = 0.0, den = 0.0;
    std::size_t off = 0;
    const bool same_n = e.size() == c.env.size();
    for (std::size_t i = 0; same_n && i < e.size(); ++i) {
      const double x = (double)e[i] - (double)c.env[i];
      d = std::max(d, std::fabs(x));
      m = std::max(m, std::fabs((double)c.env[i]));
      num += x * x;
      den += (double)c.env[i] * c.env[i];
      off += e[i] != c.env[i] ? 1 : 0;
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::printf("  %-12s frames %zu / %zu, envelope max %.2e of %.3g, "
                "rel-L2 %.2e, %zu frames off; tempo %.4f / %.4f, beats "
                "%zu / %zu %s (%.4f within 0)\n",
                c.name.c_str(), e.size(), c.env.size(), d, m, rel, off,
                t.tempo(), num_(c.meta, "tempo"), got.size(), want.size(),
                got == want ? "IDENTICAL" : "DIFFER",
                matched_(want, got, 0));
    EXPECT_TRUE(same_n);
    EXPECT_TRUE(rel < 1e-5);
    EXPECT_TRUE(t.tempo() == num_(c.meta, "tempo"));
    EXPECT_TRUE(got == want);
  }
}

// A stream that ends before its first decision is decided whole, as
// librosa would decide it: the same beats, to the frame.
TEST(beat_track, a_stream_that_ends_undecided_is_the_whole_track)
{
  for (const Case& c : cases_()) {
    if (c.pcm.size() >= 8 * 22050) { continue; }
    const Run s = run_(c.pcm, Mode::kStream, 1000);
    const Run w = run_(c.pcm, Mode::kWhole, 0);
    const auto want = ints_(c.meta, "beats");
    std::printf("  %-12s stream %zu beats, whole %zu, librosa %zu: %s\n",
                c.name.c_str(), s.beats.size(), w.beats.size(), want.size(),
                frames_(s.beats) == frames_(w.beats) ? "IDENTICAL"
                                                     : "DIFFER");
    EXPECT_TRUE(frames_(s.beats) == frames_(w.beats));
    EXPECT_TRUE(s.tempo == w.tempo);
  }
}

// The decisions follow the spectrogram's frames, not the input's cuts:
// one sample at a time, an odd size, a second, the whole at once --
// the same beats at the same tempi.
TEST(beat_track, a_stream_is_the_same_however_it_is_cut)
{
  std::vector<double> times;
  std::vector<float> y = clicks_(124.0, 22050, 24.0, 0.7, &times);
  for (const Case& c : cases_()) {
    if (c.name == "yue2-60s") { y = c.pcm; }
  }
  const Run a = run_(y, Mode::kStream, 0);
  bool same = !a.beats.empty();
  for (std::size_t chunk : {1u, 513u, 22050u}) {
    const Run b = run_(y, Mode::kStream, chunk);
    bool eq = b.beats.size() == a.beats.size();
    for (std::size_t i = 0; eq && i < a.beats.size(); ++i) {
      eq = a.beats[i].frame == b.beats[i].frame &&
           a.beats[i].bpm == b.beats[i].bpm;
    }
    std::printf("  chunk %5zu: %zu beats %s\n", chunk, b.beats.size(),
                eq ? "IDENTICAL" : "DIFFER");
    same = same && eq;
  }
  EXPECT_TRUE(same);
}

// On a whole song the stream decides on the way with what it has heard,
// and lands where librosa does: nearly every beat of the whole track's
// within a frame, handed out in order. Waiting for certainty most beats
// come out two to four seconds behind the sound; max_latency bounds the
// rest at a small cost in agreement.
TEST(beat_track, a_stream_follows_the_whole_track)
{
  const auto cs = cases_();
  if (cs.empty()) {
    std::printf("  SKIP: VPIPE_BEAT_TRACK_GOLDEN unset or empty\n");
    return;
  }
  const double fps = 22050.0 / 512.0;
  for (double cap : {0.0, 4.0}) {
    for (const Case& c : cs) {
      if (c.pcm.size() < 20 * 22050) { continue; }
      Params prm;
      prm.max_latency = cap;
      const Run s = run_(c.pcm, Mode::kStream, 4096, prm);
      const auto want = ints_(c.meta, "beats");
      const auto got = frames_(s.beats);
      bool ordered = true;
      for (std::size_t i = 1; i < got.size(); ++i) {
        ordered = ordered && got[i] > got[i - 1];
      }
      // How long each beat waited past the sound, leaving out the first
      // decision's (the warm-up's) and the end's.
      const std::int64_t n_frames = (std::int64_t)num_(c.meta, "frames");
      std::vector<double> lag;
      for (std::size_t i = 0; i < s.beats.size(); ++i) {
        if (s.heard[i] == s.heard[0] || s.heard[i] >= n_frames) { continue; }
        lag.push_back((double)(s.heard[i] - s.beats[i].frame) / fps);
      }
      std::sort(lag.begin(), lag.end());
      const double med = lag.empty() ? 0.0 : lag[lag.size() / 2];
      const double p95 = lag.empty() ? 0.0 : lag[lag.size() * 95 / 100];
      const double mx = lag.empty() ? 0.0 : lag.back();
      // The trim's threshold is the whole track's beats' in librosa and
      // the beats-so-far's here, so the two can keep or drop a beat
      // differently within a few seconds of either end; the agreement is
      // scored inside that.
      auto inner = [&](const std::vector<std::int64_t>& v) {
        std::vector<std::int64_t> o;
        for (auto f : v) {
          if (f >= 2.5 * fps && f <= n_frames - 2.5 * fps) { o.push_back(f); }
        }
        return o;
      };
      const double m0 = matched_(want, got, 0);
      const double m1 = matched_(inner(want), got, 1);
      const double back = matched_(inner(got), want, 1);
      std::printf("  %-12s max_latency %.0f: %zu beats (librosa %zu), %.3f "
                  "of librosa's on the frame, inside the ends %.3f within 1 "
                  "and %.3f of ours within 1 of one; waited median %.2f s, "
                  "95%% %.2f s, max %.2f s; first out %.2f s\n",
                  c.name.c_str(), cap, got.size(), want.size(), m0, m1, back,
                  med, p95, mx, s.heard.empty() ? 0.0 : s.heard[0] / fps);
      const auto extra = unmatched_(got, want, 1);
      const auto missed = unmatched_(want, got, 1);
      if (extra.size() + missed.size() <= 8) {
        std::string e, m;
        for (auto f : extra) { e += fmt(" {:.2f}", f / fps)(); }
        for (auto f : missed) { m += fmt(" {:.2f}", f / fps)(); }
        std::printf("    ours alone at [%s ] s, librosa's alone at [%s ] s "
                    "(of %.1f s)\n", e.c_str(), m.c_str(),
                    (double)n_frames / fps);
      }
      EXPECT_TRUE(ordered);
      EXPECT_TRUE(m1 >= (cap > 0.0 ? 0.97 : 0.98));
      EXPECT_TRUE(back >= (cap > 0.0 ? 0.96 : 0.98));
      // The cap holds to the input's chunk (4096 samples) and a frame.
      if (cap > 0.0) { EXPECT_TRUE(mx <= cap + 0.25); }
    }
  }
}

// Clicks at a tempo: every beat on a click (within the envelope's two-
// frame reach), nearly every click with a beat, the tempo the bin
// nearest the clicks' -- whole and streamed.
TEST(beat_track, click_tracks_beat_on_their_clicks)
{
  for (double bpm : {128.0, 93.0, 150.0}) {
    std::vector<double> times;
    const auto y = clicks_(bpm, 22050, 40.0, 0.9, &times);
    for (Mode mode : {Mode::kWhole, Mode::kStream}) {
      const Run r = run_(y, mode, 2048);
      std::size_t on = 0;
      bool last_on = true;
      for (const auto& b : r.beats) {
        const double t = (double)b.frame * 512.0 / 22050.0;
        double best = 1e9;
        for (double c : times) { best = std::min(best, std::fabs(t - c)); }
        on += best <= 2.0 * 512.0 / 22050.0 ? 1 : 0;
        if (&b == &r.beats.back()) { last_on = best <= 2.0 * 512.0 / 22050.0; }
      }
      const double fpb = 22050.0 / 512.0 * 60.0 / bpm;
      const double bin = 22050.0 / 512.0 * 60.0 / r.tempo;
      std::printf("  %5.1f bpm %-6s: tempo %.2f (period %.2f frames, "
                  "clicks' %.2f), %zu beats for %zu clicks, %zu on one%s\n",
                  bpm, mode == Mode::kWhole ? "whole" : "stream", r.tempo,
                  bin, fpb, r.beats.size(), times.size(), on,
                  last_on ? "" : " (not the last)");
      EXPECT_TRUE(std::fabs(bin - fpb) <= 0.5);
      // The last beat is the dynamic program's tail, which librosa picks
      // among the cumulative score's local maxima: on a grid a fraction
      // of a frame off the clicks' it can sit a few frames off the final
      // click (librosa's own answer -- the whole mode is it to the bit).
      EXPECT_TRUE(on + (last_on ? 0 : 1) == r.beats.size());
      EXPECT_TRUE(r.beats.size() + 3 >= times.size());
    }
  }
}

// Twenty minutes in a stream: every beat on a click, and the tracker
// holding a bounded window of its history however long it runs.
TEST(beat_track, a_long_stream_holds_a_window)
{
  std::vector<double> times;
  const auto y = clicks_(120.0, 22050, 1200.0, 0.4, &times);
  Params p;
  p.max_latency = 4.0;
  Tracker t(p, Mode::kStream);
  std::size_t n = 0, on = 0;
  std::int64_t most = 0;
  std::size_t ci = 0;
  auto take = [&] {
    for (const auto& b : t.take()) {
      const double s = (double)b.frame * 512.0 / 22050.0;
      while (ci + 1 < times.size() && times[ci + 1] <= s) { ++ci; }
      double best = std::fabs(times[ci] - s);
      if (ci + 1 < times.size()) {
        best = std::min(best, std::fabs(times[ci + 1] - s));
      }
      on += best <= 2.0 * 512.0 / 22050.0 ? 1 : 0;
      ++n;
    }
  };
  for (std::size_t i = 0; i < y.size(); i += 8192) {
    t.push(y.data() + i, std::min<std::size_t>(8192, y.size() - i));
    take();
    most = std::max(most, t.held_frames());
  }
  t.finish();
  take();
  std::printf("  %zu beats for %zu clicks, %zu on one; at most %lld frames "
              "held of %lld\n", n, times.size(), on, (long long)most,
              (long long)t.frames());
  EXPECT_TRUE(n + 3 >= times.size() && n <= times.size() + 1);
  EXPECT_TRUE(on + 1 >= n);
  EXPECT_TRUE(most < 12000);
}

namespace {

// The beats a stream hands out over `y`, pushed a second at a time, the
// most frames it held and, if asked, its cumulative scores at the end.
std::vector<beat::Beat>
stream_(const std::vector<float>& y, const Params& p, std::int64_t* most,
        std::vector<double>* cs = nullptr)
{
  Tracker t(p, Mode::kStream);
  std::vector<beat::Beat> out;
  for (std::size_t i = 0; i < y.size(); i += 22050) {
    t.push(y.data() + i, std::min<std::size_t>(22050, y.size() - i));
    for (const auto& b : t.take()) { out.push_back(b); }
  }
  t.finish();
  for (const auto& b : t.take()) { out.push_back(b); }
  *most = t.held_frames();
  if (cs != nullptr) { *cs = t.cumulative_score(); }
  return out;
}

bool
same_beats_(const std::vector<beat::Beat>& a,
            const std::vector<beat::Beat>& b)
{
  if (a.size() != b.size()) { return false; }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].frame != b[i].frame ||
        std::memcmp(&a[i].bpm, &b[i].bpm, sizeof(double)) != 0) {
      return false;
    }
  }
  return true;
}

// How many of `times` (seconds) from `t0` on have a beat within two
// frames.
std::size_t
clicks_hit_(const std::vector<beat::Beat>& v,
            const std::vector<double>& times, double t0)
{
  std::size_t hit = 0;
  for (double c : times) {
    if (c < t0) { continue; }
    for (const auto& b : v) {
      if (std::fabs((double)b.frame * 512.0 / 22050.0 - c) <=
          2.0 * 512.0 / 22050.0) {
        ++hit;
        break;
      }
    }
  }
  return hit;
}

}  // namespace

// A stream that opens with a long silence holds no more for it: the dB
// rows and the envelope's zeros go as they come, and the silence's part
// of the dynamic program is run at the first decision without being
// held (VPIPE_BEAT_TRACK_LEAD_DP=0 holds it, the same beats to the
// bit). And one that ends before deciding anything is still exactly
// `whole`, the silence's rows gone.
TEST(beat_track, a_leading_silence_holds_nothing)
{
  std::vector<double> times;
  const auto clicks = clicks_(128.0, 22050, 20.0, 0.5, &times);
  std::vector<float> y((std::size_t)22050 * 240, 0.0f);   // 4 minutes
  const std::size_t lead = y.size();
  y.insert(y.end(), clicks.begin(), clicks.end());
  for (double& c : times) { c += 240.0; }
  Params p;
  p.max_latency = 4.0;
  std::int64_t most_on = 0, most_off = 0;
  std::vector<double> cs_on, cs_off;
  const auto on = stream_(y, p, &most_on, &cs_on);
  setenv("VPIPE_BEAT_TRACK_LEAD_DP", "0", 1);
  const auto off = stream_(y, p, &most_off, &cs_off);
  unsetenv("VPIPE_BEAT_TRACK_LEAD_DP");
  // The silence's scores reach every later frame's through the chains
  // that start in it: the scores held at the end, the same to the bit.
  const std::size_t common = std::min(cs_on.size(), cs_off.size());
  const bool same_cs =
      common > 0 &&
      std::memcmp(cs_on.data() + (cs_on.size() - common),
                  cs_off.data() + (cs_off.size() - common),
                  common * sizeof(double)) == 0;
  // Every beat on a click. Not every click past the warm-up has one:
  // the deviation the stream divides by is still growing out of the
  // silence's zeros, so the first clicks score high, the trailing
  // trim's threshold with them, and the last seconds fall under it.
  std::size_t onc = 0;
  for (const auto& b : on) {
    for (double c : times) {
      if (std::fabs((double)b.frame * 512.0 / 22050.0 - c) <=
          2.0 * 512.0 / 22050.0) {
        ++onc;
        break;
      }
    }
  }
  const std::size_t hit = clicks_hit_(on, times, 240.0 + 9.0);
  std::size_t due = 0;
  for (double c : times) { due += c >= 249.0 ? 1 : 0; }
  std::printf("  %zu beats, %zu on a click; %zu of %zu clicks past the "
              "warm-up hit; at most %lld frames held (%lld without the "
              "lead's dynamic program) of %zu in the silence\n",
              on.size(), onc, hit, due, (long long)most_on,
              (long long)most_off, lead / 512);
  EXPECT_TRUE(same_beats_(on, off));
  EXPECT_TRUE(same_cs);
  EXPECT_TRUE(most_off > (std::int64_t)(lead / 512));
  EXPECT_TRUE(most_on < 8192);
  EXPECT_TRUE(onc == on.size() && hit * 2 >= due);
  // Five seconds of clicks after the silence: the stream ends undecided
  // and is decided whole.
  std::vector<float> s((std::size_t)22050 * 240, 0.0f);
  s.insert(s.end(), clicks.begin(), clicks.begin() + 22050 * 5);
  const beat::Result want = beat::beat_track(s.data(), s.size(), Params{});
  Tracker t(Params{}, Mode::kStream);
  for (std::size_t i = 0; i < s.size(); i += 22050) {
    t.push(s.data() + i, std::min<std::size_t>(22050, s.size() - i));
  }
  t.finish();
  const auto got = frames_(t.take());
  std::printf("  undecided: %zu beats against %zu, tempo %.4f against "
              "%.4f, at most %lld frames held\n", got.size(),
              want.beats.size(), t.tempo(), want.tempo,
              (long long)t.held_frames());
  EXPECT_TRUE(got == want.beats);
  EXPECT_TRUE(t.tempo() == want.tempo);
  EXPECT_TRUE(!want.beats.empty());
}

// Paths that do not meet -- a silence between two passages, where the
// tracker's candidate beats run side by side until the sound decides
// between them -- are committed on the best path once kHoldFrames
// (2^16) frames have passed, so a silence of any length holds a bounded
// window; the sound after it is tracked as before.
TEST(beat_track, a_long_inner_silence_holds_a_window)
{
  std::vector<double> t1, t2;
  const auto a = clicks_(120.0, 22050, 30.0, 0.5, &t1);
  const auto b = clicks_(120.0, 22050, 30.0, 0.5, &t2);
  const double gap = 30.0 * 60.0;                       // 77,519 frames
  std::vector<float> y = a;
  y.resize(y.size() + (std::size_t)(22050.0 * gap), 0.0f);
  const double t0 = (double)y.size() / 22050.0;
  y.insert(y.end(), b.begin(), b.end());
  for (double& c : t2) { c += t0; }
  for (double lat : {0.0, 4.0}) {
    Params p;
    p.max_latency = lat;
    std::int64_t most = 0;
    const auto v = stream_(y, p, &most);
    const std::size_t hit1 = clicks_hit_(v, t1, 9.0);
    const std::size_t hit2 = clicks_hit_(v, t2, t0 + 2.0);
    std::size_t due1 = 0, due2 = 0;
    for (double c : t1) { due1 += c >= 9.0 ? 1 : 0; }
    for (double c : t2) { due2 += c >= t0 + 2.0 ? 1 : 0; }
    std::printf("  max_latency %.0f: %zu beats; clicks hit %zu of %zu "
                "before, %zu of %zu after; at most %lld frames held\n",
                lat, v.size(), hit1, due1, hit2, due2, (long long)most);
    // The hold, the 4096-frame steps history is dropped in, and the
    // local score's reach.
    EXPECT_TRUE(most < (std::int64_t(1) << 16) + 4096 + 512);
    EXPECT_TRUE(hit1 + 1 >= due1);
    EXPECT_TRUE(hit2 + 1 >= due2);
  }
}

// No onset, no beats, tempo 0 -- librosa's answer for silence -- and a
// stream that starts silent decides nothing before the music and puts
// no beat in the silence.
TEST(beat_track, silence_has_no_beats)
{
  const std::vector<float> zero(22050 * 12, 0.0f);
  for (Mode mode : {Mode::kWhole, Mode::kStream}) {
    const Run r = run_(zero, mode, 4096);
    EXPECT_TRUE(r.beats.empty());
    EXPECT_TRUE(r.tempo == 0.0);
  }
  std::vector<double> times;
  auto y = std::vector<float>(22050 * 20, 0.0f);
  const auto c = clicks_(120.0, 22050, 30.0, 0.5, &times);
  y.insert(y.end(), c.begin(), c.end());
  const Run r = run_(y, Mode::kStream, 4096);
  const auto first = r.beats.empty() ? 0 : r.beats.front().frame;
  std::printf("  20 s of silence then clicks: %zu beats, the first at "
              "%.2f s\n", r.beats.size(), first * 512.0 / 22050.0);
  EXPECT_TRUE(!r.beats.empty());
  EXPECT_TRUE(first * 512.0 / 22050.0 >= 20.0);
}

// ---- The stage ----

namespace {

bool
have_avfilter_(Session& s)
{
  try {
    const FFmpegLibraries* l = s.ffmpeg_libraries();
    return l != nullptr && l->valid() && l->avfilter().valid();
  } catch (...) {
    return false;
  }
}

class PcmSource : public TypedStage<PcmSource> {
public:
  static constexpr const char* kTypeName = "ut-beat-track-source";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) {
      ctx.signal_done();
      co_return;
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(
                              TensorBeat(beats[next++])));
  }
};

class FlexSink : public TypedStage<FlexSink> {
public:
  static constexpr const char* kTypeName = "ut-beat-track-sink";
  using TypedStage::TypedStage;
  std::vector<FlexData> got;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(in.get())) {
      got.push_back(f->data);
    }
  }
};

// Planar [C, N] f32 at `rate`, cut into beats of `chunk` samples.
std::vector<TensorBeat>
pcm_beats_(const std::vector<std::vector<float>>& ch, int rate,
           std::size_t chunk)
{
  std::vector<TensorBeat> out;
  const std::size_t n = ch[0].size();
  const int C = (int)ch.size();
  for (std::size_t s = 0; s < n; s += chunk) {
    const std::size_t k = std::min(chunk, n - s);
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = C == 1 ? std::vector<std::int64_t>{(std::int64_t)k}
                      : std::vector<std::int64_t>{C, (std::int64_t)k};
    tb.resize_contiguous((std::size_t)C * k);
    float* d = tb.as_f32();
    for (int c = 0; c < C; ++c) {
      std::copy(ch[(std::size_t)c].begin() + (long)s,
                ch[(std::size_t)c].begin() + (long)(s + k),
                d + (std::size_t)c * k);
    }
    FlexData o = FlexData::make_object();
    o.as_object().insert_or_assign("sample_rate", FlexData::make_int(rate));
    o.as_object().insert_or_assign("timestamp_us",
                                   FlexData::make_uint(1000000));
    tb.sideband = std::move(o);
    out.push_back(std::move(tb));
  }
  return out;
}

struct StageRun {
  Session sess;
  std::unique_ptr<Pipeline> pl;
  FlexSink* sink = nullptr;
  BeatTrackStage* stage = nullptr;
};

bool
drive_(StageRun& r, std::vector<TensorBeat> beats, FlexData cfg)
{
  r.pl = std::make_unique<Pipeline>("p", &r.sess);
  auto src_u = std::make_unique<PcmSource>(&r.sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  src_u->beats = std::move(beats);
  auto* src = static_cast<PcmSource*>(r.pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  auto st_u = std::make_unique<BeatTrackStage>(
      &r.sess, "beats", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  r.stage = static_cast<BeatTrackStage*>(r.pl->insert_stage(std::move(st_u)));
  if (!r.stage->config_error().empty()) { return false; }
  auto sink_u = std::make_unique<FlexSink>(
      &r.sess, "sink", std::vector<InEdge>{{r.stage, 0}},
      FlexData::make_object());
  r.sink = static_cast<FlexSink*>(r.pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(r.pl.get(), &r.sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  return true;
}

FlexData
cfg_(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData o = FlexData::make_object();
  for (auto& p : kv) { o.as_object().insert_or_assign(p.first, p.second); }
  return o;
}

FlexData
field_(const FlexData& o, const char* k)
{
  if (!o.is_object() || !o.as_object().contains(k)) { return FlexData{}; }
  return o.as_object().at(k);
}

// Every object's beats, in order, as reals.
std::vector<double>
all_beats_(const std::vector<FlexData>& got, int channel = -1)
{
  std::vector<double> out;
  for (const FlexData& g : got) {
    if (channel >= 0 && field_(g, "channel").as_int(-1) != channel) {
      continue;
    }
    const FlexData a = field_(g, "beats");
    for (const FlexData& v : a.as_array()) { out.push_back(v.as_real(0.0)); }
  }
  return out;
}

}  // namespace

TEST(beat_track_stage, config_is_checked)
{
  Session s;
  auto mk = [&](FlexData c) {
    return BeatTrackStage(&s, "b", std::vector<InEdge>{}, std::move(c));
  };
  EXPECT_TRUE(mk(FlexData::make_object()).config_error().empty());
  EXPECT_TRUE(!mk(cfg_({{"units", FlexData::make_string("ms")}}))
                   .config_error().empty());
  EXPECT_TRUE(!mk(cfg_({{"mode", FlexData::make_string("live")}}))
                   .config_error().empty());
  EXPECT_TRUE(!mk(cfg_({{"group", FlexData::make_int(0)}}))
                   .config_error().empty());
  EXPECT_TRUE(!mk(cfg_({{"channels", FlexData::make_string("left")}}))
                   .config_error().empty());
  EXPECT_TRUE(mk(cfg_({{"channels", FlexData::make_string("1")}}))
                  .config_error().empty());
  EXPECT_TRUE(!mk(cfg_({{"max_latency", FlexData::make_real(-1.0)}}))
                   .config_error().empty());
  // At most 1000 envelope frames a second: 22050 / 16 is 1378.
  EXPECT_TRUE(!mk(cfg_({{"hop_length", FlexData::make_int(16)}}))
                   .config_error().empty());
  EXPECT_TRUE(mk(cfg_({{"hop_length", FlexData::make_int(32)}}))
                  .config_error().empty());
  Params p;
  p.hop_length = 16;
  EXPECT_TRUE(*Tracker(p, Mode::kStream).invalid() != '\0');
}

// The whole mode is beat_track on the channels' mean: one object of every
// beat, frames as librosa's, the time librosa.frames_to_time's.
TEST(beat_track_stage, whole_is_librosa_s_answer)
{
  std::vector<double> times;
  const auto a = clicks_(128.0, 22050, 30.0, 0.6, &times);
  auto b = a;
  for (float& x : b) { x *= 0.5f; }
  std::vector<float> mono(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    mono[i] = (float)((double)(a[i] + b[i]) / 2.0);
  }
  const beat::Result want = beat::beat_track(mono.data(), mono.size(),
                                             Params{});
  for (const char* units : {"frames", "time"}) {
    StageRun r;
    EXPECT_TRUE(drive_(r, pcm_beats_({a, b}, 22050, 4410),
                       cfg_({{"mode", FlexData::make_string("whole")},
                             {"units", FlexData::make_string(units)},
                             {"group", FlexData::make_int(1000)}})));
    EXPECT_TRUE(r.sink->got.size() == 1);
    if (r.sink->got.size() != 1) { continue; }
    const FlexData& g = r.sink->got[0];
    const auto got = all_beats_(r.sink->got);
    bool eq = got.size() == want.beats.size();
    for (std::size_t i = 0; eq && i < got.size(); ++i) {
      const double w = std::string(units) == "frames"
                           ? (double)want.beats[i]
                           : (double)(want.beats[i] * 512) / 22050.0;
      eq = got[i] == w;
    }
    std::printf("  %-6s: %zu beats, tempo %.4f / %.4f: %s\n", units,
                got.size(), field_(g, "tempo").as_real(0.0), want.tempo,
                eq ? "IDENTICAL" : "DIFFER");
    EXPECT_TRUE(eq);
    EXPECT_TRUE(field_(g, "tempo").as_real(0.0) == want.tempo);
    EXPECT_TRUE(field_(g, "index").as_uint(9) == 0);
    EXPECT_TRUE(field_(g, "units").as_string("") == std::string(units));
    EXPECT_TRUE(field_(g, "timestamp_us").as_uint(0) == 1000000);
    EXPECT_TRUE(!field_(g, "channel").is_int());
  }
}

// A 44.1 kHz stereo stream, heard at 22050 Hz as librosa.load would: the
// beats in the input's samples, on the clicks, `group` to an object with
// the last holding what is left, and the ordinals running on.
TEST(beat_track_stage, groups_in_the_input_s_samples)
{
  Session probe;
  if (!have_avfilter_(probe)) {
    std::printf("  SKIP: libavfilter not loadable\n");
    return;
  }
  std::vector<double> times;
  const auto y = clicks_(128.0, 44100, 40.0, 0.6, &times);
  StageRun r;
  EXPECT_TRUE(drive_(r, pcm_beats_({y, y}, 44100, 1411),
                     cfg_({{"units", FlexData::make_string("samples")},
                           {"group", FlexData::make_int(4)}})));
  const auto& got = r.sink->got;
  std::size_t total = 0, on = 0;
  bool shapes = !got.empty(), ordered = true;
  double prev = -1.0;
  for (std::size_t k = 0; k < got.size(); ++k) {
    const FlexData& g = got[k];
    const std::size_t n = field_(g, "beats").as_array().size();
    shapes = shapes && (n == 4 || (k + 1 == got.size() && n >= 1 && n <= 4));
    shapes = shapes && field_(g, "index").as_uint(0) == total;
    shapes = shapes && field_(g, "sample_rate").as_int(0) == 44100;
    shapes = shapes && field_(g, "analysis_rate").as_int(0) == 22050;
    total += n;
  }
  for (double s : all_beats_(got)) {
    ordered = ordered && s > prev;
    prev = s;
    double best = 1e18;
    for (double t : times) { best = std::min(best, std::fabs(s - t * 44100)); }
    // Two envelope frames: 2 x 512 samples at 22050 Hz.
    on += best <= 2.0 * 1024.0 ? 1 : 0;
  }
  std::printf("  %zu objects, %zu beats for %zu clicks, %zu on one; tempo "
              "%.2f\n", got.size(), total, times.size(), on,
              got.empty() ? 0.0 : field_(got.back(), "tempo").as_real(0.0));
  EXPECT_TRUE(shapes);
  EXPECT_TRUE(ordered);
  EXPECT_TRUE(total + 3 >= times.size());
  EXPECT_TRUE(on + 1 >= total);
  EXPECT_TRUE(r.stage->beats_out() == total);
}

// channels "each": a tracker per channel, each object naming its channel
// and carrying its tempo; and an index picks one.
TEST(beat_track_stage, each_channel_apart)
{
  std::vector<double> t0, t1;
  const auto a = clicks_(120.0, 22050, 30.0, 0.5, &t0);
  const auto b = clicks_(90.0, 22050, 30.0, 0.8, &t1);
  StageRun r;
  EXPECT_TRUE(drive_(r, pcm_beats_({a, b}, 22050, 8192),
                     cfg_({{"channels", FlexData::make_string("each")},
                           {"group", FlexData::make_int(8)}})));
  double tempo[2] = {0.0, 0.0};
  std::size_t n[2] = {0, 0};
  for (const FlexData& g : r.sink->got) {
    const int c = (int)field_(g, "channel").as_int(-1);
    if (c < 0 || c > 1) { continue; }
    tempo[c] = field_(g, "tempo").as_real(0.0);
    n[c] += field_(g, "beats").as_array().size();
  }
  std::printf("  channel 0: %zu beats at %.2f bpm; channel 1: %zu at %.2f\n",
              n[0], tempo[0], n[1], tempo[1]);
  EXPECT_TRUE(std::fabs(tempo[0] - 120.0) < 4.0);
  EXPECT_TRUE(std::fabs(tempo[1] - 90.0) < 3.0);
  EXPECT_TRUE(n[0] + 3 >= t0.size() && n[1] + 3 >= t1.size());

  StageRun one;
  EXPECT_TRUE(drive_(one, pcm_beats_({a, b}, 22050, 8192),
                     cfg_({{"channels", FlexData::make_string("1")},
                           {"mode", FlexData::make_string("whole")},
                           {"units", FlexData::make_string("frames")},
                           {"group", FlexData::make_int(1000)}})));
  const beat::Result want = beat::beat_track(b.data(), b.size(), Params{});
  const auto got = all_beats_(one.sink->got, 1);
  bool eq = got.size() == want.beats.size();
  for (std::size_t i = 0; eq && i < got.size(); ++i) {
    eq = got[i] == (double)want.beats[i];
  }
  EXPECT_TRUE(eq);
}
