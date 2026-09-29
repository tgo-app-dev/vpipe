// MotionCache (generative-models/shared/motion-cache.h) against the
// reference's own state machine.
//
// The goldens below are what ComfyUI-MiniMax-H3-MotionCache's
// MiniMaxH3MotionCacheState -- run VERBATIM, in torch float32 -- decided
// over a synthetic 20-step trajectory: a textured background with a blob
// moving across it, so the motion weights are far from uniform. This file
// rebuilds the same trajectory from the same formulas, packs it into
// MiniMax-H3's patch rows, and drives the port through H3's own gathers.
// So one comparison covers the arithmetic, the sign flip between the two
// velocity conventions, and the undoing of the packing -- any of which,
// got wrong, still produces a cache that runs and skips steps.
//
// The reference sees the video as [1, C, T, H, W] and the audio as
// [1, stereo, lanes, time]; its velocity is ComfyUI's, which is the
// NEGATIVE of H3's (x0 = x - sigma * out there, x0 = x + sigma * v here).

#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/motion-cache.h"
#include "minitest.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vpipe;
namespace accel = vpipe::genai::accel;
namespace mcache = vpipe::genai::motion_cache;

namespace {

// The trajectory's geometry. Latent 16 x 24 is an 8 x 12 patch grid.
constexpr int kC = 4, kT = 5, kH = 16, kW = 24;
constexpr int kAS = 2, kAC = 8, kAL = 40;
constexpr int kSteps = 20;
constexpr int kGH = kH / 2, kGW = kW / 2;
constexpr int kPE = kC * 4;
constexpr int kVRows = kT * kGH * kGW;
constexpr int kARows = kAS * kAL;

double
shift12(double t)
{
  return 12.0 * t / (1.0 + 11.0 * t);
}

// The formulas golden.py uses, term for term.
double
vx0(int c, int t, int h, int w)
{
  const double cx = 6.0 + 3.0 * t;
  const double blob =
      std::exp(-((w - cx) * (w - cx) + (h - 8.0) * (h - 8.0)) / 8.0);
  return 0.5 * std::sin(0.3 * h + 0.7 * c) * std::cos(0.2 * w) +
         1.5 * blob * (1 + 0.1 * c);
}
double
vn(int c, int t, int h, int w)
{
  return std::sin(1.7 * c + 2.3 * t + 0.9 * h + 1.3 * w + 0.5);
}
double
ax0(int s, int k, int l)
{
  return 0.4 * std::sin(0.25 * l + 0.6 * k + s);
}
double
an(int s, int k, int l)
{
  return std::cos(1.1 * s + 0.7 * k + 1.9 * l + 0.3);
}

std::size_t
vidx(int c, int t, int h, int w)
{
  const std::size_t row = (std::size_t)t * kGH * kGW +
                          (std::size_t)(h / 2) * kGW + (std::size_t)(w / 2);
  return row * kPE + ((std::size_t)c * 2 + (h % 2)) * 2 + (w % 2);
}
std::size_t
aidx(int s, int k, int l)
{
  return ((std::size_t)s * kAL + l) * kAC + k;
}

struct State {
  double sigma = 0.0;   // as the sampler holds it: f32
  std::vector<float> xv, vv, xa, va;
};

// Step i in H3's PACKED layout and H3's velocity sign.
State
state(int i)
{
  State st;
  const double s = shift12(1.0 - (double)i / kSteps);
  st.sigma = (double)(float)s;
  const double e = 0.04 * std::sin(0.9 * i);
  st.xv.assign((std::size_t)kVRows * kPE, 0.0f);
  st.vv.assign(st.xv.size(), 0.0f);
  for (int c = 0; c < kC; ++c) {
    for (int t = 0; t < kT; ++t) {
      for (int h = 0; h < kH; ++h) {
        for (int w = 0; w < kW; ++w) {
          const double x0 = vx0(c, t, h, w), n = vn(c, t, h, w);
          const double out =
              (n - x0) * (1 + e) + 0.03 * s * std::cos(x0 * 3 + i);
          st.xv[vidx(c, t, h, w)] = (float)((1 - s) * x0 + s * n);
          st.vv[vidx(c, t, h, w)] = -(float)out;
        }
      }
    }
  }
  st.xa.assign((std::size_t)kARows * kAC, 0.0f);
  st.va.assign(st.xa.size(), 0.0f);
  for (int sc = 0; sc < kAS; ++sc) {
    for (int k = 0; k < kAC; ++k) {
      for (int l = 0; l < kAL; ++l) {
        const double x0 = ax0(sc, k, l), n = an(sc, k, l);
        const double out =
            (n - x0) * (1 - e) + 0.02 * s * std::sin(x0 * 2 + i);
        st.xa[aidx(sc, k, l)] = (float)((1 - s) * x0 + s * n);
        st.va[aidx(sc, k, l)] = -(float)out;
      }
    }
  }
  return st;
}

genai::MetalMiniMaxH3Transformer::Config
tiny_cfg()
{
  genai::MetalMiniMaxH3Transformer::Config cfg;
  cfg.video_channels = kC;
  cfg.audio_channels = kAC;
  cfg.patch_t = 1;
  cfg.patch_h = 2;
  cfg.patch_w = 2;
  return cfg;
}

FlexData
bag(double threshold, double strength, int warmup = 4, int max_skips = 2,
    double start = 0.15, double end = 0.95, int subsample = 4)
{
  FlexData b = FlexData::make_object();
  accel::set_flag(&b, accel::kMotionCache, true);
  accel::set_real(&b, accel::kMotionCacheThreshold, threshold);
  accel::set_real(&b, accel::kMotionCacheStrength, strength);
  accel::set_integer(&b, accel::kMotionCacheWarmup, warmup);
  accel::set_integer(&b, accel::kMotionCacheMaxSkips, max_skips);
  accel::set_real(&b, accel::kMotionCacheStart, start);
  accel::set_real(&b, accel::kMotionCacheEnd, end);
  accel::set_integer(&b, accel::kMotionCacheSubsample, subsample);
  return b;
}

// Armed over H3's gathers for the trajectory.
bool
arm(mcache::Cache* c, const FlexData& b)
{
  const mcache::Config cfg = mcache::config_from_flex(&b);
  if (!c->begin(&b, mcache::percent_to_sigma(cfg.start, 12.0),
                mcache::percent_to_sigma(cfg.end, 12.0), 1.0f)) {
    return false;
  }
  genai::MotionCacheGathers g;
  std::string why;
  if (!genai::motion_cache_gathers(tiny_cfg(), kGH, kGW, kVRows, kARows,
                                   c->subsample(), &g, &why)) {
    std::printf("[motion_cache] gathers: %s\n", why.c_str());
    return false;
  }
  c->set_video(g.video.data(), g.channels, g.frames, g.plane,
               (std::size_t)kVRows * kPE);
  c->set_audio(g.audio.data(), g.audio.size(), (std::size_t)kARows * kAC);
  return true;
}

struct Golden {
  int    flags[kSteps];
  double scores[kSteps];   // -1: no score taken
  double accum[kSteps];
  // The reused output at video [c=1, t=2, h=4, w=5] and audio
  // [s=1, k=3, l=7], in the REFERENCE's sign, on reused steps only.
  struct Pred { int step; double v, a; };
  std::vector<Pred> pred;
};

struct Result {
  bool   armed = false;
  int    flag_mismatch = 0, score_mismatch = 0, pred_checked = 0;
  int    calls = 0;
  double worst_pred = 0.0;
};

Result
check_against(const Golden& g, double threshold, double strength)
{
  Result res;
  const FlexData b = bag(threshold, strength);
  mcache::Cache c;
  if (!arm(&c, b)) { return res; }
  res.armed = true;
  int flag_mismatch = 0, score_mismatch = 0, pred_checked = 0;
  double worst_score = 0.0, worst_pred = 0.0;
  for (int i = 0; i < kSteps; ++i) {
    const State st = state(i);
    const bool reuse = c.decide(st.sigma, st.xv.data(), st.xa.data());
    if ((int)reuse != g.flags[i]) {
      std::printf("[motion_cache] step %d: reuse %d, reference %d\n", i,
                  (int)reuse, g.flags[i]);
      ++flag_mismatch;
    }
    if (g.scores[i] < 0.0) {
      if (c.last_score() >= 0.0) { ++score_mismatch; }
    } else {
      const double r = std::fabs(c.last_score() - g.scores[i]) / g.scores[i];
      worst_score = std::max(worst_score, r);
      if (r > 1e-4) { ++score_mismatch; }
      const double ra = std::fabs(c.accumulated() - g.accum[i]);
      if (ra > 1e-4 * std::max(1e-3, g.accum[i])) { ++score_mismatch; }
    }
    if (reuse) {
      std::vector<float> v(st.vv.size()), a(st.va.size());
      c.predict(st.xv.data(), v.data(), st.xa.data(), a.data());
      for (const auto& p : g.pred) {
        if (p.step != i) { continue; }
        // H3's velocity is the reference's output negated.
        const double dv = std::fabs(-(double)v[vidx(1, 2, 4, 5)] - p.v);
        const double da = std::fabs(-(double)a[aidx(1, 3, 7)] - p.a);
        worst_pred = std::max(worst_pred, std::max(dv, da));
        ++pred_checked;
      }
    } else {
      c.update(st.sigma, st.xv.data(), st.vv.data(), st.xa.data(),
               st.va.data());
    }
  }
  std::printf("[motion_cache] threshold %.2f strength %.1f: %d of %d "
              "reused, flag mismatches %d, score mismatches %d (worst rel "
              "%.2e), %d reuses checked (worst %.2e)\n", threshold,
              strength, c.skipped(), c.calls(), flag_mismatch,
              score_mismatch, worst_score, pred_checked, worst_pred);
  res.flag_mismatch  = flag_mismatch;
  res.score_mismatch = score_mismatch;
  res.pred_checked   = pred_checked;
  res.worst_pred     = worst_pred;
  res.calls          = c.calls();
  return res;
}

}  // namespace

// The reference's decisions, scores and reused outputs, step for step,
// with the motion weighting on. At threshold 0.1 the THRESHOLD decides
// (the reuse runs are ragged, not the max_skips cadence) and the closest
// accumulated sum sits 0.0057 from it, far outside float noise.
TEST(motion_cache, matches_the_reference_step_for_step)
{
  Golden g = {
      {0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0},
      {-1, -1, -1, -1, 0.0344639616, 0.0730263652, 0.0422863754,
       0.0902801156, 0.03305462, 0.0712666888, 0.0295982885, 0.0646685283,
       0.106882705, 0.0438852407, 0.0989969374, 0.037754897, 0.0884880463,
       0.032026092, 0.0808277569, -1},
      {0, 0, 0, 0, 0.0344639616, 0, 0.0422863754, 0, 0.03305462, 0,
       0.0295982885, 0.0942668168, 0, 0.0438852407, 0, 0.037754897, 0,
       0.032026092, 0, 0},
      {{4, -1.2224526405334473, -0.0486111044883728},
       {6, -1.104668378829956, -0.08508762717247009},
       {8, -1.1721181869506836, -0.06664752960205078},
       {10, -1.2421636581420898, -0.044973552227020264},
       {11, -1.2237757444381714, -0.043950557708740234},
       {13, -1.0854721069335938, -0.08438637852668762},
       {15, -1.1575701236724854, -0.05334347486495972},
       {17, -1.1581673622131348, -0.05157071352005005}},
  };
  const Result r = check_against(g, 0.1, 1.0);
  ASSERT_TRUE(r.armed);
  EXPECT_TRUE(r.flag_mismatch == 0);
  EXPECT_TRUE(r.score_mismatch == 0);
  EXPECT_TRUE(r.pred_checked == (int)g.pred.size());
  EXPECT_TRUE(r.worst_pred < 2e-5);
  EXPECT_TRUE(r.calls == kSteps);
}

// The same with the motion weighting OFF -- a different decision
// sequence from the same data, which is what shows the weights above are
// live rather than all ones.
TEST(motion_cache, matches_the_reference_without_motion_weighting)
{
  Golden g = {
      {0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0},
      {-1, -1, -1, -1, 0.0385623636, 0.0817105127, 0.0453400916,
       0.0967997233, 0.0375588835, 0.0809780148, 0.0317806453,
       0.0694366932, 0.0487232394, 0.108497103, 0.0262478099, 0.0601949674,
       0.105811467, 0.0434616323, 0.109688879, -1},
      {0, 0, 0, 0, 0.0385623636, 0, 0.0453400916, 0, 0.0375588835, 0,
       0.0317806453, 0, 0.0487232394, 0, 0.0262478099, 0.0864427774, 0,
       0.0434616323, 0, 0},
      {{4, -1.2224526405334473, -0.0486111044883728},
       {6, -1.104668378829956, -0.08508762717247009},
       {8, -1.1721181869506836, -0.06664752960205078},
       {10, -1.2421636581420898, -0.044973552227020264},
       {12, -1.122422218322754, -0.07875579595565796},
       {14, -1.105666160583496, -0.0720876157283783},
       {15, -1.061563491821289, -0.06963399052619934},
       {17, -1.1581673622131348, -0.05157071352005005}},
  };
  const Result r = check_against(g, 0.1, 0.0);
  ASSERT_TRUE(r.armed);
  EXPECT_TRUE(r.flag_mismatch == 0);
  EXPECT_TRUE(r.score_mismatch == 0);
  EXPECT_TRUE(r.pred_checked == (int)g.pred.size());
  EXPECT_TRUE(r.worst_pred < 2e-5);
  EXPECT_TRUE(r.calls == kSteps);
}

// A reused velocity is the cached residual carried to the new input,
// EXACTLY: v' = (v + x) - x'. Not approximately -- this is the one piece
// of the cache with no tolerance to hide behind.
TEST(motion_cache, a_reuse_is_the_residual_carried_forward)
{
  const FlexData b = bag(1e9, 1.0, /*warmup=*/2, /*max_skips=*/1,
                         /*start=*/0.0, /*end=*/1.0);
  mcache::Cache c;
  ASSERT_TRUE(arm(&c, b));
  const State s0 = state(0), s1 = state(1), s2 = state(2);
  ASSERT_TRUE(!c.decide(s0.sigma, s0.xv.data(), s0.xa.data()));
  c.update(s0.sigma, s0.xv.data(), s0.vv.data(), s0.xa.data(),
           s0.va.data());
  ASSERT_TRUE(!c.decide(s1.sigma, s1.xv.data(), s1.xa.data()));
  c.update(s1.sigma, s1.xv.data(), s1.vv.data(), s1.xa.data(),
           s1.va.data());
  ASSERT_TRUE(c.decide(s2.sigma, s2.xv.data(), s2.xa.data()));
  std::vector<float> v(s2.vv.size()), a(s2.va.size());
  c.predict(s2.xv.data(), v.data(), s2.xa.data(), a.data());
  bool exact = true;
  for (std::size_t k = 0; k < v.size(); ++k) {
    const float want = (s1.vv[k] + s1.xv[k]) - s2.xv[k];
    if (v[k] != want) { exact = false; break; }
  }
  for (std::size_t k = 0; k < a.size(); ++k) {
    const float want = (s1.va[k] + s1.xa[k]) - s2.xa[k];
    if (a[k] != want) { exact = false; break; }
  }
  EXPECT_TRUE(exact);
}

// The schedule knobs, with the estimate taken out of it (a threshold no
// score reaches): warm-up forwards run, then reuse in runs of max_skips
// with a forward between -- and nothing outside the window.
TEST(motion_cache, warmup_window_and_the_run_cap)
{
  {
    const FlexData b = bag(1e9, 1.0, /*warmup=*/2, /*max_skips=*/2, 0.0,
                           1.0);
    mcache::Cache c;
    ASSERT_TRUE(arm(&c, b));
    std::string got;
    for (int i = 0; i < 12; ++i) {
      const State st = state(i);
      const bool r = c.decide(st.sigma, st.xv.data(), st.xa.data());
      got += r ? 'R' : 'C';
      if (!r) {
        c.update(st.sigma, st.xv.data(), st.vv.data(), st.xa.data(),
                 st.va.data());
      }
    }
    std::printf("[motion_cache] warm-up 2, cap 2: %s\n", got.c_str());
    EXPECT_TRUE(got == "CCRRCRRCRRCR");
    EXPECT_TRUE(c.skipped() == 7);
  }
  {
    // Warm-up below two changes nothing: a gain needs two forwards.
    const FlexData b = bag(1e9, 1.0, /*warmup=*/0, /*max_skips=*/3, 0.0,
                           1.0);
    mcache::Cache c;
    ASSERT_TRUE(arm(&c, b));
    std::string got;
    for (int i = 0; i < 8; ++i) {
      const State st = state(i);
      const bool r = c.decide(st.sigma, st.xv.data(), st.xa.data());
      got += r ? 'R' : 'C';
      if (!r) {
        c.update(st.sigma, st.xv.data(), st.vv.data(), st.xa.data(),
                 st.va.data());
      }
    }
    EXPECT_TRUE(got == "CCRRRCRR");
  }
  {
    // A window that closes at 0.5 of the range: every step past it runs.
    const FlexData b = bag(1e9, 1.0, 2, 2, 0.0, 0.5);
    mcache::Cache c;
    ASSERT_TRUE(arm(&c, b));
    const double end = mcache::percent_to_sigma(0.5, 12.0);
    int reused_past_end = 0;
    for (int i = 0; i < kSteps; ++i) {
      const State st = state(i);
      const bool r = c.decide(st.sigma, st.xv.data(), st.xa.data());
      if (r && st.sigma <= end) { ++reused_past_end; }
      if (!r) {
        c.update(st.sigma, st.xv.data(), st.vv.data(), st.xa.data(),
                 st.va.data());
      }
    }
    EXPECT_TRUE(c.skipped() > 0);
    EXPECT_TRUE(reused_past_end == 0);
  }
  {
    // Threshold 0 is "never": no sum is below it.
    const FlexData b = bag(0.0, 1.0, 2, 2, 0.0, 1.0);
    mcache::Cache c;
    ASSERT_TRUE(arm(&c, b));
    for (int i = 0; i < kSteps; ++i) {
      const State st = state(i);
      ASSERT_TRUE(!c.decide(st.sigma, st.xv.data(), st.xa.data()));
      c.update(st.sigma, st.xv.data(), st.vv.data(), st.xa.data(),
               st.va.data());
    }
    EXPECT_TRUE(c.skipped() == 0);
  }
}

// Off means inert: no bag, a bag that says off, an empty window.
TEST(motion_cache, off_is_inert)
{
  mcache::Cache c;
  EXPECT_TRUE(!c.begin(nullptr, 1.0, 0.0, 1.0f));
  EXPECT_TRUE(!c.armed());
  const State st = state(5);
  EXPECT_TRUE(!c.decide(st.sigma, st.xv.data(), st.xa.data()));
  EXPECT_TRUE(c.calls() == 0);

  FlexData b = bag(0.15, 1.0);
  accel::set_flag(&b, accel::kMotionCache, false);
  EXPECT_TRUE(!c.begin(&b, 1.0, 0.0, 1.0f));

  const FlexData empty = bag(0.15, 1.0, 4, 2, 0.6, 0.6);
  EXPECT_TRUE(!c.begin(&empty, 1.0, 0.0, 1.0f));

  // ...and the reader's defaults are the reference node's.
  const mcache::Config d = mcache::config_from_flex(nullptr);
  EXPECT_TRUE(!d.enabled && d.threshold == 0.15 &&
              d.motion_strength == 1.0 && d.warmup_steps == 4 &&
              d.max_skips == 2 && d.start == 0.15 && d.end == 0.95 &&
              d.subsample == 8);
  EXPECT_TRUE(accel::tiers_on(&b).empty());
  accel::set_flag(&b, accel::kMotionCache, true);
  const auto on = accel::tiers_on(&b);
  EXPECT_TRUE(on.size() == 1 && on[0] == accel::kMotionCache);
}

// The gathers undo H3's packing: fill every element with its own latent
// coordinates and read them back through the offsets.
TEST(motion_cache, the_gathers_undo_the_patch_packing)
{
  const int f = 3;   // not a divisor of the patch, on purpose
  genai::MotionCacheGathers g;
  std::string why;
  ASSERT_TRUE(genai::motion_cache_gathers(tiny_cfg(), kGH, kGW, kVRows,
                                          kARows, f, &g, &why));
  const int hs = (kH + f - 1) / f, ws = (kW + f - 1) / f;
  EXPECT_TRUE(g.channels == kC && g.frames == kT && g.plane == hs * ws);
  ASSERT_TRUE(g.video.size() == (std::size_t)kC * kT * hs * ws);
  std::vector<float> packed((std::size_t)kVRows * kPE, -1.0f);
  for (int c = 0; c < kC; ++c) {
    for (int t = 0; t < kT; ++t) {
      for (int h = 0; h < kH; ++h) {
        for (int w = 0; w < kW; ++w) {
          packed[vidx(c, t, h, w)] =
              (float)(((c * kT + t) * kH + h) * kW + w);
        }
      }
    }
  }
  int wrong = 0;
  std::size_t k = 0;
  for (int c = 0; c < kC; ++c) {
    for (int t = 0; t < kT; ++t) {
      for (int y = 0; y < hs; ++y) {
        for (int x = 0; x < ws; ++x, ++k) {
          const float want =
              (float)(((c * kT + t) * kH + y * f) * kW + x * f);
          if (packed[g.video[k]] != want) { ++wrong; }
        }
      }
    }
  }
  EXPECT_TRUE(wrong == 0);
  // Audio: every f-th step of each stereo channel, all lanes.
  const int al = (kAL + f - 1) / f;
  EXPECT_TRUE(g.audio.size() == (std::size_t)kAS * al * kAC);
  bool audio_ok = true;
  k = 0;
  for (int s = 0; s < kAS; ++s) {
    for (int l = 0; l < kAL; l += f) {
      for (int lane = 0; lane < kAC; ++lane, ++k) {
        if (g.audio[k] != (std::uint32_t)aidx(s, lane, l)) {
          audio_ok = false;
        }
      }
    }
  }
  EXPECT_TRUE(audio_ok);
  // No grid, or rows that are not whole frames: refused, and said why.
  EXPECT_TRUE(!genai::motion_cache_gathers(tiny_cfg(), 0, 0, kVRows, kARows,
                                           f, &g, &why));
  EXPECT_TRUE(!why.empty());
  EXPECT_TRUE(!genai::motion_cache_gathers(tiny_cfg(), kGH, kGW, kVRows - 1,
                                           kARows, f, &g, &why));
}
