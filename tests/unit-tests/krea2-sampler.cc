// Krea-2 sampler/scheduler refactor: the interchangeable FlowSampler (euler /
// heun / dpmpp_2m / dpmpp_sde) + FlowScheduler (simple / karras / exponential),
// the diffusion-sampler-select + scheduler-select config stages, and
// generate-image latching both specs off ports.
//
//  * specs round-trip through FlexData; method/type aliases + fallbacks.
//  * the simple euler schedule is bit-identical to the old baked-in turbo.
//  * karras / exponential are valid monotone schedules distinct from simple.
//  * every integrator drives a constant-target field to the target (terminal
//    invariant); dpmpp_2m differs from euler on a nonlinear field; dpmpp_sde is
//    deterministic given a seed.
//  * the select stages resolve defaults (built-in + from a model) + overrides.
//  * [env] end-to-end: diffusion-sampler-select + scheduler-select wired into
//    generate-image's ports yield a latent bit-identical to the default path.

#include "minitest.h"

#include <filesystem>

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/krea2/flow-sampler.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-sampler-select-stage.h"
#include "stages/scheduler-select-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/vae-decode-stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using genai::FlowSampler;
using genai::FlowSamplerSpec;
using genai::FlowSchedulerSpec;

namespace {

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Run a whole sampler loop over a denoise field, from x0. Returns x_final.
std::vector<float>
run_loop(FlowSampler& s, std::vector<float> x, const FlowSampler::DenoiseFn& f)
{
  s.reset();
  for (int i = 0; i < s.steps(); ++i) { s.step(i, x, f); }
  return x;
}

}  // namespace

// ---- spec round-trips ---------------------------------------------------

TEST(krea2_sampler, sampler_spec_roundtrips_and_aliases)
{
  FlowSamplerSpec s;
  s.method = "dpmpp_sde"; s.eta = 0.5; s.s_noise = 1.2; s.seed = 42;
  EXPECT_TRUE(FlowSamplerSpec::from_flex(s.to_flex()) == s);

  // "dpm++_2m" alias canonicalizes.
  FlexData fd = FlexData::make_object();
  fd.as_object().insert_or_assign("method", FlexData::make_string("dpm++_2m"));
  EXPECT_TRUE(FlowSamplerSpec::from_flex(fd).method == "dpmpp_2m");

  // A method this host does not implement is PASSED THROUGH, and
  // reported. It used to be rewritten to "euler", which made the
  // request unrecoverable -- a registered family runs its own denoise
  // loop and reads this spec itself, so it can understand a name this
  // tree never heard of, but only if the name survives.
  //
  // The in-tree samplers are unaffected: they fall to euler for any
  // name they do not know, which is what they did with the rewrite too.
  FlexData bad = FlexData::make_object();
  bad.as_object().insert_or_assign("method", FlexData::make_string("nope"));
  std::string err;
  EXPECT_TRUE(FlowSamplerSpec::from_flex(bad, &err).method == "nope");
  EXPECT_TRUE(!err.empty());
  // ...and the sampler still steps, on euler, rather than refusing.
  FlowSchedulerSpec sc;
  sc.steps = 2;
  FlowSampler samp(FlowSamplerSpec::from_flex(bad), sc);
  std::vector<float> x = {1.0f};
  int calls = 0;
  samp.step(0, x, [&](const std::vector<float>&, double) {
    ++calls;
    return std::vector<float>{1.0f};
  });
  EXPECT_TRUE(calls == 1);
  EXPECT_TRUE(x[0] != 1.0f);
}

TEST(krea2_sampler, scheduler_spec_roundtrips)
{
  FlowSchedulerSpec s;
  s.type = "karras"; s.steps = 12; s.shift = 0.75; s.shift_type = "linear";
  s.rho = 5.0;
  EXPECT_TRUE(FlowSchedulerSpec::from_flex(s.to_flex()) == s);

  FlexData bad = FlexData::make_object();
  bad.as_object().insert_or_assign("type", FlexData::make_string("weird"));
  std::string err;
  EXPECT_TRUE(FlowSchedulerSpec::from_flex(bad, &err).type == "simple");
  EXPECT_TRUE(!err.empty());
}

// ---- schedules ----------------------------------------------------------

// Boogu-Image's schedule + the DMD student's spec. Its sigma convention is
// INVERTED (0 = noise, 1 = clean), so unlike every other schedule here the
// array ASCENDS and terminates at 1.
TEST(krea2_sampler, boogu_v1_schedule_ascends_and_dmd_spec_roundtrips)
{
  FlowSamplerSpec s;
  s.method = "dmd"; s.conditioning_sigma = 0.001; s.seed = 7;
  EXPECT_TRUE(FlowSamplerSpec::from_flex(s.to_flex()) == s);
  // Aliases canonicalize onto "dmd".
  FlexData fd = FlexData::make_object();
  fd.as_object().insert_or_assign("method", FlexData::make_string("turbo"));
  EXPECT_TRUE(FlowSamplerSpec::from_flex(fd).method == "dmd");
  fd.as_object().insert_or_assign("method", FlexData::make_string("lcm"));
  EXPECT_TRUE(FlowSamplerSpec::from_flex(fd).method == "dmd");

  FlowSchedulerSpec sc;
  sc.type = "boogu_v1"; sc.steps = 4; sc.seq_len = 4096;
  sc.base_shift = 0.5; sc.max_shift = 1.15;
  sc.base_seq = 256; sc.max_seq = 4096;
  EXPECT_TRUE(FlowSchedulerSpec::from_flex(sc.to_flex()) == sc);

  const std::vector<double> sig = sc.sigmas();
  EXPECT_TRUE(sig.size() == 5);
  // Starts at 0 (pure noise), ends at 1 (clean), strictly increasing. The
  // reference clips the flipped time to [1e-8, 1-1e-8] before the logistic, so
  // the first sigma is that eps pushed back through it, not a hard zero.
  EXPECT_TRUE(std::abs(sig[0]) < 1e-6);
  EXPECT_TRUE(std::abs(sig[4] - 1.0) < 1e-9);
  for (std::size_t i = 1; i < sig.size(); ++i) {
    EXPECT_TRUE(sig[i] > sig[i - 1]);
  }
  // seq_len 4096 sits at the top of the base->max shift line, so mu = max_shift
  // = 1.15; the logistic shift then pulls every interior step BELOW the plain
  // linspace it started from (more time spent near the noisy end).
  for (int i = 1; i < 4; ++i) {
    EXPECT_TRUE(sig[(std::size_t)i] < (double)i / 4.0);
  }
  // A shorter sequence (smaller mu) shifts less, so its steps sit higher.
  FlowSchedulerSpec small = sc;
  small.seq_len = 256;                     // mu = base_shift = 0.5
  const std::vector<double> ssig = small.sigmas();
  for (int i = 1; i < 4; ++i) {
    EXPECT_TRUE(ssig[(std::size_t)i] > sig[(std::size_t)i]);
  }
}

TEST(krea2_sampler, simple_schedule_matches_baked_turbo)
{
  FlowSchedulerSpec s;    // simple / 8 / 1.15 / exponential
  const std::vector<double> sig = s.sigmas();
  ASSERT_TRUE((int)sig.size() == 9);
  const int S = 8;
  const double mu = 1.15, emu = std::exp(mu);
  bool ok = true;
  for (int i = 0; i < S; ++i) {
    const double raw = 1.0 + (double)i * ((1.0 / S) - 1.0) / (double)(S - 1);
    const double want = emu / (emu + (1.0 / raw - 1.0));
    if (std::abs(sig[(std::size_t)i] - want) > 1e-12) { ok = false; }
  }
  EXPECT_TRUE(ok);
  EXPECT_TRUE(sig[8] == 0.0);
}

// ComfyUI's normal / sgm_uniform / beta / linear_quadratic, against a
// float64 restatement of comfy/samplers.py over its flow ModelSampling
// (1000 sigmas for the linear shift, 10000 for the exponential), the beta
// quantile from mpmath -- not this tree's: the reference is independent of
// the bisection it checks. Includes a long beta schedule whose repeats
// ComfyUI drops (600 asked, 559 run), which FlowSampler::steps() counts.
TEST(krea2_sampler, comfyui_schedules_match_their_restatement)
{
  struct Case {
    const char* type;
    int steps;
    double shift;
    const char* shift_type;
    double a, b;
    int n;
    std::vector<double> want;
  };
  const std::vector<Case> cases = {
    {"normal", 8, 3.0, "linear", 0.6, 0.6, 9,
     {1, 0.94754253308128544, 0.88278775079197469, 0.80083732057416268,
      0.69379310344827583, 0.54804560260586321, 0.33797216699801186,
      0.0089285714285713344, 0}},
    {"normal", 6, 12.0, "linear", 0.6, 0.6, 7,
     {1, 0.97988766080811751, 0.9483480688692415, 0.89177770555736102,
      0.76077586206896552, 0.1259842519685036, 0}},
    {"normal", 7, 1.15, "exponential", 0.6, 0.6, 8,
     {1, 0.94046538089033083, 0.86337637583567228, 0.75962624452162686,
      0.61249212723052449, 0.38756847423265151, 0.00099652394670915229, 0}},
    // a shift far under 1: its last timestep's sigma is within 1e-5 of 0,
    // so ComfyUI runs one point more and appends no 0
    {"normal", 6, 0.01, "linear", 0.6, 0.6, 7,
     {0.99999999999990941, 0.047619592381185273, 0.019608131776443515,
      0.0099011863544363303, 0.0049752730379225372, 0.0019961276250948391,
      1.0010009007998765e-07}},
    {"normal", 5, 1.0, "linear", 0.6, 0.6, 6,
     {1, 0.75024999999999997, 0.50049999999999994, 0.25075000000000003,
      0.0010000000000000009, 0}},
    {"sgm_uniform", 8, 3.0, "linear", 0.6, 0.6, 9,
     {1, 0.95469387755102031, 0.90035906642728902, 0.83399800598205387,
      0.7511210762331838, 0.64468629961587709, 0.50298507462686559,
      0.30500894454382821, 0}},
    {"sgm_uniform", 6, 1.15, "exponential", 0.6, 0.6, 7,
     {1, 0.94046538089033083, 0.86337637583567228, 0.75962624452162686,
      0.61249212723052449, 0.38756847423265151, 0}},
    {"beta", 8, 3.0, "linear", 0.6, 0.6, 9,
     {1, 0.98024948024948011, 0.93353474320241703, 0.85952177625960713,
      0.75074925074925092, 0.59638554216867468, 0.39217134416543564,
      0.15591397849462366, 0}},
    {"beta", 20, 1.15, "exponential", 0.5, 0.7, 21,
     {1, 0.99010398002589939, 0.97277860369940006, 0.95020565618944386,
      0.92262405162796379, 0.89002456428625054, 0.85214222235371684,
      0.80864680245269172, 0.75914541907976807, 0.70333947507019978,
      0.64093094118385896, 0.57218434329446, 0.49726981163295214,
      0.41739031748621275, 0.33450068228994151, 0.25180114398452424,
      0.17294503272678066, 0.10331938318148372, 0.047958616721044893,
      0.012524649202339705, 0}},
    {"beta", 30, 12.0, "linear", 0.6, 0.6, 31,
     {1, 0.99949723479135255, 0.99830220713073003, 0.99653979238754331,
      0.9944282302998142, 0.99172652059278121, 0.98844744998591161,
      0.98471721989681693, 0.98033650922359628, 0.97517504774029284,
      0.96935933147632303, 0.96259703599153146, 0.95485158393614367,
      0.94568690095846641, 0.93544249679350144, 0.92336046690216567,
      0.90890890890890896, 0.89252591380250945, 0.87242138994592422,
      0.84896872920825017, 0.82061446977205155, 0.78661087866108781,
      0.74403815580286159, 0.69247626004382756, 0.62944162436548223,
      0.54870775347912526, 0.45070422535211269, 0.33907649896623016,
      0.20471161657189277, 0.077994428969359333, 0}},
    {"beta", 600, 1.0, "linear", 0.6, 0.6, 560,
     {}},   // its length only: repeats dropped
    {"linear_quadratic", 8, 3.0, "linear", 0.6, 0.6, 9,
     {1, 0.99375000000000002, 0.98750000000000004, 0.98124999999999996,
      0.97500000000000009, 0.90937500000000004, 0.72500000000000031,
      0.42187500000000022, 0}},
    {"linear_quadratic", 9, 1.15, "exponential", 0.6, 0.6, 10,
     {1, 0.99375000000000002, 0.98750000000000004, 0.98124999999999996,
      0.97500000000000009, 0.93100000000000005, 0.81150000000000022,
      0.61650000000000016, 0.3460000000000002, 0}},
    {"linear_quadratic", 1, 3.0, "linear", 0.6, 0.6, 2,
     {1, 0}},
    {"linear_quadratic", 2, 3.0, "linear", 0.6, 0.6, 3,
     {1, 0.97500000000000009, 0}},
  };
  for (const Case& c : cases) {
    FlowSchedulerSpec s;
    s.type = c.type;
    s.steps = c.steps;
    s.shift = c.shift;
    s.shift_type = c.shift_type;
    s.beta_alpha = c.a;
    s.beta_beta = c.b;
    // through the spec's FlexData form, as a scheduler-select beat
    const FlowSchedulerSpec r = FlowSchedulerSpec::from_flex(s.to_flex());
    EXPECT_TRUE(r == s);
    const std::vector<double> got = r.sigmas();
    double worst = 0.0;
    const bool len_ok = (int)got.size() == c.n;
    for (std::size_t i = 0; len_ok && i < c.want.size(); ++i) {
      worst = std::max(worst, std::fabs(got[i] - c.want[i]));
    }
    std::printf("  %-16s %3d steps, shift %5.2f %-11s: %3zu points "
                "(want %d), worst %.2g\n", c.type, c.steps, c.shift,
                c.shift_type, got.size(), c.n, worst);
    EXPECT_TRUE(len_ok);
    EXPECT_TRUE(worst <= 1e-12);
    // every grid a schedule: strictly decreasing to 0 -- exactly, but for
    // that one case of normal's, within 1e-5
    bool dec = !got.empty() && std::fabs(got.back()) <= 1e-5;
    for (std::size_t i = 1; i < got.size(); ++i) {
      dec = dec && got[i] < got[i - 1];
    }
    EXPECT_TRUE(dec);
    if (c.want.empty()) {
      FlowSampler fs(FlowSamplerSpec{}, r);
      EXPECT_TRUE(fs.steps() == c.n - 1);
    }
  }
  // An unknown type is said and runs simple; the new ones are kept.
  FlexData fd = FlexData::make_object();
  fd.as_object().insert_or_assign("type", FlexData::make_string("ays_30"));
  std::string err;
  EXPECT_TRUE(FlowSchedulerSpec::from_flex(fd, &err).type == "simple");
  EXPECT_TRUE(!err.empty());
  for (const char* t : {"normal", "sgm_uniform", "beta",
                        "linear_quadratic"}) {
    fd.as_object().insert_or_assign("type", FlexData::make_string(t));
    err.clear();
    EXPECT_TRUE(FlowSchedulerSpec::from_flex(fd, &err).type == t);
    EXPECT_TRUE(err.empty());
  }
}

TEST(krea2_sampler, karras_and_exponential_are_valid_and_distinct)
{
  FlowSchedulerSpec sim; sim.type = "simple";
  FlowSchedulerSpec kar; kar.type = "karras";
  FlowSchedulerSpec exp; exp.type = "exponential";
  const auto a = sim.sigmas(), b = kar.sigmas(), c = exp.sigmas();
  ASSERT_TRUE(a.size() == b.size() && b.size() == c.size());

  // All share endpoints: sigma[0] = shift(1) = 1, terminal 0. Strictly
  // decreasing over the interior.
  for (const auto* v : {&a, &b, &c}) {
    EXPECT_TRUE(std::abs((*v)[0] - 1.0) < 1e-9);
    EXPECT_TRUE((*v)[v->size() - 1] == 0.0);
    bool dec = true;
    for (std::size_t i = 1; i + 1 < v->size(); ++i) {
      if ((*v)[i] >= (*v)[i - 1]) { dec = false; }
    }
    EXPECT_TRUE(dec);
  }
  // The three schedules are genuinely different curves.
  EXPECT_TRUE(rel_l2(std::vector<float>(b.begin(), b.end()),
                     std::vector<float>(a.begin(), a.end())) > 1e-3);
  EXPECT_TRUE(rel_l2(std::vector<float>(c.begin(), c.end()),
                     std::vector<float>(b.begin(), b.end())) > 1e-3);
}

// ---- integrators --------------------------------------------------------

TEST(krea2_sampler, all_integrators_hit_constant_target_at_terminal)
{
  // A field whose x0-prediction (denoised) is a constant target: velocity =
  // (x - target)/sigma. Every sampler must land on `target` at sigma=0.
  const std::vector<float> target = {0.3f, -1.1f, 2.0f, 0.5f};
  auto field = [&](const std::vector<float>& x, double sigma) {
    std::vector<float> v(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      v[k] = (float)(((double)x[k] - target[k]) / sigma);
    }
    return v;
  };
  FlowSchedulerSpec sched; sched.steps = 6;
  for (const char* m : {"euler", "heun", "dpmpp_2m", "dpmpp_sde"}) {
    FlowSamplerSpec sp; sp.method = m; sp.eta = 0.0;   // deterministic
    FlowSampler s(sp, sched);
    const std::vector<float> out =
        run_loop(s, std::vector<float>{5.0f, 5.0f, 5.0f, 5.0f}, field);
    const double r = rel_l2(out, target);
    std::printf("[krea2_sampler] %-9s constant-target rel-L2 = %.2e\n", m, r);
    EXPECT_TRUE(r < 1e-3);
  }
}

TEST(krea2_sampler, dpmpp_2m_differs_from_euler_on_nonlinear_field)
{
  // A nonlinear field so the 2nd-order multistep diverges from Euler.
  auto field = [&](const std::vector<float>& x, double sigma) {
    std::vector<float> v(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      v[k] = (float)(std::sin((double)x[k]) + 0.1 * sigma);
    }
    return v;
  };
  FlowSchedulerSpec sched; sched.steps = 8;
  FlowSamplerSpec e; e.method = "euler";
  FlowSamplerSpec d; d.method = "dpmpp_2m";
  FlowSampler se(e, sched), sd(d, sched);
  const std::vector<float> x0 = {0.7f, -0.4f, 1.3f};
  const std::vector<float> oe = run_loop(se, x0, field);
  const std::vector<float> od = run_loop(sd, x0, field);
  bool finite = true;
  for (float v : od) { if (!std::isfinite(v)) { finite = false; } }
  EXPECT_TRUE(finite);
  EXPECT_TRUE(rel_l2(od, oe) > 1e-3);
}

TEST(krea2_sampler, dpmpp_sde_is_deterministic_given_seed)
{
  auto field = [&](const std::vector<float>& x, double sigma) {
    std::vector<float> v(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      v[k] = (float)(0.5 * (double)x[k] + 0.2 * sigma);
    }
    return v;
  };
  FlowSchedulerSpec sched; sched.steps = 6;
  const std::vector<float> x0 = {1.0f, -2.0f, 0.5f, 3.0f};

  FlowSamplerSpec a; a.method = "dpmpp_sde"; a.eta = 1.0; a.seed = 7;
  FlowSampler s1(a, sched), s2(a, sched);
  const std::vector<float> o1 = run_loop(s1, x0, field);
  const std::vector<float> o2 = run_loop(s2, x0, field);
  EXPECT_TRUE(rel_l2(o1, o2) < 1e-12);          // same seed => identical

  FlowSamplerSpec b = a; b.seed = 99;
  FlowSampler s3(b, sched);
  const std::vector<float> o3 = run_loop(s3, x0, field);
  EXPECT_TRUE(rel_l2(o3, o1) > 1e-4);           // different seed => different

  FlowSamplerSpec c = a; c.eta = 0.0;
  FlowSampler s4a(c, sched), s4b(c, sched);
  EXPECT_TRUE(rel_l2(run_loop(s4a, x0, field),
                     run_loop(s4b, x0, field)) < 1e-12);  // eta=0 deterministic
}

// ---- select stages ------------------------------------------------------

TEST(krea2_sampler, sampler_select_resolves_defaults_and_overrides)
{
  Session sess;
  {
    DiffusionSamplerSelectStage st(&sess, "s", {}, FlexData::make_object());
    FlexData fd = st.resolved_spec();      // bind: as_object() is a view
    auto o = fd.as_object();
    EXPECT_TRUE(std::string(o.at("method").as_string("")) == "euler");
    EXPECT_TRUE(std::abs(o.at("eta").as_real(0.0) - 1.0) < 1e-9);
  }
  {
    FlexData cfg = FlexData::make_object();
    auto c = cfg.as_object();
    c.insert_or_assign("method", FlexData::make_string("dpm++_sde"));
    c.insert_or_assign("eta", FlexData::make_real(0.0));
    c.insert_or_assign("seed", FlexData::make_int(123));
    DiffusionSamplerSelectStage st(&sess, "s", {}, std::move(cfg));
    FlexData fd = st.resolved_spec();
    auto o = fd.as_object();
    EXPECT_TRUE(std::string(o.at("method").as_string("")) == "dpmpp_sde");
    EXPECT_TRUE(o.at("eta").as_real(9.0) == 0.0);          // eta=0 honored
    EXPECT_TRUE(o.at("seed").as_int(0) == 123);
  }
  {
    // ComfyUI's lcm is the dmd step: the stage emits it by that name
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("method", FlexData::make_string("lcm"));
    DiffusionSamplerSelectStage st(&sess, "s", {}, std::move(cfg));
    FlexData fd = st.resolved_spec();
    auto o = fd.as_object();
    EXPECT_TRUE(std::string(o.at("method").as_string("")) == "dmd");
  }
}

TEST(krea2_sampler, scheduler_select_resolves_defaults_and_overrides)
{
  Session sess;
  {
    SchedulerSelectStage st(&sess, "c", {}, FlexData::make_object());
    FlexData fd = st.resolved_spec();
    auto o = fd.as_object();
    EXPECT_TRUE(std::string(o.at("type").as_string("")) == "simple");
    EXPECT_TRUE(o.at("steps").as_int(0) == 8);
    EXPECT_TRUE(std::abs(o.at("shift").as_real(0.0) - 1.15) < 1e-9);
    EXPECT_TRUE(std::string(o.at("shift_type").as_string("")) == "exponential");
  }
  {
    FlexData cfg = FlexData::make_object();
    auto c = cfg.as_object();
    c.insert_or_assign("type", FlexData::make_string("karras"));
    c.insert_or_assign("steps", FlexData::make_int(20));
    c.insert_or_assign("shift", FlexData::make_real(2.5));
    c.insert_or_assign("rho", FlexData::make_real(5.0));
    SchedulerSelectStage st(&sess, "c", {}, std::move(cfg));
    FlexData fd = st.resolved_spec();
    auto o = fd.as_object();
    EXPECT_TRUE(std::string(o.at("type").as_string("")) == "karras");
    EXPECT_TRUE(o.at("steps").as_int(0) == 20);
    EXPECT_TRUE(std::abs(o.at("shift").as_real(0.0) - 2.5) < 1e-9);
    EXPECT_TRUE(std::abs(o.at("rho").as_real(0.0) - 5.0) < 1e-9);
  }
}

// ---- model-agnostic select stages + end-to-end ports --------------------

// The sampler/scheduler select stages forward the user's choice and DO NOT read
// the model (the generate-image stage owns it). With no config they emit the
// built-in distilled-turbo defaults, and a stray `model` key is simply ignored
// (it is no longer a declared config attr) -- so the model path can no longer
// silently change the emitted spec.
TEST(krea2_sampler, select_stages_are_model_agnostic)
{
  Session sess;
  // A stray `model` key must not change the emitted spec: build one pair with
  // it and one without, and require identical turbo defaults.
  FlexData scfg = FlexData::make_object();
  scfg.as_object().insert_or_assign("model",
                                    FlexData::make_string("/some/model/dir"));
  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert_or_assign("model",
                                    FlexData::make_string("/some/model/dir"));
  DiffusionSamplerSelectStage samp(&sess, "s", {}, std::move(scfg));
  SchedulerSelectStage sched(&sess, "c", {}, std::move(ccfg));
  DiffusionSamplerSelectStage samp0(&sess, "s0", {}, FlexData::make_object());
  SchedulerSelectStage sched0(&sess, "c0", {}, FlexData::make_object());
  FlexData sfd = samp.resolved_spec(), cfd = sched.resolved_spec();
  auto so = sfd.as_object();
  auto co = cfd.as_object();
  // The turbo defaults, unaffected by the (ignored) `model` key.
  EXPECT_TRUE(std::string(so.at("method").as_string("")) == "euler");
  EXPECT_TRUE(std::string(co.at("type").as_string("")) == "simple");
  EXPECT_TRUE(std::abs(co.at("shift").as_real(0.0) - 1.15) < 1e-6);
  EXPECT_TRUE(std::string(co.at("shift_type").as_string("")) == "exponential");
  // Identical to the no-`model` build (proves `model` is inert).
  EXPECT_TRUE(FlowSamplerSpec::from_flex(sfd) ==
              FlowSamplerSpec::from_flex(samp0.resolved_spec()));
  EXPECT_TRUE(FlowSchedulerSpec::from_flex(cfd) ==
              FlowSchedulerSpec::from_flex(sched0.resolved_spec()));
}

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

class SrcText : public TypedStage<SrcText> {
public:
  static constexpr const char* kTypeName = "ut-src-text-samp";
  SrcText(const SessionContextIntf* s, std::string id, std::vector<InEdge> ip,
          FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::string prompt;
  bool done = false;
  Job process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      co_await ctx.write(0, std::make_unique<FlexDataPayload>(
                                FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};
class SinkCap : public TypedStage<SinkCap> {
public:
  static constexpr const char* kTypeName = "ut-sink-samp";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// Chain a diffusion-conditioner between `src` (prompt) and the generate-image
// stage -- the encoder half moved there. Returns the conditioner.
Stage*
add_conditioner_(Pipeline* pl, Session& sess, Stage* src,
                 const std::string& root)
{
  FlexData c = FlexData::make_object();
  c.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  return pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(c)));
}

// Optionally wire diffusion-sampler-select (iport3) + scheduler-select
// (iport4), both model-agnostic (their built-in euler + simple defaults).
// Returns the emitted latent. Those defaults equal the config default path,
// so the two runs match.
std::vector<float>
run_t2i(Session& sess, const std::string& root, const std::string& gdir,
        bool wire_ports)
{
  auto pl = std::make_unique<Pipeline>("pl", &sess);
  auto su = std::make_unique<SrcText>(&sess, "src", std::vector<InEdge>{},
                                      FlexData::make_object());
  su->prompt = "a fox in the snow";
  auto* src = static_cast<SrcText*>(pl->insert_stage(std::move(su)));

  DiffusionSamplerSelectStage* sel = nullptr;
  SchedulerSelectStage* sch = nullptr;
  if (wire_ports) {
    sel = static_cast<DiffusionSamplerSelectStage*>(pl->insert_stage(
        std::make_unique<DiffusionSamplerSelectStage>(
            &sess, "sel", std::vector<InEdge>{},
            FlexData::make_object())));
    sch = static_cast<SchedulerSelectStage*>(pl->insert_stage(
        std::make_unique<SchedulerSelectStage>(&sess, "sch",
                                               std::vector<InEdge>{},
                                               FlexData::make_object())));
  }

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert_or_assign(
      "init_latents", FlexData::make_string(gdir + "/a3_step0_latin.f32"));
  auto* cond = add_conditioner_(pl.get(), sess, src, root);
  // t2i iports: {conditioning, neg, model, sampler, scheduler, ref0, ref1};
  // sampler/scheduler are iport3/iport4 (model iport2 left unwired here).
  std::vector<InEdge> ip = wire_ports
      ? std::vector<InEdge>{{cond, 0}, InEdge{nullptr, 0}, InEdge{nullptr, 0},
                            {sel, 0}, {sch, 0}}
      : std::vector<InEdge>{{cond, 0}};
  auto* t2i = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "t2i", std::move(ip),
                                         std::move(cfg))));
  if (!t2i->config_error().empty()) { return {}; }
  auto* sink = static_cast<SinkCap*>(pl->insert_stage(
      std::make_unique<SinkCap>(&sess, "sink", std::vector<InEdge>{{t2i, 0}},
                                FlexData::make_object())));

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  if (sink->captured.size() != 1) { return {}; }
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  if (tb == nullptr || tb->dtype != TensorBeat::DType::F32) { return {}; }
  const float* f = tb->as_f32();
  return std::vector<float>(f, f + tb->element_count());
}

}  // namespace

TEST(krea2_sampler, end_to_end_via_ports_matches_default)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::vector<float> a = run_t2i(sess, root, gd, /*wire_ports=*/false);
  const std::vector<float> b = run_t2i(sess, root, gd, /*wire_ports=*/true);
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  const double r = rel_l2(b, a);
  std::printf("[krea2_sampler] via-ports vs default latent rel-L2 = %.2e\n", r);
  EXPECT_TRUE(r < 1e-6);   // euler+simple ports == the config default path
}

// A NON-default combo (dpmpp_2m + karras) driven through the real DiT + VAE:
// confirms the whole path is finite + coherent (not just the euler bit-identity
// path). Writes /tmp/krea2_dpmpp_karras.ppm for eyeballing.
TEST(krea2_sampler, dpmpp_2m_karras_end_to_end)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string rs = root;
  const int H = 256, W = 256;

  auto pl = std::make_unique<Pipeline>("pl", &sess);
  auto su = std::make_unique<SrcText>(&sess, "src", std::vector<InEdge>{},
                                      FlexData::make_object());
  su->prompt = "a fox in the snow";
  auto* src = static_cast<SrcText*>(pl->insert_stage(std::move(su)));

  FlexData sa = FlexData::make_object();
  sa.as_object().insert_or_assign("method", FlexData::make_string("dpmpp_2m"));
  auto* sel = static_cast<DiffusionSamplerSelectStage*>(pl->insert_stage(
      std::make_unique<DiffusionSamplerSelectStage>(
          &sess, "sel", std::vector<InEdge>{}, std::move(sa))));
  FlexData sc = FlexData::make_object();
  sc.as_object().insert_or_assign("type", FlexData::make_string("karras"));
  sc.as_object().insert_or_assign("steps", FlexData::make_int(8));
  auto* sch = static_cast<SchedulerSelectStage*>(pl->insert_stage(
      std::make_unique<SchedulerSelectStage>(&sess, "sch",
                                             std::vector<InEdge>{}, std::move(sc))));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(rs));
  cfg.as_object().insert_or_assign("seed", FlexData::make_int(0));
  auto* cond = add_conditioner_(pl.get(), sess, src, rs);
  // sampler/scheduler are iport3/iport4 (model iport2 left unwired here).
  std::vector<InEdge> ip{{cond, 0}, InEdge{nullptr, 0}, InEdge{nullptr, 0},
                         {sel, 0}, {sch, 0}};
  auto* t2i = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "t2i", std::move(ip),
                                         std::move(cfg))));
  ASSERT_TRUE(t2i->config_error().empty());
  FlexData vcfg = FlexData::make_object();
  vcfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(rs));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(
      std::make_unique<VaeDecodeStage>(&sess, "vae", std::vector<InEdge>{{t2i, 0}},
                                       std::move(vcfg))));
  auto* sink = static_cast<SinkCap*>(pl->insert_stage(
      std::make_unique<SinkCap>(&sess, "sink", std::vector<InEdge>{{vae, 0}},
                                FlexData::make_object())));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr && tb->dtype == TensorBeat::DType::U8 &&
              tb->shape.size() == 3);
  const std::uint8_t* u = tb->as_u8();
  const std::size_t np = (std::size_t)3 * H * W;
  double mean = 0.0;
  for (std::size_t i = 0; i < np; ++i) { mean += u[i]; }
  mean /= (double)np;
  double var = 0.0;
  for (std::size_t i = 0; i < np; ++i) { const double d = u[i] - mean; var += d * d; }
  const double sd = std::sqrt(var / (double)np);
  std::printf("[krea2_sampler] dpmpp_2m+karras image std=%.1f\n", sd);
  EXPECT_TRUE(sd > 8.0);      // coherent, not flat/NaN
  {
    std::ofstream o("/tmp/krea2_dpmpp_karras.ppm", std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c)
          o.put((char)u[((std::size_t)c * H + y) * W + x]);
  }
}
#endif

// ---- the shift a checkpoint states for itself ------------------------
//
// Most families here can hardcode their schedule because it is a
// property of the architecture. Z-Image's cannot: its two published
// checkpoints ship byte-identical transformer configs and differ in
// this number alone, so a graph with no scheduler-select stage would
// take FlowSchedulerSpec's 1.15 default -- which is neither 3.0 nor
// 6.0, and produces a picture rather than an error. That is exactly
// what it did until this reader existed.
TEST(krea2_sampler, flow_shift_comes_off_the_checkpoint)
{
  namespace fs = std::filesystem;
  const fs::path d = fs::temp_directory_path() / "vpipe_flow_shift_test";
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d / "scheduler", ec);
  auto put = [&](const char* json) {
    std::ofstream o(d / "scheduler" / "scheduler_config.json");
    o << json;
  };
  bool dyn = true;

  // Z-Image-Turbo's own file.
  put(R"({"_class_name":"FlowMatchEulerDiscreteScheduler",)"
      R"("num_train_timesteps":1000,"use_dynamic_shifting":false,)"
      R"("shift":3.0})");
  EXPECT_TRUE(genai::flow_shift_from_config(d.string(), &dyn) == 3.0);
  EXPECT_FALSE(dyn);

  // ...and the undistilled base's, which is the whole point.
  put(R"({"_class_name":"FlowMatchEulerDiscreteScheduler",)"
      R"("use_dynamic_shifting":false,"shift":6.0})");
  EXPECT_TRUE(genai::flow_shift_from_config(d.string(), &dyn) == 6.0);

  // A checkpoint asking for a per-image mu says so, and a caller
  // applying a constant anyway should be able to know.
  put(R"({"use_dynamic_shifting":true,"shift":1.0})");
  EXPECT_TRUE(genai::flow_shift_from_config(d.string(), &dyn) == 1.0);
  EXPECT_TRUE(dyn);

  // Absent, unreadable and silent all answer 0 -- "use the family's
  // default" -- rather than a number that looks deliberate.
  put(R"({"num_train_timesteps":1000})");
  EXPECT_TRUE(genai::flow_shift_from_config(d.string(), &dyn) == 0.0);
  put("{not json");
  EXPECT_TRUE(genai::flow_shift_from_config(d.string(), &dyn) == 0.0);
  fs::remove_all(d, ec);
  EXPECT_TRUE(genai::flow_shift_from_config(d.string(), &dyn) == 0.0);
  EXPECT_TRUE(genai::flow_shift_from_config("", &dyn) == 0.0);
}
