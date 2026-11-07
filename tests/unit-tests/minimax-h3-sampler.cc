// MiniMax-H3's sampler and grid as a graph names them: a
// diffusion-sampler-select spec mapped onto the model's own scheduler
// steps (euler, dmd = re-noising, res_multistep), and the model config's
// `sigmas` -- the VIDEO's sigmas as run, ComfyUI's SIGMAS -- with the
// audio's mapped through the two shifts as ComfyUI's DiT maps them
// (comfy/ldm/minimax/model.py, time_shift_sigma).
//
//   VPIPE_MINIMAX_H3_TEST_MODEL_PATH  a MiniMax-H3 FL2VA checkpoint (a
//                                     root or the partition), for the
//                                     loop on two real blocks
//
// The spec, schedule and config tests need nothing.
#include "minitest.h"
#include "minimax-h3-block-run.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "stages/diffusion-sampler-select-stage.h"
#include "stages/minimax-h3-model-config-stage.h"

#include <malloc/malloc.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

FlexData
spec_of(const char* method)
{
  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("sampler", FlexData::make_string("flow_match"));
  if (method != nullptr) {
    o.insert_or_assign("method", FlexData::make_string(method));
  }
  return fd;
}

FlexData
config_of(const char* key, FlexData v)
{
  FlexData fd = FlexData::make_object();
  fd.as_object().insert_or_assign(key, std::move(v));
  return fd;
}

// ComfyUI's time_shift_sigma, in double: back to the base grid through
// `from`, out through `to`.
double
comfy_shift_(double sigma, double from, double to)
{
#pragma clang fp contract(off)   // Python-float ops, each rounded
  const double base = sigma / (from + sigma * (1.0 - from));
  return to * base / (1.0 + (to - 1.0) * base);
}

// The reporter's ComfyUI SIGMAS (issue #55), the video's as run.
const std::vector<double> kComfySigmas = {
    1.0, 0.9844, 0.9627, 0.9304, 0.878, 0.7579, 0.5527, 0.2369, 0.0};

}  // namespace

// The spec's method onto the model's own steps; everything else said and
// left to the default, as spec-keys.h asks of a reader.
TEST(minimax_h3_sampler, a_spec_names_one_of_its_own_steps)
{
  struct Case { const char* method; StepUpdate want; bool ok; };
  const Case cases[] = {
      {"euler", StepUpdate::kEuler, true},
      {"dmd", StepUpdate::kRenoise, true},
      {"dmd_student", StepUpdate::kRenoise, true},
      {"turbo", StepUpdate::kRenoise, true},
      {"res_multistep", StepUpdate::kResMultistep, true},
      {nullptr, StepUpdate::kEuler, true},     // no method: the spec's euler
      {"heun", StepUpdate::kAuto, false},
      {"dpmpp_2m", StepUpdate::kAuto, false},
      {"dpmpp_sde", StepUpdate::kAuto, false},
      {"dpm++_sde", StepUpdate::kAuto, false},
      {"unipc", StepUpdate::kAuto, false},
      {"er_sde", StepUpdate::kAuto, false},
  };
  for (const Case& c : cases) {
    SamplerChoice ch;
    std::string why;
    const bool ok = sampler_choice_from_spec(spec_of(c.method), &ch, &why);
    std::printf("  %-14s -> %d %-14s %s\n",
                c.method != nullptr ? c.method : "(none)", ok,
                step_update_name(ch.update), why.c_str());
    EXPECT_TRUE(ok == c.ok);
    EXPECT_TRUE(ch.update == c.want);
    if (!c.ok) {
      // The reason names the method and what does run.
      EXPECT_TRUE(why.find(c.method) != std::string::npos);
      EXPECT_TRUE(why.find("res_multistep") != std::string::npos);
    }
  }
  // Two forwards a step is why dpmpp_sde is out.
  {
    SamplerChoice ch;
    std::string why;
    sampler_choice_from_spec(spec_of("dpmpp_sde"), &ch, &why);
    EXPECT_TRUE(why.find("two") != std::string::npos);
  }
  // Not a spec at all.
  {
    SamplerChoice ch;
    std::string why;
    EXPECT_FALSE(sampler_choice_from_spec(FlexData::make_string("euler"),
                                          &ch, &why));
    EXPECT_TRUE(ch.update == StepUpdate::kAuto && !why.empty());
  }
  // The seed rides along; 0 is "none named".
  {
    FlexData fd = spec_of("dmd");
    fd.as_object().insert_or_assign("seed", FlexData::make_int(1234));
    SamplerChoice ch;
    EXPECT_TRUE(sampler_choice_from_spec(fd, &ch));
    EXPECT_TRUE(ch.seed == 1234 && ch.note.empty());
  }
  // A dmd conditioning_sigma shapes the DMD student's schedule; H3 runs
  // its own and says so.
  {
    FlexData fd = spec_of("dmd");
    fd.as_object().insert_or_assign("conditioning_sigma",
                                    FlexData::make_real(0.001));
    SamplerChoice ch;
    EXPECT_TRUE(sampler_choice_from_spec(fd, &ch));
    EXPECT_TRUE(ch.note.find("conditioning_sigma") != std::string::npos);
  }
  // And through the stage that writes the spec: res_multistep is one of
  // its methods now, and every one it emits maps as above.
  Session sess;
  struct StageCase { const char* method; StepUpdate want; bool ok; };
  const StageCase sc[] = {
      {"euler", StepUpdate::kEuler, true},
      {"dmd", StepUpdate::kRenoise, true},
      {"res_multistep", StepUpdate::kResMultistep, true},
      {"dpmpp_sde", StepUpdate::kAuto, false},
  };
  for (const StageCase& c : sc) {
    DiffusionSamplerSelectStage st(
        &sess, "s", {}, config_of("method", FlexData::make_string(c.method)));
    EXPECT_TRUE(st.config_error().empty());
    SamplerChoice ch;
    EXPECT_TRUE(sampler_choice_from_spec(st.resolved_spec(), &ch) == c.ok);
    EXPECT_TRUE(ch.update == c.want);
    // The stage emits seed 0 when none is set: the clip's seed runs.
    EXPECT_TRUE(ch.seed == 0);
  }
  {
    DiffusionSamplerSelectStage st(
        &sess, "s", {}, config_of("method", FlexData::make_string("er_sde")));
    EXPECT_FALSE(st.config_error().empty());
  }
}

// Sigmas named on the video's axis: installed as given there, carried to
// the audio's by ComfyUI's rule; refused unless they are a schedule.
TEST(minimax_h3_sampler, given_sigmas_run_as_given_and_map_to_the_audio)
{
  MiniMaxH3Scheduler sv(12.0), sa(3.0);
  ASSERT_TRUE(sv.set_sigmas_mapped(kComfySigmas, 12.0));
  ASSERT_TRUE(sa.set_sigmas_mapped(kComfySigmas, 12.0));
  ASSERT_TRUE(sv.sigmas().size() == kComfySigmas.size());
  ASSERT_TRUE(sa.sigmas().size() == kComfySigmas.size());
  int video_off = 0, audio_off = 0;
  for (std::size_t i = 0; i < kComfySigmas.size(); ++i) {
    video_off += sv.sigmas()[i] != (float)kComfySigmas[i];
    audio_off += sa.sigmas()[i] !=
                 (float)comfy_shift_(kComfySigmas[i], 12.0, 3.0);
  }
  std::printf("  video %s, audio %s: [", video_off == 0 ? "as given" : "OFF",
              audio_off == 0 ? "ComfyUI's mapping" : "OFF");
  for (float x : sa.sigmas()) { std::printf(" %.4f", (double)x); }
  std::printf(" ]\n");
  EXPECT_TRUE(video_off == 0 && audio_off == 0);
  EXPECT_TRUE(sv.num_inference_steps() == 8 &&
              sa.num_inference_steps() == 8);
  // The audio sits LOWER than the video at every inner point (shift 3
  // against 12), the ends shared.
  for (std::size_t i = 1; i + 1 < kComfySigmas.size(); ++i) {
    EXPECT_TRUE(sa.sigmas()[i] < sv.sigmas()[i]);
  }
  EXPECT_TRUE(sa.sigmas().front() == 1.0f && sa.sigmas().back() == 0.0f);
  // Timesteps are 1 - sigma, as on every other grid.
  EXPECT_TRUE(sv.timesteps()[3] == 1.0f - sv.sigmas()[3]);

  // The model's own grid, given back on the video's axis, lands the
  // audio on the grid the model makes for it: one base grid, two shifts.
  {
    MiniMaxH3Scheduler v0(12.0), a0(3.0);
    ASSERT_TRUE(v0.set_timesteps(9) && a0.set_timesteps(9));
    std::vector<double> own;
    for (float x : v0.sigmas()) { own.push_back((double)x); }
    MiniMaxH3Scheduler a1(3.0);
    ASSERT_TRUE(a1.set_sigmas_mapped(own, 12.0));
    double worst = 0.0;
    for (std::size_t i = 0; i < own.size(); ++i) {
      worst = std::max(worst, std::fabs((double)a1.sigmas()[i] -
                                        (double)a0.sigmas()[i]));
    }
    std::printf("  the model's own 8-step grid mapped: audio %.2e from "
                "its own\n", worst);
    EXPECT_TRUE(worst < 1e-6);
  }

  // Not a schedule: refused, and nothing installed.
  const std::vector<std::vector<double>> bad = {
      {1.0},                         // no terminal step
      {1.0, 0.5, 0.6, 0.0},          // not falling
      {1.0, 0.5, 0.5, 0.0},          // a repeat
      {1.1, 0.5, 0.0},               // above 1
      {1.0, 0.5, 0.01},              // not ending at 0
      {1.0, -0.1, 0.0},              // negative, not falling either
      {1.0, std::numeric_limits<double>::quiet_NaN(), 0.0},
  };
  for (const auto& b : bad) {
    MiniMaxH3Scheduler s(3.0);
    std::string why;
    EXPECT_FALSE(MiniMaxH3Scheduler::sigmas_ok(b, &why));
    EXPECT_TRUE(!why.empty());
    EXPECT_FALSE(s.set_sigmas_mapped(b, 12.0));
    EXPECT_TRUE(s.sigmas().empty());
  }
}

// The key as a graph writes it, through the config stage and back out of
// the beat generate-video reads.
TEST(minimax_h3_sampler, a_sigma_list_reads_as_a_graph_writes_it)
{
  std::vector<double> v;
  std::string why;
  EXPECT_TRUE(MiniMaxH3Scheduler::parse_sigmas("1, 0.9844, 0.5, 0", &v));
  EXPECT_TRUE(v == (std::vector<double>{1.0, 0.9844, 0.5, 0.0}));
  // A ComfyUI SIGMAS list pasted: brackets, no spaces, or no commas.
  EXPECT_TRUE(MiniMaxH3Scheduler::parse_sigmas("[1.0,0.5,0.0]", &v));
  EXPECT_TRUE(v == (std::vector<double>{1.0, 0.5, 0.0}));
  EXPECT_TRUE(MiniMaxH3Scheduler::parse_sigmas(" 0.9  0.4\n0 ", &v));
  EXPECT_TRUE(v == (std::vector<double>{0.9, 0.4, 0.0}));
  EXPECT_FALSE(MiniMaxH3Scheduler::parse_sigmas("1, x, 0", &v, &why));
  EXPECT_TRUE(v.empty() && why.find("'x") != std::string::npos);
  EXPECT_FALSE(MiniMaxH3Scheduler::parse_sigmas("1, 0.5", &v, &why));
  EXPECT_TRUE(v.empty() && why.find("0") != std::string::npos);
  EXPECT_FALSE(MiniMaxH3Scheduler::parse_sigmas("", &v, &why));

  using P = MetalMiniMaxH3Transformer::GenerationParams;
  // An array of numbers, ints among them.
  {
    FlexData a = FlexData::make_array();
    a.as_array().push_back(FlexData::make_int(1));
    a.as_array().push_back(FlexData::make_real(0.25));
    a.as_array().push_back(FlexData::make_int(0));
    std::string err;
    const P p = P::from_flex(config_of("sigmas", a), &err);
    EXPECT_TRUE(p.sigmas == (std::vector<double>{1.0, 0.25, 0.0}));
    EXPECT_TRUE(err.empty());
  }
  // The string form too.
  {
    const P p = P::from_flex(
        config_of("sigmas", FlexData::make_string("1, 0.2369, 0")));
    EXPECT_TRUE(p.sigmas == (std::vector<double>{1.0, 0.2369, 0.0}));
  }
  // Not a schedule: said, and the model's grid runs -- never half-taken.
  {
    std::string err;
    const P p = P::from_flex(
        config_of("sigmas", FlexData::make_string("1, 0.5, 0.7, 0")), &err);
    EXPECT_TRUE(p.sigmas.empty());
    EXPECT_TRUE(err.find("sigmas") != std::string::npos);
  }
  {
    FlexData a = FlexData::make_array();
    a.as_array().push_back(FlexData::make_real(1.0));
    a.as_array().push_back(FlexData::make_string("half"));
    a.as_array().push_back(FlexData::make_real(0.0));
    std::string err;
    const P p = P::from_flex(config_of("sigmas", a), &err);
    EXPECT_TRUE(p.sigmas.empty() && !err.empty());
  }
  // Absent: the model's own grid.
  EXPECT_TRUE(P::from_flex(FlexData::make_object()).sigmas.empty());

  // Through the config stage: emitted as numbers, read back the same.
  Session sess;
  {
    MiniMaxH3ModelConfigStage st(
        &sess, "h3", {},
        config_of("sigmas", FlexData::make_string(
                                "1, 0.9844, 0.9627, 0.9304, 0.878, 0.7579, "
                                "0.5527, 0.2369, 0")));
    EXPECT_TRUE(st.config_error().empty());
    const FlexData beat = st.resolved_config();
    ASSERT_TRUE(beat.as_object().contains("sigmas"));
    EXPECT_TRUE(beat.as_object().at("sigmas").is_array());
    const P p = P::from_flex(beat);
    EXPECT_TRUE(p.sigmas == kComfySigmas);
  }
  {
    MiniMaxH3ModelConfigStage st(&sess, "h3", {}, FlexData::make_object());
    EXPECT_TRUE(st.config_error().empty());
    EXPECT_FALSE(st.resolved_config().as_object().contains("sigmas"));
  }
  {
    MiniMaxH3ModelConfigStage st(
        &sess, "h3", {},
        config_of("sigmas", FlexData::make_string("1, 0.5, 0.25")));
    EXPECT_FALSE(st.config_error().empty());
  }
}

// The loop on two real blocks: each named update runs as asked, Euler
// named is the default, res_multistep named is the env A/B's, the
// re-noise follows its seed, and given sigmas set the step count and
// both schedules.
TEST(minimax_h3_sampler, the_loop_runs_the_named_step_on_the_given_sigmas)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (h3root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2));
  if (br.vb.empty()) { return; }
  auto m = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, false, {});
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
  const auto& c = br.cfg;
  const std::size_t nv =
      (std::size_t)br.L.video_indices.size() * c.video_patch_elems();
  const std::size_t na =
      (std::size_t)br.L.audio_indices.size() * c.audio_channels;
  struct Run {
    std::vector<float> v, a;
    DenoiseReport rep;
    bool ok = false;
    std::string err;
  };
  struct Ask {
    StepUpdate update = StepUpdate::kAuto;
    int steps = 3;
    std::uint64_t seed = 7;
    std::vector<double> sigmas;
    std::vector<double> refine;
  };
  auto run = [&](const Ask& ask) {
    Run r;
    r.v.resize(nv);
    r.a.resize(na);
    std::mt19937_64 g(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& x : r.v) { x = nd(g); }
    for (float& x : r.a) { x = nd(g); }
    DenoiseRequest q;
    q.dit = m.get();
    q.layout = &br.L;
    q.text = &br.tb;
    q.video = r.v.data();
    q.audio = r.a.data();
    q.video_grid_h = 12 / c.patch_h;
    q.video_grid_w = 20 / c.patch_w;
    q.num_steps = ask.steps;
    q.seed = ask.seed;
    q.update = ask.update;
    q.video_sigmas = ask.sigmas;
    q.refine_sigmas = ask.refine;
    q.report = &r.rep;
    r.ok = denoise(q, &r.err);
    if (!r.ok) { std::printf("  denoise: %s\n", r.err.c_str()); }
    return r;
  };
  auto same = [](const Run& x, const Run& y) {
    return x.v == y.v && x.a == y.a;
  };

  const Run dflt = run({});
  const Run euler = run({.update = StepUpdate::kEuler});
  const Run res = run({.update = StepUpdate::kResMultistep});
  ::setenv("VPIPE_H3_RES_MULTISTEP", "1", 1);
  const Run res_env = run({});
  ::unsetenv("VPIPE_H3_RES_MULTISTEP");
  const Run rn = run({.update = StepUpdate::kRenoise});
  const Run rn2 = run({.update = StepUpdate::kRenoise});
  const Run rn_s = run({.update = StepUpdate::kRenoise, .seed = 8});
  ASSERT_TRUE(dflt.ok && euler.ok && res.ok && res_env.ok && rn.ok &&
              rn2.ok && rn_s.ok);
  std::printf("  3 steps: euler named %s the default; res_multistep "
              "%.3e from Euler, %s the env A/B; renoise %.3e from Euler, "
              "same seed %s, another seed %.3e\n",
              same(euler, dflt) ? "IS" : "is NOT",
              rel_l2_(res.v, dflt.v),
              same(res, res_env) ? "IS" : "is NOT",
              rel_l2_(rn.v, dflt.v), same(rn, rn2) ? "IDENTICAL" : "DIFFERS",
              rel_l2_(rn_s.v, rn.v));
  EXPECT_TRUE(dflt.rep.update == StepUpdate::kEuler && !dflt.rep.renoise);
  EXPECT_TRUE(same(euler, dflt));
  EXPECT_TRUE(res.rep.update == StepUpdate::kResMultistep);
  EXPECT_TRUE(same(res, res_env));
  EXPECT_TRUE(rel_l2_(res.v, dflt.v) > 1e-4);
  EXPECT_TRUE(rn.rep.update == StepUpdate::kRenoise && rn.rep.renoise);
  EXPECT_TRUE(same(rn, rn2));
  EXPECT_TRUE(rel_l2_(rn.v, dflt.v) > 1e-3);
  EXPECT_TRUE(rel_l2_(rn_s.v, rn.v) > 1e-3 && rel_l2_(rn_s.a, rn.a) > 1e-3);
  for (const Run* r : {&dflt, &euler, &res, &rn}) {
    EXPECT_TRUE(r->rep.steps == 3 && r->rep.forwards == 3);
  }

  // Given sigmas: ComfyUI's eight, whatever `steps` says, the video on
  // them as given and the audio on ComfyUI's mapping.
  const Run given = run({.steps = 3, .sigmas = kComfySigmas});
  ASSERT_TRUE(given.ok);
  int voff = 0, aoff = 0;
  if (given.rep.video_sigmas.size() == kComfySigmas.size() &&
      given.rep.audio_sigmas.size() == kComfySigmas.size()) {
    for (std::size_t i = 0; i < kComfySigmas.size(); ++i) {
      voff += given.rep.video_sigmas[i] != (float)kComfySigmas[i];
      aoff += given.rep.audio_sigmas[i] !=
              (float)comfy_shift_(kComfySigmas[i], 12.0, 3.0);
    }
  } else {
    voff = aoff = -1;
  }
  std::printf("  given sigmas: %d steps, %d forwards, video %d off, audio "
              "%d off\n", given.rep.steps, given.rep.forwards, voff, aoff);
  EXPECT_TRUE(given.rep.steps == 8 && given.rep.forwards == 8);
  EXPECT_TRUE(voff == 0 && aoff == 0);

  // The model's own 8-step grid given back: the same video schedule, the
  // audio within float rounding of its own, and so (nearly) the same
  // clip.
  const Run own8 = run({.steps = 8});
  std::vector<double> own;
  for (float x : own8.rep.video_sigmas) { own.push_back((double)x); }
  const Run back = run({.sigmas = own});
  ASSERT_TRUE(own8.ok && back.ok);
  double aworst = 1.0;
  if (back.rep.audio_sigmas.size() == own8.rep.audio_sigmas.size()) {
    aworst = 0.0;
    for (std::size_t i = 0; i < own8.rep.audio_sigmas.size(); ++i) {
      aworst = std::max(aworst,
                        std::fabs((double)back.rep.audio_sigmas[i] -
                                  (double)own8.rep.audio_sigmas[i]));
    }
  }
  std::printf("  the model's own grid given back: video %s, audio %.2e "
              "off, clip %.3e / %.3e from the grid's own run\n",
              back.rep.video_sigmas == own8.rep.video_sigmas ? "IDENTICAL"
                                                             : "DIFFERS",
              aworst, rel_l2_(back.v, own8.v), rel_l2_(back.a, own8.a));
  EXPECT_TRUE(back.rep.video_sigmas == own8.rep.video_sigmas);
  EXPECT_TRUE(aworst < 1e-6);
  EXPECT_TRUE(rel_l2_(back.v, own8.v) < 1e-2);
  EXPECT_TRUE(rel_l2_(back.a, own8.a) < 1e-2);

  // A refine: its sigmas take the place of given ones, Euler named is
  // its plain Euler, and a named re-noise runs on it.
  const std::vector<double> rs = {0.9, 0.5, 0.0};
  const Run ref = run({.refine = rs});
  const Run ref_e = run({.update = StepUpdate::kEuler, .refine = rs});
  const Run ref_rn = run({.update = StepUpdate::kRenoise, .refine = rs});
  const Run ref_g = run({.sigmas = kComfySigmas, .refine = rs});
  ASSERT_TRUE(ref.ok && ref_e.ok && ref_rn.ok && ref_g.ok);
  std::printf("  refine: euler named %s its own, renoise %.3e from it, "
              "given sigmas under it %s (%d steps)\n",
              same(ref_e, ref) ? "IS" : "is NOT", rel_l2_(ref_rn.v, ref.v),
              same(ref_g, ref) ? "ignored" : "TAKEN", ref_g.rep.steps);
  EXPECT_TRUE(same(ref_e, ref) && ref.rep.steps == 2);
  EXPECT_TRUE(ref_rn.rep.renoise && rel_l2_(ref_rn.v, ref.v) > 1e-3);
  EXPECT_TRUE(same(ref_g, ref));
}

// Over an adapter's SAMPLING CONTRACT (Pika H3 Flex: six re-noised
// steps): a named Euler keeps the contract's grid and drops its update,
// as VPIPE_H3_RENOISE=0 does; given sigmas replace the contract whole,
// stepped by Euler unless the sampler names another.
//
//   VPIPE_PIKA_H3_FLEX_PATH  the release root (fl2va/adapter.safetensors
//                            + manifest.json beside it)
TEST(minimax_h3_sampler, a_named_step_or_grid_replaces_a_contract)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_PIKA_H3_FLEX_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_PIKA_H3_FLEX_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2));
  if (br.vb.empty()) { return; }
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  auto m = MetalMiniMaxH3Transformer::load(
      br.root, mc, br.cfg, false,
      {Spec{rel + "/fl2va/adapter.safetensors", 1.0f}});
  ASSERT_TRUE(m != nullptr && m->sampling_contract() != nullptr);
  if (m == nullptr || m->sampling_contract() == nullptr) { return; }
  m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
  const auto& c = br.cfg;
  const std::size_t nv =
      (std::size_t)br.L.video_indices.size() * c.video_patch_elems();
  const std::size_t na =
      (std::size_t)br.L.audio_indices.size() * c.audio_channels;
  struct Run {
    std::vector<float> v, a;
    DenoiseReport rep;
    bool ok = false;
  };
  auto run = [&](StepUpdate u, const std::vector<double>& sigmas) {
    Run r;
    r.v.resize(nv);
    r.a.resize(na);
    std::mt19937_64 g(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& x : r.v) { x = nd(g); }
    for (float& x : r.a) { x = nd(g); }
    DenoiseRequest q;
    q.dit = m.get();
    q.layout = &br.L;
    q.text = &br.tb;
    q.video = r.v.data();
    q.audio = r.a.data();
    q.video_grid_h = 12 / c.patch_h;
    q.video_grid_w = 20 / c.patch_w;
    q.num_steps = 3;
    q.seed = 7;
    q.update = u;
    q.video_sigmas = sigmas;
    q.report = &r.rep;
    std::string err;
    r.ok = denoise(q, &err);
    if (!r.ok) { std::printf("  denoise: %s\n", err.c_str()); }
    return r;
  };
  const Run con = run(StepUpdate::kAuto, {});
  const Run eul = run(StepUpdate::kEuler, {});
  ::setenv("VPIPE_H3_RENOISE", "0", 1);
  const Run env = run(StepUpdate::kAuto, {});
  ::unsetenv("VPIPE_H3_RENOISE");
  const Run giv = run(StepUpdate::kAuto, kComfySigmas);
  const Run giv_rn = run(StepUpdate::kRenoise, kComfySigmas);
  ASSERT_TRUE(con.ok && eul.ok && env.ok && giv.ok && giv_rn.ok);
  std::printf("  contract %d steps renoise %d; euler named %d steps "
              "renoise %d, %s VPIPE_H3_RENOISE=0; given sigmas %d steps "
              "renoise %d, + dmd renoise %d\n", con.rep.steps,
              con.rep.renoise, eul.rep.steps, eul.rep.renoise,
              eul.v == env.v && eul.a == env.a ? "IS" : "is NOT",
              giv.rep.steps, giv.rep.renoise, giv_rn.rep.renoise);
  EXPECT_TRUE(con.rep.steps == 6 && con.rep.renoise);
  EXPECT_TRUE(eul.rep.steps == 6 && !eul.rep.renoise);
  EXPECT_TRUE(eul.v == env.v && eul.a == env.a);
  EXPECT_TRUE(eul.rep.video_sigmas == con.rep.video_sigmas);
  EXPECT_TRUE(giv.rep.steps == 8 && !giv.rep.renoise);
  EXPECT_TRUE(giv_rn.rep.steps == 8 && giv_rn.rep.renoise);
  EXPECT_TRUE(giv.rep.video_sigmas.size() == kComfySigmas.size() &&
              giv.rep.video_sigmas[6] == (float)kComfySigmas[6]);
}

// What the loop holds on the host, MEASURED: the heap's peak growth over
// a denoise() -- sampled from a thread -- against denoise_host_bytes(),
// which generate-video's preflight now counts as pageable memory. A
// re-noised run (its fresh noise is the extra state) at 7,680 generated
// video rows, so the loop's buffers stand far above the allocator's
// noise.
TEST(minimax_h3_sampler, the_loop_holds_on_the_host_what_it_states)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (h3root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  // 8 latent frames of 48 x 80 latents: 24 x 40 patches a frame.
  ASSERT_TRUE(br.setup(mc, h3root, 2, 8, 48, 80, 40));
  if (br.vb.empty()) { return; }
  auto m = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, false, {});
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
  const auto& c = br.cfg;
  std::vector<float> v((std::size_t)br.L.video_indices.size() *
                       c.video_patch_elems());
  std::vector<float> a((std::size_t)br.L.audio_indices.size() *
                       c.audio_channels);
  std::mt19937_64 g(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (float& x : v) { x = nd(g); }
  for (float& x : a) { x = nd(g); }
  DenoiseRequest q;
  q.dit = m.get();
  q.layout = &br.L;
  q.text = &br.tb;
  q.video = v.data();
  q.audio = a.data();
  q.video_grid_h = 48 / c.patch_h;
  q.video_grid_w = 80 / c.patch_w;
  q.num_steps = 3;
  q.seed = 7;
  q.update = StepUpdate::kRenoise;
  auto in_use = []() {
    malloc_statistics_t st{};
    malloc_zone_statistics(nullptr, &st);
    return (std::int64_t)st.size_in_use;
  };
  const std::int64_t base = in_use();
  std::atomic<bool> done{false};
  std::atomic<std::int64_t> peak{base};
  std::thread sampler([&]() {
    while (!done.load()) {
      const std::int64_t u = in_use();
      if (u > peak.load()) { peak.store(u); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  std::string err;
  const bool ok = denoise(q, &err);
  done.store(true);
  sampler.join();
  ASSERT_TRUE(ok);
  const std::size_t stated = denoise_host_bytes(br.L, c, false, false);
  const double grew = (double)(peak.load() - base);
  std::printf("  %d generated video rows: the heap grew %.2f MB at its "
              "peak, stated %.2f MB\n", br.L.num_video_rows, grew / 1048576.0,
              (double)stated / 1048576.0);
  EXPECT_TRUE(br.L.num_video_rows >= 7000);
  // Covered, and not by a wide margin.
  EXPECT_TRUE(grew <= (double)stated * 1.05 + 1048576.0);
  EXPECT_TRUE(grew >= (double)stated * 0.5);
}
