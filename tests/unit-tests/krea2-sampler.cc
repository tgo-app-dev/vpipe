// Krea-2 sampler/scheduler refactor: the interchangeable FlowSampler (euler /
// heun / dpmpp_2m / dpmpp_sde) + FlowScheduler (simple / karras / exponential),
// the sampler-select + scheduler-select config stages, and text-to-image
// latching both specs off ports.
//
//  * specs round-trip through FlexData; method/type aliases + fallbacks.
//  * the simple euler schedule is bit-identical to the old baked-in turbo.
//  * karras / exponential are valid monotone schedules distinct from simple.
//  * every integrator drives a constant-target field to the target (terminal
//    invariant); dpmpp_2m differs from euler on a nonlinear field; dpmpp_sde is
//    deterministic given a seed.
//  * the select stages resolve defaults (built-in + from a model) + overrides.
//  * [env] end-to-end: sampler-select + scheduler-select wired into
//    text-to-image's ports yield a latent bit-identical to the default path.

#include "minitest.h"

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
#include "stages/sampler-select-stage.h"
#include "stages/scheduler-select-stage.h"
#include "stages/text-to-image-stage.h"
#include "stages/vae-decode-stage.h"

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

  // Unknown -> euler + error.
  FlexData bad = FlexData::make_object();
  bad.as_object().insert_or_assign("method", FlexData::make_string("nope"));
  std::string err;
  EXPECT_TRUE(FlowSamplerSpec::from_flex(bad, &err).method == "euler");
  EXPECT_TRUE(!err.empty());
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
    SamplerSelectStage st(&sess, "s", {}, FlexData::make_object());
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
    SamplerSelectStage st(&sess, "s", {}, std::move(cfg));
    FlexData fd = st.resolved_spec();
    auto o = fd.as_object();
    EXPECT_TRUE(std::string(o.at("method").as_string("")) == "dpmpp_sde");
    EXPECT_TRUE(o.at("eta").as_real(9.0) == 0.0);          // eta=0 honored
    EXPECT_TRUE(o.at("seed").as_int(0) == 123);
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

// ---- env-gated: model scheduler + end-to-end ports ----------------------

TEST(krea2_sampler, select_stages_read_model)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  FlexData scfg = FlexData::make_object();
  scfg.as_object().insert_or_assign("model", FlexData::make_string(root));
  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert_or_assign("model", FlexData::make_string(root));
  SamplerSelectStage samp(&sess, "s", {}, std::move(scfg));
  SchedulerSelectStage sched(&sess, "c", {}, std::move(ccfg));
  FlexData sfd = samp.resolved_spec(), cfd = sched.resolved_spec();
  auto so = sfd.as_object();
  auto co = cfd.as_object();
  std::printf("[krea2_sampler] model -> sampler %s / scheduler %s %.3f %s\n",
              std::string(so.at("method").as_string("")).c_str(),
              std::string(co.at("type").as_string("")).c_str(),
              co.at("shift").as_real(0.0),
              std::string(co.at("shift_type").as_string("")).c_str());
  EXPECT_TRUE(std::string(so.at("method").as_string("")) == "euler");
  EXPECT_TRUE(std::string(co.at("type").as_string("")) == "simple");
  EXPECT_TRUE(std::abs(co.at("shift").as_real(0.0) - 1.15) < 1e-6);
  EXPECT_TRUE(std::string(co.at("shift_type").as_string("")) == "exponential");
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

// Optionally wire sampler-select (port2) + scheduler-select (port3), both
// model-derived (euler + simple defaults). Returns the emitted latent.
std::vector<float>
run_t2i(Session& sess, const std::string& root, const std::string& gdir,
        bool wire_ports)
{
  auto pl = std::make_unique<Pipeline>("pl", &sess);
  auto su = std::make_unique<SrcText>(&sess, "src", std::vector<InEdge>{},
                                      FlexData::make_object());
  su->prompt = "a fox in the snow";
  auto* src = static_cast<SrcText*>(pl->insert_stage(std::move(su)));

  SamplerSelectStage* sel = nullptr;
  SchedulerSelectStage* sch = nullptr;
  if (wire_ports) {
    FlexData a = FlexData::make_object();
    a.as_object().insert_or_assign("model", FlexData::make_string(root));
    sel = static_cast<SamplerSelectStage*>(pl->insert_stage(
        std::make_unique<SamplerSelectStage>(&sess, "sel",
                                             std::vector<InEdge>{},
                                             std::move(a))));
    FlexData b = FlexData::make_object();
    b.as_object().insert_or_assign("model", FlexData::make_string(root));
    sch = static_cast<SchedulerSelectStage*>(pl->insert_stage(
        std::make_unique<SchedulerSelectStage>(&sess, "sch",
                                               std::vector<InEdge>{},
                                               std::move(b))));
  }

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert_or_assign(
      "init_latents", FlexData::make_string(gdir + "/a3_step0_latin.f32"));
  std::vector<InEdge> ip = wire_ports
      ? std::vector<InEdge>{{src, 0}, InEdge{nullptr, 0}, {sel, 0}, {sch, 0}}
      : std::vector<InEdge>{{src, 0}};
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(
      std::make_unique<TextToImageStage>(&sess, "t2i", std::move(ip),
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
  auto* sel = static_cast<SamplerSelectStage*>(pl->insert_stage(
      std::make_unique<SamplerSelectStage>(&sess, "sel",
                                           std::vector<InEdge>{}, std::move(sa))));
  FlexData sc = FlexData::make_object();
  sc.as_object().insert_or_assign("type", FlexData::make_string("karras"));
  sc.as_object().insert_or_assign("steps", FlexData::make_int(8));
  auto* sch = static_cast<SchedulerSelectStage*>(pl->insert_stage(
      std::make_unique<SchedulerSelectStage>(&sess, "sch",
                                             std::vector<InEdge>{}, std::move(sc))));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(rs));
  cfg.as_object().insert_or_assign("seed", FlexData::make_int(0));
  std::vector<InEdge> ip{{src, 0}, InEdge{nullptr, 0}, {sel, 0}, {sch, 0}};
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(
      std::make_unique<TextToImageStage>(&sess, "t2i", std::move(ip),
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
