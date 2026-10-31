// Pika H3 Flex (Pika-Labs/Pika-H3-Flex): two rank-128 adapters on
// MiniMax-H3's FL2VA and Ref2VA DiTs, each with a SAMPLING CONTRACT in a
// manifest beside it -- six steps on the knots 1000/750/500/250/100/40/0,
// shifted 12 (video) and 3 (audio) in double, and a RE-NOISING update:
// the clean estimate at each sigma put at the next with fresh noise.
//
//   VPIPE_PIKA_H3_FLEX_PATH           the release root (fl2va/, ref2va/,
//                                     each adapter.safetensors +
//                                     manifest.json)
//   VPIPE_MINIMAX_H3_TEST_MODEL_PATH  a MiniMax-H3 FL2VA checkpoint (a
//                                     root or the partition)
//
// The manifest, grid and step tests need neither.
#include "minitest.h"
#include "minimax-h3-block-run.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "pipeline/resource-plan.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/model-memory.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
namespace fs = std::filesystem;

namespace {

using Contract = MetalMiniMaxH3Transformer::SamplingContract;

// A manifest in a directory of its own, beside an adapter path that need
// not exist: the contract is read off the manifest alone.
struct ManifestDir {
  fs::path dir;
  explicit ManifestDir(const std::string& tag)
  {
    dir = fs::temp_directory_path() /
          ("vpipe-pika-manifest-" + tag + "-" +
           std::to_string((long long)std::random_device{}()));
    fs::create_directories(dir);
  }
  ~ManifestDir()
  {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string
  write(const std::string& json) const
  {
    std::ofstream(dir / "manifest.json") << json;
    return (dir / "adapter.safetensors").string();
  }
};

std::string
manifest_(const std::string& knots, const std::string& method,
          const std::string& policy, const std::string& trunk)
{
  return "{\"format_version\": 1, \"model\": \"pika-flex\", \"task\": "
         "\"fl2va\", \"release_id\": \"test-v1\", \"rank\": 128, "
         "\"alpha\": 128, \"trunk\": \"" + trunk + "\", "
         "\"numerics_policy\": \"" + policy + "\", \"sampling\": "
         "{\"knots\": " + knots + ", \"video_shift\": 12.0, "
         "\"audio_shift\": 3.0, \"method\": \"" + method + "\"}}";
}

std::uint32_t
bits_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

}  // namespace

// The manifest read as the release writes it, and refused -- not read as
// "no contract" -- wherever it says something this sampler cannot run.
TEST(pika_h3_flex, the_contract_is_read_off_the_manifest)
{
  const std::string knots = "[1000, 750, 500, 250, 100, 40, 0]";
  {
    ManifestDir d("ok");
    const std::string f = d.write(
        manifest_(knots, "renoise", "bf16_conditioning_v1", "transformer"));
    Contract c;
    std::string err;
    ASSERT_TRUE(MetalMiniMaxH3Transformer::sampling_contract_of(f, &c,
                                                                &err));
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(c.release == "test-v1" && c.task == "fl2va");
    EXPECT_TRUE(c.sigmas.size() == 7 && c.sigmas[0] == 1.0 &&
                c.sigmas[4] == 0.1 && c.sigmas[6] == 0.0);
    EXPECT_TRUE(c.video_shift == 12.0 && c.audio_shift == 3.0);
    EXPECT_TRUE(c.renoise && c.bf16_conditioning);
    EXPECT_TRUE(MetalMiniMaxH3Transformer::contract_partition(c) ==
                "fl2va");
  }
  {
    ManifestDir d("ref");
    const std::string f = d.write(manifest_(
        knots, "renoise", "fp32_conditioning_v1", "transformer_ref"));
    Contract c;
    std::string err;
    ASSERT_TRUE(MetalMiniMaxH3Transformer::sampling_contract_of(f, &c,
                                                                &err));
    EXPECT_TRUE(!c.bf16_conditioning);
    EXPECT_TRUE(MetalMiniMaxH3Transformer::contract_partition(c) ==
                "ref2va");
  }
  // No manifest, or one without `sampling`: no contract, no error.
  {
    ManifestDir d("none");
    const std::string f = (d.dir / "adapter.safetensors").string();
    Contract c;
    std::string err = "unset";
    EXPECT_TRUE(!MetalMiniMaxH3Transformer::sampling_contract_of(f, &c,
                                                                 &err));
    EXPECT_TRUE(err.empty());
    const std::string g = d.write("{\"rank\": 8, \"alpha\": 8}");
    EXPECT_TRUE(!MetalMiniMaxH3Transformer::sampling_contract_of(g, &c,
                                                                 &err));
    EXPECT_TRUE(err.empty());
  }
  // Refused, each with a reason.
  const struct {
    const char* what;
    std::string json;
  } bad[] = {
      {"knots not ending at 0",
       manifest_("[1000, 500, 10]", "renoise", "", "transformer")},
      {"knots not decreasing",
       manifest_("[1000, 500, 500, 0]", "renoise", "", "transformer")},
      {"knots past 1000",
       manifest_("[1200, 500, 0]", "renoise", "", "transformer")},
      {"an unknown method",
       manifest_(knots, "dpmpp", "", "transformer")},
      {"an unknown numerics policy",
       manifest_(knots, "renoise", "fp8_conditioning_v9", "transformer")},
      {"an unknown trunk",
       manifest_(knots, "renoise", "", "transformer_x")},
  };
  for (const auto& b : bad) {
    ManifestDir d("bad");
    const std::string f = d.write(b.json);
    Contract c;
    std::string err;
    const bool got =
        MetalMiniMaxH3Transformer::sampling_contract_of(f, &c, &err);
    std::printf("  %s: %s\n", b.what, err.c_str());
    EXPECT_TRUE(!got && !err.empty());
  }
}

// The six-step grid as diffusers' MiniMaxH3Scheduler builds it with
// `shift_provided_sigmas`: knots / 1000 shifted in DOUBLE, then rounded to
// float32 -- the expected bits computed outside this tree. The float32
// shift (set_sigmas, the flow-map grids' rule) lands an ulp off at the
// 0.04 knot under both shifts, which is what the double path is for.
TEST(pika_h3_flex, the_grid_is_the_release_s)
{
  const std::vector<double> raw = {1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0};
  const std::uint32_t video[] = {0x3f800000, 0x3f7914c2, 0x3f6c4ec5,
                                 0x3f4ccccd, 0x3f124925, 0x3eaaaaab, 0};
  const std::uint32_t audio[] = {0x3f800000, 0x3f666666, 0x3f400000,
                                 0x3f000000, 0x3e800000, 0x3de38e39, 0};
  for (int m = 0; m < 2; ++m) {
    MiniMaxH3Scheduler s(m == 0 ? 12.0 : 3.0);
    ASSERT_TRUE(s.set_sigmas_shifted(raw));
    const std::uint32_t* want = m == 0 ? video : audio;
    ASSERT_TRUE(s.sigmas().size() == 7 && s.num_inference_steps() == 6);
    int off = 0;
    for (std::size_t i = 0; i < 7; ++i) {
      if (bits_(s.sigmas()[i]) != want[i]) { ++off; }
      if (i < 6 && s.timesteps()[i] != 1.0f - s.sigmas()[i]) { ++off; }
    }
    EXPECT_TRUE(off == 0);
    MiniMaxH3Scheduler f(m == 0 ? 12.0 : 3.0);
    std::vector<float> rf(raw.begin(), raw.end());
    ASSERT_TRUE(f.set_sigmas(rf));
    std::printf("  shift %.0f: 0.04 -> %.9g (float32 shift %.9g)\n",
                m == 0 ? 12.0 : 3.0, s.sigmas()[5], f.sigmas()[5]);
    EXPECT_TRUE(f.sigmas()[5] != s.sigmas()[5]);
  }
  // diffusers' own checks on a grid it is handed.
  MiniMaxH3Scheduler s;
  EXPECT_TRUE(!s.set_sigmas_shifted({1.0, 0.5, 0.5, 0.0}));
  EXPECT_TRUE(!s.set_sigmas_shifted({1.0, 0.5, 0.1}));
  EXPECT_TRUE(!s.set_sigmas_shifted({1.5, 0.5, 0.0}));
}

// One re-noising step against its restatement: x0 off the GRID sigma, then
// (1 - s') x0 + s' n, every op in float32; the last step is x0 exactly
// and needs no noise; any other step refuses to run without it.
TEST(pika_h3_flex, a_renoise_step_puts_x0_at_the_next_sigma)
{
  MiniMaxH3Scheduler s(12.0);
  ASSERT_TRUE(s.set_sigmas_shifted({1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0}));
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const std::size_t n = 4099;
  std::vector<float> x(n), v(n), noise(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = nd(rng);
    v[i] = nd(rng);
    noise[i] = nd(rng);
  }
  int off = 0;
  for (int i = 0; i < 6; ++i) {
    const float sg = s.sigmas()[(std::size_t)i];
    const float sn = s.sigmas()[(std::size_t)i + 1];
    std::vector<float> want(n), got = x;
    for (std::size_t k = 0; k < n; ++k) {
      volatile float sv = sg * v[k];
      volatile float x0 = x[k] + sv;
      if (sn == 0.0f) {
        want[k] = x0;
        continue;
      }
      volatile float a = (1.0f - sn) * x0;
      volatile float b = sn * noise[k];
      want[k] = a + b;
    }
    const bool last = i == 5;
    ASSERT_TRUE(s.step_renoise(v.data(), i, got.data(),
                               last ? nullptr : noise.data(), n));
    for (std::size_t k = 0; k < n; ++k) {
      if (bits_(got[k]) != bits_(want[k])) { ++off; }
    }
    if (!last) {
      std::vector<float> y = x;
      EXPECT_TRUE(!s.step_renoise(v.data(), i, y.data(), nullptr, n));
    }
  }
  EXPECT_TRUE(off == 0);
}

// The release's two manifests: six re-noised steps on the same knots, the
// FL2VA adapter's keyframes in bf16, Ref2VA's in fp32, each on its DiT.
TEST(pika_h3_flex, the_release_s_manifests)
{
  const std::string root = env_("VPIPE_PIKA_H3_FLEX_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_PIKA_H3_FLEX_PATH unset\n");
    return;
  }
  for (const char* t : {"fl2va", "ref2va"}) {
    const std::string f = root + "/" + t + "/adapter.safetensors";
    Contract c;
    std::string err;
    ASSERT_TRUE(MetalMiniMaxH3Transformer::sampling_contract_of(f, &c,
                                                                &err));
    std::printf("  %s: %s, %zu knots, shifts %.0f/%.0f, renoise %d, bf16 "
                "conditioning %d, trunk %s\n", t, c.release.c_str(),
                c.sigmas.size(), c.video_shift, c.audio_shift, c.renoise,
                c.bf16_conditioning, c.trunk.c_str());
    EXPECT_TRUE(c.sigmas.size() == 7 && c.renoise);
    EXPECT_TRUE(c.video_shift == 12.0 && c.audio_shift == 3.0);
    EXPECT_TRUE(MetalMiniMaxH3Transformer::contract_partition(c) == t);
    EXPECT_TRUE(c.bf16_conditioning == (std::string(t) == "fl2va"));
  }
}

// The FL2VA adapter through two real blocks: it binds (q|k|v banded), it
// states its sampler while live and not at strength 0, where the forward
// is the base's to the bit; the Ref2VA adapter is REFUSED on the FL2VA
// DiT; and the plan books what the load holds.
TEST(pika_h3_flex, the_adapter_binds_and_states_its_sampler)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_PIKA_H3_FLEX_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_PIKA_H3_FLEX_PATH unset\n");
    return;
  }
  ::unsetenv("VPIPE_H3_LORA_RESIDENT");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2));
  if (br.vb.empty()) { return; }
  ASSERT_TRUE(br.cfg.partition == "fl2va");
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  const std::string fl = rel + "/fl2va/adapter.safetensors";
  const std::string rf = rel + "/ref2va/adapter.safetensors";
  std::vector<float> base, at0, at1;
  int mb = -1, m1 = -1;
  ASSERT_TRUE(br.run(mc, {}, &base, &mb));
  std::unique_ptr<MetalMiniMaxH3Transformer> keep;
  ASSERT_TRUE(br.run(mc, {Spec{fl, 1.0f}}, &at1, &m1, -1.0f, false,
                     nullptr, false, &keep));
  ASSERT_TRUE(keep != nullptr);
  if (keep == nullptr) { return; }
  const auto* live = keep->sampling_contract();
  const int banded = keep->lora_banded(0);
  keep->set_lora_scale(0, 0.0f);
  const bool off_at_0 = keep->sampling_contract() == nullptr;
  keep.reset();
  ASSERT_TRUE(br.run(mc, {Spec{fl, 0.0f}}, &at0, nullptr));
  const double moved = rel_l2_(at1, base);
  std::printf("  %d modules bound (%d q|k|v banded); contract %s at 1, %s "
              "at 0; scale 0 %s the base, scale 1 moves %.3e\n", m1, banded,
              live != nullptr ? "LIVE" : "absent",
              off_at_0 ? "off" : "STILL LIVE", at0 == base ? "IS" : "is NOT",
              moved);
  EXPECT_TRUE(m1 > 0 && live != nullptr && off_at_0);
  EXPECT_TRUE(banded > 0);
  EXPECT_TRUE(at0 == base);
  EXPECT_TRUE(moved > 1e-3);

  // Trained on the other DiT: refused, not run. The refusal goes through
  // the session's error(), which reports and throws.
  bool refused = false;
  std::string why;
  try {
    auto wrong = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, false,
                                                 {Spec{rf, 1.0f}});
    refused = wrong == nullptr;
  } catch (const std::exception& e) {
    refused = true;
    why = e.what();
  }
  EXPECT_TRUE(refused &&
              why.find("trained on the ref2va DiT") != std::string::npos);

  // Booked == held, resident and streamed.
  const auto fp = MetalMiniMaxH3Transformer::lora_footprint(fl, 2);
  auto pre = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, false,
                                             {Spec{fl, 1.0f}});
  ASSERT_TRUE(pre != nullptr);
  if (pre == nullptr) { return; }
  const std::size_t held_pre = pre->lora_bytes();
  pre.reset();
  auto str = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, true,
                                             {Spec{fl, 1.0f}});
  ASSERT_TRUE(str != nullptr);
  if (str == nullptr) { return; }
  const std::size_t held_str = str->lora_bytes();
  str.reset();
  const auto all = MetalMiniMaxH3Transformer::lora_footprint(fl);
  std::printf("  booked / held over 2 blocks: resident %zu / %zu, streamed "
              "%zu / %zu bytes; the whole file books %zu MB, %zu MB "
              "streamed, rank %d\n", fp.whole, held_pre, fp.streamed,
              held_str, all.whole >> 20, all.streamed >> 20, all.rank);
  EXPECT_TRUE(fp.whole == held_pre);
  EXPECT_TRUE(fp.streamed == held_str);
}

// The denoise loop runs the contract: six steps on its grid, re-noised,
// with the clip's seed deciding the fresh noise -- the same seed the same
// latents, another seed others; VPIPE_H3_RENOISE=0 steps the same grid by
// Euler (the A/B), an explicit grid wins over the contract, and a refine
// refuses it.
TEST(pika_h3_flex, the_denoise_runs_the_contract)
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
  const std::string fl = rel + "/fl2va/adapter.safetensors";
  auto m = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, false,
                                           {Spec{fl, 1.0f}});
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
  auto run = [&](std::uint64_t seed, const std::vector<float>* grid,
                 const std::vector<double>* refine) {
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
    q.num_steps = 3;   // not the contract's: it must not be used
    q.seed = seed;
    if (grid != nullptr) { q.sigma_grid = *grid; }
    if (refine != nullptr) { q.refine_sigmas = *refine; }
    q.report = &r.rep;
    r.ok = denoise(q, &r.err);
    return r;
  };
  const Run a = run(7, nullptr, nullptr);
  const Run b = run(7, nullptr, nullptr);
  const Run s = run(8, nullptr, nullptr);
  ::setenv("VPIPE_H3_RENOISE", "0", 1);
  const Run e = run(7, nullptr, nullptr);
  ::unsetenv("VPIPE_H3_RENOISE");
  const std::vector<float> grid = {1.0f, 0.5f, 0.0f};
  const Run x = run(7, &grid, nullptr);
  const std::vector<double> rs = {0.9, 0.5, 0.0};
  const Run r = run(7, nullptr, &rs);
  ASSERT_TRUE(a.ok && b.ok && s.ok && e.ok && x.ok);
  std::printf("  contract: %d steps, %d forwards, renoise %d; same seed "
              "%s, another seed moves %.3e, Euler on the grid moves "
              "%.3e (renoise %d); an explicit grid runs %d steps "
              "(renoise %d); a refine: %s\n", a.rep.steps, a.rep.forwards,
              a.rep.renoise,
              a.v == b.v && a.a == b.a ? "IDENTICAL" : "DIFFERS",
              rel_l2_(s.v, a.v), rel_l2_(e.v, a.v), e.rep.renoise,
              x.rep.steps, x.rep.renoise,
              r.ok ? "RAN" : r.err.c_str());
  EXPECT_TRUE(a.rep.steps == 6 && a.rep.forwards == 6 && a.rep.renoise);
  EXPECT_TRUE(a.v == b.v && a.a == b.a);
  EXPECT_TRUE(rel_l2_(s.v, a.v) > 1e-3 && rel_l2_(s.a, a.a) > 1e-3);
  EXPECT_TRUE(e.rep.steps == 6 && !e.rep.renoise);
  EXPECT_TRUE(rel_l2_(e.v, a.v) > 1e-3);
  EXPECT_TRUE(x.rep.steps == 2 && !x.rep.renoise);
  EXPECT_TRUE(!r.ok && r.err.find("sampling contract") != std::string::npos);
}

// The condition noise augmentation, the one rule the keyframe anchors and
// the references now share: `aug*z + (1-aug)*noise` from a stream that
// restarts at its seed, and 1.0 (or above) leaves the values clean.
TEST(pika_h3_flex, anchors_take_the_references_noise_rule)
{
  std::vector<float> z(1000);
  for (std::size_t i = 0; i < z.size(); ++i) {
    z[i] = std::sin((float)i * 0.37f);
  }
  std::vector<float> a = z, b = z, c = z, one = z;
  minimax_h3::augment_condition_values(a.data(), a.size(), 0.999, 0);
  minimax_h3::augment_condition_values(b.data(), b.size(), 0.999, 0);
  minimax_h3::augment_condition_values(c.data(), c.size(), 0.999, 1);
  minimax_h3::augment_condition_values(one.data(), one.size(), 1.0, 0);
  double nn = 0.0;
  int off = 0;
  for (std::size_t i = 0; i < z.size(); ++i) {
    const float n = (a[i] - 0.999f * z[i]) / 0.001f;
    nn += (double)n * n;
    if (std::fabs(a[i] - z[i]) > 0.01f) { ++off; }
  }
  const double rms = std::sqrt(nn / (double)z.size());
  std::printf("  the mixed-in noise's rms %.3f; seed 0 twice %s, seed 1 "
              "%s; aug 1.0 %s\n", rms, a == b ? "IDENTICAL" : "DIFFERS",
              a == c ? "THE SAME" : "differs",
              one == z ? "leaves them" : "MOVES them");
  EXPECT_TRUE(a == b && a != c && one == z);
  EXPECT_TRUE(rms > 0.8 && rms < 1.2);
  EXPECT_TRUE(off == 0);
}

// THE KEYFRAME PRESENTATION's plan: a picture wired to the conditioner's
// ref_image port books the vision tower's working set, in the condition
// phase, at H3's widest canvas -- and unwired books none. Configs only.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA, or a root holding it).
TEST(pika_h3_flex, a_wired_keyframe_books_the_vision_tower)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (h3root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(h3root));
  // Any vertex will do as the picture's source: only the edge is read.
  DiffusionConditionerStage src(&sess, "src", std::vector<InEdge>{}, c);
  auto vision_claim = [&](bool wired) {
    std::vector<InEdge> in(4, InEdge{nullptr, 0});
    if (wired) { in[3] = InEdge{&src, 0}; }
    DiffusionConditionerStage st(&sess, "cond", std::move(in), c);
    std::size_t bytes = 0;
    std::string phase;
    for (const ResourceClaim& rc : st.declare_resources()) {
      // A scratch claim states its bytes in its key, `label|bytes`.
      const std::string lab = "vision:cond|";
      if (rc.kind == model_memory::kScratchKind &&
          rc.key.compare(0, lab.size(), lab) == 0) {
        bytes += std::stoull(rc.key.substr(lab.size()));
        phase = rc.phase;
      }
    }
    return std::pair<std::size_t, std::string>{bytes, phase};
  };
  const auto on = vision_claim(true);
  const auto off = vision_claim(false);
  MiniMaxH3TextEncoder::Config ec;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(
      MiniMaxH3TextEncoder::resolve_encoder_dir(h3root), ec, nullptr));
  const std::size_t want = MetalQwenVisionEncoder::encode_scratch_bytes(
      ec.vision, 1920LL * 1088LL);
  std::printf("  wired: %zu MB in '%s' (the tower at 1920x1088: %zu MB); "
              "unwired: %zu\n", on.first >> 20, on.second.c_str(),
              want >> 20, off.first);
  EXPECT_TRUE(on.first == want && want > 0);
  EXPECT_TRUE(on.second == model_memory::kPhaseCondition);
  EXPECT_TRUE(off.first == 0);
}

// The tower a keyframe runs through, against that booking: H3's own
// (Qwen3-VL-32B's) tower over a keyframe on each canvas tier -- the token
// count the processor's factor-32 rule gives (832 x 480 -> 390), and the
// device's peak growth over the encode within what the plan books for
// that size.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(pika_h3_flex, the_keyframe_tower_allocates_what_the_plan_books)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (h3root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  const std::string enc = MiniMaxH3TextEncoder::resolve_encoder_dir(h3root);
  MiniMaxH3TextEncoder::Config ec;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(enc, ec, nullptr));
  std::string verr;
  auto tower = MiniMaxH3TextEncoder::load_vision(enc, mc, ec, &verr);
  ASSERT_TRUE(tower != nullptr);
  if (tower == nullptr) {
    std::printf("  tower: %s\n", verr.c_str());
    return;
  }
  const struct {
    int w, h, tokens;
  } sizes[] = {{832, 480, 390}, {1344, 768, 1008}, {1920, 1088, 2040}};
  for (const auto& s : sizes) {
    std::vector<std::uint8_t> px((std::size_t)3 * s.w * s.h);
    for (std::size_t i = 0; i < px.size(); ++i) {
      px[i] = (std::uint8_t)((i * 2654435761u) >> 24);
    }
    (void)mc->memory_budget_settled();
    const std::size_t before = mc->memory_budget().allocated;
    std::atomic<std::size_t> peak{before};
    std::atomic<bool> done{false};
    std::thread sampler([&] {
      while (!done.load()) {
        peak = std::max(peak.load(), mc->memory_budget().allocated);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
    auto r = tower->encode(px.data(), s.h, s.w);
    done = true;
    sampler.join();
    const std::size_t grew = peak.load() - before;
    const std::size_t booked = MetalQwenVisionEncoder::encode_scratch_bytes(
        ec.vision, (std::int64_t)s.w * s.h);
    const int tokens = r.embeddings.empty() ? 0 : r.n_tokens;
    std::printf("  %dx%d: %d vision tokens; peak growth %.1f MB, booked "
                "%.1f MB\n", s.w, s.h, tokens, grew / 1048576.0,
                booked / 1048576.0);
    EXPECT_TRUE(tokens == s.tokens);
    EXPECT_TRUE(grew <= booked);
  }
}
