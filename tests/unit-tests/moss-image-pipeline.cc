// MOSS-Image on its REAL checkpoint, component by component, against the
// reference's own goldens (tools/dump_moss_image_golden.py):
//
//   vae       the FLUX.2 VAE with its SMALL decoder (decoder widths
//             [96, 192, 384, 384] beside the encoder's [128 .. 512]),
//             decoding a DiT-space latent through the pipeline's own
//             BatchNorm de-whitening and 2x2 unpatchify
//   real_dit  one forward of the 5B DiT at 512^2, the conditional branch
//             and the negative one with block 22 skipped
//
// Env: VPIPE_MOSS_IMAGE_TEST_MODEL_PATH (the repo ROOT) and
// VPIPE_MOSS_IMAGE_GOLDEN. Each test skips vacuously without them.

#include "minitest.h"
#include "residency-reserve.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/moss-image/metal-moss-image-transformer.h"
#include "generative-models/shared/torch-randn.h"
#include "generative-models/tokenizer.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/weight-set.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/moss-image-model-config-stage.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

#define MP_REQUIRE(cond)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v == nullptr || *v == '\0') ? std::string() : std::string(v);
}

std::vector<float>
read_f32_(const std::filesystem::path& p)
{
  std::vector<float> v;
  std::ifstream in(p, std::ios::binary);
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  v.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

FlexData
manifest_(const std::string& tag)
{
  const std::string g = env_("VPIPE_MOSS_IMAGE_GOLDEN");
  if (g.empty()) { return FlexData(); }
  std::ifstream mf(std::filesystem::path(g) / (tag + "_manifest.json"));
  if (!mf) { return FlexData(); }
  try {
    return FlexData::from_json(mf);
  } catch (...) {
    return FlexData();
  }
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& want)
{
  double num = 0.0, den = 0.0;
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::uint16_t
bf16_bits_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  return (std::uint16_t)((u + 0x7fffu + r) >> 16);
}

float
bf16_(std::uint16_t h)
{
  std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

namespace {

std::vector<std::int32_t>
read_i32_(const std::filesystem::path& p)
{
  std::vector<std::int32_t> v;
  std::ifstream in(p, std::ios::binary);
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  v.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

class SourceText : public TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-source-text-moss-image";
  SourceText(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::string> prompts;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next < prompts.size()) {
      co_await ctx.write(0, std::make_unique<FlexDataPayload>(
                                FlexData::make_string(prompts[next++])));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-moss-image";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// Emits ONE conditioning on oport0 and its negative on oport1 -- the
// diffusion-conditioner's two ports -- then ends.
class SourceCond : public TypedStage<SourceCond> {
public:
  static constexpr const char* kTypeName = "ut-source-cond-moss-image";
  SourceCond(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(2); }
  std::vector<float> pos, neg;
  int cols = 0;
  int reps = 1;           // how many (negative, positive) pairs to send
  int sent = 0;
  static std::unique_ptr<TensorBeatPayload>
  beat_(const std::vector<float>& v, int cols)
  {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {(int)(v.size() / (std::size_t)cols), cols};
    tb->resize_contiguous(v.size());
    auto* d = reinterpret_cast<std::uint16_t*>((void*)tb->as_u8());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = bf16_bits_(v[i]); }
    return tb;
  }
  Job
  process(RuntimeContext& ctx) override
  {
    if (sent < reps) {
      ++sent;
      // The negative FIRST, as the conditioner sends it.
      co_await ctx.write(1, beat_(neg, cols));
      co_await ctx.write(0, beat_(pos, cols));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

std::vector<float>
beat_f32_(const BeatPayloadIntf* b, int* rows, int* cols)
{
  std::vector<float> out;
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(b);
  if (tb == nullptr || tb->dtype != TensorBeat::DType::Bf16 ||
      tb->shape.size() != 2) {
    return out;
  }
  *rows = (int)tb->shape[0];
  *cols = (int)tb->shape[1];
  out.resize((std::size_t)*rows * *cols);
  const auto* u =
      reinterpret_cast<const std::uint16_t*>((const void*)tb->as_u8());
  for (std::size_t k = 0; k < out.size(); ++k) { out[k] = bf16_(u[k]); }
  return out;
}

}  // namespace

// The template's ids against the reference processor's: the user turn,
// the generation prompt and the open <think> enable_thinking leaves --
// split at the special tokens and each run between them encoded, which
// is what the reference tokenizer does too.
TEST(moss_image_cond, template_ids_match_the_reference)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData man = manifest_("text");
  if (root.empty() || !man.is_object()) { return; }
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_json(
      root + "/tokenizer/tokenizer.json", &sess);
  MP_REQUIRE(tok != nullptr);
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  const FlexData pr = man.as_object().at("prompts");
  auto pa = pr.as_array();
  int exact = 0;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    const std::string p(pa.at(i).as_string(""));
    std::vector<std::int32_t> ids;
    auto special = [&](const char* t) {
      ids.push_back(tok->special_token_id(t));
    };
    auto text = [&](const std::string& t) {
      for (std::int32_t x : tok->encode(t)) { ids.push_back(x); }
    };
    special("<|im_start|>");
    text("user\n" + p);
    special("<|im_end|>");
    text("\n");
    special("<|im_start|>");
    text("assistant\n");
    special("<think>");
    text("\n");
    const std::vector<std::int32_t> want =
        read_i32_(g / ("text_ids" + std::to_string(i) + ".i32"));
    const bool same = ids == want;
    if (!same) {
      std::printf("[moss_image_cond] prompt %zu: %zu ids, the reference "
                  "%zu\n", i, ids.size(), want.size());
    }
    exact += same ? 1 : 0;
  }
  std::printf("[moss_image_cond] template ids id-exact for %d / %zu "
              "prompts\n", exact, pa.size());
  EXPECT_TRUE(exact == (int)pa.size() && exact > 0);
}

// The conditioner, end to end: prompt in, [rows, 4096] bf16 out on oport0
// and the EMPTY prompt's on oport1 (emitted unasked -- the reference's
// CFG is against it), each against the reference's fp32 encoder with the
// reference's own bf16 run as the bar.
TEST(moss_image_cond, conditioning_matches_the_reference)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData man = manifest_("text");
  if (root.empty() || !man.is_object()) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  auto mo = man.as_object();
  const FlexData pr = mo.at("prompts");
  auto pa = pr.as_array();
  const FlexData br = mo.contains("bf16_rel") ? mo.at("bf16_rel") : FlexData();
  // The last prompt is the empty one, the reference's negative.
  const std::size_t NP = pa.size() - 1;
  std::vector<std::string> prompts;
  for (std::size_t i = 0; i < NP; ++i) {
    prompts.emplace_back(pa.at(i).as_string(""));
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto pl = std::make_unique<Pipeline>("moss-image-cond", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = prompts;
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(c)));
  auto* pos = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "pos",
                                    std::vector<InEdge>{{cond, 0}},
                                    FlexData::make_object())));
  auto* neg = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "neg",
                                    std::vector<InEdge>{{cond, 1}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  MP_REQUIRE(pos->captured.size() == NP);
  MP_REQUIRE(neg->captured.size() == NP);
  auto check = [&](const BeatPayloadIntf* b, std::size_t gi,
                   const char* what) {
    int rows = 0, cols = 0;
    const std::vector<float> got = beat_f32_(b, &rows, &cols);
    const std::vector<float> want =
        read_f32_(g / ("text_hidden" + std::to_string(gi) + ".f32"));
    EXPECT_TRUE(cols == 4096);
    EXPECT_TRUE((std::size_t)rows * 4096 == want.size());
    if (got.size() != want.size()) { return; }
    const double rel = rel_l2_(got, want);
    const double bar = br.is_array() && gi < br.as_array().size()
                           ? br.as_array().at(gi).as_real(0.0) : 0.0;
    std::printf("[moss_image_cond] %s %zu: [%d, %d] rel-L2 %.5f (reference "
                "bf16 %.5f)\n", what, gi, rows, cols, rel, bar);
    EXPECT_TRUE(rel < std::max(1.5 * bar, 0.01));
  };
  for (std::size_t i = 0; i < NP; ++i) {
    check(pos->captured[i].get(), i, "prompt");
    check(neg->captured[i].get(), NP, "negative");
  }
}

// A text encoder kept warm (`keep_loaded`) is handed to the next graph
// only if THAT graph's plan would build it alike. Streamed through the
// raw-source slots -- this Qwen3.5 hybrid's path now -- it carries
// nothing a plan sized (its resident set is measured again before every
// prompt), so it is taken back even under a smaller box. Through the
// pinned fallback (VPIPE_ENC_RESIDENCY=0) the prefix was sized from the
// box, and one pinned for a roomier plan holds more than this one admits:
// built again, not taken. Three graphs each way -- kept, taken back,
// asked for under VPIPE_RAM_LIMIT_MB -- and the conditioning the same
// bytes in all six.
TEST(moss_image_cond, a_warm_encoder_is_taken_back_unless_it_pinned)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData man = manifest_("text");
  if (root.empty() || !man.is_object()) { return; }
  const FlexData pr = man.as_object().at("prompts");
  const std::string prompt(pr.as_array().at(0).as_string(""));
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  // One graph: the prompt's conditioning, whether it was taken, and how
  // many layers were resident as the prompt began.
  auto run = [&](bool* taken, int* resident) {
    std::vector<float> out;
    auto pl = std::make_unique<Pipeline>("moss-image-cond-warm", &sess);
    auto srcu = std::make_unique<SourceText>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
    srcu->prompts = {prompt};
    auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(root));
    c.as_object().insert("keep_loaded", FlexData::make_real(60.0));
    c.as_object().insert("unload_when_idle", FlexData::make_string("keep"));
    auto* cond = static_cast<DiffusionConditionerStage*>(pl->insert_stage(
        std::make_unique<DiffusionConditionerStage>(
            &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(c))));
    auto* pos = static_cast<SinkCapture*>(pl->insert_stage(
        std::make_unique<SinkCapture>(&sess, "pos",
                                      std::vector<InEdge>{{cond, 0}},
                                      FlexData::make_object())));
    pl->insert_stage(std::make_unique<SinkCapture>(
        &sess, "neg", std::vector<InEdge>{{cond, 1}},
        FlexData::make_object()));
    PipelineRuntime rt(pl.get(), &sess);
    const bool launched = rt.launch();
    if (launched) { rt.wait_idle(); }
    rt.stop();
    *taken = cond->encoder_taken_warm();
    *resident = cond->encoder_layers_resident_at_prompt();
    if (launched && pos->captured.size() == 1) {
      int rows = 0, cols = 0;
      out = beat_f32_(pos->captured[0].get(), &rows, &cols);
    }
    return out;   // the graph goes here, the encoder kept warm
  };
  std::vector<float> ref;
  for (const bool pinned : {false, true}) {
    if (pinned) { setenv("VPIPE_ENC_RESIDENCY", "0", 1); }
    bool t[3] = {true, false, !pinned};
    int r[3] = {0, 0, 0};
    const std::vector<float> first = run(&t[0], &r[0]);
    const std::vector<float> again = run(&t[1], &r[1]);
    setenv("VPIPE_RAM_LIMIT_MB", "32768", 1);
    const std::vector<float> small = run(&t[2], &r[2]);
    unsetenv("VPIPE_RAM_LIMIT_MB");
    if (pinned) { unsetenv("VPIPE_ENC_RESIDENCY"); }
    MP_REQUIRE(!first.empty() && again.size() == first.size() &&
               small.size() == first.size());
    if (ref.empty()) { ref = first; }
    auto same = [&](const std::vector<float>& v) {
      return v.size() == ref.size() &&
             std::memcmp(v.data(), ref.data(), ref.size() * 4) == 0;
    };
    std::printf("[moss_image_cond] %s: taken back %d / %d / %d (the last "
                "under a 32 GB box); layers resident as each prompt began "
                "%d / %d / %d; conditioning %s\n",
                pinned ? "pinned prefix" : "residency", (int)t[0], (int)t[1],
                (int)t[2], r[0], r[1], r[2],
                same(first) && same(again) && same(small)
                    ? "byte-identical throughout" : "DIFFERS");
    EXPECT_TRUE(!t[0] && t[1] && t[2] == !pinned);
    if (!pinned) {
      // Residency engaged (not -1), grew in the first graph, and the set
      // it grew came back with the encoder.
      EXPECT_TRUE(r[0] == 0 && r[1] > 0 && r[2] > 0);
    }
    // Neither a pinned prefix nor a resident set is the arithmetic.
    EXPECT_TRUE(same(first) && same(again) && same(small));
  }
}

// The small decoder: every decoder level narrower than the encoder's,
// the mid attention at 384 wide. A VAE that read the encoder's widths
// for the decoder would refuse the checkpoint's shapes at load; one that
// mixed them would decode a picture of the wrong thing.
TEST(moss_image_vae, decodes_as_the_reference)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData man = manifest_("vae");
  if (root.empty() || !man.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  const std::vector<float> z = read_f32_(g / "vae_z.f32");
  const std::vector<float> want = read_f32_(g / "vae_image.f32");
  auto o = man.as_object();
  const int H = (int)o.at("height").as_int(0);
  const int W = (int)o.at("width").as_int(0);
  const int h16 = H / 16, w16 = W / 16;
  MP_REQUIRE(z.size() == (std::size_t)128 * h16 * w16);
  MP_REQUIRE(want.size() == (std::size_t)3 * H * W);

  auto vae = MetalFlux2Vae::load(root + "/vae", mc, MetalFlux2Vae::Config{});
  MP_REQUIRE(vae != nullptr);
  EXPECT_TRUE(vae->config().dec_block_out[0] == 96);
  EXPECT_TRUE(vae->config().dec_block_out[3] == 384);
  EXPECT_TRUE(vae->config().block_out[3] == 512);
  SharedBuffer zb = mc->make_shared_buffer(z.size() * 2);
  MP_REQUIRE(!zb.empty());
  auto* zd = static_cast<_Float16*>(zb.contents());
  for (std::size_t i = 0; i < z.size(); ++i) { zd[i] = (_Float16)z[i]; }
  std::string err;
  SharedBuffer rgb = vae->decode(zb, h16, w16, &err);
  if (rgb.empty()) {
    std::printf("[moss_image_vae] decode failed: %s\n", err.c_str());
  }
  MP_REQUIRE(rgb.byte_size() >= want.size() * 2);
  std::vector<float> got(want.size());
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  for (std::size_t i = 0; i < got.size(); ++i) {
    got[i] = std::min(1.0f, std::max(-1.0f, (float)rp[i]));
  }
  const double rel = rel_l2_(got, want);
  std::printf("[moss_image_vae] %dx%d decode rel-L2 %.6f against the fp32 "
              "reference (%d convs on a narrow hardware tile)\n", W, H, rel,
              vae->narrow_hw_convs());
  EXPECT_TRUE(rel < 5e-3);
  if (!mc->supports_matrix_cores()) { return; }
  // The 96-wide full-resolution level is the hardware conv's on a
  // 32-channel destination tile (64 does not divide it): it must have
  // ENGAGED, and must compute the conv the im2col route does -- the same
  // decode with the narrow tiles declined.
  EXPECT_TRUE(vae->narrow_hw_convs() > 0);
  ::setenv("VPIPE_VAE_NO_C32", "1", 1);
  ::setenv("VPIPE_VAE_NO_C16", "1", 1);
  auto gather = MetalFlux2Vae::load(root + "/vae", mc,
                                    MetalFlux2Vae::Config{});
  ::unsetenv("VPIPE_VAE_NO_C32");
  ::unsetenv("VPIPE_VAE_NO_C16");
  MP_REQUIRE(gather != nullptr);
  SharedBuffer rgb2 = gather->decode(zb, h16, w16, &err);
  MP_REQUIRE(rgb2.byte_size() >= want.size() * 2);
  std::vector<float> g2(want.size());
  const auto* rp2 = static_cast<const _Float16*>(rgb2.contents());
  for (std::size_t i = 0; i < g2.size(); ++i) {
    g2[i] = std::min(1.0f, std::max(-1.0f, (float)rp2[i]));
  }
  const double ab = rel_l2_(got, g2);
  std::printf("[moss_image_vae] narrow hardware tile vs im2col rel-L2 %.6f "
              "(im2col arm: %d narrow convs, %.6f from the reference)\n", ab,
              gather->narrow_hw_convs(), rel_l2_(g2, want));
  EXPECT_TRUE(gather->narrow_hw_convs() == 0);
  EXPECT_TRUE(ab < 2e-3);
}

// The real 5B DiT at 512^2 (1024 image rows + the prompt's), the
// conditional branch and the reference's negative one (the empty
// prompt, block 22 skipped), each against the fp32 reference with the
// reference's own bf16 run as the bar.
TEST(moss_image_dit, real_checkpoint_matches_the_reference)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  // real_dit (sigma 0.6) by default; VPIPE_MOSS_IMAGE_REAL_TAG names
  // another dump of the same shape (real_dit_t1: sigma 1 on the e2e
  // noise).
  std::string tag = env_("VPIPE_MOSS_IMAGE_REAL_TAG");
  if (tag.empty()) { tag = "real_dit"; }
  const FlexData man = manifest_(tag);
  if (root.empty() || !man.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  auto o = man.as_object();
  const int H = (int)o.at("height").as_int(0);
  const int W = (int)o.at("width").as_int(0);
  const int gh = H / 16, gw = W / 16;
  const float t = (float)o.at("timestep").as_real(0.0);
  const int skip = (int)o.at("skip").as_int(22);
  const double bar_c = o.contains("bf16_rel") ? o.at("bf16_rel").as_real(0) : 0;
  const double bar_n =
      o.contains("bf16_rel_neg") ? o.at("bf16_rel_neg").as_real(0) : 0;
  const std::vector<float> lat = read_f32_(g / (tag + "_latent.f32"));
  const std::vector<float> pos = read_f32_(g / (tag + "_pos.f32"));
  const std::vector<float> neg = read_f32_(g / (tag + "_neg.f32"));
  const std::vector<float> want = read_f32_(g / (tag + "_out.f32"));
  const std::vector<float> want_n = read_f32_(g / (tag + "_out_neg.f32"));
  MP_REQUIRE(!lat.empty() && !pos.empty() && !neg.empty() && !want.empty());

  const std::string tdir = root + "/transformer";
  MetalMossImageTransformer::Config cfg;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(tdir, &cfg));
  const bool stream = !env_("VPIPE_MOSS_IMAGE_STREAM").empty();
  auto m = MetalMossImageTransformer::load(tdir, mc, cfg, stream);
  MP_REQUIRE(m != nullptr);
  const int C = cfg.latent_channels, TD = cfg.text_dim;
  auto tokens = [&](const std::vector<float>& chw) {
    SharedBuffer b = mc->make_shared_buffer(chw.size() * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < gh * gw; ++p) {
        d[(std::size_t)p * C + c] =
            bf16_bits_(chw[(std::size_t)c * gh * gw + p]);
      }
    }
    return b;
  };
  auto rows = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = bf16_bits_(v[i]); }
    return b;
  };
  SharedBuffer lb = tokens(lat);
  auto run = [&](const std::vector<float>& txt, std::vector<int> sk) {
    SharedBuffer tb = rows(txt);
    MetalMossImageTransformer::Request req;
    req.latents = &lb;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &tb;
    req.text_rows = (int)(txt.size() / (std::size_t)TD);
    req.timestep = t;
    req.skip_layers = std::move(sk);
    std::string err;
    SharedBuffer v = m->forward(req, &err);
    std::vector<float> out;
    if (v.empty()) {
      std::printf("[moss_image_dit] forward failed: %s\n", err.c_str());
      return out;
    }
    out.resize((std::size_t)C * gh * gw);
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < gh * gw; ++p) {
        out[(std::size_t)c * gh * gw + p] = bf16_(vp[(std::size_t)p * C + c]);
      }
    }
    return out;
  };
  const std::vector<float> got = run(pos, {});
  MP_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, want);
  std::printf("[moss_image_dit] real %dx%d cond rel-L2 %.5f (reference "
              "bf16 %.5f)%s\n", W, H, rel, bar_c, stream ? ", streamed" : "");
  EXPECT_TRUE(rel < std::max(1.5 * bar_c, 0.01));
  if (!want_n.empty()) {
    const std::vector<float> gn = run(neg, {skip});
    MP_REQUIRE(!gn.empty());
    const double rn = rel_l2_(gn, want_n);
    std::printf("[moss_image_dit] real uncond (skip %d) rel-L2 %.5f "
                "(reference bf16 %.5f)\n", skip, rn, bar_n);
    EXPECT_TRUE(rn < std::max(1.5 * bar_n, 0.01));
  }
}

// THE SAMPLER, isolated from the encoder: the reference's own fp32
// prompt and empty-prompt embeddings into generate-image, which draws
// its noise from the seed, runs the snapped schedule with CFG against
// the negative (block 22 skipped) and emits [128, H/16, W/16]; against
// the reference pipeline's fp32 latent and its decoded picture. The
// endpoint of an N-step run is one draw of a chaotic sum, so the bar is
// the reference's OWN bf16 run's distance (the e2e_bf16 dump) when
// present, and a gross one otherwise.
TEST(moss_image_stage, sampler_matches_the_reference)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  // e2e (20 steps) by default; VPIPE_MOSS_IMAGE_E2E_TAG names another
  // dump (e2e_s2: two steps, where the drift has had no room to grow).
  std::string tag = env_("VPIPE_MOSS_IMAGE_E2E_TAG");
  if (tag.empty()) { tag = "e2e"; }
  const FlexData man = manifest_(tag);
  const FlexData tman = manifest_("text");
  if (root.empty() || !man.is_object() || !tman.is_object()) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  auto o = man.as_object();
  const int W = (int)o.at("width").as_int(0);
  const int H = (int)o.at("height").as_int(0);
  const int steps = (int)o.at("steps").as_int(0);
  const int seed = (int)o.at("seed").as_int(0);
  const std::vector<float> want = read_f32_(g / (tag + "_latent.f32"));
  const std::vector<float> want_img = read_f32_(g / (tag + "_image.f32"));
  const std::vector<float> noise = read_f32_(g / (tag + "_noise.f32"));
  const std::size_t np = tman.as_object().at("prompts").as_array().size();
  std::vector<float> pos = read_f32_(g / "text_hidden0.f32");
  std::vector<float> neg =
      read_f32_(g / ("text_hidden" + std::to_string(np - 1) + ".f32"));
  MP_REQUIRE(!want.empty() && !pos.empty() && !neg.empty());

  // The noise the stage will draw for this seed IS the reference's.
  {
    genai::TorchCpuRandn rng((std::uint64_t)seed);
    const std::vector<float> z = rng.randn(noise.size());
    EXPECT_TRUE(z == noise);
  }

  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto pl = std::make_unique<Pipeline>("moss-image-stage", &sess);
  auto su = std::make_unique<SourceCond>(&sess, "src", std::vector<InEdge>{},
                                         FlexData::make_object());
  su->pos = pos;
  su->neg = neg;
  su->cols = 4096;
  auto* src = pl->insert_stage(std::move(su));
  FlexData gc = FlexData::make_object();
  gc.as_object().insert("hf_dir", FlexData::make_string(root));
  gc.as_object().insert("width", FlexData::make_int(W));
  gc.as_object().insert("height", FlexData::make_int(H));
  gc.as_object().insert("seed", FlexData::make_int(seed));
  gc.as_object().insert("steps", FlexData::make_int(steps));
  // The golden's skip-layer guidance, through the family's config stage
  // on model_config (iport 7) -- which is how a graph says it.
  std::vector<InEdge> gin{{src, 0}, {src, 1}};
  if (o.contains("slg_scale") && o.at("slg_scale").as_real(0.0) > 0.0) {
    FlexData mcfg = FlexData::make_object();
    auto mo = mcfg.as_object();
    for (const char* k : {"slg_layers", "slg_scale", "slg_start",
                          "slg_stop"}) {
      mo.insert(k, o.at(k));
    }
    auto* cfgst = pl->insert_stage(std::make_unique<MossImageModelConfigStage>(
        &sess, "mcfg", std::vector<InEdge>{}, std::move(mcfg)));
    gin = {{src, 0}, {src, 1}, InEdge{nullptr, 0}, InEdge{nullptr, 0},
           InEdge{nullptr, 0}, InEdge{nullptr, 0}, InEdge{nullptr, 0},
           {cfgst, 0}};
  }
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "gen", gin,
                                           std::move(gc))));
  MP_REQUIRE(gen->config_error().empty());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  MP_REQUIRE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  MP_REQUIRE(tb != nullptr && tb->dtype == TensorBeat::DType::F32);
  MP_REQUIRE(tb->element_count() == want.size());
  const std::vector<float> got(tb->as_f32(), tb->as_f32() + want.size());
  const double rel = rel_l2_(got, want);
  double bar = 0.35;
  const FlexData bman = manifest_(tag + "_bf16");
  if (bman.is_object()) {
    const std::vector<float> bl = read_f32_(g / (tag + "_bf16_latent.f32"));
    if (bl.size() == want.size()) { bar = 1.35 * rel_l2_(bl, want); }
  }
  // The picture, through the VAE, as PSNR against the reference's.
  double psnr = 0.0;
  {
    auto vae = MetalFlux2Vae::load(root + "/vae", sess.metal_compute(),
                                   MetalFlux2Vae::Config{});
    MP_REQUIRE(vae != nullptr);
    const int h16 = H / 16, w16 = W / 16;
    SharedBuffer zb = sess.metal_compute()->make_shared_buffer(got.size() * 2);
    auto* zd = static_cast<_Float16*>(zb.contents());
    for (std::size_t i = 0; i < got.size(); ++i) { zd[i] = (_Float16)got[i]; }
    std::string err;
    SharedBuffer rgb = vae->decode(zb, h16, w16, &err);
    MP_REQUIRE(rgb.byte_size() >= want_img.size() * 2);
    const auto* rp = static_cast<const _Float16*>(rgb.contents());
    // VPIPE_MOSS_IMAGE_DUMP=<dir>: our picture as raw f32 [3, H, W].
    if (const std::string dd = env_("VPIPE_MOSS_IMAGE_DUMP"); !dd.empty()) {
      std::vector<float> f(want_img.size());
      for (std::size_t i = 0; i < f.size(); ++i) { f[i] = (float)rp[i]; }
      std::ofstream o(std::filesystem::path(dd) / (tag + "_ours.f32"),
                      std::ios::binary);
      o.write(reinterpret_cast<const char*>(f.data()),
              (std::streamsize)(f.size() * 4));
    }
    double se = 0.0;
    for (std::size_t i = 0; i < want_img.size(); ++i) {
      const double a =
          std::min(1.0, std::max(-1.0, (double)rp[i])) * 127.5 + 127.5;
      const double b = (double)want_img[i] * 127.5 + 127.5;
      se += (a - b) * (a - b);
    }
    const double mse = se / (double)want_img.size();
    psnr = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
  }
  std::printf("[moss_image_stage] %dx%d %d steps: latent rel-L2 %.4f (bar "
              "%.4f%s), picture %.2f dB from the reference's\n", W, H,
              steps, rel, bar, bman.is_object() ? ", the reference bf16" : "",
              psnr);
  EXPECT_TRUE(rel < bar);
  EXPECT_TRUE(psnr > 25.0);
}

// A BENCHMARK, not a check: the real DiT's forward at
// VPIPE_MOSS_IMAGE_BENCH=<side> (e.g. 1024) with a 64-row prompt -- a
// warm forward, then three timed -- with VPIPE_MOSS_IMAGE_BENCH_TIER
// (-1 off, 0 accurate, 1 fast, 2 token, 3 token_plain) for i8_gemm and
// VPIPE_MOSS_IMAGE_SAGE=1 for Sage. Quote the box, the cooling and the
// geometry with any number it prints.
TEST(moss_image_dit, bench_real_forward)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const std::string side = env_("VPIPE_MOSS_IMAGE_BENCH");
  if (root.empty() || side.empty()) { return; }
  const int S = std::atoi(side.c_str());
  if (S < 256 || (S % 16) != 0) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = root + "/transformer";
  MetalMossImageTransformer::Config cfg;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(tdir, &cfg));
  const std::string tier = env_("VPIPE_MOSS_IMAGE_BENCH_TIER");
  if (!tier.empty() && std::atoi(tier.c_str()) >= 0) {
    cfg.i8_gemm = true;
    cfg.i8_tier = std::atoi(tier.c_str());
  }
  const bool stream = !env_("VPIPE_MOSS_IMAGE_STREAM").empty();
  auto m = MetalMossImageTransformer::load(tdir, mc, cfg, stream);
  MP_REQUIRE(m != nullptr);
  if (stream) {
    m->set_residency_reserve(0);
    m->set_residency_schedule(4);
  }
  const int gh = S / 16, gw = S / 16, T = 64;
  auto rnd = [&](std::size_t n, unsigned seed) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      d[i] = bf16_bits_(((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f));
    }
    return b;
  };
  SharedBuffer lat = rnd((std::size_t)gh * gw * cfg.latent_channels, 3u);
  SharedBuffer txt = rnd((std::size_t)T * cfg.text_dim, 4u);
  MetalMossImageTransformer::Request req;
  req.latents = &lat;
  req.grid_h = gh;
  req.grid_w = gw;
  req.text = &txt;
  req.text_rows = T;
  req.timestep = 600.0f;
  std::string err;
  MP_REQUIRE(!m->forward(req, &err).empty());
  const std::uint64_t f0 = m->gpu_flops().flushed();
  const auto t0 = std::chrono::steady_clock::now();
  const int N = 3;
  for (int i = 0; i < N; ++i) { MP_REQUIRE(!m->forward(req, &err).empty()); }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / N;
  const double tf = (double)(m->gpu_flops().flushed() - f0) / N / 1e12;
  std::printf("[moss_image_bench] %dx%d (%d rows): %.0f ms a forward, %.2f "
              "TFLOP -> %.2f TFLOP/s%s%s%s%s\n", S, S, gh * gw + T, ms, tf,
              tf / (ms / 1000.0), m->uses_mma2() ? ", matmul2d" : "",
              m->uses_attn_nax() ? ", NAX attention" : "",
              m->uses_i8_gemm() ? (m->fast_tier() ? ", i8 fast/token"
                                                  : ", i8 accurate")
                                : "",
              stream ? ", streamed" : "");
}

// SECOND GENERATION IS THE TEST: two prompts in one process, the DiT
// released for the decode after the first (VPIPE_IMAGE_DIT_RECLAIM=1) and
// reloaded for the second, on the SAME seed (seed_sequence keep) -- so
// the second latent must equal the first byte for byte. A one-shot run
// never reaches the release / reload regime.
TEST(moss_image_stage, a_second_generation_matches_the_first)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData tman = manifest_("text");
  if (root.empty() || !tman.is_object()) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  const std::size_t np = tman.as_object().at("prompts").as_array().size();
  const std::vector<float> pos = read_f32_(g / "text_hidden1.f32");
  const std::vector<float> neg =
      read_f32_(g / ("text_hidden" + std::to_string(np - 1) + ".f32"));
  MP_REQUIRE(!pos.empty() && !neg.empty());
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  setenv("VPIPE_IMAGE_DIT_RECLAIM", "1", 1);
  auto pl = std::make_unique<Pipeline>("moss-image-twice", &sess);
  auto su = std::make_unique<SourceCond>(&sess, "src", std::vector<InEdge>{},
                                         FlexData::make_object());
  su->pos = pos;
  su->neg = neg;
  su->cols = 4096;
  su->reps = 2;
  auto* src = pl->insert_stage(std::move(su));
  FlexData gc = FlexData::make_object();
  auto go = gc.as_object();
  go.insert("hf_dir", FlexData::make_string(root));
  go.insert("width", FlexData::make_int(384));
  go.insert("height", FlexData::make_int(256));
  go.insert("seed", FlexData::make_int(5));
  go.insert("seed_sequence", FlexData::make_string("keep"));
  go.insert("steps", FlexData::make_int(3));
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(
          &sess, "gen", std::vector<InEdge>{{src, 0}, {src, 1}},
          std::move(gc))));
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  if (launched) { rt.wait_idle(); }
  rt.stop();
  unsetenv("VPIPE_IMAGE_DIT_RECLAIM");
  MP_REQUIRE(launched && gen->config_error().empty());
  MP_REQUIRE(sink->captured.size() == 2);
  const auto* a =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  const auto* b =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[1].get());
  MP_REQUIRE(a != nullptr && b != nullptr &&
             a->element_count() == b->element_count() &&
             a->element_count() > 0);
  const bool same = std::memcmp(a->as_f32(), b->as_f32(),
                                a->element_count() * 4) == 0;
  std::printf("[moss_image_stage] two generations, DiT released and "
              "reloaded between them: %s\n",
              same ? "byte-identical" : "DIFFER");
  EXPECT_TRUE(same);
}

// A DiT kept warm (`keep_loaded`) and taken back by the next graph makes
// the picture a fresh build makes, at THAT graph's adapter strength. The
// first graph runs its adapter at 0 and keeps the DiT; the second asks
// for 1. A strength is live, so the warm key names the adapter's file
// alone and a take sets every strength again: what the last graph left
// on the model is not this one's (before, the second graph would have
// generated at the first one's strength had it taken the DiT). A third
// graph builds fresh (keep_loaded 0) for the bytes to match.
TEST(moss_image_stage, a_dit_kept_warm_is_taken_back_at_this_strength)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData tman = manifest_("text");
  if (root.empty() || !tman.is_object()) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  const std::size_t np = tman.as_object().at("prompts").as_array().size();
  const std::vector<float> pos = read_f32_(g / "text_hidden1.f32");
  const std::vector<float> neg =
      read_f32_(g / ("text_hidden" + std::to_string(np - 1) + ".f32"));
  MP_REQUIRE(!pos.empty() && !neg.empty());
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto* mgr = sess.generative_model_manager();
  MP_REQUIRE(mgr != nullptr);

  // A rank-8 adapter on every block's q / k / v and feed-forward.
  MetalMossImageTransformer::Config cfg;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(
      root + "/transformer", &cfg));
  const int R = 8, H = cfg.hidden, F = cfg.ffn_inner;
  unsigned st = 7u;
  auto rnd = [&](std::size_t n) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.05f;
      const std::uint16_t h = genai::train::f32_to_bf16(v);
      std::memcpy(out.data() + 2 * i, &h, 2);
    }
    return out;
  };
  std::vector<genai::train::OutTensor> ts;
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".proj.";
    const struct { const char* leaf; int n, k; } mods[] = {
        {"q", H, H}, {"k", H, H}, {"v", H, H},
        {"ffn_up_gate", 2 * F, H}, {"ffn_down", H, F}};
    for (const auto& md : mods) {
      ts.push_back({p + md.leaf + ".lora_A.weight", "BF16", {R, md.k},
                    rnd((std::size_t)R * md.k)});
      ts.push_back({p + md.leaf + ".lora_B.weight", "BF16", {md.n, R},
                    rnd((std::size_t)md.n * R)});
    }
  }
  namespace fs = std::filesystem;
  const fs::path lp = fs::temp_directory_path() / "vpipe-moss-warm-lora" /
                      "adapter.safetensors";
  std::error_code ec;
  fs::create_directories(lp.parent_path(), ec);
  std::string werr;
  MP_REQUIRE(genai::train::write_safetensors(lp.string(), ts, {}, &werr));

  // One graph: the generation's latent, and whether its DiT was taken.
  auto run = [&](double keep, double strength, bool* taken) {
    std::vector<float> out;
    auto pl = std::make_unique<Pipeline>("moss-image-warm", &sess);
    auto su = std::make_unique<SourceCond>(
        &sess, "src", std::vector<InEdge>{}, FlexData::make_object());
    su->pos = pos;
    su->neg = neg;
    su->cols = 4096;
    auto* src = pl->insert_stage(std::move(su));
    FlexData gc = FlexData::make_object();
    auto go = gc.as_object();
    go.insert("hf_dir", FlexData::make_string(root));
    go.insert("width", FlexData::make_int(384));
    go.insert("height", FlexData::make_int(256));
    go.insert("seed", FlexData::make_int(5));
    go.insert("steps", FlexData::make_int(2));
    go.insert("keep_loaded", FlexData::make_real(keep));
    go.insert("lora", FlexData::make_string(lp.string()));
    go.insert("lora_scale", FlexData::make_real(strength));
    auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
        std::make_unique<GenerateImageStage>(
            &sess, "gen", std::vector<InEdge>{{src, 0}, {src, 1}},
            std::move(gc))));
    auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
        std::make_unique<SinkCapture>(&sess, "sink",
                                      std::vector<InEdge>{{gen, 0}},
                                      FlexData::make_object())));
    PipelineRuntime rt(pl.get(), &sess);
    const bool launched = rt.launch();
    if (launched) { rt.wait_idle(); }
    rt.stop();
    *taken = gen->dit_taken_warm();
    const auto* t = sink->captured.size() == 1
        ? dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get())
        : nullptr;
    if (launched && gen->config_error().empty() && t != nullptr) {
      out.assign(t->as_f32(), t->as_f32() + t->element_count());
    }
    return out;   // the graph goes here, and with it the DiT, kept warm
  };
  // Preloaded: a box the plan streams the DiT on for room -- this one,
  // beside the 18 GB encoder the stage claims -- keeps nothing warm, by
  // design, and the test would pass on three fresh builds.
  setenv("VPIPE_MOSS_IMAGE_STREAM", "0", 1);
  bool t0 = true, t1 = false, t2 = true;
  const std::vector<float> at0 = run(60.0, 0.0, &t0);   // built, kept
  const std::vector<std::string> warm = mgr->warm_dirs();
  const std::vector<float> at1 = run(60.0, 1.0, &t1);   // taken back
  const std::vector<float> fresh = run(0.0, 1.0, &t2);  // built again
  unsetenv("VPIPE_MOSS_IMAGE_STREAM");
  MP_REQUIRE(!at0.empty() && at1.size() == at0.size() &&
             fresh.size() == at0.size());
  const bool same = std::memcmp(at1.data(), fresh.data(),
                                at1.size() * 4) == 0;
  const bool moved = std::memcmp(at1.data(), at0.data(),
                                 at1.size() * 4) != 0;
  std::printf("[moss_image_stage] kept warm: %zu dir(s); taken back %d / "
              "%d / %d; at strength 1 against a fresh build: %s; against "
              "strength 0: %s\n", warm.size(), (int)t0, (int)t1, (int)t2,
              same ? "byte-identical" : "DIFFER",
              moved ? "differs" : "the SAME");
  EXPECT_TRUE(!warm.empty());
  EXPECT_TRUE(!t0 && t1 && !t2);
  EXPECT_TRUE(same);
  EXPECT_TRUE(moved);
}

// ...and a graph under ANOTHER sage_pv8 does not take it. The family reads
// its Sage settings off the config it was built with at every forward --
// which flash function, which prologue -- so a DiT kept by a graph
// without the int8 P*V and taken by one with it would run without it,
// and the reverse; an A/B of the setting in one process would measure the
// first arm twice. The warm key names it now. Asked on any box: the key
// is the config's, so it decides the take where the GPU cannot run Sage
// at all (here it then says so and runs dense). Three graphs, each
// keeping its DiT: kept without, not taken with (kept in turn), taken by
// the next one with -- which is what says the second's miss was the key.
TEST(moss_image_stage, a_dit_kept_warm_is_not_taken_under_another_sage_pv8)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const FlexData tman = manifest_("text");
  if (root.empty() || !tman.is_object()) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  const std::size_t np = tman.as_object().at("prompts").as_array().size();
  const std::vector<float> pos = read_f32_(g / "text_hidden1.f32");
  const std::vector<float> neg =
      read_f32_(g / ("text_hidden" + std::to_string(np - 1) + ".f32"));
  MP_REQUIRE(!pos.empty() && !neg.empty());
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  auto run = [&](double keep, bool pv8, bool* taken) {
    auto pl = std::make_unique<Pipeline>("moss-image-warm-pv8", &sess);
    auto su = std::make_unique<SourceCond>(
        &sess, "src", std::vector<InEdge>{}, FlexData::make_object());
    su->pos = pos;
    su->neg = neg;
    su->cols = 4096;
    auto* src = pl->insert_stage(std::move(su));
    FlexData gc = FlexData::make_object();
    auto go = gc.as_object();
    go.insert("hf_dir", FlexData::make_string(root));
    go.insert("width", FlexData::make_int(384));
    go.insert("height", FlexData::make_int(256));
    go.insert("seed", FlexData::make_int(5));
    go.insert("steps", FlexData::make_int(2));
    go.insert("keep_loaded", FlexData::make_real(keep));
    go.insert("sage_attn", FlexData::make_bool(true));
    go.insert("sage_pv8", FlexData::make_bool(pv8));
    auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
        std::make_unique<GenerateImageStage>(
            &sess, "gen", std::vector<InEdge>{{src, 0}, {src, 1}},
            std::move(gc))));
    pl->insert_stage(std::make_unique<SinkCapture>(
        &sess, "sink", std::vector<InEdge>{{gen, 0}},
        FlexData::make_object()));
    PipelineRuntime rt(pl.get(), &sess);
    const bool ok = rt.launch() && gen->config_error().empty();
    if (ok) { rt.wait_idle(); }
    rt.stop();
    *taken = gen->dit_taken_warm();
    return ok;
  };
  // Preloaded, as in the test above: a streamed DiT is never kept warm.
  setenv("VPIPE_MOSS_IMAGE_STREAM", "0", 1);
  bool t0 = true, t1 = true, t2 = false;
  const bool ok0 = run(60.0, false, &t0);   // built, kept
  const bool ok1 = run(60.0, true, &t1);    // another arm: built, kept
  const bool ok2 = run(60.0, true, &t2);    // the same arm: taken
  unsetenv("VPIPE_MOSS_IMAGE_STREAM");
  std::printf("[moss_image_stage] warm DiT taken: sage_pv8 off %d, then on "
              "%d, then on again %d\n", (int)t0, (int)t1, (int)t2);
  EXPECT_TRUE(ok0 && ok1 && ok2);
  EXPECT_TRUE(!t0 && !t1 && t2);
}

namespace {

// The real DiT forward over random inputs at (gh, gw) with a T-row
// prompt; the velocity as f32, or empty.
std::vector<float>
random_forward_(MetalMossImageTransformer* m, MetalCompute* mc, int gh,
                int gw, int T, unsigned seed)
{
  const auto& c = m->config();
  auto rnd = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      d[i] = bf16_bits_(((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f));
    }
    seed += 7u;
    return b;
  };
  SharedBuffer lat = rnd((std::size_t)gh * gw * c.latent_channels);
  SharedBuffer txt = rnd((std::size_t)T * c.text_dim);
  MetalMossImageTransformer::Request req;
  req.latents = &lat;
  req.grid_h = gh;
  req.grid_w = gw;
  req.text = &txt;
  req.text_rows = T;
  req.timestep = 750.0f;
  std::string e;
  SharedBuffer v = m->forward(req, &e);
  std::vector<float> out;
  if (v.empty()) { return out; }
  out.resize((std::size_t)gh * gw * c.latent_channels);
  const auto* p = static_cast<const std::uint16_t*>(v.contents());
  for (std::size_t i = 0; i < out.size(); ++i) { out[i] = bf16_(p[i]); }
  return out;
}

bool
same_(const std::vector<float>& a, const std::vector<float>& b)
{
  return a.size() == b.size() && !a.empty() &&
         std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

}  // namespace

// A model-quantize'd W8 PACK (VPIPE_MOSS_IMAGE_QUANT_DIT = its
// transformer/): the real 512^2 forward against the fp32 reference, and
// the pack streamed (codes placed raw, scales and biases as bf16) equal
// to the pack preloaded.
TEST(moss_image_dit, a_w8_pack_tracks_the_reference)
{
  const std::string qd = env_("VPIPE_MOSS_IMAGE_QUANT_DIT");
  const FlexData man = manifest_("real_dit");
  if (qd.empty() || !man.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  auto o = man.as_object();
  const int H = (int)o.at("height").as_int(0);
  const int W = (int)o.at("width").as_int(0);
  const int gh = H / 16, gw = W / 16;
  const std::vector<float> lat = read_f32_(g / "real_dit_latent.f32");
  const std::vector<float> pos = read_f32_(g / "real_dit_pos.f32");
  const std::vector<float> want = read_f32_(g / "real_dit_out.f32");
  MP_REQUIRE(!lat.empty() && !pos.empty() && !want.empty());
  MetalMossImageTransformer::Config cfg;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(qd, &cfg));
  const int C = cfg.latent_channels;
  auto run = [&](bool stream) {
    std::vector<float> out;
    auto m = MetalMossImageTransformer::load(qd, mc, cfg, stream);
    if (m == nullptr || m->quant_bits() != 8) { return out; }
    SharedBuffer lb = mc->make_shared_buffer(lat.size() * 2);
    auto* d = static_cast<std::uint16_t*>(lb.contents());
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < gh * gw; ++p) {
        d[(std::size_t)p * C + c] =
            bf16_bits_(lat[(std::size_t)c * gh * gw + p]);
      }
    }
    SharedBuffer tb = mc->make_shared_buffer(pos.size() * 2);
    auto* t = static_cast<std::uint16_t*>(tb.contents());
    for (std::size_t i = 0; i < pos.size(); ++i) { t[i] = bf16_bits_(pos[i]); }
    MetalMossImageTransformer::Request req;
    req.latents = &lb;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &tb;
    req.text_rows = (int)(pos.size() / (std::size_t)cfg.text_dim);
    req.timestep = (float)o.at("timestep").as_real(0.0);
    std::string e;
    SharedBuffer v = m->forward(req, &e);
    if (v.empty()) { return out; }
    out.resize(want.size());
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < gh * gw; ++p) {
        out[(std::size_t)c * gh * gw + p] = bf16_(vp[(std::size_t)p * C + c]);
      }
    }
    return out;
  };
  const std::vector<float> a = run(false);
  const std::vector<float> b = run(true);
  MP_REQUIRE(!a.empty() && !b.empty());
  const double rel = rel_l2_(a, want);
  const double bar = o.contains("bf16_rel") ? o.at("bf16_rel").as_real(0) : 0;
  std::printf("[moss_image_dit] w8 pack 512^2: rel-L2 %.5f from the fp32 "
              "reference (its bf16 %.5f); streamed %s\n", rel, bar,
              same_(a, b) ? "IDENTICAL" : "DIFFERENT");
  EXPECT_TRUE(rel < std::max(2.0 * bar, 0.02));
  EXPECT_TRUE(same_(a, b));
}

// A MIXED PACK (VPIPE_MOSS_IMAGE_MIXED_DIT = the transformer/ of a
// model-quantize bits 4 + mixed run): whole layers at w8, the rest at w4,
// so consecutive blocks differ in size and a slot sized for one width is
// refilled with the other. The pack must really be mixed (both widths
// among the main blocks -- a ranking over the wrong prefix writes a
// uniform pack and says nothing), its four holds (preloaded, streamed,
// no slots, promoted) BYTE-IDENTICAL, and the 512^2 forward within a
// gross bar of the fp32 reference -- w4 rounding, not a misread width
// (which lands near 1). MEASURED on the M4 Pro: bf16 0.035, w8 pack
// 0.066, this pack (11 of 42 blocks w8) 0.170, uniform w4 0.251. Before
// the slots re-derived each refill's width, preloaded and no-slots
// agreed and the two slot holds did not.
TEST(moss_image_dit, a_mixed_pack_holds_agree)
{
  const std::string qd = env_("VPIPE_MOSS_IMAGE_MIXED_DIT");
  const FlexData man = manifest_("real_dit");
  if (qd.empty() || !man.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(qd, &cfg));
  int n8 = 0, n4 = 0;
  {
    auto ws = genai::WeightSet::open(qd, nullptr);
    MP_REQUIRE(ws != nullptr);
    for (int L = 0; L < cfg.n_layers; ++L) {
      const std::string p = "layers." + std::to_string(L) + ".proj.q.";
      const auto* ci = ws->src().info(p + "weight");
      const auto* si = ws->src().info(p + "scales");
      if (ci == nullptr || si == nullptr || ci->shape.size() != 2 ||
          si->shape.size() != 2) {
        continue;
      }
      const long K = si->shape[1] * 64L;
      const int b = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
      n8 += b == 8;
      n4 += b == 4;
    }
  }
  std::printf("[moss_image_dit] mixed pack: %d blocks w8, %d w4\n", n8, n4);
  MP_REQUIRE(n8 > 0 && n4 > 0);
  const std::filesystem::path g(env_("VPIPE_MOSS_IMAGE_GOLDEN"));
  auto o = man.as_object();
  const int gh = (int)o.at("height").as_int(0) / 16;
  const int gw = (int)o.at("width").as_int(0) / 16;
  const std::vector<float> lat = read_f32_(g / "real_dit_latent.f32");
  const std::vector<float> pos = read_f32_(g / "real_dit_pos.f32");
  const std::vector<float> want = read_f32_(g / "real_dit_out.f32");
  MP_REQUIRE(!lat.empty() && !pos.empty() && !want.empty());
  const int C = cfg.latent_channels;
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  bool fast_on = false;
  int f8 = 0, f4 = 0;
  int promoted = 0, relaid = 0;
  int stack = 0;
  std::uint64_t grown_bytes = 0;
  bool engaged = true;
  auto run = [&](int tier, bool stream, bool slots, bool grow,
                 bool hold = false) {
    std::unique_ptr<Env> ns;
    if (!slots) {
      ns = std::make_unique<Env>("VPIPE_MOSS_IMAGE_NO_SLOTS", "1");
    }
    std::vector<float> out;
    MetalMossImageTransformer::Config c = cfg;
    c.i8_gemm = tier >= 0;
    c.i8_tier = tier < 0 ? 0 : tier;
    c.i8_hold_w4 = hold;
    auto m = MetalMossImageTransformer::load(qd, mc, c, stream);
    if (m == nullptr) { return out; }
    if (tier == 0 && !m->signed_layout()) { engaged = false; }
    if (tier > 0) {
      fast_on = m->fast_tier();
      if (!stream) {
        f8 = m->fast_weights(8);
        f4 = m->fast_weights(4);
      }
    }
    if (grow) {
      m->set_residency_reserve(test_residency_reserve(8ull << 30));
      m->set_residency_schedule(4);
    }
    SharedBuffer lb = mc->make_shared_buffer(lat.size() * 2);
    auto* d = static_cast<std::uint16_t*>(lb.contents());
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < gh * gw; ++p) {
        d[(std::size_t)p * C + c] =
            bf16_bits_(lat[(std::size_t)c * gh * gw + p]);
      }
    }
    SharedBuffer tb = mc->make_shared_buffer(pos.size() * 2);
    auto* t = static_cast<std::uint16_t*>(tb.contents());
    for (std::size_t i = 0; i < pos.size(); ++i) { t[i] = bf16_bits_(pos[i]); }
    MetalMossImageTransformer::Request req;
    req.latents = &lb;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &tb;
    req.text_rows = (int)(pos.size() / (std::size_t)cfg.text_dim);
    req.timestep = (float)o.at("timestep").as_real(0.0);
    std::string e;
    SharedBuffer v;
    for (int it = 0; it < (grow ? 2 : 1); ++it) { v = m->forward(req, &e); }
    if (v.empty()) { return out; }
    if (grow) {
      promoted = m->resident_block_count();
      stack = m->config().n_layers;
      grown_bytes = m->resident_bytes();
    }
    if (stream && slots && !grow) { relaid = m->slot_relayouts(); }
    out.resize(want.size());
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (int c = 0; c < C; ++c) {
      for (int p = 0; p < gh * gw; ++p) {
        out[(std::size_t)c * gh * gw + p] = bf16_(vp[(std::size_t)p * C + c]);
      }
    }
    return out;
  };
  // On a matrix-core box the int8 tiers too: the w4 blocks run signed
  // and (block 0 being w8 in the pack this was written against) the w8
  // ones the fast / token layout, so the slots cross that layout in
  // both directions.
  std::vector<int> tiers = {-1};
  if (mc->supports_matrix_cores()) { tiers.insert(tiers.end(), {0, 1, 2}); }
  const char* tname[4] = {"accurate (signed)", "fast", "token",
                          "token_plain"};
  for (int tier : tiers) {
    engaged = true;
    promoted = 0;
    const auto pre = run(tier, false, true, false);
    MP_REQUIRE(!pre.empty());
    if (tier > 0) {
      // BOTH WIDTHS take the tier: a w8 block's codes converted once,
      // a w4 one's into the scratch before each GEMM. Uniform w8 packs:
      // fast_tier_matches_accurate.
      std::printf("[moss_image_dit] mixed pack, %s: %d w8 and %d w4 "
                  "projections in its layout\n", tname[tier], f8, f4);
      EXPECT_TRUE(fast_on && f8 > 0 && f4 > 0);
      if (!fast_on) { continue; }
    }
    const auto str = run(tier, true, true, false);
    const auto flat = run(tier, true, false, false);
    const auto grown = run(tier, true, true, true);
    const std::uint64_t grown_pc = grown_bytes;
    // A HELD layout reached by PROMOTION: residency converts each w4
    // block it keeps (clone_block_), which then holds more than the
    // pack -- and must equal the held preload.
    const auto grown_held =
        tier > 0 ? run(tier, true, true, true, true) : grown;
    const std::uint64_t grown_h = grown_bytes;
    const int grown_n = promoted;
    // ...and with NO SLOTS, where promote_into MOVES each per-block read
    // into the resident set: the keep pass (keep_block_) must leave it
    // holding exactly what the slots' clone does.
    const auto flat_held =
        tier > 0 ? run(tier, true, false, true, true) : grown;
    const std::uint64_t flat_h = tier > 0 ? grown_bytes : grown_h;
    const int flat_n = tier > 0 ? promoted : grown_n;
    // The bytes compare only where BOTH kept the whole streamed stack:
    // a slotted run short of it still holds its slot pair, and a box
    // with little room (a 16 GB M5) keeps what its budget allows.
    const bool whole = grown_n == stack && flat_n == stack;
    // The engine's other answer for a w4 pack: its int8 layout
    // HELD (Config::i8_hold_w4) where it is converted per GEMM
    // above -- the same values, so the same bytes.
    const auto held =
        tier > 0 ? run(tier, false, true, false, true) : pre;
    const double rel = rel_l2_(pre, want);
    std::printf("[moss_image_dit] mixed pack 512^2, %s: rel-L2 %.5f from "
                "the fp32 reference; streamed %s, no slots %s, promoted (%d "
                "blocks) %s%s\n", tier < 0 ? "no i8" : tname[tier], rel,
                same_(str, pre) ? "IDENTICAL" : "DIFFERENT",
                same_(flat, pre) ? "IDENTICAL" : "DIFFERENT", promoted,
                same_(grown, pre) ? "IDENTICAL" : "DIFFERENT",
                engaged ? "" : " -- NOT ENGAGED");
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(same_(str, pre));
    EXPECT_TRUE(same_(flat, pre));
    EXPECT_TRUE(promoted > 0 && same_(grown, pre));
    if (tier > 0) {
      std::printf("[moss_image_dit] ... w4 layout held: %s; held by "
                  "promotion %s (%llu MB resident against %llu per "
                  "call); with no slots %s (%llu MB)\n",
                  same_(held, pre) ? "IDENTICAL" : "DIFFERENT",
                  same_(grown_held, pre) ? "IDENTICAL" : "DIFFERENT",
                  (unsigned long long)(grown_h >> 20),
                  (unsigned long long)(grown_pc >> 20),
                  same_(flat_held, pre) ? "IDENTICAL" : "DIFFERENT",
                  (unsigned long long)(flat_h >> 20));
      EXPECT_TRUE(same_(grown_held, pre) && grown_h > grown_pc);
      // The slots' resident figure carries no slots once residency is
      // whole (release_if_saturated); the moved blocks must hold the
      // same bytes the cloned ones do.
      EXPECT_TRUE(same_(flat_held, pre) && (!whole || flat_h == grown_h));
    }
    EXPECT_TRUE(same_(held, pre));
    // The slots crossed widths and REBUILT. A slot that refused such a
    // block instead turns itself off, and its holds still agree.
    std::printf("[moss_image_dit] ... %d streamed blocks built into their "
                "slot\n", relaid);
    EXPECT_TRUE(relaid > 0);
    EXPECT_TRUE(rel < 0.3);
  }
}

// THE INT8 TIERS ON THE PACK (matrix-core boxes): the signed layout
// (accurate) and the fast / token tiers each hold the block projections
// in their own layout -- converted in the derived entry a preloaded
// block keeps, in the slot's post_refill for a streamed one -- so the
// four holds (preloaded, streamed, no slots, promoted) must be
// BYTE-IDENTICAL; and each stays within a gross bar of the dense
// forward. Six blocks of the real DiT at 1280 + 64 rows.
TEST(moss_image_dit, i8_holds_agree_on_a_w8_pack)
{
  const std::string qd = env_("VPIPE_MOSS_IMAGE_QUANT_DIT");
  if (qd.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  MetalMossImageTransformer::Config base;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(qd, &base));
  base.n_layers = 6;
  const int gh = 32, gw = 40, T = 64;
  bool engaged = true;
  auto run = [&](bool i8, int tier, bool stream, bool grow, bool slots) {
    std::unique_ptr<Env> ns;
    if (!slots) {
      ns = std::make_unique<Env>("VPIPE_MOSS_IMAGE_NO_SLOTS", "1");
    }
    MetalMossImageTransformer::Config c = base;
    c.i8_gemm = i8;
    c.i8_tier = tier;
    auto m = MetalMossImageTransformer::load(qd, mc, c, stream);
    std::vector<float> out;
    if (m == nullptr) { return out; }
    if (i8 && tier == 0 && !m->signed_layout()) { engaged = false; }
    if (i8 && tier > 0 && !m->fast_tier()) { engaged = false; }
    if (grow) {
      m->set_residency_reserve(test_residency_reserve(4ull << 30));
      m->set_residency_schedule(4);
    }
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = random_forward_(m.get(), mc, gh, gw, T, 20261018u);
    }
    if (grow && m->resident_block_count() == 0) { engaged = false; }
    return out;
  };
  const std::vector<float> dense = run(false, 0, false, false, true);
  MP_REQUIRE(!dense.empty());
  const char* tname[4] = {"accurate (signed)", "fast", "token",
                          "token_plain"};
  double ra = 0.0;
  for (int tier = 0; tier < 4; ++tier) {
    engaged = true;
    const auto pre = run(true, tier, false, false, true);
    const auto str = run(true, tier, true, false, true);
    const auto flat = run(true, tier, true, false, false);
    const auto grown = run(true, tier, true, true, true);
    MP_REQUIRE(pre.size() == dense.size());
    const double r = rel_l2_(pre, dense);
    if (tier == 0) { ra = r; }
    std::printf("[moss_image_dit] w8 pack, %s tier: vs dense %.3e; "
                "streamed %s, no slots %s, promoted %s%s\n", tname[tier], r,
                same_(str, pre) ? "IDENTICAL" : "DIFFERENT",
                same_(flat, pre) ? "IDENTICAL" : "DIFFERENT",
                same_(grown, pre) ? "IDENTICAL" : "DIFFERENT",
                engaged ? "" : " -- NOT ENGAGED");
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(same_(str, pre));
    EXPECT_TRUE(same_(flat, pre));
    EXPECT_TRUE(same_(grown, pre));
    // A gross-error bar (random-input forwards are chaotic): what it
    // catches is a misread layout, not a ranking of the tiers.
    EXPECT_TRUE(r < (tier == 0 ? 0.25 : 6.0 * ra));
  }
}

// THE INT8 TIERS ON THE BF16 CHECKPOINT (VPIPE_MOSS_IMAGE_TEST_MODEL_PATH,
// matrix-core boxes): a preloaded block quantizes each projection once
// into the entry it keeps; a STREAMED one keeps the bf16 as its slot's
// refill target and quantizes it after every refill (QWeight::ds), and a
// promotion keeps the layout alone -- so the four holds (preloaded,
// streamed, no slots, promoted) must be BYTE-IDENTICAL on every tier
// that holds a layout, and the promoted blocks must BE in it. Six blocks
// of the real DiT at 1280 + 64 rows: past the int8 route's 1024-row
// floor, K = 3072 past its width floor.
TEST(moss_image_dit, bf16_int8_holds_agree)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const std::string dir = root + "/transformer";
  MetalMossImageTransformer::Config base;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(dir, &base));
  base.n_layers = 6;
  const int gh = 32, gw = 40, T = 64;
  bool engaged = true;
  int laid = 0;
  auto run = [&](bool i8, int tier, bool stream, bool grow, bool slots) {
    std::unique_ptr<Env> ns;
    if (!slots) {
      ns = std::make_unique<Env>("VPIPE_MOSS_IMAGE_NO_SLOTS", "1");
    }
    MetalMossImageTransformer::Config c = base;
    c.i8_gemm = i8;
    c.i8_tier = tier;
    auto m = MetalMossImageTransformer::load(dir, mc, c, stream);
    std::vector<float> out;
    if (m == nullptr) { return out; }
    if (i8 && tier > 0 && !m->fast_tier()) { engaged = false; }
    if (grow) {
      m->set_residency_reserve(test_residency_reserve(4ull << 30));
      m->set_residency_schedule(4);
    }
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = random_forward_(m.get(), mc, gh, gw, T, 20261018u);
    }
    if (grow && m->resident_block_count() == 0) { engaged = false; }
    // Resident blocks in the int8 layout: every projection of a
    // preloaded block, and of a promoted one -- which kept the layout,
    // not its slot's dense source.
    if (!stream || grow) { laid = m->fast_weights(8); }
    return out;
  };
  const std::vector<float> dense = run(false, 0, false, false, true);
  const std::vector<float> acc = run(true, 0, false, false, true);
  MP_REQUIRE(!dense.empty() && acc.size() == dense.size());
  const double ra = rel_l2_(acc, dense);
  const char* tname[4] = {"accurate", "fast", "token", "token_plain"};
  for (int tier = 1; tier < 4; ++tier) {
    engaged = true;
    const auto pre = run(true, tier, false, false, true);
    const int laid_pre = laid;
    const auto str = run(true, tier, true, false, true);
    const auto flat = run(true, tier, true, false, false);
    const auto grown = run(true, tier, true, true, true);
    const int laid_grown = laid;
    // NO SLOTS and grown: promote_into MOVES each per-block read, so the
    // keep pass is what drops the dense source (keep_block_).
    const auto flat_grown = run(true, tier, true, true, false);
    const int laid_flat = laid;
    MP_REQUIRE(pre.size() == dense.size());
    const double r = rel_l2_(pre, dense);
    std::printf("[moss_image_dit] bf16, %s tier: vs dense %.3e (accurate "
                "%.3e); streamed %s, no slots %s, promoted %s, promoted "
                "with no slots %s; %d / %d / %d projections in the layout "
                "preloaded / promoted / moved%s\n",
                tname[tier], r, ra,
                same_(str, pre) ? "IDENTICAL" : "DIFFERENT",
                same_(flat, pre) ? "IDENTICAL" : "DIFFERENT",
                same_(grown, pre) ? "IDENTICAL" : "DIFFERENT",
                same_(flat_grown, pre) ? "IDENTICAL" : "DIFFERENT",
                laid_pre, laid_grown, laid_flat,
                engaged ? "" : " -- NOT ENGAGED");
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(laid_pre > 0 && laid_grown > 0 && laid_flat > 0);
    EXPECT_TRUE(same_(str, pre));
    EXPECT_TRUE(same_(flat, pre));
    EXPECT_TRUE(same_(grown, pre));
    EXPECT_TRUE(same_(flat_grown, pre));
    // A gross-error bar (random-input forwards are chaotic): what it
    // catches is a misread layout, not a ranking of the tiers.
    EXPECT_TRUE(r < 6.0 * ra);
  }
}

// THE ANE TIERS' ROWS (VPIPE_MOSS_IMAGE_ANE_TEST=1; a box with an ANE,
// judged on the M4 series): three blocks of the real DiT at 1024^2
// (4096 + 64 rows, two 2048-row chunks to split), the ANE's SITU
// feed-forward and then q / k / v beside it, each against the GPU-only
// forward -- and again under a synthetic adapter on every projection, which
// checks the fused [hidden | gate] matrix's row offsets for both the
// weight and the adapter's B (a wrong one adds the wrong rows on the ANE's
// rows only). Twice per run: the first forward probes, the second splits.
TEST(moss_image_dit, ane_tiers_rows_are_right)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  if (root.empty() || env_("VPIPE_MOSS_IMAGE_ANE_TEST").empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = root + "/transformer";
  MetalMossImageTransformer::Config cfg;
  MP_REQUIRE(MetalMossImageTransformer::Config::read_dims(tdir, &cfg));
  cfg.n_layers = 3;
  const int gh = 64, gw = 64, T = 64;
  using Specs = std::vector<MetalMossImageTransformer::LoraSpec>;
  int bound = -1;
  auto run = [&](bool ffn, bool qkv, bool* armed,
                 const Specs& loras = {}) -> std::vector<float> {
    MetalMossImageTransformer::Config c = cfg;
    c.ane_ffn = ffn;
    c.ane_qkv = qkv;
    c.ane_rows = (ffn || qkv) ? 1 : 0;   // pinned: the whole first chunk
    auto m = MetalMossImageTransformer::load(tdir, mc, c, false, loras);
    if (m == nullptr) { return {}; }
    if (!loras.empty()) { bound = m->lora_modules(0); }
    std::vector<float> out = random_forward_(m.get(), mc, gh, gw, T, 77u);
    out = random_forward_(m.get(), mc, gh, gw, T, 77u);
    if (armed != nullptr) {
      *armed = (!ffn || (m->ane_armed() && !m->ane_gpu_only())) &&
               (!qkv || (m->ane_qkv_armed() && !m->ane_qkv_gpu_only()));
    }
    return out;
  };
  bool armed = false;
  const std::vector<float> gpu = run(false, false, nullptr);
  const std::vector<float> ff = run(true, false, &armed);
  MP_REQUIRE(!gpu.empty() && ff.size() == gpu.size());
  if (!armed) {
    std::printf("[moss_image_dit] the ANE tier did not arm here; skipped\n");
    return;
  }
  bool armed2 = false;
  const std::vector<float> both = run(true, true, &armed2);
  MP_REQUIRE(both.size() == gpu.size() && armed2);
  const double d1 = rel_l2_(ff, gpu), d2 = rel_l2_(both, gpu);
  std::printf("[moss_image_dit] ANE at %d rows, 3 blocks: feed-forward "
              "rel-L2 vs the GPU %.6f, + q/k/v %.6f\n", gh * gw + T, d1, d2);
  EXPECT_TRUE(d1 > 0.0 && d1 < 0.02);
  EXPECT_TRUE(d2 > 0.0 && d2 < 0.02);

  // AND UNDER A LoRA: rank 8 on all six projections of the three blocks.
  const int R = 8, H = cfg.hidden, F = cfg.ffn_inner;
  unsigned st = 99u;
  auto rnd = [&](std::size_t n, float k) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * k;
      const std::uint16_t h = genai::train::f32_to_bf16(v);
      std::memcpy(out.data() + 2 * i, &h, 2);
    }
    return out;
  };
  std::vector<genai::train::OutTensor> ts;
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".proj.";
    const struct { const char* leaf; int n, k; } mods[] = {
        {"q", H, H}, {"k", H, H}, {"v", H, H},
        {"ffn_up_gate", 2 * F, H}, {"ffn_down", H, F}};
    for (const auto& md : mods) {
      ts.push_back({p + md.leaf + ".lora_A.weight", "BF16", {R, md.k},
                    rnd((std::size_t)R * md.k, 0.25f)});
      ts.push_back({p + md.leaf + ".lora_B.weight", "BF16", {md.n, R},
                    rnd((std::size_t)md.n * R, 0.25f)});
    }
  }
  namespace fs = std::filesystem;
  const fs::path lp = fs::temp_directory_path() / "vpipe-moss-ane-lora" /
                      "adapter.safetensors";
  std::error_code ec;
  fs::create_directories(lp.parent_path(), ec);
  std::string werr;
  MP_REQUIRE(genai::train::write_safetensors(lp.string(), ts, {}, &werr));
  const Specs specs{{lp.string(), 1.0f}};
  const std::vector<float> gpu_l = run(false, false, nullptr, specs);
  MP_REQUIRE(bound == 5 * cfg.n_layers);
  bool armed_l = false;
  const std::vector<float> ane_l = run(true, true, &armed_l, specs);
  fs::remove_all(lp.parent_path(), ec);
  MP_REQUIRE(!gpu_l.empty() && ane_l.size() == gpu_l.size() && armed_l);
  const double moved = rel_l2_(gpu_l, gpu);
  const double dl = rel_l2_(ane_l, gpu_l);
  std::printf("[moss_image_dit] ... under a LoRA (%d modules, moving the GPU "
              "output %.4f): ANE vs GPU %.6f\n", bound, moved, dl);
  EXPECT_TRUE(moved > 0.1);
  EXPECT_TRUE(dl > 0.0 && dl < 0.02);
}

// A BENCHMARK: the VAE decode at VPIPE_MOSS_IMAGE_VAE_BENCH=<side> -- a
// warm decode, then three timed. VPIPE_VAE_BENCH_DIR names another
// AutoencoderKLFlux2 dir to time instead (FLUX.2's full-width decoder,
// to set the small one against).
TEST(moss_image_vae, bench_decode)
{
  const std::string root = env_("VPIPE_MOSS_IMAGE_TEST_MODEL_PATH");
  const std::string side = env_("VPIPE_MOSS_IMAGE_VAE_BENCH");
  if (side.empty()) { return; }
  std::string vdir = env_("VPIPE_VAE_BENCH_DIR");
  if (vdir.empty()) {
    if (root.empty()) { return; }
    vdir = root + "/vae";
  }
  const int S = std::atoi(side.c_str());
  if (S < 256 || (S % 16) != 0) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
  MP_REQUIRE(vae != nullptr);
  const int h16 = S / 16, w16 = S / 16;
  const std::size_t n = (std::size_t)128 * h16 * w16;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  auto* zd = static_cast<_Float16*>(z.contents());
  unsigned st = 5u;
  for (std::size_t i = 0; i < n; ++i) {
    st = st * 1664525u + 1013904223u;
    zd[i] = (_Float16)((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f);
  }
  std::string err;
  MP_REQUIRE(!vae->decode(z, h16, w16, &err).empty());
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 3; ++i) {
    MP_REQUIRE(!vae->decode(z, h16, w16, &err).empty());
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / 3;
  const auto& c = vae->config();
  std::printf("[moss_image_vae_bench] %dx%d decode: %.0f ms (decoder widths "
              "%d/%d/%d/%d, %.1f GFLOP)\n", S, S, ms, c.dec_block_out[0],
              c.dec_block_out[1], c.dec_block_out[2], c.dec_block_out[3],
              (double)vae->gpu_flops().flushed() / 4.0 / 1e9);
}
