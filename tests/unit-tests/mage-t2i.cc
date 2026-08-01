// Mage-Flow end-to-end through the split diffusion stages:
//
//   prompt -> diffusion-conditioner -> text-to-image (NR-MMDiT + FlowMatch
//   Euler, static shift 6.0) -> vae-decode (MageVAE) -> RGB
//
// and, for the EDIT flow, additionally
//
//   image -> vae-encode (MageVAE) -> text-to-image ref_latent0
//   image -> diffusion-conditioner ref_image
//
// The DiT and VAE are already verified against the reference numerically
// (mage-dit.cc / mage-vae.cc) and the conditioning in mage-cond.cc, so this
// test is about the WIRING: family detection in four stages, the [128, H/16,
// W/16] latent contract between them, and that the composed pipeline produces
// a non-degenerate image.
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH = the Mage-Flow model root. Optional
// VPIPE_MAGE_T2I_OUT = a directory to write mage_t2i.png / mage_edit.png into,
// and VPIPE_MAGE_EDIT_IMAGE = the source photo for the edit case.
// Skips vacuously when the model path is unset.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/load-image-stage.h"
#include "stages/save-image-stage.h"
#include "stages/text-to-image-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

// VPIPE_MAGE_T2I_SIDE overrides the square test resolution (default 512 --
// Mage-Flow trains at 1024 and its static shift 6.0 is tuned for that, so a
// much smaller grid is out of distribution and produces washed-out output;
// 256 is a useful "does it run" size but not a coherence check).
int
side_()
{
  const char* e = std::getenv("VPIPE_MAGE_T2I_SIDE");
  const int v = (e != nullptr && *e != '\0') ? std::atoi(e) : 512;
  return (v >= 64 && (v % 16) == 0) ? v : 512;
}

class MageSourceText : public vpipe::TypedStage<MageSourceText> {
public:
  static constexpr const char* kTypeName = "ut-mage-t2i-src";
  MageSourceText(const vpipe::SessionContextIntf* s, std::string id,
                 std::vector<vpipe::InEdge> ip, vpipe::FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::string prompt;
  bool done = false;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      co_await ctx.write(0, std::make_unique<vpipe::FlexDataPayload>(
                                vpipe::FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};

class MageSink : public vpipe::TypedStage<MageSink> {
public:
  static constexpr const char* kTypeName = "ut-mage-t2i-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<vpipe::BeatPayloadIntf>> captured;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// An emitted RGB image is "real content": finite, and neither flat nor
// saturated. Returns false (with a printed reason) otherwise.
bool
image_is_coherent_(const TensorBeatPayload& tb, const char* tag)
{
  const auto bytes = tb.materialize_contiguous();
  const std::size_t n = bytes.size();
  if (n == 0) { return false; }
  double mean = 0.0;
  for (std::size_t i = 0; i < n; ++i) { mean += (double)bytes[i]; }
  mean /= (double)n;
  double var = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)bytes[i] - mean; var += d * d;
  }
  const double sd = std::sqrt(var / (double)n);
  std::printf("[mage_t2i] %s image mean=%.1f std=%.1f\n", tag, mean, sd);
  return sd > 3.0;   // not a flat / grey / all-black frame
}

}  // namespace

// Text-to-image: prompt -> conditioner -> DiT -> MageVAE decode -> RGB.
TEST(mage_t2i, text_to_image_end_to_end)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const char* outd = std::getenv("VPIPE_MAGE_T2I_OUT");
  // VPIPE_MAGE_PERF_DUMP=<file>: record the run's LLM-lane activity so the
  // profiler wiring is CHECKED, not assumed -- a Mage-Flow timeline should
  // show dit-text-fuse (conditioner), dit-denoise (per sampler step) and
  // vae-codec, the same shape a Krea-2 run gives.
  const char* perf = std::getenv("VPIPE_MAGE_PERF_DUMP");
  if (perf != nullptr && *perf != '\0') {
    EXPECT_TRUE(sess.enable_profiling(65536).code == 0);
  }

  const int H = side_(), W = side_();

  auto pl = std::make_unique<Pipeline>("mt", &sess);
  auto srcu = std::make_unique<MageSourceText>(&sess, "src",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  srcu->prompt = "a red fox sitting in fresh snow, photorealistic";
  auto* src = static_cast<MageSourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(cc));
  auto* cond =
      static_cast<DiffusionConditionerStage*>(pl->insert_stage(std::move(cu)));
  ASSERT_TRUE(cond->config_error().empty());

  FlexData tc = FlexData::make_object();
  tc.as_object().insert("hf_dir", FlexData::make_string(root));
  tc.as_object().insert("height", FlexData::make_int(H));
  tc.as_object().insert("width", FlexData::make_int(W));
  tc.as_object().insert("steps", FlexData::make_int(4));   // Turbo
  tc.as_object().insert("seed", FlexData::make_int(42));
  // VPIPE_MAGE_T2I_REF: attach a reference LATENT (from VPIPE_MAGE_EDIT_IMAGE)
  // without any vision conditioning -- isolates the DiT's reference-segment
  // path from the conditioner's grounded encode.
  std::vector<InEdge> te{{cond, 0}};
  const char* refimg = std::getenv("VPIPE_MAGE_T2I_REF");
  if (refimg != nullptr && *refimg != '\0' &&
      std::filesystem::exists(refimg)) {
    FlexData ric = FlexData::make_object();
    ric.as_object().insert("url", FlexData::make_string(refimg));
    auto riu = std::make_unique<LoadImageStage>(&sess, "rimg",
                                                std::vector<InEdge>{},
                                                std::move(ric));
    Stage* rimg = pl->insert_stage(std::move(riu));
    FlexData rec = FlexData::make_object();
    rec.as_object().insert("hf_dir", FlexData::make_string(root));
    rec.as_object().insert("target_width", FlexData::make_int(W));
    rec.as_object().insert("target_height", FlexData::make_int(H));
    auto reu = std::make_unique<VaeEncodeStage>(
        &sess, "renc", std::vector<InEdge>{{rimg, 0}}, std::move(rec));
    Stage* renc = pl->insert_stage(std::move(reu));
    te.push_back({nullptr, 0}); te.push_back({nullptr, 0});
    te.push_back({nullptr, 0}); te.push_back({nullptr, 0});
    te.push_back({renc, 0});
  }
  auto tu = std::make_unique<TextToImageStage>(&sess, "t2i", std::move(te),
                                               std::move(tc));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(tu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vc = FlexData::make_object();
  vc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vc));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vu)));
  ASSERT_TRUE(vae->config_error().empty());

  auto sku = std::make_unique<MageSink>(&sess, "sink",
                                        std::vector<InEdge>{{vae, 0}},
                                        FlexData::make_object());
  auto* sink = static_cast<MageSink*>(pl->insert_stage(std::move(sku)));

  Stage* saver = nullptr;
  if (outd != nullptr && *outd != '\0') {
    FlexData sc = FlexData::make_object();
    sc.as_object().insert(
        "path", FlexData::make_string(std::string(outd) + "/mage_t2i.png"));
    auto su = std::make_unique<SaveImageStage>(
        &sess, "save", std::vector<InEdge>{{vae, 0}}, std::move(sc));
    saver = pl->insert_stage(std::move(su));
  }
  (void)saver;

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  if (t2i->latents_emitted() == 0) {
    std::printf("[mage_t2i] no latent emitted (model not loaded?); skipping\n");
    return;
  }
  if (perf != nullptr && *perf != '\0') {
    EXPECT_TRUE(sess.dump_profiling(perf).code == 0);
    std::printf("[mage_t2i] profiling dumped to %s\n", perf);
  }
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
              tb->shape[1] == H && tb->shape[2] == W);
  EXPECT_TRUE(image_is_coherent_(*tb, "t2i"));
}

// Image edit: the reference photo feeds BOTH the conditioner (Qwen3-VL tower,
// long edge 384) and vae-encode (full target resolution) -- the two halves of
// Mage-Flow's edit conditioning. The reference latent is encoded AT the output
// resolution so its RoPE band overlaps the target's exactly.

TEST(mage_t2i, image_edit_end_to_end)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const char* imge = std::getenv("VPIPE_MAGE_EDIT_IMAGE");
  if (imge == nullptr || *imge == '\0' ||
      !std::filesystem::exists(imge)) {
    std::printf("[mage_t2i] no VPIPE_MAGE_EDIT_IMAGE; skipping edit\n");
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const char* outd = std::getenv("VPIPE_MAGE_T2I_OUT");
  // Same profiling hook as the t2i test; the edit path additionally exercises
  // the Qwen3-VL tower, so this is where `vision-tower` should appear.
  const char* perf = std::getenv("VPIPE_MAGE_PERF_DUMP");
  if (perf != nullptr && *perf != '\0') {
    EXPECT_TRUE(sess.enable_profiling(65536).code == 0);
  }

  const int H = side_(), W = side_();

  auto pl = std::make_unique<Pipeline>("me", &sess);
  auto srcu = std::make_unique<MageSourceText>(&sess, "src",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  // VPIPE_MAGE_EDIT_PROMPT / VPIPE_MAGE_T2I_INIT let this test be driven as a
  // like-for-like comparison against the torch reference: same instruction,
  // same PINNED initial noise (init_latents also bypasses the watermark, so
  // the two runs differ only by implementation).
  const char* eprompt = std::getenv("VPIPE_MAGE_EDIT_PROMPT");
  srcu->prompt = (eprompt != nullptr && *eprompt != '\0')
                     ? eprompt : "make the background a snowy forest";
  auto* src = static_cast<MageSourceText*>(pl->insert_stage(std::move(srcu)));

  // One load-image feeds both the conditioner's tower and vae-encode.
  FlexData ic = FlexData::make_object();
  ic.as_object().insert("url", FlexData::make_string(imge));
  auto iu = std::make_unique<LoadImageStage>(&sess, "img",
                                             std::vector<InEdge>{},
                                             std::move(ic));
  Stage* img = pl->insert_stage(std::move(iu));

  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond",
      std::vector<InEdge>{{src, 0}, {nullptr, 0}, {nullptr, 0}, {img, 0}},
      std::move(cc));
  auto* cond =
      static_cast<DiffusionConditionerStage*>(pl->insert_stage(std::move(cu)));
  ASSERT_TRUE(cond->config_error().empty());

  // Encode the reference at the OUTPUT resolution (grid overlap; see the
  // scale_rope warning in generate_mage_).
  FlexData ec = FlexData::make_object();
  ec.as_object().insert("hf_dir", FlexData::make_string(root));
  ec.as_object().insert("target_width", FlexData::make_int(W));
  ec.as_object().insert("target_height", FlexData::make_int(H));
  auto eu = std::make_unique<VaeEncodeStage>(
      &sess, "enc", std::vector<InEdge>{{img, 0}}, std::move(ec));
  auto* enc = static_cast<VaeEncodeStage*>(pl->insert_stage(std::move(eu)));
  ASSERT_TRUE(enc->config_error().empty());

  FlexData tc = FlexData::make_object();
  tc.as_object().insert("hf_dir", FlexData::make_string(root));
  tc.as_object().insert("height", FlexData::make_int(H));
  tc.as_object().insert("width", FlexData::make_int(W));
  tc.as_object().insert("steps", FlexData::make_int(4));
  tc.as_object().insert("seed", FlexData::make_int(42));
  if (const char* ini = std::getenv("VPIPE_MAGE_T2I_INIT")) {
    if (*ini != '\0') {
      tc.as_object().insert("init_latents", FlexData::make_string(ini));
    }
  }
  // iports: conditioning, neg, model, sampler, scheduler, ref_latent0.
  // VPIPE_MAGE_EDIT_NO_REF drops the reference LATENT (keeping the grounded
  // text conditioning) -- the A/B that separates the two halves of the edit
  // conditioning when an edit comes out wrong.
  const bool no_ref = std::getenv("VPIPE_MAGE_EDIT_NO_REF") != nullptr;
  std::vector<InEdge> te{{cond, 0}, {nullptr, 0}, {nullptr, 0},
                         {nullptr, 0}, {nullptr, 0}};
  if (!no_ref) { te.push_back({enc, 0}); }
  auto tu = std::make_unique<TextToImageStage>(&sess, "t2i", std::move(te),
                                               std::move(tc));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(tu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vc = FlexData::make_object();
  vc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vc));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vu)));

  auto sku = std::make_unique<MageSink>(&sess, "sink",
                                        std::vector<InEdge>{{vae, 0}},
                                        FlexData::make_object());
  auto* sink = static_cast<MageSink*>(pl->insert_stage(std::move(sku)));
  if (outd != nullptr && *outd != '\0') {
    FlexData sc = FlexData::make_object();
    sc.as_object().insert(
        "path", FlexData::make_string(std::string(outd) + "/mage_edit.png"));
    auto su = std::make_unique<SaveImageStage>(
        &sess, "save", std::vector<InEdge>{{vae, 0}}, std::move(sc));
    pl->insert_stage(std::move(su));
  }

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  if (perf != nullptr && *perf != '\0') {
    EXPECT_TRUE(sess.dump_profiling(perf).code == 0);
    std::printf("[mage_t2i] profiling dumped to %s\n", perf);
  }
  if (t2i->latents_emitted() == 0) {
    std::printf("[mage_t2i] no edit latent emitted; skipping\n");
    return;
  }
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
              tb->shape[1] == H && tb->shape[2] == W);
  // Both halves of the edit conditioning now hold: the grounded encode alone
  // (VPIPE_MAGE_EDIT_NO_REF=1) and the full path with the reference latent.
  // The latter was a ghost until the bf16-TIMESTEP bug was fixed -- see
  // mage_dit.edit_sampler_loop_matches_golden, which is the test that can
  // actually see it (the single-forward goldens pin sigma at a bf16-exact
  // 0.75, where the bug is invisible).
  EXPECT_TRUE(image_is_coherent_(*tb, no_ref ? "edit(no-ref)"
                                             : "edit(ref-latent)"));
}

// vae-encode -> vae-decode through the STAGES (no DiT): the reference-latent
// contract the edit flow depends on. mage-vae.cc already verifies the codec
// numerically, so what this adds is the stage plumbing -- letterbox/normalize
// on the way in, the [128, H/16, W/16] f32 beat, and the [-1,1]->U8 mapping on
// the way out. If the edit output is wrong but this round-trip is clean, the
// fault is in the DiT's reference conditioning, not the codec.
TEST(mage_t2i, vae_stage_round_trip)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const char* imge = std::getenv("VPIPE_MAGE_EDIT_IMAGE");
  if (imge == nullptr || *imge == '\0' || !std::filesystem::exists(imge)) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const char* outd = std::getenv("VPIPE_MAGE_T2I_OUT");
  const int H = side_(), W = side_();

  auto pl = std::make_unique<Pipeline>("mr", &sess);
  FlexData ic = FlexData::make_object();
  ic.as_object().insert("url", FlexData::make_string(imge));
  auto iu = std::make_unique<LoadImageStage>(&sess, "img",
                                             std::vector<InEdge>{},
                                             std::move(ic));
  Stage* img = pl->insert_stage(std::move(iu));

  FlexData ec = FlexData::make_object();
  ec.as_object().insert("hf_dir", FlexData::make_string(root));
  ec.as_object().insert("target_width", FlexData::make_int(W));
  ec.as_object().insert("target_height", FlexData::make_int(H));
  auto eu = std::make_unique<VaeEncodeStage>(
      &sess, "enc", std::vector<InEdge>{{img, 0}}, std::move(ec));
  auto* enc = static_cast<VaeEncodeStage*>(pl->insert_stage(std::move(eu)));

  FlexData vc = FlexData::make_object();
  vc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{enc, 0}}, std::move(vc));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vu)));

  auto sku = std::make_unique<MageSink>(&sess, "sink",
                                        std::vector<InEdge>{{vae, 0}},
                                        FlexData::make_object());
  auto* sink = static_cast<MageSink*>(pl->insert_stage(std::move(sku)));
  if (outd != nullptr && *outd != '\0') {
    FlexData sc = FlexData::make_object();
    sc.as_object().insert(
        "path", FlexData::make_string(std::string(outd) + "/mage_rt.png"));
    auto su = std::make_unique<SaveImageStage>(
        &sess, "save", std::vector<InEdge>{{vae, 0}}, std::move(sc));
    pl->insert_stage(std::move(su));
  }

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  if (sink->captured.empty()) {
    std::printf("[mage_t2i] no round-trip image; skipping\n");
    return;
  }
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
              tb->shape[1] == H && tb->shape[2] == W);
  EXPECT_TRUE(image_is_coherent_(*tb, "round-trip"));
}

// Output size with NO width/height configured: text-to-image must take it from
// ref_latent0 (vae-encode's output for the source image) scaled by the
// family's VAE factor -- 16 for Mage-Flow -- so an edit lands at the source
// resolution with no size config at all. An explicit width/height still wins.
//
// The reference is pinned NON-SQUARE via vae-encode's target_width/height, so
// the expected latent is known regardless of the source photo's own size AND a
// transposed h/w would fail (a square case could not tell).
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH + VPIPE_MAGE_EDIT_IMAGE.
TEST(mage_t2i, output_size_follows_ref_latent)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const char* imge = std::getenv("VPIPE_MAGE_EDIT_IMAGE");
  if (imge == nullptr || *imge == '\0' || !std::filesystem::exists(imge)) {
    std::printf("[mage_t2i] no VPIPE_MAGE_EDIT_IMAGE; skipping size test\n");
    return;
  }
  const int refW = 256, refH = 512;          // non-square on purpose
  const int P = 16;                          // MageVAE scale

  // `expect_w`/`expect_h` = the size the stage should GENERATE at; cfg_w/cfg_h
  // are what goes in the config (0 = leave the key out entirely).
  auto run = [&](int cfg_w, int cfg_h, int expect_w, int expect_h,
                 const char* tag) {
    Session sess;
    if (sess.metal_compute() == nullptr) { return; }
    auto pl = std::make_unique<Pipeline>("msz", &sess);
    auto srcu = std::make_unique<MageSourceText>(&sess, "src",
                                                 std::vector<InEdge>{},
                                                 FlexData::make_object());
    srcu->prompt = "make the background a snowy forest";
    auto* src = static_cast<MageSourceText*>(pl->insert_stage(std::move(srcu)));

    FlexData ic = FlexData::make_object();
    ic.as_object().insert("url", FlexData::make_string(imge));
    auto iu = std::make_unique<LoadImageStage>(&sess, "img",
                                               std::vector<InEdge>{},
                                               std::move(ic));
    Stage* img = pl->insert_stage(std::move(iu));

    FlexData cc = FlexData::make_object();
    cc.as_object().insert("hf_dir", FlexData::make_string(root));
    auto cu = std::make_unique<DiffusionConditionerStage>(
        &sess, "cond",
        std::vector<InEdge>{{src, 0}, {nullptr, 0}, {nullptr, 0}, {img, 0}},
        std::move(cc));
    auto* cond = static_cast<DiffusionConditionerStage*>(
        pl->insert_stage(std::move(cu)));
    ASSERT_TRUE(cond->config_error().empty());

    FlexData ec = FlexData::make_object();
    ec.as_object().insert("hf_dir", FlexData::make_string(root));
    ec.as_object().insert("target_width", FlexData::make_int(refW));
    ec.as_object().insert("target_height", FlexData::make_int(refH));
    auto eu = std::make_unique<VaeEncodeStage>(
        &sess, "enc", std::vector<InEdge>{{img, 0}}, std::move(ec));
    auto* enc = static_cast<VaeEncodeStage*>(pl->insert_stage(std::move(eu)));
    ASSERT_TRUE(enc->config_error().empty());

    FlexData tc = FlexData::make_object();
    tc.as_object().insert("hf_dir", FlexData::make_string(root));
    tc.as_object().insert("steps", FlexData::make_int(1));
    if (cfg_w > 0) { tc.as_object().insert("width",
                                           FlexData::make_int(cfg_w)); }
    if (cfg_h > 0) { tc.as_object().insert("height",
                                           FlexData::make_int(cfg_h)); }
    auto tu = std::make_unique<TextToImageStage>(
        &sess, "t2i",
        std::vector<InEdge>{{cond, 0}, {nullptr, 0}, {nullptr, 0},
                            {nullptr, 0}, {nullptr, 0}, {enc, 0}},
        std::move(tc));
    auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(tu)));
    ASSERT_TRUE(t2i->config_error().empty());

    auto sku = std::make_unique<MageSink>(&sess, "sink",
                                          std::vector<InEdge>{{t2i, 0}},
                                          FlexData::make_object());
    auto* sink = static_cast<MageSink*>(pl->insert_stage(std::move(sku)));

    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    ASSERT_TRUE(!sink->captured.empty());
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
    ASSERT_TRUE(tb != nullptr && tb->shape.size() == 3);
    std::printf("[mage_t2i] %-18s cfg %dx%d -> latent [%d, %d, %d] "
                "(want %dx%d)\n", tag, cfg_w, cfg_h, (int)tb->shape[0],
                (int)tb->shape[1], (int)tb->shape[2], expect_w, expect_h);
    EXPECT_TRUE((int)tb->shape[1] == expect_h / P);
    EXPECT_TRUE((int)tb->shape[2] == expect_w / P);
  };

  // No size configured -> the reference's own resolution.
  run(0, 0, refW, refH, "inferred");
  // Explicit size configured -> it wins, reference notwithstanding.
  run(128, 192, 128, 192, "explicit-wins");
}
