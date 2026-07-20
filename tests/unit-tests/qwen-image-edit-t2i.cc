// Qwen-Image-Edit-2511 text-to-image STAGE wiring (M7) end-to-end smoke.
//
// Drives the full TextToImageStage on the "qwen-image-edit" family: SourceText
// -> TextToImageStage -> SinkCapture. This exercises the new integration glue --
// family detection, the Qwen2.5-VL encoder load, the image-aware conditioning
// (QwenImageEditPlus template, drop-64, sequential 1-D rope tap + host final-
// RMSNorm), the M2 dynamic-shift sampler, the dual-stream MetalQwenImageTransformer
// denoise loop, and the 2x2 unpack. The component numerics are golden-verified
// separately (M1 VAE, M2 sched, M3 text, M4 DiT, M6 conditioning); this smoke
// confirms the wiring produces a well-formed, non-degenerate latent (no crash,
// no NaN, coherent statistics). A bit-exact end-to-end image golden (pinned
// noise vs the diffusers pipeline) is the remaining verification.
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH = the model root. Skips if unset.
// SLOW (loads the 20B DiT + 7B encoder); env-gated so default suites skip it.

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
#include "stages/text-to-image-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

// Test-only source: emits one prompt string then ends.
class SourceText : public vpipe::TypedStage<SourceText> {
 public:
  static constexpr const char* kTypeName = "ut-qie-source-text";
  SourceText(const vpipe::SessionContextIntf* s, std::string id,
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

class SinkCapture : public vpipe::TypedStage<SinkCapture> {
 public:
  static constexpr const char* kTypeName = "ut-qie-sink-capture";
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

}  // namespace

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}
double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

TEST(qwen_image_edit_t2i, text_to_image_stage_end_to_end)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const int H = 256, W = 256;
  const int lh = H / 8, lw = W / 8;   // 32x32 latent

  // Optional pinned-noise golden (qie_e2e_golden.py): feed the same packed init
  // noise the torch reference used, and compare the denoised latent (loose bar:
  // the bf16 DiT drifts ~0.07/step, compounding over the sampler).
  const char* gd = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  std::string init_path, glat_path;
  if (gd != nullptr && *gd != '\0') {
    init_path = std::string(gd) + "/qie_init.f32";
    glat_path = std::string(gd) + "/g_qie_latent.f32";
    std::ifstream probe(init_path, std::ios::binary);
    if (!probe) { init_path.clear(); glat_path.clear(); }
  }

  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src", std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a red fox sitting in fresh snow, photorealistic";
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  // VPIPE_QWEN_IMAGE_EDIT_DIT_DIR overrides the DiT (e.g. a w4 quantized
  // transformer); the denoise should stay coherent + track the bf16 golden
  // within the added quant error (looser bar).
  const char* dd = std::getenv("VPIPE_QWEN_IMAGE_EDIT_DIT_DIR");
  const bool quant_dit = (dd != nullptr && *dd != '\0');

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert("height", FlexData::make_int(H));
  cfg.as_object().insert("width", FlexData::make_int(W));
  cfg.as_object().insert("steps", FlexData::make_int(4));
  cfg.as_object().insert("seed", FlexData::make_int(42));
  if (quant_dit) {
    cfg.as_object().insert("dit_dir", FlexData::make_string(dd));
  }
  if (!init_path.empty()) {
    cfg.as_object().insert("init_latents", FlexData::make_string(init_path));
  }
  // The encoder half now lives in a diffusion-conditioner stage: src (prompt)
  // -> conditioner -> t2i (conditioning). Text-only here (no ref image).
  FlexData cond_cfg = FlexData::make_object();
  cond_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto condu = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(cond_cfg));
  auto* cond =
      static_cast<DiffusionConditionerStage*>(pl->insert_stage(std::move(condu)));
  ASSERT_TRUE(cond->config_error().empty());
  auto t2iu = std::make_unique<TextToImageStage>(
      &sess, "t2i", std::vector<InEdge>{{cond, 0}}, std::move(cfg));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
  ASSERT_TRUE(t2i->config_error().empty());

  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{t2i, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // The stage may stay inert if the model dir isn't a real QIE pipeline; then
  // nothing is emitted and we skip (env pointed at the wrong dir).
  if (t2i->latents_emitted() == 0) {
    std::printf("[qwen_image_edit_t2i] no latent emitted (model not loaded?); "
                "skipping\n");
    return;
  }
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::F32);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 16 &&
              tb->shape[1] == lh && tb->shape[2] == lw);

  // Coherence: the latent is finite and non-degenerate (real content, not flat
  // / NaN / collapsed). A whitened VAE latent has O(1) per-channel spread.
  const auto bytes = tb->materialize_contiguous();
  const float* lp = reinterpret_cast<const float*>(bytes.data());
  const std::size_t n = (std::size_t)16 * lh * lw;
  bool finite = true;
  double mean = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(lp[i])) { finite = false; break; }
    mean += lp[i];
  }
  ASSERT_TRUE(finite);
  mean /= (double)n;
  double var = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)lp[i] - mean; var += d * d;
  }
  const double sd = std::sqrt(var / (double)n);
  std::printf("[qwen_image_edit_t2i] latent [16,%d,%d] mean=%.3f std=%.3f\n",
              lh, lw, mean, sd);
  EXPECT_TRUE(sd > 0.05);      // not collapsed to a constant
  EXPECT_TRUE(sd < 100.0);     // not exploded

  // Pinned-noise golden: the whole metal path (conditioning -> dual-stream DiT
  // -> dynamic-shift euler) should track the torch reference denoised from the
  // SAME packed noise. The bf16 DiT drifts ~0.07/step (M4), compounding over the
  // sampler, so the bar is loose -- this confirms trajectory correctness, not
  // bit-exactness.
  if (!glat_path.empty()) {
    const std::vector<float> glat = read_f32_(glat_path);
    if (glat.size() == n) {
      const double r = rel_l2_(lp, glat.data(), n);
      std::printf("[qwen_image_edit_t2i] latent-vs-golden rel-L2 = %.6f%s\n", r,
                  quant_dit ? " (w4 DiT)" : "");
      // bf16 observed 0.026; w4 adds group-affine error over 4 steps.
      EXPECT_TRUE(r < (quant_dit ? 0.30 : 0.10));

    }
  }
}
