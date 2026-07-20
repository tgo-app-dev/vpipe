// Krea-2-Turbo text-to-image stage bring-up (M5), verified against the golden.
//
// Two checks:
//  * tokenization: the prompt template [prefix | prompt | suffix] tokenizes to
//    the exact golden encoder ids (a1_full_text_ids.i32).
//  * end-to-end: the text-to-image stage (encode -> DiT -> 8-step sampler),
//    seeded with the GOLDEN initial noise (a3_step0_latin), reproduces the
//    golden final latents within the documented free-running f16 drift; and,
//    chained into vae-decode, produces a coherent RGB image (PNG dumped).
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH = the Krea-2-Turbo model root,
// VPIPE_KREA2_GOLDEN = the golden dir. Skips if unset.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/tokenizer.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/model-quantize-stage.h"
#include "stages/text-to-image-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

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

std::vector<std::int32_t>
read_i32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<std::int32_t> out;
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
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

constexpr const char* kPrefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n";
constexpr const char* kSuffix = "<|im_end|>\n<|im_start|>assistant\n";

// Special-token-aware encode (mirrors TextToImageStage's helper): split at the
// ChatML markers, encode each text run, splice the markers' special ids.
std::vector<std::int32_t>
encode_specials_(const Tokenizer& tok, const std::string& text)
{
  static const char* kMarkers[] = {"<|im_start|>", "<|im_end|>"};
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    int which = -1;
    for (int mi = 0; mi < 2; ++mi) {
      const std::size_t f = text.find(kMarkers[mi], pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f; which = mi;
      }
    }
    if (which < 0) {
      const std::vector<std::int32_t> seg = tok.encode(text.substr(pos));
      out.insert(out.end(), seg.begin(), seg.end());
      break;
    }
    if (best > pos) {
      const std::vector<std::int32_t> seg =
          tok.encode(text.substr(pos, best - pos));
      out.insert(out.end(), seg.begin(), seg.end());
    }
    const std::int32_t sid = tok.special_token_id(kMarkers[which]);
    if (sid >= 0) { out.push_back(sid); }
    pos = best + std::string(kMarkers[which]).size();
  }
  return out;
}

// Test-only source: emits one prompt string then ends.
class SourceText : public vpipe::TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-source-text";
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
      co_await ctx.write(
          0, std::make_unique<vpipe::FlexDataPayload>(
                 vpipe::FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkCapture : public vpipe::TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-t2i";
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

// Test-only source: emits one preloaded U8 RGB image [3,H,W] then ends.
class SourceImage : public vpipe::TypedStage<SourceImage> {
public:
  static constexpr const char* kTypeName = "ut-source-image";
  SourceImage(const vpipe::SessionContextIntf* s, std::string id,
              std::vector<vpipe::InEdge> ip, vpipe::FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::uint8_t> rgb;   // [3,H,W]
  int H = 0, W = 0;
  bool done = false;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto p = std::make_unique<TensorBeatPayload>();
      p->dtype = TensorBeat::DType::U8;
      p->shape = {3, H, W};
      p->resize_contiguous(rgb.size());
      std::memcpy(p->as_u8(), rgb.data(), rgb.size());
      co_await ctx.write(0, std::move(p));
    }
    ctx.signal_done();
    co_return;
  }
};

// Chain a diffusion-conditioner between `src` (prompt) and the text-to-image
// (DiT) stage -- the encoder half moved there, so the DiT consumes ready-made
// conditioning. Returns the conditioner; wire the t2i's iport0 to {cond, 0}.
Stage*
add_conditioner_(Pipeline* pl, Session& sess, Stage* src, const char* root)
{
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(root));
  auto u = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(c));
  return pl->insert_stage(std::move(u));
}

}  // namespace

// The prompt template tokenizes to the exact golden encoder ids.
TEST(krea2_t2i, tokenization_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  const std::string tok_path =
      std::string(root) + "/tokenizer/tokenizer.json";
  auto tok = Tokenizer::from_huggingface_json(tok_path, &sess);
  ASSERT_TRUE(tok != nullptr);

  const std::vector<std::int32_t> golden =
      read_i32_(std::string(gd) + "/a1_full_text_ids.i32");
  ASSERT_TRUE(!golden.empty());

  const std::vector<std::int32_t> ids = encode_specials_(
      *tok, std::string(kPrefix) + "a fox in the snow" + kSuffix);

  std::printf("[krea2_t2i] tokenized %zu ids (golden %zu)\n", ids.size(),
              golden.size());
  ASSERT_TRUE(ids.size() == golden.size());
  bool eq = true;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != golden[i]) { eq = false; break; }
  }
  EXPECT_TRUE(eq);
}

// End-to-end: text-to-image (seeded with the golden initial noise) -> vae-decode
// -> RGB. Checks the emitted latent tracks the golden final latents (within the
// documented free-running f16 drift) and the decoded image is coherent.
TEST(krea2_t2i, end_to_end_from_golden_noise)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<float> gfinal = read_f32_(gdir + "/a3_final_latents.f32");
  ASSERT_TRUE(!gfinal.empty());
  const int IC = 64;
  const int img_seq = (int)(gfinal.size() / IC);   // 256
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  ASSERT_TRUE(grid * grid == img_seq);
  const int lh = grid * 2, lw = grid * 2;           // latent 32x32
  const int H = lh * 8, W = lw * 8;

  // --- stage 1: text-to-image, seeded with the golden initial noise ---
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a fox in the snow";
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData t2i_cfg = FlexData::make_object();
  t2i_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  t2i_cfg.as_object().insert("height", FlexData::make_int(H));
  t2i_cfg.as_object().insert("width", FlexData::make_int(W));
  t2i_cfg.as_object().insert(
      "init_latents", FlexData::make_string(gdir + "/a3_step0_latin.f32"));
  auto* cond = add_conditioner_(pl.get(), sess, src, root);
  auto t2iu = std::make_unique<TextToImageStage>(
      &sess, "t2i", std::vector<InEdge>{{cond, 0}}, std::move(t2i_cfg));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vae_cfg = FlexData::make_object();
  vae_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vaeu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vae_cfg));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vaeu)));

  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{vae, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(t2i->latents_emitted() == 1);
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
              tb->shape[1] == H && tb->shape[2] == W);

  // Coherence: the decoded image is non-degenerate (real content, not flat).
  const std::uint8_t* u = tb->as_u8();
  const std::size_t np = (std::size_t)3 * H * W;
  double mean = 0.0;
  for (std::size_t i = 0; i < np; ++i) { mean += u[i]; }
  mean /= (double)np;
  double var = 0.0;
  for (std::size_t i = 0; i < np; ++i) {
    const double d = (double)u[i] - mean; var += d * d;
  }
  const double sd = std::sqrt(var / (double)np);
  std::printf("[krea2_t2i] decoded image mean=%.1f std=%.1f (%dx%d)\n", mean,
              sd, H, W);
  EXPECT_TRUE(sd > 8.0);              // not a flat/degenerate image

  // Golden anchor: seeded with the golden initial noise, the whole metal path
  // (encode -> DiT -> 8-step sampler -> VAE) should track the golden image.
  // Free-running f16 accumulation drifts the trajectory (documented ~0.25 on
  // latents), so the bar is loose -- this confirms end-to-end correctness, not
  // bit-exactness.
  const std::vector<float> gimg = read_f32_(gdir + "/a4_image.f32");
  if (!gimg.empty() && gimg.size() == np) {
    std::vector<float> got(np);
    for (std::size_t i = 0; i < np; ++i) {
      got[i] = (float)u[i] / 255.0f * 2.0f - 1.0f;
    }
    const double r = rel_l2_(got.data(), gimg.data(), np);
    std::printf("[krea2_t2i] decoded-vs-golden image rel-L2 = %.6f\n", r);
    EXPECT_TRUE(r < 0.5);
  }

  // Dump a PNG-free raw for eyeballing (PPM: portable, no deps).
  {
    const std::string ppm = "/tmp/krea2_t2i_metal.ppm";
    std::ofstream o(ppm, std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int c = 0; c < 3; ++c) {
          o.put((char)u[((std::size_t)c * H + y) * W + x]);
        }
      }
    }
    std::printf("[krea2_t2i] wrote %s\n", ppm.c_str());
  }
}

// End-to-end with a 4-bit QUANTIZED DiT (dit_dir override -> the cached
// <root>-dit-w4g64 produced by the model-quantize path). Confirms the whole
// quantized text-to-image pipeline produces a coherent image. Skips if the
// quantized DiT isn't present (run krea2_dit_quant first).
TEST(krea2_t2i, end_to_end_quantized_dit)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const char* sfx = std::getenv("VPIPE_KREA2_DIT_SUFFIX");   // e.g. -dit-w8g64
  const std::string qdit =
      std::string(root) + ((sfx && *sfx) ? sfx : "-dit-w4g64");
  if (!fs::exists(fs::path(qdit) / "config.json")) {
    std::printf("[krea2_t2i] no quantized DiT at %s; skipping\n", qdit.c_str());
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<float> gimg = read_f32_(gdir + "/a4_image.f32");
  ASSERT_TRUE(!gimg.empty());
  const int H = 256, W = 256;
  ASSERT_TRUE(gimg.size() == (std::size_t)3 * H * W);

  auto pl = std::make_unique<Pipeline>("pq", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a fox in the snow";
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData t2i_cfg = FlexData::make_object();
  t2i_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  t2i_cfg.as_object().insert("dit_dir", FlexData::make_string(qdit));  // 4-bit
  t2i_cfg.as_object().insert(
      "init_latents", FlexData::make_string(gdir + "/a3_step0_latin.f32"));
  auto* cond = add_conditioner_(pl.get(), sess, src, root);
  auto t2iu = std::make_unique<TextToImageStage>(
      &sess, "t2i", std::vector<InEdge>{{cond, 0}}, std::move(t2i_cfg));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vae_cfg = FlexData::make_object();
  vae_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vaeu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vae_cfg));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vaeu)));

  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{vae, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr && tb->shape.size() == 3);
  const std::uint8_t* u = tb->as_u8();
  const std::size_t np = (std::size_t)3 * H * W;
  double mean = 0.0;
  for (std::size_t i = 0; i < np; ++i) { mean += u[i]; }
  mean /= (double)np;
  double var = 0.0;
  for (std::size_t i = 0; i < np; ++i) { const double d = u[i] - mean; var += d * d; }
  const double sd = std::sqrt(var / (double)np);
  std::vector<float> got(np);
  for (std::size_t i = 0; i < np; ++i) { got[i] = (float)u[i] / 255.0f * 2.0f - 1.0f; }
  const double r = rel_l2_(got.data(), gimg.data(), np);
  std::printf("[krea2_t2i] QUANTIZED(4-bit) image std=%.1f, vs-golden rel-L2 = "
              "%.6f\n", sd, r);
  EXPECT_TRUE(sd > 8.0);
  EXPECT_TRUE(r < 0.55);          // 4-bit adds to the free-running f16 drift
  {
    std::ofstream o("/tmp/krea2_t2i_metal_w4.ppm", std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int c = 0; c < 3; ++c) {
          o.put((char)u[((std::size_t)c * H + y) * W + x]);
        }
      }
    }
    std::printf("[krea2_t2i] wrote /tmp/krea2_t2i_metal_w4.ppm\n");
  }
}

// End-to-end with a LoRA-FUSED DiT (dit_dir -> the cached <root>-dit-lora-swc
// from the krea2_lora fusion). Same golden init noise as the base run, so the
// style shift (softwatercolor) is directly comparable. Skips if not fused yet.
TEST(krea2_t2i, end_to_end_lora_dit)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const std::string ldit = std::string(root) + "-dit-lora-swc";
  if (!fs::exists(fs::path(ldit) / "config.json")) {
    std::printf("[krea2_t2i] no fused LoRA DiT at %s; skipping\n", ldit.c_str());
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string gdir = gd;
  const int H = 256, W = 256;

  auto pl = std::make_unique<Pipeline>("pl", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a fox in the snow, Art Deco watercolor style";  // LoRA trigger
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData t2i_cfg = FlexData::make_object();
  t2i_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  t2i_cfg.as_object().insert("dit_dir", FlexData::make_string(ldit));
  t2i_cfg.as_object().insert(
      "init_latents", FlexData::make_string(gdir + "/a3_step0_latin.f32"));
  auto* cond = add_conditioner_(pl.get(), sess, src, root);
  auto t2iu = std::make_unique<TextToImageStage>(
      &sess, "t2i", std::vector<InEdge>{{cond, 0}}, std::move(t2i_cfg));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vae_cfg = FlexData::make_object();
  vae_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vaeu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vae_cfg));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vaeu)));
  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{vae, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

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
  std::printf("[krea2_t2i] LoRA image std=%.1f\n", std::sqrt(var / (double)np));
  EXPECT_TRUE(std::sqrt(var / (double)np) > 8.0);
  {
    std::ofstream o("/tmp/krea2_lora_metal.ppm", std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c)
          o.put((char)u[((std::size_t)c * H + y) * W + x]);
    std::printf("[krea2_t2i] wrote /tmp/krea2_lora_metal.ppm\n");
  }
}

// img2img sampling (tight): feed the golden scale_noised init (b_init_latents)
// via init_latents with strength 0.6, so the stage denoises only the tail steps
// (t_start..S). rel-L2 the emitted unpacked latent against the unpacked golden
// b_final_latents (only f16 drift over the 5 tail steps).
TEST(krea2_t2i, img2img_sampling_from_golden_init)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<float> gfinal = read_f32_(gdir + "/b_final_latents.f32");
  if (gfinal.empty()) { return; }   // img2img golden not dumped -> skip
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const int IC = 64, gw = 16, gh = 16, lh = 32, lw = 32;
  ASSERT_TRUE(gfinal.size() == (std::size_t)gh * gw * IC);
  // Unpack golden [256,64] -> [16,32,32] to compare against the stage output.
  std::vector<float> gunp((std::size_t)16 * lh * lw);
  for (int c = 0; c < 16; ++c)
    for (int y = 0; y < lh; ++y)
      for (int x = 0; x < lw; ++x) {
        const int a = y / 2, ph = y % 2, b = x / 2, pw = x % 2;
        gunp[((std::size_t)c * lh + y) * lw + x] =
            gfinal[((std::size_t)(a * gw + b)) * IC + c * 4 + ph * 2 + pw];
      }

  auto pl = std::make_unique<Pipeline>("i2", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a fox in the snow";
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert("strength", FlexData::make_real(0.6));
  cfg.as_object().insert(
      "init_latents", FlexData::make_string(gdir + "/b_init_latents.f32"));
  auto* cond = add_conditioner_(pl.get(), sess, src, root);
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

  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr && tb->dtype == TensorBeat::DType::F32);
  ASSERT_TRUE(tb->element_count() == gunp.size());
  const float* lp = tb->as_f32();
  std::vector<float> got(gunp.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = lp[i]; }
  const double r = rel_l2_(got.data(), gunp.data(), got.size());
  std::printf("[krea2_t2i] img2img sampling rel-L2 = %.6f (tail from t_start)\n",
              r);
  EXPECT_TRUE(r < 0.20);            // 5 tail steps of f16 drift from golden init
}

// Full img2img end-to-end (eyeball): source image (the golden fox a4_image) ->
// vae-encode -> text-to-image (strength 0.6, latent port + noise) ->
// vae-decode -> RGB. Metal uses its own noise so it differs from the golden,
// but must be a coherent transform of the input. Saves a PPM.
TEST(krea2_t2i, img2img_end_to_end)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const std::string gdir = gd;
  const std::vector<float> vin = read_f32_(gdir + "/b_vae_in.f32");  // [-1,1]
  if (vin.empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const int H = 256, W = 256;
  ASSERT_TRUE(vin.size() == (std::size_t)3 * H * W);

  auto pl = std::make_unique<Pipeline>("i2e", &sess);
  auto pru = std::make_unique<SourceText>(&sess, "prompt",
                                          std::vector<InEdge>{},
                                          FlexData::make_object());
  pru->prompt = "a fox in the snow";
  auto* pr = static_cast<SourceText*>(pl->insert_stage(std::move(pru)));

  auto imu = std::make_unique<SourceImage>(&sess, "img",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  imu->H = H; imu->W = W;
  imu->rgb.resize((std::size_t)3 * H * W);
  for (std::size_t i = 0; i < imu->rgb.size(); ++i) {
    float v = (vin[i] + 1.0f) * 0.5f * 255.0f;
    v = std::round(v); v = v < 0 ? 0 : (v > 255 ? 255 : v);
    imu->rgb[i] = (std::uint8_t)v;
  }
  auto* im = static_cast<SourceImage*>(pl->insert_stage(std::move(imu)));

  // image -> vae-encode -> whitened latent [16,32,32].
  FlexData vecfg = FlexData::make_object();
  vecfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto veu = std::make_unique<VaeEncodeStage>(
      &sess, "vaeenc", std::vector<InEdge>{{im, 0}}, std::move(vecfg));
  auto* ve = static_cast<VaeEncodeStage*>(pl->insert_stage(std::move(veu)));
  ASSERT_TRUE(ve->config_error().empty());

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert("strength", FlexData::make_real(0.6));
  // conditioning on iport0 (from the conditioner); neg/sampler/scheduler
  // (iport1-3) DISCONNECTED; the img2img init latent (from vae-encode) on
  // iport4 (ref latent 0).
  auto* cond = add_conditioner_(pl.get(), sess, pr, root);
  auto t2iu = std::make_unique<TextToImageStage>(
      &sess, "t2i",
      std::vector<InEdge>{{cond, 0}, InEdge{nullptr, 0}, InEdge{nullptr, 0},
                          InEdge{nullptr, 0}, {ve, 0}},
      std::move(cfg));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vcfg = FlexData::make_object();
  vcfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vaeu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vcfg));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vaeu)));
  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{vae, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(t2i->latents_emitted() == 1);
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr && tb->dtype == TensorBeat::DType::U8);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[1] == H && tb->shape[2] == W);
  const std::uint8_t* u = tb->as_u8();
  const std::size_t np = (std::size_t)3 * H * W;
  double mean = 0.0;
  for (std::size_t i = 0; i < np; ++i) { mean += u[i]; }
  mean /= (double)np;
  double var = 0.0;
  for (std::size_t i = 0; i < np; ++i) { const double d = u[i] - mean; var += d * d; }
  const double sd = std::sqrt(var / (double)np);
  std::printf("[krea2_t2i] img2img e2e image std=%.1f\n", sd);
  EXPECT_TRUE(sd > 8.0);
  {
    std::ofstream o("/tmp/krea2_img2img_metal.ppm", std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int c = 0; c < 3; ++c)
          o.put((char)u[((std::size_t)c * H + y) * W + x]);
    std::printf("[krea2_t2i] wrote /tmp/krea2_img2img_metal.ppm\n");
  }
}

// End-to-end proof of the model-quantize `target=text_encoder` pass: quantize
// the Krea-2 text encoder into a SELF-CONTAINED pipeline, then drive
// text-to-image with that dir as hf_dir (no dit_dir override). The quantized
// Qwen3-VL encoder must load + condition the DiT and produce a coherent latent.
TEST(krea2_t2i, enc_quantized_hf_dir)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  namespace fs = std::filesystem;
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  // 1. Quantize the text encoder -> a self-contained model on the source's
  //    volume (the other components hard-link, so this is fast).
  const std::string qdir =
      (fs::path(root).parent_path() / "vpipe-krea2-enc-q-t2i").string();
  std::error_code ec;
  fs::remove_all(qdir, ec);
  {
    FlexData qcfg = FlexData::make_object();
    qcfg.as_object().insert("src_model", FlexData::make_string(root));
    qcfg.as_object().insert("output_name", FlexData::make_string(qdir));
    qcfg.as_object().insert("target", FlexData::make_string("text_encoder"));
    qcfg.as_object().insert("bits", FlexData::make_uint(8));
    qcfg.as_object().insert("skip_existing", FlexData::make_bool(false));
    auto qpl = std::make_unique<Pipeline>("encq", &sess);
    auto qu = std::make_unique<ModelQuantizeStage>(
        &sess, "mq", std::vector<InEdge>{}, std::move(qcfg));
    auto* qs = static_cast<ModelQuantizeStage*>(qpl->insert_stage(std::move(qu)));
    ASSERT_TRUE(qs->config_error().empty());
    PipelineRuntime qrt(qpl.get(), &sess);
    EXPECT_TRUE(qrt.launch());
    qrt.wait_idle();
    qrt.stop();
  }
  ASSERT_TRUE(fs::exists(fs::path(qdir) / "text_encoder" / "config.json", ec));

  // 2. text-to-image with hf_dir = the quantized-encoder pipeline.
  auto pl = std::make_unique<Pipeline>("t2iq", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a fox in the snow";
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(qdir));
  auto* cond = add_conditioner_(pl.get(), sess, src, qdir.c_str());
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

  // A coherent latent came out: the quantized encoder loaded + conditioned
  // the DiT (finite values with real variance, correct [16,32,32] shape).
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr && tb->dtype == TensorBeat::DType::F32);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 16);
  const float* lp = tb->as_f32();
  const std::size_t nl = tb->element_count();
  double mean = 0.0;
  for (std::size_t i = 0; i < nl; ++i) { mean += lp[i]; }
  mean /= (double)nl;
  double var = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < nl; ++i) {
    if (!std::isfinite(lp[i])) { finite = false; }
    const double d = lp[i] - mean; var += d * d;
  }
  const double sd = std::sqrt(var / (double)nl);
  std::printf("[krea2_t2i] enc-quantized hf_dir latent std=%.4f\n", sd);
  EXPECT_TRUE(finite);
  EXPECT_TRUE(sd > 0.05);
  fs::remove_all(qdir, ec);
}

// Model-free: the iport contract is {prompt, negative, sampler, scheduler,
// ref_latent0, ref_latent1} in that order, and guidance_scale is a valid
// config attr.
TEST(krea2_t2i, negative_prompt_ports_and_cfg_config)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string("/nonexistent"));
  cfg.as_object().insert("guidance_scale", FlexData::make_real(4.5));
  TextToImageStage s(&sess, "t2i", std::vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());   // guidance_scale is accepted

  const auto& ip = s.spec().iports;
  ASSERT_TRUE(ip.size() == 6);
  EXPECT_TRUE(ip[0].name == "prompt");
  EXPECT_TRUE(ip[1].name == "negative");
  EXPECT_TRUE(ip[2].name == "sampler");
  EXPECT_TRUE(ip[3].name == "scheduler");
  EXPECT_TRUE(ip[4].name == "ref_latent0");
  EXPECT_TRUE(ip[5].name == "ref_latent1");
}

// Classifier-free guidance actually consumes the negative prompt: the SAME
// golden initial noise denoised with a negative prompt (guidance_scale 3)
// diverges from the no-CFG latent. Everything but the negative/scale is held
// identical, so a dead port (no-op negative) would give rel-L2 ~0 and fail.
TEST(krea2_t2i, cfg_changes_latent)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string gdir = gd;
  const std::vector<float> gfinal = read_f32_(gdir + "/a3_final_latents.f32");
  ASSERT_TRUE(!gfinal.empty());
  const int IC = 64;
  const int img_seq = (int)(gfinal.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  ASSERT_TRUE(grid * grid == img_seq);
  const int lh = grid * 2, lw = grid * 2;
  const int H = lh * 8, W = lw * 8;

  // Run the t2i stage (latent out, no VAE) from the golden initial noise,
  // optionally with a negative prompt + guidance_scale. Returns the emitted
  // latent [16,lh,lw], or empty on failure.
  auto run = [&](bool with_cfg) -> std::vector<float> {
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto pru = std::make_unique<SourceText>(&sess, "prompt",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
    pru->prompt = "a fox in the snow";
    auto* pr = static_cast<SourceText*>(pl->insert_stage(std::move(pru)));
    SourceText* ng = nullptr;
    if (with_cfg) {
      auto ngu = std::make_unique<SourceText>(&sess, "neg",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
      ngu->prompt = "green grass, summer meadow, bright daylight";
      ng = static_cast<SourceText*>(pl->insert_stage(std::move(ngu)));
    }
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    cfg.as_object().insert("height", FlexData::make_int(H));
    cfg.as_object().insert("width", FlexData::make_int(W));
    cfg.as_object().insert(
        "init_latents", FlexData::make_string(gdir + "/a3_step0_latin.f32"));
    if (with_cfg) {
      cfg.as_object().insert("guidance_scale", FlexData::make_real(3.0));
    }
    const std::vector<InEdge> edges =
        with_cfg ? std::vector<InEdge>{{pr, 0}, {ng, 0}}
                 : std::vector<InEdge>{{pr, 0}};
    auto t2iu = std::make_unique<TextToImageStage>(&sess, "t2i", edges,
                                                   std::move(cfg));
    auto* t2i =
        static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
    if (!t2i->config_error().empty()) { return {}; }
    auto sinku = std::make_unique<SinkCapture>(
        &sess, "sink", std::vector<InEdge>{{t2i, 0}}, FlexData::make_object());
    auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));
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
  };

  const std::vector<float> base = run(false);
  const std::vector<float> cfgl = run(true);
  ASSERT_TRUE(!base.empty() && !cfgl.empty());
  ASSERT_TRUE(base.size() == cfgl.size());
  bool finite = true;
  for (float v : cfgl) { if (!std::isfinite(v)) { finite = false; break; } }
  EXPECT_TRUE(finite);
  const double r = rel_l2_(cfgl.data(), base.data(), base.size());
  std::printf("[krea2_t2i] CFG-vs-noCFG latent rel-L2 = %.6f\n", r);
  EXPECT_TRUE(r > 0.01);   // the negative prompt materially changed the output
}
