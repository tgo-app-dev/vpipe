// The whole Wan image-to-video path, end to end:
//
//   prompt  -> diffusion-conditioner (umT5-XXL)  --.
//   image   -> vae-encode (the conditioning clip) --+-> generate-video
//                                                       -> vae-decode
//                                                       -> RGB frames
//
// Separately gated behind VPIPE_WAN_E2E because of what it costs: the two
// 14B experts are ~53 GB of fp32 on disk EACH, and the run loads both (the
// schedule crosses the boundary, which is the point -- the expert swap is
// exactly the part that has no cheaper test). Even at the smallest legal
// geometry that is minutes of disk, so it must never run as part of the
// default suite.
//
// It is a SMOKE test, deliberately. There is no golden: a full-pipeline
// reference would need the two experts resident in torch as well, and the
// per-component goldens (VAE 0.00076, umT5 0.0038 at one layer, DiT 0.0204
// at full depth, UniPC 9e-8) already pin the arithmetic. What only an
// end-to-end run can show is that the pieces AGREE about shape, layout and
// ownership -- the 36-channel concatenation, the 4-D latent boundary, the
// frame arithmetic, and that one expert is dropped before the next loads.
//
// KNOWN BLOCKER (2026-08-05): this currently stops at the conditioner.
// Wan's text tower is a T5TokenizerFast, whose tokenizer.json carries a
// UNIGRAM model, and Tokenizer::from_huggingface_json supports BPE only --
// so the conditioner goes inert and nothing downstream ever receives a
// prompt. Everything else in the chain is already exercised: the VAE
// encoder and decoder both load and run here, and the DiT is verified to
// full depth by wan_dit. What is missing is a SentencePiece Unigram
// (Viterbi) encode path, which is a tokenizer feature rather than anything
// Wan-specific -- the same gap would block any other T5-conditioned model.
//
// Env: VPIPE_WAN_E2E=1 to enable, VPIPE_WAN_TEST_MODEL_PATH = the model
// root. Skips silently otherwise.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
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
#include "stages/generate-video-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/model-quantize-stage.h"
#include "stages/rgb-to-video-stage.h"
#include "stages/save-video-stage.h"
#include "stages/vae-encode-stage.h"

#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

class SourceText : public TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-wan-prompt";
  using TypedStage::TypedStage;
  std::string prompt;

  Job
  process(RuntimeContext& ctx) override
  {
    auto p = std::make_unique<FlexDataPayload>();
    p->data = FlexData::make_string(prompt);
    co_await ctx.write(0, std::move(p));
    ctx.signal_done();
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "text", .doc = "", .type = &typeid(FlexDataPayload)}};
    static const StageSpec s = {.type_name = "ut-wan-prompt", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

// A deterministic non-flat image: a diagonal ramp, so a decode that
// ignores the conditioning latent is distinguishable from one that does
// not (a flat grey would be indistinguishable from a broken encode).
class SourceImage : public TypedStage<SourceImage> {
public:
  static constexpr const char* kTypeName = "ut-wan-image";
  using TypedStage::TypedStage;
  int h = 256, w = 256;

  Job
  process(RuntimeContext& ctx) override
  {
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::U8;
    b->shape = {3, h, w};
    const std::size_t n = (std::size_t)3 * h * w;
    b->resize_contiguous(n);
    std::uint8_t* p = b->as_u8();
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          p[((std::size_t)c * h + y) * w + x] =
              (std::uint8_t)((x + y + c * 60) & 0xff);
        }
      }
    }
    co_await ctx.write(0, std::move(b));
    ctx.signal_done();
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "image", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-wan-image", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class FrameCapture : public TypedStage<FrameCapture> {
public:
  static constexpr const char* kTypeName = "ut-wan-frames";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> frames;

  Job
  process(RuntimeContext& ctx) override
  {
    auto b = co_await ctx.read(0);
    if (!b) { ctx.signal_done(); co_return; }
    if (const auto* t = dynamic_cast<const TensorBeatPayload*>(b.get())) {
      frames.push_back(*t);
    }
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "image", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-wan-frames", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

}  // namespace

TEST(wan_i2v, end_to_end_smoke)
{
  const char* on   = std::getenv("VPIPE_WAN_E2E");
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (on == nullptr || *on == '\0' || *on == '0') { return; }
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  // The smallest legal geometry: 256 is a multiple of 16, and 5 frames is
  // 4k+1 so the VAE's temporal chunking has a representation for it. That
  // gives T=2 latent frames and 2*16*16 = 512 DiT tokens -- small enough
  // that the run is dominated by loading, which is the point.
  // Overridable so the same graph can produce a LOOKABLE clip rather than
  // a 2-step smoke: VPIPE_WAN_STEPS / _SIZE / _FRAMES, and VPIPE_WAN_MP4 to
  // write one through rgb-to-video -> save-video.
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int H = envi("VPIPE_WAN_SIZE", 256), W = H;
  const int F = envi("VPIPE_WAN_FRAMES", 5);
  const int STEPS = envi("VPIPE_WAN_STEPS", 2);
  const char* mp4 = std::getenv("VPIPE_WAN_MP4");

  auto pl = std::make_unique<Pipeline>("wan", &sess);
  auto txtu = std::make_unique<SourceText>(&sess, "prompt",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  txtu->prompt = "a slow pan across a quiet field";
  txtu->allocate_oports(1);
  auto* txt = static_cast<SourceText*>(pl->insert_stage(std::move(txtu)));

  auto imgu = std::make_unique<SourceImage>(&sess, "image",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  imgu->h = H;
  imgu->w = W;
  imgu->allocate_oports(1);
  auto* img = static_cast<SourceImage*>(pl->insert_stage(std::move(imgu)));

  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{txt, 0}}, std::move(ccfg)));

  // The conditioning clip: the image followed by F-1 blanks, encoded as
  // ONE video. `frames` here must match generate-video's below.
  FlexData ecfg = FlexData::make_object();
  ecfg.as_object().insert("hf_dir", FlexData::make_string(root));
  ecfg.as_object().insert("frames", FlexData::make_int(F));
  auto* venc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
      &sess, "venc", std::vector<InEdge>{{img, 0}}, std::move(ecfg)));

  FlexData gcfg = FlexData::make_object();
  gcfg.as_object().insert("hf_dir", FlexData::make_string(root));
  gcfg.as_object().insert("height", FlexData::make_int(H));
  gcfg.as_object().insert("width", FlexData::make_int(W));
  gcfg.as_object().insert("frames", FlexData::make_int(F));
  gcfg.as_object().insert("steps", FlexData::make_int(STEPS));
  // No negative conditioning is wired, so guidance is forced to 1 and each
  // step is ONE forward rather than two.
  auto gvu = std::make_unique<GenerateVideoStage>(
      &sess, "gen",
      std::vector<InEdge>{{cond, 0}, {}, {}, {}, {}, {venc, 0}},
      std::move(gcfg));
  auto* gv = static_cast<GenerateVideoStage*>(
      pl->insert_stage(std::move(gvu)));
  ASSERT_TRUE(gv->config_error().empty());
  // T = 1 + (F-1)/4: the VAE's temporal stride is 4 with the first frame
  // standing alone. 2 at the default F, and F is overridable.
  EXPECT_TRUE(gv->latent_frames() == 1 + (F - 1) / 4);

  FlexData dcfg = FlexData::make_object();
  dcfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* vdec = pl->insert_stage(std::make_unique<VaeDecodeStage>(
      &sess, "vdec", std::vector<InEdge>{{gv, 0}}, std::move(dcfg)));

  auto capu = std::make_unique<FrameCapture>(&sess, "cap",
                                             std::vector<InEdge>{{vdec, 0}},
                                             FlexData::make_object());
  auto* cap = static_cast<FrameCapture*>(pl->insert_stage(std::move(capu)));
  if (mp4 != nullptr && *mp4 != '\0') {
    auto* r2v = pl->insert_stage(std::make_unique<RgbToVideoStage>(
        &sess, "r2v", std::vector<InEdge>{{vdec, 0}}, FlexData::make_object()));
    FlexData sv = FlexData::make_object();
    sv.as_object().insert("output_url", FlexData::make_string(mp4));
    sv.as_object().insert("enable_audio", FlexData::make_bool(false));
    pl->insert_stage(std::make_unique<SaveVideoStage>(
        &sess, "sv", std::vector<InEdge>{{r2v, 0}}, std::move(sv)));
  }

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[wan_i2v] emitted %zu frames (%llu latents)\n",
              cap->frames.size(),
              (unsigned long long)gv->latents_emitted());
  EXPECT_TRUE(gv->latents_emitted() == 1u);
  // F = 1 + 4*(T-1) -> exactly F frames, not F-1 and not 4T. The frame
  // arithmetic is the one thing a shape mismatch anywhere in the chain
  // would move.
  EXPECT_TRUE(cap->frames.size() == (std::size_t)F);
  if (cap->frames.empty()) { return; }
  for (const auto& f : cap->frames) {
    ASSERT_TRUE(f.shape.size() == 3u);
    EXPECT_TRUE(f.shape[0] == 3);
    EXPECT_TRUE(f.shape[1] == H);
    EXPECT_TRUE(f.shape[2] == W);
    EXPECT_TRUE(f.dtype == TensorBeat::DType::U8);
  }
  // Two steps of a 40-step schedule will not make a good video, so nothing
  // here judges the picture. What it does check is that the pixels are not
  // degenerate: an all-one-value clip is what a NaN latent, a dropped
  // conditioning tensor or an un-run DiT all decode to, and every one of
  // those would otherwise pass a shape-only assertion.
  int distinct = 0;
  {
    const TensorBeat& f = cap->frames.front();
    const std::uint8_t* p = f.as_u8();
    const std::size_t n = (std::size_t)3 * H * W;
    bool seen[256] = {false};
    for (std::size_t i = 0; i < n; ++i) {
      if (!seen[p[i]]) { seen[p[i]] = true; ++distinct; }
    }
  }
  std::printf("[wan_i2v] frame 0 has %d distinct byte values\n", distinct);
  EXPECT_TRUE(distinct > 8);
}

// The conditioning half of the same path: prompt -> conditioner ->
// conditioning tensor, with no DiT at all.
//
// Worth having separately because it is the part that was broken: it
// covers exactly the seam that took the pipeline offline
// -- the Unigram tokenizer feeding the encoder feeding the beat the DiT
// consumes. A tokenizer that loads but segments wrongly still passes a
// shape check, so what is asserted is the CONTRACT the DiT relies on:
// the fixed 512-token window, the encoder width, and the zeroed tail past
// the real tokens (which is what the DiT cross-attends to beyond the
// prompt).
//
// Env: VPIPE_WAN_TEST_MODEL_PATH only -- no VPIPE_WAN_E2E gate.
TEST(wan_i2v, conditioner_only)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  auto pl = std::make_unique<Pipeline>("cond", &sess);
  auto txtu = std::make_unique<SourceText>(&sess, "prompt",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  txtu->prompt = "a slow pan across a quiet field";
  txtu->allocate_oports(1);
  auto* txt = static_cast<SourceText*>(pl->insert_stage(std::move(txtu)));

  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{txt, 0}}, std::move(ccfg)));

  auto capu = std::make_unique<FrameCapture>(&sess, "cap",
                                             std::vector<InEdge>{{cond, 0}},
                                             FlexData::make_object());
  auto* cap = static_cast<FrameCapture*>(pl->insert_stage(std::move(capu)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(cap->frames.size() == 1u);
  const TensorBeat& c = cap->frames.front();
  ASSERT_TRUE(c.shape.size() == 2u);
  std::printf("[wan_i2v] conditioning [%lld, %lld]\n",
              (long long)c.shape[0], (long long)c.shape[1]);
  // Wan's text window is a fixed 512 rows -- the encoder pads to it and
  // the DiT cross-attends to all of them.
  EXPECT_TRUE(c.shape[0] == 512);
  EXPECT_TRUE(c.shape[1] == 4096);
  EXPECT_TRUE(c.dtype == TensorBeat::DType::Bf16);

  // The real rows must not be zero (an encoder that ran) and the tail
  // must be (the contract). Finding the boundary from the data also
  // tells us the tokenizer produced a plausible token count rather than,
  // say, one unk per character.
  const auto* p = c.as_bf16();
  int last_nonzero = -1;
  for (int r = 0; r < 512; ++r) {
    for (int k = 0; k < 4096; ++k) {
      if (p[(std::size_t)r * 4096 + k] != 0) { last_nonzero = r; break; }
    }
  }
  std::printf("[wan_i2v] last non-zero row = %d\n", last_nonzero);
  EXPECT_TRUE(last_nonzero >= 0);
  // "a slow pan across a quiet field" is 7 words; a sane segmentation is
  // a handful of tokens, not one per character and not the whole window.
  EXPECT_TRUE(last_nonzero < 32);
}

// Quantize BOTH A14B experts in one pass. Driven as a test because that is
// where the stage harness already lives; gated on VPIPE_WAN_QUANTIZE with
// an explicit output path, so it never runs by accident.
//
// One pass and not two: the experts are one MODEL, switched partway down
// the sigma schedule, so quantizing half a pair would produce a pipeline
// that silently changes precision mid-generation.
TEST(wan_i2v, quantize_experts)
{
  const char* on   = std::getenv("VPIPE_WAN_QUANTIZE");
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (on == nullptr || *on == '\0' || *on == '0') { return; }
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("src_model", FlexData::make_string(root));
  cfg.as_object().insert("output_name", FlexData::make_string(on));
  // 4 unless asked otherwise. The width is worth varying from outside
  // because the two are not interchangeable: w4 costs real accuracy on
  // this model where w8 is meant to be near-free, and the only way to
  // say which by how much is to build both and measure.
  int bits = 4;
  if (const char* bs = std::getenv("VPIPE_WAN_QUANTIZE_BITS")) {
    if (*bs != '\0') { bits = std::atoi(bs); }
  }
  cfg.as_object().insert("bits", FlexData::make_int(bits));
  cfg.as_object().insert("group_size", FlexData::make_int(64));
  cfg.as_object().insert("target", FlexData::make_string("dit"));
  cfg.as_object().insert("skip_existing", FlexData::make_bool(true));

  auto pl = std::make_unique<Pipeline>("qz", &sess);
  auto qu = std::make_unique<ModelQuantizeStage>(&sess, "mq",
                                                 std::vector<InEdge>{},
                                                 std::move(cfg));
  auto* q = static_cast<ModelQuantizeStage*>(pl->insert_stage(std::move(qu)));
  ASSERT_TRUE(q->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  namespace fs = std::filesystem;
  // Both experts quantized, and the small components carried across, so the
  // output is usable directly as an hf_dir.
  EXPECT_TRUE(fs::exists(fs::path(on) / "transformer" / "config.json"));
  EXPECT_TRUE(fs::exists(fs::path(on) / "transformer_2" / "config.json"));
  EXPECT_TRUE(fs::exists(fs::path(on) / "vae" / "config.json"));
  EXPECT_TRUE(fs::exists(fs::path(on) / "text_encoder"));
  EXPECT_TRUE(fs::exists(fs::path(on) / "tokenizer" / "tokenizer.json"));
}
