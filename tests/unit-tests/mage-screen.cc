// Mage-Flow's MANDATORY content screen (generative-models/mage/mage-screen.h)
// and the refusal path it drives:
//
//   diffusion-conditioner (screens; tags a refused prompt)
//     -> text-to-image (skips the denoise, carries the size)
//       -> vae-decode (paints the blank refusal image)
//
// Three layers, three kinds of test:
//
//  1. VERDICT PARSING + the policy text -- pure logic, always runs. This is
//     where fail-closed lives: everything that is not an explicit
//     {"violates": false} from the classifier must block.
//  2. The REFUSAL PATH through text-to-image and vae-decode, driven by a
//     SYNTHETIC tagged beat -- so it is checked independently of whatever the
//     classifier happens to decide.
//  3. The CLASSIFIER itself against the real Qwen3-VL weights.
//
// 2 and 3 need the model (VPIPE_MAGE_TEST_MODEL_PATH = the Mage-Flow root)
// and skip vacuously without it. They cannot run model-free: a stage whose
// model fails to load reports an error, and the runtime then drains it
// without ever calling process().

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/mage/mage-screen.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/load-image-stage.h"
#include "stages/text-to-image-stage.h"
#include "stages/vae-decode-stage.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

bool
contains_(const std::string& hay, const char* needle)
{
  return hay.find(needle) != std::string::npos;
}

bool
has_category_(const genai::MageScreenVerdict& v, const char* c)
{
  for (const auto& s : v.categories) {
    if (s == c) { return true; }
  }
  return false;
}

// Emits ONE conditioning beat, optionally tagged `content_blocked` -- what the
// conditioner produces for a refused prompt.
class CondSource : public TypedStage<CondSource> {
public:
  static constexpr const char* kTypeName = "ut-mage-screen-cond-src";
  CondSource(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }

  bool blocked = true;
  bool done = false;

  Job process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = TensorBeat::DType::Bf16;
      out->shape = {1, 8};
      out->resize_contiguous(8);
      std::memset(out->as_u8(), 0, 16);
      if (blocked) {
        FlexData sb = FlexData::make_object();
        sb.as_object().insert_or_assign("content_blocked",
                                        FlexData::make_bool(true));
        out->sideband = std::move(sb);
      }
      co_await ctx.write(0, std::move(out));
    }
    ctx.signal_done();
    co_return;
  }
};

// Emits the marker text-to-image produces for a refused prompt: a 1x1x1 f32
// beat carrying only the flag and the intended image size.
class RefusalMarkerSource : public TypedStage<RefusalMarkerSource> {
public:
  static constexpr const char* kTypeName = "ut-mage-screen-marker-src";
  RefusalMarkerSource(const SessionContextIntf* s, std::string id,
                      std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  int  h = 48, w = 64;
  bool done = false;
  Job process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = TensorBeat::DType::F32;
      out->shape = {1, 1, 1};
      out->resize_contiguous(1);
      out->as_f32()[0] = 0.0f;
      FlexData sb = FlexData::make_object();
      auto o = sb.as_object();
      o.insert_or_assign("content_blocked", FlexData::make_bool(true));
      o.insert_or_assign("refusal_height", FlexData::make_int(h));
      o.insert_or_assign("refusal_width",  FlexData::make_int(w));
      out->sideband = std::move(sb);
      co_await ctx.write(0, std::move(out));
    }
    ctx.signal_done();
    co_return;
  }
};

class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-mage-screen-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

class PromptSource : public TypedStage<PromptSource> {
public:
  static constexpr const char* kTypeName = "ut-mage-screen-prompt-src";
  PromptSource(const SessionContextIntf* s, std::string id,
               std::vector<InEdge> ip, FlexData c)
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

}  // namespace

// ---- 1. Verdict parsing (fail-closed) ---------------------------------

// The default verdict BLOCKS. Everything downstream leans on this: every
// error path in mage_screen() just returns the default-constructed verdict,
// so if this flipped to false every one of those paths would silently become
// an allow.
TEST(mage_screen, the_default_verdict_blocks)
{
  const genai::MageScreenVerdict v;
  EXPECT_TRUE(v.violates);
}

TEST(mage_screen, parses_the_reference_verdict_shape)
{
  genai::MageScreenVerdict v;
  ASSERT_TRUE(genai::mage_parse_verdict(
      R"({"violates": true, "categories": ["copyright"], )"
      R"("reason": "Pikachu is a named Pokemon."})", &v));
  EXPECT_TRUE(v.violates);
  EXPECT_TRUE(v.categories.size() == 1);
  EXPECT_TRUE(has_category_(v, "copyright"));
  EXPECT_TRUE(v.reason == "Pikachu is a named Pokemon.");

  genai::MageScreenVerdict ok;
  ASSERT_TRUE(genai::mage_parse_verdict(
      R"({"violates": false, "categories": [], "reason": "Benign."})", &ok));
  EXPECT_FALSE(ok.violates);
  EXPECT_TRUE(ok.categories.empty());
}

// The system prompt forbids markdown, but a classifier that fences its JSON
// anyway has still ANSWERED -- failing closed on it would block benign
// prompts for a formatting slip.
TEST(mage_screen, accepts_a_fenced_verdict)
{
  genai::MageScreenVerdict v;
  ASSERT_TRUE(genai::mage_parse_verdict(
      "```json\n{\"violates\": false, \"categories\": [], "
      "\"reason\": \"Benign food photography.\"}\n```", &v));
  EXPECT_FALSE(v.violates);
  EXPECT_TRUE(v.reason == "Benign food photography.");
}

// Preamble before the object, and a BRACE INSIDE the reason string. A naive
// first-`{`-to-first-`}` scan truncates this into invalid JSON and fails
// closed -- blocking a prompt the classifier cleared.
TEST(mage_screen, a_brace_in_the_reason_does_not_truncate_the_object)
{
  genai::MageScreenVerdict v;
  ASSERT_TRUE(genai::mage_parse_verdict(
      "Sure, here is the verdict:\n"
      R"({"violates": false, "categories": [], )"
      R"("reason": "The prompt has a literal } and \" in it."})", &v));
  EXPECT_FALSE(v.violates);
  EXPECT_TRUE(contains_(v.reason, "literal }"));
}

// Anything that is not a verdict fails to parse -- and mage_screen() turns a
// parse failure into a BLOCK. Note the third case: a well-formed JSON object
// with no "violates" key is NOT read as an implicit allow.
TEST(mage_screen, output_without_a_verdict_does_not_parse)
{
  genai::MageScreenVerdict v;
  EXPECT_FALSE(genai::mage_parse_verdict("", &v));
  EXPECT_FALSE(genai::mage_parse_verdict("I cannot classify that.", &v));
  EXPECT_FALSE(genai::mage_parse_verdict(R"({"categories": []})", &v));
  EXPECT_FALSE(genai::mage_parse_verdict(R"({"violates": false)", &v));
  EXPECT_FALSE(genai::mage_parse_verdict("[1, 2, 3]", &v));
  // A failed parse must not have written anything: the caller's verdict is
  // still the blocking default.
  EXPECT_TRUE(v.violates);
}

// A non-boolean "violates" reads as TRUE, not as a silent false: a classifier
// that answers {"violates": "maybe"} has not cleared anything.
TEST(mage_screen, a_non_boolean_verdict_blocks)
{
  genai::MageScreenVerdict v;
  ASSERT_TRUE(genai::mage_parse_verdict(R"({"violates": "maybe"})", &v));
  EXPECT_TRUE(v.violates);
}

// The policy text IS the gate -- the classifier is a general Qwen3-VL, so
// these clauses are what make it block. Pin the load-bearing ones so a
// well-meaning reflow of that (very long) data file cannot quietly delete a
// category or the anti-rationalization rule.
TEST(mage_screen, the_policy_prompts_carry_the_reference_rules)
{
  const std::string t(genai::kMageFilterSystem);
  const std::string e(genai::kMageFilterEditSystem);
  for (const char* cat : {"sexual", "hate", "self_harm", "violence",
                          "copyright", "public_figure"}) {
    EXPECT_TRUE(contains_(t, cat));
    EXPECT_TRUE(contains_(e, cat));
  }
  // Both demand strict JSON and forbid the "but it's only fiction/art" escape.
  EXPECT_TRUE(contains_(t, "STRICT JSON ONLY"));
  EXPECT_TRUE(contains_(e, "STRICT JSON ONLY"));
  EXPECT_TRUE(contains_(t, "Do NOT rationalize"));
  EXPECT_TRUE(contains_(e, "Do NOT rationalize"));
  // The edit policy's distinguishing rule: RECOGNIZING the source subject is
  // itself the violation, so "just change the background" does not launder it.
  EXPECT_TRUE(contains_(e, "DECISIVE RULE"));
  EXPECT_TRUE(contains_(e, "Naming it = blocking it"));
  EXPECT_TRUE(contains_(e, "SOURCE IMAGES"));
  // ... which is exactly what the text policy does NOT talk about.
  EXPECT_FALSE(contains_(t, "SOURCE IMAGES"));
}


// ---- 2. The refusal path through the stages ---------------------------
// Env: VPIPE_MAGE_TEST_MODEL_PATH = the Mage-Flow model root.

namespace {

const char*
mage_root_()
{
  const char* r = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  return (r != nullptr && *r != '\0') ? r : nullptr;
}

}  // namespace

// vae-decode turns the refusal marker into the blank image, at the size the
// marker carries -- WITHOUT decoding anything. The beat it gets is a 1x1x1
// f32 tensor: there is no latent in it, so a stage that tried to decode would
// fail the z_dim check and drop the beat instead of emitting.
TEST(mage_screen, vae_decode_paints_the_blank_refusal_at_the_carried_size)
{
  const char* root = mage_root_();
  if (root == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  auto pl = std::make_unique<Pipeline>("refuse-vae", &sess);
  auto su = std::make_unique<RefusalMarkerSource>(
      &sess, "marker", std::vector<InEdge>{}, FlexData::make_object());
  auto* src =
      static_cast<RefusalMarkerSource*>(pl->insert_stage(std::move(su)));
  src->h = 48;
  src->w = 64;              // non-square: a transposed h/w would fail

  FlexData vc = FlexData::make_object();
  vc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{src, 0}}, std::move(vc));
  auto* vae = pl->insert_stage(std::move(vu));

  auto ku = std::make_unique<Sink>(&sess, "sink", std::vector<InEdge>{{vae, 0}},
                                   FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(ku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->dtype == TensorBeat::DType::U8);
  ASSERT_TRUE(tb->shape.size() == 3);
  EXPECT_TRUE(tb->shape[0] == 3);
  EXPECT_TRUE(tb->shape[1] == 48);
  EXPECT_TRUE(tb->shape[2] == 64);
  const auto px = tb->materialize_contiguous();
  ASSERT_TRUE(px.size() == (std::size_t)3 * 48 * 64);
  bool all_white = true;
  for (const auto b : px) {
    if (b != 0xff) { all_white = false; break; }
  }
  EXPECT_TRUE(all_white);
}

// text-to-image turns a tagged CONDITIONING beat into that marker without
// denoising. The DiT does load here (the stage would otherwise be drained),
// but it must not RUN: a 1x1x1 marker out is the proof -- a generation would
// have produced a [128, H/16, W/16] latent, and taken orders of magnitude
// longer than this test does.
TEST(mage_screen, text_to_image_refuses_a_tagged_conditioning_without_denoising)
{
  const char* root = mage_root_();
  if (root == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  auto pl = std::make_unique<Pipeline>("refuse-dit", &sess);
  auto su = std::make_unique<CondSource>(&sess, "cond", std::vector<InEdge>{},
                                         FlexData::make_object());
  auto* src = static_cast<CondSource*>(pl->insert_stage(std::move(su)));

  FlexData tc = FlexData::make_object();
  tc.as_object().insert("hf_dir", FlexData::make_string(root));
  tc.as_object().insert("width", FlexData::make_int(64));
  tc.as_object().insert("height", FlexData::make_int(48));
  auto tu = std::make_unique<TextToImageStage>(
      &sess, "t2i", std::vector<InEdge>{{src, 0}}, std::move(tc));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(tu)));

  auto ku = std::make_unique<Sink>(&sess, "sink", std::vector<InEdge>{{t2i, 0}},
                                   FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(ku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->shape.size() == 3);
  EXPECT_TRUE(tb->shape[0] == 1 && tb->shape[1] == 1 && tb->shape[2] == 1);
  ASSERT_TRUE(tb->sideband.is_object());
  FlexData sb = tb->sideband;              // as_object() is a view: keep it
  auto o = sb.as_object();
  ASSERT_TRUE(o.contains("content_blocked"));
  EXPECT_TRUE(o.at("content_blocked").as_bool(false));
  ASSERT_TRUE(o.contains("refusal_height") && o.contains("refusal_width"));
  EXPECT_TRUE(o.at("refusal_height").as_int(0) == 48);
  EXPECT_TRUE(o.at("refusal_width").as_int(0) == 64);
}

// ---- 3. The classifier, on the real weights ---------------------------

namespace {

// prompt -> diffusion-conditioner -> sink. No DiT and no VAE: what these
// tests are about is the VERDICT and the beat it produces, and a 7.7 GB
// transformer contributes nothing to that.
struct ScreenGraph {
  std::unique_ptr<Pipeline>  pl;
  PromptSource*              src  = nullptr;
  DiffusionConditionerStage* cond = nullptr;
  Sink*                      sink = nullptr;
};

ScreenGraph
build_screen_graph_(Session& sess, const char* root)
{
  ScreenGraph g;
  g.pl = std::make_unique<Pipeline>("screen", &sess);

  auto su = std::make_unique<PromptSource>(&sess, "src", std::vector<InEdge>{},
                                           FlexData::make_object());
  g.src = static_cast<PromptSource*>(g.pl->insert_stage(std::move(su)));

  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{g.src, 0}}, std::move(cc));
  g.cond = static_cast<DiffusionConditionerStage*>(
      g.pl->insert_stage(std::move(cu)));

  auto ku = std::make_unique<Sink>(&sess, "sink",
                                   std::vector<InEdge>{{g.cond, 0}},
                                   FlexData::make_object());
  g.sink = static_cast<Sink*>(g.pl->insert_stage(std::move(ku)));
  return g;
}

}  // namespace

// A benign prompt clears the gate and conditions normally: nothing blocked,
// and the beat is real conditioning (many rows, no flag) rather than the
// one-row refusal marker.
TEST(mage_screen, a_benign_prompt_clears_the_gate)
{
  const char* root = mage_root_();
  if (root == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  ScreenGraph g = build_screen_graph_(sess, root);
  g.src->prompt = "A bowl of ramen with steam rising, food photography";

  PipelineRuntime rt(g.pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(g.cond->conditionings_emitted() == 1);
  EXPECT_TRUE(g.cond->blocked_by_policy() == 0);
  ASSERT_TRUE(g.sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(g.sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->shape.size() == 2);
  EXPECT_TRUE(tb->shape[0] > 1);          // real conditioning, not the marker
  bool flagged = false;
  if (tb->sideband.is_object()) {
    FlexData sb = tb->sideband;
    flagged = sb.as_object().contains("content_blocked");
  }
  EXPECT_FALSE(flagged);
}

// A prompt naming a copyrighted character is refused: the conditioner counts
// the block and emits the one-row tagged marker instead of conditioning.
TEST(mage_screen, a_violating_prompt_is_refused)
{
  const char* root = mage_root_();
  if (root == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  ScreenGraph g = build_screen_graph_(sess, root);
  g.src->prompt =
      "Anime style of Eevee, the small brown fox-like Pokemon with a fluffy "
      "cream collar";

  PipelineRuntime rt(g.pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(g.cond->conditionings_emitted() == 1);
  EXPECT_TRUE(g.cond->blocked_by_policy() == 1);
  ASSERT_TRUE(g.sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(g.sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->shape.size() == 2);
  EXPECT_TRUE(tb->shape[0] == 1);
  ASSERT_TRUE(tb->sideband.is_object());
  FlexData sb = tb->sideband;              // as_object() is a view: keep it
  EXPECT_TRUE(sb.as_object().at("content_blocked").as_bool(false));
}

// The EDIT gate is multimodal: the classifier is shown the SOURCE IMAGE as
// well as the instruction, through the same Qwen3-VL tower + deepstack
// features + mROPE positions the conditioning path uses.
//
// This asserts a benign edit CLEARS, and that is deliberate. Asserting that a
// violating edit blocks would prove nothing: garbage in the spliced vision
// rows (wrong mROPE, missing deepstack, an f16/bf16 mixup) makes the
// classifier emit nonsense, which fails to parse, which fails CLOSED -- a
// block. Only a CLEARED verdict shows the image actually arrived intact.
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH + VPIPE_MAGE_EDIT_IMAGE (a benign photo).
TEST(mage_screen, a_benign_edit_clears_the_multimodal_gate)
{
  const char* root = mage_root_();
  if (root == nullptr) { return; }
  const char* imge = std::getenv("VPIPE_MAGE_EDIT_IMAGE");
  if (imge == nullptr || *imge == '\0' || !std::filesystem::exists(imge)) {
    std::printf("[mage_screen] no VPIPE_MAGE_EDIT_IMAGE; skipping\n");
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  auto pl = std::make_unique<Pipeline>("screen-edit", &sess);
  auto su = std::make_unique<PromptSource>(&sess, "src", std::vector<InEdge>{},
                                           FlexData::make_object());
  auto* src = static_cast<PromptSource*>(pl->insert_stage(std::move(su)));
  src->prompt = "change the background to a beach";

  FlexData ic = FlexData::make_object();
  ic.as_object().insert("url", FlexData::make_string(imge));
  auto iu = std::make_unique<LoadImageStage>(&sess, "img",
                                             std::vector<InEdge>{},
                                             std::move(ic));
  auto* img = pl->insert_stage(std::move(iu));

  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond",
      std::vector<InEdge>{{src, 0}, {nullptr, 0}, {nullptr, 0}, {img, 0}},
      std::move(cc));
  auto* cond = static_cast<DiffusionConditionerStage*>(
      pl->insert_stage(std::move(cu)));

  auto ku = std::make_unique<Sink>(&sess, "sink",
                                   std::vector<InEdge>{{cond, 0}},
                                   FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(ku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(cond->conditionings_emitted() == 1);
  EXPECT_TRUE(cond->blocked_by_policy() == 0);
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  ASSERT_TRUE(tb->shape.size() == 2);
  EXPECT_TRUE(tb->shape[0] > 1);          // real conditioning, not the marker
}
