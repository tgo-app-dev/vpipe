// Live latent previews: the shared spec, the config source that carries
// it, generate-video's port contract, the x0 a preview decodes, and the
// renderer end to end through a real pipeline.
//
// The last needs a TAE checkpoint and a latent that decodes to something
// checkable; it reads both from the taeh3 golden of
// tools/dump_tae_golden.py (VPIPE_TAE_GOLDEN) and skips without it.
#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/session.h"
#include "generative-models/gen-input.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/video-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-spec.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/preview-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/flux2-model-config-stage.h"
#include "stages/krea2-model-config-stage.h"
#include "stages/qwen-image-21-model-config-stage.h"
#include "stages/z-image-model-config-stage.h"
#include "stages/latent-preview.h"
#include "stages/minimax-h3-model-config-stage.h"
#include "stages/model-catalog.h"
#include "stages/model-config-source.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace vpipe;

TEST(latent_preview, the_spec_reads_back_what_a_source_emits)
{
  LatentPreviewSpec want;
  want.vae = "/models/taeh3.safetensors";
  want.every = 3;
  want.max_edge = 640;
  want.max_frames = 17;
  FlexData cfg = FlexData::make_object();
  want.emit(cfg);
  const LatentPreviewSpec got = LatentPreviewSpec::from_model_config(cfg);
  EXPECT_TRUE(got == want);
  EXPECT_TRUE(got.enabled());

  // Nothing named, nothing emitted: a graph that asked for no preview
  // says nothing, and reads back as off.
  FlexData quiet = FlexData::make_object();
  LatentPreviewSpec{}.emit(quiet);
  EXPECT_TRUE(quiet.as_object().empty());
  EXPECT_FALSE(LatentPreviewSpec::from_model_config(quiet).enabled());
  EXPECT_FALSE(LatentPreviewSpec::from_model_config(FlexData{}).enabled());
}

// Every Nth step, and ALWAYS the last: the final clean estimate is the
// one a viewer waits on while the real decode runs.
TEST(latent_preview, the_cadence_always_includes_the_last_step)
{
  LatentPreviewSpec s;
  s.vae = "x";
  s.every = 3;
  std::vector<int> due;
  for (int i = 1; i <= 10; ++i) {
    if (s.due(i, 10)) { due.push_back(i); }
  }
  EXPECT_TRUE((due == std::vector<int>{3, 6, 9, 10}));
  s.every = 1;
  EXPECT_TRUE(s.due(1, 10) && s.due(2, 10));
  s.every = 0;
  EXPECT_FALSE(s.due(10, 10));
  LatentPreviewSpec off;
  off.every = 1;
  EXPECT_FALSE(off.due(1, 1));
}

TEST(latent_preview, the_shrink_is_the_smallest_whole_factor_that_fits)
{
  EXPECT_TRUE(latent_preview_shrink(960, 576, 512) == 2);
  EXPECT_TRUE(latent_preview_shrink(1344, 768, 512) == 3);
  EXPECT_TRUE(latent_preview_shrink(480, 288, 512) == 1);
  EXPECT_TRUE(latent_preview_shrink(1920, 1080, 0) == 1);
  EXPECT_TRUE(latent_preview_shrink(1024, 1024, 512) == 2);
}

// The H3 source carries the four keys and stamps them only when a
// preview VAE is named.
TEST(latent_preview, the_h3_config_source_carries_the_keys)
{
  Session sess;
  {
    MiniMaxH3ModelConfigStage s(&sess, "h3", {}, FlexData::make_object());
    const FlexData fd = s.resolved_config();
    EXPECT_FALSE(fd.as_object().contains(latent_preview::kVaeKey));
    EXPECT_FALSE(fd.as_object().contains(latent_preview::kEveryKey));
  }
  {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert_or_assign(latent_preview::kVaeKey,
                       FlexData::make_string("madebyollin/taeh3"));
    o.insert_or_assign(latent_preview::kEveryKey, FlexData::make_int(2));
    o.insert_or_assign(latent_preview::kFramesKey, FlexData::make_int(22));
    MiniMaxH3ModelConfigStage s(&sess, "h3", {}, cfg);
    EXPECT_TRUE(s.config_error().empty());
    const LatentPreviewSpec p =
        LatentPreviewSpec::from_model_config(s.resolved_config());
    EXPECT_TRUE(p.vae == "madebyollin/taeh3");
    EXPECT_TRUE(p.every == 2);
    EXPECT_TRUE(p.max_edge == 512);   // the default rides along
    EXPECT_TRUE(p.max_frames == 22);
  }
  {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign(latent_preview::kEveryKey,
                                     FlexData::make_int(-1));
    MiniMaxH3ModelConfigStage s(&sess, "h3", {}, cfg);
    EXPECT_FALSE(s.config_error().empty());
  }
}

// The port contract: appended at index 2 (every earlier index keeps its
// meaning), on a clock of its own, tagged so a preview stage takes it,
// and PUSH-STYLE -- DropOldest, so nothing downstream can hold the
// generation up.
TEST(latent_preview, generate_video_exposes_a_push_style_preview_port)
{
  Session sess;
  GenerateVideoStage gv(&sess, "gv", {}, FlexData::make_object());
  const auto& spec = gv.spec();
  ASSERT_TRUE(spec.oports.size() == 3);
  if (spec.oports.size() != 3) { return; }
  EXPECT_TRUE(spec.oports[2].name == "preview");
  EXPECT_TRUE(spec.oports[2].clock_group != spec.oports[0].clock_group);
  PreviewStage pv(&sess, "pv", {}, FlexData::make_object());
  EXPECT_TRUE(port_tags_compatible(spec.oports[2].tags,
                                   pv.spec().iports[0].tags));
  const OportPolicy& pol = gv.oport_policy(2);
  EXPECT_TRUE(pol.mode == OverrunPolicy::DropOldest);
  EXPECT_TRUE(pol.capacity >= 1 && pol.capacity <= 4);
  // The latent ports are untouched: they must still backpressure.
  EXPECT_TRUE(gv.oport_policy(0).mode == OverrunPolicy::Backpressure);
}

// The x0 a preview decodes is the one the sampler steps towards: on the
// last step (sigma_next = 0) the Euler step lands exactly on it.
TEST(latent_preview, h3_x0_is_what_the_last_step_lands_on)
{
  genai::MiniMaxH3Scheduler s(12.0);
  ASSERT_TRUE(s.set_timesteps(8));
  const int last = s.num_inference_steps() - 1;
  std::vector<float> x(64), v(64), x0(64);
  for (int i = 0; i < 64; ++i) {
    x[i] = std::sin(0.3f * i);
    v[i] = std::cos(0.7f * i);
  }
  ASSERT_TRUE(s.estimate_x0(v.data(), last, x.data(), x0.data(), 64));
  std::vector<float> stepped = x;
  ASSERT_TRUE(s.step(v.data(), last, stepped.data(), 64));
  double worst = 0.0;
  for (int i = 0; i < 64; ++i) {
    worst = std::max(worst, (double)std::fabs(stepped[i] - x0[i]));
  }
  EXPECT_TRUE(worst < 1e-6);
  // And mid-schedule it is NOT the state: a preview of x would be noise.
  ASSERT_TRUE(s.estimate_x0(v.data(), 0, x.data(), x0.data(), 64));
  double moved = 0.0;
  for (int i = 0; i < 64; ++i) {
    moved = std::max(moved, (double)std::fabs(x0[i] - x[i]));
  }
  EXPECT_TRUE(moved > 0.1);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

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

// Stands in for a denoiser: owns a LatentPreviewer on oport 0 (DropOldest,
// as generate-video's is) and submits `steps` latents back to back --
// faster than they can render, so the mailbox has to replace.
class PreviewProducer : public TypedStage<PreviewProducer> {
public:
  static constexpr const char* kTypeName = "ut-latent-preview-producer";
  using TypedStage::TypedStage;

  LatentPreviewSpec spec_;
  std::vector<float> latent;
  int C = 0, T = 0, h = 0, w = 0, steps = 3;
  // Generations run through ONE previewer, the way a continuous graph
  // drives generate-video: the TAE is loaded once and reused.
  int generations = 1;
  // What begin() is told: the clip's rate and length (0 / 0 for an image,
  // which makes the beat [3, H, W]) and the family's temporal rule.
  double fps = 24.0;
  int video_frames = 22;
  genai::MetalTaeDecoder::Trim trim = genai::MetalTaeDecoder::Trim::kH3Chunks;
  bool began = false;
  int began_count = 0;
  // The stage-wide policy an IMAGE denoiser sets: drop the TAE when a
  // generation's Scope is left. Exercised through a Scope, because that
  // is where the flag is read and where a stage's previews live.
  bool release_each = false;
  std::size_t resident = 0;
  std::size_t resident_mid = 0;
  std::uint64_t rendered = 0, replaced = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    LatentPreviewer pv(session(), "producer");
    pv.configure(spec_, trim);
    pv.release_each_generation(release_each);
    for (int g = 0; g < generations; ++g) {
      {
        LatentPreviewer::Scope sc(&pv, ctx, 0, fps, video_frames);
        began = sc.active();
        if (!began) { break; }
        ++began_count;
        for (int s = 1; s <= steps; ++s) {
          if (pv.due(s, steps)) { pv.submit(latent, C, T, h, w, s, steps); }
        }
        // Inside the scope the decoder is necessarily still there; the
        // question this pins is what is left once it closes.
        resident_mid = pv.resident_bytes();
      }
      rendered += pv.rendered();
      replaced += pv.replaced();
    }
    resident = pv.resident_bytes();
    ctx.signal_done();
    co_return;
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "preview", .doc = "", .type = &typeid(TensorBeatPayload),
       .tags = "rgb-clip"}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class ClipSink : public TypedStage<ClipSink> {
public:
  static constexpr const char* kTypeName = "ut-latent-preview-sink";
  using TypedStage::TypedStage;

  int beats = 0;
  int last_steps = 0;   // beats that were a generation's LAST step
  int last_step = 0;
  double fps = 0.0;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> last;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get());
    if (tb == nullptr || tb->dtype != TensorBeat::DType::U8) { co_return; }
    ++beats;
    shape.assign(tb->shape.begin(), tb->shape.end());
    last.assign(tb->as_u8(), tb->as_u8() + tb->byte_size());
    if (tb->sideband.is_object()) {
      FlexData sb = tb->sideband;
      auto o = sb.as_object();
      last_step = (int)o.at("step").as_int(0);
      if (last_step == (int)o.at("steps").as_int(-1)) { ++last_steps; }
      fps = o.contains("fps") ? o.at("fps").as_real(0.0) : 0.0;
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "clip", .doc = "", .type = &typeid(TensorBeatPayload),
       .tags = "rgb-clip"}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

struct Golden {
  std::string checkpoint;
  std::vector<float> latent, frames;
  int C = 0, T = 0, h = 0, w = 0, F = 0, H = 0, W = 0;
};

bool
golden_(Golden* g)
{
  const char* gd = std::getenv("VPIPE_TAE_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return false; }
  const std::string dir = std::string(gd) + "/taeh3_22";
  std::ifstream in(dir + "/meta.json");
  if (!in) { return false; }
  std::stringstream ss;
  ss << in.rdbuf();
  FlexData meta = FlexData::from_json(ss.str());
  auto o = meta.as_object();
  g->checkpoint = std::string(o.at("checkpoint").as_string(""));
  g->C = (int)o.at("C").as_int(0); g->T = (int)o.at("T").as_int(0);
  g->h = (int)o.at("h").as_int(0); g->w = (int)o.at("w").as_int(0);
  g->F = (int)o.at("F").as_int(0); g->H = (int)o.at("H").as_int(0);
  g->W = (int)o.at("W").as_int(0);
  g->latent = read_f32_(dir + "/latent.f32");
  g->frames = read_f32_(dir + "/frames.f32");
  return !g->latent.empty() && !g->frames.empty();
}

}  // namespace

// WHAT THE DENOISE-PHASE CLAIM PROMISES. latent_preview_claims books
// the TAE's decode scratch under kPhaseDenoise -- hundreds of MB at a
// real preview size -- so a stage that keeps the decoder past the
// generation is holding memory the plan says is not there, in the
// decode phase, where a tight box is tightest. An image denoiser
// therefore sets release_each_generation(); a video one keeps the TAE
// for the next clip and releases on its idle policy instead.
//
// BOTH ARMS, because only the pair says the flag is what moved: the
// default must still keep it (that is the reuse the video path is built
// on) and setting it must still leave nothing behind.
TEST(latent_preview, releasing_each_generation_leaves_nothing_resident)
{
  Golden g;
  if (!golden_(&g)) { return; }
  for (int arm = 0; arm < 2; ++arm) {
    const bool release_each = arm == 1;
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto prod_u = std::make_unique<PreviewProducer>(
        &sess, "prod", std::vector<InEdge>{}, FlexData::make_object());
    prod_u->spec_.vae = g.checkpoint;
    prod_u->spec_.every = 1;
    prod_u->spec_.max_edge = 0;
    prod_u->latent = g.latent;
    prod_u->C = g.C; prod_u->T = g.T; prod_u->h = g.h; prod_u->w = g.w;
    prod_u->steps = 2;
    prod_u->generations = 2;
    prod_u->release_each = release_each;
    prod_u->allocate_oports(1);
    prod_u->set_oport_policy(0, {2, OverrunPolicy::DropOldest});
    auto* prod =
        static_cast<PreviewProducer*>(pl->insert_stage(std::move(prod_u)));
    auto sink_u = std::make_unique<ClipSink>(
        &sess, "sink", std::vector<InEdge>{{prod, 0}},
        FlexData::make_object());
    pl->insert_stage(std::move(sink_u));

    PipelineRuntime rt(pl.get(), &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();

    std::printf("[latent-preview] release_each=%d: %zu bytes mid-scope, "
                "%zu after, %d generations\n", (int)release_each,
                prod->resident_mid, prod->resident, prod->began_count);
    // Both arms must have actually DECODED -- an arm that never loaded
    // the TAE reports 0 resident and would pass the release half while
    // proving nothing.
    EXPECT_TRUE(prod->began_count == 2);
    EXPECT_TRUE(prod->resident_mid > 0);
    if (release_each) {
      EXPECT_TRUE(prod->resident == 0);
    } else {
      EXPECT_TRUE(prod->resident > 0);
    }
  }
}

// End to end: a producer submits back to back, the worker renders on its
// own thread and pushes through the DropOldest port, a sink reads. What
// has to come out is the LAST step's preview, as a whole clip, at the
// clip's rate, and matching the reference decode -- however many of the
// earlier ones were replaced on the way.
TEST(latent_preview, the_last_step_arrives_as_a_whole_clip)
{
  Golden g;
  if (!golden_(&g)) { return; }
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto prod_u = std::make_unique<PreviewProducer>(
      &sess, "prod", std::vector<InEdge>{}, FlexData::make_object());
  prod_u->spec_.vae = g.checkpoint;
  prod_u->spec_.every = 1;
  prod_u->spec_.max_edge = 0;
  prod_u->latent = g.latent;
  prod_u->C = g.C; prod_u->T = g.T; prod_u->h = g.h; prod_u->w = g.w;
  prod_u->steps = 4;
  prod_u->allocate_oports(1);
  prod_u->set_oport_policy(0, {2, OverrunPolicy::DropOldest});
  auto* prod =
      static_cast<PreviewProducer*>(pl->insert_stage(std::move(prod_u)));
  auto sink_u = std::make_unique<ClipSink>(
      &sess, "sink", std::vector<InEdge>{{prod, 0}}, FlexData::make_object());
  auto* sink = static_cast<ClipSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(prod->began);
  EXPECT_TRUE(prod->resident > 0);
  EXPECT_TRUE(prod->rendered >= 1);
  EXPECT_TRUE(prod->rendered + prod->replaced == 4);
  EXPECT_TRUE(sink->beats >= 1);
  EXPECT_TRUE(sink->last_step == 4);
  EXPECT_TRUE((sink->shape ==
               std::vector<std::int64_t>{g.F, 3, g.H, g.W}));
  EXPECT_TRUE(std::fabs(sink->fps - 24.0) < 1e-9);
  if (sink->last.size() != g.frames.size()) { return; }
  double num = 0.0, den = 0.0;
  for (std::size_t k = 0; k < g.frames.size(); ++k) {
    const double d = sink->last[k] / 255.0 - g.frames[k];
    num += d * d;
    den += (double)g.frames[k] * g.frames[k];
  }
  const double r = std::sqrt(num / den);
  std::printf("[latent-preview] %d beat(s), %llu rendered, %llu replaced, "
              "last step %d, rel-L2 %.5f\n", sink->beats,
              (unsigned long long)prod->rendered,
              (unsigned long long)prod->replaced, sink->last_step, r);
  EXPECT_TRUE(r < 0.02);
}

// SECOND GENERATION IS THE TEST. A continuous graph runs many clips
// through one stage, so the previewer is begun and ended many times over
// one TAE: every generation must deliver its own last step, and the
// decoder's memory must start clean each time (a stale MemBlock state
// would show as a first frame that differs from the reference).
TEST(latent_preview, every_generation_delivers_its_last_step)
{
  Golden g;
  if (!golden_(&g)) { return; }
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto prod_u = std::make_unique<PreviewProducer>(
      &sess, "prod", std::vector<InEdge>{}, FlexData::make_object());
  prod_u->spec_.vae = g.checkpoint;
  prod_u->spec_.max_edge = 0;
  prod_u->latent = g.latent;
  prod_u->C = g.C; prod_u->T = g.T; prod_u->h = g.h; prod_u->w = g.w;
  prod_u->steps = 2;
  prod_u->generations = 3;
  prod_u->allocate_oports(1);
  // Deep enough that no generation's beat is evicted before the sink
  // reads it: this test is about the previewer, not the drop policy.
  prod_u->set_oport_policy(0, {16, OverrunPolicy::DropOldest});
  auto* prod =
      static_cast<PreviewProducer*>(pl->insert_stage(std::move(prod_u)));
  auto sink_u = std::make_unique<ClipSink>(
      &sess, "sink", std::vector<InEdge>{{prod, 0}}, FlexData::make_object());
  auto* sink = static_cast<ClipSink*>(pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  EXPECT_TRUE(prod->began_count == 3);
  EXPECT_TRUE(sink->last_steps == 3);
  EXPECT_TRUE(prod->rendered + prod->replaced == 6);
  // The last beat is the third generation's, and it still matches the
  // reference: nothing carried over from the two before it.
  if (sink->last.size() != g.frames.size()) { return; }
  double num = 0.0, den = 0.0;
  for (std::size_t k = 0; k < g.frames.size(); ++k) {
    const double d = sink->last[k] / 255.0 - g.frames[k];
    num += d * d;
    den += (double)g.frames[k] * g.frames[k];
  }
  EXPECT_TRUE(std::sqrt(num / den) < 0.02);
}

// Unwired costs nothing: begin() declines before loading anything, so a
// graph that never looks at previews never reads the TAE.
TEST(latent_preview, an_unwired_port_loads_nothing)
{
  Golden g;
  if (!golden_(&g)) { return; }
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto prod_u = std::make_unique<PreviewProducer>(
      &sess, "prod", std::vector<InEdge>{}, FlexData::make_object());
  prod_u->spec_.vae = g.checkpoint;
  prod_u->allocate_oports(1);
  prod_u->set_oport_policy(0, {2, OverrunPolicy::DropOldest});
  auto* prod =
      static_cast<PreviewProducer*>(pl->insert_stage(std::move(prod_u)));
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  EXPECT_FALSE(prod->began);
  EXPECT_TRUE(prod->resident == 0);
  EXPECT_TRUE(prod->rendered == 0);
}

// An IMAGE family's preview: one latent frame, decoded to one picture
// and sent as [3, H, W] -- the `rgb-frames` shape a preview stage shows
// as a still -- matching the reference decode of that frame alone.
TEST(latent_preview, an_image_preview_is_one_picture)
{
  const char* gd = std::getenv("VPIPE_TAE_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string dir = std::string(gd) + "/taew2_1_image";
  std::ifstream mj(dir + "/meta.json");
  if (!mj) { return; }
  std::stringstream ss;
  ss << mj.rdbuf();
  FlexData meta = FlexData::from_json(ss.str());
  auto mo = meta.as_object();
  const std::vector<float> latent = read_f32_(dir + "/latent.f32");
  const std::vector<float> frames = read_f32_(dir + "/frames.f32");
  if (latent.empty() || frames.empty()) { return; }
  const int H = (int)mo.at("H").as_int(0), W = (int)mo.at("W").as_int(0);

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto prod_u = std::make_unique<PreviewProducer>(
      &sess, "prod", std::vector<InEdge>{}, FlexData::make_object());
  prod_u->spec_.vae = std::string(mo.at("checkpoint").as_string(""));
  prod_u->spec_.max_edge = 0;
  prod_u->latent = latent;
  prod_u->C = (int)mo.at("C").as_int(0);
  prod_u->T = 1;
  prod_u->h = (int)mo.at("h").as_int(0);
  prod_u->w = (int)mo.at("w").as_int(0);
  prod_u->steps = 2;
  prod_u->fps = 0.0;
  prod_u->video_frames = 0;
  prod_u->trim = genai::MetalTaeDecoder::Trim::kLeading;
  prod_u->allocate_oports(1);
  prod_u->set_oport_policy(0, {2, OverrunPolicy::DropOldest});
  auto* prod =
      static_cast<PreviewProducer*>(pl->insert_stage(std::move(prod_u)));
  auto sink_u = std::make_unique<ClipSink>(
      &sess, "sink", std::vector<InEdge>{{prod, 0}}, FlexData::make_object());
  auto* sink = static_cast<ClipSink*>(pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(prod->began);
  EXPECT_TRUE(sink->last_step == 2);
  EXPECT_TRUE((sink->shape == std::vector<std::int64_t>{3, H, W}));
  // A still carries no rate: the preview stage decides still vs video by
  // exactly that.
  EXPECT_TRUE(sink->fps == 0.0);
  if (sink->last.size() != frames.size()) { return; }
  double num = 0.0, den = 0.0;
  for (std::size_t k = 0; k < frames.size(); ++k) {
    const double d = sink->last[k] / 255.0 - frames[k];
    num += d * d;
    den += (double)frames[k] * frames[k];
  }
  const double r = std::sqrt(num / den);
  std::printf("[latent-preview] image: %dx%d, rel-L2 %.5f\n", W, H, r);
  EXPECT_TRUE(r < 0.02);
}

// generate-image's preview port: appended at index 2 (the latent and the
// step_latent keep theirs), on its own clock, DropOldest, and tagged for
// the preview stage's frames port.
TEST(latent_preview, generate_image_exposes_a_push_style_preview_port)
{
  Session sess;
  GenerateImageStage gi(&sess, "gi", {}, FlexData::make_object());
  const auto& spec = gi.spec();
  ASSERT_TRUE(spec.oports.size() == 3);
  if (spec.oports.size() != 3) { return; }
  EXPECT_TRUE(spec.oports[0].name == "latent");
  EXPECT_TRUE(spec.oports[1].name == "step_latent");
  EXPECT_TRUE(spec.oports[2].name == "preview");
  EXPECT_TRUE(spec.oports[2].clock_group != spec.oports[0].clock_group);
  PreviewStage pv(&sess, "pv", {}, FlexData::make_object());
  EXPECT_TRUE(port_tags_compatible(spec.oports[2].tags,
                                   pv.spec().iports[0].tags));
  EXPECT_TRUE(gi.oport_policy(2).mode == OverrunPolicy::DropOldest);
  EXPECT_TRUE(gi.oport_policy(0).mode == OverrunPolicy::Backpressure);
  EXPECT_TRUE(gi.oport_policy(1).mode == OverrunPolicy::Backpressure);
}

// Krea-2's source carries the keys, and only when a preview VAE is named.
TEST(latent_preview, the_krea2_config_source_carries_the_keys)
{
  Session sess;
  {
    Krea2ModelConfigStage s(&sess, "k", {}, FlexData::make_object());
    EXPECT_FALSE(
        s.resolved_config().as_object().contains(latent_preview::kVaeKey));
  }
  {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert_or_assign(latent_preview::kVaeKey,
                       FlexData::make_string("madebyollin/taew2_1"));
    o.insert_or_assign(latent_preview::kEveryKey, FlexData::make_int(2));
    Krea2ModelConfigStage s(&sess, "k", {}, cfg);
    EXPECT_TRUE(s.config_error().empty());
    const LatentPreviewSpec p =
        LatentPreviewSpec::from_model_config(s.resolved_config());
    EXPECT_TRUE(p.vae == "madebyollin/taew2_1");
    EXPECT_TRUE(p.every == 2);
    EXPECT_TRUE(p.max_edge == 512);
    EXPECT_TRUE(model_config::family_of(s.resolved_config()) == "krea2");
  }
}

// FLUX.2's, Z-Image's and Qwen-Image-2.1's sources carry the keys too,
// stamped with their own family so generate-image books and renders the
// preview for them.
TEST(latent_preview, the_other_image_sources_carry_the_keys)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(latent_preview::kVaeKey,
                                   FlexData::make_string("some-tae"));
  cfg.as_object().insert_or_assign(latent_preview::kEveryKey,
                                   FlexData::make_int(3));
  Flux2ModelConfigStage f2(&sess, "f2", {}, cfg);
  ZImageModelConfigStage zi(&sess, "zi", {}, cfg);
  QwenImage21ModelConfigStage q21(&sess, "q21", {}, cfg);
  EXPECT_TRUE(f2.config_error().empty() && zi.config_error().empty() &&
              q21.config_error().empty());
  for (const FlexData& fd : {f2.resolved_config(), zi.resolved_config(),
                             q21.resolved_config()}) {
    const LatentPreviewSpec p = LatentPreviewSpec::from_model_config(fd);
    EXPECT_TRUE(p.vae == "some-tae");
    EXPECT_TRUE(p.every == 3);
  }
  EXPECT_TRUE(model_config::family_of(f2.resolved_config()) == "flux2");
  EXPECT_TRUE(model_config::family_of(zi.resolved_config()) == "z-image");
  EXPECT_TRUE(model_config::family_of(q21.resolved_config()) ==
              "qwen-image-21");
  Flux2ModelConfigStage quiet(&sess, "q", {}, FlexData::make_object());
  EXPECT_FALSE(
      quiet.resolved_config().as_object().contains(latent_preview::kVaeKey));
}

// ---- a registered (plugin) family, end to end ------------------------
//
// A family out of tree owns its denoise loop, so the host cannot take the
// clean estimate itself: the family hands it back through the request's
// named-output seam (ABI 6). This drives the REAL generate-video stage
// over a stub family that does exactly that with the taew2_1 golden's
// latent, so what reaches the preview port can be checked pixel for pixel
// against the reference decode.

namespace {

struct StubSeen {
  int asked = 0;       // output_wanted() calls
  int wanted = 0;      // ...that said yes
  int handed = 0;      // output() calls
};
StubSeen g_stub;
std::vector<float> g_stub_latent;
std::vector<int>   g_stub_shape;

class PreviewStubGenerator : public genai::VideoGenerator {
public:
  int latent_channels() const override { return 16; }
  int spatial_compression() const override { return 8; }
  bool
  generate(const genai::VideoGenRequest& req,
           genai::VideoGenResult* out) override
  {
    const int total = std::max(1, req.steps);
    for (int s = 1; s <= total; ++s) {
      // Asked BEFORE built, as the seam's contract says.
      ++g_stub.asked;
      if (req.output_wanted(genai::kOutputPreviewX0, s, total)) {
        ++g_stub.wanted;
        genai::NamedTensor t;
        t.data = g_stub_latent.data();
        t.shape = g_stub_shape;
        t.elem_size = 4;
        req.output(genai::kOutputPreviewX0, s, total, t);
        ++g_stub.handed;
      }
      if (!req.progress(s, total)) { return false; }
    }
    out->video = g_stub_latent;
    out->video_shape = g_stub_shape;
    return true;
  }
};

class PreviewStubFamily : public genai::VideoModelFamily {
public:
  std::string_view tag() const noexcept override
  {
    return "ut-preview-family";
  }
  bool
  claims(const std::string& root, const std::string&) const override
  {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(root) / "ut-preview-family.marker", ec);
  }
  std::unique_ptr<genai::VideoGenerator>
  load(const genai::VideoModelCreateArgs&) override
  {
    return std::make_unique<PreviewStubGenerator>();
  }
};

bool
register_preview_stub_()
{
  static const bool once = genai::VideoModelRegistry::get().add(
      std::make_unique<PreviewStubFamily>());
  return once;
}

// One beat on oport 0, then done.
class OneBeat : public TypedStage<OneBeat> {
public:
  static constexpr const char* kTypeName = "ut-latent-preview-one-beat";
  using TypedStage::TypedStage;
  std::unique_ptr<BeatPayloadIntf> beat;
  Job
  process(RuntimeContext& ctx) override
  {
    if (beat) { co_await ctx.write(0, std::move(beat)); }
    ctx.signal_done();
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "out", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class Drain : public TypedStage<Drain> {
public:
  static constexpr const char* kTypeName = "ut-latent-preview-drain";
  using TypedStage::TypedStage;
  int beats = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    ++beats;
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {{.name = "in", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

}  // namespace

TEST(latent_preview, a_plugin_family_previews_through_the_output_seam)
{
  const char* gd = std::getenv("VPIPE_TAE_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string dir = std::string(gd) + "/taew2_1";
  std::ifstream mj(dir + "/meta.json");
  if (!mj) { return; }
  std::stringstream ss;
  ss << mj.rdbuf();
  FlexData meta = FlexData::from_json(ss.str());
  auto mo = meta.as_object();
  const std::string ckpt(mo.at("checkpoint").as_string(""));
  const int C = (int)mo.at("C").as_int(0), T = (int)mo.at("T").as_int(0);
  const int h = (int)mo.at("h").as_int(0), w = (int)mo.at("w").as_int(0);
  const int F = (int)mo.at("F").as_int(0), H = (int)mo.at("H").as_int(0);
  const int W = (int)mo.at("W").as_int(0);
  g_stub_latent = read_f32_(dir + "/latent.f32");
  const std::vector<float> frames = read_f32_(dir + "/frames.f32");
  if (g_stub_latent.empty() || frames.empty()) { return; }
  g_stub_shape = {C, T, h, w};
  g_stub = StubSeen{};
  ASSERT_TRUE(register_preview_stub_() ||
              genai::VideoModelRegistry::get().find("ut-preview-family"));

  // A checkpoint directory only the stub claims.
  const auto root = std::filesystem::temp_directory_path() /
                    "vpipe-ut-preview-family";
  std::filesystem::create_directories(root);
  std::ofstream(root / "ut-preview-family.marker") << "x";

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  // bf16 [4, 8] conditioning: the stub reads none of it.
  auto cond_u = std::make_unique<OneBeat>(
      &sess, "cond", std::vector<InEdge>{}, FlexData::make_object());
  {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {4, 8};
    tb->resize_contiguous(32);
    cond_u->beat = std::move(tb);
  }
  cond_u->allocate_oports(1);
  auto* cond = static_cast<OneBeat*>(pl->insert_stage(std::move(cond_u)));
  // The family's config beat, stamped with its tag, carrying the preview
  // keys exactly as a plugin's own config source would.
  auto cfg_u = std::make_unique<OneBeat>(
      &sess, "cfg", std::vector<InEdge>{}, FlexData::make_object());
  {
    FlexData fd = FlexData::make_object();
    auto o = fd.as_object();
    o.insert_or_assign("model_family",
                       FlexData::make_string("ut-preview-family"));
    LatentPreviewSpec ps;
    ps.vae = ckpt;
    ps.every = 1;
    ps.max_edge = 0;
    ps.emit(fd);
    cfg_u->beat = std::make_unique<FlexDataPayload>(std::move(fd));
  }
  cfg_u->allocate_oports(1);
  auto* cfg = static_cast<OneBeat*>(pl->insert_stage(std::move(cfg_u)));

  FlexData gcfg = FlexData::make_object();
  {
    auto o = gcfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root.string()));
    o.insert_or_assign("height", FlexData::make_int(H));
    o.insert_or_assign("width", FlexData::make_int(W));
    o.insert_or_assign("frames", FlexData::make_int(F));
    o.insert_or_assign("fps", FlexData::make_real(16.0));
    o.insert_or_assign("steps", FlexData::make_int(3));
  }
  const InEdge none{nullptr, 0};
  auto gv_u = std::make_unique<GenerateVideoStage>(
      &sess, "gv",
      std::vector<InEdge>{{cond, 0}, none, none, none, none, none, none,
                          none, none, {cfg, 0}},
      gcfg);
  ASSERT_TRUE(gv_u->config_error().empty());
  auto* gv = static_cast<GenerateVideoStage*>(pl->insert_stage(
      std::move(gv_u)));
  auto lat_u = std::make_unique<Drain>(
      &sess, "lat", std::vector<InEdge>{{gv, 0}}, FlexData::make_object());
  auto* lat = static_cast<Drain*>(pl->insert_stage(std::move(lat_u)));
  auto sink_u = std::make_unique<ClipSink>(
      &sess, "pv", std::vector<InEdge>{{gv, 2}}, FlexData::make_object());
  auto* sink = static_cast<ClipSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  std::filesystem::remove_all(root);

  // The family asked every step and was told yes every step (every = 1);
  // the host took every tensor it was handed.
  EXPECT_TRUE(g_stub.asked == 3);
  EXPECT_TRUE(g_stub.wanted == 3);
  EXPECT_TRUE(g_stub.handed == 3);
  EXPECT_TRUE(lat->beats == 1);
  EXPECT_TRUE(sink->beats >= 1);
  EXPECT_TRUE(sink->last_step == 3);
  // The leading trim -- every TAEHV but H3's -- and the clip's own rate.
  EXPECT_TRUE((sink->shape == std::vector<std::int64_t>{F, 3, H, W}));
  EXPECT_TRUE(std::fabs(sink->fps - 16.0) < 1e-9);
  if (sink->last.size() != frames.size()) { return; }
  double num = 0.0, den = 0.0;
  for (std::size_t k = 0; k < frames.size(); ++k) {
    const double d = sink->last[k] / 255.0 - frames[k];
    num += d * d;
    den += (double)frames[k] * frames[k];
  }
  const double r = std::sqrt(num / den);
  std::printf("[latent-preview] plugin family: %d asked, %d handed, %d "
              "beat(s), rel-L2 %.5f\n", g_stub.asked, g_stub.handed,
              sink->beats, r);
  EXPECT_TRUE(r < 0.02);
}

#endif  // VPIPE_BUILD_APPLE_SILICON

// The picker on `preview_vae` filters by type, and a type nothing in the
// catalogue carries is a field that offers nothing -- which looks exactly
// like "no preview VAE exists". So the field's type must be one a
// catalogued TAE is registered under.
TEST(latent_preview, the_preview_vae_picker_has_something_to_offer)
{
  Session sess;
  MiniMaxH3ModelConfigStage h3(&sess, "h3", {}, FlexData::make_object());
  Krea2ModelConfigStage k2(&sess, "k2", {}, FlexData::make_object());
  Flux2ModelConfigStage f2(&sess, "f2", {}, FlexData::make_object());
  ZImageModelConfigStage zi(&sess, "zi", {}, FlexData::make_object());
  QwenImage21ModelConfigStage q21(&sess, "q21", {}, FlexData::make_object());
  for (const StageSpec* sp : {&h3.spec(), &k2.spec(), &f2.spec(),
                              &zi.spec(), &q21.spec()}) {
    std::string type;
    for (const ConfigKey& k : sp->attrs) {
      if (k.key == latent_preview::kVaeKey) {
        type = std::string(k.suggest_db_type);
      }
    }
    ASSERT_TRUE(!type.empty());
    bool found = false;
    for (const auto& e : model_catalog()) {
      if (e.model_type == type) { found = true; }
    }
    if (!found) {
      std::printf("[latent-preview] %s: no catalogued '%s'\n",
                  std::string(sp->type_name).c_str(), type.c_str());
    }
    EXPECT_TRUE(found);
  }
}

// madebyollin's TAEs are published in a GitHub repo, not on the Hub, so
// the catalogue fetches them by URL (`url_files` in the entry's `extra`
// bag, which grows without an ABI bump). The URL's destination must be a
// file the entry PINS: the registry record's `files` is what
// resolve_adapter_file() looks the checkpoint up by.
TEST(latent_preview, the_github_hosted_tae_is_fetched_by_url)
{
  const ModelCatalogEntry* e = catalog_by_name("madebyollin/taeh3");
  ASSERT_TRUE(e != nullptr);
  if (e == nullptr) { return; }
  const auto urls = catalog_url_files(*e);
  ASSERT_TRUE(urls.size() == 1);
  if (urls.size() != 1) { return; }
  EXPECT_TRUE(urls[0].first.rfind("https://", 0) == 0);
  EXPECT_TRUE(urls[0].second == "taeh3.safetensors");
  bool pinned = false;
  for (const auto& f : e->files) { pinned = pinned || f == urls[0].second; }
  EXPECT_TRUE(pinned);
  // A MODEL, not a dataset: dataset_files would register it with the
  // dataset flag and hide it from every model picker.
  EXPECT_TRUE(e->dataset_files.empty());
  EXPECT_TRUE(catalog_category(*e) == "supplement");
  // And the builder round-trips, with a malformed pair skipped rather
  // than half-read.
  ModelCatalogEntry x;
  x.extra = catalog_url_files_extra({{"https://a/b", "b"}, {"", "c"}});
  const auto back = catalog_url_files(x);
  EXPECT_TRUE(back.size() == 1 && back[0].second == "b");
  EXPECT_TRUE(catalog_url_files(ModelCatalogEntry{}).empty());
}
