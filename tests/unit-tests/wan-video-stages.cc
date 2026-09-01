// The stage-level half of the Wan video path: the RGB-to-ffmpeg seam and
// the generate-video denoiser's geometry contract.
//
// Neither of these needs a model, which is the point of testing them here
// rather than only end to end: a 14B generation takes minutes and needs a
// 64 GB box, so the parts that can be checked in milliseconds should be.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "interfaces/ui-delegate-intf.h"
#include <cmath>
#include <mutex>
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/video-tokens.h"
#include "stages/minimax-h3-model-config-stage.h"
#include "stages/model-config-source.h"
#include "stages/rgb-to-video-stage.h"
#include "stages/model-provenance.h"
#include "stages/trigger-beat.h"
#include "stages/wan2-model-config-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/wan/metal-wan-transformer.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "stages/generate-video-stage.h"
#include "stages/video-ref-encoder-stage.h"
#endif

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Emits `count` solid planar-U8-RGB frames of a fixed size, with the clip's
// rate on the sideband the way vae-decode does.
class RgbSource : public TypedStage<RgbSource> {
public:
  static constexpr const char* kTypeName = "ut-rgb-source";
  using TypedStage::TypedStage;

  int count = 3;
  int w = 64;
  int h = 32;
  double fps = 24.0;
  // What vae-decode carries onto a decoded frame when a model made it.
  string model_name;

  Job
  process(RuntimeContext& ctx) override
  {
    for (int i = 0; i < count; ++i) {
      auto b = make_unique<TensorBeatPayload>();
      b->dtype = TensorBeat::DType::U8;
      b->shape = {3, h, w};
      const size_t n = (size_t)3 * h * w;
      b->resize_contiguous(n);
      uint8_t* p = b->as_u8();
      for (size_t k = 0; k < n; ++k) { p[k] = (uint8_t)((i * 40 + k) & 0xff); }
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("frame", FlexData::make_int(i));
      sb.as_object().insert_or_assign("frames", FlexData::make_int(count));
      sb.as_object().insert_or_assign("fps", FlexData::make_real(fps));
      b->sideband = std::move(sb);
      provenance::set_model_name(b->sideband, model_name);
      co_await ctx.write(0, std::move(b));
    }
    ctx.signal_done();
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "image", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-rgb-source", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

// Counts the two beat kinds the video contract distinguishes.
class VideoSink : public TypedStage<VideoSink> {
public:
  static constexpr const char* kTypeName = "ut-video-sink";
  using TypedStage::TypedStage;

  int headers = 0;
  int frames = 0;
  int last_w = 0, last_h = 0, last_fmt = 0;
  int rate_num = 0, rate_den = 0;
  string last_model;

  Job
  process(RuntimeContext& ctx) override
  {
    auto b = co_await ctx.read(0);
    if (!b) { ctx.signal_done(); co_return; }
    if (const auto* hp =
            dynamic_cast<const VideoStreamParamsPayload*>(b.get())) {
      ++headers;
      last_w = hp->width;
      last_h = hp->height;
      last_fmt = hp->pix_fmt;
      rate_num = hp->frame_rate.num;
      rate_den = hp->frame_rate.den;
      last_model = hp->model_name;
    } else if (dynamic_cast<const FrameRefPayload*>(b.get()) != nullptr) {
      ++frames;
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "video", .doc = "", .type = &typeid(FrameRefPayload)}};
    static const StageSpec s = {.type_name = "ut-video-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// Emits `count` empty trigger beats, then closes. The payload does not
// matter to a config source -- receipt is the signal -- so this is the
// cheapest thing that can drive one.
class TickSource : public TypedStage<TickSource> {
public:
  static constexpr const char* kTypeName = "ut-tick-source";
  using TypedStage::TypedStage;

  int count = 3;

  Job
  process(RuntimeContext& ctx) override
  {
    for (int i = 0; i < count; ++i) {
      co_await ctx.write(0, make_payload<TriggerPayload>());
    }
    ctx.signal_done();
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "trigger", .doc = "", .type = &typeid(TriggerPayload)}};
    static const StageSpec s = {.type_name = "ut-tick-source", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

// Counts FlexData beats and keeps the last one.
class FlexSink : public TypedStage<FlexSink> {
public:
  static constexpr const char* kTypeName = "ut-flex-sink";
  using TypedStage::TypedStage;

  int      beats = 0;
  FlexData last;

  Job
  process(RuntimeContext& ctx) override
  {
    auto b = co_await ctx.read(0);
    if (!b) { ctx.signal_done(); co_return; }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(b.get())) {
      ++beats;
      last = f->data;
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "in", .doc = "", .type = &typeid(FlexDataPayload)}};
    static const StageSpec s = {.type_name = "ut-flex-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// Run one config source into a counting sink and report how many beats
// came out. `ticks < 0` leaves the trigger iport UNWIRED.
template <typename ConfigStage>
int
run_config_source(Session& sess, FlexData cfg, int ticks, FlexData* last)
{
  auto pl = make_unique<Pipeline>("p", &sess);
  vector<InEdge> in;
  if (ticks >= 0) {
    auto src_u = make_unique<TickSource>(&sess, "tick", vector<InEdge>{},
                                         FlexData::make_object());
    src_u->count = ticks;
    src_u->allocate_oports(1);
    auto* src = static_cast<TickSource*>(pl->insert_stage(std::move(src_u)));
    in.push_back({src, 0});
  }
  auto cs_u = make_unique<ConfigStage>(&sess, "cfg", in, std::move(cfg));
  auto* cs = pl->insert_stage(std::move(cs_u));

  auto sink_u = make_unique<FlexSink>(&sess, "sink", vector<InEdge>{{cs, 0}},
                                      FlexData::make_object());
  auto* sink = static_cast<FlexSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  // -1 rather than an assertion: EXPECT_* only works inside a TEST body,
  // and a launch failure is a distinct answer from "emitted nothing".
  if (!rt.launch()) { return -1; }
  rt.wait_idle();
  rt.stop();
  if (last != nullptr) { *last = sink->last; }
  return sink->beats;
}

}  // namespace

// The trigger contract, which is the same one text-prompt follows and the
// reason a config source can serve a continuously generating graph:
// unwired it is a one-shot, wired it emits once per inbound beat.
//
// Both halves matter and they fail differently. A one-shot that emitted
// per read would deadlock the second request (nothing else will ever
// arrive); a triggered source that emitted once would leave the consumer
// on stale parameters while the graph says they changed.
TEST(model_config, trigger_gates_emission)
{
  Session sess;
  FlexData last;
  // Unwired: exactly one beat for the run, whatever else happens.
  EXPECT_TRUE(run_config_source<Wan2ModelConfigStage>(
                  sess, FlexData::make_object(), -1, &last) == 1);
  EXPECT_TRUE(model_config::family_of(last) == "wan");
  EXPECT_TRUE(run_config_source<MiniMaxH3ModelConfigStage>(
                  sess, FlexData::make_object(), -1, &last) == 1);
  EXPECT_TRUE(model_config::family_of(last) == "minimax-h3");

  // Wired: one beat per trigger, and EOS upstream ends the stage rather
  // than leaving it emitting forever.
  EXPECT_TRUE(run_config_source<Wan2ModelConfigStage>(
                  sess, FlexData::make_object(), 4, &last) == 4);
  EXPECT_TRUE(run_config_source<MiniMaxH3ModelConfigStage>(
                  sess, FlexData::make_object(), 3, &last) == 3);
  // A trigger source that emits nothing must not produce a config beat
  // either -- the count is the trigger's, not a minimum of one.
  EXPECT_TRUE(run_config_source<MiniMaxH3ModelConfigStage>(
                  sess, FlexData::make_object(), 0, &last) == 0);
}

// One header then one frame per image, with the rate taken from the
// producer rather than the config. The header cannot precede the first
// frame -- nothing else in a generative graph knows the frame size -- so
// "exactly one header, and it arrived" is the contract under test.
TEST(rgb_to_video, header_then_one_frame_each)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RgbSource>(&sess, "src", vector<InEdge>{},
                                      FlexData::make_object());
  src_u->count = 5;
  src_u->w = 64;
  src_u->h = 32;
  src_u->fps = 24.0;
  src_u->allocate_oports(1);
  auto* src = static_cast<RgbSource*>(pl->insert_stage(std::move(src_u)));

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("fps", FlexData::make_real(16.0));
  auto cvt_u = make_unique<RgbToVideoStage>(&sess, "cvt",
                                            vector<InEdge>{{src, 0}}, cfg);
  auto* cvt = static_cast<RgbToVideoStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<VideoSink>(&sess, "sink",
                                       vector<InEdge>{{cvt, 0}},
                                       FlexData::make_object());
  auto* sink = static_cast<VideoSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(sink->headers == 1);
  EXPECT_TRUE(sink->frames == 5);
  EXPECT_TRUE(sink->last_w == 64);
  EXPECT_TRUE(sink->last_h == 32);
  // The SIDEBAND rate wins over the config: the stage that made the frames
  // knows the clip's rate, and a config default that silently overrode it
  // would retime every generated video.
  EXPECT_TRUE(sink->rate_den > 0);
  const double got = (double)sink->rate_num / (double)sink->rate_den;
  EXPECT_TRUE(got > 23.99 && got < 24.01);
  EXPECT_TRUE(cvt->frames_emitted() == 5u);
}

// The generating model rides from the frames' sideband onto the STREAM
// HEADER. It has to change currency here: a FrameRef is a bare AVFrame
// with no sideband, so the header is the only beat on this edge that can
// carry it -- and it is also the one save-video reads before it writes
// the container header. Losing it here would leave every generated clip
// anonymous with nothing failing.
TEST(rgb_to_video, carries_the_model_name_onto_the_header)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RgbSource>(&sess, "src", vector<InEdge>{},
                                      FlexData::make_object());
  src_u->count = 2;
  src_u->model_name = "local/Some-Video-Model-8bit";
  src_u->allocate_oports(1);
  auto* src = static_cast<RgbSource*>(pl->insert_stage(std::move(src_u)));

  auto cvt_u = make_unique<RgbToVideoStage>(&sess, "cvt",
                                            vector<InEdge>{{src, 0}},
                                            FlexData::make_object());
  auto* cvt = static_cast<RgbToVideoStage*>(
      pl->insert_stage(std::move(cvt_u)));
  auto sink_u = make_unique<VideoSink>(&sess, "sink",
                                       vector<InEdge>{{cvt, 0}},
                                       FlexData::make_object());
  auto* sink = static_cast<VideoSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(sink->headers == 1);
  EXPECT_TRUE(sink->last_model == "local/Some-Video-Model-8bit");

  // A stream nobody claimed stays unclaimed: no name in, none out.
  Session s2;
  auto pl2 = make_unique<Pipeline>("p2", &s2);
  auto s2_u = make_unique<RgbSource>(&s2, "src", vector<InEdge>{},
                                     FlexData::make_object());
  s2_u->count = 2;
  s2_u->allocate_oports(1);
  auto* src2 = static_cast<RgbSource*>(pl2->insert_stage(std::move(s2_u)));
  auto cvt2_u = make_unique<RgbToVideoStage>(&s2, "cvt",
                                             vector<InEdge>{{src2, 0}},
                                             FlexData::make_object());
  auto* cvt2 = static_cast<RgbToVideoStage*>(
      pl2->insert_stage(std::move(cvt2_u)));
  auto sink2_u = make_unique<VideoSink>(&s2, "sink",
                                        vector<InEdge>{{cvt2, 0}},
                                        FlexData::make_object());
  auto* sink2 = static_cast<VideoSink*>(pl2->insert_stage(std::move(sink2_u)));
  PipelineRuntime rt2(pl2.get(), &s2);
  EXPECT_TRUE(rt2.launch());
  rt2.wait_idle();
  rt2.stop();
  EXPECT_TRUE(sink2->headers == 1);
  EXPECT_TRUE(sink2->last_model.empty());
}

// yuv420p has no representation for an odd dimension -- its chroma planes
// are half-size in each axis. Refusing is the whole point: rounding would
// hand libx264 a frame it cannot encode, and the failure would surface
// much later as a corrupt file.
TEST(rgb_to_video, odd_size_refused_for_yuv420p)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RgbSource>(&sess, "src", vector<InEdge>{},
                                      FlexData::make_object());
  src_u->count = 2;
  src_u->w = 65;                        // odd
  src_u->h = 32;
  src_u->allocate_oports(1);
  auto* src = static_cast<RgbSource*>(pl->insert_stage(std::move(src_u)));
  auto cvt_u = make_unique<RgbToVideoStage>(&sess, "cvt",
                                            vector<InEdge>{{src, 0}},
                                            FlexData::make_object());
  auto* cvt = static_cast<RgbToVideoStage*>(
      pl->insert_stage(std::move(cvt_u)));
  auto sink_u = make_unique<VideoSink>(&sess, "sink",
                                       vector<InEdge>{{cvt, 0}},
                                       FlexData::make_object());
  auto* sink = static_cast<VideoSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(sink->headers == 0);
  EXPECT_TRUE(sink->frames == 0);

  // The same odd size in rgb24 is fine -- there is no chroma to subsample.
  Session s2;
  auto pl2 = make_unique<Pipeline>("p2", &s2);
  auto src2_u = make_unique<RgbSource>(&s2, "src", vector<InEdge>{},
                                       FlexData::make_object());
  src2_u->count = 2;
  src2_u->w = 65;
  src2_u->h = 32;
  src2_u->allocate_oports(1);
  auto* src2 = static_cast<RgbSource*>(pl2->insert_stage(std::move(src2_u)));
  auto c2 = FlexData::make_object();
  c2.as_object().insert_or_assign("pix_fmt", FlexData::make_string("rgb24"));
  auto cvt2_u = make_unique<RgbToVideoStage>(&s2, "cvt",
                                             vector<InEdge>{{src2, 0}}, c2);
  auto* cvt2 = static_cast<RgbToVideoStage*>(
      pl2->insert_stage(std::move(cvt2_u)));
  auto sink2_u = make_unique<VideoSink>(&s2, "sink",
                                        vector<InEdge>{{cvt2, 0}},
                                        FlexData::make_object());
  auto* sink2 = static_cast<VideoSink*>(pl2->insert_stage(std::move(sink2_u)));
  PipelineRuntime rt2(pl2.get(), &s2);
  EXPECT_TRUE(rt2.launch());
  rt2.wait_idle();
  rt2.stop();
  EXPECT_TRUE(sink2->headers == 1);
  EXPECT_TRUE(sink2->frames == 2);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

// The two geometry rules the video path cannot bend, both checked at
// config time so a bad graph is refused at launch rather than after the
// text encoder has loaded.
//
// F % 4 == 1 is the one that looks arbitrary and is not: the VAE's first
// temporal chunk is a single frame and every later one is four, so a frame
// count outside 4k+1 has no latent representation at all. Rounding it would
// silently generate a different clip length than asked for.
TEST(generate_video, geometry_config_validation)
{
  Session sess;
  auto mk = [&](int h, int w, int f) {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("height", FlexData::make_int(h));
    cfg.as_object().insert_or_assign("width", FlexData::make_int(w));
    cfg.as_object().insert_or_assign("frames", FlexData::make_int(f));
    return make_unique<GenerateVideoStage>(&sess, "gv", vector<InEdge>{}, cfg);
  };
  EXPECT_TRUE(mk(480, 832, 81)->config_error().empty());
  EXPECT_TRUE(mk(480, 832, 121)->config_error().empty());
  // ANY positive count is accepted. The legal counts are a per-FAMILY
  // rule (4k+1 for wan, 17n+5 for minimax-h3) that share almost no
  // values, and the family is not known until the checkpoint is read --
  // so the count is rounded UP at resolve time instead of rejected here,
  // and a graph can change families without being re-authored.
  EXPECT_TRUE(mk(480, 832, 80)->config_error().empty());
  EXPECT_TRUE(mk(480, 832, 100)->config_error().empty());
  EXPECT_TRUE(mk(480, 832, 56)->config_error().empty());   // an H3 count
  // The SIZE is accepted the same way and for the same reason: the legal
  // grid is the family's VAE stride times its DiT patch -- 16 for wan, 32
  // for minimax-h3 -- and the family is not known here either, so
  // resolve_config_ rounds UP once the checkpoint has been read. This
  // used to refuse anything off a 16 grid, which was both too strict (a
  // caller should not have to know the family's stride) and too loose: it
  // ACCEPTED H3 sizes that are illegal, and they failed much later with a
  // message about latent geometry. See the test below.
  EXPECT_TRUE(mk(484, 832, 81)->config_error().empty());
  EXPECT_TRUE(mk(480, 830, 81)->config_error().empty());
  EXPECT_TRUE(mk(768, 1360, 85)->config_error().empty());

  // The frame arithmetic itself: F = 1 + 4*(T-1) inverted.
  auto s = mk(480, 832, 81);
  EXPECT_TRUE(s->latent_frames() == 21);
  auto s2 = mk(480, 832, 1);
  EXPECT_TRUE(s2->latent_frames() == 1);
  auto s3 = mk(480, 832, 121);
  EXPECT_TRUE(s3->latent_frames() == 31);
}

// The two families' rounding rules. Tested on the rules THEMSELVES rather
// than through the stage, because the stage cannot round until it has read
// a checkpoint and named its family -- which needs a 25 GB model on disk.
//
// The property that matters is round UP, never down: a count rounded down
// silently delivers less video than was asked for, where rounding up costs
// at most three frames (wan) or sixteen (h3).
TEST(generate_video, frame_counts_round_up_per_family)
{
  namespace h3 = genai::minimax_h3;
  // wan: 4k+1, the first frame being its own chunk.
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(81) == 81);
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(80) == 81);
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(82) == 85);
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(100) == 101);
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(1) == 1);
  // Degenerate inputs must not produce a count below 1; 0 frames is not a
  // clip, and a negative one would size an allocation.
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(0) == 1);
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(-7) == 1);

  // minimax-h3: 17n+5, a 17-frame clip keeping 5 latents.
  EXPECT_TRUE(h3::align_num_frames(56, 17, 5) == 56);
  EXPECT_TRUE(h3::align_num_frames(50, 17, 5) == 56);
  EXPECT_TRUE(h3::align_num_frames(57, 17, 5) == 73);
  EXPECT_TRUE(h3::align_num_frames(22, 17, 5) == 22);

  // And the cross-family point of the whole change: each family's
  // canonical count is illegal for the other, so both must survive being
  // handed the other's number.
  EXPECT_TRUE(genai::MetalWanVae::align_num_frames(56) == 57);
  EXPECT_TRUE(h3::align_num_frames(81, 17, 5) == 90);
}

// What a rounded frame size has to BE, checked against the packer that
// used to refuse it.
//
// The reported failure was a 1360x768 request dying with "could not pack
// a 27x48x85 latent" -- a message about the latent, from a stage that had
// accepted the pixel size. 1360 is a multiple of 16, so the old check
// passed it; H3's VAE is 16x spatial, so its latent width is 85, which is
// ODD and cannot be split by the DiT's 2x patch. The grid is therefore 32
// for this family, not 16, and the fix is to round up to it rather than
// to explain any of that to the caller.
//
// Tested against build_packed_sequence itself, not against the rounding
// arithmetic, which would only confirm that ceil() is ceil().
TEST(generate_video, frame_sizes_round_up_to_a_patchable_latent)
{
  namespace h3 = genai::minimax_h3;
  genai::MetalMiniMaxH3Transformer::Config c;   // released patch_h/w = 2
  const int grid_w = 16 * c.patch_w, grid_h = 16 * c.patch_h;
  EXPECT_TRUE(grid_w == 32 && grid_h == 32);

  auto packs = [&](int px_h, int px_w) {
    h3::PackedLayout L;
    const std::vector<int> tags(8, h3::kTextTag);
    return h3::build_packed_sequence(tags, 27, px_h / 16, px_w / 16, 150,
                                     c.patch_h, c.patch_w,
                                     h3::kAudioChannels, {}, &L);
  };
  auto up = [](int v, int g) { return ((v + g - 1) / g) * g; };

  // The reported request, and why it failed.
  EXPECT_TRUE((1360 / 16) % c.patch_w != 0);   // latent width 85, odd
  EXPECT_TRUE(!packs(768, 1360));
  // Rounded to the family's grid it packs, and the height -- already a
  // multiple of 32 -- must not move.
  EXPECT_TRUE(up(1360, grid_w) == 1376);
  EXPECT_TRUE(up(768, grid_h) == 768);
  EXPECT_TRUE(packs(768, 1376));
  // The shipped geometry is untouched by the rounding, which is the
  // regression that would matter most.
  EXPECT_TRUE(up(960, grid_w) == 960 && up(544, grid_h) == 544);
  EXPECT_TRUE(packs(544, 960));
  // Round UP, never down: a size rounded down silently delivers a smaller
  // picture than was asked for.
  EXPECT_TRUE(up(1361, grid_w) == 1376 && up(1377, grid_w) == 1408);
}

// The stage is family-generic, but NOT by carrying the union of what its
// families need. It carries what every video model answers to -- geometry,
// length, steps, seed, residency -- and takes each family's own knobs as a
// BEAT on the model_config iport.
//
// What this pins is that surface, because the failure it replaced was
// silent: with a union config, a key belonging to the family that was not
// resident did nothing and said nothing. So: none of the moved keys may
// come back as an attr, and a stale pipeline that still sets one must be
// TOLD rather than quietly run on defaults.
TEST(generate_video, family_generic_surface)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("height", FlexData::make_int(480));
  cfg.as_object().insert_or_assign("width", FlexData::make_int(832));
  cfg.as_object().insert_or_assign("frames", FlexData::make_int(81));
  auto s = make_unique<GenerateVideoStage>(&sess, "gv", vector<InEdge>{}, cfg);
  EXPECT_TRUE(s->config_error().empty());

  const StageSpec& sp = s->spec();
  // The audio latent port exists on the type, so a graph can wire it
  // whether or not the resident family fills it.
  EXPECT_TRUE(sp.oports.size() == 2);
  bool has_audio = false;
  for (const auto& p : sp.oports) {
    if (std::string(p.name) == "audio_latent") { has_audio = true; }
  }
  EXPECT_TRUE(has_audio);
  // The wan-side ports keep their INDICES, with the h3 anchors, the
  // Ref2VA reference rows, the config port and the audio conditioning
  // appended -- which is what lets a graph written for an earlier port
  // set keep working.
  //
  // The COUNT is asserted last on purpose: it is the part that moves
  // every time a port is appended, while the indices below are the
  // invariant the test exists to protect. Read a failure here as "a port
  // was added" and check the indices before changing the number.
  EXPECT_TRUE(std::string(sp.iports[5].name) == "ref_latent0");
  EXPECT_TRUE(std::string(sp.iports[6].name) == "ref_latent1");
  EXPECT_TRUE(std::string(sp.iports[7].name) == "ref_video_rows");
  EXPECT_TRUE(std::string(sp.iports[8].name) == "ref_audio_rows");
  EXPECT_TRUE(std::string(sp.iports[9].name) == "model_config");
  EXPECT_TRUE(std::string(sp.iports[10].name) == "audio_conditioning");
  EXPECT_TRUE(sp.iports.size() == 11);
  // The tag is the whole compatibility check between a config source and
  // this port: the composer offers a source only if it matches.
  EXPECT_TRUE(port_tags_compatible("model-config",
                                   sp.iports[9].tags));

  // No family-specific key survives as an attr of this stage.
  for (const auto& k : sp.attrs) {
    const std::string key(k.key);
    EXPECT_FALSE(key == "video_shift" || key == "audio_shift" ||
                 key == "condition_timestep" || key == "audio_seconds" ||
                 key == "guidance_scale" || key == "guidance_scale_2" ||
                 key == "boundary_ratio");
  }
}

// A pipeline written against the old union still LOADS -- an unknown
// config key is not an error anywhere in this runtime -- so the only
// thing standing between it and silently running on defaults is that the
// stage says so. Constructing with a moved key must not be a config
// error (the graph is still valid) and must not be silent either.
TEST(generate_video, a_moved_config_key_is_reported_not_rejected)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("frames", FlexData::make_int(81));
  cfg.as_object().insert_or_assign("video_shift", FlexData::make_real(12.0));
  cfg.as_object().insert_or_assign("guidance_scale", FlexData::make_real(4.0));
  auto s = make_unique<GenerateVideoStage>(&sess, "gv", vector<InEdge>{}, cfg);
  // Still constructs and still runs: the key is inert, not wrong.
  EXPECT_TRUE(s->config_error().empty());
}

// The ref2va half of the split: a `video-ref-encoder` feeding
// generate-video's reference-row ports.
//
// What this pins is the SHAPE OF THE SEAM, which is the decision that
// would be expensive to walk back: a ref2va request carries a VARIABLE
// list of up to twelve heterogeneous references, so the list is one
// FlexData input rather than a port per reference -- twelve load-image /
// vae-encode chains cannot express "three clips and nine stills" without
// the graph being re-authored per request. Everything downstream of that
// (the latents' ragged geometry collapsing into rows, the plan riding on
// the conditioning's sideband) follows from it.
TEST(video_ref_encoder, request_surface)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("frames", FlexData::make_int(121));
  {
    auto refs = FlexData::make_array();
    refs.as_array().push_back(FlexData::make_string("/a.png"));
    refs.as_array().push_back(FlexData::make_string("/b.mp4"));
    cfg.as_object().insert_or_assign("references", std::move(refs));
  }
  auto s = make_unique<VideoRefEncoderStage>(&sess, "vre", vector<InEdge>{},
                                             cfg);
  // Deferred-validated: no hf_dir is not a construction error, because a
  // model-select source on the model iport can supply one.
  EXPECT_TRUE(s->config_error().empty());
  EXPECT_TRUE(s->requests_encoded() == 0);

  const StageSpec& sp = s->spec();
  EXPECT_TRUE(std::string(sp.type_name) == "video-ref-encoder");
  // The FILE references are config, not a port: they are paths to open,
  // which is what a file browser fills in. The six tensor iports sit
  // AFTER prompt and model, so a graph written against the original two
  // is unaffected, and they exist for the references a file list cannot
  // name -- a generated still, a cropped frame, a clip that was never
  // written to disk.
  EXPECT_TRUE(sp.iports.size() == 8);
  EXPECT_TRUE(std::string(sp.iports[0].name) == "prompt");
  EXPECT_TRUE(std::string(sp.iports[1].name) == "model");
  for (int i = 0; i < 6; ++i) {
    const std::string want = "ref" + std::to_string(i + 1);
    EXPECT_TRUE(std::string(sp.iports[(std::size_t)i + 2].name) == want);
    EXPECT_TRUE(*sp.iports[(std::size_t)i + 2].type ==
                typeid(TensorBeatPayload));
  }

  // The key the browser drives: a path field that is NOT a single
  // string, which is what makes the dialog multi-select, and a filter
  // naming every kind a reference may be -- the field cannot know which
  // one a given file is, so it must not exclude any of them.
  bool has_refs = false;
  for (const auto& k : sp.attrs) {
    if (std::string(k.key) != "references") { continue; }
    has_refs = true;
    // No longer REQUIRED: the tensor iports are a second source, and
    // whether they are wired is not knowable at construction.
    EXPECT_FALSE(k.required);
    EXPECT_TRUE(k.is_path);
    EXPECT_FALSE(k.path_write);
    EXPECT_TRUE(k.type != ConfigType::String);
    EXPECT_TRUE(std::string(k.path_filter) == "image,video,audio");
  }
  EXPECT_TRUE(has_refs);

  // Three outputs, and the pairing with generate-video is by NAME:
  // conditioning -> its iport0 (the same contract diffusion-conditioner
  // emits, so either can drive it), and the two row ports -> iport7/8.
  EXPECT_TRUE(sp.oports.size() == 3);
  EXPECT_TRUE(std::string(sp.oports[0].name) == "conditioning");
  EXPECT_TRUE(std::string(sp.oports[1].name) == "ref_video");
  EXPECT_TRUE(std::string(sp.oports[2].name) == "ref_audio");

  auto dit = make_unique<GenerateVideoStage>(&sess, "gv", vector<InEdge>{},
                                             FlexData::make_object());
  const StageSpec& dsp = dit->spec();
  EXPECT_TRUE(std::string(dsp.iports[0].name) == "conditioning");
  EXPECT_TRUE(dsp.iports[0].type == sp.oports[0].type);
  EXPECT_TRUE(dsp.iports[7].type == sp.oports[1].type);
  EXPECT_TRUE(dsp.iports[8].type == sp.oports[2].type);

  // `frames` is duplicated on both stages on purpose -- it is the
  // duration every reference is truncated to AND the size of the layout
  // the DiT builds -- so it has to be a key here, and generate-video
  // checks the two agree rather than trusting them.
  bool has_frames = false, has_short_edge = false, has_sample_fps = false;
  for (const auto& k : sp.attrs) {
    if (std::string(k.key) == "frames") { has_frames = true; }
    if (std::string(k.key) == "reference_image_short_edge") {
      has_short_edge = true;
    }
    if (std::string(k.key) == "video_sample_fps") { has_sample_fps = true; }
  }
  EXPECT_TRUE(has_frames && has_short_edge && has_sample_fps);
  std::printf("[video_ref_encoder] %zu iports / %zu oports; the reference "
              "LIST is config, not a port per reference\n",
              sp.iports.size(), sp.oports.size());
}

// The reference list is the one thing this stage cannot run without, so
// an empty or malformed one is a CONFIG error caught at launch rather
// than a warning at the first beat. A Ref2VA forward with no references
// generates video conditioned on nothing, at the full cost of a 33B
// model -- so failing the graph beats running it.
// The tensor reference ports' contract: kind from RANK, rates from the
// sideband, and a missing rate refused rather than guessed.
//
// This is the part of the port design that fails silently when it is
// wrong. A clip taken for a still conditions on one frozen frame; a
// soundtrack read at the wrong rate conditions on the wrong sound; and
// neither raises anything downstream -- the shapes are all still valid.
// attach_audio names the ports whose audio is a SOUNDTRACK rather than a
// reference of its own -- refused at launch when it names a port that
// does not exist, because a typo there is silently a different request:
// the soundtrack becomes an extra <Audio k> and renumbers nothing
// visible until the model is conditioned on it.
TEST(video_ref_encoder, attach_audio_names_reference_ports)
{
  Session sess;
  auto with = [&](std::vector<int> ports, const char* id) {
    auto arr = FlexData::make_array();
    for (int n : ports) { arr.as_array().push_back(FlexData::make_int(n)); }
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("attach_audio", std::move(arr));
    return std::string(
        VideoRefEncoderStage(&sess, id, vector<InEdge>{}, cfg)
            .config_error());
  };
  EXPECT_TRUE(with({3}, "a").empty());
  EXPECT_TRUE(with({1, 6}, "b").empty());
  // 0 and 7 are not reference iports -- 1..6 are.
  EXPECT_FALSE(with({0}, "c").empty());
  EXPECT_FALSE(with({7}, "d").empty());
  {
    // Not an array at all.
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("attach_audio", FlexData::make_int(3));
    VideoRefEncoderStage s(&sess, "e", vector<InEdge>{}, cfg);
    EXPECT_FALSE(s.config_error().empty());
  }
  // Absent is the default, and the default is nothing attached: an audio
  // beat with no instruction is a reference of its own.
  {
    auto cfg = FlexData::make_object();
    VideoRefEncoderStage s(&sess, "f", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
  }
  std::printf("[video_ref_encoder] attach_audio: 1..6 accepted, 0 and 7 "
              "refused at launch\n");
}

// A soundtrack that arrives on a PORT arrives at its producer's rate,
// and the audio VAE reads one rate only (32000 for the released
// checkpoint). Nothing downstream looks at `sample_rate` again --
// normalize_audio_reference truncates with it and then hands the
// waveform to a VAE that assumes its own -- so an unconformed 44.1 kHz
// beat was encoded as if it were 32 kHz: a fourth too high and 1.38x too
// long against the clip it shares a rotary clock with. Every shape stays
// valid, which is why it read as "the reference did nothing".
//
// The FILE path never had this: decode_references_ hands the decoder the
// VAE's rate and gets it back resampled in the same pass.
TEST(video_ref_encoder, a_port_soundtrack_is_conformed_to_the_vae_rate)
{
  namespace h3 = vpipe::genai::minimax_h3;
  Session sess;
  const FFmpegLibraries* libs = nullptr;
  try {
    libs = sess.ffmpeg_libraries();
  } catch (...) {
    return;   // no FFmpeg on this box -- skip vacuously
  }
  if (libs == nullptr || !libs->valid()) { return; }

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("references",
                                   FlexData::make_string("subject.png"));
  VideoRefEncoderStage st(&sess, "refenc", vector<InEdge>{}, cfg);

  // Two seconds of 440 Hz at 44100, stereo planar -- what an
  // `audio-to-pcm` set to 44100 emits.
  constexpr int kIn = 44100, kN = 2 * 44100;
  auto tone = [&](int rate, int n) {
    h3::MediaReference m;
    m.kind        = h3::MediaReference::Kind::kAudio;
    m.channels    = 2;
    m.sample_rate = rate;
    m.pcm.resize((std::size_t)2 * n);
    for (int i = 0; i < n; ++i) {
      const float v =
          0.5f * (float)std::sin(2.0 * M_PI * 440.0 * i / (double)rate);
      m.pcm[(std::size_t)i]     = v;
      m.pcm[(std::size_t)n + i] = v;
    }
    return m;
  };
  auto hz = [](const float* p, int n, int rate) {
    int cross = 0;
    for (int i = 1; i < n; ++i) {
      if ((p[i - 1] < 0.0f) != (p[i] < 0.0f)) { ++cross; }
    }
    return 0.5 * (double)cross * (double)rate / (double)n;
  };

  h3::MediaReference m = tone(kIn, kN);
  ASSERT_TRUE(st.conform_audio_rate(&m, 1));
  // With no audio VAE loaded the target is the released checkpoint's
  // 32000, which is the rate a graph has to hit either way.
  EXPECT_TRUE(m.sample_rate == 32000);
  const int got = (int)(m.pcm.size() / 2);
  EXPECT_TRUE(got > 63700 && got < 64300);        // 2 s at 32 kHz
  // The tone survived. Reinterpreting instead of resampling would leave
  // 88200 samples reading as 606 Hz.
  EXPECT_TRUE(std::fabs(hz(m.pcm.data(), got, 32000) - 440.0) < 4.0);
  EXPECT_TRUE(std::fabs(hz(m.pcm.data() + got, got, 32000) - 440.0) < 4.0);

  // A beat ALREADY at the VAE's rate is left exactly as it came: the
  // graph that was wired right pays nothing, and does not get a second
  // pass of the filter.
  h3::MediaReference ok = tone(32000, 64000);
  const std::vector<float> before = ok.pcm;
  ASSERT_TRUE(st.conform_audio_rate(&ok, 1));
  EXPECT_TRUE(ok.sample_rate == 32000);
  EXPECT_TRUE(ok.pcm == before);

  // Pixels are not audio, whatever their rate field says.
  h3::MediaReference img;
  img.kind        = h3::MediaReference::Kind::kImage;
  img.sample_rate = 44100;
  ASSERT_TRUE(st.conform_audio_rate(&img, 1));
  EXPECT_TRUE(img.sample_rate == 44100);

  std::printf("[video_ref_encoder] port soundtrack 44100 -> 32000: "
              "%d samples/ch, %.1f Hz\n", got,
              hz(m.pcm.data(), got, 32000));
}

TEST(video_ref_encoder, reference_beats_are_typed_by_rank)
{
  namespace h3 = vpipe::genai::minimax_h3;
  auto beat = [](std::vector<long long> shape, TensorBeat::DType dt) {
    TensorBeatPayload tb;
    tb.dtype = dt;
    tb.shape = std::move(shape);
    std::size_t n = 1;
    for (long long d : tb.shape) { n *= (std::size_t)d; }
    tb.resize_contiguous(n);
    tb.sideband = FlexData::make_object();
    return tb;
  };
  auto put = [](TensorBeatPayload& tb, const char* k, FlexData v) {
    tb.sideband.as_object().insert_or_assign(k, std::move(v));
  };
  h3::MediaReference m;
  std::string err;

  // [N] f32 + sr -> mono audio.
  {
    auto tb = beat({800}, TensorBeat::DType::F32);
    put(tb, "sr", FlexData::make_int(32000));
    ASSERT_TRUE(VideoRefEncoderStage::media_from_beat(tb, 1, &m, &err));
    EXPECT_TRUE(m.kind == h3::MediaReference::Kind::kAudio);
    EXPECT_TRUE(m.channels == 1 && m.sample_rate == 32000);
    EXPECT_TRUE(m.pcm.size() == 800);
  }
  // [C, N] f32 -> that many channels.
  {
    auto tb = beat({2, 400}, TensorBeat::DType::F32);
    put(tb, "sample_rate", FlexData::make_int(32000));
    m = h3::MediaReference{};
    ASSERT_TRUE(VideoRefEncoderStage::media_from_beat(tb, 2, &m, &err));
    EXPECT_TRUE(m.kind == h3::MediaReference::Kind::kAudio);
    EXPECT_TRUE(m.channels == 2 && m.pcm.size() == 800);
  }
  // [3, H, W] u8 -> a still, and the port default is short_edge 0 --
  // encode it at the size it arrived, which is the whole point of
  // handing pixels over a port instead of a path.
  {
    auto tb = beat({3, 64, 128}, TensorBeat::DType::U8);
    m = h3::MediaReference{};
    ASSERT_TRUE(VideoRefEncoderStage::media_from_beat(tb, 3, &m, &err));
    EXPECT_TRUE(m.kind == h3::MediaReference::Kind::kImage);
    EXPECT_TRUE(m.num_frames == 1 && m.height == 64 && m.width == 128);
    EXPECT_TRUE(m.short_edge == 0);
  }
  // ...and a per-reference short edge overrides that.
  {
    auto tb = beat({3, 64, 128}, TensorBeat::DType::U8);
    put(tb, "short_edge", FlexData::make_int(768));
    m = h3::MediaReference{};
    ASSERT_TRUE(VideoRefEncoderStage::media_from_beat(tb, 3, &m, &err));
    EXPECT_TRUE(m.short_edge == 768);
  }
  // [T, 3, H, W] u8 + fps -> a clip. Rank is what separates this from a
  // still, which is the case a container cannot state: [1, 3, H, W] is a
  // one-frame CLIP and [3, H, W] is a picture.
  {
    auto tb = beat({5, 3, 64, 128}, TensorBeat::DType::U8);
    put(tb, "fps", FlexData::make_real(30.0));
    m = h3::MediaReference{};
    ASSERT_TRUE(VideoRefEncoderStage::media_from_beat(tb, 4, &m, &err));
    EXPECT_TRUE(m.kind == h3::MediaReference::Kind::kVideo);
    EXPECT_TRUE(m.num_frames == 5 && m.fps > 29.9 && m.fps < 30.1);
  }
  {
    auto tb = beat({1, 3, 64, 128}, TensorBeat::DType::U8);
    put(tb, "fps", FlexData::make_real(24.0));
    m = h3::MediaReference{};
    ASSERT_TRUE(VideoRefEncoderStage::media_from_beat(tb, 4, &m, &err));
    EXPECT_TRUE(m.kind == h3::MediaReference::Kind::kVideo);
  }

  // The refusals. A rate that is not stated is NOT defaulted: 24 fps
  // would look right and condition the clip at the wrong speed.
  {
    auto tb = beat({5, 3, 64, 128}, TensorBeat::DType::U8);
    m = h3::MediaReference{};
    EXPECT_FALSE(VideoRefEncoderStage::media_from_beat(tb, 4, &m, &err));
    EXPECT_FALSE(err.empty());
  }
  {
    auto tb = beat({800}, TensorBeat::DType::F32);
    m = h3::MediaReference{};
    EXPECT_FALSE(VideoRefEncoderStage::media_from_beat(tb, 1, &m, &err));
  }
  // Float pixels are refused rather than guessed at: 0..1 and -1..1 are
  // both plausible and picking wrong is a brightness bug nothing reports.
  {
    auto tb = beat({3, 64, 128}, TensorBeat::DType::F32);
    m = h3::MediaReference{};
    EXPECT_FALSE(VideoRefEncoderStage::media_from_beat(tb, 3, &m, &err));
  }
  // Not planar RGB.
  {
    auto tb = beat({4, 64, 128}, TensorBeat::DType::U8);
    m = h3::MediaReference{};
    EXPECT_FALSE(VideoRefEncoderStage::media_from_beat(tb, 3, &m, &err));
  }
  std::printf("[video_ref_encoder] reference beats: rank types them, a "
              "missing rate is refused\n");
}

TEST(video_ref_encoder, references_may_be_empty_but_must_be_paths)
{
  Session sess;
  {
    // No `references` at all is NO LONGER a config error: a graph may
    // feed every reference through the tensor iports, and construction
    // cannot see whether those are wired (iport_connected is a
    // RuntimeContext question). The "conditioned on nothing" refusal
    // moved to process(), where the combined list is known.
    auto cfg = FlexData::make_object();
    VideoRefEncoderStage s(&sess, "a", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
  }
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references", FlexData::make_array());
    VideoRefEncoderStage s(&sess, "b", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
  }
  {
    // A non-path entry: caught here rather than as "will not open" per
    // file, because it cannot be a path at all.
    auto refs = FlexData::make_array();
    refs.as_array().push_back(FlexData::make_int(7));
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references", std::move(refs));
    VideoRefEncoderStage s(&sess, "c", vector<InEdge>{}, cfg);
    EXPECT_FALSE(s.config_error().empty());
  }
  {
    // ONE path as a bare string is the same request as a one-element
    // list -- a hand-written pipeline writes the first, the browser
    // writes the second, and neither is the spelling that works.
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references",
                                     FlexData::make_string("/one.png"));
    VideoRefEncoderStage s(&sess, "d", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
  }
}

// The other end of the model_config seam: the beat each config source
// emits has to be what its family's own parser reads back. This is the
// whole contract between the two stages, and neither one can check it
// alone -- so it is checked here, on the real parsers.
TEST(model_config, each_family_reads_back_its_own_beat)
{
  Session sess;
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("guidance_scale",
                                     FlexData::make_real(5.0));
    cfg.as_object().insert_or_assign("guidance_scale_2",
                                     FlexData::make_real(2.5));
    cfg.as_object().insert_or_assign("boundary_ratio",
                                     FlexData::make_real(0.875));
    Wan2ModelConfigStage s(&sess, "wc", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
    const auto p =
        genai::MetalWanTransformer::GenerationParams::from_flex(
            s.resolved_config());
    EXPECT_TRUE(p.guidance_scale == 5.0);
    EXPECT_TRUE(p.guidance_scale_2 == 2.5);
    EXPECT_TRUE(p.boundary_ratio == 0.875);
    // Set explicitly, so it must WIN against the checkpoint's
    // model_index.json rather than being overwritten by it.
    EXPECT_TRUE(p.boundary_ratio_set);
    // The expert split follows from the boundary and nothing else.
    EXPECT_TRUE(p.expert_for(0.9) == 0);
    EXPECT_TRUE(p.expert_for(0.5) == 1);
    EXPECT_TRUE(p.guidance_for(0) == 5.0);
    EXPECT_TRUE(p.guidance_for(1) == 2.5);
  }
  {
    // Omitting boundary_ratio must leave the key OUT of the beat: the
    // consumer falls back to the checkpoint's own model_index.json, and
    // a default emitted as if it were a choice would override it on
    // every graph that never mentioned the key.
    auto cfg = FlexData::make_object();
    Wan2ModelConfigStage s(&sess, "wc2", vector<InEdge>{}, cfg);
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_FALSE(o.contains("boundary_ratio"));
    const auto p =
        genai::MetalWanTransformer::GenerationParams::from_flex(fd);
    EXPECT_FALSE(p.boundary_ratio_set);
  }
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("video_shift", FlexData::make_real(9.0));
    cfg.as_object().insert_or_assign("audio_shift", FlexData::make_real(4.0));
    cfg.as_object().insert_or_assign("condition_timestep",
                                     FlexData::make_real(0.75));
    cfg.as_object().insert_or_assign("condition_audio_timestep",
                                     FlexData::make_real(0.5));
    cfg.as_object().insert_or_assign("audio_seconds",
                                     FlexData::make_real(2.0));
    MiniMaxH3ModelConfigStage s(&sess, "hc", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
    const auto p =
        genai::MetalMiniMaxH3Transformer::GenerationParams::from_flex(
            s.resolved_config());
    EXPECT_TRUE(p.video_shift == 9.0);
    EXPECT_TRUE(p.audio_shift == 4.0);
    EXPECT_TRUE(p.condition_timestep == 0.75);
    EXPECT_TRUE(p.condition_audio_timestep == 0.5);
    // 40 latents a second, so an explicit duration is exactly that many.
    EXPECT_TRUE(p.audio_latents(124, 24.0) == 80);
    // And 0 means "as long as the video", which is the default and the
    // only setting that keeps the two modalities in step.
    genai::MetalMiniMaxH3Transformer::GenerationParams d;
    EXPECT_TRUE(d.audio_latents(124, 24.0) ==
                genai::minimax_h3::audio_latent_num_frames(124, 24.0));
  }
  // Each source stamps the family its keys belong to, which is what lets
  // the consumer refuse a config wired to the wrong checkpoint instead
  // of applying nothing and running defaults.
  {
    Wan2ModelConfigStage w(&sess, "w", vector<InEdge>{},
                           FlexData::make_object());
    MiniMaxH3ModelConfigStage h(&sess, "h", vector<InEdge>{},
                                FlexData::make_object());
    EXPECT_TRUE(model_config::family_of(w.resolved_config()) == "wan");
    EXPECT_TRUE(model_config::family_of(h.resolved_config()) ==
                "minimax-h3");
  }
}

// Nonsense numbers are refused at LAUNCH, not by throwing from a ctor --
// the deferred-validation rule every stage here follows. Worth pinning
// because a shift of 0 collapses the schedule to a single sigma: the run
// would cost a full 33B generation and return noise.
TEST(model_config, out_of_range_values_are_deferred_config_errors)
{
  Session sess;
  auto bad_shift = FlexData::make_object();
  bad_shift.as_object().insert_or_assign("video_shift",
                                         FlexData::make_real(0.0));
  MiniMaxH3ModelConfigStage h(&sess, "h", vector<InEdge>{}, bad_shift);
  EXPECT_FALSE(h.config_error().empty());

  auto bad_t = FlexData::make_object();
  bad_t.as_object().insert_or_assign("condition_timestep",
                                     FlexData::make_real(1.5));
  MiniMaxH3ModelConfigStage h2(&sess, "h2", vector<InEdge>{}, bad_t);
  EXPECT_FALSE(h2.config_error().empty());

  auto bad_g = FlexData::make_object();
  bad_g.as_object().insert_or_assign("guidance_scale",
                                     FlexData::make_real(0.5));
  Wan2ModelConfigStage w(&sess, "w", vector<InEdge>{}, bad_g);
  EXPECT_FALSE(w.config_error().empty());

  auto bad_b = FlexData::make_object();
  bad_b.as_object().insert_or_assign("boundary_ratio",
                                     FlexData::make_real(2.0));
  Wan2ModelConfigStage w2(&sess, "w2", vector<InEdge>{}, bad_b);
  EXPECT_FALSE(w2.config_error().empty());
}

#endif  // VPIPE_BUILD_APPLE_SILICON

// generate-video must survive resolving its checkpoint with NO config
// beat in hand.
//
// This is a REGRESSION, and the shape of it is worth keeping: the stage
// learns its family the moment the checkpoint is identified, and it
// re-applies the model_config beat right there -- which is BEFORE the
// first beat can have arrived, since the config stage emits during
// process(). `_model_cfg` is therefore routinely a default FlexData at
// that point, and FlexData::as_object() on a default THROWS. An uncaught
// throw out of a stage does not make it inert; it takes the pipeline
// down mid-run, which is how this first showed up -- a Turbo graph that
// died at "FlexData::as_object: kind is not Object" after logging four
// healthy lines.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (an H3 DiT dir). Only the
// config.json is read here -- the 33B of weights load lazily at the
// first denoise, which never happens -- so this costs milliseconds.
namespace {

// Records what a stage SAID. Both halves of what this test asserts on --
// a stage's own info() and the runtime's warn() when initialize() throws
// -- go through the UI delegate, not the log one.
//
// It has to be the log rather than the outcome, because the runtime
// CATCHES a throw out of initialize(), reports it, and marks the stage
// inert: launch() still returns true and the graph still drains, so a
// test that only checks those passes over a stage that has silently
// stopped existing.
class UiCapture final : public UiDelegateIntf {
public:
  std::mutex               mu;
  std::vector<std::string> lines;
  void error(const VpipeFormat& f) override { add(f); }
  void warn(const VpipeFormat& f) override { add(f); }
  void info(const VpipeFormat& f) override { add(f); }
  UiInputStatus getline(const VpipeFormat&, std::string& out,
                        const std::function<bool()>&) override
  {
    out.clear();
    return UiInputStatus::Eof;
  }
  std::unique_ptr<UiTextStream> open_text_stream() override
  {
    return std::make_unique<NullUiTextStream>();
  }
  bool saw(const char* needle)
  {
    std::lock_guard<std::mutex> g(mu);
    for (const std::string& l : lines) {
      if (l.find(needle) != std::string::npos) { return true; }
    }
    return false;
  }
private:
  void add(const VpipeFormat& f)
  {
    std::lock_guard<std::mutex> g(mu);
    lines.push_back(f());
  }
};

}  // namespace

TEST(model_config, generate_video_survives_a_missing_beat)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  auto cap_u = make_unique<UiCapture>();
  auto* cap = cap_u.get();
  sess.set_ui_delegate(std::move(cap_u));

  auto cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root));
    o.insert_or_assign("width", FlexData::make_int(384));
    o.insert_or_assign("height", FlexData::make_int(256));
    o.insert_or_assign("frames", FlexData::make_int(41));
    o.insert_or_assign("steps", FlexData::make_int(1));
  }
  auto pl = make_unique<Pipeline>("p", &sess);
  // Every port UNWIRED, model_config included -- the case the crash
  // needed. With no conditioning beat the stage resolves its checkpoint
  // at init and then has nothing to do, which is the window the throw
  // happened in.
  auto gv_u = make_unique<GenerateVideoStage>(&sess, "gv", vector<InEdge>{},
                                              cfg);
  auto* gv = static_cast<GenerateVideoStage*>(
      pl->insert_stage(std::move(gv_u)));
  EXPECT_TRUE(gv->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // launch() succeeding is NOT the bar. The runtime catches a throw out
  // of initialize(), logs it and marks the stage inert -- so the graph
  // drains cleanly and every assertion above still passes while the
  // stage has silently stopped existing. What the bug looks like is this
  // line, and nothing else.
  EXPECT_FALSE(cap->saw("as_object"));
  EXPECT_FALSE(cap->saw("entering drain"));
  // And the positive: it got far enough to identify the checkpoint.
  EXPECT_TRUE(cap->saw("MiniMax-H3 partition"));
}

// `references` DECLARES itself a path (.is_path, so the composer's file
// browser fills it in) and was the only such key that never resolved
// one. Left raw, a RELATIVE path was opened against the process CWD --
// which under the web-ui's file sandbox is not the sandbox root -- so a
// path the browser had just produced did not exist, while an absolute
// one worked. That asymmetry is the bug; these pin the fix.
TEST(video_ref_encoder, references_are_confined_to_the_file_sandbox)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-vre-sandbox";
  std::error_code ec;
  fs::create_directories(root / "clips", ec);
  const std::string cfg_json =
      "{\"file_sandbox\":{\"enabled\":true,\"root\":\""
      + root.string() + "\"}}";
  // What the sandbox will answer with: weakly_canonical resolves
  // symlinks, and on macOS /var IS one (-> /private/var). Comparing
  // against the raw temp path would fail on the platform this runs on
  // while the resolution was perfectly correct.
  const fs::path croot = fs::weakly_canonical(root, ec);

  {
    Session sess(cfg_json);
    auto refs = FlexData::make_array();
    refs.as_array().push_back(FlexData::make_string("subject.png"));
    refs.as_array().push_back(FlexData::make_string("clips/motion.mp4"));
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references", std::move(refs));
    VideoRefEncoderStage s(&sess, "rel", vector<InEdge>{}, cfg);
    EXPECT_TRUE(s.config_error().empty());
    ASSERT_TRUE(s.references().size() == 2);
    if (s.references().size() == 2) {
      EXPECT_TRUE(s.references()[0] == (croot / "subject.png").string());
      EXPECT_TRUE(s.references()[1] ==
                  (croot / "clips" / "motion.mp4").string());
      if (s.references()[0] != (croot / "subject.png").string()) {
        std::printf("[video_ref_encoder] relative resolved to '%s', "
                    "wanted '%s'\n", s.references()[0].c_str(),
                    (croot / "subject.png").string().c_str());
      }
    }
  }
  {
    // A bare string is the same request as a one-element list, so it
    // must be confined the same way -- the spelling a hand-written
    // pipeline uses must not be the one that fails.
    Session sess(cfg_json);
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references",
                                     FlexData::make_string("subject.png"));
    VideoRefEncoderStage s(&sess, "bare", vector<InEdge>{}, cfg);
    ASSERT_TRUE(s.references().size() == 1);
    if (s.references().size() == 1) {
      EXPECT_TRUE(s.references()[0] == (croot / "subject.png").string());
    }
  }
  {
    // An ABSOLUTE path outside a granted prefix is RE-ROOTED, not taken
    // verbatim -- the confinement this key already claimed. What the
    // stage used to do instead was read it, sandbox or not.
    Session sess(cfg_json);
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references",
                                     FlexData::make_string("/etc/passwd"));
    VideoRefEncoderStage s(&sess, "abs", vector<InEdge>{}, cfg);
    ASSERT_TRUE(s.references().size() == 1);
    if (s.references().size() == 1) {
      EXPECT_TRUE(s.references()[0] == (croot / "etc" / "passwd").string());
      EXPECT_TRUE(s.references()[0] != "/etc/passwd");
    }
  }
  {
    // With NO sandbox the path is untouched, so every graph outside the
    // web-ui keeps resolving exactly as it did.
    Session plain;
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("references",
                                     FlexData::make_string("subject.png"));
    VideoRefEncoderStage s(&plain, "off", vector<InEdge>{}, cfg);
    ASSERT_TRUE(s.references().size() == 1);
    if (s.references().size() == 1) {
      EXPECT_TRUE(s.references()[0] == "subject.png");
    }
  }
  fs::remove_all(root, ec);
}
