// The stage-level half of the Wan video path: the RGB-to-ffmpeg seam and
// the generate-video denoiser's geometry contract.
//
// Neither of these needs a model, which is the point of testing them here
// rather than only end to end: a 14B generation takes minutes and needs a
// 64 GB box, so the parts that can be checked in milliseconds should be.

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
#include "stages/audio-video/video-tokens.h"
#include "stages/rgb-to-video-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/wan/metal-wan-vae.h"
#include "stages/generate-video-stage.h"
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

}  // namespace

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
  // 16 = the VAE's 8x spatial compression times the DiT's 2x patch, so a
  // size off that grid cannot be patchified without a partial token.
  EXPECT_TRUE(!mk(484, 832, 81)->config_error().empty());
  EXPECT_TRUE(!mk(480, 830, 81)->config_error().empty());

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

// The stage is family-generic: one stage type carrying the UNION of what
// its DiT families need, the way generate-image does. What this pins is
// the SURFACE -- a graph must not have to be rewired to change
// checkpoints, so the H3-only keys and the audio port have to exist
// regardless of which family is resident (and be inert when it is wan).
TEST(generate_video, family_generic_surface)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("height", FlexData::make_int(480));
  cfg.as_object().insert_or_assign("width", FlexData::make_int(832));
  cfg.as_object().insert_or_assign("frames", FlexData::make_int(81));
  // Keys that only minimax-h3 reads. On a wan checkpoint they are inert,
  // NOT a config error -- rejecting them would make the union useless.
  cfg.as_object().insert_or_assign("video_shift", FlexData::make_real(12.0));
  cfg.as_object().insert_or_assign("audio_shift", FlexData::make_real(3.0));
  cfg.as_object().insert_or_assign("condition_timestep",
                                   FlexData::make_real(1.0));
  cfg.as_object().insert_or_assign("audio_seconds", FlexData::make_real(5.0));
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
  // And the wan-side ports are all still there, with the h3 last-frame
  // anchor appended: the five wan ports keep their INDICES, which is what
  // lets a graph written for wan keep working unedited.
  EXPECT_TRUE(sp.iports.size() == 7);
  EXPECT_TRUE(std::string(sp.iports[5].name) == "ref_latent0");
  EXPECT_TRUE(std::string(sp.iports[6].name) == "ref_latent1");

  bool has_shift = false, has_audio_secs = false;
  for (const auto& k : sp.attrs) {
    if (std::string(k.key) == "video_shift") { has_shift = true; }
    if (std::string(k.key) == "audio_seconds") { has_audio_secs = true; }
  }
  EXPECT_TRUE(has_shift && has_audio_secs);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
