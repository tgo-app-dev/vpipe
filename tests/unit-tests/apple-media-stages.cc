// The Apple-native media stages in real pipelines: save-image's 16-bit
// and OpenEXR path, load-image's F16 path, and avf-save-video ->
// avf-load-video (ProRes 4444 with alpha; an HEVC PQ transcode whose
// colour tags carry through untouched; F32 pictures), and the avf video
// stages wired to the FFmpeg ones at their default formats.

#include "minitest.h"

#include "apple-silicon/media/color-tags.h"
#include "apple-silicon/media/image-io.h"
#include "apple-silicon/media/video-io.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/avf-load-video-stage.h"
#include "stages/avf-save-video-stage.h"
#include "stages/audio-video/video-to-rgb-stage.h"
#include "stages/load-image-stage.h"
#include "stages/load-video-stage.h"
#include "stages/rgb-to-video-stage.h"
#include "stages/save-image-stage.h"
#include "stages/save-video-stage.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;

#define REQUIRE_(cond) do { if (!(cond)) { EXPECT_TRUE(cond); return; } } \
                       while (0)

namespace {

using half = _Float16;

std::string
tmp_(const std::string& name)
{
  return "/tmp/vpipe-apple-stages-" + std::to_string(getpid()) + "-" + name;
}

// Emits the given pictures, then ends.
class PicSource : public TypedStage<PicSource> {
public:
  static constexpr const char* kTypeName = "ut-am-source";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> pics;
  std::size_t next = 0;
  Job process(RuntimeContext& ctx) override
  {
    if (next >= pics.size()) { ctx.signal_done(); co_return; }
    TensorBeat t = pics[next++];
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(t)));
  }
};

class PicSink : public TypedStage<PicSink> {
public:
  static constexpr const char* kTypeName = "ut-am-sink";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> got;
  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (auto* t = dynamic_cast<TensorBeatPayload*>(in.get())) {
      got.push_back(static_cast<TensorBeat&>(*t));
    }
  }
};

FlexData
cfg_(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData c = FlexData::make_object();
  for (auto& [k, v] : kv) { c.as_object().insert_or_assign(k, v); }
  return c;
}

FlexData
str_(const std::string& s)
{
  return FlexData::make_string(s);
}

template <class Fn>
TensorBeat
pic_(int C, int H, int W, Fn fn)
{
  TensorBeat t;
  t.dtype = TensorBeat::DType::F16;
  t.shape = {C, H, W};
  t.data.assign((std::size_t)C * H * W * 2, 0);
  auto* p = reinterpret_cast<half*>(t.data.data());
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        p[((std::size_t)c * H + y) * W + x] = (half)fn(c, y, x);
      }
    }
  }
  return t;
}

float
at_(const TensorBeat& t, int c, int y, int x)
{
  const int H = (int)t.shape[1], W = (int)t.shape[2];
  const std::size_t i = ((std::size_t)c * H + y) * W + x;
  if (t.dtype == TensorBeat::DType::U8) { return t.bytes_()[i] / 255.f; }
  if (t.dtype == TensorBeat::DType::F32) {
    return reinterpret_cast<const float*>(t.bytes_())[i];
  }
  return (float)reinterpret_cast<const half*>(t.bytes_())[i];
}

TensorBeat
to_f32_(const TensorBeat& t)
{
  TensorBeat o;
  o.dtype = TensorBeat::DType::F32;
  o.shape = t.shape;
  o.sideband = t.sideband;
  const int C = (int)t.shape[0], H = (int)t.shape[1], W = (int)t.shape[2];
  o.data.assign((std::size_t)C * H * W * 4, 0);
  auto* p = reinterpret_cast<float*>(o.data.data());
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        p[((std::size_t)c * H + y) * W + x] = at_(t, c, y, x);
      }
    }
  }
  return o;
}

// source -> sink-stage(cfg), driven to the end; `check` sees the sink
// before the pipeline (which owns it) goes.
template <class Sink>
bool
write_(Session& sess, std::vector<TensorBeat> pics, FlexData cfg,
       std::function<void(Sink&)> check = {})
{
  Pipeline pl("p", &sess);
  auto src_u = std::make_unique<PicSource>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  src_u->pics = std::move(pics);
  auto* src = static_cast<PicSource*>(pl.insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  auto sink_u = std::make_unique<Sink>(&sess, "out",
                                       std::vector<InEdge>{{src, 0}},
                                       std::move(cfg));
  auto* sink = static_cast<Sink*>(pl.insert_stage(std::move(sink_u)));
  PipelineRuntime rt(&pl, &sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  if (check) { check(*sink); }
  return true;
}

// source-stage(cfg) -> sink, driven to the end.
template <class Source>
std::vector<TensorBeat>
read_(Session& sess, FlexData cfg)
{
  Pipeline pl("p", &sess);
  auto src_u = std::make_unique<Source>(&sess, "in", std::vector<InEdge>{},
                                        std::move(cfg));
  auto* src = static_cast<Source*>(pl.insert_stage(std::move(src_u)));
  auto sink_u = std::make_unique<PicSink>(&sess, "sink",
                                          std::vector<InEdge>{{src, 0}},
                                          FlexData::make_object());
  auto* sink = static_cast<PicSink*>(pl.insert_stage(std::move(sink_u)));
  PipelineRuntime rt(&pl, &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return sink->got;
}

}  // namespace

// F16 -> 16-bit PNG -> F16, the tags carried; and an "auto" read of a
// 16-bit file is F16 by itself.
TEST(apple_media_stages, f16_png16_round_trip)
{
  Session sess;
  const std::string path = tmp_("deep.png");
  auto p = pic_(3, 4, 32, [](int c, int, int x) {
    return (x * 37 % 1000) / 1000.0f + 0.0001f * c;
  });
  apple_media::ColorTags t;
  t.primaries = 12;
  t.transfer = 13;
  t.write(p.sideband);
  REQUIRE_(write_<SaveImageStage>(sess, {p}, cfg_({{"path", str_(path)}})));
  auto got = read_<LoadImageStage>(
      sess, cfg_({{"url", str_(path)}, {"dtype", str_("auto")}}));
  REQUIRE_(got.size() == 1);
  REQUIRE_(got[0].dtype == TensorBeat::DType::F16);
  const auto back = apple_media::ColorTags::read(got[0].sideband);
  EXPECT_TRUE(back.primaries == 12 && back.transfer == 13);
  float worst = 0;
  for (int x = 0; x < 32; ++x) {
    worst = std::max(worst, std::fabs(at_(got[0], 0, 1, x) - at_(p, 0, 1, x)));
  }
  std::printf("[apple_media_stages] png16 worst %.6f\n", worst);
  EXPECT_TRUE(worst < 1.0f / 1024);
  // A U8 read of the same file is still U8 (the default).
  auto u8 = read_<LoadImageStage>(sess, cfg_({{"url", str_(path)}}));
  REQUIRE_(u8.size() == 1);
  EXPECT_TRUE(u8[0].dtype == TensorBeat::DType::U8);
  std::remove(path.c_str());
}

// F16 with alpha and values above 1 -> OpenEXR -> F16.
TEST(apple_media_stages, exr_round_trip)
{
  Session sess;
  const std::string path = tmp_("lin.exr");
  auto p = pic_(4, 2, 8, [](int c, int, int x) {
    return c == 3 ? 1.0f : 0.5f * (x + 1);
  });
  apple_media::ColorTags t;
  t.primaries = 1;
  t.transfer = 8;
  t.write(p.sideband);
  REQUIRE_(write_<SaveImageStage>(sess, {p}, cfg_({{"path", str_(path)}})));
  auto got = read_<LoadImageStage>(
      sess, cfg_({{"url", str_(path)}, {"dtype", str_("f16")},
                  {"alpha", str_("keep")}}));
  REQUIRE_(got.size() == 1);
  REQUIRE_(got[0].shape[0] == 4);
  EXPECT_TRUE(std::fabs(at_(got[0], 0, 0, 7) - 4.0f) < 0.01f);
  EXPECT_TRUE(apple_media::ColorTags::read(got[0].sideband).transfer == 8);
  std::remove(path.c_str());
}

// ProRes 4444 with alpha, written and read by the stages, timing and
// colour in the frames' sideband.
TEST(apple_media_stages, prores4444_alpha_through_the_stages)
{
  Session sess;
  const std::string path = tmp_("alpha.mov");
  std::vector<TensorBeat> frames;
  for (int i = 0; i < 5; ++i) {
    auto f = pic_(4, 64, 96, [i](int c, int, int x) {
      if (c == 3) { return x < 48 ? 1.0f : 0.5f; }
      return 0.2f + 0.1f * c + 0.02f * i;
    });
    auto o = f.sideband.is_object() ? f.sideband : FlexData::make_object();
    o.as_object().insert_or_assign(sideband::kFpsNum,
                                   FlexData::make_uint(25));
    o.as_object().insert_or_assign(sideband::kFpsDen,
                                   FlexData::make_uint(1));
    f.sideband = o;
    frames.push_back(std::move(f));
  }
  bool finished = false;
  std::uint64_t written = 0;
  REQUIRE_(write_<AvfSaveVideoStage>(
      sess, frames, cfg_({{"path", str_(path)}}),
      std::function<void(AvfSaveVideoStage&)>(
          [&](AvfSaveVideoStage& sink) {
            finished = sink.finished();
            written = sink.frames_written();
          })));
  EXPECT_TRUE(finished);
  EXPECT_TRUE(written == 5);
  auto got = read_<AvfLoadVideoStage>(sess, cfg_({{"url", str_(path)}}));
  REQUIRE_(got.size() == 5);
  REQUIRE_(got[0].shape == (std::vector<std::int64_t>{4, 64, 96}));
  EXPECT_TRUE(got[0].dtype == TensorBeat::DType::F16);
  std::printf("[apple_media_stages] prores a=%.4f r=%.4f g=%.4f "
              "(want 0.5, 0.28, 0.38)\n", at_(got[2], 3, 5, 80),
              at_(got[4], 0, 5, 5), at_(got[4], 1, 5, 5));
  EXPECT_TRUE(std::fabs(at_(got[2], 3, 5, 80) - 0.5f) < 0.01f);
  EXPECT_TRUE(std::fabs(at_(got[4], 0, 5, 5) - 0.28f) < 0.01f);
  FlexData sb = got[3].sideband;
  auto o = sb.as_object();
  EXPECT_TRUE(o.at(sideband::kFpsNum).as_uint() == 25);
  EXPECT_TRUE(o.at(sideband::kFrame).as_int() == 3);
  EXPECT_TRUE(o.at(sideband::kPtsUs).as_int() == 120000);
  const auto t = apple_media::ColorTags::read(got[0].sideband);
  std::printf("[apple_media_stages] prores tags %d/%d/%d bits %d\n",
              t.primaries, t.transfer, t.matrix, t.source_bits);
  EXPECT_TRUE(t.source_bits == 12);
  std::remove(path.c_str());
}

// A long movie's progress: the writer counts its frames against the
// `frames` it is told to expect -- else the frames' sideband, else none --
// and the reader runs at most `oport_capacity` frames ahead (4 by
// default: unbounded, a 4K export held 3.3 GB of frames).
TEST(apple_media_stages, a_long_movie_is_counted_and_paced)
{
  Session sess;
  const std::string path = tmp_("counted.mov");
  std::vector<TensorBeat> frames;
  for (int i = 0; i < 3; ++i) {
    frames.push_back(pic_(3, 32, 48, [i](int c, int, int) {
      return 0.2f + 0.1f * c + 0.05f * i;
    }));
  }
  std::uint64_t total = 1;
  REQUIRE_(write_<AvfSaveVideoStage>(
      sess, frames,
      cfg_({{"path", str_(path)}, {"frames", FlexData::make_uint(3)}}),
      std::function<void(AvfSaveVideoStage&)>(
          [&](AvfSaveVideoStage& sink) { total = sink.progress_total(); })));
  EXPECT_TRUE(total == 3);
  REQUIRE_(write_<AvfSaveVideoStage>(
      sess, frames, cfg_({{"path", str_(path)}}),
      std::function<void(AvfSaveVideoStage&)>(
          [&](AvfSaveVideoStage& sink) { total = sink.progress_total(); })));
  EXPECT_TRUE(total == 0);  // no sideband frames: uncounted
  AvfLoadVideoStage paced(&sess, "in", {}, cfg_({{"url", str_(path)}}));
  EXPECT_TRUE(paced.oport_policy(0).capacity == 4);
  EXPECT_TRUE(paced.oport_policy(0).mode == OverrunPolicy::Backpressure);
  AvfLoadVideoStage deeper(
      &sess, "in2", {},
      cfg_({{"url", str_(path)}, {"oport_capacity", FlexData::make_uint(9)}}));
  EXPECT_TRUE(deeper.oport_policy(0).capacity == 9);
  std::remove(path.c_str());
}

// HEVC Main10 PQ -> avf-load-video -> avf-save-video (hevc) -> the copy
// declares what the original did: BT.2020, PQ, its mastering display.
TEST(apple_media_stages, hevc_pq_transcode_keeps_tags)
{
  Session sess;
  const std::string a = tmp_("pq-a.mov");
  const std::string b = tmp_("pq-b.mp4");
  std::vector<TensorBeat> frames;
  for (int i = 0; i < 3; ++i) {
    auto f = pic_(3, 64, 64, [](int c, int y, int) {
      return 0.3f + 0.4f * y / 64.0f + 0.01f * c;
    });
    apple_media::ColorTags t;
    t.primaries = 9;
    t.transfer = 16;
    t.matrix = 9;
    t.mastering = {35400, 14600, 8500, 39850, 6550, 2300, 15635, 16450,
                   10000000, 50};
    t.content_light = {1000, 400};
    t.write(f.sideband);
    frames.push_back(std::move(f));
  }
  REQUIRE_(write_<AvfSaveVideoStage>(
      sess, frames, cfg_({{"path", str_(a)}, {"codec", str_("hevc")}})));
  auto mid = read_<AvfLoadVideoStage>(sess, cfg_({{"url", str_(a)}}));
  REQUIRE_(mid.size() == 3);
  REQUIRE_(write_<AvfSaveVideoStage>(sess, mid, cfg_({{"path", str_(b)}})));
  auto got = read_<AvfLoadVideoStage>(sess, cfg_({{"url", str_(b)}}));
  REQUIRE_(got.size() == 3);
  const auto t = apple_media::ColorTags::read(got[0].sideband);
  std::printf("[apple_media_stages] transcode tags %d/%d/%d bits %d "
              "mastering %zu cll %zu\n", t.primaries, t.transfer, t.matrix,
              t.source_bits, t.mastering.size(), t.content_light.size());
  EXPECT_TRUE(t.primaries == 9 && t.transfer == 16 && t.matrix == 9);
  EXPECT_TRUE(t.source_bits == 10);
  EXPECT_TRUE(t.mastering.size() == 10 && t.content_light.size() == 2);
  EXPECT_TRUE(std::fabs(at_(got[1], 1, 32, 32) - 0.51f) < 0.02f);
  std::remove(a.c_str());
  std::remove(b.c_str());
}

// The bytes after the first `box` (a four-letter MP4 box type) in `f`;
// empty without one.
std::string
box_(const std::string& f, const char* box)
{
  const auto at = f.find(box);
  return at == std::string::npos ? std::string() : f.substr(at + 4);
}

// H.264 as a delivery spec asks for it: its profile and level, CAVLC, a
// keyframe every 4 frames, no B-frames -- read back from the file's own
// boxes (avcC's profile, level and PPS; stss's sync samples; no ctts).
TEST(apple_media_stages, h264_profile_level_gop_and_entropy)
{
  Session sess;
  const std::string path = tmp_("h264.mp4");
  std::vector<TensorBeat> frames;
  for (int i = 0; i < 12; ++i) {
    frames.push_back(pic_(3, 64, 96, [i](int c, int y, int x) {
      return 0.2f + 0.5f * ((x + 3 * i) % 96) / 96.0f + 0.1f * c +
             0.001f * y;
    }));
  }
  REQUIRE_(write_<AvfSaveVideoStage>(
      sess, frames,
      cfg_({{"path", str_(path)}, {"codec", str_("h264")},
            {"profile", str_("main")}, {"level", str_("4.1")},
            {"entropy", str_("cavlc")}, {"frame_reordering", str_("off")},
            {"keyframe_interval", FlexData::make_uint(4)},
            {"bitrate", FlexData::make_int(2000000)},
            {"max_bitrate", FlexData::make_int(3000000)}})));
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  REQUIRE_(fp != nullptr);
  std::string f;
  char buf[65536];
  for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, fp)) > 0;) {
    f.append(buf, n);
  }
  std::fclose(fp);
  // avcC: version, profile_idc, compatibility, level_idc, length size,
  // the SPS count and each (16-bit length, bytes), the PPS count, ...
  const std::string avcc = box_(f, "avcC");
  REQUIRE_(avcc.size() > 8);
  const auto u8 = [&](std::size_t i) { return (unsigned char)avcc[i]; };
  std::printf("[apple_media_stages] h264 profile_idc %d level_idc %d\n",
              u8(1), u8(3));
  EXPECT_TRUE(u8(1) == 77);   // Main
  EXPECT_TRUE(u8(3) == 41);   // 4.1
  std::size_t i = 6;
  const std::size_t sps = (std::size_t(u8(i)) << 8) | u8(i + 1);
  i += 2 + sps;
  REQUIRE_(i + 4 < avcc.size() && u8(i) >= 1);
  // The PPS: its NAL header, then ue(v) pps_id and sps_id (both 0: one
  // bit each), then entropy_coding_mode_flag -- 0 for CAVLC.
  const unsigned char pps = u8(i + 4);
  EXPECT_TRUE((pps & 0xC0) == 0xC0);
  EXPECT_TRUE((pps & 0x20) == 0);
  // stss: version / flags, then the count of keyframes -- 0, 4, 8.
  const std::string stss = box_(f, "stss");
  REQUIRE_(stss.size() > 8);
  const unsigned n = (unsigned(stss[4] & 0xff) << 24) |
                     (unsigned(stss[5] & 0xff) << 16) |
                     (unsigned(stss[6] & 0xff) << 8) | unsigned(stss[7] & 0xff);
  std::printf("[apple_media_stages] h264 keyframes %u of 12\n", n);
  EXPECT_TRUE(n >= 3);
  // No B-frames: no composition offsets.
  EXPECT_TRUE(f.find("ctts") == std::string::npos);
  // A profile H.264 has not: refused.
  EXPECT_TRUE(!apple_media::h264_settings_known("main", "6.0", ""));
  EXPECT_TRUE(!apple_media::h264_settings_known("high", "", "rle"));
  EXPECT_TRUE(apple_media::h264_settings_known("baseline", "auto", ""));
  std::remove(path.c_str());
}

// F32 pictures (0..1, video-to-rgb's format) in and out of the avf
// video stages.
TEST(apple_media_stages, f32_pictures_save_and_load)
{
  Session sess;
  const std::string path = tmp_("f32.mov");
  std::vector<TensorBeat> frames;
  for (int i = 0; i < 3; ++i) {
    frames.push_back(to_f32_(pic_(4, 64, 96, [i](int c, int, int x) {
      if (c == 3) { return x < 48 ? 1.0f : 0.5f; }
      return 0.2f + 0.1f * c + 0.05f * i;
    })));
  }
  std::uint64_t written = 0;
  REQUIRE_(write_<AvfSaveVideoStage>(
      sess, frames, cfg_({{"path", str_(path)}}),
      std::function<void(AvfSaveVideoStage&)>(
          [&](AvfSaveVideoStage& sink) { written = sink.frames_written(); })));
  EXPECT_TRUE(written == 3);
  auto got = read_<AvfLoadVideoStage>(
      sess, cfg_({{"url", str_(path)}, {"dtype", str_("f32")}}));
  REQUIRE_(got.size() == 3);
  REQUIRE_(got[0].shape == (std::vector<std::int64_t>{4, 64, 96}));
  EXPECT_TRUE(got[0].dtype == TensorBeat::DType::F32);
  std::printf("[apple_media_stages] f32 a=%.4f g=%.4f (want 0.5, 0.40)\n",
              at_(got[1], 3, 5, 80), at_(got[2], 1, 5, 5));
  EXPECT_TRUE(std::fabs(at_(got[1], 3, 5, 80) - 0.5f) < 0.01f);
  EXPECT_TRUE(std::fabs(at_(got[2], 1, 5, 5) - 0.40f) < 0.01f);
  std::remove(path.c_str());
}

// Each family's video stages feed the other's at their DEFAULT formats:
// avf-load-video (F16) -> rgb-to-video -> save-video, then load-video ->
// video-to-rgb (F32) -> avf-save-video. Every frame arrives, its colour
// within 8-bit H.264's reach.
TEST(apple_media_stages, avf_and_ffmpeg_video_stages_interconnect)
{
  Session sess;
  const std::string a = tmp_("mix-a.mov");
  const std::string b = tmp_("mix-b.mp4");
  const std::string c = tmp_("mix-c.mov");
  const int kFrames = 9;
  std::vector<TensorBeat> frames;
  for (int i = 0; i < kFrames; ++i) {
    auto f = pic_(3, 64, 96, [i](int ch, int, int) {
      return 0.25f + 0.15f * ch + 0.02f * i;
    });
    auto o = FlexData::make_object();
    o.as_object().insert_or_assign(sideband::kFpsNum,
                                   FlexData::make_uint(16));
    o.as_object().insert_or_assign(sideband::kFpsDen,
                                   FlexData::make_uint(1));
    f.sideband = o;
    frames.push_back(std::move(f));
  }
  REQUIRE_(write_<AvfSaveVideoStage>(sess, frames,
                                     cfg_({{"path", str_(a)}})));

  {
    Pipeline pl("p", &sess);
    auto* lv = pl.insert_stage(std::make_unique<AvfLoadVideoStage>(
        &sess, "lv", std::vector<InEdge>{}, cfg_({{"url", str_(a)}})));
    auto* rv = pl.insert_stage(std::make_unique<RgbToVideoStage>(
        &sess, "rv", std::vector<InEdge>{{lv, 0}},
        FlexData::make_object()));
    pl.insert_stage(std::make_unique<SaveVideoStage>(
        &sess, "sv", std::vector<InEdge>{{rv, 0}},
        cfg_({{"output_url", str_(b)},
              {"enable_audio", FlexData::make_bool(false)}})));
    PipelineRuntime rt(&pl, &sess);
    REQUIRE_(rt.launch());
    rt.wait_idle();
    rt.stop();
  }

  std::uint64_t written = 0;
  {
    Pipeline pl("p2", &sess);
    auto lv_u = std::make_unique<LoadVideoStage>(
        &sess, "lv", std::vector<InEdge>{},
        cfg_({{"input_url", str_(b)},
              {"enable_audio", FlexData::make_bool(false)}}));
    lv_u->allocate_oports(1);
    auto* lv = pl.insert_stage(std::move(lv_u));
    auto* vr = pl.insert_stage(std::make_unique<VideoToRgbStage>(
        &sess, "vr", std::vector<InEdge>{{lv, 0}},
        FlexData::make_object()));
    auto* sv = static_cast<AvfSaveVideoStage*>(
        pl.insert_stage(std::make_unique<AvfSaveVideoStage>(
            &sess, "sv", std::vector<InEdge>{{vr, 0}},
            cfg_({{"path", str_(c)}}))));
    PipelineRuntime rt(&pl, &sess);
    REQUIRE_(rt.launch());
    rt.wait_idle();
    rt.stop();
    written = sv->frames_written();
  }

  auto got = read_<AvfLoadVideoStage>(sess, cfg_({{"url", str_(c)}}));
  std::printf("[apple_media_stages] avf -> ffmpeg -> avf: %zu of %d "
              "frames (avf-save-video wrote %llu)\n", got.size(), kFrames,
              (unsigned long long)written);
  EXPECT_TRUE(written == (std::uint64_t)kFrames);
  REQUIRE_(got.size() == (std::size_t)kFrames);
  REQUIRE_(got[0].shape == (std::vector<std::int64_t>{3, 64, 96}));
  for (int ch = 0; ch < 3; ++ch) {
    const float want = 0.25f + 0.15f * ch + 0.02f * 6;
    const float v = at_(got[6], ch, 32, 48);
    std::printf("[apple_media_stages]   frame 6 channel %d %.4f "
                "(want %.4f)\n", ch, v, want);
    EXPECT_TRUE(std::fabs(v - want) < 0.02f);
  }
  std::remove(a.c_str());
  std::remove(b.c_str());
  std::remove(c.c_str());
}

// A camera RAW through load-image: routed to ImageIO by itself, F16 under
// `dtype: auto`, developed upright -- and the EXIF on the metadata oport
// says Orientation 1, or every viewer of a saved copy would turn it
// again. Gated on VPIPE_TEST_RAW, as apple_media's RAW test is.
namespace {

class MetaSink : public TypedStage<MetaSink> {
public:
  static constexpr const char* kTypeName = "ut-am-meta-sink";
  using TypedStage::TypedStage;
  std::vector<FlexData> got;
  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (auto* f = dynamic_cast<FlexDataPayload*>(in.get())) {
      got.push_back(f->data);
    }
  }
};

}  // namespace

TEST(apple_media_stages, load_image_develops_a_camera_raw)
{
  const char* path = std::getenv("VPIPE_TEST_RAW");
  if (path == nullptr) {
    std::printf("[apple_media_stages] SKIP: set VPIPE_TEST_RAW to a camera "
                "RAW\n");
    return;
  }
  apple_media::StillInfo info;
  std::string err;
  REQUIRE_(apple_media::probe_still(path, &info, &err));

  Session sess;
  Pipeline pl("p", &sess);
  auto* li = pl.insert_stage(std::make_unique<LoadImageStage>(
      &sess, "li", std::vector<InEdge>{},
      cfg_({{"url", str_(path)}, {"dtype", str_("auto")}})));
  auto* pics = static_cast<PicSink*>(pl.insert_stage(
      std::make_unique<PicSink>(&sess, "pics",
                                std::vector<InEdge>{{li, 0}},
                                FlexData::make_object())));
  auto* meta = static_cast<MetaSink*>(pl.insert_stage(
      std::make_unique<MetaSink>(&sess, "meta",
                                 std::vector<InEdge>{{li, 1}},
                                 FlexData::make_object())));
  PipelineRuntime rt(&pl, &sess);
  REQUIRE_(rt.launch());
  rt.wait_idle();
  rt.stop();

  REQUIRE_(pics->got.size() == 1 && meta->got.size() == 1);
  const TensorBeat& t = pics->got[0];
  std::printf("[apple_media_stages] raw -> %s [%lld, %lld, %lld]\n",
              t.dtype == TensorBeat::DType::F16 ? "F16" : "other",
              (long long)t.shape[0], (long long)t.shape[1],
              (long long)t.shape[2]);
  EXPECT_TRUE(t.dtype == TensorBeat::DType::F16);
  EXPECT_TRUE(t.shape == (std::vector<std::int64_t>{3, info.height,
                                                    info.width}));
  const auto tags = apple_media::ColorTags::read(t.sideband);
  EXPECT_TRUE(tags.primaries == 1 && tags.transfer == 13);

  FlexData m = meta->got[0];
  auto mo = m.as_object();
  REQUIRE_(mo.contains("exif"));
  FlexData ex = mo.at("exif");
  auto eo = ex.as_object();
  const std::uint64_t orient =
      eo.contains("Orientation") ? eo.at("Orientation").as_uint() : 0;
  std::printf("[apple_media_stages] file orientation %d, carried EXIF "
              "Orientation %llu, Make %s\n", info.orientation,
              (unsigned long long)orient,
              eo.contains("Make")
                  ? std::string(eo.at("Make").as_string()).c_str() : "-");
  EXPECT_TRUE(orient == 1);
  EXPECT_TRUE(eo.contains("Make"));
}
