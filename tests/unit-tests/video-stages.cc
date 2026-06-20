#include "minitest.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "common/session.h"
#include "common/thread-pool.h"
#include "common/beat-payload-intf.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/video-file-decoder-stage.h"
#include "stages/audio-video/video-file-encoder-stage.h"
#include "stages/audio-video/video-tokens.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>
}

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

string
tmp_path_(const char* tag, const char* ext)
{
  string p = "/tmp/vpipe-";
  p += tag;
  p += "-";
  p += to_string(getpid());
  p += ext;
  return p;
}

size_t
file_size_or_zero_(const string& path)
{
  ifstream in(path, ios::binary | ios::ate);
  if (!in) {
    return 0;
  }
  return static_cast<size_t>(in.tellg());
}

// Test-only stage that emits one VideoStreamParams header followed by
// `target_frames` plain-grey YUV420P AVFrames, then closes its
// output. Used to drive the real VideoFileEncoderStage end-to-end
// without needing an input file.
class SynthVideoSource : public TypedStage<SynthVideoSource> {
public:
  static constexpr const char* kTypeName = "ut-synth-video-source";
  using TypedStage::TypedStage;

  unsigned   target_frames = 30;
  int        width         = 320;
  int        height        = 240;
  AVRational fps           = {30, 1};

  Job process(RuntimeContext& ctx) override
  {
    const FFmpegLibraries& libs = *session()->ffmpeg_libraries();
    if (!_header_sent) {
      VideoStreamParams p{
        width, height, AV_PIX_FMT_YUV420P,
        AVRational{fps.den, fps.num},   // time_base
        fps                             // frame_rate
      };
      co_await ctx.write(0,
        make_payload<VideoStreamParamsPayload>(p));
      _header_sent = true;
      co_return;
    }
    if (_pts >= static_cast<int64_t>(target_frames)) {
      ctx.signal_done();
      co_return;
    }
    AVFrame* f = libs.avutil().api.frame_alloc();
    if (!f) {
      ctx.signal_done();
      co_return;
    }
    f->format = AV_PIX_FMT_YUV420P;
    f->width  = width;
    f->height = height;
    int rc = libs.avutil().api.frame_get_buffer(f, 0);
    if (rc < 0) {
      libs.avutil().api.frame_free(&f);
      ctx.signal_done();
      co_return;
    }
    for (int y = 0; y < height; ++y) {
      memset(f->data[0] + y * f->linesize[0], 128, width);
    }
    for (int y = 0; y < height / 2; ++y) {
      memset(f->data[1] + y * f->linesize[1], 128, width / 2);
      memset(f->data[2] + y * f->linesize[2], 128, width / 2);
    }
    f->pts = _pts++;
    auto sp = FrameRef(f, [api = &libs.avutil().api](AVFrame* x) {
      api->frame_free(&x);
    });
    co_await ctx.write(0,
        make_payload<FrameRefPayload>(std::move(sp)));
  }

private:
  bool    _header_sent = false;
  int64_t _pts         = 0;
};

}

TEST(video_stages, decoder_oport_arity_follows_config) {
  Session sess;
  CerrSilencer hush;

  {
    FlexData cfg = FlexData::from_json(R"({"input_url":"x"})");
    VideoFileDecoderStage d(&sess, "d", {}, std::move(cfg));
    EXPECT_TRUE(d.num_oports() == 2);
    EXPECT_TRUE(d.video_port() == 0);
    EXPECT_TRUE(d.audio_port() == 1);
  }
  {
    FlexData cfg = FlexData::from_json(
      R"({"input_url":"x","enable_audio":false})");
    VideoFileDecoderStage d(&sess, "d", {}, std::move(cfg));
    EXPECT_TRUE(d.num_oports() == 1);
    EXPECT_TRUE(d.video_port() == 0);
    EXPECT_TRUE(d.audio_port() == -1);
  }
  {
    FlexData cfg = FlexData::from_json(
      R"({"input_url":"x","enable_video":false})");
    VideoFileDecoderStage d(&sess, "d", {}, std::move(cfg));
    EXPECT_TRUE(d.num_oports() == 1);
    EXPECT_TRUE(d.video_port() == -1);
    EXPECT_TRUE(d.audio_port() == 0);
  }
}

// Construction succeeds for any config; the problem is recorded in
// config_error() and deferred to launch.
TEST(video_stages, decoder_missing_input_url_deferred) {
  Session sess;
  VideoFileDecoderStage d(&sess, "d", {}, FlexData::make_object());
  EXPECT_FALSE(d.config_error().empty());
}

TEST(video_stages, encoder_iport_arity_validates) {
  Session sess;
  // enable_video=true, enable_audio=false -> expects 1 iport. Passing
  // 0 input edges is recorded as a config error (deferred to launch).
  FlexData cfg = FlexData::from_json(
    R"({"output_url":"/tmp/x.mp4","enable_audio":false})");
  VideoFileEncoderStage e(&sess, "e", {}, std::move(cfg));
  EXPECT_FALSE(e.config_error().empty());
}

TEST(video_stages, encoder_missing_output_url_deferred) {
  Session sess;
  VideoFileEncoderStage e(&sess, "e", {}, FlexData::make_object());
  EXPECT_FALSE(e.config_error().empty());
}

TEST(video_stages, encoder_unit_with_synth_frames) {
  Session sess;
  CerrSilencer hush;

  string out_path = tmp_path_("enc-test", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);

  auto src_u = make_unique<SynthVideoSource>(
    &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->target_frames = 30;
  src_u->allocate_oports(1);
  auto* src = static_cast<SynthVideoSource*>(
    pl->insert_stage(std::move(src_u)));

  FlexData enc_cfg = FlexData::make_object();
  {
    auto obj = enc_cfg.as_object();
    obj.insert("output_url", FlexData::make_string(out_path));
    obj.insert("enable_audio", FlexData::make_bool(false));
    FlexData v = FlexData::make_object();
    v.as_object().insert("preset", FlexData::make_string("ultrafast"));
    obj.insert("video", std::move(v));
  }

  auto enc_u = make_unique<VideoFileEncoderStage>(
    &sess, "enc", vector<InEdge>{{src, 0}}, std::move(enc_cfg));
  pl->insert_stage(std::move(enc_u));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  size_t sz = file_size_or_zero_(out_path);
  EXPECT_TRUE(sz > 0);
  remove(out_path.c_str());
}

TEST(video_stages, round_trip_or_skips) {
  const char* in_path = std::getenv("VPIPE_TEST_VIDEO");
  if (!in_path) {
    return;
  }

  Session sess;
  CerrSilencer hush;
  string out_path = tmp_path_("roundtrip", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);

  // Decoder: video only (most likely to succeed without
  // sample-format negotiation in the encoder for arbitrary inputs).
  FlexData dec_cfg = FlexData::make_object();
  dec_cfg.as_object().insert("input_url",
                             FlexData::make_string(in_path));
  dec_cfg.as_object().insert("enable_audio", FlexData::make_bool(false));
  auto dec_u = make_unique<VideoFileDecoderStage>(
    &sess, "dec", vector<InEdge>{}, std::move(dec_cfg));
  dec_u->allocate_oports(1);
  auto* dec = static_cast<VideoFileDecoderStage*>(
    pl->insert_stage(std::move(dec_u)));

  FlexData enc_cfg = FlexData::make_object();
  enc_cfg.as_object().insert("output_url",
                             FlexData::make_string(out_path));
  enc_cfg.as_object().insert("enable_audio", FlexData::make_bool(false));
  {
    FlexData v = FlexData::make_object();
    v.as_object().insert("preset",
                         FlexData::make_string("ultrafast"));
    enc_cfg.as_object().insert("video", std::move(v));
  }
  auto enc_u = make_unique<VideoFileEncoderStage>(
    &sess, "enc", vector<InEdge>{{dec, 0}}, std::move(enc_cfg));
  pl->insert_stage(std::move(enc_u));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  size_t sz = file_size_or_zero_(out_path);
  EXPECT_TRUE(sz > 0);
  remove(out_path.c_str());
}
