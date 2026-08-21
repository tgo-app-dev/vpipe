#include "minitest.h"
#include "interfaces/session-services-intf.h"
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
#include "stages/load-video-stage.h"
#include "stages/save-video-stage.h"
#include "stages/audio-video/video-tokens.h"
#include "stages/model-provenance.h"
#include "vpipe/vpipe.h"
#include "apple-silicon/tensor-beat.h"

#include <cmath>
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
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
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

// One container-level metadata value out of a written file, read back
// through the same dynamically-loaded libavformat the stage muxed with
// -- there is no other ffmpeg in this process, and reading it any other
// way would be testing a different library than the one that wrote it.
string
container_tag_(Session& sess, const string& path, const char* key)
{
  const FFmpegLibraries* libs = sess.services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->valid()) { return {}; }
  AVFormatContext* ic = nullptr;
  if (libs->avformat().api.open_input(&ic, path.c_str(), nullptr,
                                      nullptr) < 0) {
    return {};
  }
  string out;
  const auto* e = libs->avutil().api.dict_get(ic->metadata, key, nullptr, 0);
  if (e != nullptr && e->value != nullptr) { out = e->value; }
  libs->avformat().api.close_input(&ic);
  return out;
}

// Test-only stage that emits one VideoStreamParams header followed by
// `target_frames` plain-grey YUV420P AVFrames, then closes its
// output. Used to drive the real SaveVideoStage end-to-end
// without needing an input file.
class SynthVideoSource : public TypedStage<SynthVideoSource> {
public:
  static constexpr const char* kTypeName = "ut-synth-video-source";
  using TypedStage::TypedStage;

  unsigned   target_frames = 30;
  int        width         = 320;
  int        height        = 240;
  AVRational fps           = {30, 1};
  // What rgb-to-video puts on the header when a generative graph made
  // the frames. Empty is the file-sourced case.
  string     model_name;

  // The very thing Stage::reset_run_state exists for: without this the
  // source is exhausted after run 1 and a relaunch emits nothing.
  void reset_run_state() override { _header_sent = false; _pts = 0; }

  Job process(RuntimeContext& ctx) override
  {
    const FFmpegLibraries& libs = *session()->services()->ffmpeg_libraries();
    if (!_header_sent) {
      VideoStreamParams p{
        width, height, AV_PIX_FMT_YUV420P,
        AVRational{fps.den, fps.num},   // time_base
        fps                             // frame_rate
      };
      auto hdr = make_unique<VideoStreamParamsPayload>(p);
      hdr->model_name = model_name;
      co_await ctx.write(0, std::move(hdr));
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

// Counts the FrameRefs a decoder hands out, ignoring the stream header.
class FrameCounter : public TypedStage<FrameCounter> {
public:
  static constexpr const char* kTypeName = "ut-frame-counter";
  using TypedStage::TypedStage;

  unsigned frames = 0;

  void reset_run_state() override { frames = 0; }

  Job process(RuntimeContext& ctx) override
  {
    auto t = co_await ctx.read(0);
    if (!t) { ctx.signal_done(); co_return; }
    if (dynamic_cast<const FrameRefPayload*>(t.get()) != nullptr) {
      ++frames;
    }
  }
};

}

TEST(video_stages, decoder_oport_arity_follows_config) {
  Session sess;
  CerrSilencer hush;

  {
    FlexData cfg = FlexData::from_json(R"({"input_url":"x"})");
    LoadVideoStage d(&sess, "d", {}, std::move(cfg));
    EXPECT_TRUE(d.num_oports() == 2);
    EXPECT_TRUE(d.video_port() == 0);
    EXPECT_TRUE(d.audio_port() == 1);
  }
  {
    FlexData cfg = FlexData::from_json(
      R"({"input_url":"x","enable_audio":false})");
    LoadVideoStage d(&sess, "d", {}, std::move(cfg));
    EXPECT_TRUE(d.num_oports() == 1);
    EXPECT_TRUE(d.video_port() == 0);
    EXPECT_TRUE(d.audio_port() == -1);
  }
  {
    FlexData cfg = FlexData::from_json(
      R"({"input_url":"x","enable_video":false})");
    LoadVideoStage d(&sess, "d", {}, std::move(cfg));
    EXPECT_TRUE(d.num_oports() == 1);
    EXPECT_TRUE(d.video_port() == -1);
    EXPECT_TRUE(d.audio_port() == 0);
  }
}

// Construction succeeds for any config; the problem is recorded in
// config_error() and deferred to launch.
TEST(video_stages, decoder_missing_input_url_deferred) {
  Session sess;
  LoadVideoStage d(&sess, "d", {}, FlexData::make_object());
  EXPECT_FALSE(d.config_error().empty());
}

TEST(video_stages, encoder_iport_arity_validates) {
  Session sess;
  // enable_video=true, enable_audio=false -> expects 1 iport. Passing
  // 0 input edges is recorded as a config error (deferred to launch).
  FlexData cfg = FlexData::from_json(
    R"({"output_url":"/tmp/x.mp4","enable_audio":false})");
  SaveVideoStage e(&sess, "e", {}, std::move(cfg));
  EXPECT_FALSE(e.config_error().empty());
}

TEST(video_stages, encoder_missing_output_url_deferred) {
  Session sess;
  SaveVideoStage e(&sess, "e", {}, FlexData::make_object());
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

  auto enc_u = make_unique<SaveVideoStage>(
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

// save-video across a RELAUNCH. This is the sharpest case of the
// per-run-state class: drain() finalizes the file and leaves
// _video_eos / _finalized / _header_written set, so before
// reset_run_state() a second launch saw itself as already finished and
// wrote NOTHING AT ALL -- no file, no warning.
TEST(video_stages, encoder_relaunch_writes_again) {
  Session sess;
  CerrSilencer hush;

  const string out_path = tmp_path_("enc-relaunch", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SynthVideoSource>(
    &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->target_frames = 10;
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
  auto enc_u = make_unique<SaveVideoStage>(
    &sess, "enc", vector<InEdge>{{src, 0}}, std::move(enc_cfg));
  pl->insert_stage(std::move(enc_u));

  size_t first = 0;
  for (int run = 0; run < 2; ++run) {
    remove(out_path.c_str());
    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    const size_t sz = file_size_or_zero_(out_path);
    if (run == 0) { first = sz; }
    // Every launch must produce a real file, not just the first.
    EXPECT_TRUE(sz > 0);
  }
  EXPECT_TRUE(first > 0);
  remove(out_path.c_str());
}

// Every frame handed to the encoder must come back out of the file.
// It did not: the encoder leaves AVPacket::duration at 0, the mov
// muxer derives each sample's duration from the NEXT sample's
// timestamp and so has nothing for the last one, and the resulting
// track is one frame short. The edit list is written to that short
// duration, so a conformant demuxer trims the final frame and an
// N-frame clip plays back N-1. A file-size assertion cannot see this;
// only a count can.
TEST(video_stages, encoder_keeps_every_frame) {
  Session sess;
  CerrSilencer hush;

  const unsigned kFrames = 33;   // odd, and not a GOP multiple
  const string out_path = tmp_path_("enc-count", ".mp4");
  remove(out_path.c_str());

  {
    auto pl = make_unique<Pipeline>("p", &sess);
    auto src_u = make_unique<SynthVideoSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
    src_u->target_frames = kFrames;
    src_u->fps = AVRational{16, 1};
    src_u->allocate_oports(1);
    auto* src = static_cast<SynthVideoSource*>(
      pl->insert_stage(std::move(src_u)));

    FlexData enc_cfg = FlexData::make_object();
    enc_cfg.as_object().insert("output_url",
                               FlexData::make_string(out_path));
    enc_cfg.as_object().insert("enable_audio", FlexData::make_bool(false));
    auto enc_u = make_unique<SaveVideoStage>(
      &sess, "enc", vector<InEdge>{{src, 0}}, std::move(enc_cfg));
    pl->insert_stage(std::move(enc_u));

    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
  }

  ASSERT_TRUE(file_size_or_zero_(out_path) > 0);

  unsigned got = 0;
  {
    auto pl = make_unique<Pipeline>("p2", &sess);
    FlexData dec_cfg = FlexData::make_object();
    dec_cfg.as_object().insert("input_url",
                               FlexData::make_string(out_path));
    dec_cfg.as_object().insert("enable_audio", FlexData::make_bool(false));
    auto dec_u = make_unique<LoadVideoStage>(
      &sess, "dec", vector<InEdge>{}, std::move(dec_cfg));
    dec_u->allocate_oports(1);
    auto* dec = static_cast<LoadVideoStage*>(
      pl->insert_stage(std::move(dec_u)));

    auto cnt_u = make_unique<FrameCounter>(
      &sess, "cnt", vector<InEdge>{{dec, 0}}, FlexData::make_object());
    auto* cnt = static_cast<FrameCounter*>(
      pl->insert_stage(std::move(cnt_u)));

    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    got = cnt->frames;
  }

  if (got != kFrames) {
    // cerr is silenced above; the COUNT is the whole diagnosis, since
    // "one short" and "nothing at all" are entirely different bugs.
    cout << "decoded " << got << " frames, expected " << kFrames << "\n";
  }
  EXPECT_TRUE(got == kFrames);
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
  auto dec_u = make_unique<LoadVideoStage>(
    &sess, "dec", vector<InEdge>{}, std::move(dec_cfg));
  dec_u->allocate_oports(1);
  auto* dec = static_cast<LoadVideoStage*>(
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
  auto enc_u = make_unique<SaveVideoStage>(
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

namespace {

// Emits one stereo PCM TensorBeat the way `audio-vae-decode` does:
// planar f32 [channels, n_samples] with the rate and channel count on
// the sideband. There is no ffmpeg header, because nothing upstream of
// a GENERATED soundtrack has one to give.
class SynthPcmSource : public TypedStage<SynthPcmSource> {
public:
  static constexpr const char* kTypeName = "ut-synth-pcm-source";
  using TypedStage::TypedStage;

  int channels    = 2;
  int sample_rate = 32000;
  int samples     = 32000;      // 1 s
  bool send_rate  = true;
  // What audio-vae-decode carries forward from the latent it decoded.
  string model_name;

  void reset_run_state() override { _sent = false; }

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    auto b = make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::F32;
    b->shape = {channels, samples};
    b->resize_contiguous(static_cast<size_t>(channels) * samples);
    float* p = b->as_f32();
    for (int c = 0; c < channels; ++c) {
      // A different tone per channel, so a downmix or a channel swap
      // is visible in the decoded file rather than merely plausible.
      const double f = (c == 0) ? 440.0 : 660.0;
      for (int i = 0; i < samples; ++i) {
        p[static_cast<size_t>(c) * samples + i] =
          0.25f * static_cast<float>(
            sin(2.0 * 3.14159265358979 * f * i / sample_rate));
      }
    }
    FlexData sb = FlexData::make_object();
    if (send_rate) {
      sb.as_object().insert_or_assign(
        "sample_rate", FlexData::make_int(sample_rate));
    }
    sb.as_object().insert_or_assign(
      "channels", FlexData::make_int(channels));
    b->sideband = std::move(sb);
    provenance::set_model_name(b->sideband, model_name);
    co_await ctx.write(0, std::move(b));
  }

  const StageSpec& spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "audio", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-synth-pcm-source",
                                .doc = "", .display_name = "",
                                .oports = op};
    return s;
  }
private:
  bool _sent = false;
};

}  // namespace

// save-video muxing a GENERATED soundtrack: raw stereo PCM on the audio
// port, with no AudioStreamParams header anywhere. The rate and channel
// count come off the beat's sideband, which is the only place they
// exist -- a sink that defaulted them would resample or downmix the
// whole clip without saying so.
TEST(video_stages, encoder_muxes_raw_stereo_pcm) {
  Session sess;
  CerrSilencer hush;

  const string out_path = tmp_path_("enc-av", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto vsrc_u = make_unique<SynthVideoSource>(
    &sess, "vsrc", vector<InEdge>{}, FlexData::make_object());
  vsrc_u->target_frames = 24;
  vsrc_u->allocate_oports(1);
  auto* vsrc = static_cast<SynthVideoSource*>(
    pl->insert_stage(std::move(vsrc_u)));

  auto asrc_u = make_unique<SynthPcmSource>(
    &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  asrc_u->allocate_oports(1);
  auto* asrc = static_cast<SynthPcmSource*>(
    pl->insert_stage(std::move(asrc_u)));

  FlexData enc_cfg = FlexData::make_object();
  {
    auto obj = enc_cfg.as_object();
    obj.insert("output_url", FlexData::make_string(out_path));
    FlexData v = FlexData::make_object();
    v.as_object().insert("preset", FlexData::make_string("ultrafast"));
    obj.insert("video", std::move(v));
  }
  auto enc_u = make_unique<SaveVideoStage>(
    &sess, "enc", vector<InEdge>{{vsrc, 0}, {asrc, 0}},
    std::move(enc_cfg));
  auto* enc = static_cast<SaveVideoStage*>(
    pl->insert_stage(std::move(enc_u)));
  EXPECT_TRUE(enc->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const size_t sz = file_size_or_zero_(out_path);
  printf("[video_stages] a/v mux wrote %zu bytes\n", sz);
  EXPECT_TRUE(sz > 0);

  // The file must actually carry BOTH streams. A muxer that wrote only
  // the video track still produces a plausible, playable mp4 -- which
  // is exactly why the size check above is not enough.
  int n_audio = 0, n_video = 0, ach = 0, arate = 0;
  {
    const FFmpegLibraries& libs = *sess.services()->ffmpeg_libraries();
    AVFormatContext* ic = nullptr;
    if (libs.avformat().api.open_input(&ic, out_path.c_str(), nullptr,
                                       nullptr) == 0) {
      if (libs.avformat().api.find_stream_info(ic, nullptr) >= 0) {
        for (unsigned i = 0; i < ic->nb_streams; ++i) {
          const AVCodecParameters* cp = ic->streams[i]->codecpar;
          if (cp->codec_type == AVMEDIA_TYPE_AUDIO) {
            ++n_audio;
            ach   = cp->ch_layout.nb_channels;
            arate = cp->sample_rate;
          } else if (cp->codec_type == AVMEDIA_TYPE_VIDEO) {
            ++n_video;
          }
        }
      }
      libs.avformat().api.close_input(&ic);
    }
  }
  printf("[video_stages] streams: %d video, %d audio (%d ch @ %d Hz)\n",
         n_video, n_audio, ach, arate);
  EXPECT_TRUE(n_video == 1);
  EXPECT_TRUE(n_audio == 1);
  EXPECT_TRUE(ach == 2);
  EXPECT_TRUE(arate == 32000);
  remove(out_path.c_str());
}

// No sample_rate on the beat means the sink CANNOT know the rate. It
// must refuse rather than pick one: a wrong rate retimes the whole
// soundtrack against the picture, which plays fine and is wrong.
TEST(video_stages, encoder_refuses_pcm_without_a_rate) {
  Session sess;
  CerrSilencer hush;

  const string out_path = tmp_path_("enc-norate", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto vsrc_u = make_unique<SynthVideoSource>(
    &sess, "vsrc", vector<InEdge>{}, FlexData::make_object());
  vsrc_u->target_frames = 6;
  vsrc_u->allocate_oports(1);
  auto* vsrc = static_cast<SynthVideoSource*>(
    pl->insert_stage(std::move(vsrc_u)));
  auto asrc_u = make_unique<SynthPcmSource>(
    &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  asrc_u->samples   = 8000;
  asrc_u->send_rate = false;
  asrc_u->allocate_oports(1);
  auto* asrc = static_cast<SynthPcmSource*>(
    pl->insert_stage(std::move(asrc_u)));

  FlexData enc_cfg = FlexData::make_object();
  {
    auto obj = enc_cfg.as_object();
    obj.insert("output_url", FlexData::make_string(out_path));
    FlexData v = FlexData::make_object();
    v.as_object().insert("preset", FlexData::make_string("ultrafast"));
    obj.insert("video", std::move(v));
  }
  auto enc_u = make_unique<SaveVideoStage>(
    &sess, "enc", vector<InEdge>{{vsrc, 0}, {asrc, 0}},
    std::move(enc_cfg));
  pl->insert_stage(std::move(enc_u));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  // The VIDEO still gets written: losing the soundtrack must not cost
  // the picture. Before ready_to_write_header_ treated an EOS'd port as
  // settled, the unusable audio beat held the header back forever and
  // this produced NO FILE AT ALL.
  const size_t sz = file_size_or_zero_(out_path);
  printf("[video_stages] no-rate run wrote %zu bytes\n", sz);
  EXPECT_TRUE(sz > 0);
  remove(out_path.c_str());
}

// Provenance: generate-video stamps `model_name`, vae-decode carries it
// and rgb-to-video copies it onto the stream header, so a GENERATED clip
// reaches save-video knowing what produced it. That has to survive into
// the container, and MP4 makes it awkward: the default iTunes-style
// metadata holds a fixed set of atoms and drops an unknown key without a
// word, so the muxer is asked for the QuickTime `mdta` form on the way
// past. The tag being readable back is the whole point -- a file-size
// assertion cannot see a dropped key.
TEST(video_stages, generated_clip_records_the_model_in_the_container) {
  Session sess;
  CerrSilencer hush;
  const string out_path = tmp_path_("enc-prov", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SynthVideoSource>(
    &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->target_frames = 4;
  src_u->width = 64;
  src_u->height = 48;
  src_u->model_name = "local/Some-Video-Model-8bit";
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
  pl->insert_stage(make_unique<SaveVideoStage>(
    &sess, "enc", vector<InEdge>{{src, 0}}, std::move(enc_cfg)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const string sw = container_tag_(sess, out_path,
                                   "com.tgous.vpipe.software");
  printf("[video_stages] com.tgous.vpipe.software = %s\n", sw.c_str());
  EXPECT_TRUE(sw.rfind("Vpipe ", 0) == 0);
  EXPECT_TRUE(sw.find(vpipe_version_number()) != string::npos);
  EXPECT_TRUE(sw.find(vpipe_build_hash()) != string::npos);
  EXPECT_TRUE(sw.find(" with local/Some-Video-Model-8bit") != string::npos);
  remove(out_path.c_str());
}

// ...and a clip with no model on its header -- a plain transcode -- gets
// no tag at all. vpipe did not author that footage, and the same rule
// keeps the container's metadata layout untouched for every graph that
// was working before this existed.
TEST(video_stages, transcoded_clip_gets_no_provenance_tag) {
  Session sess;
  CerrSilencer hush;
  const string out_path = tmp_path_("enc-noprov", ".mp4");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SynthVideoSource>(
    &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->target_frames = 4;
  src_u->width = 64;
  src_u->height = 48;
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
  pl->insert_stage(make_unique<SaveVideoStage>(
    &sess, "enc", vector<InEdge>{{src, 0}}, std::move(enc_cfg)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(file_size_or_zero_(out_path) > 0);
  EXPECT_TRUE(container_tag_(sess, out_path,
                             "com.tgous.vpipe.software").empty());
  remove(out_path.c_str());
}

// A soundtrack-only save has no video header, so the audio beat's
// sideband is the ONLY place the model name can arrive. It is read there
// too -- otherwise a generated audio file would be the one output of the
// stack that came out anonymous.
TEST(video_stages, generated_audio_only_records_the_model) {
  Session sess;
  CerrSilencer hush;
  const string out_path = tmp_path_("enc-prov-audio", ".m4a");
  remove(out_path.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto asrc_u = make_unique<SynthPcmSource>(
    &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  asrc_u->samples = 8000;
  asrc_u->model_name = "local/Some-Audio-Model";
  asrc_u->allocate_oports(1);
  auto* asrc = static_cast<SynthPcmSource*>(
    pl->insert_stage(std::move(asrc_u)));

  FlexData enc_cfg = FlexData::make_object();
  {
    auto obj = enc_cfg.as_object();
    obj.insert("output_url", FlexData::make_string(out_path));
    obj.insert("enable_video", FlexData::make_bool(false));
  }
  pl->insert_stage(make_unique<SaveVideoStage>(
    &sess, "enc", vector<InEdge>{{asrc, 0}}, std::move(enc_cfg)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const string sw = container_tag_(sess, out_path,
                                   "com.tgous.vpipe.software");
  printf("[video_stages] audio-only tag = %s\n", sw.c_str());
  EXPECT_TRUE(sw.find(" with local/Some-Audio-Model") != string::npos);
  remove(out_path.c_str());
}
