#include "minitest.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/encoded-segment.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "common/segment-publisher.h"
#include "common/segment-writer.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/load-video-stage.h"
#include "stages/save-video-stage.h"
#include "stages/audio-video/video-to-rgb-stage.h"
#include "stages/audio-video/video-tokens.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
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
  string p = "/tmp/vpipe-v2rgb-";
  p += tag;
  p += "-";
  p += to_string(getpid());
  p += ext;
  return p;
}

// Emits N solid-luma YUV420P frames so the encoder produces a valid
// h.264 mp4 we can ingest back as an EncodedSegment.
class SolidYuvSource : public TypedStage<SolidYuvSource> {
public:
  static constexpr const char* kTypeName = "ut-solid-yuv-source";
  using TypedStage::TypedStage;

  unsigned   target_frames = 8;
  int        width         = 320;
  int        height        = 240;
  uint8_t    y_level       = 128;
  uint8_t    u_level       = 128;
  uint8_t    v_level       = 128;
  AVRational fps           = {30, 1};

  Job process(RuntimeContext& ctx) override
  {
    const FFmpegLibraries& libs = *session()->ffmpeg_libraries();
    if (!_header_sent) {
      VideoStreamParams p{
        width, height, AV_PIX_FMT_YUV420P,
        AVRational{fps.den, fps.num},
        fps
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
    f->format = AV_PIX_FMT_YUV420P;
    f->width  = width;
    f->height = height;
    if (libs.avutil().api.frame_get_buffer(f, 0) < 0) {
      libs.avutil().api.frame_free(&f);
      ctx.signal_done();
      co_return;
    }
    for (int y = 0; y < height; ++y) {
      memset(f->data[0] + y * f->linesize[0], y_level, width);
    }
    for (int y = 0; y < height / 2; ++y) {
      memset(f->data[1] + y * f->linesize[1], u_level, width / 2);
      memset(f->data[2] + y * f->linesize[2], v_level, width / 2);
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

// Emits one EncodedSegment then signals done. Used to feed the
// video-to-rgb stage exactly one segment in a unit test.
class OneSegmentSource : public TypedStage<OneSegmentSource> {
public:
  static constexpr const char* kTypeName = "ut-one-segment-source";
  using TypedStage::TypedStage;

  EncodedSegment seg;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) {
      ctx.signal_done();
      co_return;
    }
    _sent = true;
    co_await ctx.write(0,
        make_payload<EncodedSegmentPayload>(seg));
  }

private:
  bool _sent = false;
};

// Emits a vector of EncodedSegments in order, one per process()
// call. Used to simulate rtsp-capture's per-AU live emission where
// the decoder must retain SPS/PPS state across Beats.
class SegmentSequenceSource
  : public TypedStage<SegmentSequenceSource>
{
public:
  static constexpr const char* kTypeName = "ut-segment-sequence-source";
  using TypedStage::TypedStage;

  std::vector<EncodedSegment> segs;

  Job process(RuntimeContext& ctx) override
  {
    if (_i >= segs.size()) {
      ctx.signal_done();
      co_return;
    }
    auto idx = _i++;
    co_await ctx.write(0,
        make_payload<EncodedSegmentPayload>(segs[idx]));
  }

private:
  std::size_t _i = 0;
};

// Collects every TensorBeat it sees on iport 0, then exits at EOS.
// Snapshots are protected by a mutex so the test harness can read them
// after wait_idle.
class TensorBeatSink : public TypedStage<TensorBeatSink> {
public:
  static constexpr const char* kTypeName = "ut-tensor-beat-sink";
  using TypedStage::TypedStage;

  std::vector<TensorBeat>& collected() { return _collected; }

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    const TensorBeatPayload* tb =
        dynamic_cast<const TensorBeatPayload*>(in.get());
    if (tb) {
      std::lock_guard<std::mutex> g(_mu);
      _collected.push_back(*tb);
    }
  }

private:
  std::mutex              _mu;
  std::vector<TensorBeat> _collected;
};

// Produce one mp4 worth of frames via SolidYuvSource + the real
// SaveVideoStage, then read it back as an EncodedSegment.
// Returns an EncodedSegment whose `data` is AVCC NAL units and
// `extradata` is the AVCDecoderConfigurationRecord -- the same shape
// rtsp-capture emits.
bool
make_solid_color_segment_(Session&        sess,
                          uint8_t         y, uint8_t u, uint8_t v,
                          int             w, int h,
                          unsigned        frames,
                          EncodedSegment& out)
{
  string mp4 = tmp_path_("clip", ".mp4");
  remove(mp4.c_str());

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SolidYuvSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->target_frames = frames;
  src_u->width         = w;
  src_u->height        = h;
  src_u->y_level       = y;
  src_u->u_level       = u;
  src_u->v_level       = v;
  src_u->allocate_oports(1);
  auto* src = static_cast<SolidYuvSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData enc_cfg = FlexData::make_object();
  {
    auto obj = enc_cfg.as_object();
    obj.insert("output_url", FlexData::make_string(mp4));
    obj.insert("enable_audio", FlexData::make_bool(false));
    FlexData vcfg = FlexData::make_object();
    vcfg.as_object().insert("preset",
                            FlexData::make_string("ultrafast"));
    obj.insert("video", std::move(vcfg));
  }
  auto enc_u = make_unique<SaveVideoStage>(
      &sess, "enc", vector<InEdge>{{src, 0}}, std::move(enc_cfg));
  pl->insert_stage(std::move(enc_u));

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) {
    return false;
  }
  rt.wait_idle();
  rt.stop();

  // Read the mp4 back.
  SegmentInfo info;
  info.path       = mp4;
  info.start_utc  = chrono::system_clock::time_point{};
  info.end_utc    = chrono::system_clock::time_point{};
  info.duration_us = 0;
  info.has_audio  = false;

  ExtractResult er = extract_encoded_segments(
      sess.ffmpeg_libraries(), &sess, info, "ut-cam", "ut-key");
  remove(mp4.c_str());

  if (!er.video.has_value()) {
    return false;
  }
  out = std::move(*er.video);
  return !out.data.empty();
}

}

TEST(video_to_rgb_stage, shape_and_strides_match_segment) {
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 8,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  // The encoder may emit fewer packets than input frames (B-frame
  // buffering, GOP edge effects with ultrafast preset), but a
  // healthy decode loop must emit MORE than one frame per
  // segment: a single-frame return is the smoking gun for the
  // "whole AVCC blob fed as one packet" bug (where the decoder
  // consumes just the first access unit and discards the rest).
  EXPECT_TRUE(got.size() >= 4u);
  EXPECT_TRUE(got.size() <= 8u);
  for (const auto& tb : got) {
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 240);
    EXPECT_TRUE(tb.shape[2] == 320);
    if (!tb.strides.empty()) {
      EXPECT_TRUE(tb.strides.size() == 3u);
      EXPECT_TRUE(tb.strides[2] == 1);
      // Outer pitch >= width.
      EXPECT_TRUE(tb.strides[1] >= 320);
      // Storage (bytes) matches strides[0] * 3 elements.
      EXPECT_TRUE(tb.data.size()
                  == static_cast<size_t>(3) * tb.strides[0]
                                            * tb.element_byte_size());
    } else {
      EXPECT_TRUE(tb.data.size()
                  == static_cast<size_t>(3) * 240 * 320
                                            * tb.element_byte_size());
    }
  }
}

TEST(video_to_rgb_stage, normalize_default_yields_unit_range) {
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  bool any = false;
  for (const auto& tb : sink->collected()) {
    auto contig = tb.materialize_contiguous_as<float>();
    EXPECT_TRUE(contig.size() == 3u * 240u * 320u);
    for (float f : contig) {
      EXPECT_TRUE(f >= -1e-6f);
      EXPECT_TRUE(f <= 1.0f + 1e-6f);
      any = true;
    }
  }
  EXPECT_TRUE(any);
}

TEST(video_to_rgb_stage, hwaccel_auto_decodes_same_frame_count) {
  // hwaccel=auto must produce the same frame-count contract as the
  // software path (>=4, <=8 for an 8-frame solid-grey segment) on any
  // host. On Apple-Silicon this exercises the videotoolbox download
  // path; elsewhere it harmlessly falls back to software.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 8,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hwaccel", FlexData::make_string("auto"));
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  EXPECT_TRUE(got.size() >= 4u);
  EXPECT_TRUE(got.size() <= 8u);
}

TEST(video_to_rgb_stage, hwaccel_none_forces_software_decode) {
  // hwaccel=none must skip the hw_config probe and produce the same
  // frame-count contract. This is the fallback knob users will reach
  // for if the auto path misbehaves on a particular stream.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 8,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hwaccel", FlexData::make_string("none"));
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  EXPECT_TRUE(got.size() >= 4u);
  EXPECT_TRUE(got.size() <= 8u);
}

TEST(video_to_rgb_stage, u8_output_dtype_emits_byte_tensor) {
  // output_dtype="u8" emits raw RGB bytes in [0, 255] with no float
  // conversion. Mid-grey segment -> all bytes around 128.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("output_dtype", FlexData::make_string("u8"));
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!sink->collected().empty());
  for (const auto& tb : sink->collected()) {
    EXPECT_TRUE(tb.dtype == TensorBeat::DType::U8);
    EXPECT_TRUE(tb.element_byte_size() == 1u);
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 240);
    EXPECT_TRUE(tb.shape[2] == 320);
  }
  // Centre pixel of second frame should be ~128 on each channel.
  const TensorBeat& tb =
      sink->collected()[sink->collected().size() > 1 ? 1 : 0];
  auto contig = tb.materialize_contiguous_as<uint8_t>();
  const int H = 240, W = 320;
  const int cy = H / 2, cx = W / 2;
  const int r = contig[0 * H * W + cy * W + cx];
  const int g = contig[1 * H * W + cy * W + cx];
  const int b = contig[2 * H * W + cy * W + cx];
  EXPECT_TRUE(r > 128 - 12 && r < 128 + 12);
  EXPECT_TRUE(g > 128 - 12 && g < 128 + 12);
  EXPECT_TRUE(b > 128 - 12 && b < 128 + 12);
}

TEST(video_to_rgb_stage, solid_color_segment_round_trip) {
  Session sess;
  CerrSilencer hush;

  // Mid-grey (Y=128, U=V=128) under BT.601/709 round-trips to mid-
  // grey RGB ~ (128, 128, 128) / 255 ~= 0.502 on each channel.
  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!sink->collected().empty());
  // Inspect the centre pixel of the second frame (first frame can be
  // a keyframe with slightly different rounding through the encoder).
  const TensorBeat& tb =
      sink->collected()[sink->collected().size() > 1 ? 1 : 0];
  auto contig = tb.materialize_contiguous_as<float>();
  const int H = 240, W = 320;
  const int cy = H / 2, cx = W / 2;
  const float r = contig[0 * H * W + cy * W + cx];
  const float g = contig[1 * H * W + cy * W + cx];
  const float b = contig[2 * H * W + cy * W + cx];
  // Mid-grey: every channel should be ~0.5. Tolerance is loose
  // because the YUV420 -> RGB pipeline plus h264 encoder rounding
  // can drift a few units of 1/255.
  const float kTol = 12.0f / 255.0f;
  EXPECT_TRUE(r > 0.5f - kTol && r < 0.5f + kTol);
  EXPECT_TRUE(g > 0.5f - kTol && g < 0.5f + kTol);
  EXPECT_TRUE(b > 0.5f - kTol && b < 0.5f + kTol);
}

namespace {

// In-place convert an AVCC-framed EncodedSegment to annex-B framing.
// Used to exercise the live-path framing without spinning up an RTSP
// server: take the mp4-derived AVCC bytes and swap each 4-byte BE
// length prefix for a {00 00 00 01} start code; expand the
// AVCDecoderConfigurationRecord into start-code-prefixed SPS+PPS so
// the H.264 decoder selects its annex-B parser.
void
to_annexb_(EncodedSegment& seg)
{
  std::vector<uint8_t> ed_out;
  if (!seg.extradata.empty() && seg.extradata[0] == 0x01) {
    const uint8_t* e  = seg.extradata.data();
    const size_t   es = seg.extradata.size();
    auto append_nal = [&](const uint8_t* nal, size_t len) {
      ed_out.push_back(0); ed_out.push_back(0);
      ed_out.push_back(0); ed_out.push_back(1);
      ed_out.insert(ed_out.end(), nal, nal + len);
    };
    if (es >= 7) {
      const int nb_sps = e[5] & 0x1F;
      size_t p = 6;
      for (int i = 0; i < nb_sps && p + 2 <= es; ++i) {
        const uint16_t len =
            (static_cast<uint16_t>(e[p]) << 8) | e[p + 1];
        p += 2;
        if (p + len > es) { break; }
        append_nal(e + p, len);
        p += len;
      }
      if (p < es) {
        const int nb_pps = e[p++];
        for (int i = 0; i < nb_pps && p + 2 <= es; ++i) {
          const uint16_t len =
              (static_cast<uint16_t>(e[p]) << 8) | e[p + 1];
          p += 2;
          if (p + len > es) { break; }
          append_nal(e + p, len);
          p += len;
        }
      }
    }
  }
  seg.extradata = std::move(ed_out);

  std::vector<uint8_t> data_out;
  data_out.reserve(seg.data.size());
  const uint8_t* d  = seg.data.data();
  const size_t   ds = seg.data.size();
  size_t pos = 0;
  while (pos + 4 <= ds) {
    const uint32_t nal_len =
        (static_cast<uint32_t>(d[pos    ]) << 24)
      | (static_cast<uint32_t>(d[pos + 1]) << 16)
      | (static_cast<uint32_t>(d[pos + 2]) <<  8)
      |  static_cast<uint32_t>(d[pos + 3]);
    if (nal_len == 0 || pos + 4 + nal_len > ds) { break; }
    data_out.push_back(0); data_out.push_back(0);
    data_out.push_back(0); data_out.push_back(1);
    data_out.insert(data_out.end(),
                    d + pos + 4, d + pos + 4 + nal_len);
    pos += 4 + nal_len;
  }
  seg.data = std::move(data_out);
}

}

TEST(video_to_rgb_stage, annexb_framing_decodes_same_frames) {
  // The rtsp-capture live path emits per-GOP Beats in annex-B framing
  // (raw NALs with 00 00 00 01 start codes -- what FFmpeg's rtp
  // depacketizer and global-header encoder both produce). This test
  // takes an AVCC fixture, converts to annex-B in place, and confirms
  // video-to-rgb decodes the same frame count it would have for the
  // AVCC original.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 8,
                                 seg)) {
    return;
  }
  to_annexb_(seg);
  // Sanity-check: data now begins with a start code.
  EXPECT_TRUE(seg.data.size() >= 4);
  EXPECT_TRUE(seg.data[0] == 0x00 && seg.data[1] == 0x00
              && seg.data[2] == 0x00 && seg.data[3] == 0x01);

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hwaccel", FlexData::make_string("none"));
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  // Same lower bound as the AVCC fixture-shape test: more than one
  // frame proves the AU splitter sliced the annex-B blob correctly
  // rather than handing the whole thing to send_packet once.
  EXPECT_TRUE(got.size() >= 4u);
  EXPECT_TRUE(got.size() <= 8u);
  for (const auto& tb : got) {
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 240);
    EXPECT_TRUE(tb.shape[2] == 320);
  }
}

namespace {

// Walk the annex-B `seg` and split it into per-AU EncodedSegments
// (one VCL NAL plus any leading non-VCL NALs per output). Mimics
// what rtsp-capture's per-AU publisher would deliver to video-to-rgb
// over the wire; lets us assert that the decoder retains SPS/PPS
// state across Beat boundaries (the regression that broke when
// flush_buffers fired between Beats).
std::vector<EncodedSegment>
split_into_per_au_(const EncodedSegment& seg)
{
  std::vector<EncodedSegment> out;
  if (seg.data.size() < 3) { return out; }
  const uint8_t* d = seg.data.data();
  const size_t   n = seg.data.size();

  auto scan_sc = [&](size_t i) -> int {
    if (i + 3 < n
        && d[i]   == 0 && d[i+1] == 0
        && d[i+2] == 0 && d[i+3] == 1) {
      return 4;
    }
    if (i + 2 < n
        && d[i] == 0 && d[i+1] == 0 && d[i+2] == 1) {
      return 3;
    }
    return 0;
  };

  size_t au_start = 0;
  bool   have_au  = false;
  bool   have_vcl = false;
  std::vector<std::pair<size_t, size_t>> aus;
  size_t i = 0;
  while (i + 2 < n) {
    int sc = scan_sc(i);
    if (sc == 0) { ++i; continue; }
    const size_t nal_hdr = i + sc;
    if (nal_hdr >= n) { break; }
    const uint8_t t = d[nal_hdr] & 0x1F;
    const bool is_vcl = (t == 1 || t == 5);
    if (!have_au) {
      au_start = i;
      have_au  = true;
    } else if (is_vcl && have_vcl) {
      aus.emplace_back(au_start, i - au_start);
      au_start = i;
      have_vcl = false;
    }
    if (is_vcl) { have_vcl = true; }
    i = nal_hdr + 1;
  }
  if (have_au && au_start < n) {
    aus.emplace_back(au_start, n - au_start);
  }

  out.reserve(aus.size());
  for (const auto& [off, len] : aus) {
    EncodedSegment s = seg;          // copy codec params + extradata
    s.data.assign(seg.data.begin() + static_cast<ptrdiff_t>(off),
                  seg.data.begin()
                    + static_cast<ptrdiff_t>(off + len));
    out.push_back(std::move(s));
  }
  return out;
}

}

TEST(video_to_rgb_stage, per_au_emission_decodes_continuous_stream) {
  // Regression guard for the rtsp-capture per-AU live path:
  // each Beat carries one access unit. The H.264 decoder must
  // retain SPS/PPS state across Beats so P-frame Beats decode
  // against the SPS/PPS that arrived in the earlier IDR Beat.
  // The pre-per-AU code called flush_buffers() at the tail of
  // decode_segment_, which wiped that state -- only IDR Beats
  // ever decoded. This test feeds N per-AU annex-B Beats in
  // order and asserts the decoder still emits ~one frame per AU.
  Session sess;
  CerrSilencer hush;

  EncodedSegment whole;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 8,
                                 whole)) {
    return;
  }
  to_annexb_(whole);
  auto per_au = split_into_per_au_(whole);
  // Sanity: we got several AUs out (8 frames in => 8 AUs).
  EXPECT_TRUE(per_au.size() >= 4u);
  EXPECT_TRUE(per_au.size() <= 10u);

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SegmentSequenceSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->segs = per_au;
  src_u->allocate_oports(1);
  auto* src = static_cast<SegmentSequenceSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hwaccel", FlexData::make_string("none"));
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  // With per-AU Beats and no per-Beat flush, the decoder retains
  // SPS/PPS through P-frame Beats. Lower bound 4 mirrors the other
  // shape tests; the upper bound is the AU count.
  EXPECT_TRUE(got.size() >= 4u);
  EXPECT_TRUE(got.size() <= per_au.size());
  for (const auto& tb : got) {
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 240);
    EXPECT_TRUE(tb.shape[2] == 320);
  }
}

namespace {

// Find the index of the AU containing the first IDR (NAL type 5).
// Helper for the mid-GOP-startup test below.
std::size_t
first_idr_index_(const std::vector<EncodedSegment>& aus)
{
  for (std::size_t i = 0; i < aus.size(); ++i) {
    const auto& d = aus[i].data;
    // Cheap annex-B scan for a NAL type-5 byte after a start code.
    for (std::size_t p = 0; p + 4 < d.size(); ++p) {
      if (d[p] == 0 && d[p + 1] == 0
          && (d[p + 2] == 1 || (d[p + 2] == 0 && d[p + 3] == 1))) {
        const std::size_t nal = (d[p + 2] == 1) ? p + 3 : p + 4;
        if (nal < d.size() && (d[nal] & 0x1F) == 5) {
          return i;
        }
      }
    }
  }
  return aus.size();
}

}

TEST(video_to_rgb_stage, mid_gop_startup_drops_until_idr) {
  // Regression guard for the live-stream startup hang we hit at
  // 2880×1616 + VideoToolbox: a freshly-opened decoder cannot
  // consume a P-slice as its first packet (VT returns
  // AVERROR_UNKNOWN and libavcodec stays stuck thereafter, so even
  // a subsequent IDR fails). The stage's sync gate must drop
  // non-IDR AUs until the first IDR arrives, then sync the
  // decoder. We simulate the mid-GOP join by rotating the fixture
  // so all P-frames come before the IDR. With the gate the IDR
  // (now last in the sequence) still decodes -- one frame out.
  // Without the gate, send_packet on the leading P-slice errors,
  // leaves libavcodec broken, the IDR also fails, zero frames out.
  Session sess;
  CerrSilencer hush;

  EncodedSegment whole;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 8,
                                 whole)) {
    return;
  }
  to_annexb_(whole);
  auto per_au = split_into_per_au_(whole);
  EXPECT_TRUE(per_au.size() >= 2u);
  // First AU of an ultrafast-encoded clip is the IDR; rotate it
  // to the end so the sequence is [P, P, ..., P, IDR].
  const std::size_t idr_idx = first_idr_index_(per_au);
  EXPECT_TRUE(idr_idx < per_au.size());
  std::rotate(per_au.begin(),
              per_au.begin() + static_cast<ptrdiff_t>(idr_idx + 1),
              per_au.end());
  // Sanity: IDR is now the last entry.
  EXPECT_TRUE(first_idr_index_(per_au) == per_au.size() - 1);

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<SegmentSequenceSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->segs = per_au;
  src_u->allocate_oports(1);
  auto* src = static_cast<SegmentSequenceSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hwaccel", FlexData::make_string("none"));
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  // At least one frame must come out -- the IDR at the tail of
  // the sequence. Pre-fix this would be 0.
  EXPECT_TRUE(got.size() >= 1u);
  for (const auto& tb : got) {
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 240);
    EXPECT_TRUE(tb.shape[2] == 320);
  }
}

TEST(video_to_rgb_stage, output_dims_require_both_keys) {
  // output_width / output_height must be set together and be > 0.
  // Construction succeeds for any config; the problem is recorded in
  // config_error() and deferred to launch.
  Session sess;
  {
    FlexData cfg = FlexData::from_json(R"({"output_width": 224})");
    VideoToRgbStage s(&sess, "cvt", vector<InEdge>{}, std::move(cfg));
    EXPECT_FALSE(s.config_error().empty());
  }
  {
    FlexData cfg = FlexData::from_json(
        R"({"output_width": 0, "output_height": 224})");
    VideoToRgbStage s(&sess, "cvt", vector<InEdge>{}, std::move(cfg));
    EXPECT_FALSE(s.config_error().empty());
  }
}

TEST(video_to_rgb_stage, rescale_landscape_to_square_emits_correct_shape) {
  // 320×240 source decoded then center-cropped to 240×240 + bilinearly
  // rescaled to 224×224. The emitted TensorBeat must have shape
  // [3, 224, 224] and (for a uniform-grey source) the center pixel
  // must remain ~mid-grey post-rescale.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::from_json(
      R"({"output_dtype":"u8","output_width":224,"output_height":224})");
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!sink->collected().empty());
  for (const auto& tb : sink->collected()) {
    EXPECT_TRUE(tb.dtype == TensorBeat::DType::U8);
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 224);
    EXPECT_TRUE(tb.shape[2] == 224);
  }
  // Mid-grey input -> the entire rescaled output is mid-grey; spot-
  // check the center pixel.
  const TensorBeat& tb =
      sink->collected()[sink->collected().size() > 1 ? 1 : 0];
  auto contig = tb.materialize_contiguous_as<uint8_t>();
  const int H = 224, W = 224;
  const int cy = H / 2, cx = W / 2;
  const int r = contig[0 * H * W + cy * W + cx];
  const int g = contig[1 * H * W + cy * W + cx];
  const int b = contig[2 * H * W + cy * W + cx];
  EXPECT_TRUE(r > 128 - 12 && r < 128 + 12);
  EXPECT_TRUE(g > 128 - 12 && g < 128 + 12);
  EXPECT_TRUE(b > 128 - 12 && b < 128 + 12);
}

TEST(video_to_rgb_stage, rescale_landscape_to_portrait_emits_correct_shape) {
  // 320×240 -> 200×400. Output is taller than wide; src aspect is
  // wider, so the crop should be horizontal (drop columns), leaving
  // a 120×240 src region centered on column 100. The output is then
  // rescaled to 200×400 by the kernel (or sws on the CPU path).
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::from_json(
      R"({"output_dtype":"u8","output_width":200,"output_height":400})");
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!sink->collected().empty());
  for (const auto& tb : sink->collected()) {
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 400);
    EXPECT_TRUE(tb.shape[2] == 200);
  }
}

TEST(video_to_rgb_stage, sideband_timestamp_us_propagates_from_au) {
  // rtsp-capture stamps each AU Beat with start_utc = wall-clock at
  // arrival; video-to-rgb must surface that as a sideband object
  // `{"timestamp_us": <uint64>}` on every emitted TensorBeat so
  // downstream stages can correlate frames with their wall-clock
  // capture time.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }
  // Stamp a deterministic UTC point so we can read it back.
  const std::uint64_t kTs = 1'700'000'123'456ull;
  seg.start_utc = std::chrono::system_clock::time_point{
      std::chrono::microseconds{kTs}};

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  EXPECT_TRUE(!got.empty());
  for (const auto& tb : got) {
    EXPECT_TRUE(tb.sideband.is_object());
    auto obj = tb.sideband.as_object();
    EXPECT_TRUE(obj.contains("timestamp_us"));
    EXPECT_TRUE(obj.at("timestamp_us").get_uint() == kTs);
  }
}

TEST(video_to_rgb_stage, sideband_fps_propagates_from_segment) {
  // rtsp-capture carries the source stream's frame rate on each
  // EncodedSegment (fps_num/fps_den); video-to-rgb must forward it onto
  // every emitted TensorBeat's sideband so a downstream sink
  // (hls-broadcast) can adopt the original cadence.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }
  // 23.976 fps as a rational, the sort of value cameras report.
  seg.fps_num = 24000;
  seg.fps_den = 1001;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  EXPECT_TRUE(!got.empty());
  for (const auto& tb : got) {
    EXPECT_TRUE(tb.sideband.is_object());
    auto obj = tb.sideband.as_object();
    EXPECT_TRUE(obj.contains("fps_num"));
    EXPECT_TRUE(obj.contains("fps_den"));
    EXPECT_TRUE(obj.at("fps_num").get_uint() == 24000u);
    EXPECT_TRUE(obj.at("fps_den").get_uint() == 1001u);
  }
}

TEST(video_to_rgb_stage, rescale_strong_downsample_uses_bicubic) {
  // 320×240 source rescaled to 80×80. After centred crop to the
  // square 240×240, the resample factor is 3.0× (downsample),
  // exceeding the bicubic threshold (2×). The kernel switches to
  // Catmull-Rom 4×4 sampling on both Y and CbCr. For a uniform-
  // grey source the visible behaviour is the same as bilinear —
  // shape is correct, output stays mid-grey, no aliasing artefacts
  // — but this exercises the bicubic branch in CI.
  Session sess;
  CerrSilencer hush;

  EncodedSegment seg;
  if (!make_solid_color_segment_(sess, 128, 128, 128, 320, 240, 4,
                                 seg)) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<OneSegmentSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->seg = seg;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneSegmentSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::from_json(
      R"({"output_dtype":"u8","output_width":80,"output_height":80})");
  auto cvt_u = make_unique<VideoToRgbStage>(
      &sess, "cvt", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* cvt = static_cast<VideoToRgbStage*>(
      pl->insert_stage(std::move(cvt_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{cvt, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!sink->collected().empty());
  for (const auto& tb : sink->collected()) {
    EXPECT_TRUE(tb.dtype == TensorBeat::DType::U8);
    EXPECT_TRUE(tb.shape.size() == 3u);
    EXPECT_TRUE(tb.shape[0] == 3);
    EXPECT_TRUE(tb.shape[1] == 80);
    EXPECT_TRUE(tb.shape[2] == 80);
  }
  // Mid-grey input → centre pixel survives the bicubic round-trip.
  const TensorBeat& tb =
      sink->collected()[sink->collected().size() > 1 ? 1 : 0];
  auto contig = tb.materialize_contiguous_as<uint8_t>();
  const int H = 80, W = 80;
  const int cy = H / 2, cx = W / 2;
  const int r = contig[0 * H * W + cy * W + cx];
  const int g = contig[1 * H * W + cy * W + cx];
  const int b = contig[2 * H * W + cy * W + cx];
  EXPECT_TRUE(r > 128 - 12 && r < 128 + 12);
  EXPECT_TRUE(g > 128 - 12 && g < 128 + 12);
  EXPECT_TRUE(b > 128 - 12 && b < 128 + 12);
}
