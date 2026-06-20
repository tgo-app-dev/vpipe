#ifndef VIDEO_TOKENS_H
#define VIDEO_TOKENS_H

#include "common/beat-payload-intf.h"

#include <memory>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
#include <libavutil/samplefmt.h>
}

namespace vpipe {

// Stream contract on each video / audio EdgeBuffer between a
// VideoFileDecoderStage and a VideoFileEncoderStage:
//
//   1. The producer writes exactly one StreamParams header Beat
//      (VideoStreamParams on a video port, AudioStreamParams on an
//      audio port) before any frames.
//   2. The producer then writes any number of FrameRef Tokens (each
//      wraps a refcounted AVFrame).
//   3. EOS is signalled by closing the EdgeBuffer (RuntimeContext's
//      driver does this when the stage finishes).
//
// Receivers use Beat::try_get<T> to distinguish a header from a
// frame. Headers are immutable POD; frames are shared via
// shared_ptr<AVFrame> with a deleter that calls av_frame_free, so
// fanout copies are cheap.

struct VideoStreamParams {
  int        width      = 0;
  int        height     = 0;
  int        pix_fmt    = AV_PIX_FMT_NONE;   // AVPixelFormat as int
  AVRational time_base  = {0, 1};
  AVRational frame_rate = {0, 1};
};

struct AudioStreamParams {
  int             sample_rate = 0;
  int             sample_fmt  = AV_SAMPLE_FMT_NONE;
  AVChannelLayout ch_layout   {};            // ffmpeg 5.1+ value type
  AVRational      time_base   = {0, 1};
};

// Refcounted AVFrame handle. Construct with the deleter that calls
// av_frame_free via the appropriate function pointer:
//
//   auto sp = std::shared_ptr<AVFrame>(raw,
//     [api = &libs.avutil().api](AVFrame* f){ api->frame_free(&f); });
using FrameRef = std::shared_ptr<AVFrame>;

// Pipeline-edge payload form of the video-stream header. POD-sized;
// clone is trivial deep-copy.
class VideoStreamParamsPayload : public VideoStreamParams,
                                 public BeatPayloadIntf {
public:
  VideoStreamParamsPayload() = default;
  explicit
  VideoStreamParamsPayload(const VideoStreamParams& p)
    : VideoStreamParams(p)
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<VideoStreamParamsPayload>(
        static_cast<const VideoStreamParams&>(*this));
  }

  std::string
  describe() const override
  {
    std::string s = "VideoStreamParams ";
    s += std::to_string(width);
    s += "x";
    s += std::to_string(height);
    return s;
  }
};

// Pipeline-edge payload form of the audio-stream header. POD-sized;
// clone is trivial deep-copy.
class AudioStreamParamsPayload : public AudioStreamParams,
                                 public BeatPayloadIntf {
public:
  AudioStreamParamsPayload() = default;
  explicit
  AudioStreamParamsPayload(const AudioStreamParams& p)
    : AudioStreamParams(p)
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<AudioStreamParamsPayload>(
        static_cast<const AudioStreamParams&>(*this));
  }

  std::string
  describe() const override
  {
    std::string s = "AudioStreamParams ";
    s += std::to_string(sample_rate);
    s += "Hz";
    return s;
  }
};

// Pipeline-edge payload wrapping a FrameRef (shared_ptr<AVFrame>).
// Clone is a refcount bump on the shared_ptr -- O(1).
class FrameRefPayload : public BeatPayloadIntf {
public:
  FrameRef ref;

  FrameRefPayload() = default;
  explicit
  FrameRefPayload(FrameRef r) noexcept
    : ref(std::move(r))
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<FrameRefPayload>(ref);
  }

  std::string
  describe() const override
  {
    return ref ? "FrameRef" : "FrameRef(empty)";
  }
};

}

#endif
