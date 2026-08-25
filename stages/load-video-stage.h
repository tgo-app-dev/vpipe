#ifndef LOAD_VIDEO_STAGE_H
#define LOAD_VIDEO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "common/encoded-segment.h"
#include "common/ffmpeg-libraries.h"
#include "stages/audio-video/video-tokens.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Reads a video file (or network URL: rtsp://, http://, ...), demuxes
// it, and emits its ENCODED packets downstream. Source stage; 0 input
// ports, 1 or 2 output ports depending on `enable_video` /
// `enable_audio`.
//
// One EncodedSegment per packet, the same contract rtsp-capture
// publishes -- so the file and the camera are interchangeable sources
// for everything downstream, and `video-to-rgb` / `audio-to-pcm` own the
// decode for both.
//
// It decoded, once, and emitted FrameRefs. Nothing consumed them: the
// only FrameRef consumer in the tree is save-video, which is fed by
// rgb-to-video, so a file could reach no stage that works in tensors.
// Decoding here also duplicated video-to-rgb's decoder without its
// hardware path, its format conversion, or its cadence sideband. Moving
// the decode downstream is what gives a file the whole existing chain --
// resample, decimate, stack, encode -- instead of a private one.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   input_url            (string, required)
//   format               (string, default "")  -- forced demuxer; "" auto
//   enable_video         (bool,   default true)
//   enable_audio         (bool,   default true)
//   video_stream_index   (int,    default -1)  -- -1 = first video
//   audio_stream_index   (int,    default -1)
//   options              (object<string,string>) -- av_dict for open
//   read_timeout_ms      (int,    default 0)   -- network timeout
class LoadVideoStage final
  : public TypedStage<LoadVideoStage>
{
public:
  static constexpr const char* kTypeName = "load-video";

  LoadVideoStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);

  ~LoadVideoStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Decoder is a source (no iports). Its video and audio oports
  // advance on independent clocks -- video on the video stream's
  // frame rate, audio on the audio stream's sample / frame rate --
  // so each oport reports its own clock group. The port indices are
  // assigned dynamically (video/audio optional), so this stays an
  // explicit override rather than reading static groups from kSpec.
  unsigned
  oport_clock_group(unsigned p) const noexcept override
  {
    if (_video_port >= 0 &&
        p == static_cast<unsigned>(_video_port)) {
      return 0;
    }
    return 1;
  }

  // Test-only accessors.
  int video_port() const noexcept { return _video_port; }
  int audio_port() const noexcept { return _audio_port; }

private:
  void open_input_();
  int  pick_stream_(int media_type, int requested) const noexcept;
  // Cache the per-stream metadata every emitted segment repeats:
  // codec_id, extradata and the geometry or rate. Read once at open,
  // because codecpar lives in _fctx and a segment must be readable
  // after this stage is gone.
  void cache_stream_(int stream_idx, bool video);
  std::string av_err_(int rc) const;

  // Build one segment from the packet currently in `_pkt`.
  std::unique_ptr<EncodedSegmentPayload> segment_(bool video);

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _input_url;
  std::string _format;
  bool        _enable_video{};
  bool        _enable_audio{};
  int         _video_stream_index{};
  int         _audio_stream_index{};
  FlexData    _open_options;
  int         _read_timeout_ms{};

  // Derived port indices: -1 = disabled.
  int _video_port = -1;
  int _audio_port = -1;

  // FFmpeg API tables. Owned by the Session and shared across every
  // stage that needs them; bound here in the ctor's member-init list
  // and stable for the stage's lifetime. Non-null in a fully-
  // constructed stage; ctor calls session()->error if the session
  // can't supply one.
  const FFmpegLibraries* _libs;
  AVFormatContext*       _fctx          = nullptr;
  int              _v_stream_idx  = -1;
  int              _a_stream_idx  = -1;
  AVPacket*        _pkt           = nullptr;

  // Per-stream metadata, cached at open (see cache_stream_).
  struct StreamMeta {
    unsigned codec_id = 0;
    unsigned width = 0, height = 0;
    unsigned fps_num = 0, fps_den = 0;
    unsigned sample_rate = 0, channels = 0;
    std::vector<std::uint8_t> extradata;
    AVRational time_base { 0, 1 };
    // MEDIA time of the next packet, in microseconds from the start of
    // the file. A packet with no pts repeats the previous one's end,
    // which keeps the stream monotonic where a zero would jump it back
    // to the start mid-clip.
    std::int64_t last_us = 0;
  };
  StreamMeta _vmeta, _ameta;

  bool _eof = false;
  std::uint64_t _v_packets = 0, _a_packets = 0;
};

}

#endif
