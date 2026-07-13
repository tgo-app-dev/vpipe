#ifndef LOAD_VIDEO_STAGE_H
#define LOAD_VIDEO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "common/ffmpeg-libraries.h"
#include "stages/audio-video/video-tokens.h"
#include <string>

namespace vpipe {

// Reads a video file (or network URL: rtsp://, http://, ...), demuxes,
// decodes, and emits decoded frames downstream. Source stage; 0 input
// ports, 1 or 2 output ports depending on `enable_video` /
// `enable_audio`.
//
// Per port, the contract is the StreamParams header followed by N
// FrameRef beats followed by EOS (driver closes the buffer when this
// stage signals done). Receivers distinguish header vs frame via
// Beat::try_get<T>.
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
  Job drain     (RuntimeContext& ctx) override;

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
  void open_codec_(int stream_idx, AVCodecContext** out);
  std::string av_err_(int rc) const;

  // Drains the receive_frame loop after a send_packet. `pkt == nullptr`
  // means flush. Yields each decoded frame downstream on `port`.
  Job drain_codec_(RuntimeContext& ctx, AVCodecContext* cctx,
                   unsigned port, AVPacket* pkt);

  VideoStreamParams make_video_params_() const noexcept;
  AudioStreamParams make_audio_params_() const noexcept;

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
  AVCodecContext*  _vctx          = nullptr;
  AVCodecContext*  _actx          = nullptr;
  AVPacket*        _pkt           = nullptr;

  bool _eof = false;
};

}

#endif
