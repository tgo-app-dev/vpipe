#ifndef SAVE_VIDEO_STAGE_H
#define SAVE_VIDEO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "common/ffmpeg-libraries.h"
#include "stages/audio-video/video-tokens.h"
#include <string>

namespace vpipe {

// Receives StreamParams headers + FrameRef beats on its input ports
// and encodes them with H.264 (libx264 by default) and AAC (default
// audio encoder), muxing into a container written to disk.
//
// Sink stage; 0 output ports. Input port count is 1 or 2 depending on
// `enable_video` / `enable_audio`. With both enabled, port 0 is video
// and port 1 is audio. With only one enabled, that one is port 0.
//
// Configuration (FlexData object):
//   output_url       (string, required)
//   format           (string, default "")     -- container; "" = inferred
//   enable_video     (bool,   default true)
//   enable_audio     (bool,   default true)
//   video.codec      (string, default "libx264")
//   video.bitrate    (int,    default 2000000)
//   video.preset     (string, default "medium")
//   video.crf        (int,    optional)       -- when present overrides bitrate
//   video.gop_size   (int,    default 60)
//   video.options    (object<string,string>)  -- extra encoder options
//   audio.codec      (string, default "aac")
//   audio.bitrate    (int,    default 128000)
//   audio.options    (object<string,string>)
//   muxer_options    (object<string,string>)  -- e.g. {"movflags":"+faststart"}
//
// The encoder inherits pixel format / sample format / channel layout
// from the incoming StreamParams headers; mismatches between input
// format and what the encoder accepts surface as clean error logs at
// avcodec_open2 time (no swscale / swresample yet).
class SaveVideoStage final
  : public TypedStage<SaveVideoStage>
{
public:
  static constexpr const char* kTypeName = "save-video";

  SaveVideoStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);

  ~SaveVideoStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Encoder is a sink (no oports). When both video and audio are
  // enabled the iports run on independent clocks (video frame
  // rate vs audio sample rate); declare distinct clock groups so
  // upstream domains stay separate. The port indices are assigned
  // dynamically (video/audio optional), so this stays an explicit
  // override rather than reading static groups from kSpec.
  unsigned
  iport_clock_group(unsigned p) const noexcept override
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
  void ensure_output_format_();
  void init_video_encoder_(const VideoStreamParams& p);
  void init_audio_encoder_(const AudioStreamParams& p);
  void open_output_and_write_header_();
  bool ready_to_write_header_() const noexcept;
  void encode_and_mux_(unsigned port, AVFrame* frame);
  void drain_encoder_(AVCodecContext* enc, AVStream* st);
  void finalize_();
  std::string av_err_(int rc) const;

  // Config attributes. Flat-key defaults live in kSpec.attrs (read via
  // attr_*); the nested video.* / audio.* sub-object defaults have no
  // flat ConfigKey representation and are seeded once at the top of the
  // constructor. Declarations carry no non-zero default.
  std::string _output_url;
  std::string _format;
  bool        _enable_video{};
  bool        _enable_audio{};
  std::string _video_codec;
  int64_t     _video_bitrate{};
  std::string _video_preset;
  bool        _video_has_crf{};
  int         _video_crf{};
  int         _video_gop_size{};
  FlexData    _video_options;
  std::string _audio_codec;
  int64_t     _audio_bitrate{};
  FlexData    _audio_options;
  FlexData    _muxer_options;

  // Derived port indices: -1 = disabled.
  int _video_port = -1;
  int _audio_port = -1;

  // FFmpeg API tables. Owned by the Session and shared across every
  // stage that needs them; bound here in the ctor's member-init list.
  // Non-null in a fully-constructed stage; ctor calls
  // session()->error if the session can't supply one.
  const FFmpegLibraries* _libs;
  AVFormatContext*       _ofctx           = nullptr;
  AVStream*        _vstream         = nullptr;
  AVStream*        _astream         = nullptr;
  AVCodecContext*  _venc            = nullptr;
  AVCodecContext*  _aenc            = nullptr;
  AVPacket*        _enc_pkt         = nullptr;

  bool _video_initialized = false;
  bool _audio_initialized = false;
  bool _video_eos         = false;
  bool _audio_eos         = false;
  bool _header_written    = false;
  bool _finalized         = false;

  int _next_port = 0;
};

}

#endif
