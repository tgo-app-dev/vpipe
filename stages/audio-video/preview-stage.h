#ifndef VIDEO_PREVIEW_STAGE_H
#define VIDEO_PREVIEW_STAGE_H

#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "common/preview-channel.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

namespace vpipe {

struct TensorBeat;

// Sink stage that broadcasts a live video preview to the web-ui over a
// low-latency custom protocol (fragmented MP4 over WebSocket -> Media
// Source Extensions). MSE is not gated to secure contexts, so this plays
// over plain-HTTP LAN origins with no HTTPS required.
//
// The stage is SELF-CLOCKED: it runs a cadence (default 25 fps) and always
// produces output -- a black frame before any input, then the latest
// received frame. Faster-than-cadence input is sampled (only the newest
// frame per tick is shown); slower/absent input repeats the last frame
// (so a single still image plays as continuous video). When an input
// frame's sideband carries fps, the cadence adopts it (pass-through at the
// source rate); otherwise the 25 fps image cadence applies.
//
// Video is encoded with the hardware H.264 encoder (h264_videotoolbox) at
// the input's native resolution (the pre-input black frame uses a small
// default resolution; the stream re-initializes once when the first real
// frame establishes the native size). Audio (optional) is passed through
// as planar float and rendered by WebAudio in the browser.
//
// iport 0: video RGB TensorBeat ([3,H,W], F32 or U8). The first frame sets
//          the output resolution; later size changes are dropped.
// iport 1: (optional) audio PCM TensorBeat -- F32 rank-1 [n] (mono) or
//          rank-2 [channels, n]. sideband.sample_rate honoured.
// oports : none (sink).
//
// Config (FlexData object):
//   bitrate          (int bps, default 2'000'000)
//   input_normalized (bool, default true) -- F32 input in [0,1] vs [0,255]
//   title            (string, default "") -- optional picker label; empty
//                    = the stage id.
class PreviewStage final
  : public TypedStage<PreviewStage>
  , public PreviewSource
{
public:
  static constexpr const char* kTypeName = "preview";

  PreviewStage(const SessionContextIntf* session,
               std::string               id,
               std::vector<InEdge>       iports,
               FlexData                  config);
  ~PreviewStage() override;

  Job process(RuntimeContext& ctx) override;
  Job drain  (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  std::shared_ptr<PreviewChannel> preview_channel() const override
  { return _channel; }

  // Test-only inspectors.
  bool encoder_initialized() const noexcept { return _enc != nullptr; }
  const std::string& codec_string() const noexcept { return _codec_string; }
  int  cadence_fps() const noexcept
  { return _cadence_den ? _cadence_num / _cadence_den : 0; }
  int  output_width()  const noexcept { return _out_w; }
  int  output_height() const noexcept { return _out_h; }

private:
  void resolve_roles_(RuntimeContext& ctx);

  // Build the encoder + fMP4 muxer + output frame at WxH, emit the init
  // segment. Tears down any prior pipeline first (used for the one-time
  // native-resolution re-init). Sets _fatal on failure.
  bool build_pipeline_(int W, int H);
  void teardown_media_();   // free ffmpeg objects (not the channel)
  void teardown_();         // flush + close channel + teardown_media_

  // Derive the WebCodecs/MSE codec string ("avc1.PPCCLL") from extradata.
  void parse_extradata_();
  void emit_config_();

  // Ingest one input video frame: adopt native res on the first frame,
  // adopt fps from the sideband, and convert RGB -> the output frame.
  void handle_video_frame_(const TensorBeat& tb);
  void adopt_fps_(const TensorBeat& tb);
  bool ensure_sws_(int W, int H);
  void convert_to_frame_(const TensorBeat& tb);   // RGB [3,H,W] -> _frame
  void fill_black_();

  void handle_audio_(const TensorBeat& tb);

  // Encode + mux the current output frame (one cadence tick); flush a
  // fragment on the fragment boundary.
  void encode_tick_();

  std::chrono::steady_clock::duration cadence_period_() const;

  std::string av_err_(int rc) const;

  // AVIO write callback: append muxer output to _mux_buf.
  static int write_cb_(void* opaque, const std::uint8_t* buf, int size);

  const FFmpegLibraries* _libs = nullptr;

  // Config.
  std::int64_t _bitrate{};
  bool         _input_normalized{};
  std::string  _title;

  // Cadence (default 25 fps; adopts an input frame's sideband fps).
  int _cadence_num = 25;
  int _cadence_den = 1;

  // Output resolution (default black size until the first frame adopts
  // the native size).
  int _out_w = 0;
  int _out_h = 0;

  // FFmpeg encode + fMP4 mux state.
  AVCodecContext*  _enc      = nullptr;
  AVPacket*        _pkt      = nullptr;
  SwsContext*      _sws      = nullptr;
  AVFrame*         _gbrp     = nullptr;   // input-res GBRP scratch
  AVFrame*         _frame    = nullptr;   // output YUV (black or latest)
  AVFormatContext* _mux      = nullptr;
  AVStream*        _vstream  = nullptr;
  AVIOContext*     _avio     = nullptr;
  std::uint8_t*    _avio_buf = nullptr;
  std::vector<std::uint8_t> _mux_buf;     // AVIO capture (one fragment)
  std::vector<std::uint8_t> _init;        // fMP4 init segment
  std::string      _codec_string;         // "avc1.PPCCLL"

  // Cadence / fragment state.
  bool         _have_frame        = false;  // a real input frame arrived
  bool         _first_encoded     = true;   // next encode forces IDR
  bool         _force_next_key    = false;  // start next fragment on a key
  int          _frames_since_flush = 0;
  int          _frag_len          = 12;     // frames per fragment
  // Constant-cadence encode timestamps (90 kHz): a uniform PTS + duration
  // per frame, so the muxer's DTS stays monotonic + self-describing.
  std::int64_t _enc_pts           = 0;
  std::int64_t _frame_dur_90k     = 3600;   // 90000 / 25 fps

  std::chrono::steady_clock::time_point _next_tick;
  bool _clock_started = false;

  // Audio passthrough.
  bool _audio_seen = false;
  int  _audio_rate = 0;
  int  _audio_ch   = 0;

  // Roles + config readiness.
  bool _roles_resolved = false;
  bool _want_video     = false;
  bool _want_audio     = false;
  bool _video_cfg_ready = false;
  bool _audio_cfg_ready = false;

  bool _fatal = false;
  bool _torn  = false;
  std::uint64_t _frames_in = 0;

  std::shared_ptr<PreviewChannel> _channel;
};

}

#endif
