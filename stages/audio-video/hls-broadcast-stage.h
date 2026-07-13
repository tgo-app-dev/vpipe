#ifndef VIDEO_HLS_BROADCAST_STAGE_H
#define VIDEO_HLS_BROADCAST_STAGE_H

#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "common/static-file-server.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace vpipe {

struct TensorBeat;

// Sink stage that consumes RGB TensorBeats and broadcasts them as an
// HLS stream:
//   * libavcodec encodes the frames (default h264_videotoolbox on
//     Apple Silicon; libx264 elsewhere via the `codec` config).
//   * libavformat's `hls` muxer writes a rolling .m3u8 playlist plus
//     mpegts (.ts) segments. Both the playlist and the segments are
//     intercepted via the muxer's io_open / io_close2 callbacks and
//     captured into an in-memory blob registry — no filesystem
//     interaction.
//   * An in-process StaticFileServer makes that registry available
//     over HTTP so clients can fetch the playlist and segments
//     directly without a separate web server.
//
// iport 0: video RGB TensorBeat ([3, H, W], F32 or U8). Resolution
//          must be stable for the stage's lifetime; mid-stream resizes
//          warn + drop.
// iport 1: (optional) audio PCM TensorBeat -- F32, rank-1 [n_samples]
//          (mono) or rank-2 [channels, n_samples]. sideband.sample_rate
//          gives the source rate (else the encoder rate is assumed).
// oports : none (sink).
//
// Video/audio modes -- iports are positional and optional, with STRICT
// roles: iport 0 is always video, iport 1 is always audio. Which
// streams the output carries is decided purely by which iports are
// connected:
//   * video only  -- wire video to iport 0 (audio port left unwired).
//   * video+audio -- wire video to iport 0, audio to iport 1. The two
//                    streams are muxed into one HLS output.
//   * audio only  -- leave iport 0 (video) DISCONNECTED and wire the
//                    PCM producer to iport 1.
// A beat that arrives on the wrong-role port (e.g. PCM audio wired to
// the video port 0, or a frame on the audio port 1) is dropped with a
// warning -- audio must go to iport 1, video to iport 0.
//
// A/V synchronisation: when a beat carries a `timestamp_us` sideband
// (video-to-rgb / audio-to-pcm set it from capture time), both streams'
// PTS derive from one shared UTC epoch, so audio and video stay locked.
// When the audio has NO timestamp (raw PCM, e.g. text-to-speech), the
// stage buffers `audio_buffer_seconds` of audio before it begins muxing
// and then advances the audio clock by sample count -- an initial
// jitter cushion that hides a generator lagging behind real time. That
// cushion is a ONE-TIME startup delay: past it the stage streams
// continuously and only "pauses" when the FIFO is empty, at which point
// it emits silence (chasing the video/wall clock) so the audio track
// never gaps.
//
// Why HLS instead of RTSP: FFmpeg's RTSP muxer is a push client only
// (it connects to an existing RTSP server and pushes via ANNOUNCE).
// It does not implement a listening server, so we cannot host an
// RTSP service in-process with libavformat. HLS is the simplest
// in-tree-only alternative: pure file I/O plus a static HTTP server,
// no RTSP/RTP packetization, and any HLS-capable client (VLC,
// ffplay, browsers, Safari) can consume it.
//
// Latency: standard HLS is high-latency (typically several seconds);
// drop hls_time + playlist_max_size for lower latency at the cost of
// less buffering room.
//
// Config (FlexData object):
//   playlist_name       (string, default "stream.m3u8")
//   segment_duration    (real seconds, default 2.0) -- hls_time. May
//                        be fractional (e.g. 0.5) for low-latency
//                        operation. Pair with `gop_size` so an IDR
//                        lands on each segment boundary; otherwise
//                        the muxer rolls late at the next keyframe.
//                        Example low-latency tuning at 30 fps:
//                        segment_duration=0.5, gop_size=15.
//   live_start_offset   (real seconds, default 0 = off) -- when > 0,
//                        a `#EXT-X-START:TIME-OFFSET=-N,PRECISE=YES`
//                        tag is injected into every playlist so
//                        compliant players anchor playback N
//                        seconds behind the live edge instead of
//                        their default 3 × target_duration. Set to
//                        ~2× segment_duration for a tight live
//                        window. Players that ignore the tag
//                        (e.g. MPC-HC) fall back to their default
//                        seek-from-end heuristic.
//   playlist_max_size   (int, default 5)           -- hls_list_size
//                                                    0 == unbounded
//   hls_flags           (string, default "omit_endlist")
//                        -- a 'delete_segments' token is ignored:
//                        segments live in memory and are evicted
//                        internally, so the muxer's on-disk unlink
//                        would only fail.
//   codec               (string, default "h264_videotoolbox") -- on
//                       Apple-Silicon hosts this engages the platform
//                       hardware encoder. Set to "libx264" (or another
//                       software encoder) on hosts without VideoToolbox.
//   bitrate             (int bps, default 2'000'000)
//   gop_size            (int, default 60)
//   preset              (string, default "veryfast")  -- libx264 only
//   tune                (string, default "zerolatency") -- libx264 only
//   fps_num             (int, default 0 = auto) -- encode frame rate
//   fps_den             (int, default 0 = auto)    numerator/denominator.
//                       When fps_num is 0 the stage adopts the rate
//                       carried on the first input frame's sideband
//                       (fps_num/fps_den, set by video-to-rgb from the
//                       capture stream), falling back to 30/1 when the
//                       sideband has none. A non-zero fps_num pins the
//                       rate (fps_den defaults to 1) and ignores the
//                       sideband.
//   input_normalized    (bool, default true)
//   realtime            (bool, default true)  -- when true, PTS
//                       advances at wall-clock rate (so HLS
//                       segments close every hls_time seconds
//                       regardless of how fast frames arrive) and
//                       a keyframe is forced every
//                       segment_duration wall-clock seconds. When
//                       false, PTS = frame_index in the
//                       fps_num/fps_den timebase (useful for
//                       deterministic tests where frames flow in
//                       a burst).
//   serve_http          (bool, default true)
//   bind_address        (string, default "" = auto) -- the HTTP
//                       server's bind interface. Empty selects, in
//                       order: the address the web-ui is served on
//                       (when this session is driven by the web-ui, so
//                       the stream lands on the same interface as the
//                       UI), else this machine's LAN address (en0) so
//                       other devices can play it, else "0.0.0.0". Set
//                       explicitly (e.g. "127.0.0.1" or "0.0.0.0") to
//                       override.
//   port                (int, default 8080)
//   audio_codec         (string, default "aac") -- audio encoder name
//                       for the (optional) audio stream. AAC is the
//                       HLS-standard codec; broadly playable.
//   audio_bitrate       (int bps, default 128000)
//   audio_sample_rate   (int, default 0 = auto) -- the audio encoder's
//                       output sample rate. 0 selects 48000. Each PCM
//                       beat's own sideband.sample_rate is the swr INPUT
//                       rate (resampled to this), so mismatched sources
//                       still mux cleanly.
//   audio_channels      (int, default 1) -- audio encoder output channel
//                       count. Mono PCM is up/down-mixed to this via
//                       swresample.
//   audio_buffer_seconds (real, default 0.5) -- initial jitter buffer,
//                       in seconds, applied ONLY to timestamp-less audio
//                       (raw PCM with no sideband.timestamp_us). The
//                       stage accumulates this much audio ONCE before it
//                       starts muxing, so a generator that lags behind
//                       real time doesn't produce choppy playback. This
//                       is a one-time startup delay: after it, the stage
//                       drains the FIFO to empty and never pauses the
//                       audio track again -- when the FIFO runs dry it
//                       streams silence to keep the clock advancing
//                       rather than stalling. 0 disables the cushion
//                       (lowest latency). Ignored when the audio carries
//                       timestamps (those pace the stream directly).
//   log_input_stats_every (int, default 0) -- when > 0, log
//                         min/mean/max of every Nth input
//                         TensorBeat at INFO level. Useful for
//                         triaging "all black" / wrong-range
//                         issues — confirms whether the data
//                         reaching this stage is actually full-
//                         range float in [0, 1] (or whatever the
//                         producer is emitting).
class HlsBroadcastStage final : public TypedStage<HlsBroadcastStage>
{
public:
  static constexpr const char* kTypeName = "hls-broadcast";

  HlsBroadcastStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);
  ~HlsBroadcastStage() override;

  Job process(RuntimeContext& ctx) override;
  Job drain  (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Up to 2 iports (video clock group 0, audio clock group 1), 0
  // oports. The stage multiplexes the two inputs with read_any, so no
  // clock-group override is needed.

  // Test-only inspectors.
  const std::string& playlist_url()  const noexcept { return _playlist_url; }
  const std::string& playlist_name() const noexcept { return _playlist_name; }
  double             segment_duration()  const noexcept
  { return _segment_duration; }
  double             live_start_offset() const noexcept
  { return _live_start_offset; }
  bool               encoder_initialized() const noexcept
  { return _enc != nullptr; }
  // Audio inspectors (test-only). audio_encoder_initialized() is true
  // once the AAC encoder + swresample are built; audio_stream_present()
  // is true once an audio stream has been added to the muxer output.
  bool               audio_encoder_initialized() const noexcept
  { return _aenc != nullptr; }
  bool               audio_stream_present() const noexcept
  { return _astream != nullptr; }
  // Latched "a video / audio stream was added to the muxer output" --
  // unlike encoder_initialized() / audio_stream_present() these survive
  // teardown, so a test can assert them after the pipeline stops.
  bool               video_muxed() const noexcept { return _video_muxed; }
  bool               audio_muxed() const noexcept { return _audio_muxed; }
  bool               server_running() const noexcept
  { return _http && _http->running(); }
  int                http_port() const noexcept
  { return _http ? _http->bound_port() : 0; }
  // Encode frame rate. 0/0 until the first frame resolves the "auto"
  // default (config value, else input sideband, else 30/1); test-only.
  int                fps_num() const noexcept { return _fps_num; }
  int                fps_den() const noexcept { return _fps_den; }
  // Resolved bind address (after the "auto" default has been resolved
  // to the web-ui / en0 / 0.0.0.0 fallback in the constructor).
  const std::string& bind_address() const noexcept
  { return _bind_address; }
  const InMemoryBlobStore& blobs() const noexcept { return _blobs; }

  // Internal: FFmpeg io callbacks (public because they need to be
  // pointed-at by C function pointers; not part of the user API).
  int io_open_ (AVIOContext** pb, const std::string& url);
  int io_close_(AVIOContext*  pb);

private:
  struct InMemSink;

  // Resolve the "auto" fps default (fps_num == 0 in config) from the
  // first frame's sideband, else 30/1. No-op semantics live in the .cc;
  // only called when config didn't pin fps.
  void resolve_default_fps_(const TensorBeat& tb);

  // Decide which streams the muxer will carry from iport connectedness.
  // Port roles are strict and positional: iport 0 = video, iport 1 =
  // audio. Either may be left disconnected (audio-only leaves iport 0
  // unwired); a beat that arrives on the wrong-role port is dropped.
  void resolve_roles_(RuntimeContext& ctx);

  // Latch the shared UTC epoch from the first ts-carrying beat.
  void note_epoch_(uint64_t ts_us) noexcept
  {
    if (!_av_epoch_set) { _av_epoch_us = ts_us; _av_epoch_set = true; }
  }

  bool ensure_encoder_   (int H, int W);
  bool ensure_sws_       (int H, int W);
  // Build the AAC encoder + reusable input frame/packet. swresample is
  // (re)built lazily per input (rate, channels) in push_audio_samples_.
  bool ensure_audio_encoder_();
  // Open the muxer once every wanted encoder is ready (video needs the
  // first frame's W/H; audio needs only ensure_audio_encoder_). Adds the
  // video and/or audio stream, writes the header. Idempotent.
  bool maybe_open_muxer_ ();
  bool open_output_      ();

  // Encode + mux one video frame (builds the video encoder / opens the
  // muxer on the first frame). No-op-drops a resolution change.
  void handle_video_     (const TensorBeat& tb);
  AVFrame* tensor_to_yuv_(const TensorBeat& tb);
  int  send_and_mux_     (AVFrame* yuv);
  int  drain_packets_    ();

  // Start the in-process HTTP server (once, if serve_http). Called from
  // maybe_open_muxer_ so the server comes up as soon as output exists.
  void start_http_       ();

  // Audio path. handle_audio_ resamples the PCM into the encoder FIFO;
  // pump_audio_ drains full frames through the encoder into the muxer
  // once the muxer is open and the jitter buffer (ts-less audio) has
  // filled. flush=true also emits the final zero-padded partial frame.
  void handle_audio_        (const TensorBeat& tb);
  // (Re)build swresample for a new (input rate, input channels) pair.
  bool ensure_swr_          (int in_rate, int in_ch);
  void push_audio_samples_  (const TensorBeat& tb);
  void pump_audio_          (bool flush);
  int  drain_audio_packets_ ();
  // True when an audio-only realtime stream is mid-stall: the PCM
  // producer is wired but idle (no backlog, not yet at EOS) so no beat
  // will wake read_any(). process() then self-clocks a silence pump on
  // the pool timer instead of freezing the audio track. Only audio-only
  // streams need this -- a video+audio stream is already pumped at video
  // cadence, and a closed input ends the stage rather than stalling.
  bool audio_stall_silence_due_(RuntimeContext& ctx) const noexcept;
  // Audio-only realtime: open the muxer + audio encoder and start the audio
  // clock at startup (pts 0, no jitter wait) so a silent track broadcasts
  // before the first PCM beat. Real audio later resumes from the advanced pts.
  // No-op unless prime_silence + audio-only + realtime + not ts-paced.
  void prime_audio_silence_(RuntimeContext& ctx);
  // Planar-float FIFO (one plane per encoder channel), at the encoder's
  // sample rate. Buffers swr output until a full frame_size frame is
  // available for the AAC encoder (which has a fixed frame size).
  void   afifo_push_        (const uint8_t* const* planes, int n);
  int    afifo_filled_      () const;
  bool   afifo_pull_frame_  (uint8_t* const* dst);
  int    afifo_pull_padded_ (uint8_t* const* dst);

  void close_output_     ();
  void teardown_         ();

  std::string av_err_(int rc) const;

  const FFmpegLibraries* _libs = nullptr;

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _playlist_name;
  double      _segment_duration{};
  double      _live_start_offset{};   // 0 = no EXT-X-START tag
  int         _playlist_max_size{};
  std::string _hls_flags;
  std::string _codec_name;
  int64_t     _bitrate{};
  int         _gop_size{};
  std::string _preset;
  std::string _tune;
  int         _fps_num{};
  int         _fps_den{};
  // True when fps was pinned in config (fps_num > 0); otherwise the
  // rate is resolved from the first frame's sideband on the fly.
  bool        _fps_from_config{};
  bool        _input_normalized{};
  bool        _realtime{};
  int         _log_input_stats_every{};
  bool        _serve_http{};
  std::string _bind_address;
  int         _http_port{};
  // Audio config.
  std::string _audio_codec_name;
  int64_t     _audio_bitrate{};
  int         _audio_sample_rate{};   // encoder out rate (0 -> 48000)
  int         _audio_channels{};      // encoder out channels
  double      _audio_buffer_seconds{};
  bool        _prime_silence{};       // audio-only: go live w/ silence at start

  // Derived
  std::string _playlist_url;      // http://<bind>:<port>/<playlist_name>

  // FFmpeg state (lazily initialized on first frame).
  AVCodecContext*  _enc           = nullptr;
  AVPacket*        _pkt           = nullptr;
  AVFormatContext* _ofctx         = nullptr;
  AVStream*        _vstream       = nullptr;
  SwsContext*      _sws           = nullptr;
  AVFrame*         _gbrp_scratch  = nullptr;
  int              _enc_w         = 0;
  int              _enc_h         = 0;
  int64_t          _pts           = 0;       // used when !_realtime
  int64_t          _last_pts      = -1;      // monotonicity guard
  bool             _header_written = false;
  bool             _trailer_written = false;

  // Wall-clock anchor for _realtime mode + forced-keyframe pacing.
  std::chrono::steady_clock::time_point _epoch;
  std::chrono::steady_clock::time_point _last_kf_time;
  bool                                  _clock_started = false;

  // --- Audio state (lazily initialized on the first audio beat / on
  //     muxer open when audio is wanted). ---
  AVCodecContext*  _aenc          = nullptr;
  AVStream*        _astream       = nullptr;
  AVPacket*        _apkt          = nullptr;
  SwrContext*      _swr           = nullptr;
  AVFrame*         _aframe        = nullptr;   // encoder input (frame_size)
  int              _aenc_frame_size = 0;
  int              _swr_in_rate   = 0;         // current swr input rate
  int              _swr_in_ch     = 0;         // current swr input channels
  // Planar-float FIFO at the encoder rate: one plane per enc channel.
  std::vector<std::vector<float>>       _afifo;
  int64_t          _audio_pts     = 0;         // next frame pts, enc samples
  bool             _audio_started = false;     // jitter buffer released
  bool             _audio_seen    = false;     // first audio beat handled
  bool             _audio_ts_mode = false;     // audio carries timestamp_us
  int64_t          _audio_anchor_us = 0;       // ts-mode start media offset
  // Wall-clock anchor latched when the no-TC cushion is released; drives
  // the silence-fill target for audio-only streams (video-locked streams
  // chase _video_media_us instead). See pump_audio_.
  std::chrono::steady_clock::time_point _audio_epoch;

  // Set when a build step fails unrecoverably; process() then signals
  // done so the runtime tears the stage down instead of spinning.
  bool             _fatal         = false;

  // Role / setup. Determined on the first beat.
  bool             _roles_resolved = false;
  bool             _want_video      = false;
  bool             _want_audio      = false;

  // Shared UTC epoch for A/V timestamp sync. Set from the first beat
  // that carries a `timestamp_us` sideband (video or audio); all
  // ts-based PTS are measured from it.
  bool             _av_epoch_set   = false;
  uint64_t         _av_epoch_us    = 0;
  // Latched at muxer open (see video_muxed()/audio_muxed()).
  bool             _video_muxed    = false;
  bool             _audio_muxed    = false;
  // Most recent video media time (microseconds from epoch), used to
  // anchor the start of timestamp-less audio to the current video edge.
  int64_t          _video_media_us = 0;

  // Diagnostic counter.
  uint64_t                              _frames_in = 0;

  // In-memory backing for the HLS muxer. Each io_open() callback
  // produces an InMemSink that buffers writes until io_close, at
  // which point the bytes are atomically published to `_blobs`
  // under the HTTP path "/<basename>". `_segment_paths` tracks the
  // FIFO insertion order so segments past `_playlist_max_size` can
  // be evicted from the registry (mirroring on-disk
  // `delete_segments` behaviour without touching the filesystem).
  InMemoryBlobStore                          _blobs;
  std::unordered_map<AVIOContext*,
                     std::unique_ptr<InMemSink>> _open_sinks;
  std::deque<std::string>                    _segment_paths;

  std::unique_ptr<StaticFileServer>          _http;
};

}

#endif
