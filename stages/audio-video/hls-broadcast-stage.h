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
// iport 0: TensorBeat (float32 [3, H, W]). Resolution must be stable
//          for the stage's lifetime; mid-stream resizes warn + drop.
// oports : none (sink).
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
//   fps_num             (int, default 30)
//   fps_den             (int, default 1)
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
//   bind_address        (string, default "0.0.0.0")
//   port                (int, default 8080)
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

  // 1 iport, 0 oports. Single clock domain — Stage defaults are
  // correct, no override.

  // Test-only inspectors.
  const std::string& playlist_url()  const noexcept { return _playlist_url; }
  const std::string& playlist_name() const noexcept { return _playlist_name; }
  double             segment_duration()  const noexcept
  { return _segment_duration; }
  double             live_start_offset() const noexcept
  { return _live_start_offset; }
  bool               encoder_initialized() const noexcept
  { return _enc != nullptr; }
  bool               server_running() const noexcept
  { return _http && _http->running(); }
  int                http_port() const noexcept
  { return _http ? _http->bound_port() : 0; }
  const InMemoryBlobStore& blobs() const noexcept { return _blobs; }

  // Internal: FFmpeg io callbacks (public because they need to be
  // pointed-at by C function pointers; not part of the user API).
  int io_open_ (AVIOContext** pb, const std::string& url);
  int io_close_(AVIOContext*  pb);

private:
  struct InMemSink;

  bool ensure_encoder_   (int H, int W);
  bool ensure_sws_       (int H, int W);
  bool open_output_      ();
  AVFrame* tensor_to_yuv_(const TensorBeat& tb);
  int  send_and_mux_     (AVFrame* yuv);
  int  drain_packets_    ();
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
  bool        _input_normalized{};
  bool        _realtime{};
  int         _log_input_stats_every{};
  bool        _serve_http{};
  std::string _bind_address;
  int         _http_port{};

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
