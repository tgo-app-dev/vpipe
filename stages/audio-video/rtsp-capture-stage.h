#ifndef RTSP_CAPTURE_STAGE_H
#define RTSP_CAPTURE_STAGE_H

#include "pipeline/typed-stage.h"

#include <chrono>
#include <cstdint>
#include <string>

namespace vpipe {

// Long-running source stage (0-or-1 iport, 2 oports). Looks up a
// camera by `camera_name` in the LMDB "cameras" sub-db (populated by
// onvif-discovery-stage), decrypts the stored ONVIF password,
// connects to the RTSP stream via FFmpeg, and writes IDR-aligned
// MP4 segments to disk. Each finalized segment is indexed in the
// "<camera_name>-videos" sub-db keyed by the segment's UTC start
// time (8-byte big-endian microseconds-since-epoch).
//
// Iport (optional):
//   * iport 0 -- watchdog tick stream (typically TriggerPayload from
//                a chrono stage). The iport lives in a different clock
//                domain from the oports -- ticks arrive on the chrono
//                schedule, not the RTSP packet rate. Each tick checks
//                whether at least one RTSP packet was received since
//                the previous tick; if none, the stage forces an
//                FFmpeg abort on the current input, finalises the
//                segment, and reconnects. Leave the iport unwired to
//                disable the watchdog.
//
// Oports:
//   * oport 0 -- one Beat<EncodedSegment> per finalised file, carrying
//                the file's h.264 elementary stream (AVCC framing,
//                SPS/PPS in extradata) plus metadata.
//   * oport 1 -- one Beat<EncodedSegment> per finalised file, carrying
//                the file's AAC elementary stream (raw frames, ASC in
//                extradata) plus metadata.
// Both oports use the DropOldest overrun policy so a slow or absent
// downstream stage cannot backpressure the live capture loop. If
// neither oport has a consumer wired up, the demux step that
// produces the bytes is skipped entirely.
//
// On connection failure -- and on RTP errors/warnings observed in
// the FFmpeg log stream during capture -- the stage finalizes the
// current segment, closes the input, and reopens. If the camera
// was registered with `supports_onvif=true` and the first reopen
// also fails, the stage re-runs WS-Discovery, matches by UUID, and
// refreshes `rtsp_uri` in the cameras DB before the next try.
//
// Stop honouring: an AVIOInterruptCB is installed on every input
// context so a session-level stop_requested promptly aborts
// blocking FFmpeg I/O instead of waiting for `stimeout_us` to
// expire. The watchdog uses the same interrupt path to force a
// disconnect when its silent-camera trigger fires.
//
// APPLE-only in v1 (depends on credential-cipher + onvif-refresh).
class RtspCaptureStage final : public TypedStage<RtspCaptureStage> {
public:
  static constexpr const char* kTypeName = "rtsp-capture";

  RtspCaptureStage(const SessionContextIntf* s,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // The whole FFmpeg connect/segment/packet loop. Runs on a thread
  // this stage owns (spawned by process()) rather than on a session
  // worker, so the number of concurrent capture stages is bounded by
  // cameras, not by the worker-pool size. Pushes to the oports through
  // RuntimeContext::write_sync (non-coroutine). Returns once the
  // runtime requests stop.
  void capture_loop_(RuntimeContext& ctx);

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  // The iport (watchdog tick) and the two oports (encoded segments)
  // sit in different clock domains -- declared per-port in kSpec
  // (iport clock 0, oports clock 1). The oport ring depth defaults to
  // ~5 GOPs (300 AUs at gop_size=60 / 30 fps ~= 10 s of media);
  // DropOldest keeps the live capture loop unblocked on a slow
  // downstream consumer.
  std::string _camera_name;          // required
  std::string _output_dir;           // required
  std::chrono::seconds      _segment_seconds{};
  std::string _cameras_db;
  std::string _videos_db_suffix;
  std::string _rtsp_transport;
  unsigned    _stimeout_us{};
  unsigned    _connect_timeout_ms{};
  unsigned    _reconnect_delay_ms{};
  unsigned    _rediscover_timeout_ms{};
  uint64_t    _video_bitrate{};
  uint64_t    _audio_bitrate{};
  unsigned    _oport_depth{};
};

}

#endif
