#ifndef VPIPE_STAGES_AUDIO_CAPTURE_STAGE_H
#define VPIPE_STAGES_AUDIO_CAPTURE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>

namespace vpipe {

// Live audio capture source stage (0 iports, 1 oport).
//
// Opens a microphone / line-in audio device via FFmpeg's
// "avfoundation" input demuxer (macOS-only) and emits one
// Beat<EncodedSegment> per audio packet on oport 0. The emitted
// payload is shaped so a downstream `audio-to-pcm` stage can decode
// it directly:
//
//   * kind        = Audio
//   * codec_id    = the stream's AVCodecID as reported by avfoundation
//                   (typically AV_CODEC_ID_PCM_S16LE or AV_CODEC_ID_PCM_F32LE
//                   -- raw PCM at the device's native rate). audio-to-pcm
//                   resolves a PCM decoder + resamples to mono float32 at
//                   its configured output_sample_rate.
//   * sample_rate / channels = as advertised by the device.
//   * extradata   = empty (PCM doesn't need any).
//   * data        = the raw packet bytes verbatim.
//
// Each beat carries UTC wall-clock timestamps in `start_utc` /
// `end_utc`; `duration_us` is the packet's duration in microseconds
// derived from the demuxer's PTS unit (`AVStream::time_base`).
//
// Device selection (configuration must set exactly one of):
//   * device_id      (uint)    -- avfoundation device index (the leading
//                                 "[N]" in `ffmpeg -f avfoundation
//                                 -list_devices true -i ""`).
//   * device_name    (string)  -- device's human-readable name. A
//                                 case-insensitive substring match is
//                                 used; for built-in microphones the
//                                 default macOS name is e.g.
//                                 "Built-in Microphone" or
//                                 "MacBook Pro Microphone".
//
// Optional configuration:
//   sample_rate      (uint)    request a specific input sample rate
//                              (the device may reject unsupported rates;
//                              the stage falls back to the device default
//                              after logging a warning).
//   channels         (uint)    request a specific channel count.
//   reconnect_delay_ms (uint)  backoff between open attempts on error
//                              (default 2000).
//   oport_depth      (uint)    output ring capacity (default 256). The
//                              oport uses DropOldest so a slow consumer
//                              never backpressures the capture loop.
//
// Stop honouring: an AVIOInterruptCB is installed on the input format
// context so a session-level stop_requested promptly aborts any
// blocking read in libavformat's avfoundation demuxer.
class AudioCaptureStage final : public TypedStage<AudioCaptureStage> {
public:
  static constexpr const char* kTypeName = "audio-capture";

  AudioCaptureStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  uint64_t      packets_emitted() const noexcept { return _packets_emitted; }
  unsigned      input_sample_rate() const noexcept { return _input_sample_rate; }
  unsigned      input_channels() const noexcept { return _input_channels; }
  unsigned      input_codec_id() const noexcept { return _input_codec_id; }
  bool          has_device_id() const noexcept { return _has_device_id; }
  std::uint64_t device_id() const noexcept { return _device_id; }
  const std::string& device_name() const noexcept { return _device_name; }

private:
  // Resolve `device_name` against `ffmpeg -f avfoundation
  // -list_devices true -i ""` output and return the matching index,
  // or -1 if no match. Used only when `device_name` was configured.
  int probe_device_index_by_name_();

  // Either `_has_device_id` is true and `_device_id` is the resolved
  // avfoundation index, or `_device_name` is non-empty.
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  bool        _has_device_id = false;
  std::uint64_t _device_id   = 0;
  std::string _device_name;
  unsigned    _req_sample_rate    = 0;   // 0 = device default
  unsigned    _req_channels       = 0;   // 0 = device default
  unsigned    _reconnect_delay_ms{};
  unsigned    _oport_depth{};

  // Filled in at runtime after a successful open.
  unsigned      _input_sample_rate = 0;
  unsigned      _input_channels    = 0;
  unsigned      _input_codec_id    = 0;
  std::uint64_t _packets_emitted   = 0;
};

}

#endif
