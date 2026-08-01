#ifndef VPIPE_STAGES_VIDEO_CAPTURE_STAGE_H
#define VPIPE_STAGES_VIDEO_CAPTURE_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>

namespace vpipe {

// Live camera capture source stage (0 iports, 1 oport). The video twin of
// `audio-capture`: same device_id / device_name selection, same reconnect
// loop, same DropOldest oport, same AVIOInterruptCB stop honouring.
//
// Opens a camera via FFmpeg's "avfoundation" input demuxer (macOS-only) and
// emits one planar RGB TensorBeat [3, H, W] per frame on oport 0.
//
// WHY RGB AND NOT EncodedSegment (the shape audio-capture emits). The audio
// twin hands raw PCM to a downstream `audio-to-pcm`, so the video-shaped
// mirror would be to hand raw frames to `video-to-rgb`. That does not work:
// avfoundation delivers RAWVIDEO, whose pixel format is not carried by
// EncodedSegment, and video-to-rgb picks its access-unit splitter by sniffing
// H.264 AVCC / annex-B framing, which raw frames do not have. So this stage
// does the decode + colour conversion itself and emits video-to-rgb's OUTPUT
// contract instead -- identical payload, so a camera drops straight into
// every existing frame consumer (preview, save-image, image-resample,
// yolo-detection, vae-encode...).
//
// Each beat carries the same sideband object video-to-rgb attaches:
//   { "timestamp_us": <uint>,        wall-clock arrival of the frame
//     "camera_name":  <string>,      only when `camera_name` is configured
//     "fps_num", "fps_den": <uint> } the negotiated capture cadence
//
// Device selection (configuration must set exactly one of):
//   * device_id      (uint)    -- avfoundation VIDEO device index (the
//                                 leading "[N]" in the video block of
//                                 `ffmpeg -f avfoundation -list_devices
//                                 true -i ""`). Video and audio indices are
//                                 numbered separately -- "[0]" in the video
//                                 block is not the "[0]" audio device.
//   * device_name    (string)  -- device's human-readable name, matched
//                                 case-insensitively as a substring against
//                                 the VIDEO block only (e.g. "FaceTime",
//                                 "MacBook Air Camera").
//
// Optional configuration -- the avfoundation knobs are passed through
// verbatim, so `ffmpeg -f avfoundation -list_devices true` and the demuxer
// docs remain the reference for legal values:
//   width, height    (uint)    requested capture resolution, sent as
//                              avfoundation's `video_size` "WxH". BOTH must
//                              be set together. Unset = device default.
//                              avfoundation REJECTS a mode it does not
//                              support and lists the legal ones in its error
//                              (e.g. 1280x720@[15 30]fps), which this stage
//                              surfaces verbatim.
//   framerate        (real)    requested frames per second, avfoundation's
//                              `framerate`. Unset/0 = device default. Also
//                              rejected if unsupported for the chosen size.
//   pixel_format     (string)  requested capture pixel format (avfoundation
//                              `pixel_format`), e.g. "uyvy422", "nv12",
//                              "bgr0". Unset = device default. The frames are
//                              converted to RGB regardless; this only selects
//                              what the device hands over.
//   output_dtype     (string)  "u8" (default) or "f32" -- the emitted
//                              TensorBeat's element type, matching
//                              video-to-rgb's option of the same name. f32 is
//                              normalized to [0, 1].
//   camera_name      (string)  label copied into each beat's sideband, so a
//                              multi-camera graph can tell sources apart.
//   reconnect_delay_ms (uint)  backoff between open attempts on error
//                              (default 2000).
//   oport_depth      (uint)    output ring capacity (default 8; frames are
//                              far larger than audio packets, so the default
//                              is much shallower than audio-capture's 256).
//                              DropOldest, so a slow consumer never
//                              backpressures the camera.
class VideoCaptureStage final : public TypedStage<VideoCaptureStage> {
public:
  static constexpr const char* kTypeName = "video-capture";

  VideoCaptureStage(const SessionContextIntf* session,
                    std::string               id,
                    std::vector<InEdge>       iports,
                    FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  std::uint64_t frames_emitted() const noexcept { return _frames_emitted; }
  unsigned      input_width() const noexcept { return _input_width; }
  unsigned      input_height() const noexcept { return _input_height; }
  unsigned      input_codec_id() const noexcept { return _input_codec_id; }
  bool          has_device_id() const noexcept { return _has_device_id; }
  std::uint64_t device_id() const noexcept { return _device_id; }
  const std::string& device_name() const noexcept { return _device_name; }
  unsigned      req_width() const noexcept { return _req_width; }
  unsigned      req_height() const noexcept { return _req_height; }
  double        req_framerate() const noexcept { return _req_framerate; }
  const std::string& pixel_format() const noexcept { return _pixel_format; }
  TensorBeat::DType  output_dtype() const noexcept { return _output_dtype; }

private:
  // Resolve `device_name` against the VIDEO block of `ffmpeg -f avfoundation
  // -list_devices true -i ""` and return the matching index, or -1 if no
  // match. Used only when `device_name` was configured.
  int probe_device_index_by_name_();

  // Either `_has_device_id` is true and `_device_id` is the resolved
  // avfoundation video index, or `_device_name` is non-empty.
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  bool          _has_device_id = false;
  std::uint64_t _device_id     = 0;
  std::string   _device_name;
  unsigned      _req_width     = 0;   // 0 = device default (with height)
  unsigned      _req_height    = 0;
  double        _req_framerate = 0.0; // 0 = device default
  std::string   _pixel_format;        // empty = device default
  std::string   _camera_name;
  TensorBeat::DType _output_dtype = TensorBeat::DType::U8;
  unsigned      _reconnect_delay_ms{};
  unsigned      _oport_depth{};

  // Filled in at runtime after a successful open.
  unsigned      _input_width    = 0;
  unsigned      _input_height   = 0;
  unsigned      _input_codec_id = 0;
  unsigned      _fps_num        = 0;
  unsigned      _fps_den        = 0;
  std::uint64_t _frames_emitted = 0;
};

}

#endif
