#ifndef VPIPE_STAGES_RGB_TO_VIDEO_STAGE_H
#define VPIPE_STAGES_RGB_TO_VIDEO_STAGE_H

#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Adapts a stream of planar-U8-RGB image beats into the header + frames
// contract the video encoder speaks.
//
// It exists because the two halves of a video graph were designed against
// different currencies and both are right. Everything on the generative
// side -- load-image, vae-decode, save-image -- passes a TensorBeat of
// planar U8 RGB, which is what the metal kernels produce and consume.
// save-video speaks ffmpeg: one VideoStreamParams header followed by
// FrameRef beats wrapping AVFrames. Rather than teach either side the
// other's format, this stage is the seam: one small, testable conversion
// with no model, no GPU and no state beyond the header it has already
// sent.
//
//   iport0  TensorBeatPayload, planar U8 RGB [3, H, W] -- one per frame,
//           in presentation order. The frame's index and the clip's frame
//           count and rate ride on the beat's sideband ({frame, frames,
//           fps}) when the producer knows them (vae-decode does); the
//           `fps` config is the fallback for producers that do not.
//
//   oport0  the video stream: exactly one VideoStreamParamsPayload,
//           emitted when the FIRST frame arrives and its size is known,
//           then one FrameRefPayload per frame.
//
// The header cannot be sent before the first frame because nothing else
// knows the frame size: a generative graph's dimensions come from the
// prompt and the DiT's config, not from this stage's own configuration.
// So the size is LATCHED from frame 0 and every later frame is checked
// against it -- a video stream whose dimensions change mid-clip is not
// something the encoder can be handed, and silently letting one through
// would surface much later as a corrupt file.
//
// Config (FlexData object):
//   fps        (real, default 16)  -- frame rate to declare when the
//                                    producer's sideband does not carry
//                                    one. 16 is the Wan default.
//   pix_fmt    (string, default "yuv420p") -- "yuv420p" | "rgb24".
//                                    yuv420p is what H.264 wants; rgb24
//                                    skips the conversion for a lossless
//                                    codec.
class RgbToVideoStage final : public TypedStage<RgbToVideoStage> {
public:
  static constexpr const char* kTypeName = "rgb-to-video";

  RgbToVideoStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);
  ~RgbToVideoStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  std::uint64_t frames_emitted() const noexcept { return _frames; }
  bool          header_sent()    const noexcept { return _header_sent; }
  int           width()          const noexcept { return _w; }
  int           height()         const noexcept { return _h; }

private:
  const FFmpegLibraries* _libs = nullptr;
  double        _fps = 16.0;
  std::string   _pix_fmt_name;
  int           _pix_fmt = 0;
  std::uint64_t _frames = 0;
  bool          _header_sent = false;
  int           _w = 0, _h = 0;
};

}  // namespace vpipe

#endif
