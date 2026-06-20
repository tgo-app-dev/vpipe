#ifndef VPIPE_STAGES_LOAD_IMAGE_STAGE_H
#define VPIPE_STAGES_LOAD_IMAGE_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Source-ish stage: 0 or 1 inputs, 1 output. Reads still images from
// local files or network URLs via FFmpeg, decodes the first video
// frame of each input, converts to planar RGB, and emits one
// TensorBeat on out-port 0 per image with shape [3, H, W] and dtype
// U8 (raw 0..255 bytes, channel order R, G, B).
//
// Pacing: when an iport is wired (typically to a `chrono` stage),
// each upstream beat triggers exactly one decode+emit. With no iport
// the stage emits every URL back-to-back and signals done.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   url   (string or array<string>, required) -- one path or a list
//                          of paths/URLs. Local paths, "file://",
//                          "http(s)://", "rtsp://", etc. are all
//                          handled by FFmpeg's demuxer autodetection.
class LoadImageStage final : public TypedStage<LoadImageStage> {
public:
  static constexpr const char* kTypeName = "load-image";

  LoadImageStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~LoadImageStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessor.
  const std::vector<std::string>& urls() const noexcept { return _urls; }

private:
  std::unique_ptr<BeatPayloadIntf>
  decode_url_(const std::string& url) const;
  std::string av_err_(int rc) const;

  std::vector<std::string> _urls;
  std::size_t              _next = 0;

  const FFmpegLibraries*   _libs = nullptr;
};

}

#endif
