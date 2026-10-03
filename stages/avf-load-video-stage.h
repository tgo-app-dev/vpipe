#ifndef VPIPE_STAGES_AVF_LOAD_VIDEO_STAGE_H
#define VPIPE_STAGES_AVF_LOAD_VIDEO_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace apple_media { class VideoReader; }

// Source: a movie file's frames through AVFoundation and VideoToolbox, as
// planar pictures -- what the FFmpeg path (load-video -> video-to-rgb,
// 8-bit, BT.601/709) flattens is kept: ProRes 4444's 12 bits and alpha,
// HEVC Main10's 10 bits and its PQ / HLG transfer with the HDR10
// metadata. One TensorBeat [3|4,H,W] per frame, F16 (default) or U8,
// tagged with the colour the file declares (sideband color_*,
// mastering_display, content_light) and the clip's timing (fps,
// fps_num/fps_den, frame, frames, pts_us). Video only; the file's audio
// is not read. Apple builds only.
class AvfLoadVideoStage final : public TypedStage<AvfLoadVideoStage> {
public:
  static constexpr const char* kTypeName = "avf-load-video";

  AvfLoadVideoStage(const SessionContextIntf* session, std::string id,
                    std::vector<InEdge> iports, FlexData config);
  ~AvfLoadVideoStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  // Test-only.
  std::uint64_t frames_emitted() const noexcept { return _emitted; }

private:
  std::string _url;
  TensorBeat::DType _dtype = TensorBeat::DType::F16;
  bool        _keep_alpha = true;
  double      _start_s = 0;
  double      _duration_s = 0;
  std::uint64_t _max_frames = 0;

  std::unique_ptr<apple_media::VideoReader> _reader;
  bool          _failed = false;
  std::uint64_t _emitted = 0;
};

}  // namespace vpipe

#endif
