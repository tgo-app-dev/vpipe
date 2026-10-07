#ifndef VPIPE_STAGES_AVF_SAVE_VIDEO_STAGE_H
#define VPIPE_STAGES_AVF_SAVE_VIDEO_STAGE_H

#include "common/job.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace apple_media { class VideoWriter; }

// Sink: planar pictures into a movie through AVFoundation and
// VideoToolbox -- ProRes 4444 (with alpha) / 422, HEVC Main10 (PQ / HLG
// with the HDR10 metadata), HEVC 8-bit, H.264 -- on the hardware
// encoders. Frames come in as F16 (10 and 12 bits reach the encoder) or
// U8, and the file declares the colour the frames are tagged with (or
// what the config says). The movie is finished at end of stream. Video
// only. Apple builds only.
class AvfSaveVideoStage final : public TypedStage<AvfSaveVideoStage> {
public:
  static constexpr const char* kTypeName = "avf-save-video";

  AvfSaveVideoStage(const SessionContextIntf* session, std::string id,
                    std::vector<InEdge> iports, FlexData config);
  ~AvfSaveVideoStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  // Test-only.
  std::uint64_t frames_written() const noexcept { return _written; }
  bool finished() const noexcept { return _finished; }
  // What its progress counted against (0: uncounted).
  std::uint64_t progress_total() const noexcept { return _total; }

private:
  void finish_();

  std::string _path;
  std::string _codec;
  double      _fps = 0;          // 0 = from the frames
  bool        _keep_alpha = true;
  double      _quality = 0;
  std::int64_t _bitrate = 0;
  int         _primaries = 0;    // 0 = from the frames
  int         _transfer = 0;
  int         _matrix = 0;

  std::uint64_t _expected = 0;   // config.frames; 0 = from the frames

  std::unique_ptr<apple_media::VideoWriter> _writer;
  bool          _failed = false;
  bool          _finished = false;
  std::uint64_t _written = 0;
  // Frames written of how many: a long movie's progress ("save video").
  UiProgress    _bar;
  std::uint64_t _total = 0;
};

}  // namespace vpipe

#endif
