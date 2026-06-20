#ifndef VPIPE_STAGES_BYTE_TRACK_STAGE_H
#define VPIPE_STAGES_BYTE_TRACK_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// ByteTrack multi-object tracker.
//
// Postprocesses yolo-detection output: assigns persistent integer
// track ids to detected boxes across frames using a constant-velocity
// Kalman filter (state = [cx, cy, aspect, h, vx, vy, va, vh]) and a
// two-stage IoU + Hungarian (lapjv) assignment. High-confidence
// detections drive the first association; low-confidence detections
// only get to re-associate already-tracked boxes (the key
// ByteTrack idea).
//
// Pipeline shape:
//   yolo-detection --> byte-track
//   iport 0: Beat<FlexData> -- yolo-detection's frame record:
//              { frame_width, frame_height, detections: [
//                  { class_id, [class_name], score, x1, y1, x2, y2 }, ...
//              ] }
//   oport 0: Beat<FlexData> -- same shape, but only detections that
//              the tracker confirmed are emitted, each enriched with
//              an integer `track_id`. Track ids are 1-based and
//              monotonically increasing for the life of the stage.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   track_thresh   (real, default 0.5)  -- detections with score
//                                          >= track_thresh enter the
//                                          high-score pool.
//   high_thresh    (real, default 0.6)  -- unmatched high-score
//                                          detections must clear this
//                                          bar to spawn a new track.
//   match_thresh   (real, default 0.8)  -- cost-matrix cap for the
//                                          first association round
//                                          (cost = 1 - IoU).
//   frame_rate     (int,  default 30)   -- used together with
//   track_buffer   (int,  default 30)   -- to derive max_time_lost,
//                                          the # of frames a lost
//                                          track waits before being
//                                          removed.
//   oport_capacity (int,  default 4)
//
// Adapted from FoundationVision/ByteTrack
// (https://github.com/FoundationVision/ByteTrack); see
// THIRD_PARTY_LICENSES.md for the verbatim license.
class ByteTrackStage final : public TypedStage<ByteTrackStage> {
public:
  static constexpr const char* kTypeName = "byte-track";

  ByteTrackStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);

  ~ByteTrackStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}

#endif
