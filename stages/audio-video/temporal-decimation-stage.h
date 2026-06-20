#ifndef VPIPE_STAGES_TEMPORAL_DECIMATION_STAGE_H
#define VPIPE_STAGES_TEMPORAL_DECIMATION_STAGE_H

#include "apple-silicon/tensor-storage.h"
#include "common/flex-data.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace vpipe {

namespace metal_compute { class MetalCompute; }

// Apple-Silicon temporal-decimation stage.
//
//   iport0  TensorBeat planar u8 [3, H, W] RGB video frames (the
//           `sideband.timestamp_us` key, when present, drives the
//           token-bucket pacing).
//
//   iport1  *Optional* FlexData with the YOLO-style detection schema
//           (`{frame_width, frame_height, detections: [{class_id,
//           class_name, score, x1, y1, x2, y2}]}`). When wired the
//           stage reads it 1:1 with iport0; when not wired the stage
//           runs detection-blind. Optionality is established at
//           graph-build time: connect zero or one InEdge to iport1.
//
//   oport0  Forwarded TensorBeatPayload on keep decisions. Dropped
//           frames produce no output. The output sideband is the
//           input sideband augmented with:
//             - decim_motion_score   (real, 0..1 normalised)
//             - decim_focus_share    (real, 0..1; only present when
//                                     iport1 was wired)
//             - decim_consec_drops   (uint, drops BEFORE this kept
//                                     frame)
//             - decim_kept           (bool true)
//
// Drop policy: a token-bucket capping the average keep rate at
// `max_avg_fps`. Each frame arrival refills the bucket by
// (dt * max_avg_fps) where dt is the inter-frame interval inferred
// from `timestamp_us` (or 1/max_avg_fps when timestamps are absent).
// The cap is a CEILING, not a target — staying below it is fine, and
// the policy never force-keeps just to drain accumulated tokens.
//
// A frame is kept when:
//   (a) the consecutive-drops counter would otherwise exceed
//       `max_consecutive_drops` (the floor mechanism), OR
//   (b) motion priority is at or above `motion_threshold` AND the
//       bucket has >= 1 token to spend.
//
// Long-run behaviour:
//   - Idle motion (below threshold): keep rate ≈ source / (max_drops
//     + 1), independent of `max_avg_fps`.
//   - Above-threshold motion: keep rate ≤ `max_avg_fps`, with bursts
//     up to `bucket_capacity` keeps when accumulated tokens permit.
//   - Rolling-window keep rate over ~32 source frames converges to
//     ≤ `max_avg_fps` + (bucket_capacity / window_seconds).
//
// `max_avg_fps` accepts sub-1 values for slow-motion / occasional-
// keep workloads. Motion priority blends total-frame motion with a
// focus-class boost:
//     priority = total_motion + focus_motion_gain * focus_motion_share
// where `focus_motion_share` is the fraction of per-tile motion that
// falls inside any focus-class detection bbox (zero when iport1 isn't
// wired or no detection's class is in focus_classes).
class TemporalDecimationStage final
    : public TypedStage<TemporalDecimationStage> {
public:
  static constexpr const char* kTypeName = "temporal-decimation";

  TemporalDecimationStage(const SessionContextIntf* session,
                          std::string               id,
                          std::vector<InEdge>       iports,
                          FlexData                  config);
  ~TemporalDecimationStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test / introspection accessors.
  int  tile_w() const noexcept { return _tile_w; }
  int  tile_h() const noexcept { return _tile_h; }
  bool consumes_detections() const noexcept { return _has_iport1; }
  std::uint64_t kept_count()    const noexcept { return _kept_count; }
  std::uint64_t dropped_count() const noexcept { return _dropped_count; }

  // Per-frame decision helper, exposed for unit tests that feed a
  // synthetic motion-score sequence. Updates internal state (token
  // bucket, consec-drops counter, kept/dropped counters) and returns
  // true iff the frame should be kept.
  bool decide_keep_for_test(double           total_motion,
                            double           focus_motion_share,
                            std::uint64_t    timestamp_us);

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  double                        _max_avg_fps{};
  unsigned                      _max_consec_drops{};
  int                           _tile_w{};
  int                           _tile_h{};
  double                        _motion_threshold{};
  double                        _focus_motion_gain{};
  double                        _bucket_capacity{};
  std::unordered_set<int>       _focus_class_ids;
  std::unordered_set<std::string> _focus_class_names;

  // Runtime state.
  metal_compute::MetalCompute*  _mc                  = nullptr;
  bool                          _has_iport1          = false;
  // Ping-pong signatures so the just-computed one becomes prev next
  // iteration without a copy.
  std::unique_ptr<ExternalStorageHandle> _sig_a;
  std::unique_ptr<ExternalStorageHandle> _sig_b;
  std::unique_ptr<ExternalStorageHandle> _diff;
  // _sig_cur points at the buffer the next frame will write into;
  // _sig_prev holds the previous frame's signature.
  ExternalStorageHandle*        _sig_cur             = nullptr;
  ExternalStorageHandle*        _sig_prev            = nullptr;
  bool                          _have_prev           = false;
  int                           _src_w               = 0;
  int                           _src_h               = 0;
  // The src staging buffer is allocated lazily once we know the
  // input dims; reused across frames.
  std::unique_ptr<ExternalStorageHandle> _src_stage;

  // Pacing.
  double                        _tokens              = 0.0;
  double                        _last_refill         = 0.0;
  std::uint64_t                 _prev_ts_us          = 0;
  bool                          _have_prev_ts        = false;
  unsigned                      _consec_drops        = 0;

  // Bookkeeping.
  std::uint64_t                 _kept_count          = 0;
  std::uint64_t                 _dropped_count       = 0;

  // Rolling-window stats for the periodic info summary. Reset every
  // kSummaryEveryFrames frames; the debug-level per-frame log carries
  // the raw values.
  static constexpr std::uint64_t kSummaryEveryFrames = 100;
  std::uint64_t                 _win_frames          = 0;
  std::uint64_t                 _win_kept            = 0;
  std::uint64_t                 _win_dropped         = 0;
  double                        _win_motion_sum      = 0.0;
  double                        _win_motion_max      = 0.0;
  double                        _win_focus_sum       = 0.0;
  double                        _win_focus_max       = 0.0;
  double                        _win_tokens_sum      = 0.0;
  std::uint64_t                 _win_with_score      = 0;
  std::uint64_t                 _win_dets_total      = 0;
  std::uint64_t                 _win_focus_matches   = 0;

  // Internal helpers. Defined in the .cc.
  bool ensure_buffers_(int in_w, int in_h);
  // Returns total_motion in [0, 1] and (when det_fd != nullptr)
  // focus_motion_share in [0, 1]. focus_motion_share is 0 when no
  // focus-class detection is present. `out_n_dets` (when non-null)
  // receives the total detection count read from det_fd; `out_n_focus`
  // (when non-null) receives the count of those detections that
  // matched a configured focus_class. Both are 0 when det_fd is null.
  bool score_frame_(const std::uint8_t* src_bytes,
                    std::size_t         src_bytes_len,
                    int                 in_w,
                    int                 in_h,
                    const FlexData*     det_fd,
                    double*             out_total_motion,
                    double*             out_focus_share,
                    int*                out_n_dets,
                    int*                out_n_focus);
};

}

#endif
