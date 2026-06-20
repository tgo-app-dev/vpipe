#ifndef VIDEO_DETECTION_OVERLAY_STAGE_H
#define VIDEO_DETECTION_OVERLAY_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpipe {

// Consumes (RGB TensorBeat, detection FlexData) one pair per frame
// and emits a new TensorBeat with the top-N bounding boxes + labels
// drawn on top.
//
// iport 0: TensorBeat ([3, H, W], optionally pitch-padded). Dtype
//          may be F32 or U8; both pass through dtype-faithfully.
// iport 1: FlexData object matching the schema produced by
//          yolo-detection-stage (or byte-track-stage downstream):
//            { frame_width:int, frame_height:int,
//              detections: [ { class_id, class_name?, score,
//                              track_id?, x1, y1, x2, y2 }, ... ] }
//          Bbox coords are xyxy in original-frame pixels (doubles).
//          The optional `track_id` field is honoured: when present
//          the bbox + label are coloured by track_id (so each
//          tracked object keeps a stable colour across frames),
//          otherwise they are coloured by class_id.
// iport 2: (optional) FlexData from an audio-tagging stage:
//            { timestamp_us:uint, tags:[ { label:str, ... }, ... ] }.
//          NOT in the same clock domain as iport 0/1 -- the audio
//          tagger runs on its own clock. Each frame the stage drains
//          every classification whose window timestamp is strictly
//          before the frame's `timestamp_us` (reading until it goes
//          beyond the video time); the most recent window's top class
//          is drawn just above the timestamp label, but only when its
//          score exceeds `audio_label_threshold` -- a window whose
//          tags are all below the threshold clears the label and the
//          field is skipped. Unwired (num_iports() < 3) disables the
//          audio overlay.
// oport 0: TensorBeat (same dtype as input, [3, H, W], always
//          contiguous on output).
//
// iport 0 and iport 1 share one clock domain: a frame's detection set
// on iport 1 arrives 1:1 with its image on iport 0, read sequentially
// (one cycle handles one frame). iport 2 (audio) is a separate clock
// domain, drained non-blockingly and timestamp-gated against the
// frame.
//
// Drawing semantics:
//  - Each detection's box is rendered as a rectangular border of
//    configurable thickness in a deterministic HSV-derived colour
//    keyed on `track_id` when the detection carries one (so the
//    same tracked object keeps its colour across frames), or on
//    `class_id` otherwise.
//  - Each detection optionally gets a filled label rectangle above
//    the box (or just below the top edge inside the box if the
//    box hugs the top of the frame). The label text is
//    "[#<track_id> ]<class_name> <score>" where score has
//    configurable precision; class_name falls back to "cls<id>"
//    when the FlexData entry omits a name, and the "#<track_id>"
//    prefix is added only when the detection carries a track_id.
//  - Detections below score_threshold are dropped before top-N
//    selection. After sorting by descending score, the first
//    top_n are drawn; top_n == 0 means draw all surviving.
//
// Output values: copied from the input (deep-copy, since beats are
// immutable). Coloured pixels are written in the same numeric
// range as the input. For F32 with input_normalized=true the range
// is [0, 1]; otherwise [0, 255]. For U8 the range is always
// [0, 255] (input_normalized is ignored).
//
// Config (FlexData object, all optional):
//   top_n             (int,    default 10)    0 = draw all
//   score_threshold   (real,   default 0.0)   drop below
//   box_thickness     (int,    default 2)     clamped to [1, 16]
//   font_scale        (int,    default 2)     integer >=1, clamped <=8
//   label_padding     (int,    default 2)     pixels around glyphs in bg
//   draw_label        (bool,   default true)
//   draw_score        (bool,   default true)
//   score_precision   (int,    default 2)     digits after decimal
//   input_normalized  (bool,   default true)  F32 only; ignored for U8
//   class_names       (array<string>, optional)  -- label lookup
//                     keyed by class_id. Used when the detection
//                     FlexData entry does not carry a per-detection
//                     `class_name` field, and when class_id is in
//                     range and the entry is non-empty. Falls back
//                     to "cls<id>" otherwise.
//   draw_timestamp        (bool, default false) -- when true and the
//                          incoming TensorBeat's sideband carries a
//                          `timestamp_us` (uint64, microseconds-since
//                          -UTC-epoch) field, render the time in
//                          human-readable form ("YYYY-MM-DD HH:MM:SS
//                          .fff") in the bottom-right corner of the
//                          frame. No-op when the sideband is absent.
//   timestamp_use_local   (bool, default false) -- when true the
//                          timestamp is formatted in the host's local
//                          time zone; otherwise UTC (default).
//   audio_label_threshold (real, default 0.6) -- iport2 audio label is
//                          shown only when its top tag's score exceeds
//                          this; otherwise the label is cleared and the
//                          field skipped. Clamped to [0, 1].
//   oport_capacity    (int,    default 4)
class DetectionOverlayStage final
    : public TypedStage<DetectionOverlayStage>
{
public:
  static constexpr const char* kTypeName = "detection-overlay";

  DetectionOverlayStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);
  ~DetectionOverlayStage() override = default;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // iport2 (audio tags) is on its own clock; iport0 / iport1 and the
  // output share the video clock (group 0). The clock groups are
  // static, declared per-port in kSpec (iport0/1 + oport0 = group 0,
  // iport2 = group 1), so no clock-group override is needed.

  // Internal: per-class pre-rendered label tile. The cache key is
  // class_id; the value holds the class-name text (used to validate
  // cache hits against changing labels) plus a contiguous planar-RGB
  // tile (3 × th × tw, glyph foreground composited on class bg, no
  // surrounding padding). Score digits and the separator space are
  // drawn per-frame so the tile only depends on stage config.
  struct CachedLabelTileU8 {
    std::string          text;
    int                  tw = 0;
    int                  th = 0;
    std::vector<std::uint8_t> bytes;
  };
  struct CachedLabelTileF32 {
    std::string          text;
    int                  tw = 0;
    int                  th = 0;
    std::vector<float>   bytes;
  };

  std::unordered_map<int, CachedLabelTileU8>&  u8_label_tiles()
  { return _u8_label_tiles; }
  std::unordered_map<int, CachedLabelTileF32>& f32_label_tiles()
  { return _f32_label_tiles; }

private:
  // Drain audio-tagging classifications from iport2 whose window
  // timestamp_us is strictly before `boundary_ts_us` (the current
  // frame's timestamp), advancing the iport2 cursor past them. The
  // most recent window's gated top class becomes _cur_audio_label
  // (drawn above the timestamp); a window whose tags are all below
  // audio_label_threshold yields an empty label and clears it, so the
  // field is dropped rather than left showing a stale tag. No-op when
  // iport2 is unwired (num_iports() < 3) or has reached EOS.
  Job consume_audio_(RuntimeContext& ctx, std::uint64_t boundary_ts_us);

  // All drawing helpers are templated on the pixel type T (float for
  // F32, uint8_t for U8) and live in the .cc. The header just
  // exposes the knobs they consult.
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  int                      _top_n{};
  double                   _score_threshold{};
  int                      _box_thickness{};
  int                      _font_scale{};
  int                      _label_padding{};
  bool                     _draw_label{};
  bool                     _draw_score{};
  int                      _score_precision{};
  bool                     _input_normalized{};
  std::vector<std::string> _class_names;
  bool                     _draw_timestamp{};
  bool                     _timestamp_use_local{};
  // Minimum audio-tag score required to display the audio label, [0,1].
  // The dominant class must score strictly above this or no label shows.
  double                   _audio_label_threshold{};
  unsigned                 _oport_capacity{};

  int _last_warn_w = -1;
  int _last_warn_h = -1;

  // Current audio top class to draw above the timestamp. Updated from
  // the most recent iport2 window consumed before a frame: a window
  // above audio_label_threshold sets it; a window all below threshold
  // clears it (empty -> the field is skipped). Persists across frames
  // only until the next window arrives.
  std::string _cur_audio_label;
  bool        _audio_eos = false;

  std::unordered_map<int, CachedLabelTileU8>  _u8_label_tiles;
  std::unordered_map<int, CachedLabelTileF32> _f32_label_tiles;
};

}

#endif
