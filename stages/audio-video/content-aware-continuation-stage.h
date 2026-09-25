#ifndef VPIPE_STAGES_CONTENT_AWARE_CONTINUATION_STAGE_H
#define VPIPE_STAGES_CONTENT_AWARE_CONTINUATION_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "pipeline/typed-stage.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Matches a CONTINUATION clip to the clip it continues, at the seam.
//
// A generated continuation opens where its guide ended, but not exactly
// as it looked: it can come back a few code values darker in the
// shadows, or a fraction of a percent taller or shorter, and then hold
// that for the rest of the clip. Joined end to end, that is a visible
// step at every seam. This stage measures the step on the frames either
// side of the seam and takes it out of the whole continuation.
//
//   iport0  the continuation's frames, planar RGB [3,H,W] u8 or f32
//           (what vae-decode emits), in order.
//   iport1  the reference: the clip being continued, stacked
//           [T,3,H,W] (what temporal-stack emits for a guide) or one
//           [3,H,W] frame. Only its LAST `window` frames are used.
//   oport0  the continuation's frames, same shape and dtype, corrected.
//
// TWO CORRECTIONS, each estimated once from the first `window` frames
// of the continuation against the last `window` frames of the reference
// and then applied to every frame:
//
//   geometry  an axis-aligned scale about the frame centre plus a shift,
//             x' = sx (x - cx) + cx + tx, and the same in y. Fitted to a
//             block-matched displacement field, so it follows the
//             picture rather than its statistics. No rotation or shear:
//             the drift this is for is a scale, and a looser model fits
//             noise.
//   tone      one monotone curve per channel, fitted to the aligned
//             pixel pairs. A curve rather than a gain and offset, because
//             the step is not uniform: it is largest in the shadows.
//
// Each is REFUSED rather than applied when it does not look like a
// seam: too few matching blocks, a scale beyond `max_scale`, or a curve
// that moves some level by more than `max_tone`. A continuation that
// does not resemble its reference is a scene change, and bending it
// toward the reference would be the error. A refusal is said once.
//
// The estimate is taken before the first frame is emitted, so the stage
// holds `window` frames and then streams. One continuation per run.
class ContentAwareContinuationStage final
  : public TypedStage<ContentAwareContinuationStage> {
public:
  static constexpr const char* kTypeName = "content-aware-continuation";

  ContentAwareContinuationStage(const SessionContextIntf* session,
                                std::string               id,
                                std::vector<InEdge>       iports,
                                FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // ---- the estimate, exposed so a test can pin it without a graph ----

  // Axis-aligned scale about the centre plus a shift, mapping a pixel of
  // the REFERENCE to where the same content sits in the continuation.
  struct Geometry {
    double sx = 1.0, sy = 1.0;   // scale about the frame centre
    double tx = 0.0, ty = 0.0;   // shift, pixels
    int    blocks_used  = 0;     // blocks that matched well enough
    int    blocks_total = 0;
    bool   ok = false;           // a fit was possible at all
    bool identity() const noexcept;
  };

  // One 256-entry curve per channel, continuation level -> reference
  // level, on a 0..255 scale whatever the frame's dtype.
  struct Tone {
    std::array<std::array<float, 256>, 3> lut{};
    double max_shift = 0.0;      // largest |curve - identity|, codes
    bool   ok = false;
    bool identity() const noexcept;
  };

  // `ref` and `cont` are planar RGB on a 0..255 scale, [3][H][W].
  static Geometry estimate_geometry(const std::vector<float>& ref,
                                    const std::vector<float>& cont, int w,
                                    int h);
  // `cont` is warped by `g` first; the curve is fitted on the aligned
  // pairs, so a geometric step is not read as a tonal one.
  static Tone estimate_tone(const std::vector<float>& ref,
                            const std::vector<float>& cont, int w, int h,
                            const Geometry& g);
  // Resample `src` [3][H][W] through `g` (bilinear, edges clamped):
  // out(x, y) = src(sx (x - cx) + cx + tx, ...).
  static void warp(const std::vector<float>& src, int w, int h,
                   const Geometry& g, std::vector<float>* out);

private:
  // One frame through the estimate; null when it would be unchanged,
  // so the caller forwards the original beat.
  std::unique_ptr<BeatPayloadIntf> correct_(const TensorBeatPayload& t);

  bool   _tone      = true;
  bool   _geometry  = true;
  int    _window    = 4;
  double _max_scale = 0.05;
  double _max_tone  = 32.0;
  int    _fade      = 0;

  // Frames held until the estimate is taken, and the reference's last
  // `window` frames averaged, planar on a 0..255 scale.
  std::vector<std::unique_ptr<BeatPayloadIntf>> _pending;
  std::vector<float> _ref;
  int  _rw = 0, _rh = 0;
  bool _have_ref = false, _ref_missing = false;

  bool     _estimated = false;
  Geometry _geo;
  Tone     _curve;
  std::int64_t _frames_out = 0;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_CONTENT_AWARE_CONTINUATION_STAGE_H
