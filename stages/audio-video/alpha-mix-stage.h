#ifndef VPIPE_STAGES_ALPHA_MIX_STAGE_H
#define VPIPE_STAGES_ALPHA_MIX_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

struct TensorBeat;

// Composites one image OVER another -- Porter-Duff `over`, which is what
// every layer stack, every "paste with transparency" and every matte
// means by mixing.
//
//   out_a   = aF + aB * (1 - aF)
//   out_rgb = (rgbF*aF + rgbB*aB*(1 - aF)) / out_a
//
// accumulated PREMULTIPLIED and divided back out once at the end, which
// is the form in which the blend is a plain weighted sum. Note the
// back's weight is `aB * (1 - aF)` and not `(1 - aF)`: the two agree
// whenever the back is opaque, so dropping aB is invisible in the
// ordinary case and wrong only when BOTH layers are transparent.
//
// EITHER INPUT MAY BE RGB OR RGBA, independently: a beat with three
// channels is opaque, which makes `front` RGB + a mask the ordinary
// stencil case and `back` RGB the ordinary flatten case.
//
// THE OPTIONAL `alpha` IPORT IS A MASK, in create-mask's own output
// format -- [1,H,W] or [H,W], U8 0..255 or F32 coverage in [0,1]. It
// MULTIPLIES the front image's alpha rather than replacing it, and
// that one rule covers both things a caller wants: against an RGB
// front (alpha 1 everywhere) the mask IS the alpha, and against an
// RGBA front it restricts where the front shows without discarding
// the transparency it already had. So a create-mask wired here needs
// no mode switch -- only a canvas the same size as the images.
//
// `output` decides what leaves:
//   rgba  the composite as it is, alpha kept
//   rgb   the composite then flattened onto `background`, which is
//         what "a background colour for fusing transparent inputs"
//         means: out = rgb*a + bg*(1-a)
//
// The output dtype follows the FRONT input, so a U8 graph stays U8 and
// an f32 one stays f32 without a conversion nobody asked for.
//
// iport 0: front -- planar RGB [3,H,W] or RGBA [4,H,W] (F32 or U8).
// iport 1: back  -- the same. Both are required.
// iport 2: alpha -- OPTIONAL mask, [1,H,W] or [H,W] (F32 or U8).
// oport 0: the composite, [3,H,W] or [4,H,W] per `output`.
class AlphaMixStage final : public TypedStage<AlphaMixStage> {
public:
  static constexpr const char* kTypeName = "alpha-mix";

  AlphaMixStage(const SessionContextIntf* session,
                std::string               id,
                std::vector<InEdge>       iports,
                FlexData                  config);

  const StageSpec& spec() const noexcept override;

  Job process(RuntimeContext& ctx) override;

private:
  // One input, normalized to planar float: `rgb[c]` in 0..1 and `a` in
  // 0..1, whatever the beat's dtype and channel count were.
  struct Plane {
    std::vector<float> rgb;    // 3 * H * W
    std::vector<float> a;      // H * W, all 1 when the beat had none
    int w = 0, h = 0;
    bool had_alpha = false;
    bool ok = false;
  };
  Plane read_image_(const TensorBeat& tb) const;
  // The mask, as H*W coverage in 0..1. `ok` false when the beat is not
  // a mask shape or does not match `w`/`h`.
  Plane read_mask_(const TensorBeat& tb, int w, int h) const;

  bool          _rgba_out = false;
  bool          _input_normalized = true;
  std::uint8_t  _bg_r = 255, _bg_g = 255, _bg_b = 255;
};

}  // namespace vpipe

#endif
