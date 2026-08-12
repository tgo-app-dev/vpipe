#ifndef GENERATIVE_MODELS_SHARED_GROUNDED_ENCODE_PARAMS_H
#define GENERATIVE_MODELS_SHARED_GROUNDED_ENCODE_PARAMS_H

#include "common/flex-data.h"

#include <cstddef>
#include <string>

namespace vpipe {
namespace genai {

// How one family wants a REFERENCE IMAGE prepared before its vision tower
// sees it, for the grounded encode that produces image-aware
// conditioning.
//
// These numbers come from each reference pipeline's own preprocessing and
// they are not interchangeable: Mage-Flow's `vl_cond_long_edge` is 384
// where Krea-2's grounding node uses 768, and Boogu-Image caps the long
// side at 768 but the AREA at 384x384. Feeding one family another's
// numbers produces a well-formed conditioning that lands somewhere the
// DiT was never trained against -- an edit that mis-targets rather than a
// failure, which is the expensive kind of wrong.
//
// They lived as literals in the conditioner's family branches, which made
// them invisible and unadjustable. Here they are the model layer's
// statement of what its family needs, with `for_family()` as the
// authority and a config beat able to override -- so a checkpoint
// fine-tuned at a different grounding resolution is a config change
// rather than a code change.
struct GroundedEncodeParams {
  // Cap on the LONGEST side, applied before the tower's own smart-resize.
  // 0 = no cap. The tower resizes internally anyway; this bound exists to
  // keep the image inside the training distribution of the family's
  // grounding LoRA, which is tighter than the tower's ~1M-pixel limit.
  int long_edge = 0;
  // Cap on total PIXELS, applied together with `long_edge`. 0 = none.
  // Separate from the long edge because a wide image can satisfy one and
  // not the other, and Boogu-Image bounds both.
  std::size_t pixel_budget = 0;
  // The image processor's own bounds, passed to the vision tower's
  // config. 0 = keep the tower's default. `min_pixels` is what makes a
  // small or very wide reference get UPSCALED before patching; the
  // Mage-Flow families set it far above the Qwen default and a reference
  // that falls under it is silently not upscaled without this.
  std::size_t min_pixels = 0;
  std::size_t max_pixels = 0;

  // What `family` needs, before any config. The family vocabulary is the
  // conditioner's own `_family` tag ("krea2", "mage-flow", "boogu-image",
  // "qwen-image-edit"); anything else gets an all-zero set, which means
  // "no capping, tower defaults" -- the right answer for a family with no
  // grounded path at all.
  static GroundedEncodeParams for_family(const std::string& family);

  // Overlay a `model-config` beat's keys onto this. Absent keys leave the
  // family default alone, which is what makes a config stage that names
  // one knob not silently reset the other three.
  void merge_flex(const FlexData& fd, std::string* err = nullptr);
};

}  // namespace genai
}  // namespace vpipe

#endif
