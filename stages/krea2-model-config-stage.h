#ifndef VPIPE_STAGES_KREA2_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_KREA2_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: the KREA-2-specific parameters of a diffusion graph.
//
// Krea-2's grounded edit encodes the instruction WITH the source
// image, and the ComfyUI-Krea2Edit node caps that image's long edge
// at 768 before the Qwen3-VL tower sees it -- looser than the
// Mage-Flow families' 384 and tighter than the tower's own ~1M-pixel
// limit. The number is the identity-edit LoRA's training
// distribution, so it is the family's, not the stage's.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
//
// Configuration (FlexData object): vl_long_edge, vl_pixel_budget,
// vl_min_pixels, vl_max_pixels -- see genai::GroundedEncodeParams. Each
// is emitted ONLY when set, so omitting one keeps the family's own
// number rather than replacing it with a default.
class Krea2ModelConfigStage final
  : public ModelConfigSourceStage<Krea2ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "krea2-model-config";

  Krea2ModelConfigStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
