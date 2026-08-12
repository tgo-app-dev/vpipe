#ifndef VPIPE_STAGES_BOOGU_IMAGE_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_BOOGU_IMAGE_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: the BOOGU-IMAGE-specific parameters of a diffusion graph.
//
// BooguImagePipeline bounds its VLM conditioning image TWICE --
// long side 768 AND area 384x384 -- which is why its long edge is
// double Mage-Flow's rather than equal to it. A reference that
// satisfies one bound can fail the other, so neither implies the
// other and both travel together.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
//
// Configuration (FlexData object): vl_long_edge, vl_pixel_budget,
// vl_min_pixels, vl_max_pixels -- see genai::GroundedEncodeParams. Each
// is emitted ONLY when set, so omitting one keeps the family's own
// number rather than replacing it with a default.
class BooguImageModelConfigStage final
  : public ModelConfigSourceStage<BooguImageModelConfigStage> {
public:
  static constexpr const char* kTypeName = "boogu-image-model-config";

  BooguImageModelConfigStage(const SessionContextIntf* session,
                             std::string               id,
                             std::vector<InEdge>       iports,
                             FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
