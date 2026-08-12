#ifndef VPIPE_STAGES_QWEN_IMAGE_EDIT_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_QWEN_IMAGE_EDIT_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: the QWEN-IMAGE-EDIT-specific parameters of a diffusion
// graph.
//
// Qwen2.5-VL takes a PIXEL BUDGET directly and smart-resizes from
// it, so this family has no separate long-edge cap the way the
// Qwen3-VL families do -- setting one here does nothing this model
// can act on.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
//
// Configuration (FlexData object): vl_long_edge, vl_pixel_budget,
// vl_min_pixels, vl_max_pixels -- see genai::GroundedEncodeParams. Each
// is emitted ONLY when set, so omitting one keeps the family's own
// number rather than replacing it with a default.
class QwenImageEditModelConfigStage final
  : public ModelConfigSourceStage<QwenImageEditModelConfigStage> {
public:
  static constexpr const char* kTypeName = "qwen-image-edit-model-config";

  QwenImageEditModelConfigStage(const SessionContextIntf* session,
                                std::string               id,
                                std::vector<InEdge>       iports,
                                FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
