#ifndef VPIPE_STAGES_QWEN_IMAGE_21_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_QWEN_IMAGE_21_MODEL_CONFIG_STAGE_H

#include "stages/latent-preview.h"
#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: the QWEN-IMAGE-2.1-specific parameters of a diffusion graph.
//
// Only what no other family can answer. Guidance is NOT here: it is
// `generate-image`'s own `guidance_scale`, which every family shares,
// and a second spelling of it on this stage would be two knobs for one
// number.
//
// The conditioning bounds are a CONTRACT rather than a preference here,
// which is why they are worth stating even though they have family
// defaults. One resize feeds both the text encoder and the VAE, and the
// DiT requires the tower's merged grid to be exactly half the VAE's
// latent grid -- so a `vl_pixel_budget` that does not survive the
// alignment to 32 produces a reference the DiT refuses to lay out.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
class QwenImage21ModelConfigStage final
  : public ModelConfigSourceStage<QwenImage21ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "qwen-image-21-model-config";

  QwenImage21ModelConfigStage(const SessionContextIntf* session,
                              std::string               id,
                              std::vector<InEdge>       iports,
                              FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;

private:
  // The live-preview keys (preview_vae / _every / _max_edge). No
  // `preview_frames`: a picture is one frame.
  LatentPreviewSpec _preview;
};

}  // namespace vpipe

#endif
