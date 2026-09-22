#ifndef VPIPE_STAGES_Z_IMAGE_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_Z_IMAGE_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source: the Z-IMAGE-specific parameters of a diffusion graph.
//
// This family needs one MORE than most, and for an unusual reason: its
// two published checkpoints -- the distilled Turbo and the undistilled
// base -- ship BYTE-IDENTICAL transformer configs. Nothing that reads
// the DiT can tell them apart, so which one is loaded has to be said
// rather than detected, and `guidance_scale` is where it is said.
// Turbo is distilled to guidance 0 and answers no negative prompt;
// the base model wants 3-5 over 28-50 steps and responds to one.
//
// Guidance IS here rather than on generate-image's shared
// `guidance_scale` because this model's CFG is not the shared formula:
// it is `pos + g*(pos - neg)` where every other family here computes
// `neg + g*(pos - neg)`. The two differ by exactly one in the scale, so
// a model card's "guidance 4" fed to the shared knob is the wrong
// picture rather than an obviously broken one.
//
// See stages/model-config-source.h for the beat contract and the
// trigger rule (unwired = one beat for the run; wired = one per
// inbound beat).
class ZImageModelConfigStage final
  : public ModelConfigSourceStage<ZImageModelConfigStage> {
public:
  static constexpr const char* kTypeName = "z-image-model-config";

  ZImageModelConfigStage(const SessionContextIntf* session,
                         std::string               id,
                         std::vector<InEdge>       iports,
                         FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
