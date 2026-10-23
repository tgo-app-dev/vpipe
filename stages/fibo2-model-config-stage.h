#ifndef VPIPE_STAGES_FIBO2_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_FIBO2_MODEL_CONFIG_STAGE_H

#include "stages/latent-preview.h"
#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the Fibo 2.0 parameters of a `generate-image` /
// `diffusion-conditioner` request.
//
// The guidance (Fibo 2.0 Dev's 5: the two checkpoints cannot be told
// apart by their files, so the graph says which recipe it runs), runtime
// LoRA slots, the live preview, and the bounds of the pictures Qwen3-VL
// reads for an edit. None is needed for an Edit Turbo run: the
// checkpoint states its own schedule and the conditioner its own
// thumbnail rule; this stage is for a graph that wants something else.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
class Fibo2ModelConfigStage final
  : public ModelConfigSourceStage<Fibo2ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "fibo2-model-config";

  Fibo2ModelConfigStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;

private:
  LatentPreviewSpec _preview;
};

}  // namespace vpipe

#endif
