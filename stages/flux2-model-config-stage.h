#ifndef VPIPE_STAGES_FLUX2_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_FLUX2_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the FLUX.2-specific parameters of a `generate-image`
// request.
//
// One key, and it is the most consequential in the image stack.
// FLUX.2-klein-9b-kv is distilled with its reference tokens ISOLATED from
// the rest of the sequence, and nothing on disk distinguishes it from
// plain klein-9B -- same model_index.json, same transformer/config.json,
// same tensor names. So it cannot be detected, only declared; and getting
// it wrong is silent, because both combinations load and run and produce
// an image, just the wrong one.
//
// That is exactly why it belongs on a stage of its own rather than in
// `generate-image`'s config: as a key in a union config it read as one
// optional knob among a dozen, most of which do not apply to the resident
// checkpoint. Here it is the whole content of a stage a graph either
// wires or does not.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
//
// Configuration (FlexData object):
//   klein_kv (bool) -- the checkpoint is FLUX.2-klein-9b-kv. REQUIRED for
//                      that checkpoint and WRONG for any other.
class Flux2ModelConfigStage final
  : public ModelConfigSourceStage<Flux2ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "flux2-model-config";

  Flux2ModelConfigStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;

private:
  bool _klein_kv = false;
};

}  // namespace vpipe

#endif
