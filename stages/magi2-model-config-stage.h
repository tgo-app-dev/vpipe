#ifndef VPIPE_STAGES_MAGI2_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_MAGI2_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// MAGI-2-preview's sampling knobs (see generative-models/magi2/): the
// step count, the flow shift and the two guidance scales.
class Magi2ModelConfigStage final
  : public ModelConfigSourceStage<Magi2ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "magi2-model-config";

  Magi2ModelConfigStage(const SessionContextIntf* session, std::string id,
                        std::vector<InEdge> iports, FlexData config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;
};

}  // namespace vpipe

#endif
