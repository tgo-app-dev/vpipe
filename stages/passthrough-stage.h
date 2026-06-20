#ifndef PASSTHROUGH_STAGE_H
#define PASSTHROUGH_STAGE_H

#include "pipeline/typed-stage.h"
#include <string>
#include <vector>

namespace vpipe {

// Built-in 1-in / 1-out identity stage. Useful as a baseline test
// fixture and as an example of the minimal process() body. Forwards
// each input beat unchanged; both ports are type-erased ("any").
class PassthroughStage final : public TypedStage<PassthroughStage> {
public:
  static constexpr const char* kTypeName = "passthrough";

  PassthroughStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config =
                       FlexData::make_object());

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;
};

}

#endif
