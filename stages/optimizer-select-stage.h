#ifndef VPIPE_STAGES_OPTIMIZER_SELECT_STAGE_H
#define VPIPE_STAGES_OPTIMIZER_SELECT_STAGE_H

// optimizer-select -- the update rule and learning-rate schedule a
// train-lora stage runs, as one FlexData beat. The training twin of
// scheduler-select and diffusion-sampler-select: no inputs, one beat,
// then end of stream. Several of these beats into one train-lora (a
// sweep) run one training job each, by the generation pairing rule.

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

class OptimizerSelectStage final : public TypedStage<OptimizerSelectStage> {
public:
  static constexpr const char* kTypeName = "optimizer-select";

  OptimizerSelectStage(const SessionContextIntf* session, std::string id,
                       std::vector<InEdge> iports, FlexData config);
  ~OptimizerSelectStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  // The spec that would be emitted (tests).
  FlexData resolved_spec() const { return _spec; }

private:
  FlexData _spec;
  std::uint64_t _emitted = 0;
};

}  // namespace vpipe

#endif
