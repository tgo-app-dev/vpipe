#ifndef VPIPE_STAGES_SCHEDULER_SELECT_STAGE_H
#define VPIPE_STAGES_SCHEDULER_SELECT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: choose a diffusion SCHEDULER (the sigma schedule) and emit its
// spec as a FlexData beat, once, on oport0. A `generate-image` stage latches it
// off its optional scheduler iport. Pairs with `diffusion-sampler-select` (the
// integrator) -- scheduler and sampler are the two independent choices.
//
// The spec is:
//   { scheduler:"flow_match", type, steps, shift, shift_type, rho }
// type       -- "simple" (default, the flow-match linspace+shift, token-exact)
//               | "karras" | "exponential".
// steps      -- denoising steps.
// shift      -- mu, the flow-matching time-shift strength.
// shift_type -- "exponential" | "linear".
// rho        -- karras curvature (default 7).
//
// This stage is MODEL-AGNOSTIC: it forwards the user's schedule choice and does
// not read any model's scheduler config (the generate-image stage owns the
// model). The built-in distilled turbo defaults apply unless a config field
// overrides (simple / 8 / 1.15 / exponential / 7).
//
// Config (FlexData object):
//   type       (string, optional) -- "simple" (default)/"karras"/"exponential".
//   steps      (int,    optional) -- override step count (default 8).
//   shift      (real,   optional) -- override mu (default 1.15).
//   shift_type (string, optional) -- override "exponential"/"linear".
//   rho        (real,   optional) -- override karras curvature (default 7).
class SchedulerSelectStage final : public TypedStage<SchedulerSelectStage> {
public:
  static constexpr const char* kTypeName = "scheduler-select";

  SchedulerSelectStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);
  ~SchedulerSelectStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only: the resolved spec that would be emitted.
  FlexData resolved_spec() const;

private:
  std::string  _type;
  std::string  _shift_type;
  std::int64_t _steps = 0;
  double       _shift = 0.0;
  double       _rho = 0.0;
  // boogu_v1 only (0 = unset -> the schedule's own default).
  std::int64_t _seq_len = 0;
  double       _base_shift = 0.0;
  double       _max_shift = 0.0;
  std::uint64_t _emitted = 0;
};

}

#endif
