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
// spec as a FlexData beat, once, on oport0. A `text-to-image` stage latches it
// off its optional scheduler iport. Pairs with `sampler-select` (the
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
// Defaults come from a model's scheduler config when `model` is set
// (time_shift_type -> shift_type, and max_shift/shift -> shift); any explicit
// config field overrides. With no `model`, the built-in distilled turbo defaults
// apply (simple / 8 / 1.15 / exponential / 7).
//
// Config (FlexData object):
//   model      (string, optional) -- model dir/key to read scheduler defaults.
//   type       (string, optional) -- override "simple"/"karras"/"exponential".
//   steps      (int,    optional) -- override step count (default 8).
//   shift      (real,   optional) -- override mu (default 1.15).
//   shift_type (string, optional) -- override "exponential"/"linear".
//   rho        (real,   optional) -- override karras curvature (default 7).
//   models_db  (string, default "models") -- registry for resolve_model_dir.
class SchedulerSelectStage final : public TypedStage<SchedulerSelectStage> {
public:
  static constexpr const char* kTypeName = "scheduler-select";

  SchedulerSelectStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);
  ~SchedulerSelectStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only: the resolved spec that would be emitted.
  FlexData resolved_spec() const;

private:
  std::string  _model;
  std::string  _type;
  std::string  _shift_type;
  std::string  _models_db;
  std::int64_t _steps = 0;
  double       _shift = 0.0;
  double       _rho = 0.0;
  std::uint64_t _emitted = 0;
};

}

#endif
