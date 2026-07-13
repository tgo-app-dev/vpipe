#ifndef VPIPE_STAGES_SAMPLER_SELECT_STAGE_H
#define VPIPE_STAGES_SAMPLER_SELECT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: choose a diffusion SAMPLER (the integrator) and emit its spec as
// a FlexData beat, once, on oport0. A `text-to-image` stage latches the spec off
// its optional sampler iport. Pairs with `scheduler-select` (the sigma
// schedule) -- sampler and scheduler are the two independent choices.
//
// The spec is:
//   { sampler:"flow_match", method, eta, s_noise, seed }
// method  -- "euler" (default, token-exact turbo) | "heun" | "dpmpp_2m"
//            | "dpmpp_sde"  (aliases "dpm++_2m"/"dpm++_sde" accepted).
// eta     -- dpmpp_sde stochasticity (0 => deterministic).
// s_noise -- dpmpp_sde added-noise scale.
// seed    -- dpmpp_sde noise seed.
//
// The method default comes from a model's scheduler config when `model` is set
// (_class_name: FlowMatchEuler -> euler, FlowMatchHeun -> heun); any explicit
// config field overrides. With no `model`, method defaults to euler.
//
// Config (FlexData object):
//   model     (string, optional) -- model dir/key for the scheduler default.
//   method    (string, optional) -- override sampler method.
//   eta       (real,   optional) -- override dpmpp_sde eta (default 1.0).
//   s_noise   (real,   optional) -- override dpmpp_sde s_noise (default 1.0).
//   seed      (int,    optional) -- override dpmpp_sde noise seed.
//   models_db (string, default "models") -- registry for resolve_model_dir.
class SamplerSelectStage final : public TypedStage<SamplerSelectStage> {
public:
  static constexpr const char* kTypeName = "sampler-select";

  SamplerSelectStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);
  ~SamplerSelectStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only: the resolved spec that would be emitted.
  FlexData resolved_spec() const;

private:
  std::string   _model;
  std::string   _method;
  std::string   _models_db;
  double        _eta = 1.0;
  double        _s_noise = 1.0;
  std::int64_t  _seed = 0;
  bool          _eta_set = false;      // eta=0 is valid, so track presence
  bool          _s_noise_set = false;
  bool          _seed_set = false;
  std::uint64_t _emitted = 0;
};

}

#endif
