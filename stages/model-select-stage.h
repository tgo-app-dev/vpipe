#ifndef VPIPE_STAGES_MODEL_SELECT_STAGE_H
#define VPIPE_STAGES_MODEL_SELECT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: pick ONE diffusion model directory and emit it as a FlexData
// beat, once, on oport0. A text-to-image graph's diffusion-conditioner, DiT
// (text-to-image), vae-encode and vae-decode stages all need the SAME model;
// wiring this one source into each of their optional `model` iports keeps that
// choice in a single place -- change the model here rather than in four
// separate configs. The emitted beat OVERRIDES each consumer's `hf_dir` config
// key.
//
// The beat is:  { "hf_dir": <dir-or-registry-key>, "models_db": <db> }
// which apply_model_select_beat() (model-registry.h) parses on the consumer
// side; each consumer then resolve_model_dir()s it as usual, so a registry key
// or a filesystem path both work.
//
// Config (FlexData object):
//   hf_dir     (string, required) -- the model dir/registry key shared by the
//                                    conditioner / DiT / vae-encode / vae-decode
//                                    stages.
//   models_db  (string, default "models") -- the registry db the consumers pass
//                                    to resolve_model_dir.
class ModelSelectStage final : public TypedStage<ModelSelectStage> {
public:
  static constexpr const char* kTypeName = "model-select";

  ModelSelectStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);
  ~ModelSelectStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only: the beat that would be emitted.
  FlexData resolved_beat() const;

private:
  std::string   _hf_dir;
  std::string   _models_db;
  std::uint64_t _emitted = 0;
};

}

#endif
