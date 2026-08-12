#include "stages/flux2-model-config-stage.h"

#include "common/flex-data.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "klein_kv", .type = ConfigType::Bool, .required = false,
   .doc = "the checkpoint is FLUX.2-klein-9b-kv rather than plain klein-9B: "
          "reference tokens are attention-isolated and modulated at a fixed "
          "timestep 0, and their K/V are cached across denoise steps (BFL "
          "measure 1.21-2.66x on multi-reference edits). REQUIRED for that "
          "checkpoint and WRONG for any other -- the two are indistinguishable "
          "on disk (same config.json, same tensor names), so it cannot be "
          "detected. Default false"},
};
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc = "OPTIONAL beat that gates re-emitting the config (a chrono tick, a "
          "prompt source, a feedback loop). Any payload -- receipt is the "
          "signal. Unwired, the stage emits once for the run",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "model_config",
   .doc = "FLUX.2 parameters as one FlexData object {model_family: flux2, "
          "klein_kv}, for a generate-image model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "flux2-model-config",
  .doc       = "Source: the FLUX.2-specific parameters -- which today means "
               "declaring a klein-9b-kv checkpoint, whose isolated-reference "
               "recipe nothing on disk reveals and which produces wrong "
               "images if it is run either way by accident. One beat then "
               "done; with a trigger iport, one beat per inbound beat.",
  .display_name = "FLUX.2 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Flux2ModelConfigStage::Flux2ModelConfigStage(const SessionContextIntf* s,
                                             std::string               id,
                                             std::vector<InEdge>       iports,
                                             FlexData                  config)
  : ModelConfigSourceStage<Flux2ModelConfigStage>(s, std::move(id),
                                                  std::move(iports),
                                                  std::move(config))
{
  _klein_kv = attr_bool("klein_kv");
  allocate_oports(spec().oports.size());
}

const StageSpec&
Flux2ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
Flux2ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("flux2");
  fd.as_object().insert_or_assign("klein_kv", FlexData::make_bool(_klein_kv));
  return fd;
}

VPIPE_REGISTER_STAGE(Flux2ModelConfigStage)
VPIPE_REGISTER_SPEC(Flux2ModelConfigStage, kSpec)

}  // namespace vpipe
