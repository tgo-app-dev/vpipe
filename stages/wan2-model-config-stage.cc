#include "stages/wan2-model-config-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-config-source.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance; on a two-expert checkpoint (A14B) the "
          "HIGH-noise expert's. Wan is not distilled, so this is what steers "
          "it toward the prompt -- but it needs a negative conditioning wired "
          "into generate-video, which without one forces guidance to 1",
   .def_real = 3.5},
  {.key = "guidance_scale_2", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance for the LOW-noise expert (two-expert "
          "checkpoints, e.g. A14B). Separate from the first because the two "
          "experts are trained over different sigma ranges",
   .def_real = 3.5},
  {.key = "boundary_ratio", .type = ConfigType::Real, .required = false,
   .doc = "the sigma at which the low-noise expert takes over from the "
          "high-noise one; 0 means a single expert. OMIT IT to let the "
          "checkpoint's own model_index.json decide, which is right for "
          "every stock checkpoint -- setting it here overrides the file",
   .def_real = 0.9},
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
   .doc = "Wan generation parameters as one FlexData object "
          "{model_family: wan, guidance_scale, guidance_scale_2, "
          "+boundary_ratio}, for a generate-video model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "wan2-model-config",
  .doc       = "Source: the Wan-specific generation parameters (classifier-"
               "free guidance, and the two-expert boundary A14B switches at) "
               "as one FlexData beat for generate-video to latch. The family "
               "that needs them is the family that carries them, so a graph "
               "shows which checkpoint it is built for. One beat then done; "
               "with a trigger iport, one beat per inbound beat.",
  .display_name = "Wan 2.x Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Wan2ModelConfigStage::Wan2ModelConfigStage(const SessionContextIntf* s,
                                           std::string               id,
                                           std::vector<InEdge>       iports,
                                           FlexData                  config)
  : ModelConfigSourceStage<Wan2ModelConfigStage>(s, std::move(id),
                                                 std::move(iports),
                                                 std::move(config))
{
  _guidance   = attr_real("guidance_scale");
  _guidance_2 = attr_real("guidance_scale_2");
  // PRESENCE, not value: `boundary_ratio` is the one key whose default
  // must not be emitted, because the consumer falls back to the
  // checkpoint's model_index.json and an unmentioned key has to leave
  // that fallback alone.
  auto o = this->config().as_object();
  if (o.contains("boundary_ratio")) {
    _boundary     = attr_real("boundary_ratio");
    _boundary_set = true;
  }
  // Deferred validation: the ctor never throws, so a nonsense number is
  // reported and the runtime skips the stage at launch.
  if (_guidance < 1.0 || _guidance_2 < 1.0) {
    fail_config(fmt(
        "Wan2ModelConfigStage('{}'): guidance scales must be >= 1 (got "
        "{:.3f} / {:.3f}); below 1 points away from the prompt",
        this->id(), _guidance, _guidance_2));
  }
  if (_boundary_set && (_boundary < 0.0 || _boundary > 1.0)) {
    fail_config(fmt(
        "Wan2ModelConfigStage('{}'): boundary_ratio is a sigma and must be "
        "in [0, 1] (got {:.3f})", this->id(), _boundary));
  }
  allocate_oports(spec().oports.size());
}

const StageSpec&
Wan2ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
Wan2ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("wan");
  auto o = fd.as_object();
  o.insert_or_assign("guidance_scale", FlexData::make_real(_guidance));
  o.insert_or_assign("guidance_scale_2", FlexData::make_real(_guidance_2));
  if (_boundary_set) {
    o.insert_or_assign("boundary_ratio", FlexData::make_real(_boundary));
  }
  return fd;
}

void
Wan2ModelConfigStage::report_config(const FlexData& fd) const
{
  (void)fd;
  session()->info(fmt(
      "Wan2ModelConfigStage('{}'): guidance {:.2f} / {:.2f}{}", this->id(),
      _guidance, _guidance_2,
      _boundary_set ? fmt(", boundary {:.3f}", _boundary)()
                    : std::string(", boundary from the checkpoint")));
}

VPIPE_REGISTER_STAGE(Wan2ModelConfigStage)
VPIPE_REGISTER_SPEC(Wan2ModelConfigStage, kSpec)

}  // namespace vpipe
