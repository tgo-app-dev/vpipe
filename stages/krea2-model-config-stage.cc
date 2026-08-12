#include "stages/krea2-model-config-stage.h"

#include "common/flex-data.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "vl_long_edge", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: cap on the reference image's LONGEST side before "
          "the vision tower. Unset => 768"},
  {.key = "vl_pixel_budget", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: cap on total PIXELS, applied together with "
          "vl_long_edge. Unset => none for this family"},
  {.key = "vl_min_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's lower bound, which is what "
          "makes a small or very wide reference get UPSCALED before patching. "
          "Unset => the tower default"},
  {.key = "vl_max_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's upper bound. Unset => "
          "the tower default"},
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
   .doc = "krea2 parameters as one FlexData object {model_family: "
          "krea2, +vl_*}, for a diffusion-conditioner model_config iport "
          "(the grounded encode)",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "krea2-model-config",
  .doc       = "Source: the Krea-2-specific parameters -- the resolution its "
               "grounded edit encodes a source image at, which is the "
               "identity-edit LoRA's training distribution rather than a free "
               "choice. One beat then done; with a trigger iport, one beat per "
               "inbound beat.",
  .display_name = "Krea-2 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Krea2ModelConfigStage::Krea2ModelConfigStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : ModelConfigSourceStage<Krea2ModelConfigStage>(s, std::move(id),
                                                  std::move(iports),
                                                  std::move(config))
{
  allocate_oports(spec().oports.size());
}

const StageSpec&
Krea2ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
Krea2ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("krea2");
  model_config::copy_grounded_keys(config(), fd);
  return fd;
}

VPIPE_REGISTER_STAGE(Krea2ModelConfigStage)
VPIPE_REGISTER_SPEC(Krea2ModelConfigStage, kSpec)

}  // namespace vpipe
