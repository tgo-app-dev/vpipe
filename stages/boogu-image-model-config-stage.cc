#include "stages/boogu-image-model-config-stage.h"

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
          "vl_long_edge. Unset => the family's own 384x384 (147456)"},
  {.key = "vl_min_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's lower bound, which is what "
          "makes a small or very wide reference get UPSCALED before patching. "
          "Unset => the family's own 65536"},
  {.key = "vl_max_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's upper bound. Unset => "
          "the family's own 16777216"},
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
   .doc = "boogu-image parameters as one FlexData object {model_family: "
          "boogu-image, +vl_*}, for a diffusion-conditioner "
          "model_config iport (the grounded encode)",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "boogu-image-model-config",
  .doc       = "Source: the Boogu-Image-specific parameters -- the two bounds "
               "its pipeline puts on a VLM conditioning image, which are not "
               "redundant with each other. One beat then done; with a trigger "
               "iport, one beat per inbound beat.",
  .display_name = "Boogu-Image Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

BooguImageModelConfigStage::BooguImageModelConfigStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : ModelConfigSourceStage<BooguImageModelConfigStage>(s, std::move(id),
                                                       std::move(iports),
                                                       std::move(config))
{
  allocate_oports(spec().oports.size());
}

const StageSpec&
BooguImageModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
BooguImageModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("boogu-image");
  model_config::copy_grounded_keys(config(), fd);
  return fd;
}

VPIPE_REGISTER_STAGE(BooguImageModelConfigStage)
VPIPE_REGISTER_SPEC(BooguImageModelConfigStage, kSpec)

}  // namespace vpipe
