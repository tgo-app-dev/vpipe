#include "stages/z-image-model-config-stage.h"

#include "common/flex-data.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance, in THIS model's parametrization: "
          "pred = pos + g*(pos - neg), which is one LESS than the usual "
          "neg + g*(pos - neg). Unset => 0, which is the distilled "
          "Z-Image-Turbo's setting and turns CFG off entirely (one "
          "forward per step, no negative prompt). The undistilled "
          "Z-Image base wants 3-5 here, and 28-50 steps -- the two "
          "checkpoints ship identical transformer configs, so this key "
          "is the only thing that says which one is loaded",
   .def_real = 0.0},
  {.key = "cfg_normalization", .type = ConfigType::Real, .required = false,
   .doc = "clamp the guided prediction's norm to this multiple of the "
          "conditional one's, which bounds the over-saturation a high "
          "guidance produces. Unset => 0 (off), matching the reference. "
          "Only read when guidance_scale > 0",
   .def_real = 0.0},
  {.key = "cfg_truncation", .type = ConfigType::Real, .required = false,
   .doc = "drop guidance for the steps where (1 - sigma) exceeds this, "
          "i.e. the CLEAN end of the schedule, where CFG mostly adds "
          "artefacts. Unset => 1.0, which never fires. Only read when "
          "guidance_scale > 0",
   .def_real = 1.0},
};
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc = "OPTIONAL beat that gates re-emitting the config (a chrono tick, "
          "a prompt source, a feedback loop). Any payload -- receipt is "
          "the signal. Unwired, the stage emits once for the run",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "model_config",
   .doc = "z-image parameters as one FlexData object {model_family: "
          "z-image, +guidance_scale, +cfg_normalization, "
          "+cfg_truncation}, for a generate-image model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "z-image-model-config",
  .doc       = "Source: the Z-Image-specific parameters -- the guidance "
               "scale in this model's own parametrization, and the two "
               "refinements the reference applies on top of it. The "
               "guidance is also what distinguishes the distilled Turbo "
               "checkpoint (0) from the undistilled base (3-5), which "
               "nothing else can. One beat then done; with a trigger "
               "iport, one beat per inbound beat.",
  .display_name = "Z-Image Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

ZImageModelConfigStage::ZImageModelConfigStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : ModelConfigSourceStage<ZImageModelConfigStage>(s, std::move(id),
                                                   std::move(iports),
                                                   std::move(config))
{
  allocate_oports(spec().oports.size());
}

const StageSpec&
ZImageModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
ZImageModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("z-image");
  // Emitted ONLY where the graph set it: an unset key has to read
  // downstream as "no opinion" rather than as a deliberate 0, because
  // a beat that carries other keys and not this one is saying nothing
  // about guidance.
  if (config().is_object()) {
    auto in = config().as_object();
    auto carry = [&](const char* k) {
      if (!in.contains(k)) { return; }
      fd.as_object().insert_or_assign(
          k, FlexData::make_real(in.at(k).as_real(0.0)));
    };
    carry("guidance_scale");
    carry("cfg_normalization");
    carry("cfg_truncation");
  }
  return fd;
}

VPIPE_REGISTER_STAGE(ZImageModelConfigStage)
VPIPE_REGISTER_SPEC(ZImageModelConfigStage, kSpec)

}  // namespace vpipe
