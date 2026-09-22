#include "stages/qwen-image-21-model-config-stage.h"

#include "common/flex-data.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "use_kv_cache", .type = ConfigType::Bool, .required = false,
   .doc = "reuse the text + condition-image prefix's attention keys and "
          "values across denoising steps. Sound only because this model "
          "modulates that prefix from t=0, which makes it independent of "
          "the step -- so it is an exact optimization, not an "
          "approximation, and the cached and uncached paths are "
          "bit-identical. Worth little on text-to-image (the prefix is a "
          "few dozen rows against thousands) and a great deal on a "
          "multi-reference edit, where the references ARE most of the "
          "sequence. Unset => on",
   .def_bool = true},
  {.key = "vl_long_edge", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: cap on the reference image's LONGEST side "
          "before the vision tower. Unset => 0 (none) -- this family "
          "bounds by AREA, not by a long edge, because its own pipeline "
          "resizes to a target area at a fixed aspect"},
  {.key = "vl_pixel_budget", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: cap on total PIXELS for a reference image. "
          "Unset => the family's own 1024x1024 (1048576), which is its "
          "pipeline's `output_resolution` squared. CHANGING THIS IS NOT "
          "FREE: the same resized pixels feed the VAE, and the DiT needs "
          "the tower's merged grid to be exactly half the VAE's latent "
          "grid, so a budget whose result does not align to 32 makes the "
          "reference unlayoutable",
   .def_uint = 0},
  {.key = "vl_min_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's lower bound, which is "
          "what makes a small or very wide reference get UPSCALED before "
          "patching. Unset => the family's own 65536"},
  {.key = "vl_max_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's upper bound. Unset => "
          "the family's own 16777216"},
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
   .doc = "qwen-image-2.1 parameters as one FlexData object "
          "{model_family: qwen-image-21, +use_kv_cache, +vl_*}, for a "
          "generate-image or diffusion-conditioner model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "qwen-image-21-model-config",
  .doc       = "Source: the Qwen-Image-2.1-specific parameters -- the "
               "cross-step prefix KV cache and the bounds on a reference "
               "image, which this model shares with the VAE rather than "
               "choosing freely. One beat then done; with a trigger "
               "iport, one beat per inbound beat.",
  .display_name = "Qwen-Image-2.1 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

QwenImage21ModelConfigStage::QwenImage21ModelConfigStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : ModelConfigSourceStage<QwenImage21ModelConfigStage>(s, std::move(id),
                                                        std::move(iports),
                                                        std::move(config))
{
  allocate_oports(spec().oports.size());
}

const StageSpec&
QwenImage21ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
QwenImage21ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("qwen-image-21");
  model_config::copy_grounded_keys(config(), fd);
  // Emitted ONLY when the graph set it. An unset key has to read
  // downstream as "no opinion" rather than as a deliberate `false`,
  // because the family's own default is `true` and the two are not the
  // same instruction.
  if (config().is_object()) {
    auto in = config().as_object();
    if (in.contains("use_kv_cache")) {
      fd.as_object().insert_or_assign(
          "use_kv_cache",
          FlexData::make_bool(in.at("use_kv_cache").as_bool(true)));
    }
  }
  return fd;
}

VPIPE_REGISTER_STAGE(QwenImage21ModelConfigStage)
VPIPE_REGISTER_SPEC(QwenImage21ModelConfigStage, kSpec)

}  // namespace vpipe
