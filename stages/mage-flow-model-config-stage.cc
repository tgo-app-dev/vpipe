#include "stages/mage-flow-model-config-stage.h"

#include "common/flex-data.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "no_watermark", .type = ConfigType::Bool, .required = false,
   .doc = "DISABLE the Gaussian-Shading provenance watermark in the initial "
          "noise. The watermark is ON by default -- the reference applies it "
          "unconditionally and it is distribution-preserving, so it costs no "
          "image quality. Negative-named so the safe default needs no config. "
          "Ignored by any run pinned to init_latents"},
  {.key = "watermark_key", .type = ConfigType::String, .required = false,
   .doc = "Gaussian-Shading key: an integer or a passphrase. Unset => "
          "$MAGEFLOW_GS_KEY, else $MAGEFLOW_GS_KEY_FILE / ~/.mageflow/gs_key, "
          "else the published default. The detector needs the SAME key"},
  {.key = "vl_long_edge", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: cap on the reference image's LONGEST side before "
          "the vision tower. Unset => the family's own 384 (pipeline.py "
          "vl_cond_long_edge). Raising it leaves the grounding LoRA's "
          "training distribution"},
  {.key = "vl_pixel_budget", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: cap on total PIXELS, applied with vl_long_edge. "
          "Unset => none for this family"},
  {.key = "vl_min_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's lower bound, which is what "
          "makes a small or very wide reference get UPSCALED before patching. "
          "Unset => the family's own 65536 (preprocessor_config.json "
          "shortest_edge), far above the Qwen default of 3136"},
  {.key = "vl_max_pixels", .type = ConfigType::Int, .required = false,
   .doc = "grounded encode: the image processor's upper bound. Unset => the "
          "family's own 16777216"},
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
   .doc = "Mage-Flow parameters as one FlexData object {model_family: "
          "mage-flow, no_watermark, watermark_key, +vl_*}. Wire it to BOTH "
          "generate-image (the watermark) and diffusion-conditioner (the "
          "grounded encode) -- one checkpoint, one source",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "mage-flow-model-config",
  .doc       = "Source: the Mage-Flow-specific parameters -- its provenance "
               "watermark (unique to this family) and the resolution its "
               "reference pipeline grounds a conditioning image at. One beat "
               "then done; with a trigger iport, one beat per inbound beat.",
  .display_name = "Mage-Flow Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MageFlowModelConfigStage::MageFlowModelConfigStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : ModelConfigSourceStage<MageFlowModelConfigStage>(s, std::move(id),
                                                     std::move(iports),
                                                     std::move(config))
{
  _no_watermark  = attr_bool("no_watermark");
  _watermark_key = attr_str("watermark_key");
  allocate_oports(spec().oports.size());
}

const StageSpec&
MageFlowModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
MageFlowModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("mage-flow");
  auto o = fd.as_object();
  o.insert_or_assign("no_watermark", FlexData::make_bool(_no_watermark));
  // The KEY is forwarded verbatim, including empty: empty is not "no
  // key", it is "let resolve_key()'s environment precedence decide", and
  // that has to be resolved where the noise is built rather than here.
  o.insert_or_assign("watermark_key",
                     FlexData::make_string(_watermark_key));
  model_config::copy_grounded_keys(config(), fd);
  return fd;
}

VPIPE_REGISTER_STAGE(MageFlowModelConfigStage)
VPIPE_REGISTER_SPEC(MageFlowModelConfigStage, kSpec)

}  // namespace vpipe
