#include "stages/krea2-model-config-stage.h"
#include "stages/model-registry.h"

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
  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x) rather than having the delta folded into "
          "the weights, which for a 4-bit base is the difference between "
          "keeping a small correction and rounding it away. A registered "
          "model, a directory holding one .safetensors, or a path to one. "
          "Both key conventions bind: this model's own diffusers names and "
          "the ai-toolkit / ComfyUI spelling. LOAD-TIME: it changes how the "
          "weights are BUILT (an adapted ff.gate/ff.up forbids the "
          "fused-SwiGLU weave), so a beat that changes it once the DiT is "
          "up is reported and ignored. Unset => no adapter",
   // A MODEL picker, not a file browser: an adapter is a catalogued
   // model here, and `krea2-lora` is what keeps a FLUX.2 or H3 adapter
   // out of the list. Free text still works, so a bare path is
   // unaffected -- see ConfigKey::suggest_db.
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "krea2-lora"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "the adapter's strength, applied PER FORWARD. Live: it rides the "
          "GEMM as a constant, so it can be swept without a reload. 1.0 = as "
          "trained; 0 skips both adapter GEMMs, so off is exactly off",
   .def_real = 1.0},
  {.key = "lora2", .type = ConfigType::String, .required = false,
   .doc = "a SECOND runtime LoRA, applied alongside the first: every "
          "adapted projection computes W x + s1 B1 (A1 x) + s2 B2 (A2 x). "
          "The pairing this exists for is an identity or few-step adapter "
          "in `lora` and a style one here, which is the one actually being "
          "dialled -- but nothing distinguishes the slots: either may hold "
          "either, and `lora2` alone is an ordinary request. Same accepted "
          "forms and same conventions as `lora`. Costs one more pair of "
          "skinny GEMMs; the base weight is still read once. LOAD-time "
          "like `lora`",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "krea2-lora"},
  {.key = "lora2_scale", .type = ConfigType::Real, .required = false,
   .doc = "`lora2`'s strength, per FORWARD and independent of "
          "`lora_scale` -- which is the point of a second slot: one "
          "adapter stays where it was trained while the other is swept. "
          "0 skips its two GEMMs", .def_real = 1.0},
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
   .doc = "krea2 parameters as one FlexData object {model_family: krea2, "
          "+vl_*, +lora, +lora2 and their scales}. The vl_* keys are "
          "for a "
          "diffusion-conditioner's model_config iport (the grounded "
          "encode); the lora keys are for generate-image's. Wire it to "
          "both -- each reads only what it owns",
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
  // The adapter. Written only when SET, on the same argument as the
  // grounded keys: a default emitted as though it were a choice would
  // read downstream as "the graph asked for scale 1.0" when the graph
  // said nothing at all.
  {
    auto in = config().as_object();
    auto o  = fd.as_object();
    for (const char* k : {"lora", "lora_scale",
                          "lora2", "lora2_scale"}) {
      if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
    }
  }
  return fd;
}

VPIPE_REGISTER_STAGE(Krea2ModelConfigStage)
VPIPE_REGISTER_SPEC(Krea2ModelConfigStage, kSpec)

}  // namespace vpipe
