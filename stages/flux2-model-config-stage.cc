#include "stages/flux2-model-config-stage.h"
#include "stages/model-registry.h"

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
  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x) rather than having the delta folded into "
          "the weights, which for a 4-bit base is the difference between "
          "keeping a small correction and rounding it away. A registered "
          "model, a directory holding one .safetensors, or a path to one. "
          "Module names must be the DIFFUSERS spelling the checkpoint "
          "itself uses (transformer_blocks.N.attn.to_q, "
          "single_transformer_blocks.N.attn.to_qkv_mlp_proj, ...), "
          "optionally under a `diffusion_model.` prefix; a BFL/ComfyUI "
          "repack that renames the modules binds nothing and says so. "
          "LOAD-time: an adapter on a fused projection turns the "
          "fused-SwiGLU weave off, so it is read before the DiT is built "
          "and a beat that changes it afterwards is reported and ignored",
   .suggest_db = kModelRegistryDb,
   // Named for the same reason H3's is: a typeless field falls back to
   // plain models and shows no adapter at all, and the type is what
   // keeps a Krea-2 or H3 adapter out of a FLUX.2 field.
   .suggest_db_type = "flux2-lora"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "the adapter's strength, applied PER FORWARD. Live: it rides the "
          "GEMM as a constant, so it can be swept without a reload. 1.0 = "
          "as trained; 0 skips both adapter GEMMs, so off is exactly off",
   .def_real = 1.0},
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
          "klein_kv, +lora, +lora_scale}, for a generate-image "
          "model_config iport",
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
  // The adapter, written only when SET. This family builds its beat by
  // hand, so declaring the keys above emits nothing on its own -- and a
  // default emitted as though it were a choice would read downstream as
  // "the graph asked for scale 1.0" when the graph said nothing at all.
  {
    auto in = config().as_object();
    auto o  = fd.as_object();
    for (const char* k : {"lora", "lora_scale"}) {
      if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
    }
  }
  return fd;
}

VPIPE_REGISTER_STAGE(Flux2ModelConfigStage)
VPIPE_REGISTER_SPEC(Flux2ModelConfigStage, kSpec)

}  // namespace vpipe
