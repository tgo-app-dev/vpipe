#include "stages/fibo2-model-config-stage.h"
#include "stages/latent-preview.h"
#include "stages/lora-slots.h"
#include "stages/model-registry.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "vl_long_edge", .type = ConfigType::Int, .required = false,
   .doc = "edit: the longest side Qwen3-VL sees a picture to edit at, "
          "thumbnailed as the reference pipeline does (bicubic, aspect "
          "kept, never enlarged); the VAE takes the picture itself. Unset "
          "=> the family's own 768"},
  {.key = "vl_min_pixels", .type = ConfigType::Int, .required = false,
   .doc = "edit: the image processor's lower bound on a thumbnail's pixels "
          "(it is enlarged to reach it). Unset => the checkpoint's own 3136"},
  {.key = "vl_max_pixels", .type = ConfigType::Int, .required = false,
   .doc = "edit: the image processor's upper bound on a thumbnail's pixels. "
          "Unset => the checkpoint's own 1605632"},
  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x). A registered model, a directory holding "
          "one .safetensors, or a path to one. Module names are the "
          "checkpoint's own (noise_refiner.N.attention.to_q, layers.N."
          "feed_forward.w1, ...), optionally under a `diffusion_model.` or "
          "`transformer.` prefix; an adapter for another model binds "
          "nothing and says so. LOAD-time: read before the DiT is built, so "
          "a beat that changes it afterwards is reported and ignored (the "
          "strength stays live)",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "fibo2-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "the adapter's strength, applied PER FORWARD: live, so it can be "
          "swept without a reload. 1.0 = as trained; 0 skips both adapter "
          "GEMMs, so off is exactly off",
   .def_real = 1.0},
  lora_slots::path_key(1, "fibo2-lora"), lora_slots::scale_key(1),
  lora_slots::path_key(2, "fibo2-lora"), lora_slots::scale_key(2),
  lora_slots::path_key(3, "fibo2-lora"), lora_slots::scale_key(3),
  lora_slots::path_key(4, "fibo2-lora"), lora_slots::scale_key(4),
  lora_slots::path_key(5, "fibo2-lora"), lora_slots::scale_key(5),
  lora_slots::path_key(6, "fibo2-lora"), lora_slots::scale_key(6),
  lora_slots::path_key(7, "fibo2-lora"), lora_slots::scale_key(7),
  // The live-preview keys every family's config source shares; see
  // stages/latent-preview.h. Fibo 2.0 decodes through FLUX.2's VAE, so
  // FLUX.2's TAE (madebyollin's taef2) previews it unchanged.
  {.key = latent_preview::kVaeKey, .type = ConfigType::String,
   .required = false, .doc = latent_preview::kVaeDoc,
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "flux2-tae"},
  {.key = latent_preview::kEveryKey, .type = ConfigType::Int,
   .required = false, .doc = latent_preview::kEveryDoc, .def_int = 1},
  {.key = latent_preview::kMaxEdgeKey, .type = ConfigType::Int,
   .required = false, .doc = latent_preview::kMaxEdgeDoc, .def_int = 512},
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
   .doc = "Fibo 2.0 parameters as one FlexData object {model_family: "
          "fibo2, +vl_long_edge / vl_min_pixels / vl_max_pixels, +lora .. "
          "+lora8 and their scales, +preview_vae and its knobs}, for a "
          "generate-image or diffusion-conditioner model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "fibo2-model-config",
  .doc       = "Source: the Fibo 2.0 parameters -- the bounds of the "
               "pictures Qwen3-VL reads for an edit, runtime LoRA slots and "
               "the live preview. None is needed for an ordinary run. One "
               "beat then done; with a trigger iport, one beat per inbound "
               "beat.",
  .display_name = "Fibo 2.0 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Fibo2ModelConfigStage::Fibo2ModelConfigStage(const SessionContextIntf* s,
                                             std::string               id,
                                             std::vector<InEdge>       iports,
                                             FlexData                  config)
  : ModelConfigSourceStage<Fibo2ModelConfigStage>(s, std::move(id),
                                                  std::move(iports),
                                                  std::move(config))
{
  for (const char* k : {"vl_long_edge", "vl_min_pixels", "vl_max_pixels"}) {
    if (attr_int(k) < 0) {
      fail_config(fmt("Fibo2ModelConfigStage('{}'): {} must be >= 0 (got "
                      "{})", this->id(), k, attr_int(k)));
    }
  }
  _preview.vae      = attr_str(latent_preview::kVaeKey);
  _preview.every    = (int)attr_int(latent_preview::kEveryKey);
  _preview.max_edge = (int)attr_int(latent_preview::kMaxEdgeKey);
  if (_preview.every < 0 || _preview.max_edge < 0) {
    fail_config(fmt(
        "Fibo2ModelConfigStage('{}'): preview_every and "
        "preview_max_edge must be >= 0 (got {} / {})", this->id(),
        _preview.every, _preview.max_edge));
  }
  allocate_oports(spec().oports.size());
}

const StageSpec&
Fibo2ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
Fibo2ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("fibo2");
  // Each key written only when SET: a default emitted as though it were
  // a choice reads downstream as "the graph asked for it".
  model_config::copy_grounded_keys(config(), fd);
  {
    auto in = config().as_object();
    auto o  = fd.as_object();
    for (int i = 0; i < lora_slots::kSlots; ++i) {
      for (std::string_view k :
           {lora_slots::kPath[i], lora_slots::kScale[i]}) {
        if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
      }
    }
  }
  _preview.emit(fd);
  return fd;
}

VPIPE_REGISTER_STAGE(Fibo2ModelConfigStage)
VPIPE_REGISTER_SPEC(Fibo2ModelConfigStage, kSpec)

}  // namespace vpipe
