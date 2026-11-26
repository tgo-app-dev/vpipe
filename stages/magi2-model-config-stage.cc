#include "stages/magi2-model-config-stage.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "denoising steps, each one batched forward of the conditional "
          "and negative passes (the release's 100 when unset; over "
          "generate-video's own steps when set)",
   .def_int = 100},
  {.key = "shift", .type = ConfigType::Real, .required = false,
   .doc = "the flow shift of the sampling grid (the release's 7 when "
          "unset): higher spends more of the steps at high noise",
   .def_real = 7.0},
  {.key = "video_guidance", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance of the video against the negative "
          "prompt (the release's 5 when unset; 1 runs one pass a step and "
          "no negative)",
   .def_real = 5.0},
  {.key = "audio_guidance", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance of the soundtrack (the release's 7 "
          "when unset)",
   .def_real = 7.0},
  {.key = "refine_width", .type = ConfigType::Int, .required = false,
   .doc = "REFINE the preview to this width (pixels; with refine_height): "
          "the release's 1080p preset, a 896 x 512 preview refined to "
          "1920 x 1088 by its refiner (refiner/), the clip at twice the "
          "preview's rate (25 fps, 2T - 1 latent frames). Floored to "
          "whole 16-pixel cells. Unset: the preview is the clip",
   .def_int = 0},
  {.key = "refine_height", .type = ConfigType::Int, .required = false,
   .doc = "the refine's height (pixels; with refine_width)",
   .def_int = 0},
  {.key = "refine_steps", .type = ConfigType::Int, .required = false,
   .doc = "the refine's denoising steps, each a conditional and a negative "
          "forward (the release's 5 when unset)",
   .def_int = 5},
  {.key = "refine_shift", .type = ConfigType::Real, .required = false,
   .doc = "the flow shift of the refine's grid (the release's 5 when unset)",
   .def_real = 5.0},
  {.key = "refine_guidance", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance of the refine against the negative "
          "prompt (the release's 2 when unset; 1 runs one forward a step)",
   .def_real = 2.0},
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
   .doc = "MAGI-2 parameters as one FlexData object {model_family: magi2, "
          "+steps, +shift, +video_guidance, +audio_guidance, "
          "+refine_width, +refine_height, +refine_steps, +refine_shift, "
          "+refine_guidance}, for a generate-video model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "magi2-model-config",
  .doc       = "Source: the MAGI-2-preview sampling parameters -- steps, the "
               "flow shift and the video and audio guidance scales -- and "
               "the refine to a larger clip at twice the rate (the "
               "release's 1080p preset). One beat then done; with a "
               "trigger iport, one beat per inbound beat.",
  .display_name = "MAGI-2 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Magi2ModelConfigStage::Magi2ModelConfigStage(const SessionContextIntf* s,
                                             std::string id,
                                             std::vector<InEdge> iports,
                                             FlexData config)
  : ModelConfigSourceStage<Magi2ModelConfigStage>(s, std::move(id),
                                                  std::move(iports),
                                                  std::move(config))
{
  allocate_oports(spec().oports.size());
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) { return; }
  auto o = cfg.as_object();
  if (o.contains("steps") && o.at("steps").as_int(0) < 1) {
    fail_config(fmt("Magi2ModelConfigStage('{}'): steps must be >= 1",
                    this->id()));
  }
  if (o.contains("shift") && o.at("shift").as_real(0.0) <= 0.0) {
    fail_config(fmt("Magi2ModelConfigStage('{}'): shift must be > 0",
                    this->id()));
  }
  for (const char* k :
       {"video_guidance", "audio_guidance", "refine_guidance"}) {
    if (o.contains(k) && o.at(k).as_real(0.0) < 1.0) {
      fail_config(fmt("Magi2ModelConfigStage('{}'): {} must be >= 1",
                      this->id(), k));
    }
  }
  if (o.contains("refine_width") != o.contains("refine_height")) {
    fail_config(fmt("Magi2ModelConfigStage('{}'): refine_width and "
                    "refine_height go together", this->id()));
  }
  for (const char* k : {"refine_width", "refine_height"}) {
    if (o.contains(k) && o.at(k).as_int(0) < 16) {
      fail_config(fmt("Magi2ModelConfigStage('{}'): {} must be >= 16",
                      this->id(), k));
    }
  }
  if (o.contains("refine_steps") && o.at("refine_steps").as_int(0) < 1) {
    fail_config(fmt("Magi2ModelConfigStage('{}'): refine_steps must be >= "
                    "1", this->id()));
  }
  if (o.contains("refine_shift") && o.at("refine_shift").as_real(0.0) <= 0) {
    fail_config(fmt("Magi2ModelConfigStage('{}'): refine_shift must be > 0",
                    this->id()));
  }
}

const StageSpec&
Magi2ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
Magi2ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("magi2");
  const FlexData& cfg = config();
  if (cfg.is_object()) {
    auto in = cfg.as_object();
    auto out = fd.as_object();
    for (const char* k :
         {"steps", "shift", "video_guidance", "audio_guidance",
          "refine_width", "refine_height", "refine_steps", "refine_shift",
          "refine_guidance"}) {
      if (in.contains(k)) { out.insert_or_assign(k, in.at(k)); }
    }
  }
  return fd;
}

VPIPE_REGISTER_STAGE(Magi2ModelConfigStage)
VPIPE_REGISTER_SPEC(Magi2ModelConfigStage, kSpec)

}  // namespace vpipe
