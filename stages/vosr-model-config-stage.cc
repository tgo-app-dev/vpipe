#include "stages/vosr-model-config-stage.h"

#include "common/flex-data.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "tile_size", .type = ConfigType::Int, .required = false,
   .doc = "tile the restorer in LATENT space, this many output PIXELS per "
          "tile side (rounded to a multiple of 16). UNSET tiles at the "
          "resolution the checkpoint was distilled at whenever the output "
          "is larger, which is what the reference does and what keeps "
          "every tile on the token grid the weights know. 0 forces ONE "
          "pass: correct at that resolution, and above it a grid the model "
          "never saw, which weaves a periodic pattern through faces",
   .def_int = 0},
  {.key = "tile_overlap", .type = ConfigType::Int, .required = false,
   .doc = "overlap between tiles, in output pixels; raised to an eighth of "
          "the tile when smaller. Overlapping tiles are Gaussian-blended "
          "and share one noise field, so this trades work for seam-free "
          "blending. Ignored when tile_size is 0",
   .def_int = 32},
  {.key = "color_fix", .type = ConfigType::String, .required = false,
   .doc = "wavelet | adain | none: the colour correction the release's "
          "script applies to its output against the upscaled picture it "
          "restored (its --align_method). `wavelet` (default, as there): "
          "the restored picture keeps its own detail and takes its low "
          "frequencies -- a Gaussian blur at sigma 5 -- from the "
          "picture. `adain`: each channel's mean and spread moved to the "
          "picture's. `none`: the decode as it is (the script's nofix). "
          "Needs the picture on vae-decode's color_reference input -- the "
          "same pixels the restorer saw",
   .def_str = "wavelet"},
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
   .doc = "VOSR parameters as one FlexData object {model_family: vosr, "
          "+tile_size, +tile_overlap, +color_fix}, for a generate-image "
          "model_config iport (the restorer's tiling and the colour fix "
          "its decode takes)",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "vosr-model-config",
  .doc       = "Source: the VOSR-specific parameters -- how the one-step "
               "restorer is split over a picture larger than the grid its "
               "weights were distilled at, and how its output's colour is "
               "corrected against the picture. One beat then done; with a "
               "trigger iport, one beat per inbound beat.",
  .display_name = "VOSR Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

VosrModelConfigStage::VosrModelConfigStage(const SessionContextIntf* s,
                                           std::string               id,
                                           std::vector<InEdge>       iports,
                                           FlexData                  config)
  : ModelConfigSourceStage<VosrModelConfigStage>(s, std::move(id),
                                                 std::move(iports),
                                                 std::move(config))
{
  allocate_oports(spec().oports.size());
}

const StageSpec&
VosrModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
VosrModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("vosr");
  // ONLY WHAT THE GRAPH SET. Emitting a default as though it were a
  // choice is the one thing this must not do: `tile_size` unset means
  // "the trained grid" and `tile_size` 0 means "one pass whatever the
  // size", and a source that always wrote a number would turn the first
  // into the second for every graph that never touched the key.
  const FlexData& cfg = config();
  if (cfg.is_object()) {
    auto in = cfg.as_object();
    auto out = fd.as_object();
    for (const char* k : {"tile_size", "tile_overlap", "color_fix"}) {
      if (in.contains(k)) { out.insert_or_assign(k, in.at(k)); }
    }
  }
  return fd;
}

VPIPE_REGISTER_STAGE(VosrModelConfigStage)
VPIPE_REGISTER_SPEC(VosrModelConfigStage, kSpec)

}  // namespace vpipe
