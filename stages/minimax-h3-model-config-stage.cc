#include "stages/minimax-h3-model-config-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-config-source.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "video_shift", .type = ConfigType::Real, .required = false,
   .doc = "sigma shift for the VIDEO schedule. 12.0 is the released "
          "checkpoint's; it is not interchangeable with the audio one",
   .def_real = 12.0},
  {.key = "audio_shift", .type = ConfigType::Real, .required = false,
   .doc = "sigma shift for the AUDIO schedule, stepped in lockstep with the "
          "video one over the same step count", .def_real = 3.0},
  {.key = "condition_timestep", .type = ConfigType::Real, .required = false,
   .doc = "the level the pinned keyframe rows are conditioned at; 1.0 is "
          "CLEAN in this model's t = 1 - sigma convention. Lower it only for "
          "a checkpoint trained with noise-augmented anchors",
   .def_real = 1.0},
  {.key = "condition_audio_timestep", .type = ConfigType::Real,
   .required = false,
   .doc = "the same, for a Ref2VA reference SOUNDTRACK's rows. Separate "
          "because a reference's frames and its audio are encoded by "
          "different VAEs and enter as different rows", .def_real = 1.0},
  {.key = "audio_seconds", .type = ConfigType::Real, .required = false,
   .doc = "soundtrack duration; 0 derives it from the video's frames / fps, "
          "which is what keeps the two modalities the same length by "
          "construction", .def_real = 0.0},
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
   .doc = "MiniMax-H3 generation parameters as one FlexData object "
          "{model_family: minimax-h3, video_shift, audio_shift, "
          "condition_timestep, condition_audio_timestep, audio_seconds}, for "
          "a generate-video model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "minimax-h3-model-config",
  .doc       = "Source: the MiniMax-H3-specific generation parameters (the "
               "two sigma shifts its video and audio schedules run at, the "
               "level its pinned condition rows sit at, and the soundtrack "
               "duration) as one FlexData beat for generate-video to latch. "
               "H3 is guidance-distilled, so there is deliberately no "
               "guidance scale here. One beat then done; with a trigger "
               "iport, one beat per inbound beat.",
  .display_name = "MiniMax-H3 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MiniMaxH3ModelConfigStage::MiniMaxH3ModelConfigStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : ModelConfigSourceStage<MiniMaxH3ModelConfigStage>(s, std::move(id),
                                                      std::move(iports),
                                                      std::move(config))
{
  _video_shift   = attr_real("video_shift");
  _audio_shift   = attr_real("audio_shift");
  _cond_timestep = attr_real("condition_timestep");
  _cond_audio_timestep = attr_real("condition_audio_timestep");
  _audio_seconds = attr_real("audio_seconds");
  // Deferred validation: the ctor never throws, so a nonsense number is
  // reported and the runtime skips the stage at launch. A non-positive
  // shift collapses the schedule to a single sigma, which generates
  // noise at full cost rather than failing.
  if (!(_video_shift > 0.0) || !(_audio_shift > 0.0)) {
    fail_config(fmt(
        "MiniMaxH3ModelConfigStage('{}'): the sigma shifts must be > 0 (got "
        "{:.3f} / {:.3f})", this->id(), _video_shift, _audio_shift));
  }
  auto bad_t = [](double v) { return v < 0.0 || v > 1.0; };
  if (bad_t(_cond_timestep) || bad_t(_cond_audio_timestep)) {
    fail_config(fmt(
        "MiniMaxH3ModelConfigStage('{}'): the condition timesteps are levels "
        "in t = 1 - sigma and must be in [0, 1] (got {:.3f} / {:.3f})",
        this->id(), _cond_timestep, _cond_audio_timestep));
  }
  if (_audio_seconds < 0.0) {
    fail_config(fmt(
        "MiniMaxH3ModelConfigStage('{}'): audio_seconds must be >= 0; 0 "
        "derives it from the video (got {:.3f})", this->id(),
        _audio_seconds));
  }
  allocate_oports(spec().oports.size());
}

const StageSpec&
MiniMaxH3ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
MiniMaxH3ModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("minimax-h3");
  auto o = fd.as_object();
  o.insert_or_assign("video_shift", FlexData::make_real(_video_shift));
  o.insert_or_assign("audio_shift", FlexData::make_real(_audio_shift));
  o.insert_or_assign("condition_timestep",
                     FlexData::make_real(_cond_timestep));
  o.insert_or_assign("condition_audio_timestep",
                     FlexData::make_real(_cond_audio_timestep));
  o.insert_or_assign("audio_seconds", FlexData::make_real(_audio_seconds));
  return fd;
}

void
MiniMaxH3ModelConfigStage::report_config(const FlexData& fd) const
{
  (void)fd;
  session()->info(fmt(
      "MiniMaxH3ModelConfigStage('{}'): shifts {:.1f}/{:.1f}, condition "
      "t {:.3f}/{:.3f}, audio {}", this->id(), _video_shift, _audio_shift,
      _cond_timestep, _cond_audio_timestep,
      _audio_seconds > 0.0 ? fmt("{:.3f} s", _audio_seconds)()
                           : std::string("as long as the video")));
}

VPIPE_REGISTER_STAGE(MiniMaxH3ModelConfigStage)
VPIPE_REGISTER_SPEC(MiniMaxH3ModelConfigStage, kSpec)

}  // namespace vpipe
