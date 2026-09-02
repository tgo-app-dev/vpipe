#include "stages/minimax-h3-model-config-stage.h"
#include "stages/model-registry.h"

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
  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x) rather than having the delta folded into "
          "the weights. A registered model (the key a `model-fetch` of a "
          "Turbo adapter writes), a directory holding one .safetensors, "
          "or a direct path to one. Preferred over lora-fuse for a "
          "few-step adapter: the MiniMax-H3 Turbo update is 2-4e-4 "
          "relative to the weights where bf16's step is ~4e-3, so merging "
          "rounds most of it away (46-78% of the delta survives) and a "
          "4-bit base is coarser again. Costs ~1.5% of the projections' "
          "FLOPs. LOAD-time: it is read before the DiT is built, and a "
          "beat that changes it afterwards is reported and ignored",
   .suggest_db = kModelRegistryDb,
   // Without this the picker shows NOTHING: a field with no type falls
   // back to plain models only, and every LoRA is catalogued as a
   // `supplement` (it attaches to a parent rather than standing alone).
   // Naming the type is also what keeps a Krea-2 or Wan adapter out of
   // an H3 field.
   .suggest_db_type = "minimax-h3-lora"},
  {.key = "lora_qkv_layout", .type = ConfigType::String, .required = false,
   .doc = "which row order a FUSED attn.qkv_proj adapter is in. The two "
          "publishers group that projection's rows differently -- "
          "Comfy-Org's repack flat [all q | all k | all v], every "
          "MiniMaxAI release per head -- and this tree re-orders neither, "
          "so an adapter built for the other one lands on the wrong "
          "channels in every block. `auto` (the default) reads FLAT, "
          "which every published H3 adapter is, and permutes the rows on "
          "a per-head DiT so one adapter serves both checkpoints. "
          "`per_head` is the escape hatch for an adapter actually trained "
          "on MiniMaxAI's weights. Only the fused projection is affected",
   .def_str = "auto"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "adapter strength, per FORWARD. Live: a trigger-driven config "
          "can sweep it across beats without a reload, and 0 skips the "
          "adapter's two GEMMs so off is exactly off. 1.0 is as trained "
          "and is what the Turbo adapter is tuned for; nudge up for "
          "blurry ghosting, down for over-sharp grain", .def_real = 1.0},
  {.key = "lora2", .type = ConfigType::String, .required = false,
   .doc = "a SECOND runtime LoRA, applied alongside the first: every "
          "adapted projection computes W x + s1 B1 (A1 x) + s2 B2 (A2 x). "
          "The pairing this exists for is a few-step Turbo distillation "
          "in `lora` -- a correction to the whole model, belonging at the "
          "strength it was trained at -- and a style or identity adapter "
          "here, which is the one actually being dialled. Nothing "
          "distinguishes the slots though: either may hold either, and "
          "the strengths move independently. Same accepted forms as "
          "`lora`. Costs one more pair of skinny GEMMs, ~1.5% of a "
          "projection at rank 64; the base weight is still read once. "
          "LOAD-time like `lora`",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "minimax-h3-lora"},
  {.key = "lora2_qkv_layout", .type = ConfigType::String, .required = false,
   .doc = "the row order of `lora2`'s FUSED attn.qkv_proj, exactly as "
          "`lora_qkv_layout` is for `lora`. Per slot because the order is "
          "a property of the FILE: two adapters from different "
          "publishers need different answers",
   .def_str = "auto"},
  {.key = "lora2_scale", .type = ConfigType::Real, .required = false,
   .doc = "`lora2`'s strength, per FORWARD and independent of "
          "`lora_scale` -- which is the point of a second slot: the "
          "distillation stays where it was trained while this one is "
          "swept. 0 skips its two GEMMs", .def_real = 1.0},
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
          "condition_timestep, condition_audio_timestep, audio_seconds, "
          "+lora, +lora2 and their scales}, for a generate-video "
          "model_config iport",
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
  _lora          = attr_str("lora");
  _lora_scale    = attr_real("lora_scale");
  _lora_qkv      = attr_str("lora_qkv_layout");
  _lora2         = attr_str("lora2");
  _lora2_scale   = attr_real("lora2_scale");
  _lora2_qkv     = attr_str("lora2_qkv_layout");
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
  // Emitted only when SET. An empty `lora` key would read as "the graph
  // asked for no adapter", which is indistinguishable from "the graph
  // said nothing" at the consumer -- and the consumer's default is
  // already no adapter.
  if (!_lora.empty()) {
    o.insert_or_assign("lora", FlexData::make_string(_lora));
    o.insert_or_assign("lora_scale", FlexData::make_real(_lora_scale));
    if (!_lora_qkv.empty()) {
      o.insert_or_assign("lora_qkv_layout",
                         FlexData::make_string(_lora_qkv));
    }
  }
  // The second slot is independent of the first: `lora2` with no `lora`
  // is a graph with one style adapter and no distillation, which is a
  // perfectly ordinary request and not a hole to be filled.
  if (!_lora2.empty()) {
    o.insert_or_assign("lora2", FlexData::make_string(_lora2));
    o.insert_or_assign("lora2_scale", FlexData::make_real(_lora2_scale));
    if (!_lora2_qkv.empty()) {
      o.insert_or_assign("lora2_qkv_layout",
                         FlexData::make_string(_lora2_qkv));
    }
  }
  return fd;
}

void
MiniMaxH3ModelConfigStage::report_config(const FlexData& fd) const
{
  (void)fd;
  session()->info(fmt(
      "MiniMaxH3ModelConfigStage('{}'): shifts {:.1f}/{:.1f}, condition "
      "t {:.3f}/{:.3f}, audio {}{}{}", this->id(), _video_shift,
      _audio_shift,
      _cond_timestep, _cond_audio_timestep,
      _audio_seconds > 0.0 ? fmt("{:.3f} s", _audio_seconds)()
                           : std::string("as long as the video"),
      _lora.empty() ? std::string()
                    : fmt(", runtime LoRA at {:.2f}", _lora_scale)(),
      _lora2.empty() ? std::string()
                     : fmt(", second at {:.2f}", _lora2_scale)()));
}

VPIPE_REGISTER_STAGE(MiniMaxH3ModelConfigStage)
VPIPE_REGISTER_SPEC(MiniMaxH3ModelConfigStage, kSpec)

}  // namespace vpipe
