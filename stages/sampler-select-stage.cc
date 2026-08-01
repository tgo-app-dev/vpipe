#include "stages/sampler-select-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "temperature", .type = ConfigType::Real, .required = false,
   .doc = "softmax temperature; <= 0 forces argmax (greedy)",
   .def_real = 1.0},
  {.key = "top_k", .type = ConfigType::Int, .required = false,
   .doc = "keep only the top-k logits; 0 = disabled", .def_int = 0},
  {.key = "top_p", .type = ConfigType::Real, .required = false,
   .doc = "nucleus sampling: smallest prefix summing to p; 1.0 = disabled",
   .def_real = 1.0},
  {.key = "min_p", .type = ConfigType::Real, .required = false,
   .doc = "drop tokens below min_p * max_prob; 0 = disabled", .def_real = 0.0},
  {.key = "repetition_penalty", .type = ConfigType::Real, .required = false,
   .doc = "penalise already-seen tokens; 1.0 = disabled", .def_real = 1.0},
  {.key = "presence_penalty", .type = ConfigType::Real, .required = false,
   .doc = "flat penalty on already-seen tokens; 0.0 = disabled",
   .def_real = 0.0},
  {.key = "seed", .type = ConfigType::Uint, .required = false,
   .doc = "sampling RNG seed; 0 = fresh non-deterministic seed",
   .def_uint = 0},
};
const PortSpec kOports[] = {
  {.name = "sampler",
   .doc = "token-sampler spec {sampler:\"token\",temperature,top_k,top_p,"
          "min_p,repetition_penalty,presence_penalty,seed}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "sampler-select",
  .doc       = "Program the LLM token sampler (temperature / top_k / top_p / "
               "min_p / penalties / seed) and emit its spec as a FlexData "
               "beat for a text-chat, visual-qa, realtime-vqa, "
               "audio-transcribe or text-to-speech stage to latch. Every "
               "knob at its default is "
               "argmax, so configure at least temperature to sample. For the "
               "diffusion integrator see diffusion-sampler-select instead. "
               "0 in / 1 out (emits once).",
  .display_name = "Sampler",
  .category  = StageCategory::Generative,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

SamplerSelectStage::SamplerSelectStage(const SessionContextIntf* s,
                                       std::string               id,
                                       std::vector<InEdge>       iports,
                                       FlexData                  config)
  : TypedStage<SamplerSelectStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  // attr_* fall back to the ConfigKey defaults above, which are exactly
  // genai::SamplerParams's defaults -- i.e. greedy.
  _temperature        = attr_real("temperature");
  _top_k              = attr_int("top_k");
  _top_p              = attr_real("top_p");
  _min_p              = attr_real("min_p");
  _repetition_penalty = attr_real("repetition_penalty");
  _presence_penalty   = attr_real("presence_penalty");
  _seed               = attr_uint("seed");

  if (_top_k < 0) {
    fail_config(fmt("SamplerSelectStage('{}'): top_k must be >= 0 (got {})",
                    this->id(), _top_k));
  }
  if (_top_p < 0.0 || _top_p > 1.0) {
    fail_config(fmt("SamplerSelectStage('{}'): top_p must be in [0, 1] "
                    "(got {})", this->id(), _top_p));
  }
  if (_min_p < 0.0 || _min_p > 1.0) {
    fail_config(fmt("SamplerSelectStage('{}'): min_p must be in [0, 1] "
                    "(got {})", this->id(), _min_p));
  }
  if (_repetition_penalty <= 0.0) {
    fail_config(fmt("SamplerSelectStage('{}'): repetition_penalty must be > 0 "
                    "(got {})", this->id(), _repetition_penalty));
  }
  allocate_oports(spec().oports.size());
}

SamplerSelectStage::~SamplerSelectStage() = default;

const StageSpec&
SamplerSelectStage::spec() const noexcept
{
  return kSpec;
}

FlexData
SamplerSelectStage::resolved_spec() const
{
  // Model-agnostic: every knob is the operator's choice, and the defaults are
  // genai::SamplerParams's own, so an unconfigured stage emits the greedy
  // spec. The "sampler" discriminator lets a consumer tell this apart from a
  // diffusion-sampler-select beat (which tags itself "flow_match") -- both
  // travel as FlexDataPayload, so the port type alone can't catch a swap.
  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("sampler", FlexData::make_string("token"));
  o.insert_or_assign("temperature", FlexData::make_real(_temperature));
  o.insert_or_assign("top_k", FlexData::make_int(_top_k));
  o.insert_or_assign("top_p", FlexData::make_real(_top_p));
  o.insert_or_assign("min_p", FlexData::make_real(_min_p));
  o.insert_or_assign("repetition_penalty",
                     FlexData::make_real(_repetition_penalty));
  o.insert_or_assign("presence_penalty",
                     FlexData::make_real(_presence_penalty));
  o.insert_or_assign("seed", FlexData::make_uint(_seed));
  return fd;
}

void
SamplerSelectStage::reset_run_state()
{
  // Per-launch reset. Stopping a pipeline destroys the RUNTIME, not the
  // stages: only unload / re-materialize destroys a Stage, so a plain
  // Stop-then-Start re-enters initialize() with this source already
  // exhausted from the previous run. Without this it would emit nothing
  // and signal done immediately, and every stage downstream of the sampler beat
  // would sit idle while the pipeline "completed" in milliseconds.
  _emitted = 0;
}

Job
SamplerSelectStage::process(RuntimeContext& ctx)
{
  if (_emitted > 0) { ctx.signal_done(); co_return; }
  FlexData fd = resolved_spec();
  session()->info(fmt(
      "SamplerSelectStage('{}'): temperature {}, top_k {}, top_p {}, min_p {}, "
      "rep {}, presence {}, seed {}", this->id(), _temperature, _top_k, _top_p,
      _min_p, _repetition_penalty, _presence_penalty, _seed));
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  ++_emitted;
  ctx.signal_done();   // one-shot source: emit then close.
}

VPIPE_REGISTER_STAGE(SamplerSelectStage)
VPIPE_REGISTER_SPEC(SamplerSelectStage, kSpec)

}
