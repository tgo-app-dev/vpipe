#include "stages/scheduler-select-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "type", .type = ConfigType::String, .required = false,
   .doc = "schedule: simple (default) | karras | exponential | boogu_v1 "
          "(Boogu-Image's logistic time shift; its sigmas ASCEND -- 0 is noise, "
          "1 is clean -- so it only fits a Boogu DiT)"},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "override denoising steps (default 8)"},
  {.key = "shift", .type = ConfigType::Real, .required = false,
   .doc = "override mu / time-shift strength (default 1.15)"},
  {.key = "shift_type", .type = ConfigType::String, .required = false,
   .doc = "time-shift form: exponential (default) | linear"},
  {.key = "rho", .type = ConfigType::Real, .required = false,
   .doc = "karras curvature (default 7)"},
  {.key = "seq_len", .type = ConfigType::Int, .required = false,
   .doc = "boogu_v1 only: the STATIC token count its mu is read off, i.e. the "
          "checkpoint scheduler_config seq_len (default 4096 = 1K square)"},
  {.key = "base_shift", .type = ConfigType::Real, .required = false,
   .doc = "boogu_v1 only: mu at base_seq tokens (default 0.5)"},
  {.key = "max_shift", .type = ConfigType::Real, .required = false,
   .doc = "boogu_v1 only: mu at max_seq tokens (default 1.15)"},
};
const PortSpec kOports[] = {
  {.name = "scheduler",
   .doc = "scheduler spec {scheduler,type,steps,shift,shift_type,rho} "
          "(+ base_shift/max_shift/seq_len for boogu_v1)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "scheduler-select",
  .doc       = "Choose a diffusion sigma schedule (simple | karras | "
               "exponential | boogu_v1) + steps/shift and emit its spec as a "
               "FlexData beat "
               "for a text-to-image stage to latch. Model-agnostic (forwards the "
               "user's choice); config fields override the turbo defaults. Pairs "
               "with diffusion-sampler-select. 0 in / 1 out (emits once).",
  .display_name = "Scheduler Select",
  .category  = StageCategory::Generative,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

SchedulerSelectStage::SchedulerSelectStage(const SessionContextIntf* s,
                                           std::string               id,
                                           std::vector<InEdge>       iports,
                                           FlexData                  config)
  : TypedStage<SchedulerSelectStage>(s, std::move(id), std::move(iports),
                                     std::move(config))
{
  _type       = attr_str("type");
  _shift_type = attr_str("shift_type");
  _steps      = attr_int("steps");
  _shift      = attr_real("shift");
  _rho        = attr_real("rho");
  _seq_len    = attr_int("seq_len");
  _base_shift = attr_real("base_shift");
  _max_shift  = attr_real("max_shift");
  if (!_type.empty() && _type != "simple" && _type != "karras" &&
      _type != "exponential" && _type != "boogu_v1") {
    fail_config(fmt("SchedulerSelectStage('{}'): type must be simple | karras "
                    "| exponential | boogu_v1 (got \"{}\")", this->id(),
                    _type));
  }
  if (!_shift_type.empty() && _shift_type != "exponential" &&
      _shift_type != "linear") {
    fail_config(fmt("SchedulerSelectStage('{}'): shift_type must be exponential "
                    "| linear (got \"{}\")", this->id(), _shift_type));
  }
  allocate_oports(spec().oports.size());
}

SchedulerSelectStage::~SchedulerSelectStage() = default;

const StageSpec&
SchedulerSelectStage::spec() const noexcept
{
  return kSpec;
}

FlexData
SchedulerSelectStage::resolved_spec() const
{
  // Model-agnostic: this stage forwards the user's schedule choice and does NOT
  // read the model's scheduler config (the text-to-image stage owns the model).
  // The built-in distilled-turbo defaults apply unless a config field overrides.
  std::string type = "simple", shift_type = "exponential";
  std::int64_t steps = 8;
  double shift = 1.15, rho = 7.0;

  // Explicit config fields override the built-in defaults.
  if (!_type.empty()) { type = _type; }
  if (_steps > 0) { steps = _steps; }
  if (_shift > 0.0) { shift = _shift; }
  if (!_shift_type.empty()) { shift_type = _shift_type; }
  if (_rho > 0.0) { rho = _rho; }

  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("scheduler", FlexData::make_string("flow_match"));
  o.insert_or_assign("type", FlexData::make_string(type));
  o.insert_or_assign("steps", FlexData::make_int(steps));
  o.insert_or_assign("shift", FlexData::make_real(shift));
  o.insert_or_assign("shift_type", FlexData::make_string(shift_type));
  o.insert_or_assign("rho", FlexData::make_real(rho));
  if (type == "boogu_v1") {
    o.insert_or_assign("seq_len",
                       FlexData::make_int(_seq_len > 0 ? _seq_len : 4096));
    o.insert_or_assign("base_shift",
                       FlexData::make_real(_base_shift > 0.0 ? _base_shift
                                                             : 0.5));
    o.insert_or_assign("max_shift",
                       FlexData::make_real(_max_shift > 0.0 ? _max_shift
                                                            : 1.15));
    o.insert_or_assign("base_seq", FlexData::make_int(256));
    o.insert_or_assign("max_seq", FlexData::make_int(4096));
  }
  return fd;
}

void
SchedulerSelectStage::reset_run_state()
{
  // Per-launch reset. Stopping a pipeline destroys the RUNTIME, not the
  // stages: only unload / re-materialize destroys a Stage, so a plain
  // Stop-then-Start re-enters initialize() with this source already
  // exhausted from the previous run. Without this it would emit nothing
  // and signal done immediately, and every stage downstream of the
  // scheduler beat would sit idle while the pipeline "completed"
  // in milliseconds.
  _emitted = 0;
}

Job
SchedulerSelectStage::process(RuntimeContext& ctx)
{
  if (_emitted > 0) { ctx.signal_done(); co_return; }
  FlexData fd = resolved_spec();
  {
    auto o = fd.as_object();
    session()->info(fmt(
        "SchedulerSelectStage('{}'): scheduler = {} ({} steps, shift {} {}, "
        "rho {})", this->id(), std::string(o.at("type").as_string("")),
        o.at("steps").as_int(0), o.at("shift").as_real(0.0),
        std::string(o.at("shift_type").as_string("")),
        o.at("rho").as_real(0.0)));
  }
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  ++_emitted;
  ctx.signal_done();   // one-shot source: emit then close.
}

VPIPE_REGISTER_STAGE(SchedulerSelectStage)
VPIPE_REGISTER_SPEC(SchedulerSelectStage, kSpec)

}
