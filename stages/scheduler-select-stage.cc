#include "stages/scheduler-select-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "model", .type = ConfigType::String, .required = false,
   .doc = "model dir/key to read scheduler defaults (scheduler_config.json)",
   .suggest_db = "models"},
  {.key = "type", .type = ConfigType::String, .required = false,
   .doc = "schedule: simple (default) | karras | exponential"},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "override denoising steps (default 8)"},
  {.key = "shift", .type = ConfigType::Real, .required = false,
   .doc = "override mu / time-shift strength (default 1.15)"},
  {.key = "shift_type", .type = ConfigType::String, .required = false,
   .doc = "time-shift form: exponential (default) | linear"},
  {.key = "rho", .type = ConfigType::Real, .required = false,
   .doc = "karras curvature (default 7)"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
};
const PortSpec kOports[] = {
  {.name = "scheduler",
   .doc = "scheduler spec {scheduler,type,steps,shift,shift_type,rho}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "scheduler-select",
  .doc       = "Choose a diffusion sigma schedule (simple | karras | "
               "exponential) + steps/shift and emit its spec as a FlexData beat "
               "for a text-to-image stage to latch. Defaults come from a model's "
               "scheduler config; config fields override. Pairs with "
               "sampler-select. 0 in / 1 out (emits once).",
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
  _model      = attr_str("model");
  _type       = attr_str("type");
  _shift_type = attr_str("shift_type");
  _models_db  = attr_str("models_db");
  _steps      = attr_int("steps");
  _shift      = attr_real("shift");
  _rho        = attr_real("rho");
  if (_models_db.empty()) { _models_db = "models"; }
  if (!_type.empty() && _type != "simple" && _type != "karras" &&
      _type != "exponential") {
    fail_config(fmt("SchedulerSelectStage('{}'): type must be simple | karras "
                    "| exponential (got \"{}\")", this->id(), _type));
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
  // Built-in distilled-turbo defaults.
  std::string type = "simple", shift_type = "exponential";
  std::int64_t steps = 8;
  double shift = 1.15, rho = 7.0;

  // Model scheduler config -> shift / shift_type defaults.
  if (!_model.empty()) {
    namespace fs = std::filesystem;
    const std::string dir = resolve_model_dir(session(), _models_db, _model);
    std::ifstream in(fs::path(dir) / "scheduler" / "scheduler_config.json");
    if (!in) { in.open((fs::path(dir) / "scheduler_config.json")); }
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("time_shift_type")) {
          const std::string ts(o.at("time_shift_type").as_string(""));
          shift_type = (ts == "linear") ? "linear" : "exponential";
        }
        // Distilled turbo pipelines use max_shift as the fixed mu when dynamic
        // shifting is enabled; otherwise the static `shift`.
        const bool dyn = o.contains("use_dynamic_shifting") &&
                         o.at("use_dynamic_shifting").as_bool(false);
        if (dyn && o.contains("max_shift")) {
          shift = o.at("max_shift").as_real(shift);
        } else if (o.contains("shift")) {
          shift = o.at("shift").as_real(shift);
        }
      }
    } else {
      session()->warn(fmt("SchedulerSelectStage('{}'): no scheduler config "
                          "under '{}'; using built-in defaults", this->id(),
                          dir));
    }
  }

  // Explicit config fields override the model-derived defaults.
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
  return fd;
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
