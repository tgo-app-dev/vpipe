#include "stages/sampler-select-stage.h"

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

// Canonicalize a sampler method (accepting the "dpm++_*" spelling); returns ""
// if unknown. Kept in sync with genai::FlowSamplerSpec::canon_method, but this
// stage stays backend-agnostic (no genai dependency) so it builds everywhere.
std::string
canon_sampler_method(const std::string& m)
{
  if (m == "euler" || m == "heun" || m == "dpmpp_2m" || m == "dpmpp_sde") {
    return m;
  }
  if (m == "dpm++_2m" || m == "dpmpp2m") { return "dpmpp_2m"; }
  if (m == "dpm++_sde" || m == "dpmppsde") { return "dpmpp_sde"; }
  return {};
}

const ConfigKey kAttrs[] = {
  {.key = "model", .type = ConfigType::String, .required = false,
   .doc = "model dir/key to read the scheduler default method",
   .suggest_db = "models"},
  {.key = "method", .type = ConfigType::String, .required = false,
   .doc = "sampler method: euler (default) | heun | dpmpp_2m | dpmpp_sde"},
  {.key = "eta", .type = ConfigType::Real, .required = false,
   .doc = "dpmpp_sde stochasticity, 0 = deterministic (default 1.0)"},
  {.key = "s_noise", .type = ConfigType::Real, .required = false,
   .doc = "dpmpp_sde added-noise scale (default 1.0)"},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "dpmpp_sde noise seed (default 0)"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
};
const PortSpec kOports[] = {
  {.name = "sampler",
   .doc = "sampler spec {sampler,method,eta,s_noise,seed}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "sampler-select",
  .doc       = "Choose a diffusion sampler/integrator (euler | heun | dpmpp_2m "
               "| dpmpp_sde) and emit its spec as a FlexData beat for a "
               "text-to-image stage to latch. Pairs with scheduler-select. "
               "0 in / 1 out (emits once).",
  .display_name = "Sampler Select",
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
  _model     = attr_str("model");
  _method    = attr_str("method");
  _models_db = attr_str("models_db");
  if (_models_db.empty()) { _models_db = "models"; }
  auto o = this->config().as_object();
  if (o.contains("eta")) { _eta = attr_real("eta"); _eta_set = true; }
  if (o.contains("s_noise")) {
    _s_noise = attr_real("s_noise"); _s_noise_set = true;
  }
  if (o.contains("seed")) { _seed = attr_int("seed"); _seed_set = true; }
  if (!_method.empty()) {
    const std::string c = canon_sampler_method(_method);
    if (c.empty()) {
      fail_config(fmt("SamplerSelectStage('{}'): method must be euler | heun | "
                      "dpmpp_2m | dpmpp_sde (got \"{}\")", this->id(), _method));
    }
    _method = c;
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
  std::string method = "euler";

  // Model scheduler config -> default method (_class_name).
  if (!_model.empty()) {
    namespace fs = std::filesystem;
    const std::string dir = resolve_model_dir(session(), _models_db, _model);
    std::ifstream in(fs::path(dir) / "scheduler" / "scheduler_config.json");
    if (!in) { in.open((fs::path(dir) / "scheduler_config.json")); }
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("_class_name")) {
          const std::string cn(o.at("_class_name").as_string(""));
          if (cn.find("Heun") != std::string::npos) { method = "heun"; }
          else if (cn.find("Euler") != std::string::npos) { method = "euler"; }
        }
      }
    }
  }

  // Explicit config overrides (method already canonicalized in the ctor).
  if (!_method.empty()) { method = _method; }
  const double eta = _eta_set ? _eta : 1.0;
  const double s_noise = _s_noise_set ? _s_noise : 1.0;
  const std::int64_t seed = _seed_set ? _seed : 0;

  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("sampler", FlexData::make_string("flow_match"));
  o.insert_or_assign("method", FlexData::make_string(method));
  o.insert_or_assign("eta", FlexData::make_real(eta));
  o.insert_or_assign("s_noise", FlexData::make_real(s_noise));
  o.insert_or_assign("seed", FlexData::make_int(seed));
  return fd;
}

Job
SamplerSelectStage::process(RuntimeContext& ctx)
{
  if (_emitted > 0) { ctx.signal_done(); co_return; }
  FlexData fd = resolved_spec();
  {
    auto o = fd.as_object();
    session()->info(fmt(
        "SamplerSelectStage('{}'): sampler = {} (eta {}, s_noise {}, seed {})",
        this->id(), std::string(o.at("method").as_string("")),
        o.at("eta").as_real(0.0), o.at("s_noise").as_real(0.0),
        o.at("seed").as_int(0)));
  }
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  ++_emitted;
  ctx.signal_done();   // one-shot source: emit then close.
}

VPIPE_REGISTER_STAGE(SamplerSelectStage)
VPIPE_REGISTER_SPEC(SamplerSelectStage, kSpec)

}
