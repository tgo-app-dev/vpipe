#include "stages/model-select-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "the model dir/registry key shared by the diffusion-conditioner, "
          "text-to-image (DiT), vae-encode and vae-decode stages; emitted as a "
          "beat that overrides each of their hf_dir config keys",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type =
       "krea2,flux2,qwen-image-edit,mage-flow,mage-flow-edit,"
       "boogu-image,boogu-image-edit"},
};
const PortSpec kOports[] = {
  {.name = "model",
   .doc = "the shared model reference { hf_dir } as a FlexData beat "
          "for a diffusion stage's model iport to latch",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "model-select",
  .doc       = "Pick one diffusion model dir and emit it as a FlexData beat for "
               "the diffusion-conditioner / text-to-image / vae-encode / "
               "vae-decode stages to share -- each stage's model iport overrides "
               "its hf_dir config. 0 in / 1 out (emits once).",
  .display_name = "Model Select",
  .category  = StageCategory::Generative,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

ModelSelectStage::ModelSelectStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : TypedStage<ModelSelectStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  // Deferred validation (Stage::fail_config): construct for any config so a
  // graph can be built/edited before the model is supplied.
  _hf_dir    = attr_str("hf_dir");
  if (_hf_dir.empty()) {
    fail_config(fmt("ModelSelectStage('{}'): config.hf_dir is required (the "
                    "model dir/key to share)", this->id()));
  }
  allocate_oports(spec().oports.size());
}

ModelSelectStage::~ModelSelectStage() = default;

const StageSpec&
ModelSelectStage::spec() const noexcept
{
  return kSpec;
}

FlexData
ModelSelectStage::resolved_beat() const
{
  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("hf_dir", FlexData::make_string(_hf_dir));
  return fd;
}

void
ModelSelectStage::reset_run_state()
{
  // Per-launch reset. Stopping a pipeline destroys the RUNTIME, not the
  // stages: only unload / re-materialize destroys a Stage, so a plain
  // Stop-then-Start re-enters initialize() with this source already
  // exhausted from the previous run. Without this it would emit nothing
  // and signal done immediately, and every stage downstream of the model beat
  // would sit idle while the pipeline "completed" in milliseconds.
  _emitted = 0;
}

Job
ModelSelectStage::process(RuntimeContext& ctx)
{
  if (_emitted > 0) { ctx.signal_done(); co_return; }
  session()->info(fmt("ModelSelectStage('{}'): model = '{}'",
                      this->id(), _hf_dir));
  co_await ctx.write(0, make_payload<FlexDataPayload>(resolved_beat()));
  ++_emitted;
  ctx.signal_done();   // one-shot source: emit then close.
}

VPIPE_REGISTER_STAGE(ModelSelectStage)
VPIPE_REGISTER_SPEC(ModelSelectStage, kSpec)

}
