#include "stages/optimizer-select-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/train/optimizer.h"
#include "interfaces/session-context-intf.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

using genai::train::OptimizerSpec;

constexpr SpecExtra kOptChoices[] = {
  {spec_key::kChoices, OptimizerSpec::kOptimizers},
};
constexpr SpecExtra kSchedChoices[] = {
  {spec_key::kChoices, OptimizerSpec::kSchedules},
};

const ConfigKey kAttrs[] = {
  {.key = "optimizer",
   .type = ConfigType::String,
   .doc = "adamw (default), prodigy (estimates its own step size -- set "
          "learning_rate 1.0; forgiving on a small dataset), lion, or sgd",
   .def_str = "adamw",
   .extra = kOptChoices},
  {.key = "learning_rate",
   .type = ConfigType::Real,
   .doc = "0 (default) takes the optimizer's own: 1e-4 adamw, 1.0 "
          "prodigy, 3e-5 lion, 1e-2 sgd"},
  {.key = "beta1",
   .type = ConfigType::Real,
   .doc = "first-moment decay (sgd: momentum), default 0.9",
   .def_real = 0.9},
  {.key = "beta2",
   .type = ConfigType::Real,
   .doc = "second-moment decay, default 0.999",
   .def_real = 0.999},
  {.key = "eps",
   .type = ConfigType::Real,
   .doc = "default 1e-8",
   .def_real = 1e-8},
  {.key = "weight_decay",
   .type = ConfigType::Real,
   .doc = "decoupled weight decay, default 0.01",
   .def_real = 0.01},
  {.key = "lr_schedule",
   .type = ConfigType::String,
   .doc = "how the learning rate moves over the run: auto (default: "
          "constant for a small dataset, cosine for one of 64 pictures or "
          "more), constant, linear, cosine, cosine_restarts. Called "
          "lr_schedule because a \"scheduler\" here is the noise schedule.",
   .def_str = "auto",
   .extra = kSchedChoices},
  {.key = "warmup_steps",
   .type = ConfigType::Int,
   .doc = "a linear ramp from zero; -1 (default) is 2% of the run",
   .def_int = -1},
  {.key = "min_lr_ratio",
   .type = ConfigType::Real,
   .doc = "where linear and cosine end, as a fraction of the rate "
          "(default 0)"},
  {.key = "restarts",
   .type = ConfigType::Int,
   .doc = "cosine_restarts: how many cycles (default 1)",
   .def_int = 1},
  {.key = "grad_clip",
   .type = ConfigType::Real,
   .doc = "the global L2 norm gradients are clipped to; 0 disables it "
          "(default 1.0)",
   .def_real = 1.0},
  {.key = "ema",
   .type = ConfigType::Real,
   .doc = "EMA decay of the adapter's weights, e.g. 0.999; each checkpoint "
          "then also saves the averaged adapter. 0 (default) is off."},
  {.key = "prodigy_d_coef",
   .type = ConfigType::Real,
   .doc = "prodigy: scales its step-size estimate (default 1.0)",
   .def_real = 1.0},
  {.key = "bias_correction",
   .type = ConfigType::Bool,
   .doc = "prodigy: Adam's bias correction (default off, as prodigyopt)"},
};

const PortSpec kOports[] = {
  {.name = "optimizer",
   .doc = "optimizer spec {kind: optimizer, optimizer, learning_rate, ...}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "optimizer-select",
  .doc = "Choose the optimizer and learning-rate schedule a train-lora "
         "stage runs, emitted as one FlexData beat. Pairs with "
         "training-dataset and train-lora. 0 in / 1 out (emits once).",
  .display_name = "Optimizer",
  .category = StageCategory::Generative,
  .iports = {},
  .oports = kOports,
  .attrs = kAttrs,
};

}  // namespace

OptimizerSelectStage::OptimizerSelectStage(const SessionContextIntf* s,
                                           std::string id,
                                           std::vector<InEdge> iports,
                                           FlexData config)
  : TypedStage<OptimizerSelectStage>(s, std::move(id), std::move(iports),
                                     std::move(config))
{
  OptimizerSpec o;
  o.optimizer = attr_str("optimizer");
  o.lr = attr_real("learning_rate");
  o.beta1 = attr_real("beta1");
  o.beta2 = attr_real("beta2");
  o.eps = attr_real("eps");
  o.weight_decay = attr_real("weight_decay");
  o.lr_schedule = attr_str("lr_schedule");
  o.warmup_steps = (int)attr_int("warmup_steps");
  o.min_lr_ratio = attr_real("min_lr_ratio");
  o.restarts = (int)attr_int("restarts");
  o.grad_clip = attr_real("grad_clip");
  o.ema = attr_real("ema");
  o.prodigy_d_coef = attr_real("prodigy_d_coef");
  o.bias_correction = attr_bool("bias_correction");
  // Validated by the one parser train-lora reads it with, so the two
  // cannot disagree about what is legal.
  std::string err;
  OptimizerSpec check;
  if (!OptimizerSpec::from_flex(o.to_flex(), &check, &err)) {
    fail_config(fmt("OptimizerSelectStage('{}'): {}", this->id(), err));
  }
  _spec = o.to_flex();
  allocate_oports(spec().oports.size());
}

OptimizerSelectStage::~OptimizerSelectStage() = default;

const StageSpec&
OptimizerSelectStage::spec() const noexcept
{
  return kSpec;
}

void
OptimizerSelectStage::reset_run_state()
{
  _emitted = 0;
}

Job
OptimizerSelectStage::process(RuntimeContext& ctx)
{
  if (_emitted > 0) { ctx.signal_done(); co_return; }
  {
    FlexData sp = _spec;
    auto o = sp.as_object();
    session()->info(fmt(
        "OptimizerSelectStage('{}'): {} (learning rate {}, {} schedule)",
        this->id(), std::string(o.at("optimizer").as_string("")),
        o.at("learning_rate").as_real(0.0),
        std::string(o.at("lr_schedule").as_string(""))));
  }
  FlexData out = _spec;
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(out)));
  ++_emitted;
  ctx.signal_done();
}

VPIPE_REGISTER_STAGE(OptimizerSelectStage)
VPIPE_REGISTER_SPEC(OptimizerSelectStage, kSpec)

}  // namespace vpipe
