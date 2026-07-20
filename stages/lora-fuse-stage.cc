#include "stages/lora-fuse-stage.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "generative-models/lora-fusion.h"
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

// The registry model_type for a fused DiT dir, from its config.json
// _class_name. Krea-2 is the historical default; FLUX.2 and Qwen-Image-Edit
// fused outputs must register under their own family so downstream family
// detection (text-to-image, quantize) picks the right loader.
std::string
fused_model_type_(const std::string& dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("_class_name")) {
        const std::string cls(o.at("_class_name").as_string(""));
        if (cls == "Flux2Transformer2DModel") { return "flux2"; }
        if (cls == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
      }
    }
  }
  return "krea2";
}

}  // namespace

LoraFuseStage::LoraFuseStage(const SessionContextIntf* s,
                             std::string               id,
                             std::vector<InEdge>       iports,
                             FlexData                  config)
  : TypedStage<LoraFuseStage>(s, std::move(id), std::move(iports),
                              std::move(config))
{
  _base_model  = attr_str("base_model");
  _lora        = attr_str("lora");
  _output_name = attr_str("output_name");
  _models_db   = attr_str("models_db");
  _scale       = attr_real("scale");
  if (_models_db.empty()) { _models_db = "models"; }
  if (_scale == 0.0) { _scale = 1.0; }
  if (_base_model.empty()) {
    fail_config(fmt("LoraFuseStage('{}'): config.base_model is required",
                    this->id()));
  }
  if (_lora.empty()) {
    fail_config(fmt("LoraFuseStage('{}'): config.lora is required", this->id()));
  }
  if (_output_name.empty()) {
    fail_config(fmt("LoraFuseStage('{}'): config.output_name is required",
                    this->id()));
  }
  allocate_oports(spec().oports.size());
}

LoraFuseStage::~LoraFuseStage() = default;

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "base_model", .type = ConfigType::String, .required = true,
   .doc = "base model dir or models-DB key (for Krea-2, the transformer/ DiT)",
   .suggest_db = "models"},
  {.key = "lora", .type = ConfigType::String, .required = true,
   .doc = "LoRA .safetensors file, or a dir/key with one .safetensors"},
  {.key = "output_name", .type = ConfigType::String, .required = true,
   .doc = "result name -> <cwd>/models/<output_name> (registered), or an "
          "explicit path"},
  {.key = "scale", .type = ConfigType::Real,
   .doc = "LoRA fusion strength (default 1.0)", .def_real = 1.0},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db for base lookup + output registration", .def_str = "models"},
};
// Trigger iport (optional, any beat) + summary oport -- see model-fetch /
// model-quantize for the shared "preparation recipe" rationale.
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc  = "optional pacing trigger (any beat type); when wired, the fusion "
           "waits for one beat before running -- lets these preparation "
           "stages cascade into a recipe",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "summary",
   .doc  = "FlexData summary of the completed fusion; its `text` field "
           "renders a report via save-text, and the beat also triggers "
           "the next stage in a recipe",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "lora-fuse",
  .doc       = "Source: fuse a base model with a LoRA adapter (W + scale*dW; "
               "dW = B@A or, for a LoKr adapter, kron(w1,w2)) into a new "
               "registered model. Handles diffusers and ai-toolkit / ComfyUI "
               "(diffusion_model.*) adapter naming. For Krea-2 point base_model "
               "at the transformer/ DiT and use the result via text-to-image "
               "dit_dir. Optional trigger in / summary out.",
  .display_name = "LoRA Fuse",
  .category  = StageCategory::Preparation,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
LoraFuseStage::spec() const noexcept
{
  return kSpec;
}

void
LoraFuseStage::register_output_(const std::string& key, const std::string& dir)
{
  LmdbEnv* env = session() ? session()->lmdb_env() : nullptr;
  if (env == nullptr) {
    session()->warn(fmt("LoraFuseStage('{}'): no lmdb_env(); not registering "
                        "'{}'", this->id(), key));
    return;
  }
  try {
    FlexData rec = FlexData::make_object();
    auto ro = rec.as_object();
    ro.insert_or_assign("local_path", FlexData::make_string(dir));
    ro.insert_or_assign("source", FlexData::make_string(_base_model));
    ro.insert_or_assign("lora", FlexData::make_string(_lora));
    ro.insert_or_assign("lora_fused", FlexData::make_bool(true));
    ro.insert_or_assign("model_type",
                        FlexData::make_string(fused_model_type_(dir)));
    LmdbDb  db(*env, _models_db);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    db.put(txn, key, rec.to_binary());
    txn.commit();
    session()->info(fmt("LoraFuseStage('{}'): registered '{}' -> '{}'",
                        this->id(), key, dir));
  } catch (const std::exception& e) {
    session()->warn(fmt("LoraFuseStage('{}'): registry write for '{}' failed: "
                        "{}", this->id(), key, e.what()));
  }
}

bool
LoraFuseStage::fuse_once(const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  const std::string base_dir =
      resolve_model_dir(session(), _models_db, _base_model);

  // Resolve the LoRA to a single .safetensors file: a direct path, or the one
  // .safetensors inside a resolved dir.
  std::string lora_file = resolve_model_dir(session(), _models_db, _lora);
  if (fs::is_directory(lora_file, ec)) {
    std::string found;
    for (const auto& e : fs::directory_iterator(lora_file, ec)) {
      if (e.path().extension() == ".safetensors") {
        if (!found.empty()) {
          session()->warn(fmt("LoraFuseStage('{}'): multiple .safetensors in "
                              "'{}'; using '{}'", this->id(), lora_file, found));
          break;
        }
        found = e.path().string();
      }
    }
    if (found.empty()) {
      session()->warn(fmt("LoraFuseStage('{}'): no .safetensors in '{}'",
                          this->id(), lora_file));
      return false;
    }
    lora_file = found;
  }

  const bool explicit_path =
      _output_name[0] == '/' || _output_name.rfind("./", 0) == 0 ||
      _output_name.rfind("../", 0) == 0;
  const std::string out_dir = explicit_path
      ? _output_name
      : (fs::current_path() / "models" / _output_name).string();
  _out_dir = out_dir;

  metal_compute::MetalCompute* mc = session()->metal_compute();
  if (mc == nullptr) {
    session()->warn(fmt("LoraFuseStage('{}'): no metal-compute backend",
                        this->id()));
    return false;
  }

  session()->info(fmt("LoraFuseStage('{}'): fusing '{}' + LoRA '{}' (scale {}) "
                      "-> '{}'", this->id(), base_dir, lora_file, _scale,
                      out_dir));
  std::string err;
  if (!genai::fuse_lora(mc, base_dir, lora_file, out_dir, (float)_scale, &err,
                        stop)) {
    if (stop()) {
      session()->info(fmt("LoraFuseStage('{}'): fusion stopped; output '{}' "
                          "incomplete", this->id(), out_dir));
    } else {
      session()->warn(fmt("LoraFuseStage('{}'): {}", this->id(), err));
    }
    return false;
  }
  session()->log_normal(fmt("LoraFuseStage('{}'): fused '{}' -> '{}'",
                            this->id(), base_dir, out_dir));
  if (!explicit_path) { register_output_(_output_name, out_dir); }
  return true;
}

Job
LoraFuseStage::process(RuntimeContext& ctx)
{
  if (ctx.stop_requested()) { ctx.signal_done(); co_return; }
  // Optional trigger (see model-fetch / model-quantize): gate the work on one
  // beat when the iport is wired so this stage can cascade in a recipe.
  if (ctx.iport_connected(0)) {
    auto trig = co_await ctx.read(0);
    if (!trig) { ctx.signal_done(); co_return; }
  }
  // Inputs are checked HERE, AFTER the trigger -- in a recipe the upstream
  // model-fetch / model-quantize may not have produced them at config time.
  // Missing base or LoRA => log + halt WITHOUT emitting a summary, so the
  // cascade stops here instead of fusing a nonexistent model.
  if (!model_dir_available(session(), _models_db, _base_model)) {
    session()->error(fmt("LoraFuseStage('{}'): base model '{}' is not available "
                         "(not downloaded yet?); skipping fusion", this->id(),
                         _base_model));
    ctx.signal_done();
    co_return;
  }
  if (!model_dir_available(session(), _models_db, _lora)) {
    session()->error(fmt("LoraFuseStage('{}'): LoRA '{}' is not available "
                         "(not downloaded yet?); skipping fusion", this->id(),
                         _lora));
    ctx.signal_done();
    co_return;
  }
  const bool ok = fuse_once([&ctx] { return ctx.stop_requested(); });
  // Emit the summary only on success, so a failed fusion halts the cascade
  // (mirrors model-quantize / -benchmark / -eval).
  if (ok && ctx.has_consumers(0)) {
    FlexData summary = FlexData::make_object();
    auto so = summary.as_object();
    so.insert_or_assign("stage", FlexData::make_string("lora-fuse"));
    so.insert_or_assign("base_model", FlexData::make_string(_base_model));
    so.insert_or_assign("lora", FlexData::make_string(_lora));
    so.insert_or_assign("output", FlexData::make_string(_output_name));
    so.insert_or_assign("local_path", FlexData::make_string(_out_dir));
    so.insert_or_assign("models_db", FlexData::make_string(_models_db));
    so.insert_or_assign("scale", FlexData::make_real(_scale));
    so.insert_or_assign("fused", FlexData::make_bool(true));
    so.insert_or_assign("text", FlexData::make_string(
        fmt("[lora-fuse] {} + {} (scale {}) -> {} [ok]",
            _base_model, _lora, _scale, _output_name)()));
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(summary)));
  }
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(LoraFuseStage)
VPIPE_REGISTER_SPEC(LoraFuseStage, kSpec)

}
