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
#include "interfaces/session-services-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "stages/model-detect.h"
#include "stages/model-registry.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// Recursively replicate `src` into `dst`, hard-linking files where possible
// (same filesystem -- no data duplication for the ~GB encoder/vae shards) and
// falling back to a byte copy across devices. Used to place the base pipeline's
// non-transformer components next to the fused DiT.
bool
link_or_copy_tree_(const std::filesystem::path& src,
                   const std::filesystem::path& dst, std::error_code& ec,
                   const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  if (stop()) { return false; }
  if (fs::is_directory(src, ec)) {
    fs::create_directories(dst, ec);
    if (ec) { return false; }
    for (const auto& e : fs::directory_iterator(src, ec)) {
      if (!link_or_copy_tree_(e.path(), dst / e.path().filename(), ec, stop)) {
        return false;
      }
    }
    return true;
  }
  std::error_code le;
  fs::create_hard_link(src, dst, le);
  if (le) {   // cross-device / unsupported -> real copy
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) { return false; }
  }
  return true;
}

}  // namespace

LoraFuseStage::LoraFuseStage(const SessionContextIntf* s,
                             std::string               id,
                             std::vector<InEdge>       iports,
                             FlexData                  config)
  : TypedStage<LoraFuseStage>(s, std::move(id), std::move(iports),
                              std::move(config))
{
  _base_model    = attr_str("base_model");
  _lora          = attr_str("lora");
  _output_name   = attr_str("output_name");
  _base_pipeline = attr_str("base_pipeline");
  _scale         = attr_real("scale");
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
   .suggest_db = kModelRegistryDb},
  {.key = "lora", .type = ConfigType::String, .required = true,
   .doc = "LoRA .safetensors file, or a dir/key with one .safetensors"},
  {.key = "output_name", .type = ConfigType::String, .required = true,
   .doc = "result name -> <cwd>/models/<output_name> (registered), or an "
          "explicit path"},
  {.key = "base_pipeline", .type = ConfigType::String,
   .doc = "optional base diffusers pipeline ROOT (dir or key); when set, the "
          "fused DiT is written under <output>/transformer/ and the pipeline's "
          "other components (text_encoder/, vae/, tokenizer/, scheduler/, "
          "model_index.json) are hard-linked/copied alongside -> a "
          "self-contained model that chain-quantizes like the stock one. Empty "
          "=> bare DiT output.", .suggest_db = kModelRegistryDb},
  {.key = "scale", .type = ConfigType::Real,
   .doc = "LoRA fusion strength (default 1.0)", .def_real = 1.0},
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
               "at the transformer/ DiT and use the result via generate-image "
               "dit_dir; set base_pipeline to also copy the encoder/vae/tokenizer "
               "for a self-contained, chain-quantizable model. Optional trigger "
               "in / summary out.",
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
  LmdbEnv* env = session() ? session()->services()->lmdb_env() : nullptr;
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
    // Describe the fused output by probing it (model-detect.h): a
    // self-contained pipeline registers under its family, a BARE DiT
    // under "<family>-dit" -- which is the type generate-image's `dit_dir`
    // field filters on, and where a bare fused DiT actually belongs.
    const DetectedModel d = detect_model_dir(dir);
    record_detected_fields(ro, d);
    if (d.model_type.empty()) {
      session()->warn(fmt(
          "LoraFuseStage('{}'): could not determine a runtime type for the "
          "fused output '{}'; it is registered but will not be offered by "
          "the model pickers", this->id(), key));
    }
    LmdbDb  db(*env, kModelRegistryDb);
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
      resolve_model_dir(session(), _base_model);

  // Resolve the LoRA to a single .safetensors: a direct path, a registry
  // key whose record names its file, or a directory holding exactly one.
  //
  // Shared with generate-video's runtime adapter rather than scanned here,
  // and the difference is not cosmetic: this used to take the FIRST
  // .safetensors the directory iterator produced and warn. Two Turbo
  // checkpoints published from one repo land side by side, so that was a
  // coin flip between two adapters, decided by directory order.
  std::string lerr;
  std::string lora_file = resolve_adapter_file(session(), _lora, &lerr);
  if (lora_file.empty()) {
    session()->warn(fmt("LoraFuseStage('{}'): {}", this->id(), lerr));
    return false;
  }

  const bool explicit_path =
      _output_name[0] == '/' || _output_name.rfind("./", 0) == 0 ||
      _output_name.rfind("../", 0) == 0;
  const std::string out_dir = explicit_path
      ? _output_name
      : (fs::current_path() / "models" / _output_name).string();
  _out_dir = out_dir;

  metal_compute::MetalCompute* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->warn(fmt("LoraFuseStage('{}'): no metal-compute backend",
                        this->id()));
    return false;
  }

  // Self-contained mode: fuse the DiT into <out>/transformer/ and place the base
  // pipeline's other components next to it; bare-DiT mode fuses into <out>.
  std::string pipe_root;
  std::string fuse_out = out_dir;
  if (!_base_pipeline.empty()) {
    pipe_root = resolve_model_dir(session(), _base_pipeline);
    if (!fs::is_directory(pipe_root, ec)) {
      session()->warn(fmt("LoraFuseStage('{}'): base_pipeline '{}' is not a "
                          "directory", this->id(), pipe_root));
      return false;
    }
    fuse_out = (fs::path(out_dir) / "transformer").string();
  }

  session()->info(fmt("LoraFuseStage('{}'): fusing '{}' + LoRA '{}' (scale {}) "
                      "-> '{}'{}", this->id(), base_dir, lora_file, _scale,
                      fuse_out,
                      pipe_root.empty() ? ""
                          : (" (self-contained from '" + pipe_root + "')")));
  std::string err;
  {   // in-place fusion progress (per base tensor written)
    UiProgress bar = session()->open_progress("fuse");
    const bool ok = genai::fuse_lora(
        mc, base_dir, lora_file, fuse_out, (float)_scale, &err, stop,
        [&](std::size_t done, std::size_t total) {
          bar.update(done, total);
        });
    bar.finish();
    if (!ok) {
      if (stop()) {
        session()->info(fmt("LoraFuseStage('{}'): fusion stopped; output '{}' "
                            "incomplete", this->id(), fuse_out));
      } else {
        session()->warn(fmt("LoraFuseStage('{}'): {}", this->id(), err));
      }
      return false;
    }
  }

  // Copy the base pipeline's non-transformer components alongside the fused DiT
  // (progress bar over the components; the big encoder/vae shards hard-link).
  if (!pipe_root.empty()) {
    std::vector<fs::path> comps;
    for (const auto& e : fs::directory_iterator(pipe_root, ec)) {
      if (e.path().filename() != "transformer") { comps.push_back(e.path()); }
    }
    UiProgress cbar = session()->open_progress("copy components");
    for (std::size_t i = 0; i < comps.size(); ++i) {
      cbar.update(i, comps.size());
      if (stop()) { return false; }
      if (!link_or_copy_tree_(comps[i], fs::path(out_dir) / comps[i].filename(),
                              ec, stop)) {
        session()->warn(fmt("LoraFuseStage('{}'): failed to copy component '{}' "
                            "from '{}'", this->id(),
                            comps[i].filename().string(), pipe_root));
        return false;
      }
    }
    cbar.finish();
    session()->log_normal(fmt("LoraFuseStage('{}'): assembled self-contained "
                              "model at '{}'", this->id(), out_dir));
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
  if (!model_dir_available(session(), _base_model)) {
    session()->error(fmt("LoraFuseStage('{}'): base model '{}' is not available "
                         "(not downloaded yet?); skipping fusion", this->id(),
                         _base_model));
    ctx.signal_done();
    co_return;
  }
  if (!model_dir_available(session(), _lora)) {
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
