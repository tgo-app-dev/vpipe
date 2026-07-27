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
#include "interfaces/ui-delegate-intf.h"
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

// The registry model_type for a fused DiT dir, from its config.json
// _class_name. Krea-2 is the historical default; FLUX.2 and Qwen-Image-Edit
// fused outputs must register under their own family so downstream family
// detection (text-to-image, quantize) picks the right loader.
std::string
fused_model_type_(const std::string& dir)
{
  namespace fs = std::filesystem;
  // A bare-DiT output has config.json at the root; a self-contained pipeline
  // carries the DiT config under transformer/. Check both.
  const fs::path cfgs[] = {fs::path(dir) / "config.json",
                           fs::path(dir) / "transformer" / "config.json"};
  for (const fs::path& cfg : cfgs) {
    std::ifstream in(cfg);
    if (!in) { continue; }
    FlexData fd = FlexData::from_json(in);
    if (!fd.is_object()) { continue; }
    auto o = fd.as_object();
    if (!o.contains("_class_name")) { continue; }
    const std::string cls(o.at("_class_name").as_string(""));
    if (cls == "Flux2Transformer2DModel") { return "flux2"; }
    if (cls == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
    return "krea2";
  }
  return "krea2";
}

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

// Throttled in-place progress bar on the user-facing text stream, mirroring the
// model-quantize / text-to-image style: redrawn on a carriage return only when
// the integer percentage changes, space-padded so a shorter redraw overwrites a
// longer prior one.
void
fuse_progress_(UiTextStream* bar, std::size_t done, std::size_t total,
               int& last_pct, const char* phase)
{
  if (bar == nullptr || total == 0) { return; }
  int pct = (int)(done * 100 / total);
  if (pct < 0) { pct = 0; } else if (pct > 100) { pct = 100; }
  if (pct == last_pct) { return; }
  last_pct = pct;
  constexpr int W = 24;
  const int fill = pct * W / 100;
  std::string b(static_cast<std::size_t>(fill), '#');
  b += std::string(static_cast<std::size_t>(W - fill), '-');
  std::string line =
      fmt("\r[{}] {}% {} ({}/{})", b, pct, phase, done, total)();
  while (line.size() < 52) { line += ' '; }
  bar->write(line);
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
  _models_db     = attr_str("models_db");
  _scale         = attr_real("scale");
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
  {.key = "base_pipeline", .type = ConfigType::String,
   .doc = "optional base diffusers pipeline ROOT (dir or key); when set, the "
          "fused DiT is written under <output>/transformer/ and the pipeline's "
          "other components (text_encoder/, vae/, tokenizer/, scheduler/, "
          "model_index.json) are hard-linked/copied alongside -> a "
          "self-contained model that chain-quantizes like the stock one. Empty "
          "=> bare DiT output.", .suggest_db = "models"},
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

  // Self-contained mode: fuse the DiT into <out>/transformer/ and place the base
  // pipeline's other components next to it; bare-DiT mode fuses into <out>.
  std::string pipe_root;
  std::string fuse_out = out_dir;
  if (!_base_pipeline.empty()) {
    pipe_root = resolve_model_dir(session(), _models_db, _base_pipeline);
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
  {   // in-place fusion progress bar (per base tensor written)
    std::unique_ptr<UiTextStream> bar = session()->open_text_stream();
    int last_pct = -1;
    const bool ok = genai::fuse_lora(
        mc, base_dir, lora_file, fuse_out, (float)_scale, &err, stop,
        [&](std::size_t done, std::size_t total) {
          fuse_progress_(bar.get(), done, total, last_pct, "fuse");
        });
    bar->end();
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
    std::unique_ptr<UiTextStream> cbar = session()->open_text_stream();
    int cpct = -1;
    for (std::size_t i = 0; i < comps.size(); ++i) {
      fuse_progress_(cbar.get(), i, comps.size(), cpct, "copy");
      if (stop()) { cbar->end(); return false; }
      if (!link_or_copy_tree_(comps[i], fs::path(out_dir) / comps[i].filename(),
                              ec, stop)) {
        cbar->end();
        session()->warn(fmt("LoraFuseStage('{}'): failed to copy component '{}' "
                            "from '{}'", this->id(),
                            comps[i].filename().string(), pipe_root));
        return false;
      }
    }
    fuse_progress_(cbar.get(), comps.size(), comps.size(), cpct, "copy");
    cbar->end();
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
