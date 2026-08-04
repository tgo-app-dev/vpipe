#include "stages/model-register-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/runtime-context.h"
#include "stages/model-catalog.h"
#include "stages/model-detect.h"
#include "stages/model-registry.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  {.key = "model_dir", .type = ConfigType::String, .required = true,
   .doc = "model directory on disk to register (or a bare .mlpackage / "
          ".gguf file). Everything recorded about it -- runtime type, "
          "input/output modalities, size labels -- is detected from the "
          "directory",
   .is_path = true, .path_kind = "dir"},
  {.key = "key", .type = ConfigType::String,
   .doc = "registry key to register under; empty => \"<owner>/<repo>\" from "
          "the two trailing path components (the layout model-fetch writes), "
          "else the directory name"},
  {.key = "model_type", .type = ConfigType::String,
   .doc = "override the DETECTED runtime hint (e.g. qwen3.5, gemma4, flux2, "
          "krea2); empty => detected from the directory, and left empty when "
          "it cannot be determined"},
  {.key = "overwrite_existing", .type = ConfigType::Bool,
   .doc = "let this key be taken over from a DIFFERENT directory (off by "
          "default: re-registering over another model's key is refused). "
          "Re-registering the SAME directory is idempotent and needs no "
          "opt-in",
   .def_bool = false},
};
// Trigger iport (optional, any beat) + summary oport -- see model-fetch
// for the shared "preparation recipe" rationale.
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc  = "optional pacing trigger (any beat type); when wired, the work "
           "waits for one beat before running -- lets these preparation "
           "stages cascade into a recipe",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "summary",
   .doc  = "FlexData summary of the registration; its `text` field renders "
           "a report via save-text, and the beat also triggers the next "
           "stage in a recipe",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "model-register",
  .doc       = "One-shot: register a model directory already on disk into "
               "the model registry (model-fetch without the download). The "
               "runtime type and the input/output modalities are DETECTED "
               "from the directory -- catalogue match, config.json, "
               "diffusers transformer, GGUF or CoreML package -- so the "
               "usual config is just the path. Optional trigger in / "
               "summary out.",
  .display_name = "Model Register",
  .category  = StageCategory::Preparation,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

// Render the detected modality lists as "text, image -> text".
string
io_text_(const DetectedModel& d)
{
  auto join = [](const vector<string>& v) {
    string s;
    for (const auto& m : v) {
      if (!s.empty()) { s += ", "; }
      s += m;
    }
    return s.empty() ? string("-") : s;
  };
  return join(d.inputs) + " -> " + join(d.outputs);
}

void
put_strings_(FlexData::ObjectView& o, const char* key,
             const vector<string>& v)
{
  FlexData arr = FlexData::make_array();
  {
    auto a = arr.as_array();
    for (const auto& s : v) { a.push_back(FlexData::make_string(s)); }
  }
  o.insert_or_assign(key, std::move(arr));
}

}  // namespace

ModelRegisterStage::ModelRegisterStage(const SessionContextIntf* s,
                                       string                    id,
                                       vector<InEdge>            iports,
                                       FlexData                  config)
  : TypedStage<ModelRegisterStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  _model_dir          = attr_str("model_dir");
  _key                = attr_str("key");
  _model_type         = attr_str("model_type");
  _overwrite_existing = attr_bool("overwrite_existing");

  // Deferred validation: existence is NOT checked here. In a recipe the
  // directory may be produced by an upstream model-quantize / lora-fuse
  // that has not run yet at config time (the same reason model-benchmark
  // checks availability after its trigger fires).
  if (_model_dir.empty()) {
    fail_config(fmt("ModelRegisterStage('{}'): config.model_dir is required",
                    this->id()));
  }

  allocate_oports(spec().oports.size());
}

const StageSpec&
ModelRegisterStage::spec() const noexcept
{
  return kSpec;
}

ModelRegisterStage::RegisterResult
ModelRegisterStage::register_once()
{
  RegisterResult r;
  const SessionContextIntf* s = session();

  // -------- 1. The directory must exist NOW ----------------------------
  error_code ec;
  if (!fs::exists(_model_dir, ec) || ec) {
    s->warn(fmt("ModelRegisterStage('{}'): '{}' does not exist; nothing "
                "to register", this->id(), _model_dir));
    return r;   // ok stays false -> halts the cascade
  }
  // Register the ABSOLUTE path: the record outlives this process, and a
  // relative path would resolve against whatever cwd a later run has.
  const string abs = fs::absolute(_model_dir, ec).lexically_normal().string();
  const string container = ec ? _model_dir : abs;

  // A CoreML supplement is loaded as a BUNDLE, not as the directory it
  // sits in, so the record has to name the bundle -- resolve_model_dir
  // hands its answer straight to CoreMLModelManager::load, and a bare
  // container directory there fails at modelWithContentsOfURL. This is
  // the same rule model-fetch applies when it unpacks an archive; a
  // registered supplement and a fetched one must be indistinguishable to
  // every reader, and that includes what local_path points AT.
  //
  // Empty means "not a CoreML container" -- the overwhelmingly common
  // case, a safetensors checkpoint whose loader wants the directory.
  const string artifact = coreml_artifact(container);
  r.local_path = artifact.empty() ? container : artifact;
  if (!artifact.empty() && artifact != container) {
    s->info(fmt("ModelRegisterStage('{}'): '{}' holds the CoreML bundle "
                "'{}'; registering the bundle, since that is what loads",
                this->id(), container,
                fs::path(artifact).filename().string()));
  }

  LmdbEnv* env = s->services()->lmdb_env();
  if (!env) {
    s->warn(fmt("ModelRegisterStage('{}'): session lmdb_env() unavailable",
                this->id()));
    return r;
  }

  // -------- 2. Detect what this directory IS ---------------------------
  // Keyed off the CONTAINER, not the resolved bundle: the key, the
  // hf_path and the catalogue hint all describe where the model lives,
  // and a bundle path would derive them from the archive's stem
  // ("<name>/<stem>_w8.mlpackage") instead. Detection runs on the
  // resolved path so a supplement is classified from the bundle suffix
  // (model-detect step 5), but keeps the container-derived hint so a
  // catalogue lookup still sees the name the model was published under.
  const string hf_path = hf_path_from_local(container);
  r.detected = detect_model_dir(r.local_path, hf_path);
  if (!_model_type.empty() && _model_type != r.detected.model_type) {
    // An explicit type replaces the detected one AND its derived I/O, so
    // the record stays self-consistent.
    r.detected.model_type = _model_type;
    r.detected.inputs.clear();
    r.detected.outputs.clear();
    catalog_default_io(_model_type, r.detected.inputs, r.detected.outputs);
    r.detected.detected_by = "config-key";
  }

  r.key = !_key.empty() ? _key
        : !hf_path.empty() ? hf_path
                           : fs::path(container).filename().string();
  if (r.key.empty()) {
    s->warn(fmt("ModelRegisterStage('{}'): could not derive a registry key "
                "from '{}'; set config.key", this->id(), _model_dir));
    return r;
  }

  // -------- 3. Refuse to clobber ANOTHER model, unless asked ------------
  // Construct the db handle BEFORE any user txn (its ctor opens its own
  // RW boot txn; opening it while another RW txn is live deadlocks).
  LmdbDb db(*env, kModelRegistryDb);
  std::string existing_path;
  {
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadOnly);
    auto view = db.get(txn, r.key);
    if (view) {
      const string bytes(*view);          // copy before the txn ends
      r.replaced = true;
      try {
        FlexData old = FlexData::from_binary(bytes);
        if (old.is_object()) {
          auto oo = old.as_object();
          if (oo.contains("local_path")) {
            existing_path = string(oo.at("local_path").as_string(""));
          }
        }
      } catch (...) {
      }                                   // corrupt record: treat as other
    }
    txn.abort();
  }
  // Re-registering the SAME directory is idempotent, not a clobber: the
  // desired end-state already holds (the model-remove/missing_ok rule).
  // It still rewrites the record, since detection may have improved
  // since. Without this a relaunched recipe would halt on its second
  // run, having done nothing wrong.
  const bool same_model = r.replaced && existing_path == r.local_path;
  if (r.replaced && !same_model && !_overwrite_existing) {
    s->warn(fmt("ModelRegisterStage('{}'): '{}' is already registered to a "
                "DIFFERENT directory ('{}'); set overwrite_existing=true to "
                "replace it", this->id(), r.key, existing_path));
    r.replaced = false;
    return r;
  }

  // -------- 4. Write the record ----------------------------------------
  // Field-for-field what model-fetch writes, so a registered model and a
  // fetched one are indistinguishable to every reader (resolve_model_dir,
  // the web-ui browser, model-remove).
  FlexData rec = FlexData::make_object();
  auto ro = rec.as_object();
  ro.insert_or_assign("local_path", FlexData::make_string(r.local_path));
  if (!hf_path.empty()) {
    ro.insert_or_assign("hf_path", FlexData::make_string(hf_path));
  }
  if (r.key != hf_path) {
    // The browser's catalogue enrichment matches `name` first; carrying
    // it also lets a supplement registered under its own key be found.
    ro.insert_or_assign("name", FlexData::make_string(r.key));
  }
  const DetectedModel& d = r.detected;
  record_detected_fields(ro, d);
  ro.insert_or_assign("file_count", FlexData::make_uint(d.file_count));
  ro.insert_or_assign("total_bytes", FlexData::make_uint(d.total_bytes));
  // Provenance: this model was NOT downloaded by model-fetch. A reader
  // that cares (e.g. a re-fetch) can tell the two apart.
  ro.insert_or_assign("registered_from_disk", FlexData::make_bool(true));

  try {
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    const string bytes = rec.to_binary();
    db.put(txn, r.key, bytes);
    txn.commit();
  } catch (const std::exception& e) {
    s->warn(fmt("ModelRegisterStage('{}'): registry write for '{}' failed: "
                "{}", this->id(), r.key, e.what()));
    return r;
  }

  s->info(fmt(
      "ModelRegisterStage('{}'): registered '{}' -> '{}' [{}{}] {}",
      this->id(), r.key, r.local_path,
      d.model_type.empty() ? string("type unknown") : d.model_type,
      d.detected_by.empty() ? string() : fmt(", by {}", d.detected_by)(),
      io_text_(d)));
  if (d.model_type.empty()) {
    // Registering still worked -- the model resolves by key -- but no
    // stage picker will offer it, which is worth saying out loud.
    s->warn(fmt(
        "ModelRegisterStage('{}'): could not determine a runtime type for "
        "'{}'; it is registered but will not be offered by the model "
        "pickers -- set config.model_type to fix that",
        this->id(), r.key));
  }
  r.ok = true;
  return r;
}

Job
ModelRegisterStage::process(RuntimeContext& ctx)
{
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }

  // Optional trigger: when the iport is wired, wait for one beat so this
  // stage can cascade in a preparation recipe (any beat type; payload
  // ignored). Upstream EOS -> nothing to do.
  if (ctx.iport_connected(0)) {
    auto trig = co_await ctx.read(0);
    if (!trig) {
      ctx.signal_done();
      co_return;
    }
  }

  const RegisterResult r = register_once();

  // Emit the summary only on success, so a failed registration halts the
  // cascade (mirrors model-quantize / -benchmark / -eval / -remove).
  if (r.ok && ctx.has_consumers(0)) {
    const DetectedModel& d = r.detected;
    FlexData sum = FlexData::make_object();
    auto so = sum.as_object();
    so.insert_or_assign("stage", FlexData::make_string("model-register"));
    so.insert_or_assign("model", FlexData::make_string(r.key));
    so.insert_or_assign("local_path", FlexData::make_string(r.local_path));
    so.insert_or_assign("replaced", FlexData::make_bool(r.replaced));
    so.insert_or_assign("model_type", FlexData::make_string(d.model_type));
    so.insert_or_assign("detected_by",
                        FlexData::make_string(d.detected_by));
    put_strings_(so, "inputs", d.inputs);
    put_strings_(so, "outputs", d.outputs);
    so.insert_or_assign("total_bytes", FlexData::make_uint(d.total_bytes));
    so.insert_or_assign("text", FlexData::make_string(fmt(
        "[model-register] {}\n  -> {}\n  type: {} ({})\n  I/O:  {}\n"
        "  {} file(s), {} bytes{}",
        r.key, r.local_path,
        d.model_type.empty() ? string("unknown") : d.model_type,
        d.detected_by.empty() ? string("not detected") : d.detected_by,
        io_text_(d), d.file_count, d.total_bytes,
        r.replaced ? "\n  (replaced an existing record)" : "")()));
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(sum)));
  }
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelRegisterStage)
VPIPE_REGISTER_SPEC(ModelRegisterStage, kSpec)

}  // namespace vpipe
