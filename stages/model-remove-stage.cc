#include "stages/model-remove-stage.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

namespace vpipe {

namespace {

// Guard the opt-in on-disk deletion: refuse an empty path or a filesystem
// root (path == its own root_path, i.e. "/" or "C:\\"). The recorded
// local_path is a model directory model-fetch wrote, but never trust a
// registry value enough to rm -rf "/".
bool
safe_to_delete_(const fs::path& p)
{
  if (p.empty()) {
    return false;
  }
  const fs::path abs = p.lexically_normal();
  if (abs.empty() || abs == abs.root_path() || abs == ".") {
    return false;
  }
  return true;
}

// Read a model's registered `local_path` (empty if the record has none).
string
record_local_path_(const FlexData& rec)
{
  if (!rec.is_object()) {
    return {};
  }
  auto obj = rec.as_object();
  if (obj.contains("local_path")) {
    return string(obj.at("local_path").as_string(""));
  }
  return {};
}

constexpr ConfigKey kAttrs[] = {
  {.key = "model", .type = ConfigType::String, .required = true,
   .doc = "models-DB key to remove (the key model-fetch / model-quantize "
          "registered the model under)",
   .suggest_db = "models"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db to remove the record from", .def_str = "models"},
  {.key = "delete_files", .type = ConfigType::Bool,
   .doc = "also delete the model's recorded local_path directory from disk "
          "(opt-in; destructive -- the record alone is removed otherwise)",
   .def_bool = false},
  {.key = "missing_ok", .type = ConfigType::Bool,
   .doc = "treat a model that is not registered as a success (already "
          "removed) instead of halting a cascade", .def_bool = true},
};
// Trigger iport (optional, any beat) + summary oport -- see model-fetch for
// the shared "preparation recipe" rationale.
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc  = "optional pacing trigger (any beat type); when wired, the work "
           "waits for one beat before running -- lets these preparation "
           "stages cascade into a recipe",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "summary",
   .doc  = "FlexData summary of the completed work; its `text` field "
           "renders a report via save-text, and the beat also triggers "
           "the next stage in a recipe",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "model-remove",
  .doc       = "One-shot: remove a registered model from the models DB "
               "(the inverse of model-fetch). Deletes the record keyed by "
               "`model`; with delete_files=true also removes the recorded "
               "local_path directory from disk. A model already absent is a "
               "success when missing_ok. Optional trigger in / summary out.",
  .display_name = "Model Remove",
  .category  = StageCategory::Preparation,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

ModelRemoveStage::ModelRemoveStage(const SessionContextIntf* s,
                                   string                    id,
                                   vector<InEdge>            iports,
                                   FlexData                  config)
  : TypedStage<ModelRemoveStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  _model        = attr_str("model");
  _models_db    = attr_str("models_db");
  _delete_files = attr_bool("delete_files");
  _missing_ok   = attr_bool("missing_ok");

  if (_models_db.empty()) {
    _models_db = "models";
  }
  if (_model.empty()) {
    fail_config(fmt("ModelRemoveStage('{}'): config.model is required",
                    this->id()));
  }

  allocate_oports(spec().oports.size());
}

const StageSpec&
ModelRemoveStage::spec() const noexcept
{
  return kSpec;
}

ModelRemoveStage::RemoveResult
ModelRemoveStage::remove_once()
{
  RemoveResult r;
  const SessionContextIntf* s = session();

  LmdbEnv* env = s->lmdb_env();
  if (!env) {
    // No registry to remove from -> failure (halt the cascade). warn (not
    // error) so process() exits cleanly without throwing through the
    // coroutine and without emitting a summary.
    s->warn(fmt("ModelRemoveStage('{}'): session lmdb_env() unavailable",
                this->id()));
    return r;   // ok stays false
  }

  // Construct the db handle BEFORE any user txn (its ctor opens its own RW
  // boot txn; opening it while another RW txn is live on the env deadlocks).
  LmdbDb db(*env, _models_db);

  // -------- 1. Read the record (does the model exist?) -----------------
  FlexData rec;
  {
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadOnly);
    auto view = db.get(txn, _model);
    if (view) {
      const string bytes(*view);          // copy before the txn ends
      r.present = true;
      try {
        rec = FlexData::from_binary(bytes);
      } catch (...) {
        rec = FlexData::make_object();     // corrupt record: still removable
      }
    }
    txn.abort();
  }

  if (!r.present) {
    if (_missing_ok) {
      s->info(fmt(
          "ModelRemoveStage('{}'): '{}' not registered in sub-db '{}'; "
          "nothing to remove", this->id(), _model, _models_db));
      r.ok = true;   // desired end-state ("not registered") already holds
    } else {
      // A missing model is a failure -- halt the cascade (no summary).
      s->warn(fmt(
          "ModelRemoveStage('{}'): '{}' is not registered in sub-db '{}'; "
          "set missing_ok=true to treat this as a no-op", this->id(),
          _model, _models_db));
    }
    return r;
  }

  // -------- 2. Optional on-disk deletion (best-effort) -----------------
  r.local_path = record_local_path_(rec);
  if (_delete_files) {
    const fs::path lp(r.local_path);
    if (!safe_to_delete_(lp)) {
      r.delete_note = r.local_path.empty()
          ? "no local_path recorded" : "unsafe local_path; skipped";
      s->warn(fmt(
          "ModelRemoveStage('{}'): refusing to delete files for '{}' ({})",
          this->id(), _model, r.delete_note));
    } else {
      std::error_code ec;
      if (fs::exists(lp, ec)) {
        const auto n = fs::remove_all(lp, ec);
        if (ec) {
          r.delete_note = fmt("delete failed: {}", ec.message())();
          s->warn(fmt(
              "ModelRemoveStage('{}'): deleting '{}' failed: {}",
              this->id(), r.local_path, ec.message()));
        } else {
          r.files_deleted = true;
          r.delete_note = fmt("removed {} entr{}", n, n == 1 ? "y" : "ies")();
          s->info(fmt(
              "ModelRemoveStage('{}'): deleted '{}' ({})",
              this->id(), r.local_path, r.delete_note));
        }
      } else {
        r.delete_note = "local_path did not exist on disk";
        s->info(fmt(
            "ModelRemoveStage('{}'): local_path '{}' already gone",
            this->id(), r.local_path));
      }
    }
  }

  // -------- 3. Remove the registry record ------------------------------
  {
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    r.removed = db.del(txn, _model);
    txn.commit();
  }
  if (!r.removed) {
    // Raced with another remover between the read and the delete: the
    // end-state ("not registered") still holds, so treat it as success.
    s->info(fmt(
        "ModelRemoveStage('{}'): '{}' was already gone at delete time",
        this->id(), _model));
  }
  s->info(fmt(
      "ModelRemoveStage('{}'): removed '{}' from sub-db '{}'{}",
      this->id(), _model, _models_db,
      r.files_deleted ? " (+ deleted files)" : ""));
  r.ok = true;
  return r;
}

Job
ModelRemoveStage::process(RuntimeContext& ctx)
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

  const RemoveResult r = remove_once();

  // Emit the summary only on success, so a failed removal halts the
  // cascade (mirrors model-quantize / -benchmark / -eval).
  if (r.ok && ctx.has_consumers(0)) {
    FlexData sum = FlexData::make_object();
    auto so = sum.as_object();
    so.insert_or_assign("stage", FlexData::make_string("model-remove"));
    so.insert_or_assign("model", FlexData::make_string(_model));
    so.insert_or_assign("models_db", FlexData::make_string(_models_db));
    so.insert_or_assign("present", FlexData::make_bool(r.present));
    so.insert_or_assign("removed", FlexData::make_bool(r.removed));
    so.insert_or_assign("files_deleted",
                        FlexData::make_bool(r.files_deleted));
    if (!r.local_path.empty()) {
      so.insert_or_assign("local_path",
                          FlexData::make_string(r.local_path));
    }
    so.insert_or_assign("text", FlexData::make_string(
        r.present
            ? fmt("[model-remove] {}\n  <- sub-db '{}'{}",
                  _model, _models_db,
                  _delete_files
                      ? fmt("\n  files: {}", r.delete_note.empty()
                                ? string("(none)") : r.delete_note)()
                      : string())()
            : fmt("[model-remove] {} not registered (nothing to remove)",
                  _model)()));
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(sum)));
  }
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelRemoveStage)
VPIPE_REGISTER_SPEC(ModelRemoveStage, kSpec)

}  // namespace vpipe
