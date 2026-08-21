#include "apps/web-ui/model-browser.h"

#include "common/lmdb-env.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-catalog.h"
#include "stages/model-registry.h"

#include <lmdb.h>

#include <string>
#include <string_view>
#include <vector>

using namespace std;

namespace vpipe::webui {

namespace {

string
fstr(const FlexData::ConstObjectView& o, const char* key)
{
  return o.contains(key) ? string(o.at(key).as_string("")) : string();
}

// A modality list with repeats removed, order preserved.
//
// Records written before model-detect stopped appending its I/O twice
// carry lists like ["text","image","text","image"] -- the compatibility
// filter uses includes() and survived them, but the picker draws one
// badge per entry. Normalising on the way OUT fixes what is already in
// the database, which re-registering every model would otherwise be the
// only way to reach.
FlexData
dedup_modalities(const FlexData& v)
{
  if (!v.is_array()) { return v; }
  FlexData out = FlexData::make_array();
  auto oa = out.as_array();
  auto in = v.as_array();
  std::vector<string> seen;
  for (std::size_t i = 0; i < in.size(); ++i) {
    string s(in.at(i).as_string(""));
    if (s.empty()) { continue; }
    bool dup = false;
    for (const string& t : seen) {
      if (t == s) { dup = true; break; }
    }
    if (dup) { continue; }
    seen.push_back(s);
    oa.push_back(FlexData::make_string(s));
  }
  return out;
}

}  // namespace

FlexData
list_installed_models(SessionContextIntf* sctx, string& err)
{
  FlexData doc = FlexData::make_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();

  LmdbEnv* env = sctx ? sctx->services()->lmdb_env() : nullptr;
  if (!env || !env->valid()) {
    err = "no database available";
    doc.as_object().insert("models", std::move(arr));
    return doc;
  }
  MDB_env* e = env->raw();
  MDB_txn* txn = nullptr;
  if (mdb_txn_begin(e, nullptr, MDB_RDONLY, &txn) != 0) {
    err = "could not open a read transaction";
    doc.as_object().insert("models", std::move(arr));
    return doc;
  }
  MDB_dbi dbi = 0;
  if (mdb_dbi_open(txn, string(kModelRegistryDb).c_str(), 0, &dbi) != 0) {
    // No registry sub-db yet (nothing fetched) -> empty list, not an error.
    mdb_txn_abort(txn);
    doc.as_object().insert("models", std::move(arr));
    return doc;
  }
  MDB_cursor* cur = nullptr;
  mdb_cursor_open(txn, dbi, &cur);

  // Descriptive fields copied verbatim from each registry record.
  static const char* kRecFields[] = {
      "hf_path", "local_path", "model_type", "family",
      "version", "param_class", "variant", "name"};

  MDB_val k, v;
  int rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
  while (rc == 0) {
    string key(static_cast<const char*>(k.mv_data), k.mv_size);
    string_view val(static_cast<const char*>(v.mv_data), v.mv_size);
    FlexData rec;
    bool ok = false;
    try {
      rec = FlexData::from_binary(val);
      ok = rec.is_object();
    } catch (...) {
      ok = false;
    }
    if (ok) {
      auto ro = rec.as_object();
      FlexData m = FlexData::make_object();
      auto mo = m.as_object();
      mo.insert("key", FlexData::make_string(key));
      for (const char* f : kRecFields) {
        if (ro.contains(f)) {
          mo.insert(f, FlexData::make_string(string(ro.at(f).as_string(""))));
        }
      }
      // Enrich from the catalogue: by `name` first (the vpipe-supplement
      // CoreML models share one hf_path, so hf_path can't disambiguate
      // them), else by hf_path. A miss falls back to what the RECORD
      // carries -- model-register detects category + I/O for a model it
      // registers from disk, and dropping that would hide a perfectly
      // well-described model from the compatibility filters -- and only
      // then to a plain "model" with empty I/O.
      const string name = fstr(ro, "name");
      const string hf = fstr(ro, "hf_path");
      const string mtype = fstr(ro, "model_type");
      const ModelCatalogEntry* ce = nullptr;
      if (!name.empty()) {
        ce = catalog_by_name(name);
      }
      if (!ce && !hf.empty()) {
        // BY model_type WHEN ONE REPO PUBLISHES SEVERAL MODELS.
        //
        // catalog_by_path answers with the FIRST entry for a path, which
        // is right only when there is one -- and MiniMax-H3 ships its
        // FL2VA and Ref2VA partitions from a single Comfy-Org repo, so
        // every Ref2VA record here was enriched from the FL2VA entry.
        // The visible cost was its I/O: Ref2VA reads text+image+VIDEO+
        // AUDIO and was published as text+image, which is what the
        // browser filters stages against -- so a stage needing a video
        // or audio input would not offer a model that accepts both.
        //
        // The record's own model_type is what tells them apart; it is
        // written at registration and is the same field the stage
        // allow-list matches on.
        for (const ModelCatalogEntry* c : catalog_all_by_path(hf)) {
          if (c != nullptr && !mtype.empty() && c->model_type == mtype) {
            ce = c;
            break;
          }
        }
        // No model_type on the record, or a repo that publishes one
        // model: the first entry IS the answer.
        if (ce == nullptr) { ce = catalog_by_path(hf); }
      }
      if (ce) {
        FlexData meta = catalog_entry_to_flex(*ce);
        auto meo = meta.as_object();
        for (const char* f : {"category", "parent_model_type",
                              "parent_param_class"}) {
          if (meo.contains(f)) {
            mo.insert(f,
                      FlexData::make_string(string(meo.at(f).as_string(""))));
          }
        }
        if (meo.contains("inputs")) {
          mo.insert("inputs", dedup_modalities(meo.at("inputs")));
        }
        if (meo.contains("outputs")) {
          mo.insert("outputs", dedup_modalities(meo.at("outputs")));
        }
      } else {
        for (const char* f : {"category", "parent_model_type",
                              "parent_param_class"}) {
          if (ro.contains(f)) {
            mo.insert(f, FlexData::make_string(string(ro.at(f).as_string(""))));
          }
        }
        FlexData in = ro.contains("inputs") ? dedup_modalities(ro.at("inputs"))
                                            : FlexData::make_array();
        FlexData out = ro.contains("outputs")
                           ? dedup_modalities(ro.at("outputs"))
                           : FlexData::make_array();
        // DERIVE what an old record never recorded.
        //
        // The browser filters on these: a stage declaring need_inputs
        // hides any model whose `inputs` does not cover them, and a
        // record with NONE covers nothing -- so a locally registered
        // MiniMax-H3 written before detection learned the quantized
        // repack layout is filtered out of every picker that asks for
        // text+image, which is every picker that would use it. The
        // model_type is on the record; the table that turns it into
        // modalities is the same one the catalogue uses, so deriving
        // here reads identically to a freshly registered sibling.
        //
        // Re-registering would also fix it, and nobody knows to.
        if (in.as_array().size() == 0 && out.as_array().size() == 0 &&
            !mtype.empty()) {
          std::vector<string> di, dout;
          // A CATALOGUED SIBLING FIRST -- any entry of the same
          // model_type -- and the static table only if there is none.
          //
          // The table is the host's and cannot know a PLUGIN's types, so
          // an LTX-2.5 pack registered from disk would get nothing from
          // it however complete the table became. It also does not know
          // "minimax-h3-ref2va", which is why every locally registered
          // Ref2VA had no modalities at all. A sibling knows both,
          // because a plugin's entries are appended to the same
          // catalogue the built-ins live in.
          for (const ModelCatalogEntry& c : model_catalog()) {
            if (c.model_type != mtype) { continue; }
            if (c.inputs.empty() && c.outputs.empty()) { continue; }
            di = c.inputs;
            dout = c.outputs;
            break;
          }
          if (di.empty() && dout.empty()) {
            catalog_default_io(mtype, di, dout);
          }
          for (const string& x : di) {
            in.as_array().push_back(FlexData::make_string(x));
          }
          for (const string& x : dout) {
            out.as_array().push_back(FlexData::make_string(x));
          }
        }
        if (in.as_array().size() > 0)  { mo.insert("inputs", std::move(in)); }
        if (out.as_array().size() > 0) { mo.insert("outputs", std::move(out)); }
        if (!ro.contains("category")) {
          mo.insert("category", FlexData::make_string("model"));
        }
      }
      a.push_back(std::move(m));
    }
    rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
  }
  mdb_cursor_close(cur);
  mdb_txn_abort(txn);

  doc.as_object().insert("models", std::move(arr));
  return doc;
}

}
