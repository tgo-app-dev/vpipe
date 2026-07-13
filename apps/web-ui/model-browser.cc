#include "apps/web-ui/model-browser.h"

#include "common/lmdb-env.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-catalog.h"

#include <lmdb.h>

#include <string>
#include <string_view>

using namespace std;

namespace vpipe::webui {

namespace {

string
fstr(const FlexData::ConstObjectView& o, const char* key)
{
  return o.contains(key) ? string(o.at(key).as_string("")) : string();
}

}  // namespace

FlexData
list_installed_models(SessionContextIntf* sctx, const string& db_name,
                      string& err)
{
  FlexData doc = FlexData::make_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();

  LmdbEnv* env = sctx ? sctx->lmdb_env() : nullptr;
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
  if (mdb_dbi_open(txn, db_name.c_str(), 0, &dbi) != 0) {
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
      // them), else by hf_path. Misses (user-typed paths) fall back to a
      // plain "model" with empty I/O.
      const string name = fstr(ro, "name");
      const string hf = fstr(ro, "hf_path");
      const ModelCatalogEntry* ce = nullptr;
      if (!name.empty()) {
        ce = catalog_by_name(name);
      }
      if (!ce && !hf.empty()) {
        ce = catalog_by_path(hf);
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
        if (meo.contains("inputs")) { mo.insert("inputs", meo.at("inputs")); }
        if (meo.contains("outputs")) {
          mo.insert("outputs", meo.at("outputs"));
        }
      } else {
        mo.insert("category", FlexData::make_string("model"));
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
