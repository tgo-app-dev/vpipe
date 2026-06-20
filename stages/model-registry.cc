#include "stages/model-registry.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

namespace vpipe {

std::string
resolve_model_dir(const SessionContextIntf* session,
                  const std::string&        models_db,
                  const std::string&        ref)
{
  if (!session || models_db.empty()) {
    return ref;
  }
  LmdbEnv* env = session->lmdb_env();
  if (!env) {
    return ref;
  }
  std::string local;
  try {
    // LmdbDb opens (creates if absent) the sub-db; a read txn then looks
    // up the key. No write txn is held, so the open-during-RW deadlock
    // doesn't apply.
    LmdbDb  db(*env, models_db);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadOnly);
    auto    view = db.get(txn, ref);
    if (!view) {
      return ref;
    }
    const std::string bytes(*view);   // copy before the txn ends
    txn.abort();
    FlexData rec = FlexData::from_binary(bytes);
    if (rec.is_object()) {
      auto obj = rec.as_object();
      if (obj.contains("local_path")) {
        local = std::string(obj.at("local_path").as_string(""));
      }
    }
  } catch (...) {
    return ref;
  }
  if (local.empty()) {
    return ref;
  }
  session->info(fmt(
      "model registry: '{}' -> '{}' (models DB '{}')",
      ref, local, models_db));
  return local;
}

}
