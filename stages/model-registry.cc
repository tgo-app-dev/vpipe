#include "stages/model-registry.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>

namespace vpipe {

std::string
resolve_model_dir(const SessionContextIntf* session,
                  const std::string&        ref)
{
  if (!session) {
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
    LmdbDb  db(*env, kModelRegistryDb);
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
  session->info(fmt("model registry: '{}' -> '{}'", ref, local));
  return local;
}

bool
model_dir_available(const SessionContextIntf* session,
                    const std::string&        ref)
{
  if (ref.empty()) {
    return false;
  }
  const std::string dir = resolve_model_dir(session, ref);
  std::error_code ec;
  return std::filesystem::exists(dir, ec) && !ec;
}

bool
apply_model_select_beat(const FlexData& beat,
                        std::string&    hf_dir)
{
  std::string ref;
  if (beat.is_string()) {
    ref = std::string(beat.as_string(""));
  } else if (beat.is_object()) {
    auto o = beat.as_object();
    if (o.contains("hf_dir")) {
      ref = std::string(o.at("hf_dir").as_string(""));
    } else if (o.contains("model")) {
      ref = std::string(o.at("model").as_string(""));
    }
  }
  if (ref.empty()) {
    return false;
  }
  hf_dir = ref;
  return true;
}

}
