#include "stages/model-registry.h"

#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <filesystem>

namespace vpipe {

ResolvedModel
resolve_model(const SessionContextIntf* session, const std::string& ref)
{
  ResolvedModel out;
  out.dir = ref;
  if (!session) {
    return out;
  }
  // An empty reference cannot name a model, and LMDB will not be asked
  // whether it does: mdb_get rejects a zero-length key with
  // MDB_BAD_VALSIZE, which surfaces as a registry ERROR naming neither
  // the caller nor what it was looking for. A stage whose model arrives
  // on an iport has an empty hf_dir until that beat lands, so reaching
  // here with "" is ordinary rather than exceptional -- it deserves an
  // empty answer, not a database error.
  if (ref.empty()) {
    return out;
  }
  LmdbEnv* env = session->services()->lmdb_env();
  if (!env) {
    return out;
  }
  std::string local;
  std::string mtype;
  std::vector<std::string> pinned;
  try {
    // LmdbDb opens (creates if absent) the sub-db; a read txn then looks
    // up the key. No write txn is held, so the open-during-RW deadlock
    // doesn't apply.
    LmdbDb  db(*env, kModelRegistryDb);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadOnly);
    auto    view = db.get(txn, ref);
    if (!view) {
      return out;
    }
    const std::string bytes(*view);   // copy before the txn ends
    txn.abort();
    FlexData rec = FlexData::from_binary(bytes);
    if (rec.is_object()) {
      auto obj = rec.as_object();
      if (obj.contains("local_path")) {
        local = std::string(obj.at("local_path").as_string(""));
      }
      if (obj.contains("model_type")) {
        mtype = std::string(obj.at("model_type").as_string(""));
      }
      if (obj.contains("files")) {
        const FlexData fl = obj.at("files");
        if (fl.is_array()) {
          const auto fa = fl.as_array();
          for (std::size_t i = 0; i < fa.size(); ++i) {
            const std::string f(fa[i].as_string(""));
            if (!f.empty()) { pinned.push_back(f); }
          }
        }
      }
    }
  } catch (...) {
    return out;
  }
  if (local.empty()) {
    return out;
  }
  session->info(fmt("model registry: '{}' -> '{}'", ref, local));
  out.dir           = local;
  out.key           = ref;
  out.model_type    = mtype;
  out.files         = std::move(pinned);
  out.from_registry = true;
  return out;
}

std::string
resolved_subtree_dir(const ResolvedModel& rm)
{
  namespace fs = std::filesystem;
  if (rm.files.empty()) { return rm.dir; }
  // The common DIRECTORY prefix, component by component. A file's own
  // name never contributes -- two records pinning `a/x` and `a/y` share
  // `a`, and one pinning `a/x` alone shares `a` with itself.
  std::vector<std::string> pre;
  bool first = true;
  for (const std::string& f : rm.files) {
    std::vector<std::string> parts;
    const fs::path p(f);
    for (auto it = p.begin(); it != p.end(); ++it) {
      parts.push_back(it->string());
    }
    if (parts.empty()) { return rm.dir; }
    parts.pop_back();                       // drop the file name
    if (first) {
      pre = std::move(parts);
      first = false;
      continue;
    }
    std::size_t n = 0;
    while (n < pre.size() && n < parts.size() && pre[n] == parts[n]) { ++n; }
    pre.resize(n);
    if (pre.empty()) { return rm.dir; }
  }
  fs::path out(rm.dir);
  for (const std::string& c : pre) { out /= c; }
  return out.string();
}

std::string
resolve_model_dir(const SessionContextIntf* session,
                  const std::string&        ref)
{
  return resolve_model(session, ref).dir;
}

std::string
resolve_adapter_file(const SessionContextIntf* session,
                     const std::string& ref, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return std::string();
  };
  if (ref.empty()) { return fail("no adapter named"); }
  std::error_code ec;
  if (fs::is_regular_file(fs::path(ref), ec) && !ec) { return ref; }

  const ResolvedModel rm = resolve_model(session, ref);
  if (!rm.from_registry && !fs::exists(fs::path(rm.dir), ec)) {
    return fail("'" + ref + "' is neither a file, a registered model, nor a "
                "directory");
  }
  // The record's OWN pinned file wins. Two records can share a directory
  // -- both MiniMax-H3 Turbo checkpoints are published from one repo --
  // and only the record says which of them this key meant.
  for (const std::string& f : rm.files) {
    const fs::path p = fs::path(rm.dir) / f;
    if (p.extension() == ".safetensors" && fs::is_regular_file(p, ec)) {
      return p.string();
    }
  }
  if (fs::is_regular_file(fs::path(rm.dir), ec)) { return rm.dir; }
  if (!fs::is_directory(fs::path(rm.dir), ec)) {
    return fail("'" + ref + "' resolves to '" + rm.dir + "', which is not a "
                "file or a directory");
  }
  // A bare directory. One .safetensors is an answer; several are not --
  // directory-iteration order is not a choice anybody made, so this
  // names them and refuses rather than picking.
  std::vector<std::string> hits;
  for (const auto& e : fs::directory_iterator(fs::path(rm.dir), ec)) {
    if (e.path().extension() == ".safetensors") {
      hits.push_back(e.path().string());
    }
  }
  if (hits.empty()) {
    return fail("no .safetensors in '" + rm.dir + "'");
  }
  if (hits.size() > 1) {
    std::sort(hits.begin(), hits.end());
    std::string list;
    for (const std::string& h : hits) {
      list += "\n  - " + fs::path(h).filename().string();
    }
    return fail("'" + rm.dir + "' holds " + std::to_string(hits.size()) +
                " .safetensors and nothing says which one is meant; name the "
                "file, or use the registry key that pinned it:" + list);
  }
  return hits[0];
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
