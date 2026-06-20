#include "pipeline/stage-registry.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <exception>
#include <mutex>
#include <utility>

using namespace std;

namespace vpipe {

StageRegistry&
StageRegistry::get() noexcept
{
  static StageRegistry x;
  return x;
}

StageTypeId
StageRegistry::register_type(string_view type_name, Factory f)
{
  lock_guard<mutex> lk(_mu);
  auto it = _by_name.find(type_name);
  if (it != _by_name.end()) {
    return it->second;
  }
  unsigned id = static_cast<unsigned>(_entries.size()) + 1;
  StageTypeId tid = static_cast<StageTypeId>(id);
  _entries.push_back(Entry{string(type_name), f});
  _by_name.emplace(_entries.back().name, tid);
  return tid;
}

StageTypeId
StageRegistry::find_id(string_view type_name) const noexcept
{
  lock_guard<mutex> lk(_mu);
  auto it = _by_name.find(type_name);
  if (it == _by_name.end()) {
    return StageTypeId::unknown;
  }
  return it->second;
}

void
StageRegistry::set_spec(string_view type_name,
                        const StageSpec* spec) noexcept
{
  lock_guard<mutex> lk(_mu);
  // First registration sticks (mirrors register_type's duplicate
  // handling); a second include of the same stage is a no-op.
  _specs.emplace(string(type_name), spec);
}

const StageSpec*
StageRegistry::spec(string_view type_name) const noexcept
{
  lock_guard<mutex> lk(_mu);
  auto it = _specs.find(type_name);
  return it == _specs.end() ? nullptr : it->second;
}

string_view
StageRegistry::find_name(StageTypeId tid) const noexcept
{
  lock_guard<mutex> lk(_mu);
  unsigned id = static_cast<unsigned>(tid);
  if (id == 0 || id > _entries.size()) {
    return {};
  }
  return _entries[id - 1].name;
}

StagePtr
StageRegistry::create(string_view               type_name,
                      const SessionContextIntf* session,
                      string                    id,
                      vector<InEdge>            iports,
                      FlexData                  config) const
{
  Factory f;
  {
    lock_guard<mutex> lk(_mu);
    auto it = _by_name.find(type_name);
    if (it == _by_name.end()) {
      if (session) {
        session->warn(fmt(
            "StageRegistry::create: unknown stage type '{}'",
            type_name));
      }
      return nullptr;
    }
    unsigned tid = static_cast<unsigned>(it->second);
    f = _entries[tid - 1].factory;
  }
  // A stage ctor that detects a malformed config calls
  // session->error(...), which logs at Error level and then throws
  // (see Session::error). Catch here so a bad config does not
  // unwind the caller -- the public PipelineHandle::insert_stage
  // path expects nullptr on construction failure, and direct
  // callers of the registry get the same.
  try {
    return f(session, std::move(id), std::move(iports),
             std::move(config));
  } catch (const exception& e) {
    if (session) {
      session->warn(fmt(
          "StageRegistry::create('{}'): stage ctor failed: {}",
          type_name, e.what()));
    }
    return nullptr;
  } catch (...) {
    if (session) {
      session->warn(fmt(
          "StageRegistry::create('{}'): stage ctor threw a "
          "non-std exception",
          type_name));
    }
    return nullptr;
  }
}

vector<pair<StageTypeId, string>>
StageRegistry::all() const
{
  lock_guard<mutex> lk(_mu);
  vector<pair<StageTypeId, string>> out;
  out.reserve(_entries.size());
  for (size_t i = 0; i < _entries.size(); ++i) {
    out.emplace_back(static_cast<StageTypeId>(i + 1),
                     _entries[i].name);
  }
  return out;
}

}
