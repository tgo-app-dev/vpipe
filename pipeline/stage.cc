#include "pipeline/stage.h"
#include "common/job.h"
#include "common/perf-event.h"
#include <string>

using namespace std;

namespace vpipe {

Stage::Stage(const SessionContextIntf* s,
             string id,
             vector<InEdge> iports,
             FlexData config)
  : Vertex(s, std::move(id), std::move(iports))
  , _config(std::move(config))
{
}

// Default lifecycle hooks: empty coroutines. Stages override only
// the phases they care about; process() is pure-virtual and must be
// supplied.
Job
Stage::initialize(RuntimeContext& /*ctx*/)
{
  co_return;
}

Job
Stage::drain(RuntimeContext& /*ctx*/)
{
  co_return;
}

const StageSpec&
Stage::spec() const noexcept
{
  // Base default: an empty Generic spec. Stages override to return
  // their file-static kSpec.
  static const StageSpec kGeneric{};
  return kGeneric;
}

namespace {

// Lookup the ConfigKey for `key` in a spec span (nullptr if absent).
const ConfigKey*
find_key_(span<const ConfigKey> spec, string_view key)
{
  for (const ConfigKey& k : spec) {
    if (k.key == key) { return &k; }
  }
  return nullptr;
}

}  // namespace

bool
Stage::attr_present_(string_view key, FlexData& out) const
{
  const FlexData& cfg = config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    auto it   = root.find(key);
    if (it != root.end()) {
      out = (*it).second;
      return true;
    }
  }
  return false;
}

bool
Stage::attr_bool(string_view key) const
{
  FlexData v;
  if (attr_present_(key, v)) { return v.as_bool(); }
  const ConfigKey* k = find_key_(config_spec(), key);
  return k ? k->def_bool : false;
}

int64_t
Stage::attr_int(string_view key) const
{
  FlexData v;
  if (attr_present_(key, v)) { return v.as_int(); }
  const ConfigKey* k = find_key_(config_spec(), key);
  return k ? k->def_int : 0;
}

uint64_t
Stage::attr_uint(string_view key) const
{
  FlexData v;
  if (attr_present_(key, v)) { return v.as_uint(); }
  const ConfigKey* k = find_key_(config_spec(), key);
  return k ? k->def_uint : 0;
}

double
Stage::attr_real(string_view key) const
{
  FlexData v;
  if (attr_present_(key, v)) { return v.as_real(); }
  const ConfigKey* k = find_key_(config_spec(), key);
  return k ? k->def_real : 0.0;
}

string
Stage::attr_str(string_view key) const
{
  FlexData v;
  if (attr_present_(key, v)) { return string(v.as_string()); }
  const ConfigKey* k = find_key_(config_spec(), key);
  return k ? string(k->def_str) : string();
}

FlexData
Stage::attr(string_view key) const
{
  FlexData v;
  if (attr_present_(key, v)) { return v; }
  const ConfigKey* k = find_key_(config_spec(), key);
  return k ? config_default_value(*k) : FlexData::make_null();
}

string
Stage::perf_event_name(uint32_t type) const
{
  // Runtime-emitted worker scheduling events (see common/perf-event.h).
  // A stage that overrides this for its own events should chain to the
  // base for the reserved range.
  switch (type) {
    case kPerfSchedule:   return "schedule";
    case kPerfUnschedule: return "unschedule";
    default: break;
  }
  return "event_" + to_string(type);
}

vector<ConfigParam>
Stage::config_params() const
{
  return resolve_config_params(config_spec(), _config);
}

FlexData
Stage::config_schema() const
{
  return config_params_to_flex(config_params());
}

void
Stage::fail_config(const VpipeFormat& message)
{
  if (_config_error.empty()) {
    _config_error = message();
  }
}

void
StageLifecycleAccess::set_running(Stage* s, bool running)
{
  s->_running.store(running, memory_order_release);
}

void
StageLifecycleAccess::set_needs_init(Stage* s, bool needs_init)
{
  s->_needs_init.store(needs_init, memory_order_release);
}

}
