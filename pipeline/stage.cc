#include "pipeline/stage.h"
#include "pipeline/stage-consumer-types.h"
#include "common/job.h"
#include "pipeline/stage-command.h"
#include "common/path-sandbox.h"
#include "common/perf-event.h"
#include <string>

using namespace std;

namespace vpipe {

// Stage's own state (see the comment on Stage::_st). Host-private: no
// plugin sees this layout, so it grows freely.
class StageState {
public:
  explicit StageState(FlexData c) : config(std::move(c)) {}

  // Set by the runtime for the span of a launch; null otherwise. Atomic
  // because a revision comes from the stage's own thread while stop() may
  // be clearing it from another.
  std::atomic<MemoryPlanSink*> mem_sink{nullptr};

  FlexData          config;
  std::string       config_error;
  std::atomic<bool> running{false};
  std::atomic<bool> needs_init{false};
  // Set and cleared by the runtime (StageLifecycleAccess) while callers
  // on other threads read it, hence the lock.
  mutable std::mutex            inbox_mu;
  std::shared_ptr<CommandInbox> inbox;
};

Stage::Stage(const SessionContextIntf* s,
             string id,
             vector<InEdge> iports,
             FlexData config)
  : Vertex(s, std::move(id), std::move(iports))
  , _st(std::make_unique<StageState>(std::move(config)))
{
}

Stage::~Stage() = default;

const FlexData&
Stage::config() const noexcept
{
  return _st->config;
}

const string&
Stage::config_error() const noexcept
{
  return _st->config_error;
}

bool
Stage::running() const noexcept
{
  return _st->running.load(memory_order_acquire);
}

bool
Stage::needs_init() const noexcept
{
  return _st->needs_init.load(memory_order_acquire);
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

string
Stage::confine_local_(string_view raw, bool for_write)
{
  if (raw.empty() || is_network_url(raw)) {
    return string(raw);   // empty (handled by the stage) or a network URL
  }
  string_view p = raw;
  if (p.rfind("file://", 0) == 0) { p.remove_prefix(7); }
  string err;
  const SessionContextIntf* sess = session();
  string resolved = sess
      ? sess->confine_path(p, for_write, &err).string()
      : string(p);
  if (!err.empty()) {
    fail_config(fmt(
        "path '{}' is blocked by the file sandbox: {}", string(raw), err));
    return {};
  }
  return resolved;
}

string
Stage::attr_path(string_view key, bool for_write)
{
  return confine_local_(attr_str(key), for_write);
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
  vector<ConfigParam> params = resolve_config_params(config_spec(),
                                                     _st->config);
  // A model picker that is a channel CONSUMER (it declares the types it
  // runs) also offers the families a plugin registered into THIS stage's
  // registry: its static list was compiled before they existed. A
  // channel SOURCE already offers them through the channel union.
  const vector<string> extra = consumer_types(spec().type_name);
  if (!extra.empty()) {
    const auto keys = config_spec();
    for (size_t i = 0; i < params.size() && i < keys.size(); ++i) {
      if (keys[i].model_channel.empty() ||
          keys[i].suggest_db_type.empty()) {
        continue;
      }
      string& csv = params[i].suggest_db_type;
      for (const string& t : extra) {
        // Whole-entry match: "foo" must not be taken as present because
        // "foo-21" is.
        const string padded = "," + csv + ",";
        if (padded.find("," + t + ",") != string::npos) { continue; }
        if (!csv.empty()) { csv += ','; }
        csv += t;
      }
    }
  }
  return params;
}

FlexData
Stage::config_schema() const
{
  return config_params_to_flex(config_params());
}

void
Stage::fail_config(const VpipeFormat& message)
{
  if (_st->config_error.empty()) {
    _st->config_error = message();
  }
}

void
StageLifecycleAccess::set_running(Stage* s, bool running)
{
  s->_st->running.store(running, memory_order_release);
}

void
Stage::revise_memory(const StageMemory& m) const
{
  if (MemoryPlanSink* sink = _st->mem_sink.load(std::memory_order_acquire)) {
    sink->revise(this, m);
  }
}

void
StageLifecycleAccess::set_memory_sink(Stage* s, MemoryPlanSink* sink)
{
  if (s != nullptr) {
    s->_st->mem_sink.store(sink, std::memory_order_release);
  }
}

void
StageLifecycleAccess::set_command_inbox(Stage* s,
                                        shared_ptr<CommandInbox> inbox)
{
  if (s == nullptr) { return; }
  lock_guard<mutex> lk(s->_st->inbox_mu);
  s->_st->inbox = std::move(inbox);
}

shared_ptr<CommandInbox>
Stage::command_inbox() const
{
  lock_guard<mutex> lk(_st->inbox_mu);
  return _st->inbox;
}

void
StageLifecycleAccess::set_needs_init(Stage* s, bool needs_init)
{
  s->_st->needs_init.store(needs_init, memory_order_release);
}

}
