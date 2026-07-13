#include "common/session.h"
#include "common/db-log-delegate.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/i18n.h"
#include "common/library-handle.h"
#include "common/lmdb-env.h"
#include "common/session-config.h"
#include "common/stdio-ui-delegate.h"
#include "common/stdout-log-delegate.h"
#include "common/thread-pool.h"
#include "common/vpipe-format.h"

#include <filesystem>
#include <system_error>
#include "common/perf-buffer.h"
#include "common/perf-event.h"
#include "interfaces/log-delegate-intf.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline-spec.h"
#include "pipeline/pipeline.h"
#include "pipeline/stage.h"
#include "plugin/plugin-manager.h"
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_set>
// Header is portable (forward-declares CML::* on Apple; nothing
// CoreML-specific). Include unconditionally so unique_ptr<...>
// member destruction sees a complete type even on builds where
// VPIPE_BUILD_APPLE_SILICON is off and the .cc impl isn't linked.
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "generative-models/generative-model-manager.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

using namespace std;

namespace vpipe {

namespace {

// Parsed view of the config's `log` block. Read once at boot.
// `log.delegate=db` opts the session into the DB delegate; the
// path / map_size live at the top-level `db` block (parsed below)
// because the env is shared by every SessionMember, not just by
// logging.
struct LogConfigState {
  LogLevel level = LogLevel::Normal;
  string   kind  = "stdout";  // "stdout" | "db"
};

void
parse_log_config(const FlexData& config, LogConfigState* out)
{
  // Note on lifetimes: ConstObjectView::iterator's operator* yields a
  // pair<string_view, FlexData> *by value*; the FlexData is a clone
  // owned by the returned pair. A view obtained from such a temporary
  // dangles as soon as the full expression ends. So at every nesting
  // level we copy the child FlexData into a named local before
  // taking a view from it.
  if (!config.is_object()) {
    return;
  }
  auto root = config.as_object();
  if (!root.contains("log")) {
    return;
  }
  FlexData log_val = root.at("log");
  if (!log_val.is_object()) {
    return;
  }
  auto log_obj = log_val.as_object();

  if (log_obj.contains("level")) {
    FlexData v = log_obj.at("level");
    if (v.is_string()) {
      out->level = parse_log_level(v.get_string(), out->level);
    }
  }
  if (log_obj.contains("delegate")) {
    FlexData v = log_obj.at("delegate");
    if (v.is_string()) {
      out->kind = string(v.get_string());
    }
  }
}

// Reads the top-level `db.path` / `db.map_size_mb` from the session
// config. An empty `db.path` is fine: `Session::lmdb_env()` falls
// back to the process CWD (".") at first-open time.
void
parse_db_config(const FlexData& config,
                string*         out_path,
                size_t*         out_map_size)
{
  *out_path     = "";
  *out_map_size = DbLogDelegate::kDefaultMapSize;
  if (!config.is_object()) {
    return;
  }
  auto root = config.as_object();
  if (!root.contains("db")) {
    return;
  }
  FlexData db_val = root.at("db");
  if (!db_val.is_object()) {
    return;
  }
  auto db_obj = db_val.as_object();
  if (db_obj.contains("path")) {
    FlexData v = db_obj.at("path");
    if (v.is_string()) {
      *out_path = string(v.get_string());
    }
  }
  if (db_obj.contains("map_size_mb")) {
    FlexData v = db_obj.at("map_size_mb");
    if (v.is_uint() || v.is_int()) {
      uint64_t mb = v.is_uint()
        ? v.get_uint()
        : static_cast<uint64_t>(v.get_int());
      *out_map_size = static_cast<size_t>(mb) << 20;
    }
  }
}

// Read the "file_sandbox" config: { enabled: bool, root: string }. When
// enabled, stage file paths are confined (chroot-like) to `root`
// (default "<cwd>/sandbox"), which is created here. Unset / disabled ->
// a disabled PathSandbox (native access). Fail-soft like the other
// parse_*_config helpers.
PathSandbox
parse_sandbox_config(const FlexData& config)
{
  if (!config.is_object()) {
    return PathSandbox{};
  }
  auto root = config.as_object();
  if (!root.contains("file_sandbox")) {
    return PathSandbox{};
  }
  FlexData fs_val = root.at("file_sandbox");
  if (!fs_val.is_object()) {
    return PathSandbox{};
  }
  auto obj = fs_val.as_object();
  bool enabled = false;
  if (obj.contains("enabled")) {
    FlexData v = obj.at("enabled");
    if (v.is_bool()) { enabled = v.get_bool(); }
  }
  if (!enabled) {
    return PathSandbox{};
  }
  std::string dir;
  if (obj.contains("root")) {
    FlexData v = obj.at("root");
    if (v.is_string()) { dir = std::string(v.get_string()); }
  }
  if (dir.empty()) {
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    dir = ((ec ? std::filesystem::path(".") : cwd) / "sandbox").string();
  }
  // Optional whitelist: real host prefixes granted pass-through access
  // (an array of path strings). These are existing directories the
  // operator explicitly exposes -- do NOT create them here.
  std::vector<std::filesystem::path> whitelist;
  if (obj.contains("whitelist")) {
    FlexData v = obj.at("whitelist");
    if (v.is_array()) {
      auto arr = v.as_array();
      for (size_t i = 0; i < arr.size(); ++i) {
        FlexData e = arr.at(i);
        if (e.is_string()) {
          std::string p(e.get_string());
          if (!p.empty()) { whitelist.emplace_back(std::move(p)); }
        }
      }
    }
  }
  // Self-maintained: create the root so confine() has a real directory
  // to canonicalize against and land paths under.
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return PathSandbox{std::filesystem::path(dir), std::move(whitelist)};
}

// Read the top-level "language" config key (an IETF tag) and normalize
// it. Unset / unsupported -> the default locale. Mirrors the fail-soft
// style of the other parse_*_config helpers.
string
parse_language_config(const FlexData& config)
{
  if (config.is_object()) {
    auto root = config.as_object();
    if (root.contains("language")) {
      FlexData v = root.at("language");
      if (v.is_string()) {
        string norm = normalize_language(v.get_string());
        if (!norm.empty()) {
          return norm;
        }
      }
    }
  }
  return string(default_language());
}

unique_ptr<LogDelegateIntf>
build_delegate(const LogConfigState& state, const Session* reporter)
{
  if (state.kind == "db") {
    LmdbEnv* env = reporter->lmdb_env();
    if (!env) {
      reporter->warn(fmt(
        "log.delegate=db: lmdb_env unavailable (open of '{}' "
        "failed); falling back to stdout",
        state.kind));
      return make_unique<StdoutLogDelegate>(state.level);
    }
    try {
      return make_unique<DbLogDelegate>(env, state.level);
    } catch (const exception& e) {
      reporter->warn(fmt(
        "DbLogDelegate construction failed: {}; falling back to "
        "stdout", e.what()));
      return make_unique<StdoutLogDelegate>(state.level);
    }
  }
  if (state.kind != "stdout") {
    reporter->warn(fmt(
      "unknown log.delegate '{}'; falling back to stdout",
      state.kind));
  }
  return make_unique<StdoutLogDelegate>(state.level);
}

LogLevel
clamp_log_level(unsigned u)
{
  // LogLevel valid range: 0..6 (Error..Always). Map out-of-range to
  // Debug so callers asking for "more verbose than the enum" still
  // get the most-verbose informational level (Always is a sentinel
  // and not a sensible threshold).
  if (u > static_cast<unsigned>(LogLevel::Debug)) {
    return LogLevel::Debug;
  }
  return static_cast<LogLevel>(u);
}

unsigned
default_workers() noexcept
{
  // Leave a few cores for the OS, the UI/log delegates, and the GPU /
  // ANE driver threads the LLM + vision stages lean on: max(1, cores-3).
  // hardware_concurrency() can report 0 (unknown) -- treat as 1 core.
  unsigned hc = std::thread::hardware_concurrency();
  int n = static_cast<int>(hc) - 3;
  return static_cast<unsigned>(n > 1 ? n : 1);
}

unsigned
read_pool_workers(const FlexData& config) noexcept
{
  if (!config.is_object()) {
    return default_workers();
  }
  try {
    auto root = config.as_object();
    if (!root.contains("pool")) {
      return default_workers();
    }
    FlexData pool_val = root.at("pool");
    if (!pool_val.is_object()) {
      return default_workers();
    }
    auto pool_obj = pool_val.as_object();
    if (!pool_obj.contains("num_workers")) {
      return default_workers();
    }
    FlexData v = pool_obj.at("num_workers");
    if (v.is_uint()) {
      uint64_t n = v.get_uint();
      return n > 0 ? static_cast<unsigned>(n) : 1u;
    }
    if (v.is_int()) {
      int64_t n = v.get_int();
      return n > 0 ? static_cast<unsigned>(n) : 1u;
    }
  } catch (...) {
  }
  return default_workers();
}

// If `d` is a StdoutLogDelegate, hand it the session so it can
// switch from the sync (bootstrap) path to the per-worker async
// path. No-op for other delegate types.
void
attach_if_stdout_(LogDelegateIntf* d, const SessionContextIntf* sess)
{
  if (auto* sd = dynamic_cast<StdoutLogDelegate*>(d)) {
    sd->attach(sess);
  }
}

unsigned
read_default_edge_capacity(const FlexData& config) noexcept
{
  constexpr unsigned kDefault = 4;
  if (!config.is_object()) {
    return kDefault;
  }
  try {
    auto root = config.as_object();
    if (!root.contains("pipeline")) {
      return kDefault;
    }
    FlexData pl_val = root.at("pipeline");
    if (!pl_val.is_object()) {
      return kDefault;
    }
    auto pl_obj = pl_val.as_object();
    if (!pl_obj.contains("default_edge_capacity")) {
      return kDefault;
    }
    FlexData v = pl_obj.at("default_edge_capacity");
    if (v.is_uint()) {
      uint64_t n = v.get_uint();
      return n > 0 ? static_cast<unsigned>(n) : 1u;
    }
    if (v.is_int()) {
      int64_t n = v.get_int();
      return n > 0 ? static_cast<unsigned>(n) : 1u;
    }
  } catch (...) {
  }
  return kDefault;
}


}

// Collect plugin .dylib paths to load: the session config's `plugins`
// array (path strings) followed by the colon-separated VPIPE_PLUGINS
// environment variable. Config entries come first.
static vector<string>
collect_plugin_paths_(const FlexData& config)
{
  vector<string> paths;
  if (config.is_object()) {
    auto obj = config.as_object();
    if (obj.contains("plugins")) {
      FlexData v = obj.at("plugins");
      if (v.is_array()) {
        auto arr = v.as_array();
        for (size_t i = 0; i < arr.size(); ++i) {
          FlexData e = arr.at(i);
          if (e.is_string()) {
            string p(e.get_string());
            if (!p.empty()) { paths.push_back(std::move(p)); }
          }
        }
      }
    }
  }
  if (const char* env = std::getenv("VPIPE_PLUGINS")) {
    const string s(env);
    size_t start = 0;
    for (;;) {
      const size_t colon = s.find(':', start);
      string p = (colon == string::npos) ? s.substr(start)
                                          : s.substr(start, colon - start);
      if (!p.empty()) { paths.push_back(std::move(p)); }
      if (colon == string::npos) { break; }
      start = colon + 1;
    }
  }
  return paths;
}

Session::Session(string_view cfg)
  : _config(FlexData::make_object())
  , _delegate(make_unique<StdoutLogDelegate>(LogLevel::Normal))
  , _ui_delegate(make_unique<StdioUiDelegate>())
  , _db_map_size(DbLogDelegate::kDefaultMapSize)
{
  // Phase 1: parse config. Fail-soft: a parse error becomes a warning
  // through the bootstrap stdout delegate, and we keep going with an
  // empty config object.
  try {
    _config = parse_session_config(cfg);
  } catch (const exception& e) {
    warn(fmt("session config parse failed: {}; using defaults",
             e.what()));
    _config = FlexData::make_object();
    _pool = make_unique<ThreadPool>(default_workers(), this);
    attach_if_stdout_(_delegate.get(), this);
    return;
  }

  // Phase 2: read top-level db config. lmdb_env() is lazy -- the env
  // doesn't open here unless build_delegate() pulls it.
  parse_db_config(_config, &_db_path, &_db_map_size);
  _path_sandbox = parse_sandbox_config(_config);

  // Phase 2b: UI/message locale (default en-us when unset/unsupported).
  _language = parse_language_config(_config);

  // Phase 3: build the chosen log delegate. The helper itself is
  // fail-soft -- on any inner failure it warns through `this`
  // (which still has the bootstrap delegate at this point) and
  // returns a StdoutLogDelegate.
  LogConfigState log_state;
  parse_log_config(_config, &log_state);
  auto chosen = build_delegate(log_state, this);
  if (chosen) {
    _delegate = std::move(chosen);
  }

  // Phase 4: read pool / pipeline knobs and construct the pool. The
  // pool logs through `this` so the delegate must be in place first.
  unsigned workers       = read_pool_workers(_config);
  _default_edge_capacity = read_default_edge_capacity(_config);
  _pool = make_unique<ThreadPool>(workers, this);
  // Pool is up; switch a StdoutLogDelegate (bootstrap or chosen)
  // from sync to per-worker async logging.
  attach_if_stdout_(_delegate.get(), this);

  // Phase 5: load plugin .dylibs named by the `plugins` config array and
  // the VPIPE_PLUGINS env, now that the delegate + pool are up so their
  // registrations + diagnostics are visible. Process-wide + dedup'd, so a
  // plugin loads once even across multiple sessions.
  const vector<string> plugin_paths = collect_plugin_paths_(_config);
  if (!plugin_paths.empty()) {
    PluginManager::get().load_all(this, plugin_paths);
  }
}

Session::Session(unique_ptr<LogDelegateIntf> d)
  : _config(FlexData::make_object())
  , _delegate(d ? std::move(d)
                : make_unique<StdoutLogDelegate>(LogLevel::Normal))
  , _ui_delegate(make_unique<StdioUiDelegate>())
  , _db_map_size(DbLogDelegate::kDefaultMapSize)
{
  _pool = make_unique<ThreadPool>(default_workers(), this);
  attach_if_stdout_(_delegate.get(), this);
}

Session::~Session()
{
  // Drop pipelines (and their runtimes) before the pool is destroyed:
  // each PipelineHandleImpl's dtor calls PipelineRuntime::stop()
  // which schedules wakes onto _pool. After this scope, no more
  // pipeline-driven schedules will happen, so the pool can join
  // workers safely.
  {
    lock_guard<mutex> lk(_pipelines_mu);
    _pipelines.clear();
  }
  // Tear the delegate down BEFORE the pool. A StdoutLogDelegate's
  // dtor calls detach() which schedules a wake on the pool to let
  // its consumer coroutine reach final_suspend; the pool must
  // still be alive to service that wake. A DbLogDelegate dtor only
  // touches _env (handled below), so the order is fine for that
  // case too.
  _delegate.reset();
  _pool.reset();
  // The DbLogDelegate would have borrowed _env; with the delegate
  // gone above, the env can safely be torn down.
  _env.reset();
  // _ffmpeg's dtor needs the full FFmpegLibraries type, which is
  // available here (the .cc includes it). Default member dtor order
  // would do the same thing -- explicit reset just makes the
  // dependency on the include obvious.
  _ffmpeg.reset();
  // _config destroyed by member dtor as usual.
}

const FFmpegLibraries*
Session::ffmpeg_libraries() const
{
  // call_once guarantees exactly one successful construction across
  // all racing callers. If construction throws, the once_flag is
  // *not* set as fulfilled and a later call retries -- which is the
  // desired behavior for a Required-mode load that failed because
  // of a transient environment issue.
  std::call_once(_ffmpeg_once, [this] {
    _ffmpeg = make_unique<FFmpegLibraries>(
      this, LibraryHandle::LoadMode::Required);
  });
  return _ffmpeg.get();
}

CoreMLModelManager*
Session::coreml_model_manager() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The manager itself does no I/O at construction; failure
  // surfaces from individual `manager->load(...)` calls. So a plain
  // call_once with no try/catch is enough.
  std::call_once(_coreml_mgr_once, [this] {
    _coreml_mgr = make_unique<CoreMLModelManager>(this);
  });
  return _coreml_mgr.get();
#else
  return nullptr;
#endif
}

metal_compute::MetalCompute*
Session::metal_compute() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Construction acquires MTL::CreateSystemDefaultDevice; failure is
  // reported via valid()==false on the instance, not exceptions, so
  // a single call_once is enough.
  std::call_once(_mc_once, [this] {
    _mc = make_unique<metal_compute::MetalCompute>(this);
  });
  return _mc.get();
#else
  return nullptr;
#endif
}

genai::GenerativeModelManager*
Session::generative_model_manager() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The LM subsystem runs on the metal-compute backend without MLX, so
  // the manager is available under VPIPE_BUILD_APPLE_SILICON. Manager
  // construction is trivial (just stores the session pointer); the
  // heavy lifting happens inside `load()` against a specific LoadSpec.
  std::call_once(_llm_mgr_once, [this] {
    _llm_mgr = make_unique<genai::GenerativeModelManager>(this);
  });
  return _llm_mgr.get();
#else
  return nullptr;
#endif
}

std::filesystem::path
Session::confine_path(std::string_view user_path, bool for_write,
                      std::string* err) const
{
  return _path_sandbox.confine(user_path, for_write, err);
}

LmdbEnv*
Session::lmdb_env() const
{
  // Resolve the path: if the config didn't set db.path we fall
  // back to "." (the process CWD at first-open time). LMDB
  // resolves "." against CWD inside mdb_env_open, so we capture
  // whatever the caller's working directory is when the env is
  // first materialized.
  const string path = _db_path.empty() ? string(".") : _db_path;

  // Lazy open. If the open throws, _env stays null and the
  // once_flag is *not* fulfilled -- a later call will retry. The
  // env reports its construction failures through `this` (the
  // session), so the message lands in the active log delegate.
  std::call_once(_env_once, [this, &path] {
    try {
      _env = make_unique<LmdbEnv>(this, path, _db_map_size);
    } catch (const exception& e) {
      warn(fmt(
        "Session: lmdb_env open failed for '{}': {}; lmdb_env() "
        "will return nullptr",
        path, e.what()));
      _env.reset();
    }
  });
  return _env.get();
}

PipelineHandleImpl*
Session::resolve(PipelineHandle h) const
{
  PipelineHandleImpl* p = HandleAccess::impl(h);
  if (!p) {
    return nullptr;
  }
  lock_guard<mutex> lk(_pipelines_mu);
  return _pipelines.count(p) ? p : nullptr;
}

namespace {

// Read an entire file into a string. Returns nullopt on I/O error.
optional<string>
slurp_file_(const string& path)
{
  ifstream in(path, ios::binary);
  if (!in) {
    return nullopt;
  }
  ostringstream buf;
  buf << in.rdbuf();
  if (!in && !in.eof()) {
    return nullopt;
  }
  return buf.str();
}

// Peek the first non-whitespace byte. Returns true if it's `{` or
// `[`, indicating JSON; false otherwise (treat as binary FlexData).
bool
looks_like_json_(string_view s)
{
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      continue;
    }
    return c == '{' || c == '[';
  }
  return false;
}

// Lower-case ASCII suffix from a path. Returns "" if no `.` is in
// the basename.
string
ext_of_(const string& path)
{
  auto slash = path.find_last_of("/\\");
  string base = (slash == string::npos)
              ? path : path.substr(slash + 1);
  auto dot = base.rfind('.');
  if (dot == string::npos) {
    return "";
  }
  string ext = base.substr(dot);
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext;
}

}

PipelineHandle
Session::load_pipeline(string_view spec_sv)
{
  string input(spec_sv);
  if (input.empty()) {
    warn(fmt("load_pipeline: empty input"));
    return HandleAccess::make_pipeline(nullptr);
  }

  // A leading '{' or '[' (after whitespace) marks an inline JSON spec
  // document; anything else is a filesystem path to a JSON / binary-
  // FlexData spec file. Only a path becomes the handle's storage path --
  // an inline spec has no file to round-trip back to, so a later no-arg
  // store_pipeline() on it reports "no URL" until one is set.
  const bool is_inline = looks_like_json_(input);

  FlexData spec;
  string   storage;
  try {
    if (is_inline) {
      spec = FlexData::from_json(input);
    } else {
      auto contents = slurp_file_(input);
      if (!contents) {
        warn(fmt("load_pipeline: failed to read '{}'", input));
        return HandleAccess::make_pipeline(nullptr);
      }
      spec = looks_like_json_(*contents) ? FlexData::from_json(*contents)
                                         : FlexData::from_binary(*contents);
      storage = input;
    }
  } catch (const exception& e) {
    if (is_inline) {
      warn(fmt("load_pipeline: parse failed for inline spec: {}",
               e.what()));
    } else {
      warn(fmt("load_pipeline: parse failed for '{}': {}", input,
               e.what()));
    }
    return HandleAccess::make_pipeline(nullptr);
  }

  auto pipeline = pipeline_from_spec(spec, this);
  if (!pipeline) {
    // pipeline_from_spec already logged the underlying cause.
    return HandleAccess::make_pipeline(nullptr);
  }

  auto impl = make_unique<PipelineHandleImpl>(std::move(pipeline),
                                              this);
  if (!storage.empty()) {
    impl->storage_path(storage);
  }
  PipelineHandleImpl* raw = impl.get();
  {
    lock_guard<mutex> lk(_pipelines_mu);
    _pipelines.emplace(raw, std::move(impl));
  }
  return HandleAccess::make_pipeline(raw);
}

PipelineHandle
Session::create_pipeline(string id)
{
  auto pl = make_unique<Pipeline>(std::move(id), this);
  auto impl = make_unique<PipelineHandleImpl>(std::move(pl), this);
  PipelineHandleImpl* raw = impl.get();
  {
    lock_guard<mutex> lk(_pipelines_mu);
    _pipelines.emplace(raw, std::move(impl));
  }
  return HandleAccess::make_pipeline(raw);
}

Status
Session::launch_pipeline(PipelineHandle h)
{
  PipelineHandleImpl* impl = resolve(h);
  if (!impl) {
    warn(fmt("launch_pipeline: unknown handle"));
    return Status{1};
  }
  return impl->launch() ? Status{0} : Status{2};
}

Status
Session::pause_pipeline(PipelineHandle h)
{
  PipelineHandleImpl* impl = resolve(h);
  if (!impl) {
    return Status{1};
  }
  impl->pause();
  return Status{0};
}

Status
Session::stop_pipeline(PipelineHandle h)
{
  PipelineHandleImpl* impl = resolve(h);
  if (!impl) {
    return Status{1};
  }
  impl->stop();
  return Status{0};
}

Status
Session::unload_pipeline(PipelineHandle h)
{
  PipelineHandleImpl* impl = resolve(h);
  if (!impl) {
    return Status{1};
  }
  if (impl->launched()) {
    impl->stop();
  }
  unique_ptr<PipelineHandleImpl> removed;
  {
    lock_guard<mutex> lk(_pipelines_mu);
    auto it = _pipelines.find(impl);
    if (it != _pipelines.end()) {
      removed = std::move(it->second);
      _pipelines.erase(it);
    }
  }
  // 'removed' destructed here, after the map lock is released.
  return Status{0};
}

Status
Session::store_pipeline(PipelineHandle h)
{
  PipelineHandleImpl* impl = resolve(h);
  if (!impl) {
    warn(fmt("store_pipeline: unknown handle"));
    return Status{1};
  }
  const string& path = impl->storage_path();
  if (path.empty()) {
    warn(fmt(
      "store_pipeline: pipeline '{}' has no associated storage "
      "URL; use the path-taking overload to set one",
      impl->pipeline()->id()));
    return Status{1};
  }

  FlexData spec = pipeline_to_spec(*impl->pipeline());
  string ext = ext_of_(path);
  string serialized;
  try {
    if (ext == ".json") {
      serialized = spec.to_json(/* pretty */ true);
    } else {
      serialized = spec.to_binary();
    }
  } catch (const exception& e) {
    warn(fmt(
      "store_pipeline: serialization failed for '{}': {}",
      path, e.what()));
    return Status{2};
  }

  ofstream out(path, ios::binary);
  if (!out) {
    warn(fmt("store_pipeline: failed to open '{}' for writing", path));
    return Status{2};
  }
  out.write(serialized.data(),
            static_cast<streamsize>(serialized.size()));
  if (!out) {
    warn(fmt("store_pipeline: write failed for '{}'", path));
    return Status{2};
  }
  return Status{0};
}

Status
Session::wait_pipelines(int timeout_ms)
{
  // Snapshot the launched impl pointers under the map lock, then
  // release the lock before waiting. Per the SessionIntf contract,
  // concurrent stop_pipeline / unload_pipeline from another thread
  // is the caller's responsibility to serialize: those would
  // either destroy the impl behind our back (unload) or reset its
  // runtime (stop), and we hold raw pointers here.
  vector<PipelineHandleImpl*> snap;
  {
    lock_guard<mutex> lk(_pipelines_mu);
    snap.reserve(_pipelines.size());
    for (const auto& kv : _pipelines) {
      if (kv.first->launched()) {
        snap.push_back(kv.first);
      }
    }
  }
  if (snap.empty()) {
    return Status{0};
  }

  // Negative timeout: wait forever, one pipeline at a time.
  if (timeout_ms < 0) {
    for (auto* impl : snap) {
      impl->wait_idle(-1);
    }
    return Status{0};
  }

  // Bounded timeout: split the budget across the remaining
  // pipelines, taking from the elapsed clock. Each pipeline gets
  // the full remaining budget; the loop short-circuits on the
  // first timeout. This is simpler than apportioning evenly and
  // matches the contract ("wait up to timeout_ms total").
  using clock = chrono::steady_clock;
  const auto deadline =
      clock::now() + chrono::milliseconds(timeout_ms);
  for (auto* impl : snap) {
    const auto now = clock::now();
    int remaining;
    if (now >= deadline) {
      remaining = 0;
    } else {
      remaining = static_cast<int>(
          chrono::duration_cast<chrono::milliseconds>(
              deadline - now).count());
    }
    if (!impl->wait_idle(remaining)) {
      return Status{4};
    }
  }
  return Status{0};
}

Status
Session::store_pipeline(PipelineHandle h, string_view path_sv)
{
  PipelineHandleImpl* impl = resolve(h);
  if (!impl) {
    warn(fmt("store_pipeline: unknown handle"));
    return Status{1};
  }
  string path(path_sv);
  if (path.empty()) {
    warn(fmt(
      "store_pipeline: empty path for pipeline '{}'",
      impl->pipeline()->id()));
    return Status{1};
  }
  impl->storage_path(std::move(path));
  return store_pipeline(h);
}

// error/warn/info are the user-facing channel and route to the UI
// delegate; the lower log_* levels remain diagnostic logging on the
// log delegate. error() still reports first, then throws.
void
Session::error(const VpipeFormat& f) const
{
  _ui_delegate->error(f);
  throw runtime_error(f() + "\nError reported\n");
}

void
Session::warn(const VpipeFormat& f) const
{
  _ui_delegate->warn(f);
}

void
Session::info(const VpipeFormat& f) const
{
  _ui_delegate->info(f);
}

void
Session::log_debug(const VpipeFormat& f) const
{
  _delegate->log(LogLevel::Debug, f);
}

void
Session::log_verbose(const VpipeFormat& f) const
{
  _delegate->log(LogLevel::Verbose, f);
}

void
Session::log_normal(const VpipeFormat& f) const
{
  _delegate->log(LogLevel::Normal, f);
}

void
Session::log_always(const VpipeFormat& f) const
{
  _delegate->log(LogLevel::Always, f);
}

UiInputStatus
Session::getline(const VpipeFormat&           prompt,
                 string&                      out,
                 const function<bool()>&      should_cancel) const
{
  return _ui_delegate->getline(prompt, out, should_cancel);
}

UiInputStatus
Session::getpasswd(const VpipeFormat&           prompt,
                   string&                      out,
                   const function<bool()>&      should_cancel) const
{
  return _ui_delegate->getpasswd(prompt, out, should_cancel);
}

UiInputStatus
Session::getmedialine(const VpipeFormat&           prompt,
                      string&                      out,
                      const function<bool()>&      should_cancel) const
{
  return _ui_delegate->getmedialine(prompt, out, should_cancel);
}

unique_ptr<UiTextStream>
Session::open_text_stream() const
{
  return _ui_delegate->open_text_stream();
}

void
Session::set_ui_delegate(unique_ptr<UiDelegateIntf> d)
{
  if (d) {
    _ui_delegate = std::move(d);
  }
}

void
Session::set_log_delegate(unique_ptr<LogDelegateIntf> d)
{
  if (!d) {
    return;
  }
  // Carry the current threshold across so the configured log.level (or
  // any debug_level already applied) survives the swap.
  d->set_threshold(_delegate->threshold());
  // Reassigning _delegate destroys the old one; a StdoutLogDelegate's
  // dtor detaches its async consumer (it schedules a wake on the pool,
  // which is up by now -- same as log_to_stdout()).
  _delegate = std::move(d);
  // If the new delegate is a StdoutLogDelegate, switch it to async;
  // a no-op for the web-ui's in-memory ring delegate.
  attach_if_stdout_(_delegate.get(), this);
}

std::string
Session::language() const
{
  lock_guard<mutex> lk(_lang_mu);
  return _language;
}

Status
Session::set_language(string_view tag)
{
  string norm = normalize_language(tag);
  if (norm.empty()) {
    warn(fmt("set_language: unsupported language tag '{}'", string(tag)));
    return Status{2};
  }
  {
    lock_guard<mutex> lk(_lang_mu);
    if (_language == norm) {
      return Status{0};
    }
    _language = norm;
  }
  info(fmt("session UI language set to '{}'", norm));
  return Status{0};
}

bool
Session::any_pipeline_launched() const
{
  lock_guard<mutex> lk(_pipelines_mu);
  for (const auto& kv : _pipelines) {
    if (kv.first->launched()) {
      return true;
    }
  }
  return false;
}

Status
Session::debug_level(unsigned u)
{
  if (any_pipeline_launched()) {
    warn(fmt(
      "debug_level: rejected -- a pipeline is currently launched"));
    return Status{3};
  }
  _delegate->set_threshold(clamp_log_level(u));
  return Status{0};
}

Status
Session::debug_level(string_view name)
{
  if (any_pipeline_launched()) {
    warn(fmt(
      "debug_level: rejected -- a pipeline is currently launched"));
    return Status{3};
  }
  // parse_log_level returns the supplied default for unknown input;
  // use a sentinel-default to detect "did not parse".
  const LogLevel sentinel = LogLevel::Always;  // not a valid threshold
  const LogLevel parsed = parse_log_level(name, sentinel);
  if (parsed == sentinel && name != "always") {
    warn(fmt("debug_level: unknown level '{}'; ignored", name));
    return Status{1};
  }
  _delegate->set_threshold(parsed);
  return Status{0};
}

Status
Session::log_to_stdout()
{
  if (any_pipeline_launched()) {
    warn(fmt(
      "log_to_stdout: rejected -- a pipeline is currently launched"));
    return Status{3};
  }
  // Carry the current threshold across so callers don't have to
  // re-issue debug_level after the swap.
  const LogLevel level = _delegate->threshold();
  _delegate = make_unique<StdoutLogDelegate>(level);
  // The previous delegate (replaced above) had its dtor run as the
  // unique_ptr was reassigned -- if it was a StdoutLogDelegate,
  // detach happened there. Now switch the new delegate to async.
  attach_if_stdout_(_delegate.get(), this);
  return Status{0};
}

Status
Session::log_to_db()
{
  if (any_pipeline_launched()) {
    warn(fmt(
      "log_to_db: rejected -- a pipeline is currently launched"));
    return Status{3};
  }
  LmdbEnv* env = lmdb_env();
  if (!env) {
    warn(fmt(
      "log_to_db: lmdb_env open failed; ignored"));
    return Status{1};
  }
  const LogLevel level = _delegate->threshold();
  unique_ptr<LogDelegateIntf> fresh;
  try {
    fresh = make_unique<DbLogDelegate>(env, level);
  } catch (const exception& e) {
    warn(fmt(
      "log_to_db: DbLogDelegate construction failed: {}; keeping "
      "previous delegate", e.what()));
    return Status{2};
  }
  _delegate = std::move(fresh);
  return Status{0};
}

// ----------------------------------------------------------------------
// Performance profiling.
// ----------------------------------------------------------------------

namespace {

// Walk every Stage in `pl`, then recurse into each sub-pipeline.
template <class F>
void
for_each_stage_(Pipeline& pl, const F& visit)
{
  for (auto it = pl.begin(); it != pl.end(); ++it) {
    if (auto* s = dynamic_cast<Stage*>(*it)) {
      visit(s, &pl);
    }
  }
  for (const auto& kv : pl.graphs()) {
    if (auto* child = dynamic_cast<Pipeline*>(kv.second.get())) {
      for_each_stage_(*child, visit);
    }
  }
}

}

void
Session::record_perf_event(uint32_t stage_gvid,
                           uint32_t type,
                           uint64_t value) const noexcept
{
  // The hot path. Caller (Stage::record_perf_event) already gated
  // on profiling_enabled(); we don't re-check here. Routing:
  // worker buffer index = worker_id; non-worker callers fall into
  // the overflow buffer at index num_workers().
  if (_perf_buffers.empty()) {
    return;
  }
  unsigned wid = ThreadPool::not_a_worker;
  if (_pool) {
    wid = _pool->worker_id_of_current_thread();
  }
  // Workers -> their own buffer; any non-worker -> the overflow buffer
  // (NOT the last buffer, which is now an auxiliary lane).
  size_t idx = (wid < _perf_overflow_index) ? wid : _perf_overflow_index;
  _perf_buffers[idx]->record(stage_gvid, type, value);
}

void
Session::record_perf_event_aux(unsigned lane,
                               uint32_t gvid,
                               uint32_t type,
                               uint64_t value) const noexcept
{
  // Auxiliary lanes route by `lane`, not by the calling thread, so the
  // LLM/MLX worker and CoreML callback threads land in their dedicated
  // timelines instead of the shared overflow lane.
  if (_perf_buffers.empty() || lane >= kPerfAuxLaneCount) {
    return;
  }
  size_t idx = _perf_aux_base + lane;
  if (idx < _perf_buffers.size()) {
    _perf_buffers[idx]->record(gvid, type, value);
  }
}

Status
Session::enable_profiling(unsigned max_events_per_thread)
{
  if (max_events_per_thread == 0) {
    warn(fmt(
      "enable_profiling: max_events_per_thread must be > 0"));
    return Status{1};
  }
  // Capture both clocks at the same moment so the dump can map
  // steady_clock-based event ns deltas to wall-clock time.
  _profiling_anchor          = chrono::steady_clock::now();
  _profiling_realtime_anchor = chrono::system_clock::now();
  _profiling_max             = max_events_per_thread;

  // Allocate (num_workers + 1 + aux) buffers: N for workers, 1
  // overflow, then one per auxiliary lane (LLM, ANE).
  unsigned n_workers = _pool ? _pool->num_workers() : 0;
  _perf_overflow_index = n_workers;
  _perf_aux_base       = static_cast<size_t>(n_workers) + 1;
  const size_t total   = _perf_aux_base + kPerfAuxLaneCount;
  _perf_buffers.clear();
  _perf_buffers.reserve(total);
  for (size_t i = 0; i < total; ++i) {
    _perf_buffers.push_back(make_unique<PerfBuffer>(
        max_events_per_thread, _profiling_anchor));
  }

  // Publish the master switch last so producers don't see "on" with
  // empty/incomplete buffers.
  _profiling_enabled.store(true, memory_order_release);
  return Status{0};
}

Status
Session::disable_profiling()
{
  // Set the flag off first so producers stop calling record_perf_
  // event; we then drop the buffers. Note: a producer that already
  // entered record_perf_event before the flag flip may still be
  // mid-write; freeing the buffers underneath is unsafe in that
  // window. So we wait for any active producers to finish by
  // pausing/stopping pipelines is the user's responsibility -- the
  // contract is that disable_profiling is called when no recording
  // is in flight (mirrors the log-delegate-swap contract).
  _profiling_enabled.store(false, memory_order_release);
  _perf_buffers.clear();
  _perf_overflow_index = 0;
  _perf_aux_base       = 0;
  _profiling_max = 0;
  return Status{0};
}

namespace {

// Build a `gvid -> Stage*` map by walking every stage in every
// session-owned pipeline. Used at dump time to translate event
// stage_gvid back to a live Stage for perf_event_name().
struct StageRef {
  Stage*    stage;
  Pipeline* pipeline;
};

unordered_map<uint32_t, StageRef>
build_gvid_to_stage_(
    const unordered_map<PipelineHandleImpl*,
                        unique_ptr<PipelineHandleImpl>>& pipelines)
{
  unordered_map<uint32_t, StageRef> out;
  for (const auto& kv : pipelines) {
    Pipeline* pl = kv.first->pipeline();
    if (!pl) {
      continue;
    }
    for_each_stage_(*pl, [&](Stage* s, Pipeline* parent) {
      out[VertexGraphAccess::gvid(s)] = StageRef{s, parent};
    });
  }
  return out;
}

}

Status
Session::dump_profiling(string_view path_sv)
{
  string path(path_sv);
  if (path.empty()) {
    warn(fmt("dump_profiling: empty path"));
    return Status{1};
  }

  FlexData root = FlexData::make_object();
  root.as_object().insert_or_assign(
      "anchor_steady_ns",
      FlexData::make_uint(static_cast<uint64_t>(
        chrono::duration_cast<chrono::nanoseconds>(
          _profiling_anchor.time_since_epoch()).count())));
  root.as_object().insert_or_assign(
      "anchor_realtime_ns",
      FlexData::make_uint(static_cast<uint64_t>(
        chrono::duration_cast<chrono::nanoseconds>(
          _profiling_realtime_anchor.time_since_epoch()).count())));
  root.as_object().insert_or_assign(
      "max_events_per_thread",
      FlexData::make_uint(_profiling_max));
  root.as_object().insert_or_assign(
      "enabled",
      FlexData::make_bool(
        _profiling_enabled.load(memory_order_relaxed)));
  root.as_object().insert_or_assign(
      "num_workers",
      FlexData::make_uint(_pool ? _pool->num_workers() : 0u));

  // Phase 1: per-thread raw event arrays. Each buffer index in
  // _perf_buffers becomes one entry: worker buffers (0 ..
  // num_workers()-1) and the overflow buffer (num_workers()).
  // This is the canonical layout -- it mirrors the in-memory
  // storage and includes every event regardless of stage.
  FlexData threads = FlexData::make_array();
  {
    lock_guard<mutex> lk(_pipelines_mu);
    for (size_t bi = 0; bi < _perf_buffers.size(); ++bi) {
      const PerfBuffer& pb = *_perf_buffers[bi];
      size_t n = pb.size();

      FlexData ns_arr    = FlexData::make_array();
      FlexData type_arr  = FlexData::make_array();
      FlexData gvid_arr  = FlexData::make_array();
      FlexData value_arr = FlexData::make_array();
      for (size_t i = 0; i < n; ++i) {
        const PerfEvent& e = pb.at(i);
        ns_arr.as_array().push_back(FlexData::make_uint(e.ns));
        type_arr.as_array().push_back(FlexData::make_uint(e.type));
        gvid_arr.as_array().push_back(
            FlexData::make_uint(e.stage_gvid));
        value_arr.as_array().push_back(FlexData::make_uint(e.value));
      }

      // Lane kind from the buffer index: worker (bi < overflow),
      // overflow (bi == overflow), or an auxiliary lane (bi >= aux
      // base) which carries a human label (LLM / ANE).
      const bool is_overflow = (bi == _perf_overflow_index);
      const bool is_aux      = (bi >= _perf_aux_base);
      FlexData entry = FlexData::make_object();
      entry.as_object().insert_or_assign(
          "worker_id",
          FlexData::make_uint(
            (is_overflow || is_aux)
              ? static_cast<uint64_t>(ThreadPool::not_a_worker)
              : static_cast<uint64_t>(bi)));
      entry.as_object().insert_or_assign(
          "is_overflow", FlexData::make_bool(is_overflow));
      if (is_aux) {
        const size_t lane = bi - _perf_aux_base;
        if (lane < kPerfAuxLaneCount) {
          entry.as_object().insert_or_assign(
              "label", FlexData::make_string(kPerfAuxLaneName[lane]));
        }
      }
      FlexData ev = FlexData::make_object();
      ev.as_object().insert_or_assign("ns",         std::move(ns_arr));
      ev.as_object().insert_or_assign("type",       std::move(type_arr));
      ev.as_object().insert_or_assign("stage_gvid", std::move(gvid_arr));
      ev.as_object().insert_or_assign("value",      std::move(value_arr));
      entry.as_object().insert_or_assign("events", std::move(ev));
      entry.as_object().insert_or_assign(
          "events_count", FlexData::make_uint(n));
      entry.as_object().insert_or_assign(
          "dropped",      FlexData::make_uint(pb.dropped()));
      entry.as_object().insert_or_assign(
          "capacity",     FlexData::make_uint(pb.capacity()));
      threads.as_array().push_back(std::move(entry));
    }
  }
  root.as_object().insert_or_assign("threads", std::move(threads));

  // Phase 2: per-stage event-name tables. The threads array holds
  // raw events grouped by worker; consumers that want a per-stage
  // view re-bucket by stage_gvid and join with this table to map
  // type -> human-readable name.
  FlexData stages = FlexData::make_array();
  {
    lock_guard<mutex> lk(_pipelines_mu);
    auto gvid_to_stage = build_gvid_to_stage_(_pipelines);
    // Collect per-gvid distinct types from all buffers.
    unordered_map<uint32_t, unordered_set<uint32_t>> types_by_stage;
    for (const auto& pb_ptr : _perf_buffers) {
      const PerfBuffer& pb = *pb_ptr;
      size_t n = pb.size();
      for (size_t i = 0; i < n; ++i) {
        const PerfEvent& e = pb.at(i);
        types_by_stage[e.stage_gvid].insert(e.type);
      }
    }
    for (const auto& kv : gvid_to_stage) {
      uint32_t gvid    = kv.first;
      const StageRef& sr = kv.second;

      FlexData entry = FlexData::make_object();
      entry.as_object().insert_or_assign(
          "gvid", FlexData::make_uint(gvid));
      entry.as_object().insert_or_assign(
          "id",   FlexData::make_string(sr.stage->id()));
      entry.as_object().insert_or_assign(
          "type", FlexData::make_string(sr.stage->type_name()));
      if (sr.pipeline) {
        entry.as_object().insert_or_assign(
            "pipeline", FlexData::make_string(sr.pipeline->id()));
      }
      FlexData names = FlexData::make_object();
      auto it = types_by_stage.find(gvid);
      if (it != types_by_stage.end()) {
        for (uint32_t t : it->second) {
          names.as_object().insert_or_assign(
              std::to_string(t),
              FlexData::make_string(sr.stage->perf_event_name(t)));
        }
      }
      entry.as_object().insert_or_assign(
          "event_names", std::move(names));
      stages.as_array().push_back(std::move(entry));
    }

    // Synthetic stage entries naming the LLM auxiliary-lane activities
    // (text-prefill / text-decode / vision-tower / audio-encoder), so
    // the viewer colors + labels them like real stages. Emitted only
    // when events for that activity were actually recorded. The ANE
    // lane reuses real CoreML stage gvids, so it needs none here.
    for (const auto& d : kPerfAuxStages) {
      if (types_by_stage.find(d.gvid) == types_by_stage.end()) {
        continue;
      }
      FlexData entry = FlexData::make_object();
      entry.as_object().insert_or_assign(
          "gvid", FlexData::make_uint(d.gvid));
      entry.as_object().insert_or_assign(
          "id",   FlexData::make_string(d.id));
      entry.as_object().insert_or_assign(
          "type", FlexData::make_string("llm"));
      FlexData names = FlexData::make_object();
      names.as_object().insert_or_assign(
          std::to_string(d.begin_type), FlexData::make_string("begin"));
      names.as_object().insert_or_assign(
          std::to_string(d.begin_type + 1u),
          FlexData::make_string("end"));
      entry.as_object().insert_or_assign(
          "event_names", std::move(names));
      stages.as_array().push_back(std::move(entry));
    }
  }
  root.as_object().insert_or_assign("stages", std::move(stages));

  string ext = ext_of_(path);
  string serialized;
  try {
    if (ext == ".json") {
      serialized = root.to_json(/* pretty */ true);
    } else {
      serialized = root.to_binary();
    }
  } catch (const exception& e) {
    warn(fmt(
      "dump_profiling: serialization failed for '{}': {}",
      path, e.what()));
    return Status{1};
  }

  ofstream out(path, ios::binary);
  if (!out) {
    warn(fmt("dump_profiling: failed to open '{}' for writing", path));
    return Status{1};
  }
  out.write(serialized.data(),
            static_cast<streamsize>(serialized.size()));
  if (!out) {
    warn(fmt("dump_profiling: write failed for '{}'", path));
    return Status{1};
  }
  return Status{0};
}

}
