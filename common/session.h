#ifndef SESSION_H
#define SESSION_H

#include "common/flex-data.h"
#include "common/path-sandbox.h"
#include "interfaces/log-delegate-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "interfaces/session-context-intf.h"
#include "common/perf-buffer.h"
#include "vpipe/session-intf.h"
#include <atomic>
#include <functional>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vpipe {

class PipelineHandleImpl;
class ThreadPool;

namespace genai { class GenerativeModelManager; }
namespace metal_compute { class MetalCompute; }

class Session final : public SessionIntf,
                      public SessionContextIntf
{
public:
  // Config string is one of:
  //   * empty / all-whitespace          -> built-in defaults
  //   * starts with '{' or '['          -> inline JSON
  //   * otherwise                       -> filesystem path to a JSON
  //                                        or binary-FlexData config
  Session(std::string_view = "");

  // Test / library-user injection. Bypasses config-driven delegate
  // selection. A null pointer falls back to a default StdoutLogDelegate.
  // Worker count and edge-buffer capacity use built-in defaults.
  explicit Session(std::unique_ptr<LogDelegateIntf>);

  // Defined in .cc so unique_ptr<PipelineHandleImpl> / unique_ptr<ThreadPool>
  // can be destroyed at a point where the impl types are complete.
  ~Session();

  /* external */
  PipelineHandle load_pipeline(std::string_view) override;
  PipelineHandle create_pipeline(std::string) override;
  Status launch_pipeline(PipelineHandle) override;
  Status pause_pipeline(PipelineHandle) override;
  Status stop_pipeline(PipelineHandle) override;
  Status unload_pipeline(PipelineHandle) override;
  Status store_pipeline(PipelineHandle) override;
  Status store_pipeline(PipelineHandle,
                        std::string_view path) override;
  Status wait_pipelines(int timeout_ms = -1) override;

  Status debug_level(unsigned) override;
  Status debug_level(std::string_view) override;
  Status log_to_stdout() override;
  Status log_to_db() override;
  Status set_language(std::string_view) override;

  Status enable_profiling(unsigned max_events_per_stage) override;
  Status disable_profiling() override;
  Status dump_profiling(std::string_view path) override;

  /* internal */
  void error(const VpipeFormat&) const override;
  void warn(const VpipeFormat&) const override;
  void info(const VpipeFormat&) const override;
  void log_debug(const VpipeFormat&) const override;
  void log_verbose(const VpipeFormat&) const override;
  void log_normal(const VpipeFormat&) const override;
  void log_always(const VpipeFormat&) const override;

  UiInputStatus
  getline(const VpipeFormat& prompt, std::string& out,
          const std::function<bool()>& should_cancel) const override;

  UiInputStatus
  getpasswd(const VpipeFormat& prompt, std::string& out,
            const std::function<bool()>& should_cancel) const override;

  UiInputStatus
  getmedialine(const VpipeFormat& prompt, std::string& out,
               const std::function<bool()>& should_cancel) const override;

  std::unique_ptr<UiTextStream> open_text_stream() const override;

  std::string language() const override;

  // Install the user-facing I/O delegate (error/warn/info + getline).
  // Defaults to a StdioUiDelegate at construction. The web-ui app
  // swaps in a delegate that diverts these channels to the browser.
  // A null pointer is ignored. Like the log-delegate swaps, this is
  // not safe to race against active worker-thread log/UI calls; the
  // caller installs it before launching pipelines.
  void set_ui_delegate(std::unique_ptr<UiDelegateIntf>);

  // Install the diagnostic log delegate (the log_* channels). Defaults
  // to a StdoutLogDelegate; the web-ui app swaps in one that diverts
  // the leveled log stream to the browser's Session Log view. The
  // current threshold is carried across so the configured log.level
  // survives the swap. A null pointer is ignored. Like the log_to_*
  // swaps, this must not race against worker-thread log() calls; the
  // caller installs it before launching pipelines.
  void set_log_delegate(std::unique_ptr<LogDelegateIntf>);

  // Record the address the web-ui's HTTP server bound to, so stages
  // that host their own HTTP endpoint (hls-broadcast) can default to
  // the same interface. Called once by the web-ui app at startup,
  // before pipelines launch; empty in every other front end. See
  // SessionContextIntf::web_ui_bind_address.
  void set_web_ui_bind_address(std::string addr)
  {
    _web_ui_bind_address = std::move(addr);
  }
  std::string web_ui_bind_address() const override
  {
    return _web_ui_bind_address;
  }

  // Session-level resources surfaced through SessionContextIntf so
  // every SessionMember can reach them via session()->...().
  ThreadPool* thread_pool() const noexcept override
  {
    return _pool.get();
  }
  unsigned default_edge_capacity() const noexcept override
  {
    return _default_edge_capacity;
  }

  // Live profiling state. record_perf_event is the producer hot
  // path; stages reach it via their inline Stage::record_perf_event
  // wrapper which checks profiling_enabled() first.
  bool profiling_enabled() const noexcept override
  {
    return _profiling_enabled.load(std::memory_order_relaxed);
  }
  unsigned profiling_max_events_per_thread() const noexcept override
  {
    return _profiling_max;
  }
  std::chrono::steady_clock::time_point
  profiling_anchor() const noexcept override
  {
    return _profiling_anchor;
  }
  void record_perf_event(std::uint32_t stage_gvid,
                         std::uint32_t type,
                         std::uint64_t value) const noexcept override;
  void record_perf_event_aux(unsigned      lane,
                             std::uint32_t gvid,
                             std::uint32_t type,
                             std::uint64_t value) const noexcept override;

  // Lazily constructs (on first call) the per-session FFmpegLibraries
  // instance under a once_flag and returns it. See SessionContextIntf
  // for the contract.
  const FFmpegLibraries* ffmpeg_libraries() const override;

  // Lazily opens (on first call) the per-session LmdbEnv at
  // db.path / db.map_size_mb (configured at session boot). When
  // db.path is missing or empty the env opens at "." -- i.e. the
  // process working directory at first-open time. Returns nullptr
  // only if the env failed to open (the failure is reported
  // through the active log delegate). Concurrent first calls
  // serialise on an internal once_flag.
  LmdbEnv* lmdb_env() const override;

  // Confine a stage-supplied local file path to the session filesystem
  // sandbox (see SessionContextIntf::confine_path). Passthrough when the
  // sandbox is disabled.
  std::filesystem::path
  confine_path(std::string_view user_path, bool for_write,
               std::string* err = nullptr) const override;

  // Filesystem-sandbox state (see SessionContextIntf). Reflect the
  // session's PathSandbox so the web-ui file browser can present the
  // sandbox's chroot-like namespace.
  bool fs_sandboxed() const override { return _path_sandbox.enabled(); }
  std::filesystem::path fs_sandbox_root() const override
  {
    return _path_sandbox.root();
  }
  std::vector<std::filesystem::path> fs_whitelist() const override
  {
    return _path_sandbox.whitelist();
  }

  // Session-shared CoreML model cache. Lazily constructed on first
  // call. Returns nullptr on non-Apple builds. See SessionContextIntf
  // for the contract.
  CoreMLModelManager* coreml_model_manager() const override;

  // Session-shared language-model manager. Lazily constructed on
  // first call. Returns nullptr when the metal-compute LLM subsystem
  // is unavailable. See SessionContextIntf for the contract.
  genai::GenerativeModelManager* generative_model_manager() const override;

  // Session-shared Metal compute framework. Lazily constructed on
  // first call. Returns nullptr on non-Apple builds. See
  // SessionContextIntf for the contract.
  metal_compute::MetalCompute* metal_compute() const override;

  // Parsed config tree. Empty object when no config was supplied.
  const FlexData& config() const noexcept { return _config; }

  // Non-const accessor kept as a convenience for code that already
  // holds a Session* (vs. a SessionContextIntf*).
  ThreadPool* pool() noexcept { return _pool.get(); }

private:
  // Resolve a PipelineHandle to its owning Impl, or nullptr if the
  // handle is unknown to this Session.
  PipelineHandleImpl* resolve(PipelineHandle) const;

  // True iff any pipeline owned by this Session is currently launched
  // (running). Used as a precondition gate for the runtime log-config
  // mutators (debug_level / log_to_*), which must not race against
  // worker-thread log() calls.
  bool any_pipeline_launched() const;

  FlexData                         _config;

  // Hot-swappable at runtime via log_to_stdout() / log_to_db(), but
  // only when no pipeline is currently launched -- that contract is
  // checked in those methods. With it, a swap can never race against
  // a worker-thread log() call so no extra synchronization is needed.
  std::unique_ptr<LogDelegateIntf> _delegate;

  // User-facing I/O delegate (error/warn/info + getline). Defaults to
  // a StdioUiDelegate; swappable via set_ui_delegate(). Declared right
  // after _delegate so error/warn/info routing and getline are always
  // backed by a live object during construction (the ctor body warns
  // through it on a config parse failure). Never null after
  // construction -- set_ui_delegate ignores a null argument.
  std::unique_ptr<UiDelegateIntf>  _ui_delegate;

  // Boot-time db config retained so log_to_db() and the lazy
  // lmdb_env() opener don't have to re-parse the config tree. An
  // empty `_db_path` means "no path configured" -- lmdb_env()
  // falls back to the process CWD (".") in that case.
  std::string                      _db_path;
  std::size_t                      _db_map_size = 0;

  // Filesystem sandbox for stage file paths (web-ui default; disabled
  // for the CLI and under --expose-native-file-system). Parsed once from
  // the "file_sandbox" config key at construction. Model-manager file
  // access is intentionally not routed through it.
  PathSandbox                      _path_sandbox;

  // Set once by the web-ui app (set_web_ui_bind_address) before any
  // pipeline launches; empty otherwise. Read by HTTP-hosting stages
  // via SessionContextIntf::web_ui_bind_address to pick a default
  // bind interface. No synchronization: written before launch, only
  // read afterward.
  std::string                      _web_ui_bind_address;

  std::unique_ptr<ThreadPool>      _pool;
  unsigned                         _default_edge_capacity = 4;

  // UI/message locale (normalized IETF tag). Seeded from the "language"
  // config key (default en-us) and mutable at runtime via set_language;
  // guarded by _lang_mu since language() can be read off worker threads
  // while a UI thread changes it.
  mutable std::mutex               _lang_mu;
  std::string                      _language = "en-us";

  // Profiling control. _profiling_enabled is the master switch
  // queried by Stage::record_perf_event. _profiling_max is the
  // per-thread capacity passed to enable_profiling. _profiling_
  // anchor anchors every event's ns delta. _profiling_realtime_
  // anchor lets the dump correlate the steady_clock anchor to
  // wall-clock time. _perf_buffers holds (num_workers + 1) buffers
  // -- one per worker plus one shared overflow at index num_workers
  // for non-worker callers; mutable because record_perf_event is
  // const-callable (it's a producer hot path with no logical state
  // change visible to readers).
  std::atomic<bool>                _profiling_enabled{false};
  unsigned                         _profiling_max = 0;
  std::chrono::steady_clock::time_point _profiling_anchor;
  std::chrono::system_clock::time_point _profiling_realtime_anchor;
  // Buffer layout: [worker 0 .. worker N-1][overflow][aux 0 .. aux K-1]
  // where K == kPerfAuxLaneCount. _perf_overflow_index == N routes
  // non-worker thread-keyed events; _perf_aux_base == N+1 is the first
  // auxiliary-lane buffer (record_perf_event_aux indexes by lane off
  // this base). Both are 0 when profiling is disabled.
  mutable std::vector<std::unique_ptr<PerfBuffer>> _perf_buffers;
  std::size_t                                      _perf_overflow_index = 0;
  std::size_t                                      _perf_aux_base = 0;

  // Session-shared FFmpegLibraries. Lazily constructed on first call
  // to ffmpeg_libraries(); guarded by _ffmpeg_once. mutable because
  // ffmpeg_libraries() is logically const-observable but needs to
  // perform the one-shot init.
  mutable std::once_flag                   _ffmpeg_once;
  mutable std::unique_ptr<FFmpegLibraries> _ffmpeg;

  // Session-shared LmdbEnv. Lazily opened on first lmdb_env() call.
  // mutable because lmdb_env() is logically const-observable.
  mutable std::once_flag                   _env_once;
  mutable std::unique_ptr<LmdbEnv>         _env;

  // Session-shared CoreML model cache. Lazily constructed on first
  // coreml_model_manager() call. On non-Apple builds the manager is
  // never constructed and the accessor returns nullptr. mutable so
  // the const accessor can lazy-init.
  mutable std::once_flag                     _coreml_mgr_once;
  mutable std::unique_ptr<CoreMLModelManager> _coreml_mgr;

  // Session-shared LLM manager. Lazily constructed on first
  // generative_model_manager() call. On non-apple-silicon builds the
  // manager is never constructed and the accessor returns nullptr.
  // mutable so the const accessor can lazy-init.
  mutable std::once_flag                          _llm_mgr_once;
  mutable std::unique_ptr<genai::GenerativeModelManager> _llm_mgr;

  // Session-shared Metal compute framework. Lazily constructed on
  // first metal_compute() call. On non-Apple builds it is never
  // constructed and the accessor returns nullptr. mutable so the
  // const accessor can lazy-init.
  mutable std::once_flag                                  _mc_once;
  mutable std::unique_ptr<metal_compute::MetalCompute>    _mc;

  // Owns every PipelineHandleImpl we hand out. Keyed on the impl's
  // own pointer (which is the value carried inside PipelineHandle).
  mutable std::mutex                                              _pipelines_mu;
  std::unordered_map<PipelineHandleImpl*,
                     std::unique_ptr<PipelineHandleImpl>>         _pipelines;
};

}

#endif
