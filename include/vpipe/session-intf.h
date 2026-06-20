// vpipe::SessionIntf -- the abstract surface a vpipe library user
// drives. Concrete sessions are constructed through SessionManager
// (vpipe/session-manager.h) and outlive every PipelineHandle /
// StageHandle they hand out.
//
// Threading model:
//   * `load_pipeline`, `create_pipeline`, and the lifecycle
//     methods (`launch_pipeline`, `pause_pipeline`,
//     `stop_pipeline`, `unload_pipeline`, `store_pipeline`) are
//     serialized by an internal mutex; they are safe to call from
//     any thread.
//   * `debug_level`, `log_to_stdout`, `log_to_db` are *not*
//     guaranteed safe to interleave with active log writes from
//     worker threads; they are rejected with Status{3} while any
//     pipeline owned by the session is currently launched.
//
// Lifetime:
//   Every PipelineHandle returned from a SessionIntf is opaque and
//   borrowed -- the impl is owned by the session. When the session
//   is destroyed every handle it produced becomes invalid; using a
//   stale handle is a Status{1} (bad request), not undefined
//   behaviour.

#ifndef SESSION_INTF_H
#define SESSION_INTF_H

#include "vpipe/pipeline-handle.h"
#include "vpipe/status.h"
#include <string>
#include <string_view>

namespace vpipe {

class SessionIntf {
public:
  SessionIntf() {};
  virtual ~SessionIntf() = default;

  // Load a pipeline from a filesystem path. The file format is
  // auto-detected: a leading `{` or `[` (after whitespace) is
  // parsed as JSON, anything else as the FlexData binary
  // encoding. The returned handle is null (use !valid()) if the
  // file cannot be read, the spec fails to parse, or any stage
  // fails to instantiate; failures are reported through the
  // session log delegate. The path is remembered on the handle's
  // impl so a subsequent no-arg `store_pipeline(handle)` call can
  // round-trip back to the same place.
  virtual PipelineHandle load_pipeline(std::string_view path) = 0;

  // Create a new empty pipeline with the given id. The returned
  // handle is owned by this session and remains valid until the
  // session is destroyed or `unload_pipeline` is called on it.
  virtual PipelineHandle create_pipeline(std::string) = 0;

  // Lifecycle -- all four take a handle previously returned by
  // load_/create_pipeline. An unknown handle gets Status{1}; a
  // runtime-level failure gets Status{2}.
  //
  //   launch_pipeline -- spin up the runtime: allocate edge buffers,
  //                      schedule one driver coroutine per stage on
  //                      the session ThreadPool. Returns once every
  //                      driver has been scheduled (not once they
  //                      have finished).
  //   pause_pipeline  -- ask the runtime to stop on the next
  //                      iteration boundary; non-blocking.
  //   stop_pipeline   -- pause + close every edge buffer + wait for
  //                      every driver to reach final_suspend, then
  //                      destroy the runtime.
  //   unload_pipeline -- stop (if running) + destroy the pipeline
  //                      graph + drop the impl.
  //   store_pipeline  -- persist the pipeline to its storage URL
  //                      (set by `load_pipeline` or by the
  //                      path-taking overload). The format is
  //                      chosen from the path extension:
  //                      `.json` -> pretty JSON; anything else ->
  //                      FlexData binary. Returns Status{1} if
  //                      no URL has been associated with this
  //                      handle yet.
  virtual Status launch_pipeline(PipelineHandle) = 0;
  virtual Status pause_pipeline(PipelineHandle) = 0;
  virtual Status stop_pipeline(PipelineHandle) = 0;
  virtual Status unload_pipeline(PipelineHandle) = 0;
  virtual Status store_pipeline(PipelineHandle) = 0;
  // Path-taking overload of store_pipeline: associates `path` with
  // the handle (so subsequent no-arg calls reuse it) and persists
  // immediately. Format is selected from the extension as above.
  virtual Status store_pipeline(PipelineHandle,
                                std::string_view path) = 0;

  // Block the calling thread until every pipeline owned by this
  // session that is currently launched has reached *data-driven*
  // idle -- i.e. every driver coroutine has returned. Pipelines
  // that have one-shot work (e.g. a single inference, a finite
  // transcode) signal their own completion via
  // RuntimeContext::signal_done() and the EOS cascades through
  // the graph; wait_pipelines blocks until that cascade finishes
  // for every launched pipeline. Compare to stop_pipeline(), which
  // *preempts* in-flight work by closing edge buffers and waking
  // suspended drivers -- it does not let one-shot stages finish.
  //
  // `timeout_ms`:
  //   negative   -- wait forever (the default).
  //   0          -- non-blocking probe.
  //   positive   -- wait up to this many milliseconds.
  //
  // Returns:
  //   Status{0}  -- every pipeline in the snapshot taken at call
  //                 time reached idle.
  //   Status{4}  -- timeout elapsed before all pipelines reached
  //                 idle. Pipelines still launched continue to
  //                 run; the caller may retry, stop_pipeline, or
  //                 unload_pipeline.
  //
  // Snapshot semantics: the wait operates on the set of launched
  // pipelines captured at entry. A pipeline launched while the
  // wait is in progress is NOT picked up -- call wait_pipelines
  // again to include it.
  //
  // Thread-safety note: this method is intentionally NOT safe to
  // call from one thread while another thread calls stop_pipeline
  // / unload_pipeline on the same session. The typical pattern is
  // a single control thread (e.g. Python's main thread) doing
  // launch -> wait -> unload sequentially. SIGINT handling is
  // expected to interrupt the wait via the binding layer
  // (Python's signal machinery; the binding releases the GIL with
  // periodic PyErr_CheckSignals), after which the caller's
  // exception path can sequentially issue stop_pipeline /
  // unload_pipeline from the same thread that was blocked here.
  virtual Status wait_pipelines(int timeout_ms = -1) = 0;

  // Runtime overrides on top of the boot-time `log.*` config block.
  // All four are rejected with Status{3} while any pipeline owned
  // by this session is currently launched -- log-delegate swaps
  // must not race against worker-thread log() calls.
  //
  // debug_level(unsigned): numeric form, mapped to LogLevel by
  //   integer value -- 0=Error, 1=Warn, 2=Info, 3=Normal,
  //   4=Verbose, 5=Debug. Out-of-range values are clamped to Debug.
  // debug_level(string_view): name form, parsed via parse_log_level
  //   ("error"/"warn"/"info"/"normal"/"verbose"/"debug"/"always");
  //   unknown names leave the threshold unchanged and warn.
  // log_to_stdout(): swap the active delegate to a fresh
  //   StdoutLogDelegate carrying the current threshold.
  // log_to_db(): swap the active delegate to a fresh DbLogDelegate
  //   that writes into the session-shared LmdbEnv (opened on
  //   demand from the boot config's top-level `db.path` /
  //   `db.map_size_mb`). When `db.path` is missing the env opens
  //   in the process CWD; only an env open failure makes this a
  //   no-op + warn that returns Status{1}.
  virtual Status debug_level(unsigned) = 0;
  virtual Status debug_level(std::string_view) = 0;
  virtual Status log_to_stdout() = 0;
  virtual Status log_to_db() = 0;

  // Set the UI/message locale (an IETF tag, e.g. "en-us", "zh-cn",
  // "zh-tw"). The tag is normalized; an unsupported tag is rejected with
  // Status{2} and the language is left unchanged. Safe to call at any
  // time (it only affects how user-facing strings are rendered). Read it
  // back via SessionContextIntf::language(). This is the application UI
  // locale, NOT a per-stage model/ASR language hint.
  virtual Status set_language(std::string_view tag) = 0;

  // ---- Performance profiling -----------------------------------
  //
  // A low-overhead per-stage event tracer. When enabled, each
  // Stage carries a fixed-size PerfBuffer of `max_events_per_stage`
  // slots. From inside a stage, code calls
  // Stage::record_perf_event(type, value) -- a single-branch no-op
  // when profiling is off, and a relaxed-atomic claim-then-write
  // when on. Events are dropped (newest-wins) once the buffer is
  // full so the producer cost stays constant.
  //
  // enable_profiling(N):
  //   * Captures a session-level steady_clock anchor.
  //   * Walks every stage in every pipeline (including sub-
  //     pipelines) and allocates a PerfBuffer of capacity N.
  //   * Sets a session flag so stages constructed AFTER this call
  //     also self-allocate at construction.
  //   * Calling enable_profiling again clears any previously
  //     captured events and re-allocates with the new size.
  //   Returns Status{1} if N == 0.
  //
  // disable_profiling():
  //   * Clears the flag and drops every stage's PerfBuffer. After
  //     disable, record_perf_event is a single-branch no-op. Dump
  //     anything you care about BEFORE disabling.
  //
  // dump_profiling(path):
  //   * Snapshots all per-stage buffers into a FlexData document
  //     and writes it to the given path. Format is selected from
  //     the extension: `.json` -> pretty JSON; anything else ->
  //     FlexData binary. Safe to call while profiling is still
  //     enabled (captures a consistent prefix of each buffer).
  //   Returns Status{1} on path / IO failure.
  virtual Status enable_profiling(unsigned max_events_per_stage) = 0;
  virtual Status disable_profiling() = 0;
  virtual Status dump_profiling(std::string_view path) = 0;

};

}

#endif
