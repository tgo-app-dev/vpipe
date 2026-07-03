#ifndef SESSION_CONTEXT_INTF_H
#define SESSION_CONTEXT_INTF_H

#include "interfaces/ui-delegate-intf.h"
#include "common/i18n.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace vpipe {

struct VpipeFormat;
class CoreMLModelManager;
class FFmpegLibraries;
class LmdbEnv;
class ThreadPool;

namespace genai { class GenerativeModelManager; }
namespace metal_compute { class MetalCompute; }

// Resources every SessionMember (Vertex, Stage, EdgeBuffer, ...)
// reaches for during normal operation: structured logging plus the
// session-level thread pool and default edge-buffer capacity. Held by
// pointer through SessionMember::session().
class SessionContextIntf {
public:
  SessionContextIntf() {};
  virtual ~SessionContextIntf() = default;

  virtual void error(const VpipeFormat&) const = 0;
  virtual void warn(const VpipeFormat&) const = 0;
  virtual void info(const VpipeFormat&) const = 0;
  virtual void log_debug(const VpipeFormat&) const = 0;
  virtual void log_verbose(const VpipeFormat&) const = 0;
  virtual void log_normal(const VpipeFormat&) const = 0;
  virtual void log_always(const VpipeFormat&) const = 0;

  // Current UI/message locale as an IETF tag (e.g. "en-us", "zh-cn").
  // Drives tr() and any user-facing string the session formats. The
  // default is English; the concrete Session returns the configured or
  // last-set tag. (Not the model/ASR language hint -- that is separate.)
  virtual std::string language() const
  {
    return std::string(default_language());
  }

  // Translate an application message `key` for the current language,
  // falling back to en-us and then to the key itself. Convenience over
  // the free vpipe::localize(); the key catalogue lives in common/i18n.
  virtual std::string tr(std::string_view key) const
  {
    return localize(language(), key);
  }

  // Blocking request for one line of interactive user input, routed to
  // the session's UI delegate. `prompt` is shown before the read;
  // `should_cancel` (if set) is polled while waiting so a pipeline
  // stop is observed promptly. See UiDelegateIntf::getline. The
  // default returns Eof (no interactive input available) so adapter
  // contexts that never service a UI need not override it.
  virtual UiInputStatus
  getline(const VpipeFormat& /*prompt*/, std::string& /*out*/,
          const std::function<bool()>& /*should_cancel*/) const
  {
    return UiInputStatus::Eof;
  }

  // Blocking request for one line of SECRET interactive input (a
  // password), routed to the session's UI delegate, which masks it on
  // screen. See UiDelegateIntf::getpasswd. The default returns Eof so
  // adapter contexts that never service a UI need not override it.
  virtual UiInputStatus
  getpasswd(const VpipeFormat& /*prompt*/, std::string& /*out*/,
            const std::function<bool()>& /*should_cancel*/) const
  {
    return UiInputStatus::Eof;
  }

  // Blocking request for one line of interactive user input that MAY
  // carry inline media-line attachment markers (images/audio; see
  // common/media-line.h and UiDelegateIntf::getmedialine), routed to
  // the session's UI delegate. Calling this instead of getline tells
  // the delegate the caller will parse markers, so media-capable
  // front ends (web-ui) offer attach/drop controls. The default
  // returns Eof so adapter contexts that never service a UI need not
  // override it.
  virtual UiInputStatus
  getmedialine(const VpipeFormat& /*prompt*/, std::string& /*out*/,
               const std::function<bool()>& /*should_cancel*/) const
  {
    return UiInputStatus::Eof;
  }

  // Open a live text-output stream routed to the session's UI delegate
  // (see UiDelegateIntf::open_text_stream / UiTextStream). The default
  // returns a no-op stream so adapter contexts that never service a UI
  // need not override it. Never returns null.
  virtual std::unique_ptr<UiTextStream>
  open_text_stream() const
  {
    return std::make_unique<NullUiTextStream>();
  }

  // The shared worker pool that drives stage drivers and wakes
  // suspended awaiters. Always non-null on a fully-constructed
  // Session. Members schedule resumed coroutines through it.
  virtual ThreadPool* thread_pool() const noexcept = 0;

  // Per-session knob. Currently used by EdgeBuffer when no explicit
  // capacity is supplied at construction.
  virtual unsigned default_edge_capacity() const noexcept = 0;

  // Lazily-materialized FFmpeg dlopen wrapper, shared across every
  // SessionMember in this session. The first caller pays the cost of
  // probing sonames and resolving the curated symbol table; every
  // subsequent caller gets the same instance back. Construction is
  // serialized internally; safe to call concurrently.
  //
  // The implementation may throw on first call if a Required-mode
  // load fails (e.g. no compatible FFmpeg on the system); subsequent
  // calls after a successful first construction never throw and
  // return the same non-null pointer. Callers that participate in
  // session-bootstrap contexts where FFmpeg is intentionally
  // unavailable (e.g. log delegates) may return nullptr instead.
  virtual const FFmpegLibraries* ffmpeg_libraries() const = 0;

  // Lazily-materialized session-shared LMDB environment. The path
  // and map size come from the session config's top-level `db.path`
  // and `db.map_size_mb`. When `db.path` is missing or empty the
  // env opens at "." -- i.e. the process CWD at first-open time.
  // Returns nullptr only if opening the env failed (the failure is
  // reported through the session's log delegate). Concrete sub-
  // databases (LmdbDb) live inside this single env -- both the log
  // delegate and any application stage that needs persistent KV
  // storage share it. Safe to call concurrently; the first call
  // serializes construction internally.
  virtual LmdbEnv* lmdb_env() const = 0;

  // ---- Session-level performance profiling ---------------------
  //
  // Sessions own a small fixed set of PerfBuffers -- one per
  // ThreadPool worker plus one shared "overflow" buffer used by
  // non-worker callers. Stages call record_perf_event(), which
  // routes to the right buffer based on the calling thread's
  // worker id. Memory bound = (num_workers + 1) *
  // max_events_per_thread * sizeof(PerfEvent); the user controls
  // total memory via max_events_per_thread.
  //
  // Defaults: profiling_enabled() returns false; the other two
  // accessors return zero / a default-constructed time_point;
  // record_perf_event is a no-op. Sub-contexts built outside a
  // real Session (e.g. DbLogDelegate's internal CerrSessionContext)
  // inherit these defaults.

  virtual bool
  profiling_enabled() const noexcept { return false; }

  virtual unsigned
  profiling_max_events_per_thread() const noexcept { return 0; }

  virtual std::chrono::steady_clock::time_point
  profiling_anchor() const noexcept
  {
    return std::chrono::steady_clock::time_point{};
  }

  // Producer hot path. Routes the event to the calling thread's
  // per-worker PerfBuffer (or the overflow buffer for non-worker
  // callers). Stage::record_perf_event is the inline wrapper that
  // forwards to this; user code reaches in here only via Stage.
  // Default impl is a no-op so adapter contexts (CerrSessionContext)
  // don't need to implement it.
  virtual void
  record_perf_event(std::uint32_t /*stage_gvid*/,
                    std::uint32_t /*type*/,
                    std::uint64_t /*value*/) const noexcept
  {
  }

  // Producer hot path for an AUXILIARY (non-worker) lane -- a logical
  // activity timeline (LLM forward pass, ANE jobs) that is not a
  // pipeline worker thread. Unlike record_perf_event, routing is by
  // `lane` (see perf-event.h PerfAuxLane), not the calling thread, so
  // it works from the dedicated LLM/MLX worker or a CoreML callback
  // thread. Callers reach this via the PerfAuxScope RAII helper
  // (common/perf-scope.h). Default impl is a no-op so adapter contexts
  // don't need to implement it.
  virtual void
  record_perf_event_aux(unsigned      /*lane*/,
                        std::uint32_t /*gvid*/,
                        std::uint32_t /*type*/,
                        std::uint64_t /*value*/) const noexcept
  {
  }

  // Session-shared CoreML model cache. Stages call
  // `coreml_model_manager()->load(path, compute_units)` to obtain a
  // shared_ptr to a loaded model; duplicate requests share one
  // load. Returns nullptr on non-Apple builds (and on Apple if
  // VPIPE_BUILD_APPLE_SILICON was disabled at build time, since the
  // manager type is then a forward declaration with no
  // implementation). Safe to call concurrently.
  virtual CoreMLModelManager* coreml_model_manager() const = 0;

  // Session-shared LLM manager (text + multi-modal). Lazily
  // constructed on first call. Backed by the metal-compute LM
  // subsystem, so the manager is only available on apple-silicon
  // builds; on any other build configuration the accessor returns
  // nullptr. Safe to call concurrently. See `generative-models/` for
  // the managed types.
  virtual genai::GenerativeModelManager* generative_model_manager() const = 0;

  // Session-shared Metal compute kernel framework (CUDA-shape
  // surface: load_library / make buffer / encode dispatch on a
  // stream). Lazily constructed on first call. Returns nullptr on
  // non-Apple builds. Even on Apple-Silicon the returned pointer
  // may be valid()==false (Metal unavailable); callers must check
  // before dispatching. Safe to call concurrently. See
  // apple-silicon/metal-compute/metal-compute.h.
  //
  // Defaulted here (returns nullptr) so adapter contexts that don't
  // need GPU resources (e.g. CerrSessionContext) need not override.
  virtual metal_compute::MetalCompute* metal_compute() const
  {
    return nullptr;
  }
};

}

#endif
