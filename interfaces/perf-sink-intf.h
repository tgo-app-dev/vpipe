// perf-sink-intf.h -- the session-level performance event sink.
//
// Sessions own a small fixed set of PerfBuffers -- one per ThreadPool
// worker plus one shared "overflow" buffer used by non-worker callers.
// Stages call record_perf_event(), which routes to the right buffer
// based on the calling thread's worker id. Memory bound =
// (num_workers + 1) * max_events_per_thread * sizeof(PerfEvent); the
// user controls total memory via max_events_per_thread.
//
// Everything here is defaulted to inert, so a context built outside a
// real Session inherits a no-op profiler rather than implementing one.
// PerfAuxScope (common/perf-scope.h) needs only this role, not a whole
// session context.

#ifndef PERF_SINK_INTF_H
#define PERF_SINK_INTF_H

#include <chrono>
#include <cstdint>

namespace vpipe {

class PerfSinkIntf {
public:
  virtual ~PerfSinkIntf() = default;

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
  // it works from the dedicated LLM worker or a CoreML callback thread.
  virtual void
  record_perf_event_aux(unsigned      /*lane*/,
                        std::uint32_t /*gvid*/,
                        std::uint32_t /*type*/,
                        std::uint64_t /*value*/) const noexcept
  {
  }
};

}

#endif
