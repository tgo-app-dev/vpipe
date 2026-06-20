#ifndef PIPELINE_RUNTIME_H
#define PIPELINE_RUNTIME_H

#include "common/job.h"
#include "common/session-member.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe {

class EdgeReader;
class InitBarrier;
class OportBuffer;
class Pipeline;
class RuntimeContext;
class Stage;

// Per-pipeline runtime: builds the edge buffers and runtime contexts,
// spawns one driver coroutine per Stage, schedules them on the
// session ThreadPool, and exposes pause/stop/wait_idle.
//
// Lifetime: one launch per instance. Construct, launch(), then either
// wait_idle() (data-driven shutdown) or stop() (force shutdown), then
// destroy. Destructor calls stop() if still running so that the
// drivers are guaranteed to be at final_suspend before their Job
// objects are destroyed.
//
// Pool and default-edge-capacity are reached through SessionMember's
// `session()->thread_pool()` / `session()->default_edge_capacity()`.
class PipelineRuntime : public SessionMember {
public:
  PipelineRuntime(Pipeline*                 pipeline,
                  const SessionContextIntf* session);

  ~PipelineRuntime() override;

  PipelineRuntime(const PipelineRuntime&)            = delete;
  PipelineRuntime& operator=(const PipelineRuntime&) = delete;

  // Build buffers / contexts / drivers and start them. Returns true
  // on success, false if the graph is malformed or has already been
  // launched. Any failure logs through the SessionContextIntf.
  bool launch();

  // Request stop. Drivers exit on the next iteration boundary. Does
  // not block. Idempotent.
  void pause();

  // pause() + close all buffers (wakes any suspended drivers) +
  // wait_idle(). Idempotent.
  void stop();

  // Block the calling thread until every driver has reached its
  // co_return.
  void wait_idle();

  // Bounded variant: wait up to `timeout_ms` milliseconds. Returns
  // true if every driver finished, false on timeout. A negative
  // timeout is equivalent to the unbounded wait_idle() above and
  // always returns true. timeout_ms == 0 is a non-blocking probe.
  bool wait_idle(int timeout_ms);

  bool running() const noexcept
  {
    return _running.load(std::memory_order_acquire);
  }

  // True once every driver has reached co_return ON ITS OWN -- i.e.
  // every stage signalled done() and the pipeline drained -- with NO
  // pause()/stop() having been requested. This is the "all stages
  // signalled done" condition a long-lived host (the web-ui) uses to
  // auto-stop a self-terminating pipeline. False before launch(), after
  // pause()/stop() (those set _stop), and while any driver is still
  // running. All reads are atomic, so it is safe to poll from any
  // thread.
  bool self_completed() const noexcept
  {
    return _running.load(std::memory_order_acquire)
        && !_stop.load(std::memory_order_acquire)
        && _expected > 0
        && _completed.load(std::memory_order_acquire) >= _expected;
  }

  unsigned num_drivers() const noexcept
  {
    return static_cast<unsigned>(_drivers.size());
  }

  // Per-edge buffer-utilization snapshot, for diagnostics and the
  // web-ui overlay. One entry per consumer iport (== one EdgeReader),
  // labelled by the producer/consumer stage ids and port indices so a
  // caller can join it to the rendered graph edges. Read while the
  // pipeline runs: the counts come from relaxed atomics / short locks,
  // so a snapshot may be slightly stale but never corrupt. Empty before
  // launch().
  struct EdgeBufferStat {
    std::string   from_id;     // producer stage id
    unsigned      from_port;   // producer oport index
    std::string   to_id;       // consumer stage id
    unsigned      to_port;     // consumer iport index
    std::uint32_t backlog;     // items pending for THIS consumer cursor
    unsigned      capacity;    // producer buffer hard cap (ring depth)
    std::uint32_t dropped;     // items dropped for this consumer
    bool          closed;      // producer signalled EOS
  };
  std::vector<EdgeBufferStat> edge_buffer_stats() const;

private:
  // Spin until every driver's coroutine handle reports done(),
  // i.e. has reached final_suspend. Caller must already have
  // satisfied _completed >= _expected via the done condvar.
  void wait_drivers_suspended_();

public:

private:
  Pipeline*                                    _pipeline;

  // The real (post-inline) stages this runtime drives, captured at
  // launch. Used by stop() to clear each stage's running flag once the
  // drivers have drained, so post-stop topology edits are permitted.
  std::vector<Stage*>                          _live_stages;

  std::vector<std::unique_ptr<OportBuffer>>    _oport_bufs;
  std::vector<std::unique_ptr<EdgeReader>>     _readers;
  // Parallel to _readers: the producer/consumer labels for each edge,
  // captured at launch so edge_buffer_stats() can name them without
  // re-walking the graph.
  std::vector<EdgeBufferStat>                  _edge_labels;
  std::vector<std::unique_ptr<RuntimeContext>> _contexts;
  std::vector<Job>                             _drivers;
  std::unique_ptr<InitBarrier>                 _init_barrier;

  std::atomic<bool>                            _stop{false};
  std::atomic<bool>                            _running{false};
  bool                                         _launched = false;

  std::atomic<unsigned>                        _completed{0};
  unsigned                                     _expected = 0;
  std::mutex                                   _done_mu;
  std::condition_variable                      _done_cv;
};

}

#endif
