#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vpipe {

class SessionContextIntf;

// Generic worker pool. The pool only knows about
// std::coroutine_handle<>; all of vpipe's runnable work (per-stage
// drivers, woken-up awaiters from EdgeBuffer, etc.) is wrapped in a
// coroutine and submitted via schedule().
//
// Lifetime: workers are launched in the constructor and joined in the
// destructor. The destructor sets a stopping flag, drains any handles
// still in the queue (each is resumed, exactly once), then joins.
// Callers are responsible for ensuring all running coroutines have
// reached a stable state (final_suspend) before the pool is
// destroyed; the runtime layer does this via PipelineRuntime::stop().
//
// Thread safety: schedule() and num_workers() are MT-safe. The pool
// must not be moved or copied.
class ThreadPool {
public:
  ThreadPool(unsigned num_workers, const SessionContextIntf*);
  ~ThreadPool();

  ThreadPool(const ThreadPool&)            = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&)                 = delete;
  ThreadPool& operator=(ThreadPool&&)      = delete;

  // Enqueue a coroutine handle to be resumed on a worker thread.
  // The pool calls h.resume() exactly once. If the coroutine
  // suspends, the awaiter is responsible for re-scheduling it. If
  // it runs to final_suspend the symmetric-transfer chain unwinds
  // naturally; the pool never .destroy()s a handle.
  void schedule(std::coroutine_handle<> h);

  // Like schedule(), but the handle is enqueued only after `delay`
  // elapses. A single dedicated timer thread holds the pending
  // deadlines and hands each handle to the normal worker queue when it
  // fires, so the awaiting coroutine does NOT pin a worker thread while
  // it waits -- the worker is returned to the pool immediately. This is
  // how periodic/timed source stages (e.g. chrono) wait without burning
  // a worker on a blocking sleep. A non-positive delay schedules
  // immediately. On pool teardown any still-pending handles are flushed
  // to the queue so the drain can run them to final_suspend.
  void
  schedule_after(std::chrono::steady_clock::duration delay,
                 std::coroutine_handle<>             h);

  unsigned num_workers() const noexcept { return _num_workers; }

  // Sentinel returned by worker_id_of_current_thread() when the
  // calling thread is not one of *this* pool's workers. Stable;
  // safe to use as a "no worker" tag in callers that want a flat
  // unsigned-keyed table.
  static constexpr unsigned not_a_worker =
      std::numeric_limits<unsigned>::max();

  // Returns 0..num_workers()-1 if the calling thread is one of this
  // pool's workers; not_a_worker otherwise. O(1), never blocks.
  // Workers of a *different* ThreadPool instance also map to
  // not_a_worker -- the lookup is per-instance so two pools
  // alive in the same process don't see each other's workers.
  // Used by per-worker queue routing in StdoutLogDelegate.
  unsigned worker_id_of_current_thread() const noexcept;

private:
  void worker_loop(unsigned my_idx);
  void timer_loop();

  const SessionContextIntf*            _session;
  unsigned                             _num_workers;
  std::vector<std::thread>             _workers;
  std::mutex                           _mu;
  std::condition_variable              _cv;
  std::queue<std::coroutine_handle<>>  _q;
  std::atomic<bool>                    _stopping{false};

  // Delayed-schedule timer: one thread waits on the earliest deadline
  // in `_timers` and moves due handles into the worker queue via
  // schedule(). Separate mutex/cv from the worker queue so timer
  // bookkeeping never contends with the hot schedule() path.
  std::thread                          _timer;
  std::mutex                           _timer_mu;
  std::condition_variable              _timer_cv;
  std::multimap<std::chrono::steady_clock::time_point,
                std::coroutine_handle<>> _timers;
};

}

#endif
