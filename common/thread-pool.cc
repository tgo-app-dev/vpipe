#include "common/thread-pool.h"
#include "common/job.h"
#include "common/perf-event.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <algorithm>
#include <cstdint>
#include <exception>

using namespace std;

namespace vpipe {

namespace {

// Per-OS-thread "which pool's worker am I, and at what index" stamp.
// Set by worker_loop() on entry, cleared on exit. The pair makes the
// lookup safe even when multiple ThreadPool instances coexist in one
// process: a ThreadPool only claims its own workers.
struct WorkerCtx {
  const ThreadPool* pool = nullptr;
  unsigned          idx  = ThreadPool::not_a_worker;
};
thread_local WorkerCtx tls_worker_ctx;

}

ThreadPool::ThreadPool(unsigned num_workers, const SessionContextIntf* s)
  : _session(s)
  , _num_workers(max(1u, num_workers))
{
  _workers.reserve(_num_workers);
  for (unsigned i = 0; i < _num_workers; ++i) {
    _workers.emplace_back([this, i] { worker_loop(i); });
  }
  _timer = thread([this] { timer_loop(); });
}

ThreadPool::~ThreadPool()
{
  // Set the stop flag under _mu so a worker between its predicate check
  // and wait() can't miss the wakeup (the flag is atomic so the timer
  // thread can read it without _mu).
  {
    lock_guard<mutex> lk(_mu);
    _stopping.store(true, memory_order_release);
  }
  _cv.notify_all();
  // Wake the timer under its mutex so a stop set between the timer's
  // predicate check and its wait() isn't lost (it would hang the join).
  {
    lock_guard<mutex> lk(_timer_mu);
  }
  _timer_cv.notify_all();
  if (_timer.joinable()) {
    _timer.join();
  }
  // The timer's shutdown flush may have pushed handles into the queue;
  // workers drain them before they exit (see worker_loop drain
  // semantics).
  _cv.notify_all();
  for (auto& t : _workers) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void
ThreadPool::schedule(coroutine_handle<> h)
{
  if (!h) {
    return;
  }
  {
    lock_guard<mutex> lk(_mu);
    _q.push(h);
  }
  _cv.notify_one();
}

void
ThreadPool::schedule_after(chrono::steady_clock::duration delay,
                           coroutine_handle<>            h)
{
  if (!h) {
    return;
  }
  if (delay <= chrono::steady_clock::duration::zero()) {
    schedule(h);
    return;
  }
  const auto deadline = chrono::steady_clock::now() + delay;
  {
    lock_guard<mutex> lk(_timer_mu);
    _timers.emplace(deadline, h);
  }
  _timer_cv.notify_one();
}

void
ThreadPool::timer_loop()
{
  unique_lock<mutex> lk(_timer_mu);
  while (!_stopping.load(memory_order_acquire)) {
    if (_timers.empty()) {
      // Wait for an insert or for stop. Holding lk across the empty
      // check + wait() closes the lost-wakeup window with
      // schedule_after (which inserts + notifies under _timer_mu).
      _timer_cv.wait(lk);
      continue;
    }
    const auto next = _timers.begin()->first;
    const auto now  = chrono::steady_clock::now();
    if (next > now) {
      // May wake early on a nearer insert or on stop; re-evaluate.
      _timer_cv.wait_until(lk, next);
      continue;
    }
    // Fire the earliest due handle. Drop the lock across schedule() so
    // a worker can pick it up while we keep servicing the heap.
    coroutine_handle<> h = _timers.begin()->second;
    _timers.erase(_timers.begin());
    lk.unlock();
    schedule(h);
    lk.lock();
  }
  // Shutdown: flush any still-pending handles to the worker queue so
  // the pool's drain runs them to final_suspend rather than stranding
  // a coroutine frame.
  while (!_timers.empty()) {
    coroutine_handle<> h = _timers.begin()->second;
    _timers.erase(_timers.begin());
    lk.unlock();
    schedule(h);
    lk.lock();
  }
}

unsigned
ThreadPool::worker_id_of_current_thread() const noexcept
{
  return (tls_worker_ctx.pool == this)
       ? tls_worker_ctx.idx
       : not_a_worker;
}

void
ThreadPool::worker_loop(unsigned my_idx)
{
  // Stamp the per-thread identity so worker_id_of_current_thread()
  // (and any other lookup that wants to know "which worker am I")
  // can resolve in O(1) without taking any of the pool's locks.
  tls_worker_ctx.pool = this;
  tls_worker_ctx.idx  = my_idx;

  bool exiting = false;
  while (!exiting) {
    coroutine_handle<> h;
    bool more = false;
    {
      unique_lock<mutex> lk(_mu);
      _cv.wait(lk, [this] {
        return _stopping.load(memory_order_acquire) || !_q.empty();
      });
      // Drain semantics: even after _stopping is set we keep popping
      // until the queue is empty so any in-flight chain can run to
      // final_suspend cleanly.
      if (_q.empty()) {
        exiting = true;
        break;
      }
      h = _q.front();
      _q.pop();
      more = !_q.empty();
    }

    // Cascade the wake. schedule() signals exactly one waiter per
    // push, and CV wakeups are edge-triggered and ordering-biased
    // (Darwin wakes ~LIFO, so the just-finished warm worker that
    // re-entered wait() keeps winning). Under sustained load a small
    // hot set then monopolizes the backlog while workers parked at the
    // bottom of the wait stack are never reached -- they show up as
    // permanently-idle lanes in the profiler even though runnable work
    // is queued. Re-signalling one waiter whenever work remains after a
    // pop forces the wake to propagate: each newly-woken worker that
    // still finds a backlog wakes the next, so the pool engages every
    // worker up to the depth of available work, then stops.
    if (more) {
      _cv.notify_one();
    }

    // Bracket the resume with schedule/unschedule perf events: the
    // worker records "schedule" (begin) right before running the
    // coroutine and "unschedule" (end) right after it suspends or
    // finishes. Both land in THIS worker's buffer (a resume and its
    // return run on this thread), so each thread's timeline shows clean
    // schedule->unschedule pairs. Gated to a single atomic-load no-op
    // when profiling is off, and skipped for untagged coroutines (e.g.
    // the log-delegate consumer). The tag is read BEFORE resume because
    // the handle may be at final_suspend afterwards.
    const bool          prof = _session && _session->profiling_enabled();
    const std::uint32_t tag  = prof ? perf_tag_of(h) : kNoPerfTag;
    if (prof && tag != kNoPerfTag) {
      _session->record_perf_event(tag, kPerfSchedule, 0);
    }
    // Resume must not propagate exceptions out of a worker. The
    // coroutine's promise captures any unhandled exception in
    // promise_type::unhandled_exception(); the only way exceptions
    // reach this catch is if resume() itself, or some non-coroutine
    // path inside it, throws.
    try {
      h.resume();
    } catch (const exception& e) {
      _session->warn(fmt(
        "ThreadPool: worker swallowed exception during resume: {}",
        e.what()));
    } catch (...) {
      _session->warn(fmt(
        "ThreadPool: worker swallowed non-std exception during resume"));
    }
    if (prof && tag != kNoPerfTag) {
      _session->record_perf_event(tag, kPerfUnschedule, 0);
    }
  }

  // Clear the stamp so a thread that for any reason is reused outside
  // the pool (it isn't, today) doesn't keep claiming this worker id.
  tls_worker_ctx.pool = nullptr;
  tls_worker_ctx.idx  = not_a_worker;
}

}
