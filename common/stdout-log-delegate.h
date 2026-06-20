#ifndef STDOUT_LOG_DELEGATE_H
#define STDOUT_LOG_DELEGATE_H

#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/log-delegate-intf.h"
#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <mutex>
#include <vector>

namespace vpipe {

class SessionContextIntf;
class ThreadPool;

// Writes log messages to std::cout / std::cerr. Filters on a level
// threshold: a message is emitted iff
//     level == LogLevel::Always || level <= threshold
// (smaller LogLevel ints are more severe). Format cost is paid only
// for messages that are going to be emitted -- the VpipeFormat
// callable is left untouched for filtered messages.
//
// Two operating modes:
//
//   * Sync (unattached): the original behaviour. log() runs the
//     formatter and writes to cout/cerr inline, serialised by an
//     internal mutex so lines never tear. Used during Session boot
//     before the pool exists, in tests that construct the delegate
//     directly, and after detach().
//
//   * Async (attached): after Session calls attach() with the
//     active SessionContextIntf, log() routes the entry into a
//     per-worker queue keyed by ThreadPool::worker_id_of_current_
//     thread() (or a shared overflow queue for non-worker callers).
//     A single consumer Job, scheduled on the same pool, drains
//     all queues serially -- format work happens on the consumer,
//     so producer-side cost is just an enqueue. The consumer
//     writes through one process-global I/O mutex (shared with the
//     sync path) so async and sync writes never tear against each
//     other.
//
// Shutdown: detach() drains every queue synchronously, waits for
// the consumer to reach final_suspend, then drops the pool pointer.
// The destructor calls detach() if still attached. Session is
// expected to call detach() (or destroy the delegate) BEFORE the
// pool itself is destroyed.
//
// Thread safety: log(), set_threshold(), and threshold() are MT-
// safe in both modes. attach() / detach() / flush() are not
// expected to race against each other -- Session sequences them.
class StdoutLogDelegate final : public LogDelegateIntf {
public:
  explicit StdoutLogDelegate(LogLevel threshold = LogLevel::Normal);
  ~StdoutLogDelegate() override;

  void log(LogLevel, const VpipeFormat&) override;

  void set_threshold(LogLevel level) override
  {
    _threshold.store(level, std::memory_order_relaxed);
  }
  LogLevel threshold() const noexcept override
  {
    return _threshold.load(std::memory_order_relaxed);
  }

  // Called by Session AFTER its ThreadPool is up. Allocates the
  // per-worker queues sized to pool->num_workers(), launches the
  // consumer Job onto the pool, and switches log() to the async
  // path. Idempotent for the same session pointer.
  void attach(const SessionContextIntf* session);

  // Called by Session BEFORE the pool is torn down. Sets the
  // stopping flag, wakes the parked consumer, waits for it to
  // reach final_suspend, then drains anything that snuck in on the
  // calling thread. After detach the delegate is back in sync
  // mode; safe to attach again.
  void detach();

  // Synchronously drain all queued messages on the calling thread.
  // No-op when not attached. Safe to call from any thread; takes
  // an internal I/O mutex so it can't tear against the consumer.
  void flush();

  // Test-only inspection: number of queued entries for a given
  // worker queue (0 .. num_workers()-1). Returns 0 for invalid
  // indices and 0 when not attached. Brief lock on the queue.
  std::size_t queue_size_for_test(unsigned worker_id) const;

  // Test-only inspection: number of queued entries in the overflow
  // queue (used by callers whose thread is not one of the pool's
  // workers). Returns 0 when not attached.
  std::size_t overflow_queue_size_for_test() const;

private:
  struct LogEntry {
    LogLevel    level;
    VpipeFormat fmt;
  };

  // Per-worker queue. Producer (the worker that owns this index)
  // appends under `mu`; consumer swap-takes the buffer under `mu`
  // and writes outside the lock. With one producer per queue this
  // is essentially uncontended in steady state.
  struct LogQueue {
    mutable std::mutex    mu;
    std::vector<LogEntry> buf;
  };

  // Park / wake awaiters used by the consumer coroutine. Defined
  // out-of-line in the .cc.
  struct ParkAwaiter;
  struct YieldAwaiter;

  // Entry path for the sync (unattached) fallback. Format under no
  // lock, then write under _io_mu. Mirrors the pre-async behaviour.
  void emit_sync_(LogLevel, const VpipeFormat&);

  // Append a copy of `f` to the queue routed by the current worker
  // identity. Does NOT format -- the consumer does that.
  void enqueue_(LogLevel, const VpipeFormat&);

  // Wake the consumer if it has parked itself. Producers call this
  // after enqueue_(). Lock held only briefly to take the parked
  // handle.
  void wake_consumer_if_parked_();

  // Drain a single queue: swap-take its buffer, then format + emit
  // each entry under _io_mu. Returns true iff at least one entry
  // was emitted. Safe to call from any thread (used by both the
  // consumer and flush()).
  bool drain_one_(LogQueue& q);

  // Drain every queue once. Returns true iff any entry was emitted.
  bool drain_all_();

  // Format `entry` and write to cout/cerr under _io_mu. Tolerant
  // of formatter exceptions (replaces with a marker).
  void emit_locked_(const LogEntry& entry);

  // The consumer body. Loops while !_stopping: drain, then either
  // park (if the pass was empty) or yield to the back of the pool
  // queue (otherwise). On stopping, performs one final drain pass
  // and signals _consumer_done.
  Job consumer_loop_();

  // True iff any per-worker or overflow queue currently has
  // entries. Cheap; just walks vectors and reads `.empty()` under
  // each queue's mu.
  bool any_queue_nonempty_() const;

  std::atomic<LogLevel> _threshold;

  // I/O serialisation. Used by both the sync fallback and the
  // async consumer's emit_locked_; that way a flush() called on a
  // background thread can't interleave with the consumer.
  std::mutex _io_mu;

  // Async-mode state. _pool == nullptr means "not attached".
  const SessionContextIntf*  _session = nullptr;
  ThreadPool*                _pool    = nullptr;

  std::vector<LogQueue>      _worker_qs;
  LogQueue                   _overflow_q;

  // Wake handoff: consumer parks its handle here; producer takes
  // and schedules. Guarded by _wake_mu.
  std::mutex                 _wake_mu;
  std::coroutine_handle<>    _parked{};

  std::atomic<bool>          _stopping{false};

  // Consumer-done sync: the coroutine sets _consumer_done and
  // notifies _done_cv at the bottom of its body. detach() waits
  // here.
  std::mutex                 _done_mu;
  std::condition_variable    _done_cv;
  bool                       _consumer_done = false;

  // Owns the consumer coroutine. Default-constructed Job is a
  // tombstone; attach() assigns a started-suspended coroutine into
  // it. detach() waits for the coroutine to complete and then
  // assigns a fresh tombstone, which destroys the frame.
  Job                        _consumer;
};

}

#endif
