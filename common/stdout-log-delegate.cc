#include "common/stdout-log-delegate.h"
#include "common/thread-pool.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

using namespace std;

namespace vpipe {

// ---------------------------------------------------------------------
// Awaiters used by the consumer coroutine.
//
// ParkAwaiter: park the consumer's handle in the delegate so the
// next producer can resume it. The double check (await_ready +
// re-check after parking) closes the lost-wakeup race when a
// producer enqueues between the ready check and the parked store.
//
// YieldAwaiter: re-schedule the consumer onto the back of the pool
// queue, giving other workers a chance to run before we loop back
// for another drain pass.
// ---------------------------------------------------------------------

struct StdoutLogDelegate::ParkAwaiter {
  StdoutLogDelegate* self;

  bool
  await_ready() noexcept
  {
    return self->_stopping.load(memory_order_acquire) ||
           self->any_queue_nonempty_();
  }

  void
  await_suspend(coroutine_handle<> h) noexcept
  {
    // Hoist `self` into a function-local BEFORE publishing the
    // coroutine handle. Once `_parked = h` is observable, another
    // worker can resume this coroutine concurrently with us; that
    // resume can advance the body past this co_await, reuse the
    // awaiter's storage in the coroutine frame, or (if the frame
    // is destroyed) leave the slot freed. Any subsequent
    // `this->self` reload would be a use-after-free of the
    // awaiter. Reading through a stack local (or callee-saved
    // register) sidesteps that: the local lives on this worker's
    // stack, not in the coroutine frame.
    StdoutLogDelegate* d = self;
    {
      lock_guard<mutex> lk(d->_wake_mu);
      d->_parked = h;
    }
    // From here, do NOT touch `this`, `self`, or any other
    // awaiter member -- only `d` (and through it, the
    // StdoutLogDelegate, which lives independently of the frame).
    if (d->_stopping.load(memory_order_acquire) ||
        d->any_queue_nonempty_()) {
      coroutine_handle<> retaken;
      {
        lock_guard<mutex> lk(d->_wake_mu);
        if (d->_parked == h) {
          retaken = d->_parked;
          d->_parked = {};
        }
      }
      if (retaken) {
        d->_pool->schedule(retaken);
      }
    }
  }

  void await_resume() noexcept {}
};

struct StdoutLogDelegate::YieldAwaiter {
  ThreadPool* pool;

  bool await_ready() noexcept { return false; }

  void
  await_suspend(coroutine_handle<> h) noexcept
  {
    pool->schedule(h);
  }

  void await_resume() noexcept {}
};

// ---------------------------------------------------------------------
// Construction / destruction.
// ---------------------------------------------------------------------

StdoutLogDelegate::StdoutLogDelegate(LogLevel threshold)
  : _threshold(threshold)
{
}

StdoutLogDelegate::~StdoutLogDelegate()
{
  // If the owner forgot to detach (we expect Session to do it
  // explicitly), drop the consumer cleanly here. detach() is a
  // no-op if not attached.
  detach();
}

// ---------------------------------------------------------------------
// log() -- the producer-side hot path.
// ---------------------------------------------------------------------

void
StdoutLogDelegate::log(LogLevel level, const VpipeFormat& f)
{
  // 1. Threshold filter: skip everything for filtered messages.
  //    The formatter is NOT invoked.
  LogLevel thr = _threshold.load(memory_order_relaxed);
  if (level != LogLevel::Always &&
      static_cast<int>(level) > static_cast<int>(thr)) {
    return;
  }

  // 2. Sync fallback: no pool attached -> write inline.
  if (!_pool) {
    emit_sync_(level, f);
    return;
  }

  // 3. Async path: route to a per-worker queue and wake the
  //    consumer if it has parked.
  enqueue_(level, f);
  wake_consumer_if_parked_();
}

void
StdoutLogDelegate::emit_sync_(LogLevel level, const VpipeFormat& f)
{
  // Format outside the I/O mutex so the critical section is just
  // the stream write.
  string line;
  line.reserve(64);
  line.push_back('[');
  line.append(to_cstr(level));
  line.append("] ");
  try {
    line.append(f());
  } catch (...) {
    line.append("<formatter threw>");
  }
  line.push_back('\n');

  const bool to_err =
      (level == LogLevel::Error || level == LogLevel::Warn);

  lock_guard<mutex> lk(_io_mu);
  if (to_err) {
    cerr << line;
    cerr.flush();
  } else {
    // Explicit flush: when stdout is redirected to a file it is
    // fully buffered (not line-buffered), so without this the log
    // is invisible to an external tail-er / Python parent until the
    // userspace buffer accumulates ~4 KiB. cout.flush() only flushes
    // the C++ streambuf; we also fflush(stdout) so the underlying
    // stdio FILE*'s 4 KiB buffer drops to the kernel too. Logging
    // is low-volume so the extra syscalls per line are fine.
    cout << line;
    cout.flush();
    std::fflush(stdout);
  }
}

void
StdoutLogDelegate::enqueue_(LogLevel level, const VpipeFormat& f)
{
  unsigned wid = _pool->worker_id_of_current_thread();
  LogQueue& q  = (wid < _worker_qs.size())
               ? _worker_qs[wid]
               : _overflow_q;
  lock_guard<mutex> lk(q.mu);
  q.buf.push_back(LogEntry{level, f});
}

void
StdoutLogDelegate::wake_consumer_if_parked_()
{
  coroutine_handle<> h;
  {
    lock_guard<mutex> lk(_wake_mu);
    if (_parked) {
      h = _parked;
      _parked = {};
    }
  }
  if (h) {
    _pool->schedule(h);
  }
}

// ---------------------------------------------------------------------
// Drain helpers (consumer & flush share these).
// ---------------------------------------------------------------------

bool
StdoutLogDelegate::any_queue_nonempty_() const
{
  for (const auto& q : _worker_qs) {
    lock_guard<mutex> lk(q.mu);
    if (!q.buf.empty()) {
      return true;
    }
  }
  {
    lock_guard<mutex> lk(_overflow_q.mu);
    if (!_overflow_q.buf.empty()) {
      return true;
    }
  }
  return false;
}

bool
StdoutLogDelegate::drain_one_(LogQueue& q)
{
  vector<LogEntry> taken;
  {
    lock_guard<mutex> lk(q.mu);
    if (q.buf.empty()) {
      return false;
    }
    taken.swap(q.buf);
  }
  // Format + emit serially under _io_mu so any concurrent flush()
  // or sync writer can't tear against us.
  lock_guard<mutex> lk(_io_mu);
  for (const auto& e : taken) {
    emit_locked_(e);
  }
  return true;
}

bool
StdoutLogDelegate::drain_all_()
{
  bool did_any = false;
  for (auto& q : _worker_qs) {
    if (drain_one_(q)) {
      did_any = true;
    }
  }
  if (drain_one_(_overflow_q)) {
    did_any = true;
  }
  return did_any;
}

void
StdoutLogDelegate::emit_locked_(const LogEntry& entry)
{
  // Caller holds _io_mu. We format here -- inside the lock -- so
  // that the producer's enqueue() never pays the format cost. A
  // formatter that throws is replaced with a marker so we still
  // emit something useful (LogDelegateIntf MUST NOT throw out of
  // log()).
  string line;
  line.reserve(64);
  line.push_back('[');
  line.append(to_cstr(entry.level));
  line.append("] ");
  try {
    line.append(entry.fmt());
  } catch (...) {
    line.append("<formatter threw>");
  }
  line.push_back('\n');

  const bool to_err =
      (entry.level == LogLevel::Error || entry.level == LogLevel::Warn);
  if (to_err) {
    cerr << line;
    cerr.flush();
  } else {
    // Same reasoning as emit_sync_(): without this flush a parent
    // process reading our redirected stdout sees nothing until the
    // 4 KiB buffer fills, which makes the chat interaction look
    // hung even when inference is making steady progress. Flush
    // both the C++ streambuf and stdio FILE* layer for stability
    // across libc / libc++ implementations.
    cout << line;
    cout.flush();
    std::fflush(stdout);
  }
}

// ---------------------------------------------------------------------
// Consumer coroutine.
// ---------------------------------------------------------------------

Job
StdoutLogDelegate::consumer_loop_()
{
  while (!_stopping.load(memory_order_acquire)) {
    bool did_any = drain_all_();
    if (!did_any) {
      co_await ParkAwaiter{this};
    } else {
      // Yield: drop to the back of the pool queue between bursts
      // so we don't hog one worker.
      co_await YieldAwaiter{_pool};
    }
  }
  // Final drain pass after _stopping is set so anything that
  // arrived between the last loop iteration and the stop signal
  // still makes it out.
  drain_all_();

  {
    lock_guard<mutex> lk(_done_mu);
    _consumer_done = true;
  }
  _done_cv.notify_all();
  co_return;
}

// ---------------------------------------------------------------------
// attach / detach / flush.
// ---------------------------------------------------------------------

void
StdoutLogDelegate::attach(const SessionContextIntf* session)
{
  if (!session) {
    return;
  }
  ThreadPool* pool = session->thread_pool();
  if (!pool) {
    return;
  }
  // Idempotent for the same session.
  if (_pool == pool && _session == session) {
    return;
  }

  _session       = session;
  _pool          = pool;
  _stopping.store(false, memory_order_release);
  _consumer_done = false;
  _worker_qs     = vector<LogQueue>(pool->num_workers());
  // _overflow_q is a member; its buf is already empty.

  _consumer = consumer_loop_();
  _pool->schedule(_consumer.handle());
}

void
StdoutLogDelegate::detach()
{
  if (!_pool) {
    return;
  }

  // Ask the consumer to stop, then wake it (it may be parked).
  _stopping.store(true, memory_order_release);
  wake_consumer_if_parked_();

  // Wait for the consumer to reach final_suspend. The flag
  // `_consumer_done` is set inside the coroutine body BEFORE the
  // body returns, which means there is a small window between
  // "_consumer_done = true; notify_all()" and the coroutine
  // actually reaching the suspended-at-final_suspend state. If
  // we destroyed the Job (_h.destroy()) in that window the
  // coroutine frame would be torn down while still executing its
  // return epilogue, scribbling on libmalloc's freelist and
  // producing the long-standing rare "memory corruption of free
  // block" abort observed in the test binary.
  //
  // Two-phase wait: (1) cv on _consumer_done so we sleep cheaply
  // until the coroutine is essentially done, then (2) spin on
  // _h.done() until the coroutine is officially at
  // final_suspend. The spin window is typically a handful of
  // instructions; yield keeps it polite even if the scheduler
  // is unkind.
  {
    unique_lock<mutex> lk(_done_mu);
    _done_cv.wait(lk, [this] { return _consumer_done; });
  }
  while (_consumer.handle() && !_consumer.handle().done()) {
    std::this_thread::yield();
  }

  // The consumer's final pass already drained everything. Reset
  // the Job (destroys the coroutine frame, which is now at
  // final_suspend), the queues, and the pool pointer.
  _consumer = Job{};
  _worker_qs.clear();
  // overflow_q was drained by the consumer; defensively clear it.
  {
    lock_guard<mutex> lk(_overflow_q.mu);
    _overflow_q.buf.clear();
  }
  _pool          = nullptr;
  _session       = nullptr;
  _stopping.store(false, memory_order_release);
  _consumer_done = false;
}

void
StdoutLogDelegate::flush()
{
  if (!_pool) {
    return;
  }
  // Drain on the calling thread. _io_mu inside drain_one_ keeps us
  // serialised with the consumer; we may race the consumer for
  // entries (whoever takes a queue's buf wins), but the entries
  // themselves are not lost.
  while (drain_all_()) {
    // Loop in case the consumer concurrently appended a fresh batch
    // to a queue we already drained. In steady state one pass is
    // enough.
  }
}

// ---------------------------------------------------------------------
// Test-only accessors.
// ---------------------------------------------------------------------

size_t
StdoutLogDelegate::queue_size_for_test(unsigned worker_id) const
{
  if (!_pool || worker_id >= _worker_qs.size()) {
    return 0;
  }
  const LogQueue& q = _worker_qs[worker_id];
  lock_guard<mutex> lk(q.mu);
  return q.buf.size();
}

size_t
StdoutLogDelegate::overflow_queue_size_for_test() const
{
  if (!_pool) {
    return 0;
  }
  lock_guard<mutex> lk(_overflow_q.mu);
  return _overflow_q.buf.size();
}

}
