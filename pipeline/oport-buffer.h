#ifndef OPORT_BUFFER_H
#define OPORT_BUFFER_H

#include "common/beat-payload-intf.h"
#include "common/oport-policy.h"
#include "common/session-member.h"

#include <atomic>
#include <coroutine>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace vpipe {

// Per-producer-oport shared buffer. Holds payloads (unique_ptr<
// BeatPayloadIntf>) in a ring sized to next_pow2(hard_cap). Sequence
// numbers are std::atomic<uint32_t>; (wp - rp) is bounded by
// OportPolicy::kMaxCapacity (1<<20) so uint32 subtract-with-
// wraparound yields the correct distance.
//
// Per-cursor reader semantics:
//   * peek(offset) -- const-borrow at rp+offset; suspends until that
//     seq is written or EOS. Does NOT advance rp.
//   * acquire() -- move-out at rp; succeeds only when the caller is
//     the slowest (or only) cursor. Suspends only for "not yet
//     written" / EOS; the not-slowest case returns nullopt
//     immediately so the caller can fall back to peek+clone+release.
//     Advances rp by 1 on success.
//   * release_read(n) -- explicit advance. Non-blocking; wakes the
//     writer if it was backpressured.
//
// Overrun policy:
//   * Backpressure (default) -- the producer suspends when the
//     buffer is full. "Full" iff active >= hard_cap where active =
//     wp - min(cursor.next_seq).
//   * DropOldest -- on overflow the producer snaps the (single)
//     cursor forward to discard the oldest slot, increments the
//     cursor's drop counter, then writes the new slot. Enforced to
//     be single-cursor at attach_cursor() time.
//
// Default-depth mode (capacity == 0 in OportPolicy): the ring is
// sized to next_pow2(soft_error) (default 2048). Soft thresholds
// fire on push:
//   * active >= soft_warn -- one-shot session->warn().
//   * active >= soft_error -- the write attempt throws
//     std::runtime_error (caller's process() unwinds).
// Backpressure never kicks in in default mode.
//
// EOS: producer calls close() when done. A cursor that finds its
// slot range empty and the buffer closed gets a null pointer back
// from its ReadAwaiter / PeekAwaiter / AcquireAwaiter. A writer
// that races close() gets a runtime_error out of
// WriteAwaiter::await_resume().
//
// No-consumer behavior: with zero attached cursors, push succeeds
// without storing anything. Producers don't have to gate every
// write. Use RuntimeContext::has_consumers() if the producer wants
// to skip the data production work too.
//
// Thread safety: hot-path advances (wp store, cursor.next_seq
// fetch_add) are atomic. _mu guards waiter slots, slot writes, the
// slot-unique_ptr move on acquire, DropOldest's cursor snap,
// _cursors.push_back, and _closed. Wakeups via thread_pool->schedule
// happen *after* _mu is released.
class EdgeReader;
class ThreadPool;

// Shared wake-state for a "wait on any of N iports" suspend
// (RuntimeContext::read_any). One coroutine handle is registered as a
// pending waiter on several cursors at once; the FIRST cursor to become
// readable (a write satisfying its seq, or close()) wakes it, exactly
// once. The race that makes this delicate -- a producer resuming the
// handle on another worker while await_suspend is still registering the
// remaining cursors -- is closed with an armed/fired handshake under
// `mu`: a satisfied cursor calls notify(), which marks `fired` but only
// schedules the handle once `armed` is set (the last thing await_suspend
// does, after all registrations). So the handle is never resumed until
// the suspend is fully published, and never resumed twice.
struct MultiReadWaiter {
  std::mutex              mu;
  std::coroutine_handle<> h{};
  ThreadPool*             pool  = nullptr;
  bool                    fired = false;   // a cursor has claimed the wake
  bool                    armed = false;   // await_suspend finished publishing

  // Called by a satisfied cursor (under no lock). Schedules `h` at most
  // once, and only after await_suspend armed the waiter; if it fires
  // before arming, await_suspend sees `fired` and resumes inline.
  void notify();
};

class OportBuffer : public SessionMember {
public:
  // capacity == 0 selects default mode (ring sized next_pow2(
  // soft_error), soft thresholds armed). Non-zero capacity is
  // clamped to [1, OportPolicy::kMaxCapacity].
  explicit
  OportBuffer(const SessionContextIntf* session,
              OportPolicy policy = {});
  ~OportBuffer() override;

  OportBuffer(const OportBuffer&)            = delete;
  OportBuffer& operator=(const OportBuffer&) = delete;

  // Producer-side awaiter for writing a payload pointer into the
  // buffer. await_resume() throws std::runtime_error if the buffer
  // was closed mid-write, or if the soft-error threshold was
  // exceeded in default-depth mode.
  struct WriteAwaiter {
    OportBuffer*                       _buf;
    std::unique_ptr<BeatPayloadIntf>   _value;
    bool                               _delivered     = false;
    bool                               _saw_closed    = false;
    bool                               _saw_threshold = false;

    bool await_ready();
    bool await_suspend(std::coroutine_handle<> h);
    void await_resume();
  };

  WriteAwaiter
  write(std::unique_ptr<BeatPayloadIntf> t)
  {
    return WriteAwaiter{this, std::move(t)};
  }

  // Non-coroutine push for a producer running on a thread it owns
  // (not a pool worker / coroutine), so it cannot co_await the
  // WriteAwaiter. Performs the same push + reader-wakeup as
  // WriteAwaiter's ready path. Returns false iff the buffer is closed
  // -- the caller should stop producing. Intended for non-
  // backpressure oports (DropOldest / no-consumer): a Backpressure
  // oport that is momentarily full, or a default-mode soft-error,
  // cannot suspend a plain thread, so the payload is dropped and the
  // call still returns true (buffer open). Use the coroutine
  // WriteAwaiter when backpressure must actually be honoured.
  bool push_sync(std::unique_ptr<BeatPayloadIntf> t);

  // Producer signals end-of-stream. After close():
  //   * every suspended cursor reader is woken; their next
  //     read/peek/acquire returns null once their position
  //     reaches the end of writes;
  //   * a suspended writer is woken; its await_resume throws;
  //   * any subsequent write throws.
  void close();

  // Called at pipeline launch only. Appends a fresh cursor and
  // returns its index. Throws in DropOldest mode if a cursor is
  // already attached.
  unsigned attach_cursor();

  bool          closed() const noexcept;
  unsigned      capacity() const noexcept { return _hard_cap; }
  OverrunPolicy mode()     const noexcept { return _mode; }
  unsigned      size() const;                 // active count
  unsigned      num_cursors() const;
  unsigned      dropped() const;              // total drops
  unsigned      dropped_for_cursor(unsigned cursor_idx) const;

private:
  friend class EdgeReader;

  struct Cursor {
    std::atomic<std::uint32_t>             next_seq{0};
    std::atomic<std::uint32_t>             dropped{0};
    // Guarded by _mu. `waiter` (single-port read/peek/acquire) and
    // `multi` (read_any) are mutually exclusive -- a cursor has one
    // consumer coroutine, so only one wait is pending at a time. Both
    // use `waiter_seq` as the at-least wp that satisfies the wait.
    std::optional<std::coroutine_handle<>> waiter;
    std::shared_ptr<MultiReadWaiter>       multi;
    std::uint32_t                          waiter_seq = 0;
  };

  // Computes (wp - min(cursor.next_seq)) using relaxed loads of
  // cursor seqs. Stale reads under-estimate progress (cursors only
  // advance), so the returned count is a safe upper bound on
  // active.
  std::uint32_t active_count_relaxed_() const noexcept;

  // Locked helpers (caller holds _mu).
  struct PushResult {
    bool                                          success     = false;
    bool                                          closed      = false;
    bool                                          threshold   = false;
    std::vector<std::coroutine_handle<>>          wake_readers;
    std::vector<std::shared_ptr<MultiReadWaiter>> wake_multi;
  };
  PushResult
  push_locked_(std::unique_ptr<BeatPayloadIntf>& v);

  // Borrow at cursor.next_seq + offset. Returns the slot pointer if
  // available (saw_eos false), or sets saw_eos true if buffer is
  // closed and the seq is past _wp.
  struct PeekResult {
    const BeatPayloadIntf* p = nullptr;
    bool                   saw_eos = false;
    bool                   not_ready = false;
  };
  PeekResult
  peek_locked_(unsigned cursor_idx, std::uint32_t offset) const;

  // Move-out at cursor.next_seq. Returns the moved unique_ptr on
  // success and advances next_seq. Otherwise sets one of the
  // not_slowest / not_ready / saw_eos flags appropriately.
  struct AcquireResult {
    std::unique_ptr<BeatPayloadIntf> p;
    bool                             not_slowest = false;
    bool                             not_ready   = false;
    bool                             saw_eos     = false;
    std::optional<std::coroutine_handle<>> wake_writer;
  };
  AcquireResult
  acquire_locked_(unsigned cursor_idx);

  // Register a waiter on cursor for at-least seq waiter_seq.
  void
  register_reader_waiter_locked_(unsigned                cursor_idx,
                                 std::uint32_t           waiter_seq,
                                 std::coroutine_handle<> h);

  // Register a multi-port (read_any) waiter on cursor for at-least seq
  // waiter_seq. Clears any stale single waiter (mutually exclusive).
  void
  register_reader_multi_waiter_locked_(
      unsigned                         cursor_idx,
      std::uint32_t                    waiter_seq,
      std::shared_ptr<MultiReadWaiter> w);

  // Drop a multi-port waiter from cursor iff it is still `w` (locks
  // _mu). Called on resume to clear the stale registrations on the
  // cursors that did not win the wake.
  void
  deregister_reader_multi_waiter_(
      unsigned                                cursor_idx,
      const std::shared_ptr<MultiReadWaiter>& w);

  // Advance cursor.next_seq by n (clamped to wp); returns the
  // writer handle to wake if the writer was backpressured and the
  // buffer just fell below hard_cap. Lock is needed only for the
  // writer-waiter handoff; the next_seq fetch_add itself is atomic.
  std::optional<std::coroutine_handle<>>
  release_read_(unsigned cursor_idx, std::uint32_t n);

  unsigned         _hard_cap;
  std::uint32_t    _ring_size;     // _slots.size(), pow2
  std::uint32_t    _ring_mask;     // _ring_size - 1
  OverrunPolicy    _mode;

  // Default-mode soft thresholds. Both 0 means thresholds disabled
  // (user-specified capacity).
  std::uint32_t    _soft_warn;
  std::uint32_t    _soft_error;
  bool             _warn_logged = false;

  std::uint32_t    _dropped = 0;   // total drops, all causes

  struct Slot {
    std::unique_ptr<BeatPayloadIntf> p;
  };
  std::vector<Slot> _slots;

  std::atomic<std::uint32_t> _wp{0};

  std::vector<std::unique_ptr<Cursor>> _cursors;

  std::optional<std::coroutine_handle<>> _writer_waiter;
  bool                                   _closed = false;
  mutable std::mutex                     _mu;
};

}

#endif
