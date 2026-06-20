#include "pipeline/oport-buffer.h"
#include "common/thread-pool.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

uint32_t
round_up_pow2_(uint32_t v) noexcept
{
  if (v <= 1) { return 1; }
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

}

OportBuffer::OportBuffer(const SessionContextIntf* session,
                         OportPolicy               policy)
  : SessionMember(session)
  , _mode(policy.mode)
{
  // Resolve hard_cap. capacity == 0 -> default-mode (ring sized to
  // soft_error; soft thresholds armed). capacity != 0 -> user-set,
  // clamped to [1, kMaxCapacity]; soft thresholds disabled.
  if (policy.capacity == 0) {
    // Default mode: warn at soft_warn, error+throw at soft_error.
    // Soft thresholds default in OportPolicy to 1024 / 2048.
    _soft_warn  = policy.soft_warn;
    _soft_error = policy.soft_error;
    _hard_cap   = max<uint32_t>(1, _soft_error);
  } else {
    unsigned cap = policy.capacity;
    if (cap > OportPolicy::kMaxCapacity) {
      cap = OportPolicy::kMaxCapacity;
    }
    _hard_cap   = cap;
    _soft_warn  = 0;
    _soft_error = 0;
  }
  _ring_size = round_up_pow2_(_hard_cap);
  _ring_mask = _ring_size - 1;
  _slots.resize(_ring_size);
}

OportBuffer::~OportBuffer() = default;

void
MultiReadWaiter::notify()
{
  coroutine_handle<> to_wake{};
  {
    lock_guard<mutex> lk(mu);
    if (fired) {
      return;
    }
    fired = true;
    // Only schedule once await_suspend has armed us; if we fire first,
    // await_suspend sees `fired` and resumes the coroutine inline.
    if (armed) {
      to_wake = h;
    }
  }
  if (to_wake && pool) {
    pool->schedule(to_wake);
  }
}

void
OportBuffer::close()
{
  optional<coroutine_handle<>>             writer;
  vector<coroutine_handle<>>               readers;
  vector<shared_ptr<MultiReadWaiter>>      multis;
  {
    lock_guard<mutex> lk(_mu);
    if (_closed) {
      return;
    }
    _closed = true;
    writer  = std::exchange(_writer_waiter, nullopt);
    for (auto& c : _cursors) {
      if (c->waiter) {
        readers.push_back(*c->waiter);
        c->waiter.reset();
      }
      if (c->multi) {
        multis.push_back(std::move(c->multi));
        c->multi.reset();
      }
    }
  }
  if (writer) {
    session()->thread_pool()->schedule(*writer);
  }
  for (auto h : readers) {
    session()->thread_pool()->schedule(h);
  }
  // EOS is "readable" for a read_any waiter -- wake it so its next
  // read() on the closed port returns null.
  for (auto& m : multis) {
    m->notify();
  }
}

unsigned
OportBuffer::attach_cursor()
{
  lock_guard<mutex> lk(_mu);
  if (_mode == OverrunPolicy::DropOldest && !_cursors.empty()) {
    throw runtime_error(
        "OportBuffer: DropOldest mode allows only one cursor");
  }
  auto c = make_unique<Cursor>();
  c->next_seq.store(_wp.load(memory_order_relaxed),
                    memory_order_relaxed);
  unsigned idx = static_cast<unsigned>(_cursors.size());
  _cursors.push_back(std::move(c));
  return idx;
}

bool
OportBuffer::closed() const noexcept
{
  lock_guard<mutex> lk(_mu);
  return _closed;
}

unsigned
OportBuffer::size() const
{
  return active_count_relaxed_();
}

unsigned
OportBuffer::num_cursors() const
{
  lock_guard<mutex> lk(_mu);
  return static_cast<unsigned>(_cursors.size());
}

unsigned
OportBuffer::dropped() const
{
  lock_guard<mutex> lk(_mu);
  return _dropped;
}

unsigned
OportBuffer::dropped_for_cursor(unsigned cursor_idx) const
{
  lock_guard<mutex> lk(_mu);
  return _cursors.at(cursor_idx)->dropped.load(memory_order_relaxed);
}

uint32_t
OportBuffer::active_count_relaxed_() const noexcept
{
  // Caller may or may not hold _mu. Reads cursor.next_seq with
  // relaxed ordering; since cursors only advance, a stale read
  // under-estimates progress -- the returned active is a safe
  // upper bound.
  uint32_t wp = _wp.load(memory_order_relaxed);
  uint32_t min_seq = wp;
  bool any = false;
  for (auto& c : _cursors) {
    uint32_t s = c->next_seq.load(memory_order_relaxed);
    if (!any || (wp - s) > (wp - min_seq)) {
      min_seq = s;
      any = true;
    }
  }
  if (!any) {
    return 0;          // no cursors -> producer drops silently
  }
  return wp - min_seq;
}

OportBuffer::PushResult
OportBuffer::push_locked_(unique_ptr<BeatPayloadIntf>& v)
{
  PushResult r;
  if (_closed) {
    r.closed = true;
    return r;
  }
  if (_cursors.empty()) {
    // No consumer wired: drop silently (preserves the historical
    // "disconnected oport: no-op" semantic).
    r.success = true;
    v.reset();
    return r;
  }
  uint32_t wp     = _wp.load(memory_order_relaxed);
  uint32_t active = active_count_relaxed_();

  // Soft thresholds (default-depth mode only).
  if (_soft_error != 0 && active >= _soft_error) {
    r.threshold = true;
    return r;
  }
  if (_soft_warn != 0 && !_warn_logged && active >= _soft_warn) {
    _warn_logged = true;
    // Log outside _mu? Format is cheap (no allocation in steady
    // state); the warn() call may itself acquire a log-delegate
    // mutex. Calling session()->warn under _mu is safe as long as
    // the log delegate doesn't call back into this buffer -- which
    // it doesn't.
    session()->warn(fmt(
        "OportBuffer: active beats {} crossed soft_warn={} "
        "threshold (soft_error={})",
        active, _soft_warn, _soft_error));
  }

  if (_soft_error == 0 && active >= _hard_cap) {
    if (_mode == OverrunPolicy::DropOldest) {
      // attach_cursor enforced single-cursor in this mode.
      Cursor& c = *_cursors.front();
      uint32_t need = active - _hard_cap + 1;
      uint32_t cur  = c.next_seq.load(memory_order_relaxed);
      c.next_seq.store(cur + need, memory_order_relaxed);
      c.dropped.fetch_add(need, memory_order_relaxed);
      _dropped += need;
    } else {
      // Backpressure: caller will suspend.
      return r;
    }
  }

  // Place the payload in the slot at wp. Advance _wp last with
  // release ordering so readers acquiring _wp see the slot.
  _slots[wp & _ring_mask].p = std::move(v);
  _wp.store(wp + 1, memory_order_release);
  r.success = true;

  uint32_t new_wp = wp + 1;
  for (auto& c : _cursors) {
    // waiter_seq is the minimum wp value at which this reader's target
    // slot is available (semantically "at-least"). The distance check
    // rejects wraparound / waiter-set-for-future-seq cases; `<=
    // kMaxCapacity` accepts an exact match (the common interactive
    // case: reader registered with waiter_seq = cs + 1, then a single
    // write makes new_wp == cs + 1). Using `> 0` here was wrong: it
    // required TWO writes to wake a single-beat-at-a-time waiter.
    if (c->multi) {
      uint32_t d = new_wp - c->waiter_seq;
      if (d <= OportPolicy::kMaxCapacity) {
        r.wake_multi.push_back(std::move(c->multi));
        c->multi.reset();
      }
    } else if (c->waiter) {
      uint32_t d = new_wp - c->waiter_seq;
      if (d <= OportPolicy::kMaxCapacity) {
        r.wake_readers.push_back(*c->waiter);
        c->waiter.reset();
      }
    }
  }
  return r;
}

OportBuffer::PeekResult
OportBuffer::peek_locked_(unsigned cursor_idx,
                          uint32_t offset) const
{
  PeekResult r;
  uint32_t wp        = _wp.load(memory_order_acquire);
  uint32_t cursor_sq = _cursors[cursor_idx]->next_seq.load(
      memory_order_relaxed);
  uint32_t target    = cursor_sq + offset;
  if ((wp - target) > 0 && (wp - target) < (1u << 31)) {
    r.p = _slots[target & _ring_mask].p.get();
    return r;
  }
  if (_closed) {
    r.saw_eos = true;
    return r;
  }
  r.not_ready = true;
  return r;
}

OportBuffer::AcquireResult
OportBuffer::acquire_locked_(unsigned cursor_idx)
{
  AcquireResult r;
  Cursor& c = *_cursors[cursor_idx];
  uint32_t wp        = _wp.load(memory_order_acquire);
  uint32_t cursor_sq = c.next_seq.load(memory_order_relaxed);
  if ((wp - cursor_sq) == 0) {
    if (_closed) {
      r.saw_eos = true;
    } else {
      r.not_ready = true;
    }
    return r;
  }
  // Check we're STRICTLY slower than every other cursor. Tied
  // cursors must clone -- the slot's unique_ptr can only be moved
  // out once, so the last reader (strictly slowest) takes it.
  for (auto& cc : _cursors) {
    if (cc.get() == &c) { continue; }
    uint32_t s = cc->next_seq.load(memory_order_relaxed);
    // (cursor_sq - s) mod 2^32 interpretation:
    //   == 0           -> tied; another cursor is at the same seq;
    //   in (0, kMax]   -> s is strictly behind me; not slowest;
    //   > kMax         -> wraparound; s is ahead; OK so far.
    if ((cursor_sq - s) <= OportPolicy::kMaxCapacity) {
      r.not_slowest = true;
      return r;
    }
  }
  // We are slowest (or only). Move the slot out.
  r.p = std::move(_slots[cursor_sq & _ring_mask].p);
  c.next_seq.store(cursor_sq + 1, memory_order_release);
  // Wake the writer if it was backpressured and we just opened a
  // slot.
  if (_writer_waiter) {
    uint32_t active_after = active_count_relaxed_();
    if (active_after < _hard_cap) {
      r.wake_writer = std::exchange(_writer_waiter, nullopt);
    }
  }
  return r;
}

void
OportBuffer::register_reader_waiter_locked_(unsigned                cursor_idx,
                                            uint32_t                waiter_seq,
                                            coroutine_handle<>      h)
{
  Cursor& c = *_cursors[cursor_idx];
  c.waiter     = h;
  c.waiter_seq = waiter_seq;
}

void
OportBuffer::register_reader_multi_waiter_locked_(
    unsigned                    cursor_idx,
    uint32_t                    waiter_seq,
    shared_ptr<MultiReadWaiter> w)
{
  Cursor& c = *_cursors[cursor_idx];
  c.waiter.reset();          // single + multi are mutually exclusive
  c.multi      = std::move(w);
  c.waiter_seq = waiter_seq;
}

void
OportBuffer::deregister_reader_multi_waiter_(
    unsigned                          cursor_idx,
    const shared_ptr<MultiReadWaiter>& w)
{
  lock_guard<mutex> lk(_mu);
  Cursor& c = *_cursors[cursor_idx];
  if (c.multi == w) {
    c.multi.reset();
  }
}

optional<coroutine_handle<>>
OportBuffer::release_read_(unsigned cursor_idx, uint32_t n)
{
  if (n == 0) { return nullopt; }
  Cursor& c = *_cursors[cursor_idx];
  c.next_seq.fetch_add(n, memory_order_release);
  optional<coroutine_handle<>> wake;
  {
    lock_guard<mutex> lk(_mu);
    if (_writer_waiter) {
      uint32_t active_after = active_count_relaxed_();
      if (active_after < _hard_cap) {
        wake = std::exchange(_writer_waiter, nullopt);
      }
    }
  }
  return wake;
}

// ----- WriteAwaiter -----

bool
OportBuffer::WriteAwaiter::await_ready()
{
  vector<coroutine_handle<>>          wake_readers;
  vector<shared_ptr<MultiReadWaiter>> wake_multi;
  {
    lock_guard<mutex> lk(_buf->_mu);
    auto r = _buf->push_locked_(_value);
    if (r.success) {
      _delivered   = true;
      wake_readers = std::move(r.wake_readers);
      wake_multi   = std::move(r.wake_multi);
    } else if (r.closed) {
      _saw_closed = true;
    } else if (r.threshold) {
      _saw_threshold = true;
    } else {
      return false;          // backpressure: caller will suspend
    }
  }
  for (auto h : wake_readers) {
    _buf->session()->thread_pool()->schedule(h);
  }
  for (auto& m : wake_multi) {
    m->notify();
  }
  return true;
}

bool
OportBuffer::WriteAwaiter::await_suspend(coroutine_handle<> h)
{
  vector<coroutine_handle<>>          wake_readers;
  vector<shared_ptr<MultiReadWaiter>> wake_multi;
  bool suspend = true;
  {
    lock_guard<mutex> lk(_buf->_mu);
    auto r = _buf->push_locked_(_value);
    if (r.success) {
      _delivered   = true;
      wake_readers = std::move(r.wake_readers);
      wake_multi   = std::move(r.wake_multi);
      suspend      = false;
    } else if (r.closed) {
      _saw_closed = true;
      suspend     = false;
    } else if (r.threshold) {
      _saw_threshold = true;
      suspend        = false;
    } else {
      _buf->_writer_waiter = h;
    }
  }
  for (auto wh : wake_readers) {
    _buf->session()->thread_pool()->schedule(wh);
  }
  for (auto& m : wake_multi) {
    m->notify();
  }
  return suspend;
}

void
OportBuffer::WriteAwaiter::await_resume()
{
  if (_saw_closed) {
    throw runtime_error("OportBuffer: write to closed buffer");
  }
  if (_saw_threshold) {
    throw runtime_error(
        "OportBuffer: soft-error threshold exceeded (default-depth "
        "mode); pipeline structural imbalance");
  }
  if (_delivered) {
    return;
  }
  // Resumed by a consumer release/acquire freeing space, or by
  // close. Retry.
  vector<coroutine_handle<>>          wake_readers;
  vector<shared_ptr<MultiReadWaiter>> wake_multi;
  bool ok            = false;
  bool closed_       = false;
  bool threshold_    = false;
  {
    lock_guard<mutex> lk(_buf->_mu);
    auto r = _buf->push_locked_(_value);
    if (r.success) {
      ok           = true;
      wake_readers = std::move(r.wake_readers);
      wake_multi   = std::move(r.wake_multi);
    } else if (r.closed) {
      closed_ = true;
    } else if (r.threshold) {
      threshold_ = true;
    }
  }
  for (auto h : wake_readers) {
    _buf->session()->thread_pool()->schedule(h);
  }
  for (auto& m : wake_multi) {
    m->notify();
  }
  if (!ok && closed_) {
    throw runtime_error("OportBuffer: write to closed buffer");
  }
  if (!ok && threshold_) {
    throw runtime_error(
        "OportBuffer: soft-error threshold exceeded (default-depth "
        "mode); pipeline structural imbalance");
  }
  // Invariant: if we were woken without close or threshold, a slot
  // is free.
}

}
