#include "pipeline/edge-reader.h"
#include "common/thread-pool.h"
#include "interfaces/session-context-intf.h"

#include <mutex>
#include <utility>

using namespace std;

namespace vpipe {

EdgeReader::EdgeReader(OportBuffer* parent, unsigned cursor_idx)
  : _parent(parent)
  , _cursor_idx(cursor_idx)
{
}

uint32_t
EdgeReader::dropped() const
{
  return _parent->dropped_for_cursor(_cursor_idx);
}

uint32_t
EdgeReader::backlog() const
{
  // Active-from-this-cursor distance: wp - cursor.next_seq.
  // Relaxed loads are safe -- under-estimating progress yields a
  // larger backlog, never a negative or wrapped value.
  uint32_t wp = _parent->_wp.load(memory_order_relaxed);
  uint32_t cs = _parent->_cursors[_cursor_idx]->next_seq.load(
      memory_order_relaxed);
  uint32_t d = wp - cs;
  // Out-of-bound distance (huge unsigned) is impossible with the
  // (wp - rp) <= kMaxCapacity invariant; if seen, clamp to 0.
  if (d > OportPolicy::kMaxCapacity) {
    return 0;
  }
  return d;
}

void
EdgeReader::release_read(uint32_t n)
{
  if (n == 0) { return; }
  auto wake = _parent->release_read_(_cursor_idx, n);
  if (wake) {
    _parent->session()->thread_pool()->schedule(*wake);
  }
}

// ----- multi-port wait (RuntimeContext::read_any) -----------------

bool
EdgeReader::readable_now() const
{
  // A beat to read, or closed (EOS) -- either way read() won't suspend.
  return backlog() > 0 || _parent->closed();
}

bool
EdgeReader::at_eos() const
{
  return backlog() == 0 && _parent->closed();
}

EdgeReader::MultiReg
EdgeReader::register_multi(const shared_ptr<MultiReadWaiter>& w)
{
  OportBuffer* buf = _parent;
  lock_guard<mutex> lk(buf->_mu);
  auto r = buf->peek_locked_(_cursor_idx, 0);
  if (r.p || r.saw_eos) {
    return MultiReg::Ready;
  }
  if (!w->pool) {
    w->pool = buf->session()->thread_pool();
  }
  uint32_t cs = buf->_cursors[_cursor_idx]->next_seq.load(
      memory_order_relaxed);
  buf->register_reader_multi_waiter_locked_(_cursor_idx, cs + 1, w);
  return MultiReg::Registered;
}

void
EdgeReader::deregister_multi(const shared_ptr<MultiReadWaiter>& w)
{
  _parent->deregister_reader_multi_waiter_(_cursor_idx, w);
}

// ----- PeekAwaiter ------------------------------------------------

bool
EdgeReader::PeekAwaiter::await_ready()
{
  OportBuffer* buf = _reader->_parent;
  lock_guard<mutex> lk(buf->_mu);
  auto r = buf->peek_locked_(_reader->_cursor_idx, _offset);
  if (r.p) {
    _stash = r.p;
    return true;
  }
  if (r.saw_eos) {
    _is_eos = true;
    return true;
  }
  return false;
}

bool
EdgeReader::PeekAwaiter::await_suspend(coroutine_handle<> h)
{
  OportBuffer* buf = _reader->_parent;
  lock_guard<mutex> lk(buf->_mu);
  auto r = buf->peek_locked_(_reader->_cursor_idx, _offset);
  if (r.p) {
    _stash = r.p;
    return false;
  }
  if (r.saw_eos) {
    _is_eos = true;
    return false;
  }
  uint32_t cs = buf->_cursors[_reader->_cursor_idx]->next_seq.load(
      memory_order_relaxed);
  buf->register_reader_waiter_locked_(_reader->_cursor_idx,
                                      cs + _offset + 1, h);
  return true;
}

const BeatPayloadIntf*
EdgeReader::PeekAwaiter::await_resume()
{
  if (_is_eos) {
    return nullptr;
  }
  if (_stash) {
    return _stash;
  }
  OportBuffer* buf = _reader->_parent;
  lock_guard<mutex> lk(buf->_mu);
  auto r = buf->peek_locked_(_reader->_cursor_idx, _offset);
  if (r.p) {
    _stash = r.p;
    return _stash;
  }
  if (r.saw_eos) {
    _is_eos = true;
  }
  return nullptr;
}

// ----- AcquireAwaiter ---------------------------------------------

bool
EdgeReader::AcquireAwaiter::await_ready()
{
  OportBuffer* buf = _reader->_parent;
  optional<coroutine_handle<>> wake_writer;
  {
    lock_guard<mutex> lk(buf->_mu);
    auto r = buf->acquire_locked_(_reader->_cursor_idx);
    if (r.p) {
      _stash      = std::move(r.p);
      wake_writer = r.wake_writer;
    } else if (r.saw_eos) {
      _is_eos = true;
    } else if (r.not_slowest) {
      _not_slowest = true;
    } else {
      return false;
    }
  }
  if (wake_writer) {
    buf->session()->thread_pool()->schedule(*wake_writer);
  }
  return true;
}

bool
EdgeReader::AcquireAwaiter::await_suspend(coroutine_handle<> h)
{
  OportBuffer* buf = _reader->_parent;
  optional<coroutine_handle<>> wake_writer;
  bool suspend = true;
  {
    lock_guard<mutex> lk(buf->_mu);
    auto r = buf->acquire_locked_(_reader->_cursor_idx);
    if (r.p) {
      _stash      = std::move(r.p);
      wake_writer = r.wake_writer;
      suspend     = false;
    } else if (r.saw_eos) {
      _is_eos = true;
      suspend = false;
    } else if (r.not_slowest) {
      _not_slowest = true;
      suspend      = false;
    } else {
      uint32_t cs = buf->_cursors[_reader->_cursor_idx]
                        ->next_seq.load(memory_order_relaxed);
      buf->register_reader_waiter_locked_(_reader->_cursor_idx,
                                          cs + 1, h);
    }
  }
  if (wake_writer) {
    buf->session()->thread_pool()->schedule(*wake_writer);
  }
  return suspend;
}

unique_ptr<BeatPayloadIntf>
EdgeReader::AcquireAwaiter::await_resume()
{
  if (_is_eos || _not_slowest) {
    return nullptr;
  }
  if (_stash) {
    return std::move(_stash);
  }
  // Resumed by a producer push or close. Retry.
  OportBuffer* buf = _reader->_parent;
  optional<coroutine_handle<>> wake_writer;
  unique_ptr<BeatPayloadIntf>  result;
  {
    lock_guard<mutex> lk(buf->_mu);
    auto r = buf->acquire_locked_(_reader->_cursor_idx);
    if (r.p) {
      result      = std::move(r.p);
      wake_writer = r.wake_writer;
    } else if (r.saw_eos) {
      _is_eos = true;
    } else if (r.not_slowest) {
      _not_slowest = true;
    }
  }
  if (wake_writer) {
    buf->session()->thread_pool()->schedule(*wake_writer);
  }
  return result;
}

// ----- ReadAwaiter (compat) ---------------------------------------
//
// Semantics today:
//   * Try acquire (move-out, slowest-only).
//   * On not-slowest: peek + clone + release.
//   * On EOS: return null.
//
// For the v1 of this refactor we keep the implementation simple:
// just delegate to acquire_locked_ + (optional) peek+clone+release
// all under one acquisition of _mu in await_ready and await_suspend.

bool
EdgeReader::ReadAwaiter::await_ready()
{
  OportBuffer* buf = _reader->_parent;
  optional<coroutine_handle<>> wake_writer;
  {
    lock_guard<mutex> lk(buf->_mu);
    auto r = buf->acquire_locked_(_reader->_cursor_idx);
    if (r.p) {
      _stash      = std::move(r.p);
      wake_writer = r.wake_writer;
    } else if (r.saw_eos) {
      _is_eos = true;
    } else if (r.not_slowest) {
      // Fallback path: peek the slot, clone the payload, release.
      auto pk = buf->peek_locked_(_reader->_cursor_idx, 0);
      if (pk.p) {
        _stash = pk.p->clone();
        buf->_cursors[_reader->_cursor_idx]
            ->next_seq.fetch_add(1, memory_order_release);
        if (buf->_writer_waiter) {
          uint32_t active_after =
              buf->active_count_relaxed_();
          if (active_after < buf->_hard_cap) {
            wake_writer =
                std::exchange(buf->_writer_waiter, nullopt);
          }
        }
      } else if (pk.saw_eos) {
        _is_eos = true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  if (wake_writer) {
    buf->session()->thread_pool()->schedule(*wake_writer);
  }
  return true;
}

bool
EdgeReader::ReadAwaiter::await_suspend(coroutine_handle<> h)
{
  OportBuffer* buf = _reader->_parent;
  optional<coroutine_handle<>> wake_writer;
  bool suspend = true;
  {
    lock_guard<mutex> lk(buf->_mu);
    auto r = buf->acquire_locked_(_reader->_cursor_idx);
    if (r.p) {
      _stash      = std::move(r.p);
      wake_writer = r.wake_writer;
      suspend     = false;
    } else if (r.saw_eos) {
      _is_eos = true;
      suspend = false;
    } else if (r.not_slowest) {
      auto pk = buf->peek_locked_(_reader->_cursor_idx, 0);
      if (pk.p) {
        _stash = pk.p->clone();
        buf->_cursors[_reader->_cursor_idx]
            ->next_seq.fetch_add(1, memory_order_release);
        if (buf->_writer_waiter) {
          uint32_t active_after =
              buf->active_count_relaxed_();
          if (active_after < buf->_hard_cap) {
            wake_writer =
                std::exchange(buf->_writer_waiter, nullopt);
          }
        }
        suspend = false;
      } else if (pk.saw_eos) {
        _is_eos = true;
        suspend = false;
      } else {
        uint32_t cs = buf->_cursors[_reader->_cursor_idx]
                          ->next_seq.load(memory_order_relaxed);
        buf->register_reader_waiter_locked_(_reader->_cursor_idx,
                                            cs + 1, h);
      }
    } else {
      uint32_t cs = buf->_cursors[_reader->_cursor_idx]
                        ->next_seq.load(memory_order_relaxed);
      buf->register_reader_waiter_locked_(_reader->_cursor_idx,
                                          cs + 1, h);
    }
  }
  if (wake_writer) {
    buf->session()->thread_pool()->schedule(*wake_writer);
  }
  return suspend;
}

unique_ptr<BeatPayloadIntf>
EdgeReader::ReadAwaiter::await_resume()
{
  if (_is_eos) {
    return nullptr;
  }
  if (_stash) {
    return std::move(_stash);
  }
  // Resumed; retry the read path.
  OportBuffer* buf = _reader->_parent;
  optional<coroutine_handle<>> wake_writer;
  unique_ptr<BeatPayloadIntf>  result;
  {
    lock_guard<mutex> lk(buf->_mu);
    auto r = buf->acquire_locked_(_reader->_cursor_idx);
    if (r.p) {
      result      = std::move(r.p);
      wake_writer = r.wake_writer;
    } else if (r.saw_eos) {
      _is_eos = true;
    } else if (r.not_slowest) {
      auto pk = buf->peek_locked_(_reader->_cursor_idx, 0);
      if (pk.p) {
        result = pk.p->clone();
        buf->_cursors[_reader->_cursor_idx]
            ->next_seq.fetch_add(1, memory_order_release);
        if (buf->_writer_waiter) {
          uint32_t active_after =
              buf->active_count_relaxed_();
          if (active_after < buf->_hard_cap) {
            wake_writer =
                std::exchange(buf->_writer_waiter, nullopt);
          }
        }
      } else if (pk.saw_eos) {
        _is_eos = true;
      }
    }
  }
  if (wake_writer) {
    buf->session()->thread_pool()->schedule(*wake_writer);
  }
  return result;
}

}
