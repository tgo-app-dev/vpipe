#ifndef EDGE_READER_H
#define EDGE_READER_H

#include "common/beat-payload-intf.h"
#include "pipeline/oport-buffer.h"

#include <coroutine>
#include <cstdint>
#include <memory>

namespace vpipe {

// Per-consumer-iport cursor handle into an OportBuffer. One
// EdgeReader is constructed at pipeline launch for every (producer,
// oport, consumer, iport) edge; its cursor index inside the parent
// OportBuffer is allocated by OportBuffer::attach_cursor().
//
// Three reader operations:
//   * read() -- compat awaiter. Returns
//     unique_ptr<BeatPayloadIntf> (null on EOS). Internally either
//     acquires (move-out) when caller is slowest, or peeks + clones
//     + releases as fallback.
//   * peek(offset) -- borrow at rp+offset; suspends until written.
//   * acquire() -- move-out at rp; suspends only for "not yet
//     written" or EOS. Returns null + records "not-slowest" when
//     this cursor is not the slowest (caller falls back to peek).
//   * release_read(n) -- non-blocking advance.
class EdgeReader {
public:
  EdgeReader(OportBuffer* parent, unsigned cursor_idx);

  EdgeReader(const EdgeReader&)            = delete;
  EdgeReader& operator=(const EdgeReader&) = delete;

  // ---- compat one-shot read ---------------------------------------
  struct ReadAwaiter {
    EdgeReader*                       _reader;
    std::unique_ptr<BeatPayloadIntf>  _stash;
    bool                              _is_eos = false;

    bool await_ready();
    bool await_suspend(std::coroutine_handle<> h);
    std::unique_ptr<BeatPayloadIntf> await_resume();
  };
  ReadAwaiter
  read()
  {
    return ReadAwaiter{this, {}, false};
  }

  // ---- window peek ------------------------------------------------
  // Suspends until rp + offset is written or EOS is reached.
  // Returns a pointer borrowed from the buffer slot (nullptr on
  // EOS). The borrow is valid until release_read advances past
  // offset or the buffer is destroyed.
  struct PeekAwaiter {
    EdgeReader*            _reader;
    std::uint32_t          _offset;
    const BeatPayloadIntf* _stash   = nullptr;
    bool                   _is_eos  = false;

    bool await_ready();
    bool await_suspend(std::coroutine_handle<> h);
    const BeatPayloadIntf* await_resume();
  };
  PeekAwaiter
  peek(std::uint32_t offset = 0)
  {
    return PeekAwaiter{this, offset, nullptr, false};
  }

  // ---- move-out acquire -------------------------------------------
  // Returns the moved unique_ptr on success (advances rp by 1).
  // Returns null + sets _not_slowest if this cursor is not the
  // slowest; caller should fall back to peek+clone+release.
  // Returns null + sets _is_eos on EOS.
  struct AcquireAwaiter {
    EdgeReader*                       _reader;
    std::unique_ptr<BeatPayloadIntf>  _stash;
    bool                              _is_eos       = false;
    bool                              _not_slowest  = false;

    bool await_ready();
    bool await_suspend(std::coroutine_handle<> h);
    std::unique_ptr<BeatPayloadIntf> await_resume();
    bool not_slowest() const noexcept { return _not_slowest; }
  };
  AcquireAwaiter
  acquire()
  {
    return AcquireAwaiter{this, {}, false, false};
  }

  // Non-blocking advance. Wakes the writer if backpressured.
  void release_read(std::uint32_t n = 1);

  // ---- multi-port wait (RuntimeContext::read_any) -----------------
  // Non-blocking: true iff a subsequent read() would NOT suspend --
  // either a beat is available or the buffer is closed at this cursor
  // (EOS). Relaxed snapshot; the authoritative check is under-lock in
  // register_multi.
  bool readable_now() const;

  // Non-blocking: true iff this cursor is drained AND its producer has
  // closed (a subsequent read() returns null / EOS). Distinguishes the
  // closed-and-empty case from "data still buffered" so a read_any
  // driver can end instead of spinning on the always-"readable" EOS.
  bool at_eos() const;

  enum class MultiReg { Ready, Registered };
  // Under _mu: if this cursor is already readable (beat or EOS) return
  // Ready without registering; otherwise register `w` as the cursor's
  // pending multi-waiter and return Registered. Sets w->pool on first
  // registration.
  MultiReg register_multi(const std::shared_ptr<MultiReadWaiter>& w);

  // Drop `w` from this cursor's pending multi-waiter slot iff still set.
  void deregister_multi(const std::shared_ptr<MultiReadWaiter>& w);

  OportBuffer* parent() const noexcept { return _parent; }
  unsigned     cursor_idx() const noexcept { return _cursor_idx; }

  std::uint32_t dropped() const;
  std::uint32_t backlog() const;

private:
  OportBuffer* _parent;
  unsigned     _cursor_idx;
};

}

#endif
