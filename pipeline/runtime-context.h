#ifndef RUNTIME_CONTEXT_H
#define RUNTIME_CONTEXT_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/edge-reader.h"
#include "pipeline/oport-buffer.h"

#include <atomic>
#include <coroutine>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace vpipe {

// One RuntimeContext per Stage instance per running Pipeline. The
// PipelineRuntime constructs it during launch with pointers to the
// EdgeReader for each iport and the OportBuffer for each oport, and
// a back-reference to the runtime's atomic stop flag.
//
// Stage code uses it only inside process_one():
//   auto p = co_await ctx.read(0);
//   if (!p) { ctx.signal_done(); co_return; }
//   const auto& payload = static_cast<const FooPayload&>(*p);
//   co_await ctx.write(0, make_payload<FooPayload>(std::move(...)));
//
// Window-peek consumers can also use:
//   auto* p0 = co_await ctx.peek(0, 0);
//   auto* p1 = co_await ctx.peek(0, 1);
//   ... process window ...
//   ctx.release_read(0, 1);   // slide forward by 1
//
// Fanout: one OportBuffer per (producer, oport) receives the
// produced payload once. Cursors borrow via peek; the slowest cursor
// can acquire by moving the slot out. Compat read() does both
// transparently.
class RuntimeContext {
public:
  RuntimeContext(std::vector<EdgeReader*>  in_readers,
                 std::vector<OportBuffer*> out_bufs,
                 std::atomic<bool>*        stop)
    : _in_readers(std::move(in_readers))
    , _out_bufs(std::move(out_bufs))
    , _stop(stop)
  {}

  RuntimeContext(const RuntimeContext&)            = delete;
  RuntimeContext& operator=(const RuntimeContext&) = delete;

  unsigned num_iports() const noexcept
  {
    return static_cast<unsigned>(_in_readers.size());
  }

  // True iff positional iport `p` is wired to a producer. A declared
  // but DISCONNECTED (optional) iport reads as immediate EOS; this
  // distinguishes "unwired" from "wired producer that has finished".
  // Stages with optional inputs (e.g. hls-broadcast video vs audio)
  // branch on this at launch instead of a config flag.
  bool
  iport_connected(unsigned p) const noexcept
  {
    return p < _in_readers.size()
        && _in_readers[p] != nullptr
        && _in_readers[p]->parent() != nullptr;
  }

  // Non-blocking count of unread items on iport `p`. Returns 0 when
  // the cursor is fully drained (a subsequent read() would suspend
  // until the producer writes again) and a positive value when at
  // least that many beats are immediately readable. Useful for
  // stages that multiplex across multiple iports: a typical pattern
  // is to wait for one driver port (e.g. a chrono trigger) and then
  // drain the other port's backlog non-blockingly per tick. The
  // result is a relaxed snapshot — it may under-count progress but
  // never returns a stale value above the true backlog.
  std::uint32_t
  backlog(unsigned p) const noexcept
  {
    return _in_readers[p]->backlog();
  }

  // True iff iport `p` is drained AND closed -- a subsequent read()
  // returns null (EOS). Pairs with read_any(): a backlog-bounded drain
  // loop reads nothing on an empty-but-closed port, so the driver tests
  // eos() to end the stage instead of spinning (read_any treats a
  // closed port as perpetually "ready").
  bool
  eos(unsigned p) const
  {
    return _in_readers[p]->at_eos();
  }

  unsigned num_oports() const noexcept
  {
    return static_cast<unsigned>(_out_bufs.size());
  }

  // True iff the named oport has at least one consumer wired up at
  // pipeline launch time. The graph is frozen after launch so the
  // answer is stable for the life of the stage.
  bool
  has_consumers(unsigned out_port) const noexcept
  {
    return out_port < _out_bufs.size()
        && _out_bufs[out_port]
        && _out_bufs[out_port]->num_cursors() > 0;
  }

  // Compat one-shot read. Returns null on EOS.
  EdgeReader::ReadAwaiter
  read(unsigned in_port)
  {
    return _in_readers[in_port]->read();
  }

  // Window peek: borrow at cursor + offset. Suspends until written.
  // Returns nullptr on EOS at that offset. Does not advance.
  EdgeReader::PeekAwaiter
  peek(unsigned in_port, std::uint32_t offset = 0)
  {
    return _in_readers[in_port]->peek(offset);
  }

  // Suspend until ANY of `ports` is readable (a beat is available or
  // EOS), then resume; the caller drains whichever port(s) are ready
  // via backlog()/read(). For stages that must react to whichever of
  // several inputs arrives first -- e.g. realtime-vqa waking on a video
  // frame (iport0) OR a chrono tick (iport1), rather than blocking on
  // the tick and only draining frames once per tick. Wakes EXACTLY
  // once and is safe against a producer resuming the coroutine while
  // await_suspend is still registering the remaining ports (see
  // MultiReadWaiter's armed/fired handshake). await_ready short-
  // circuits when a port is already readable (no suspend).
  struct ReadAnyAwaiter {
    std::vector<EdgeReader*>         _readers;
    std::shared_ptr<MultiReadWaiter> _state;

    bool
    await_ready()
    {
      for (auto* r : _readers) {
        if (r->readable_now()) { return true; }
      }
      return false;
    }

    bool
    await_suspend(std::coroutine_handle<> h)
    {
      _state = std::make_shared<MultiReadWaiter>();
      _state->h = h;
      // Register on every not-yet-ready port. A port that is already
      // readable returns Ready (no registration) -> resume inline.
      bool any_ready = false;
      for (auto* r : _readers) {
        if (r->register_multi(_state) == EdgeReader::MultiReg::Ready) {
          any_ready = true;
        }
      }
      // Arm last, under the state lock: if a registered port already
      // fired (or one was Ready), resume inline instead of suspending,
      // and claim `fired` so a late notify() becomes a no-op.
      bool inline_resume = any_ready;
      {
        std::lock_guard<std::mutex> lk(_state->mu);
        if (_state->fired) { inline_resume = true; }
        _state->armed = true;
        if (inline_resume) { _state->fired = true; }
      }
      if (inline_resume) {
        for (auto* r : _readers) { r->deregister_multi(_state); }
        return false;
      }
      return true;
    }

    void
    await_resume()
    {
      // Clear our registrations on the ports that did not win the wake
      // (the winner already cleared its own slot).
      if (_state) {
        for (auto* r : _readers) { r->deregister_multi(_state); }
      }
    }
  };

  ReadAnyAwaiter
  read_any(std::vector<unsigned> ports)
  {
    std::vector<EdgeReader*> rs;
    rs.reserve(ports.size());
    for (unsigned p : ports) { rs.push_back(_in_readers[p]); }
    return ReadAnyAwaiter{ std::move(rs), {} };
  }

  // Move-out acquire at cursor. Suspends only for "not-yet-written"
  // or EOS. Returns null + records "not-slowest" if this cursor is
  // not the slowest (in fanout); the caller should fall back to
  // peek+clone+release.
  EdgeReader::AcquireAwaiter
  acquire(unsigned in_port)
  {
    return _in_readers[in_port]->acquire();
  }

  // Explicit advance. Non-blocking.
  void
  release_read(unsigned in_port, std::uint32_t n = 1)
  {
    _in_readers[in_port]->release_read(n);
  }

  // Push a payload to the oport buffer.
  OportBuffer::WriteAwaiter
  write(unsigned out_port, std::unique_ptr<BeatPayloadIntf> t)
  {
    return _out_bufs[out_port]->write(std::move(t));
  }

  // Non-coroutine push to an oport, for a stage that produces from a
  // thread it owns rather than from its process() coroutine. Returns
  // false iff the oport buffer is closed (teardown) -- the caller
  // should stop producing. See OportBuffer::push_sync.
  bool
  write_sync(unsigned out_port, std::unique_ptr<BeatPayloadIntf> t)
  {
    return _out_bufs[out_port]->push_sync(std::move(t));
  }

  // Stage signals it has produced its last output (or read past EOS
  // and has no more work). The driver loop closes outputs and exits
  // after the current process_one returns.
  void signal_done() noexcept { _done = true; }
  bool done()        const noexcept { return _done; }

  bool
  stop_requested() const noexcept
  {
    return _stop && _stop->load(std::memory_order_acquire);
  }

  // Close every downstream OportBuffer. Called by the driver after
  // the stage signals done or after an exception.
  void
  close_outputs()
  {
    for (auto* buf : _out_bufs) {
      if (buf) {
        buf->close();
      }
    }
  }

private:
  std::vector<EdgeReader*>  _in_readers;
  std::vector<OportBuffer*> _out_bufs;
  std::atomic<bool>*        _stop;
  bool                      _done = false;
};

}

#endif
