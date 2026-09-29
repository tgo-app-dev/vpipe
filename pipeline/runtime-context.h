#ifndef RUNTIME_CONTEXT_H
#define RUNTIME_CONTEXT_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/edge-reader.h"
#include "pipeline/oport-buffer.h"
#include "pipeline/stage-command.h"

#include <atomic>
#include <coroutine>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

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
//
// Plugin ABI: nothing here is inline, so a plugin compiles no knowledge of
// the members. The awaiters these methods return live in the caller's
// coroutine frame, and THEIR layouts are frozen (abi/ snapshot).
class RuntimeContext {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
public:
  RuntimeContext(std::vector<EdgeReader*>  in_readers,
                 std::vector<OportBuffer*> out_bufs,
                 std::atomic<bool>*        stop);
  ~RuntimeContext();

  RuntimeContext(const RuntimeContext&)            = delete;
  RuntimeContext& operator=(const RuntimeContext&) = delete;

  unsigned num_iports() const noexcept;

private:
  // Reading a positional iport a stage does NOT have -- `ctx.read(0)` on a
  // stage launched with no inputs at all (e.g. `vpipe --launch-stage
  // video-to-rgb`) -- used to index past the end of `_in_readers` and
  // dereference garbage, segfaulting the whole process. There is a correct
  // answer already in the model: the runtime gives a declared-but-UNWIRED
  // iport a parent-less EdgeReader that reads as permanent EOS, so an absent
  // port behaves the same way. Every stage's existing
  // `if (!beat) { signal_done(); }` path then ends the stage cleanly instead
  // of crashing.
  EdgeReader* reader_(unsigned p) const noexcept;

public:

  // True iff positional iport `p` is wired to a producer. A declared
  // but DISCONNECTED (optional) iport reads as immediate EOS; this
  // distinguishes "unwired" from "wired producer that has finished".
  // Stages with optional inputs (e.g. hls-broadcast video vs audio)
  // branch on this at launch instead of a config flag.
  bool iport_connected(unsigned p) const noexcept;

  // Non-blocking count of unread items on iport `p`. Returns 0 when
  // the cursor is fully drained (a subsequent read() would suspend
  // until the producer writes again) and a positive value when at
  // least that many beats are immediately readable. Useful for
  // stages that multiplex across multiple iports: a typical pattern
  // is to wait for one driver port (e.g. a chrono trigger) and then
  // drain the other port's backlog non-blockingly per tick. The
  // result is a relaxed snapshot — it may under-count progress but
  // never returns a stale value above the true backlog.
  std::uint32_t backlog(unsigned p) const noexcept;

  // True iff iport `p` is drained AND closed -- a subsequent read()
  // returns null (EOS). Pairs with read_any(): a backlog-bounded drain
  // loop reads nothing on an empty-but-closed port, so the driver tests
  // eos() to end the stage instead of spinning (read_any treats a
  // closed port as perpetually "ready").
  bool eos(unsigned p) const;

  unsigned num_oports() const noexcept;

  // True iff the named oport has at least one consumer wired up at
  // pipeline launch time. The graph is frozen after launch so the
  // answer is stable for the life of the stage.
  bool has_consumers(unsigned out_port) const noexcept;

  // Compat one-shot read. Returns null on EOS.
  EdgeReader::ReadAwaiter read(unsigned in_port);

  // Window peek: borrow at cursor + offset. Suspends until written.
  // Returns nullptr on EOS at that offset. Does not advance.
  EdgeReader::PeekAwaiter peek(unsigned in_port, std::uint32_t offset = 0);

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
  //
  // With `commands` set it ALSO wakes when a command reaches the stage's
  // inbox (or the inbox shuts at stop), so a stage that answers commands
  // can sleep on its inputs and its commands at once. With commands set
  // the port list may be empty -- then it waits on commands alone.
  //
  // The awaiter lives in the stage's coroutine frame, so its LAYOUT is
  // part of the plugin ABI; its logic is out of line.
  struct ReadAnyAwaiter {
    std::vector<EdgeReader*>         _readers;
    std::shared_ptr<MultiReadWaiter> _state;
    CommandInbox*                    _inbox = nullptr;

    bool await_ready();
    bool await_suspend(std::coroutine_handle<> h);
    void await_resume();
  };

  ReadAnyAwaiter read_any(std::vector<unsigned> ports,
                          bool commands = false);

  // ---- command channels (pipeline/stage-command.h) --------------------
  //
  // Present only when the stage's spec declares commands. Commands
  // arrive first in first out; each one taken is Active and the stage's
  // to answer (reply or fail) -- an unanswered one leaves its caller
  // waiting until the pipeline stops.

  bool has_commands() const noexcept;

  // The next queued command, or null. Never suspends.
  std::shared_ptr<StageCommand> try_command();

  // `auto cmd = co_await ctx.next_command();` -- the next command,
  // suspending until one arrives. Null means no more will: the pipeline
  // is stopping, or the stage has no command channels at all.
  NextCommandAwaiter next_command();

  // Cancel every command this stage took and has not seen closed, e.g.
  // on reaching the end of its stream with a read still open. The
  // inbox stays open for new commands.
  void cancel_open_commands(std::string_view why);

  // Runtime only: bind the stage's inbox, and shut it when the stage's
  // driver has finished so a caller is refused rather than left
  // waiting on a stage that will never read again.
  void attach_commands(CommandInbox* inbox) noexcept;
  void shut_commands(std::string_view why);

  // Move-out acquire at cursor. Suspends only for "not-yet-written"
  // or EOS. Returns null + records "not-slowest" if this cursor is
  // not the slowest (in fanout); the caller should fall back to
  // peek+clone+release.
  EdgeReader::AcquireAwaiter acquire(unsigned in_port);

  // Explicit advance. Non-blocking.
  void release_read(unsigned in_port, std::uint32_t n = 1);

  // Push a payload to the oport buffer.
  OportBuffer::WriteAwaiter write(unsigned out_port,
                                  std::unique_ptr<BeatPayloadIntf> t);

  // Non-coroutine push to an oport, for a stage that produces from a
  // thread it owns rather than from its process() coroutine. Returns
  // false iff the oport buffer is closed (teardown) -- the caller
  // should stop producing. See OportBuffer::push_sync.
  bool write_sync(unsigned out_port, std::unique_ptr<BeatPayloadIntf> t);

  // Stage signals it has produced its last output (or read past EOS
  // and has no more work). The driver loop closes outputs and exits
  // after the current process_one returns.
  void signal_done() noexcept;
  bool done() const noexcept;

  bool stop_requested() const noexcept;

  // Close every downstream OportBuffer. Called by the driver after
  // the stage signals done or after an exception.
  void close_outputs();

private:
  // Every method is out of line, so this layout is the host's alone: a
  // plugin's stage holds a RuntimeContext& and never reaches a member.
  std::vector<EdgeReader*>  _in_readers;
  std::vector<OportBuffer*> _out_bufs;
  // Parent-less => permanently at EOS. Stands in for any iport index the
  // stage does not actually have (see reader_). `mutable` so the const
  // accessors above can hand out a usable pointer.
  mutable EdgeReader        _eos_reader{nullptr, 0};
  std::atomic<bool>*        _stop;
  bool                      _done = false;
  CommandInbox*             _inbox = nullptr;
};

}

VPIPE_API_END

#endif
