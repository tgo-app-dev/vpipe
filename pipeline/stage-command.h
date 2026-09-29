#ifndef VPIPE_PIPELINE_STAGE_COMMAND_H
#define VPIPE_PIPELINE_STAGE_COMMAND_H

// The STAGE side of command channels, and the runtime plumbing between
// a caller and a running stage. The caller's side is the public
// vpipe/stage-command.h; the declaration a stage makes is CommandSpec in
// pipeline/stage-spec.h; the whole contract is docs/STAGE-COMMANDS.md.
//
// A command is one shared object, StageCommand, with two faces. The
// caller holds it through a CommandHandle and the stage through a
// shared_ptr taken from its inbox; whichever lets go last frees it,
// together with every buffer owner it holds.
//
// HOW A STAGE SERVES ONE. Commands queue in the stage's CommandInbox,
// first in first out, from launch until stop. The stage's coroutine
// takes them -- `ctx.try_command()` between beats, or suspends on
// `co_await ctx.next_command()` / `co_await ctx.read_any(ports, true)`
// to wake on a command as readily as on a beat -- and answers each
// with reply() or fail():
//
//   auto cmd = ctx.try_command();
//   if (cmd && cmd->name() == "read") {
//     cmd->reply(result, { view_of_my_bytes });
//     co_await cmd->until_closed();   // HOLD: nothing downstream moves
//     ...release or reuse the bytes, emit the beat...
//   }
//
// Waiting in until_closed() SUSPENDS the coroutine; it does not block a
// worker thread. Everything downstream starves and everything upstream
// backs up, which is exactly what keeps the exposed bytes valid, and a
// stop cancels the command so the wait always ends.
//
// StageCommand and CommandInbox are created by libvpipe, reached only
// through pointers and touched only through out-of-line methods, so
// their members can change without moving anything a plugin compiled.
// The two awaiters live in a plugin's coroutine frame and are part of
// the plugin ABI.

#include "vpipe/stage-command.h"

#include "common/flex-data.h"
#include "pipeline/stage-spec.h"

#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

class Stage;
class ThreadPool;
struct MultiReadWaiter;

class StageCommand : public std::enable_shared_from_this<StageCommand> {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
public:
  // ---- what was asked ------------------------------------------------
  std::string_view   name() const noexcept;
  // The declaration it was validated against; null for a command the
  // host refused before it had one.
  const CommandSpec* spec() const noexcept;
  // 1-based, per stage; 0 for a refused command.
  std::uint64_t      id() const noexcept;
  // The arguments, validated: an object holding every declared key --
  // the caller's value, or the declared default -- and nothing else.
  const FlexData&    args() const noexcept;
  // The caller's buffers, validated against spec()->in: every name is
  // declared, every required one is present, every layout matches, and
  // a "bytes" buffer carries its type as Bytes and the spec's format.
  // Read-only: they are the caller's memory.
  const std::vector<DataBuffer>& inputs() const noexcept;
  const DataBuffer*  input(std::string_view name) const noexcept;

  // ---- the stage's answer ----------------------------------------------
  // Answer once. `out` must match spec()->out -- the host checks it as
  // it checked the inputs, and a mismatch FAILS the command (naming the
  // buffer) rather than handing the caller a view it cannot trust.
  // Returns false when the reply was not delivered: already answered,
  // refused for that mismatch, or the caller closed / the runtime
  // cancelled first. After a false return nothing is exposed, so there
  // is nothing to hold for.
  bool reply(FlexData result = FlexData::make_object(),
             std::vector<DataBuffer> out = {});

  // Refuse it. `message` reaches the caller as error().
  void fail(std::string message);

  // True once the caller closed it or the runtime cancelled it. From
  // then on the stage may release or reuse anything it exposed.
  bool closed() const noexcept;
  // True when it was the RUNTIME that ended it (a stop, or the stage's
  // own end of stream) -- the stage should then drop rather than emit
  // what the command was about.
  bool cancelled() const noexcept;

  // `co_await cmd->until_closed()`: suspend until closed(). Returns at
  // once when it already is.
  struct CloseAwaiter {
    std::shared_ptr<StageCommand> _cmd;
    bool await_ready() const noexcept;
    bool await_suspend(std::coroutine_handle<> h);
    void await_resume() const noexcept {}
  };
  CloseAwaiter until_closed();

  // ---- the caller's side (CommandHandle forwards here) -----------------
  CommandState state() const noexcept;
  CommandState wait(int timeout_ms) const;
  FlexData     result() const;
  std::string  error() const;
  std::vector<DataBuffer> outputs() const;
  void         close();

  // ---- construction (the host) -----------------------------------------
  StageCommand(const CommandSpec* spec, FlexData args,
               std::vector<DataBuffer> in);
  // A command refused before it reached a stage.
  static std::shared_ptr<StageCommand> refused(std::string error);

  StageCommand(const StageCommand&)            = delete;
  StageCommand& operator=(const StageCommand&) = delete;

private:
  friend class CommandInbox;

  CommandState state_locked_() const noexcept;
  void set_queued_(std::uint64_t id, ThreadPool* pool) noexcept;
  void activate_() noexcept;
  // The runtime's end for a command it will not see answered or closed:
  // the pipeline stopped, or the stage ended its stream.
  void cancel_(std::string_view why);
  // Wake a stage suspended in until_closed(). Called with _mu NOT held.
  void wake_closer_(std::coroutine_handle<> h) noexcept;

  const CommandSpec*       _spec = nullptr;
  std::uint64_t            _id   = 0;
  FlexData                 _args;
  std::vector<DataBuffer>  _in;

  mutable std::mutex              _mu;
  mutable std::condition_variable _cv;
  // The OUTCOME: Pending -> Active -> Replied | Failed. Once answered it
  // never changes -- a later stop cannot turn a delivered reply into a
  // cancellation, which is what a caller's wait would otherwise race.
  CommandState             _state = CommandState::Pending;
  bool                     _closed        = false;   // either side ended it
  bool                     _caller_closed = false;
  bool                     _cancelled     = false;   // the runtime did
  FlexData                 _result;
  std::string              _error;
  std::vector<DataBuffer>  _out;
  ThreadPool*              _pool = nullptr;
  std::coroutine_handle<>  _closer{};
};

// A stage's queue of commands, alive from launch to stop. The runtime
// creates one for each stage whose spec declares commands; a caller
// reaches it through Stage::command_inbox().
class CommandInbox {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
public:
  CommandInbox(std::string stage_id, ThreadPool* pool,
               std::size_t capacity = 64);
  ~CommandInbox();

  CommandInbox(const CommandInbox&)            = delete;
  CommandInbox& operator=(const CommandInbox&) = delete;

  // ---- the caller ------------------------------------------------------
  // Queue `cmd` and wake the stage. False + *err when the inbox is shut
  // or holds `capacity` unanswered commands already.
  bool post(const std::shared_ptr<StageCommand>& cmd, std::string* err);

  // ---- the stage -------------------------------------------------------
  // The oldest queued command, now Active; null when there is none.
  std::shared_ptr<StageCommand> take();
  // A command is queued, or the inbox is shut (nothing more will come).
  bool ready() const;
  bool shut() const;

  // Multi-waiter registration, for read_any: false when ready() already
  // (nothing registered -- resume inline), true when `w` will be
  // notified by the next post() or by shutdown().
  bool register_waiter(const std::shared_ptr<MultiReadWaiter>& w);
  void deregister_waiter(const std::shared_ptr<MultiReadWaiter>& w);

  // ---- the runtime -------------------------------------------------------
  // End the inbox: every queued command and every command the stage
  // took and has not seen closed is CANCELLED with `why`, and every
  // waiter wakes. Idempotent.
  void shutdown(std::string_view why);

  // Cancel what the stage took and still holds open, without shutting
  // the inbox -- for a stage that has reached the end of its stream.
  void cancel_open(std::string_view why);

private:
  std::string                                  _stage_id;
  ThreadPool*                                  _pool;
  std::size_t                                  _capacity;
  mutable std::mutex                           _mu;
  std::deque<std::shared_ptr<StageCommand>>    _queue;
  std::vector<std::weak_ptr<StageCommand>>     _taken;
  std::vector<std::shared_ptr<MultiReadWaiter>> _waiters;
  std::uint64_t                                _next_id = 1;
  bool                                         _shut = false;
  std::string                                  _shut_why;
};

// `co_await ctx.next_command()`: the next command, suspending until one
// arrives; null once the inbox shuts (the pipeline is stopping).
struct NextCommandAwaiter {
  CommandInbox*                    _inbox = nullptr;
  std::shared_ptr<StageCommand>    _got;
  std::shared_ptr<MultiReadWaiter> _state;

  bool await_ready();
  bool await_suspend(std::coroutine_handle<> h);
  std::shared_ptr<StageCommand> await_resume();
};

// ---- the host's entry points --------------------------------------------

// Send stage `s` the command `name`. Never throws and never returns
// null: validation failures, an unknown command and a stage that is not
// running all come back as a refused (Failed) command whose error()
// names the stage and the cause.
std::shared_ptr<StageCommand>
open_stage_command(Stage& s, std::string_view name, FlexData args,
                   std::vector<DataBuffer> in);

// The commands `spec` declares, as the JSON-able array
// StageHandle::commands_json() returns.
FlexData describe_commands(const StageSpec& spec);

// Check `bufs` against `spec` for one direction, normalising in place
// (a "bytes" buffer's type becomes Bytes and it takes the spec's
// format). `outbound` selects the out-side rules (writable must match
// the spec). False + *err naming the buffer on the first mismatch.
bool check_buffers(std::span<const BufferSpec> spec,
                   std::vector<DataBuffer>& bufs, bool outbound,
                   std::string* err);

// Check `args` (null or an object) against the declared arguments;
// normalises null to {}. False + *err on an unknown key, a missing
// required one, or a value of the wrong JSON type.
bool check_args(std::span<const ConfigKey> spec, FlexData& args,
                std::string* err);

// Does `shape` match the pattern `pattern` (the BufferSpec::shape
// grammar)? `symbols` carries NAME bindings across calls; it is only
// extended when the match succeeds.
struct ShapeSymbols {
  std::vector<std::pair<std::string, std::int64_t>> bound;
};
bool match_shape(std::string_view pattern,
                 const std::vector<std::int64_t>& shape,
                 ShapeSymbols* symbols);

}

VPIPE_API_END

#endif
