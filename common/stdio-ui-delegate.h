#ifndef STDIO_UI_DELEGATE_H
#define STDIO_UI_DELEGATE_H

#include "interfaces/ui-delegate-intf.h"
#include <atomic>
#include <chrono>
#include <mutex>

namespace vpipe {

// Default UiDelegateIntf: error/warn -> stderr, info -> stdout (each
// emitted as a `[LEVEL] message` line, matching StdoutLogDelegate's
// framing so console output is unchanged from when these levels rode
// the log delegate), getline -> stdin.
//
// getline() reproduces the cooperative-cancellation behaviour the
// text-input stage used to implement inline: it masks SIGINT/SIGTERM/
// SIGHUP on the calling worker for the duration of the read (so the
// signal is routed to the process's main thread, where a Python driver
// can act on it) and poll()s stdin on a short timeout, re-checking the
// supplied cancel predicate between polls.
//
// Thread safety: all four methods are safe to call concurrently. An
// internal mutex serialises writes so lines never tear; getline holds
// it only for the prompt write, not for the (blocking) read.
class StdioUiDelegate final : public UiDelegateIntf {
public:
  void error(const VpipeFormat&) override;
  void warn(const VpipeFormat&) override;
  void info(const VpipeFormat&) override;

  UiInputStatus
  getline(const VpipeFormat& prompt, std::string& out,
          const std::function<bool()>& should_cancel) override;

  // Password variant: disables terminal echo for the read on a real
  // tty (nothing is shown as the user types), otherwise identical to
  // getline().
  UiInputStatus
  getpasswd(const VpipeFormat& prompt, std::string& out,
            const std::function<bool()>& should_cancel) override;

  // Returns a stream that writes chunks verbatim to stdout (flushed)
  // and emits a terminating newline on end(). Shares the delegate's
  // I/O mutex so streamed text never tears against emit_(); the
  // delegate must outlive every stream it hands out.
  std::unique_ptr<UiTextStream> open_text_stream() override;

  // ---- Ctrl-C (SIGINT) policy --------------------------------------
  //
  // Two-stroke. The FIRST Ctrl-C interrupts whatever work stages
  // registered interrupt handlers for (see
  // UiDelegateIntf::register_interrupt_handler) -- typically a running
  // text generation -- and the process keeps going. A SECOND Ctrl-C
  // struck within kDoubleTapMs of the first means "quit", which is
  // what a lone SIGINT has always meant: drain the pipelines and
  // terminate. A first Ctrl-C that NO handler consumes also means
  // quit, so a pipeline with nothing interruptible still stops on one
  // keystroke rather than swallowing it.
  //
  // Split in two because a signal handler may not do the work:
  //   note_sigint()  -- call from the process's SIGINT handler. Bumps
  //                     an atomic and nothing else, so it is
  //                     async-signal-safe.
  //   poll_sigint()  -- call from the app's ordinary wait loop. Applies
  //                     the policy (which may run handlers and write to
  //                     the console) and returns true when the caller
  //                     should stop the pipelines and exit.
  // poll_sigint() keeps unsynchronized state and expects a single
  // caller -- the app's control thread; note_sigint() may be called
  // from anywhere.
  static constexpr int kDoubleTapMs = 2000;
  void note_sigint() noexcept;
  bool poll_sigint();

private:
  // Format `[tag] msg\n` and write to stderr (to_err) or stdout.
  void emit_(const char* tag, bool to_err, const VpipeFormat&);

  // Shared body of getline()/getpasswd(): prints the prompt, masks
  // SIGINT/SIGTERM/SIGHUP, poll()s stdin with cancel re-checks, then
  // reads one line. When `mask` is set, terminal echo is disabled for
  // the read on a real tty (and a newline is emitted afterward, since
  // the user's Enter wasn't echoed).
  UiInputStatus
  read_line_(const VpipeFormat& prompt, std::string& out,
             const std::function<bool()>& should_cancel, bool mask);

  std::mutex _io_mu;

  // SIGINT bookkeeping. `_sigint_count` is the only field the signal
  // handler touches; the rest belong to poll_sigint()'s caller.
  std::atomic<unsigned> _sigint_count{0};
  unsigned              _sigint_seen = 0;      // last count acted on
  bool                  _sigint_armed = false; // a 2nd tap now quits
  std::chrono::steady_clock::time_point _sigint_at{};
};

}

#endif
