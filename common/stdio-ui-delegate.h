#ifndef STDIO_UI_DELEGATE_H
#define STDIO_UI_DELEGATE_H

#include "interfaces/ui-delegate-intf.h"
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
};

}

#endif
