#ifndef STDIO_UI_DELEGATE_H
#define STDIO_UI_DELEGATE_H

#include "interfaces/ui-delegate-intf.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
  ~StdioUiDelegate() override;

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

  // ---- progress footer ---------------------------------------------
  //
  // Live reports are drawn as a block of bars pinned to the bottom of
  // the terminal, one row each, while everything else scrolls above
  // them (see common/console-writer.h for how that is arbitrated).
  //
  // A repaint THREAD rather than painting from update(): a download's
  // progress callback fires hundreds of times a second and a denoise
  // step once every few seconds, so redrawing per update would either
  // thrash the terminal or need every producer to throttle. The thread
  // wakes at kRepaintMs, compares progress_version(), and repaints only
  // when something actually moved. It starts on the first report and
  // exits once the last one closes, so a session that never reports
  // progress never spawns it.
  static constexpr int kRepaintMs = 100;

  // OFF A TTY there is nothing to pin a footer to, but silence is not
  // the right answer either: a redirected stdout -- a log file, or an
  // agent reading our pipe -- then shows nothing whatsoever between the
  // line before a long generation and the line after it, which reads as
  // a hang. The same reports are emitted there as plain lines instead,
  // one per ~10% of a report's span, so the destination gains bounded,
  // greppable evidence of progress rather than a repaint stream.
  static constexpr int kMilestonePct = 10;

protected:
  void on_progress_opened() override;

private:
  void start_progress_thread_();
  void stop_progress_thread_();

  // Format `[tag] msg\n` and write to stderr (to_err) or stdout.
  void emit_(const char* tag, bool to_err, const VpipeFormat&);

  // Off a tty, what has already been said about one report: the last
  // ~10% milestone announced, and the last state seen -- which is all
  // there is left to report with once the id disappears, since a
  // closed report is gone from the registry entirely.
  struct Announced {
    int           bucket = -1;
    std::string   desc;
    std::uint64_t done  = 0;
    std::uint64_t total = 0;
  };

  // The non-tty half of the progress thread: emit a line for every live
  // report that has crossed into a new ~10% milestone since the last
  // call, and a closing line for every id that has gone away without
  // reaching 100%. `seen` is the per-id state above, pruned as reports
  // close.
  void emit_milestones_(const std::vector<UiProgressRegistry::Item>& items,
                        std::map<std::uint64_t, Announced>& seen);

  // Shared body of getline()/getpasswd(): prints the prompt, masks
  // SIGINT/SIGTERM/SIGHUP, poll()s stdin with cancel re-checks, then
  // reads one line. When `mask` is set, terminal echo is disabled for
  // the read on a real tty (and a newline is emitted afterward, since
  // the user's Enter wasn't echoed).
  UiInputStatus
  read_line_(const VpipeFormat& prompt, std::string& out,
             const std::function<bool()>& should_cancel, bool mask);

  std::mutex _io_mu;

  // Repaint thread state. `_progress_run` is what the thread polls to
  // know it should exit; `_progress_mu`/`_progress_cv` let a stop wake
  // it immediately instead of waiting out a kRepaintMs tick.
  std::thread             _progress_thread;
  std::mutex              _progress_mu;
  std::condition_variable _progress_cv;
  bool                    _progress_run = false;

  // Off-tty milestone state. A MEMBER rather than a thread local
  // because the thread is not always the one that gets to close a
  // report out: the last report of a run typically closes within a
  // tick of the session tearing down, and the thread is joined before
  // it can observe the empty list. stop_progress_thread_() flushes it
  // after the join, where the thread is provably gone and no lock is
  // needed to touch it.
  std::map<std::uint64_t, Announced> _milestones;

  // SIGINT bookkeeping. `_sigint_count` is the only field the signal
  // handler touches; the rest belong to poll_sigint()'s caller.
  std::atomic<unsigned> _sigint_count{0};
  unsigned              _sigint_seen = 0;      // last count acted on
  bool                  _sigint_armed = false; // a 2nd tap now quits
  std::chrono::steady_clock::time_point _sigint_at{};
};

}

#endif
