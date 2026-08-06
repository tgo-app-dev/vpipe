#include "common/console-writer.h"
#include "common/media-line.h"
#include "common/stdio-ui-delegate.h"
#include "common/vpipe-format.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <poll.h>
#include <pthread.h>
#include <string>
#include <string_view>
#include <termios.h>
#include <unistd.h>

using namespace std;

namespace vpipe {

namespace {

// Verbatim stdout stream: chunks go out immediately (flushed), end()
// terminates the line. Shares the delegate's I/O mutex so it can't
// tear against [LEVEL] message lines.
class StdioTextStream final : public UiTextStream {
public:
  explicit StdioTextStream(mutex& io_mu) : _io_mu(io_mu) {}
  ~StdioTextStream() override { end(); }

  void write(string_view chunk) override
  {
    if (chunk.empty()) {
      return;
    }
    // Thinking markers arrive as whole chunks (the detokenizer emits
    // each as one piece); render them as readable tags on a terminal.
    string plain;
    if (chunk.find(media_line::kThinkStart) != string_view::npos
        || chunk.find(media_line::kThinkEnd) != string_view::npos) {
      plain = media_line::render_think_markers_plain(chunk);
      chunk = plain;
    }
    lock_guard<mutex> lk(_io_mu);
    // write_text, not write_line: a token chunk rarely ends in a
    // newline, and the ConsoleWriter has to know the cursor is left
    // mid-line so it suppresses the progress footer until the stream
    // finishes rather than painting bars into the middle of a sentence.
    console_writer().write_text(/*to_err=*/false, chunk);
    _wrote = true;
  }

  void end() override
  {
    if (_ended) {
      return;
    }
    _ended = true;
    if (!_wrote) {
      return;
    }
    lock_guard<mutex> lk(_io_mu);
    console_writer().write_line(/*to_err=*/false, "\n");
  }

private:
  mutex& _io_mu;
  bool   _wrote = false;
  bool   _ended = false;
};

// RAII: disable stdin echo on a real tty for the lifetime of the
// guard (password reads), restoring the prior termios on destruction
// -- so every return path of read_line_ leaves the terminal as it
// found it. A no-op when stdin is not a tty (pipe / file / web-ui).
class NoEchoGuard {
public:
  NoEchoGuard()
  {
    if (!::isatty(STDIN_FILENO)) {
      return;
    }
    if (::tcgetattr(STDIN_FILENO, &_old) != 0) {
      return;
    }
    termios t = _old;
    t.c_lflag &= ~static_cast<tcflag_t>(ECHO);
    if (::tcsetattr(STDIN_FILENO, TCSANOW, &t) == 0) {
      _active = true;
    }
  }
  ~NoEchoGuard()
  {
    if (_active) {
      ::tcsetattr(STDIN_FILENO, TCSANOW, &_old);
    }
  }
  NoEchoGuard(const NoEchoGuard&)            = delete;
  NoEchoGuard& operator=(const NoEchoGuard&) = delete;
  bool active() const { return _active; }

private:
  termios _old{};
  bool    _active = false;
};

}  // namespace

// ---- progress footer -------------------------------------------------

namespace {

// One report as a footer row:
//   [########----------------]  34%  fetch qwen3-4b   1.2 / 3.0 GB
// The bar is 24 cells, matching what the model-fetch stage drew before
// this moved behind the UI delegate.
constexpr int kBarCells = 24;

string
render_row_(const UiProgressRegistry::Item& it, int tick)
{
  string bar = "[";
  if (it.total > 0) {
    int pct = static_cast<int>((it.done * 100) / it.total);
    pct = pct < 0 ? 0 : (pct > 100 ? 100 : pct);
    const int filled = pct * kBarCells / 100;
    for (int i = 0; i < kBarCells; ++i) {
      bar += (i < filled ? '#' : '-');
    }
    bar += "] ";
    const string p = std::to_string(pct);
    bar.append(3 - std::min<size_t>(3, p.size()), ' ');
    bar += p;
    bar += "%  ";
  } else {
    // INDETERMINATE: no percentage exists, so show motion instead of a
    // fill -- a pip bouncing across the cells. Without it a download
    // that has not yet reported a Content-Length looks frozen.
    const int span = kBarCells * 2 - 2;          // there and back
    int pos = span > 0 ? (tick % span) : 0;
    if (pos >= kBarCells) { pos = span - pos; }
    for (int i = 0; i < kBarCells; ++i) {
      bar += (i == pos ? '#' : '-');
    }
    bar += "]  --  ";
  }
  bar += it.desc;
  if (!it.detail.empty()) {
    bar += "   ";
    bar += it.detail;
  }
  return bar;
}

}  // namespace

void
StdioUiDelegate::on_progress_opened()
{
  start_progress_thread_();
}

void
StdioUiDelegate::start_progress_thread_()
{
  // Nothing to pin a footer to when stdout is a file or a pipe, and
  // painting one there would write escape sequences into a log.
  if (!console_writer().tty()) { return; }
  // Reap a PREVIOUS run's thread outside the lock. The thread takes
  // _progress_mu at the top of every tick, so joining while holding it
  // deadlocks: the joiner waits for a thread that is waiting for the
  // joiner's mutex. Moving the handle out first means the join below
  // touches nothing the thread still needs.
  std::thread stale;
  {
    lock_guard<mutex> lk(_progress_mu);
    if (_progress_run) { return; }             // a live thread has it
    stale = std::move(_progress_thread);       // exited, or never started
  }
  if (stale.joinable()) { stale.join(); }
  lock_guard<mutex> lk(_progress_mu);
  if (_progress_run) { return; }               // a peer won the race
  _progress_run = true;
  _progress_thread = std::thread([this] {
    std::uint64_t last_version = 0;
    int  tick    = 0;
    bool painted = false;
    for (;;) {
      {
        unique_lock<mutex> lk(_progress_mu);
        if (!_progress_run) { break; }
        _progress_cv.wait_for(lk, std::chrono::milliseconds(kRepaintMs));
        if (!_progress_run) { break; }
      }
      ++tick;
      const std::uint64_t v = progress_version();
      const auto items = progress_snapshot();
      if (items.empty()) {
        // Last report closed: wipe the block and let the thread go.
        // A later report starts a fresh one through on_progress_opened.
        if (painted) { console_writer().set_footer(this, {}); }
        lock_guard<mutex> lk(_progress_mu);
        _progress_run = false;
        break;
      }
      // Repaint on a real change, or on every tick while an
      // indeterminate report is live -- its pip is animated by `tick`,
      // which the version counter knows nothing about.
      bool animating = false;
      for (const auto& it : items) {
        if (it.total == 0) { animating = true; break; }
      }
      if (v != last_version || animating || !painted) {
        vector<string> rows;
        rows.reserve(items.size());
        for (const auto& it : items) { rows.push_back(render_row_(it, tick)); }
        console_writer().set_footer(this, std::move(rows));
        last_version = v;
        painted      = true;
      }
    }
  });
}

void
StdioUiDelegate::stop_progress_thread_()
{
  {
    lock_guard<mutex> lk(_progress_mu);
    _progress_run = false;
  }
  _progress_cv.notify_all();
  if (_progress_thread.joinable()) { _progress_thread.join(); }
  console_writer().set_footer(this, {});
}

StdioUiDelegate::~StdioUiDelegate()
{
  stop_progress_thread_();
}

void
StdioUiDelegate::emit_(const char* tag, bool to_err, const VpipeFormat& f)
{
  string line;
  line.reserve(64);
  line.push_back('[');
  line.append(tag);
  line.append("] ");
  try {
    line.append(f());
  } catch (...) {
    line.append("<formatter threw>");
  }
  // Reply text relayed through info() (e.g. visual-qa answers) may
  // carry the unified thinking markers; render them readably.
  if (line.find(media_line::kThinkStart) != string::npos
      || line.find(media_line::kThinkEnd) != string::npos) {
    line = media_line::render_think_markers_plain(line);
  }
  line.push_back('\n');

  // Through the ConsoleWriter, not straight to the stream: it owns the
  // progress footer's rows and has to erase them before this line
  // scrolls the terminal, then repaint below it. Off a tty that is
  // exactly the old write + flush + fflush.
  lock_guard<mutex> lk(_io_mu);
  console_writer().write_line(to_err, line);
}

void
StdioUiDelegate::error(const VpipeFormat& f)
{
  emit_("ERROR", /*to_err=*/true, f);
}

void
StdioUiDelegate::warn(const VpipeFormat& f)
{
  emit_("WARN", /*to_err=*/true, f);
}

void
StdioUiDelegate::info(const VpipeFormat& f)
{
  emit_("INFO", /*to_err=*/false, f);
}

void
StdioUiDelegate::note_sigint() noexcept
{
  _sigint_count.fetch_add(1, memory_order_relaxed);
}

bool
StdioUiDelegate::poll_sigint()
{
  const unsigned n = _sigint_count.load(memory_order_relaxed);
  if (n == _sigint_seen) {
    return false;
  }
  const auto now = chrono::steady_clock::now();
  // A double tap is either two strokes since the last poll (the user
  // hit Ctrl-C twice inside one poll interval) or a fresh stroke
  // arriving while the previous one is still armed.
  const bool burst = (n - _sigint_seen) >= 2;
  const bool armed_recently =
      _sigint_armed
      && (now - _sigint_at) < chrono::milliseconds(kDoubleTapMs);
  _sigint_seen = n;
  _sigint_at   = now;
  if (burst || armed_recently) {
    _sigint_armed = false;
    return true;
  }
  // First stroke: hand it to the stages. If nobody had anything to
  // interrupt, fall through to the historical meaning of a lone SIGINT
  // so the process still stops on one keystroke.
  const int acted = dispatch_interrupt();
  if (acted <= 0) {
    _sigint_armed = false;
    return true;
  }
  _sigint_armed = true;
  info(fmt("interrupted {} running task(s). Press Ctrl-C again within "
           "{:.1f}s to stop the pipelines and quit.",
           acted, kDoubleTapMs / 1000.0));
  return false;
}

UiInputStatus
StdioUiDelegate::getline(const VpipeFormat&           prompt,
                         string&                      out,
                         const function<bool()>&      should_cancel)
{
  return read_line_(prompt, out, should_cancel, /*mask=*/false);
}

UiInputStatus
StdioUiDelegate::getpasswd(const VpipeFormat&           prompt,
                           string&                      out,
                           const function<bool()>&      should_cancel)
{
  return read_line_(prompt, out, should_cancel, /*mask=*/true);
}

UiInputStatus
StdioUiDelegate::read_line_(const VpipeFormat&           prompt,
                            string&                      out,
                            const function<bool()>&      should_cancel,
                            bool                         mask)
{
  // Print the prompt (flushed) so the user sees it before we block.
  {
    string p;
    try {
      p = prompt();
    } catch (...) {
      p.clear();
    }
    if (!p.empty()) {
      // A prompt ends without a newline, so this leaves the cursor
      // mid-line and the footer stays suppressed while the user types.
      lock_guard<mutex> lk(_io_mu);
      console_writer().write_text(/*to_err=*/false, p);
    }
  }

  // For a password read, disable terminal echo so nothing is shown as
  // the user types. RAII restores the prior state on every return path.
  optional<NoEchoGuard> echo_guard;
  if (mask) {
    echo_guard.emplace();
  }

  // Block SIGINT/SIGTERM/SIGHUP on this worker for the duration of the
  // read. Without this the kernel can deliver SIGINT here -- the only
  // thread parked in a syscall -- where a Python driver's trip flag is
  // set but only the main thread's PyErr_CheckSignals can observe it,
  // so the interrupt is missed. Masking re-routes the signal to the
  // main thread. The mask is restored before we return.
  sigset_t block_set, prev_set;
  sigemptyset(&block_set);
  sigaddset(&block_set, SIGINT);
  sigaddset(&block_set, SIGTERM);
  sigaddset(&block_set, SIGHUP);
  pthread_sigmask(SIG_BLOCK, &block_set, &prev_set);

  // Poll stdin on a short timeout so a cancel request is observed
  // within ~50ms; std::getline by itself blocks the worker forever.
  constexpr int kPollMs = 50;
  bool data_ready = false;
  while (!data_ready) {
    if (should_cancel && should_cancel()) {
      pthread_sigmask(SIG_SETMASK, &prev_set, nullptr);
      return UiInputStatus::Canceled;
    }
    struct pollfd pfd { STDIN_FILENO, POLLIN, 0 };
    int rc = ::poll(&pfd, 1, kPollMs);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      // Unexpected error; fall through to getline which surfaces the
      // real failure via the stream state.
      break;
    }
    if (rc == 0) {
      continue;  // timeout: re-check cancel and poll again
    }
    if (pfd.revents & (POLLIN | POLLHUP)) {
      data_ready = true;
    }
  }

  pthread_sigmask(SIG_SETMASK, &prev_set, nullptr);

  out.clear();
  const bool ok = static_cast<bool>(std::getline(cin, out));

  // With echo suppressed, the Enter the user pressed left no visible
  // newline; emit one so subsequent output starts on a fresh line.
  if (mask && echo_guard && echo_guard->active()) {
    lock_guard<mutex> lk(_io_mu);
    console_writer().write_line(/*to_err=*/false, "\n");
  }

  return ok ? UiInputStatus::Ok : UiInputStatus::Eof;
}

unique_ptr<UiTextStream>
StdioUiDelegate::open_text_stream()
{
  return make_unique<StdioTextStream>(_io_mu);
}

}
