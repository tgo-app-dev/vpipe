#include "common/media-line.h"
#include "common/stdio-ui-delegate.h"
#include "common/vpipe-format.h"

#include <cerrno>
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
    cout << chunk;
    cout.flush();
    fflush(stdout);
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
    cout << '\n';
    cout.flush();
    fflush(stdout);
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

  lock_guard<mutex> lk(_io_mu);
  if (to_err) {
    cerr << line;
    cerr.flush();
  } else {
    // Explicit flush + fflush so redirected/fully-buffered stdout (a
    // file, or a Python parent reading our pipe) sees the line now.
    cout << line;
    cout.flush();
    fflush(stdout);
  }
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
      lock_guard<mutex> lk(_io_mu);
      cout << p;
      cout.flush();
      fflush(stdout);
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
    cout << '\n';
    cout.flush();
    fflush(stdout);
  }

  return ok ? UiInputStatus::Ok : UiInputStatus::Eof;
}

unique_ptr<UiTextStream>
StdioUiDelegate::open_text_stream()
{
  return make_unique<StdioTextStream>(_io_mu);
}

}
