#include "common/console-writer.h"
#include "common/media-line.h"
#include "common/stdio-ui-delegate.h"
#include "common/vpipe-format.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <map>
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

string
local_stamp_()
{
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  ::localtime_r(&now, &tm);
  char stamp[16] = "??:??:??";
  std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &tm);
  return stamp;
}

// One report as a plain line for a destination that is not a terminal:
//   40% of 'denoise 1024x1024' completed at 00:52:14 (640/1600)
//
// Local time, not elapsed: the reader of a redirected log is asking
// "is this still moving", and the answer is the gap between two stamps
// -- which they can only take if the stamps are absolute. The counts
// ride along because they are what substantiates the percentage; there
// is no bar to draw here.
string
render_milestone_(const string& desc, int pct, std::uint64_t done,
                  std::uint64_t total)
{
  string line = std::to_string(pct);
  line += "% of '";
  line += desc;
  line += "' completed at ";
  line += local_stamp_();
  line += " (";
  line += std::to_string(done);
  line += "/";
  line += std::to_string(total);
  line += ")";
  return line;
}

// A report that closed without ever reporting 100%.
//
// Needed for the milestone lines to answer the question they exist for.
// A stage is free to stop updating before the end -- the LM benchmark
// ticks its bar BEFORE each test, so its last word is 8/9 -- and a log
// that simply stops at 88% leaves the reader exactly where they were
// with no output at all: unable to tell finished from wedged. This says
// which it was, and says nothing about SUCCESS, because the registry
// cannot distinguish a normal finish from an abandoned one.
string
render_ended_(const string& desc, std::uint64_t done, std::uint64_t total)
{
  string line = "'";
  line += desc;
  line += "' ended at ";
  line += local_stamp_();
  if (total > 0) {
    int pct = static_cast<int>((done * 100) / total);
    pct = pct < 0 ? 0 : (pct > 100 ? 100 : pct);
    line += ", last reported ";
    line += std::to_string(pct);
    line += "% (";
    line += std::to_string(done);
    line += "/";
    line += std::to_string(total);
    line += ")";
  }
  return line;
}

}  // namespace

void
StdioUiDelegate::emit_milestones_(
    const std::vector<UiProgressRegistry::Item>& items,
    std::map<std::uint64_t, Announced>&          seen)
{
  auto say = [this](string line) {
    emit_("PROGRESS", /*to_err=*/false,
          VpipeFormat([l = std::move(line)] { return l; }));
  };

  for (const auto& it : items) {
    auto& a = seen[it.id];
    a.desc  = it.desc;
    a.done  = it.done;
    a.total = it.total;
    // INDETERMINATE reports have no percentage to cross a milestone
    // with, so they say nothing here. A download whose server sent no
    // Content-Length is therefore still silent off a tty -- the tty
    // path answers it with a bouncing pip, and there is no equivalent
    // that would not be a timer writing lines about nothing.
    if (it.total == 0) { continue; }
    int pct = static_cast<int>((it.done * 100) / it.total);
    pct = pct < 0 ? 0 : (pct > 100 ? 100 : pct);
    // 0 and 100 each get their own bucket, so a report announces itself
    // as soon as it knows its size and again when it finishes: 11 lines
    // at most for a whole report, however many times it updates.
    const int bucket = pct / kMilestonePct;
    if (bucket <= a.bucket) { continue; }
    a.bucket = bucket;
    say(render_milestone_(it.desc, pct, it.done, it.total));
  }

  // Ids that have gone away. Ids are never reused, so an erased entry
  // cannot be resurrected by a later report -- and the erase is what
  // keeps this from growing across a long run of back-to-back reports.
  for (auto i = seen.begin(); i != seen.end();) {
    const bool live = std::any_of(
        items.begin(), items.end(),
        [id = i->first](const auto& it) { return it.id == id; });
    if (live) { ++i; continue; }
    // Silent when the report already announced 100%: that line said
    // everything this one would.
    const bool complete = i->second.total > 0
                          && i->second.done >= i->second.total;
    if (!complete) {
      say(render_ended_(i->second.desc, i->second.done, i->second.total));
    }
    i = seen.erase(i);
  }
}

void
StdioUiDelegate::on_progress_opened()
{
  start_progress_thread_();
}

void
StdioUiDelegate::start_progress_thread_()
{
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
  // Decided ONCE, here rather than per tick: the two halves are
  // different renderings of the same poll, and a stream cannot become a
  // terminal midway through a run.
  const bool tty = console_writer().tty();
  _progress_thread = std::thread([this, tty] {
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
        // Off a tty the same moment is the ONLY chance to say that the
        // last report ended -- one more pass over an empty list, which
        // closes out everything still in the map. Skipping it is how
        // the terminating line goes missing for exactly the report the
        // reader is waiting on.
        if (!tty) { emit_milestones_(items, _milestones); }
        lock_guard<mutex> lk(_progress_mu);
        _progress_run = false;
        break;
      }
      if (!tty) {
        // No footer, and no animation to keep alive either -- a
        // milestone can only be crossed by something actually moving,
        // which is exactly what the version counter reports.
        if (v != last_version) {
          emit_milestones_(items, _milestones);
          last_version = v;
        }
        continue;
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
  // The thread is joined, so _milestones is ours alone. Anything still
  // in it is a report that closed inside the last tick -- which is the
  // NORMAL case for the last report of a run, since a stage tends to
  // finish its bar and return within a few milliseconds of the session
  // going away. Without this the run's final report is the one that
  // never gets a terminating line.
  if (!_milestones.empty()) { emit_milestones_({}, _milestones); }
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
