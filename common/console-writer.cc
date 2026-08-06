#include "common/console-writer.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

namespace vpipe {

using std::string;
using std::string_view;
using std::vector;

namespace {

// Cursor up N rows.
string cursor_up_(int n)
{
  return "\033[" + std::to_string(n) + "A";
}

// Erase from the cursor to the end of the screen.
constexpr const char* kEraseBelow = "\033[J";

}  // namespace

ConsoleWriter&
console_writer()
{
  // Function-local static: constructed on first use and never
  // destroyed before the last writer, which matters because delegates
  // are torn down at static-destruction time in some apps.
  static ConsoleWriter* w = new ConsoleWriter();
  return *w;
}

ConsoleWriter::ConsoleWriter()
{
  // stdout decides: the footer lives there, and a run with stdout
  // redirected but stderr on a terminal must still not emit escapes
  // (the log file is what gets parsed).
  _tty = ::isatty(::fileno(stdout)) != 0;
}

int
ConsoleWriter::width() const
{
  if (!_tty) { return 80; }
  struct winsize ws {};
  if (::ioctl(::fileno(stdout), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return static_cast<int>(ws.ws_col);
  }
  return 80;
}

void
ConsoleWriter::emit_locked_(bool to_err, string_view s)
{
  if (s.empty()) { return; }
  if (to_err) {
    std::cerr.write(s.data(), static_cast<std::streamsize>(s.size()));
    std::cerr.flush();
    return;
  }
  // Flush BOTH layers. When stdout is redirected to a file it is fully
  // buffered, not line-buffered, so without this a parent process
  // tailing our output sees nothing until ~4 KiB accumulates and a
  // steadily-progressing run looks hung. cout.flush() only drains the
  // C++ streambuf; fflush drops the stdio FILE* buffer to the kernel
  // too. This output is low-volume, so the extra syscalls are fine.
  std::cout.write(s.data(), static_cast<std::streamsize>(s.size()));
  std::cout.flush();
  std::fflush(stdout);
}

void
ConsoleWriter::erase_footer_locked_()
{
  if (_painted <= 0) { return; }
  // One sequence: jump back to the block's first row and wipe
  // everything below. Clearing row by row would flicker.
  const string seq = cursor_up_(_painted) + kEraseBelow;
  emit_locked_(/*to_err=*/false, seq);
  _painted = 0;
}

void
ConsoleWriter::paint_footer_locked_()
{
  if (!_tty || _mid_line) { return; }
  const int cols = width();
  string out;
  int rows = 0;
  for (const Owned& f : _footers) {
    for (const string& r : f.rows) {
      // Elide rather than let the terminal wrap: a wrapped row occupies
      // two screen lines, so the cursor-up count would no longer match
      // what is on screen and the next erase would eat real output.
      if (cols > 1 && static_cast<int>(r.size()) > cols - 1) {
        out.append(r, 0, static_cast<std::size_t>(cols - 1));
      } else {
        out.append(r);
      }
      out.push_back('\n');
      ++rows;
    }
  }
  if (rows == 0) { return; }
  emit_locked_(/*to_err=*/false, out);
  _painted = rows;
}

void
ConsoleWriter::write_line(bool to_err, string_view line)
{
  std::lock_guard<std::mutex> lk(_mu);
  erase_footer_locked_();
  emit_locked_(to_err, line);
  // A complete line puts the cursor back at column 0, so whatever a
  // partial write left open is finished as far as the footer cares.
  if (!line.empty()) { _mid_line = line.back() != '\n'; }
  paint_footer_locked_();
}

void
ConsoleWriter::write_text(bool to_err, string_view chunk)
{
  if (chunk.empty()) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  erase_footer_locked_();
  emit_locked_(to_err, chunk);
  _mid_line = chunk.back() != '\n';
  paint_footer_locked_();      // no-ops while _mid_line
}

void
ConsoleWriter::set_footer(const void* owner, vector<string> rows)
{
  if (owner == nullptr) { return; }
  std::lock_guard<std::mutex> lk(_mu);
  if (!_tty) {
    // Keep no state at all off a terminal: nothing will ever paint it,
    // and holding rows would only grow with every update.
    return;
  }
  auto it = std::find_if(_footers.begin(), _footers.end(),
                         [owner](const Owned& o) {
                           return o.owner == owner;
                         });
  if (rows.empty()) {
    if (it != _footers.end()) { _footers.erase(it); }
  } else if (it != _footers.end()) {
    it->rows = std::move(rows);
  } else {
    _footers.push_back(Owned{owner, std::move(rows)});
  }
  erase_footer_locked_();
  paint_footer_locked_();
}

void
ConsoleWriter::clear_footer()
{
  std::lock_guard<std::mutex> lk(_mu);
  erase_footer_locked_();
  _footers.clear();
}

}
