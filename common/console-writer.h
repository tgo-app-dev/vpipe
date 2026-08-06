#ifndef CONSOLE_WRITER_H
#define CONSOLE_WRITER_H

#include <cstddef>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// The single owner of the process's terminal.
//
// A progress FOOTER -- bars pinned to the bottom rows while ordinary
// output scrolls above them -- only holds together if exactly one
// object decides when those rows are erased and repainted. Before this,
// two objects wrote to stdout with unrelated mutexes: StdioUiDelegate
// (error/warn/info, text streams, the getline prompt) and
// StdoutLogDelegate (log lines, emitted from its own drain thread).
// Either one printing mid-frame would scroll the footer away and leave
// a torn copy behind.
//
// So both now route their writes here. write_line() erases the footer,
// emits the line, and repaints -- atomically under one mutex.
//
// A SINGLETON, reached through console_writer(), rather than an
// instance threaded through construction: there is one terminal per
// process, but the two delegates are built independently at several
// call sites (Session's two constructors, the web-ui app) with no
// shared owner to pass it through.
//
// LOCK ORDER is strictly delegate -> console and never the reverse.
// This class therefore MUST NOT call back into a delegate: it renders
// from rows it was handed, never by asking anyone for them.
//
// NOT A TTY (redirected output, vpipe_test, a Python parent reading our
// stdout) is the common case, and there the footer is meaningless: it
// would emit escape sequences into a log file. set_footer() then does
// nothing and write_line() degrades to exactly what the delegates did
// before -- write, flush the streambuf, fflush the FILE* -- so a
// redirected log is byte-for-byte what it always was.
class ConsoleWriter {
public:
  ConsoleWriter();

  // Emit one COMPLETE line (callers pass their own trailing newline),
  // to stderr when `to_err` is set. Footer-safe.
  void write_line(bool to_err, std::string_view line);

  // Emit a partial, un-framed chunk -- token-by-token model output, a
  // getline prompt. Anything not ending in a newline leaves the cursor
  // MID-LINE, where a footer cannot be painted below it without
  // breaking the line being written. The footer is therefore suppressed
  // until the line completes; a token stream in flight simply wins over
  // the bars, which is the right precedence (the user is reading it).
  void write_text(bool to_err, std::string_view chunk);

  // Install or replace `owner`'s footer rows; empty `rows` removes the
  // owner entirely. Keyed by owner so several Sessions in one process
  // (the test binary) contribute rows instead of overwriting one
  // another's block.
  void set_footer(const void* owner, std::vector<std::string> rows);

  // Erase the footer and forget every owner, leaving the terminal
  // clean. Idempotent.
  void clear_footer();

  bool tty() const { return _tty; }

  // Current terminal width in columns (80 when unknown), for callers
  // that elide their rows to fit. Re-queried per call so a resized
  // window is picked up without a restart.
  int width() const;

private:
  // Both assume the cursor sits at column 0 of the line just below the
  // block -- which holds as long as every write goes through this
  // class. Callers must hold _mu.
  void erase_footer_locked_();
  void paint_footer_locked_();
  void emit_locked_(bool to_err, std::string_view s);

  struct Owned {
    const void*              owner = nullptr;
    std::vector<std::string> rows;
  };

  mutable std::mutex _mu;
  std::vector<Owned> _footers;
  int                _painted  = 0;      // rows currently on screen
  bool               _mid_line = false;  // last write left the cursor mid-line
  bool               _tty      = false;
};

// The process-wide instance. Never null; safe from any thread.
ConsoleWriter& console_writer();

}

#endif
