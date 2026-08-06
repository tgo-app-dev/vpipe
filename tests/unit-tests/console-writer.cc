#include "minitest.h"
#include "common/console-writer.h"
#include "common/stdio-ui-delegate.h"
#include "common/vpipe-format.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using namespace std;
using namespace vpipe;

namespace {

// Redirect cout + cerr to in-memory buffers, as ui-delegate.cc does.
class IoCapture {
public:
  IoCapture()
    : _o(cout.rdbuf(_ob.rdbuf())), _e(cerr.rdbuf(_eb.rdbuf())) {}
  ~IoCapture() { cout.rdbuf(_o); cerr.rdbuf(_e); }
  string out() const { return _ob.str(); }
  string err() const { return _eb.str(); }
private:
  stringstream _ob, _eb;
  streambuf*   _o;
  streambuf*   _e;
};

}  // namespace

// The test binary's stdout is a pipe or a file under every runner we
// use, so these pin the NON-TTY contract -- the one that matters for a
// redirected log. If this ever runs on a terminal the footer assertions
// would be about escape sequences instead, so they are skipped there
// rather than made to pass both ways.

// write_line is byte-for-byte what the delegates emitted before the
// ConsoleWriter existed: the text, nothing added.
TEST(console_writer, write_line_verbatim) {
  IoCapture cap;
  console_writer().write_line(/*to_err=*/false, "[NORMAL] hello\n");
  EXPECT_TRUE(cap.out() == "[NORMAL] hello\n");
  EXPECT_TRUE(cap.err().empty());
}

// to_err routes to stderr and leaves stdout untouched -- warn/error
// keep their stream.
TEST(console_writer, write_line_to_err) {
  IoCapture cap;
  console_writer().write_line(/*to_err=*/true, "[WARN] careful\n");
  EXPECT_TRUE(cap.err() == "[WARN] careful\n");
  EXPECT_TRUE(cap.out().empty());
}

// A partial chunk is emitted as-is with no newline of its own: this is
// the token-streaming path, where the delegate owns the framing.
TEST(console_writer, write_text_partial_verbatim) {
  IoCapture cap;
  console_writer().write_text(/*to_err=*/false, "tok");
  console_writer().write_text(/*to_err=*/false, "en");
  EXPECT_TRUE(cap.out() == "token");
}

// Off a tty a footer must produce NOTHING. Emitting escape sequences
// into a redirected log is the failure this guards.
TEST(console_writer, footer_silent_off_tty) {
  if (console_writer().tty()) { return; }   // see the note above
  IoCapture cap;
  console_writer().set_footer(&cap, {"[####----] 50% work"});
  EXPECT_TRUE(cap.out().empty());
  console_writer().write_line(/*to_err=*/false, "a line\n");
  EXPECT_TRUE(cap.out() == "a line\n");
  console_writer().set_footer(&cap, {});
  EXPECT_TRUE(cap.out() == "a line\n");
}

// clear_footer is safe with nothing painted, and stays silent off a
// tty -- it runs on every StdioUiDelegate teardown, including the many
// the test binary builds.
TEST(console_writer, clear_footer_idempotent) {
  IoCapture cap;
  console_writer().clear_footer();
  console_writer().clear_footer();
  if (!console_writer().tty()) { EXPECT_TRUE(cap.out().empty()); }
}

// An empty write is a no-op rather than a flush storm; the delegates
// call through unconditionally.
TEST(console_writer, empty_write_is_noop) {
  IoCapture cap;
  console_writer().write_text(/*to_err=*/false, "");
  console_writer().write_line(/*to_err=*/false, "");
  EXPECT_TRUE(cap.out().empty());
}

// width() always answers something usable, so a caller eliding rows
// never has to special-case "unknown".
TEST(console_writer, width_is_positive) {
  EXPECT_TRUE(console_writer().width() > 0);
}

// ---- on a real terminal ----------------------------------------------

// The footer only exists on a tty, so this one is driven through a pty:
//
//   script -q /dev/null env VPIPE_CONSOLE_FOOTER_DEMO=1 \
//       ./vpipe_test --filter 'console_writer.footer_on_tty'
//
// It opens two concurrent reports, updates them while ordinary lines are
// emitted through the same writer, and asserts the terminal control it
// must have used. Gated because without a pty there is nothing to
// assert and the interesting path never runs.
TEST(console_writer, footer_on_tty) {
  if (getenv("VPIPE_CONSOLE_FOOTER_DEMO") == nullptr) { return; }
  ASSERT_TRUE(console_writer().tty());
  StdioUiDelegate d;
  UiProgress a = d.open_progress("fetch shard-1");
  UiProgress b = d.open_progress("denoise");
  for (int i = 0; i <= 10; ++i) {
    a.update(static_cast<uint64_t>(i), 10,
             to_string(i) + " / 10 GB");
    b.update(static_cast<uint64_t>(10 - i), 10);
    // Interleave output on BOTH streams: on a terminal stderr and
    // stdout are the same device, so a warn tears the block exactly
    // like an info does.
    d.info(fmt("interleaved line {}", i));
    if (i % 4 == 0) { d.warn(fmt("a warning at {}", i)); }
    this_thread::sleep_for(chrono::milliseconds(120));
  }
  a.finish();
  b.finish();
  // Let the repaint thread notice the empty registry and exit.
  this_thread::sleep_for(chrono::milliseconds(400));

  // RESTART after the thread exited: start_progress_thread_ has to reap
  // the finished thread, and it must do that OUTSIDE its own mutex --
  // the thread takes that mutex every tick, so joining under it
  // deadlocks. Reaching the end of this test is the assertion.
  UiProgress again = d.open_progress("second wave");
  for (int i = 0; i <= 4; ++i) {
    again.update(static_cast<uint64_t>(i), 4);
    d.info(fmt("after restart {}", i));
    this_thread::sleep_for(chrono::milliseconds(80));
  }
  again.finish();

  // An INDETERMINATE report (total 0) animates on its own clock rather
  // than on the version counter, so it must keep repainting.
  UiProgress unknown = d.open_progress("sizing");
  for (int i = 0; i < 6; ++i) {
    unknown.update(static_cast<uint64_t>(i) * 1024, 0);
    this_thread::sleep_for(chrono::milliseconds(80));
  }
  unknown.finish();
  this_thread::sleep_for(chrono::milliseconds(250));
}
