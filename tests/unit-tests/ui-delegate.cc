#include "minitest.h"
#include "common/stdio-ui-delegate.h"
#include "common/vpipe-format.h"
#include "interfaces/ui-delegate-intf.h"

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace std;
using namespace vpipe;

namespace {

// Minimal delegate that implements only the required surface (and
// getline). Used to verify the interface's getpasswd default forwards
// to getline -- the graceful-degradation contract for delegates that
// cannot mask.
class GetlineOnlyUi final : public UiDelegateIntf {
public:
  void error(const VpipeFormat&) override {}
  void warn(const VpipeFormat&) override {}
  void info(const VpipeFormat&) override {}
  UiInputStatus
  getline(const VpipeFormat&, string& out,
          const function<bool()>&) override
  {
    getline_called = true;
    out = "from-getline";
    return UiInputStatus::Ok;
  }
  unique_ptr<UiTextStream> open_text_stream() override
  {
    return make_unique<NullUiTextStream>();
  }
  bool getline_called = false;
};

// Redirect cout + cerr to in-memory buffers for the lifetime of the
// guard so emit tests don't pollute the runner's output.
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

// getline returns Canceled promptly when the cancel predicate is
// already true -- the predicate is checked before stdin is touched,
// so this is deterministic without any stdin fixture.
TEST(ui_delegate, stdio_getline_cancels_without_reading_stdin) {
  IoCapture cap;
  StdioUiDelegate d;
  string out = "unchanged";
  UiInputStatus st = d.getline(fmt(""), out, [] { return true; });
  EXPECT_TRUE(st == UiInputStatus::Canceled);
}

// getpasswd shares getline's cancel path (the no-echo guard is a no-op
// when stdin is not a tty, as in the test runner), so it too cancels
// promptly without reading stdin.
TEST(ui_delegate, stdio_getpasswd_cancels_without_reading_stdin) {
  IoCapture cap;
  StdioUiDelegate d;
  string out = "unchanged";
  UiInputStatus st = d.getpasswd(fmt(""), out, [] { return true; });
  EXPECT_TRUE(st == UiInputStatus::Canceled);
}

// The UiDelegateIntf::getpasswd default forwards to getline for
// delegates that don't override it (input is just not masked).
TEST(ui_delegate, getpasswd_defaults_to_getline) {
  GetlineOnlyUi d;
  string out;
  UiInputStatus st = d.getpasswd(fmt("pw: "), out, nullptr);
  EXPECT_TRUE(st == UiInputStatus::Ok);
  EXPECT_TRUE(d.getline_called);
  EXPECT_TRUE(out == "from-getline");
}

// The UiDelegateIntf::getmedialine default forwards to getline for
// delegates that don't collect attachments (any markers the user types
// literally still travel through and parse downstream).
TEST(ui_delegate, getmedialine_defaults_to_getline) {
  GetlineOnlyUi d;
  string out;
  UiInputStatus st = d.getmedialine(fmt(">> "), out, nullptr);
  EXPECT_TRUE(st == UiInputStatus::Ok);
  EXPECT_TRUE(d.getline_called);
  EXPECT_TRUE(out == "from-getline");
}

// error/warn -> stderr, info -> stdout, each a single framed line.
TEST(ui_delegate, stdio_emit_routes_and_frames) {
  IoCapture cap;
  StdioUiDelegate d;
  d.info(fmt("hello {}", 1));
  d.warn(fmt("careful"));
  d.error(fmt("boom"));
  EXPECT_TRUE(cap.out() == "[INFO] hello 1\n");
  EXPECT_TRUE(cap.err() == "[WARN] careful\n[ERROR] boom\n");
}

// open_text_stream writes chunks verbatim to stdout (no framing) and
// emits a single terminating newline on end().
TEST(ui_delegate, stdio_text_stream_writes_verbatim_then_newline) {
  IoCapture cap;
  StdioUiDelegate d;
  {
    auto s = d.open_text_stream();
    s->write("Par");
    s->write("is");
    s->end();
    s->end();   // idempotent
  }
  EXPECT_TRUE(cap.out() == "Paris\n");
  EXPECT_TRUE(cap.err().empty());
}

// A stream with no writes emits nothing (not even a newline) on end.
TEST(ui_delegate, stdio_text_stream_empty_emits_nothing) {
  IoCapture cap;
  StdioUiDelegate d;
  { auto s = d.open_text_stream(); }   // dtor calls end()
  EXPECT_TRUE(cap.out().empty());
}
