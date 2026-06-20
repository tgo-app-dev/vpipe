#include "minitest.h"
#include "common/session.h"
#include "common/stdout-log-delegate.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/log-delegate-intf.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/stage.h"
#include "vpipe/pipeline-handle.h"
#include "vpipe/status.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Counts warn() emissions so the negative-path tests can assert the
// public insert_stage path produced its diagnostic. warn() now routes
// to the session UI delegate (not the log delegate), so this is a
// UiDelegateIntf installed via Session::set_ui_delegate.
struct CountingDelegate final : public UiDelegateIntf {
  mutex    mu;
  unsigned warns = 0;
  string   last_warn;

  void error(const VpipeFormat&) override {}
  void warn(const VpipeFormat& f) override {
    lock_guard<mutex> lk(mu);
    ++warns;
    last_warn = f();
  }
  void info(const VpipeFormat&) override {}
  UiInputStatus getline(const VpipeFormat&, string&,
                        const function<bool()>&) override {
    return UiInputStatus::Eof;
  }
  unique_ptr<UiTextStream> open_text_stream() override {
    return make_unique<NullUiTextStream>();
  }
};

// Silent log delegate: keeps insert_stage's log_* output off the test
// runner's stdout/stderr (warn/info/error go to the UI delegate above).
struct SilentLogDelegate final : public LogDelegateIntf {
  void log(LogLevel, const VpipeFormat&) override {}
  void set_threshold(LogLevel) override {}
  LogLevel threshold() const noexcept override { return LogLevel::Always; }
};

string
tmp_path_(const char* tag)
{
  string p = "/tmp/vpipe-pipeline-handle-";
  p += tag;
  p += "-";
  p += to_string(getpid());
  p += ".out";
  return p;
}

// Reach into a handle's live Pipeline graph to find a stage by id, so
// the move_iport tests can assert the actual edges (the public handle
// API is intentionally edge-opaque).
Stage*
live_stage_(PipelineHandle& pl, const string& id)
{
  Pipeline* g = HandleAccess::impl(pl)->pipeline();
  if (!g) { return nullptr; }
  for (auto it = g->begin(); it != g->end(); ++it) {
    if (Stage* s = dynamic_cast<Stage*>(*it)) {
      if (s->id() == id) { return s; }
    }
  }
  return nullptr;
}

}

TEST(pipeline_handle, insert_stage_builds_chrono_shell_pipeline) {
  Session sess;
  string out = tmp_path_("chrono-shell");
  ::unlink(out.c_str());

  PipelineHandle pl = sess.create_pipeline("p");
  ASSERT_TRUE(pl.valid());

  StageHandle src = pl.insert_stage(
      "chrono", "src", {},
      R"({"frequency_hz": 100.0, "count": 3})");
  EXPECT_TRUE(src.valid());
  EXPECT_TRUE(src.num_oports() == 1u);

  string cmd = R"({"command":"echo tick >> )" + out + R"("})";
  StageHandle sink = pl.insert_stage(
      "shell", "sink", {{src, 0u}}, cmd);
  EXPECT_TRUE(sink.valid());

  EXPECT_TRUE(sess.launch_pipeline(pl).code == 0u);
  // 3 ticks at 100Hz ~= 30ms; sleep generously and stop.
  this_thread::sleep_for(chrono::milliseconds(150));
  EXPECT_TRUE(sess.stop_pipeline(pl).code == 0u);

  ifstream in(out);
  unsigned lines = 0;
  string line;
  while (getline(in, line)) {
    ++lines;
  }
  EXPECT_TRUE(lines == 3u);

  ::unlink(out.c_str());
}

TEST(pipeline_handle, insert_stage_unknown_type_returns_null) {
  auto delegate = make_unique<CountingDelegate>();
  CountingDelegate* cap = delegate.get();
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(std::move(delegate));

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle s = pl.insert_stage("does-not-exist", "x", {});
  EXPECT_FALSE(s.valid());
  EXPECT_TRUE(cap->warns >= 1u);
}

TEST(pipeline_handle, insert_stage_bad_config_still_constructs) {
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(make_unique<CountingDelegate>());

  PipelineHandle pl = sess.create_pipeline("p");
  // chrono requires either frequency_hz or period_*, but validation is
  // now deferred to launch: construction succeeds for any config so the
  // graph can be built/edited first. The bad config is reported (and
  // the stage skipped) at launch instead of failing insertion.
  StageHandle s = pl.insert_stage("chrono", "src", {}, "{}");
  EXPECT_TRUE(s.valid());
}

TEST(pipeline_handle, insert_stage_oport_index_out_of_range) {
  auto delegate = make_unique<CountingDelegate>();
  CountingDelegate* cap = delegate.get();
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(std::move(delegate));

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle src = pl.insert_stage(
      "chrono", "src", {},
      R"({"frequency_hz": 10.0, "count": 1})");
  ASSERT_TRUE(src.valid());

  // chrono allocates exactly one oport; index 1 is out of range.
  StageHandle bad = pl.insert_stage(
      "shell", "bad", {{src, 1u}},
      R"({"command":"true"})");
  EXPECT_FALSE(bad.valid());
  EXPECT_TRUE(cap->warns >= 1u);
}

TEST(pipeline_handle, insert_stage_invalid_json_returns_null) {
  auto delegate = make_unique<CountingDelegate>();
  CountingDelegate* cap = delegate.get();
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(std::move(delegate));

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle s = pl.insert_stage("chrono", "src", {},
                                  "this is not json");
  EXPECT_FALSE(s.valid());
  EXPECT_TRUE(cap->warns >= 1u);
}

TEST(pipeline_handle, session_dtor_stops_running_pipeline) {
  // Leave the pipeline running indefinitely and let the session
  // destructor tear it down. The whole sequence must complete within
  // a small fraction of a second.
  string out = tmp_path_("dtor-stops");
  ::unlink(out.c_str());

  auto t0 = chrono::steady_clock::now();
  {
    Session sess;
    PipelineHandle pl = sess.create_pipeline("p");
    StageHandle src = pl.insert_stage(
        "chrono", "src", {},
        R"({"frequency_hz": 100.0})");  // no count -> indefinite
    ASSERT_TRUE(src.valid());
    string cmd = R"({"command":"echo tick >> )" + out + R"("})";
    StageHandle sink = pl.insert_stage(
        "shell", "sink", {{src, 0u}}, cmd);
    ASSERT_TRUE(sink.valid());
    EXPECT_TRUE(sess.launch_pipeline(pl).code == 0u);
    // Let it run briefly so we know it's actually working.
    this_thread::sleep_for(chrono::milliseconds(50));
    // sess dtor runs at end of scope.
  }
  auto elapsed = chrono::steady_clock::now() - t0;
  EXPECT_TRUE(elapsed < chrono::seconds(2));
  ::unlink(out.c_str());
}

TEST(pipeline_handle, move_iport_repoints_input) {
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(make_unique<CountingDelegate>());

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle s1 = pl.insert_stage("chrono", "src1", {},
                                   R"({"frequency_hz": 10.0})");
  StageHandle s2 = pl.insert_stage("chrono", "src2", {},
                                   R"({"frequency_hz": 10.0})");
  // sink starts wired to src1's oport 0.
  StageHandle sink = pl.insert_stage("shell", "sink", {{s1, 0u}},
                                     R"({"command":"true"})");
  ASSERT_TRUE(s1.valid() && s2.valid() && sink.valid());

  Stage* lsrc1 = live_stage_(pl, "src1");
  Stage* lsrc2 = live_stage_(pl, "src2");
  Stage* lsink = live_stage_(pl, "sink");
  ASSERT_TRUE(lsrc1 && lsrc2 && lsink);
  EXPECT_TRUE(lsink->iport_edges()[0].v == lsrc1);
  EXPECT_TRUE(lsrc1->oport_edges(0).count(OutEdge{lsink, 0}) == 1u);

  // Re-point sink's iport 0 from src1 to src2 (in place).
  EXPECT_TRUE(pl.move_iport("sink", 0, "src2", 0));
  EXPECT_TRUE(lsink->iport_edges()[0].v == lsrc2);
  EXPECT_TRUE(lsrc1->oport_edges(0).empty());
  EXPECT_TRUE(lsrc2->oport_edges(0).count(OutEdge{lsink, 0}) == 1u);
}

TEST(pipeline_handle, move_iport_empty_src_disconnects) {
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(make_unique<CountingDelegate>());

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle s1 = pl.insert_stage("chrono", "src", {},
                                   R"({"frequency_hz": 10.0})");
  StageHandle sink = pl.insert_stage("shell", "sink", {{s1, 0u}},
                                     R"({"command":"true"})");
  ASSERT_TRUE(s1.valid() && sink.valid());

  Stage* lsrc  = live_stage_(pl, "src");
  Stage* lsink = live_stage_(pl, "sink");
  ASSERT_TRUE(lsrc && lsink);
  EXPECT_TRUE(lsrc->oport_edges(0).count(OutEdge{lsink, 0}) == 1u);

  // Empty source disconnects iport 0.
  EXPECT_TRUE(pl.move_iport("sink", 0, "", 0));
  EXPECT_TRUE(lsink->iport_edges()[0].v == nullptr);
  EXPECT_TRUE(lsrc->oport_edges(0).empty());
}

TEST(pipeline_handle, move_iport_rejects_unknown_and_out_of_range) {
  auto delegate = make_unique<CountingDelegate>();
  CountingDelegate* cap = delegate.get();
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(std::move(delegate));

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle s1 = pl.insert_stage("chrono", "src", {},
                                   R"({"frequency_hz": 10.0})");
  StageHandle sink = pl.insert_stage("shell", "sink", {{s1, 0u}},
                                     R"({"command":"true"})");
  ASSERT_TRUE(s1.valid() && sink.valid());

  EXPECT_FALSE(pl.move_iport("nope", 0, "src", 0));   // unknown consumer
  EXPECT_FALSE(pl.move_iport("sink", 5, "src", 0));   // iport out of range
  EXPECT_FALSE(pl.move_iport("sink", 0, "nope", 0));  // unknown source
  EXPECT_FALSE(pl.move_iport("sink", 0, "src", 9));   // src oport oob
  EXPECT_TRUE(cap->warns >= 4u);

  // The original edge survives every rejected move.
  Stage* lsrc  = live_stage_(pl, "src");
  Stage* lsink = live_stage_(pl, "sink");
  EXPECT_TRUE(lsink->iport_edges()[0].v == lsrc);
}

TEST(pipeline_handle, move_iport_refused_while_launched) {
  auto delegate = make_unique<CountingDelegate>();
  CountingDelegate* cap = delegate.get();
  Session sess(make_unique<SilentLogDelegate>());
  sess.set_ui_delegate(std::move(delegate));

  PipelineHandle pl = sess.create_pipeline("p");
  StageHandle s1 = pl.insert_stage("chrono", "src", {},
                                   R"({"frequency_hz": 100.0})");
  StageHandle sink = pl.insert_stage("shell", "sink", {{s1, 0u}},
                                     R"({"command":"true"})");
  ASSERT_TRUE(s1.valid() && sink.valid());

  EXPECT_TRUE(sess.launch_pipeline(pl).code == 0u);
  // Editing a launched pipeline's edges is refused.
  EXPECT_FALSE(pl.move_iport("sink", 0, "src", 0));
  EXPECT_TRUE(cap->warns >= 1u);
  EXPECT_TRUE(sess.stop_pipeline(pl).code == 0u);
}
