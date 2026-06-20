#include "minitest.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/stage-registry.h"
#include "stages/chrono-stage.h"
#include "stages/shell-stage.h"
#include "vpipe/pipeline-handle.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

string
tmp_path_(const char* tag)
{
  string p = "/tmp/vpipe-";
  p += tag;
  p += "-";
  p += to_string(getpid());
  p += ".out";
  return p;
}

size_t
count_lines_(const string& path)
{
  ifstream in(path);
  if (!in) {
    return 0;
  }
  size_t n = 0;
  string line;
  while (getline(in, line)) {
    ++n;
  }
  return n;
}

FlexData
chrono_cfg_(double period_seconds, int64_t count)
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "period_seconds", FlexData::make_real(period_seconds));
  cfg.as_object().insert_or_assign(
      "count", FlexData::make_int(count));
  return cfg;
}

FlexData
shell_cfg_(const string& cmd)
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "command", FlexData::make_string(cmd));
  return cfg;
}

}

// -------- chrono ---------------------------------------------------

TEST(chrono_stage, type_is_registered) {
  EXPECT_TRUE(StageRegistry::get().find_id("chrono") !=
              StageTypeId::unknown);
}

// Construction succeeds for any config; an invalid period spec is
// recorded in config_error() and deferred to launch.
TEST(chrono_stage, missing_period_deferred) {
  Session sess;
  ChronoStage s(&sess, "c", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(chrono_stage, conflicting_hz_and_period_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(10.0));
  cfg.as_object().insert_or_assign(
      "period_seconds", FlexData::make_real(0.1));
  ChronoStage s(&sess, "c", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(chrono_stage, frequency_hz_sets_period) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(20.0));
  ChronoStage s(&sess, "c", {}, std::move(cfg));
  EXPECT_TRUE(s.period() == chrono::milliseconds(50));
  EXPECT_TRUE(s.count() == 0u);
}

TEST(chrono_stage, period_units_stack) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "period_minutes", FlexData::make_real(1.0));
  cfg.as_object().insert_or_assign(
      "period_seconds", FlexData::make_real(30.0));
  ChronoStage s(&sess, "c", {}, std::move(cfg));
  EXPECT_TRUE(s.period() == chrono::seconds(90));
}

// -------- shell ----------------------------------------------------

TEST(shell_stage, type_is_registered) {
  EXPECT_TRUE(StageRegistry::get().find_id("shell") !=
              StageTypeId::unknown);
}

// Construction succeeds with any config; a missing/empty command is
// recorded in config_error() and deferred to launch.
TEST(shell_stage, missing_command_deferred) {
  Session sess;
  ShellStage s(&sess, "sh", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(shell_stage, empty_command_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "command", FlexData::make_string(""));
  ShellStage s(&sess, "sh", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// -------- pipeline -------------------------------------------------

TEST(chrono_shell_pipeline, fast_count_invokes_command_n_times) {
  // 10 ticks at 20Hz; total wall time ~0.5s. Each tick appends one
  // line to a temp file via /bin/sh. After the runtime drains, the
  // line count must equal the tick count.
  Session sess;
  string out = tmp_path_("chrono-shell-fast");
  ::unlink(out.c_str());

  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(20.0));
  ccfg.as_object().insert_or_assign(
      "count", FlexData::make_int(10));

  auto pl = make_unique<Pipeline>("p", &sess);
  auto* src = pl->insert_stage("chrono", "src", {}, std::move(ccfg));
  ASSERT_TRUE(src != nullptr);

  string cmd = "echo tick >> ";
  cmd += out;
  auto* sink = pl->insert_stage("shell", "sink",
                                vector<InEdge>{{src, 0}},
                                shell_cfg_(cmd));
  ASSERT_TRUE(sink != nullptr);

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();

  EXPECT_TRUE(count_lines_(out) == 10);

  auto* shell = dynamic_cast<ShellStage*>(sink);
  ASSERT_TRUE(shell != nullptr);
  EXPECT_TRUE(shell->invocations() == 10u);

  ::unlink(out.c_str());
}

TEST(chrono_shell_pipeline, prints_system_time_5_times_1s_apart) {
  // The reference example: chrono(period=1s, count=5) -> shell(date
  // appended to a temp file). Total wall time ~5s; we accept ±0.5s.
  Session sess;
  string out = tmp_path_("chrono-shell-1s");
  ::unlink(out.c_str());

  string cmd = "date >> ";
  cmd += out;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto* src = pl->insert_stage("chrono", "src", {},
                               chrono_cfg_(1.0, 5));
  ASSERT_TRUE(src != nullptr);
  auto* sink = pl->insert_stage("shell", "sink",
                                vector<InEdge>{{src, 0}},
                                shell_cfg_(cmd));
  ASSERT_TRUE(sink != nullptr);

  auto t0 = chrono::steady_clock::now();
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  auto elapsed = chrono::steady_clock::now() - t0;

  EXPECT_TRUE(count_lines_(out) == 5);

  // Lower bound is the strict one: 5 ticks at 1s each must take at
  // least ~5s. Upper bound is generous to absorb scheduling jitter.
  EXPECT_TRUE(elapsed >= chrono::milliseconds(4800));
  EXPECT_TRUE(elapsed <= chrono::milliseconds(7000));

  ::unlink(out.c_str());
}

TEST(chrono_shell_pipeline, stop_cuts_indefinite_run_short) {
  // count=0 means "run forever". stop() at 200ms must terminate the
  // pipeline quickly even though the chrono stage is mid-sleep.
  Session sess;
  string out = tmp_path_("chrono-shell-stop");
  ::unlink(out.c_str());

  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(2.0));   // 0.5s period
  // count omitted -> indefinite

  string cmd = "echo tick >> ";
  cmd += out;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto* src = pl->insert_stage("chrono", "src", {}, std::move(ccfg));
  ASSERT_TRUE(src != nullptr);
  auto* sink = pl->insert_stage("shell", "sink",
                                vector<InEdge>{{src, 0}},
                                shell_cfg_(cmd));
  ASSERT_TRUE(sink != nullptr);

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  this_thread::sleep_for(chrono::milliseconds(200));

  auto t0 = chrono::steady_clock::now();
  rt.stop();
  auto stop_latency = chrono::steady_clock::now() - t0;

  EXPECT_TRUE(!rt.running());
  // 50ms chunked sleep means stop should land within ~150ms.
  EXPECT_TRUE(stop_latency <= chrono::milliseconds(500));

  ::unlink(out.c_str());
}

// -------- wait_pipelines (Session-level) ---------------------------

// Helper: build a chrono+shell pipeline through SessionIntf so we can
// exercise wait_pipelines against it. Returns the handle. The shell
// command appends to `out_path` so the caller can count completed
// ticks via count_lines_.
namespace {

PipelineHandle
build_chrono_shell_via_session_(SessionIntf&  sess,
                                const string& out_path,
                                double        frequency_hz,
                                int           count)
{
  PipelineHandle h = sess.create_pipeline("p");
  string ccfg = "{\"frequency_hz\": "
              + to_string(frequency_hz)
              + ", \"count\": " + to_string(count) + "}";
  StageHandle src = h.insert_stage("chrono", "src", {}, ccfg);
  string scfg = "{\"command\": \"echo tick >> " + out_path + "\"}";
  StageHandle sink = h.insert_stage(
      "shell", "sink",
      vector<StagePortHandle>{{src, 0u}},
      scfg);
  (void)sink;
  return h;
}

}

TEST(chrono_shell_pipeline,
     wait_pipelines_lets_oneshot_chain_finish)
{
  // Pipeline runs ~250ms (5 ticks at 20Hz). wait_pipelines() must
  // block until every tick has gone through shell -- otherwise the
  // file ends up with < 5 lines because the runtime got torn down
  // mid-flight. This is the failure mode the user observed.
  Session sess;
  string out = tmp_path_("chrono-shell-wait-finish");
  ::unlink(out.c_str());

  auto h = build_chrono_shell_via_session_(sess, out, /*hz*/ 20.0,
                                                       /*count*/ 5);

  auto t0 = chrono::steady_clock::now();
  EXPECT_TRUE(sess.launch_pipeline(h).code == 0u);
  Status w = sess.wait_pipelines(/*timeout_ms*/ -1);
  auto elapsed = chrono::steady_clock::now() - t0;

  EXPECT_TRUE(w.code == 0u);
  EXPECT_TRUE(count_lines_(out) == 5);
  // 5 ticks at 20Hz = 250ms minimum. Allow generous upper bound for
  // shell-fork jitter.
  EXPECT_TRUE(elapsed >= chrono::milliseconds(240));
  EXPECT_TRUE(elapsed <= chrono::milliseconds(3000));

  EXPECT_TRUE(sess.unload_pipeline(h).code == 0u);
  ::unlink(out.c_str());
}

TEST(chrono_shell_pipeline,
     wait_pipelines_timeout_returns_status_4_not_idle)
{
  // count=0 -> chrono ticks forever. wait_pipelines(200ms) must
  // return Status{4} (timeout) since the pipeline never reaches
  // idle on its own. After timeout, stop_pipeline + unload_pipeline
  // tear it down cleanly.
  Session sess;
  string out = tmp_path_("chrono-shell-wait-timeout");
  ::unlink(out.c_str());

  auto h = build_chrono_shell_via_session_(sess, out, /*hz*/ 20.0,
                                                       /*count*/ 0);

  EXPECT_TRUE(sess.launch_pipeline(h).code == 0u);

  auto t0 = chrono::steady_clock::now();
  Status w = sess.wait_pipelines(/*timeout_ms*/ 200);
  auto elapsed = chrono::steady_clock::now() - t0;

  EXPECT_TRUE(w.code == 4u);
  // Timeout latency must be in the right ballpark -- not zero
  // (would mean the wait didn't actually block), not unbounded
  // (would mean the timeout did not fire).
  EXPECT_TRUE(elapsed >= chrono::milliseconds(150));
  EXPECT_TRUE(elapsed <= chrono::milliseconds(1500));

  EXPECT_TRUE(sess.stop_pipeline(h).code   == 0u);
  EXPECT_TRUE(sess.unload_pipeline(h).code == 0u);
  ::unlink(out.c_str());
}

TEST(chrono_shell_pipeline,
     wait_pipelines_with_no_launched_pipeline_is_noop)
{
  // No pipelines launched -> the wait returns Status{0} immediately
  // (no work to wait for, no timeout to elapse).
  Session sess;
  auto t0 = chrono::steady_clock::now();
  Status w = sess.wait_pipelines(/*timeout_ms*/ 10000);
  auto elapsed = chrono::steady_clock::now() - t0;
  EXPECT_TRUE(w.code == 0u);
  EXPECT_TRUE(elapsed <= chrono::milliseconds(100));
}

TEST(chrono_shell_pipeline,
     wait_pipelines_zero_timeout_probes_without_blocking)
{
  // timeout_ms == 0 is documented as a non-blocking probe: it must
  // return immediately (Status{4} if anything's still running).
  Session sess;
  string out = tmp_path_("chrono-shell-wait-probe");
  ::unlink(out.c_str());

  auto h = build_chrono_shell_via_session_(sess, out, /*hz*/ 1.0,
                                                       /*count*/ 0);
  EXPECT_TRUE(sess.launch_pipeline(h).code == 0u);

  auto t0 = chrono::steady_clock::now();
  Status w = sess.wait_pipelines(/*timeout_ms*/ 0);
  auto elapsed = chrono::steady_clock::now() - t0;
  EXPECT_TRUE(w.code == 4u);
  EXPECT_TRUE(elapsed <= chrono::milliseconds(50));

  EXPECT_TRUE(sess.stop_pipeline(h).code   == 0u);
  EXPECT_TRUE(sess.unload_pipeline(h).code == 0u);
  ::unlink(out.c_str());
}
