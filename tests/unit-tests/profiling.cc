#include "minitest.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/session.h"
#include "common/thread-pool.h"
#include "common/vertex.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/stage.h"
#include "vpipe/pipeline-handle.h"
#include "vpipe/status.h"

#include <memory>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace {

string make_tmp_file(string_view ext) {
  auto base = filesystem::temp_directory_path() /
              "vpipe_profiling_test_XXXXXX";
  string tmpl = base.string();
  int fd = mkstemp(tmpl.data());
  if (fd < 0) {
    throw runtime_error("mkstemp failed");
  }
  close(fd);
  string out = tmpl + string(ext);
  rename(tmpl.c_str(), out.c_str());
  return out;
}

// Helper: insert a chrono stage with a real frequency_hz config so
// the ctor doesn't throw.
Stage*
insert_chrono_(Pipeline& pl, string id) {
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(50.0));
  return pl.insert_stage("chrono", std::move(id), {}, std::move(cfg));
}

}

// ----------------------------------------------------------------------
// Session-level profiling state.
// ----------------------------------------------------------------------

TEST(profiling, disabled_by_default) {
  Session sess;
  EXPECT_FALSE(sess.profiling_enabled());
  EXPECT_TRUE(sess.profiling_max_events_per_thread() == 0u);
}

TEST(profiling, enable_zero_capacity_is_rejected) {
  Session sess;
  Status st = sess.enable_profiling(0);
  EXPECT_TRUE(st.code == 1);
  EXPECT_FALSE(sess.profiling_enabled());
}

TEST(profiling, enable_then_disable_round_trip) {
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(64).code == 0);
  EXPECT_TRUE(sess.profiling_enabled());
  EXPECT_TRUE(sess.profiling_max_events_per_thread() == 64u);

  EXPECT_TRUE(sess.disable_profiling().code == 0);
  EXPECT_FALSE(sess.profiling_enabled());
  EXPECT_TRUE(sess.profiling_max_events_per_thread() == 0u);
}

// ----------------------------------------------------------------------
// Stage::record_perf_event routes via the session.
// ----------------------------------------------------------------------

TEST(profiling, record_routes_to_overflow_buffer_when_off_pool) {
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(8).code == 0);

  Pipeline pl("p", &sess);
  Stage* s = insert_chrono_(pl, "src");
  ASSERT_TRUE(s != nullptr);

  // We're on the test's main thread (not a pool worker), so the
  // event lands in the overflow buffer at index num_workers.
  s->record_perf_event(1, 100);
  s->record_perf_event(2, 200);

  // No direct accessor from outside; verify via dump round-trip.
  string path = make_tmp_file(".bin");
  EXPECT_TRUE(sess.dump_profiling(path).code == 0);

  ifstream in(path, ios::binary);
  ostringstream buf; buf << in.rdbuf();
  FlexData root = FlexData::from_binary(buf.str());
  ASSERT_TRUE(root.is_object());
  auto root_obj = root.as_object();
  ASSERT_TRUE(root_obj.contains("threads"));
  FlexData threads_v = root_obj.at("threads");
  ASSERT_TRUE(threads_v.is_array());
  auto threads = threads_v.as_array();

  // Find the overflow entry (is_overflow = true) and check counts.
  bool saw_overflow = false;
  for (size_t i = 0; i < threads.size(); ++i) {
    FlexData entry_v = threads.at(i);
    auto entry = entry_v.as_object();
    if (entry.contains("is_overflow") &&
        entry.at("is_overflow").get_bool()) {
      saw_overflow = true;
      EXPECT_TRUE(entry.at("events_count").get_uint() == 2u);
      break;
    }
  }
  EXPECT_TRUE(saw_overflow);
  filesystem::remove(path);
}

TEST(profiling, record_routes_to_worker_buffer_when_inside_job) {
  // A coroutine running on a real pool worker writes into THAT
  // worker's buffer, not the overflow buffer.
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(8).code == 0);

  Pipeline pl("p", &sess);
  Stage* s = insert_chrono_(pl, "src");
  ASSERT_TRUE(s != nullptr);

  promise<unsigned> wid_p;
  auto wid_fut = wid_p.get_future();
  ThreadPool* pool = sess.thread_pool();
  ASSERT_TRUE(pool != nullptr);

  auto job = [](Stage* stage, ThreadPool* p,
                promise<unsigned> pp) -> Job {
    pp.set_value(p->worker_id_of_current_thread());
    stage->record_perf_event(7, 777);
    co_return;
  }(s, pool, std::move(wid_p));
  pool->schedule(job.handle());
  ASSERT_TRUE(wid_fut.wait_for(chrono::seconds(2)) ==
              future_status::ready);
  unsigned wid = wid_fut.get();
  EXPECT_TRUE(wid < pool->num_workers());

  // Dump and verify the worker buffer at index `wid` got the event.
  string path = make_tmp_file(".bin");
  EXPECT_TRUE(sess.dump_profiling(path).code == 0);
  ifstream in(path, ios::binary);
  ostringstream buf; buf << in.rdbuf();
  FlexData root = FlexData::from_binary(buf.str());
  auto root_obj = root.as_object();
  FlexData threads_v = root_obj.at("threads");
  auto threads = threads_v.as_array();
  // num_workers worker lanes + 1 overflow + the auxiliary lanes
  // (LLM, ANE). Worker lanes stay first, so threads.at(wid) is valid.
  ASSERT_TRUE(threads.size()
              == pool->num_workers() + 1u + vpipe::kPerfAuxLaneCount);

  FlexData entry_v = threads.at(wid);
  auto entry = entry_v.as_object();
  EXPECT_FALSE(entry.at("is_overflow").get_bool());
  EXPECT_TRUE(entry.at("events_count").get_uint() == 1u);
  FlexData events_v = entry.at("events");
  auto events = events_v.as_object();
  FlexData type_arr_v  = events.at("type");
  FlexData value_arr_v = events.at("value");
  EXPECT_TRUE(type_arr_v.as_array().size() == 1u);
  EXPECT_TRUE(type_arr_v.as_array().at(0).get_uint() == 7u);
  EXPECT_TRUE(value_arr_v.as_array().at(0).get_uint() == 777u);
  filesystem::remove(path);
}

TEST(profiling, aux_lanes_appear_with_labels_and_synthetic_stages) {
  // record_perf_event_aux routes to dedicated lanes (LLM / ANE) by lane
  // id rather than the calling thread, and the dump labels them + emits
  // synthetic stage entries naming the LLM activities. No model needed.
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(8).code == 0);

  // An LLM text-prefill begin/end (synthetic gvid) on the LLM lane.
  sess.record_perf_event_aux(vpipe::kPerfLaneLLM, vpipe::kGvidLlmPrefill,
                             vpipe::kPerfLlmPrefillBegin, 5);
  sess.record_perf_event_aux(vpipe::kPerfLaneLLM, vpipe::kGvidLlmPrefill,
                             vpipe::kPerfLlmPrefillBegin + 1u, 0);
  // An ANE predict begin/end (a real-ish stage gvid) on the ANE lane.
  sess.record_perf_event_aux(vpipe::kPerfLaneANE, 42u,
                             vpipe::kPerfAnePredictBegin, 0);
  sess.record_perf_event_aux(vpipe::kPerfLaneANE, 42u,
                             vpipe::kPerfAnePredictBegin + 1u, 0);

  string path = make_tmp_file(".bin");
  EXPECT_TRUE(sess.dump_profiling(path).code == 0);
  ifstream in(path, ios::binary);
  ostringstream buf; buf << in.rdbuf();
  FlexData root = FlexData::from_binary(buf.str());
  auto root_obj = root.as_object();

  // Lanes: an "LLM" and an "ANE" labelled lane, each holding 2 events.
  FlexData threads_v = root_obj.at("threads");
  auto threads = threads_v.as_array();
  bool saw_llm = false, saw_ane = false;
  for (size_t i = 0; i < threads.size(); ++i) {
    FlexData entry_v = threads.at(i);
    auto entry = entry_v.as_object();
    if (!entry.contains("label")) { continue; }
    string lbl(entry.at("label").as_string(""));
    if (lbl == "LLM") {
      saw_llm = true;
      EXPECT_TRUE(entry.at("events_count").get_uint() == 2u);
    } else if (lbl == "ANE") {
      saw_ane = true;
      EXPECT_TRUE(entry.at("events_count").get_uint() == 2u);
    }
  }
  EXPECT_TRUE(saw_llm);
  EXPECT_TRUE(saw_ane);

  // Synthetic stage entry naming the prefill activity "text-prefill".
  FlexData stages_v = root_obj.at("stages");
  auto stages = stages_v.as_array();
  bool saw_prefill = false;
  for (size_t i = 0; i < stages.size(); ++i) {
    FlexData s_v = stages.at(i);
    auto s = s_v.as_object();
    if (s.contains("gvid")
        && s.at("gvid").get_uint() == vpipe::kGvidLlmPrefill) {
      saw_prefill = true;
      EXPECT_TRUE(string(s.at("id").as_string("")) == "text-prefill");
    }
  }
  EXPECT_TRUE(saw_prefill);
  filesystem::remove(path);
}

TEST(profiling, aux_scope_records_value_on_begin_and_end) {
  // PerfAuxScope records its payload (e.g. a token count) on BOTH the
  // begin and the end event, so a viewer can derive a rate (tok/s).
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(8).code == 0);
  {
    vpipe::PerfAuxScope p(&sess, vpipe::kPerfLaneLLM,
                          vpipe::kGvidLlmPrefill,
                          vpipe::kPerfLlmPrefillBegin, 42);
  }  // begin(42) then end(42) recorded on destruction

  string path = make_tmp_file(".bin");
  EXPECT_TRUE(sess.dump_profiling(path).code == 0);
  ifstream in(path, ios::binary);
  ostringstream buf; buf << in.rdbuf();
  FlexData root = FlexData::from_binary(buf.str());
  auto root_obj = root.as_object();

  // Locate the LLM lane and confirm both of its events carry value 42.
  FlexData threads_v = root_obj.at("threads");
  auto threads = threads_v.as_array();
  bool checked = false;
  for (size_t i = 0; i < threads.size(); ++i) {
    FlexData entry_v = threads.at(i);
    auto entry = entry_v.as_object();
    if (!entry.contains("label")
        || string(entry.at("label").as_string("")) != "LLM") {
      continue;
    }
    FlexData events_v = entry.at("events");
    auto events = events_v.as_object();
    FlexData value_arr_v = events.at("value");
    auto values = value_arr_v.as_array();
    ASSERT_TRUE(values.size() == 2u);
    EXPECT_TRUE(values.at(0).get_uint() == 42u);   // begin
    EXPECT_TRUE(values.at(1).get_uint() == 42u);   // end
    checked = true;
  }
  EXPECT_TRUE(checked);
  filesystem::remove(path);
}

TEST(profiling, record_when_disabled_is_a_noop) {
  Session sess;
  Pipeline pl("p", &sess);
  Stage* s = insert_chrono_(pl, "src");
  ASSERT_TRUE(s != nullptr);

  // Profiling never enabled. record_perf_event must be safe and
  // dump_profiling must produce an empty per-thread array.
  s->record_perf_event(1, 100);

  string path = make_tmp_file(".bin");
  EXPECT_TRUE(sess.dump_profiling(path).code == 0);
  ifstream in(path, ios::binary);
  ostringstream buf; buf << in.rdbuf();
  FlexData root = FlexData::from_binary(buf.str());
  auto root_obj = root.as_object();
  EXPECT_FALSE(root_obj.at("enabled").get_bool());
  EXPECT_TRUE(root_obj.at("threads").as_array().size() == 0u);
  filesystem::remove(path);
}

TEST(profiling, dump_writes_json_and_binary) {
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(8).code == 0);
  string path_bin  = make_tmp_file(".bin");
  string path_json = make_tmp_file(".json");
  EXPECT_TRUE(sess.dump_profiling(path_bin).code  == 0);
  EXPECT_TRUE(sess.dump_profiling(path_json).code == 0);

  ifstream in_bin(path_bin, ios::binary);
  ostringstream buf_bin; buf_bin << in_bin.rdbuf();
  FlexData root_bin = FlexData::from_binary(buf_bin.str());
  ASSERT_TRUE(root_bin.is_object());
  auto root_obj = root_bin.as_object();
  EXPECT_TRUE(root_obj.contains("anchor_steady_ns"));
  EXPECT_TRUE(root_obj.contains("anchor_realtime_ns"));
  EXPECT_TRUE(root_obj.contains("max_events_per_thread"));
  EXPECT_TRUE(root_obj.contains("enabled"));
  EXPECT_TRUE(root_obj.contains("num_workers"));
  EXPECT_TRUE(root_obj.contains("threads"));
  EXPECT_TRUE(root_obj.contains("stages"));
  EXPECT_TRUE(root_obj.at("max_events_per_thread").get_uint() == 8u);

  ifstream in_json(path_json);
  ostringstream buf_json; buf_json << in_json.rdbuf();
  string data_json = buf_json.str();
  EXPECT_TRUE(!data_json.empty() && data_json[0] == '{');
  FlexData root_json = FlexData::from_json(data_json);
  EXPECT_TRUE(root_json.is_object());

  filesystem::remove(path_bin);
  filesystem::remove(path_json);
}

TEST(profiling, custom_perf_event_name_is_callable) {
  Session sess;
  Pipeline pl("p", &sess);
  Stage* s = insert_chrono_(pl, "src");
  ASSERT_TRUE(s != nullptr);
  EXPECT_TRUE(s->perf_event_name(0)  == "event_0");
  EXPECT_TRUE(s->perf_event_name(42) == "event_42");
}

// Reserved runtime event ids map to readable schedule/unschedule
// labels, and follow the even-begin / odd-end pairing convention.
TEST(profiling, runtime_perf_event_names) {
  Session sess;
  Pipeline pl("p", &sess);
  Stage* s = insert_chrono_(pl, "src");
  ASSERT_TRUE(s != nullptr);
  EXPECT_TRUE(s->perf_event_name(kPerfSchedule) == "schedule");
  EXPECT_TRUE(s->perf_event_name(kPerfUnschedule) == "unschedule");
  // schedule is an even begin; unschedule is its odd end.
  EXPECT_FALSE(perf_type_is_end(kPerfSchedule));
  EXPECT_TRUE(perf_type_is_end(kPerfUnschedule));
  EXPECT_TRUE(perf_type_begin_of(kPerfUnschedule) == kPerfSchedule);
}

// The ThreadPool worker loop auto-emits a schedule/unschedule pair
// around each resume-slice of a LAUNCHED pipeline. Drive a short
// chrono -> shell pipeline with profiling on, then assert the schedule
// and unschedule events landed in the per-worker buffers AND that each
// worker buffer alternates schedule, unschedule, schedule, ... -- the
// property that was broken when the begin/end were emitted by the
// driver around process() (which can migrate workers mid-slice).
TEST(profiling, worker_emits_clean_schedule_unschedule_pairs) {
  Session sess;
  EXPECT_TRUE(sess.enable_profiling(1024).code == 0);

  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(200.0));   // ~5ms/tick
  ccfg.as_object().insert_or_assign(
      "count", FlexData::make_int(4));                // auto-drains

  auto pl = make_unique<Pipeline>("p", &sess);
  Stage* src = pl->insert_stage("chrono", "src", {}, std::move(ccfg));
  ASSERT_TRUE(src != nullptr);
  FlexData scfg = FlexData::make_object();
  scfg.as_object().insert_or_assign(
      "command", FlexData::make_string("true"));      // harmless no-op
  Stage* sink = pl->insert_stage("shell", "sink",
                                 vector<InEdge>{{src, 0}}, std::move(scfg));
  ASSERT_TRUE(sink != nullptr);

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();   // all drivers at final_suspend -> all pairs closed

  string path = make_tmp_file(".bin");
  EXPECT_TRUE(sess.dump_profiling(path).code == 0);
  ifstream in(path, ios::binary);
  ostringstream buf; buf << in.rdbuf();
  FlexData root = FlexData::from_binary(buf.str());
  auto root_obj = root.as_object();

  FlexData threads_v = root_obj.at("threads");
  auto threads = threads_v.as_array();
  size_t sched = 0, unsched = 0;
  bool alternates = true;
  for (size_t i = 0; i < threads.size(); ++i) {
    FlexData entry_v = threads.at(i);
    auto entry = entry_v.as_object();
    FlexData ev_v = entry.at("events");
    auto ev = ev_v.as_object();
    FlexData ty_v = ev.at("type");
    auto ty = ty_v.as_array();
    // Per buffer, scheduling events (in append/time order) must strictly
    // alternate schedule -> unschedule: a worker records schedule, runs
    // the coroutine, then records unschedule before the next dequeue.
    bool expect_sched = true;
    for (size_t k = 0; k < ty.size(); ++k) {
      uint32_t t = static_cast<uint32_t>(ty.at(k).get_uint());
      if (t == kPerfSchedule) {
        ++sched;
        if (!expect_sched) { alternates = false; }
        expect_sched = false;
      } else if (t == kPerfUnschedule) {
        ++unsched;
        if (expect_sched) { alternates = false; }
        expect_sched = true;
      }
    }
    // Each buffer must end on a closed pair (no dangling schedule).
    if (!expect_sched) { alternates = false; }
  }
  EXPECT_TRUE(sched > 0);
  EXPECT_TRUE(unsched > 0);
  EXPECT_TRUE(sched == unsched);   // every begin has its end
  EXPECT_TRUE(alternates);          // clean S,U,S,U per worker

  sess.disable_profiling();
  filesystem::remove(path);
}
