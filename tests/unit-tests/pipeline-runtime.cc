#include "minitest.h"
#include "common/job.h"
#include "common/session.h"
#include "common/thread-pool.h"
#include "common/beat-payload-intf.h"
#include "tests/unit-tests/payload-types.h"
#include "common/oport-policy.h"
#include "common/vertex.h"
#include "stages/passthrough-stage.h"
#include "interfaces/session-services-intf.h"
#include "common/lmdb-env.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"
#include "common/vpipe-format.h"
#include "interfaces/ui-delegate-intf.h"

#include <cstdio>
#include <functional>
#include <string>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

namespace {

// Source: emits N consecutive unsigned tokens then signals done.
class CountingSource : public vpipe::TypedStage<CountingSource> {
public:
  static constexpr const char* kTypeName = "ut-counting-source";
  using TypedStage::TypedStage;

  unsigned target = 0;
  unsigned next   = 0;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (next >= target) {
      ctx.signal_done();
      co_return;
    }
    unsigned v = next++;
    co_await ctx.write(0,
        vpipe::make_payload<vpipe::test::UintPayload>(v));
  }
};
VPIPE_REGISTER_STAGE(CountingSource)

// Sink: collects every received unsigned. Stops on EOS.
class CollectingSink : public vpipe::TypedStage<CollectingSink> {
public:
  static constexpr const char* kTypeName = "ut-collecting-sink";
  using TypedStage::TypedStage;

  std::mutex            mu;
  std::vector<unsigned> received;
  std::chrono::microseconds slow{0};

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
    if (slow.count() > 0) {
      std::this_thread::sleep_for(slow);
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      received.push_back(
          static_cast<const vpipe::test::UintPayload&>(*t).value);
    }
  }
};
VPIPE_REGISTER_STAGE(CollectingSink)

struct Wired {
  std::unique_ptr<vpipe::Pipeline> pipeline;
  CountingSource*                  src;
  vpipe::Stage*                    mid;
  CollectingSink*                  sink;
};

// Build a 3-stage pipeline: src -> passthrough -> sink. The Pipeline
// owns the stages; raw pointers in Wired are non-owning.
// edge_depth == 0 leaves every oport on the default OportPolicy
// (soft-threshold mode: warn at 1024 active Beats, throw at 2048).
// A non-zero value sets a fixed ring depth on each producer, which is
// the only way to get a small buffer -- there is no session-wide knob.
Wired
build_pipeline(vpipe::Session& sess, unsigned target,
               unsigned edge_depth = 0)
{
  auto pl = std::make_unique<vpipe::Pipeline>("p", &sess);

  auto src_u = std::make_unique<CountingSource>(
    &sess, "src", std::vector<vpipe::InEdge>{});
  src_u->allocate_oports(1);
  src_u->target = target;
  auto* src = static_cast<CountingSource*>(
    pl->insert_stage(std::move(src_u)));

  auto mid_u = std::make_unique<vpipe::PassthroughStage>(
    &sess, "mid", std::vector<vpipe::InEdge>{{src, 0}});
  mid_u->allocate_oports(1);
  auto* mid = pl->insert_stage(std::move(mid_u));

  auto sink_u = std::make_unique<CollectingSink>(
    &sess, "sink", std::vector<vpipe::InEdge>{{mid, 0}});
  auto* sink = static_cast<CollectingSink*>(
    pl->insert_stage(std::move(sink_u)));

  if (edge_depth > 0) {
    const vpipe::OportPolicy pol{edge_depth,
                                 vpipe::OverrunPolicy::Backpressure};
    src->set_oport_policy(0, pol);
    mid->set_oport_policy(0, pol);
  }

  return Wired{std::move(pl), src, mid, sink};
}

}

TEST(pipeline_runtime, passthrough_basic) {
  vpipe::Session sess;
  auto w = build_pipeline(sess, 100);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();

  std::lock_guard<std::mutex> lk(w.sink->mu);
  EXPECT_TRUE(w.sink->received.size() == 100);
  bool ordered = true;
  for (size_t i = 0; i < w.sink->received.size(); ++i) {
    if (w.sink->received[i] != static_cast<unsigned>(i)) {
      ordered = false;
      break;
    }
  }
  EXPECT_TRUE(ordered);
}

TEST(pipeline_runtime, backpressure_small_buffer) {
  vpipe::Session sess;
  // Depth 1 on both producers, so a slow sink forces the source to
  // suspend rather than run ahead.
  //
  // The assertion has to be about HOW FAR AHEAD the source got, not
  // about what eventually arrived. "All N items, in order" holds at
  // any depth, so the earlier version of this test passed without ever
  // reaching the backpressure path it names: it selected the depth
  // through a session config key that nothing read, and the
  // assertions could not tell the difference either way.
  auto w = build_pipeline(sess, 400, /*edge_depth=*/1);
  w.sink->slow = std::chrono::microseconds(200);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(rt.launch());

  // Sample mid-run. Unbuffered the source would already be at 400
  // while the sink is a handful of items in; held to depth 1 it can
  // only be a couple of Beats ahead of what the sink has taken.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  unsigned produced = w.src->next;
  size_t   consumed = 0;
  {
    std::lock_guard<std::mutex> lk(w.sink->mu);
    consumed = w.sink->received.size();
  }
  EXPECT_TRUE(produced < 400u);
  EXPECT_TRUE(produced - consumed <= 8u);

  rt.wait_idle();

  std::lock_guard<std::mutex> lk(w.sink->mu);
  EXPECT_TRUE(w.sink->received.size() == 400);
  bool ordered = true;
  for (size_t i = 0; i < w.sink->received.size(); ++i) {
    if (w.sink->received[i] != static_cast<unsigned>(i)) {
      ordered = false;
      break;
    }
  }
  EXPECT_TRUE(ordered);
}

// ---- how long it ran --------------------------------------------------

// The one line a stop leaves behind. It is the only record of how long a
// pipeline was up: a long-lived graph is started once and looked at
// later, and "when did this start" is not in the log by the time anyone
// asks.
//
// Asserted as a COUNT as well as a match, because the two ways to get
// this wrong are both silent -- stop() is reachable from the destructor
// after an explicit stop (two lines for one run) and on a runtime that
// never launched (a line for a run that never happened).
TEST(pipeline_runtime, stop_reports_how_long_it_ran) {
  vpipe::Session sess;
  struct RanLines final : public vpipe::UiDelegateIntf {
    std::mutex               mu;
    std::vector<std::string> ran;
    void error(const vpipe::VpipeFormat&) override {}
    void warn(const vpipe::VpipeFormat&) override {}
    void info(const vpipe::VpipeFormat& f) override {
      const std::string s = f();
      if (s.find("ran for") == std::string::npos) { return; }
      std::lock_guard<std::mutex> lk(mu);
      ran.push_back(s);
    }
    vpipe::UiInputStatus getline(const vpipe::VpipeFormat&, std::string&,
                                 const std::function<bool()>&) override {
      return vpipe::UiInputStatus::Eof;
    }
    std::unique_ptr<vpipe::UiTextStream> open_text_stream() override {
      return std::make_unique<vpipe::NullUiTextStream>();
    }
  };
  auto owned = std::make_unique<RanLines>();
  RanLines* ui = owned.get();
  sess.set_ui_delegate(std::move(owned));

  {
    auto w = build_pipeline(sess, 100);
    vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
    // A runtime that never launched has no run to report, and stop() on
    // one is a normal call -- PipelineHandleImpl makes it on any handle.
    rt.stop();
    {
      std::lock_guard<std::mutex> lk(ui->mu);
      EXPECT_TRUE(ui->ran.empty());
    }
  }

  auto w = build_pipeline(sess, 100);
  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  rt.stop();          // idempotent: the second one ended nothing
  {
    std::lock_guard<std::mutex> lk(ui->mu);
    ASSERT_TRUE(ui->ran.size() == 1);
    const std::string& line = ui->ran.front();
    // It names the pipeline, because a session runs several at once and
    // a bare duration belongs to none of them.
    EXPECT_TRUE(line.find("'p'") != std::string::npos);
    std::printf("[pipeline_runtime] %s\n", line.c_str());
  }
}

// The scale the line has to cover, which is the whole reason it is not
// a bare second count: the same graph is a 40 ms unit test and a camera
// that has been up since spring.
TEST(pipeline_runtime, a_duration_reads_at_every_scale) {
  using vpipe::human_duration;
  EXPECT_TRUE(human_duration(0.0)        == "0 ms");
  EXPECT_TRUE(human_duration(-1.0)       == "0 ms");   // never negative
  EXPECT_TRUE(human_duration(0.0004)     == "0.40 ms");
  EXPECT_TRUE(human_duration(0.847)      == "847 ms");
  EXPECT_TRUE(human_duration(42.34)      == "42.3 s");
  // Rounded ONCE: 59.96 s must not print as "60.0 s".
  EXPECT_TRUE(human_duration(59.96)      == "1 m 00 s");
  EXPECT_TRUE(human_duration(90.0)       == "1 m 30 s");
  EXPECT_TRUE(human_duration(3660.0)     == "1 h 01 m");
  EXPECT_TRUE(human_duration(86400 + 3600.0)     == "1 d 1 h");
  EXPECT_TRUE(human_duration(30.0 * 86400)       == "1 mo 0 d");
  EXPECT_TRUE(human_duration(75.0 * 86400)       == "2 mo 15 d");
  // A year is TWELVE of those months, so the remainder can never read
  // "1 y 12 mo".
  EXPECT_TRUE(human_duration(360.0 * 86400)      == "1 y 0 mo");
  EXPECT_TRUE(human_duration(359.9 * 86400)      == "11 mo 29 d");
  EXPECT_TRUE(human_duration(400.0 * 86400)      == "1 y 1 mo");
}

TEST(pipeline_runtime, stop_mid_stream) {
  vpipe::Session sess;
  // Large target so the source can't finish naturally before stop().
  auto w = build_pipeline(sess, 1000000);
  w.sink->slow = std::chrono::microseconds(100);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(rt.launch());
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  rt.stop();
  EXPECT_TRUE(!rt.running());
  // Source should be far from finished.
  EXPECT_TRUE(w.src->next < 1000000u);
}

TEST(pipeline_runtime, self_completed_when_all_stages_done) {
  vpipe::Session sess;
  auto w = build_pipeline(sess, 100);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(!rt.self_completed());   // not yet launched
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  // Every stage signalled done and the pipeline drained on its own,
  // with no pause()/stop(): self_completed() reports the auto-stop
  // condition. running() is still set until stop() clears it.
  EXPECT_TRUE(rt.self_completed());
  EXPECT_TRUE(rt.running());
  rt.stop();
  EXPECT_TRUE(!rt.self_completed());
  EXPECT_TRUE(!rt.running());
}

TEST(pipeline_runtime, not_self_completed_after_external_stop) {
  vpipe::Session sess;
  // Large target so the source can't finish naturally before stop().
  auto w = build_pipeline(sess, 1000000);
  w.sink->slow = std::chrono::microseconds(100);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(rt.launch());
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  rt.stop();
  // stop() drained every driver (so _completed >= _expected) but this
  // was an external stop, not "all stages signalled done" -- so it is
  // NOT a self-completion.
  EXPECT_TRUE(!rt.self_completed());
}

TEST(pipeline_runtime, eos_propagates_through_passthrough) {
  vpipe::Session sess;
  auto w = build_pipeline(sess, 7);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();

  std::lock_guard<std::mutex> lk(w.sink->mu);
  EXPECT_TRUE(w.sink->received.size() == 7);
}

// edge_buffer_stats() is the introspection feeding the web-ui buffer
// overlay: one labelled entry per edge (consumer iport), with the
// producer/consumer ids + port indices and the live ring depth.
TEST(pipeline_runtime, edge_buffer_stats_labels_and_depth) {
  vpipe::Session sess;
  auto w = build_pipeline(sess, 100);

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  // Nothing before launch builds the buffers.
  EXPECT_TRUE(rt.edge_buffer_stats().empty());

  EXPECT_TRUE(rt.launch());
  // Two edges: src[0] -> mid[0] and mid[0] -> sink[0], each labelled
  // with the real stage ids and a positive ring capacity.
  auto stats = rt.edge_buffer_stats();
  EXPECT_TRUE(stats.size() == 2);
  bool saw_src_mid = false, saw_mid_sink = false;
  for (const auto& e : stats) {
    EXPECT_TRUE(e.capacity > 0);
    if (e.from_id == "src" && e.to_id == "mid"
        && e.from_port == 0 && e.to_port == 0) { saw_src_mid = true; }
    if (e.from_id == "mid" && e.to_id == "sink"
        && e.from_port == 0 && e.to_port == 0) { saw_mid_sink = true; }
  }
  EXPECT_TRUE(saw_src_mid);
  EXPECT_TRUE(saw_mid_sink);

  rt.wait_idle();
  // Fully drained + EOS: every edge reports an empty, closed buffer.
  for (const auto& e : rt.edge_buffer_stats()) {
    EXPECT_TRUE(e.backlog == 0);
    EXPECT_TRUE(e.closed);
  }
}

// edge_buffer_stats() is polled from the web-ui's HTTP thread WHILE the
// driver threads stream. Hammer it during active streaming to prove the
// concurrent read is crash-free and stays coherent (the relaxed/locked
// snapshots never tear into a bad size or null label).
TEST(pipeline_runtime, edge_buffer_stats_concurrent_safe) {
  vpipe::Session sess;
  auto w = build_pipeline(sess, 300000);     // long enough to poll mid-run
  w.sink->slow = std::chrono::microseconds(40);   // keep buffers non-empty

  vpipe::PipelineRuntime rt(w.pipeline.get(), &sess);
  EXPECT_TRUE(rt.launch());

  bool coherent = true;
  for (int i = 0; i < 3000 && rt.running(); ++i) {
    auto stats = rt.edge_buffer_stats();
    if (stats.size() != 2) { coherent = false; break; }
    for (const auto& e : stats) {
      if (e.capacity == 0 || e.from_id.empty() || e.to_id.empty()) {
        coherent = false;
      }
    }
    if (!coherent) { break; }
  }
  rt.stop();
  EXPECT_TRUE(coherent);
}

namespace {

// Stages that record, at the very first call to process(), how many
// stages in the pipeline had already returned from initialize(). The
// runtime guarantees this count equals the total stage count for
// every stage, even if individual initialize() bodies finish at very
// different times.
class InitBarrierProbeSource : public vpipe::TypedStage<InitBarrierProbeSource> {
public:
  static constexpr const char* kTypeName = "ut-initbar-source";
  using TypedStage::TypedStage;

  std::atomic<unsigned>*    init_done_count = nullptr;
  std::atomic<unsigned>     observed{0};
  std::chrono::milliseconds init_delay{0};
  unsigned                  emit_n = 0;
  unsigned                  next   = 0;

  vpipe::Job
  initialize(vpipe::RuntimeContext&) override
  {
    if (init_delay.count() > 0) {
      std::this_thread::sleep_for(init_delay);
    }
    init_done_count->fetch_add(1, std::memory_order_release);
    co_return;
  }

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (observed.load(std::memory_order_acquire) == 0) {
      observed.store(
        init_done_count->load(std::memory_order_acquire),
        std::memory_order_release);
    }
    if (next >= emit_n) {
      ctx.signal_done();
      co_return;
    }
    unsigned v = next++;
    co_await ctx.write(0,
        vpipe::make_payload<vpipe::test::UintPayload>(v));
  }
};
VPIPE_REGISTER_STAGE(InitBarrierProbeSource)

class InitBarrierProbeMid : public vpipe::TypedStage<InitBarrierProbeMid> {
public:
  static constexpr const char* kTypeName = "ut-initbar-mid";
  using TypedStage::TypedStage;

  std::atomic<unsigned>*    init_done_count = nullptr;
  std::atomic<unsigned>     observed{0};
  std::chrono::milliseconds init_delay{0};

  vpipe::Job
  initialize(vpipe::RuntimeContext&) override
  {
    if (init_delay.count() > 0) {
      std::this_thread::sleep_for(init_delay);
    }
    init_done_count->fetch_add(1, std::memory_order_release);
    co_return;
  }

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (observed.load(std::memory_order_acquire) == 0) {
      observed.store(
        init_done_count->load(std::memory_order_acquire),
        std::memory_order_release);
    }
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
    co_await ctx.write(0, std::move(t));
  }
};
VPIPE_REGISTER_STAGE(InitBarrierProbeMid)

class InitBarrierProbeSink : public vpipe::TypedStage<InitBarrierProbeSink> {
public:
  static constexpr const char* kTypeName = "ut-initbar-sink";
  using TypedStage::TypedStage;

  std::atomic<unsigned>*    init_done_count = nullptr;
  std::atomic<unsigned>     observed{0};
  std::chrono::milliseconds init_delay{0};

  vpipe::Job
  initialize(vpipe::RuntimeContext&) override
  {
    if (init_delay.count() > 0) {
      std::this_thread::sleep_for(init_delay);
    }
    init_done_count->fetch_add(1, std::memory_order_release);
    co_return;
  }

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (observed.load(std::memory_order_acquire) == 0) {
      observed.store(
        init_done_count->load(std::memory_order_acquire),
        std::memory_order_release);
    }
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
  }
};
VPIPE_REGISTER_STAGE(InitBarrierProbeSink)

}

// Every stage's first process() call must observe that all 3 stages
// have already returned from initialize(). Without the pipeline-wide
// barrier the fast source would race ahead and start emitting beats
// before the slow sink has finished initialize().
TEST(pipeline_runtime, no_process_before_all_initialize_returns) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("p", &sess);

  std::atomic<unsigned> init_done_count{0};

  auto src_u = std::make_unique<InitBarrierProbeSource>(
    &sess, "src", std::vector<vpipe::InEdge>{});
  src_u->allocate_oports(1);
  src_u->init_done_count = &init_done_count;
  src_u->init_delay      = std::chrono::milliseconds(0);
  src_u->emit_n          = 5;
  auto* src = static_cast<InitBarrierProbeSource*>(
    pl->insert_stage(std::move(src_u)));

  auto mid_u = std::make_unique<InitBarrierProbeMid>(
    &sess, "mid", std::vector<vpipe::InEdge>{{src, 0}});
  mid_u->allocate_oports(1);
  mid_u->init_done_count = &init_done_count;
  mid_u->init_delay      = std::chrono::milliseconds(20);
  auto* mid = static_cast<InitBarrierProbeMid*>(
    pl->insert_stage(std::move(mid_u)));

  auto sink_u = std::make_unique<InitBarrierProbeSink>(
    &sess, "sink", std::vector<vpipe::InEdge>{{mid, 0}});
  sink_u->init_done_count = &init_done_count;
  sink_u->init_delay      = std::chrono::milliseconds(80);
  auto* sink = static_cast<InitBarrierProbeSink*>(
    pl->insert_stage(std::move(sink_u)));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();

  EXPECT_TRUE(
    src->observed.load(std::memory_order_acquire)  == 3u);
  EXPECT_TRUE(
    mid->observed.load(std::memory_order_acquire)  == 3u);
  EXPECT_TRUE(
    sink->observed.load(std::memory_order_acquire) == 3u);
}

// ---------------------------------------------------------------------------
// A malformed graph must FAIL THIS PIPELINE, not the process.
//
// The build steps report faults by throwing (a payload type mismatch between
// two wired ports being the one users hit, from a hand-written .vpipeline or
// an edited layout). An escaping exception reaches std::terminate and aborts
// the host -- for the web-ui, every other pipeline and the operator's session
// with it. launch() therefore catches, warns, and returns false.

namespace {

// Two stages whose declared payload types cannot be wired together: the
// source emits UintPayload, the sink demands a different type.
class TypedSourceStage final : public vpipe::TypedStage<TypedSourceStage> {
public:
  static constexpr const char* kTypeName = "ut-typed-source";
  TypedSourceStage(const vpipe::SessionContextIntf* s, std::string id,
                   std::vector<vpipe::InEdge> in, vpipe::FlexData cfg)
    : TypedStage<TypedSourceStage>(s, std::move(id), std::move(in),
                                   std::move(cfg))
  { allocate_oports(spec().oports.size()); }
  vpipe::Job process(vpipe::RuntimeContext& ctx) override {
    ctx.signal_done();
    co_return;
  }
  const vpipe::StageSpec& spec() const noexcept override;
};
const vpipe::PortSpec kTsOports[] = {
  {.name = "out", .doc = "uint", .type = &typeid(vpipe::test::UintPayload),
   .clock_group = 0},
};
const vpipe::StageSpec kTsSpec = {
  .type_name = "ut-typed-source",
  .doc = "test source with a typed oport",
  .oports = kTsOports,
};
const vpipe::StageSpec& TypedSourceStage::spec() const noexcept
{ return kTsSpec; }
VPIPE_REGISTER_STAGE(TypedSourceStage)
VPIPE_REGISTER_SPEC(TypedSourceStage, kTsSpec)

class TypedSinkStage final : public vpipe::TypedStage<TypedSinkStage> {
public:
  static constexpr const char* kTypeName = "ut-typed-sink";
  TypedSinkStage(const vpipe::SessionContextIntf* s, std::string id,
                 std::vector<vpipe::InEdge> in, vpipe::FlexData cfg)
    : TypedStage<TypedSinkStage>(s, std::move(id), std::move(in),
                                 std::move(cfg)) {}
  vpipe::Job process(vpipe::RuntimeContext& ctx) override {
    auto b = co_await ctx.read(0);
    if (!b) { ctx.signal_done(); }
  }
  const vpipe::StageSpec& spec() const noexcept override;
};
const vpipe::PortSpec kTkIports[] = {
  // IntPayload, NOT the source's UintPayload -- distinct types.
  {.name = "in", .doc = "a DIFFERENT payload type",
   .type = &typeid(vpipe::test::IntPayload), .clock_group = 0},
};
const vpipe::StageSpec kTkSpec = {
  .type_name = "ut-typed-sink",
  .doc = "test sink whose iport type cannot accept the source's",
  .iports = kTkIports,
};
const vpipe::StageSpec& TypedSinkStage::spec() const noexcept
{ return kTkSpec; }
VPIPE_REGISTER_STAGE(TypedSinkStage)
VPIPE_REGISTER_SPEC(TypedSinkStage, kTkSpec)

}  // namespace

TEST(pipeline_runtime, type_mismatch_fails_launch_without_throwing) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("mistyped", &sess);
  auto src_u = std::make_unique<TypedSourceStage>(
      &sess, "src", std::vector<vpipe::InEdge>{},
      vpipe::FlexData::make_object());
  auto* src = pl->insert_stage(std::move(src_u));
  auto sink_u = std::make_unique<TypedSinkStage>(
      &sess, "sink", std::vector<vpipe::InEdge>{{src, 0}},
      vpipe::FlexData::make_object());
  pl->insert_stage(std::move(sink_u));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  bool threw = false;
  bool launched = true;
  try {
    launched = rt.launch();
  } catch (...) {
    threw = true;          // the old behaviour: straight to std::terminate
  }
  EXPECT_FALSE(threw);
  EXPECT_FALSE(launched);
  // A failed launch must leave nothing running, so stop()/destruction is safe
  // and the operator can fix the wiring and try again.
  rt.stop();
}

// A stage that reads an iport it does NOT have must see EOS, not crash.
//
// `vpipe --launch-stage video-to-rgb` builds a one-stage pipeline with no
// edges at all, so the stage has ZERO iports while its process() still does
// `co_await ctx.read(0)`. That indexed past the end of the reader vector and
// dereferenced garbage -- SIGSEGV, taking the whole host down (video-to-rgb,
// audio-to-pcm and temporal-decimation all did it). An absent port now reads
// as permanent EOS, exactly like a declared-but-unwired one, so the stage's
// ordinary `if (!beat) { signal_done(); }` path retires it.
TEST(pipeline_runtime, reading_an_absent_iport_reads_eos) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("no-inputs", &sess);
  // CollectingSink declares no ports of its own and is wired to nothing, so
  // its process() reads iport 0 on a context with an EMPTY reader list.
  auto sink_u = std::make_unique<CollectingSink>(
      &sess, "sink", std::vector<vpipe::InEdge>{},
      vpipe::FlexData::make_object());
  auto* sink = static_cast<CollectingSink*>(
      pl->insert_stage(std::move(sink_u)));
  ASSERT_TRUE(sink != nullptr);

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());        // launching is fine; the read is the bug
  rt.wait_idle();
  rt.stop();
  // Reached EOS immediately and consumed nothing, rather than faulting.
  std::lock_guard<std::mutex> lk(sink->mu);
  EXPECT_TRUE(sink->received.empty());
}

// The same guard on the peek/window path, which stages like
// temporal-decimation use instead of read().
TEST(pipeline_runtime, peeking_an_absent_iport_reads_eos) {
  vpipe::Session sess;
  vpipe::RuntimeContext ctx({}, {}, nullptr);
  EXPECT_TRUE(ctx.num_iports() == 0u);
  EXPECT_FALSE(ctx.iport_connected(0));
  // Out-of-range queries answer as a finished port instead of faulting.
  EXPECT_TRUE(ctx.eos(0));
  EXPECT_TRUE(ctx.backlog(0) == 0u);
  ctx.release_read(0, 1);          // must be a no-op, not a crash
}

// ---- declared service dependencies ----------------------------------
//
// A stage that needs a session service used to discover its absence by
// getting a null pointer from services() partway through initialize(),
// after the launch had already committed. Declaring the dependency moves
// that to a refusal at launch which names the stage and the service.

namespace {

// A service no session provides, so the "missing" case is deterministic
// rather than depending on what this box happens to have.
struct AbsentService {};

// A source that requires it. Everything else about it is ordinary.
class NeedyStage : public vpipe::TypedStage<NeedyStage> {
public:
  static constexpr const char* kTypeName = "ut-needy-stage";
  using TypedStage::TypedStage;
  bool required = true;

  std::vector<vpipe::ServiceReq> declare_services() const override
  {
    return {vpipe::ServiceReq{"vpipe.test.absent/1", required}};
  }
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

// One that requires a service the session really does provide.
class LmdbNeedyStage : public vpipe::TypedStage<LmdbNeedyStage> {
public:
  static constexpr const char* kTypeName = "ut-lmdb-needy-stage";
  using TypedStage::TypedStage;

  std::vector<vpipe::ServiceReq> declare_services() const override
  {
    return {vpipe::require_service<vpipe::LmdbEnv>()};
  }
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

}  // namespace

TEST(pipeline_runtime, missing_required_service_refuses_launch) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("needy", &sess);
  pl->insert_stage(std::make_unique<NeedyStage>(
      &sess, "needy", std::vector<vpipe::InEdge>{},
      vpipe::FlexData::make_object()));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  bool threw = false, launched = true;
  try {
    launched = rt.launch();
  } catch (...) {
    threw = true;
  }
  // Refused, not thrown -- same contract as the type-mismatch check, so
  // the operator can fix the session and try again.
  EXPECT_FALSE(threw);
  EXPECT_FALSE(launched);
  rt.stop();
}

// The same stage declaring the dependency as OPTIONAL launches: absence
// is a fact the stage said it can live with, not a failure.
TEST(pipeline_runtime, optional_service_does_not_block_launch) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("tolerant", &sess);
  auto s_u = std::make_unique<NeedyStage>(
      &sess, "tolerant", std::vector<vpipe::InEdge>{},
      vpipe::FlexData::make_object());
  s_u->required = false;
  pl->insert_stage(std::move(s_u));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.stop();
}

// A requirement the session DOES satisfy resolves by type, through the
// keyed registry, and launches.
TEST(pipeline_runtime, satisfied_service_requirement_launches) {
  vpipe::Session sess;
  // Prove the registry answers for a real service before relying on it.
  EXPECT_TRUE(sess.services()->service<vpipe::LmdbEnv>() != nullptr);
  EXPECT_TRUE(sess.services()->find_service("vpipe.lmdb/1") != nullptr);
  EXPECT_TRUE(sess.services()->find_service("nope/1") == nullptr);

  auto pl = std::make_unique<vpipe::Pipeline>("satisfied", &sess);
  pl->insert_stage(std::make_unique<LmdbNeedyStage>(
      &sess, "reader", std::vector<vpipe::InEdge>{},
      vpipe::FlexData::make_object()));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.stop();
}
