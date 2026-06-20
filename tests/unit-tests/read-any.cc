// Exercises RuntimeContext::read_any -- the multi-port "wait on any of
// these iports" suspend used by realtime-vqa to wake on a video frame OR
// a chrono tick. Drives a real pipeline (suspend/resume through the
// Session ThreadPool), so it covers the race-prone wake path: a producer
// resuming the coroutine while await_suspend is still registering the
// other ports, and the exactly-once wake guarantee. (The realtime-vqa
// integration that uses read_any is env-gated on a model, so this is the
// primitive's standalone coverage.)

#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"
#include "tests/unit-tests/payload-types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Source with two oports. Writes n0 beats on oport0 and n1 on oport1 in
// a configurable order, then ends -- which closes BOTH oports (EOS).
class TwoPortSource : public TypedStage<TwoPortSource> {
public:
  static constexpr const char* kTypeName = "ut-read-any-source";
  using TypedStage::TypedStage;

  enum class Order { P0ThenP1, P1ThenP0, Interleaved };
  unsigned n0 = 0;
  unsigned n1 = 0;
  Order    order = Order::P0ThenP1;

  Job
  process(RuntimeContext& ctx) override
  {
    if (_done) {
      ctx.signal_done();          // closes both oports -> probe sees EOS
      co_return;
    }
    _done = true;
    if (order == Order::P1ThenP0) {
      for (unsigned i = 0; i < n1; ++i) {
        co_await ctx.write(1, make_payload<test::UintPayload>(1000 + i));
      }
      for (unsigned i = 0; i < n0; ++i) {
        co_await ctx.write(0, make_payload<test::UintPayload>(i));
      }
    } else if (order == Order::Interleaved) {
      unsigned a = 0, b = 0;
      while (a < n0 || b < n1) {
        if (a < n0) {
          co_await ctx.write(0, make_payload<test::UintPayload>(a));
          ++a;
        }
        if (b < n1) {
          co_await ctx.write(1, make_payload<test::UintPayload>(1000 + b));
          ++b;
        }
      }
    } else {
      for (unsigned i = 0; i < n0; ++i) {
        co_await ctx.write(0, make_payload<test::UintPayload>(i));
      }
      for (unsigned i = 0; i < n1; ++i) {
        co_await ctx.write(1, make_payload<test::UintPayload>(1000 + i));
      }
    }
  }

private:
  bool _done = false;
};
VPIPE_REGISTER_STAGE(TwoPortSource)

// Consumer driven by read_any({0,1}): wakes on whichever port has data
// (or EOS), drains both ports' backlogs, and ends when either port is at
// EOS. Counts per-port beats and the number of wakes. A double-resume of
// the multi-waiter would corrupt counts / crash / hang here.
class ReadAnyProbe : public TypedStage<ReadAnyProbe> {
public:
  static constexpr const char* kTypeName = "ut-read-any-probe";
  using TypedStage::TypedStage;

  std::mutex mu;
  unsigned   got0  = 0;
  unsigned   got1  = 0;
  unsigned   wakes = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    co_await ctx.read_any({0, 1});
    { lock_guard<mutex> lk(mu); ++wakes; }

    const std::uint32_t b0 = ctx.backlog(0);
    for (std::uint32_t i = 0; i < b0; ++i) {
      auto p = co_await ctx.read(0);
      if (!p) { ctx.signal_done(); co_return; }
      { lock_guard<mutex> lk(mu); ++got0; }
    }
    const std::uint32_t b1 = ctx.backlog(1);
    for (std::uint32_t i = 0; i < b1; ++i) {
      auto p = co_await ctx.read(1);
      if (!p) { ctx.signal_done(); co_return; }
      { lock_guard<mutex> lk(mu); ++got1; }
    }
    // End only once BOTH ports are drained AND closed: backlog() is a
    // relaxed under-count, so a wake may read fewer than were written;
    // looping until both are at EOS guarantees every beat is collected
    // (ending on `||` could quit while one port still has unread beats
    // once the other closes). The brief window where one port is closed
    // but the other isn't only re-runs read_any a few times (close()
    // notifies the multi-waiter), never spins indefinitely.
    if (ctx.eos(0) && ctx.eos(1)) { ctx.signal_done(); co_return; }
  }
};
VPIPE_REGISTER_STAGE(ReadAnyProbe)

struct Wired {
  std::unique_ptr<Pipeline> pipeline;
  TwoPortSource*            src;
  ReadAnyProbe*             probe;
};

Wired
build(Session& sess, unsigned n0, unsigned n1,
      TwoPortSource::Order order)
{
  auto pl = std::make_unique<Pipeline>("read-any", &sess);

  auto src_u = std::make_unique<TwoPortSource>(
      &sess, "src", std::vector<InEdge>{});
  src_u->allocate_oports(2);
  src_u->n0    = n0;
  src_u->n1    = n1;
  src_u->order = order;
  auto* src = static_cast<TwoPortSource*>(pl->insert_stage(std::move(src_u)));

  auto probe_u = std::make_unique<ReadAnyProbe>(
      &sess, "probe", std::vector<InEdge>{ { src, 0 }, { src, 1 } });
  auto* probe = static_cast<ReadAnyProbe*>(
      pl->insert_stage(std::move(probe_u)));

  return Wired{ std::move(pl), src, probe };
}

// Launch + drain to completion. Returns whether launch succeeded;
// wait_idle returns only if the probe terminated (so a read_any spin or
// deadlock manifests as a hang, caught by the test harness, not a pass).
// EXPECT_/ASSERT_ live in the TEST bodies (the minitest macros need the
// fixture scope), so the helper only runs.
bool
run_pipeline(Session& sess, Wired& w)
{
  PipelineRuntime rt(w.pipeline.get(), &sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  return true;
}

}  // namespace

// Frames-then-ticks ordering (the realtime-vqa shape): every beat from
// both ports is delivered exactly once.
TEST(read_any, drains_both_ports_p0_then_p1) {
  Session sess;
  auto w = build(sess, 50, 30, TwoPortSource::Order::P0ThenP1);
  ASSERT_TRUE(run_pipeline(sess, w));
  lock_guard<mutex> lk(w.probe->mu);
  EXPECT_TRUE(w.probe->got0 == 50u);
  EXPECT_TRUE(w.probe->got1 == 30u);
}

// Small edge buffers force the source to backpressure and the probe to
// suspend on read_any repeatedly -- stressing register/notify/deregister.
TEST(read_any, drains_both_ports_under_backpressure) {
  Session sess(R"({"pipeline":{"default_edge_capacity":4}})");
  auto w = build(sess, 64, 48, TwoPortSource::Order::Interleaved);
  ASSERT_TRUE(run_pipeline(sess, w));
  lock_guard<mutex> lk(w.probe->mu);
  EXPECT_TRUE(w.probe->got0 == 64u);
  EXPECT_TRUE(w.probe->got1 == 48u);
}

// No data ever arrives on iport0: read_any must wake on iport1.
TEST(read_any, wakes_on_second_port_only) {
  Session sess;
  auto w = build(sess, 0, 25, TwoPortSource::Order::P1ThenP0);
  ASSERT_TRUE(run_pipeline(sess, w));
  lock_guard<mutex> lk(w.probe->mu);
  EXPECT_TRUE(w.probe->got0 == 0u);
  EXPECT_TRUE(w.probe->got1 == 25u);
}

// Interleaved writes across both ports, tight buffers.
TEST(read_any, interleaved_both_ports) {
  Session sess(R"({"pipeline":{"default_edge_capacity":2}})");
  auto w = build(sess, 40, 40, TwoPortSource::Order::Interleaved);
  ASSERT_TRUE(run_pipeline(sess, w));
  lock_guard<mutex> lk(w.probe->mu);
  EXPECT_TRUE(w.probe->got0 == 40u);
  EXPECT_TRUE(w.probe->got1 == 40u);
}

// Source closes without ever writing: read_any must wake on EOS (close
// notifies the multi-waiter) so the probe terminates instead of hanging.
TEST(read_any, terminates_on_eos_with_no_data) {
  Session sess;
  auto w = build(sess, 0, 0, TwoPortSource::Order::P0ThenP1);
  ASSERT_TRUE(run_pipeline(sess, w));
  lock_guard<mutex> lk(w.probe->mu);
  EXPECT_TRUE(w.probe->got0 == 0u);
  EXPECT_TRUE(w.probe->got1 == 0u);
}
