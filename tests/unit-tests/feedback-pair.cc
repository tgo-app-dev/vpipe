// End-to-end tests for the {feedback-tx, feedback-rx} pair: the
// lookup-by-name wiring, the lag-by-one-beat data flow, and the
// pipeline-runtime clock-domain validation that rejects launches
// where the pair would span clock domains.

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/feedback-rx-stage.h"
#include "pipeline/feedback-tx-stage.h"
#include "stages/passthrough-stage.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "tests/unit-tests/payload-types.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

namespace {

// Source: emits N consecutive ints then signals done.
class FbCountingSource : public vpipe::TypedStage<FbCountingSource> {
public:
  static constexpr const char* kTypeName = "ut-fb-counting-source";
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
    int v = static_cast<int>(next++);
    co_await ctx.write(0,
        vpipe::make_payload<vpipe::test::IntPayload>(v));
  }
};
VPIPE_REGISTER_STAGE(FbCountingSource)

// Sink: collects every received int.
class FbCollectingSink : public vpipe::TypedStage<FbCollectingSink> {
public:
  static constexpr const char* kTypeName = "ut-fb-collecting-sink";
  using TypedStage::TypedStage;

  std::mutex       mu;
  std::vector<int> received;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
    int v = static_cast<const vpipe::test::IntPayload&>(*t).value;
    {
      std::lock_guard<std::mutex> lk(mu);
      received.push_back(v);
    }
  }
};
VPIPE_REGISTER_STAGE(FbCollectingSink)

// Loop driver: 1 iport (from tx), 1 oport (forward to chat side).
// On round 0 it fires unconditionally (the analog of text-input's
// `present_first_without_beat=true`). On subsequent rounds it reads
// the feedback beat from tx, records the relayed int, and decides
// whether to emit another beat or stop. Stops after `target` rounds.
class FbLoopDriver : public vpipe::TypedStage<FbLoopDriver> {
public:
  static constexpr const char* kTypeName = "ut-fb-loop-driver";
  using TypedStage::TypedStage;

  int target = 0;
  int next   = 0;

  std::mutex       mu;
  std::vector<int> received;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    const bool first = (next == 0);
    if (!first) {
      auto trig = co_await ctx.read(0);
      if (!trig) {
        ctx.signal_done();
        co_return;
      }
      int v = static_cast<const vpipe::test::IntPayload&>(*trig).value;
      {
        std::lock_guard<std::mutex> lk(mu);
        received.push_back(v);
      }
    }
    if (next >= target) {
      ctx.signal_done();
      co_return;
    }
    int v = next++;
    co_await ctx.write(0,
        vpipe::make_payload<vpipe::test::IntPayload>(v));
  }
};
VPIPE_REGISTER_STAGE(FbLoopDriver)

}

// End-to-end loop topology:
//   tx -> loop_driver -> chat (passthrough) -> rx
//                |                              |
//                +------ (config "from: rx") --+
//
// loop_driver fires N beats (values 0..N-1) into chat -> rx; rx
// caches each; tx wakes up and re-emits the cached payload into
// loop_driver's iport; loop_driver records the relayed value. After
// the last beat round-trips, loop_driver signals done and the chain
// shuts down. The recorded list on loop_driver should be {0, ..., N-2}
// (the last beat triggers loop_driver to stop before reading it back).
TEST(feedback_pair, tx_relays_rx_cached_beats) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("p", &sess);

  // tx first (so we can reference it as the iport source of driver).
  vpipe::FlexData tx_cfg = vpipe::FlexData::make_object();
  tx_cfg.as_object().insert("from",
      vpipe::FlexData::make_string("rx"));
  auto tx_u = std::make_unique<vpipe::FeedbackTxStage>(
    &sess, "tx", std::vector<vpipe::InEdge>{},
    std::move(tx_cfg));
  auto* tx = pl->insert_stage(std::move(tx_u));

  auto drv_u = std::make_unique<FbLoopDriver>(
    &sess, "driver", std::vector<vpipe::InEdge>{{tx, 0}});
  drv_u->allocate_oports(1);
  drv_u->target = 5;
  auto* drv = static_cast<FbLoopDriver*>(
    pl->insert_stage(std::move(drv_u)));

  auto chat_u = std::make_unique<vpipe::PassthroughStage>(
    &sess, "chat", std::vector<vpipe::InEdge>{{drv, 0}});
  chat_u->allocate_oports(1);
  auto* chat = pl->insert_stage(std::move(chat_u));

  vpipe::FlexData rx_cfg = vpipe::FlexData::make_object();
  auto rx_u = std::make_unique<vpipe::FeedbackRxStage>(
    &sess, "rx", std::vector<vpipe::InEdge>{{chat, 0}},
    std::move(rx_cfg));
  pl->insert_stage(std::move(rx_u));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();

  std::lock_guard<std::mutex> lk(drv->mu);
  EXPECT_TRUE(drv->received.size() == 5);
  bool ordered = drv->received.size() == 5;
  for (size_t i = 0; i < drv->received.size(); ++i) {
    if (drv->received[i] != static_cast<int>(i)) {
      ordered = false;
      break;
    }
  }
  EXPECT_TRUE(ordered);
}

// When config.from names a stage that does not exist, the runtime
// must refuse to launch.
TEST(feedback_pair, missing_rx_refuses_launch) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("p", &sess);

  auto src_u = std::make_unique<FbCountingSource>(
    &sess, "src", std::vector<vpipe::InEdge>{});
  src_u->allocate_oports(1);
  src_u->target = 1;
  auto* src = static_cast<FbCountingSource*>(
    pl->insert_stage(std::move(src_u)));

  // No rx stage in the pipeline. tx names "rx-missing".
  vpipe::FlexData rx_cfg = vpipe::FlexData::make_object();
  auto rx_u = std::make_unique<vpipe::FeedbackRxStage>(
    &sess, "rx-here", std::vector<vpipe::InEdge>{{src, 0}},
    std::move(rx_cfg));
  pl->insert_stage(std::move(rx_u));

  vpipe::FlexData tx_cfg = vpipe::FlexData::make_object();
  tx_cfg.as_object().insert("from",
      vpipe::FlexData::make_string("rx-missing"));
  auto tx_u = std::make_unique<vpipe::FeedbackTxStage>(
    &sess, "tx", std::vector<vpipe::InEdge>{},
    std::move(tx_cfg));
  auto* tx = pl->insert_stage(std::move(tx_u));

  auto sink_u = std::make_unique<FbCollectingSink>(
    &sess, "sink", std::vector<vpipe::InEdge>{{tx, 0}});
  pl->insert_stage(std::move(sink_u));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(!rt.launch());
}

namespace {

// A stage that puts iport and oport into DIFFERENT clock groups so a
// feedback pair spanning it ends up in distinct clock domains and the
// runtime check rejects the launch.
class FbClockCrosser
  : public vpipe::TypedStage<FbClockCrosser>
{
public:
  static constexpr const char* kTypeName = "ut-fb-clock-crosser";
  using TypedStage::TypedStage;

  unsigned
  iport_clock_group(unsigned /*p*/) const noexcept override
  { return 0; }

  unsigned
  oport_clock_group(unsigned /*p*/) const noexcept override
  { return 1; }

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
    co_await ctx.write(0, std::move(t));
  }
};
VPIPE_REGISTER_STAGE(FbClockCrosser)

}

// Wire the feedback pair across a stage that puts its iport and oport
// on distinct clock groups; the pair then straddles two clock domains
// and the runtime must reject the launch.
TEST(feedback_pair, spans_clock_domains_refused) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("p", &sess);

  // src(domain A) -> crosser.iport(A) ; crosser.oport(B) -> rx(B)
  auto src_u = std::make_unique<FbCountingSource>(
    &sess, "src", std::vector<vpipe::InEdge>{});
  src_u->allocate_oports(1);
  src_u->target = 1;
  auto* src = static_cast<FbCountingSource*>(
    pl->insert_stage(std::move(src_u)));

  auto cross_u = std::make_unique<FbClockCrosser>(
    &sess, "cross", std::vector<vpipe::InEdge>{{src, 0}});
  cross_u->allocate_oports(1);
  auto* cross = pl->insert_stage(std::move(cross_u));

  vpipe::FlexData rx_cfg = vpipe::FlexData::make_object();
  auto rx_u = std::make_unique<vpipe::FeedbackRxStage>(
    &sess, "rx", std::vector<vpipe::InEdge>{{cross, 0}},
    std::move(rx_cfg));
  pl->insert_stage(std::move(rx_u));

  // tx.oport feeds a sink that lives in domain A (connected to the
  // src side). Concretely, we wire a passthrough fed by tx into a
  // sink with iport_clock_group 0; the tx.oport then lands in
  // domain 0 while rx.iport landed in domain 1.
  vpipe::FlexData tx_cfg = vpipe::FlexData::make_object();
  tx_cfg.as_object().insert("from",
      vpipe::FlexData::make_string("rx"));
  auto tx_u = std::make_unique<vpipe::FeedbackTxStage>(
    &sess, "tx", std::vector<vpipe::InEdge>{},
    std::move(tx_cfg));
  auto* tx = pl->insert_stage(std::move(tx_u));

  auto sink_u = std::make_unique<FbCollectingSink>(
    &sess, "sink", std::vector<vpipe::InEdge>{{tx, 0}});
  pl->insert_stage(std::move(sink_u));

  vpipe::PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(!rt.launch());
}
