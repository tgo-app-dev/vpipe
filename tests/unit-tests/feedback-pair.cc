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
#include "common/vpipe-format.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/feedback-rx-stage.h"
#include "pipeline/feedback-tx-stage.h"
#include "stages/audio-video/temporal-slice-stage.h"
#include "stages/passthrough-stage.h"
#include "stages/text-input-stage.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "tests/unit-tests/payload-types.h"
#include "apple-silicon/tensor-beat.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
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

  // Per-launch reset, mirroring what real loop drivers (text-input)
  // do: stage objects survive a stop/relaunch, so round-counting
  // state must not leak into the next run.
  vpipe::Job
  initialize(vpipe::RuntimeContext&) override
  {
    next = 0;
    {
      std::lock_guard<std::mutex> lk(mu);
      received.clear();
    }
    co_return;
  }

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

// RELAUNCH: the stage objects survive a stop (only the runtime dies),
// so the pair must reset its cross-run state in initialize(). A run
// that ended (rx saw EOS at drain) used to leave rx._eos sticky --
// on the next launch tx's wait_new_beat resolved instantly on the
// stale flag, saw no new beat, signalled done, and the closed trigger
// edge EOS-cascaded the whole loop ("auto-stopped: all stages done"
// before the first round). Build the loop once, run it to completion
// TWICE over the same Pipeline, and require the second run to relay
// the full round count again.
TEST(feedback_pair, relaunch_after_stop_runs_again) {
  vpipe::Session sess;
  auto pl = std::make_unique<vpipe::Pipeline>("p", &sess);

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

  auto run_once = [&]() -> size_t {
    vpipe::PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) {
      return static_cast<size_t>(-1);
    }
    // Bounded wait: a relaunch regression that deadlocks (instead of
    // auto-stopping) must fail the test, not hang the suite.
    const bool idle = rt.wait_idle(20000);
    rt.stop();
    if (!idle) {
      return static_cast<size_t>(-1);
    }
    std::lock_guard<std::mutex> lk(drv->mu);
    return drv->received.size();
  };

  const size_t first = run_once();
  EXPECT_TRUE(first == 5);
  // Same Pipeline, same stage objects, fresh runtime: the loop must
  // run all 5 rounds again (0 = the stale-EOS instant shutdown).
  const size_t second = run_once();
  EXPECT_TRUE(second == 5);
}

namespace {

// UI delegate that answers every getline immediately (no stdin) and
// counts the calls -- lets a text-input-driven loop run headless.
class ScriptedUi final : public vpipe::UiDelegateIntf {
public:
  std::atomic<int> calls{0};

  void error(const vpipe::VpipeFormat&) override {}
  void warn(const vpipe::VpipeFormat&) override {}
  void info(const vpipe::VpipeFormat&) override {}
  vpipe::UiInputStatus
  getline(const vpipe::VpipeFormat&, std::string& out,
          const std::function<bool()>&) override
  {
    ++calls;
    out = "hello";
    return vpipe::UiInputStatus::Ok;
  }
  std::unique_ptr<vpipe::UiTextStream> open_text_stream() override
  {
    return std::make_unique<vpipe::NullUiTextStream>();
  }
};

}

// The reported chat-loop topology, relaunched: {feedback-tx ->
// text-input -> chat(passthrough) -> feedback-rx} with count=2 per
// run. The first run reads two lines and self-terminates; the second
// launch over the SAME stage objects must read two more. Before the
// per-launch resets (rx._eos / tx._last_seen / text-input's
// _first_round_seen + _emitted) the relaunch showed no prompt and
// auto-stopped instantly with zero reads.
TEST(feedback_pair, text_input_chat_loop_relaunches) {
  vpipe::Session sess;
  auto ui_owned = std::make_unique<ScriptedUi>();
  ScriptedUi* ui = ui_owned.get();
  sess.set_ui_delegate(std::move(ui_owned));

  auto pl = std::make_unique<vpipe::Pipeline>("chat", &sess);

  vpipe::FlexData tx_cfg = vpipe::FlexData::make_object();
  tx_cfg.as_object().insert("from",
      vpipe::FlexData::make_string("frx"));
  auto tx_u = std::make_unique<vpipe::FeedbackTxStage>(
    &sess, "ftx", std::vector<vpipe::InEdge>{},
    std::move(tx_cfg));
  auto* tx = pl->insert_stage(std::move(tx_u));

  vpipe::FlexData tin_cfg = vpipe::FlexData::make_object();
  tin_cfg.as_object().insert("count", vpipe::FlexData::make_int(2));
  auto tin_u = std::make_unique<vpipe::TextInputStage>(
    &sess, "tin", std::vector<vpipe::InEdge>{{tx, 0}},
    std::move(tin_cfg));
  auto* tin = pl->insert_stage(std::move(tin_u));

  auto chat_u = std::make_unique<vpipe::PassthroughStage>(
    &sess, "chat", std::vector<vpipe::InEdge>{{tin, 0}});
  chat_u->allocate_oports(1);
  auto* chat = pl->insert_stage(std::move(chat_u));

  vpipe::FlexData rx_cfg = vpipe::FlexData::make_object();
  auto rx_u = std::make_unique<vpipe::FeedbackRxStage>(
    &sess, "frx", std::vector<vpipe::InEdge>{{chat, 0}},
    std::move(rx_cfg));
  pl->insert_stage(std::move(rx_u));

  auto run_once = [&]() -> bool {
    vpipe::PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) {
      return false;
    }
    const bool idle = rt.wait_idle(20000);
    rt.stop();
    return idle;
  };

  EXPECT_TRUE(run_once());
  EXPECT_TRUE(ui->calls == 2);
  // Relaunch over the same stage objects: two MORE prompts must be
  // presented and read (0 new reads = the stale-state instant stop;
  // a wait_idle timeout = the startup deadlock variant).
  EXPECT_TRUE(run_once());
  EXPECT_TRUE(ui->calls == 4);
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

namespace {

// The shape of a real feedback LOOP, minus the models: the beat this
// stage sends to the rx is caused by the beat it read from the tx, so
// on round one there is nothing behind the edge and nobody can go
// first. That is the deadlock `prime` exists to break, and it is the
// only reason a graph like
//
//   video-ref-encoder -> generate-video -> vae-decode -> feedback-rx
//   feedback-tx -------> video-ref-encoder (a reference port)
//
// cannot start.
class FbLoopRelay : public vpipe::TypedStage<FbLoopRelay> {
public:
  static constexpr const char* kTypeName = "ut-fb-loop-relay";
  using TypedStage::TypedStage;

  std::mutex mu;
  int  rounds  = 0;
  int  limit   = 3;
  std::vector<bool> saw_empty;   // was round k's inbound beat empty?

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    bool done = false;
    {
      std::lock_guard<std::mutex> lk(mu);
      bool empty = true;
      if (const auto* tb =
              dynamic_cast<const vpipe::TensorBeatPayload*>(in.get())) {
        empty = tb->shape.empty() || tb->byte_size() == 0;
      }
      saw_empty.push_back(empty);
      ++rounds;
      done = rounds >= limit;
    }
    if (done) {
      // Stop WITHOUT writing: closing the outputs and then writing to
      // them is not a thing a stage may do.
      ctx.signal_done();
      co_return;
    }
    // The round's "output": a NON-empty tensor, so the next round can
    // tell a relayed beat from a primed one.
    // Clip-shaped, so a frames-mode temporal-slice in the loop has a
    // leading axis to index: [2, 3, 1, 1].
    vpipe::TensorBeat out;
    out.dtype = vpipe::TensorBeat::DType::U8;
    out.shape = {2, 3, 1, 1};
    out.data.assign(6, (std::uint8_t)7);
    // Bound to its own local BEFORE the suspend. Building the payload
    // inside the co_await argument leaves the temporary's lifetime
    // tangled with the frame layout across the resume -- the crash
    // audio-to-pcm's emit_chunk_ carries a comment about.
    auto payload = vpipe::make_payload<vpipe::TensorBeatPayload>(
        std::move(out));
    co_await ctx.write(0, std::move(payload));
  }

  const vpipe::StageSpec&
  spec() const noexcept override
  {
    static const vpipe::PortSpec ip[] = {
      {.name = "in", .doc = "", .type = nullptr, .clock_group = 0}};
    static const vpipe::PortSpec op[] = {
      {.name = "out", .doc = "", .type = nullptr, .clock_group = 0}};
    static const vpipe::StageSpec sp = {
        .type_name = "ut-fb-loop-relay", .doc = "", .display_name = "",
        .iports = ip, .oports = op};
    return sp;
  }
};
VPIPE_REGISTER_STAGE(FbLoopRelay)

// Build tx -> relay -> rx (the pair itself is wired BY NAME, so the
// graph stays a DAG) and run it. Returns the relay so the caller can
// see how far the loop actually got.
struct LoopRun {
  vpipe::Session sess;
  std::unique_ptr<vpipe::Pipeline> pl;
  FbLoopRelay* relay = nullptr;
  bool launched = false;
  bool went_idle = false;
};

void
drive_loop_(LoopRun& r, const char* prime, int limit, int timeout_ms)
{
  using namespace vpipe;
  r.pl = std::make_unique<Pipeline>("loop", &r.sess);

  auto tcfg = FlexData::make_object();
  tcfg.as_object().insert_or_assign("from", FlexData::make_string("rx"));
  if (prime != nullptr) {
    tcfg.as_object().insert_or_assign("prime", FlexData::make_string(prime));
  }
  auto tx_u = std::make_unique<FeedbackTxStage>(
      &r.sess, "tx", std::vector<InEdge>{}, tcfg);
  auto* tx = static_cast<FeedbackTxStage*>(r.pl->insert_stage(std::move(tx_u)));

  auto relay_u = std::make_unique<FbLoopRelay>(
      &r.sess, "relay", std::vector<InEdge>{{tx, 0}}, FlexData::make_object());
  relay_u->limit = limit;
  r.relay = static_cast<FbLoopRelay*>(r.pl->insert_stage(std::move(relay_u)));
  // A writing stage has to be given its oports explicitly; TypedStage
  // does not infer them for a test stage, and the rx wired below reads
  // this one's oport 0.
  r.relay->allocate_oports(1);

  auto rx_u = std::make_unique<FeedbackRxStage>(
      &r.sess, "rx", std::vector<InEdge>{{r.relay, 0}},
      FlexData::make_object());
  r.pl->insert_stage(std::move(rx_u));

  PipelineRuntime rt(r.pl.get(), &r.sess);
  r.launched = rt.launch();
  if (!r.launched) { return; }
  r.went_idle = rt.wait_idle(timeout_ms);
  rt.stop();
}

}  // namespace

// WITHOUT priming the loop cannot start. This is the deadlock stated as
// a test rather than as a comment: the relay's beat is what the rx
// receives, and the tx waits for the rx, so round one never happens.
TEST(feedback_pair, an_unprimed_loop_never_starts)
{
  LoopRun r;
  drive_loop_(r, nullptr, /*limit=*/3, /*timeout_ms=*/400);
  ASSERT_TRUE(r.launched);
  if (r.relay == nullptr) { return; }
  std::lock_guard<std::mutex> lk(r.relay->mu);
  std::printf("[feedback_pair] unprimed loop: %d round(s), idle=%d\n",
              r.relay->rounds, (int)r.went_idle);
  EXPECT_TRUE(r.relay->rounds == 0);
  // And it does not merely run slowly -- it never finishes.
  EXPECT_FALSE(r.went_idle);
}

// WITH priming it starts, and the primed beat is EMPTY -- the declared
// "nothing this time" that video-ref-encoder already skips, so the
// consumer needs no first-round special case.
TEST(feedback_pair, a_primed_loop_starts_and_the_first_beat_is_empty)
{
  LoopRun r;
  drive_loop_(r, "empty-tensor", /*limit=*/3, /*timeout_ms=*/4000);
  ASSERT_TRUE(r.launched);
  if (r.relay == nullptr) { return; }
  std::lock_guard<std::mutex> lk(r.relay->mu);
  std::printf("[feedback_pair] primed loop: %d round(s), idle=%d\n",
              r.relay->rounds, (int)r.went_idle);
  EXPECT_TRUE(r.went_idle);
  EXPECT_TRUE(r.relay->rounds == 3);
  ASSERT_TRUE(r.relay->saw_empty.size() >= 2);
  if (r.relay->saw_empty.size() >= 2) {
    // Round 1 is the primed beat...
    EXPECT_TRUE(r.relay->saw_empty[0]);
    // ... and every round after it is REAL feedback, not more priming.
    // A stage that primed every round would pass the first assertion
    // and fail this one.
    for (std::size_t i = 1; i < r.relay->saw_empty.size(); ++i) {
      EXPECT_FALSE(r.relay->saw_empty[i]);
    }
  }
}

// A misspelt kind is refused at CONFIG. Defaulting it to "none" would
// deadlock the loop the setting exists to start, with the config
// LOOKING right -- the worst of the available failures.
TEST(feedback_pair, an_unknown_prime_kind_is_refused)
{
  vpipe::Session sess;
  auto cfg = vpipe::FlexData::make_object();
  cfg.as_object().insert_or_assign("from", vpipe::FlexData::make_string("rx"));
  cfg.as_object().insert_or_assign("prime",
                                   vpipe::FlexData::make_string("empty"));
  auto tx = std::make_unique<vpipe::FeedbackTxStage>(
      &sess, "tx", std::vector<vpipe::InEdge>{}, cfg);
  EXPECT_FALSE(tx->config_error().empty());

  // The two spellings that ARE accepted.
  for (const char* ok : {"none", "empty-tensor"}) {
    auto c = vpipe::FlexData::make_object();
    c.as_object().insert_or_assign("from", vpipe::FlexData::make_string("rx"));
    c.as_object().insert_or_assign("prime", vpipe::FlexData::make_string(ok));
    auto t = std::make_unique<vpipe::FeedbackTxStage>(
        &sess, "tx", std::vector<vpipe::InEdge>{}, c);
    EXPECT_TRUE(t->config_error().empty());
  }
}

namespace {

// A stage that changes the beat RATE, declared the way temporal-slice
// declares it: iport in clock group 0, oport in group 1. The analyzer
// unifies connected port-groups, so it does NOT unify across this one
// -- which is the whole point of the declaration.
class FbRateChanger : public vpipe::TypedStage<FbRateChanger> {
public:
  static constexpr const char* kTypeName = "ut-fb-rate-changer";
  using TypedStage::TypedStage;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    auto out = in->clone();
    co_await ctx.write(0, std::move(out));
  }

  const vpipe::StageSpec&
  spec() const noexcept override
  {
    static const vpipe::PortSpec ip[] = {
      {.name = "in", .doc = "", .type = nullptr, .clock_group = 0}};
    static const vpipe::PortSpec op[] = {
      {.name = "out", .doc = "", .type = nullptr, .clock_group = 1}};
    static const vpipe::StageSpec sp = {
        .type_name = "ut-fb-rate-changer", .doc = "", .display_name = "",
        .iports = ip, .oports = op};
    return sp;
  }
};
VPIPE_REGISTER_STAGE(FbRateChanger)

}  // namespace

// WHAT PRIMING DOES NOT FIX, stated as a test so the limitation is not
// something a graph author discovers at launch.
//
// `prime` removes the "nobody can go first" deadlock. It does NOT relax
// the rule that a feedback pair lives inside ONE clock domain, and a
// loop that picks ONE beat out of many -- `temporal-slice` selecting
// each clip's last frame, which is the obvious way to feed the next
// generation -- changes the rate and so splits the domain.
//
// So the frame-accurate version of cross-segment continuity needs more
// than priming: either the selection has to happen without a rate
// change, or the feedback has to be taken from a port that is already
// in the consumer's domain.
TEST(feedback_pair, priming_does_not_lift_the_one_clock_domain_rule)
{
  using namespace vpipe;
  Session sess;
  auto pl = std::make_unique<Pipeline>("loop", &sess);

  auto tcfg = FlexData::make_object();
  tcfg.as_object().insert_or_assign("from", FlexData::make_string("rx"));
  tcfg.as_object().insert_or_assign("prime",
                                    FlexData::make_string("empty-tensor"));
  auto tx_u = std::make_unique<FeedbackTxStage>(
      &sess, "tx", std::vector<InEdge>{}, tcfg);
  auto* tx = static_cast<FeedbackTxStage*>(pl->insert_stage(std::move(tx_u)));

  auto relay_u = std::make_unique<FbLoopRelay>(
      &sess, "relay", std::vector<InEdge>{{tx, 0}}, FlexData::make_object());
  auto* relay =
      static_cast<FbLoopRelay*>(pl->insert_stage(std::move(relay_u)));
  relay->allocate_oports(1);

  // The rate change sits between the loop body and the rx, exactly
  // where a `temporal-slice` picking the last frame of each clip would.
  auto rc_u = std::make_unique<FbRateChanger>(
      &sess, "slice", std::vector<InEdge>{{relay, 0}}, FlexData::make_object());
  auto* rc = static_cast<FbRateChanger*>(pl->insert_stage(std::move(rc_u)));
  rc->allocate_oports(1);

  auto rx_u = std::make_unique<FeedbackRxStage>(
      &sess, "rx", std::vector<InEdge>{{rc, 0}}, FlexData::make_object());
  pl->insert_stage(std::move(rx_u));

  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  std::printf("[feedback_pair] primed loop across a rate change: "
              "launched=%d (expected 0)\n", (int)launched);
  EXPECT_FALSE(launched);
  if (launched) { rt.wait_idle(200); rt.stop(); }
}

// THE PAYOFF, and the counterpart of the refusal above.
//
// `temporal-slice` in `sequence: frames` is one beat in and one beat
// out, so it reports the iport's clock group and the analyzer unifies
// them. Put in the same place the rate changer was refused -- between
// the loop body and the rx, picking the last frame of each clip -- it
// LAUNCHES, and the primed loop runs through it.
//
// That is the whole cross-segment-continuity wiring in miniature:
// generate -> decode a clip -> slice its last frame -> feed it back as
// the next round's reference, with `prime` supplying round one.
TEST(feedback_pair, a_frames_mode_slice_keeps_the_loop_in_one_domain)
{
  using namespace vpipe;
  Session sess;
  auto pl = std::make_unique<Pipeline>("loop", &sess);

  auto tcfg = FlexData::make_object();
  tcfg.as_object().insert_or_assign("from", FlexData::make_string("rx"));
  tcfg.as_object().insert_or_assign("prime",
                                    FlexData::make_string("empty-tensor"));
  auto tx_u = std::make_unique<FeedbackTxStage>(
      &sess, "tx", std::vector<InEdge>{}, tcfg);
  auto* tx = static_cast<FeedbackTxStage*>(pl->insert_stage(std::move(tx_u)));

  auto relay_u = std::make_unique<FbLoopRelay>(
      &sess, "relay", std::vector<InEdge>{{tx, 0}}, FlexData::make_object());
  relay_u->limit = 3;
  auto* relay =
      static_cast<FbLoopRelay*>(pl->insert_stage(std::move(relay_u)));
  relay->allocate_oports(1);

  // The last frame of each clip, as a STILL -- exactly what a reference
  // port wants, and the shape a one-frame clip would get wrong.
  auto scfg = FlexData::make_object();
  scfg.as_object().insert_or_assign("sequence",
                                    FlexData::make_string("frames"));
  scfg.as_object().insert_or_assign("start", FlexData::make_int(-1));
  scfg.as_object().insert_or_assign("squeeze", FlexData::make_bool(true));
  auto sl_u = std::make_unique<TemporalSliceStage>(
      &sess, "slice", std::vector<InEdge>{{relay, 0}}, scfg);
  auto* sl =
      static_cast<TemporalSliceStage*>(pl->insert_stage(std::move(sl_u)));

  auto rx_u = std::make_unique<FeedbackRxStage>(
      &sess, "rx", std::vector<InEdge>{{sl, 0}}, FlexData::make_object());
  pl->insert_stage(std::move(rx_u));

  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  std::printf("[feedback_pair] primed loop through a frames-mode slice: "
              "launched=%d (expected 1)\n", (int)launched);
  EXPECT_TRUE(launched);
  if (!launched) { return; }
  const bool idle = rt.wait_idle(4000);
  rt.stop();

  std::lock_guard<std::mutex> lk(relay->mu);
  std::printf("[feedback_pair] ... ran %d round(s), idle=%d\n",
              relay->rounds, (int)idle);
  EXPECT_TRUE(idle);
  EXPECT_TRUE(relay->rounds == 3);
  // Round one is the primed empty beat; the rounds after it carry the
  // sliced still, so the loop really did turn over.
  ASSERT_TRUE(relay->saw_empty.size() >= 2);
  if (relay->saw_empty.size() >= 2) {
    EXPECT_TRUE(relay->saw_empty[0]);
    EXPECT_FALSE(relay->saw_empty[1]);
  }
  EXPECT_TRUE(sl->emitted() >= 1);
}
