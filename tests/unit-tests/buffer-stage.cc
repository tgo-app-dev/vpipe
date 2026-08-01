#include "minitest.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/buffer-stage.h"
#include "stages/trigger-beat.h"
#include "tests/unit-tests/payload-types.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Scripted source. `script` is a comma-separated op list, each op
// addressing one of this stage's oports:
//
//   "<port>:<int>"  write an IntPayload to that oport
//   "<port>"        write a TriggerPayload to that oport
//   "sleep"         pause, so a later write is observably later
//
// The whole script runs inside ONE process() call, so its writes are
// strictly ordered: by the time op N+1 is visible to a consumer, op N
// already is. That is what makes the assertions below deterministic
// without a feedback edge -- a trigger written after another trigger
// is never serviced before it.
class ScriptSrc : public TypedStage<ScriptSrc> {
public:
  static constexpr const char* kTypeName = "ut-buf-script-src";

  ScriptSrc(const SessionContextIntf* s, string id,
            vector<InEdge> iports,
            FlexData config = FlexData::make_object())
    : TypedStage<ScriptSrc>(s, std::move(id), std::move(iports),
                            std::move(config))
  {
    unsigned n = 3;
    if (config_ref_().is_object()
        && config_ref_().as_object().contains("num_oports")) {
      n = static_cast<unsigned>(
          config_ref_().as_object().at("num_oports").as_uint(3));
    }
    allocate_oports(n);
    if (config_ref_().is_object()
        && config_ref_().as_object().contains("script")) {
      _script = config_ref_().as_object().at("script").as_string("");
    }
  }

  Job
  process(RuntimeContext& ctx) override
  {
    size_t i = 0;
    while (i < _script.size()) {
      size_t end = _script.find(',', i);
      if (end == string::npos) {
        end = _script.size();
      }
      string op = _script.substr(i, end - i);
      i = end + 1;
      if (op.empty()) {
        continue;
      }
      if (op == "sleep") {
        this_thread::sleep_for(chrono::milliseconds(50));
        continue;
      }
      size_t colon = op.find(':');
      unsigned port = static_cast<unsigned>(stoul(op.substr(0, colon)));
      if (colon == string::npos) {
        co_await ctx.write(port, make_payload<TriggerPayload>());
      } else {
        int v = stoi(op.substr(colon + 1));
        co_await ctx.write(port, make_payload<test::IntPayload>(v));
      }
    }
    ctx.signal_done();
    co_return;
  }

private:
  // config() is protected in Stage; this keeps the ctor readable.
  const FlexData& config_ref_() const { return config(); }

  string _script;
};
VPIPE_REGISTER_STAGE(ScriptSrc)

// 1 oport: writes `count` IntPayloads (value == index + 1) and ends,
// closing its edge. Used to give buffer.in its OWN producer so the
// source can reach EOS while the trigger source keeps running.
class DataSrc : public TypedStage<DataSrc> {
public:
  static constexpr const char* kTypeName = "ut-buf-data-src";

  DataSrc(const SessionContextIntf* s, string id,
          vector<InEdge> iports,
          FlexData config = FlexData::make_object())
    : TypedStage<DataSrc>(s, std::move(id), std::move(iports),
                          std::move(config))
  {
    allocate_oports(1);
  }

  Job
  process(RuntimeContext& ctx) override
  {
    if (_i >= 1) {
      ctx.signal_done();
      co_return;
    }
    ++_i;
    co_await ctx.write(0, make_payload<test::IntPayload>(_i));
  }

private:
  int _i = 0;
};
VPIPE_REGISTER_STAGE(DataSrc)

// Collects every beat that reaches buffer.out.
class IntCollector : public TypedStage<IntCollector> {
public:
  static constexpr const char* kTypeName = "ut-buf-collector";

  IntCollector(const SessionContextIntf* s, string id,
               vector<InEdge> iports,
               FlexData config = FlexData::make_object())
    : TypedStage<IntCollector>(s, std::move(id), std::move(iports),
                               std::move(config)) {}

  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) {
      ctx.signal_done();
      co_return;
    }
    int v = static_cast<const test::IntPayload&>(*p).value;
    if (out_) {
      lock_guard<mutex> lk(*mu_);
      out_->push_back(v);
    }
  }

  // The collector runs on a worker thread, so these are plain statics
  // guarded by the same mutex the test reads under (see call-stage.cc).
  static vector<int>* out_;
  static mutex*       mu_;
};
vector<int>* IntCollector::out_ = nullptr;
mutex*       IntCollector::mu_  = nullptr;
VPIPE_REGISTER_STAGE(IntCollector)

FlexData
script_cfg_(const char* script, unsigned num_oports = 3)
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("script",
                                   FlexData::make_string(script));
  cfg.as_object().insert_or_assign("num_oports",
                                   FlexData::make_uint(num_oports));
  return cfg;
}

// Run `src`(3 oports: 0=in, 1=advance, 2=emit) -> buffer -> collector
// and return everything the collector saw.
vector<int>
run_script_(const char* script)
{
  Session  sess;
  Pipeline pl("buf", &sess);

  Stage* src = pl.insert_stage(make_unique<ScriptSrc>(
      &sess, "src", vector<InEdge>{}, script_cfg_(script)));

  Stage* buf = pl.insert_stage(make_unique<BufferStage>(
      &sess, "buf", vector<InEdge>{{src, 0}, {src, 1}, {src, 2}}));

  vector<int> got;
  mutex       got_mu;
  IntCollector::out_ = &got;
  IntCollector::mu_  = &got_mu;
  pl.insert_stage(make_unique<IntCollector>(
      &sess, "sink", vector<InEdge>{{buf, 0}}));

  PipelineRuntime rt(&pl, &sess);
  if (!rt.launch()) {
    IntCollector::out_ = nullptr;
    IntCollector::mu_  = nullptr;
    return {};
  }
  rt.wait_idle();
  rt.stop();

  IntCollector::out_ = nullptr;
  IntCollector::mu_  = nullptr;
  lock_guard<mutex> lk(got_mu);
  return got;
}

}  // namespace

// ---------------------------------------------------------------------
// The stage registers itself and declares its shape.
// ---------------------------------------------------------------------

TEST(buffer_stage, spec_declares_three_iports_under_control) {
  Session sess;
  BufferStage s(&sess, "b", vector<InEdge>{});
  const StageSpec& sp = s.spec();
  EXPECT_TRUE(sp.type_name == "buffer");
  EXPECT_TRUE(sp.category == StageCategory::Control);
  ASSERT_TRUE(sp.iports.size() == 3u);
  ASSERT_TRUE(sp.oports.size() == 1u);
  EXPECT_TRUE(sp.iports[0].name == "in");
  EXPECT_TRUE(sp.iports[1].name == "advance");
  EXPECT_TRUE(sp.iports[2].name == "emit");
  // Every port is type-erased: the stage buffers any beat.
  EXPECT_TRUE(sp.iports[0].type == nullptr);
  EXPECT_TRUE(sp.oports[0].type == nullptr);
  // The three inputs and the output must NOT collapse into one clock
  // domain -- decoupling those rates is the point of the stage. `out`
  // is paced by the emit trigger, so it shares that group.
  EXPECT_TRUE(sp.iports[0].clock_group != sp.iports[1].clock_group);
  EXPECT_TRUE(sp.iports[1].clock_group != sp.iports[2].clock_group);
  EXPECT_TRUE(sp.oports[0].clock_group == sp.iports[2].clock_group);
}

// ---------------------------------------------------------------------
// Emit sends a COPY, so the held beat can be emitted repeatedly.
// ---------------------------------------------------------------------

TEST(buffer_stage, emit_repeats_the_held_beat) {
  // in:1, advance (buffer=1), emit, emit.
  vector<int> got = run_script_("0:1,1,2,2");
  ASSERT_TRUE(got.size() == 2u);
  EXPECT_TRUE(got[0] == 1);
  EXPECT_TRUE(got[1] == 1);
}

// ---------------------------------------------------------------------
// Advancing faster than emitting drops the un-emitted input.
// ---------------------------------------------------------------------

TEST(buffer_stage, advance_outpacing_emit_drops_input) {
  // in:1, in:2, advance (buffer=1), advance (buffer=2, drops 1), emit.
  vector<int> got = run_script_("0:1,0:2,1,1,2");
  ASSERT_TRUE(got.size() == 1u);
  EXPECT_TRUE(got[0] == 2);
}

TEST(buffer_stage, one_input_beat_consumed_per_advance) {
  // Three inputs but only two advances: the buffer holds the SECOND
  // beat (not the newest), because each advance consumes exactly one.
  vector<int> got = run_script_("0:1,0:2,0:3,1,1,2");
  ASSERT_TRUE(got.size() == 1u);
  EXPECT_TRUE(got[0] == 2);
}

// ---------------------------------------------------------------------
// An emit before the first advance has nothing to send.
// ---------------------------------------------------------------------

TEST(buffer_stage, emit_before_any_advance_sends_nothing) {
  // in:1 is available but never sampled -- no advance trigger at all.
  vector<int> got = run_script_("0:1,2,2");
  EXPECT_TRUE(got.empty());
}

// ---------------------------------------------------------------------
// The advance trigger is a rendezvous: it waits for the input beat.
// Written non-blocking (take-if-queued), this emits nothing instead.
// ---------------------------------------------------------------------

TEST(buffer_stage, advance_waits_for_a_later_input_beat) {
  // advance and emit BOTH arrive before any input; the input lands
  // 50 ms later. The advance must block on it, and the emit that
  // queued behind it must then send it.
  vector<int> got = run_script_("1,2,sleep,0:7");
  ASSERT_TRUE(got.size() == 1u);
  EXPECT_TRUE(got[0] == 7);
}

// ---------------------------------------------------------------------
// The held beat outlives its source: `in` reaching EOS freezes the
// buffer, it does not clear it or end the stage.
// ---------------------------------------------------------------------

TEST(buffer_stage, held_beat_survives_input_eos) {
  Session  sess;
  Pipeline pl("buf", &sess);

  // buffer.in has its own single-beat producer, which closes right
  // after that beat; the triggers keep coming from ScriptSrc.
  Stage* data = pl.insert_stage(make_unique<DataSrc>(
      &sess, "data", vector<InEdge>{}));

  // 2 oports: 0 = advance, 1 = emit (emitted twice, after the
  // advance and after `data` is long gone).
  Stage* trig = pl.insert_stage(make_unique<ScriptSrc>(
      &sess, "trig", vector<InEdge>{},
      script_cfg_("0,sleep,1,1", 2)));

  Stage* buf = pl.insert_stage(make_unique<BufferStage>(
      &sess, "buf",
      vector<InEdge>{{data, 0}, {trig, 0}, {trig, 1}}));

  vector<int> got;
  mutex       got_mu;
  IntCollector::out_ = &got;
  IntCollector::mu_  = &got_mu;
  pl.insert_stage(make_unique<IntCollector>(
      &sess, "sink", vector<InEdge>{{buf, 0}}));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  IntCollector::out_ = nullptr;
  IntCollector::mu_  = nullptr;

  lock_guard<mutex> lk(got_mu);
  ASSERT_TRUE(got.size() == 2u);
  EXPECT_TRUE(got[0] == 1);
  EXPECT_TRUE(got[1] == 1);
}
