#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "tests/unit-tests/payload-types.h"
#include "common/flex-data.h"
#include "common/graph.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/call-stage.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// ---- Test stages -----------------------------------------------

// 0 iports, configurable oports -- usable as an iport carrier OR
// as a counter source.
class CarrierIn : public TypedStage<CarrierIn> {
public:
  static constexpr const char* kTypeName = "ut-carrier-in";
  CarrierIn(const SessionContextIntf* s, string id,
            vector<InEdge> iports, FlexData config = FlexData::make_object())
    : TypedStage<CarrierIn>(s, std::move(id), std::move(iports),
                            std::move(config)) {}
  Job process(RuntimeContext& ctx) override {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(CarrierIn)

// configurable iports, 0 oports -- usable as an oport carrier or
// as a discarding sink.
class CarrierOut : public TypedStage<CarrierOut> {
public:
  static constexpr const char* kTypeName = "ut-carrier-out";
  CarrierOut(const SessionContextIntf* s, string id,
             vector<InEdge> iports, FlexData config = FlexData::make_object())
    : TypedStage<CarrierOut>(s, std::move(id), std::move(iports),
                             std::move(config)) {}
  Job process(RuntimeContext& ctx) override {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(CarrierOut)

// Source: 0 iports, 1 oport. Emits N integer beats then done.
class CountSrc : public TypedStage<CountSrc> {
public:
  static constexpr const char* kTypeName = "ut-count-src";
  CountSrc(const SessionContextIntf* s, string id,
           vector<InEdge> iports,
           FlexData config = FlexData::make_object())
    : TypedStage<CountSrc>(s, std::move(id), std::move(iports),
                           std::move(config))
  {
    allocate_oports(1);
    if (config.is_object() && config.as_object().contains("count")) {
      _count = static_cast<int>(
          config.as_object().at("count").as_uint(5));
    }
  }
  Job process(RuntimeContext& ctx) override {
    if (_i >= _count) {
      ctx.signal_done();
      co_return;
    }
    co_await ctx.write(0, make_payload<test::IntPayload>(_i));
    ++_i;
  }
private:
  int _count = 5;
  int _i     = 0;
};
VPIPE_REGISTER_STAGE(CountSrc)

// 1 iport, 1 oport. Reads int beat, multiplies by 2, writes.
class Doubler : public TypedStage<Doubler> {
public:
  static constexpr const char* kTypeName = "ut-doubler";
  Doubler(const SessionContextIntf* s, string id,
          vector<InEdge> iports,
          FlexData config = FlexData::make_object())
    : TypedStage<Doubler>(s, std::move(id), std::move(iports),
                          std::move(config))
  {
    allocate_oports(1);
  }
  Job process(RuntimeContext& ctx) override {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
    int v = static_cast<const test::IntPayload&>(*t).value;
    co_await ctx.write(0, make_payload<test::IntPayload>(v * 2));
  }
};
VPIPE_REGISTER_STAGE(Doubler)

// 1 iport, 0 oports. Collects every received beat into a shared
// vector under a mutex. Test reads the vector after the runtime
// stops.
class IntSink : public TypedStage<IntSink> {
public:
  static constexpr const char* kTypeName = "ut-int-sink";
  IntSink(const SessionContextIntf* s, string id,
          vector<InEdge> iports,
          FlexData config = FlexData::make_object())
    : TypedStage<IntSink>(s, std::move(id), std::move(iports),
                          std::move(config)) {}
  Job process(RuntimeContext& ctx) override {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
    int v = static_cast<const test::IntPayload&>(*t).value;
    if (out_target_) {
      lock_guard<mutex> lk(*out_mu_);
      out_target_->push_back(v);
    }
  }
  // Process-global wiring used by the test fixtures. The sink runs
  // on a worker thread, so a thread_local would be invisible from
  // there; plain statics under the consumer-side mutex are fine.
  static vector<int>* out_target_;
  static mutex*       out_mu_;
};
vector<int>* IntSink::out_target_ = nullptr;
mutex*       IntSink::out_mu_     = nullptr;
VPIPE_REGISTER_STAGE(IntSink)

// Suppress cerr noise during expected-failure tests so CI logs
// stay clean.
class CerrCapture {
public:
  CerrCapture() : _saved(cerr.rdbuf()) { cerr.rdbuf(_buf.rdbuf()); }
  ~CerrCapture() { cerr.rdbuf(_saved); }
private:
  streambuf*   _saved;
  stringstream _buf;
};

// Build the {dbl} sub-pipeline:
//   carrier-in "in" (0 iports, 1 oport)
//   doubler   "mul2" (1 iport from in.oport[0], 1 oport)
//   carrier-out "out" (1 iport from mul2.oport[0], 0 oports)
// And assign iports/oports of the sub-pipeline to expose those.
GraphPtr
build_dbl_subpipeline(const SessionContextIntf* sess)
{
  auto pl = make_unique<Pipeline>("dbl", sess);

  auto in_u = make_unique<CarrierIn>(sess, "in",
                                     vector<InEdge>{});
  in_u->allocate_oports(1);
  Stage* in = pl->insert_stage(std::move(in_u));

  Stage* mul2 = pl->insert_stage(make_unique<Doubler>(
      sess, "mul2", vector<InEdge>{{in, 0}}));

  Stage* out = pl->insert_stage(make_unique<CarrierOut>(
      sess, "out", vector<InEdge>{{mul2, 0}}));

  pl->assign_iports(vector<InEdge>{{in,  0}});
  pl->assign_oports(vector<OutEdge>{{out, 0}});
  return pl;
}

}

// ---------------------------------------------------------------------
// Graph::iports() / oports() accessors.
// ---------------------------------------------------------------------

TEST(call_stage, iports_oports_accessors_round_trip) {
  Session sess;
  Pipeline pl("p", &sess);
  auto in_u = make_unique<CarrierIn>(&sess, "in", vector<InEdge>{});
  in_u->allocate_oports(2);
  Stage* in = pl.insert_stage(std::move(in_u));

  Stage* out = pl.insert_stage(make_unique<CarrierOut>(
      &sess, "out", vector<InEdge>{{in, 0}, {in, 1}}));

  pl.assign_iports(vector<InEdge>{{in, 0}, {in, 1}});
  pl.assign_oports(vector<OutEdge>{{out, 0}, {out, 1}});

  EXPECT_TRUE(pl.iports().size() == 2u);
  EXPECT_TRUE(pl.iports()[0].v == in);
  EXPECT_TRUE(pl.iports()[0].p == 0u);
  EXPECT_TRUE(pl.iports()[1].v == in);
  EXPECT_TRUE(pl.iports()[1].p == 1u);

  EXPECT_TRUE(pl.oports().size() == 2u);
  EXPECT_TRUE(pl.oports()[0].v == out);
  EXPECT_TRUE(pl.oports()[0].p == 0u);
  EXPECT_TRUE(pl.oports()[1].v == out);
  EXPECT_TRUE(pl.oports()[1].p == 1u);
}

// ---------------------------------------------------------------------
// lexical_lookup_pipeline.
// ---------------------------------------------------------------------

TEST(call_stage, lexical_lookup_finds_sibling) {
  Session sess;
  Pipeline root("root", &sess);
  auto a_u = make_unique<Pipeline>("A", &sess);
  Pipeline* a = static_cast<Pipeline*>(root.insert_graph(std::move(a_u)));
  auto b_u = make_unique<Pipeline>("B", &sess);
  Pipeline* b = static_cast<Pipeline*>(root.insert_graph(std::move(b_u)));
  // From inside A, "B" is a sibling -- visible.
  EXPECT_TRUE(lexical_lookup_pipeline(a, "B") == b);
  EXPECT_TRUE(lexical_lookup_pipeline(b, "A") == a);
}

TEST(call_stage, lexical_lookup_walks_up_chain) {
  Session sess;
  Pipeline root("root", &sess);
  auto p1_u = make_unique<Pipeline>("P1", &sess);
  Pipeline* p1 = static_cast<Pipeline*>(root.insert_graph(std::move(p1_u)));
  auto p2_u = make_unique<Pipeline>("P2", &sess);
  Pipeline* p2 = static_cast<Pipeline*>(p1->insert_graph(std::move(p2_u)));
  auto util_u = make_unique<Pipeline>("Util", &sess);
  Pipeline* util = static_cast<Pipeline*>(
      root.insert_graph(std::move(util_u)));
  // From inside P2, "Util" is reached by walking up to root.
  EXPECT_TRUE(lexical_lookup_pipeline(p2, "Util") == util);
  // P1 is also visible from P2 (defined alongside Util at root).
  EXPECT_TRUE(lexical_lookup_pipeline(p2, "P1") == p1);
}

TEST(call_stage, lexical_lookup_innermost_wins) {
  Session sess;
  Pipeline root("root", &sess);
  auto outer_x_u = make_unique<Pipeline>("X", &sess);
  Pipeline* outer_x =
      static_cast<Pipeline*>(root.insert_graph(std::move(outer_x_u)));
  auto inner_u = make_unique<Pipeline>("inner", &sess);
  Pipeline* inner =
      static_cast<Pipeline*>(root.insert_graph(std::move(inner_u)));
  auto inner_x_u = make_unique<Pipeline>("X", &sess);
  Pipeline* inner_x = static_cast<Pipeline*>(
      inner->insert_graph(std::move(inner_x_u)));
  // From inner, "X" should resolve to the inner-defined X (own
  // child wins over the outer X).
  EXPECT_TRUE(lexical_lookup_pipeline(inner, "X") == inner_x);
  EXPECT_TRUE(inner_x != outer_x);
}

TEST(call_stage, lexical_lookup_does_not_cross_subtrees) {
  Session sess;
  Pipeline root("root", &sess);
  auto a_u = make_unique<Pipeline>("A", &sess);
  Pipeline* a = static_cast<Pipeline*>(root.insert_graph(std::move(a_u)));
  auto q_u = make_unique<Pipeline>("Q", &sess);
  (void)a->insert_graph(std::move(q_u));
  auto b_u = make_unique<Pipeline>("B", &sess);
  Pipeline* b = static_cast<Pipeline*>(root.insert_graph(std::move(b_u)));
  // Q is a child of A; from B (a sibling of A) Q is not visible.
  EXPECT_TRUE(lexical_lookup_pipeline(b, "Q") == nullptr);
  // From A itself, Q is a child -> visible.
  EXPECT_TRUE(lexical_lookup_pipeline(a, "Q") != nullptr);
}

// ---------------------------------------------------------------------
// End-to-end: launch a pipeline that calls a sub-pipeline.
// ---------------------------------------------------------------------

TEST(call_stage, basic_inline_doubles_through_callee) {
  Session sess;
  Pipeline pl("main", &sess);

  // Sub-pipeline `dbl`: in -> mul2 -> out, exposed via assign_*.
  pl.insert_graph(build_dbl_subpipeline(&sess));

  // Main: src -> call(dbl) -> sink.
  FlexData src_cfg = FlexData::make_object();
  src_cfg.as_object().insert_or_assign(
      "count", FlexData::make_uint(5));
  Stage* src = pl.insert_stage(make_unique<CountSrc>(
      &sess, "src", vector<InEdge>{}, std::move(src_cfg)));

  FlexData call_cfg = FlexData::make_object();
  call_cfg.as_object().insert_or_assign(
      "pipeline", FlexData::make_string("dbl"));
  call_cfg.as_object().insert_or_assign(
      "num_oports", FlexData::make_uint(1));
  Stage* call = pl.insert_stage(make_unique<CallStage>(
      &sess, "callee", vector<InEdge>{{src, 0}}, std::move(call_cfg)));

  vector<int> received;
  mutex      received_mu;
  IntSink::out_target_ = &received;
  IntSink::out_mu_     = &received_mu;
  pl.insert_stage(make_unique<IntSink>(
      &sess, "sink", vector<InEdge>{{call, 0}}));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  // Source emits N beats and signals done; the EOS propagates and
  // every driver reaches final_suspend. wait_idle drains naturally
  // without forcing the source to abort mid-stream.
  rt.wait_idle();
  rt.stop();

  IntSink::out_target_ = nullptr;
  IntSink::out_mu_     = nullptr;

  // Five inputs (0..4) doubled = {0, 2, 4, 6, 8}. Order is
  // deterministic because the graph is a single straight pipe.
  ASSERT_TRUE(received.size() == 5u);
  EXPECT_TRUE(received[0] == 0);
  EXPECT_TRUE(received[1] == 2);
  EXPECT_TRUE(received[2] == 4);
  EXPECT_TRUE(received[3] == 6);
  EXPECT_TRUE(received[4] == 8);
}

TEST(call_stage, unknown_pipeline_rejected) {
  CerrCapture cap;
  Session sess;
  Pipeline pl("main", &sess);
  Stage* src = pl.insert_stage(make_unique<CountSrc>(
      &sess, "src", vector<InEdge>{}));
  FlexData call_cfg = FlexData::make_object();
  call_cfg.as_object().insert_or_assign(
      "pipeline", FlexData::make_string("does-not-exist"));
  call_cfg.as_object().insert_or_assign(
      "num_oports", FlexData::make_uint(1));
  pl.insert_stage(make_unique<CallStage>(
      &sess, "c", vector<InEdge>{{src, 0}}, std::move(call_cfg)));
  PipelineRuntime rt(&pl, &sess);
  EXPECT_FALSE(rt.launch());
}

TEST(call_stage, port_count_mismatch_rejected) {
  CerrCapture cap;
  Session sess;
  Pipeline pl("main", &sess);
  pl.insert_graph(build_dbl_subpipeline(&sess));
  Stage* src = pl.insert_stage(make_unique<CountSrc>(
      &sess, "src", vector<InEdge>{}));
  // dbl has 1 oport; we lie and ask for 2 -> launch must fail.
  FlexData call_cfg = FlexData::make_object();
  call_cfg.as_object().insert_or_assign(
      "pipeline", FlexData::make_string("dbl"));
  call_cfg.as_object().insert_or_assign(
      "num_oports", FlexData::make_uint(2));
  pl.insert_stage(make_unique<CallStage>(
      &sess, "c", vector<InEdge>{{src, 0}}, std::move(call_cfg)));
  PipelineRuntime rt(&pl, &sess);
  EXPECT_FALSE(rt.launch());
}

TEST(call_stage, recursion_rejected) {
  CerrCapture cap;
  Session sess;
  // Build a sub-pipeline that contains a call back to itself --
  // launching its parent must reject (cycle).
  auto inner_u = make_unique<Pipeline>("loop", &sess);
  Pipeline* inner = inner_u.get();
  auto in_u = make_unique<CarrierIn>(&sess, "in", vector<InEdge>{});
  in_u->allocate_oports(1);
  Stage* in = inner->insert_stage(std::move(in_u));
  // call inside the sub-pipeline that names itself
  FlexData c_cfg = FlexData::make_object();
  c_cfg.as_object().insert_or_assign(
      "pipeline", FlexData::make_string("loop"));
  c_cfg.as_object().insert_or_assign(
      "num_oports", FlexData::make_uint(1));
  Stage* recur = inner->insert_stage(make_unique<CallStage>(
      &sess, "recur", vector<InEdge>{{in, 0}}, std::move(c_cfg)));
  Stage* out = inner->insert_stage(make_unique<CarrierOut>(
      &sess, "out", vector<InEdge>{{recur, 0}}));
  inner->assign_iports(vector<InEdge>{{in,  0}});
  inner->assign_oports(vector<OutEdge>{{out, 0}});

  Pipeline parent("parent", &sess);
  parent.insert_graph(std::move(inner_u));

  Stage* src = parent.insert_stage(make_unique<CountSrc>(
      &sess, "src", vector<InEdge>{}));
  FlexData call_cfg = FlexData::make_object();
  call_cfg.as_object().insert_or_assign(
      "pipeline", FlexData::make_string("loop"));
  call_cfg.as_object().insert_or_assign(
      "num_oports", FlexData::make_uint(1));
  parent.insert_stage(make_unique<CallStage>(
      &sess, "kick", vector<InEdge>{{src, 0}}, std::move(call_cfg)));

  PipelineRuntime rt(&parent, &sess);
  EXPECT_FALSE(rt.launch());
}

TEST(call_stage, double_reference_rejected) {
  CerrCapture cap;
  Session sess;
  Pipeline pl("main", &sess);
  pl.insert_graph(build_dbl_subpipeline(&sess));

  Stage* src1 = pl.insert_stage(make_unique<CountSrc>(
      &sess, "src1", vector<InEdge>{}));
  Stage* src2 = pl.insert_stage(make_unique<CountSrc>(
      &sess, "src2", vector<InEdge>{}));

  auto mk_call = [&](string id, Stage* upstream) {
    FlexData c = FlexData::make_object();
    c.as_object().insert_or_assign(
        "pipeline", FlexData::make_string("dbl"));
    c.as_object().insert_or_assign(
        "num_oports", FlexData::make_uint(1));
    return make_unique<CallStage>(
        &sess, std::move(id), vector<InEdge>{{upstream, 0}},
        std::move(c));
  };
  Stage* c1 = pl.insert_stage(mk_call("c1", src1));
  Stage* c2 = pl.insert_stage(mk_call("c2", src2));
  pl.insert_stage(make_unique<CarrierOut>(
      &sess, "sink1", vector<InEdge>{{c1, 0}}));
  pl.insert_stage(make_unique<CarrierOut>(
      &sess, "sink2", vector<InEdge>{{c2, 0}}));

  PipelineRuntime rt(&pl, &sess);
  EXPECT_FALSE(rt.launch());
}
