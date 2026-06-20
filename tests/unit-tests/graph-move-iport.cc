#include "minitest.h"
#include "common/graph.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"

#include <memory>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Plain vertex with a configurable iport list; exercises the generic
// Graph::move_iport_to primitive without the Pipeline guard layered on
// top.
struct TestVertex : public Vertex {
  TestVertex(const SessionContextIntf* s, string id, vector<InEdge> in)
    : Vertex(s, std::move(id), std::move(in))
  {}
};

// Stage with one output port, used to exercise Pipeline's running /
// needs-init semantics on top of the move primitive.
class MoveTestStage : public TypedStage<MoveTestStage> {
public:
  static constexpr const char* kTypeName = "move-test";
  MoveTestStage(const SessionContextIntf* s, string id,
                vector<InEdge> in, FlexData cfg)
    : TypedStage(s, std::move(id), std::move(in), std::move(cfg))
  {
    allocate_oports(1);
  }

  Job
  process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(MoveTestStage)

}

// ---- Generic Graph::move_iport_to mechanics -------------------------

TEST(graph_move_iport, connect_attaches_fanout_and_updates_head)
{
  Session s;
  Graph g("g", &s);

  auto src_u = make_unique<TestVertex>(&s, "src", vector<InEdge>{});
  TestVertex* src = src_u.get();
  src->allocate_oports(1);
  g.insert_vertex(std::move(src_u));

  auto dst_u =
      make_unique<TestVertex>(&s, "dst", vector<InEdge>{ InEdge{nullptr, 0} });
  TestVertex* dst = dst_u.get();
  g.insert_vertex(std::move(dst_u));

  // Initially dst has a null iport -> it's a head vertex.
  EXPECT_TRUE(g.head_vertices().count(dst) == 1);
  EXPECT_TRUE(dst->iport_edges()[0].v == nullptr);

  EXPECT_TRUE(g.move_iport_to(dst, 0, InEdge{src, 0}));

  EXPECT_TRUE(dst->iport_edges()[0].v == src);
  EXPECT_TRUE(dst->iport_edges()[0].p == 0);
  EXPECT_TRUE(src->oport_edges(0).count(OutEdge{dst, 0}) == 1);
  EXPECT_TRUE(src->oport_edges(0).size() == 1);
  // dst now has a wired source -> no longer a head vertex.
  EXPECT_TRUE(g.head_vertices().count(dst) == 0);
}

TEST(graph_move_iport, disconnect_to_null_detaches_and_rejoins_head)
{
  Session s;
  Graph g("g", &s);

  auto src_u = make_unique<TestVertex>(&s, "src", vector<InEdge>{});
  TestVertex* src = src_u.get();
  src->allocate_oports(1);
  g.insert_vertex(std::move(src_u));

  auto dst_u =
      make_unique<TestVertex>(&s, "dst", vector<InEdge>{ InEdge{src, 0} });
  TestVertex* dst = dst_u.get();
  g.insert_vertex(std::move(dst_u));

  // Committed with a real source: fanout present, not a head.
  EXPECT_TRUE(src->oport_edges(0).count(OutEdge{dst, 0}) == 1);
  EXPECT_TRUE(g.head_vertices().count(dst) == 0);

  // Default new_src is null -> disconnect.
  EXPECT_TRUE(g.move_iport_to(dst, 0));

  EXPECT_TRUE(dst->iport_edges()[0].v == nullptr);
  EXPECT_TRUE(src->oport_edges(0).empty());
  EXPECT_TRUE(g.head_vertices().count(dst) == 1);
}

TEST(graph_move_iport, repoint_moves_fanout_between_sources)
{
  Session s;
  Graph g("g", &s);

  auto a_u = make_unique<TestVertex>(&s, "a", vector<InEdge>{});
  TestVertex* a = a_u.get();
  a->allocate_oports(1);
  g.insert_vertex(std::move(a_u));

  auto b_u = make_unique<TestVertex>(&s, "b", vector<InEdge>{});
  TestVertex* b = b_u.get();
  b->allocate_oports(1);
  g.insert_vertex(std::move(b_u));

  auto dst_u =
      make_unique<TestVertex>(&s, "dst", vector<InEdge>{ InEdge{a, 0} });
  TestVertex* dst = dst_u.get();
  g.insert_vertex(std::move(dst_u));

  EXPECT_TRUE(a->oport_edges(0).count(OutEdge{dst, 0}) == 1);

  EXPECT_TRUE(g.move_iport_to(dst, 0, InEdge{b, 0}));

  EXPECT_TRUE(a->oport_edges(0).empty());
  EXPECT_TRUE(b->oport_edges(0).count(OutEdge{dst, 0}) == 1);
  EXPECT_TRUE(dst->iport_edges()[0].v == b);
}

TEST(graph_move_iport, out_of_range_iport_is_rejected)
{
  Session s;
  Graph g("g", &s);

  auto src_u = make_unique<TestVertex>(&s, "src", vector<InEdge>{});
  TestVertex* src = src_u.get();
  src->allocate_oports(1);
  g.insert_vertex(std::move(src_u));

  auto dst_u =
      make_unique<TestVertex>(&s, "dst", vector<InEdge>{ InEdge{nullptr, 0} });
  TestVertex* dst = dst_u.get();
  g.insert_vertex(std::move(dst_u));

  EXPECT_FALSE(g.move_iport_to(dst, 5, InEdge{src, 0}));
  // Unchanged.
  EXPECT_TRUE(dst->iport_edges()[0].v == nullptr);
  EXPECT_TRUE(src->oport_edges(0).empty());
}

TEST(graph_move_iport, source_outside_graph_is_rejected)
{
  Session s;
  Graph g("g", &s);

  auto dst_u =
      make_unique<TestVertex>(&s, "dst", vector<InEdge>{ InEdge{nullptr, 0} });
  TestVertex* dst = dst_u.get();
  g.insert_vertex(std::move(dst_u));

  // Never inserted -> graph() == nullptr.
  TestVertex outsider(&s, "out", vector<InEdge>{});
  outsider.allocate_oports(1);

  EXPECT_FALSE(g.move_iport_to(dst, 0, InEdge{&outsider, 0}));
  EXPECT_TRUE(dst->iport_edges()[0].v == nullptr);
}

TEST(graph_move_iport, source_oport_out_of_range_is_rejected)
{
  Session s;
  Graph g("g", &s);

  auto src_u = make_unique<TestVertex>(&s, "src", vector<InEdge>{});
  TestVertex* src = src_u.get();
  src->allocate_oports(1);  // only oport 0 exists
  g.insert_vertex(std::move(src_u));

  auto dst_u =
      make_unique<TestVertex>(&s, "dst", vector<InEdge>{ InEdge{nullptr, 0} });
  TestVertex* dst = dst_u.get();
  g.insert_vertex(std::move(dst_u));

  EXPECT_FALSE(g.move_iport_to(dst, 0, InEdge{src, 3}));
  EXPECT_TRUE(dst->iport_edges()[0].v == nullptr);
}

// ---- Pipeline guard + needs-init semantics --------------------------

TEST(graph_move_iport, pipeline_move_sets_needs_init)
{
  Session s;
  Pipeline pl("pl", &s);
  Stage* src = pl.insert_stage("move-test", "src", {});
  Stage* dst =
      pl.insert_stage("move-test", "dst", { InEdge{nullptr, 0} });
  EXPECT_TRUE(src != nullptr);
  EXPECT_TRUE(dst != nullptr);

  // A fresh stage is wired as constructed -> not flagged.
  EXPECT_FALSE(dst->needs_init());

  EXPECT_TRUE(pl.move_iport_to(dst, 0, InEdge{src, 0}));
  EXPECT_TRUE(dst->iport_edges()[0].v == src);
  // The move changed dst's inputs -> it must re-initialize.
  EXPECT_TRUE(dst->needs_init());
}

TEST(graph_move_iport, pipeline_refuses_move_on_running_stage)
{
  Session s;
  Pipeline pl("pl", &s);
  Stage* src = pl.insert_stage("move-test", "src", {});
  Stage* dst =
      pl.insert_stage("move-test", "dst", { InEdge{src, 0} });

  // Simulate the runtime marking dst's driver live.
  StageLifecycleAccess::set_running(dst, true);
  EXPECT_TRUE(dst->running());

  // Rewiring a running stage is refused; topology is untouched.
  EXPECT_FALSE(pl.move_iport_to(dst, 0));
  EXPECT_TRUE(dst->iport_edges()[0].v == src);
  EXPECT_TRUE(src->oport_edges(0).count(OutEdge{dst, 0}) == 1);

  // Once stopped, the move proceeds.
  StageLifecycleAccess::set_running(dst, false);
  EXPECT_TRUE(pl.move_iport_to(dst, 0));
  EXPECT_TRUE(dst->iport_edges()[0].v == nullptr);
  EXPECT_TRUE(src->oport_edges(0).empty());
}
