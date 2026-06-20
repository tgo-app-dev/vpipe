#include "minitest.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/clock-domain.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"

#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace {

// Trivial single-clock stage. Its driver loop is never run by these
// tests; we only exercise the analyzer / launch validation.
class CdSync : public vpipe::TypedStage<CdSync> {
public:
  static constexpr const char* kTypeName = "ut-cd-sync";
  using TypedStage::TypedStage;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(CdSync)

// Crosser modelled on the video decoder: every oport reports its own
// clock group (group 0 for port 0, group 1 for port 1).
class CdCrosserSrc : public vpipe::TypedStage<CdCrosserSrc> {
public:
  static constexpr const char* kTypeName = "ut-cd-crosser-src";
  using TypedStage::TypedStage;
  unsigned oport_clock_group(unsigned p) const noexcept override
  {
    return p == 0 ? 0u : 1u;
  }
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(CdCrosserSrc)

// Crosser modelled on the video encoder: each iport on its own clock,
// no oports.
class CdCrosserSink : public vpipe::TypedStage<CdCrosserSink> {
public:
  static constexpr const char* kTypeName = "ut-cd-crosser-sink";
  using TypedStage::TypedStage;
  unsigned iport_clock_group(unsigned p) const noexcept override
  {
    return p == 0 ? 0u : 1u;
  }
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(CdCrosserSink)

}

TEST(clock_domain, single_chain_is_one_domain) {
  vpipe::Session sess;
  vpipe::Pipeline pl("p", &sess);

  auto a_u = make_unique<CdSync>(&sess, "a", vector<vpipe::InEdge>{});
  a_u->allocate_oports(1);
  auto* a = pl.insert_stage(std::move(a_u));

  auto b_u = make_unique<CdSync>(&sess, "b",
                                 vector<vpipe::InEdge>{{a, 0}});
  b_u->allocate_oports(1);
  auto* b = pl.insert_stage(std::move(b_u));

  auto c_u = make_unique<CdSync>(&sess, "c",
                                 vector<vpipe::InEdge>{{b, 0}});
  auto* c = pl.insert_stage(std::move(c_u));

  vector<vpipe::Stage*> stages{a, b, c};
  auto cd = vpipe::compute_clock_domains(stages);

  EXPECT_TRUE(cd.num_domains == 1u);
  EXPECT_TRUE(cd.crossers.empty());
  EXPECT_TRUE(cd.stages_in_domain[0].size() == 3u);
  EXPECT_FALSE(cd.intra_domain_cycle);
}

TEST(clock_domain, crosser_src_splits_two_domains) {
  vpipe::Session sess;
  vpipe::Pipeline pl("p", &sess);

  auto src_u = make_unique<CdCrosserSrc>(&sess, "src",
                                         vector<vpipe::InEdge>{});
  src_u->allocate_oports(2);
  auto* src = pl.insert_stage(std::move(src_u));

  auto v_u = make_unique<CdSync>(&sess, "v",
                                 vector<vpipe::InEdge>{{src, 0}});
  auto* v = pl.insert_stage(std::move(v_u));

  auto a_u = make_unique<CdSync>(&sess, "a",
                                 vector<vpipe::InEdge>{{src, 1}});
  auto* a = pl.insert_stage(std::move(a_u));

  vector<vpipe::Stage*> stages{src, v, a};
  auto cd = vpipe::compute_clock_domains(stages);

  EXPECT_TRUE(cd.num_domains == 2u);
  EXPECT_TRUE(cd.crossers.size() == 1u);
  EXPECT_TRUE(cd.crossers[0] == src);
  // The two sync sinks must end up in different domains.
  vpipe::PortKey kv{v, vpipe::PortKey::Kind::In, 0};
  vpipe::PortKey ka{a, vpipe::PortKey::Kind::In, 0};
  EXPECT_TRUE(cd.port_domain[kv] != cd.port_domain[ka]);
  EXPECT_FALSE(cd.intra_domain_cycle);
}

TEST(clock_domain, crosser_sink_separates_iport_domains) {
  vpipe::Session sess;
  vpipe::Pipeline pl("p", &sess);

  auto v_u = make_unique<CdSync>(&sess, "v",
                                 vector<vpipe::InEdge>{});
  v_u->allocate_oports(1);
  auto* v = pl.insert_stage(std::move(v_u));

  auto a_u = make_unique<CdSync>(&sess, "a",
                                 vector<vpipe::InEdge>{});
  a_u->allocate_oports(1);
  auto* a = pl.insert_stage(std::move(a_u));

  auto sink_u = make_unique<CdCrosserSink>(
      &sess, "sink",
      vector<vpipe::InEdge>{{v, 0}, {a, 0}});
  auto* sink = pl.insert_stage(std::move(sink_u));

  vector<vpipe::Stage*> stages{v, a, sink};
  auto cd = vpipe::compute_clock_domains(stages);

  EXPECT_TRUE(cd.num_domains == 2u);
  EXPECT_TRUE(cd.crossers.size() == 1u);
  EXPECT_TRUE(cd.crossers[0] == sink);
  EXPECT_FALSE(cd.intra_domain_cycle);
}

TEST(clock_domain, intra_domain_cycle_detected) {
  // a -> b -> c -> a. Every stage is single-clock (default), so all
  // three end up in one domain and the cycle becomes an
  // intra-domain cycle.
  vpipe::Session sess;
  vpipe::Pipeline pl("p", &sess);

  // The graph builder requires upstream stages to exist before
  // downstream constructs; build first, then patch the back-edge by
  // hand via the Vertex API.
  auto a_u = make_unique<CdSync>(&sess, "a", vector<vpipe::InEdge>{});
  a_u->allocate_oports(1);
  auto* a = pl.insert_stage(std::move(a_u));

  auto b_u = make_unique<CdSync>(&sess, "b",
                                 vector<vpipe::InEdge>{{a, 0}});
  b_u->allocate_oports(1);
  auto* b = pl.insert_stage(std::move(b_u));

  auto c_u = make_unique<CdSync>(&sess, "c",
                                 vector<vpipe::InEdge>{{b, 0}});
  c_u->allocate_oports(1);
  auto* c = pl.insert_stage(std::move(c_u));

  // Register c -> a as an additional fanout on c so the cycle is
  // visible to the analyzer. We walk through VertexGraphAccess.
  vpipe::VertexGraphAccess::attach_fanout(c, 0, vpipe::OutEdge{a, 0});

  vector<vpipe::Stage*> stages{a, b, c};
  auto cd = vpipe::compute_clock_domains(stages);

  EXPECT_TRUE(cd.num_domains == 1u);
  EXPECT_TRUE(cd.intra_domain_cycle);
  EXPECT_TRUE(cd.cycle_stage != nullptr);
}

TEST(clock_domain, runtime_rejects_intra_domain_cycle) {
  // Same shape as the prior test, but driven through
  // PipelineRuntime::launch -- expect a clean false return rather
  // than an exception or a hung worker.
  vpipe::Session sess;
  vpipe::Pipeline pl("p", &sess);

  auto a_u = make_unique<CdSync>(&sess, "a", vector<vpipe::InEdge>{});
  a_u->allocate_oports(1);
  auto* a = pl.insert_stage(std::move(a_u));

  auto b_u = make_unique<CdSync>(&sess, "b",
                                 vector<vpipe::InEdge>{{a, 0}});
  b_u->allocate_oports(1);
  auto* b = pl.insert_stage(std::move(b_u));

  auto c_u = make_unique<CdSync>(&sess, "c",
                                 vector<vpipe::InEdge>{{b, 0}});
  c_u->allocate_oports(1);
  auto* c = pl.insert_stage(std::move(c_u));

  vpipe::VertexGraphAccess::attach_fanout(c, 0, vpipe::OutEdge{a, 0});
  // Patching c -> a as a fanout doesn't add the InEdge on a; do that
  // by re-creating the iport list. Instead, rely on the analyzer's
  // signal: launch should fail, and we don't actually run drivers.

  vpipe::PipelineRuntime rt(&pl, &sess);
  EXPECT_FALSE(rt.launch());
  EXPECT_FALSE(rt.running());
}
