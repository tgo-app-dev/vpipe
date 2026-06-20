#include "minitest.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"

#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace {

// Trivial stages used to exercise the registry and Pipeline graph
// machinery. process_one is a no-op coroutine that immediately
// signals done; these tests don't drive the runtime.
class FakeSourceStage : public vpipe::TypedStage<FakeSourceStage> {
public:
  static constexpr const char* kTypeName = "fake-source";
  using TypedStage::TypedStage;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(FakeSourceStage)

class FakeSinkStage : public vpipe::TypedStage<FakeSinkStage> {
public:
  static constexpr const char* kTypeName = "fake-sink";
  using TypedStage::TypedStage;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(FakeSinkStage)

}

TEST(stage_registry, registers_at_static_init) {
  auto& reg = vpipe::StageRegistry::get();
  EXPECT_TRUE(reg.find_id("fake-source")
              != vpipe::StageTypeId::unknown);
  EXPECT_TRUE(reg.find_id("fake-sink")
              != vpipe::StageTypeId::unknown);
}

TEST(stage_registry, type_id_round_trip) {
  auto& reg = vpipe::StageRegistry::get();
  vpipe::StageTypeId id = reg.find_id("fake-source");
  EXPECT_TRUE(reg.find_name(id) == "fake-source");
}

TEST(stage_registry, duplicate_register_returns_same_id) {
  auto& reg = vpipe::StageRegistry::get();
  vpipe::StageTypeId first = reg.find_id("fake-source");
  vpipe::StageTypeId again = reg.register_type("fake-source", nullptr);
  EXPECT_TRUE(first == again);
}

TEST(stage_registry, missing_type) {
  auto& reg = vpipe::StageRegistry::get();
  vpipe::Session sess;
  EXPECT_TRUE(reg.find_id("nope") == vpipe::StageTypeId::unknown);
  EXPECT_TRUE(reg.find_name(vpipe::StageTypeId::unknown).empty());
  EXPECT_TRUE(reg.create("nope", &sess, "x", {}) == nullptr);
}

TEST(stage_registry, create_constructs_typed_instance) {
  auto& reg = vpipe::StageRegistry::get();
  vpipe::Session sess;
  vpipe::StagePtr p = reg.create("fake-source", &sess, "s0", {});
  EXPECT_TRUE(p != nullptr);
  EXPECT_TRUE(p->type_name() == "fake-source");
  EXPECT_TRUE(p->type() == FakeSourceStage::type_id());
}

TEST(typed_stage, type_id_matches_registry) {
  auto& reg = vpipe::StageRegistry::get();
  EXPECT_TRUE(FakeSourceStage::type_id()
              == reg.find_id("fake-source"));
}

TEST(typed_stage, distinct_types_have_distinct_ids) {
  EXPECT_TRUE(FakeSourceStage::type_id() != FakeSinkStage::type_id());
}

TEST(pipeline, insert_stage_succeeds) {
  vpipe::Session sess;
  vpipe::Pipeline pl("pl1", &sess);
  auto stage = make_unique<FakeSourceStage>(&sess, "s0",
                                            vector<vpipe::InEdge>{});
  vpipe::Stage* sp = pl.insert_stage(std::move(stage));
  EXPECT_TRUE(sp != nullptr);
  EXPECT_TRUE(pl.num_vertices() == 1);
  EXPECT_TRUE(sp->type_name() == "fake-source");
}

TEST(pipeline, insert_stage_by_name) {
  vpipe::Session sess;
  vpipe::Pipeline pl("pl1", &sess);
  vpipe::Stage* sp = pl.insert_stage("fake-source", "s0", {});
  EXPECT_TRUE(sp != nullptr);
  EXPECT_TRUE(sp->type_name() == "fake-source");
  EXPECT_TRUE(pl.num_vertices() == 1);
}

TEST(pipeline, insert_stage_unknown_name_returns_null) {
  vpipe::Session sess;
  vpipe::Pipeline pl("pl1", &sess);
  vpipe::Stage* sp = pl.insert_stage("nope", "s0", {});
  EXPECT_TRUE(sp == nullptr);
  EXPECT_TRUE(pl.num_vertices() == 0);
}
