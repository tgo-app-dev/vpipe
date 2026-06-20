#include "minitest.h"
#include "common/flex-data.h"
#include "common/graph.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-spec.h"
#include "pipeline/pipeline.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Build a flat 2-stage pipeline (chrono -> shell) directly through
// the Pipeline API so we don't depend on file I/O for the
// to_spec tests.
unique_ptr<Pipeline>
build_chrono_shell_(Session& sess, const string& pl_id,
                    const string& cmd)
{
  auto pl = make_unique<Pipeline>(pl_id, &sess);

  FlexData chrono_cfg = FlexData::make_object();
  chrono_cfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(50.0));
  chrono_cfg.as_object().insert_or_assign(
      "count", FlexData::make_int(3));
  Stage* src = pl->insert_stage(
      "chrono", "src", {}, std::move(chrono_cfg));

  FlexData shell_cfg = FlexData::make_object();
  shell_cfg.as_object().insert_or_assign(
      "command", FlexData::make_string(cmd));
  Stage* sink = pl->insert_stage(
      "shell", "sink",
      vector<InEdge>{{src, 0u}}, std::move(shell_cfg));

  (void)sink;
  return pl;
}

}

TEST(pipeline_spec, to_spec_emits_id_stages_and_empty_subs) {
  // FlexData ConstObjectView::iterator yields entries by value
  // (the FlexData is a fresh clone). A view obtained from such a
  // temporary dangles. We bind every nested FlexData to a named
  // local before taking a view.
  Session sess;
  auto pl = build_chrono_shell_(sess, "p", "true");

  FlexData spec = pipeline_to_spec(*pl);
  ASSERT_TRUE(spec.is_object());
  auto root = spec.as_object();

  // id
  ASSERT_TRUE(root.contains("id"));
  FlexData id_v = root.at("id");
  EXPECT_TRUE(id_v.get_string() == "p");

  // stages: 2 entries with ids "src" and "sink", in dependency
  // order (src first, then sink).
  ASSERT_TRUE(root.contains("stages"));
  FlexData stages_v = root.at("stages");
  ASSERT_TRUE(stages_v.is_array());
  auto stages = stages_v.as_array();
  ASSERT_TRUE(stages.size() == 2u);

  // subpipelines: empty array.
  ASSERT_TRUE(root.contains("subpipelines"));
  FlexData subs_v = root.at("subpipelines");
  ASSERT_TRUE(subs_v.is_array());
  EXPECT_TRUE(subs_v.as_array().size() == 0u);

  // Walk stages in declared order.
  vector<string> ids;
  for (size_t i = 0; i < stages.size(); ++i) {
    FlexData s_v = stages.at(i);
    auto s = s_v.as_object();
    FlexData sid_v  = s.at("id");
    FlexData type_v = s.at("type");
    string sid(sid_v.get_string());
    string typ(type_v.get_string());
    ids.push_back(sid);
    if (sid == "src") {
      EXPECT_TRUE(typ == "chrono");
      FlexData iports_v = s.at("iports");
      EXPECT_TRUE(iports_v.as_array().size() == 0u);
    } else if (sid == "sink") {
      EXPECT_TRUE(typ == "shell");
      FlexData iports_v = s.at("iports");
      auto ip = iports_v.as_array();
      ASSERT_TRUE(ip.size() == 1u);
      FlexData e_v = ip.at(0);
      auto e = e_v.as_object();
      FlexData src_v   = e.at("src");
      FlexData oport_v = e.at("oport");
      EXPECT_TRUE(src_v.get_string() == "src");
      EXPECT_TRUE(oport_v.get_int() == 0);
    }
  }
  // Topo order means src comes before sink.
  ASSERT_TRUE(ids.size() == 2u);
  EXPECT_TRUE(ids[0] == "src");
  EXPECT_TRUE(ids[1] == "sink");
}

TEST(pipeline_spec, round_trip_through_flexdata_preserves_topology) {
  Session sess;
  auto src_pl = build_chrono_shell_(sess, "p", "echo hi");
  FlexData spec1 = pipeline_to_spec(*src_pl);

  auto rebuilt = pipeline_from_spec(spec1, &sess);
  ASSERT_TRUE(rebuilt != nullptr);
  EXPECT_TRUE(rebuilt->id() == "p");
  EXPECT_TRUE(rebuilt->num_vertices() == 2u);

  // A second to_spec should match the first by JSON pretty-printed
  // string -- key ordering inside FlexData objects is preserved
  // across copy + write, so the round trip is byte-stable.
  FlexData spec2 = pipeline_to_spec(*rebuilt);
  EXPECT_TRUE(spec1.to_json(true) == spec2.to_json(true));
}

TEST(pipeline_spec, rejects_unknown_stage_type) {
  Session sess;
  FlexData spec = FlexData::make_object();
  spec.as_object().insert_or_assign(
      "id", FlexData::make_string("p"));
  FlexData stages = FlexData::make_array();
  FlexData s = FlexData::make_object();
  s.as_object().insert_or_assign("id",
      FlexData::make_string("x"));
  s.as_object().insert_or_assign("type",
      FlexData::make_string("type-that-does-not-exist"));
  stages.as_array().push_back(std::move(s));
  spec.as_object().insert_or_assign("stages", std::move(stages));

  auto pl = pipeline_from_spec(spec, &sess);
  EXPECT_TRUE(pl == nullptr);
}

TEST(pipeline_spec, rejects_forward_reference) {
  Session sess;
  // sink declared first, references "src" which appears later.
  FlexData spec = FlexData::make_object();
  spec.as_object().insert_or_assign(
      "id", FlexData::make_string("p"));
  FlexData stages = FlexData::make_array();

  FlexData sink = FlexData::make_object();
  sink.as_object().insert_or_assign("id",
      FlexData::make_string("sink"));
  sink.as_object().insert_or_assign("type",
      FlexData::make_string("shell"));
  FlexData iports = FlexData::make_array();
  FlexData e = FlexData::make_object();
  e.as_object().insert_or_assign("src",
      FlexData::make_string("src"));
  e.as_object().insert_or_assign("oport",
      FlexData::make_int(0));
  iports.as_array().push_back(std::move(e));
  sink.as_object().insert_or_assign("iports", std::move(iports));
  FlexData sink_cfg = FlexData::make_object();
  sink_cfg.as_object().insert_or_assign(
      "command", FlexData::make_string("true"));
  sink.as_object().insert_or_assign("config", std::move(sink_cfg));
  stages.as_array().push_back(std::move(sink));
  spec.as_object().insert_or_assign("stages", std::move(stages));

  auto pl = pipeline_from_spec(spec, &sess);
  EXPECT_TRUE(pl == nullptr);
}

TEST(pipeline_spec, rejects_oport_out_of_range) {
  Session sess;
  // chrono has 1 oport; reference oport 5 -> reject.
  FlexData spec = FlexData::make_object();
  spec.as_object().insert_or_assign(
      "id", FlexData::make_string("p"));
  FlexData stages = FlexData::make_array();

  FlexData src = FlexData::make_object();
  src.as_object().insert_or_assign("id",
      FlexData::make_string("src"));
  src.as_object().insert_or_assign("type",
      FlexData::make_string("chrono"));
  FlexData src_cfg = FlexData::make_object();
  src_cfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(10.0));
  src_cfg.as_object().insert_or_assign(
      "count", FlexData::make_int(1));
  src.as_object().insert_or_assign("config", std::move(src_cfg));
  stages.as_array().push_back(std::move(src));

  FlexData sink = FlexData::make_object();
  sink.as_object().insert_or_assign("id",
      FlexData::make_string("sink"));
  sink.as_object().insert_or_assign("type",
      FlexData::make_string("shell"));
  FlexData iports = FlexData::make_array();
  FlexData e = FlexData::make_object();
  e.as_object().insert_or_assign("src",
      FlexData::make_string("src"));
  e.as_object().insert_or_assign("oport",
      FlexData::make_int(5));
  iports.as_array().push_back(std::move(e));
  sink.as_object().insert_or_assign("iports", std::move(iports));
  FlexData sink_cfg = FlexData::make_object();
  sink_cfg.as_object().insert_or_assign(
      "command", FlexData::make_string("true"));
  sink.as_object().insert_or_assign("config", std::move(sink_cfg));
  stages.as_array().push_back(std::move(sink));

  spec.as_object().insert_or_assign("stages", std::move(stages));
  auto pl = pipeline_from_spec(spec, &sess);
  EXPECT_TRUE(pl == nullptr);
}

TEST(pipeline_spec, subpipeline_round_trip) {
  Session sess;
  auto parent = build_chrono_shell_(sess, "parent", "true");
  // Insert a child pipeline into the parent.
  auto child = build_chrono_shell_(sess, "child", "true");
  parent->insert_graph(std::move(child));

  FlexData spec = pipeline_to_spec(*parent);
  ASSERT_TRUE(spec.is_object());
  auto root = spec.as_object();
  ASSERT_TRUE(root.contains("subpipelines"));
  EXPECT_TRUE(root.at("subpipelines").as_array().size() == 1u);

  auto rebuilt = pipeline_from_spec(spec, &sess);
  ASSERT_TRUE(rebuilt != nullptr);
  EXPECT_TRUE(rebuilt->id() == "parent");
  EXPECT_TRUE(rebuilt->num_vertices() == 2u);
  // The child shows up via Graph::graphs().
  EXPECT_TRUE(rebuilt->graphs().size() == 1u);
  ASSERT_TRUE(rebuilt->graphs().count("child") == 1u);
  EXPECT_TRUE(rebuilt->graphs().at("child")->num_vertices() == 2u);
}
