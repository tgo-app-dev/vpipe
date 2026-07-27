// ModelSelectStage + apply_model_select_beat() tests (CPU-only).
//
// The model-select source emits one { hf_dir, models_db } FlexData beat that
// the diffusion-conditioner / text-to-image / vae-encode / vae-decode stages
// latch on their `model` iport to share a single model choice (overriding each
// stage's hf_dir config). These tests cover the source's beat shape + one-shot
// emission (incl. fan-out to several consumers) and the shared beat parser.
//
//   vpipe_test --filter '*model_select*'

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "pipeline/stage-registry.h"
#include "stages/model-registry.h"
#include "stages/model-select-stage.h"

#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

// Test-only sink: 1 iport, captures every payload until EOS.
class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-model";
  using TypedStage::TypedStage;
  vector<unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// The (hf_dir, models_db) of a captured model beat.
pair<string, string>
beat_dirs_(const unique_ptr<BeatPayloadIntf>& p)
{
  const auto* fd = dynamic_cast<const FlexDataPayload*>(p.get());
  if (fd == nullptr || !fd->data.is_object()) { return {}; }
  auto o = fd->data.as_object();
  return {string(o.at("hf_dir").as_string("")),
          string(o.at("models_db").as_string(""))};
}

}  // namespace

// Construction succeeds for any config; a missing hf_dir is deferred to launch.
TEST(model_select, config_hf_dir_required_deferred)
{
  Session sess;
  ModelSelectStage s(&sess, "ms", vector<InEdge>{}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());   // hf_dir required (deferred)
  EXPECT_TRUE(s.num_oports() == 1);
}

// resolved_beat() carries hf_dir + the models_db (defaulting to "models").
TEST(model_select, resolved_beat_shape)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string("/models/krea2"));
  ModelSelectStage s(&sess, "ms", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  FlexData beat = s.resolved_beat();       // bind: as_object() is a view
  auto o = beat.as_object();
  EXPECT_TRUE(string(o.at("hf_dir").as_string("")) == "/models/krea2");
  EXPECT_TRUE(string(o.at("models_db").as_string("")) == "models");

  FlexData cfg2 = FlexData::make_object();
  cfg2.as_object().insert("hf_dir", FlexData::make_string("org/repo"));
  cfg2.as_object().insert("models_db", FlexData::make_string("my_db"));
  ModelSelectStage s2(&sess, "ms2", vector<InEdge>{}, std::move(cfg2));
  FlexData beat2 = s2.resolved_beat();
  auto o2 = beat2.as_object();
  EXPECT_TRUE(string(o2.at("hf_dir").as_string("")) == "org/repo");
  EXPECT_TRUE(string(o2.at("models_db").as_string("")) == "my_db");
}

// End-to-end: the source emits exactly one model beat, and it fans out to
// SEVERAL consumers (the shared-model wiring the four diffusion stages use).
TEST(model_select, emits_one_beat_to_all_consumers)
{
  Session sess;
  CerrSilencer hush;
  auto pl = make_unique<Pipeline>("p", &sess);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string("/models/flux2"));
  auto ms_u = make_unique<ModelSelectStage>(
      &sess, "ms", vector<InEdge>{}, std::move(cfg));
  auto* ms = static_cast<ModelSelectStage*>(pl->insert_stage(std::move(ms_u)));

  // Two consumers on the one oport (fan-out).
  auto s0_u = make_unique<SinkCapture>(
      &sess, "s0", vector<InEdge>{{ms, 0}}, FlexData::make_object());
  auto* s0 = static_cast<SinkCapture*>(pl->insert_stage(std::move(s0_u)));
  auto s1_u = make_unique<SinkCapture>(
      &sess, "s1", vector<InEdge>{{ms, 0}}, FlexData::make_object());
  auto* s1 = static_cast<SinkCapture*>(pl->insert_stage(std::move(s1_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(s0->captured.size() == 1);
  ASSERT_TRUE(s1->captured.size() == 1);
  const auto d0 = beat_dirs_(s0->captured[0]);
  const auto d1 = beat_dirs_(s1->captured[0]);
  EXPECT_TRUE(d0.first == "/models/flux2" && d0.second == "models");
  EXPECT_TRUE(d1.first == "/models/flux2" && d1.second == "models");
}

// The shared beat parser: overrides hf_dir (string or object; "model" alias)
// and only overrides models_db when the beat carries one.
TEST(model_select, apply_beat_parses_forms)
{
  // Plain string -> hf_dir; models_db untouched.
  {
    string hf = "old", db = "models";
    EXPECT_TRUE(apply_model_select_beat(
        FlexData::make_string("/m/krea2"), hf, db));
    EXPECT_TRUE(hf == "/m/krea2" && db == "models");
  }
  // Object with hf_dir + models_db -> both overridden.
  {
    FlexData b = FlexData::make_object();
    b.as_object().insert("hf_dir", FlexData::make_string("org/repo"));
    b.as_object().insert("models_db", FlexData::make_string("db2"));
    string hf = "old", db = "models";
    EXPECT_TRUE(apply_model_select_beat(b, hf, db));
    EXPECT_TRUE(hf == "org/repo" && db == "db2");
  }
  // Object with the "model" alias, no models_db -> db kept.
  {
    FlexData b = FlexData::make_object();
    b.as_object().insert("model", FlexData::make_string("/m/qie"));
    string hf = "old", db = "keepme";
    EXPECT_TRUE(apply_model_select_beat(b, hf, db));
    EXPECT_TRUE(hf == "/m/qie" && db == "keepme");
  }
  // Empty / no usable ref -> false, fields untouched.
  {
    string hf = "old", db = "models";
    EXPECT_FALSE(apply_model_select_beat(FlexData::make_string(""), hf, db));
    EXPECT_FALSE(apply_model_select_beat(FlexData::make_object(), hf, db));
    EXPECT_TRUE(hf == "old" && db == "models");
  }
}

// Every stage in the split diffusion flow shares ONE model, so their model
// pickers must offer the SAME families -- a model-select that cannot pick a
// family the conditioner/DiT/VAE support is a dead end for the user, and that
// is exactly how the Mage-Flow families were missed when they were added to
// the other four. This pins the set across all five specs so the next family
// cannot be wired into some of them only.
TEST(model_select, diffusion_pickers_offer_the_same_families)
{
  auto types_of = [](const StageSpec& sp) -> std::string {
    for (const auto& a : sp.attrs) {
      if (std::string_view(a.key) == "hf_dir") {
        return std::string(a.suggest_db_type);
      }
    }
    return "<no hf_dir key>";
  };
  const char* names[] = {"model-select", "diffusion-conditioner",
                         "text-to-image", "vae-encode", "vae-decode"};
  const StageRegistry& reg = StageRegistry::get();
  const StageSpec* specs[5] = {};
  for (int i = 0; i < 5; ++i) {
    specs[i] = reg.spec(names[i]);
    ASSERT_TRUE(specs[i] != nullptr);
  }
  const std::string want = types_of(*specs[0]);
  ASSERT_TRUE(!want.empty() && want != "<no hf_dir key>");
  bool all_same = true;
  for (int i = 0; i < 5; ++i) {
    const std::string got = types_of(*specs[i]);
    std::printf("[model_select] %-22s hf_dir suggest_db_type = %s\n", names[i],
                got.c_str());
    if (got != want) { all_same = false; }
  }
  EXPECT_TRUE(all_same);
  // And the set must actually contain every diffusion family the flow
  // supports, so a future family shows up here rather than silently nowhere.
  for (const char* fam : {"krea2", "flux2", "qwen-image-edit", "mage-flow",
                          "mage-flow-edit"}) {
    const bool present = want.find(fam) != std::string::npos;
    if (!present) {
      std::printf("[model_select] MISSING family '%s' from the pickers\n", fam);
    }
    EXPECT_TRUE(present);
  }
}
