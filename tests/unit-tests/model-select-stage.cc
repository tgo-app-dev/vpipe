// ModelSelectStage + apply_model_select_beat() tests (CPU-only).
//
// The model-select source emits one { hf_dir } FlexData beat that
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

// The hf_dir of a captured model beat.
string
beat_dir_(const unique_ptr<BeatPayloadIntf>& p)
{
  const auto* fd = dynamic_cast<const FlexDataPayload*>(p.get());
  if (fd == nullptr || !fd->data.is_object()) { return {}; }
  auto o = fd->data.as_object();
  return string(o.at("hf_dir").as_string(""));
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

// resolved_beat() carries the hf_dir. The registry sub-db is fixed
// (kModelRegistryDb) rather than travelling in the beat, so a consumer
// cannot be pointed at a different db than the one model-fetch writes.
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
  EXPECT_FALSE(o.contains("models_db"));

  FlexData cfg2 = FlexData::make_object();
  cfg2.as_object().insert("hf_dir", FlexData::make_string("org/repo"));
  ModelSelectStage s2(&sess, "ms2", vector<InEdge>{}, std::move(cfg2));
  FlexData beat2 = s2.resolved_beat();
  auto o2 = beat2.as_object();
  EXPECT_TRUE(string(o2.at("hf_dir").as_string("")) == "org/repo");
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
  EXPECT_TRUE(beat_dir_(s0->captured[0]) == "/models/flux2");
  EXPECT_TRUE(beat_dir_(s1->captured[0]) == "/models/flux2");
}

// A source must emit again on a RELAUNCH. Stopping a pipeline destroys
// the runtime, not the stages, so a Stop-then-Start re-enters
// initialize() on the same ModelSelectStage with its one-shot latch
// still set from the previous run. Before the per-launch reset it
// emitted nothing on run 2 and signalled done immediately -- which
// silently killed every consumer waiting on the model beat (the whole
// diffusion chain) while the pipeline "completed" in milliseconds.
TEST(model_select, emits_again_on_relaunch)
{
  Session sess;
  CerrSilencer hush;
  auto pl = make_unique<Pipeline>("p", &sess);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string("/models/flux2"));
  auto ms_u = make_unique<ModelSelectStage>(
      &sess, "ms", vector<InEdge>{}, std::move(cfg));
  auto* ms = static_cast<ModelSelectStage*>(pl->insert_stage(std::move(ms_u)));
  auto sk_u = make_unique<SinkCapture>(
      &sess, "s0", vector<InEdge>{{ms, 0}}, FlexData::make_object());
  auto* sk = static_cast<SinkCapture*>(pl->insert_stage(std::move(sk_u)));

  // Three launches over the SAME stage objects: one beat each time.
  for (int run = 1; run <= 3; ++run) {
    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    ASSERT_TRUE(sk->captured.size() == static_cast<size_t>(run));
    EXPECT_TRUE(beat_dir_(sk->captured[run - 1]) == "/models/flux2");
  }
}

// The shared beat parser: overrides hf_dir (string or object; "model"
// alias). A "models_db" carried by an OLD beat is ignored -- the registry
// sub-db is fixed system-wide.
TEST(model_select, apply_beat_parses_forms)
{
  // Plain string -> hf_dir.
  {
    string hf = "old";
    EXPECT_TRUE(apply_model_select_beat(
        FlexData::make_string("/m/krea2"), hf));
    EXPECT_TRUE(hf == "/m/krea2");
  }
  // Object with hf_dir (+ a stale models_db, which is ignored).
  {
    FlexData b = FlexData::make_object();
    b.as_object().insert("hf_dir", FlexData::make_string("org/repo"));
    b.as_object().insert("models_db", FlexData::make_string("db2"));
    string hf = "old";
    EXPECT_TRUE(apply_model_select_beat(b, hf));
    EXPECT_TRUE(hf == "org/repo");
  }
  // Object with the "model" alias.
  {
    FlexData b = FlexData::make_object();
    b.as_object().insert("model", FlexData::make_string("/m/qie"));
    string hf = "old";
    EXPECT_TRUE(apply_model_select_beat(b, hf));
    EXPECT_TRUE(hf == "/m/qie");
  }
  // Empty / no usable ref -> false, hf_dir untouched.
  {
    string hf = "old";
    EXPECT_FALSE(apply_model_select_beat(FlexData::make_string(""), hf));
    EXPECT_FALSE(apply_model_select_beat(FlexData::make_object(), hf));
    EXPECT_TRUE(hf == "old");
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
