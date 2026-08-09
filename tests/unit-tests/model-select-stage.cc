// ModelSelectStage + apply_model_select_beat() tests (CPU-only).
//
// The model-select source emits one { hf_dir } FlexData beat that
// the diffusion-conditioner / generate-image / vae-encode / vae-decode stages
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

#include <algorithm>
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

// Every stage in the split diffusion flow shares ONE model, so a family the
// consumers can run must be pickable at the SOURCE -- a model-select that
// cannot pick a family the conditioner/DiT/VAE support is a dead end for the
// user, and that is exactly how the Mage-Flow families were missed when they
// were added to the other four, and later how minimax-h3-fl2va was missing
// from model-select while a whole text-to-video graph ran it.
//
// This used to demand that all five lists be IDENTICAL, which held only
// while every consumer ran every family. It stopped being true once the
// flow grew a second DiT: generate-image runs the image families and
// generate-video the video ones, and neither can run the other's. So the
// invariant is now the one that actually generalises --
//
//   model-select's list == the UNION of its consumers' lists
//
// which is two properties at once: nothing runnable is unpickable at the
// source (a new family wired into a consumer only), and nothing pickable
// at the source is unrunnable everywhere (a family removed from the last
// consumer that had it, or a typo in one list).
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
  auto split = [](const std::string& csv) -> std::vector<std::string> {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i <= csv.size()) {
      const std::size_t k = csv.find(',', i);
      const std::size_t e = (k == std::string::npos) ? csv.size() : k;
      if (e > i) { out.push_back(csv.substr(i, e - i)); }
      if (k == std::string::npos) { break; }
      i = k + 1;
    }
    return out;
  };
  // Every stage that latches model-select's beat on a `model` iport.
  const char* consumers[] = {"diffusion-conditioner", "generate-image",
                             "generate-video", "vae-encode", "vae-decode",
                             "audio-vae-decode"};
  const StageRegistry& reg = StageRegistry::get();
  const StageSpec* src = reg.spec("model-select");
  ASSERT_TRUE(src != nullptr);
  const std::string want = types_of(*src);
  ASSERT_TRUE(!want.empty() && want != "<no hf_dir key>");
  std::printf("[model_select] %-22s hf_dir suggest_db_type = %s\n",
              "model-select", want.c_str());

  const std::vector<std::string> offered = split(want);
  std::vector<std::string> seen;          // the union over the consumers
  for (const char* nm : consumers) {
    const StageSpec* sp = reg.spec(nm);
    ASSERT_TRUE(sp != nullptr);
    const std::string got = types_of(*sp);
    std::printf("[model_select] %-22s hf_dir suggest_db_type = %s\n", nm,
                got.c_str());
    for (const std::string& fam : split(got)) {
      // A consumer offering what the shared source cannot emit is the
      // same dead end seen from the other side.
      const bool at_source =
          std::find(offered.begin(), offered.end(), fam) != offered.end();
      if (!at_source) {
        std::printf("[model_select] '%s' offers family '%s', model-select does "
                    "NOT\n", nm, fam.c_str());
      }
      EXPECT_TRUE(at_source);
      if (std::find(seen.begin(), seen.end(), fam) == seen.end()) {
        seen.push_back(fam);
      }
    }
  }
  // ...and the other direction: nothing the source offers is orphaned.
  for (const std::string& fam : offered) {
    const bool runnable =
        std::find(seen.begin(), seen.end(), fam) != seen.end();
    if (!runnable) {
      std::printf("[model_select] model-select offers '%s' but NO consumer "
                  "runs it\n", fam.c_str());
    }
    EXPECT_TRUE(runnable);
  }
  // And the set must actually contain every diffusion family the flow
  // supports, so a future family shows up here rather than silently nowhere.
  // The video families are named for the same reason the image ones are:
  // they were added to the DiT and the VAEs before the picker.
  for (const char* fam : {"krea2", "flux2", "qwen-image-edit", "mage-flow",
                          "mage-flow-edit", "boogu-image", "boogu-image-edit",
                          "wan-t2v", "wan-i2v", "minimax-h3-fl2va"}) {
    const bool present =
        std::find(offered.begin(), offered.end(), fam) != offered.end();
    if (!present) {
      std::printf("[model_select] MISSING family '%s' from the pickers\n", fam);
    }
    EXPECT_TRUE(present);
  }
}
