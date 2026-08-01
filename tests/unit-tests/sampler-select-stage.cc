// SamplerSelectStage tests -- the LLM TOKEN sampler source.
//
// `sampler-select` emits one spec beat that text-chat / visual-qa /
// realtime-vqa / text-to-speech latch on their optional sampler iport. These
// tests cover the beat shape, the greedy default, config validation, and the
// mis-wire guard that keeps a `diffusion-sampler-select` beat (which shares
// the payload TYPE but none of the keys) from silently reading as a no-op.
//
//   vpipe_test --filter '*sampler_select*'

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
#include "stages/diffusion-sampler-select-stage.h"
#include "stages/sampler-select-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/sampler.h"
#include "stages/sampler-spec.h"
#endif

#include <cmath>
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
  static constexpr const char* kTypeName = "ut-sink-capture-sampler";
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

FlexData
cfg_of(initializer_list<pair<const char*, FlexData>> kv)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  for (auto& [k, v] : kv) { o.insert_or_assign(k, v); }
  return c;
}

}  // namespace

// Both stages register, under DISTINCT type names -- the rename is what a
// pipeline JSON sees, and a stale registration would resolve one name to the
// wrong stage rather than fail loudly.
TEST(sampler_select, both_selectors_register_under_distinct_names)
{
  EXPECT_TRUE(string_view(SamplerSelectStage::kTypeName) == "sampler-select");
  EXPECT_TRUE(string_view(DiffusionSamplerSelectStage::kTypeName)
              == "diffusion-sampler-select");
  const StageSpec* tok = StageRegistry::get().spec("sampler-select");
  const StageSpec* dif = StageRegistry::get().spec("diffusion-sampler-select");
  ASSERT_TRUE(tok != nullptr && dif != nullptr);
  EXPECT_TRUE(tok != dif);
  // "sampler-select" must now resolve to the TOKEN sampler, not to the
  // diffusion one it used to name -- that is the substance of the rename.
  EXPECT_TRUE(tok->display_name == string_view("Sampler"));
  EXPECT_TRUE(dif->display_name == string_view("Diffusion Sampler"));
  // Each declares exactly one oport and no iports (both are sources).
  EXPECT_TRUE(tok->oports.size() == 1u && tok->iports.size() == 0u);
  EXPECT_TRUE(dif->oports.size() == 1u && dif->iports.size() == 0u);
}

// A bare stage emits the SamplerParams defaults, which are argmax. That makes
// "wire a sampler-select but configure nothing" identical to "wire nothing",
// so adding the stage never silently changes a graph's output.
TEST(sampler_select, bare_stage_emits_the_greedy_defaults)
{
  Session sess;
  SamplerSelectStage st(&sess, "s", {}, FlexData::make_object());
  EXPECT_TRUE(st.config_error().empty());
  EXPECT_TRUE(st.num_oports() == 1);

  FlexData fd = st.resolved_spec();     // bind: as_object() is a view
  auto o = fd.as_object();
  EXPECT_TRUE(string(o.at("sampler").as_string("")) == "token");
  EXPECT_TRUE(abs(o.at("temperature").as_real(0.0) - 1.0) < 1e-9);
  EXPECT_TRUE(o.at("top_k").as_int(-1) == 0);
  EXPECT_TRUE(abs(o.at("top_p").as_real(0.0) - 1.0) < 1e-9);
  EXPECT_TRUE(o.at("min_p").as_real(9.0) == 0.0);
  EXPECT_TRUE(abs(o.at("repetition_penalty").as_real(0.0) - 1.0) < 1e-9);
  EXPECT_TRUE(o.at("presence_penalty").as_real(9.0) == 0.0);
  EXPECT_TRUE(o.at("seed").as_uint(9) == 0);
}

TEST(sampler_select, config_overrides_round_trip)
{
  Session sess;
  SamplerSelectStage st(&sess, "s", {}, cfg_of({
      {"temperature", FlexData::make_real(0.7)},
      {"top_k", FlexData::make_int(40)},
      {"top_p", FlexData::make_real(0.9)},
      {"min_p", FlexData::make_real(0.05)},
      {"repetition_penalty", FlexData::make_real(1.1)},
      {"presence_penalty", FlexData::make_real(0.5)},
      {"seed", FlexData::make_uint(1234)},
  }));
  EXPECT_TRUE(st.config_error().empty());
  FlexData fd = st.resolved_spec();
  auto o = fd.as_object();
  EXPECT_TRUE(abs(o.at("temperature").as_real(0.0) - 0.7) < 1e-9);
  EXPECT_TRUE(o.at("top_k").as_int(0) == 40);
  EXPECT_TRUE(abs(o.at("top_p").as_real(0.0) - 0.9) < 1e-9);
  EXPECT_TRUE(abs(o.at("min_p").as_real(0.0) - 0.05) < 1e-9);
  EXPECT_TRUE(abs(o.at("repetition_penalty").as_real(0.0) - 1.1) < 1e-9);
  EXPECT_TRUE(abs(o.at("presence_penalty").as_real(0.0) - 0.5) < 1e-9);
  EXPECT_TRUE(o.at("seed").as_uint(0) == 1234);
}

// Out-of-range knobs are deferred config errors (the ctor never throws), so a
// graph still builds and the runtime skips the stage at launch.
TEST(sampler_select, rejects_out_of_range_knobs)
{
  Session sess;
  {
    SamplerSelectStage st(&sess, "s", {},
                          cfg_of({{"top_k", FlexData::make_int(-1)}}));
    EXPECT_FALSE(st.config_error().empty());
  }
  {
    SamplerSelectStage st(&sess, "s", {},
                          cfg_of({{"top_p", FlexData::make_real(1.5)}}));
    EXPECT_FALSE(st.config_error().empty());
  }
  {
    SamplerSelectStage st(&sess, "s", {},
                          cfg_of({{"min_p", FlexData::make_real(-0.1)}}));
    EXPECT_FALSE(st.config_error().empty());
  }
  {
    SamplerSelectStage st(&sess, "s", {},
                          cfg_of({{"repetition_penalty",
                                   FlexData::make_real(0.0)}}));
    EXPECT_FALSE(st.config_error().empty());
  }
}

// End-to-end: one beat, fanned out to several consumers (a single
// sampler-select can program every LLM stage in a graph).
TEST(sampler_select, emits_one_beat_to_all_consumers)
{
  Session sess;
  CerrSilencer hush;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto* sel = static_cast<SamplerSelectStage*>(pl->insert_stage(
      make_unique<SamplerSelectStage>(
          &sess, "sel", vector<InEdge>{},
          cfg_of({{"temperature", FlexData::make_real(0.8)}}))));
  auto* s0 = static_cast<SinkCapture*>(pl->insert_stage(
      make_unique<SinkCapture>(&sess, "s0", vector<InEdge>{{sel, 0}},
                               FlexData::make_object())));
  auto* s1 = static_cast<SinkCapture*>(pl->insert_stage(
      make_unique<SinkCapture>(&sess, "s1", vector<InEdge>{{sel, 0}},
                               FlexData::make_object())));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(s0->captured.size() == 1);
  ASSERT_TRUE(s1->captured.size() == 1);
  for (const auto* sink : {s0, s1}) {
    const auto* fd =
        dynamic_cast<const FlexDataPayload*>(sink->captured[0].get());
    ASSERT_TRUE(fd != nullptr && fd->data.is_object());
    auto o = fd->data.as_object();
    EXPECT_TRUE(string(o.at("sampler").as_string("")) == "token");
    EXPECT_TRUE(abs(o.at("temperature").as_real(0.0) - 0.8) < 1e-9);
  }
}

// The two select stages are NOT interchangeable: they emit the same payload
// type onto ports that look alike, so each tags its spec and consumers check
// the tag.
TEST(sampler_select, spec_tag_distinguishes_it_from_the_diffusion_selector)
{
  Session sess;
  SamplerSelectStage tok(&sess, "t", {}, FlexData::make_object());
  DiffusionSamplerSelectStage dif(&sess, "d", {}, FlexData::make_object());
  FlexData tf = tok.resolved_spec();
  FlexData df = dif.resolved_spec();
  auto to = tf.as_object();
  auto dobj = df.as_object();
  EXPECT_TRUE(string(to.at("sampler").as_string("")) == "token");
  EXPECT_TRUE(string(dobj.at("sampler").as_string("")) == "flow_match");
  // Neither spec carries the other's keys -- which is exactly why an unchecked
  // swap would read as "everything default" rather than as an error.
  EXPECT_FALSE(to.contains("method"));
  EXPECT_FALSE(dobj.contains("temperature"));
}

#ifdef VPIPE_BUILD_APPLE_SILICON

// The consumer-side parse. The `accepted` flag is what makes the mis-wire
// guard observable: a rejected beat ALSO returns greedy params, so asserting
// only on the params would pass whether or not the guard exists. Callers
// depend on the flag for real -- text-to-speech's audio channel must keep
// MOSS's non-greedy defaults when a beat is rejected.
TEST(sampler_select, consumer_parse_and_miswire_guard)
{
  Session sess;
  CerrSilencer hush;
  const string who = "ut";

  // A real token spec is accepted and resolves to its knobs.
  {
    SamplerSelectStage st(&sess, "s", {},
                          cfg_of({{"temperature", FlexData::make_real(0.5)},
                                  {"top_k", FlexData::make_int(7)}}));
    auto beat = make_payload<FlexDataPayload>(st.resolved_spec());
    bool ok = false;
    const genai::SamplerParams p =
        sampler_params_from_beat(beat.get(), &sess, who, &ok);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(abs(p.temperature - 0.5f) < 1e-6f);
    EXPECT_TRUE(p.top_k == 7);
    EXPECT_FALSE(genai::Sampler(p).is_argmax());
  }
  // A DIFFUSION spec on the same port is REJECTED (and warns). Without the
  // tag check it would parse silently to "every knob default".
  {
    DiffusionSamplerSelectStage dif(&sess, "d", {},
                                    cfg_of({{"method",
                                             FlexData::make_string("heun")}}));
    auto beat = make_payload<FlexDataPayload>(dif.resolved_spec());
    bool ok = true;
    const genai::SamplerParams p =
        sampler_params_from_beat(beat.get(), &sess, who, &ok);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(genai::Sampler(p).is_argmax());
  }
  // A non-FlexData payload likewise.
  {
    bool ok = true;
    const genai::SamplerParams p =
        sampler_params_from_beat(nullptr, &sess, who, &ok);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(genai::Sampler(p).is_argmax());
  }
}

TEST(sampler_select, is_diffusion_sampler_spec_predicate)
{
  Session sess;
  SamplerSelectStage tok(&sess, "t", {}, FlexData::make_object());
  DiffusionSamplerSelectStage dif(&sess, "d", {}, FlexData::make_object());
  EXPECT_TRUE(genai::is_diffusion_sampler_spec(dif.resolved_spec()));
  EXPECT_FALSE(genai::is_diffusion_sampler_spec(tok.resolved_spec()));
  EXPECT_FALSE(genai::is_diffusion_sampler_spec(FlexData::make_object()));
  EXPECT_FALSE(genai::is_diffusion_sampler_spec(FlexData::make_null()));
}

#endif  // VPIPE_BUILD_APPLE_SILICON
