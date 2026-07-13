#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"

#include <exception>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// A concrete payload type to exercise port typing.
struct SpecTestPayload : public BeatPayloadIntf {
  unique_ptr<BeatPayloadIntf> clone() const override {
    return make_unique<SpecTestPayload>(*this);
  }
  string describe() const override { return "SpecTestPayload"; }
};

// A stage that declares a full StageSpec: a description, the Audio
// category, one typed output port on clock group 1, and three config
// attributes with non-default defaults that live ONLY in the spec.
class SpecTestStage final : public TypedStage<SpecTestStage> {
public:
  static constexpr const char* kTypeName = "ut-spec-test";

  SpecTestStage(const SessionContextIntf* s, string id,
                vector<InEdge> in, FlexData cfg)
    : TypedStage<SpecTestStage>(s, std::move(id), std::move(in),
                                std::move(cfg))
  {
    // Members initialized from the spec defaults (no non-zero member
    // initializers in the class).
    _gain   = attr_real("gain");
    _enable = attr_bool("enable");
    _label  = attr_str("label");
    allocate_oports(spec().oports.size());
  }

  Job process(RuntimeContext& ctx) override {
    ctx.signal_done();
    co_return;
  }

  const StageSpec& spec() const noexcept override;

  double      gain()   const { return _gain; }
  bool        enable() const { return _enable; }
  std::string label()  const { return _label; }

private:
  double      _gain;
  bool        _enable;
  std::string _label;
};

constexpr ConfigKey kAttrs[] = {
  {.key = "gain",   .type = ConfigType::Real, .doc = "linear gain",
   .def_real = 2.5},
  {.key = "enable", .type = ConfigType::Bool, .doc = "on/off",
   .def_bool = true},
  {.key = "label",  .type = ConfigType::String, .doc = "tag",
   .def_str = "hi"},
};
const PortSpec kOports[] = {
  {.name = "out", .doc = "the only output", .type = &typeid(SpecTestPayload),
   .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "ut-spec-test",
  .doc       = "A test stage exercising the StageSpec machinery.",
  .category  = StageCategory::Audio,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};

const StageSpec& SpecTestStage::spec() const noexcept { return kSpec; }

VPIPE_REGISTER_STAGE(SpecTestStage)
VPIPE_REGISTER_SPEC(SpecTestStage, kSpec)

// --- payload-tag test stages ---------------------------------------
// A source and two sinks that all carry the SAME beat type
// (SpecTestPayload) but differ in port TAGS, so a wiring's acceptance
// turns purely on tag compatibility (the beat-type check always passes).

// Source: one typed oport tagged {alpha, beta}. Produces nothing and
// closes -- enough to exercise launch-time edge validation.
class TagSourceStage final : public TypedStage<TagSourceStage> {
public:
  static constexpr const char* kTypeName = "ut-tag-source";
  TagSourceStage(const SessionContextIntf* s, string id,
                 vector<InEdge> in, FlexData cfg)
    : TypedStage<TagSourceStage>(s, std::move(id), std::move(in),
                                 std::move(cfg))
  { allocate_oports(spec().oports.size()); }
  Job process(RuntimeContext& ctx) override {
    ctx.signal_done();
    co_return;
  }
  const StageSpec& spec() const noexcept override;
};
const PortSpec kTagSrcOports[] = {
  {.name = "out", .doc = "tagged output", .type = &typeid(SpecTestPayload),
   .tags = "alpha,beta", .clock_group = 0},
};
const StageSpec kTagSrcSpec = {
  .type_name = "ut-tag-source",
  .doc = "test source with a tagged oport",
  .oports = kTagSrcOports,
};
const StageSpec& TagSourceStage::spec() const noexcept { return kTagSrcSpec; }
VPIPE_REGISTER_STAGE(TagSourceStage)
VPIPE_REGISTER_SPEC(TagSourceStage, kTagSrcSpec)

// Sink whose iport tag {beta} INTERSECTS the source (compatible).
class TagSinkOkStage final : public TypedStage<TagSinkOkStage> {
public:
  static constexpr const char* kTypeName = "ut-tag-sink-ok";
  TagSinkOkStage(const SessionContextIntf* s, string id,
                 vector<InEdge> in, FlexData cfg)
    : TypedStage<TagSinkOkStage>(s, std::move(id), std::move(in),
                                 std::move(cfg)) {}
  Job process(RuntimeContext& ctx) override {
    auto t = co_await ctx.read(0);
    if (!t) { ctx.signal_done(); }
  }
  const StageSpec& spec() const noexcept override;
};
const PortSpec kTagSinkOkIports[] = {
  {.name = "in", .doc = "accepts beta", .type = &typeid(SpecTestPayload),
   .tags = "beta", .clock_group = 0},
};
const StageSpec kTagSinkOkSpec = {
  .type_name = "ut-tag-sink-ok",
  .doc = "test sink whose iport tag intersects the source",
  .iports = kTagSinkOkIports,
};
const StageSpec& TagSinkOkStage::spec() const noexcept {
  return kTagSinkOkSpec;
}
VPIPE_REGISTER_STAGE(TagSinkOkStage)
VPIPE_REGISTER_SPEC(TagSinkOkStage, kTagSinkOkSpec)

// Sink whose iport tag {gamma} is DISJOINT from the source (rejected).
class TagSinkBadStage final : public TypedStage<TagSinkBadStage> {
public:
  static constexpr const char* kTypeName = "ut-tag-sink-bad";
  TagSinkBadStage(const SessionContextIntf* s, string id,
                  vector<InEdge> in, FlexData cfg)
    : TypedStage<TagSinkBadStage>(s, std::move(id), std::move(in),
                                  std::move(cfg)) {}
  Job process(RuntimeContext& ctx) override {
    auto t = co_await ctx.read(0);
    if (!t) { ctx.signal_done(); }
  }
  const StageSpec& spec() const noexcept override;
};
const PortSpec kTagSinkBadIports[] = {
  {.name = "in", .doc = "accepts only gamma", .type = &typeid(SpecTestPayload),
   .tags = "gamma", .clock_group = 0},
};
const StageSpec kTagSinkBadSpec = {
  .type_name = "ut-tag-sink-bad",
  .doc = "test sink whose iport tag is disjoint from the source",
  .iports = kTagSinkBadIports,
};
const StageSpec& TagSinkBadStage::spec() const noexcept {
  return kTagSinkBadSpec;
}
VPIPE_REGISTER_STAGE(TagSinkBadStage)
VPIPE_REGISTER_SPEC(TagSinkBadStage, kTagSinkBadSpec)

}

TEST(stage_spec, category_names_round_trip) {
  EXPECT_TRUE(stage_category_name(StageCategory::Generic) == "generic");
  EXPECT_TRUE(stage_category_name(StageCategory::Audio) == "audio");
  EXPECT_TRUE(stage_category_name(StageCategory::Network) == "network");
}

TEST(stage_spec, defaults_come_from_spec) {
  Session sess;
  // Empty config -> every attribute resolves to its spec default.
  SpecTestStage s(&sess, "s0", {}, FlexData::make_object());
  EXPECT_TRUE(s.gain() == 2.5);
  EXPECT_TRUE(s.enable() == true);
  EXPECT_TRUE(s.label() == "hi");
  EXPECT_TRUE(s.category() == StageCategory::Audio);
  EXPECT_TRUE(s.description()
              == "A test stage exercising the StageSpec machinery.");
}

TEST(stage_spec, config_overrides_spec_default) {
  Session sess;
  FlexData cfg = FlexData::from_json(
      R"({"gain": 4.0, "enable": false, "label": "bye"})");
  SpecTestStage s(&sess, "s0", {}, std::move(cfg));
  EXPECT_TRUE(s.gain() == 4.0);
  EXPECT_TRUE(s.enable() == false);
  EXPECT_TRUE(s.label() == "bye");
}

TEST(stage_spec, ports_and_types_from_spec) {
  Session sess;
  SpecTestStage s(&sess, "s0", {}, FlexData::make_object());
  // allocate_oports(spec().oports.size()) -> exactly one oport.
  EXPECT_TRUE(s.num_oports() == 1u);
  // Payload type + clock group flow from the spec defaults.
  EXPECT_TRUE(s.oport_payload_type(0) == &typeid(SpecTestPayload));
  EXPECT_TRUE(s.oport_clock_group(0) == 1u);
  // config_spec() defaults to spec().attrs (3 keys).
  EXPECT_TRUE(s.config_spec().size() == 3u);
}

TEST(stage_spec, registry_exposes_spec_without_instance) {
  const StageSpec* sp = StageRegistry::get().spec("ut-spec-test");
  ASSERT_TRUE(sp != nullptr);
  EXPECT_TRUE(sp->category == StageCategory::Audio);
  EXPECT_TRUE(sp->oports.size() == 1u);
  EXPECT_TRUE(sp->attrs.size() == 3u);
  // A stage that never registered a spec returns nullptr.
  EXPECT_TRUE(StageRegistry::get().spec("fake-source") == nullptr);
}

// -------- payload tags ---------------------------------------------

TEST(port_tags, compatible_helper_semantics) {
  // No constraint on either side -> compatible.
  EXPECT_TRUE(port_tags_compatible("", ""));
  EXPECT_TRUE(port_tags_compatible("", "rgb-frames"));
  EXPECT_TRUE(port_tags_compatible("rgb-frames", ""));
  // Same single tag intersects.
  EXPECT_TRUE(port_tags_compatible("rgb-frames", "rgb-frames"));
  // OR semantics: any shared tag suffices.
  EXPECT_TRUE(port_tags_compatible("alpha,beta", "beta"));
  EXPECT_TRUE(port_tags_compatible("beta", "alpha,beta"));
  EXPECT_TRUE(port_tags_compatible("a, b ,c", "x,c"));   // whitespace ok
  // Disjoint sets are incompatible.
  EXPECT_FALSE(port_tags_compatible("alpha,beta", "gamma"));
  EXPECT_FALSE(port_tags_compatible("video-encoder-segments",
                                    "audio-encoder-segments"));
}

TEST(port_tags, accessors_read_tags_from_spec) {
  Session sess;
  TagSourceStage src(&sess, "s", {}, FlexData::make_object());
  EXPECT_TRUE(src.oport_payload_tags(0) == "alpha,beta");
  // An out-of-range port yields the empty (no-constraint) view.
  EXPECT_TRUE(src.oport_payload_tags(9).empty());
}

TEST(port_tags, runtime_accepts_intersecting_tags) {
  // beta ∈ {alpha,beta} -> the edge is tag-compatible; launch succeeds.
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);
  auto* src = pl->insert_stage("ut-tag-source", "src", {},
                               FlexData::make_object());
  ASSERT_TRUE(src != nullptr);
  auto* sink = pl->insert_stage("ut-tag-sink-ok", "sink",
                                vector<InEdge>{{src, 0}},
                                FlexData::make_object());
  ASSERT_TRUE(sink != nullptr);
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
}

TEST(port_tags, runtime_rejects_disjoint_tags) {
  // {gamma} ∩ {alpha,beta} = ∅ -> launch throws on the tag mismatch even
  // though both ports carry the same beat type (SpecTestPayload).
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);
  auto* src = pl->insert_stage("ut-tag-source", "src", {},
                               FlexData::make_object());
  ASSERT_TRUE(src != nullptr);
  auto* sink = pl->insert_stage("ut-tag-sink-bad", "sink",
                                vector<InEdge>{{src, 0}},
                                FlexData::make_object());
  ASSERT_TRUE(sink != nullptr);
  PipelineRuntime rt(pl.get(), &sess);
  bool threw = false;
  try {
    rt.launch();
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

// The video chain is the motivating example: rtsp-capture emits both its
// video and audio as EncodedSegment, so the beat type alone can't stop an
// audio segment from being wired into video-to-rgb -- the tags can.
TEST(port_tags, real_video_chain_declares_and_gates_tags) {
  const StageSpec* rtsp = StageRegistry::get().spec("rtsp-capture");
  const StageSpec* v2r  = StageRegistry::get().spec("video-to-rgb");
  const StageSpec* dec  = StageRegistry::get().spec("temporal-decimation");
  ASSERT_TRUE(rtsp != nullptr);
  ASSERT_TRUE(v2r != nullptr);
  ASSERT_TRUE(dec != nullptr);
  // Declared tags.
  ASSERT_TRUE(rtsp->oports.size() >= 2u);
  EXPECT_TRUE(rtsp->oports[0].tags == "video-encoder-segments");
  EXPECT_TRUE(rtsp->oports[1].tags == "audio-encoder-segments");
  EXPECT_TRUE(v2r->iports[0].tags == "video-encoder-segments");
  EXPECT_TRUE(v2r->oports[0].tags == "rgb-frames");
  EXPECT_TRUE(dec->iports[0].tags == "rgb-frames");
  // rtsp VIDEO -> video-to-rgb: compatible. rtsp AUDIO -> video-to-rgb:
  // refused despite the identical beat type.
  EXPECT_TRUE(port_tags_compatible(rtsp->oports[0].tags,
                                   v2r->iports[0].tags));
  EXPECT_FALSE(port_tags_compatible(rtsp->oports[1].tags,
                                    v2r->iports[0].tags));
  // video-to-rgb frames -> temporal-decimation frames: compatible.
  EXPECT_TRUE(port_tags_compatible(v2r->oports[0].tags,
                                   dec->iports[0].tags));
}
