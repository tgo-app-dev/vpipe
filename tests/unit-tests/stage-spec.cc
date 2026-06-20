#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "pipeline/stage.h"
#include "pipeline/typed-stage.h"

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
