// Plugin foundation tests: the ABI-version gate (pure logic) and the
// VpipePluginContext facade registering a stage end-to-end in-process
// (no dlopen -- that path is exercised by the plugin-loader test once a
// real test-plugin .dylib exists).

#include "minitest.h"

#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/typed-stage.h"
#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"
#include "plugin/plugin-manager.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace vpipe;

namespace {

// A trivial stage defined ONLY here (no VPIPE_REGISTER_STAGE macro),
// mirroring what a plugin does. A TypedStage self-registers its factory
// at load via its vtable, so the facade's material effect we assert below
// is the StageSpec attachment + instantiability, not first-registration.
class PluginAbiTestStage : public TypedStage<PluginAbiTestStage> {
public:
  static constexpr const char* kTypeName = "plugin-abi-test-stage";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

const StageSpec kPluginAbiTestSpec = {
  .type_name = "plugin-abi-test-stage",
  .doc = "plugin ABI test stage",
  .display_name = "Plugin ABI Test",
};

}  // namespace

// The support window is the last TWO versions, never reaching below the
// version that introduced it -- eight, which therefore stands alone.
TEST(plugin_abi, version_window)
{
  EXPECT_TRUE(VPIPE_PLUGIN_ABI_OLDEST <= VPIPE_PLUGIN_ABI_VERSION);
  EXPECT_TRUE(VPIPE_PLUGIN_ABI_VERSION - VPIPE_PLUGIN_ABI_OLDEST <= 1u);
  EXPECT_TRUE(VPIPE_PLUGIN_ABI_OLDEST >= 8u);
  for (std::uint32_t v = VPIPE_PLUGIN_ABI_OLDEST;
       v <= VPIPE_PLUGIN_ABI_VERSION; ++v) {
    EXPECT_TRUE(PluginManager::is_abi_compatible(v));
  }
  EXPECT_FALSE(
      PluginManager::is_abi_compatible(VPIPE_PLUGIN_ABI_VERSION + 1u));
  EXPECT_FALSE(
      PluginManager::is_abi_compatible(VPIPE_PLUGIN_ABI_OLDEST - 1u));
  EXPECT_FALSE(PluginManager::is_abi_compatible(0u));
  // The pre-window versions by name: seven had no window, and nothing
  // built for it can be kept. Literals, so a future bump cannot quietly
  // turn these into restatements of the lines above.
  EXPECT_FALSE(PluginManager::is_abi_compatible(7u));
  EXPECT_FALSE(PluginManager::is_abi_compatible(1u));
}

TEST(plugin_abi, context_introspection)
{
  Session sess;
  VpipePluginContext ctx(&sess, "test-plugin");
  EXPECT_TRUE(ctx.abi_version() == VPIPE_PLUGIN_ABI_VERSION);
  EXPECT_TRUE(ctx.plugin_abi() == VPIPE_PLUGIN_ABI_VERSION);
  EXPECT_FALSE(ctx.host_version().empty());
  VpipePluginContext old(&sess, "older-plugin", VPIPE_PLUGIN_ABI_OLDEST);
  EXPECT_TRUE(old.plugin_abi() == VPIPE_PLUGIN_ABI_OLDEST);
}

// Features are strings a host lists and a plugin asks for. Every one this
// SDK names must be present -- a host that shipped a feature macro it
// does not provide would refuse its own plugins.
TEST(plugin_abi, features)
{
  Session sess;
  VpipePluginContext ctx(&sess, "test-plugin");
  for (const char* f : {VPIPE_FEATURE_STAGE_COMMANDS,
                        VPIPE_FEATURE_KERNEL_CONTRACT,
                        VPIPE_FEATURE_NAMED_OUTPUTS,
                        VPIPE_FEATURE_FAMILY_PROFILES,
                        VPIPE_FEATURE_ACCEL_BAG}) {
    EXPECT_TRUE(ctx.has_feature(f));
    EXPECT_TRUE(host_has_feature(f));
  }
  EXPECT_FALSE(ctx.has_feature("no-such-feature/1"));
  // A revision is part of the name: /2 is not /1.
  EXPECT_FALSE(host_has_feature("stage-commands/2"));
  EXPECT_TRUE(host_features().size() >= 5);
}

namespace {
class QuarantineStageA : public TypedStage<QuarantineStageA> {
public:
  static constexpr const char* kTypeName = "ut-quarantine-a";
  using TypedStage::TypedStage;
  Job process(RuntimeContext& ctx) override { ctx.signal_done(); co_return; }
};
StagePtr
quarantine_factory_(const SessionContextIntf* s, std::string id,
                    std::vector<InEdge> iports, FlexData cfg)
{
  return std::make_unique<QuarantineStageA>(s, std::move(id),
                                            std::move(iports),
                                            std::move(cfg));
}
const StageSpec kQuarantineSpec = {.type_name = "ut-quarantine-b"};
}  // namespace

// What a plugin registers while it loads is invisible until the handshake
// admits it, and gone for good when it does not.
TEST(plugin_abi, registry_quarantine)
{
  Session sess;
  StageRegistry& r = StageRegistry::get();

  r.begin_quarantine();
  const StageTypeId b = r.register_type("ut-quarantine-b",
                                        &quarantine_factory_);
  r.set_spec("ut-quarantine-b", &kQuarantineSpec);
  EXPECT_TRUE(b != StageTypeId::unknown);             // an id is minted...
  EXPECT_TRUE(r.find_id("ut-quarantine-b") == StageTypeId::unknown);
  EXPECT_TRUE(r.spec("ut-quarantine-b") == nullptr);  // ...but unseen
  EXPECT_TRUE(r.create("ut-quarantine-b", &sess, "x", {}) == nullptr);
  r.end_quarantine(false);
  EXPECT_TRUE(r.find_id("ut-quarantine-b") == StageTypeId::unknown);
  EXPECT_TRUE(r.spec("ut-quarantine-b") == nullptr);

  // The name is free again, and a registration that is admitted is
  // ordinary from then on.
  r.begin_quarantine();
  const StageTypeId c = r.register_type("ut-quarantine-b",
                                        &quarantine_factory_);
  EXPECT_TRUE(c != b);
  r.end_quarantine(true);
  EXPECT_TRUE(r.find_id("ut-quarantine-b") == c);
  EXPECT_TRUE(r.create("ut-quarantine-b", &sess, "x", {}) != nullptr);
}

TEST(plugin_abi, context_registers_and_instantiates_stage)
{
  Session sess;
  VpipePluginContext ctx(&sess, "test-plugin");

  ctx.register_stage<PluginAbiTestStage>(&kPluginAbiTestSpec);

  // Registered + describable (the spec attach is the facade's own effect).
  EXPECT_TRUE(StageRegistry::get().find_id("plugin-abi-test-stage")
              != StageTypeId::unknown);
  EXPECT_TRUE(StageRegistry::get().spec("plugin-abi-test-stage")
              == &kPluginAbiTestSpec);

  // And it constructs through the registry factory.
  auto s = StageRegistry::get().create(
      "plugin-abi-test-stage", &sess, "s0", std::vector<InEdge>{},
      FlexData::make_object());
  ASSERT_TRUE(s != nullptr);
  EXPECT_TRUE(s->type_name() == "plugin-abi-test-stage");

  // Re-registering the same type is a first-wins no-op (no crash).
  ctx.register_stage<PluginAbiTestStage>(&kPluginAbiTestSpec);
  EXPECT_TRUE(StageRegistry::get().find_id("plugin-abi-test-stage")
              != StageTypeId::unknown);
}
