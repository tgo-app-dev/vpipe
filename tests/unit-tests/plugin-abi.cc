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

#include <string_view>

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

TEST(plugin_abi, version_gate)
{
  EXPECT_TRUE(PluginManager::is_abi_compatible(VPIPE_PLUGIN_ABI_VERSION));
  EXPECT_FALSE(
      PluginManager::is_abi_compatible(VPIPE_PLUGIN_ABI_VERSION + 1u));
  EXPECT_FALSE(PluginManager::is_abi_compatible(0u));
}

TEST(plugin_abi, context_introspection)
{
  Session sess;
  VpipePluginContext ctx(&sess, "test-plugin");
  EXPECT_TRUE(ctx.abi_version() == VPIPE_PLUGIN_ABI_VERSION);
  EXPECT_FALSE(ctx.host_version().empty());
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
