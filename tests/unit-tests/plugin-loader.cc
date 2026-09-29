// End-to-end plugin loader test: dlopen a real MODULE .dylib built against
// libvpipe (tests/plugins/echo-plugin), confirm its stage type registers
// in the host StageRegistry and instantiates. The plugin lives ONLY in
// that .dylib, so "plugin-echo" is unknown until the load.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"
#include "plugin/plugin-abi.h"
#include "plugin/plugin-manager.h"

#include <iostream>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

using namespace vpipe;

#ifndef VPIPE_TEST_ECHO_PLUGIN
#define VPIPE_TEST_ECHO_PLUGIN ""
#endif

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(std::cerr.rdbuf()) { std::cerr.rdbuf(&_null); }
  ~CerrSilencer() { std::cerr.rdbuf(_saved); }
private:
  struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
  };
  std::streambuf* _saved;
  NullBuf         _null;
};

}  // namespace

TEST(plugin_loader, dlopen_register_instantiate)
{
  const std::string path = VPIPE_TEST_ECHO_PLUGIN;
  ASSERT_TRUE(!path.empty());   // wired by the build (compile definition)

  Session sess;

  // The plugin's stage is defined only in its own .dylib -> not in the
  // host registry until we load it.
  EXPECT_TRUE(StageRegistry::get().find_id("plugin-echo")
              == StageTypeId::unknown);

  EXPECT_TRUE(PluginManager::get().load(&sess, path));
  EXPECT_TRUE(StageRegistry::get().find_id("plugin-echo")
              != StageTypeId::unknown);

  // Loading the same path again is an idempotent no-op (dedup).
  EXPECT_TRUE(PluginManager::get().load(&sess, path));

  // Its StageSpec was attached by the plugin's register call.
  EXPECT_TRUE(StageRegistry::get().spec("plugin-echo") != nullptr);

  // It reported the version it was built for, and it was recorded.
  bool recorded = false;
  for (const auto& r : PluginManager::get().records()) {
    if (r.name == "plugin-echo" || r.path.find("echo") != std::string::npos) {
      recorded = recorded || r.abi == VPIPE_PLUGIN_ABI_VERSION;
    }
  }
  EXPECT_TRUE(recorded);
  // And it constructs through the registry factory.
  auto s = StageRegistry::get().create(
      "plugin-echo", &sess, "e0", std::vector<InEdge>{},
      FlexData::make_object());
  ASSERT_TRUE(s != nullptr);
  EXPECT_TRUE(s->type_name() == "plugin-echo");
}

#ifndef VPIPE_TEST_HANDSHAKE_OLD_ABI
#define VPIPE_TEST_HANDSHAKE_OLD_ABI ""
#define VPIPE_TEST_HANDSHAKE_NO_CONSTANT ""
#define VPIPE_TEST_HANDSHAKE_MISSING_FEATURE ""
#endif

// Outside the window, and read from the FILE: refused without the dylib
// ever being loaded, so its static initialiser never registered a thing
// (the registry did not even mint an id).
TEST(plugin_loader, a_plugin_outside_the_window_is_never_loaded)
{
  Session sess;
  CerrSilencer hush;
  for (const char* path : {VPIPE_TEST_HANDSHAKE_OLD_ABI,
                           VPIPE_TEST_HANDSHAKE_NO_CONSTANT}) {
    ASSERT_TRUE(std::string(path).size() > 0);
    const StageTypeId before = StageRegistry::get().next_id();
    EXPECT_FALSE(PluginManager::get().load(&sess, path));
    EXPECT_TRUE(StageRegistry::get().next_id() == before);
  }
  EXPECT_TRUE(StageRegistry::get().find_id("plugin-handshake-old-abi")
              == StageTypeId::unknown);
  EXPECT_TRUE(StageRegistry::get().find_id("plugin-handshake-no-constant")
              == StageTypeId::unknown);
  EXPECT_FALSE(PluginManager::get().is_loaded_path(
      VPIPE_TEST_HANDSHAKE_OLD_ABI));
}

// Inside the window but asking for a feature this host lacks: the dylib is
// loaded -- its static initialiser DID register a stage -- and refused
// after. The registration is withdrawn: nothing can find or create it.
TEST(plugin_loader, a_missing_feature_is_refused_and_its_stages_withdrawn)
{
  Session sess;
  CerrSilencer hush;
  const std::string path = VPIPE_TEST_HANDSHAKE_MISSING_FEATURE;
  ASSERT_TRUE(!path.empty());
  const StageTypeId before = StageRegistry::get().next_id();
  EXPECT_FALSE(PluginManager::get().load(&sess, path));
  // It ran -- an id was minted while it loaded -- ...
  EXPECT_TRUE(static_cast<unsigned>(StageRegistry::get().next_id()) >
              static_cast<unsigned>(before));
  // ...and left nothing reachable.
  EXPECT_TRUE(StageRegistry::get().find_id(
                  "plugin-handshake-missing-feature") == StageTypeId::unknown);
  EXPECT_TRUE(StageRegistry::get().create(
                  "plugin-handshake-missing-feature", &sess, "x", {}) ==
              nullptr);
}

TEST(plugin_loader, bad_path_is_rejected)
{
  Session sess;
  CerrSilencer hush;   // the load warns through the UI delegate
  EXPECT_FALSE(PluginManager::get().load(&sess,
                                         "/no/such/vpipe-plugin.dylib"));
}
