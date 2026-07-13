// End-to-end plugin loader test: dlopen a real MODULE .dylib built against
// libvpipe (tests/plugins/echo-plugin), confirm its stage type registers
// in the host StageRegistry and instantiates. The plugin lives ONLY in
// that .dylib, so "plugin-echo" is unknown until the load.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"
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

  // And it constructs through the registry factory.
  auto s = StageRegistry::get().create(
      "plugin-echo", &sess, "e0", std::vector<InEdge>{},
      FlexData::make_object());
  ASSERT_TRUE(s != nullptr);
  EXPECT_TRUE(s->type_name() == "plugin-echo");
}

TEST(plugin_loader, bad_path_is_rejected)
{
  Session sess;
  CerrSilencer hush;   // the load warns through the UI delegate
  EXPECT_FALSE(PluginManager::get().load(&sess,
                                         "/no/such/vpipe-plugin.dylib"));
}
