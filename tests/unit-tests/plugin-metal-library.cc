// Runtime metal-library registration (the plugin shader extension point):
// register a metallib's bytes at runtime through the plugin facade, then
// prove it loads by name and its entry point resolves to a pipeline state
// -- exactly the path a plugin's offline-compiled shaders take. Skips
// vacuously when no Metal device is available.

#include "minitest.h"

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "plugin/plugin-context.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <streambuf>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

#ifndef VPIPE_TEST_PROBE_METALLIB
#define VPIPE_TEST_PROBE_METALLIB ""
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

TEST(plugin_metal_library, register_load_resolve)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return;   // no Metal device -> skip
  }

  const std::string path = VPIPE_TEST_PROBE_METALLIB;
  ASSERT_TRUE(!path.empty());   // wired by the build (compile definition)
  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(static_cast<bool>(in));
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  ASSERT_TRUE(!bytes.empty());

  // Register through the plugin facade (facade -> MetalCompute).
  VpipePluginContext ctx(&sess, "metal-test");
  EXPECT_TRUE(ctx.register_metal_library("plugin_probe_rt",
                                         bytes.data(), bytes.size()));

  // Now it loads by name like a built-in kernel, and its entry point
  // resolves to a compute pipeline state.
  ComputeLibrary lib = mc->load_library("plugin_probe_rt");
  EXPECT_TRUE(lib.valid());
  ComputeFunction fn = lib.function("plugin_probe");
  EXPECT_TRUE(fn.valid());

  // First-wins: a second registration under the same name is rejected.
  {
    CerrSilencer hush;   // the reject path warns through the delegate
    EXPECT_FALSE(ctx.register_metal_library("plugin_probe_rt",
                                            bytes.data(), bytes.size()));
  }

  // The from-file entry point works too (under a fresh name).
  EXPECT_TRUE(mc->register_metal_library_file("plugin_probe_rt2", path));
  EXPECT_TRUE(mc->load_library("plugin_probe_rt2")
                  .function("plugin_probe").valid());
}
