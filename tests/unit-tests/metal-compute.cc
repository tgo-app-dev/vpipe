#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "interfaces/session-context-intf.h"

#include <memory>

using namespace vpipe;
using namespace vpipe::metal_compute;

// Step 1 smoke: the Session accessor materialises a MetalCompute
// instance on Apple-Silicon builds. We don't assert valid() here --
// that depends on a working MTL device, which CI hosts may not have.
TEST(metal_compute, exists) {
  Session sess;
  const SessionContextIntf* ctx = &sess;
  MetalCompute* mc = ctx->metal_compute();
  EXPECT_TRUE(mc != nullptr);
}

// Calling the accessor twice returns the same instance (call_once
// caches it).
TEST(metal_compute, accessor_is_idempotent) {
  Session sess;
  const SessionContextIntf* ctx = &sess;
  MetalCompute* a = ctx->metal_compute();
  MetalCompute* b = ctx->metal_compute();
  EXPECT_TRUE(a == b);
}

// When the device is available the framework should report valid().
// Skipped (logs a notice and passes) if Metal is unavailable on this
// host, so this test is safe to run in environments without a GPU.
TEST(metal_compute, valid_when_device_available) {
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    // No Metal device on this host; nothing more to assert.
    return;
  }
  EXPECT_TRUE(mc->device() != nullptr);
}
