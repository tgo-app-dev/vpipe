#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/texture.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

}  // namespace

TEST(metal_compute_residency, support_probe_returns_bool) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Just check it doesn't crash; older hosts return false.
  const bool supported = mc->residency_set_supported();
  (void)supported;
  EXPECT_TRUE(true);
}

TEST(metal_compute_residency, add_buffer_increments_counter) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(4096);
  if (b.empty()) {
    return;
  }
  const auto before = mc->residency_stats();
  EXPECT_TRUE(mc->residency_add(b));
  EXPECT_TRUE(mc->residency_commit());
  const auto after = mc->residency_stats();
  EXPECT_TRUE(after.add_calls == before.add_calls + 1);
  EXPECT_TRUE(after.current >= before.current + 1);
}

TEST(metal_compute_residency, remove_buffer_decrements) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(4096);
  if (b.empty()) {
    return;
  }
  mc->residency_add(b);
  mc->residency_commit();
  const auto mid = mc->residency_stats();

  EXPECT_TRUE(mc->residency_remove(b));
  EXPECT_TRUE(mc->residency_commit());
  const auto after = mc->residency_stats();
  EXPECT_TRUE(after.remove_calls == mid.remove_calls + 1);
  EXPECT_TRUE(after.current <= mid.current);
}

TEST(metal_compute_residency, add_texture_works) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 16;
  d.height = 16;
  Texture t = mc->make_texture(d);
  if (!t.valid()) {
    return;
  }
  EXPECT_TRUE(mc->residency_add(t));
  EXPECT_TRUE(mc->residency_commit());
  mc->residency_remove(t);
  mc->residency_commit();
}

TEST(metal_compute_residency, request_and_end_are_no_op_safe) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  // Even with no allocations these should not crash.
  EXPECT_TRUE(mc->residency_request());
  EXPECT_TRUE(mc->residency_end());
}

TEST(metal_compute_residency, add_empty_buffer_is_rejected) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer empty;
  EXPECT_FALSE(mc->residency_add(empty));
}
