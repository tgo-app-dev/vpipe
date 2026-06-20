#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <vector>

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

TEST(metal_compute_heap, small_tracked_alloc_goes_to_heap) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  const auto before = mc->alloc_stats();
  SharedBuffer b = mc->make_shared_buffer(4096);
  EXPECT_FALSE(b.empty());
  const auto after = mc->alloc_stats();
  EXPECT_TRUE(after.buffers_from_heap == before.buffers_from_heap + 1);
  EXPECT_TRUE(after.buffers_from_device == before.buffers_from_device);
}

TEST(metal_compute_heap, large_alloc_bypasses_heap) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  const auto before = mc->alloc_stats();
  // > 64KB threshold -> direct device alloc.
  SharedBuffer b = mc->make_shared_buffer(256 * 1024);
  EXPECT_FALSE(b.empty());
  const auto after = mc->alloc_stats();
  EXPECT_TRUE(after.buffers_from_heap == before.buffers_from_heap);
  EXPECT_TRUE(after.buffers_from_device
              == before.buffers_from_device + 1);
}

TEST(metal_compute_heap, untracked_bypasses_heap) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  const auto before = mc->alloc_stats();
  // Small but Untracked -> bypass heap (heap is Tracked-fixed).
  SharedBuffer b = mc->make_shared_buffer(
      4096, 64, HazardTracking::Untracked);
  EXPECT_FALSE(b.empty());
  const auto after = mc->alloc_stats();
  EXPECT_TRUE(after.buffers_from_heap == before.buffers_from_heap);
  EXPECT_TRUE(after.buffers_from_device
              == before.buffers_from_device + 1);
}

TEST(metal_compute_heap, many_small_allocs_share_heap_space) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Allocate 64 x 16KB = 1 MB worth of small buffers; the 4 MB heap
  // should accommodate all of them without spilling to the device.
  const auto before = mc->alloc_stats();
  std::vector<SharedBuffer> bs;
  bs.reserve(64);
  for (int i = 0; i < 64; ++i) {
    bs.push_back(mc->make_shared_buffer(16 * 1024));
  }
  const auto after = mc->alloc_stats();
  EXPECT_TRUE(after.buffers_from_heap
              >= before.buffers_from_heap + 64);
  // None should have spilled to direct device alloc.
  EXPECT_TRUE(after.buffers_from_device == before.buffers_from_device);
}

TEST(metal_compute_heap, released_heap_buffer_returns_space) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Fill the heap by allocating until it spills, then release and
  // verify a subsequent small alloc can come from the heap again.
  std::vector<SharedBuffer> bs;
  bs.reserve(256);
  for (int i = 0; i < 256; ++i) {
    bs.push_back(mc->make_shared_buffer(64 * 1024));
  }
  // Drop everything.
  bs.clear();
  // Even if the prior loop spilled, a fresh small alloc post-clear
  // should fit in the heap again.
  const auto before = mc->alloc_stats();
  SharedBuffer b = mc->make_shared_buffer(4096);
  EXPECT_FALSE(b.empty());
  const auto after = mc->alloc_stats();
  EXPECT_TRUE(after.buffers_from_heap == before.buffers_from_heap + 1);
}

TEST(metal_compute_heap, heap_buffer_is_usable) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(1024);
  EXPECT_FALSE(b.empty());
  EXPECT_TRUE(b.contents() != nullptr);
  // Heap-sub-allocated buffers must respect alignment and UMA.
  auto* p = static_cast<std::uint8_t*>(b.contents());
  for (std::size_t i = 0; i < 1024; ++i) {
    p[i] = static_cast<std::uint8_t>(i & 0xff);
  }
  for (std::size_t i = 0; i < 1024; ++i) {
    EXPECT_TRUE(p[i] == static_cast<std::uint8_t>(i & 0xff));
  }
}
