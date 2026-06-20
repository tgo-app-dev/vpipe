#include "minitest.h"
#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <cstring>
#include <utility>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

// Helper: get a valid() MetalCompute on this host, or nullptr if
// Metal is unavailable (in which case the test self-skips).
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

TEST(metal_compute_shared_buffer, alloc_and_release) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  {
    SharedBuffer buf = mc->make_shared_buffer(4096);
    EXPECT_FALSE(buf.empty());
    EXPECT_TRUE(buf.contents() != nullptr);
    EXPECT_TRUE(buf.byte_size() == 4096u);
    EXPECT_TRUE(buf.mtl_buffer() != nullptr);
  }
  // Destruction completes without crashing.
  EXPECT_TRUE(true);
}

TEST(metal_compute_shared_buffer, contents_is_64_byte_aligned) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(1024);
  const auto ptr = reinterpret_cast<std::uintptr_t>(buf.contents());
  EXPECT_TRUE((ptr & 63u) == 0u);
}

TEST(metal_compute_shared_buffer, zero_size_is_empty) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(0);
  EXPECT_TRUE(buf.empty());
  EXPECT_TRUE(buf.contents() == nullptr);
  EXPECT_TRUE(buf.byte_size() == 0u);
}

TEST(metal_compute_shared_buffer, contents_round_trips_through_cpu) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(256);
  auto* p = static_cast<std::uint8_t*>(buf.contents());
  for (std::size_t i = 0; i < buf.byte_size(); ++i) {
    p[i] = static_cast<std::uint8_t>(i ^ 0xA5);
  }
  for (std::size_t i = 0; i < buf.byte_size(); ++i) {
    EXPECT_TRUE(p[i] == static_cast<std::uint8_t>(i ^ 0xA5));
  }
}

TEST(metal_compute_shared_buffer, view_setter_round_trips) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(64);
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 2;
  v.shape[0]   = 4;
  v.shape[1]   = 4;
  v.strides[0] = 4;
  v.strides[1] = 1;
  v.offset     = 0;
  buf.set_view(v);
  EXPECT_TRUE(buf.view().dtype == DType::F32);
  EXPECT_TRUE(buf.view().rank == 2);
  EXPECT_TRUE(buf.view().shape[0] == 4);
  EXPECT_TRUE(buf.view().shape[1] == 4);
  EXPECT_TRUE(buf.view().strides[0] == 4);
  EXPECT_TRUE(buf.view().strides[1] == 1);
}

TEST(metal_compute_shared_buffer, move_ctor_transfers_ownership) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer src = mc->make_shared_buffer(128);
  EXPECT_FALSE(src.empty());
  void*       contents_before = src.contents();
  MTL::Buffer* mtl_before     = src.mtl_buffer();

  SharedBuffer dst(std::move(src));
  EXPECT_TRUE(src.empty());
  EXPECT_TRUE(src.contents() == nullptr);
  EXPECT_TRUE(src.byte_size() == 0u);
  EXPECT_TRUE(dst.contents() == contents_before);
  EXPECT_TRUE(dst.mtl_buffer() == mtl_before);
  EXPECT_TRUE(dst.byte_size() == 128u);
}

TEST(metal_compute_shared_buffer, move_assignment_releases_old) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer a = mc->make_shared_buffer(256);
  SharedBuffer b = mc->make_shared_buffer(512);
  a = std::move(b);
  EXPECT_TRUE(b.empty());
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a.byte_size() == 512u);
}

TEST(metal_compute_shared_buffer, set_wired_round_trip) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Use one page so the lock fits under the default RLIMIT_MEMLOCK
  // (64KB) on a non-root macOS process.
  SharedBuffer buf = mc->make_shared_buffer(4096);
  void* contents_before = buf.contents();

  // mlock can fail on hosts with tight RLIMIT_MEMLOCK or in
  // sandboxes; if so the test self-skips the wired assertions.
  if (!buf.set_wired(true)) {
    return;
  }
  EXPECT_TRUE(buf.is_wired());
  EXPECT_TRUE(buf.contents() == contents_before);

  // Idempotent: second set_wired(true) is a no-op success.
  EXPECT_TRUE(buf.set_wired(true));
  EXPECT_TRUE(buf.is_wired());

  EXPECT_TRUE(buf.set_wired(false));
  EXPECT_FALSE(buf.is_wired());
  EXPECT_TRUE(buf.contents() == contents_before);

  // Idempotent: second set_wired(false) is a no-op success.
  EXPECT_TRUE(buf.set_wired(false));
  EXPECT_FALSE(buf.is_wired());
}

TEST(metal_compute_shared_buffer, destructor_unwires_automatically) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  {
    SharedBuffer buf = mc->make_shared_buffer(4096);
    if (!buf.set_wired(true)) {
      return;
    }
    EXPECT_TRUE(buf.is_wired());
    // Destructor runs at end of scope and should munlock + release
    // without crashing.
  }
  EXPECT_TRUE(true);
}

TEST(metal_compute_shared_buffer, set_wired_on_empty_is_noop) {
  SharedBuffer empty;
  EXPECT_TRUE(empty.set_wired(true));
  EXPECT_FALSE(empty.is_wired());
  EXPECT_TRUE(empty.set_wired(false));
}

TEST(metal_compute_shared_buffer, default_constructed_is_empty) {
  SharedBuffer buf;
  EXPECT_TRUE(buf.empty());
  EXPECT_TRUE(buf.contents() == nullptr);
  EXPECT_TRUE(buf.byte_size() == 0u);
  EXPECT_TRUE(buf.mtl_buffer() == nullptr);
  EXPECT_FALSE(buf.is_wired());
}
