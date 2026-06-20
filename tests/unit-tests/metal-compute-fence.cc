#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/fence.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <cstring>

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

TEST(metal_compute_fence, make_fence_is_valid) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  Fence f = mc->make_fence();
  EXPECT_TRUE(f.valid());
  EXPECT_TRUE(f.mtl_fence() != nullptr);
}

TEST(metal_compute_fence, default_fence_is_invalid) {
  Fence f;
  EXPECT_FALSE(f.valid());
  EXPECT_TRUE(f.mtl_fence() == nullptr);
}

TEST(metal_compute_fence, untracked_buffer_allocates) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(
      1024, 64, HazardTracking::Untracked);
  EXPECT_FALSE(b.empty());
  EXPECT_TRUE(b.byte_size() == 1024u);
}

TEST(metal_compute_fence, two_encoders_with_fence_correct) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("fence_demo");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fill = lib.function("fill_with_tid");
  ComputeFunction copy = lib.function("copy_uint");
  if (!fill.valid() || !copy.valid()) {
    return;
  }
  constexpr std::size_t N = 256;
  SharedBuffer src = mc->make_shared_buffer(
      N * sizeof(std::uint32_t), 64, HazardTracking::Untracked);
  SharedBuffer dst = mc->make_shared_buffer(
      N * sizeof(std::uint32_t), 64, HazardTracking::Untracked);
  if (src.empty() || dst.empty()) {
    return;
  }
  // Zero dst so any miss is obvious; src can be uninitialised
  // because the fill kernel writes every slot.
  std::memset(dst.contents(), 0, N * sizeof(std::uint32_t));

  Fence f = mc->make_fence();
  if (!f.valid()) {
    return;
  }

  CommandStream s = mc->make_command_stream();
  {
    ComputeEncoder a = s.begin_compute();
    a.set_function(fill);
    a.set_buffer(0, src, 0);
    const std::uint32_t nn = static_cast<std::uint32_t>(N);
    a.set_constant(1, nn);
    a.dispatch({static_cast<unsigned>(N), 1, 1}, {32, 1, 1});
    a.update_fence(f);
  }
  {
    ComputeEncoder b = s.begin_compute();
    b.wait_for_fence(f);
    b.set_function(copy);
    b.set_buffer(0, src, 0);
    b.set_buffer(1, dst, 0);
    const std::uint32_t nn = static_cast<std::uint32_t>(N);
    b.set_constant(2, nn);
    b.dispatch({static_cast<unsigned>(N), 1, 1}, {32, 1, 1});
  }
  CommandStream::Fence cb = s.commit();
  cb.wait();
  EXPECT_TRUE(cb.completed());

  const auto* gpu = static_cast<const std::uint32_t*>(dst.contents());
  bool ok = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (gpu[i] != static_cast<std::uint32_t>(i)) {
      ok = false;
      break;
    }
  }
  EXPECT_TRUE(ok);
}
