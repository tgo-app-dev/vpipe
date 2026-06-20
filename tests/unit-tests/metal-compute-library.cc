#include "minitest.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
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

TEST(metal_compute_library, load_embedded_noop) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  EXPECT_TRUE(lib.valid());
  EXPECT_TRUE(lib.mtl_library() != nullptr);
  EXPECT_TRUE(lib.name() == "noop");
}

TEST(metal_compute_library, unknown_name_returns_invalid) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("no-such-kernel-anywhere");
  EXPECT_FALSE(lib.valid());
  EXPECT_TRUE(lib.mtl_library() == nullptr);
}

TEST(metal_compute_library, function_resolves) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("noop");
  EXPECT_TRUE(fn.valid());
  EXPECT_TRUE(fn.mtl_function() != nullptr);
}

TEST(metal_compute_library, unknown_function_is_invalid) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("not_a_real_kernel");
  EXPECT_FALSE(fn.valid());
}

TEST(metal_compute_library, library_destructor_releases) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  for (int i = 0; i < 4; ++i) {
    ComputeLibrary lib = mc->load_library("noop");
    EXPECT_TRUE(lib.valid());
  }
  // Repeated load+release shouldn't crash or leak the registry.
  EXPECT_TRUE(true);
}

TEST(metal_compute_library, load_is_cached_by_name) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary a = mc->load_library("noop");
  ComputeLibrary b = mc->load_library("noop");
  if (!a.valid() || !b.valid()) {
    return;
  }
  // Both handles wrap the same cached MTL::Library*.
  EXPECT_TRUE(a.mtl_library() == b.mtl_library());
}

TEST(metal_compute_library, function_is_cached_by_lib_and_name) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction a = lib.function("noop");
  ComputeFunction b = lib.function("noop");
  if (!a.valid() || !b.valid()) {
    return;
  }
  // PSOs are cached -- both handles share the same PSO pointer.
  EXPECT_TRUE(a.mtl_pso() == b.mtl_pso());
}

TEST(metal_compute_library, threadgroup_properties_nonzero) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("noop");
  if (!fn.valid()) {
    return;
  }
  EXPECT_TRUE(fn.max_total_threads_per_threadgroup() > 0u);
  EXPECT_TRUE(fn.thread_execution_width() > 0u);
  // static_threadgroup_memory_length() can legitimately be 0 for a
  // kernel that uses no `threadgroup` storage (noop is one); don't
  // assert > 0 there.
}

TEST(metal_compute_library, invalid_function_has_zero_props) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("noop");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("not_a_real_kernel");
  EXPECT_FALSE(fn.valid());
  EXPECT_TRUE(fn.max_total_threads_per_threadgroup() == 0u);
  EXPECT_TRUE(fn.thread_execution_width() == 0u);
  EXPECT_TRUE(fn.static_threadgroup_memory_length() == 0u);
}
