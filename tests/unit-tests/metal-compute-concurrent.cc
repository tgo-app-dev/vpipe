#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <cstring>
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

TEST(metal_compute_concurrent,
     two_independent_dispatches_concurrent) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("concurrent_demo");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction write = lib.function("write_constant");
  if (!write.valid()) {
    return;
  }
  constexpr std::size_t N = 256;
  SharedBuffer a = mc->make_shared_buffer(
      N * sizeof(std::uint32_t), 64, HazardTracking::Untracked);
  SharedBuffer b = mc->make_shared_buffer(
      N * sizeof(std::uint32_t), 64, HazardTracking::Untracked);
  if (a.empty() || b.empty()) {
    return;
  }
  std::memset(a.contents(), 0, N * sizeof(std::uint32_t));
  std::memset(b.contents(), 0, N * sizeof(std::uint32_t));

  CommandStream s = mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute(DispatchType::Concurrent);
    const std::uint32_t n = N;
    const std::uint32_t va = 42;
    const std::uint32_t vb = 7;
    // Two dispatches writing to DIFFERENT buffers: no hazard, can
    // legitimately run concurrently. No barrier needed.
    enc.set_function(write);
    enc.set_buffer(0, a, 0);
    enc.set_constant(1, n);
    enc.set_constant(2, va);
    enc.dispatch({N, 1, 1}, {32, 1, 1});

    enc.set_function(write);
    enc.set_buffer(0, b, 0);
    enc.set_constant(1, n);
    enc.set_constant(2, vb);
    enc.dispatch({N, 1, 1}, {32, 1, 1});
  }
  CommandStream::Fence f = s.commit();
  f.wait();
  EXPECT_TRUE(f.completed());

  const auto* pa = static_cast<const std::uint32_t*>(a.contents());
  const auto* pb = static_cast<const std::uint32_t*>(b.contents());
  bool ok = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (pa[i] != 42u || pb[i] != 7u) { ok = false; break; }
  }
  EXPECT_TRUE(ok);
}

TEST(metal_compute_concurrent,
     dependent_chain_via_explicit_barrier) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("concurrent_demo");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction write = lib.function("write_constant");
  ComputeFunction read_write = lib.function("read_then_write");
  if (!write.valid() || !read_write.valid()) {
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
  std::memset(dst.contents(), 0, N * sizeof(std::uint32_t));

  CommandStream s = mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute(DispatchType::Concurrent);
    const std::uint32_t n = N;
    const std::uint32_t init  = 100;
    const std::uint32_t add   = 5;

    // First fill src.
    enc.set_function(write);
    enc.set_buffer(0, src, 0);
    enc.set_constant(1, n);
    enc.set_constant(2, init);
    enc.dispatch({N, 1, 1}, {32, 1, 1});

    // Concurrent encoder + Untracked buffers means Metal will NOT
    // insert a barrier on its own; without this call the read
    // below could race the write above.
    enc.memory_barrier(BarrierScope::Buffers);

    // Now read src + add value, write to dst.
    enc.set_function(read_write);
    enc.set_buffer(0, src, 0);
    enc.set_buffer(1, dst, 0);
    enc.set_constant(2, n);
    enc.set_constant(3, add);
    enc.dispatch({N, 1, 1}, {32, 1, 1});
  }
  CommandStream::Fence f = s.commit();
  f.wait();
  EXPECT_TRUE(f.completed());

  const auto* p = static_cast<const std::uint32_t*>(dst.contents());
  bool ok = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (p[i] != 105u) { ok = false; break; }
  }
  EXPECT_TRUE(ok);
}

TEST(metal_compute_concurrent, serial_encoder_is_default) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  // begin_compute() with no arg should still work (default Serial).
  ComputeEncoder enc = s.begin_compute();
  EXPECT_TRUE(enc.valid());
}

TEST(metal_compute_concurrent, memory_barrier_no_op_on_empty) {
  ComputeEncoder enc;
  // No crash; just covers the empty-encoder guard.
  enc.memory_barrier(BarrierScope::Both);
  EXPECT_TRUE(true);
}

TEST(metal_compute_concurrent,
     serial_encoder_orders_dispatches_implicitly) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("concurrent_demo");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction write = lib.function("write_constant");
  ComputeFunction read_write = lib.function("read_then_write");
  if (!write.valid() || !read_write.valid()) {
    return;
  }
  // Default Serial encoder + Tracked buffers: Metal inserts
  // implicit barriers between dependent dispatches. Same workload
  // as the explicit-barrier test, NO explicit barrier, but result
  // is correct because Metal orders us.
  constexpr std::size_t N = 256;
  SharedBuffer src = mc->make_shared_buffer(N * sizeof(std::uint32_t));
  SharedBuffer dst = mc->make_shared_buffer(N * sizeof(std::uint32_t));
  CommandStream s = mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();   // Serial (default)
    const std::uint32_t n = N;
    const std::uint32_t init = 50;
    const std::uint32_t add  = 1;
    enc.set_function(write);
    enc.set_buffer(0, src, 0);
    enc.set_constant(1, n);
    enc.set_constant(2, init);
    enc.dispatch({N, 1, 1}, {32, 1, 1});
    enc.set_function(read_write);
    enc.set_buffer(0, src, 0);
    enc.set_buffer(1, dst, 0);
    enc.set_constant(2, n);
    enc.set_constant(3, add);
    enc.dispatch({N, 1, 1}, {32, 1, 1});
  }
  s.commit().wait();
  const auto* p = static_cast<const std::uint32_t*>(dst.contents());
  bool ok = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (p[i] != 51u) { ok = false; break; }
  }
  EXPECT_TRUE(ok);
}
