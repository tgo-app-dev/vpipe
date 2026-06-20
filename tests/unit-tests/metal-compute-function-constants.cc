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

// Run the specialized_add kernel with given (bias, mult) and
// inputs; return the (host-readable) output buffer.
std::vector<std::int32_t>
run_specialized_add_(MetalCompute& mc, std::int32_t bias,
                     std::uint32_t mult,
                     const std::vector<std::int32_t>& x)
{
  std::vector<std::int32_t> y(x.size(), 0);
  ComputeLibrary lib = mc.load_library("specialized_add");
  if (!lib.valid()) {
    return y;
  }
  FunctionConstants k;
  k.set_int(0, bias).set_uint(1, mult);
  ComputeFunction fn = lib.function("specialized_add", k);
  if (!fn.valid()) {
    return y;
  }
  const std::size_t n = x.size();
  SharedBuffer xbuf = mc.make_shared_buffer(n * sizeof(std::int32_t));
  SharedBuffer ybuf = mc.make_shared_buffer(n * sizeof(std::int32_t));
  if (xbuf.empty() || ybuf.empty()) {
    return y;
  }
  std::memcpy(xbuf.contents(), x.data(), n * sizeof(std::int32_t));
  std::memset(ybuf.contents(), 0, n * sizeof(std::int32_t));

  CommandStream s = mc.make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, ybuf, 0);
    enc.set_buffer(1, xbuf, 0);
    const std::uint32_t nn = static_cast<std::uint32_t>(n);
    enc.set_constant(2, nn);
    enc.dispatch({static_cast<unsigned>(n), 1, 1}, {32, 1, 1});
  }
  CommandStream::Fence f = s.commit();
  f.wait();

  std::memcpy(y.data(), ybuf.contents(),
              n * sizeof(std::int32_t));
  return y;
}

}  // namespace

TEST(metal_compute_function_constants, set_writes_signature) {
  FunctionConstants a;
  EXPECT_TRUE(a.empty());
  EXPECT_TRUE(a.signature().empty());

  a.set_int(0, 7).set_uint(1, 3);
  EXPECT_FALSE(a.empty());
  // Copy: a.signature() returns a const&, and set_* invalidates
  // the cache backing it.
  std::string s1 = a.signature();
  EXPECT_FALSE(s1.empty());

  // Setting the same slot updates value; signature changes.
  a.set_int(0, 99);
  std::string s2 = a.signature();
  EXPECT_TRUE(s1 != s2);
}

TEST(metal_compute_function_constants,
     signature_is_index_order_stable) {
  FunctionConstants a;
  FunctionConstants b;
  a.set_int(1, 7).set_uint(0, 3);   // set order swapped
  b.set_uint(0, 3).set_int(1, 7);
  EXPECT_TRUE(a.signature() == b.signature());
}

TEST(metal_compute_function_constants,
     different_constants_yield_different_pso) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("specialized_add");
  if (!lib.valid()) {
    return;
  }
  FunctionConstants k1;
  k1.set_int(0, 10).set_uint(1, 3);
  FunctionConstants k2;
  k2.set_int(0, 0).set_uint(1, 2);

  ComputeFunction f1 = lib.function("specialized_add", k1);
  ComputeFunction f2 = lib.function("specialized_add", k2);
  if (!f1.valid() || !f2.valid()) {
    return;
  }
  EXPECT_TRUE(f1.mtl_pso() != f2.mtl_pso());
}

TEST(metal_compute_function_constants,
     same_constants_share_pso) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("specialized_add");
  if (!lib.valid()) {
    return;
  }
  FunctionConstants k;
  k.set_int(0, 42).set_uint(1, 5);
  ComputeFunction a = lib.function("specialized_add", k);
  ComputeFunction b = lib.function("specialized_add", k);
  if (!a.valid() || !b.valid()) {
    return;
  }
  EXPECT_TRUE(a.mtl_pso() == b.mtl_pso());
}

TEST(metal_compute_function_constants, kernel_uses_constants) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  std::vector<std::int32_t> x = {0, 1, 2, 3, 4, 5, 6, 7};

  // bias=10, mult=3 -> y = 3*x + 10
  auto y1 = run_specialized_add_(*mc, 10, 3, x);
  if (y1.size() != x.size()) {
    return;
  }
  bool ok1 = true;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (y1[i] != 3 * x[i] + 10) {
      ok1 = false;
      break;
    }
  }
  EXPECT_TRUE(ok1);

  // bias=0, mult=2 -> y = 2*x
  auto y2 = run_specialized_add_(*mc, 0, 2, x);
  bool ok2 = true;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (y2[i] != 2 * x[i]) {
      ok2 = false;
      break;
    }
  }
  EXPECT_TRUE(ok2);
}
