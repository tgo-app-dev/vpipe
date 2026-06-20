#include "minitest.h"
#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

// y = a*x + y on the CPU, as the reference oracle.
void
cpu_saxpy_ref_(float a,
               const std::vector<float>& x,
               std::vector<float>* y)
{
  for (std::size_t i = 0; i < x.size(); ++i) {
    (*y)[i] = a * x[i] + (*y)[i];
  }
}

// Build a contiguous rank-1 BufferView over n F32 elements.
BufferView
contig_view_(std::size_t n)
{
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 1;
  v.shape[0]   = static_cast<std::int64_t>(n);
  v.strides[0] = 1;
  v.offset     = 0;
  return v;
}

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

// Fill buf->contents() (cast to float*) with values [0..n).
void
fill_iota_(SharedBuffer& buf, std::size_t n, float scale = 1.0f)
{
  auto* p = static_cast<float*>(buf.contents());
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = static_cast<float>(i) * scale;
  }
}

bool
max_abs_diff_under_(const float* a, const float* b,
                    std::size_t n, float tol)
{
  for (std::size_t i = 0; i < n; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(metal_compute_saxpy, contiguous_1024) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("saxpy");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("saxpy");
  if (!fn.valid()) {
    return;
  }

  constexpr std::size_t N = 1024;
  constexpr float       A = 2.5f;

  SharedBuffer xbuf = mc->make_shared_buffer(N * sizeof(float));
  SharedBuffer ybuf = mc->make_shared_buffer(N * sizeof(float));
  EXPECT_FALSE(xbuf.empty());
  EXPECT_FALSE(ybuf.empty());

  fill_iota_(xbuf, N, 1.0f);          // x = 0,1,2,3,...
  fill_iota_(ybuf, N, 0.5f);          // y = 0,0.5,1,1.5,...

  // Capture CPU reference using y's initial values.
  std::vector<float> x_ref(N), y_ref(N);
  std::memcpy(x_ref.data(), xbuf.contents(), N * sizeof(float));
  std::memcpy(y_ref.data(), ybuf.contents(), N * sizeof(float));
  cpu_saxpy_ref_(A, x_ref, &y_ref);

  xbuf.set_view(contig_view_(N));
  ybuf.set_view(contig_view_(N));

  struct SaxpyParams { float a; };
  SaxpyParams params{ A };

  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    EXPECT_TRUE(enc.valid());
    enc.set_function(fn);
    enc.set_buffer_view(/*buf*/ 0, xbuf, /*meta*/ 1);
    enc.set_buffer_view(/*buf*/ 2, ybuf, /*meta*/ 3);
    enc.set_constant(/*index*/ 4, params);
    const unsigned tew = fn.thread_execution_width();
    const unsigned tg  = tew > 0 ? tew : 32;
    enc.dispatch({N, 1, 1}, {tg, 1, 1});
  }
  CommandStream::Fence fence = stream.commit();
  EXPECT_TRUE(fence.valid());
  fence.wait();
  EXPECT_TRUE(fence.completed());

  const auto* gpu = static_cast<const float*>(ybuf.contents());
  EXPECT_TRUE(max_abs_diff_under_(gpu, y_ref.data(), N, 1e-5f));
}

TEST(metal_compute_saxpy, strided_every_other_element) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("saxpy");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("saxpy");
  if (!fn.valid()) {
    return;
  }

  // Buffers hold 2048 floats; logical view covers indices 0,2,4,...
  // i.e. shape=1024, stride=2, offset=0. The other half of each
  // buffer must remain untouched.
  constexpr std::size_t N_PHYS = 2048;
  constexpr std::size_t N_LOG  = 1024;
  constexpr float       A      = -1.25f;

  SharedBuffer xbuf = mc->make_shared_buffer(N_PHYS * sizeof(float));
  SharedBuffer ybuf = mc->make_shared_buffer(N_PHYS * sizeof(float));
  if (xbuf.empty() || ybuf.empty()) {
    return;
  }
  fill_iota_(xbuf, N_PHYS, 0.25f);
  fill_iota_(ybuf, N_PHYS, 0.125f);

  // Reference: apply saxpy only at even indices.
  std::vector<float> y_ref(N_PHYS);
  std::memcpy(y_ref.data(), ybuf.contents(),
              N_PHYS * sizeof(float));
  const auto* xp = static_cast<const float*>(xbuf.contents());
  for (std::size_t i = 0; i < N_LOG; ++i) {
    y_ref[2 * i] = A * xp[2 * i] + y_ref[2 * i];
  }

  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 1;
  v.shape[0]   = static_cast<std::int64_t>(N_LOG);
  v.strides[0] = 2;
  v.offset     = 0;
  xbuf.set_view(v);
  ybuf.set_view(v);

  struct SaxpyParams { float a; };
  SaxpyParams params{ A };

  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer_view(0, xbuf, 1);
    enc.set_buffer_view(2, ybuf, 3);
    enc.set_constant(4, params);
    const unsigned tew = fn.thread_execution_width();
    const unsigned tg  = tew > 0 ? tew : 32;
    enc.dispatch({N_LOG, 1, 1}, {tg, 1, 1});
  }
  CommandStream::Fence fence = stream.commit();
  fence.wait();
  EXPECT_TRUE(fence.completed());

  const auto* gpu = static_cast<const float*>(ybuf.contents());
  EXPECT_TRUE(max_abs_diff_under_(gpu, y_ref.data(),
                                  N_PHYS, 1e-5f));
}

TEST(metal_compute_saxpy, repeated_dispatches_accumulate) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("saxpy");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("saxpy");
  if (!fn.valid()) {
    return;
  }

  // Sanity that running saxpy three times on the same y buffer
  // chains: y <- a*x + y, repeated three times => y_final =
  // 3*a*x + y_initial. Exercises the PSO cache + buffer
  // statefulness via UMA.
  constexpr std::size_t N = 64;
  constexpr float       A = 0.5f;

  SharedBuffer xbuf = mc->make_shared_buffer(N * sizeof(float));
  SharedBuffer ybuf = mc->make_shared_buffer(N * sizeof(float));
  if (xbuf.empty() || ybuf.empty()) {
    return;
  }
  fill_iota_(xbuf, N, 1.0f);
  fill_iota_(ybuf, N, 0.0f);   // y = 0,0,0,...
  // y will receive 3 * 0.5 * x = 1.5*x = 0,1.5,3,4.5,...

  xbuf.set_view(contig_view_(N));
  ybuf.set_view(contig_view_(N));

  struct SaxpyParams { float a; };
  SaxpyParams params{ A };

  for (int rep = 0; rep < 3; ++rep) {
    CommandStream stream = mc->make_command_stream();
    {
      ComputeEncoder enc = stream.begin_compute();
      enc.set_function(fn);
      enc.set_buffer_view(0, xbuf, 1);
      enc.set_buffer_view(2, ybuf, 3);
      enc.set_constant(4, params);
      const unsigned tg = fn.thread_execution_width();
      enc.dispatch({N, 1, 1}, {tg > 0 ? tg : 32, 1, 1});
    }
    CommandStream::Fence f = stream.commit();
    f.wait();
    EXPECT_TRUE(f.completed());
  }

  std::vector<float> y_ref(N);
  for (std::size_t i = 0; i < N; ++i) {
    y_ref[i] = 3.0f * A * static_cast<float>(i);
  }
  const auto* gpu = static_cast<const float*>(ybuf.contents());
  EXPECT_TRUE(max_abs_diff_under_(gpu, y_ref.data(), N, 1e-5f));
}
