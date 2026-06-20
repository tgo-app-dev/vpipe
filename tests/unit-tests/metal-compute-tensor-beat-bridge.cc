#include "minitest.h"
#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/tensor-beat-bridge.h"
#include "apple-silicon/tensor-beat.h"
#include "apple-silicon/tensor-storage.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <cstring>
#include <utility>
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

// Build a row-major contiguous CpuCached TensorBeat with shape `s`
// and dtype F32. Fills with `1, 2, 3, ...` for distinctness.
TensorBeat
make_cpu_f32_(const std::vector<int64_t>& s)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = s;
  std::size_t n = 1;
  for (auto d : s) {
    n *= static_cast<std::size_t>(d);
  }
  tb.resize_contiguous(n);
  float* p = tb.as_f32();
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = static_cast<float>(i + 1);
  }
  return tb;
}

}  // namespace

TEST(metal_compute_tensor_beat_bridge, from_empty_yields_empty) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TensorBeat tb;  // default: CpuCached, empty data
  SharedBuffer sb = from_tensor_beat(*mc, tb);
  EXPECT_TRUE(sb.empty());
}

TEST(metal_compute_tensor_beat_bridge, from_cpu_cached_copies_bytes) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TensorBeat tb = make_cpu_f32_({4, 8});  // 32 elements
  SharedBuffer sb = from_tensor_beat(*mc, tb);
  if (sb.empty()) {
    return;
  }
  EXPECT_TRUE(sb.byte_size() == tb.byte_size());

  // The shared buffer should have its own bytes that match the
  // source byte-for-byte.
  const auto* dst = static_cast<const float*>(sb.contents());
  for (std::size_t i = 0; i < 32; ++i) {
    EXPECT_TRUE(dst[i] == static_cast<float>(i + 1));
  }

  // Mutating the SharedBuffer must not affect the original tb
  // (allocate-and-copy semantics).
  auto* mut = static_cast<float*>(sb.contents());
  mut[0] = 99.0f;
  EXPECT_TRUE(tb.as_f32()[0] == 1.0f);
}

TEST(metal_compute_tensor_beat_bridge,
     from_cpu_cached_view_matches_shape_strides_dtype) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TensorBeat tb = make_cpu_f32_({4, 8});
  SharedBuffer sb = from_tensor_beat(*mc, tb);
  if (sb.empty()) {
    return;
  }
  const BufferView& v = sb.view();
  EXPECT_TRUE(v.dtype == DType::F32);
  EXPECT_TRUE(v.rank == 2);
  EXPECT_TRUE(v.shape[0] == 4);
  EXPECT_TRUE(v.shape[1] == 8);
  // Row-major contiguous strides for [4, 8]: outer 8, inner 1.
  EXPECT_TRUE(v.strides[0] == 8);
  EXPECT_TRUE(v.strides[1] == 1);
  EXPECT_TRUE(v.offset == 0);
}

TEST(metal_compute_tensor_beat_bridge,
     from_shared_returns_same_mtl_buffer) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Build a Shared TensorBeat via the to_tensor_beat path.
  SharedBuffer source = mc->make_shared_buffer(16 * sizeof(float));
  if (source.empty()) {
    return;
  }
  BufferView src_view{};
  src_view.dtype      = DType::F32;
  src_view.rank       = 1;
  src_view.shape[0]   = 16;
  src_view.strides[0] = 1;
  source.set_view(src_view);
  auto* original_buf = source.mtl_buffer();

  TensorBeat tb = to_tensor_beat(std::move(source));
  EXPECT_TRUE(tb.storage_class() == TensorStorageClass::Shared);
  EXPECT_TRUE(tb.mtl_buffer() == original_buf);

  // Now bridge it BACK and verify zero-copy: the round-tripped
  // SharedBuffer must point at the same MTL::Buffer.
  SharedBuffer round_trip = from_tensor_beat(*mc, tb);
  EXPECT_FALSE(round_trip.empty());
  EXPECT_TRUE(round_trip.mtl_buffer() == original_buf);
  EXPECT_TRUE(round_trip.contents() == tb.external->contents);
}

TEST(metal_compute_tensor_beat_bridge,
     to_tensor_beat_transfers_buffer_and_clears_source) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer sb = mc->make_shared_buffer(64 * sizeof(float));
  if (sb.empty()) {
    return;
  }
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 1;
  v.shape[0]   = 64;
  v.strides[0] = 1;
  v.offset     = 0;
  sb.set_view(v);

  auto* contents_before = sb.contents();
  auto* buf_before      = sb.mtl_buffer();
  auto  byte_size_before = sb.byte_size();

  TensorBeat tb = to_tensor_beat(std::move(sb));

  // Source is cleared by the move.
  EXPECT_TRUE(sb.empty());

  // TensorBeat receives the buffer with intact metadata.
  EXPECT_TRUE(tb.storage_class() == TensorStorageClass::Shared);
  EXPECT_TRUE(tb.mtl_buffer() == buf_before);
  EXPECT_TRUE(tb.byte_size() == byte_size_before);
  EXPECT_TRUE(tb.dtype == TensorBeat::DType::F32);
  EXPECT_TRUE(tb.shape.size() == 1);
  EXPECT_TRUE(tb.shape[0] == 64);
  EXPECT_TRUE(tb.strides.size() == 1);
  EXPECT_TRUE(tb.strides[0] == 1);
  EXPECT_TRUE(tb.storage_offset == 0);

  // contents pointer is the same (UMA-shared).
  EXPECT_TRUE(tb.external->contents
              == static_cast<std::uint8_t*>(contents_before));
}

TEST(metal_compute_tensor_beat_bridge,
     to_tensor_beat_preserves_strided_view) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer sb = mc->make_shared_buffer(64 * sizeof(float));
  if (sb.empty()) {
    return;
  }
  // Transposed [4, 8] view: shape stays, strides reorder.
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 2;
  v.shape[0]   = 8;
  v.shape[1]   = 4;
  v.strides[0] = 1;     // transposed: was inner, now outer stride 1
  v.strides[1] = 8;
  v.offset     = 0;
  sb.set_view(v);

  TensorBeat tb = to_tensor_beat(std::move(sb));
  EXPECT_TRUE(tb.shape.size() == 2);
  EXPECT_TRUE(tb.shape[0] == 8);
  EXPECT_TRUE(tb.shape[1] == 4);
  EXPECT_TRUE(tb.strides.size() == 2);
  EXPECT_TRUE(tb.strides[0] == 1);
  EXPECT_TRUE(tb.strides[1] == 8);
}

TEST(metal_compute_tensor_beat_bridge,
     to_tensor_beat_preserves_storage_offset) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer sb = mc->make_shared_buffer(128 * sizeof(float));
  if (sb.empty()) {
    return;
  }
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 1;
  v.shape[0]   = 64;
  v.strides[0] = 1;
  v.offset     = 32;   // 32-element slice starting at index 32
  sb.set_view(v);

  TensorBeat tb = to_tensor_beat(std::move(sb));
  EXPECT_TRUE(tb.storage_offset == 32);
  EXPECT_TRUE(tb.shape[0] == 64);
}

TEST(metal_compute_tensor_beat_bridge, round_trip_cpu_cached_bytes) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TensorBeat src = make_cpu_f32_({3, 5});  // 15 elements

  SharedBuffer sb = from_tensor_beat(*mc, src);
  if (sb.empty()) {
    return;
  }
  TensorBeat dst = to_tensor_beat(std::move(sb));

  EXPECT_TRUE(dst.dtype == TensorBeat::DType::F32);
  EXPECT_TRUE(dst.shape.size() == 2);
  EXPECT_TRUE(dst.shape[0] == 3);
  EXPECT_TRUE(dst.shape[1] == 5);
  EXPECT_TRUE(dst.byte_size() == src.byte_size());

  const auto* sp = src.as_f32();
  const auto* dp = dst.as_f32();
  for (std::size_t i = 0; i < 15; ++i) {
    EXPECT_TRUE(dp[i] == sp[i]);
  }
}

TEST(metal_compute_tensor_beat_bridge,
     round_trip_shared_zero_copy_pointer_identity) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer original = mc->make_shared_buffer(8 * sizeof(float));
  if (original.empty()) {
    return;
  }
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 1;
  v.shape[0]   = 8;
  v.strides[0] = 1;
  original.set_view(v);

  auto* original_buf = original.mtl_buffer();

  // sb -> tb -> sb' -> tb'. All three should keep the same MTL::Buffer.
  TensorBeat   t1 = to_tensor_beat(std::move(original));
  SharedBuffer s2 = from_tensor_beat(*mc, t1);
  EXPECT_TRUE(s2.mtl_buffer() == original_buf);
  TensorBeat   t2 = to_tensor_beat(std::move(s2));
  EXPECT_TRUE(t2.mtl_buffer() == original_buf);
}

TEST(metal_compute_tensor_beat_bridge, to_from_empty_safe) {
  // Empty SharedBuffer -> empty (default) TensorBeat round-trip.
  SharedBuffer empty;
  TensorBeat tb = to_tensor_beat(std::move(empty));
  EXPECT_TRUE(tb.byte_size() == 0u);
  EXPECT_TRUE(tb.shape.empty());
  EXPECT_TRUE(tb.storage_class() == TensorStorageClass::CpuCached);
}
