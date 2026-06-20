#include "apple-silicon/metal-compute/compute-encoder.h"

#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/fence.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/texture.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdint>
#include <utility>

namespace vpipe::metal_compute {

namespace {

// Wire-format mirror of MSL's `struct vpipe_tensor_view` (see
// msl/vpipe_tensor_view.metal). Field order, types, and array
// widths must stay byte-identical to the MSL side -- any change
// here requires a matching change there.
//
// Shape / strides / offset are stored as int32 even though
// BufferView uses int64; tensors that fit on one Apple Silicon
// GPU fit in int32 element offsets (>= 2 GiB elements would be
// a >= 8 GiB float32 buffer, which already exceeds practical
// allocation limits).
struct BufferViewWire {
  std::uint32_t rank;
  std::uint32_t dtype;
  std::int32_t  shape[BufferView::kMaxRank];
  std::int32_t  strides[BufferView::kMaxRank];
  std::int32_t  offset;
};

static_assert(sizeof(BufferViewWire) == 4 + 4 + 4 * 8 + 4 * 8 + 4,
              "BufferViewWire byte size must match the MSL struct");

}  // namespace

ComputeEncoder::ComputeEncoder(MTL::ComputeCommandEncoder* enc) noexcept
  : _enc(enc)
{
}

ComputeEncoder::ComputeEncoder(ComputeEncoder&& o) noexcept
  : _enc(std::exchange(o._enc, nullptr))
{
}

ComputeEncoder&
ComputeEncoder::operator=(ComputeEncoder&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  end();
  _enc = std::exchange(o._enc, nullptr);
  return *this;
}

ComputeEncoder::~ComputeEncoder()
{
  end();
}

void
ComputeEncoder::end()
{
  if (_enc != nullptr) {
    _enc->endEncoding();
    _enc->release();
    _enc = nullptr;
  }
}

void
ComputeEncoder::set_function(const ComputeFunction& fn)
{
  if (_enc == nullptr || !fn.valid()) {
    return;
  }
  _enc->setComputePipelineState(fn.mtl_pso());
}

void
ComputeEncoder::set_buffer(unsigned index, const SharedBuffer& buf,
                           std::size_t byte_offset)
{
  if (_enc == nullptr || buf.empty()) {
    return;
  }
  _enc->setBuffer(buf.mtl_buffer(),
                  static_cast<NS::UInteger>(byte_offset),
                  static_cast<NS::UInteger>(index));
}

void
ComputeEncoder::set_mtl_buffer(unsigned index, MTL::Buffer* buf,
                               std::size_t byte_offset)
{
  if (_enc == nullptr || buf == nullptr) {
    return;
  }
  _enc->setBuffer(buf,
                  static_cast<NS::UInteger>(byte_offset),
                  static_cast<NS::UInteger>(index));
}

void
ComputeEncoder::set_buffer_view(unsigned buf_index,
                                const SharedBuffer& buf,
                                unsigned meta_index)
{
  if (_enc == nullptr || buf.empty()) {
    return;
  }
  // Data buffer at byte offset 0 -- the view's element offset is
  // applied inside the kernel by vpipe_tv_index(), not at bind
  // time. This mirrors how MLX and other strided libraries
  // separate base pointer from logical view.
  _enc->setBuffer(buf.mtl_buffer(), 0,
                  static_cast<NS::UInteger>(buf_index));

  const BufferView& v = buf.view();
  BufferViewWire wire{};
  wire.rank   = static_cast<std::uint32_t>(v.rank);
  wire.dtype  = static_cast<std::uint32_t>(v.dtype);
  wire.offset = static_cast<std::int32_t>(v.offset);
  for (int i = 0; i < BufferView::kMaxRank; ++i) {
    wire.shape[i]   = static_cast<std::int32_t>(v.shape[i]);
    wire.strides[i] = static_cast<std::int32_t>(v.strides[i]);
  }
  _enc->setBytes(&wire, sizeof(wire),
                 static_cast<NS::UInteger>(meta_index));
}

void
ComputeEncoder::set_constant_bytes(unsigned index,
                                   const void* bytes,
                                   std::size_t size)
{
  if (_enc == nullptr || bytes == nullptr || size == 0) {
    return;
  }
  _enc->setBytes(bytes,
                 static_cast<NS::UInteger>(size),
                 static_cast<NS::UInteger>(index));
}

void
ComputeEncoder::set_threadgroup_memory_length(unsigned index,
                                              std::size_t byte_size)
{
  if (_enc == nullptr) {
    return;
  }
  _enc->setThreadgroupMemoryLength(
      static_cast<NS::UInteger>(byte_size),
      static_cast<NS::UInteger>(index));
}

void
ComputeEncoder::set_texture(unsigned index, const Texture& tex)
{
  if (_enc == nullptr || !tex.valid()) {
    return;
  }
  _enc->setTexture(tex.mtl_texture(),
                   static_cast<NS::UInteger>(index));
}

void
ComputeEncoder::update_fence(const Fence& fence)
{
  if (_enc == nullptr || !fence.valid()) {
    return;
  }
  _enc->updateFence(fence.mtl_fence());
}

void
ComputeEncoder::wait_for_fence(const Fence& fence)
{
  if (_enc == nullptr || !fence.valid()) {
    return;
  }
  _enc->waitForFence(fence.mtl_fence());
}

void
ComputeEncoder::memory_barrier(BarrierScope scope)
{
  if (_enc == nullptr) {
    return;
  }
  MTL::BarrierScope bits = static_cast<MTL::BarrierScope>(0);
  if (static_cast<std::uint8_t>(scope) & 1) {
    bits = static_cast<MTL::BarrierScope>(
        bits | MTL::BarrierScopeBuffers);
  }
  if (static_cast<std::uint8_t>(scope) & 2) {
    bits = static_cast<MTL::BarrierScope>(
        bits | MTL::BarrierScopeTextures);
  }
  _enc->memoryBarrier(bits);
}

void
ComputeEncoder::dispatch(LaunchDims threads_per_grid,
                         LaunchDims threads_per_threadgroup)
{
  if (_enc == nullptr) {
    return;
  }
  ++_n_dispatch;
  MTL::Size grid(
      static_cast<NS::UInteger>(threads_per_grid.x),
      static_cast<NS::UInteger>(threads_per_grid.y),
      static_cast<NS::UInteger>(threads_per_grid.z));
  MTL::Size tg(
      static_cast<NS::UInteger>(threads_per_threadgroup.x),
      static_cast<NS::UInteger>(threads_per_threadgroup.y),
      static_cast<NS::UInteger>(threads_per_threadgroup.z));
  _enc->dispatchThreads(grid, tg);
}

}  // namespace vpipe::metal_compute
