#include "apple-silicon/metal-compute/compute-encoder.h"

#include "apple-silicon/metal-compute/command-stream.h"
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

ComputeEncoder::ComputeEncoder(MTL::ComputeCommandEncoder* enc,
                               MTL::CommandBuffer* cb) noexcept
  : _enc(enc), _cb(cb)
{
}

ComputeEncoder::ComputeEncoder(ComputeEncoder&& o) noexcept
  : _enc(std::exchange(o._enc, nullptr)),
    _cb(std::exchange(o._cb, nullptr)),
    _n_dispatch(std::exchange(o._n_dispatch, 0)),
    _stream(std::exchange(o._stream, nullptr)),
    _split_every(std::exchange(o._split_every, 0)),
    _since_split(std::exchange(o._since_split, 0)),
    _scope_depth(std::exchange(o._scope_depth, 0)),
    _state_dirty(std::exchange(o._state_dirty, false)),
    _pending(std::move(o._pending)),
    _pending_bytes(std::move(o._pending_bytes)),
    _pending_pso(std::exchange(o._pending_pso, nullptr)),
    _dt(o._dt)
{
}

ComputeEncoder&
ComputeEncoder::operator=(ComputeEncoder&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  end();
  _enc           = std::exchange(o._enc, nullptr);
  _cb            = std::exchange(o._cb, nullptr);
  _n_dispatch    = std::exchange(o._n_dispatch, 0);
  _stream        = std::exchange(o._stream, nullptr);
  _split_every   = std::exchange(o._split_every, 0);
  _since_split   = std::exchange(o._since_split, 0);
  _scope_depth   = std::exchange(o._scope_depth, 0);
  _state_dirty   = std::exchange(o._state_dirty, false);
  _pending       = std::move(o._pending);
  _pending_bytes = std::move(o._pending_bytes);
  _pending_pso   = std::exchange(o._pending_pso, nullptr);
  _dt            = o._dt;
  return *this;
}

void
ComputeEncoder::reencode_(DispatchType dt)
{
  if (_enc == nullptr || _cb == nullptr) {
    return;
  }
  _enc->endEncoding();
  _enc->release();
  _enc = nullptr;
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  const MTL::DispatchType mtl_dt = dt == DispatchType::Concurrent
                                       ? MTL::DispatchTypeConcurrent
                                       : MTL::DispatchTypeSerial;
  MTL::ComputeCommandEncoder* e = _cb->computeCommandEncoder(mtl_dt);
  if (e != nullptr) {
    e->retain();
  }
  _enc = e;
  pool->release();
  // The retired encoder took the pending op's bindings with it. A scope
  // boundary is not deferrable (see the header), so re-bind them here.
  replay_pending_();
}

void
ComputeEncoder::replay_pending_()
{
  if (_enc == nullptr) {
    return;
  }
  if (_pending_pso != nullptr) {
    _enc->setComputePipelineState(_pending_pso);
  }
  for (const Pending& p : _pending) {
    switch (p.kind) {
      case PendKind::Buffer:
        _enc->setBuffer(p.buf,
                        static_cast<NS::UInteger>(p.off),
                        static_cast<NS::UInteger>(p.index));
        break;
      case PendKind::Bytes:
        _enc->setBytes(_pending_bytes.data() + p.bytes_off,
                       static_cast<NS::UInteger>(p.bytes_len),
                       static_cast<NS::UInteger>(p.index));
        break;
      case PendKind::Texture:
        _enc->setTexture(p.tex, static_cast<NS::UInteger>(p.index));
        break;
      case PendKind::TgMem:
        _enc->setThreadgroupMemoryLength(
            static_cast<NS::UInteger>(p.off),
            static_cast<NS::UInteger>(p.index));
        break;
    }
  }
}

ComputeEncoder::ConcurrentScope::ConcurrentScope(ComputeEncoder* e,
                                                 bool active) noexcept
  : _e(active ? e : nullptr)
{
  if (_e != nullptr) {
    ++_e->_scope_depth;   // suppress auto-split for the duration of the group
    _e->reencode_(DispatchType::Concurrent);
  }
}

ComputeEncoder::ConcurrentScope::ConcurrentScope(ConcurrentScope&& o) noexcept
  : _e(std::exchange(o._e, nullptr))
{
}

ComputeEncoder::ConcurrentScope::~ConcurrentScope()
{
  if (_e != nullptr) {
    _e->reencode_(DispatchType::Serial);   // reopen Serial for the chain
    if (_e->_scope_depth > 0) { --_e->_scope_depth; }
  }
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
  // Auto command-buffer split: at an OP BOUNDARY (set_function starts a new op),
  // once _split_every dispatches have accumulated, commit the current buffer +
  // reopen a fresh one, so the CPU-encode of the next chunk pipelines against
  // the GPU-exec of this one. Must be here (before the pipeline/buffers are set
  // on the encoder), NOT in dispatch() -- splitting mid-op would strand the
  // just-set state on the old encoder. Deferred inside a concurrent_scope, and
  // deferred again when state is already pending (_state_dirty): a call site
  // that binds its buffers BEFORE naming its function reaches here mid-op, and
  // splitting there would strand exactly the bindings it is about to dispatch
  // with. Waiting for the next clean boundary costs a few dispatches of
  // pipelining and nothing else.
  if (_split_every > 0 && _scope_depth == 0 && _stream != nullptr &&
      _since_split >= _split_every && !_state_dirty) {
    _stream->split_encoder_(*this);
    _since_split = 0;
  }
  _enc->setComputePipelineState(fn.mtl_pso());
  _pending_pso = fn.mtl_pso();
}

void
ComputeEncoder::set_buffer(unsigned index, const SharedBuffer& buf,
                           std::size_t byte_offset)
{
  if (_enc == nullptr || buf.empty()) {
    return;
  }
  _enc->setBuffer(buf.mtl_buffer(),
                  static_cast<NS::UInteger>(byte_offset + buf.byte_offset()),
                  static_cast<NS::UInteger>(index));
  _state_dirty = true;
  Pending p;
  p.kind  = PendKind::Buffer;
  p.index = index;
  p.buf   = buf.mtl_buffer();
  p.off   = byte_offset + buf.byte_offset();
  _pending.push_back(p);
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
  _state_dirty = true;
  Pending p;
  p.kind  = PendKind::Buffer;
  p.index = index;
  p.buf   = buf;
  p.off   = byte_offset;
  _pending.push_back(p);
}

void
ComputeEncoder::set_buffer_view(unsigned buf_index,
                                const SharedBuffer& buf,
                                unsigned meta_index)
{
  if (_enc == nullptr || buf.empty()) {
    return;
  }
  // Data buffer at the subview's baked-in base offset (0 for a normal
  // buffer) -- the view's element offset is applied inside the kernel by
  // vpipe_tv_index(), not at bind time. This mirrors how MLX and other
  // strided libraries separate base pointer from logical view.
  _enc->setBuffer(buf.mtl_buffer(),
                  static_cast<NS::UInteger>(buf.byte_offset()),
                  static_cast<NS::UInteger>(buf_index));
  {
    Pending p;
    p.kind  = PendKind::Buffer;
    p.index = buf_index;
    p.buf   = buf.mtl_buffer();
    p.off   = buf.byte_offset();
    _pending.push_back(p);
  }

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
  _state_dirty = true;
  record_bytes_(meta_index, &wire, sizeof(wire));
}

void
ComputeEncoder::record_bytes_(unsigned index, const void* bytes,
                              std::size_t size)
{
  Pending p;
  p.kind      = PendKind::Bytes;
  p.index     = index;
  p.bytes_off = _pending_bytes.size();
  p.bytes_len = size;
  const auto* src = static_cast<const std::uint8_t*>(bytes);
  _pending_bytes.insert(_pending_bytes.end(), src, src + size);
  _pending.push_back(p);
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
  _state_dirty = true;
  record_bytes_(index, bytes, size);
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
  _state_dirty = true;
  Pending p;
  p.kind  = PendKind::TgMem;
  p.index = index;
  p.off   = byte_size;
  _pending.push_back(p);
}

void
ComputeEncoder::set_texture(unsigned index, const Texture& tex)
{
  if (_enc == nullptr || !tex.valid()) {
    return;
  }
  _enc->setTexture(tex.mtl_texture(),
                   static_cast<NS::UInteger>(index));
  _state_dirty = true;
  Pending p;
  p.kind  = PendKind::Texture;
  p.index = index;
  p.tex   = tex.mtl_texture();
  _pending.push_back(p);
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
ComputeEncoder::memory_barrier_buffer(const SharedBuffer& buf)
{
  if (_enc == nullptr || buf.empty()) {
    return;
  }
  const MTL::Resource* res = buf.mtl_buffer();
  _enc->memoryBarrier(&res, 1);
}

void
ComputeEncoder::dispatch(LaunchDims threads_per_grid,
                         LaunchDims threads_per_threadgroup)
{
  if (_enc == nullptr) {
    return;
  }
  ++_n_dispatch;
  ++_since_split;
  // Op complete -> the encoder is at a clean boundary and nothing is pending.
  // clear() keeps both vectors' capacity, so steady-state encoding does not
  // allocate.
  _state_dirty = false;
  _pending.clear();
  _pending_bytes.clear();
  _pending_pso = nullptr;
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
