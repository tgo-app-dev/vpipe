#include "apple-silicon/metal-compute/command-stream.h"

#include "apple-silicon/metal-compute/event.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <utility>

namespace vpipe::metal_compute {

// ---- CommandStream ----------------------------------------------

CommandStream::CommandStream(MTL::CommandQueue* queue) noexcept
  : _queue(queue)
{
}

CommandStream::CommandStream(CommandStream&& o) noexcept
  : _queue(std::exchange(o._queue, nullptr)),
    _cb(std::exchange(o._cb, nullptr))
{
}

CommandStream&
CommandStream::operator=(CommandStream&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_cb != nullptr) {
    _cb->release();
  }
  if (_queue != nullptr) {
    _queue->release();
  }
  _queue = std::exchange(o._queue, nullptr);
  _cb    = std::exchange(o._cb, nullptr);
  return *this;
}

CommandStream::~CommandStream()
{
  // An open, uncommitted command buffer is silently released. Any
  // commands the user already encoded are dropped on the floor;
  // Metal does not require commit() on every command buffer.
  if (_cb != nullptr) {
    _cb->release();
    _cb = nullptr;
  }
  if (_queue != nullptr) {
    _queue->release();
    _queue = nullptr;
  }
}

MTL::CommandBuffer*
CommandStream::ensure_cb_()
{
  if (_queue == nullptr) {
    return nullptr;
  }
  if (_cb != nullptr) {
    return _cb;
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::CommandBuffer* cb = _queue->commandBuffer();
  if (cb != nullptr) {
    cb->retain();
    _cb = cb;
  }
  pool->release();
  return _cb;
}

ComputeEncoder
CommandStream::begin_compute(DispatchType dispatch_type)
{
  MTL::CommandBuffer* cb = ensure_cb_();
  if (cb == nullptr) {
    return ComputeEncoder{};
  }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  const MTL::DispatchType mtl_dt =
      dispatch_type == DispatchType::Concurrent
          ? MTL::DispatchTypeConcurrent
          : MTL::DispatchTypeSerial;
  MTL::ComputeCommandEncoder* enc =
      cb->computeCommandEncoder(mtl_dt);
  if (enc != nullptr) {
    enc->retain();
  }
  pool->release();
  return ComputeEncoder{enc};
}

void
CommandStream::encode_signal(const Event& ev, std::uint64_t value)
{
  if (!ev.valid()) {
    return;
  }
  MTL::CommandBuffer* cb = ensure_cb_();
  if (cb == nullptr) {
    return;
  }
  cb->encodeSignalEvent(ev.mtl_event(),
                        static_cast<NS::UInteger>(value));
}

void
CommandStream::encode_wait(const Event& ev, std::uint64_t value)
{
  if (!ev.valid()) {
    return;
  }
  MTL::CommandBuffer* cb = ensure_cb_();
  if (cb == nullptr) {
    return;
  }
  cb->encodeWait(ev.mtl_event(),
                 static_cast<NS::UInteger>(value));
}

void
CommandStream::on_completion(std::function<void()> handler)
{
  MTL::CommandBuffer* cb = ensure_cb_();
  if (cb == nullptr) {
    return;
  }
  // MTL::HandlerFunction is std::function<void(MTL::CommandBuffer*)>.
  // Adapt by ignoring the CB arg so callers can write void()
  // lambdas. addCompletedHandler must be called before commit; the
  // caller-side contract is documented in command-stream.h.
  MTL::HandlerFunction wrapped =
      [h = std::move(handler)](MTL::CommandBuffer*) { h(); };
  cb->addCompletedHandler(wrapped);
}

CommandStream::Fence
CommandStream::commit()
{
  if (_cb == nullptr) {
    return Fence{};
  }
  _cb->commit();
  // Transfer the refcount to the Fence and clear our slot so the
  // next ensure_cb_() opens a fresh buffer.
  MTL::CommandBuffer* cb = _cb;
  _cb = nullptr;
  return Fence{cb};
}

// ---- CommandStream::Fence ---------------------------------------

CommandStream::Fence::Fence(MTL::CommandBuffer* cb) noexcept
  : _cb(cb)
{
}

CommandStream::Fence::Fence(Fence&& o) noexcept
  : _cb(std::exchange(o._cb, nullptr))
{
}

CommandStream::Fence&
CommandStream::Fence::operator=(Fence&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_cb != nullptr) {
    _cb->release();
  }
  _cb = std::exchange(o._cb, nullptr);
  return *this;
}

CommandStream::Fence::~Fence()
{
  if (_cb != nullptr) {
    _cb->release();
    _cb = nullptr;
  }
}

void
CommandStream::Fence::wait()
{
  if (_cb == nullptr) {
    return;
  }
  _cb->waitUntilCompleted();
}

bool
CommandStream::Fence::completed() const noexcept
{
  if (_cb == nullptr) {
    return false;
  }
  return _cb->status() == MTL::CommandBufferStatusCompleted;
}

}  // namespace vpipe::metal_compute
