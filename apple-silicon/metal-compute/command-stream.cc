#include "apple-silicon/metal-compute/command-stream.h"

#include "apple-silicon/metal-compute/event.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdlib>
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
  ComputeEncoder e{enc, cb};
  // Arm the auto command-buffer split. Default 50 commands/buffer (matches the
  // measured decode sweet spot + MLX's commit cadence); VPIPE_MC_CMDBUF_SPLIT=N
  // overrides, 0 disables. Only long streams (>N dispatches) ever trip it;
  // short one-off encodes never split.
  static const int kSplit = []() {
    const char* s = std::getenv("VPIPE_MC_CMDBUF_SPLIT");
    return s ? std::atoi(s) : 50;
  }();
  e._stream = this;
  e._dt = dispatch_type;
  e._split_every = kSplit;
  e._since_split = 0;
  return e;
}

void
CommandStream::split_encoder_(ComputeEncoder& enc)
{
  const DispatchType dt = enc._dt;
  enc.end();                       // endEncoding + release enc._enc
  if (_cb != nullptr) {
    _cb->commit();                 // fire-and-forget; final commit() waits
    _cb->release();
    _cb = nullptr;
  }
  ComputeEncoder fresh = begin_compute(dt);   // opens a new cb + configures it
  enc._enc = fresh._enc;
  enc._cb  = fresh._cb;
  fresh._enc = nullptr;            // keep fresh's dtor from ending our encoder
  fresh._cb  = nullptr;
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

bool
CommandStream::Fence::errored() const noexcept
{
  return _cb != nullptr
      && _cb->status() == MTL::CommandBufferStatusError;
}

long
CommandStream::Fence::error_code() const noexcept
{
  if (_cb == nullptr) {
    return 0;
  }
  NS::Error* e = _cb->error();
  return e != nullptr ? static_cast<long>(e->code()) : 0;
}

bool
CommandStream::Fence::out_of_memory() const noexcept
{
  // A PageFault is the GPU touching non-resident memory -- the exact
  // symptom of over-committing UMA -- so treat it alongside OutOfMemory.
  const long c = error_code();
  return c == static_cast<long>(MTL::CommandBufferErrorOutOfMemory)
      || c == static_cast<long>(MTL::CommandBufferErrorPageFault);
}

std::string
CommandStream::Fence::error_message() const noexcept
{
  if (_cb == nullptr || _cb->status() != MTL::CommandBufferStatusError) {
    return {};
  }
  NS::Error* e = _cb->error();
  if (e == nullptr) {
    return "GPU command buffer error (no NS::Error attached)";
  }
  const long code = static_cast<long>(e->code());
  const char* kind =
      code == static_cast<long>(MTL::CommandBufferErrorOutOfMemory) ? "out of memory"
    : code == static_cast<long>(MTL::CommandBufferErrorPageFault)   ? "page fault (non-resident memory)"
    : code == static_cast<long>(MTL::CommandBufferErrorTimeout)     ? "timeout"
    : "error";
  std::string msg = "GPU command buffer ";
  msg += kind;
  msg += " (code ";
  msg += std::to_string(code);
  NS::String* desc = e->localizedDescription();
  if (desc != nullptr && desc->utf8String() != nullptr) {
    msg += ": ";
    msg += desc->utf8String();
  }
  msg += ")";
  return msg;
}

bool
CommandStream::Fence::wait_ok(std::string* reason)
{
  if (_cb == nullptr) {
    return true;                 // nothing ran, nothing failed
  }
  _cb->waitUntilCompleted();
  if (_cb->status() != MTL::CommandBufferStatusError) {
    return true;
  }
  if (reason != nullptr) {
    *reason = error_message();
  }
  return false;
}

CommandStream::Fence::GpuTimes
CommandStream::Fence::gpu_times() const noexcept
{
  if (_cb == nullptr) {
    return {0.0, 0.0};
  }
  return {_cb->GPUEndTime() - _cb->GPUStartTime(),
          _cb->kernelEndTime() - _cb->kernelStartTime()};
}

}  // namespace vpipe::metal_compute
