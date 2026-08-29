#include "apple-silicon/metal-compute/command-stream.h"

#include "apple-silicon/metal-compute/event.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdlib>
#include <atomic>
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

namespace {
// See CommandStream::fire_and_forget_stats.
std::atomic<unsigned long long> g_ff_completed{0};
std::atomic<unsigned long long> g_ff_errored{0};
}  // namespace

void
CommandStream::fire_and_forget_stats(unsigned long long* completed,
                                     unsigned long long* errored)
{
  if (completed != nullptr) {
    *completed = g_ff_completed.load(std::memory_order_relaxed);
  }
  if (errored != nullptr) {
    *errored = g_ff_errored.load(std::memory_order_relaxed);
  }
}

void
CommandStream::split_encoder_(ComputeEncoder& enc)
{
  const DispatchType dt = enc._dt;
  enc.end();                       // endEncoding + release enc._enc
  if (_cb != nullptr) {
    // NOBODY WAITS ON THIS BUFFER, so nothing else is in a position to
    // notice it fail. Latch the failure from a completion handler and
    // let the Fence the caller does wait on report it -- otherwise an
    // intermediate out-of-memory skips its dispatches and the stream
    // still reports success, which is a silently wrong result rather
    // than a failed one.
    if (!_fail) { _fail = std::make_shared<Failure>(); }
    auto f = _fail;
    MTL::HandlerFunction note = [f](MTL::CommandBuffer* cb) {
      // Counted FIRST and unconditionally: this is what proves the
      // handler runs at all, which is what makes a zero error count mean
      // "nothing failed" rather than "nothing was watching".
      g_ff_completed.fetch_add(1, std::memory_order_relaxed);
      if (cb == nullptr ||
          cb->status() != MTL::CommandBufferStatusError) {
        return;
      }
      g_ff_errored.fetch_add(1, std::memory_order_relaxed);
      // FIRST failure wins: it is the one that explains the rest, and a
      // later buffer failing because an earlier one did would otherwise
      // overwrite the cause with a symptom.
      bool expected = false;
      if (!f->failed.compare_exchange_strong(expected, true)) { return; }
      std::string why = "GPU command buffer error";
      if (NS::Error* e = cb->error()) {
        if (NS::String* d = e->localizedDescription()) {
          const char* c = d->utf8String();
          if (c != nullptr) { why = c; }
        }
      }
      std::lock_guard<std::mutex> lk(f->mu);
      f->reason = "an earlier command buffer in this stream failed: " + why;
    };
    _cb->addCompletedHandler(note);
    _cb->commit();                 // fire-and-forget; the latch reports
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
  // next ensure_cb_() opens a fresh buffer. The latch goes with it: this
  // Fence is what the caller waits on, so it is where a failure in any
  // buffer this stream already let go has to surface.
  MTL::CommandBuffer* cb = _cb;
  _cb = nullptr;
  return Fence{cb, _fail};
}

// ---- CommandStream::Fence ---------------------------------------

CommandStream::Fence::Fence(MTL::CommandBuffer* cb) noexcept
  : _cb(cb)
{
}

CommandStream::Fence::Fence(MTL::CommandBuffer* cb,
                            std::shared_ptr<Failure> f) noexcept
  : _earlier(std::move(f)), _cb(cb)
{
}

CommandStream::Fence::Fence(Fence&& o) noexcept
  : _earlier(std::move(o._earlier)), _cb(std::exchange(o._cb, nullptr))
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
  _earlier = std::move(o._earlier);
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
  // This buffer's own status FIRST: when both failed, the one the caller
  // was waiting on is the more specific answer.
  if (_cb->status() == MTL::CommandBufferStatusError) {
    if (reason != nullptr) { *reason = error_message(); }
    return false;
  }
  // Then any buffer this stream committed and let go before it. Ordering
  // is safe: a stream's buffers complete in submission order, so by the
  // time this one has completed every earlier one has run its handler.
  if (_earlier && _earlier->failed.load()) {
    if (reason != nullptr) {
      std::lock_guard<std::mutex> lk(_earlier->mu);
      *reason = _earlier->reason;
    }
    return false;
  }
  return true;
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
