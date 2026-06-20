#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_COMMAND_STREAM_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_COMMAND_STREAM_H

#include "apple-silicon/metal-compute/compute-encoder.h"

#include <cstdint>
#include <functional>

namespace MTL {
class CommandBuffer;
class CommandQueue;
}

namespace vpipe::metal_compute {

class Event;
class MetalCompute;

// One independently-ordered stream of GPU work, analogous to a CUDA
// stream. Wraps an MTL::CommandQueue. Commands accumulate into a
// command buffer that begin_compute() / encode_signal() / encode_wait()
// open on demand; commit() submits it and returns a Fence the caller
// can wait on.
//
// Move-only. Default-constructed instances are empty no-ops.
class CommandStream {
public:
  CommandStream() noexcept = default;
  CommandStream(CommandStream&&) noexcept;
  CommandStream& operator=(CommandStream&&) noexcept;
  CommandStream(const CommandStream&)            = delete;
  CommandStream& operator=(const CommandStream&) = delete;
  ~CommandStream();

  bool valid() const noexcept { return _queue != nullptr; }

  // Open the current command buffer (auto-creates one if none is
  // in flight) and return an RAII encoder. The encoder must be
  // ended (drop it, or call end()) before the next encoder is
  // started on the same stream or before commit() submits the
  // buffer.
  //
  // `dispatch_type` controls whether Metal inserts implicit
  // barriers between successive dispatches on the returned
  // encoder. Concurrent + Untracked-hazard buffers + explicit
  // memory_barrier() is the explicit-control path; Serial (the
  // default) matches Tracked-hazard buffers.
  ComputeEncoder
  begin_compute(DispatchType dispatch_type = DispatchType::Serial);

  // Append a GPU-side signal/wait of `value` on `ev` to the
  // current command buffer (auto-opened if needed). The op is
  // encoded immediately but executes only when commit() submits
  // the buffer. No-op if the stream or event is empty.
  void encode_signal(const Event& ev, std::uint64_t value);
  void encode_wait  (const Event& ev, std::uint64_t value);

  // Attach a completion callback to the currently-open command
  // buffer (auto-opens one if none is open). MUST be called
  // before commit(); Metal asserts if a handler is added to an
  // already-committed buffer. Multiple calls chain in
  // registration order. The callback runs on a Metal-internal
  // thread, not the caller's, so synchronize access to anything
  // it captures.
  void on_completion(std::function<void()> handler);

  // RAII handle on a committed MTL::CommandBuffer. Lets the caller
  // poll status, wait, or attach a completion callback.
  class Fence {
  public:
    Fence() noexcept = default;
    Fence(Fence&&) noexcept;
    Fence& operator=(Fence&&) noexcept;
    Fence(const Fence&)            = delete;
    Fence& operator=(const Fence&) = delete;
    ~Fence();

    bool valid() const noexcept { return _cb != nullptr; }

    // Block until the command buffer has completed. No-op on an
    // empty Fence.
    void wait();

    // Non-blocking poll. Returns true iff status() == Completed.
    bool completed() const noexcept;

    MTL::CommandBuffer* mtl_command_buffer() const noexcept
    {
      return _cb;
    }

  private:
    friend class CommandStream;
    explicit Fence(MTL::CommandBuffer* cb) noexcept;

    MTL::CommandBuffer* _cb = nullptr;
  };

  // Submit the current command buffer (if any) and return a Fence
  // wrapping it. After commit() the next begin_compute() /
  // encode_signal() / encode_wait() opens a fresh buffer. Returns
  // an empty Fence if no buffer was open.
  Fence commit();

  MTL::CommandQueue* mtl_queue() const noexcept { return _queue; }

private:
  friend class MetalCompute;
  explicit CommandStream(MTL::CommandQueue* queue) noexcept;

  // Open the lazy command buffer if none is in flight, returning
  // the current cb (or nullptr if the stream is invalid).
  MTL::CommandBuffer* ensure_cb_();

  MTL::CommandQueue*  _queue = nullptr;
  MTL::CommandBuffer* _cb    = nullptr;
};

}  // namespace vpipe::metal_compute

#endif
