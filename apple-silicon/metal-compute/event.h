#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_EVENT_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_EVENT_H

#include <chrono>
#include <cstdint>

namespace MTL { class SharedEvent; }

namespace vpipe::metal_compute {

class MetalCompute;

// Cross-stream / CPU<->GPU synchronization primitive. Wraps a
// MTL::SharedEvent: an event holds a monotonically-rising 64-bit
// counter; producers raise it (either GPU-side via
// CommandStream::encode_signal or CPU-side via set_signaled_value),
// and waiters block until the value crosses a threshold.
//
// Move-only. Default-constructed Events are empty and safe to
// destroy. Allocate via MetalCompute::make_event().
class Event {
public:
  Event() noexcept = default;
  Event(Event&&) noexcept;
  Event& operator=(Event&&) noexcept;
  Event(const Event&)            = delete;
  Event& operator=(const Event&) = delete;
  ~Event();

  bool valid() const noexcept { return _ev != nullptr; }

  // Read the event's current counter value (last value any signaler
  // has written). Returns 0 on an empty Event.
  std::uint64_t signaled_value() const noexcept;

  // CPU-side signaler. Sets the counter directly; equivalent to a
  // GPU-side encodeSignalEvent + waitUntilCompleted but synchronous.
  void set_signaled_value(std::uint64_t value) noexcept;

  // Block the calling thread until the counter reaches >= `value`
  // or until `timeout` elapses. Returns true on success, false on
  // timeout or invalid event.
  bool wait(std::uint64_t value,
            std::chrono::nanoseconds timeout) noexcept;

  MTL::SharedEvent* mtl_event() const noexcept { return _ev; }

private:
  friend class MetalCompute;
  explicit Event(MTL::SharedEvent* ev) noexcept;

  MTL::SharedEvent* _ev = nullptr;
};

}  // namespace vpipe::metal_compute

#endif
