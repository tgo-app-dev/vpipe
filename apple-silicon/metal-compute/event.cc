#include "apple-silicon/metal-compute/event.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <utility>

namespace vpipe::metal_compute {

Event::Event(MTL::SharedEvent* ev) noexcept
  : _ev(ev)
{
}

Event::Event(Event&& o) noexcept
  : _ev(std::exchange(o._ev, nullptr))
{
}

Event&
Event::operator=(Event&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_ev != nullptr) {
    _ev->release();
  }
  _ev = std::exchange(o._ev, nullptr);
  return *this;
}

Event::~Event()
{
  if (_ev != nullptr) {
    _ev->release();
    _ev = nullptr;
  }
}

std::uint64_t
Event::signaled_value() const noexcept
{
  if (_ev == nullptr) {
    return 0;
  }
  return _ev->signaledValue();
}

void
Event::set_signaled_value(std::uint64_t value) noexcept
{
  if (_ev == nullptr) {
    return;
  }
  _ev->setSignaledValue(value);
}

bool
Event::wait(std::uint64_t value,
            std::chrono::nanoseconds timeout) noexcept
{
  if (_ev == nullptr) {
    return false;
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      timeout).count();
  // waitUntilSignaledValue takes milliseconds as uint64; clamp.
  const std::uint64_t timeout_ms =
      ms < 0 ? 0 : static_cast<std::uint64_t>(ms);
  return _ev->waitUntilSignaledValue(value, timeout_ms);
}

}  // namespace vpipe::metal_compute
