#include "apple-silicon/metal-compute/fence.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <utility>

namespace vpipe::metal_compute {

Fence::Fence(MTL::Fence* f) noexcept : _fence(f) {}

Fence::Fence(Fence&& o) noexcept
  : _fence(std::exchange(o._fence, nullptr))
{
}

Fence&
Fence::operator=(Fence&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_fence != nullptr) {
    _fence->release();
  }
  _fence = std::exchange(o._fence, nullptr);
  return *this;
}

Fence::~Fence()
{
  if (_fence != nullptr) {
    _fence->release();
    _fence = nullptr;
  }
}

}  // namespace vpipe::metal_compute
