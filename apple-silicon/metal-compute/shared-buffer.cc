#include "apple-silicon/metal-compute/shared-buffer.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <utility>

namespace vpipe::metal_compute {

namespace {

// Round `n` up to the next multiple of `page_size`. `page_size` is
// always a power of two on Darwin/Linux, but stay safe with the
// generic ceil-div form.
inline std::size_t
round_up_to_page_(std::size_t n, std::size_t page_size) noexcept
{
  return ((n + page_size - 1) / page_size) * page_size;
}

}  // namespace

SharedBuffer::SharedBuffer(MTL::Buffer* buf, void* contents,
                           std::size_t byte_size) noexcept
  : _buf(buf),
    _contents(contents),
    _byte_size(byte_size)
{
}

SharedBuffer
SharedBuffer::wrap(MTL::Buffer* buf, void* contents,
                   std::size_t byte_size) noexcept
{
  if (buf == nullptr || byte_size == 0) {
    return SharedBuffer{};
  }
  void* c = contents != nullptr ? contents : buf->contents();
  return SharedBuffer{buf, c, byte_size};
}

SharedBuffer
SharedBuffer::subview(std::size_t offset, std::size_t size) const noexcept
{
  if (_buf == nullptr || size == 0) {
    return SharedBuffer{};
  }
  _buf->retain();                          // the subview holds its own ref
  SharedBuffer sv{_buf,
                  static_cast<void*>(
                      static_cast<std::uint8_t*>(_contents) + offset),
                  size};
  sv._base_off = _base_off + offset;       // composes with this view's offset
  return sv;
}

SharedBuffer::SharedBuffer(SharedBuffer&& o) noexcept
  : _buf(std::exchange(o._buf, nullptr)),
    _contents(std::exchange(o._contents, nullptr)),
    _byte_size(std::exchange(o._byte_size, 0)),
    _base_off(std::exchange(o._base_off, 0)),
    _view(o._view),
    _wired(std::exchange(o._wired, false))
{
  o._view = {};
}

SharedBuffer&
SharedBuffer::operator=(SharedBuffer&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  teardown_();
  _buf       = std::exchange(o._buf, nullptr);
  _contents  = std::exchange(o._contents, nullptr);
  _byte_size = std::exchange(o._byte_size, 0);
  _base_off  = std::exchange(o._base_off, 0);
  _view      = o._view;
  _wired     = std::exchange(o._wired, false);
  o._view    = {};
  return *this;
}

SharedBuffer::~SharedBuffer()
{
  teardown_();
}

void
SharedBuffer::teardown_() noexcept
{
  if (_wired && _contents != nullptr && _byte_size != 0) {
    // mlock() rounded the address down and the length up to page
    // boundaries; munlock() must call with the same bounds. The
    // contents pointer of an MTL Shared buffer is already page-
    // aligned on Darwin, so just round the length up.
    const std::size_t page_size =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t unlock_size =
        round_up_to_page_(_byte_size, page_size);
    ::munlock(_contents, unlock_size);
    _wired = false;
  }
  if (_buf != nullptr) {
    _buf->release();
    _buf = nullptr;
  }
  _contents  = nullptr;
  _byte_size = 0;
  _base_off  = 0;
}

bool
SharedBuffer::set_wired(bool on) noexcept
{
  if (_buf == nullptr || _contents == nullptr || _byte_size == 0) {
    // Nothing to wire; treat as a successful no-op so callers don't
    // have to special-case empty buffers.
    return true;
  }
  if (on == _wired) {
    return true;
  }
  const std::size_t page_size =
      static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  const std::size_t lock_size =
      round_up_to_page_(_byte_size, page_size);

  if (on) {
    if (::mlock(_contents, lock_size) != 0) {
      // errno preserved by mlock(); caller can inspect.
      return false;
    }
    _buf->setPurgeableState(MTL::PurgeableStateNonVolatile);
    _wired = true;
  } else {
    // Best-effort unlock. A munlock failure is non-fatal -- the OS
    // may unwire automatically when the address is released. We
    // flip the flag regardless so a subsequent set_wired(true)
    // re-attempts the lock.
    ::munlock(_contents, lock_size);
    _buf->setPurgeableState(MTL::PurgeableStateVolatile);
    _wired = false;
  }
  return true;
}

}  // namespace vpipe::metal_compute
