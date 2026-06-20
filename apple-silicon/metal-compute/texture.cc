#include "apple-silicon/metal-compute/texture.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <CoreFoundation/CoreFoundation.h>

#include <utility>

namespace vpipe::metal_compute {

Texture::Texture(MTL::Texture* tex, PixelFormat format,
                 void* cv_handle) noexcept
  : _tex(tex),
    _format(format),
    _cv_handle(cv_handle)
{
}

Texture::Texture(Texture&& o) noexcept
  : _tex(std::exchange(o._tex, nullptr)),
    _format(o._format),
    _cv_handle(std::exchange(o._cv_handle, nullptr))
{
  o._format = PixelFormat::Unknown;
}

Texture&
Texture::operator=(Texture&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  if (_cv_handle != nullptr) {
    ::CFRelease(_cv_handle);
  }
  if (_tex != nullptr) {
    _tex->release();
  }
  _tex       = std::exchange(o._tex, nullptr);
  _format    = o._format;
  _cv_handle = std::exchange(o._cv_handle, nullptr);
  o._format  = PixelFormat::Unknown;
  return *this;
}

Texture::~Texture()
{
  // Release in CV-then-MTL order. CVMetalTextureRef holds the
  // MTL::Texture* alive; CFRelease decrements its retain count and
  // (if zero) releases the underlying texture. For natively-
  // allocated textures _cv_handle is nullptr and we just release
  // the MTL::Texture* directly.
  if (_cv_handle != nullptr) {
    ::CFRelease(_cv_handle);
    _cv_handle = nullptr;
    // For CV-backed textures, the MTL::Texture* refcount is owned
    // by the CVMetalTextureRef; do not double-release here.
    _tex = nullptr;
  }
  if (_tex != nullptr) {
    _tex->release();
    _tex = nullptr;
  }
}

std::uint32_t
Texture::width() const noexcept
{
  if (_tex == nullptr) {
    return 0;
  }
  return static_cast<std::uint32_t>(_tex->width());
}

std::uint32_t
Texture::height() const noexcept
{
  if (_tex == nullptr) {
    return 0;
  }
  return static_cast<std::uint32_t>(_tex->height());
}

void
Texture::replace_region(const void* src, std::size_t bytes_per_row)
{
  if (_tex == nullptr || src == nullptr || bytes_per_row == 0) {
    return;
  }
  // CV-backed textures map back to a CVPixelBuffer's IOSurface; the
  // replaceRegion path bypasses that and would write garbage into
  // an aliased surface. Force callers through the CVPixelBuffer
  // they originally bridged in.
  if (_cv_handle != nullptr) {
    return;
  }
  if (_tex->storageMode() != MTL::StorageModeShared
      && _tex->storageMode() != MTL::StorageModeManaged) {
    // Private-storage textures need a blit encoder; not yet wrapped.
    return;
  }
  MTL::Region region(0, 0, _tex->width(), _tex->height());
  _tex->replaceRegion(region, 0, src, bytes_per_row);
}

// Allocation factory lives in metal-compute.cc next to the existing
// make_shared_buffer / make_event factories; this TU just owns the
// wrapper lifecycle + helpers.

}  // namespace vpipe::metal_compute
