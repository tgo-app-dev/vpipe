#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_TEXTURE_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_TEXTURE_H

#include <cstddef>
#include <cstdint>

namespace MTL { class Texture; }

namespace vpipe::metal_compute {

class MetalCompute;

// Pixel formats the framework understands. Maps 1:1 to a curated
// subset of MTL::PixelFormat values; extend the enum + the
// to_mtl_pixel_format() helper in texture.cc as new ones become
// load-bearing.
enum class PixelFormat : std::uint8_t {
  Unknown      = 0,
  R8Unorm      = 1,
  RGBA8Unorm   = 2,
  BGRA8Unorm   = 3,
  R32Float     = 4,
  RGBA32Float  = 5,
  R16Float     = 6,
  RGBA16Float  = 7,
};

// What a kernel does with a texture. Affects allocation flags;
// declared at make_texture time and cannot be widened after.
enum class TextureUsage : std::uint8_t {
  ShaderRead      = 1,
  ShaderWrite     = 2,
  ShaderReadWrite = 3,
};

// Where the bytes live. Shared (UMA, CPU-visible) is the default;
// Private (GPU-only) needs a separate copy path (BlitEncoder, not
// yet exposed) to populate.
enum class TextureStorage : std::uint8_t {
  Shared  = 1,
  Private = 2,
};

struct TextureDesc {
  PixelFormat    format       = PixelFormat::RGBA8Unorm;
  std::uint32_t  width        = 0;
  std::uint32_t  height       = 0;
  TextureUsage   usage        = TextureUsage::ShaderRead;
  TextureStorage storage_mode = TextureStorage::Shared;
};

// Pair of textures produced by bridging a bi-planar YUV
// CVPixelBuffer (e.g. NV12 from VideoToolbox).
//   * luma   -- plane 0 at full resolution, R8Unorm.
//   * chroma -- plane 1 at half resolution, RG8Unorm
//               (interleaved Cb/Cr).
// Kernels read both and do YCbCr->RGB themselves; the bridge
// does NOT do colour conversion.
struct YuvBiplanarTextures;

// RAII handle on an MTL::Texture. Move-only; default-constructed
// is empty.
//
// Allocated through MetalCompute::make_texture(desc) for fresh
// textures, or MetalCompute::texture_from_cv_pixel_buffer() for
// the zero-copy CVPixelBuffer bridge.
class Texture {
public:
  Texture() noexcept = default;
  Texture(Texture&&) noexcept;
  Texture& operator=(Texture&&) noexcept;
  Texture(const Texture&)            = delete;
  Texture& operator=(const Texture&) = delete;
  ~Texture();

  bool valid() const noexcept { return _tex != nullptr; }

  std::uint32_t width()  const noexcept;
  std::uint32_t height() const noexcept;
  PixelFormat   format() const noexcept { return _format; }

  // CPU bytes-in helper for Shared-storage textures. Writes a
  // tightly-packed buffer of `bytes_per_row * height` bytes into
  // the texture covering (0,0)..(width,height) at mip level 0.
  // No-op for Private storage or invalid texture.
  void replace_region(const void* src, std::size_t bytes_per_row);

  MTL::Texture* mtl_texture() const noexcept { return _tex; }

private:
  friend class MetalCompute;
  Texture(MTL::Texture* tex, PixelFormat format,
          void* cv_handle) noexcept;

  MTL::Texture* _tex       = nullptr;
  PixelFormat   _format    = PixelFormat::Unknown;
  // Retained CVMetalTextureRef for textures created via the CV
  // bridge; the MTL::Texture's lifetime is tied to it. nullptr
  // for natively-allocated textures.
  void*         _cv_handle = nullptr;
};

struct YuvBiplanarTextures {
  Texture luma;     // R8Unorm,  width  x height
  Texture chroma;   // RG8Unorm, width/2 x height/2

  bool valid() const noexcept
  {
    return luma.valid() && chroma.valid();
  }
};

}  // namespace vpipe::metal_compute

#endif
