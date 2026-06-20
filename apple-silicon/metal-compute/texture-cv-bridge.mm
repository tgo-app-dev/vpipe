// texture-cv-bridge.mm -- Objective-C++ shim for CVMetalTextureCache.
// CoreVideo's CV-to-Metal bridge prototypes take id<MTLDevice>, an
// Obj-C protocol type that the pure C++ TU can't accept. Keep this
// shim minimal: one entry point that creates a CVMetalTextureRef +
// hands back the underlying MTL::Texture*.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreVideo/CVMetalTextureCache.h>
#import <CoreVideo/CVMetalTexture.h>

#include "apple-silicon/metal-compute/texture.h"

#include <Metal/Metal.hpp>  // for MTL::Texture forward use

#include <mutex>

namespace vpipe::metal_compute::_texture_cv {

namespace {

// Cache is per-process (one MTLDevice on Apple Silicon). Lazily
// created on first CV-bridge call; never destroyed -- it caches
// CVMetalTextureRefs for the life of the process.
std::mutex            g_cache_mu;
CVMetalTextureCacheRef g_cache = nullptr;

CVMetalTextureCacheRef
ensure_cache_(MTL::Device* device)
{
  std::lock_guard<std::mutex> g(g_cache_mu);
  if (g_cache != nullptr) {
    return g_cache;
  }
  id<MTLDevice> mtl_dev = (__bridge id<MTLDevice>)
      reinterpret_cast<void*>(device);
  CVReturn r = CVMetalTextureCacheCreate(
      kCFAllocatorDefault, nullptr, mtl_dev, nullptr, &g_cache);
  if (r != kCVReturnSuccess) {
    g_cache = nullptr;
  }
  return g_cache;
}

MTLPixelFormat
to_obj_mtl_pixel_format_(PixelFormat f)
{
  switch (f) {
    case PixelFormat::R8Unorm:     return MTLPixelFormatR8Unorm;
    case PixelFormat::RGBA8Unorm:  return MTLPixelFormatRGBA8Unorm;
    case PixelFormat::BGRA8Unorm:  return MTLPixelFormatBGRA8Unorm;
    case PixelFormat::R32Float:    return MTLPixelFormatR32Float;
    case PixelFormat::RGBA32Float: return MTLPixelFormatRGBA32Float;
    case PixelFormat::R16Float:    return MTLPixelFormatR16Float;
    case PixelFormat::RGBA16Float: return MTLPixelFormatRGBA16Float;
    case PixelFormat::Unknown:     return MTLPixelFormatInvalid;
  }
  return MTLPixelFormatInvalid;
}

}  // namespace

// Shared helper: create one plane texture. Returns the retained
// CVMetalTextureRef on success and writes the MTL::Texture* to
// out_texture; returns nullptr on failure.
static void*
create_plane_texture_(CVMetalTextureCacheRef cache,
                      CVPixelBufferRef pb,
                      MTLPixelFormat fmt,
                      size_t width,
                      size_t height,
                      size_t plane_index,
                      MTL::Texture** out_texture)
{
  CVMetalTextureRef cv_tex = nullptr;
  CVReturn r = CVMetalTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, cache, pb, nullptr,
      fmt, width, height, plane_index, &cv_tex);
  if (r != kCVReturnSuccess || cv_tex == nullptr) {
    if (cv_tex != nullptr) { CFRelease(cv_tex); }
    return nullptr;
  }
  id<MTLTexture> mtl_tex = CVMetalTextureGetTexture(cv_tex);
  if (mtl_tex == nil) {
    CFRelease(cv_tex);
    return nullptr;
  }
  *out_texture = (__bridge MTL::Texture*)
      reinterpret_cast<void*>(mtl_tex);
  return cv_tex;
}

// Returns a retained CVMetalTextureRef on success (caller's
// Texture takes ownership and CFReleases on destruction). The
// underlying MTL::Texture* is written through `out_texture`; its
// lifetime is tied to the CVMetalTextureRef -- do NOT release it
// independently.
//
// Returns nullptr on:
//   * unsupported PixelFormat
//   * cache creation failure
//   * CVMetalTextureCacheCreateTextureFromImage failure
void*
create_texture_from_cv_pixel_buffer(MTL::Device* device,
                                    void* cv_pixel_buffer,
                                    PixelFormat format,
                                    MTL::Texture** out_texture)
{
  *out_texture = nullptr;
  if (device == nullptr || cv_pixel_buffer == nullptr) {
    return nullptr;
  }
  const MTLPixelFormat mtl_fmt = to_obj_mtl_pixel_format_(format);
  if (mtl_fmt == MTLPixelFormatInvalid) {
    return nullptr;
  }
  CVMetalTextureCacheRef cache = ensure_cache_(device);
  if (cache == nullptr) {
    return nullptr;
  }
  CVPixelBufferRef pb = (CVPixelBufferRef)cv_pixel_buffer;
  const size_t w = CVPixelBufferGetWidth(pb);
  const size_t h = CVPixelBufferGetHeight(pb);
  return create_plane_texture_(cache, pb, mtl_fmt, w, h,
                               /*plane=*/0, out_texture);
}

// Bridge an NV12 (4:2:0 bi-planar YUV) CVPixelBuffer to a pair
// of textures: Y (R8Unorm, full res) + CbCr (RG8Unorm, half res).
// Each plane gets its own retained CVMetalTextureRef; both must
// outlive any MTL::Texture* the caller uses, which is why we
// return them in the two `out_*_cv_handle` slots.
//
// Returns true on success; false on format mismatch (the
// CVPixelBuffer is not a 4:2:0 bi-planar format) or cache
// failure. On failure, all output slots are nullptr.
bool
create_nv12_textures_from_cv_pixel_buffer(MTL::Device* device,
                                          void* cv_pixel_buffer,
                                          MTL::Texture** out_luma_tex,
                                          void** out_luma_cv_handle,
                                          MTL::Texture** out_chroma_tex,
                                          void** out_chroma_cv_handle)
{
  *out_luma_tex      = nullptr;
  *out_chroma_tex    = nullptr;
  *out_luma_cv_handle   = nullptr;
  *out_chroma_cv_handle = nullptr;
  if (device == nullptr || cv_pixel_buffer == nullptr) {
    return false;
  }
  CVPixelBufferRef pb = (CVPixelBufferRef)cv_pixel_buffer;
  const OSType pix_fmt = CVPixelBufferGetPixelFormatType(pb);
  const bool is_biplanar_420 =
      (pix_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
   || (pix_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
  if (!is_biplanar_420 || CVPixelBufferGetPlaneCount(pb) < 2) {
    return false;
  }
  CVMetalTextureCacheRef cache = ensure_cache_(device);
  if (cache == nullptr) {
    return false;
  }

  // Plane sizes come from the pixel buffer (not the parent dims);
  // CoreVideo may pad the planes for alignment, and the bridge
  // expects the per-plane logical width/height.
  const size_t y_w = CVPixelBufferGetWidthOfPlane(pb, 0);
  const size_t y_h = CVPixelBufferGetHeightOfPlane(pb, 0);
  const size_t c_w = CVPixelBufferGetWidthOfPlane(pb, 1);
  const size_t c_h = CVPixelBufferGetHeightOfPlane(pb, 1);

  void* luma_handle = create_plane_texture_(
      cache, pb, MTLPixelFormatR8Unorm, y_w, y_h,
      /*plane=*/0, out_luma_tex);
  if (luma_handle == nullptr) {
    return false;
  }
  void* chroma_handle = create_plane_texture_(
      cache, pb, MTLPixelFormatRG8Unorm, c_w, c_h,
      /*plane=*/1, out_chroma_tex);
  if (chroma_handle == nullptr) {
    CFRelease(luma_handle);
    *out_luma_tex = nullptr;
    return false;
  }
  *out_luma_cv_handle   = luma_handle;
  *out_chroma_cv_handle = chroma_handle;
  return true;
}

}  // namespace vpipe::metal_compute::_texture_cv
