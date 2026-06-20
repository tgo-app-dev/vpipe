#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/texture.h"
#include "common/session.h"

#include <CoreVideo/CVPixelBuffer.h>
#include <Metal/Metal.hpp>

#include <cstdint>
#include <cstring>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

// Build an 8x8 NV12 CVPixelBuffer in memory, filled with:
//   Y plane     -- (x*16 + y) clamped to byte (so cell (x,y) holds
//                  a known value the test can verify).
//   CbCr plane  -- interleaved (Cb=x*32, Cr=y*32) per cell at
//                  half resolution (4x4 cells covering the 8x8
//                  source).
// Returns nullptr on allocation failure (test self-skips).
CVPixelBufferRef
make_test_nv12_buffer_(int w, int h)
{
  CFDictionaryRef empty = CFDictionaryCreate(
      nullptr, nullptr, nullptr, 0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFStringRef keys[] = {
      kCVPixelBufferIOSurfacePropertiesKey,
      kCVPixelBufferMetalCompatibilityKey };
  CFTypeRef vals[] = { empty, kCFBooleanTrue };
  CFDictionaryRef attrs = CFDictionaryCreate(
      nullptr, (const void**)keys, (const void**)vals, 2,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFRelease(empty);

  CVPixelBufferRef pb = nullptr;
  CVReturn r = CVPixelBufferCreate(
      nullptr, w, h,
      kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
      attrs, &pb);
  CFRelease(attrs);
  if (r != kCVReturnSuccess || pb == nullptr) {
    return nullptr;
  }

  CVPixelBufferLockBaseAddress(pb, 0);
  // Y plane
  auto* y_base = static_cast<std::uint8_t*>(
      CVPixelBufferGetBaseAddressOfPlane(pb, 0));
  const std::size_t y_bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
  for (int y = 0; y < h; ++y) {
    auto* row = y_base + y * y_bpr;
    for (int x = 0; x < w; ++x) {
      row[x] = static_cast<std::uint8_t>((x * 16 + y) & 0xff);
    }
  }
  // CbCr plane (half res, interleaved Cb then Cr per cell)
  auto* c_base = static_cast<std::uint8_t*>(
      CVPixelBufferGetBaseAddressOfPlane(pb, 1));
  const std::size_t c_bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
  const int ch = h / 2;
  const int cw = w / 2;
  for (int y = 0; y < ch; ++y) {
    auto* row = c_base + y * c_bpr;
    for (int x = 0; x < cw; ++x) {
      row[x * 2 + 0] = static_cast<std::uint8_t>((x * 32) & 0xff);
      row[x * 2 + 1] = static_cast<std::uint8_t>((y * 32) & 0xff);
    }
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);
  return pb;
}

}  // namespace

TEST(metal_compute_nv12, bridge_returns_two_textures) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CVPixelBufferRef pb = make_test_nv12_buffer_(8, 8);
  if (pb == nullptr) {
    return;
  }
  YuvBiplanarTextures t = mc->nv12_textures_from_cv_pixel_buffer(pb);
  EXPECT_TRUE(t.valid());
  EXPECT_TRUE(t.luma.width() == 8u);
  EXPECT_TRUE(t.luma.height() == 8u);
  EXPECT_TRUE(t.chroma.width() == 4u);
  EXPECT_TRUE(t.chroma.height() == 4u);
  CFRelease(pb);
}

TEST(metal_compute_nv12, bridge_rejects_bgra_buffer) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // BGRA CVPixelBuffer is single-plane; bridge should reject.
  CFDictionaryRef empty = CFDictionaryCreate(
      nullptr, nullptr, nullptr, 0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFStringRef keys[] = {
      kCVPixelBufferIOSurfacePropertiesKey,
      kCVPixelBufferMetalCompatibilityKey };
  CFTypeRef vals[] = { empty, kCFBooleanTrue };
  CFDictionaryRef attrs = CFDictionaryCreate(
      nullptr, (const void**)keys, (const void**)vals, 2,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFRelease(empty);
  CVPixelBufferRef pb = nullptr;
  CVPixelBufferCreate(nullptr, 4, 4,
                       kCVPixelFormatType_32BGRA, attrs, &pb);
  CFRelease(attrs);
  if (pb == nullptr) {
    return;
  }
  YuvBiplanarTextures t = mc->nv12_textures_from_cv_pixel_buffer(pb);
  EXPECT_FALSE(t.valid());
  CFRelease(pb);
}

TEST(metal_compute_nv12, bridge_rejects_null) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  YuvBiplanarTextures t = mc->nv12_textures_from_cv_pixel_buffer(nullptr);
  EXPECT_FALSE(t.valid());
}

TEST(metal_compute_nv12, kernel_reads_both_planes) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CVPixelBufferRef pb = make_test_nv12_buffer_(8, 8);
  if (pb == nullptr) {
    return;
  }
  YuvBiplanarTextures yc = mc->nv12_textures_from_cv_pixel_buffer(pb);
  if (!yc.valid()) {
    CFRelease(pb);
    return;
  }
  ComputeLibrary lib = mc->load_library("nv12_read");
  if (!lib.valid()) {
    CFRelease(pb);
    return;
  }
  ComputeFunction fn = lib.function("nv12_read");
  if (!fn.valid()) {
    CFRelease(pb);
    return;
  }
  // Sample pixel (x=4, y=2): luma  = (4*16+2) = 66 -> 66/255
  //                           cbcr_index (2, 1) -> Cb=2*32=64,
  //                                                Cr=1*32=32
  SharedBuffer out = mc->make_shared_buffer(4 * sizeof(float));
  CommandStream s = mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    enc.set_function(fn);
    enc.set_texture(0, yc.luma);
    enc.set_texture(1, yc.chroma);
    enc.set_buffer(0, out, 0);
    struct ReadParams { std::uint32_t x, y; } p{ 4, 2 };
    enc.set_constant(1, p);
    enc.dispatch({1, 1, 1}, {1, 1, 1});
  }
  s.commit().wait();

  float pixel[4];
  std::memcpy(pixel, out.contents(), sizeof(pixel));
  // Luma 66/255 ~= 0.2588
  EXPECT_TRUE(pixel[0] > 0.25f && pixel[0] < 0.27f);
  // Cb 64/255 ~= 0.251
  EXPECT_TRUE(pixel[1] > 0.24f && pixel[1] < 0.26f);
  // Cr 32/255 ~= 0.125
  EXPECT_TRUE(pixel[2] > 0.12f && pixel[2] < 0.13f);

  CFRelease(pb);
}
