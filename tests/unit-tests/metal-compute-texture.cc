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
#include <utility>

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

// Read pixel (px, py) from `tex` via the texture_read_pixel_rgba8
// kernel and return the (float4) result through out_pixel.
bool
gpu_read_pixel_(MetalCompute& mc, const Texture& tex,
                std::uint32_t px, std::uint32_t py,
                float out_pixel[4])
{
  ComputeLibrary lib = mc.load_library("texture_read_pixel");
  if (!lib.valid()) {
    return false;
  }
  ComputeFunction fn = lib.function("texture_read_pixel_rgba8");
  if (!fn.valid()) {
    return false;
  }
  SharedBuffer out = mc.make_shared_buffer(4 * sizeof(float));
  if (out.empty()) {
    return false;
  }
  CommandStream s = mc.make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    enc.set_function(fn);
    enc.set_texture(/*index*/ 0, tex);
    enc.set_buffer(/*index*/ 0, out, 0);
    struct ReadParams { std::uint32_t x, y; } params{ px, py };
    enc.set_constant(1, params);
    enc.dispatch({1, 1, 1}, {1, 1, 1});
  }
  CommandStream::Fence f = s.commit();
  f.wait();
  std::memcpy(out_pixel, out.contents(), 4 * sizeof(float));
  return true;
}

}  // namespace

TEST(metal_compute_texture, make_texture_rgba8) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 8;
  d.height = 4;
  d.usage  = TextureUsage::ShaderRead;
  Texture t = mc->make_texture(d);
  EXPECT_TRUE(t.valid());
  EXPECT_TRUE(t.width() == 8u);
  EXPECT_TRUE(t.height() == 4u);
  EXPECT_TRUE(t.format() == PixelFormat::RGBA8Unorm);
  EXPECT_TRUE(t.mtl_texture() != nullptr);
}

TEST(metal_compute_texture, make_texture_zero_size_is_empty) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 0;
  d.height = 4;
  Texture t = mc->make_texture(d);
  EXPECT_FALSE(t.valid());
}

TEST(metal_compute_texture, make_texture_unknown_format_is_empty) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::Unknown;
  d.width  = 8;
  d.height = 4;
  Texture t = mc->make_texture(d);
  EXPECT_FALSE(t.valid());
}

TEST(metal_compute_texture, move_ctor_transfers_ownership) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 4;
  d.height = 4;
  Texture a = mc->make_texture(d);
  EXPECT_TRUE(a.valid());
  auto* mt = a.mtl_texture();

  Texture b(std::move(a));
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_TRUE(b.mtl_texture() == mt);
}

TEST(metal_compute_texture, default_constructed_is_empty) {
  Texture t;
  EXPECT_FALSE(t.valid());
  EXPECT_TRUE(t.mtl_texture() == nullptr);
  EXPECT_TRUE(t.width() == 0u);
  EXPECT_TRUE(t.height() == 0u);
}

TEST(metal_compute_texture, set_texture_no_op_on_empty) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  ComputeEncoder enc = s.begin_compute();
  Texture empty;
  enc.set_texture(0, empty);   // should not crash
  EXPECT_TRUE(enc.valid());
}

TEST(metal_compute_texture, kernel_reads_replaceregion_bytes) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // 4x4 RGBA8 texture; fill with a known pattern; dispatch the
  // read_pixel kernel and verify the sampled float4 matches.
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 4;
  d.height = 4;
  d.usage  = TextureUsage::ShaderRead;
  Texture t = mc->make_texture(d);
  if (!t.valid()) {
    return;
  }
  // Pattern: pixel (x,y) = (x*8, y*8, x+y, 255).
  std::uint8_t pixels[4 * 4 * 4];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      const int idx = (y * 4 + x) * 4;
      pixels[idx + 0] = static_cast<std::uint8_t>(x * 8);
      pixels[idx + 1] = static_cast<std::uint8_t>(y * 8);
      pixels[idx + 2] = static_cast<std::uint8_t>(x + y);
      pixels[idx + 3] = 255;
    }
  }
  t.replace_region(pixels, /*bytes_per_row=*/4 * 4);

  float pixel[4];
  if (!gpu_read_pixel_(*mc, t, /*px*/ 2, /*py*/ 1, pixel)) {
    return;
  }
  // Unorm: byte / 255. (16/255, 8/255, 3/255, 1.0)
  EXPECT_TRUE(pixel[0] > 0.06f && pixel[0] < 0.07f);  // 16/255 ≈ 0.0627
  EXPECT_TRUE(pixel[1] > 0.03f && pixel[1] < 0.04f);  // 8/255 ≈ 0.0314
  EXPECT_TRUE(pixel[2] > 0.01f && pixel[2] < 0.02f);  // 3/255 ≈ 0.0118
  EXPECT_TRUE(pixel[3] > 0.99f);                       // 255/255 = 1.0
}

TEST(metal_compute_texture, cv_pixel_buffer_bridge_bgra) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Create an in-memory CVPixelBuffer (BGRA8, 4x4). IOSurface
  // backing is required so CVMetalTextureCache can wrap it.
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
      nullptr, 4, 4, kCVPixelFormatType_32BGRA, attrs, &pb);
  CFRelease(attrs);
  if (r != kCVReturnSuccess || pb == nullptr) {
    return;
  }
  // Fill with: pixel (x,y) = BGRA(x*16, y*16, x+y, 255)
  CVPixelBufferLockBaseAddress(pb, 0);
  auto* base = static_cast<std::uint8_t*>(
      CVPixelBufferGetBaseAddress(pb));
  const std::size_t bpr = CVPixelBufferGetBytesPerRow(pb);
  for (int y = 0; y < 4; ++y) {
    auto* row = base + y * bpr;
    for (int x = 0; x < 4; ++x) {
      row[x * 4 + 0] = static_cast<std::uint8_t>(x * 16);  // B
      row[x * 4 + 1] = static_cast<std::uint8_t>(y * 16);  // G
      row[x * 4 + 2] = static_cast<std::uint8_t>(x + y);   // R
      row[x * 4 + 3] = 255;
    }
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);

  Texture tex = mc->texture_from_cv_pixel_buffer(
      pb, PixelFormat::BGRA8Unorm);
  EXPECT_TRUE(tex.valid());
  EXPECT_TRUE(tex.width() == 4u);
  EXPECT_TRUE(tex.height() == 4u);

  CFRelease(pb);
}
