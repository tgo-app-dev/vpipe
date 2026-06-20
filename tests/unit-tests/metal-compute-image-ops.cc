// Tests for the metal-compute image-preprocessing ops (formerly
// MetalRuntime; retired in favour of metal-compute). Each op is
// validated against a CPU reference. Skips gracefully when metal-compute
// is unavailable.

#include "minitest.h"
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-storage.h"
#include "common/session.h"
#include "interfaces/session-context-intf.h"

#include <CoreVideo/CVPixelBuffer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace vpipe;
namespace mc = vpipe::metal_compute;

namespace {

void
cpu_letterbox_ref_(const uint8_t* src,
                   int            in_w,
                   int            in_h,
                   int            out_w,
                   int            out_h,
                   float*         dst)
{
  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = std::min(sx, sy);
  const int new_w = static_cast<int>(std::round(scale * in_w));
  const int new_h = static_cast<int>(std::round(scale * in_h));
  const int pad_x = (out_w - new_w) / 2;
  const int pad_y = (out_h - new_h) / 2;
  const float inv = 1.0f / scale;
  const float pad_v = 114.0f / 255.0f;
  const size_t plane_src = static_cast<size_t>(in_w) * in_h;
  const size_t plane_dst = static_cast<size_t>(out_w) * out_h;
  for (int c = 0; c < 3; ++c) {
    const uint8_t* src_c = src + c * plane_src;
    float*         dst_c = dst + c * plane_dst;
    for (int y = 0; y < out_h; ++y) {
      for (int x = 0; x < out_w; ++x) {
        const int xo = x;
        const int yo = y;
        const bool out =
            xo < pad_x || xo >= pad_x + new_w
         || yo < pad_y || yo >= pad_y + new_h;
        float v;
        if (out) {
          v = pad_v;
        } else {
          float sxf = (xo - pad_x + 0.5f) * inv - 0.5f;
          float syf = (yo - pad_y + 0.5f) * inv - 0.5f;
          int ix0 = static_cast<int>(std::floor(sxf));
          int iy0 = static_cast<int>(std::floor(syf));
          int ix1 = ix0 + 1;
          int iy1 = iy0 + 1;
          float fx = sxf - ix0;
          float fy = syf - iy0;
          ix0 = std::clamp(ix0, 0, in_w - 1);
          ix1 = std::clamp(ix1, 0, in_w - 1);
          iy0 = std::clamp(iy0, 0, in_h - 1);
          iy1 = std::clamp(iy1, 0, in_h - 1);
          float v00 = src_c[iy0 * in_w + ix0] * (1.0f / 255.0f);
          float v01 = src_c[iy0 * in_w + ix1] * (1.0f / 255.0f);
          float v10 = src_c[iy1 * in_w + ix0] * (1.0f / 255.0f);
          float v11 = src_c[iy1 * in_w + ix1] * (1.0f / 255.0f);
          float v0  = v00 + fx * (v01 - v00);
          float v1  = v10 + fx * (v11 - v10);
          v = v0 + fy * (v1 - v0);
        }
        dst_c[y * out_w + x] = v;
      }
    }
  }
}

}  // namespace

TEST(metal_compute_image_ops, make_shared_storage_returns_handle)
{
  Session sess(R"({})");
  auto* dev = sess.metal_compute();
  if (!dev || !dev->valid()) {
    return;
  }
  auto s = mc::make_shared_storage(*dev, 1024, &sess);
  EXPECT_TRUE((bool)s);
  EXPECT_TRUE(s->mtl_buffer != nullptr);
  EXPECT_TRUE(s->contents   != nullptr);
  EXPECT_TRUE(s->byte_size  == 1024u);
  EXPECT_TRUE(s->deleter    != nullptr);
  for (size_t i = 0; i < 16; ++i) {
    s->contents[i] = static_cast<uint8_t>(i ^ 0x5a);
  }
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_TRUE(s->contents[i] == static_cast<uint8_t>(i ^ 0x5a));
  }
}

TEST(metal_compute_image_ops, letterbox_planar_u8_matches_cpu_reference)
{
  Session sess(R"({})");
  auto* dev = sess.metal_compute();
  if (!dev || !dev->valid()) { return; }
  constexpr int in_w = 320;
  constexpr int in_h = 200;
  constexpr int S    = 256;
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  const size_t need_dst = static_cast<size_t>(3) * S * S * sizeof(float);
  auto src = mc::make_shared_storage(*dev, need_src, &sess);
  auto dst = mc::make_shared_storage(*dev, need_dst, &sess);
  if (!src || !dst) { return; }
  uint8_t* sbytes = src->contents;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < in_h; ++y) {
      for (int x = 0; x < in_w; ++x) {
        sbytes[c * in_w * in_h + y * in_w + x] =
            static_cast<uint8_t>((c * 71 + y * 31 + x) & 0xff);
      }
    }
  }
  float scale = 0.0f;
  int px = 0, py = 0;
  const bool ok = mc::letterbox_planar_u8_to_f32_chw(
      *dev, *src, in_w, in_h, *dst, S, S, &scale, &px, &py, &sess);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(std::abs(scale - 0.8f) < 1e-6f);
  EXPECT_TRUE(px == 0);
  EXPECT_TRUE(py == 48);
  std::vector<float> ref(static_cast<size_t>(3) * S * S);
  cpu_letterbox_ref_(sbytes, in_w, in_h, S, S, ref.data());
  const float* gpu = reinterpret_cast<const float*>(dst->contents);
  size_t mismatches = 0;
  float  max_abs    = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float diff = std::abs(gpu[i] - ref[i]);
    if (diff > max_abs) { max_abs = diff; }
    if (diff > 2.0e-4f) { ++mismatches; }
  }
  EXPECT_TRUE(mismatches == 0u);
  EXPECT_TRUE(max_abs < 2.0e-4f);
  const float pad_v = 114.0f / 255.0f;
  for (int c = 0; c < 3; ++c) {
    for (int y = py + 160; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        EXPECT_TRUE(
            std::abs(gpu[c * S * S + y * S + x] - pad_v) < 1e-6f);
      }
    }
  }
}

// The realtime-vqa VLM-input resampler uses letterbox_planar_u8_to_u8_chw
// (u8 in -> u8 out), which the tests above don't cover (they hit the f32
// and bgra variants). Gate that it RESAMPLES (aspect-preserving resize +
// centred 114-grey pad) rather than CROPS the top-left: compare to the
// (true-resize) CPU reference rounded to u8, and assert the top rows are
// padding (a crop would have image content there).
TEST(metal_compute_image_ops, letterbox_u8_to_u8_resamples_not_crops)
{
  Session sess(R"({})");
  auto* dev = sess.metal_compute();
  if (!dev || !dev->valid()) { return; }
  constexpr int in_w = 320, in_h = 200, out_w = 256, out_h = 256;
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  const size_t need_dst = static_cast<size_t>(3) * out_w * out_h;
  auto src = mc::make_shared_storage(*dev, need_src, &sess);
  auto dst = mc::make_shared_storage(*dev, need_dst, &sess);
  if (!src || !dst) { return; }
  uint8_t* sb = src->contents;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < in_h; ++y) {
      for (int x = 0; x < in_w; ++x) {
        sb[c * in_w * in_h + y * in_w + x] =
            static_cast<uint8_t>((c * 71 + y * 31 + x) & 0xff);
      }
    }
  }
  const bool ok = mc::letterbox_planar_u8_to_u8_chw(
      *dev, *src, in_w, in_h, *dst, out_w, out_h, &sess);
  EXPECT_TRUE(ok);

  // 320x200 -> scale 0.8 -> content 256x160, pad_x=0, pad_y=48.
  std::vector<float> ref(static_cast<size_t>(3) * out_w * out_h);
  cpu_letterbox_ref_(sb, in_w, in_h, out_w, out_h, ref.data());
  const uint8_t* gpu = dst->contents;
  size_t mismatches = 0;
  int    max_abs    = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const int want = static_cast<int>(
        std::lround(std::clamp(ref[i] * 255.0f, 0.0f, 255.0f)));
    const int d = std::abs(static_cast<int>(gpu[i]) - want);
    if (d > max_abs) { max_abs = d; }
    if (d > 1) { ++mismatches; }   // allow +/-1 u8 rounding
  }
  EXPECT_TRUE(mismatches == 0u);
  EXPECT_TRUE(max_abs <= 1);
  // Crop detector: rows 0..47 are the top letterbox pad -> must be 114
  // grey on every channel. A top-left crop would have image content here.
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 48; ++y) {
      for (int x = 0; x < out_w; ++x) {
        EXPECT_TRUE(gpu[c * out_w * out_h + y * out_w + x] == 114);
      }
    }
  }
}

TEST(metal_compute_image_ops, letterbox_non_square_matches_cpu_reference)
{
  Session sess(R"({})");
  auto* dev = sess.metal_compute();
  if (!dev || !dev->valid()) { return; }
  constexpr int in_w  = 320;
  constexpr int in_h  = 240;
  constexpr int out_w = 640;
  constexpr int out_h = 384;
  const size_t need_src =
      static_cast<size_t>(3) * in_w * in_h;
  const size_t need_dst =
      static_cast<size_t>(3) * out_w * out_h * sizeof(float);
  auto src = mc::make_shared_storage(*dev, need_src, &sess);
  auto dst = mc::make_shared_storage(*dev, need_dst, &sess);
  if (!src || !dst) { return; }
  uint8_t* sbytes = src->contents;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < in_h; ++y) {
      for (int x = 0; x < in_w; ++x) {
        sbytes[c * in_w * in_h + y * in_w + x] =
            static_cast<uint8_t>((c * 53 + y * 17 + x * 11) & 0xff);
      }
    }
  }
  float scale = 0.0f;
  int px = 0, py = 0;
  const bool ok = mc::letterbox_planar_u8_to_f32_chw(
      *dev, *src, in_w, in_h, *dst, out_w, out_h, &scale, &px, &py,
      &sess);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(std::abs(scale - 1.6f) < 1e-6f);
  EXPECT_TRUE(px == 64);
  EXPECT_TRUE(py == 0);
  std::vector<float> ref(static_cast<size_t>(3) * out_w * out_h);
  cpu_letterbox_ref_(sbytes, in_w, in_h, out_w, out_h, ref.data());
  const float* gpu = reinterpret_cast<const float*>(dst->contents);
  size_t mismatches = 0;
  float  max_abs    = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float diff = std::abs(gpu[i] - ref[i]);
    if (diff > max_abs) { max_abs = diff; }
    if (diff > 2.0e-4f) { ++mismatches; }
  }
  EXPECT_TRUE(mismatches == 0u);
  EXPECT_TRUE(max_abs < 2.0e-4f);
  const float pad_v = 114.0f / 255.0f;
  for (int y = 0; y < out_h; ++y) {
    EXPECT_TRUE(std::abs(gpu[y * out_w + 0]   - pad_v) < 1e-6f);
    EXPECT_TRUE(std::abs(gpu[y * out_w + 63]  - pad_v) < 1e-6f);
    EXPECT_TRUE(std::abs(gpu[y * out_w + 576] - pad_v) < 1e-6f);
    EXPECT_TRUE(std::abs(gpu[y * out_w + 639] - pad_v) < 1e-6f);
  }
}

namespace {

void
cpu_letterbox_bgra_ref_(const uint8_t* src,
                        int            in_w,
                        int            in_h,
                        int            out_w,
                        int            out_h,
                        size_t         bpr,
                        uint8_t*       dst)
{
  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = std::min(sx, sy);
  const int new_w = static_cast<int>(std::round(scale * in_w));
  const int new_h = static_cast<int>(std::round(scale * in_h));
  const int pad_x = (out_w - new_w) / 2;
  const int pad_y = (out_h - new_h) / 2;
  const float inv = 1.0f / scale;
  const size_t plane_src = static_cast<size_t>(in_w) * in_h;
  const uint8_t* r_plane = src + 0 * plane_src;
  const uint8_t* g_plane = src + 1 * plane_src;
  const uint8_t* b_plane = src + 2 * plane_src;
  for (int y = 0; y < out_h; ++y) {
    uint8_t* row = dst + y * bpr;
    for (int x = 0; x < out_w; ++x) {
      const bool outside =
          x < pad_x || x >= pad_x + new_w
       || y < pad_y || y >= pad_y + new_h;
      if (outside) {
        row[x * 4 + 0] = 114;
        row[x * 4 + 1] = 114;
        row[x * 4 + 2] = 114;
        row[x * 4 + 3] = 255;
        continue;
      }
      const float sxf = (x - pad_x + 0.5f) * inv - 0.5f;
      const float syf = (y - pad_y + 0.5f) * inv - 0.5f;
      int ix0 = static_cast<int>(std::floor(sxf));
      int iy0 = static_cast<int>(std::floor(syf));
      int ix1 = ix0 + 1;
      int iy1 = iy0 + 1;
      const float fx = sxf - ix0;
      const float fy = syf - iy0;
      ix0 = std::clamp(ix0, 0, in_w - 1);
      ix1 = std::clamp(ix1, 0, in_w - 1);
      iy0 = std::clamp(iy0, 0, in_h - 1);
      iy1 = std::clamp(iy1, 0, in_h - 1);
      auto sample_ = [&](const uint8_t* p) -> uint8_t {
        const float v00 = p[iy0 * in_w + ix0];
        const float v01 = p[iy0 * in_w + ix1];
        const float v10 = p[iy1 * in_w + ix0];
        const float v11 = p[iy1 * in_w + ix1];
        const float v0  = v00 + fx * (v01 - v00);
        const float v1  = v10 + fx * (v11 - v10);
        const float v   = v0 + fy * (v1 - v0);
        return static_cast<uint8_t>(std::clamp(v + 0.5f, 0.0f, 255.0f));
      };
      row[x * 4 + 0] = sample_(b_plane);   // B
      row[x * 4 + 1] = sample_(g_plane);   // G
      row[x * 4 + 2] = sample_(r_plane);   // R
      row[x * 4 + 3] = 255;
    }
  }
}

}  // namespace

TEST(metal_compute_image_ops, letterbox_bgra_matches_cpu_reference)
{
  Session sess(R"({})");
  auto* dev = sess.metal_compute();
  if (!dev || !dev->valid()) { return; }
  constexpr int in_w  = 320;
  constexpr int in_h  = 240;
  constexpr int out_w = 416;
  constexpr int out_h = 416;
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  auto src = mc::make_shared_storage(*dev, need_src, &sess);
  if (!src) { return; }
  uint8_t* sbytes = src->contents;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < in_h; ++y) {
      for (int x = 0; x < in_w; ++x) {
        sbytes[c * in_w * in_h + y * in_w + x] =
            static_cast<uint8_t>((c * 91 + y * 23 + x * 7) & 0xff);
      }
    }
  }

  CVPixelBufferRef pb = nullptr;
  CVReturn cv_rc = CVPixelBufferCreate(
      kCFAllocatorDefault,
      static_cast<size_t>(out_w),
      static_cast<size_t>(out_h),
      kCVPixelFormatType_32BGRA,
      nullptr,
      &pb);
  EXPECT_TRUE(cv_rc == kCVReturnSuccess && pb != nullptr);
  if (!pb) { return; }

  float scale = 0.0f;
  int   px    = 0;
  int   py    = 0;
  const bool ok = mc::letterbox_planar_u8_to_bgra_cvpixelbuffer(
      *dev, *src, in_w, in_h, pb, &scale, &px, &py, &sess);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(std::abs(scale - 1.3f) < 1e-6f);
  EXPECT_TRUE(px == 0);
  EXPECT_TRUE(py == 52);

  CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
  const auto*   gpu = static_cast<const uint8_t*>(
      CVPixelBufferGetBaseAddress(pb));
  const size_t  bpr = CVPixelBufferGetBytesPerRow(pb);
  std::vector<uint8_t> ref(bpr * out_h);
  cpu_letterbox_bgra_ref_(sbytes, in_w, in_h, out_w, out_h, bpr,
                          ref.data());
  size_t mismatches = 0;
  int    max_abs    = 0;
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      for (int c = 0; c < 4; ++c) {
        const int gv = gpu[y * bpr + x * 4 + c];
        const int rv = ref[y * bpr + x * 4 + c];
        const int d  = std::abs(gv - rv);
        if (d > max_abs) { max_abs = d; }
        if (d > 1)       { ++mismatches; }
      }
    }
  }
  EXPECT_TRUE(mismatches == 0u);
  EXPECT_TRUE(max_abs <= 1);
  for (int y = 0; y < py; ++y) {
    EXPECT_TRUE(gpu[y * bpr + 0] == 114);
    EXPECT_TRUE(gpu[y * bpr + 1] == 114);
    EXPECT_TRUE(gpu[y * bpr + 2] == 114);
    EXPECT_TRUE(gpu[y * bpr + 3] == 255);
  }
  CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
  CFRelease(pb);
}

TEST(metal_compute_image_ops, letterbox_rejects_null_handle)
{
  Session sess(R"({})");
  auto* dev = sess.metal_compute();
  if (!dev || !dev->valid()) { return; }
  ExternalStorageHandle empty;
  auto dst = mc::make_shared_storage(
      *dev, static_cast<size_t>(3) * 16 * 16 * sizeof(float), &sess);
  if (!dst) { return; }
  const bool ok = mc::letterbox_planar_u8_to_f32_chw(
      *dev, empty, 16, 16, *dst, 16, 16, nullptr, nullptr, nullptr,
      &sess);
  EXPECT_TRUE(!ok);
}
