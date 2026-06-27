// Unit tests for the letterbox_planar_u8_to_rgb_f16 kernel -- the GPU
// resampler that feeds the CoreML vision tower's MLMultiArray (zero-copy)
// input path. Validates: RGB channel order is PRESERVED (no B/R swap), the
// NCHW vs NHWC layout, exact integer pixel values stored as f16, and the
// aspect-preserving 114 grey letterbox padding. No CoreML model needed.

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  return (mc == nullptr || !mc->valid()) ? nullptr : mc;
}

// Dispatch letterbox_planar_u8_to_rgb_f16 over a planar [3,in_h,in_w] u8 RGB
// source into a fresh f16 buffer of [3,out_h,out_w] (chw) or [out_h,out_w,3].
// Geometry params mirror the host helper. Returns the f16 values as floats.
bool
run_letterbox_(MetalCompute& mc, const std::vector<std::uint8_t>& rgb,
               int in_w, int in_h, int out_w, int out_h, bool chw,
               std::vector<float>* out)
{
  ComputeLibrary lib = mc.load_library("letterbox_planar_u8_to_rgb_f16");
  if (!lib.valid()) { return false; }
  ComputeFunction fn = lib.function("letterbox_planar_u8_to_rgb_f16");
  if (!fn.valid()) { return false; }

  SharedBuffer src = mc.make_shared_buffer(rgb.size());
  if (src.empty()) { return false; }
  std::memcpy(src.contents(), rgb.data(), rgb.size());

  const std::size_t dst_elems = (std::size_t)3 * out_w * out_h;
  SharedBuffer dst = mc.make_shared_buffer(dst_elems * 2);
  if (dst.empty()) { return false; }

  const float sx = (float)out_w / (float)in_w;
  const float sy = (float)out_h / (float)in_h;
  const float scale = sx < sy ? sx : sy;
  const int new_w = (int)(scale * in_w + 0.5f);
  const int new_h = (int)(scale * in_h + 0.5f);
  const int pad_x = (out_w - new_w) / 2;
  const int pad_y = (out_h - new_h) / 2;
  const float inv = 1.0f / scale;

  std::uint32_t params[4]  = { (std::uint32_t)in_w, (std::uint32_t)in_h,
                               (std::uint32_t)out_w, (std::uint32_t)out_h };
  std::uint32_t params2[4] = { (std::uint32_t)new_w, (std::uint32_t)new_h,
                               (std::uint32_t)pad_x, (std::uint32_t)pad_y };
  float params3[4]         = { inv, 0.0f, 0.0f, 0.0f };
  std::uint32_t chw_u      = chw ? 1u : 0u;

  CommandStream s = mc.make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    if (!enc.valid()) { return false; }
    enc.set_function(fn);
    enc.set_buffer(0, src, 0);
    enc.set_buffer(1, dst, 0);
    enc.set_constant_bytes(2, params,  sizeof(params));
    enc.set_constant_bytes(3, params2, sizeof(params2));
    enc.set_constant_bytes(4, params3, sizeof(params3));
    enc.set_constant_bytes(5, &chw_u, sizeof(chw_u));
    enc.dispatch({(unsigned)out_w, (unsigned)out_h, 1u}, {8u, 8u, 1u});
  }
  s.commit().wait();

  const _Float16* p = static_cast<const _Float16*>(dst.contents());
  out->resize(dst_elems);
  for (std::size_t i = 0; i < dst_elems; ++i) { (*out)[i] = (float)p[i]; }
  return true;
}

// Planar [3,h,w] RGB with distinct, channel-separable values per pixel.
std::vector<std::uint8_t>
make_planar_(int w, int h)
{
  std::vector<std::uint8_t> v((std::size_t)3 * w * h);
  const std::size_t plane = (std::size_t)w * h;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = (std::size_t)y * w + x;
      v[0 * plane + p] = (std::uint8_t)(10 + x);          // R
      v[1 * plane + p] = (std::uint8_t)(50 + y);          // G
      v[2 * plane + p] = (std::uint8_t)(100 + x + y);     // B
    }
  }
  return v;
}

}  // namespace

// in == out: a pure copy. NCHW output must equal the planar input exactly
// (RGB order preserved, integer values exact as f16).
TEST(letterbox_rgb_f16, identity_nchw_preserves_rgb) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  const int w = 4, h = 2;
  auto rgb = make_planar_(w, h);
  std::vector<float> got;
  ASSERT_TRUE(run_letterbox_(*mc, rgb, w, h, w, h, /*chw=*/true, &got));
  const std::size_t plane = (std::size_t)w * h;
  bool ok = true;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = (std::size_t)y * w + x;
      if (got[0 * plane + p] != (float)(10 + x))      { ok = false; }
      if (got[1 * plane + p] != (float)(50 + y))      { ok = false; }
      if (got[2 * plane + p] != (float)(100 + x + y)) { ok = false; }
    }
  }
  EXPECT_TRUE(ok);
}

// Same content, NHWC interleaved layout: dst[(y*w+x)*3 + c].
TEST(letterbox_rgb_f16, identity_nhwc_interleaved) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  const int w = 4, h = 2;
  auto rgb = make_planar_(w, h);
  std::vector<float> got;
  ASSERT_TRUE(run_letterbox_(*mc, rgb, w, h, w, h, /*chw=*/false, &got));
  bool ok = true;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t o = ((std::size_t)y * w + x) * 3;
      if (got[o + 0] != (float)(10 + x))      { ok = false; }
      if (got[o + 1] != (float)(50 + y))      { ok = false; }
      if (got[o + 2] != (float)(100 + x + y)) { ok = false; }
    }
  }
  EXPECT_TRUE(ok);
}

// 2x2 -> 4x2 letterbox: 1-column grey (114) pad on each side, content
// (unscaled, scale==1) centered in columns [1,2]. NCHW.
TEST(letterbox_rgb_f16, pads_with_grey_114) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  const int in_w = 2, in_h = 2, out_w = 4, out_h = 2;
  auto rgb = make_planar_(in_w, in_h);
  std::vector<float> got;
  ASSERT_TRUE(run_letterbox_(*mc, rgb, in_w, in_h, out_w, out_h,
                             /*chw=*/true, &got));
  const std::size_t plane = (std::size_t)out_w * out_h;
  bool pad_ok = true, content_ok = true;
  for (int y = 0; y < out_h; ++y) {
    // Padded columns 0 and 3 are grey on every channel.
    for (int xc : {0, 3}) {
      const std::size_t p = (std::size_t)y * out_w + xc;
      for (int c = 0; c < 3; ++c) {
        if (got[(std::size_t)c * plane + p] != 114.0f) { pad_ok = false; }
      }
    }
    // Column 1 holds source column 0; column 2 holds source column 1.
    for (int sx = 0; sx < 2; ++sx) {
      const std::size_t p = (std::size_t)y * out_w + (1 + sx);
      if (got[0 * plane + p] != (float)(10 + sx))      { content_ok = false; }
      if (got[1 * plane + p] != (float)(50 + y))       { content_ok = false; }
      if (got[2 * plane + p] != (float)(100 + sx + y)) { content_ok = false; }
    }
  }
  EXPECT_TRUE(pad_ok);
  EXPECT_TRUE(content_ok);
}
