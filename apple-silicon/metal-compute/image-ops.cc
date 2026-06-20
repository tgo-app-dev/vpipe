#include "apple-silicon/metal-compute/image-ops.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

// metal-cpp types for our MTL::Buffer / MTL::Device signatures. The
// MTL_PRIVATE_IMPLEMENTATION / Foundation macros are defined elsewhere
// (metal-compute.cc, or libmlx under VPIPE_FOUNDATION_FROM_MLX); we just
// pull in the headers.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <CoreVideo/CVPixelBuffer.h>
#include <CoreVideo/CVPixelBufferIOSurface.h>

#include <cmath>
#include <cstring>

using namespace std;

namespace vpipe::metal_compute {

namespace {

// Centered-crop helper for the NV12 path. Aspect-correct crop that
// keeps even pixel dims (NV12 chroma is 4:2:0 -- odd crop offsets
// would mis-align the chroma plane sampler).
struct CenteredCrop {
  int crop_w;
  int crop_h;
  int crop_left;
  int crop_top;
};

CenteredCrop
centered_crop_(int in_w, int in_h, int out_w, int out_h)
{
  const long long in_x_out_h  =
      static_cast<long long>(in_w) * out_h;
  const long long out_x_in_h  =
      static_cast<long long>(out_w) * in_h;

  CenteredCrop c{};
  if (in_x_out_h == out_x_in_h) {
    c.crop_w    = in_w;
    c.crop_h    = in_h;
    c.crop_left = 0;
    c.crop_top  = 0;
    return c;
  }
  if (in_x_out_h > out_x_in_h) {
    c.crop_h = in_h;
    c.crop_w =
        static_cast<int>(static_cast<long long>(in_h) * out_w / out_h);
    if (c.crop_w & 1) { --c.crop_w; }
    if (c.crop_w > in_w) { c.crop_w = in_w & ~1; }
    c.crop_left = (in_w - c.crop_w) / 2;
    if (c.crop_left & 1) { --c.crop_left; }
    c.crop_top  = 0;
  } else {
    c.crop_w = in_w;
    c.crop_h =
        static_cast<int>(static_cast<long long>(in_w) * out_h / out_w);
    if (c.crop_h & 1) { --c.crop_h; }
    if (c.crop_h > in_h) { c.crop_h = in_h & ~1; }
    c.crop_top  = (in_h - c.crop_h) / 2;
    if (c.crop_top & 1) { --c.crop_top; }
    c.crop_left = 0;
  }
  return c;
}

struct Dims2D {
  LaunchDims grid;
  LaunchDims threadgroup;
};

Dims2D
dims2d_(const ComputeFunction& fn, int grid_w, int grid_h)
{
  const unsigned tew = fn.thread_execution_width();
  const unsigned max_total = fn.max_total_threads_per_threadgroup();
  const unsigned tgh = tew == 0 ? 1 : max_total / tew;
  Dims2D out;
  out.grid        = { static_cast<unsigned>(grid_w),
                      static_cast<unsigned>(grid_h),
                      1 };
  out.threadgroup = { tew == 0 ? 1u : tew,
                      tgh == 0 ? 1u : tgh,
                      1 };
  return out;
}

// Common shape of every dispatcher: load_library + function (cached),
// open a CommandStream + ComputeEncoder, run `body`, commit, wait.
template <class BodyFn>
bool
dispatch_(MetalCompute&     mc,
          std::string_view  lib_name,
          std::string_view  fn_name,
          const char*       what,
          const SessionContextIntf* session,
          BodyFn&&          body)
{
  ComputeLibrary lib = mc.load_library(lib_name);
  if (!lib.valid()) {
    if (session) {
      session->warn(fmt(
          "metal_compute::image_ops: load_library('{}') failed -- {} "
          "disabled", string(lib_name), what));
    }
    return false;
  }
  ComputeFunction fn = lib.function(fn_name);
  if (!fn.valid()) {
    if (session) {
      session->warn(fmt(
          "metal_compute::image_ops: function('{}::{}') failed -- {} "
          "disabled", string(lib_name), string(fn_name), what));
    }
    return false;
  }
  CommandStream stream = mc.make_command_stream();
  bool encode_ok = true;
  {
    ComputeEncoder enc = stream.begin_compute();
    if (!enc.valid()) {
      return false;
    }
    enc.set_function(fn);
    encode_ok = body(enc, fn);
  }
  if (!encode_ok) {
    return false;
  }
  CommandStream::Fence cb = stream.commit();
  cb.wait();
  if (!cb.completed()) {
    if (session) {
      session->warn(fmt(
          "metal_compute::image_ops: {} command buffer ended "
          "uncompleted", what));
    }
    return false;
  }
  return true;
}

// Stage `byte_size` bytes from `src` into a fresh Shared MTL::Buffer.
SharedBuffer
staging_from_bytes_(MetalCompute& mc, const void* src,
                    std::size_t byte_size)
{
  SharedBuffer b = mc.make_shared_buffer(byte_size);
  if (b.empty()) {
    return b;
  }
  std::memcpy(b.contents(), src, byte_size);
  return b;
}

void
release_shared_buffer_(void* p)
{
  if (!p) { return; }
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  static_cast<MTL::Buffer*>(p)->release();
  pool->release();
}

bool
dispatch_nv12_kernel_(MetalCompute&            mc,
                      void*                    cv_pixel_buffer,
                      MTL::Buffer*             dst_buf,
                      int                      src_width,
                      int                      src_height,
                      int                      out_width,
                      int                      out_height,
                      const SessionContextIntf* session)
{
  CVPixelBufferRef pb =
      static_cast<CVPixelBufferRef>(cv_pixel_buffer);
  const OSType pix_fmt = CVPixelBufferGetPixelFormatType(pb);
  const bool is_nv12 =
      (pix_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
   || (pix_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
  if (!is_nv12) { return false; }
  if (CVPixelBufferGetPlaneCount(pb) < 2) { return false; }

  CVReturn lock = CVPixelBufferLockBaseAddress(
      pb, kCVPixelBufferLock_ReadOnly);
  if (lock != kCVReturnSuccess) { return false; }
  void* y_base    = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
  void* cbcr_base = CVPixelBufferGetBaseAddressOfPlane(pb, 1);
  const size_t y_stride =
      CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
  const size_t cbcr_stride =
      CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
  if (!y_base || !cbcr_base
      || y_stride < static_cast<size_t>(src_width)) {
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return false;
  }

  const size_t y_bytes    = y_stride    * src_height;
  const size_t cbcr_bytes = cbcr_stride * (src_height / 2);

  SharedBuffer y_buf    = staging_from_bytes_(mc, y_base,    y_bytes);
  SharedBuffer cbcr_buf =
      staging_from_bytes_(mc, cbcr_base, cbcr_bytes);
  if (y_buf.empty() || cbcr_buf.empty()) {
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return false;
  }

  const CenteredCrop cc =
      centered_crop_(src_width, src_height, out_width, out_height);
  const float scale_x =
      static_cast<float>(cc.crop_w) / static_cast<float>(out_width);
  const float scale_y =
      static_cast<float>(cc.crop_h) / static_cast<float>(out_height);

  uint32_t params[4] = {
      static_cast<uint32_t>(src_width),
      static_cast<uint32_t>(src_height),
      static_cast<uint32_t>(y_stride),
      static_cast<uint32_t>(cbcr_stride),
  };
  uint32_t params2[4] = {
      static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(out_height),
      0u, 0u,
  };
  float params3[4] = {
      static_cast<float>(cc.crop_left),
      static_cast<float>(cc.crop_top),
      scale_x,
      scale_y,
  };
  const uint32_t use_bicubic =
      (scale_x > 2.0f || scale_y > 2.0f) ? 1u : 0u;

  const bool ok = dispatch_(mc, "nv12_to_planar_rgb_u8",
      "nv12_to_planar_rgb_u8", "nv12->rgb", session,
      [&](ComputeEncoder& enc, const ComputeFunction& fn) {
        enc.set_buffer(0, y_buf, 0);
        enc.set_buffer(1, cbcr_buf, 0);
        enc.set_mtl_buffer(2, dst_buf, 0);
        enc.set_constant_bytes(3, params,       sizeof(params));
        enc.set_constant_bytes(4, params2,      sizeof(params2));
        enc.set_constant_bytes(5, params3,      sizeof(params3));
        enc.set_constant_bytes(6, &use_bicubic, sizeof(use_bicubic));
        const Dims2D d = dims2d_(fn, out_width, out_height);
        enc.dispatch(d.grid, d.threadgroup);
        return true;
      });

  CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
  return ok;
}

}  // namespace

bool
nv12_to_planar_rgb_u8(
    MetalCompute& mc, void* cv_pixel_buffer,
    uint8_t* dst_bytes, size_t dst_capacity_bytes,
    int src_width, int src_height, int out_width, int out_height,
    const SessionContextIntf* session)
{
  if (!mc.valid() || !cv_pixel_buffer || !dst_bytes
      || src_width <= 0 || src_height <= 0
      || out_width <= 0 || out_height <= 0) {
    return false;
  }
  const size_t need =
      static_cast<size_t>(3) * out_width * out_height;
  if (dst_capacity_bytes < need) {
    return false;
  }
  SharedBuffer out_buf = mc.make_shared_buffer(need);
  if (out_buf.empty()) {
    if (session) {
      session->warn(fmt(
          "metal_compute::image_ops: make_shared_buffer(out, {}) failed",
          need));
    }
    return false;
  }
  const bool ok = dispatch_nv12_kernel_(
      mc, cv_pixel_buffer, out_buf.mtl_buffer(),
      src_width, src_height, out_width, out_height, session);
  if (ok) {
    std::memcpy(dst_bytes, out_buf.contents(), need);
  }
  return ok;
}

bool
nv12_to_planar_rgb_u8_shared(
    MetalCompute& mc, void* cv_pixel_buffer,
    const ExternalStorageHandle& dst,
    int src_width, int src_height, int out_width, int out_height,
    const SessionContextIntf* session)
{
  if (!mc.valid() || !cv_pixel_buffer
      || src_width <= 0 || src_height <= 0
      || out_width <= 0 || out_height <= 0) {
    return false;
  }
  auto* dst_buf = static_cast<MTL::Buffer*>(dst.mtl_buffer);
  if (!dst_buf) { return false; }
  const size_t need =
      static_cast<size_t>(3) * out_width * out_height;
  if (dst.byte_size < need) {
    return false;
  }
  return dispatch_nv12_kernel_(
      mc, cv_pixel_buffer, dst_buf,
      src_width, src_height, out_width, out_height, session);
}

bool
letterbox_planar_u8_to_f32_chw(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst,
    int out_w, int out_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    const SessionContextIntf* session)
{
  if (!mc.valid()
      || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
    return false;
  }
  auto* src_buf = static_cast<MTL::Buffer*>(src.mtl_buffer);
  auto* dst_buf = static_cast<MTL::Buffer*>(dst.mtl_buffer);
  if (!src_buf || !dst_buf) { return false; }
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  const size_t need_dst =
      static_cast<size_t>(3) * out_w * out_h * sizeof(float);
  if (src.byte_size < need_src || dst.byte_size < need_dst) {
    return false;
  }

  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = sx < sy ? sx : sy;
  const int new_w = static_cast<int>(std::round(scale * in_w));
  const int new_h = static_cast<int>(std::round(scale * in_h));
  const int pad_x = (out_w - new_w) / 2;
  const int pad_y = (out_h - new_h) / 2;
  const float inv = 1.0f / scale;

  if (out_scale) { *out_scale = scale; }
  if (out_pad_x) { *out_pad_x = pad_x; }
  if (out_pad_y) { *out_pad_y = pad_y; }

  uint32_t params[4] = {
      static_cast<uint32_t>(in_w),
      static_cast<uint32_t>(in_h),
      static_cast<uint32_t>(out_w),
      static_cast<uint32_t>(out_h),
  };
  uint32_t params2[4] = {
      static_cast<uint32_t>(new_w),
      static_cast<uint32_t>(new_h),
      static_cast<uint32_t>(pad_x),
      static_cast<uint32_t>(pad_y),
  };
  float params3[4] = { inv, 0.0f, 0.0f, 0.0f };

  return dispatch_(mc, "letterbox_planar_u8_to_f32_chw",
      "letterbox_planar_u8_to_f32_chw", "letterbox-f32", session,
      [&](ComputeEncoder& enc, const ComputeFunction& fn) {
        enc.set_mtl_buffer(0, src_buf, 0);
        enc.set_mtl_buffer(1, dst_buf, 0);
        enc.set_constant_bytes(2, params,  sizeof(params));
        enc.set_constant_bytes(3, params2, sizeof(params2));
        enc.set_constant_bytes(4, params3, sizeof(params3));
        const Dims2D d = dims2d_(fn, out_w, out_h);
        enc.dispatch(d.grid, d.threadgroup);
        return true;
      });
}

bool
letterbox_planar_u8_to_u8_chw(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst,
    int out_w, int out_h, const SessionContextIntf* session)
{
  if (!mc.valid()
      || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
    return false;
  }
  auto* src_buf = static_cast<MTL::Buffer*>(src.mtl_buffer);
  auto* dst_buf = static_cast<MTL::Buffer*>(dst.mtl_buffer);
  if (!src_buf || !dst_buf) { return false; }
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  const size_t need_dst = static_cast<size_t>(3) * out_w * out_h;
  if (src.byte_size < need_src || dst.byte_size < need_dst) {
    return false;
  }

  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = sx < sy ? sx : sy;
  const int new_w = static_cast<int>(std::round(scale * in_w));
  const int new_h = static_cast<int>(std::round(scale * in_h));
  const int pad_x = (out_w - new_w) / 2;
  const int pad_y = (out_h - new_h) / 2;
  const float inv = 1.0f / scale;

  uint32_t params[4] = {
      static_cast<uint32_t>(in_w), static_cast<uint32_t>(in_h),
      static_cast<uint32_t>(out_w), static_cast<uint32_t>(out_h),
  };
  uint32_t params2[4] = {
      static_cast<uint32_t>(new_w), static_cast<uint32_t>(new_h),
      static_cast<uint32_t>(pad_x), static_cast<uint32_t>(pad_y),
  };
  float params3[4] = { inv, 0.0f, 0.0f, 0.0f };

  return dispatch_(mc, "letterbox_planar_u8_to_u8_chw",
      "letterbox_planar_u8_to_u8_chw", "letterbox-u8", session,
      [&](ComputeEncoder& enc, const ComputeFunction& fn) {
        enc.set_mtl_buffer(0, src_buf, 0);
        enc.set_mtl_buffer(1, dst_buf, 0);
        enc.set_constant_bytes(2, params,  sizeof(params));
        enc.set_constant_bytes(3, params2, sizeof(params2));
        enc.set_constant_bytes(4, params3, sizeof(params3));
        const Dims2D d = dims2d_(fn, out_w, out_h);
        enc.dispatch(d.grid, d.threadgroup);
        return true;
      });
}

bool
letterbox_planar_u8_to_bgra_cvpixelbuffer(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, void* cv_pixel_buffer,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    const SessionContextIntf* session)
{
  if (!mc.valid() || in_w <= 0 || in_h <= 0 || !cv_pixel_buffer) {
    return false;
  }
  CVPixelBufferRef pb =
      static_cast<CVPixelBufferRef>(cv_pixel_buffer);
  if (CVPixelBufferGetPixelFormatType(pb)
      != kCVPixelFormatType_32BGRA) {
    return false;
  }
  const int out_w = static_cast<int>(CVPixelBufferGetWidth(pb));
  const int out_h = static_cast<int>(CVPixelBufferGetHeight(pb));
  if (out_w <= 0 || out_h <= 0) { return false; }

  auto* src_buf = static_cast<MTL::Buffer*>(src.mtl_buffer);
  if (!src_buf) { return false; }
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  if (src.byte_size < need_src) { return false; }

  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = sx < sy ? sx : sy;
  const int   new_w = static_cast<int>(std::round(scale * in_w));
  const int   new_h = static_cast<int>(std::round(scale * in_h));
  const int   pad_x = (out_w - new_w) / 2;
  const int   pad_y = (out_h - new_h) / 2;
  const float inv   = 1.0f / scale;

  if (out_scale) { *out_scale = scale; }
  if (out_pad_x) { *out_pad_x = pad_x; }
  if (out_pad_y) { *out_pad_y = pad_y; }

  CVReturn lk = CVPixelBufferLockBaseAddress(pb, 0);
  if (lk != kCVReturnSuccess) { return false; }
  auto* base =
      static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
  const size_t bpr = CVPixelBufferGetBytesPerRow(pb);
  if (!base || bpr < static_cast<size_t>(out_w) * 4u) {
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return false;
  }

  const size_t total_bytes = bpr * static_cast<size_t>(out_h);
  SharedBuffer dst_buf = mc.make_shared_buffer(total_bytes);
  if (dst_buf.empty()) {
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return false;
  }

  uint32_t params[4] = {
      static_cast<uint32_t>(in_w),
      static_cast<uint32_t>(in_h),
      static_cast<uint32_t>(out_w),
      static_cast<uint32_t>(out_h),
  };
  uint32_t params2[4] = {
      static_cast<uint32_t>(new_w),
      static_cast<uint32_t>(new_h),
      static_cast<uint32_t>(pad_x),
      static_cast<uint32_t>(pad_y),
  };
  float    params3[4] = { inv, 0.0f, 0.0f, 0.0f };
  uint32_t dst_bpr    = static_cast<uint32_t>(bpr);

  const bool ok = dispatch_(mc, "letterbox_planar_u8_to_bgra_u8",
      "letterbox_planar_u8_to_bgra_u8", "letterbox-bgra", session,
      [&](ComputeEncoder& enc, const ComputeFunction& fn) {
        enc.set_mtl_buffer(0, src_buf, 0);
        enc.set_buffer(1, dst_buf, 0);
        enc.set_constant_bytes(2, params,   sizeof(params));
        enc.set_constant_bytes(3, params2,  sizeof(params2));
        enc.set_constant_bytes(4, params3,  sizeof(params3));
        enc.set_constant_bytes(5, &dst_bpr, sizeof(dst_bpr));
        const Dims2D d = dims2d_(fn, out_w, out_h);
        enc.dispatch(d.grid, d.threadgroup);
        return true;
      });

  if (ok) {
    std::memcpy(base, dst_buf.contents(), total_bytes);
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);
  return ok;
}

bool
motion_signature_u8(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst_sig,
    int tile_w, int tile_h, const SessionContextIntf* session)
{
  if (!mc.valid()
      || in_w <= 0 || in_h <= 0
      || tile_w <= 0 || tile_h <= 0
      || tile_w > in_w || tile_h > in_h) {
    return false;
  }
  auto* src_buf = static_cast<MTL::Buffer*>(src.mtl_buffer);
  auto* dst_buf = static_cast<MTL::Buffer*>(dst_sig.mtl_buffer);
  if (!src_buf || !dst_buf) { return false; }
  const size_t need_src = static_cast<size_t>(3) * in_w * in_h;
  const size_t need_dst =
      static_cast<size_t>(tile_w) * static_cast<size_t>(tile_h);
  if (src.byte_size < need_src || dst_sig.byte_size < need_dst) {
    return false;
  }

  uint32_t params[4] = {
      static_cast<uint32_t>(in_w),
      static_cast<uint32_t>(in_h),
      static_cast<uint32_t>(tile_w),
      static_cast<uint32_t>(tile_h),
  };

  return dispatch_(mc, "motion_signature_u8",
      "motion_signature_u8", "motion-signature", session,
      [&](ComputeEncoder& enc, const ComputeFunction& fn) {
        enc.set_mtl_buffer(0, src_buf, 0);
        enc.set_mtl_buffer(1, dst_buf, 0);
        enc.set_constant_bytes(2, params, sizeof(params));
        const Dims2D d = dims2d_(fn, tile_w, tile_h);
        enc.dispatch(d.grid, d.threadgroup);
        return true;
      });
}

bool
motion_diff_u8(
    MetalCompute& mc, const ExternalStorageHandle& sig_a,
    const ExternalStorageHandle& sig_b, const ExternalStorageHandle& dst,
    int tile_w, int tile_h, const SessionContextIntf* session)
{
  if (!mc.valid() || tile_w <= 0 || tile_h <= 0) {
    return false;
  }
  auto* a_buf = static_cast<MTL::Buffer*>(sig_a.mtl_buffer);
  auto* b_buf = static_cast<MTL::Buffer*>(sig_b.mtl_buffer);
  auto* d_buf = static_cast<MTL::Buffer*>(dst.mtl_buffer);
  if (!a_buf || !b_buf || !d_buf) { return false; }
  const size_t need =
      static_cast<size_t>(tile_w) * static_cast<size_t>(tile_h);
  if (sig_a.byte_size < need || sig_b.byte_size < need
      || dst.byte_size < need) {
    return false;
  }

  uint32_t params[2] = {
      static_cast<uint32_t>(tile_w),
      static_cast<uint32_t>(tile_h),
  };

  return dispatch_(mc, "motion_diff_u8",
      "motion_diff_u8", "motion-diff", session,
      [&](ComputeEncoder& enc, const ComputeFunction& fn) {
        enc.set_mtl_buffer(0, a_buf, 0);
        enc.set_mtl_buffer(1, b_buf, 0);
        enc.set_mtl_buffer(2, d_buf, 0);
        enc.set_constant_bytes(3, params, sizeof(params));
        const Dims2D d = dims2d_(fn, tile_w, tile_h);
        enc.dispatch(d.grid, d.threadgroup);
        return true;
      });
}

std::unique_ptr<ExternalStorageHandle>
make_shared_storage(MetalCompute& mc, size_t byte_size,
                    const SessionContextIntf* session)
{
  if (!mc.valid() || byte_size == 0) {
    return nullptr;
  }
  SharedBuffer sb = mc.make_shared_buffer(byte_size);
  if (sb.empty()) {
    if (session) {
      session->warn(fmt(
          "metal_compute::image_ops: make_shared_storage({}) -- "
          "make_shared_buffer failed", byte_size));
    }
    return nullptr;
  }
  // Transfer ownership from SharedBuffer to ExternalStorageHandle.
  // Pre-retain so sb's destructor (-1) is balanced and the handle holds
  // the surviving +1 refcount (same pattern as to_tensor_beat()).
  auto* buf = sb.mtl_buffer();
  void* contents = sb.contents();
  buf->retain();

  auto h = std::make_unique<ExternalStorageHandle>();
  h->mtl_buffer = buf;
  h->contents   = static_cast<uint8_t*>(contents);
  h->byte_size  = byte_size;
  h->deleter    = &release_shared_buffer_;
  return h;
}

}  // namespace vpipe::metal_compute
