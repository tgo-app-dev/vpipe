#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_IMAGE_OPS_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_IMAGE_OPS_H

// GPU image-preprocessing ops on the metal-compute backend. These were
// formerly MetalRuntime methods; MetalRuntime has been retired and its
// (already metal-compute-backed) operations live here as free functions
// in vpipe::metal_compute. Each takes a valid MetalCompute& and an
// optional SessionContextIntf* (for warning logs), dispatches the
// matching metal-compute kernel, and returns false on failure.

#include "apple-silicon/tensor-storage.h"   // ExternalStorageHandle

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vpipe { class SessionContextIntf; }

namespace vpipe::metal_compute {

class MetalCompute;

// NV12 CVPixelBuffer -> planar [3,out_h,out_w] RGB u8, center-cropped +
// bilinear-rescaled, written into a host byte buffer (`dst_bytes`).
bool nv12_to_planar_rgb_u8(
    MetalCompute& mc, void* cv_pixel_buffer,
    std::uint8_t* dst_bytes, std::size_t dst_capacity_bytes,
    int src_width, int src_height, int out_width, int out_height,
    const SessionContextIntf* session);

// Same as above but writes directly into a Shared MTL::Buffer (the
// destination TensorBeat's ExternalStorageHandle) -- no host readback.
bool nv12_to_planar_rgb_u8_shared(
    MetalCompute& mc, void* cv_pixel_buffer,
    const ExternalStorageHandle& dst,
    int src_width, int src_height, int out_width, int out_height,
    const SessionContextIntf* session);

// Bilinear letterbox planar u8 RGB [3,in_h,in_w] -> f32 CHW
// [3,out_h,out_w] normalised to [0,1], aspect-preserving with 114/255
// grey pad. Reports the inverse-mapping scale + pad for unprojection.
bool letterbox_planar_u8_to_f32_chw(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst,
    int out_w, int out_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    const SessionContextIntf* session);

// Bilinear letterbox planar u8 RGB [3,in_h,in_w] -> planar u8 RGB
// [3,out_h,out_w], aspect-preserving with 114 grey pad. Like the f32
// variant but the output stays 8-bit, for feeding a vision encoder that
// itself takes planar u8 RGB (the GPU VLM-input resampler).
bool letterbox_planar_u8_to_u8_chw(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst,
    int out_w, int out_h, const SessionContextIntf* session);

// Destination-rectangle + source-mapping parameters for the resample
// kernel (see compute_resample_geom). The dest rectangle
// [pad_x, pad_x+new_w) x [pad_y, pad_y+new_h) samples src at
// (dst - pad) * inv + src0; outside it is the pad colour.
struct ResampleGeom {
  int   new_w = 0, new_h = 0, pad_x = 0, pad_y = 0;
  float inv_x = 1.0f, inv_y = 1.0f, src_x0 = 0.0f, src_y0 = 0.0f;
};

// Compute the resample geometry for a fit mode (0 pad / 1 crop / 2 stretch
// / 3 manual). `src_x/src_y/scale` are used only by manual. Pure; shared by
// the GPU path (below) and a CPU fallback.
ResampleGeom compute_resample_geom(int in_w, int in_h, int out_w, int out_h,
                                   int mode, int src_x, int src_y,
                                   float scale);

// Bilinear resample of planar u8 RGB [3,in_h,in_w] -> planar u8 RGB
// [3,out_h,out_w] with the fit mode + solid pad colour above. Reuses the
// (generalised) letterbox kernel.
bool resample_planar_u8_to_u8(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst,
    int out_w, int out_h, int mode, int src_x, int src_y, float scale,
    std::uint8_t pad_r, std::uint8_t pad_g, std::uint8_t pad_b,
    const SessionContextIntf* session);

// Bilinear letterbox + RGB->BGRA pack into a kCVPixelFormatType_32BGRA
// CVPixelBuffer (destination dims read off the pixel buffer).
bool letterbox_planar_u8_to_bgra_cvpixelbuffer(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, void* cv_pixel_buffer,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    const SessionContextIntf* session);

// Bilinear letterbox + RGB->BGRA8888 pack into a Shared byte buffer
// `dst` laid out tightly (out_w*4 bytes/row, size out_w*4*out_h). Same
// pixel math as the CVPixelBuffer variant but CoreVideo-free: the bytes
// are handed to CoreMLLoadedModel::predict() as an image input, which
// owns the CVPixelBuffer. Reports the applied scale + pad.
bool letterbox_planar_u8_to_bgra_u8(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst,
    int out_w, int out_h,
    float* out_scale, int* out_pad_x, int* out_pad_y,
    const SessionContextIntf* session);

// Downsampled BT.601-luma signature [tile_h,tile_w] u8 from planar u8
// RGB [3,in_h,in_w] (paired with motion_diff_u8 for frame decimation).
bool motion_signature_u8(
    MetalCompute& mc, const ExternalStorageHandle& src,
    int in_w, int in_h, const ExternalStorageHandle& dst_sig,
    int tile_w, int tile_h, const SessionContextIntf* session);

// Per-tile |sig_a - sig_b| u8 [tile_h,tile_w].
bool motion_diff_u8(
    MetalCompute& mc, const ExternalStorageHandle& sig_a,
    const ExternalStorageHandle& sig_b, const ExternalStorageHandle& dst,
    int tile_w, int tile_h, const SessionContextIntf* session);

// Allocate a Shared MTL::Buffer of `byte_size` wrapped in an
// ExternalStorageHandle (ready for TensorBeat::external). nullptr on
// failure. The handle owns the buffer and releases it on destruction.
std::unique_ptr<ExternalStorageHandle> make_shared_storage(
    MetalCompute& mc, std::size_t byte_size,
    const SessionContextIntf* session);

}  // namespace vpipe::metal_compute

#endif
