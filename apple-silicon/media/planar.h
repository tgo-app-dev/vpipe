#ifndef VPIPE_APPLE_MEDIA_PLANAR_H
#define VPIPE_APPLE_MEDIA_PLANAR_H

#include "apple-silicon/tensor-beat.h"

#include <cstddef>
#include <string>

namespace vpipe::apple_media {

// The interleaved layouts the Apple frameworks read and write, four
// channels each: RGBA 8/16-bit (Core Graphics, ImageIO), RGBA half
// (kCVPixelFormatType_64RGBAHalf, a float CGImage), BGRA 8-bit
// (kCVPixelFormatType_32BGRA).
enum class Interleaved { RGBA8, RGBA16, RGBAHalf, BGRA8 };

std::size_t bytes_per_pixel(Interleaved);

// Picture samples are U8 (0..255), F16 or F32 (0..1, a linear picture
// above 1) -- F32 as video-to-rgb and image-resample have it.
//
// A planar picture [C,H,W] (U8, F16 or F32, C 3 or 4, any strides) into
// interleaved rows of `row_bytes`. A 3-channel picture gets opaque
// alpha, and so does any with `opaque`. Integer layouts clamp to 0..1;
// half keeps what F16 holds (a linear picture above 1). `premultiply`:
// colour times alpha.
bool planar_to_interleaved(const TensorBeat& src, Interleaved fmt,
                           void* dst, std::size_t row_bytes,
                           bool premultiply, std::string* err,
                           bool opaque = false);

// Interleaved rows into a new planar picture of `dtype` (U8, F16, F32)
// with `channels` (3 drops alpha, 4 keeps it). `unpremultiply`: the
// source colour is premultiplied and the picture is to be straight.
bool interleaved_to_planar(const void* src, std::size_t row_bytes,
                           int width, int height, Interleaved fmt,
                           int channels, TensorBeat::DType dtype,
                           bool unpremultiply, TensorBeat* out,
                           std::string* err);

}  // namespace vpipe::apple_media

#endif
