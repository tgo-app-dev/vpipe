#include "apple-silicon/media/planar.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace vpipe::apple_media {

namespace {

using half = _Float16;

float
clamp01(float v)
{
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// One planar sample as a float in the picture's own units (U8 / 255).
struct PlanarReader {
  const std::uint8_t* base = nullptr;
  TensorBeat::DType dtype = TensorBeat::DType::U8;
  std::int64_t sc = 0, sy = 0, sx = 0, off = 0;

  float at(int c, int y, int x) const
  {
    const std::int64_t i = off + c * sc + y * sy + x * sx;
    if (dtype == TensorBeat::DType::U8) { return base[i] / 255.0f; }
    if (dtype == TensorBeat::DType::F32) {
      return reinterpret_cast<const float*>(base)[i];
    }
    return (float)reinterpret_cast<const half*>(base)[i];
  }
};

}  // namespace

std::size_t
bytes_per_pixel(Interleaved f)
{
  switch (f) {
  case Interleaved::RGBA8:
  case Interleaved::BGRA8:    return 4;
  case Interleaved::RGBA16:
  case Interleaved::RGBAHalf: return 8;
  }
  return 4;
}

bool
planar_to_interleaved(const TensorBeat& src, Interleaved fmt, void* dst,
                      std::size_t row_bytes, bool premultiply,
                      std::string* err, bool opaque)
{
  if (src.shape.size() != 3 || (src.shape[0] != 3 && src.shape[0] != 4) ||
      (src.dtype != TensorBeat::DType::U8 &&
       src.dtype != TensorBeat::DType::F16 &&
       src.dtype != TensorBeat::DType::F32)) {
    if (err) {
      *err = "a picture is a U8, F16 or F32 TensorBeat [3|4,H,W]";
    }
    return false;
  }
  const int C = (int)src.shape[0];
  const int H = (int)src.shape[1];
  const int W = (int)src.shape[2];
  PlanarReader r;
  r.base = src.bytes_();
  r.dtype = src.dtype;
  r.off = src.storage_offset;
  if (src.strides.size() == 3) {
    r.sc = src.strides[0];
    r.sy = src.strides[1];
    r.sx = src.strides[2];
  } else {
    r.sc = (std::int64_t)H * W;
    r.sy = W;
    r.sx = 1;
  }
  auto* out = static_cast<std::uint8_t*>(dst);
  for (int y = 0; y < H; ++y) {
    std::uint8_t* row = out + (std::size_t)y * row_bytes;
    for (int x = 0; x < W; ++x) {
      float px[4] = {r.at(0, y, x), r.at(1, y, x), r.at(2, y, x),
                     C == 4 && !opaque ? r.at(3, y, x) : 1.0f};
      if (premultiply && C == 4) {
        const float a = clamp01(px[3]);
        px[0] *= a;
        px[1] *= a;
        px[2] *= a;
      }
      switch (fmt) {
      case Interleaved::RGBA8:
      case Interleaved::BGRA8: {
        std::uint8_t* p = row + (std::size_t)x * 4;
        const int order[4] = {fmt == Interleaved::BGRA8 ? 2 : 0, 1,
                              fmt == Interleaved::BGRA8 ? 0 : 2, 3};
        for (int k = 0; k < 4; ++k) {
          p[k] = (std::uint8_t)std::lround(clamp01(px[order[k]]) * 255.0f);
        }
        break;
      }
      case Interleaved::RGBA16: {
        auto* p = reinterpret_cast<std::uint16_t*>(row) + (std::size_t)x * 4;
        for (int k = 0; k < 4; ++k) {
          p[k] = (std::uint16_t)std::lround(clamp01(px[k]) * 65535.0f);
        }
        break;
      }
      case Interleaved::RGBAHalf: {
        auto* p = reinterpret_cast<half*>(row) + (std::size_t)x * 4;
        for (int k = 0; k < 4; ++k) { p[k] = (half)px[k]; }
        break;
      }
      }
    }
  }
  return true;
}

bool
interleaved_to_planar(const void* src, std::size_t row_bytes, int width,
                      int height, Interleaved fmt, int channels,
                      TensorBeat::DType dtype, bool unpremultiply,
                      TensorBeat* out, std::string* err)
{
  if ((channels != 3 && channels != 4) ||
      (dtype != TensorBeat::DType::U8 && dtype != TensorBeat::DType::F16 &&
       dtype != TensorBeat::DType::F32) ||
      width <= 0 || height <= 0) {
    if (err) { *err = "interleaved_to_planar: bad geometry or dtype"; }
    return false;
  }
  const std::size_t plane = (std::size_t)width * height;
  out->dtype = dtype;
  out->shape = {channels, height, width};
  out->strides.clear();
  out->storage_offset = 0;
  out->data.assign(plane * channels * TensorBeat::byte_size_of(dtype), 0);
  const auto* in = static_cast<const std::uint8_t*>(src);
  std::uint8_t* d8 = out->data.data();
  auto* d16 = reinterpret_cast<half*>(out->data.data());
  auto* d32 = reinterpret_cast<float*>(out->data.data());
  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = in + (std::size_t)y * row_bytes;
    for (int x = 0; x < width; ++x) {
      float px[4];
      switch (fmt) {
      case Interleaved::RGBA8:
      case Interleaved::BGRA8: {
        const std::uint8_t* p = row + (std::size_t)x * 4;
        const bool bgra = fmt == Interleaved::BGRA8;
        px[0] = p[bgra ? 2 : 0] / 255.0f;
        px[1] = p[1] / 255.0f;
        px[2] = p[bgra ? 0 : 2] / 255.0f;
        px[3] = p[3] / 255.0f;
        break;
      }
      case Interleaved::RGBA16: {
        const auto* p =
            reinterpret_cast<const std::uint16_t*>(row) + (std::size_t)x * 4;
        for (int k = 0; k < 4; ++k) { px[k] = p[k] / 65535.0f; }
        break;
      }
      case Interleaved::RGBAHalf: {
        const auto* p =
            reinterpret_cast<const half*>(row) + (std::size_t)x * 4;
        for (int k = 0; k < 4; ++k) { px[k] = (float)p[k]; }
        break;
      }
      }
      if (unpremultiply && px[3] > 0.0f && px[3] < 1.0f) {
        px[0] /= px[3];
        px[1] /= px[3];
        px[2] /= px[3];
      }
      const std::size_t i = (std::size_t)y * width + x;
      for (int c = 0; c < channels; ++c) {
        if (dtype == TensorBeat::DType::U8) {
          d8[(std::size_t)c * plane + i] =
              (std::uint8_t)std::lround(clamp01(px[c]) * 255.0f);
        } else if (dtype == TensorBeat::DType::F32) {
          d32[(std::size_t)c * plane + i] = px[c];
        } else {
          d16[(std::size_t)c * plane + i] = (half)px[c];
        }
      }
    }
  }
  return true;
}

}  // namespace vpipe::apple_media
