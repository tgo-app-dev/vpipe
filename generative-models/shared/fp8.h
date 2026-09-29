#ifndef VPIPE_GENERATIVE_MODELS_SHARED_FP8_H
#define VPIPE_GENERATIVE_MODELS_SHARED_FP8_H

// fp8.h -- decoding the two 8-bit float formats a safetensors header can
// name, "F8_E4M3" and "F8_E5M2".
//
//   F8_E4M3   the OCP "E4M3FN" encoding: bias 7, subnormals, NO
//             infinities, and S.1111.111 as the only NaN -- which is what
//             buys it a top value of 448 rather than 240.
//   F8_E5M2   IEEE-shaped: bias 15, subnormals, infinities and NaNs. It is
//             exactly the top byte of an IEEE half.
//
// BOTH DECODE EXACTLY INTO BF16. E4M3's smallest subnormal is 2^-9 and its
// mantissa is 3 bits; E5M2's are 2^-16 and 2 bits -- bf16 has an 8-bit
// exponent and a 7-bit mantissa, so every finite code of either format is
// a bf16 value. Widening an FP8 checkpoint to bf16 therefore loses
// nothing, which is what makes it a DEQUANTIZATION rather than one more
// rounding: the bf16 checkpoint computes exactly what the FP8 one does.
//
// This is the CODE alone. "Scaled" FP8, where a tensor carries a
// multiplier beside it (`<name>.scale_weight`, `<name>.weight_scale`,
// ComfyUI's `scaled_fp8` marker), means code * scale, and which scale
// goes with which tensor is a matter of names -- shared/fp8-layout.h.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

namespace vpipe::genai::fp8 {

enum class Format { kNone, kE4M3, kE5M2 };

// The FP8 format a safetensors dtype string names; kNone for any other
// dtype.
inline Format
format_of(std::string_view safetensors_dtype)
{
  if (safetensors_dtype == "F8_E4M3") { return Format::kE4M3; }
  if (safetensors_dtype == "F8_E5M2") { return Format::kE5M2; }
  return Format::kNone;
}

// One code, as the float it stands for. NaN codes decode to a quiet NaN,
// E5M2's infinities to infinities.
inline float
to_f32(std::uint8_t code, Format f)
{
  const float sign = (code & 0x80u) ? -1.0f : 1.0f;
  if (f == Format::kE4M3) {
    const int e = (code >> 3) & 0xF;
    const int m = code & 0x7;
    if (e == 0xF && m == 0x7) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    // Subnormal: m/8 * 2^(1-7). Normal: (1 + m/8) * 2^(e-7).
    const float v = e == 0 ? (float)m * 0x1p-9f
                           : (1.0f + (float)m / 8.0f) *
                                 (float)(1u << e) * 0x1p-7f;
    return sign * v;
  }
  if (f == Format::kE5M2) {
    const int e = (code >> 2) & 0x1F;
    const int m = code & 0x3;
    if (e == 0x1F) {
      return m == 0 ? sign * std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::quiet_NaN();
    }
    const float v = e == 0 ? (float)m * 0x1p-16f
                           : (1.0f + (float)m / 4.0f) *
                                 (float)(1u << e) * 0x1p-15f;
    return sign * v;
  }
  return 0.0f;
}

namespace detail {

// A float that is known to be a bf16 value, as its bf16 bits. The low 16
// bits are zero for every finite FP8 value (see the header), so this is a
// truncation that drops nothing; NaN keeps a mantissa bit set.
inline std::uint16_t
bf16_bits_(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  if (v != v) { return (std::uint16_t)((u >> 16) | 0x0040u); }
  return (std::uint16_t)(u >> 16);
}

inline std::array<std::uint16_t, 256>
make_lut_(Format f)
{
  std::array<std::uint16_t, 256> t{};
  for (int c = 0; c < 256; ++c) {
    t[(std::size_t)c] = bf16_bits_(to_f32((std::uint8_t)c, f));
  }
  return t;
}

}  // namespace detail

// The 256-entry code -> bf16 table for a format. Built once.
inline const std::array<std::uint16_t, 256>&
bf16_table(Format f)
{
  static const std::array<std::uint16_t, 256> kE4M3 =
      detail::make_lut_(Format::kE4M3);
  static const std::array<std::uint16_t, 256> kE5M2 =
      detail::make_lut_(Format::kE5M2);
  return f == Format::kE5M2 ? kE5M2 : kE4M3;
}

// The 256-entry code -> f32 table for a format. Built once.
inline const std::array<float, 256>&
f32_table(Format f)
{
  auto make = [](Format fm) {
    std::array<float, 256> t{};
    for (int c = 0; c < 256; ++c) {
      t[(std::size_t)c] = to_f32((std::uint8_t)c, fm);
    }
    return t;
  };
  static const std::array<float, 256> kE4M3 = make(Format::kE4M3);
  static const std::array<float, 256> kE5M2 = make(Format::kE5M2);
  return f == Format::kE5M2 ? kE5M2 : kE4M3;
}

// Decode `n` codes into bf16. `f` must not be kNone.
inline void
decode_bf16(const std::uint8_t* src, std::uint16_t* dst, std::size_t n,
            Format f)
{
  const std::array<std::uint16_t, 256>& t = bf16_table(f);
  for (std::size_t i = 0; i < n; ++i) { dst[i] = t[src[i]]; }
}

}  // namespace vpipe::genai::fp8

#endif
