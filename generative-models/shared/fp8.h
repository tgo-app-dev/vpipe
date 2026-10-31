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
//
// INT8 rides the same machinery as a third format, kI8: ComfyUI's
// `int8_tensorwise` (a comfy_quant record) stores signed codes with one
// scale or one per row, value = code * scale. A code is its own small
// integer, exact in bf16. It may also be ROTATED (ConvRot, the record's
// `convrot`): the codes then quantize W * H^T, H a regular Hadamard over
// groups of the input columns, undone on the codes by convrot_unrotate
// below. kI8 is never a dtype -- an I8 tensor is int8 only where a record
// says so.
//
// W4A8 (kW4A8; ComfyUI's `asym_w4a8_int8`) is int8 STORED at four bits: a
// row's codes are nibbles, two a byte (the even column low), each the
// index of a 16-entry codebook (or, with none, code - 8); a group of
// `group_size` (16) columns carries an E4M3 relative scale, and
// round(codebook[code] * s_rel), clamped to +-127, IS an int8 code --
// rotated (ConvRot) and scaled by its row (`s_channel`) exactly as kI8's.
// Not elementwise: the code tables below do not apply to it
// (fp8-layout.h's decode_w4a8_bf16 does).


#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe::genai::fp8 {

enum class Format { kNone, kE4M3, kE5M2, kI8, kW4A8 };

// The FP8 format a safetensors dtype string names; kNone for any other
// dtype (I8 among them: see kI8 above).
inline Format
format_of(std::string_view safetensors_dtype)
{
  if (safetensors_dtype == "F8_E4M3") { return Format::kE4M3; }
  if (safetensors_dtype == "F8_E5M2") { return Format::kE5M2; }
  return Format::kNone;
}

// One code, as the float it stands for. NaN codes decode to a quiet NaN,
// E5M2's infinities to infinities; an int8 code to its integer.
inline float
to_f32(std::uint8_t code, Format f)
{
  if (f == Format::kI8) { return (float)(std::int8_t)code; }
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
  static const std::array<std::uint16_t, 256> kI8 =
      detail::make_lut_(Format::kI8);
  return f == Format::kE5M2 ? kE5M2 : f == Format::kI8 ? kI8 : kE4M3;
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
  static const std::array<float, 256> kI8 = make(Format::kI8);
  return f == Format::kE5M2 ? kE5M2 : f == Format::kI8 ? kI8 : kE4M3;
}

// Whether `g` is a ConvRot group: a power of four, 4 to 1024 (what
// comfy_kitchen builds, and what one GPU threadgroup holds).
inline bool
convrot_group_ok(int g)
{
  return g >= 4 && g <= 1024 && (g & (g - 1)) == 0 &&
         (__builtin_ctz((unsigned)g) % 2) == 0;
}

// ConvRot undone in place ON THE CODES: each group of `g` integer codes
// of `v` (n of them, a multiple of g) multiplied by H^T * sqrt(g), H the
// normalized REGULAR Hadamard comfy_kitchen rotates with -- kron powers
// of
//   [[ 1, 1, 1,-1], [ 1, 1,-1, 1], [ 1,-1, 1, 1], [-1, 1, 1, 1]] / 2.
// H is symmetric and orthogonal, so the rotation is its own inverse. Each
// base-4 digit of an index is one radix-4 stage: with s the four values
// that differ in that digit summed, the d-th becomes s - 2 * the
// (3 - d)-th. A row's scale is common to its groups, so it waits until
// after: every value stays an integer of magnitude at most 128 * 1024 <
// 2^24, so the sums are EXACT, in any order and even in f32, and the
// weight is v * scale * convrot_norm(g), a product rounded once -- which
// the GPU kernel (affine_dequant's i8_convrot_dequant) computes alike.
inline void
convrot_unrotate(std::int32_t* v, std::size_t n, int g)
{
  for (std::size_t g0 = 0; g0 + (std::size_t)g <= n; g0 += (std::size_t)g) {
    std::int32_t* x = v + g0;
    for (int st = 1; st < g; st *= 4) {
      for (int q = 0; q < g / 4; ++q) {
        const int base = (q / st) * st * 4 + q % st;
        const std::int32_t a0 = x[base], a1 = x[base + st];
        const std::int32_t a2 = x[base + 2 * st], a3 = x[base + 3 * st];
        const std::int32_t s = a0 + a1 + a2 + a3;
        x[base] = s - 2 * a3;
        x[base + st] = s - 2 * a2;
        x[base + 2 * st] = s - 2 * a1;
        x[base + 3 * st] = s - 2 * a0;
      }
    }
  }
}

// 1 / sqrt(g) for a ConvRot group, exact: g is a power of four.
inline float
convrot_norm(int g)
{
  return 1.0f / (float)(1 << (__builtin_ctz((unsigned)g) / 2));
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
