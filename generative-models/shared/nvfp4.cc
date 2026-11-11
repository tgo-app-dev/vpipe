#include "generative-models/shared/nvfp4.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace vpipe::genai::nvfp4 {

namespace {

constexpr float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                             4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                             -2.0f, -3.0f, -4.0f, -6.0f};

}  // namespace

float
e4m3_to_float(std::uint8_t b)
{
  const int e = (b >> 3) & 0xF, m = b & 7;
  if (e == 0xF && m == 7) { return std::numeric_limits<float>::quiet_NaN(); }
  // Normal: (1 + m/8) 2^(e-7) = (8 + m) 2^(e-10); subnormal: m 2^-9.
  const float v = e == 0 ? std::ldexp((float)m, -9)
                         : std::ldexp((float)(8 + m), e - 10);
  return (b & 0x80) != 0 ? -v : v;
}

float
e2m1_value(std::uint8_t code)
{
  return kE2M1[code & 0xF];
}

bool
pack(const std::uint8_t* codes, const std::uint8_t* scales, float scale_2,
     int N, int K, std::uint8_t* dst)
{
  if (N <= 0 || K <= 0 || K % kSuper != 0) { return false; }
  const int supers = K / kSuper;
  for (int n = 0; n < N; ++n) {
    const std::uint8_t* crow = codes + (std::size_t)n * (K / 2);
    const std::uint8_t* srow = scales + (std::size_t)n * (K / kBlock);
    for (int s = 0; s < supers; ++s) {
      std::uint8_t* sb =
          dst + ((std::size_t)n * supers + s) * kSuperBytes;
      std::memcpy(sb, &scale_2, 4);
      std::memcpy(sb + 4, srow + (std::size_t)s * (kSuper / kBlock),
                  kSuper / kBlock);
      std::memcpy(sb + 20, crow + (std::size_t)s * (kSuper / 2),
                  kSuper / 2);
    }
  }
  return true;
}

float
served_value(const std::uint8_t* sb, int pos)
{
  float g;
  std::memcpy(&g, sb, 4);
  const std::uint8_t byte = sb[20 + pos / 2];
  const std::uint8_t code = (pos & 1) != 0 ? byte >> 4 : byte & 0xF;
  return e2m1_value(code) * (e4m3_to_float(sb[4 + pos / kBlock]) * g);
}

void
dequantize(const std::uint8_t* codes, const std::uint8_t* scales,
           float scale_2, int N, int K, float* out)
{
  for (int n = 0; n < N; ++n) {
    const std::uint8_t* crow = codes + (std::size_t)n * (K / 2);
    const std::uint8_t* srow = scales + (std::size_t)n * (K / kBlock);
    float* orow = out + (std::size_t)n * K;
    for (int k = 0; k < K; ++k) {
      const std::uint8_t byte = crow[k / 2];
      const std::uint8_t code = (k & 1) != 0 ? byte >> 4 : byte & 0xF;
      orow[k] = e2m1_value(code) * (e4m3_to_float(srow[k / kBlock]) * scale_2);
    }
  }
}

}  // namespace vpipe::genai::nvfp4
