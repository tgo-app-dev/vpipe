#ifndef VPIPE_GENERATIVE_MODELS_SHARED_NVFP4_H
#define VPIPE_GENERATIVE_MODELS_SHARED_NVFP4_H

// NVFP4 -- NVIDIA's 4-bit float weights as ModelOpt exports them
// (`quantization_config.quant_method` "modelopt", `quant_algo` "NVFP4").
//
// A quantized linear `<base>` is THREE tensors (a fourth, `input_scale`,
// is the activation scale of Blackwell's FP4 x FP4 path; a weight-only
// run ignores it):
//   <base>.weight         U8       [N, K/2]   two E2M1 codes a byte: the
//                                             LOW nibble element 2i, the
//                                             HIGH nibble 2i+1
//   <base>.weight_scale   F8_E4M3  [N, K/16]  a scale per 16 inputs
//   <base>.weight_scale_2 F32      []         the tensor's own scale
// and its weight is, as ModelOpt's NVFP4QTensor.dequantize computes it,
//   w[n, k] = e2m1[code(n, k)] * (float(scale[n, k/16]) * scale_2)
// in f32, e2m1 = {0, .5, 1, 1.5, 2, 3, 4, 6} with bit 3 the sign.
//
// Run natively it is SERVED as one buffer of super-blocks of 256 inputs
// of a row, 148 bytes each (4.625 bits a weight against the file's 4.5):
//   +0   f32       the tensor's scale_2 (repeated: a row of a fused q|k
//                  keeps its own tensor's)
//   +4   u8 [16]   the 16 block scales, E4M3
//   +20  u8 [128]  the codes, as the file packs them
// rows back to back -- the k-quant (GGUF) route's shape, so the Qwen
// stack's qmv / dequant-and-GEMM machinery takes it as another type.
// K must be a multiple of 256.

#include <cstddef>
#include <cstdint>

namespace vpipe::genai::nvfp4 {

inline constexpr int kBlock = 16;              // inputs a block scale covers
inline constexpr int kSuper = 256;             // inputs a served super-block
inline constexpr int kSuperBytes = 4 + 16 + 128;

// E4M3 (float8_e4m3fn: bias 7, no infinity) to f32; 0x7F / 0xFF -> NaN.
float e4m3_to_float(std::uint8_t b);
// The E2M1 code's value (bit 3 the sign).
float e2m1_value(std::uint8_t code);

// Bytes the served form of an [N, K] weight takes.
inline std::size_t
served_bytes(int N, int K)
{
  return (std::size_t)N * (std::size_t)(K / kSuper) * kSuperBytes;
}

// Pack an [N, K] weight's file tensors into the served super-blocks at
// `dst` (served_bytes(N, K)). False when K is not a multiple of 256.
bool pack(const std::uint8_t* codes, const std::uint8_t* scales,
          float scale_2, int N, int K, std::uint8_t* dst);

// The value at input `pos` (0..255) of a served super-block -- the
// reference the kernels are held to.
float served_value(const std::uint8_t* super_block, int pos);

// The file tensors' weight [N, K] in f32, as ModelOpt dequantizes it.
void dequantize(const std::uint8_t* codes, const std::uint8_t* scales,
                float scale_2, int N, int K, float* out);

}  // namespace vpipe::genai::nvfp4

#endif
