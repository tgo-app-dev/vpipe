// ggml's block quantization formats, read on the CPU: every format the
// llama.cpp GGUFs of Qwen3.8-Flash-Next and DeepSeek-V4.1 are written in
// (the k-quants, the block-32 legacy quants, the IQ codebooks), decoded
// to f32 as ggml's dequantize_row_* does, and -- for the formats whose
// values lie on an affine grid a group of 32 at a time -- re-expressed
// EXACTLY as affine codes, so a block-format weight can run on the affine
// kernels with nothing rounded but each group's scale and bias as the
// model stores them.
#ifndef VPIPE_GENERATIVE_MODELS_GGML_BLOCKS_H
#define VPIPE_GENERATIVE_MODELS_GGML_BLOCKS_H

#include <cstdint>
#include <vector>

namespace vpipe::genai::ggml {

// ggml_type values.
enum Type : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ4_0 = 2,
  kQ5_0 = 6,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ3_K = 11,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kIQ2_XXS = 16,
  kIQ2_XS = 17,
  kIQ3_XXS = 18,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ2_S = 22,
  kIQ4_XS = 23,
  kIQ1_M = 29,
  kBF16 = 30,
};

// A block: `elems` values in `bytes` bytes. {0, 0} for a type this file
// does not know.
struct Geom {
  int elems = 0;
  int bytes = 0;
};
Geom geom(std::uint32_t type);
const char* name(std::uint32_t type);

// `n` values (whole blocks) of `type` at `src` into f32, as ggml's
// dequantize_row_* computes them. False for a type this file does not
// decode, or `n` not a whole number of blocks.
bool decode(std::uint32_t type, const void* src, std::int64_t n,
            float* out);

// AFFINE AT GROUP 32. A group of 32 values of a block format, as integer
// codes and one (scale, bias): value = scale * code + bias, which in f32
// is the value ggml computes. `off` is nonzero where the format is
// symmetric (bias = -off * scale): write the bias from the scale AS
// STORED then, so a rounded scale moves every value of the group by the
// same relative amount and none by an offset.
struct Affine32 {
  float scale = 0.0f;
  float bias = 0.0f;
  float off = 0.0f;
};
// The least code width a kernel runs that holds `type` exactly, 0 where
// no group-32 affine form does (Q6_K, IQ2_XS, IQ2_S and IQ1_M: a scale a
// 16 values, two a group). The k-quants and the legacy quants keep their
// own width; IQ3_S is 4 bits (its odd grid values, 2 apart), IQ3_XXS 6
// (its grid values, even, halved), IQ4_NL / IQ4_XS and Q3_K 8 (a
// non-uniform codebook; two 16-value scales a group, as d * (sc * (q -
// 4)) with the product the code), IQ2_XXS 8 (its grid values {8, 25, 43}
// signed: 7 would do, and no kernel takes 7).
int affine_bits(std::uint32_t type);
// `n` values (whole blocks; a multiple of 32) at `src` as codes of
// `bits` (>= affine_bits(type), <= 8) into `codes` (one byte each) and a
// group's form into groups[n / 32]. False when the type has no form or a
// code does not fit.
bool to_affine32(std::uint32_t type, const void* src, std::int64_t n,
                 int bits, std::uint8_t* codes, Affine32* groups);

// One row of `K` values (whole blocks) at `src` as affine codes of `bits`
// at group 32 -- one byte a code, unpacked -- and each group's scale and
// bias in bf16 (else f16): the EXACT form where `type` has one at `bits`
// (to_affine32; a free bias FITTED to the stored scale), else, at 8 bits
// only, the decoded values rounded to the nearest of 255 steps over the
// group's range (the bias the group's minimum, the scale the next step
// up from range / 255). False where neither applies.
bool row_affine32(std::uint32_t type, const void* src, int K, int bits,
                  bool bf16, std::uint8_t* codes, std::uint16_t* scales,
                  std::uint16_t* biases);
// The width row_affine32 takes `type` at: affine_bits(type), else 8 for
// a type this file decodes; 0 for one it does not.
int affine_width(std::uint32_t type);

// The grids and signs ggml_to_affine.metal reads, as one table of 2832
// u64: the IQ2_XXS grid (256), the IQ2_XS grid (512), the IQ1_S / IQ1_M
// grid (2048), then ksigns_iq2xs's 128 bytes (16).
std::vector<std::uint64_t> gpu_iq_tables();
// The same, then the IQ1_S / IQ1_M grid as nibbles (value + 1) a u32 an
// entry (1024 u64 more, at 2832), IQ2_S's grid (1024 u64, at 3856),
// IQ3_XXS's (256 u32, at 4880), IQ3_S's (512 u32, at 5008) and IQ4_NL's
// values (16 int8, at 5264): what the native expert kernels read
// (ops/ggml_native.h).
std::vector<std::uint64_t> gpu_iq_tables_native();
// Whether ggml_to_affine.metal converts `type` (at 8 bits): IQ2_XXS,
// IQ2_XS, IQ1_M.
bool gpu_affine8(std::uint32_t type);

// `n` codes of `bits` packed as MLX packs them: a little-endian bit
// stream, code j at bits [j * bits, (j + 1) * bits). `out` holds
// n * bits / 8 bytes (n * bits a multiple of 8).
void pack_codes(const std::uint8_t* codes, std::int64_t n, int bits,
                std::uint8_t* out);

}  // namespace vpipe::genai::ggml

#endif  // VPIPE_GENERATIVE_MODELS_GGML_BLOCKS_H
