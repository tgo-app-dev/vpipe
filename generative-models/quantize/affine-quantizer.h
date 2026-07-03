#ifndef VPIPE_GENAI_QUANTIZE_AFFINE_QUANTIZER_H
#define VPIPE_GENAI_QUANTIZE_AFFINE_QUANTIZER_H

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>

namespace vpipe::metal_compute {
class MetalCompute;
}

namespace vpipe::genai {

// Forward group-affine quantizer: dense fp weight [N,K] -> packed U32 codes +
// F16 scales + F16 biases in the MLX-affine layout (bias = group min,
// w = scale*q + bias). Wraps the affine_quant_* metal kernels which live in
// the affine_dequant metallib (f16) / affine_dequant_bf16 (bf16 input).
// scales/biases are always F16 on output.
class AffineQuantizer {
public:
  explicit AffineQuantizer(metal_compute::MetalCompute* mc);

  bool valid() const noexcept { return _ok; }

  // Quantize w_in[N,K] (src_bf16 selects the bf16-input kernel; scales/biases
  // stay F16) into freshly-allocated (w_out U32, s_out F16, b_out F16).
  // bits in {4,8}, group in {32,64}, clip 1.0 = plain min/max. One
  // self-contained command stream (commit+wait). Returns false on bad args.
  bool quantize_linear(const metal_compute::SharedBuffer& w_in, bool src_bf16,
                       int N, int K, int bits, int group, float clip,
                       metal_compute::SharedBuffer& w_out,
                       metal_compute::SharedBuffer& s_out,
                       metal_compute::SharedBuffer& b_out) const;

  // Byte sizes of the three outputs for a [N,K] bits/group quant.
  static std::size_t weight_bytes(int N, int K, int bits)
  { return (std::size_t)N * (K * bits / 32) * 4; }
  static std::size_t scale_bytes(int N, int K, int group)
  { return (std::size_t)N * (K / group) * 2; }   // F16

private:
  // [src_bf16][bits 8->0,4->1][group 64->0,32->1]
  const metal_compute::ComputeFunction*
  fn_(bool bf16, int bits, int group) const;

  metal_compute::MetalCompute*  _mc = nullptr;
  metal_compute::ComputeLibrary _lib;        // half input
  metal_compute::ComputeLibrary _lib_bf16;   // bfloat input
  metal_compute::ComputeFunction _fn[2][2][2];
  bool _ok = false;
};

}  // namespace vpipe::genai

#endif
