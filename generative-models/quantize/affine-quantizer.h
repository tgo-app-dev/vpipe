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
// scales + biases in the MLX-affine layout (bias = group min,
// w = scale*q + bias). Wraps the affine_quant_* metal kernels which live in
// the affine_dequant metallib (f16) / affine_dequant_bf16 (bf16 input).
class AffineQuantizer {
public:
  // How the scales and biases are written. kF16 is the original format:
  // F16, the codes fit to the unrounded f32 scale. The other two round the
  // pair to the type a model RUNS the pack in, write it in that type, and
  // fit the codes to the rounded pair (affine_quant_fit_*): kFitF16 for a
  // model that runs f16, kFitBf16 (scales and biases BF16) for one that
  // runs bf16 -- which holds an F16 pack's scales rounded to bf16 after the
  // codes were fit to neither.
  enum class Scales { kF16, kFitF16, kFitBf16 };
  // The safetensors dtype each writes its scales and biases in.
  static const char* scale_dtype(Scales sc)
  { return sc == Scales::kFitBf16 ? "BF16" : "F16"; }

  explicit AffineQuantizer(metal_compute::MetalCompute* mc);

  bool valid() const noexcept { return _ok; }

  // Quantize w_in[N,K] (src_bf16 selects the bf16-input kernel) into
  // freshly-allocated (w_out U32, s_out, b_out -- F16 or BF16 as `sc`
  // says, scale_dtype()). bits in {4,8}, group in {32,64}, clip 1.0 =
  // plain min/max. One self-contained command stream (commit+wait).
  // Returns false on bad args.
  bool quantize_linear(const metal_compute::SharedBuffer& w_in, bool src_bf16,
                       int N, int K, int bits, int group, float clip,
                       metal_compute::SharedBuffer& w_out,
                       metal_compute::SharedBuffer& s_out,
                       metal_compute::SharedBuffer& b_out,
                       Scales sc = Scales::kF16) const;

  // Byte sizes of the three outputs for a [N,K] bits/group quant.
  static std::size_t weight_bytes(int N, int K, int bits)
  { return (std::size_t)N * (K * bits / 32) * 4; }
  static std::size_t scale_bytes(int N, int K, int group)
  { return (std::size_t)N * (K / group) * 2; }   // F16 or BF16

private:
  // [src_bf16][bits 8->0,4->1][group 64->0,32->1]; [sc] for the scales
  const metal_compute::ComputeFunction*
  fn_(bool bf16, int bits, int group, Scales sc) const;

  metal_compute::MetalCompute*  _mc = nullptr;
  metal_compute::ComputeLibrary _lib;        // half input
  metal_compute::ComputeLibrary _lib_bf16;   // bfloat input
  metal_compute::ComputeFunction _fn[2][2][2];
  metal_compute::ComputeFunction _fn_fit[2][2][2][2];   // [.][.][.][bf16 s]
  bool _ok = false;
};

}  // namespace vpipe::genai

#endif
