#ifndef VPIPE_GENERATIVE_MODELS_SHARED_FP8_EXPAND_H
#define VPIPE_GENERATIVE_MODELS_SHARED_FP8_EXPAND_H

// fp8-expand.h -- an FP8 weight kept AS FP8 and widened on the GPU right
// before the GEMM that reads it.
//
// A model that stores the checkpoint's codes as they are holds an FP8
// file at its own size -- half of bf16 -- and a streaming model reads
// half the bytes a step. The price is one elementwise pass per GEMM,
// writing the [N, K] bf16 weight into a scratch the model already owns
// for dequantized weights; every GEMM path, LoRA fold and i8 route then
// sees an ordinary dense operand.
//
// The expansion is value = code (x scale), the product taken in f32 and
// rounded ONCE to bf16 -- what fp8::decode_bf16 computes on the host, so
// running the file and running its dense conversion agree to the bit
// (shared/fp8-layout.h). The scale is one value, or one per output row.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/fp8-layout.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class ComputeEncoder;
class MetalCompute;
}

namespace vpipe::genai {
class MetalLlamaWeights;
}

namespace vpipe::genai::fp8 {

// One FP8 weight as a model keeps it, BORROWED for an expansion: the
// file's [N, K] codes, their format, and the scale as f32 (null or empty
// when unscaled). The buffers stay the model's.
struct Codes {
  const metal_compute::SharedBuffer* codes   = nullptr;
  Format                             format  = Format::kNone;
  const metal_compute::SharedBuffer* scale   = nullptr;  // f32: 1, or N
  bool                               per_row = false;
  bool any() const noexcept {
    return format != Format::kNone && codes != nullptr;
  }
  bool scaled() const noexcept { return scale != nullptr && !scale->empty(); }
};

// The four expansion kernels (fp8_{e4m3,e5m2}_dequant[_scaled] in the
// affine_dequant library), loaded once per model.
class Expander {
 public:
  // Load them from the bf16 library. False when any is missing -- which
  // a model storing FP8 cannot run without, so it refuses at load.
  bool load(metal_compute::MetalCompute* mc);
  bool valid() const noexcept;

  // Encode codes -> `dst`, [N, K] bf16, and return true. False, with
  // nothing encoded, when the buffers are short, the element count
  // does not fit the kernels' 32-bit index, or the kernels are absent.
  bool encode(metal_compute::ComputeEncoder& enc, const Codes& w,
              const metal_compute::SharedBuffer& dst, std::size_t N,
              std::size_t K) const;

 private:
  metal_compute::ComputeLibrary  _lib;
  metal_compute::ComputeFunction _e4m3, _e5m2, _e4m3_s, _e5m2_s;
};

// The scale of the FP8 weight `name` as the f32 buffer Codes::scale is:
// empty for an unscaled weight (and *ok true), one value or `rows` of
// them otherwise. *ok false when the scale cannot be read.
metal_compute::SharedBuffer scale_buffer(const MetalLlamaWeights& w,
                                         const Weight& x, std::size_t rows,
                                         metal_compute::MetalCompute* mc,
                                         bool* ok);

// One FP8 tensor decoded whole on the host, to bf16 or f32 -- for the
// small tensors a model converts rather than keeps (norms, embeddings,
// host-side MLPs) and for fallbacks. False + *err when `name` is not an
// FP8 tensor this build reads, or `n` is not its element count.
bool decode_tensor_bf16(const MetalLlamaWeights& w, const std::string& name,
                        std::uint16_t* dst, std::size_t n, std::string* err);
bool decode_tensor_f32(const MetalLlamaWeights& w, const std::string& name,
                       float* dst, std::size_t n, std::string* err);

}  // namespace vpipe::genai::fp8

#endif
