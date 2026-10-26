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
//
// An INT8 weight (fp8::Format::kI8, a model that opted in) widens the
// same way, code x scale; a ConvRot one has its CODES un-rotated in the
// same pass, exactly, before the scale and the one rounding
// (fp8::convrot_unrotate's arithmetic, a threadgroup a 1024 elements --
// whole groups, since a group is a power of four no larger than that
// and divides each row).

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/fp8-layout.h"

#include <cstddef>
#include <string>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

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
  int                                rot_group = 0;      // int8 ConvRot
  bool any() const noexcept {
    return format != Format::kNone && codes != nullptr;
  }
  bool scaled() const noexcept { return scale != nullptr && !scale->empty(); }
};

// The expansion kernels (fp8_{e4m3,e5m2}_dequant[_scaled], and int8's
// i8_dequant_scaled / i8_convrot_dequant, in the affine_dequant
// library), loaded once per model.
class Expander {
 public:
  // Load them from the bf16 library. False when an FP8 kernel is missing
  // -- which a model storing FP8 cannot run without, so it refuses at
  // load. The int8 pair is OPTIONAL (kernel-contract.h): a host older
  // than them leaves int8_valid() false and FP8 working, and a model
  // that reads int8 refuses an int8 checkpoint there by name.
  bool load(metal_compute::MetalCompute* mc);
  bool valid() const noexcept;
  bool int8_valid() const noexcept;

  // Encode codes -> `dst`, [N, K] bf16, and return true. False, with
  // nothing encoded, when the buffers are short, the element count
  // does not fit the kernels' 32-bit index, or the kernels are absent
  // (for int8, int8_valid()).
  bool encode(metal_compute::ComputeEncoder& enc, const Codes& w,
              const metal_compute::SharedBuffer& dst, std::size_t N,
              std::size_t K) const;

 private:
  metal_compute::ComputeLibrary  _lib;
  metal_compute::ComputeFunction _e4m3, _e5m2, _e4m3_s, _e5m2_s;
  metal_compute::ComputeFunction _i8_s, _i8_rot;   // int8; and ConvRot
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
