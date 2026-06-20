#ifndef VPIPE_GENERATIVE_MODELS_GGUF_CONVERT_H
#define VPIPE_GENERATIVE_MODELS_GGUF_CONVERT_H

// Convert a Gemma-4 GGUF checkpoint into the exact weight layout vpipe's
// existing Gemma-4 executors expect, so a `.gguf` loads token-for-token
// identically on BOTH the MLX and the metal-compute backend.
//
// The mapping (gemma4 / gemma4_unified family):
//   * Q4_0 linears  -> MLX-affine 4-bit, group 32. This is a LOSSLESS
//     repack: q4_0 dequant is (q-8)*d, which is affine `scale*q + bias`
//     with scale=d, bias=-8d (both exact in fp16) -- only the nibble
//     packing differs (q4_0 splits a 32-block into low/high nibbles;
//     MLX packs 8 consecutive 4-bit values per u32).
//   * Q6_K token_embd (tied lm_head) -> MLX-affine 8-bit, group 32.
//     q6_K isn't affine, so this is a dequant + requant (relL2 ~0.5%);
//     8-bit keeps the logit head high-fidelity.
//   * RMSNorm weights pass through unchanged: Gemma folds the +1 into
//     the stored weight (~1.023) and both vpipe execs apply it directly,
//     exactly matching the GGUF convention.
//   * per-layer `layer_output_scale` -> `layer_scalar` (raw passthrough).
//
// Tensor names are renamed blk.N.* -> language_model.model.layers.N.*.
//
// The converter is MLX-free; the MLX loader and the metal weight reader
// both drive it (a stateless `convert(spec, dst)` per tensor, plus a
// small auto-freed cache for the one expensive q6_K embedding) so neither
// path needs a multi-GB staging arena.

#include "generative-models/model-loader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai {

class GgufFile;

// Describes one output tensor (an HF name + its backend dtype/shape) and
// how to produce it from the GGUF. `nbytes` is the size of the converted
// payload `convert()` writes.
struct ConvertedTensorSpec {
  enum class Op {
    kQ4Weight,      // q4_0 source -> affine4 g32 packed u32 weight
    kQ4Scales,      // q4_0 source -> affine4 g32 scales (f16)
    kQ4Biases,      // q4_0 source -> affine4 g32 biases (f16)
    kEmbedWeight,   // q6_K source -> affine8 g32 packed u32 weight
    kEmbedScales,   // q6_K source -> affine8 g32 scales (f16)
    kEmbedBiases,   // q6_K source -> affine8 g32 biases (f16)
    kEmbedQ6KRaw,   // q6_K source -> raw Q6_K bytes (metal-only, lossless)
    kFloatPass,     // f32/f16/bf16/q8_0 source -> f32 (norms, a/b, conv, dt)
    kKQuantRaw,     // any k-quant source -> raw k-quant bytes (Qwen native)
    kSsmALog,       // f32 ssm_a (= -exp(A_log)) -> A_log = log(-ssm_a)
  };
  std::string               hf_name;
  std::string               dtype;     // "U32"/"F16"/"F32"/"Q4K"/"Q5K"/"Q6K"
  std::vector<std::int64_t> shape;
  std::uint64_t             nbytes = 0;
  std::string               gguf_name;
  Op                        op = Op::kFloatPass;
};

// Polymorphic GGUF->vpipe tensor converter. open_model() picks the
// arch-specific implementation; MetalLlamaWeights drives it uniformly
// (read specs(), then convert() each into a caller-owned SharedBuffer).
class GgufConverterBase {
public:
  virtual ~GgufConverterBase() = default;
  virtual const std::vector<ConvertedTensorSpec>& specs() const = 0;
  virtual bool convert(const ConvertedTensorSpec& spec,
                       std::uint8_t* dst) = 0;
};

// Locate the primary model `.gguf` in `dir` (skips an `mmproj-*` vision
// projector). Returns the absolute path, or "" if none is present.
std::string find_gguf_in_dir(const std::string& dir);

// Populate `*out` from a gemma4 GGUF's metadata. False if the file isn't
// a recognised gemma4 family checkpoint.
bool gguf_to_model_config(const GgufFile& g, ModelConfig* out);

// Plans + executes the per-tensor conversion. Construct once over an open
// GgufFile (held by the caller for the converter's lifetime); read
// `specs()` to learn the output tensors, then `convert()` each into a
// caller-owned buffer of `spec.nbytes`.
class GgufGemma4Converter : public GgufConverterBase {
public:
  GgufGemma4Converter(const GgufFile* gguf, const ModelConfig& cfg);
  ~GgufGemma4Converter() override;

  GgufGemma4Converter(const GgufGemma4Converter&)            = delete;
  GgufGemma4Converter& operator=(const GgufGemma4Converter&) = delete;

  // The complete set of HF output tensors (built in the ctor).
  const std::vector<ConvertedTensorSpec>& specs() const override {
    return _specs;
  }

  // Write the converted bytes for `spec` into `dst` (>= spec.nbytes).
  // False on a missing/unsupported source tensor.
  bool convert(const ConvertedTensorSpec& spec, std::uint8_t* dst) override;

private:
  void build_specs_();
  bool ensure_embed_();          // build the q6_K->affine8 cache once
  void maybe_release_embed_();   // free the cache after all 3 served

  const GgufFile*                 _g;
  ModelConfig                     _cfg;
  std::vector<ConvertedTensorSpec> _specs;
  // q6_K embedding cache (weight/scales/biases), built lazily, freed once
  // all three sub-tensors have been served.
  std::vector<std::uint32_t>      _embed_w;
  std::vector<std::uint16_t>      _embed_s;
  std::vector<std::uint16_t>      _embed_b;
  bool                            _embed_ready = false;
  unsigned                        _embed_served = 0;   // bit per sub-tensor
};

// Converts a `qwen35` (Qwen3.5 hybrid GDN) GGUF into the layout the metal
// Qwen exec expects, NATIVELY (no requant): k-quant linear weights pass
// through as raw blocks (the metal forward dispatches q4_K/q5_K/q6_K qmv +
// dequant-to-f16 GEMM per tensor), and the f32 side-tensors get their small
// transforms (A_log = log(-ssm_a); norms / conv1d / dt_bias passthrough;
// the two tiny Q8_0 alpha/beta projections dequant to f32). token_embd is a
// raw Q6_K table (tied lm_head). MLX has no native k-quant path, so this
// converter is metal-only.
class GgufQwen35Converter : public GgufConverterBase {
public:
  GgufQwen35Converter(const GgufFile* gguf, const ModelConfig& cfg);
  ~GgufQwen35Converter() override = default;

  GgufQwen35Converter(const GgufQwen35Converter&)            = delete;
  GgufQwen35Converter& operator=(const GgufQwen35Converter&) = delete;

  const std::vector<ConvertedTensorSpec>& specs() const override {
    return _specs;
  }
  bool convert(const ConvertedTensorSpec& spec, std::uint8_t* dst) override;

private:
  void build_specs_();
  const GgufFile*                  _g;
  ModelConfig                      _cfg;
  std::vector<ConvertedTensorSpec> _specs;
};

}  // namespace vpipe::genai

#endif
