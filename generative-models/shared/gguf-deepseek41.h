// DeepSeek-V4.1-Flash as a GGUF writes it (architecture `deepseek41`:
// ISTA-DASLab's GSQ-RCO builds and AtomicChat's edits of them): its
// config, and its tensors presented under the MLX pack's names
// MetalDeepseekV41Model reads (oMLX's layout: `language_model.` before
// the layers, the embedding, the head and the final norm).
//
// MEASURED against the oQ4e pack of the same release, tensor by tensor:
// nothing is permuted or re-centred -- the norms, the hyper-connections,
// the sinks, the router and its bias, the compressors and the indexers
// are the release's values as stored (F32 / BF16, identical); wo_a's
// [4096, 1024, 8] is the release's [8 x 1024, 4096] row order. So the
// conversion is the block formats alone:
//   * a quantized projection (Q8_0, Q5_K, ...) as affine codes at group
//     32, exactly where the format has the form (ggml-blocks.h), a bf16
//     scale and bias -- what the pack's affine modules are;
//   * the norms as bf16 (they were bf16 before llama.cpp widened them);
//   * a layer's routed experts as affine stacks at that layer's widths
//     (w1 and w3 at the wider of the two, w2 at its own) -- IQ2_XXS and
//     IQ3_XXS exactly, the formats with a scale a 16 values (IQ1_M,
//     IQ2_XS, IQ2_S) rounded to 8 bits over each group of 32;
//   * the engram's hash tables (token map, multipliers, primes, offsets)
//     as stored, under `gguf.engram_*`.
// The engram's n-gram TABLE is a file of its own (the key
// `deepseek41.engram.file`, beside the shards), read a row at a time by
// the model.
#ifndef VPIPE_GENERATIVE_MODELS_GGUF_DEEPSEEK41_H
#define VPIPE_GENERATIVE_MODELS_GGUF_DEEPSEEK41_H

#include "generative-models/shared/gguf-convert.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::genai {

class GgufFile;
struct ModelConfig;

// The config.json text a `deepseek41` GGUF stands for (model_type
// deepseek_v41; an omlx_deepseek_v41 record saying its modules are
// affine), for DeepseekV41Config::from_json_text. False with `err` set
// when a key the model needs is missing.
bool gguf_to_deepseek41_json(const GgufFile& g, std::string* json,
                             std::string* err);

// The same as the ModelConfig the loader reads (architecture
// DeepseekV41ForCausalLM: the language-model family and the plan key on
// it).
bool gguf_to_deepseek41_config(const GgufFile& g, ModelConfig* out);

// The engram table's file named by the GGUF (`deepseek41.engram.file`):
// beside `first`, the backbone's first shard, or -- fetched as a record
// of its own -- in another repo directory of the same models root
// (<root>/<org>/<repo>/); where found nowhere, the path beside `first`
// (the load then says it is missing). "" where the GGUF names none.
std::string gguf_deepseek41_engram_path(const GgufFile& g,
                                        const std::string& first);

class GgufDeepseek41Converter : public GgufConverterBase {
public:
  explicit GgufDeepseek41Converter(const GgufFile* gguf);
  ~GgufDeepseek41Converter() override = default;
  GgufDeepseek41Converter(const GgufDeepseek41Converter&) = delete;
  GgufDeepseek41Converter& operator=(const GgufDeepseek41Converter&) =
      delete;

  const std::vector<ConvertedTensorSpec>& specs() const override {
    return _specs;
  }
  bool convert(const ConvertedTensorSpec& spec, std::uint8_t* dst) override;

private:
  enum class Kind {
    kAffine,    // a projection: affine codes at group 32 (3 parts)
    kBf16,      // decoded into bf16
    kF32,       // decoded into f32
    kRaw,       // the bytes as stored (the engram's integer tables)
    kExpert,    // a routed-expert stack (3 parts)
  };
  struct Src {
    Kind kind = Kind::kBf16;
    std::string gguf;
    int part = 0;       // 0 codes, 1 scales, 2 biases
    int bits = 8;
  };
  // A projection's three parts, built once and kept until all are served
  // (its codes only where a scale part was asked for first).
  struct Built {
    std::vector<std::uint8_t> codes;          // packed
    std::vector<std::uint16_t> scales, biases;
    int served = 0;
  };
  void build_();
  void add_affine_(const std::string& hf, const std::string& gguf);
  void add_float_(const std::string& hf, const std::string& gguf, Kind k);
  void add_expert_(const std::string& hf, const std::string& gguf,
                   int bits);
  // The codes into `codes` (nullptr: kept in `b`), the groups into `b`.
  bool build_affine_(const Src& s, Built& b, std::uint8_t* codes) const;

  const GgufFile* _g;
  std::vector<ConvertedTensorSpec> _specs;
  std::unordered_map<std::string, Src> _src;
  std::unordered_map<std::string, Built> _built;   // by projection
};

// One routed expert's three GGUF slabs (w1 = ffn_gate_exps, w3 =
// ffn_up_exps, w2 = ffn_down_exps, each the expert's [N, K] rows of
// blocks) into the DeepSeek cache's nine parts: w1 codes, scales,
// biases, w3's, w2's -- codes packed at `b13` / `b2` bits, group 32,
// bf16 scales and biases. False where a format takes neither.
bool deepseek41_expert_to_affine(std::uint32_t w1_type,
                                 const std::uint8_t* w1,
                                 std::uint32_t w3_type,
                                 const std::uint8_t* w3,
                                 std::uint32_t w2_type,
                                 const std::uint8_t* w2, int I, int H,
                                 int b13, int b2, std::uint8_t* const* dst);

// `rows` rows of `K` values of `type` at `src` into affine codes of
// `bits` (packed at `w`), group 32, bf16 scales and biases at `s` / `b`
// (a slab of deepseek41_expert_to_affine's, for a caller splitting one).
bool deepseek41_rows_to_affine(std::uint32_t type, const std::uint8_t* src,
                               int rows, int K, int bits, std::uint8_t* w,
                               std::uint8_t* s, std::uint8_t* b);

// The width an expert role of `type` is held at: the format's exact form
// where it has one (IQ2_XXS 8, IQ3_XXS 6, ...), else 8.
int deepseek41_expert_bits(std::uint32_t type);

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_GGUF_DEEPSEEK41_H
