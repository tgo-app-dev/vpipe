// Qwen3.8-Flash-Next as llama.cpp writes it (GGUF architecture
// `qwen4exp`): its config, its tensors presented under the Hugging Face
// names MetalQwenModel's hyper-connection loader reads, and its routed
// experts converted -- exactly -- into the expert cache's affine layout.
//
// What llama.cpp's converter changed, each restored here (MEASURED
// against the MLX oQ5e pack of the same BF16 source):
//   * every zero-centred norm is stored as 1 + w (hyper-connection,
//     q / k, the indexer's and the PLE's); the GDN's gated norm is not;
//   * ssm_a = -exp(A_log);
//   * the GDN's 48 value heads are STRIDED, GGUF head j * 16 + k holding
//     HF head k * 3 + j -- in in_proj_qkv's v rows, in_proj_z, a and b,
//     dt_bias, A_log, conv1d's v channels and out_proj's columns; put
//     back in HF order here, so the model runs one layout;
//   * the indexer's q | k is two tensors, the n-gram table one tensor
//     of 320M rows, its hash tables metadata.
//
// The non-expert projections are held as affine w8 at group 32 -- the
// form every block format here but Q6_K and IQ2_S takes EXACTLY (a
// group's scale in bf16 the only rounding); those two and the float
// sources re-encoded to it -- and the router stays dense. A routed expert is converted from its blocks to
// affine codes at group 32 with NOTHING rounded but each group's scale
// and bias in the compute dtype (ggml-blocks.h: to_affine32), at one
// gate|up width and one down width for the whole stack (the widest any
// layer needs; a narrower layer's codes widen exactly).
#ifndef VPIPE_GENERATIVE_MODELS_GGUF_QWEN4EXP_H
#define VPIPE_GENERATIVE_MODELS_GGUF_QWEN4EXP_H

#include "generative-models/shared/gguf-convert.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::genai {

class GgufFile;
struct ModelConfig;

// A `qwen4exp` GGUF's config, as the safetensors release's config.json
// would give it (architecture Qwen4ExpForConditionalGeneration, its
// text_config, the w8 group).
bool gguf_to_qwen4exp_config(const GgufFile& g, ModelConfig* out);

class GgufQwen4ExpConverter : public GgufConverterBase {
public:
  explicit GgufQwen4ExpConverter(const GgufFile* gguf);
  ~GgufQwen4ExpConverter() override = default;
  GgufQwen4ExpConverter(const GgufQwen4ExpConverter&) = delete;
  GgufQwen4ExpConverter& operator=(const GgufQwen4ExpConverter&) = delete;

  const std::vector<ConvertedTensorSpec>& specs() const override {
    return _specs;
  }
  bool convert(const ConvertedTensorSpec& spec, std::uint8_t* dst) override;

  // The routed experts' widths (gate|up, down) and group.
  int gate_bits() const { return _gb; }
  int down_bits() const { return _db; }

private:
  enum class Kind {
    kW8,        // decoded, re-encoded affine w8 g64 (weight/scales/biases)
    kF32,       // a float tensor as f32 (minus one, permuted, A_log)
    kI64,       // a metadata int array
    kExpert,    // a routed-expert stack (weight/scales/biases)
  };
  struct Src {
    Kind kind = Kind::kF32;
    int part = 0;                        // 0 weight, 1 scales, 2 biases
    std::vector<std::string> gguf;       // row-concatenated sources
    std::string key;                     // kI64: the metadata key
    bool minus_one = false;              // zero-centred norm: 1 + w stored
    bool a_log = false;                  // ssm_a -> log(-ssm_a)
    int vrows = 0;      // permute v-head row blocks from this row on
    int vcols = 0;      // permute v-head column blocks (out_proj)
    int vunit = 0;      // rows (cols) a v-head (0: no permutation)
    int bits = 8;       // kExpert
  };
  void build_();
  void add_w8_(const std::string& hf, std::vector<std::string> gguf,
               int vrows = -1, int vunit = 0, bool vcols = false);
  void add_f32_(const std::string& hf, const std::string& gguf,
                bool minus_one = false, int vunit = 0, int vrow0 = 0,
                bool a_log = false);
  void add_i64_(const std::string& hf, const std::string& key);
  void add_expert_(const std::string& hf, const std::string& gguf, int bits);
  // HF v-head h's GGUF head (llama.cpp strides them).
  int v_src_(int h) const;
  // One projection's scales and biases (bf16), kept from its codes'
  // build until all three parts are served.
  struct W8 {
    std::vector<std::uint16_t> scales, biases;
    int served = 0;
  };
  // The codes into `codes` (nullptr: a scratch) and the groups into `out`.
  bool build_w8_(const Src& s, W8& out, std::uint8_t* codes) const;

  const GgufFile* _g;
  std::vector<ConvertedTensorSpec> _specs;
  std::unordered_map<std::string, Src> _src;
  std::unordered_map<std::string, W8> _w8;    // by projection base name
  int _nk = 16, _nv = 48;                     // GDN key / value heads
  int _gb = 0, _db = 0;                       // the experts' widths
};

// A routed expert's GGUF rows into the expert cache's affine parts (see
// LayerLoad::affine_expert_parts: gate|up interleaved by row, then down;
// group 32). `gate`, `up` and `down` are one expert's slab of each stack
// as the file holds it ([N, K] rows of blocks); `bf16` the compute dtype
// the scales and biases are written in. False where a format has no
// exact affine form at the stack's widths.
bool qwen4exp_expert_to_affine(std::uint32_t gate_type,
                               const std::uint8_t* gate,
                               std::uint32_t up_type, const std::uint8_t* up,
                               std::uint32_t down_type,
                               const std::uint8_t* down, int I, int H,
                               int gb, int db, bool bf16,
                               std::uint8_t* const* dst);

// The PLE table's rows (per_layer_token_embd) decode: `n` rows of `dim`
// values of `type` at `src` into bf16 or f16.
bool qwen4exp_rows(std::uint32_t type, const std::uint8_t* src,
                   std::int64_t n, int dim, bool bf16, std::uint16_t* out);

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_GGUF_QWEN4EXP_H
