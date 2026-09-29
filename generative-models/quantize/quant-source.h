#ifndef VPIPE_GENAI_QUANTIZE_QUANT_SOURCE_H
#define VPIPE_GENAI_QUANTIZE_QUANT_SOURCE_H

// The quantizer's view of its source checkpoint: every tensor as the
// OUTPUT will name and type it.
//
// Two things can stand between a checkpoint on disk and what the
// quantizer is written against, and this is where both are undone so
// that nothing downstream has to know:
//
//   * FP8 storage, plain or SCALED (shared/fp8-layout.h). FP8 tensors are
//     reported, and loaded, as BF16: plain ones exactly (every FP8 value
//     is a bf16 value), scaled ones as code * scale rounded once to bf16.
//     The encoding's own tensors -- scales, the `scaled_fp8` marker,
//     comfy_quant records, activation scales -- are HIDDEN, since the
//     values they describe are what is presented. An encoding this
//     cannot decode (MXFP8, NVFP4, int8, block scales) is refused at open,
//     by name.
//   * A foreign LAYOUT. A single-file Krea-2 DiT in Krea's own naming
//     (krea2/krea2-native-checkpoint.h) is read under the diffusers names
//     every loader here uses -- MetalLlamaWeights::open_model translates
//     it -- and this brings the config.json it lacks, synthesized from
//     its shapes.
//
// Everything else is the MetalLlamaWeights it wraps, unchanged: the same
// names, dtypes and bytes.

#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/fp8.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class SharedBuffer;
}

namespace vpipe::genai {

class QuantSource {
public:
  using TensorInfo = MetalLlamaWeights::TensorInfo;

  // Open `path` (a model directory, or a single .safetensors file) the way
  // MetalLlamaWeights::open_model does. nullopt + *err on failure --
  // including a scaled-FP8 checkpoint, or a native layout with a tensor
  // this build cannot place.
  static std::optional<QuantSource> open(const std::string& path,
                                         std::string*       err);

  QuantSource(QuantSource&&) noexcept            = default;
  QuantSource& operator=(QuantSource&&) noexcept = default;

  // Output-side names, dtypes (BF16 for a decoded FP8 tensor) and shapes.
  const TensorInfo*        info(const std::string& name) const;
  std::vector<std::string> tensor_names() const;

  // The tensor's bytes as info() describes them: decoded to BF16 when it
  // is stored as FP8, the raw bytes otherwise. Empty on failure.
  metal_compute::SharedBuffer load(const std::string&           name,
                                   metal_compute::MetalCompute* mc) const;

  // A config.json for a source that has none of its own (the native
  // layout); null when the caller should read or lift the source's own.
  const FlexData* config() const {
    return _has_config ? &_config : nullptr;
  }

  // "krea2-native" when the source was translated, "" otherwise.
  const std::string& layout() const { return _layout; }

  // How many tensors are stored as FP8 and read back as BF16, and how
  // many of those carry a scale (folded into the values).
  int n_fp8() const { return _n_fp8; }
  int n_scaled() const { return _n_scaled; }

  // Tensors in the file that are not part of the model being read -- a
  // full ComfyUI checkpoint bundles its text encoder and VAE beside the
  // DiT, under other prefixes. Reported, never read.
  int n_ignored() const { return _n_ignored; }

private:
  QuantSource() = default;

  struct Entry {
    std::string        src;    // name in the reader
    TensorInfo         info;   // as presented: BF16 where the file has FP8
    fp8::Format        f8 = fp8::Format::kNone;
    std::size_t        rows = 0, cols = 0;
    std::vector<float> scale;  // scaled FP8: one value, or one per row
  };

  std::optional<MetalLlamaWeights>       _w;
  std::unordered_map<std::string, Entry> _map;
  FlexData                               _config;
  bool                                   _has_config = false;
  std::string                            _layout;
  int                                    _n_fp8     = 0;
  int                                    _n_scaled  = 0;
  int                                    _n_ignored = 0;
};

}  // namespace vpipe::genai

#endif
