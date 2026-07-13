#ifndef VPIPE_GENERATIVE_MODELS_METAL_LLAMA_WEIGHTS_H
#define VPIPE_GENERATIVE_MODELS_METAL_LLAMA_WEIGHTS_H

// MetalLlamaWeights -- thin, MLX-free safetensors reader that mmaps a
// model's safetensors and hands individual tensors to the metal-compute
// LLM kernels as SharedBuffers (one memcpy per requested tensor from
// the mmap into a UMA buffer). The on-disk quantized layout already
// matches the kernels: weight U32 [N, K/8], scales/biases F16
// [N, K/64]. Used by the procedural metal Llama exec (M5) and its
// tests; deliberately does not depend on MLX.
//
// Handles both single-file checkpoints (model.safetensors) and the
// multi-shard layout larger models in a family ship as
// (model-00001-of-0000N.safetensors + model.safetensors.index.json):
// every shard is mmapped and its tensors merged into one namespace, so
// callers never see the split. Use open_model(dir) for that; open(path)
// remains for a single named file (tests).

#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

// Holds the state for a GGUF-backed checkpoint (the GgufFile + the
// on-demand converter + the name->spec map). Defined in the .cc so the
// header stays free of the gguf/MLX headers. Null for safetensors.
struct GgufBacking;

class MetalLlamaWeights {
public:
  struct TensorInfo {
    std::string          dtype;    // "U32", "F16", ...
    std::vector<int64_t> shape;
    std::uint64_t        offset;   // bytes from the start of the data blob
    std::uint64_t        nbytes;
    int                  shard = 0;  // index into _shards; -2 => GGUF-backed
  };

  // Open a model directory. Three layouts are recognised, in order:
  //   * a `.gguf` file (GGUF checkpoint): parsed + converted on demand to
  //     the affine layout the metal execs expect (q4_0 -> 4-bit g32,
  //     q6_K token table -> 8-bit g32). No staging arena -- each tensor
  //     is converted straight into its SharedBuffer in load().
  //   * model.safetensors.index.json: mmap every shard it references and
  //     merge their tensors.
  //   * model.safetensors: mmap the single file.
  // Returns nullopt on any error.
  static std::optional<MetalLlamaWeights> open_model(
      const std::string& model_dir);

  // Parse + mmap a single named `safetensors_path`. Returns nullopt on
  // any error. (Shard-unaware; prefer open_model for a model directory.)
  static std::optional<MetalLlamaWeights> open(
      const std::string& safetensors_path);

  MetalLlamaWeights(MetalLlamaWeights&&) noexcept;
  MetalLlamaWeights& operator=(MetalLlamaWeights&&) noexcept;
  MetalLlamaWeights(const MetalLlamaWeights&) = delete;
  MetalLlamaWeights& operator=(const MetalLlamaWeights&) = delete;
  ~MetalLlamaWeights();

  bool has(const std::string& name) const;
  const TensorInfo* info(const std::string& name) const;

  // All tensor names in the checkpoint (unordered). Used by the model
  // quantizer to enumerate + classify every tensor.
  std::vector<std::string> tensor_names() const;

  // Allocate a SharedBuffer and copy the tensor's bytes into it.
  // Returns an empty SharedBuffer if the tensor is missing.
  metal_compute::SharedBuffer load(
      const std::string& name, metal_compute::MetalCompute* mc) const;

  // Zero-copy variant: return a READ-ONLY SharedBuffer that is a view into
  // the mmapped shard (newBufferWithBytesNoCopy over the whole shard, then a
  // subview at the tensor's offset) -- NO memcpy, so the tensor stays on clean
  // file-backed pages the OS reclaims under pressure and re-faults from disk.
  //
  // The returned buffer MUST be treated as read-only (the mapping is PROT_READ)
  // and is valid only while THIS MetalLlamaWeights stays alive (it owns the
  // mmap) -- so callers that use it must retain the weights for the model's
  // lifetime. Falls back to a copying load() (allocating an owned buffer) for
  // GGUF-backed tensors, when the no-copy wrap fails, or when the tensor's file
  // offset is not GPU-bindable (misaligned / out of range) -- so the result is
  // always correct, just not always zero-copy. Empty if the tensor is missing.
  metal_compute::SharedBuffer load_mapped(
      const std::string& name, metal_compute::MetalCompute* mc) const;

private:
  MetalLlamaWeights() = default;

  // One mmapped safetensors file. A checkpoint is one shard (single
  // file) or several (sharded). data_start is the byte offset of the
  // tensor data blob (8 + header_len), to which TensorInfo::offset is
  // relative.
  struct Shard {
    int           fd = -1;
    void*         base = nullptr;
    std::size_t   map_size = 0;
    std::uint64_t data_start = 0;
  };

  // mmap one safetensors file and merge its tensors into _tensors,
  // tagging each with this shard's index. Returns false on any error.
  bool map_shard_(const std::string& safetensors_path);

  std::vector<Shard>                          _shards;
  // Lazily-created whole-shard newBufferWithBytesNoCopy wrappers, one per
  // _shards entry (empty until load_mapped() first wraps that shard). Declared
  // after _shards so it is destroyed FIRST -- the MTL buffers release before
  // ~MetalLlamaWeights munmaps the pages they reference. `mutable`: the wraps
  // are a lazy cache built by the const load_mapped().
  mutable std::vector<metal_compute::SharedBuffer> _shard_maps;
  std::unordered_map<std::string, TensorInfo> _tensors;
  // Non-null when this checkpoint was loaded from a `.gguf`; owns the
  // GgufFile + converter and backs the shard==-2 tensors in load().
  std::unique_ptr<GgufBacking>                _gguf;
};

}  // namespace vpipe::genai

#endif
