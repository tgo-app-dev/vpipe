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
  // True when this checkpoint came from a `.gguf`, i.e. every tensor is
  // produced by the on-demand converter rather than copied from a mapped
  // shard. Callers that want to parallelise loads need this: the
  // converter is not re-entrant, while the safetensors memcpy is.
  bool is_gguf() const noexcept;

  // How much of this checkpoint is 16-byte aligned, which is what a
  // Metal buffer offset must be for a tensor to be used where it lies.
  //
  // Reported at OPEN rather than only where it bites. load_mapped() has
  // said so for a while, but only a caller asking for Residency::Mapped
  // ever reaches it -- and the models that suffer most are the ones
  // reading Copied, which never do. A misaligned pack costs them a copy
  // per tensor per read with nothing in the log to connect the cost to
  // the cause.
  struct Alignment {
    std::size_t tensors    = 0;
    std::size_t misaligned = 0;
    int         shards     = 0;
    int         bad_shards = 0;   // shards whose data section is itself off
  };
  Alignment alignment() const;
  const TensorInfo* info(const std::string& name) const;

  // All tensor names in the checkpoint (unordered). Used by the model
  // quantizer to enumerate + classify every tensor.
  std::vector<std::string> tensor_names() const;

  // Where a load's wall time went. The two halves answer different
  // questions for a streaming model: `alloc` is the SharedBuffer, which
  // on a fresh anonymous allocation includes the kernel zero-filling it,
  // and `fetch` is the memcpy out of the read-only shard mmap -- which
  // is a pure copy when the source pages are already in the file cache
  // and a demand-paged DISK read when they are not.
  //
  // The distinction decides what a weight prefetch can buy. Real IO
  // overlaps GPU work on another thread; a memcpy on unified memory
  // mostly relocates memory-bandwidth contention instead of hiding it.
  // The two are fused inside the memcpy (the fault happens there), so
  // this does not try to separate them by bracketing -- the achieved
  // rate does that: ~1.5 GB/s is a disk, ~12 GB/s is the page cache.
  struct LoadCost {
    double alloc_ms = 0.0;
    double fetch_ms = 0.0;
  };

  // Read a tensor's RAW bytes straight from the shard into memory the
  // CALLER owns, with pread(2) -- no allocation, and not a byte through
  // the shard's mmap.
  //
  // The point is the read shape, not the destination. MEASURED on an M5
  // over a MiniMax-H3 shard, 206 MB per block, arms interleaved and their
  // ORDER rotated: a fresh allocation plus a memcpy out of the mapping
  // runs at 0.86-1.48 GB/s and a pread into a warm buffer at 6.7-6.9
  // GB/s. The variance is the other half of it -- the mapped path's rate
  // moves by 2x between rounds while pread does not, and a block-streamed
  // forward turns that into GPU occupancy that will not sit still.
  //
  // That also corrects what the streaming rate was taken to mean: ~1.5
  // GB/s is not this machine's disk, it is the cost of demand-faulting a
  // mapping. The same files read at 6.8 GB/s through this path.
  //
  // Source alignment does not matter -- measured flat across page-,
  // 8-byte- and mid-page-aligned tensor offsets, which is what makes this
  // usable on safetensors at all. Nor does read size punish the small
  // tensors: 27 us at 10 KB, so a block's norms and scales can come the
  // same way as its matrices.
  //
  // `uncached` sets F_NOCACHE, which does NOT make the read faster (a
  // cached pread measured identically) -- it keeps the read from growing
  // the file cache. For a model that re-reads its whole checkpoint every
  // forward that cache is pure pressure, and it is the pressure that
  // squeezes out the blocks a resident set is trying to keep.
  //
  // Returns false, having written NOTHING, when the tensor is missing,
  // when it is not backed by a plain safetensors shard (GGUF is converted
  // rather than copied, so there are no raw bytes to place), or when
  // `cap` is smaller than the tensor. The caller then takes read_into()
  // or load(), which are always correct and merely slower.
  //
  // No conversion, deliberately: the bytes land exactly as the file holds
  // them. A caller wanting a dtype the checkpoint does not store has to
  // convert after the fact, which is only in-place-able when the widths
  // agree.
  bool pread_into(const std::string& name, void* dst, std::size_t cap,
                  bool uncached = true) const;

  // Allocate a SharedBuffer and copy the tensor's bytes into it.
  // Returns an empty SharedBuffer if the tensor is missing. `cost`, when
  // non-null, receives the split above (added to, not overwritten).
  metal_compute::SharedBuffer load(
      const std::string& name, metal_compute::MetalCompute* mc,
      LoadCost* cost = nullptr) const;

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

  // Re-read a tensor into memory the CALLER already owns. The point is
  // in-place refresh: a buffer whose pages the kernel reclaimed while
  // parked keeps its allocation (and therefore its GPU address and every
  // alias handed out from it), so restoring it has to write back into
  // THAT pointer rather than allocate a replacement. Handles GGUF-backed
  // tensors too -- the converter already writes through a caller-supplied
  // destination. Returns false if the tensor is missing or `cap` is
  // smaller than its bytes.
  bool read_into(const std::string& name, void* dst,
                 std::size_t cap) const;

private:
  MetalLlamaWeights() = default;

  // One mmapped safetensors file. A checkpoint is one shard (single
  // file) or several (sharded). data_start is the byte offset of the
  // tensor data blob (8 + header_len), to which TensorInfo::offset is
  // relative.
  struct Shard {
    int           fd = -1;
    // A SECOND descriptor on the same file, opened F_NOCACHE, for
    // pread_into(). Separate from `fd` because F_NOCACHE is a property of
    // the descriptor and `fd` owns the mmap every other reader goes
    // through -- a checkpoint that is both streamed and mapped must not
    // have one mode decide the other. Opened eagerly at map time so the
    // streaming path needs no lazy init and is therefore thread-safe
    // without a lock.
    int           fd_stream = -1;
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
  // Has this shard already reported that zero-copy mapping is off for
  // it? One shard's data offset decides EVERY tensor in it, so the first
  // fallback is the whole story and the other four thousand would be
  // noise. `mutable` for the same reason as _shard_maps: load_mapped is
  // const and this is bookkeeping, not state a caller can observe.
  mutable std::vector<bool>                       _shard_unmappable_said;
  std::unordered_map<std::string, TensorInfo> _tensors;
  // Non-null when this checkpoint was loaded from a `.gguf`; owns the
  // GgufFile + converter and backs the shard==-2 tensors in load().
  std::unique_ptr<GgufBacking>                _gguf;
};

}  // namespace vpipe::genai

#endif
