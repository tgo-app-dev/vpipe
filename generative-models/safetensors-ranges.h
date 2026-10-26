#ifndef VPIPE_GENERATIVE_MODELS_SAFETENSORS_RANGES_H
#define VPIPE_GENERATIVE_MODELS_SAFETENSORS_RANGES_H

// BYTE RANGES OF A SAFETENSORS CHECKPOINT'S TENSORS, read with pread(2) on
// descriptors of its own -- a slice of one tensor without the rest of it:
// one expert's rows of a stacked [experts, rows, cols] MoE weight.
//
// Reads go UNCACHED (F_NOCACHE): what they fill is a cache of its own (the
// expert cache), and the page cache keeping a second copy would compete with
// it for the same memory. F_NOCACHE bypasses the buffer cache only on a
// page-aligned offset -- an unaligned read silently falls back to the cached
// path -- so every read is widened to whole pages into a staging buffer of
// the calling thread and the wanted bytes copied out.
//
// The tensor table is the checkpoint's own (MetalLlamaWeights); the files
// are the shards the index names (or the one model.safetensors).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpipe::genai {

class MetalLlamaWeights;

class SafetensorsRanges {
public:
  // `cached`: read THROUGH the page cache instead -- plain descriptors, a
  // read straight into its destination. For small reads the cache should
  // keep: the rows of a table indexed by token (the n-gram embedding), a
  // few hundred bytes each, many of them read again.
  // `want`, if given, names the tensors this reader serves, and only their
  // shards are opened. A reader keeps a descriptor a shard, and a process
  // starts on macOS with room for 256: an 82-shard checkpoint read by the
  // weight set and two readers of every shard did not open. Opening also
  // raises the process's soft limit toward its hard one when the shards
  // need it.
  static std::unique_ptr<SafetensorsRanges> open(
      const std::string& dir, const MetalLlamaWeights& src,
      std::string* err = nullptr, bool cached = false,
      const std::function<bool(std::string_view)>& want = {});
  ~SafetensorsRanges();

  // `n` bytes of tensor `name`, from byte `off` of its data, into `dst`.
  // Thread-safe.
  bool read(const std::string& name, std::size_t off, std::size_t n,
            void* dst) const;
  // The tensor's size in bytes, 0 when there is none.
  std::size_t nbytes(const std::string& name) const;
  // An uncached read widened to `bytes` (a power of two, at most the
  // page) instead of the page: what a reader of many small ranges wants
  // -- MEASURED on the M4 Pro's internal SSD, random reads of a few
  // hundred bytes at 64 in flight 316K a second at 4 KB, 167K at the
  // page's 16 KB (and 162K cached). Set before any read.
  void set_alignment(std::size_t bytes) { _align = bytes; }
  // Several tensors in ONE read, when they lie together in one file (a
  // checkpoint written in name order stores an expert's tensors side by
  // side): `at[i]` points at names[i] inside `stage`. False -- nothing
  // read -- when they are in different files or spread wider than 1.25x
  // their bytes; the caller then reads them one by one. A single 1.5 MB
  // read takes a third of the time six reads of its pieces do.
  bool read_together(const std::vector<std::string>& names,
                     std::vector<std::uint8_t>& stage,
                     std::vector<const std::uint8_t*>& at) const;

private:
  SafetensorsRanges() = default;
  struct Where {
    int file = -1;
    std::uint64_t at = 0;            // absolute offset of the tensor's data
    std::uint64_t n = 0;
  };
  std::vector<int> _fds;
  std::unordered_map<std::string, Where> _where;
  bool _cached = false;
  std::size_t _align = 0;          // 0: the page
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_SAFETENSORS_RANGES_H
