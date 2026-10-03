#ifndef VPIPE_GENERATIVE_MODELS_TRAIN_SAMPLE_CACHE_H
#define VPIPE_GENERATIVE_MODELS_TRAIN_SAMPLE_CACHE_H

// The encoded dataset on disk: what LoRA training keeps between runs so
// a caption or a picture is encoded once, ever.
//
// A cache is one directory per (dataset, model) -- the training-dataset
// stage names it -- holding safetensors SHARDS and an index.json. Every
// entry is keyed by a hash of whatever changes its bytes (the caption's
// final text; the picture's file identity, bucket and flip), so a re-run
// looks its work up instead of encoding it, and an edit to one picture
// re-encodes that picture alone.
//
// The index is self-sufficient: it records each entry's shard, absolute
// byte offset, dtype, shape and sideband, so a read is one pread() with no
// header to parse. Shards are written whole (tmp + rename) and the index
// after them, the same way, so a kill leaves at worst a shard the index
// does not name, which the next write replaces.
//
// One writer at a time: two runs filling one cache at once is not
// supported.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
namespace genai {
namespace train {

// FNV-1a 64 over the parts (each followed by a separator), as 16 hex
// digits. Stable across runs and machines.
std::string cache_key(std::initializer_list<std::string_view> parts);

class SampleCache {
 public:
  struct Entry {
    int shard = -1;
    std::uint64_t offset = 0;      // absolute, in the shard file
    std::uint64_t bytes = 0;
    std::string dtype;             // "BF16", "F32"
    std::vector<std::int64_t> shape;
    std::string sideband;          // JSON, or empty
  };

  SampleCache() = default;
  ~SampleCache();
  SampleCache(const SampleCache&) = delete;
  SampleCache& operator=(const SampleCache&) = delete;

  // Opens `dir`, creating it, and reads its index when there is one. An
  // index this code cannot read is treated as an empty cache (and is
  // replaced by the next flush) rather than failing the run.
  bool open(const std::string& dir, std::string* err);
  // Drops everything staged and closes the shards.
  void close();
  bool is_open() const noexcept { return !_dir.empty(); }
  const std::string& dir() const noexcept { return _dir; }

  // The keys an index names, without opening anything for writing -- what
  // the dataset stage asks to decide which encodes it can skip.
  static std::set<std::string> keys_in(const std::string& dir);

  bool has(const std::string& key) const;
  const Entry* find(const std::string& key) const;
  std::size_t size() const noexcept { return _entries.size(); }

  // Staged in memory until flush(); a key already present is replaced.
  void put(const std::string& key, std::string dtype,
           std::vector<std::int64_t> shape, std::vector<std::uint8_t> bytes,
           std::string sideband);
  std::size_t pending_bytes() const noexcept { return _pending_bytes; }
  // Writes the staged entries as one new shard, then the index.
  bool flush(std::string* err);

  // An entry's bytes into `dst` (at least entry->bytes long).
  bool read(const std::string& key, void* dst, std::size_t cap,
            std::string* err) const;

 private:
  struct Pending {
    std::string key;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
    std::string sideband;
  };
  std::string _dir;
  std::vector<std::string> _shards;           // file names, by index
  std::map<std::string, Entry> _entries;
  std::vector<Pending> _pending;
  std::size_t _pending_bytes = 0;
  mutable std::vector<int> _fds;              // open shards, -1 if not

  bool write_index_(std::string* err) const;
};

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#endif
