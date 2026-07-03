#ifndef VPIPE_GENAI_QUANTIZE_SAFETENSORS_WRITER_H
#define VPIPE_GENAI_QUANTIZE_SAFETENSORS_WRITER_H

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai {

// Streaming MLX-affine safetensors writer.
//
// Tensors are appended one at a time; their bytes are streamed straight to a
// per-shard temp file, so peak host RAM is ~one tensor regardless of model
// size (the >system-memory requirement). Tensors are rolled into shards by a
// byte budget. close() finalises each shard as
//   [u64 LE header_len][JSON header][data blob]
// (data_offsets relative to the blob, matching the MetalLlamaWeights reader
// byte-for-byte) and writes model.safetensors.index.json (weight_map
// name -> shard filename).
//
// dtype strings are safetensors-canonical: "U32", "F16", "BF16", "F32".
class SafetensorsWriter {
public:
  explicit SafetensorsWriter(std::string out_dir,
                             std::uint64_t shard_max_bytes = 5ull << 30);
  ~SafetensorsWriter();
  SafetensorsWriter(const SafetensorsWriter&)            = delete;
  SafetensorsWriter& operator=(const SafetensorsWriter&) = delete;

  // Stage one tensor into the current shard. `data`/`nbytes` are copied to
  // the shard temp immediately -- the pointer need not outlive the call.
  // Returns false on an IO error or after close().
  bool add(const std::string& name, const std::string& dtype,
           const std::vector<std::int64_t>& shape, const void* data,
           std::uint64_t nbytes);

  // Finalise every shard and write the index. Idempotent; returns false on
  // any IO error. The writer is unusable afterwards.
  bool close();

private:
  struct Entry {
    std::string               name;
    std::string               dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t             offset;   // within the shard data blob
    std::uint64_t             nbytes;
  };
  struct Shard {
    std::string        tmp_path;
    int                fd = -1;         // temp data file (append)
    std::uint64_t      size = 0;        // running data bytes
    std::vector<Entry> entries;
  };

  bool ensure_shard_();
  bool finalize_shard_(int idx, int total, std::string& out_name);

  std::string        _out_dir;
  std::uint64_t      _shard_max;
  std::vector<Shard> _shards;
  bool               _closed = false;
};

}  // namespace vpipe::genai

#endif
