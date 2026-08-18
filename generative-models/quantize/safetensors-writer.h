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
  // A tensor whose SIZE is not a multiple of 16, held back to the end of
  // its shard. See the note on `deferred` below.
  struct Deferred {
    Entry                     e;
    std::vector<std::uint8_t> bytes;
  };
  struct Shard {
    std::string        tmp_path;
    int                fd = -1;         // temp data file (append)
    std::uint64_t      size = 0;        // running data bytes
    std::vector<Entry> entries;
    // Tensors are packed CONTIGUOUSLY, so one whose size is not a
    // multiple of 16 shifts every tensor after it off the 16-byte
    // boundary the zero-copy read path needs -- and the offenders are
    // small and numerous. MEASURED on a quantized Gemma-4 12B: one
    // 2-byte `layer_scalar` per layer, 48 of them interleaved with the
    // weights, plus four JSON asset blobs -- 53 tensors that between
    // them misaligned 85% of the file.
    //
    // Writing them LAST fixes every tensor before them and costs
    // nothing: the format says only that the byte ranges tile the blob,
    // not in which order the header lists them. Gaps would be the
    // obvious alternative and are not available -- the reference
    // safetensors reader validates that the ranges are contiguous.
    //
    // Bounded, because this is the one place the writer stops being
    // streaming: only a SMALL odd tensor is held in RAM. A large one is
    // written in place and the tensors after it are misaligned, which
    // the loader reports at read time.
    std::vector<Deferred> deferred;
    std::uint64_t         deferred_bytes = 0;
  };
  // Per-tensor and per-shard ceilings on what `deferred` will hold.
  static constexpr std::uint64_t kDeferMax      = 1ull << 20;   // 1 MB
  static constexpr std::uint64_t kDeferTotalMax = 64ull << 20;  // 64 MB

  bool ensure_shard_();
  bool finalize_shard_(int idx, int total, std::string& out_name);

  std::string        _out_dir;
  std::uint64_t      _shard_max;
  std::vector<Shard> _shards;
  bool               _closed = false;
};

}  // namespace vpipe::genai

#endif
