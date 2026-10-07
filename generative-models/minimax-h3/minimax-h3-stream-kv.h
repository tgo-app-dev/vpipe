#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_STREAM_KV_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_STREAM_KV_H

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

// The persistent CLEAN audio/video key-value cache of TaoMate-H3's
// streaming method (upstream streaming/cache.py CleanAVKVCache), one
// pair per main block.
//
// What it holds: the post-QK-norm, post-RoPE keys and values of the
// media rows of every chunk COMMITTED so far -- the rows the chunk's
// clean forward (t = 1) produced -- trimmed after each commit to
//   * chunk 0's VIDEO rows, for the whole run (the "sink"), and
//   * the two most recent chunks, audio and video,
// with every audio row dropped once each 12 requests. Text rows are
// never cached: a chunk's prompt is its own, and text attends to text.
//
// How a forward uses it, per main block (see the attention in
// MetalMiniMaxH3Transformer::forward):
//
//   gather   staging = [effective cache rows | this forward's K/V rows]
//   attend   text rows  -> staging[cache + text]
//            media rows -> staging[all of it]
//   commit   (clean forward only) the block's buffer is rewritten from
//            staging as the rows that SURVIVE this commit plus the new
//            chunk's media rows -- after the attention has read the old
//            contents, so trimming and appending never need more than
//            the retained set plus one chunk.
//
// Layout: per layer, K and V are [heads, capacity, head_dim] bf16, rows
// valid from 0. A committed chunk's rows sit as [audio | video], the
// packed order. Dropping audio does not move bytes until the next
// commit rewrites the buffer; until then the gather simply skips them.
class StreamKv {
public:
  struct Commit {
    int audio = 0;          // rows physically present
    int video = 0;
    bool audio_dropped = false;
    int effective() const { return (audio_dropped ? 0 : audio) + video; }
  };

  // Allocate `layers` K/V pairs of `capacity` rows. Returns false, with
  // the reason, when the GPU cannot hold them. The bytes are held until
  // clear() or destruction.
  bool init(metal_compute::MetalCompute* mc, int layers, int heads,
            int head_dim, int capacity, std::string* err);
  bool ready() const { return !_k.empty(); }
  void clear();          // forget every commit; buffers kept

  // Rows a forward's attention sees from the cache.
  int rows() const;
  int capacity() const { return _cap; }
  int layers() const { return (int)_k.size(); }
  int commits() const { return (int)_commits.size(); }
  int audio_rows() const;
  int video_rows() const;
  // What the cache holds on the GPU: the K/V pairs plus staging.
  std::size_t bytes() const;
  // Bytes init() would allocate for this geometry, staging included at
  // `max_seq` rows past the cache.
  static std::size_t bytes_for(int layers, int heads, int head_dim,
                               int capacity, int max_seq);

  // Make room in staging for a `seq`-row forward. Grows only.
  bool ensure_staging(int seq);
  const metal_compute::SharedBuffer& staging_k() const { return _sk; }
  const metal_compute::SharedBuffer& staging_v() const { return _sv; }
  // Rows per head in staging for this forward: rows() + seq. The head
  // stride the attention binds.
  int staging_rows(int seq) const { return rows() + seq; }

  // The upstream drop_audio_history, applied lazily: from now on the
  // gather skips every committed audio row, and the next commit
  // compacts them away. Returns the rows dropped.
  int drop_audio();

  // A commit's retention, decided BEFORE its forward is encoded so the
  // per-layer write-back and the host bookkeeping cannot disagree.
  struct CommitPlan {
    struct Range { int src = 0, n = 0; };   // staging rows
    std::vector<Range> ranges;              // concatenated into the layer
    std::vector<Commit> after;              // the commit list it leaves
    int rows_after = 0;
  };
  // Plan committing a chunk of `audio` + `video` media rows that sit at
  // staging rows [rows() + n_text, rows() + n_text + audio + video) --
  // i.e. the forward's [text | audio | video] layout. False when the
  // retained set would not fit the capacity.
  bool plan_commit(int n_text, int audio, int video, CommitPlan* out,
                   std::string* err) const;

  // GPU, per main block: fill staging from the cache and the forward's
  // head-major [heads, seq, head_dim] keys and values.
  void encode_gather(metal_compute::ComputeEncoder& enc,
                     const metal_compute::ComputeFunction& copy_rect,
                     int layer, const metal_compute::SharedBuffer& kh,
                     const metal_compute::SharedBuffer& vh, int seq) const;
  // GPU, per main block of a CLEAN forward, after its attention: rewrite
  // the layer's pair from staging per `plan`.
  void encode_commit(metal_compute::ComputeEncoder& enc,
                     const metal_compute::ComputeFunction& copy_rect,
                     int layer, const CommitPlan& plan, int seq) const;
  // Host, once the clean forward has COMPLETED on the GPU.
  void finish_commit(const CommitPlan& plan);

private:
  // One [heads, n, head_dim] row block between two head-major buffers.
  void copy_rows_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::ComputeFunction& copy_rect,
                  const metal_compute::SharedBuffer& src, int src_rows,
                  int src_row0, const metal_compute::SharedBuffer& dst,
                  int dst_rows, int dst_row0, int n) const;

  metal_compute::MetalCompute* _mc = nullptr;
  int _heads = 0, _hd = 0, _cap = 0;
  std::vector<metal_compute::SharedBuffer> _k, _v;
  metal_compute::SharedBuffer _sk, _sv;
  int _staging_rows = 0;
  std::vector<Commit> _commits;
};

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
