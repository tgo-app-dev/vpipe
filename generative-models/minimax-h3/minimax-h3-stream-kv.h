#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_STREAM_KV_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_STREAM_KV_H

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <future>
#include <memory>
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
//
// ON DISK (Storage::disk). The cache above is 50 layers x the retained
// rows: 17.4 GB at 832x480, more than a 24 GB Mac keeps beside the
// streamed DiT, so the compressor took it between forwards and every
// forward paid to decompress it -- MEASURED, 34.5 s a chunk forward, ~10
// GB compressed and 3 GB of swap; on disk in 8 bits, 7.9 s, the compressor
// under 2 GB. Yet a forward reads each layer's rows
// exactly once, in layer order, which is the access a file serves best.
// So each COMMITTED CHUNK is a file of its own (unlinked at creation),
// one page-aligned record per layer -- K then V, [heads, rows, head_dim]
// -- written once by the commit that made it and dropped whole when the
// retention lets it go; a chunk aged to its video keeps its audio bytes
// in the file and the gather skips them. A forward holds two SLOTS on the
// GPU: layer L's records are read (uncached, F_NOCACHE) into slot L % 2
// while the GPU runs layer L - 1, and a clean forward's new chunk is
// written from the end of the slot once its layer has run. The forward
// waits for each block (begin_forward / acquire / prefetch / layer_done
// / end_forward below). In 8 bits (Storage::bits) the records are
// affine groups of 64 (kv_quant_u8g64) and the gather dequantizes them
// into the bf16 staging; the forward's own rows stay bf16.
class StreamKv {
public:
  struct Segment;    // a committed chunk's file (disk)

  struct Commit {
    int audio = 0;          // rows physically present
    int video = 0;
    bool audio_dropped = false;
    std::shared_ptr<Segment> seg;   // disk: its file
    int effective() const { return (audio_dropped ? 0 : audio) + video; }
  };

  // Where the committed chunks live.
  struct Storage {
    // In files, read a layer ahead into two GPU slots. False: every
    // layer's pair on the GPU for the run (upstream's own layout).
    bool disk = false;
    // 16: bf16, as computed. 8: affine 8-bit groups of 64 along head_dim
    // (kv_quant_u8g64), dequantized into the staging. Disk only.
    int bits = 16;
    // Disk: the most rows one layer's committed chunks take AS STORED --
    // the audio a file keeps after its chunk aged included -- and the most
    // media rows one commit adds. They size the slots.
    int slot_rows = 0;
    int new_rows = 0;
  };

  // Allocate `layers` K/V pairs of `capacity` rows. Returns false, with
  // the reason, when the GPU cannot hold them. The bytes are held until
  // clear() or destruction.
  bool init(metal_compute::MetalCompute* mc, int layers, int heads,
            int head_dim, int capacity, std::string* err);
  // The same, stored as `st` says: on disk, two slots instead of the
  // layers' pairs. `capacity` still bounds the rows a forward sees.
  bool init(metal_compute::MetalCompute* mc, int layers, int heads,
            int head_dim, int capacity, const Storage& st,
            std::string* err);
  ~StreamKv();
  bool ready() const { return _layers > 0; }
  bool on_disk() const { return _disk; }
  int bits() const { return _bits; }
  void clear();          // forget every commit; buffers kept

  // Rows a forward's attention sees from the cache.
  int rows() const;
  int capacity() const { return _cap; }
  int layers() const { return _layers; }
  int commits() const { return (int)_commits.size(); }
  int audio_rows() const;
  int video_rows() const;
  // What the cache holds on the GPU: the K/V pairs (or the two slots)
  // plus staging of its own.
  std::size_t bytes() const;
  // What its files hold now (disk).
  std::size_t disk_bytes() const;
  // Bytes init() would allocate for this geometry, staging included at
  // `max_seq` rows past the cache.
  static std::size_t bytes_for(int layers, int heads, int head_dim,
                               int capacity, int max_seq);
  // The same stored as `st` says (on disk: the slots and the staging).
  static std::size_t bytes_for(int layers, int heads, int head_dim,
                               int capacity, int max_seq,
                               const Storage& st);

  // Make room in staging for a `seq`-row forward. Grows only.
  bool ensure_staging(int seq);
  // Staging carved out of memory the caller already holds -- the DiT's
  // q|k|v scratch, dead from the q / k / v split until the feed-forward,
  // which is exactly the gather -> attention -> commit window staging
  // lives in. `buf` holds both halves (K first), each `rows` rows per
  // head; a staging of the stream's own is released. Taken again before
  // every forward, so a rebuilt scratch never leaves it stale.
  bool adopt_staging(const metal_compute::SharedBuffer& buf, int rows);
  // Bytes a `seq`-row forward's staging takes (K and V).
  std::size_t staging_bytes(int seq) const;
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
    // Which commits it keeps, in order (-1 the new one; video_only: aged
    // to its video), and the new chunk: its staging row and size.
    struct Pick { int index = -1; bool video_only = false; };
    std::vector<Pick> picks;
    int new_src = 0, new_audio = 0, new_video = 0;
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
  // Host, once the clean forward has COMPLETED on the GPU (and, on disk,
  // end_forward has joined its writes).
  void finish_commit(const CommitPlan& plan);

  // ---- a forward's I/O, on disk (each a no-op in memory) -------------
  //
  // Before anything of the forward is encoded: the slots laid out for
  // the commits held now, a clean forward's file opened (`plan`), and
  // layer 0's read started.
  bool begin_forward(const CommitPlan* plan, std::string* err);
  // Before layer `layer`'s attention is encoded: its records in their
  // slot -- the read issued ahead joined, or made now.
  bool acquire(int layer, std::string* err);
  // Layer `layer`'s read, started into its slot. Issue it once the GPU
  // has finished the layer that slot last served (layer - 2).
  void prefetch(int layer);
  // The GPU has finished `layer`: a clean forward's new rows go from the
  // slot to their file.
  void layer_done(int layer);
  // Every read and write of the forward joined. False if one failed.
  bool end_forward(std::string* err);
  // What this forward read and wrote, and how long acquire() waited.
  struct IoStats {
    std::size_t read = 0, written = 0;
    double wait_ms = 0.0;
  };
  IoStats io_stats() const { return _stats; }

private:
  // One [heads, n, head_dim] row block between two head-major buffers,
  // each from a byte offset.
  void copy_rows_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::ComputeFunction& copy_rect,
                  const metal_compute::SharedBuffer& src,
                  std::size_t src_off, int src_rows, int src_row0,
                  const metal_compute::SharedBuffer& dst,
                  std::size_t dst_off, int dst_rows, int dst_row0,
                  int n) const;
  // A layer's record of `n` rows, as stored: K's part (V's is the same),
  // and both rounded up to a page -- the file's stride and the slot's.
  std::size_t part_bytes_(int n) const;
  std::size_t record_bytes_(int n) const;
  // In 8 bits: where a part's scales and minimums start.
  std::size_t q_bytes_(int n) const;
  std::size_t g_bytes_(int n) const;
  // Rows [row0, row0 + n) of a stored part (at `off` in `buf`, `rows`
  // rows a head) into staging rows from `dst_row0`, either width.
  void load_part_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::ComputeFunction& copy_rect,
                  const metal_compute::SharedBuffer& buf, std::size_t off,
                  int rows, int row0, int n,
                  const metal_compute::SharedBuffer& dst, int dst_rows,
                  int dst_row0) const;
  // And back: staging rows [src_row0, src_row0 + n) into a stored part.
  void store_part_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::ComputeFunction& copy_rect,
                   const metal_compute::SharedBuffer& src, int src_rows,
                   int src_row0, int n,
                   const metal_compute::SharedBuffer& buf,
                   std::size_t off) const;
  // Whatever slot `s` still has in flight, finished.
  bool join_(int s);

  metal_compute::MetalCompute* _mc = nullptr;
  int _layers = 0, _heads = 0, _hd = 0, _cap = 0;
  std::vector<metal_compute::SharedBuffer> _k, _v;
  metal_compute::SharedBuffer _sk, _sv;
  int _staging_rows = 0;
  std::vector<Commit> _commits;

  // ---- on disk -------------------------------------------------------
  bool _disk = false;
  int _bits = 16;
  std::size_t _page = 16384;
  metal_compute::ComputeFunction _fn_quant, _fn_dequant;
  metal_compute::SharedBuffer _slot[2];
  // This forward's slot layout: commit i's records at _slot_off[i], the
  // new chunk's at _new_off.
  std::vector<std::size_t> _slot_off;
  std::size_t _new_off = 0;
  int _slot_layer[2] = {-1, -1};
  std::shared_ptr<Segment> _pending;   // the clean forward's file
  int _pending_rows = 0;
  std::future<bool> _io[2];            // each slot's reads and writes
  IoStats _stats;
};

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
