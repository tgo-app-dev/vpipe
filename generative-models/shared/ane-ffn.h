#ifndef VPIPE_GENERATIVE_MODELS_SHARED_ANE_FFN_H
#define VPIPE_GENERATIVE_MODELS_SHARED_ANE_FFN_H

// The block feed-forward on the Apple Neural Engine, beside the GPU.
//
// A SwiGLU feed-forward is row-independent, so a DiT can hand the ANE a
// contiguous band of its rows and keep the rest on the GPU, the two
// running concurrently -- exact, with no seam. This is the family-neutral
// half of that: ONE runtime-weight module per model (its weights are
// IOSurface-bound inputs, staged per block, so it compiles once per shape
// and its memory is fixed), the staging of dense bf16 or 4/8-bit affine
// weights into it, the chunked predict over the ANE's rows, and the
// balancer that moves the split toward whichever engine is waiting.
//
// What stays with the family: WHICH blocks are eligible, where its
// feed-forward weights are, and the command-stream choreography around
// the split -- commit the attention without waiting, stage under it,
// drain, begin(), run the GPU's rows, finish().

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/ane-module.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
class AneWorker;
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }

namespace genai {

// One projection's weights as staging reads them: a dense bf16 matrix, or
// a 4/8-bit affine one (u32 codes, bf16 scales and biases per group), in
// the checkpoint's [out,in] layout. `stride` and `offset` select the
// source rows one slot row comes from, so a fused [gate; up] matrix is
// read as two halves (stride 1, offset 0 / ffn) and an interleaved
// [g0,u0,g1,u1,...] one as every other row (stride 2, offset 0 / 1).
struct AneFfnSource {
  const metal_compute::SharedBuffer* w      = nullptr;
  const metal_compute::SharedBuffer* codes  = nullptr;
  const metal_compute::SharedBuffer* scales = nullptr;
  const metal_compute::SharedBuffer* qbias  = nullptr;
  bool        quantized = false;
  int         bits      = 0;
  std::size_t stride    = 1;
  std::size_t offset    = 0;
  // The projection's bias, bf16, indexed like its rows: element
  // r * stride + offset belongs to slot row r. Read only when the module was
  // created with `biases`; null there stages zeros.
  const metal_compute::SharedBuffer* bias = nullptr;
  // WHERE in the module's weight slots this source lands. By default (-1)
  // source i fills slot i whole. A multi-part matmul -- Krea-2's separate
  // q, k, v and gate stacked into one [out,in] slot -- sets `slot` to the
  // shared slot, `slot_row` to its first row there and `rows` to its own
  // output width.
  int         slot     = -1;
  std::size_t slot_row = 0;
  std::size_t rows     = 0;          // 0: the whole slot past slot_row

  // Low-rank adapters merged into the staged rows: row += scale * B[br] A,
  // with br = r * b_stride + b_offset for slot row r. A [rank, cols] and B
  // [rows, rank], both bf16. B's rows follow the ADAPTER's layout, which is
  // not always the weight's: MiniMax-H3's fc1 adapter is [gate; up] like
  // fc1 itself (so it takes fc1's stride and offset), while Krea-2 keeps
  // separate gate and up adapters (1, 0) even beside a fused weight. The
  // GPU applies the same adapters as side GEMMs on its rows; the ANE's
  // module has no adapter inputs, so its rows get them folded into the
  // weights instead.
  struct Delta {
    const metal_compute::SharedBuffer* a = nullptr;
    const metal_compute::SharedBuffer* b = nullptr;
    int   rank  = 0;
    float scale = 0.0f;
    std::size_t b_stride = 1;
    std::size_t b_offset = 0;
    // A BANDED pair (lora::Factors::banded): `rank` is then the PER-PART
    // rank B holds, A has parts * rank rows, and B row `br` contracts
    // against A's rows [p*rank, p*rank + rank), p = (br / group) % parts.
    // 0 = an ordinary dense pair.
    int group = 0;
    int parts = 1;
  };
  static constexpr int kMaxDeltas = 2;
  Delta delta[kMaxDeltas];
  int   deltas = 0;

  bool stageable() const noexcept;
};

class AneFeedForward {
 public:
  // The module's row count; the host calls it as many times as the share
  // needs. 2048 because the ANE's rate saturates by ~2048 rows.
  static constexpr int kChunkDefault = 2048;
  // The in-graph tile. Runtime operands measured ~10% faster at 1024 than
  // at 2048 (11.2 vs 10.2 TOPS on a Krea-2 feed-forward), where a module
  // with baked weights is fastest at 2048. A width that does not divide
  // takes a narrower last tile.
  static constexpr int kTile = 1024;

  struct Options {
    const SessionContextIntf*    session = nullptr;
    metal_compute::MetalCompute* mc      = nullptr;
    std::string tag = "ane";     // log prefix: the family's name
    int   hidden = 0;
    int   ffn    = 0;            // SwiGLU inner width
    int   seq    = 0;            // rows of the first forward
    // The ANE's share of the rows, rounded down to whole chunks. <= 0
    // means AUTO: open at half and balance from the measured times.
    float rows    = 0.0f;
    int   chunk   = kChunkDefault;
    int   tile    = kTile;
    bool  profile = false;       // a timing line per block, at info
    // The Linears carry biases: the module takes them as three more
    // runtime inputs, staged with the weights.
    bool  biases  = false;
    // ONE projection instead of a feed-forward: y = x W^T with W
    // [ffn, hidden], so `ffn` is its output width. One weight slot, no
    // biases, and no fp16 headroom retry -- a single matmul has no product
    // of two projections to overflow. MiniMax-H3's fused qkv is one.
    bool  matmul  = false;
    // A two-projection GELU (tanh approximation) feed-forward: staged as
    // {up, down}, and the fp16 headroom rides the DOWN projection, since
    // gelu(x / s) is not gelu(x) / s.
    bool  gelu    = false;
  };

  // Builds the module and its weight slots, or explains at warn/info why
  // not and returns null -- the caller then keeps its GPU path.
  static std::unique_ptr<AneFeedForward> create(const Options& o);

  // Bytes the tier holds for a `seq`-row forward (<= 0 books one chunk).
  // ONE module serves every block, so this is fixed however many use it.
  // Three terms, each MEASURED on Krea-2:
  //   slots    the IOSurface weight inputs, one block's fp16 weights W;
  //   staging  CoreML's wired copy of those plus the graph's activations
  //            at one chunk, W + chunk*(3*ffn + 2*hidden)*2 -- fitted to
  //            836 / 953 / 1078 MB against 836 / 938 / 1059 measured at
  //            2048 / 3072 / 4096 rows;
  //   host     ONE chunk's fp16 input and output, reused chunk by chunk
  //            -- so the sequence no longer scales it (kept as a parameter
  //            for callers that pass it).
  static std::size_t runtime_bytes(int hidden, int ffn, int seq,
                                   int chunk) noexcept;
  // The same for a matmul tier, `in` -> `out` wide: the slot, CoreML's
  // copy of it plus one chunk's activations, CoreML's own runtime buffers
  // (~5/8 of the weights, whatever the chunk), and one chunk of host rows
  // at both widths.
  static std::size_t matmul_runtime_bytes(int in, int out, int seq,
                                          int chunk) noexcept;
  // `env`'s value when it names a chunk in [256, 16384], else the default.
  static int chunk_rows(const char* env) noexcept;

  ~AneFeedForward();
  AneFeedForward(const AneFeedForward&)            = delete;
  AneFeedForward& operator=(const AneFeedForward&) = delete;

  // OUT OF LINE, every accessor: this header is installed for plugin
  // families, and an inline read of a private member compiles its offset
  // into the plugin -- a member added later would then be misread by a
  // binary that passed every check.
  int  chunk() const noexcept;
  int  rows_per_block() const noexcept;
  bool timing() const noexcept;

  // runtime_bytes() for a GELU module (Options::gelu): two projections and
  // their biases rather than three. Independent of the sequence, like it.
  static std::size_t gelu_runtime_bytes(int hidden, int ffn,
                                        int chunk) noexcept;
  std::size_t host_bytes() const noexcept;
  AneWorker& worker() const;

  // Dispatch `layer`'s gate, up and down into the slots on the ANE worker
  // and return at once, so the conversion runs under the GPU's attention
  // for the same block. The sources' buffers must outlive join_stage().
  // `group` is the affine group size of a quantized source.
  void stage(int layer, const AneFfnSource& gate, const AneFfnSource& up,
             const AneFfnSource& down, int group);
  // The matmul tier's one projection.
  void stage(int layer, const AneFfnSource& weight, int group);
  // ...or several, stacked along the rows of its one slot (see
  // AneFfnSource::slot_row).
  void stage(int layer, const std::vector<AneFfnSource>& parts, int group);
  // Wait for staging; true when `layer`'s weights are in the slots.
  bool join_stage(int layer);
  // Wait for whatever job is outstanding. For an early exit between
  // stage() and join_stage(), before the block's buffers go away.
  void join();

  // Whether to split this block, asked at its start and before staging.
  //
  // A faster GPU can make the split a LOSS: it adds a mid-block commit and
  // a staging job per block, a fixed cost the rows/ms balancer cannot see,
  // and whether the ANE's rows pay for it depends on the geometry (a
  // MiniMax-H3 clip loses at 124 frames on an M5 Pro and wins at 243). So
  // under an auto share the tier MEASURES both: a kSplit block runs the
  // split as always, a kProbe block commits at the split point with no
  // staging and runs the GPU's feed-forward over every row (report its two
  // times to note_probe()), and kGpu is the family's unsplit path. The
  // better of the two is kept, and the other re-measured now and then. A
  // pinned share always splits.
  enum class Plan { kSplit, kProbe, kGpu };
  Plan plan_block();
  void note_probe(double drain_ms, double gpu_ms);
  // True when the block just planned is measured (kSplit or kProbe) and the
  // one before it ran unsplit. An unsplit block commits nothing mid-block,
  // so on a stack that defers its commits the split point's drain would
  // wait for EVERY such block queued before it -- MEASURED at 1857 ms
  // against a real ~240 on an M5 Pro, which then kept the tier off for the
  // rest of the run. The family commits and waits for what it has queued
  // BEFORE staging and encoding this block when this says so.
  bool needs_barrier() const noexcept;

  // Start the ANE's rows of one feed-forward on the worker: the LAST rows
  // of the `seq` in `in` (bf16), into the same rows of `out` (bf16). `in`
  // may be `out`, and must be readable now -- drain the GPU first. The GPU
  // takes the head rows, [0, seq - returned). Returns the ANE's row count,
  // 0 when it takes none.
  int begin(const metal_compute::SharedBuffer& in,
            const metal_compute::SharedBuffer& out, int seq);
  // The same with the output COLUMNS scattered into several buffers, in
  // order: a multi-part matmul writes each part's rows straight into its
  // own destination (widths must sum to the module's output width).
  struct OutSeg {
    const metal_compute::SharedBuffer* buf = nullptr;
    int width = 0;
    // Row `r` of the input lands at row `row_offset + r` of this buffer --
    // for an output that shares its plane with rows the split does not
    // compute (FLUX.2's image q/k/v sit after the text rows).
    std::size_t row_offset = 0;
  };
  int begin(const metal_compute::SharedBuffer& in,
            const std::vector<OutSeg>& outs, int seq);
  // Join the ANE, move the balance using the GPU's time for its rows, and
  // log the block under `profile`. False when the predict failed, which
  // leaves those rows uncomputed.
  bool finish(int layer, int seq, double gpu_ms, double drain_ms);

 private:
  AneFeedForward() = default;
  void stage_(int layer, std::vector<AneFfnSource> src, int group);

  const SessionContextIntf*       _session = nullptr;
  metal_compute::MetalCompute*    _mc      = nullptr;
  std::string                     _tag;
  int _hidden = 0, _ffn = 0, _chunk = kChunkDefault;
  bool _matmul = false;
  bool _gelu   = false;      // {up, down}: the headroom rides slot 1 = down
  int  _out_w  = 0;          // output row width: hidden, or ffn for matmul
  std::unique_ptr<AneModule>      _rt;
  std::unique_ptr<AneWeightSlots> _w;
  int  _rows_per_block = 0;
  // WIRED ON OUR BEHALF. CoreML wires the module's program and its
  // IOSurface slots itself, outside the wired pool -- MEASURED ~500 MB at
  // LTX-2.5's 4096x16384 feed-forward, most of it at the first predict --
  // so the growth in system-wide wired from create to that predict, less
  // what the pool itself wired meanwhile, is charged to the pool then and
  // released in the destructor. `_estimate` bounds it from above against
  // other processes moving the same system figure.
  std::size_t _wired0       = 0;
  std::size_t _pool0        = 0;
  std::size_t _estimate     = 0;
  std::size_t _ext_wired    = 0;
  bool        _ext_measured = false;
  // AUTO-BALANCE. The feed-forward runs n_blocks x n_steps times with the
  // same shape, so the split does not have to be known in advance. Off
  // when the graph pins the share: a run that must be reproducible cannot
  // have its arithmetic depend on how busy the machine was.
  bool _auto    = false;
  bool _profile = false;
  int  _chunks  = 0;
  // Running rates the balancer predicts with.
  double _gpu_row_ms = 0.0;
  double _chunk_ms   = 0.0;
  // Written by jobs on the ANE worker; read only after joining it.
  int    _staged   = -1;
  double _stage_ms = 0.0;
  bool   _ok       = false;
  double _t_cin = 0.0, _t_pred = 0.0, _t_cout = 0.0;
  int    _a_rows = 0;                              // this block's
  // ON/OFF: block-section times, from the split point to the end of the
  // feed-forward -- split = drain + GPU half + exposed ANE, probe = drain +
  // GPU over every row -- kept as the MEDIAN of the last kWindow of each,
  // and which mode is kept. A median, not a running mean, because layers
  // are not all alike: Sol-Attn keeps block 0 dense, which on an M5 Pro
  // made that block's section ~25% (124 frames) to ~2x (243 frames) longer,
  // and one such sample in a 0.5-weight mean flipped the tier off a split
  // that was 4.5-6.3% faster.
  static constexpr int kReprobeBlocks = 48;
  int    _blocks_seen = 0;
  int    _since_probe = 0;
  bool   _gpu_mode    = false;
  bool   _barrier     = false;
  Plan   _prev_plan   = Plan::kSplit;
  static constexpr int kWindow = 9;
  double _split_s[kWindow] = {};
  double _probe_s[kWindow] = {};
  int    _split_n = 0, _probe_n = 0;       // samples taken, ever
  double _split_ms    = 0.0;               // medians
  double _probe_ms    = 0.0;
  static double push_median_(double* ring, int* n, double v);
  void decide_();
  // fp16 HEADROOM. The graph's silu(gate) * up is a product of two
  // projections, and on real activations it overflows fp16 where the GPU's
  // bf16 does not -- MEASURED on MiniMax-H3 at block 36 of 50 from an input
  // of |x| <= 9.8: one inf, which the residual stream then spread to every
  // row of every later block. So the up projection is staged divided by a
  // power of two and the ANE's output multiplied back, which is exact
  // arithmetic on the GPU side of the boundary. The scale starts at 1 and
  // only grows: a chunk that comes back non-finite has its up slot divided
  // by 4 in place and is predicted again, and later blocks stage at the new
  // scale. Written only by jobs on the ANE worker.
  static constexpr float kMaxUpScale = 4096.0f;
  float _up_scale   = 1.0f;    // the next staging divides up by this
  float _slot_scale = 1.0f;    // what the up slot is divided by now
  int   _rescales   = 0;       // this block's retries
  // VPIPE_ANE_CHECK: per block, the input's largest |x| and how many
  // elements the fp16 narrowing overflowed, and how many outputs came back
  // non-finite. Diagnostic only -- two serial passes over the ANE's rows.
  bool        _check    = false;
  float       _in_max   = 0.0f;
  std::size_t _in_over  = 0;
  std::size_t _out_bad  = 0;
  metal_compute::SharedBuffer _in, _out;           // fp16, ONE chunk each
};

}  // namespace genai
}  // namespace vpipe

#endif
