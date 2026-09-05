#ifndef GENERATIVE_MODELS_MINIMAX_H3_VDN_GEOMETRY_H
#define GENERATIVE_MODELS_MINIMAX_H3_VDN_GEOMETRY_H

// VDN-H3 (Video DeltaNet) hybrid attention: the GEOMETRY half.
//
// VDN replaces MiniMax-H3's dense attention with two branches that
// PARTITION the key set rather than duplicating it:
//
//   softmax   exact attention, but video->video pairs restricted to a
//             window of whole frames around the query's frame
//   linear    a bidirectional delta-rule recurrence over frames, read
//             out at every video token, carrying exactly what the
//             window CANNOT see
//
// Which is why the two halves have to agree about the window to the
// frame: a frame counted by both is counted twice, and a frame counted
// by neither is silently dropped. Everything in this file is that
// agreement, and it is pure integer/float arithmetic over frame indices
// -- no GPU, no weights -- which is what makes it testable on its own,
// exactly as minimax-h3-layout.h is.
//
// Reference: github.com/OpenVDN/vdn-minimax-h3, src/models/
// softmax_attention/window.py and src/models/linear_attention/scan.py.

#include <cstddef>
#include <string_view>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

// How frames 0 and F-1 sit in the softmax mask.
//
// Not a tuning knob: it is a CROSS-BRANCH fact. Under kBoth the two
// frames are exact softmax in both directions, so the linear branch
// drops them from its input entirely and the partition stays exact.
// Under kColumns or kRows alone it would not be exact, so the branch
// keeps covering them. The released checkpoints are kBoth.
enum class AnchorFrames { kNone, kColumns, kRows, kBoth };

// The spelling used in model_spec.json / linear_branch/config.json
// ("none", "columns", "rows", "both"). False on anything else -- an
// unrecognised mode must not fall back to a default, because every
// wrong answer here is a silently mis-partitioned attention rather
// than an error.
bool parse_anchor_frames(std::string_view name, AnchorFrames* out);
const char* anchor_frames_name(AnchorFrames mode);

// One query frame's inclusive softmax window, in FRAME indices.
//
// DELIBERATELY UNCLAMPED: `lo < 0` and `hi >= num_frames` are what tell
// the linear branch that this query has no frames outside the window on
// that side. Clamping here would make a boundary frame look like it had
// a neighbour whose state should be gathered, which is the difference
// between reading the text state and reading frame 0 twice.
struct Bound {
  int lo = 0;
  int hi = 0;

  bool operator==(const Bound& o) const { return lo == o.lo && hi == o.hi; }
  bool operator!=(const Bound& o) const { return !(*this == o); }
};

// Per-frame windows.
//
//   chunk <= 0   FRAME mode: the centred window |t_q - t_k| <= radius.
//   chunk == K   CHUNK-ALIGNED mode: frame t belongs to chunk t / K and
//                sees whole chunks [c - radius, c + radius].
//
// The released config is radius 1, chunk 5, and the chunk is not a
// wider frame window wearing a different name. The video VAE encodes
// every 5 latent frames as ONE unit, so a frame seeing part of a
// neighbouring chunk sees a fragment of something that was never coded
// separably. Alignment is what chunk mode buys; width is incidental. A
// frame window cannot express it -- centring any radius on t gives
// frames 5 and 9 different spans, and one of them straddles a boundary.
//
// The last chunk is short when K does not divide num_frames (102 =
// 20*5 + 2) and that is correct: it is still a whole chunk.
std::vector<Bound> window_bounds(int num_frames, int radius, int chunk);

// Does the softmax branch keep the (query row, key row) pair?
//
// Every pair involving a NON-video row stays dense in both directions --
// the prompt and the soundtrack are small and the model reads them
// exactly. Only video->video is windowed. `bounds` is indexed by the
// query's frame, so the query frame is clamped into range (the
// predicate is also evaluated on global rows, where the division runs
// out of range) while the key frame is not: it is only consulted when
// the key IS video, where it cannot be out of range.
struct WindowMask {
  int seq_len          = 0;
  int video_start      = 0;   // first video row of the packed sequence
  int num_frames       = 0;
  int tokens_per_frame = 0;
  AnchorFrames anchors = AnchorFrames::kNone;
  std::vector<Bound> bounds;

  int video_end() const
  {
    return video_start + num_frames * tokens_per_frame;
  }

  bool allows(int query_row, int key_row) const;
};

// Where each query frame reads the state of everything OUTSIDE its
// window, and how far that state has to be decayed to reach it.
//
// prefix[j] holds frames 0..j and suffix[j] holds frames j..F-1, so for
// a query frame t with window [lo, hi] the complement is exactly
// prefix[lo-1] plus suffix[hi+1]: one frame outside on each side,
// nothing counted twice.
//
// THE BRIDGE INDEX IS NOT THE GATHER INDEX, and the two differ at the
// ends for a reason that is invisible until the text state exists. A
// boundary row gathers a CLAMPED state which is then discarded (or
// replaced by the text state), but it must still decay over the frames
// it really skipped: from the scan's virtual index -1 that is [0..t],
// from virtual F it is [t..F-1]. Clamping both the same way decays the
// text state over one frame too few -- a wrong answer that is a few
// percent off and looks entirely plausible.
struct GatherIndex {
  std::vector<int>  before_idx;      // prefix row to read (clamped)
  std::vector<int>  after_idx;       // suffix row to read (clamped)
  std::vector<char> has_before;      // was there anything to the left?
  std::vector<char> has_after;
  std::vector<int>  bridge_before;   // log-prefix row, NOT before_idx
  std::vector<int>  bridge_after;    // may be F -- the bank has F+1 rows
};

GatherIndex gather_indices(const std::vector<Bound>& bounds, int num_frames);

// prod over the frames between the boundary and t of alpha, as a
// difference of EXCLUSIVE log-prefix sums so any span is one
// subtraction rather than a product loop. alpha is [F, H, d_k] and
// per KEY channel; both outputs are the same shape.
//
// fp32 throughout: the scan multiplies alpha across every frame, so a
// per-element error compounds -- at ~100 frames a bf16 alpha moves the
// retention of the far state by tens of percent on the worst channels.
void alpha_bridge(const float* alpha, int num_frames, int heads, int d_k,
                  const GatherIndex& idx, float* from_before,
                  float* from_after);

// linear_state[f, h, d_v, d_k]: everything outside the window, in the
// query frame's frame of reference.
//
// `text_state` [H, d_v, d_k] is the state both scans STARTED from. With
// it, a query whose window already touches a clip end reads the prompt
// from that side instead of nothing, decayed in over exactly the frames
// between the boundary and t. Without it (nullptr) the ends contribute
// zero, which is the plain complement-of-the-window semantics.
//
// `bridge_alpha` false uses the states exactly as gathered.
void gather_linear_state(const float* prefix, const float* suffix,
                         const float* alpha, const GatherIndex& idx,
                         int num_frames, int heads, int d_v, int d_k,
                         bool bridge_alpha, const float* text_state,
                         float* out);

// The mask as SPANS, which is what a kernel can actually consume.
//
// `WindowMask::allows` is the predicate; materialising it is a [seq, seq]
// boolean, which at production geometry is 105k x 105k. But the allowed
// set of a video query is a union of at most five CONTIGUOUS runs of
// rows -- the globals before the video block, the globals after it, the
// chunk window, and the two anchor frames -- and every query in the same
// frame has the SAME one, because the window is a property of the chunk.
//
// So the spans are stored per GROUP, CSR-style: group 0 is every global
// row (dense, one span), group 1 + f is video frame f. A kernel derives
// its group from the row index and walks [off[g], off[g+1]).
//
// The runs come back SORTED and COALESCED. That is not tidiness: the
// online softmax accumulates one pass over the union, so a key appearing
// in two spans would be counted twice and get double its softmax mass --
// silently, since the result is still a normalised distribution.
struct WindowSpans {
  int groups = 0;                  // 1 + num_frames
  std::vector<int> off;            // [groups + 1]
  std::vector<int> start;          // [off[groups]]
  std::vector<int> end;            // exclusive
};

WindowSpans build_window_spans(const WindowMask& mask);

// The same mask for a BLOCK-SPARSE kernel: which key blocks of `bk` rows
// each query block of `bq` rows may attend to.
//
// A DIFFERENT SHAPE FROM WindowSpans, and not merely a coarser one. That
// one is per GROUP -- every query in a frame shares its window, so 103
// groups serve 105k rows -- and it is exact to the row. This is per
// QUERY BLOCK, because a steel kernel's threadgroup owns a block of
// rows and has to load whole key blocks for all of them at once; so the
// answer is the UNION over the block's rows, ROUNDED OUTWARD to whole
// key blocks. A block at a span's edge therefore carries keys the mask
// forbids and the kernel must still test them element by element.
// Rounding outward is safe; rounding inward would silently drop keys.
//
// `off` has one entry per query block plus a terminator; `blocks` lists
// the key block indices, ascending, for each.
struct BlockSpans {
  int bq = 0, bk = 0;
  std::vector<int> off;      // [num_query_blocks + 1]
  std::vector<int> blocks;   // key block indices
};

BlockSpans build_block_spans(const WindowMask& mask, int bq, int bk);

// The kBoth partner: frames 0 and F-1 leave the linear branch's INPUT
// entirely, so the branch runs over a sub-clip of F-2 frames whose
// window bounds rebase by one. Empty when the anchors ARE the clip
// (F <= 2), which the caller answers with an all-zero readout.
std::vector<Bound> rebase_for_anchor_skip(const std::vector<Bound>& bounds,
                                          int num_frames);

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
