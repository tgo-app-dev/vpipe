#include "generative-models/minimax-h3/vdn-geometry.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

namespace {

int
clamp_int(int v, int lo, int hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

bool
parse_anchor_frames(std::string_view name, AnchorFrames* out)
{
  AnchorFrames mode;
  if      (name == "none")    { mode = AnchorFrames::kNone; }
  else if (name == "columns") { mode = AnchorFrames::kColumns; }
  else if (name == "rows")    { mode = AnchorFrames::kRows; }
  else if (name == "both")    { mode = AnchorFrames::kBoth; }
  else                        { return false; }
  if (out != nullptr) { *out = mode; }
  return true;
}

const char*
anchor_frames_name(AnchorFrames mode)
{
  switch (mode) {
    case AnchorFrames::kColumns: return "columns";
    case AnchorFrames::kRows:    return "rows";
    case AnchorFrames::kBoth:    return "both";
    default:                     return "none";
  }
}

std::vector<Bound>
window_bounds(int num_frames, int radius, int chunk)
{
  std::vector<Bound> out;
  if (num_frames <= 0) { return out; }
  out.reserve((std::size_t)num_frames);
  for (int t = 0; t < num_frames; ++t) {
    if (chunk <= 0) {
      out.push_back({t - radius, t + radius});
    } else {
      const int c = t / chunk;
      out.push_back({(c - radius) * chunk, (c + radius + 1) * chunk - 1});
    }
  }
  return out;
}

bool
WindowMask::allows(int query_row, int key_row) const
{
  const int ve = video_end();
  const bool query_is_video = query_row >= video_start && query_row < ve;
  const bool key_is_video   = key_row >= video_start && key_row < ve;
  // Anything touching a global row (text, audio) is dense both ways.
  if (!(query_is_video && key_is_video)) { return true; }
  if (bounds.empty() || tokens_per_frame <= 0) { return true; }

  const int qf = clamp_int((query_row - video_start) / tokens_per_frame, 0,
                           num_frames - 1);
  const int kf = (key_row - video_start) / tokens_per_frame;
  const Bound& b = bounds[(std::size_t)qf];
  if (kf >= b.lo && kf <= b.hi) { return true; }

  const bool key_anchor   = kf == 0 || kf == num_frames - 1;
  const bool query_anchor = qf == 0 || qf == num_frames - 1;
  if ((anchors == AnchorFrames::kColumns || anchors == AnchorFrames::kBoth)
      && key_anchor) {
    return true;
  }
  if ((anchors == AnchorFrames::kRows || anchors == AnchorFrames::kBoth)
      && query_anchor) {
    return true;
  }
  return false;
}

GatherIndex
gather_indices(const std::vector<Bound>& bounds, int num_frames)
{
  GatherIndex g;
  const std::size_t n = bounds.size();
  g.before_idx.resize(n);
  g.after_idx.resize(n);
  g.has_before.resize(n);
  g.has_after.resize(n);
  g.bridge_before.resize(n);
  g.bridge_after.resize(n);
  for (std::size_t t = 0; t < n; ++t) {
    const int last_before = bounds[t].lo - 1;
    const int first_after = bounds[t].hi + 1;
    g.before_idx[t]    = last_before < 0 ? 0 : last_before;
    g.after_idx[t]     = std::min(first_after, num_frames - 1);
    g.has_before[t]    = (char)(last_before >= 0);
    g.has_after[t]     = (char)(first_after < num_frames);
    // NOT the gather indices -- see the header. before is
    // last_before + 1 clamped at 0, after is first_after clamped at F
    // (the log-prefix bank has F + 1 rows, so F is in range).
    g.bridge_before[t] = std::max(last_before + 1, 0);
    g.bridge_after[t]  = std::min(first_after, num_frames);
  }
  return g;
}

void
alpha_bridge(const float* alpha, int num_frames, int heads, int d_k,
             const GatherIndex& idx, float* from_before, float* from_after)
{
  if (num_frames <= 0 || heads <= 0 || d_k <= 0) { return; }
  const std::size_t per_frame = (std::size_t)heads * (std::size_t)d_k;
  // EXCLUSIVE log-prefix with a leading zero row, so the empty product
  // is 1 and any span [a, b) is one subtraction. [F + 1, H, d_k].
  std::vector<double> log_prefix((std::size_t)(num_frames + 1) * per_frame,
                                 0.0);
  for (int f = 0; f < num_frames; ++f) {
    const double* prev = log_prefix.data() + (std::size_t)f * per_frame;
    double* cur = log_prefix.data() + (std::size_t)(f + 1) * per_frame;
    const float* a = alpha + (std::size_t)f * per_frame;
    for (std::size_t i = 0; i < per_frame; ++i) {
      const float clamped = a[i] < 1e-12f ? 1e-12f : a[i];
      cur[i] = prev[i] + std::log((double)clamped);
    }
  }
  for (int t = 0; t < num_frames; ++t) {
    const double* end_b =
        log_prefix.data() + (std::size_t)(t + 1) * per_frame;
    const double* beg_b =
        log_prefix.data() + (std::size_t)idx.bridge_before[t] * per_frame;
    const double* end_a =
        log_prefix.data() + (std::size_t)idx.bridge_after[t] * per_frame;
    const double* beg_a = log_prefix.data() + (std::size_t)t * per_frame;
    float* ob = from_before + (std::size_t)t * per_frame;
    float* oa = from_after + (std::size_t)t * per_frame;
    for (std::size_t i = 0; i < per_frame; ++i) {
      ob[i] = (float)std::exp(end_b[i] - beg_b[i]);
      oa[i] = (float)std::exp(end_a[i] - beg_a[i]);
    }
  }
}

void
gather_linear_state(const float* prefix, const float* suffix,
                    const float* alpha, const GatherIndex& idx,
                    int num_frames, int heads, int d_v, int d_k,
                    bool bridge_alpha, const float* text_state, float* out)
{
  if (num_frames <= 0 || heads <= 0 || d_v <= 0 || d_k <= 0) { return; }
  const std::size_t mat = (std::size_t)d_v * (std::size_t)d_k;
  const std::size_t per_frame_state = (std::size_t)heads * mat;
  const std::size_t per_frame_alpha = (std::size_t)heads * (std::size_t)d_k;

  std::vector<float> ab, aa;
  if (bridge_alpha) {
    ab.resize((std::size_t)num_frames * per_frame_alpha);
    aa.resize(ab.size());
    alpha_bridge(alpha, num_frames, heads, d_k, idx, ab.data(), aa.data());
  }

  for (int t = 0; t < num_frames; ++t) {
    const float* pb =
        prefix + (std::size_t)idx.before_idx[t] * per_frame_state;
    const float* sa =
        suffix + (std::size_t)idx.after_idx[t] * per_frame_state;
    float* o = out + (std::size_t)t * per_frame_state;
    for (int h = 0; h < heads; ++h) {
      const std::size_t base = (std::size_t)h * mat;
      for (int v = 0; v < d_v; ++v) {
        for (int k = 0; k < d_k; ++k) {
          const std::size_t e = base + (std::size_t)v * d_k + k;
          // The text state replaces the CLAMPED read on a side that has
          // nothing outside the window, before any decay is applied.
          float before = pb[e];
          float after  = sa[e];
          if (text_state != nullptr) {
            if (!idx.has_before[t]) { before = text_state[e]; }
            if (!idx.has_after[t])  { after  = text_state[e]; }
          }
          if (bridge_alpha) {
            // alpha is per KEY channel and broadcasts over d_v.
            const std::size_t ai =
                (std::size_t)t * per_frame_alpha + (std::size_t)h * d_k + k;
            before *= ab[ai];
            after  *= aa[ai];
          }
          if (text_state == nullptr) {
            before *= (float)(idx.has_before[t] != 0);
            after  *= (float)(idx.has_after[t] != 0);
          }
          o[e] = before + after;
        }
      }
    }
  }
}

namespace {

// Append a row run, merging into the previous one when they touch.
//
// `group_begin` is where THIS group's runs start in the shared arrays,
// and it is load-bearing rather than bookkeeping: without it the merge
// looks at end.back(), which at a group boundary is the PREVIOUS
// group's last run -- so group 0's dense span swallows the next group's
// first run and that group comes out empty. Silent, because attention
// over the wrong key set still returns a normalised distribution.
//
// The caller adds runs in ascending order, which is what makes a single
// backward look sufficient.
void
push_span_(std::vector<int>& start, std::vector<int>& end,
           std::size_t group_begin, int lo, int hi)
{
  if (hi <= lo) { return; }
  if (start.size() > group_begin && lo <= end.back()) {
    if (hi > end.back()) { end.back() = hi; }
    return;
  }
  start.push_back(lo);
  end.push_back(hi);
}

}  // namespace

WindowSpans
build_window_spans(const WindowMask& mask)
{
  WindowSpans out;
  const int F = mask.num_frames;
  const int S = mask.tokens_per_frame;
  const int vs = mask.video_start;
  const int ve = mask.video_end();
  out.groups = 1 + (F > 0 ? F : 0);
  out.off.reserve((std::size_t)out.groups + 1);
  out.off.push_back(0);

  // Group 0: the global rows. Every pair touching one is dense in both
  // directions, so the whole sequence.
  push_span_(out.start, out.end, 0, 0, mask.seq_len);
  out.off.push_back((int)out.start.size());

  const bool anchor_cols = mask.anchors == AnchorFrames::kColumns
                           || mask.anchors == AnchorFrames::kBoth;
  const bool anchor_rows = mask.anchors == AnchorFrames::kRows
                           || mask.anchors == AnchorFrames::kBoth;

  for (int f = 0; f < F; ++f) {
    const std::size_t gb = out.start.size();
    const bool row_dense = anchor_rows && (f == 0 || f == F - 1);
    if (row_dense) {
      push_span_(out.start, out.end, gb, 0, mask.seq_len);
      out.off.push_back((int)out.start.size());
      continue;
    }
    // Ascending order: leading globals, then the video runs in frame
    // order, then trailing globals.
    push_span_(out.start, out.end, gb, 0, vs);
    std::vector<std::pair<int, int>> frames;
    if (!mask.bounds.empty()) {
      const Bound& b = mask.bounds[(std::size_t)f];
      const int lo = b.lo < 0 ? 0 : b.lo;
      const int hi = b.hi > F - 1 ? F - 1 : b.hi;
      if (lo <= hi) { frames.push_back({lo, hi}); }
    }
    if (anchor_cols) {
      frames.push_back({0, 0});
      frames.push_back({F - 1, F - 1});
    }
    std::sort(frames.begin(), frames.end());
    for (const auto& fr : frames) {
      push_span_(out.start, out.end, gb, vs + fr.first * S,
                 vs + (fr.second + 1) * S);
    }
    push_span_(out.start, out.end, gb, ve, mask.seq_len);
    out.off.push_back((int)out.start.size());
  }
  return out;
}

std::vector<Bound>
rebase_for_anchor_skip(const std::vector<Bound>& bounds, int num_frames)
{
  std::vector<Bound> out;
  if (num_frames <= 2) { return out; }
  out.reserve((std::size_t)(num_frames - 2));
  for (int t = 1; t < num_frames - 1; ++t) {
    out.push_back({bounds[(std::size_t)t].lo - 1,
                   bounds[(std::size_t)t].hi - 1});
  }
  return out;
}

BlockSpans
build_block_spans(const WindowMask& mask, int bq, int bk)
{
  BlockSpans out;
  out.bq = bq;
  out.bk = bk;
  if (bq <= 0 || bk <= 0 || mask.seq_len <= 0) { return out; }

  const int nq = (mask.seq_len + bq - 1) / bq;
  const int nk = (mask.seq_len + bk - 1) / bk;
  const int tpf = mask.tokens_per_frame;
  const int nf = mask.num_frames;
  const int ve = mask.video_end();
  const bool anchor_cols = mask.anchors == AnchorFrames::kColumns
                           || mask.anchors == AnchorFrames::kBoth;
  const bool anchor_rows = mask.anchors == AnchorFrames::kRows
                           || mask.anchors == AnchorFrames::kBoth;
  out.off.reserve((std::size_t)nq + 1);
  out.off.push_back(0);

  // Add the key blocks covering rows [r0, r1), merged against whatever
  // is already there. Ranges are produced in ascending order per query
  // block, so a single tail comparison keeps `blocks` sorted and free of
  // duplicates -- which the kernel needs, since it only moves forward.
  const std::size_t base_guard = 0;
  auto add_rows = [&](int r0, int r1, std::size_t first) {
    r0 = clamp_int(r0, 0, mask.seq_len);
    r1 = clamp_int(r1, 0, mask.seq_len);
    if (r1 <= r0) { return; }
    const int b0 = r0 / bk, b1 = (r1 - 1) / bk;
    for (int b = b0; b <= b1 && b < nk; ++b) {
      if (out.blocks.size() > first && out.blocks.back() >= b) { continue; }
      out.blocks.push_back(b);
    }
  };
  (void)base_guard;

  for (int qb = 0; qb < nq; ++qb) {
    const std::size_t first = out.blocks.size();
    const int r0 = qb * bq;
    const int r1 = std::min(r0 + bq, mask.seq_len);

    // A query block holding ANY row outside the video block, or any
    // anchor row under an anchor-rows mode, sees the whole sequence --
    // there is nothing to skip and no edge to mask.
    bool dense = mask.bounds.empty() || tpf <= 0;
    if (!dense && (r0 < mask.video_start || r1 > ve)) { dense = true; }
    int qf0 = 0, qf1 = 0;
    if (!dense) {
      qf0 = clamp_int((r0 - mask.video_start) / tpf, 0, nf - 1);
      qf1 = clamp_int((r1 - 1 - mask.video_start) / tpf, 0, nf - 1);
      if (anchor_rows) {
        for (int f = qf0; f <= qf1 && !dense; ++f) {
          if (f == 0 || f == nf - 1) { dense = true; }
        }
      }
    }
    if (dense) {
      for (int b = 0; b < nk; ++b) { out.blocks.push_back(b); }
      out.off.push_back((int)out.blocks.size());
      continue;
    }

    // The union of the block's rows' windows. Taken as min/max over the
    // frames rather than as bounds[qf0].lo / bounds[qf1].hi, because
    // nothing in the type says the bounds are monotone and a chunked
    // window that ever stopped being so would silently drop keys.
    int lo = mask.bounds[(std::size_t)qf0].lo;
    int hi = mask.bounds[(std::size_t)qf0].hi;
    for (int f = qf0 + 1; f <= qf1; ++f) {
      lo = std::min(lo, mask.bounds[(std::size_t)f].lo);
      hi = std::max(hi, mask.bounds[(std::size_t)f].hi);
    }
    lo = clamp_int(lo, 0, nf - 1);
    hi = clamp_int(hi, 0, nf - 1);

    // Ascending: the global rows before the video block, then the first
    // anchor frame, the window, the last anchor frame, and finally the
    // rows after the video block.
    add_rows(0, mask.video_start, first);
    if (anchor_cols && 0 < lo) {
      add_rows(mask.video_start, mask.video_start + tpf, first);
    }
    add_rows(mask.video_start + lo * tpf,
             mask.video_start + (hi + 1) * tpf, first);
    if (anchor_cols && hi < nf - 1) {
      add_rows(mask.video_start + (nf - 1) * tpf, ve, first);
    }
    add_rows(ve, mask.seq_len, first);
    out.off.push_back((int)out.blocks.size());
  }
  return out;
}

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
