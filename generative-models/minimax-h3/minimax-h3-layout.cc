#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <cmath>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

namespace {

constexpr double kMinAspect = 1.0 / 4.0;
constexpr double kMaxAspect = 4.0;

// Python's round() is banker's rounding (ties to even) and the canvas
// resolver rounds a half-pixel exactly often enough to matter -- a 16:9
// canvas at short edge 768 lands on 1344.0 with no fraction, but an
// arbitrary keyframe aspect does not.
double
round_half_even_(double v)
{
  const double r = std::nearbyint(v);   // FE_TONEAREST is ties-to-even
  return r;
}

}  // namespace

bool
resolve_canvas_size(double aspect_w, double aspect_h, int multiple,
                    int short_edge, std::int64_t max_pixels, int* out_h,
                    int* out_w)
{
  if (out_h == nullptr || out_w == nullptr) { return false; }
  if (!(aspect_w > 0.0) || !(aspect_h > 0.0) || multiple <= 0 ||
      short_edge <= 0 || max_pixels <= 0) {
    return false;
  }
  const double ratio = aspect_w / aspect_h;
  if (!(ratio >= kMinAspect && ratio <= kMaxAspect)) { return false; }

  double width  = 0.0;
  double height = 0.0;
  if (ratio >= 1.0) {
    width  = (double)short_edge * ratio;
    height = (double)short_edge;
  } else {
    width  = (double)short_edge;
    height = (double)short_edge / ratio;
  }
  const double area = width * height;
  if (area > (double)max_pixels) {
    const double scale = std::sqrt((double)max_pixels / area);
    width *= scale;
    height *= scale;
  }
  const double m = (double)multiple;
  *out_h = (int)std::max(m, round_half_even_(height / m) * m);
  *out_w = (int)std::max(m, round_half_even_(width / m) * m);
  return true;
}

int
align_num_frames(int num_frames, int frames_per_chunk, int latents_per_chunk)
{
  if (num_frames < 1 || frames_per_chunk < 1) { return 0; }
  while (num_frames % frames_per_chunk != latents_per_chunk) { ++num_frames; }
  return num_frames;
}

int
video_latent_num_frames(int num_frames, int frames_per_chunk,
                        int latents_per_chunk)
{
  if (frames_per_chunk < 1 ||
      num_frames % frames_per_chunk != latents_per_chunk) {
    return 0;
  }
  return (num_frames - latents_per_chunk) / frames_per_chunk *
             latents_per_chunk +
         2;
}

int
audio_latent_num_frames(int num_frames, double fps, int latents_per_second)
{
  if (num_frames < 0 || !(fps > 0.0)) { return 0; }
  return (int)std::llround((double)num_frames / fps *
                           (double)latents_per_second);
}

std::vector<double>
spatial_position_grid(int dim, int patch, double sqrt_area)
{
  std::vector<double> grid;
  if (dim <= 0 || patch <= 0 || !(sqrt_area > 0.0)) { return grid; }
  const int num = dim / patch;
  if (num <= 0) { return grid; }
  const double ratio = (double)dim / sqrt_area;
  const double left  = (1.0 - ratio) / 2.0;
  // numpy linspace(start, stop, num, endpoint=False):
  //   start + arange(num) * (stop - start) / num, with stop - start =
  //   ratio. torch.linspace divides by num-1 and includes the endpoint,
  //   so it is a different grid -- not a rounding difference.
  const double step = ratio / (double)num;
  grid.reserve((std::size_t)num);
  for (int i = 0; i < num; ++i) {
    grid.push_back((left + (double)i * step) * (double)kRopeSpatialScale);
  }
  return grid;
}

std::vector<double>
temporal_position_grid(int num_latent_frames, double origin)
{
  std::vector<double> grid;
  if (num_latent_frames <= 0) { return grid; }
  grid.reserve((std::size_t)num_latent_frames);
  double acc = origin;
  for (int i = 0; i < num_latent_frames; ++i) {
    grid.push_back(acc);
    acc += kRopeFrameRescale * (double)kRopeFramesPerLatent[i % 5];
  }
  return grid;
}

bool
build_packed_sequence(const std::vector<int>& text_token_tags,
                      int num_latent_frames, int latent_height,
                      int latent_width, int num_audio_latents, int patch_h,
                      int patch_w, int audio_channels,
                      const std::vector<Anchor>& keyframe_anchors,
                      PackedLayout* out)
{
  if (out == nullptr) { return false; }
  if (num_latent_frames <= 0 || latent_height <= 0 || latent_width <= 0 ||
      patch_h <= 0 || patch_w <= 0 || audio_channels <= 0 ||
      num_audio_latents < 0) {
    return false;
  }
  if (latent_height % patch_h != 0 || latent_width % patch_w != 0) {
    return false;
  }

  const int ph = latent_height / patch_h;
  const int pw = latent_width / patch_w;
  const int rows_per_frame = ph * pw;
  const int num_text = (int)text_token_tags.size();
  const int num_cond = (int)keyframe_anchors.size() * rows_per_frame;
  const int num_audio_rows = num_audio_latents * audio_channels;
  const int num_video_rows = num_latent_frames * rows_per_frame;
  const int seq = num_text + num_cond + num_audio_rows + num_video_rows;
  if (seq <= 0) { return false; }

  PackedLayout L;
  L.seq_len            = seq;
  L.num_text_rows      = num_text;
  L.condition_start    = num_text;
  L.num_condition_rows = num_cond;
  L.audio_start        = num_text + num_cond;
  L.num_audio_rows     = num_audio_rows;
  L.video_start        = L.audio_start + num_audio_rows;
  L.num_video_rows     = num_video_rows;
  L.position_ids.assign((std::size_t)seq * 3, 0.0);
  L.token_tags.assign((std::size_t)seq, kVideoTag);
  auto pos = [&](int row, int axis) -> double& {
    return L.position_ids[(std::size_t)row * 3 + (std::size_t)axis];
  };

  // 1. Text rows sit on the TIME axis at their row index, and every media
  // row's clock starts at `num_text` -- so the prompt length shifts the
  // whole media grid. Their h/w stay 0.
  for (int i = 0; i < num_text; ++i) { pos(i, 0) = (double)i; }

  // 2. The (h, w) grid of one latent frame, shared by the conditioning
  // rows and the target rows.
  const double sqrt_area =
      std::sqrt((double)latent_height * (double)latent_width);
  const std::vector<double> h_grid =
      spatial_position_grid(latent_height, patch_h, sqrt_area);
  const std::vector<double> w_grid =
      spatial_position_grid(latent_width, patch_w, sqrt_area);
  if ((int)h_grid.size() != ph || (int)w_grid.size() != pw) { return false; }
  auto frame_h = [&](int r) { return h_grid[(std::size_t)(r / pw)]; };
  auto frame_w = [&](int r) { return w_grid[(std::size_t)(r % pw)]; };

  // 3. Keyframe conditioning blocks. "first" pins the block at the first
  // latent frame's rotary time; "last" pins it one frame span short of
  // the end of the span the generated frames cover.
  //
  // The reference sums that span with numpy's pairwise summation, which
  // differs from a sequential sum in the LAST ULP of a double from 16
  // latent frames on. That is ~2e-16 relative on a rotary angle whose
  // cosine is then stored as bf16, so this sums sequentially and the
  // difference cannot reach the model.
  double last_anchor = 0.0;
  {
    double span = 0.0;
    for (int i = 0; i < num_latent_frames; ++i) {
      span += kRopeFrameRescale * (double)kRopeFramesPerLatent[i % 5];
    }
    last_anchor = (double)num_text + span - kRopeFrameRescale;
  }
  for (std::size_t k = 0; k < keyframe_anchors.size(); ++k) {
    const double t = (keyframe_anchors[k] == Anchor::kFirst)
                         ? (double)num_text
                         : last_anchor;
    const int base = L.condition_start + (int)k * rows_per_frame;
    for (int r = 0; r < rows_per_frame; ++r) {
      pos(base + r, 0) = t;
      pos(base + r, 1) = frame_h(r);
      pos(base + r, 2) = frame_w(r);
    }
  }

  // 4. Audio rows are CHANNEL-MAJOR and share the video's rotary clock --
  // one unit per latent, since 40 latents/s is exactly 24 fps * 5/3. They
  // carry NO height coordinate and are pinned to the two extremes of the
  // width grid, which is how the model tells the two stereo channels
  // apart positionally.
  for (int c = 0; c < audio_channels; ++c) {
    for (int i = 0; i < num_audio_latents; ++i) {
      const int row = L.audio_start + c * num_audio_latents + i;
      pos(row, 0) = (double)num_text + (double)i;
      pos(row, 2) = (c == 0) ? w_grid.front() : w_grid.back();
    }
  }

  // 5. Target video rows.
  const std::vector<double> t_grid =
      temporal_position_grid(num_latent_frames, (double)num_text);
  for (int f = 0; f < num_latent_frames; ++f) {
    for (int r = 0; r < rows_per_frame; ++r) {
      const int row = L.video_start + f * rows_per_frame + r;
      pos(row, 0) = t_grid[(std::size_t)f];
      pos(row, 1) = frame_h(r);
      pos(row, 2) = frame_w(r);
    }
  }

  // 6. Tags. Assigned text -> audio -> video in that order, matching the
  // reference: a text row that belongs to a keyframe's vision block is
  // tagged VIDEO by the caller, and the video pass must not overwrite the
  // text pass (it does not -- the two index disjoint ranges).
  for (int i = 0; i < num_text; ++i) {
    L.token_tags[(std::size_t)i] = text_token_tags[(std::size_t)i];
  }
  for (int i = 0; i < num_audio_rows; ++i) {
    L.token_tags[(std::size_t)(L.audio_start + i)] = kAudioTag;
  }
  L.video_indices.reserve((std::size_t)(num_cond + num_video_rows));
  for (int i = 0; i < num_cond; ++i) {
    L.video_indices.push_back(L.condition_start + i);
  }
  for (int i = 0; i < num_video_rows; ++i) {
    L.video_indices.push_back(L.video_start + i);
  }
  for (int r : L.video_indices) { L.token_tags[(std::size_t)r] = kVideoTag; }

  *out = std::move(L);
  return true;
}

void
build_row_timesteps(const PackedLayout& layout, float video_timestep,
                    float audio_timestep, float condition_video_timestep,
                    std::vector<float>* timesteps_out,
                    std::vector<int>* row_index_out)
{
  if (timesteps_out == nullptr || row_index_out == nullptr) { return; }
  const std::size_t seq = (std::size_t)layout.seq_len;
  std::vector<float> row(seq, video_timestep);
  // Conditioning rows first, then the generated audio rows -- the
  // reference's order, which matters only if the two ranges overlapped
  // (they do not).
  for (int i = 0; i < layout.num_condition_rows; ++i) {
    row[(std::size_t)(layout.condition_start + i)] = condition_video_timestep;
  }
  for (int i = 0; i < layout.num_audio_rows; ++i) {
    row[(std::size_t)(layout.audio_start + i)] = audio_timestep;
  }

  // torch.unique(sorted=True, return_inverse=True).
  std::vector<float> uniq = row;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::vector<int> idx(seq, 0);
  for (std::size_t i = 0; i < seq; ++i) {
    idx[i] = (int)(std::lower_bound(uniq.begin(), uniq.end(), row[i]) -
                   uniq.begin());
  }
  *timesteps_out = std::move(uniq);
  *row_index_out = std::move(idx);
}

std::vector<int>
build_adaln_indices(const PackedLayout& layout,
                    const std::vector<int>& row_timestep_index)
{
  std::vector<int> out;
  if ((int)row_timestep_index.size() != layout.seq_len) { return out; }
  out.resize((std::size_t)layout.seq_len);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = row_timestep_index[i] * kModalityNum + layout.token_tags[i];
  }
  return out;
}

TileSplit
split_tiles(int length, int tile_size, int min_overlap, int ratio)
{
  TileSplit out;
  if (length <= 0 || tile_size <= 0) { return out; }
  if (tile_size >= length) {
    out.start.push_back(0);
    out.length.push_back(length);
    return out;                        // one tile, no overlaps
  }
  int n = (length + tile_size - 1) / tile_size;
  // The ceiling can still be one short once the overlaps are paid for,
  // so grow until the union genuinely covers the axis.
  while (tile_size * n - min_overlap * (n - 1) - length < 0) { ++n; }
  out.overlap.assign((std::size_t)(n - 1), min_overlap);
  int total = 0;
  for (int v : out.overlap) { total += v; }
  const int slack = tile_size * n - total - length;
  if (ratio > 0) {
    for (int i = 0; i < slack / ratio; ++i) {
      out.overlap[(std::size_t)(i % (n - 1))] += ratio;
    }
  }
  out.start.push_back(0);
  out.length.push_back(tile_size);
  for (int i = 0; i + 1 < n; ++i) {
    out.start.push_back(out.start.back() + tile_size -
                        out.overlap[(std::size_t)i]);
    out.length.push_back(tile_size);
  }
  return out;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
