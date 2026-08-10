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

  // 7. The same rows in the general shape, so that a consumer does not
  // have to know which task built the layout. Here every modality is
  // one range (video two), which is exactly what `ref2va` stops being
  // true.
  L.audio_indices.reserve((std::size_t)num_audio_rows);
  for (int i = 0; i < num_audio_rows; ++i) {
    L.audio_indices.push_back(L.audio_start + i);
  }
  if (num_cond > 0) {
    L.video_runs.push_back({L.condition_start, num_cond});
  }
  if (num_video_rows > 0) {
    L.video_runs.push_back({L.video_start, num_video_rows});
  }
  if (num_audio_rows > 0) {
    L.audio_runs.push_back({L.audio_start, num_audio_rows});
  }
  L.num_condition_video_rows = num_cond;
  L.num_condition_audio_rows = 0;

  *out = std::move(L);
  return true;
}

bool
build_ref2va_packed_sequence(const std::vector<int>& text_token_tags,
                             const std::vector<Reference>& references,
                             int num_latent_frames, int latent_height,
                             int latent_width, int num_audio_latents,
                             int patch_h, int patch_w, int audio_channels,
                             PackedLayout* out)
{
  if (out == nullptr || references.empty()) { return false; }
  if (num_latent_frames <= 0 || latent_height <= 0 || latent_width <= 0 ||
      patch_h <= 0 || patch_w <= 0 || audio_channels <= 0 ||
      num_audio_latents < 0) {
    return false;
  }
  if (latent_height % patch_h != 0 || latent_width % patch_w != 0) {
    return false;
  }
  // Every reference's own geometry has to divide the patch too -- it is
  // encoded at a resolution of its own, so the target's divisibility
  // says nothing about it.
  for (const Reference& r : references) {
    const bool visual = r.kind != Reference::Kind::kAudio;
    if (visual) {
      if (r.num_latent_frames <= 0 || r.latent_height <= 0 ||
          r.latent_width <= 0) {
        return false;
      }
      if (r.latent_height % patch_h != 0 || r.latent_width % patch_w != 0) {
        return false;
      }
      if (r.kind == Reference::Kind::kImage && r.num_latent_frames != 1) {
        return false;                    // an image is a single frame
      }
    } else if (r.num_audio_latents <= 0) {
      return false;                      // an audio reference IS its rows
    }
    if (r.kind == Reference::Kind::kImage && r.has_audio()) {
      return false;                      // an image carries no soundtrack
    }
  }

  const int num_text = (int)text_token_tags.size();
  const int tph = latent_height / patch_h;
  const int tpw = latent_width / patch_w;
  const int target_rows_per_frame = tph * tpw;
  const int num_target_video = num_latent_frames * target_rows_per_frame;
  const int num_target_audio = num_audio_latents * audio_channels;

  auto ref_video_rows = [&](const Reference& r) {
    return r.num_latent_frames * (r.latent_height / patch_h) *
           (r.latent_width / patch_w);
  };

  int num_ref_video = 0;
  int num_ref_audio = 0;
  for (const Reference& r : references) {
    if (r.kind != Reference::Kind::kAudio) { num_ref_video += ref_video_rows(r); }
    if (r.has_audio()) { num_ref_audio += r.num_audio_latents * audio_channels; }
  }
  const int seq = num_text + num_ref_video + num_ref_audio + num_target_audio +
                  num_target_video;
  if (seq <= 0) { return false; }

  PackedLayout L;
  L.seq_len       = seq;
  L.num_text_rows = num_text;
  L.position_ids.assign((std::size_t)seq * 3, 0.0);
  L.token_tags.assign((std::size_t)seq, kVideoTag);
  auto pos = [&](int row, int axis) -> double& {
    return L.position_ids[(std::size_t)row * 3 + (std::size_t)axis];
  };

  // 1. Text rows on the time axis at their own index, as in every task.
  for (int i = 0; i < num_text; ++i) { pos(i, 0) = (double)i; }

  const double target_sqrt_area =
      std::sqrt((double)latent_height * (double)latent_width);
  const std::vector<double> target_h =
      spatial_position_grid(latent_height, patch_h, target_sqrt_area);
  const std::vector<double> target_w =
      spatial_position_grid(latent_width, patch_w, target_sqrt_area);
  if ((int)target_h.size() != tph || (int)target_w.size() != tpw) {
    return false;
  }

  // Place one channel-major audio block. Audio rows carry no height and
  // are pinned to the two extremes of the width grid of THEIR OWN block
  // -- the target grid for a standalone audio reference, the video's own
  // grid for a soundtrack -- which is how the model tells the two stereo
  // channels apart positionally.
  auto fill_audio = [&](int start, int latents, double clock,
                        const std::vector<double>& w_grid) {
    for (int c = 0; c < audio_channels; ++c) {
      for (int i = 0; i < latents; ++i) {
        const int row = start + c * latents + i;
        pos(row, 0) = clock + (double)i;
        pos(row, 2) = (c == 0) ? w_grid.front() : w_grid.back();
      }
    }
  };

  // 2. The reference blocks, in request order, on the shared clock.
  int    cursor = num_text;
  double clock  = (double)num_text;
  for (const Reference& r : references) {
    if (r.kind == Reference::Kind::kImage) {
      const int rows = ref_video_rows(r);
      const int rpw  = r.latent_width / patch_w;
      const double area =
          std::sqrt((double)r.latent_height * (double)r.latent_width);
      const std::vector<double> rh =
          spatial_position_grid(r.latent_height, patch_h, area);
      const std::vector<double> rw =
          spatial_position_grid(r.latent_width, patch_w, area);
      for (int i = 0; i < rows; ++i) {
        pos(cursor + i, 0) = clock;
        pos(cursor + i, 1) = rh[(std::size_t)(i / rpw)];
        pos(cursor + i, 2) = rw[(std::size_t)(i % rpw)];
      }
      L.video_runs.push_back({cursor, rows});
      cursor += rows;
      // An image is a single frame and takes a single INTEGER slot, not
      // a latent frame's 5/3 units.
      clock += 1.0;
    } else if (r.kind == Reference::Kind::kAudio) {
      const int rows = r.num_audio_latents * audio_channels;
      fill_audio(cursor, r.num_audio_latents, clock, target_w);
      L.audio_runs.push_back({cursor, rows});
      cursor += rows;
      clock += (double)r.num_audio_latents;
    } else {
      // A video reference's soundtrack rows are packed immediately
      // BEFORE its video rows and share their origin, so the two are
      // rotary-aligned exactly as the generated audio and video are.
      const int a_rows = r.num_audio_latents * audio_channels;
      const int v_rows = ref_video_rows(r);
      const int rpw    = r.latent_width / patch_w;
      const double area =
          std::sqrt((double)r.latent_height * (double)r.latent_width);
      const std::vector<double> rh =
          spatial_position_grid(r.latent_height, patch_h, area);
      const std::vector<double> rw =
          spatial_position_grid(r.latent_width, patch_w, area);
      if (a_rows > 0) {
        fill_audio(cursor, r.num_audio_latents, clock, rw);
        L.audio_runs.push_back({cursor, a_rows});
        cursor += a_rows;
      }
      const std::vector<double> t_grid =
          temporal_position_grid(r.num_latent_frames, clock);
      const int rows_per_frame = (r.latent_height / patch_h) * rpw;
      for (int f = 0; f < r.num_latent_frames; ++f) {
        for (int i = 0; i < rows_per_frame; ++i) {
          const int row = cursor + f * rows_per_frame + i;
          pos(row, 0) = t_grid[(std::size_t)f];
          pos(row, 1) = rh[(std::size_t)(i / rpw)];
          pos(row, 2) = rw[(std::size_t)(i % rpw)];
        }
      }
      L.video_runs.push_back({cursor, v_rows});
      cursor += v_rows;
      // The clock advances by the LONGER of the two streams this
      // reference carries. Summed sequentially: the reference sums the
      // same series with numpy's pairwise summation, and the two differ
      // in the last ulp of a double from 16 latent frames on -- ~2e-16
      // relative on an angle whose cosine is then stored as bf16, so it
      // cannot reach the model. Same call as the `fl2va` anchor.
      double span = 0.0;
      for (int i = 0; i < r.num_latent_frames; ++i) {
        span += kRopeFrameRescale * (double)kRopeFramesPerLatent[i % 5];
      }
      clock += std::max((double)r.num_audio_latents, span);
    }
  }

  // 3. The generated rows share the origin the references left behind.
  const int audio_start = cursor;
  const int video_start = audio_start + num_target_audio;
  L.audio_start    = audio_start;
  L.num_audio_rows = num_target_audio;
  L.video_start    = video_start;
  L.num_video_rows = num_target_video;
  // The `fl2va` conditioning fields describe a keyframe block that does
  // not exist here; the reference blocks are what a `ref2va` request
  // conditions on and they are addressed through the runs.
  L.condition_start    = num_text;
  L.num_condition_rows = num_ref_video;

  if (num_target_audio > 0) {
    fill_audio(audio_start, num_audio_latents, clock, target_w);
    L.audio_runs.push_back({audio_start, num_target_audio});
  }
  const std::vector<double> t_grid =
      temporal_position_grid(num_latent_frames, clock);
  for (int f = 0; f < num_latent_frames; ++f) {
    for (int i = 0; i < target_rows_per_frame; ++i) {
      const int row = video_start + f * target_rows_per_frame + i;
      pos(row, 0) = t_grid[(std::size_t)f];
      pos(row, 1) = target_h[(std::size_t)(i / tpw)];
      pos(row, 2) = target_w[(std::size_t)(i % tpw)];
    }
  }
  L.video_runs.push_back({video_start, num_target_video});

  // 4. Indices, then tags. The reference writes text, then audio, then
  // video, and the video pass must win: a vision block's text rows are
  // tagged VIDEO by the caller and are inside the text range, so the
  // three passes are not disjoint the way they are in `fl2va`.
  L.video_indices.reserve((std::size_t)(num_ref_video + num_target_video));
  for (const RowRun& r : L.video_runs) {
    for (int i = 0; i < r.count; ++i) { L.video_indices.push_back(r.start + i); }
  }
  L.audio_indices.reserve((std::size_t)(num_ref_audio + num_target_audio));
  for (const RowRun& r : L.audio_runs) {
    for (int i = 0; i < r.count; ++i) { L.audio_indices.push_back(r.start + i); }
  }
  for (int i = 0; i < num_text; ++i) {
    L.token_tags[(std::size_t)i] = text_token_tags[(std::size_t)i];
  }
  for (int r : L.audio_indices) { L.token_tags[(std::size_t)r] = kAudioTag; }
  for (int r : L.video_indices) { L.token_tags[(std::size_t)r] = kVideoTag; }

  L.num_condition_video_rows = num_ref_video;
  L.num_condition_audio_rows = num_ref_audio;

  *out = std::move(L);
  return true;
}

void
build_row_timesteps(const PackedLayout& layout, float video_timestep,
                    float audio_timestep, float condition_video_timestep,
                    std::vector<float>* timesteps_out,
                    std::vector<int>* row_index_out,
                    float condition_audio_timestep)
{
  if (timesteps_out == nullptr || row_index_out == nullptr) { return; }
  const std::size_t seq = (std::size_t)layout.seq_len;
  std::vector<float> row(seq, video_timestep);
  // Through the INDEX vectors rather than the ranges: a `ref2va` layout
  // interleaves the two modalities, so there is no one range to walk.
  // The reference's order is conditioning video, generated audio, then
  // reference audio, which matters only if those overlapped (they do
  // not) -- it is kept anyway so a future overlap resolves the same way
  // in both.
  const int ncv = layout.num_condition_video_rows;
  const int nca = layout.num_condition_audio_rows;
  for (int i = 0; i < ncv && i < (int)layout.video_indices.size(); ++i) {
    row[(std::size_t)layout.video_indices[(std::size_t)i]] =
        condition_video_timestep;
  }
  for (int i = nca; i < (int)layout.audio_indices.size(); ++i) {
    row[(std::size_t)layout.audio_indices[(std::size_t)i]] = audio_timestep;
  }
  for (int i = 0; i < nca && i < (int)layout.audio_indices.size(); ++i) {
    row[(std::size_t)layout.audio_indices[(std::size_t)i]] =
        condition_audio_timestep;
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
