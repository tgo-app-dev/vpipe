#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_LAYOUT_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_LAYOUT_H

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

// MiniMax-H3 runs ONE stack of blocks over ONE packed 1-D sequence
// holding the text condition, the keyframe conditioning rows, the audio
// rows and the target video rows. The transformer itself knows none of
// that -- it is handed a row-per-modality tag, a row-per-timestep index
// and a (t, h, w) rotary coordinate per row, and does full
// self-attention over whatever it is given. So the layout below IS the
// model's structure, and everything downstream addresses rows through
// it.
//
// This file is the layout and nothing else: pure arithmetic over ints
// and doubles, no GPU and no weights, which is what makes it testable
// on its own.

// The three modality tags. The AdaLN table row of a sequence row is
// `timestep_index * kModalityNum + tag`, so these values are load-bearing
// and not just labels.
constexpr int kVideoTag    = 0;
constexpr int kTextTag     = 1;
constexpr int kAudioTag    = 2;
constexpr int kModalityNum = 3;

constexpr double kFps                   = 24.0;
constexpr int    kAudioLatentsPerSecond = 40;
constexpr int    kAudioChannels         = 2;

// The rotary grid constants. Spatial coordinates are normalized by the
// canvas aspect and scaled up by 32; the temporal axis advances by
// 5/3 * (1, 4, 4, 4, 4) per latent frame, i.e. NON-uniformly -- the
// first latent of each group of five covers one pixel frame and the
// other four cover four each, which is the video VAE's 17-frames-to-5-
// latents chunking seen from the rotary side.
constexpr double kRopeFrameRescale   = 5.0 / 3.0;
constexpr int    kRopeSpatialScale   = 32;
constexpr int    kRopeFramesPerLatent[5] = {1, 4, 4, 4, 4};

// ---- geometry ---------------------------------------------------------

// Resolve a display aspect ratio to a canvas. The short edge starts at
// `short_edge`, the area is capped at `max_pixels`, then BOTH axes round
// to `multiple` -- so the rounded area can land slightly above the cap.
// Only the ratio of the first two arguments matters. False when the
// ratio is non-positive or outside the trained 1:4 .. 4:1 range.
bool resolve_canvas_size(double aspect_w, double aspect_h, int multiple,
                         int short_edge, std::int64_t max_pixels, int* out_h,
                         int* out_w);

// Snap a frame count UP to the next `frames_per_chunk * n +
// latents_per_chunk` the video VAE can encode (17n + 5 for the released
// checkpoint). 0 when `num_frames` is not positive.
int align_num_frames(int num_frames, int frames_per_chunk,
                     int latents_per_chunk);

// Latent frames produced for an ALIGNED frame count: 5n + 2. 0 when
// `num_frames` is not aligned.
int video_latent_num_frames(int num_frames, int frames_per_chunk,
                            int latents_per_chunk);

// Audio latents covering `num_frames` video frames, rounded at the
// latent grid.
int audio_latent_num_frames(int num_frames, double fps = kFps,
                            int latents_per_second = kAudioLatentsPerSecond);

// ---- rotary grids -----------------------------------------------------

// One aspect-normalized spatial axis: `dim / patch` coordinates centred
// on the unit interval and scaled by 32, with the RIGHT ENDPOINT
// EXCLUDED. Reproduces numpy's `linspace(..., endpoint=False)`, which is
// `start + arange(num) * (stop - start) / num` -- not what torch's
// linspace computes, and the two disagree at every point but the first.
std::vector<double> spatial_position_grid(int dim, int patch,
                                          double sqrt_area);

// The rotary time of every latent frame, starting at `origin`.
std::vector<double> temporal_position_grid(int num_latent_frames,
                                           double origin);

// ---- the packed sequence ----------------------------------------------

// Which end of the clip a keyframe conditioning block is anchored to.
enum class Anchor { kFirst, kLast };

// A contiguous destination range of one modality's rows.
//
// `t2va` / `fl2va` give every modality ONE range (video two: the
// conditioning block and the target block), which is why the layout
// below also carries plain start/count fields for them. `ref2va` does
// not: a video reference's soundtrack rows are packed immediately
// before its own video rows, so with several references neither audio
// nor video is a single range any more, and the rows of one modality
// are interleaved with the other's.
//
// A run is still CONTIGUOUS, though, which is what makes this the right
// shape rather than a per-row index: each run is one GEMM with a
// destination offset, exactly as the single-range case already was.
struct RowRun {
  int start = 0;   // first packed row of the run
  int count = 0;   // rows in it
};

struct PackedLayout {
  // Row-major [seq_len, 3] rotary coordinates, in the (t, h, w) order
  // the transformer's rope reads. Kept in DOUBLE all the way to the
  // cos/sin evaluation: these are absolute positions scaled by 32 over
  // sequences tens of thousands long, so the angles are large and
  // accumulating them in float would quantize the grid visibly.
  std::vector<double> position_ids;
  std::vector<int>    token_tags;    // [seq_len], one of the three tags

  // The packed order is [text | keyframe conditions | target audio |
  // target video], so every modality's rows are CONTIGUOUS and the
  // "indices" the reference passes as tensors are ranges here. Video is
  // the one exception: its rows are the conditioning block AND the
  // target block, which are not adjacent, so it takes two ranges.
  int seq_len            = 0;
  int num_text_rows      = 0;
  int condition_start    = 0;   // first keyframe conditioning row
  int num_condition_rows = 0;   // leading VIDEO rows that are conditioning
  int audio_start        = 0;
  int num_audio_rows     = 0;
  int video_start        = 0;   // first GENERATED video row
  int num_video_rows     = 0;   // generated video rows only

  // Positions of the video rows in packed order (conditioning first),
  // i.e. the reference's `video_indices`. Materialized because the two
  // ranges are not adjacent and the output head's row selection needs
  // them in this order.
  std::vector<int> video_indices;

  // The same for audio, reference rows first. `t2va` / `fl2va` have no
  // reference audio, so this is just [audio_start, +num_audio_rows) --
  // it is materialized anyway so that everything downstream reads ONE
  // shape whichever task built the layout.
  std::vector<int> audio_indices;

  // Each modality's rows as contiguous runs, in the order the indices
  // above list them. Video is {conditioning, target} for `fl2va` and
  // one run per image/video reference plus the target for `ref2va`;
  // audio is a single target run for `fl2va`.
  std::vector<RowRun> video_runs;
  std::vector<RowRun> audio_runs;

  // How many LEADING entries of the index vectors are conditioning
  // rather than generated rows. `num_condition_video_rows` is
  // `num_condition_rows` under the name the general path uses; audio
  // conditioning exists only in `ref2va`.
  int num_condition_video_rows = 0;
  int num_condition_audio_rows = 0;
};

// One reference of a `ref2va` request, as the LAYOUT sees it: what
// modality it is and the shape of what the encoders made of it. The
// media itself never reaches here.
//
// `num_latent_frames` / `latent_height` / `latent_width` are the video
// conditioning latent's own geometry -- a reference is encoded at a
// resolution of its own and never binds the generated canvas, so these
// are genuinely per-reference and not derivable from the target.
struct Reference {
  enum class Kind { kImage, kVideo, kAudio };

  Kind kind = Kind::kImage;

  // Video geometry, for kImage (one latent frame) and kVideo.
  int num_latent_frames = 0;
  int latent_height     = 0;
  int latent_width      = 0;

  // Audio latents PER CHANNEL, or 0 for a reference with no soundtrack.
  // Always 0 for kImage; a kAudio reference is nothing but this.
  int num_audio_latents = 0;

  bool has_audio() const { return num_audio_latents > 0; }
};

// Build the `[text | keyframe conditions | target audio | target video]`
// layout of the `t2va` / `fl2va` tasks.
//
// `text_token_tags` is the modality tag of every text row -- text is
// tagged 1, EXCEPT the rows of a keyframe's vision block, which
// MiniMax-H3 tags 0 (video). That is why this is a per-row input rather
// than a count: the caller building the prompt is the only thing that
// knows where the vision blocks are.
//
// False on an inconsistent request (non-positive geometry, a latent
// size not divisible by the patch).
bool build_packed_sequence(const std::vector<int>& text_token_tags,
                           int num_latent_frames, int latent_height,
                           int latent_width, int num_audio_latents,
                           int patch_h, int patch_w, int audio_channels,
                           const std::vector<Anchor>& keyframe_anchors,
                           PackedLayout* out);

// Build the `[text | reference blocks | target audio | target video]`
// layout of the `ref2va` task.
//
// `references` is in the order the model READS them, which is load-
// bearing twice over: it numbers the `<Picture i>` / `<Audio j>` /
// `<Video k>` labels of the prompt presentation, and it advances the
// shared audio/video rotary clock. A different order is a different
// request, not a different spelling of one.
//
// The clock starts where the text rows end and every reference pushes
// it forward by the time that reference occupies: an image takes a
// single integer slot (NOT a latent frame's 5/3 units), a soundtrack
// one unit per audio latent, and a video the LONGER of its own two.
// A video reference's soundtrack rows are packed immediately before its
// video rows and share their origin, which is why the two modalities
// interleave and `audio_runs` exists.
//
// False on an inconsistent request: a reference whose geometry is not
// positive or not divisible by the patch, or an empty list (a request
// with no references is `t2va`, not a degenerate `ref2va`).
bool build_ref2va_packed_sequence(const std::vector<int>& text_token_tags,
                                  const std::vector<Reference>& references,
                                  int num_latent_frames, int latent_height,
                                  int latent_width, int num_audio_latents,
                                  int patch_h, int patch_w, int audio_channels,
                                  PackedLayout* out);

// Assign a timestep to every row and reduce it to the transformer's
// `(distinct timesteps, per-row index)` pair.
//
// One forward serves rows at DIFFERENT noise levels: the generated video
// and audio rows step down their own schedules while the conditioning
// rows stay pinned at their noise-augmentation level. Text rows never
// reach an output head and inherit the video timestep.
//
// `timesteps_out` comes back sorted and distinct; `row_index_out` is
// per-row into it, which is exactly what the AdaLN table indexes with.
//
// `condition_audio_timestep` is the level a `ref2va` REFERENCE
// soundtrack sits at -- it is conditioning the model reads rather than
// a stream it denoises, so it never moves. It is unused by `t2va` /
// `fl2va`, which have no reference audio rows at all.
void build_row_timesteps(const PackedLayout& layout, float video_timestep,
                         float audio_timestep, float condition_video_timestep,
                         std::vector<float>* timesteps_out,
                         std::vector<int>* row_index_out,
                         float condition_audio_timestep = 1.0f);

// `timestep_index * kModalityNum + tag` for every row -- the AdaLN table
// row each sequence row reads its six modulation parameters from.
std::vector<int> build_adaln_indices(const PackedLayout& layout,
                                     const std::vector<int>& row_timestep_index);

// ---- VAE tiling -------------------------------------------------------

// One axis's tile layout, in PIXELS. `overlap` has one entry fewer than
// `start` / `length`: it is the overlap BETWEEN neighbours.
struct TileSplit {
  std::vector<int> start;
  std::vector<int> length;
  std::vector<int> overlap;
};

// Lay `tile_size`-wide tiles over `length` pixels.
//
// The video VAE has to tile -- its decoder is a transformer, so
// attention is quadratic in latent voxels and a full frame is not a slow
// decode but an impossible one. The encoder does not have that problem,
// but it MUST tile the same way regardless: a tile decoded on its own
// sees a rope grid built from its own extent, so an encode and a decode
// that disagree about the boundaries do not merely lose efficiency, they
// evaluate different functions.
//
// The count is the fewest tiles whose union covers `length` while every
// overlap stays at least `min_overlap`. The slack is then spread
// round-robin over the overlaps in whole `ratio` steps, so every tile
// boundary lands on a latent cell -- the leftover under `ratio` is
// absorbed by the tiles themselves rather than by an unaligned overlap.
//
// A `tile_size` at or above `length` gives one tile and no overlaps.
TileSplit split_tiles(int length, int tile_size, int min_overlap, int ratio);

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
