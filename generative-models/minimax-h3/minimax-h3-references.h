#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_REFERENCES_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_REFERENCES_H

#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

// Putting a `ref2va` reference onto MiniMax-H3's own rates and
// resolutions.
//
// A reference arrives at whatever the file carried; the model was
// conditioned at 24 fps, at one of two canvas rules, and at the audio
// VAE's sample rate. Nothing here decodes media -- the caller does that
// -- and nothing here touches the GPU: it is the geometry and the
// resampling, which is exactly the part that is wrong silently. A clip
// conditioned at the wrong speed still generates a video.
//
// The two visual rules are NOT the same one with different numbers:
//
//   * an IMAGE is encoded at high detail, at a short edge of its own
//     (2048 for the released checkpoint), upscaling included and with
//     NO area cap. It never binds the generated canvas.
//   * a VIDEO is put on the same canvas rule as the target -- short
//     edge, area cap, both axes rounded -- resolved from its OWN aspect
//     ratio, so two references of different shapes land on different
//     canvases.

// The canvas a reference is encoded at when the encoder may only
// REDUCE -- what a `short_edge` of 0 selects on both the image and the
// video path.
//
// Starts at the source's OWN extent, brings it under `max_pixels` if it
// is over (aspect preserved), and then FLOORS both axes to `multiple`.
// Floors rather than rounds to nearest, because the nearest grid line
// sits above as often as below: a 1080-tall source rounds UP to 1088,
// which is an upsample of exactly the kind this rule exists to refuse.
//
// The one upward move it cannot avoid is a source under one `multiple`.
// The grid is not a preference -- the VAE's stride times the DiT's patch
// is 32, and a latent the patch does not divide is refused 50 layers
// later -- so a 20-pixel edge becomes 32 and there is nowhere else to
// put it.
//
// `max_pixels` of 0 is uncapped. False on a non-positive extent, or
// outside the trained 1:4 .. 4:1 aspect.
bool resolve_canvas_within(int height, int width, int multiple,
                           std::int64_t max_pixels, int* out_h, int* out_w);

// The canvas a reference IMAGE is encoded at: the short edge scaled to
// `short_edge`, both axes then rounded to `multiple`, held under
// `max_pixels` (0 = uncapped, which is the released checkpoint's rule --
// an image reference is read at high detail and never binds the
// generated canvas). An image smaller than `short_edge` is scaled UP.
//
// `short_edge` of 0 is resolve_canvas_within above: take the picture at
// the size it arrived, which is what a caller that has already sized it
// -- a resample stage, a crop, a generated still -- is asking for.
//
// False when the geometry is not positive or the aspect is outside the
// trained 1:4 .. 4:1 range -- the same bound the target canvas holds to.
bool resolve_reference_image_size(int height, int width, int short_edge,
                                  int multiple, std::int64_t max_pixels,
                                  int* out_h, int* out_w);

// How many times each source frame is held to put `num_frames` frames at
// `src_fps` onto `dst_fps`.
//
// The reference resamples a CONSTANT frame rate the way ffmpeg's `fps`
// filter does -- every frame is held until the slot of the next one, and
// the last until the slot the stream's end rounds to -- so frames are
// dropped and duplicated WHOLE and none is ever blended. The returned
// vector has one count per source frame; a count of 0 means that frame
// is dropped.
//
// Empty when the rates are not positive. Rates that are equal give all
// ones, which is the parity-exact route: media already at 24 fps flows
// through untouched.
std::vector<int> frame_resample_counts(int num_frames, double src_fps,
                                       double dst_fps);

// PIL-faithful LANCZOS-3 resize of planar u8 RGB [3, H, W].
//
// LANCZOS specifically: the reference rescaled with ffmpeg's own while
// decoding, and a bilinear or box stand-in is visibly softer -- on the
// image path this is what the tower then reads, so the difference does
// not wash out.
bool resize_lanczos_planar_rgb(const std::uint8_t* rgb, int height, int width,
                               int out_h, int out_w,
                               std::vector<std::uint8_t>* out);

// The same resize with its per-GEOMETRY work lifted out of the per-FRAME
// work, for a caller that runs many frames through one (src -> dst)
// pair.
//
// A video reference is dozens of frames of ONE geometry, and the
// one-shot spelling above rebuilds the coefficient tables, reallocates
// the intermediate and re-derives the tap bounds for every one of them.
// MEASURED on a 39-frame 960x544 -> 1344x768 clip: 507 ms one-shot,
// 350 ms prepared, and the 120 MB results hash the same -- nothing here
// is a different filter, only the same one that stops re-deriving
// itself.
//
// The scratch lives in the plan rather than in the caller because the
// intermediate's size is a property of the geometry (`src_h` rows of
// `dst_w`), so a caller that held it would be re-deriving the one thing
// this type exists to remember.
struct PlanarResize {
  int src_h = 0, src_w = 0, dst_h = 0, dst_w = 0;
  // Pillow's coefficients: `bx[x]` is the first source column output
  // column x reads and `wx[x * kx ...]` its weights, and likewise down.
  int                kx = 0, ky = 0;
  std::vector<int>   bx, by;
  std::vector<float> wx, wy;
  // How many of those taps land INSIDE the source. The builder zeroes
  // the weights of the ones that do not, so stopping early and adding
  // 0 x whatever-is-there agree exactly -- this is the bounds test out
  // of the innermost loop, not a change of filter.
  std::vector<int>   nx, ny;
  // The horizontal pass's u8 intermediate ([src_h, dst_w]) and one
  // output row of the vertical pass's accumulator.
  std::vector<std::uint8_t> mid;
  std::vector<double>       acc;

  bool empty() const { return kx <= 0 || ky <= 0; }
};

// Build the plan for one (src -> dst) geometry. False on a non-positive
// extent.
bool prepare_planar_resize(int src_h, int src_w, int dst_h, int dst_w,
                           PlanarResize* out);

// Run one planar [3, src_h, src_w] u8 frame through it, into a caller-
// owned [3, dst_h, dst_w]. Not re-entrant on one plan -- the scratch is
// shared -- and false when the plan is empty.
bool run_planar_resize(PlanarResize& plan, const std::uint8_t* rgb,
                       std::uint8_t* out);

// An image reference onto its own canvas. `out_h` / `out_w` come back as
// the resolved size; the pixels are untouched when they already match,
// which is the parity-exact route.
bool normalize_image_reference(const std::uint8_t* rgb, int height, int width,
                               int short_edge, int multiple,
                               std::int64_t max_pixels,
                               std::vector<std::uint8_t>* out, int* out_h,
                               int* out_w, std::string* err = nullptr);

// A video reference onto 24 fps, truncated to the generated length, and
// onto the canvas its OWN aspect ratio resolves to.
//
// `frames` is `num_frames` consecutive planar [3, H, W] u8 planes. The
// three passes run in the reference's order -- rate first, truncate
// second, rescale third -- which matters: rescaling before the drop
// would resample frames that are then thrown away, and truncating
// before the rate resample would cut the wrong ones.
//
// `target_frames` is the generated frame count the reference is
// truncated to; a clip shorter than that is left short (the layout is
// built from what this produces, so a short reference is a smaller
// conditioning block, not an error).
//
// `short_edge` of 0 means THE SOURCE'S OWN short edge, which is how a
// caller asks for a clip that is never scaled UP. It is not a different
// rule -- the aspect resolve, the `max_pixels` cap and the rounding all
// still run, and the cap is still what binds a large source. What it
// changes is the small one: the released checkpoint's 768 puts a
// 960x544 clip on a 1344x768 canvas, which is 1.98x the pixels and
// carries no more information than the 960x544 it was interpolated
// from.
//
// Cheaper is not the same as equivalent, and this canvas reaches
// further than this function does. Downstream it sets the VAE encode,
// the reference rows the DiT attends to on every step, AND -- the one
// that is easy to miss from here -- how many vision tokens the
// conditioner gets, because the tower is handed these same pixels.
// See ReferencePlan::canvas_short_edge; costed in docs/MINIMAX-H3.md.
bool normalize_video_reference(const std::uint8_t* frames, int num_frames,
                               int height, int width, double src_fps,
                               int target_frames, int multiple, int short_edge,
                               std::int64_t max_pixels,
                               std::vector<std::uint8_t>* out, int* out_frames,
                               int* out_h, int* out_w, double dst_fps = kFps,
                               std::string* err = nullptr);

// A reference soundtrack: truncated to the generated duration and
// upmixed to stereo.
//
// TRUNCATION HAPPENS AT THE SOURCE RATE, before any resampling -- the
// reference cuts `int(seconds * sample_rate)` samples off the decoded
// waveform and resamples once, and cutting after would land on a
// different sample.
//
// Resampling is deliberately NOT here. The rate a reference arrives at
// is the decoder's business, and every decoder in this tree already goes
// through swresample; a second resampler here would be a worse one, and
// two of them in the same path is how a clip ends up filtered twice.
// Pass `pcm` already at the audio VAE's rate.
//
// `pcm` is planar `[channels, samples]`, mono or stereo. Mono is upmixed
// by REPEATING the channel, which is what the reference does -- the VAE
// is mono and takes the pair as two batch items, so a silent second
// channel would be conditioning on silence.
bool normalize_audio_reference(const float* pcm, int channels, int samples,
                               int sample_rate, double max_seconds,
                               std::vector<float>* out, int* out_samples,
                               std::string* err = nullptr);

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
