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

// The canvas a reference IMAGE is encoded at: the short edge scaled to
// `short_edge`, both axes then rounded to `multiple`. No area cap, and
// an image smaller than `short_edge` is scaled UP.
//
// False when the geometry is not positive or the aspect is outside the
// trained 1:4 .. 4:1 range -- the same bound the target canvas holds to.
bool resolve_reference_image_size(int height, int width, int short_edge,
                                  int multiple, int* out_h, int* out_w);

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

// An image reference onto its own canvas. `out_h` / `out_w` come back as
// the resolved size; the pixels are untouched when they already match,
// which is the parity-exact route.
bool normalize_image_reference(const std::uint8_t* rgb, int height, int width,
                               int short_edge, int multiple,
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
