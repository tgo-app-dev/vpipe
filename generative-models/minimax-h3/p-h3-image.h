#ifndef VPIPE_GENERATIVE_MODELS_MINIMAX_H3_P_H3_IMAGE_H
#define VPIPE_GENERATIVE_MODELS_MINIMAX_H3_P_H3_IMAGE_H

// p-h3-image.h -- P-H3-Image (PrunaAI), an IMAGE model on MiniMax-H3.
//
// Text-to-image and instruction editing (up to 8 pictures) on H3's own
// DiT: the image is ONE latent frame, denoised in H3's packed sequence
// beside two audio latents that are thrown away -- `t2va`'s layout for
// text-to-image, `ref2va`'s with image references only for an edit (the
// pictures packed ahead of the generated rows, each on one integer slot
// of the rotary clock). The weights are FL2VA's plus a rank-128 adapter
// (PrunaAI ship both the merged DiT and the adapter), guidance-distilled:
// 8 Euler steps over the checkpoint's own grid, no CFG, no negative.
//
// What is new around the DiT, every piece of it read off the checkpoint:
//   * the TEXT ENCODER is Qwen3-VL-4B, not H3's 32B, and conditions
//     through a CONNECTOR (`text_connector/`, PH3ImageTextConnector):
//     hidden_states [13, 19, 25, 31, 36] concatenated -- the last is the
//     FINAL NORMED state, every other the raw residual -- standardized,
//     projected to H3's 5120 in f32, destandardized, and the first row
//     replaced by a learned sink. The prompt is used VERBATIM, as H3's.
//   * the VAE (`AutoencoderKLPH3Image`) is H3's encoder BYTE FOR BYTE --
//     so the latent space is H3's -- with a ViT decoder retrained to
//     decode one latent frame into one picture (see
//     MetalMiniMaxH3VideoVae::Config::dec_patch_t / t1_time).
//   * the manifest (`modular_model_index.json`, PH3ImageModularPipeline)
//     carries the sampling grid and the edit's sizing rules (below).
//
// This header is the RULES, nothing that runs a model: the manifest, the
// reference pipeline's sizing arithmetic (Python's round, half to even)
// and its edit prompt. Each is restated from diffusers'
// modular_pipelines/p_h3_image, read as data.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
namespace genai {
namespace ph3 {

// `_class_name`s the checkpoint's configs carry.
inline constexpr const char* kPipelineClass  = "PH3ImageModularPipeline";
inline constexpr const char* kVaeClass       = "AutoencoderKLPH3Image";
inline constexpr const char* kConnectorClass = "PH3ImageTextConnector";
// The manifest's file name (a modular pipeline's, not model_index.json).
inline constexpr const char* kManifestFile = "modular_model_index.json";

// The image decoder's rotary time: where the first token of a 7-token
// chunk sits on [-1, 1) (PH3ImageViTDecoder's FRAME_POSITION).
inline constexpr double kFramePosition = 2.0 * (0.5 / 7.0) - 1.0;
// Audio latents per stereo channel denoised beside the image, discarded.
inline constexpr int kAudioLatents = 2;
// The t an edit's picture rows are noised to and held at (H3's t: 1 is
// clean), and the seed their VAE posterior is SAMPLED under -- a fresh
// generator per picture, independent of the request's.
inline constexpr double        kConditionNoiseAug = 0.999;
inline constexpr std::uint64_t kConditionEncodeSeed = 42;
// What a canvas side must be a multiple of: the VAE's 16 times the DiT's
// patch of 2.
inline constexpr int kCanvasMultiple = 32;
// Text-to-image's default canvas (8 megapixels, 1:1) and the most
// pictures an edit takes.
inline constexpr int kDefaultSide   = 2912;
inline constexpr int kMaxEditImages = 8;

// The manifest's figures, each defaulted to the released checkpoint's.
struct Manifest {
  // The sampling grid AS RUN (1 -> 0; no shift is applied to it): 8
  // forwards over its 9 points.
  std::vector<float> sample_sigmas = {1.0f,      0.984753f, 0.967905f,
                                      0.951688f, 0.925372f, 0.881923f,
                                      0.810083f, 0.528718f, 0.0f};
  // An edit's pictures keep their size inside this AREA range (else are
  // scaled to its nearer end); the picture to edit is the canvas, held
  // at least at `canvas_min_pixels` when the request names no size.
  std::int64_t reference_min_pixels = 1024LL * 1024LL;
  std::int64_t reference_max_pixels = 8LL * 1024LL * 1024LL;
  std::int64_t canvas_min_pixels    = 4LL * 1024LL * 1024LL;
  // The vision tower reads its own copy of each picture at about this
  // many pixels -- not the VAE's.
  std::int64_t vision_pixels = 1000000;
};

// Whether `dir` is a P-H3-Image root: its manifest names the pipeline.
bool is_root(const std::string& dir);

// The connector's directory (`text_connector/`) of a P-H3-Image root,
// "" for any other root or when it is not there. One answer for the
// stages that claim it, phase it and release it with the encoder: the
// encoder holds it, so it lives and goes with the encoder.
std::string connector_dir(const std::string& root);

// Read `root`'s manifest. False (with `err`) when it is absent, not this
// pipeline's, or carries a grid that is not strictly decreasing from 1
// to exactly 0.
bool read_manifest(const std::string& root, Manifest* out,
                   std::string* err = nullptr);

// ---- the reference's sizing (Python's round(): half to even) ----------

// `round(x)` as Python's builtin rounds a float.
long long py_round(double x);

// A picture resized to ABOUT `pixels` at its own aspect, both sides
// multiples of `multiple`; one already on the grid and within 20% of
// the area keeps its size (`_size_at_area`).
void size_at_area(int width, int height, std::int64_t pixels, int multiple,
                  int* out_w, int* out_h);

// A picture that keeps its size, or is scaled at its own aspect to the
// nearer end of [min_pixels, max_pixels] when its area is outside it,
// both sides then rounded to `multiple` (`_size_in_area_range`).
void size_in_area_range(int width, int height, std::int64_t min_pixels,
                        std::int64_t max_pixels, int multiple, int* out_w,
                        int* out_h);

// A requested canvas rounded to the nearest multiples (`round_canvas`).
void round_canvas(int height, int width, int multiple, int* out_h,
                  int* out_w);

// An edit's instruction as the encoder reads it: prefixed "Edit <Picture
// 1>. " unless it already opens with that sentence or with an upsampled
// brief's first section ("subject_definitions:").
std::string edit_prompt(std::string_view prompt);

}  // namespace ph3
}  // namespace genai
}  // namespace vpipe

#endif
