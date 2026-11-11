#ifndef VPIPE_GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_IMAGE_H
#define VPIPE_GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_IMAGE_H

// minimax-h3-image.h -- MiniMax-H3-Image (MiniMaxAI), the H3 family's
// IMAGE model: text-to-image and editing with up to 16 pictures, about
// 16 megapixels at any aspect, 6 steps, no guidance.
//
// The DiT is H3's stack WITHOUT its audio branch (`audio_enabled`
// false: no audio patch projection or head), distilled twice over:
// guidance is in the weights, and the 6 steps of its fixed grid
// (`pdd_config.json`) each have their OWN output head, which replaces the
// checkpoint's `final_layer.video_out` -- `pdd_fused_heads.safetensors`,
// `video_out.weight` [steps, 96, hidden] (Parallel Decoding
// Distillation). The packed sequence is H3's reference layout with no
// audio rows at all: [text | reference pictures | the picture], the
// picture ONE latent frame, each reference on one integer slot of the
// rotary clock after the text, the picture on the slot after them.
//
// Around it:
//   * the TEXT ENCODER is MiniMax-H3's own (Qwen3-VL-32B, hidden state
//     50), which this repository does not carry -- `model_index.json`
//     names `MiniMaxAI/MiniMax-H3`'s `text_encoder/`. The prompt is
//     used VERBATIM, truncated to 4096 tokens; an edit puts a
//     "Picture i: " label (no angle brackets, unlike H3's
//     "<Picture i>: ") and a vision block per picture ahead of it.
//   * the VAE (`MiniMaxH3ImageVAE`) is H3's encoder with a 2D ViT
//     decoder: one latent frame into one picture, rotary over (h, w)
//     only (see MetalMiniMaxH3VideoVae::Config::rope_axes), tiled at
//     256 px with a 64 px overlap as released.
//   * SAMPLING: the grid's sigmas as the release steps them (each
//     rounded through its float32 timestep 1 - sigma), the Euler update
//     in its VELOCITY form, x + (sigma - sigma_next) v in float32 (the
//     heads were distilled with it; the blend form differs in the last
//     bit, which the bf16 forward amplifies), step i through head i.
//   * an edit's pictures keep their size within short edges [1024,
//     2048] (else are scaled to the nearer end), the tower's copy at a
//     1024 short edge; each posterior is SAMPLED off the request's
//     generator, rounded to float16, cropped to whole patches, whitened,
//     and noised to t = 0.999 with noise drawn after the picture's.
//   * a picture smaller than `min_generation_area` (default 4 MP) is
//     generated on a larger canvas of its aspect and resized back with
//     Lanczos; canvas sides are multiples of 32.
//
// This header is the RULES, nothing that runs a model. Each is restated
// from diffusers' modular_pipelines/minimax_h3_image (huggingface/
// diffusers#15230) and the release's own files, read as data.

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace h3i {

// `_class_name`s the checkpoint's configs carry.
inline constexpr const char* kPipelineClass = "MiniMaxH3ImagePipeline";
inline constexpr const char* kVaeClass      = "MiniMaxH3ImageVAE";
// The two files beside model_index.json.
inline constexpr const char* kScheduleFile = "pdd_config.json";
inline constexpr const char* kHeadsFile    = "pdd_fused_heads.safetensors";
// The heads' tensors: [steps, out, hidden] and [steps, out].
inline constexpr const char* kHeadsWeight = "video_out.weight";
inline constexpr const char* kHeadsBias   = "video_out.bias";

// What a canvas side must be a multiple of: the VAE's 16 times the DiT's
// patch of 2.
inline constexpr int kCanvasMultiple = 32;
// The returned picture's size when the request names none, and the
// smallest area generated (a smaller picture is generated larger and
// resized back): the release's defaults, 2048 x 2048.
inline constexpr int          kDefaultSide    = 2048;
inline constexpr std::int64_t kDefaultMinArea = 2048LL * 2048LL;
// The range `min_generation_area` is taken in.
inline constexpr std::int64_t kMinAreaFloor = 1024;
inline constexpr std::int64_t kMinAreaCeil  = 4096LL * 4096LL;
// The t an edit's picture rows are held at (H3's t: 1 is clean).
inline constexpr float kReferenceNoiseAug = 0.999f;
// An edit picture's short edge for the VAE (kept inside, else scaled to
// the nearer end), and for the vision tower.
inline constexpr int kReferenceMinShortEdge = 1024;
inline constexpr int kReferenceMaxShortEdge = 2048;
inline constexpr int kVisionShortEdge       = 1024;
// The most tokens the prompt, and each picture's label, keep.
inline constexpr int kMaxSequenceLength = 4096;
// The most pictures an edit takes (the model card's).
inline constexpr int kMaxReferences = 16;
// The text encoder's tap: `hidden_states[50]`, as MiniMax-H3's.
inline constexpr int kTextEncoderLayer = 50;

// The release's figures, each defaulted to the shipped checkpoint's.
struct Manifest {
  // pdd_config.json: the grid as WRITTEN (double), 1 -> exactly 0, one
  // forward and one head per step.
  std::vector<double> sigmas = {1.0,
                                0.9675236940383911,
                                0.8875485062599182,
                                0.75,
                                0.5327739119529724,
                                0.23200887441635132,
                                0.0};
  int steps() const { return (int)sigmas.size() - 1; }
  // The per-step heads' file, absolute; empty when it is not there.
  std::string heads_file;
  // Where the text encoder is: model_index.json's `text_encoder` entry,
  // a repository and a folder in it.
  std::string text_encoder_repo      = "MiniMaxAI/MiniMax-H3";
  std::string text_encoder_subfolder = "text_encoder";
};

// Whether `dir` is a MiniMax-H3-Image root: its model_index.json names
// the pipeline.
bool is_root(const std::string& dir);

// Read `root`'s model_index.json and pdd_config.json. False (with `err`)
// when the root is not this pipeline's, or its grid is not strictly
// decreasing from 1 to exactly 0 with one step per head.
bool read_manifest(const std::string& root, Manifest* out,
                   std::string* err = nullptr);

// The grid as the release STEPS it: each sigma but the last rounded
// through its float32 timestep, sigma' = f32(1 - f32(1 - sigma)), the
// terminal 0 kept. The timesteps the DiT reads are f32(1 - sigma').
std::vector<float> grid_as_run(const std::vector<double>& sigmas);

// The canvas a `height` x `width` picture is generated on: scaled up at
// its aspect to `min_area` when it is smaller, both sides then rounded
// to the nearest `multiple` (Python's round, half to even), at least
// `multiple` (get_image_canvas_size).
void canvas_size(int height, int width, std::int64_t min_area, int multiple,
                 int* out_h, int* out_w);

// An edit picture's size for one consumer: kept when its short edge is
// within [min_edge, max_edge], else scaled at its aspect so the short
// edge lands on the nearer end, both sides rounded to `multiple`
// (bound_short_edge).
void bound_short_edge(int width, int height, int min_edge, int max_edge,
                      int multiple, int* out_w, int* out_h);

// A reference picture's label as the encoder reads it: "Picture i: "
// (1-based), no angle brackets.
std::string picture_label(int index);

}  // namespace h3i
}  // namespace genai
}  // namespace vpipe

#endif
