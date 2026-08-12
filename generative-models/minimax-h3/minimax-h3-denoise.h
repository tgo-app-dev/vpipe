#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_DENOISE_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_DENOISE_H

#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <functional>
#include <string>

namespace vpipe {
namespace genai {

// The MiniMax-H3 denoise loop: one packed sequence, two modalities, two
// sigma schedules stepped in lockstep.
//
// Four things about this loop are unlike the other diffusion drivers
// here, and all four come straight from the model's structure:
//
//   * ONE forward per step. The checkpoint is guidance-distilled, so
//     there is no unconditional pass and no CFG -- a negative prompt has
//     nothing to guide away from, and adding a second forward would
//     double the cost of a 33B model for nothing.
//   * TWO schedules. Video runs at shift 12.0 and audio at 3.0, over the
//     same step count, so step i of one always pairs with step i of the
//     other. They are stepped on their own slices of one sequence.
//   * The CONDITION rows are never stepped. They lead their modality's
//     buffer, carry their own timestep in the AdaLN table, and are read
//     by every other row through attention -- but only the generated
//     rows are ever written, so they survive the whole loop unchanged.
//     Video has always had them (a keyframe anchor); `ref2va` adds them
//     on the AUDIO side too, one block per reference soundtrack.
//   * The state is float32 while the transformer is bf16. The loop runs
//     20+ times over a latent that starts as unit noise, and the
//     reference steps in float32 for exactly that reason; the bf16 round
//     trip happens per forward, not per accumulation.
struct DenoiseRequest {
  MetalMiniMaxH3Transformer*      dit    = nullptr;
  const minimax_h3::PackedLayout* layout = nullptr;
  // Text conditioning [num_text_rows, text_dim] bf16 -- the Qwen3-VL
  // layer-50 tap, unchanged across steps.
  const metal_compute::SharedBuffer* text = nullptr;

  // Caller-owned f32 state, updated IN PLACE.
  //
  // `video` is [num_condition_rows + num_video_rows, video_patch_elems]
  // in the layout's `video_indices` order -- conditioning rows FIRST,
  // which is what makes "step only the tail" a contiguous slice.
  float* video = nullptr;
  // [num_audio_rows, audio_channels].
  float* audio = nullptr;

  int    num_steps   = 32;
  double video_shift = 12.0;
  double audio_shift = 3.0;
  // The timestep the pinned keyframe rows are conditioned on. They are
  // real encoded frames, so 1.0 -- CLEAN in this model's convention,
  // where t = 1 - sigma and t = 1 means no noise. Exposed because it is
  // the reference's `condition_video_timestep` rather than a constant,
  // and a checkpoint trained with noise-augmented anchors would want it
  // lower.
  float condition_timestep = 1.0f;
  // The level a `ref2va` REFERENCE soundtrack sits at. Unused by
  // `t2va` / `fl2va`, which carry no reference audio rows at all: those
  // layouts report `num_condition_audio_rows == 0` and the whole audio
  // buffer is generated.
  float condition_audio_timestep = 1.0f;

  // Per-step progress. Return false to stop early; the state keeps
  // whatever steps have run.
  std::function<bool(int step, int total)> progress;

  // Adopt a caller's per-generation choices. Here rather than at each
  // call site so a field added to GenerationParams reaches the loop
  // without every driver having to be found and updated -- which is the
  // failure mode a knob silently keeping its default looks like.
  void
  set_params(const MetalMiniMaxH3Transformer::GenerationParams& p)
  {
    video_shift = p.video_shift;
    audio_shift = p.audio_shift;
    condition_timestep = (float)p.condition_timestep;
    condition_audio_timestep = (float)p.condition_audio_timestep;
  }
};

// Run the loop. False on failure, with a reason in `err`. On success the
// request's `video` / `audio` hold the denoised latents, still in packed
// row order -- unpatchifying them into a latent grid is the caller's job
// (it owns the geometry the layout was built from).
bool denoise(const DenoiseRequest& req, std::string* err = nullptr);

}  // namespace genai
}  // namespace vpipe

#endif
