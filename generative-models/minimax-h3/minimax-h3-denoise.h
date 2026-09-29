#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_DENOISE_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_DENOISE_H

#include "common/flex-data.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <functional>
#include <string>
#include <vector>

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
//
// What a denoise did, for a caller that wants to report or assert it.
// `forwards` is the transformer calls that RAN; `reused` the steps
// MotionCache answered from its residual instead. They sum to `steps`.
// `motion_cache` says the cache was ARMED for the clip, which is what
// tells "asked, and nothing was close enough to reuse" from "asked, and
// the geometry turned it off".
struct DenoiseReport {
  int  steps        = 0;
  int  forwards     = 0;
  int  reused       = 0;
  bool motion_cache = false;
};

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

  // The GENERATED video's latent grid, patches not pixels. Only VDN's
  // hybrid attention reads it -- its window is over whole frames and its
  // short conv is a (t, h, w) stencil, neither of which can be recovered
  // from a row count -- and left at zero the hybrid stays off. So a
  // driver that forgets it does not get a wrong answer, it gets the
  // unhybridised model; which is the failure worth having, and the
  // reason the transformer's own switch is the same field.
  int    video_grid_h = 0;
  int    video_grid_w = 0;

  // Denoising steps -- one transformer FORWARD each, which is what
  // diffusers' `num_inference_steps` and ComfyUI's `steps` count: N steps
  // run N forwards over the N + 1 sigmas of linspace(1, 0, N + 1), the
  // last of them the terminal zero.
  //
  // MiniMax's own reference scheduler counts differently -- its
  // `num_inference_steps` is the number of GRID POINTS, terminal zero
  // included, so it runs one forward fewer than it says -- and
  // MiniMaxH3Scheduler::set_timesteps keeps that signature so it can be
  // checked against the reference directly. The loop asks it for
  // num_steps + 1 points. Matching the tools people come from matters
  // more than matching one reference's argument: a step count copied from
  // a ComfyUI workflow, or a Turbo adapter's "4-step" / "8-step" name,
  // means forwards, and read the other way it lands one short -- off the
  // grid a few-step distillation was trained on.
  int    num_steps   = 32;
  // An explicit RAW sigma grid (1 -> 0, unshifted), overriding
  // `num_steps`. Each modality pushes it through its own shift, as with
  // the linspace grid.
  //
  // Left empty, a live FLOW-MAP adapter on the DiT supplies its own --
  // the grid it was distilled on is a property of its weights, the way
  // upstream's loader configures the pipeline from the file -- and
  // `num_steps` is ignored. An explicit grid here still wins.
  std::vector<float> sigma_grid;
  // Leading steps whose attention runs DENSE with Sol-Attn on (see
  // Step::dense_attention). 0 routes every step.
  int sol_dense_steps = 0;
  // The acceleration bag (shared/accel-settings.h), for the tiers the
  // SAMPLING LOOP implements rather than the transformer -- MotionCache
  // (shared/motion-cache.h), which decides per step whether to run the
  // forward at all. Borrowed; null reads as every default, i.e. off.
  const FlexData* accel = nullptr;
  // Filled when set.
  DenoiseReport* report = nullptr;
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

  // A LATENT PREVIEW. After step `step` (1-based, of `total`) for which
  // `preview_due` says yes, `preview` gets the model's clean estimate
  // x0 = x + sigma * v of the GENERATED video rows --
  // [num_video_rows, video_patch_elems] f32, in packed order, the tail
  // `video` holds past its conditioning rows. Borrowed for the call.
  //
  // x0 rather than the state: the state is mostly noise for the first
  // half of the schedule, and x0 is what the model currently believes
  // the clip is. Asking first means a step nobody previews builds
  // nothing; both empty is no previews.
  std::function<bool(int step, int total)> preview_due;
  std::function<void(int step, int total, const float* x0)> preview;

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

// MotionCache's view of this loop's packed state (shared/motion-cache.h):
// the offsets that pick its subsampled grids out of the GENERATED tails.
//
// Video is [C][T][H'][W'] over every `subsample`-th latent ROW and
// COLUMN -- not patch; that is what the reference's [..., ::f, ::f]
// keeps -- mapped through the packing: row t * gh * gw + cell, element
// (c * patch_h + y) * patch_w + x. Audio is every `subsample`-th latent
// step of each stereo channel, all lanes; its rows are channel-major in
// time. False, with the reason, when the geometry has no such view.
struct MotionCacheGathers {
  std::vector<std::uint32_t> video;
  std::vector<std::uint32_t> audio;
  int channels = 0;
  int frames   = 0;
  int plane    = 0;   // H' * W'
};
bool motion_cache_gathers(const MetalMiniMaxH3Transformer::Config& cfg,
                          int grid_h, int grid_w, int video_rows,
                          int audio_rows, int subsample,
                          MotionCacheGathers* out, std::string* why);

// Run the loop. False on failure, with a reason in `err`. On success the
// request's `video` / `audio` hold the denoised latents, still in packed
// row order -- unpatchifying them into a latent grid is the caller's job
// (it owns the geometry the layout was built from).
bool denoise(const DenoiseRequest& req, std::string* err = nullptr);

}  // namespace genai
}  // namespace vpipe

#endif
