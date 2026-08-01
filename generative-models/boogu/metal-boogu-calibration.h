#ifndef GENERATIVE_MODELS_BOOGU_METAL_BOOGU_CALIBRATION_H
#define GENERATIVE_MODELS_BOOGU_METAL_BOOGU_CALIBRATION_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe {
namespace genai {

// On-device AWQ calibration for the Boogu-Image DiT. Like Krea-2 / FLUX.2 the
// activation distribution is TIMESTEP-DEPENDENT, so faithful calibration runs
// the real denoising trajectory: for each prompt, encode it through the Qwen3-VL
// mllm (last hidden state + the final RMSNorm, the whole templated sequence --
// Boogu drops no prefix) and run a `steps`-step sweep over the DMD student's
// ASCENDING sigma schedule, calling forward_dit with the DiT's calib taps on so
// the per-input-channel |activation| abs-max accumulates across prompts x
// sigmas. Writes one <group>.f32 per activation group ([rows*dim]) -- the format
// the quantizer's boogu dit_act (clip-only AWQ) consumes.
//
// Loads mllm/ + processor/ under `model_root` FIRST, encodes + caches every
// prompt's conditioning, frees the encoder, THEN loads the transformer -- so
// peak memory is ~max(encoder, DiT), not their sum (the pair is ~36 GB bf16).
// `stop` is polled at prompt boundaries. Returns false + *err on failure.
// Reuses default_dit_calibration_prompts() when the caller passes none.
bool collect_boogu_calibration(
    metal_compute::MetalCompute* mc, const std::string& model_root,
    const std::vector<std::string>& prompts, int steps, int height, int width,
    std::uint64_t seed, const std::string& out_dir, std::string* err,
    const std::function<bool()>& stop = [] { return false; });

}  // namespace genai
}  // namespace vpipe

#endif
