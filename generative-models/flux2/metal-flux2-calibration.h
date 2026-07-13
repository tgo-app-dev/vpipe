#ifndef GENERATIVE_MODELS_FLUX2_METAL_FLUX2_CALIBRATION_H
#define GENERATIVE_MODELS_FLUX2_METAL_FLUX2_CALIBRATION_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe {
namespace genai {

// On-device AWQ calibration for the FLUX.2 DiT. Like Krea-2 the activation
// distribution is TIMESTEP-DEPENDENT, so faithful calibration runs the real
// denoising trajectory: for each prompt, encode it (Qwen3 dense, {9,18,27}
// hidden-state tap -> concat 7680) and run a `steps`-step FlowMatchEuler sweep,
// calling forward_dit with the DiT's calib taps on so the per-input-channel
// |activation| abs-max accumulates across prompts x sigmas. Writes one
// <group>.f32 per activation group ([rows*dim]) -- the format the quantizer's
// flux2 dit_act (clip-only AWQ) consumes.
//
// Loads text_encoder/ + tokenizer/ under `model_root` FIRST, encodes + caches
// every prompt's context, frees the encoder, THEN loads the transformer -- so
// peak memory is ~max(encoder, DiT), not their sum. `stop` is polled at prompt
// boundaries. Returns false + *err on failure. Reuses
// default_dit_calibration_prompts() (Krea-2 calibration) when the caller passes
// none.
bool collect_flux2_calibration(
    metal_compute::MetalCompute* mc, const std::string& model_root,
    const std::vector<std::string>& prompts, int steps, int height, int width,
    std::uint64_t seed, const std::string& out_dir, std::string* err,
    const std::function<bool()>& stop = [] { return false; });

}  // namespace genai
}  // namespace vpipe

#endif
