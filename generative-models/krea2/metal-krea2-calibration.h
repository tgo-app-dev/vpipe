#ifndef GENERATIVE_MODELS_KREA2_METAL_KREA2_CALIBRATION_H
#define GENERATIVE_MODELS_KREA2_METAL_KREA2_CALIBRATION_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe {
namespace genai {

// On-device AWQ calibration for the Krea-2 DiT. Unlike an LM, the DiT's
// activation distribution is TIMESTEP-DEPENDENT, so faithful calibration must
// run the real denoising trajectory: for each prompt, encode it (Qwen3-VL
// 12-layer tap -> text fusion) and run the `steps`-step turbo FlowMatchEuler
// sampler, calling forward_dit with the DiT's calib taps on so the per-input-
// channel |activation| abs-max accumulates across prompts x sigmas. Writes
// calib_{qkv,o,gateup,down}.f32 -- [n_layers, hidden] (qkv/o/gateup) and
// [n_layers, ffn] (down) -- the format the AWQ fold consumes.
//
// Loads text_encoder/ + transformer/ + tokenizer/ under `model_root` (bf16;
// the DiT unquantized for fidelity). `stop` is polled at prompt boundaries so a
// pipeline stop aborts cooperatively. Returns false + *err on failure.
bool collect_dit_calibration(
    metal_compute::MetalCompute* mc, const std::string& model_root,
    const std::vector<std::string>& prompts, int steps, int height, int width,
    std::uint64_t seed, const std::string& out_dir, std::string* err,
    const std::function<bool()>& stop = [] { return false; });

// A small diverse default image-prompt corpus (objects / scenes / animals /
// people / styles / abstract) for calibration when the caller supplies none.
std::vector<std::string> default_dit_calibration_prompts();

}  // namespace genai
}  // namespace vpipe

#endif
