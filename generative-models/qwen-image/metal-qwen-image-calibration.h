#ifndef GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN_IMAGE_CALIBRATION_H
#define GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN_IMAGE_CALIBRATION_H

#include "apple-silicon/metal-compute/metal-compute.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

// On-device AWQ activation calibration for the Qwen-Image-Edit-2511 DiT.
//
// Runs the REAL conditioning + denoise: the Qwen2.5-VL text encoder over the
// QwenImageEditPlus template (drop-64, sequential 1-D rope, host final-norm),
// then the dual-stream DiT over a dynamic-shift sigma schedule, with the DiT's
// per-input-channel col-absmax taps enabled. Accumulates over `prompts` x
// `steps` sigmas and writes one raw-f32 file per calib group -- img_qkv, txt_qkv,
// img_o, txt_o, img_fc1, txt_fc1 ([n_layers*hidden]) and img_fc2, txt_fc2
// ([n_layers*ffn]) -- into `out_dir`, which model-quantize target=dit awq=true
// reads to activation-aware-clip the quantized leaf weights.
//
// `model_root` is the pipeline root (text_encoder/, transformer/, tokenizer/ or
// processor/tokenizer.json). Returns false + sets *err on failure; `stop`
// aborts cooperatively. Frees the encoder before the DiT denoise loop.
bool collect_qwen_image_calibration(
    metal_compute::MetalCompute* mc, const std::string& model_root,
    const std::vector<std::string>& prompts, int steps, int height, int width,
    std::uint64_t seed, const std::string& out_dir, std::string* err,
    const std::function<bool()>& stop = [] { return false; });

}  // namespace genai
}  // namespace vpipe

#endif
