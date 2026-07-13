#ifndef GENERATIVE_MODELS_LORA_FUSION_H
#define GENERATIVE_MODELS_LORA_FUSION_H

#include <cstdint>
#include <functional>
#include <string>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe {
namespace genai {

// Fuse a base safetensors model (`base_dir`) with a LoRA adapter
// (`lora_path`, a single .safetensors) into `out_dir`. Each base weight
// W[N,K] that has a matching adapter becomes W + scale * dW; every other tensor
// passes through byte-for-byte, and config.json is copied. The output reloads
// through the usual bf16 path (e.g. the Krea-2 DiT via a text-to-image
// `dit_dir`).
//
// Two adapter formats are recognised per module:
//   - low-rank LoRA -- `<module>.lora_{A,B}.weight` (lora_A[rank,K] +
//     lora_B[N,rank]); dW = lora_B @ lora_A.
//   - LoKr (Kronecker) -- `<module>.lokr_w1`/`.lokr_w2` (each optionally a
//     low-rank product `lokr_w{1,2}_a @ lokr_w{1,2}_b`); dW = kron(w1, w2).
// A sibling `<module>.alpha` rescales by alpha/rank (kohya / ai-toolkit); a
// full-matrix LoKr uses no alpha rescaling (LyCORIS scale=1), diffusers LoRAs
// omit alpha entirely.
//
// The base weight for a `<module>` is matched by trying `<module>.weight`,
// then `<module>` with its leading component segment stripped (diffusers'
// "transformer." prefix), then an ai-toolkit / ComfyUI name remap
// (`diffusion_model.*` block + attn/mlp submodule names -> diffusers). The
// fuse runs in f32 and writes each weight back in its source dtype
// (BF16/F16/F32), one tensor at a time (peak RAM ~one weight). `stop` is polled
// at tensor boundaries. Returns false + *err on failure; logs counts via mc's
// session.
bool fuse_lora(metal_compute::MetalCompute* mc, const std::string& base_dir,
               const std::string& lora_path, const std::string& out_dir,
               float scale, std::string* err,
               const std::function<bool()>& stop = [] { return false; });

}  // namespace genai
}  // namespace vpipe

#endif
