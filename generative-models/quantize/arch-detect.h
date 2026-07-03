#ifndef VPIPE_GENAI_QUANTIZE_ARCH_DETECT_H
#define VPIPE_GENAI_QUANTIZE_ARCH_DETECT_H

#include <string>

#include "generative-models/qwen3/metal-qwen-model.h"

namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

// Architecture introspection for the model-quantize stage. Combines the
// source model's config.json (model_type + the text-config dims) with a probe
// of its safetensors tensor names to derive, generically across the families
// vpipe supports (Llama-3, Qwen2/Qwen3 dense, Qwen3.5 hybrid + MoE, Gemma-4,
// MOSS-TTS-Local), everything the quantizer's per-layer passes need:
//
//   * layer_prefix  -- the safetensors prefix up to and including "layers."
//     for the LM stack (handles the multimodal "language_model.model.layers."
//     wrap, the plain "model.layers." text model, and MOSS "transformer.
//     layers." -- probed, not guessed by family).
//   * n_layers      -- total transformer layers (max layer index + 1).
//   * awq_ok        -- whether the SmoothQuant/AWQ fold is STRUCTURALLY valid:
//     every layer is either a standard attention block (q/k/v/o) OR a Qwen3.5
//     gated-DeltaNet block (linear_attn.in_proj_* / out_proj) -- both start
//     with input_layernorm (the fold target for the in-projections) -- with
//     the standard input/post-attention layernorm pair and a plain dense MLP.
//     False for Gemma (its gate/up input is pre_feedforward_layernorm, so the
//     fold target would be wrong) and for MoE (no dense gate_proj to fold).
//   * calib_ok      -- whether on-device auto-calibration can run the backbone
//     through MetalQwenModel (the Qwen3 family: dense full-attention AND the
//     Qwen3.5 full-attn + gated-DeltaNet hybrid; not MoE).
//   * backbone      -- the MetalQwenModel::Config for that calibration
//     (valid only when calib_ok), built via MetalQwenModel::config_from for
//     the Qwen3.5 family so the GDN dims / full-attn interval / mROPE are
//     correct.
//
// Plain group-affine quantization and per-LAYER mixed precision work for any
// standard-named model (they gracefully skip non-matching tensors) and only
// need layer_prefix + n_layers; AWQ additionally needs awq_ok; on-device
// auto-calibration needs calib_ok + backbone.
struct QuantArchInfo {
  std::string arch;          // vpipe family tag (informational + registry)
  std::string layer_prefix;  // e.g. "model.layers." / "transformer.layers."
  int  n_layers      = 0;    // total transformer layers
  int  n_attn_layers = 0;    // layers carrying self_attn.q_proj
  bool detected      = false; // layer layout was probed from the weights
  bool awq_ok        = false; // AWQ/SmoothQuant fold structurally valid
  bool calib_ok      = false; // on-device MetalQwenModel auto-calibration
  MetalQwenModel::Config backbone;  // valid when calib_ok
};

// Inspect a source model directory (config.json + safetensors). Never throws;
// fields that can't be determined are left at their defaults (detected=false
// when the weights can't be opened / carry no recognizable layer stack). The
// session is borrowed for the config parse (ModelLoader) -- may be null.
QuantArchInfo detect_quant_arch(const SessionContextIntf* session,
                                const std::string& src_dir);

}  // namespace vpipe::genai

#endif
