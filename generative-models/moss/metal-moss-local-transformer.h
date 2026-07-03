#ifndef VPIPE_GENAI_MOSS_METAL_MOSS_LOCAL_TRANSFORMER_H
#define VPIPE_GENAI_MOSS_METAL_MOSS_LOCAL_TRANSFORMER_H

// MOSS-TTS-Local-Transformer-v1.5 depth/"local" transformer: a tiny GPT2-style
// decoder (config gpt2_config, n_layer=1) that, seeded by the backbone's frame
// hidden, autoregressively emits the n_vq RVQ codebook tokens for one frame.
// Runs in f16 (its bf16 weights are converted at load -- f16 >= bf16 mantissa,
// and it is a single layer) reusing the existing f16 metal kernels:
//   ln_1/ln_2/ln_f = layer_norm_bias_f16 (LayerNorm w/ bias, eps 1e-6),
//   c_attn/c_proj/fc_in/fc_out = dense_gemv_t_f16 + bias_add_rows_f16,
//   RoPE = rope_interleaved_f16 (GPT-J interleaved, base rope_base),
//   silu = mul_sigmoid_f16(z,z), residual = residual_add_f16,
//   attention = local_attn_step_f16 (single-query causal over the per-frame
//   KV cache, head_dim, scale 1/sqrt(head_dim)).
// step() advances one position; reset_frame() clears the per-frame KV cache.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <memory>
#include <string>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class MetalLlamaWeights;

class MetalMossLocalTransformer {
public:
  struct Config {
    int   hidden     = 2560;
    int   n_head     = 32;
    int   head_dim   = 80;     // hidden / n_head
    int   inner      = 9728;
    int   n_vq       = 12;     // -> per-frame cache capacity n_vq (+1 slack)
    float ln_eps     = 1e-6f;
    float rope_base  = 1.0e6f;
  };

  // Load from the model dir's local_transformer.* tensors (bf16 -> f16).
  static std::unique_ptr<MetalMossLocalTransformer> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc,
      const Config& cfg);

  static std::unique_ptr<MetalMossLocalTransformer> load(
      const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Clear the per-frame KV cache (call before each frame's codebook loop).
  void reset_frame() { _pos = 0; }

  // Advance one position: x_in is a [hidden] f16 row (the backbone seed at
  // pos 0, else audio_embeddings[k-1](code) at pos k). Returns a pointer to
  // the internal [hidden] f16 ln_f hidden (valid until the next step()).
  // Uses the current internal position, then increments it. Standalone form:
  // owns its command buffer + a synchronous wait (teacher-forced / debug).
  const metal_compute::SharedBuffer* step(
      const metal_compute::SharedBuffer& x_in);

  // Encode one position into a caller-owned encoder WITHOUT committing, so a
  // whole frame's codebook loop (heads + on-GPU sample + embed-gather + the
  // next step) fuses into a single command buffer (see MetalMossLocalModel).
  // `pos` is explicit (0..n_vq); the internal _pos is not touched. Output
  // lands in hidden() and is valid until the next encode_step on this encoder.
  void encode_step(metal_compute::ComputeEncoder& e,
                   const metal_compute::SharedBuffer& x_in, int pos);
  const metal_compute::SharedBuffer& hidden() const { return _hidden; }

  const Config& config() const { return _cfg; }
  int pos() const { return _pos; }

private:
  bool init_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
             const Config& cfg);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int    _pos = 0;

  // f16 weights + biases.
  metal_compute::SharedBuffer _ca_w, _ca_b, _cp_w, _cp_b;
  metal_compute::SharedBuffer _fi_w, _fi_b, _fo_w, _fo_b;
  metal_compute::SharedBuffer _ln1_w, _ln1_b, _ln2_w, _ln2_b, _lnf_w, _lnf_b;
  metal_compute::SharedBuffer _inv_freq;             // f32 [head_dim/2]
  metal_compute::SharedBuffer _kc, _vc;              // [n_head*pmax*head_dim]

  // Scratch (all f16).
  metal_compute::SharedBuffer _ln1, _qkv, _attn, _attn2, _resid1;
  metal_compute::SharedBuffer _ln2, _fc, _mlp, _resid2, _hidden;

  // Kernel handles.
  metal_compute::ComputeLibrary  _lib_vis, _lib_dense, _lib_elt, _lib_rope;
  metal_compute::ComputeFunction _fn_ln, _fn_gemv, _fn_bias, _fn_residual;
  metal_compute::ComputeFunction _fn_silu, _fn_rope, _fn_attn;
};

}  // namespace vpipe::genai

#endif
