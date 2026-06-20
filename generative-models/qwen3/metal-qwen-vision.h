#ifndef VPIPE_GENERATIVE_MODELS_METAL_QWEN_VISION_H
#define VPIPE_GENERATIVE_MODELS_METAL_QWEN_VISION_H

// MetalQwenVisionEncoder -- procedural Qwen3-VL vision tower (ViT) on the
// metal-compute framework, no MLX in the forward. The dense counterpart
// of the MLX Qwen3VLVisionEncoder (generative-models/shared/vision-encoder.cc):
// CPU preprocess (smart-resize + bilinear + normalize) -> merger-order
// patchify -> patch-embed GEMM -> bilinear pos-embed -> N pre-LN ViT
// blocks (LayerNorm, dense qkv, 2D-RoPE, full SDPA, dense proj, GELU-tanh
// MLP) -> patch merger (LayerNorm, dense fc1, GELU-erf, dense fc2) ->
// [n_tokens, out_hidden] image embeddings, ready to splice into the LM at
// the image-token placeholders. Vision weights are dense BF16 (converted
// to F16 at load); every kernel it calls is oracle-verified in
// tests/unit-tests/metal-compute-llm-ops.cc.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

struct ModelConfig;

class MetalQwenVisionEncoder {
public:
  struct Config {
    int depth          = 24;
    int hidden         = 1024;
    int n_heads        = 16;
    int patch_size     = 16;
    int spatial_merge  = 2;
    int temporal_patch = 2;
    int out_hidden     = 2560;
    int num_pos_embed  = 2304;   // G*G learned position table (G=48)
    int intermediate   = 4096;   // ViT MLP inner; also merger fc1 out
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3]  = {0.5f, 0.5f, 0.5f};
    float ln_eps       = 1e-6f;
    // When non-empty, the tower loads from this mmproj-*.gguf (llama.cpp
    // CLIP layout) instead of the safetensors `model_dir`. Same BF16/F32
    // weights, just CLIP tensor names + a split patch-embed conv.
    std::string gguf_mmproj;
    int head_dim() const { return hidden / n_heads; }   // 64
  };

  struct Result {
    // Native f16 [n_tokens * out_hidden] row-major, on the GPU (UMA).
    // Splice straight into the LM via prefill_multimodal_metal -- no
    // host f32 round-trip. Empty buffer / n_tokens==0 on failure.
    metal_compute::SharedBuffer embeddings;
    int n_tokens   = 0;
    int out_hidden = 0;
    int grid_h     = 0;
    int grid_w     = 0;
  };

  static std::unique_ptr<MetalQwenVisionEncoder> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Derive the ViT Config from a parsed ModelConfig's vision_config so
  // the tower self-sizes to any family member (the 9B ViT is depth 27 /
  // hidden 1152 / out_hidden 4096 vs the 4B's 24 / 1024 / 2560). Single
  // source of truth, shared by the production loader and the tests.
  static Config config_from(const ModelConfig& c);

  // Encode one RGB image: rgb is [3, H, W] U8 channel-first contiguous.
  Result encode(const std::uint8_t* rgb, int H, int W);

  const Config& config() const { return _cfg; }

  // Attach a session so encode() is recorded on the profiler's LLM lane
  // (vision-tower category) when no CoreML tower is configured and the
  // ViT runs on the GPU. Optional; without it the encode still runs but
  // is invisible to the profiler. Mirrors MetalGemma4VisionEncoder.
  void set_session(const SessionContextIntf* s) { _session = s; }

private:
  MetalQwenVisionEncoder() = default;

  struct Block {
    metal_compute::SharedBuffer n1w, n1b, qkvw, qkvb, ow, ob;
    metal_compute::SharedBuffer n2w, n2b, fc1w, fc1b, fc2w, fc2b;
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;   // profiler, optional
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_attn, _lib_dense_mma, _lib_sdpa_mma, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm, _fn_ln, _fn_gelu_tanh, _fn_gelu_erf,
      _fn_vrope, _fn_sdpa_full, _fn_head_slice, _fn_transpose, _fn_residual;
  // Steel MMA dense GEMM (fast path; the scalar _fn_gemm is the fallback
  // when this isn't available). Bias is folded into the kernel.
  metal_compute::ComputeFunction _fn_gemm_t;
  // M5 matrix-core (matmul2d) dense GEMM -- the same NAX path the LM prefill
  // uses (dense_gemm_mma, 128x128 for K<6144 else 128x256). The kernel omits
  // bias, so _fn_bias_add folds the vision linear bias back over the rows.
  // _use_mma2 gates it (tensor GPU only); else the steel/scalar path above.
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep, _fn_bias_add;
  bool _use_mma2 = false;
  // M5 matrix-core (matmul2d/NAX) non-causal flash attention for the ViT
  // (head_dim 64). Replaces the steel simdgroup_matrix attn -- the O(n^2)
  // term that dominates the tower at large grids. _use_mma2_attn gates it
  // (tensor GPU + hd==64); else the steel/scalar path stays.
  metal_compute::ComputeFunction _fn_sdpa_mma_d64;
  bool _use_mma2_attn = false;
  // MLX's matrix-core (NAX) steel flash attention -- the kernel MLX itself
  // dispatches on M5 (attn_steel_nax, bq=64/bk=32, register-resident softmax).
  // The preferred steel path when available; the ALU _lib_attn steel is the
  // M4 fallback. _use_attn_nax gates it (supports_matrix_cores + hd==64).
  bool _use_attn_nax = false;
  // Steel flash-attention (MMA) param block; the per-encode pipeline (with
  // align/causal function constants) is created in encode() from _lib_attn.
  metal_compute::SharedBuffer _attn_params;

  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _pe_w, _pe_b;   // patch embed [hidden,1536],[hidden]
  metal_compute::SharedBuffer _pos_w;          // pos table [num_pos_embed, hidden] f16
  metal_compute::SharedBuffer _mnw, _mnb, _mfc1w, _mfc1b, _mfc2w, _mfc2b;
  std::vector<float> _rope_inv_freq;           // [head_dim/4]
  int _feat_dim = 0;                           // temporal*patch*patch*3
};

}  // namespace vpipe::genai

#endif
