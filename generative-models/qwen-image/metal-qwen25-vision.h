#ifndef GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN25_VISION_H
#define GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN25_VISION_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd

// Qwen2.5-VL vision tower (the Qwen-Image-Edit-2511 text_encoder's `visual`).
// A 32-block ViT: RMSNorm, fused-qkv attention (per-head, head_dim 80, 2D
// half-split RoPE), SwiGLU MLP, then a 2x2 spatial-merge patch merger
// (RMSNorm + Linear + GELU + Linear) to the LLM hidden (3584). Runs bf16.
//
// Takes ALREADY-PATCHIFIED pixel_values [seq, 1176] (3*temporal*patch*patch)
// + per-patch (h,w) RoPE positions. Window attention (fullatt_block_indexes
// [7,15,23,31]) collapses to full attention when the whole grid fits one
// window (the M5 verification case); the multi-window reorder is a follow-up.
class MetalQwen25Vision {
 public:
  struct Config {
    int depth        = 32;
    int hidden       = 1280;
    int n_heads      = 16;
    int head_dim     = 80;
    int patch_in     = 1176;   // 3 * temporal_patch(2) * patch(14)^2
    int ffn          = 3420;
    int out_hidden   = 3584;
    int merge        = 2;      // spatial_merge_size
    float norm_eps   = 1e-6f;
    int rope_theta   = 10000;
    // Window attention: window_size(112) / patch(14) / merge(2) = 4 merged
    // tokens per window side. 28 of 32 blocks use window attention; blocks
    // {7,15,23,31} (fullatt_block_indexes) use full attention. When the whole
    // grid fits one window this degenerates to full attention (the M5 case).
    int window_merge = 4;
  };

  static std::unique_ptr<MetalQwen25Vision>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg);
  ~MetalQwen25Vision();

  // Encode patchified pixels -> vision tokens [seq/merge^2, out_hidden] f16(bf16).
  // `pos` is [seq*2] (h,w per patch). Currently assumes a single attention
  // window (grid fits window_size); asserts otherwise via `full_attention`.
  metal_compute::SharedBuffer
  encode(const metal_compute::SharedBuffer& pixels, int seq,
         const std::vector<int>& pos);

  // Preprocess a raw RGB image [3,H,W] U8 (channel-first, load-image format) the
  // QwenImageEditPlus condition way -- resize to the `cond_area` (=384*384)
  // condition dims, smart_resize to the 28-multiple ViT grid, CLIP-normalize,
  // patchify (merge-block order) -- then run the tower. Returns [n_tokens,
  // out_hidden] bf16 vision tokens (natural merged order) and sets grid_h/grid_w
  // (the PATCH grid; n_tokens = grid_h*grid_w / merge^2). Bilinear resize (HF
  // uses bicubic; close enough for conditioning). Empty on failure.
  metal_compute::SharedBuffer
  encode_rgb(const std::uint8_t* rgb, int H, int W, int cond_area,
             int& grid_h, int& grid_w);

  const Config& config() const { return _cfg; }

 private:
  MetalQwen25Vision() = default;
  metal_compute::SharedBuffer to_elt_(const MetalLlamaWeights& wts,
                                      const std::string& name);
  bool load_linear_(const MetalLlamaWeights& wts, const std::string& pre,
                    metal_compute::SharedBuffer& w,
                    metal_compute::SharedBuffer& b);

  struct Block {
    metal_compute::SharedBuffer n1, n2;                  // RMSNorm [hidden]
    metal_compute::SharedBuffer qkv_w, qkv_b, proj_w, proj_b;
    metal_compute::SharedBuffer gw, gb, uw, ub, dw, db;  // SwiGLU
  };

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  metal_compute::SharedBuffer _patch_w;                  // [hidden, patch_in]
  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _merge_ln;                 // RMSNorm [hidden]
  metal_compute::SharedBuffer _merge0_w, _merge0_b;      // [4h,4h]
  metal_compute::SharedBuffer _merge2_w, _merge2_b;      // [out,4h]

  // Host-built half-split RoPE cos/sin tables [seq, head_dim/2] (f32).
  void build_rope_(int seq, const std::vector<int>& pos,
                   metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out);

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa,
      _lib_rope;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_rms, _fn_silu,
      _fn_mul, _fn_gelu, _fn_residual, _fn_transpose, _fn_sdpa, _fn_rope,
      _fn_swiglu, _fn_varwindow;
};

}  // namespace genai
}  // namespace vpipe

#endif
