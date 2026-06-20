#ifndef VPIPE_GENERATIVE_MODELS_METAL_GEMMA4_VISION_H
#define VPIPE_GENERATIVE_MODELS_METAL_GEMMA4_VISION_H

// MetalGemma4VisionEncoder -- procedural Gemma-4 vision tower (ViT) on the
// metal-compute framework, no MLX in the forward. The dense counterpart of
// the MLX Gemma4VisionEncoder (generative-models/shared/vision-encoder.cc): CPU
// preprocess (aspect-resize + bilinear + rescale) -> raster-order patchify
// (2*(x-0.5)) -> patch-embed GEMM -> learned 2-axis pos embed (host gather)
// -> 16 pre-RMSNorm ViT blocks (RMSNorm, q/k/v/o proj, q/k RMSNorm +
// weightless v RMSNorm, Gemma 2-D RoPE, full SDPA scale 1.0, geglu MLP)
// -> 3x3 position avg-pool (host) x sqrt(hidden) -> embed_vision
// (RMSNormNoScale + 4-bit 768->out_hidden projection) -> [n_tokens,
// out_hidden] image embeddings, native f16 (zero-copy LM splice). Vision
// weights are dense BF16 (converted to F16 at load); only embed_vision's
// projection is 4-bit affine-quantized.

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

class MetalGemma4VisionEncoder {
public:
  struct Config {
    int   depth          = 16;
    int   hidden         = 768;
    int   n_heads        = 12;
    int   patch_size     = 16;
    int   pool_kernel    = 3;
    int   out_length     = 280;     // soft-token budget (stills)
    int   video_out_length = 70;    // per-frame budget (video)
    int   pos_size       = 10240;   // learned 2-axis table rows
    int   intermediate   = 3072;
    int   out_hidden     = 2560;
    int   group_size     = 64;      // embed_vision quant
    int   bits           = 4;
    float rope_theta     = 100.0f;
    float rms_eps        = 1e-6f;
    int   head_dim() const { return n_heads ? hidden / n_heads : 0; }
  };

  struct Result {
    // Native f16 [n_tokens * out_hidden] row-major on the GPU (UMA).
    metal_compute::SharedBuffer embeddings;
    int n_tokens   = 0;
    int out_hidden = 0;
    int grid_h     = 0;
    int grid_w     = 0;
  };

  static std::unique_ptr<MetalGemma4VisionEncoder> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  static Config config_from(const ModelConfig& c);

  // Encode one RGB image: rgb is [3, H, W] U8 channel-first contiguous.
  // `max_soft_tokens` > 0 overrides the soft-token budget for this call
  // (video frames pass the smaller video budget); <=0 uses the still
  // budget (out_length).
  Result encode(const std::uint8_t* rgb, int H, int W,
                int max_soft_tokens = -1);

  // Per-frame video soft-token budget (Config::video_out_length), or -1
  // if unset. Mirrors the MLX Gemma4VisionEncoder accessor so the stage
  // can query the metal tower the same way.
  int video_soft_token_budget() const
  { return _cfg.video_out_length > 0 ? _cfg.video_out_length : -1; }

  const Config& config() const { return _cfg; }

  // Attach a session so encode() shows up on the profiler (vision-tower
  // lane), mirroring the MLX vision encoder. Optional; null => no-op.
  void set_session(const SessionContextIntf* s) { _session = s; }

private:
  MetalGemma4VisionEncoder() = default;

  struct Block {
    metal_compute::SharedBuffer in_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln;
    metal_compute::SharedBuffer qw, kw, vw, ow, qn, kn;
    metal_compute::SharedBuffer gw, uw, dw;
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;   // profiler, optional
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_rms, _lib_qmm, _lib_attn, _lib_dense_mma, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm_t, _fn_rms, _fn_grope, _fn_sdpa_full,
      _fn_sdpa_full_mma, _fn_transpose, _fn_residual, _fn_geglu, _fn_qmm;
  // M5 matrix-core (matmul2d/NAX) dense GEMM -- the same NAX path the LM
  // prefill and the Qwen ViT use (dense_gemm_mma, 128x128 for K<6144 else
  // 128x256). The Gemma vision linears are bias-free, so unlike the Qwen
  // tower no bias-add fold is needed. _use_mma2 gates it (tensor GPU only);
  // else the steel _fn_gemm_t path stays (M4 byte-identical, unchanged).
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep;
  bool _use_mma2 = false;
  // MLX's matrix-core (NAX) steel flash attention -- the kernel MLX itself
  // dispatches on M5 (attn_steel_nax, bq=64/bk=32, register-resident softmax).
  // ~1.2-1.7x the ALU steel attn at large grids (the ViT's O(n^2) term). Built
  // correct only with deployment target 26.2 (see metal-compute CMakeLists /
  // VPIPE_NAX_MIN_OS). _use_attn_nax gates it (supports_matrix_cores + hd==64);
  // else the ALU _lib_attn steel is the M4 fallback.
  bool _use_attn_nax = false;

  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _patch_w;     // [hidden, 3*P*P]
  metal_compute::SharedBuffer _pos_table;   // [2, pos_size, hidden] f16
  metal_compute::SharedBuffer _ev_w, _ev_s, _ev_b;   // embed_vision proj
  metal_compute::SharedBuffer _ones_hd, _ones_hid, _zero_bias;
  // SteelAttnParams scratch for the vendored MLX steel flash kernel
  // (attn_steel_h_bd64); the per-encode function (align/causal constants)
  // is created in encode() from _lib_attn.
  metal_compute::SharedBuffer _attn_params;
};

}  // namespace vpipe::genai

#endif
