#ifndef VPIPE_GENERATIVE_MODELS_METAL_AUDIO_ENCODER_H
#define VPIPE_GENERATIVE_MODELS_METAL_AUDIO_ENCODER_H

// MetalAudioEncoder -- procedural Qwen3-ASR audio tower on the
// metal-compute framework, no MLX in the forward. The dense counterpart
// of the MLX Qwen3AsrAudioEncoder (generative-models/shared/audio-encoder.cc):
// CPU WhisperFeatureExtractor (log-mel) -> per-chunk Conv2d x3 stem
// (+GELU-erf) -> conv_out GEMM -> per-chunk sinusoidal position embed ->
// block-windowed pre-LN encoder blocks (LayerNorm, fused q|k|v GEMM,
// windowed bidirectional SDPA, GELU-erf MLP) -> ln_post + proj1 + GELU +
// proj2 -> [n_tokens, output_dim] host f32 embeddings, ready to splice
// into the LM at the audio-token placeholders.
//
// Reuses the Qwen3-VL vision tower's dense kernels (dense_gemm_bias_f16,
// layer_norm_bias_f16, gelu_erf_f16, sdpa) + the new conv stem kernel
// (audio_encoder.metal) + the windowed SDPA (sdpa_window_f16). Encoder
// weights are dense BF16 (converted to F16 at load).

#include "generative-models/shared/whisper-feature-extractor.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
// The make_metal_audio_encoder() adapter wraps the metal encoder in an
// AudioEncoder (whose EncodedAudio carries an mlx::core::array), so its
// declaration + definition are MLX-only. The pure MetalAudioEncoder
// class below is MLX-free and used directly by the no-MLX LM path.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalAudioEncoder {
public:
  struct Config {
    int   n_mel          = 128;
    int   d_model        = 1024;
    int   n_layers       = 24;
    int   n_heads        = 16;
    int   ffn            = 4096;   // encoder_ffn_dim
    int   output_dim     = 2048;
    int   conv_hidden    = 480;    // downsample_hidden_size
    int   n_window       = 50;
    int   n_window_infer = 800;
    int   sample_rate    = 16000;
    float ln_eps         = 1e-5f;
    int head_dim() const { return d_model / n_heads; }   // 64
    int chunk_mel() const { return n_window * 2; }        // 100 mel frames
    // Audio tokens per window after the conv stem (block-attention size).
    int window_tokens() const {
      return 13 * (n_window_infer / (n_window * 2));       // 104
    }
  };

  struct Result {
    std::vector<float> embeddings;   // [n_tokens * output_dim] row-major
    int n_tokens = 0;
  };

  static std::unique_ptr<MetalAudioEncoder> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Encode one f32 mono PCM clip at `sample_rate` Hz (must equal the
  // config sample rate). Empty result on failure / rate mismatch.
  Result encode(const float* pcm, std::size_t n_samples, int sample_rate);

  const Config& config() const { return _cfg; }

private:
  MetalAudioEncoder() = default;

  struct Block {
    metal_compute::SharedBuffer n1w, n1b, qkvw, qkvb, ow, ob;
    metal_compute::SharedBuffer n2w, n2b, fc1w, fc1b, fc2w, fc2b;
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_conv, _lib_dense_mma, _lib_attn, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm, _fn_ln, _fn_gelu_erf, _fn_conv,
      _fn_sdpa_window, _fn_transpose, _fn_head_slice, _fn_residual;
  // Steel MMA dense GEMM (fast path; scalar _fn_gemm is the fallback).
  metal_compute::ComputeFunction _fn_gemm_t;
  // im2col + last-two-axis swap: turn the conv stem into a dense MMA GEMM
  // (the scalar _fn_conv is the fallback).
  metal_compute::ComputeFunction _fn_im2col, _fn_swap_last2;
  // M5 matrix-core (matmul2d/NAX) dense GEMM -- the same NAX path the LM
  // prefill and the Gemma/Qwen ViTs use (dense_gemm_mma, 128x128 for K<6144
  // else 128x256). The ASR encoder linears (and the im2col conv GEMMs) are
  // biased except conv_out, so _fn_bias_add folds that bias over the rows.
  // _use_mma2 gates it (tensor GPU only); else the steel _fn_gemm_t path
  // stays (M4 byte-identical, performance unchanged).
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep, _fn_bias_add;
  bool _use_mma2 = false;
  // M5 matrix-core (NAX) BLOCK-WINDOWED steel flash attention. The ASR
  // block-windowed attention (sdpa_window) is block-diagonal: query qi attends
  // to its window-aligned block [(qi/W)*W, +W). That maps onto the steel/NAX
  // kernel by viewing the contiguous head-major [heads, seq, hd] qt/kt/vt as
  // [n_blocks, heads, W, hd] -- batch stride W*hd, head stride seq*hd -- and
  // running attn_steel_nax (hd=64, the vision kernel) batched over the blocks.
  // Two dispatches: the n_full full blocks (B=n_full, qL=kL=W) + the partial
  // last block (B=1, qL=kL=seq-n_full*W, bound at a byte offset). Subsumes the
  // seq<=window full-attention case (n_full=0). _use_attn_nax gates it
  // (supports_matrix_cores + hd==64); the scalar windowed kernel stays on M4 /
  // VPIPE_ASR_NO_NAX_ATTN=1. _attn_params holds the full-block SteelAttnParams,
  // _attn_params_last the partial-block one (filled once -- seq is constant).
  metal_compute::SharedBuffer _attn_params, _attn_params_last;
  bool _use_attn_nax = false;

  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _c1w, _c1b, _c2w, _c2b, _c3w, _c3b;
  metal_compute::SharedBuffer _convout_w;   // [d_model, conv_hidden*16]
  metal_compute::SharedBuffer _ln_post_w, _ln_post_b;
  metal_compute::SharedBuffer _proj1_w, _proj1_b, _proj2_w, _proj2_b;
  std::unique_ptr<WhisperFeatureExtractor> _fx;
};


}  // namespace vpipe::genai

#endif
