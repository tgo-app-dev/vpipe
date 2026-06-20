#ifndef VPIPE_GENERATIVE_MODELS_METAL_GEMMA4_AUDIO_H
#define VPIPE_GENERATIVE_MODELS_METAL_GEMMA4_AUDIO_H

// MetalGemma4AudioEncoder -- Gemma-4 USM Conformer audio tower on the
// metal-compute framework, no MLX in the forward. The dense counterpart
// of the MLX Gemma4AudioEncoder (generative-models/gemma4/gemma4-audio-
// encoder.cc): host USM log-mel front-end -> SSCP subsample -> 12
// Conformer blocks (macaron FFN, chunked-local attention with relative-
// position embeddings + logit softcap, light depthwise Conv1d + GLU) ->
// output_proj -> embed_audio -> [n_tokens, lm_hidden] host f32
// embeddings, ready to splice at the audio-token placeholders.
//
// Strategy: the matmul-heavy ops (the linears + the quantized
// embed_audio) run on metal-compute (dense_gemm_t_f16 /
// affine_qmm_steel_w4g64); the intricate-but-tiny glue (RMSNorm, SiLU,
// ClippableLinear clamps, GLU, the chunked-local rel-pos attention, the
// depthwise Conv1d, residuals) runs on host f32. Encoder weights are
// dense BF16 (-> F16); embed_audio is 4-bit affine-quantized.

#include "generative-models/gemma4/gemma4-audio-feature-extractor.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

struct ModelConfig;

class MetalGemma4AudioEncoder {
public:
  struct Config {
    int   d_model      = 1024;
    int   n_heads      = 8;
    int   n_layers     = 12;
    int   ffn          = 4096;
    int   output_dim   = 1536;
    int   out_hidden   = 2560;
    int   n_mel        = 128;
    int   conv_kernel  = 5;
    int   chunk        = 12;
    int   ctx_left     = 13;   // attention_context_left (max_past = -1)
    int   ctx_right    = 0;
    int   group_size   = 64;
    int   bits         = 4;
    float softcap      = 50.0f;
    float grad_clip    = 1.0e10f;
    float residual_w   = 0.5f;
    float rms_eps      = 1.0e-6f;
    std::vector<int> sscp_channels;   // {128, 32}
    int head_dim() const { return n_heads ? d_model / n_heads : 0; }
  };

  struct Result {
    std::vector<float> embeddings;   // [n_tokens * out_hidden] row-major
    int n_tokens = 0;
  };

  static std::unique_ptr<MetalGemma4AudioEncoder> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  static Config config_from(const ModelConfig& c);

  // Encode one f32 mono PCM clip at 16 kHz. Empty result on failure.
  Result encode(const float* pcm, std::size_t n_samples, int sample_rate);

  const Config& config() const { return _cfg; }

  // Attach a session so encode() shows up on the profiler (audio-encoder
  // lane), mirroring the MLX audio encoder. Optional; null => no-op.
  void set_session(const SessionContextIntf* s) { _session = s; }

private:
  MetalGemma4AudioEncoder() = default;

  // ClippableLinear bounds (host floats).
  struct Clip { float imin = 0, imax = 0, omin = 0, omax = 0; bool on = false; };

  // A dense f16 weight [N, K] on the GPU + (optional) host clip bounds.
  struct Lin {
    metal_compute::SharedBuffer w;   // [N, K] f16
    int n = 0, k = 0;
    Clip clip;
  };

  struct Layer {
    Lin ff1_w1, ff1_w2, ff2_w1, ff2_w2;
    std::vector<float> ff1_pre, ff1_post, ff2_pre, ff2_post;     // RMSNorm
    Lin q, k, v, post, rel_k;
    std::vector<float> per_dim_scale;                            // [head_dim]
    std::vector<float> norm_pre_attn, norm_post_attn, norm_out;
    Lin lc_start, lc_end;
    std::vector<float> lc_pre, lc_norm;
    std::vector<float> lc_dw;     // depthwise conv1d [d_model, kernel]
    // f16 GPU copies of the per-layer norm/scale weights (built once by
    // prepare_gpu_) for the fully GPU-resident block forward.
    metal_compute::SharedBuffer ff1_pre_b, ff1_post_b, ff2_pre_b, ff2_post_b;
    metal_compute::SharedBuffer npa_b, npost_b, nout_b, lc_pre_b, lc_norm_b;
    metal_compute::SharedBuffer pds_b;     // softplus(per_dim_scale) [Hd] f16
    metal_compute::SharedBuffer lc_dw_b;   // [d_model, kernel] f16
  };

  // ---- GPU helpers (round-trip a host f32 matmul through metal) ----
  std::vector<float> gemm_(const std::vector<float>& x, int M, int K,
                           const Lin& lin) const;
  std::vector<float> qmm_(const std::vector<float>& x, int M) const;
  // Build the f16 GPU norm/scale/timing buffers from the host weights
  // (idempotent; runs on the first encode).
  void prepare_gpu_();

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;   // profiler, optional
  metal_compute::ComputeLibrary _lib_gemm, _lib_qmm, _lib_elt, _lib_rms,
      _lib_sdpa, _lib_audio, _lib_dense_mma;
  metal_compute::ComputeFunction _fn_gemm_t, _fn_qmm, _fn_rms, _fn_clamp,
      _fn_silu, _fn_glu, _fn_dwconv, _fn_residual, _fn_scale, _fn_local_attn,
      _fn_conv2d, _fn_ln_relu;
  // M5 matrix-core (matmul2d/NAX) dense GEMM -- the same NAX path the LM
  // prefill and the Gemma/Qwen ViTs use (dense_gemm_mma, 128x128 for K<6144
  // else 128x256). The conformer linears are bias-free except output_proj, so
  // _fn_bias_add folds that one bias over the rows. _use_mma2 gates it (tensor
  // GPU only); else the steel _fn_gemm_t path stays (M4 byte-identical).
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep, _fn_bias_add;
  bool _use_mma2 = false;
  metal_compute::SharedBuffer _zero_bias;
  bool _gpu_prepared = false;

  std::vector<Layer> _layers;
  // SSCP conv stem: host f32 weights + their f16 GPU copies (built by
  // prepare_gpu_) + a zero conv bias; input projection (GPU Lin).
  std::vector<float> _conv0_w, _conv0_norm, _conv1_w, _conv1_norm;
  metal_compute::SharedBuffer _conv0_w_b, _conv1_w_b, _conv0_norm_b,
      _conv1_norm_b, _conv_zero_bias;
  Lin _in_proj;
  Lin _out_proj;
  std::vector<float> _out_proj_bias;
  metal_compute::SharedBuffer _out_proj_bias_b;   // [output_dim] f16
  metal_compute::SharedBuffer _ones_outdim;       // [output_dim] f16 ones
  metal_compute::SharedBuffer _timing_b;          // [span * d_model] f16
  // embed_audio: 4-bit affine projection.
  metal_compute::SharedBuffer _ev_w, _ev_s, _ev_b;
  std::vector<float> _timing;     // [span * d_model] sinusoidal table
  Gemma4AudioFeatureExtractor _fx{Gemma4AudioFeatureExtractor::Params{}};
};

}  // namespace vpipe::genai

#endif
