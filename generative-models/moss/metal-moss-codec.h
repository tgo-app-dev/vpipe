#ifndef VPIPE_GENERATIVE_MODELS_MOSS_METAL_MOSS_CODEC_H
#define VPIPE_GENERATIVE_MODELS_MOSS_METAL_MOSS_CODEC_H

// MetalMossCodec -- the no-MLX metal DECODE path of the MOSS Audio Tokenizer
// (the companion codec of MOSS-TTS). Turns the 32-codebook RVQ audio codes
// the TTS LM emits into a 24 kHz PCM waveform. The decoder is a CNN-free
// transformer: a Residual-LFQ codebook decode (per-codebook embed + 1x1
// weight-normed conv, summed) followed by FOUR ProjectedTransformer stages
// (32L@1280 + 12L@768 x3, causal + RoPE + LayerScale, exact-GELU FFN) with
// patch-reshape upsamplers between them (x2,x2,x2,x240 = x1920). Reuses the
// vision/audio-encoder metal kernels (layer_norm_bias / gelu_erf / dense GEMM
// / sdpa_causal_window / residual) + an interleaved-pair RoPE kernel.
//
// The checkpoint is F32; we run f16 (better audio precision than bf16, and
// the codec is not the pipeline bottleneck). Weight-norm convs and the
// per-channel LayerScale are folded into plain weight matrices at load.
// Metal-only; no MLX path. v1 is correctness-first (staged rel-L2 verified).

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalMossCodec {
public:
  // Load a MOSS-Audio-Tokenizer directory (config.json + sharded
  // safetensors, F32). Returns nullptr on failure.
  static std::unique_ptr<MetalMossCodec> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc);

  bool valid() const { return _ok; }
  int sample_rate() const { return _sample_rate; }
  int n_quantizers() const { return _n_vq; }

  // Decode RVQ codes `codes[T][n_vq]` (0..codebook_size-1) into a [T*1920]
  // f32 PCM waveform at sample_rate. When `stages` is non-null it is filled
  // with the per-stage intermediates in CHANNEL-major [C][T] order (index 0 =
  // post-RVQ [768,T], then one per decoder module) for rel-L2 verification.
  std::vector<float> decode(
      const std::vector<std::vector<std::int32_t>>& codes,
      std::vector<std::vector<float>>* stages = nullptr);

private:
  MetalMossCodec() = default;

  struct StageCfg {
    int in_dim, d_model, n_layers, n_heads, ff, out_dim, patch, context;
  };
  struct Layer {
    metal_compute::SharedBuffer n1w, n1b, n2w, n2b;  // LayerNorm w/b
    metal_compute::SharedBuffer qkvw;                // in_proj [3d, d]
    metal_compute::SharedBuffer ow;                  // out_proj [d,d] (ls1 fold)
    metal_compute::SharedBuffer fc1;                 // linear1 [ff, d]
    metal_compute::SharedBuffer fc2;                 // linear2 [d, ff] (ls2 fold)
  };
  struct Stage {
    StageCfg cfg;
    metal_compute::SharedBuffer in_proj;     // [d_model, in_dim]
    metal_compute::SharedBuffer out_proj;    // [out_dim, d_model] (empty=Identity)
    std::vector<Layer> layers;
  };

  metal_compute::SharedBuffer run_stage_(const Stage& st, int T,
                                         const metal_compute::SharedBuffer& in);
  metal_compute::SharedBuffer rvq_decode_(
      const std::vector<std::vector<std::int32_t>>& codes, int T);

  metal_compute::MetalCompute* _mc = nullptr;
  bool _ok = false;
  int _sample_rate = 24000;
  int _n_vq = 32;
  int _codebook_dim = 8;
  int _rvq_dim = 512;
  int _code_dim = 768;   // hidden fed to the first transformer

  // RVQ: per-codebook embed [1024,8] + folded out_proj [512,8]+bias[512];
  // shared output_proj [768,512]+bias[768].
  std::vector<metal_compute::SharedBuffer> _codebook;       // n_vq x [Csz,8]
  std::vector<metal_compute::SharedBuffer> _q_outw, _q_outb; // n_vq x [512,8],[512]
  metal_compute::SharedBuffer _rvq_outw, _rvq_outb;          // [768,512],[768]
  int _codebook_size = 1024;

  std::vector<Stage> _stages;     // 4 transformer stages
  metal_compute::SharedBuffer _inv_freq;   // [hd/2] f32, head_dim 64

  // kernels
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_rope;
  metal_compute::ComputeFunction _fn_gemm, _fn_ln, _fn_gelu, _fn_hslice,
      _fn_transpose, _fn_residual, _fn_sdpa, _fn_rope;
};

}  // namespace vpipe::genai

#endif
