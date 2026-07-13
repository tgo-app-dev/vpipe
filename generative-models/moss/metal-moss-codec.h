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
namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

class MetalMossCodec {
public:
  // Load a MOSS-Audio-Tokenizer directory (config.json + sharded
  // safetensors, F32). Returns nullptr on failure. When `int8` is set, the
  // transformer GEMM weights are stored as int8 group-32 affine (~half the
  // resident footprint), dequantized to f16 per stage at decode -- opt-in
  // (a small audio-quality cost). A converted-weight cache (f16 or int8)
  // sidecars the model dir so later loads skip the F32 read + conversion.
  //
  // `with_encoder` additionally loads the ENCODE path (the 4 encoder
  // transformer stages + the quantizer input/in_proj weights) so encode()
  // is available -- it roughly doubles the resident weights, so it is
  // opt-in (voice cloning needs it; plain TTS does not). The encoder cache
  // sidecars under a distinct "-enc" filename, independent of the
  // decoder-only cache.
  static std::unique_ptr<MetalMossCodec> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc,
      bool int8 = false, bool with_encoder = false);

  bool valid() const { return _ok; }
  bool has_encoder() const { return _with_encoder; }
  int sample_rate() const { return _sample_rate; }
  int n_quantizers() const { return _n_vq; }

  // Test / A-B hooks for the M5 matrix-core (matmul2d/NAX) decode paths (the
  // f16 GEMM and the windowed-causal flash attention). No-ops unless the NAX
  // kernels loaded (M5 + not disabled via VPIPE_MOSS_CODEC_NO_MMA2 /
  // NO_ATTN_MMA); production leaves the load-time default (on for M5).
  bool set_use_mma2(bool on)
  { _use_mma2 = on && _mma_available; return _use_mma2; }
  bool use_mma2() const { return _use_mma2; }
  bool set_use_attn_mma(bool on)
  { _use_attn_mma = on && _attn_mma_available; return _use_attn_mma; }
  bool use_attn_mma() const { return _use_attn_mma; }

  // Optional profiling sink. When set (and profiling enabled), decode()
  // brackets onto the LLM perf lane as an "audio-codec" block. No-op if null.
  void set_session(const SessionContextIntf* s) { _session = s; }

  // Decode RVQ codes `codes[T][n_vq]` (0..codebook_size-1) into a [T*1920]
  // f32 PCM waveform at sample_rate. When `stages` is non-null it is filled
  // with the per-stage intermediates in CHANNEL-major [C][T] order (index 0 =
  // post-RVQ [768,T], then one per decoder module) for rel-L2 verification.
  // `n_active` (0 => all n_quantizers) decodes with only the FIRST n_active
  // residual codebooks (residual truncation) -- MOSS-TTS-Realtime emits 16 of
  // this codec's codebooks, so its rows are 16 wide and n_active=16.
  std::vector<float> decode(
      const std::vector<std::vector<std::int32_t>>& codes,
      std::vector<std::vector<float>>* stages = nullptr, int n_active = 0);

  // --- Streaming (incremental) decode -------------------------------------
  // The decoder is causal (windowed-causal attention, patch-reshape upsamples
  // only within a frame), so audio streams chunk-by-chunk as the LM emits
  // codes. Each stage keeps its per-layer K/V in a windowed RING (ring_cap =
  // context + the chunk's frame count), so decode_stream_chunk() is O(chunk),
  // total O(T) like one-shot decode() but low-latency. Concatenating the chunk
  // PCM reproduces decode()'s mono waveform bit-for-bit (f16 noise only).
  struct StreamState {
    std::vector<std::vector<metal_compute::SharedBuffer>> kc, vc;  // [stage][L]
    std::vector<int> pos;   // per-stage absolute input-frame position so far
    std::vector<int> cap;   // per-stage ring capacity (frames)
    int max_chunk = 0;
    int n_active = 0;       // active codebooks (RT emits 16); 0 => all
    // Re-arm for a NEW utterance without reallocating the rings (sized by chunk,
    // not utterance length). Only positions reset; stale ring contents are never
    // read (attention only reaches this utterance's [0, pos+T)).
    void reset() { for (int& p : pos) { p = 0; } }
  };
  // Allocate a fresh streaming state (per utterance). `max_chunk_frames` is the
  // largest latent frame count a single decode_stream_chunk() will pass;
  // `n_active` matches decode()'s (0 => all codebooks). Null if not loaded.
  std::unique_ptr<StreamState> decode_stream_begin(int max_chunk_frames,
                                                   int n_active = 0) const;
  // Decode ONLY the new frames `codes[Cnew][n_vq]` (Cnew <= max_chunk_frames),
  // advancing `st`. Returns the chunk's mono PCM [Cnew*hop] f32; concatenating
  // successive chunks == decode() of the whole code stream.
  std::vector<float> decode_stream_chunk(
      StreamState& st, const std::vector<std::vector<std::int32_t>>& codes);

  // Encode a mono `wave` (f32 @ sample_rate; the caller resamples) into RVQ
  // codes `codes[T][n_vq]` -- the inverse of decode(): the waveform is
  // zero-padded to a multiple of the 1920x downsample, run through the 4
  // encoder transformer stages (each preceded by a DOWN patch-reshape), then
  // RVQ-encoded (the LFQ residual loop: cosine-nearest code per codebook +
  // residual subtraction). T = ceil(wave.size() / 1920). Empty unless the
  // codec was loaded with_encoder (or on a zero-length input).
  std::vector<std::vector<std::int32_t>> encode(const std::vector<float>& wave);

private:
  MetalMossCodec() = default;

  struct StageCfg {
    int in_dim, d_model, n_layers, n_heads, ff, out_dim, patch, context;
  };
  // A GEMM weight [N,K]: f16 (w only), or int8 group-32 affine (w = uint8
  // [N,K] + scale/bias [N,K/32] f16). is_int8() distinguishes; the codec
  // dequants int8 to an f16 temp per stage at decode.
  struct QuantWeight {
    metal_compute::SharedBuffer w;            // f16 [N,K] OR uint8 [N,K]
    metal_compute::SharedBuffer scale, bias;  // [N,K/32] f16 (int8 only)
    int N = 0, K = 0;
    bool empty() const { return w.empty(); }
    bool is_int8() const { return !scale.empty(); }
  };
  struct Layer {
    metal_compute::SharedBuffer n1w, n1b, n2w, n2b;  // LayerNorm w/b
    QuantWeight qkvw;                // in_proj [3d, d]
    QuantWeight ow;                  // out_proj [d,d] (ls1 fold)
    QuantWeight fc1;                 // linear1 [ff, d]
    QuantWeight fc2;                 // linear2 [d, ff] (ls2 fold)
  };
  struct Stage {
    StageCfg cfg;
    QuantWeight in_proj;     // [d_model, in_dim]
    QuantWeight out_proj;    // [out_dim, d_model] (empty=Identity)
    std::vector<Layer> layers;
  };

  // kc/vc null => one-shot; non-null => streaming (T = new frames, `pos` the
  // absolute start, `ring_cap` the K/V ring capacity). See codec-v2 for detail.
  metal_compute::SharedBuffer run_stage_(
      const Stage& st, int T, const metal_compute::SharedBuffer& in,
      std::vector<metal_compute::SharedBuffer>* kc = nullptr,
      std::vector<metal_compute::SharedBuffer>* vc = nullptr, int pos = 0,
      int ring_cap = 0);
  metal_compute::SharedBuffer rvq_decode_(
      const std::vector<std::vector<std::int32_t>>& codes, int T,
      int n_active);
  // RVQ encode (host): the [T, code_dim] encoder hidden -> codes[T][n_vq].
  // input_proj (768->512) then the LFQ residual loop (per codebook: in_proj
  // 512->8, L2-normalized cosine argmax over the codebook, subtract
  // out_proj(raw codebook[idx])).
  std::vector<std::vector<std::int32_t>> encode_rvq_(
      const metal_compute::SharedBuffer& hidden, int T);

  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf*    _session = nullptr;   // profiling sink
  bool _ok = false;
  bool _int8 = false;                                // int8 g32 GEMM weights
  bool _with_encoder = false;                        // encode path loaded
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

  std::vector<Stage> _stages;     // 4 decoder transformer stages

  // Encode path (loaded only when _with_encoder). The 4 encoder transformer
  // stages mirror the decoder (reuse run_stage_); StageCfg::patch here is the
  // DOWN patch applied BEFORE the stage (240,2,2,2). Plus the quantizer
  // encode-side weights: input_proj [rvq_dim,code_dim]+bias, per-codebook
  // in_proj [codebook_dim,rvq_dim]+bias, and the L2-normalized codebook used
  // for the cosine-nearest search (the raw _codebook drives the residual
  // subtraction via _q_outw/_q_outb).
  std::vector<Stage> _enc_stages;
  metal_compute::SharedBuffer _rvq_inw, _rvq_inb;            // [512,768],[512]
  std::vector<metal_compute::SharedBuffer> _q_inw, _q_inb;   // n_vq x [8,512],[8]
  std::vector<metal_compute::SharedBuffer> _codebook_norm;   // n_vq x [Csz,8]

  metal_compute::SharedBuffer _inv_freq;   // [hd/2] f32, head_dim 64

  // kernels
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_rope, _lib_qmm;
  metal_compute::ComputeFunction _fn_gemm, _fn_ln, _fn_gelu, _fn_hslice,
      _fn_transpose, _fn_residual, _fn_sdpa, _fn_rope, _fn_ring_append;
  metal_compute::ComputeFunction _fn_quant;        // int8 g32 quant (load)
  metal_compute::ComputeFunction _fn_qmm8g32;      // fused int8 g32 GEMM

  // M5 matrix-core (matmul2d/NAX) decode acceleration (mirrors CodecV2): the
  // f16 dense GEMM -> dense_gemm_mma, and the scalar windowed-causal attention
  // -> sdpa_causal_mma2_d64_f16 (head_dim 64). The int8 GEMM path stays on the
  // fused affine steel kernel. Loaded only when the GPU has matrix cores.
  bool _mma_available = false;
  bool _use_mma2 = false;
  metal_compute::ComputeLibrary  _lib_dense_mma;
  metal_compute::ComputeFunction _fn_gemm_mma, _fn_gemm_mma_deep;
  bool _attn_mma_available = false;
  bool _use_attn_mma = false;
  metal_compute::ComputeLibrary  _lib_sdpa_mma;
  metal_compute::ComputeFunction _fn_sdpa_mma;

  // Int8 (w8 group-32) matrix-core GEMM path: dequant the weight to f16 once
  // (affine_dequant_w8g32) then run the f16 dense_gemm_mma. This beats both the
  // fused affine STEEL w8 kernel and a fused dequant-in-matmul2d at the codec's
  // prefill-like M (the fused kernel re-dequants the weight per M-tile). Gated
  // on _use_mma2 (the same GEMM lever as the f16 path); int8 mode only.
  metal_compute::ComputeLibrary  _lib_dequant;
  metal_compute::ComputeFunction _fn_dequant_w8g32;
};

}  // namespace vpipe::genai

#endif
