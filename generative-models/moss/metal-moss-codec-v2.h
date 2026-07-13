#ifndef VPIPE_GENERATIVE_MODELS_MOSS_METAL_MOSS_CODEC_V2_H
#define VPIPE_GENERATIVE_MODELS_MOSS_METAL_MOSS_CODEC_V2_H

// MetalMossCodecV2 -- the no-MLX metal DECODE path of MOSS-Audio-Tokenizer-v2
// (the companion codec of MOSS-TTS-Local-v1.5). Turns the 12 RVQ codes/frame
// the v1.5 LM emits into a 48 kHz STEREO waveform. Same architecture family as
// the 24 kHz MetalMossCodec but a deeper decoder (SIX ProjectedTransformer
// stages: 32L@1280 + 12L@768 x5, causal + RoPE + LayerScale, exact-GELU FFN)
// with patch-reshape upsamplers between them (x2 x2 x2 x2 x2 x240 = x7680) and
// a final channel de-interleave (the codec runs the 2 channels interleaved in
// time, then splits flat[2k]->L, flat[2k+1]->R). The v2 checkpoint names the
// per-layer projections self_attn.in_proj / out_proj + ffn.0 / ffn.2 (v1 uses
// in_projs.0), so this is a sibling class rather than a reconfig of v1.
//
// The checkpoint is F32; we run f16 (better audio precision than bf16, and the
// codec is not the pipeline bottleneck). Weight-norm 1x1 convs (the RLFQ
// projections) and the two per-channel LayerScales are folded into plain
// weight matrices at load. Decode-only (correctness-first, staged rel-L2
// verified vs the HF golden); int8/encoder are v1 extras, not ported here.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }
namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

class MetalLlamaWeights;

class MetalMossCodecV2 {
public:
  // `with_encoder` additionally loads the ENCODE path (the 6 encoder
  // transformer stages + the RLFQ quantizer's input projections), so encode()
  // works. Decode-only callers pass false (the default) and pay nothing.
  static std::unique_ptr<MetalMossCodecV2> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc,
      bool with_encoder = false);

  bool valid() const { return _ok; }
  bool has_encoder() const { return _with_encoder; }
  int sample_rate() const { return _sample_rate; }
  int channels() const { return _channels; }
  int n_quantizers() const { return _n_vq; }

  // Test / A-B hook for the M5 matrix-core (matmul2d/NAX) decode GEMM path.
  // set_use_mma2 is a no-op unless the NAX kernels loaded (M5 + not disabled
  // via VPIPE_MOSS_CODEC_NO_MMA2); production leaves the load-time default
  // (on for M5). Returns the effective state.
  bool set_use_mma2(bool on)
  { _use_mma2 = on && _mma_available; return _use_mma2; }
  bool use_mma2() const { return _use_mma2; }

  // Test / A-B hook for the M5 matrix-core windowed-causal flash attention
  // (sdpa_causal_mma2_d64_f16) that replaces the scalar sdpa_causal_window_f16.
  // No-op unless the NAX attn kernel loaded (M5 + not disabled via
  // VPIPE_MOSS_CODEC_NO_ATTN_MMA). Returns the effective state.
  bool set_use_attn_mma(bool on)
  { _use_attn_mma = on && _attn_mma_available; return _use_attn_mma; }
  bool use_attn_mma() const { return _use_attn_mma; }

  // Decode codes `codes[T][n_vq]` (n_vq=12, 0..1023) into a stereo waveform,
  // returned channel-major flat: [ch0 (T*3840 samples) | ch1 (T*3840)] f32.
  // When `stages` is non-null it is filled with the per-stage intermediates in
  // CHANNEL-major [C][T'] order (index 0 = post-RLFQ latent [768,T], then one
  // per decoder transformer + its patch upsample) for rel-L2 verification.
  std::vector<float> decode(
      const std::vector<std::vector<std::int32_t>>& codes,
      std::vector<std::vector<float>>* stages = nullptr);

  // C1 helper: just the RLFQ-dequantized latent [T, 768] f16 (decoder input).
  metal_compute::SharedBuffer decode_latent(
      const std::vector<std::vector<std::int32_t>>& codes, int T);

  // --- Streaming (incremental) decode -------------------------------------
  // The decoder is causal (windowed-causal attention, patch-reshape upsamples
  // only within a frame), so audio can be produced chunk-by-chunk as the LM
  // emits codes instead of decoding the whole utterance at once. Each stage
  // keeps its per-layer K/V in a windowed RING (ring_cap = that stage's
  // attention context), so decode_stream_chunk() is O(chunk) -- total work
  // O(T), same as one-shot decode() but streamed at low latency. Absolute
  // RoPE positions are preserved across chunks, so concatenating the chunk
  // PCM reproduces decode()'s waveform bit-for-bit (f16 reduction noise only).
  struct StreamState {
    // [stage][layer] windowed K/V ring caches, each [n_heads, cap, hd]. The
    // ring capacity `cap` (>= context + the stage's per-chunk frame count) is
    // sized so a chunk's K/V appends never evict context that same chunk's
    // earliest query still needs; attention masking stays at `context`.
    std::vector<std::vector<metal_compute::SharedBuffer>> kc, vc;
    std::vector<int> pos;   // per-stage absolute input-frame position so far
    std::vector<int> cap;   // per-stage ring capacity (frames)
    int max_chunk = 0;      // max latent frames/chunk this state was sized for
    // Re-arm for a NEW utterance without reallocating the rings (they are sized
    // by chunk, not utterance length). Only the absolute positions reset; stale
    // ring contents are never read -- attention only reaches this utterance's
    // [0, pos+T), all freshly appended after the reset.
    void reset() { for (int& p : pos) { p = 0; } }
  };
  // Allocate a fresh streaming state (per utterance). `max_chunk_frames` is the
  // largest number of LATENT frames a single decode_stream_chunk() call will
  // pass -- the rings are sized for it. Null if not loaded.
  std::unique_ptr<StreamState> decode_stream_begin(int max_chunk_frames) const;
  // Decode ONLY the new frames `codes[Cnew][n_vq]` (Cnew <= max_chunk_frames),
  // advancing `st`. Returns the chunk's PCM channel-major flat
  // [ch0(Cnew*hop) | ch1(Cnew*hop)] f32; bit-concatenating successive chunks
  // == decode() of the full code stream.
  std::vector<float> decode_stream_chunk(
      StreamState& st, const std::vector<std::vector<std::int32_t>>& codes);

  // Encode a 48 kHz STEREO waveform into the 12 RVQ code streams the decoder
  // consumes -- the inverse of decode(). `wave` is channel-major flat:
  // [ch0 (N samples) | ch1 (N samples)] f32 (the same layout decode() returns).
  // The per-channel signal is zero-padded to a multiple of the 3840x hop, run
  // through the channel-interleave + first patch-reshape + 6 encoder
  // transformer stages (each preceded by a DOWN patch-reshape), then RLFQ-
  // encoded (per-codebook in_proj -> cosine-nearest -> residual subtraction).
  // Returns codes[T][n_vq] (T = ceil(N / 3840)). Empty unless loaded
  // with_encoder (or on empty input). When `stages` is non-null it is filled
  // with the per-stage encoder intermediates in CHANNEL-major [C][T'] order
  // (one per transformer stage output, then the final [768,T] latent) for
  // rel-L2 verification.
  std::vector<std::vector<std::int32_t>> encode(
      const std::vector<float>& wave,
      std::vector<std::vector<float>>* stages = nullptr);

private:
  MetalMossCodecV2() = default;

  struct StageCfg {
    int in_dim, d_model, n_layers, n_heads, ff, out_dim, patch, context;
  };
  struct Layer {
    metal_compute::SharedBuffer n1w, n1b, n2w, n2b;  // LayerNorm w/b
    metal_compute::SharedBuffer qkvw;                // self_attn.in_proj [3d,d]
    metal_compute::SharedBuffer ow;                  // out_proj [d,d] (ls1 fold)
    metal_compute::SharedBuffer fc1;                 // ffn.0 [ff,d]
    metal_compute::SharedBuffer fc2;                 // ffn.2 [d,ff] (ls2 fold)
  };
  struct Stage {
    StageCfg cfg;
    metal_compute::SharedBuffer in_proj;     // [d_model, in_dim] (always here)
    metal_compute::SharedBuffer out_proj;    // [out_dim, d_model] (empty=Id)
    std::vector<Layer> layers;
  };

  bool init_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
             bool with_encoder);
  // Runs one decoder transformer stage. In one-shot mode (kc/vc null) T is the
  // full frame count and K/V come from the local per-call buffers. In STREAMING
  // mode (kc/vc = the stage's per-layer ring caches) T is the new-frame count,
  // `pos` the absolute start position; new K/V are appended into the rings and
  // attention reads them windowed -- the layer math is otherwise identical.
  metal_compute::SharedBuffer run_stage_(
      const Stage& st, int T, const metal_compute::SharedBuffer& in,
      std::vector<metal_compute::SharedBuffer>* kc = nullptr,
      std::vector<metal_compute::SharedBuffer>* vc = nullptr, int pos = 0,
      int ring_cap = 0);
  // Host RLFQ encode: encoder latent [T, code_dim] (f16) -> codes[T][n_vq].
  std::vector<std::vector<std::int32_t>> encode_rvq_(
      const metal_compute::SharedBuffer& hidden, int T);

  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;   // profiling sink (null=off)
  bool _ok = false;
  bool _with_encoder = false;
  int _sample_rate = 48000;
  int _channels = 2;
  int _n_vq = 12;            // codes/frame the LM emits (codec defines 32)
  int _codebook_size = 1024;
  int _codebook_dim = 8;
  int _rvq_dim = 512;
  int _code_dim = 768;

  // RLFQ: per-codebook embed [1024,8] + folded out_proj [512,8]+bias[512];
  // shared output_proj [768,512]+bias[768].
  std::vector<metal_compute::SharedBuffer> _codebook;        // n_vq x [1024,8]
  std::vector<metal_compute::SharedBuffer> _q_outw, _q_outb; // [512,8],[512]
  metal_compute::SharedBuffer _rvq_outw, _rvq_outb;          // [768,512],[768]

  std::vector<Stage> _stages;     // 6 decoder transformer stages

  // Encode path (loaded only when _with_encoder). The 6 encoder transformer
  // stages (each StageCfg.patch = the DOWN patch-reshape applied BEFORE it),
  // the RLFQ input projections, and the L2-normalized codebooks for the
  // cosine-nearest search.
  std::vector<Stage> _enc_stages;                            // 6 encoder stages
  metal_compute::SharedBuffer _rvq_inw, _rvq_inb;            // [512,768],[512]
  std::vector<metal_compute::SharedBuffer> _q_inw, _q_inb;   // [8,512],[8] each
  std::vector<metal_compute::SharedBuffer> _codebook_norm;   // n_vq x [1024,8]

  metal_compute::SharedBuffer _inv_freq;   // [hd/2] f32, head_dim 64

  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_sdpa,
      _lib_rope;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_ln, _fn_gelu,
      _fn_hslice, _fn_transpose, _fn_residual, _fn_sdpa, _fn_rope,
      _fn_ring_append;   // windowed K/V ring scatter (streaming decode)

  // M5-only matrix-core (matmul2d/NAX) dense GEMM -- the same NAX path the LM
  // prefill and the Gemma/ASR encoders use. Loaded only when the GPU supports
  // matrix cores (Apple10+); the transformer stages are GEMM-dominated at the
  // long upsampled sequence lengths (later stages run at 2..32x the frame
  // count), so routing y = x @ w^T through matmul2d is the decode win. The
  // codec GEMMs carry no bias, so no bias-add fold is needed.
  bool _mma_available = false;
  bool _use_mma2 = false;
  metal_compute::ComputeLibrary  _lib_dense_mma;
  metal_compute::ComputeFunction _fn_gemm_mma, _fn_gemm_mma_deep;

  // M5 matrix-core windowed-causal flash attention (sdpa_causal_mma2_d64_f16),
  // replacing the scalar per-query sdpa_causal_window_f16 -- the dominant
  // decode cost (attention is ~86% of decode at the long upsampled sequence
  // lengths; the scalar kernel wasted 32x on redundant per-key simd_sum/exp).
  bool _attn_mma_available = false;
  bool _use_attn_mma = false;
  metal_compute::ComputeLibrary  _lib_sdpa_mma;
  metal_compute::ComputeFunction _fn_sdpa_mma;
};

}  // namespace vpipe::genai

#endif
