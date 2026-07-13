#ifndef VPIPE_GENAI_MOSS_METAL_MOSS_RT_MODEL_H
#define VPIPE_GENAI_MOSS_METAL_MOSS_RT_MODEL_H

// MetalMossRtModel -- the no-MLX metal forward + streaming generation for
// MOSS-TTS-Realtime (`MossTTSRealtime`, OpenMOSS). Two PLAIN Qwen3 transformers
// reused via MetalQwenModel (backbone_only, dense, 8-bit affine):
//   * BACKBONE  = Qwen3-1.7B (28 layers, hidden 2048, 16/8 GQA, head_dim 128,
//     SwiGLU 6144, rope 1e6). Driven by a [seq, 1+n_vq]=17 token grid whose
//     per-position input embedding is text_embed(ch0) + sum audio_embed_i(ch i)
//     (i=1..16). Owns the streaming KV; NO text output head (text is DRIVEN in,
//     not predicted).
//   * LOCAL ("depth") = a 4-layer Qwen3 (same shape) that, seeded PER FRAME by
//     the backbone's last hidden, autoregressively emits the 16 RVQ codebooks:
//     pos 0 input = the backbone hidden, pos k input = local_embed[k-1](code
//     k-1); head[k] (bf16) -> codebook k. Per-frame KV is a fresh context.
//
// Generation (mirrors inferencer.py `_generate_from_ids`, single-turn TTS):
// prefill = prompt grid + the first min(len,12) text-token rows (BOS 1025 at
// ch1 of the last prefill row); backbone forward -> local frame 0. Then per
// step: feed [next_text_token (or text_pad 151655), prev-frame 16 codes] ->
// backbone 1-step -> local frame. STOP when the local codebook-0 sample == EOS
// (1026). Audio codes are SAMPLED (greedy degenerates); the emitted codes feed
// the 24 kHz MOSS Audio Tokenizer (MetalMossCodec, decoded with 16 codebooks).
//
// The embedding tables (17 top + 15 local feedback) + 16 heads are plain bf16
// in the checkpoint; embeddings are assembled on the host (bf16 gather + sum)
// and the heads run as f16 dense GEMV. Metal-only; there is no MLX path.
//
// The LM dir may be EITHER the unquantized bf16 checkpoint (runs as-is: the
// transformers' linears load raw, MetalQwenModel auto-detects the dense f16/
// bf16 GEMM/GEMV path) OR a model-quantize'd 8-bit dir (~2x faster, half the
// resident bytes). Same Config for both -- MetalQwenModel picks raw vs 8-bit
// affine from the checkpoint (a `.weight` with no `.scales` => dense). Plain
// Qwen3 => Config.zero_centered_norm MUST be false (the dense path folds +1
// into the norms otherwise; no effect on the quantized path).

#include "generative-models/moss/metal-moss-tts-model.h"   // MossSampling
#include "generative-models/qwen3/metal-qwen-model.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <functional>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; class ComputeEncoder; }
namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

class MetalLlamaWeights;

class MetalMossRtModel {
public:
  struct Config {
    MetalQwenModel::Config backbone;   // Qwen3-1.7B (28L), 8-bit dense
    MetalQwenModel::Config local;      // 4-layer depth Qwen3, 8-bit dense
    int n_vq           = 16;           // RVQ codebooks / frame
    int audio_vocab    = 1027;         // head width (0..1023 codes,1024 pad,
                                       // 1025 BOS, 1026 EOS)
    int audio_pad_code = 1024;
    int audio_bos      = 1025;
    int audio_eos      = 1026;
    int text_pad       = 151655;       // fed on ch0 once the text is exhausted
    int hidden         = 2048;
    int sampling_rate  = 24000;
  };

  static std::unique_ptr<MetalMossRtModel> load(
      const std::string& quant_dir, metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Streaming single-turn TTS. `prompt_grid` is the fixed system/assistant
  // prefix ([seq][1+n_vq], from moss_rt_build_prompt_grid). `text_ids` are the
  // target-text token ids to speak (channel 0 stream). Generates up to
  // `max_frames` frames of n_vq codes; stops when the local codebook-0 sample
  // == audio_eos. `audio_sp.temperature <= 0` => greedy (tests / golden);
  // otherwise the per-codebook logits are sampled (audio MUST be sampled in
  // production -- greedy degenerates into silence). `seed` seeds the sampler
  // (0 => fresh).
  // `on_frame`, when set, is called with each generated frame's n_vq codes as
  // soon as it is finalized -- the streaming hook the TTS stage uses to
  // interleave codec decode + PCM emission. Returning false stops generation
  // early. Returned frames are unaffected; default {} keeps one-shot behavior.
  using FrameCb = std::function<bool(const std::vector<int>&)>;
  std::vector<std::vector<int>> generate(
      const std::vector<std::vector<std::int32_t>>& prompt_grid,
      const std::vector<std::int32_t>& text_ids, int max_frames,
      const MossSampling& audio_sp = {}, std::uint64_t seed = 0,
      const FrameCb& on_frame = {});

  const Config& config() const { return _cfg; }
  int sampling_rate() const { return _cfg.sampling_rate; }

  // Test hooks (metal_lm_smoke): assemble a grid's bf16 embedding stream and
  // run ONE local depth frame from a backbone hidden. Public so the smoke test
  // can compare the backbone hidden + leading codebooks vs the python golden.
  metal_compute::SharedBuffer assemble_embeds(
      const std::vector<std::vector<std::int32_t>>& grid, int start, int n)
  { return assemble_embeds_(grid, start, n); }
  MetalQwenModel* backbone() { return _bb.get(); }
  // Depth-decode ONE frame from the backbone seed hidden (bf16 [hidden]);
  // `sp.temperature <= 0` => greedy. Advances `rng` (per-frame LCG). Public so
  // the smoke test can compare the leading codebooks vs the python golden.
  std::vector<int> local_frame(const metal_compute::SharedBuffer& seed_hidden,
                               const MossSampling& sp, std::uint64_t& rng);

private:
  bool init_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
             const Config& cfg, const std::string& quant_dir);
  // Assemble n grid rows -> bf16 [n*hidden] (text embed + sum audio embeds).
  metal_compute::SharedBuffer assemble_embeds_(
      const std::vector<std::vector<std::int32_t>>& grid, int start, int n);
  // Fully-fused whole-frame depth decode: the entire 16-codebook loop (per k:
  // gather prev code's embed -> local decode step -> head GEMV -> on-GPU
  // argmax/sample -> _codes[k]) runs in ONE command buffer (one commit+wait per
  // frame). The local transformer decodes in f16 via MetalQwenModel's
  // encode_decode_prewritten (its decode input written on-GPU by the gather).
  // `sp == nullptr` => greedy argmax. `seed64` seeds the per-codebook GPU
  // sampler. Returns the n_vq codes (host-read after the single commit).
  std::vector<int> decode_frame_fused_(const metal_compute::SharedBuffer& seed_bb,
                                       const MossSampling* sp,
                                       std::uint64_t seed64);
  // Encoder helpers (no commit): codebook-k head GEMV (f16) -> _logits;
  // on-GPU argmax / sample -> _codes[k]; embed-gather _local_embed[k](_codes[k])
  // -> `out` (the local transformer's decode input buffer).
  void encode_head_(metal_compute::ComputeEncoder& e, int k,
                    const metal_compute::SharedBuffer& h);
  void encode_argmax_(metal_compute::ComputeEncoder& e, int k);
  void encode_sample_(metal_compute::ComputeEncoder& e, int k,
                      const MossSampling& sp, std::uint32_t seed);
  void encode_gather_(metal_compute::ComputeEncoder& e, int k,
                      const metal_compute::SharedBuffer& out);
  // Apply codebook-k's cross-frame repetition penalty to _logits in-place
  // (reads _hist[k], length _hist_len[k]).
  void encode_rep_penalty_(metal_compute::ComputeEncoder& e, int k,
                           float penalty);

  static constexpr int kRepWindow = 50;   // repetition-penalty history window

  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;
  Config _cfg;
  std::unique_ptr<MetalQwenModel> _bb;     // backbone (bf16, 8-bit)
  std::unique_ptr<MetalQwenModel> _lt;     // local / depth (f16, 8-bit)

  metal_compute::SharedBuffer _text_embed;                // bf16 [vocab,hidden]
  std::vector<metal_compute::SharedBuffer> _audio_embed;  // n_vq x bf16 [V,h]
  std::vector<metal_compute::SharedBuffer> _local_embed;  // n_vq-1 x f16 [V,h]
  std::vector<metal_compute::SharedBuffer> _heads;        // n_vq x f16 [V,h]
  metal_compute::SharedBuffer _logits;   // f16 [audio_vocab]
  metal_compute::SharedBuffer _codes;    // int32 [n_vq] (on-GPU sampled codes)
  metal_compute::SharedBuffer _lc_ws;    // f16 [audio_vocab] sampler workspace
  metal_compute::SharedBuffer _hist;     // int32 [n_vq*kRepWindow] rep-pen hist
  std::vector<int> _hist_len;                  // valid rep-pen entries / codebook
  std::vector<std::vector<int>> _hist_host;    // rolling per-codebook history

  metal_compute::ComputeLibrary  _lib_dense, _lib_elt;
  metal_compute::ComputeFunction _fn_gemv, _fn_argmax, _fn_sample, _fn_gather;
  metal_compute::ComputeFunction _fn_rep;
};

}  // namespace vpipe::genai

#endif
