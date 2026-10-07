#ifndef VPIPE_GENERATIVE_MODELS_MOSS_METAL_MOSS_TTS_MODEL_H
#define VPIPE_GENERATIVE_MODELS_MOSS_METAL_MOSS_TTS_MODEL_H

// MetalMossTtsModel -- the no-MLX metal forward + delay-pattern generation
// for MOSS-TTS-8B (`MossTTSDelay`, OpenMOSS). A dense Qwen3-8B backbone
// (reused via MetalQwenModel in backbone_only mode, 8-bit affine) driven by
// a [seq, 1+n_vq] token grid: channel 0 = text, channels 1..n_vq = the audio
// RVQ codebooks. The per-position input embedding is the SUM of the text
// embedding + n_vq audio-code embeddings; the backbone hidden feeds 1+n_vq
// output heads (text -> vocab, each codebook -> audio_vocab+1). Generation
// follows the multi-head DELAY pattern: codebook k only becomes active k
// steps after audio starts, and a text-channel state machine emits the
// audio start/gen/delay/end control tokens that drive it. The audio codes
// this produces are decoded to a 24 kHz waveform by the MOSS Audio Tokenizer
// (a separate codec model -- not here). Metal-only; there is no MLX path.
//
// The embedding tables (text + n_vq audio) and all 1+n_vq heads are plain
// bf16 in the checkpoint (only the 36 backbone layers are quantized), so the
// embeddings are assembled on the host (UMA bf16 gather + sum) and the heads
// run as plain bf16 dense GEMV. Everything is bf16 to match the reference's
// dtype (and avoid any conversion). v1 is correctness-first: prefill +
// decode both route through MetalQwenModel::forward_embeddings_hidden (n=1
// per generated row), and the full text head runs every step.

#include "generative-models/qwen3/metal-qwen-model.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }
namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

class WeightSet;       // generative-models/weight-set.h

// Per-channel sampling controls for MOSS-TTS delay-pattern content choices
// (the audio codebooks and the free text token). temperature <= 0 => greedy
// argmax; top_k <= 0 / top_p >= 1 => that cap off; repetition_penalty 1.0 =>
// disabled. The delay-pattern CONTROL transitions stay deterministic.
struct MossSampling {
  float temperature        = 0.0f;   // <=0 => greedy
  int   top_k              = 0;
  float top_p              = 1.0f;
  float repetition_penalty = 1.0f;
};

// How a MOSS-TTS backbone is HELD -- the memory decisions a stage takes
// from the plan before it loads (docs/MODEL-MEMORY.md). Only meaningful
// for an unquantized checkpoint (MOSS-TTS / MOSS-TTS-v1.5 as published);
// a model-quantize'd or MLX 8-bit dir loads as it sits.
struct MossTtsLoadOptions {
  // 8: quantize the backbone's projections to affine w8 group-64 in memory
  // as each layer is built (~53% of the bf16 bytes). 0: keep bf16.
  int  quant_bits = 8;
  // Stream backbone layers that do not fit (decode included), growing the
  // resident set by measurement. An irreversible, load-time choice. An
  // affine w8 pack streams too, as it sits.
  bool stream = false;
  // With quant_bits 8, hold the text embedding table and the text head --
  // the two [vocab, H] tensors, 2.4 GB of bf16 on the 8B -- as affine w8
  // group-64 as well (~53%). The rows every audio step reads (the slot
  // tokens, in and out) are kept exact bf16, so only prompt text and the
  // free text phase see the w8 values. The audio heads and audio
  // embeddings stay bf16: small, and read for every code.
  bool trunk_w8 = true;
};

class MetalMossTtsModel {
public:
  struct Config {
    int n_vq            = 32;      // audio RVQ codebooks (= input channels-1)
    int audio_vocab     = 1024;    // valid codes 0..audio_vocab-1
    int audio_pad_code  = 1024;    // per-codebook pad index (== audio_vocab)
    int hidden          = 4096;
    int vocab           = 155648;
    int sampling_rate   = 24000;
    // Text-channel control token ids (from config.json).
    int audio_start     = 151652;
    int audio_end       = 151653;
    int audio_user_slot = 151654;
    int audio_gen_slot  = 151656;
    int audio_delay_slot = 151662;
    int pad_token       = 151643;
    int im_start        = 151644;
    int im_end          = 151645;
  };

  using LoadOptions = MossTtsLoadOptions;

  // Load from a MOSS-TTS model directory (config.json + sharded
  // safetensors). Builds the dense Qwen3 backbone (backbone_only) and loads
  // the bf16 embedding tables + output heads. Returns nullptr on failure.
  static std::unique_ptr<MetalMossTtsModel> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc,
      const LoadOptions& opts = {});

  // Preferred: the backbone and the embedding/head tables come from ONE
  // checkpoint, which this used to open twice over.
  //
  // The returned model KEEPS the set (its tensors come from it).
  static std::unique_ptr<MetalMossTtsModel> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      const LoadOptions& opts = {});

  // What a checkpoint will cost, from its tensor table alone -- answerable
  // before anything loads, which is when a stage has to declare it.
  struct MemoryPlan {
    std::size_t disk     = 0;   // the checkpoint's bytes on disk
    std::size_t preload  = 0;   // held fully resident, as `quant_bits` builds
    std::size_t floor    = 0;   // held streaming: trunk + two layer slots
    std::size_t retires  = 0;   // disk - preload: what quantizing sheds
    std::size_t trunk    = 0;   // embeddings, heads, final norm
    std::size_t layer    = 0;   // one built layer (a residency unit)
    int         n_layers = 0;
    bool        raw      = false;   // the backbone can be built / streamed
    bool        pack     = false;   // ... from an affine w8 pack
  };
  // The delay-pattern model, its trunk held as LoadOptions::trunk_w8 says.
  static MemoryPlan plan_memory(const std::string& model_dir,
                                int quant_bits, bool trunk_w8 = true);
  // The same for ANY MOSS checkpoint whose backbone is a dense Qwen3
  // stack of `n_layers` under `layer_prefix` -- the delay-pattern model
  // ("language_model.layers."), the realtime one
  // ("language_model.model.layers.") and Local-v1.5 ("transformer.layers.").
  // Everything outside the stack is the trunk; `trunk_w8` (delay layout
  // only) books its text embedding and text head at w8.
  static MemoryPlan plan_memory(const std::string& model_dir,
                                const std::string& layer_prefix,
                                int n_layers, int quant_bits,
                                bool trunk_w8 = false);
  // Whether the text embedding and text head are held w8 (trunk_w8).
  bool trunk_quantized() const { return _trunk_w8; }

  // The weight bytes this model holds NOW: backbone (built layers, slots)
  // plus the embedding tables and heads. Grows while residency promotes
  // layers; read it after load and again after each utterance.
  std::size_t resident_bytes() const;
  // Residency, for a streaming backbone (no-ops otherwise).
  void set_residency_reserve(std::size_t bytes);
  void set_residency_schedule(int forwards);
  std::size_t release_resident_layers(std::size_t bytes);
  bool streams() const;
  // Whether the backbone streams an affine w8 pack as it sits (rather
  // than building its layers from a raw checkpoint).
  bool streams_pack() const;
  bool quantized_on_load() const;
  int  resident_layer_count() const;
  int  n_layers() const;
  // K/V the backbone's pools have grown to.
  std::size_t kv_bytes() const;
  // The backbone's dynamic-int8 prefill tier (LOSSY, matrix cores only;
  // see MetalQwenModel::set_i8_gemm), whether it is on, and how many GEMMs
  // it has taken -- it declines everything under 1024 rows, so only a long
  // prompt reaches it.
  void set_i8_gemm(bool on);
  bool i8_gemm_enabled() const;
  std::uint64_t i8_gemm_count() const;

  bool valid() const { return _backbone != nullptr; }
  const Config& config() const { return _cfg; }
  int n_channels() const { return 1 + _cfg.n_vq; }   // 33

  // Optional profiling sink. When set (and profiling is enabled on the
  // session), generate_delay_greedy brackets its prefill and decode phases
  // onto the LLM perf lane (text-prefill / text-decode). No-op if null.
  void set_session(const SessionContextIntf* s) { _session = s; }

  // Called with every generated row [1+n_vq] as soon as it is chosen;
  // returning false ends generation after that row.
  using RowCb = std::function<bool(const std::vector<std::int32_t>&)>;

  // Delay-pattern generation. `prompt` is the [seq][1+n_vq] int32 input grid
  // (channel 0 text id; 1..n_vq audio codes, audio_pad_code where inactive).
  // `audio` samples the per-codebook audio codes, `text` the free text token
  // (both greedy by default). `seed` 0 => nondeterministic. Returns the
  // generated rows [G][1+n_vq]; stops at <|im_end|> or max_new_tokens.
  // `on_row` sees each row as it is generated -- what the TTS stage streams
  // from (a rolling de-delay turns rows into codec frames) and its barge-in
  // hook (return false to abort an in-flight utterance). The rows produced
  // so far are still returned; the delay-pattern tail that never completes
  // is dropped by the de-delay. It does not perturb sampling: the same seed
  // gives the same rows with or without it.
  std::vector<std::vector<std::int32_t>> generate_delay(
      const std::vector<std::vector<std::int32_t>>& prompt,
      int max_new_tokens, const MossSampling& audio = {},
      const MossSampling& text = {}, std::uint64_t seed = 0,
      const RowCb& on_row = {});

  // Greedy convenience wrapper (used by the load-time warmup + verification).
  std::vector<std::vector<std::int32_t>> generate_delay_greedy(
      const std::vector<std::vector<std::int32_t>>& prompt, int max_new_tokens)
  {
    return generate_delay(prompt, max_new_tokens);
  }

  // Verification: a single active-codebook disagreement with a reference
  // generation. The MOSS audio heads are full of exact bf16 logit ties, so
  // autoregressive token-exactness across two bf16 implementations is
  // impossible -- but the FORWARD is verified correct if every disagreement
  // is a numerical near-tie (my_logit ~= ref_code's logit).
  struct AudioMismatch {
    int   row;
    int   codebook;
    int   my_code;
    int   ref_code;
    float my_logit;       // logit at my (argmax) code
    float ref_logit;      // logit at the reference's code
  };
  // Teacher-force `ref_rows` (a reference [G][1+n_vq] generation) on top of
  // `prompt`: at each row feed the reference row as input, and report every
  // ACTIVE audio codebook (ref code != audio_pad_code) whose greedy argmax
  // disagrees with the reference. A correct forward yields only near-tie
  // disagreements (my_logit - ref_logit ~ 0).
  std::vector<AudioMismatch> teacher_force_audio_mismatches(
      const std::vector<std::vector<std::int32_t>>& prompt,
      const std::vector<std::vector<std::int32_t>>& ref_rows);

  // Verification: teacher-force `rows` on top of `prompt` through the
  // prefill + per-row decode path generation uses, and hand `cb` the head
  // logits that PREDICT row r (before row r is fed): audio[n_vq][Ac+1]
  // and, when `full_text`, the whole [vocab] text head (else empty). The
  // logits are the bf16 values the sampler reads, widened to f32.
  using LogitsCb = std::function<void(
      int r, const std::vector<std::vector<float>>& audio,
      const std::vector<float>& text)>;
  void teacher_force_logits(
      const std::vector<std::vector<std::int32_t>>& prompt,
      const std::vector<std::vector<std::int32_t>>& rows, bool full_text,
      const LogitsCb& cb);

private:
  MetalMossTtsModel() = default;

  // Assemble the [n*hidden] bf16 embedding stream for grid rows
  // [start, start+n): per row, text embed + sum of n_vq audio-code embeds.
  // trunk_w8: quantize one [vocab, H] bf16 table into ONE buffer laid out
  // [codes | scales | biases] (byte offsets of the last two out through
  // `soff` / `boff`), keeping the rows `exact` names in bf16. Empty when
  // the tensor is not a bf16 [N, K] with K a multiple of 64.
  metal_compute::SharedBuffer quantize_table_(
      WeightSet& wts, const std::string& name, const std::vector<int>& exact,
      std::unordered_map<int, std::vector<std::uint16_t>>& rows,
      std::size_t* soff, std::size_t* boff);
  // Row `id` of the text embedding in bf16, from wherever it is held.
  void embed_row_(int id, std::uint16_t* out) const;
  metal_compute::SharedBuffer assemble_embeds_(
      const std::vector<std::vector<std::int32_t>>& rows, int start, int n);

  // Run the output heads on `hn` (the last position's normed hidden) ->
  // audio_logits[n_vq][audio_vocab+1] (always) + text_logits[vocab]. The full
  // 155k-row text head GEMV is the dominant head cost but is only needed for
  // the rare OUTSIDE-audio text sampling (need_full_text); inside audio the
  // text channel only chooses between the gen/delay slot tokens, so those 2
  // logits are computed with a cheap host dot product (need_2slots); when the
  // text token is forced, no text logits are produced.
  void head_logits_(const metal_compute::SharedBuffer& hn, bool need_full_text,
                    bool need_2slots, std::vector<float>& text_logits,
                    std::vector<std::vector<float>>& audio_logits);
  // Dispatch the heads into _logits_buf on a caller-provided encoder (so they
  // can fuse into the decode command buffer): 32 audio heads + the text head
  // when need_full_text. read_heads_ then pulls the results to host.
  void dispatch_heads_(metal_compute::ComputeEncoder& enc,
                       const metal_compute::SharedBuffer& hn,
                       bool need_full_text);
  void read_heads_(const metal_compute::SharedBuffer& hn, bool need_full_text,
                   bool need_2slots, std::vector<float>& text_logits,
                   std::vector<std::vector<float>>& audio_logits);

  metal_compute::MetalCompute*    _mc = nullptr;
  const SessionContextIntf*       _session = nullptr;   // profiling sink
  std::unique_ptr<MetalQwenModel> _backbone;
  Config _cfg;

  // bf16 embedding tables (raw UMA, host-readable): text + n_vq audio.
  metal_compute::SharedBuffer              _embed_tokens;  // [vocab, H]
  std::vector<metal_compute::SharedBuffer> _emb_ext;       // n_vq x [Ac+1, H]
  // bf16 output heads: head 0 [vocab, H], heads 1..n_vq [Ac+1, H].
  std::vector<metal_compute::SharedBuffer> _heads;
  std::vector<int>                         _head_out;      // per-head N

  // trunk_w8: the text embedding and text head as affine w8 group-64, each
  // ONE buffer [codes | scales | biases] the backbone owns and wires with
  // its trunk (scales / biases at the byte offsets beside it);
  // _embed_tokens and _heads[0] stay empty. The slot-token rows are kept
  // exact.
  bool                         _trunk_w8 = false;
  metal_compute::SharedBuffer* _embed_q  = nullptr;
  metal_compute::SharedBuffer* _head0_q  = nullptr;
  std::size_t                  _e_soff = 0, _e_boff = 0;
  std::size_t                  _h_soff = 0, _h_boff = 0;
  std::unordered_map<int, std::vector<std::uint16_t>> _embed_exact;
  std::unordered_map<int, std::vector<std::uint16_t>> _head_exact;
  metal_compute::ComputeLibrary  _lib_qmv;
  metal_compute::ComputeFunction _fn_qmv8;      // affine_qmv_w8g64

  metal_compute::ComputeLibrary  _lib_dense;
  metal_compute::ComputeFunction _fn_dense_gemv;
  metal_compute::SharedBuffer    _logits_buf;   // bf16 [vocab + n_vq*(Ac+1)]
  // The checkpoint, held for this model's whole life.
  std::shared_ptr<WeightSet> _ws;
};

}  // namespace vpipe::genai

#endif
