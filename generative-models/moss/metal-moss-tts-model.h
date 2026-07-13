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
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }
namespace vpipe { class SessionContextIntf; }

namespace vpipe::genai {

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

  // Load from a MOSS-TTS model directory (config.json + sharded
  // safetensors). Builds the dense Qwen3 backbone (backbone_only) and loads
  // the bf16 embedding tables + output heads. Returns nullptr on failure.
  static std::unique_ptr<MetalMossTtsModel> load(
      const std::string& model_dir, metal_compute::MetalCompute* mc);

  bool valid() const { return _backbone != nullptr; }
  const Config& config() const { return _cfg; }
  int n_channels() const { return 1 + _cfg.n_vq; }   // 33

  // Optional profiling sink. When set (and profiling is enabled on the
  // session), generate_delay_greedy brackets its prefill and decode phases
  // onto the LLM perf lane (text-prefill / text-decode). No-op if null.
  void set_session(const SessionContextIntf* s) { _session = s; }

  // Delay-pattern generation. `prompt` is the [seq][1+n_vq] int32 input grid
  // (channel 0 text id; 1..n_vq audio codes, audio_pad_code where inactive).
  // `audio` samples the per-codebook audio codes, `text` the free text token
  // (both greedy by default). `seed` 0 => nondeterministic. Returns the
  // generated rows [G][1+n_vq]; stops at <|im_end|> or max_new_tokens.
  // `should_stop`, when set, is polled once per generated step; returning true
  // ends generation early (after the current row) -- the barge-in hook the TTS
  // stage uses to abort an in-flight utterance when new text arrives. The rows
  // produced so far are still returned (and decoded); the delay-pattern tail
  // that never completes is dropped by the caller's de-delay. Default {} runs
  // to <|im_end|> or max_new_tokens as before.
  std::vector<std::vector<std::int32_t>> generate_delay(
      const std::vector<std::vector<std::int32_t>>& prompt,
      int max_new_tokens, const MossSampling& audio = {},
      const MossSampling& text = {}, std::uint64_t seed = 0,
      const std::function<bool()>& should_stop = {});

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

private:
  MetalMossTtsModel() = default;

  // Assemble the [n*hidden] bf16 embedding stream for grid rows
  // [start, start+n): per row, text embed + sum of n_vq audio-code embeds.
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

  metal_compute::ComputeLibrary  _lib_dense;
  metal_compute::ComputeFunction _fn_dense_gemv;
  metal_compute::SharedBuffer    _logits_buf;   // bf16 [vocab + n_vq*(Ac+1)]
};

}  // namespace vpipe::genai

#endif
