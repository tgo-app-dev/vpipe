#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TEXT_ENCODER_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TEXT_ENCODER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class SessionContextIntf;

namespace genai {

class Tokenizer;
class WeightSet;

// MiniMax-H3's text conditioning: a Qwen3-VL-32B backbone tapped at an
// INTERMEDIATE layer.
//
// Three things about this are unlike every other text encoder here, and
// each is silent if got wrong:
//
//   * The prompt is used VERBATIM. No chat template, no BOS/EOS, no
//     padding to a fixed length -- the reference tokenizes with
//     `add_special_tokens=False` and feeds exactly those tokens. A
//     template that another family needs (FLUX.2's `<think>` block,
//     Krea-2's grounded wrapper) is a different conditioning here, not a
//     harmless prefix.
//   * The conditioning is `hidden_states[50]` of 64 layers, NOT the
//     final hidden state. HF's tuple puts the embedding output at index
//     0, so index 50 is the UN-NORMED residual stream after 50 decoder
//     layers -- the model's own `norm` is never applied to it. Only the
//     LAST entry of that tuple is normed, which is what makes this easy
//     to reproduce incorrectly with a truncated reference.
//   * Because the tap is at layer 50, layers 50..63 are never evaluated
//     and are NOT LOADED. That is ~22% of a 62 GB checkpoint, which on
//     a 64 GB box is the difference between fitting and not.
//
// Below 64 GB even the tapped 50 layers do not fit (~48 GB bf16, ~26 GB
// at w8), so `Config::lm.stream_layers` builds each layer inside the
// prefill and frees it after -- ~one layer plus the embedding table
// resident. An encoder only ever PREFILLS, which is what makes that a
// sane trade here and not for a chat model. The stage decides via the
// same model_memory::plan_streaming rule the DiTs use.
//
// Text-only prompts take plain 1-D rope even though Qwen3-VL is an
// mROPE model: with no vision block every one of the three position
// axes carries the sequence position, so channel c sees
// `pos * inv_freq[c]` whichever section it belongs to -- identical to
// standard rope, interleaved or not.
class MiniMaxH3TextEncoder {
 public:
  struct Config {
    // The HF `output_hidden_states` index the DiT conditions on, which
    // is also the number of decoder layers that have to run. The
    // reference's `get_qwen3vl_prompt_embeds()` defaults it to 50.
    int tap = 50;
    // Layers the checkpoint actually has, for the range check. Not the
    // number loaded -- that is `tap`.
    int total_layers = 64;
    int text_dim = 5120;
    // Whether the checkpoint carries a `quantization` block. Distinct
    // from lm.quant_bits, which has a non-zero DEFAULT and so cannot
    // tell a dense checkpoint from a 4-bit one.
    bool quantized = false;
    MetalQwenModel::Config lm;
  };

  // `path` may be the `text_encoder` directory, the partition root, or
  // the repository root.
  static std::string resolve_encoder_dir(const std::string& path);

  // Size a Config from the encoder's `config.json` (`text_config`).
  static bool config_from_json(const std::string& enc_dir, Config& out,
                               std::string* err = nullptr);

  static std::unique_ptr<MiniMaxH3TextEncoder>
  load(const std::string& enc_dir, metal_compute::MetalCompute* mc,
       SessionContextIntf* session, const Config& cfg);

  ~MiniMaxH3TextEncoder();

  // Whether the backbone streams its layers through the prefill instead
  // of holding them (Config::lm.stream_layers -- the 16 GB path), and
  // how many leading layers stayed pinned. For the stage's log line.
  bool streaming() const { return _lm && _lm->streaming_layers(); }
  int  pinned_layers() const { return _lm ? _lm->pinned_layers() : 0; }

  // Tokenize VERBATIM -- no chat template, no special tokens.
  std::vector<std::int32_t> tokenize(std::string_view prompt) const;

  // Conditioning for `prompt`: [n_tokens, text_dim] bf16, the row
  // layout the DiT's text rows take.
  metal_compute::SharedBuffer
  encode(std::string_view prompt, int* n_tokens = nullptr,
         std::string* err = nullptr);

  // Same, from ids the caller already has. Exists so a test can pin the
  // forward pass on fixed ids without also pinning the tokenizer, and
  // so the multimodal path can splice vision rows in first.
  metal_compute::SharedBuffer
  encode_ids(const std::vector<std::int32_t>& ids, std::string* err = nullptr);

  const Config& config() const { return _cfg; }

 private:
  MiniMaxH3TextEncoder() = default;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;
  std::unique_ptr<MetalQwenModel> _lm;
  std::unique_ptr<Tokenizer> _tok;
  metal_compute::SharedBuffer _embed;
};

}  // namespace genai
}  // namespace vpipe

#endif
