#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TEXT_ENCODER_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TEXT_ENCODER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"

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

    // The vision tower, for `ref2va` references. Only the ref2va task
    // has any, which is why the tower is loaded on demand rather than
    // with the backbone.
    //
    // The DEFAULTS here are the released Qwen3-VL-32B geometry, because
    // the Comfy-Org repack's `minimax_h3_te` metadata carries only the
    // tap depth -- there is no vision_config in it to read. A diffusers
    // checkout has one and config_from_json prefers it; a repack falls
    // back to these, and a mismatch shows up as a shape failure at load
    // rather than as wrong pixels.
    MetalQwenVisionEncoder::Config vision;
  };

  // `path` may be the `text_encoder` directory, the partition root, or
  // the repository root.
  static std::string resolve_encoder_dir(const std::string& path);

  // The LEAST this encoder holds when it streams its layers: everything
  // outside `model.layers.N.` plus the two in-flight slots.
  //
  // Read from the checkpoint before anything loads, because that is when
  // the planning phase asks. Without it a 48 GB encoder is counted at
  // full size in the conditioning phase, and since that is usually the
  // widest phase of a generation graph it decides the whole peak -- a
  // graph whose real requirement is ~9 GB reports 58 GB and is refused.
  static std::size_t streaming_floor_bytes(const std::string& enc_dir);

  // Size a Config from the encoder's `config.json` (`text_config`).
  static bool config_from_json(const std::string& enc_dir, Config& out,
                               std::string* err = nullptr);

  static std::unique_ptr<MiniMaxH3TextEncoder>
  load(const std::string& enc_dir, metal_compute::MetalCompute* mc,
       SessionContextIntf* session, const Config& cfg);

  // The vision tower on its own, over the SAME checkpoint.
  //
  // Separate from load() because the two are wanted at different times:
  // `t2va` / `fl2va` need the backbone and never the tower, and a
  // `ref2va` request runs the tower over every reference before it has a
  // prompt to encode. The tower is ~380M parameters next to the
  // backbone's 32B, so building it is cheap; building the backbone to
  // get at it would not be.
  static std::unique_ptr<MetalQwenVisionEncoder>
  load_vision(const std::string& enc_dir, metal_compute::MetalCompute* mc,
              const Config& cfg, std::string* err = nullptr);

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

  // ---- ref2va: the reference presentation ----------------------------

  // One reference, as the CONDITIONER sees it.
  //
  // The pixels are already encoded: the caller owns the media and runs
  // the tower (see load_vision), so nothing here opens a file or knows
  // what a frame is. An AUDIO reference reaches the conditioner as a
  // label and nothing else -- a waveform is never encoded by it -- which
  // is why it carries no vision at all.
  struct Reference {
    enum class Kind { kImage, kVideo, kAudio };
    Kind kind = Kind::kImage;

    // The tower's output for this reference. One block for an image;
    // `vision->grid_t` blocks for a video, each labelled with its own
    // timestamp. Null for kAudio.
    const MetalQwenVisionEncoder::Result* vision = nullptr;

    // The timestamp of every video block, in seconds -- `grid_t` of
    // them. Rendered "<%.1f seconds>", so these are the labels the model
    // reads and not merely metadata. See video_block_seconds().
    std::vector<float> block_seconds;

    // Whether this reference contributes a soundtrack. A video that
    // carries one is labelled "<Audio j>: " BEFORE its "<Video k>: ",
    // mirroring the order its ROWS are packed in.
    bool has_audio = false;
  };

  // The timestamp of every merged block of a video read at `sample_fps`.
  //
  // Qwen3-VL merges the sampled frames in groups of `temporal_patch`,
  // repeating the last one when the count does not divide, and labels a
  // group with the MEAN of its timestamps. At 2 fps that makes the first
  // block 0.25 s, which "%.1f" renders as "<0.2 seconds>" rather than
  // "<0.3>" -- both C's and Python's formatting round half to even, so
  // the two agree, but the value is surprising enough to be worth
  // stating.
  static std::vector<float> video_block_seconds(int num_sampled_frames,
                                                float sample_fps = 2.0f,
                                                int temporal_patch = 2);

  // Encode MiniMax-H3's presentation of a `ref2va` request.
  //
  // The presentation is a label per reference, numbered PER MODALITY in
  // the order the references are read -- "<Picture i>: " plus a vision
  // block, "<Audio j>: " alone, "<Video k>: " plus one timestamped
  // vision block per merged frame pair -- followed by the prompt
  // verbatim. No chat template and no special tokens, exactly as the
  // text-only path.
  //
  // `token_tags`, when non-null, receives MiniMax-H3's own per-row
  // modality tag: text rows are tagged 1 and a VISION BLOCK's rows are
  // tagged 0 (video). That is not the same thing as the Qwen-internal
  // token-type ids that drive the rotary layout, and the DiT reads this
  // one -- it is what `build_ref2va_packed_sequence` takes as
  // `text_token_tags`.
  //
  // Returns [n_tokens, text_dim] bf16, the same rows encode() returns.
  metal_compute::SharedBuffer
  encode_references(const std::vector<Reference>& refs,
                    std::string_view prompt,
                    std::vector<int>* token_tags = nullptr,
                    int* n_tokens = nullptr, std::string* err = nullptr);

  // The presentation, WITHOUT running the conditioner.
  //
  // Split out because it is the part most likely to be wrong and the
  // part cheapest to check: the labels, their per-modality numbering,
  // the audio-before-video ordering, the block timestamps, the pad
  // counts and the rotary layout are all decided here, and every one of
  // them fails silently. Checking them needs a tokenizer and the
  // reference GRIDS -- not 50 layers of a 32B backbone.
  struct Presentation {
    std::vector<std::int32_t> ids;
    std::vector<int>          tags;    // MiniMax-H3's per-row modality tag
    std::vector<std::int32_t> mrope;   // [3 * n], rows t / h / w

    // Where each vision block's rows landed, in presentation order.
    // `cell` selects the temporal cell within its reference's tower
    // output, which is one buffer across the whole clip.
    struct Run {
      int start = 0, rows = 0, mh = 0, mw = 0, cell = 0;
      const MetalQwenVisionEncoder::Result* vision = nullptr;
    };
    std::vector<Run> runs;

    int size() const { return (int)ids.size(); }
  };

  bool build_presentation(const std::vector<Reference>& refs,
                          std::string_view prompt, Presentation* out,
                          std::string* err = nullptr) const;

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
