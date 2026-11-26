#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_TEXT_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_TEXT_H

// MAGI-2-preview's text conditioning, as the release's
// inference/model/qwen35.py and inference_engine.py do it (read, not run):
//
//   * the RAW prompt, no chat template -- a JSON caption (the release's
//     prompt enhancer writes one, {"global_layer": ..., "dynamic_layer":
//     ...}) is first turned into a compact markdown (json_to_compact_
//     markdown), anything else passed as written;
//   * Qwen's tokenizer with a pre-tokenizer step in front that ISOLATES
//     every CJK character (a Split, behavior "isolated") -- and, as its
//     regex engine reads the pattern, every character from '0' to 'U'
//     too (see magi2-text.cc) -- truncated at 7000 tokens;
//   * the condition is Qwen3.5-27B's hidden_states[-3] -- the un-normed
//     output of its third-to-last layer -- every row kept;
//   * the negative is a fixed prompt (DEFAULT_NEGATIVE_PROMPT);
//   * image-to-video addresses the image from the prompt: the text gets
//     "reference_layer:The first frame refers to <Figure 1>" appended (a
//     JSON caption a reference_layer field instead), and the image's
//     special token is the MEAN of the condition's rows over the tokens
//     that overlap "<Figure 1>".

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class Tokenizer;

namespace magi2 {

inline constexpr int kMaxTextTokens = 7000;
// The condition's tap counted from the end: hidden_states[-3] of the
// text model -- the output of layer n_layers - 3 (0-based).
inline constexpr int kTapFromEnd = 3;
inline constexpr const char* kFigureToken = "<Figure 1>";

// The release's default negative prompt.
const std::string& negative_prompt();

// _normalize_prompt: a prompt that parses as a JSON object or array goes
// through json_to_compact_markdown -- which returns it as written unless
// it is an object with a "global_layer" -- anything else as written.
std::string normalize_prompt(const std::string& prompt);

// _ensure_figure_tokens: a JSON object prompt gets
// "reference_layer": ["The first frame refers to <Figure 1>"] (and is
// re-serialized as Python's json.dumps(ensure_ascii=False) writes it); any
// other prompt has "reference_layer:The first frame refers to <Figure 1>"
// appended.
std::string with_figure_reference(const std::string& prompt);

// The token ids of an already-normalized text: special tokens split out
// as the HF tokenizer does, every CJK character its own pre-token, the
// rest through the tokenizer's own pre-tokenizer; at most kMaxTextTokens.
std::vector<std::int32_t> encode(const Tokenizer& tok,
                                 const std::string& text);

// The indices of the tokens of `ids` (the encode of `text`) whose bytes
// overlap `target`'s first occurrence in `text`; empty when it is absent.
std::vector<int> target_span(const Tokenizer& tok,
                             const std::vector<std::int32_t>& ids,
                             const std::string& text,
                             const std::string& target);

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_TEXT_H
