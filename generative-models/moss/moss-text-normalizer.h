#ifndef VPIPE_GENERATIVE_MODELS_MOSS_MOSS_TEXT_NORMALIZER_H
#define VPIPE_GENERATIVE_MODELS_MOSS_MOSS_TEXT_NORMALIZER_H

// The MOSS-TTS-v1.5 input normalizer: a port of the checkpoint's own
// `normalize_tts_text` (tts_robust_normalizer_single_script.py), which
// v1.5's processor runs on every user text before it builds the prompt.
//
// It is a ROBUSTNESS clean-up, not text normalization in the TN sense --
// numbers, dates and units are left for the model to read:
//
//   * control / format / zero-width characters dropped, CRLF and U+3000
//     folded;
//   * markdown links become "text url", heading / quote / list markers
//     are stripped, and every line break becomes a sentence break "。";
//   * URLs, e-mails, @mentions, #hashtags, u/ r/ names, ".dotfiles" and
//     file-like tokens (app.js.map, v2.3.1) are protected from the rest;
//   * spaces: collapsed inside Latin runs, removed inside CJK, kept or
//     added at a CJK / Latin boundary;
//   * structural brackets 【】〖〗『』「」 at a sentence start, a stand-alone
//     《title》, arrows (->, =>, →) and long dashes become boundaries;
//   * runs of punctuation collapse (……, ???!!! -> ？！).
//
// [] and {} content is preserved, so inline controls such as
// "[pause 3.2s]" pass through untouched.
//
// Matching is character-for-character with the Python reference,
// including its Unicode classes (\s, \d, \w and category C come from
// tables generated off Python's unicodedata). MOSS-TTS 1.0 ships no such
// script and does none of this; see moss_tts_has_text_normalizer().

#include <string>
#include <string_view>

namespace vpipe::genai {

// Normalize UTF-8 text exactly as v1.5's processor does. Invalid UTF-8
// bytes decode as U+FFFD (Python would have refused the bytes outright).
std::string moss_tts_normalize_text(std::string_view utf8);

// True when a MOSS-TTS checkpoint dir carries the v1.5 normalizer script,
// i.e. its own processor normalizes. model-quantize copies the script
// along with every other sidecar, so a quantized v1.5 dir answers true.
bool moss_tts_has_text_normalizer(const std::string& model_dir);

}  // namespace vpipe::genai

#endif
