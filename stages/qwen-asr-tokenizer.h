#ifndef STAGES_QWEN_ASR_TOKENIZER_H
#define STAGES_QWEN_ASR_TOKENIZER_H

#include <string>

namespace vpipe {

// Native (no-transformers) preparation of the tokenizer.json our runtime
// needs, scoped to the Qwen3-ASR case.
//
// The mlx-community Qwen3-ASR repos ship the tokenizer in the "slow"
// form -- vocab.json + merges.txt + tokenizer_config.json -- but no
// consolidated tokenizer.json. Qwen is a GPT-2-style BYTE-LEVEL BPE, so
// the consolidated file can be reconstructed directly from those inputs
// without the transformers library: model.vocab is vocab.json verbatim,
// model.merges is merges.txt, the special tokens come from
// tokenizer_config.json's added_tokens_decoder, and the pre-tokenizer is
// the fixed Qwen \p{L}/\p{N} regex (which vpipe's Tokenizer routes to its
// hand-coded Qwen/Llama-3 scanner). No metaspace normaliser is emitted,
// so the tokenizer stays byte-level.

// Build a tokenizer.json STRING from the three source-file contents.
// Pure (no I/O). Returns "" and sets `err` on failure (e.g. malformed
// vocab.json). `tokenizer_config_json` may be empty -- the result is
// then valid but carries no special tokens.
std::string
build_qwen_asr_tokenizer_json(const std::string& vocab_json,
                              const std::string& merges_txt,
                              const std::string& tokenizer_config_json,
                              std::string&       err);

// Read vocab.json + merges.txt (+ tokenizer_config.json) from
// `model_dir`, synthesize tokenizer.json, and write it into the same
// directory. Returns false + sets `err` when a required input is absent
// or the write fails. Idempotent callers should skip when tokenizer.json
// already exists.
bool
prepare_qwen_asr_tokenizer_json(const std::string& model_dir,
                                std::string&       err);

}

#endif
