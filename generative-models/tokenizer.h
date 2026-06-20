#ifndef VPIPE_GENERATIVE_MODELS_TOKENIZER_H
#define VPIPE_GENERATIVE_MODELS_TOKENIZER_H

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::genai {

class GgufFile;

// Hugging Face style byte-level BPE tokenizer parsed from a
// `tokenizer.json` file. Covers the Llama / Qwen / GPT-2 family of
// vocabularies: model.type == "BPE", a byte-level alphabet, and a
// (optionally absent) pre-tokenizer regex that splits raw input
// into chunks before BPE merging.
//
// v1 scope (this commit):
//
//   * Encoder: pre-tokenizer regex (std::regex / ECMAScript syntax,
//     no \p{L}/\p{N}) followed by byte-level encoding and BPE
//     merging. Real-Llama-3.1's Unicode-aware regex requires a
//     follow-up backend.
//   * Decoder: bulk + streaming. The streaming decoder buffers
//     partial UTF-8 sequences across tokens so callers never see
//     a half-codepoint chunk.
//   * Special tokens: looked up by symbolic name (e.g.
//     "<|begin_of_text|>"). They are NOT auto-recognized in input
//     text -- callers prepend the appropriate id explicitly.
//
// Thread safety: a constructed Tokenizer is immutable; encode(),
// decode(), step(), and special_token_id() are MT-safe. The
// StreamDecoder holds per-stream state and is owned by one thread
// at a time.
class Tokenizer {
public:
  // Build from a tokenizer.json file. Returns nullptr on failure
  // (failure reported through `session->warn()`).
  static std::unique_ptr<Tokenizer>
  from_huggingface_json(std::string_view          path,
                        const SessionContextIntf* session);

  // For round-trip tests and other callers that already have the
  // tokenizer.json content in memory: parse from a literal string.
  // `tag` appears in error messages so the caller knows what was
  // being parsed.
  // Build from an open GGUF checkpoint's embedded tokenizer (the
  // `tokenizer.ggml.*` metadata: tokens / merges / token_type / model / pre).
  // Used for pure-GGUF model dirs that ship no tokenizer.json. The
  // `tokenizer.ggml.model` field selects the scheme, matching the HF fast
  // tokenizer: "gpt2" -> GPT-2 byte-level BPE (Qwen, Llama-3 -- the Ġ/Ċ
  // alphabet + the \p{L}/\p{N} pre-tokenizer); "llama" / other SentencePiece
  // (Gemma) -> metaspace BPE (U+2581 marker + ByteFallback). nullptr on failure.
  static std::unique_ptr<Tokenizer>
  from_gguf(const GgufFile& gguf, const SessionContextIntf* session);

  // For round-trip tests and other callers that already have the
  // tokenizer.json content in memory: parse from a literal string.
  // `tag` appears in error messages so the caller knows what was
  // being parsed.
  static std::unique_ptr<Tokenizer>
  from_huggingface_string(std::string_view          json,
                          std::string_view          tag,
                          const SessionContextIntf* session);

  ~Tokenizer();

  Tokenizer(const Tokenizer&)            = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // Encode UTF-8 text. Does NOT add BOS/EOS or any other special
  // marker; callers (typically the chat template) prepend the
  // appropriate special id via special_token_id() if needed.
  std::vector<std::int32_t>
  encode(std::string_view text) const;

  // Bulk decode. Same byte-level + special-token rules as the
  // streaming decoder, but materialises the whole string at once.
  std::string
  decode(std::span<const std::int32_t> ids) const;

  // Per-generation streaming decoder state. Holds at most a few
  // pending UTF-8 continuation bytes between calls.
  struct StreamDecoder {
    std::string pending;   // bytes that don't yet form a complete
                           // UTF-8 sequence
  };

  StreamDecoder
  make_stream_decoder() const;

  // Feed one token id, get back the next renderable text chunk.
  // The chunk is the longest prefix of the cumulative byte stream
  // that ends on a complete UTF-8 codepoint. Empty when the new
  // id closes a multi-byte sequence whose start is still in the
  // pending buffer (rare; a 4-byte codepoint split across tokens
  // can produce up to three empty step() returns in a row).
  std::string
  step(StreamDecoder& sd, std::int32_t id) const;

  // Look up a special token id by symbolic name (e.g.
  // "<|begin_of_text|>"). Returns -1 when the name is unknown.
  std::int32_t
  special_token_id(std::string_view name) const;

  // Vocab size including special tokens. The id space is dense in
  // [0, vocab_size()).
  std::int32_t vocab_size() const noexcept;

  // For diagnostics + tests.
  bool has_pre_tokenizer() const noexcept;
  std::size_t merge_count() const noexcept;
  std::size_t special_token_count() const noexcept;

private:
  Tokenizer();
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}

#endif
