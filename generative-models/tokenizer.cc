#include "generative-models/tokenizer.h"

#include "generative-models/shared/gguf-file.h"
#include "common/flex-data.h"
#include "common/media-line.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::genai {

// ---------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------

namespace {

// Decode one codepoint starting at `data + pos`. Returns the codepoint
// and writes the number of bytes consumed to `*out_len`. On invalid
// input returns 0xFFFD and consumes 1 byte (best-effort recovery).
uint32_t
utf8_next_(const char* data, size_t len, size_t pos, size_t* out_len)
{
  auto b0 = static_cast<unsigned char>(data[pos]);
  if (b0 < 0x80) {
    *out_len = 1;
    return b0;
  }
  size_t need = 0;
  uint32_t cp = 0;
  if ((b0 & 0xE0) == 0xC0) { need = 2; cp = b0 & 0x1F; }
  else if ((b0 & 0xF0) == 0xE0) { need = 3; cp = b0 & 0x0F; }
  else if ((b0 & 0xF8) == 0xF0) { need = 4; cp = b0 & 0x07; }
  else { *out_len = 1; return 0xFFFD; }

  if (pos + need > len) { *out_len = 1; return 0xFFFD; }
  for (size_t i = 1; i < need; ++i) {
    auto bi = static_cast<unsigned char>(data[pos + i]);
    if ((bi & 0xC0) != 0x80) { *out_len = 1; return 0xFFFD; }
    cp = (cp << 6) | (bi & 0x3F);
  }
  *out_len = need;
  return cp;
}

// Append the UTF-8 encoding of `cp` to `out`.
void
utf8_append_(string* out, uint32_t cp)
{
  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Longest prefix of `s` that ends on a complete UTF-8 codepoint.
// Used by the streaming decoder to hold back partial multi-byte
// sequences across token boundaries.
size_t
utf8_longest_valid_prefix_(string_view s)
{
  size_t i = 0;
  while (i < s.size()) {
    auto b0 = static_cast<unsigned char>(s[i]);
    size_t need = 0;
    if (b0 < 0x80)               { need = 1; }
    else if ((b0 & 0xE0) == 0xC0){ need = 2; }
    else if ((b0 & 0xF0) == 0xE0){ need = 3; }
    else if ((b0 & 0xF8) == 0xF0){ need = 4; }
    else { ++i; continue; }  // illegal leading byte: emit as-is
    if (i + need > s.size()) {
      // Incomplete trailing sequence -- hold it back.
      return i;
    }
    i += need;
  }
  return i;
}

// ---------------------------------------------------------------------
// Unicode property classifiers + Llama-3 pre-tokenizer scanner.
//
// std::regex does not support \p{L} / \p{N}, so tokenizer.json files
// that ship the Llama-3 (and Qwen-2.5) pre-tokenizer regex fail to
// compile and the encoder silently falls back to "treat the input as
// one chunk". That feeds the BPE incoherent token sequences. The
// scanner below mirrors the seven alternatives of that regex byte-
// faithfully for ASCII and applies coarse Unicode-block heuristics
// to non-ASCII codepoints.
//
//   regex:
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}{1,3}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// We pick the leftmost matching alternative at each position (the
// std::regex / ECMAScript alternation semantics).
// ---------------------------------------------------------------------

bool
is_letter_cp_(uint32_t cp)
{
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) {
    return true;
  }
  if (cp < 0x80) { return false; }
  // Best-effort Unicode coverage. False positives (treating a
  // symbol as a letter) are tolerable; the BPE encoder still
  // produces well-formed output via byte-level fallback. Excluding
  // a real letter would corrupt the model's input distribution
  // far worse.
  if (cp >= 0x00C0 && cp <= 0x02FF) {
    // Latin-1 supplement + Latin Extended A/B + IPA. Skip the
    // multiplication and division signs that share the range.
    return cp != 0x00D7 && cp != 0x00F7;
  }
  if (cp >= 0x0370 && cp <= 0x07FF) { return true; }    // Greek..Arabic
  if (cp >= 0x0900 && cp <= 0x0DFF) { return true; }    // Indic
  if (cp >= 0x0E00 && cp <= 0x0FFF) { return true; }    // Thai..Tibetan
  if (cp >= 0x1100 && cp <= 0x1FFF) { return true; }    // Hangul Jamo..
  if (cp >= 0x3040 && cp <= 0x30FF) { return true; }    // Hiragana/Katakana
  if (cp >= 0x3400 && cp <= 0x4DBF) { return true; }    // CJK Ext A
  if (cp >= 0x4E00 && cp <= 0x9FFF) { return true; }    // CJK Unified
  if (cp >= 0xAC00 && cp <= 0xD7AF) { return true; }    // Hangul Syllables
  if (cp >= 0xF900 && cp <= 0xFAFF) { return true; }    // CJK Compat
  return false;
}

bool
is_digit_cp_(uint32_t cp)
{
  if (cp >= '0' && cp <= '9') { return true; }
  if (cp >= 0x0660 && cp <= 0x0669) { return true; }   // Arabic-Indic
  if (cp >= 0x06F0 && cp <= 0x06F9) { return true; }   // Extended Arabic-Indic
  if (cp >= 0x0966 && cp <= 0x096F) { return true; }   // Devanagari
  return false;
}

// ASCII whitespace only. We deliberately avoid Unicode whitespace
// (NBSP, ideographic space, ...) because the alt-5 / alt-6 backtrack
// logic walks byte-by-byte and a multi-byte whitespace codepoint
// would land us in the middle of a UTF-8 sequence. Llama-3 prompts
// almost never carry non-ASCII whitespace.
bool
is_ws_(unsigned char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r'
      || c == 0x0B || c == 0x0C;
}

// One leftmost-alternative match starting at `pos`. Returns the byte
// length of the matched chunk, or 0 if no alternative matched
// (caller advances by one codepoint to make progress).
size_t
llama3_match_one_(const char* data, size_t len, size_t pos)
{
  // --- Alt 1: contractions (case-insensitive ASCII) -----------------
  if (pos < len && data[pos] == '\'') {
    static const struct { const char* s; size_t n; } k[] = {
      {"'re", 3}, {"'ve", 3}, {"'ll", 3}, {"'s", 2},
      {"'t", 2}, {"'m", 2},  {"'d", 2},
    };
    for (const auto& c : k) {
      if (pos + c.n > len) { continue; }
      bool match = true;
      for (size_t i = 0; i < c.n; ++i) {
        char a = data[pos + i];
        char b = c.s[i];
        if (a >= 'A' && a <= 'Z') { a = static_cast<char>(a + 32); }
        if (b >= 'A' && b <= 'Z') { b = static_cast<char>(b + 32); }
        if (a != b) { match = false; break; }
      }
      if (match) { return c.n; }
    }
  }

  // --- Alt 2: [^\r\n\p{L}\p{N}]?\p{L}+ ------------------------------
  {
    auto try_word_at = [&](size_t start) -> size_t {
      size_t cp_len;
      uint32_t cp = utf8_next_(data, len, start, &cp_len);
      if (!is_letter_cp_(cp)) { return 0; }
      size_t end = start + cp_len;
      while (end < len) {
        size_t cl;
        uint32_t c2 = utf8_next_(data, len, end, &cl);
        if (!is_letter_cp_(c2)) { break; }
        end += cl;
      }
      return end - start;
    };

    size_t lead = 0;
    if (pos < len) {
      size_t cl;
      uint32_t cp0 = utf8_next_(data, len, pos, &cl);
      const bool ok_lead = cp0 != '\r' && cp0 != '\n'
                         && !is_letter_cp_(cp0)
                         && !is_digit_cp_(cp0);
      if (ok_lead) { lead = cl; }
    }
    if (lead > 0) {
      size_t w = try_word_at(pos + lead);
      if (w > 0) { return lead + w; }
    }
    size_t w = try_word_at(pos);
    if (w > 0) { return w; }
  }

  // --- Alt 3: \p{N}{1,3} --------------------------------------------
  {
    size_t end = pos;
    int count = 0;
    while (count < 3 && end < len) {
      size_t cl;
      uint32_t cp = utf8_next_(data, len, end, &cl);
      if (!is_digit_cp_(cp)) { break; }
      end += cl;
      ++count;
    }
    if (count >= 1) { return end - pos; }
  }

  // --- Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]* -----------------------------
  {
    auto try_punct_run_at = [&](size_t start) -> size_t {
      size_t end = start;
      while (end < len) {
        size_t cl;
        uint32_t cp = utf8_next_(data, len, end, &cl);
        const bool ws = cp < 0x80
            ? is_ws_(static_cast<unsigned char>(cp)) : false;
        if (ws || is_letter_cp_(cp) || is_digit_cp_(cp)) { break; }
        end += cl;
      }
      return end - start;
    };
    size_t lead = (pos < len && data[pos] == ' ') ? 1 : 0;
    size_t run = try_punct_run_at(pos + lead);
    if (run > 0) {
      size_t end = pos + lead + run;
      while (end < len && (data[end] == '\r' || data[end] == '\n')) {
        ++end;
      }
      return end - pos;
    }
    if (lead > 0) {
      // Backtrack: drop the leading space, try without.
      run = try_punct_run_at(pos);
      if (run > 0) {
        size_t end = pos + run;
        while (end < len
               && (data[end] == '\r' || data[end] == '\n')) {
          ++end;
        }
        return end - pos;
      }
    }
  }

  // --- Alt 5: \s*[\r\n]+ --------------------------------------------
  {
    size_t wp = pos;
    while (wp < len && is_ws_(static_cast<unsigned char>(data[wp]))) {
      ++wp;
    }
    // Find the largest m in (pos, wp] where data[m-1] is \r or \n.
    size_t m = wp;
    while (m > pos && data[m - 1] != '\r' && data[m - 1] != '\n') {
      --m;
    }
    if (m > pos) { return m - pos; }
  }

  // --- Alt 6: \s+(?!\S) ---------------------------------------------
  {
    size_t wp = pos;
    while (wp < len && is_ws_(static_cast<unsigned char>(data[wp]))) {
      ++wp;
    }
    if (wp > pos) {
      // Greedy match; backtrack one whitespace char if the next
      // char is non-whitespace (i.e. EOF satisfies lookahead).
      if (wp >= len) {
        return wp - pos;
      }
      if (wp - 1 > pos) {
        return (wp - 1) - pos;
      }
      // Only one ws char left and the next is non-ws -- alt 6
      // requires >=1 chars in the match, so fall through.
    }
  }

  // --- Alt 7: \s+ ---------------------------------------------------
  {
    size_t wp = pos;
    while (wp < len && is_ws_(static_cast<unsigned char>(data[wp]))) {
      ++wp;
    }
    if (wp > pos) { return wp - pos; }
  }

  // No alternative matched: advance one codepoint to make progress.
  if (pos < len) {
    size_t cl;
    (void)utf8_next_(data, len, pos, &cl);
    return cl;
  }
  return 0;
}

// ---------------------------------------------------------------------
// Byte-level alphabet (the GPT-2 / HF byte_level mapping)
// ---------------------------------------------------------------------
//
// Each byte 0..255 is mapped to a unique unicode codepoint that is
// guaranteed printable (no control chars, no whitespace -- the
// pre-tokenizer's whitespace splitter can't accidentally re-split a
// byte-level encoded chunk). 188 bytes (the standard "printable
// punctuation, letters, digits, and Latin-1 supplement minus a few
// control-ish chars") map to themselves; the other 68 bytes map to
// U+0100..U+0143 in a fixed order. The inverse map is sparse over
// codepoint space and stored as an unordered_map.

struct ByteLevelMap {
  array<uint32_t, 256>           to_unicode{};
  unordered_map<uint32_t, uint8_t> from_unicode;

  ByteLevelMap()
  {
    // Bytes that map to themselves: 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF.
    array<bool, 256> self_mapped{};
    for (int b = 0x21; b <= 0x7E; ++b) { self_mapped[b] = true; }
    for (int b = 0xA1; b <= 0xAC; ++b) { self_mapped[b] = true; }
    for (int b = 0xAE; b <= 0xFF; ++b) { self_mapped[b] = true; }

    // Walk bytes 0..255 in order; the n-th "missing" byte gets
    // U+0100 + n. This matches HF's bytes_to_unicode() exactly.
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
      if (self_mapped[b]) {
        to_unicode[b] = static_cast<uint32_t>(b);
      } else {
        to_unicode[b] = 0x100u + n;
        ++n;
      }
    }
    for (int b = 0; b < 256; ++b) {
      from_unicode.emplace(to_unicode[b], static_cast<uint8_t>(b));
    }
  }
};

const ByteLevelMap&
byte_level_()
{
  static const ByteLevelMap m;
  return m;
}

// Encode raw bytes as a byte-level UTF-8 string (one mapped
// codepoint per input byte).
string
bytes_to_byte_level_(string_view raw)
{
  const auto& m = byte_level_();
  string out;
  out.reserve(raw.size() * 2);
  for (char c : raw) {
    utf8_append_(&out, m.to_unicode[static_cast<unsigned char>(c)]);
  }
  return out;
}

// Reverse the byte-level encoding: byte-level UTF-8 -> raw bytes.
// Unknown codepoints are dropped (rare; would indicate a vocab/byte
// mismatch).
string
byte_level_to_bytes_(string_view encoded)
{
  const auto& m = byte_level_();
  string out;
  out.reserve(encoded.size());
  size_t i = 0;
  while (i < encoded.size()) {
    size_t step = 0;
    uint32_t cp = utf8_next_(encoded.data(), encoded.size(), i, &step);
    auto it = m.from_unicode.find(cp);
    if (it != m.from_unicode.end()) {
      out.push_back(static_cast<char>(it->second));
    }
    i += step;
  }
  return out;
}

// Split a byte-level encoded string into one substring per UTF-8
// codepoint. Each substring is the unit of BPE merging.
vector<string>
split_codepoints_(string_view s)
{
  vector<string> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    size_t step = 0;
    (void)utf8_next_(s.data(), s.size(), i, &step);
    out.emplace_back(s.data() + i, step);
    i += step;
  }
  return out;
}

}

// ---------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------

class Tokenizer::Impl {
public:
  bool parse(const FlexData& root, string_view tag,
             const SessionContextIntf* session);
  bool load_gguf(const GgufFile& g, const SessionContextIntf* session);

  vector<int32_t> encode(string_view text) const;
  string          decode(span<const int32_t> ids) const;
  string          step(StreamDecoder& sd, int32_t id) const;
  int32_t         special_token_id(string_view name) const;

  int32_t     vocab_size() const noexcept { return _vocab_size; }
  bool        has_pre_tokenizer() const noexcept
  { return _use_llama3_pre || !_pre_regexes.empty(); }
  size_t      merge_count() const noexcept { return _merge_priority.size(); }
  size_t      special_count() const noexcept { return _special_by_name.size(); }

private:
  // (first, second) -> merge priority (lower wins).
  struct PairHash {
    size_t operator()(const pair<string, string>& p) const noexcept
    {
      auto h1 = hash<string>{}(p.first);
      auto h2 = hash<string>{}(p.second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  unordered_map<string, int32_t>                            _vocab;
  unordered_map<int32_t, string>                            _inv_vocab;
  unordered_map<pair<string, string>, int32_t, PairHash>    _merge_priority;
  unordered_map<string, int32_t>                            _special_by_name;
  unordered_map<int32_t, string>                            _special_by_id;
  vector<regex>                                             _pre_regexes;
  // When true, the model declares the Llama-3 / Qwen-2.5 family
  // pre-tokenizer pattern (the one with \p{L} / \p{N} that std::regex
  // can't compile). pre_tokenize_ dispatches to a hand-coded scanner
  // that walks the seven alternatives of that regex directly, with
  // best-effort Unicode property classification.
  bool                                                      _use_llama3_pre = false;
  int32_t                                                   _vocab_size = 0;

  // SentencePiece-style metaspace mode (Gemma): a `Replace " " -> "▁"`
  // normalizer + plain BPE, NOT GPT-2 ByteLevel. When set, encode
  // normalizes spaces to `_ms_marker`, BPEs on raw UTF-8 (skipping the
  // byte-level encoding that Llama/Qwen need), and ByteFallback-encodes
  // out-of-vocab pieces; decode reverses it. Distinguishes the two
  // tokenizer families that ship as HF tokenizer.json.
  bool                                                      _metaspace = false;
  string                                                    _ms_marker;

  // Family-specific reasoning begin/end token ids, detected once at
  // load (Qwen3 `<think>`/`</think>`, else Gemma-4
  // `<|channel>`/`<channel|>`; -1 when the vocab has neither). decode
  // and step rewrite these ids to the UNIFIED vpipe thinking markers
  // (media_line::kThinkStart/kThinkEnd) instead of their literal
  // content, so every consumer sees one marker pair regardless of the
  // model family.
  int32_t                                                   _think_open_id  = -1;
  int32_t                                                   _think_close_id = -1;

  void detect_thinking_markers_();

  void bpe_(vector<string>* pieces) const;
  string normalize_ms_(string_view text) const;
  string metaspace_decode_(const string& s) const;

  // Encode one pre-tokenized chunk: byte-level encode + BPE + vocab
  // lookup. Appends ids to `out`.
  void encode_chunk_(string_view chunk, vector<int32_t>* out) const;

  // Apply all pre-tokenizer regexes in sequence, producing the
  // final list of chunks. If no regex is configured, returns the
  // whole input as a single chunk.
  vector<string> pre_tokenize_(string_view text) const;
};

namespace {

// Walk a pre_tokenizer object (Sequence, Split, ByteLevel, ...) and
// collect every Split's regex pattern in document order.
void
collect_split_regexes_(const FlexData& node, vector<string>* out)
{
  if (!node.is_object()) { return; }
  auto obj = node.as_object();
  // FlexData::ObjectView::at() returns a temporary FlexData; any
  // string_view it yields dangles at the end of the expression.
  // Snapshot to a stable string.
  string type;
  if (obj.contains("type")) {
    type = string(obj.at("type").as_string(""));
  }
  if (type == "Sequence" && obj.contains("pretokenizers")) {
    auto arr_fd = obj.at("pretokenizers");
    if (arr_fd.is_array()) {
      auto arr = arr_fd.as_array();
      for (size_t i = 0; i < arr.size(); ++i) {
        collect_split_regexes_(arr.at(i), out);
      }
    }
    return;
  }
  if (type == "Split" && obj.contains("pattern")) {
    auto pat = obj.at("pattern");
    if (pat.is_object()) {
      auto po = pat.as_object();
      if (po.contains("Regex")) {
        out->emplace_back(po.at("Regex").as_string(""));
      }
    }
    return;
  }
  // ByteLevel, Whitespace, Metaspace, etc. -- ignored in v1. The
  // byte-level encoding is applied unconditionally by the encoder.
}

}

bool
Tokenizer::Impl::parse(const FlexData&            root,
                       string_view                tag,
                       const SessionContextIntf*  session)
{
  // Async log delegate renders fmt() lambdas on a worker thread, so
  // any string_view captured into a closure must reference memory
  // that outlives this call. Snapshot tag to a stable local string
  // and use that in every diagnostic below.
  const string tag_str(tag);

  if (!root.is_object()) {
    if (session) {
      session->warn(fmt("Tokenizer({}): root is not a JSON object",
                        tag_str));
    }
    return false;
  }
  auto rootobj = root.as_object();

  // ---- model.vocab + model.merges ----------------------------------
  if (!rootobj.contains("model")) {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): tokenizer.json has no `model` field",
          tag_str));
    }
    return false;
  }
  auto model_fd = rootobj.at("model");
  if (!model_fd.is_object()) {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): `model` is not an object", tag_str));
    }
    return false;
  }
  auto model = model_fd.as_object();
  // FlexData::ObjectView::at() returns a fresh FlexData by value;
  // string_view obtained from it dangles at the end of the
  // expression. Snapshot to string immediately.
  string model_type;
  if (model.contains("type")) {
    model_type = string(model.at("type").as_string(""));
  }
  if (model_type != "BPE") {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): unsupported model.type='{}' (v1 supports "
          "BPE only)", tag_str, model_type));
    }
    return false;
  }

  if (!model.contains("vocab")) {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): model.vocab is missing", tag_str));
    }
    return false;
  }
  auto vocab_fd = model.at("vocab");
  if (!vocab_fd.is_object()) {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): model.vocab is not an object", tag_str));
    }
    return false;
  }
  auto vocab = vocab_fd.as_object();
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    auto entry = *it;
    int32_t id = static_cast<int32_t>(entry.second.as_int(-1));
    if (id < 0) { continue; }
    string key(entry.first);
    _vocab.emplace(key, id);
    _inv_vocab.emplace(id, std::move(key));
    if (id + 1 > _vocab_size) { _vocab_size = id + 1; }
  }

  if (model.contains("merges")) {
    auto merges_fd = model.at("merges");
    if (merges_fd.is_array()) {
      auto merges = merges_fd.as_array();
      for (size_t i = 0; i < merges.size(); ++i) {
        auto m = merges.at(i);
        string first;
        string second;
        if (m.is_string()) {
          // Format: "a b" (single string with a space separator).
          string s(m.as_string(""));
          auto sp = s.find(' ');
          if (sp == string::npos) { continue; }
          first  = s.substr(0, sp);
          second = s.substr(sp + 1);
        } else if (m.is_array()) {
          auto arr = m.as_array();
          if (arr.size() < 2) { continue; }
          first  = string(arr.at(0).as_string(""));
          second = string(arr.at(1).as_string(""));
        } else {
          continue;
        }
        _merge_priority.emplace(
            pair<string, string>(std::move(first), std::move(second)),
            static_cast<int32_t>(i));
      }
    }
  }

  // ---- added_tokens (special tokens) -------------------------------
  if (rootobj.contains("added_tokens")) {
    auto added_fd = rootobj.at("added_tokens");
    if (added_fd.is_array()) {
      auto arr = added_fd.as_array();
      for (size_t i = 0; i < arr.size(); ++i) {
        auto t = arr.at(i);
        if (!t.is_object()) { continue; }
        auto to = t.as_object();
        if (!to.contains("id") || !to.contains("content")) { continue; }
        int32_t id = static_cast<int32_t>(to.at("id").as_int(-1));
        if (id < 0) { continue; }
        string content(to.at("content").as_string(""));
        _special_by_name.emplace(content, id);
        _special_by_id.emplace(id, std::move(content));
        if (id + 1 > _vocab_size) { _vocab_size = id + 1; }
      }
    }
  }

  // ---- normalizer: detect the SentencePiece metaspace scheme -------
  // A `Replace " " -> "<marker>"` normalizer (single or inside a
  // Sequence) means the model BPEs on normalized UTF-8 (marker for
  // space) instead of GPT-2 byte-level. Gemma uses marker U+2581 (▁).
  if (rootobj.contains("normalizer")) {
    auto try_replace = [&](const FlexData::ConstObjectView& no) {
      if (!no.contains("type")
          || string(no.at("type").as_string("")) != "Replace") {
        return;
      }
      string pat, content;
      if (no.contains("pattern")) {
        auto p = no.at("pattern");
        if (p.is_object() && p.as_object().contains("String")) {
          pat = string(p.as_object().at("String").as_string(""));
        }
      }
      if (no.contains("content")) {
        content = string(no.at("content").as_string(""));
      }
      if (pat == " " && !content.empty()) {
        _metaspace = true;
        _ms_marker = content;
      }
    };
    auto nrm = rootobj.at("normalizer");
    if (nrm.is_object()) {
      auto no = nrm.as_object();
      const string ty = no.contains("type")
          ? string(no.at("type").as_string("")) : "";
      if (ty == "Sequence" && no.contains("normalizers")
          && no.at("normalizers").is_array()) {
        for (FlexData s : no.at("normalizers").as_array()) {
          if (s.is_object()) { try_replace(s.as_object()); }
        }
      } else {
        try_replace(no);
      }
    }
    if (_metaspace && session) {
      session->info(fmt("Tokenizer({}): SentencePiece metaspace mode "
                        "(marker U+2581); byte-level encoding disabled",
                        tag_str));
    }
  }

  // ---- pre_tokenizer regex (optional) ------------------------------
  if (rootobj.contains("pre_tokenizer")) {
    auto pre = rootobj.at("pre_tokenizer");
    vector<string> patterns;
    collect_split_regexes_(pre, &patterns);
    for (const auto& p : patterns) {
      // Llama-3 / Qwen-2.5 family: \p{L}/\p{N} are not in
      // std::regex. Detect the pattern and hand the input to a
      // hand-coded scanner (see llama3_match_one_).
      if (p.find("\\p{L}") != string::npos
          && p.find("\\p{N}") != string::npos) {
        _use_llama3_pre = true;
        continue;
      }
      try {
        // ECMAScript syntax (the std::regex default).
        _pre_regexes.emplace_back(p);
      } catch (const exception& e) {
        if (session) {
          session->warn(fmt(
              "Tokenizer({}): pre_tokenizer regex compile failed for "
              "'{}': {}; falling back to no-pre-tokenize",
              tag_str, p, e.what()));
        }
      }
    }
  }

  detect_thinking_markers_();
  return !_vocab.empty();
}

bool
Tokenizer::Impl::load_gguf(const GgufFile& g,
                           const SessionContextIntf* session)
{
  vector<string> tokens = g.get_string_array("tokenizer.ggml.tokens");
  vector<string> merges = g.get_string_array("tokenizer.ggml.merges");
  vector<int64_t> types = g.get_int_array("tokenizer.ggml.token_type");
  if (tokens.empty()) {
    if (session) {
      session->warn(fmt(
          "Tokenizer::from_gguf: tokenizer.ggml.tokens missing/empty"));
    }
    return false;
  }
  // vocab + per-token specials. GGUF token types: 2=UNKNOWN, 3=CONTROL,
  // 4=USER_DEFINED are "special" (looked up by name, never BPE-split);
  // 1=NORMAL, 6=BYTE stay regular (byte tokens feed ByteFallback).
  for (int32_t id = 0; id < static_cast<int32_t>(tokens.size()); ++id) {
    const string& s = tokens[static_cast<size_t>(id)];
    _vocab.emplace(s, id);
    _inv_vocab.emplace(id, s);
    if (id + 1 > _vocab_size) { _vocab_size = id + 1; }
    if (id < static_cast<int32_t>(types.size())) {
      const int t = static_cast<int>(types[static_cast<size_t>(id)]);
      if (t == 2 || t == 3 || t == 4) {
        _special_by_name.emplace(s, id);
        _special_by_id.emplace(id, s);
      }
    }
  }
  // merges: "first second" -> priority (= array index, lower wins).
  for (size_t i = 0; i < merges.size(); ++i) {
    const string& s = merges[i];
    const auto sp = s.find(' ');
    if (sp == string::npos) { continue; }
    _merge_priority.emplace(
        pair<string, string>(s.substr(0, sp), s.substr(sp + 1)),
        static_cast<int32_t>(i));
  }
  // The GGUF tokenizer model gates encode + decode. "gpt2" = GPT-2 byte-level
  // BPE (Qwen, Llama-3): the Ġ/Ċ byte-level alphabet, encoded via
  // bytes_to_byte_level_ and decoded via byte_level_to_bytes_. "llama" (and
  // other SentencePiece, e.g. Gemma) = metaspace: normalize space -> U+2581 and
  // BPE on raw UTF-8 with ByteFallback, matching the HF fast tokenizer.
  // Defaulting every GGUF to metaspace mangled byte-level models -- the raw
  // Ġ (space) / Ċ (newline) alphabet chars leaked into the detokenized text.
  const std::string tk_model =
      g.get_string("tokenizer.ggml.model").value_or("");
  _metaspace = (tk_model != "gpt2");
  if (_metaspace) {
    _ms_marker = "\xE2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK
  } else {
    // GPT-2 byte-level pre-tokenization. Qwen / Llama-3 use the same
    // \p{L}/\p{N}-family split regex that from_json hands to the hand-coded
    // llama3_match_one_ scanner (std::regex can't express \p{L}). Gate on the
    // declared pre-tokenizer so an unrecognised byte-level pre falls back to
    // whole-chunk BPE rather than a mismatched scanner. Without this the
    // byte-level encode would BPE the whole turn as one chunk (wrong splits).
    const std::string pre = g.get_string("tokenizer.ggml.pre").value_or("");
    if (pre.find("qwen") != string::npos
        || pre.find("llama") != string::npos) {
      _use_llama3_pre = true;
    }
  }
  if (session) {
    session->info(fmt(
        "Tokenizer::from_gguf: {} tokens, {} merges, {} special",
        _vocab.size(), _merge_priority.size(), _special_by_name.size()));
  }
  detect_thinking_markers_();
  return !_vocab.empty();
}

void
Tokenizer::Impl::detect_thinking_markers_()
{
  auto sp = [this](const char* name) -> int32_t {
    auto it = _special_by_name.find(name);
    return it != _special_by_name.end() ? it->second : -1;
  };
  _think_open_id  = sp("<think>");
  _think_close_id = sp("</think>");
  if (_think_open_id < 0 && _think_close_id < 0) {
    // Gemma-4 reasoning channel. (`<|think|>` is Gemma's arm-reasoning
    // marker, not the block delimiter -- the channel tokens are.)
    _think_open_id  = sp("<|channel>");
    _think_close_id = sp("<channel|>");
  }
}

void
Tokenizer::Impl::bpe_(vector<string>* pieces) const
{
  if (pieces->size() < 2) { return; }
  while (true) {
    size_t best_idx = numeric_limits<size_t>::max();
    int32_t best_prio = numeric_limits<int32_t>::max();
    for (size_t i = 0; i + 1 < pieces->size(); ++i) {
      auto it = _merge_priority.find({(*pieces)[i], (*pieces)[i + 1]});
      if (it != _merge_priority.end() && it->second < best_prio) {
        best_prio = it->second;
        best_idx  = i;
      }
    }
    if (best_idx == numeric_limits<size_t>::max()) { break; }
    (*pieces)[best_idx] += (*pieces)[best_idx + 1];
    pieces->erase(pieces->begin() + static_cast<long>(best_idx) + 1);
  }
}

void
Tokenizer::Impl::encode_chunk_(string_view chunk, vector<int32_t>* out) const
{
  if (chunk.empty()) { return; }
  // Metaspace (Gemma): BPE on the raw UTF-8 (marker already substituted
  // for spaces); Llama/Qwen: GPT-2 byte-level encode first.
  auto pieces = _metaspace
      ? split_codepoints_(string(chunk))
      : split_codepoints_(bytes_to_byte_level_(chunk));
  bpe_(&pieces);
  for (const auto& p : pieces) {
    auto it = _vocab.find(p);
    if (it != _vocab.end()) {
      out->push_back(it->second);
      continue;
    }
    if (_metaspace) {
      // ByteFallback: emit one <0xNN> token per UTF-8 byte.
      for (unsigned char b : p) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        auto bit = _vocab.find(buf);
        if (bit != _vocab.end()) { out->push_back(bit->second); }
      }
    }
    // Non-metaspace miss: byte-level guarantees single-codepoint pieces
    // are in vocab, so a miss is a merges/vocab mismatch -- drop it.
  }
}

string
Tokenizer::Impl::normalize_ms_(string_view text) const
{
  string out;
  out.reserve(text.size());
  for (char c : text) {
    if (c == ' ') { out += _ms_marker; }
    else          { out += c; }
  }
  return out;
}

// Reverse the metaspace encoding for decode: marker -> space, and
// <0xNN> byte tokens -> the raw byte. Operates on the concatenated
// vocab strings of a run of regular ids.
string
Tokenizer::Impl::metaspace_decode_(const string& s) const
{
  string out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    if (!_ms_marker.empty()
        && s.compare(i, _ms_marker.size(), _ms_marker) == 0) {
      out += ' ';
      i += _ms_marker.size();
      continue;
    }
    // <0xNN> byte token.
    if (s[i] == '<' && i + 5 < s.size() && s[i + 1] == '0'
        && (s[i + 2] == 'x' || s[i + 2] == 'X') && s[i + 5] == '>') {
      auto hex = [](char h) -> int {
        if (h >= '0' && h <= '9') { return h - '0'; }
        if (h >= 'a' && h <= 'f') { return h - 'a' + 10; }
        if (h >= 'A' && h <= 'F') { return h - 'A' + 10; }
        return -1;
      };
      const int hi = hex(s[i + 3]), lo = hex(s[i + 4]);
      if (hi >= 0 && lo >= 0) {
        out += static_cast<char>((hi << 4) | lo);
        i += 6;
        continue;
      }
    }
    out += s[i++];
  }
  return out;
}

vector<string>
Tokenizer::Impl::pre_tokenize_(string_view text) const
{
  if (_use_llama3_pre) {
    // Hand-coded scanner for the Llama-3 / Qwen-2.5 pre-tokenizer
    // regex. Walks the seven alternatives at each position; emits
    // chunks in input order.
    vector<string> out;
    const char* data = text.data();
    const size_t len = text.size();
    size_t pos = 0;
    while (pos < len) {
      size_t n = llama3_match_one_(data, len, pos);
      if (n == 0) { break; }
      out.emplace_back(data + pos, n);
      pos += n;
    }
    return out;
  }
  if (_pre_regexes.empty()) {
    return { string(text) };
  }
  // For each regex in document order, scan the current chunk set
  // and replace each chunk with its in-order match list. With
  // Llama-style "Split + ByteLevel" pre_tokenizers, only the first
  // (Split) regex matters in v1; ByteLevel is handled by the
  // byte-level encoding inside encode_chunk_.
  vector<string> chunks = { string(text) };
  for (const auto& re : _pre_regexes) {
    vector<string> next;
    next.reserve(chunks.size());
    for (const auto& chunk : chunks) {
      auto begin = sregex_iterator(chunk.begin(), chunk.end(), re);
      auto end   = sregex_iterator();
      bool any = false;
      for (auto it = begin; it != end; ++it) {
        next.push_back(it->str());
        any = true;
      }
      if (!any) {
        // No matches: keep the chunk as-is so the BPE still has
        // something to chew on.
        next.push_back(chunk);
      }
    }
    chunks.swap(next);
  }
  return chunks;
}

vector<int32_t>
Tokenizer::Impl::encode(string_view text) const
{
  vector<int32_t> out;
  if (text.empty()) { return out; }
  // Metaspace: normalize spaces to the marker BEFORE pre-tokenization
  // (Gemma's `Split " "` pre-tokenizer then finds nothing to split, so
  // the normalized text BPEs as one chunk -- matching SentencePiece).
  const string norm = _metaspace ? normalize_ms_(text) : string();
  const string_view in = _metaspace ? string_view(norm) : text;
  auto chunks = pre_tokenize_(in);
  for (const auto& c : chunks) {
    encode_chunk_(c, &out);
  }
  return out;
}

string
Tokenizer::Impl::decode(span<const int32_t> ids) const
{
  string out;
  string byte_buf;  // accumulated byte-level chars across regular ids
  for (int32_t id : ids) {
    auto sit = _special_by_id.find(id);
    if (sit != _special_by_id.end()) {
      // Flush pending regular tokens, then emit the special's
      // literal content -- except the family's reasoning begin/end
      // tokens, which rewrite to the unified vpipe thinking markers.
      if (!byte_buf.empty()) {
        out.append(_metaspace ? metaspace_decode_(byte_buf)
                              : byte_level_to_bytes_(byte_buf));
        byte_buf.clear();
      }
      if (id == _think_open_id) {
        out.append(media_line::kThinkStart);
      } else if (id == _think_close_id) {
        out.append(media_line::kThinkEnd);
      } else {
        out.append(sit->second);
      }
      continue;
    }
    auto it = _inv_vocab.find(id);
    if (it == _inv_vocab.end()) { continue; }
    byte_buf.append(it->second);
  }
  if (!byte_buf.empty()) {
    out.append(_metaspace ? metaspace_decode_(byte_buf)
                          : byte_level_to_bytes_(byte_buf));
  }
  return out;
}

string
Tokenizer::Impl::step(StreamDecoder& sd, int32_t id) const
{
  // Append the decoded bytes (special: literal content, regular:
  // byte-level decode) to the pending buffer, then emit the longest
  // valid UTF-8 prefix. The family's reasoning begin/end tokens
  // rewrite to the unified vpipe thinking markers so streamed chat
  // output carries one marker pair regardless of the model family.
  auto sit = _special_by_id.find(id);
  if (sit != _special_by_id.end()) {
    if (id == _think_open_id) {
      sd.pending.append(media_line::kThinkStart);
    } else if (id == _think_close_id) {
      sd.pending.append(media_line::kThinkEnd);
    } else {
      sd.pending.append(sit->second);
    }
  } else {
    auto it = _inv_vocab.find(id);
    if (it != _inv_vocab.end()) {
      sd.pending.append(_metaspace ? metaspace_decode_(it->second)
                                   : byte_level_to_bytes_(it->second));
    }
  }
  size_t n = utf8_longest_valid_prefix_(sd.pending);
  string out(sd.pending, 0, n);
  sd.pending.erase(0, n);
  return out;
}

int32_t
Tokenizer::Impl::special_token_id(string_view name) const
{
  auto it = _special_by_name.find(string(name));
  return it != _special_by_name.end() ? it->second : -1;
}

// ---------------------------------------------------------------------
// Tokenizer (public API forwarding to Impl)
// ---------------------------------------------------------------------

Tokenizer::Tokenizer() : _impl(make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

unique_ptr<Tokenizer>
Tokenizer::from_huggingface_json(string_view               path,
                                 const SessionContextIntf* session)
{
  const string path_str(path);
  ifstream in(path_str);
  if (!in) {
    if (session) {
      session->warn(fmt(
          "Tokenizer::from_huggingface_json('{}'): file not readable",
          path_str));
    }
    return nullptr;
  }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (const exception& e) {
    if (session) {
      session->warn(fmt(
          "Tokenizer::from_huggingface_json('{}'): JSON parse failed: "
          "{}", path_str, e.what()));
    }
    return nullptr;
  }
  unique_ptr<Tokenizer> tok(new Tokenizer);
  if (!tok->_impl->parse(fd, path_str, session)) {
    return nullptr;
  }
  return tok;
}

unique_ptr<Tokenizer>
Tokenizer::from_huggingface_string(string_view               json,
                                   string_view               tag,
                                   const SessionContextIntf* session)
{
  const string tag_str(tag);
  FlexData fd;
  try {
    fd = FlexData::from_json(json);
  } catch (const exception& e) {
    if (session) {
      session->warn(fmt(
          "Tokenizer::from_huggingface_string('{}'): JSON parse "
          "failed: {}", tag_str, e.what()));
    }
    return nullptr;
  }
  unique_ptr<Tokenizer> tok(new Tokenizer);
  if (!tok->_impl->parse(fd, tag_str, session)) {
    return nullptr;
  }
  return tok;
}

unique_ptr<Tokenizer>
Tokenizer::from_gguf(const GgufFile& gguf, const SessionContextIntf* session)
{
  unique_ptr<Tokenizer> tok(new Tokenizer);
  if (!tok->_impl->load_gguf(gguf, session)) {
    return nullptr;
  }
  return tok;
}

vector<int32_t>
Tokenizer::encode(string_view text) const
{
  return _impl->encode(text);
}

string
Tokenizer::decode(span<const int32_t> ids) const
{
  return _impl->decode(ids);
}

Tokenizer::StreamDecoder
Tokenizer::make_stream_decoder() const
{
  return {};
}

string
Tokenizer::step(StreamDecoder& sd, int32_t id) const
{
  return _impl->step(sd, id);
}

int32_t
Tokenizer::special_token_id(string_view name) const
{
  return _impl->special_token_id(name);
}

int32_t
Tokenizer::vocab_size() const noexcept
{
  return _impl->vocab_size();
}

bool
Tokenizer::has_pre_tokenizer() const noexcept
{
  return _impl->has_pre_tokenizer();
}

size_t
Tokenizer::merge_count() const noexcept
{
  return _impl->merge_count();
}

size_t
Tokenizer::special_token_count() const noexcept
{
  return _impl->special_count();
}

}
