#ifndef VPIPE_COMMON_TEXT_STREAM_CHUNK_H
#define VPIPE_COMMON_TEXT_STREAM_CHUNK_H

// Progressive text chunking for streaming a language-model reply as
// speakable units. A producer (text-chat's `stream` oport) accumulates
// decoded token text and flushes a beat roughly every ~N words, cut at a
// sentence/clause punctuation, so a downstream consumer (text-to-speech)
// can begin work before the full reply is decoded.
//
// Word counting and punctuation both handle English (space-delimited,
// one word per Latin run) and Chinese (one word per CJK character, plus
// the full-width punctuation set). Header-only + inline so it is trivially
// unit-testable and adds no translation unit.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace vpipe::text_stream {

// Decode one UTF-8 codepoint at s[i], advancing i past it. A malformed
// lead byte or truncated sequence yields U+FFFD and consumes one byte,
// so the scan always makes progress.
inline std::uint32_t
next_codepoint(std::string_view s, std::size_t& i)
{
  const unsigned char c = static_cast<unsigned char>(s[i]);
  std::uint32_t cp;
  int           n;
  if (c < 0x80)              { cp = c;        n = 1; }
  else if ((c >> 5) == 0x6)  { cp = c & 0x1F; n = 2; }
  else if ((c >> 4) == 0xE)  { cp = c & 0x0F; n = 3; }
  else if ((c >> 3) == 0x1E) { cp = c & 0x07; n = 4; }
  else { ++i; return 0xFFFD; }
  if (i + static_cast<std::size_t>(n) > s.size()) { ++i; return 0xFFFD; }
  for (int k = 1; k < n; ++k) {
    const unsigned char cc = static_cast<unsigned char>(s[i + k]);
    if ((cc >> 6) != 0x2) { ++i; return 0xFFFD; }
    cp = (cp << 6) | (cc & 0x3F);
  }
  i += static_cast<std::size_t>(n);
  return cp;
}

// CJK ideographs + Japanese kana: each such codepoint counts as one
// "word" (there are no spaces between them).
inline bool
is_cjk(std::uint32_t cp)
{
  return (cp >= 0x3040 && cp <= 0x30FF)      // Hiragana + Katakana
      || (cp >= 0x3400 && cp <= 0x4DBF)      // CJK Ext-A
      || (cp >= 0x4E00 && cp <= 0x9FFF)      // CJK Unified Ideographs
      || (cp >= 0xF900 && cp <= 0xFAFF)      // CJK Compatibility
      || (cp >= 0x20000 && cp <= 0x2A6DF);   // CJK Ext-B
}

// Sentence / clause punctuation a chunk may end on -- English ASCII and
// the Chinese full-width equivalents.
inline bool
is_break_punct(std::uint32_t cp)
{
  switch (cp) {
    case '.': case ',': case '!': case '?': case ';': case ':':
    case 0x3002:  // U+3002 。 ideographic full stop
    case 0xFF0E:  // U+FF0E ．fullwidth full stop
    case 0xFF0C:  // U+FF0C ，fullwidth comma
    case 0x3001:  // U+3001 、ideographic comma
    case 0xFF01:  // U+FF01 ！fullwidth exclamation mark
    case 0xFF1F:  // U+FF1F ？fullwidth question mark
    case 0xFF1B:  // U+FF1B ；fullwidth semicolon
    case 0xFF1A:  // U+FF1A ：fullwidth colon
    case 0x2026:  // U+2026 … horizontal ellipsis
      return true;
    default:
      return false;
  }
}

inline bool
is_space(std::uint32_t cp)
{
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'
      || cp == 0x3000;   // ideographic space
}

// A word-forming codepoint: ASCII alphanumeric / apostrophe, or any
// non-CJK, non-space, non-break-punct codepoint >= 0x80 (accented
// Latin, Cyrillic, ...). CJK ideographs are counted separately, one per
// character.
inline bool
is_word_cp(std::uint32_t cp)
{
  if (cp < 0x80) {
    return (cp >= '0' && cp <= '9') || (cp >= 'A' && cp <= 'Z')
        || (cp >= 'a' && cp <= 'z') || cp == '\'';
  }
  return !is_cjk(cp) && !is_space(cp) && !is_break_punct(cp);
}

// Count words in `s`: each CJK codepoint is one word; a maximal run of
// Latin word codepoints is one word.
inline int
count_words(std::string_view s)
{
  int  words  = 0;
  bool in_run = false;
  for (std::size_t i = 0; i < s.size(); ) {
    const std::uint32_t cp = next_codepoint(s, i);
    if (is_cjk(cp)) {
      ++words;
      in_run = false;
    } else if (is_word_cp(cp)) {
      if (!in_run) { ++words; in_run = true; }
    } else {
      in_run = false;
    }
  }
  return words;
}

// True iff the last non-space codepoint of `s` is a break punctuation.
inline bool
ends_at_break(std::string_view s)
{
  std::uint32_t last = 0;
  for (std::size_t i = 0; i < s.size(); ) {
    const std::uint32_t cp = next_codepoint(s, i);
    if (!is_space(cp)) { last = cp; }
  }
  return last != 0 && is_break_punct(last);
}

// Accumulates decoded text chunks and reports when the pending buffer
// has reached a flush boundary: at least `word_target` words AND ending
// at a sentence/clause punctuation, OR a hard cap of `word_max` words
// (bounds first-audio latency on a runaway clause that never punctuates).
class Chunker {
public:
  explicit Chunker(int word_target = 20, int word_max = 60)
      : _word_target(word_target), _word_max(word_max)
  {
  }

  // Append a decoded chunk; returns true iff a flush boundary is now
  // reached (call take() to consume the pending text).
  bool push(std::string_view chunk)
  {
    _buf.append(chunk);
    return ready();
  }

  bool ready() const
  {
    const int w = count_words(_buf);
    return (w >= _word_target && ends_at_break(_buf)) || w >= _word_max;
  }

  bool empty() const { return _buf.empty(); }

  // Move out the pending text, clearing the buffer.
  std::string take()
  {
    std::string out = std::move(_buf);
    _buf.clear();
    return out;
  }

private:
  std::string _buf;
  int         _word_target;
  int         _word_max;
};

// Folds model-output META spans out of a streamed reply so a consumer that
// only speaks/reads the answer (e.g. text-to-speech) never sees the
// reasoning or tool-call text. Fed the same per-token chunks as the live
// stream, it forwards to a caller `sink` (any callable
// `void(const std::string&)`) ONLY the text OUTSIDE a reasoning block
// (`think_open`..`think_close`) or a tool-call block
// (`tool_open`..`tool_close`); suppressed text is dropped. Any marker may
// straddle a chunk boundary, so a partial-marker tail is held back until it
// completes or flush() runs. An unterminated block runs to end-of-message
// and its text is dropped (mirrors how front ends fold reasoning). An empty
// marker disarms that span kind. Blocks are treated as non-nesting (a
// reasoning block and a tool-call block never overlap in the wire format).
class MetaFilter {
public:
  MetaFilter(std::string_view think_open, std::string_view think_close,
             std::string_view tool_open,  std::string_view tool_close)
      : _think_open(think_open), _think_close(think_close),
        _tool_open(tool_open),   _tool_close(tool_close)
  {
  }

  // Append a chunk; forward the now-decidable visible text to `sink`.
  template <class Sink>
  void feed(std::string_view chunk, const Sink& sink)
  {
    _buf.append(chunk);
    scan(sink, /*final_flush=*/false);
  }

  // End of message: flush any held-back visible tail; drop an unterminated
  // suppressed block.
  template <class Sink>
  void flush(const Sink& sink)
  {
    scan(sink, /*final_flush=*/true);
  }

private:
  enum class State { Visible, Thinking, Tool };

  // Longest suffix of `buf` that is a proper prefix of `marker`: the bytes
  // to hold back so a marker split across chunks is not missed. 0 when
  // `marker` is empty or no suffix matches.
  static std::size_t
  hold_len(const std::string& buf, std::string_view marker)
  {
    if (marker.empty()) { return 0; }
    const std::size_t maxn = std::min(buf.size(), marker.size() - 1);
    for (std::size_t n = maxn; n > 0; --n) {
      if (std::string_view(buf).substr(buf.size() - n)
          == marker.substr(0, n)) {
        return n;
      }
    }
    return 0;
  }

  // Take `marker` (when armed) if it occurs earlier than the current best.
  void
  pick(std::string_view marker, State target, std::size_t& pos,
       std::size_t& len, State& to) const
  {
    if (marker.empty()) { return; }
    const std::size_t p = _buf.find(marker);
    if (p != std::string::npos && p < pos) {
      pos = p;
      len = marker.size();
      to  = target;
    }
  }

  template <class Sink>
  void scan(const Sink& sink, bool final_flush)
  {
    for (;;) {
      if (_state == State::Visible) {
        // Earliest of the two enter markers wins.
        std::size_t pos = std::string::npos;
        std::size_t len = 0;
        State       to  = State::Visible;
        pick(_think_open, State::Thinking, pos, len, to);
        pick(_tool_open,  State::Tool,     pos, len, to);
        if (to != State::Visible) {
          if (pos > 0) { sink(_buf.substr(0, pos)); }
          _buf.erase(0, pos + len);
          _state = to;
          continue;
        }
        // No enter marker: emit all but a possible partial-marker tail.
        const std::size_t hold = final_flush
            ? 0
            : std::max(hold_len(_buf, _think_open),
                       hold_len(_buf, _tool_open));
        if (_buf.size() > hold) {
          sink(_buf.substr(0, _buf.size() - hold));
          _buf.erase(0, _buf.size() - hold);
        }
        if (final_flush) { _buf.clear(); }
        return;
      }
      // Suppressed (Thinking / Tool): drop text up to and including the
      // matching close marker; hold back a possible partial tail.
      const std::string_view close =
          (_state == State::Thinking) ? _think_close : _tool_close;
      const std::size_t p =
          close.empty() ? std::string::npos : _buf.find(close);
      if (p != std::string::npos) {
        _buf.erase(0, p + close.size());
        _state = State::Visible;
        continue;
      }
      const std::size_t hold = final_flush ? 0 : hold_len(_buf, close);
      if (_buf.size() > hold) { _buf.erase(0, _buf.size() - hold); }
      if (final_flush) { _buf.clear(); }
      return;
    }
  }

  std::string      _buf;
  State            _state = State::Visible;
  std::string_view _think_open;
  std::string_view _think_close;
  std::string_view _tool_open;
  std::string_view _tool_close;
};

}  // namespace vpipe::text_stream

#endif
