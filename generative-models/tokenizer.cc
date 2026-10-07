#include "generative-models/tokenizer.h"

#include "generative-models/shared/gguf-file.h"
#include "common/flex-data.h"
#include "common/media-line.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <regex>
#include <span>
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
llama3_match_one_(const char* data, size_t len, size_t pos,
                  int max_digits)
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

  // --- Alt 3: \p{N}{1,3} (Llama-3), or \p{N} (Qwen) ---------------
  {
    size_t end = pos;
    int count = 0;
    while (count < max_digits && end < len) {
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
// Tekken pre-tokenizer scanner (Mistral: Ministral 3, Pixtral, Mistral
// Small 3.x). The Split pattern is GPT-4o's minus the contractions, with
// single digits and a trailing [\r\n/]*:
//
//     [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*
//                       [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+
//                       [\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n/]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// It splits letter runs on CASE ("HelloWorld" -> "Hello" "World"), so
// the coarse is_letter_cp_ above cannot serve it. The classes come from
// CoreFoundation's Unicode tables, plus the Nl/No numbers (superscripts,
// fractions, roman numerals...) it has no set for.
// ---------------------------------------------------------------------

enum class UCls : uint8_t {
  Other,         // punctuation, symbols, controls
  Upper,         // Lu, Lt
  Lower,         // Ll
  LetterOther,   // Lm, Lo -- in BOTH case classes of the pattern
  Mark,          // M* -- in both case classes, but NOT \p{L}
  Number,        // N*
  Space,         // White_Space
};

// General category Nl + No (Unicode 16), which CoreFoundation's
// decimal-digit set (Nd only) leaves out.
constexpr uint32_t kNumberLetterOther[][2] = {
  {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE}, {0x09F4, 0x09F9},
  {0x0B72, 0x0B77}, {0x0BF0, 0x0BF2}, {0x0C78, 0x0C7E}, {0x0D58, 0x0D5E},
  {0x0D70, 0x0D78}, {0x0F2A, 0x0F33}, {0x1369, 0x137C}, {0x16EE, 0x16F0},
  {0x17F0, 0x17F9}, {0x19DA, 0x19DA}, {0x2070, 0x2070}, {0x2074, 0x2079},
  {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189}, {0x2460, 0x249B},
  {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x2CFD, 0x2CFD}, {0x3007, 0x3007},
  {0x3021, 0x3029}, {0x3038, 0x303A}, {0x3192, 0x3195}, {0x3220, 0x3229},
  {0x3248, 0x324F}, {0x3251, 0x325F}, {0x3280, 0x3289}, {0x32B1, 0x32BF},
  {0xA6E6, 0xA6EF}, {0xA830, 0xA835}, {0x10107, 0x10133},
  {0x10140, 0x10178}, {0x1018A, 0x1018B}, {0x102E1, 0x102FB},
  {0x10320, 0x10323}, {0x10341, 0x10341}, {0x1034A, 0x1034A},
  {0x103D1, 0x103D5}, {0x10858, 0x1085F}, {0x10879, 0x1087F},
  {0x108A7, 0x108AF}, {0x108FB, 0x108FF}, {0x10916, 0x1091B},
  {0x109BC, 0x109BD}, {0x109C0, 0x109CF}, {0x109D2, 0x109FF},
  {0x10A40, 0x10A48}, {0x10A7D, 0x10A7E}, {0x10A9D, 0x10A9F},
  {0x10AEB, 0x10AEF}, {0x10B58, 0x10B5F}, {0x10B78, 0x10B7F},
  {0x10BA9, 0x10BAF}, {0x10CFA, 0x10CFF}, {0x10E60, 0x10E7E},
  {0x10F1D, 0x10F26}, {0x10F51, 0x10F54}, {0x10FC5, 0x10FCB},
  {0x11052, 0x11065}, {0x111E1, 0x111F4}, {0x1173A, 0x1173B},
  {0x118EA, 0x118F2}, {0x11C5A, 0x11C6C}, {0x11FC0, 0x11FD4},
  {0x12400, 0x1246E}, {0x16B5B, 0x16B61}, {0x16E80, 0x16E96},
  {0x1D2C0, 0x1D2D3}, {0x1D2E0, 0x1D2F3}, {0x1D360, 0x1D378},
  {0x1E8C7, 0x1E8CF}, {0x1EC71, 0x1ECAB}, {0x1ECAD, 0x1ECAF},
  {0x1ECB1, 0x1ECB4}, {0x1ED01, 0x1ED2D}, {0x1ED2F, 0x1ED3D},
  {0x1F100, 0x1F10C},
};

UCls
uni_class_(uint32_t cp)
{
  if (cp < 0x80) {
    if (cp >= 'a' && cp <= 'z') { return UCls::Lower; }
    if (cp >= 'A' && cp <= 'Z') { return UCls::Upper; }
    if (cp >= '0' && cp <= '9') { return UCls::Number; }
    if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) { return UCls::Space; }
    return UCls::Other;
  }
  // The predefined sets are immutable singletons owned by CF.
  static const CFCharacterSetRef ws =
      CFCharacterSetGetPredefined(kCFCharacterSetWhitespaceAndNewline);
  static const CFCharacterSetRef upper =
      CFCharacterSetGetPredefined(kCFCharacterSetUppercaseLetter);
  static const CFCharacterSetRef lower =
      CFCharacterSetGetPredefined(kCFCharacterSetLowercaseLetter);
  static const CFCharacterSetRef letter =
      CFCharacterSetGetPredefined(kCFCharacterSetLetter);   // L* + M*
  static const CFCharacterSetRef mark =
      CFCharacterSetGetPredefined(kCFCharacterSetNonBase);  // M*
  static const CFCharacterSetRef digit =
      CFCharacterSetGetPredefined(kCFCharacterSetDecimalDigit);
  const auto c = static_cast<UTF32Char>(cp);
  if (CFCharacterSetIsLongCharacterMember(ws, c))     { return UCls::Space; }
  if (CFCharacterSetIsLongCharacterMember(upper, c))  { return UCls::Upper; }
  if (CFCharacterSetIsLongCharacterMember(lower, c))  { return UCls::Lower; }
  if (CFCharacterSetIsLongCharacterMember(mark, c))   { return UCls::Mark; }
  if (CFCharacterSetIsLongCharacterMember(letter, c)) {
    return UCls::LetterOther;
  }
  if (CFCharacterSetIsLongCharacterMember(digit, c))  { return UCls::Number; }
  for (const auto& r : kNumberLetterOther) {
    if (cp < r[0]) { break; }
    if (cp <= r[1]) { return UCls::Number; }
  }
  return UCls::Other;
}

// One leftmost-alternative match of the Tekken pattern at `pos`; the
// byte length of the chunk, never 0 while pos < len.
size_t
tekken_match_one_(const char* data, size_t len, size_t pos)
{
  auto cls_at = [&](size_t p, size_t* cl) {
    const uint32_t cp = utf8_next_(data, len, p, cl);
    return std::make_pair(cp, uni_class_(cp));
  };
  auto in_u = [](UCls c) {
    return c == UCls::Upper || c == UCls::LetterOther || c == UCls::Mark;
  };
  auto in_l = [](UCls c) {
    return c == UCls::Lower || c == UCls::LetterOther || c == UCls::Mark;
  };
  auto is_l = [](UCls c) {   // \p{L}
    return c == UCls::Upper || c == UCls::Lower || c == UCls::LetterOther;
  };
  // End of the maximal run of `pred` codepoints from `p`.
  auto run = [&](size_t p, auto pred) {
    while (p < len) {
      size_t cl;
      if (!pred(cls_at(p, &cl).second)) { break; }
      p += cl;
    }
    return p;
  };

  // Optional lead [^\r\n\p{L}\p{N}] (greedy: tried consumed first).
  size_t lead = 0;
  {
    size_t cl;
    const auto [cp, c] = cls_at(pos, &cl);
    if (cp != '\r' && cp != '\n' && !is_l(c) && c != UCls::Number) {
      lead = cl;
    }
  }

  // --- Alt 1: lead? U* L+ -------------------------------------------
  // U* is greedy and gives codepoints back until L+ can start; only a
  // codepoint in both classes (Lm/Lo/M) can start it inside the U run.
  auto alt1 = [&](size_t s) -> size_t {
    const size_t u_end = run(s, in_u);
    size_t k = u_end;
    while (true) {
      const size_t l_end = run(k, in_l);
      if (l_end > k) { return l_end; }
      if (k == s) { return 0; }
      // Step back one codepoint (to the start of the previous one).
      do { --k; } while (k > s && (data[k] & 0xC0) == 0x80);
    }
  };
  // --- Alt 2: lead? U+ L* -------------------------------------------
  auto alt2 = [&](size_t s) -> size_t {
    const size_t u_end = run(s, in_u);
    return u_end > s ? run(u_end, in_l) : 0;
  };
  if (lead > 0) {
    if (size_t e = alt1(pos + lead)) { return e - pos; }
  }
  if (size_t e = alt1(pos)) { return e - pos; }
  if (lead > 0) {
    if (size_t e = alt2(pos + lead)) { return e - pos; }
  }
  if (size_t e = alt2(pos)) { return e - pos; }

  // --- Alt 3: \p{N} (one codepoint) ----------------------------------
  {
    size_t cl;
    if (cls_at(pos, &cl).second == UCls::Number) { return cl; }
  }

  // --- Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]* ------------------------------
  {
    auto punct = [&](UCls c) {
      return c != UCls::Space && !is_l(c) && c != UCls::Number;
    };
    auto tail = [&](size_t e) {
      while (e < len && (data[e] == '\r' || data[e] == '\n'
                         || data[e] == '/')) {
        ++e;
      }
      return e;
    };
    const size_t sp = (data[pos] == ' ') ? 1 : 0;
    size_t e = run(pos + sp, punct);
    if (e > pos + sp) { return tail(e) - pos; }
    if (sp > 0) {
      e = run(pos, punct);
      if (e > pos) { return tail(e) - pos; }
    }
  }

  // Whitespace run from pos (codepoints, Unicode White_Space).
  const size_t wp = run(pos, [](UCls c) { return c == UCls::Space; });
  if (wp > pos) {
    // --- Alt 5: \s*[\r\n]+ -- up to the LAST newline in the run.
    // ASCII bytes never occur inside a multi-byte sequence, so a byte
    // scan backwards finds it.
    for (size_t m = wp; m > pos; --m) {
      if (data[m - 1] == '\r' || data[m - 1] == '\n') { return m - pos; }
    }
    // --- Alt 6: \s+(?!\S) -- all of it at the end of the text, else
    // all but the last codepoint (which then precedes the \S).
    if (wp >= len) { return wp - pos; }
    size_t last = wp - 1;
    while (last > pos && (data[last] & 0xC0) == 0x80) { --last; }
    if (last > pos) { return last - pos; }
    // --- Alt 7: \s+
    return wp - pos;
  }

  // No alternative matched: advance one codepoint to make progress.
  size_t cl;
  (void)utf8_next_(data, len, pos, &cl);
  return cl;
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
  bool load_tiktoken(string_view path,
                     span<const string> specials,
                     const SessionContextIntf* session);

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
  // Mistral's Tekken split pattern (case-aware letter runs, single
  // digits, no contractions); pre_tokenize_ hands it to
  // tekken_match_one_. Takes precedence over _use_llama3_pre.
  bool                                                      _use_tekken_pre = false;
  // BPE `ignore_merges` (Tekken sets it): a pre-token whose byte-level
  // spelling is itself a vocab entry is emitted as that one token,
  // without running the merges -- which need not reach it.
  bool                                                      _ignore_merges = false;
  // Longest digit run the scanner keeps in one pre-token: the Llama-3
  // pattern spells \p{N}{1,3}, Qwen's spells a bare \p{N}. Left at 3
  // for tokenizer.json (whose Qwen vocabularies carry no digit merges,
  // so the two spellings agree there); a tiktoken rank file sets 1.
  int                                                       _pre_digit_run = 3;
  // tiktoken rank-file BPE (qwen.tiktoken): there is no merges list.
  // The pair to merge is the adjacent pair whose CONCATENATION has the
  // lowest rank, and a token's rank is its id -- so the priority is a
  // vocab lookup and no merges table is built. Ids at or above
  // `_n_ordinary` are specials and never take part.
  bool                                                      _rank_merges = false;
  int32_t                                                   _n_ordinary = 0;
  int32_t                                                   _vocab_size = 0;

  // SentencePiece-style metaspace mode (Gemma): a `Replace " " -> "▁"`
  // normalizer + plain BPE, NOT GPT-2 ByteLevel. When set, encode
  // normalizes spaces to `_ms_marker`, BPEs on raw UTF-8 (skipping the
  // byte-level encoding that Llama/Qwen need), and ByteFallback-encodes
  // out-of-vocab pieces; decode reverses it. Distinguishes the two
  // tokenizer families that ship as HF tokenizer.json.
  bool                                                      _metaspace = false;
  string                                                    _ms_marker;

  // ---- SentencePiece Unigram (T5 / umT5, i.e. every T5-conditioned
  // diffusion model) ------------------------------------------------
  // A different MODEL from BPE, not a different pre-tokenizer: where BPE
  // merges greedily by rank, Unigram scores every token with a log
  // probability and picks the segmentation that MAXIMIZES the total --
  // a Viterbi over the sentence, not a merge loop. The two disagree on
  // real text, so this cannot be approximated with the BPE path.
  bool                                                      _unigram = false;
  vector<float>                                             _uni_score;   // by id
  int32_t                                                   _unk_id = -1;
  float                                                     _unk_score = 0.0f;
  size_t                                                    _uni_max_piece = 0;
  // Metaspace PRE-TOKENIZER settings (the BPE families take metaspace
  // from the normalizer instead; see _metaspace above).
  bool                                                      _ms_prepend = false;
  bool                                                      _ms_split   = false;
  // Normalizer Replace rules, applied in document order before
  // anything else. Collected only for Unigram: the BPE families' one
  // interesting Replace (" " -> marker) is already folded into
  // normalize_ms_, and re-applying it here would double-substitute.
  vector<pair<regex, string>>                               _norm_replaces;

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
  // Unigram: the maximum-score segmentation of one pre-token, by
  // Viterbi over codepoint boundaries. Appends ids to `out`.
  void viterbi_(string_view seg, vector<int32_t>* out) const;
  vector<int32_t> encode_unigram_(string_view text) const;
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
  if (model_type != "BPE" && model_type != "Unigram") {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): unsupported model.type='{}' (BPE and Unigram "
          "are supported)", tag_str, model_type));
    }
    return false;
  }
  _unigram = (model_type == "Unigram");
  if (model.contains("ignore_merges")) {
    _ignore_merges = model.at("ignore_merges").as_bool(false);
  }

  if (!model.contains("vocab")) {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): model.vocab is missing", tag_str));
    }
    return false;
  }
  auto vocab_fd = model.at("vocab");
  // Unigram's vocab is an ARRAY of [piece, log_prob] and the id is the
  // POSITION -- not the {piece: id} object BPE uses. The scores are the
  // whole point of the model, so they are read here rather than
  // reconstructed.
  if (_unigram) {
    if (!vocab_fd.is_array()) {
      if (session) {
        session->warn(fmt(
            "Tokenizer({}): Unigram model.vocab is not an array", tag_str));
      }
      return false;
    }
    auto arr = vocab_fd.as_array();
    _uni_score.resize(arr.size(), 0.0f);
    float min_score = 0.0f;
    bool have_min = false;
    for (size_t i = 0; i < arr.size(); ++i) {
      auto e = arr.at(i);
      if (!e.is_array()) { continue; }
      auto pair_v = e.as_array();
      if (pair_v.size() < 2) { continue; }
      string piece(pair_v.at(0).as_string(""));
      const float score = (float)pair_v.at(1).as_real(0.0);
      const int32_t id = (int32_t)i;
      _uni_score[i] = score;
      if (!have_min || score < min_score) { min_score = score; have_min = true; }
      if (piece.size() > _uni_max_piece) { _uni_max_piece = piece.size(); }
      _vocab.emplace(piece, id);
      _inv_vocab.emplace(id, std::move(piece));
      if (id + 1 > _vocab_size) { _vocab_size = id + 1; }
    }
    if (model.contains("unk_id")) {
      _unk_id = (int32_t)model.at("unk_id").as_int(-1);
    }
    // The penalty an unknown CHARACTER pays, which is what makes the
    // Viterbi prefer any real segmentation over falling back. 10.0 below
    // the worst token in the vocab is the reference's constant
    // (tokenizers' K_UNK_PENALTY); a different value would silently
    // change segmentations near the tail of the vocab.
    _unk_score = min_score - 10.0f;
    if (session) {
      session->info(fmt(
          "Tokenizer({}): SentencePiece Unigram, {} pieces (unk id {})",
          tag_str, _uni_score.size(), _unk_id));
    }
  } else if (!vocab_fd.is_object()) {
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): model.vocab is not an object", tag_str));
    }
    return false;
  }
  if (!_unigram) {
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
      if (ty == "Sequence" && no.contains("normalizers")) {
        // The array has to be bound to a NAMED local: at() returns a
        // FlexData by value and as_array() is a view into it, so
        // iterating `no.at(...).as_array()` directly walks a view whose
        // owner has already been destroyed. (Latent until a checkpoint
        // arrived whose normalizer is actually a Sequence.)
        FlexData seq = no.at("normalizers");
        if (seq.is_array()) {
          auto arr = seq.as_array();
          for (size_t i = 0; i < arr.size(); ++i) {
            FlexData s = arr.at(i);
            if (s.is_object()) { try_replace(s.as_object()); }
          }
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
    // Unigram: collect the Replace rules to APPLY, rather than only
    // sniffing them for the metaspace marker. T5's normalizer is a
    // Replace of the regex " {2,}" with " " -- collapsing runs of
    // spaces, which changes the token sequence of any prompt that has
    // one. Only for Unigram: the BPE families' " " -> marker rule is
    // already folded into normalize_ms_ and applying it twice here
    // would substitute the marker into itself.
    if (_unigram) {
      auto collect_replace = [&](const FlexData::ConstObjectView& no) {
        if (!no.contains("type")
            || string(no.at("type").as_string("")) != "Replace") {
          return;
        }
        string pat, content, kind;
        if (no.contains("pattern")) {
          auto p = no.at("pattern");
          if (p.is_object()) {
            auto po = p.as_object();
            if (po.contains("Regex")) {
              pat = string(po.at("Regex").as_string(""));
              kind = "Regex";
            } else if (po.contains("String")) {
              pat = string(po.at("String").as_string(""));
              kind = "String";
            }
          }
        }
        if (no.contains("content")) {
          content = string(no.at("content").as_string(""));
        }
        if (pat.empty()) { return; }
        if (kind == "String") {
          // A literal, so escape it into a regex rather than adding a
          // second substitution mechanism.
          string esc;
          for (char c : pat) {
            if (std::strchr("\\^$.|?*+()[]{}", c) != nullptr) { esc += '\\'; }
            esc += c;
          }
          pat = esc;
        }
        try {
          _norm_replaces.emplace_back(regex(pat), content);
        } catch (const std::regex_error&) {
          if (session) {
            session->warn(fmt(
                "Tokenizer({}): normalizer pattern '{}' did not compile; "
                "skipping it", tag_str, pat));
          }
        }
      };
      auto nrm2 = rootobj.at("normalizer");
      if (nrm2.is_object()) {
        auto no = nrm2.as_object();
        const string ty = no.contains("type")
            ? string(no.at("type").as_string("")) : "";
        if (ty == "Sequence" && no.contains("normalizers")) {
          // at() returns a FlexData BY VALUE and as_array() is a VIEW into
          // it, so the array has to be bound to a named local first --
          // iterating `no.at(...).as_array()` directly walks a view whose
          // owner was already destroyed.
          FlexData seq = no.at("normalizers");
          if (seq.is_array()) {
            auto arr = seq.as_array();
            for (size_t i = 0; i < arr.size(); ++i) {
              FlexData s = arr.at(i);
              if (s.is_object()) { collect_replace(s.as_object()); }
            }
          }
        } else {
          collect_replace(no);
        }
      }
    }
  }

  // ---- Metaspace PRE-tokenizer (SentencePiece / T5) -----------------
  // Distinct from the metaspace NORMALIZER the BPE families use: here
  // the space -> marker substitution, the leading-marker prepend and
  // the split into per-word segments all come from the pre_tokenizer,
  // and the split matters -- it bounds what a single Unigram token may
  // span, so a piece can never straddle two words.
  if (_unigram && rootobj.contains("pre_tokenizer")) {
    auto pre = rootobj.at("pre_tokenizer");
    if (pre.is_object()) {
      auto po = pre.as_object();
      if (po.contains("type")
          && string(po.at("type").as_string("")) == "Metaspace") {
        _metaspace = true;
        _ms_marker = po.contains("replacement")
            ? string(po.at("replacement").as_string("\xe2\x96\x81"))
            : string("\xe2\x96\x81");
        const string scheme = po.contains("prepend_scheme")
            ? string(po.at("prepend_scheme").as_string("always"))
            : string("always");
        _ms_prepend = (scheme != "never");
        _ms_split = po.contains("split") ? po.at("split").as_bool(true) : true;
      }
    }
  }
  if (_unigram && !_metaspace) {
    // Every Unigram checkpoint this supports is SentencePiece-shaped.
    // One without the metaspace pre-tokenizer would need a different
    // pre-tokenization entirely, so refuse rather than encode it wrong.
    if (session) {
      session->warn(fmt(
          "Tokenizer({}): Unigram without a Metaspace pre-tokenizer is not "
          "supported", tag_str));
    }
    return false;
  }

  // ---- pre_tokenizer regex (optional) ------------------------------
  if (rootobj.contains("pre_tokenizer")) {
    auto pre = rootobj.at("pre_tokenizer");
    vector<string> patterns;
    collect_split_regexes_(pre, &patterns);
    for (const auto& p : patterns) {
      // Mistral Tekken: the only pattern here that names the letter
      // CASE classes (see tekken_match_one_).
      if (p.find("\\p{Lu}") != string::npos
          && p.find("\\p{Ll}") != string::npos) {
        _use_tekken_pre = true;
        continue;
      }
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

bool
Tokenizer::Impl::load_tiktoken(string_view path, span<const string> specials,
                               const SessionContextIntf* session)
{
  const string path_str(path);
  ifstream in(path_str, ios::binary);
  if (!in) {
    if (session) {
      session->warn(fmt("Tokenizer::from_tiktoken('{}'): cannot open",
                        path_str));
    }
    return false;
  }
  // One "<base64 bytes> <rank>" per line. The vocab is keyed in the
  // byte-level alphabet like every other BPE here, so encode_chunk_ and
  // the decoder need no tiktoken arm of their own.
  string line;
  int32_t max_rank = -1;
  while (getline(in, line)) {
    if (!line.empty() && line.back() == '\r') { line.pop_back(); }
    if (line.empty()) { continue; }
    const auto sp = line.find(' ');
    if (sp == string::npos) { return false; }
    auto bytes = media_line::base64_decode(string_view(line).substr(0, sp));
    if (!bytes) { return false; }
    int32_t rank = -1;
    try {
      rank = static_cast<int32_t>(stol(line.substr(sp + 1)));
    } catch (const exception&) {
      return false;
    }
    if (rank < 0) { return false; }
    const string tok = bytes_to_byte_level_(
        string_view(reinterpret_cast<const char*>(bytes->data()),
                    bytes->size()));
    _vocab.emplace(tok, rank);
    _inv_vocab.emplace(rank, tok);
    max_rank = std::max(max_rank, rank);
  }
  if (max_rank < 0 || (size_t)(max_rank + 1) != _vocab.size()) {
    if (session) {
      session->warn(fmt(
          "Tokenizer::from_tiktoken('{}'): {} ranks are not dense in "
          "[0, {})", path_str, _vocab.size(), max_rank + 1));
    }
    return false;
  }
  _n_ordinary = max_rank + 1;
  _vocab_size = _n_ordinary;
  // Specials follow the ordinary ranks, in the order given.
  for (const auto& name : specials) {
    const int32_t id = _vocab_size++;
    _special_by_name.emplace(name, id);
    _special_by_id.emplace(id, name);
  }
  _rank_merges = true;
  _use_llama3_pre = true;
  _pre_digit_run = 1;
  detect_thinking_markers_();
  return true;
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
      int32_t prio = numeric_limits<int32_t>::max();
      if (_rank_merges) {
        auto it = _vocab.find((*pieces)[i] + (*pieces)[i + 1]);
        if (it != _vocab.end() && it->second < _n_ordinary) {
          prio = it->second;
        }
      } else {
        auto it = _merge_priority.find({(*pieces)[i], (*pieces)[i + 1]});
        if (it != _merge_priority.end()) { prio = it->second; }
      }
      // Strictly less: the LEFTMOST of equal priorities merges first,
      // which is also what tiktoken does.
      if (prio < best_prio) {
        best_prio = prio;
        best_idx  = i;
      }
    }
    if (best_idx == numeric_limits<size_t>::max()) { break; }
    (*pieces)[best_idx] += (*pieces)[best_idx + 1];
    pieces->erase(pieces->begin() + static_cast<long>(best_idx) + 1);
  }
}

// The maximum-score segmentation of one pre-token.
//
// This is what makes Unigram a different model rather than a different
// pre-tokenizer: BPE commits to each merge in rank order and never
// reconsiders, while this considers EVERY segmentation and takes the one
// whose token log-probabilities sum highest. A greedy longest-match would
// agree on most words and differ on exactly the ones that matter.
//
// Boundaries are codepoint boundaries, not byte offsets: a vocab piece is
// a sequence of characters, and stepping by bytes would let the unknown
// fallback split a multi-byte character in half.
void
Tokenizer::Impl::viterbi_(string_view seg, vector<int32_t>* out) const
{
  if (seg.empty()) { return; }
  // Codepoint boundary offsets, plus the end.
  vector<size_t> bnd;
  bnd.reserve(seg.size() + 1);
  for (size_t i = 0; i < seg.size();) {
    bnd.push_back(i);
    const unsigned char c = (unsigned char)seg[i];
    size_t n = 1;
    if ((c & 0xF8) == 0xF0)      { n = 4; }
    else if ((c & 0xF0) == 0xE0) { n = 3; }
    else if ((c & 0xE0) == 0xC0) { n = 2; }
    i += n;
    if (i > seg.size()) { i = seg.size(); }
  }
  bnd.push_back(seg.size());
  const size_t N = bnd.size() - 1;         // characters

  constexpr float kNeg = -1e30f;
  vector<float>   best(N + 1, kNeg);
  vector<int32_t> from(N + 1, -1);
  vector<int32_t> tok(N + 1, -1);
  best[0] = 0.0f;

  for (size_t i = 0; i < N; ++i) {
    if (best[i] == kNeg) { continue; }
    bool single_char_hit = false;
    for (size_t j = i + 1; j <= N; ++j) {
      const size_t len = bnd[j] - bnd[i];
      if (len > _uni_max_piece) { break; }
      auto it = _vocab.find(string(seg.substr(bnd[i], len)));
      if (it == _vocab.end()) { continue; }
      if (j == i + 1) { single_char_hit = true; }
      const float s = best[i] + _uni_score[(size_t)it->second];
      if (s > best[j]) { best[j] = s; from[j] = (int32_t)i; tok[j] = it->second; }
    }
    // Unknown-character fallback, added only when the vocab has no
    // single-character token here -- the reference's rule. Adding it
    // unconditionally would let a heavily-penalised unk beat a real
    // token whose score happens to be lower still.
    if (!single_char_hit && _unk_id >= 0) {
      const float s = best[i] + _unk_score;
      if (s > best[i + 1]) {
        best[i + 1] = s; from[i + 1] = (int32_t)i; tok[i + 1] = _unk_id;
      }
    }
  }
  if (best[N] == kNeg) { return; }         // nothing reached the end
  vector<int32_t> rev;
  for (size_t j = N; j > 0;) {
    if (from[j] < 0) { return; }           // broken chain: emit nothing
    rev.push_back(tok[j]);
    j = (size_t)from[j];
  }
  out->insert(out->end(), rev.rbegin(), rev.rend());
}

vector<int32_t>
Tokenizer::Impl::encode_unigram_(string_view text) const
{
  // 1. normalizer (T5: collapse runs of spaces).
  string s(text);
  for (const auto& r : _norm_replaces) {
    s = std::regex_replace(s, r.first, r.second);
  }
  // 2. metaspace: space -> marker, THEN prepend one -- but only when the
  //    result does not already start with a marker. The order and the
  //    condition are both load-bearing. The prepend is what makes the
  //    first word carry the same word-initial marker every later word
  //    does ("▁the" is a different token from "the"); the condition is
  //    what stops a leading space from producing two markers, which
  //    would encode as a spurious extra token before the prompt.
  string m;
  m.reserve(s.size() + _ms_marker.size());
  for (char c : s) {
    if (c == ' ') { m += _ms_marker; }
    else          { m += c; }
  }
  if (_ms_prepend && m.compare(0, _ms_marker.size(), _ms_marker) != 0) {
    m.insert(0, _ms_marker);
  }
  // 3. split at each marker so a token cannot span two words, then
  //    Viterbi each segment.
  vector<int32_t> out;
  if (!_ms_split) {
    viterbi_(m, &out);
    return out;
  }
  // Each segment runs from one marker up to (not including) the next, so
  // the marker stays attached to the word that FOLLOWS it -- which is
  // what makes "▁the" a different token from "the". Searching from
  // pos + 1 steps past the current marker's first byte without being
  // able to re-find it: the marker is a valid UTF-8 sequence, so it
  // cannot match at a non-boundary.
  size_t pos = 0;
  while (pos < m.size()) {
    const size_t next = m.find(_ms_marker, pos + 1);
    const size_t end = (next == string::npos) ? m.size() : next;
    viterbi_(string_view(m).substr(pos, end - pos), &out);
    pos = end;
  }
  return out;
}

void
Tokenizer::Impl::encode_chunk_(string_view chunk, vector<int32_t>* out) const
{
  if (chunk.empty()) { return; }
  if (_ignore_merges && !_metaspace) {
    const auto it = _vocab.find(bytes_to_byte_level_(chunk));
    if (it != _vocab.end()) {
      out->push_back(it->second);
      return;
    }
  }
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
  if (_use_tekken_pre) {
    vector<string> out;
    const char* data = text.data();
    const size_t len = text.size();
    size_t pos = 0;
    while (pos < len) {
      size_t n = tekken_match_one_(data, len, pos);
      if (n == 0) { break; }
      out.emplace_back(data + pos, n);
      pos += n;
    }
    return out;
  }
  if (_use_llama3_pre) {
    // Hand-coded scanner for the Llama-3 / Qwen-2.5 pre-tokenizer
    // regex. Walks the seven alternatives at each position; emits
    // chunks in input order.
    vector<string> out;
    const char* data = text.data();
    const size_t len = text.size();
    size_t pos = 0;
    while (pos < len) {
      size_t n = llama3_match_one_(data, len, pos, _pre_digit_run);
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
  if (_unigram) { return encode_unigram_(text); }
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

unique_ptr<Tokenizer>
Tokenizer::from_tiktoken(string_view path, span<const string> specials,
                         const SessionContextIntf* session)
{
  unique_ptr<Tokenizer> tok(new Tokenizer);
  if (!tok->_impl->load_tiktoken(path, specials, session)) {
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

string
Tokenizer::whitespace_clean(string_view text)
{
  // Any ASCII whitespace, not just ' ': a tab or newline that survived
  // to the metaspace substitution would become an unknown token rather
  // than a word break.
  auto is_ws = [](unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'
           || c == '\v';
  };
  string out;
  out.reserve(text.size());
  bool pending = false;
  for (char ch : text) {
    if (is_ws((unsigned char)ch)) {
      pending = !out.empty();       // never emit a LEADING space
      continue;
    }
    if (pending) { out += ' '; pending = false; }
    out += ch;
  }
  return out;                       // `pending` at the end == trailing, dropped
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
