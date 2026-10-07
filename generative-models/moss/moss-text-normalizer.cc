#include "generative-models/moss/moss-text-normalizer.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

// A literal port of normalize_tts_text. Each Python regex step is one
// function below, named after the Python helper it ports, operating on
// UTF-32 so a "character" means what it means to Python's str. Where a
// regex could backtrack, the comment says why the scan here takes the
// same branch; re.sub's left-to-right, non-overlapping search is the
// shape every step shares.

namespace vpipe::genai {

namespace {

#include "generative-models/moss/moss-text-unicode.inc"

using U32 = std::u32string;

template <std::size_t N>
bool
in_ranges_(const std::uint32_t (&r)[N][2], char32_t c)
{
  std::size_t lo = 0, hi = N;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (c < r[mid][0]) {
      hi = mid;
    } else if (c > r[mid][1]) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

bool is_space_(char32_t c)   { return in_ranges_(kSpaceRanges, c); }
bool is_decimal_(char32_t c) { return in_ranges_(kDecimalRanges, c); }
bool is_word_(char32_t c)    { return in_ranges_(kWordRanges, c); }
bool is_control_(char32_t c) { return in_ranges_(kControlRanges, c); }

bool
is_ascii_letter_(char32_t c)
{
  return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z');
}
bool
is_ascii_alnum_(char32_t c)
{
  return is_ascii_letter_(c) || (c >= U'0' && c <= U'9');
}
// [A-Za-z0-9_]
bool is_ascii_word_(char32_t c) { return is_ascii_alnum_(c) || c == U'_'; }

// _CJK: [㐀-䶿一-鿿぀-ヿ]
bool
is_cjk_(char32_t c)
{
  return (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) ||
         (c >= 0x3040 && c <= 0x30FF);
}

bool
one_of_(char32_t c, std::u32string_view set)
{
  return set.find(c) != std::u32string_view::npos;
}

// [A-Za-z0-9._/+:-] -- the file-like / LATINISH class (it has '_').
bool
is_file_(char32_t c)
{
  return is_ascii_alnum_(c) || one_of_(c, U"._/+:-");
}

// Length of the run of `pred` characters starting at p.
template <class Pred>
std::size_t
run_(const U32& t, std::size_t p, Pred pred)
{
  std::size_t e = p;
  while (e < t.size() && pred(t[e])) { ++e; }
  return e - p;
}

std::size_t
spaces_(const U32& t, std::size_t p)
{
  return run_(t, p, is_space_);
}

// A pattern matched at position s: the end of the match, or npos.
constexpr std::size_t kNo = U32::npos;

// re.sub(pattern, repl, t) for a pattern given as `match(t, s) -> end`
// and a replacement given as `repl(t, s, end, out)`.
template <class Match, class Repl>
U32
sub_(const U32& t, Match match, Repl repl)
{
  U32 out;
  out.reserve(t.size());
  std::size_t s = 0;
  while (s < t.size()) {
    const std::size_t e = match(t, s);
    if (e != kNo && e > s) {
      repl(t, s, e, out);
      s = e;
    } else {
      out.push_back(t[s]);
      ++s;
    }
  }
  return out;
}

// ---- UTF-8 <-> UTF-32 -------------------------------------------------

U32
decode_utf8_(std::string_view s)
{
  U32 out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    std::size_t need = 0;
    char32_t cp = 0;
    if (b0 < 0x80) {
      out.push_back(b0);
      ++i;
      continue;
    } else if ((b0 & 0xE0) == 0xC0) {
      need = 1; cp = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
      need = 2; cp = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
      need = 3; cp = b0 & 0x07;
    } else {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    bool ok = i + need < s.size();
    for (std::size_t k = 1; ok && k <= need; ++k) {
      const auto b = static_cast<unsigned char>(s[i + k]);
      if ((b & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (b & 0x3F);
    }
    if (!ok) {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    out.push_back(cp);
    i += need + 1;
  }
  return out;
}

std::string
encode_utf8_(const U32& t)
{
  std::string out;
  out.reserve(t.size());
  for (char32_t c : t) {
    if (c < 0x80) {
      out.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (c >> 6)));
      out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (c >> 12)));
      out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (c >> 18)));
      out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
  }
  return out;
}

U32
strip_(const U32& t)
{
  std::size_t b = 0, e = t.size();
  while (b < e && is_space_(t[b])) { ++b; }
  while (e > b && is_space_(t[e - 1])) { --e; }
  return t.substr(b, e - b);
}

// ---- _base_cleanup ----------------------------------------------------

// CRLF / CR -> LF, U+3000 -> space, zero-width [​-‍﻿]
// removed, then every category-C character except \n \t and space.
U32
base_cleanup_(const U32& t)
{
  U32 out;
  out.reserve(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    char32_t c = t[i];
    if (c == U'\r') {
      out.push_back(U'\n');
      if (i + 1 < t.size() && t[i + 1] == U'\n') { ++i; }
      continue;
    }
    if (c == 0x3000) { c = U' '; }
    if ((c >= 0x200B && c <= 0x200D) || c == 0xFEFF) { continue; }
    if (c == U'\n' || c == U'\t' || c == U' ' || !is_control_(c)) {
      out.push_back(c);
    }
  }
  return out;
}

// ---- _normalize_markdown_and_lines -------------------------------------

// \[([^\[\]]+?)\]\((https?://[^)\s]+)\)  ->  "\1 \2". The label cannot
// hold a bracket, so it ends at the first ']' whatever the laziness.
U32
markdown_links_(const U32& t)
{
  U32 out;
  out.reserve(t.size());
  std::size_t s = 0;
  while (s < t.size()) {
    std::size_t e = kNo, lab_e = 0, url_b = 0, url_e = 0;
    if (t[s] == U'[') {
      const std::size_t lb = s + 1;
      const std::size_t le =
          lb + run_(t, lb, [](char32_t c) { return c != U'[' && c != U']'; });
      if (le > lb && le + 1 < t.size() && t[le] == U']' &&
          t[le + 1] == U'(') {
        std::size_t p = le + 2;
        const U32 http = U"http";
        if (t.compare(p, http.size(), http) == 0) {
          p += http.size();
          if (p < t.size() && t[p] == U's') { ++p; }
          if (t.compare(p, 3, U"://") == 0) {
            const std::size_t ub = le + 2;
            p += 3;
            const std::size_t n = run_(t, p, [](char32_t c) {
              return c != U')' && !is_space_(c);
            });
            if (n >= 1 && p + n < t.size() && t[p + n] == U')') {
              e = p + n + 1;
              lab_e = le;
              url_b = ub;
              url_e = p + n;
            }
          }
        }
      }
    }
    if (e != kNo) {
      out.append(t, s + 1, lab_e - (s + 1));
      out.push_back(U' ');
      out.append(t, url_b, url_e - url_b);
      s = e;
    } else {
      out.push_back(t[s]);
      ++s;
    }
  }
  return out;
}

// str.splitlines() boundaries.
bool
is_line_break_(char32_t c)
{
  return c == U'\n' || c == U'\r' || c == 0x0B || c == 0x0C ||
         c == 0x1C || c == 0x1D || c == 0x1E || c == 0x85 ||
         c == 0x2028 || c == 0x2029;
}

// The four ^-anchored prefix strips, applied in order to one line.
U32
strip_line_markers_(U32 line)
{
  // ^#{1,6}\s+ : more than six '#' leaves '#' after the quantifier, which
  // is not \s, so it fails at every split.
  {
    const std::size_t n = run_(line, 0, [](char32_t c) { return c == U'#'; });
    if (n >= 1 && n <= 6) {
      const std::size_t w = spaces_(line, n);
      if (w >= 1) { line.erase(0, n + w); }
    }
  }
  // ^>\s+ and ^[-*+]\s+
  for (std::u32string_view set : {std::u32string_view(U">"),
                                  std::u32string_view(U"-*+")}) {
    if (!line.empty() && one_of_(line[0], set)) {
      const std::size_t w = spaces_(line, 1);
      if (w >= 1) { line.erase(0, 1 + w); }
    }
  }
  // ^\d+[.)]\s+
  {
    const std::size_t n = run_(line, 0, is_decimal_);
    if (n >= 1 && n < line.size() && (line[n] == U'.' || line[n] == U')')) {
      const std::size_t w = spaces_(line, n + 1);
      if (w >= 1) { line.erase(0, n + 1 + w); }
    }
  }
  return line;
}

U32
normalize_markdown_and_lines_(const U32& in)
{
  const U32 t = markdown_links_(in);
  U32 out;
  bool first = true;
  std::size_t b = 0;
  while (b <= t.size()) {
    std::size_t e = b;
    while (e < t.size() && !is_line_break_(t[e])) { ++e; }
    U32 line = strip_(t.substr(b, e - b));
    if (!line.empty()) {
      line = strip_line_markers_(std::move(line));
      if (!first) { out += U"。"; }
      out += line;
      first = false;
    }
    if (e >= t.size()) { break; }
    // \r\n is one boundary to splitlines; base_cleanup already folded it.
    b = e + 1;
  }
  return out;
}

// ---- _protect_spans / _restore_spans -----------------------------------

U32
placeholder_(std::size_t idx)
{
  const std::string n = std::to_string(idx);
  U32 p = U"___PROT";
  for (char c : n) { p.push_back(static_cast<char32_t>(c)); }
  p += U"___";
  return p;
}

// ___PROT\d+___ at p: its end, or npos.
std::size_t
placeholder_at_(const U32& t, std::size_t p)
{
  static const U32 head = U"___PROT";
  if (t.compare(p, head.size(), head) != 0) { return kNo; }
  std::size_t q = p + head.size();
  const std::size_t d = run_(t, q, is_decimal_);
  if (d == 0) { return kNo; }
  q += d;
  if (t.compare(q, 3, U"___") != 0) { return kNo; }
  return q + 3;
}

bool
ascii_lookbehind_ok_(const U32& t, std::size_t s)
{
  return s == 0 || !is_ascii_word_(t[s - 1]);
}

// https?://[^\s　，。！？；、）】》〉」』]+
std::size_t
match_url_(const U32& t, std::size_t s)
{
  if (t.compare(s, 4, U"http") != 0) { return kNo; }
  std::size_t p = s + 4;
  if (p < t.size() && t[p] == U's') { ++p; }
  if (t.compare(p, 3, U"://") != 0) { return kNo; }
  p += 3;
  const std::size_t n = run_(t, p, [](char32_t c) {
    return !is_space_(c) && !one_of_(c, U"　，。！？；、）】》〉」』");
  });
  return n >= 1 ? p + n : kNo;
}

// (?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])
//
// '@' is outside the local class, so the local part is the whole run
// before it. The TLD's letters can only stop at a non-[\w.-] character
// if they are the END of the domain run (anything inside the run is
// [\w.-]), so the match exists iff the domain run's last segment is two
// or more letters after a '.' that is not the run's first character.
std::size_t
match_email_(const U32& t, std::size_t s)
{
  if (s > 0) {
    const char32_t b = t[s - 1];
    if (is_word_(b) || b == U'.' || b == U'+' || b == U'-') { return kNo; }
  }
  auto local = [](char32_t c) {
    return is_ascii_alnum_(c) || one_of_(c, U"._%+-");
  };
  const std::size_t a = s + run_(t, s, local);
  if (a == s || a >= t.size() || t[a] != U'@') { return kNo; }
  auto dom = [](char32_t c) { return is_ascii_alnum_(c) || one_of_(c, U".-"); };
  const std::size_t d0 = a + 1;
  const std::size_t d1 = d0 + run_(t, d0, dom);
  std::size_t letters = 0;
  while (d1 - letters > d0 && is_ascii_letter_(t[d1 - letters - 1])) {
    ++letters;
  }
  if (letters < 2) { return kNo; }
  const std::size_t dot = d1 - letters - 1;
  if (dot <= d0 || dot >= d1 || t[dot] != U'.') { return kNo; }
  if (d1 < t.size() && is_word_(t[d1])) { return kNo; }
  return d1;
}

// (?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}
std::size_t
match_mention_(const U32& t, std::size_t s)
{
  if (t[s] != U'@' || !ascii_lookbehind_ok_(t, s)) { return kNo; }
  const std::size_t n = run_(t, s + 1, is_ascii_word_);
  return n >= 1 ? s + 1 + std::min<std::size_t>(n, 32) : kNo;
}

// (?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+
std::size_t
match_reddit_(const U32& t, std::size_t s)
{
  if ((t[s] != U'u' && t[s] != U'r') || !ascii_lookbehind_ok_(t, s)) {
    return kNo;
  }
  if (s + 1 >= t.size() || t[s + 1] != U'/') { return kNo; }
  const std::size_t n = run_(t, s + 2, is_ascii_word_);
  return n >= 1 ? s + 2 + n : kNo;
}

// (?<![A-Za-z0-9_])#(?!\s)[^\s#]+
std::size_t
match_hashtag_(const U32& t, std::size_t s)
{
  if (t[s] != U'#' || !ascii_lookbehind_ok_(t, s)) { return kNo; }
  const std::size_t n = run_(t, s + 1, [](char32_t c) {
    return !is_space_(c) && c != U'#';
  });
  return n >= 1 ? s + 1 + n : kNo;
}

// (?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+
// The lookahead holds iff the run after the dot has an alnum in it.
std::size_t
match_dot_token_(const U32& t, std::size_t s)
{
  if (t[s] != U'.' || !ascii_lookbehind_ok_(t, s)) { return kNo; }
  auto cls = [](char32_t c) {
    return is_ascii_alnum_(c) || one_of_(c, U"._-");
  };
  const std::size_t n = run_(t, s + 1, cls);
  for (std::size_t i = s + 1; i < s + 1 + n; ++i) {
    if (is_ascii_alnum_(t[i])) { return s + 1 + n; }
  }
  return kNo;
}

// (?<![A-Za-z0-9_])
// (?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[._/+:-])
// [A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?(?![A-Za-z0-9_])
//
// Both lookaheads test the whole class run R from s. The body then ends
// on an alnum of R; greedy, so the LAST alnum whose next character is
// not [A-Za-z0-9_] wins (an '_' after it makes that end fail).
std::size_t
match_filelike_(const U32& t, std::size_t s)
{
  if (!is_ascii_alnum_(t[s]) || !ascii_lookbehind_ok_(t, s)) { return kNo; }
  const std::size_t r1 = s + run_(t, s, is_file_);
  bool letter = false, punct = false;
  for (std::size_t i = s; i < r1; ++i) {
    letter = letter || is_ascii_letter_(t[i]);
    punct = punct || one_of_(t[i], U"._/+:-");
  }
  if (!letter || !punct) { return kNo; }
  for (std::size_t e = r1; e > s; --e) {
    if (!is_ascii_alnum_(t[e - 1])) { continue; }
    if (e >= t.size() || !is_ascii_word_(t[e])) { return e; }
  }
  return kNo;
}

U32
protect_spans_(const U32& in, std::vector<U32>& prot)
{
  using Fn = std::size_t (*)(const U32&, std::size_t);
  static const Fn kPatterns[] = {
    match_url_, match_email_, match_mention_, match_reddit_,
    match_hashtag_, match_dot_token_, match_filelike_,
  };
  U32 t = in;
  for (Fn fn : kPatterns) {
    t = sub_(t, fn, [&](const U32& src, std::size_t s, std::size_t e,
                        U32& out) {
      out += placeholder_(prot.size());
      prot.push_back(src.substr(s, e - s));
    });
  }
  return t;
}

// str.replace for each placeholder in order -- a span that swallowed an
// earlier placeholder restores it literally, as the reference does.
U32
restore_spans_(U32 t, const std::vector<U32>& prot)
{
  for (std::size_t i = 0; i < prot.size(); ++i) {
    const U32 ph = placeholder_(i);
    U32 out;
    std::size_t p = 0;
    while (true) {
      const std::size_t f = t.find(ph, p);
      if (f == U32::npos) {
        out.append(t, p, U32::npos);
        break;
      }
      out.append(t, p, f - p);
      out += prot[i];
      p = f + ph.size();
    }
    t = std::move(out);
  }
  return t;
}

// ---- _normalize_spaces --------------------------------------------------

// _LATINISH at p: ___PROT\d+___ | (?=[class]*[A-Za-z])[A-Za-z0-9][class]*
bool
latinish_at_(const U32& t, std::size_t p)
{
  if (p >= t.size()) { return false; }
  if (placeholder_at_(t, p) != kNo) { return true; }
  if (!is_ascii_alnum_(t[p])) { return false; }
  const std::size_t r1 = p + run_(t, p, is_file_);
  for (std::size_t i = p; i < r1; ++i) {
    if (is_ascii_letter_(t[i])) { return true; }
  }
  return false;
}

// "X\s+(?=Y)" -> "X" for one-character classes X and Y.
template <class X, class Y>
U32
drop_spaces_between_(const U32& t, X x, Y y)
{
  return sub_(
      t,
      [&](const U32& s_, std::size_t s) {
        if (!x(s_[s])) { return kNo; }
        const std::size_t w = spaces_(s_, s + 1);
        if (w == 0 || s + 1 + w >= s_.size() || !y(s_[s + 1 + w])) {
          return kNo;
        }
        return s + 1 + w;
      },
      [](const U32& s_, std::size_t s, std::size_t, U32& out) {
        out.push_back(s_[s]);
      });
}

// " {2,}" -> " "
U32
squeeze_spaces_(const U32& t)
{
  return sub_(
      t,
      [](const U32& s_, std::size_t s) {
        const std::size_t n = run_(s_, s, [](char32_t c) { return c == U' '; });
        return n >= 2 ? s + n : kNo;
      },
      [](const U32&, std::size_t, std::size_t, U32& out) {
        out.push_back(U' ');
      });
}

U32
normalize_spaces_(const U32& in)
{
  // [ \t\r\f\v]+ -> " "
  U32 t = sub_(
      in,
      [](const U32& s_, std::size_t s) {
        const std::size_t n =
            run_(s_, s, [](char32_t c) { return one_of_(c, U" \t\r\f\v"); });
        return n >= 1 ? s + n : kNo;
      },
      [](const U32&, std::size_t, std::size_t, U32& out) {
        out.push_back(U' ');
      });

  t = drop_spaces_between_(t, is_cjk_, is_cjk_);
  t = drop_spaces_between_(t, is_cjk_, is_decimal_);
  t = drop_spaces_between_(t, is_decimal_, is_cjk_);

  // (CJK)(?=LATINISH) -> "\1 "
  t = sub_(
      t,
      [](const U32& s_, std::size_t s) {
        return is_cjk_(s_[s]) && latinish_at_(s_, s + 1) ? s + 1 : kNo;
      },
      [](const U32& s_, std::size_t s, std::size_t, U32& out) {
        out.push_back(s_[s]);
        out.push_back(U' ');
      });

  // (LATINISH)(?=CJK) -> "\1 ". The class run can only be followed by a
  // CJK character at its own end, so no shorter body can succeed.
  t = sub_(
      t,
      [](const U32& s_, std::size_t s) {
        const std::size_t ph = placeholder_at_(s_, s);
        if (ph != kNo && ph < s_.size() && is_cjk_(s_[ph])) { return ph; }
        if (!latinish_at_(s_, s) || !is_ascii_alnum_(s_[s])) { return kNo; }
        const std::size_t r1 = s + run_(s_, s, is_file_);
        return r1 < s_.size() && is_cjk_(s_[r1]) ? r1 : kNo;
      },
      [](const U32& s_, std::size_t s, std::size_t e, U32& out) {
        out.append(s_, s, e - s);
        out.push_back(U' ');
      });

  t = squeeze_spaces_(t);

  // \s+([，。！？；：、”’」』】）》]) -> "\1"
  t = sub_(
      t,
      [](const U32& s_, std::size_t s) {
        const std::size_t w = spaces_(s_, s);
        if (w == 0 || s + w >= s_.size() ||
            !one_of_(s_[s + w], U"，。！？；：、”’」』】）》")) {
          return kNo;
        }
        return s + w + 1;
      },
      [](const U32& s_, std::size_t, std::size_t e, U32& out) {
        out.push_back(s_[e - 1]);
      });

  // ([（【「『《“‘])\s+ -> "\1"
  t = sub_(
      t,
      [](const U32& s_, std::size_t s) {
        if (!one_of_(s_[s], U"（【「『《“‘")) { return kNo; }
        const std::size_t w = spaces_(s_, s + 1);
        return w >= 1 ? s + 1 + w : kNo;
      },
      [](const U32& s_, std::size_t s, std::size_t, U32& out) {
        out.push_back(s_[s]);
      });

  // ([，。！？；：、])\s* -> "\1"
  t = sub_(
      t,
      [](const U32& s_, std::size_t s) {
        if (!one_of_(s_[s], U"，。！？；：、")) { return kNo; }
        return s + 1 + spaces_(s_, s + 1);
      },
      [](const U32& s_, std::size_t s, std::size_t, U32& out) {
        out.push_back(s_[s]);
      });

  // \s+([,.;!?]) -> "\1"
  t = sub_(
      t,
      [](const U32& s_, std::size_t s) {
        const std::size_t w = spaces_(s_, s);
        if (w == 0 || s + w >= s_.size() || !one_of_(s_[s + w], U",.;!?")) {
          return kNo;
        }
        return s + w + 1;
      },
      [](const U32& s_, std::size_t, std::size_t e, U32& out) {
        out.push_back(s_[e - 1]);
      });

  return strip_(squeeze_spaces_(t));
}

// ---- _normalize_structural_punctuation ----------------------------------

constexpr std::u32string_view kSentenceEnd = U"。！？!?；;";

// (^|[。！？!?；;]\s*) at s: the alternatives' end positions (the ^ arm
// first, as re tries it), each paired with the group's own end.
template <class Rest>
std::size_t
lead_then_(const U32& t, std::size_t s, Rest rest, std::size_t* g_end)
{
  if (s == 0) {
    const std::size_t e = rest(t, s);
    if (e != kNo) { *g_end = s; return e; }
  }
  if (one_of_(t[s], kSentenceEnd)) {
    const std::size_t p = s + 1 + spaces_(t, s + 1);
    const std::size_t e = rest(t, p);
    if (e != kNo) { *g_end = p; return e; }
  }
  return kNo;
}

U32
structural_brackets_(const U32& t)
{
  // [【〖『「]([^】〗』」]+)[】〗』」]\s*  after the lead.
  auto rest = [](const U32& s_, std::size_t p) {
    if (p >= s_.size() || !one_of_(s_[p], U"【〖『「")) { return kNo; }
    const std::size_t n = run_(s_, p + 1, [](char32_t c) {
      return !one_of_(c, U"】〗』」");
    });
    const std::size_t q = p + 1 + n;
    if (n == 0 || q >= s_.size()) { return kNo; }
    return q + 1 + spaces_(s_, q + 1);
  };
  std::size_t g = 0;
  return sub_(
      t,
      [&](const U32& s_, std::size_t s) { return lead_then_(s_, s, rest, &g); },
      [&](const U32& s_, std::size_t s, std::size_t, U32& out) {
        // "\1\2。": the lead, then the bracket's content.
        out.append(s_, s, g - s);
        const std::size_t cb = g + 1;
        const std::size_t ce = cb + run_(s_, cb, [](char32_t c) {
          return !one_of_(c, U"】〗』」");
        });
        out.append(s_, cb, ce - cb);
        out += U"。";
      });
}

bool
is_dash_(char32_t c)
{
  return c == U'—' || c == U'–' || c == U'―' || c == U'-';
}

U32
standalone_titles_(const U32& t)
{
  // 《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))
  auto rest = [](const U32& s_, std::size_t p) {
    if (p >= s_.size() || s_[p] != U'《') { return kNo; }
    const std::size_t n = run_(s_, p + 1, [](char32_t c) {
      return c != U'》';
    });
    const std::size_t q = p + 1 + n;
    if (n == 0 || q >= s_.size()) { return kNo; }
    const std::size_t r = q + 1 + spaces_(s_, q + 1);
    const bool ok = r >= s_.size() || placeholder_at_(s_, r) != kNo ||
                    run_(s_, r, is_dash_) >= 2 ||
                    one_of_(s_[r], U"。！？!?；;，,");
    return ok ? q + 1 : kNo;
  };
  std::size_t g = 0;
  return sub_(
      t,
      [&](const U32& s_, std::size_t s) { return lead_then_(s_, s, rest, &g); },
      [&](const U32& s_, std::size_t s, std::size_t e, U32& out) {
        out.append(s_, s, g - s);
        out.append(s_, g + 1, e - 1 - (g + 1));
      });
}

// \s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s* -> "，"
// Every arm starts on a non-space, so the leading \s* never gives back.
std::size_t
match_arrow_(const U32& t, std::size_t s)
{
  const std::size_t p = s + spaces_(t, s);
  if (p >= t.size()) { return kNo; }
  auto eqdash = [](char32_t c) { return c == U'-' || c == U'='; };
  std::size_t q = kNo;
  if (t[p] == U'<') {
    const std::size_t k = run_(t, p + 1, eqdash);
    if (k >= 1 && p + 1 + k < t.size() && t[p + 1 + k] == U'>') {
      q = p + 2 + k;
    }
  }
  if (q == kNo) {
    const std::size_t k = run_(t, p, eqdash);
    if (k >= 1 && p + k < t.size() && t[p + k] == U'>') { q = p + k + 1; }
  }
  if (q == kNo && t[p] == U'<') {
    const std::size_t k = run_(t, p + 1, eqdash);
    if (k >= 1) { q = p + 1 + k; }
  }
  if (q == kNo && one_of_(t[p], U"→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩")) { q = p + 1; }
  return q == kNo ? kNo : q + spaces_(t, q);
}

// \s*(?:—|–|―|-){2,}\s* -> "。"
std::size_t
match_long_dash_(const U32& t, std::size_t s)
{
  const std::size_t p = s + spaces_(t, s);
  const std::size_t k = run_(t, p, is_dash_);
  if (k < 2) { return kNo; }
  return p + k + spaces_(t, p + k);
}

U32
normalize_structural_punctuation_(const U32& in)
{
  U32 t = structural_brackets_(in);
  t = structural_brackets_(t);   // twice, so adjacent blocks converge
  t = standalone_titles_(t);
  auto put = [](std::u32string_view r) {
    return [r](const U32&, std::size_t, std::size_t, U32& out) { out += r; };
  };
  t = sub_(t, match_arrow_, put(U"，"));
  t = sub_(t, match_long_dash_, put(U"。"));
  return t;
}

// ---- _normalize_repeated_punctuation ------------------------------------

// A run of >= `min` characters from `set` -> `repl` (or a computed one).
template <class Repl>
U32
collapse_runs_(const U32& t, std::u32string_view set, std::size_t min,
               Repl repl)
{
  return sub_(
      t,
      [&](const U32& s_, std::size_t s) {
        const std::size_t n =
            run_(s_, s, [&](char32_t c) { return one_of_(c, set); });
        return n >= min ? s + n : kNo;
      },
      [&](const U32& s_, std::size_t s, std::size_t e, U32& out) {
        out += repl(std::u32string_view(s_).substr(s, e - s));
      });
}

U32
normalize_repeated_punctuation_(const U32& in)
{
  auto fixed = [](std::u32string_view r) {
    return [r](std::u32string_view) { return U32(r); };
  };
  // (?:\.{3,}|…{2,}|……+) -> "。"; the third arm is the second again.
  U32 t = sub_(
      in,
      [](const U32& s_, std::size_t s) {
        const std::size_t dots =
            run_(s_, s, [](char32_t c) { return c == U'.'; });
        if (dots >= 3) { return s + dots; }
        const std::size_t ell =
            run_(s_, s, [](char32_t c) { return c == U'…'; });
        return ell >= 2 ? s + ell : kNo;
      },
      [](const U32&, std::size_t, std::size_t, U32& out) { out += U"。"; });
  t = collapse_runs_(t, U"。．", 2, fixed(U"。"));
  t = collapse_runs_(t, U"，,", 2, fixed(U"，"));
  t = collapse_runs_(t, U"!！", 2, fixed(U"！"));
  t = collapse_runs_(t, U"?？", 2, fixed(U"？"));
  t = collapse_runs_(t, U"!?！？", 2, [](std::u32string_view s) {
    const bool q = s.find_first_of(U"?？") != std::u32string_view::npos;
    const bool e = s.find_first_of(U"!！") != std::u32string_view::npos;
    if (q && e) { return U32(U"？！"); }
    return U32(q ? U"？" : U"！");
  });
  return t;
}

}  // namespace

std::string
moss_tts_normalize_text(std::string_view utf8)
{
  U32 t = base_cleanup_(decode_utf8_(utf8));
  t = normalize_markdown_and_lines_(t);
  std::vector<U32> prot;
  t = protect_spans_(t, prot);
  t = normalize_spaces_(t);
  t = normalize_structural_punctuation_(t);
  t = normalize_repeated_punctuation_(t);
  t = normalize_spaces_(t);
  t = restore_spans_(std::move(t), prot);
  return encode_utf8_(strip_(t));
}

bool
moss_tts_has_text_normalizer(const std::string& model_dir)
{
  std::error_code ec;
  return std::filesystem::is_regular_file(
      std::filesystem::path(model_dir) /
          "tts_robust_normalizer_single_script.py",
      ec);
}

}  // namespace vpipe::genai
