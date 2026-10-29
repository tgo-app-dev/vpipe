#include "generative-models/minimax-music3/music3-protocol.h"

#include "common/flex-data.h"
#include "generative-models/tokenizer.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

// The grids below are numpy's and torch's arithmetic, operation by
// operation: a fused multiply-add would move a node by an ulp.
#pragma clang fp contract(off)

namespace vpipe::genai::music3 {

namespace {

// ---- UTF-8 <-> code points ------------------------------------------------

std::u32string
to_u32_(std::string_view s)
{
  std::u32string out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = (unsigned char)s[i];
    char32_t cp = 0xfffd;
    std::size_t n = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
      cp = ((char32_t)(c & 0x1f) << 6) | ((unsigned char)s[i + 1] & 0x3f);
      n = 2;
    } else if ((c >> 4) == 0xe && i + 2 < s.size()) {
      cp = ((char32_t)(c & 0x0f) << 12) |
           ((char32_t)((unsigned char)s[i + 1] & 0x3f) << 6) |
           ((unsigned char)s[i + 2] & 0x3f);
      n = 3;
    } else if ((c >> 3) == 0x1e && i + 3 < s.size()) {
      cp = ((char32_t)(c & 0x07) << 18) |
           ((char32_t)((unsigned char)s[i + 1] & 0x3f) << 12) |
           ((char32_t)((unsigned char)s[i + 2] & 0x3f) << 6) |
           ((unsigned char)s[i + 3] & 0x3f);
      n = 4;
    }
    out.push_back(cp);
    i += n;
  }
  return out;
}

void
put_utf8_(char32_t cp, std::string* o)
{
  if (cp < 0x80) {
    o->push_back((char)cp);
  } else if (cp < 0x800) {
    o->push_back((char)(0xc0 | (cp >> 6)));
    o->push_back((char)(0x80 | (cp & 0x3f)));
  } else if (cp < 0x10000) {
    o->push_back((char)(0xe0 | (cp >> 12)));
    o->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
    o->push_back((char)(0x80 | (cp & 0x3f)));
  } else {
    o->push_back((char)(0xf0 | (cp >> 18)));
    o->push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
    o->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
    o->push_back((char)(0x80 | (cp & 0x3f)));
  }
}

std::string
to_utf8_(std::u32string_view s)
{
  std::string o;
  o.reserve(s.size());
  for (char32_t c : s) { put_utf8_(c, &o); }
  return o;
}

using U = std::u32string;

// Python's str.splitlines() boundaries (\r\n counts once).
bool
line_break_(char32_t c)
{
  switch (c) {
    case 0x0a: case 0x0b: case 0x0c: case 0x0d: case 0x1c: case 0x1d:
    case 0x1e: case 0x85: case 0x2028: case 0x2029:
      return true;
    default:
      return false;
  }
}

std::vector<U>
splitlines_(const U& s)
{
  std::vector<U> out;
  std::size_t i = 0, start = 0;
  while (i < s.size()) {
    if (line_break_(s[i])) {
      out.emplace_back(s.substr(start, i - start));
      if (s[i] == 0x0d && i + 1 < s.size() && s[i + 1] == 0x0a) { ++i; }
      ++i;
      start = i;
    } else {
      ++i;
    }
  }
  if (start < s.size()) { out.emplace_back(s.substr(start)); }
  return out;
}

U
strip_(const U& s, bool left, bool right)
{
  std::size_t a = 0, b = s.size();
  if (left) {
    while (a < b && py_isspace(s[a])) { ++a; }
  }
  if (right) {
    while (b > a && py_isspace(s[b - 1])) { --b; }
  }
  return s.substr(a, b - a);
}

U
replace_all_(const U& s, const U& from, const U& to)
{
  if (from.empty()) { return s; }
  U out;
  std::size_t i = 0;
  while (i < s.size()) {
    if (s.compare(i, from.size(), from) == 0) {
      out += to;
      i += from.size();
    } else {
      out.push_back(s[i]);
      ++i;
    }
  }
  return out;
}

// <|inner|> -> "a is b" (inner split once on whitespace) or inner.
U
rewrite_tag_(const U& raw)
{
  const U inner = strip_(raw, true, true);
  std::size_t i = 0;
  while (i < inner.size() && !py_isspace(inner[i])) { ++i; }
  if (i == inner.size()) { return inner; }
  std::size_t j = i;
  while (j < inner.size() && py_isspace(inner[j])) { ++j; }
  U out = inner.substr(0, i);
  out += U" is ";
  out += inner.substr(j);
  return out;
}

// re.sub(r"<\|([^|]*)\|>", rewrite, s)
U
special_tags_(const U& s)
{
  U out;
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    if (s[i] == U'<' && i + 1 < n && s[i + 1] == U'|') {
      std::size_t j = i + 2;
      while (j < n && s[j] != U'|') { ++j; }
      if (j + 1 < n && s[j + 1] == U'>') {
        out += rewrite_tag_(s.substr(i + 2, j - i - 2));
        i = j + 2;
        continue;
      }
    }
    out.push_back(s[i]);
    ++i;
  }
  return out;
}

// re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
U
strip_heading_(const U& s)
{
  std::size_t i = 0;
  while (i < s.size() && i < 3 && py_isspace(s[i])) { ++i; }
  std::size_t h = i;
  while (h < s.size() && s[h] == U'#') { ++h; }
  const std::size_t run = h - i;
  if (run < 1 || run > 6 || h >= s.size() || !py_isspace(s[h])) {
    return s;
  }
  while (h < s.size() && py_isspace(s[h])) { ++h; }
  return s.substr(h);
}

// re.sub(r"^\s*[<marks>]\s+", "", line)
U
strip_bullet_(const U& s, std::u32string_view marks)
{
  std::size_t i = 0;
  while (i < s.size() && py_isspace(s[i])) { ++i; }
  if (i >= s.size() || marks.find(s[i]) == std::u32string_view::npos) {
    return s;
  }
  std::size_t j = i + 1;
  if (j >= s.size() || !py_isspace(s[j])) { return s; }
  while (j < s.size() && py_isspace(s[j])) { ++j; }
  return s.substr(j);
}

// re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
U
unbold_once_(const U& s)
{
  U out;
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    if (s[i] == U'*' && i + 1 < n && s[i + 1] == U'*') {
      std::size_t j = i + 2;
      while (j < n && s[j] != U'*') { ++j; }
      if (j > i + 2 && j + 1 < n && s[j + 1] == U'*') {
        out += s.substr(i + 2, j - i - 2);
        i = j + 2;
        continue;
      }
    }
    out.push_back(s[i]);
    ++i;
  }
  return out;
}

// re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
U
unemph_(const U& s)
{
  U out;
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    if (s[i] == U'*' && (i == 0 || s[i - 1] != U'*')) {
      std::size_t j = i + 1;
      while (j < n && s[j] != U'*' && s[j] != U'\n') { ++j; }
      if (j > i + 1 && j < n && s[j] == U'*' &&
          (j + 1 >= n || s[j + 1] != U'*')) {
        out += s.substr(i + 1, j - i - 1);
        i = j + 1;
        continue;
      }
    }
    out.push_back(s[i]);
    ++i;
  }
  return out;
}

// re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
U
strip_rules_(const U& s)
{
  U out;
  const std::size_t n = s.size();
  std::size_t p = 0;
  auto rule = [](char32_t c) {
    return c == U'-' || c == U'*' || c == U'_';
  };
  while (p < n) {
    if (p == 0 || s[p - 1] == U'\n') {
      std::size_t a = p;
      while (a < n && py_isspace(s[a])) { ++a; }
      std::size_t b = a;
      while (b < n && rule(s[b])) { ++b; }
      if (b - a >= 3) {
        std::size_t c = b;
        while (c < n && py_isspace(s[c])) { ++c; }
        // $: the end, or just before a newline -- the LAST such place
        // the greedy \s* can back off to.
        std::size_t q = c + 1;
        bool hit = false;
        while (q-- > b) {
          if (q == n || s[q] == U'\n') {
            hit = true;
            break;
          }
        }
        if (hit) {
          p = q;
          continue;
        }
      }
    }
    out.push_back(s[p]);
    ++p;
  }
  return out;
}

// re.sub(r"\n{2,}", "\n", text)
U
collapse_newlines_(const U& s)
{
  U out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == U'\n' && !out.empty() && out.back() == U'\n' && i > 0 &&
        s[i - 1] == U'\n') {
      continue;
    }
    out.push_back(s[i]);
  }
  return out;
}

U
lower_u32_(const U& s)
{
  return to_u32_(unicode_lower(to_utf8_(s)));
}

std::string
nfc_(std::string_view s)
{
  bool ascii = true;
  for (char c : s) {
    if ((unsigned char)c >= 0x80) {
      ascii = false;
      break;
    }
  }
  if (ascii || s.empty()) { return std::string(s); }
  CFStringRef in = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
      (CFIndex)s.size(), kCFStringEncodingUTF8, false);
  if (in == nullptr) { return std::string(s); }
  CFMutableStringRef m = CFStringCreateMutableCopy(kCFAllocatorDefault, 0,
                                                   in);
  CFRelease(in);
  if (m == nullptr) { return std::string(s); }
  CFStringNormalize(m, kCFStringNormalizationFormC);
  const CFIndex len = CFStringGetLength(m);
  CFIndex need = 0;
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   nullptr, 0, &need);
  std::string out((std::size_t)need, '\0');
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   reinterpret_cast<UInt8*>(out.data()), need, nullptr);
  CFRelease(m);
  return out;
}

}  // namespace

bool
py_isspace(char32_t c)
{
  // str.isspace(): bidirectional class WS, B or S, or category Zs.
  if (c >= 0x09 && c <= 0x0d) { return true; }
  if (c >= 0x1c && c <= 0x20) { return true; }
  switch (c) {
    case 0x85: case 0xa0: case 0x1680: case 0x2028: case 0x2029:
    case 0x202f: case 0x205f: case 0x3000:
      return true;
    default:
      return c >= 0x2000 && c <= 0x200a;
  }
}

std::string
unicode_lower(std::string_view s)
{
  bool ascii = true;
  for (char c : s) {
    if ((unsigned char)c >= 0x80) {
      ascii = false;
      break;
    }
  }
  if (ascii) {
    std::string o(s);
    for (char& c : o) {
      if (c >= 'A' && c <= 'Z') { c = (char)(c - 'A' + 'a'); }
    }
    return o;
  }
  CFStringRef in = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
      (CFIndex)s.size(), kCFStringEncodingUTF8, false);
  if (in == nullptr) { return std::string(s); }
  CFMutableStringRef m = CFStringCreateMutableCopy(kCFAllocatorDefault, 0,
                                                   in);
  CFRelease(in);
  if (m == nullptr) { return std::string(s); }
  // A null locale is the Unicode default mapping (SpecialCasing included)
  // -- what Python's str.lower() applies.
  CFStringLowercase(m, nullptr);
  const CFIndex len = CFStringGetLength(m);
  CFIndex need = 0;
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   nullptr, 0, &need);
  std::string out((std::size_t)need, '\0');
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   reinterpret_cast<UInt8*>(out.data()), need, nullptr);
  CFRelease(m);
  return out;
}

std::string
clean_caption(std::string_view caption)
{
  const U text = special_tags_(to_u32_(caption));
  U joined;
  bool first = true;
  for (const U& raw : splitlines_(text)) {
    U line = strip_heading_(raw);
    line = strip_bullet_(line, U"*+-");
    line = strip_bullet_(line, U"*");
    while (line.find(U"**") != U::npos) {
      U next = unbold_once_(line);
      if (next == line) { break; }
      line = std::move(next);
    }
    line = unemph_(line);
    line = strip_(line, false, true);
    if (!first) { joined.push_back(U'\n'); }
    first = false;
    joined += line;
  }
  U t = strip_rules_(joined);
  t = replace_all_(t, U"• ", U"");
  t = replace_all_(t, U"    ", U"");
  return to_utf8_(collapse_newlines_(t));
}

std::string
normalize_lyrics(std::string_view lyrics)
{
  const U in = to_u32_(lyrics);
  // split("\n"): exactly the newline, unlike splitlines().
  std::vector<U> lines;
  {
    std::size_t start = 0;
    for (std::size_t i = 0; i <= in.size(); ++i) {
      if (i == in.size() || in[i] == U'\n') {
        lines.emplace_back(in.substr(start, i - start));
        start = i + 1;
      }
    }
  }
  U text;
  for (std::size_t li = 0; li < lines.size(); ++li) {
    const U& line = lines[li];
    // ^[ \t]*((?:\[[^\]]+\][ \t]*)+)
    std::size_t i = 0;
    while (i < line.size() && (line[i] == U' ' || line[i] == U'\t')) {
      ++i;
    }
    const std::size_t g0 = i;
    std::size_t end = g0;
    std::size_t k = g0;
    while (k < line.size() && line[k] == U'[') {
      std::size_t j = k + 1;
      while (j < line.size() && line[j] != U']') { ++j; }
      if (j >= line.size() || j == k + 1) { break; }
      j += 1;
      while (j < line.size() && (line[j] == U' ' || line[j] == U'\t')) {
        ++j;
      }
      end = j;
      k = j;
    }
    if (li > 0) { text.push_back(U'\n'); }
    if (end > g0) {
      text += strip_(line.substr(g0, end - g0), true, true);
    } else {
      text += line;
    }
  }
  text = replace_all_(text, U"] ", U"]\n");
  text = replace_all_(text, U" [", U"\n[");
  text = replace_all_(text, U" ^ ", U"\n");
  // re.sub(r"\[([^\]]+)\]", lower)
  U out;
  {
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
      if (text[i] == U'[') {
        std::size_t j = i + 1;
        while (j < n && text[j] != U']') { ++j; }
        if (j < n && j > i + 1) {
          out.push_back(U'[');
          out += lower_u32_(text.substr(i + 1, j - i - 1));
          out.push_back(U']');
          i = j + 1;
          continue;
        }
      }
      out.push_back(text[i]);
      ++i;
    }
  }
  return "[start]\n" + to_utf8_(out);
}

std::string
prompt_text(std::string_view caption, std::string_view lyrics)
{
  return "<|im_start|><|caption_start|>" + clean_caption(caption) +
         "<|caption_end|><|lyrics_start|>" + normalize_lyrics(lyrics) +
         "<|lyrics_end|><|im_end|><|audio_start|>";
}

bool
read_added_tokens(const std::string& path, AddedTokens* out,
                  std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::ifstream in(path, std::ios::binary);
  if (!in) { return fail("cannot read " + path); }
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string s = ss.str();
  // The array holds flat objects only, so its first unquoted ']' closes
  // it; the rest of the file (the 11 MB vocabulary) is never parsed.
  const std::size_t key = s.find("\"added_tokens\"");
  if (key == std::string::npos) { return fail("no added_tokens"); }
  const std::size_t a = s.find('[', key);
  if (a == std::string::npos) { return fail("malformed added_tokens"); }
  std::size_t b = a + 1;
  bool quoted = false;
  for (; b < s.size(); ++b) {
    const char c = s[b];
    if (quoted) {
      if (c == '\\') {
        ++b;
      } else if (c == '"') {
        quoted = false;
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ']') {
      break;
    }
  }
  if (b >= s.size()) { return fail("malformed added_tokens"); }
  FlexData fd;
  try {
    fd = FlexData::from_json(s.substr(a, b - a + 1));
  } catch (const std::exception& e) {
    return fail(std::string("added_tokens: ") + e.what());
  }
  if (!fd.is_array()) { return fail("added_tokens is not an array"); }
  out->clear();
  for (const FlexData& t : fd.as_array()) {
    if (!t.is_object()) { continue; }
    auto o = t.as_object();
    if (!o.contains("content") || !o.contains("id")) { continue; }
    out->emplace_back(std::string(o.at("content").as_string()),
                      (std::int32_t)o.at("id").as_int());
  }
  return true;
}

std::vector<std::int32_t>
encode_text(const Tokenizer& tok, const AddedTokens& added,
            std::string_view text)
{
  std::vector<std::int32_t> ids;
  auto run = [&](std::string_view seg) {
    if (seg.empty()) { return; }
    const auto v = tok.encode(nfc_(seg));
    ids.insert(ids.end(), v.begin(), v.end());
  };
  std::size_t i = 0, start = 0;
  while (i < text.size()) {
    // Leftmost match first, then the longest at that position.
    std::size_t best = 0;
    std::int32_t id = -1;
    for (const auto& [content, tid] : added) {
      if (content.size() > best &&
          text.compare(i, content.size(), content) == 0) {
        best = content.size();
        id = tid;
      }
    }
    if (best > 0) {
      run(text.substr(start, i - start));
      ids.push_back(id);
      i += best;
      start = i;
    } else {
      ++i;
    }
  }
  run(text.substr(start));
  return ids;
}

std::vector<std::int32_t>
unconditional_ids(const std::vector<std::int32_t>& ids)
{
  std::vector<std::int32_t> u = ids;
  for (std::size_t i = 1; i + 2 < u.size(); ++i) { u[i] = kAudioCfg; }
  return u;
}

int
latent_length(int frames)
{
  const double v = (double)frames * 44100.0 / 24000.0 * 960.0 / 512.0;
  return std::max(1, (int)v);
}

std::vector<int>
window_starts(int frames)
{
  std::vector<int> s;
  if (frames <= kChunkFrames) {
    s.push_back(0);
    return s;
  }
  for (int f = 0; f < frames - kChunkHop; f += kChunkHop) { s.push_back(f); }
  return s;
}

std::vector<int>
nearest_indices(int in, int out)
{
  std::vector<int> idx((std::size_t)std::max(out, 0));
  if (out == in) {
    for (int i = 0; i < out; ++i) { idx[(std::size_t)i] = i; }
    return idx;
  }
  if (out == 2 * in) {
    for (int i = 0; i < out; ++i) { idx[(std::size_t)i] = i >> 1; }
    return idx;
  }
  const float scale = (float)in / (float)out;
  for (int i = 0; i < out; ++i) {
    const float x = (float)i * scale;
    idx[(std::size_t)i] =
        std::min((int)std::floor(x), in - 1);
  }
  return idx;
}

std::pair<int, int>
kept_latents(int window, int windows, int latents)
{
  const int a = window == 0 ? 0 : kCropLeft;
  const int b = window == windows - 1 ? latents : latents - kCropRight;
  return {a, std::max(a, b)};
}

int
stitched_latents(int frames)
{
  if (frames <= 0) { return 0; }
  const std::vector<int> starts = window_starts(frames);
  const int n = (int)starts.size();
  int total = 0;
  for (int k = 0; k < n; ++k) {
    const int e = std::min(starts[(std::size_t)k] + kChunkFrames, frames);
    const auto kept = kept_latents(
        k, n, latent_length(e - starts[(std::size_t)k]));
    total += kept.second - kept.first;
  }
  return total;
}

std::vector<float>
flow_times(int steps)
{
  const int N = std::max(steps, 1);
  std::vector<float> t((std::size_t)N + 1);
  const double start = 1.0;
  const double stop = 1.0 / (double)N;
  const double step = N > 1 ? (stop - start) / (double)(N - 1) : 0.0;
  for (int k = 0; k < N; ++k) {
    double y = N > 1 ? (double)k * step + start : start;
    if (N > 1 && k == N - 1) { y = stop; }
    const float s = (float)y;
    t[(std::size_t)k] = 1.0f - s;
  }
  t[(std::size_t)N] = 1.0f;
  return t;
}

// ---- picks ----------------------------------------------------------------

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// The k-th largest of `v` (1-based), ties counted -- torch.topk's last
// value.
float
kth_largest_(const std::vector<float>& v, int k, std::vector<float>* tmp)
{
  *tmp = v;
  const int kk = std::min<int>(k, (int)tmp->size());
  std::nth_element(tmp->begin(), tmp->begin() + (kk - 1), tmp->end(),
                   std::greater<float>());
  return (*tmp)[(std::size_t)(kk - 1)];
}

// _sample_top_k's distribution: -inf (and NaN) become -1e9, the top 50
// are kept, the rest -inf.
void
top_k_mask_(std::vector<float>* s, std::vector<float>* tmp)
{
  for (float& x : *s) {
    if (std::isnan(x)) {
      x = -1e9f;
    } else if (std::isinf(x)) {
      x = x > 0 ? 1e9f : -1e9f;
    }
  }
  const float thr = kth_largest_(*s, kArTopK, tmp);
  for (float& x : *s) {
    if (x < thr) { x = kNegInf; }
  }
}

bool
semantic_allowed_(int i)
{
  // Window index 0 is <|audio_end|>; 1..4 are the four ids between it
  // and the codes; 5 + c is code c.
  return i == 0 || i >= kSemanticBase;
}

}  // namespace

void
FramePicker::semantic_scores(const float* cond, const float* uncond,
                             std::vector<float>* out, float cfg)
{
  const int n = kHeadRows;
  std::vector<float> c((std::size_t)n);
  out->assign((std::size_t)n, kNegInf);
  for (int i = 0; i < n; ++i) {
    c[(std::size_t)i] = semantic_allowed_(i) ? cond[i] : kNegInf;
  }
  std::vector<float> tmp;
  const float thr = kth_largest_(c, kArTopK, &tmp);
  for (int i = 0; i < n; ++i) {
    if (!semantic_allowed_(i) || c[(std::size_t)i] < thr) { continue; }
    const float u = uncond[i];
    (*out)[(std::size_t)i] = u + (cond[i] - u) * cfg;
  }
  top_k_mask_(out, &tmp);
}

void
FramePicker::residual_scores(const float* cond, const float* uncond, int n,
                             std::vector<float>* out, float cfg)
{
  out->resize((std::size_t)n);
  for (int i = 0; i < n; ++i) {
    const float u = uncond[i];
    (*out)[(std::size_t)i] = u + (cond[i] - u) * cfg;
  }
  std::vector<float> tmp;
  top_k_mask_(out, &tmp);
}

int
FramePicker::draw_(std::vector<float>& s)
{
  float m = kNegInf;
  for (float x : s) { m = std::max(m, x); }
  double sum = 0.0;
  for (float& x : s) {
    x = x == kNegInf ? 0.0f : std::exp(x - m);
    sum += (double)x;
  }
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double r = uni(_rng) * sum;
  double acc = 0.0;
  int last = -1;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] <= 0.0f) { continue; }
    last = (int)i;
    acc += (double)s[i];
    if (r < acc) { return (int)i; }
  }
  return std::max(last, 0);
}

int
FramePicker::pick_semantic(const float* cond, const float* uncond)
{
  semantic_scores(cond, uncond, &_s, _cfg);
  return draw_(_s);
}

int
FramePicker::pick_residual(const float* cond, const float* uncond)
{
  residual_scores(cond, uncond, kResidualCodes, &_s, _cfg);
  return draw_(_s);
}

}  // namespace vpipe::genai::music3
