#include "stages/output-path.h"

#include "common/path-sandbox.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <string>

namespace vpipe::outpath {
namespace {

using std::size_t;
using std::string;
using std::string_view;

constexpr string_view kFlags = "-+ #0";
constexpr string_view kConvs = "diuxX";

bool
is_flag(char c)
{
  return kFlags.find(c) != string_view::npos;
}

bool
is_conv(char c)
{
  return kConvs.find(c) != string_view::npos;
}

// One parsed "%[flags][width]<conv>". `end` is one past the conversion
// letter. Only meaningful when `ok`.
struct Conv {
  bool   ok    = false;
  bool   left  = false;    // '-'
  bool   zero  = false;    // '0'
  bool   alt   = false;    // '#', hex only
  int    width = 0;
  char   conv  = 'd';
  size_t end   = 0;
};

// Parse a conversion starting at tmpl[i] == '%'. Does not handle "%%"
// or "%t" -- the caller takes those first.
Conv
parse_conv(string_view tmpl, size_t i)
{
  Conv c;
  size_t j = i + 1;
  for (; j < tmpl.size() && is_flag(tmpl[j]); ++j) {
    if (tmpl[j] == '-') { c.left = true; }
    if (tmpl[j] == '0') { c.zero = true; }
    if (tmpl[j] == '#') { c.alt  = true; }
  }
  // Clamped as it is read, so a pathological width in a config can
  // never reach the padding loop below as a length.
  for (; j < tmpl.size() && (std::isdigit((unsigned char)tmpl[j]) != 0);
       ++j) {
    c.width = std::min(c.width * 10 + (tmpl[j] - '0'), kMaxWidth + 1);
  }
  if (j >= tmpl.size() || !is_conv(tmpl[j])) { return c; }
  c.conv = tmpl[j];
  c.end  = j + 1;
  c.ok   = true;
  return c;
}

// Render `v` under `c`. Written out rather than handed to snprintf: the
// format would be user text, and the whole point of this file is that a
// path template is data.
string
render_int(const Conv& c, std::uint64_t v)
{
  const bool hex   = (c.conv == 'x' || c.conv == 'X');
  const char* digs = (c.conv == 'X') ? "0123456789ABCDEF"
                                     : "0123456789abcdef";
  const std::uint64_t base = hex ? 16u : 10u;
  string body;
  do {
    body.push_back(digs[v % base]);
    v /= base;
  } while (v != 0);
  std::reverse(body.begin(), body.end());
  if (hex && c.alt && body != "0") {
    body.insert(0, (c.conv == 'X') ? "0X" : "0x");
  }
  const size_t w = (size_t)std::max(0, std::min(c.width, kMaxWidth));
  if (body.size() >= w) { return body; }
  const size_t pad = w - body.size();
  if (c.left) { return body + string(pad, ' '); }
  // '0' is ignored with '-' (printf does the same); with the alternate
  // form the zeroes go after the 0x, which is what printf means by it.
  if (c.zero) {
    const size_t at = (hex && c.alt && body.rfind("0", 0) == 0
                       && body.size() > 1) ? 2 : 0;
    body.insert(at, string(pad, '0'));
    return body;
  }
  return string(pad, ' ') + body;
}

string
format_time(const string& spec, std::time_t when)
{
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &when);
#else
  localtime_r(&when, &tm);
#endif
  std::array<char, 512> buf{};
  const size_t n = std::strftime(buf.data(), buf.size(), spec.c_str(), &tm);
  // strftime returns 0 both for "did not fit" and for a format whose
  // result is legitimately empty. Either way an empty component is a
  // name the operator did not ask for, so fall back to the default
  // rather than silently dropping the token.
  if (n == 0) {
    const size_t d = std::strftime(buf.data(), buf.size(),
                                   kDefaultTimeFormat, &tm);
    return string(buf.data(), d);
  }
  return string(buf.data(), n);
}

// The "%t" or "%t{FMT}" at tmpl[i]. `end` is one past the token.
struct TimeTok {
  bool   ok      = false;
  bool   braced  = false;
  bool   closed  = false;
  string spec    = kDefaultTimeFormat;
  size_t end     = 0;
};

TimeTok
parse_time(string_view tmpl, size_t i)
{
  TimeTok t;
  if (i + 1 >= tmpl.size() || tmpl[i + 1] != 't') { return t; }
  t.ok  = true;
  t.end = i + 2;
  if (t.end >= tmpl.size() || tmpl[t.end] != '{') { return t; }
  t.braced = true;
  const size_t close = tmpl.find('}', t.end + 1);
  if (close == string_view::npos) { return t; }   // unterminated
  t.closed = true;
  t.spec.assign(tmpl, t.end + 1, close - t.end - 1);
  t.end = close + 1;
  return t;
}

// Insert `suffix` before the final extension of `path`, or append it
// when there is none. The extension test ignores a dot that belongs to
// a directory ("out.d/name"), which is the same rule save-image has
// always used.
string
with_suffix(const string& path, const string& suffix)
{
  const size_t dot   = path.find_last_of('.');
  const size_t slash = path.find_last_of("/\\");
  if (dot != string::npos
      && (slash == string::npos || dot > slash)) {
    return path.substr(0, dot) + suffix + path.substr(dot);
  }
  return path + suffix;
}

}   // namespace

string
check(string_view tmpl, Tokens* out)
{
  Tokens tok;
  for (size_t i = 0; i < tmpl.size(); ) {
    if (tmpl[i] != '%') { ++i; continue; }
    if (i + 1 < tmpl.size() && tmpl[i + 1] == '%') { i += 2; continue; }
    const TimeTok t = parse_time(tmpl, i);
    if (t.ok) {
      if (t.braced && !t.closed) {
        return "a %t{...} timestamp format is missing its closing '}'";
      }
      if (t.spec.find('/') != string::npos
          || t.spec.find('\\') != string::npos) {
        return "a %t{...} timestamp format may not contain a path "
               "separator -- it names one part of a filename, and the "
               "directory it would ask for is never created";
      }
      tok.time = true;
      i = t.end;
      continue;
    }
    const Conv c = parse_conv(tmpl, i);
    if (c.ok) {
      if (c.width > kMaxWidth) {
        return "a sequence conversion asks for a field wider than "
               + std::to_string(kMaxWidth) + " characters";
      }
      tok.sequence = true;
      i = c.end;
      continue;
    }
    ++i;                       // a bare '%': part of the name
  }
  if (out != nullptr) { *out = tok; }
  return {};
}

string
expand(string_view tmpl, std::uint64_t seq, int pad, std::time_t when)
{
  string out;
  out.reserve(tmpl.size() + 32);
  bool used_seq = false;
  for (size_t i = 0; i < tmpl.size(); ) {
    if (tmpl[i] != '%') { out.push_back(tmpl[i++]); continue; }
    if (i + 1 < tmpl.size() && tmpl[i + 1] == '%') {
      out.push_back('%');
      i += 2;
      continue;
    }
    const TimeTok t = parse_time(tmpl, i);
    if (t.ok && (!t.braced || t.closed)) {
      out += format_time(t.spec, when);
      i = t.end;
      continue;
    }
    const Conv c = parse_conv(tmpl, i);
    if (c.ok) {
      out += render_int(c, seq);
      used_seq = true;
      i = c.end;
      continue;
    }
    out.push_back(tmpl[i++]);
  }
  // No conversion of its own: file 0 keeps the name as written and each
  // later one is suffixed, so a stream cannot clobber itself.
  if (!used_seq && seq > 0) {
    Conv c;
    c.zero  = true;
    c.width = std::max(0, std::min(pad, kMaxWidth));
    out = with_suffix(out, "-" + render_int(c, seq));
  }
  return out;
}

string
SeqPicker::take(string_view tmpl, int pad)
{
  // ONE clock reading for the whole call. The scan below can cross a
  // second boundary, and a %t that changed mid-scan would have us
  // testing one name and writing another.
  const std::time_t now = std::time(nullptr);
  string path = expand(tmpl, _next, pad, now);
  // Nothing to stat on a network URL (rtmp://, udp://): save-video
  // accepts one, and "does it already exist" has no answer there.
  if (_skip && !is_network_url(tmpl)) {
    std::error_code ec;
    std::uint64_t probes = 0;
    while (std::filesystem::exists(path, ec)) {
      if (++probes > kMaxProbe) { return {}; }
      ++_next;
      path = expand(tmpl, _next, pad, now);
    }
  }
  ++_next;
  return path;
}

}   // namespace vpipe::outpath
