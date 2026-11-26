#include "generative-models/magi2/magi2-text.h"

#include "generative-models/tokenizer.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>

namespace vpipe {
namespace genai {
namespace magi2 {

namespace {

// ---- a JSON value that keeps its keys' order --------------------------------
//
// The release reads the caption with Python's json, whose dicts keep their
// keys in the file's order -- and one path writes it back out (json.dumps),
// so the order is part of the text the encoder sees. Numbers keep whether
// they were integers, as Python's str() of them depends on it.

struct JValue {
  enum Kind { kNull, kBool, kInt, kReal, kString, kArray, kObject };
  Kind kind = kNull;
  bool b = false;
  long long i = 0;
  double r = 0.0;
  std::string s;
  std::vector<JValue> a;
  std::vector<std::pair<std::string, JValue>> o;

  const JValue* get(const std::string& k) const
  {
    for (const auto& kv : o) {
      if (kv.first == k) { return &kv.second; }
    }
    return nullptr;
  }
  bool empty_like() const
  {
    return kind == kNull || (kind == kArray && a.empty()) ||
           (kind == kObject && o.empty());
  }
};

struct Parser {
  const std::string& t;
  std::size_t p = 0;
  bool ok = true;

  void ws()
  {
    while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' ||
                            t[p] == '\r')) {
      ++p;
    }
  }
  static void put_utf8(std::string& o, unsigned cp)
  {
    if (cp < 0x80) {
      o += (char)cp;
    } else if (cp < 0x800) {
      o += (char)(0xc0 | (cp >> 6));
      o += (char)(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
      o += (char)(0xe0 | (cp >> 12));
      o += (char)(0x80 | ((cp >> 6) & 0x3f));
      o += (char)(0x80 | (cp & 0x3f));
    } else {
      o += (char)(0xf0 | (cp >> 18));
      o += (char)(0x80 | ((cp >> 12) & 0x3f));
      o += (char)(0x80 | ((cp >> 6) & 0x3f));
      o += (char)(0x80 | (cp & 0x3f));
    }
  }
  bool hex4(unsigned* v)
  {
    if (p + 4 > t.size()) { return false; }
    unsigned x = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = t[p + k];
      x <<= 4;
      if (c >= '0' && c <= '9') { x |= (unsigned)(c - '0'); }
      else if (c >= 'a' && c <= 'f') { x |= (unsigned)(c - 'a' + 10); }
      else if (c >= 'A' && c <= 'F') { x |= (unsigned)(c - 'A' + 10); }
      else { return false; }
    }
    p += 4;
    *v = x;
    return true;
  }
  bool str(std::string* out)
  {
    if (p >= t.size() || t[p] != '"') { return false; }
    ++p;
    while (p < t.size()) {
      const char c = t[p++];
      if (c == '"') { return true; }
      if ((unsigned char)c < 0x20) { return false; }
      if (c != '\\') { *out += c; continue; }
      if (p >= t.size()) { return false; }
      const char e = t[p++];
      switch (e) {
        case '"': *out += '"'; break;
        case '\\': *out += '\\'; break;
        case '/': *out += '/'; break;
        case 'b': *out += '\b'; break;
        case 'f': *out += '\f'; break;
        case 'n': *out += '\n'; break;
        case 'r': *out += '\r'; break;
        case 't': *out += '\t'; break;
        case 'u': {
          unsigned cp = 0;
          if (!hex4(&cp)) { return false; }
          if (cp >= 0xd800 && cp < 0xdc00 && p + 6 <= t.size() &&
              t[p] == '\\' && t[p + 1] == 'u') {
            p += 2;
            unsigned lo = 0;
            if (!hex4(&lo)) { return false; }
            if (lo >= 0xdc00 && lo < 0xe000) {
              cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            } else {
              put_utf8(*out, cp);
              cp = lo;
            }
          }
          put_utf8(*out, cp);
          break;
        }
        default: return false;
      }
    }
    return false;
  }
  bool value(JValue* v)
  {
    ws();
    if (p >= t.size()) { return false; }
    const char c = t[p];
    if (c == '{') {
      ++p;
      v->kind = JValue::kObject;
      ws();
      if (p < t.size() && t[p] == '}') { ++p; return true; }
      while (true) {
        ws();
        std::string k;
        if (!str(&k)) { return false; }
        ws();
        if (p >= t.size() || t[p] != ':') { return false; }
        ++p;
        JValue x;
        if (!value(&x)) { return false; }
        // A repeated key keeps its first place and its last value.
        bool seen = false;
        for (auto& kv : v->o) {
          if (kv.first == k) { kv.second = std::move(x); seen = true; break; }
        }
        if (!seen) { v->o.emplace_back(std::move(k), std::move(x)); }
        ws();
        if (p < t.size() && t[p] == ',') { ++p; continue; }
        if (p < t.size() && t[p] == '}') { ++p; return true; }
        return false;
      }
    }
    if (c == '[') {
      ++p;
      v->kind = JValue::kArray;
      ws();
      if (p < t.size() && t[p] == ']') { ++p; return true; }
      while (true) {
        JValue x;
        if (!value(&x)) { return false; }
        v->a.push_back(std::move(x));
        ws();
        if (p < t.size() && t[p] == ',') { ++p; continue; }
        if (p < t.size() && t[p] == ']') { ++p; return true; }
        return false;
      }
    }
    if (c == '"') {
      v->kind = JValue::kString;
      return str(&v->s);
    }
    auto lit = [&](const char* w) {
      const std::size_t n = std::strlen(w);
      if (t.compare(p, n, w) == 0) { p += n; return true; }
      return false;
    };
    if (lit("true")) { v->kind = JValue::kBool; v->b = true; return true; }
    if (lit("false")) { v->kind = JValue::kBool; v->b = false; return true; }
    if (lit("null")) { v->kind = JValue::kNull; return true; }
    // A number: an int unless it has a fraction or an exponent.
    const std::size_t st = p;
    if (p < t.size() && t[p] == '-') { ++p; }
    bool real = false;
    while (p < t.size()) {
      const char d = t[p];
      if (d >= '0' && d <= '9') { ++p; continue; }
      if (d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') {
        real = true;
        ++p;
        continue;
      }
      break;
    }
    if (p == st) { return false; }
    const std::string num = t.substr(st, p - st);
    if (!real) {
      v->kind = JValue::kInt;
      auto r = std::from_chars(num.data(), num.data() + num.size(), v->i);
      if (r.ec != std::errc()) {
        // Out of range for 64 bits: Python keeps it exact; read as real.
        v->kind = JValue::kReal;
        v->r = std::strtod(num.c_str(), nullptr);
      }
      return true;
    }
    v->kind = JValue::kReal;
    v->r = std::strtod(num.c_str(), nullptr);
    return true;
  }
};

bool
parse_json(const std::string& text, JValue* out)
{
  Parser ps{text};
  if (!ps.value(out)) { return false; }
  ps.ws();
  return ps.p == text.size();
}

// Python's repr of a float: the shortest that round-trips, with ".0" on an
// integral value and Python's exponent form past 1e16 / under 1e-4.
std::string
py_float(double v)
{
  if (std::isnan(v)) { return "nan"; }
  if (std::isinf(v)) { return v > 0 ? "inf" : "-inf"; }
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, v);
  std::string s(buf, r.ptr);
  // to_chars' shortest form: "1e+16", "1e-05", "123.5", "10".
  const std::size_t e = s.find('e');
  const double a = std::fabs(v);
  if (e == std::string::npos) {
    if (a != 0.0 && (a >= 1e16 || a < 1e-4)) {
      std::snprintf(buf, sizeof buf, "%.17g", v);
      s = buf;
    } else if (s.find('.') == std::string::npos) {
      s += ".0";
    }
    return s;
  }
  // Python: mantissa as is ("1e+16" -> "1e+16"), exponent at least two
  // digits -- which to_chars already writes.
  return s;
}

// Python's str() of a JSON value (what _one_line does to a non-string).
std::string
py_str(const JValue& v, bool repr = false)
{
  switch (v.kind) {
    case JValue::kNull: return "None";
    case JValue::kBool: return v.b ? "True" : "False";
    case JValue::kInt: return std::to_string(v.i);
    case JValue::kReal: return py_float(v.r);
    case JValue::kString: {
      if (!repr) { return v.s; }
      // repr: single quotes unless the text holds one and no double.
      const bool dq = v.s.find('\'') != std::string::npos &&
                      v.s.find('"') == std::string::npos;
      const char q = dq ? '"' : '\'';
      std::string o(1, q);
      for (char c : v.s) {
        if (c == q || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') { o += "\\n"; }
        else if (c == '\r') { o += "\\r"; }
        else if (c == '\t') { o += "\\t"; }
        else { o += c; }
      }
      return o + q;
    }
    case JValue::kArray: {
      std::string o = "[";
      for (std::size_t i = 0; i < v.a.size(); ++i) {
        if (i) { o += ", "; }
        o += py_str(v.a[i], true);
      }
      return o + "]";
    }
    case JValue::kObject: {
      std::string o = "{";
      for (std::size_t i = 0; i < v.o.size(); ++i) {
        if (i) { o += ", "; }
        o += "'" + v.o[i].first + "': " + py_str(v.o[i].second, true);
      }
      return o + "}";
    }
  }
  return {};
}

// Python's json.dumps(v, ensure_ascii=False).
void
py_dumps(const JValue& v, std::string& o)
{
  switch (v.kind) {
    case JValue::kNull: o += "null"; return;
    case JValue::kBool: o += v.b ? "true" : "false"; return;
    case JValue::kInt: o += std::to_string(v.i); return;
    case JValue::kReal:
      o += std::isnan(v.r) ? "NaN" : std::isinf(v.r)
                                         ? (v.r > 0 ? "Infinity"
                                                    : "-Infinity")
                                         : py_float(v.r);
      return;
    case JValue::kString:
      o += '"';
      for (unsigned char c : v.s) {
        switch (c) {
          case '"': o += "\\\""; break;
          case '\\': o += "\\\\"; break;
          case '\n': o += "\\n"; break;
          case '\r': o += "\\r"; break;
          case '\t': o += "\\t"; break;
          case '\b': o += "\\b"; break;
          case '\f': o += "\\f"; break;
          default:
            if (c < 0x20) {
              char b[8];
              std::snprintf(b, sizeof b, "\\u%04x", c);
              o += b;
            } else {
              o += (char)c;
            }
        }
      }
      o += '"';
      return;
    case JValue::kArray:
      o += '[';
      for (std::size_t i = 0; i < v.a.size(); ++i) {
        if (i) { o += ", "; }
        py_dumps(v.a[i], o);
      }
      o += ']';
      return;
    case JValue::kObject:
      o += '{';
      for (std::size_t i = 0; i < v.o.size(); ++i) {
        if (i) { o += ", "; }
        JValue k;
        k.kind = JValue::kString;
        k.s = v.o[i].first;
        py_dumps(k, o);
        o += ": ";
        py_dumps(v.o[i].second, o);
      }
      o += '}';
      return;
  }
}

// strip_empty: drop None, [] and {} values from objects (after stripping
// them), None from arrays.
JValue
strip_empty(const JValue& v)
{
  if (v.kind == JValue::kObject) {
    JValue o;
    o.kind = JValue::kObject;
    for (const auto& kv : v.o) {
      JValue x = strip_empty(kv.second);
      if (!x.empty_like()) { o.o.emplace_back(kv.first, std::move(x)); }
    }
    return o;
  }
  if (v.kind == JValue::kArray) {
    JValue a;
    a.kind = JValue::kArray;
    for (const JValue& x : v.a) {
      if (x.kind != JValue::kNull) { a.a.push_back(strip_empty(x)); }
    }
    return a;
  }
  return v;
}

// Python's str.split() whitespace: one code point at a time.
bool
py_space(unsigned cp)
{
  return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) ||
         cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

// Decode one UTF-8 code point at s[i]; its length in *n (1 on a bad byte).
unsigned
cp_at(const std::string& s, std::size_t i, int* n)
{
  const unsigned char c = (unsigned char)s[i];
  if (c < 0x80) { *n = 1; return c; }
  int len = (c >= 0xf0) ? 4 : (c >= 0xe0) ? 3 : (c >= 0xc0) ? 2 : 1;
  if (len == 1 || i + (std::size_t)len > s.size()) { *n = 1; return c; }
  unsigned cp = c & (0xff >> (len + 1));
  for (int k = 1; k < len; ++k) {
    cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3f);
  }
  *n = len;
  return cp;
}

// _one_line: " ".join(str(v).split()); None -> "".
std::string
one_line(const JValue* v)
{
  if (v == nullptr || v->kind == JValue::kNull) { return {}; }
  const std::string s = py_str(*v);
  std::string o;
  std::size_t i = 0;
  bool in_word = false;
  while (i < s.size()) {
    int n = 1;
    const unsigned cp = cp_at(s, i, &n);
    if (py_space(cp)) {
      in_word = false;
    } else {
      if (!in_word && !o.empty()) { o += ' '; }
      o.append(s, i, (std::size_t)n);
      in_word = true;
    }
    i += (std::size_t)n;
  }
  return o;
}

const JValue&
empty_object()
{
  static const JValue e = [] {
    JValue v;
    v.kind = JValue::kObject;
    return v;
  }();
  return e;
}

const JValue&
as_dict(const JValue* v)
{
  return v != nullptr && v->kind == JValue::kObject ? *v : empty_object();
}

std::string
join(const std::vector<std::string>& xs, const char* sep)
{
  std::string o;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) { o += sep; }
    o += xs[i];
  }
  return o;
}

bool
has_text(const std::string& s)
{
  for (char c : s) {
    if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
          c == '\v')) {
      return true;
    }
  }
  return false;
}

// json_to_compact_markdown over a parsed, stripped object with a
// "global_layer".
std::string
compact_markdown(const JValue& obj)
{
  const JValue& gl = as_dict(obj.get("global_layer"));
  const JValue& dl = as_dict(obj.get("dynamic_layer"));
  const JValue* rl = obj.get("reference_layer");
  std::vector<std::string> lines;
  const std::string context = one_line(gl.get("context"));
  const std::string desc = one_line(gl.get("description"));
  if (!context.empty() || !desc.empty()) {
    lines.push_back(!context.empty() ? "context: " + context : "context");
    if (!desc.empty()) { lines.push_back(desc); }
  }
  {
    const JValue& ae = as_dict(gl.get("aesthetics"));
    std::vector<std::string> parts;
    const std::pair<const char*, const char*> keys[] = {
        {"style", "style"}, {"mood", "mood_atmosphere"},
        {"color", "color_scheme"}};
    for (const auto& [label, key] : keys) {
      const std::string v = one_line(ae.get(key));
      if (!v.empty()) { parts.push_back(std::string(label) + "=" + v); }
    }
    if (!parts.empty()) { lines.push_back("aesthetics: " + join(parts, "; ")); }
  }
  {
    const JValue& au = as_dict(gl.get("audio_baseline"));
    const JValue& dia = as_dict(au.get("dialogue"));
    std::vector<std::string> parts;
    const std::string lang = one_line(dia.get("language"));
    std::vector<std::string> spk;
    if (const JValue* st = dia.get("speaker_tags")) {
      if (st->kind == JValue::kArray) {
        for (const JValue& x : st->a) { spk.push_back(one_line(&x)); }
      } else if (st->kind != JValue::kNull) {
        spk.push_back(one_line(st));
      }
    }
    const std::string amb = one_line(au.get("ambience"));
    if (!lang.empty()) { parts.push_back("language=" + lang); }
    if (!spk.empty()) { parts.push_back("speakers=" + join(spk, ",")); }
    if (!amb.empty()) { parts.push_back("ambience=" + amb); }
    if (!parts.empty()) { lines.push_back("audio: " + join(parts, "; ")); }
  }
  auto nonempty = [](const JValue* v) {
    return v != nullptr && v->kind == JValue::kArray && !v->a.empty();
  };
  if (const JValue* subs = gl.get("alive_subjects_static"); nonempty(subs)) {
    lines.push_back("subjects:");
    for (const JValue& sb : subs->a) {
      if (sb.kind != JValue::kObject) { continue; }
      const std::string sid = one_line(sb.get("subject_id"));
      const std::string d = one_line(sb.get("description"));
      const std::string pos = one_line(sb.get("position"));
      const std::string ori = one_line(sb.get("orientation"));
      std::vector<std::string> attrs;
      const JValue& va = as_dict(sb.get("visual_attributes"));
      for (const char* k : {"gender", "age_group", "ethnicity", "clothing",
                            "appearance_details"}) {
        const JValue* x = va.get(k);
        if (x == nullptr || x->empty_like()) { continue; }
        const std::string v = one_line(x);
        if (!v.empty()) { attrs.push_back(std::string(k) + "=" + v); }
      }
      std::vector<std::string> head;
      if (!sid.empty()) { head.push_back(sid); }
      if (!d.empty()) { head.push_back(d); }
      std::string row = " - " + join(head, " | ");
      std::vector<std::string> ex;
      if (!pos.empty()) { ex.push_back(pos); }
      if (!ori.empty()) { ex.push_back(ori); }
      if (!ex.empty()) { row += " (" + join(ex, "; ") + ")"; }
      if (!attrs.empty()) { row += " :: " + join(attrs, "; "); }
      lines.push_back(row);
    }
  }
  if (const JValue* objs = gl.get("objects_static"); nonempty(objs)) {
    lines.push_back("objects:");
    for (const JValue& ob : objs->a) {
      if (ob.kind != JValue::kObject) { continue; }
      const std::string oid = one_line(ob.get("object_id"));
      const std::string d = one_line(ob.get("description"));
      const std::string sh = one_line(ob.get("shape_and_color"));
      const std::string pos = one_line(ob.get("position"));
      std::vector<std::string> head;
      if (!oid.empty()) { head.push_back(oid); }
      if (!d.empty()) { head.push_back(d); }
      std::string row = " - " + join(head, " | ");
      std::vector<std::string> det;
      if (!sh.empty()) { det.push_back(sh); }
      if (!pos.empty()) { det.push_back(pos); }
      if (!det.empty()) { row += " :: " + join(det, "; "); }
      lines.push_back(row);
    }
  }
  if (const JValue* segs = dl.get("timeline_segments"); nonempty(segs)) {
    lines.push_back("timeline:");
    for (const JValue& sg : segs->a) {
      if (sg.kind != JValue::kObject) { continue; }
      const JValue& bi = as_dict(sg.get("segment_basic_info"));
      const std::string ts = one_line(bi.get("timestamp_range"));
      const std::string d = one_line(bi.get("segment_description"));
      std::string head = !ts.empty() ? " - " + ts : " -";
      if (!d.empty()) { head += " " + d; }
      while (!head.empty() && (head.back() == ' ' || head.back() == '\t' ||
                               head.back() == '\n')) {
        head.pop_back();
      }
      lines.push_back(head);
      const JValue& sa = as_dict(sg.get("audio"));
      if (const JValue* dls = sa.get("dialogue_lines");
          dls != nullptr && dls->kind == JValue::kArray) {
        for (const JValue& dlg : dls->a) {
          if (dlg.kind != JValue::kObject) { continue; }
          const std::string sp = one_line(dlg.get("speaker"));
          const std::string tx = one_line(dlg.get("text"));
          const std::string tt = one_line(dlg.get("timestamp"));
          if (tx.empty()) { continue; }
          std::string ln = (!sp.empty() ? "   - dialogue " + sp + ": "
                                        : std::string("   - dialogue: ")) +
                           tx;
          if (!tt.empty()) { ln += " (" + tt + ")"; }
          lines.push_back(ln);
        }
      }
      if (const JValue* al = sg.get("alive_subjects");
          al != nullptr && al->kind == JValue::kArray) {
        for (const JValue& x : al->a) {
          if (x.kind != JValue::kObject) { continue; }
          const std::string sid = one_line(x.get("subject_id"));
          const JValue& act = as_dict(x.get("action"));
          std::vector<std::string> ps;
          for (const char* k :
               {"primary_action", "interaction", "facial_expression"}) {
            const std::string v = one_line(act.get(k));
            if (!v.empty()) { ps.push_back(v); }
          }
          if (!ps.empty()) {
            lines.push_back("   - action " + sid + ": " + join(ps, "; "));
          }
        }
      }
      if (const JValue* os = sg.get("objects");
          os != nullptr && os->kind == JValue::kArray) {
        for (const JValue& x : os->a) {
          if (x.kind != JValue::kObject) { continue; }
          const std::string oid = one_line(x.get("object_id"));
          const JValue& st = as_dict(x.get("dynamic_state"));
          std::vector<std::string> ps;
          for (const char* k : {"state_change", "motion_detail"}) {
            const std::string v = one_line(st.get(k));
            if (!v.empty()) { ps.push_back(v); }
          }
          if (!ps.empty()) {
            lines.push_back("   - objects " + oid + ": " + join(ps, "; "));
          }
        }
      }
    }
  }
  if (rl != nullptr && rl->kind == JValue::kArray) {
    for (const JValue& x : rl->a) {
      const std::string s = py_str(x);
      if (has_text(s)) { lines.push_back(s); }
    }
  }
  std::vector<std::string> keep;
  for (const std::string& l : lines) {
    if (has_text(l)) { keep.push_back(l); }
  }
  return join(keep, "\n");
}

// What the release's Split ISOLATES -- as its regex engine reads it, not
// as it was written. The class ends in \U00020000-\U0002FFFF, which
// Oniguruma (the tokenizers library's engine) does not take as an escape:
// it reads a literal "U00020000", a range "0-U" and more literals. So
// besides the BMP CJK blocks, every character from '0' to 'U' (0x30-0x55:
// the digits, ":;<=>?@" and the capitals A-U) is its own pre-token, and
// the supplementary CJK block is NOT -- MEASURED by probing the release's
// own pre-tokenizer at every code point below 0x30000. ("Bright" encodes
// as "B" + "right".) The model was trained on this tokenizer, so it is
// reproduced, not fixed.
bool
cjk(unsigned cp)
{
  return (cp >= 0x30 && cp <= 0x55) || (cp >= 0x1100 && cp <= 0x11ff) ||
         (cp >= 0x2e80 && cp <= 0xa4cf) || (cp >= 0xa840 && cp <= 0xd7af) ||
         (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0xfe30 && cp <= 0xfe4f) ||
         (cp >= 0xff65 && cp <= 0xffdc);
}

void
encode_plain(const Tokenizer& tok, const std::string& text,
             std::vector<std::int32_t>* out)
{
  // Runs of non-CJK text through the tokenizer's own pre-tokenizer, each
  // CJK character on its own.
  std::size_t run = 0, i = 0;
  auto flush = [&](std::size_t end) {
    if (end > run) {
      const auto ids = tok.encode(std::string_view(text).substr(run,
                                                               end - run));
      out->insert(out->end(), ids.begin(), ids.end());
    }
  };
  while (i < text.size()) {
    int n = 1;
    const unsigned cp = cp_at(text, i, &n);
    if (cjk(cp)) {
      flush(i);
      const auto ids =
          tok.encode(std::string_view(text).substr(i, (std::size_t)n));
      out->insert(out->end(), ids.begin(), ids.end());
      run = i + (std::size_t)n;
    }
    i += (std::size_t)n;
  }
  flush(text.size());
}

}  // namespace

const std::string&
negative_prompt()
{
  static const std::string s =
      "Bright tones, overexposed, static, blurred details, subtitles, "
      "style, works, paintings, images, static, overall gray, worst "
      "quality, low quality, JPEG compression residue, ugly, incomplete, "
      "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
      "disfigured, misshapen limbs, fused fingers, still picture, messy "
      "background, three legs, many people in the background, walking "
      "backwards"
      ", low quality, worst quality, poor quality, noise, background noise, "
      "hiss, hum, buzz, crackle, static, compression artifacts, MP3 "
      "artifacts, digital clipping, distortion, muffled, muddy, unclear, "
      "echo, reverb, room echo, over-reverberated, hollow sound, distant, "
      "washed out, harsh, shrill, piercing, grating, tinny, thin sound, "
      "boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring "
      "transition, sudden silence, looping artifact, music, instrumental, "
      "sirens, alarms, crowd noise, unrelated sound effects, chaotic, "
      "disorganized, messy, cheap sound"
      ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, "
      "mechanical, monotone, flat intonation, undynamic, boring, reading "
      "from a script, AI voice, synthetic, text-to-speech, TTS, insincere, "
      "fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, "
      "cringey, hesitant, unconfident, tired, weak voice, stuttering, "
      "stammering, mumbling, slurred speech, mispronounced, bad "
      "articulation, lisp, vocal fry, creaky voice, mouth clicks, lip "
      "smacks, wet mouth sounds, heavy breathing, audible inhales, "
      "plosives, p-pops, coughing, clearing throat, sneezing, speaking too "
      "fast, rushed, speaking too slow, dragged out, unnatural pauses, "
      "awkward silence, choppy, disconnected, multiple speakers, two "
      "voices, background talking, out of tune, off-key, autotune "
      "artifacts";
  return s;
}

std::string
normalize_prompt(const std::string& prompt)
{
  JValue v;
  if (!parse_json(prompt, &v)) { return prompt; }
  if (v.kind != JValue::kObject && v.kind != JValue::kArray) {
    return prompt;
  }
  const JValue s = strip_empty(v);
  if (s.kind != JValue::kObject || s.get("global_layer") == nullptr) {
    return prompt;
  }
  return compact_markdown(s);
}

std::string
with_figure_reference(const std::string& prompt)
{
  static const char* kLine = "The first frame refers to <Figure 1>";
  JValue v;
  if (parse_json(prompt, &v) && v.kind == JValue::kObject) {
    JValue arr;
    arr.kind = JValue::kArray;
    JValue line;
    line.kind = JValue::kString;
    line.s = kLine;
    arr.a.push_back(std::move(line));
    bool seen = false;
    for (auto& kv : v.o) {
      if (kv.first == "reference_layer") { kv.second = arr; seen = true; }
    }
    if (!seen) { v.o.emplace_back("reference_layer", std::move(arr)); }
    std::string out;
    py_dumps(v, out);
    return out;
  }
  return prompt + "reference_layer:" + kLine;
}

std::vector<std::int32_t>
encode(const Tokenizer& tok, const std::string& text)
{
  // The added tokens the HF tokenizer splits out of raw text before its
  // pre-tokenizer runs -- those this vocabulary has.
  static const char* kSpecials[] = {"<|endoftext|>", "<|im_start|>",
                                    "<|im_end|>", "<|vision_start|>",
                                    "<|vision_end|>", "<|image_pad|>",
                                    "<|video_pad|>", "<think>", "</think>"};
  std::vector<std::pair<std::string, std::int32_t>> marks;
  for (const char* m : kSpecials) {
    const std::int32_t id = tok.special_token_id(m);
    if (id >= 0) { marks.emplace_back(m, id); }
  }
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    int which = -1;
    for (int k = 0; k < (int)marks.size(); ++k) {
      const std::size_t f = text.find(marks[(std::size_t)k].first, pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f;
        which = k;
      }
    }
    if (which < 0) {
      encode_plain(tok, text.substr(pos), &out);
      break;
    }
    if (best > pos) { encode_plain(tok, text.substr(pos, best - pos), &out); }
    out.push_back(marks[(std::size_t)which].second);
    pos = best + marks[(std::size_t)which].first.size();
  }
  if ((int)out.size() > kMaxTextTokens) { out.resize(kMaxTextTokens); }
  return out;
}

std::vector<int>
target_span(const Tokenizer& tok, const std::vector<std::int32_t>& ids,
            const std::string& text, const std::string& target)
{
  std::vector<int> out;
  const std::size_t st = text.find(target);
  if (st == std::string::npos) { return out; }
  const std::size_t en = st + target.size();
  auto sd = tok.make_stream_decoder();
  std::size_t emitted = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::size_t a = emitted + sd.pending.size();
    emitted += tok.step(sd, ids[i]).size();
    const std::size_t b = emitted + sd.pending.size();
    if (std::max(a, st) < std::min(b, en)) { out.push_back((int)i); }
  }
  return out;
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe
