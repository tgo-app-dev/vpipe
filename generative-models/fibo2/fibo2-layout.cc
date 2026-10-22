#include "generative-models/fibo2/fibo2-layout.h"

#include "common/flex-data.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace vpipe {
namespace genai {
namespace fibo2 {

std::string
prompt_template(const std::string& text)
{
  return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

bool
build_layout(int grid_h, int grid_w,
             const std::vector<std::pair<int, int>>& ctx_grids, int gist,
             int max_gist, Layout* out, std::string* err)
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (out == nullptr || grid_h <= 0 || grid_w <= 0 || gist <= 0 ||
      gist > max_gist) {
    return fail("fibo2 layout: degenerate geometry");
  }
  Layout l;
  l.grid_h = grid_h;
  l.grid_w = grid_w;
  l.img_tokens = grid_h * grid_w;
  l.ctx_grids = ctx_grids;
  for (const auto& g : ctx_grids) {
    if (g.first <= 0 || g.second <= 0) {
      return fail("fibo2 layout: a context image has no rows");
    }
    l.ctx_tokens += g.first * g.second;
  }
  l.gist = gist;
  const int n = l.joint_len();
  l.pos_t.resize((std::size_t)n);
  l.pos_h.resize((std::size_t)n);
  l.pos_w.resize((std::size_t)n);
  int r = 0;
  auto grid = [&](int plane, int gh, int gw) {
    for (int y = 0; y < gh; ++y) {
      for (int x = 0; x < gw; ++x, ++r) {
        l.pos_t[(std::size_t)r] = plane;
        l.pos_h[(std::size_t)r] = y;
        l.pos_w[(std::size_t)r] = x;
      }
    }
  };
  const int image_plane = max_gist + 1;
  grid(image_plane, grid_h, grid_w);
  for (std::size_t k = 0; k < ctx_grids.size(); ++k) {
    grid(image_plane + 1 + (int)k, ctx_grids[k].first, ctx_grids[k].second);
  }
  for (int k = 0; k < gist; ++k, ++r) {
    l.pos_t[(std::size_t)r] = 1 + k;
    l.pos_h[(std::size_t)r] = 0;
    l.pos_w[(std::size_t)r] = 0;
  }
  *out = std::move(l);
  return true;
}

int
gist_count(int text_len, int min_gist, int max_gist, int step, int min_len,
           int max_len)
{
  const double span = (double)(max_len - min_len);
  double ratio = span > 0.0 ? ((double)text_len - min_len) / span : 1.0;
  ratio = std::min(1.0, std::max(0.0, ratio));
  const double raw = min_gist + ratio * (max_gist - min_gist);
  const int q = (int)std::ceil(raw / step) * step;
  return std::max(min_gist, std::min(max_gist, q));
}

void
pack_latents(const float* z, int C, int H, int W, int p, float* rows)
{
  const int gh = H / p, gw = W / p, F = C * p * p;
  for (int y = 0; y < gh; ++y) {
    for (int x = 0; x < gw; ++x) {
      float* row = rows + ((std::size_t)y * gw + x) * F;
      for (int c = 0; c < C; ++c) {
        for (int dy = 0; dy < p; ++dy) {
          for (int dx = 0; dx < p; ++dx) {
            row[(c * p + dy) * p + dx] =
                z[((std::size_t)c * H + (y * p + dy)) * W + (x * p + dx)];
          }
        }
      }
    }
  }
}

void
unpack_latents(const float* rows, int C, int H, int W, int p, float* z)
{
  const int gh = H / p, gw = W / p, F = C * p * p;
  for (int y = 0; y < gh; ++y) {
    for (int x = 0; x < gw; ++x) {
      const float* row = rows + ((std::size_t)y * gw + x) * F;
      for (int c = 0; c < C; ++c) {
        for (int dy = 0; dy < p; ++dy) {
          for (int dx = 0; dx < p; ++dx) {
            z[((std::size_t)c * H + (y * p + dy)) * W + (x * p + dx)] =
                row[(c * p + dy) * p + dx];
          }
        }
      }
    }
  }
}

std::pair<int, int>
image_size(double aspect, long long area, int multiple)
{
  const int min_side = 4 * multiple;
  std::pair<int, int> best{0, 0};
  double best_crop = 0.0;
  long long best_dist = 0;
  bool have = false;
  for (long long w = min_side; w < 2 * area / min_side; w += multiple) {
    for (long long h = min_side; h < 2 * area / w; h += multiple) {
      const long long a = w * h;
      if (!(0.9 * (double)area <= (double)a &&
            (double)a <= 1.1 * (double)area)) {
        continue;
      }
      const double ratio = (double)w / (double)h;
      // round(crop, 9): the reference compares crops at 9 decimals.
      const double crop = std::nearbyint(
          (1.0 - std::min(ratio / aspect, aspect / ratio)) * 1e9) / 1e9;
      const long long dist = std::llabs(a - area);
      // min over (crop, |a - area|, -w): the wider wins the last tie.
      const bool better =
          !have || crop < best_crop ||
          (crop == best_crop &&
           (dist < best_dist ||
            (dist == best_dist && w > (long long)best.first)));
      if (better) {
        best = {(int)w, (int)h};
        best_crop = crop;
        best_dist = dist;
        have = true;
      }
    }
  }
  return best;
}

namespace {

// ---- the reference's JSON, restated -------------------------------------
//
// json.dumps(x, separators=(",", ":"), ensure_ascii=False): FlexData's
// writer matches it but for one thing -- a float with an integral value is
// "2.0" in Python and "2" there -- so the prompt is written here.
void
py_string(std::string_view s, std::string& out)
{
  out.push_back('"');
  for (char ch : s) {
    const auto uc = static_cast<unsigned char>(ch);
    switch (ch) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (uc < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)uc);
          out += buf;
        } else {
          out.push_back(ch);
        }
    }
  }
  out.push_back('"');
}

// Python's repr of a float: the shortest round-trip digits, fixed when
// the decimal exponent is in [-4, 16), else d.ddde+XX; ".0" on a fixed
// integral value.
void
py_float(double v, std::string& out)
{
  if (std::isnan(v)) { out += "NaN"; return; }
  if (std::isinf(v)) { out += v > 0 ? "Infinity" : "-Infinity"; return; }
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), v,
                         std::chars_format::scientific);
  std::string s(buf, r.ptr);
  bool neg = false;
  if (!s.empty() && s[0] == '-') { neg = true; s.erase(0, 1); }
  const auto epos = s.find('e');
  std::string mant = s.substr(0, epos);
  const int exp = std::atoi(s.c_str() + epos + 1);
  std::string digits;
  for (char c : mant) {
    if (c != '.') { digits.push_back(c); }
  }
  while (digits.size() > 1 && digits.back() == '0') { digits.pop_back(); }
  if (neg) { out.push_back('-'); }
  if (exp >= -4 && exp < 16) {
    if (exp < 0) {
      out += "0.";
      out.append((std::size_t)(-exp - 1), '0');
      out += digits;
    } else if ((int)digits.size() <= exp + 1) {
      out += digits;
      out.append((std::size_t)(exp + 1 - (int)digits.size()), '0');
      out += ".0";
    } else {
      out += digits.substr(0, (std::size_t)exp + 1);
      out.push_back('.');
      out += digits.substr((std::size_t)exp + 1);
    }
    return;
  }
  out.push_back(digits[0]);
  if (digits.size() > 1) {
    out.push_back('.');
    out += digits.substr(1);
  }
  char eb[16];
  std::snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
  out += eb;
}

void
py_dumps(const FlexData& v, std::string& out)
{
  switch (v.kind()) {
    case FlexKind::Null: out += "null"; break;
    case FlexKind::Bool: out += v.get_bool() ? "true" : "false"; break;
    case FlexKind::Int: out += std::to_string(v.get_int()); break;
    case FlexKind::Uint: out += std::to_string(v.get_uint()); break;
    case FlexKind::Real: py_float(v.get_real(), out); break;
    case FlexKind::String: py_string(v.get_string(), out); break;
    case FlexKind::Array: {
      out.push_back('[');
      bool first = true;
      for (const FlexData e : v.as_array()) {
        if (!first) { out.push_back(','); }
        first = false;
        py_dumps(e, out);
      }
      out.push_back(']');
      break;
    }
    case FlexKind::Object: {
      out.push_back('{');
      bool first = true;
      for (const auto& [k, e] : v.as_object()) {
        if (!first) { out.push_back(','); }
        first = false;
        py_string(k, out);
        out.push_back(':');
        py_dumps(e, out);
      }
      out.push_back('}');
      break;
    }
  }
}

// None, "", {}, [] or NaN -- what training removed from captions.
bool
empty_value(const FlexData& v)
{
  switch (v.kind()) {
    case FlexKind::Null: return true;
    case FlexKind::Real: return std::isnan(v.get_real());
    case FlexKind::String: return v.get_string().empty();
    case FlexKind::Array: return v.as_array().empty();
    case FlexKind::Object: return v.as_object().empty();
    default: return false;
  }
}

FlexData
clean(const FlexData& v)
{
  if (v.is_object()) {
    FlexData o = FlexData::make_object();
    for (const auto& [k, e] : v.as_object()) {
      FlexData c = clean(e);
      if (!empty_value(c)) { o.as_object().insert_or_assign(k, std::move(c)); }
    }
    return o;
  }
  if (v.is_array()) {
    FlexData a = FlexData::make_array();
    for (const FlexData e : v.as_array()) {
      FlexData c = clean(e);
      if (!empty_value(c)) { a.as_array().push_back(std::move(c)); }
    }
    return a;
  }
  return v;
}

bool
is_score(std::string_view k)
{
  return k == "aesthetic_score" || k == "preference_score";
}

// _parse_prompt: the prompt's caption (an object, a string, or none) and
// its edit instruction (a value, or none). `is_json` says whether the
// prompt was a JSON object at all.
struct Parsed {
  bool is_json = false;
  bool has_caption = false;
  FlexData caption;            // object or string
  bool has_instruction = false;
  FlexData instruction;
};

Parsed
parse_prompt(const std::string& prompt)
{
  Parsed p;
  FlexData j;
  try {
    j = FlexData::from_json(prompt);
  } catch (const std::exception&) {
    p.has_caption = true;
    p.caption = FlexData::make_string(prompt);
    return p;
  }
  if (!j.is_object()) {
    p.has_caption = true;
    p.caption = FlexData::make_string(prompt);
    return p;
  }
  p.is_json = true;
  FlexData cap = FlexData::make_object();
  for (const auto& [k, v] : j.as_object()) {
    if (k == "edit_instruction") {
      // pop(..., None): a null instruction is no instruction.
      p.has_instruction = !v.is_null();
      p.instruction = v;
    } else {
      cap.as_object().insert_or_assign(k, v);
    }
  }
  bool keys_sd_tr = false;
  {
    std::size_t n = 0;
    bool sd = false, tr = false;
    for (const auto& [k, v] : cap.as_object()) {
      ++n;
      sd |= k == "short_description";
      tr |= k == "text_render";
    }
    keys_sd_tr = n == 2 && sd && tr;
  }
  for (const auto& [k, v] : cap.as_object()) {
    if (k == "structured_caption") {
      if (!v.is_object()) {
        // A string caption (or anything else): kept when truthy.
        p.has_caption = !empty_value(v) &&
                        !(v.is_bool() && !v.get_bool()) &&
                        !(v.is_int() && v.get_int() == 0);
        p.caption = v;
        return p;
      }
      FlexData inner = v;
      cap = std::move(inner);
      // The key set test reads the caption it replaced, as there.
      std::size_t n = 0;
      bool sd = false, tr = false;
      for (const auto& [k2, v2] : cap.as_object()) {
        ++n;
        sd |= k2 == "short_description";
        tr |= k2 == "text_render";
      }
      keys_sd_tr = n == 2 && sd && tr;
      break;
    }
  }
  const FlexData cl = clean(cap);
  FlexData out = FlexData::make_object();
  for (const auto& [k, v] : cl.as_object()) {
    if (is_score(k)) { continue; }
    if (v.is_object()) {
      FlexData o = FlexData::make_object();
      for (const auto& [k2, v2] : v.as_object()) {
        if (!is_score(k2)) { o.as_object().insert_or_assign(k2, v2); }
      }
      if (o.as_object().empty()) { continue; }
      out.as_object().insert_or_assign(k, std::move(o));
    } else {
      out.as_object().insert_or_assign(k, v);
    }
  }
  if (keys_sd_tr) {
    FlexData o = FlexData::make_object();
    bool have_sd = false, have_tr = false;
    for (const auto& [k, v] : out.as_object()) {
      if (k == "short_description") {
        o.as_object().insert_or_assign("short_description", v);
        have_sd = true;
      }
    }
    if (!have_sd) {
      o.as_object().insert_or_assign("short_description",
                                     FlexData::make_string(""));
    }
    for (const auto& [k, v] : out.as_object()) {
      if (k == "text_render") {
        o.as_object().insert_or_assign("text_render", v);
        have_tr = true;
      }
    }
    if (!have_tr) {
      o.as_object().insert_or_assign("text_render", FlexData::make_array());
    }
    out = std::move(o);
  }
  p.has_caption = !out.as_object().empty();
  p.caption = std::move(out);
  return p;
}

// Each text_render entry with text gets the marker of its image.
void
mark_text_images(FlexData& caption, std::string_view marker)
{
  if (!caption.is_object()) { return; }
  for (const auto& [k, v] : caption.as_object()) {
    if (k != "text_render" || !v.is_array()) { continue; }
    FlexData arr = FlexData::make_array();
    for (FlexData e : v.as_array()) {
      if (e.is_object()) {
        bool has_text = false;
        for (const auto& [k2, v2] : e.as_object()) {
          if (k2 == "text" && v2.is_string() && !v2.get_string().empty()) {
            has_text = true;
          }
        }
        if (has_text) {
          e.as_object().insert_or_assign("image_tokens",
                                         FlexData::make_string(marker));
        }
      }
      arr.as_array().push_back(std::move(e));
    }
    caption.as_object().insert_or_assign("text_render", std::move(arr));
    return;
  }
}

}  // namespace

bool
edit_prompt(const std::string& prompt, int n_images, bool text_images,
            std::string* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (out == nullptr || n_images < 1 || n_images > kMaxImages) {
    return fail("fibo2: an edit takes 1 to " + std::to_string(kMaxImages) +
                " images");
  }
  // check_inputs: a JSON edit prompt names its instruction, and a
  // "structured_caption" key stands beside it alone.
  try {
    const FlexData j = FlexData::from_json(prompt);
    if (j.is_object()) {
      bool instr = false, sc = false, sc_ok = true;
      std::size_t n = 0;
      for (const auto& [k, v] : j.as_object()) {
        ++n;
        instr |= k == "edit_instruction";
        if (k == "structured_caption") {
          sc = true;
          sc_ok = v.is_object() || v.is_string();
        }
      }
      if (!instr) {
        return fail("fibo2: to edit, the prompt is the edit instruction, or "
                    "a JSON caption of the result with an \"edit_instruction\""
                    " key");
      }
      if (sc && (!sc_ok || n > 2)) {
        return fail("fibo2: an edit prompt with a \"structured_caption\" "
                    "holds the caption there, an object or a string, and "
                    "nothing else but the \"edit_instruction\"");
      }
    }
  } catch (const std::exception&) {
  }
  Parsed p = parse_prompt(prompt);
  FlexData caption;
  bool has_caption = false;
  FlexData instruction;
  if (!p.has_instruction) {
    instruction = FlexData::make_string(prompt);
  } else {
    caption = p.caption;
    has_caption = p.has_caption;
    instruction = p.instruction;
  }
  if (text_images && has_caption) { mark_text_images(caption, kVisionMarker); }
  FlexData edit = FlexData::make_object();
  if (n_images == 1) {
    edit.as_object().insert_or_assign("image",
                                      FlexData::make_string(kVisionMarker));
  } else {
    for (int k = 1; k <= n_images; ++k) {
      edit.as_object().insert_or_assign("image_" + std::to_string(k),
                                        FlexData::make_string(kVisionMarker));
    }
  }
  if (has_caption) {
    edit.as_object().insert_or_assign("structured_caption", caption);
  }
  edit.as_object().insert_or_assign("edit_instruction", instruction);
  std::string s;
  py_dumps(edit, s);
  *out = std::move(s);
  return true;
}

std::string
generation_prompt(const std::string& prompt)
{
  const Parsed p = parse_prompt(prompt);
  if (!p.is_json) { return prompt; }
  // A string caption (a "structured_caption" short description) keeps
  // the prompt as it was given.
  if (p.has_caption && p.caption.is_string()) { return prompt; }
  std::string s;
  py_dumps(p.has_caption ? p.caption : FlexData::make_object(), s);
  return s;
}

std::vector<std::string>
caption_texts(const std::string& prompt)
{
  std::vector<std::string> out;
  const Parsed p = parse_prompt(prompt);
  if (!p.has_caption || !p.caption.is_object()) { return out; }
  for (const auto& [k, v] : p.caption.as_object()) {
    if (k != "text_render" || !v.is_array()) { continue; }
    for (const FlexData e : v.as_array()) {
      if (!e.is_object()) { continue; }
      for (const auto& [k2, v2] : e.as_object()) {
        if (k2 == "text" && v2.is_string() && !v2.get_string().empty()) {
          out.emplace_back(v2.get_string());
        }
      }
    }
  }
  return out;
}

}  // namespace fibo2
}  // namespace genai
}  // namespace vpipe
