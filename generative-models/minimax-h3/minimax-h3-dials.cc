#include "generative-models/minimax-h3/minimax-h3-dials.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

namespace {

bool
fail_(std::string* err, std::string m)
{
  if (err != nullptr) { *err = std::move(m); }
  return false;
}

float
bf16_f_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

bool
is_number_(const FlexData& v)
{
  return v.is_int() || v.is_uint() || v.is_real();
}

// A Python-falsy `time` (absent, null, false, "", {}, []) is no curve: the
// rack tests `r.get("time")`.
bool
falsy_(const FlexData& v)
{
  if (v.is_null()) { return true; }
  if (v.is_bool()) { return !v.as_bool(); }
  if (v.is_string()) { return v.as_string().empty(); }
  if (v.is_object()) { return v.as_object().size() == 0; }
  if (v.is_array()) { return v.as_array().size() == 0; }
  if (is_number_(v)) { return v.as_real(0.0) == 0.0; }
  return false;
}

bool
parse_curve_(const FlexData& t, DialCurve* c, std::string* err)
{
  if (!t.is_object()) {
    return fail_(err, "`time` must be an object ({\"type\": \"keys\", "
                      "\"points\": [[t, m], ...], ...})");
  }
  const auto o = t.as_object();
  const std::string type =
      o.contains("type") ? std::string(o.at("type").as_string("")) : "";
  if (type == "audio") {
    return fail_(err, "an audio-driven time curve (`\"type\": \"audio\"`, "
                      "M from a feature of the music) is not supported "
                      "here; give the curve as keys");
  }
  if (type != "keys") {
    return fail_(err, fmt("unknown time curve type '{}'; this reads "
                          "`\"type\": \"keys\"`", type)());
  }
  if (o.contains("steps")) {
    const FlexData s = o.at("steps");
    if (!s.is_int() && !s.is_uint() &&
        !(s.is_real() && s.as_real() == std::floor(s.as_real()))) {
      return fail_(err, "time curve `steps` (blend steps) must be a whole "
                        "number");
    }
    c->steps = (int)s.as_int(2);
  }
  if (c->steps < 1) {
    return fail_(err, "time curve `steps` must be at least 1");
  }
  const std::string audio =
      o.contains("audio") ? std::string(o.at("audio").as_string("")) : "";
  if (!audio.empty() && audio != "follow" && audio != "protect") {
    return fail_(err, "time curve `audio` must be 'follow' or 'protect'");
  }
  c->protect_audio = audio == "protect";
  const std::string unit =
      o.contains("unit") ? std::string(o.at("unit").as_string("")) : "";
  const std::string interp =
      o.contains("interp") ? std::string(o.at("interp").as_string("")) : "";
  if (!unit.empty() && unit != "seconds" && unit != "normalized") {
    return fail_(err, "time curve `unit` must be 'seconds' or "
                      "'normalized'");
  }
  if (!interp.empty() && interp != "linear" && interp != "step") {
    return fail_(err, "time curve `interp` must be 'linear' or 'step'");
  }
  c->seconds = unit == "seconds";
  c->step = interp == "step";
  if (o.contains("smooth")) {
    const FlexData s = o.at("smooth");
    if (!is_number_(s)) {
      return fail_(err, "time curve `smooth` must be a number from 0 to 1");
    }
    c->smooth = s.as_real(0.0);
  }
  if (!(c->smooth >= 0.0 && c->smooth <= 1.0)) {
    return fail_(err, fmt("time curve `smooth` (rate smoothing) is a number "
                          "from 0 to 1 (0 = linear), not {}", c->smooth)());
  }
  if (!o.contains("points") || !o.at("points").is_array()) {
    return fail_(err, "time curve keys need `points`: [[time, value], ...]");
  }
  const FlexData pts = o.at("points");
  const auto a = pts.as_array();
  c->points.clear();
  for (std::size_t i = 0; i < a.size(); ++i) {
    const FlexData p = a[i];
    if (!p.is_array() || p.as_array().size() != 2 ||
        !is_number_(p.as_array()[0]) || !is_number_(p.as_array()[1])) {
      return fail_(err, fmt("time curve point {} is not [time, value]",
                            i)());
    }
    const double x = p.as_array()[0].as_real(0.0);
    const double y = p.as_array()[1].as_real(0.0);
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return fail_(err, fmt("time curve point {} is not finite", i)());
    }
    if (!c->points.empty() && !(x > c->points.back().first)) {
      return fail_(err, "time curve keys need strictly increasing times");
    }
    c->points.emplace_back(x, y);
  }
  if (c->points.empty()) {
    return fail_(err, "time curve keys need at least one point");
  }
  return true;
}

bool
parse_row_(const FlexData& r, std::size_t i, DialRow* row, std::string* err)
{
  auto bad = [&](const std::string& m) {
    return fail_(err, fmt("dial row {}: {}", i, m)());
  };
  if (!r.is_object()) { return bad("not an object"); }
  const auto o = r.as_object();
  if (o.contains("input") && !o.at("input").is_null()) {
    return bad("a wired dial (`input`) has nothing to come from here; "
               "name its file in `dial`");
  }
  if (o.contains("hold") && o.at("hold").as_bool(false)) {
    return bad("hold rows (a direction projected out) are not supported "
               "here");
  }
  if (o.contains("envelope") && !falsy_(o.at("envelope"))) {
    return bad("step envelopes (`envelope`) are not supported here; give "
               "the row a time curve or none");
  }
  if (o.contains("blocks") && !falsy_(o.at("blocks"))) {
    return bad("per-block strengths (`blocks`) are not supported here");
  }
  if (!o.contains("dial") || !o.at("dial").is_string() ||
      o.at("dial").as_string().empty()) {
    return bad("needs `dial`, the dial file");
  }
  row->dial = std::string(o.at("dial").as_string());
  if (!o.contains("strength") || !is_number_(o.at("strength"))) {
    return bad("needs `strength`, a number (negative toward the dial's A "
               "side, positive toward its B side)");
  }
  row->strength = o.at("strength").as_real(0.0);
  if (!std::isfinite(row->strength)) { return bad("`strength` is not finite"); }
  if (o.contains("on")) { row->on = o.at("on").as_bool(true); }
  if (o.contains("solo")) { row->solo = o.at("solo").as_bool(false); }
  if (o.contains("time") && !falsy_(o.at("time"))) {
    row->timed = true;
    std::string why;
    if (!parse_curve_(o.at("time"), &row->time, &why)) { return bad(why); }
  }
  return true;
}

// MAINodes' envelopes.keyframes: linear keys whose rate ramps across each
// key over `smooth`/2 of each neighbouring interval -- C1, piecewise
// quadratic, no overshoot -- held flat outside the keys.
double
keyframes_(const std::vector<double>& x, const std::vector<double>& y,
           double smooth, double t)
{
  const std::size_t n = x.size();
  if (n == 0) { return 0.0; }
  if (n == 1) { return y[0]; }
  if (smooth == 0.0) {
    if (t <= x[0]) { return y[0]; }
    if (t >= x[n - 1]) { return y[n - 1]; }
    const std::size_t i =
        (std::size_t)(std::upper_bound(x.begin(), x.end(), t) - x.begin()) -
        1;
    return y[i] + (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
  }
  const std::size_t ni = n - 1;
  std::vector<double> h(ni), d(ni), m(n, 0.0), w(ni), k(ni);
  for (std::size_t i = 0; i < ni; ++i) {
    h[i] = x[i + 1] - x[i];
    d[i] = (y[i + 1] - y[i]) / h[i];
  }
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const double a = d[i - 1], b = d[i];
    m[i] = a * b > 0.0 ? 2.0 * a * b / (a + b) : 0.0;
  }
  for (std::size_t i = 0; i < ni; ++i) {
    w[i] = smooth * h[i] / 2.0;
    k[i] = (d[i] - smooth * (m[i] + m[i + 1]) / 4.0) / (1.0 - smooth / 2.0);
  }
  std::size_t i =
      (std::size_t)(std::upper_bound(x.begin(), x.end(), t) - x.begin());
  i = i == 0 ? 0 : std::min(i - 1, ni - 1);
  const double u = std::clamp(t - x[i], 0.0, h[i]);
  const double v = h[i] - u;
  const double m0 = m[i], m1 = m[i + 1], ki = k[i], wi = w[i];
  if (u < wi) {
    return y[i] + m0 * u + (ki - m0) * u * u / (2.0 * wi);
  }
  if (v < wi) {
    return y[i + 1] - m1 * v - (ki - m1) * v * v / (2.0 * wi);
  }
  return y[i] + (m0 + ki) * wi / 2.0 + ki * (u - wi);
}

std::string
strip_prefix_(std::string n)
{
  for (const char* p : {"model.diffusion_model.", "diffusion_model."}) {
    const std::string s(p);
    if (n.compare(0, s.size(), s) == 0) { return n.substr(s.size()); }
  }
  return n;
}

}  // namespace

bool
read_dial(const std::string& file, int hidden, Dial* out, std::string* err)
{
  namespace fs = std::filesystem;
  auto bad = [&](const std::string& m) {
    return fail_(err, fmt("dial '{}': {}", file, m)());
  };
  auto w = MetalLlamaWeights::open(file);
  if (!w.has_value()) { return bad("cannot be read as a safetensors file"); }
  const std::vector<std::string> names = w->tensor_names();
  if (names.size() != 1) {
    return bad(fmt("holds {} tensors; a dial is ONE "
                   "(diffusion_model.token_refiner.dials_shift or "
                   "diffusion_model.condition_proj.diff_b)",
                   names.size())());
  }
  const std::string key = strip_prefix_(names[0]);
  Dial d;
  d.path = file;
  if (key == "token_refiner.dials_shift") {
    d.seam = DialSeam::kPost;
  } else if (key == "condition_proj.diff_b") {
    d.seam = DialSeam::kPre;
  } else {
    return bad(fmt("its tensor '{}' is not a MiniMax H3 dial's "
                   "(token_refiner.dials_shift or condition_proj.diff_b)",
                   names[0])());
  }
  const auto& md = w->metadata();
  auto meta = [&](const char* k) -> std::string {
    auto it = md.find(k);
    return it == md.end() ? std::string() : it->second;
  };
  d.family = meta("model_family");
  if (d.family.empty()) { d.family = kDialFamily; }
  if (d.family != kDialFamily) {
    return bad(fmt("was made for {}, not {}", d.family, kDialFamily)());
  }
  d.name = meta("name");
  if (d.name.empty()) { d.name = fs::path(file).stem().string(); }
  const auto* info = w->info(names[0]);
  if (info == nullptr) { return bad("its tensor cannot be read"); }
  std::int64_t numel = 1;
  for (std::int64_t s : info->shape) { numel *= s; }
  if (numel != hidden) {
    return bad(fmt("is {} wide, and this DiT's hidden size is {}", numel,
                   hidden)());
  }
  const int esz = info->dtype == "F32" ? 4
                  : info->dtype == "BF16" || info->dtype == "F16" ? 2
                                                                  : 0;
  if (esz == 0 || info->nbytes != (std::uint64_t)numel * esz) {
    return bad(fmt("is {}; a dial is BF16, F16 or F32", info->dtype)());
  }
  std::vector<std::uint8_t> raw((std::size_t)info->nbytes);
  if (!w->pread_into(names[0], raw.data(), raw.size(), false)) {
    return bad("its tensor cannot be read");
  }
  d.vec.resize((std::size_t)numel);
  for (std::size_t i = 0; i < d.vec.size(); ++i) {
    if (esz == 4) {
      std::memcpy(&d.vec[i], &raw[i * 4], 4);
    } else {
      std::uint16_t b;
      std::memcpy(&b, &raw[i * 2], 2);
      _Float16 h;
      std::memcpy(&h, &b, 2);
      d.vec[i] = info->dtype == "BF16" ? bf16_f_(b) : (float)h;
    }
    if (!std::isfinite(d.vec[i])) {
      return bad(fmt("element {} is not finite", i)());
    }
  }
  *out = std::move(d);
  return true;
}

double
DialCurve::at(double t, double duration) const
{
  const double scale = seconds ? 1.0 : duration;
  std::vector<double> x, y;
  x.reserve(points.size());
  y.reserve(points.size());
  for (const auto& p : points) {
    x.push_back(p.first * scale);
    y.push_back(p.second);
  }
  if (x.empty()) { return 0.0; }
  if (step) {
    const std::size_t i =
        (std::size_t)(std::upper_bound(x.begin(), x.end(), t) - x.begin());
    return y[i == 0 ? 0 : i - 1];
  }
  return keyframes_(x, y, smooth, t);
}

std::string
DialCurve::key() const
{
  std::string s = fmt("{}|{}|{}|{}|{:.17g}|", steps,
                      protect_audio ? "protect" : "follow",
                      seconds ? "seconds" : "normalized",
                      step ? "step" : "linear", smooth)();
  for (const auto& p : points) {
    s += fmt("{:.17g},{:.17g};", p.first, p.second)();
  }
  return s;
}

bool
parse_dial_rows(const FlexData& v, std::vector<DialRow>* out,
                std::string* err)
{
  out->clear();
  if (v.is_null()) { return true; }
  if (v.is_string()) {
    const std::string s(v.as_string());
    if (s.find_first_not_of(" \t\r\n") == std::string::npos) { return true; }
    FlexData j;
    try {
      j = FlexData::from_json(s);
    } catch (const std::exception& e) {
      return fail_(err, fmt("dials: not JSON ({})", e.what())());
    }
    if (j.is_string()) {
      return fail_(err, "dials: a JSON string of a string");
    }
    return parse_dial_rows(j, out, err);
  }
  if (v.is_object()) {
    DialRow r;
    if (!parse_row_(v, 0, &r, err)) { return false; }
    out->push_back(std::move(r));
    return true;
  }
  if (!v.is_array()) {
    return fail_(err, "dials: an array of rows ({\"dial\": file, "
                      "\"strength\": s, \"time\": {...}})");
  }
  const auto a = v.as_array();
  for (std::size_t i = 0; i < a.size(); ++i) {
    DialRow r;
    if (!parse_row_(a[i], i, &r, err)) {
      out->clear();
      return false;
    }
    out->push_back(std::move(r));
  }
  return true;
}

std::vector<DialRow>
active_dial_rows(const std::vector<DialRow>& rows)
{
  std::vector<DialRow> on;
  bool solo = false;
  for (const DialRow& r : rows) {
    if (r.on && r.strength != 0.0) {
      on.push_back(r);
      solo = solo || r.solo;
    }
  }
  if (solo) {
    std::erase_if(on, [](const DialRow& r) { return !r.solo; });
  }
  return on;
}

FlexData
dial_rows_to_flex(const std::vector<DialRow>& rows)
{
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const DialRow& r : rows) {
    FlexData o = FlexData::make_object();
    auto ov = o.as_object();
    ov.insert_or_assign("dial", FlexData::make_string(r.dial));
    ov.insert_or_assign("strength", FlexData::make_real(r.strength));
    if (!r.on) { ov.insert_or_assign("on", FlexData::make_bool(false)); }
    if (r.solo) { ov.insert_or_assign("solo", FlexData::make_bool(true)); }
    if (r.timed) {
      const DialCurve& c = r.time;
      FlexData t = FlexData::make_object();
      auto tv = t.as_object();
      tv.insert_or_assign("type", FlexData::make_string("keys"));
      FlexData pts = FlexData::make_array();
      auto pv = pts.as_array();
      for (const auto& p : c.points) {
        FlexData q = FlexData::make_array();
        auto qv = q.as_array();
        qv.push_back(p.first);
        qv.push_back(p.second);
        pv.push_back(std::move(q));
      }
      tv.insert_or_assign("points", std::move(pts));
      tv.insert_or_assign("unit", FlexData::make_string(
                                      c.seconds ? "seconds" : "normalized"));
      tv.insert_or_assign("interp",
                          FlexData::make_string(c.step ? "step" : "linear"));
      tv.insert_or_assign("smooth", FlexData::make_real(c.smooth));
      tv.insert_or_assign("steps", FlexData::make_int(c.steps));
      tv.insert_or_assign("audio", FlexData::make_string(
                                       c.protect_audio ? "protect" : "follow"));
      ov.insert_or_assign("time", std::move(t));
    }
    a.push_back(std::move(o));
  }
  return arr;
}

void
dial_frame_weights(const DialCurve& c, int latent_frames, int audio_latents,
                   double fps, std::vector<float>* video,
                   std::vector<float>* audio)
{
  video->assign((std::size_t)std::max(0, latent_frames), 0.0f);
  audio->assign((std::size_t)std::max(0, audio_latents), 0.0f);
  if (!(fps > 0.0)) { return; }
  std::vector<int> starts(1, 0);
  for (int i = 0; i < latent_frames; ++i) {
    starts.push_back(starts.back() + kRopeFramesPerLatent[i % 5]);
  }
  const double duration = (double)starts.back() / fps;
  for (int i = 0; i < latent_frames; ++i) {
    double s = 0.0;
    for (int f = starts[(std::size_t)i]; f < starts[(std::size_t)i + 1]; ++f) {
      s += c.at((double)f / fps, duration);
    }
    (*video)[(std::size_t)i] =
        (float)(s / (double)(starts[(std::size_t)i + 1] -
                             starts[(std::size_t)i]));
  }
  if (c.protect_audio) { return; }
  for (int j = 0; j < audio_latents; ++j) {
    (*audio)[(std::size_t)j] =
        (float)c.at((double)j / (double)kAudioLatentsPerSecond, duration);
  }
}

bool
compile_dial_plan(const std::vector<DialRow>& rows,
                  const std::vector<Dial>& dials, DialPlan* out,
                  std::string* err)
{
  *out = DialPlan{};
  if (rows.size() != dials.size()) {
    return fail_(err, "dials: a row without its dial");
  }
  if (rows.empty()) { return true; }
  const std::size_t H = dials[0].vec.size();
  // The rack's sums: one row as `vec * strength` in f32; several summed
  // in double, then f32 (core.rack_post_shift).
  auto shift = [&](const std::vector<std::size_t>& idx) {
    std::vector<float> s(H, 0.0f);
    if (idx.size() == 1) {
      const float k = (float)rows[idx[0]].strength;
      const auto& v = dials[idx[0]].vec;
      for (std::size_t c = 0; c < H; ++c) { s[c] = v[c] * k; }
      return s;
    }
    std::vector<double> acc(H, 0.0);
    for (std::size_t i : idx) {
      const auto& v = dials[i].vec;
      for (std::size_t c = 0; c < H; ++c) {
        acc[c] += rows[i].strength * (double)v[c];
      }
    }
    for (std::size_t c = 0; c < H; ++c) { s[c] = (float)acc[c]; }
    return s;
  };
  std::vector<std::size_t> pre, post;
  std::map<std::string, std::size_t> by_curve;
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const DialRow& r = rows[i];
    const Dial& d = dials[i];
    if (d.vec.size() != H) {
      return fail_(err, "dials: rows of different widths");
    }
    const char* seam = d.seam == DialSeam::kPre ? "pre" : "post";
    if (!r.timed) {
      (d.seam == DialSeam::kPre ? pre : post).push_back(i);
      out->report += fmt("\n  {} ({}) {:+.4g}: constant", d.name, seam,
                         r.strength)();
      continue;
    }
    if (d.seam == DialSeam::kPre) {
      return fail_(err, fmt("dial '{}': time curves vary the post seam "
                            "only; a pre dial (condition_proj.diff_b) "
                            "cannot follow one -- give it no `time`",
                            d.name)());
    }
    const std::string k = r.time.key();
    auto it = by_curve.find(k);
    if (it == by_curve.end()) {
      it = by_curve.emplace(k, groups.size()).first;
      groups.emplace_back();
    }
    groups[it->second].push_back(i);
    out->report += fmt("\n  {} ({}) {:+.4g}: {} curve over {} key{}, "
                       "blend steps 0-{}, audio {}", d.name, seam,
                       r.strength, r.time.step ? "step" : "linear",
                       r.time.points.size(),
                       r.time.points.size() == 1 ? "" : "s",
                       r.time.steps - 1,
                       r.time.protect_audio ? "protect" : "follow")();
  }
  if (!pre.empty()) { out->pre = shift(pre); }
  if (!post.empty()) { out->post = shift(post); }
  for (const auto& g : groups) {
    DialPlan::Group grp;
    for (std::size_t i : g) {
      grp.label += (grp.label.empty() ? "" : " + ") + dials[i].name;
    }
    grp.shift = shift(g);
    grp.curve = rows[g[0]].time;
    out->groups.push_back(std::move(grp));
  }
  return true;
}

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
