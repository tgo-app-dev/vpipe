#include "generative-models/minimax-h3/p-h3-image.h"

#include "common/flex-data.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace vpipe {
namespace genai {
namespace ph3 {

namespace {

bool
read_json_(const std::filesystem::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  try {
    *out = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  return out->is_object();
}

}  // namespace

bool
is_root(const std::string& dir)
{
  FlexData fd;
  if (!read_json_(std::filesystem::path(dir) / kManifestFile, &fd)) {
    return false;
  }
  auto o = fd.as_object();
  if (!o.contains("_class_name")) { return false; }
  const FlexData c = o.at("_class_name");   // a VIEW: bind it
  return c.is_string() && c.as_string("") == kPipelineClass;
}

std::string
connector_dir(const std::string& root)
{
  if (root.empty() || !is_root(root)) { return {}; }
  const std::filesystem::path d =
      std::filesystem::path(root) / "text_connector";
  FlexData fd;
  if (!read_json_(d / "config.json", &fd)) { return {}; }
  auto o = fd.as_object();
  if (!o.contains("_class_name")) { return {}; }
  const FlexData c = o.at("_class_name");   // a VIEW: bind it
  return c.is_string() && c.as_string("") == kConnectorClass ? d.string()
                                                              : std::string();
}

bool
read_manifest(const std::string& root, Manifest* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const std::filesystem::path p = std::filesystem::path(root) / kManifestFile;
  FlexData fd;
  if (!read_json_(p, &fd)) { return fail("cannot read " + p.string()); }
  auto o = fd.as_object();
  {
    const FlexData c =
        o.contains("_class_name") ? o.at("_class_name") : FlexData();
    if (!c.is_string() || c.as_string("") != kPipelineClass) {
      return fail(p.string() + " is not a " + std::string(kPipelineClass) +
                  " manifest");
    }
  }
  Manifest m;
  if (o.contains("sample_sigmas")) {
    FlexData a = o.at("sample_sigmas");
    if (!a.is_array()) { return fail(p.string() + ": sample_sigmas"); }
    m.sample_sigmas.clear();
    // The reference reads them as float64 and runs them as float32.
    for (auto e : a.as_array()) {
      m.sample_sigmas.push_back((float)e.as_real(-1.0));
    }
  }
  const std::vector<float>& s = m.sample_sigmas;
  bool ok = s.size() >= 2 && s.front() <= 1.0f && s.back() == 0.0f;
  for (std::size_t i = 1; ok && i < s.size(); ++i) { ok = s[i] < s[i - 1]; }
  if (!ok) {
    return fail(p.string() + ": sample_sigmas is not a strictly decreasing "
                "grid from at most 1 to exactly 0");
  }
  auto px = [&](const char* k, std::int64_t* dst) {
    if (o.contains(k)) {
      const std::int64_t v = o.at(k).as_int(0);
      if (v > 0) { *dst = v; }
    }
  };
  px("reference_min_pixels", &m.reference_min_pixels);
  px("reference_max_pixels", &m.reference_max_pixels);
  px("canvas_min_pixels", &m.canvas_min_pixels);
  px("vision_pixels", &m.vision_pixels);
  if (m.reference_min_pixels > m.reference_max_pixels) {
    return fail(p.string() + ": reference_min_pixels is above "
                "reference_max_pixels");
  }
  *out = std::move(m);
  return true;
}

long long
py_round(double x)
{
  // Python 3's round() of a float: to nearest, ties to EVEN -- the IEEE
  // default rounding mode, which nearbyint honours.
  return (long long)std::nearbyint(x);
}

void
size_at_area(int width, int height, std::int64_t pixels, int multiple,
             int* out_w, int* out_h)
{
  const double area = (double)width * (double)height;
  if (width % multiple == 0 && height % multiple == 0 &&
      0.8 * (double)pixels <= area && area <= 1.2 * (double)pixels) {
    *out_w = width;
    *out_h = height;
    return;
  }
  const double ratio = (double)width / (double)height;
  const long long w =
      py_round(std::sqrt((double)pixels * ratio) / multiple) * multiple;
  const long long h =
      py_round(std::sqrt((double)pixels / ratio) / multiple) * multiple;
  *out_w = (int)std::max<long long>(multiple, w);
  *out_h = (int)std::max<long long>(multiple, h);
}

void
size_in_area_range(int width, int height, std::int64_t min_pixels,
                   std::int64_t max_pixels, int multiple, int* out_w,
                   int* out_h)
{
  const double area = (double)width * (double)height;
  const double clamped =
      std::min(std::max(area, (double)min_pixels), (double)max_pixels);
  const double scale = std::sqrt(clamped / area);
  auto side = [&](int v) {
    return (int)std::max<long long>(
        multiple, py_round((double)v * scale / multiple) * multiple);
  };
  *out_w = side(width);
  *out_h = side(height);
}

void
round_canvas(int height, int width, int multiple, int* out_h, int* out_w)
{
  auto side = [&](int v) {
    return (int)std::max<long long>(
        multiple, py_round((double)v / multiple) * multiple);
  };
  *out_h = side(height);
  *out_w = side(width);
}

std::string
edit_prompt(std::string_view prompt)
{
  std::string_view t = prompt;
  while (!t.empty() &&
         (t.front() == ' ' || t.front() == '\t' || t.front() == '\n' ||
          t.front() == '\r' || t.front() == '\f' || t.front() == '\v')) {
    t.remove_prefix(1);
  }
  constexpr std::string_view kEdit = "Edit <Picture 1>.";
  constexpr std::string_view kBrief = "subject_definitions:";
  if (t.substr(0, kEdit.size()) == kEdit ||
      t.substr(0, kBrief.size()) == kBrief) {
    return std::string(prompt);
  }
  return "Edit <Picture 1>. " + std::string(prompt);
}

}  // namespace ph3
}  // namespace genai
}  // namespace vpipe
