#include "generative-models/minimax-h3/minimax-h3-image.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/p-h3-image.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace vpipe {
namespace genai {
namespace h3i {

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

std::string
str_(const FlexData& v)
{
  return v.is_string() ? std::string(v.as_string("")) : std::string();
}

}  // namespace

bool
is_root(const std::string& dir)
{
  if (dir.empty()) { return false; }
  FlexData fd;
  if (!read_json_(std::filesystem::path(dir) / "model_index.json", &fd)) {
    return false;
  }
  auto o = fd.as_object();
  if (!o.contains("_class_name")) { return false; }
  const FlexData c = o.at("_class_name");   // a VIEW: bind it
  return str_(c) == kPipelineClass;
}

bool
read_manifest(const std::string& root, Manifest* out, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out == nullptr) { return fail("null manifest"); }
  FlexData mi;
  const fs::path r(root);
  if (!read_json_(r / "model_index.json", &mi)) {
    return fail(root + ": no model_index.json");
  }
  auto o = mi.as_object();
  {
    const FlexData c =
        o.contains("_class_name") ? o.at("_class_name") : FlexData();
    if (str_(c) != kPipelineClass) {
      return fail(root + " is not a MiniMax-H3-Image root (model_index.json "
                         "does not name " + std::string(kPipelineClass) + ")");
    }
  }
  Manifest m;
  // The heads' file: named under `_minimax_h3`, the release's default
  // when it is not.
  std::string heads = kHeadsFile;
  if (o.contains("_minimax_h3")) {
    const FlexData h = o.at("_minimax_h3");
    if (h.is_object() && h.as_object().contains("pdd_fused_heads")) {
      const FlexData f = h.as_object().at("pdd_fused_heads");
      if (!str_(f).empty()) { heads = str_(f); }
    }
  }
  std::error_code ec;
  if (fs::exists(r / heads, ec)) { m.heads_file = (r / heads).string(); }
  // Where the text encoder lives: ["transformers", "<class>", {...}].
  if (o.contains("text_encoder")) {
    const FlexData te = o.at("text_encoder");
    if (te.is_array() && te.as_array().size() >= 3) {
      const FlexData spec = te.as_array().at(2);
      if (spec.is_object()) {
        auto so = spec.as_object();
        if (so.contains("pretrained_model_name_or_path")) {
          const FlexData v = so.at("pretrained_model_name_or_path");
          if (!str_(v).empty()) { m.text_encoder_repo = str_(v); }
        }
        if (so.contains("subfolder")) {
          const FlexData v = so.at("subfolder");
          if (!str_(v).empty()) { m.text_encoder_subfolder = str_(v); }
        }
      }
    }
  }
  // The grid. Absent, the release's own; present, it must be one.
  FlexData pc;
  if (read_json_(r / kScheduleFile, &pc)) {
    auto po = pc.as_object();
    if (po.contains("sigmas")) {
      const FlexData s = po.at("sigmas");
      if (!s.is_array()) { return fail(std::string(kScheduleFile) +
                                       ": `sigmas` is not an array"); }
      std::vector<double> g;
      auto a = s.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        g.push_back(a.at(i).as_real(-1.0));
      }
      m.sigmas = std::move(g);
    }
    if (po.contains("num_inference_steps")) {
      const int n = (int)po.at("num_inference_steps").as_int(-1);
      if (n != m.steps()) {
        return fail(std::string(kScheduleFile) + " has " + std::to_string(n) +
                    " steps for a grid of " +
                    std::to_string(m.sigmas.size()) + " points");
      }
    }
  }
  if (m.sigmas.size() < 2 || m.sigmas.front() != 1.0 ||
      m.sigmas.back() != 0.0) {
    return fail(std::string(kScheduleFile) +
                ": the grid must run from 1 to exactly 0");
  }
  for (std::size_t i = 1; i < m.sigmas.size(); ++i) {
    if (!(m.sigmas[i] < m.sigmas[i - 1])) {
      return fail(std::string(kScheduleFile) +
                  ": the grid is not strictly decreasing");
    }
  }
  *out = std::move(m);
  return true;
}

std::vector<float>
grid_as_run(const std::vector<double>& sigmas)
{
#pragma clang fp contract(off)
  std::vector<float> out;
  if (sigmas.empty()) { return out; }
  out.reserve(sigmas.size());
  for (std::size_t i = 0; i + 1 < sigmas.size(); ++i) {
    // torch.tensor([1.0 - s], float32): the difference in double, then
    // narrowed; `1.0 - timesteps` then in float32.
    const float t = (float)(1.0 - sigmas[i]);
    out.push_back(1.0f - t);
  }
  out.push_back(0.0f);
  return out;
}

void
canvas_size(int height, int width, std::int64_t min_area, int multiple,
            int* out_h, int* out_w)
{
  const double area = (double)width * (double)height;
  const double scale =
      std::max(1.0, std::sqrt((double)min_area / std::max(area, 1.0)));
  const long long m = std::max(1, multiple);
  *out_h = (int)std::max<long long>(
      m, ph3::py_round((double)height * scale / (double)m) * m);
  *out_w = (int)std::max<long long>(
      m, ph3::py_round((double)width * scale / (double)m) * m);
}

void
bound_short_edge(int width, int height, int min_edge, int max_edge,
                 int multiple, int* out_w, int* out_h)
{
  const int s = std::min(width, height);
  if (s >= min_edge && s <= max_edge) {
    *out_w = width;
    *out_h = height;
    return;
  }
  const double scale =
      (double)(s < min_edge ? min_edge : max_edge) / (double)std::max(s, 1);
  const long long m = std::max(1, multiple);
  *out_w = (int)std::max<long long>(
      m, ph3::py_round((double)width * scale / (double)m) * m);
  *out_h = (int)std::max<long long>(
      m, ph3::py_round((double)height * scale / (double)m) * m);
}

std::string
picture_label(int index)
{
  return "Picture " + std::to_string(index) + ": ";
}

}  // namespace h3i
}  // namespace genai
}  // namespace vpipe
