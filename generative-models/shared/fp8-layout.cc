#include "generative-models/shared/fp8-layout.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <cstring>
#include <string_view>

namespace vpipe::genai::fp8 {

namespace {

bool
ends_(std::string_view s, std::string_view suf)
{
  return s.size() >= suf.size() && s.substr(s.size() - suf.size()) == suf;
}

// The layer a `<layer>.weight` belongs to, "" for any other name.
std::string
layer_of_(const std::string& name)
{
  static constexpr std::string_view kW = ".weight";
  return ends_(name, kW) ? name.substr(0, name.size() - kW.size())
                         : std::string();
}

// The scale spellings, current first.
constexpr const char* kScaleSuffixes[] = {".weight_scale", ".scale_weight"};

// The format a comfy_quant record or the header metadata declares for
// `layer`, "" when neither does. *err on a record that cannot be read.
std::string
declared_format_(const MetalLlamaWeights& w, const std::string& layer,
                 std::string* err)
{
  const std::string cq = layer + ".comfy_quant";
  if (const auto* ti = w.info(cq); ti != nullptr) {
    std::string txt((std::size_t)ti->nbytes, '\0');
    if (!w.read_into(cq, txt.data(), txt.size())) {
      if (err != nullptr) { *err = "cannot read '" + cq + "'"; }
      return {};
    }
    try {
      const FlexData j = FlexData::from_json(txt);
      if (j.is_object() && j.as_object().contains("format")) {
        return std::string(j.as_object().at("format").as_string(""));
      }
    } catch (...) {
    }
    if (err != nullptr) { *err = "'" + cq + "' is not a JSON record"; }
    return {};
  }
  const auto& md = w.metadata();
  auto it = md.find("_quantization_metadata");
  if (it == md.end()) { return {}; }
  try {
    const FlexData j = FlexData::from_json(it->second);
    if (!j.is_object() || !j.as_object().contains("layers")) { return {}; }
    const FlexData layers = j.as_object().at("layers");
    if (!layers.is_object() || !layers.as_object().contains(layer)) {
      return {};
    }
    const FlexData one = layers.as_object().at(layer);
    if (one.is_object() && one.as_object().contains("format")) {
      return std::string(one.as_object().at("format").as_string(""));
    }
  } catch (...) {
    if (err != nullptr) {
      *err = "the header's _quantization_metadata is not JSON";
    }
  }
  return {};
}

Format
format_named_(const std::string& s)
{
  if (s == "float8_e4m3fn" || s == "float8_e4m3") { return Format::kE4M3; }
  if (s == "float8_e5m2") { return Format::kE5M2; }
  return Format::kNone;
}

std::size_t
numel_(const MetalLlamaWeights::TensorInfo& ti)
{
  std::size_t n = 1;
  for (auto d : ti.shape) { n *= (std::size_t)d; }
  return n;
}

bool
fail_(std::string* err, const std::string& m)
{
  if (err != nullptr) { *err = m; }
  return false;
}

}  // namespace

bool
Layout::scaled() const noexcept
{
  for (const auto& kv : weights) {
    if (!kv.second.scale.empty()) { return true; }
  }
  return false;
}

const Weight*
Layout::find(const std::string& name) const noexcept
{
  auto it = weights.find(name);
  return it == weights.end() ? nullptr : &it->second;
}

bool
is_aux_name(const std::string& name) noexcept
{
  for (const char* suf : {".weight_scale", ".scale_weight", ".input_scale",
                          ".scale_input", ".comfy_quant", ".weight_scale_2",
                          ".pre_quant_scale", ".scaled_fp8"}) {
    if (ends_(name, suf)) { return true; }
  }
  return name == "scaled_fp8";
}

bool
resolve(const MetalLlamaWeights& w, const std::string& name, Weight* out,
        std::string* err)
{
  if (err != nullptr) { err->clear(); }
  const auto* ti = w.info(name);
  if (ti == nullptr || is_aux_name(name)) { return false; }
  const std::string layer = layer_of_(name);
  Format f = format_of(ti->dtype);
  // What the encoding records say about this layer, when anything does.
  std::string declared;
  if (!layer.empty()) {
    std::string e;
    declared = declared_format_(w, layer, &e);
    if (!e.empty()) { return fail_(err, e); }
  }
  if (!declared.empty()) {
    const Format df = format_named_(declared);
    if (df == Format::kNone) {
      return fail_(err, fmt("'{}' is stored as '{}', which this build does "
                            "not read -- plain and scaled FP8 "
                            "(float8_e4m3fn / float8_e5m2) are",
                            name, declared)());
    }
    // The record wins over a U8 dtype (the legacy on-disk form); a real
    // FP8 dtype has to agree with it.
    if (f != Format::kNone && f != df) {
      return fail_(err, fmt("'{}' is {} on disk but its record says '{}'",
                            name, ti->dtype, declared)());
    }
    if (f == Format::kNone) {
      if (ti->dtype != "U8") {
        return fail_(err, fmt("'{}' is recorded as '{}' but stored as {}",
                              name, declared, ti->dtype)());
      }
      f = df;
    }
  }
  std::string scale;
  if (!layer.empty()) {
    for (const char* suf : kScaleSuffixes) {
      if (w.has(layer + suf)) { scale = layer + suf; break; }
    }
    for (const char* suf : {".weight_scale_2", ".pre_quant_scale"}) {
      if (w.has(layer + suf)) {
        return fail_(err, fmt("'{}' carries '{}', a second-stage scale "
                              "(NVFP4 / AWQ-style), which this build does "
                              "not read", name, layer + suf)());
      }
    }
  }
  if (f == Format::kNone) {
    // A scale beside a weight that is NOT FP8 means an encoding this
    // cannot see (int8, a 4-bit pack): refused, since dropping the scale
    // would load the codes as values.
    if (!scale.empty()) {
      return fail_(err, fmt("'{}' is {} with a scale ('{}') beside it -- "
                            "not an FP8 encoding this build reads",
                            name, ti->dtype, scale)());
    }
    return false;
  }
  Weight x;
  x.format = f;
  if (!scale.empty()) {
    const auto* si = w.info(scale);
    const std::size_t rows = ti->shape.empty() ? 0 : (std::size_t)ti->shape[0];
    const std::size_t n = si != nullptr ? numel_(*si) : 0;
    const bool col = si != nullptr &&
                     (si->shape.size() == 1 ||
                      (si->shape.size() == 2 && si->shape[1] == 1));
    if (si == nullptr ||
        (si->dtype != "F32" && si->dtype != "BF16" && si->dtype != "F16")) {
      return fail_(err, fmt("'{}': scale '{}' is not a float tensor", name,
                            scale)());
    }
    if (n == 1) {
      x.per_row = false;
    } else if (n == rows && col) {
      x.per_row = true;
    } else {
      return fail_(err, fmt("'{}': scale '{}' has {} values for {} rows -- "
                            "a BLOCK-scaled encoding, which this build does "
                            "not read (one scale, or one per row, is)", name,
                            scale, n, rows)());
    }
    x.scale = scale;
  }
  *out = x;
  return true;
}

bool
scan(const MetalLlamaWeights& w, Layout* out, std::string* err)
{
  Layout l;
  const std::vector<std::string> names = w.tensor_names();
  for (const std::string& n : names) {
    if (is_aux_name(n)) { continue; }
    Weight x;
    std::string e;
    if (resolve(w, n, &x, &e)) {
      l.weights.emplace(n, std::move(x));
    } else if (!e.empty()) {
      return fail_(err, e);
    }
  }
  // Every bookkeeping tensor must belong to something this decodes, or
  // the encoding is one it does not understand.
  for (const std::string& n : names) {
    if (!is_aux_name(n)) { continue; }
    if (ends_(n, ".weight_scale_2") || ends_(n, ".pre_quant_scale")) {
      return fail_(err, fmt("'{}' is a second-stage scale (NVFP4 / "
                            "AWQ-style), which this build does not read",
                            n)());
    }
    for (const char* suf : kScaleSuffixes) {
      if (!ends_(n, suf)) { continue; }
      const std::string wname =
          n.substr(0, n.size() - std::strlen(suf)) + ".weight";
      if (l.weights.count(wname) == 0 && w.has(wname)) {
        return fail_(err, fmt("'{}' scales '{}', which is not stored as "
                              "FP8", n, wname)());
      }
    }
    l.aux.insert(n);
  }
  *out = std::move(l);
  return true;
}

bool
read_scale(const MetalLlamaWeights& w, const Weight& x, std::size_t rows,
           std::vector<float>* out, std::string* err)
{
  out->clear();
  if (x.scale.empty()) { return true; }
  const auto* si = w.info(x.scale);
  if (si == nullptr) { return fail_(err, "missing '" + x.scale + "'"); }
  const std::size_t n = numel_(*si);
  if (x.per_row ? n != rows : n != 1) {
    return fail_(err, fmt("'{}' has {} values, expected {}", x.scale, n,
                          x.per_row ? rows : 1)());
  }
  std::vector<std::uint8_t> raw((std::size_t)si->nbytes);
  if (!w.read_into(x.scale, raw.data(), raw.size())) {
    return fail_(err, "cannot read '" + x.scale + "'");
  }
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (si->dtype == "F32") {
      std::memcpy(&(*out)[i], raw.data() + 4 * i, 4);
    } else if (si->dtype == "BF16") {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      const std::uint32_t u = (std::uint32_t)b << 16;
      std::memcpy(&(*out)[i], &u, 4);
    } else {
      _Float16 h;
      std::memcpy(&h, raw.data() + 2 * i, 2);
      (*out)[i] = (float)h;
    }
  }
  return true;
}

std::size_t
translate_torchao(MetalLlamaWeights& w)
{
  static constexpr std::string_view kQ = "._weight_qdata";
  static constexpr std::string_view kS = "._weight_scale";
  std::size_t n = 0;
  for (const std::string& name : w.tensor_names()) {
    if (ends_(name, kQ)) { ++n; }
  }
  if (n == 0) { return 0; }
  const bool ok = w.rename_tensors(
      [](const std::string& name, MetalLlamaWeights::TensorInfo&) {
        if (ends_(name, kQ)) {
          return name.substr(0, name.size() - kQ.size()) + ".weight";
        }
        if (ends_(name, kS)) {
          return name.substr(0, name.size() - kS.size()) + ".weight_scale";
        }
        return name;
      });
  return ok ? n : 0;
}

void
decode_f32(const std::uint8_t* codes, float* dst, std::size_t rows,
           std::size_t cols, Format f, const std::vector<float>& scale)
{
  const std::array<float, 256>& t = f32_table(f);
  const bool per_row = scale.size() > 1;
  const float s0 = scale.empty() ? 1.0f : scale[0];
  for (std::size_t r = 0; r < rows; ++r) {
    const float s = per_row ? scale[r] : s0;
    const std::uint8_t* c = codes + r * cols;
    float* d = dst + r * cols;
    for (std::size_t k = 0; k < cols; ++k) { d[k] = t[c[k]] * s; }
  }
}

void
decode_bf16(const std::uint8_t* codes, std::uint16_t* dst, std::size_t rows,
            std::size_t cols, Format f, const std::vector<float>& scale)
{
  if (scale.empty()) {
    decode_bf16(codes, dst, rows * cols, f);   // exact
    return;
  }
  const std::array<float, 256>& t = f32_table(f);
  const bool per_row = scale.size() > 1;
  for (std::size_t r = 0; r < rows; ++r) {
    const float s = per_row ? scale[r] : scale[0];
    const std::uint8_t* c = codes + r * cols;
    std::uint16_t* d = dst + r * cols;
    for (std::size_t k = 0; k < cols; ++k) {
      const float v = t[c[k]] * s;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      // Round to nearest even -- the rounding every converter in this
      // tree uses. NaN cannot arise: no weight code is NaN.
      d[k] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
  }
}

}  // namespace vpipe::genai::fp8
