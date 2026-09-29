#include "generative-models/quantize/quant-source.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/vpipe-format.h"
#include "generative-models/krea2/krea2-native-checkpoint.h"
#include "generative-models/shared/fp8-layout.h"

#include <cstdint>
#include <filesystem>
#include <utility>

namespace vpipe::genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

std::optional<QuantSource>
QuantSource::open(const std::string& path, std::string* err)
{
  auto fail = [&](const std::string& m) -> std::optional<QuantSource> {
    if (err != nullptr) { *err = m; }
    return std::nullopt;
  };
  auto w = MetalLlamaWeights::open_model(path);
  if (!w.has_value()) {
    return fail("model-quantize: cannot open source model: " + path);
  }
  QuantSource s;
  const std::vector<std::string> names = w->tensor_names();

  // Which layout the FILE is in. open_model has already translated a
  // native Krea-2 file to the diffusers names, and does not say so (see
  // MetalLlamaWeights::rename_tensors) -- open() does not translate, so
  // the untranslated view answers. A file that still reads natively
  // through open_model is one the translation REFUSED, and the reason is
  // a tensor it could not place.
  {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_regular_file(path, ec) && !ec) {
      auto raw = MetalLlamaWeights::open(path);
      const std::optional<std::string> pre =
          raw.has_value() ? krea2::native_prefix(*raw) : std::nullopt;
      if (pre.has_value()) {
        if (krea2::native_prefix(*w).has_value()) {
          return fail(fmt(
              "model-quantize: '{}' holds a native Krea-2 DiT, but tensor "
              "'{}' has no diffusers name -- refusing rather than writing "
              "a checkpoint without it", path,
              krea2::untranslatable_tensor(*raw))());
        }
        s._layout = "krea2-native";
        for (const std::string& n : raw->tensor_names()) {
          if (n.compare(0, pre->size(), *pre) != 0) { ++s._n_ignored; }
        }
      }
    }
  }
  if (!s._layout.empty()) {
    std::string cerr;
    if (!krea2::diffusers_config(*w, s._config, &cerr, s._layout)) {
      return fail("model-quantize: " + path + ": " + cerr);
    }
    s._has_config = true;
  }

  // The FP8 encoding: which tensors are codes, with which scale, and
  // which tensors are bookkeeping (scales, markers, comfy_quant records)
  // that the output must not carry -- they are folded into the values.
  fp8::Layout f8;
  {
    std::string ferr;
    if (!fp8::scan(*w, &f8, &ferr)) {
      return fail("model-quantize: '" + path + "': " + ferr);
    }
  }
  for (const std::string& n : names) {
    if (f8.is_aux(n)) { continue; }
    const TensorInfo* ti = w->info(n);
    if (ti == nullptr) { continue; }
    Entry e;
    e.src  = n;
    e.info = *ti;
    if (const fp8::Weight* fw = f8.find(n); fw != nullptr) {
      std::size_t numel = 1;
      for (auto d : ti->shape) { numel *= (std::size_t)d; }
      e.f8   = fw->format;
      e.rows = ti->shape.empty() ? 1 : (std::size_t)ti->shape[0];
      e.cols = e.rows > 0 ? numel / e.rows : 0;
      std::string serr;
      if (!fp8::read_scale(*w, *fw, e.rows, &e.scale, &serr)) {
        return fail("model-quantize: '" + path + "': " + serr);
      }
      e.info.dtype  = "BF16";
      e.info.nbytes = numel * 2;
      ++s._n_fp8;
      if (!fw->scale.empty()) { ++s._n_scaled; }
    }
    s._map.emplace(n, std::move(e));
  }
  s._w = std::move(w);
  return s;
}

const QuantSource::TensorInfo*
QuantSource::info(const std::string& name) const
{
  auto it = _map.find(name);
  return it == _map.end() ? nullptr : &it->second.info;
}

std::vector<std::string>
QuantSource::tensor_names() const
{
  std::vector<std::string> out;
  out.reserve(_map.size());
  for (const auto& kv : _map) { out.push_back(kv.first); }
  return out;
}

SharedBuffer
QuantSource::load(const std::string& name, MetalCompute* mc) const
{
  auto it = _map.find(name);
  if (it == _map.end() || !_w.has_value()) { return {}; }
  const Entry& e = it->second;
  if (e.f8 == fp8::Format::kNone) { return _w->load(e.src, mc); }
  // Decode straight out of the raw read into the bf16 destination, the
  // scale folded in; the codes are dropped with the scratch.
  const std::size_t n = (std::size_t)(e.info.nbytes / 2);
  SharedBuffer raw = _w->load(e.src, mc);
  if (raw.empty() || raw.byte_size() < n || e.rows * e.cols != n) {
    return {};
  }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  fp8::decode_bf16(static_cast<const std::uint8_t*>(raw.contents()),
                   static_cast<std::uint16_t*>(out.contents()), e.rows,
                   e.cols, e.f8, e.scale);
  return out;
}

}  // namespace vpipe::genai
