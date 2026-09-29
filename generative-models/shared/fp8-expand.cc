#include "generative-models/shared/fp8-expand.h"

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <cstdint>
#include <cstring>

namespace vpipe::genai::fp8 {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

bool
Expander::load(MetalCompute* mc)
{
  if (mc == nullptr) { return false; }
  _lib    = mc->load_library("affine_dequant_bf16");
  _e4m3   = _lib.function("fp8_e4m3_dequant");
  _e5m2   = _lib.function("fp8_e5m2_dequant");
  _e4m3_s = _lib.function("fp8_e4m3_dequant_scaled");
  _e5m2_s = _lib.function("fp8_e5m2_dequant_scaled");
  return valid();
}

bool
Expander::valid() const noexcept
{
  return _e4m3.valid() && _e5m2.valid() && _e4m3_s.valid() &&
         _e5m2_s.valid();
}

bool
Expander::encode(ComputeEncoder& enc, const Codes& w, const SharedBuffer& dst,
                 std::size_t N, std::size_t K) const
{
  if (!w.any() || !valid()) { return false; }
  const std::size_t n = N * K;
  // One launch expands the whole [N, K], four codes a thread, so the
  // count has to fit the kernels' 32-bit index: 4 G elements, far past
  // any single weight these models carry.
  if (n == 0 || n > 0xFFFFFFFFull || w.codes->byte_size() < n ||
      dst.byte_size() < n * 2) {
    return false;
  }
  const bool e5 = w.format == Format::kE5M2;
  if (!w.scaled()) {
    enc.set_function(e5 ? _e5m2 : _e4m3);
  } else {
    // Element e takes scale[e / div]: div = K gives one scale a row,
    // div = n the one scale to every element.
    const std::size_t want = w.per_row ? N : 1;
    if (w.scale->byte_size() < want * sizeof(float)) { return false; }
    enc.set_function(e5 ? _e5m2_s : _e4m3_s);
    enc.set_buffer(3, *w.scale);
    enc.set_constant(4, (std::uint32_t)(w.per_row ? K : n));
  }
  enc.set_buffer(0, *w.codes);
  enc.set_buffer(1, dst);
  enc.set_constant(2, (std::uint32_t)n);
  enc.dispatch({(unsigned)((n + 3) / 4), 1, 1}, {256, 1, 1});
  return true;
}

SharedBuffer
scale_buffer(const MetalLlamaWeights& w, const Weight& x, std::size_t rows,
             MetalCompute* mc, bool* ok)
{
  if (ok != nullptr) { *ok = true; }
  if (x.scale.empty()) { return {}; }
  std::vector<float> v;
  if (mc == nullptr || !read_scale(w, x, rows, &v, nullptr) || v.empty()) {
    if (ok != nullptr) { *ok = false; }
    return {};
  }
  SharedBuffer b = mc->make_shared_buffer(v.size() * sizeof(float));
  if (b.empty()) {
    if (ok != nullptr) { *ok = false; }
    return {};
  }
  std::memcpy(b.contents(), v.data(), v.size() * sizeof(float));
  return b;
}

namespace {

// The codes and scale of `name`, read for a host decode.
bool
host_codes_(const MetalLlamaWeights& w, const std::string& name,
            std::size_t n, Weight* x, std::vector<std::uint8_t>* codes,
            std::vector<float>* scale, std::size_t* rows, std::string* err)
{
  std::string e;
  if (!resolve(w, name, x, &e)) {
    if (err != nullptr) {
      *err = e.empty() ? "'" + name + "' is not an FP8 tensor" : e;
    }
    return false;
  }
  const auto* ti = w.info(name);
  std::size_t count = 1;
  for (auto d : ti->shape) { count *= (std::size_t)d; }
  if (count != n || (std::size_t)ti->nbytes != n) {
    if (err != nullptr) {
      *err = fmt("'{}' holds {} codes, not {}", name, count, n)();
    }
    return false;
  }
  *rows = ti->shape.empty() ? 1 : (std::size_t)ti->shape[0];
  codes->resize(n);
  if (!w.read_into(name, codes->data(), n)) {
    if (err != nullptr) { *err = "cannot read '" + name + "'"; }
    return false;
  }
  return read_scale(w, *x, *rows, scale, err);
}

}  // namespace

bool
decode_tensor_bf16(const MetalLlamaWeights& w, const std::string& name,
                   std::uint16_t* dst, std::size_t n, std::string* err)
{
  Weight x;
  std::vector<std::uint8_t> codes;
  std::vector<float> scale;
  std::size_t rows = 0;
  if (!host_codes_(w, name, n, &x, &codes, &scale, &rows, err)) {
    return false;
  }
  decode_bf16(codes.data(), dst, rows, rows > 0 ? n / rows : 0, x.format,
              scale);
  return true;
}

bool
decode_tensor_f32(const MetalLlamaWeights& w, const std::string& name,
                  float* dst, std::size_t n, std::string* err)
{
  Weight x;
  std::vector<std::uint8_t> codes;
  std::vector<float> scale;
  std::size_t rows = 0;
  if (!host_codes_(w, name, n, &x, &codes, &scale, &rows, err)) {
    return false;
  }
  // Through bf16 rather than straight to f32: a scaled product rounds
  // there first in the dense conversion of this file, and in the GPU
  // expansion, so an f32 reader has to see the same value or the two
  // stop agreeing to the bit. Unscaled codes are exact either way.
  std::vector<std::uint16_t> b(n);
  decode_bf16(codes.data(), b.data(), rows, rows > 0 ? n / rows : 0,
              x.format, scale);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t u = (std::uint32_t)b[i] << 16;
    std::memcpy(&dst[i], &u, sizeof(float));
  }
  return true;
}

}  // namespace vpipe::genai::fp8
