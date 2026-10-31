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
  _i8_s   = _lib.function("i8_dequant_scaled");
  _i8_rot = _lib.function("i8_convrot_dequant");
  _w4a8   = _lib.function("w4a8_convrot_dequant");
  return valid();
}

bool
Expander::valid() const noexcept
{
  return _e4m3.valid() && _e5m2.valid() && _e4m3_s.valid() &&
         _e5m2_s.valid();
}

bool
Expander::int8_valid() const noexcept
{
  return _i8_s.valid() && _i8_rot.valid();
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
  if (n == 0 || n > 0xFFFFFFFFull || dst.byte_size() < n * 2) {
    return false;
  }
  if (w.format == Format::kW4A8) {
    // Nibbles, two a byte; the int8 pass after the decode (a 1024-element
    // threadgroup, whole rotation groups -- 1 when unrotated).
    const int g = w.rot_group > 0 ? w.rot_group : 1;
    if (!w4a8_valid() || !w.scaled() || w.s_rel == nullptr ||
        w.qgroup <= 0 || K % 2 != 0 || K % (std::size_t)w.qgroup != 0 ||
        w.codes->byte_size() < n / 2 ||
        w.s_rel->byte_size() < n / (std::size_t)w.qgroup ||
        w.scale->byte_size() < N * sizeof(float) ||
        (w.codebook != nullptr && !w.codebook->empty() &&
         w.codebook->byte_size() < 16 * sizeof(float)) ||
        (w.rot_group > 0 &&
         (!convrot_group_ok(w.rot_group) || K % w.rot_group != 0))) {
      return false;
    }
    const bool cb = w.codebook != nullptr && !w.codebook->empty();
    enc.set_function(_w4a8);
    enc.set_buffer(0, *w.codes);
    enc.set_buffer(1, dst);
    enc.set_constant(2, (std::uint32_t)n);
    enc.set_buffer(3, *w.scale);
    enc.set_constant(4, (std::uint32_t)K);
    enc.set_constant(5, (std::uint32_t)g);
    enc.set_buffer(6, *w.s_rel);
    // The codebook slot is bound either way; without one it is not read.
    enc.set_buffer(7, cb ? *w.codebook : *w.scale);
    enc.set_constant(8, (std::uint32_t)w.qgroup);
    enc.set_constant(9, (std::uint32_t)(cb ? 1 : 0));
    enc.dispatch({(unsigned)((n + 1023) / 1024 * 256), 1, 1}, {256, 1, 1});
    return true;
  }
  if (w.codes->byte_size() < n) { return false; }
  const bool e5 = w.format == Format::kE5M2;
  if (w.format == Format::kI8) {
    // int8 is always scaled (fp8-layout.h); ConvRot whole groups a
    // threadgroup, a 1024 elements each, its group in a constant. The
    // kernels are optional (int8_valid).
    const std::size_t want = w.per_row ? N : 1;
    if (!int8_valid() || !w.scaled() ||
        w.scale->byte_size() < want * sizeof(float) ||
        (w.rot_group > 0 &&
         (!convrot_group_ok(w.rot_group) || K % w.rot_group != 0))) {
      return false;
    }
    enc.set_function(w.rot_group > 0 ? _i8_rot : _i8_s);
    enc.set_buffer(0, *w.codes);
    enc.set_buffer(1, dst);
    enc.set_constant(2, (std::uint32_t)n);
    enc.set_buffer(3, *w.scale);
    enc.set_constant(4, (std::uint32_t)(w.per_row ? K : n));
    if (w.rot_group > 0) {
      // Whole threadgroups of 256, one a chunk of 1024 elements.
      enc.set_constant(5, (std::uint32_t)w.rot_group);
      enc.dispatch({(unsigned)((n + 1023) / 1024 * 256), 1, 1},
                   {256, 1, 1});
    } else {
      enc.dispatch({(unsigned)((n + 3) / 4), 1, 1}, {256, 1, 1});
    }
    return true;
  }
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
  // int8 too: a model whose scan refused it never asks for one, and the
  // decode below carries its rotation.
  if (!resolve(w, name, x, &e, /*int8=*/true)) {
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

// A W4A8 tensor decoded on the host: `n` the LOGICAL element count.
bool
w4a8_host_(const MetalLlamaWeights& w, const std::string& name,
           const Weight& x, std::uint16_t* dst, std::size_t n,
           std::string* err)
{
  const auto* ti = w.info(name);
  const auto* ri = w.info(x.s_rel);
  if (ti == nullptr || ri == nullptr || ti->shape.size() != 2) {
    if (err != nullptr) { *err = "'" + name + "' is not a W4A8 tensor"; }
    return false;
  }
  const std::size_t rows = (std::size_t)ti->shape[0];
  const std::size_t cols = (std::size_t)ti->shape[1] * 2;
  if (rows * cols != n) {
    if (err != nullptr) {
      *err = fmt("'{}' holds {} values, not {}", name, rows * cols, n)();
    }
    return false;
  }
  std::vector<std::uint8_t> codes((std::size_t)ti->nbytes),
      sr((std::size_t)ri->nbytes);
  std::vector<float> cb, scale;
  if (!w.read_into(name, codes.data(), codes.size()) ||
      !w.read_into(x.s_rel, sr.data(), sr.size()) ||
      !read_scale(w, x, rows, &scale, err)) {
    if (err != nullptr && err->empty()) { *err = "cannot read " + name; }
    return false;
  }
  if (!x.codebook.empty()) {
    cb.resize(16);
    if (!w.read_into(x.codebook, cb.data(), 16 * sizeof(float))) {
      if (err != nullptr) { *err = "cannot read '" + x.codebook + "'"; }
      return false;
    }
  }
  decode_w4a8_bf16(codes.data(), sr.data(), cb.empty() ? nullptr : cb.data(),
                   scale, rows, cols, x.qgroup, x.rot_group, dst);
  return true;
}

}  // namespace

bool
decode_tensor_bf16(const MetalLlamaWeights& w, const std::string& name,
                   std::uint16_t* dst, std::size_t n, std::string* err)
{
  {
    Weight x;
    std::string e;
    if (resolve(w, name, &x, &e, /*int8=*/true) &&
        x.format == Format::kW4A8) {
      return w4a8_host_(w, name, x, dst, n, err);
    }
  }
  Weight x;
  std::vector<std::uint8_t> codes;
  std::vector<float> scale;
  std::size_t rows = 0;
  if (!host_codes_(w, name, n, &x, &codes, &scale, &rows, err)) {
    return false;
  }
  decode_bf16(codes.data(), dst, rows, rows > 0 ? n / rows : 0, x.format,
              scale, x.rot_group);
  return true;
}

bool
decode_tensor_f32(const MetalLlamaWeights& w, const std::string& name,
                  float* dst, std::size_t n, std::string* err)
{
  {
    Weight x;
    std::string e;
    if (resolve(w, name, &x, &e, /*int8=*/true) &&
        x.format == Format::kW4A8) {
      std::vector<std::uint16_t> b(n);
      if (!w4a8_host_(w, name, x, b.data(), n, err)) { return false; }
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = (std::uint32_t)b[i] << 16;
        std::memcpy(&dst[i], &u, sizeof(float));
      }
      return true;
    }
  }
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
              x.format, scale, x.rot_group);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t u = (std::uint32_t)b[i] << 16;
    std::memcpy(&dst[i], &u, sizeof(float));
  }
  return true;
}

}  // namespace vpipe::genai::fp8
