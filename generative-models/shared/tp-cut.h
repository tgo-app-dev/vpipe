#ifndef VPIPE_GENERATIVE_MODELS_SHARED_TP_CUT_H
#define VPIPE_GENERATIVE_MODELS_SHARED_TP_CUT_H

// A RANK'S SHARE OF A WEIGHT (tensor-parallel.h): `rows` -- [first, end)
// each -- of an [N, rb]-byte buffer, each row's bytes [c0, c1) (c1 0: to
// its end), copied in order into a buffer of their own. An output cut
// takes rows, an input cut each row's columns; a codes / scales / biases
// triple is cut alike at its own bytes a column. Empty when the buffer
// is not N rows or a range falls outside it.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

inline metal_compute::SharedBuffer
tp_cut(metal_compute::MetalCompute* mc, const metal_compute::SharedBuffer& src,
       int N, const std::vector<std::pair<int, int>>& rows, std::size_t c0,
       std::size_t c1)
{
  if (src.empty() || N <= 0 || src.byte_size() % (std::size_t)N != 0) {
    return {};
  }
  const std::size_t rb = src.byte_size() / (std::size_t)N;
  if (c1 == 0) { c1 = rb; }
  if (c0 >= c1 || c1 > rb) { return {}; }
  std::size_t n = 0;
  for (const auto& [a, b] : rows) {
    if (a < 0 || b > N || a >= b) { return {}; }
    n += (std::size_t)(b - a);
  }
  metal_compute::SharedBuffer out = mc->make_shared_buffer(n * (c1 - c0));
  if (out.empty()) { return {}; }
  const auto* sp = static_cast<const std::uint8_t*>(src.contents());
  auto* dp = static_cast<std::uint8_t*>(out.contents());
  for (const auto& [a, b] : rows) {
    if (c0 == 0 && c1 == rb) {
      std::memcpy(dp, sp + (std::size_t)a * rb, (std::size_t)(b - a) * rb);
      dp += (std::size_t)(b - a) * rb;
      continue;
    }
    for (int r = a; r < b; ++r) {
      std::memcpy(dp, sp + (std::size_t)r * rb + c0, c1 - c0);
      dp += c1 - c0;
    }
  }
  return out;
}

// A SHARE CUT BY ITS ROWS, READ AS ONE: `rows` of tensor `name` -- [N,
// ...] as the checkpoint stores it, in `dtype` -- read straight into a
// buffer of their own (MetalLlamaWeights::pread_range_into: a range a
// row band, uncached), the rest of the tensor never read. Empty where
// the tensor is another dtype or another N rows, or is not a plain
// safetensors one: the caller then reads it whole and cuts it (tp_cut).
// Host-only: pread_range_into is libvpipe's own.
inline metal_compute::SharedBuffer
tp_read_rows(metal_compute::MetalCompute* mc, const MetalLlamaWeights& src,
             const std::string& name, const char* dtype, int N,
             const std::vector<std::pair<int, int>>& rows)
{
  const auto* ti = src.info(name);
  if (ti == nullptr || ti->dtype != dtype || ti->shape.empty() ||
      ti->shape[0] != N || N <= 0 || ti->nbytes % (std::size_t)N != 0) {
    return {};
  }
  const std::size_t rb = (std::size_t)ti->nbytes / (std::size_t)N;
  std::size_t n = 0;
  for (const auto& [a, b] : rows) {
    if (a < 0 || b > N || a >= b) { return {}; }
    n += (std::size_t)(b - a);
  }
  metal_compute::SharedBuffer out = mc->make_shared_buffer(n * rb);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint8_t*>(out.contents());
  for (const auto& [a, b] : rows) {
    if (!src.pread_range_into(name, (std::uint64_t)a * rb,
                              (std::size_t)(b - a) * rb, d)) {
      return {};
    }
    d += (std::size_t)(b - a) * rb;
  }
  return out;
}

}  // namespace vpipe::genai

#endif
