#ifndef VPIPE_GENERATIVE_MODELS_LFM3_LFM3_WEIGHTS_H
#define VPIPE_GENERATIVE_MODELS_LFM3_LFM3_WEIGHTS_H

// The LFM3 checkpoint's tensors as the metal models keep them: F16,
// converted once from the stored BF16 / F16 / F32 into a derived entry of
// the manager's set (keyed by `ns` + the name, so every model naming the
// checkpoint shares one copy), and row-concatenated fusions (q|k|v|gate,
// w1|w3) built the same way -- the pieces read uncached, consumed.

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace vpipe::genai {

class Lfm3Weights {
public:
  Lfm3Weights(WeightSet* ws, metal_compute::MetalCompute* mc,
              std::string ns)
    : _ws(ws), _mc(mc), _ns(std::move(ns))
  {}

  bool has(const std::string& name) const
  {
    return _ws->src().info(name) != nullptr;
  }

  // The tensor's element count, 0 when absent.
  std::size_t numel(const std::string& name) const
  {
    const auto* i = _ws->src().info(name);
    if (i == nullptr) { return 0; }
    std::size_t n = 1;
    for (auto d : i->shape) { n *= (std::size_t)d; }
    return n;
  }

  // One tensor as F16 (any stored float type).
  metal_compute::SharedBuffer f16(const std::string& name)
  {
    if (!has(name)) { return {}; }
    return _ws->derived(_ns + name, [&]() -> metal_compute::SharedBuffer {
      const std::size_t n = numel(name);
      metal_compute::SharedBuffer out = _mc->make_shared_buffer(n * 2);
      if (out.empty() || !convert_(name, out.contents(), n)) { return {}; }
      return out;
    });
  }

  // Tensors stacked along their first axis (each [rows_i, K]), as F16.
  metal_compute::SharedBuffer fused(const std::vector<std::string>& names)
  {
    std::string key = _ns + "fused";
    std::size_t total = 0;
    for (const auto& n : names) {
      if (!has(n)) { return {}; }
      key += "|" + n;
      total += numel(n);
    }
    return _ws->derived(key, [&]() -> metal_compute::SharedBuffer {
      metal_compute::SharedBuffer out = _mc->make_shared_buffer(total * 2);
      if (out.empty()) { return {}; }
      auto* dst = static_cast<std::uint8_t*>(out.contents());
      for (const auto& n : names) {
        const std::size_t k = numel(n);
        if (!convert_(n, dst, k)) { return {}; }
        dst += k * 2;
      }
      return out;
    });
  }

private:
  // Read `name` uncached and write it as F16 into dst (n elements).
  bool convert_(const std::string& name, void* dst, std::size_t n)
  {
    const auto* info = _ws->src().info(name);
    if (info == nullptr) { return false; }
    metal_compute::SharedBuffer raw =
        _ws->read(name, _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return false; }
    const std::string& dt = info->dtype;
    if (dt != "BF16" && dt != "F16" && dt != "F32") { return false; }
    if (dt == "F16") {
      std::memcpy(dst, raw.contents(), n * 2);
      return true;
    }
    auto* o = static_cast<_Float16*>(dst);
    const void* s = raw.contents();
    auto run = [&](std::size_t lo, std::size_t hi) {
      if (dt == "BF16") {
        const auto* b = static_cast<const std::uint16_t*>(s);
        for (std::size_t i = lo; i < hi; ++i) {
          const std::uint32_t bits = (std::uint32_t)b[i] << 16;
          float f;
          std::memcpy(&f, &bits, 4);
          o[i] = (_Float16)f;
        }
      } else {
        const auto* f = static_cast<const float*>(s);
        for (std::size_t i = lo; i < hi; ++i) { o[i] = (_Float16)f[i]; }
      }
    };
    const std::size_t per = 1u << 22;
    if (n <= per) {
      run(0, n);
      return true;
    }
    const unsigned nt = std::clamp(
        std::thread::hardware_concurrency(), 1u, 8u);
    std::vector<std::thread> th;
    const std::size_t step = (n + nt - 1) / nt;
    for (unsigned t = 0; t < nt; ++t) {
      const std::size_t lo = t * step, hi = std::min(n, lo + step);
      if (lo >= hi) { break; }
      th.emplace_back(run, lo, hi);
    }
    for (auto& t : th) { t.join(); }
    return true;
  }

  WeightSet* _ws;
  metal_compute::MetalCompute* _mc;
  std::string _ns;
};

}  // namespace vpipe::genai

#endif
