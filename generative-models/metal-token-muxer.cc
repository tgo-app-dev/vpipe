#include "generative-models/metal-token-muxer.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <cstring>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

MetalTokenMuxer::MetalTokenMuxer(metal_compute::MetalCompute* mc,
                                 const SharedBuffer* w,
                                 const SharedBuffer* scales,
                                 const SharedBuffer* biases, int hidden,
                                 bool bf16, int quant_bits)
    : _mc(mc), _w(w), _scales(scales), _biases(biases), _hidden(hidden)
{
  if (mc != nullptr && mc->valid()) {
    _lib = mc->load_library(bf16 ? "llm_elementwise_bf16" : "llm_elementwise");
    _fn = _lib.function(quant_bits == 8 ? "dequant_embed_gather_w8_f16"
                                        : "dequant_embed_gather_f16");
  }
}

MetalTokenMuxer::MetalTokenMuxer(metal_compute::MetalCompute* mc,
                                 const SharedBuffer* q6k, int hidden,
                                 bool bf16)
    : _mc(mc), _hidden(hidden), _q6k(true), _q6k_table(q6k)
{
  if (mc != nullptr && mc->valid()) {
    _lib = mc->load_library(bf16 ? "llm_elementwise_bf16" : "llm_elementwise");
    _fn = _lib.function("embed_gather_q6k_f16");
  }
}

MetalTokenMuxer::MetalTokenMuxer(metal_compute::MetalCompute* mc,
                                 const SharedBuffer* kq, int hidden,
                                 bool bf16, bool table_is_q4k)
    : _mc(mc), _hidden(hidden), _q6k(true), _q6k_table(kq)
{
  if (mc != nullptr && mc->valid()) {
    _lib = mc->load_library(bf16 ? "llm_elementwise_bf16" : "llm_elementwise");
    // The raw-table fetch path (the `_q6k` branch of fetch_text) is identical
    // for Q4_K and Q6_K -- only the per-row gather kernel differs.
    _fn = _lib.function(table_is_q4k ? "embed_gather_q4k_f16"
                                     : "embed_gather_q6k_f16");
  }
}

bool
MetalTokenMuxer::valid() const noexcept
{
  if (_mc == nullptr || !_fn.valid() || _hidden <= 0) { return false; }
  if (_q6k) { return _q6k_table != nullptr; }
  return _w != nullptr && _scales != nullptr && _biases != nullptr;
}

SharedBuffer
MetalTokenMuxer::fetch_text(std::span<const std::int32_t> ids) const
{
  if (!valid() || ids.empty()) {
    return {};
  }
  const int n = (int)ids.size();
  SharedBuffer idbuf = _mc->make_shared_buffer((std::size_t)n * 4);
  auto* idp = static_cast<std::int32_t*>(idbuf.contents());
  for (int i = 0; i < n; ++i) { idp[i] = ids[i]; }

  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)n * _hidden * 2);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(_fn);
    if (_q6k) {
      // embed_gather_q6k_f16(ids, table, out, H); grid (H, n, 1).
      enc.set_buffer(0, idbuf);
      enc.set_buffer(1, *_q6k_table);
      enc.set_buffer(2, out);
      enc.set_constant(3, _hidden);
    } else {
      enc.set_buffer(0, idbuf);
      enc.set_buffer(1, *_w);
      enc.set_buffer(2, *_scales);
      enc.set_buffer(3, *_biases);
      enc.set_buffer(4, out);
      enc.set_constant(5, _hidden);
    }
    enc.dispatch({(unsigned)_hidden, (unsigned)n, 1}, {256, 1, 1});
  }
  stream.commit().wait();
  return out;
}

}  // namespace vpipe::genai
