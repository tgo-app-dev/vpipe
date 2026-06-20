#ifndef VPIPE_GENERATIVE_MODELS_METAL_TOKEN_MUXER_H
#define VPIPE_GENERATIVE_MODELS_METAL_TOKEN_MUXER_H

// MetalTokenMuxer -- metal-compute-native counterpart of TokenMuxer.
// Builds the [n, hidden] embedding tensor fed into the LLM's first
// layer as a SharedBuffer, gathering + dequantizing the rows of a
// 4-bit affine embedding table on the GPU (no MLX, no dense table
// materialized). Step 1 of making the LLM core backend-native so the
// metal Llama path plugs into LoadedLanguageModel like the MLX path.
// Text-only for now; image/audio TokenRefs come with metal multimodal.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <span>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class MetalTokenMuxer {
public:
  // Borrowed quantized embedding table (must outlive the muxer):
  //   w      [vocab, hidden*bits/32] uint32 packed
  //   scales [vocab, hidden/64] f16
  //   biases [vocab, hidden/64] f16   (group_size 64, affine)
  // bf16=true selects the bf16 embed-gather metallib (scales/biases must
  // then be bf16); default false = f16. quant_bits is 4 (default) or 8
  // (Qwen3-ASR) and selects the nibble vs byte gather kernel.
  MetalTokenMuxer(metal_compute::MetalCompute* mc,
                  const metal_compute::SharedBuffer* w,
                  const metal_compute::SharedBuffer* scales,
                  const metal_compute::SharedBuffer* biases,
                  int hidden, bool bf16 = false, int quant_bits = 4);

  // Native Q6_K embedding table (GGUF, no affine requant): gathers rows
  // with embed_gather_q6k_f16. `q6k` is the raw [vocab, hidden] Q6_K
  // table (must outlive the muxer).
  MetalTokenMuxer(metal_compute::MetalCompute* mc,
                  const metal_compute::SharedBuffer* q6k, int hidden,
                  bool bf16 = false);

  bool valid() const noexcept;

  // Gather + dequantize the rows for `ids` into a fresh [n, hidden]
  // f16 SharedBuffer. Empty buffer on failure / empty input.
  metal_compute::SharedBuffer
  fetch_text(std::span<const std::int32_t> ids) const;

private:
  metal_compute::MetalCompute*       _mc = nullptr;
  const metal_compute::SharedBuffer* _w = nullptr;
  const metal_compute::SharedBuffer* _scales = nullptr;
  const metal_compute::SharedBuffer* _biases = nullptr;
  int                                _hidden = 0;
  bool                               _q6k = false;
  const metal_compute::SharedBuffer* _q6k_table = nullptr;
  metal_compute::ComputeLibrary      _lib;
  metal_compute::ComputeFunction     _fn;
};

}  // namespace vpipe::genai

#endif
