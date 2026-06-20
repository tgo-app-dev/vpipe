#ifndef VPIPE_GENERATIVE_MODELS_TOKEN_MUXER_H
#define VPIPE_GENERATIVE_MODELS_TOKEN_MUXER_H


#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace vpipe { namespace metal_compute { class SharedBuffer; } }

namespace vpipe::genai {

// One unit of input to the first attention layer. Three variants:
//   * Text          - a vocab id, looked up in the embedding table.
//   * ImageTokens   - a row of a vision-encoder's pre-computed
//                     embedding sequence (Qwen3-VL et al).
//   * AudioTokens   - a row of an audio-encoder's pre-computed
//                     embedding sequence (Qwen3-ASR et al).
//
// For ImageTokens / AudioTokens the embedding tensor lives in the
// encoder's output; the muxer never owns it. The `*_token_offset`
// field indicates which row of that tensor THIS TokenRef stands
// for, so a sequence of N image / audio tokens is represented as
// N consecutive TokenRefs sharing the same tensor pointer with
// offsets 0..N-1.
struct TokenRef {
  enum class Kind : std::uint8_t { Text, ImageTokens, AudioTokens };
  Kind                    kind                = Kind::Text;
  // Text variant.
  std::int32_t            text_id             = 0;
  // Image / Audio token row index into the encoder's output sequence.
  int                     image_token_offset  = 0;
  int                     audio_token_offset  = 0;
  // Metal backend (MLX-free): borrowed pointer to the encoder's
  // [rows, host_hidden] row-major f32 buffer; the row used is
  // image_token_offset / audio_token_offset.
  const std::vector<float>* embeddings_host   = nullptr;
  int                       host_hidden       = 0;
  // Native-f16 zero-copy (metal): borrowed pointer to the encoder's
  // [rows, host_hidden] row-major f16 SharedBuffer (GPU/UMA). When set,
  // the metal splice copies the f16 row straight in (no f32 round-trip).
  // Takes priority over embeddings_host for the indicated row.
  const metal_compute::SharedBuffer* embeddings_buf = nullptr;
};


}

#endif
