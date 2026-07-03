#ifndef VPIPE_GENERATIVE_MODELS_GEMMA4_UNIFIED_EMBEDDER_H
#define VPIPE_GENERATIVE_MODELS_GEMMA4_UNIFIED_EMBEDDER_H

// Gemma4UnifiedEmbedder -- the multimodal front-end for gemma-4-12B
// (`gemma4_unified`). UNLIKE e4b there is NO ViT and NO Conformer: a shallow
// patch embedder (vision) and a single linear projection (audio) turn
// pixels/audio directly into LM soft tokens; the 48-layer LM does all the
// cross-modal work (Fuyu-style). Weights come from llama.cpp's mmproj GGUF
// (projector_type gemma4uv / gemma4ua) -- 11 tensors, no attention blocks.
//
// The forward runs ONCE per image/clip, so v1 is host-f32 (no GPU kernels;
// the two matmuls are naive host loops -- a GPU-GEMM perf pass is a TODO).
// Output rows are f32 [n_tokens, out_hidden] for the metal prefill_mm splice.
//
// MLX-free: depends only on GgufFile. Lives on the no-MLX/GGUF metal path.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class GgufFile;

class Gemma4UnifiedEmbedder {
public:
  struct EncodedImage {
    std::vector<float> rows;   // [n_tokens * out_hidden] row-major f32
    int n_tokens = 0;
    int grid_h = 0;            // patch rows
    int grid_w = 0;            // patch cols
  };
  struct EncodedAudio {
    std::vector<float> rows;   // [n_tokens * out_hidden] row-major f32
    int n_tokens = 0;
  };

  // Open an mmproj GGUF (projector_type gemma4uv/gemma4ua) and dequantise the
  // 11 tensors to f32. Returns nullptr if the file is not a gemma4-unified
  // mmproj.
  static std::unique_ptr<Gemma4UnifiedEmbedder>
  load(const std::string& mmproj_path);

  // Load the SAME shallow adaptor weights from a raw safetensors 12B
  // checkpoint (`model.vision_embedder.* / model.embed_vision.* /
  // model.embed_audio.*`), converting bf16 -> f32 into the identical
  // internal buffers the GGUF path fills (encode_image/encode_audio are
  // source-agnostic). `mc` is used only to stage tensors through the
  // MetalLlamaWeights reader; no GPU kernels run. nullptr if neither
  // adaptor is present.
  static std::unique_ptr<Gemma4UnifiedEmbedder>
  load_safetensors(const std::string& model_dir,
                   metal_compute::MetalCompute* mc);

  // True if `model_dir`'s safetensors carries the unified vision/audio
  // adaptor tensors (used by the loader to flag config before load).
  static bool has_unified_safetensors(const std::string& model_dir);

  // Locate a sibling `mmproj*.gguf` in `model_dir`; empty string if none.
  static std::string find_mmproj(const std::string& model_dir);

  bool has_vision() const noexcept { return _has_vision; }
  bool has_audio() const noexcept { return _has_audio; }
  int out_hidden() const noexcept { return _embed; }   // 3840

  // Attach a session so encode_image/encode_audio show up on the profiler
  // (vision-tower / audio-encoder lanes). Optional; null => no-op.
  void set_session(const SessionContextIntf* s) { _session = s; }

  // Vision. `rgb_chw` is planar [3,H,W] u8 (R plane, G plane, B plane;
  // row-major within a plane) -- the engine's standard layout. The image is
  // smart-resized internally. nullopt on shape error / missing weights.
  std::optional<EncodedImage>
  encode_image(const std::uint8_t* rgb_chw, int H, int W) const;

  // Audio. `pcm` is mono f32 @ 16 kHz. One soft token per 640 samples.
  std::optional<EncodedAudio>
  encode_audio(const float* pcm, std::size_t n) const;

  // Smart-resize target: both multiples of 48, pixel count clamped to
  // [40, 280] * 48^2. (Exposed for the preprocessing-match test.)
  void smart_resize(int H, int W, int* th, int* tw) const;

private:
  Gemma4UnifiedEmbedder() = default;

  // Vision weights (f32, [out, in] row-major).
  std::vector<float> _w_patch;     // [embed, 6912]
  std::vector<float> _b_patch;     // [embed]
  std::vector<float> _ln1_w, _ln1_b;  // [6912]
  std::vector<float> _ln2_w, _ln2_b;  // [embed]
  std::vector<float> _ln3_w, _ln3_b;  // [embed]
  std::vector<float> _pos;         // [2 * pos_max * embed]
  std::vector<float> _w_proj;      // [embed, embed]
  // Audio weights.
  std::vector<float> _w_aproj;     // [embed, 640]

  int  _embed       = 3840;
  int  _patch_px    = 48;          // 16 * pooling 3
  int  _patch_in    = 6912;        // 48*48*3
  int  _pos_max     = 1120;
  int  _audio_frame = 640;
  float _eps_ln     = 1.0e-5f;     // patch LayerNorms (hardcoded in llama.cpp)
  float _eps_rms    = 1.0e-6f;     // weightless pre-projection RMSNorm
  bool _has_vision  = false;
  bool _has_audio   = false;
  const SessionContextIntf* _session = nullptr;   // profiler, optional
};

}  // namespace vpipe::genai

#endif
