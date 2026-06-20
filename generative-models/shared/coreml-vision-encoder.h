#ifndef VPIPE_GENERATIVE_MODELS_COREML_VISION_ENCODER_H
#define VPIPE_GENERATIVE_MODELS_COREML_VISION_ENCODER_H

// The CoreML vision tower exposes an MLX-free host API
// (encode_host/encode_pair_host -> Result of host f32) used by the
// metal LM path to splice CoreML image embeddings via
// prefill_multimodal_metal().

#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class MlxRuntime;
class SessionContextIntf;
}

namespace vpipe::genai {

// CoreML-backed vision tower: delegates the ViT body + patch merger to a
// pre-compiled CoreML mlpackage instead of running it in MLX. Use this
// when the user has converted the vision tower to CoreML (e.g. via
// coremltools) and wants the Apple Neural Engine / optimised GPU
// compiler in the inference path.
//
// Contract:
//   * Input: u8 [3, H, W] RGB planar bytes.
//   * MLX path (encode/encode_pair): EncodedImage{ embeddings
//     [n_tokens, lm_hidden] in compute_dtype }.
//   * Metal/no-MLX path (encode_host/encode_pair_host): Result{
//     embeddings (host f32 [n_tokens*hidden] row-major), n_tokens,
//     grid_h, grid_w } -- splice via prefill_multimodal_metal().
//   * DeepStack features: not surfaced (checkpoints needing DeepStack
//     stay on the MLX tower).
//
// Preprocessing (both paths): the source image is letterboxed
// (aspect-preserving, 114/255 grey padding) and RGB->BGRA8888 packed
// into a CVPixelBuffer of the model's fixed input size via the
// metal-compute `letterbox_planar_u8_to_bgra_u8` kernel (no MLX, no
// MetalRuntime). The model's input pixel dimensions are read off the
// MLModelDescription at construction and never change.
class CoreMLVisionEncoder
{
public:
  // MLX-free host result: [n_tokens, hidden] row-major f32 embeddings,
  // ready to splice via prefill_multimodal_metal(). Mirrors
  // MetalQwenVisionEncoder::Result. n_tokens == 0 signals failure.
  struct Result {
    // Native f16 [n_tokens * out_hidden] row-major, on the GPU (UMA) --
    // splice via prefill_multimodal_metal with no host f32 round-trip.
    metal_compute::SharedBuffer embeddings;
    int n_tokens   = 0;
    int out_hidden = 0;
    int grid_h     = 0;
    int grid_w     = 0;
  };

  struct LoadSpec {
    // Filesystem path to an .mlpackage / .mlmodelc directory.
    std::string      mlpackage_path;
    // 0 = CPU-only, 1 = CPU+GPU, 2 = CPU+GPU+ANE. Default 2 (ALL).
    int              compute_units  = 2;
    // Post-merger grid dims, used to populate grid_h / grid_w. When both
    // are 0 the encoder reads them from
    //   grid = input_pixels / (patch_size * spatial_merge_size).
    int              grid_h            = 0;
    int              grid_w            = 0;
    int              patch_size        = 16;
    int              spatial_merge_size = 2;
  };

  // Build and bind the model. Returns nullptr on failure (and logs
  // through session). `runtime` may be null (used only by the MLX path
  // to build the EncodedImage array on the runtime worker).
  static std::unique_ptr<CoreMLVisionEncoder>
  create(const LoadSpec&             spec,
         MlxRuntime*                 runtime,
         const SessionContextIntf*   session);

  ~CoreMLVisionEncoder();

  // ---- MLX-free host API (both builds) ----
  // Encode one RGB-planar [3,H,W] u8 image into native-f16 embeddings.
  Result encode_host(const std::uint8_t* rgb, int H, int W);
  // Zero-copy input overload: `src` is a Shared/UMA [3,H,W] u8 frame
  // (e.g. bridged from a metal-compute-backed TensorBeat) bound straight
  // into the letterbox kernel -- no host staging copy.
  Result encode_host(const metal_compute::SharedBuffer& src, int H, int W);
  // Temporal pair: feed two DISTINCT frames into the two-input video
  // export so its in-model merge yields one token grid. Empty Result on
  // a single-input model.
  Result encode_pair_host(const std::uint8_t* rgbA,
                          const std::uint8_t* rgbB, int H, int W);

  bool implemented() const noexcept
      { return _valid; }

  // True only when the loaded mlpackage has the two-input ("image0" +
  // "image1") video layout.
  bool supports_temporal_pair() const noexcept
      ;

  std::string_view family_name() const noexcept
      { return "coreml-vision"; }


  // Test/debug accessors.
  int model_input_width()  const noexcept;
  int model_input_height() const noexcept;
  int output_n_tokens()    const noexcept;
  int output_hidden()      const noexcept;

private:
  CoreMLVisionEncoder();

  // Shared MLX-free encode core: letterbox each frame into its own BGRA
  // pixel buffer (metal-compute), run one CoreML prediction binding the
  // image input(s), and return the host f32 [n_tokens, hidden]
  // embeddings. std::nullopt on any failure.
  std::optional<Result> encode_frames_host_(
      const std::vector<const std::uint8_t*>& frames, int H, int W,
      const metal_compute::SharedBuffer* src_buf = nullptr);

  struct Impl;
  std::unique_ptr<Impl> _p;
  bool                  _valid = false;
};

}

#endif
