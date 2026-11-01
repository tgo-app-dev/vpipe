#ifndef VPIPE_GENERATIVE_MODELS_LFM3_METAL_LFM3_VISION_H
#define VPIPE_GENERATIVE_MODELS_LFM3_METAL_LFM3_VISION_H

// MetalLfm3VisionEncoder -- LFM3-VL's image path on metal-compute: the
// vision tower and the multi-modal projector, no MLX in the forward.
//
//   resize (smart_resize: sides multiples of patch * merge, the pixel
//   count within the processor's bounds; Pillow BICUBIC on 8 bits) ->
//   rescale + normalise (mean / std 0.5) -> patchify in MERGE order, the
//   temporal patch a copy of the frame -> patchifier GEMM -> RMSNorm ->
//   N layers, each a gated 3x3 depthwise conv over the patch grid OR a
//   bidirectional GQA attention (q / k RMSNorm per head, axial 2-D RoPE:
//   the head's first quarter of pairs turn with the patch row, the next
//   with its column), then a SwiGLU feed-forward, pre-norm residuals ->
//   projector: LayerNorm per patch, the 2x2 block's four patches side by
//   side (merge order makes it a reshape), linear_1, GELU, linear_2.
//
// One merged token a 2x2 block of 16-pixel patches. The image's rows are
// the LM's <|image|> placeholders, 1-D positions. Weights dense F16.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class WeightSet;

class MetalLfm3VisionEncoder {
public:
  struct Config {
    int   layers       = 14;
    int   hidden       = 1024;
    int   n_heads      = 16;
    int   n_kv_heads   = 8;
    int   head_dim     = 64;
    int   intermediate = 2560;
    int   patch        = 16;
    int   merge        = 2;
    int   temporal     = 2;
    int   conv_k       = 3;
    float rope_theta   = 1.0e6f;
    float eps          = 1e-5f;
    float proj_eps     = 1e-6f;
    int   proj_hidden  = 4096;
    int   out_hidden   = 2048;     // the LM's hidden
    std::vector<bool> attn;        // per layer: attention, else conv
    // The processor's pixel-count bounds (image_processor.size).
    std::int64_t min_pixels = 65536;
    std::int64_t max_pixels = 16777216;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float stdv[3] = {0.5f, 0.5f, 0.5f};

    // config.json's vision_config and projector fields, and
    // processor_config.json's image_processor.
    static bool from_dir(const std::string& dir, Config* out,
                         std::string* err);
  };

  struct Result {
    // F16 [n_tokens * out_hidden] row-major (UMA).
    metal_compute::SharedBuffer embeddings;
    int n_tokens   = 0;
    int out_hidden = 0;
    int grid_h     = 0;   // merged-token grid
    int grid_w     = 0;
  };

  // The checkpoint the LM loads from; the encoder KEEPS the set.
  static std::unique_ptr<MetalLfm3VisionEncoder> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      const Config& cfg, std::string* err = nullptr);

  // What the tower and projector hold once loaded (dense F16).
  static std::size_t weight_bytes(const std::string& model_dir);

  // smart_resize for an H x W input: each side a multiple of patch *
  // merge, the pixels within [min_pixels, max] where max is `max_pixels`
  // when > 0 (a live frame's budget), else the processor's own.
  void target_size(int H, int W, int* th, int* tw,
                   std::int64_t max_pixels = 0) const;
  static void smart_resize(int H, int W, int factor, std::int64_t lo,
                           std::int64_t hi, int* th, int* tw);

  // The processor's half: resize (Pillow BICUBIC on 8 bits) + normalise
  // -> px [3, th, tw] f32.
  void preprocess(const std::uint8_t* rgb, int H, int W,
                  std::int64_t max_pixels, std::vector<float>* px, int* th,
                  int* tw) const;

  // One RGB image, rgb [3, H, W] u8 channel-first. Empty on failure.
  Result encode(const std::uint8_t* rgb, int H, int W,
                std::int64_t max_pixels = 0);

  // From normalised pixels [3, h, w], h and w multiples of patch * merge.
  // `vit_out` (tests): the tower's output [n_patches, hidden] f32 in
  // MERGE order; `proj_out`: the projector's [n_tokens, out_hidden].
  Result encode_normalized(const float* px, int h, int w,
                           std::vector<float>* vit_out = nullptr,
                           std::vector<float>* proj_out = nullptr);

  // The device memory an encode allocates beside the weights at the
  // largest picture of at most `pixels` pixels (<= 0: the processor's
  // own bound) -- its buffers grow with the patch count.
  static std::size_t encode_scratch_bytes(const Config& c,
                                          std::int64_t pixels);

  const Config& config() const { return _cfg; }
  bool uses_matrix_cores() const noexcept { return _use_mma2; }
  bool uses_nax_attention() const noexcept { return _use_attn_nax; }
  void set_session(const SessionContextIntf* s) { _session = s; }

private:
  MetalLfm3VisionEncoder() = default;

  struct Layer {
    bool attn = false;
    metal_compute::SharedBuffer op_norm, ffn_norm;
    metal_compute::SharedBuffer in_w, conv_w, out_w;       // conv
    metal_compute::SharedBuffer qkv_w, q_norm, k_norm, o_w;  // attention
    metal_compute::SharedBuffer w13, w2;
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;
  std::shared_ptr<WeightSet> _ws;
  metal_compute::ComputeLibrary _lib_dense, _lib_mma, _lib_rms, _lib_rope,
      _lib_elt, _lib_vis, _lib_attn, _lib_attn_nax, _lib_lfm3;
  metal_compute::ComputeFunction _fn_gemm, _fn_mma, _fn_mma_deep, _fn_rms,
      _fn_norm_rope, _fn_heads, _fn_tokens, _fn_residual, _fn_swiglu,
      _fn_conv, _fn_ln, _fn_gelu, _fn_bias;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;
  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _patch_w, _embed_norm;
  metal_compute::SharedBuffer _ln_w, _ln_b, _l1_w, _l1_b, _l2_w, _l2_b;
  metal_compute::SharedBuffer _attn_params;
};

}  // namespace vpipe::genai

#endif
