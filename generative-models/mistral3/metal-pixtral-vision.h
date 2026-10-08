#ifndef VPIPE_GENERATIVE_MODELS_MISTRAL3_METAL_PIXTRAL_VISION_H
#define VPIPE_GENERATIVE_MODELS_MISTRAL3_METAL_PIXTRAL_VISION_H

// MetalPixtralVisionEncoder -- the Mistral3 image path (Ministral-3,
// Mistral Small 3.x) on metal-compute, no MLX in the forward: the Pixtral
// ViT and the Mistral3 multi-modal projector.
//
//   resize (PIL BICUBIC on 8 bits; longest edge within the budget, each
//   side rounded UP to a multiple of patch*merge) -> rescale + CLIP
//   normalise -> patchify -> patch_conv GEMM -> ln_pre RMSNorm -> N blocks
//   (RMSNorm, q/k/v, axial 2-D RoPE, bidirectional attention, o,
//   RMSNorm, SwiGLU MLP) -> projector: RMSNorm, the 2x2 patch merge (one
//   GEMM over the four patches' channels), linear_1, GELU, linear_2.
//
// The output carries the image's [IMG_BREAK] / [IMG_END] rows as well:
// a gh x gw grid of merged tokens is gh*(gw+1) rows, row-major, each grid
// row closed by the LM's own embedding of [IMG_BREAK] and the last by
// [IMG_END] (set_layout_rows). So the image is one placeholder per row
// in the prompt -- what the reference's prompt holds, ids aside -- and a
// template needs nothing but the count.
//
// Patches run through the ViT in MERGE order (each 2x2 block's four
// patches adjacent), which attention and the per-token layers do not
// see, so the merge is a reshape; the merging layer's columns are
// permuted to match once, at load (dequantised, since a permutation does
// not keep 4-bit groups whole). Weights: the ViT dense BF16 -> F16; the
// projector's linear_1 / linear_2 affine 4- or 8-bit as stored.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

struct ModelConfig;
class WeightSet;       // generative-models/weight-set.h

class MetalPixtralVisionEncoder {
public:
  struct Config {
    int   depth        = 24;
    int   hidden       = 1024;
    int   n_heads      = 16;
    int   head_dim     = 64;
    int   intermediate = 4096;
    int   patch_size   = 14;
    int   merge        = 2;       // spatial_merge_size
    int   out_hidden   = 5120;    // the LM's hidden
    int   longest_edge = 1540;    // the processor's size.longest_edge
    float rope_theta   = 10000.0f;
    float rms_eps      = 1e-5f;   // ViT norms
    float proj_eps     = 1e-5f;   // projector norm (text rms_norm_eps)
    int   quant_bits   = 4;       // projector linears (group 64)
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
  };

  struct Result {
    // Native f16 [n_tokens * out_hidden] row-major on the GPU (UMA).
    metal_compute::SharedBuffer embeddings;
    int n_tokens   = 0;   // grid_h * (grid_w + 1): image + break rows
    int out_hidden = 0;
    int grid_h     = 0;   // merged-token grid
    int grid_w     = 0;
  };

  // The checkpoint the LM loads from; the encoder KEEPS the set.
  static std::unique_ptr<MetalPixtralVisionEncoder> load(
      std::shared_ptr<WeightSet> ws,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  static Config config_from(const ModelConfig& c);

  // The processor's geometry for an H x W input: the size the image is
  // resized to (each a multiple of patch*merge). `max_tokens` > 0 also
  // caps the merged grid at that many tokens (a smaller longest edge,
  // e.g. for video frames); <= 0 is the processor's own budget.
  void target_size(int H, int W, int* th, int* tw,
                   int max_tokens = -1) const;

  // The rows a [IMG_BREAK] and an [IMG_END] stand for: the LM's own
  // embeddings, host f32 [out_hidden] each. encode() refuses until set.
  void set_layout_rows(const std::vector<float>& image_break,
                       const std::vector<float>& image_end);

  // The processor's half alone: resize to target_size (PIL BICUBIC on
  // 8 bits), rescale and normalise -> `px` [3, *th, *tw] f32. What
  // encode() hands encode_normalized().
  void preprocess(const std::uint8_t* rgb, int H, int W, int max_tokens,
                  std::vector<float>* px, int* th, int* tw) const;

  // Encode one RGB image: rgb is [3, H, W] U8 channel-first contiguous.
  // `max_tokens` as for target_size. Empty embeddings on failure.
  Result encode(const std::uint8_t* rgb, int H, int W,
                int max_tokens = -1);

  // The same from already-normalised pixels [3, h, w] f32, h and w
  // multiples of patch*merge (what the reference's processor hands its
  // tower). When given, `vit_out` gets the ViT's output [n_patches,
  // hidden] f32 in RASTER patch order and `proj_out` the projector's
  // [gh*gw, out_hidden] f32 -- the layers a golden checks one at a time.
  Result encode_normalized(const float* px, int h, int w,
                           std::vector<float>* vit_out = nullptr,
                           std::vector<float>* proj_out = nullptr);

  const Config& config() const { return _cfg; }

  // Which M5 matrix-core paths this encoder runs: the matmul2d dense
  // GEMMs and MLX's NAX steel attention (both off on a GPU without
  // matrix cores, or under VPIPE_PIXTRAL_NO_MMA2 / _NO_NAX_ATTN).
  bool uses_matrix_cores() const noexcept { return _use_mma2; }
  bool uses_nax_attention() const noexcept { return _use_attn_nax; }

  // Attach a session so encode() shows up on the profiler (vision-tower
  // lane) and logs; optional.
  void set_session(const SessionContextIntf* s) { _session = s; }

private:
  MetalPixtralVisionEncoder() = default;

  struct Block {
    metal_compute::SharedBuffer attn_norm, ffn_norm;
    metal_compute::SharedBuffer qw, kw, vw, ow;
    metal_compute::SharedBuffer gw, uw, dw;
  };
  struct QLinear {
    metal_compute::SharedBuffer w, s, b;   // codes u32, scales, biases
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  const SessionContextIntf* _session = nullptr;
  metal_compute::ComputeLibrary _lib_gemm, _lib_vis, _lib_elt, _lib_rms,
      _lib_qmm, _lib_attn, _lib_attn_nax, _lib_dense_mma;
  metal_compute::ComputeFunction _fn_gemm, _fn_rms, _fn_rope, _fn_gelu,
      _fn_transpose, _fn_residual, _fn_swiglu, _fn_qmm;
  // M4: the widest steel tile that loaded (BM x BN).
  int _gemm_bm = 32, _gemm_bn = 32;
  // M5: matrix-core matmul2d dense GEMM (n128, and n128x256 for deep K)
  // and MLX's NAX steel attention. VPIPE_PIXTRAL_NO_MMA2 /
  // VPIPE_PIXTRAL_NO_NAX_ATTN fall back to the M4 kernels for an A/B.
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;

  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _patch_w;    // [hidden, P*P*3] (kh, kw, c)
  metal_compute::SharedBuffer _ln_pre;
  metal_compute::SharedBuffer _proj_norm;
  metal_compute::SharedBuffer _merge_w;    // [hidden, merge^2*hidden], k-major
  QLinear _lin1, _lin2;
  metal_compute::SharedBuffer _zero_bias;
  metal_compute::SharedBuffer _attn_params;
  std::vector<std::uint16_t> _break_row, _end_row;   // f16 bits
  std::shared_ptr<WeightSet> _ws;
};

}  // namespace vpipe::genai

#endif
