#ifndef VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_VISION_H
#define VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_VISION_H

// DeepSeek-V4.1's vision path (the release's vision.py and
// image_processor.py): the picture resized and padded onto a grid of
// 14-pixel patches (Pillow's ImageOps.pad, bicubic, gray 127), a
// bidirectional ViT with a 2-D rotary (half the pairs turn by the patch's
// row, half by its column), and the ALIGNER -- 3x3 patches unfolded into
// one row, two linears with a GELU between -- giving one LM row per cell.
//
// What the LM reads is the image's SPAN: [IMAGE_START], then each grid
// row's cells followed by [IMAGE_NEW_LINE], then [IMAGE_END], the three
// markers learned embeddings of their own. Every span position carries
// image_token_id in the token stream; the rows here replace their
// embeddings.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/deepseek-v41/deepseek-v41-config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class WeightSet;

class MetalDeepseekV41Vision {
public:
  struct Plan {
    int vit_h = 0, vit_w = 0;      // patch grid
    int llm_h = 0, llm_w = 0;      // cell grid
    int best_h = 0, best_w = 0;    // pixels the picture is padded to
    int span() const { return llm_h * (llm_w + 1) + 2; }
  };
  struct Result {
    std::vector<float> rows;       // [n_tokens, dim]: the image's span
    int n_tokens = 0;
    int grid_h = 0, grid_w = 0;    // the cell grid
  };

  static std::unique_ptr<MetalDeepseekV41Vision> load(
      const std::shared_ptr<WeightSet>& ws, metal_compute::MetalCompute* mc,
      const DeepseekV41Config& cfg, bool bf16, std::string* err = nullptr);
  ~MetalDeepseekV41Vision();

  // image_processor.plan_image_grid for a picture of this size.
  static Plan plan(const DeepseekV41Config& c, int W, int H);
  // load_image: the padded picture's patches, normalized to [-1, 1] and
  // rounded to the compute dtype as the reference does, [vit_h * vit_w,
  // 3 * 14 * 14] (channel, row, column within a patch). `rgb` is planar
  // [3, H, W].
  static std::vector<float> patches(const DeepseekV41Config& c,
                                    const std::uint8_t* rgb, int H, int W,
                                    Plan* plan, bool bf16);

  // The span's rows for a planar [3, H, W] picture; empty on failure.
  Result encode(const std::uint8_t* rgb, int H, int W);
  // ... from patches already made (tests: the reference's own).
  Result encode_patches(const std::vector<float>& patches, const Plan& p);

  std::size_t resident_bytes() const;
  // What encode() allocates for the largest picture (a span of
  // vis_max_tokens): the plan's figure for a picture turn.
  static std::size_t encode_scratch_bytes(const DeepseekV41Config& c);

private:
  MetalDeepseekV41Vision() = default;
  bool init_(const std::shared_ptr<WeightSet>& ws, std::string* err);

  struct Block {
    metal_compute::SharedBuffer norm1, norm2, wqkv, bqkv, wo, bo, w1, w2;
  };
  metal_compute::MetalCompute* _mc = nullptr;
  DeepseekV41Config _cfg;
  bool _bf16 = true;
  std::shared_ptr<WeightSet> _ws;
  metal_compute::ComputeLibrary _lib, _lib_dense, _lib_attn;
  metal_compute::ComputeFunction _fn_rms, _fn_rope, _fn_add, _fn_unfold;
  metal_compute::ComputeFunction _fn_gelu, _fn_swiglu, _fn_gemm, _fn_attn;
  metal_compute::SharedBuffer _patch_w, _patch_b, _norm;
  metal_compute::SharedBuffer _al_w1, _al_b1, _al_w2, _al_b2;
  metal_compute::SharedBuffer _img_start, _img_end, _img_newline;
  metal_compute::SharedBuffer _attn_params;
  metal_compute::CommandStream _stream;
  std::vector<Block> _blocks;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_VISION_H
