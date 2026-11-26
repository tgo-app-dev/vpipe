#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_IMAGE_ENCODER_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_IMAGE_ENCODER_H

// MAGI-2-preview's image-to-video picture, encoded as the release does it
// (inference_engine._encode_image over model/vae2_2.py, read, not run): the
// Wan2.2 VAE's ENCODER on ONE frame, its mean, whitened by the latent
// statistics -- [48, h, w] at 16 pixels a cell.
//
// One frame makes the causal 3D network a 2D one: a causal 3x3x3 conv
// over a single frame (its two predecessors zero-padded) is the 3x3 conv
// of its LAST temporal tap; a temporal downsample's time conv never runs
// on a first chunk; the residual shortcut's average-down pads time to its
// factor with a ZERO frame, which that average then includes. So: a 2x2
// patchify to 12 channels, a 3x3 conv to 160, four down blocks (two
// residual blocks each -- RMSNorm (F.normalize x sqrt(C) x gamma), SiLU,
// conv, twice -- a stride-2 downsample padded right and bottom in the
// first three, and the average-down shortcut of the block's input), a
// middle (residual, single-head attention over the pixels, residual), a
// head (RMSNorm, SiLU, conv to 96) and a 1x1 conv; the mean is its first
// 48 channels.
//
// IN f32 ON THE HOST (Accelerate's sgemm), as the release runs it in f32:
// one picture a clip, ~250 GFLOP at 896 x 512 -- the GPU is not worth
// the code here.
//
// The picture must be on the 32-pixel grid the release floors it to
// (diffusers' VideoProcessor, vae_scale_factor 32); one that is not is
// resized there the release's way -- a PIL LANCZOS of its bytes.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace metal_compute { class MetalCompute; }
namespace genai {

class WeightSet;
class MetalLlamaWeights;

class Magi2ImageEncoder {
 public:
  // `ws`: the release's vae/Wan2.2_VAE.pth (a plain state dict).
  static std::unique_ptr<Magi2ImageEncoder>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       std::string* err);
  ~Magi2ImageEncoder();

  // `rgb` planar [3, H, W] in [-1, 1]. Out: [48, h, w] whitened (h = H' /
  // 16, w = W' / 16 for the picture's 32-pixel floor H' x W').
  bool encode(const float* rgb, int H, int W, std::vector<float>* z, int* h,
              int* w, std::string* err) const;

  std::size_t weight_bytes() const noexcept;
  // The same off the file's headers, before a load: what load() keeps on
  // the host -- every encoder conv as its LAST temporal tap, f32, with its
  // bias, and the norms' gammas (the temporal downsamples' time convs,
  // which never run on one frame, left out).
  static std::size_t held_bytes(const MetalLlamaWeights& src);
  // What an encode of an H x W picture holds on the host beside the
  // weights, at most: the picture resized to its 32-pixel floor, the five
  // activations alive at once in a top-level residual block (its input,
  // the running value, the norm's copy, the conv's output and the
  // shortcut) at 160 channels, an im2col chunk, and the middle's
  // attention over the pixels at a sixteenth -- its scores [N, N].
  static std::size_t working_bytes(int H, int W);

  struct Impl;                    // the weights (defined in the .cc)

 private:
  Magi2ImageEncoder() = default;
  std::unique_ptr<Impl> _p;
};

}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_IMAGE_ENCODER_H
