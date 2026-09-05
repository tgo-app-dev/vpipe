#include "generative-models/minimax-h3/vdn-linear-features.h"

#include <cmath>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

void
short_conv_spatial(const float* tokens, int frames, int grid_h, int grid_w,
                   int channels, const ShortConv& conv, float* out)
{
  const int k = conv.kernel;
  const int pad = k / 2;
  const std::size_t per_frame = (std::size_t)grid_h * grid_w * channels;
  for (int f = 0; f < frames; ++f) {
    const float* src = tokens + (std::size_t)f * per_frame;
    float* dst = out + (std::size_t)f * per_frame;
    for (int y = 0; y < grid_h; ++y) {
      for (int x = 0; x < grid_w; ++x) {
        float* o = dst + ((std::size_t)y * grid_w + x) * channels;
        for (int c = 0; c < channels; ++c) { o[c] = 0.0f; }
        for (int ky = 0; ky < k; ++ky) {
          const int sy = y + ky - pad;
          if (sy < 0 || sy >= grid_h) { continue; }   // zero padding
          for (int kx = 0; kx < k; ++kx) {
            const int sx = x + kx - pad;
            if (sx < 0 || sx >= grid_w) { continue; }
            const float* s =
                src + ((std::size_t)sy * grid_w + sx) * channels;
            const float* w = conv.spatial + (std::size_t)ky * k + kx;
            const std::size_t stride = (std::size_t)k * k;
            for (int c = 0; c < channels; ++c) {
              o[c] += s[c] * w[(std::size_t)c * stride];
            }
          }
        }
      }
    }
  }
}

void
short_conv_temporal(const float* x, int frames, int tokens_per_frame,
                    int channels, const ShortConv& conv, float* out)
{
  const int k = conv.kernel;
  const int pad = k / 2;
  const std::size_t per_frame = (std::size_t)tokens_per_frame * channels;
  for (int f = 0; f < frames; ++f) {
    float* o = out + (std::size_t)f * per_frame;
    for (std::size_t i = 0; i < per_frame; ++i) { o[i] = 0.0f; }
    for (int dt = 0; dt < k; ++dt) {
      // A CORRELATION with zero padding, both directions: the reference
      // shifts a zero-padded copy by dt and weights it by w[:, dt], so
      // the source frame is t + dt - pad and frames outside the clip
      // contribute nothing.
      const int sf = f + dt - pad;
      if (sf < 0 || sf >= frames) { continue; }
      const float* s = x + (std::size_t)sf * per_frame;
      for (int t = 0; t < tokens_per_frame; ++t) {
        const float* sp = s + (std::size_t)t * channels;
        float* op = o + (std::size_t)t * channels;
        for (int c = 0; c < channels; ++c) {
          op[c] += sp[c] * conv.temporal[(std::size_t)c * k + dt];
        }
      }
    }
  }
}

void
linear_features(const float* tokens, int frames, int grid_h, int grid_w,
                int heads, int head_dim, const ShortConv* conv, bool l2norm,
                float* out)
{
  const int channels = heads * head_dim;
  const int per_frame_tokens = grid_h * grid_w;
  const std::size_t n =
      (std::size_t)frames * per_frame_tokens * (std::size_t)channels;

  if (conv != nullptr && conv->spatial != nullptr
      && conv->temporal != nullptr) {
    std::vector<float> mid(n);
    short_conv_spatial(tokens, frames, grid_h, grid_w, channels, *conv,
                       mid.data());
    short_conv_temporal(mid.data(), frames, per_frame_tokens, channels,
                        *conv, out);
  } else if (out != tokens) {
    for (std::size_t i = 0; i < n; ++i) { out[i] = tokens[i]; }
  }

  // SiLU, then the fla L2 norm over head_dim (q and k only). The norm is
  // accumulated in fp32 with eps 1e-6 -- the eps is inside the sqrt's
  // argument the way F.normalize spells it, i.e. a clamp on the norm
  // rather than a term added to the sum of squares.
  const std::size_t rows = (std::size_t)frames * per_frame_tokens
                           * (std::size_t)heads;
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = out + r * (std::size_t)head_dim;
    for (int i = 0; i < head_dim; ++i) {
      const float v = row[i];
      row[i] = v / (1.0f + std::exp(-v));
    }
    if (!l2norm) { continue; }
    double sum = 0.0;
    for (int i = 0; i < head_dim; ++i) { sum += (double)row[i] * row[i]; }
    double norm = std::sqrt(sum);
    if (norm < 1e-6) { norm = 1e-6; }
    for (int i = 0; i < head_dim; ++i) {
      row[i] = (float)((double)row[i] / norm);
    }
  }
}

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
