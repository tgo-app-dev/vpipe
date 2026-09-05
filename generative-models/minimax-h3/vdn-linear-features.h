#ifndef GENERATIVE_MODELS_MINIMAX_H3_VDN_LINEAR_FEATURES_H
#define GENERATIVE_MODELS_MINIMAX_H3_VDN_LINEAR_FEATURES_H

// VDN-H3 linear branch, stage 1: the q/k/v features.
//
//     [ShortConv ->] SiLU [-> L2Norm]
//
// The branch SHARES the softmax branch's projections -- it consumes the
// raw pre-QK-norm, pre-RoPE q/k/v and adds no projection of its own,
// which is why its width is exactly H3's 56 x 128 and why a Stage-B LoRA
// on attn.orig.to_{q,k,v} feeds both branches at once. There is no RoPE
// here: the branch is NoPE by construction, and position enters only
// through the frame recurrence.
//
// FOUR THINGS A PORT GETS WRONG SILENTLY, all of them pinned by the
// goldens in tests/unit-tests/vdn-linear-features.cc:
//
//   1. WHICH projections are convolved. The released checkpoints
//      convolve K and V and leave Q as the raw NoPE features. A conv on
//      Q is a plausible-looking blur.
//   2. The temporal half is a CORRELATION over frames with ZERO padding
//      and no causality: out[t] = sum_dt w[dt] x[t + dt - 2], crossing
//      VAE chunk boundaries in both directions. A causal reading loses
//      the future half of every stencil.
//   3. V is NOT L2-normalised. Only q and k are. Normalising v changes
//      what B means and the branch norm downstream hides the magnitude
//      error, so it survives every scale-invariant check.
//
// The two halves COMMUTE -- they act on disjoint axes, the 5x5 being
// identical for every frame and the 5-tap identical for every spatial
// position -- so an implementation may fuse them in either order, which
// is what lets the inference path put its epilogue on the temporal half.
// The effective 3D kernel is the outer product w_tm x w_sp either way:
// 30 free parameters per channel rather than 125, restricted to rank-1
// space-time.
//
// The L2 norm is fla's: accumulated in fp32 with eps 1e-6.

#include <cstddef>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

// One projection's depthwise short conv. Both halves are bias-free and
// per CHANNEL, where a channel is (head, head_dim) flattened -- 7168 of
// them at H3's shape.
struct ShortConv {
  const float* spatial  = nullptr;   // [channels, K, K], row-major
  const float* temporal = nullptr;   // [channels, K]
  int kernel = 5;
};

// tokens / out: [frames * grid_h * grid_w, heads, head_dim], the spatial
// index row-major over (grid_h, grid_w).
//
// `conv` null leaves the projection unconvolved, which is what Q takes.
// `l2norm` false is what V takes.
//
// In-place is allowed (out may alias tokens).
void linear_features(const float* tokens, int frames, int grid_h, int grid_w,
                     int heads, int head_dim, const ShortConv* conv,
                     bool l2norm, float* out);

// The two halves on their own, for a caller that wants to fuse the
// epilogue (the inference path does) and for tests that need to see
// which half a mismatch is in.
void short_conv_spatial(const float* tokens, int frames, int grid_h,
                        int grid_w, int channels, const ShortConv& conv,
                        float* out);
void short_conv_temporal(const float* x, int frames, int tokens_per_frame,
                         int channels, const ShortConv& conv, float* out);

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
