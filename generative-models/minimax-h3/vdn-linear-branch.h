#ifndef GENERATIVE_MODELS_MINIMAX_H3_VDN_LINEAR_BRANCH_H
#define GENERATIVE_MODELS_MINIMAX_H3_VDN_LINEAR_BRANCH_H

// VDN-H3's bidirectional linear branch: everything the softmax window
// CANNOT see, summarised for every video token.
//
//   1. features        q, k, v   (short conv on k/v, SiLU, L2-normed q/k)
//   2. per-frame stats A = sum_s beta k k^T,  B = sum_s beta v k^T
//   3. alpha           the KDA double-exponential gate, per frame
//   4. text state      the prompt as ONE delta-rule chunk, halved
//   5. two scans       forward and reverse state banks over frames
//   6. gather          the state just outside the window, decayed in
//   7. readout         q . state, RMSNorm, output gate
//
// This is the CPU reference -- the oracle a Metal port is checked
// against, and slow on purpose. Every stage is separately callable so a
// mismatch says WHICH one, which is the whole reason the goldens tap
// each of them.
//
// FIVE THINGS THE STAGES BELOW GET WRONG SILENTLY:
//
//   * `alpha` is a DOUBLE exponential, exp(-exp(A_log) softplus(...)),
//     with A_log per HEAD and dt_bias per CHANNEL. Reading A_log as
//     per-channel type-checks and changes the retention spectrum.
//   * The frame mean feeding it is taken in fp32 BEFORE any downcast.
//     Rounding it first cannot be recovered by promoting afterwards.
//   * A must be SYMMETRISED. It is formed as (k*beta)^T k, so its (i,j)
//     and (j,i) entries come from different reductions; the Cholesky
//     reads ONE triangle, so an unsymmetrised A means the matrix
//     factorised is not the matrix meant -- and the discarded triangle
//     is exactly the evidence of how wrong the kept one is. This
//     reference's own asymmetry is below the goldens' bar (removing the
//     step here changes nothing measurable), so the step is for the
//     Metal path, where A comes out of a tensor-core GEMM. See the test
//     an_unsymmetrised_A_factorises_a_different_matrix.
//   * BOTH scan directions start from the SAME half text state. Each
//     direction carries the whole prompt, so half each keeps the sum the
//     gather forms at roughly one copy.
//   * With anchor_frames "both" the branch drops frames 0 and F-1 from
//     its INPUT -- not masked afterwards -- and their readout rows are
//     exactly zero, because the softmax side covers them in both
//     directions and the partition has to stay exact.

#include "generative-models/minimax-h3/vdn-geometry.h"
#include "generative-models/minimax-h3/vdn-linear-features.h"

#include <cstddef>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

struct LinearBranchWeights {
  ShortConv k_conv, v_conv;
  const float* beta_proj      = nullptr;   // [heads, hidden]
  const float* alpha_down     = nullptr;   // [head_dim, hidden]
  const float* alpha_up       = nullptr;   // [heads*head_dim, head_dim]
  const float* alpha_dt_bias  = nullptr;   // [heads*head_dim]
  const float* alpha_a_log    = nullptr;   // [heads]
  const float* gate_down      = nullptr;   // [head_dim, hidden]
  const float* gate_up_w      = nullptr;   // [heads*head_dim, head_dim]
  const float* gate_up_b      = nullptr;   // [heads*head_dim]
  const float* norm           = nullptr;   // [head_dim]
};

struct LinearBranchConfig {
  int heads    = 56;
  int head_dim = 128;
  int hidden   = 5376;
  bool bridge_alpha      = true;
  bool enable_text_state = true;
  // anchor_frames == "both": drop frames 0 and F-1 from the input.
  bool skip_ends         = true;
  float text_state_scale = 0.5f;
  float norm_eps         = 1e-6f;
};

// --- the stages, each usable on its own ------------------------------

// beta = sigmoid(beta_proj x). [rows, heads].
void branch_beta(const float* x, int rows, const LinearBranchConfig& cfg,
                 const float* beta_proj, float* beta);

// A [frames, heads, d, d] and B [frames, heads, d, d], from features
// already in [frames*tokens, heads, d] order. A is symmetrised.
void frame_statistics(const float* key, const float* value,
                      const float* beta, int frames, int tokens_per_frame,
                      int heads, int head_dim, float* a, float* b);

// alpha = exp(-exp(A_log) * softplus(up(down(mean_s x)) + dt_bias)),
// [frames, heads, head_dim], fp32 throughout. `frame_mean` is optional:
// pass null to have it taken from `x` here (in fp32, as it must be).
void branch_alpha(const float* x, int frames, int tokens_per_frame,
                  const LinearBranchConfig& cfg,
                  const LinearBranchWeights& w, float* frame_mean,
                  float* alpha);

// The prompt written into a zero state as ONE delta-rule chunk, scaled
// by text_state_scale. [heads, d, d]. False when the solve fails.
bool branch_text_state(const float* text_x, const float* text_k_raw,
                       const float* text_v_raw, int text_len,
                       const LinearBranchConfig& cfg,
                       const LinearBranchWeights& w, float* text_state);

// prefix[t] = frames 0..t, suffix[t] = frames t..F-1, both seeded with
// `text_state` (null for a zero start). [frames, heads, d, d] each.
bool branch_scans(const float* a, const float* b, const float* alpha,
                  int frames, int heads, int head_dim,
                  const float* text_state, float* prefix, float* suffix);

// readout = (q . state), RMSNormed and gated. [frames*tokens, heads*d].
void branch_readout(const float* query, const float* state, const float* x,
                    int frames, int tokens_per_frame,
                    const LinearBranchConfig& cfg,
                    const LinearBranchWeights& w, float* out);

// --- the whole branch -------------------------------------------------
//
// `qkv_raw` are the BACKBONE's raw q/k/v for the video rows -- the
// branch has no projections of its own. `out` is [frames*tokens,
// heads*head_dim]; with skip_ends the two anchor frames' rows are zero.
bool linear_branch_forward(const float* x, const float* q_raw,
                           const float* k_raw, const float* v_raw,
                           int frames, int grid_h, int grid_w,
                           const std::vector<Bound>& bounds,
                           const float* text_x, const float* text_k_raw,
                           const float* text_v_raw, int text_len,
                           const LinearBranchConfig& cfg,
                           const LinearBranchWeights& w, float* out);

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
