#ifndef VPIPE_GENERATIVE_MODELS_SHARED_MMA_TILE_H
#define VPIPE_GENERATIVE_MODELS_SHARED_MMA_TILE_H

// Which dense matmul2d tile a prefill GEMM should dispatch on M5.
//
// Lives here because qwen3 and gemma4 each have more than one call site and
// they must agree: a tile rule that drifts between the two families is a
// silent per-model perf difference.
//
// The two production tiles are dense_gemm_mma_t_n128_f16 (128x128) and
// dense_gemm_mma_t_n128x256_f16 (128x256), both 8 simdgroups. They are
// BIT-IDENTICAL -- verified over 56M outputs at four shapes including ragged
// M and N (optiq_blocks.tile_choice_is_bit_identical) -- because both pass
// the full K as a dynamic contraction extent and differ only in how many
// output columns a threadgroup owns. So this is a pure rate knob: changing
// it owes a benchmark, not a token-exactness run.

#include <cstdlib>

namespace vpipe::genai {

// K at or above which the 128x256 tile wins: the deep-K weight stream
// outpaces the square tile's x-reuse.
constexpr int kMmaDeepK = 6144;

// NEGATIVE RESULT, recorded so it is not re-tried: the rule does NOT want an
// N term. The four large OptiQ checkpoints have enormous fused gate|up
// projections (N = 2*ffn: 34816 on Qwen3.6-27B, 43008 on gemma-4-31B) at
// shallow K, and a first pass at this file routed N >= 16384 to the 128x256
// tile on the strength of a probe that measured 1.07-1.22x there.
//
// That probe was WRONG, and wrong in a way worth remembering: it ran all of
// arm A's rounds and then all of arm B's, so the first arm measured absorbed
// the first-touch cost of freshly allocated buffers and the two arms sat in
// different SoC power states. Re-run with the arms truly interleaved
// (optiq_blocks.mlp_tile_probe, which now warms both before timing either),
// 128x256 measures 0.85-0.95x at M=1024 and 1.03-1.07x at M>=2048 -- no
// consistent winner, and a loss in the prefill regime that matters most.
//
// End-to-end confirms the interleaved reading: Qwen3.5-4B-OptiQ (gate|up
// N=18432, K=2560) prefills 1160 tok/s @1k on the K-only rule and 1074 with
// the N term added -- a 7% REGRESSION. The N term is not in the rule.
//
// VPIPE_MMA_WIDE_N is kept as the re-probe hook: set it to an N threshold to
// re-enable the wide tile for N >= that, e.g. to retest on a future GPU.
// Unset (the default), the rule is K-only.
inline bool mma_use_wide_tile(int N, int K)
{
  static const int wide_n = []() {
    const char* e = std::getenv("VPIPE_MMA_WIDE_N");
    const int v = (e != nullptr) ? std::atoi(e) : 0;
    return v > 0 ? v : 0;
  }();
  return K >= kMmaDeepK || (wide_n > 0 && N >= wide_n);
}

// ---------------------------------------------------------------------
// How many ROWS one matmul2d dispatch may cover, over an N-wide
// destination contracting a K-wide source, at 2 bytes per element.
//
// The mma kernels build their tensors as `dextents<int32_t, 2>`, so MPP
// computes addresses in 32 bits and a tile whose base passes 2^31 BYTES
// silently stops storing -- no error, no partial write, the rows come back
// holding whatever was already in the buffer. Splitting M and rebasing each
// dispatch through the buffer offset keeps every offset the kernel computes
// small. Below the line the band is the whole thing and nothing about the
// encoding changes, so this costs nothing at the shapes that already work.
//
// BOTH extents, because the limit is a property of EVERY operand and not of
// the destination. This lives here, and takes both, because banding on one
// of them is the mistake that has now been made twice:
//
//   MiniMax-H3's DiT banded on N. Three of its four projections cannot show
//   the difference; fc2 can, having the block's smallest N (5376, so its
//   band was 199680 and never fired) and its largest K (14336, so at 75136
//   packed rows the SwiGLU output it contracts is 2.15 GB). It came back as
//   whole-latent video noise.
//
//   The Wan VAE splits by a fixed ROW cap, which bounds its destination
//   (cout <= 384) and not its source. Its causal conv3d has 27 taps where a
//   2D VAE has 9, so K = 27 * cin reaches 10368 and the im2col band alone
//   permits 2.7 GB.
//
// Floored to 128, the mma tiles' row step; a band ending mid-tile would
// still hand one tile an offset past the line. 128 is a multiple of the
// 64-row step the scaled/LoRA tiles use, so one floor serves both.
//
// MEASURED, minimax_h3_blocks.tail_rows_match_a_small_reference at
// [76800, 14336] bf16: the 128-row tiles return 9633792 wrong elements,
// every row from 75008 = ceil128(2^31 / 28672) to the end, and 0 at the
// band. steel, the int8 arm and split-K are clean either way -- all three
// are int64 internally, which is why this is a matrix-core path only.
inline int mma_row_band(int N, int K)
{
  const int W = N > K ? N : K;
  if (W <= 0) { return 1 << 30; }
  const long long lim = (((long long)1 << 31) - 1) / ((long long)W * 2);
  long long band = (lim / 128) * 128;
  if (band < 128) { band = 128; }
  if (band > (1 << 30)) { band = 1 << 30; }
  return (int)band;
}

}  // namespace vpipe::genai

#endif
