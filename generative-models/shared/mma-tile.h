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

}  // namespace vpipe::genai

#endif
