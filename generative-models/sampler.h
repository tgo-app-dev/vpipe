#ifndef VPIPE_GENERATIVE_MODELS_SAMPLER_H
#define VPIPE_GENERATIVE_MODELS_SAMPLER_H

#include "common/flex-data.h"

#include <cstdint>
#include <random>
#include <span>
#include <unordered_set>

namespace vpipe::genai {

// Parameters for the token sampler. Defaults reduce to greedy argmax
// so a caller that doesn't configure anything keeps the legacy
// deterministic behaviour.
//
// Parameter order matters for both correctness and clarity:
//   1. repetition_penalty / presence_penalty (modify raw logits).
//   2. temperature (scale logits).
//   3. softmax to get probabilities.
//   4. top_k filter (keep only the top k probabilities).
//   5. min_p filter (drop tokens below min_p * max_prob).
//   6. top_p filter (nucleus: keep the smallest prefix summing to p).
//   7. renormalise and multinomial-sample.
//
// Notes:
//   * temperature <= 0 forces argmax (greedy). Useful for unit tests
//     and for callers that want deterministic decode independent of
//     the sampler's other knobs.
//   * top_k = 0  -> disabled.
//   * top_p >= 1 -> disabled.
//   * min_p <= 0 -> disabled.
//   * repetition_penalty == 1.0 -> disabled.
//   * presence_penalty == 0.0   -> disabled.
//   * seed == 0 -> fresh non-deterministic seed at construction; any
//     other value seeds the engine reproducibly.
struct SamplerParams {
  float         temperature        = 1.0f;
  int           top_k              = 0;
  float         top_p              = 1.0f;
  float         min_p              = 0.0f;
  float         repetition_penalty = 1.0f;
  float         presence_penalty   = 0.0f;
  std::uint64_t seed               = 0;
};

class Sampler {
public:
  explicit Sampler(SamplerParams p);
  ~Sampler() = default;

  Sampler(const Sampler&)            = delete;
  Sampler& operator=(const Sampler&) = delete;
  Sampler(Sampler&&)                 = default;
  Sampler& operator=(Sampler&&)      = default;

  // True when every parameter is at its no-op default. Callers can
  // short-circuit to a cheap argmax over the logits in this case
  // (and the batched-decode path can keep its argmax fast path).
  bool is_argmax() const noexcept;

  // Sample one token id from `logits` (length vocab_size). Updates
  // the internal seen-token set so the next call's repetition /
  // presence penalties see this id. Returns -1 if vocab_size <= 0.
  std::int32_t
  sample(std::span<const float> logits);

  // Sample one token from a PRE-EXTRACTED top-K view of the raw
  // logits. `top_vals` (length K) holds the K largest raw logits in
  // any order; `top_idx` (length K) holds the corresponding vocab
  // ids (1:1 with top_vals). Used by the batched-decode path that
  // does the top-K on GPU before host-pulling: pulling [N, K] is
  // orders of magnitude cheaper than [N, vocab], and the per-branch
  // CPU work drops from O(vocab) to O(K).
  //
  // Equivalence: when params.top_k > 0 and top_k <= K, the returned
  // distribution is mathematically identical to sample(full_logits),
  // because tokens outside top-K of unpenalized logits are also
  // outside top-K of penalized logits (penalty only depresses) and
  // would be dropped by the top_k filter inside sample() anyway.
  // When params.top_k == 0 but K is large enough to cover the
  // effective support of top_p / min_p (e.g. K=512 with default
  // temperature), the divergence is below FP32 noise.
  //
  // Updates the seen-token set with the picked id, same as sample().
  // Returns -1 if K <= 0 or shapes mismatch.
  std::int32_t
  sample_topk(std::span<const float>        top_vals,
              std::span<const std::int32_t> top_idx);

  // Prime the seen-token set with prompt / prior-context tokens so
  // the first sampled token already penalises them under the
  // repetition / presence penalty rules. Idempotent — call after
  // every prefill if you want the penalty to remember the prompt.
  void
  prime(std::span<const std::int32_t> tokens);

  // Forget all seen tokens. Useful when reusing a Sampler across
  // independent generation contexts.
  void reset();

  const SamplerParams& params() const noexcept { return _params; }

private:
  SamplerParams                          _params;
  std::unordered_set<std::int32_t>       _seen;
  std::mt19937_64                        _rng;
  // Reusable buffer so per-step sampling doesn't churn the heap.
  std::vector<float>                     _scratch;
};

// Parse a stage's "sampler" config block (a FlexData object) into a
// SamplerParams. Unknown keys are ignored; missing keys fall back to
// the struct defaults (which already reduce to greedy argmax).
//
// Expected schema (all fields optional):
//   {
//     "temperature":        real    (default 1.0; <= 0 = argmax),
//     "top_k":              int     (default 0   = disabled),
//     "top_p":              real    (default 1.0 = disabled),
//     "min_p":              real    (default 0.0 = disabled),
//     "repetition_penalty": real    (default 1.0 = disabled),
//     "presence_penalty":   real    (default 0.0 = disabled),
//     "seed":               uint    (default 0   = non-deterministic)
//   }
SamplerParams
parse_sampler_config(const FlexData& obj);

}

#endif
