#ifndef VPIPE_GENAI_EVAL_LM_EVAL_H
#define VPIPE_GENAI_EVAL_LM_EVAL_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::genai {

class LoadedLanguageModel;

// ---- dataset loaders --------------------------------------------------------
// Evaluation datasets are NOT shipped in the binary; they are fetched on demand
// (model-fetch catalogue entries "wikitext-2-raw-test" / "arc-challenge-test")
// into a directory of HuggingFace datasets-server rows-*.json pages, which
// these loaders read. `dir` is the registry-resolved dataset directory.

// One ARC-Challenge multiple-choice question.
struct ArcItem {
  std::string              question;
  std::vector<std::string> choices;
  int                      answer = -1;   // index into choices
};

// Concatenate the WikiText test rows (each page's rows[].row.text) into one
// text blob. false + err when the dir has no readable rows-*.json pages.
bool load_wikitext_dir(const std::string& dir, std::string& out_text,
                       std::string& err);

// Parse the ARC-Challenge rows (rows[].row.{question,choices,answerKey}) into
// items. false + err when the dir has no readable pages / usable rows.
bool load_arc_dir(const std::string& dir, std::vector<ArcItem>& out,
                  std::string& err);

// ---- results ----------------------------------------------------------------
struct PerplexityResult {
  bool        ok         = false;
  double      perplexity = 0.0;   // exp(avg_nll)
  double      avg_nll    = 0.0;   // mean per-token negative log-likelihood (nats)
  long long   n_tokens   = 0;     // scored next-token predictions
};

struct ArcResult {
  bool   ok       = false;
  int    n        = 0;            // questions scored
  int    correct  = 0;           // by length-normalized log-likelihood
  double accuracy = 0.0;         // correct / n
};

// A-vs-B output-distribution divergence over the SAME token ids. For models
// whose absolute perplexity is meaningless (e.g. a TTS/codec backbone whose
// tied text head is not a language model), this compares two models' per-token
// outputs directly instead.
struct AbDivergenceResult {
  bool   ok     = false;
  double kl     = 0.0;   // mean KL(P_A || P_B) over scored positions (nats)
  double rel_l2 = 0.0;   // mean per-token logit relative-L2 error
  int    n      = 0;     // scored positions
};

// ---- evaluations ------------------------------------------------------------
// Teacher-forced WikiText-2 perplexity over a single contiguous window of up
// to `max_tokens` tokens from the built-in corpus. Prepends BOS (when the
// tokenizer has one), prefills, then feeds each true token (next_token) and
// accumulates the per-position negative log-likelihood of the actual next
// token. Greedy/no sampling.
//
// Caveats: (1) perplexity is over each model's OWN tokenization, so absolute
// values are not directly comparable across different tokenizer families
// (use it within a family, e.g. a model vs its quantization). (2) It uses the
// backend's exposed last-position logits; a family whose production path
// post-processes logits (e.g. Gemma's final soft-cap) may read high here. ARC
// accuracy (below) is robust to both.
PerplexityResult
eval_wikitext2_perplexity(LoadedLanguageModel& lm, std::string_view text,
                          int max_tokens,
                          const SessionContextIntf* session = nullptr);

// ARC-Challenge accuracy over `n_samples` questions drawn deterministically
// (mt19937_64 seeded with `seed`) from `items`. Per question, each choice's
// continuation log-likelihood given the stem ("Question: ...\nAnswer:") is
// scored and the length-normalized argmax is compared to the gold answer.
ArcResult
eval_arc_challenge(LoadedLanguageModel& lm, const std::vector<ArcItem>& items,
                   int n_samples, std::uint64_t seed,
                   const SessionContextIntf* session = nullptr);

// ---- A-vs-B output divergence ----------------------------------------------
// Teacher-forced over a single window of up to `max_tokens` tokens from `text`
// (BOS-prepended like the perplexity probe). Mirrors that loop: prefills the
// first id, then per scored position captures the full [vocab] logit vector
// (last_logits_host()) into `out_logits` (flat, row-major n*vocab) and records
// the EXACT input id sequence fed into `out_ids` so model B can be driven
// identically (do NOT re-tokenize with B). out_vocab = config().vocab_size.
// Returns false if the window is too short to score a position.
//
// NOTE: `out_logits` is n*vocab floats -- e.g. vocab ~152k * 256 tokens is
// ~155 MB. Bound `max_tokens` for huge-vocab models.
bool
capture_token_logits(LoadedLanguageModel& lm, std::string_view text,
                     int max_tokens, std::vector<std::int32_t>& out_ids,
                     std::vector<float>& out_logits, int& out_n, int& out_vocab,
                     const SessionContextIntf* session = nullptr);

// Drive model B with the SAME `ids` captured by capture_token_logits (prefill
// ids[0], then next_token over the rest) and, at each captured position,
// compute over the full [vocab]:
//   P_A    = softmax(a_logits[t])      (numerically stable: subtract row max)
//   logP_B = log_softmax(b_logits)
//   KL(P_A||P_B) = sum P_A (logP_A - logP_B)   (>= 0)
//   logit rel-L2 = ||a - b||_2 / max(||a||_2, eps)
// and averages each over the n scored positions. `ok=false` when B's vocab
// does not match `vocab` (KL needs a shared tokenizer/vocab).
AbDivergenceResult
ab_divergence(LoadedLanguageModel& lm_b, const std::vector<std::int32_t>& ids,
              const std::vector<float>& a_logits, int a_n, int vocab,
              const SessionContextIntf* session = nullptr);

}  // namespace vpipe::genai

#endif
