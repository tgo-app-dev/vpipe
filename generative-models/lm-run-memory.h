// lm-run-memory.h -- what a language model allocates to RUN, beside its
// weights, for the plan an LM stage makes before it loads.
//
// Host-only (not installed): the LM stages book it as a scratch claim and
// the loader sizes a routed-expert cache beside it; no plugin sizes a
// host model.

#ifndef VPIPE_GENERATIVE_MODELS_LM_RUN_MEMORY_H
#define VPIPE_GENERATIVE_MODELS_LM_RUN_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::genai {

// The built-in families a config.json `architecture` loads as (the one
// list LoadedLanguageModel picks its exec by); kNone for any other -- a
// plugin's arch, or one no exec takes.
enum class LmFamily { kNone, kQwen, kGemma, kLlama, kDeepseekV41 };
LmFamily lm_family(const std::string& architecture);

// HOW A STAGE RUNS its language model -- what moves the figure.
struct LmRunShape {
  // Tokens of K/V the stage may hold at once, across its contexts: its
  // page pool (page_tokens * max_pages), or less where it knows its own
  // bound. A context's prompt is at most the model's own per-context cap.
  std::int64_t kv_tokens = 0;
  // The pool the model is loaded with (LoadSpec's), which caps how far
  // the paged K/V grows.
  int page_tokens = 512;
  std::uint32_t max_pages = 4096;
  // Contexts held at once: each keeps its own recurrent state and rings.
  int contexts = 1;
  // > 0: the contexts past the first are BRANCHES of it, sharing its pages
  // and each adding this many tokens of its own (a question's turn and
  // answer); the K/V is then the base -- at most the model's per-context
  // cap -- and the branches'. 0: the contexts are independent, each at
  // most that cap. Either way never past the pool.
  std::int64_t branch_tokens = 0;
  // The whole prompt in one forward: a Qwen MTP head's prefix seed, a
  // DFlash drafter's taps.
  bool one_pass = false;
  // DeepSeek-V4.1's own draft head (DSpark) loaded.
  bool draft_head = false;
};

// The bytes a load of `dir` allocates to run in `shape` -- its K/V and
// per-context state, a prefill's activations, the decode scratch -- by
// the family's own figure (MetalQwenModel / MetalGemmaModel /
// MetalLlamaModel / MetalDeepseekV41Model::run_bytes). Not its weights,
// not its vision tower's encode (lm_claims books that), not a drafter's
// verify scratch (text-chat books that). 0 for a config it cannot read
// or a family it does not know (a plugin's).
std::size_t lm_run_bytes(const SessionContextIntf* session,
                         const std::string& dir, const LmRunShape& shape);

// The plan's label for it: one arena per checkpoint, as the stages that
// name one directory share its loaded model -- and with it the pool and
// the scratch (two claims under one label count once, at the larger).
std::string lm_run_label(const std::string& dir);

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_LM_RUN_MEMORY_H
