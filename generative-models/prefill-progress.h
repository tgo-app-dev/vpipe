#ifndef VPIPE_GENERATIVE_MODELS_PREFILL_PROGRESS_H
#define VPIPE_GENERATIVE_MODELS_PREFILL_PROGRESS_H

// A PREFILL'S PIECES, AS THEY FINISH. A long prompt is read in pieces by
// the execs that cut it (a piece's activations scale with its rows); a
// caller timing what a model reads -- a chat's live speed -- is told
// after each piece how far the prefill has come. An exec that can do this
// answers ModelExec::query_extension(kExtensionId) with a
// PrefillProgressExt*; one that cannot (or reads the prompt in one
// forward) answers null, and the caller times the prefill whole.
// LoadedLanguageModel::set_prefill_progress wraps it.
//
// Called on the forward's thread once each piece's GPU work is done:
// `done` the prompt's tokens read so far, `total` the prefill call's --
// the last call `done == total`. A prefill read in one forward calls it
// once, at its end, or not at all.

#include <functional>
#include <string_view>

namespace vpipe::genai {

namespace prefill_progress {

inline constexpr std::string_view kExtensionId = "vpipe.prefill-progress/1";

using Fn = std::function<void(int done, int total)>;

}  // namespace prefill_progress

struct PrefillProgressExt {
  virtual ~PrefillProgressExt() = default;
  // `fn` for every prefill from now on; an empty one: none.
  virtual void set_prefill_progress(prefill_progress::Fn fn) = 0;
};

}  // namespace vpipe::genai

#endif
