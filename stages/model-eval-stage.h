#ifndef VPIPE_STAGES_MODEL_EVAL_STAGE_H
#define VPIPE_STAGES_MODEL_EVAL_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>

namespace vpipe {

// 1 optional trigger iport (any beat) + 1 FlexData "summary" oport (for
// cascading a preparation recipe / a save-text report). One-shot offline
// model evaluation.
// Loads one or two language models (each a models-DB key or a directory path),
// runs WikiText-2 perplexity + a random-sampled ARC-Challenge accuracy probe on
// each, and logs a Markdown report through session()->info(). With two models a
// side-by-side comparison (with deltas) is produced. Pure read-only;
// emits no Beats. Models are loaded + evaluated one at a time (the first is
// released before the second loads) so peak memory stays at one model.
//
// Config (FlexData object):
//   model_a     (string, required) -- first model: a models-DB key or a path.
//   model_b     (string, optional) -- second model; empty => single-model eval.
//   models_db   (string, default "models") -- LMDB sub-db for key resolution.
//   wikitext    (string, default "wikitext-2-raw-test") -- the WikiText-2
//               dataset: a models-DB key (fetch it with model-fetch) or a dir
//               path. Empty / not-found => the perplexity probe is skipped.
//   arc         (string, default "arc-challenge-test") -- the ARC-Challenge
//               dataset (key or path); empty / not-found => ARC is skipped.
//   ppl_tokens  (uint, default 1024) -- WikiText-2 tokens to score (0 => all).
//   arc_samples (uint, default 40) -- ARC-Challenge questions to sample
//               (0 => the whole fetched pool).
//   seed        (uint, default 1234) -- RNG seed for the ARC sample.
//   divergence  (bool, default false) -- A-vs-B output KL divergence + logit
//               relative error (for models whose absolute perplexity is
//               meaningless, e.g. TTS/codec backbones); needs model_b.
//
// The evaluation datasets are NOT shipped with vpipe; fetch them on demand via
// the model-fetch catalogue entries "wikitext-2-raw-test" / "arc-challenge-
// test" (keeps their licenses off vpipe's Apache-2.0 binary).
class ModelEvalStage final : public TypedStage<ModelEvalStage> {
public:
  static constexpr const char* kTypeName = "model-eval";

  ModelEvalStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only inspectors.
  const std::string& model_a() const noexcept { return _model_a; }
  const std::string& model_b() const noexcept { return _model_b; }
  int ppl_tokens()  const noexcept { return _ppl_tokens; }
  int arc_samples() const noexcept { return _arc_samples; }
  bool divergence() const noexcept { return _divergence; }

  // Test seam: run the evaluation once. Returns true on success (a report was
  // produced); logs + returns false on error (no manager / load failure).
  // `summary` is filled with the FlexData work-summary (scores + the
  // Markdown report in its `text` field) on success; untouched on error.
  bool evaluate_once(FlexData& summary);
  // Convenience overload for callers that don't need the summary (tests).
  bool evaluate_once();

private:
  std::string   _model_a;
  std::string   _model_b;
  std::string   _models_db;
  std::string   _wikitext;
  std::string   _arc;
  int           _ppl_tokens{};
  int           _arc_samples{};
  std::uint64_t _seed{};
  bool          _divergence{};
};

}  // namespace vpipe

#endif
