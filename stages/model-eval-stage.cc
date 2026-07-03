#include "stages/model-eval-stage.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/eval/lm-eval.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include <memory>
#include <utility>
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

ModelEvalStage::ModelEvalStage(const SessionContextIntf* s,
                               std::string               id,
                               std::vector<InEdge>       iports,
                               FlexData                  config)
  : TypedStage<ModelEvalStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  _model_a     = attr_str("model_a");
  _model_b     = attr_str("model_b");
  _models_db   = attr_str("models_db");
  _wikitext    = attr_str("wikitext");
  _arc         = attr_str("arc");
  _ppl_tokens  = static_cast<int>(attr_uint("ppl_tokens"));
  _arc_samples = static_cast<int>(attr_uint("arc_samples"));
  _seed        = static_cast<std::uint64_t>(attr_uint("seed"));
  _divergence  = attr_bool("divergence");
  if (_models_db.empty()) { _models_db = "models"; }

  if (_model_a.empty()) {
    fail_config(fmt("ModelEvalStage('{}'): config.model_a is required",
                    this->id()));
  }
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "model_a", .type = ConfigType::String, .required = true,
   .doc = "first model: a models-DB key (model-fetch / model-quantize) or a "
          "model directory path",
   .suggest_db = "models"},
  {.key = "model_b", .type = ConfigType::String,
   .doc = "OPTIONAL second model (key or path); when set, the report is a "
          "side-by-side comparison with deltas",
   .def_str = "", .suggest_db = "models"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db for model-key resolution", .def_str = "models"},
  {.key = "wikitext", .type = ConfigType::String,
   .doc = "WikiText-2 dataset: a models-DB key (fetch with model-fetch) or a "
          "dir path; empty/not-found skips the perplexity probe",
   .def_str = "wikitext-2-raw-test",
   .suggest_db = "models", .suggest_db_type = "eval-wikitext2"},
  {.key = "arc", .type = ConfigType::String,
   .doc = "ARC-Challenge dataset: a models-DB key or a dir path; empty/not-"
          "found skips the ARC probe",
   .def_str = "arc-challenge-test",
   .suggest_db = "models", .suggest_db_type = "eval-arc-challenge"},
  {.key = "ppl_tokens", .type = ConfigType::Uint,
   .doc = "WikiText-2 tokens to score for perplexity (0 => all)",
   .def_uint = 1024},
  {.key = "arc_samples", .type = ConfigType::Uint,
   .doc = "ARC-Challenge questions to random-sample (0 => the whole pool)",
   .def_uint = 40},
  {.key = "seed", .type = ConfigType::Uint,
   .doc = "RNG seed for the ARC-Challenge sample", .def_uint = 1234},
  {.key = "divergence", .type = ConfigType::Bool,
   .doc = "A-vs-B output KL divergence + logit relative error (for models "
          "whose absolute perplexity is meaningless, e.g. TTS/codec backbones "
          "like MOSS-TTS-Local-v1.5); needs model_b. A's per-token logits are "
          "buffered (ppl_tokens * vocab floats, ~155 MB at vocab 152k / 256 "
          "tokens) -- lower ppl_tokens for huge-vocab models.",
   .def_bool = false},
};
const StageSpec kSpec = {
  .type_name = "model-eval",
  .doc       = "Source: one-shot offline evaluation of one or two language "
               "models (models-DB keys or paths). Runs WikiText-2 perplexity "
               "+ a random-sampled ARC-Challenge accuracy probe and logs a "
               "Markdown report (a comparison when two models are given). "
               "0 in / 0 out.",
  .display_name = "Model Eval",
  .category  = StageCategory::Preparation,
  .iports    = {},
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ModelEvalStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

struct Outcome {
  bool                     loaded = false;
  std::string              ref;
  genai::PerplexityResult  ppl;
  genai::ArcResult         arc;
  genai::AbDivergenceResult div;   // A->B divergence (populated on B's pass)
};

std::string ppl_cell_(const genai::PerplexityResult& p)
{
  return p.ok ? fmt("{:.3f}", p.perplexity)() : std::string("n/a");
}
std::string arc_cell_(const genai::ArcResult& a)
{
  return a.ok ? fmt("{:.1f}% ({}/{})", a.accuracy * 100.0, a.correct, a.n)()
              : std::string("n/a");
}
std::string kl_cell_(const genai::AbDivergenceResult& d)
{
  return d.ok ? fmt("{:.6f}", d.kl)() : std::string("n/a");
}
std::string rel_cell_(const genai::AbDivergenceResult& d)
{
  return d.ok ? fmt("{:.6f}", d.rel_l2)() : std::string("n/a");
}

// Built-in fallback paragraph for the A-vs-B divergence when no WikiText-2
// corpus is loaded -- it only needs tokens to feed both models. Includes CJK
// so multilingual (MOSS-class) models are exercised.
constexpr const char* kDivergenceFallback =
  "The quick brown fox jumps over the lazy dog. Paris is the capital of "
  "France, and the river Seine flows through the heart of the city. "
  "Machine learning models predict the next token from the preceding "
  "context. 人工智能正在迅速改变我们的世界。今天天气晴朗，适合出去散步。"
  "The cat sat quietly on the warm windowsill in the afternoon sun. "
  "東京是日本的首都，也是世界上人口最多的城市之一。Water boils at one "
  "hundred degrees Celsius and freezes at zero. 学而时习之，不亦说乎？"
  "The stars appeared one by one as the evening sky slowly darkened. "
  "Numbers, letters, and symbols all become tokens for the model to read.";

}  // namespace
#endif  // VPIPE_BUILD_APPLE_SILICON

bool
ModelEvalStage::evaluate_once()
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  auto* mgr = session() ? session()->generative_model_manager() : nullptr;
  if (mgr == nullptr) {
    session()->warn(fmt(
        "ModelEvalStage('{}'): no GenerativeModelManager (is this an "
        "apple-silicon build?)", this->id()));
    return false;
  }

  // A-vs-B output divergence (KL + logit rel-L2) for models whose absolute
  // perplexity is meaningless; needs two models and does NOT need a dataset.
  const bool want_div = _divergence && !_model_b.empty();
  if (_divergence && _model_b.empty()) {
    session()->warn(fmt(
        "ModelEvalStage('{}'): divergence=true but model_b is empty; the "
        "A-vs-B divergence needs two models (skipping it)", this->id()));
  }

  // Load the fetched datasets once (resolved via the models DB). A missing /
  // unfetched dataset just disables that probe (reported as n/a).
  std::string wt_text;
  std::vector<genai::ArcItem> arc_items;
  bool have_wt = false, have_arc = false;
  if (!_wikitext.empty()) {
    const std::string dir = resolve_model_dir(session(), _models_db, _wikitext);
    std::string e;
    have_wt = genai::load_wikitext_dir(dir, wt_text, e);
    if (!have_wt) {
      session()->warn(fmt(
          "ModelEvalStage('{}'): WikiText-2 unavailable ({}); skipping "
          "perplexity", this->id(), e));
    }
  }
  if (!_arc.empty()) {
    const std::string dir = resolve_model_dir(session(), _models_db, _arc);
    std::string e;
    have_arc = genai::load_arc_dir(dir, arc_items, e);
    if (!have_arc) {
      session()->warn(fmt(
          "ModelEvalStage('{}'): ARC-Challenge unavailable ({}); skipping ARC",
          this->id(), e));
    }
  }
  if (!have_wt && !have_arc && !want_div) {
    session()->warn(fmt(
        "ModelEvalStage('{}'): no evaluation datasets available -- fetch "
        "'wikitext-2-raw-test' / 'arc-challenge-test' with model-fetch",
        this->id()));
    return false;
  }

  // Divergence comparison text: the loaded WikiText-2 corpus if present, else
  // a built-in multilingual fallback (the divergence only needs tokens).
  const std::string div_text = have_wt ? wt_text
                                       : std::string(kDivergenceFallback);
  // A's captured per-token logits (flat n*vocab) -- filled on A's pass and
  // replayed against B, then dropped.
  std::vector<std::int32_t> div_ids;
  std::vector<float>        a_logits;
  int                       a_n = 0, a_vocab = 0;

  auto run_one = [&](const std::string& ref, bool is_a) -> Outcome {
    Outcome o; o.ref = ref;
    const std::string dir = resolve_model_dir(session(), _models_db, ref);
    session()->info(fmt(
        "ModelEvalStage('{}'): loading '{}' for evaluation", this->id(), ref));
    genai::LoadSpec spec;
    spec.hf_dir = dir;
    std::shared_ptr<genai::LoadedLanguageModel> lm = mgr->load(spec);
    if (!lm || !lm->valid()) {
      session()->warn(fmt(
          "ModelEvalStage('{}'): failed to load '{}'", this->id(), ref));
      return o;
    }
    o.loaded = true;
    // A-vs-B divergence: capture A's logits now (before A is released); on B's
    // pass, replay the SAME ids against B and compute KL + logit rel-L2.
    if (want_div && is_a) {
      session()->info(fmt(
          "ModelEvalStage('{}'): [{}] capturing logits for A-vs-B divergence "
          "(<= {} tokens)...", this->id(), ref, _ppl_tokens));
      if (!genai::capture_token_logits(*lm, div_text, _ppl_tokens, div_ids,
                                       a_logits, a_n, a_vocab, session())) {
        session()->warn(fmt(
            "ModelEvalStage('{}'): logit capture for divergence failed "
            "(text too short?)", this->id()));
      }
    }
    if (want_div && !is_a && a_n > 0) {
      session()->info(fmt(
          "ModelEvalStage('{}'): [{}] A-vs-B output divergence...",
          this->id(), ref));
      o.div = genai::ab_divergence(*lm, div_ids, a_logits, a_n, a_vocab,
                                   session());
    }
    if (have_wt) {
      session()->info(fmt(
          "ModelEvalStage('{}'): [{}] WikiText-2 perplexity ({} tokens)...",
          this->id(), ref, _ppl_tokens));
      o.ppl = genai::eval_wikitext2_perplexity(*lm, wt_text, _ppl_tokens,
                                               session());
    }
    if (have_arc) {
      session()->info(fmt(
          "ModelEvalStage('{}'): [{}] ARC-Challenge ({} samples)...",
          this->id(), ref, _arc_samples));
      o.arc = genai::eval_arc_challenge(*lm, arc_items, _arc_samples, _seed,
                                        session());
    }
    lm.reset();   // release before the next model loads (peak memory = 1 model)
    return o;
  };

  const Outcome a = run_one(_model_a, /*is_a=*/true);
  if (!a.loaded) { return false; }
  const bool two = !_model_b.empty();
  Outcome b;
  if (two) {
    b = run_one(_model_b, /*is_a=*/false);
    if (!b.loaded) {
      session()->warn(fmt(
          "ModelEvalStage('{}'): model_b failed to load; reporting model_a "
          "only", this->id()));
    }
  }

  // ---- Markdown report ----
  std::string md;
  if (!two || !b.loaded) {
    md += fmt("## Model evaluation: {}\n\n", a.ref)();
    md += "| Metric | Value |\n|---|---|\n";
    md += fmt("| WikiText-2 perplexity (lower is better) | {} |\n",
              ppl_cell_(a.ppl))();
    md += fmt("| ARC-Challenge accuracy (higher is better) | {} |\n",
              arc_cell_(a.arc))();
  } else {
    md += "## Model evaluation comparison\n\n";
    md += fmt("- **A**: {}\n- **B**: {}\n\n", a.ref, b.ref)();
    md += "| Metric | A | B | Δ (B − A) |\n|---|---|---|---|\n";
    // Perplexity (lower better).
    std::string dppl = "n/a";
    if (a.ppl.ok && b.ppl.ok) {
      const double d = b.ppl.perplexity - a.ppl.perplexity;
      dppl = fmt("{}{:.3f}", d >= 0.0 ? "+" : "", d)();
    }
    md += fmt("| WikiText-2 perplexity ↓ | {} | {} | {} |\n",
              ppl_cell_(a.ppl), ppl_cell_(b.ppl), dppl)();
    // ARC accuracy (higher better), delta in percentage points.
    std::string darc = "n/a";
    if (a.arc.ok && b.arc.ok) {
      const double d = (b.arc.accuracy - a.arc.accuracy) * 100.0;
      darc = fmt("{}{:.1f}pp", d >= 0.0 ? "+" : "", d)();
    }
    md += fmt("| ARC-Challenge accuracy ↑ | {} | {} | {} |\n",
              arc_cell_(a.arc), arc_cell_(b.arc), darc)();
    // A-vs-B output divergence (single A->B value; for non-text models whose
    // absolute perplexity is meaningless). The value sits in the A column.
    if (want_div) {
      md += fmt("| A→B KL divergence (nats) ↓ | {} |  |  |\n",
                kl_cell_(b.div))();
      md += fmt("| A→B logit rel-L2 ↓ | {} |  |  |\n",
                rel_cell_(b.div))();
    }
  }
  session()->info(fmt("\n{}", md));
  session()->log_normal(fmt(
      "ModelEvalStage('{}'): evaluation complete", this->id()));
  return true;
#else
  session()->warn(fmt(
      "ModelEvalStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; the LM "
      "subsystem is unavailable", this->id()));
  return false;
#endif
}

Job
ModelEvalStage::process(RuntimeContext& ctx)
{
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }
  session()->info(fmt(
      "ModelEvalStage('{}'): evaluating '{}'{}", this->id(), _model_a,
      _model_b.empty() ? std::string()
                       : fmt(" vs '{}'", _model_b)()));
  evaluate_once();
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelEvalStage)
VPIPE_REGISTER_SPEC(ModelEvalStage, kSpec)

}  // namespace vpipe
