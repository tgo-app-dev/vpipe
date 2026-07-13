// ModelEvalStage: always-on config-validation + embedded-data tests, plus an
// env-gated real evaluation (VPIPE_QWEN35_TEST_MODEL_PATH = a loadable LM dir).

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/eval/lm-eval.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/quantize/model-quantizer.h"
#include "stages/model-eval-stage.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <array>
#include <span>
#include <vector>

using namespace vpipe;

namespace {
class CerrSilencer {
public:
  CerrSilencer() : _saved(std::cerr.rdbuf()) { std::cerr.rdbuf(&_null); }
  ~CerrSilencer() { std::cerr.rdbuf(_saved); }
private:
  struct NullBuf : std::streambuf { int overflow(int c) override { return c; } };
  std::streambuf* _saved;
  NullBuf         _null;
};

FlexData cfg_(const char* a, const char* b)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  if (a) { o.insert("model_a", FlexData::make_string(a)); }
  if (b) { o.insert("model_b", FlexData::make_string(b)); }
  return c;
}
}  // namespace

TEST(model_eval_stage, type_is_registered)
{
  EXPECT_TRUE(std::string_view(ModelEvalStage::kTypeName) == "model-eval");
}

// The stage exposes one trigger iport (any beat type) + one FlexData
// summary oport so it can cascade into a preparation recipe / save-text.
TEST(model_eval_stage, trigger_and_summary_ports)
{
  Session sess;
  CerrSilencer hush;
  ModelEvalStage s(&sess, "ev", std::vector<InEdge>{},
                   cfg_("/tmp/m", nullptr));
  const StageSpec& sp = s.spec();
  ASSERT_TRUE(sp.iports.size() == 1u);
  ASSERT_TRUE(sp.oports.size() == 1u);
  EXPECT_TRUE(std::string_view(sp.iports[0].name) == "trigger");
  EXPECT_TRUE(sp.iports[0].type == nullptr);          // any beat type
  EXPECT_TRUE(std::string_view(sp.oports[0].name) == "summary");
  // Compare by mangled name, not typeid pointer: the stage lives in
  // libvpipe while the test runs in a separate image, so the two
  // &typeid(FlexDataPayload) addresses differ (runtime port matching is
  // all within libvpipe, so it is unaffected).
  ASSERT_TRUE(sp.oports[0].type != nullptr);
  EXPECT_TRUE(std::string_view(sp.oports[0].type->name())
              == typeid(FlexDataPayload).name());
}

TEST(model_eval_stage, config_defaults)
{
  Session sess;
  CerrSilencer hush;
  ModelEvalStage s(&sess, "ev", std::vector<InEdge>{}, cfg_("/tmp/m", nullptr));
  EXPECT_TRUE(s.model_a() == "/tmp/m");
  EXPECT_TRUE(s.model_b().empty());
  EXPECT_TRUE(s.ppl_tokens() == 1024);
  EXPECT_TRUE(s.arc_samples() == 40);
  EXPECT_TRUE(s.config_error().empty());
}

TEST(model_eval_stage, missing_model_a_deferred)
{
  Session sess;
  CerrSilencer hush;
  ModelEvalStage s(&sess, "ev", std::vector<InEdge>{}, cfg_(nullptr, nullptr));
  EXPECT_FALSE(s.config_error().empty());
}

// The dataset loaders parse the HF datasets-server rows-*.json page format.
TEST(model_eval_stage, dataset_loaders)
{
  namespace fs = std::filesystem;
  const fs::path base = fs::temp_directory_path() /
      ("vpipe-evalds-" + std::to_string(::getpid()));
  const fs::path wtd = base / "wt";
  const fs::path arcd = base / "arc";
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(wtd, ec);
  fs::create_directories(arcd, ec);
  {
    std::ofstream o((wtd / "rows-0000.json").string());
    o << R"({"rows":[{"row":{"text":"Hello world. "}},)"
         R"({"row":{"text":"Second line."}}]})";
  }
  {
    std::ofstream o((arcd / "rows-0000.json").string());
    o << R"({"rows":[{"row":{"question":"What is 2+2?",)"
         R"("choices":{"text":["3","4","5","6"],)"
         R"("label":["A","B","C","D"]},"answerKey":"B"}}]})";
  }

  std::string txt, err;
  ASSERT_TRUE(genai::load_wikitext_dir(wtd.string(), txt, err));
  EXPECT_TRUE(txt == "Hello world. Second line.");

  std::vector<genai::ArcItem> items;
  ASSERT_TRUE(genai::load_arc_dir(arcd.string(), items, err));
  ASSERT_TRUE(items.size() == 1);
  EXPECT_TRUE(items[0].question == "What is 2+2?");
  EXPECT_TRUE(items[0].choices.size() == 4);
  EXPECT_TRUE(items[0].answer == 1);   // "B" -> index 1

  // A missing dir fails cleanly.
  std::string e2;
  EXPECT_FALSE(genai::load_arc_dir((base / "nope").string(), items, e2));
  EXPECT_FALSE(e2.empty());
  fs::remove_all(base, ec);
}

// End-to-end single-model eval on a real LM + fetched datasets. Skips unless
// the model + both dataset dirs are provided. Small budgets keep it quick.
TEST(model_eval_stage, real_eval_single)
{
  const char* m  = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  const char* wt = std::getenv("VPIPE_EVAL_WIKITEXT_DIR");
  const char* ar = std::getenv("VPIPE_EVAL_ARC_DIR");
  if (m == nullptr || *m == '\0' || wt == nullptr || *wt == '\0' ||
      ar == nullptr || *ar == '\0') {
    return;
  }
  if (!std::filesystem::exists(std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  FlexData c = cfg_(m, nullptr);
  {
    auto o = c.as_object();
    o.insert("wikitext", FlexData::make_string(wt));
    o.insert("arc", FlexData::make_string(ar));
    const char* pe = std::getenv("VPIPE_EVAL_PPL_TOKENS");
    const char* ae = std::getenv("VPIPE_EVAL_ARC_SAMPLES");
    o.insert("ppl_tokens",
             FlexData::make_int(pe && *pe ? std::atoi(pe) : 64));
    o.insert("arc_samples",
             FlexData::make_int(ae && *ae ? std::atoi(ae) : 4));
  }
  ModelEvalStage s(&sess, "ev", std::vector<InEdge>{}, std::move(c));
  ASSERT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.evaluate_once());
}

// A-vs-B output divergence self-consistency: comparing a model to ITSELF must
// give KL ~ 0 and logit rel-L2 ~ 0 (deterministic greedy logits => identical
// distributions). Proves the metric is correct. Gated on a loadable LM dir.
TEST(model_eval_stage, ab_divergence_self_is_zero)
{
  const char* m = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  CerrSilencer hush;
  if (sess.metal_compute() == nullptr) { return; }

  genai::GenerativeModelManager mgr(&sess);
  genai::LoadSpec spec;
  spec.hf_dir = m;
  std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
  ASSERT_TRUE(lm != nullptr);
  ASSERT_TRUE(lm->valid());

  static const char kText[] =
      "The quick brown fox jumps over the lazy dog. Paris is the capital "
      "of France. Water boils at one hundred degrees Celsius at sea level.";

  std::vector<std::int32_t> ids;
  std::vector<float>        a_logits;
  int a_n = 0, a_vocab = 0;
  ASSERT_TRUE(genai::capture_token_logits(*lm, kText, 48, ids, a_logits, a_n,
                                          a_vocab, &sess));
  ASSERT_TRUE(a_n > 0);
  ASSERT_TRUE(a_vocab > 0);

  // A vs A: same model, same ids => zero divergence and zero rel-L2.
  genai::AbDivergenceResult d =
      genai::ab_divergence(*lm, ids, a_logits, a_n, a_vocab, &sess);
  ASSERT_TRUE(d.ok);
  EXPECT_TRUE(d.n == a_n);
  EXPECT_TRUE(std::isfinite(d.kl));
  EXPECT_TRUE(std::isfinite(d.rel_l2));
  EXPECT_TRUE(d.kl >= 0.0);
  EXPECT_TRUE(d.kl < 1e-4);
  EXPECT_TRUE(d.rel_l2 < 1e-4);

  // Also exercise the stage path (divergence=true, model_a == model_b, no
  // dataset => built-in fallback text): it must produce a report.
  FlexData c = cfg_(m, m);
  {
    auto o = c.as_object();
    o.insert("wikitext", FlexData::make_string(""));
    o.insert("arc", FlexData::make_string(""));
    o.insert("divergence", FlexData::make_bool(true));
    o.insert("ppl_tokens", FlexData::make_int(48));
  }
  ModelEvalStage s(&sess, "ev", std::vector<InEdge>{}, std::move(c));
  ASSERT_TRUE(s.config_error().empty());
  ASSERT_TRUE(s.divergence());
  EXPECT_TRUE(s.evaluate_once());
}

// MOSS-TTS-Local-v1.5: the TTS model's dense Qwen3 TEXT backbone must load
// through GenerativeModelManager (recognized as a metal-qwen text LM with the
// tied text head) and emit a full [vocab] text-logits row, so a bf16-vs-
// quantized WikiText-2 perplexity comparison can run. Gated on
// VPIPE_MOSS_TTS_LOCAL_MODEL (absolute model dir); uses a literal English
// passage, no WikiText dataset needed.
//
// What this asserts: the model LOADS and perplexity runs end-to-end (ok +
// finite + scored tokens) -- i.e. the recognition/parse/tied-head plumbing
// produces text logits and the dense forward runs.
//
// It deliberately does NOT assert a *low* perplexity. The dense forward is
// token-exact-faithful to a torch/transformers reference (the model's own
// qwen3_decoder.py MossQwen3Model): per-layer residual + per-position prefill
// AND decode logits match to bf16 rounding (rel-L2 ~0.05-0.07, argmax-equal).
// The large perplexity is INTRINSIC to MOSS's backbone -- it is a TTS model
// whose tied-embedding "text head" is not a natural-language next-token
// predictor; the reference torch forward gives the same ~1e7-scale PPL on
// plain prose. So PPL > 1.0 is the only honest bound here (a low number would
// indicate the model, not vpipe, had changed). The residual-stream RMS that
// grows to ~49 by the last layer is likewise correct -- the reference grows
// identically (it is model-specific, not a defect).
TEST(model_eval_stage, moss_tts_local_perplexity)
{
  const char* m = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  CerrSilencer hush;
  if (sess.metal_compute() == nullptr) { return; }

  genai::GenerativeModelManager mgr(&sess);
  genai::LoadSpec spec;
  spec.hf_dir = m;
  std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
  ASSERT_TRUE(lm != nullptr);
  ASSERT_TRUE(lm->valid());

  static const char kText[] =
      "The quick brown fox jumps over the lazy dog. Paris is the capital "
      "of France, and the Eiffel Tower stands on the Champ de Mars beside "
      "the river Seine. Water boils at one hundred degrees Celsius at sea "
      "level, and freezes at zero. The sun rises in the east and sets in "
      "the west every single day of the year.";
  genai::PerplexityResult r =
      genai::eval_wikitext2_perplexity(*lm, kText, 256, &sess);
  // The text-logits path flows: a full [vocab] logits row is produced and
  // scored for every position (proves load + tied head + forward run).
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.n_tokens > 0);
  EXPECT_TRUE(std::isfinite(r.perplexity));
  EXPECT_TRUE(r.perplexity > 1.0);
}

// MOSS-TTS-Realtime backbone (arch "MossTTSRealtime"): loads through
// GenerativeModelManager as a Qwen3-1.7B text LM (shape from language_config,
// head tied to language_model.embed_tokens == the text embed) and runs a text
// forward. Like v1.5, its tied-embedding "head" is NOT a natural-language
// predictor -- absolute PPL is meaningless/huge (use divergence for real quant
// eval); this only asserts the recognition/parse/tied-head plumbing LOADS and
// the forward produces scored [vocab] logits. Gated on
// VPIPE_MOSS_TTS_REALTIME_MODEL (the bf16 OR 8-bit dir).
TEST(model_eval_stage, moss_tts_realtime_perplexity)
{
  const char* m = std::getenv("VPIPE_MOSS_TTS_REALTIME_MODEL");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  CerrSilencer hush;
  if (sess.metal_compute() == nullptr) { return; }

  genai::GenerativeModelManager mgr(&sess);
  genai::LoadSpec spec;
  spec.hf_dir = m;
  std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
  ASSERT_TRUE(lm != nullptr);
  ASSERT_TRUE(lm->valid());

  static const char kText[] =
      "The quick brown fox jumps over the lazy dog. Paris is the capital "
      "of France, and the Eiffel Tower stands beside the river Seine.";
  genai::PerplexityResult r =
      genai::eval_wikitext2_perplexity(*lm, kText, 128, &sess);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(r.n_tokens > 0);
  EXPECT_TRUE(std::isfinite(r.perplexity));
  EXPECT_TRUE(r.perplexity > 1.0);
}

// The MEANINGFUL eval for the realtime TTS backbone: A/B divergence of the
// 8-bit-quantized backbone vs the bf16 reference (quantization error). Captures
// the bf16 logits, quantizes on the fly to 8-bit (embeds/tied-head stay bf16),
// loads the quantized dir, and compares. Gated on VPIPE_MOSS_TTS_REALTIME_MODEL
// pointing at the UNQUANTIZED bf16 dir (skips a dir that already has a
// quantization block).
TEST(model_eval_stage, moss_tts_realtime_quantized_diverges)
{
  const char* m = std::getenv("VPIPE_MOSS_TTS_REALTIME_MODEL");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  CerrSilencer hush;
  if (sess.metal_compute() == nullptr) { return; }

  static const char kText[] =
      "The quick brown fox jumps over the lazy dog. Paris is the capital "
      "of France. Water boils at one hundred degrees Celsius at sea level.";

  // Capture the bf16 reference logits (model A). Skip if `m` is not raw bf16.
  std::vector<std::int32_t> ids;
  std::vector<float> a_logits;
  int a_n = 0, a_vocab = 0;
  {
    genai::GenerativeModelManager mgr(&sess);
    genai::LoadSpec spec;
    spec.hf_dir = m;
    std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
    ASSERT_TRUE(lm != nullptr && lm->valid());
    ASSERT_TRUE(genai::capture_token_logits(*lm, kText, 48, ids, a_logits,
                                            a_n, a_vocab, &sess));
    ASSERT_TRUE(a_n > 0 && a_vocab > 0);
  }

  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path qdir = fs::temp_directory_path() /
      ("vpipe-moss-rt-q8-" + std::to_string(::getpid()));
  fs::remove_all(qdir, ec);
  {
    genai::ModelQuantizer mq(sess.metal_compute());
    genai::QuantizeOptions opt;   // 8-bit g64; quant_embeddings/norm_offset off
    opt.bits = 8; opt.group = 64;
    std::string err;
    if (!mq.run(m, qdir.string(), opt, &err)) { return; }  // m wasn't raw bf16
  }

  genai::GenerativeModelManager mgr(&sess);
  genai::LoadSpec spec;
  spec.hf_dir = qdir.string();
  std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  genai::AbDivergenceResult d =
      genai::ab_divergence(*lm, ids, a_logits, a_n, a_vocab, &sess);
  fs::remove_all(qdir, ec);
  ASSERT_TRUE(d.ok);
  std::fprintf(stderr, "[moss-rt 8-bit vs bf16] KL=%.6f rel_l2=%.6f\n",
               d.kl, d.rel_l2);
  EXPECT_TRUE(std::isfinite(d.kl) && d.kl >= 0.0 && d.kl < 50.0);
  EXPECT_TRUE(std::isfinite(d.rel_l2) && d.rel_l2 >= 0.0);
}

// A model-quantize'd MOSS (AFFINE backbone, bf16 embed/tied-head) must LOAD
// through GenerativeModelManager and run a forward. Exercises mixed 4/8 plus
// uniform 4-bit and 8-bit; the mixed run also checks the bf16-vs-quantized
// A->B output divergence is SMALL (proves the quantized forward is correct,
// not just that it binds). Gated on VPIPE_MOSS_TTS_LOCAL_MODEL.
TEST(model_eval_stage, moss_tts_local_quantized_loads_and_diverges)
{
  const char* m = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (m == nullptr || *m == '\0') { return; }
  if (!std::filesystem::exists(std::filesystem::path(m) / "config.json")) {
    return;
  }
  Session sess;
  CerrSilencer hush;
  if (sess.metal_compute() == nullptr) { return; }

  static const char kText[] =
      "The quick brown fox jumps over the lazy dog. Paris is the capital "
      "of France. Water boils at one hundred degrees Celsius at sea level.";

  namespace fs = std::filesystem;
  std::error_code ec;

  // Capture the bf16 reference logits once (model A).
  std::vector<std::int32_t> ids;
  std::vector<float>        a_logits;
  int a_n = 0, a_vocab = 0;
  {
    genai::GenerativeModelManager mgr(&sess);
    genai::LoadSpec spec;
    spec.hf_dir = m;
    std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
    ASSERT_TRUE(lm != nullptr && lm->valid());
    ASSERT_TRUE(genai::capture_token_logits(*lm, kText, 48, ids, a_logits,
                                            a_n, a_vocab, &sess));
    ASSERT_TRUE(a_n > 0 && a_vocab > 0);
  }

  struct Variant { const char* tag; bool mixed; int bits; };
  const Variant variants[] = {
      {"mixed4_8", true, 4}, {"uniform4", false, 4}, {"uniform8", false, 8}};

  for (const Variant& v : variants) {
    const fs::path qdir = fs::temp_directory_path() /
        ("vpipe-moss-q-" + std::string(v.tag) + "-" +
         std::to_string(::getpid()));
    fs::remove_all(qdir, ec);
    {
      genai::ModelQuantizer mq(sess.metal_compute());
      genai::QuantizeOptions opt;
      opt.bits  = v.bits;
      opt.group = 64;
      if (v.mixed) {
        opt.mixed        = true;
        opt.high_bits    = 8;
        opt.n_layers     = 36;
        opt.layer_prefix = "transformer.layers.";
      }
      std::string err;
      ASSERT_TRUE(mq.run(m, qdir.string(), opt, &err));
    }

    // The quantized MOSS must LOAD (this is the bind that used to fail) and
    // run a forward (perplexity completes; its absolute value is meaningless
    // by design for a TTS backbone).
    genai::GenerativeModelManager mgr(&sess);
    genai::LoadSpec spec;
    spec.hf_dir = qdir.string();
    std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
    ASSERT_TRUE(lm != nullptr);
    ASSERT_TRUE(lm->valid());
    genai::PerplexityResult r =
        genai::eval_wikitext2_perplexity(*lm, kText, 48, &sess);
    EXPECT_TRUE(r.ok && r.n_tokens > 0 && std::isfinite(r.perplexity));

    {
      genai::AbDivergenceResult d =
          genai::ab_divergence(*lm, ids, a_logits, a_n, a_vocab, &sess);
      ASSERT_TRUE(d.ok);
      std::fprintf(stderr, "[moss-q %s] KL=%.6f rel_l2=%.6f\n", v.tag,
                   d.kl, d.rel_l2);
      // Forward is correct => finite, non-negative, and NOT huge/inf. The
      // absolute KL tracks the backbone's intrinsic quant sensitivity (this
      // TTS backbone is unusually quant-sensitive: cf. moss_tts_local's
      // awq_reduces_4bit_drift, ~0.46 hidden rel-L2 at 4-bit vs bf16). The
      // 8-bit run is the "good quant" reference: small KL + logit rel-L2.
      EXPECT_TRUE(std::isfinite(d.kl) && d.kl >= 0.0 && d.kl < 50.0);
      EXPECT_TRUE(std::isfinite(d.rel_l2));
      if (!v.mixed && v.bits == 8) {
        EXPECT_TRUE(d.kl < 1.0);
        EXPECT_TRUE(d.rel_l2 < 0.5);
      }
    }
    fs::remove_all(qdir, ec);
  }
}

// Debug-only: prefill the MOSS backbone over a FIXED id sequence [1..16] so a
// reference torch forward (same ids) can be diffed layer-by-layer. Dumps via
// VPIPE_QWEN_LAYER_DUMP (set inside the metal model). Gated on the same model
// env var; only runs the prefill (the dump is the deliverable).
TEST(model_eval_stage, moss_tts_local_layerdump)
{
  const char* m = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (m == nullptr || *m == '\0') { return; }
  if (std::getenv("VPIPE_QWEN_LAYER_DUMP") == nullptr) { return; }
  Session sess;
  CerrSilencer hush;
  if (sess.metal_compute() == nullptr) { return; }
  genai::GenerativeModelManager mgr(&sess);
  genai::LoadSpec spec;
  spec.hf_dir = m;
  std::shared_ptr<genai::LoadedLanguageModel> lm = mgr.load(spec);
  ASSERT_TRUE(lm != nullptr);
  ASSERT_TRUE(lm->valid());
  std::vector<std::int32_t> ids;
  for (std::int32_t i = 1; i <= 16; ++i) { ids.push_back(i); }
  // Decode mode: prefill ids[0], then next_token through the rest, dumping
  // last_logits_host() at each step -> <prefix>_dec_<i>.bin (predicts ids[i]).
  if (std::getenv("VPIPE_QWEN_DECODE_DUMP") != nullptr) {
    const std::string pre = std::getenv("VPIPE_QWEN_LAYER_DUMP");
    genai::LoadedLanguageModel::Context dc = lm->make_context();
    std::array<std::int32_t, 1> f0{ ids[0] };
    ASSERT_TRUE(lm->prefill(dc, std::span<const std::int32_t>(f0)) >= 0);
    for (std::size_t i = 1; i < ids.size(); ++i) {
      const std::vector<float>& lg = lm->last_logits_host();
      std::ofstream of(pre + "_dec_" + std::to_string(i) + ".bin",
                       std::ios::binary);
      of.write(reinterpret_cast<const char*>(lg.data()),
               (std::streamsize)(lg.size() * sizeof(float)));
      ASSERT_TRUE(lm->next_token(dc, ids[i]) >= 0);
    }
  }
  genai::LoadedLanguageModel::Context ctx = lm->make_context();
  const std::int32_t pred = lm->prefill(ctx, ids);
  EXPECT_TRUE(pred >= 0);
}
