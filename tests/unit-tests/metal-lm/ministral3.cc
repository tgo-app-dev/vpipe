// ministral3.cc -- Mistral3 / Ministral-3 (Tekken tokenizer, a dense
// decoder with YaRN RoPE, the Pixtral vision tower + its projector)
// against the mlx-vlm reference on the SAME MLX checkpoint.
//
// Env (absolute paths; unset -> the test skips):
//   VPIPE_MINISTRAL3_TEST_MODEL_PATH  mlx-community/
//                                     Ministral-3-14B-Instruct-2512-4bit
//   VPIPE_MINISTRAL3_GOLDEN           a dir written by
//                                     tools/dump_ministral3_golden.py

#include "metal-lm-test-common.h"

#include "common/flex-bag.h"
#include "common/media-decode.h"
#include "generative-models/mistral3/metal-pixtral-vision.h"

namespace {

const char*
model_dir_()
{
  const char* p = std::getenv("VPIPE_MINISTRAL3_TEST_MODEL_PATH");
  return (p && *p) ? p : nullptr;
}

const char*
golden_dir_()
{
  const char* p = std::getenv("VPIPE_MINISTRAL3_GOLDEN");
  return (p && *p) ? p : nullptr;
}

FlexData
read_json_(const std::string& path)
{
  std::ifstream in(path);
  if (!in) { return FlexData(); }
  return FlexData::from_json(in);
}

std::vector<std::int32_t>
ids_of_(const FlexData& arr)
{
  std::vector<std::int32_t> out;
  if (!arr.is_array()) { return out; }
  for (const auto& v : arr.as_array()) {
    out.push_back((std::int32_t)v.as_int(-1));
  }
  return out;
}

}  // namespace

// The Tekken pre-tokenizer (case-split letter runs, single digits, no
// contractions) + `ignore_merges`, against the HF `tokenizers` ids the
// reference produced for the same strings. A case that holds a special
// token's spelling is skipped: HF matches added tokens inside text,
// vpipe's encode() deliberately does not (templates push special ids).
TEST(ministral3, tokenizer_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(md) + "/tokenizer.json", nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  const FlexData g = read_json_(std::string(gd) + "/tokenizer.json");
  ASSERT_TRUE(g.is_object());
  if (!g.is_object()) { return; }
  const FlexData cases = g.as_object().at("cases");
  int n_checked = 0, n_bad = 0;
  for (const auto& c : cases.as_array()) {
    const std::string text(c.as_object().at("text").as_string(""));
    if (text.find("[INST]") != std::string::npos) { continue; }
    const auto want = ids_of_(c.as_object().at("ids"));
    const auto got = tok->encode(text);
    ++n_checked;
    if (got != want) {
      ++n_bad;
      std::printf("[ministral3.tokenizer] MISMATCH '%s'\n  want:",
                  text.c_str());
      for (auto i : want) { std::printf(" %d", i); }
      std::printf("\n  got: ");
      for (auto i : got) { std::printf(" %d", i); }
      std::printf("\n");
    }
    // Decode must reproduce the text byte-for-byte.
    const std::string back = tok->decode(
        std::span<const std::int32_t>(got.data(), got.size()));
    EXPECT_TRUE(back == text);
  }
  std::printf("[ministral3.tokenizer] %d cases, %d mismatched\n",
              n_checked, n_bad);
  EXPECT_TRUE(n_checked >= 10);
  EXPECT_TRUE(n_bad == 0);
}

namespace {

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> v;
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const auto n = (std::size_t)in.tellg() / sizeof(float);
  in.seekg(0);
  v.resize(n);
  in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * 4));
  return v;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double d = 0.0, r = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = (double)a[i] - (double)b[i];
    d += x * x;
    r += (double)b[i] * (double)b[i];
  }
  return r > 0.0 ? std::sqrt(d / r) : 0.0;
}

std::shared_ptr<genai::LoadedLanguageModel>
load_lm_(Session& sess, const char* dir, const char* dtype)
{
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return nullptr; }
  genai::LoadSpec spec;
  spec.hf_dir        = dir;
  spec.compute_dtype = dtype;
  spec.page_tokens   = 256;
  spec.max_pages     = 16;
  return mgr->load(spec);
}

// Greedy-decode `n` tokens after prefilling `ids`; stops at </s> (2).
std::vector<std::int32_t>
greedy_(genai::LoadedLanguageModel& lm,
        const std::vector<std::int32_t>& ids, int n,
        std::vector<float>* logits0)
{
  std::vector<std::int32_t> out;
  auto ctx = lm.make_context();
  if (!ctx.valid()) { return out; }
  std::int32_t t = lm.prefill(ctx, ids);
  if (logits0) { *logits0 = lm.last_logits_host(); }
  while (t >= 0 && (int)out.size() < n) {
    out.push_back(t);
    if (t == 2) { break; }
    t = lm.next_token(ctx);
  }
  return out;
}

// First index where `got` leaves `want`, or -1 when one is a prefix of
// the other up to the shorter length.
int
first_divergence_(const std::vector<std::int32_t>& got,
                  const std::vector<std::int32_t>& want)
{
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) { return (int)i; }
  }
  return got.size() == want.size() ? -1 : (int)n;
}

}  // namespace

// The Mistral template, fed the golden's system message, renders the
// reference's prompt ids exactly (<s>, [SYSTEM_PROMPT] ...
// [/SYSTEM_PROMPT], [INST] ... [/INST]); and from the checkpoint's
// config it carries the default system message the Jinja declares.
TEST(ministral3, chat_template_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(md) + "/tokenizer.json", nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  const FlexData g = read_json_(std::string(gd) + "/text.json");
  ASSERT_TRUE(g.is_object());
  if (!g.is_object()) { return; }
  const auto obj = g.as_object();
  genai::ModelConfig cfg;
  cfg.architecture = "Mistral3ForConditionalGeneration";
  bag::set_text(cfg.extra, genai::model_config::kDefaultSystemPrompt,
                std::string(obj.at("system").as_string("")));
  auto tpl = genai::make_chat_template(cfg, *tok);
  ASSERT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(std::string(obj.at("user").as_string("")),
                        /*is_first_turn=*/true, &ids);
  EXPECT_TRUE(ids == ids_of_(obj.at("prompt_ids")));
  EXPECT_TRUE(tpl->image_pad_token_id() == 10);   // [IMG]
  EXPECT_TRUE(tpl->is_stop_token(2) && !tpl->is_stop_token(4));

  // The loader lifts the checkpoint's own default system message.
  genai::ModelLoader loader(nullptr);
  auto mc = loader.load_config(md);
  ASSERT_TRUE(mc.has_value());
  if (!mc) { return; }
  const std::string sys =
      bag::text(mc->extra, genai::model_config::kDefaultSystemPrompt);
  std::printf("[ministral3.template] default system: %zu chars\n",
              sys.size());
  EXPECT_TRUE(sys.find("You are Ministral-3-14B-Instruct-2512") == 0);
  EXPECT_TRUE(sys.find("{today}") != std::string::npos);
  EXPECT_TRUE(sys.find("\\n") == std::string::npos);   // unescaped
}

// Teacher forcing against a greedy reference: after `prefill` (which
// returns the first prediction) every step is fed the REFERENCE's token,
// so one flipped near-tie does not derail the rest. A pick the reference
// made by at least kDecisive (two bf16 ulps of a logit in the 16..32
// range) must agree; nearer ones are rounding and are only counted.
namespace {

constexpr double kDecisive = 0.25;

struct ForcedRun {
  int steps = 0, decisive = 0, decisive_agree = 0, agree = 0;
  int first_bad = -1;            // first DECISIVE disagreement
};

ForcedRun
teacher_force_(genai::LoadedLanguageModel& lm,
               genai::LoadedLanguageModel::Context& ctx,
               std::int32_t first,
               const std::vector<std::int32_t>& want,
               const std::vector<double>& margins)
{
  ForcedRun f;
  std::int32_t pred = first;
  for (std::size_t i = 0; i < want.size() && pred >= 0; ++i) {
    ++f.steps;
    const bool decisive = i < margins.size() && margins[i] >= kDecisive;
    const bool same = pred == want[i];
    f.agree += same ? 1 : 0;
    if (decisive) {
      ++f.decisive;
      f.decisive_agree += same ? 1 : 0;
      if (!same && f.first_bad < 0) { f.first_bad = (int)i; }
    }
    if (want[i] == 2 || i + 1 >= want.size()) { break; }
    pred = lm.next_token(ctx, want[i]);
  }
  return f;
}

}  // namespace

// Text: the reference's prompt ids on the metal backend (the Qwen dense
// path: no q/k norm, YaRN RoPE) against mlx-vlm's greedy run on the same
// 4-bit weights -- first-step logits, then every decisive pick under
// teacher forcing. The free-running greedy is reported, not asserted: it
// may part ways at a bf16 near-tie (the golden records each step's
// top-2 margin), after which nothing is comparable.
namespace {

struct GreedyRun {
  bool   ran = false;       // false: skipped (env unset) or failed load
  bool   loaded = false;
  ForcedRun forced;
  std::size_t n_tok = 0;
  double logits_rel = -1.0;
  int    divergence = -1;
  double margin = -1.0;     // the reference's top-2 margin there
};

GreedyRun
run_text_greedy_(const char* dtype)
{
  GreedyRun r;
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return r; }
  r.ran = true;
  const FlexData g = read_json_(std::string(gd) + "/text.json");
  if (!g.is_object()) { return r; }
  const auto obj = g.as_object();
  const auto prompt = ids_of_(obj.at("prompt_ids"));
  const auto want = ids_of_(obj.at("greedy"));
  std::vector<double> margins;
  const FlexData marr = obj.at("margins");   // own it: views dangle
  for (const auto& v : marr.as_array()) {
    margins.push_back(v.as_real(0.0));
  }
  Session sess;
  auto lm = load_lm_(sess, md, dtype);
  if (!lm || !lm->valid()) { return r; }
  r.loaded = true;
  std::vector<float> lg0;
  const auto t0 = std::chrono::steady_clock::now();
  const auto got = greedy_(*lm, prompt, (int)want.size(), &lg0);
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  const auto ref0 = read_f32_(std::string(gd) + "/text_logits0.f32");
  if (!ref0.empty() && lg0.size() == ref0.size()) {
    r.logits_rel = rel_l2_(lg0.data(), ref0.data(), ref0.size());
  }
  {
    auto fctx = lm->make_context();
    const std::int32_t first = lm->prefill(fctx, prompt);
    r.forced = teacher_force_(*lm, fctx, first, want, margins);
  }
  r.n_tok = got.size();
  r.divergence = first_divergence_(got, want);
  if (r.divergence >= 0 && r.divergence < (int)margins.size()) {
    r.margin = margins[(std::size_t)r.divergence];
  }
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(got.data(), got.size()));
  std::printf("[ministral3.text %s] %zu tok in %.2f s | logits0 rel %.4f |"
              " free-run divergence %d (ref margin there %.3f) | forced: "
              "%d/%d decisive agree, %d/%d all\n  '%s'\n", dtype,
              got.size(), secs, r.logits_rel, r.divergence, r.margin,
              r.forced.decisive_agree, r.forced.decisive, r.forced.agree,
              r.forced.steps, text.c_str());
  return r;
}

}  // namespace

TEST(ministral3, text_greedy_matches_reference_f16)
{
  const GreedyRun r = run_text_greedy_("f16");
  if (!r.ran) { return; }
  ASSERT_TRUE(r.loaded && r.n_tok > 0);
  EXPECT_TRUE(r.logits_rel >= 0.0 && r.logits_rel < 0.05);
  EXPECT_TRUE(r.forced.decisive >= 32);
  EXPECT_TRUE(r.forced.first_bad < 0);
}

TEST(ministral3, text_greedy_matches_reference_bf16)
{
  const GreedyRun r = run_text_greedy_("bf16");
  if (!r.ran) { return; }
  ASSERT_TRUE(r.loaded && r.n_tok > 0);
  EXPECT_TRUE(r.logits_rel >= 0.0 && r.logits_rel < 0.05);
  EXPECT_TRUE(r.forced.decisive >= 32);
  EXPECT_TRUE(r.forced.first_bad < 0);
}

// The Pixtral tower + Mistral3 projector on the reference's own
// normalised pixels (560 x 812: a 20 x 29 grid of merged tokens, so the
// row / column RoPE axes are told apart), one stage at a time. The bar
// is the reference's own bf16 run's distance from its fp32 run: the
// port computes in f16, finer than bf16, so it must land no farther from
// fp32 than that.
TEST(ministral3, vision_tower_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  genai::ModelLoader loader(nullptr);
  auto cfg = loader.load_config(md);
  ASSERT_TRUE(cfg.has_value() && cfg->vision.present);
  if (!cfg) { return; }
  const auto vc = genai::MetalPixtralVisionEncoder::config_from(*cfg);
  std::printf("[ministral3.vision] depth %d hidden %d heads %d hd %d "
              "inter %d patch %d merge %d out %d edge %d theta %g\n",
              vc.depth, vc.hidden, vc.n_heads, vc.head_dim,
              vc.intermediate, vc.patch_size, vc.merge, vc.out_hidden,
              vc.longest_edge, (double)vc.rope_theta);
  auto enc = genai::MetalPixtralVisionEncoder::load(
      genai::WeightSet::open(md, nullptr), mc, vc);
  ASSERT_TRUE(enc != nullptr);
  if (!enc) { return; }
  // Layout rows are irrelevant to the stages checked here.
  enc->set_layout_rows(std::vector<float>((std::size_t)vc.out_hidden, 0.f),
                       std::vector<float>((std::size_t)vc.out_hidden, 0.f));

  const FlexData g = read_json_(std::string(gd) + "/image.json");
  ASSERT_TRUE(g.is_object());
  if (!g.is_object()) { return; }
  const FlexData sz = g.as_object().at("image_size");
  const int h = (int)sz.as_array()[0].as_int(0);
  const int w = (int)sz.as_array()[1].as_int(0);
  const auto px = read_f32_(std::string(gd) + "/pixel_values.f32");
  ASSERT_TRUE(px.size() == (std::size_t)3 * h * w);
  if (px.size() != (std::size_t)3 * h * w) { return; }

  std::vector<float> vit, proj;
  const auto t0 = std::chrono::steady_clock::now();
  const auto r = enc->encode_normalized(px.data(), h, w, &vit, &proj);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  const int unit = vc.patch_size * vc.merge;
  ASSERT_TRUE(r.grid_h == h / unit && r.grid_w == w / unit);
  ASSERT_TRUE(r.n_tokens == r.grid_h * (r.grid_w + 1));

  const auto vit32 = read_f32_(std::string(gd) + "/vit_fp32.f32");
  const auto vit16 = read_f32_(std::string(gd) + "/vit.f32");
  const auto prj32 = read_f32_(std::string(gd) + "/proj_fp32.f32");
  const auto prj16 = read_f32_(std::string(gd) + "/proj.f32");
  ASSERT_TRUE(vit.size() == vit32.size() && proj.size() == prj32.size());
  if (vit.size() != vit32.size() || proj.size() != prj32.size()) { return; }
  const double v_ours = rel_l2_(vit.data(), vit32.data(), vit.size());
  const double v_ref = rel_l2_(vit16.data(), vit32.data(), vit.size());
  const double p_ours = rel_l2_(proj.data(), prj32.data(), proj.size());
  const double p_ref = rel_l2_(prj16.data(), prj32.data(), proj.size());
  std::printf("[ministral3.vision] %dx%d -> %dx%d tokens in %.1f ms | vs "
              "fp32: vit %.4f (ref bf16 %.4f) proj %.4f (ref bf16 %.4f)\n",
              h, w, r.grid_h, r.grid_w, ms, v_ours, v_ref, p_ours, p_ref);
  EXPECT_TRUE(v_ours < std::max(0.02, v_ref));
  EXPECT_TRUE(p_ours < std::max(0.02, p_ref));
}

// End to end on an image: the PNG decoded and encoded by the production
// path (no resize at this size: it is already a multiple of 28), its rows
// spliced at the reference's 600 image positions ([IMG] rows with their
// [IMG_BREAK]s and the [IMG_END]), greedy-decoded against mlx-vlm.
TEST(ministral3, image_greedy_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  const FlexData g = read_json_(std::string(gd) + "/image.json");
  ASSERT_TRUE(g.is_object());
  if (!g.is_object()) { return; }
  const auto obj = g.as_object();
  const auto prompt = ids_of_(obj.at("prompt_ids"));
  const auto want = ids_of_(obj.at("greedy"));
  std::vector<double> margins;
  const FlexData marr = obj.at("margins");
  for (const auto& v : marr.as_array()) { margins.push_back(v.as_real(0)); }

  Session sess;
  auto lm = load_lm_(sess, md, "f16");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  auto* enc = lm->metal_pixtral_vision_encoder();
  ASSERT_TRUE(enc != nullptr);
  if (!enc) { return; }
  std::string derr;
  auto img = decode_image_file(sess.ffmpeg_libraries(),
                               std::string(gd) + "/image.png", &derr);
  ASSERT_TRUE(img.has_value());
  if (!img) { return; }
  const auto t0 = std::chrono::steady_clock::now();
  const auto r = enc->encode(img->rgb.data(), img->height, img->width);
  const double enc_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(!r.embeddings.empty());
  if (r.embeddings.empty()) { return; }

  // The template renders the reference's stream with every image row a
  // placeholder ([IMG_BREAK] / [IMG_END] carried by the tower's rows).
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  std::vector<std::int32_t> mine;
  {
    genai::ModelConfig cfg;
    cfg.architecture = "Mistral3ForConditionalGeneration";
    bag::set_text(cfg.extra, genai::model_config::kDefaultSystemPrompt,
                  std::string(obj.at("system").as_string("")));
    auto t2 = genai::make_chat_template(cfg, lm->tokenizer());
    genai::MediaChunk im, tx;
    im.kind = genai::MediaChunk::Kind::Image;
    im.n_tokens = r.n_tokens;
    tx.kind = genai::MediaChunk::Kind::Text;
    tx.text = std::string(obj.at("user").as_string(""));
    const genai::MediaChunk chunks[2] = {im, tx};
    ASSERT_TRUE(t2 && t2->render_user_turn_media(
        std::span<const genai::MediaChunk>(chunks, 2), true, &mine));
  }
  std::vector<std::int32_t> want_ids = prompt;
  for (auto& id : want_ids) { if (id == 12 || id == 13) { id = 10; } }
  EXPECT_TRUE(mine == want_ids);

  std::vector<genai::TokenRef> refs;
  int off = 0;
  for (std::int32_t id : prompt) {
    genai::TokenRef t;
    if (id == 10 || id == 12 || id == 13) {
      t.kind = genai::TokenRef::Kind::ImageTokens;
      t.embeddings_buf = &r.embeddings;
      t.image_token_offset = off++;
    } else {
      t.kind = genai::TokenRef::Kind::Text;
      t.text_id = id;
    }
    refs.push_back(t);
  }
  ASSERT_TRUE(off == r.n_tokens);
  auto ctx = lm->make_context();
  const std::pair<int, int> grid{r.grid_h, r.grid_w};
  const auto t1 = std::chrono::steady_clock::now();
  std::int32_t t = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>(&grid, 1));
  const double pre_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t1).count();
  const std::vector<float> lg0 = lm->last_logits_host();
  std::vector<std::int32_t> got;
  while (t >= 0 && got.size() < want.size()) {
    got.push_back(t);
    if (t == 2) { break; }
    t = lm->next_token(ctx);
  }
  const auto ref0 = read_f32_(std::string(gd) + "/image_logits0.f32");
  double rl = -1.0;
  if (lg0.size() == ref0.size() && !ref0.empty()) {
    rl = rel_l2_(lg0.data(), ref0.data(), ref0.size());
  }
  const int div = first_divergence_(got, want);
  const double m = (div >= 0 && div < (int)margins.size())
      ? margins[(std::size_t)div] : -1.0;
  ForcedRun f;
  {
    auto fctx = lm->make_context();
    const std::int32_t first = lm->prefill_multimodal_metal(
        fctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>(&grid, 1));
    f = teacher_force_(*lm, fctx, first, want, margins);
  }
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(got.data(), got.size()));
  std::printf("[ministral3.image] encode %.0f ms, prefill %zu tok %.0f ms |"
              " logits0 rel %.4f | free-run divergence %d (ref margin %.3f)"
              " | forced: %d/%d decisive agree, %d/%d all\n  '%s'\n",
              enc_ms, prompt.size(), pre_ms, rl, div, m, f.decisive_agree,
              f.decisive, f.agree, f.steps, text.c_str());
  EXPECT_TRUE(rl >= 0.0 && rl < 0.05);
  EXPECT_TRUE(f.decisive >= 24);
  EXPECT_TRUE(f.first_bad < 0);
}

// Batched decode (what realtime-vqa runs for its question branches) on
// the no-q/k-norm dense path, against serial decode of the same
// branches: branches at different lengths, split-K tiles (the default)
// and the row-identical ones. A small share of near-tie flips is the
// expected reduction-order noise (metal_lm_smoke.qwen_batched_decode_*);
// a missing guard or a wrong RoPE position shows as a cascade instead.
TEST(ministral3, batched_decode_matches_serial)
{
  const char* md = model_dir_();
  if (!md) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(md);
  ASSERT_TRUE(cfg.has_value());
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  ASSERT_TRUE(!mcfg.qk_norm && mcfg.rope_yarn_factor > 1.0f);
  ASSERT_TRUE(mcfg.max_seq == 262144);
  ASSERT_TRUE(mcfg.attn_temp_beta > 0.0f && mcfg.attn_temp_orig == 16384);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 256;
  mcfg.max_pages = 64;
  auto model = genai::MetalQwenModel::load(md, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(md) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  auto prompt = tok->encode("The three most useful things to pack for a "
                            "week of hiking are");
  auto root = ctxm->acquire_root();
  ASSERT_TRUE(root.valid() && !model->prefill(root, prompt).empty());

  const int N = 4, n_steps = 12;
  auto argmax_of = [](const std::vector<float>& lg) {
    return (std::int32_t)(std::max_element(lg.begin(), lg.end())
                          - lg.begin());
  };
  for (const bool ks : {false, true}) {
    model->set_qmv_ksplit(true, ks);
    auto batched = ctxm->branch(root, N);
    auto serial  = ctxm->branch(root, N);
    ASSERT_TRUE((int)batched.size() == N && (int)serial.size() == N);
    std::vector<std::int32_t> first((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      // Distinct content and length per branch.
      const auto suffix = tok->encode(std::string(" (") +
                                      std::string((std::size_t)i + 1, 'x'));
      auto lb = model->prefill(batched[(std::size_t)i], suffix);
      auto ls = model->prefill(serial[(std::size_t)i], suffix);
      ASSERT_TRUE(!lb.empty() && !ls.empty());
      first[(std::size_t)i] = argmax_of(lb);
    }
    std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      std::int32_t cur = first[(std::size_t)i];
      for (int s = 0; s < n_steps; ++s) {
        const std::int32_t nxt =
            model->decode_step_fast(serial[(std::size_t)i], cur);
        if (nxt < 0) { break; }
        ref[(std::size_t)i].push_back(nxt);
        cur = nxt;
      }
    }
    auto got = model->decode_batched_argmax(
        std::span<const genai::ContextId>(batched.data(), batched.size()),
        std::span<const std::int32_t>(first.data(), first.size()), n_steps);
    ASSERT_TRUE((int)got.size() == N);
    int total = 0, flips = 0;
    for (int i = 0; i < N && (int)got.size() == N; ++i) {
      const auto& g = got[(std::size_t)i];
      const auto& r = ref[(std::size_t)i];
      for (std::size_t s = 0; s < r.size() && s < g.size(); ++s) {
        ++total;
        flips += (g[s] != r[s]) ? 1 : 0;
      }
    }
    std::printf("[ministral3.batched] %sN=%d: %d/%d steps match serial\n",
                ks ? "split-K " : "", N, total - flips, total);
    EXPECT_TRUE(total == N * n_steps);
    EXPECT_TRUE(flips * 20 <= total);
    for (auto id : batched) { ctxm->release(id); }
    for (auto id : serial) { ctxm->release(id); }
  }
}

// The processor's half on an image that needs a resize (1024 -> 1036 a
// side, the next multiple of 28): PIL's 8-bit BICUBIC, then rescale and
// CLIP normalisation, against mlx-vlm's processor on the same lossless
// PNG. PIL rounds its fixed-point taps where this rounds float ones, so
// a pixel may land one 8-bit level off; anything more is a wrong filter.
// Also the geometry a token budget gives (realtime-vqa's frames).
TEST(ministral3, preprocess_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  genai::ModelLoader loader(nullptr);
  auto cfg = loader.load_config(md);
  ASSERT_TRUE(cfg.has_value());
  if (!cfg) { return; }
  auto enc = genai::MetalPixtralVisionEncoder::load(
      genai::WeightSet::open(md, nullptr), mc,
      genai::MetalPixtralVisionEncoder::config_from(*cfg));
  ASSERT_TRUE(enc != nullptr);
  if (!enc) { return; }
  const FlexData g = read_json_(std::string(gd) + "/resize.json");
  ASSERT_TRUE(g.is_object());
  if (!g.is_object()) { return; }
  const FlexData dst = g.as_object().at("dst");
  const int want_h = (int)dst.as_array()[0].as_int(0);
  const int want_w = (int)dst.as_array()[1].as_int(0);
  std::string derr;
  auto img = decode_image_file(sess.ffmpeg_libraries(),
                               std::string(gd) + "/resize_src.png", &derr);
  ASSERT_TRUE(img.has_value());
  if (!img) { return; }
  std::vector<float> px;
  int th = 0, tw = 0;
  enc->preprocess(img->rgb.data(), img->height, img->width, -1, &px, &th,
                  &tw);
  ASSERT_TRUE(th == want_h && tw == want_w);
  const auto ref = read_f32_(std::string(gd) + "/resize_pixel_values.f32");
  ASSERT_TRUE(ref.size() == px.size());
  if (ref.size() != px.size()) { return; }
  // One 8-bit level, normalised: 1/255 over the smallest std.
  const float level = (1.0f / 255.0f) / 0.26130258f;
  std::size_t exact = 0, one = 0, worse = 0;
  for (std::size_t i = 0; i < px.size(); ++i) {
    const float d = std::fabs(px[i] - ref[i]);
    if (d < 1e-5f) { ++exact; }
    else if (d < 1.01f * level) { ++one; }
    else { ++worse; }
  }
  std::printf("[ministral3.preprocess] %dx%d -> %dx%d: %zu exact, %zu one "
              "level off, %zu worse (of %zu)\n", img->height, img->width,
              th, tw, exact, one, worse, px.size());
  EXPECT_TRUE(worse == 0);
  EXPECT_TRUE(exact * 100 >= px.size() * 99);

  // A token budget scales the image down until the grid fits.
  int bh = 0, bw = 0;
  enc->target_size(720, 1280, &bh, &bw, 256);
  std::printf("[ministral3.preprocess] 720x1280 at 256 tokens -> %dx%d "
              "(%d tokens)\n", bh, bw, (bh / 28) * (bw / 28));
  EXPECT_TRUE(bh % 28 == 0 && bw % 28 == 0);
  EXPECT_TRUE((bh / 28) * (bw / 28) <= 256 && (bh / 28) * (bw / 28) >= 200);
  enc->target_size(720, 1280, &bh, &bw);
  EXPECT_TRUE(bh == 728 && bw == 1288);   // the processor's own budget
}

// Past 16k: a needle-in-a-haystack prompt of ~34k tokens, so the queries
// cross BOTH steps of the Llama-4 attention temperature (1.069 from 16384,
// 1.110 from 32768), prefilled in one pass against mlx-vlm on the same
// weights -- the post-prefill logits, every decisive pick of the answer
// under teacher forcing, and the answer itself (the code planted ~4% in).
// mlx-vlm rounds the temperature to bf16 (1.0703 for 1.0693) where vpipe
// keeps it exact, as an fp32 run does; the logits bar allows for that.
// Needs long.json from `dump_ministral3_golden.py --long`; ~2 min.
TEST(ministral3, long_context_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  const FlexData g = read_json_(std::string(gd) + "/long.json");
  if (!g.is_object()) { return; }   // no long golden: skipped
  const auto obj = g.as_object();
  const auto prompt = ids_of_(obj.at("prompt_ids"));
  const auto want = ids_of_(obj.at("greedy"));
  std::vector<double> margins;
  const FlexData marr = obj.at("margins");
  for (const auto& v : marr.as_array()) { margins.push_back(v.as_real(0)); }
  ASSERT_TRUE(prompt.size() > 32768);

  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = md;
  spec.compute_dtype = "f16";
  spec.page_tokens   = 1024;
  spec.max_pages     = 48;      // 49152 tokens: prompt + answer + forcing
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }

  // Free-running first; its context is released before the teacher-forced
  // pass prefills the prompt again (the pool holds one of them).
  std::vector<float> lg0;
  std::vector<std::int32_t> got;
  double pre_s = 0.0, dec_s = 0.0;
  {
    auto ctx = lm->make_context();
    const auto t0 = std::chrono::steady_clock::now();
    std::int32_t t = lm->prefill(ctx, prompt);
    pre_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(t >= 0);
    lg0 = lm->last_logits_host();
    const auto t1 = std::chrono::steady_clock::now();
    while (t >= 0 && got.size() < want.size()) {
      got.push_back(t);
      if (t == 2) { break; }
      t = lm->next_token(ctx);
    }
    dec_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t1).count();
  }
  const auto ref0 = read_f32_(std::string(gd) + "/long_logits0.f32");
  double rl = -1.0;
  if (lg0.size() == ref0.size() && !ref0.empty()) {
    rl = rel_l2_(lg0.data(), ref0.data(), ref0.size());
  }
  ForcedRun f;
  {
    auto fctx = lm->make_context();
    const std::int32_t first = lm->prefill(fctx, prompt);
    f = teacher_force_(*lm, fctx, first, want, margins);
  }
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(got.data(), got.size()));
  std::printf("[ministral3.long] %zu-token prompt: prefill %.1f s (%.0f "
              "tok/s), decode %zu tok at %.1f tok/s | logits0 rel %.4f | "
              "forced %d/%d decisive, %d/%d all\n  '%s'\n",
              prompt.size(), pre_s, prompt.size() / pre_s, got.size(),
              got.size() / std::max(dec_s, 1e-6), rl, f.decisive_agree,
              f.decisive, f.agree, f.steps, text.c_str());
  EXPECT_TRUE(rl >= 0.0 && rl < 0.05);
  EXPECT_TRUE(f.decisive >= 5 && f.first_bad < 0);
  EXPECT_TRUE(text.find("4719-KESTREL") != std::string::npos);
}
