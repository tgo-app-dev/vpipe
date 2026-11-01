// lfm3.cc -- LFM3-VL (Liquid AI's LFM3 language model + its vision tower
// and projector) against a torch restatement of the checkpoint's own
// modeling code, in float32 (the reference) with its bfloat16 run as the
// bar.
//
// Env (absolute paths; unset -> the test skips):
//   VPIPE_LFM3_TEST_MODEL_PATH  LiquidAI/LFM3-VL-2B
//   VPIPE_LFM3_GOLDEN           a dir written by tools/dump_lfm3_golden.py

#include "metal-lm-test-common.h"

#include "common/flex-bag.h"
#include "generative-models/lfm3/metal-lfm3-model.h"
#include "generative-models/lfm3/metal-lfm3-model-exec.h"
#include "generative-models/lfm3/metal-lfm3-vision.h"
#include "generative-models/lm-run-memory.h"
#include "stages/model-detect.h"

#include <atomic>
#include <thread>

namespace {

const char*
model_dir_()
{
  const char* p = std::getenv("VPIPE_LFM3_TEST_MODEL_PATH");
  return (p && *p) ? p : nullptr;
}

const char*
golden_dir_()
{
  const char* p = std::getenv("VPIPE_LFM3_GOLDEN");
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

std::vector<double>
reals_of_(const FlexData& arr)
{
  std::vector<double> out;
  if (!arr.is_array()) { return out; }
  for (const auto& v : arr.as_array()) { out.push_back(v.as_real(0.0)); }
  return out;
}

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

int
argmax_(const float* v, int n)
{
  int b = 0;
  for (int i = 1; i < n; ++i) {
    if (v[i] > v[b]) { b = i; }
  }
  return b;
}

// A pick the reference made by at least this much must agree; nearer
// ones are rounding.
constexpr double kDecisive = 0.25;

struct Golden {
  std::vector<std::int32_t> prompt, cont, picks;
  std::vector<double> margins, bars;
  std::vector<float> logits;   // [rows, vocab]: prompt's last, then cont
  bool ok = false;
};

Golden
golden_(const char* gd, const std::string& name)
{
  Golden g;
  const FlexData j = read_json_(std::string(gd) + "/" + name + ".json");
  if (!j.is_object()) { return g; }
  const auto o = j.as_object();
  g.prompt = ids_of_(o.at("prompt_ids"));
  g.cont = ids_of_(o.at("continuation"));
  g.picks = ids_of_(o.at("picks"));
  g.margins = reals_of_(o.at("margins"));
  g.bars = reals_of_(o.at("bf16_rel"));
  g.logits = read_f32_(std::string(gd) + "/" + name + "_logits.f32");
  g.ok = !g.prompt.empty() && g.picks.size() == g.cont.size() + 1;
  return g;
}

std::unique_ptr<genai::MetalLfm3Model>
load_model_(metal_compute::MetalCompute* mc, const char* md)
{
  genai::MetalLfm3Model::Config cfg;
  std::string err;
  if (!genai::MetalLfm3Model::Config::from_dir(md, &cfg, &err)) {
    std::printf("[lfm3] config: %s\n", err.c_str());
    return nullptr;
  }
  cfg.page_tokens = 256;
  cfg.max_pages = 64;
  auto m = genai::MetalLfm3Model::load(genai::WeightSet::open(md, nullptr),
                                       mc, cfg, &err);
  if (!m) { std::printf("[lfm3] load: %s\n", err.c_str()); }
  return m;
}

// Every row's logits against the golden's: rel-L2 within `bar_x` of the
// bf16 run's own distance (and an absolute floor), decisive picks equal.
struct RowCheck {
  int rows = 0, decisive = 0, agree = 0;
  double worst = 0.0, worst_bar = 0.0, median = 0.0, median_bar = 0.0;
};

RowCheck
check_rows_(const Golden& g, const std::vector<float>& got, int vocab)
{
  RowCheck rc;
  rc.rows = (int)g.picks.size();
  std::vector<double> rel, bar;
  for (int r = 0; r < rc.rows; ++r) {
    const float* a = got.data() + (std::size_t)r * vocab;
    const float* b = g.logits.data() + (std::size_t)r * vocab;
    const double d = rel_l2_(a, b, (std::size_t)vocab);
    rel.push_back(d);
    bar.push_back(g.bars[(std::size_t)r]);
    rc.worst = std::max(rc.worst, d);
    rc.worst_bar = std::max(rc.worst_bar, g.bars[(std::size_t)r]);
    if (g.margins[(std::size_t)r] >= kDecisive) {
      ++rc.decisive;
      rc.agree += argmax_(a, vocab) == g.picks[(std::size_t)r];
    }
  }
  std::sort(rel.begin(), rel.end());
  std::sort(bar.begin(), bar.end());
  rc.median = rel[rel.size() / 2];
  rc.median_bar = bar[bar.size() / 2];
  return rc;
}

}  // namespace

// The text model teacher-forced over a prompt and the reference's own
// continuation, in ONE forward: every row's logits against fp32. The
// residual after each layer is read back too and printed against the
// reference's, which locates a wrong layer when the logits are off.
TEST(lfm3, text_forward_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto m = load_model_(mc, md);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  const Golden g = golden_(gd, "text");
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  std::vector<std::int32_t> ids = g.prompt;
  ids.insert(ids.end(), g.cont.begin(), g.cont.end());
  const int n0 = (int)g.prompt.size(), n = (int)ids.size();
  const int V = m->config().vocab, H = m->config().hidden;
  auto* cm = m->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  const genai::MetalLfm3Model::Segment s{cid, 0, n};
  std::vector<float> logits;
  std::vector<std::vector<float>> cap;
  ASSERT_TRUE(m->forward(std::span<const genai::MetalLfm3Model::Segment>(
                             &s, 1), ids, nullptr, n0 - 1, &logits, &cap));
  const auto hid = read_f32_(std::string(gd) + "/text_hidden.f32");
  if (hid.size() == (std::size_t)m->config().n_layers * n0 * H
      && (int)cap.size() == m->config().n_layers) {
    for (int L = 0; L < m->config().n_layers; ++L) {
      const double d = rel_l2_(cap[(std::size_t)L].data(),
                               hid.data() + (std::size_t)L * n0 * H,
                               (std::size_t)n0 * H);
      std::printf("[lfm3.text] layer %2d residual rel %.5f\n", L, d);
    }
  }
  const RowCheck rc = check_rows_(g, logits, V);
  std::printf("[lfm3.text] %d rows: rel median %.4f worst %.4f (ref bf16 "
              "median %.4f worst %.4f), decisive %d/%d\n", rc.rows,
              rc.median, rc.worst, rc.median_bar, rc.worst_bar, rc.agree,
              rc.decisive);
  EXPECT_TRUE(rc.median <= rc.median_bar);
  EXPECT_TRUE(rc.worst <= std::max(rc.worst_bar, 0.02));
  EXPECT_TRUE(rc.decisive > 0 && rc.agree == rc.decisive);
  cm->release(cid);
}

// The same rows through the decode path: the prompt prefilled, then the
// continuation fed one token a forward -- the conv state carried, the
// K/V read back from the pages, every row against the golden.
TEST(lfm3, decode_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto m = load_model_(mc, md);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  for (const char* name : {"text", "nothink", "image"}) {
    const Golden g = golden_(gd, name);
    if (!g.ok) { continue; }
    if (std::string(name) == "image") { continue; }   // needs the tower
    const int V = m->config().vocab;
    auto* cm = m->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    std::vector<float> all, lg;
    ASSERT_TRUE(m->prefill(cid, g.prompt, nullptr, &lg));
    all = lg;
    for (std::int32_t t : g.cont) {
      const genai::MetalLfm3Model::Segment s{cid, 0, 1};
      ASSERT_TRUE(m->forward(
          std::span<const genai::MetalLfm3Model::Segment>(&s, 1),
          std::span<const std::int32_t>(&t, 1), nullptr, 0, &lg));
      all.insert(all.end(), lg.begin(), lg.end());
    }
    const RowCheck rc = check_rows_(g, all, V);
    std::printf("[lfm3.decode] %s %d rows: rel median %.4f worst %.4f (ref "
                "bf16 median %.4f worst %.4f), decisive %d/%d\n", name,
                rc.rows, rc.median, rc.worst, rc.median_bar, rc.worst_bar,
                rc.agree, rc.decisive);
    EXPECT_TRUE(rc.median <= rc.median_bar);
    EXPECT_TRUE(rc.worst <= std::max(rc.worst_bar, 0.02));
    EXPECT_TRUE(rc.agree == rc.decisive);
    cm->release(cid);
  }
}

// The vision tower and projector from the golden's normalised pixels.
TEST(lfm3, vision_tower_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  genai::MetalLfm3VisionEncoder::Config vc;
  std::string err;
  ASSERT_TRUE(genai::MetalLfm3VisionEncoder::Config::from_dir(md, &vc, &err));
  auto enc = genai::MetalLfm3VisionEncoder::load(
      genai::WeightSet::open(md, nullptr), mc, vc, &err);
  if (!enc) { std::printf("[lfm3.vision] %s\n", err.c_str()); }
  ASSERT_TRUE(enc != nullptr);
  if (!enc) { return; }
  for (const char* tag : {"image", "image2"}) {
    const FlexData j = read_json_(std::string(gd) + "/" + tag + ".json");
    if (!j.is_object()) { continue; }
    const int h = (int)bag::integer(j, "h"), w = (int)bag::integer(j, "w");
    const auto px =
        read_f32_(std::string(gd) + "/" + tag + "_pixel_values.f32");
    ASSERT_TRUE(px.size() == (std::size_t)3 * h * w);
    if (px.size() != (std::size_t)3 * h * w) { continue; }
    std::vector<float> vit, proj;
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = enc->encode_normalized(px.data(), h, w, &vit, &proj);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(r.n_tokens == (int)bag::integer(j, "n_tokens"));
    const std::string b = std::string(gd) + "/" + tag;
    const auto v32 = read_f32_(b + "_vit.f32");
    const auto v16 = read_f32_(b + "_vit_bf16.f32");
    const auto p32 = read_f32_(b + "_proj.f32");
    const auto p16 = read_f32_(b + "_proj_bf16.f32");
    ASSERT_TRUE(vit.size() == v32.size() && proj.size() == p32.size());
    if (vit.size() != v32.size() || proj.size() != p32.size()) { continue; }
    const double vo = rel_l2_(vit.data(), v32.data(), vit.size());
    const double vr = rel_l2_(v16.data(), v32.data(), vit.size());
    const double po = rel_l2_(proj.data(), p32.data(), proj.size());
    const double pr = rel_l2_(p16.data(), p32.data(), proj.size());
    std::printf("[lfm3.vision] %s %dx%d -> %d tokens in %.1f ms | vs fp32: "
                "tower %.4f (ref bf16 %.4f) projector %.4f (ref bf16 "
                "%.4f) | mma2 %d nax %d\n", tag, h, w, r.n_tokens, ms, vo,
                vr, po, pr, (int)enc->uses_matrix_cores(),
                (int)enc->uses_nax_attention());
    EXPECT_TRUE(vo < vr);
    EXPECT_TRUE(po < pr);
  }
}

// The image turn: the tower's rows spliced at the <|image|> positions of
// the reference's prompt, prefilled, the continuation decoded; every row
// against fp32. The prompt is 276 rows, past the sliding window (256),
// so the windowed prefill and the shifted decode window both run.
TEST(lfm3, image_turn_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto m = load_model_(mc, md);
  genai::MetalLfm3VisionEncoder::Config vc;
  ASSERT_TRUE(genai::MetalLfm3VisionEncoder::Config::from_dir(md, &vc,
                                                              nullptr));
  auto enc = genai::MetalLfm3VisionEncoder::load(
      genai::WeightSet::open(md, nullptr), mc, vc);
  ASSERT_TRUE(m != nullptr && enc != nullptr);
  if (!m || !enc) { return; }
  const Golden g = golden_(gd, "image");
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  const FlexData j = read_json_(std::string(gd) + "/image.json");
  const int h = (int)bag::integer(j, "h"), w = (int)bag::integer(j, "w");
  // Through the production path: the u8 picture, no resize at this size.
  std::vector<std::uint8_t> rgb((std::size_t)3 * h * w);
  {
    std::ifstream in(std::string(gd) + "/image.rgb", std::ios::binary);
    in.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)rgb.size());
    ASSERT_TRUE((bool)in);
  }
  const auto r = enc->encode(rgb.data(), h, w);
  ASSERT_TRUE(r.n_tokens == (int)bag::integer(j, "n_tokens"));
  const int H = m->config().hidden, V = m->config().vocab;
  const std::int32_t img_id = 125030;
  std::vector<float> emb = m->embed_rows(g.prompt);
  const auto* er = static_cast<const _Float16*>(r.embeddings.contents());
  int k = 0;
  for (std::size_t i = 0; i < g.prompt.size(); ++i) {
    if (g.prompt[i] != img_id) { continue; }
    for (int d = 0; d < H; ++d) {
      emb[i * H + d] = (float)er[(std::size_t)k * H + d];
    }
    ++k;
  }
  ASSERT_TRUE(k == r.n_tokens);
  auto* cm = m->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  std::vector<float> all, lg;
  ASSERT_TRUE(m->prefill(cid, {}, &emb, &lg));
  all = lg;
  for (std::int32_t t : g.cont) {
    const genai::MetalLfm3Model::Segment s{cid, 0, 1};
    ASSERT_TRUE(m->forward(
        std::span<const genai::MetalLfm3Model::Segment>(&s, 1),
        std::span<const std::int32_t>(&t, 1), nullptr, 0, &lg));
    all.insert(all.end(), lg.begin(), lg.end());
  }
  const RowCheck rc = check_rows_(g, all, V);
  std::printf("[lfm3.image] %d image rows of %zu, %d rows: rel median %.4f "
              "worst %.4f (ref bf16 median %.4f worst %.4f), decisive "
              "%d/%d\n", k, g.prompt.size(), rc.rows, rc.median, rc.worst,
              rc.median_bar, rc.worst_bar, rc.agree, rc.decisive);
  EXPECT_TRUE(rc.median <= std::max(rc.median_bar, 0.01));
  EXPECT_TRUE(rc.worst <= std::max(rc.worst_bar, 0.02));
  EXPECT_TRUE(rc.agree == rc.decisive);
  cm->release(cid);
}

// A prompt in two forwards (its second continuing the first's K/V, conv
// state and pages) against one forward of it, and a batched step over
// branches against each branch decoded alone -- the weights read once for
// every branch, each branch's arithmetic its own: BIT-IDENTICAL.
TEST(lfm3, pieces_and_branches_agree)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto m = load_model_(mc, md);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  const Golden g = golden_(gd, "text");
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  using Seg = genai::MetalLfm3Model::Segment;
  const int V = m->config().vocab;
  auto* cm = m->context_manager();
  const std::vector<std::int32_t>& p = g.prompt;
  const int n = (int)p.size(), cut = n / 2 + 3;

  std::vector<float> one, two;
  const genai::ContextId a = cm->acquire_root();
  ASSERT_TRUE(m->prefill(a, p, nullptr, &one));
  const genai::ContextId b = cm->acquire_root();
  {
    const Seg s1{b, 0, cut};
    ASSERT_TRUE(m->forward(std::span<const Seg>(&s1, 1),
                           std::span<const std::int32_t>(p.data(),
                                                         (std::size_t)cut),
                           nullptr, cut - 1, nullptr));
    const Seg s2{b, 0, n - cut};
    ASSERT_TRUE(m->forward(std::span<const Seg>(&s2, 1),
                           std::span<const std::int32_t>(
                               p.data() + cut, (std::size_t)(n - cut)),
                           nullptr, n - cut - 1, &two));
  }
  const double d = rel_l2_(two.data(), one.data(), (std::size_t)V);
  std::printf("[lfm3.pieces] %d + %d rows vs %d: rel %.5f, picks %d / %d\n",
              cut, n - cut, n, d, argmax_(two.data(), V),
              argmax_(one.data(), V));
  EXPECT_TRUE(d < 0.01);
  EXPECT_TRUE(argmax_(two.data(), V) == argmax_(one.data(), V));

  // Branches: three children of `a`, each fed a different token, alone
  // and then in one batched step from fresh branches.
  const std::int32_t toks[3] = {g.cont[0], g.cont[1], g.cont[2]};
  std::vector<std::vector<float>> alone;
  for (int i = 0; i < 3; ++i) {
    const genai::ContextId c = cm->branch(a);
    const Seg s{c, 0, 1};
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(std::span<const Seg>(&s, 1),
                           std::span<const std::int32_t>(&toks[i], 1),
                           nullptr, 0, &lg));
    alone.push_back(lg);
    cm->release(c);
  }
  std::vector<genai::ContextId> kids;
  std::vector<Seg> segs;
  for (int i = 0; i < 3; ++i) {
    kids.push_back(cm->branch(a));
    segs.push_back({kids.back(), i, 1});
  }
  std::vector<float> batched;
  ASSERT_TRUE(m->forward(segs, std::span<const std::int32_t>(toks, 3),
                         nullptr, 0, &batched));
  int same = 0;
  for (int i = 0; i < 3; ++i) {
    same += std::memcmp(batched.data() + (std::size_t)i * V,
                        alone[(std::size_t)i].data(),
                        (std::size_t)V * 4) == 0;
  }
  std::printf("[lfm3.branches] batched == alone on %d / 3 branches\n", same);
  EXPECT_TRUE(same == 3);
  for (auto c : kids) { cm->release(c); }
  cm->release(a);
  cm->release(b);
}

// The tokenizer (byte-level BPE, the Llama-3 split with \p{N}{1,3},
// ignore_merges) against the `tokenizers` ids for the same strings. A
// case holding a special token's spelling is skipped: HF matches added
// tokens inside text, vpipe's encode() does not (templates push them).
TEST(lfm3, tokenizer_matches_reference)
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
  int checked = 0, bad = 0;
  const FlexData cases = g.as_object().at("cases");
  for (const auto& c : cases.as_array()) {
    const std::string text(c.as_object().at("text").as_string(""));
    if (text.find("<|") != std::string::npos) { continue; }
    const auto want = ids_of_(c.as_object().at("ids"));
    const auto got = tok->encode(text);
    ++checked;
    if (got != want) {
      ++bad;
      std::printf("[lfm3.tokenizer] MISMATCH '%s'\n  want:", text.c_str());
      for (auto i : want) { std::printf(" %d", i); }
      std::printf("\n  got: ");
      for (auto i : got) { std::printf(" %d", i); }
      std::printf("\n");
    }
    EXPECT_TRUE(tok->decode(std::span<const std::int32_t>(
                    got.data(), got.size())) == text);
  }
  std::printf("[lfm3.tokenizer] %d cases, %d mismatched\n", checked, bad);
  EXPECT_TRUE(checked >= 10);
  EXPECT_TRUE(bad == 0);
}

// The template renders the reference's prompt ids: a system message and
// a user turn, thinking on (`<think>`) and off (`<think></think>`), each
// opened by <|startoftext|>; and the image turn's vision block.
TEST(lfm3, chat_template_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(md) + "/tokenizer.json", nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  genai::ModelLoader loader(nullptr);
  auto cfg = loader.load_config(md);
  ASSERT_TRUE(cfg.has_value());
  if (!cfg) { return; }
  std::printf("[lfm3.template] architecture %s, vision %d, mean %.2f, "
              "min/max pixels %lld / %lld\n", cfg->architecture.c_str(),
              (int)cfg->vision.present, (double)cfg->vision.image_mean[0],
              (long long)bag::integer(cfg->extra,
                                      genai::model_config::kVisionMinPixels),
              (long long)bag::integer(
                  cfg->extra, genai::model_config::kVisionMaxPixels));
  for (const char* name : {"text", "nothink"}) {
    const FlexData j = read_json_(std::string(gd) + "/" + name + ".json");
    ASSERT_TRUE(j.is_object());
    if (!j.is_object()) { continue; }
    const bool think = bag::flag(j, "thinking", true);
    auto tpl = genai::make_chat_template(*cfg, *tok, !think, {});
    ASSERT_TRUE(tpl != nullptr);
    if (!tpl) { return; }
    std::vector<std::int32_t> ids;
    tpl->render_system_message(std::string(bag::text(j, "system")), {},
                               /*is_first_turn=*/true, &ids);
    tpl->render_user_turn(std::string(bag::text(j, "user")),
                          /*is_first_turn=*/false, &ids);
    const auto want = ids_of_(j.as_object().at("prompt_ids"));
    if (ids != want) {
      std::printf("[lfm3.template] %s MISMATCH\n  want:", name);
      for (auto i : want) { std::printf(" %d", i); }
      std::printf("\n  got: ");
      for (auto i : ids) { std::printf(" %d", i); }
      std::printf("\n");
    }
    EXPECT_TRUE(ids == want);
    EXPECT_TRUE(tpl->assistant_prompt_opens_thinking() == think);
    EXPECT_TRUE(tpl->is_stop_token(124900));    // <|im_end|>
  }
  const FlexData j = read_json_(std::string(gd) + "/image.json");
  ASSERT_TRUE(j.is_object());
  if (!j.is_object()) { return; }
  auto tpl = genai::make_chat_template(*cfg, *tok, true, {});
  ASSERT_TRUE(tpl != nullptr && tpl->image_pad_token_id() == 125030);
  if (!tpl) { return; }
  std::vector<genai::MediaChunk> chunks(2);
  chunks[0].kind = genai::MediaChunk::Kind::Image;
  chunks[0].n_tokens = (int)bag::integer(j, "n_tokens");
  chunks[1].kind = genai::MediaChunk::Kind::Text;
  chunks[1].text = std::string(bag::text(j, "user"));
  std::vector<std::int32_t> ids;
  ASSERT_TRUE(tpl->render_user_turn_media(chunks, true, &ids));
  EXPECT_TRUE(ids == ids_of_(j.as_object().at("prompt_ids")));
}

// Through the manager, as text-chat and realtime-vqa load it: the
// template the LM carries, the exec, the tower. A text turn (thinking
// off) and the picture turn decoded greedily give the reference's own
// greedy replies token for token.
TEST(lfm3, manager_chats_and_reads_a_picture)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = md;
  spec.page_tokens = 256;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm) { return; }
  ASSERT_TRUE(lm->chat_template() != nullptr);
  EXPECT_TRUE(lm->chat_template()->family_name() == "lfm3");
  auto* vis = lm->metal_lfm3_vision();
  ASSERT_TRUE(vis != nullptr);
  const auto& tok = lm->tokenizer();
  auto greedy = [&](genai::LoadedLanguageModel::Context& ctx,
                    std::int32_t t, const genai::ChatTemplate& tpl,
                    int max) {
    std::vector<std::int32_t> out;
    while (t >= 0 && (int)out.size() < max && !tpl.is_stop_token(t)) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  {
    const Golden g = golden_(gd, "nothink");
    const FlexData j = read_json_(std::string(gd) + "/nothink.json");
    auto tpl = genai::make_chat_template(lm->config(), tok, true, {});
    ASSERT_TRUE(tpl != nullptr);
    if (!tpl) { return; }
    std::vector<std::int32_t> ids;
    tpl->render_system_message(std::string(bag::text(j, "system")), {},
                               true, &ids);
    tpl->render_user_turn(std::string(bag::text(j, "user")), false, &ids);
    EXPECT_TRUE(ids == g.prompt);
    auto ctx = lm->make_context();
    const auto got = greedy(ctx, lm->prefill(ctx, ids), *tpl, 64);
    const std::string text = tok.decode(
        std::span<const std::int32_t>(got.data(), got.size()));
    std::printf("[lfm3.manager] text: '%s'\n", text.c_str());
    EXPECT_TRUE(got == g.cont);
  }

  {
    const Golden g = golden_(gd, "image");
    const FlexData j = read_json_(std::string(gd) + "/image.json");
    const int h = (int)bag::integer(j, "h"), w = (int)bag::integer(j, "w");
    std::vector<std::uint8_t> rgb((std::size_t)3 * h * w);
    std::ifstream in(std::string(gd) + "/image.rgb", std::ios::binary);
    in.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)rgb.size());
    ASSERT_TRUE((bool)in);
    const auto r = vis->encode(rgb.data(), h, w);
    ASSERT_TRUE(r.n_tokens == (int)bag::integer(j, "n_tokens"));
    auto tpl = genai::make_chat_template(lm->config(), tok, true, {});
    std::vector<genai::MediaChunk> chunks(2);
    chunks[0].kind = genai::MediaChunk::Kind::Image;
    chunks[0].n_tokens = r.n_tokens;
    chunks[1].kind = genai::MediaChunk::Kind::Text;
    chunks[1].text = std::string(bag::text(j, "user"));
    std::vector<std::int32_t> ids;
    ASSERT_TRUE(tpl->render_user_turn_media(chunks, true, &ids));
    EXPECT_TRUE(ids == g.prompt);
    std::vector<genai::TokenRef> refs(ids.size());
    int k = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == tpl->image_pad_token_id()) {
        refs[i].kind = genai::TokenRef::Kind::ImageTokens;
        refs[i].image_token_offset = k++;
        refs[i].embeddings_buf = &r.embeddings;
        refs[i].host_hidden = r.out_hidden;
        refs[i].text_id = ids[i];
      } else {
        refs[i].text_id = ids[i];
      }
    }
    auto ctx = lm->make_context();
    const std::pair<int, int> grid{r.grid_h, r.grid_w};
    const std::int32_t first = lm->prefill_multimodal_metal(
        ctx, refs, std::span<const std::pair<int, int>>(&grid, 1));
    const auto got = greedy(ctx, first, *tpl, 64);
    const std::string text = tok.decode(
        std::span<const std::int32_t>(got.data(), got.size()));
    std::printf("[lfm3.manager] picture: '%s'\n", text.c_str());
    EXPECT_TRUE(got == g.cont);
  }
}

// Speed (VPIPE_LFM3_BENCH=1): a prefill of 512 / 2048 / 8192 tokens, a
// decode token a forward, a batched step over 4 branches, the tower on a
// 512^2 and a ~1 MP picture. Prints only.
TEST(lfm3, bench)
{
  const char* md = model_dir_();
  if (!md || std::getenv("VPIPE_LFM3_BENCH") == nullptr) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  genai::MetalLfm3Model::Config cfg;
  ASSERT_TRUE(genai::MetalLfm3Model::Config::from_dir(md, &cfg, nullptr));
  cfg.page_tokens = 256;
  cfg.max_pages = 160;
  auto m = genai::MetalLfm3Model::load(genai::WeightSet::open(md, nullptr),
                                       mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  using Clock = std::chrono::steady_clock;
  using Seg = genai::MetalLfm3Model::Segment;
  auto* cm = m->context_manager();
  std::mt19937 rng(7);
  auto ids_of = [&](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (auto& x : v) { x = (std::int32_t)(rng() % 120000) + 300; }
    return v;
  };
  // Three runs a length, the last two printed: with the int8 prefill on
  // (VPIPE_I8_GEMM=1) the first records its shapes, the second's piece
  // tunes them (once a session), the third runs settled.
  for (int n : {512, 2048, 8192}) {
    for (int rep = 0; rep < 3; ++rep) {
      const genai::ContextId c = cm->acquire_root();
      const auto ids = ids_of(n);
      std::vector<float> lg;
      const auto t0 = Clock::now();
      ASSERT_TRUE(m->prefill(c, ids, nullptr, &lg));
      const double s =
          std::chrono::duration<double>(Clock::now() - t0).count();
      if (rep >= 1) {
        std::printf("[lfm3.bench] prefill %5d tokens (run %d): %.3f s, "
                    "%.0f tok/s\n", n, rep + 1, s, n / s);
      }
      if (n == 2048 && rep == 2) {
        // decode 128 tokens a forward each, after this prompt
        std::int32_t t = 1000;
        const auto t1 = Clock::now();
        for (int i = 0; i < 128; ++i) {
          const Seg sg{c, 0, 1};
          ASSERT_TRUE(m->forward(std::span<const Seg>(&sg, 1),
                                 std::span<const std::int32_t>(&t, 1),
                                 nullptr, 0, &lg));
          t = (std::int32_t)(std::max_element(lg.begin(), lg.end()) -
                             lg.begin());
        }
        const double d =
            std::chrono::duration<double>(Clock::now() - t1).count();
        std::printf("[lfm3.bench] decode at 2k context: %.2f ms a token, "
                    "%.1f tok/s\n", d * 1000.0 / 128, 128 / d);
        // the same without the head and the logits' read-back
        const auto t1b = Clock::now();
        for (int i = 0; i < 128; ++i) {
          const Seg sg{c, 0, 1};
          ASSERT_TRUE(m->forward(std::span<const Seg>(&sg, 1),
                                 std::span<const std::int32_t>(&t, 1),
                                 nullptr, 0, nullptr));
        }
        const double d2 =
            std::chrono::duration<double>(Clock::now() - t1b).count();
        std::printf("[lfm3.bench] decode without the head: %.2f ms a "
                    "token\n", d2 * 1000.0 / 128);
        // 4 branches of it, a batched step each
        std::vector<genai::ContextId> kids;
        std::vector<Seg> segs;
        for (int b = 0; b < 4; ++b) {
          kids.push_back(cm->branch(c));
          segs.push_back({kids.back(), b, 1});
        }
        std::int32_t tk[4] = {1000, 1001, 1002, 1003};
        const auto t2 = Clock::now();
        for (int i = 0; i < 64; ++i) {
          ASSERT_TRUE(m->forward(segs, std::span<const std::int32_t>(tk, 4),
                                 nullptr, 0, &lg));
        }
        const double b4 =
            std::chrono::duration<double>(Clock::now() - t2).count();
        std::printf("[lfm3.bench] batched 4 branches: %.2f ms a step, "
                    "%.1f tokens/s in all\n", b4 * 1000.0 / 64,
                    4 * 64 / b4);
        for (auto k : kids) { cm->release(k); }
      }
      cm->release(c);
    }
  }
  genai::MetalLfm3VisionEncoder::Config vc;
  ASSERT_TRUE(genai::MetalLfm3VisionEncoder::Config::from_dir(md, &vc,
                                                              nullptr));
  auto enc = genai::MetalLfm3VisionEncoder::load(
      genai::WeightSet::open(md, nullptr), mc, vc);
  ASSERT_TRUE(enc != nullptr);
  if (!enc) { return; }
  for (auto [h, w] : {std::pair<int, int>{512, 512}, {768, 1344}}) {
    std::vector<std::uint8_t> rgb((std::size_t)3 * h * w, 128);
    for (int rep = 0; rep < 2; ++rep) {
      const auto t0 = Clock::now();
      const auto r = enc->encode(rgb.data(), h, w);
      const double ms = std::chrono::duration<double, std::milli>(
          Clock::now() - t0).count();
      if (rep == 1) {
        std::printf("[lfm3.bench] tower %dx%d -> %d tokens: %.1f ms\n", h,
                    w, r.n_tokens, ms);
      }
    }
  }
}

// The decode GEMV's bandwidth at LFM3's shapes (VPIPE_LFM3_GEMV_BENCH=1):
// 200 dependent dispatches a shape in one command buffer. Prints only.
TEST(lfm3, gemv_bench)
{
  if (std::getenv("VPIPE_LFM3_GEMV_BENCH") == nullptr) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto lib = mc->load_library("lfm3");
  auto dense = mc->load_library("dense_gemm");
  const char* names[] = {"lfm3_gemv_batch_f16", "dense_gemv_t_f16"};
  const int shapes[][2] = {{6144, 2048}, {2048, 2048}, {3088, 2048},
                           {15360, 2048}, {2048, 7680}, {128000, 2048}};
  for (const auto& sh : shapes) {
    const int N = sh[0], K = sh[1];
    auto w = mc->make_shared_buffer((std::size_t)N * K * 2);
    auto x = mc->make_shared_buffer((std::size_t)K * 2);
    auto y = mc->make_shared_buffer((std::size_t)N * 2);
    auto* wp = static_cast<_Float16*>(w.contents());
    for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
      wp[i] = (_Float16)((float)(i % 97) * 1e-3f);
    }
    auto* xp = static_cast<_Float16*>(x.contents());
    for (int i = 0; i < K; ++i) { xp[i] = (_Float16)0.01f; }
    for (int f = 0; f < 2; ++f) {
      auto fn = f == 0 ? lib.function(names[0]) : dense.function(names[1]);
      const int reps = N >= 100000 ? 20 : 200;
      double best = 1e9;
      for (int round = 0; round < 3; ++round) {
        auto st = mc->make_command_stream();
        auto enc = st.begin_compute();
        for (int r = 0; r < reps; ++r) {
          enc.set_function(fn);
          enc.set_buffer(0, x);
          enc.set_buffer(1, w);
          enc.set_buffer(2, y);
          enc.set_constant(3, K);
          enc.set_constant(4, N);
          const int one = 1;
          enc.set_constant(5, one);
          enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
        }
        enc.end();
        const auto t0 = std::chrono::steady_clock::now();
        ASSERT_TRUE(st.commit().wait_ok());
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, s);
      }
      std::printf("[lfm3.gemv] %-20s N %6d K %5d: %.1f us, %.0f GB/s\n",
                  names[f], N, K, best * 1e6 / reps,
                  (double)N * K * 2 * reps / best / 1e9);
    }
  }
}

// The checkpoint on disk is detected as the catalogue names it (a model
// registered from a folder reads as one fetched), its I/O kept: it
// carries a vision_config.
TEST(lfm3, detected_as_catalogued)
{
  const char* md = model_dir_();
  if (!md) { return; }
  const DetectedModel d = detect_model_dir(md, "LiquidAI/LFM3-VL-2B");
  const DetectedModel p = detect_model_dir(md, "local/some-lfm3");
  std::printf("[lfm3.detect] '%s' / '%s' by %s, %zu inputs\n",
              d.model_type.c_str(), p.model_type.c_str(),
              p.detected_by.c_str(), p.inputs.size());
  EXPECT_TRUE(d.model_type == "lfm3-vl" && p.model_type == "lfm3-vl");
  EXPECT_TRUE(std::find(p.inputs.begin(), p.inputs.end(), "image") !=
              p.inputs.end());
}

// A prompt past one prefill piece (2048 rows) and far past the sliding
// window (256): 2584 rows prefilled in two pieces, then its last 16 rows
// run once as a continuing segment (the steel prefill over pages, the
// window's twin) and once a token a decode (the shifted window), every
// row against fp32. Golden: dump_lfm3_golden.py --long ... 2600.
TEST(lfm3, long_prompt_matches_reference)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  const FlexData j = read_json_(std::string(gd) + "/long.json");
  if (!j.is_object()) { return; }   // the --long golden is optional
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto m = load_model_(mc, md);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  const auto ids = ids_of_(j.as_object().at("ids"));
  const auto picks = ids_of_(j.as_object().at("picks"));
  const auto margins = reals_of_(j.as_object().at("margins"));
  const auto bars = reals_of_(j.as_object().at("bf16_rel"));
  const auto want = read_f32_(std::string(gd) + "/long_logits.f32");
  const int V = m->config().vocab, n = (int)ids.size(), k = 16;
  ASSERT_TRUE(want.size() == (std::size_t)k * V && picks.size() == 16u);
  if (want.size() != (std::size_t)k * V) { return; }
  using Seg = genai::MetalLfm3Model::Segment;
  auto* cm = m->context_manager();
  const std::span<const std::int32_t> head(ids.data(), (std::size_t)(n - k));
  auto check = [&](const char* how, const std::vector<float>& got) {
    std::vector<double> rel, bar(bars.begin(), bars.end());
    int decisive = 0, agree = 0;
    double worst = 0.0;
    for (int r = 0; r < k; ++r) {
      const double d = rel_l2_(got.data() + (std::size_t)r * V,
                               want.data() + (std::size_t)r * V,
                               (std::size_t)V);
      rel.push_back(d);
      worst = std::max(worst, d);
      if (margins[(std::size_t)r] >= kDecisive) {
        ++decisive;
        agree += argmax_(got.data() + (std::size_t)r * V, V) ==
                 picks[(std::size_t)r];
      }
    }
    std::sort(rel.begin(), rel.end());
    std::sort(bar.begin(), bar.end());
    std::printf("[lfm3.long] %s: %d rows at %d..%d: rel median %.4f worst "
                "%.4f (ref bf16 median %.4f worst %.4f), decisive %d/%d\n",
                how, k, n - k, n - 1, rel[k / 2], worst, bar[k / 2],
                bar.back(), agree, decisive);
    EXPECT_TRUE(rel[k / 2] <= bar[k / 2]);
    EXPECT_TRUE(worst <= std::max(bar.back(), 0.03));
    EXPECT_TRUE(agree == decisive);
  };
  {
    const genai::ContextId c = cm->acquire_root();
    std::vector<float> lg;
    ASSERT_TRUE(m->prefill(c, head, nullptr, &lg));
    const Seg s{c, 0, k};
    ASSERT_TRUE(m->forward(std::span<const Seg>(&s, 1),
                           std::span<const std::int32_t>(
                               ids.data() + n - k, (std::size_t)k),
                           nullptr, 0, &lg));
    check("segment", lg);
    cm->release(c);
  }
  {
    const genai::ContextId c = cm->acquire_root();
    std::vector<float> lg, all;
    ASSERT_TRUE(m->prefill(c, head, nullptr, &lg));
    for (int r = 0; r < k; ++r) {
      const Seg s{c, 0, 1};
      ASSERT_TRUE(m->forward(std::span<const Seg>(&s, 1),
                             std::span<const std::int32_t>(
                                 ids.data() + n - k + r, 1),
                             nullptr, 0, &lg));
      all.insert(all.end(), lg.begin(), lg.end());
    }
    check("decode", all);
    cm->release(c);
  }
}

// On a GPU with matrix cores (M5): the language model's prefill GEMMs on
// matmul2d and the tower's GEMMs and attention on the matrix units
// ENGAGE, and the text rows and the tower stay inside the bar. Skips
// elsewhere (an M4 runs the steel paths the other tests cover).
TEST(lfm3, matrix_core_paths_engage)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc || !mc->supports_matrix_cores()) { return; }
  auto m = load_model_(mc, md);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  EXPECT_TRUE(m->uses_matrix_cores());
  const Golden g = golden_(gd, "text");
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  // 64+ rows reach matmul2d: the prompt and continuation twice over.
  std::vector<std::int32_t> ids = g.prompt;
  ids.insert(ids.end(), g.cont.begin(), g.cont.end());
  auto* cm = m->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  const genai::MetalLfm3Model::Segment s{cid, 0, (int)ids.size()};
  std::vector<float> logits;
  ASSERT_TRUE(m->forward(std::span<const genai::MetalLfm3Model::Segment>(
                             &s, 1), ids, nullptr,
                         (int)g.prompt.size() - 1, &logits));
  const RowCheck rc = check_rows_(g, logits, m->config().vocab);
  std::printf("[lfm3.mma] %d rows (%zu in the forward): rel median %.4f "
              "worst %.4f (ref bf16 median %.4f), decisive %d/%d\n",
              rc.rows, ids.size(), rc.median, rc.worst, rc.median_bar,
              rc.agree, rc.decisive);
  EXPECT_TRUE(rc.median <= rc.median_bar);
  EXPECT_TRUE(rc.agree == rc.decisive);
  cm->release(cid);
  genai::MetalLfm3VisionEncoder::Config vc;
  ASSERT_TRUE(genai::MetalLfm3VisionEncoder::Config::from_dir(md, &vc,
                                                              nullptr));
  auto enc = genai::MetalLfm3VisionEncoder::load(
      genai::WeightSet::open(md, nullptr), mc, vc);
  ASSERT_TRUE(enc != nullptr);
  if (!enc) { return; }
  EXPECT_TRUE(enc->uses_matrix_cores() && enc->uses_nax_attention());
}

// The int8 prefill (the stages' i8_prefill) on matrix cores: the long
// golden's 2584 rows prefilled with it -- the 2048-row piece clears the
// int8 context's row floor (1024), the 536-row piece and the 16 rows
// after do not -- then those 16 rows against fp32 and against the f16
// run. What took the route: the four projections of each of the 24
// layers in the one piece, the head never. Twice, so the second run's
// piece meets the shapes the tuner settled after the first. Without
// matrix cores the context declines and the flag must leave it off.
TEST(lfm3, i8_prefill_engages_and_stays_close)
{
  const char* md = model_dir_();
  const char* gd = golden_dir_();
  if (!md || !gd) { return; }
  const FlexData j = read_json_(std::string(gd) + "/long.json");
  if (!j.is_object()) { return; }
  Session sess;
  auto* mc = sess.services()->metal_compute();
  ASSERT_TRUE(mc != nullptr && mc->valid());
  if (!mc) { return; }
  auto m = load_model_(mc, md);
  ASSERT_TRUE(m != nullptr);
  if (!m) { return; }
  EXPECT_TRUE(!m->i8_gemm_enabled() ||
              std::getenv("VPIPE_I8_GEMM") != nullptr);
  const auto ids = ids_of_(j.as_object().at("ids"));
  const auto picks = ids_of_(j.as_object().at("picks"));
  const auto margins = reals_of_(j.as_object().at("margins"));
  const auto bars = reals_of_(j.as_object().at("bf16_rel"));
  const auto want = read_f32_(std::string(gd) + "/long_logits.f32");
  const int V = m->config().vocab, n = (int)ids.size(), k = 16;
  ASSERT_TRUE(want.size() == (std::size_t)k * V && picks.size() == 16u);
  if (want.size() != (std::size_t)k * V) { return; }
  using Seg = genai::MetalLfm3Model::Segment;
  auto* cm = m->context_manager();
  auto run = [&](std::vector<float>* lg) {
    const genai::ContextId c = cm->acquire_root();
    std::vector<float> last;
    bool ok = m->prefill(c, std::span<const std::int32_t>(
                                ids.data(), (std::size_t)(n - k)),
                         nullptr, &last);
    const Seg s{c, 0, k};
    ok = ok && m->forward(std::span<const Seg>(&s, 1),
                          std::span<const std::int32_t>(
                              ids.data() + n - k, (std::size_t)k),
                          nullptr, 0, lg);
    cm->release(c);
    return ok;
  };
  std::vector<float> plain;
  ASSERT_TRUE(run(&plain));
  m->set_i8_gemm(true);
  if (!mc->supports_matrix_cores()) {
    EXPECT_TRUE(!m->i8_gemm_enabled());
    return;
  }
  ASSERT_TRUE(m->i8_gemm_enabled());
  std::vector<double> bar(bars.begin(), bars.end());
  std::sort(bar.begin(), bar.end());
  for (int pass = 0; pass < 2; ++pass) {
    const std::uint64_t n0 = m->i8_gemm_count();
    std::vector<float> got;
    ASSERT_TRUE(run(&got));
    if (got.size() != plain.size()) { return; }
    const std::uint64_t took = m->i8_gemm_count() - n0;
    std::vector<double> rel, off;
    int decisive = 0, agree = 0;
    for (int r = 0; r < k; ++r) {
      const std::size_t o = (std::size_t)r * V;
      rel.push_back(rel_l2_(got.data() + o, want.data() + o,
                            (std::size_t)V));
      off.push_back(rel_l2_(got.data() + o, plain.data() + o,
                            (std::size_t)V));
      if (margins[(std::size_t)r] >= kDecisive) {
        ++decisive;
        agree += argmax_(got.data() + o, V) == picks[(std::size_t)r];
      }
    }
    std::sort(rel.begin(), rel.end());
    std::sort(off.begin(), off.end());
    std::printf("[lfm3.i8] run %d: %llu GEMMs took int8 | %d rows: rel "
                "median %.4f worst %.4f from fp32 (ref bf16 median %.4f "
                "worst %.4f), %.4f / %.4f from f16, decisive %d/%d\n",
                pass + 1, (unsigned long long)took, k, rel[k / 2],
                rel.back(), bar[k / 2], bar.back(), off[k / 2],
                off.back(), agree, decisive);
    if (std::getenv("VPIPE_I8_GEMM_MIN_M") == nullptr) {
      EXPECT_TRUE(took == 4u * (std::uint64_t)m->config().n_layers);
    }
    // From fp32 inside the reference's own bf16 spread, and moved from
    // the f16 run by no more than that spread: the int8 rounding costs
    // what bf16 rounding does.
    EXPECT_TRUE(rel[k / 2] <= bar[k / 2]);
    EXPECT_TRUE(rel.back() <= std::max(bar.back(), 0.03));
    EXPECT_TRUE(off[k / 2] <= bar[k / 2] && off.back() <= bar.back());
    EXPECT_TRUE(agree == decisive);
  }
  // Asked again for the state it is in (every stage a shared model is
  // handed to asks), the model keeps its context and what it settled;
  // asked off, it lets the context and its scratch go.
  const genai::I8GemmContext* held = m->i8_context();
  m->set_i8_gemm(true);
  EXPECT_TRUE(held != nullptr && m->i8_context() == held);
  m->set_i8_gemm(false);
  EXPECT_TRUE(m->i8_context() == nullptr && !m->i8_gemm_enabled());
}

// The int8 prefill's scratch is in the plan: through the manager, with
// i8_prefill on, an 8192-token prefill + 8 steps grows the device's
// allocation by what lm_run_bytes books for the shape -- more than the
// f16 run's figure, which it would pass if the context's scratches were
// left out. Matrix cores only.
TEST(lfm3, i8_prefill_books_its_scratch)
{
  const char* md = model_dir_();
  if (!md) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.services()->metal_compute();
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mc != nullptr && mgr != nullptr);
  if (!mc || !mgr || !mc->supports_matrix_cores()) { return; }
  (void)mc->memory_budget_settled(15000);
  genai::LoadSpec spec;
  spec.hf_dir = md;
  spec.page_tokens = 512;
  spec.max_pages = 17;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm) { return; }
  lm->set_i8_prefill(true);
  genai::LmRunShape shape;
  shape.page_tokens = spec.page_tokens;
  shape.max_pages = spec.max_pages;
  shape.kv_tokens = (std::int64_t)spec.page_tokens * spec.max_pages;
  const std::size_t f16 = genai::lm_run_bytes(&sess, md, shape);
  shape.i8_prefill = true;
  const std::size_t booked = genai::lm_run_bytes(&sess, md, shape);
  const int kPrompt = 8192, kSteps = 8;
  std::vector<std::int32_t> ids((std::size_t)kPrompt);
  for (int i = 0; i < kPrompt; ++i) {
    ids[(std::size_t)i] = 1000 + (std::int32_t)((i * 7919) % 20000);
  }
  const std::size_t alloc0 = mc->memory_budget().allocated;
  std::atomic<std::size_t> peak{alloc0};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  auto ctx = lm->make_context();
  std::int32_t t = lm->prefill(ctx, ids);
  for (int i = 0; i < kSteps && t >= 0; ++i) {
    t = lm->next_token_greedy(ctx);
  }
  done.store(true);
  sampler.join();
  ASSERT_TRUE(t >= 0);
  const std::size_t top = std::max(peak.load(),
                                   mc->memory_budget().allocated);
  const std::size_t grew = top > alloc0 ? top - alloc0 : 0;
  std::printf("[lfm3.i8] %d-token prefill + %d steps with int8 peaked "
              "%.1f MB past the load; booked %.1f MB (f16 run's figure "
              "%.1f MB)\n", kPrompt, kSteps, grew / 1048576.0,
              booked / 1048576.0, f16 / 1048576.0);
  EXPECT_TRUE(booked > f16);
  EXPECT_TRUE(grew > f16);
  EXPECT_TRUE(booked + ((std::size_t)2 << 20) >= grew);
  EXPECT_TRUE((double)booked <= 1.3 * (double)grew + (32 << 20));
}
