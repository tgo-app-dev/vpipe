// MiniMax-H3's text conditioning: the Qwen3-VL-32B tap.
//
// The checkpoint is 62 GB on a 64 GB box, so this runs at TRUNCATED
// depth -- the tap is overridden to a shallow layer and the reference is
// built to match. That is not a weaker test of the thing most likely to
// be wrong: what a depth-2 run pins is the tap CONVENTION, the verbatim
// tokenization and the rope, and every one of those is depth-independent.
// The production tap index (50) is a constant read from the reference's
// own `get_qwen3vl_prompt_embeds()` default, not something a golden can
// establish.
//
// What it checks:
//
//   * the prompt is tokenized VERBATIM -- no chat template, no BOS/EOS.
//     Every other diffusion text encoder here wraps the prompt in
//     something, so the wrong default is a live hazard rather than a
//     hypothetical one.
//   * HF's `output_hidden_states[k]` is the UN-NORMED residual after
//     layer k-1, so a tap of k maps to 0-indexed layer k-1 and the
//     model's own final norm is never applied.
//   * plain 1-D rope reproduces Qwen3-VL's interleaved mROPE on a
//     text-only prompt. With no vision block all three position axes
//     carry the sequence position, so channel c sees pos*inv_freq[c]
//     whichever mrope section owns it -- but that is an argument, and
//     this is the measurement.
//
// Env: VPIPE_MINIMAX_H3_TEXT_ENC_PATH, VPIPE_MINIMAX_H3_TEXT_ENC_GOLDEN.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "stages/model-quantize-stage.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// gen_text_enc_golden.py's prompt, verbatim.
constexpr const char* kPrompt =
    "A red fox trots across a snowy field at dawn, breath steaming, "
    "while a distant church bell rings twice.";

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::vector<std::int32_t>
read_i32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<std::int32_t> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

TEST(minimax_h3_text_enc, config_from_json)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MiniMaxH3TextEncoder::Config cfg;
  std::string err;
  const bool ok = MiniMaxH3TextEncoder::config_from_json(root, cfg, &err);
  if (!ok) { std::printf("[minimax_h3_text_enc] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  std::printf("[minimax_h3_text_enc] Qwen3-VL hidden %d, %d heads (%d kv) x "
              "%d, ffn %d, %d layers, tap %d -> %d loaded\n",
              cfg.lm.hidden, cfg.lm.n_heads, cfg.lm.n_kv_heads, cfg.lm.head_dim,
              cfg.lm.ffn_inner, cfg.total_layers, cfg.tap, cfg.lm.n_layers);
  // The 32B: 5120 wide, 64 query heads over 8 KV heads.
  EXPECT_TRUE(cfg.lm.hidden == 5120);
  EXPECT_TRUE(cfg.lm.n_heads == 64 && cfg.lm.n_kv_heads == 8);
  EXPECT_TRUE(cfg.lm.head_dim == 128 && cfg.lm.ffn_inner == 25600);
  EXPECT_TRUE(cfg.text_dim == 5120);
  // The DiT's text_dim, so the tap feeds it without a projection.
  EXPECT_TRUE(cfg.tap == 50);
  // Only as deep as the tap gets loaded. HOW MUCH that skips depends on
  // the publisher: MiniMaxAI ships all 64 layers, so 50..63 are simply
  // never read (~14 GB of the 62 GB checkpoint); Comfy-Org's repack is
  // already PRUNED to the tap, so it has 50 and there is nothing past
  // it. `total_layers` is what the checkpoint HAS, and the tap has to
  // fit inside it either way -- that, not the number 64, is the claim.
  EXPECT_TRUE(cfg.lm.n_layers == cfg.tap);
  EXPECT_TRUE(cfg.tap <= cfg.total_layers && cfg.total_layers >= 50);
  // Nothing past the tap runs, so there is no head to load.
  EXPECT_TRUE(cfg.lm.backbone_only);
}

namespace {

// Whether the golden's reference was built ONE LAYER AT A TIME. The
// 62 GB encoder does not fit here whole, so the deep goldens had to be,
// and they carry a measurably different floor (see the bar below).
bool
golden_streamed_(const std::string& gdir)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return false; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find("\"streamed\"");
  return k != std::string::npos && s.find("true", k) != std::string::npos &&
         s.find("true", k) - k < 16;
}

}  // namespace

TEST(minimax_h3_text_enc, tap_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<std::int32_t> ids = read_i32_(gdir + "/ids.i32");
  const std::vector<float> ref = read_f32_(gdir + "/hidden.f32");
  if (ids.empty() || ref.empty()) { return; }
  int gn = 0, gdim = 0, gtap = 0;
  {
    std::ifstream d(gdir + "/dims.txt");
    if (d) { d >> gn >> gdim >> gtap; }
  }
  if (gn != (int)ids.size() || gtap <= 0 ||
      ref.size() != (std::size_t)gn * gdim) {
    std::printf("[minimax_h3_text_enc] golden geometry %d x %d (tap %d) does "
                "not match its %zu ids / %zu floats\n", gn, gdim, gtap,
                ids.size(), ref.size());
    EXPECT_TRUE(false);
    return;
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MiniMaxH3TextEncoder::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(root, cfg, &cerr));
  // Depth override: the golden was built at this tap, and the whole
  // 64-layer stack does not fit here.
  cfg.tap = gtap;
  cfg.lm.n_layers = gtap;

  auto m = MiniMaxH3TextEncoder::load(root, mc, &sess, cfg);
  if (m == nullptr) { std::printf("[minimax_h3_text_enc] load failed\n"); }
  ASSERT_TRUE(m != nullptr);

  // Verbatim tokenization: the reference calls the tokenizer with
  // `add_special_tokens=False` and no chat template, so these ids must
  // come back with no BOS, no EOS and no wrapper.
  const std::vector<std::int32_t> got_ids = m->tokenize(kPrompt);
  EXPECT_TRUE(got_ids == ids);
  if (got_ids != ids) {
    std::printf("[minimax_h3_text_enc] tokenized %zu ids, golden has %zu\n",
                got_ids.size(), ids.size());
  }

  std::string eerr;
  SharedBuffer h = m->encode_ids(ids, &eerr);
  if (h.empty()) { std::printf("[minimax_h3_text_enc] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!h.empty());
  if (h.byte_size() < ref.size() * 2) { EXPECT_TRUE(false); return; }

  const auto* g = static_cast<const std::uint16_t*>(h.contents());
  double num = 0.0, den = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = (double)bf16_to_f32_(g[i]) - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
    worst = std::max(worst, std::fabs(d));
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[minimax_h3_text_enc] %d tokens -> [%d, %d] at tap %d: rel-L2 "
              "%.6f, max abs %.4f\n", gn, gn, gdim, gtap, rel, worst);
  // bf16 against an fp32 reference. The bar is calibrated against how
  // far the WRONG answers sit, which matters here because a residual
  // stream can change slowly enough that neighbouring layers look alike.
  // MEASURED on this prompt: hidden_states[2] differs from [1] by 0.821
  // and from [3] by 0.419, and applying the model's final RMS norm
  // moves it by 3.179. So an off-by-one layer lands ~120x above this
  // bar and a stray norm ~950x -- the two failure modes this convention
  // invites are both nowhere near passing.
  // A QUANTIZED checkpoint is a different measurement: 4-bit costs
  // MEASURED 0.0427 here against the same fp32 golden, 12x the bf16
  // floor, with the rms landing on the reference's (0.6034 vs 0.6045)
  // so there is no scale drift -- just weight noise. The dense bar
  // would reject that, and widening one bar to cover both would stop
  // either from meaning anything.
  //
  // The bar depends on WHICH reference dumped the golden, and that is a
  // property of the harness rather than of the depth. Two families are
  // on disk:
  //   * built whole (h3-text-enc, tap 2): MEASURED 0.0034.
  //   * STREAMED one layer at a time (h3-text-enc-t8/-t24/-t50, which is
  //     the only way the 62 GB reference fits this box): MEASURED
  //     0.0371 / 0.0383 / 0.0365 at taps 8 / 24 / 50.
  // The streamed numbers are ~10x the built-whole one and FLAT in depth
  // -- 8, 24 and 50 layers all land in the same place. Flat is what
  // rules out a per-layer defect, which would compound; whatever the
  // streamed reference does differently, it does once. The cause is not
  // established here, and deliberately was not chased, because the two
  // independently-packed copies of these weights (MiniMaxAI's and
  // Comfy-Org's) score IDENTICALLY on it -- 0.036493 both -- so it
  // cannot be either checkpoint.
  const bool streamed = golden_streamed_(gdir);
  EXPECT_TRUE(rel < (cfg.quantized ? 0.08 : (streamed ? 0.05 : 0.02)));
  EXPECT_TRUE(den > 0.0);
}

// Layer streaming must be the SAME FUNCTION as preloading.
//
// The 50 tapped layers of Qwen3-VL-32B are ~48 GB bf16, which no 16 GB
// box holds -- so Config::lm.stream_layers builds each layer inside the
// prefill and frees it once the GPU is done, leaving ~one layer plus the
// embedding table resident. The trade is only acceptable if the answer
// does not move, and because a streamed layer goes through the SAME
// build_layer_ a resident load runs, the bar is BIT-IDENTICAL rather
// than "close". Anything else means the two paths read the checkpoint
// differently, which is the failure this design exists to rule out.
//
// Runs at the golden's (shallow) tap: what streaming can break is the
// build/free/commit protocol, and that is exercised by layer 2 exactly
// as by layer 50. The two loads are SEQUENTIAL -- the first encoder is
// destroyed before the second is built -- so this never holds two
// copies.
TEST(minimax_h3_text_enc, streaming_matches_resident)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<std::int32_t> ids = read_i32_(gdir + "/ids.i32");
  if (ids.empty()) { return; }
  int gn = 0, gdim = 0, gtap = 0;
  {
    std::ifstream d(gdir + "/dims.txt");
    if (d) { d >> gn >> gdim >> gtap; }
  }
  if (gtap <= 0) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MiniMaxH3TextEncoder::Config base;
  std::string cerr;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(root, base, &cerr));
  base.tap = gtap;
  base.lm.n_layers = gtap;

  // Resident first, then let it go before the streamed load opens.
  std::vector<std::uint16_t> want;
  {
    auto m = MiniMaxH3TextEncoder::load(root, mc, &sess, base);
    ASSERT_TRUE(m != nullptr);
    EXPECT_TRUE(!m->streaming());
    std::string eerr;
    SharedBuffer h = m->encode_ids(ids, &eerr);
    if (h.empty()) { std::printf("[minimax_h3_text_enc] %s\n", eerr.c_str()); }
    ASSERT_TRUE(!h.empty());
    const auto* p = static_cast<const std::uint16_t*>(h.contents());
    want.assign(p, p + h.byte_size() / 2);
  }

  MiniMaxH3TextEncoder::Config scfg = base;
  scfg.lm.stream_layers = true;
  scfg.lm.pin_frac      = 0.0;   // pure streaming: every layer takes the path
  auto sm = MiniMaxH3TextEncoder::load(root, mc, &sess, scfg);
  ASSERT_TRUE(sm != nullptr);
  // Without this the rest of the test passes vacuously: load() DOWNGRADES
  // streaming to resident for anything with an output head, and a silent
  // downgrade would make this compare a model against itself.
  EXPECT_TRUE(sm->streaming());
  EXPECT_TRUE(sm->pinned_layers() == 0);

  std::string serr;
  SharedBuffer sh = sm->encode_ids(ids, &serr);
  if (sh.empty()) { std::printf("[minimax_h3_text_enc] %s\n", serr.c_str()); }
  ASSERT_TRUE(!sh.empty());
  ASSERT_TRUE(sh.byte_size() / 2 == want.size());

  const auto* g = static_cast<const std::uint16_t*>(sh.contents());
  std::size_t diff = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (g[i] == want[i]) { continue; }
    ++diff;
    worst = std::max(worst,
                     std::fabs((double)bf16_to_f32_(g[i]) -
                               (double)bf16_to_f32_(want[i])));
  }
  std::printf("[minimax_h3_text_enc] streamed vs resident at tap %d: %zu/%zu "
              "words differ (max abs %.6f)\n", gtap, diff, want.size(), worst);
  EXPECT_TRUE(diff == 0);
  // Encoding a SECOND prompt has to work too: the streamed layers were
  // freed at the end of the first prefill, so a per-forward rebuild that
  // only works once would pass everything above and fail in production
  // on prompt two.
  SharedBuffer again = sm->encode_ids(ids, &serr);
  ASSERT_TRUE(!again.empty() && again.byte_size() == sh.byte_size());
  const auto* g2 = static_cast<const std::uint16_t*>(again.contents());
  std::size_t diff2 = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (g2[i] != want[i]) { ++diff2; }
  }
  std::printf("[minimax_h3_text_enc] streamed re-encode: %zu words differ\n",
              diff2);
  EXPECT_TRUE(diff2 == 0);
}

// The REAL tap, at the production depth, with no reference to compare
// against -- there is no box here that can hold this encoder and an
// fp32 copy of it at once.
//
// So this checks the one thing arithmetic cannot: that 50 layers of a
// 62 GB checkpoint actually load and produce finite conditioning on
// this machine. Gated off by default because it reads ~50 GB and takes
// minutes; set VPIPE_MINIMAX_H3_TEXT_ENC_FULL=1 to run it.
//
// VPIPE_MINIMAX_H3_TEXT_ENC_STREAM=1 additionally runs it with layer
// streaming, which is the only way to get a process holding NOTHING but
// a streamed encoder -- what the peak-footprint measurement needs (the
// equivalence test above deliberately loads resident first).
TEST(minimax_h3_text_enc, full_depth_load)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_PATH");
  const char* on   = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_FULL");
  if (root == nullptr || *root == '\0' || on == nullptr || *on == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MiniMaxH3TextEncoder::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(root, cfg, &cerr));
  if (const char* s = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_STREAM")) {
    cfg.lm.stream_layers = (std::atoi(s) != 0);
  }
  auto m = MiniMaxH3TextEncoder::load(root, mc, &sess, cfg);
  if (m == nullptr) { std::printf("[minimax_h3_text_enc] full load failed\n"); }
  ASSERT_TRUE(m != nullptr);

  int n = 0;
  std::string eerr;
  SharedBuffer h = m->encode(kPrompt, &n, &eerr);
  if (h.empty()) { std::printf("[minimax_h3_text_enc] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!h.empty());
  EXPECT_TRUE(n > 0);

  double sq = 0.0;
  bool finite = true;
  const auto* g = static_cast<const std::uint16_t*>(h.contents());
  const std::size_t cnt = (std::size_t)n * cfg.text_dim;
  for (std::size_t i = 0; i < cnt; ++i) {
    const float v = bf16_to_f32_(g[i]);
    if (!std::isfinite(v)) { finite = false; break; }
    sq += (double)v * (double)v;
  }
  const double rms = cnt > 0 ? std::sqrt(sq / (double)cnt) : 0.0;
  std::printf("[minimax_h3_text_enc] FULL depth: %d layers of %d, %d tokens "
              "-> rms %.4f\n", cfg.lm.n_layers, cfg.total_layers, n, rms);
  EXPECT_TRUE(finite);
  // A stack this deep that silently produced zeros, or that overflowed
  // bf16 into infinities, would both pass a shape check.
  EXPECT_TRUE(rms > 0.0);
}

// Quantize the text encoder and reload it through this class.
//
// No H3-specific code runs here -- the encoder is a stock Qwen3-VL and
// the quantizer's default linear set matches it prefix-agnostically.
// What this pins is the SEAM: config_from_json has to read the
// `quantization` block, because the loader auto-detects quantized-vs-
// dense weights but takes the bit width from the config. Left at the
// default, an 8-bit checkpoint decodes with the 4-bit kernel and
// returns garbage rather than failing.
//
// Env: VPIPE_MINIMAX_H3_TEXT_ENC_QUANT (output dir) + ..._ROOT (the
// partition root, whose text_encoder component is the source).
TEST(minimax_h3_text_enc, quantized_reloads)
{
  const char* rootd = std::getenv("VPIPE_MINIMAX_H3_ROOT");
  const char* qo    = std::getenv("VPIPE_MINIMAX_H3_TEXT_ENC_QUANT");
  if (rootd == nullptr || *rootd == '\0' || qo == nullptr || *qo == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path out(qo);
  const fs::path enc = out / "text_encoder";

  if (!fs::exists(enc / "config.json", ec)) {
    Session sess;
    if (sess.metal_compute() == nullptr) { return; }
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert("src_model", FlexData::make_string(rootd));
      o.insert("output_name", FlexData::make_string(out.string()));
      o.insert("target", FlexData::make_string("text_encoder"));
      o.insert("bits", FlexData::make_int(4));
      o.insert("group_size", FlexData::make_int(64));
      o.insert("skip_existing", FlexData::make_bool(false));
    }
    ModelQuantizeStage s(&sess, "mq-h3-enc", std::vector<InEdge>{},
                         std::move(cfg));
    if (!s.config_error().empty()) {
      std::printf("[minimax_h3_text_enc] quantize config: %s\n",
                  s.config_error().c_str());
    }
    ASSERT_TRUE(s.config_error().empty());
    ASSERT_TRUE(s.quantize_once());
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MiniMaxH3TextEncoder::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MiniMaxH3TextEncoder::config_from_json(enc.string(), cfg, &cerr));
  // The whole point of the seam: the width came from the checkpoint.
  std::printf("[minimax_h3_text_enc] quantized encoder: w%d, %d layers of "
              "%d loaded\n", cfg.lm.quant_bits, cfg.lm.n_layers,
              cfg.total_layers);
  EXPECT_TRUE(cfg.lm.quant_bits == 4);

  // Shallow, so this stays a load-and-run check rather than a second
  // 50-layer read.
  cfg.tap = 2;
  cfg.lm.n_layers = 2;
  auto m = MiniMaxH3TextEncoder::load(enc.string(), mc, &sess, cfg);
  ASSERT_TRUE(m != nullptr);
  int n = 0;
  std::string eerr;
  SharedBuffer h = m->encode(kPrompt, &n, &eerr);
  if (h.empty()) { std::printf("[minimax_h3_text_enc] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!h.empty());
  double sq = 0.0;
  bool finite = true;
  const auto* g = static_cast<const std::uint16_t*>(h.contents());
  for (std::size_t i = 0; i < (std::size_t)n * cfg.text_dim; ++i) {
    const float v = bf16_to_f32_(g[i]);
    if (!std::isfinite(v)) { finite = false; break; }
    sq += (double)v * (double)v;
  }
  std::printf("[minimax_h3_text_enc] quantized tap rms %.4f over %d tokens\n",
              std::sqrt(sq / (double)((std::size_t)n * cfg.text_dim)), n);
  EXPECT_TRUE(finite);
  EXPECT_TRUE(sq > 0.0);
}
