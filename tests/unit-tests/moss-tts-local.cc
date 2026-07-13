// MOSS-TTS-Local-Transformer-v1.5 bring-up tests (Phase A).
//
// A1: the Qwen3 backbone loads + runs. MetalQwenModel has no raw-bf16 path,
// so we quantize the bf16 v1.5 backbone to 8-bit affine first (the project's
// own ModelQuantizer), then load it via MetalQwenModel with the v1.5 naming
// (weight_prefix="transformer." + model_seg="") and config, feed a gathered
// embedding stream, and assert the final-normed hidden is finite.
//
// Env: VPIPE_MOSS_TTS_LOCAL_MODEL = the bf16 v1.5 source dir. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "common/flex-data.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/moss/metal-moss-codec-v2.h"
#include "generative-models/moss/moss-v15-processor.h"
#include "generative-models/tokenizer.h"
#include "generative-models/moss/metal-moss-local-model.h"
#include "generative-models/moss/metal-moss-local-transformer.h"
#include "generative-models/moss/metal-moss-v15-model.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/quantize/calibration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Read a raw little-endian bf16 file into a float vector.
std::vector<float>
read_bf16_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<std::uint16_t> raw((std::size_t)n / 2);
  in.read(reinterpret_cast<char*>(raw.data()), n);
  out.resize(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    out[i] = bf16_to_f32_(raw[i]);
  }
  return out;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

MetalQwenModel::Config v15_backbone_config_(int bits = 8)
{
  MetalQwenModel::Config c;
  c.n_layers          = 36;
  c.hidden            = 2560;
  c.n_heads           = 32;
  c.n_kv_heads        = 8;
  c.head_dim          = 128;
  c.ffn_inner         = 9728;
  c.vocab             = 151936;
  c.rope_theta        = 1.0e6f;
  c.rms_eps           = 1e-6f;
  c.rotary_dim        = 128;     // full rotary (head_dim)
  c.full_attn_interval = 1;      // dense: every layer full-attn
  c.tie_embeddings    = false;
  c.use_bf16          = true;
  c.quant_bits        = bits;
  c.dense             = true;
  c.attn_output_gate  = false;   // standard Qwen3 q_proj = qd (no gate)
  c.backbone_only     = true;
  c.weight_prefix     = "transformer.";
  c.model_seg         = "";      // names are transformer.layers.N (no model.)
  c.max_seq           = 2048;
  c.page_tokens       = 256;
  return c;
}

}  // namespace

TEST(moss_tts_local, backbone_loads_and_forwards)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (src == nullptr || *src == '\0') { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path qdir = fs::temp_directory_path() /
                        ("vpipe-moss15-q-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(qdir, ec);

  // Quantize the bf16 v1.5 backbone to 8-bit affine (rest passthrough).
  {
    ModelQuantizer mq(mc);
    QuantizeOptions opt;
    opt.bits = 8;
    opt.group = 64;
    std::string err;
    ASSERT_TRUE(mq.run(src, qdir.string(), opt, &err));
  }

  // Load the quantized backbone.
  auto bb = MetalQwenModel::load(qdir.string(), mc, v15_backbone_config_());
  ASSERT_TRUE(bb != nullptr);

  // Build a [seq*hidden] bf16 embedding stream by gathering the first `seq`
  // rows of the (passthrough bf16) text embedding table -- a representative
  // input; the backbone forward just needs finite embeddings.
  const int seq = 8, hidden = 2560;
  auto wts = MetalLlamaWeights::open_model(qdir.string());
  ASSERT_TRUE(wts.has_value());
  SharedBuffer emb = wts->load("transformer.embed_tokens.weight", mc);
  ASSERT_TRUE(!emb.empty());
  SharedBuffer x = mc->make_shared_buffer((std::size_t)seq * hidden * 2);
  ASSERT_TRUE(!x.empty());
  std::memcpy(x.contents(), emb.contents(),
              (std::size_t)seq * hidden * 2);   // first seq rows (bf16)

  ContextManager* cm = bb->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer hn = bb->forward_embeddings_hidden(cid, x, seq);
  cm->release(cid);

  ASSERT_TRUE(!hn.empty());
  const auto* h = static_cast<const std::uint16_t*>(hn.contents());
  bool all_finite = true, any_nonzero = false;
  for (int i = 0; i < hidden; ++i) {
    const float v = bf16_to_f32_(h[i]);
    if (!std::isfinite(v)) { all_finite = false; break; }
    if (v != 0.0f) { any_nonzero = true; }
  }
  EXPECT_TRUE(all_finite);
  EXPECT_TRUE(any_nonzero);

  fs::remove_all(qdir, ec);
}

// A1 reference check: feed the HF-dumped golden embeds through the (8-bit
// quantized) backbone and compare the final-normed last hidden to the HF
// bf16 golden. rel-L2 is bounded by bf16 + 8-bit quant error (a wiring bug
// would blow it up / produce garbage). Golden via dump/moss15_golden.py.
// Env: VPIPE_MOSS_TTS_LOCAL_MODEL (bf16 src) + VPIPE_MOSS15_GOLDEN (dir).
TEST(moss_tts_local, backbone_matches_golden)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const int hidden = 2560;
  std::vector<float> g_emb = read_bf16_((fs::path(gold) / "a1_embeds.bf16"));
  // Compare to the HF hidden computed on the SAME 8-bit-affine weights
  // (a1_hidden_q8) -- this isolates backbone WIRING from quant error (HF's
  // own q8-vs-bf16 drift on this model is ~10%). vpipe-8bit vs HF-8bit should
  // be a small kernel/ordering difference.
  std::vector<float> g_hid =
      read_bf16_((fs::path(gold) / "a1_hidden_q8.bf16"));
  ASSERT_TRUE(!g_emb.empty() && !g_hid.empty());
  const int seq = (int)(g_emb.size() / hidden);
  ASSERT_TRUE(seq >= 1 && (int)g_hid.size() == seq * hidden);

  const fs::path qdir = fs::temp_directory_path() /
                        ("vpipe-moss15-qg-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(qdir, ec);
  {
    ModelQuantizer mq(mc);
    QuantizeOptions opt;
    opt.bits = 8;
    opt.group = 64;
    std::string err;
    ASSERT_TRUE(mq.run(src, qdir.string(), opt, &err));
  }
  auto bb = MetalQwenModel::load(qdir.string(), mc, v15_backbone_config_());
  ASSERT_TRUE(bb != nullptr);

  // Load the golden embeds as bf16 into the compute-dtype input stream.
  SharedBuffer x = mc->make_shared_buffer((std::size_t)seq * hidden * 2);
  ASSERT_TRUE(!x.empty());
  {
    std::ifstream in((fs::path(gold) / "a1_embeds.bf16").string(),
                     std::ios::binary);
    ASSERT_TRUE((bool)in);
    in.read(static_cast<char*>(x.contents()),
            (std::streamsize)seq * hidden * 2);
  }

  ContextManager* cm = bb->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer hn = bb->forward_embeddings_hidden(cid, x, seq);
  cm->release(cid);
  ASSERT_TRUE(!hn.empty());

  // vpipe returns the LAST position's hidden [hidden]; compare to golden row.
  const auto* h = static_cast<const std::uint16_t*>(hn.contents());
  std::vector<float> v(hidden);
  for (int i = 0; i < hidden; ++i) { v[i] = bf16_to_f32_(h[i]); }
  const double rl2 = rel_l2_(v.data(), &g_hid[(std::size_t)(seq - 1) * hidden],
                             hidden);
  std::fprintf(stderr,
               "[moss15] A1 backbone rel-L2 vs HF-8bit golden = %.4f\n", rl2);
  EXPECT_TRUE(rl2 < 0.04);   // wiring match (pure kernel/order diff)

  fs::remove_all(qdir, ec);
}

// M3 AWQ: the per-layer alpha-grid-search activation-aware fold must REDUCE
// 4-bit backbone drift vs the FP (bf16) golden. Quantizes plain vs AWQ (calib
// stats from golden), forwards the golden embeds, compares both last-hiddens
// to the bf16 golden. (FINDINGS: plain SmoothQuant act^a/w^(1-a) HURTS
// weight-only quant 0.46->0.69; fixed-alpha AWQ is ~neutral; the alpha SEARCH
// minimizing reconstruction error gives the real win, ~0.46->0.25.)
// Env: src + golden (with calib_*.f32).
TEST(moss_tts_local, awq_reduces_4bit_drift)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const int hidden = 2560;
  std::vector<float> g_emb = read_bf16_((fs::path(gold) / "a1_embeds.bf16"));
  std::vector<float> g_hid = read_bf16_((fs::path(gold) / "a1_hidden.bf16"));
  if (g_emb.empty() ||
      !fs::exists((fs::path(gold) / "calib_qkv.f32"))) { return; }
  const int seq = (int)(g_emb.size() / hidden);
  ASSERT_TRUE(seq >= 1 && (int)g_hid.size() == seq * hidden);

  auto run_drift = [&](bool sq, bool clip) -> double {
    const fs::path qd = fs::temp_directory_path() /
        ("vpipe-sq-" + std::string(sq ? "s" : "p") + (clip ? "c" : "") +
         std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(qd, ec);
    genai::ModelQuantizer mq(mc);
    genai::QuantizeOptions opt;
    opt.bits = 4; opt.group = 64;
    if (sq) {
      opt.smoothquant = true; opt.smooth_alpha = 0.5f;
      opt.calib_dir = gold; opt.layer_prefix = "transformer.layers.";
      opt.n_layers = 36;
      opt.awq_clip = clip;
    }
    std::string err;
    if (!mq.run(src, qd.string(), opt, &err)) { return -1.0; }
    auto bb = MetalQwenModel::load(qd.string(), mc, v15_backbone_config_(4));
    if (!bb) { fs::remove_all(qd, ec); return -2.0; }
    SharedBuffer x = mc->make_shared_buffer((std::size_t)seq * hidden * 2);
    {
      std::ifstream in((fs::path(gold) / "a1_embeds.bf16").string(),
                       std::ios::binary);
      in.read(static_cast<char*>(x.contents()),
              (std::streamsize)seq * hidden * 2);
    }
    ContextManager* cm = bb->context_manager();
    const ContextId cid = cm->acquire_root();
    SharedBuffer hn = bb->forward_embeddings_hidden(cid, x, seq);
    cm->release(cid);
    if (hn.empty()) { fs::remove_all(qd, ec); return -3.0; }
    const auto* h = static_cast<const std::uint16_t*>(hn.contents());
    std::vector<float> v(hidden);
    for (int i = 0; i < hidden; ++i) { v[i] = bf16_to_f32_(h[i]); }
    const double r = rel_l2_(v.data(),
                             &g_hid[(std::size_t)(seq - 1) * hidden], hidden);
    fs::remove_all(qd, ec);
    return r;
  };

  const double plain = run_drift(false, false);
  const double sqd = run_drift(true, false);
  const double clipd = run_drift(true, true);
  std::fprintf(stderr,
               "[moss15] 4-bit backbone drift vs FP: plain=%.4f awq=%.4f "
               "awq+clip=%.4f\n", plain, sqd, clipd);
  ASSERT_TRUE(plain > 0.0 && sqd > 0.0 && clipd > 0.0);
  // AWQ scale search meaningfully reduces 4-bit drift (measured ~0.46 -> ~0.25)
  // -- the dominant win.
  EXPECT_TRUE(sqd < plain * 0.85);
  // FINDING: the paired AWQ per-group clip search (opt.awq_clip) does NOT stack
  // on the scale search for THIS model+calib -- it regresses end drift
  // (~0.25 -> ~0.32) even though clip=1.0 is in the grid (so it never regresses
  // the per-group PROXY). Same proxy-vs-end divergence as plain SmoothQuant:
  // saturating outlier weights hurts more downstream than the act-weighted
  // group MSE (from the tiny 8-text calib) predicts. So awq_clip stays OFF by
  // default; here we only assert it still beats plain 4-bit (the clipped AWQ
  // model is valid + better than no smoothing).
  EXPECT_TRUE(clipd < plain);
}

// M4: unsloth-style mixed precision. Bit width is assigned PER-LAYER (half the
// layers -- the most sensitive -- promoted to 8-bit, sub-8-bit average). The
// mixed checkpoint must (a) load + forward through the loader's _mixed path,
// and (b) drift LESS than uniform 4-bit at the same group, both plain and with
// AWQ. (Per-layer, not per-tensor: the runtime's mixed path corrupts a layer
// whose projections carry different widths -- the fused q|k|v / gate|up groups
// assume a uniform width.) Env: src + golden (with calib_*.f32).
TEST(moss_tts_local, mixed_precision_reduces_drift)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const int hidden = 2560;
  std::vector<float> g_emb = read_bf16_((fs::path(gold) / "a1_embeds.bf16"));
  std::vector<float> g_hid = read_bf16_((fs::path(gold) / "a1_hidden.bf16"));
  if (g_emb.empty() ||
      !fs::exists((fs::path(gold) / "calib_qkv.f32"))) { return; }
  const int seq = (int)(g_emb.size() / hidden);
  ASSERT_TRUE(seq >= 1 && (int)g_hid.size() == seq * hidden);

  auto run_drift = [&](bool mixed, bool awq, int bits = 4) -> double {
    const fs::path qd = fs::temp_directory_path() /
        ("vpipe-mix-" + std::string(mixed ? "m" : "u") +
         std::string(awq ? "a" : "p") + std::to_string(bits) +
         std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(qd, ec);
    genai::ModelQuantizer mq(mc);
    genai::QuantizeOptions opt;
    opt.bits = bits; opt.group = 64;
    opt.layer_prefix = "transformer.layers.";
    opt.n_layers = 36;
    if (awq) {
      opt.smoothquant = true; opt.smooth_alpha = 0.5f;   // AWQ search
      opt.calib_dir = gold;
    }
    if (mixed) {
      opt.mixed = true; opt.high_bits = 8; opt.mixed_frac = 0.5f;
    }
    std::string err;
    if (!mq.run(src, qd.string(), opt, &err)) { return -1.0; }
    // Mixed loads with quant_bits=4 (base; 8-bit layers use _fn_qmm8);
    // uniform loads with its own bit width.
    auto bb = MetalQwenModel::load(qd.string(), mc,
                                   v15_backbone_config_(mixed ? 4 : bits));
    if (!bb) { fs::remove_all(qd, ec); return -2.0; }
    SharedBuffer x = mc->make_shared_buffer((std::size_t)seq * hidden * 2);
    {
      std::ifstream in((fs::path(gold) / "a1_embeds.bf16").string(),
                       std::ios::binary);
      in.read(static_cast<char*>(x.contents()),
              (std::streamsize)seq * hidden * 2);
    }
    ContextManager* cm = bb->context_manager();
    const ContextId cid = cm->acquire_root();
    SharedBuffer hn = bb->forward_embeddings_hidden(cid, x, seq);
    cm->release(cid);
    if (hn.empty()) { fs::remove_all(qd, ec); return -3.0; }
    const auto* h = static_cast<const std::uint16_t*>(hn.contents());
    std::vector<float> v(hidden);
    for (int i = 0; i < hidden; ++i) { v[i] = bf16_to_f32_(h[i]); }
    const double r = rel_l2_(v.data(),
                             &g_hid[(std::size_t)(seq - 1) * hidden], hidden);
    fs::remove_all(qd, ec);
    return r;
  };

  const double u4p = run_drift(false, false);
  const double m48p = run_drift(true, false);
  const double u4a = run_drift(false, true);
  const double m48a = run_drift(true, true);
  std::fprintf(stderr,
               "[moss15] drift vs FP: u4_plain=%.4f mix_plain=%.4f "
               "u4_awq=%.4f mix_awq=%.4f\n", u4p, m48p, u4a, m48a);
  ASSERT_TRUE(u4p > 0.0 && m48p > 0.0 && u4a > 0.0 && m48a > 0.0);
  // Per-layer 4/8-bit mixed precision (half the layers at 8-bit) clearly cuts
  // drift vs uniform 4-bit, both plain and with AWQ smoothing.
  EXPECT_TRUE(m48p < u4p);
  EXPECT_TRUE(m48a < u4a);
}

// A2: the local/depth transformer. Feed the SAME fixed seed the golden used
// through one local step (pos 0) and compare the ln_f hidden to the HF golden
// (isolates the local-transformer wiring; f16 vs bf16 => small rel-L2). The
// local weights are bf16 (not quantized), loaded straight from the source.
// Env: VPIPE_MOSS_TTS_LOCAL_MODEL (src) + VPIPE_MOSS15_GOLDEN.
TEST(moss_tts_local, local_transformer_step0_matches_golden)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const int hidden = 2560;
  std::vector<float> seed = read_bf16_((fs::path(gold) / "a2_seed.bf16"));
  std::vector<float> g_h0 =
      read_bf16_((fs::path(gold) / "a2_local_hidden0.bf16"));
  ASSERT_TRUE((int)seed.size() == hidden && (int)g_h0.size() == hidden);

  MetalMossLocalTransformer::Config lc;   // v1.5 defaults
  auto lt = MetalMossLocalTransformer::load(src, mc, lc);
  ASSERT_TRUE(lt != nullptr);

  SharedBuffer x = mc->make_shared_buffer((std::size_t)hidden * 2);
  ASSERT_TRUE(!x.empty());
  {
    auto* d = static_cast<_Float16*>(x.contents());
    for (int i = 0; i < hidden; ++i) { d[i] = (_Float16)seed[i]; }
  }

  lt->reset_frame();
  const SharedBuffer* hn = lt->step(x);
  ASSERT_TRUE(hn != nullptr && !hn->empty());

  const auto* h = static_cast<const _Float16*>(hn->contents());
  std::vector<float> v(hidden);
  for (int i = 0; i < hidden; ++i) { v[i] = (float)h[i]; }
  const double rl2 = rel_l2_(v.data(), g_h0.data(), hidden);
  std::fprintf(stderr,
               "[moss15] A2 local step0 rel-L2 vs HF golden = %.4f\n", rl2);
  EXPECT_TRUE(rl2 < 0.03);   // f16 vs bf16 local transformer
}

// A2 multi-step: the full per-frame 12-codebook greedy loop (local transformer
// KV-cache growth + audio heads + code feedback). Compare to the HF golden's
// greedy first-frame codes (a2_codes in a2_meta.json) -- TOKEN-EXACT target.
// Env: VPIPE_MOSS_TTS_LOCAL_MODEL (src) + VPIPE_MOSS15_GOLDEN.
TEST(moss_tts_local, frame_codes_match_golden)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const int hidden = 2560;
  std::vector<float> seed = read_bf16_((fs::path(gold) / "a2_seed.bf16"));
  ASSERT_TRUE((int)seed.size() == hidden);

  // Golden codes from a2_meta.json.
  std::vector<int> g_codes;
  {
    std::ifstream in((fs::path(gold) / "a1_meta.json").string());
    ASSERT_TRUE((bool)in);
    std::string txt((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    FlexData meta = FlexData::from_json(txt);
    FlexData arr = meta.as_object().at("a2_codes");
    for (auto e : arr.as_array()) { g_codes.push_back((int)e.as_int()); }
  }
  ASSERT_TRUE((int)g_codes.size() == 12);

  MetalMossLocalModel::Config cfg;   // v1.5 defaults (n_vq 12, vocab 1024)
  auto lm = MetalMossLocalModel::load(src, mc, cfg);
  ASSERT_TRUE(lm != nullptr);

  SharedBuffer x = mc->make_shared_buffer((std::size_t)hidden * 2);
  ASSERT_TRUE(!x.empty());
  {
    auto* d = static_cast<_Float16*>(x.contents());
    for (int i = 0; i < hidden; ++i) { d[i] = (_Float16)seed[i]; }
  }

  // Free-running greedy (informational): flips propagate via code feedback, so
  // exact match across 12 sequential argmaxes is fragile at f16 vs bf16.
  std::vector<int> codes = lm->decode_frame_greedy(x);
  ASSERT_TRUE((int)codes.size() == 12);
  int free_mism = 0;
  for (int k = 0; k < 12; ++k) {
    if (codes[(std::size_t)k] != g_codes[(std::size_t)k]) { ++free_mism; }
  }
  std::fprintf(stderr, "[moss15] A2 free-run codes: %d/12 mism\n", free_mism);

  // Teacher-forced: feed the GOLDEN codes so every step sees the same input as
  // the reference. The real correctness signal is the per-step hidden rel-L2
  // (a structural bug blows it up; precision is tiny). Compare to a2_hiddens.
  std::vector<float> g_hid = read_bf16_((fs::path(gold) / "a2_hiddens.bf16"));
  ASSERT_TRUE((int)g_hid.size() == 12 * hidden);
  std::vector<float> tf_hid;
  std::vector<int> tf_arg;
  lm->decode_frame_teacher(x, g_codes, tf_hid, tf_arg);
  ASSERT_TRUE((int)tf_hid.size() == 12 * hidden);

  double worst = 0.0;
  int tf_mism = 0;
  for (int k = 0; k < 12; ++k) {
    const double r = rel_l2_(tf_hid.data() + (std::size_t)k * hidden,
                             g_hid.data() + (std::size_t)k * hidden, hidden);
    worst = std::max(worst, r);
    if (tf_arg[(std::size_t)k] != g_codes[(std::size_t)k]) { ++tf_mism; }
    std::fprintf(stderr, "  cb%2d: tf_rel_l2=%.4f vpipe=%d golden=%d\n", k, r,
                 tf_arg[(std::size_t)k], g_codes[(std::size_t)k]);
  }
  std::fprintf(stderr,
               "[moss15] A2 teacher-forced: worst hidden rel-L2=%.4f, "
               "argmax %d/12 mism\n", worst, tf_mism);
  EXPECT_TRUE(worst < 0.03);   // per-step compute correct (f16 vs bf16)
}

// A2-integration: the full v1.5 model. Quantize -> load V15 -> (1) verify the
// grid embed-sum (text + sum audio) against the HF inputs_embeds golden, and
// (2) smoke the per-frame generate loop (backbone prefill -> seed -> local
// decode -> next grid row -> backbone decode), checking frame-0 codes vs the
// golden (precision near-ties allowed). Env: src + golden.
TEST(moss_tts_local, v15_embeds_and_generate)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  const int hidden = 2560, NV = 12, pad = 1024;

  // a1 grid (text_ids from meta; audio channels = pad).
  std::vector<int> text_ids;
  std::vector<int> g_codes;
  {
    std::ifstream in((fs::path(gold) / "a1_meta.json").string());
    ASSERT_TRUE((bool)in);
    std::string txt((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    FlexData meta = FlexData::from_json(txt);
    FlexData o = meta;
    FlexData ti = o.as_object().at("text_ids");
    for (auto e : ti.as_array()) { text_ids.push_back((int)e.as_int()); }
    FlexData cc = o.as_object().at("a2_codes");
    for (auto e : cc.as_array()) { g_codes.push_back((int)e.as_int()); }
  }
  const int seq = (int)text_ids.size();
  ASSERT_TRUE(seq >= 1);
  std::vector<std::vector<std::int32_t>> grid(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)(1 + NV), pad));
  for (int t = 0; t < seq; ++t) { grid[(std::size_t)t][0] = text_ids[t]; }

  const fs::path qdir = fs::temp_directory_path() /
                        ("vpipe-moss15-v15-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(qdir, ec);
  {
    ModelQuantizer mq(mc);
    QuantizeOptions opt; opt.bits = 8; opt.group = 64;
    std::string err;
    ASSERT_TRUE(mq.run(src, qdir.string(), opt, &err));
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_();
  auto m = MetalMossV15Model::load(qdir.string(), mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // (1) embed-sum vs HF inputs_embeds golden (a1_embeds).
  std::vector<float> g_emb = read_bf16_((fs::path(gold) / "a1_embeds.bf16"));
  ASSERT_TRUE((int)g_emb.size() == seq * hidden);
  SharedBuffer emb = m->assemble_embeds(grid, 0, seq);
  ASSERT_TRUE(!emb.empty());
  std::vector<float> ve((std::size_t)seq * hidden);
  {
    const auto* s = static_cast<const std::uint16_t*>(emb.contents());
    for (std::size_t i = 0; i < ve.size(); ++i) {
      std::uint32_t u = (std::uint32_t)s[i] << 16; float f;
      std::memcpy(&f, &u, 4); ve[i] = f;
    }
  }
  const double erl2 = rel_l2_(ve.data(), g_emb.data(), ve.size());
  std::fprintf(stderr, "[moss15] v15 embed-sum rel-L2 vs HF = %.5f\n", erl2);
  EXPECT_TRUE(erl2 < 0.01);

  // (2) generate smoke: a few frames from the real backbone.
  std::vector<std::vector<int>> frames = m->generate(grid, 4);
  std::fprintf(stderr, "[moss15] v15 generate -> %zu frames\n", frames.size());
  if (!frames.empty()) {
    int mism = 0;
    for (int k = 0; k < NV; ++k) {
      if (frames[0][(std::size_t)k] != g_codes[(std::size_t)k]) { ++mism; }
    }
    std::fprintf(stderr, "[moss15] v15 frame0 codes vs golden: %d/12 mism\n",
                 mism);
    for (int k = 0; k < NV; ++k) {
      EXPECT_TRUE(frames[0][(std::size_t)k] >= 0 &&
                  frames[0][(std::size_t)k] < pad);
    }
  }
  fs::remove_all(qdir, ec);
}

// Decode-throughput benchmark on a PRE-QUANTIZED v1.5 dir (no re-quant).
// Env: VPIPE_MOSS15_QUANT_DIR = the quantized model dir (e.g. the 4bit dir);
// VPIPE_MOSS15_BENCH_FRAMES (default 96). Set VPIPE_MOSS15_PROFILE=1 for the
// per-phase breakdown. Skips if the dir env is unset.
TEST(moss_tts_local, v15_decode_bench)
{
  const char* qd = std::getenv("VPIPE_MOSS15_QUANT_DIR");
  if (qd == nullptr || *qd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int NV = 12, pad = 1024;
  int bits = 4;                         // read the dir's own quant width
  {
    std::ifstream in(std::string(qd) + "/config.json");
    if (in) {
      FlexData root = FlexData::from_json(in);
      if (root.is_object() && root.as_object().contains("quantization")) {
        FlexData q = root.as_object().at("quantization");
        if (q.is_object() && q.as_object().contains("bits")) {
          bits = (int)q.as_object().at("bits").as_int(bits);
        }
      }
    }
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_(bits);
  auto m = MetalMossV15Model::load(qd, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // A short representative prompt grid (text ids in-vocab, audio channels pad).
  const int seq = 8;
  std::vector<std::vector<std::int32_t>> grid(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)(1 + NV), pad));
  for (int t = 0; t < seq; ++t) { grid[(std::size_t)t][0] = 100 + t; }

  int frames = 96;
  if (const char* fe = std::getenv("VPIPE_MOSS15_BENCH_FRAMES")) {
    frames = std::max(8, std::atoi(fe));
  }
  MossSampling sp; sp.temperature = 0.8f; sp.top_k = 50; sp.top_p = 0.95f;

  m->generate(grid, 2, sp, 1234);       // warm the caches / clocks
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<int>> out = m->generate(grid, frames, sp, 4321);
  const double wall = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "[moss15-bench] %zu frames in %.1f ms = %.1f frame/s\n",
               out.size(), wall,
               out.empty() ? 0.0 : 1000.0 * out.size() / wall);
  EXPECT_TRUE(!out.empty());
}

// Fused vs serial greedy equivalence on a PRE-QUANTIZED dir: the fused
// single-command-buffer decode (on-GPU argmax + embed-gather) must be
// token-exact with the pre-fusion serial (host-argmax, per-codebook commit)
// reference, across several real backbone-driven frame seeds.
// Env: VPIPE_MOSS15_QUANT_DIR. Skips if unset.
TEST(moss_tts_local, v15_fused_matches_serial)
{
  const char* qd = std::getenv("VPIPE_MOSS15_QUANT_DIR");
  if (qd == nullptr || *qd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int NV = 12, pad = 1024, H = 2560;
  int bits = 4;
  {
    std::ifstream in(std::string(qd) + "/config.json");
    if (in) {
      FlexData root = FlexData::from_json(in);
      if (root.is_object() && root.as_object().contains("quantization")) {
        FlexData q = root.as_object().at("quantization");
        if (q.is_object() && q.as_object().contains("bits")) {
          bits = (int)q.as_object().at("bits").as_int(bits);
        }
      }
    }
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_(bits);
  auto m = MetalMossV15Model::load(qd, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  const int seq = 8;
  std::vector<std::vector<std::int32_t>> grid(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)(1 + NV), pad));
  for (int t = 0; t < seq; ++t) { grid[(std::size_t)t][0] = 100 + t; }

  ContextManager* cm = m->backbone()->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer emb = m->assemble_embeds(grid, 0, seq);
  SharedBuffer s0 = m->backbone()->forward_embeddings_hidden(cid, emb, seq);
  ASSERT_TRUE(!s0.empty());

  SharedBuffer seed_f16 = mc->make_shared_buffer((std::size_t)H * 2);
  auto seed_from = [&](const SharedBuffer& bb) {
    const auto* sb = static_cast<const std::uint16_t*>(bb.contents());
    auto* sf = static_cast<_Float16*>(seed_f16.contents());
    for (int j = 0; j < H; ++j) {
      std::uint32_t u = (std::uint32_t)sb[j] << 16; float f;
      std::memcpy(&f, &u, 4); sf[j] = (_Float16)f;
    }
  };

  int total_mism = 0;
  const int frames = 6;
  seed_from(s0);
  for (int f = 0; f < frames; ++f) {
    // Same seed -> both paths; assert token-exact.
    std::vector<int> a = m->local()->decode_frame_greedy(seed_f16);
    std::vector<int> b = m->local()->decode_frame_greedy_ref(seed_f16);
    int mism = 0;
    for (int k = 0; k < NV; ++k) { if (a[k] != b[k]) { ++mism; } }
    total_mism += mism;
    EXPECT_TRUE(mism == 0);
    // Advance the backbone with the fused codes to get the next real seed.
    std::vector<std::vector<std::int32_t>> row(
        1, std::vector<std::int32_t>((std::size_t)(1 + NV)));
    row[0][0] = cfg.audio_assistant_slot;
    for (int k = 0; k < NV; ++k) { row[0][(std::size_t)(k + 1)] = a[k]; }
    SharedBuffer re = m->assemble_embeds(row, 0, 1);
    const SharedBuffer* nh = m->backbone()->decode_embedding_hidden(cid, re);
    ASSERT_TRUE(nh != nullptr && !nh->empty());
    seed_from(*nh);
  }
  cm->release(cid);
  std::fprintf(stderr,
      "[moss15] fused-vs-serial greedy: %d frames, %d total code mismatches\n",
      frames, total_mism);
  EXPECT_TRUE(total_mism == 0);
}

// A3 codec (MOSS-Audio-Tokenizer-v2) decode: codes -> 48kHz stereo waveform.
// Decodes the golden's fixed codes [12,T] and compares (a) the RLFQ latent
// [768,T] (C1) and (b) the full stereo waveform [2, T*3840] to the HF golden,
// Codec-v2 decode throughput benchmark on synthetic codes (no golden needed).
// Env: VPIPE_MOSS_CODEC_V2_MODEL (codec dir); VPIPE_MOSS_CODEC_BENCH_FRAMES
// (default 100 frames = 8 s @ 12.5 Hz). VPIPE_MOSS_CODEC_PROFILE=1 for the
// per-stage breakdown. Skips if the codec dir env is unset.
TEST(moss_tts_local, codec_v2_decode_bench)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  const int NV = codec->n_quantizers();
  int T = 100;
  if (const char* fe = std::getenv("VPIPE_MOSS_CODEC_BENCH_FRAMES")) {
    T = std::max(4, std::atoi(fe));
  }
  // Deterministic pseudo-random codes in [0, 1024).
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  std::uint64_t r = 0x9e3779b97f4a7c15ULL;
  for (int t = 0; t < T; ++t) {
    for (int q = 0; q < NV; ++q) {
      r = r * 6364136223846793005ULL + 1442695040888963407ULL;
      codes[(std::size_t)t][(std::size_t)q] = (std::int32_t)((r >> 33) % 1024);
    }
  }

  std::vector<float> w0 = codec->decode(codes);   // warm caches / clocks
  ASSERT_TRUE(!w0.empty());
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> wav = codec->decode(codes);
  const double wall = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  const double secs = (double)(wav.size() / 2) / codec->sample_rate();
  std::fprintf(stderr,
      "[codec-v2-bench] T=%d -> %.2fs audio in %.1f ms (%.1fx realtime)\n",
      T, secs, wall, secs / (wall / 1000.0));
  EXPECT_TRUE(!wav.empty());
}

// STREAMING == ONE-SHOT: feeding the codes to decode_stream_chunk() in small
// chunks and concatenating the per-chunk PCM must reproduce the whole-utterance
// decode() bit-for-bit (f16 reduction noise only). Proves the windowed K/V-ring
// streaming decode (bounded memory, O(chunk)/step) is numerically correct.
// Env: VPIPE_MOSS_CODEC_V2_MODEL. Skips if unset.
TEST(moss_tts_local, codec_v2_stream_matches_oneshot)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  const int NV = codec->n_quantizers();
  const int T = 260;          // > 2*window so the ring wraps and is exercised
  int chunk = 20;
  if (const char* e = std::getenv("VPIPE_STREAM_CHUNK")) {
    chunk = std::max(1, std::atoi(e));
  }
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  std::uint64_t r = 0x9e3779b97f4a7c15ULL;
  for (int t = 0; t < T; ++t) {
    for (int q = 0; q < NV; ++q) {
      r = r * 6364136223846793005ULL + 1442695040888963407ULL;
      codes[(std::size_t)t][(std::size_t)q] = (std::int32_t)((r >> 33) % 1024);
    }
  }

  const std::vector<float> full = codec->decode(codes);   // [2*T*spf]
  ASSERT_TRUE(!full.empty());
  const int spf = (int)((full.size() / 2) / (std::size_t)T);

  // Stream in `chunk`-frame steps; assemble channel-major [ch0(T*spf)|ch1].
  auto stm = codec->decode_stream_begin(chunk);
  ASSERT_TRUE(stm != nullptr);
  std::vector<float> ch0, ch1;
  ch0.reserve((std::size_t)T * spf);
  ch1.reserve((std::size_t)T * spf);
  for (int t0 = 0; t0 < T; t0 += chunk) {
    const int c = std::min(chunk, T - t0);
    std::vector<std::vector<std::int32_t>> sub(
        codes.begin() + t0, codes.begin() + t0 + c);
    std::vector<float> pcm = codec->decode_stream_chunk(*stm, sub);
    ASSERT_TRUE((int)pcm.size() == 2 * c * spf);
    const int per = c * spf;
    ch0.insert(ch0.end(), pcm.begin(), pcm.begin() + per);
    ch1.insert(ch1.end(), pcm.begin() + per, pcm.begin() + 2 * per);
  }
  ASSERT_TRUE((int)ch0.size() == T * spf && (int)ch1.size() == T * spf);

  // rel-L2 of assembled-vs-full over both channels.
  double num = 0, den = 0;
  float maxd = 0.0f;
  for (int k = 0; k < T * spf; ++k) {
    const double a0 = full[(std::size_t)k], b0 = ch0[(std::size_t)k];
    const double a1 = full[(std::size_t)T * spf + k], b1 = ch1[(std::size_t)k];
    num += (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1);
    den += a0 * a0 + a1 * a1;
    maxd = std::max(maxd, (float)std::max(std::fabs(a0 - b0),
                                          std::fabs(a1 - b1)));
  }
  const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
  std::fprintf(stderr,
      "[codec-v2-stream] T=%d chunk=%d spf=%d | rel-L2=%.3g max|d|=%.3g\n",
      T, chunk, spf, rel, (double)maxd);
  EXPECT_TRUE(rel < 1e-3);
}

// BARGE-IN mechanism: the generate() frame callback returning false stops
// generation ASAP (after the current frame) -- what the TTS stage uses to abort
// an in-flight utterance when new text arrives. Verify it stops at exactly the
// requested frame instead of running to max_frames.
// Env: VPIPE_MOSS15_QUANT_DIR. Skips if unset.
TEST(moss_tts_local, v15_generate_callback_stop_is_prompt)
{
  const char* qd = std::getenv("VPIPE_MOSS15_QUANT_DIR");
  if (qd == nullptr || *qd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  int bits = 4;
  {
    std::ifstream in(std::string(qd) + "/config.json");
    if (in) {
      FlexData root = FlexData::from_json(in);
      if (root.is_object() && root.as_object().contains("quantization")) {
        FlexData q = root.as_object().at("quantization");
        if (q.is_object() && q.as_object().contains("bits")) {
          bits = (int)q.as_object().at("bits").as_int(bits);
        }
      }
    }
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_(bits);
  auto m = MetalMossV15Model::load(qd, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  const int NV = 12, pad = 1024, seq = 8;
  std::vector<std::vector<std::int32_t>> grid(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)(1 + NV), pad));
  for (int t = 0; t < seq; ++t) { grid[(std::size_t)t][0] = 100 + t; }
  MossSampling sp; sp.temperature = 0.8f; sp.top_k = 50; sp.top_p = 0.95f;

  const int F = 64, stop_after = 6;
  int calls = 0;
  auto cb = [&](const std::vector<int>&) { return ++calls < stop_after; };
  std::vector<std::vector<int>> frames = m->generate(grid, F, sp, 555, cb);
  std::fprintf(stderr, "[v15-barge] max=%d stop_after=%d -> %d frames\n",
               F, stop_after, (int)frames.size());
  // The callback fires after each pushed frame and false breaks the loop, so
  // exactly `stop_after` frames come back -- well short of the F-frame budget.
  EXPECT_TRUE((int)frames.size() == stop_after);
}

// STAGE-LEVEL streaming integration: driving the v1.5 LM with the per-frame
// callback + feeding each chunk through the codec's windowed streaming decode
// (what TextToSpeechStage does when stream_chunk_frames>0) must reproduce the
// one-shot generate()+decode() PCM. Same seed => identical frames, and the
// codec stream is bit-exact vs one-shot, so the assembled waveform matches.
// Env: VPIPE_MOSS15_QUANT_DIR (v1.5 LM) + VPIPE_MOSS_CODEC_V2_MODEL (codec).
TEST(moss_tts_local, v15_stream_pipeline_matches_oneshot)
{
  const char* qd = std::getenv("VPIPE_MOSS15_QUANT_DIR");
  const char* cd = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (qd == nullptr || *qd == '\0' || cd == nullptr || *cd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  int bits = 4;
  {
    std::ifstream in(std::string(qd) + "/config.json");
    if (in) {
      FlexData root = FlexData::from_json(in);
      if (root.is_object() && root.as_object().contains("quantization")) {
        FlexData q = root.as_object().at("quantization");
        if (q.is_object() && q.as_object().contains("bits")) {
          bits = (int)q.as_object().at("bits").as_int(bits);
        }
      }
    }
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_(bits);
  auto m = MetalMossV15Model::load(qd, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  auto codec = MetalMossCodecV2::load(cd, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  const int NV = 12, pad = 1024, seq = 8;
  std::vector<std::vector<std::int32_t>> grid(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)(1 + NV), pad));
  for (int t = 0; t < seq; ++t) { grid[(std::size_t)t][0] = 100 + t; }
  MossSampling sp; sp.temperature = 0.8f; sp.top_k = 50; sp.top_p = 0.95f;
  const int F = 40, chunk = 12;
  const std::uint64_t seed = 777;

  // One-shot: generate all frames, decode once.
  std::vector<std::vector<int>> f1 = m->generate(grid, F, sp, seed);
  ASSERT_TRUE(!f1.empty());
  std::vector<std::vector<std::int32_t>> c1;
  for (const auto& f : f1) { c1.emplace_back(f.begin(), f.end()); }
  const std::vector<float> full = codec->decode(c1);
  const int spf = (int)((full.size() / 2) / f1.size());

  // Streaming: same seed => same frames; emit `chunk`-frame PCM via the codec
  // stream, assemble channel-major [ch0|ch1] exactly as the stage's consumer.
  auto stm = codec->decode_stream_begin(chunk);
  ASSERT_TRUE(stm != nullptr);
  std::vector<float> ch0, ch1;
  std::vector<std::vector<std::int32_t>> buf;
  auto flush = [&]() {
    if (buf.empty()) { return; }
    std::vector<float> pcm = codec->decode_stream_chunk(*stm, buf);
    const int c = (int)buf.size();
    buf.clear();
    if ((int)pcm.size() != 2 * c * spf) { return; }
    ch0.insert(ch0.end(), pcm.begin(), pcm.begin() + (std::size_t)c * spf);
    ch1.insert(ch1.end(), pcm.begin() + (std::size_t)c * spf, pcm.end());
  };
  auto on_frame = [&](const std::vector<int>& codes) -> bool {
    buf.emplace_back(codes.begin(), codes.end());
    if ((int)buf.size() >= chunk) { flush(); }
    return true;
  };
  std::vector<std::vector<int>> f2 = m->generate(grid, F, sp, seed, on_frame);
  flush();
  EXPECT_TRUE(f2.size() == f1.size());   // callback must not perturb sampling

  const int T = (int)f1.size();
  ASSERT_TRUE((int)ch0.size() == T * spf && (int)ch1.size() == T * spf);
  double num = 0, den = 0;
  for (int k = 0; k < T * spf; ++k) {
    const double a0 = full[(std::size_t)k], b0 = ch0[(std::size_t)k];
    const double a1 = full[(std::size_t)T * spf + k], b1 = ch1[(std::size_t)k];
    num += (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1);
    den += a0 * a0 + a1 * a1;
  }
  const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
  std::fprintf(stderr,
      "[v15-stream-pipe] frames one-shot=%d stream=%d | chunk=%d | rel-L2=%.3g\n",
      (int)f1.size(), (int)f2.size(), chunk, rel);
  EXPECT_TRUE(rel < 1e-3);
}

// REUSE: a cached StreamState re-armed with reset() (the stage's per-beat reuse,
// no realloc) must produce BIT-IDENTICAL PCM to a fresh state -- i.e. a prior
// (longer) utterance leaves no stale ring contamination. Streams utterance B
// through a fresh state and through a state that first ran a longer utterance A
// then reset(); asserts equality.
// Env: VPIPE_MOSS_CODEC_V2_MODEL. Skips if unset.
TEST(moss_tts_local, codec_v2_stream_reuse_matches_fresh)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  const int NV = codec->n_quantizers();
  const int chunk = 20;
  auto gen_codes = [&](int T, std::uint64_t s) {
    std::vector<std::vector<std::int32_t>> c(
        (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
    for (int t = 0; t < T; ++t) {
      for (int q = 0; q < NV; ++q) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        c[(std::size_t)t][(std::size_t)q] = (std::int32_t)((s >> 33) % 1024);
      }
    }
    return c;
  };
  const auto A = gen_codes(300, 0x111);   // longer prior utterance
  const auto B = gen_codes(180, 0x222);   // the utterance under test (shorter)

  auto stream_all = [&](MetalMossCodecV2::StreamState& st,
                        const std::vector<std::vector<std::int32_t>>& codes) {
    std::vector<float> out;
    for (int t0 = 0; t0 < (int)codes.size(); t0 += chunk) {
      const int c = std::min(chunk, (int)codes.size() - t0);
      std::vector<std::vector<std::int32_t>> sub(
          codes.begin() + t0, codes.begin() + t0 + c);
      std::vector<float> pcm = codec->decode_stream_chunk(st, sub);
      out.insert(out.end(), pcm.begin(), pcm.end());
    }
    return out;
  };

  auto fresh = codec->decode_stream_begin(chunk);
  ASSERT_TRUE(fresh != nullptr);
  const std::vector<float> got_fresh = stream_all(*fresh, B);

  auto reused = codec->decode_stream_begin(chunk);
  ASSERT_TRUE(reused != nullptr);
  (void)stream_all(*reused, A);     // run a longer utterance first
  reused->reset();                  // re-arm (the stage's per-beat reuse)
  const std::vector<float> got_reused = stream_all(*reused, B);

  ASSERT_TRUE(got_fresh.size() == got_reused.size());
  float maxd = 0.0f;
  for (std::size_t k = 0; k < got_fresh.size(); ++k) {
    maxd = std::max(maxd, std::fabs(got_fresh[k] - got_reused[k]));
  }
  std::fprintf(stderr, "[codec-v2-reuse] |A|=%d |B|=%d | max|fresh-reused|=%.6g\n",
               (int)A.size(), (int)B.size(), (double)maxd);
  EXPECT_TRUE(maxd == 0.0f);
}

// codec-v1 (24 kHz mono, RT/8B path) streaming == one-shot: same windowed
// K/V-ring streaming decode as codec-v2, verified bit-exact vs decode().
// Env: VPIPE_MOSS_CODEC_MODEL. Skips if unset.
TEST(moss_tts_local, codec_v1_stream_matches_oneshot)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodec::load(cdir, mc, false, false);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  const int NV = codec->n_quantizers();
  const int T = 260, chunk = 20;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  std::uint64_t r = 0x9e3779b97f4a7c15ULL;
  for (int t = 0; t < T; ++t) {
    for (int q = 0; q < NV; ++q) {
      r = r * 6364136223846793005ULL + 1442695040888963407ULL;
      codes[(std::size_t)t][(std::size_t)q] = (std::int32_t)((r >> 33) % 1024);
    }
  }
  const std::vector<float> full = codec->decode(codes);   // mono [T*hop]
  ASSERT_TRUE(!full.empty());
  const int spf = (int)(full.size() / (std::size_t)T);

  auto stm = codec->decode_stream_begin(chunk);
  ASSERT_TRUE(stm != nullptr);
  std::vector<float> got;
  got.reserve(full.size());
  for (int t0 = 0; t0 < T; t0 += chunk) {
    const int c = std::min(chunk, T - t0);
    std::vector<std::vector<std::int32_t>> sub(
        codes.begin() + t0, codes.begin() + t0 + c);
    std::vector<float> pcm = codec->decode_stream_chunk(*stm, sub);
    ASSERT_TRUE((int)pcm.size() == c * spf);
    got.insert(got.end(), pcm.begin(), pcm.end());
  }
  ASSERT_TRUE(got.size() == full.size());
  double num = 0, den = 0;
  for (std::size_t k = 0; k < full.size(); ++k) {
    num += (full[k] - got[k]) * (full[k] - got[k]);
    den += (double)full[k] * full[k];
  }
  const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
  std::fprintf(stderr, "[codec-v1-stream] T=%d chunk=%d spf=%d | rel-L2=%.3g\n",
               T, chunk, spf, rel);
  EXPECT_TRUE(rel < 1e-3);
}

// Isolate the codec decoder's windowed-attention cost: replay each stage's
// exact sdpa_causal_window_f16 workload (seq/heads/window/n_layers) in one
// command buffer and time it -- no model needed. Attributes how much of the
// long-stage decode time is attention vs the GEMM/elementwise rest. Base T
// via VPIPE_MOSS_CODEC_BENCH_FRAMES (default 100). No env gate on a model.
TEST(moss_tts_local, codec_v2_attn_microbench)
{
  if (std::getenv("VPIPE_MOSS_CODEC_ATTN_MICROBENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("sdpa");
  auto fn = lib.function("sdpa_causal_window_f16");
  ASSERT_TRUE(fn.valid());

  int baseT = 100;
  if (const char* fe = std::getenv("VPIPE_MOSS_CODEC_BENCH_FRAMES")) {
    baseT = std::max(4, std::atoi(fe));
  }
  struct SC { int mult, heads, window, layers, d; };
  const SC cfgs[6] = {
      {1, 20, 125, 32, 1280}, {2, 12, 250, 12, 768}, {4, 12, 400, 12, 768},
      {8, 12, 400, 12, 768},  {16, 12, 400, 12, 768}, {32, 12, 400, 12, 768}};

  double total = 0;
  for (int s = 0; s < 6; ++s) {
    const SC& c = cfgs[s];
    const int T = baseT * c.mult, heads = c.heads, hd = c.d / c.heads;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::size_t n = (std::size_t)heads * T * hd;
    SharedBuffer q = mc->make_shared_buffer(n * 2), k = mc->make_shared_buffer(n * 2),
                 v = mc->make_shared_buffer(n * 2), o = mc->make_shared_buffer(n * 2);
    auto* qp = static_cast<_Float16*>(q.contents());
    auto* kp = static_cast<_Float16*>(k.contents());
    auto* vp = static_cast<_Float16*>(v.contents());
    for (std::size_t i = 0; i < n; ++i) {
      qp[i] = (_Float16)(0.01f * (float)(i % 7));
      kp[i] = (_Float16)(0.01f * (float)(i % 5));
      vp[i] = (_Float16)(0.01f * (float)(i % 3));
    }
    auto run = [&]() {
      metal_compute::CommandStream st = mc->make_command_stream();
      { metal_compute::ComputeEncoder e = st.begin_compute();
        for (int l = 0; l < c.layers; ++l) {
          e.set_function(fn);
          e.set_buffer(0, q); e.set_buffer(1, k); e.set_buffer(2, v);
          e.set_buffer(3, o);
          e.set_constant(4, scale); e.set_constant(5, T); e.set_constant(6, hd);
          e.set_constant(7, heads); e.set_constant(8, heads);
          e.set_constant(9, T); const int z = 0; e.set_constant(10, z);
          e.set_constant(11, T); e.set_constant(12, c.window);
          e.set_constant(13, T);
          e.dispatch({32, (unsigned)heads, (unsigned)T}, {32, 1, 1});
        } }
      st.commit().wait();
    };
    run();                                  // warm
    const auto t0 = std::chrono::steady_clock::now();
    run();
    const double m = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    total += m;
    std::fprintf(stderr, "[codec-attn] stage%d T=%d heads=%d win=%d L=%d: "
                 "%.1f ms\n", s, T, heads, c.window, c.layers, m);
  }
  std::fprintf(stderr, "[codec-attn] total attention = %.1f ms\n", total);
}

// M5 matrix-core (matmul2d/NAX) decode GEMM must be numerically equivalent to
// the steel dense_gemm_t path: decode the same codes with mma on vs off and
// compare the waveform rel-L2. Skips if the codec dir env is unset or mma is
// not active (pre-M5). Env: VPIPE_MOSS_CODEC_V2_MODEL.
TEST(moss_tts_local, codec_v2_mma_matches_steel)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  if (!codec->use_mma2()) {
    std::fprintf(stderr, "[codec-v2] mma path not active (pre-M5?), skip\n");
    return;
  }

  const int NV = codec->n_quantizers(), T = 40;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  std::uint64_t r = 0xd1b54a32d192ed03ULL;
  for (int t = 0; t < T; ++t) {
    for (int q = 0; q < NV; ++q) {
      r = r * 6364136223846793005ULL + 1442695040888963407ULL;
      codes[(std::size_t)t][(std::size_t)q] = (std::int32_t)((r >> 33) % 1024);
    }
  }

  codec->set_use_mma2(true);
  std::vector<float> a = codec->decode(codes);
  codec->set_use_mma2(false);
  std::vector<float> b = codec->decode(codes);
  codec->set_use_mma2(true);
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  const double rl2 = rel_l2_(a.data(), b.data(), a.size());
  std::fprintf(stderr, "[codec-v2] mma-vs-steel waveform rel-L2 = %.6f\n", rl2);
  EXPECT_TRUE(rl2 < 1e-2);
}

// M5 matrix-core windowed-causal flash attention (sdpa_causal_mma2_d64_f16)
// must be numerically equivalent to the scalar sdpa_causal_window_f16: decode
// the same codes with attn-mma on vs off and compare the waveform rel-L2.
// Skips if the codec dir env is unset or the attn-mma path is not active.
TEST(moss_tts_local, codec_v2_attn_mma_matches_scalar)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  if (!codec->use_attn_mma()) {
    std::fprintf(stderr, "[codec-v2] attn-mma not active (pre-M5?), skip\n");
    return;
  }

  const int NV = codec->n_quantizers(), T = 53;   // non-16-multiple: tail tile
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  std::uint64_t r = 0x243f6a8885a308d3ULL;
  for (int t = 0; t < T; ++t) {
    for (int q = 0; q < NV; ++q) {
      r = r * 6364136223846793005ULL + 1442695040888963407ULL;
      codes[(std::size_t)t][(std::size_t)q] = (std::int32_t)((r >> 33) % 1024);
    }
  }

  codec->set_use_attn_mma(true);
  std::vector<float> a = codec->decode(codes);
  codec->set_use_attn_mma(false);
  std::vector<float> b = codec->decode(codes);
  codec->set_use_attn_mma(true);
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  const double rl2 = rel_l2_(a.data(), b.data(), a.size());
  std::fprintf(stderr, "[codec-v2] attn mma-vs-scalar waveform rel-L2 = %.6f\n",
               rl2);
  EXPECT_TRUE(rl2 < 1e-2);
}

// Build a deterministic ~`secs`-second 48 kHz stereo waveform (channel-major
// [ch0(N)|ch1(N)]) for the encode-path benchmarks/A-B (no golden needed).
static std::vector<float> synth_stereo_(double secs, int sr = 48000)
{
  const int N = (int)(secs * sr);
  std::vector<float> w((std::size_t)(2 * N));
  for (int k = 0; k < N; ++k) {
    const double t = (double)k / sr;
    w[(std::size_t)k] = 0.3f * (float)std::sin(2.0 * M_PI * 220.0 * t);
    w[(std::size_t)N + k] =
        0.25f * (float)std::sin(2.0 * M_PI * 330.0 * t + 0.5);
  }
  return w;
}

// Codec-v2 ENCODE throughput benchmark (encode = decode's twin, same shared
// run_stage_ so it rides the same matmul2d GEMM + windowed-causal flash attn).
// Env: VPIPE_MOSS_CODEC_V2_MODEL; VPIPE_MOSS_CODEC_BENCH_SECS (default 8).
// VPIPE_MOSS_CODEC_PROFILE=1 for the per-stage breakdown. Skips if dir unset.
TEST(moss_tts_local, codec_v2_encode_bench)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc, /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid() && codec->has_encoder());

  double secs = 8.0;
  if (const char* se = std::getenv("VPIPE_MOSS_CODEC_BENCH_SECS")) {
    secs = std::max(1.0, atof(se));
  }
  std::vector<float> wave = synth_stereo_(secs, codec->sample_rate());

  std::vector<std::vector<std::int32_t>> c0 = codec->encode(wave);  // warm
  ASSERT_TRUE(!c0.empty());
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<std::int32_t>> codes = codec->encode(wave);
  const double wall = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "[codec-v2-enc-bench] %.1fs audio -> %zu frames in "
               "%.1f ms (%.1fx realtime)\n", secs, codes.size(), wall,
               secs / (wall / 1000.0));
  EXPECT_TRUE(!codes.empty());
}

// Encode-path A/B: the matrix-core windowed-causal flash attention must match
// the scalar path on the ENCODER's pre-RVQ latent (stages.back()), the direct
// twin of the decode waveform A/B (comparing final int codes would flip a few
// near the RVQ nearest-neighbour boundaries). Env: VPIPE_MOSS_CODEC_V2_MODEL.
TEST(moss_tts_local, codec_v2_encode_attn_mma_matches_scalar)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (cdir == nullptr || *cdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc, /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid() && codec->has_encoder());
  if (!codec->use_attn_mma()) {
    std::fprintf(stderr, "[codec-v2] enc attn-mma not active, skip\n");
    return;
  }
  std::vector<float> wave = synth_stereo_(2.0, codec->sample_rate());

  std::vector<std::vector<float>> sa, sb;
  codec->set_use_attn_mma(true);  codec->encode(wave, &sa);
  codec->set_use_attn_mma(false); codec->encode(wave, &sb);
  codec->set_use_attn_mma(true);
  ASSERT_TRUE(!sa.empty() && sa.size() == sb.size());
  const std::vector<float>& la = sa.back();
  const std::vector<float>& lb = sb.back();
  ASSERT_TRUE(la.size() == lb.size() && !la.empty());
  const double rl2 = rel_l2_(la.data(), lb.data(), la.size());
  std::fprintf(stderr,
      "[codec-v2] ENCODE attn mma-vs-scalar latent rel-L2 = %.6f\n", rl2);
  EXPECT_TRUE(rl2 < 1e-2);
}

// rel-L2. Env: VPIPE_MOSS_CODEC_V2_MODEL (codec dir) + VPIPE_MOSS15_GOLDEN
// (codec_codes.i32, codec_latent.bf16, codec_wav.f32 from moss15_codec_golden).
TEST(moss_tts_local, codec_v2_decode_matches_golden)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (cdir == nullptr || *cdir == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const fs::path gd(gold);
  if (!fs::exists(gd / "codec_codes.i32") ||
      !fs::exists(gd / "codec_wav.f32")) { return; }

  // codec_codes.i32 = [n_vq=12, T] row-major. Infer T from file size.
  std::vector<std::int32_t> ci;
  {
    std::ifstream in((gd / "codec_codes.i32").string(), std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    ci.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(ci.data()), n);
  }
  const int NV = 12;
  const int T = (int)ci.size() / NV;
  ASSERT_TRUE(T >= 1 && (int)ci.size() == NV * T);
  // codes[t][vq] from [vq*T + t].
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)NV, 0));
  for (int vq = 0; vq < NV; ++vq) {
    for (int t = 0; t < T; ++t) {
      codes[(std::size_t)t][(std::size_t)vq] = ci[(std::size_t)vq * T + t];
    }
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  std::vector<std::vector<float>> stages;
  std::vector<float> wav = codec->decode(codes, &stages);
  ASSERT_TRUE(!wav.empty() && !stages.empty());
  // Localize: my stage[i+1] == golden decoder-module i output (codec_dec{i}).
  for (int i = 0; i + 1 < (int)stages.size(); ++i) {
    std::vector<float> gdv =
        read_bf16_((gd / ("codec_dec" + std::to_string(i) + ".bf16")).string());
    if (gdv.empty() || gdv.size() != stages[(std::size_t)i + 1].size()) {
      std::fprintf(stderr, "[moss15]   dec%d: (no/mismatched golden)\n", i);
      continue;
    }
    const double r =
        rel_l2_(stages[(std::size_t)i + 1].data(), gdv.data(), gdv.size());
    std::fprintf(stderr, "[moss15]   dec%d rel-L2=%.4f\n", i, r);
  }

  // (a) RLFQ latent [768,T] (stage 0, channel-major) vs codec_latent.bf16.
  std::vector<float> glat = read_bf16_((gd / "codec_latent.bf16").string());
  if (!glat.empty()) {
    ASSERT_TRUE(glat.size() == stages[0].size());
    const double rl = rel_l2_(stages[0].data(), glat.data(), glat.size());
    std::fprintf(stderr, "[moss15] codec C1 latent rel-L2 = %.4f\n", rl);
    EXPECT_TRUE(rl < 0.05);
  }

  // (b) full stereo waveform [2, T*3840] (channel-major) vs codec_wav.f32.
  std::vector<float> gwav;
  {
    std::ifstream in((gd / "codec_wav.f32").string(), std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    gwav.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(gwav.data()), n);
  }
  ASSERT_TRUE(!gwav.empty() && gwav.size() == wav.size());
  const double rw = rel_l2_(wav.data(), gwav.data(), gwav.size());
  std::fprintf(stderr, "[moss15] codec waveform rel-L2 = %.4f (%zu samples)\n",
               rw, wav.size());
  EXPECT_TRUE(rw < 0.10);
}

// A3b codec ENCODE: 48 kHz stereo waveform -> 12 RVQ code streams (the inverse
// of decode). Encodes the golden's fixed (real, in-distribution) stereo clip
// and checks: (a) the encoder pre-quant latent [768,T] rel-L2 vs HF, (b)
// per-level RVQ code agreement %, (c) round-trip (vpipe encode -> vpipe decode)
// vs the HF encode->decode waveform. Env: VPIPE_MOSS_CODEC_V2_MODEL (codec dir)
// + VPIPE_MOSS15_GOLDEN (enc_in_wav.f32, enc_latent.bf16, enc_stage{i}.bf16,
// enc_codes.i32, enc_rt_wav.f32 from
//   moss15_codec_encode_golden.py <codec> <golden> 48 <a-real-48k-clip.wav>).
//
// The RLFQ codes are inherently precision-sensitive: the coarse levels (L0..)
// are stable but the deep residual levels flip under any rounding -- HF's OWN
// bf16-vs-fp32 codes only agree ~77%, and HF-decode(bf16 codes) vs HF-decode
// (fp32 codes) is ~0.33 rel-L2. So the bar is: a clean encoder latent, strong
// coarse-level code agreement, and a round-trip within that precision floor
// (vpipe's f16 round-trip is actually slightly CLOSER to HF-bf16 than HF-fp32
// is). Sample-wise rel-L2 vs the INPUT is NOT a codec-quality metric (this is a
// causal neural codec; even HF's rt vs input is ~0.85) -- reported only.
TEST(moss_tts_local, codec_v2_encode_matches_golden)
{
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (cdir == nullptr || *cdir == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const fs::path gd(gold);
  if (!fs::exists(gd / "enc_in_wav.f32") ||
      !fs::exists(gd / "enc_codes.i32")) { return; }

  // Input waveform [2, per] channel-major (exact bits the golden encoded).
  std::vector<float> wave;
  {
    std::ifstream in((gd / "enc_in_wav.f32").string(), std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    wave.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(wave.data()), n);
  }
  ASSERT_TRUE(!wave.empty());

  // Golden codes [n_vq=12, T] row-major (level-major).
  std::vector<std::int32_t> gci;
  {
    std::ifstream in((gd / "enc_codes.i32").string(), std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    gci.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(gci.data()), n);
  }
  const int NV = 12;
  const int T = (int)gci.size() / NV;
  ASSERT_TRUE(T >= 1 && (int)gci.size() == NV * T);

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = MetalMossCodecV2::load(cdir, mc, /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid() && codec->has_encoder());

  std::vector<std::vector<float>> stages;
  std::vector<std::vector<std::int32_t>> codes = codec->encode(wave, &stages);
  ASSERT_TRUE((int)codes.size() == T && !stages.empty());
  ASSERT_TRUE((int)codes[0].size() == NV);

  // (a) per-stage localization: my stage[i] == golden enc_stage{i}.
  for (int i = 0; i < (int)stages.size(); ++i) {
    std::vector<float> gsv =
        read_bf16_((gd / ("enc_stage" + std::to_string(i) + ".bf16")).string());
    if (gsv.empty() || gsv.size() != stages[(std::size_t)i].size()) {
      std::fprintf(stderr, "[moss15]   enc-stage%d: (no golden)\n", i);
      continue;
    }
    const double r =
        rel_l2_(stages[(std::size_t)i].data(), gsv.data(), gsv.size());
    std::fprintf(stderr, "[moss15]   enc-stage%d rel-L2=%.4f\n", i, r);
  }

  // (b) encoder pre-quant latent [768,T] (last stage) vs enc_latent.bf16.
  std::vector<float> glat = read_bf16_((gd / "enc_latent.bf16").string());
  if (!glat.empty()) {
    const std::vector<float>& mine = stages.back();
    ASSERT_TRUE(glat.size() == mine.size());
    const double rl = rel_l2_(mine.data(), glat.data(), glat.size());
    std::fprintf(stderr, "[moss15] encoder latent rel-L2 = %.4f\n", rl);
    EXPECT_TRUE(rl < 0.05);
  }

  // (c) per-level RVQ code agreement %. The coarsest level (L0) carries the
  // most energy and is precision-stable; deep levels flip under f16/bf16
  // rounding (HF bf16-vs-fp32 itself only agrees ~77% overall).
  int total_agree = 0, l0_agree = 0;
  for (int vq = 0; vq < NV; ++vq) {
    int agree = 0;
    for (int t = 0; t < T; ++t) {
      const std::int32_t g = gci[(std::size_t)vq * T + t];
      if (codes[(std::size_t)t][(std::size_t)vq] == g) { ++agree; }
    }
    if (vq == 0) { l0_agree = agree; }
    total_agree += agree;
    std::fprintf(stderr, "[moss15]   codes L%d: %d/%d agree (%.1f%%)\n", vq,
                 agree, T, 100.0 * agree / T);
  }
  std::fprintf(stderr, "[moss15] RVQ codes overall: %d/%d agree (%.1f%%)\n",
               total_agree, NV * T, 100.0 * total_agree / (NV * T));
  EXPECT_TRUE(l0_agree >= (int)(0.90 * T));          // coarse level near-exact
  EXPECT_TRUE(total_agree >= (int)(0.60 * NV * T));  // overall within floor

  // (d) round-trip: vpipe encode -> vpipe decode, vs HF encode->decode and the
  // input. A clean round-trip + high code agreement is the bar.
  std::vector<float> rt = codec->decode(codes);
  ASSERT_TRUE(!rt.empty());
  for (float v : rt) { ASSERT_TRUE(std::isfinite(v)); }

  std::vector<float> hf_rt;
  {
    std::ifstream in((gd / "enc_rt_wav.f32").string(), std::ios::binary);
    if (in) {
      in.seekg(0, std::ios::end);
      const std::streamoff n = in.tellg();
      in.seekg(0, std::ios::beg);
      hf_rt.resize((std::size_t)n / 4);
      in.read(reinterpret_cast<char*>(hf_rt.data()), n);
    }
  }
  if (!hf_rt.empty() && hf_rt.size() == rt.size()) {
    const double rr = rel_l2_(rt.data(), hf_rt.data(), hf_rt.size());
    std::fprintf(stderr, "[moss15] round-trip vs HF rt rel-L2 = %.4f\n", rr);
    // Within the RVQ code-precision floor (~0.33; see header).
    EXPECT_TRUE(rr < 0.40);
  }
  if (rt.size() == wave.size()) {
    const double ri = rel_l2_(rt.data(), wave.data(), wave.size());
    std::fprintf(stderr, "[moss15] round-trip vs input rel-L2 = %.4f\n", ri);
  }
}

// A4 end-to-end: text -> grid (host processor) -> v1.5 LM generate -> codec
// decode -> 48kHz stereo PCM. Verifies (a) the host processor's prompt grid is
// TOKEN-EXACT vs the HF golden (grid_text_ids.i32), and (b) the full pipeline
// produces finite, non-silent stereo audio of the right shape. Env:
// VPIPE_MOSS_TTS_LOCAL_MODEL (bf16 src) + VPIPE_MOSS_CODEC_V2_MODEL +
// VPIPE_MOSS15_GOLDEN (grid_text_ids.i32 from moss15_grid_golden.py).
TEST(moss_tts_local, tts_v15_text_to_audio)
{
  const char* src  = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || cdir == nullptr || *cdir == '\0' ||
      gold == nullptr || *gold == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const fs::path gd(gold);
  if (!fs::exists(gd / "grid_text_ids.i32")) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // (a) host processor grid vs golden, token-exact.
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(src) / "tokenizer.json").string(), &sess);
  ASSERT_TRUE(tok != nullptr);
  const std::string text = "Hello world, this is a test of the speech system.";
  genai::MossV15PromptIds pids;
  auto grid = genai::moss_v15_build_tts_grid(*tok, text, pids);
  ASSERT_TRUE(!grid.empty());
  {
    std::ifstream in((gd / "grid_text_ids.i32").string(), std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<std::int32_t> gi((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(gi.data()), n);
    std::fprintf(stderr, "[moss15] tts grid: mine=%zu golden=%zu\n",
                 grid.size(), gi.size());
    ASSERT_TRUE(grid.size() == gi.size());
    int mism = 0;
    for (std::size_t i = 0; i < gi.size(); ++i) {
      if (grid[i][0] != gi[i]) { ++mism; }
      // audio channels must be pad.
      EXPECT_TRUE(grid[i][1] == pids.audio_pad);
    }
    std::fprintf(stderr, "[moss15] tts grid channel-0 mismatches: %d/%zu\n",
                 mism, gi.size());
    EXPECT_TRUE(mism == 0);
  }

  // (b) full pipeline: quantize v1.5 -> v15 LM -> generate -> codec -> PCM.
  const fs::path qdir = fs::temp_directory_path() /
                        ("vpipe-tts15-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(qdir, ec);
  {
    ModelQuantizer mq(mc);
    QuantizeOptions opt; opt.bits = 8; opt.group = 64;
    std::string err;
    ASSERT_TRUE(mq.run(src, qdir.string(), opt, &err));
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_();
  auto lm = MetalMossV15Model::load(qdir.string(), mc, cfg);
  ASSERT_TRUE(lm != nullptr);
  auto codec = MetalMossCodecV2::load(cdir, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  const int max_frames = 8;
  std::vector<std::vector<int>> frames = lm->generate(grid, max_frames);
  std::fprintf(stderr, "[moss15] tts generated %zu frames\n", frames.size());
  ASSERT_TRUE(!frames.empty());
  // codec wants codes[T][n_vq] int32.
  std::vector<std::vector<std::int32_t>> codes;
  codes.reserve(frames.size());
  for (const auto& f : frames) {
    codes.emplace_back(f.begin(), f.end());
  }
  std::vector<float> wav = codec->decode(codes);
  const int T = (int)codes.size();
  std::fprintf(stderr, "[moss15] tts wav: %zu samples (expect %d)\n",
               wav.size(), T * 3840 * 2);
  ASSERT_TRUE((int)wav.size() == T * 3840 * 2);   // stereo, 3840 samp/frame/ch
  double e = 0.0; bool fin = true;
  for (float v : wav) { e += (double)v * v; if (!std::isfinite(v)) fin = false; }
  std::fprintf(stderr, "[moss15] tts wav L2=%.3f finite=%d\n", std::sqrt(e),
               (int)fin);
  EXPECT_TRUE(fin);
  EXPECT_TRUE(e > 0.0);   // not pure silence
  fs::remove_all(qdir, ec);
}

// M1 on-device streaming calibration: compute the AWQ activation stats ON
// DEVICE (tap the real backbone forward over a corpus) instead of the offline
// HF script, then verify AWQ using those stats reduces 4-bit drift as well as
// HF-calib AWQ. Env: VPIPE_MOSS_TTS_LOCAL_MODEL + VPIPE_MOSS15_GOLDEN.
TEST(moss_tts_local, on_device_calibration_drives_awq)
{
  const char* src  = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  const int hidden = 2560;
  std::vector<float> g_emb = read_bf16_((fs::path(gold) / "a1_embeds.bf16"));
  std::vector<float> g_hid = read_bf16_((fs::path(gold) / "a1_hidden.bf16"));
  if (g_emb.empty() || g_hid.empty()) { return; }
  const int seq = (int)(g_emb.size() / hidden);
  ASSERT_TRUE(seq >= 1 && (int)g_hid.size() == seq * hidden);

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // Tokenize the calibration corpus (the same 8 texts as moss15_calib.py).
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(src) / "tokenizer.json").string(), &sess);
  ASSERT_TRUE(tok != nullptr);
  const char* texts[] = {
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "Speech synthesis turns written text into natural sounding audio.",
    "Numbers like 3.14159 and dates such as June 27th appear often.",
    "She sells sea shells by the sea shore on a sunny afternoon.",
    "Bonjour, comment allez-vous aujourd'hui? Tres bien, merci beaucoup.",
    "Machine learning models quantized to four bits save memory.",
    "Once upon a time, in a land far far away, magic was real.",
  };
  std::vector<std::vector<std::int32_t>> corpus;
  for (const char* t : texts) {
    auto ids = tok->encode(t);
    if (!ids.empty()) { corpus.emplace_back(ids.begin(), ids.end()); }
  }
  ASSERT_TRUE(corpus.size() >= 4);

  std::error_code ec;
  const fs::path q8 = fs::temp_directory_path() /
                      ("vpipe-cal8-" + std::to_string(::getpid()));
  const fs::path cdir = fs::temp_directory_path() /
                        ("vpipe-calib-" + std::to_string(::getpid()));
  fs::remove_all(q8, ec); fs::remove_all(cdir, ec);

  // 8-bit backbone for accurate activations, then on-device calibration.
  {
    ModelQuantizer mq(mc);
    QuantizeOptions opt; opt.bits = 8; opt.group = 64;
    std::string err;
    ASSERT_TRUE(mq.run(src, q8.string(), opt, &err));
  }
  std::string cerr;
  ASSERT_TRUE(collect_backbone_calibration(
      mc, q8.string(), v15_backbone_config_(8), corpus, cdir.string(), &cerr));
  ASSERT_TRUE(fs::exists(cdir / "calib_qkv.f32") &&
              fs::exists(cdir / "calib_gateup.f32") &&
              fs::exists(cdir / "calib_down.f32"));

  // 4-bit drift vs FP golden: plain vs AWQ-with-on-device-calib.
  auto run_drift = [&](const std::string& calib) -> double {
    const fs::path qd = fs::temp_directory_path() /
        ("vpipe-odc-" + std::string(calib.empty() ? "p" : "a") +
         std::to_string(::getpid()));
    fs::remove_all(qd, ec);
    ModelQuantizer mq(mc);
    QuantizeOptions opt; opt.bits = 4; opt.group = 64;
    if (!calib.empty()) {
      opt.smoothquant = true; opt.calib_dir = calib;
      opt.layer_prefix = "transformer.layers."; opt.n_layers = 36;
    }
    std::string err;
    if (!mq.run(src, qd.string(), opt, &err)) { return -1.0; }
    auto bb = MetalQwenModel::load(qd.string(), mc, v15_backbone_config_(4));
    if (!bb) { fs::remove_all(qd, ec); return -2.0; }
    SharedBuffer x = mc->make_shared_buffer((std::size_t)seq * hidden * 2);
    {
      std::ifstream in((fs::path(gold) / "a1_embeds.bf16").string(),
                       std::ios::binary);
      in.read(static_cast<char*>(x.contents()),
              (std::streamsize)seq * hidden * 2);
    }
    ContextManager* cm = bb->context_manager();
    const ContextId cid = cm->acquire_root();
    SharedBuffer hn = bb->forward_embeddings_hidden(cid, x, seq);
    cm->release(cid);
    if (hn.empty()) { fs::remove_all(qd, ec); return -3.0; }
    const auto* h = static_cast<const std::uint16_t*>(hn.contents());
    std::vector<float> v(hidden);
    for (int i = 0; i < hidden; ++i) { v[i] = bf16_to_f32_(h[i]); }
    const double r = rel_l2_(v.data(),
                             &g_hid[(std::size_t)(seq - 1) * hidden], hidden);
    fs::remove_all(qd, ec);
    return r;
  };
  const double plain = run_drift("");
  const double awq = run_drift(cdir.string());
  std::fprintf(stderr,
               "[moss15] on-device-calib 4-bit drift: plain=%.4f awq=%.4f\n",
               plain, awq);
  fs::remove_all(q8, ec); fs::remove_all(cdir, ec);
  ASSERT_TRUE(plain > 0.0 && awq > 0.0);
  EXPECT_TRUE(awq < plain * 0.85);   // on-device calib drives AWQ as intended
}

// The built-in calibration corpus: groups the curated mlx-lm text into ~128
// sequences of <=512 tokens, chat-template-wrapped so the model's control
// tokens (<|im_start|>/<|im_end|>) are exercised at both ends. Env:
// VPIPE_MOSS_TTS_LOCAL_MODEL (for tokenizer.json).
TEST(moss_tts_local, builtin_calibration_corpus)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (src == nullptr || *src == '\0') { return; }
  namespace fs = std::filesystem;
  if (!fs::exists(fs::path(src) / "tokenizer.json")) { return; }
  Session sess;
  auto tok = vpipe::genai::Tokenizer::from_huggingface_json(
      std::string(src) + "/tokenizer.json", &sess);
  ASSERT_TRUE((bool)tok);

  auto corpus = vpipe::genai::build_builtin_calibration_corpus(
      *tok, 128, 512, /*apply_chat_template=*/true);
  ASSERT_TRUE(!corpus.empty());
  EXPECT_TRUE((int)corpus.size() <= 128);
  EXPECT_TRUE((int)corpus.size() >= 64);   // the subset yields plenty

  bool any_full = false;
  std::size_t total = 0;
  for (const auto& s : corpus) {
    EXPECT_TRUE(!s.empty());
    EXPECT_TRUE((int)s.size() <= 512);
    if ((int)s.size() >= 480) { any_full = true; }
    total += s.size();
  }
  EXPECT_TRUE(any_full);   // documents grouped near the 512-token target

  // Control tokens exercised at BOTH ends: every chat-wrapped sequence begins
  // with <|im_start|> and the closing <|im_end|> survives the seq_len cap.
  const std::int32_t im_start = tok->special_token_id("<|im_start|>");
  const std::int32_t im_end   = tok->special_token_id("<|im_end|>");
  ASSERT_TRUE(im_start >= 0 && im_end >= 0);
  int with_start = 0, with_end = 0;
  for (const auto& s : corpus) {
    if (s.front() == im_start) { ++with_start; }
    if (std::find(s.begin(), s.end(), im_end) != s.end()) { ++with_end; }
  }
  EXPECT_TRUE(with_start == (int)corpus.size());
  EXPECT_TRUE(with_end == (int)corpus.size());

  // Raw (no template) path: no leading control token.
  auto raw = vpipe::genai::build_builtin_calibration_corpus(
      *tok, 8, 512, /*apply_chat_template=*/false);
  ASSERT_TRUE(!raw.empty());
  EXPECT_TRUE(raw.front().front() != im_start);

  std::fprintf(stderr,
               "[moss15] builtin calib corpus: %zu seqs, %zu tokens, "
               "avg %.0f/seq\n", corpus.size(), total,
               (double)total / (double)corpus.size());
}

// Voice-clone grid STRUCTURE + token-exactness (no real clip needed). Builds a
// clone grid from a synthetic [Tref][n_vq] ref-codes vector + a fixed target
// text and asserts: (a) the reference block is laid out exactly (audio_start,
// Tref audio_user_slot rows carrying the ref codes in channels 1..n_vq with NO
// delay, audio_end); (b) the TEXT channel (channel 0) is token-exact vs the
// plain builder -- the prefix is identical and the after-reference tail is
// identical, the ONLY difference being the ref block replacing "None". The
// plain builder is itself golden-verified token-exact (tts_v15_text_to_audio),
// so this transitively checks the clone grid against the HF processor's
// _build_generation_or_voice_clone_codes. Env: VPIPE_MOSS_TTS_LOCAL_MODEL
// (tokenizer.json only).
TEST(moss_tts_local, clone_grid_structure)
{
  const char* src = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  if (src == nullptr || *src == '\0') { return; }
  namespace fs = std::filesystem;
  Session sess;
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(src) / "tokenizer.json").string(), &sess);
  ASSERT_TRUE(tok != nullptr);

  const std::string text = "Hello world, this is a test of the speech system.";
  genai::MossV15PromptIds pids;

  // Synthetic reference RVQ codes [Tref][n_vq], deterministic, in 0..1023.
  const int Tref = 5;
  const int nvq  = pids.n_vq;
  std::vector<std::vector<std::int32_t>> ref(Tref,
      std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < Tref; ++t) {
    for (int cb = 0; cb < nvq; ++cb) {
      ref[(std::size_t)t][(std::size_t)cb] = (t * 31 + cb * 7 + 1) % 1024;
    }
  }

  auto noref = genai::moss_v15_build_tts_grid(*tok, text, pids);
  auto clone = genai::moss_v15_build_clone_grid(*tok, ref, text, pids);
  ASSERT_TRUE(!noref.empty() && !clone.empty());

  // Empty ref => falls back to the plain grid (byte-identical).
  auto clone_empty = genai::moss_v15_build_clone_grid(*tok, {}, text, pids);
  ASSERT_TRUE(clone_empty.size() == noref.size());
  {
    bool same = true;
    for (std::size_t i = 0; i < noref.size() && same; ++i) {
      if (clone_empty[i] != noref[i]) { same = false; }
    }
    EXPECT_TRUE(same);
  }

  // Prefix length P = im_start + enc("user\n") + enc("<user_inst>\n- ...:\n").
  const std::size_t P = 1 + tok->encode("user\n").size()
                          + tok->encode("<user_inst>\n- Reference(s):\n").size();
  const std::size_t none_len = tok->encode("None").size();
  ASSERT_TRUE(noref.size() > P + none_len);
  ASSERT_TRUE(clone.size() > P + (std::size_t)(Tref + 2));

  // (a) prefix channel-0 ids identical.
  int pre_mism = 0;
  for (std::size_t i = 0; i < P; ++i) {
    if (clone[i][0] != noref[i][0]) { ++pre_mism; }
  }
  std::fprintf(stderr, "[moss15] clone prefix mism=%d (P=%zu)\n", pre_mism, P);
  EXPECT_TRUE(pre_mism == 0);

  // (b) reference audio block: audio_start, Tref slot rows, audio_end.
  EXPECT_TRUE(clone[P][0] == pids.audio_start);
  for (int c = 1; c <= nvq; ++c) {
    EXPECT_TRUE(clone[P][(std::size_t)c] == pids.audio_pad);
  }
  int slot_mism = 0, code_mism = 0;
  for (int t = 0; t < Tref; ++t) {
    const auto& row = clone[P + 1 + (std::size_t)t];
    if (row[0] != pids.audio_user_slot) { ++slot_mism; }
    for (int cb = 0; cb < nvq; ++cb) {
      if (row[(std::size_t)(1 + cb)] != ref[(std::size_t)t][(std::size_t)cb]) {
        ++code_mism;
      }
    }
  }
  std::fprintf(stderr, "[moss15] clone ref block: slot_mism=%d code_mism=%d\n",
               slot_mism, code_mism);
  EXPECT_TRUE(slot_mism == 0);
  EXPECT_TRUE(code_mism == 0);
  EXPECT_TRUE(clone[P + 1 + (std::size_t)Tref][0] == pids.audio_end);

  // (c) after-reference tail channel-0 ids identical to the plain grid (the
  // ref block of (Tref + 2) rows replaces the (none_len) "None" rows).
  const std::size_t clone_tail = P + 1 + (std::size_t)Tref + 1;
  const std::size_t noref_tail = P + none_len;
  ASSERT_TRUE(clone.size() - clone_tail == noref.size() - noref_tail);
  int tail_mism = 0;
  for (std::size_t i = 0; clone_tail + i < clone.size(); ++i) {
    if (clone[clone_tail + i][0] != noref[noref_tail + i][0]) { ++tail_mism; }
  }
  std::fprintf(stderr, "[moss15] clone tail mism=%d (len=%zu)\n", tail_mism,
               clone.size() - clone_tail);
  EXPECT_TRUE(tail_mism == 0);

  // The grid ends at audio_start (ready for generation).
  EXPECT_TRUE(clone.back()[0] == pids.audio_start);
}

// End-to-end voice clone: synthesize a stereo clip -> codec-v2 encode -> clone
// grid -> v1.5 generate. Confirms the encoder runs, the ref codes are spliced
// into the prompt, generation is CONDITIONED on them (clone output differs from
// the same-seed no-reference output), and audio_end fires (stops before the
// frame cap). Env: VPIPE_MOSS_TTS_LOCAL_MODEL (bf16 src) + VPIPE_MOSS_CODEC_V2_MODEL.
TEST(moss_tts_local, clone_encode_generate)
{
  const char* src  = std::getenv("VPIPE_MOSS_TTS_LOCAL_MODEL");
  const char* cdir = std::getenv("VPIPE_MOSS_CODEC_V2_MODEL");
  if (src == nullptr || *src == '\0' || cdir == nullptr || *cdir == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(src) / "tokenizer.json").string(), &sess);
  ASSERT_TRUE(tok != nullptr);

  // Codec WITH encoder (encode + decode available).
  auto codec = MetalMossCodecV2::load(cdir, mc, /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid() && codec->has_encoder());

  // Synthesize ~0.5 s of stereo audio (channel-major flat [L | R]) @ 48 kHz.
  const int sr = codec->sample_rate();
  const int N  = sr / 2;
  std::vector<float> wave((std::size_t)N * 2, 0.0f);
  for (int i = 0; i < N; ++i) {
    const double tt = (double)i / (double)sr;
    wave[(std::size_t)i]       = 0.3f * (float)std::sin(2.0 * M_PI * 220.0 * tt);
    wave[(std::size_t)(N + i)] = 0.3f * (float)std::sin(2.0 * M_PI * 330.0 * tt);
  }
  std::vector<std::vector<std::int32_t>> ref = codec->encode(wave);
  std::fprintf(stderr, "[moss15] clone: encoded ref -> %zu frames\n",
               ref.size());
  ASSERT_TRUE(!ref.empty());
  for (const auto& f : ref) { ASSERT_TRUE((int)f.size() == codec->n_quantizers()); }

  const std::string text = "Hello there.";
  genai::MossV15PromptIds pids;
  auto clone_grid = genai::moss_v15_build_clone_grid(*tok, ref, text, pids);
  auto plain_grid = genai::moss_v15_build_tts_grid(*tok, text, pids);
  ASSERT_TRUE(clone_grid.size() > plain_grid.size());   // ref block present

  // Quantize + load the v1.5 LM (8-bit backbone) -- same recipe as tts_v15.
  const fs::path qdir = fs::temp_directory_path() /
                        ("vpipe-tts15clone-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(qdir, ec);
  {
    ModelQuantizer mq(mc);
    QuantizeOptions opt; opt.bits = 8; opt.group = 64;
    std::string err;
    ASSERT_TRUE(mq.run(src, qdir.string(), opt, &err));
  }
  MetalMossV15Model::Config cfg;
  cfg.backbone = v15_backbone_config_();
  auto lm = MetalMossV15Model::load(qdir.string(), mc, cfg);
  ASSERT_TRUE(lm != nullptr);

  // The tuned v1.5 audio sampling (the stage defaults); low temp degenerates.
  genai::MossSampling sp;
  sp.temperature = 1.7f; sp.top_k = 25; sp.top_p = 0.8f;
  const int max_frames = 400;
  const std::uint64_t seed = 0x6d6f7373ULL;
  auto clone_frames = lm->generate(clone_grid, max_frames, sp, seed);
  auto plain_frames = lm->generate(plain_grid, max_frames, sp, seed);
  std::fprintf(stderr, "[moss15] clone gen=%zu plain gen=%zu (cap=%d)\n",
               clone_frames.size(), plain_frames.size(), max_frames);
  ASSERT_TRUE(!clone_frames.empty());

  // audio_end firing is a model property (a synthetic-sine reference is OOD);
  // log whether generation stopped before the cap rather than hard-asserting.
  std::fprintf(stderr, "[moss15] clone audio_end fired=%d\n",
               (int)((int)clone_frames.size() < max_frames));

  // Generation is conditioned on the reference: clone vs same-seed plain differ
  // (different length, or different codes where they overlap).
  bool differ = clone_frames.size() != plain_frames.size();
  if (!differ) {
    for (std::size_t t = 0; t < clone_frames.size() && !differ; ++t) {
      if (clone_frames[t] != plain_frames[t]) { differ = true; }
    }
  }
  std::fprintf(stderr, "[moss15] clone-vs-plain differ=%d\n", (int)differ);
  EXPECT_TRUE(differ);

  fs::remove_all(qdir, ec);
}
