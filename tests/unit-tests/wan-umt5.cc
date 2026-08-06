// The umT5-XXL encoder (Wan's text tower), verified against the
// transformers reference.
//
// The golden carries the TOKEN IDS as well as the hidden states, so this
// tests the encoder and nothing else: a tokenizer disagreement would
// otherwise surface here as an encoder error and send the search to the
// wrong place. The tokenizer is checked where the conditioner uses it.
//
// What is actually at risk in this model, and what a mismatch would mean:
//   * the relative-position bias is the ONLY position signal (no rotary),
//     and it is looked up through a bucketed distance whose float log
//     truncates -- an off-by-one bucket shows as a small, position-
//     dependent error rather than garbage.
//   * T5 does NOT scale scores by 1/sqrt(d). Scaling would be wrong by a
//     constant factor per layer and compound across 24 of them.
//   * umT5 carries a bias table in EVERY layer where T5 shares layer 0's.
//     Reading layer 0's everywhere would match at layer 0 and drift after.
//
// Env: VPIPE_WAN_TEST_MODEL_PATH = the Wan model root, VPIPE_WAN_UMT5_GOLDEN
// = the dir tools/dump_umt5_golden.py wrote. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/wan/metal-umt5-encoder.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

template <typename T>
std::vector<T>
read_raw_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<T> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / sizeof(T));
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

}  // namespace

TEST(wan_umt5, config_from_json)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalUmt5Encoder::Config cfg;
  std::string err;
  const bool ok = MetalUmt5Encoder::config_from_json(
      std::string(root) + "/text_encoder", cfg, &err);
  if (!ok) { std::printf("[wan_umt5] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(cfg.d_model == 4096);
  EXPECT_TRUE(cfg.d_ff == 10240);
  EXPECT_TRUE(cfg.n_heads == 64);
  EXPECT_TRUE(cfg.d_kv == 64);
  EXPECT_TRUE(cfg.n_layers == 24);
  EXPECT_TRUE(cfg.vocab == 256384);
  EXPECT_TRUE(cfg.n_heads * cfg.d_kv == cfg.d_model);
}

TEST(wan_umt5, encode_matches_golden)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_WAN_UMT5_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<std::int32_t> ids =
      read_raw_<std::int32_t>(gdir + "/ids.i32");
  const std::vector<float> ref = read_raw_<float>(gdir + "/hidden.f32");
  if (ids.empty() || ref.empty()) { return; }   // golden not dumped -> skip

  MetalUmt5Encoder::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalUmt5Encoder::config_from_json(
      std::string(root) + "/text_encoder", cfg, &cerr));
  const int L = (int)ids.size();
  ASSERT_TRUE(ref.size() == (std::size_t)L * cfg.d_model);

  // The golden's real token count: the reference zeroes every row past it,
  // so the first all-zero row IS n_valid.
  int n_valid = L;
  for (int i = 0; i < L; ++i) {
    bool zero = true;
    for (int c = 0; c < cfg.d_model && zero; ++c) {
      if (ref[(std::size_t)i * cfg.d_model + c] != 0.0f) { zero = false; }
    }
    if (zero) { n_valid = i; break; }
  }
  ASSERT_TRUE(n_valid > 0);

  auto m = MetalUmt5Encoder::load(std::string(root) + "/text_encoder", mc, cfg);
  ASSERT_TRUE(m != nullptr);

  std::string err;
  SharedBuffer h = m->encode(ids, n_valid, &err);
  if (h.empty()) { std::printf("[wan_umt5] encode: %s\n", err.c_str()); }
  ASSERT_TRUE(!h.empty());
  ASSERT_TRUE(h.byte_size() >= ref.size() * 2);

  const auto* hp = static_cast<const std::uint16_t*>(h.contents());
  std::vector<float> got(ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_to_f32_(hp[i]); }

  // Compare only the REAL rows: the padded tail is zero on both sides by
  // construction, so including it would dilute the metric with agreement
  // that was never in question.
  const std::size_t real = (std::size_t)n_valid * cfg.d_model;
  const double r = rel_l2_(got.data(), ref.data(), real);
  std::printf("[wan_umt5] encode rel-L2 = %.6f (%d real of %d tokens, d=%d)\n",
              r, n_valid, L, cfg.d_model);
  // The golden is the FP32 reference, and this tower runs bf16 -- so the
  // bar has to be read against what bf16 itself costs, not against the
  // VAE's f16 bar. MEASURED on this checkpoint: running the transformers
  // reference in bfloat16 and comparing it to its own fp32 output scores
  // 0.068 over these tokens, because 24 layers of residual accumulate
  // through an 8-bit mantissa. So 0.05 is not a loose bar -- it is a bar
  // the reference in bf16 would FAIL, and this implementation passes it
  // (~0.043) because its GEMMs accumulate in f32 where torch's do not.
  // A real bug here reads as a much larger number: an off-by-one
  // position bucket or a shared (rather than per-layer) bias table both
  // land well past 0.1.
  EXPECT_TRUE(r < 0.05);

  // The zero tail is a contract, not a side effect: it is what the DiT
  // cross-attends to past the prompt.
  bool tail_zero = true;
  for (std::size_t i = real; i < got.size(); ++i) {
    if (got[i] != 0.0f) { tail_zero = false; break; }
  }
  EXPECT_TRUE(tail_zero);
}

// ONE layer, against a one-layer reference. The full-depth test above can
// only bound the total, and at bf16 that bound is loose enough to hide a
// small systematic error -- an off-by-one position bucket, say, which is
// exactly the kind of mistake this model invites. At depth 1 there is no
// accumulation left to hide behind, so the per-layer math is either right
// or it is not: the attention (with its bucketed bias and its ABSENT
// 1/sqrt(d)), the RMS norm and the gated feed-forward all show up here at
// full contrast.
TEST(wan_umt5, single_layer_matches_golden)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_WAN_UMT5_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<std::int32_t> ids =
      read_raw_<std::int32_t>(gdir + "/ids.i32");
  const std::vector<float> ref = read_raw_<float>(gdir + "/hidden_1layer.f32");
  if (ids.empty() || ref.empty()) { return; }

  MetalUmt5Encoder::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalUmt5Encoder::config_from_json(
      std::string(root) + "/text_encoder", cfg, &cerr));
  cfg.n_layers = 1;                        // block 0 + the final norm
  const int L = (int)ids.size();
  ASSERT_TRUE(ref.size() == (std::size_t)L * cfg.d_model);

  int n_valid = L;
  for (int i = 0; i < L; ++i) {
    bool zero = true;
    for (int c = 0; c < cfg.d_model && zero; ++c) {
      if (ref[(std::size_t)i * cfg.d_model + c] != 0.0f) { zero = false; }
    }
    if (zero) { n_valid = i; break; }
  }
  ASSERT_TRUE(n_valid > 0);

  auto m = MetalUmt5Encoder::load(std::string(root) + "/text_encoder", mc, cfg);
  ASSERT_TRUE(m != nullptr);
  std::string err;
  SharedBuffer h = m->encode(ids, n_valid, &err);
  if (h.empty()) { std::printf("[wan_umt5] encode: %s\n", err.c_str()); }
  ASSERT_TRUE(!h.empty());

  const auto* hp = static_cast<const std::uint16_t*>(h.contents());
  const std::size_t real = (std::size_t)n_valid * cfg.d_model;
  std::vector<float> got(real);
  for (std::size_t i = 0; i < real; ++i) { got[i] = bf16_to_f32_(hp[i]); }
  const double r = rel_l2_(got.data(), ref.data(), real);
  std::printf("[wan_umt5] single-layer rel-L2 = %.6f\n", r);
  // One layer of bf16 rounding only -- an order of magnitude tighter than
  // the 24-layer bar, and tight enough that a bucket error cannot pass.
  EXPECT_TRUE(r < 0.005);
}
