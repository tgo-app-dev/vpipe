// The Wan denoiser (WanTransformer3DModel), against the diffusers
// reference.
//
// The golden is a TRUNCATED model -- two blocks out of forty, dumped by
// tools/dump_wan_dit_golden.py -- and that is the point rather than a
// compromise. One A14B expert is ~53 GB in fp32, so a full-depth fp32
// reference does not exist on any machine this runs on, and a bf16
// reference would be comparing two approximations to each other. At depth
// 2 the reference is exact, the comparison is against fp32, and every
// piece of the model is still exercised:
//
//   * the two patch flattenings, which run OPPOSITE ways round -- the
//     input patch is channel-slowest (it is a Conv3d weight) and the
//     output patch is channel-fastest (the reference reshapes proj_out's
//     columns as (p_t, p_h, p_w, c)). Getting both the same way produces a
//     result that is the right shape and the right magnitude and wrong.
//   * the conditioning tower: sinusoidal timestep, the two time MLPs, the
//     6-way modulation table, the text projection.
//   * inside a block: fp32 LayerNorm with NO affine, adaLN modulation, the
//     ACROSS-HEADS q/k RMS norm (not the per-head norm every other model
//     here uses), 3-axis 44/42/42 RoPE, self-attention, cross-attention
//     into the text with no position signal at all, and the UNGATED
//     feed-forward.
//
// Two blocks and not one because block 1's weights differ from block 0's:
// an implementation that read any per-block tensor from block 0 would pass
// at depth 1 and fail here.
//
// The DEPTH comes from the golden's manifest, not from this file, so the
// same test runs against a full 40-block golden by pointing the env var at
// one (`--layers 40 --dtype bfloat16`, which is what fits: fp32 at full
// depth is 53 GB on the reference side alone). That run costs ~28 GB on
// each side and is what proves the blocks accumulate correctly rather than
// just individually; the committed default stays at depth 2 so the suite
// does not need a 64 GB box to be honest.
//
// Env: VPIPE_WAN_TEST_MODEL_PATH = the Wan model root, VPIPE_WAN_DIT_GOLDEN
// = the dir the dumper wrote. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/wan/metal-wan-transformer.h"

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

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer
to_bf16_buf_(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = f32_to_bf16_(v[i]); }
  return b;
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

TEST(wan_dit, config_from_json)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalWanTransformer::Config cfg;
  std::string err;
  const bool ok = MetalWanTransformer::config_from_json(
      std::string(root) + "/transformer", cfg, &err);
  if (!ok) { std::printf("[wan_dit] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(cfg.n_heads == 40);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.hidden == 5120);
  EXPECT_TRUE(cfg.ffn == 13824);
  EXPECT_TRUE(cfg.n_layers == 40);
  EXPECT_TRUE(cfg.text_dim == 4096);
  // I2V conditioning is channel-wise: 16 noise + 4 mask + 16 image latent.
  // A text-to-video checkpoint would say 16 here, and both are valid --
  // what is NOT valid is a Wan 2.1-style checkpoint with a CLIP tower,
  // which config_from_json refuses outright.
  EXPECT_TRUE(cfg.in_channels == 36);
  EXPECT_TRUE(cfg.out_channels == 16);
  // The RoPE split is derived, not configured: h = w = 2*(head_dim/6) and
  // t takes the remainder. 44/42/42 at head_dim 128, and they must sum to
  // the head dim or the rotation would run off the end of a head.
  EXPECT_TRUE(cfg.rope_t() == 44);
  EXPECT_TRUE(cfg.rope_h() == 42);
  EXPECT_TRUE(cfg.rope_w() == 42);
  EXPECT_TRUE(cfg.rope_t() + cfg.rope_h() + cfg.rope_w() == cfg.head_dim);
  EXPECT_TRUE(cfg.patch_elems() == 36 * 4);
  EXPECT_TRUE(cfg.out_patch_elems() == 16 * 4);
}

TEST(wan_dit, forward_matches_golden)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_WAN_DIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  std::ifstream mf(gdir + "/manifest.json");
  if (!mf) { return; }   // golden not dumped -> skip
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  ASSERT_TRUE(man.is_object());
  auto mo = man.as_object();
  const int layers   = (int)mo.at("layers").as_int(2);
  const int T        = (int)mo.at("T").as_int(3);
  const int h        = (int)mo.at("h").as_int(8);
  const int w        = (int)mo.at("w").as_int(12);
  const int text_seq = (int)mo.at("text_seq").as_int(512);
  const float ts     = (float)mo.at("timestep").as_real(750.0);
  const std::string gdtype =
      mo.contains("dtype") ? std::string(mo.at("dtype").as_string("float32"))
                           : std::string("float32");

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> lat = read_f32_(gdir + "/latents.f32");
  const std::vector<float> txt = read_f32_(gdir + "/text.f32");
  const std::vector<float> ref = read_f32_(gdir + "/out.f32");
  if (lat.empty() || txt.empty() || ref.empty()) { return; }

  MetalWanTransformer::Config cfg;
  std::string cerr;
  const std::string ddir = std::string(root) + "/transformer";
  ASSERT_TRUE(MetalWanTransformer::config_from_json(ddir, cfg, &cerr));
  cfg.n_layers = layers;

  // A quantized checkpoint is a DIFFERENT question from a correctness
  // one, and it has to be asked with a different bar or the same test
  // means two things. The width comes from the checkpoint's own config
  // rather than an env var, so pointing the test at a directory is all
  // it takes and there is nothing to keep in sync.
  int qbits = 0;
  {
    std::ifstream cf(ddir + "/config.json");
    if (cf) {
      try {
        FlexData c = FlexData::from_json(cf);
        if (c.is_object() && c.as_object().contains("quantization")) {
          FlexData q = c.as_object().at("quantization");
          if (q.is_object()) {
            qbits = (int)q.as_object().at("bits").as_int(0);
          }
        }
      } catch (...) {
      }
    }
  }
  ASSERT_TRUE(lat.size() ==
              (std::size_t)cfg.in_channels * T * h * w);
  ASSERT_TRUE(txt.size() == (std::size_t)text_seq * cfg.text_dim);
  ASSERT_TRUE(ref.size() == (std::size_t)cfg.out_channels * T * h * w);

  auto m = MetalWanTransformer::load(std::string(root) + "/transformer", mc,
                                     cfg);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer lb = to_bf16_buf_(mc, lat);
  SharedBuffer tb = to_bf16_buf_(mc, txt);
  ASSERT_TRUE(!lb.empty() && !tb.empty());

  std::string err;
  SharedBuffer tp = m->encode_text(tb, text_seq, &err);
  if (tp.empty()) { std::printf("[wan_dit] encode_text: %s\n", err.c_str()); }
  ASSERT_TRUE(!tp.empty());

  SharedBuffer out = m->forward(lb, T, h, w, tp, text_seq, ts, &err);
  if (out.empty()) { std::printf("[wan_dit] forward: %s\n", err.c_str()); }
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= ref.size() * 2);

  const auto* op = static_cast<const std::uint16_t*>(out.contents());
  std::vector<float> got(ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_to_f32_(op[i]); }
  const double r = rel_l2_(got.data(), ref.data(), ref.size());
  std::printf("[wan_dit] velocity rel-L2 = %.6f (%d blocks, %d tokens, %s "
              "golden, %s weights)\n", r, layers, T * (h / 2) * (w / 2),
              gdtype.c_str(),
              qbits > 0 ? (qbits == 4 ? "w4" : "w8") : "bf16");
  // The bar depends on which KIND of golden this is, because against a
  // bf16 golden the reference is itself an approximation and the two
  // rounding paths are independent.
  //
  // MEASURED on this checkpoint, both at T=3, 8x12:
  //   depth 2, fp32 golden : this implementation 0.0135, and the
  //     transformers reference run in bfloat16 scores 0.0137 against its
  //     OWN fp32 output. So 0.0135 is not "close enough" -- it is the bf16
  //     floor, and a bar tighter than ~0.014 is one the reference itself
  //     would fail. (These GEMMs accumulate in f32 where torch's do not,
  //     which is why it lands a hair under.)
  //   depth 40, bf16 golden: 0.0204.
  //
  // The second number is the one that carries the argument. Twenty times
  // the depth moved the disagreement 0.0135 -> 0.0204, not 0.0135 ->
  // 0.06 (sqrt(20)x) and not linearly: the error is uncorrelated rounding
  // that partly cancels, not a systematic difference compounding block
  // over block. A real defect does the opposite -- it grows WITH depth.
  //
  // What the bar has to separate is that floor from an actual mistake, and
  // every mistake this model invites -- a swapped patch flattening,
  // per-head instead of across-head q/k norm, an off-by-one RoPE axis
  // split, the modulation chunks in the wrong order -- lands one to three
  // orders of magnitude above it, not just outside it.
  //
  // Against a QUANTIZED checkpoint none of that applies: the weights
  // are not the reference's weights, so what is measured is the
  // quantizer's error and not the forward pass's. MEASURED on this
  // checkpoint, same goldens, same session:
  //
  //             depth 2 / fp32     depth 40 / bf16
  //   bf16          0.0135             0.0204
  //   w8            0.0186             0.0317
  //   w4            0.1539             0.1466
  //
  // Subtracting the bf16 baseline in quadrature -- the two error
  // sources are independent -- leaves w8 contributing 0.013 and 0.024
  // of its own, against w4's 0.153 and 0.145. So w8 costs about what
  // bf16 rounding already costs and w4 costs an order of magnitude
  // more: a real quality decision, not noise.
  //
  // Both widths are FLAT to slightly falling with depth (w4 0.1539 ->
  // 0.1466 over twenty times the blocks), which is the signature of
  // per-block weight noise that partly cancels rather than a bias
  // compounding down the stack. Quantization error here is a fixed
  // tax, not something that runs away in a deep model.
  //
  // The bars sit just above each measurement so a REGRESSION in the
  // quantizer still fails, while the accuracy the widths cost is
  // recorded rather than asserted away.
  double bar = (gdtype == "bfloat16" ? 0.03 : 0.02);
  if (qbits == 8) { bar = (gdtype == "bfloat16" ? 0.04 : 0.025); }
  if (qbits == 4) { bar = 0.20; }
  EXPECT_TRUE(r < bar);
}
