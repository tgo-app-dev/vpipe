// Qwen-Image-Edit-2511 dual-stream DiT (M4): MetalQwenImageTransformer, one
// denoiser step on controlled inputs, verified against the diffusers golden
// (dit_golden.py -- transformer only, no refs). Feeds the exact packed noise
// latent + prompt embeds + sigma, and rel-L2s the predicted velocity vs g_dit;
// staged block-0 output vs g_dit_blk0 to localize.
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH (uses <root>/transformer),
// VPIPE_QWEN_IMAGE_EDIT_GOLDEN. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/qwen-image/metal-qwen-image-transformer.h"

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

double
rel_l2_(const std::vector<float>& a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// The DiT runs bf16 (residual stream exceeds f16 range).
SharedBuffer
upload_(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = f32_to_bf16_(v[i]); }
  return b;
}

std::vector<float>
readback_(const SharedBuffer& b, std::size_t n)
{
  std::vector<float> out(n);
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(p[i]); }
  return out;
}

}  // namespace

// M5 matrix-core matmul2d (NAX) A/B: the mma path (default) must match the
// steel dense_gemm_t / affine_qmm_steel path (VPIPE_QIE_NO_MMA2=1) to the bf16
// noise floor -- dequant-once -> dense matmul2d is the same math as the fused
// steel qmm modulo f32-rounding order, compounded over 60 dual-stream blocks.
// Needs only the model (no golden); skips on M4/older (no matrix cores -> both
// loads run steel, so the A/B is vacuous but still passes at ~0). Random inputs
// suffice: both paths process identical bytes, so agreement is the invariant.
TEST(qwen_image_edit_dit, mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* dd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_DIT_DIR");
  const std::string dit_dir =
      (dd != nullptr && *dd != '\0') ? std::string(dd)
      : (root != nullptr && *root != '\0') ? std::string(root) + "/transformer"
      : std::string();
  if (dit_dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // Deterministic random inputs (fixed seed): hidden [gen_seq, in_channels=64],
  // txt [txt_seq, txt_dim=3584]. gen_seq = grid_h*grid_w.
  const int gh = 16, gw = 16, gen_seq = gh * gw, txt_seq = 96;
  const int IC = 64, TXTD = 3584;
  auto rnd = [](std::uint32_t& s) {         // xorshift -> [-0.5, 0.5)
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return (float)((s >> 8) & 0xffffff) / 16777216.0f - 0.5f;
  };
  std::uint32_t s1 = 0x1234567u, s2 = 0x89abcdefu;
  std::vector<float> hidden((std::size_t)gen_seq * IC),
                     txt((std::size_t)txt_seq * TXTD);
  for (auto& v : hidden) { v = rnd(s1); }
  for (auto& v : txt)    { v = rnd(s2); }
  const SharedBuffer h = upload_(mc, hidden);
  const SharedBuffer t = upload_(mc, txt);
  const float sigma = 0.7f;

  // Each load runs the SINGLE-block staged output (stop=0, [gen_seq, hidden])
  // -- the clean discriminator: it exercises the qkv/out/ff mma GEMMs without
  // the 60-block residual compounding that random (untrained-regime) inputs
  // amplify -- AND the full 60-block velocity (a diagnostic; random inputs put
  // the net in a high-gain regime where the per-block bf16 floor compounds
  // geometrically, so this is loose). Free each DiT before the twin loads so
  // only one ~10GB model is resident (16GB box).
  auto run = [&](std::vector<float>& blk0, std::vector<float>& vel) -> bool {
    auto m = MetalQwenImageTransformer::load(dit_dir, mc, {});
    if (m == nullptr) { return false; }
    const bool mma = m->uses_mma2();
    SharedBuffer b0 = m->forward(h, gen_seq, t, txt_seq, gh, gw, sigma, {}, 0);
    SharedBuffer v  = m->forward(h, gen_seq, t, txt_seq, gh, gw, sigma);
    if (b0.empty() || v.empty()) { return false; }
    blk0 = readback_(b0, (std::size_t)gen_seq * 3072);
    vel  = readback_(v,  (std::size_t)gen_seq * IC);
    return mma;
  };

  // (1) matmul2d path (default).
  std::vector<float> blk0_mma, vel_mma;
  const bool mma_engaged = run(blk0_mma, vel_mma);
  ASSERT_TRUE(!blk0_mma.empty());
  if (!mma_engaged) {
    std::printf("[qwen_image_edit_dit] matmul2d NOT engaged (no matrix cores "
                "or NO_MMA2 set) -- A/B is vacuous\n");
  }

  // (2) steel path (force NO_MMA2 for this load; restore after).
  ::setenv("VPIPE_QIE_NO_MMA2", "1", 1);
  std::vector<float> blk0_steel, vel_steel;
  const bool steel_mma = run(blk0_steel, vel_steel);
  ::unsetenv("VPIPE_QIE_NO_MMA2");
  ASSERT_TRUE(!blk0_steel.empty());
  EXPECT_TRUE(!steel_mma);   // NO_MMA2 -> steel

  const double rb = rel_l2_(blk0_mma, blk0_steel.data(), blk0_mma.size());
  const double rv = rel_l2_(vel_mma,  vel_steel.data(),  vel_mma.size());
  std::printf("[qwen_image_edit_dit] matmul2d vs steel: block0 rel-L2 = %.6f, "
              "velocity rel-L2 = %.6f (mma %s)\n", rb, rv,
              mma_engaged ? "engaged" : "OFF");
  // Per-block: dequant-once -> matmul2d vs the fused steel qmm are both
  // f32-accumulate, differing only by the extra bf16 rounding of the dequanted
  // weight + accumulation order -> a small fraction of the (identical) residual.
  EXPECT_TRUE(rb < 0.01);
  // Velocity is diagnostic only: bound it loosely to still catch NaN/garbage.
  EXPECT_TRUE(std::isfinite(rv) && rv < 0.2);
}

TEST(qwen_image_edit_dit, step_matches_golden)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<float> hidden = read_f32_(gdir + "/dit_hidden.f32");
  const std::vector<float> txt = read_f32_(gdir + "/dit_txt.f32");
  const std::vector<float> meta = read_f32_(gdir + "/dit_meta.f32");
  const std::vector<float> sigv = read_f32_(gdir + "/dit_sigma.f32");
  const std::vector<float> gdit = read_f32_(gdir + "/g_dit.f32");
  if (hidden.empty() || txt.empty() || meta.size() < 4 || gdit.empty()) {
    return;
  }
  const int gen_seq = (int)meta[0], txt_seq = (int)meta[1];
  const int gh = (int)meta[2], gw = (int)meta[3];
  const float sigma = sigv.empty() ? 0.7f : sigv[0];

  // VPIPE_QWEN_IMAGE_EDIT_DIT_DIR overrides the DiT dir (e.g. a w4/w8 quantized
  // transformer) -- the velocity should still track the bf16 golden within the
  // quant error added on top of the bf16 noise floor (looser bar).
  const char* dd = std::getenv("VPIPE_QWEN_IMAGE_EDIT_DIT_DIR");
  const std::string dit_dir = (dd != nullptr && *dd != '\0')
      ? std::string(dd) : std::string(root) + "/transformer";
  const bool quantized = (dd != nullptr && *dd != '\0');
  auto m = MetalQwenImageTransformer::load(dit_dir, mc, {});
  ASSERT_TRUE(m != nullptr);

  const SharedBuffer h = upload_(mc, hidden);
  const SharedBuffer t = upload_(mc, txt);

  // Staged: post img_in embedder vs g_dit_imgin.
  const std::vector<float> gimgin = read_f32_(gdir + "/g_dit_imgin.f32");
  if (!gimgin.empty()) {
    SharedBuffer e =
        m->forward(h, gen_seq, t, txt_seq, gh, gw, sigma, {}, /*stop=*/-2);
    ASSERT_TRUE(!e.empty());
    const auto got = readback_(e, (std::size_t)gen_seq * 3072);
    const double r = rel_l2_(got, gimgin.data(), got.size());
    std::printf("[qwen_image_edit_dit] img_in rel-L2 = %.6f\n", r);
  }

  // Debug: post-attention (pre-MLP) finiteness of block 0.
  {
    SharedBuffer d3 =
        m->forward(h, gen_seq, t, txt_seq, gh, gw, sigma, {}, /*stop=*/-3);
    const auto got = readback_(d3, (std::size_t)gen_seq * 3072);
    int nf = 0; double mx = 0;
    for (float v : got) { if (!std::isfinite(v)) { ++nf; } else { mx = std::max(mx, (double)std::fabs(v)); } }
    std::printf("[qwen_image_edit_dit] post-attn: nonfinite=%d maxabs=%.3f\n",
                nf, mx);
  }

  // Staged: block-0 image-stream hidden vs g_dit_blk0.
  const std::vector<float> gblk0 = read_f32_(gdir + "/g_dit_blk0.f32");
  if (!gblk0.empty()) {
    SharedBuffer b0 =
        m->forward(h, gen_seq, t, txt_seq, gh, gw, sigma, {}, /*stop=*/0);
    ASSERT_TRUE(!b0.empty());
    const auto got = readback_(b0, (std::size_t)gen_seq * 3072);
    const double r = rel_l2_(got, gblk0.data(), got.size());
    std::printf("[qwen_image_edit_dit] block0 img rel-L2 = %.6f\n", r);
    EXPECT_TRUE(r < 0.03);
  }

  // Full step: velocity vs g_dit. NOTE: the QwenImage DiT residual stream
  // reaches ~1e7, forcing bf16 (f16 overflows); the reference is bf16 too, so
  // two valid bf16 implementations agree only to the bf16 noise floor (~0.004
  // /block, ~sqrt(60)-compounded). We check both the rel-L2 band AND that the
  // velocity DIRECTION matches (cosine ~= 1 => coherent denoising).
  SharedBuffer vel = m->forward(h, gen_seq, t, txt_seq, gh, gw, sigma);
  ASSERT_TRUE(!vel.empty());
  const auto got = readback_(vel, (std::size_t)gen_seq * 64);
  const double r = rel_l2_(got, gdit.data(), got.size());
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    dot += (double)got[i] * gdit[i]; na += (double)got[i] * got[i];
    nb += (double)gdit[i] * gdit[i];
  }
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb));
  std::printf("[qwen_image_edit_dit] velocity rel-L2 = %.6f  cosine = %.6f "
              "(gen=%d txt=%d)\n", r, cos, gen_seq, txt_seq);
  // bf16 noise floor for a 1e7-magnitude residual; the quantized DiT adds w4
  // group-affine error on top (attention linears quantized), so a looser bar.
  EXPECT_TRUE(r < (quantized ? 0.20 : 0.10));
  EXPECT_TRUE(cos > (quantized ? 0.98 : 0.995));   // direction -> coherent

  // Streaming-mode equality: the memory-bounded stream_blocks path (load + free
  // each block on demand from the retained mmap) must be numerically IDENTICAL
  // to the preloaded path -- same weights, kernels, and dispatch order; only
  // the command-buffer boundaries differ (a flush per block). This is what lets
  // AWQ calibration / a full denoise run on a 16 GB box. Free the preloaded DiT
  // first so both need not be resident at once.
  {
    const std::vector<float> got_pre = got;   // preloaded velocity
    m.reset();                                 // release the preloaded DiT
    // (a) pure streaming (pin_frac = 0): one block resident, tail streamed.
    {
      auto ms = MetalQwenImageTransformer::load(dit_dir, mc, {},
                                                /*stream_blocks=*/true,
                                                /*pin_frac=*/0.0);
      ASSERT_TRUE(ms != nullptr);
      EXPECT_TRUE(ms->pinned_blocks() == 0);
      SharedBuffer vs = ms->forward(h, gen_seq, t, txt_seq, gh, gw, sigma);
      ASSERT_TRUE(!vs.empty());
      const auto got_s = readback_(vs, (std::size_t)gen_seq * 64);
      const double rs = rel_l2_(got_s, got_pre.data(), got_pre.size());
      std::printf("[qwen_image_edit_dit] STREAM vs PRELOAD rel-L2 = %.3e\n", rs);
      EXPECT_TRUE(rs < 1e-5);   // bit-identical up to GPU nondeterminism
    }
    // (b) pinned-prefix streaming (60% RAM budget): pins a leading block prefix
    // resident and streams the rest. Pinning changes only residency, not the
    // math, so it must ALSO be bit-identical to preload.
    {
      auto mp = MetalQwenImageTransformer::load(dit_dir, mc, {},
                                                /*stream_blocks=*/true,
                                                /*pin_frac=*/0.60);
      ASSERT_TRUE(mp != nullptr);
      EXPECT_TRUE(mp->pinned_blocks() > 0);   // spare RAM -> some blocks pinned
      SharedBuffer vp = mp->forward(h, gen_seq, t, txt_seq, gh, gw, sigma);
      ASSERT_TRUE(!vp.empty());
      const auto got_p = readback_(vp, (std::size_t)gen_seq * 64);
      const double rp = rel_l2_(got_p, got_pre.data(), got_pre.size());
      std::printf("[qwen_image_edit_dit] STREAM+PIN(%d/%d) vs PRELOAD "
                  "rel-L2 = %.3e\n", mp->pinned_blocks(), 60, rp);
      EXPECT_TRUE(rp < 1e-5);
    }
  }
}
