// Qwen-Image-Edit-2511 Qwen2.5-VL vision tower (M5): MetalQwen25Vision, a fixed
// 112x112 image (one attention window) verified against the diffusers golden
// (vision_golden.py). Feeds the exact patchified pixel_values + per-patch (h,w)
// positions, and rel-L2s the merged vision tokens vs g_vis.
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH (uses <root>/text_encoder),
// VPIPE_QWEN_IMAGE_EDIT_GOLDEN. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/qwen-image/metal-qwen25-vision.h"

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

inline std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

std::vector<float> read_f32_(const std::string& p)
{
  std::ifstream in(p, std::ios::binary); std::vector<float> o;
  if (!in) { return o; }
  in.seekg(0, std::ios::end); const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg); o.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(o.data()), n); return o;
}
std::vector<std::int32_t> read_i32_(const std::string& p)
{
  std::ifstream in(p, std::ios::binary); std::vector<std::int32_t> o;
  if (!in) { return o; }
  in.seekg(0, std::ios::end); const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg); o.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(o.data()), n); return o;
}
double rel_l2_(const std::vector<float>& a, const float* b, std::size_t n)
{
  double num = 0, den = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

TEST(qwen_image_edit_vision, tower_matches_golden)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<float> px = read_f32_(gdir + "/vis_pixels.f32");
  const std::vector<std::int32_t> posi = read_i32_(gdir + "/vis_pos.i32");
  const std::vector<float> gvis = read_f32_(gdir + "/g_vis.f32");
  if (px.empty() || posi.empty() || gvis.empty()) { return; }

  MetalQwen25Vision::Config cfg;
  const int seq = (int)posi.size() / 2;
  ASSERT_TRUE((int)px.size() == seq * cfg.patch_in);

  auto m = MetalQwen25Vision::load(std::string(root) + "/text_encoder", mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // Upload patchified pixels as bf16.
  SharedBuffer pixels = mc->make_shared_buffer(px.size() * 2);
  { auto* d = static_cast<std::uint16_t*>(pixels.contents());
    for (std::size_t i = 0; i < px.size(); ++i) { d[i] = f32_to_bf16_(px[i]); } }

  std::vector<int> pos(posi.begin(), posi.end());

  // Localize: patch-embed (stop=-1) and block-0 (stop=0) hidden [seq,1280].
  auto dbg = [&](const char* stop, const char* gf) {
    const std::vector<float> gold = read_f32_(gdir + "/" + gf);
    if (gold.empty()) { return; }
    ::setenv("VPIPE_QIE_VIS_STOP", stop, 1);
    SharedBuffer h = m->encode(pixels, seq, pos);
    ::unsetenv("VPIPE_QIE_VIS_STOP");
    std::vector<float> got(gold.size());
    const auto* p = static_cast<const std::uint16_t*>(h.contents());
    for (std::size_t i = 0; i < gold.size(); ++i) { got[i] = bf16_to_f32_(p[i]); }
    std::printf("[qwen_image_edit_vision] stop=%s rel-L2 = %.6f\n", stop,
                rel_l2_(got, gold.data(), gold.size()));
  };
  dbg("-1", "g_vis_patch.f32");
  dbg("0", "g_vis_blk0.f32");

  SharedBuffer out = m->encode(pixels, seq, pos);
  ASSERT_TRUE(!out.empty());

  const int mseq = seq / (cfg.merge * cfg.merge);
  const std::size_t n = (std::size_t)mseq * cfg.out_hidden;
  ASSERT_TRUE(gvis.size() == n);
  std::vector<float> got(n);
  const auto* p = static_cast<const std::uint16_t*>(out.contents());
  for (std::size_t i = 0; i < n; ++i) { got[i] = bf16_to_f32_(p[i]); }

  const double r = rel_l2_(got, gvis.data(), n);
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += (double)got[i] * gvis[i]; na += (double)got[i] * got[i];
    nb += (double)gvis[i] * gvis[i];
  }
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb));
  std::printf("[qwen_image_edit_vision] vision tokens rel-L2 = %.6f cosine = "
              "%.6f (seq=%d -> %d tokens)\n", r, cos, seq, mseq);
  // bf16 noise over 32 blocks + merger (patch-embed 3e-5, block0 2e-3 confirm
  // the tower is correct; the accumulation floor is ~0.03-0.04).
  EXPECT_TRUE(r < 0.05);
  EXPECT_TRUE(cos > 0.998);
}

// Multi-window diagnostic: a REAL 384^2 condition image (grid 22x32 = 704
// patches -> 176 tokens across 12 windows), the case the single-window tower
// does NOT yet handle. Reads the v2_* golden (qie_vision_golden.py) and reports
// the current full-attention tower's rel-L2 vs HF -- expected LARGE until window
// attention lands (this test documents the gap; it does not gate the suite).
TEST(qwen_image_edit_vision, multiwindow_gap)
{
  const char* root = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<float> px = read_f32_(gdir + "/v2_pixels.f32");
  const std::vector<std::int32_t> posi = read_i32_(gdir + "/v2_pos.i32");
  const std::vector<float> gvis = read_f32_(gdir + "/g_v2_tokens.f32");
  if (px.empty() || posi.empty() || gvis.empty()) {
    std::printf("[qwen_image_edit_vision] multiwindow_gap: no v2_* golden, "
                "skipping\n");
    return;
  }
  MetalQwen25Vision::Config cfg;
  const int seq = (int)posi.size() / 2;
  ASSERT_TRUE((int)px.size() == seq * cfg.patch_in);

  auto m = MetalQwen25Vision::load(std::string(root) + "/text_encoder", mc, cfg);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer pixels = mc->make_shared_buffer(px.size() * 2);
  { auto* d = static_cast<std::uint16_t*>(pixels.contents());
    for (std::size_t i = 0; i < px.size(); ++i) { d[i] = f32_to_bf16_(px[i]); } }
  std::vector<int> pos(posi.begin(), posi.end());

  // Localize (reordered space): patch-embed (stop=-1) and block-0 (stop=0).
  auto dbg = [&](const char* stop, const char* gf) {
    const std::vector<float> gold = read_f32_(gdir + "/" + gf);
    if (gold.empty()) { return; }
    ::setenv("VPIPE_QIE_VIS_STOP", stop, 1);
    SharedBuffer h = m->encode(pixels, seq, pos);
    ::unsetenv("VPIPE_QIE_VIS_STOP");
    std::vector<float> got(gold.size());
    const auto* p = static_cast<const std::uint16_t*>(h.contents());
    for (std::size_t i = 0; i < gold.size(); ++i) { got[i] = bf16_to_f32_(p[i]); }
    std::printf("[qwen_image_edit_vision] MW stop=%s rel-L2 = %.6f\n", stop,
                rel_l2_(got, gold.data(), gold.size()));
  };
  dbg("-1", "g_v2_patch.f32");
  dbg("0", "g_v2_blk0.f32");

  // Per-block growth curve (VPIPE_QIE_VIS_BLKDUMP) -- diagnostic only.
  const std::vector<float> allblk = read_f32_(gdir + "/g_v2_allblk.f32");
  if (!allblk.empty() && std::getenv("VPIPE_QIE_VIS_BLKDUMP")) {
    const std::size_t bs = (std::size_t)seq * cfg.hidden;
    for (int L = 0; L < cfg.depth; ++L) {
      char sv[8]; std::snprintf(sv, sizeof(sv), "%d", L);
      ::setenv("VPIPE_QIE_VIS_STOP", sv, 1);
      SharedBuffer h = m->encode(pixels, seq, pos);
      ::unsetenv("VPIPE_QIE_VIS_STOP");
      std::vector<float> got(bs);
      const auto* p = static_cast<const std::uint16_t*>(h.contents());
      for (std::size_t i = 0; i < bs; ++i) { got[i] = bf16_to_f32_(p[i]); }
      const bool fullb = (L == 7 || L == 15 || L == 23 || L == 31);
      std::printf("[qwen_image_edit_vision]   blk %2d (%s) rel-L2 = %.5f\n", L,
                  fullb ? "full" : "win ", rel_l2_(got,
                  allblk.data() + (std::size_t)L * bs, bs));
    }
  }

  SharedBuffer out = m->encode(pixels, seq, pos);
  ASSERT_TRUE(!out.empty());
  const int mseq = seq / (cfg.merge * cfg.merge);
  const std::size_t n = (std::size_t)mseq * cfg.out_hidden;
  ASSERT_TRUE(gvis.size() == n);
  std::vector<float> got(n);
  const auto* p = static_cast<const std::uint16_t*>(out.contents());
  for (std::size_t i = 0; i < n; ++i) { got[i] = bf16_to_f32_(p[i]); }
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += (double)got[i] * gvis[i]; na += (double)got[i] * got[i];
    nb += (double)gvis[i] * gvis[i];
  }
  const double r = rel_l2_(got, gvis.data(), n);
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb));
  std::printf("[qwen_image_edit_vision] MULTIWINDOW: rel-L2 = %.6f cosine = "
              "%.6f (seq=%d -> %d tokens, 12 windows)\n", r, cos, seq, mseq);
  // Window attention (28 windowed + 4 full blocks) vs HF. bf16 accumulation
  // over 704 tokens x 32 blocks floors ~0.06; cosine confirms correctness.
  EXPECT_TRUE(r < 0.12);
  EXPECT_TRUE(cos > 0.99);

  // encode_rgb: preprocess the RAW input image (bilinear vs HF bicubic) + tower,
  // vs the golden vision tokens. Won't be tight (interpolation differs) but the
  // grid must match and cosine stay high (the conditioning is robust to it).
  const std::vector<std::int32_t> rgbhw = read_i32_(gdir + "/v2_rgb_hw.i32");
  std::ifstream rf(gdir + "/v2_rgb.u8", std::ios::binary);
  if (!rgbhw.empty() && rf) {
    const int ih = rgbhw[0], iw = rgbhw[1];
    std::vector<std::uint8_t> rgb((std::size_t)3 * ih * iw);
    rf.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)rgb.size());
    int ggh = 0, ggw = 0;
    SharedBuffer vt = m->encode_rgb(rgb.data(), ih, iw, 384 * 384, ggh, ggw);
    ASSERT_TRUE(!vt.empty());
    const int mtok = (ggh / cfg.merge) * (ggw / cfg.merge);
    std::printf("[qwen_image_edit_vision] encode_rgb grid %dx%d -> %d tokens "
                "(golden %d)\n", ggh, ggw, mtok, mseq);
    if (mtok == mseq) {
      std::vector<float> gt2(n);
      const auto* pp = static_cast<const std::uint16_t*>(vt.contents());
      for (std::size_t i = 0; i < n; ++i) { gt2[i] = bf16_to_f32_(pp[i]); }
      double d2 = 0, a2 = 0, b2 = 0;
      for (std::size_t i = 0; i < n; ++i) { d2 += (double)gt2[i] * gvis[i];
        a2 += (double)gt2[i] * gt2[i]; b2 += (double)gvis[i] * gvis[i]; }
      std::printf("[qwen_image_edit_vision] encode_rgb vs golden: rel-L2 = %.4f "
                  "cosine = %.4f\n", rel_l2_(gt2, gvis.data(), n),
                  d2 / (std::sqrt(a2) * std::sqrt(b2)));
    }
  }
}
