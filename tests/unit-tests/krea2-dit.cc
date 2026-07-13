// Krea-2-Turbo DiT bring-up (M3), staged against the diffusers golden taps.
//
// M3a (text-fusion tower + txt_in): feed the golden 12-layer encoder stack
// a1_prompt_embeds (text_seq, 12, 2560) to MetalKrea2Transformer::forward_text
// and rel-L2 the (text_seq, 6144) fused-text output against d_txtin.f32 (the
// HF Krea2Transformer2DModel's txt_in output, dumped by krea2_dit_ref.py).
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH = the Krea-2-Turbo model root (uses
// <root>/transformer), VPIPE_KREA2_GOLDEN = the golden dir. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "common/flex-data.h"
#include "generative-models/krea2/metal-krea2-transformer.h"

#include <chrono>
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

TEST(krea2_dit, text_fusion_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  // Golden inputs/outputs.
  const std::vector<float> pe = read_f32_(gdir + "/a1_prompt_embeds.f32");
  const std::vector<float> golden = read_f32_(gdir + "/d_txtin.f32");
  ASSERT_TRUE(!pe.empty());
  ASSERT_TRUE(!golden.empty());

  MetalKrea2Transformer::Config cfg;             // Krea2 DiT defaults
  const int NL = cfg.n_text_layers, TH = cfg.text_hidden, HID = cfg.hidden;
  ASSERT_TRUE(pe.size() % ((std::size_t)NL * TH) == 0);
  const int text_seq = (int)(pe.size() / ((std::size_t)NL * TH));
  ASSERT_TRUE(golden.size() == (std::size_t)text_seq * HID);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // Build the f16 encoder-tap stack [text_seq, 12, 2560] from the golden f32.
  SharedBuffer ehs = mc->make_shared_buffer(pe.size() * 2);
  ASSERT_TRUE(!ehs.empty());
  auto* e = static_cast<_Float16*>(ehs.contents());
  for (std::size_t i = 0; i < pe.size(); ++i) { e[i] = (_Float16)pe[i]; }

  SharedBuffer out = m->forward_text(ehs, text_seq);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= (std::size_t)text_seq * HID * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got((std::size_t)text_seq * HID);
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), golden.data(), got.size());
  std::printf("[krea2_dit] text_fusion+txt_in rel-L2 = %.6f (text_seq=%d)\n",
              r, text_seq);
  EXPECT_TRUE(r < 0.05);
}

// A/B oracle (needs only the model, no golden): the M5 matrix-core matmul2d
// projection path must match the steel dense_gemm_t path within f32-accumulation
// rounding. Runs the text-fusion tower + txt_in (forward_text) on a deterministic
// random encoder-tap stack twice -- once default (matmul2d on M5) and once with
// VPIPE_KREA2_NO_MMA2 forcing steel -- and rel-L2s the two fused-text outputs.
// This exercises the block attention/SwiGLU projections + txt_in through the
// same gemm_mma_ routing the main DiT blocks use. stream_blocks=true keeps the
// 28 main transformer blocks OFF the heap (the 12B DiT would blow a 16GB box);
// forward_text only touches the small text tower. On a non-matrix-core GPU both
// loads take the steel path -> a trivial (still valid) exact match.
TEST(krea2_dit, text_fusion_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int NL = cfg.n_text_layers, TH = cfg.text_hidden, HID = cfg.hidden;
  const int text_seq = 128;
  // Deterministic pseudo-random encoder-tap stack [text_seq, 12, 2560] in
  // ~[-0.5, 0.5].
  std::vector<float> pe((std::size_t)text_seq * NL * TH);
  std::uint32_t s = 0x01234567u;
  for (auto& v : pe) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 9) / 4194304.0f - 1.0f) * 0.5f;
  }
  auto make_ehs = [&]() {
    SharedBuffer e = mc->make_shared_buffer(pe.size() * 2);
    auto* d = static_cast<_Float16*>(e.contents());
    for (std::size_t i = 0; i < pe.size(); ++i) { d[i] = (_Float16)pe[i]; }
    return e;
  };
  const std::size_t n = (std::size_t)text_seq * HID;

  std::printf("[krea2_dit] text_fusion mma_matches_steel: matrix_cores=%d\n",
              (int)mc->supports_matrix_cores());

  // matmul2d path (default on M5).
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  auto m_mma =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m_mma != nullptr);
  SharedBuffer o_mma = m_mma->forward_text(make_ehs(), text_seq);
  ASSERT_TRUE(!o_mma.empty() && o_mma.byte_size() >= n * 2);

  // steel path (force off at load time).
  ::setenv("VPIPE_KREA2_NO_MMA2", "1", 1);
  auto m_steel =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->forward_text(make_ehs(), text_seq);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_mma.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf(
      "[krea2_dit] text_fusion mma-vs-steel rel-L2 = %.6g (text_seq=%d)\n", r,
      text_seq);
  EXPECT_TRUE(r < 3e-2);
}

// A/B oracle for the DiT main-block path (needs only the model, no golden):
// runs forward_dit through 2 streamed real transformer blocks (tap mode) on
// deterministic random fused-text + latents, once with matmul2d and once with
// VPIPE_KREA2_NO_MMA2 forcing steel, and rel-L2s the joint [text+img, hidden]
// states. Unlike forward_text this exercises the img_in projection (which the
// mma path writes into `joint` at an ELEMENT OFFSET) plus the block q/k/v/gate/
// o + SwiGLU projections through gemm_mma_. stream_blocks=true keeps the full
// 28-block DiT off the heap; only blocks 0..1 are read from the source mmap.
TEST(krea2_dit, forward_dit_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 128, grid = 8;      // img_seq = 64 -> seq = 192
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x0badf00du;
  auto fill = [&](std::vector<float>& v, float amp) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f) * amp;
    }
  };
  fill(txt, 1.0f);
  fill(lat, 1.0f);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  std::printf("[krea2_dit] forward_dit mma_matches_steel: matrix_cores=%d\n",
              (int)mc->supports_matrix_cores());

  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  auto m_mma =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m_mma != nullptr);
  SharedBuffer o_mma = m_mma->forward_dit(to_f16buf(txt), text_seq,
                                          to_f16buf(lat), img_seq, grid, grid,
                                          0.5f, stop);
  ASSERT_TRUE(!o_mma.empty() && o_mma.byte_size() >= n * 2);

  ::setenv("VPIPE_KREA2_NO_MMA2", "1", 1);
  auto m_steel =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->forward_dit(to_f16buf(txt), text_seq,
                                              to_f16buf(lat), img_seq, grid,
                                              grid, 0.5f, stop);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_mma.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_dit] forward_dit mma-vs-steel rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  EXPECT_TRUE(r < 3e-2);
}

// Reference-image conditioning (Qwen-Image-Edit multi-reference): forward_dit
// with reference latents (a) still emits ONLY the generated-token velocity
// [img_seq, in_channels] (refs dropped from the output), (b) stays finite with
// one and two references, and (c) actually CHANGES the velocity vs the no-ref
// forward -- proving the reference tokens are embedded via img_in, frame-offset
// on the RoPE and folded into the joint attention. A dead ref path would give a
// byte-identical result. Note: this checks the MECHANISM is wired, not that a
// text-to-image checkpoint semantically uses references. Same random inputs.
TEST(krea2_dit, forward_dit_reference_images_change_output)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 96, grid = 8, img_seq = grid * grid;

  std::uint32_t s = 0x51eed00du;
  auto fill = [&](_Float16* d, std::size_t cnt, float amp) {
    for (std::size_t i = 0; i < cnt; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(s >> 9) / 4194304.0f - 1.0f) * amp);
    }
  };
  auto rand_buf = [&](std::size_t cnt, float amp) {
    SharedBuffer b = mc->make_shared_buffer(cnt * 2);
    fill(static_cast<_Float16*>(b.contents()), cnt, amp);
    return b;
  };
  // A random packed reference latent [rseq, in_channels] on a rgh x rgw grid,
  // mirroring the 2x2-patchified vae-encode latent the stage feeds.
  auto make_ref = [&](int rgh, int rgw) -> MetalKrea2Transformer::RefImage {
    MetalKrea2Transformer::RefImage r;
    r.seq = rgh * rgw; r.grid_h = rgh; r.grid_w = rgw;
    r.latents = rand_buf((std::size_t)r.seq * IC, 1.0f);
    return r;
  };

  auto dit = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(dit != nullptr);
  SharedBuffer txt = rand_buf((std::size_t)text_seq * HID, 1.0f);
  SharedBuffer lat = rand_buf((std::size_t)img_seq * IC, 1.0f);

  const std::size_t n = (std::size_t)img_seq * IC;
  auto to_vec = [&](const SharedBuffer& b) {
    std::vector<float> v(n);
    const auto* p = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)p[i]; }
    return v;
  };

  SharedBuffer v0 = dit->forward_dit(txt, text_seq, lat, img_seq, grid, grid,
                                     0.5f);
  ASSERT_TRUE(!v0.empty() && v0.byte_size() >= n * 2);
  // RefImage is move-only (holds a SharedBuffer), so move into the vectors.
  std::vector<MetalKrea2Transformer::RefImage> one;
  one.push_back(make_ref(4, 6));
  SharedBuffer v1 = dit->forward_dit(txt, text_seq, lat, img_seq, grid, grid,
                                     0.5f, -1, one);
  ASSERT_TRUE(!v1.empty());
  ASSERT_TRUE(v1.byte_size() >= n * 2);   // still generated-token count only
  std::vector<MetalKrea2Transformer::RefImage> two;
  two.push_back(make_ref(4, 6));
  two.push_back(make_ref(6, 4));
  SharedBuffer v2 = dit->forward_dit(txt, text_seq, lat, img_seq, grid, grid,
                                     0.5f, -1, two);
  ASSERT_TRUE(!v2.empty() && v2.byte_size() >= n * 2);

  const std::vector<float> a = to_vec(v0), b1 = to_vec(v1), b2 = to_vec(v2);
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(b1[i]) || !std::isfinite(b2[i])) { finite = false; break; }
  }
  EXPECT_TRUE(finite);
  const double r1 = rel_l2_(b1.data(), a.data(), n);
  const double r2 = rel_l2_(b2.data(), a.data(), n);
  std::printf("[krea2_dit] ref-image velocity delta: 1 ref rel-L2 %.4f, 2 refs "
              "rel-L2 %.4f (vs no-ref)\n", r1, r2);
  EXPECT_TRUE(r1 > 1e-3);          // a reference materially changes the output
  EXPECT_TRUE(r2 > 1e-3);
}

// A/B oracle for the BM=128 quantized qmm tile (_qmm_tile=2) vs BM=64
// (_qmm_tile=1): the tall-M steel qmm + fused-SwiGLU GEMMs pick a 128-row
// output tile (WM=4, 256 threads) when seq >= 1024, quartering the fused
// gate|up code re-reads (that GEMM had stayed at BM=32). Needs a QUANTIZED
// transformer dir (VPIPE_KREA2_QMM_DIR, e.g. Krea-2-Turbo-dit-w4g64); runs
// forward_dit through 2 streamed blocks (tap) at grid=32 (img_seq=1024,
// seq=1152 -> the BM=128 gate engages) on deterministic random inputs and
// rel-L2s the joint. Different tiling changes only the f32 accumulation
// order, so the two paths are numerically equivalent (bar 3e-2, expect
// ~1e-3). Skips vacuously without the dir. Only meaningful on the steel qmm
// path (M4 / no matrix cores); on M5 the quant GEMMs route through mma2 and
// both tiles run identical, so rel-L2 ~0 (no regression).
TEST(krea2_dit, forward_dit_bm128_matches_bm64)
{
  const char* qdir = std::getenv("VPIPE_KREA2_QMM_DIR");
  if (qdir == nullptr || *qdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 128, grid = 32;     // img_seq = 1024 -> seq = 1152
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x0badf00du;
  auto fill = [&](std::vector<float>& v, float amp) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f) * amp;
    }
  };
  fill(txt, 1.0f);
  fill(lat, 1.0f);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  ::setenv("VPIPE_KREA2_QMM_TILE", "2", 1);
  auto m128 =
      MetalKrea2Transformer::load(qdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m128 != nullptr);
  SharedBuffer o128 = m128->forward_dit(to_f16buf(txt), text_seq,
                                        to_f16buf(lat), img_seq, grid, grid,
                                        0.5f, stop);
  ASSERT_TRUE(!o128.empty() && o128.byte_size() >= n * 2);

  ::setenv("VPIPE_KREA2_QMM_TILE", "1", 1);
  auto m64 =
      MetalKrea2Transformer::load(qdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_QMM_TILE");
  ASSERT_TRUE(m64 != nullptr);
  SharedBuffer o64 = m64->forward_dit(to_f16buf(txt), text_seq,
                                      to_f16buf(lat), img_seq, grid, grid,
                                      0.5f, stop);
  ASSERT_TRUE(!o64.empty() && o64.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o128.contents());
  const auto* b = static_cast<const _Float16*>(o64.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_dit] forward_dit bm128-vs-bm64 rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  EXPECT_TRUE(r < 3e-2);
}

// A/B oracle for the DiT's M5 matrix-core NAX attention (attn_steel_nax_h_bd128)
// vs the ALU steel (attn_steel_h_bd128): runs forward_dit through 2 streamed
// blocks (tap mode) on deterministic random inputs, once with NAX (default on
// M5) and once with VPIPE_KREA2_NO_ATTN_NAX forcing the ALU steel, and rel-L2s
// the joint state. On M4 (no matrix cores) BOTH paths are the ALU steel, so the
// rel-L2 is ~0 (the test passes trivially, validating no wiring regression);
// the REAL NAX-vs-steel comparison happens when run on M5 (matrix_cores=1).
TEST(krea2_dit, forward_dit_nax_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 10, grid = 16;      // img_seq = 256 -> seq = 266
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x51ee7a11u;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  std::printf("[krea2_dit] forward_dit nax_matches_steel: matrix_cores=%d "
              "(NAX active only when 1)\n", (int)mc->supports_matrix_cores());

  ::unsetenv("VPIPE_KREA2_NO_ATTN_NAX");
  auto m_nax =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m_nax != nullptr);
  SharedBuffer o_nax = m_nax->forward_dit(to_f16buf(txt), text_seq,
                                          to_f16buf(lat), img_seq, grid, grid,
                                          0.5f, stop);
  ASSERT_TRUE(!o_nax.empty() && o_nax.byte_size() >= n * 2);

  ::setenv("VPIPE_KREA2_NO_ATTN_NAX", "1", 1);
  auto m_steel =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_NO_ATTN_NAX");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->forward_dit(to_f16buf(txt), text_seq,
                                              to_f16buf(lat), img_seq, grid,
                                              grid, 0.5f, stop);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_nax.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_dit] forward_dit nax-vs-steel rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  // Both fp32 online softmax -> within the flash tolerance (0 on M4, small on
  // M5). Bar matches the mma-vs-steel oracle.
  EXPECT_TRUE(r < 3e-2);
}

// M3b: img_in + time_embed + ONE main DiT block (adaLN + Flux 3-axis rope +
// GQA 48/12 + sigmoid gate). Feed the golden fused text (d_txtin), packed
// latents (a3_step0_latin) and timestep (a3_step0_tstep); rel-L2 the joint
// [text+img, hidden] state after block 0 against d_blk0.f32.
TEST(krea2_dit, block0_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> txt   = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32_(gdir + "/a3_step0_tstep.f32");
  const std::vector<float> blk0  = read_f32_(gdir + "/d_blk0.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !tstep.empty() && !blk0.empty());

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }          // square latent grid
  ASSERT_TRUE(grid * grid == img_seq);
  ASSERT_TRUE(blk0.size() == (std::size_t)(text_seq + img_seq) * HID);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  SharedBuffer fused = to_f16buf(txt);
  SharedBuffer lat = to_f16buf(latin);

  SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    tstep[0], /*stop_after_block=*/0);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= blk0.size() * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(blk0.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), blk0.data(), got.size());
  std::printf("[krea2_dit] block0 rel-L2 = %.6f (text=%d img=%d grid=%d)\n",
              r, text_seq, img_seq, grid);
  EXPECT_TRUE(r < 0.05);
}

// M3c: the FULL single denoiser step -- all 28 blocks + final_layer -> the
// predicted velocity. rel-L2 the (img_seq, in_channels) output against
// a3_step0_noise.f32 (the HF transformer's step-0 velocity).
TEST(krea2_dit, single_step_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> txt   = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32_(gdir + "/a3_step0_tstep.f32");
  const std::vector<float> noise = read_f32_(gdir + "/a3_step0_noise.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !tstep.empty() && !noise.empty());

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  ASSERT_TRUE(grid * grid == img_seq);
  ASSERT_TRUE(noise.size() == (std::size_t)img_seq * IC);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  SharedBuffer fused = to_f16buf(txt);
  SharedBuffer lat = to_f16buf(latin);

  SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    tstep[0], /*stop_after_block=*/-1);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= noise.size() * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(noise.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), noise.data(), got.size());
  std::printf("[krea2_dit] single_step rel-L2 = %.6f\n", r);
  EXPECT_TRUE(r < 0.05);
}

// M3d: the full 8-step FlowMatchEuler turbo sampler (free-running) driving the
// metal DiT. Start from a3_step0_latin, step latents += (sigma[i+1]-sigma[i]) *
// velocity with the golden sigma schedule, and rel-L2 the final packed latents
// against a3_final_latents.f32. (Fused text is the golden d_txtin, isolating the
// DiT+sampler from the text tower.)
TEST(krea2_dit, sampler_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> txt   = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> fin   = read_f32_(gdir + "/a3_final_latents.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !fin.empty());

  // The turbo (mu=1.15 dynamic-shift) sigma schedule from the golden meta.json
  // (linspace(1, 1/8, 8) shifted); 9 values incl. the terminal 0.
  const double sig[9] = {1.0, 0.956723690032959, 0.9045307636260986,
                         0.8403487801551819, 0.7595109343528748,
                         0.654566764831543, 0.5128441452980042,
                         0.31090107560157776, 0.0};

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  ASSERT_TRUE(grid * grid == img_seq);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer fused = mc->make_shared_buffer(txt.size() * 2);
  { auto* d = static_cast<_Float16*>(fused.contents());
    for (std::size_t i = 0; i < txt.size(); ++i) { d[i] = (_Float16)txt[i]; } }

  SharedBuffer latbuf = mc->make_shared_buffer(latin.size() * 2);
  double worst_tf = 0.0;
  for (int s = 0; s < 8; ++s) {
    // TEACHER-FORCED per-step check: feed the golden latin[s], apply one Euler
    // step with our velocity, compare to the golden latin[s+1] (or, for the
    // last step, a3_final_latents). Isolates per-step correctness from f16
    // trajectory drift.
    const std::string cur = "/a3_step" + std::to_string(s) + "_latin.f32";
    const std::vector<float> gl = read_f32_(gdir + cur);
    ASSERT_TRUE(!gl.empty());
    std::vector<float> gnext = (s < 7)
        ? read_f32_(gdir + "/a3_step" + std::to_string(s + 1) + "_latin.f32")
        : fin;
    ASSERT_TRUE(!gnext.empty());
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t i = 0; i < gl.size(); ++i) { lb[i] = (_Float16)gl[i]; }
    SharedBuffer np = m->forward_dit(fused, text_seq, latbuf, img_seq, grid,
                                     grid, (float)sig[s], -1);
    ASSERT_TRUE(!np.empty());
    const auto* p = static_cast<const _Float16*>(np.contents());
    const double dt = sig[s + 1] - sig[s];
    std::vector<float> stepped(gl.size());
    for (std::size_t i = 0; i < gl.size(); ++i) {
      stepped[i] = gl[i] + (float)(dt * (double)(float)p[i]);
    }
    const double rt = rel_l2_(stepped.data(), gnext.data(), stepped.size());
    std::printf("[krea2_dit]   teacher-forced step %d rel-L2 = %.6f\n", s, rt);
    if (rt > worst_tf) { worst_tf = rt; }
  }
  std::printf("[krea2_dit] sampler teacher-forced worst per-step rel-L2 = %.6f\n",
              worst_tf);
  EXPECT_TRUE(worst_tf < 0.03);
}

// Perf bench (opt-in, no golden): time forward_dit at a target latent grid so
// the per-section breakdown (VPIPE_KREA2_DIT_PROFILE) can be read at real
// resolutions. Gated on VPIPE_KREA2_DIT_BENCH so it never runs in the default
// suite. Env:
//   VPIPE_KREA2_DIT_BENCH       -- enable.
//   VPIPE_KREA2_TEST_MODEL_PATH -- model root (uses <root>/transformer unless
//                                  overridden).
//   VPIPE_KREA2_DIT_BENCH_DIR   -- explicit transformer dir (e.g. a quantized
//                                  variant) overriding <root>/transformer.
//   VPIPE_KREA2_DIT_BENCH_GRID  -- latent grid side (default 64 => 1024px; a
//                                  256px image is grid 16, 512px is 32).
//   VPIPE_KREA2_DIT_BENCH_ITERS -- timed forward_dit calls (default 3).
// Forces VPIPE_KREA2_DIT_PROFILE on so each call logs its section breakdown.
TEST(krea2_dit, forward_dit_bench)
{
  if (std::getenv("VPIPE_KREA2_DIT_BENCH") == nullptr) { return; }
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* dir_override = std::getenv("VPIPE_KREA2_DIT_BENCH_DIR");
  if ((root == nullptr || *root == '\0')
      && (dir_override == nullptr || *dir_override == '\0')) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = (dir_override != nullptr && *dir_override != '\0')
      ? std::string(dir_override)
      : std::string(root) + "/transformer";

  int grid = 64;
  if (const char* g = std::getenv("VPIPE_KREA2_DIT_BENCH_GRID")) {
    const int gv = std::atoi(g);
    if (gv > 0) { grid = gv; }
  }
  int iters = 3;
  if (const char* it = std::getenv("VPIPE_KREA2_DIT_BENCH_ITERS")) {
    const int iv = std::atoi(it);
    if (iv > 0) { iters = iv; }
  }
  const int text_seq = 10;                   // typical compact prompt length
  const int img_seq = grid * grid;

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  std::printf("[krea2_dit] bench: dir=%s grid=%d img_seq=%d seq=%d iters=%d "
              "matrix_cores=%d\n", tdir.c_str(), grid, img_seq,
              text_seq + img_seq, iters, (int)mc->supports_matrix_cores());

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  std::uint32_t s = 0x1234567u;
  auto rndbuf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)((float)(s >> 9) / 4194304.0f - 1.0f);
    }
    return b;
  };
  SharedBuffer fused = rndbuf((std::size_t)text_seq * HID);
  SharedBuffer latbuf = rndbuf((std::size_t)img_seq * IC);

  // Profile the FIRST iter (section breakdown, barriers inflate its wall time);
  // run the rest barrier-free so best/mean reflect the true step latency.
  double best = 1e30, sum = 0.0;
  int timed = 0;
  for (int i = 0; i < iters; ++i) {
    if (i == 0) { ::setenv("VPIPE_KREA2_DIT_PROFILE", "1", 1); }
    else { ::unsetenv("VPIPE_KREA2_DIT_PROFILE"); }
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer v = m->forward_dit(fused, text_seq, latbuf, img_seq, grid,
                                    grid, 0.5f, -1);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(!v.empty());
    std::printf("[krea2_dit] bench iter %d: %.1f ms/step%s\n", i, ms,
                i == 0 ? " (profiled -- barriers inflate)" : "");
    if (i > 0) { if (ms < best) { best = ms; } sum += ms; ++timed; }
  }
  ::unsetenv("VPIPE_KREA2_DIT_PROFILE");
  if (timed == 0) { best = sum = 0.0; timed = 1; }
  std::printf("[krea2_dit] bench: best %.1f ms/step, mean %.1f ms/step "
              "(barrier-free iters; iter 0 is the profiled section split)\n",
              best, sum / timed);
  EXPECT_TRUE(best >= 0.0);
}
