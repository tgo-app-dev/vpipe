// Krea-2-Turbo VAE decoder (M4), verified against the diffusers golden.
//
// Feed the golden un-whitened single-frame latent a4_vae_in [16, 1, 32, 32] to
// MetalKrea2Vae::decode and rel-L2 the decoded RGB [3, 256, 256] against
// a4_image.f32 (the HF AutoencoderKLQwenImage decode, dumped by
// krea2_golden.py). Single-frame decode collapses the 3D causal-conv VAE to a
// 2D conv net (see metal-krea2-vae.h).
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH = the Krea-2-Turbo model root (uses
// <root>/vae), VPIPE_KREA2_GOLDEN = the golden dir. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/vae-decode-stage.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
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

// Per-channel latent statistics (vae/config.json latents_mean/std). Used to
// invert the golden's un-whitening so the stage sees a whitened latent (the
// boundary the DiT stage would emit) and un-whitens it back internally.
const float kMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f,
                         0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f,
                         0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f,
                         -0.2921f};
const float kStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f,
                        2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f,
                        1.6382f, 1.1253f, 2.8251f, 1.9160f};

// Test-only source: emits one preloaded TensorBeat then ends.
class SourceOne : public vpipe::TypedStage<SourceOne> {
public:
  static constexpr const char* kTypeName = "ut-source-one";
  SourceOne(const vpipe::SessionContextIntf* s, std::string id,
            std::vector<vpipe::InEdge> ip, vpipe::FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::unique_ptr<vpipe::BeatPayloadIntf> payload;
  bool done = false;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (!done && payload) {
      done = true;
      co_await ctx.write(0, std::move(payload));
    }
    ctx.signal_done();
    co_return;
  }
};

// Test-only sink: captures every received payload until EOS.
class SinkCapture : public vpipe::TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-vae";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<vpipe::BeatPayloadIntf>> captured;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

}  // namespace

// The VAE ENCODER (img2img): encode the golden preprocessed image b_vae_in
// [3,256,256] and rel-L2 the whitened latent [16,32,32] against b_image_latents
// (the HF vae.encode mode + whiten, from krea2_img2img_golden.py).
TEST(krea2_vae, encode_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";
  const std::string gdir = gd;

  const std::vector<float> vin = read_f32_(gdir + "/b_vae_in.f32");   // [3,H,W]
  const std::vector<float> gl  = read_f32_(gdir + "/b_image_latents.f32");
  if (vin.empty() || gl.empty()) { return; }   // golden not dumped -> skip
  ASSERT_TRUE(vin.size() % 3 == 0);
  const std::size_t hw = vin.size() / 3;
  int H = 1;
  while ((std::size_t)H * H < hw) { ++H; }
  ASSERT_TRUE((std::size_t)H * H == hw);
  const int W = H;

  MetalKrea2Vae::Config cfg;
  for (int c = 0; c < 16; ++c) {
    cfg.latents_mean.push_back(kMean[c]);
    cfg.latents_std.push_back(kStd[c]);
  }
  auto m = MetalKrea2Vae::load(vdir, mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  ASSERT_TRUE(m->has_encoder());

  SharedBuffer img = mc->make_shared_buffer(vin.size() * 2);
  { auto* d = static_cast<_Float16*>(img.contents());
    for (std::size_t i = 0; i < vin.size(); ++i) { d[i] = (_Float16)vin[i]; } }

  SharedBuffer lat = m->encode(img, H, W);
  ASSERT_TRUE(!lat.empty());
  ASSERT_TRUE(lat.byte_size() >= gl.size() * 2);
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  std::vector<float> got(gl.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)lp[i]; }

  const double r = rel_l2_(got.data(), gl.data(), got.size());
  std::printf("[krea2_vae] encode rel-L2 = %.6f (%dx%d -> [16,%d,%d])\n", r, H,
              W, H / 8, W / 8);
  EXPECT_TRUE(r < 0.05);
}

TEST(krea2_vae, decode_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string vdir = std::string(root) + "/vae";
  const std::string gdir = gd;

  // Golden un-whitened latent [16, 1, 32, 32] -> [16, 32, 32]; RGB [3,256,256].
  const std::vector<float> vin = read_f32_(gdir + "/a4_vae_in.f32");
  const std::vector<float> img = read_f32_(gdir + "/a4_image.f32");
  ASSERT_TRUE(!vin.empty());
  ASSERT_TRUE(!img.empty());

  MetalKrea2Vae::Config cfg;                        // Qwen-Image VAE defaults
  const int Cz = cfg.z_dim;                         // 16
  ASSERT_TRUE(vin.size() % (std::size_t)Cz == 0);
  const std::size_t hw = vin.size() / (std::size_t)Cz;   // 32*32
  int h8 = 1;
  while ((std::size_t)h8 * h8 < hw) { ++h8; }
  ASSERT_TRUE((std::size_t)h8 * h8 == hw);          // square latent
  const int w8 = h8;
  const int H = h8 * 8, W = w8 * 8;
  ASSERT_TRUE(img.size() == (std::size_t)3 * H * W);

  auto m = MetalKrea2Vae::load(vdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // Build the f16 channel-first latent [16, h8, w8] from the golden f32.
  SharedBuffer z = mc->make_shared_buffer(vin.size() * 2);
  ASSERT_TRUE(!z.empty());
  { auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < vin.size(); ++i) { d[i] = (_Float16)vin[i]; } }

  SharedBuffer out = m->decode(z, h8, w8);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= img.size() * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(img.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), img.data(), got.size());
  std::printf("[krea2_vae] decode rel-L2 = %.6f (%dx%d)\n", r, H, W);
  EXPECT_TRUE(r < 0.05);
}

// A/B oracle (needs only the model, no golden): the M5 matrix-core matmul2d
// conv path must match the steel dense_gemm_bias path within f32-accumulation
// rounding. Loads the VAE twice -- once default (matmul2d on M5) and once with
// VPIPE_KREA2_NO_MMA2 forcing steel -- decodes the same deterministic random
// latent and rel-L2s the two RGB outputs. This exercises the whole decoder
// (conv_in, mid resblocks + attention, up-blocks, upsample, conv_out) through
// the shared conv_gemm_bias_ (which the encoder path also uses). On a GPU
// without matrix cores both loads take the steel path -> a trivial (but still
// valid) exact match.
TEST(krea2_vae, decode_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";

  MetalKrea2Vae::Config cfg;                        // Qwen-Image VAE defaults
  const int Cz = cfg.z_dim, h8 = 32, w8 = 32;       // -> 256x256 decode
  const std::size_t hw = (std::size_t)h8 * w8;
  // Deterministic pseudo-random latent [Cz, h8, w8] in ~[-3, 3].
  std::vector<float> lat((std::size_t)Cz * hw);
  std::uint32_t s = 0x9e3779b9u;
  for (auto& v : lat) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f;
  }
  auto make_z = [&]() {
    SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
    return z;
  };
  const int H = h8 * 8, W = w8 * 8;
  const std::size_t n = (std::size_t)3 * H * W;

  std::printf("[krea2_vae] mma_matches_steel: matrix_cores=%d\n",
              (int)mc->supports_matrix_cores());

  // matmul2d path (default on M5).
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  auto m_mma = MetalKrea2Vae::load(vdir, mc, cfg);
  ASSERT_TRUE(m_mma != nullptr);
  SharedBuffer o_mma = m_mma->decode(make_z(), h8, w8);
  ASSERT_TRUE(!o_mma.empty() && o_mma.byte_size() >= n * 2);

  // steel path (force off at load time) -- ALSO disable the NAX hardware
  // conv, or the "steel" VAE would still take the conv2d op for its 3x3s.
  ::setenv("VPIPE_KREA2_NO_MMA2", "1", 1);
  ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1);
  auto m_steel = MetalKrea2Vae::load(vdir, mc, cfg);
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  ::unsetenv("VPIPE_VAE_NO_HWCONV");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->decode(make_z(), h8, w8);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_mma.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_vae] decode mma-vs-steel rel-L2 = %.6g (%dx%d)\n", r, H, W);
  EXPECT_TRUE(r < 3e-2);
}

// conv2d-on vs conv2d-off (both matrix-core; off = im2col + matmul2d): correct-
// ness AND perf of the fused 3x3 conv2d. Decodes the same latent with the fused
// conv (default) and with VPIPE_KREA2_NO_CONV2D (im2col path), rel-L2s the two
// RGB outputs (~0: same matmul2d f32 accumulation, only the im2col DRAM scratch
// differs) and times both to report the fused-conv speedup. 512x512 so the
// full-res convs dominate. Timing is a back-to-back relative A/B (same thermal
// state); absolute ms is DVFS-sensitive on M5.
TEST(krea2_vae, decode_conv2d_vs_im2col)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const std::string vdir = std::string(root) + "/vae";

  MetalKrea2Vae::Config cfg;
  const int Cz = cfg.z_dim, h8 = 64, w8 = 64;       // -> 512x512 decode
  const std::size_t hw = (std::size_t)h8 * w8;
  std::vector<float> lat((std::size_t)Cz * hw);
  std::uint32_t s = 0x51ced00du;
  for (auto& v : lat) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f;
  }
  auto make_z = [&]() {
    SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
    return z;
  };
  const int H = h8 * 8, W = w8 * 8;
  const std::size_t n = (std::size_t)3 * H * W;

  // Both legs disable the NAX hardware conv: this test compares the PARKED
  // tgmem-staged fused conv against im2col + matmul2d, and the hw op would
  // otherwise preempt both.
  ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1);
  ::setenv("VPIPE_KREA2_CONV2D", "1", 1);            // opt-in the fused conv
  auto m_conv = MetalKrea2Vae::load(vdir, mc, cfg);
  ::unsetenv("VPIPE_KREA2_CONV2D");
  ASSERT_TRUE(m_conv != nullptr);
  auto m_im2col = MetalKrea2Vae::load(vdir, mc, cfg);  // default: im2col+matmul2d
  ::unsetenv("VPIPE_VAE_NO_HWCONV");
  ASSERT_TRUE(m_im2col != nullptr);

  SharedBuffer o_conv = m_conv->decode(make_z(), h8, w8);
  SharedBuffer o_im = m_im2col->decode(make_z(), h8, w8);
  ASSERT_TRUE(!o_conv.empty() && !o_im.empty());
  const auto* a = static_cast<const _Float16*>(o_conv.contents());
  const auto* b = static_cast<const _Float16*>(o_im.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_vae] conv2d-vs-im2col rel-L2 = %.6g (%dx%d)\n", r, H, W);
  EXPECT_TRUE(r < 3e-2);

  auto bench = [&](MetalKrea2Vae* m) {
    SharedBuffer z = make_z();
    for (int i = 0; i < 2; ++i) { m->decode(z, h8, w8); }   // warm
    const int iters = 6;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { m->decode(z, h8, w8); }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };
  const double ms_conv = bench(m_conv.get());
  const double ms_im = bench(m_im2col.get());
  std::printf("[krea2_vae] decode %dx%d: fused-conv2d %.2f ms  im2col %.2f ms  "
              "(%.2fx)\n", H, W, ms_conv, ms_im, ms_im / ms_conv);
}

// Regression for the >=1024px matmul2d corruption: the MPP matmul2d op returns
// garbage for output rows past M ~= 2^19, so a 1024px decode (M = H*W = 2^20)
// went grey from image row ~512 down. _mma_max_m routes those large-M GEMMs to
// steel. Decode a deterministic random latent at 1024x1024 through the DEFAULT
// (hwconv + mma2) path and assert every horizontal band keeps real signal (a
// collapsed / grey region drops the per-band std from ~0.5 to ~0.01) and no
// output is non-finite. Env-gated on the model like the other decode tests.
TEST(krea2_vae, decode_1024_no_matmul2d_corruption)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const std::string vdir = std::string(root) + "/vae";

  MetalKrea2Vae::Config cfg;
  const int Cz = cfg.z_dim, h8 = 128, w8 = 128;      // -> 1024x1024 decode
  const std::size_t hw = (std::size_t)h8 * w8;
  std::vector<float> lat((std::size_t)Cz * hw);
  std::uint32_t s = 0x51ced00du;
  for (auto& v : lat) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f;
  }
  SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
  {
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
  }
  auto m = MetalKrea2Vae::load(vdir, mc, cfg);       // default: hwconv + mma2
  ASSERT_TRUE(m != nullptr);
  std::string err;
  SharedBuffer o = m->decode(z, h8, w8, &err);
  ASSERT_TRUE(!o.empty());
  const int H = h8 * 8, W = w8 * 8;

  // Per-band std of channel 0 across 8 horizontal bands: a corrupt region
  // collapses to a near-constant grey (std ~0.01), a healthy decode stays
  // ~0.4-0.5. Also flag any non-finite output.
  const auto* a = static_cast<const _Float16*>(o.contents());
  const int NB = 8;
  int nonfinite = 0;
  double min_std = 1e9;
  for (int bidx = 0; bidx < NB; ++bidx) {
    const int r0 = bidx * H / NB, r1 = (bidx + 1) * H / NB;
    double sum = 0, sq = 0; std::size_t cnt = 0;
    for (int r = r0; r < r1; ++r) {
      for (int c = 0; c < W; ++c) {
        const float v = (float)a[(std::size_t)r * W + c];
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        sum += v; sq += (double)v * v; ++cnt;
      }
    }
    const double mean = cnt ? sum / cnt : 0.0;
    const double sd = cnt ? std::sqrt(sq / cnt - mean * mean) : 0.0;
    if (sd < min_std) { min_std = sd; }
  }
  std::printf("[krea2_vae] 1024 decode: min band std = %.4f, non-finite = %d\n",
              min_std, nonfinite);
  EXPECT_TRUE(nonfinite == 0);
  EXPECT_TRUE(min_std > 0.2);          // >> the ~0.01 of a collapsed grey band
}

// The mid-block self-attention on the matrix-core FULL flash kernel
// (sdpa_full_mma2_dN) must match the scalar sdpa_full_f16 it replaces. Decode
// the same latent both ways (VPIPE_KREA2_NO_MMA_ATTN forces scalar) and rel-L2.
TEST(krea2_vae, decode_flash_attn_matches_scalar)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const std::string vdir = std::string(root) + "/vae";
  MetalKrea2Vae::Config cfg;
  const int Cz = cfg.z_dim, h8 = 64, w8 = 64;        // 512x512 -> 4096 mid tokens
  const std::size_t hw = (std::size_t)h8 * w8;
  std::vector<float> lat((std::size_t)Cz * hw);
  std::uint32_t s = 0x51ced00du;
  for (auto& v : lat) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f;
  }
  auto make_z = [&]() {
    SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
    return z;
  };
  auto decode_rgb = [&](bool scalar) {
    if (scalar) { ::setenv("VPIPE_KREA2_NO_MMA_ATTN", "1", 1); }
    auto m = MetalKrea2Vae::load(vdir, mc, cfg);
    ::unsetenv("VPIPE_KREA2_NO_MMA_ATTN");
    std::vector<float> out;
    if (m == nullptr) { return out; }
    SharedBuffer o = m->decode(make_z(), h8, w8);
    if (o.empty()) { return out; }
    const std::size_t n = (std::size_t)3 * h8 * 8 * w8 * 8;
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };
  const std::vector<float> flash = decode_rgb(false);
  const std::vector<float> scal  = decode_rgb(true);
  ASSERT_TRUE(!flash.empty() && flash.size() == scal.size());
  const double r = rel_l2_(flash.data(), scal.data(), flash.size());
  std::printf("[krea2_vae] flash-attn vs scalar rel-L2 = %.6g (512x512)\n", r);
  EXPECT_TRUE(r < 3e-2);
}

// Decode wall-clock at 512 and 1024 (default hwconv + matmul2d path). Warm 2x,
// best-of-5 min (least DVFS-throttled). VPIPE_KREA2_TEST_MODEL_PATH gated.
TEST(krea2_vae, decode_bench)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";
  MetalKrea2Vae::Config cfg;
  auto m = MetalKrea2Vae::load(vdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  const int Cz = cfg.z_dim;
  for (int side : {64, 128}) {                       // 512, 1024
    const std::size_t hw = (std::size_t)side * side;
    SharedBuffer z = mc->make_shared_buffer((std::size_t)Cz * hw * 2);
    std::uint32_t s = 0x51ced00du;
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)Cz * hw; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f);
    }
    std::string err;
    m->decode(z, side, side, &err);                  // warm
    double best = 1e18;
    for (int i = 0; i < 3; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      SharedBuffer o = m->decode(z, side, side, &err);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (!o.empty() && ms < best) { best = ms; }
    }
    std::printf("[krea2_vae] decode %dx%d: %.1f ms (peak est %llu MB)\n",
                side * 8, side * 8, best,
                (unsigned long long)(m->decode_peak_bytes(side, side) >> 20));
  }
}

// End-to-end through the vae-decode stage: feed a WHITENED latent (inverse of
// the golden un-whitening) on the stage's iport, run the pipeline, and check
// the emitted planar U8 RGB image matches a4_image (the stage un-whitens +
// decodes + maps [-1,1] -> U8 internally).
TEST(krea2_vae, decode_stage_end_to_end)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const std::string gdir = gd;

  const std::vector<float> vin = read_f32_(gdir + "/a4_vae_in.f32");
  const std::vector<float> img = read_f32_(gdir + "/a4_image.f32");
  ASSERT_TRUE(!vin.empty() && !img.empty());
  const int Cz = 16;
  const std::size_t hw = vin.size() / (std::size_t)Cz;
  int h8 = 1;
  while ((std::size_t)h8 * h8 < hw) { ++h8; }
  ASSERT_TRUE((std::size_t)h8 * h8 == hw);
  const int w8 = h8, H = h8 * 8, W = w8 * 8;
  ASSERT_TRUE(img.size() == (std::size_t)3 * H * W);

  // Build the whitened latent = (a4_vae_in - mean) / std, channel-first f32.
  auto src = std::make_unique<TensorBeatPayload>();
  src->dtype = TensorBeat::DType::F32;
  src->shape = {Cz, h8, w8};
  src->resize_contiguous(vin.size());
  { float* d = src->as_f32();
    for (int c = 0; c < Cz; ++c) {
      for (std::size_t p = 0; p < hw; ++p) {
        const std::size_t i = (std::size_t)c * hw + p;
        d[i] = (vin[i] - kMean[c]) / kStd[c];
      }
    }
  }

  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto srcu = std::make_unique<SourceOne>(&sess, "src",
                                          std::vector<InEdge>{},
                                          FlexData::make_object());
  srcu->payload = std::move(src);
  auto* srcs = static_cast<SourceOne*>(pl->insert_stage(std::move(srcu)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vaeu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{srcs, 0}}, std::move(cfg));
  auto* vaes = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vaeu)));
  ASSERT_TRUE(vaes->config_error().empty());

  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{vaes, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
              tb->shape[1] == H && tb->shape[2] == W);

  // Map the U8 image back to [-1,1] and rel-L2 vs the golden (U8 quantization
  // adds ~1/255 per pixel on top of the module's decode error).
  const std::uint8_t* u = tb->as_u8();
  std::vector<float> got((std::size_t)3 * H * W);
  for (std::size_t i = 0; i < got.size(); ++i) {
    got[i] = (float)u[i] / 255.0f * 2.0f - 1.0f;
  }
  const double r = rel_l2_(got.data(), img.data(), got.size());
  std::printf("[krea2_vae] stage end-to-end rel-L2 = %.6f\n", r);
  EXPECT_TRUE(r < 0.05);
  EXPECT_TRUE(vaes->images_emitted() == 1);
}
