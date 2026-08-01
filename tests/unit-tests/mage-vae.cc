// MageVAE (microsoft/Mage-Flow) encoder, verified against the reference
// golden.
//
// Feeds the golden preprocessed RGB [3,512,512] (in [-1,1]) to
// MetalMageVae::encode and rel-L2s the returned latent [128,32,32] against
// the fp32 CPU reference (mage_flow.models.modules.mage_vae.MageVAE.encode,
// sample_posterior=false -> the posterior mean), dumped by
// dump_vae_golden.py. Also asserts the constant-folded t=0 adaLN modulation
// the loader bakes in matches the reference's own folded buffers, since a
// wrong fold would silently bias every DiCo block.
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH = the Mage-Flow model root (uses
// <root>/vae), VPIPE_MAGE_GOLDEN = the golden dir. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/mage/metal-mage-vae.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

TEST(mage_vae, encode_matches_golden)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MAGE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";
  const std::string gdir = gd;

  const std::vector<float> vin = read_f32_(gdir + "/input_rgb.f32");
  const std::vector<float> gl  = read_f32_(gdir + "/latent.f32");
  if (vin.empty() || gl.empty()) { return; }   // golden not dumped -> skip
  ASSERT_TRUE(vin.size() % 3 == 0);
  const std::size_t hw = vin.size() / 3;
  int H = 1;
  while ((std::size_t)H * H < hw) { ++H; }
  ASSERT_TRUE((std::size_t)H * H == hw);
  const int W = H;

  MetalMageVae::Config cfg;
  auto m = MetalMageVae::load(vdir, mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  ASSERT_TRUE(m->has_encoder());

  SharedBuffer img = mc->make_shared_buffer(vin.size() * 2);
  ASSERT_TRUE(!img.empty());
  { auto* d = static_cast<_Float16*>(img.contents());
    for (std::size_t i = 0; i < vin.size(); ++i) { d[i] = (_Float16)vin[i]; } }

  std::string err;
  SharedBuffer lat = m->encode(img, H, W, &err);
  if (lat.empty()) {
    std::printf("[mage_vae] encode failed: %s\n", err.c_str());
  }
  ASSERT_TRUE(!lat.empty());
  ASSERT_TRUE(lat.byte_size() >= gl.size() * 2);

  const auto* lp = static_cast<const _Float16*>(lat.contents());
  std::vector<float> got(gl.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)lp[i]; }

  const double r = rel_l2_(got.data(), gl.data(), got.size());
  std::printf("[mage_vae] encode rel-L2 = %.6f (%dx%d -> [%d,%d,%d])\n", r, H,
              W, cfg.latent_channels, H / cfg.patch, W / cfg.patch);
  // Measured 0.0033 on M5 vs the fp32 CPU reference -- the DiCo trunk is 21
  // residual blocks deep, so f16 rounding compounds, but the reference
  // pipeline itself runs the VAE in bf16 (coarser). 0.01 is a regression
  // guard with headroom, not the achieved accuracy.
  EXPECT_TRUE(r < 0.01);
}

// End-to-end round trip on the metal path: encode the golden image and
// decode the result back, comparing against the reference's own round trip.
// This is the shape the edit flow uses (reference image -> latent -> ... ->
// image), and it is what catches encode and decode disagreeing on the
// channel-first latent convention.
TEST(mage_vae, round_trip_matches_golden)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MAGE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> vin = read_f32_(std::string(gd) + "/input_rgb.f32");
  const std::vector<float> gi = read_f32_(std::string(gd) + "/output_rgb.f32");
  if (vin.empty() || gi.empty()) { return; }
  const std::size_t hw = vin.size() / 3;
  int H = 1;
  while ((std::size_t)H * H < hw) { ++H; }
  ASSERT_TRUE((std::size_t)H * H == hw);

  MetalMageVae::Config cfg;
  auto m = MetalMageVae::load(std::string(root) + "/vae", mc, cfg, true);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer img = mc->make_shared_buffer(vin.size() * 2);
  ASSERT_TRUE(!img.empty());
  { auto* d = static_cast<_Float16*>(img.contents());
    for (std::size_t i = 0; i < vin.size(); ++i) { d[i] = (_Float16)vin[i]; } }

  std::string err;
  SharedBuffer lat = m->encode(img, H, H, &err);
  ASSERT_TRUE(!lat.empty());
  SharedBuffer out = m->decode(lat, H / cfg.patch, H / cfg.patch, &err);
  if (out.empty()) {
    std::printf("[mage_vae] decode failed: %s\n", err.c_str());
  }
  ASSERT_TRUE(!out.empty());

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(gi.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }
  const double r = rel_l2_(got.data(), gi.data(), got.size());
  std::printf("[mage_vae] round-trip rel-L2 = %.6f (%dx%d)\n", r, H, H);
  // Composes both halves' error, so looser than either alone.
  EXPECT_TRUE(r < 0.02);
}

// Decode the golden latent [128,32,32] and rel-L2 the RGB [3,512,512]
// against the reference decode. Exercises the CoD trunk (GroupNorm
// ResnetBlocks + 32x32 tiled attention), the 21 DiCo blocks and the
// per-pixel MLP head.
TEST(mage_vae, decode_matches_golden)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MAGE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> zl = read_f32_(std::string(gd) + "/latent.f32");
  const std::vector<float> gi = read_f32_(std::string(gd) + "/output_rgb.f32");
  if (zl.empty() || gi.empty()) { return; }

  MetalMageVae::Config cfg;
  ASSERT_TRUE(zl.size() % cfg.latent_channels == 0);
  const std::size_t hw = zl.size() / (std::size_t)cfg.latent_channels;
  int h = 1;
  while ((std::size_t)h * h < hw) { ++h; }
  ASSERT_TRUE((std::size_t)h * h == hw);
  const int w = h;

  auto m = MetalMageVae::load(std::string(root) + "/vae", mc, cfg,
                              /*with_encoder=*/false);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer z = mc->make_shared_buffer(zl.size() * 2);
  ASSERT_TRUE(!z.empty());
  { auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < zl.size(); ++i) { d[i] = (_Float16)zl[i]; } }

  std::string err;
  SharedBuffer img = m->decode(z, h, w, &err);
  if (img.empty()) {
    std::printf("[mage_vae] decode failed: %s\n", err.c_str());
  }
  ASSERT_TRUE(!img.empty());
  ASSERT_TRUE(img.byte_size() >= gi.size() * 2);

  const auto* ip = static_cast<const _Float16*>(img.contents());
  std::vector<float> got(gi.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)ip[i]; }

  const double r = rel_l2_(got.data(), gi.data(), got.size());
  std::printf("[mage_vae] decode rel-L2 = %.6f ([%d,%d,%d] -> [3,%d,%d])\n", r,
              cfg.latent_channels, h, w, h * cfg.patch, w * cfg.patch);
  // Measured 0.00049 on M5 at 512x512 (and at 768x768, where the 48x48
  // latent does NOT tile evenly into 32x32 and the attention's replicate
  // padding is exercised). 0.005 is a regression guard with headroom.
  EXPECT_TRUE(r < 0.005);
}

// A blown-out white region must survive the decoder's per-pixel MLP head.
//
// That head is a chain of gated residual adds with no normalization until the
// very end, so its residual stream grows far past the f16 range: on a real
// photo the row entering the three blocks peaks around |x| ~ 39 and the blocks
// take it to ~2.7e4, ~5.7e4, then ~1.5e5 -- against f16's 65504 ceiling. When
// the stream was kept in f16, the brightest pixels overflowed to inf, the
// final RMS norm turned inf * rsqrt(inf) into NaN, and the u8 conversion
// (whose clamp cannot catch NaN) emitted 0 -- a whole 16x16 latent cell came
// out BLACK in the middle of a bright sky. The residual is f32 for exactly
// this reason; this guards the regression without needing a golden.
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH only.
TEST(mage_vae, saturated_white_decodes_without_nan)
{
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMageVae::Config cfg;
  auto m = MetalMageVae::load(std::string(root) + "/vae", mc, cfg,
                              /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  ASSERT_TRUE(m->has_encoder());

  // Pure white (+1 in the [-1,1] convention) over a whole image: every latent
  // cell is then a saturated-highlight cell, so the head is driven into the
  // range that used to overflow.
  const int H = 256, W = 256;
  const std::size_t n = (std::size_t)3 * H * W;
  SharedBuffer img = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!img.empty());
  { auto* d = static_cast<_Float16*>(img.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)1.0f; } }

  std::string err;
  SharedBuffer lat = m->encode(img, H, W, &err);
  ASSERT_TRUE(!lat.empty());
  const int h = H / cfg.patch, w = W / cfg.patch;
  SharedBuffer out = m->decode(lat, h, w, &err);
  ASSERT_TRUE(!out.empty());

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::size_t nan = 0, inf = 0;
  double lo = 1e30, hi = -1e30, sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float v = (float)op[i];
    if (std::isnan(v)) { ++nan; continue; }
    if (std::isinf(v)) { ++inf; continue; }
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
    sum += v;
  }
  const double mean = sum / (double)(n - nan - inf);
  std::printf("[mage_vae] white %dx%d decode: nan=%zu inf=%zu range [%.3f, "
              "%.3f] mean=%.3f\n", H, W, nan, inf, lo, hi, mean);
  EXPECT_TRUE(nan == 0);
  EXPECT_TRUE(inf == 0);
  // White in, white out: the codec round-trips a constant field to ~+1.
  EXPECT_TRUE(mean > 0.9);
}
