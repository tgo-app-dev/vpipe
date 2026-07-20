// Qwen-Image-Edit-2511 VAE (M1): the model's `AutoencoderKLQwenImage` is the
// SAME VAE as Krea-2, so it runs on the existing MetalKrea2Vae with NO new code
// -- this test proves it against the diffusers golden.
//
//  * encode: feed g_vae_input [3,256,256] in [-1,1] -> whitened latent
//    [16,32,32], rel-L2 vs g_vae_enc (HF vae.encode.mode then whiten).
//  * decode: feed the RAW (un-whitened) latent g_vae_declat [16,32,32] ->
//    RGB [3,256,256], rel-L2 vs g_vae_dec (HF vae.decode).
//
// Env: VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH = model root (uses <root>/vae),
// VPIPE_QWEN_IMAGE_EDIT_GOLDEN = golden dir (vae_golden.py output). Skips unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"

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

SharedBuffer
upload_f16_(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  auto* d = static_cast<_Float16*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
  return b;
}

// Per-channel latent statistics (vae/config.json latents_mean/std) -- identical
// to the Krea-2 Qwen-Image VAE. The encode() whitening needs these populated
// (the bare Config leaves them empty; decode() never touches them).
const float kMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f,
                         0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f,
                         0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f,
                         -0.2921f};
const float kStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f,
                        2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f,
                        1.6382f, 1.1253f, 2.8251f, 1.9160f};

MetalKrea2Vae::Config
vae_config_()
{
  MetalKrea2Vae::Config cfg;
  for (int c = 0; c < 16; ++c) {
    cfg.latents_mean.push_back(kMean[c]);
    cfg.latents_std.push_back(kStd[c]);
  }
  return cfg;
}

struct Env {
  std::string root, gdir;
  bool ok() const { return !root.empty() && !gdir.empty(); }
};
Env env_()
{
  const char* r = std::getenv("VPIPE_QWEN_IMAGE_EDIT_TEST_MODEL_PATH");
  const char* g = std::getenv("VPIPE_QWEN_IMAGE_EDIT_GOLDEN");
  Env e;
  if (r && *r) { e.root = r; }
  if (g && *g) { e.gdir = g; }
  return e;
}

}  // namespace

TEST(qwen_image_edit_vae, encode_matches_golden)
{
  const Env e = env_();
  if (!e.ok()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> vin = read_f32_(e.gdir + "/g_vae_input.f32");
  const std::vector<float> gl  = read_f32_(e.gdir + "/g_vae_enc.f32");
  if (vin.empty() || gl.empty()) { return; }
  ASSERT_TRUE(vin.size() % 3 == 0);
  const std::size_t hw = vin.size() / 3;
  int H = 1;
  while ((std::size_t)H * H < hw) { ++H; }
  ASSERT_TRUE((std::size_t)H * H == hw);
  const int W = H;

  MetalKrea2Vae::Config cfg = vae_config_();
  auto m = MetalKrea2Vae::load(e.root + "/vae", mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  ASSERT_TRUE(m->has_encoder());

  SharedBuffer lat = m->encode(upload_f16_(mc, vin), H, W);
  ASSERT_TRUE(!lat.empty());
  ASSERT_TRUE(lat.byte_size() >= gl.size() * 2);
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  std::vector<float> got(gl.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)lp[i]; }

  const double r = rel_l2_(got.data(), gl.data(), got.size());
  std::printf("[qwen_image_edit_vae] encode rel-L2 = %.6f (%dx%d -> [16,%d,%d])\n",
              r, H, W, H / 8, W / 8);
  EXPECT_TRUE(r < 0.05);
}

TEST(qwen_image_edit_vae, decode_matches_golden)
{
  const Env e = env_();
  if (!e.ok()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> lat = read_f32_(e.gdir + "/g_vae_declat.f32");
  const std::vector<float> img = read_f32_(e.gdir + "/g_vae_dec.f32");
  if (lat.empty() || img.empty()) { return; }

  MetalKrea2Vae::Config cfg = vae_config_();
  const int Cz = cfg.z_dim;
  ASSERT_TRUE(lat.size() % (std::size_t)Cz == 0);
  const std::size_t hw = lat.size() / (std::size_t)Cz;
  int h8 = 1;
  while ((std::size_t)h8 * h8 < hw) { ++h8; }
  ASSERT_TRUE((std::size_t)h8 * h8 == hw);
  const int w8 = h8, H = h8 * 8, W = w8 * 8;
  ASSERT_TRUE(img.size() == (std::size_t)3 * H * W);

  auto m = MetalKrea2Vae::load(e.root + "/vae", mc, cfg);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer out = m->decode(upload_f16_(mc, lat), h8, w8);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= img.size() * 2);
  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(img.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), img.data(), got.size());
  std::printf("[qwen_image_edit_vae] decode rel-L2 = %.6f (%dx%d)\n", r, H, W);
  EXPECT_TRUE(r < 0.05);
}
