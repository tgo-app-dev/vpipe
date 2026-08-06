// The Wan video VAE (AutoencoderKLWan), verified against the diffusers
// golden and against the image VAE it generalizes.
//
// Three things need proving, and only the first two are about the net:
//
//   decode_matches_golden -- a T-frame latent decodes to 1 + 4*(T-1) video
//     frames matching the reference. This is the test that actually
//     exercises the time axis: the per-conv feature carry across chunks,
//     the two temporal upsamples, and the first chunk's pass-through.
//   encode_matches_golden -- the mirror, including the chunked temporal
//     downsample (one frame, then fours).
//   single_frame_matches_image_vae -- a one-frame decode reproduces
//     MetalKrea2Vae on the SAME weights. The Qwen-Image VAE is this net
//     with time pinned to one frame, so if the causal padding is right
//     the two must agree; if this fails while the goldens pass, the
//     disagreement is in the kt=2 slice convention rather than in either
//     implementation.
//
// Env: VPIPE_WAN_TEST_MODEL_PATH = the Wan2.2 model root (uses <root>/vae),
// VPIPE_WAN_GOLDEN = the dir tools/dump_wan_vae_golden.py wrote. Skips if
// unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/wan/metal-wan-vae.h"

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

// The golden manifest's fixed geometry (tools/dump_wan_vae_golden.py
// defaults). Read from the file sizes rather than parsed, so the test
// stays honest about what it actually loaded.
struct Geom { int T = 0, h = 0, w = 0, F = 0, H = 0, W = 0; };

bool
geom_from_sizes_(std::size_t lat_elems, std::size_t vid_elems, int z_dim,
                 Geom& g)
{
  // A NON-SQUARE golden cannot be recovered from the element counts alone
  // (h*w is known, h and w are not), and non-square is exactly where a
  // row/column transpose hides -- so let the caller pin the latent grid
  // and only fall back to the square inference.
  const char* eh = std::getenv("VPIPE_WAN_GOLDEN_H");
  const char* ew = std::getenv("VPIPE_WAN_GOLDEN_W");
  if (eh != nullptr && ew != nullptr) {
    const int h = std::atoi(eh), w = std::atoi(ew);
    if (h > 0 && w > 0 && lat_elems % ((std::size_t)z_dim * h * w) == 0) {
      const int T = (int)(lat_elems / ((std::size_t)z_dim * h * w));
      const int F = 1 + 4 * (T - 1);
      g = Geom{T, h, w, F, h * 8, w * 8};
      return (std::size_t)3 * F * g.H * g.W == vid_elems;
    }
    return false;
  }
  // latent [z_dim, T, h, w] with h == w, video [3, F, h*8, w*8].
  if (lat_elems == 0 || vid_elems == 0) { return false; }
  const std::size_t per_ch = lat_elems / (std::size_t)z_dim;
  for (int T = 1; T <= 64; ++T) {
    if (per_ch % (std::size_t)T != 0) { continue; }
    const std::size_t hw = per_ch / (std::size_t)T;
    int h = 1;
    while ((std::size_t)h * h < hw) { ++h; }
    if ((std::size_t)h * h != hw) { continue; }
    const int F = 1 + 4 * (T - 1);
    const int H = h * 8;
    if ((std::size_t)3 * F * H * H == vid_elems) {
      g = Geom{T, h, h, F, H, H};
      return true;
    }
  }
  return false;
}

MetalWanVae::Config
wan_cfg_(const std::string& vdir)
{
  MetalWanVae::Config cfg;
  std::string err;
  MetalWanVae::config_from_json(vdir, cfg, &err);
  return cfg;
}

}  // namespace

TEST(wan_vae, config_from_json)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalWanVae::Config cfg;
  std::string err;
  const bool ok =
      MetalWanVae::config_from_json(std::string(root) + "/vae", cfg, &err);
  if (!ok) { std::printf("[wan_vae] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(cfg.z_dim == 16);
  EXPECT_TRUE(cfg.base_dim == 96);
  EXPECT_TRUE(cfg.num_res_blocks == 2);
  // Two temporal stages => the 4x temporal compression the frame
  // arithmetic assumes.
  int n_temporal = 0;
  for (bool b : cfg.temperal_downsample) { n_temporal += b ? 1 : 0; }
  EXPECT_TRUE(n_temporal == 2);
  EXPECT_TRUE(cfg.latents_mean.size() == 16);
  EXPECT_TRUE(cfg.latents_std.size() == 16);
  // The frame arithmetic itself, which everything below depends on.
  EXPECT_TRUE(MetalWanVae::video_frames(1) == 1);
  EXPECT_TRUE(MetalWanVae::video_frames(3) == 9);
  EXPECT_TRUE(MetalWanVae::video_frames(21) == 81);
  EXPECT_TRUE(MetalWanVae::latent_frames(81) == 21);
}

TEST(wan_vae, decode_matches_golden)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_WAN_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";
  const std::string gdir = gd;

  const std::vector<float> zin = read_f32_(gdir + "/dec_in.f32");
  const std::vector<float> ref = read_f32_(gdir + "/dec_out.f32");
  if (zin.empty() || ref.empty()) { return; }   // golden not dumped -> skip

  MetalWanVae::Config cfg = wan_cfg_(vdir);
  Geom g;
  ASSERT_TRUE(geom_from_sizes_(zin.size(), ref.size(), cfg.z_dim, g));

  auto m = MetalWanVae::load(vdir, mc, cfg, /*with_encoder=*/false);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer z = mc->make_shared_buffer(zin.size() * 2);
  ASSERT_TRUE(!z.empty());
  { auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < zin.size(); ++i) { d[i] = (_Float16)zin[i]; } }

  // Collect the streamed chunks back into one [3, F, H, W] clip so the
  // comparison is against the whole reference tensor -- and so a chunk
  // landing at the wrong frame index shows up as a mismatch rather than
  // being quietly reassembled.
  const std::size_t hw = (std::size_t)g.H * g.W;
  std::vector<float> got((std::size_t)3 * g.F * hw, 0.0f);
  int frames_seen = 0;
  std::string err;
  const bool ok = m->decode(
      z, g.T, g.h, g.w,
      [&](const SharedBuffer& rgb, int frame0, int n) {
        const auto* s = static_cast<const _Float16*>(rgb.contents());
        for (int c = 0; c < 3; ++c) {
          for (int f = 0; f < n; ++f) {
            if (frame0 + f >= g.F) { continue; }
            for (std::size_t p = 0; p < hw; ++p) {
              got[((std::size_t)c * g.F + (frame0 + f)) * hw + p] =
                  (float)s[((std::size_t)c * n + f) * hw + p];
            }
          }
        }
        frames_seen += n;
        return true;
      },
      &err);
  if (!ok) { std::printf("[wan_vae] decode: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  EXPECT_TRUE(frames_seen == g.F);

  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[wan_vae] decode rel-L2 = %.6f  ([16,%d,%d,%d] -> [3,%d,%d,%d])\n",
              r, g.T, g.h, g.w, g.F, g.H, g.W);
  EXPECT_TRUE(r < 0.05);
}

TEST(wan_vae, encode_matches_golden)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_WAN_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";
  const std::string gdir = gd;

  const std::vector<float> vin = read_f32_(gdir + "/enc_in.f32");
  const std::vector<float> ref = read_f32_(gdir + "/enc_out.f32");
  if (vin.empty() || ref.empty()) { return; }

  MetalWanVae::Config cfg = wan_cfg_(vdir);
  Geom g;
  ASSERT_TRUE(geom_from_sizes_(ref.size(), vin.size(), cfg.z_dim, g));

  auto m = MetalWanVae::load(vdir, mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  ASSERT_TRUE(m->has_encoder());

  SharedBuffer video = mc->make_shared_buffer(vin.size() * 2);
  ASSERT_TRUE(!video.empty());
  { auto* d = static_cast<_Float16*>(video.contents());
    for (std::size_t i = 0; i < vin.size(); ++i) { d[i] = (_Float16)vin[i]; } }

  std::string err;
  SharedBuffer lat = m->encode(video, g.F, g.H, g.W, &err);
  if (lat.empty()) { std::printf("[wan_vae] encode: %s\n", err.c_str()); }
  ASSERT_TRUE(!lat.empty());
  ASSERT_TRUE(lat.byte_size() >= ref.size() * 2);

  // encode() returns the WHITENED latent (the boundary the DiT stage
  // consumes); the golden is the raw posterior mode, so un-whiten before
  // comparing rather than comparing two different quantities.
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  const std::size_t per_ch = (std::size_t)g.T * g.h * g.w;
  std::vector<float> got(ref.size());
  for (int c = 0; c < cfg.z_dim; ++c) {
    const float mu = cfg.latents_mean[(std::size_t)c];
    const float sd = cfg.latents_std[(std::size_t)c];
    for (std::size_t p = 0; p < per_ch; ++p) {
      const std::size_t i = (std::size_t)c * per_ch + p;
      got[i] = (float)lp[i] * sd + mu;
    }
  }
  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[wan_vae] encode rel-L2 = %.6f  ([3,%d,%d,%d] -> [16,%d,%d,%d])\n",
              r, g.F, g.H, g.W, g.T, g.h, g.w);
  EXPECT_TRUE(r < 0.05);
}

// The reuse claim, made testable: with one frame this net IS the
// Qwen-Image VAE, so both implementations over the SAME Wan checkpoint
// have to produce the same picture. They are separate code paths -- 27
// gathered taps here against a kt=2 weight slice there -- so agreement is
// evidence about the causal padding convention, not a tautology.
TEST(wan_vae, single_frame_matches_image_vae)
{
  const char* root = std::getenv("VPIPE_WAN_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vdir = std::string(root) + "/vae";

  MetalWanVae::Config wcfg = wan_cfg_(vdir);
  auto wan = MetalWanVae::load(vdir, mc, wcfg, /*with_encoder=*/false);
  ASSERT_TRUE(wan != nullptr);

  MetalKrea2Vae::Config kcfg;
  kcfg.base_dim = wcfg.base_dim;
  kcfg.z_dim = wcfg.z_dim;
  for (int i = 0; i < 4; ++i) { kcfg.dim_mult[i] = wcfg.dim_mult[i]; }
  kcfg.num_res_blocks = wcfg.num_res_blocks;
  kcfg.latents_mean = wcfg.latents_mean;
  kcfg.latents_std = wcfg.latents_std;
  auto img = MetalKrea2Vae::load(vdir, mc, kcfg, /*with_encoder=*/false);
  ASSERT_TRUE(img != nullptr);

  const int h8 = 8, w8 = 8;
  const std::size_t n = (std::size_t)wcfg.z_dim * h8 * w8;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!z.empty());
  {
    auto* d = static_cast<_Float16*>(z.contents());
    std::uint32_t s = 12345u;
    for (std::size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(s >> 8) / 8388608.0f) - 1.0f);
    }
  }

  std::string werr, kerr;
  SharedBuffer a = wan->decode_frame(z, h8, w8, &werr);
  if (a.empty()) { std::printf("[wan_vae] wan decode: %s\n", werr.c_str()); }
  ASSERT_TRUE(!a.empty());
  SharedBuffer b = img->decode(z, h8, w8, &kerr);
  if (b.empty()) { std::printf("[wan_vae] image decode: %s\n", kerr.c_str()); }
  ASSERT_TRUE(!b.empty());

  const std::size_t px = (std::size_t)3 * h8 * 8 * w8 * 8;
  ASSERT_TRUE(a.byte_size() >= px * 2 && b.byte_size() >= px * 2);
  const auto* ap = static_cast<const _Float16*>(a.contents());
  const auto* bp = static_cast<const _Float16*>(b.contents());
  std::vector<float> av(px), bv(px);
  for (std::size_t i = 0; i < px; ++i) {
    av[i] = (float)ap[i];
    bv[i] = (float)bp[i];
  }
  const double r = rel_l2_(av.data(), bv.data(), px);
  std::printf("[wan_vae] single-frame vs image VAE rel-L2 = %.6f\n", r);
  // Both run f16 over the same weights but reduce in a different order
  // (27-tap GEMM vs 9-tap), so this is an agreement bound, not equality.
  EXPECT_TRUE(r < 0.02);
}
