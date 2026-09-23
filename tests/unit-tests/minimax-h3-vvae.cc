// The MiniMax-H3 video VAE's ViT decoder, against the diffusers
// reference at FULL depth (36 blocks).
//
// Full depth is affordable here where the DiT's was not: the decoder is
// ~2.4B parameters, so an fp32 reference fits. What it checks that the
// DiT test does not:
//
//   * a rope built from CELL-CENTRE coordinates normalized to [-1, 1)
//     per axis and multiplied by 2*pi, so the grid depends on the TILE's
//     extent rather than on absolute position. Reusing the DiT's
//     absolute-coordinate rope here produces a smooth, wrong image.
//   * the 4 register tokens and the trailing all-zero token, which are
//     appended at position 0, attended over, and dropped again. They
//     change every other token's attention, so leaving them out is not
//     a truncation -- it is a different function.
//   * q/k RMS with NO affine, attention projections WITH biases, and a
//     learned per-CHANNEL residual gate (`scale1`/`scale2`) rather than
//     the DiT's per-row modulation.
//   * the unpatchify, which expands one token into a 4 x 16 x 16 pixel
//     block with the CHANNEL slowest.
//
// The ENCODER is the other half, and shaped nothing like the decoder: a
// causal 3D CNN whose whole difficulty is padding. What its test checks
// that nothing else does:
//
//   * spatial padding by REFLECTION. Zero padding darkens the frame
//     border into the latent and survives a round trip as a vignette,
//     which is a plausible-looking image rather than an obvious break.
//   * CAUSAL temporal padding -- two zero frames prepended and none
//     appended -- so 5 pixel frames encode to 2 latent frames rather
//     than 1, and frame 0 sees only itself.
//   * group norm isolated PER FRAME, where the reference folds time
//     into the batch axis so statistics never mix across frames.
//   * the stride-2 downsample, which carries no padding of its own: an
//     asymmetric bottom/right pad of 1 comes first, offsetting the
//     sampling grid by half a tap from what a symmetric pad would give.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH, VPIPE_MINIMAX_H3_VVAE_GOLDEN,
// VPIPE_MINIMAX_H3_VVAE_ENC_GOLDEN.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/media-decode.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"

#include <algorithm>
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

// gen_vvae_golden.py's DEFAULT latent tile; a golden may declare its
// own geometry in meta.json (see golden_int_ below).
constexpr int kT = 2, kH = 4, kW = 4;

// One integer out of a golden's meta.json, so one harness serves both
// the small tile and a production-size grid.
int
golden_int_(const std::string& gdir, const char* key, int dflt)
{
  std::ifstream f(gdir + "/meta.json");
  if (!f.good()) { return dflt; }
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const std::size_t k = s.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) { return dflt; }
  const std::size_t c = s.find(':', k);
  if (c == std::string::npos) { return dflt; }
  const int n = std::atoi(s.c_str() + c + 1);
  return n > 0 ? n : dflt;
}

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

}  // namespace

TEST(minimax_h3_vvae, config_from_json)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  const bool ok = MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &err);
  if (!ok) { std::printf("[minimax_h3_vvae] config: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  std::printf("[minimax_h3_vvae] ViT decoder: %d blocks, dim %d, %d heads x "
              "%d, rope %d of %d, patch %dx%dx%d -> %d\n",
              cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.rope_dim(),
              cfg.head_dim, cfg.patch_t, cfg.patch, cfg.patch,
              cfg.patch_elems());
  EXPECT_TRUE(cfg.dim == cfg.n_heads * cfg.head_dim);
  // 16x spatial and 4x temporal, 24 latent channels -- one token becomes
  // a whole 4x16x16 block of pixels, which is what makes the decoder a
  // ViT rather than a stack of upsampling convolutions.
  EXPECT_TRUE(cfg.patch == 16 && cfg.patch_t == 4 && cfg.z_channels == 24);
  EXPECT_TRUE(cfg.patch_elems() == 3 * 4 * 16 * 16);
  // 48 of 64 channels rotate; the rope table needs rope_dim/6 = 8
  // frequencies per axis.
  EXPECT_TRUE(cfg.rope_dim() == 48 && cfg.rope_freqs() == 8);
}

TEST(minimax_h3_vvae, decode_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_VVAE_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<float> z   = read_f32_(gdir + "/z.f32");
  const std::vector<float> ref = read_f32_(gdir + "/rgb.f32");
  if (z.empty() || ref.empty()) { return; }
  // Geometry from the GOLDEN, not a constant. The historical 2x4x4 is a
  // 16-token grid; production decodes 7x16x16 and 17x30x54, and a
  // defect that only appears once the grid is big enough to matter
  // cannot show up at 4x4 no matter how tight the bar is.
  const int gT = golden_int_(gdir, "T", kT);
  const int gH = golden_int_(gdir, "H", kH);
  const int gW = golden_int_(gdir, "W", kW);

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &cerr));
  // Depth override, for bisecting a mismatch: a 0-block golden isolates
  // the input/output plumbing from the block math.
  if (const char* d = std::getenv("VPIPE_MINIMAX_H3_VVAE_LAYERS")) {
    cfg.n_layers = std::atoi(d);
  }
  ASSERT_TRUE(z.size() ==
              (std::size_t)cfg.z_channels * gT * gH * gW);
  ASSERT_TRUE(ref.size() == (std::size_t)cfg.out_channels * (gT * cfg.patch_t) *
                                (gH * cfg.patch) * (gW * cfg.patch));

  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer zb = mc->make_shared_buffer(z.size() * 2);
  ASSERT_TRUE(!zb.empty());
  {
    auto* d = static_cast<std::uint16_t*>(zb.contents());
    for (std::size_t i = 0; i < z.size(); ++i) { d[i] = f32_to_bf16_(z[i]); }
  }

  std::string derr;
  SharedBuffer rgb = m->decode(zb, gT, gH, gW, &derr);
  if (rgb.empty()) { std::printf("[minimax_h3_vvae] %s\n", derr.c_str()); }
  ASSERT_TRUE(!rgb.empty());
  ASSERT_TRUE(rgb.byte_size() >= ref.size() * 2);

  const auto* g = static_cast<const std::uint16_t*>(rgb.contents());
  double num = 0.0, den = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = (double)bf16_to_f32_(g[i]) - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
    worst = std::max(worst, std::fabs(d));
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[minimax_h3_vvae] %dx%dx%d latent -> %dx%dx%d rgb, %d blocks: "
              "rel-L2 %.6f, max abs %.4f\n", gT, gH, gW, gT * cfg.patch_t,
              gH * cfg.patch, gW * cfg.patch, cfg.n_layers, rel, worst);
  // The reference runs the block stack in bf16, so this compares two
  // bf16 evaluations whose accumulation orders differ across 36 blocks.
  // MEASURED: 0.0041 with no blocks at all (the input/output plumbing's
  // own bf16 floor), 0.0043 at depth 1 and 0.0080 at full depth -- so
  // the blocks add almost nothing and the bar is set just above the
  // floor rather than at a level a real error could hide under.
  EXPECT_TRUE(rel < 0.02);
  // A decoder that returned zeros, or that dropped the residual gates,
  // would have a small relative error against a small reference; pin the
  // magnitude too.
  EXPECT_TRUE(den > 0.0);
}

TEST(minimax_h3_vvae, encode_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_VVAE_ENC_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<float> rgb = read_f32_(gdir + "/rgb.f32");
  const std::vector<float> ref = read_f32_(gdir + "/moments.f32");
  if (rgb.empty() || ref.empty()) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &cerr));
  // The clip geometry comes from the golden rather than being repeated
  // here, so regenerating at another size cannot leave the two
  // disagreeing about what they are comparing.
  int T = 5, H = 32, W = 32;
  {
    std::ifstream d(gdir + "/dims.txt");
    if (d) { d >> T >> H >> W; }
  }
  // minitest's ASSERT_TRUE is EXPECT_TRUE: it records the failure and
  // keeps going. A geometry that disagrees with the golden would then
  // read past the end of both buffers, so stop by hand.
  if (T <= 0 || H <= 0 || W <= 0 ||
      rgb.size() != (std::size_t)cfg.in_channels * T * H * W) {
    std::printf("[minimax_h3_vvae] golden geometry %dx%dx%d does not match "
                "its %zu input floats\n", T, H, W, rgb.size());
    EXPECT_TRUE(false);
    return;
  }

  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  // 17 pixel frames become 5 latent frames, not 4: the causal pad makes
  // each stride-2 temporal convolution emit floor((T-1)/2)+1, so the
  // count rounds UP twice and a plain division loses the last frame.
  EXPECT_TRUE(m->encoded_frames(17) == 5);
  EXPECT_TRUE(m->encoded_frames(5) == 2);
  EXPECT_TRUE(m->encoded_frames(cfg.clip_length) == 5);

  SharedBuffer xb = mc->make_shared_buffer(rgb.size() * 2);
  ASSERT_TRUE(!xb.empty());
  {
    auto* d = static_cast<std::uint16_t*>(xb.contents());
    for (std::size_t i = 0; i < rgb.size(); ++i) {
      d[i] = f32_to_bf16_(rgb[i]);
    }
  }

  std::string eerr;
  SharedBuffer mom = m->encode(xb, T, H, W, &eerr);
  if (mom.empty()) { std::printf("[minimax_h3_vvae] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!mom.empty());
  // The depth override changes the channel count, so size the
  // comparison from the golden rather than from the config.
  if (mom.byte_size() < ref.size() * 2) {
    std::printf("[minimax_h3_vvae] encode returned %zu bytes, golden wants "
                "%zu\n", mom.byte_size(), ref.size() * 2);
    EXPECT_TRUE(false);
    return;
  }

  const auto* g = static_cast<const std::uint16_t*>(mom.contents());
  double num = 0.0, den = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = (double)bf16_to_f32_(g[i]) - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
    worst = std::max(worst, std::fabs(d));
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[minimax_h3_vvae] encode %dx%dx%d -> %zu moments: rel-L2 "
              "%.6f, max abs %.4f\n", T, H, W, ref.size(), rel, worst);
  // The reference runs the encoder in fp32 (the checkpoint pins it
  // there), so this compares a bf16 evaluation against a higher-
  // precision one through 12 residual blocks -- a wider gap than the
  // decoder's bf16-vs-bf16 comparison, and the bar is set from what was
  // MEASURED at each depth rather than from what looks tidy.
  // MEASURED: 0.0020 with conv_in alone (the bf16 floor against an fp32
  // reference), then 0.0027 / 0.0033 / 0.0034 as levels are added, and
  // 0.0039 at full depth on this 5x32x32 clip -- 0.0025 on a 17x64x48
  // one. The growth is smooth, with no step at the level that
  // introduces the spatial stride-2 or at either temporal one, which is
  // what says the padding is right rather than merely close.
  EXPECT_TRUE(rel < 0.02);
  EXPECT_TRUE(den > 0.0);
}

// The two halves have to agree about the latent they hand each other,
// and that is not checked by comparing either one against the reference
// on its own: both goldens would still pass if encode() wrote its
// channel-first output in an order decode() does not read. So encode a
// smooth ramp, feed the MEAN half of the moments straight back into
// decode(), and require the reconstruction to track the input.
//
// The bar is a correlation rather than a PSNR on purpose. This is a
// lossy VAE and the clip is synthetic, so the absolute error is not
// meaningful -- but a transposed axis or a mismatched frame count
// destroys the correlation outright, which is the failure being looked
// for.
TEST(minimax_h3_vvae, encode_decode_round_trip)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &cerr));
  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // At the checkpoint's own 256-pixel tile. The size is not a free knob
  // here: the ViT decoder's rope is normalized to the TILE's extent, so
  // a 32x32 clip (a 2x2 latent) puts the decoder far outside anything
  // it was trained on and reconstructs poorly for reasons that have
  // nothing to do with the seam being tested. MEASURED: corr 0.39 at
  // 32x32 against 0.97 here.
  const int T = 5, H = 256, W = 256, IC = cfg.in_channels;
  const std::size_t n = (std::size_t)IC * T * H * W;
  SharedBuffer xb = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!xb.empty());
  std::vector<float> src(n);
  {
    auto* d = static_cast<std::uint16_t*>(xb.contents());
    for (int ch = 0; ch < IC; ++ch) {
      for (int t = 0; t < T; ++t) {
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            // A ramp whose period is wider than the tile. This VAE
            // compresses 4x16x16x3 pixels into 24 numbers, so anything
            // with detail near the latent cell size does not survive
            // the round trip at all -- the reference itself scores
            // corr 0.41 on a ramp four times this frequency, against
            // 0.66 on this one.
            const float v =
                std::sin((float)(x + 2 * ch) * 0.02f) *
                std::cos((float)(y + t) * 0.015f) * 0.8f;
            const std::size_t i =
                (((std::size_t)ch * T + t) * H + y) * W + x;
            src[i] = v;
            d[i] = f32_to_bf16_(v);
          }
        }
      }
    }
  }

  std::string eerr;
  SharedBuffer mom = m->encode(xb, T, H, W, &eerr);
  if (mom.empty()) { std::printf("[minimax_h3_vvae] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!mom.empty());

  const int lt = m->encoded_frames(T);
  const int lh = H / cfg.patch, lw = W / cfg.patch;
  const std::size_t voxels = (std::size_t)lt * lh * lw;
  // The moments are mean|logvar on the channel axis, so the mean is the
  // leading z_channels planes -- already the channel-first layout
  // decode() wants, with no repacking in between.
  SharedBuffer zb = mc->make_shared_buffer(voxels * cfg.z_channels * 2);
  ASSERT_TRUE(!zb.empty());
  std::memcpy(zb.contents(), mom.contents(),
              voxels * (std::size_t)cfg.z_channels * 2);

  std::string derr;
  SharedBuffer rgb = m->decode(zb, lt, lh, lw, &derr);
  if (rgb.empty()) { std::printf("[minimax_h3_vvae] %s\n", derr.c_str()); }
  ASSERT_TRUE(!rgb.empty());

  // The decode emits lt*patch_t frames, which is more than went in: the
  // clip's length is not a multiple of the temporal ratio, so the
  // reference's `frame_pre_padding` leading frames correspond to the
  // implicit pad and the original clip starts after them.
  const int ot = lt * cfg.patch_t;
  const int lead = ((-T) % cfg.patch_t + cfg.patch_t) % cfg.patch_t;
  EXPECT_TRUE(ot - lead == T);
  const std::size_t plane = (std::size_t)H * W;
  if (rgb.byte_size() < (std::size_t)cfg.out_channels * ot * plane * 2) {
    EXPECT_TRUE(false);
    return;
  }

  const auto* g = static_cast<const std::uint16_t*>(rgb.contents());
  double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
  std::size_t cnt = 0;
  for (int ch = 0; ch < IC; ++ch) {
    for (int t = 0; t < T; ++t) {
      for (std::size_t p = 0; p < plane; ++p) {
        const double a = src[(((std::size_t)ch * T + t) * plane) + p];
        const double b = bf16_to_f32_(
            g[((std::size_t)ch * ot + (t + lead)) * plane + p]);
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b;
        ++cnt;
      }
    }
  }
  const double nn = (double)cnt;
  const double cov = sxy / nn - (sx / nn) * (sy / nn);
  const double va  = sxx / nn - (sx / nn) * (sx / nn);
  const double vb  = syy / nn - (sy / nn) * (sy / nn);
  const double corr = (va > 0.0 && vb > 0.0) ? cov / std::sqrt(va * vb) : 0.0;
  std::printf("[minimax_h3_vvae] round trip %dx%dx%d -> %dx%dx%d latent -> "
              "%d frames: corr %.4f\n", T, H, W, lt, lh, lw, ot, corr);
  // MEASURED: the diffusers reference, run end to end on this exact
  // ramp, scores 0.656 -- so the bar is set below its OWN number rather
  // than at what a lossless codec would give. vpipe tracked it to four
  // decimals on the higher-frequency ramp this started with (0.4120 vs
  // 0.41205), which is what established that the halves agree; a
  // transposed axis or an off-by-one frame count collapses this to
  // roughly zero, which is the failure the bar has to catch.
  EXPECT_TRUE(corr > 0.5);
}

// The WHOLE-VIDEO path: clip chunking on top of spatial tiling, against
// the reference's own `_encode` / `_decode`.
//
// This is orchestration, and it is where the geometry gets counter-
// intuitive. Three things it pins that the per-clip goldens cannot:
//
//   * `token_drop` makes the round trip LOSSY IN LENGTH on purpose --
//     35 pixel frames encode to 12 latent frames and decode back to 39,
//     not 35. Treating encode and decode as inverses gets this wrong in
//     a way that looks like an off-by-one until the frame counts are
//     worked out.
//   * a chunk's leading `frame_pre_pad` frames belong to an implicit
//     pad and are dropped, and consecutive chunks are cross-faded over
//     `frame_overlap` frames -- so the decoded stream is neither a
//     concatenation nor a simple overlap-add.
//   * the spatial stitch, including a CORNER cell where a tile blends
//     against both its upper and left neighbours, each read in its
//     ORIGINAL (unblended) state.
//
// The tile size is turned down from the released 256 so a 96-pixel clip
// gives a 2x2 grid. Both sides use the same value, so this tests the
// two agreeing rather than image quality at that tile size.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH, VPIPE_MINIMAX_H3_VVAE_VIDEO_GOLDEN.
// THE ENCODE TICKS PER TILE, as the decode always has. A reference encode
// is ~1.8 s per frame at 896x512, and before this the bar above it moved
// once per REFERENCE -- minutes of one number for a long clip.
//
// Three things are pinned. The VAE's own count spans the whole clip
// (clips x tiles, one total, reaching it). Its counters are the CALL's:
// a second encode starts again from 0 against its own total, where a
// counter left set would report against the first clip's. And the
// reference encoder turns those ticks into a bar that moves INSIDE the
// reference, landing exactly on the share it planned for it.
//
// Geometry chosen to clear the tiling floor: 256x512 is two tiles or
// more across at the 256 tile, and 18 frames is two clips of 17.
TEST(minimax_h3_vvae, encode_reports_progress_per_tile)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &cerr));
  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  const int T = 18, H = 256, W = 512, IC = cfg.in_channels;
  const int tiles =
      (int)(minimax_h3::split_tiles(H, cfg.tile_size, cfg.tile_overlap_min,
                                    cfg.patch).start.size() *
            minimax_h3::split_tiles(W, cfg.tile_size, cfg.tile_overlap_min,
                                    cfg.patch).start.size());
  const int clips = (T + cfg.clip_length - 1) / cfg.clip_length;
  std::printf("[minimax_h3_vvae] encode %dx%dx%d: %d clips x %d tiles\n",
              T, W, H, clips, tiles);
  EXPECT_TRUE(tiles >= 2 && clips == 2);

  auto run = [&](int frames, std::vector<std::pair<int, int>>* seen) {
    const std::size_t n = (std::size_t)IC * frames * H * W;
    SharedBuffer xb = mc->make_shared_buffer(n * 2);
    if (xb.empty()) { return false; }
    auto* d = static_cast<std::uint16_t*>(xb.contents());
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = f32_to_bf16_(std::sin((float)(i % 977) * 0.01f) * 0.5f);
    }
    m->set_tile_progress([seen](int done, int total) {
      seen->emplace_back(done, total);
    });
    int lf = 0;
    std::string err;
    SharedBuffer out = m->encode_video(xb, frames, H, W, &lf, &err);
    m->set_tile_progress(nullptr);
    return !out.empty() && lf > 0;
  };

  std::vector<std::pair<int, int>> clip, still;
  ASSERT_TRUE(run(T, &clip));
  ASSERT_TRUE(run(1, &still));
  auto check = [&](const std::vector<std::pair<int, int>>& v, int want,
                  const char* what) {
    bool steady = !v.empty(), forward = true;
    int last = 0;
    for (const auto& p : v) {
      steady = steady && p.second == want;
      forward = forward && p.first >= last;
      last = p.first;
    }
    std::printf("[minimax_h3_vvae] %s: %zu reports, %d/%d .. %d/%d\n", what,
                v.size(), v.empty() ? -1 : v.front().first,
                v.empty() ? -1 : v.front().second,
                v.empty() ? -1 : v.back().first,
                v.empty() ? -1 : v.back().second);
    EXPECT_TRUE(steady);
    EXPECT_TRUE(forward);
    // Both edges of every tile: entry and exit.
    EXPECT_TRUE((int)v.size() == 2 * want);
    EXPECT_TRUE(!v.empty() && v.front().first == 0 &&
                v.back().first == want);
  };
  check(clip, clips * tiles, "clip");
  // The second call's own total, from 0 -- not the first one's.
  check(still, tiles, "still");

  // ---- through the reference encoder ---------------------------------
  namespace h3 = vpipe::genai::minimax_h3;
  h3::ReferencePlan plan;
  plan.target_frames     = 21;
  plan.canvas_short_edge = 0;           // the clip's own 512x256
  h3::MediaReference ref;
  ref.kind       = h3::MediaReference::Kind::kVideo;
  ref.num_frames = T;
  ref.height     = H;
  ref.width      = W;
  ref.fps        = 24.0;
  ref.rgb.resize((std::size_t)T * 3 * H * W);
  for (std::size_t i = 0; i < ref.rgb.size(); ++i) {
    ref.rgb[i] = (std::uint8_t)((i * 7) % 251);
  }
  std::vector<h3::MediaReference> refs;
  refs.push_back(std::move(ref));

  struct Seen {
    std::uint64_t done, total;
    std::string detail;
  };
  std::vector<Seen> seen;
  h3::ReferenceEncoders models;
  models.mc        = mc;
  models.video_vae = m.get();
  models.progress  = [&seen](std::uint64_t done, std::uint64_t total,
                             const std::string& detail) {
    seen.push_back({done, total, detail});
  };
  h3::EncodedReferences enc;
  std::string rerr;
  ASSERT_TRUE(h3::encode_references(refs, "p", plan, models, &enc, &rerr));
  if (seen.empty()) { return; }

  // The plan: one canvas frame for the rest, plus 2 clips x 17 frames of
  // it through the VAE.
  const std::uint64_t frame = (std::uint64_t)H * W;
  const std::uint64_t want = frame + (std::uint64_t)clips *
                                         cfg.clip_length * frame;
  int tile_reports = 0, distinct = 0;
  std::uint64_t prev = ~0ull;
  bool steady = true;
  for (const auto& p : seen) {
    steady = steady && p.total == want;
    if (p.detail.find("video VAE tile") != std::string::npos) {
      ++tile_reports;
      if (p.done != prev) { ++distinct; prev = p.done; }
    }
  }
  std::printf("[minimax_h3_vvae] refenc bar: total %llu, %d tile reports, "
              "%d distinct positions, last '%s'\n",
              (unsigned long long)want, tile_reports, distinct,
              seen.back().detail.c_str());
  EXPECT_TRUE(steady);
  EXPECT_TRUE(tile_reports == 2 * clips * tiles);
  // It MOVES inside the reference: one position per finished tile, plus
  // the start.
  EXPECT_TRUE(distinct == clips * tiles + 1);
  EXPECT_TRUE(seen.back().done == want);
  // The VAE is left as it was found.
  std::vector<std::pair<int, int>> after;
  ASSERT_TRUE(run(1, &after));
  EXPECT_TRUE((int)after.size() == 2 * tiles);
}

TEST(minimax_h3_vvae, video_chunking_matches_golden)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MINIMAX_H3_VVAE_VIDEO_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<float> rgb = read_f32_(gdir + "/rgb.f32");
  const std::vector<float> refm = read_f32_(gdir + "/moments.f32");
  const std::vector<float> refd = read_f32_(gdir + "/decoded.f32");
  if (rgb.empty() || refm.empty() || refd.empty()) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &cerr));
  int T = 0, H = 0, W = 0, tile = 0, tov = 0;
  {
    std::ifstream d(gdir + "/dims.txt");
    if (d) { d >> T >> H >> W >> tile >> tov; }
  }
  if (T <= 0 || H <= 0 || W <= 0 || tile <= 0 ||
      rgb.size() != (std::size_t)cfg.in_channels * T * H * W) {
    std::printf("[minimax_h3_vvae] video golden geometry %dx%dx%d does not "
                "match its %zu floats\n", T, H, W, rgb.size());
    EXPECT_TRUE(false);
    return;
  }
  cfg.tile_size        = tile;
  cfg.tile_overlap_min = tov;
  // VPIPE_MINIMAX_H3_VVAE_GOLDEN_ANE: the same golden with the ANE
  // feed-forward tier on -- real encoded latents through the whole decoder.
  // Pair with VPIPE_H3_VVAE_ANE_CHUNK (and _ANE_ROWS) for tiles shorter
  // than one default chunk.
  if (std::getenv("VPIPE_MINIMAX_H3_VVAE_GOLDEN_ANE") != nullptr) {
    cfg.ane_ffn = true;
    if (const char* r = std::getenv("VPIPE_MINIMAX_H3_VVAE_ANE_ROWS")) {
      cfg.ane_rows = (float)std::atof(r);
    }
  }

  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // The released chunk geometry, spelled out: 5 latent frames per
  // 17-frame clip, a 3-frame implicit lead pad, a 2-latent chunk
  // overlap and a 5-frame cross-fade.
  EXPECT_TRUE(cfg.tokens_per_chunk() == 5);
  EXPECT_TRUE(cfg.frame_pre_pad() == 3);
  EXPECT_TRUE(cfg.token_overlap() == 2);
  EXPECT_TRUE(cfg.frame_overlap() == 5);

  const int lh = H / cfg.patch, lw = W / cfg.patch;
  const int lt = m->video_latent_frames(T);
  const int ot = m->decoded_frames(lt);
  std::printf("[minimax_h3_vvae] video %d frames -> %d latent -> %d frames "
              "(tile %d/%d, %dx%d latent)\n", T, lt, ot, tile, tov, lh, lw);
  if (refm.size() != (std::size_t)(2 * cfg.z_channels) * lt * lh * lw ||
      refd.size() != (std::size_t)cfg.out_channels * ot * H * W) {
    std::printf("[minimax_h3_vvae] predicted %d latent / %d decoded frames, "
                "golden has %zu / %zu floats\n", lt, ot, refm.size(),
                refd.size());
    EXPECT_TRUE(false);
    return;
  }

  SharedBuffer xb = mc->make_shared_buffer(rgb.size() * 2);
  ASSERT_TRUE(!xb.empty());
  {
    auto* d = static_cast<std::uint16_t*>(xb.contents());
    for (std::size_t i = 0; i < rgb.size(); ++i) {
      d[i] = f32_to_bf16_(rgb[i]);
    }
  }

  auto rel_l2 = [](const std::uint16_t* got, const std::vector<float>& ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double d = (double)bf16_to_f32_(got[i]) - (double)ref[i];
      num += d * d;
      den += (double)ref[i] * (double)ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  int lt_got = 0, ot_got = 0;
  std::string eerr;
  SharedBuffer mom = m->encode_video(xb, T, H, W, &lt_got, &eerr);
  if (mom.empty()) { std::printf("[minimax_h3_vvae] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!mom.empty());
  EXPECT_TRUE(lt_got == lt);
  if (mom.byte_size() < refm.size() * 2) { EXPECT_TRUE(false); return; }
  const double erel =
      rel_l2(static_cast<const std::uint16_t*>(mom.contents()), refm);

  // Decode the encoder's OWN mean, so a decode mismatch cannot be
  // blamed on a different latent going in.
  SharedBuffer zb =
      mc->make_shared_buffer((std::size_t)cfg.z_channels * lt * lh * lw * 2);
  ASSERT_TRUE(!zb.empty());
  std::memcpy(zb.contents(), mom.contents(),
              (std::size_t)cfg.z_channels * lt * lh * lw * 2);

  std::string derr;
  SharedBuffer dec = m->decode_video(zb, lt, lh, lw, &ot_got, &derr);
  if (dec.empty()) { std::printf("[minimax_h3_vvae] %s\n", derr.c_str()); }
  ASSERT_TRUE(!dec.empty());
  EXPECT_TRUE(ot_got == ot);
  if (dec.byte_size() < refd.size() * 2) { EXPECT_TRUE(false); return; }
  const double drel =
      rel_l2(static_cast<const std::uint16_t*>(dec.contents()), refd);

  std::printf("[minimax_h3_vvae] video encode rel-L2 %.6f, decode rel-L2 "
              "%.6f\n", erel, drel);
  // Both compare bf16 against an fp32 reference. The decode's own input
  // is this encode's output, so its error carries the encode's forward
  // -- it is not a second independent measurement.
  //
  // What makes these bars mean something: tiling is LOAD-BEARING here.
  // MEASURED, by running the reference against ITSELF with tiling on
  // and off, the tiled result differs by 0.091 / 0.090 -- so an
  // implementation that skipped the tiling, or stitched it differently,
  // would land ~28x above what is observed (0.0032 / 0.0090). The
  // agreement is with the TILED reference specifically, not with the
  // net's output in general.
  EXPECT_TRUE(erel < 0.02);
  EXPECT_TRUE(drel < 0.05);

  // The degenerate ends of the chunking, which are easy to get wrong in
  // opposite directions. A single frame skips the temporal path
  // entirely and keeps its one latent frame -- `token_drop` does not
  // apply. And a latent video shorter than one chunk plus the drop
  // cannot be decoded at all: it must report that rather than return a
  // truncated video.
  EXPECT_TRUE(m->video_latent_frames(1) == 1);
  EXPECT_TRUE(m->decoded_frames(2) == 0);
  std::string serr;
  EXPECT_TRUE(m->decode_video(zb, 2, lh, lw, nullptr, &serr).empty());
  EXPECT_TRUE(!serr.empty());
}

// The latent whitening the two stages exchange.
//
// The DiT generates in NORMALIZED space, so vae-encode divides by these
// and vae-decode multiplies back. If config_from_json failed to read
// them both stages fall back to identity and STILL RUN -- producing a
// washed-out clip with the wrong per-channel offset rather than obvious
// noise. So the thing worth pinning is that they arrived at all, with
// one entry per latent channel.
TEST(minimax_h3_vvae, latent_whitening_is_read)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &err));
  EXPECT_TRUE((int)cfg.latents_mean.size() == cfg.z_channels);
  EXPECT_TRUE((int)cfg.latents_std.size() == cfg.z_channels);
  if ((int)cfg.latents_mean.size() != cfg.z_channels) { return; }

  // A std of 0 would make the encode side divide by zero; a mean that
  // was all zeros AND a std all ones would mean the file parsed but
  // carried nothing, which is the identity case this test exists to
  // tell apart from a real read.
  bool all_identity = true;
  for (int i = 0; i < cfg.z_channels; ++i) {
    EXPECT_TRUE(cfg.latents_std[(std::size_t)i] > 0.0f);
    if (cfg.latents_mean[(std::size_t)i] != 0.0f ||
        cfg.latents_std[(std::size_t)i] != 1.0f) {
      all_identity = false;
    }
  }
  EXPECT_TRUE(!all_identity);

  // Round trip: whiten then un-whiten is the identity the two stages
  // rely on being inverses of each other.
  double worst = 0.0;
  for (int c = 0; c < cfg.z_channels; ++c) {
    const float mu = cfg.latents_mean[(std::size_t)c];
    const float sd = cfg.latents_std[(std::size_t)c];
    for (float raw : {-3.0f, 0.0f, 2.5f}) {
      const float norm = (raw - mu) / sd;
      const float back = norm * sd + mu;
      worst = std::max(worst, (double)std::fabs(back - raw));
    }
  }
  std::printf("[minimax_h3_vvae] latent whitening: %d channels, mean[0] "
              "%.4f std[0] %.4f, round-trip worst %.2e\n", cfg.z_channels,
              cfg.latents_mean[0], cfg.latents_std[0], worst);
  EXPECT_TRUE(worst < 1e-5);
}

// Decode throughput at a chosen geometry, on a SYNTHETIC latent.
//
// Synthetic because the question is timing, not fidelity, and a golden
// pins one small shape while the cost here is driven entirely by the
// latent grid: the decoder is a transformer, so attention is quadratic in
// tokens per tile and a 960x544 clip is a different regime from the
// 256x256 the goldens use.
//
// Reports ms per PIXEL FRAME as well as per call, because that is the
// number that composes -- a clip is decoded in chunks and the per-call
// figure hides how many frames each chunk carried.
//
// Env: VPIPE_MINIMAX_H3_VVAE_BENCH=1 + VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
// Geometry: ..._BENCH_{LT,LH,LW} (default the production 960x544 clip).
TEST(minimax_h3_vvae, decode_bench)
{
  const char* on   = std::getenv("VPIPE_MINIMAX_H3_VVAE_BENCH");
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (on == nullptr || *on == '\0' || *on == '0') { return; }
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &err));
  auto vae = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(vae != nullptr);

  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int LT = envi("VPIPE_MINIMAX_H3_VVAE_BENCH_LT", 17);
  const int lh = envi("VPIPE_MINIMAX_H3_VVAE_BENCH_LH", 34);   // 544 / 16
  const int lw = envi("VPIPE_MINIMAX_H3_VVAE_BENCH_LW", 60);   // 960 / 16

  const std::size_t n = (std::size_t)cfg.z_channels * LT * lh * lw;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!z.empty());
  {
    auto* d = static_cast<std::uint16_t*>(z.contents());
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = f32_to_bf16_(0.1f * (float)((int)(i * 37 % 19) - 9));
    }
  }
  int frames = 0;
  SharedBuffer out = vae->decode_video(z, LT, lh, lw, &frames, &err);  // warm
  if (out.empty()) { std::printf("[minimax_h3_vvae] %s\n", err.c_str()); }
  ASSERT_TRUE(!out.empty() && frames > 0);

  const auto t0 = std::chrono::steady_clock::now();
  out = vae->decode_video(z, LT, lh, lw, &frames, &err);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(!out.empty());
  // Optional dump, so a sweep can ask what a setting did to the OUTPUT
  // and not only to the clock. Tile size is not a pure speed knob here --
  // it decides what each tile's attention can see -- so a faster arm is
  // only interesting alongside what it changed.
  if (const char* dp = std::getenv("VPIPE_MINIMAX_H3_VVAE_BENCH_DUMP")) {
    std::FILE* f = std::fopen(dp, "wb");
    if (f != nullptr) {
      const auto* d = static_cast<const std::uint16_t*>(out.contents());
      const std::size_t n = out.byte_size() / 2;
      std::vector<float> v(n);
      for (std::size_t k = 0; k < n; ++k) { v[k] = bf16_to_f32_(d[k]); }
      std::fwrite(v.data(), 4, n, f);
      std::fclose(f);
      std::printf("[minimax_h3_vvae] dumped %zu floats to %s\n", n, dp);
    }
  }
  std::printf("[minimax_h3_vvae] decode %dx%dx%d latent -> %d frames of "
              "%dx%d in %.0f ms (%.1f ms/frame, tile %d)\n",
              LT, lh, lw, frames, lw * cfg.patch, lh * cfg.patch, ms,
              ms / (double)frames, cfg.tile_size);
  EXPECT_TRUE(ms > 0.0);
}

// The ANE feed-forward tier against the GPU alone, one decode() tile.
//
// The split is exact arithmetic -- the feed-forward is row-independent -- so
// what separates the runs is fp16 on the ANE's rows against bf16. Small, and
// not zero: zero would mean no row reached the ANE.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH, geometry ..._ANE_{LT,LH,LW}
// (default 7x16x16: one production tile, 1797 rows), VPIPE_H3_VVAE_ANE_ROWS
// pins the share.
TEST(minimax_h3_vvae, decode_ane_matches_gpu)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int LT = envi("VPIPE_MINIMAX_H3_VVAE_ANE_LT", 7);
  const int lh = envi("VPIPE_MINIMAX_H3_VVAE_ANE_LH", 16);
  const int lw = envi("VPIPE_MINIMAX_H3_VVAE_ANE_LW", 16);
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &err));

  const std::size_t n = (std::size_t)cfg.z_channels * LT * lh * lw;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!z.empty());
  if (z.empty()) { return; }
  {
    std::uint32_t sd = 0x51ced00du;
    auto* d = static_cast<std::uint16_t*>(z.contents());
    for (std::size_t i = 0; i < n; ++i) {
      sd = sd * 1664525u + 1013904223u;
      d[i] = f32_to_bf16_(((float)(sd >> 9) / 4194304.0f - 1.0f));
    }
  }
  struct Arm {
    std::vector<float> rgb;
    double best_ms = 0.0;
    bool   armed   = false;
  };
  auto run = [&](bool ane) {
    Arm arm;
    MetalMiniMaxH3VideoVae::Config rc = cfg;
    rc.ane_ffn = ane;
    if (const char* r = std::getenv("VPIPE_H3_VVAE_ANE_ROWS")) {
      rc.ane_rows = (float)std::atof(r);
    }
    auto m = MetalMiniMaxH3VideoVae::load(root, mc, rc);
    if (m == nullptr) { return arm; }
    for (int i = 0; i < 3; ++i) {         // 0 warms, then best of two
      const auto t0 = std::chrono::steady_clock::now();
      std::string derr;
      SharedBuffer out = m->decode(z, LT, lh, lw, &derr);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (out.empty()) {
        std::printf("[minimax_h3_vvae] decode: %s\n", derr.c_str());
        return Arm{};
      }
      if (i == 0) {
        const auto* p = static_cast<const std::uint16_t*>(out.contents());
        arm.rgb.resize(out.byte_size() / 2);
        for (std::size_t k = 0; k < arm.rgb.size(); ++k) {
          arm.rgb[k] = bf16_to_f32_(p[k]);
        }
      } else if (arm.best_ms == 0.0 || ms < arm.best_ms) {
        arm.best_ms = ms;
      }
    }
    arm.armed = m->ane_armed();
    return arm;
  };
  const Arm gpu = run(false);
  const Arm ane = run(true);
  ASSERT_TRUE(!gpu.rgb.empty() && gpu.rgb.size() == ane.rgb.size());
  if (gpu.rgb.empty() || gpu.rgb.size() != ane.rgb.size()) { return; }
  double num = 0.0, den = 0.0, worst = 0.0;
  std::size_t bad = 0;
  for (std::size_t k = 0; k < gpu.rgb.size(); ++k) {
    if (!std::isfinite(ane.rgb[k])) { ++bad; continue; }
    const double d = (double)ane.rgb[k] - (double)gpu.rgb[k];
    num += d * d;
    den += (double)gpu.rgb[k] * (double)gpu.rgb[k];
    worst = std::max(worst, std::fabs(d));
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
  std::printf("[minimax_h3_vvae] ANE vs GPU decode %dx%dx%d: rel-L2 %.6f, max "
              "abs %.4f, %zu non-finite | GPU-only %.0f ms, ANE split %.0f "
              "ms, %.3fx\n", LT, lh, lw, rel, worst, bad, gpu.best_ms,
              ane.best_ms, ane.best_ms > 0.0 ? gpu.best_ms / ane.best_ms : 0.0);
  EXPECT_TRUE(ane.armed);
  EXPECT_TRUE(bad == 0);
  EXPECT_TRUE(rel < 0.02 && rel > 0.0);
}

// ---------------------------------------------------------------------
// The SINGLE-LATENT-FRAME image decode.
//
// H3's encoder turns one still into exactly one latent frame, and the
// chunked video decoder cannot turn that back into a picture: it spends
// `token_drop` frames priming its temporal window, so it needs 7 latent
// frames before it emits anything and refuses 1 outright. An
// image-trained checkpoint closes the gap by fine-tuning the decoder to
// read one token alone -- the encoder, `quant_conv` and the whitening
// are left frozen, so the LATENT SPACE is the same one and a latent from
// any H3 model decodes here.
//
// What these check that nothing else does:
//
//   * the capability is read from the CHECKPOINT and never assumed.
//     Both files carry the same 562 tensor names at the same shapes, so
//     nothing about the weights distinguishes them and the video
//     decoder would return a correctly shaped picture of the wrong
//     thing.
//   * which of the `patch_t` pixel frames a latent token expands to is
//     the picture. Off by one is not a crash and not noise -- it is a
//     plausible, blurred image, because the neighbouring frames of a
//     still's block are near-copies of it.
//
// Env: VPIPE_MINIMAX_H3_IMAGE_VAE_PATH (the image checkpoint),
// VPIPE_MINIMAX_H3_IMAGE_SRC (a photograph to round-trip).
// ---------------------------------------------------------------------

namespace {

// The reference's pixel normalization, which is IMAGENET's and not
// [-1, 1]: a round trip run in the wrong space reconstructs a
// contrast-shifted image rather than a broken one.
constexpr float kInMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kInStd[3]  = {0.229f, 0.224f, 0.225f};

}  // namespace

TEST(minimax_h3_vvae, image_capability_is_declared_by_the_checkpoint)
{
  const char* img = std::getenv("VPIPE_MINIMAX_H3_IMAGE_VAE_PATH");
  if (img == nullptr || *img == '\0') { return; }

  MetalMiniMaxH3VideoVae::Config icfg;
  std::string ierr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(img, icfg, &ierr));
  if (!ierr.empty()) { std::printf("[minimax_h3_vvae] %s\n", ierr.c_str()); }
  EXPECT_TRUE(icfg.t1_direct);
  // The released checkpoint's slice, which is also the pad the chunked
  // path already discards -- `frame_pre_pad()` for a 17-frame clip at a
  // temporal stride of 4. Agreeing with that derivation is why the
  // default is safe; it is still READ, because a later fine-tune is
  // free to park the picture elsewhere.
  EXPECT_TRUE(icfg.t1_output_slice == 3);
  EXPECT_TRUE(icfg.t1_output_slice == icfg.frame_pre_pad());
  // Same geometry and same latent space as the video checkpoint: this is
  // what makes it a drop-in decoder rather than a second model.
  EXPECT_TRUE(icfg.z_channels == 24 && icfg.patch == 16 && icfg.patch_t == 4);

  // The tiling must land on the patch grid, or `split_tiles` lays a
  // union that overruns the image -- see
  // minimax_h3_layout.vae_tile_split_needs_patch_aligned_geometry.
  EXPECT_TRUE(icfg.tile_size % icfg.patch == 0);
  EXPECT_TRUE(icfg.tile_overlap_min % icfg.patch == 0);

  // And the env override cannot put it off that grid. A value that is
  // not a whole number of patches is IGNORED rather than honoured,
  // which is the same answer the config path gives by refusing.
  const int was = icfg.tile_size;
  ::setenv("VPIPE_H3_VVAE_TILE", "250", 1);
  MetalMiniMaxH3VideoVae::Config bad;
  const bool ok = MetalMiniMaxH3VideoVae::config_from_json(img, bad, nullptr);
  ::unsetenv("VPIPE_H3_VVAE_TILE");
  EXPECT_TRUE(ok);
  EXPECT_TRUE(bad.tile_size == was);

  // The VIDEO checkpoint must NOT claim it.
  const char* vid = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (vid == nullptr || *vid == '\0') { return; }
  MetalMiniMaxH3VideoVae::Config vcfg;
  std::string verr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(vid, vcfg, &verr));
  EXPECT_TRUE(!vcfg.t1_direct);
}

TEST(minimax_h3_vvae, image_round_trip)
{
  const char* img = std::getenv("VPIPE_MINIMAX_H3_IMAGE_VAE_PATH");
  const char* src = std::getenv("VPIPE_MINIMAX_H3_IMAGE_SRC");
  if (img == nullptr || *img == '\0' || src == nullptr || *src == '\0') {
    return;
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  std::string derr;
  auto pic = decode_image_file(sess.ffmpeg_libraries(), src, &derr);
  if (!pic.has_value()) { std::printf("[minimax_h3_vvae] %s\n", derr.c_str()); }
  ASSERT_TRUE(pic.has_value());
  if (!pic.has_value()) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(img, cfg, &cerr));
  ASSERT_TRUE(cfg.t1_direct);
  if (!cfg.t1_direct) { return; }

  // A CENTRE CROP to a whole number of latent cells, not a resample: a
  // resample is a second lossy step and this measures one. 512 is the
  // size the checkpoint's own validation number is quoted at.
  int side = 512;
  if (const char* s = std::getenv("VPIPE_MINIMAX_H3_IMAGE_SIDE")) {
    side = std::atoi(s);
  }
  side = std::min({side, pic->width, pic->height});
  side -= side % cfg.patch;
  ASSERT_TRUE(side >= cfg.patch);
  if (side < cfg.patch) { return; }
  const int x0 = (pic->width - side) / 2, y0 = (pic->height - side) / 2;

  const int IC = cfg.in_channels, H = side, W = side;
  const std::size_t plane = (std::size_t)H * W;
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)IC * plane * 2);
  ASSERT_TRUE(!xb.empty());
  std::vector<float> ref((std::size_t)IC * plane);   // the source, in [0,1]
  {
    auto* d = static_cast<std::uint16_t*>(xb.contents());
    for (int c = 0; c < IC; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const std::size_t s =
              ((std::size_t)c * pic->height + (y0 + y)) * pic->width + x0 + x;
          const float p = (float)pic->rgb[s] / 255.0f;
          const std::size_t i = (std::size_t)c * plane + (std::size_t)y * W + x;
          ref[i] = p;
          d[i] = f32_to_bf16_((p - kInMean[c]) / kInStd[c]);
        }
      }
    }
  }

  // Slice override, for the sweep that shows the checkpoint's own answer
  // is the right one: the four pixel frames of a still's block are near
  // copies of each other, so a wrong slice is a blurred picture rather
  // than a broken one and no shape check would catch it.
  if (const char* s = std::getenv("VPIPE_MINIMAX_H3_IMAGE_SLICE")) {
    cfg.t1_output_slice = std::atoi(s);
  }
  auto m = MetalMiniMaxH3VideoVae::load(img, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  EXPECT_TRUE(m->can_decode_image());

  // ONE still in, ONE latent frame out -- the encoder's own answer, not
  // an assumption: `encoded_frames(1)` is 1 because the causal temporal
  // padding leaves a length-1 axis at length 1.
  std::string eerr;
  SharedBuffer mom = m->encode(xb, 1, H, W, &eerr);
  if (mom.empty()) { std::printf("[minimax_h3_vvae] %s\n", eerr.c_str()); }
  ASSERT_TRUE(!mom.empty());
  if (mom.empty()) { return; }
  ASSERT_TRUE(m->encoded_frames(1) == 1);

  const int lh = H / cfg.patch, lw = W / cfg.patch;
  const std::size_t voxels = (std::size_t)lh * lw;
  // Moments are mean|logvar on the channel axis, so the mean is the
  // leading z_channels planes -- already channel-first, no repacking.
  SharedBuffer zb = mc->make_shared_buffer(voxels * cfg.z_channels * 2);
  ASSERT_TRUE(!zb.empty());
  std::memcpy(zb.contents(), mom.contents(),
              voxels * (std::size_t)cfg.z_channels * 2);

  // PSNR over the two, in [0, 1] -- the quantity the checkpoint's own
  // validation number is stated in, so the two are comparable.
  auto psnr_of = [&](const SharedBuffer& out) {
    const auto* o = static_cast<const std::uint16_t*>(out.contents());
    double se = 0.0;
    for (int c = 0; c < IC; ++c) {
      for (std::size_t i = 0; i < plane; ++i) {
        const std::size_t k = (std::size_t)c * plane + i;
        const float v = bf16_to_f32_(o[k]) * kInStd[c] + kInMean[c];
        const double d = (double)v - (double)ref[k];
        se += d * d;
      }
    }
    const double mse = se / (double)((std::size_t)IC * plane);
    return mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
  };

  // TILED against WHOLE-IMAGE, because the two are different functions
  // and only one of them is what the checkpoint was validated as. The
  // decoder's rope is normalized to the extent it is handed, so a 256
  // tile and a whole 512 image put every token at a different
  // coordinate; the video path tiles because attention is quadratic,
  // which is a cost argument and not a correctness one at image sizes.
  //
  // ONE MODEL RESIDENT AT A TIME. `load(dir, ...)` opens a PRIVATE
  // weight set, so holding both arms at once is two copies of a 5.2 GB
  // checkpoint -- which on a 16 GB box is not a slow test but a wedged
  // machine. The first is released before the second is opened, and the
  // latent outlives both because it is an ordinary buffer.
  std::string terr;
  SharedBuffer tiled = m->decode_image(zb, lh, lw, &terr);
  if (tiled.empty()) { std::printf("[minimax_h3_vvae] %s\n", terr.c_str()); }
  ASSERT_TRUE(!tiled.empty());
  if (tiled.empty()) { return; }
  const double tiled_db = psnr_of(tiled);
  const int tile_was = cfg.tile_size;
  tiled = SharedBuffer{};
  m.reset();

  MetalMiniMaxH3VideoVae::Config wcfg = cfg;
  wcfg.tile_size = std::max(H, W);          // one tile = no tiling
  auto wm = MetalMiniMaxH3VideoVae::load(img, mc, wcfg);
  ASSERT_TRUE(wm != nullptr);
  if (wm == nullptr) { return; }
  std::string werr;
  SharedBuffer whole = wm->decode_image(zb, lh, lw, &werr);
  if (whole.empty()) { std::printf("[minimax_h3_vvae] %s\n", werr.c_str()); }
  ASSERT_TRUE(!whole.empty());
  if (whole.empty()) { return; }
  const double whole_db = psnr_of(whole);

  std::printf("[minimax_h3_vvae] image round trip %dx%d (latent %dx%d), slice "
              "%d: PSNR %.2f dB tiled@%d, %.2f dB whole-image | the "
              "checkpoint reports 30.44 dB over a 64-image 512px set\n",
              H, W, lh, lw, cfg.t1_output_slice, tiled_db, tile_was, whole_db);

  // A FLOOR, not the published number: that one is a mean over 64
  // images and this is one crop. What it catches is the failure that
  // looks like success -- a wrong output slice, the un-fine-tuned
  // decoder, or the wrong pixel normalization, all of which land far
  // below this while still producing a picture-shaped buffer.
  EXPECT_TRUE(tiled_db > 25.0);

  // TILING AT THE CHECKPOINT'S OWN 256 IS REQUIRED, not an optimization,
  // and this is the assertion that says so. MEASURED on a 512px centre
  // crop: 30.97 dB tiled against 23.27 dB whole-image -- 7.7 dB, and the
  // tiled arm lands on the 30.44 dB the checkpoint reports while the
  // whole-image arm is nowhere near it.
  //
  // The reason is the rope, not the seams. `build_rope_` normalizes
  // every axis to the extent it is HANDED, so a 512px decode puts each
  // token at a coordinate no 256px tile ever occupies -- and this
  // decoder was fine-tuned on a 256 -> 384 px curriculum. Bigger tiles
  // are not "less seam for more compute" here; they are a different
  // function, and a worse one.
  //
  // ONLY ABOVE THE TILE. At or below `tile_size` the image is a single
  // tile and the two arms are the SAME function -- MEASURED 28.14 dB
  // against 28.14 dB at 256px -- so asserting a strict win there fails
  // on an equality rather than on a defect.
  if (H > tile_was || W > tile_was) {
    EXPECT_TRUE(tiled_db > whole_db);
  } else {
    EXPECT_TRUE(std::fabs(tiled_db - whole_db) < 1e-6);
  }
}

TEST(minimax_h3_vvae, decode_image_refuses_a_video_checkpoint)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalMiniMaxH3VideoVae::Config cfg;
  std::string cerr;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &cerr));
  ASSERT_TRUE(!cfg.t1_direct);
  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  EXPECT_TRUE(!m->can_decode_image());
  const int lh = 4, lw = 4;
  SharedBuffer zb =
      mc->make_shared_buffer((std::size_t)lh * lw * cfg.z_channels * 2);
  ASSERT_TRUE(!zb.empty());
  std::memset(zb.contents(), 0, zb.byte_size());
  std::string err;
  // Refused, and the refusal must SAY the checkpoint is the problem --
  // the weights are all present and would produce a picture-shaped
  // buffer, so a caller has no other way to tell.
  EXPECT_TRUE(m->decode_image(zb, lh, lw, &err).empty());
  EXPECT_TRUE(err.find("h3_t1_direct") != std::string::npos);
}
