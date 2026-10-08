// FlashVSR's decoder: the Wan 2.1 VAE, read from a NATIVELY-named
// checkpoint.
//
// The v1.1 release ships two decoders over one denoiser -- the full
// variant decodes through this VAE, the tiny variant through its own
// conditional decoder -- and that is exactly what the latent boundary
// between generate-video and vae-decode buys. This tier is the full one.
//
// WHAT IS NEW HERE IS THE NAMING, not the net. `Wan2.1_VAE.pth` carries
// the upstream research spelling (`decoder.upsamples.4.residual.2`)
// where every loader in this tree reads the diffusers one
// (`decoder.up_blocks.1.resnets.0.conv1`). The two are a bijection over
// the same 194 tensors, and shared/wan-vae-names.h is the structural map
// between them -- previously used only by the image VAE. A WRONG INDEX
// IS SILENT: every tensor at one level has the same shape, so a
// mis-grouped resnet loads cleanly and decodes a plausible, wrong clip.
// That is what this test exists to catch.
//
// The latent is the WHITENED one generate-video emits, so it is
// un-whitened here first -- the decoder takes un-whitened input, and the
// statistics live in the checkpoint's config rather than in the caller.
//
// Env, both required or the test skips:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  the converted FlashVSR-v1.1 dir
//   VPIPE_FLASHVSR_GOLDEN           a dir `gen_goldens.py --decode` wrote

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& ref)
{
  if (got.size() != ref.size() || got.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::vector<float>
read_f32_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<float> out((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

}  // namespace

TEST(flashvsr_vae, native_names_decode_matches_reference)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  const char* gold = std::getenv("VPIPE_FLASHVSR_GOLDEN");
  if (root == nullptr || gold == nullptr || *root == '\0' || *gold == '\0') {
    return;
  }
  const fs::path g(gold);
  const auto zin = read_f32_(g / "latent.f32");
  const auto ref = read_f32_(g / "vae_frames.f32");
  // The decode golden is opt-in on the generator, so a golden set
  // without it skips rather than failing.
  if (zin.empty() || ref.empty()) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // `vae/` when converted; `Wan2.1_VAE.pth` as published, which carries
  // no config and takes Wan 2.1's own from its tensors.
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout) && !layout.vae.empty());
  const std::string vdir = layout.vae;
  MetalWanVae::Config cfg;
  std::string err;
  const bool have_cfg =
      MetalWanVae::config_from_json(vdir, cfg, &err) ||
      MetalWanVae::config_for_native_checkpoint(vdir, cfg, &err);
  ASSERT_TRUE(have_cfg);
  if (!have_cfg) {
    std::printf("  vae config: %s\n", err.c_str());
    return;
  }
  // The whitening statistics have to be THERE, not defaulted: an
  // un-whiten with a unit scale decodes a washed-out clip rather than
  // failing.
  EXPECT_TRUE(cfg.latents_mean.size() == (std::size_t)cfg.z_dim);
  EXPECT_TRUE(cfg.latents_std.size() == (std::size_t)cfg.z_dim);
  auto m = MetalWanVae::load(vdir, mc, cfg, /*with_encoder=*/false);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  // Geometry comes from the golden's own sizes, so a shape change in the
  // generator cannot silently compare the wrong tensors.
  const std::size_t nz = zin.size() / (std::size_t)cfg.z_dim;
  const std::size_t npx = ref.size() / 3;
  int T = 0, hh = 0, ww = 0, F = 0, H = 0, W = 0;
  for (int t = 1; t <= 64 && T == 0; ++t) {
    if (nz % (std::size_t)t != 0) { continue; }
    const std::size_t hw = nz / (std::size_t)t;
    const int side = (int)std::lround(std::sqrt((double)hw));
    if ((std::size_t)side * side != hw) { continue; }
    const int f = MetalWanVae::video_frames(t);
    if ((std::size_t)f * (std::size_t)(side * 8) * (side * 8) != npx) {
      continue;
    }
    T = t; hh = side; ww = side; F = f; H = side * 8; W = side * 8;
  }
  ASSERT_TRUE(T > 0);
  if (T == 0) { return; }

  SharedBuffer z = mc->make_shared_buffer(zin.size() * 2);
  ASSERT_TRUE(!z.empty());
  if (z.empty()) { return; }
  {
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < zin.size(); ++i) { d[i] = (_Float16)zin[i]; }
  }
  SharedBuffer zw = m->unwhiten(z, T, hh, ww);
  ASSERT_TRUE(!zw.empty());
  if (zw.empty()) { return; }

  const std::size_t hw = (std::size_t)H * W;
  std::vector<float> got((std::size_t)3 * F * hw, 0.0f);
  int seen = 0;
  const bool ok = m->decode(
      zw, T, hh, ww,
      [&](const SharedBuffer& rgb, int frame0, int n) {
        const auto* s = static_cast<const _Float16*>(rgb.contents());
        for (int c = 0; c < 3; ++c) {
          for (int f = 0; f < n; ++f) {
            if (frame0 + f >= F) { continue; }
            for (std::size_t p = 0; p < hw; ++p) {
              // The reference clamps its decode to [-1, 1]; comparing
              // against an unclamped one would charge this port for a
              // step the reference took.
              const float v = (float)s[((std::size_t)c * n + f) * hw + p];
              got[((std::size_t)c * F + (frame0 + f)) * hw + p] =
                  v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
            }
          }
        }
        seen += n;
        return true;
      },
      &err);
  if (!ok) { std::printf("  decode: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  if (!ok) { return; }
  EXPECT_TRUE(seen == F);

  const double r = rel_l2_(got, ref);
  std::printf("  vae decode rel-L2 %.6f  ([16,%d,%d,%d] -> [3,%d,%d,%d])\n", r,
              T, hh, ww, F, H, W);
  EXPECT_TRUE(r < 0.05);
}

// THE FRAME SPLIT IS FREE, AND THE ESTIMATE IS A MEASUREMENT.
//
// decode() runs everything past the last temporal upsample one frame at a
// time. Nothing there mixes frames except through a carry, so it has to be
// the same arithmetic as the whole chunk -- BIT-identical, not close. And
// it has to cost what decode_peak_bytes() says, because that figure is
// what the plan books and what the preflight refuses on: under-stated it
// admits a decode into swap, over-stated it refuses one that fits.
//
// No golden needed. A synthetic latent exercises both, and three latent
// frames is the shortest clip with a four-frame chunk in it. The band is
// pinned so it is a known constant to take out of the peak rather than
// whatever headroom this box has.
//
// Env: VPIPE_FLASHVSR_TEST_MODEL_PATH; VPIPE_WAN_VAE_TEST_GRID=<h8> sizes
// the square latent (default 32).
TEST(flashvsr_vae, frame_split_is_exact_and_bounded)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout) && !layout.vae.empty());
  if (layout.vae.empty()) { return; }
  MetalWanVae::Config cfg;
  std::string err;
  const bool have_cfg =
      MetalWanVae::config_from_json(layout.vae, cfg, &err) ||
      MetalWanVae::config_for_native_checkpoint(layout.vae, cfg, &err);
  ASSERT_TRUE(have_cfg);
  if (!have_cfg) { return; }
  auto m = MetalWanVae::load(layout.vae, mc, cfg, /*with_encoder=*/false);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  int g = 32;
  if (const char* e = std::getenv("VPIPE_WAN_VAE_TEST_GRID")) {
    if (std::atoi(e) > 0) { g = std::atoi(e); }
  }
  const int T = 3;
  const int F = MetalWanVae::video_frames(T);
  const std::size_t nz = (std::size_t)cfg.z_dim * T * g * g;
  SharedBuffer z = mc->make_shared_buffer(nz * 2);
  ASSERT_TRUE(!z.empty());
  if (z.empty()) { return; }
  {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)nd(rng); }
  }

  const int band_rows = 16;
  const std::size_t band_bytes = (std::size_t)band_rows * 27 *
                                 cfg.base_dim * cfg.dim_mult[1] * 2;
  ::setenv("VPIPE_WAN_VAE_BAND_ROWS", std::to_string(band_rows).c_str(), 1);

  const std::size_t hw = (std::size_t)g * 8 * g * 8;
  struct Arm {
    std::vector<std::uint16_t> px;
    std::size_t peak = 0;
    bool ok = false;
  };
  auto run = [&](bool split) {
    Arm a;
    if (split) {
      ::unsetenv("VPIPE_WAN_VAE_NO_FRAME_SPLIT");
    } else {
      ::setenv("VPIPE_WAN_VAE_NO_FRAME_SPLIT", "1", 1);
    }
    a.px.assign((std::size_t)3 * F * hw, 0);
    const std::size_t live0 =
        metal_compute::shared_buffer_memory_stats().live_bytes;
    metal_compute::shared_buffer_reset_peak();
    std::string e;
    a.ok = m->decode(
        z, T, g, g,
        [&](const SharedBuffer& rgb, int frame0, int n) {
          const auto* s = static_cast<const std::uint16_t*>(rgb.contents());
          for (int c = 0; c < 3; ++c) {
            for (int f = 0; f < n; ++f) {
              std::memcpy(
                  a.px.data() + ((std::size_t)c * F + frame0 + f) * hw,
                  s + ((std::size_t)c * n + f) * hw, hw * 2);
            }
          }
          return true;
        },
        &e);
    if (!a.ok) { std::printf("  decode: %s\n", e.c_str()); }
    const std::size_t pk =
        metal_compute::shared_buffer_memory_stats().peak_bytes;
    a.peak = pk > live0 ? pk - live0 : 0;
    return a;
  };
  const Arm whole = run(false);
  const Arm split = run(true);
  ::unsetenv("VPIPE_WAN_VAE_NO_FRAME_SPLIT");
  ::unsetenv("VPIPE_WAN_VAE_BAND_ROWS");
  ASSERT_TRUE(whole.ok && split.ok);
  if (!whole.ok || !split.ok) { return; }

  EXPECT_TRUE(std::memcmp(whole.px.data(), split.px.data(),
                          whole.px.size() * 2) == 0);
  for (const bool s : {false, true}) {
    const Arm& a = s ? split : whole;
    const std::size_t used = a.peak > band_bytes ? a.peak - band_bytes : 0;
    const std::size_t est = MetalWanVae::decode_peak_bytes(cfg, g, g, s);
    const double ratio = used > 0 ? (double)est / (double)used : 0.0;
    std::printf("  %s %dx%d: measured %.1f MB (+%.1f MB band), "
                "estimate %.1f MB (x%.2f)\n", s ? "split" : "whole",
                g * 8, g * 8, (double)used / 1048576.0,
                (double)band_bytes / 1048576.0, (double)est / 1048576.0,
                ratio);
    EXPECT_TRUE(est >= used);
    EXPECT_TRUE(ratio < 1.15);
  }
  EXPECT_TRUE(split.peak < whole.peak);
}

// The SPATIALLY TILED tail against the whole-plane one, at a size where
// both fit. The claim tiling rests on is that it is not an approximation:
// the mid attention -- the only layer here that reads the whole plane --
// runs above the split, and every layer below it is local, so a tile
// carrying the receptive field on each interior side reproduces the
// untiled pixels EXACTLY. Bit-for-bit, not within a tolerance.
//
// Env: VPIPE_FLASHVSR_TEST_MODEL_PATH; VPIPE_WAN_VAE_TEST_GRID=<h8>.
TEST(flashvsr_vae, tiled_tail_is_exact_and_smaller)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout) && !layout.vae.empty());
  if (layout.vae.empty()) { return; }
  MetalWanVae::Config cfg;
  std::string err;
  const bool have_cfg =
      MetalWanVae::config_from_json(layout.vae, cfg, &err) ||
      MetalWanVae::config_for_native_checkpoint(layout.vae, cfg, &err);
  ASSERT_TRUE(have_cfg);
  if (!have_cfg) { return; }
  auto m = MetalWanVae::load(layout.vae, mc, cfg, /*with_encoder=*/false);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  int g = 32;
  if (const char* e = std::getenv("VPIPE_WAN_VAE_TEST_GRID")) {
    if (std::atoi(e) > 0) { g = std::atoi(e); }
  }
  const int T = 3;                       // one opening chunk plus a steady one
  const int F = MetalWanVae::video_frames(T);
  const std::size_t nz = (std::size_t)cfg.z_dim * T * g * g;
  SharedBuffer z = mc->make_shared_buffer(nz * 2);
  ASSERT_TRUE(!z.empty());
  if (z.empty()) { return; }
  {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)nd(rng); }
  }

  // A fixed band, so what is measured is the decode and not the im2col
  // scratch a box without the hardware conv sizes from its free memory
  // (it shrank with the tiles and hid that tiling saved nothing).
  const int band_rows = 16;
  const std::size_t band_bytes = (std::size_t)band_rows * 27 *
                                 cfg.base_dim * cfg.dim_mult[1] * 2;
  ::setenv("VPIPE_WAN_VAE_BAND_ROWS", std::to_string(band_rows).c_str(), 1);
  const std::size_t hw = (std::size_t)g * 8 * g * 8;
  struct Arm {
    std::vector<std::uint16_t> px;
    std::size_t peak = 0;
    bool ok = false;
  };
  auto run = [&](const char* grid) {
    Arm a;
    if (grid == nullptr) {
      ::unsetenv("VPIPE_WAN_VAE_TILE");
    } else {
      ::setenv("VPIPE_WAN_VAE_TILE", grid, 1);
    }
    a.px.assign((std::size_t)3 * F * hw, 0);
    const std::size_t live0 =
        metal_compute::shared_buffer_memory_stats().live_bytes;
    metal_compute::shared_buffer_reset_peak();
    std::string e;
    a.ok = m->decode(
        z, T, g, g,
        [&](const SharedBuffer& rgb, int frame0, int n) {
          const auto* s = static_cast<const std::uint16_t*>(rgb.contents());
          for (int c = 0; c < 3; ++c) {
            for (int f = 0; f < n; ++f) {
              std::memcpy(
                  a.px.data() + ((std::size_t)c * F + frame0 + f) * hw,
                  s + ((std::size_t)c * n + f) * hw, hw * 2);
            }
          }
          return true;
        },
        &e);
    if (!a.ok) { std::printf("  decode: %s\n", e.c_str()); }
    const std::size_t pk =
        metal_compute::shared_buffer_memory_stats().peak_bytes;
    a.peak = pk > live0 ? pk - live0 : 0;
    return a;
  };

  const Arm whole = run(nullptr);
  const Arm tiled = run("2x2");
  const Arm tiled3 = run("3x3");
  // DEEP grids -- below the mid block, on the latent plane -- where the
  // plane is wide enough for a tile past its 16-pixel halo: 2x2 from
  // 512x512, 4x4 from 1024x1024. Skipped, and said so, below that.
  struct Deep { const char* grid; int n; Arm a; bool ran = false; };
  std::vector<Deep> deep = {{"d2x2", 2, {}}, {"d4x4", 4, {}}};
  for (Deep& dd : deep) {
    const auto tt = m->tail_tiles_for(g, g, dd.n, dd.n, /*deep=*/true);
    if (tt.ty <= 1 && tt.tx <= 1) {
      std::printf("  %s: no deep %dx%d grid at %dx%d (tiles under the "
                  "halo)\n", dd.grid, dd.n, dd.n, g * 8, g * 8);
      continue;
    }
    dd.a = run(dd.grid);
    dd.ran = true;
  }
  // THE RE-CUT. Each tile runs the whole clip on its own, so a tile that
  // no longer fits what is free when its turn comes is split then and
  // there. Driven by telling the tiles there is only what a deep 2x2
  // decode needs in all -- of which a tile may take tile_share_() -- so
  // they must be cut finer, hold less, and stay exact.
  Arm recut;
  bool recut_ran = false;
  const auto t22 = m->tail_tiles_for(g, g, 2, 2, /*deep=*/true);
  if (t22.ty > 1 || t22.tx > 1) {
    const std::size_t est22 =
        MetalWanVae::decode_peak_bytes(cfg, g, g, true, t22.frac, T, true);
    ::setenv("VPIPE_WAN_VAE_TILE_ROOM_MB",
             std::to_string(est22 >> 20).c_str(), 1);
    recut = run("d2x2");
    recut_ran = true;
  }
  // THE BAND PLANNER, the same squeeze: the tiles are laid out a band at a
  // time from what the hook says is free, rather than from a forced grid.
  Arm planned;
  bool planned_ran = false;
  if (recut_ran) {
    planned = run("d");
    planned_ran = true;
  }
  ::unsetenv("VPIPE_WAN_VAE_TILE_ROOM_MB");
  ::unsetenv("VPIPE_WAN_VAE_TILE");
  ::unsetenv("VPIPE_WAN_VAE_BAND_ROWS");
  ASSERT_TRUE(whole.ok && tiled.ok && tiled3.ok);
  if (!whole.ok || !tiled.ok || !tiled3.ok) { return; }
  if (planned_ran) {
    ASSERT_TRUE(planned.ok);
    if (planned.ok) {
      EXPECT_TRUE(std::memcmp(whole.px.data(), planned.px.data(),
                              whole.px.size() * 2) == 0);
      std::printf("  %dx%d planned deep bands under the same squeeze: "
                  "%.1f MB\n", g * 8, g * 8,
                  (double)planned.peak / 1048576.0);
      EXPECT_TRUE(planned.peak < whole.peak);
    }
  }
  if (recut_ran) {
    ASSERT_TRUE(recut.ok);
    if (recut.ok) {
      EXPECT_TRUE(std::memcmp(whole.px.data(), recut.px.data(),
                              whole.px.size() * 2) == 0);
      const Arm* d22 = nullptr;
      for (const Deep& dd : deep) {
        if (dd.ran && dd.n == 2) { d22 = &dd.a; }
      }
      if (d22 != nullptr) {
        std::printf("  %dx%d d2x2 re-cut to fit its own figure, shared: "
                    "%.1f MB against %.1f MB uncut\n", g * 8, g * 8,
                    (double)recut.peak / 1048576.0,
                    (double)d22->peak / 1048576.0);
        EXPECT_TRUE(recut.peak < d22->peak);
      }
    }
  }
  for (const Deep& dd : deep) {
    if (!dd.ran) { continue; }
    ASSERT_TRUE(dd.a.ok);
    if (!dd.a.ok) { return; }
  }

  const bool exact = std::memcmp(whole.px.data(), tiled.px.data(),
                                 whole.px.size() * 2) == 0;
  if (!exact) {
    // Where it diverges says WHICH boundary leaked: a halo one short
    // shows up as a seam, not as noise everywhere.
    std::size_t diff = 0, first = 0;
    for (std::size_t i = 0; i < whole.px.size(); ++i) {
      if (whole.px[i] != tiled.px[i]) {
        if (diff == 0) { first = i; }
        ++diff;
      }
    }
    std::printf("  tiled differs in %zu of %zu samples, first at %zu\n",
                diff, whole.px.size(), first);
  }
  EXPECT_TRUE(exact);
  EXPECT_TRUE(std::memcmp(whole.px.data(), tiled3.px.data(),
                          whole.px.size() * 2) == 0);
  // THE POINT OF THE EXERCISE, and what the first tiling missed: a tiled
  // decode HOLDS less -- each tile runs the clip with only its carries --
  // and holds no more than the figure its chooser was told. That tiling
  // kept every tile's carries for the clip and measured 2x2 at 98% of
  // whole on the M5 while its estimate said 47%.
  struct Row { const char* name; const Arm* a; int ty; bool deep; };
  std::vector<Row> rows = {{"whole", &whole, 1, false},
                           {"2x2", &tiled, 2, false},
                           {"3x3", &tiled3, 3, false}};
  for (const Deep& dd : deep) {
    if (!dd.ran) { continue; }
    // The deep pixels are the untiled ones too: every layer below the mid
    // attention is local, so a 16-latent-pixel halo carries all of it.
    const bool dsame = std::memcmp(whole.px.data(), dd.a.px.data(),
                                   whole.px.size() * 2) == 0;
    EXPECT_TRUE(dsame);
    if (!dsame) {
      // WHERE it differs says what leaked: a halo one short is a seam at
      // the tile borders; a different kernel is noise everywhere.
      const int W8 = g * 8;
      std::size_t diff = 0;
      double maxd = 0.0;
      std::vector<std::size_t> per_row((std::size_t)W8, 0),
          per_col((std::size_t)W8, 0), per_frame((std::size_t)F, 0);
      for (std::size_t i = 0; i < whole.px.size(); ++i) {
        if (whole.px[i] == dd.a.px[i]) { continue; }
        ++diff;
        const double a = (double)static_cast<_Float16>(0);
        (void)a;
        _Float16 x, y;
        std::memcpy(&x, &whole.px[i], 2);
        std::memcpy(&y, &dd.a.px[i], 2);
        maxd = std::max(maxd, std::fabs((double)x - (double)y));
        const std::size_t p = i % hw;
        per_row[p / W8]++;
        per_col[p % W8]++;
        per_frame[(i / hw) % F]++;
      }
      std::printf("  %s differs in %zu of %zu samples, max |d| %.4g\n",
                  dd.grid, diff, whole.px.size(), maxd);
      std::printf("   rows:");
      for (int y = 0; y < W8; y += 8) {
        std::size_t c = 0;
        for (int k = 0; k < 8; ++k) { c += per_row[(std::size_t)(y + k)]; }
        std::printf(" %zu", c);
      }
      std::printf("\n   frames:");
      for (int f = 0; f < F; ++f) { std::printf(" %zu", per_frame[f]); }
      std::printf("\n");
    }
    rows.push_back({dd.grid, &dd.a, dd.n, true});
  }
  for (const Row& r : rows) {
    const double frac =
        r.ty > 1 ? m->tail_tiles_for(g, g, r.ty, r.ty, r.deep).frac : 1.0;
    const std::size_t used =
        r.a->peak > band_bytes ? r.a->peak - band_bytes : 0;
    const std::size_t est =
        MetalWanVae::decode_peak_bytes(cfg, g, g, true, frac, T, r.deep);
    const double ratio = used > 0 ? (double)est / (double)used : 0.0;
    std::printf("  %dx%d %s: measured %.1f MB (+%.1f MB band), estimate "
                "%.1f MB (x%.2f)\n", g * 8, g * 8, r.name,
                (double)used / 1048576.0, (double)band_bytes / 1048576.0,
                (double)est / 1048576.0, ratio);
    EXPECT_TRUE(est >= used);
    EXPECT_TRUE(ratio < 1.15);
  }
  // Finer is never larger -- and past ~3x3 not smaller either: the head,
  // which runs the whole plane, is then what the decode peaks at.
  EXPECT_TRUE(tiled.peak < whole.peak);
  EXPECT_TRUE(tiled3.peak <= tiled.peak);
}

// A DECODE BENCH, not a check. VPIPE_WAN_VAE_BENCH="h8,w8,T" decodes a
// synthetic latent of that size and prints the wall time of every chunk,
// so a slow decode can be attributed without a denoiser in front of it.
// The arm is chosen by the decoder's own env toggles
// (VPIPE_WAN_VAE_BAND_ROWS, VPIPE_WAN_VAE_NO_FRAME_SPLIT, VPIPE_WAN_NO_MMA2);
// VPIPE_WAN_VAE_LOG prints the band it picked. A two-latent-frame warmup
// at the same size runs first, so kernel builds and tuning stay out of it.
TEST(flashvsr_vae, decode_bench)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  const char* spec = std::getenv("VPIPE_WAN_VAE_BENCH");
  if (root == nullptr || spec == nullptr || *root == '\0') { return; }
  int h8 = 0, w8 = 0, T = 0;
  ASSERT_TRUE(std::sscanf(spec, "%d,%d,%d", &h8, &w8, &T) == 3 && h8 > 0 &&
              w8 > 0 && T > 0);
  if (h8 <= 0 || w8 <= 0 || T <= 0) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout) && !layout.vae.empty());
  if (layout.vae.empty()) { return; }
  MetalWanVae::Config cfg;
  std::string err;
  const bool have_cfg =
      MetalWanVae::config_from_json(layout.vae, cfg, &err) ||
      MetalWanVae::config_for_native_checkpoint(layout.vae, cfg, &err);
  ASSERT_TRUE(have_cfg);
  if (!have_cfg) { return; }
  auto m = MetalWanVae::load(layout.vae, mc, cfg, /*with_encoder=*/false);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  const std::size_t nz = (std::size_t)cfg.z_dim * T * h8 * w8;
  SharedBuffer z = mc->make_shared_buffer(nz * 2);
  ASSERT_TRUE(!z.empty());
  if (z.empty()) { return; }
  {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)nd(rng); }
  }

  using clk = std::chrono::steady_clock;
  {
    const auto w0 = clk::now();
    const bool ok = m->decode(
        z, std::min(T, 2), h8, w8,
        [](const SharedBuffer&, int, int) { return true; }, &err);
    ASSERT_TRUE(ok);
    if (!ok) { std::printf("  warmup: %s\n", err.c_str()); return; }
    std::printf("  warmup (T=%d): %.0f ms\n", std::min(T, 2),
                std::chrono::duration<double, std::milli>(clk::now() - w0)
                    .count());
  }
  // What the warmup left behind. A decode frees everything it allocated,
  // so a footprint far above the live buffers is memory the process
  // still holds for buffers that no longer exist.
  // Sampled over time, because the driver releases a freed GPU buffer's
  // pages asynchronously: a single sample right after the decode reads
  // them as held.
  {
    const auto rs = mc->residency_stats();
    const auto w1 = clk::now();
    for (int ms : {0, 50, 200, 1000, 3000}) {
      const auto until = w1 + std::chrono::milliseconds(ms);
      while (clk::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      const auto mb = mc->memory_budget();
      std::printf("  after warmup +%4d ms: footprint %zu MB (graphics %zu "
                  "MB), live GPU buffers %zu MB, reclaimable %zu MB\n", ms,
                  mb.self_footprint >> 20, mb.self_graphics >> 20,
                  metal_compute::shared_buffer_memory_stats().live_bytes >>
                      20,
                  mb.available_physical >> 20);
    }
    std::printf("  residency set %zu (+%zu/-%zu)\n", rs.current,
                rs.add_calls, rs.remove_calls);
  }

  metal_compute::shared_buffer_reset_peak();
  const std::size_t live0 =
      metal_compute::shared_buffer_memory_stats().live_bytes;
  std::vector<double> ms;
  int frames = 0;
  const auto t0 = clk::now();
  auto last = t0;
  const bool ok = m->decode(
      z, T, h8, w8,
      [&](const SharedBuffer&, int, int n) {
        const auto now = clk::now();
        ms.push_back(
            std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
        frames += n;
        return true;
      },
      &err);
  const double total =
      std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  if (!ok) { std::printf("  decode: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
  if (!ok || ms.empty()) { return; }

  double steady = 0.0;
  for (std::size_t i = 1; i < ms.size(); ++i) { steady += ms[i]; }
  const std::size_t n_steady = ms.size() - 1;
  const std::size_t peak =
      metal_compute::shared_buffer_memory_stats().peak_bytes - live0;
  std::printf("  %dx%d T=%d -> %d frames: total %.0f ms, chunk0 %.0f ms, "
              "steady %.0f ms/chunk = %.0f ms/frame, peak %zu MB\n",
              w8 * 8, h8 * 8, T, frames, total, ms[0],
              n_steady ? steady / (double)n_steady : 0.0,
              n_steady ? steady / (double)(4 * n_steady) : 0.0, peak >> 20);
}

// THE HARDWARE CONV DECODES THE SAME CLIP AS THE GATHER.
//
// On a GPU with the MPP convolution2d op, every 3x3 and 3x3x3 conv of the
// decoder skips im2col: a causal conv runs as one 2D conv over its three
// taps concatenated channel-wise, against an HWIO twin of its weight. A
// wrong channel order in that twin is not a crash -- it is a plausible,
// wrong clip -- so the whole decoder is held to the gathered one here.
// Two loads, because the route is decided at load (VPIPE_VAE_NO_HWCONV).
// Per conv the two agree to ~2e-5 (conv2d_mma.hw_op_wan_causal_conv3d);
// across the stack that compounds, hence the bound.
TEST(flashvsr_vae, hwconv_decode_matches_gather)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout) && !layout.vae.empty());
  if (layout.vae.empty()) { return; }
  MetalWanVae::Config cfg;
  std::string err;
  const bool have_cfg =
      MetalWanVae::config_from_json(layout.vae, cfg, &err) ||
      MetalWanVae::config_for_native_checkpoint(layout.vae, cfg, &err);
  ASSERT_TRUE(have_cfg);
  if (!have_cfg) { return; }

  const int g = 32, T = 3;
  const int F = MetalWanVae::video_frames(T);
  const std::size_t hw = (std::size_t)g * 8 * g * 8;
  const std::size_t nz = (std::size_t)cfg.z_dim * T * g * g;
  SharedBuffer z = mc->make_shared_buffer(nz * 2);
  ASSERT_TRUE(!z.empty());
  if (z.empty()) { return; }
  {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)nd(rng); }
  }

  auto decode = [&](bool hwconv, std::vector<float>* px) {
    if (hwconv) {
      ::unsetenv("VPIPE_VAE_NO_HWCONV");
    } else {
      ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1);
    }
    auto m = MetalWanVae::load(layout.vae, mc, cfg, /*with_encoder=*/false);
    ::unsetenv("VPIPE_VAE_NO_HWCONV");
    if (m == nullptr) { return false; }
    px->assign((std::size_t)3 * F * hw, 0.0f);
    std::string e;
    const bool ok = m->decode(
        z, T, g, g,
        [&](const SharedBuffer& rgb, int frame0, int n) {
          const auto* s = static_cast<const _Float16*>(rgb.contents());
          for (int c = 0; c < 3; ++c) {
            for (int f = 0; f < n; ++f) {
              for (std::size_t p = 0; p < hw; ++p) {
                (*px)[((std::size_t)c * F + frame0 + f) * hw + p] =
                    (float)s[((std::size_t)c * n + f) * hw + p];
              }
            }
          }
          return true;
        },
        &e);
    if (!ok) { std::printf("  decode: %s\n", e.c_str()); }
    return ok;
  };
  std::vector<float> gathered, hw_out;
  const bool ok_g = decode(false, &gathered);
  const bool ok_h = decode(true, &hw_out);
  ASSERT_TRUE(ok_g && ok_h);
  if (!ok_g || !ok_h) { return; }
  const double r = rel_l2_(hw_out, gathered);
  std::printf("  hardware conv vs gather, %dx%d x%d frames: rel-L2 %.3e\n",
              g * 8, g * 8, F, r);
  EXPECT_TRUE(r < 2e-3);
}
