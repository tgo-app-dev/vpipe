// Boogu-Image-0.1 bring-up smoke. Three tiers, mirroring the FLUX.2 tests:
//   * boogu_smoke.*  -- plumbing + SELF-consistency: the model classes load, a
//     forward produces a finite output of the expected shape, the reference-
//     image path actually changes the velocity, the zero-padded steel attention
//     matches the scalar SDPA it replaces, and the VAE round-trips an image.
//     None of these need an external reference.
//   * boogu_golden.* -- numerical rel-L2 vs the CANONICAL implementation, when
//     a golden is supplied (VPIPE_BOOGU_GOLDEN). Produce one with
//     tools/dump_boogu_golden.py, which runs the reference
//     BooguImageTransformer2DModel on CPU in bf16 over deterministic inputs.
//     Measured on Edit-Turbo: 0.036 (64 img + 16 txt, with and without a
//     reference image) and 0.021 (256 img + 64 txt) -- the error SHRINKS with
//     sequence length, i.e. bf16 noise averaging out rather than a systematic
//     rope/masking error.
//   * boogu_e2e.*    -- the conditioner -> generate-image -> vae-decode pipeline
//     produces a coherent image (opt-in, heavy: the 10B DiT + the 8B mllm).
//
// Env: VPIPE_BOOGU_TEST_MODEL_PATH = the Boogu-Image model root.
// VPIPE_BOOGU_E2E set = run the end-to-end pipeline test. Unset => skip.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "apple-silicon/tensor-beat.h"
#include "generative-models/boogu/metal-boogu-transformer.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/load-image-stage.h"
#include "stages/save-image-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/vae-decode-stage.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::ComputeLibrary;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

// minitest's ASSERT_TRUE is #defined to EXPECT_TRUE -- it reports and CONTINUES.
// Every precondition that guards a later buffer access therefore needs an
// explicit return, or a mismatch turns into a heap over-read that can take the
// whole test binary down with it (this file's golden tests compare
// externally-sized arrays, so that is not hypothetical).
#define BOOGU_REQUIRE(cond)                                                    \
  do { if (!(cond)) { EXPECT_TRUE(cond); return; } } while (0)

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

bool all_finite_bf16_(const SharedBuffer& b, std::size_t n)
{
  if (b.empty() || b.byte_size() < n * 2) { return false; }
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(bf16_to_f32_(p[i]))) { return false; }
  }
  return true;
}

// Relative L2 between two bf16 buffers.
double rel_l2_bf16_(const SharedBuffer& a, const SharedBuffer& b, std::size_t n)
{
  const auto* x = static_cast<const std::uint16_t*>(a.contents());
  const auto* y = static_cast<const std::uint16_t*>(b.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double u = bf16_to_f32_(x[i]), v = bf16_to_f32_(y[i]);
    num += (u - v) * (u - v);
    den += v * v;
  }
  return den > 0.0 ? std::sqrt(num / den) : (num == 0.0 ? 0.0 : 1.0);
}

void fill_normal_bf16_(const SharedBuffer& b, std::size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { p[i] = f32_to_bf16_(nd(rng)); }
}

std::string model_root_()
{
  const char* r = std::getenv("VPIPE_BOOGU_TEST_MODEL_PATH");
  return (r != nullptr && *r != '\0') ? std::string(r) : std::string();
}

}  // namespace

// The DiT loads and one forward yields a finite velocity of the right shape.
// VPIPE_BOOGU_BENCH_GRID/_TS/_ITERS parametrize it for profiling (pair with
// VPIPE_BOOGU_DIT_PROFILE for the per-section breakdown).
TEST(boogu_smoke, dit_forward_shape_finite)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto dit = MetalBooguTransformer::load(root + "/transformer", mc,
                                         MetalBooguTransformer::Config{});
  BOOGU_REQUIRE(dit != nullptr);
  const auto& c = dit->config();
  // The whole point of the config-driven load: these come off config.json and
  // the weight shapes, not the Config defaults.
  EXPECT_TRUE(c.hidden == 3360 && c.n_heads == 28 && c.n_kv_heads == 7);
  EXPECT_TRUE(c.head_dim == 120);
  EXPECT_TRUE(c.n_double == 8 && c.n_single == 32 && c.n_refiner == 2);
  EXPECT_TRUE(c.ff_inner == 13568);       // 4*3360 rounded up to 256
  EXPECT_TRUE(c.out_channels == 64);      // patch^2 * 16
  EXPECT_TRUE(c.instruct_dim == 4096);

  auto envi = [](const char* k, int d) {
    const char* e = std::getenv(k); return (e && *e) ? std::atoi(e) : d; };
  const int grid = envi("VPIPE_BOOGU_BENCH_GRID", 4);   // token grid (latent/2)
  const int TS = envi("VPIPE_BOOGU_BENCH_TS", 8);
  const int iters = envi("VPIPE_BOOGU_BENCH_ITERS", 1);
  const int lh = grid * c.patch, lw = grid * c.patch;
  const int img_seq = grid * grid;
  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
  SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
  BOOGU_REQUIRE(!ctx.empty() && !lat.empty());
  fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 99);
  fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 7);

  // VPIPE_BOOGU_BENCH_REFS reference images at the same grid, so the bench can
  // reproduce the edit graph's joint sequence (refs + target + instruction) and
  // not just the text-to-image one.
  std::vector<MetalBooguTransformer::RefImage> refs;
  for (int r = 0; r < envi("VPIPE_BOOGU_BENCH_REFS", 0); ++r) {
    MetalBooguTransformer::RefImage ri;
    ri.latents = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
    if (ri.latents.empty()) { break; }
    fill_normal_bf16_(ri.latents, (std::size_t)img_seq * c.x_in(),
                      (unsigned)(21 + r));
    ri.seq = img_seq; ri.grid_h = lh; ri.grid_w = lw;
    refs.push_back(std::move(ri));
  }

  SharedBuffer vel;
  double best_ms = 1e30, sum_ms = 0.0;
  for (int it = 0; it < iters; ++it) {
    const auto t0 = std::chrono::steady_clock::now();
    vel = dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.5f, refs);
    BOOGU_REQUIRE(!vel.empty());
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    sum_ms += ms;
    if (it > 0 || iters == 1) { best_ms = ms < best_ms ? ms : best_ms; }
  }
  if (iters > 1) {
    // Iteration 0 pays the pipeline-state builds and the first touch of 20 GB
    // of weights; the steady-state number is the one to quote.
    std::printf("[boogu_smoke] DiT forward seq=%d (img %d x%d ref + txt %d): "
                "best %.0f ms, mean %.0f ms over %d iters\n",
                img_seq * (1 + (int)refs.size()) + TS, img_seq,
                (int)refs.size(), TS, best_ms, sum_ms / iters, iters);
  }
  const std::size_t n = (std::size_t)img_seq * c.out_channels;
  BOOGU_REQUIRE(vel.byte_size() >= n * 2);
  EXPECT_TRUE(all_finite_bf16_(vel, n));
  double mag = 0.0;
  {
    const auto* p = static_cast<const std::uint16_t*>(vel.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const double v = bf16_to_f32_(p[i]); mag += v * v;
    }
    mag = std::sqrt(mag / (double)n);
  }
  // A flow-matching velocity on N(0,1) latents is O(1): a dead path gives 0 and
  // a broken residual stream blows up. This is a coarse sanity band, not a
  // golden.
  EXPECT_TRUE(mag > 1e-3 && mag < 1e3);
  std::printf("[boogu_smoke] DiT forward -> velocity [%d, %d] rms %.4f\n",
              img_seq, c.out_channels, mag);
}

// The reference-image path is LIVE: a forward with a reference latent must stay
// finite, still return only the generated-token velocity (references dropped
// from the output), and DIFFER from the reference-less forward. A dead ref path
// (weights not loaded, tokens not embedded, rope band not applied) would give a
// byte-identical result.
// STREAMED must equal PRELOADED, byte for byte.
//
// Both stacks read each block into a reusable slot with pread and issue
// the next block's read under the current block's GPU work
// (shared/block-slots.h). That moves where the bytes come from and when
// they arrive; it must not move a single bit of the result.
//
// This model is the harder case for that mechanism: two block shapes,
// so two slot sets, and a FUSED gate|up in each -- a tensor with no
// checkpoint name, rebuilt from the refilled halves after every read.
// A fuse that ran on stale bytes, or a slot sized for the other stack,
// produces plausible numbers and nothing else would catch it.
//
// VPIPE_BOOGU_STREAM_DUMP writes the streamed velocity, so the same
// test on two builds can be diffed against each other.
TEST(boogu_smoke, streamed_matches_preloaded)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int grid = 4, TS = 8;
  std::vector<float> pre, str;
  double fwd_s = 0.0;

  auto run = [&](bool stream, std::vector<float>& out) -> bool {
    auto dit = MetalBooguTransformer::load(root + "/transformer", mc,
                                           MetalBooguTransformer::Config{},
                                           stream);
    if (dit == nullptr) { return false; }
    const auto& c = dit->config();
    const int lh = grid * c.patch, lw = grid * c.patch;
    const int img_seq = grid * grid;
    SharedBuffer ctx =
        mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
    SharedBuffer lat =
        mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
    if (ctx.empty() || lat.empty()) { return false; }
    // The SAME deterministic inputs either way -- both paths process
    // identical bytes, so agreement is the invariant.
    fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 99);
    fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 7);
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer v = dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.5f);
    fwd_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (v.empty()) { return false; }
    const std::size_t n = (std::size_t)img_seq * c.out_channels;
    out.resize(n);
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(vp[i]); }
    return true;
  };

  BOOGU_REQUIRE(run(false, pre));
  const double pre_s = fwd_s;
  BOOGU_REQUIRE(run(true, str));
  std::printf("[boogu_smoke] forward: preloaded %.1f s, streamed %.1f s "
              "(slots %s)\n", pre_s, fwd_s,
              std::getenv("VPIPE_BOOGU_NO_SLOTS") != nullptr ? "OFF" : "on");
  BOOGU_REQUIRE(pre.size() == str.size() && !pre.empty());

  std::size_t diff = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < pre.size(); ++i) {
    if (pre[i] != str[i]) {
      ++diff;
      worst = std::max(worst, (double)std::fabs(pre[i] - str[i]));
    }
  }
  std::printf("[boogu_smoke] streamed vs preloaded: %zu of %zu differ "
              "(worst %.3e)\n", diff, pre.size(), worst);
  EXPECT_TRUE(diff == 0);

  if (const char* out = std::getenv("VPIPE_BOOGU_STREAM_DUMP")) {
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(str.data()),
            (std::streamsize)(str.size() * sizeof(float)));
  }
}

TEST(boogu_smoke, dit_reference_image_changes_output)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto dit = MetalBooguTransformer::load(root + "/transformer", mc,
                                         MetalBooguTransformer::Config{});
  BOOGU_REQUIRE(dit != nullptr);
  const auto& c = dit->config();
  const int grid = 4, TS = 8;
  const int lh = grid * c.patch, lw = grid * c.patch;
  const int img_seq = grid * grid;
  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
  SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
  fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 99);
  fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 7);

  SharedBuffer v0 = dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.5f);
  BOOGU_REQUIRE(!v0.empty());

  MetalBooguTransformer::RefImage r;
  r.latents = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
  fill_normal_bf16_(r.latents, (std::size_t)img_seq * c.x_in(), 31);
  r.seq = img_seq; r.grid_h = lh; r.grid_w = lw;
  // RefImage owns a move-only SharedBuffer, so build the vector by move.
  std::vector<MetalBooguTransformer::RefImage> refs;
  refs.push_back(std::move(r));
  SharedBuffer v1 = dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.5f, refs);
  BOOGU_REQUIRE(!v1.empty());

  const std::size_t n = (std::size_t)img_seq * c.out_channels;
  // Same output length: the reference tokens are NOT returned.
  EXPECT_TRUE(v1.byte_size() >= n * 2);
  EXPECT_TRUE(all_finite_bf16_(v1, n));
  const double d = rel_l2_bf16_(v1, v0, n);
  EXPECT_TRUE(d > 1e-3);
  std::printf("[boogu_smoke] reference latent shifts the velocity by rel-L2 "
              "%.4f\n", d);
}

// The vec4 adaLN / tanh-gate / residual twins do the SAME arithmetic per
// element as the scalar kernels they replace (only the thread mapping and the
// access width change), so the velocity must come out BIT-IDENTICAL. Anything
// else means the 2-D grid indexing or the row stride is wrong.
TEST(boogu_smoke, vec4_elementwise_matches_scalar)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int grid = 4, TS = 8;
  MetalBooguTransformer::Config cfg;
  const int lh = grid * cfg.patch, lw = grid * cfg.patch;
  const int img_seq = grid * grid;

  SharedBuffer ctx, lat;
  auto run = [&](bool v4) -> SharedBuffer {
    if (v4) { unsetenv("VPIPE_BOOGU_NO_ELT_V4"); }
    else    { setenv("VPIPE_BOOGU_NO_ELT_V4", "1", 1); }
    auto dit = MetalBooguTransformer::load(root + "/transformer", mc, cfg);
    if (!dit) { return {}; }
    const auto& c = dit->config();
    if (ctx.empty()) {
      ctx = mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
      lat = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
      fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 7);
      fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 13);
    }
    return dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.3f);
  };
  SharedBuffer v_v4 = run(true);
  SharedBuffer v_sc = run(false);
  unsetenv("VPIPE_BOOGU_NO_ELT_V4");
  BOOGU_REQUIRE(!v_v4.empty() && !v_sc.empty());
  const std::size_t n = (std::size_t)img_seq * 64;
  const auto* a4 = static_cast<const std::uint16_t*>(v_v4.contents());
  const auto* a1 = static_cast<const std::uint16_t*>(v_sc.contents());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < n; ++i) { diff += (a4[i] != a1[i]) ? 1 : 0; }
  EXPECT_TRUE(diff == 0);
  std::printf("[boogu_smoke] vec4 vs scalar elementwise: %zu/%zu bf16 words "
              "differ\n", diff, n);
}

// The zero-padded-to-128 steel flash attention must match the scalar
// sdpa_full_f16 it replaces. head_dim is 120, so the padded path is only exact
// if the pad lanes contribute nothing -- this is the test that proves it (and
// the only way to catch a pad/unpad stride bug without a golden).
TEST(boogu_smoke, padded_steel_attention_matches_scalar)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int grid = 4, TS = 8;
  MetalBooguTransformer::Config cfg;
  const int lh = grid * cfg.patch, lw = grid * cfg.patch;
  const int img_seq = grid * grid;

  SharedBuffer ctx, lat;
  auto run = [&](bool scalar) -> SharedBuffer {
    if (scalar) { setenv("VPIPE_BOOGU_NO_STEEL_ATTN", "1", 1); }
    else        { unsetenv("VPIPE_BOOGU_NO_STEEL_ATTN"); }
    auto dit = MetalBooguTransformer::load(root + "/transformer", mc, cfg);
    if (!dit) { return {}; }
    const auto& c = dit->config();
    if (ctx.empty()) {
      ctx = mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
      lat = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
      fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 5);
      fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 11);
    }
    return dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.3f);
  };
  SharedBuffer v_steel = run(false);
  SharedBuffer v_scalar = run(true);
  unsetenv("VPIPE_BOOGU_NO_STEEL_ATTN");
  BOOGU_REQUIRE(!v_steel.empty() && !v_scalar.empty());

  MetalBooguTransformer::Config c2;
  const std::size_t n = (std::size_t)img_seq * 64;
  const double d = rel_l2_bf16_(v_steel, v_scalar, n);
  // Not bit-identical: the two kernels accumulate in a different order, and the
  // difference compounds over 46 blocks. Tight enough to catch a real indexing
  // bug (which lands at O(1)), loose enough for bf16 reassociation.
  EXPECT_TRUE(d < 5e-2);
  std::printf("[boogu_smoke] padded-steel vs scalar attention rel-L2 %.5f\n", d);
  (void)c2;
}

// VAE round-trip: encode an image, decode it back, measure PSNR. The FLUX.1
// AutoencoderKL is a strong codec, so a correct patch-1 / scalar-whitening
// generalization round-trips a smooth synthetic image well above 20 dB; a
// broken latent scaling or a wrong pixel scale collapses it.
TEST(boogu_smoke, vae_roundtrip_psnr)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalFlux2Vae::Config vcfg;
  auto vae = MetalFlux2Vae::load(root + "/vae", mc, vcfg, /*with_encoder=*/true);
  BOOGU_REQUIRE(vae != nullptr);
  const auto& c = vae->config();
  // The plain AutoencoderKL shape, all read from vae/config.json.
  EXPECT_TRUE(c.patch == 1);
  EXPECT_TRUE(c.latent_channels == 16);
  EXPECT_TRUE(c.dit_channels() == 16);
  EXPECT_FALSE(c.use_quant_conv);
  EXPECT_FALSE(c.use_post_quant_conv);
  EXPECT_TRUE(std::fabs(c.scaling_factor - 0.3611f) < 1e-4f);
  EXPECT_TRUE(std::fabs(c.shift_factor - 0.1159f) < 1e-4f);

  const int H = 128, W = 128;
  std::vector<float> img((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      // Smooth gradients + a couple of hard edges: enough structure that a
      // dead/mis-scaled codec cannot fake a high PSNR.
      const float u = (float)x / (float)(W - 1), v = (float)y / (float)(H - 1);
      const float box = (x > W / 4 && x < W / 2 && y > H / 4 && y < H / 2)
                            ? 0.6f : -0.4f;
      img[((std::size_t)0 * H + y) * W + x] = 2.0f * u - 1.0f;
      img[((std::size_t)1 * H + y) * W + x] = 2.0f * v - 1.0f;
      img[((std::size_t)2 * H + y) * W + x] = box;
    }
  }
  SharedBuffer in = mc->make_shared_buffer(img.size() * 2);
  BOOGU_REQUIRE(!in.empty());
  { auto* d = static_cast<_Float16*>(in.contents());
    for (std::size_t i = 0; i < img.size(); ++i) { d[i] = (_Float16)img[i]; } }

  SharedBuffer z = vae->encode(in, H, W);
  BOOGU_REQUIRE(!z.empty());
  const int lh = H / 8, lw = W / 8;
  BOOGU_REQUIRE(z.byte_size() >= (std::size_t)16 * lh * lw * 2);
  // The whitened latent should be O(1) -- that is what the shift/scale is for.
  double zrms = 0.0;
  { const auto* p = static_cast<const _Float16*>(z.contents());
    const std::size_t nz = (std::size_t)16 * lh * lw;
    for (std::size_t i = 0; i < nz; ++i) {
      const double v = (double)(float)p[i]; zrms += v * v;
    }
    zrms = std::sqrt(zrms / (double)nz); }
  EXPECT_TRUE(zrms > 0.05 && zrms < 20.0);

  std::string derr;
  SharedBuffer rgb = vae->decode(z, lh, lw, &derr);
  BOOGU_REQUIRE(!rgb.empty());
  // patch 1 => 8 pixels per latent cell.
  BOOGU_REQUIRE(rgb.byte_size() >= (std::size_t)3 * H * W * 2);
  double mse = 0.0;
  { const auto* p = static_cast<const _Float16*>(rgb.contents());
    for (std::size_t i = 0; i < img.size(); ++i) {
      const double d = (double)(float)p[i] - (double)img[i]; mse += d * d;
    }
    mse /= (double)img.size(); }
  // Signal range is [-1,1] => peak-to-peak 2.
  const double psnr = 10.0 * std::log10(4.0 / (mse > 0.0 ? mse : 1e-12));
  std::printf("[boogu_smoke] VAE round-trip %dx%d: latent rms %.3f, PSNR "
              "%.2f dB\n", W, H, zrms, psnr);
  EXPECT_TRUE(psnr > 20.0);
}

// The VAE's mid-block self-attention runs on the matrix-core FULL flash kernel
// (sdpa_full_mma2_d512) on a matrix-core GPU and on the simdgroup-matrix flash
// (sdpa_full_mma_f16) elsewhere; the two must agree, in the ENCODER and in the
// DECODER. NOTE the matrix-core skip below, and see
// vae_selected_mid_attn_matches_scalar at the end of this file for the check
// that covers the member the autotune ACTUALLY picked on any GPU -- this one
// is blind on the machines where that is neither of the two named above. Boogu's VAE is a plain AutoencoderKL at latent 16 -- its mid block is
// 16x16 = 256 tokens, a shape flux2_smoke.vae_decode_flash_attn_matches_scalar
// (patch 2, latent 32, 4096 mid tokens, decode only) never reaches. The two
// halves are reported SEPARATELY because they fail independently: the encoder's
// mid attention feeds a residual, so a wrong one moves the latent without
// moving its rms, which is exactly how vae_roundtrip_psnr sees it (a bad PSNR
// under an innocent-looking `latent rms`).
TEST(boogu_smoke, vae_flash_attn_matches_scalar)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  // Without matrix cores _use_attn_mma2 is false and both arms are the same
  // kernel -- the comparison would be vacuously 0.
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }

  // Sweep the mid-block token count: 256 (16x16), 1024, 4096. The flux2 test
  // only ever reaches 4096, and a tiled flash kernel can be exact at one tile
  // count and wrong at another.
  for (const int px : {128, 256, 512}) {
  const int H = px, W = px, lh = H / 8, lw = W / 8;
  std::vector<float> img((std::size_t)3 * H * W);
  for (std::size_t i = 0; i < img.size(); ++i) {
    img[i] = std::sin((float)i * 0.017f);
  }
  std::vector<float> lat((std::size_t)16 * lh * lw);
  { std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : lat) { v = nd(rng); } }

  // Both halves in ONE load per arm, so the arms differ only in the kernel.
  auto run = [&](bool scalar, std::vector<float>* z_out,
                 std::vector<float>* rgb_out) {
    if (scalar) { ::setenv("VPIPE_FLUX2_NO_MMA_ATTN", "1", 1); }
    auto vae = MetalFlux2Vae::load(root + "/vae", mc,
                                   MetalFlux2Vae::Config{},
                                   /*with_encoder=*/true);
    if (scalar) { ::unsetenv("VPIPE_FLUX2_NO_MMA_ATTN"); }
    if (vae == nullptr) { return; }

    SharedBuffer in = mc->make_shared_buffer(img.size() * 2);
    if (in.empty()) { return; }
    { auto* d = static_cast<_Float16*>(in.contents());
      for (std::size_t i = 0; i < img.size(); ++i) { d[i] = (_Float16)img[i]; } }
    SharedBuffer z = vae->encode(in, H, W);
    if (!z.empty() && z.byte_size() >= lat.size() * 2) {
      const auto* p = static_cast<const _Float16*>(z.contents());
      z_out->resize(lat.size());
      for (std::size_t i = 0; i < lat.size(); ++i) {
        (*z_out)[i] = (float)p[i];
      }
    }

    SharedBuffer zin = mc->make_shared_buffer(lat.size() * 2);
    if (zin.empty()) { return; }
    { auto* d = static_cast<_Float16*>(zin.contents());
      for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; } }
    std::string derr;
    SharedBuffer rgb = vae->decode(zin, lh, lw, &derr);
    const std::size_t n = img.size();
    if (!rgb.empty() && rgb.byte_size() >= n * 2) {
      const auto* p = static_cast<const _Float16*>(rgb.contents());
      rgb_out->resize(n);
      for (std::size_t i = 0; i < n; ++i) { (*rgb_out)[i] = (float)p[i]; }
    }
  };

  std::vector<float> z_mma, rgb_mma, z_ref, rgb_ref;
  run(/*scalar=*/false, &z_mma, &rgb_mma);
  run(/*scalar=*/true, &z_ref, &rgb_ref);
  BOOGU_REQUIRE(!z_mma.empty() && z_mma.size() == z_ref.size());
  BOOGU_REQUIRE(!rgb_mma.empty() && rgb_mma.size() == rgb_ref.size());

  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d; den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : (num == 0.0 ? 0.0 : 1.0);
  };
  const double re = rel(z_mma, z_ref), rd = rel(rgb_mma, rgb_ref);
  std::printf("[boogu_smoke] VAE flash-attn vs simdgroup flash: encode rel-L2 "
              "%.6g, decode rel-L2 %.6g (mid block %dx%d = %d tokens)\n",
              re, rd, lh, lw, lh * lw);
  EXPECT_TRUE(std::isfinite(re) && re < 2e-2);
  EXPECT_TRUE(std::isfinite(rd) && rd < 2e-2);
  }
}

// ---- boogu_golden: numerical rel-L2 vs the reference implementation --------
// VPIPE_BOOGU_GOLDEN = a dir written by dump_boogu_golden.py, which runs the
// canonical BooguImageTransformer2DModel on CPU in bf16 (the precision vpipe
// computes in) over deterministic inputs. Comparing in LATENT space also
// exercises the stage's patch pack/unpack convention -- "(h w) (p1 p2 c)",
// channel fastest inside a token -- which cancels out of every self-consistency
// check and so is only observable against a reference.
TEST(boogu_golden, dit_velocity_rel_l2)
{
  const std::string root = model_root_();
  const char* gd = std::getenv("VPIPE_BOOGU_GOLDEN");
  if (root.empty() || gd == nullptr || *gd == '\0') { return; }
  const std::string g(gd);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto read_f32 = [](const std::string& p, std::size_t n) {
    std::vector<float> v(n);
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (f == nullptr) { return std::vector<float>(); }
    const std::size_t got = std::fread(v.data(), 4, n, f);
    std::fclose(f);
    if (got != n) { return std::vector<float>(); }
    return v;
  };
  int TS = 0, lh = 0, lw = 0, Z = 0, P = 0, idim = 0, n_refs = 0;
  float sigma = 0.0f;
  bool has_ref = false;
  std::vector<std::pair<int, int>> ref_hw;   // per-reference LATENT h, w
  {
    std::ifstream in(g + "/meta.json");
    BOOGU_REQUIRE((bool)in);
    FlexData m = FlexData::from_json(in);
    BOOGU_REQUIRE(m.is_object());
    auto o = m.as_object();
    TS = (int)o.at("ts").as_int(0);
    lh = (int)o.at("lh").as_int(0);
    lw = (int)o.at("lw").as_int(0);
    Z  = (int)o.at("channels").as_int(0);
    P  = (int)o.at("patch").as_int(0);
    idim = (int)o.at("instruct_dim").as_int(0);
    sigma = (float)o.at("sigma").as_real(0.0);
    has_ref = o.at("has_ref").as_bool(false);
    if (o.contains("n_refs")) { n_refs = (int)o.at("n_refs").as_int(0); }
    else { n_refs = has_ref ? 1 : 0; }
    if (o.contains("ref_shapes")) {
      FlexData rs = o.at("ref_shapes");
      if (rs.is_array()) {
        auto av = rs.as_array();
        for (std::size_t i = 0; i < av.size(); ++i) {
          FlexData pr = av[i];
          if (!pr.is_array()) { continue; }
          auto pv = pr.as_array();
          if (pv.size() >= 2) {
            ref_hw.push_back({(int)pv[0].as_int(0), (int)pv[1].as_int(0)});
          }
        }
      }
    }
  }
  BOOGU_REQUIRE(TS > 0 && lh > 0 && lw > 0 && Z > 0 && P > 0);
  const int gh = lh / P, gw = lw / P, img_seq = gh * gw, IC = P * P * Z;

  auto dit = MetalBooguTransformer::load(root + "/transformer", mc,
                                         MetalBooguTransformer::Config{});
  BOOGU_REQUIRE(dit != nullptr);

  const std::vector<float> instr =
      read_f32(g + "/instruct.f32", (std::size_t)TS * idim);
  const std::vector<float> lat =
      read_f32(g + "/latent.f32", (std::size_t)Z * lh * lw);
  const std::vector<float> want =
      read_f32(g + "/velocity.f32", (std::size_t)Z * lh * lw);
  BOOGU_REQUIRE(!instr.empty() && !lat.empty() && !want.empty());

  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * idim * 2);
  { auto* p = static_cast<std::uint16_t*>(ctx.contents());
    for (std::size_t i = 0; i < instr.size(); ++i) {
      p[i] = f32_to_bf16_(instr[i]);
    } }
  // Pack [Z, h, w] -> [seq, IC] exactly as the stage does (any latent size, so
  // the same helper serves the target and differently-sized references).
  auto pack = [&](const std::vector<float>& chw, int h, int w) {
    const int tgh = h / P, tgw = w / P, tseq = tgh * tgw;
    SharedBuffer b = mc->make_shared_buffer((std::size_t)tseq * IC * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (int c = 0; c < Z; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const int a = y / P, ph = y % P, bc = x / P, pw = x % P;
          const std::size_t t = (std::size_t)a * tgw + bc;
          d[t * IC + ((std::size_t)ph * P + pw) * Z + c] =
              f32_to_bf16_(chw[((std::size_t)c * h + y) * w + x]);
        }
      }
    }
    return b;
  };
  SharedBuffer latb = pack(lat, lh, lw);
  std::vector<MetalBooguTransformer::RefImage> refs;
  for (int i = 0; i < n_refs; ++i) {
    const int rh = i < (int)ref_hw.size() ? ref_hw[(std::size_t)i].first : lh;
    const int rw = i < (int)ref_hw.size() ? ref_hw[(std::size_t)i].second : lw;
    // Older goldens wrote a single "ref.f32"; newer ones "ref0.f32", ...
    std::vector<float> rf =
        read_f32(g + "/ref" + std::to_string(i) + ".f32",
                 (std::size_t)Z * rh * rw);
    if (rf.empty() && i == 0) {
      rf = read_f32(g + "/ref.f32", (std::size_t)Z * rh * rw);
    }
    BOOGU_REQUIRE(!rf.empty());
    MetalBooguTransformer::RefImage r;
    r.latents = pack(rf, rh, rw);
    r.seq = (rh / P) * (rw / P); r.grid_h = rh; r.grid_w = rw;
    refs.push_back(std::move(r));
  }

  SharedBuffer vel =
      dit->forward_dit(ctx, TS, latb, img_seq, lh, lw, sigma, refs);
  BOOGU_REQUIRE(!vel.empty());
  // Unpack back to [Z, lh, lw] and compare against the reference.
  const auto* vp = static_cast<const std::uint16_t*>(vel.contents());
  double num = 0.0, den = 0.0;
  for (int c = 0; c < Z; ++c) {
    for (int y = 0; y < lh; ++y) {
      for (int x = 0; x < lw; ++x) {
        const int a = y / P, ph = y % P, bc = x / P, pw = x % P;
        const std::size_t t = (std::size_t)a * gw + bc;
        const double got =
            bf16_to_f32_(vp[t * IC + ((std::size_t)ph * P + pw) * Z + c]);
        const double ref = want[((std::size_t)c * lh + y) * lw + x];
        num += (got - ref) * (got - ref);
        den += ref * ref;
      }
    }
  }
  const double rel = std::sqrt(num / den);
  int ref_tok = 0;
  for (const auto& r : refs) { ref_tok += r.seq; }
  std::printf("[boogu_golden] DiT velocity rel-L2 = %.5f (seq %d img + %d txt, "
              "%d reference(s) = %d tokens)\n", rel, img_seq, TS,
              (int)refs.size(), ref_tok);
  // bf16 vs bf16 through 46 blocks: the sibling DiTs' golden bounds are 0.10
  // (FLUX.2, 25 blocks) and 0.20 (QIE, 60 blocks); 0.10 sits between them.
  EXPECT_TRUE(std::isfinite(rel) && rel < 0.10);
}

// ---- conditioning golden -------------------------------------------------
// tools/dump_boogu_cond_golden.py runs the reference
// _get_instruction_feature_embeds: the adaptive system prompt, the stock
// Qwen3-VL chat template with NO generation prompt, and the mllm's LAST hidden
// state (post final norm) over the WHOLE templated sequence -- Boogu drops
// nothing. It also saves the PREPROCESSED VLM image, so this test can be run
// two ways:
//   VPIPE_BOOGU_COND_IMAGE unset -> feed the golden's own vlm_input.png, which
//     isolates the template + vision tower + deepstack + mROPE + splice + final
//     norm from any resize difference.
//   VPIPE_BOOGU_COND_IMAGE=<original> -> additionally exercise vpipe's own
//     long-edge cap (a box filter where the reference uses LANCZOS).
namespace {

// Test-only prompt source: emits one string then ends.
class BooguCondSrc : public vpipe::TypedStage<BooguCondSrc> {
public:
  static constexpr const char* kTypeName = "ut-boogu-cond-src";
  BooguCondSrc(const vpipe::SessionContextIntf* s, std::string id,
               std::vector<vpipe::InEdge> ip, vpipe::FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::string prompt;
  bool done = false;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      co_await ctx.write(0, std::make_unique<vpipe::FlexDataPayload>(
                                vpipe::FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};

class BooguCondSink : public vpipe::TypedStage<BooguCondSink> {
public:
  static constexpr const char* kTypeName = "ut-boogu-cond-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<vpipe::BeatPayloadIntf>> captured;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

}  // namespace

TEST(boogu_golden, conditioning_rel_l2)
{
  const std::string root = model_root_();
  const char* gd = std::getenv("VPIPE_BOOGU_COND_GOLDEN");
  if (root.empty() || gd == nullptr || *gd == '\0') { return; }
  const std::string g(gd);
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  int n_g = 0, dim = 0, n_images = 1;
  std::string instruction, vlm_png;
  {
    std::ifstream in(g + "/meta.json");
    BOOGU_REQUIRE((bool)in);
    FlexData m = FlexData::from_json(in);
    BOOGU_REQUIRE(m.is_object());
    auto o = m.as_object();
    n_g = (int)o.at("n").as_int(0);
    dim = (int)o.at("dim").as_int(0);
    instruction = std::string(o.at("instruction").as_string(""));
    if (o.contains("vlm_input") && o.at("vlm_input").is_string()) {
      vlm_png = g + "/" + std::string(o.at("vlm_input").as_string(""));
    }
    if (o.contains("n_images")) {
      n_images = (int)o.at("n_images").as_int(1);
    }
  }
  BOOGU_REQUIRE(n_g > 0 && dim > 0 && !instruction.empty());
  // Default to the golden's OWN preprocessed pixels; override to test the
  // full path including vpipe's resize.
  if (const char* im = std::getenv("VPIPE_BOOGU_COND_IMAGE")) {
    if (*im != '\0') { vlm_png = im; }
  }
  const bool grounded = !vlm_png.empty();   // text-only golden => no ref image
  // A multi-image golden saves vlm_input.png, vlm_input1.png, ... -- wire one
  // ref_image iport per picture so the VLM sees them all.
  std::vector<std::string> ref_pngs;
  if (grounded) {
    ref_pngs.push_back(vlm_png);
    for (int i = 1; i < n_images; ++i) {
      ref_pngs.push_back(g + "/vlm_input" + std::to_string(i) + ".png");
    }
  }

  std::vector<float> want((std::size_t)n_g * dim);
  {
    std::FILE* f = std::fopen((g + "/cond.f32").c_str(), "rb");
    BOOGU_REQUIRE(f != nullptr);
    const std::size_t got = std::fread(want.data(), 4, want.size(), f);
    std::fclose(f);
    BOOGU_REQUIRE(got == want.size());
  }
  std::vector<std::int32_t> ids((std::size_t)n_g);
  {
    std::FILE* f = std::fopen((g + "/ids.i32").c_str(), "rb");
    if (f != nullptr) {
      const std::size_t got = std::fread(ids.data(), 4, ids.size(), f);
      std::fclose(f);
      if (got != ids.size()) { ids.clear(); }
    } else {
      ids.clear();
    }
  }

  // Run the conditioner: prompt on iport0, the reference image on iport3.
  auto pl = std::make_unique<Pipeline>("bc", &sess);
  auto srcu = std::make_unique<BooguCondSrc>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  srcu->prompt = instruction;
  auto* src = static_cast<BooguCondSrc*>(pl->insert_stage(std::move(srcu)));
  std::vector<InEdge> ce{{src, 0}};
  if (grounded) {
    ce.push_back({nullptr, 0});    // iport1 negative -- unconnected
    ce.push_back({nullptr, 0});    // iport2 model    -- unconnected
    for (int i = 0; i < (int)ref_pngs.size(); ++i) {
      FlexData ic = FlexData::make_object();
      ic.as_object().insert("url", FlexData::make_string(ref_pngs[(std::size_t)i]));
      auto iu = std::make_unique<LoadImageStage>(
          &sess, "img" + std::to_string(i), std::vector<InEdge>{},
          std::move(ic));
      Stage* imgstage = pl->insert_stage(std::move(iu));
      ce.push_back({imgstage, 0});   // iport3 / iport4 ref_image[2]
    }
  }
  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(&sess, "cond",
                                                        std::move(ce),
                                                        std::move(cc));
  auto* cond =
      static_cast<DiffusionConditionerStage*>(pl->insert_stage(std::move(cu)));
  auto sinku = std::make_unique<BooguCondSink>(
      &sess, "sink", std::vector<InEdge>{{cond, 0}}, FlexData::make_object());
  auto* sink = static_cast<BooguCondSink*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  BOOGU_REQUIRE(rt.launch());
  rt.wait_idle();
  rt.stop();
  BOOGU_REQUIRE(!sink->captured.empty());
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  BOOGU_REQUIRE(tb != nullptr && tb->shape.size() == 2);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::Bf16);
  const int n_v = (int)tb->shape[0];
  std::printf("[boogu_golden] conditioning rows: vpipe %d, golden %d (dim %d)\n",
              n_v, n_g, (int)tb->shape[1]);
  // Token count must agree EXACTLY -- Boogu drops no prefix, so a template or
  // vision-token-count difference shows up right here.
  BOOGU_REQUIRE(n_v == n_g);
  BOOGU_REQUIRE((int)tb->shape[1] == dim);
  const auto bytes = tb->materialize_contiguous();
  const auto* bp = reinterpret_cast<const std::uint16_t*>(bytes.data());
  std::vector<float> got((std::size_t)n_g * dim);
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = bf16_to_f32_(bp[i]); }

  // Overall + per-row-block (image rows vs text rows): a grounding bug lives in
  // the image rows and a template bug in the text rows, so reporting them apart
  // is what makes a mismatch diagnosable.
  auto stats = [&](const std::vector<int>& rows, const char* tag) {
    if (rows.empty()) { return; }
    double num = 0.0, den = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (int r : rows) {
      for (int k = 0; k < dim; ++k) {
        const double a = got[(std::size_t)r * dim + k];
        const double b = want[(std::size_t)r * dim + k];
        num += (a - b) * (a - b); den += b * b;
        dot += a * b; na += a * a; nb += b * b;
      }
    }
    std::printf("[boogu_golden]   %-11s rows %3zu  rel-L2 %.5f  cos %.6f\n",
                tag, rows.size(), std::sqrt(num / den),
                dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
  };
  std::vector<int> all, imgrows, txtrows;
  const std::int32_t kImagePad = 151655;
  for (int r = 0; r < n_g; ++r) {
    all.push_back(r);
    if (!ids.empty() && ids[(std::size_t)r] == kImagePad) { imgrows.push_back(r); }
    else { txtrows.push_back(r); }
  }
  double num = 0.0, den = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double a = got[i], b = want[i];
    num += (a - b) * (a - b); den += b * b;
    dot += a * b; na += a * a; nb += b * b;
  }
  const double rel = std::sqrt(num / den);
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  std::printf("[boogu_golden] conditioning rel-L2 %.5f  cos %.6f\n", rel, cos);
  stats(imgrows, "image");
  stats(txtrows, "text");
  // Bounds calibrated by EXPERIMENT, not by loosening until green. Feeding the
  // REFERENCE LM vpipe's own tower output (tools/dump_boogu_cond_golden.py
  // --inject-vision) reproduces the image-row error to 4 significant figures
  // (0.17825 vs vpipe's 0.17837), so the whole image-row gap is the LM
  // amplifying the tower's 0.035 bf16 difference ~5x -- not a vpipe LM or
  // splice/deepstack/mROPE error. A random 3.5% tower perturbation
  // (--perturb-vision) only reaches 0.091, i.e. a SYSTEMATIC rounding pattern
  // amplifies about twice as much as noise does.
  //
  // So the sharp assertions are the ones the tower does NOT dominate: the token
  // count (template + vision-token count) and the TEXT rows (template,
  // tokenizer, LM backbone, mROPE, final norm). Text-only goldens land ~0.02
  // overall, which is the same floor.
  EXPECT_TRUE(std::isfinite(rel) && rel < (grounded ? 0.25 : 0.05));
  EXPECT_TRUE(cos > 0.99);
  if (!txtrows.empty()) {
    double tn = 0.0, td = 0.0;
    for (int r : txtrows) {
      for (int k = 0; k < dim; ++k) {
        const double a = got[(std::size_t)r * dim + k];
        const double b = want[(std::size_t)r * dim + k];
        tn += (a - b) * (a - b); td += b * b;
      }
    }
    EXPECT_TRUE(std::sqrt(tn / td) < 0.05);
  }
}

// Partition a conditioning mismatch: compare vpipe's Qwen3-VL tower output (the
// merged image embeddings that get spliced over the <|image_pad|> rows, plus the
// deepstack features added after LM layers 0..k-1) against the reference tower
// on the SAME preprocessed pixels. If this matches but conditioning_rel_l2 does
// not, the fault is in the splice / deepstack injection / LM; if it does not
// match, it is the tower (or its preprocessing).
TEST(boogu_golden, vision_tower_rel_l2)
{
  const std::string root = model_root_();
  const char* gd = std::getenv("VPIPE_BOOGU_COND_GOLDEN");
  if (root.empty() || gd == nullptr || *gd == '\0') { return; }
  const std::string g(gd);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  int n_img = 0, n_ds = 0, dim = 0;
  std::string vlm_png;
  {
    std::ifstream in(g + "/meta.json");
    BOOGU_REQUIRE((bool)in);
    FlexData m = FlexData::from_json(in);
    auto o = m.as_object();
    n_img = (int)o.at("n_img").as_int(0);
    n_ds  = (int)o.at("n_deepstack").as_int(0);
    dim   = (int)o.at("dim").as_int(0);
    if (o.contains("vlm_input") && o.at("vlm_input").is_string()) {
      vlm_png = g + "/" + std::string(o.at("vlm_input").as_string(""));
    }
  }
  if (n_img <= 0 || vlm_png.empty()) { return; }   // text-only golden

  // Load the preprocessed PNG the reference actually fed the tower.
  std::vector<std::uint8_t> rgb;
  int H = 0, W = 0;
  {
    Session s2;
    FlexData ic = FlexData::make_object();
    ic.as_object().insert("url", FlexData::make_string(vlm_png));
    auto pl = std::make_unique<Pipeline>("li", &s2);
    auto iu = std::make_unique<LoadImageStage>(&s2, "img",
                                              std::vector<InEdge>{},
                                              std::move(ic));
    Stage* im = pl->insert_stage(std::move(iu));
    auto sinku = std::make_unique<BooguCondSink>(
        &s2, "sink", std::vector<InEdge>{{im, 0}}, FlexData::make_object());
    auto* sink = static_cast<BooguCondSink*>(pl->insert_stage(std::move(sinku)));
    PipelineRuntime rt(pl.get(), &s2);
    BOOGU_REQUIRE(rt.launch());
    rt.wait_idle();
    rt.stop();
    BOOGU_REQUIRE(!sink->captured.empty());
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
    BOOGU_REQUIRE(tb != nullptr && tb->shape.size() == 3);
    H = (int)tb->shape[1]; W = (int)tb->shape[2];
    const auto bytes = tb->materialize_contiguous();
    rgb.assign(bytes.begin(), bytes.end());
  }
  BOOGU_REQUIRE(H > 0 && W > 0 && rgb.size() >= (std::size_t)3 * H * W);

  // The tower exactly as the conditioner builds it for boogu-image.
  const std::string enc_dir = root + "/mllm";
  ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(enc_dir);
  BOOGU_REQUIRE(mcfg.has_value());
  auto vcfg = MetalQwenVisionEncoder::config_from(*mcfg);
  vcfg.weight_prefix = "model.visual.";
  vcfg.use_bf16 = true;
  vcfg.min_pixels = 65536;
  vcfg.max_pixels = 16777216;
  for (int i = 0; i < 3; ++i) { vcfg.image_mean[i] = 0.5f; vcfg.image_std[i] = 0.5f; }
  auto tower = MetalQwenVisionEncoder::load(enc_dir, mc, vcfg);
  BOOGU_REQUIRE(tower != nullptr);
  tower->set_session(&sess);
  auto r = tower->encode(rgb.data(), H, W);
  BOOGU_REQUIRE(!r.embeddings.empty() && r.n_tokens > 0);
  std::printf("[boogu_golden] tower -> %d tokens (patch grid %dx%d), %zu "
              "deepstack; golden %d tokens, %d deepstack\n",
              r.n_tokens, r.grid_h, r.grid_w, r.deepstack.size(), n_img, n_ds);
  BOOGU_REQUIRE(r.n_tokens == n_img);

  auto cmp = [&](const std::string& file, const SharedBuffer& buf,
                 const char* tag) {
    std::vector<float> want((std::size_t)n_img * dim);
    std::FILE* f = std::fopen((g + "/" + file).c_str(), "rb");
    if (f == nullptr) { std::printf("  (no %s)\n", file.c_str()); return; }
    const std::size_t got = std::fread(want.data(), 4, want.size(), f);
    std::fclose(f);
    if (got != want.size()) { std::printf("  (short %s)\n", file.c_str()); return; }
    const bool bf = tower->is_bf16();
    double num = 0.0, den = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      double a;
      if (bf) {
        a = bf16_to_f32_(static_cast<const std::uint16_t*>(buf.contents())[i]);
      } else {
        a = (double)static_cast<const _Float16*>(buf.contents())[i];
      }
      const double b = want[i];
      num += (a - b) * (a - b); den += b * b;
      dot += a * b; na += a * a; nb += b * b;
    }
    std::printf("[boogu_golden]   %-11s rel-L2 %.5f  cos %.6f\n", tag,
                std::sqrt(num / den),
                dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
  };
  // VPIPE_BOOGU_COND_DUMP=<dir>: write vpipe's OWN tower output as f32 so the
  // reference LM can be re-run on it (tools/dump_boogu_cond_golden.py
  // --inject-vision), which partitions the conditioning gap into "tower error
  // propagated by the LM" vs "vpipe's LM".
  if (const char* dd = std::getenv("VPIPE_BOOGU_COND_DUMP")) {
    if (*dd != '\0') {
      const bool bf = tower->is_bf16();
      auto wr = [&](const std::string& nm, const SharedBuffer& b) {
        std::vector<float> v((std::size_t)n_img * dim);
        for (std::size_t i = 0; i < v.size(); ++i) {
          v[i] = bf ? bf16_to_f32_(
                          static_cast<const std::uint16_t*>(b.contents())[i])
                    : (float)static_cast<const _Float16*>(b.contents())[i];
        }
        std::FILE* f = std::fopen((std::string(dd) + "/" + nm).c_str(), "wb");
        if (f != nullptr) {
          std::fwrite(v.data(), 4, v.size(), f);
          std::fclose(f);
        }
      };
      wr("vpipe_vision.f32", r.embeddings);
      for (int i = 0; i < (int)r.deepstack.size(); ++i) {
        wr("vpipe_deepstack" + std::to_string(i) + ".f32",
           r.deepstack[(std::size_t)i]);
      }
      std::printf("[boogu_golden] dumped vpipe tower output to %s\n", dd);
    }
  }
  cmp("vision.f32", r.embeddings, "merged");
  for (int i = 0; i < (int)r.deepstack.size() && i < n_ds; ++i) {
    cmp("deepstack" + std::to_string(i) + ".f32", r.deepstack[(std::size_t)i],
        ("deepstack" + std::to_string(i)).c_str());
  }
}

// Shape-level GEMM bench for the Boogu DiT's actual projection / FF shapes,
// so a tile or fusion decision can be settled in seconds instead of a
// two-minute pipeline run. Opt-in: VPIPE_BOOGU_GEMM_BENCH (no model needed --
// the operands are synthetic). VPIPE_BOOGU_BENCH_SEQ overrides the sequence
// length (default 2271 = the 512x512 edit graph: 1024 target + 1024 reference
// + 223 instruction tokens).
//
// What it answers:
//   * bm32 vs bm64 vs bm64/bn64 per shape (the VPIPE_BOOGU_GEMM_TILE knob),
//   * whether fusing q|k|v into ONE [H + 2*KD, H] GEMM beats three,
//   * whether the fused-SwiGLU FF beats gate + up + elementwise.
TEST(boogu_perf, gemm_shapes)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_bf16");
  ComputeLibrary elt = mc->load_library("llm_elementwise_bf16");
  ComputeFunction f32t = lib.function("dense_gemm_t_f16");
  ComputeFunction f64t = lib.function("dense_gemm_t_bm64_f16");
  ComputeFunction f6464 = lib.function("dense_gemm_t_bm64bn64_f16");
  ComputeFunction fswi = lib.function("dense_gemm_swiglu_bm64_f16");
  ComputeFunction fswi_elt = elt.function("swiglu_f16");
  BOOGU_REQUIRE(f32t.valid() && f64t.valid() && f6464.valid());
  BOOGU_REQUIRE(fswi.valid() && fswi_elt.valid());

  MetalBooguTransformer::Config c;          // the shipped Boogu-Image config
  // ff_inner is read off the checkpoint at load; this bench has no checkpoint,
  // so use Boogu-Image-0.1's value (verified against the 942-tensor manifest).
  const int H = c.hidden, KD = c.n_kv_heads * c.head_dim, FFI = 13568;
  int M = 2271;
  if (const char* s = std::getenv("VPIPE_BOOGU_BENCH_SEQ")) {
    M = std::atoi(s);
    if (M < 32) { M = 32; }
  }
  const int KMAX = FFI > H ? FFI : H;
  const int NMAX = 2 * FFI;
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * KMAX * 2);
  SharedBuffer wb = mc->make_shared_buffer((std::size_t)NMAX * KMAX * 2);
  SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * NMAX * 2);
  SharedBuffer y2 = mc->make_shared_buffer((std::size_t)M * FFI * 2);
  BOOGU_REQUIRE(!xb.empty() && !wb.empty() && !yb.empty() && !y2.empty());
  fill_normal_bf16_(xb, (std::size_t)M * KMAX, 1);
  fill_normal_bf16_(wb, (std::size_t)NMAX * KMAX, 2);

  const int ITERS = 8;
  // One timed run of `n` dispatches, best of 3 (rep 0 warms up).
  auto time_ms = [&](const std::function<void(ComputeEncoder&)>& body) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) { body(enc); }
      }
      const auto t0 = std::chrono::steady_clock::now();
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (rep > 0 && ms < best) { best = ms; }
    }
    return best / ITERS;
  };
  auto gemm = [&](ComputeEncoder& enc, const ComputeFunction& fn, int bm,
                  int bn, int N, int K, std::size_t we) {
    enc.set_function(fn);
    enc.set_buffer(0, xb); enc.set_buffer(1, wb, we * 2);
    enc.set_buffer(2, wb, we * 2); enc.set_buffer(3, yb);
    enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
    enc.set_constant(7, 0);
    const unsigned tgz = 2;
    enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                  (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
  };
  auto report = [&](const char* tag, int N, int K, double ms) {
    std::printf("[boogu_perf] %-18s M=%4d N=%5d K=%5d | %7.2f ms | %5.2f "
                "TFLOP/s\n", tag, M, N, K, ms,
                2.0 * M * N * K / (ms * 1e-3) / 1e12);
  };
  // Per-shape tile sweep.
  struct Sh { const char* tag; int N; int K; };
  const Sh shapes[] = {
      {"q/o", H, H}, {"k or v", KD, H}, {"qkv-fused", H + 2 * KD, H},
      {"ff-down", H, FFI},
  };
  for (const auto& sh : shapes) {
    const double m32 = time_ms([&](ComputeEncoder& e) {
      gemm(e, f32t, 32, 32, sh.N, sh.K, 0);
    });
    const double m64 = time_ms([&](ComputeEncoder& e) {
      gemm(e, f64t, 64, 32, sh.N, sh.K, 0);
    });
    const double m6464 = time_ms([&](ComputeEncoder& e) {
      gemm(e, f6464, 64, 64, sh.N, sh.K, 0);
    });
    double best = m32 < m64 ? m32 : m64;
    if (m6464 < best) { best = m6464; }
    report((std::string(sh.tag) + " bm32").c_str(), sh.N, sh.K, m32);
    report((std::string(sh.tag) + " bm64").c_str(), sh.N, sh.K, m64);
    report((std::string(sh.tag) + " bm64bn64").c_str(), sh.N, sh.K, m6464);
    std::printf("[boogu_perf] %-18s best %.2f ms\n", sh.tag, best);
  }
  // q/k/v: three GEMMs vs one fused [H + 2*KD, H].
  {
    auto three = [&](const ComputeFunction& fn, int bm, int bn) {
      return time_ms([&](ComputeEncoder& e) {
        gemm(e, fn, bm, bn, H, H, 0);
        gemm(e, fn, bm, bn, KD, H, 0);
        gemm(e, fn, bm, bn, KD, H, 0);
      });
    };
    auto one = [&](const ComputeFunction& fn, int bm, int bn) {
      return time_ms([&](ComputeEncoder& e) {
        gemm(e, fn, bm, bn, H + 2 * KD, H, 0);
      });
    };
    const double s32 = three(f32t, 32, 32), o32 = one(f32t, 32, 32);
    const double s64 = three(f64t, 64, 32), o64 = one(f64t, 64, 32);
    std::printf("[boogu_perf] qkv 3-gemm bm32 %.2f ms -> fused %.2f ms "
                "(%.2fx) | bm64 %.2f -> %.2f (%.2fx)\n",
                s32, o32, s32 / o32, s64, o64, s64 / o64);
  }
  // FF: fused SwiGLU vs gate + up + elementwise.
  {
    const double fused = time_ms([&](ComputeEncoder& e) {
      e.set_function(fswi);
      e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, yb);
      e.set_constant(3, H); e.set_constant(4, 2 * FFI);
      e.set_constant(5, M); e.set_constant(6, 0); e.set_constant(7, 0);
      e.dispatch({(unsigned)(((2 * FFI + 31) / 32) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    });
    const double unfused = time_ms([&](ComputeEncoder& e) {
      gemm(e, f64t, 64, 32, FFI, H, 0);
      gemm(e, f64t, 64, 32, FFI, H, (std::size_t)FFI * H);
      e.set_function(fswi_elt);
      e.set_buffer(0, yb); e.set_buffer(1, y2); e.set_buffer(2, y2);
      e.set_constant(3, M * FFI);
      e.dispatch({(unsigned)(M * FFI), 1, 1}, {256, 1, 1});
    });
    std::printf("[boogu_perf] ff gate|up: fused %.2f ms vs unfused %.2f ms "
                "(%.2fx) | fused %.2f TFLOP/s\n", fused, unfused,
                unfused / fused,
                2.0 * M * 2.0 * FFI * H / (fused * 1e-3) / 1e12);
  }
  EXPECT_TRUE(true);
}

// The non-GEMM passes of a Boogu DiT block, at the shipped shapes. The pipeline
// profile shows ~15 ms per single-stream block that is not GEMM and not
// attention; this attributes it per kernel so the fusion work targets the right
// one. Opt-in: VPIPE_BOOGU_GEMM_BENCH.
TEST(boogu_perf, elementwise_shapes)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  ComputeLibrary elt = mc->load_library("llm_elementwise_bf16");
  ComputeLibrary rmsl = mc->load_library("rms_norm_bf16");
  ComputeLibrary ropel = mc->load_library("rope_bf16");
  ComputeFunction f_rms = rmsl.function("rms_norm_fast_f16");
  ComputeFunction f_adaln = elt.function("adaln_modulate_f16");
  ComputeFunction f_gated = elt.function("gated_residual_tanh_f16");
  ComputeFunction f_trrope = ropel.function("transpose_rope_pair_ftab_pad_f16");
  ComputeFunction f_trpad = elt.function("transpose_abd_pad_f16");
  ComputeFunction f_trunpad = elt.function("transpose_abd_unpad_f16");
  BOOGU_REQUIRE(f_rms.valid() && f_adaln.valid() && f_gated.valid());
  BOOGU_REQUIRE(f_trrope.valid() && f_trpad.valid() && f_trunpad.valid());

  MetalBooguTransformer::Config c;
  const int H = c.hidden, HD = c.head_dim, HED = c.n_heads;
  const int KVH = c.n_kv_heads, DP = 128;
  int M = 2271;
  if (const char* s = std::getenv("VPIPE_BOOGU_BENCH_SEQ")) {
    M = std::atoi(s);
    if (M < 32) { M = 32; }
  }
  SharedBuffer a = mc->make_shared_buffer((std::size_t)M * H * 2);
  SharedBuffer b = mc->make_shared_buffer((std::size_t)M * H * 2);
  SharedBuffer w = mc->make_shared_buffer((std::size_t)H * 2);
  SharedBuffer pad = mc->make_shared_buffer((std::size_t)HED * M * DP * 2);
  SharedBuffer tab = mc->make_shared_buffer((std::size_t)M * HD * 4);
  BOOGU_REQUIRE(!a.empty() && !b.empty() && !pad.empty() && !tab.empty());
  fill_normal_bf16_(a, (std::size_t)M * H, 3);
  fill_normal_bf16_(b, (std::size_t)M * H, 4);
  fill_normal_bf16_(w, (std::size_t)H, 5);
  std::memset(tab.contents(), 0, tab.byte_size());

  const int ITERS = 20;
  auto time_ms = [&](const std::function<void(ComputeEncoder&)>& body) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) { body(enc); }
      }
      const auto t0 = std::chrono::steady_clock::now();
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (rep > 0 && ms < best) { best = ms; }
    }
    return best / ITERS;
  };
  auto report = [&](const char* tag, double ms, double bytes) {
    std::printf("[boogu_perf] %-22s %6.3f ms | %6.1f GB/s\n", tag, ms,
                bytes / (ms * 1e-3) / 1e9);
  };
  const double rowb = (double)M * H * 2;
  report("rms [M,H]", time_ms([&](ComputeEncoder& e) {
    e.set_function(f_rms);
    e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, b);
    e.set_constant(3, H); e.set_constant(4, 1e-5f);
    e.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
  }), 2 * rowb);
  report("adaln modulate", time_ms([&](ComputeEncoder& e) {
    e.set_function(f_adaln);
    e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, w);
    e.set_buffer(3, b);
    e.set_constant(4, H); e.set_constant(5, M * H);
    e.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
  }), 2 * rowb);
  report("gated_residual_tanh", time_ms([&](ComputeEncoder& e) {
    e.set_function(f_gated);
    e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, b);
    e.set_constant(3, H); e.set_constant(4, M * H);
    e.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
  }), 3 * rowb);
  ComputeFunction f_adaln4 = elt.function("adaln_modulate_v4_f16");
  ComputeFunction f_gated4 = elt.function("gated_residual_tanh_v4_f16");
  if (f_adaln4.valid() && f_gated4.valid()) {
    report("adaln v4", time_ms([&](ComputeEncoder& e) {
      e.set_function(f_adaln4);
      e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, w);
      e.set_buffer(3, b);
      e.set_constant(4, H / 4); e.set_constant(5, M);
      e.dispatch({(unsigned)(H / 4), (unsigned)M, 1}, {256, 1, 1});
    }), 2 * rowb);
    report("gated_tanh v4", time_ms([&](ComputeEncoder& e) {
      e.set_function(f_gated4);
      e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, b);
      e.set_constant(3, H / 4); e.set_constant(4, M);
      e.dispatch({(unsigned)(H / 4), (unsigned)M, 1}, {256, 1, 1});
    }), 3 * rowb);
  } else {
    std::printf("[boogu_perf] v4 elementwise twins MISSING\n");
  }
  report("rms per-head q", time_ms([&](ComputeEncoder& e) {
    e.set_function(f_rms);
    e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, b);
    e.set_constant(3, HD); e.set_constant(4, 1e-5f);
    e.dispatch({256, (unsigned)(M * HED), 1}, {256, 1, 1});
  }), 2 * rowb);
  for (unsigned tg : {32u, 64u, 128u}) {
    char tag[48];
    std::snprintf(tag, sizeof(tag), "rms per-head q tg%u", tg);
    report(tag, time_ms([&](ComputeEncoder& e) {
      e.set_function(f_rms);
      e.set_buffer(0, a); e.set_buffer(1, w); e.set_buffer(2, b);
      e.set_constant(3, HD); e.set_constant(4, 1e-5f);
      e.dispatch({tg, (unsigned)(M * HED), 1}, {tg, 1, 1});
    }), 2 * rowb);
  }
  report("tr_rope_pad q", time_ms([&](ComputeEncoder& e) {
    e.set_function(f_trrope);
    e.set_buffer(0, a); e.set_buffer(1, pad);
    e.set_buffer(2, tab); e.set_buffer(3, tab);
    e.set_constant(4, HED); e.set_constant(5, M); e.set_constant(6, HD);
    e.set_constant(7, DP);
    e.dispatch({(unsigned)(DP / 2), (unsigned)M, (unsigned)HED},
               {(unsigned)(DP / 2), 1, 1});
  }), rowb + (double)HED * M * DP * 2);
  report("tr_unpad out", time_ms([&](ComputeEncoder& e) {
    e.set_function(f_trunpad);
    e.set_buffer(0, pad); e.set_buffer(1, a);
    e.set_constant(2, HED); e.set_constant(3, M); e.set_constant(4, HD);
    e.set_constant(5, DP);
    e.dispatch({(unsigned)HD, (unsigned)M, (unsigned)HED},
               {(unsigned)HD, 1, 1});
  }), rowb + (double)HED * M * DP * 2);
  std::printf("[boogu_perf] (kv variants scale by %d/%d)\n", KVH, HED);

  // The adaLN / gate twins are shared by every DiT here, so measure them at the
  // OTHER families' widths too -- alternating in one process, which is the only
  // way to resolve a few percent on this box (a cross-process end-to-end A/B
  // has a ~4% thermal spread). rows = a representative joint sequence.
  if (f_adaln4.valid() && f_gated4.valid()) {
    struct FamShape { const char* fam; int H; int rows; };
    const FamShape fams[] = {
        {"boogu@512", 3360, 2271}, {"flux2@512", 3072, 1280},
        {"krea2/QIE@512", 3072, 1152}, {"QIE@1024", 3072, 4192},
    };
    for (const auto& fs : fams) {
      SharedBuffer x = mc->make_shared_buffer((std::size_t)fs.rows * fs.H * 2);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)fs.rows * fs.H * 2);
      SharedBuffer w2 = mc->make_shared_buffer((std::size_t)fs.H * 2);
      if (x.empty() || y.empty() || w2.empty()) { continue; }
      fill_normal_bf16_(x, (std::size_t)fs.rows * fs.H, 41);
      fill_normal_bf16_(w2, (std::size_t)fs.H, 42);
      const int tot = fs.rows * fs.H;
      auto pair = [&](const char* nm, const ComputeFunction& fs1,
                      const ComputeFunction& fv4, bool three) {
        const double t1 = time_ms([&](ComputeEncoder& e) {
          e.set_function(fs1);
          e.set_buffer(0, x); e.set_buffer(1, w2);
          if (three) { e.set_buffer(2, y); }
          else { e.set_buffer(2, w2); e.set_buffer(3, y); }
          e.set_constant(three ? 3 : 4, fs.H);
          e.set_constant(three ? 4 : 5, tot);
          e.dispatch({(unsigned)tot, 1, 1}, {256, 1, 1});
        });
        const double t4 = time_ms([&](ComputeEncoder& e) {
          e.set_function(fv4);
          e.set_buffer(0, x); e.set_buffer(1, w2);
          if (three) { e.set_buffer(2, y); }
          else { e.set_buffer(2, w2); e.set_buffer(3, y); }
          e.set_constant(three ? 3 : 4, fs.H / 4);
          e.set_constant(three ? 4 : 5, fs.rows);
          e.dispatch({(unsigned)(fs.H / 4), (unsigned)fs.rows, 1},
                     {256, 1, 1});
        });
        std::printf("[boogu_perf] %-14s %-12s [%4d x %4d] scalar %6.3f ms -> "
                    "v4 %6.3f ms (%.2fx)\n", fs.fam, nm, fs.rows, fs.H, t1, t4,
                    t1 / t4);
      };
      pair("adaln", f_adaln, f_adaln4, false);
      pair("gated_tanh", f_gated, f_gated4, true);
    }
  }

  // forward_dit allocates its whole scratch set per call. At seq 2271 that is
  // ~1.2 GB of wired SharedBuffer created and destroyed every sampler step;
  // this is what that costs, i.e. what pooling the scratch would recover.
  {
    const std::size_t elems[] = {
        (std::size_t)M * H, (std::size_t)M * H, (std::size_t)M * H,
        (std::size_t)M * H, (std::size_t)M * H, (std::size_t)M * H,
        (std::size_t)M * KVH * HD, (std::size_t)M * KVH * HD,
        (std::size_t)HED * M * DP, (std::size_t)KVH * M * DP,
        (std::size_t)KVH * M * DP, (std::size_t)HED * M * DP,
        (std::size_t)M * H, (std::size_t)M * H, (std::size_t)M * 13568};
    double total_mb = 0.0;
    for (std::size_t e : elems) { total_mb += (double)e * 2 / 1e6; }
    double best = 1e30;
    for (int rep = 0; rep < 4; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      {
        std::vector<SharedBuffer> pool;
        pool.reserve(sizeof(elems) / sizeof(elems[0]));
        for (std::size_t e : elems) {
          pool.push_back(mc->make_shared_buffer(e * 2));
        }
        // Touch one byte per buffer: an untouched allocation may not be paged.
        for (auto& b : pool) {
          if (!b.empty()) { *static_cast<char*>(b.contents()) = 0; }
        }
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (rep > 0 && ms < best) { best = ms; }
    }
    std::printf("[boogu_perf] scratch alloc+free (%.0f MB, %zu buffers) "
                "%.2f ms per forward\n", total_mb,
                sizeof(elems) / sizeof(elems[0]), best);
  }
  EXPECT_TRUE(true);
}

// Does the MetalPerformancePrimitives matmul2d dense GEMM beat the scalar-FMA
// one at Boogu's shapes on THIS GPU? The DiTs bind matmul2d only when
// supports_matrix_cores() (Apple10 / M5) reports true, on the assumption that
// it needs hardware matrix units; this measures the same entry points the
// forward pass would use, so the gate can be decided by data on any box.
// Opt-in: VPIPE_BOOGU_GEMM_BENCH.
TEST(boogu_perf, mma_vs_scalar_gemm)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_bf16");
  ComputeLibrary lmma = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction f32t = lib.function("dense_gemm_t_f16");
  ComputeFunction fswi = lib.function("dense_gemm_swiglu_bm64_f16");
  ComputeFunction f_n128 = lmma.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_deep = lmma.function("dense_gemm_mma_t_n128x256_f16");
  ComputeFunction f_tn2 = lmma.function("dense_gemm_mma_t_n128x256_tn2_f16");
  BOOGU_REQUIRE(f32t.valid() && fswi.valid());
  std::printf("[boogu_perf] matrix-cores reported: %s | mma kernels: n128 %s "
              "deep %s tn2 %s\n", mc->supports_matrix_cores() ? "yes" : "NO",
              f_n128.valid() ? "ok" : "MISSING",
              f_deep.valid() ? "ok" : "MISSING",
              f_tn2.valid() ? "ok" : "MISSING");
  if (!f_n128.valid() || !f_deep.valid()) { return; }

  MetalBooguTransformer::Config c;
  const int H = c.hidden, KD = c.n_kv_heads * c.head_dim, FFI = 13568;
  int M = 2271;
  if (const char* s = std::getenv("VPIPE_BOOGU_BENCH_SEQ")) {
    M = std::atoi(s);
    if (M < 32) { M = 32; }
  }
  const int KMAX = FFI > H ? FFI : H, NMAX = 2 * FFI;
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * KMAX * 2);
  SharedBuffer wb = mc->make_shared_buffer((std::size_t)NMAX * KMAX * 2);
  SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * NMAX * 2);
  BOOGU_REQUIRE(!xb.empty() && !wb.empty() && !yb.empty());
  fill_normal_bf16_(xb, (std::size_t)M * KMAX, 1);
  fill_normal_bf16_(wb, (std::size_t)NMAX * KMAX, 2);

  const int ITERS = 8;
  auto time_ms = [&](const std::function<void(ComputeEncoder&)>& body) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) { body(enc); }
      }
      const auto t0 = std::chrono::steady_clock::now();
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (rep > 0 && ms < best) { best = ms; }
    }
    return best / ITERS;
  };
  auto bind = [&](ComputeEncoder& e, const ComputeFunction& fn, int N, int K) {
    e.set_function(fn);
    e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, wb);
    e.set_buffer(3, yb);
    e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
    e.set_constant(7, 0);
  };
  // The forward pass's own routing: K < 6144 -> n128 (RN 128), K < 12288 ->
  // tn2 (RN 512) when present, else the deep n128x256 (RN 256).
  auto mma = [&](ComputeEncoder& e, int N, int K) {
    int RN = 256;
    const ComputeFunction* fn = &f_deep;
    if (K < 6144) { fn = &f_n128; RN = 128; }
    else if (K < 12288 && f_tn2.valid()) { fn = &f_tn2; RN = 512; }
    bind(e, *fn, N, K);
    e.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  };
  auto scalar = [&](ComputeEncoder& e, int N, int K) {
    bind(e, f32t, N, K);
    e.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  };
  struct Sh { const char* tag; int N; int K; };
  const Sh shapes[] = {
      {"q/o", H, H}, {"k or v", KD, H}, {"ff-gate or up", FFI, H},
      {"ff-down", H, FFI},
  };
  double tot_s = 0.0, tot_m = 0.0;
  for (const auto& sh : shapes) {
    const double ms = time_ms([&](ComputeEncoder& e) {
      scalar(e, sh.N, sh.K);
    });
    const double mm = time_ms([&](ComputeEncoder& e) { mma(e, sh.N, sh.K); });
    const double fl = 2.0 * M * sh.N * sh.K;
    std::printf("[boogu_perf] %-14s N=%5d K=%5d | scalar %6.2f ms (%.2f "
                "TFLOP/s) | mma %6.2f ms (%.2f TFLOP/s) | %.2fx\n",
                sh.tag, sh.N, sh.K, ms, fl / (ms * 1e-3) / 1e12, mm,
                fl / (mm * 1e-3) / 1e12, ms / mm);
    // One single-stream block: q, k, v, o, gate, up, down.
    const int mult = (std::string(sh.tag) == "q/o") ? 2
                     : (std::string(sh.tag) == "k or v") ? 2
                     : (std::string(sh.tag) == "ff-gate or up") ? 2 : 1;
    tot_s += ms * mult;
    tot_m += mm * mult;
  }
  // The shipped scalar path fuses gate|up; charge it that way for a fair total.
  const double fused = time_ms([&](ComputeEncoder& e) {
    e.set_function(fswi);
    e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, yb);
    e.set_constant(3, H); e.set_constant(4, 2 * FFI);
    e.set_constant(5, M); e.set_constant(6, 0); e.set_constant(7, 0);
    e.dispatch({(unsigned)(((2 * FFI + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  });
  const double sep = time_ms([&](ComputeEncoder& e) {
    scalar(e, FFI, H); scalar(e, FFI, H);
  });
  std::printf("[boogu_perf] one single block GEMM total: scalar-unfused %.2f "
              "ms | scalar+fused-ff %.2f ms | mma %.2f ms (%.2fx vs shipped)\n",
              tot_s, tot_s - sep + fused, tot_m,
              (tot_s - sep + fused) / tot_m);
  EXPECT_TRUE(true);
}

// END-TO-END A/B of the two matmul2d routing decisions (split-K on the K=13568
// ff-down, TN=2 on the N=13568 ff-gate/up), ALTERNATING inside one process on
// ONE loaded model.
//
// This exists because the obvious way to measure it does not work. Running the
// four arms as four processes gave +10.7% at seq 4104 and -14% at seq 1032 for
// the SAME build -- the M5's clock is gated by the SoC power budget, so a
// batch of runs drifts by more than the effect. And two DiTs will not fit on a
// 16 GB box, so the arms have to share one set of weights and be re-routed
// between forwards (MetalBooguTransformer::set_gemm_route). Arms are cycled
// A,B,C,D,A,B,C,D... and each keeps its BEST time, so a monotone clock drift
// hits all four equally and the comparison survives it.
//
// Opt-in: VPIPE_BOOGU_GEMM_BENCH. VPIPE_BOOGU_BENCH_GRID picks the token grid
// (32 = 512px, 64 = 1024px), VPIPE_BOOGU_BENCH_ITERS the number of cycles.
TEST(boogu_perf, gemm_route_ab)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  auto dit = MetalBooguTransformer::load(root + "/transformer", mc,
                                         MetalBooguTransformer::Config{});
  BOOGU_REQUIRE(dit != nullptr);
  const auto& c = dit->config();
  auto envi = [](const char* k, int d) {
    const char* e = std::getenv(k); return (e && *e) ? std::atoi(e) : d; };
  const int grid = envi("VPIPE_BOOGU_BENCH_GRID", 64);
  const int cycles = envi("VPIPE_BOOGU_BENCH_ITERS", 4);
  const int TS = envi("VPIPE_BOOGU_BENCH_TS", 8);
  const int img_seq = grid * grid;
  SharedBuffer ctx =
      mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
  SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
  BOOGU_REQUIRE(!ctx.empty() && !lat.empty());
  fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 99);
  fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 7);

  struct Arm { const char* tag; bool sk; bool tn2; double best; };
  Arm arms[] = {
    {"splitk+tn2 (shipped)", true,  true,  1e30},
    {"splitk only",          true,  false, 1e30},
    {"tn2 only",             false, true,  1e30},
    {"neither (pre-change)", false, false, 1e30},
  };
  // The first forward pays pipeline-state compilation for whichever route it
  // takes, so warm EVERY arm once before timing any of them.
  for (Arm& a : arms) {
    dit->set_gemm_route(a.sk, a.tn2);
    dit->forward_dit(ctx, TS, lat, img_seq, grid * c.patch, grid * c.patch,
                     0.5f, {});
  }
  std::vector<float> ref;
  for (int cy = 0; cy < cycles; ++cy) {
    for (Arm& a : arms) {
      dit->set_gemm_route(a.sk, a.tn2);
      // A route that could not be enabled (kernel absent) would silently
      // duplicate another arm; report what actually ran.
      const auto t0 = std::chrono::steady_clock::now();
      SharedBuffer v =
          dit->forward_dit(ctx, TS, lat, img_seq, grid * c.patch,
                           grid * c.patch, 0.5f, {});
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (v.empty()) { EXPECT_TRUE(!v.empty()); return; }
      if (ms < a.best) { a.best = ms; }
      // Every arm must compute the same velocity. split-K folds its two
      // partial planes with one extra bf16 rounding, so it is close, not
      // bit-equal; a routing mistake would be O(1).
      const std::size_t n = (std::size_t)img_seq * c.out_channels;
      std::vector<float> got(n);
      const auto* p = static_cast<const std::uint16_t*>(v.contents());
      for (std::size_t i = 0; i < n; ++i) { got[i] = bf16_to_f32_(p[i]); }
      if (ref.empty()) { ref = got; }
      else {
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          const double d = (double)got[i] - (double)ref[i];
          num += d * d; den += (double)ref[i] * (double)ref[i];
        }
        const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
        EXPECT_TRUE(std::isfinite(rel) && rel < 2e-2);
      }
    }
  }
  const double base = arms[3].best;
  std::printf("[boogu_perf] gemm route A/B, seq=%d (grid %d, %d cycles):\n",
              TS + img_seq, grid, cycles);
  for (const Arm& a : arms) {
    std::printf("[boogu_perf]   %-22s %8.0f ms | %.3fx\n", a.tag, a.best,
                a.best > 0.0 ? base / a.best : 0.0);
  }
  EXPECT_TRUE(true);
}

// Which matmul2d VARIANT should each Boogu GEMM shape dispatch on a matrix-core
// GPU? mma_vs_scalar_gemm answers "matmul2d or not"; this answers "which one",
// which the M4-Pro bring-up could not measure (no matrix cores, so the whole
// matmul2d path was inert there and the routing was inherited from Krea-2).
//
// The inherited rule is K<6144 -> n128, K in [6144,12288) -> tn2, else the deep
// n128x256, plus split-K when K is an exact multiple of a KC the kernel is
// instantiated at. Krea-2's KC is 8192 for its K=16384 ff-down; Boogu's ff-down
// is K=13568, which 8192 does not divide -- so the deepest, slowest shape in the
// model had NO split-K available. 13568 = 2*6784 = 4*3392, both now built.
//
// Every variant is timed in ONE process, alternating, and each is checked
// against the shipped route's output (rel-L2) so a fast wrong tile cannot win.
// Opt-in: VPIPE_BOOGU_GEMM_BENCH. VPIPE_BOOGU_BENCH_SEQ picks M.
TEST(boogu_perf, mma_tile_sweep)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[boogu_perf] no matrix cores -- mma tile sweep skipped\n");
    return;
  }
  ComputeLibrary lmma = mc->load_library("dense_gemm_mma_bf16");
  ComputeLibrary elt = mc->load_library("llm_elementwise_bf16");
  ComputeFunction f_n128 = lmma.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_deep = lmma.function("dense_gemm_mma_t_n128x256_f16");
  ComputeFunction f_tn2 = lmma.function("dense_gemm_mma_t_n128x256_tn2_f16");
  ComputeFunction f_sk2 =
      lmma.function("dense_gemm_mma_splitk_n128x256_k6784_f16");
  ComputeFunction f_sk4 =
      lmma.function("dense_gemm_mma_splitk_n128x256_k3392_f16");
  ComputeFunction f_res = elt.function("residual_add_f16");
  BOOGU_REQUIRE(f_n128.valid() && f_deep.valid());

  MetalBooguTransformer::Config c;
  const int H = c.hidden, KD = c.n_kv_heads * c.head_dim, FFI = 13568;
  int M = 2271;
  if (const char* s = std::getenv("VPIPE_BOOGU_BENCH_SEQ")) {
    M = std::atoi(s);
    if (M < 32) { M = 32; }
  }
  const int KMAX = FFI > H ? FFI : H, NMAX = 2 * FFI;
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * KMAX * 2);
  SharedBuffer wb = mc->make_shared_buffer((std::size_t)NMAX * KMAX * 2);
  SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * NMAX * 2);
  SharedBuffer ref = mc->make_shared_buffer((std::size_t)M * NMAX * 2);
  // 4 planes of [M, H] for the widest split we bench.
  SharedBuffer skb = mc->make_shared_buffer((std::size_t)M * H * 4 * 2);
  BOOGU_REQUIRE(!xb.empty() && !wb.empty() && !yb.empty() && !ref.empty());
  fill_normal_bf16_(xb, (std::size_t)M * KMAX, 1);
  fill_normal_bf16_(wb, (std::size_t)NMAX * KMAX, 2);

  const int ITERS = 8;
  auto time_ms = [&](const std::function<void(ComputeEncoder&)>& body) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) { body(enc); } }
      const auto t0 = std::chrono::steady_clock::now();
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (rep > 0 && ms < best) { best = ms; }
    }
    return best / ITERS;
  };
  auto plain = [&](ComputeEncoder& e, const ComputeFunction& fn, int RN, int N,
                   int K, const SharedBuffer& out) {
    e.set_function(fn);
    e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, wb);
    e.set_buffer(3, out);
    e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
    e.set_constant(7, 0);
    e.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  };
  // Split into `splits` planes, then fold with residual_add -- the same shape
  // the forward pass would use.
  auto splitk = [&](ComputeEncoder& e, const ComputeFunction& fn, int splits,
                    int N, int K, const SharedBuffer& out) {
    const std::size_t plane = (std::size_t)M * N;
    e.set_function(fn);
    e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, skb);
    e.set_constant(3, K); e.set_constant(4, N); e.set_constant(5, M);
    e.dispatch({(unsigned)(((N + 255) / 256) * 256),
                (unsigned)((M + 127) / 128), (unsigned)splits}, {256, 1, 1});
    e.set_function(f_res);
    e.set_buffer(0, skb, 0); e.set_buffer(1, skb, plane * 2);
    e.set_buffer(2, out);
    e.set_constant(3, (int)plane);
    e.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
    for (int s = 2; s < splits; ++s) {
      e.set_function(f_res);
      e.set_buffer(0, out); e.set_buffer(1, skb, plane * (std::size_t)s * 2);
      e.set_buffer(2, out);
      e.set_constant(3, (int)plane);
      e.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
    }
  };

  struct Sh { const char* tag; int N; int K; };
  const Sh shapes[] = {
      {"q/o", H, H}, {"k or v", KD, H}, {"ff-gate or up", FFI, H},
      {"ff-down", H, FFI},
  };
  for (const auto& sh : shapes) {
    // The shipped route for this K, into `ref` -- the correctness baseline.
    int RN0 = 256;
    const ComputeFunction* f0 = &f_deep;
    if (sh.K < 6144) { f0 = &f_n128; RN0 = 128; }
    else if (sh.K < 12288 && f_tn2.valid()) { f0 = &f_tn2; RN0 = 512; }
    const double base = time_ms([&](ComputeEncoder& e) {
      plain(e, *f0, RN0, sh.N, sh.K, ref);
    });
    const double fl = 2.0 * M * sh.N * sh.K;
    std::printf("[boogu_perf] %-14s M=%d N=%5d K=%5d | SHIPPED %6.2f ms "
                "(%.2f TFLOP/s)\n", sh.tag, M, sh.N, sh.K, base,
                fl / (base * 1e-3) / 1e12);
    struct Cand { const char* name; const ComputeFunction* fn; int rn;
                  int splits; };
    const Cand cands[] = {
      {"n128",       &f_n128, 128, 0},
      {"n128x256",   &f_deep, 256, 0},
      {"tn2",        &f_tn2,  512, 0},
      {"splitk x2",  &f_sk2,  0,   2},
      {"splitk x4",  &f_sk4,  0,   4},
    };
    for (const Cand& cd : cands) {
      if (cd.fn == nullptr || !cd.fn->valid()) { continue; }
      if (cd.splits > 0 &&
          (skb.empty() || !f_res.valid() || sh.K % cd.splits != 0 ||
           (sh.K / cd.splits) != (cd.splits == 2 ? 6784 : 3392))) {
        continue;
      }
      const double ms = time_ms([&](ComputeEncoder& e) {
        if (cd.splits > 0) { splitk(e, *cd.fn, cd.splits, sh.N, sh.K, yb); }
        else { plain(e, *cd.fn, cd.rn, sh.N, sh.K, yb); }
      });
      const double rel = rel_l2_bf16_(yb, ref, (std::size_t)M * sh.N);
      std::printf("[boogu_perf]   %-10s %6.2f ms (%5.2f TFLOP/s) | %.3fx "
                  "| rel-L2 %.2e\n", cd.name, ms, fl / (ms * 1e-3) / 1e12,
                  base / ms, rel);
      // A variant that computes something else is not a candidate. Split-K
      // folds partials in f16, so it is close-but-not-equal by construction.
      EXPECT_TRUE(std::isfinite(rel) && rel < 5e-2);
    }
  }
  EXPECT_TRUE(true);
}

// How much does the head_dim-120 -> 128 zero-pad actually cost? Times the WHOLE
// attention stage -- transpose+rope, the flash kernel, and the un-transpose --
// both ways, ALTERNATING in one process so thermal drift cannot favour either.
// (End-to-end A/B across two processes could not resolve it: at seq 8415 the
// run-to-run spread is ~4%, larger than the effect.)
// Opt-in: VPIPE_BOOGU_GEMM_BENCH. VPIPE_BOOGU_BENCH_SEQ picks the length.
TEST(boogu_perf, attention_native_vs_padded)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  ComputeLibrary la = mc->load_library("attn_steel");
  ComputeLibrary elt = mc->load_library("llm_elementwise_bf16");
  ComputeLibrary rope = mc->load_library("rope_bf16");
  BOOGU_REQUIRE(la.valid());
  ComputeFunction f_tr = elt.function("transpose_abd_f16");
  ComputeFunction f_trp = elt.function("transpose_abd_pad_f16");
  ComputeFunction f_tru = elt.function("transpose_abd_unpad_f16");
  ComputeFunction f_rr = rope.function("transpose_rope_pair_ftab_f16");
  ComputeFunction f_rrp = rope.function("transpose_rope_pair_ftab_pad_f16");
  BOOGU_REQUIRE(f_tr.valid() && f_trp.valid() && f_tru.valid());
  BOOGU_REQUIRE(f_rr.valid() && f_rrp.valid());

  MetalBooguTransformer::Config c;
  const int HD = c.head_dim, HED = c.n_heads, KVH = c.n_kv_heads;
  int L = 2271;
  if (const char* s = std::getenv("VPIPE_BOOGU_BENCH_SEQ")) {
    L = std::atoi(s);
    if (L < 64) { L = 64; }
  }
  // C++ mirror of mlx::steel::AttnParams (same layout the DiT writes).
  struct P {
    int B, H, D; int qL, kL; int gqa_factor; float scale;
    int NQ, NK; int NQ_aligned, NK_aligned; int qL_rem, kL_rem, qL_off;
    std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  };
  const int A_BQ = 32, A_BK = 16;
  auto mkparams = [&](int D) {
    SharedBuffer pb = mc->make_shared_buffer(sizeof(P));
    auto* p = static_cast<P*>(pb.contents());
    p->B = 1; p->H = HED; p->D = D; p->qL = L; p->kL = L;
    p->gqa_factor = HED / KVH; p->scale = 1.0f / std::sqrt((float)HD);
    p->NQ = (L + A_BQ - 1) / A_BQ; p->NK = (L + A_BK - 1) / A_BK;
    p->NQ_aligned = L / A_BQ; p->NK_aligned = L / A_BK;
    p->qL_rem = L - p->NQ_aligned * A_BQ;
    p->kL_rem = L - p->NK_aligned * A_BK;
    p->qL_off = 0;
    p->Q_strides[0] = (std::int64_t)HED * L * D;
    p->Q_strides[1] = (std::int64_t)L * D; p->Q_strides[2] = D;
    p->K_strides[0] = (std::int64_t)KVH * L * D;
    p->K_strides[1] = (std::int64_t)L * D; p->K_strides[2] = D;
    p->V_strides[0] = p->K_strides[0];
    p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = D;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = D;
    return pb;
  };
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, (L % A_BQ) == 0).set_bool(201, (L % A_BK) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false);
  ComputeFunction a120 = la.function("attn_steel_h_bd120_bf16", fc);
  ComputeFunction a128 = la.function("attn_steel_h_bd128_bf16", fc);
  BOOGU_REQUIRE(a128.valid());
  if (!a120.valid()) {
    std::printf("[boogu_perf] attn_steel_h_bd120_bf16 MISSING\n");
    return;
  }
  SharedBuffer p120 = mkparams(120), p128 = mkparams(128);
  // Token-major projections (what the block GEMMs produce) + head-major scratch.
  SharedBuffer qb = mc->make_shared_buffer((std::size_t)L * HED * HD * 2);
  SharedBuffer kb = mc->make_shared_buffer((std::size_t)L * KVH * HD * 2);
  SharedBuffer tab = mc->make_shared_buffer((std::size_t)L * HD * 4);
  SharedBuffer out = mc->make_shared_buffer((std::size_t)L * HED * HD * 2);
  auto scratch = [&](int D) {
    return std::array<SharedBuffer, 4>{
        mc->make_shared_buffer((std::size_t)HED * L * D * 2),
        mc->make_shared_buffer((std::size_t)KVH * L * D * 2),
        mc->make_shared_buffer((std::size_t)KVH * L * D * 2),
        mc->make_shared_buffer((std::size_t)HED * L * D * 2)};
  };
  auto s120 = scratch(120), s128 = scratch(128);
  BOOGU_REQUIRE(!qb.empty() && !out.empty() && !s120[0].empty() &&
                !s128[0].empty());
  fill_normal_bf16_(qb, (std::size_t)L * HED * HD, 5);
  fill_normal_bf16_(kb, (std::size_t)L * KVH * HD, 6);
  std::memset(tab.contents(), 0, tab.byte_size());

  // One full attention stage at head width D.
  auto stage = [&](ComputeEncoder& e, int D, const ComputeFunction& fn,
                   const SharedBuffer& params,
                   const std::array<SharedBuffer, 4>& sc) {
    const bool pad = (D != HD);
    auto trrope = [&](const SharedBuffer& in, const SharedBuffer& o, int nh) {
      e.set_function(pad ? f_rrp : f_rr);
      e.set_buffer(0, in); e.set_buffer(1, o);
      e.set_buffer(2, tab); e.set_buffer(3, tab);
      e.set_constant(4, nh); e.set_constant(5, L); e.set_constant(6, HD);
      if (pad) { e.set_constant(7, D); }
      e.dispatch({(unsigned)(D / 2), (unsigned)L, (unsigned)nh},
                 {(unsigned)(D / 2), 1, 1});
    };
    trrope(qb, sc[0], HED);
    trrope(kb, sc[1], KVH);
    e.set_function(pad ? f_trp : f_tr);                      // v transpose
    e.set_buffer(0, kb); e.set_buffer(1, sc[2]);
    e.set_constant(2, L); e.set_constant(3, KVH); e.set_constant(4, HD);
    if (pad) { e.set_constant(5, D); }
    e.dispatch({(unsigned)D, (unsigned)KVH, (unsigned)L},
               {(unsigned)D, 1, 1});
    e.set_function(fn);                                      // flash attention
    e.set_buffer(0, sc[0]); e.set_buffer(1, sc[1]); e.set_buffer(2, sc[2]);
    e.set_buffer(3, sc[3]); e.set_buffer(4, params);
    e.dispatch({32 * (unsigned)((L + A_BQ - 1) / A_BQ), 4 * (unsigned)HED, 1},
               {32, 4, 1});
    e.set_function(pad ? f_tru : f_tr);                      // un-transpose
    e.set_buffer(0, sc[3]); e.set_buffer(1, out);
    e.set_constant(2, HED); e.set_constant(3, L); e.set_constant(4, HD);
    if (pad) { e.set_constant(5, D); }
    e.dispatch({(unsigned)HD, (unsigned)L, (unsigned)HED},
               {(unsigned)HD, 1, 1});
  };
  auto once = [&](bool native) {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      if (native) { stage(e, 120, a120, p120, s120); }
      else        { stage(e, 128, a128, p128, s128); }
    }
    const auto t0 = std::chrono::steady_clock::now();
    st.commit().wait();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };
  double b120 = 1e30, b128 = 1e30;
  for (int rep = 0; rep < 7; ++rep) {          // alternate; rep 0 warms up
    const double t1 = once(true), t2 = once(false);
    if (rep > 0) {
      b120 = t1 < b120 ? t1 : b120;
      b128 = t2 < b128 ? t2 : b128;
    }
  }
  // Useful attention FLOPs (QK + PV over the REAL 120 dims), so both routes are
  // credited the same work and the pad shows up as lost throughput.
  const double fl = 2.0 * 2.0 * (double)L * L * HD * HED;
  std::printf("[boogu_perf] attention stage L=%d: bd120 %.2f ms (%.2f TFLOP/s) "
              "| bd128+pad %.2f ms (%.2f TFLOP/s) | %.3fx\n", L, b120,
              fl / (b120 * 1e-3) / 1e12, b128, fl / (b128 * 1e-3) / 1e12,
              b128 / b120);
  EXPECT_TRUE(true);
}

// The steel BlockMMA dense GEMM (gemm/dense_gemm_steel.metal) vs the scalar-FMA
// dense_gemm_t at Boogu's shapes. simdgroup MMA measures 10.14 TFLOP/s on this
// box against 7.88 for scalar FMA (gemm_mma.alu_rate_f16_vs_f32), and the DiT's
// GEMMs sit at 6.9-7.2 -- i.e. ~90% of the SCALAR roofline but only ~70% of the
// MMA one. This asks whether a real memory-fed MMA GEMM can collect that gap.
// Correctness first (rel-L2 vs the scalar kernel), then throughput per tile.
// Opt-in: VPIPE_BOOGU_GEMM_BENCH.
TEST(boogu_perf, steel_mma_dense_gemm)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_bf16");
  ComputeLibrary ls = mc->load_library("dense_gemm_steel_bf16");
  ComputeFunction f_sc = lib.function("dense_gemm_t_f16");
  BOOGU_REQUIRE(f_sc.valid());
  if (!ls.valid()) {
    std::printf("[boogu_perf] dense_gemm_steel_bf16 MISSING\n");
    return;
  }
  struct Tile { const char* fn; int bm, bn; };
  const Tile tiles[] = {
      {"dense_gemm_steel_64x64x16_f16", 64, 64},
      {"dense_gemm_steel_64x64x32_f16", 64, 64},
      {"dense_gemm_steel_32x32x32_f16", 32, 32},
      {"dense_gemm_steel_64x32x32_f16", 64, 32},
      {"dense_gemm_steel_32x64x32_f16", 32, 64},
      {"dense_gemm_steel_128x64x16_f16", 128, 64},
      {"dense_gemm_steel_64x128x16_f16", 64, 128},
      {"dense_gemm_steel_128x128x16_f16", 128, 128},
  };
  // The 8-simdgroup tiles need a 256-thread threadgroup.
  auto tg_threads = [](const char* nm) {
    return (std::string(nm).find("128x") != std::string::npos ||
            std::string(nm).find("x128x") != std::string::npos) ? 256u : 128u;
  };

  MetalBooguTransformer::Config c;
  const int H = c.hidden, KD = c.n_kv_heads * c.head_dim, FFI = 13568;
  int M = 2271;
  if (const char* s = std::getenv("VPIPE_BOOGU_BENCH_SEQ")) {
    M = std::atoi(s);
    if (M < 32) { M = 32; }
  }
  const int KMAX = FFI > H ? FFI : H, NMAX = FFI;
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * KMAX * 2);
  SharedBuffer wb = mc->make_shared_buffer((std::size_t)NMAX * KMAX * 2);
  SharedBuffer ya = mc->make_shared_buffer((std::size_t)M * NMAX * 2);
  SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * NMAX * 2);
  BOOGU_REQUIRE(!xb.empty() && !wb.empty() && !ya.empty() && !yb.empty());
  fill_normal_bf16_(xb, (std::size_t)M * KMAX, 21);
  fill_normal_bf16_(wb, (std::size_t)NMAX * KMAX, 22);

  auto run_scalar = [&](ComputeEncoder& e, const SharedBuffer& out, int N,
                        int K) {
    e.set_function(f_sc);
    e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, wb);
    e.set_buffer(3, out);
    e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
    e.set_constant(7, 0);
    e.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
  };
  auto run_steel = [&](ComputeEncoder& e, const ComputeFunction& fn, int bm,
                       int bn, unsigned tgz, const SharedBuffer& out, int N,
                       int K) {
    e.set_function(fn);
    e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, wb);
    e.set_buffer(3, out);
    e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
    e.set_constant(7, 0);
    e.dispatch({(unsigned)(((N + bn - 1) / bn) * tgz),
                (unsigned)((M + bm - 1) / bm), 1}, {tgz, 1, 1});
  };
  const int ITERS = 6;
  auto time_ms = [&](const std::function<void(ComputeEncoder&)>& body) {
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) { body(e); }
      }
      const auto t0 = std::chrono::steady_clock::now();
      st.commit().wait();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      if (rep > 0 && ms < best) { best = ms; }
    }
    return best / ITERS;
  };
  struct Sh { const char* tag; int N; int K; };
  const Sh shapes[] = {{"q/o", H, H}, {"k or v", KD, H},
                       {"ff-gate/up", FFI, H}, {"ff-down", H, FFI}};
  for (const auto& sh : shapes) {
    const double fl = 2.0 * M * sh.N * sh.K;
    const double ms_sc = time_ms([&](ComputeEncoder& e) {
      run_scalar(e, ya, sh.N, sh.K);
    });
    std::printf("[boogu_perf] %-11s N=%5d K=%5d | scalar     %7.2f ms "
                "(%.2f TFLOP/s)\n", sh.tag, sh.N, sh.K, ms_sc,
                fl / (ms_sc * 1e-3) / 1e12);
    for (const auto& t : tiles) {
      ComputeFunction fn = ls.function(t.fn);
      if (!fn.valid()) { continue; }
      // Correctness against the scalar kernel (same inputs, bf16 both, so the
      // only difference is accumulation order).
      {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          run_steel(e, fn, t.bm, t.bn, tg_threads(t.fn), yb, sh.N, sh.K);
        }
        st.commit().wait();
      }
      const double d =
          rel_l2_bf16_(yb, ya, (std::size_t)M * sh.N);
      const double ms = time_ms([&](ComputeEncoder& e) {
        run_steel(e, fn, t.bm, t.bn, tg_threads(t.fn), yb, sh.N, sh.K);
      });
      std::printf("[boogu_perf] %-11s N=%5d K=%5d | %-14s %7.2f ms "
                  "(%.2f TFLOP/s) | %.3fx | rel-L2 %.2e\n", sh.tag, sh.N, sh.K,
                  t.fn + 17, ms, fl / (ms * 1e-3) / 1e12, ms_sc / ms, d);
      EXPECT_TRUE(d < 1e-2);
    }
  }
}

// A quantized DiT's velocity against the SAME inputs through the bf16 one --
// the check that says whether a quantization choice damaged the model, without
// needing an external golden. Both DiTs are resident at once (bf16 ~20 GB +
// quantized ~7 GB), so this wants a roomy box.
//   VPIPE_BOOGU_TEST_MODEL_PATH  the bf16 reference pipeline
//   VPIPE_BOOGU_Q4_MODEL_PATH    the quantized pipeline to judge
// Reference bands (Boogu-Image-0.1-Edit-Turbo, 4-bit g32, seq 80): the DiT
// golden's own bf16-vs-reference floor is 0.035, so anything at or under ~0.10
// is quantization noise of the same order as the port's arithmetic noise.
TEST(boogu_golden, quantized_velocity_vs_bf16)
{
  const std::string root = model_root_();
  const char* qp = std::getenv("VPIPE_BOOGU_Q4_MODEL_PATH");
  if (root.empty() || qp == nullptr || *qp == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const int grid = 4, TS = 8;
  MetalBooguTransformer::Config cfg;
  const int lh = grid * cfg.patch, lw = grid * cfg.patch;
  const int img_seq = grid * grid;
  SharedBuffer ctx, lat, ref;
  auto run = [&](const std::string& dir) -> SharedBuffer {
    auto dit = MetalBooguTransformer::load(dir + "/transformer", mc,
                                           MetalBooguTransformer::Config{});
    if (!dit) { return {}; }
    const auto& c = dit->config();
    if (ctx.empty()) {
      ctx = mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
      lat = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
      ref = mc->make_shared_buffer((std::size_t)img_seq * c.x_in() * 2);
      fill_normal_bf16_(ctx, (std::size_t)TS * c.instruct_dim, 31);
      fill_normal_bf16_(lat, (std::size_t)img_seq * c.x_in(), 32);
      fill_normal_bf16_(ref, (std::size_t)img_seq * c.x_in(), 33);
    }
    std::vector<MetalBooguTransformer::RefImage> refs;
    MetalBooguTransformer::RefImage ri;
    ri.latents = mc->make_shared_buffer(ref.byte_size());
    std::memcpy(ri.latents.contents(), ref.contents(), ref.byte_size());
    ri.seq = img_seq; ri.grid_h = lh; ri.grid_w = lw;
    refs.push_back(std::move(ri));
    return dit->forward_dit(ctx, TS, lat, img_seq, lh, lw, 0.35f, refs);
  };
  SharedBuffer v_bf16 = run(root);
  BOOGU_REQUIRE(!v_bf16.empty());
  // Keep the bf16 result, then load the quantized DiT (the first is destroyed
  // by then, so peak is one DiT plus a velocity).
  std::vector<std::uint16_t> keep((std::size_t)img_seq * 64);
  std::memcpy(keep.data(), v_bf16.contents(), keep.size() * 2);
  v_bf16 = SharedBuffer{};
  SharedBuffer v_q = run(std::string(qp));
  BOOGU_REQUIRE(!v_q.empty());
  const std::size_t n = keep.size();
  BOOGU_REQUIRE(v_q.byte_size() >= n * 2);
  const auto* q = static_cast<const std::uint16_t*>(v_q.contents());
  double num = 0.0, den = 0.0, qm = 0.0, bm = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = bf16_to_f32_(q[i]), b = bf16_to_f32_(keep[i]);
    num += (a - b) * (a - b); den += b * b;
    qm += a * a; bm += b * b;
  }
  const double rl2 = den > 0.0 ? std::sqrt(num / den) : 1.0;
  std::printf("[boogu_golden] quantized velocity vs bf16: rel-L2 %.5f "
              "(|v| ratio %.4f)\n", rl2,
              bm > 0.0 ? std::sqrt(qm / bm) : 0.0);
  // Generous: this is a whole-model quantization, judged against the port's own
  // bf16 output. A wrecked modulation shows up at O(0.5)+, not O(0.1).
  EXPECT_TRUE(rl2 < 0.25);
  // Magnitude must not drift either -- a broken scale/gate path changes |v|
  // even when the direction survives.
  EXPECT_TRUE(bm > 0.0 && std::sqrt(qm / bm) > 0.8 &&
              std::sqrt(qm / bm) < 1.25);
}

// ---- boogu_e2e: the whole model, on one box -------------------------------
// conditioner (Qwen3-VL 8B mllm) -> generate-image (the 10B NextDiT, 4 DMD
// steps) -> vae-decode (the FLUX.1 AutoencoderKL at patch 1). Everything above
// this line tests one component against a golden or against itself; this is
// the only test that says the three fit together and that a 16 GB box can hold
// the sequence. Opt-in (VPIPE_BOOGU_E2E) because it loads ~13 GB in turn and
// runs for minutes.
//   VPIPE_BOOGU_E2E_PX   output side (default 512)
//   VPIPE_BOOGU_E2E_OUT  directory to write boogu_t2i.png into
namespace {

class BooguSourceText : public vpipe::TypedStage<BooguSourceText> {
public:
  static constexpr const char* kTypeName = "ut-boogu-t2i-src";
  BooguSourceText(const vpipe::SessionContextIntf* s, std::string id,
                  std::vector<vpipe::InEdge> ip, vpipe::FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::string prompt;
  bool done = false;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      co_await ctx.write(0, std::make_unique<vpipe::FlexDataPayload>(
                                vpipe::FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};

class BooguSink : public vpipe::TypedStage<BooguSink> {
public:
  static constexpr const char* kTypeName = "ut-boogu-t2i-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<vpipe::BeatPayloadIntf>> captured;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (p) { captured.push_back(std::move(p)); }
    ctx.signal_done();
    co_return;
  }
};

}  // namespace

TEST(boogu_e2e, text_to_image_end_to_end)
{
  if (std::getenv("VPIPE_BOOGU_E2E") == nullptr) { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const char* outd = std::getenv("VPIPE_BOOGU_E2E_OUT");
  int px = 512;
  if (const char* p = std::getenv("VPIPE_BOOGU_E2E_PX")) {
    px = std::atoi(p);
    if (px < 256) { px = 256; }
  }

  auto pl = std::make_unique<Pipeline>("bt", &sess);
  auto srcu = std::make_unique<BooguSourceText>(&sess, "src",
                                                std::vector<InEdge>{},
                                                FlexData::make_object());
  srcu->prompt = "a red fox sitting in fresh snow, photorealistic";
  auto* src = static_cast<BooguSourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(cc));
  auto* cond =
      static_cast<DiffusionConditionerStage*>(pl->insert_stage(std::move(cu)));
  ASSERT_TRUE(cond->config_error().empty());

  FlexData tc = FlexData::make_object();
  tc.as_object().insert("hf_dir", FlexData::make_string(root));
  tc.as_object().insert("height", FlexData::make_int(px));
  tc.as_object().insert("width", FlexData::make_int(px));
  tc.as_object().insert("steps", FlexData::make_int(4));    // Turbo / DMD
  tc.as_object().insert("seed", FlexData::make_int(42));
  auto tu = std::make_unique<GenerateImageStage>(
      &sess, "t2i", std::vector<InEdge>{{cond, 0}}, std::move(tc));
  auto* t2i = static_cast<GenerateImageStage*>(pl->insert_stage(std::move(tu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vc = FlexData::make_object();
  vc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vc));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vu)));
  ASSERT_TRUE(vae->config_error().empty());

  auto sku = std::make_unique<BooguSink>(&sess, "sink",
                                         std::vector<InEdge>{{vae, 0}},
                                         FlexData::make_object());
  auto* sink = static_cast<BooguSink*>(pl->insert_stage(std::move(sku)));

  if (outd != nullptr && *outd != '\0') {
    FlexData sc = FlexData::make_object();
    sc.as_object().insert(
        "path", FlexData::make_string(std::string(outd) + "/boogu_t2i.png"));
    auto su = std::make_unique<SaveImageStage>(
        &sess, "save", std::vector<InEdge>{{vae, 0}}, std::move(sc));
    pl->insert_stage(std::move(su));
  }

  const auto t0 = std::chrono::steady_clock::now();
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();

  if (t2i->latents_emitted() == 0) {
    std::printf("[boogu_e2e] no latent emitted (model not loaded?)\n");
    EXPECT_TRUE(false);
    return;
  }
  BOOGU_REQUIRE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  BOOGU_REQUIRE(tb != nullptr);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8);
  BOOGU_REQUIRE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
                tb->shape[1] == px && tb->shape[2] == px);
  // Coherence, not identity: a broken conditioning or a dead VAE gives a flat
  // or saturated field. Require real spread and that it is not almost all one
  // value (the two ways this pipeline has actually failed).
  const std::uint8_t* d = tb->data.data();
  const std::size_t n = (std::size_t)3 * px * px;
  double mean = 0.0;
  std::array<std::size_t, 256> hist{};
  for (std::size_t i = 0; i < n; ++i) { mean += d[i]; ++hist[d[i]]; }
  mean /= (double)n;
  double var = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)d[i] - mean; var += e * e;
  }
  const double sd = std::sqrt(var / (double)n);
  std::size_t top = 0;
  for (std::size_t h : hist) { if (h > top) { top = h; } }
  std::printf("[boogu_e2e] %dx%d in %.1f s | mean %.1f sd %.1f | most common "
              "value %.1f%% of pixels\n", px, px, secs, mean, sd,
              100.0 * (double)top / (double)n);
  EXPECT_TRUE(sd > 12.0);
  EXPECT_TRUE((double)top / (double)n < 0.35);
}

// The Qwen3-VL tower's cost in Boogu's edit conditioning, and what its
// attention actually runs on.
//
// The tower gets matmul2d for its GEMMs on a matrix-core GPU like every other
// encoder here, but NOT a flash attention kernel, and for two compounding
// reasons: Boogu's ViT is 1152 hidden / 16 heads = head_dim 72, and every
// flash kernel in the tree is instantiated at 64/128/256/384/512 -- so the
// matmul2d (sdpa_full_mma2_d64) and NAX (attn_steel_nax bd64) paths are both
// gated off; and the tower runs bf16 (to match the reference the conditioning
// was tuned against), where the steel simdgroup-matrix flash is f16-only. What
// is left is the scalar O(n^2) sdpa over 27 layers. This measures what that
// costs at the shape the pipeline actually uses -- the VLM image is capped at
// 384x384, so 24x24 merge-2 patches = 576 tokens -- so the gap is a number
// rather than a worry. Opt-in: VPIPE_BOOGU_GEMM_BENCH.
TEST(boogu_perf, vision_tower_cost)
{
  if (std::getenv("VPIPE_BOOGU_GEMM_BENCH") == nullptr) { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string enc_dir = root + "/mllm";
  ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(enc_dir);
  BOOGU_REQUIRE(mcfg.has_value());
  auto vcfg = MetalQwenVisionEncoder::config_from(*mcfg);
  vcfg.weight_prefix = "model.visual.";
  vcfg.use_bf16 = true;
  vcfg.min_pixels = 65536;
  vcfg.max_pixels = 16777216;
  for (int i = 0; i < 3; ++i) {
    vcfg.image_mean[i] = 0.5f; vcfg.image_std[i] = 0.5f;
  }
  std::printf("[boogu_perf] tower: hidden %d heads %d head_dim %d, bf16=%d "
              "(flash kernels exist at 64/128/256/384/512)\n",
              vcfg.hidden, vcfg.n_heads, vcfg.head_dim(),
              vcfg.use_bf16 ? 1 : 0);
  auto tower = MetalQwenVisionEncoder::load(enc_dir, mc, vcfg);
  BOOGU_REQUIRE(tower != nullptr);

  const int side = 384;                     // the pipeline's VLM cap
  std::vector<std::uint8_t> rgb((std::size_t)3 * side * side);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = (std::uint8_t)((i * 37 + (i >> 8) * 11) & 0xff);
  }
  double best = 1e30;
  int tokens = 0;
  for (int rep = 0; rep < 4; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    auto r = tower->encode(rgb.data(), side, side);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    if (r.embeddings.empty()) { EXPECT_TRUE(false); return; }
    tokens = r.n_tokens;
    if (rep > 0 && ms < best) { best = ms; }
  }
  std::printf("[boogu_perf] tower encode %dx%d -> %d merged tokens: %.1f ms\n",
              side, side, tokens, best);
  EXPECT_TRUE(tokens > 0 && best < 5000.0);
}

// The zero-padded flash attention must equal the scalar SDPA it replaces.
//
// Boogu's tower is head_dim 72 and every flash kernel is instantiated at
// 64/128/256/384/512, so the tower padded q/k/v out to bd128 to reach one at
// all. That is exact ONLY if the pad lanes are zero -- a zero q/k lane adds
// nothing to the dot, and the widened V's pad columns are dropped by the
// unpad -- and only if `scale` stays 1/sqrt(72) rather than 1/sqrt(128). Both
// are easy to get wrong and neither changes the output SHAPE, so compare the
// embeddings and the deepstack features against VPIPE_QWEN_VISION_NO_ATTN_PAD.
TEST(boogu_smoke, padded_tower_attention_matches_scalar)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string enc_dir = root + "/mllm";
  ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(enc_dir);
  BOOGU_REQUIRE(mcfg.has_value());
  auto vcfg = MetalQwenVisionEncoder::config_from(*mcfg);
  vcfg.weight_prefix = "model.visual.";
  vcfg.use_bf16 = true;
  vcfg.min_pixels = 65536;
  vcfg.max_pixels = 16777216;
  for (int i = 0; i < 3; ++i) {
    vcfg.image_mean[i] = 0.5f; vcfg.image_std[i] = 0.5f;
  }
  if (vcfg.head_dim() == 64) { return; }   // native; nothing to pad

  const int side = 384;
  std::vector<std::uint8_t> rgb((std::size_t)3 * side * side);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = (std::uint8_t)((i * 61 + (i >> 9) * 7) & 0xff);
  }
  // Result::embeddings follows the encoder's dtype -- ask, do not assume.
  auto dec = [](std::uint16_t v, bool bf16) -> float {
    if (bf16) { return bf16_to_f32_(v); }
    _Float16 h; std::memcpy(&h, &v, 2); return (float)h;
  };
  auto run = [&](bool no_pad, std::vector<float>* emb,
                 std::vector<std::vector<float>>* ds) {
    if (no_pad) { ::setenv("VPIPE_QWEN_VISION_NO_ATTN_PAD", "1", 1); }
    auto tower = MetalQwenVisionEncoder::load(enc_dir, mc, vcfg);
    if (no_pad) { ::unsetenv("VPIPE_QWEN_VISION_NO_ATTN_PAD"); }
    if (tower == nullptr) { return; }
    auto r = tower->encode(rgb.data(), side, side);
    if (r.embeddings.empty() || r.n_tokens <= 0) { return; }
    const std::size_t n = (std::size_t)r.n_tokens * r.out_hidden;
    const auto* p = static_cast<const std::uint16_t*>(r.embeddings.contents());
    emb->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      (*emb)[i] = dec(p[i], tower->is_bf16());
    }
    ds->resize(r.deepstack.size());
    for (std::size_t d = 0; d < r.deepstack.size(); ++d) {
      const auto* q =
          static_cast<const std::uint16_t*>(r.deepstack[d].contents());
      (*ds)[d].resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        (*ds)[d][i] = dec(q[i], tower->is_bf16());
      }
    }
  };
  std::vector<float> a, b;
  std::vector<std::vector<float>> da, db;
  run(/*no_pad=*/false, &a, &da);
  run(/*no_pad=*/true, &b, &db);
  BOOGU_REQUIRE(!a.empty() && a.size() == b.size());
  auto rel = [](const std::vector<float>& x, const std::vector<float>& y) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double d = (double)x[i] - (double)y[i];
      num += d * d; den += (double)y[i] * (double)y[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : (num == 0.0 ? 0.0 : 1.0);
  };
  const double re = rel(a, b);
  std::printf("[boogu_smoke] padded-flash vs scalar tower: embeddings rel-L2 "
              "%.6g (head_dim %d -> bd128)\n", re, vcfg.head_dim());
  // Not bit-equal, and the bound is loose for a reason worth recording. The
  // measured matrix, same image, pad-vs-scalar:
  //        bf16 + NAX 0.0659 | bf16 + ALU steel 0.0580
  //        f16  + NAX 0.0114 | f16  + ALU steel 0.0086
  // Changing DTYPE moves it 5.7x; changing KERNEL moves it 1.14x. That is the
  // signature of accumulated rounding (bf16 has 3 fewer mantissa bits, and 27
  // blocks compound it), not of a broken pad -- a non-zeroed pad lane or a
  // scale left at 1/sqrt(128) would be dtype-INDEPENDENT and would still be
  // ~0.06 in f16. It is also the same order as the bf16 tower's own 0.052
  // distance from an fp32 oracle, i.e. inside the noise that mode already
  // accepts by construction (Config::use_bf16 is fidelity, not accuracy).
  EXPECT_TRUE(std::isfinite(re) && re < 8e-2);
  BOOGU_REQUIRE(da.size() == db.size());
  for (std::size_t d = 0; d < da.size(); ++d) {
    const double rd = rel(da[d], db[d]);
    std::printf("[boogu_smoke]   deepstack%zu rel-L2 %.6g\n", d, rd);
    EXPECT_TRUE(std::isfinite(rd) && rd < 8e-2);
  }
}

// THE MEMBER THE AUTOTUNE ACTUALLY PICKED, against the scalar reference.
//
// vae_flash_attn_matches_scalar above skips without matrix cores, so on an M4
// Pro -- where the tuner picks the MATERIALIZED member, which the matrix-core
// boxes never see -- nothing checked the mid attention at all. The tuner
// itself cannot: it ranks candidates by TIME, on a synthetic block whose
// scores are tiny (q.k ~ -15 at D=512), so a member that is fast and wrong
// wins and says nothing.
//
// Swept over mid-block token counts, because the SHAPE of the error is what
// names the cause: flat across 256/1024/4096 is a range or algebra failure,
// while an error that tracks the tile count is a tiling bug.
TEST(boogu_smoke, vae_selected_mid_attn_matches_scalar)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  for (const int px : {128, 256, 512}) {
    const int H = px, W = px, lh = H / 8, lw = W / 8;
    std::vector<float> img((std::size_t)3 * H * W);
    for (std::size_t i = 0; i < img.size(); ++i) {
      img[i] = std::sin((float)i * 0.017f);
    }
    std::vector<float> lat((std::size_t)16 * lh * lw);
    { std::mt19937 rng(4242);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (float& v : lat) { v = nd(rng); } }

    auto run = [&](bool scalar, std::vector<float>* z_out,
                   std::vector<float>* rgb_out) {
      if (scalar) { ::setenv("VPIPE_FLUX2_NO_MMA_ATTN", "1", 1); }
      auto vae = MetalFlux2Vae::load(root + "/vae", mc,
                                     MetalFlux2Vae::Config{},
                                     /*with_encoder=*/true);
      if (scalar) { ::unsetenv("VPIPE_FLUX2_NO_MMA_ATTN"); }
      if (vae == nullptr) { return; }
      SharedBuffer in = mc->make_shared_buffer(img.size() * 2);
      if (in.empty()) { return; }
      { auto* d = static_cast<_Float16*>(in.contents());
        for (std::size_t i = 0; i < img.size(); ++i) {
          d[i] = (_Float16)img[i];
        } }
      SharedBuffer z = vae->encode(in, H, W);
      if (!z.empty() && z.byte_size() >= lat.size() * 2) {
        const auto* p = static_cast<const _Float16*>(z.contents());
        z_out->resize(lat.size());
        for (std::size_t i = 0; i < lat.size(); ++i) {
          (*z_out)[i] = (float)p[i];
        }
      }
      SharedBuffer zin = mc->make_shared_buffer(lat.size() * 2);
      if (zin.empty()) { return; }
      { auto* d = static_cast<_Float16*>(zin.contents());
        for (std::size_t i = 0; i < lat.size(); ++i) {
          d[i] = (_Float16)lat[i];
        } }
      std::string derr;
      SharedBuffer rgb = vae->decode(zin, lh, lw, &derr);
      const std::size_t n = img.size();
      if (!rgb.empty() && rgb.byte_size() >= n * 2) {
        const auto* p = static_cast<const _Float16*>(rgb.contents());
        rgb_out->resize(n);
        for (std::size_t i = 0; i < n; ++i) { (*rgb_out)[i] = (float)p[i]; }
      }
    };

    std::vector<float> z_sel, rgb_sel, z_ref, rgb_ref;
    run(/*scalar=*/false, &z_sel, &rgb_sel);
    run(/*scalar=*/true, &z_ref, &rgb_ref);
    BOOGU_REQUIRE(!z_sel.empty() && z_sel.size() == z_ref.size());
    BOOGU_REQUIRE(!rgb_sel.empty() && rgb_sel.size() == rgb_ref.size());
    auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d; den += (double)b[i] * (double)b[i];
      }
      return den > 0.0 ? std::sqrt(num / den) : (num == 0.0 ? 0.0 : 1.0);
    };
    const double re = rel(z_sel, z_ref), rd = rel(rgb_sel, rgb_ref);
    std::printf("[boogu_smoke] selected mid-attn vs scalar: encode rel-L2 "
                "%.6g, decode rel-L2 %.6g (%d mid tokens)\n",
                re, rd, (lh / 2) * (lw / 2) * 4);
    EXPECT_TRUE(re < 0.02);
    EXPECT_TRUE(rd < 0.02);
  }
}
