// Open-H3-Regenerate-2K (Efficient-Large-Model/Open-H3-Regenerate-2K): the
// pieces it adds to MiniMax-H3 -- the latent upscaler, the LMS Sharp curve
// branch on the DiT's AdaLN, and the refine schedule.
//
// Goldens come from tools/dump_open_h3_regenerate_golden.py, which RESTATES
// the upscaler from its node's architecture over the release's own file
// and runs it on the CPU in fp32 (the truth) and with every op rounded to
// bf16 (the release's precision, the bar):
//
//   VPIPE_OPEN_H3_TEST_MODEL_PATH  the release directory (lora/,
//                                  latent_upsampler/, ...), with
//                                  lora/adaln_t_table.bin beside the curves
//   VPIPE_OPEN_H3_GOLDEN           the dumper's output (`upscaler`,
//                                  `curve` and `fastvae`)
//
// The fast decoder's test also needs VPIPE_MINIMAX_H3_TEST_MODEL_PATH (a
// MiniMax-H3 root or partition) for the released VAE's config: its latent
// statistics.
//
// The geometry tests need neither.

#include "minitest.h"
#include "residency-reserve.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/latent-upscale-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/vae-encode-stage.h"
#include "stages/model-memory.h"
#include "stages/model-memory-host.h"
#include "common/flex-bag.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "generative-models/minimax-h3/metal-minimax-h3-fast-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-latent-upscaler.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/quantize/safetensors-writer.h"
#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <atomic>
#include <random>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
namespace fs = std::filesystem;
namespace h3 = vpipe::genai::minimax_h3;

namespace {

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
from_bf16_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

metal_compute::SharedBuffer
bf16_buf_(MetalCompute* mc, const std::vector<float>& v)
{
  metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { d[i] = to_bf16_(v[i]); }
  return b;
}

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return {}; }
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> v((std::size_t)n / 4);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

FlexData
read_json_(const std::string& path)
{
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return FlexData::from_json(ss.str());
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr ? std::string(v) : std::string();
}

}  // namespace

// The node's TARGET_DIMENSIONS rule at the release's sizes: 1080 and
// 2160 rows are not whole 32-px alignments, so they compute at 1088 and
// 2176 (Python's round, half to even: 67.5 -> 68) and the caller crops.
TEST(minimax_h3_regenerate, upscaler_targets_follow_the_node)
{
  using U = MetalMiniMaxH3LatentUpscaler;
  struct Case { int h, w, wpx, hpx, H, W; };
  const Case cases[] = {
    {48, 84, 2560, 1440, 90, 160},    // 768P -> 2K, the release's default
    {48, 84, 1920, 1080, 68, 120},    // -> 1080P, computed at 1088
    {90, 160, 3840, 2160, 136, 240},  // 2K -> 4K, computed at 2176
    {36, 64, 1344, 768, 48, 84},      // a 576P draft -> 768P
    {48, 84, 1344, 768, 48, 84},      // same size: nothing to do
  };
  for (const Case& c : cases) {
    const U::Target t = U::target_for(c.h, c.w, c.wpx, c.hpx, 32);
    EXPECT_TRUE(t.h == c.H && t.w == c.W);
    if (t.h != c.H || t.w != c.W) {
      std::printf("  %dx%d -> %dx%d px: %dx%d, expected %dx%d\n", c.h, c.w,
                  c.wpx, c.hpx, t.h, t.w, c.H, c.W);
    }
  }
  // The embedding's scale is the mean of the two pixel ratios.
  const U::Target t = U::target_for(48, 84, 2560, 1440, 32);
  const double want = (2560.0 / (84 * 16) + 1440.0 / (48 * 16)) / 2.0;
  EXPECT_TRUE(std::fabs(t.scale - want) < 1e-12);
}

// Against the fp32 restatement on two geometries: a non-integer ratio
// (6 x 10 -> 10 x 18, scale 1.733), which exercises the bilinear
// weights, and an exact 2x. f16 activations land well inside the
// release's own bf16 distance.
TEST(minimax_h3_regenerate, upscaler_matches_reference)
{
  const std::string root = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  const std::string gold = env_("VPIPE_OPEN_H3_GOLDEN");
  if (root.empty() || gold.empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_GOLDEN unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  std::string err;
  const std::string file =
      MetalMiniMaxH3LatentUpscaler::resolve_weights(root, &err);
  ASSERT_TRUE(!file.empty());
  if (file.empty()) { std::printf("  %s\n", err.c_str()); return; }
  auto up = MetalMiniMaxH3LatentUpscaler::load(WeightSet::open(file, nullptr),
                                               mc, &err);
  ASSERT_TRUE(up != nullptr);
  if (!up) { std::printf("  load: %s\n", err.c_str()); return; }
  const auto& cfg = up->config();
  EXPECT_TRUE(cfg.channels == 512 && cfg.in_entries.size() == 18 &&
              cfg.out_entries.size() == 18 && cfg.temporal_kernel == 5);
  std::printf("  %d channels, %zu + %zu entries, %zu MB held, GEMMs on %s\n",
              cfg.channels, cfg.in_entries.size(), cfg.out_entries.size(),
              up->weight_bytes() >> 20,
              up->matrix_core_gemm() ? "matmul2d" : "steel");
  // What the plan books (off the headers) is what the load holds.
  EXPECT_TRUE(MetalMiniMaxH3LatentUpscaler::weight_bytes_for(file, -1) ==
              up->weight_bytes());

  const FlexData meta = read_json_(gold + "/meta.json");
  ASSERT_TRUE(meta.is_object());
  if (!meta.is_object()) { return; }
  FlexData cases = meta.as_object().at("cases");
  for (const FlexData& c : cases.as_array()) {
    FlexData cc = c;
    auto o = cc.as_object();
    const std::string name = std::string(o.at("name").as_string(""));
    const int T = (int)o.at("T").as_int(0), h = (int)o.at("h").as_int(0);
    const int w = (int)o.at("w").as_int(0), H = (int)o.at("H").as_int(0);
    const int W = (int)o.at("W").as_int(0);
    const double scale = o.at("scale").as_real(0.0);
    const double bar = o.at("bf16_rel_l2").as_real(0.0);
    const auto in  = read_f32_(gold + "/" + name + "_in.f32");
    const auto ref = read_f32_(gold + "/" + name + "_out.f32");
    ASSERT_TRUE(in.size() == (std::size_t)24 * T * h * w &&
                ref.size() == (std::size_t)24 * T * H * W);
    if (in.empty() || ref.empty()) { continue; }
    const auto tgt = MetalMiniMaxH3LatentUpscaler::target_for(
        h, w, (int)o.at("width_px").as_int(0),
        (int)o.at("height_px").as_int(0), 32);
    EXPECT_TRUE(tgt.h == H && tgt.w == W &&
                std::fabs(tgt.scale - scale) < 1e-9);
    std::vector<float> out;
    const bool ok = up->upscale(
        in.data(), T, h, w, H, W, scale,
        MetalMiniMaxH3LatentUpscaler::Normalize::kRelease, &out, &err);
    ASSERT_TRUE(ok);
    if (!ok) { std::printf("  upscale: %s\n", err.c_str()); continue; }
    const double d = rel_l2_(out, ref);
    // What the pass counted, against the shapes: 48 3x3x3 convs, 6 + 6
    // 1x1s and depthwise convs, the head and tail -- live taps only.
    const double C = 512, Z = 24;
    auto conv27 = [&](double rows, double cin, double cout) {
      // Interior frames read 3 taps, the two end frames 2.
      return 2.0 * rows * cout * 9 * cin * (3.0 * (T - 2) + 2 * 2) / T;
    };
    const double vin = (double)T * h * w, vout = (double)T * H * W;
    const double want =
        conv27(vin, Z, C) + 24 * conv27(vin, C, C) +
        6 * (2 * vin * C * C + 2 * vin * C * 5) + 24 * conv27(vout, C, C) +
        6 * (2 * vout * C * C + 2 * vout * C * 5) + conv27(vout, C, Z);
    const double got = (double)up->last_flops();
    std::printf("  %s: [24, %d, %d, %d] -> [.., %d, %d], scale %.4f: rel-L2 "
                "%.5f from fp32 (the release's bf16 %.5f); %.4f GFLOP "
                "counted, %.4f derived\n", name.c_str(), T, h, w, H, W,
                scale, d, bar, got * 1e-9, want * 1e-9);
    EXPECT_TRUE(d < 0.5 * bar);
    EXPECT_TRUE(std::fabs(got - want) <= 1e-6 * want);
    // The int8 tiers (matrix cores only): engaged, and within a GROSS
    // bar -- one a wrong layout or a dropped scale breaks (rel-L2 ~0.3 and
    // up); the accuracy claim is the GEMMs' own. MEASURED on the base M5,
    // frac / twox: accurate 0.0114 / 0.0128, fast 0.022 / 0.014, token
    // 0.012 / 0.017, token_plain 0.044 / 0.086 -- 48 convs of int8 error
    // through a GroupNorm stack, against the release's bf16 0.007 / 0.010.
    // These frames are 60 and 180 rows, under the accurate tier's
    // 1024-row floor (the held tiers lower their own), so that arm runs
    // with the floor lowered from the environment.
    if (!mc->supports_matrix_cores()) { continue; }
    const char* nm[4] = {"accurate", "fast", "token", "token_plain"};
    const double gross[4] = {0.03, 0.06, 0.06, 0.2};
    for (int t = 0; t <= 3; ++t) {
      if (t == 0) { ::setenv("VPIPE_I8_GEMM_MIN_M", "1", 1); }
      auto u8 = MetalMiniMaxH3LatentUpscaler::load(
          WeightSet::open(file, nullptr), mc, &err, t);
      if (t == 0) { ::unsetenv("VPIPE_I8_GEMM_MIN_M"); }
      ASSERT_TRUE(u8 != nullptr);
      if (!u8) { continue; }
      std::vector<float> o8;
      ASSERT_TRUE(u8->upscale(
          in.data(), T, h, w, H, W, scale,
          MetalMiniMaxH3LatentUpscaler::Normalize::kRelease, &o8, &err));
      const double d8 = rel_l2_(o8, ref);
      std::printf("    i8 %s: rel-L2 %.5f from fp32, %lld int8 GEMMs, %zu "
                  "MB held\n", nm[t], d8, u8->i8_gemms(),
                  u8->weight_bytes() >> 20);
      EXPECT_TRUE(u8->i8_tier() == t && u8->i8_gemms() > 0);
      EXPECT_TRUE(d8 < gross[t]);
      EXPECT_TRUE(MetalMiniMaxH3LatentUpscaler::weight_bytes_for(file, t) ==
                  u8->weight_bytes());
    }
  }
}

// The release's own geometry, timed: a 768P latent (37 x 48 x 84) to 2K
// (37 x 90 x 160). Opt-in (VPIPE_OPEN_H3_UPSCALE_BENCH=1, or =<frames>
// for a shorter clip); ~232 TFLOP at 37. VPIPE_OPEN_H3_UPSCALE_I8=<0..3>
// runs an int8 tier.
TEST(minimax_h3_regenerate, upscaler_bench_768p_to_2k)
{
  const std::string root = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (root.empty() || env_("VPIPE_OPEN_H3_UPSCALE_BENCH").empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_UPSCALE_BENCH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  std::string err;
  const std::string tier_s = env_("VPIPE_OPEN_H3_UPSCALE_I8");
  const int tier = tier_s.empty() ? -1 : std::atoi(tier_s.c_str());
  auto up = MetalMiniMaxH3LatentUpscaler::load(
      WeightSet::open(MetalMiniMaxH3LatentUpscaler::resolve_weights(root),
                      nullptr),
      mc, &err, tier);
  ASSERT_TRUE(up != nullptr);
  if (!up) { return; }
  const int nb = std::atoi(env_("VPIPE_OPEN_H3_UPSCALE_BENCH").c_str());
  const int T = nb > 1 ? nb : 37, h = 48, w = 84;
  const auto tgt = MetalMiniMaxH3LatentUpscaler::target_for(h, w, 2560, 1440);
  std::vector<float> in((std::size_t)24 * T * h * w);
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = (float)std::sin(0.37 * (double)i) * 0.9f;
  }
  const std::size_t booked = MetalMiniMaxH3LatentUpscaler::scratch_bytes(
      up->config(), T, h, w, tgt.h, tgt.w, up->i8_tier());
  for (int rep = 0; rep < 2; ++rep) {
    std::vector<float> out;
    // The pass's device allocation over the settled baseline, sampled
    // from a thread: what the plan's scratch claim has to cover.
    const std::size_t alloc0 = mc->memory_budget_settled().allocated;
    std::atomic<std::size_t> peak{alloc0};
    std::atomic<bool> done{false};
    std::thread sampler([&]() {
      while (!done.load()) {
        const std::size_t a = mc->memory_budget().allocated;
        if (a > peak.load()) { peak.store(a); }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = up->upscale(
        in.data(), T, h, w, tgt.h, tgt.w, tgt.scale,
        MetalMiniMaxH3LatentUpscaler::Normalize::kRelease, &out, &err);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    done.store(true);
    sampler.join();
    ASSERT_TRUE(ok);
    if (!ok) { std::printf("  %s\n", err.c_str()); return; }
    const std::size_t grew = peak.load() - alloc0;
    std::printf("  768P -> 2K, %d frames, i8 tier %d (%lld int8 GEMMs): "
                "%.2f s, %.1f TFLOP at %.2f TFLOP/s; peak allocation %zu "
                "MB, booked %zu MB (%zd bytes past it)\n", T,
                up->i8_tier(), up->i8_gemms(), s,
                (double)up->last_flops() * 1e-12,
                (double)up->last_flops() * 1e-12 / s, grew >> 20,
                booked >> 20, (std::ptrdiff_t)grew - (std::ptrdiff_t)booked);
    // A sampled peak can only under-read the true one; 2 MB covers the
    // pass's small per-call buffers and allocation rounding.
    EXPECT_TRUE(booked + (std::size_t(2) << 20) >= grew);
    EXPECT_TRUE((double)booked <= 1.25 * (double)grew);
  }
}


// c(t) read off the curve table as the reference reads it (torch.lerp),
// at the refine's timesteps, both ends and a knot.
TEST(minimax_h3_regenerate, curve_coords_match_reference)
{
  const std::string root = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  const std::string gold = env_("VPIPE_OPEN_H3_GOLDEN");
  if (root.empty() || gold.empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_GOLDEN unset\n");
    return;
  }
  const auto table = read_f32_(root + "/lora/adaln_t_table.bin");
  const auto t = read_f32_(gold + "/curve_t.f32");
  const auto want = read_f32_(gold + "/curve_coords.f32");
  ASSERT_TRUE(table.size() == 1025 * 8 && !t.empty() &&
              want.size() == t.size() * 8);
  if (want.size() != t.size() * 8 || table.size() != 1025 * 8) { return; }
  std::vector<float> got(t.size() * 8);
  MetalMiniMaxH3Transformer::curve_coords(table, 1025, t, got.data());
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, (double)std::fabs(got[i] - want[i]));
  }
  std::printf("  %zu timesteps: worst |c - ref| %.3e\n", t.size(), worst);
  EXPECT_TRUE(worst <= 1e-6);
}

// The curve add itself, against the host: mod[t, j] += s * wc[j] . c[t],
// summed in f32 and rounded once into the bf16 modulation.
TEST(minimax_h3_regenerate, curve_kernel_adds_its_delta)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("minimax_h3_regenerate_bf16");
  auto fn = lib.function("h3_adaln_curve_add_f16");
  ASSERT_TRUE(fn.valid());
  if (!fn.valid()) { return; }
  const int D = 1000, n_t = 3;
  const float scale = 0.75f;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> mod((std::size_t)n_t * D), wc((std::size_t)D * 8),
      cc((std::size_t)n_t * 8);
  for (auto& v : mod) { v = nd(rng); }
  for (auto& v : wc) { v = 0.1f * nd(rng); }
  for (auto& v : cc) { v = nd(rng); }
  auto mb = bf16_buf_(mc, mod);
  auto wb = mc->make_shared_buffer(wc.size() * 4);
  auto cb = mc->make_shared_buffer(cc.size() * 4);
  std::memcpy(wb.contents(), wc.data(), wc.size() * 4);
  std::memcpy(cb.contents(), cc.data(), cc.size() * 4);
  {
    auto st = mc->make_command_stream();
    {
      auto enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, mb);
      enc.set_buffer(1, wb);
      enc.set_buffer(2, cb);
      enc.set_constant(3, D);
      enc.set_constant(4, n_t);
      enc.set_constant(5, scale);
      enc.dispatch({(unsigned)D, (unsigned)n_t, 1}, {256, 1, 1});
    }
    ASSERT_TRUE(st.commit().wait_ok());
  }
  const auto* d = static_cast<const std::uint16_t*>(mb.contents());
  int off = 0;
  for (int t = 0; t < n_t; ++t) {
    for (int j = 0; j < D; ++j) {
      float dot = 0.0f;
      for (int k = 0; k < 8; ++k) {
        dot += wc[(std::size_t)j * 8 + k] * cc[(std::size_t)t * 8 + k];
      }
      const float base = from_bf16_(to_bf16_(mod[(std::size_t)t * D + j]));
      const float want = from_bf16_(to_bf16_(base + scale * dot));
      const float got = from_bf16_(d[(std::size_t)t * D + j]);
      // One bf16 rounding of the same f32 sum: equal, or an ulp apart
      // where the two f32 sums straddle a rounding boundary.
      if (got != want &&
          std::fabs(got - want) > std::fabs(want) * (1.0f / 128.0f)) {
        ++off;
      }
    }
  }
  EXPECT_TRUE(off == 0);
}

namespace {

// Two real blocks of the DiT through one forward, with the adapter specs
// given; the video velocity back as f32.
struct BlockRun {
  std::string root;
  MetalMiniMaxH3Transformer::Config cfg;
  h3::PackedLayout L;
  std::vector<float> uniq;
  std::vector<int> row_idx;
  metal_compute::SharedBuffer vb, ab, tb;
  int n_video = 0;

  bool
  setup(MetalCompute* mc, const std::string& r, int layers, int T = 2,
        int h = 12, int w = 20, int audio = 8)
  {
    root = r;
    std::string cerr;
    if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
      std::printf("  config: %s\n", cerr.c_str());
      return false;
    }
    cfg.n_layers = layers;
    const std::vector<int> tags(8, h3::kTextTag);
    if (!h3::build_packed_sequence(tags, T, h, w, audio, cfg.patch_h,
                                   cfg.patch_w, h3::kAudioChannels, {}, &L)) {
      return false;
    }
    // The refine's plan: the video at 1 - 0.909375, the audio clean.
    h3::build_row_timesteps(L, 0.090625f, 1.0f, 1.0f, &uniq, &row_idx);
    n_video = (int)L.video_indices.size();
    auto ramp = [](std::size_t n, float k) {
      std::vector<float> v(n);
      for (std::size_t i = 0; i < n; ++i) {
        v[i] = std::sin((float)i * k) * 0.5f;
      }
      return v;
    };
    vb = bf16_buf_(mc, ramp((std::size_t)n_video * cfg.video_patch_elems(),
                            0.017f));
    ab = bf16_buf_(mc, ramp((std::size_t)L.num_audio_rows *
                                cfg.audio_channels, 0.031f));
    tb = bf16_buf_(mc, ramp(tags.size() * (std::size_t)cfg.text_dim, 0.005f));
    return !vb.empty() && !ab.empty() && !tb.empty();
  }

  // `with`: the config to load with (the tiers), else `cfg`; `routes`:
  // the engine's GEMM routes, else steel's 32-row tile pinned.
  bool
  run(MetalCompute* mc,
      const std::vector<MetalMiniMaxH3Transformer::LoraSpec>& specs,
      std::vector<float>* out, int* mods, float turn_to = -1.0f,
      bool baked = false,
      const MetalMiniMaxH3Transformer::Config* with = nullptr,
      bool routes = false,
      std::unique_ptr<MetalMiniMaxH3Transformer>* keep = nullptr)
  {
    auto m = MetalMiniMaxH3Transformer::load(
        root, mc, with != nullptr ? *with : cfg, false, specs);
    if (m == nullptr) { return false; }
    if (!routes) {
      m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    }
    if (mods != nullptr) { *mods = m->lora_modules(); }
    if (turn_to >= 0.0f) { m->set_lora_scale(0, turn_to); }
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    // The denoise's route: the modulation baked for the schedule, the
    // curve added to the baked slice per forward.
    if (baked) {
      std::string berr;
      if (!m->bake_adaln({uniq}, &berr)) {
        std::printf("  bake: %s\n", berr.c_str());
        return false;
      }
      step.schedule_index = 0;
    }
    std::string ferr;
    const auto v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("  forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    if (keep != nullptr) { *keep = std::move(m); }
    return true;
  }

  // One more forward on a model run() kept; its wall clock in `ms`.
  bool
  forward_on(MetalMiniMaxH3Transformer& m, std::vector<float>* out,
             double* ms)
  {
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb;  step.audio = &ab;  step.text = &tb;
    step.layout = &L;  step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    const auto t0 = std::chrono::steady_clock::now();
    const auto v = m.forward(step, &ferr);
    *ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t0).count();
    if (v.empty()) {
      std::printf("  forward: %s\n", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)n_video * cfg.video_patch_elems();
    out->resize(n);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16_(p[i]); }
    return true;
  }
};

}  // namespace

// Curve branches through two real blocks: the release's file (every B
// zero) binds nothing and is the base forward to the bit; a TRAINED
// branch -- a synthetic one, written here over the release's table --
// moves the velocity at strength 1, is the base forward at 0, and an
// adapter loaded at 0 and turned up is the one loaded at 1.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (a MiniMax-H3 root or partition)
// + VPIPE_OPEN_H3_TEST_MODEL_PATH.
TEST(minimax_h3_regenerate, curve_branches_through_real_blocks)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2));
  if (br.vb.empty()) { return; }
  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  std::vector<float> base, released;
  int m_base = -1, m_rel = -1;
  ASSERT_TRUE(br.run(mc, {}, &base, &m_base));
  ASSERT_TRUE(br.run(mc, {Spec{rel + "/lora/sharp_curve_branches.safetensors",
                               1.0f}}, &released, &m_rel));
  EXPECT_TRUE(m_rel == 0);
  EXPECT_TRUE(released == base);

  // A trained branch on both blocks: A [4, 8], B [adaln_out, 4].
  const fs::path dir = fs::temp_directory_path() / "vpipe-h3-curve-test";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  {
    SafetensorsWriter wr(dir.string());
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int D = br.cfg.adaln_out(), r = 4;
    for (int L = 0; L < 2; ++L) {
      std::vector<float> A((std::size_t)r * 8), B((std::size_t)D * r);
      for (auto& v : A) { v = 0.3f * nd(rng); }
      for (auto& v : B) { v = 0.05f * nd(rng); }
      const std::string pre =
          "transformer.transformer_blocks." + std::to_string(L) +
          ".adaln_proj.linear.";
      ASSERT_TRUE(wr.add(pre + "curve_A.weight", "F32", {r, 8}, A.data(),
                         A.size() * 4));
      ASSERT_TRUE(wr.add(pre + "curve_B.weight", "F32", {D, r}, B.data(),
                         B.size() * 4));
    }
    ASSERT_TRUE(wr.close());
  }
  fs::copy_file(rel + "/lora/adaln_t_table.bin", dir / "adaln_t_table.bin",
                fs::copy_options::overwrite_existing, ec);
  std::string file;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (e.path().extension() == ".safetensors") { file = e.path().string(); }
  }
  ASSERT_TRUE(!file.empty());
  std::vector<float> at0, at1, turned;
  int m0 = -1, m1 = -1, mt = -1;
  ASSERT_TRUE(br.run(mc, {Spec{file, 0.0f}}, &at0, &m0));
  ASSERT_TRUE(br.run(mc, {Spec{file, 1.0f}}, &at1, &m1));
  ASSERT_TRUE(br.run(mc, {Spec{file, 0.0f}}, &turned, &mt, 1.0f));
  // The same through the baked modulation: the base baked, and the curve
  // on top of it, which must move it as the unbaked path does.
  std::vector<float> base_b, at1_b;
  ASSERT_TRUE(br.run(mc, {}, &base_b, nullptr, -1.0f, true));
  ASSERT_TRUE(br.run(mc, {Spec{file, 1.0f}}, &at1_b, nullptr, -1.0f, true));
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");
  const double moved = rel_l2_(at1, base);
  const double moved_b = rel_l2_(at1_b, base_b);
  const double paths = rel_l2_(at1_b, at1);
  std::printf("  baked: the curve moves the base %.3e; baked against "
              "unbaked at 1: %.3e\n", moved_b, paths);
  EXPECT_TRUE(moved_b > 0.5 * moved && moved_b < 2.0 * moved);
  EXPECT_TRUE(paths < 0.1 * moved);
  std::printf("  release's file: %d branches bound, forward %s the base; "
              "a trained branch: %d bound, scale 0 %s the base, scale 1 "
              "moves %.3e, turned up %s loaded at 1\n", m_rel,
              released == base ? "IS" : "is NOT", m1,
              at0 == base ? "IS" : "is NOT", moved,
              turned == at1 ? "IS" : "is NOT");
  EXPECT_TRUE(m0 == 2 && m1 == 2);
  EXPECT_TRUE(at0 == base);
  EXPECT_TRUE(moved > 1e-3);
  EXPECT_TRUE(turned == at1);
  fs::remove_all(dir, ec);
}

// The release's merged LoRA (refiner + Turbo + Sharp, rank 576 on every
// projection, alpha == rank) binds through the diffusers decomposition,
// every block banded, is the base forward at 0 and moves it at 1.
TEST(minimax_h3_regenerate, merged_lora_binds_banded)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2));
  if (br.vb.empty()) { return; }
  ::setenv("VPIPE_H3_NO_FUSED_FF", "1", 1);
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  const std::string lp = rel + "/lora/open_h3_regenerate_2k_lora.safetensors";
  std::vector<float> base, at0, at1;
  int mb = -1, m0 = -1, m1 = -1;
  ASSERT_TRUE(br.run(mc, {}, &base, &mb));
  ASSERT_TRUE(br.run(mc, {Spec{lp, 0.0f}}, &at0, &m0));
  ASSERT_TRUE(br.run(mc, {Spec{lp, 1.0f}}, &at1, &m1));
  ::unsetenv("VPIPE_H3_NO_FUSED_FF");
  const double moved = rel_l2_(at1, base);
  std::printf("  %d modules bound over 2 blocks + the refiner; scale 0 %s "
              "the base, scale 1 moves %.3e\n", m1,
              at0 == base ? "IS" : "is NOT", moved);
  EXPECT_TRUE(m1 > 0 && m0 == m1);
  EXPECT_TRUE(at0 == base);
  EXPECT_TRUE(moved > 1e-3);

  // What the plan books for the file -- lora_footprint, off its header --
  // is what a load holds: every factor preloaded; the refiner's and the
  // largest block's set when the blocks stream (the rest spilled); and
  // the scratch's rank, q|k|v fused.
  ::unsetenv("VPIPE_H3_LORA_RESIDENT");
  const auto fp = MetalMiniMaxH3Transformer::lora_footprint(lp, 2);
  const auto all = MetalMiniMaxH3Transformer::lora_footprint(lp);
  auto pre = MetalMiniMaxH3Transformer::load(h3root, mc, br.cfg, false,
                                             {Spec{lp, 1.0f}});
  ASSERT_TRUE(pre != nullptr);
  if (pre == nullptr) { return; }
  const std::size_t held_pre = pre->lora_bytes();
  const int rank = pre->lora_scratch_rank();
  pre.reset();
  auto str = MetalMiniMaxH3Transformer::load(h3root, mc, br.cfg, true,
                                             {Spec{lp, 1.0f}});
  ASSERT_TRUE(str != nullptr);
  if (str == nullptr) { return; }
  const std::size_t held_str = str->lora_bytes();
  str.reset();
  std::printf("  booked / held over 2 blocks: preloaded %zu / %zu, streamed "
              "%zu / %zu bytes, rank %d / %d; the whole file books %zu MB "
              "preloaded, %zu MB streamed, rank %d\n", fp.whole, held_pre,
              fp.streamed, held_str, fp.rank, rank, all.whole >> 20,
              all.streamed >> 20, all.rank);
  EXPECT_TRUE(fp.whole == held_pre);
  EXPECT_TRUE(fp.streamed == held_str);
  EXPECT_TRUE(held_str < held_pre);
  EXPECT_TRUE(fp.rank == rank);
}

// THE BANDED FORWARD against the whole one: two real blocks under the
// release's merged LoRA at 8 x 32 x 64 latents (4,096 video rows, 4,120
// with the prompt and the soundtrack) in bands of at most 1,024 --
// balanced, four of 896 and a last of 536, so both banded pipelines run
// and the last clears the matrix-core floor (cut at 1,024 it was 24 rows,
// and its adapter GEMMs ran steel where the whole sequence's ran
// matmul2d: 2.4e-3 off on an M5). The velocity must be the whole
// forward's to the bit with steel's 32-row GEMM tile pinned AND on the
// engine's own routes (a band takes the whole sequence's tuned route).
// The band must be TAKEN, and the banded scratch is what scratch_bytes
// books for it.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_OPEN_H3_TEST_MODEL_PATH.
TEST(minimax_h3_regenerate, banded_forward_is_the_whole_one)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2, 8, 32, 64));
  if (br.vb.empty()) { return; }
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  const std::string lp = rel + "/lora/open_h3_regenerate_2k_lora.safetensors";
  struct Arm {
    std::vector<float> v;
    int band = -1;
    std::size_t resident = 0, booked = 0;
  };
  auto arm = [&](const char* band, bool routes, Arm* a,
                 const MetalMiniMaxH3Transformer::Config* with = nullptr) {
    ::setenv("VPIPE_H3_BAND_ROWS", band, 1);
    std::unique_ptr<MetalMiniMaxH3Transformer> keep;
    const bool ok = br.run(mc, {Spec{lp, 1.0f}}, &a->v, nullptr, -1.0f,
                           false, with, routes, &keep);
    ::unsetenv("VPIPE_H3_BAND_ROWS");
    if (!ok || !keep) { return false; }
    a->band = keep->last_band_rows();
    a->resident = keep->scratch_resident_bytes();
    a->booked = MetalMiniMaxH3Transformer::scratch_bytes(
        br.cfg, br.L.seq_len, 8, (int)br.uniq.size(), false,
        keep->ff_scratch_narrow(), 0, a->band);
    return true;
  };
  Arm whole, banded, whole_r, banded_r;
  ASSERT_TRUE(arm("0", false, &whole));
  ASSERT_TRUE(arm("1024", false, &banded));
  ASSERT_TRUE(arm("0", true, &whole_r));
  ASSERT_TRUE(arm("1024", true, &banded_r));
  std::vector<float> base;
  ASSERT_TRUE(br.run(mc, {}, &base, nullptr));
  const double moved = rel_l2_(whole.v, base);
  const double d_pin = rel_l2_(banded.v, whole.v);
  const double d_eng = rel_l2_(banded_r.v, whole_r.v);
  std::printf("  %d rows in bands of %d: pinned tile %s (rel-L2 %.3e), the "
              "engine's routes %.3e, against the adapter's own %.3e; "
              "scratch %zu MB banded (booked %zu) against %zu whole "
              "(booked %zu)\n", br.L.seq_len, banded.band,
              banded.v == whole.v ? "IDENTICAL" : "differs", d_pin, d_eng,
              moved, banded.resident >> 20, banded.booked >> 20,
              whole.resident >> 20, whole.booked >> 20);
  const int last = br.L.seq_len - (br.L.seq_len - 1) / banded.band *
                                       banded.band;
  std::printf("  bands of %d, the last %d\n", banded.band, last);
  EXPECT_TRUE(whole.band == 0 && banded.band > 0 && banded.band <= 1024);
  EXPECT_TRUE(last >= 512);
  EXPECT_TRUE(banded.v == whole.v);
  EXPECT_TRUE(banded_r.v == whole_r.v);
  EXPECT_TRUE(d_eng == 0.0 && moved > 0.1);
  EXPECT_TRUE(banded.resident == banded.booked);
  EXPECT_TRUE(whole.resident == whole.booked);
  EXPECT_TRUE(banded.resident < whole.resident);

  // THE ATTENTION TIERS BAND TOO, each the same tier run whole to the
  // bit: Sol (a second Sol for the 24-row last band), Sage (K quantized
  // once a layer, each band's Q), Sage's int8 P*V (band 0's prompt and
  // soundtrack rows on the int8 QK alone, as the whole sequence's), Sol
  // with Sage on its exact half. Each must also MOVE the dense forward,
  // or the arm compared two dense runs. Sage needs matrix cores.
  struct Tier { const char* name; bool sol, sage, pv8; };
  const bool mx = mc->supports_matrix_cores();
  for (const Tier& t : {Tier{"sol", true, false, false},
                        Tier{"sage", false, true, false},
                        Tier{"sage + int8 P*V", false, true, true},
                        Tier{"sol + sage", true, true, false}}) {
    if (t.sage && !mx) {
      std::printf("  %s: SKIP (no matrix cores)\n", t.name);
      continue;
    }
    MetalMiniMaxH3Transformer::Config tc = br.cfg;
    tc.sol.enabled = t.sol;
    tc.sol.dense_layers = 0;
    tc.sage.enabled = t.sage;
    tc.sage.dense_layers = 0;
    tc.sage.pv8 = t.pv8;
    Arm tw, tb;
    ASSERT_TRUE(arm("0", false, &tw, &tc));
    ASSERT_TRUE(arm("1024", false, &tb, &tc));
    const double moved_t = rel_l2_(tw.v, whole.v);
    std::printf("  %s: banded %s whole (rel-L2 %.3e), band %d; the tier "
                "moves the dense forward %.3e\n", t.name,
                tb.v == tw.v ? "IS" : "is NOT", rel_l2_(tb.v, tw.v), tb.band,
                moved_t);
    EXPECT_TRUE(tw.band == 0 && tb.band == banded.band);
    EXPECT_TRUE(tb.v == tw.v);
    EXPECT_TRUE(moved_t > 0.0);
  }
}

// The banded forward's cost at a real size: two real blocks under the
// merged LoRA, one model, forwards alternating whole and banded (the
// default band) with the order reversed each round, after one warm-up of
// each. Opt-in: VPIPE_OPEN_H3_BAND_BENCH="T,h,w" (or 1: the 39-frame 2K
// clip, 12 x 90 x 160 latents, 43k rows); VPIPE_OPEN_H3_BAND_BENCH_TIER
// = sol | sage | pv8 (Sage's int8 P*V) | sol+sage runs that attention
// tier in both arms (Sage needs matrix cores).
TEST(minimax_h3_regenerate, banded_forward_bench)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  const std::string geo = env_("VPIPE_OPEN_H3_BAND_BENCH");
  if (h3root.empty() || rel.empty() || geo.empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_BAND_BENCH unset\n");
    return;
  }
  int T = 12, h = 90, w = 160;
  if (geo != "1") { std::sscanf(geo.c_str(), "%d,%d,%d", &T, &h, &w); }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2, T, h, w, T * 4 * 40 / 24 + 1));
  if (br.vb.empty()) { return; }
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  const std::string lp = rel + "/lora/open_h3_regenerate_2k_lora.safetensors";
  const std::string tier = env_("VPIPE_OPEN_H3_BAND_BENCH_TIER");
  MetalMiniMaxH3Transformer::Config tc = br.cfg;
  tc.sol.enabled = tier.find("sol") != std::string::npos;
  tc.sage.enabled = tier.find("sage") != std::string::npos || tier == "pv8";
  tc.sage.pv8 = tier == "pv8";
  tc.sol.dense_layers = 0;
  tc.sage.dense_layers = 0;
  std::unique_ptr<MetalMiniMaxH3Transformer> m;
  std::vector<float> v;
  ASSERT_TRUE(br.run(mc, {Spec{lp, 1.0f}}, &v, nullptr, -1.0f, false,
                     &tc, true, &m));
  if (!m) { return; }
  const int band = MetalMiniMaxH3Transformer::band_rows(br.cfg,
                                                        br.L.seq_len);
  std::printf("  %d rows; the default band %d (the last %d); attention "
              "tier: %s\n", br.L.seq_len, band,
              band > 0 ? br.L.seq_len - (br.L.seq_len - 1) / band * band
                       : 0,
              tier.empty() ? "dense" : tier.c_str());
  std::vector<float> ref;
  double ms_w = 0.0, ms_b = 0.0, t = 0.0;
  std::size_t res_w = 0, res_b = 0;
  bool same = true;
  for (int round = 0; round < 3; ++round) {
    for (int k = 0; k < 2; ++k) {
      const bool banded = ((round + k) % 2) == 1;
      if (banded) { ::unsetenv("VPIPE_H3_BAND_ROWS"); }
      else        { ::setenv("VPIPE_H3_BAND_ROWS", "0", 1); }
      ASSERT_TRUE(br.forward_on(*m, &v, &t));
      (banded ? res_b : res_w) = m->scratch_resident_bytes();
      if (round == 0) { continue; }       // warm-up: each arm's first
      (banded ? ms_b : ms_w) += t;
      if (ref.empty()) { ref = v; } else { same = same && v == ref; }
    }
  }
  ::unsetenv("VPIPE_H3_BAND_ROWS");
  std::printf("  whole %.0f ms, banded %.0f ms a 2-block forward (%.3fx); "
              "scratch %zu MB whole, %zu MB banded; the velocity %s\n",
              ms_w / 2, ms_b / 2, ms_w / ms_b, res_w >> 20, res_b >> 20,
              same ? "IDENTICAL" : "DIFFERS");
  EXPECT_TRUE(band > 0 && res_b < res_w);
}

// The refine under every tier the box has, with the release's merged LoRA
// on: two real blocks at 8 latent frames x 32 x 64 (4096 video rows --
// past i8_gemm's 1024-row floor and the ANE's 2048-row chunk, which a
// smaller geometry would leave untouched). Against the engine's own
// dense forward with the adapter: on an M4 the ANE taking the feed-
// forward's and the qkv's rows (the adapter merged into the staged
// weights), on an M5 every i8 tier (the adapter on top of the int8
// route). Each must ENGAGE and land far closer to the dense forward than
// the adapter moves it.
TEST(minimax_h3_regenerate, refine_tiers_take_the_merged_lora)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2, 8, 32, 64, 33));
  if (br.vb.empty()) { return; }
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  const std::string lp = rel + "/lora/open_h3_regenerate_2k_lora.safetensors";
  std::vector<float> base, dense;
  ASSERT_TRUE(br.run(mc, {}, &base, nullptr, -1.0f, false, nullptr, true));
  ASSERT_TRUE(br.run(mc, {Spec{lp, 1.0f}}, &dense, nullptr, -1.0f, false,
                     nullptr, true));
  const double moved = rel_l2_(dense, base);
  std::printf("  %d video rows; the adapter moves the dense forward "
              "%.3e\n", br.n_video, moved);
  EXPECT_TRUE(moved > 1e-2);
  auto arm = [&](const char* name, MetalMiniMaxH3Transformer::Config c,
                 double bar) {
    std::vector<float> got;
    std::unique_ptr<MetalMiniMaxH3Transformer> m;
    ASSERT_TRUE(br.run(mc, {Spec{lp, 1.0f}}, &got, nullptr, -1.0f, false,
                       &c, true, &m));
    if (!m) { return; }
    const double d = rel_l2_(got, dense);
    std::printf("  %s: %.3e from the dense forward (%.2f of the adapter's "
                "move); int8 layout %s, ANE %s%s\n", name, d, d / moved,
                m->i8_layout() ? "held" : "no",
                m->ane_armed() ? "armed" : "off",
                m->ane_gpu_only() ? " (lost rows: GPU only)" : "");
    if (c.ane_ffn) {
      EXPECT_TRUE(m->ane_armed() && !m->ane_gpu_only());
    }
    if (c.i8_gemm && c.i8_tier >= 1) { EXPECT_TRUE(m->i8_layout()); }
    if (c.i8_gemm && c.i8_tier == 0) { EXPECT_TRUE(got != dense); }
    EXPECT_TRUE(d < bar * moved);
  };
  if (!mc->supports_matrix_cores()) {
    MetalMiniMaxH3Transformer::Config c = br.cfg;
    c.ane_ffn = true;
    arm("ane_ffn", c, 0.25);
    c.ane_qkv = true;
    arm("ane_ffn + ane_qkv", c, 0.25);
    std::printf("  (no matrix cores: the i8 tiers do not run here)\n");
  } else {
    const char* nm[4] = {"i8 accurate", "i8 fast", "i8 token",
                         "i8 token_plain"};
    for (int t = 0; t <= 3; ++t) {
      MetalMiniMaxH3Transformer::Config c = br.cfg;
      c.i8_gemm = true;
      c.i8_tier = t;
      arm(nm[t], c, t == 3 ? 0.6 : 0.3);
    }
  }
}

// STREAM, THEN RETAIN under the merged LoRA: two real blocks streamed,
// each block's factors spilled at load, and residency given room for
// both. A kept block takes its factors back, so the run converges on
// what a preload holds -- and books it: residency's bytes grow by
// exactly the factors it brought back (they were not counted, ~124 MB a
// block of the full merge), the DiT's own holding is the one it has
// without the adapter (the factors are the adapter claim's, which books
// all of them), the factors held are that claim's figure to the byte
// (the streamed block's set goes with the slot pair), and a forward on
// the kept blocks is the preloaded model's to the bit.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH + VPIPE_OPEN_H3_TEST_MODEL_PATH.
TEST(minimax_h3_regenerate, kept_blocks_keep_their_factors_booked)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  ::unsetenv("VPIPE_H3_LORA_RESIDENT");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  BlockRun br;
  ASSERT_TRUE(br.setup(mc, h3root, 2, 4, 16, 32));
  if (br.vb.empty()) { return; }
  using Spec = MetalMiniMaxH3Transformer::LoraSpec;
  const std::string lp = rel + "/lora/open_h3_regenerate_2k_lora.safetensors";
  struct Out {
    std::vector<float> v;
    int kept = -1;
    std::size_t resid = 0, holding = 0;
    std::size_t lora_before = 0, lora_mid = 0, lora_after = 0;
  };
  auto streamed = [&](const std::vector<Spec>& specs, Out* o) {
    auto m = MetalMiniMaxH3Transformer::load(br.root, mc, br.cfg, true,
                                             specs);
    if (m == nullptr) { return false; }
    m->set_gemm_route(MetalMiniMaxH3Transformer::GemmRoute::kSteelBm32);
    m->set_residency_reserve(test_residency_reserve(2ull << 30));
    o->lora_before = m->lora_bytes();
    double ms = 0.0;
    // The first forward keeps both blocks (their factors with them); the
    // second, every block kept, gives the slot pair and the factors'
    // streamed set back at its top.
    for (int i = 0; i < 2; ++i) {
      if (!br.forward_on(*m, &o->v, &ms)) { return false; }
      if (i == 0) { o->lora_mid = m->lora_bytes(); }
    }
    o->kept = m->resident_block_count();
    o->resid = m->resident_block_bytes();
    o->holding = m->streamed_holding_bytes();
    o->lora_after = m->lora_bytes();
    return true;
  };
  Out bare, lora;
  ASSERT_TRUE(streamed({}, &bare));
  ASSERT_TRUE(streamed({Spec{lp, 1.0f}}, &lora));
  std::vector<float> pre;
  ASSERT_TRUE(br.run(mc, {Spec{lp, 1.0f}}, &pre, nullptr));
  const auto fp = MetalMiniMaxH3Transformer::lora_footprint(lp, 2);
  const std::size_t kept_factors = lora.lora_mid - lora.lora_before;
  const std::size_t stream_set = lora.lora_mid - lora.lora_after;
  std::printf("  kept %d / %d blocks; residency %zu MB bare, %zu MB with "
              "the adapter (+%zu MB; the factors kept %zu MB); the DiT's "
              "holding %zu / %zu MB; factors held %zu MB of %zu booked "
              "(the streamed set, %zu MB, given back); the velocity %s "
              "the resident model's\n", bare.kept, lora.kept,
              bare.resid >> 20, lora.resid >> 20,
              (lora.resid - bare.resid) >> 20, kept_factors >> 20,
              bare.holding >> 20, lora.holding >> 20,
              lora.lora_after >> 20, fp.whole >> 20, stream_set >> 20,
              lora.v == pre ? "IS" : "is NOT");
  EXPECT_TRUE(bare.kept == 2 && lora.kept == 2);
  EXPECT_TRUE(kept_factors > 0 && stream_set > 0);
  EXPECT_TRUE(lora.resid - bare.resid == kept_factors);
  EXPECT_TRUE(lora.holding == bare.holding);
  EXPECT_TRUE(lora.lora_after == fp.whole);
  EXPECT_TRUE(lora.v == pre);
}

// The refine's memory plan, off the stage before anything loads: the
// merged LoRA booked at every factor a streamed DiT's kept blocks
// converge on, streamable to the streamed block's set (its header's
// footprint -- not the 12.45 GB F32 file, and not nothing, which is what
// it had been), the audio VAE that encodes the held silence claimed in
// the denoise, and no soundtrack decode, since a refine writes none. The
// same stage with init_latent unwired books a soundtrack and no audio
// VAE. Configs and headers only.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH (FL2VA, or a root holding it)
// + VPIPE_OPEN_H3_TEST_MODEL_PATH.
TEST(minimax_h3_regenerate, refine_plan_books_what_it_holds)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string rel = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (h3root.empty() || rel.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  ::unsetenv("VPIPE_PRELOAD");
  ::unsetenv("VPIPE_H3_STREAM");
  Session sess;
  const std::string lp = rel + "/lora/open_h3_regenerate_2k_lora.safetensors";
  auto cfg = [&](const std::string& dir) {
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(dir));
    c.as_object().insert("width", FlexData::make_int(2560));
    c.as_object().insert("height", FlexData::make_int(1440));
    c.as_object().insert("frames", FlexData::make_int(121));
    c.as_object().insert("fps", FlexData::make_real(24.0));
    return c;
  };
  FlexData mcfg = FlexData::make_object();
  mcfg.as_object().insert("lora", FlexData::make_string(lp));
  // Any upstream vertex on iport 11 (init_latent) makes it a refine.
  LatentUpscaleStage up(&sess, "up", std::vector<InEdge>{},
                        cfg(rel + "/latent_upsampler"));
  std::vector<InEdge> wired(12, InEdge{nullptr, 0});
  wired[11] = InEdge{&up, 0};

  struct Seen {
    std::int64_t lora_held = -1;
    std::size_t lora_floor = 0;
    bool audio_vae = false, audio_decode = false;
    std::string audio_phase;
  };
  const std::string adir =
      genai::MetalMiniMaxH3AudioVae::resolve_vae_dir(h3root);
  auto scan = [&](const std::vector<InEdge>& edges) {
    GenerateVideoStage gen(&sess, "gen", edges, cfg(h3root));
    gen.apply_constant(9, mcfg);
    Seen s;
    for (const ResourceClaim& c : gen.declare_resources()) {
      if (c.key == lp) {
        s.lora_held = bag::integer(c.extra, model_memory::kClaimHeldBytes, 0);
        s.lora_floor = c.floor_bytes;
      }
      if (!adir.empty() && c.key == adir) {
        s.audio_vae = true;
        s.audio_phase = c.phase;
      }
      if (c.key.rfind("gen-audio-decode|", 0) == 0) { s.audio_decode = true; }
    }
    return s;
  };
  const Seen r = scan(wired);
  const Seen g = scan({});
  MetalMiniMaxH3Transformer::Config tc;
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(h3root, tc,
                                                          nullptr));
  const auto fp = MetalMiniMaxH3Transformer::lora_footprint(lp, tc.n_layers);
  std::printf("  the LoRA booked %lld MB (floor %zu MB; every factor %zu "
              "MB); "
              "refine: audio VAE %s in '%s', soundtrack decode %s; "
              "generation: audio VAE %s, soundtrack decode %s\n",
              (long long)(r.lora_held >> 20), r.lora_floor >> 20,
              fp.whole >> 20, r.audio_vae ? "claimed" : "NOT claimed",
              r.audio_phase.c_str(), r.audio_decode ? "BOOKED" : "not booked",
              g.audio_vae ? "CLAIMED" : "not claimed",
              g.audio_decode ? "booked" : "NOT booked");
  EXPECT_TRUE(!adir.empty());
  EXPECT_TRUE(fp.streamed > 0 && fp.streamed < fp.whole / 10);
  // Stream, then retain: a kept block keeps its factors, so the adapter
  // is booked at every factor -- what a box with room converges on --
  // streamable to the streamed block's set.
  EXPECT_TRUE(r.lora_held == (std::int64_t)fp.whole);
  EXPECT_TRUE(r.lora_floor == fp.streamed);
  EXPECT_TRUE(g.lora_held == (std::int64_t)fp.whole);
  EXPECT_TRUE(r.audio_vae && r.audio_phase == model_memory::kPhaseDenoise);
  EXPECT_TRUE(!r.audio_decode);
  EXPECT_TRUE(!g.audio_vae);
  EXPECT_TRUE(g.audio_decode);
  // The preload A/B holds the same.
  ::setenv("VPIPE_PRELOAD", "1", 1);
  const Seen p = scan(wired);
  ::unsetenv("VPIPE_PRELOAD");
  EXPECT_TRUE(p.lora_held == (std::int64_t)fp.whole);
  // A DiT let go after each clip (the shipped graphs' `always`) is phased
  // to the denoise, and its adapters with it: the model owns them.
  FlexData ac = cfg(h3root);
  ac.as_object().insert("unload_when_idle", FlexData::make_string("always"));
  GenerateVideoStage dg(&sess, "gen", wired, ac);
  dg.apply_constant(9, mcfg);
  std::string lphase = "(none)";
  for (const ResourceClaim& c : dg.decide_resources()) {
    if (c.key == lp) { lphase = c.phase; }
  }
  std::printf("  under `always` the adapter is phased to '%s'\n",
              lphase.c_str());
  EXPECT_TRUE(lphase == model_memory::kPhaseDenoise);
}

// The draft's encoder at what it holds: MiniMax-H3's video VAE builds
// each half at its first use, and vae-encode only encodes -- so its claim
// is the encoder half (0.36 GB, not the 9.9 GB F32 file), phased to the
// CONDITION when the stage lets go after each beat (`destroy`, the
// regenerate graph's), and left unphased under `auto`. Configs only.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH.
TEST(minimax_h3_regenerate, the_draft_encoder_books_its_half)
{
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (h3root.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  auto stage = [&](const char* unload) {
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(h3root));
    c.as_object().insert("unload_when_idle", FlexData::make_string(unload));
    return std::make_unique<VaeEncodeStage>(&sess, "enc",
                                            std::vector<InEdge>{}, c);
  };
  using Half = MetalMiniMaxH3VideoVae::Half;
  const std::string vdir = MetalMiniMaxH3VideoVae::resolve_vae_dir(h3root);
  const std::size_t enc = MetalMiniMaxH3VideoVae::held_bytes(vdir,
                                                             Half::kEncoder);
  for (const char* unload : {"destroy", "auto"}) {
    auto st = stage(unload);
    std::int64_t held = -1;
    std::string key;
    for (const ResourceClaim& c : st->declare_resources()) {
      key = c.key;
      held = bag::integer(c.extra, model_memory::kClaimHeldBytes, 0);
    }
    std::string phase = "(unphased)";
    for (const ResourceClaim& c : st->decide_resources()) {
      if (c.key == key) { phase = c.phase; }
    }
    std::printf("  %s: claims '%s' at %lld MB (encoder half %zu MB, file "
                "%zu MB), %s\n", unload, key.c_str(),
                (long long)(held >> 20), enc >> 20,
                model_memory::dir_weights_bytes(key) >> 20, phase.c_str());
    EXPECT_TRUE(enc > 0 && held == (std::int64_t)enc);
    EXPECT_TRUE(std::string(unload) == "destroy"
                    ? phase == model_memory::kPhaseCondition
                    : phase == "(unphased)");
  }
}

// The refine's sigmas, as open_h3/sigmas.py has them (its own test's
// numbers): the release's 3-step grid literally, the 2-step one and the
// shift-6 rule from any start; and the timesteps and Euler steps the
// scheduler takes from them.
TEST(minimax_h3_regenerate, refine_sigmas_follow_the_release)
{
  using S = MiniMaxH3Scheduler;
  const auto three = S::refine_sigmas(3);
  ASSERT_TRUE(three.size() == 4);
  EXPECT_TRUE(three[0] == 0.909375 && three[1] == 6.0 / 7.0 &&
              three[2] == 2.0 / 3.0 && three[3] == 0.0);
  const auto two = S::refine_sigmas(2);
  ASSERT_TRUE(two.size() == 3);
  EXPECT_TRUE(two[0] == 0.7 && std::fabs(two[1] - 0.494118) < 1e-6 &&
              two[2] == 0.0);
  // shifted_grid(0.909375, 4): the refiner's original 4-step grid.
  const auto four = S::refine_sigmas(4, 0.909375);
  const double want4[] = {0.909375, 0.841446, 0.732075, 0.526697, 0.0};
  ASSERT_TRUE(four.size() == 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(std::fabs(four[(std::size_t)i] - want4[i]) < 1e-6);
  }
  // start 0.8 (the ladder's), 3 steps.
  const auto ladder = S::refine_sigmas(3, 0.8);
  EXPECT_TRUE(ladder.size() == 4 && ladder[0] == 0.8 && ladder[3] == 0.0 &&
              ladder[1] < 0.8 && ladder[2] < ladder[1]);
  EXPECT_TRUE(S::refine_sigmas(0).empty() &&
              S::refine_sigmas(3, 1.5).empty());

  // Timesteps 1 - sigma in double, rounded once; Euler as the reference
  // steps it, x + float(s_i - s_i+1) * v.
  S sch;
  ASSERT_TRUE(sch.set_sigmas_as_run(three));
  ASSERT_TRUE(sch.num_inference_steps() == 3);
  EXPECT_TRUE(sch.timesteps()[0] == (float)(1.0 - 0.909375) &&
              sch.timesteps()[1] == (float)(1.0 - 6.0 / 7.0));
  std::vector<float> x = {1.0f, -2.0f, 0.5f}, v = {0.25f, 1.0f, -3.0f};
  const std::vector<float> x0 = x;
  ASSERT_TRUE(sch.step_plain(v.data(), 1, x.data(), x.size()));
  const float d = (float)(6.0 / 7.0 - 2.0 / 3.0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    // Two roundings, as torch's mul then add; contracted to an FMA the
    // comparison would round once.
    volatile float dv = d * v[i];
    EXPECT_TRUE(x[i] == x0[i] + dv);
  }
  EXPECT_TRUE(!sch.set_sigmas_as_run({0.9, 0.95, 0.0}) &&
              !sch.set_sigmas_as_run({0.9, 0.5}));
}

namespace {

class LatentSource : public TypedStage<LatentSource> {
public:
  static constexpr const char* kTypeName = "ut-h3regen-source";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    auto p = make_payload<TensorBeatPayload>(TensorBeat(beats[next++]));
    co_await ctx.write(0, std::move(p));
    co_return;
  }
};

class LatentSink : public TypedStage<LatentSink> {
public:
  static constexpr const char* kTypeName = "ut-h3regen-sink";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> got;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      got.push_back(TensorBeat(*tb));
    }
    co_return;
  }
};

// One latent through a `latent-upscale` stage; the beats out.
std::vector<TensorBeat>
upscale_through_stage_(Session& sess, const TensorBeat& in, FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<LatentSource>(
      &sess, "src", std::vector<InEdge>{}, FlexData::make_object());
  src_u->beats.push_back(in);
  auto* src = static_cast<LatentSource*>(pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  auto* st = pl->insert_stage(std::make_unique<LatentUpscaleStage>(
      &sess, "up", std::vector<InEdge>{{src, 0}}, std::move(cfg)));
  auto* sink = static_cast<LatentSink*>(pl->insert_stage(
      std::make_unique<LatentSink>(&sess, "sink",
                                   std::vector<InEdge>{{st, 0}},
                                   FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return sink->got;
}

}  // namespace

// The stage: the node's target rule (152 rows compute at 160, and the
// latent carries a centred crop box for vae-decode), the input's sideband
// carried, the values the class gives; and a target the latent already
// has passes it through untouched.
TEST(minimax_h3_regenerate, latent_upscale_stage_crops_and_carries)
{
  const std::string root = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const int T = 2, h = 6, w = 10;
  TensorBeat in;
  in.dtype = TensorBeat::DType::F32;
  in.shape = {24, T, h, w};
  in.resize_contiguous((std::size_t)24 * T * h * w);
  for (std::size_t i = 0; i < in.element_count(); ++i) {
    in.as_f32()[i] = 0.8f * (float)std::sin(0.11 * (double)i);
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign(sideband::kFrames, FlexData::make_int(5));
  sb.as_object().insert_or_assign(sideband::kFps, FlexData::make_real(24.0));
  in.sideband = sb;

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  cfg.as_object().insert_or_assign("width", FlexData::make_int(288));
  cfg.as_object().insert_or_assign("height", FlexData::make_int(152));
  const auto out = upscale_through_stage_(sess, in, cfg);
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  const TensorBeat& o = out[0];
  ASSERT_TRUE(o.shape.size() == 4 && o.shape[2] == 10 && o.shape[3] == 18);
  FlexData osb = o.sideband;
  ASSERT_TRUE(osb.is_object());
  auto oo = osb.as_object();
  EXPECT_TRUE(oo.contains(sideband::kFrames) &&
              oo.at(sideband::kFrames).as_int(0) == 5);
  ASSERT_TRUE(oo.contains(sideband::kCropBox));
  FlexData box = oo.at(sideband::kCropBox);
  ASSERT_TRUE(box.is_array() && box.as_array().size() == 4);
  int b[4];
  for (int i = 0; i < 4; ++i) {
    b[i] = (int)box.as_array()[(std::size_t)i].as_int(-1);
  }
  std::printf("  [24, %d, %d, %d] -> [24, %d, %d, %d], crop box [%d, %d, "
              "%d, %d]\n", T, h, w, (int)o.shape[1], (int)o.shape[2],
              (int)o.shape[3], b[0], b[1], b[2], b[3]);
  EXPECT_TRUE(b[0] == 0 && b[1] == 4 && b[2] == 288 && b[3] == 156);

  // The stage's values are the class's.
  std::string err;
  auto up = MetalMiniMaxH3LatentUpscaler::load(
      WeightSet::open(MetalMiniMaxH3LatentUpscaler::resolve_weights(root),
                      nullptr),
      mc, &err);
  ASSERT_TRUE(up != nullptr);
  if (up) {
    const auto t = MetalMiniMaxH3LatentUpscaler::target_for(h, w, 288, 152);
    std::vector<float> want;
    ASSERT_TRUE(up->upscale(in.as_f32(), T, h, w, t.h, t.w, t.scale,
                            MetalMiniMaxH3LatentUpscaler::Normalize::kRelease,
                            &want, &err));
    const std::vector<float> got(o.as_f32(), o.as_f32() + o.element_count());
    EXPECT_TRUE(got == want);
  }

  // Already the target's size: through, byte for byte, with no box.
  FlexData same = FlexData::make_object();
  same.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  same.as_object().insert_or_assign("width", FlexData::make_int(160));
  same.as_object().insert_or_assign("height", FlexData::make_int(96));
  const auto thru = upscale_through_stage_(sess, in, same);
  ASSERT_TRUE(thru.size() == 1);
  if (thru.size() == 1) {
    EXPECT_TRUE(thru[0].shape == in.shape &&
                std::memcmp(thru[0].as_f32(), in.as_f32(),
                            in.element_count() * 4) == 0);
    FlexData tsb = thru[0].sideband;
    EXPECT_TRUE(!tsb.as_object().contains(sideband::kCropBox));
  }
}


// The FAST decoder (vae_fast/) against the dumper's fp32 restatement at
// two geometries chosen to cut windows at every edge -- 7 latents (a full
// block and a 2-latent one) at 18 x 34, and 12 (two blocks and a part) at
// 17 x 20, where two blocks finish in the shifted grid's last temporal
// window. The bar is the release's own bf16 (its autocast restated:
// operands and outputs of every GEMM rounded), the FLOPs are counted ==
// derived from the windows (grouped here by brute force, not by the
// class's tables), and the frame count is the released decoder's.
//
// Arms beside the default: the ANE taking a share of every feed-forward
// (ane_ffn; fp16, lossy) where the box has one, and every i8 tier where it
// has matrix cores -- each must ENGAGE and stay within a gross bar.
TEST(minimax_h3_regenerate, fast_vae_matches_reference)
{
  const std::string root = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  const std::string gold = env_("VPIPE_OPEN_H3_GOLDEN");
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root.empty() || gold.empty() || h3root.empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_TEST_MODEL_PATH / "
                "VPIPE_OPEN_H3_GOLDEN / VPIPE_MINIMAX_H3_TEST_MODEL_PATH "
                "unset\n");
    return;
  }
  const FlexData meta = read_json_(gold + "/fastvae.json");
  if (!meta.is_object()) {
    std::printf("  SKIP: no fastvae.json (run the dumper's `fastvae`)\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  std::string err;
  const std::string file = MetalMiniMaxH3FastVae::resolve_weights(root, &err);
  ASSERT_TRUE(!file.empty());
  if (file.empty()) { std::printf("  %s\n", err.c_str()); return; }
  auto ws = WeightSet::open(file, nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  const std::string vae_dir = MetalMiniMaxH3VideoVae::resolve_vae_dir(h3root);
  MetalMiniMaxH3FastVae::Config base;
  const bool cfg_ok =
      MetalMiniMaxH3FastVae::config_from(*ws, vae_dir, &base, &err);
  ASSERT_TRUE(cfg_ok);
  if (!cfg_ok) { std::printf("  config: %s\n", err.c_str()); return; }
  EXPECT_TRUE(base.n_layers == 24 && base.dim == 2048 &&
              base.ffn_inner == 8192);

  struct Case {
    std::string name;
    int T, H, W, frames;
    double bar;
    std::vector<float> z, ref;
  };
  std::vector<Case> cases;
  {
    FlexData cs = meta.as_object().at("cases");
    for (const FlexData& c : cs.as_array()) {
      FlexData cc = c;
      auto o = cc.as_object();
      Case k;
      k.name = std::string(o.at("name").as_string(""));
      k.T = (int)o.at("T").as_int(0);
      k.H = (int)o.at("H").as_int(0);
      k.W = (int)o.at("W").as_int(0);
      k.frames = (int)o.at("frames").as_int(0);
      k.bar = o.at("bf16_rel_l2").as_real(0.0);
      k.z = read_f32_(gold + "/fastvae_" + k.name + "_z.f32");
      k.ref = read_f32_(gold + "/fastvae_" + k.name + "_out.f32");
      cases.push_back(std::move(k));
    }
  }
  ASSERT_TRUE(cases.size() == 2);

  // The work, derived: every GEMM at its rows, every attention pair.
  auto derived_flops = [&](const MetalMiniMaxH3FastVae::Config& c,
                           const Case& k) {
    const double D = c.dim, F = c.ffn_inner, S = c.global_tokens();
    const int tb = c.block, ws2 = c.window;
    const double N = (double)k.T * k.H * k.W;
    const int nb = (k.T + tb - 1) / tb;
    const double G = nb * S;
    const double qkv = 2.0 * 3 * D * D, out = 2.0 * D * D,
                 ff = 2.0 * (2 * F * D + D * F);
    double f = 2.0 * N * D * c.z_channels;
    for (int L = 0; L < c.n_layers; ++L) {
      const bool last = L == c.n_layers - 1;
      const int s = (L & 1) ? ws2 / 2 : 0, st = (L & 1) ? c.t_shift : 0;
      f += (N + G) * qkv + (last ? N : N + G) * (out + ff);
      // Windows by brute force: a token's window from its padded place.
      std::map<long long, long long> count;
      std::map<long long, int> ng;
      for (int t = 0; t < k.T; ++t) {
        for (int y = 0; y < k.H; ++y) {
          for (int x = 0; x < k.W; ++x) {
            const long long id =
                (((long long)(t + st) / tb) * 1000 + (y + s) / ws2) * 1000 +
                (x + s) / ws2;
            ++count[id];
            const int j = (t + st) / tb;
            const int lo = std::max(j * tb - st, 0);
            const int hi = std::min((j + 1) * tb - st, k.T) - 1;
            ng[id] = (lo / tb != hi / tb) ? 2 : 1;
          }
        }
      }
      double pairs = 0.0;
      for (const auto& [id, n] : count) {
        pairs += (double)n * ((double)n + ng[id] * S);
      }
      if (!last) {
        for (int b = 0; b < nb; ++b) {
          const int nt = std::min((b + 1) * tb, k.T) - b * tb;
          pairs += S * ((double)nt * k.H * k.W + S);
        }
      }
      f += 4.0 * c.n_heads * pairs * c.head_dim;
    }
    f += 2.0 * N * D * c.patch_elems();
    return f;
  };

  auto run = [&](const char* arm, MetalMiniMaxH3FastVae::Config cfg,
                 double bar_mult, bool check_flops) {
    auto vae = MetalMiniMaxH3FastVae::load(ws, mc, cfg, &err);
    ASSERT_TRUE(vae != nullptr);
    if (!vae) { std::printf("  %s load: %s\n", arm, err.c_str()); return; }
    // Booked (off the headers, at the tier that engaged) == held.
    const std::size_t booked =
        MetalMiniMaxH3FastVae::weight_bytes(file, vae->i8_tier());
    if (booked != vae->weight_bytes()) {
      std::printf("  %s: weights booked %zu, held %zu\n", arm, booked,
                  vae->weight_bytes());
    }
    EXPECT_TRUE(booked == vae->weight_bytes());
    for (const Case& k : cases) {
      ASSERT_TRUE(k.z.size() == (std::size_t)24 * k.T * k.H * k.W);
      if (k.z.empty()) { continue; }
      EXPECT_TRUE(vae->decoded_frames(k.T) == k.frames);
      const std::uint64_t f0 = vae->gpu_flops().flushed();
      metal_compute::SharedBuffer px =
          vae->decode(k.z.data(), k.T, k.H, k.W, &err);
      ASSERT_TRUE(!px.empty());
      if (px.empty()) { std::printf("  decode: %s\n", err.c_str()); return; }
      const std::size_t n = (std::size_t)3 * k.frames * k.H * 16 * k.W * 16;
      ASSERT_TRUE(k.ref.size() == n && px.byte_size() >= n * 2);
      if (k.ref.size() != n) { return; }
      std::vector<float> got(n);
      const auto* p = static_cast<const std::uint16_t*>(px.contents());
      for (std::size_t i = 0; i < n; ++i) { got[i] = from_bf16_(p[i]); }
      const double d = rel_l2_(got, k.ref);
      const double counted = (double)(vae->gpu_flops().flushed() - f0);
      const double want = derived_flops(vae->config(), k);
      std::printf("  %s %s: [24, %d, %d, %d] -> %d frames: rel-L2 %.5f "
                  "from fp32 (the release's bf16 %.5f); GEMMs %s%s, "
                  "attention %s; i8 %lld / dense %lld GEMMs, ANE %lld rows; "
                  "%.3f GFLOP counted, %.3f derived\n", arm, k.name.c_str(),
                  k.T, k.H, k.W, k.frames, d, k.bar,
                  vae->matrix_cores() ? "matmul2d" : "steel",
                  vae->i8_tier() >= 0 ? " + int8" : "",
                  vae->nax_attention() ? "NAX" : "steel", vae->i8_gemms(),
                  vae->dense_gemms(), vae->ane_rows_taken(), counted * 1e-9,
                  want * 1e-9);
      EXPECT_TRUE(d < bar_mult * k.bar);
      if (check_flops) {
        EXPECT_TRUE(std::fabs(counted - want) <= 1e-6 * want);
      }
      if (cfg.i8_tier >= 0) { EXPECT_TRUE(vae->i8_gemms() > 0); }
      if (cfg.ane_ffn) { EXPECT_TRUE(vae->ane_rows_taken() > 0); }
    }
  };
  run("bf16", base, 1.0, true);
  if (!mc->supports_matrix_cores()) {
    MetalMiniMaxH3FastVae::Config a = base;
    a.ane_ffn = true;
    run("ane_ffn", a, 2.0, false);
    std::printf("  (no matrix cores: the i8 tiers do not run here)\n");
  } else {
    for (int t = 0; t <= 3; ++t) {
      MetalMiniMaxH3FastVae::Config a = base;
      a.i8_tier = t;
      const char* nm[4] = {"i8 accurate", "i8 fast", "i8 token",
                           "i8 token_plain"};
      run(nm[t], a, t == 3 ? 12.0 : 6.0, false);
    }
  }
}

// The fast decoder at a real geometry, timed, its device allocation held
// against scratch_bytes. Opt-in: VPIPE_OPEN_H3_FVAE_BENCH="T,h,w" (or 1:
// the 39-frame 2K clip, 12 x 90 x 160); VPIPE_OPEN_H3_FVAE_ANE=1 and
// VPIPE_OPEN_H3_FVAE_I8=<0..3> pick the arm.
TEST(minimax_h3_regenerate, fast_vae_bench)
{
  const std::string root = env_("VPIPE_OPEN_H3_TEST_MODEL_PATH");
  const std::string h3root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string geo = env_("VPIPE_OPEN_H3_FVAE_BENCH");
  if (root.empty() || h3root.empty() || geo.empty()) {
    std::printf("  SKIP: VPIPE_OPEN_H3_FVAE_BENCH unset\n");
    return;
  }
  int T = 12, h = 90, w = 160;
  if (geo != "1") { std::sscanf(geo.c_str(), "%d,%d,%d", &T, &h, &w); }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  std::string err;
  // VPIPE_OPEN_H3_FVAE_RELEASED=1: the RELEASED decoder at the same
  // geometry instead -- its time and its device high water beside the
  // weights, which the stage books only as a floor (its output).
  if (!env_("VPIPE_OPEN_H3_FVAE_RELEASED").empty()) {
    const std::string vdir = MetalMiniMaxH3VideoVae::resolve_vae_dir(h3root);
    MetalMiniMaxH3VideoVae::Config vc;
    ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(vdir, vc, &err));
    const std::size_t a_load = mc->memory_budget_settled().allocated;
    auto rel = MetalMiniMaxH3VideoVae::load(WeightSet::open(vdir, nullptr),
                                            mc, vc);
    ASSERT_TRUE(rel != nullptr);
    if (!rel) { return; }
    // The decoder half loads at its first decode: a small one first, so
    // the peak below is the decode's alone.
    {
      const int wt = vc.tokens_per_chunk() + vc.token_drop;
      metal_compute::SharedBuffer wz = bf16_buf_(
          mc, std::vector<float>((std::size_t)24 * wt * 4 * 4, 0.0f));
      int wf = 0;
      ASSERT_TRUE(!rel->decode_video(wz, wt, 4, 4, &wf, &err).empty());
    }
    const std::size_t a0 = mc->memory_budget_settled().allocated;
    std::printf("  released decoder: %zu MB allocated by the load and a "
                "first (small) decode\n", (a0 - a_load) >> 20);
    std::vector<float> zf((std::size_t)24 * T * h * w);
    for (std::size_t i = 0; i < zf.size(); ++i) {
      zf[i] = (float)std::sin(0.37 * (double)i) * 0.9f;
    }
    const metal_compute::SharedBuffer zb = bf16_buf_(mc, zf);
    const std::size_t a1 = mc->memory_budget_settled().allocated;
    std::atomic<std::size_t> peak{a1};
    std::atomic<bool> done{false};
    std::thread sampler([&]() {
      while (!done.load()) {
        const std::size_t a = mc->memory_budget().allocated;
        if (a > peak.load()) { peak.store(a); }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
    int F = 0;
    const auto t0 = std::chrono::steady_clock::now();
    metal_compute::SharedBuffer px = rel->decode_video(zb, T, h, w, &F, &err);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    done.store(true);
    sampler.join();
    ASSERT_TRUE(!px.empty());
    const std::size_t out = (std::size_t)3 * F * h * 16 * w * 16 * 2;
    const std::size_t booked =
        MetalMiniMaxH3VideoVae::decode_video_bytes(vc, T, h, w);
    const std::size_t grew = peak.load() - a1;
    std::printf("  released [24, %d, %d, %d] -> %d frames %dx%d: %.1f s; "
                "peak allocation %zu MB beside the weights (booked %zu), "
                "of which the bf16 output is %zu MB\n", T, h, w, F, w * 16,
                h * 16, s, grew >> 20, booked >> 20, out >> 20);
    EXPECT_TRUE(grew <= booked + (std::size_t(16) << 20));
    return;
  }
  auto ws = WeightSet::open(MetalMiniMaxH3FastVae::resolve_weights(root),
                            nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  MetalMiniMaxH3FastVae::Config cfg;
  ASSERT_TRUE(MetalMiniMaxH3FastVae::config_from(
      *ws, MetalMiniMaxH3VideoVae::resolve_vae_dir(h3root), &cfg, &err));
  cfg.ane_ffn = !env_("VPIPE_OPEN_H3_FVAE_ANE").empty();
  if (!env_("VPIPE_OPEN_H3_FVAE_I8").empty()) {
    cfg.i8_tier = std::atoi(env_("VPIPE_OPEN_H3_FVAE_I8").c_str());
  }
  const auto t_l0 = std::chrono::steady_clock::now();
  auto vae = MetalMiniMaxH3FastVae::load(ws, mc, cfg, &err);
  ASSERT_TRUE(vae != nullptr);
  if (!vae) { std::printf("  load: %s\n", err.c_str()); return; }
  std::printf("  loaded in %.1f s, %zu MB held\n",
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t_l0).count(),
              vae->weight_bytes() >> 20);
  std::vector<float> z((std::size_t)24 * T * h * w);
  for (std::size_t i = 0; i < z.size(); ++i) {
    z[i] = (float)std::sin(0.37 * (double)i) * 0.9f;
  }
  MetalMiniMaxH3FastVae::Config eff = vae->config();
  eff.i8_tier = vae->i8_tier();
  // The ANE module is the plan's other claim (CoreML units), its host
  // rows and staging made with the device's allocator.
  const std::size_t booked =
      MetalMiniMaxH3FastVae::scratch_bytes(eff, T, h, w) +
      (cfg.ane_ffn ? MetalMiniMaxH3FastVae::ane_runtime_bytes(eff) : 0);
  for (int rep = 0; rep < 2; ++rep) {
    const std::size_t alloc0 = mc->memory_budget_settled().allocated;
    std::atomic<std::size_t> peak{alloc0};
    std::atomic<bool> done{false};
    std::thread sampler([&]() {
      while (!done.load()) {
        const std::size_t a = mc->memory_budget().allocated;
        if (a > peak.load()) { peak.store(a); }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
    const std::uint64_t f0 = vae->gpu_flops().flushed();
    const auto t0 = std::chrono::steady_clock::now();
    metal_compute::SharedBuffer px = vae->decode(z.data(), T, h, w, &err);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    done.store(true);
    sampler.join();
    ASSERT_TRUE(!px.empty());
    if (px.empty()) { std::printf("  %s\n", err.c_str()); return; }
    const double tf = (double)(vae->gpu_flops().flushed() - f0) * 1e-12;
    const std::size_t grew = peak.load() - alloc0;
    std::printf("  [24, %d, %d, %d] -> %d frames %dx%d: %.2f s, %.1f TFLOP "
                "on the GPU at %.2f TFLOP/s (ANE %lld rows, i8 %lld GEMMs); "
                "peak allocation %zu MB, booked %zu MB\n", T, h, w,
                vae->decoded_frames(T), w * 16, h * 16, s, tf, tf / s,
                vae->ane_rows_taken(), vae->i8_gemms(), grew >> 20,
                booked >> 20);
    EXPECT_TRUE(booked + (std::size_t(8) << 20) >= grew);
  }
}
