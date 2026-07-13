// FLUX.2-klein bring-up smoke + golden (both the 4B and 9B sizes -- the whole
// flux2 stack is config-driven off config.json, so ONE set of tests serves
// both). Three tiers:
//   * flux2_smoke.*  -- plumbing: the model classes load + a forward produces a
//     finite output of the expected shape (no reference).
//   * flux2_golden.* -- numerical rel-L2 vs the diffusers golden
//     (dump_flux2_golden.py): DiT velocity, embedded-guidance velocity (when the
//     model has a guidance_embedder), and VAE decode.
//   * flux2_e2e.*    -- the full text-to-image -> vae-decode pipeline produces a
//     coherent image (opt-in, heavy: loads the encoder + DiT together).
//
// Env: VPIPE_FLUX2_TEST_MODEL_PATH = the FLUX.2-klein model root (4B or 9B).
// VPIPE_FLUX2_GOLDEN = the golden dir (for flux2_golden.*). VPIPE_FLUX2_E2E set
// = run the end-to-end pipeline test. Unset => the relevant tier skips.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/flux2/metal-flux2-calibration.h"
#include "generative-models/flux2/metal-flux2-transformer.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/text-to-image-stage.h"
#include "stages/vae-decode-stage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// True if every element is finite (no NaN/Inf).
bool
all_finite_(const SharedBuffer& b, std::size_t n)
{
  if (b.empty() || b.byte_size() < n * 2) { return false; }
  const auto* p = static_cast<const _Float16*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite((float)p[i])) { return false; }
  }
  return true;
}

std::vector<float>
read_f32_file_(const std::string& path)
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

// True if every bf16 (uint16) element is finite.
bool
all_finite_bf16_(const SharedBuffer& b, std::size_t n)
{
  if (b.empty() || b.byte_size() < n * 2) { return false; }
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t u = (std::uint32_t)p[i] << 16;
    float f; std::memcpy(&f, &u, 4);
    if (!std::isfinite(f)) { return false; }
  }
  return true;
}

// rel-L2 = ||got - ref||_2 / ||ref||_2.
double
rel_l2_(const float* got, const float* ref, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Test-only source: emits one prompt string then ends.
class SourceText : public TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-source-text-flux2";
  SourceText(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::string prompt;
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      co_await ctx.write(
          0, std::make_unique<FlexDataPayload>(FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};

// Test-only sink: captures the beats it receives.
class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-flux2";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// Test-only source: emits one f32 [C,H,W] TensorBeat then ends (a stand-in for
// a vae-encode reference latent on a ref-latent iport).
class SourceTensor : public TypedStage<SourceTensor> {
public:
  static constexpr const char* kTypeName = "ut-source-tensor-flux2";
  SourceTensor(const SessionContextIntf* s, std::string id,
               std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<float> chw;
  int C = 0, H = 0, W = 0;
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {C, H, W};
      tb->resize_contiguous(chw.size());
      std::memcpy(tb->as_f32(), chw.data(), chw.size() * sizeof(float));
      co_await ctx.write(0, std::move(tb));
    }
    ctx.signal_done();
    co_return;
  }
};

}  // namespace

// The VAE loads and decodes a small random latent into a finite RGB image of
// the expected size.
TEST(flux2_smoke, vae_decode_shape_finite)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  MetalFlux2Vae::Config cfg;
  auto vae = MetalFlux2Vae::load(vdir, mc, cfg);
  ASSERT_TRUE(vae != nullptr);

  int gh = 4, gw = 4;                     // -> 64x64 image
  // Large-resolution decode is env-selectable so the per-level command-buffer
  // split (see MetalFlux2Vae::decode) can be exercised (VPIPE_VAE_GRID=64 ->
  // 1024^2, which OOMs as a single command buffer).
  if (const char* g = std::getenv("VPIPE_VAE_GRID")) {
    const int v = std::atoi(g);
    if (v >= 4) { gh = gw = v; }
  }
  const int C = vae->config().dit_channels();   // 128
  SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
  ASSERT_TRUE(!z.empty());
  {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
      d[i] = (_Float16)nd(rng);
    }
  }
  std::string err;
  SharedBuffer rgb = vae->decode(z, gh, gw, &err);
  if (rgb.empty()) {
    std::printf("[flux2_smoke] VAE decode FAILED: %s\n", err.c_str());
  }
  ASSERT_TRUE(!rgb.empty());
  const int H = gh * 16, W = gw * 16;
  ASSERT_TRUE(rgb.byte_size() >= (std::size_t)3 * H * W * 2);
  EXPECT_TRUE(all_finite_(rgb, (std::size_t)3 * H * W));
  std::printf("[flux2_smoke] VAE decoded [3, %d, %d] (finite)\n", H, W);
}

// M5 matrix-core matmul2d A/B for the VAE conv/1x1 GEMMs (mirrors
// krea2_vae): decode the same deterministic random latent with the default
// (matmul2d) route and with VPIPE_FLUX2_NO_MMA2 forcing steel, rel-L2 the two
// images, and report both wall-clocks. On a non-matrix-core GPU both runs
// take steel and the rel-L2 is ~0.
TEST(flux2_smoke, vae_decode_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  const int gh = 32, gw = 32;                   // -> 512x512 image
  std::printf("[flux2_smoke] vae_decode_mma_matches_steel: matrix_cores=%d\n",
              (int)mc->supports_matrix_cores());

  auto run = [&](bool no_mma, double* ms) {
    // The steel leg must ALSO disable the NAX hardware conv, or the
    // "steel" VAE would still take the conv2d op for its 3x3s.
    if (no_mma) {
      ::setenv("VPIPE_FLUX2_NO_MMA2", "1", 1);
      ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1);
    } else {
      ::unsetenv("VPIPE_FLUX2_NO_MMA2");
      ::unsetenv("VPIPE_VAE_NO_HWCONV");
    }
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    ::unsetenv("VPIPE_FLUX2_NO_MMA2");
    ::unsetenv("VPIPE_VAE_NO_HWCONV");
    std::vector<float> out;
    if (vae == nullptr) { return out; }
    const int C = vae->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    if (z.empty()) { return out; }
    std::mt19937 rng(1234);                     // same latent both runs
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
      d[i] = (_Float16)nd(rng);
    }
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer rgb = vae->decode(z, gh, gw, &err);
    *ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const std::size_t n = (std::size_t)3 * (gh * 16) * (gw * 16);
    if (rgb.empty() || rgb.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(rgb.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };

  double ms_mma = 0.0, ms_steel = 0.0;
  const std::vector<float> v_mma   = run(/*no_mma=*/false, &ms_mma);
  const std::vector<float> v_steel = run(/*no_mma=*/true, &ms_steel);
  ASSERT_TRUE(!v_mma.empty());
  ASSERT_TRUE(v_mma.size() == v_steel.size());
  const double r = rel_l2_(v_mma.data(), v_steel.data(), v_mma.size());
  std::printf("[flux2_smoke] VAE decode mma-vs-steel rel-L2 = %.6g "
              "(512x512; mma %.0f ms, steel %.0f ms)\n", r, ms_mma, ms_steel);
  EXPECT_TRUE(r < 3e-2);
}

// Regression for the >=1024px matmul2d corruption (see the Krea-2 VAE and
// MetalFlux2Vae::conv_gemm_bias_): the MPP matmul2d op returns garbage for
// output rows past M ~= 2^19, so a 1024px decode (M = H*W = 2^20) went grey from
// image row ~512 down. _mma_max_m splits those large-M GEMMs into row-chunks.
// Decode a deterministic random latent at 1024x1024 through the DEFAULT (hwconv
// + mma2) path and assert every horizontal band keeps real signal (a collapsed /
// grey region drops the per-band std from ~0.5 to ~0.01) and no output is
// non-finite. Env-gated on the model like the other VAE tests.
TEST(flux2_smoke, vae_decode_1024_no_matmul2d_corruption)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
  ASSERT_TRUE(vae != nullptr);
  const int gh = 64, gw = 64;                   // -> 1024x1024 image
  const int C = vae->config().dit_channels();
  SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
  ASSERT_TRUE(!z.empty());
  {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
      d[i] = (_Float16)nd(rng);
    }
  }
  std::string err;
  SharedBuffer rgb = vae->decode(z, gh, gw, &err);
  ASSERT_TRUE(!rgb.empty());
  const int H = gh * 16, W = gw * 16;

  // Per-band std of channel 0 across 8 horizontal bands: a corrupt region
  // collapses to a near-constant grey (std ~0.01), a healthy decode stays high.
  const auto* a = static_cast<const _Float16*>(rgb.contents());
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
  std::printf("[flux2_smoke] 1024 decode: min band std = %.4f, non-finite = %d\n",
              min_std, nonfinite);
  EXPECT_TRUE(nonfinite == 0);
  EXPECT_TRUE(min_std > 0.05);         // >> the ~0.01 of a collapsed grey band
}

// The mid-block self-attention on the matrix-core FULL flash kernel
// (sdpa_full_mma2_d512) must match the scalar sdpa_full_f16 it replaces. Decode
// the same latent both ways (VPIPE_FLUX2_NO_MMA_ATTN forces scalar) and rel-L2.
TEST(flux2_smoke, vae_decode_flash_attn_matches_scalar)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }
  const int gh = 32, gw = 32;                        // 512x512 -> 1024 mid tokens
  auto decode_rgb = [&](bool scalar) {
    if (scalar) { ::setenv("VPIPE_FLUX2_NO_MMA_ATTN", "1", 1); }
    auto m = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    ::unsetenv("VPIPE_FLUX2_NO_MMA_ATTN");
    std::vector<float> out;
    if (m == nullptr) { return out; }
    const int C = m->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
      d[i] = (_Float16)nd(rng);
    }
    std::string err;
    SharedBuffer o = m->decode(z, gh, gw, &err);
    if (o.empty()) { return out; }
    const std::size_t n = (std::size_t)3 * (gh * 16) * (gw * 16);
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };
  const std::vector<float> flash = decode_rgb(false);
  const std::vector<float> scal  = decode_rgb(true);
  ASSERT_TRUE(!flash.empty() && flash.size() == scal.size());
  const double r = rel_l2_(flash.data(), scal.data(), flash.size());
  std::printf("[flux2_smoke] flash-attn vs scalar rel-L2 = %.6g (512x512)\n", r);
  EXPECT_TRUE(r < 3e-2);
}

// Decode wall-clock at 512 and 1024 (default hwconv + matmul2d path). Warm 2x,
// best-of-5 min (least DVFS-throttled). VPIPE_FLUX2_TEST_MODEL_PATH gated.
TEST(flux2_smoke, vae_decode_bench)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }
  auto m = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
  ASSERT_TRUE(m != nullptr);
  const int C = m->config().dit_channels();
  for (int side : {32, 64}) {                        // 512, 1024
    const std::size_t hw = (std::size_t)side * side;
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * hw * 2);
    std::uint32_t s = 0x51ced00du;
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * hw; ++i) {
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
    std::printf("[flux2_smoke] decode %dx%d: %.1f ms (peak est %llu MB)\n",
                side * 16, side * 16, best,
                (unsigned long long)(m->decode_peak_bytes(side, side) >> 20));
  }
}

// NAX hardware conv A/B for the ENCODER (stride-2 downsample convs +
// stride-1): encode the same deterministic random image with the hw conv
// (default on matrix-core GPUs) and with VPIPE_VAE_NO_HWCONV forcing
// im2col + matmul2d, rel-L2 the latents and report both wall-clocks.
TEST(flux2_smoke, vae_encode_hwconv_matches_im2col)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  const int H = 256, W = 256;                   // -> 16x16 latent grid
  auto run = [&](bool no_hw, double* ms) {
    if (no_hw) { ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1); }
    else       { ::unsetenv("VPIPE_VAE_NO_HWCONV"); }
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{},
                                   /*with_encoder=*/true);
    ::unsetenv("VPIPE_VAE_NO_HWCONV");
    std::vector<float> out;
    if (vae == nullptr || !vae->has_encoder()) { return out; }
    SharedBuffer img = mc->make_shared_buffer((std::size_t)3 * H * W * 2);
    if (img.empty()) { return out; }
    std::mt19937 rng(31415);                    // same image both runs
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto* p = static_cast<_Float16*>(img.contents());
    for (std::size_t i = 0; i < (std::size_t)3 * H * W; ++i) {
      p[i] = (_Float16)d(rng);
    }
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer z = vae->encode(img, H, W);
    *ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (z.empty()) { return out; }
    const std::size_t n = z.byte_size() / 2;
    out.resize(n);
    const auto* zp = static_cast<const _Float16*>(z.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)zp[i]; }
    return out;
  };

  double ms_hw = 0.0, ms_i2c = 0.0;
  const std::vector<float> v_hw = run(false, &ms_hw);
  if (v_hw.empty()) {
    std::printf("[flux2_smoke] no encoder in this checkpoint -- skip\n");
    return;
  }
  const std::vector<float> v_i2c = run(true, &ms_i2c);
  ASSERT_TRUE(v_hw.size() == v_i2c.size());
  const double r = rel_l2_(v_hw.data(), v_i2c.data(), v_hw.size());
  std::printf("[flux2_smoke] VAE encode hwconv-vs-im2col rel-L2 = %.6g "
              "(%dx%d; hw %.0f ms, im2col %.0f ms)\n", r, H, W, ms_hw,
              ms_i2c);
  EXPECT_TRUE(r < 3e-2);
}

// The DiT loads and one forward step produces a finite velocity of the packed
// latent shape [img_seq, in_channels].
TEST(flux2_smoke, dit_forward_shape_finite)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string tdir = std::string(root) + "/transformer";
  auto dit = MetalFlux2Transformer::load(
      tdir, mc, MetalFlux2Transformer::Config{});
  ASSERT_TRUE(dit != nullptr);

  const auto& c = dit->config();
  const int TS = 8;                     // a short fake prompt
  const int gh = 4, gw = 4, img_seq = gh * gw;
  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
  SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq *
                                            c.in_channels * 2);
  ASSERT_TRUE(!ctx.empty() && !lat.empty());
  {
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* cp = static_cast<_Float16*>(ctx.contents());
    for (std::size_t i = 0; i < (std::size_t)TS * c.joint_dim; ++i) {
      cp[i] = (_Float16)nd(rng);
    }
    auto* lp = static_cast<_Float16*>(lat.contents());
    for (std::size_t i = 0; i < (std::size_t)img_seq * c.in_channels; ++i) {
      lp[i] = (_Float16)nd(rng);
    }
  }
  SharedBuffer vel = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
  ASSERT_TRUE(!vel.empty());
  const std::size_t n = (std::size_t)img_seq * c.out_channels;
  ASSERT_TRUE(vel.byte_size() >= n * 2);
  EXPECT_TRUE(all_finite_(vel, n));
  std::printf("[flux2_smoke] DiT forward -> velocity [%d, %d] (finite)\n",
              img_seq, c.out_channels);
}

// Reference-image conditioning: forward_dit with reference latents (a) still
// emits ONLY the generated-token velocity [img_seq, out_channels] (refs dropped
// from the output), (b) stays finite with one and two references, and (c)
// actually CHANGES the velocity vs the no-ref forward -- proving the reference
// tokens are embedded, position-offset and folded into the joint attention (a
// dead ref path would give a byte-identical result). Same fixed random inputs.
TEST(flux2_smoke, dit_reference_images_change_output)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string tdir = std::string(root) + "/transformer";
  auto dit = MetalFlux2Transformer::load(
      tdir, mc, MetalFlux2Transformer::Config{});
  ASSERT_TRUE(dit != nullptr);
  const auto& c = dit->config();

  const int TS = 8;
  const int gh = 4, gw = 4, img_seq = gh * gw;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto fill = [&](SharedBuffer& b, std::size_t cnt) {
    auto* p = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < cnt; ++i) { p[i] = (_Float16)nd(rng); }
  };
  // A random packed reference latent [rseq, in_channels] on a grid_h x grid_w
  // grid, mirroring the vae-encode output the stage patchify-packs.
  auto make_ref = [&](int rgh, int rgw) -> MetalFlux2Transformer::RefImage {
    MetalFlux2Transformer::RefImage r;
    r.seq = rgh * rgw; r.grid_h = rgh; r.grid_w = rgw;
    r.latents = mc->make_shared_buffer((std::size_t)r.seq * c.in_channels * 2);
    fill(r.latents, (std::size_t)r.seq * c.in_channels);
    return r;
  };

  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
  SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq *
                                            c.in_channels * 2);
  ASSERT_TRUE(!ctx.empty() && !lat.empty());
  fill(ctx, (std::size_t)TS * c.joint_dim);
  fill(lat, (std::size_t)img_seq * c.in_channels);

  const std::size_t n = (std::size_t)img_seq * c.out_channels;
  auto to_vec = [&](const SharedBuffer& b) {
    std::vector<float> v(n);
    const auto* p = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)p[i]; }
    return v;
  };

  // Baseline (no refs) and one ref (2x3 grid) + two refs.
  SharedBuffer v0 = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
  ASSERT_TRUE(!v0.empty() && all_finite_(v0, n));
  // RefImage is move-only (holds a SharedBuffer), so move into the vectors.
  std::vector<MetalFlux2Transformer::RefImage> one;
  one.push_back(make_ref(2, 3));
  SharedBuffer v1 = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f,
                                     -1.0f, one);
  ASSERT_TRUE(!v1.empty());
  ASSERT_TRUE(v1.byte_size() >= n * 2);   // still generated-token count only
  EXPECT_TRUE(all_finite_(v1, n));
  std::vector<MetalFlux2Transformer::RefImage> two;
  two.push_back(make_ref(2, 3));
  two.push_back(make_ref(3, 2));
  SharedBuffer v2 = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f,
                                     -1.0f, two);
  ASSERT_TRUE(!v2.empty() && v2.byte_size() >= n * 2);
  EXPECT_TRUE(all_finite_(v2, n));

  const std::vector<float> a = to_vec(v0), b1 = to_vec(v1), b2 = to_vec(v2);
  const double r1 = rel_l2_(b1.data(), a.data(), n);
  const double r2 = rel_l2_(b2.data(), a.data(), n);
  std::printf("[flux2_smoke] DiT ref-image velocity delta: 1 ref rel-L2 %.4f, "
              "2 refs rel-L2 %.4f (vs no-ref)\n", r1, r2);
  EXPECT_TRUE(r1 > 1e-3);          // a reference materially changes the output
  EXPECT_TRUE(r2 > 1e-3);
}

// M5 matrix-core matmul2d A/B (mirrors krea2_dit.forward_dit_mma_matches_
// steel): run the same deterministic random forward twice -- default (the
// matmul2d route on matrix-core GPUs) and VPIPE_FLUX2_NO_MMA2 forcing steel --
// and rel-L2 the velocities. Exercises gemm_mma_ end to end: dequant-once (on
// a quantized DiT via VPIPE_FLUX2_DIT_DIR), the tile routing, split-K on the
// single-stream to_out (K = H + SMLP), and element-offset x reads (the double
// blocks' img-half GEMMs). The steel run keeps its fused-SwiGLU FF (the mma
// run defaults it off), so the paths differ by GEMM accumulation order + the
// FF fusion -- both f16-bounded. stream_blocks=true keeps the DiT off the
// heap (verified bit-identical to preloaded), so this also runs the bf16 9B
// on a 16 GB box. On a non-matrix-core GPU the two runs take the same steel
// path and the rel-L2 is ~0.
TEST(flux2_smoke, dit_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string tdir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";

  // seq = 64 txt + 256 img = 320: everything past _mma_min_m so the block
  // GEMMs all route through matmul2d in the default run.
  const int TS = 64, gh = 16, gw = 16, img_seq = gh * gw;

  std::printf("[flux2_smoke] dit_mma_matches_steel: matrix_cores=%d (%s)\n",
              (int)mc->supports_matrix_cores(), tdir.c_str());

  auto run = [&](bool no_mma) {
    if (no_mma) { ::setenv("VPIPE_FLUX2_NO_MMA2", "1", 1); }
    else        { ::unsetenv("VPIPE_FLUX2_NO_MMA2"); }
    auto dit = MetalFlux2Transformer::load(
        tdir, mc, MetalFlux2Transformer::Config{}, /*stream_blocks=*/true);
    ::unsetenv("VPIPE_FLUX2_NO_MMA2");
    std::vector<float> out;
    if (dit == nullptr) { return out; }
    const auto& c = dit->config();
    SharedBuffer ctx =
        mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
    SharedBuffer lat =
        mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
    if (ctx.empty() || lat.empty()) { return out; }
    std::mt19937 rng(4242);              // same inputs both runs
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* cp = static_cast<_Float16*>(ctx.contents());
    for (std::size_t i = 0; i < (std::size_t)TS * c.joint_dim; ++i) {
      cp[i] = (_Float16)nd(rng);
    }
    auto* lp = static_cast<_Float16*>(lat.contents());
    for (std::size_t i = 0; i < (std::size_t)img_seq * c.in_channels; ++i) {
      lp[i] = (_Float16)nd(rng);
    }
    SharedBuffer vel = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    const std::size_t n = (std::size_t)img_seq * c.out_channels;
    if (vel.empty() || vel.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* vp = static_cast<const _Float16*>(vel.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)vp[i]; }
    return out;
  };

  const std::vector<float> v_mma   = run(/*no_mma=*/false);
  const std::vector<float> v_steel = run(/*no_mma=*/true);
  ASSERT_TRUE(!v_mma.empty());
  ASSERT_TRUE(v_mma.size() == v_steel.size());
  const double r = rel_l2_(v_mma.data(), v_steel.data(), v_mma.size());
  std::printf("[flux2_smoke] DiT mma-vs-steel velocity rel-L2 = %.6g "
              "(seq=%d)\n", r, TS + img_seq);
  EXPECT_TRUE(r < 3e-2);
}

// Accelerated mode A/B: the same deterministic forward with the dynamic-
// int8 GEMMs (VPIPE_I8_GEMM=1 -> shared/i8-gemm.h route in gemm_mma_) vs
// the default f16 path. LOSSY by design (per-GEMM int8 ~1e-2), so the
// bound here is the DiT-level drift budget, not exactness. Uses a big
// enough grid that the block GEMMs pass the M >= 1024 gate. stream_blocks
// keeps the DiT off the heap. Skips vacuously off matrix-core GPUs.
TEST(flux2_smoke, dit_i8_gemm_matches_f16)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string tdir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";

  // 32x40 grid -> img_seq 1280 (past the i8 M-gate), txt 64.
  const int TS = 64, gh = 32, gw = 40, img_seq = gh * gw;

  auto run = [&](bool i8) {
    ::setenv("VPIPE_I8_GEMM", i8 ? "1" : "0", 1);
    auto dit = MetalFlux2Transformer::load(
        tdir, mc, MetalFlux2Transformer::Config{}, /*stream_blocks=*/true);
    ::unsetenv("VPIPE_I8_GEMM");
    std::vector<float> out;
    if (dit == nullptr) { return out; }
    const auto& c = dit->config();
    SharedBuffer ctx =
        mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
    SharedBuffer lat =
        mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
    if (ctx.empty() || lat.empty()) { return out; }
    std::mt19937 rng(777);                    // same inputs both runs
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* cp = static_cast<_Float16*>(ctx.contents());
    for (std::size_t i = 0; i < (std::size_t)TS * c.joint_dim; ++i) {
      cp[i] = (_Float16)nd(rng);
    }
    auto* lp = static_cast<_Float16*>(lat.contents());
    for (std::size_t i = 0; i < (std::size_t)img_seq * c.in_channels; ++i) {
      lp[i] = (_Float16)nd(rng);
    }
    SharedBuffer vel = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    const std::size_t n = (std::size_t)img_seq * c.out_channels;
    if (vel.empty() || vel.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* vp = static_cast<const _Float16*>(vel.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)vp[i]; }
    return out;
  };

  const std::vector<float> v_i8 = run(true);
  const std::vector<float> v_f16 = run(false);
  ASSERT_TRUE(!v_i8.empty());
  ASSERT_TRUE(v_i8.size() == v_f16.size());
  const double r = rel_l2_(v_i8.data(), v_f16.data(), v_i8.size());
  std::printf("[flux2_smoke] DiT i8-accel-vs-f16 velocity rel-L2 = %.4e "
              "(seq=%d)\n", r, TS + img_seq);
  EXPECT_TRUE(r < 0.10);
}

// ===== numerical rel-L2 vs the diffusers golden (dump_flux2_golden.py) =====
// Env: VPIPE_FLUX2_TEST_MODEL_PATH + VPIPE_FLUX2_GOLDEN. Skips if unset.

// DiT: feed the golden random hidden/context + the same fixed geometry, rel-L2
// the velocity against the reference. Isolates the DiT math (SwiGLU FF, norm_out
// order, 4-axis RoPE, modulation) -- random context, so no encoder/padding.
TEST(flux2_golden, dit_velocity_rel_l2)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_FLUX2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string g = gd;

  std::ifstream mf(g + "/meta.json");
  FlexData meta = FlexData::from_json(mf);
  ASSERT_TRUE(meta.is_object());
  auto mo = meta.as_object();
  const int gh = (int)mo.at("gh").as_int(0);
  const int gw = (int)mo.at("gw").as_int(0);
  const int img_seq = (int)mo.at("img_seq").as_int(0);
  const int TS = (int)mo.at("text_seq").as_int(0);
  const int IN = (int)mo.at("in_channels").as_int(0);
  const int JD = (int)mo.at("joint_dim").as_int(0);
  const int OUT = (int)mo.at("out_channels").as_int(0);
  const float ts = (float)mo.at("timestep").as_real(0.0);

  const std::vector<float> hid = read_f32_file_(g + "/dit_hidden.f32");
  const std::vector<float> ctx = read_f32_file_(g + "/dit_context.f32");
  const std::vector<float> ref = read_f32_file_(g + "/dit_velocity.f32");
  ASSERT_TRUE(hid.size() == (std::size_t)img_seq * IN);
  ASSERT_TRUE(ctx.size() == (std::size_t)TS * JD);
  ASSERT_TRUE(ref.size() == (std::size_t)img_seq * OUT);

  // Optional quantized-DiT override (VPIPE_FLUX2_DIT_DIR) so plain-w4 vs
  // awq-w4 can be rel-L2'd against the same f32 golden. VPIPE_FLUX2_STREAM=1
  // forces block streaming (must be bit-identical to preloaded).
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const bool quantized = (ddir != nullptr && *ddir != '\0');
  const char* strv = std::getenv("VPIPE_FLUX2_STREAM");
  const bool stream = (strv != nullptr && std::atoi(strv) != 0);
  auto dit = MetalFlux2Transformer::load(
      quantized ? std::string(ddir) : std::string(root) + "/transformer", mc,
      MetalFlux2Transformer::Config{}, stream);
  ASSERT_TRUE(dit != nullptr);

  SharedBuffer latb = mc->make_shared_buffer(hid.size() * 2);
  SharedBuffer ctxb = mc->make_shared_buffer(ctx.size() * 2);
  { auto* d = static_cast<_Float16*>(latb.contents());
    for (std::size_t i = 0; i < hid.size(); ++i) { d[i] = (_Float16)hid[i]; } }
  { auto* d = static_cast<_Float16*>(ctxb.contents());
    for (std::size_t i = 0; i < ctx.size(); ++i) { d[i] = (_Float16)ctx[i]; } }

  SharedBuffer vel = dit->forward_dit(ctxb, TS, latb, img_seq, gh, gw, ts);
  ASSERT_TRUE(!vel.empty());
  std::vector<float> got(ref.size());
  const auto* vp = static_cast<const _Float16*>(vel.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)vp[i]; }

  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[flux2_golden] DiT velocity rel-L2 = %.5f (%s, %d blocks)\n", r,
              quantized ? "quantized" : "f16 vs f32 ref", 5 + 20);
  // Dense f16 drifts ~0.005. Quantized is looser and width-dependent: plain w8
  // ~0.04, plain w4 ~0.31 on this random-input golden (AWQ/mixed are the quality
  // path). The bound here is a coarse "did it dispatch + stay sane" smoke.
  EXPECT_TRUE(r < (quantized ? 0.40 : 0.02));
}

// Embedded guidance (guidance-distilled variants, e.g. klein-9B): re-feed the
// same golden hidden/context but pass an embedded guidance scale and rel-L2 the
// velocity against the guidance-on reference. Skips vacuously unless the golden
// carries guidance_embeds + a dit_velocity_guided.f32 (i.e. the model has a
// guidance_embedder -- the 4B does not, so this only runs for the 9B golden).
TEST(flux2_golden, dit_velocity_guided_rel_l2)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_FLUX2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string g = gd;
  std::ifstream mf(g + "/meta.json");
  FlexData meta = FlexData::from_json(mf);
  if (!meta.is_object()) { return; }
  auto mo = meta.as_object();
  if (!mo.contains("guidance_embeds") ||
      !mo.at("guidance_embeds").as_bool(false)) {
    return;   // model has no guidance_embedder -> nothing to verify (4B)
  }
  const float guidance = (float)mo.at("guidance").as_real(0.0);
  const std::vector<float> ref = read_f32_file_(g + "/dit_velocity_guided.f32");
  if (ref.empty()) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const int gh = (int)mo.at("gh").as_int(0);
  const int gw = (int)mo.at("gw").as_int(0);
  const int img_seq = (int)mo.at("img_seq").as_int(0);
  const int TS = (int)mo.at("text_seq").as_int(0);
  const int IN = (int)mo.at("in_channels").as_int(0);
  const int JD = (int)mo.at("joint_dim").as_int(0);
  const int OUT = (int)mo.at("out_channels").as_int(0);
  const float ts = (float)mo.at("timestep").as_real(0.0);

  const std::vector<float> hid = read_f32_file_(g + "/dit_hidden.f32");
  const std::vector<float> ctx = read_f32_file_(g + "/dit_context.f32");
  ASSERT_TRUE(hid.size() == (std::size_t)img_seq * IN);
  ASSERT_TRUE(ctx.size() == (std::size_t)TS * JD);
  ASSERT_TRUE(ref.size() == (std::size_t)img_seq * OUT);

  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const bool quantized = (ddir != nullptr && *ddir != '\0');
  auto dit = MetalFlux2Transformer::load(
      quantized ? std::string(ddir) : std::string(root) + "/transformer", mc,
      MetalFlux2Transformer::Config{});
  ASSERT_TRUE(dit != nullptr);
  ASSERT_TRUE(dit->config().guidance_embeds);   // loader saw guidance_embeds

  SharedBuffer latb = mc->make_shared_buffer(hid.size() * 2);
  SharedBuffer ctxb = mc->make_shared_buffer(ctx.size() * 2);
  { auto* d = static_cast<_Float16*>(latb.contents());
    for (std::size_t i = 0; i < hid.size(); ++i) { d[i] = (_Float16)hid[i]; } }
  { auto* d = static_cast<_Float16*>(ctxb.contents());
    for (std::size_t i = 0; i < ctx.size(); ++i) { d[i] = (_Float16)ctx[i]; } }

  SharedBuffer vel =
      dit->forward_dit(ctxb, TS, latb, img_seq, gh, gw, ts, guidance);
  ASSERT_TRUE(!vel.empty());
  std::vector<float> got(ref.size());
  const auto* vp = static_cast<const _Float16*>(vel.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)vp[i]; }

  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[flux2_golden] DiT guided velocity rel-L2 = %.5f "
              "(guidance=%.2f, %s)\n", r, guidance,
              quantized ? "quantized" : "f16 vs f32 ref");
  EXPECT_TRUE(r < (quantized ? 0.20 : 0.02));
}

// VAE: decode the golden DiT-facing latent [dit_channels, gh, gw], rel-L2 the
// RGB against the reference (un-bn + unpatchify + decode).
TEST(flux2_golden, vae_decode_rel_l2)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_FLUX2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string g = gd;

  std::ifstream mf(g + "/meta.json");
  FlexData meta = FlexData::from_json(mf);
  ASSERT_TRUE(meta.is_object());
  auto mo = meta.as_object();
  const int gh = (int)mo.at("gh").as_int(0);
  const int gw = (int)mo.at("gw").as_int(0);
  const int C = (int)mo.at("dit_channels").as_int(0);
  const int H = (int)mo.at("image_h").as_int(0);
  const int W = (int)mo.at("image_w").as_int(0);

  const std::vector<float> z = read_f32_file_(g + "/vae_z.f32");
  const std::vector<float> ref = read_f32_file_(g + "/vae_image.f32");
  ASSERT_TRUE(z.size() == (std::size_t)C * gh * gw);
  ASSERT_TRUE(ref.size() == (std::size_t)3 * H * W);

  std::string vdir = std::string(root) + "/vae";
  auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
  ASSERT_TRUE(vae != nullptr);
  ASSERT_TRUE(C == vae->config().dit_channels());

  SharedBuffer zb = mc->make_shared_buffer(z.size() * 2);
  { auto* d = static_cast<_Float16*>(zb.contents());
    for (std::size_t i = 0; i < z.size(); ++i) { d[i] = (_Float16)z[i]; } }
  std::string err;
  SharedBuffer rgb = vae->decode(zb, gh, gw, &err);
  ASSERT_TRUE(!rgb.empty());
  std::vector<float> got(ref.size());
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)rp[i]; }

  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[flux2_golden] VAE decode rel-L2 = %.5f (%dx%d)\n", r, H, W);
  EXPECT_TRUE(r < 0.01);
}

// Per-component quantize (chained DiT + text_encoder passes): confirm the flux2
// Qwen3 dense text encoder LOADS + runs its {9,18,27} taps, whether bf16 or
// affine-quantized (VPIPE_FLUX2_ENC_DIR override -> a model-quantize'd
// text_encoder dir). Exercises the runtime load of a quantized encoder.
TEST(flux2_golden, text_encoder_loads_runs)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  const char* eov = std::getenv("VPIPE_FLUX2_ENC_DIR");
  const std::string edir = (eov != nullptr && *eov != '\0')
      ? std::string(eov) : std::string(root) + "/text_encoder";

  MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 2560; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936; c.rope_theta = 1.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = true; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "model."; c.max_seq = 1024;
  c.page_tokens = 256;
  bool declared_quant = false;   // config.json carries a quantization block
  { std::ifstream in(fs::path(edir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto obj = fd.as_object();               // bind view to a local
        auto geti = [&](const char* k, int cur) -> int {
          return obj.contains(k) ? (int)obj.at(k).as_int(cur) : cur;
        };
        auto getf = [&](const char* k, float cur) -> float {
          return obj.contains(k) ? (float)obj.at(k).as_real(cur) : cur;
        };
        // Size from config.json so the 8B (9B pipeline) encoder validates too.
        c.n_layers   = geti("num_hidden_layers", c.n_layers);
        c.hidden     = geti("hidden_size", c.hidden);
        c.n_heads    = geti("num_attention_heads", c.n_heads);
        c.n_kv_heads = geti("num_key_value_heads", c.n_kv_heads);
        c.head_dim   = geti("head_dim",
                            c.n_heads > 0 ? c.hidden / c.n_heads : c.head_dim);
        c.rotary_dim = c.head_dim;
        c.ffn_inner  = geti("intermediate_size", c.ffn_inner);
        c.vocab      = geti("vocab_size", c.vocab);
        c.rope_theta = getf("rope_theta", c.rope_theta);
        c.rms_eps    = getf("rms_norm_eps", c.rms_eps);
        if (obj.contains("tie_word_embeddings")) {
          c.tie_embeddings = obj.at("tie_word_embeddings").as_bool(true);
        }
        if (obj.contains("quantization")) {
          FlexData q = obj.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();             // bind view to a local
            if (qo.contains("bits")) {
              const int b = (int)qo.at("bits").as_int(0);
              if (b == 4 || b == 8) { c.quant_bits = b; declared_quant = true; }
            }
          }
        }
      }
    } }

  auto enc = MetalQwenModel::load(edir, mc, c);
  ASSERT_TRUE(enc != nullptr);
  auto wts = MetalLlamaWeights::open_model(edir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer embed = wts->load("model.embed_tokens.weight", mc);
  ASSERT_TRUE(!embed.empty());

  const int EH = c.hidden, n = 8;
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
  { const auto* tbl = static_cast<const std::uint8_t*>(embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      std::memcpy(xb + (std::size_t)i * EH * 2,
                  tbl + (std::size_t)(i + 100) * EH * 2, (std::size_t)EH * 2);
    } }
  genai::ContextManager* cm = enc->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  SharedBuffer taps = enc->forward_embeddings_taps(cid, x, n, {8, 17, 26});
  cm->release(cid);
  ASSERT_TRUE(!taps.empty());
  EXPECT_TRUE(all_finite_bf16_(taps, (std::size_t)3 * n * EH));
  std::printf("[flux2_golden] Qwen3 encoder (%s, hidden %d) {9,18,27} taps "
              "finite\n", declared_quant ? "quantized" : "bf16", EH);
}

// End-to-end: the full text-to-image (tokenize -> Qwen3 encode {9,18,27} tap
// concat -> FLUX.2 DiT sampler) -> vae-decode pipeline produces a COHERENT RGB
// image from a prompt. Free-running (no golden anchor), so this checks the whole
// path dispatches + the image is non-degenerate (not flat), and dumps a PPM.
// Heavy (loads the encoder + DiT together, ~34 GB for the 9B) -> OPT-IN via
// VPIPE_FLUX2_E2E. Env: VPIPE_FLUX2_TEST_MODEL_PATH + VPIPE_FLUX2_E2E.
TEST(flux2_e2e, text_to_image_produces_image)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  if (std::getenv("VPIPE_FLUX2_E2E") == nullptr) { return; }   // opt-in (heavy)
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  // Resolution is env-selectable (VPIPE_FLUX2_E2E_HW, default 256) so the
  // resolution-dependent flow-shift (mu) can be exercised at 1024 etc.
  int H = 256, W = 256;         // multiple of 16
  if (const char* hw = std::getenv("VPIPE_FLUX2_E2E_HW")) {
    const int v = std::atoi(hw);
    if (v >= 16 && (v % 16) == 0) { H = W = v; }
  }
  const int steps = 4;          // klein distilled default

  auto pl = std::make_unique<Pipeline>("flux2-e2e", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src", std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompt = "a fox in the snow";
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

  FlexData t2i_cfg = FlexData::make_object();
  t2i_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  t2i_cfg.as_object().insert("height", FlexData::make_int(H));
  t2i_cfg.as_object().insert("width", FlexData::make_int(W));
  t2i_cfg.as_object().insert("steps", FlexData::make_int(steps));
  t2i_cfg.as_object().insert("seed", FlexData::make_int(0));
  auto t2iu = std::make_unique<TextToImageStage>(
      &sess, "t2i", std::vector<InEdge>{{src, 0}}, std::move(t2i_cfg));
  auto* t2i = static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
  ASSERT_TRUE(t2i->config_error().empty());

  FlexData vae_cfg = FlexData::make_object();
  vae_cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto vaeu = std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{t2i, 0}}, std::move(vae_cfg));
  auto* vae = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vaeu)));

  auto sinku = std::make_unique<SinkCapture>(
      &sess, "sink", std::vector<InEdge>{{vae, 0}}, FlexData::make_object());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(t2i->latents_emitted() == 1);
  ASSERT_TRUE(sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  ASSERT_TRUE(tb != nullptr);
  EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8);
  ASSERT_TRUE(tb->shape.size() == 3 && tb->shape[0] == 3 &&
              tb->shape[1] == H && tb->shape[2] == W);

  const std::uint8_t* u = tb->as_u8();
  const std::size_t np = (std::size_t)3 * H * W;
  double mean = 0.0;
  for (std::size_t i = 0; i < np; ++i) { mean += u[i]; }
  mean /= (double)np;
  double var = 0.0;
  for (std::size_t i = 0; i < np; ++i) {
    const double d = (double)u[i] - mean; var += d * d;
  }
  const double sd = std::sqrt(var / (double)np);
  std::printf("[flux2_e2e] decoded image mean=%.1f std=%.1f (%dx%d, %d steps)\n",
              mean, sd, H, W, steps);
  EXPECT_TRUE(sd > 8.0);              // not a flat/degenerate image

  {
    const std::string ppm = "/tmp/flux2_t2i_metal.ppm";
    std::ofstream o(ppm, std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int cc = 0; cc < 3; ++cc) {
          o.put((char)u[((std::size_t)cc * H + y) * W + x]);
        }
      }
    }
    std::printf("[flux2_e2e] wrote %s\n", ppm.c_str());
  }
}

// Stage-level reference-image path: the SAME prompt + seed generates a
// DIFFERENT latent when a reference latent is wired on iport4 (ref_latent0) than
// without one. Exercises the full stage plumbing -- iport4 read -> patchify-pack
// -> forward_dit with reference tokens -> generated-only output -- not just the
// DiT. A dead iport4 (ref ignored) would give a byte-identical latent and fail.
// Opt-in (loads encoder + DiT): VPIPE_FLUX2_TEST_MODEL_PATH + VPIPE_FLUX2_E2E.
TEST(flux2_e2e, reference_latent_iport_changes_latent)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  if (std::getenv("VPIPE_FLUX2_E2E") == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const int H = 256, W = 256, steps = 4;

  // Run the t2i stage (latent out, no VAE) for the fixed prompt/seed, optionally
  // with a reference latent [128, 8, 8] on iport4. Returns the emitted DiT
  // latent [128, H/16, W/16], or empty on failure.
  auto run = [&](bool with_ref) -> std::vector<float> {
    auto pl = std::make_unique<Pipeline>("flux2-ref", &sess);
    auto srcu = std::make_unique<SourceText>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
    srcu->prompt = "a fox in the snow";
    auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));

    SourceTensor* rt_src = nullptr;
    if (with_ref) {
      auto ru = std::make_unique<SourceTensor>(&sess, "ref",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
      ru->C = 128; ru->H = 8; ru->W = 8;
      ru->chw.resize((std::size_t)ru->C * ru->H * ru->W);
      std::mt19937 rng(123);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& v : ru->chw) { v = nd(rng); }
      rt_src = static_cast<SourceTensor*>(pl->insert_stage(std::move(ru)));
    }

    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    cfg.as_object().insert("height", FlexData::make_int(H));
    cfg.as_object().insert("width", FlexData::make_int(W));
    cfg.as_object().insert("steps", FlexData::make_int(steps));
    cfg.as_object().insert("seed", FlexData::make_int(0));
    // iports {prompt, negative, sampler, scheduler, ref0, ref1}; wire ref0.
    std::vector<InEdge> edges{{src, 0}};
    if (with_ref) {
      edges = std::vector<InEdge>{{src, 0}, InEdge{nullptr, 0},
                                  InEdge{nullptr, 0}, InEdge{nullptr, 0},
                                  {rt_src, 0}};
    }
    auto t2iu = std::make_unique<TextToImageStage>(&sess, "t2i", edges,
                                                   std::move(cfg));
    auto* t2i =
        static_cast<TextToImageStage*>(pl->insert_stage(std::move(t2iu)));
    if (!t2i->config_error().empty()) { return {}; }
    auto sinku = std::make_unique<SinkCapture>(
        &sess, "sink", std::vector<InEdge>{{t2i, 0}}, FlexData::make_object());
    auto* sink = static_cast<SinkCapture*>(pl->insert_stage(std::move(sinku)));
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) { return {}; }
    rt.wait_idle();
    rt.stop();
    if (sink->captured.size() != 1) { return {}; }
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
    if (tb == nullptr || tb->dtype != TensorBeat::DType::F32) { return {}; }
    const float* f = tb->as_f32();
    return std::vector<float>(f, f + tb->element_count());
  };

  const std::vector<float> base = run(false);
  const std::vector<float> refd = run(true);
  ASSERT_TRUE(!base.empty() && !refd.empty());
  ASSERT_TRUE(base.size() == refd.size());
  bool finite = true;
  for (float v : refd) { if (!std::isfinite(v)) { finite = false; break; } }
  EXPECT_TRUE(finite);
  const double r = rel_l2_(refd.data(), base.data(), base.size());
  std::printf("[flux2_e2e] reference-vs-noref latent rel-L2 = %.6f\n", r);
  EXPECT_TRUE(r > 0.01);   // the reference materially steered the generation
}

// DiT-only bench: run forward_dit at a target resolution with random context
// (no encoder -> loads only the ~18 GB DiT). Reports ms/step; set
// VPIPE_FLUX2_DIT_PROFILE for the per-section breakdown. Opt-in (heavy).
// Env: VPIPE_FLUX2_TEST_MODEL_PATH + VPIPE_FLUX2_BENCH; VPIPE_FLUX2_BENCH_HW
// (default 1024), VPIPE_FLUX2_BENCH_ITERS (default 3).
TEST(flux2_bench, forward_dit)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  if (std::getenv("VPIPE_FLUX2_BENCH") == nullptr) { return; }   // opt-in
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  int HW = 1024;
  if (const char* h = std::getenv("VPIPE_FLUX2_BENCH_HW")) { HW = std::atoi(h); }
  int iters = 3;
  if (const char* it = std::getenv("VPIPE_FLUX2_BENCH_ITERS")) {
    iters = std::max(1, std::atoi(it));
  }
  // Optional quantized-DiT override (VPIPE_FLUX2_DIT_DIR -> a model-quantize'd
  // w8/w4 DiT dir) so bf16 vs w8 vs w4 can be benched against the same shapes.
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string dit_dir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";
  auto dit = MetalFlux2Transformer::load(dit_dir, mc,
                                         MetalFlux2Transformer::Config{});
  ASSERT_TRUE(dit != nullptr);
  const auto& c = dit->config();
  const int gh = HW / 16, gw = HW / 16, img_seq = gh * gw;
  const int TS = 40;                         // representative prompt length
  const int IC = c.in_channels, JD = c.joint_dim;

  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * JD * 2);
  SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
  { auto* p = static_cast<_Float16*>(ctx.contents());
    for (std::size_t i = 0; i < (std::size_t)TS * JD; ++i) {
      p[i] = (_Float16)nd(rng);
    } }
  { auto* p = static_cast<_Float16*>(lat.contents());
    for (std::size_t i = 0; i < (std::size_t)img_seq * IC; ++i) {
      p[i] = (_Float16)nd(rng);
    } }

  SharedBuffer v0 = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
  ASSERT_TRUE(!v0.empty());                  // warm-up
  double best = 1e30, sum = 0.0;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer v = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(!v.empty());
    best = std::min(best, ms); sum += ms;
  }
  std::printf("[flux2_bench] forward_dit %dx%d (seq %d = %d img + %d txt, "
              "hidden %d): %.0f ms/step best, %.0f avg (%d iters)\n",
              HW, HW, TS + img_seq, img_seq, TS, c.hidden, best,
              sum / iters, iters);
}

// Quantize <root>/transformer -> <root>-dit-w<bits>g64 (plain affine, the DiT
// Linear leaf set), reusing the dir if present. Opt-in: VPIPE_FLUX2_QUANTIZE =
// 8 | 4. Produces the dirs the bench / golden feed via VPIPE_FLUX2_DIT_DIR.
TEST(flux2_quant, make)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  const char* qb   = std::getenv("VPIPE_FLUX2_QUANTIZE");
  if (root == nullptr || *root == '\0' || qb == nullptr || *qb == '\0') {
    return;
  }
  const int bits = std::atoi(qb);
  if (bits != 4 && bits != 8) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  const std::string src = std::string(root) + "/transformer";
  const std::string dst =
      std::string(root) + "-dit-w" + std::to_string(bits) + "g64";
  if (fs::exists(fs::path(dst) / "config.json")) {
    std::printf("[flux2_quant] reuse %s\n", dst.c_str());
    return;
  }
  QuantizeOptions opt;
  opt.bits = bits;
  opt.group = 64;
  opt.dit_family = "flux2";
  // The FLUX.2 DiT quant leaf set (kept in lock-step with dit_quant_linears_
  // in model-quantize-stage.cc): the big per-block compute Linears; embedders +
  // modulation + norm_out stay bf16.
  opt.quant_linears = {"to_q", "to_k", "to_v", "to_out", "to_add_out",
                       "add_q_proj", "add_k_proj", "add_v_proj",
                       "to_qkv_mlp_proj", "linear_in", "linear_out"};
  ModelQuantizer q(mc);
  std::string err;
  const bool ok = q.run(src, dst, opt, &err);
  ASSERT_TRUE(ok);
  std::printf("[flux2_quant] wrote %s (w%d)\n", dst.c_str(), bits);
}

// On-device DiT AWQ calibration end-to-end: run collect_flux2_calibration (the
// path the model-quantize stage's flux2 dit_act AWQ takes) over a couple of
// prompts x a couple of sigmas, and confirm it paints the two-phase progress
// bar (encode, then denoise, on the session text stream) and writes the
// per-group abs-max .f32 files the quantizer consumes. Heavy (loads the encoder
// then the DiT) -> OPT-IN via VPIPE_FLUX2_CALIB. Env: VPIPE_FLUX2_TEST_MODEL_PATH.
TEST(flux2_calib, on_device_dit_awq_progress_and_files)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  if (std::getenv("VPIPE_FLUX2_CALIB") == nullptr) { return; }   // opt-in (heavy)
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  namespace fs = std::filesystem;

  const std::vector<std::string> prompts = {
    "a fox in the snow", "a red bicycle on a cobblestone street"};
  const fs::path out = fs::temp_directory_path() / "vpipe-flux2-calib-smoke";
  std::error_code ec;
  fs::remove_all(out, ec);

  std::string err;
  // 2 prompts x 2 steps @ 256 -> the bar advances 2 encode + 4 denoise ticks.
  const bool ok = collect_flux2_calibration(
      sess.metal_compute(), root, prompts, /*steps=*/2, 256, 256, /*seed=*/0,
      out.string(), &err);
  ASSERT_TRUE(ok);   // err is set on failure

  // The quantizer's flux2 dit_act consumes per-group abs-max files; a couple of
  // representative ones must exist and be non-empty.
  for (const char* g : {"dbl_norm1_img", "sgl_cat", "emb_x"}) {
    const fs::path f = out / (std::string(g) + ".f32");
    ASSERT_TRUE(fs::exists(f));
    EXPECT_TRUE(fs::file_size(f, ec) > 0);
  }
  std::printf("[flux2_calib] on-device DiT AWQ calibration wrote group files "
              "to %s\n", out.string().c_str());
  fs::remove_all(out, ec);
}
