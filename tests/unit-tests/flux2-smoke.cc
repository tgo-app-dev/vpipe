// FLUX.2-klein bring-up smoke + golden (both the 4B and 9B sizes -- the whole
// flux2 stack is config-driven off config.json, so ONE set of tests serves
// both). Three tiers:
//   * flux2_smoke.*  -- plumbing: the model classes load + a forward produces a
//     finite output of the expected shape (no reference).
//   * flux2_golden.* -- numerical rel-L2 vs the diffusers golden
//     (dump_flux2_golden.py): DiT velocity, embedded-guidance velocity (when the
//     model has a guidance_embedder), and VAE decode.
//   * flux2_e2e.*    -- the full generate-image -> vae-decode pipeline produces a
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
#include "generative-models/generative-model-manager.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
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

// Chain a diffusion-conditioner between `src` (prompt) and the generate-image
// (DiT) stage -- the encoder half moved there. Returns the conditioner; wire
// the t2i's iport0 to {cond, 0}.
Stage*
add_conditioner_(Pipeline* pl, Session& sess, Stage* src, const char* root)
{
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(root));
  auto u = std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(c));
  return pl->insert_stage(std::move(u));
}

// The FLUX.2 text encoder's Config, exactly as diffusion-conditioner
// builds it (encoder_config_flux2_): sized from config.json so the 8B
// validates as well as the 4B, backbone-only because this encoder taps
// hidden states and never decodes.
//
// Hoisted so the streaming-equality test runs against the SAME encoder
// the pipeline loads. Two copies of a forty-field config drift, and a
// drifted copy would compare streaming against a model no stage builds.
MetalQwenModel::Config
flux2_encoder_config_(const std::string& edir, bool* quantized)
{
  namespace fs = std::filesystem;
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
  if (quantized != nullptr) { *quantized = declared_quant; }
  return c;
}

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
  // "N" -> N x N; "HxW" -> a non-square grid (the real aspect ratios a
  // generation actually decodes, e.g. 64x48 -> 1024x768).
  if (const char* g = std::getenv("VPIPE_VAE_GRID")) {
    const int v = std::atoi(g);
    const char* x = std::strchr(g, 'x');
    const int v2 = (x != nullptr) ? std::atoi(x + 1) : 0;
    if (v >= 4) { gh = gw = v; }
    if (v >= 4 && v2 >= 4) { gw = v2; }
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
// The direct small-cout 3x3 must agree with the im2col+GEMM it replaces.
// Both accumulate in f32 but in a different order (register accumulators per
// pixel vs a GEMM's K-loop), so this is a rel-L2 bound, not bit-equality.
TEST(flux2_smoke, vae_decode_small_cout_conv_matches_im2col)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  const int gh = 24, gw = 32;                   // -> 384x512, non-square
  // The reference arm must reach the ACTUAL im2col path, so it has to switch
  // off the direct 3x3 conv too -- that also intercepts tiled_conv3x3_, and
  // without this the "im2col" side silently became a second direct-conv run
  // (the two agreed trivially, and the test verified nothing).
  auto run = [&](bool no_small, double* ms) {
    if (no_small) {
      ::setenv("VPIPE_VAE_NO_SMALL_COUT_CONV", "1", 1);
      ::setenv("VPIPE_VAE_NO_DIRECT_CONV", "1", 1);
    } else {
      ::unsetenv("VPIPE_VAE_NO_SMALL_COUT_CONV");
      ::unsetenv("VPIPE_VAE_NO_DIRECT_CONV");
    }
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    ::unsetenv("VPIPE_VAE_NO_SMALL_COUT_CONV");
    ::unsetenv("VPIPE_VAE_NO_DIRECT_CONV");
    std::vector<float> out;
    if (vae == nullptr) { return out; }
    const int C = vae->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    if (z.empty()) { return out; }
    std::mt19937 rng(2468);                     // same latent both runs
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

  double ms_small = 0.0, ms_i2c = 0.0;
  const std::vector<float> v_small = run(/*no_small=*/false, &ms_small);
  const std::vector<float> v_i2c   = run(/*no_small=*/true,  &ms_i2c);
  ASSERT_TRUE(!v_small.empty());
  ASSERT_TRUE(v_small.size() == v_i2c.size());
  const double r = rel_l2_(v_small.data(), v_i2c.data(), v_small.size());
  std::printf("[flux2_smoke] VAE small-cout conv vs im2col rel-L2 = %g "
              "(%dx%d; direct %.0f ms, im2col %.0f ms)\n",
              r, gh * 16, gw * 16, ms_small, ms_i2c);
  EXPECT_TRUE(r < 2e-3);
}

// The direct (im2col-free) 3x3 conv must agree with the im2col + GEMM pair it
// replaces. Both run the same simdgroup MMA in f32; they differ only in where
// the activation tile comes from (gathered on-chip vs read back from the
// [H*W, 9*cin] scratch), so this is a tight rel-L2 bound.
TEST(flux2_smoke, vae_decode_direct_conv_matches_im2col)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  int gh = 24, gw = 32;                         // -> 384x512, non-square
  // VPIPE_VAE_DEC_HW re-points this A/B at a bigger grid. That matters for more
  // than timing: the im2col path's flat element index H*W*9*cin crosses 2^32 at
  // ~2Kx2K (2048*2048*9*256 = 9.7e9), which is the regime its 64-bit index math
  // exists for. Running the comparison THERE is what proves the 32-bit index
  // decomposition (the addresses stay 64-bit) did not silently truncate.
  if (const char* e = std::getenv("VPIPE_VAE_DEC_HW")) {
    const int hpx = std::atoi(e);
    const char* x = std::strchr(e, 'x');
    const int wpx = (x != nullptr) ? std::atoi(x + 1) : hpx;
    if (hpx >= 64 && wpx >= 64) { gh = hpx / 16; gw = wpx / 16; }
  }
  // Both arms hold the small-cout conv OFF so the ONLY difference under test is
  // direct-gather vs materialized im2col on the big 3x3 convs.
  auto run = [&](bool no_direct, double* ms) {
    ::setenv("VPIPE_VAE_NO_SMALL_COUT_CONV", "1", 1);
    if (no_direct) {
      ::setenv("VPIPE_VAE_NO_DIRECT_CONV", "1", 1);
      ::unsetenv("VPIPE_VAE_DIRECT_CONV_MMA2");
    } else {
      ::unsetenv("VPIPE_VAE_NO_DIRECT_CONV");
      // direct_conv3x3_ declines on a MATRIX-CORE GPU unless this is set, so
      // without it both arms are im2col and the test compares a path against
      // itself -- rel-L2 exactly 0, "direct" measuring SLOWER than the thing
      // it replaces, and nothing verified. Same trap the small-cout A/B hit.
      ::setenv("VPIPE_VAE_DIRECT_CONV_MMA2", "1", 1);
    }
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    ::unsetenv("VPIPE_VAE_NO_SMALL_COUT_CONV");
    ::unsetenv("VPIPE_VAE_NO_DIRECT_CONV");
    // NOT VPIPE_VAE_DIRECT_CONV_MMA2 -- unlike the two above (read once at
    // load), that gate is consulted per CALL inside direct_conv3x3_, so
    // clearing it here would close the path again before decode() runs.
    std::vector<float> out;
    if (vae == nullptr) { return out; }
    const int C = vae->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    if (z.empty()) { return out; }
    std::mt19937 rng(1357);                     // same latent both runs
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

  double ms_direct = 0.0, ms_i2c = 0.0;
  const std::vector<float> v_direct = run(/*no_direct=*/false, &ms_direct);
  ::unsetenv("VPIPE_VAE_DIRECT_CONV_MMA2");
  const std::vector<float> v_i2c    = run(/*no_direct=*/true,  &ms_i2c);
  ASSERT_TRUE(!v_direct.empty());
  ASSERT_TRUE(v_direct.size() == v_i2c.size());
  const double r = rel_l2_(v_direct.data(), v_i2c.data(), v_direct.size());
  std::printf("[flux2_smoke] VAE direct conv vs im2col rel-L2 = %g "
              "(%dx%d; direct %.0f ms, im2col %.0f ms)\n",
              r, gh * 16, gw * 16, ms_direct, ms_i2c);
  EXPECT_TRUE(r < 2e-3);
}

// Tiled decode is the MEMORY fallback (see decode_tiled_), so what matters is
// not bit-equality -- it cannot be equal, the mid-block attention is global and
// a window only sees itself -- but that the cross-fade leaves no visible seam
// and the image stays close to the whole-image decode. Report PSNR, and also
// check the seam explicitly: the worst per-column error must not spike at a
// window boundary relative to the image's own error level.
TEST(flux2_smoke, vae_decode_tiled_matches_whole)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  const int gh = 48, gw = 48;                    // 768x768, 2x2 windows of 32
  auto run = [&](int tile, std::vector<float>* out) {
    if (tile > 0) { ::setenv("VPIPE_VAE_TILE", std::to_string(tile).c_str(), 1); }
    else          { ::unsetenv("VPIPE_VAE_TILE"); }
    // NOTE: VPIPE_VAE_TILE is read inside decode(), NOT at load -- it must stay
    // set across the decode call. Clearing it right after load left both arms
    // on the whole-image path, and the test reported a perfect match for
    // comparing a run against itself.
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    struct Clear { ~Clear() { ::unsetenv("VPIPE_VAE_TILE"); } } clear_on_exit;
    if (vae == nullptr) { return; }
    const int C = vae->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    if (z.empty()) { return; }
    std::mt19937 rng(97531);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
      d[i] = (_Float16)nd(rng);
    }
    std::string err;
    SharedBuffer rgb = vae->decode(z, gh, gw, &err);
    const std::size_t n = (std::size_t)3 * (gh * 16) * (gw * 16);
    if (rgb.empty() || rgb.byte_size() < n * 2) { return; }
    out->resize(n);
    const auto* p = static_cast<const _Float16*>(rgb.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = (float)p[i]; }
  };

  std::vector<float> whole, tiled;
  run(0, &whole);
  run(32, &tiled);
  ASSERT_TRUE(!whole.empty());
  ASSERT_TRUE(whole.size() == tiled.size());

  const int H = gh * 16, W = gw * 16;
  double se = 0.0, pk = 0.0;
  for (std::size_t i = 0; i < whole.size(); ++i) {
    const double d = whole[i] - tiled[i];
    se += d * d;
    pk = std::max(pk, (double)std::fabs(whole[i]));
  }
  const double mse = se / (double)whole.size();
  const double psnr = (mse > 0.0 && pk > 0.0)
      ? 10.0 * std::log10(pk * pk / mse) : 99.0;
  // Per-column mean abs error, to locate any seam. Window 32 with overlap 8
  // steps by 24 latent cells -> the seam bands are around x = 24*16 = 384.
  std::vector<double> col((std::size_t)W, 0.0);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      const std::size_t row = ((std::size_t)c * H + y) * W;
      for (int x = 0; x < W; ++x) {
        col[x] += std::fabs(whole[row + x] - tiled[row + x]);
      }
    }
  }
  double cmax = 0.0, csum = 0.0;
  int cmax_x = 0;
  for (int x = 0; x < W; ++x) {
    col[x] /= (double)(3 * H);
    csum += col[x];
    if (col[x] > cmax) { cmax = col[x]; cmax_x = x; }
  }
  const double cmean = csum / (double)W;
  std::printf("[flux2_smoke] VAE tiled vs whole: PSNR %.1f dB, col-err mean "
              "%.4g worst %.4g at x=%d (ratio %.2f)\n",
              psnr, cmean, cmax, cmax_x, cmean > 0 ? cmax / cmean : 0.0);
  // The seam must not stand out: worst column within 4x the mean column error.
  EXPECT_TRUE(cmean <= 0.0 || cmax / cmean < 4.0);
  EXPECT_TRUE(psnr > 20.0);
}

// The AUTO-SWITCH itself: a decode whose peak exceeds the budget must fall back
// to tiling and still produce an image, where it previously returned an error.
// VPIPE_VAE_BUDGET_MB stands in for the constrained box (a 64 GB machine can
// never take this branch on its own), and VPIPE_VAE_NO_TILE must restore the
// old hard failure -- otherwise this test would pass on a box where the decode
// simply fit and nothing was exercised.
TEST(flux2_smoke, vae_decode_auto_tiles_when_short)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }
  auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
  ASSERT_TRUE(vae != nullptr);

  const int gh = 48, gw = 48;                    // 768x768
  const int C = vae->config().dit_channels();
  SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
  ASSERT_TRUE(!z.empty());
  std::mt19937 rng(24680);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto* d = static_cast<_Float16*>(z.contents());
  for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
    d[i] = (_Float16)nd(rng);
  }
  const std::size_t need = vae->decode_peak_bytes(gh, gw);
  // Budget that the whole-image decode cannot meet but a window can.
  const int budget_mb = (int)std::max<std::size_t>(1, (need >> 20) / 3);
  ::setenv("VPIPE_VAE_BUDGET_MB", std::to_string(budget_mb).c_str(), 1);

  std::string err;
  SharedBuffer rgb = vae->decode(z, gh, gw, &err);
  const bool tiled_ok = !rgb.empty();

  // Same budget, tiling disabled -> must fail cleanly rather than silently fit.
  ::setenv("VPIPE_VAE_NO_TILE", "1", 1);
  std::string err2;
  SharedBuffer rgb2 = vae->decode(z, gh, gw, &err2);
  const bool refused = rgb2.empty();
  ::unsetenv("VPIPE_VAE_NO_TILE");
  ::unsetenv("VPIPE_VAE_BUDGET_MB");

  std::printf("[flux2_smoke] VAE auto-tile: need %zu MB, budget %d MB -> "
              "tiled %s, no-tile refused %s (%s)\n",
              need >> 20, budget_mb, tiled_ok ? "ok" : "FAILED",
              refused ? "yes" : "no", err2.c_str());
  EXPECT_TRUE(refused);      // proves the budget really was binding
  EXPECT_TRUE(tiled_ok);     // and that tiling rescued it
}

// The two-pass group norm must be a pure speedup: same normalization, only
// split so both passes saturate the GPU. It is not bit-identical (the
// partials are summed in a different order), so this is a rel-L2 bound -- if
// anything the block-partial sum is the better-conditioned of the two.
TEST(flux2_smoke, vae_decode_fast_gnorm_matches_legacy)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }

  const int gh = 24, gw = 32;                   // -> 384x512, non-square
  auto run = [&](bool legacy, double* ms) {
    // Read at load(), so it has to be set before the load, not the decode.
    if (legacy) { ::setenv("VPIPE_VAE_NO_FAST_GNORM", "1", 1); }
    else        { ::unsetenv("VPIPE_VAE_NO_FAST_GNORM"); }
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    ::unsetenv("VPIPE_VAE_NO_FAST_GNORM");
    std::vector<float> out;
    if (vae == nullptr) { return out; }
    const int C = vae->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    if (z.empty()) { return out; }
    std::mt19937 rng(4321);                     // same latent both runs
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

  double ms_fast = 0.0, ms_legacy = 0.0;
  const std::vector<float> v_fast   = run(/*legacy=*/false, &ms_fast);
  const std::vector<float> v_legacy = run(/*legacy=*/true,  &ms_legacy);
  ASSERT_TRUE(!v_fast.empty());
  ASSERT_TRUE(v_fast.size() == v_legacy.size());
  const double r = rel_l2_(v_fast.data(), v_legacy.data(), v_fast.size());
  std::printf("[flux2_smoke] VAE fast-gnorm vs legacy rel-L2 = %g "
              "(%dx%d; fast %.0f ms, legacy %.0f ms)\n",
              r, gh * 16, gw * 16, ms_fast, ms_legacy);
  EXPECT_TRUE(r < 2e-3);
  bool finite = true;
  for (float v : v_fast) { if (!std::isfinite(v)) { finite = false; break; } }
  EXPECT_TRUE(finite);
}

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

// Regression for the row-tiled im2col path at 1024 (VPIPE_VAE_NO_HWCONV -- the
// non-matrix-core M4 fallback and the conv_out fallback on every box). The
// matmul2d op corrupts output rows past M ~ 2^18-2^19 for the large-K 3x3 im2col
// GEMMs (LOWER than the small-K _mma_max_m=2^19 the chunk split assumes), so a
// 1024 decode on the im2col path went grey-BANDED until the band was safe-capped
// below the threshold. The band std check above is too coarse to catch a thin
// stripe, so decode the SAME latent through the default hw-conv path and the
// adaptive-tiled im2col path and rel-L2 the two full 1024x1024 images -- a banded
// region spikes rel-L2 far past the ~7e-4 hw-vs-im2col numerical floor. Uses the
// DEFAULT adaptive band (no BAND_ROWS override) so a broken safe-cap (which would
// pick a >2^18 band on the roomy test box) is caught. Gated on model + mma.
TEST(flux2_smoke, vae_decode_1024_im2col_tiled_matches_hwconv)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }
  const int gh = 64, gw = 64;                   // -> 1024x1024

  auto run = [&](bool no_hwconv) -> std::vector<float> {
    if (no_hwconv) { ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1);
      // ...and pin the fallback to im2col. Without this the conv autotune
      // sends some shapes to the on-chip gather, so the arm is a MIX and the
      // exact-0 that pins the im2col kernel itself is lost (it read 2.98e-4).
      ::setenv("VPIPE_VAE_NO_DIRECT_CONV", "1", 1); }
    else           { ::unsetenv("VPIPE_VAE_NO_HWCONV");
      ::unsetenv("VPIPE_VAE_NO_DIRECT_CONV"); }
    auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{});
    ::unsetenv("VPIPE_VAE_NO_HWCONV");
    std::vector<float> out;
    if (vae == nullptr) { return out; }
    const int C = vae->config().dit_channels();
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * gh * gw * 2);
    if (z.empty()) { return out; }
    std::mt19937 rng(1234);                     // same latent both legs
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * gh * gw; ++i) {
      d[i] = (_Float16)nd(rng);
    }
    std::string err;
    SharedBuffer rgb = vae->decode(z, gh, gw, &err);
    const std::size_t n = (std::size_t)3 * (gh * 16) * (gw * 16);
    if (rgb.empty() || rgb.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(rgb.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };

  const std::vector<float> hw  = run(/*no_hwconv=*/false);
  const std::vector<float> tld = run(/*no_hwconv=*/true);
  ASSERT_TRUE(!hw.empty());
  ASSERT_TRUE(hw.size() == tld.size());
  const double r = rel_l2_(hw.data(), tld.data(), hw.size());
  std::printf("[flux2_smoke] 1024 im2col-tiled vs hwconv rel-L2 = %.6g\n", r);
  EXPECT_TRUE(std::isfinite(r) && r < 3e-2);   // a banded region would be >> this
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
  // Latent grids to bench, as {h16, w16} (image pixels / 16). Default 512
  // and 1024 square. VPIPE_VAE_DEC_HW=1024x768 benches one real, NON-SQUARE
  // generation size instead -- the counterpart to VPIPE_VAE_ENC_HW, and the
  // shape a decode actually runs at.
  std::vector<std::pair<int, int>> grids = {{32, 32}, {64, 64}};
  if (const char* e = std::getenv("VPIPE_VAE_DEC_HW")) {
    const int hpx = std::atoi(e);
    const char* x = std::strchr(e, 'x');
    const int wpx = (x != nullptr) ? std::atoi(x + 1) : hpx;
    if (hpx >= 64 && wpx >= 64) {
      grids = {{hpx / 16, wpx / 16}};
    }
  }
  for (const auto& g : grids) {
    const int h16 = g.first, w16 = g.second;
    const std::size_t hw = (std::size_t)h16 * w16;
    SharedBuffer z = mc->make_shared_buffer((std::size_t)C * hw * 2);
    std::uint32_t s = 0x51ced00du;
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < (std::size_t)C * hw; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f);
    }
    std::string err;
    m->decode(z, h16, w16, &err);                    // warm
    double best = 1e18;
    for (int i = 0; i < 3; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      SharedBuffer o = m->decode(z, h16, w16, &err);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (!o.empty() && ms < best) { best = ms; }
    }
    std::printf("[flux2_smoke] decode %dhx%dw: %.1f ms (peak est %llu MB)\n",
                h16 * 16, w16 * 16, best,
                (unsigned long long)(m->decode_peak_bytes(h16, w16) >> 20));
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

  int H = 256, W = 256;                         // -> 16x16 latent grid
  // VPIPE_VAE_ENC_HW=1024x768 drives the encoder at a real generation size
  // (the encoder runs its widest convs at FULL resolution, so 256x256 hides
  // where its time actually goes).
  if (const char* e = std::getenv("VPIPE_VAE_ENC_HW")) {
    const int v = std::atoi(e);
    const char* x = std::strchr(e, 'x');
    const int v2 = (x != nullptr) ? std::atoi(x + 1) : 0;
    if (v >= 64) { H = W = v; }
    if (v >= 64 && v2 >= 64) { W = v2; }
  }
  auto run = [&](bool no_hw, double* ms) {
    if (no_hw) { ::setenv("VPIPE_VAE_NO_HWCONV", "1", 1);
      // Pin the fallback to im2col: the conv autotune would otherwise route
      // some shapes to the on-chip gather, making this arm a mix rather than
      // the im2col the test names.
      ::setenv("VPIPE_VAE_NO_DIRECT_CONV", "1", 1); }
    else       { ::unsetenv("VPIPE_VAE_NO_HWCONV");
      ::unsetenv("VPIPE_VAE_NO_DIRECT_CONV"); }
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

// VAE ENCODE golden at 1024: a deterministic normalized image [3,1024,1024] ->
// DiT-facing reference latent [128,64,64] vs the diffusers golden
// (dump_vae_encode_1024.py). This is the reference-conditioning latent an edit
// feeds the DiT; the DiT + decode both pass at 1024, so a mismatch here is the
// remaining place "block patches" can enter. VPIPE_FLUX2_ENCODE_GOLDEN = dir.
TEST(flux2_golden, vae_encode_1024_rel_l2)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_FLUX2_ENCODE_GOLDEN");
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
  const int H = (int)mo.at("H").as_int(0);
  const int W = (int)mo.at("W").as_int(0);
  const int C = (int)mo.at("dit_channels").as_int(0);
  const int lh = (int)mo.at("lh").as_int(0);
  const int lw = (int)mo.at("lw").as_int(0);

  const std::vector<float> img = read_f32_file_(g + "/img.f32");
  const std::vector<float> ref = read_f32_file_(g + "/z.f32");
  ASSERT_TRUE(img.size() == (std::size_t)3 * H * W);
  ASSERT_TRUE(ref.size() == (std::size_t)C * lh * lw);

  namespace fs = std::filesystem;
  std::string vdir = std::string(root) + "/vae";
  if (!fs::exists(fs::path(vdir) / "config.json")) { vdir = root; }
  auto vae = MetalFlux2Vae::load(vdir, mc, MetalFlux2Vae::Config{},
                                 /*with_encoder=*/true);
  ASSERT_TRUE(vae != nullptr);
  if (!vae->has_encoder()) {
    std::printf("[flux2_golden] no encoder in checkpoint -- skip\n");
    return;
  }

  SharedBuffer ib = mc->make_shared_buffer(img.size() * 2);
  { auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < img.size(); ++i) { d[i] = (_Float16)img[i]; } }
  SharedBuffer z = vae->encode(ib, H, W);
  ASSERT_TRUE(!z.empty());
  std::vector<float> got(ref.size());
  const auto* zp = static_cast<const _Float16*>(z.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)zp[i]; }

  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[flux2_golden] VAE encode rel-L2 = %.5f (%dx%d -> [%d,%d,%d])\n",
              r, H, W, C, lh, lw);
  EXPECT_TRUE(r < 0.02);
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
  // Resolution-parametrized profiling: VPIPE_FLUX2_BENCH_GRID=<n> sets a
  // square gh=gw=n token grid (n=64 -> 1024x1024), VPIPE_FLUX2_BENCH_TS the
  // conditioning length, VPIPE_FLUX2_BENCH_ITERS the (warm) forward count. With
  // VPIPE_FLUX2_DIT_PROFILE set this prints the per-section timing per forward.
  auto envi = [](const char* k, int d) {
    const char* e = std::getenv(k); return (e && *e) ? std::atoi(e) : d; };
  const int grid = envi("VPIPE_FLUX2_BENCH_GRID", 4);
  const int TS = envi("VPIPE_FLUX2_BENCH_TS", 8);   // a short fake prompt
  const int iters = envi("VPIPE_FLUX2_BENCH_ITERS", 1);
  const int gh = grid, gw = grid, img_seq = gh * gw;
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
  SharedBuffer vel;
  for (int it = 0; it < iters; ++it) {   // warm iterations (profile prints each)
    vel = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    ASSERT_TRUE(!vel.empty());
  }
  const std::size_t n = (std::size_t)img_seq * c.out_channels;
  ASSERT_TRUE(vel.byte_size() >= n * 2);
  EXPECT_TRUE(all_finite_(vel, n));
  std::printf("[flux2_smoke] DiT forward -> velocity [%d, %d] (finite)\n",
              img_seq, c.out_channels);
}

// BLOCK STREAMING MUST NOT CHANGE A SINGLE BIT.
//
// The streamed path reads every block off disk per forward into two
// reusable slots (shared/block-slots.h) instead of allocating one per
// block, and both of this model's stacks rebuild something after the
// read: the double blocks riffle ff.linear_in / ff_context.linear_in
// into the fused-SwiGLU order, and the single blocks slice
// to_qkv_mlp_proj into qkv + an interleaved gate|up. Any of that done
// even slightly differently from the load-time route is SILENT -- right
// shapes, plausible numbers, a slightly different picture -- so the bar
// is byte equality with the preloaded forward, not closeness.
//
// It reads the whole checkpoint twice, which is the price of the only
// check that can catch a wrong permutation.
TEST(flux2_smoke, streamed_matches_preloaded)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string tdir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";

  const int TS = 16, gh = 4, gw = 4, img_seq = gh * gw;

  std::vector<float> pre, str, grown;
  double fwd_s = 0.0;
  int nd_blocks = 0, ns_blocks = 0, kept = 0;
  // `grow` runs the streamed model with residency ON and takes the
  // SECOND forward, where some blocks come out of the resident set and
  // the rest are still refilled. That is the arm that exercises
  // promotion -- a block copied out of a slot -- and a mixed pass; with
  // growth off nothing is ever promoted and clone(copy=true) is dead
  // code.
  auto run = [&](bool stream, bool grow, std::vector<float>& out) -> bool {
    auto dit = MetalFlux2Transformer::load(
        tdir, mc, MetalFlux2Transformer::Config{}, stream);
    if (dit == nullptr) { return false; }
    // Growth OFF by default: a promoted block would be served out of
    // _double / _single on the later forwards, which is a different
    // question and would hide the refill this test exists to check.
    dit->set_residency_reserve(grow ? (2ull << 30) : 0);
    if (grow) { dit->set_residency_schedule(4); }
    const auto& c = dit->config();
    nd_blocks = c.n_double;
    ns_blocks = c.n_single;
    SharedBuffer ctx =
        mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
    SharedBuffer lat =
        mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
    if (ctx.empty() || lat.empty()) { return false; }
    std::mt19937 rng(20260823);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    auto* cp = static_cast<_Float16*>(ctx.contents());
    for (std::size_t i = 0; i < (std::size_t)TS * c.joint_dim; ++i) {
      cp[i] = (_Float16)ndist(rng);
    }
    auto* lp = static_cast<_Float16*>(lat.contents());
    for (std::size_t i = 0; i < (std::size_t)img_seq * c.in_channels; ++i) {
      lp[i] = (_Float16)ndist(rng);
    }
    SharedBuffer v;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      v = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    }
    fwd_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    kept = dit->resident_block_count();
    const std::size_t n = (std::size_t)img_seq * c.out_channels;
    if (v.empty() || v.byte_size() < n * 2) { return false; }
    const auto* p = static_cast<const _Float16*>(v.contents());
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return true;
  };

  ASSERT_TRUE(run(false, false, pre));
  const double pre_s = fwd_s;
  ASSERT_TRUE(run(true, false, str));
  std::printf("[flux2_smoke] %d+%d blocks -- forward: preloaded %.1f s, "
              "streamed %.1f s (slots %s)\n", nd_blocks, ns_blocks, pre_s,
              fwd_s,
              std::getenv("VPIPE_FLUX2_NO_SLOTS") != nullptr ? "OFF" : "on");

  ASSERT_TRUE(pre.size() == str.size() && !pre.empty());
  std::size_t diff = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < pre.size(); ++i) {
    if (pre[i] != str[i]) {
      ++diff;
      worst = std::max(worst, (double)std::fabs(pre[i] - str[i]));
    }
  }
  std::printf("[flux2_smoke] streamed vs preloaded: %zu of %zu differ "
              "(worst %.3e)\n", diff, pre.size(), worst);
  EXPECT_TRUE(diff == 0);

  ASSERT_TRUE(run(true, true, grown));
  ASSERT_TRUE(grown.size() == pre.size());
  std::size_t gdiff = 0;
  for (std::size_t i = 0; i < pre.size(); ++i) {
    if (pre[i] != grown[i]) { ++gdiff; }
  }
  std::printf("[flux2_smoke] resident-grown (%d of %d blocks kept) vs "
              "preloaded: %zu of %zu differ\n", kept,
              nd_blocks + ns_blocks, gdiff, pre.size());
  EXPECT_TRUE(gdiff == 0);
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

// ---- FLUX.2-klein-9b-kv recipe + cross-step reference K/V cache -----------
// Both tests run on whatever checkpoint VPIPE_FLUX2_TEST_MODEL_PATH names,
// including plain klein-9B, because what they assert is STRUCTURAL rather
// than numerical: that `klein_kv` actually changes the computation, and that
// the cache reproduces the recipe it accelerates. Those hold for any weights,
// so the -kv checkpoint is not needed to catch a broken mask, a mis-set
// stride or a stale splice. Verifying that the -kv WEIGHTS then produce the
// right image still needs a diffusers golden (VPIPE_FLUX2_EDIT_GOLDEN).
namespace {
struct KvFixture {
  Session sess;
  MetalCompute* mc = nullptr;
  std::unique_ptr<MetalFlux2Transformer> dit;
  std::mt19937 rng{11};
  SharedBuffer ctx, lat;
  int TS = 8, gh = 4, gw = 4, img_seq = 16;
  std::size_t n = 0;

  bool build(bool klein_kv) {
    const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
    if (root == nullptr || *root == '\0') { return false; }
    // VPIPE_FLUX2_KV_BENCH=<grid> runs the REAL generation geometry (32 -> a
    // 512x512 edit: 1024 image tokens against a 1024-token reference), where
    // the reference is half the sequence and the cache has something to save.
    // The default 4x4 keeps the correctness tests fast.
    if (const char* g = std::getenv("VPIPE_FLUX2_KV_BENCH")) {
      const int v = std::atoi(g);
      if (v >= 4) { gh = gw = v; TS = 512; }
    }
    mc = sess.metal_compute();
    if (mc == nullptr) { return false; }
    MetalFlux2Transformer::Config cfg;
    cfg.klein_kv = klein_kv;
    dit = MetalFlux2Transformer::load(std::string(root) + "/transformer", mc,
                                      cfg);
    if (dit == nullptr) { return false; }
    const auto& c = dit->config();
    img_seq = gh * gw;
    n = (std::size_t)img_seq * c.out_channels;
    ctx = mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
    lat = mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
    if (ctx.empty() || lat.empty()) { return false; }
    fill(ctx, (std::size_t)TS * c.joint_dim);
    fill(lat, (std::size_t)img_seq * c.in_channels);
    return true;
  }
  void fill(SharedBuffer& b, std::size_t cnt) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* p = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < cnt; ++i) { p[i] = (_Float16)nd(rng); }
  }
  MetalFlux2Transformer::RefImage make_ref(int rgh, int rgw) {
    MetalFlux2Transformer::RefImage r;
    const int ic = dit->config().in_channels;
    r.seq = rgh * rgw; r.grid_h = rgh; r.grid_w = rgw;
    r.latents = mc->make_shared_buffer((std::size_t)r.seq * ic * 2);
    fill(r.latents, (std::size_t)r.seq * ic);
    return r;
  }
  std::vector<float> vec(const SharedBuffer& b) const {
    std::vector<float> v(n);
    const auto* p = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)p[i]; }
    return v;
  }
};
}  // namespace

// What the cache is FOR: a cached step has to be cheaper than the step it
// replaces. Pricing it needs the cache toggled with the RECIPE HELD FIXED --
// turning klein_kv off would change the attention, so the two arms would not be
// computing the same thing and the ratio would be meaningless.
//
// Both arms also have to run in ONE process, ALTERNATING. A cross-process A/B
// on this box does not resolve the effect: model reload, the page cache and the
// SoC-power-gated GPU clock all move more than the cache does. MEASURED over
// four full generations per arm, the SAME arm landed anywhere from 32.9 s to
// 43.9 s, and reversing which arm ran first flipped the sign of the result.
// Alternating inside one process holds the model, the pages and the clock
// roughly constant across the pair.
//
// Set VPIPE_FLUX2_KV_BENCH=32 for the real 512x512-edit geometry; at the
// default 4x4 the reference is 12 tokens and there is nothing to save, so the
// test only asserts that the cache does not make things WORSE.
TEST(flux2_kv, cached_step_is_cheaper)
{
  KvFixture f;
  if (!f.build(/*klein_kv=*/true)) { return; }
  const int rg = (f.gh >= 32) ? f.gh : 3;      // reference grid
  auto refs = [&](unsigned seed) {
    f.rng.seed(seed);
    std::vector<MetalFlux2Transformer::RefImage> r;
    r.push_back(f.make_ref(rg, rg));
    return r;
  };
  MetalFlux2Transformer::KvCache kv;
  // Prime the cache (this call is the EXTRACT step, priced with the uncached
  // arm below, not with the cached one).
  const auto rp = refs(7);
  SharedBuffer p = f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh, f.gw,
                                      0.5f, -1.0f, rp, &kv);
  ASSERT_TRUE(!p.empty() && kv.populated());

  double c_best = 1e30, u_best = 1e30, c_sum = 0.0, u_sum = 0.0;
  const int reps = 3;
  for (int i = 0; i < reps; ++i) {
    for (int arm = 0; arm < 2; ++arm) {
      const auto rr = refs(7);
      const auto t0 = std::chrono::steady_clock::now();
      SharedBuffer v = f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh,
                                          f.gw, 0.5f, -1.0f, rr,
                                          arm == 0 ? &kv : nullptr);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      ASSERT_TRUE(!v.empty());
      if (arm == 0) { c_best = std::min(c_best, ms); c_sum += ms; }
      else          { u_best = std::min(u_best, ms); u_sum += ms; }
    }
  }
  std::printf("[flux2_kv] step cost: cached %.0f ms (mean %.0f), uncached "
              "%.0f ms (mean %.0f) -> %.2fx  (text %d + img %d + ref %d)\n",
              c_best, c_sum / reps, u_best, u_sum / reps, u_best / c_best,
              f.TS, f.img_seq, kv.ref_seq);
  EXPECT_TRUE(c_best <= u_best * 1.05);
}

// A cached step must reproduce the uncached recipe step it stands in for.
// The first call EXTRACTS (same arithmetic as uncached, plus lifting the
// reference band out), so it should match to the bit; the second REUSES the
// cache in place of recomputing, which is where a wrong stride, a stale band
// or a mis-scoped mask would show up as a diverged velocity.
TEST(flux2_kv, cached_matches_uncached)
{
  KvFixture f;
  if (!f.build(/*klein_kv=*/true)) { return; }
  // Re-seed per set so all three runs see the SAME reference latents --
  // make_ref draws from the fixture RNG, so without this each call would
  // condition on different references and nothing would be comparable.
  auto refs_of = [&](unsigned seed) {
    f.rng.seed(seed);
    std::vector<MetalFlux2Transformer::RefImage> r;
    r.push_back(f.make_ref(2, 3));
    r.push_back(f.make_ref(3, 2));
    return r;
  };
  const auto ra = refs_of(99), rb = refs_of(99), rc = refs_of(99);
  SharedBuffer a = f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh,
                                      f.gw, 0.5f, -1.0f, ra, nullptr);
  ASSERT_TRUE(!a.empty() && all_finite_(a, f.n));

  MetalFlux2Transformer::KvCache kv;
  SharedBuffer b0 = f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh,
                                       f.gw, 0.5f, -1.0f, rb, &kv);
  ASSERT_TRUE(!b0.empty());
  ASSERT_TRUE(kv.populated());     // the extracting step filled it
  SharedBuffer b1 = f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh,
                                       f.gw, 0.5f, -1.0f, rc, &kv);
  ASSERT_TRUE(!b1.empty() && all_finite_(b1, f.n));

  const std::vector<float> va = f.vec(a), v0 = f.vec(b0), v1 = f.vec(b1);
  const double r0 = rel_l2_(v0.data(), va.data(), f.n);
  const double r1 = rel_l2_(v1.data(), va.data(), f.n);
  std::printf("[flux2_kv] extract-vs-uncached rel-L2 %.3e, "
              "cached-vs-uncached rel-L2 %.3e (refs=%d)\n",
              r0, r1, kv.ref_seq);
  EXPECT_TRUE(r0 < 1e-6);          // same arithmetic
  EXPECT_TRUE(r1 < 5e-3);          // cache stands in for recomputation
}

// A changed geometry must REBUILD the cache rather than splice a band of the
// wrong length: the guard is what keeps a stale cache a slow step instead of
// a wrong one.
TEST(flux2_kv, geometry_change_rebuilds_cache)
{
  KvFixture f;
  if (!f.build(/*klein_kv=*/true)) { return; }
  std::vector<MetalFlux2Transformer::RefImage> r1, r2;
  r1.push_back(f.make_ref(2, 3));
  r2.push_back(f.make_ref(2, 3));
  r2.push_back(f.make_ref(3, 2));
  MetalFlux2Transformer::KvCache kv;
  ASSERT_TRUE(!f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh, f.gw,
                                  0.5f, -1.0f, r1, &kv).empty());
  const int first = kv.ref_seq;
  EXPECT_TRUE(first == 6);
  // A second reference changes ref_seq; the cache must refill, not reuse.
  ASSERT_TRUE(!f.dit->forward_dit(f.ctx, f.TS, f.lat, f.img_seq, f.gh, f.gw,
                                  0.5f, -1.0f, r2, &kv).empty());
  std::printf("[flux2_kv] cache ref_seq %d -> %d on geometry change\n",
              first, kv.ref_seq);
  EXPECT_TRUE(kv.ref_seq == 12);
}

// The recipe must not be a silent no-op: isolating the references and
// modulating them at timestep 0 has to move the velocity away from the plain
// fully-joint path. If these ever match, `klein_kv` is not reaching the
// forward and the -kv checkpoint would be running under the wrong attention.
TEST(flux2_kv, recipe_differs_from_plain)
{
  KvFixture plain, recipe;
  if (!plain.build(/*klein_kv=*/false)) { return; }
  if (!recipe.build(/*klein_kv=*/true)) { return; }
  // Same inputs on both: the fixtures seed identically, so ctx/lat/refs match.
  std::vector<MetalFlux2Transformer::RefImage> rp, rr;
  rp.push_back(plain.make_ref(2, 3));
  rr.push_back(recipe.make_ref(2, 3));
  SharedBuffer vp = plain.dit->forward_dit(plain.ctx, plain.TS, plain.lat,
                                           plain.img_seq, plain.gh, plain.gw,
                                           0.5f, -1.0f, rp);
  SharedBuffer vr = recipe.dit->forward_dit(recipe.ctx, recipe.TS, recipe.lat,
                                            recipe.img_seq, recipe.gh,
                                            recipe.gw, 0.5f, -1.0f, rr);
  ASSERT_TRUE(!vp.empty() && !vr.empty());
  const std::vector<float> a = plain.vec(vp), b = recipe.vec(vr);
  const double r = rel_l2_(b.data(), a.data(), plain.n);
  std::printf("[flux2_kv] recipe-vs-plain velocity rel-L2 %.4f\n", r);
  EXPECT_TRUE(r > 1e-3);
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
  // bf16 floor: with the DiT now bf16 (f16 overflowed on real conditioning),
  // the per-GEMM matmul2d-vs-steel difference compounds through 32 blocks' deep
  // residual stream to ~0.045 (kernel-level bf16 mma-vs-steel is ~1.6e-3 --
  // gemm_mma.correctness -- so this is accumulation, not a logic error). The
  // 3e-2 f16-era bound was only ever hit steel-vs-steel (~0) on the M4 stub; on
  // real M5 matrix cores it needs the same loosening the DiT golden got
  // (0.02->0.10). Still far under the sibling QIE velocity bound (0.2 / 60 blk).
  EXPECT_TRUE(std::isfinite(r) && r < 0.10);
}

// M5 matrix-core NAX attention A/B. The bf16 DiT's joint attention runs the
// matmul2d NAX flash-attention (attn_steel_nax_h_bd128_bf16) on matrix-core
// GPUs; VPIPE_FLUX2_NO_ATTN_NAX forces the non-nax bf16 steel attention
// (attn_steel_h_bd128_bf16). The GEMM path is the SAME (matmul2d) in both runs,
// so the velocity rel-L2 isolates the NAX-vs-steel attention kernel. Both are
// bf16 with f32-accumulate online softmax, so they match to the bf16 floor.
// The commit that added the bf16 nax attention could only compile it to its M4
// stub (no matrix cores) -- this is the missing M5 verification it asked for.
// Skips vacuously off matrix-core GPUs (there the nax path is capability-gated
// off and both runs take the same steel attention).
TEST(flux2_smoke, dit_attn_nax_matches_nonax)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string tdir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";

  const int TS = 64, gh = 16, gw = 16, img_seq = gh * gw;

  auto run = [&](bool no_nax) {
    if (no_nax) { ::setenv("VPIPE_FLUX2_NO_ATTN_NAX", "1", 1); }
    else        { ::unsetenv("VPIPE_FLUX2_NO_ATTN_NAX"); }
    auto dit = MetalFlux2Transformer::load(
        tdir, mc, MetalFlux2Transformer::Config{}, /*stream_blocks=*/true);
    ::unsetenv("VPIPE_FLUX2_NO_ATTN_NAX");
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

  const std::vector<float> v_nax   = run(/*no_nax=*/false);
  const std::vector<float> v_steel = run(/*no_nax=*/true);
  ASSERT_TRUE(!v_nax.empty());
  ASSERT_TRUE(v_nax.size() == v_steel.size());
  const double r = rel_l2_(v_nax.data(), v_steel.data(), v_nax.size());
  std::printf("[flux2_smoke] DiT nax-attn-vs-steel velocity rel-L2 = %.6g "
              "(seq=%d)\n", r, TS + img_seq);
  EXPECT_TRUE(std::isfinite(r) && r < 0.10);
}

// DiT step wall-clock (preloaded w4g64, bf16 compute). Warm once, best-of-5 min
// (least DVFS-throttled -- the M5 GPU clock tracks the shared SoC power budget,
// so measure cold). VPIPE_FLUX2_BENCH_SIDE overrides the square latent grid side
// (default 48 -> 2304 img tokens ~ 768px; 64 -> 4096 ~ 1024px). Pair with
// VPIPE_FLUX2_DIT_PROFILE=1 for the per-op (gemm / attn) breakdown, and A/B the
// paths across runs: default (nax matmul2d + nax attn) vs VPIPE_FLUX2_NO_MMA2
// (steel GEMM) vs VPIPE_FLUX2_NO_ATTN_NAX (steel attn). Gated on the model path.
TEST(flux2_smoke, dit_bench)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string tdir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";
  int side = 48;
  if (const char* s = std::getenv("VPIPE_FLUX2_BENCH_SIDE")) {
    const int v = std::atoi(s);
    if (v >= 8) { side = v; }
  }
  const int TS = 64, gh = side, gw = side, img_seq = gh * gw;

  // Preloaded (not streaming) so compute -- not per-block weight re-reads --
  // dominates the wall-clock. The w4g64 DiT is ~5.4 GB, fits a 16 GB box.
  auto dit = MetalFlux2Transformer::load(
      tdir, mc, MetalFlux2Transformer::Config{}, /*stream_blocks=*/false);
  ASSERT_TRUE(dit != nullptr);
  const auto& c = dit->config();
  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
  SharedBuffer lat =
      mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
  ASSERT_TRUE(!ctx.empty() && !lat.empty());
  std::mt19937 rng(4242);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto* cp = static_cast<_Float16*>(ctx.contents());
  for (std::size_t i = 0; i < (std::size_t)TS * c.joint_dim; ++i) {
    cp[i] = (_Float16)nd(rng);
  }
  auto* lp = static_cast<_Float16*>(lat.contents());
  for (std::size_t i = 0; i < (std::size_t)img_seq * c.in_channels; ++i) {
    lp[i] = (_Float16)nd(rng);
  }
  dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);       // warm
  double best = 1e18;
  for (int i = 0; i < 5; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer vel = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (!vel.empty() && ms < best) { best = ms; }
  }
  std::printf("[flux2_smoke] DiT step (side=%d, img=%d, seq=%d): %.1f ms\n",
              side, img_seq, TS + img_seq, best);
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
              quantized ? "quantized" : "bf16 vs f32 ref", 5 + 20);
  // The DiT runs BF16 (f16's 65504 range overflows on real conditioning
  // outliers -> NaN; see the flux2_edit test). bf16-vs-f32 drifts ~0.06 on this
  // tiny random golden (16 img tokens; QIE's bf16 DiT is cosine 0.9974 ~= 0.072
  // rel-L2 -- same family, same floor). Real-image cases average lower (the
  // 64x48 edit is ~0.018). Quantized adds group-affine error on top.
  EXPECT_TRUE(r < (quantized ? 0.40 : 0.10));
}

// EDITING path: feed the DiT a generated + reference token stream exactly as
// pipeline_flux2_klein does for image editing (hidden = [gen; ref];
// img_ids = [gen T=0 ; ref T=10]) and rel-L2 the generated-portion velocity
// against the diffusers golden (dump_edit_golden.py). VPIPE_FLUX2_EDIT_GOLDEN
// points at a size subdir (small / big). Isolates the reference-conditioning
// path (RoPE T band, ref embed, joint attention) from the encoder/template.
TEST(flux2_edit, dit_ref_velocity_rel_l2)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_FLUX2_EDIT_GOLDEN");
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
  const int gen_seq = (int)mo.at("gen_seq").as_int(0);
  const int rgh = (int)mo.at("rgh").as_int(0);
  const int rgw = (int)mo.at("rgw").as_int(0);
  const int ref_seq = (int)mo.at("ref_seq").as_int(0);
  const int TSq = (int)mo.at("text_seq").as_int(0);
  const int IN = (int)mo.at("in_channels").as_int(0);
  const int JD = (int)mo.at("joint_dim").as_int(0);
  const int OUT = (int)mo.at("out_channels").as_int(0);
  const float ts = (float)mo.at("timestep").as_real(0.0);

  const std::vector<float> gen = read_f32_file_(g + "/gen.f32");
  const std::vector<float> rfl = read_f32_file_(g + "/ref.f32");
  const std::vector<float> ctx = read_f32_file_(g + "/ctx.f32");
  const std::vector<float> ref = read_f32_file_(g + "/vel.f32");
  ASSERT_TRUE(gen.size() == (std::size_t)gen_seq * IN);
  ASSERT_TRUE(rfl.size() == (std::size_t)ref_seq * IN);
  ASSERT_TRUE(ctx.size() == (std::size_t)TSq * JD);
  ASSERT_TRUE(ref.size() == (std::size_t)gen_seq * OUT);

  auto dit = MetalFlux2Transformer::load(
      std::string(root) + "/transformer", mc,
      MetalFlux2Transformer::Config{}, /*stream=*/false);
  ASSERT_TRUE(dit != nullptr);

  SharedBuffer latb = mc->make_shared_buffer(gen.size() * 2);
  SharedBuffer ctxb = mc->make_shared_buffer(ctx.size() * 2);
  SharedBuffer refb = mc->make_shared_buffer(rfl.size() * 2);
  { auto* d = static_cast<_Float16*>(latb.contents());
    for (std::size_t i = 0; i < gen.size(); ++i) { d[i] = (_Float16)gen[i]; } }
  { auto* d = static_cast<_Float16*>(ctxb.contents());
    for (std::size_t i = 0; i < ctx.size(); ++i) { d[i] = (_Float16)ctx[i]; } }
  { auto* d = static_cast<_Float16*>(refb.contents());
    for (std::size_t i = 0; i < rfl.size(); ++i) { d[i] = (_Float16)rfl[i]; } }

  std::vector<MetalFlux2Transformer::RefImage> refs;
  MetalFlux2Transformer::RefImage ri;
  ri.latents = std::move(refb);
  ri.seq = ref_seq; ri.grid_h = rgh; ri.grid_w = rgw;
  refs.push_back(std::move(ri));

  SharedBuffer vel = dit->forward_dit(ctxb, TSq, latb, gen_seq, gh, gw, ts,
                                      -1.0f, refs);
  ASSERT_TRUE(!vel.empty());
  std::vector<float> got(ref.size());
  const auto* vp = static_cast<const _Float16*>(vel.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)vp[i]; }

  const double r = rel_l2_(got.data(), ref.data(), got.size());
  std::printf("[flux2_edit] DiT ref velocity rel-L2 = %.5f (gen %dx%d + ref "
              "%dx%d, seq %d)\n", r, gh, gw, rgh, rgw,
              TSq + gen_seq + ref_seq);
  // BF16 DiT (fixes the f16 overflow -> NaN on real conditioning): the real-
  // image edit drifts ~0.018 vs the f32 ref. Was NaN (f16 overflow) before bf16.
  EXPECT_TRUE(r < 0.05);
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

  bool declared_quant = false;
  const MetalQwenModel::Config c = flux2_encoder_config_(edir, &declared_quant);

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
  auto* cond = add_conditioner_(pl.get(), sess, src, root);
  auto t2iu = std::make_unique<GenerateImageStage>(
      &sess, "t2i", std::vector<InEdge>{{cond, 0}}, std::move(t2i_cfg));
  auto* t2i = static_cast<GenerateImageStage*>(pl->insert_stage(std::move(t2iu)));
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
    // iports {conditioning, neg_conditioning, model, sampler, scheduler, ref0,
    // ref1}; the conditioner feeds conditioning; wire ref0 (iport5) for the
    // with_ref case (model iport2 left unwired here).
    auto* cond = add_conditioner_(pl.get(), sess, src, root);
    std::vector<InEdge> edges{{cond, 0}};
    if (with_ref) {
      edges = std::vector<InEdge>{{cond, 0}, InEdge{nullptr, 0},
                                  InEdge{nullptr, 0}, InEdge{nullptr, 0},
                                  InEdge{nullptr, 0}, {rt_src, 0}};
    }
    auto t2iu = std::make_unique<GenerateImageStage>(&sess, "t2i", edges,
                                                   std::move(cfg));
    auto* t2i =
        static_cast<GenerateImageStage*>(pl->insert_stage(std::move(t2iu)));
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

// The vec4 adaLN / gate twins do the SAME arithmetic per element as the scalar
// kernels they replace -- only the thread mapping and the access width change --
// so the velocity must come out BIT-IDENTICAL. What this actually guards is the
// PLUMBING: the (N, total) -> (N/4, total/N) rewrite and the 4-alignment of
// every bound element offset. An unaligned vec4 access is undefined, so a
// wrong offset here would not be a rounding difference, it would be garbage.
TEST(flux2_smoke, vec4_elementwise_matches_scalar)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  const int grid = 4, TS = 8, img_seq = grid * grid;
  SharedBuffer ctx, lat;
  auto run = [&](bool v4) -> std::vector<std::uint16_t> {
    if (v4) { unsetenv("VPIPE_NO_ELT_V4"); }
    else    { setenv("VPIPE_NO_ELT_V4", "1", 1); }
    auto dit = MetalFlux2Transformer::load(
        tdir, mc, MetalFlux2Transformer::Config{});
    if (!dit) { return {}; }
    const auto& c = dit->config();
    if (ctx.empty()) {
      ctx = mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
      lat = mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
      std::mt19937 rng(4242);
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
    SharedBuffer v = dit->forward_dit(ctx, TS, lat, img_seq, grid, grid, 0.5f);
    const std::size_t n = (std::size_t)img_seq * c.out_channels;
    if (v.empty() || v.byte_size() < n * 2) { return {}; }
    std::vector<std::uint16_t> out(n);
    std::memcpy(out.data(), v.contents(), n * 2);
    return out;
  };
  const std::vector<std::uint16_t> a = run(true);
  const std::vector<std::uint16_t> b = run(false);
  unsetenv("VPIPE_NO_ELT_V4");
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { diff += (a[i] != b[i]) ? 1 : 0; }
  EXPECT_TRUE(diff == 0);
  std::printf("[flux2_smoke] vec4 vs scalar elementwise: %zu/%zu words differ\n",
              diff, a.size());
}

// ---- conv phase attribution: im2col vs GEMM ---------------------------
//
// The VAE decode profile shows the up-block resblock convs dominating on a
// GPU with no matrix cores, and after routing the conv GEMM through the
// simdgroup-MMA kernels `up3` was still ~23x off this box's ALU roofline.
// That is not a number the GEMM alone explains, so time the two halves of
// one conv separately at the real shape rather than reason about it.
//
// Shape is up3's inner conv at a 256x256 decode: [65536, 1152] x [128, 1152]^T.
// VPIPE_VAE_CONV_PHASE=<hw> picks another square output size.
TEST(flux2_smoke, vae_conv_phase_bench)
{
  if (std::getenv("VPIPE_VAE_CONV_PHASE") == nullptr) { return; }
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  int HW = std::atoi(std::getenv("VPIPE_VAE_CONV_PHASE"));
  if (HW < 32) { HW = 256; }
  const int H = HW, W = HW, cin = 128, cout = 128;
  const int M = H * W, K = 9 * cin, N = cout;

  auto lib_elt  = mc->load_library("llm_elementwise");
  auto lib_gemm = mc->load_library("dense_gemm");
  auto f_im2col = lib_elt.function("im2col_hwc_3x3_tiled_f16");
  auto f_gemm   = lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  auto f_scalar = lib_gemm.function("dense_gemm_bias_f16");
  auto f_dc64   = lib_gemm.function("conv3x3_gemm_s1_bn64_f16");
  auto f_dc128  = lib_gemm.function("conv3x3_gemm_s1_bn128_f16");
  ASSERT_TRUE(f_im2col.valid() && f_gemm.valid() && f_scalar.valid());

  auto act = mc->make_shared_buffer((std::size_t)M * cin * 2);
  auto col = mc->make_shared_buffer((std::size_t)M * K * 2);
  auto wgt = mc->make_shared_buffer((std::size_t)N * K * 2);
  auto outb = mc->make_shared_buffer((std::size_t)M * N * 2);
  ASSERT_TRUE(!act.empty() && !col.empty() && !wgt.empty() && !outb.empty());

  auto time_it = [&](const char* what, auto&& body) {
    double best = 1e18;
    for (int it = 0; it < 4; ++it) {
      auto st = mc->make_command_stream();
      auto e  = st.begin_compute();
      const auto t0 = std::chrono::steady_clock::now();
      body(e);
      e.end();
      std::string err;
      const bool ok = st.commit().wait_ok(&err);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      EXPECT_TRUE(ok);
      if (it > 0 && ms < best) { best = ms; }
    }
    std::printf("[flux2_smoke]   %-28s %.2f ms\n", what, best);
    return best;
  };

  std::printf("[flux2_smoke] conv phases at %dx%d cin=%d cout=%d "
              "(M=%d K=%d N=%d)\n", H, W, cin, cout, M, K, N);
  const double t_im = time_it("im2col only", [&](vpipe::metal_compute::ComputeEncoder& e) {
    e.set_function(f_im2col);
    e.set_buffer(0, act); e.set_buffer(1, col);
    e.set_constant(2, H); e.set_constant(3, W); e.set_constant(4, cin);
    e.set_constant(5, 0); e.set_constant(6, M);
    e.dispatch({(unsigned)K, (unsigned)M, 1}, {64, 1, 1});
  });
  const double t_gm = time_it("gemm only (simd MMA)", [&](vpipe::metal_compute::ComputeEncoder& e) {
    e.set_function(f_gemm);
    e.set_buffer(0, col); e.set_buffer(1, wgt); e.set_buffer(2, wgt);
    e.set_buffer(3, outb);
    e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
    e.set_constant(7, 0);
    e.dispatch({(unsigned)(((N + 63) / 64) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  });
  time_it("gemm only (scalar)", [&](vpipe::metal_compute::ComputeEncoder& e) {
    e.set_function(f_scalar);
    e.set_buffer(0, col); e.set_buffer(1, wgt); e.set_buffer(2, wgt);
    e.set_buffer(3, outb);
    e.set_constant(4, M); e.set_constant(5, N); e.set_constant(6, K);
    e.set_constant(7, 0);
    e.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
  });
  // The proposed alternative: no im2col at all. A 3x3 conv is the SUM of 9
  // 1x1 convs over shifted activations, i.e. 9 GEMMs of K=cin accumulating
  // into the same output. Same total FLOPs, but each GEMM reads the [M,cin]
  // activation directly (9x smaller, and warm across taps) instead of a
  // materialized [M,9*cin]. Timed here as 9 back-to-back K=cin GEMMs -- an
  // upper bound on the arithmetic, ignoring that a real implementation folds
  // the shift into the tap's address and accumulates in place.
  // The direct conv against the SAME GEMM it embeds: t_gm is the floor (the
  // MMA with a materialized operand), so direct - t_gm is what the on-chip
  // gather costs, and that is the number to optimize.
  auto direct = [&](const vpipe::metal_compute::ComputeFunction& f, int bn) {
    return [&, bn](vpipe::metal_compute::ComputeEncoder& e) {
      e.set_function(f);
      e.set_buffer(0, act); e.set_buffer(1, wgt); e.set_buffer(2, wgt);
      e.set_buffer(3, outb);
      e.set_constant(4, H);   e.set_constant(5, W);
      e.set_constant(6, cin); e.set_constant(7, cout);
      e.set_constant(8, H);   e.set_constant(9, W);
      e.set_constant(10, 0);
      e.dispatch({(unsigned)(((cout + bn - 1) / bn) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    };
  };
  if (f_dc64.valid())  { time_it("direct conv BN=64",  direct(f_dc64, 64)); }
  if (f_dc128.valid()) { time_it("direct conv BN=128", direct(f_dc128, 128)); }
  time_it("9x gemm K=cin (no im2col)", [&](vpipe::metal_compute::ComputeEncoder& e) {
    for (int tap = 0; tap < 9; ++tap) {
      e.set_function(f_gemm);
      e.set_buffer(0, act); e.set_buffer(1, wgt); e.set_buffer(2, wgt);
      e.set_buffer(3, outb);
      e.set_constant(4, cin); e.set_constant(5, N); e.set_constant(6, M);
      e.set_constant(7, 0);
      e.dispatch({(unsigned)(((N + 63) / 64) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    }
  });
  const double gflop = 2.0 * M * N * K / 1e9;
  const double gb_col = (double)M * K * 2 / 1e9;
  std::printf("[flux2_smoke]   conv = %.1f GFLOP, im2col writes %.2f GB "
              "-> im2col %.0f GB/s, gemm %.2f TFLOP/s\n",
              gflop, gb_col, gb_col / (t_im / 1e3),
              gflop / 1e3 / (t_gm / 1e3));
}


// ---- streamed residency, wired ------------------------------------------
//
// The end of the policy every DiT shares: a streamed block that is KEPT
// gets mlock'd through the manager's pool, and admission is gated on the
// pool having room for it. What this pins is the pair of properties that
// bound memory rather than the ones that make it fast.
//
//   1. THE RESULT DOES NOT MOVE. Keeping a block and streaming it must
//      produce the same velocity to the bit -- a resident block is the
//      same bytes read once instead of every step, so anything else is a
//      bug in the promotion (a buffer aliased where it should be copied,
//      a fused weight kept unfused).
//   2. NOTHING GROWS WITHOUT A CEILING. Wired memory is the one
//      allocation the kernel cannot take back, so the resident set must
//      stop at the pool's limit and the process must never pass it --
//      the failure mode is a panicked box, not a slow one.
//
// Run against a small pool on purpose: with the box's real pool a 9B DiT
// fits entirely and the gate never fires, which is the case that proves
// nothing.
// LAYER STREAMING MUST NOT CHANGE A SINGLE BIT, for the text encoder as
// much as for the DiT.
//
// declare_resources() claims this encoder with a floor -- what it holds
// if it streams -- and the resource phase may admit a graph on the
// strength of that. The claim is only honest if the encoder really can
// stream, and the trade is only acceptable if the answer does not move.
// Because a streamed layer is built by the SAME build_layer_ a resident
// load runs, the bar is BIT-IDENTICAL rather than close; anything else
// means the two paths read the checkpoint differently.
//
// Runs the {9,18,27} MULTI-LAYER tap FLUX.2 actually conditions on,
// which the single-tap encoders elsewhere do not cover: what streaming
// can break is the build/free/commit protocol around a tap, and a tap
// at layer 9 that is later freed is a different case from the last one.
// The two loads are SEQUENTIAL -- the first is destroyed before the
// second is built -- so this never holds two copies.
TEST(flux2_smoke, encoder_layer_streaming_matches_resident)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  const char* eov = std::getenv("VPIPE_FLUX2_ENC_DIR");
  const std::string edir = (eov != nullptr && *eov != '\0')
      ? std::string(eov) : (fs::path(root) / "text_encoder").string();
  if (!fs::exists(fs::path(edir) / "config.json")) { return; }

  const MetalQwenModel::Config base = flux2_encoder_config_(edir, nullptr);
  auto wts = MetalLlamaWeights::open_model(edir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer embed = wts->load("model.embed_tokens.weight", mc);
  ASSERT_TRUE(!embed.empty());

  const int EH = base.hidden, n = 8;
  const std::vector<int> taps_l = {8, 17, 26};
  auto make_x = [&]() {
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * EH * 2);
    const auto* tbl = static_cast<const std::uint8_t*>(embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      std::memcpy(xb + (std::size_t)i * EH * 2,
                  tbl + (std::size_t)(i + 100) * EH * 2, (std::size_t)EH * 2);
    }
    return x;
  };
  auto run = [&](bool stream, std::vector<std::uint16_t>& out) -> bool {
    MetalQwenModel::Config c = base;
    c.stream_layers = stream;
    c.pin_frac      = 0.0;   // pure streaming: every layer takes the path
    auto enc = MetalQwenModel::load(edir, mc, c);
    if (enc == nullptr) { return false; }
    // Without this the streamed arm can pass vacuously: load() DOWNGRADES
    // streaming to resident for anything with an output head, and a silent
    // downgrade would compare a model against itself.
    if (enc->streaming_layers() != stream) { return false; }
    genai::ContextManager* cm = enc->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer t = enc->forward_embeddings_taps(cid, make_x(), n, taps_l);
    cm->release(cid);
    if (t.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(t.contents());
    out.assign(p, p + t.byte_size() / 2);
    return true;
  };

  std::vector<std::uint16_t> want, got;
  ASSERT_TRUE(run(false, want));
  ASSERT_TRUE(run(true, got));
  ASSERT_TRUE(want.size() == got.size() && !want.empty());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i] != got[i]) { ++diff; }
  }
  std::printf("[flux2_smoke] encoder {9,18,27} taps, streamed vs resident: "
              "%zu of %zu words differ\n", diff, want.size());
  EXPECT_TRUE(diff == 0);
}

TEST(flux2_smoke, kept_blocks_are_wired_and_bounded_by_the_pool)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }
  const char* ddir = std::getenv("VPIPE_FLUX2_DIT_DIR");
  const std::string tdir = (ddir != nullptr && *ddir != '\0')
      ? std::string(ddir) : std::string(root) + "/transformer";

  const int TS = 64, gh = 8, gw = 8, img_seq = gh * gw;

  // Small enough that the gate binds well before the stack is resident,
  // and small enough to sit under RLIMIT_MEMLOCK on any box.
  const std::size_t kPool = 512ull << 20;
  mgr->set_wired_pool_bytes(kPool);
  const std::size_t limit = mgr->wired_pool_limit();

  auto dit = MetalFlux2Transformer::load(
      tdir, mc, MetalFlux2Transformer::Config{}, /*stream_blocks=*/true);
  if (dit == nullptr) { mgr->set_wired_pool_pct(0); return; }
  // A zero reserve is an ANSWER, not a default: this test frees the DiT
  // itself, so there is no peer to hold room clear for. Without one,
  // BlockResidency keeps growth off entirely.
  dit->set_residency_reserve(0);
  dit->set_residency_schedule(4);

  const auto& c = dit->config();
  SharedBuffer ctx = mc->make_shared_buffer((std::size_t)TS * c.joint_dim * 2);
  SharedBuffer lat =
      mc->make_shared_buffer((std::size_t)img_seq * c.in_channels * 2);
  ASSERT_TRUE(!ctx.empty() && !lat.empty());
  std::mt19937 rng(9182);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto* cp = static_cast<_Float16*>(ctx.contents());
  for (std::size_t i = 0; i < (std::size_t)TS * c.joint_dim; ++i) {
    cp[i] = (_Float16)nd(rng);
  }
  auto* lp = static_cast<_Float16*>(lat.contents());
  for (std::size_t i = 0; i < (std::size_t)img_seq * c.in_channels; ++i) {
    lp[i] = (_Float16)nd(rng);
  }

  auto forward = [&]() {
    std::vector<std::uint16_t> out;
    SharedBuffer v = dit->forward_dit(ctx, TS, lat, img_seq, gh, gw, 0.5f);
    const std::size_t n = (std::size_t)img_seq * c.out_channels;
    if (v.empty() || v.byte_size() < n * 2) { return out; }
    out.assign(static_cast<const std::uint16_t*>(v.contents()),
               static_cast<const std::uint16_t*>(v.contents()) + n);
    return out;
  };

  // Forward 1 streams every block and keeps what the pool allows;
  // forwards 2-3 run against a partly resident stack.
  const std::vector<std::uint16_t> a = forward();
  ASSERT_TRUE(!a.empty());
  const int kept1 = dit->resident_block_count();
  EXPECT_TRUE(mgr->wired_pool_used() <= limit);

  for (int i = 0; i < 2; ++i) {
    const std::vector<std::uint16_t> b = forward();
    ASSERT_TRUE(b.size() == a.size());
    // BIT-identical, not close: same weights, same inputs, same kernels.
    EXPECT_TRUE(std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
    EXPECT_TRUE(mgr->wired_pool_used() <= limit);
  }
  const int kept3 = dit->resident_block_count();
  std::printf("[flux2_smoke] wired residency: %d -> %d blocks kept, "
              "pool %zu of %zu MB\n", kept1, kept3,
              (std::size_t)(mgr->wired_pool_used() >> 20),
              (std::size_t)(limit >> 20));
  // The set may stop at zero on a box already under pressure -- that is
  // the policy working, not a failure. What must hold is that it never
  // outran the ceiling and that it did not shrink the answer.
  EXPECT_TRUE(dit->resident_block_bytes() <= limit);

  // And the pool comes BACK when the model goes. A DiT freed for the
  // vae-decode and reloaded per prompt would otherwise leak its whole
  // share of the budget every prompt until nothing could wire at all.
  dit.reset();
  EXPECT_TRUE(mgr->wired_pool_used() == 0u);
  mgr->set_wired_pool_pct(0);
}
