// Qwen-Image-2.1's DiT: the loader.
//
// The forward is not here yet. What this pins is the half that fails
// SILENTLY -- names and dtypes -- and it is cheap enough to run on a
// 16 GB box because streaming mode reads only the ~272 MB of
// model-level weights rather than the 14.2 GB stack.
//
// Two of these earn their place specifically:
//
//  * the ZERO-CENTERED text norm. txt_in.text_norm stores `scale - 1`,
//    so the loader has to add 1. Read as an ordinary gain it leaves the
//    text stream scaled by ~0, and the model still denoises -- it just
//    conditions on nothing, which no shape check can see.
//  * read_dims REFUSING the 20B sibling. "QwenImageTransformer2DModel"
//    is a different network whose config would otherwise load here
//    without complaint.
//
// Env: VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH (uses <root>/transformer).

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/weight-set.h"
#include "stages/model-quantize-stage.h"

#include <unistd.h>
#include "stages/model-memory.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

#define Q21D_REQUIRE(cond)                                                 \
  do {                                                                     \
    ASSERT_TRUE(cond);                                                     \
    if (!(cond)) { return; }                                               \
  } while (0)

std::string
dit_dir_()
{
  const char* r = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH");
  if (r == nullptr || *r == '\0') { return {}; }
  return (std::filesystem::path(r) / "transformer").string();
}

}  // namespace

TEST(qwen_image_21_dit, read_dims_reads_the_real_config)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  EXPECT_TRUE(cfg.n_layers == 32);
  EXPECT_TRUE(cfg.n_heads == 32);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.hidden == 4096);
  EXPECT_TRUE(cfg.in_channels == 64);
  EXPECT_TRUE(cfg.out_channels == 64);
  EXPECT_TRUE(cfg.txt_dim == 4096);
  EXPECT_TRUE(cfg.mlp_ratio == 3);
  EXPECT_TRUE(cfg.causal_condition);
  // The rotary axes sum to the head dim; a config where they do not is
  // one this forward cannot run.
  EXPECT_TRUE(cfg.axes[0] + cfg.axes[1] + cfg.axes[2] == cfg.head_dim);
  EXPECT_TRUE(cfg.axes[0] == 16 && cfg.axes[1] == 56 && cfg.axes[2] == 56);
}

// The 20B sibling's config must NOT load here.
TEST(qwen_image_21_dit, read_dims_refuses_the_dual_stream_sibling)
{
  namespace fs = std::filesystem;
  const fs::path d = fs::temp_directory_path() / "vpipe_qi21_cfg_test";
  std::error_code ec;
  fs::create_directories(d, ec);
  {
    std::ofstream o(d / "config.json");
    o << R"({"_class_name":"QwenImageTransformer2DModel","num_layers":60})";
  }
  MetalQwenImage21Transformer::Config cfg;
  EXPECT_FALSE(MetalQwenImage21Transformer::Config::read_dims(d.string(),
                                                              &cfg));
  // And a patched patch_size, which would silently change the stride.
  {
    std::ofstream o(d / "config.json");
    o << R"({"_class_name":"QwenImage21Transformer2DModel","patch_size":2})";
  }
  EXPECT_FALSE(MetalQwenImage21Transformer::Config::read_dims(d.string(),
                                                              &cfg));
  fs::remove_all(d, ec);
}

TEST(qwen_image_21_dit, loads_every_model_level_weight)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // STREAMING on purpose: this reads the model-level weights and none of
  // the 32 blocks, so it costs ~272 MB rather than 14.2 GB and runs on a
  // 16 GB box beside everything else.
  auto m = MetalQwenImage21Transformer::load(dir, mc, cfg,
                                             /*stream_blocks=*/true);
  Q21D_REQUIRE(m != nullptr);
  EXPECT_TRUE(m->streaming());
  EXPECT_TRUE(m->config().n_layers == 32);
  // The published checkpoint is bf16 throughout; a pack would carry
  // `.scales` beside its codes and set this.
  EXPECT_TRUE(m->quant_bits() == 0);
  std::printf("[qwen_image_21_dit] loaded: hidden %d, %d blocks, %s, "
              "trunk %.0f MB\n", m->config().hidden, m->config().n_layers,
              m->quant_bits() > 0 ? "quantized" : "bf16",
              (double)m->resident_bytes() / (1024.0 * 1024.0));
  // The trunk alone, with no block resident: the shared modulation is
  // 16384x4096 and dominates it, so a figure near zero would mean the
  // fixed weights did not load at all.
  EXPECT_TRUE(m->resident_bytes() > (200ull << 20));
  EXPECT_TRUE(m->resident_bytes() < (1024ull << 20));
}

// The +1. Checked against the checkpoint's own bytes rather than against
// a remembered constant.
TEST(qwen_image_21_dit, text_norm_is_stored_zero_centered)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto ws = WeightSet::open(dir, nullptr);
  Q21D_REQUIRE(ws != nullptr);
  const MetalLlamaWeights& src = ws->src();
  const auto* info = src.info("txt_in.text_norm.weight");
  Q21D_REQUIRE(info != nullptr);
  metal_compute::SharedBuffer raw = src.load("txt_in.text_norm.weight", mc);
  Q21D_REQUIRE(!raw.empty());

  // An ORDINARY norm gain is centred near 1; this one is not, so the +1
  // the loader adds is the right reading. MEASURED on the published
  // checkpoint: the raw mean is -0.807, i.e. an effective scale of about
  // 0.19 -- a real trained value, and nowhere near the 1.0 an ordinary
  // gain would sit at. The bound below rules out that reading rather
  // than pinning the number, which is free to move with a re-release.
  //
  // diffusers computes `weight + 1` in QwenImage21ZeroCenterRMSNorm; the
  // only question this test can answer locally is whether THIS file is
  // stored the way that class expects, and the mean answers it.
  double sum = 0.0;
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  Q21D_REQUIRE(n > 0);
  const auto* s = static_cast<const std::uint16_t*>(raw.contents());
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t u = (std::uint32_t)s[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    sum += f;
  }
  const double mean = sum / (double)n;
  std::printf("[qwen_image_21_dit] text_norm raw mean %.4f over %zu -> "
              "effective scale %.4f (an ordinary gain would be ~1.0)\n",
              mean, n, mean + 1.0);
  EXPECT_TRUE(mean < 0.5);
  // ...and the effective scale still has to be a usable positive number.
  EXPECT_TRUE(mean + 1.0 > 0.0);
}

// ---- the forward, with no reference -------------------------------

// Runs the whole 32-block stack over a deliberately small joint
// sequence. It proves the plumbing -- shapes, the segment dispatches,
// the streamed block path, finiteness -- and nothing about the
// numerics, which is the golden's job.
//
// STREAMING, so the stack is read one block at a time and never more
// than two are resident: the alternative is 14.2 GB on a 16 GB box.
TEST(qwen_image_21_dit, forward_runs_and_is_finite)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  auto m = MetalQwenImage21Transformer::load(dir, mc, cfg, true);
  Q21D_REQUIRE(m != nullptr);

  // 3 text tokens, one 2x2 condition image, 2 text, a 4x4 target.
  const std::vector<std::uint8_t> slot = {0, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  const std::vector<qi21::ImgBlock> blocks = {{1, 2, 2}, {1, 4, 4}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  EXPECT_TRUE(lay.target_len == 16);

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  metal_compute::SharedBuffer lat =
      rnd((std::size_t)lay.image_len * cfg.in_channels, 11u);
  metal_compute::SharedBuffer txt =
      rnd((std::size_t)slot.size() * cfg.txt_dim, 22u);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());

  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = 0.75f;
  std::string ferr;
  metal_compute::SharedBuffer out = m->forward(req, &ferr);
  if (out.empty()) {
    std::printf("[qwen_image_21_dit] forward failed: %s\n", ferr.c_str());
  }
  Q21D_REQUIRE(!out.empty());
  Q21D_REQUIRE(out.byte_size() >=
               (std::size_t)lay.target_len * cfg.out_channels * 2);

  // Finite, and not identically zero -- a stack whose blocks all
  // no-opped would still produce the right shape.
  const auto* d = static_cast<const std::uint16_t*>(out.contents());
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
  double sum2 = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t u = (std::uint32_t)d[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    if (!std::isfinite(f)) { finite = false; }
    sum2 += (double)f * (double)f;
  }
  const double rms = std::sqrt(sum2 / (double)n);
  std::printf("[qwen_image_21_dit] forward: joint %d -> [%d, %d], rms %.5f\n",
              lay.joint_len, lay.target_len, cfg.out_channels, rms);
  EXPECT_TRUE(finite);
  EXPECT_TRUE(rms > 1e-4);
}

// ---- against the reference, at a size that fits anywhere -----------

// The published DiT is 14.2 GB, so a CPU reference of it wants a bigger
// box than this bring-up always has. Nothing about the forward's MATH
// depends on the width, though: `--what dit` writes a 2-layer, 2-head
// instance of the SAME class with random weights, plus its output, and
// that exercises the shared modulation, the tanh gates, the
// block-causal segments, the centred rotary grid and the SwiGLU
// half-order exactly as 32 layers would.
//
// What it does not cover is the real checkpoint's names and dtypes --
// that is loads_every_model_level_weight's job. Between them the two
// cover the whole path.
TEST(qwen_image_21_dit, matches_reference_on_a_small_model)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));
  EXPECT_TRUE(cfg.n_layers == geti("layers", -1));
  EXPECT_TRUE(cfg.hidden == geti("hidden", -1));
  auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, false);
  Q21D_REQUIRE(m != nullptr);

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  EXPECT_TRUE(lay.joint_len == geti("joint", -1));
  EXPECT_TRUE(lay.target_len == geti("target", -1));

  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  };
  const std::vector<float> lat_f = read_f32("dit_latents");
  const std::vector<float> txt_f = read_f32("dit_txt");
  const std::vector<float> want = read_f32("dit_out");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty() && !want.empty());

  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      // round-to-nearest-even into bf16
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };
  metal_compute::SharedBuffer lat = up(lat_f);
  metal_compute::SharedBuffer txt = up(txt_f);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());

  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = (float)(o.contains("timestep")
                             ? o.at("timestep").as_real(0.75) : 0.75);
  std::string ferr;
  metal_compute::SharedBuffer out = m->forward(req, &ferr);
  if (out.empty()) {
    std::printf("[qwen_image_21_dit] forward failed: %s\n", ferr.c_str());
  }
  Q21D_REQUIRE(!out.empty());
  Q21D_REQUIRE(out.byte_size() >= want.size() * 2);

  const auto* d = static_cast<const std::uint16_t*>(out.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    std::uint32_t u = (std::uint32_t)d[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    const double e = (double)f - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[qwen_image_21_dit] velocity rel-L2 = %.6f over %zu "
              "(%d layers, joint %d)\n", rel, want.size(), cfg.n_layers,
              lay.joint_len);
  // bf16 weights and activations against an fp32 reference, through two
  // blocks: the floor is ~1e-2, not 1e-3.
  EXPECT_TRUE(rel < 0.05);
}

// ---- the prefix KV cache ------------------------------------------

// The cache is a pure OPTIMIZATION: a cached step must produce what an
// uncached one would, because it is the same arithmetic with the
// prefix's K and V read back instead of recomputed. So the test is an
// equivalence, not a golden -- and it runs against the small model, so
// it needs no big box either.
//
// It is only sound because causal_condition modulates the prefix from
// t=0, which makes those K and V independent of the step. A run that
// got that wrong would still denoise; it would just drift, which is why
// this compares EVERY step rather than only the first.
TEST(qwen_image_21_dit, cached_step_matches_uncached)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, false);
  Q21D_REQUIRE(m != nullptr);

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  Q21D_REQUIRE(lay.prefix_len > 0);

  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  };
  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };
  metal_compute::SharedBuffer lat = up(read_f32("dit_latents"));
  metal_compute::SharedBuffer txt = up(read_f32("dit_txt"));
  Q21D_REQUIRE(!lat.empty() && !txt.empty());

  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;

  auto to_vec = [&](const metal_compute::SharedBuffer& b, std::size_t n) {
    std::vector<float> v(n);
    const auto* d = static_cast<const std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      std::memcpy(&v[i], &u, 4);
    }
    return v;
  };
  const std::size_t n =
      (std::size_t)lay.target_len * cfg.out_channels;

  // Several timesteps, because the cache is filled at ONE of them and
  // reused at the others: a cache that quietly carried the fill step's
  // modulation would agree at step one and drift after.
  const float steps[] = {0.95f, 0.70f, 0.40f, 0.10f};
  MetalQwenImage21Transformer::KvCache kv;
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    req.timestep = steps[i];

    req.kv = nullptr;
    req.kv_mode = MetalQwenImage21Transformer::KvMode::kOff;
    std::string e1;
    metal_compute::SharedBuffer plain = m->forward(req, &e1);
    if (plain.empty()) {
      std::printf("[qwen_image_21_dit] uncached failed: %s\n", e1.c_str());
    }
    Q21D_REQUIRE(!plain.empty());

    req.kv = &kv;
    req.kv_mode = i == 0 ? MetalQwenImage21Transformer::KvMode::kFill
                         : MetalQwenImage21Transformer::KvMode::kCached;
    std::string e2;
    metal_compute::SharedBuffer cached = m->forward(req, &e2);
    if (cached.empty()) {
      std::printf("[qwen_image_21_dit] cached failed: %s\n", e2.c_str());
    }
    Q21D_REQUIRE(!cached.empty());

    const std::vector<float> a = to_vec(plain, n);
    const std::vector<float> b = to_vec(cached, n);
    double num = 0.0, den = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      const double d = (double)a[j] - (double)b[j];
      num += d * d;
      den += (double)a[j] * (double)a[j];
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    worst = std::max(worst, rel);
    std::printf("[qwen_image_21_dit] step %d (t=%.2f, %s): cached vs "
                "uncached rel-L2 %.3e\n", i, steps[i],
                i == 0 ? "fill" : "cached", rel);
  }
  // The fill step runs the same arithmetic as the uncached one, and a
  // cached step differs only in reading K/V back rather than recomputing
  // them -- bit-identical inputs through identical kernels. The bound is
  // loose only to leave room for a non-deterministic reduction order.
  EXPECT_TRUE(worst < 1e-3);
  // And the cache really was used: geometry stamped, one entry per block.
  EXPECT_TRUE(kv.populated());
  EXPECT_TRUE((int)kv.k.size() == cfg.n_layers);
  EXPECT_TRUE(kv.prefix == lay.prefix_len);
}

// A cache filled at one geometry must be REFUSED at another, not
// silently reused: the rows would line up and the picture would be
// wrong.
TEST(qwen_image_21_dit, cache_refuses_a_different_geometry)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  if (!std::filesystem::exists(gd / "dit" / "config.json")) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(
      (gd / "dit").string(), &cfg));
  auto m = MetalQwenImage21Transformer::load((gd / "dit").string(), mc, cfg,
                                             false);
  Q21D_REQUIRE(m != nullptr);

  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout({0, 0, 1, 1, 1, 1, 1}, {},
                                  {{1, 2, 2}, {1, 4, 4}}, &lay, &lerr));
  metal_compute::SharedBuffer lat =
      mc->make_shared_buffer((std::size_t)lay.image_len * cfg.in_channels * 2);
  metal_compute::SharedBuffer txt =
      mc->make_shared_buffer((std::size_t)7 * cfg.txt_dim * 2);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());
  std::memset(lat.contents(), 0, lat.byte_size());
  std::memset(txt.contents(), 0, txt.byte_size());

  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = 0.5f;
  // An empty cache used as if filled.
  MetalQwenImage21Transformer::KvCache kv;
  req.kv = &kv;
  req.kv_mode = MetalQwenImage21Transformer::KvMode::kCached;
  std::string e;
  EXPECT_TRUE(m->forward(req, &e).empty());
  EXPECT_FALSE(e.empty());
  // And a cache with no home at all.
  req.kv = nullptr;
  e.clear();
  EXPECT_TRUE(m->forward(req, &e).empty());
  EXPECT_FALSE(e.empty());
}

// ---- streaming equivalence ----------------------------------------

// STREAMED MUST EQUAL PRELOADED, BIT FOR BIT. Block streaming rebuilds
// every block from the checkpoint on the fly, and every way of getting
// that wrong is silent: a wrong permutation, a slot overwritten while
// the GPU still points at it, a prefetch with no free destination. None
// of them produce a buffer of the wrong size and none of them fail --
// they produce plausible numbers, which a rel-L2 bar against a
// reference is too slack to catch through two blocks.
//
// This also covers the boundary that makes the slot pair legal at all.
// A preloaded run encodes the whole stack into one command stream; a
// streamed one cannot, because the two slots are reused block after
// block and anything encoded-but-not-run still points at bytes the next
// read would overwrite. So a streamed forward commits per block and
// issues the next read between that commit and its wait.
//
// Run with VPIPE_QWEN_IMAGE21_NO_SLOTS=1 as well: that is the arm with
// one destination and no prefetch, and it is where a read issued into
// the block currently in use shows up -- as a GPU fault mid-stack
// rather than as a wrong answer.
TEST(qwen_image_21_dit, streamed_matches_preloaded)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  };
  const std::vector<float> lat_f = read_f32("dit_latents");
  const std::vector<float> txt_f = read_f32("dit_txt");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty());

  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };
  const float ts = (float)(o.contains("timestep")
                               ? o.at("timestep").as_real(0.75) : 0.75);

  // One forward, either way. Fresh input buffers per arm: the forward
  // does not write its inputs, but sharing them would hide it if it did.
  auto run = [&](bool stream_blocks, std::vector<std::uint16_t>* outv) {
    auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, stream_blocks);
    if (m == nullptr) { return false; }
    if (m->streaming() != stream_blocks) { return false; }
    metal_compute::SharedBuffer lat = up(lat_f);
    metal_compute::SharedBuffer txt = up(txt_f);
    if (lat.empty() || txt.empty()) { return false; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = ts;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty()) {
      std::printf("[qwen_image_21_dit] %s forward failed: %s\n",
                  stream_blocks ? "streamed" : "preloaded", ferr.c_str());
      return false;
    }
    const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
    if (out.byte_size() < n * 2) { return false; }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    outv->assign(d, d + n);
    return true;
  };

  std::vector<std::uint16_t> pre, str;
  Q21D_REQUIRE(run(false, &pre));
  Q21D_REQUIRE(run(true, &str));
  Q21D_REQUIRE(pre.size() == str.size() && !pre.empty());

  std::size_t diff = 0;
  for (std::size_t i = 0; i < pre.size(); ++i) {
    if (pre[i] != str[i]) { ++diff; }
  }
  std::printf("[qwen_image_21_dit] streamed vs preloaded: %zu/%zu words "
              "differ (%d layers)\n", diff, pre.size(), cfg.n_layers);
  EXPECT_TRUE(diff == 0);
}

// A PROMOTED BLOCK TAKES A DIFFERENT PATH THROUGH THE FORWARD, and
// nothing above reaches it. Residency keeps blocks as the run goes, so
// step 1 reads block L out of a slot and step 2 reads the same block out
// of `_blocks` -- a branch that only exists from the second forward on,
// which is exactly the regime a one-shot CLI run never enters.
//
// Two forwards at the same timestep must agree bit for bit, and with
// them the count of what was kept: a promotion that lands in the wrong
// slot, or a `held` test that reads an unfilled entry as filled, shows
// up here and nowhere else.
TEST(qwen_image_21_dit, promoted_blocks_match_streamed)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)n / 4);
    in.read(reinterpret_cast<char*>(v.data()), n);
    return v;
  };
  const std::vector<float> lat_f = read_f32("dit_latents");
  const std::vector<float> txt_f = read_f32("dit_txt");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty());
  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };
  const float ts = (float)(o.contains("timestep")
                               ? o.at("timestep").as_real(0.75) : 0.75);

  auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, true);
  Q21D_REQUIRE(m != nullptr);
  Q21D_REQUIRE(m->streaming());
  // DECLARE A RESERVE, or nothing is ever admitted. Residency refuses
  // every block until it has been told what else needs the box, and 0
  // is the real answer here -- no VAE decode runs beside this test.
  // Leaving it unsaid is not the same answer, and a run that forgets it
  // streams forever while looking perfectly healthy.
  m->set_residency_reserve(0);
  // Then the rates: this is what arms growth for the run.
  m->set_residency_schedule(4);

  std::vector<std::uint16_t> first, second;
  int kept_after_1 = 0, kept_after_2 = 0;
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
  for (int step = 0; step < 2; ++step) {
    metal_compute::SharedBuffer lat = up(lat_f);
    metal_compute::SharedBuffer txt = up(txt_f);
    Q21D_REQUIRE(!lat.empty() && !txt.empty());
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = ts;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty()) {
      std::printf("[qwen_image_21_dit] step %d failed: %s\n", step,
                  ferr.c_str());
    }
    Q21D_REQUIRE(!out.empty() && out.byte_size() >= n * 2);
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    (step == 0 ? first : second).assign(d, d + n);
    (step == 0 ? kept_after_1 : kept_after_2) = m->resident_block_count();
  }
  std::printf("[qwen_image_21_dit] kept %d/%d blocks after step 1, %d after "
              "step 2 (%.1f MB)\n", kept_after_1, cfg.n_layers, kept_after_2,
              (double)m->resident_block_bytes() / (1024.0 * 1024.0));
  Q21D_REQUIRE(first.size() == second.size() && !first.empty());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    if (first[i] != second[i]) { ++diff; }
  }
  std::printf("[qwen_image_21_dit] step 2 vs step 1: %zu/%zu words differ\n",
              diff, first.size());
  EXPECT_TRUE(diff == 0);

  // Giving the blocks back must leave the forward correct, not just
  // survivable -- a release that drops a block the loop still thinks is
  // held would read an empty one.
  if (kept_after_2 > 0) {
    // What the model SAYS it holds has to move by exactly what it gave
    // back. Blocks are counted once from their buffers; adding the
    // residency ledger on top would double every promoted one, and this
    // drop would come out at twice `freed` -- a number large enough to
    // make peers size against a model twice this one's weight, and
    // plausible enough that nothing else would notice.
    const std::uint64_t held_before = m->resident_bytes();
    const std::size_t freed = m->release_resident_blocks((std::size_t)-1);
    EXPECT_TRUE(freed > 0);
    EXPECT_TRUE(m->resident_block_count() == 0);
    const std::uint64_t held_after = m->resident_bytes();
    std::printf("[qwen_image_21_dit] resident_bytes %llu -> %llu, freed %zu\n",
                (unsigned long long)held_before,
                (unsigned long long)held_after, freed);
    EXPECT_TRUE(held_before - held_after == (std::uint64_t)freed);
    metal_compute::SharedBuffer lat = up(lat_f);
    metal_compute::SharedBuffer txt = up(txt_f);
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = ts;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    Q21D_REQUIRE(!out.empty() && out.byte_size() >= n * 2);
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    std::size_t d3 = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
      if (first[i] != d[i]) { ++d3; }
    }
    std::printf("[qwen_image_21_dit] after release: %zu/%zu words differ\n",
                d3, first.size());
    EXPECT_TRUE(d3 == 0);
  }
}

// WHAT THIS BOX ACTUALLY KEEPS, on the real 32-block stack.
//
// The equivalence tests above run a 2-block toy, where everything fits
// and growth is never refused -- so they prove the mechanism and say
// nothing about the policy. This one runs the checkpoint: 32 blocks of
// ~440 MB against whatever the box has, over three forwards, which is
// long enough for the per-forward cap to probe and double twice.
//
// It asserts that growth HAPPENS at all and that the forward stays
// correct as blocks move from slot to resident underneath it. The count
// is REPORTED rather than pinned, because it is a property of the box
// and the moment -- a number fixed here would fail on the 64 GB machine
// for being too small and on a loaded 16 GB one for being too large.
//
// It deliberately does NOT assert the count only ever rises. MEASURED
// on the M5 16 GB box: 18 blocks admitted on the first forward, then
// residency sampled its own pages, found 2 of 96 paged out, and gave
// one back -- settling at 17. That shed is the ratchet working, not a
// regression, and an assertion against it would be an assertion that
// the box must never be tight.
TEST(qwen_image_21_dit, residency_grows_on_the_real_stack)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  auto m = MetalQwenImage21Transformer::load(dir, mc, cfg, true);
  Q21D_REQUIRE(m != nullptr && m->streaming());
  m->set_residency_reserve(0);
  m->set_residency_schedule(3);

  const std::vector<std::uint8_t> slot = {0, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  const std::vector<qi21::ImgBlock> blocks = {{1, 2, 2}, {1, 4, 4}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
  std::vector<std::uint16_t> ref;
  for (int step = 0; step < 3; ++step) {
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 12345u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 999u);
    Q21D_REQUIRE(!lat.empty() && !txt.empty());
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    const auto t0 = std::chrono::steady_clock::now();
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (out.empty()) {
      std::printf("[qwen_image_21_dit] step %d failed: %s\n", step,
                  ferr.c_str());
    }
    Q21D_REQUIRE(!out.empty() && out.byte_size() >= n * 2);
    const int kept = m->resident_block_count();
    std::printf("[qwen_image_21_dit] step %d: %.0f ms, kept %d/%d blocks "
                "(%.1f MB resident)\n", step, ms, kept, cfg.n_layers,
                (double)m->resident_block_bytes() / (1024.0 * 1024.0));
    // Growth has to reach SOMETHING, or streaming is just a slow
    // preload -- but a later forward is free to shed what it measured
    // as not fitting, so this is a floor and not a monotonicity claim.
    EXPECT_TRUE(kept > 0);
    // And the answer must not change as blocks move from slot to
    // resident underneath it.
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    if (step == 0) {
      ref.assign(d, d + n);
    } else {
      std::size_t diff = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != d[i]) { ++diff; }
      }
      std::printf("[qwen_image_21_dit] step %d vs step 0: %zu/%zu differ\n",
                  step, diff, ref.size());
      EXPECT_TRUE(diff == 0);
    }
  }
}

// THE REAL GENERATION GEOMETRY, on the real bf16 stack.
//
// The residency test above runs a 25-row joint sequence, which answers
// the weight question and nothing else: at that size the activations
// are noise next to 13 GB of blocks. A 1024x1024 text-to-image is a
// 64x64 latent -- 4096 image rows, 1024 encoder slots at four rows each
// -- and that is where attention, the per-block scratch and the
// resident set have to fit in the same box at once.
//
// Gated on its own variable rather than the model path: it is the
// slowest thing in this file and it answers a question about a BOX, so
// it is run deliberately and its numbers quoted with the machine.
TEST(qwen_image_21_dit, real_1024_geometry_streams)
{
  const char* big = std::getenv("VPIPE_QWEN_IMAGE21_BIG");
  if (big == nullptr || *big == '\0') { return; }
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  const int px = std::atoi(big) > 0 ? std::atoi(big) : 1024;
  const int lg = px / 16;                    // the /16 VAE's latent grid
  Q21D_REQUIRE(lg > 0 && (lg % 2) == 0);

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  auto m = MetalQwenImage21Transformer::load(dir, mc, cfg, true);
  Q21D_REQUIRE(m != nullptr && m->streaming());
  m->set_residency_reserve(0);
  m->set_residency_schedule(4);

  // Text-to-image: a text run, then the target's own slots. Each set
  // entry is four joint rows, so a 64x64 latent is 1024 slots.
  const int n_text = 64;
  const int n_slot = lg * lg / 4;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)n_slot, 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  if (!qi21::build_layout(slot, {}, blocks, &lay, &lerr)) {
    std::printf("[qwen_image_21_dit] layout: %s\n", lerr.c_str());
  }
  Q21D_REQUIRE(lay.joint_len == n_text + lg * lg);

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
  std::vector<std::uint16_t> ref;
  for (int step = 0; step < 3; ++step) {
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 4242u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 77u);
    Q21D_REQUIRE(!lat.empty() && !txt.empty());
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    const auto t0 = std::chrono::steady_clock::now();
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (out.empty()) {
      std::printf("[qwen_image_21_dit] %dpx step %d failed: %s\n", px, step,
                  ferr.c_str());
    }
    Q21D_REQUIRE(!out.empty() && out.byte_size() >= n * 2);
    const auto mb = mc->memory_budget();
    std::printf("[qwen_image_21_dit] %dpx joint %d: step %d %.0f ms, kept "
                "%d/%d (%.1f MB); %llu MB compressed (self %llu), %llu MB "
                "swap, %llu MB phys avail%s\n", px, lay.joint_len,
                step, ms, m->resident_block_count(), cfg.n_layers,
                (double)m->resident_block_bytes() / (1024.0 * 1024.0),
                (unsigned long long)(mb.compressed >> 20),
                (unsigned long long)(mb.self_compressed >> 20),
                (unsigned long long)(mb.swap_used >> 20),
                (unsigned long long)(mb.available_physical >> 20),
                mb.paging() ? " PAGING" : "");
    // Finite, and unchanged as blocks are promoted underneath it.
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      if (!std::isfinite(f)) { ++bad; }
    }
    EXPECT_TRUE(bad == 0);
    if (step == 0) {
      ref.assign(d, d + n);
    } else {
      std::size_t diff = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != d[i]) { ++diff; }
      }
      std::printf("[qwen_image_21_dit] %dpx step %d vs 0: %zu/%zu differ\n",
                  px, step, diff, ref.size());
      EXPECT_TRUE(diff == 0);
    }
  }
}

// THE DECLARATION HAS TO TRACK THE WORKING SET, and the weight set
// cannot tell it what that is.
//
// A streaming DiT's retained trunk goes through tensor()/derived() and
// is cached in the WeightSet, so `stats().bytes` sees it. The blocks
// residency PROMOTES are the model's own buffers -- not cache entries
// -- so the set never sees them, and they are the large half. Revising
// a declaration from the set therefore reports a model that grew to
// several GB as though it had stayed at its trunk, and a peer sizing
// against that keeps weights the box no longer has room for. Every
// figure in that chain is one somebody really said, which is what makes
// it quiet.
//
// This measures the gap on the real stack: it is the reason
// GenerateImageStage asks the model rather than the set, and asks again
// after every generation rather than only at load.
TEST(qwen_image_21_dit, holding_is_not_visible_to_the_weight_set)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // Through the manager, so the set is the registered one a stage would
  // revise from -- not the private set the by-directory load opens.
  auto ws = vpipe::genai::open_weight_set(dir, &sess);
  Q21D_REQUIRE((bool)ws);
  auto m = MetalQwenImage21Transformer::load(ws, mc, cfg, true);
  Q21D_REQUIRE(m != nullptr && m->streaming());
  m->set_residency_reserve(0);
  m->set_residency_schedule(3);

  const std::vector<std::uint8_t> slot = {0, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  const std::vector<qi21::ImgBlock> blocks = {{1, 2, 2}, {1, 4, 4}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  for (int step = 0; step < 2; ++step) {
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 7u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 8u);
    Q21D_REQUIRE(!lat.empty() && !txt.empty());
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    Q21D_REQUIRE(!out.empty());
  }

  const std::size_t cached = ws->stats().bytes;
  const std::uint64_t held = m->resident_bytes();
  const int kept = m->resident_block_count();
  // The DECLARED floor, which is what a revision from the set gets
  // clamped up to -- trunk plus one slot pair, and the whole distance
  // between it and `held` is what the plan never hears about.
  static const std::vector<std::string_view> kStems = {"transformer_blocks."};
  const std::size_t floor =
      vpipe::model_memory::streaming_floor_bytes(dir, kStems);
  std::printf("[qwen_image_21_dit] floor %.1f MB; weight set sees %.1f MB; "
              "the model holds %.1f MB (%d/%d blocks promoted)\n",
              (double)floor / (1024.0 * 1024.0),
              (double)cached / (1024.0 * 1024.0),
              (double)held / (1024.0 * 1024.0), kept, cfg.n_layers);
  // Only meaningful once something was actually promoted -- on a box
  // with no room the two agree and there is nothing to see.
  if (kept > 0) {
    EXPECT_TRUE(held > cached);
    // And the gap is the promoted blocks, not a rounding difference.
    EXPECT_TRUE(held - cached >= (std::uint64_t)m->resident_block_bytes());
  }
}

// THE MATRIX-CORE GEMM AGAINST THE STEEL ONE, at a shape that reaches
// it.
//
// Nothing above verifies this path. The golden runs a 2-block toy whose
// joint sequence is 25 rows, and gemm_mma_ declines below 64 -- the
// 128-row tile does not amortize there -- so every existing correctness
// test exercises the steel arm and would pass with the matrix-core one
// arbitrarily wrong. This runs a geometry that engages it.
//
// NOT bit-identical, and it should not be: the tiles reduce in a
// different order, so the two arms differ wherever a bf16 intermediate
// rounds the other way. What separates that from a wrong tile, a wrong
// N-region or a wrong operand binding -- each of which also yields a
// full-sized buffer of plausible numbers -- is the SHAPE of the
// divergence, and it was measured rather than assumed.
//
// MEASURED, per block, from VPIPE_QWEN_IMAGE21_TRACE run on both arms:
// 2.9e-3 after block 0, rising smoothly to 2.6e-2 by block 12 and
// ~3-5e-2 at block 31. 2.9e-3 is the bf16 epsilon (2^-8 = 3.9e-3), so
// one block's disagreement is one rounding of the intermediate, and the
// rest is that compounding through a residual stack. A wrong tile does
// not look like this -- it puts the error in at block 0 and it does not
// start at the representational floor.
//
// And the arms were checked against TRUTH, which is the test that does
// not depend on reading a growth curve: on the dit_wide golden at the
// real widths, steel is 0.022126 from the fp32 reference and the
// matrix-core arm 0.022005 -- the same distance, marginally closer. So
// the number below is two arms each ~2.2e-2 from the reference, not one
// arm drifting from the other. The bar is set above that and well under
// the O(1) a broken tile gives.
TEST(qwen_image_21_dit, mma_matches_steel)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }   // nothing to compare
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // 256x256 -> a 16x16 latent, 64 slots, 256 image rows; with 64 text
  // rows that is a 320-row joint, comfortably past the 64-row floor.
  const int lg = 16;
  const int n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  Q21D_REQUIRE(lay.joint_len == n_text + lg * lg);

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;

  // The arm is chosen at LOAD, from the environment, so the two arms
  // are two loads in one process -- which also keeps the box, its
  // clocks and its page cache the same for both.
  auto run = [&](bool steel, std::vector<float>* outv, bool* engaged) {
    if (steel) { ::setenv("VPIPE_QWEN_IMAGE21_NO_MMA2", "1", 1); }
    else { ::unsetenv("VPIPE_QWEN_IMAGE21_NO_MMA2"); }
    auto m = MetalQwenImage21Transformer::load(dir, mc, cfg, true);
    if (m == nullptr) { return false; }
    *engaged = m->uses_mma2();
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 31u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 57u);
    if (lat.empty() || txt.empty()) { return false; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < n * 2) { return false; }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    outv->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      (*outv)[i] = f;
    }
    return true;
  };

  std::vector<float> a, b;
  bool steel_engaged = true, mma_engaged = false;
  Q21D_REQUIRE(run(true, &a, &steel_engaged));
  Q21D_REQUIRE(run(false, &b, &mma_engaged));
  ::unsetenv("VPIPE_QWEN_IMAGE21_NO_MMA2");
  // The arms have to actually BE the two arms -- a run where the
  // matrix-core path never engaged would compare steel against steel
  // and pass for the wrong reason.
  EXPECT_TRUE(!steel_engaged);
  EXPECT_TRUE(mma_engaged);
  Q21D_REQUIRE(a.size() == b.size() && !a.empty());

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = (double)b[i] - (double)a[i];
    num += e * e;
    den += (double)a[i] * (double)a[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[qwen_image_21_dit] mma vs steel: rel-L2 %.3e over %zu "
              "(joint %d, %d layers)\n", rel, a.size(), lay.joint_len,
              cfg.n_layers);
  // bf16 intermediates compounding through 32 blocks, with both arms
  // ~2.2e-2 from the fp32 reference. See the header for why this is the
  // floor and not a slack bar.
  EXPECT_TRUE(rel < 0.06);
  EXPECT_TRUE(den > 0.0);
}

// THE MATRIX-CORE GEMM AGAINST THE fp32 REFERENCE.
//
// `matches_reference_on_a_small_model` cannot reach this path: its
// joint sequence is 25 rows and gemm_mma_ declines below 64, so it
// verifies the STEEL arm and would pass with the matrix-core one
// arbitrarily wrong -- which is the arm every real generation takes.
//
// `dit_wide` is the same 2-layer random instance at the real WIDTHS
// (hidden 4096, ffn 12288) over a 12x12 target, so the joint is 153
// rows and K takes both values the model has. That is what selects the
// tile: 4096 for qkv/out/ff-up takes the 128x128 one, 12288 for ff-down
// takes the deep 128x256. A golden at hidden 256 would exercise only
// the first and leave the deep tile unverified.
//
// Dumped by:
//   dump_qwen_image21_golden.py --what dit --dit-grid 12 \
//       --dit-tag dit_wide --dit-heads 32
TEST(qwen_image_21_dit, mma_matches_reference_at_real_widths)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit_wide.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit_wide").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, false);
  Q21D_REQUIRE(m != nullptr);

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  // The whole point of this golden: past the row floor, so the
  // matrix-core arm is the one under test.
  Q21D_REQUIRE(lay.joint_len >= 64);
  if (mc->supports_matrix_cores()) { EXPECT_TRUE(m->uses_mma2()); }

  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff nn = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)nn / 4);
    in.read(reinterpret_cast<char*>(v.data()), nn);
    return v;
  };
  // The data files carry the tag too, so a second geometry cannot
  // overwrite the first's numbers while leaving its manifest standing.
  const std::vector<float> lat_f = read_f32("dit_wide_latents");
  const std::vector<float> txt_f = read_f32("dit_wide_txt");
  const std::vector<float> want = read_f32("dit_wide_out");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty() && !want.empty());

  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };
  metal_compute::SharedBuffer lat = up(lat_f);
  metal_compute::SharedBuffer txt = up(txt_f);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());

  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = (float)(o.contains("timestep")
                             ? o.at("timestep").as_real(0.75) : 0.75);
  std::string ferr;
  metal_compute::SharedBuffer out = m->forward(req, &ferr);
  if (out.empty()) {
    std::printf("[qwen_image_21_dit] wide forward failed: %s\n", ferr.c_str());
  }
  Q21D_REQUIRE(!out.empty());
  Q21D_REQUIRE(out.byte_size() >= want.size() * 2);

  const auto* d = static_cast<const std::uint16_t*>(out.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    std::uint32_t u = (std::uint32_t)d[i] << 16;
    float f;
    std::memcpy(&f, &u, 4);
    const double e = (double)f - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[qwen_image_21_dit] mma vs fp32 reference: rel-L2 %.6f over "
              "%zu (joint %d, hidden %d, ffn %d)\n", rel, want.size(),
              lay.joint_len, cfg.hidden, cfg.hidden * cfg.mlp_ratio);
  // bf16 weights and activations against an fp32 reference through two
  // blocks -- the same bar the narrow golden is held to.
  EXPECT_TRUE(rel < 0.05);
}

// A BOX WITHOUT MATRIX CORES MUST NOT LAND ON A SCALAR KERNEL.
//
// M5 and later take matmul2d for the GEMMs and the NAX flash kernel for
// attention. Everything before that -- an M4 -- declines both, and what
// it declines TO is the whole of its performance. This pins that the
// decline goes to the tiled kernels and not to the naive ones.
//
// The strong form of the claim is structural and holds on every box:
// this model never LOADS a scalar GEMM or a scalar attention.
// `dense_gemm_bias_f16` (the 16x16 one-thread-per-output GEMM that the
// Qwen-Image-Edit sibling keeps as a last resort) and `sdpa_full_f16`
// are not referenced anywhere in it, and load() refuses outright if the
// tiled entries are missing rather than substituting something worse.
// Attention goes further and refuses inside forward(), because the
// block-causal decomposition IS the masking and a dense scalar kernel
// would answer a different question.
//
// What is left for a test is the part that could regress silently: that
// the WIDE tiles are loaded and chosen, and that the arm is still right.
TEST(qwen_image_21_dit, no_matrix_cores_stays_on_tiled_kernels)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit_wide.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit_wide").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff nn = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)nn / 4);
    in.read(reinterpret_cast<char*>(v.data()), nn);
    return v;
  };
  const std::vector<float> lat_f = read_f32("dit_wide_latents");
  const std::vector<float> txt_f = read_f32("dit_wide_txt");
  const std::vector<float> want = read_f32("dit_wide_out");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty() && !want.empty());
  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };

  // The M4 arm, forced. Every tile must give the SAME answer as the
  // others and as the reference -- they are one piece of arithmetic
  // over different tilings, and a tile that changed the answer would be
  // a tile that dropped part of the output.
  //
  // BOTH flags, because they are two mechanisms. On a real M4
  // `supports_matrix_cores()` is false and that one answer turns off
  // the GEMM tiles AND the NAX attention kernel; the env overrides are
  // per-feature, so setting only NO_MMA2 leaves the NAX attention on
  // and emulates a machine that does not exist.
  for (int tile = 0; tile <= 2; ++tile) {
    ::setenv("VPIPE_QWEN_IMAGE21_NO_MMA2", "1", 1);
    ::setenv("VPIPE_QWEN_IMAGE21_NO_NAX_ATTN", "1", 1);
    ::setenv("VPIPE_QWEN_IMAGE21_GEMM_TILE",
             std::to_string(tile).c_str(), 1);
    auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, false);
    if (m == nullptr) {
      ::unsetenv("VPIPE_QWEN_IMAGE21_NO_MMA2");
      ::unsetenv("VPIPE_QWEN_IMAGE21_NO_NAX_ATTN");
      ::unsetenv("VPIPE_QWEN_IMAGE21_GEMM_TILE");
    }
    Q21D_REQUIRE(m != nullptr);
    // The matrix-core arms are off, which is the situation under test.
    EXPECT_TRUE(!m->uses_mma2());
    EXPECT_TRUE(!m->uses_attn_nax());
    EXPECT_TRUE(m->gemm_tile() == tile);

    metal_compute::SharedBuffer lat = up(lat_f);
    metal_compute::SharedBuffer txt = up(txt_f);
    Q21D_REQUIRE(!lat.empty() && !txt.empty());
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = (float)(o.contains("timestep")
                               ? o.at("timestep").as_real(0.75) : 0.75);
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    Q21D_REQUIRE(!out.empty() && out.byte_size() >= want.size() * 2);
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      const double e = (double)f - (double)want[i];
      num += e * e;
      den += (double)want[i] * (double)want[i];
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("[qwen_image_21_dit] no-mma tile %d: rel-L2 %.6f\n", tile,
                rel);
    EXPECT_TRUE(rel < 0.05);
  }
  ::unsetenv("VPIPE_QWEN_IMAGE21_NO_MMA2");
  ::unsetenv("VPIPE_QWEN_IMAGE21_NO_NAX_ATTN");
  ::unsetenv("VPIPE_QWEN_IMAGE21_GEMM_TILE");

  // And the DEFAULT on such a box is a wide tile, not the 32x32 one --
  // which is what this model shipped with, because it was the only
  // dense entry it loaded.
  ::setenv("VPIPE_QWEN_IMAGE21_NO_MMA2", "1", 1);
  auto def = MetalQwenImage21Transformer::load(ckpt, mc, cfg, false);
  ::unsetenv("VPIPE_QWEN_IMAGE21_NO_MMA2");
  Q21D_REQUIRE(def != nullptr);
  std::printf("[qwen_image_21_dit] no-mma default tile = %d\n",
              def->gemm_tile());
  EXPECT_TRUE(def->gemm_tile() >= 1);
}

// A PIPELINE STOP HAS TO LAND WITHIN ABOUT ONE BLOCK, not one forward.
//
// The stop is cooperative and checked at the top of every block. What
// bounds the latency is therefore how long a block takes, plus whatever
// is already in flight when the flag flips -- on the streamed path a
// block's read and its committed GPU work, on the preloaded path the
// command buffers Metal already has queued.
//
// This is a LATENCY test, not a throughput one, so it states the
// number it measured and asserts only a bound loose enough to survive a
// clamped box: a whole 32-block forward at this geometry is seconds,
// and the point is that a stop does not wait for one.
TEST(qwen_image_21_dit, stop_lands_within_about_one_block)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  auto m = MetalQwenImage21Transformer::load(dir, mc, cfg, true);
  Q21D_REQUIRE(m != nullptr);
  m->set_residency_reserve(0);
  m->set_residency_schedule(2);

  // 512x512: a 32x32 latent, 256 slots, 1024 image rows -- big enough
  // that a block is real work and a whole forward is clearly longer
  // than the bound below.
  const int lg = 32, n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  metal_compute::SharedBuffer lat =
      rnd((std::size_t)lay.image_len * cfg.in_channels, 11u);
  metal_compute::SharedBuffer txt =
      rnd((std::size_t)lay.text_len * cfg.txt_dim, 12u);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());
  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = 0.75f;

  // First, how long an UNINTERRUPTED forward takes -- the thing a stop
  // must not have to wait for.
  std::string ferr;
  const auto w0 = std::chrono::steady_clock::now();
  metal_compute::SharedBuffer full = m->forward(req, &ferr);
  const double whole_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - w0).count();
  Q21D_REQUIRE(!full.empty());

  // GIVE THE BLOCKS BACK before the stopped run, so every block streams
  // again. That is the SLOWER arm on purpose: a resident block is
  // encoded into a stream that is not committed until the end, so a
  // stop there abandons work that never started and returns in
  // milliseconds. A streamed block commits and waits, so its stop has
  // to sit out whatever is already on the GPU -- which is the latency
  // worth bounding.
  m->release_resident_blocks((std::size_t)-1);

  // Now stop it partway. The flag flips on the Nth check rather than on
  // a timer, so the measurement is "how long after the stop was visible
  // did forward return", with no sleep to confuse it.
  const int trip_at = cfg.n_layers / 2;
  int seen = 0;
  std::chrono::steady_clock::time_point tripped{};
  m->set_stream_stop([&]() {
    ++seen;
    if (seen < trip_at) { return false; }
    if (tripped.time_since_epoch().count() == 0) {
      tripped = std::chrono::steady_clock::now();
    }
    return true;
  });
  // Left DEFAULT-CONSTRUCTED, exactly as the stage declares it: a stop
  // must not write a reason into it, and "stayed empty" is the test.
  std::string serr;
  const auto s0 = std::chrono::steady_clock::now();
  metal_compute::SharedBuffer stopped = m->forward(req, &serr);
  const auto s1 = std::chrono::steady_clock::now();
  m->set_stream_stop({});

  Q21D_REQUIRE(tripped.time_since_epoch().count() != 0);
  const double latency_ms =
      std::chrono::duration<double, std::milli>(s1 - tripped).count();
  const double total_ms =
      std::chrono::duration<double, std::milli>(s1 - s0).count();
  std::printf("[qwen_image_21_dit] stop at block %d/%d: returned %.1f ms "
              "after the flag (%.0f ms into the forward; a whole forward is "
              "%.0f ms)\n", trip_at, cfg.n_layers, latency_ms, total_ms,
              whole_ms);

  // A STOP IS NOT A FAILURE: empty, and with no reason, which is how
  // the stage tells the two apart and avoids logging a user stop as a
  // fault.
  EXPECT_TRUE(stopped.empty());
  EXPECT_TRUE(serr.empty());
  // It must not have had to wait out the rest of the stack. Half the
  // blocks remained, so anything near `whole_ms/2` would mean the check
  // is not doing its job; the bound is deliberately loose.
  EXPECT_TRUE(latency_ms < whole_ms / 4.0);
}

// THE INT8 GEMM TIER: that it engages, and what it costs in accuracy.
//
// This is not a weight quantization -- the checkpoint is untouched.
// The bf16 activation is quantized per (row, group of 64) on the fly
// and the dense weight per (out-channel, group), and the product runs
// on the matrix units' int8 pipe. So what it risks is the rounding of
// one matmul, not stored precision.
//
// It was DECLARED and never dispatched: `Config::i8_gemm` was read by
// nobody and `_i8` had no uses. A tier in that state costs nothing and
// does nothing, and no timing test would notice -- which is why this
// asserts ENGAGEMENT first and accuracy second.
//
// The tier's own floor is M >= 1024, so the 153-row wide golden does
// not reach it in a normal run; VPIPE_I8_GEMM_MIN_M lowers the floor so
// the arithmetic can be checked against the fp32 reference at all. The
// per-element quantization is the same at any M, so what this measures
// is the tier's accuracy, not the accuracy of the shape it ran on.
TEST(qwen_image_21_dit, i8_gemm_engages_and_stays_accurate)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit_wide.json");
  if (!mf) { return; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return;
  }
  if (!man.is_object()) { return; }
  auto o = man.as_object();
  auto arr = [&](const char* k) {
    std::vector<int> v;
    if (!o.contains(k)) { return v; }
    const FlexData f = o.at(k);
    if (!f.is_array()) { return v; }
    auto a = f.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      v.push_back((int)a.at(i).as_int(0));
    }
    return v;
  };
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }   // the tier needs them
  MetalQwenImage21Transformer::Config cfg;
  const std::string ckpt = (gd / "dit_wide").string();
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg));

  const std::vector<int> slot_i = arr("slot");
  const std::vector<int> shapes = arr("img_shapes");
  Q21D_REQUIRE(!slot_i.empty() && shapes.size() % 3 == 0);
  std::vector<std::uint8_t> slot(slot_i.begin(), slot_i.end());
  std::vector<qi21::ImgBlock> blocks;
  for (std::size_t i = 0; i + 2 < shapes.size(); i += 3) {
    blocks.push_back(qi21::ImgBlock{shapes[i], shapes[i + 1], shapes[i + 2]});
  }
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  auto read_f32 = [&](const char* nm) {
    std::ifstream in(gd / (std::string(nm) + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff nn = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)nn / 4);
    in.read(reinterpret_cast<char*>(v.data()), nn);
    return v;
  };
  const std::vector<float> lat_f = read_f32("dit_wide_latents");
  const std::vector<float> txt_f = read_f32("dit_wide_txt");
  const std::vector<float> want = read_f32("dit_wide_out");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty() && !want.empty());
  auto up = [&](const std::vector<float>& v) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &v[i], 4);
      const std::uint32_t r = (u >> 16) & 1u;
      d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
    }
    return b;
  };

  auto run = [&](bool i8, double* rel, bool* engaged) {
    ::setenv("VPIPE_I8_GEMM", i8 ? "1" : "0", 1);
    if (i8) { ::setenv("VPIPE_I8_GEMM_MIN_M", "128", 1); }
    MetalQwenImage21Transformer::Config c = cfg;
    c.i8_gemm = i8;
    auto m = MetalQwenImage21Transformer::load(ckpt, mc, c, false);
    ::unsetenv("VPIPE_I8_GEMM");
    ::unsetenv("VPIPE_I8_GEMM_MIN_M");
    if (m == nullptr) { return false; }
    *engaged = m->uses_i8_gemm();
    metal_compute::SharedBuffer lat = up(lat_f);
    metal_compute::SharedBuffer txt = up(txt_f);
    if (lat.empty() || txt.empty()) { return false; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = (float)(o.contains("timestep")
                               ? o.at("timestep").as_real(0.75) : 0.75);
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < want.size() * 2) { return false; }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      const double e = (double)f - (double)want[i];
      num += e * e;
      den += (double)want[i] * (double)want[i];
    }
    *rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    return true;
  };

  double rel_off = 0.0, rel_on = 0.0;
  bool eng_off = true, eng_on = false;
  Q21D_REQUIRE(run(false, &rel_off, &eng_off));
  Q21D_REQUIRE(run(true, &rel_on, &eng_on));
  std::printf("[qwen_image_21_dit] i8 off rel-L2 %.6f, i8 on %.6f (%.2fx the "
              "error)\n", rel_off, rel_on,
              rel_off > 0.0 ? rel_on / rel_off : 0.0);
  // ENGAGEMENT is the first claim: the tier was declared and unwired
  // once, and nothing else here would have noticed.
  EXPECT_TRUE(!eng_off);
  EXPECT_TRUE(eng_on);
  // And it must stay a ROUNDING difference. MEASURED 0.022 -> 0.042
  // through two blocks, i.e. about twice the bf16 distance from the
  // reference -- real, and worth knowing before turning it on, but the
  // same order. A broken quantization is not this close.
  EXPECT_TRUE(rel_on < 0.08);
  EXPECT_TRUE(rel_on > rel_off);
}

// SAGE: int8 Q/K for the QK^T half of attention.
//
// The half that grows with the SQUARE of the sequence -- MEASURED on
// this model, attention is 10.4% of a 1024^2 step and 36.3% of a
// 2048^2 one -- so it is where a quantized-compute tier is worth the
// most, and the only one of the tiers whose value rises with the
// picture.
//
// This model's layout makes the wiring different from every other
// family here. Attention is dispatched PER LAYOUT SEGMENT, and the
// kernel indexes its int8 operands by params->qL / params->kL, so the
// quantized operands belong to one segment's geometry. The
// single-group families prepare once per block; this one prepares per
// qualifying segment, and skips segments too small to pay for their own
// quantization pass.
//
// Checked at a geometry where a segment actually qualifies (the
// kSageMinSegment floor is 512 rows, so the 320-row A/B geometry would
// silently test nothing) and against the SAME model with sage off.
//
// AND IT IS A NET LOSS HERE, which is why it stays off by default and
// why this test asserts a BAND rather than a small number. MEASURED at
// 2048^2, where attention is 36% of the step and this tier should be
// worth the most:
//
//   attention  15276 -> 15575 ms      sage is 2% SLOWER
//   step       41602 -> 44859 ms      7.8% slower
//   accuracy   rel-L2 0.163 against the dense arm
//
// The quantization pass costs more than the int8 QK^T saves, because
// the dense arm is already the NAX flash kernel at ~9.4 TFLOP/s -- the
// same reason a better attention TILE has so little headroom here. The
// per-block trace enters at 3.5e-2 in block 0, so the error is the int8
// attention itself and not compounding.
//
// The wiring is kept because a graph may ask for the tier and because
// the shape may favour it elsewhere; the numbers are here so nobody
// turns it on expecting a win.
// SOL-ATTN: skip the key blocks that contribute almost nothing.
//
// The only tier here whose value RISES with the picture. Attention is
// 10.4% of a 1024^2 step and 36.3% of a 2048^2 one, and unlike sage --
// which tried to make the same arithmetic cheaper and could not, the
// dense kernel already being GEMM-class -- this does less of it.
//
// MEASURED at 2048^2: attention 15351 -> 4171 ms (3.68x), step 42290
// -> 34356 (1.23x).
//
// It is APPROXIMATE, so the number that decides whether it ships is
// this one, not that one. Non-causal segments only: Sol's exact half is
// the steel kernel under a CSR, built without the causal predicate, so
// it serves an image block and declines a text run -- which is the
// cheap part anyway.
TEST(qwen_image_21_dit, sol_attention_engages_and_stays_close)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // 512x512 -> 1024 image rows, which clears kSolMinSegment, and 17 key
  // blocks at the default 64, which clears Sol's own "nothing left to
  // skip" floor of 16.
  //
  // SPARSITY IS A PROPERTY OF LENGTH. 512 is the SHORT case and so the
  // least favourable: there is least to skip and the approximation has
  // the largest share. VPIPE_QWEN_IMAGE21_SOL_PX runs it where the tier
  // is actually for.
  int px = 512;
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_SOL_PX")) {
    px = std::atoi(e) > 0 ? std::atoi(e) : px;
  }
  const int lg = px / 16, n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;

  auto run = [&](bool on, float tau, std::vector<float>* outv, bool* armed) {
    MetalQwenImage21Transformer::Config c = cfg;
    c.sol.enabled = on;
    c.sol.tau = tau;
    auto m = MetalQwenImage21Transformer::load(dir, mc, c, true);
    if (m == nullptr) { return false; }
    *armed = m->uses_sol_attn();
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 41u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 42u);
    if (lat.empty() || txt.empty()) { return false; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < n * 2) {
      std::printf("[qwen_image_21_dit] sol=%d forward failed: %s\n",
                  (int)on, ferr.c_str());
      return false;
    }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    outv->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      (*outv)[i] = f;
    }
    return true;
  };

  auto rel = [&](const std::vector<float>& x, const std::vector<float>& y) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double e = (double)x[i] - (double)y[i];
      num += e * e;
      den += (double)y[i] * (double)y[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  std::vector<float> dense, exact, routed;
  bool a_off = true, a_exact = false, a_on = false;
  Q21D_REQUIRE(run(false, 1.0f, &dense, &a_off));
  // EVERY BLOCK KEPT. The routing, both partials and the merge all
  // still run, so this pins the whole machine against the one case with
  // a known answer -- and it is the band plumbing's end-to-end check in
  // a real model, not a rig.
  Q21D_REQUIRE(run(true, -1.0e30f, &exact, &a_exact));
  float tau = 1.0f;
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_SOL_TAU")) {
    tau = (float)std::atof(e);
  }
  Q21D_REQUIRE(run(true, tau, &routed, &a_on));
  EXPECT_TRUE(!a_off);
  EXPECT_TRUE(a_exact && a_on);
  Q21D_REQUIRE(dense.size() == exact.size() && !dense.empty());

  const double e0 = rel(exact, dense);
  const double e1 = rel(routed, dense);
  std::printf("[qwen_image_21_dit] sol %dpx tau=-inf vs dense: rel-L2 "
              "%.3e; tau=%.2f: %.4f (joint %d)\n", px, e0, tau, e1,
              lay.joint_len);
  // Keeping everything must reproduce the dense answer.
  EXPECT_TRUE(e0 < 0.02);
  // And routing must actually approximate something rather than
  // degenerate -- a run that silently kept every block would pass any
  // upper bound alone.
  EXPECT_TRUE(e1 > 1e-4);
  // A BAND, and a wide one: the routed error is a property of the
  // sequence's redundancy, so it moves with the picture. See the note
  // above -- the default geometry is the least favourable one.
  EXPECT_TRUE(e1 < 0.45);
}

TEST(qwen_image_21_dit, sage_attention_engages_and_matches_dense)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }   // nax-entry only
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // 512x512 -> a 32x32 latent, 1024 image rows: past the 512-row floor,
  // so the image segment takes the int8 path while the text run does
  // not -- which is the split the wiring is meant to produce.
  const int lg = 32, n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;

  auto run = [&](bool sage, std::vector<float>* outv, bool* armed) {
    MetalQwenImage21Transformer::Config c = cfg;
    c.sage.enabled = sage;
    auto m = MetalQwenImage21Transformer::load(dir, mc, c, true);
    if (m == nullptr) { return false; }
    *armed = m->uses_sage_attn();
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 21u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 22u);
    if (lat.empty() || txt.empty()) { return false; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < n * 2) {
      std::printf("[qwen_image_21_dit] sage=%d forward failed: %s\n",
                  (int)sage, ferr.c_str());
      return false;
    }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    outv->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      (*outv)[i] = f;
    }
    return true;
  };

  std::vector<float> dense, s8;
  bool armed_off = true, armed_on = false;
  Q21D_REQUIRE(run(false, &dense, &armed_off));
  Q21D_REQUIRE(run(true, &s8, &armed_on));
  // ENGAGEMENT first: this tier was declared and unwired, and nothing
  // about a correct picture would have said so.
  EXPECT_TRUE(!armed_off);
  EXPECT_TRUE(armed_on);
  Q21D_REQUIRE(dense.size() == s8.size() && !dense.empty());

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < dense.size(); ++i) {
    const double e = (double)s8[i] - (double)dense[i];
    num += e * e;
    den += (double)dense[i] * (double)dense[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[qwen_image_21_dit] sage vs dense attention: rel-L2 %.4e "
              "(joint %d, %d layers)\n", rel, lay.joint_len, cfg.n_layers);
  // A BAND, not a small number: 0.163 is what this model measures, and
  // a test that asserted "small" would be asserting something untrue.
  // The floor catches a dense-vs-dense run (the tier not engaging); the
  // ceiling catches a wiring break, which does not land near 0.16.
  EXPECT_TRUE(rel < 0.30);
  EXPECT_TRUE(rel > 0.02);
}

// THE ANE FEED-FORWARD: that its rows are right, and what it costs.
//
// A SwiGLU feed-forward is row-independent, so the ANE takes a
// contiguous band of a block's rows and the GPU the rest, concurrently.
// There is no seam to blend -- the two compute the same function over
// disjoint rows -- so the whole claim is that the ANE's rows come back
// with the same numbers. They are not BIT-identical: the module is
// fp16 where the GPU is bf16, which has MORE mantissa, so the split
// arm's own rows are if anything the more accurate of the two.
//
// WHAT THE TIER IS FOR, and it is not this box. The ANE's value is
// largest where the GPU is slowest -- the M4 series, which has no
// matrix cores at all, so its GEMMs run the tiled path rather than
// matmul2d. On an M5 the GPU is already 3x faster at the same work and
// there is proportionally less for the ANE to take.
//
// MEASURED HERE, an M5 16 GB FANLESS MacBook Air, interleaved and
// unprofiled: 1024^2 6851 -> 7389 ms and 2048^2 43909 -> 47783, i.e.
// 8-9% SLOWER. That figure is a property of THIS MACHINE and must not
// be quoted as the tier's: a fanless Air shares one SoC power budget
// between the ANE and the GPU, so the two do not truly overlap, while
// an M5 Pro or an M4 Mac mini does overlap them. The per-block
// telemetry here still reports the ANE's rows 100% hidden (2048 of
// 4160 at 1024^2) -- the wall clock disagrees because the engines are
// contending for watts, which the balancer cannot see.
//
// So this test asserts CORRECTNESS, which is box-independent, and
// leaves the speed question to the machines the tier is aimed at.
TEST(qwen_image_21_dit, ane_feed_forward_rows_are_right)
{
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // 1024x1024: 4160 joint rows. The module's chunk is 2048, so a
  // shorter sequence gives the ANE less than one chunk and the tier
  // declines outright -- which is a correct outcome that tests nothing,
  // and is what 512^2 did here.
  const int lg = 64, n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));

  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<std::uint16_t*>(b.contents());
    unsigned st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      const float v = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;

  auto run = [&](bool ane, std::vector<float>* outv, bool* armed) {
    MetalQwenImage21Transformer::Config c = cfg;
    c.ane_ffn = ane;
    auto m = MetalQwenImage21Transformer::load(dir, mc, c, true);
    if (m == nullptr) { return false; }
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 61u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 62u);
    if (lat.empty() || txt.empty()) { return false; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    // Asked AFTER the forward: the module is created on the first one,
    // when the row count is finally known.
    *armed = m->ane_armed();
    if (out.empty() || out.byte_size() < n * 2) {
      std::printf("[qwen_image_21_dit] ane=%d forward failed: %s\n",
                  (int)ane, ferr.c_str());
      return false;
    }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    outv->resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)d[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      (*outv)[i] = f;
    }
    return true;
  };

  std::vector<float> gpu, split;
  bool armed_off = true, armed_on = false;
  Q21D_REQUIRE(run(false, &gpu, &armed_off));
  Q21D_REQUIRE(run(true, &split, &armed_on));
  EXPECT_TRUE(!armed_off);
  if (!armed_on) {
    // No ANE on this box, or the module would not compile: the tier
    // declining is a valid outcome and not a failure.
    std::printf("[qwen_image_21_dit] ANE did not arm -- SKIPPED\n");
    return;
  }
  Q21D_REQUIRE(gpu.size() == split.size() && !gpu.empty());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    const double e = (double)split[i] - (double)gpu[i];
    num += e * e;
    den += (double)gpu[i] * (double)gpu[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[qwen_image_21_dit] ANE split vs GPU-only: rel-L2 %.4e "
              "(joint %d, %d layers)\n", rel, lay.joint_len, cfg.n_layers);
  // The same function over disjoint rows, one side fp16: a rounding
  // difference through 32 blocks, not a different answer.
  EXPECT_TRUE(rel < 0.05);
}

// QUANTIZING THIS FAMILY, end to end: the stage's leaf set, the
// quantizer, and the DiT's own loader reading what came out.
//
// The leaf set is the part that goes wrong silently. A leaf is matched
// on the LAST dot-component, so a name that collides with a
// model-level tensor quantizes something that must stay bf16, and a
// name that matches nothing leaves a "4-bit" checkpoint the size of
// its source. This model's seven block Linears are to_q / to_k / to_v
// / "0" (attn.to_out.0) / gate_layer / proj / out, and none of them
// collides with the nine model-level leaves -- so the check is that
// the blocks came out quantized AND the model-level tensors did not.
//
// Run against `dit_wide`: two layers at the REAL widths, which is what
// makes the leaf names and the group arithmetic the real ones.
TEST(qwen_image_21_dit, quantize_produces_a_loadable_checkpoint)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::filesystem::path gd(g);
  const std::string src = (gd / "dit_wide").string();
  std::error_code ec;
  if (!std::filesystem::exists(src + "/config.json", ec)) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::filesystem::path out =
      std::filesystem::temp_directory_path() /
      ("vpipe-qi21-q-" + std::to_string(::getpid()));
  std::filesystem::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(4));
    o.insert("skip_existing", FlexData::make_bool(false));
  }
  vpipe::ModelQuantizeStage st(&sess, "q21", std::vector<InEdge>{},
                               std::move(cfg));
  Q21D_REQUIRE(st.config_error().empty());
  if (!st.quantize_once()) {
    std::printf("[qwen_image_21_dit] quantize did not run\n");
    std::filesystem::remove_all(out, ec);
  }
  Q21D_REQUIRE(std::filesystem::exists(out / "config.json", ec));

  // WHICH tensors were quantized. A quantized one grows a `.scales`
  // sibling; a passed-through one stays bf16.
  auto wts = vpipe::genai::MetalLlamaWeights::open_model(out.string());
  Q21D_REQUIRE(wts.has_value());
  int blk_q = 0, blk_dense = 0, top_q = 0, top_dense = 0;
  for (const auto& n : wts->tensor_names()) {
    if (n.size() > 7 && n.compare(n.size() - 7, 7, ".scales") == 0) {
      const std::string base = n.substr(0, n.size() - 7);
      if (base.rfind("transformer_blocks.", 0) == 0) { ++blk_q; }
      else { ++top_q; }
      continue;
    }
    if (n.size() > 7 && n.compare(n.size() - 7, 7, ".biases") == 0) {
      continue;
    }
    // A PASSTHROUGH is a tensor with no `.scales` sibling, whatever
    // dtype it ended up in -- keying this on BF16 counted zero of them,
    // because what the quantizer passes through is not necessarily the
    // dtype the source held.
    // The codes keep the `.weight` name and the scales hang off the
    // base WITHOUT it, so the sibling of `X.weight` is `X.scales` --
    // looking for `X.weight.scales` finds nothing and counts every
    // quantized tensor as a passthrough.
    std::string base = n;
    if (base.size() > 7 && base.compare(base.size() - 7, 7, ".weight") == 0) {
      base.resize(base.size() - 7);
    }
    const std::string want = base + ".scales";
    bool quantized = false;
    for (const auto& o : wts->tensor_names()) {
      if (o == want) { quantized = true; break; }
    }
    if (quantized) { continue; }
    if (n.rfind("transformer_blocks.", 0) == 0) { ++blk_dense; }
    else { ++top_dense; }
  }
  std::printf("[qwen_image_21_dit] quantized: %d block tensors, %d "
              "model-level; left dense: %d block, %d model-level\n",
              blk_q, top_q, blk_dense, top_dense);
  // The seven Linears of each of two blocks.
  EXPECT_TRUE(blk_q == 14);
  // attn.norm_q / norm_k are vectors, not matrices: 2 per block.
  EXPECT_TRUE(blk_dense == 4);
  // NOTHING model-level: img_in, modulation, txt_in, the timestep MLP,
  // norm_out and proj_out all carry the picture and stay bf16.
  EXPECT_TRUE(top_q == 0);
  EXPECT_TRUE(top_dense > 0);

  // AND THE DiT'S OWN LOADER READS IT. A checkpoint the quantizer is
  // happy with that the model cannot open is not a checkpoint.
  MetalQwenImage21Transformer::Config qc;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(out.string(),
                                                              &qc));
  auto m = MetalQwenImage21Transformer::load(out.string(), mc, qc, false);
  if (m == nullptr) {
    std::printf("[qwen_image_21_dit] the quantized checkpoint did not "
                "load\n");
  }
  Q21D_REQUIRE(m != nullptr);
  std::printf("[qwen_image_21_dit] quantized DiT loaded, %d-bit\n",
              m->quant_bits());
  EXPECT_TRUE(m->quant_bits() == 4);
  std::filesystem::remove_all(out, ec);
}
