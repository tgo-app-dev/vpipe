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
#include "residency-reserve.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/weight-set.h"
#include "stages/model-quantize-stage.h"

#include <unistd.h>
#include "stages/model-memory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

// `cond` is evaluated ONCE. An ASSERT_TRUE(cond) followed by
// if (!(cond)) evaluates it twice, which is invisible for a
// comparison and doubles the work for a call -- a model loaded, or a
// forward run, twice per use.
#define Q21D_REQUIRE(cond)                                                 \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
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

// ---- perf-flops ----------------------------------------------------
//
// The forward's arithmetic, as the FLOP/s meter is credited with it
// (common/perf-flops.h, shared/dit-flops.h). Three things:
//
//   * EXACT against a count derived here from the model's structure: every
//     projection at 2*M*N*K and the block-causal attention at its pairs --
//     a GEMM counted twice or missed is a wrong figure on every plot;
//   * the meter is credited with all of it, from the completion handlers,
//     once the buffers retire;
//   * with the feed-forward's rows split onto the ANE, GPU + ANE is the
//     same count -- moving rows moves no FLOPs.
//
// The ANE half needs a sequence past one 2048-row chunk (the tier
// declines below it), hence 2112 image rows. Streams, like the smoke
// test above. Env: VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH.
TEST(qwen_image_21_dit, forward_credits_its_flops)
{
  namespace pf = vpipe::perf_flops;
  const std::string dir = dit_dir_();
  if (dir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));

  // 20 prompt tokens, then a 48x44 target (528 slots of four rows).
  std::vector<std::uint8_t> slot(20, 0);
  slot.resize(20 + 528, 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, 48, 44}};
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
      rnd((std::size_t)slot.size() * cfg.txt_dim, 22u);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());

  // The count, from the structure.
  const long long H = cfg.hidden, FF = (long long)cfg.hidden * cfg.mlp_ratio;
  const long long J = lay.joint_len, IMG = lay.image_len;
  const long long E = (long long)slot.size();
  const long long MR = cfg.causal_condition ? 2 : 1;
  std::uint64_t attn = 0;
  for (const qi21::Segment& sg : lay.segments) {
    const long long q = sg.end - sg.start;
    attn += attention_flops(cfg.n_heads,
                            sg.is_text ? causal_pairs(q, sg.start)
                                       : q * sg.end,
                            cfg.head_dim);
  }
  const std::uint64_t per_block =
      4 * gemm_flops(J, H, H) + 3 * gemm_flops(J, FF, H) + attn;
  const std::uint64_t model =
      gemm_flops(MR, H, cfg.time_proj) + gemm_flops(MR, H, H) +
      gemm_flops(MR, 4 * H, H) + gemm_flops(MR, H, H) +
      gemm_flops(IMG, H, cfg.in_channels) +
      gemm_flops(E, H, cfg.txt_dim) + gemm_flops(E, H, H) +
      gemm_flops(J, cfg.out_channels, H);
  const std::uint64_t want =
      model + (std::uint64_t)cfg.n_layers * per_block;

  // One forward on a fresh model; GPU and ANE credited during it.
  struct Run { std::uint64_t flushed, gpu, ane; double ms; bool ane_on; };
  auto run = [&](bool ane) -> Run {
    Run r{0, 0, 0, 0.0, false};
    MetalQwenImage21Transformer::Config c = cfg;
    c.ane_ffn = ane;
    c.ane_rows = ane ? 1 : 0;       // pinned: the whole first chunk
    auto m = MetalQwenImage21Transformer::load(dir, mc, c, true);
    if (m == nullptr) { return r; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    const pf::Totals g0 = pf::totals(pf::kGpu);
    const pf::Totals a0 = pf::totals(pf::kAne);
    const auto t0 = std::chrono::steady_clock::now();
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    r.ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (out.empty()) {
      std::printf("  forward failed: %s\n", ferr.c_str());
      return r;
    }
    r.flushed = m->gpu_flops().flushed();
    // The last buffer's handler may run after the wait that saw it.
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0.flops <
                                   r.flushed; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    r.gpu = pf::totals(pf::kGpu).flops - g0.flops;
    r.ane = pf::totals(pf::kAne).flops - a0.flops;
    r.ane_on = m->ane_armed() && !m->ane_gpu_only();
    return r;
  };

  const Run a = run(false);
  Q21D_REQUIRE(a.flushed > 0);
  std::printf("  joint %lld rows, %d blocks: counted %.4f TFLOP, derived "
              "%.4f, credited %.4f; %.0f ms -> %.2f TFLOP/s\n", J,
              cfg.n_layers, a.flushed / 1e12, want / 1e12, a.gpu / 1e12,
              a.ms, a.gpu / 1e12 / (a.ms / 1000.0));
  EXPECT_TRUE(a.flushed == want);
  EXPECT_TRUE(a.gpu == a.flushed);
  EXPECT_TRUE(a.ane == 0);

  const Run b = run(true);
  if (!b.ane_on || b.ane == 0) {
    std::printf("  no ANE split on this box; the conservation half is "
                "skipped\n");
    return;
  }
  // The rows moved are whole chunks of the SwiGLU's 6*H*FF a row. A
  // headroom retry is ANE work the GPU never does, so the ANE may exceed
  // what moved -- by whole chunks too -- and never fall short.
  const std::uint64_t chunk = gemm_flops(2048, 3 * FF, H);
  const std::uint64_t moved = a.gpu - b.gpu;
  std::printf("  ANE split: GPU %.4f + ANE %.4f TFLOP (moved %.4f, %llu "
              "chunks)\n", b.gpu / 1e12, b.ane / 1e12, moved / 1e12,
              (unsigned long long)(moved / chunk));
  EXPECT_TRUE(b.gpu == b.flushed);
  EXPECT_TRUE(moved > 0 && moved % chunk == 0);
  EXPECT_TRUE(b.ane >= moved && (b.ane - moved) % chunk == 0);
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

// A MIXED PACK (VPIPE_QWEN_IMAGE21_MIXED_DIT = the transformer/ of a
// model-quantize bits 4 + mixed run): whole blocks at w8, the rest at w4,
// so a slot built for one width is asked to hold the other
// (BlockSlots::Ops::same_layout). The pack must really mix widths, and
// its four holds -- preloaded, streamed, no slots, promoted -- must be
// BYTE-IDENTICAL with i8 off, and on a matrix-core box under the signed
// layout and (when block 0, which gates the fast tier, is w8) the fast
// and token tiers. Before the slots rebuilt such a block, preloaded and
// no-slots agreed and the two slot holds did not.
TEST(qwen_image_21_dit, a_mixed_pack_holds_agree)
{
  const char* qd = std::getenv("VPIPE_QWEN_IMAGE21_MIXED_DIT");
  if (qd == nullptr || *qd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(qd, &cfg));
  int n8 = 0, n4 = 0;
  {
    auto ws = genai::WeightSet::open(qd, nullptr);
    Q21D_REQUIRE(ws != nullptr);
    for (int L = 0; L < cfg.n_layers; ++L) {
      const int b = genai::stored_affine_bits(
          *ws, "transformer_blocks." + std::to_string(L) + ".attn.to_q", 64);
      n8 += b == 8;
      n4 += b == 4;
    }
  }
  std::printf("[qwen_image_21_dit] mixed pack: %d blocks w8, %d w4\n", n8,
              n4);
  Q21D_REQUIRE(n8 > 0 && n4 > 0);
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, TS = 64;
  std::vector<std::uint8_t> slot((std::size_t)TS, 0);
  slot.resize((std::size_t)TS + (std::size_t)(gh * gw / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, gh, gw}};
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
  bool engaged = true;
  bool fast_on = false;
  int f8 = 0, f4 = 0;
  std::uint64_t held = 0, grown_bytes = 0;
  int promoted = 0, relaid = 0;
  int stack = 0;
  auto run = [&](int tier, bool stream, bool slots, bool grow,
                 bool hold = false) {
    std::vector<std::uint16_t> v;
    std::unique_ptr<Env> ns;
    if (!slots) {
      ns = std::make_unique<Env>("VPIPE_QWEN_IMAGE21_NO_SLOTS", "1");
    }
    MetalQwenImage21Transformer::Config c = cfg;
    c.i8_gemm = tier >= 0;
    c.i8_tier = tier < 0 ? 0 : tier;
    c.i8_hold_w4 = hold;
    auto m = MetalQwenImage21Transformer::load(qd, mc, c, stream);
    if (m == nullptr) { return v; }
    if (tier == 0 && !m->signed_layout()) { engaged = false; }
    if (!stream) { held = m->resident_bytes(); }
    if (tier > 0) {
      fast_on = m->fast_tier();
      if (!stream) {
        f8 = m->fast_weights(8);
        f4 = m->fast_weights(4);
      }
    }
    m->set_residency_reserve(grow ? test_residency_reserve(8ull << 30) : 0);
    if (grow) { m->set_residency_schedule(4); }
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 11u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)slot.size() * cfg.txt_dim, 22u);
    if (lat.empty() || txt.empty()) { return v; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out;
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = m->forward(req, &ferr);
    }
    if (grow) {
      promoted = m->resident_block_count();
      stack = m->config().n_layers;
      grown_bytes = m->resident_bytes();
    }
    if (stream && slots && !grow) { relaid = m->slot_relayouts(); }
    const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
    if (out.empty() || out.byte_size() < n * 2) { return v; }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    v.assign(d, d + n);
    return v;
  };
  std::vector<int> tiers = {-1};
  if (mc->supports_matrix_cores()) { tiers.insert(tiers.end(), {0, 1, 2}); }
  const char* tname[3] = {"accurate (signed)", "fast", "token"};
  for (int tier : tiers) {
    engaged = true;
    const auto pre = run(tier, false, true, false);
    Q21D_REQUIRE(!pre.empty());
    // What the model holds preloaded, against what the plan books for it
    // (image_memory_plan.the_int8_tiers_book_a_w4_pack_s_scratch).
    std::printf("[qwen_image_21_dit] mixed pack, %s: preloaded, it holds "
                "%llu MB\n", tier < 0 ? "no i8" : tname[tier],
                (unsigned long long)(held >> 20));
    if (tier > 0) {
      // BOTH WIDTHS take the tier: a w8 block's codes converted once,
      // a w4 one's into the scratch before each GEMM. Uniform w8 packs:
      // fast_tier_matches_accurate.
      std::printf("[qwen_image_21_dit] mixed pack, %s: %d w8 and %d w4 "
                  "projections in its layout\n", tname[tier], f8, f4);
      EXPECT_TRUE(fast_on && f8 > 0 && f4 > 0);
      if (!fast_on) { continue; }
    }
    const auto str = run(tier, true, true, false);
    const auto flat = run(tier, true, false, false);
    const auto grown = run(tier, true, true, true);
    const std::uint64_t grown_pc = grown_bytes;
    // A HELD layout reached by PROMOTION: residency converts each w4
    // block it keeps (clone_block_), which then holds more than the
    // pack -- and must equal the held preload.
    const auto grown_held =
        tier > 0 ? run(tier, true, true, true, true) : grown;
    const std::uint64_t grown_h = grown_bytes;
    const int grown_n = promoted;
    // ...and with NO SLOTS, where promote_into MOVES each per-block read
    // into the resident set: the keep pass (keep_block_) must leave it
    // holding exactly what the slots' clone does.
    const auto flat_held =
        tier > 0 ? run(tier, true, false, true, true) : grown;
    const std::uint64_t flat_h = tier > 0 ? grown_bytes : grown_h;
    const int flat_n = tier > 0 ? promoted : grown_n;
    // The bytes compare only where BOTH kept the whole streamed stack:
    // a slotted run short of it still holds its slot pair, and a box
    // with little room (a 16 GB M5) keeps what its budget allows.
    const bool whole = grown_n == stack && flat_n == stack;
    // The engine's other answer for a w4 pack: its int8 layout
    // HELD (Config::i8_hold_w4) where it is converted per GEMM
    // above -- the same values, so the same bytes.
    const auto held =
        tier > 0 ? run(tier, false, true, false, true) : pre;
    std::printf("[qwen_image_21_dit] mixed pack, %s: streamed %s, no slots "
                "%s, promoted (%d blocks) %s%s\n",
                tier < 0 ? "no i8" : tname[tier],
                str == pre ? "IDENTICAL" : "DIFFERENT",
                flat == pre ? "IDENTICAL" : "DIFFERENT", promoted,
                grown == pre ? "IDENTICAL" : "DIFFERENT",
                engaged ? "" : " -- NOT ENGAGED");
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(str == pre);
    EXPECT_TRUE(flat == pre);
    EXPECT_TRUE(promoted > 0 && grown == pre);
    if (tier > 0) {
      std::printf("[qwen_image_21_dit] ... w4 layout held: %s; held by "
                  "promotion %s (%llu MB resident against %llu per "
                  "call); with no slots %s (%llu MB)\n",
                  held == pre ? "IDENTICAL" : "DIFFERENT",
                  grown_held == pre ? "IDENTICAL" : "DIFFERENT",
                  (unsigned long long)(grown_h >> 20),
                  (unsigned long long)(grown_pc >> 20),
                  flat_held == pre ? "IDENTICAL" : "DIFFERENT",
                  (unsigned long long)(flat_h >> 20));
      EXPECT_TRUE(grown_held == pre && grown_h > grown_pc);
      // The slots' resident figure carries no slots once residency is
      // whole (release_if_saturated); the moved blocks must hold the
      // same bytes the cloned ones do.
      EXPECT_TRUE(flat_held == pre && (!whole || flat_h == grown_h));
    }
    EXPECT_TRUE(held == pre);
    // The slots crossed widths and REBUILT. A slot that refused such a
    // block instead turns itself off, and its holds still agree.
    std::printf("[qwen_image_21_dit] ... %d streamed blocks built into their "
                "slot\n", relaid);
    EXPECT_TRUE(relaid > 0);
  }
}

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
// THE SIGNED INT8 LAYOUT on a real quantized pack
// (VPIPE_QWEN_IMAGE21_QUANT_DIT at a model-quantize'd DiT; i8_gemm on; M5
// only), as flux2_smoke.signed_layout_matches_native: every arm --
// preloaded, streamed through the slots, streamed without them, promoted
// -- BYTE-IDENTICAL, and within a gross-error bar of the route this
// family took before (dequant + requant; VPIPE_I8_SIGNED=0) in distance
// from the dense (i8 off) forward. One joint sequence -- this model has
// no separate text stream -- at three lengths, one per route a signed
// weight takes:
// 1280 target rows + 64 prompt (the int8 GEMM), 64 + 8 (the signed
// dequant into matmul2d, under the int8 floor) and 32 + 8 (the steel
// tiles through the signed dequant, under the matmul2d tile's).
TEST(qwen_image_21_dit, signed_layout_matches_native)
{
  const char* qd = std::getenv("VPIPE_QWEN_IMAGE21_QUANT_DIT");
  if (qd == nullptr || *qd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(qd, &cfg));
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  struct Geo { int gh, gw, TS; };
  for (const Geo& g : {Geo{32, 40, 64}, Geo{8, 8, 8}, Geo{4, 8, 8}}) {
    const int gh = g.gh, gw = g.gw, TS = g.TS;
    std::vector<std::uint8_t> slot((std::size_t)TS, 0);
    slot.resize((std::size_t)TS + (std::size_t)(gh * gw / 4), 1);
    const std::vector<qi21::ImgBlock> blocks = {{1, gh, gw}};
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
        const float v =
            ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        d[i] = (std::uint16_t)(u >> 16);
      }
      return b;
    };
    bool engaged = true;
    auto run = [&](bool sgn, bool i8, bool stream, bool grow, bool slots,
                   const std::string& dir = std::string()) {
      std::vector<std::uint16_t> v;
      std::unique_ptr<Env> e1, e2;
      if (!sgn) { e1 = std::make_unique<Env>("VPIPE_I8_SIGNED", "0"); }
      if (!slots) {
        e2 = std::make_unique<Env>("VPIPE_QWEN_IMAGE21_NO_SLOTS", "1");
      }
      MetalQwenImage21Transformer::Config c = cfg;
      c.i8_gemm = i8;
      auto m = MetalQwenImage21Transformer::load(dir.empty() ? qd : dir, mc,
                                                 c, stream);
      if (m == nullptr) { return v; }
      if (dir.empty() && m->signed_layout() != (sgn && i8)) {
        engaged = false;
      }
      m->set_residency_reserve(grow ? test_residency_reserve(4ull << 30) : 0);
      if (grow) { m->set_residency_schedule(4); }
      metal_compute::SharedBuffer lat =
          rnd((std::size_t)lay.image_len * cfg.in_channels, 11u);
      metal_compute::SharedBuffer txt =
          rnd((std::size_t)slot.size() * cfg.txt_dim, 22u);
      if (lat.empty() || txt.empty()) { return v; }
      MetalQwenImage21Transformer::Request req;
      req.latents = &lat;
      req.txt = &txt;
      req.layout = &lay;
      req.timestep = 0.75f;
      std::string ferr;
      metal_compute::SharedBuffer out;
      for (int it = 0; it < (grow ? 2 : 1); ++it) {
        out = m->forward(req, &ferr);
      }
      if (grow && m->resident_block_count() == 0) { engaged = false; }
      const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
      if (out.empty() || out.byte_size() < n * 2) { return v; }
      const auto* d = static_cast<const std::uint16_t*>(out.contents());
      v.assign(d, d + n);
      return v;
    };
    const auto dense = run(false, false, false, false, true);
    const auto nat = run(false, true, false, false, true);
    const auto pre = run(true, true, false, false, true);
    const auto str = run(true, true, true, false, true);
    const auto flat = run(true, true, true, false, false);
    const auto grown = run(true, true, true, true, true);
    EXPECT_TRUE(engaged);
    Q21D_REQUIRE(!pre.empty() && nat.size() == pre.size() &&
                 dense.size() == pre.size());
    auto f = [](std::uint16_t h) {
      const std::uint32_t u = (std::uint32_t)h << 16;
      float x;
      std::memcpy(&x, &u, 4);
      return (double)x;
    };
    auto rel = [&](const std::vector<std::uint16_t>& a,
                   const std::vector<std::uint16_t>& b) {
      double n2 = 0, d2 = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        n2 += (f(a[i]) - f(b[i])) * (f(a[i]) - f(b[i]));
        d2 += f(b[i]) * f(b[i]);
      }
      return std::sqrt(n2 / d2);
    };
    const double rs = rel(pre, dense), rn = rel(nat, dense);
    // The yardstick, where the bf16 checkpoint is at hand: how far the
    // pack itself (dense, i8 off) sits from the weights it quantized.
    const std::string bdir = dit_dir_();
    if (!bdir.empty()) {
      const auto orig = run(false, false, false, false, true, bdir);
      if (orig.size() == pre.size()) {
        std::printf("[qwen_image_21_dit]   vs the bf16 checkpoint: the "
                    "pack %.3e, signed %.3e, unsigned %.3e\n",
                    rel(dense, orig), rel(pre, orig), rel(nat, orig));
      }
    }
    std::printf("[qwen_image_21_dit] signed layout, %d+%d rows: vs the "
                "dense forward signed %.3e, unsigned %.3e; streamed %s, no "
                "slots %s, promoted %s\n", gh * gw, TS, rs, rn,
                str == pre ? "IDENTICAL" : "DIFFERENT",
                flat == pre ? "IDENTICAL" : "DIFFERENT",
                grown == pre ? "IDENTICAL" : "DIFFERENT");
    EXPECT_TRUE(str == pre);
    EXPECT_TRUE(flat == pre);
    EXPECT_TRUE(grown == pre);
    // A GROSS-ERROR guard, not a ranking: on random inputs this forward
    // is chaotic -- QI21's dense forward moved 4.8e-2 for a per-weight
    // change of 5e-5 (gemm_i8.signed_dequant_on_a_real_pack: both
    // dequants 7.3e-3 from the bf16 weights, equal to four digits) -- so
    // two equally accurate routes land a few percent apart either way.
    // gemm_i8.signed_layout_matches_oracle is the accuracy statement; a
    // misread layout is O(1) here.
    // Under the int8 floor the old route is the dense one, exactly.
    if (rn > 0.0) { EXPECT_TRUE(rs <= 1.25 * rn); }
  }
}

// THE INT8 FAST AND TOKEN TIERS on the real pack
// (VPIPE_QWEN_IMAGE21_QUANT_DIT): the block projections held as rotated
// int8 at a coarse group (I8GemmContext::Tier::kFast), or as int8 per
// channel x per token (kToken). Each must ENGAGE, be BYTE-IDENTICAL however
// the stack is held (preloaded, streamed, no slots, promoted -- every one
// of them converts somewhere else: the derived entry, the slot's
// post_refill, a clone), and sit within a gross-error bar of the accurate
// tier's distance from the dense forward. The geometry clears the int8
// row floor (1280 + 64 rows); with VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH set
// the bf16 checkpoint's fast tier (quantized straight from bf16, into the
// entry) is measured against its own dense forward as well.
TEST(qwen_image_21_dit, fast_tier_matches_accurate)
{
  const char* qd = std::getenv("VPIPE_QWEN_IMAGE21_QUANT_DIT");
  if (qd == nullptr || *qd == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(qd, &cfg));
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, TS = 64;
  std::vector<std::uint8_t> slot((std::size_t)TS, 0);
  slot.resize((std::size_t)TS + (std::size_t)(gh * gw / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, gh, gw}};
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
  bool engaged = true;
  std::string kernel;                // the token GEMM the last run took
  auto run = [&](const std::string& dir, bool i8, int tier, bool stream,
                 bool grow, bool slots, bool db = false, bool wide = false) {
    std::vector<std::uint16_t> v;
    std::unique_ptr<Env> e2;
    if (!slots) {
      e2 = std::make_unique<Env>("VPIPE_QWEN_IMAGE21_NO_SLOTS", "1");
    }
    MetalQwenImage21Transformer::Config c = cfg;
    c.i8_gemm = i8;
    c.i8_tier = tier;
    // The engine's own kernel unless the A/B knob names another.
    std::unique_ptr<Env> ek;
    if (db || wide) {
      ek = std::make_unique<Env>(
          "VPIPE_I8_TOKEN_KERNEL",
          db ? (wide ? "d64x128" : "d128x64") : "m64x128");
    }
    auto m = MetalQwenImage21Transformer::load(dir, mc, c, stream);
    if (m == nullptr) { return v; }
    kernel = m->i8_token_kernel();
    if (m->fast_tier() != (i8 && tier != 0) ||
        (i8 && tier >= 2 &&
         m->fast_group() != I8GemmContext::kTokenGroup)) {
      engaged = false;
    }
    m->set_residency_reserve(grow ? test_residency_reserve(4ull << 30) : 0);
    if (grow) { m->set_residency_schedule(4); }
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 11u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)slot.size() * cfg.txt_dim, 22u);
    if (lat.empty() || txt.empty()) { return v; }
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out;
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = m->forward(req, &ferr);
    }
    if (grow && m->resident_block_count() == 0) { engaged = false; }
    const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
    if (out.empty() || out.byte_size() < n * 2) { return v; }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    v.assign(d, d + n);
    return v;
  };
  auto f = [](std::uint16_t h) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return (double)x;
  };
  auto rel = [&](const std::vector<std::uint16_t>& a,
                 const std::vector<std::uint16_t>& b) {
    double n2 = 0, d2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      n2 += (f(a[i]) - f(b[i])) * (f(a[i]) - f(b[i]));
      d2 += f(b[i]) * f(b[i]);
    }
    return std::sqrt(n2 / d2);
  };
  const auto dense = run(qd, false, 0, false, false, true);
  const auto acc = run(qd, true, 0, false, false, true);
  Q21D_REQUIRE(!acc.empty() && dense.size() == acc.size());
  const double ra = rel(acc, dense);
  // A GROSS-ERROR guard (the forward is chaotic on random inputs; see
  // signed_layout_matches_native): the fast tier's GEMMs are 1.3-1.45x
  // the accurate tier's error on outlier-free operands
  // (gemm_i8.coarse_tier), the plain token tier's ~1.7x on real weights
  // and the rotated one's about the fast tier's, so a misread layout is
  // what this catches.
  const double bar[4] = {0.0, 2.5, 3.0, 4.0};
  const char* tname[4] = {"accurate", "fast", "token", "token_plain"};
  for (int tier : {1, 2, 3}) {
    engaged = true;
    const auto pre = run(qd, true, tier, false, false, true);
    const auto str = run(qd, true, tier, true, false, true);
    const auto flat = run(qd, true, tier, true, false, false);
    const auto grown = run(qd, true, tier, true, true, true);
    EXPECT_TRUE(engaged);
    if (pre.size() != acc.size()) { ASSERT_TRUE(false); continue; }
    const double rf = rel(pre, dense);
    std::printf("[qwen_image_21_dit] %s tier, %d+%d rows: vs the dense "
                "forward %.3e, accurate %.3e; streamed %s, no slots %s, "
                "promoted %s\n", tname[tier], gh * gw, TS, rf, ra,
                str == pre ? "IDENTICAL" : "DIFFERENT",
                flat == pre ? "IDENTICAL" : "DIFFERENT",
                grown == pre ? "IDENTICAL" : "DIFFERENT");
    EXPECT_TRUE(str == pre);
    EXPECT_TRUE(flat == pre);
    EXPECT_TRUE(grown == pre);
    EXPECT_TRUE(rf <= bar[tier] * ra);
    if (tier < 2) { continue; }
    // The token GEMM's other K loop and tile (VPIPE_I8_TOKEN_KERNEL):
    // one int32 sum and one epilogue whichever kernel runs, so every
    // pairing is BYTE-IDENTICAL to the engine's own choice -- and must
    // have been the kernel asked for, or the comparison proves nothing.
    for (int k = 1; k < 4; ++k) {
      const bool db = (k & 1) != 0, wide = (k & 2) != 0;
      const auto alt = run(qd, true, tier, false, false, true, db, wide);
      const std::string want = std::string(db ? "double-buffered "
                                              : "blocked ") +
                               (wide ? "64x128" : "128x64");
      std::printf("[qwen_image_21_dit]   %s GEMM: %s\n", kernel.c_str(),
                  alt == pre ? "IDENTICAL" : "DIFFERENT");
      EXPECT_TRUE(kernel == want);
      EXPECT_TRUE(alt == pre);
    }
  }
  const std::string bdir = dit_dir_();
  if (!bdir.empty()) {
    const auto bd = run(bdir, false, 0, false, false, true);
    const auto ba = run(bdir, true, 0, false, false, true);
    for (int tier : {1, 2, 3}) {
      engaged = true;
      const auto bf = run(bdir, true, tier, false, false, true);
      if (bd.empty() || ba.size() != bd.size() || bf.size() != bd.size()) {
        continue;
      }
      // A STREAMED bf16 block quantizes its refilled source into the
      // layout (QWeight::ds) and a promotion keeps the layout alone, so
      // every hold of the checkpoint runs the tier the preload runs.
      const auto bs = run(bdir, true, tier, true, false, true);
      const auto bn = run(bdir, true, tier, true, false, false);
      const auto bg = run(bdir, true, tier, true, true, true);
      std::printf("[qwen_image_21_dit]   bf16 checkpoint: vs its dense "
                  "forward %s %.3e, accurate %.3e; streamed %s, no slots "
                  "%s, promoted %s\n", tname[tier], rel(bf, bd),
                  rel(ba, bd), bs == bf ? "IDENTICAL" : "DIFFERENT",
                  bn == bf ? "IDENTICAL" : "DIFFERENT",
                  bg == bf ? "IDENTICAL" : "DIFFERENT");
      EXPECT_TRUE(engaged);
      EXPECT_TRUE(rel(bf, bd) <= bar[tier] * rel(ba, bd));
      EXPECT_TRUE(bs == bf && bn == bf && bg == bf);
    }
  }
}

// A DRAW THINGS i8x CHECKPOINT (VPIPE_DRAWTHINGS_QI21_CKPT, Draw Things'
// qwen_image_2.1_i8x.ckpt) as the DiT: on the unrotated token tier its
// codes and row scales are read as stored -- it must ENGAGE there, be
// BYTE-IDENTICAL preloaded, streamed and with no slots (the slot refills
// the codes raw and rebuilds the f32 scales by name), and sit near the
// bf16 forward (dit_dir_()) -- and with int8 off it runs DENSE from the
// same file, dequantized on the way out. Same geometry as the tier test.
TEST(qwen_image_21_dit, drawthings_i8x_runs_native)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_QI21_CKPT");
  const std::string bdir = dit_dir_();
  if (ck == nullptr || *ck == '\0' || bdir.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(ck, &cfg));
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, TS = 64;
  std::vector<std::uint8_t> slot((std::size_t)TS, 0);
  slot.resize((std::size_t)TS + (std::size_t)(gh * gw / 4), 1);
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, {{1, gh, gw}}, &lay, &lerr));
  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
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
  bool native = true;
  auto run = [&](const std::string& dir, bool i8, int tier, bool stream,
                 bool slots) {
    std::vector<std::uint16_t> v;
    std::unique_ptr<Env> e2;
    if (!slots) {
      e2 = std::make_unique<Env>("VPIPE_QWEN_IMAGE21_NO_SLOTS", "1");
    }
    MetalQwenImage21Transformer::Config c = cfg;
    c.i8_gemm = i8;
    c.i8_tier = tier;
    auto m = MetalQwenImage21Transformer::load(dir, mc, c, stream);
    if (m == nullptr) { return v; }
    if (i8 && tier == 3 &&
        m->fast_group() != I8GemmContext::kTokenGroup) {
      native = false;
    }
    metal_compute::SharedBuffer lat =
        rnd((std::size_t)lay.image_len * cfg.in_channels, 11u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)slot.size() * cfg.txt_dim, 22u);
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
    if (out.empty() || out.byte_size() < n * 2) { return v; }
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    v.assign(d, d + n);
    return v;
  };
  auto rel = [&](const std::vector<std::uint16_t>& a,
                 const std::vector<std::uint16_t>& b) {
    auto f = [](std::uint16_t h) {
      const std::uint32_t u = (std::uint32_t)h << 16;
      float x;
      std::memcpy(&x, &u, 4);
      return (double)x;
    };
    double n2 = 0, d2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      n2 += (f(a[i]) - f(b[i])) * (f(a[i]) - f(b[i]));
      d2 += f(b[i]) * f(b[i]);
    }
    return std::sqrt(n2 / d2);
  };
  const auto dense = run(bdir, false, 0, false, true);
  const auto acc = run(bdir, true, 0, false, true);
  const auto pre = run(ck, true, 3, false, true);
  const auto str = run(ck, true, 3, true, true);
  const auto flat = run(ck, true, 3, true, false);
  const auto deq = run(ck, false, 0, false, true);
  const auto rot = run(ck, true, 2, false, true);
  Q21D_REQUIRE(!dense.empty() && acc.size() == dense.size() &&
               pre.size() == dense.size() && deq.size() == dense.size() &&
               rot.size() == dense.size());
  EXPECT_TRUE(native);
  const double ra = rel(acc, dense), rt = rel(pre, dense);
  const double rd = rel(deq, dense), rr = rel(rot, dense);
  std::printf("[qwen_image_21_dit] Draw Things i8x, %d+%d rows: vs the "
              "bf16 forward native token %.3e, requantized rotated %.3e, "
              "dense from the file %.3e (bf16 accurate tier %.3e); "
              "streamed %s, no slots %s\n",
              gh * gw, TS, rt, rr, rd, ra,
              str == pre ? "IDENTICAL" : "DIFFERENT",
              flat == pre ? "IDENTICAL" : "DIFFERENT");
  EXPECT_TRUE(str == pre);
  EXPECT_TRUE(flat == pre);
  // Gross-error bars (the forward is chaotic on random inputs): the
  // file's weights are an int8 per row, and the native route adds the
  // per-token activation on top.
  EXPECT_TRUE(rd <= 4.0 * ra);
  EXPECT_TRUE(rt <= 5.0 * ra);
  EXPECT_TRUE(rr <= 5.0 * ra);
}

// THE SIGNED LAYOUT'S WORTH in the Qwen-Image-2.1 forward, i8_gemm on, at
// 1024^2 (4096 target rows + a 42-token prompt), preloaded. Arms: bf16
// (VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH's transformer/), the quantized pack
// (VPIPE_QWEN_IMAGE21_QUANT_DIT) signed, and the pack on the route it took
// before (dequant + requant, VPIPE_I8_SIGNED=0). Each arm warms two
// forwards and keeps the median of three; the arms rotate through
// VPIPE_QWEN_IMAGE21_SIGNED_BENCH_ROUNDS rounds, reversed every other one.
// VPIPE_QWEN_IMAGE21_SIGNED_BENCH=1 runs it.
TEST(qwen_image_21_dit, signed_layout_forward)
{
  const char* qd = std::getenv("VPIPE_QWEN_IMAGE21_QUANT_DIT");
  const std::string bdir = dit_dir_();
  if (qd == nullptr || *qd == '\0' || bdir.empty() ||
      std::getenv("VPIPE_QWEN_IMAGE21_SIGNED_BENCH") == nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  int rounds = 3;
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_SIGNED_BENCH_ROUNDS")) {
    rounds = std::max(1, std::atoi(e));
  }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(qd, &cfg));
  std::vector<std::uint8_t> slot(42, 0);
  slot.resize(42 + 1024, 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, 64, 64}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  struct Arm { const char* name; std::string dir; bool sgn; };
  const Arm arms[] = {{"bf16", bdir, true},
                      {"w8 signed", qd, true},
                      {"w8 requant", qd, false}};
  std::vector<std::vector<double>> per(3);
  for (int r = 0; r < rounds; ++r) {
    for (int k = 0; k < 3; ++k) {
      const int a = (r % 2 == 0) ? k : 2 - k;
      if (!arms[a].sgn) { ::setenv("VPIPE_I8_SIGNED", "0", 1); }
      MetalQwenImage21Transformer::Config c = cfg;
      c.i8_gemm = true;
      auto m = MetalQwenImage21Transformer::load(arms[a].dir, mc, c, false);
      ::unsetenv("VPIPE_I8_SIGNED");
      Q21D_REQUIRE(m != nullptr);
      metal_compute::SharedBuffer lat = mc->make_shared_buffer(
          (std::size_t)lay.image_len * cfg.in_channels * 2);
      metal_compute::SharedBuffer txt = mc->make_shared_buffer(
          slot.size() * (std::size_t)cfg.txt_dim * 2);
      Q21D_REQUIRE(!lat.empty() && !txt.empty());
      std::memset(lat.contents(), 0x3c, lat.byte_size());
      std::memset(txt.contents(), 0x3c, txt.byte_size());
      MetalQwenImage21Transformer::Request req;
      req.latents = &lat;
      req.txt = &txt;
      req.layout = &lay;
      req.timestep = 0.75f;
      std::string ferr;
      for (int w = 0; w < 2; ++w) {
        Q21D_REQUIRE(!m->forward(req, &ferr).empty());
      }
      std::vector<double> t;
      for (int i = 0; i < 3; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        Q21D_REQUIRE(!m->forward(req, &ferr).empty());
        t.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(t.begin(), t.end());
      per[(std::size_t)a].push_back(t[1]);
      std::printf("[qwen_image_21_dit] round %d %-10s %7.1f ms/forward "
                  "(signed layout %s)\n", r, arms[a].name, t[1],
                  m->signed_layout() ? "on" : "off");
    }
  }
  double mean[3] = {0, 0, 0};
  for (int a = 0; a < 3; ++a) {
    for (double v : per[(std::size_t)a]) { mean[a] += v / rounds; }
  }
  for (int a = 0; a < 3; ++a) {
    std::printf("[qwen_image_21_dit]   %-10s %7.1f ms (mean of %d)  %.3fx "
                "bf16\n", arms[a].name, mean[a], rounds, mean[0] / mean[a]);
  }
}

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

// THE DENSE PATH AGAINST THE REFERENCE, ON REAL WEIGHTS. A released
// transformer's trunk and first blocks, cut out by
//   dump_qwen_image21_golden.py --what dit --dit-from <transformer>
//       --dit-layers 8 --dit-grid 48 --dit-tag real8
// with the reference run in fp32 AND in bf16 (`ref_bf16`, the bar a bf16
// forward is held to). VPIPE_QWEN_IMAGE21_REAL_GOLDEN names the dir
// (VPIPE_QWEN_IMAGE21_REAL_TAG the tag, real8 by default). A 48x48
// target puts 2304 image rows in the joint, so every block projection
// clears matmul2d's floor AND the int8 one -- the rows the two paths
// differ on. Random weights cannot show a checkpoint's own magnitudes
// and outlier channels; these can.
//
// Every arm runs the same bytes: dense (matmul2d + NAX on matrix cores,
// steel elsewhere), then each i8_gemm tier where there are matrix cores.
// Dense is held to the reference's own bf16 distance; an int8 arm is
// asserted ENGAGED (not bit-identical to dense) and reported, since it
// is a lossy tier by design.
TEST(qwen_image_21_dit, real_weights_dense_and_int8_against_reference)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_REAL_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const char* te = std::getenv("VPIPE_QWEN_IMAGE21_REAL_TAG");
  const std::string tag = (te != nullptr && *te != '\0') ? te : "real8";
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / (tag + ".json"));
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
  const double bar = o.contains("ref_bf16") ? o.at("ref_bf16").as_real(0.0)
                                            : 0.0;
  Q21D_REQUIRE(bar > 0.0);
  auto read_f32 = [&](const std::string& nm) {
    std::ifstream in(gd / (nm + ".f32"), std::ios::binary);
    std::vector<float> v;
    if (!in) { return v; }
    in.seekg(0, std::ios::end);
    const std::streamoff nn = in.tellg();
    in.seekg(0, std::ios::beg);
    v.resize((std::size_t)nn / 4);
    in.read(reinterpret_cast<char*>(v.data()), nn);
    return v;
  };
  const std::vector<float> lat_f = read_f32(tag + "_latents");
  const std::vector<float> txt_f = read_f32(tag + "_txt");
  const std::vector<float> want = read_f32(tag + "_out");
  const std::vector<float> want16 = read_f32(tag + "_out_bf16");
  Q21D_REQUIRE(!lat_f.empty() && !txt_f.empty() && !want.empty() &&
               want16.size() == want.size());

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
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
  Q21D_REQUIRE(lay.joint_len >= 1024);   // both floors cleared
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
  auto rel_to = [&](const std::vector<float>& a,
                    const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      const double e = (double)a[i] - (double)b[i];
      num += e * e;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const std::string ckpt = (gd / tag).string();
  // tier < 0: dense.
  auto run = [&](int tier, std::string* what) {
    std::vector<float> got;
    MetalQwenImage21Transformer::Config cfg;
    if (!MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg)) {
      return got;
    }
    cfg.i8_gemm = tier >= 0;
    cfg.i8_tier = tier < 0 ? 0 : tier;
    auto m = MetalQwenImage21Transformer::load(ckpt, mc, cfg, false);
    if (m == nullptr) { return got; }
    if (tier >= 0 && !m->uses_i8_gemm()) { return got; }
    *what = fmt("{}{}{}", m->uses_mma2() ? "matmul2d" : "steel",
                m->uses_attn_nax() ? " + NAX attention" : "",
                tier < 0 ? "" : " + i8")();
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = (float)(o.contains("timestep")
                               ? o.at("timestep").as_real(0.75) : 0.75);
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < want.size() * 2) {
      std::printf("[qwen_image_21_dit] forward failed: %s\n", ferr.c_str());
      return got;
    }
    got.resize(want.size());
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    for (std::size_t i = 0; i < want.size(); ++i) {
      const std::uint32_t u = (std::uint32_t)d[i] << 16;
      std::memcpy(&got[i], &u, 4);
    }
    return got;
  };
  std::string what;
  const std::vector<float> dense = run(-1, &what);
  Q21D_REQUIRE(!dense.empty());
  const double rd = rel_to(dense, want);
  std::printf("[qwen_image_21_dit] real weights, %d blocks, joint %d: the "
              "reference's own bf16 %.5f from its fp32\n",
              (int)o.at("layers").as_int(0), lay.joint_len, bar);
  std::printf("[qwen_image_21_dit]   dense (%s): %.5f from fp32, %.5f from "
              "the reference's bf16\n", what.c_str(), rd,
              rel_to(dense, want16));
  // A bf16 forward at the reference's own bf16 distance, give or take
  // the summation order of a different machine.
  EXPECT_TRUE(rd <= 1.25 * bar);
  if (!mc->supports_matrix_cores()) { return; }
  const char* tiers[] = {"accurate", "fast", "token", "token_plain"};
  for (int t = 0; t < 4; ++t) {
    std::string w8;
    const std::vector<float> got = run(t, &w8);
    if (got.empty()) {
      std::printf("[qwen_image_21_dit]   i8 %s: unavailable\n", tiers[t]);
      continue;
    }
    const bool same = std::memcmp(got.data(), dense.data(),
                                  got.size() * sizeof(float)) == 0;
    const double r = rel_to(got, want);
    std::printf("[qwen_image_21_dit]   i8 %-11s (%s): %.5f from fp32 "
                "(%.2fx dense), %.5f from dense%s\n", tiers[t], w8.c_str(),
                r, rd > 0.0 ? r / rd : 0.0, rel_to(got, dense),
                same ? " -- IDENTICAL TO DENSE: NOT ENGAGED" : "");
    EXPECT_TRUE(!same);
  }
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
  //
  // Opt-in, for the machines the tier is aimed at:
  //   VPIPE_QWEN_IMAGE21_ANE_BENCH=<reps>  hold both arms, warm them, and
  //       time their forwards INTERLEAVED, order alternating per rep
  //   VPIPE_QWEN_IMAGE21_ANE_PX=<px>       square size (default 1024)
  //   VPIPE_QWEN_IMAGE21_ANE_STREAM=0      hold the blocks (default: stream,
  //       which is what a 16 GB box must do)
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int px = envi("VPIPE_QWEN_IMAGE21_ANE_PX", 1024);
  const bool stream = envi("VPIPE_QWEN_IMAGE21_ANE_STREAM", 1) != 0;
  const int reps = envi("VPIPE_QWEN_IMAGE21_ANE_BENCH", 0);
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
  metal_compute::SharedBuffer lat =
      rnd((std::size_t)lay.image_len * cfg.in_channels, 61u);
  metal_compute::SharedBuffer txt =
      rnd((std::size_t)lay.text_len * cfg.txt_dim, 62u);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());
  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = 0.75f;
  // Three arms: the GPU alone, the feed-forward on the ANE, and the
  // feed-forward AND q/k/v on it -- so each tier is measured against
  // the one before it, not only against nothing.
  enum Arm { kGpu, kFf, kQkv, kArms };
  const char* const kName[kArms] = {"GPU-only", "ANE ff", "ANE ff+qkv"};
  // VPIPE_QWEN_IMAGE21_ANE_LORA=<adapter>[:scale]: the same runtime
  // adapter on EVERY arm. The GPU applies it as side GEMMs and the
  // ANE's rows take it merged into their staged weights, so the three
  // still agree -- where rows staged WITHOUT it would run the base
  // model under a GPU running the adapted one, the adapter at part
  // strength and no error anywhere.
  std::vector<MetalQwenImage21Transformer::LoraSpec> loras;
  if (const char* lp = std::getenv("VPIPE_QWEN_IMAGE21_ANE_LORA")) {
    std::string v(lp);
    MetalQwenImage21Transformer::LoraSpec sp;
    const std::size_t c = v.rfind(':');
    sp.path  = c == std::string::npos ? v : v.substr(0, c);
    sp.scale = c == std::string::npos ? 1.0f
                                      : (float)std::atof(v.c_str() + c + 1);
    loras.push_back(sp);
  }
  auto make = [&](int arm) {
    MetalQwenImage21Transformer::Config c = cfg;
    c.ane_ffn = arm != kGpu;
    c.ane_qkv = arm == kQkv;
    return MetalQwenImage21Transformer::load(dir, mc, c, stream, loras);
  };
  auto fwd = [&](MetalQwenImage21Transformer* m, std::vector<float>* outv) {
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < n * 2) {
      std::printf("[qwen_image_21_dit] forward failed: %s\n", ferr.c_str());
      return false;
    }
    if (outv == nullptr) { return true; }
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
  // Every arm held at once, so a bench can interleave them.
  std::unique_ptr<MetalQwenImage21Transformer> m[kArms];
  std::vector<float> out[kArms];
  for (int a = 0; a < kArms; ++a) {
    m[a] = make(a);
    Q21D_REQUIRE(m[a] != nullptr);
    Q21D_REQUIRE(fwd(m[a].get(), &out[a]));
  }
  // Asked AFTER the forward: the modules are created on the first one,
  // when the row count is finally known.
  EXPECT_TRUE(!m[kGpu]->ane_armed() && !m[kGpu]->ane_qkv_armed());
  EXPECT_TRUE(!m[kFf]->ane_qkv_armed());
  if (!m[kFf]->ane_armed()) {
    // No ANE on this box, or the module would not compile: the tier
    // declining is a valid outcome and not a failure.
    std::printf("[qwen_image_21_dit] ANE did not arm -- SKIPPED\n");
    return;
  }
  // ENGAGED before accurate: the q/k/v arm has to have built its own
  // module, or its numbers are the feed-forward arm's again.
  EXPECT_TRUE(m[kQkv]->ane_armed() && m[kQkv]->ane_qkv_armed());
  auto rel = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      const double e = (double)a[i] - (double)b[i];
      num += e * e;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const double r_ff = rel(out[kFf], out[kGpu]);
  const double r_qkv = rel(out[kQkv], out[kGpu]);
  std::printf("[qwen_image_21_dit] ANE split vs GPU-only: rel-L2 %.4e ff, "
              "%.4e ff+qkv (joint %d, %d layers%s)\n", r_ff, r_qkv,
              lay.joint_len, cfg.n_layers,
              loras.empty() ? "" : ", runtime LoRA on every arm");
  if (!loras.empty()) {
    EXPECT_TRUE(m[kGpu]->lora_modules(0) > 0);
  }
  // The same function over disjoint rows, one side fp16: a rounding
  // difference through 32 blocks, not a different answer. The q/k/v
  // rows are fp16 too, and are normalized per head right after, so
  // they add rounding and nothing else.
  EXPECT_TRUE(r_ff < 0.05);
  EXPECT_TRUE(r_qkv < 0.05);

  // THE GPU FALLBACK, forced (VPIPE_ANE_FAIL: the tiers read it when the
  // first forward builds them). Two cases, both tiers at once:
  //
  //  * the FIRST split loses every ANE row (chunk 0). The rows go back
  //    to the GPU and the tiers stay there, so the whole forward is the
  //    GPU's -- it matches the GPU arm far closer than any split does,
  //    which is what proves no later block split again;
  //  * a PINNED two-chunk share loses only its SECOND chunk. The feed-
  //    forward writes in place, so this is the case where rows before
  //    the loss are already outputs and rows after it must still be
  //    inputs: recomputing the wrong range lands far from the GPU arm.
  auto finite = [](const std::vector<float>& v) {
    for (float f : v) {
      if (!std::isfinite(f)) { return false; }
    }
    return !v.empty();
  };
  auto failing = [&](const char* spec, int rows,
                     std::vector<float>* outv, bool* latched) {
    MetalQwenImage21Transformer::Config c = cfg;
    c.ane_ffn  = true;
    c.ane_qkv  = true;
    c.ane_rows = rows;
    auto mf = MetalQwenImage21Transformer::load(dir, mc, c, stream, loras);
    if (!mf) { return false; }
    setenv("VPIPE_ANE_FAIL", spec, 1);
    const bool ok = fwd(mf.get(), outv);
    unsetenv("VPIPE_ANE_FAIL");
    *latched = mf->ane_gpu_only() && mf->ane_qkv_gpu_only();
    return ok;
  };
  std::vector<float> f0, f1;
  bool l0 = false, l1 = false;
  Q21D_REQUIRE(failing("0:0", 0, &f0, &l0));
  Q21D_REQUIRE(failing("0:1", 1, &f1, &l1));
  const double r_f0 = rel(f0, out[kGpu]);
  const double r_f1 = rel(f1, out[kGpu]);
  std::printf("[qwen_image_21_dit] ANE rows lost -> GPU: rel-L2 %.4e "
              "(first split, every row), %.4e (second chunk of a pinned "
              "share) against GPU-only\n", r_f0, r_f1);
  EXPECT_TRUE(l0);                 // engaged: both tiers fell back
  EXPECT_TRUE(l1);
  EXPECT_TRUE(finite(f0));
  EXPECT_TRUE(finite(f1));
  EXPECT_TRUE(r_f0 < 1e-6);        // every row the GPU's
  EXPECT_TRUE(r_f0 < 0.01 * r_qkv);
  EXPECT_TRUE(r_f1 < 0.05);        // the right rows were recomputed
  EXPECT_TRUE(r_f1 > r_f0);        // and the ANE's first chunk kept

  if (reps <= 0) { return; }
  double t[kArms] = {};
  std::vector<double> per[kArms];
  for (int k = 0; k < reps; ++k) {
    // The order ROTATES per rep, so a clock that drifts over the run
    // charges every arm alike.
    for (int j = 0; j < kArms; ++j) {
      const int a = (j + k) % kArms;
      const auto t0 = std::chrono::steady_clock::now();
      Q21D_REQUIRE(fwd(m[a].get(), nullptr));
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      t[a] += ms;
      per[a].push_back(ms);
    }
  }
  auto list = [](const std::vector<double>& v) {
    std::string s;
    for (double x : v) { s += std::to_string((int)(x + 0.5)) + " "; }
    return s;
  };
  std::printf("[qwen_image_21_dit] ANE bench %dx%d%s, %d reps rotated: "
              "%s %.0f, %s %.0f (%.3fx), %s %.0f ms/fwd (%.3fx; %.3fx over "
              "ff) | %s| %s| %s\n", px, px, stream ? " streamed" : "",
              reps, kName[kGpu], t[kGpu] / reps, kName[kFf], t[kFf] / reps,
              t[kGpu] / t[kFf], kName[kQkv], t[kQkv] / reps,
              t[kGpu] / t[kQkv], t[kFf] / t[kQkv], list(per[kGpu]).c_str(),
              list(per[kFf]).c_str(), list(per[kQkv]).c_str());
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

// Two checkpoints of the SAME model, one forward each on the same
// request: an FP8 file against its own dense conversion, which have to
// agree to the BIT (shared/fp8-expand.h), or against the bf16 original,
// to a stated rel-L2.
//
// Env: VPIPE_QWEN_IMAGE21_CKPT_A / _B (a transformer directory or a
// single-file DiT); _LAYERS (default 4); _STREAM=1 to stream the blocks,
// which with 3+ layers takes the slot refill; _PX (default 512); _TOL, a
// rel-L2 bound in place of bit-identity.
TEST(qwen_image_21_dit, checkpoints_agree)
{
  const char* pa = std::getenv("VPIPE_QWEN_IMAGE21_CKPT_A");
  const char* pb = std::getenv("VPIPE_QWEN_IMAGE21_CKPT_B");
  if (pa == nullptr || *pa == '\0' || pb == nullptr || *pb == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int layers = envi("VPIPE_QWEN_IMAGE21_CKPT_LAYERS", 4);
  const bool stream = envi("VPIPE_QWEN_IMAGE21_CKPT_STREAM", 0) != 0;
  const int px = envi("VPIPE_QWEN_IMAGE21_CKPT_PX", 512);
  const char* tol_s = std::getenv("VPIPE_QWEN_IMAGE21_CKPT_TOL");
  const int lg = px / 16, n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  auto run = [&](const char* dir) -> std::vector<float> {
    MetalQwenImage21Transformer::Config cfg;
    if (!MetalQwenImage21Transformer::Config::read_dims(dir, &cfg)) {
      std::printf("[checkpoints_agree] %s: read_dims failed\n", dir);
      return {};
    }
    std::printf("[checkpoints_agree] %s: hidden %d heads %d layers %d "
                "mlp_ratio %d in %d out %d txt %d\n", dir, cfg.hidden,
                cfg.n_heads, cfg.n_layers, cfg.mlp_ratio, cfg.in_channels,
                cfg.out_channels, cfg.txt_dim);
    cfg.n_layers = std::min(cfg.n_layers, layers);
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
        rnd((std::size_t)lay.image_len * cfg.in_channels, 61u);
    metal_compute::SharedBuffer txt =
        rnd((std::size_t)lay.text_len * cfg.txt_dim, 62u);
    MetalQwenImage21Transformer::Request req;
    req.latents = &lat;
    req.txt = &txt;
    req.layout = &lay;
    req.timestep = 0.75f;
    auto m = MetalQwenImage21Transformer::load(dir, mc, cfg, stream);
    if (m == nullptr) {
      std::printf("[checkpoints_agree] %s: load failed\n", dir);
      return {};
    }
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
    if (out.empty() || out.byte_size() < n * 2) {
      std::printf("[checkpoints_agree] %s: forward: %s\n", dir, ferr.c_str());
      return {};
    }
    std::vector<float> v(n);
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)d[i] << 16;
      std::memcpy(&v[i], &u, 4);
    }
    return v;
  };
  const std::vector<float> a = run(pa);
  const std::vector<float> b = run(pb);
  Q21D_REQUIRE(!a.empty() && a.size() == b.size());
  double num = 0.0, den = 0.0, mx = 0.0;
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
    mx = std::max(mx, std::fabs(d));
    diff += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
  std::printf("[checkpoints_agree] %zu of %zu differ, max |d| %.3g, rel-L2 "
              "%.3e\n", diff, a.size(), mx, rel);
  if (tol_s != nullptr) {
    EXPECT_TRUE(rel <= std::atof(tol_s));
  } else {
    EXPECT_TRUE(diff == 0);
  }
}

// Runtime LoRA against the same adapter FUSED into the base weights --
// W + (alpha/rank) B A, computed in f32 and rounded once to bf16 by a
// script outside this tree -- over the same blocks, on the same
// request. Asserts what an adapter test has to, in order:
//   it BOUND modules (an adapter for another model binds none and says
//     nothing), and as many as the caller expects when told;
//   it MOVES the base well past the bf16 floor, or "matches the fused
//     reference" would be vacuous;
//   runtime and fused agree to within that floor;
//   a streamed stack takes the same adapters as a held one, to the bit;
//   scale 0 reproduces the base EXACTLY, not to rounding.
//
// Env: VPIPE_QWEN_IMAGE21_LORA_BASE (the base transformer), _LORA (the
// adapter), _LORA_FUSED (base + adapter, fused); _LORA_LAYERS (default
// 4, the fused file's depth); _LORA_EXPECT, the modules the adapter must
// bind at that depth; _LORA_PX (default 512).
TEST(qwen_image_21_dit, runtime_lora_matches_fused)
{
  const char* pbase = std::getenv("VPIPE_QWEN_IMAGE21_LORA_BASE");
  const char* plora = std::getenv("VPIPE_QWEN_IMAGE21_LORA");
  const char* pfuse = std::getenv("VPIPE_QWEN_IMAGE21_LORA_FUSED");
  if (pbase == nullptr || plora == nullptr || pfuse == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int layers = envi("VPIPE_QWEN_IMAGE21_LORA_LAYERS", 4);
  const int expect = envi("VPIPE_QWEN_IMAGE21_LORA_EXPECT", 0);
  const int px = envi("VPIPE_QWEN_IMAGE21_LORA_PX", 512);
  const int lg = px / 16, n_text = 64;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, lg, lg}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(pbase, &cfg));
  cfg.n_layers = std::min(cfg.n_layers, layers);
  auto rnd = [&](std::size_t n, unsigned seed) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
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
      rnd((std::size_t)lay.image_len * cfg.in_channels, 61u);
  metal_compute::SharedBuffer txt =
      rnd((std::size_t)lay.text_len * cfg.txt_dim, 62u);
  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = 0.75f;
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;
  auto fwd = [&](MetalQwenImage21Transformer* m) -> std::vector<float> {
    std::string ferr;
    metal_compute::SharedBuffer out = m->forward(req, &ferr);
    if (out.empty() || out.byte_size() < n * 2) {
      std::printf("[runtime_lora] forward failed: %s\n", ferr.c_str());
      return {};
    }
    std::vector<float> v(n);
    const auto* d = static_cast<const std::uint16_t*>(out.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)d[i] << 16;
      std::memcpy(&v[i], &u, 4);
    }
    return v;
  };
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      num += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
      den += (double)b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
  };
  // _LORA_SCALE: the strength both here and in the fused reference (the
  // fuse script's own argument) -- how a rounding floor, which stays
  // put, is told apart from a mapping error, which grows with it.
  const char* sc_s = std::getenv("VPIPE_QWEN_IMAGE21_LORA_SCALE");
  const float lscale = sc_s != nullptr ? (float)std::atof(sc_s) : 1.0f;
  const std::vector<MetalQwenImage21Transformer::LoraSpec> specs = {
      {plora, lscale}};
  auto base_m = MetalQwenImage21Transformer::load(pbase, mc, cfg, false);
  Q21D_REQUIRE(base_m != nullptr);
  const std::vector<float> base = fwd(base_m.get());
  base_m.reset();
  auto lora_m = MetalQwenImage21Transformer::load(pbase, mc, cfg, false,
                                                  specs);
  Q21D_REQUIRE(lora_m != nullptr);
  const int bound = lora_m->lora_modules(0);
  const std::vector<float> got = fwd(lora_m.get());
  MetalQwenImage21Transformer::Config fcfg = cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(pfuse, &fcfg));
  fcfg.n_layers = cfg.n_layers;
  auto fused_m = MetalQwenImage21Transformer::load(pfuse, mc, fcfg, false);
  Q21D_REQUIRE(fused_m != nullptr);
  const std::vector<float> want = fwd(fused_m.get());
  fused_m.reset();
  auto stream_m = MetalQwenImage21Transformer::load(pbase, mc, cfg, true,
                                                    specs);
  Q21D_REQUIRE(stream_m != nullptr);
  const std::vector<float> streamed = fwd(stream_m.get());
  stream_m.reset();
  lora_m->set_lora_scale(0, 0.0f);
  const std::vector<float> off = fwd(lora_m.get());
  Q21D_REQUIRE(!base.empty() && !got.empty() && !want.empty() &&
               !streamed.empty() && !off.empty());
  const double moved = rel(base, want);
  const double err = rel(got, want);
  std::printf("[runtime_lora] bound %d modules at %d blocks; the adapter "
              "moves the base %.4f; runtime vs fused %.4f; streamed vs "
              "held %.3e; scale 0 vs base %.3e\n", bound, cfg.n_layers,
              moved, err, rel(streamed, got), rel(off, base));
  EXPECT_TRUE(bound > 0);
  if (expect > 0) { EXPECT_TRUE(bound == expect); }
  // The floor is bf16 rounding -- the fused weight's one rounding
  // against the runtime path's rounded intermediate and double-rounded
  // accumulate -- and it does NOT grow with the strength: MEASURED on
  // Viggle's turbo (modulation + timestep Linears only), 0.0105 / 0.0172
  // / 0.0173 at scales 0.25 / 1 / 4 while the adapter's effect went
  // 0.018 / 0.039 / 0.117. So the bound is the floor, stated, and the
  // effect has to clear it several times over -- run a weak adapter at
  // a higher _LORA_SCALE (fused at the same) rather than loosen this.
  EXPECT_TRUE(moved > 0.02);
  EXPECT_TRUE(err < 0.03);
  EXPECT_TRUE(err < 0.25 * moved);
  EXPECT_TRUE(std::memcmp(streamed.data(), got.data(),
                          n * sizeof(float)) == 0);
  EXPECT_TRUE(std::memcmp(off.data(), base.data(), n * sizeof(float)) == 0);
}

// ---- eight runtime adapters at once -----------------------------------
//
// A DiT carries up to lora::Stack::kMax adapters at once (eight), each
// its own pair of GEMMs, its own region of the [rows, rank] scratch and
// its own live strength. Eight DIFFERENT synthetic adapters on the small
// model, and five things that hold only if the slots are really apart:
//
//   ENGAGEMENT  every slot binds, in the order given. Adapter i adapts
//               attn.to_out.0 in every block and, when i is odd,
//               img_mlp.out as well, so a slot that shifted would show
//               up as the wrong module count.
//   ORDER       naming them in reverse is the same function. A shared
//               scratch region would make the last writer's
//               intermediate what every B factor multiplies, and the
//               two orders would compute two different things.
//   SUM         the eight are ONE adapter: the same factors stacked on
//               the rank axis (A = [A0; ..; A7], B = [B0 | .. | B7]) in
//               a single slot is the same arithmetic up to rounding.
//   DIAL        slots 1..7 at strength 0 is adapter 0 alone, EXACTLY: a
//               zero strength encodes nothing.
//   HOLE        an adapter that cannot be opened keeps its slot, EMPTY,
//               so the ones after it keep their indices -- a live
//               strength is addressed by index, and a gap closed up
//               would send it to the next adapter.
//
// Env: VPIPE_QWEN_IMAGE21_GOLDEN (the 2-layer model with random weights,
// which is all this needs).
TEST(qwen_image_21_dit, eight_adapters_ride_one_dit)
{
  namespace fs = std::filesystem;
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const std::string ckpt = (fs::path(g) / "dit").string();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalQwenImage21Transformer::Config cfg;
  if (!MetalQwenImage21Transformer::Config::read_dims(ckpt, &cfg)) {
    return;
  }
  constexpr int kSlots = MetalQwenImage21Transformer::kMaxLoraSlots;
  EXPECT_TRUE(kSlots == 8);
  const int H = cfg.hidden, FF = cfg.hidden * cfg.mlp_ratio, R = 4;
  const int nl = cfg.n_layers;

  // Every factor, f32, distinct per (adapter, block, module, factor).
  // [rows, cols] row-major; spread chosen so each adapter visibly moves
  // the output on its own.
  auto factor = [](int rows, int cols, unsigned seed) {
    std::vector<float> v((std::size_t)rows * cols);
    unsigned st = seed * 2654435761u + 12345u;
    for (float& x : v) {
      st = st * 1664525u + 1013904223u;
      x = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.25f;
    }
    return v;
  };
  struct Mod { const char* leaf; int n, k; };
  const Mod kOut{"attn.to_out.0", H, H}, kMlp{"img_mlp.out", H, FF};
  auto adapts = [&](int i, const Mod& m) {
    return &m == &kOut || (i % 2) == 1;
  };
  auto a_of = [&](int i, int L, const Mod& m) {
    return factor(R, m.k, (unsigned)(i * 1000 + L * 10 + (&m == &kMlp)));
  };
  auto b_of = [&](int i, int L, const Mod& m) {
    return factor(m.n, R,
                  (unsigned)(i * 1000 + L * 10 + (&m == &kMlp) + 500));
  };
  auto bf16 = [](const std::vector<float>& v) {
    std::vector<std::uint8_t> out(v.size() * 2);
    for (std::size_t i = 0; i < v.size(); ++i) {
      const std::uint16_t h = train::f32_to_bf16(v[i]);
      std::memcpy(out.data() + 2 * i, &h, 2);
    }
    return out;
  };
  const fs::path dir = fs::temp_directory_path() / "vpipe-qi21-8-loras";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  auto pre = [](int L) {
    return "transformer_blocks." + std::to_string(L) + ".";
  };
  std::vector<std::string> files;
  for (int i = 0; i < kSlots; ++i) {
    std::vector<train::OutTensor> ts;
    for (int L = 0; L < nl; ++L) {
      for (const Mod* m : {&kOut, &kMlp}) {
        if (!adapts(i, *m)) { continue; }
        const std::string p = pre(L) + m->leaf;
        ts.push_back({p + ".lora_A.weight", "BF16", {R, m->k},
                      bf16(a_of(i, L, *m))});
        ts.push_back({p + ".lora_B.weight", "BF16", {m->n, R},
                      bf16(b_of(i, L, *m))});
      }
    }
    files.push_back((dir / ("a" + std::to_string(i) + ".safetensors"))
                        .string());
    std::string werr;
    Q21D_REQUIRE(train::write_safetensors(files.back(), ts, {}, &werr));
  }
  // The same factors stacked on the rank axis, one module at a time:
  // A rows are slot-major, B columns likewise.
  const std::string cat = (dir / "stacked.safetensors").string();
  {
    std::vector<train::OutTensor> ts;
    for (int L = 0; L < nl; ++L) {
      for (const Mod* m : {&kOut, &kMlp}) {
        std::vector<float> a, b;
        std::vector<std::vector<float>> bs;
        for (int i = 0; i < kSlots; ++i) {
          if (!adapts(i, *m)) { continue; }
          const std::vector<float> ai = a_of(i, L, *m);
          a.insert(a.end(), ai.begin(), ai.end());
          bs.push_back(b_of(i, L, *m));
        }
        const int rk = R * (int)bs.size();
        b.resize((std::size_t)m->n * rk);
        for (int n = 0; n < m->n; ++n) {
          for (std::size_t j = 0; j < bs.size(); ++j) {
            for (int r = 0; r < R; ++r) {
              b[(std::size_t)n * rk + j * R + r] =
                  bs[j][(std::size_t)n * R + r];
            }
          }
        }
        const std::string p = pre(L) + m->leaf;
        ts.push_back({p + ".lora_A.weight", "BF16", {rk, m->k}, bf16(a)});
        ts.push_back({p + ".lora_B.weight", "BF16", {m->n, rk}, bf16(b)});
      }
    }
    std::string werr;
    Q21D_REQUIRE(train::write_safetensors(cat, ts, {}, &werr));
  }

  // A layout the small model takes: 16 text rows, one 256^2 target.
  const int lg = 256 / 16, n_text = 16;
  std::vector<std::uint8_t> slot((std::size_t)n_text, 0);
  slot.insert(slot.end(), (std::size_t)(lg * lg / 4), 1);
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, {{1, lg, lg}}, &lay, &lerr));
  auto rnd = [&](std::size_t n, unsigned seed) {
    const std::vector<float> f = factor(1, (int)n, seed);
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (!b.empty()) {
      const std::vector<std::uint8_t> h = bf16(f);
      std::memcpy(b.contents(), h.data(), h.size());
    }
    return b;
  };
  metal_compute::SharedBuffer lat =
      rnd((std::size_t)lay.image_len * cfg.in_channels, 71u);
  metal_compute::SharedBuffer txt =
      rnd((std::size_t)lay.text_len * cfg.txt_dim, 72u);
  Q21D_REQUIRE(!lat.empty() && !txt.empty());
  MetalQwenImage21Transformer::Request req;
  req.latents = &lat;
  req.txt = &txt;
  req.layout = &lay;
  req.timestep = 0.75f;
  const std::size_t n = (std::size_t)lay.target_len * cfg.out_channels;

  using Specs = std::vector<MetalQwenImage21Transformer::LoraSpec>;
  std::vector<int> mods;
  int slots = 0;
  // One load and one forward; `turn` dials slots after the load.
  auto arm = [&](const Specs& specs, const std::vector<float>& turn,
                 std::vector<float>* out) {
    auto m =
        MetalQwenImage21Transformer::load(ckpt, mc, cfg, false, specs);
    if (m == nullptr) { return false; }
    for (std::size_t i = 0; i < turn.size(); ++i) {
      m->set_lora_scale((int)i, turn[i]);
    }
    slots = m->lora_slots();
    mods.assign((std::size_t)kSlots, 0);
    for (int i = 0; i < kSlots; ++i) {
      mods[(std::size_t)i] = m->lora_modules(i);
    }
    std::string ferr;
    metal_compute::SharedBuffer o = m->forward(req, &ferr);
    if (o.empty() || o.byte_size() < n * 2) {
      std::printf("[eight_adapters] forward failed: %s\n", ferr.c_str());
      return false;
    }
    out->resize(n);
    const auto* d = static_cast<const std::uint16_t*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) {
      (*out)[i] = train::bf16_to_f32(d[i]);
    }
    return true;
  };
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double e = (double)a[i] - (double)b[i];
      num += e * e;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  Specs fwd_specs, rev_specs;
  for (int i = 0; i < kSlots; ++i) {
    fwd_specs.push_back({files[(std::size_t)i], 1.0f});
    rev_specs.push_back({files[(std::size_t)(kSlots - 1 - i)], 1.0f});
  }
  std::vector<float> v_base, v_all, v_rev, v_cat, v_first, v_dial;
  Q21D_REQUIRE(arm({}, {}, &v_base));
  Q21D_REQUIRE(arm(fwd_specs, {}, &v_all));
  // ENGAGEMENT, in the order given.
  EXPECT_TRUE(slots == kSlots);
  bool order_ok = true;
  for (int i = 0; i < kSlots; ++i) {
    order_ok = order_ok && mods[(std::size_t)i] == nl * (1 + i % 2);
  }
  EXPECT_TRUE(order_ok);
  Q21D_REQUIRE(arm(rev_specs, {}, &v_rev));
  Q21D_REQUIRE(arm({{cat, 1.0f}}, {}, &v_cat));
  Q21D_REQUIRE(arm({{files[0], 1.0f}}, {}, &v_first));
  {
    std::vector<float> turn((std::size_t)kSlots, 0.0f);
    turn[0] = 1.0f;
    Q21D_REQUIRE(arm(fwd_specs, turn, &v_dial));
  }

  const double moved  = rel(v_all, v_base);
  const double first  = rel(v_first, v_base);
  const double order  = rel(v_rev, v_all);
  const double summed = rel(v_cat, v_all);
  const double dial   = rel(v_dial, v_first);
  std::printf("[eight_adapters] the eight move the output %.3e (adapter 0 "
              "alone %.3e) | reversed order %.3e | stacked on the rank axis "
              "%.3e | slots 1..7 at 0 vs adapter 0 alone %.3e\n", moved,
              first, order, summed, dial);
  // The adapters have to be DOING something, or every comparison below
  // is zero against zero.
  EXPECT_TRUE(moved > 1e-2 && first > 1e-3 && moved > first);
  // ORDER and SUM differ from the eight only in how many times y is
  // rounded to bf16 (eps 2^-8): MEASURED 9.6e-3 and 8.9e-3 on the M4
  // Pro against a move of 1.02. A slot dropped is ~1/8 of that move,
  // and a scratch region shared computes a different function, both
  // well above the bound.
  EXPECT_TRUE(order < 0.03 && order < 0.1 * moved);
  EXPECT_TRUE(summed < 0.03 && summed < 0.1 * moved);
  // DIAL: the same dispatches as adapter 0 alone.
  EXPECT_TRUE(dial == 0.0);

  // HOLE: slot 3 cannot be opened. It stays a slot that binds nothing,
  // and slot 4 is still the fifth file -- which, being even, binds only
  // the attention output.
  {
    Specs holed = fwd_specs;
    holed[3].path = (dir / "missing.safetensors").string();
    std::vector<float> v_holed, v_off3;
    Q21D_REQUIRE(arm(holed, {}, &v_holed));
    EXPECT_TRUE(slots == kSlots);
    EXPECT_TRUE(mods[3] == 0 && mods[4] == nl && mods[5] == 2 * nl);
    // ...and is the full set with adapter 3 turned off, to rounding:
    // the scratch layout differs (the hole has no rank), the GEMMs do
    // not.
    std::vector<float> turn((std::size_t)kSlots, 1.0f);
    turn[3] = 0.0f;
    Q21D_REQUIRE(arm(fwd_specs, turn, &v_off3));
    const double hole = rel(v_holed, v_off3);
    std::printf("[eight_adapters] a hole at slot 3 vs slot 3 dialled to 0: "
                "%.3e\n", hole);
    EXPECT_TRUE(hole == 0.0);
  }
  fs::remove_all(dir, ec);
}

// FORWARD A/B over environment arms, one model load per arm per round,
// interleaved with the order reversed every round, median of each arm's
// timed forwards. VPIPE_QI21_AB names the DiT; VPIPE_QI21_AB_ARMS is
// "name:ENV=v,ENV=v;name:..." (an arm with no assignments is the
// baseline); _ROUNDS, _SIDE (pixels, 1024), _CACHED=1 times cached steps
// (target rows against the kept prefix) as a generation's steps 2.. run,
// _STOP=1 installs a stop check as generate-image does, _STREAM=1 streams
// every block; an arm's VPIPE_I8_W4_HOLD=1 holds a w4 pack's int8 layout.
TEST(qwen_image_21_dit, forward_ab)
{
  const char* dir = std::getenv("VPIPE_QI21_AB");
  const char* spec = std::getenv("VPIPE_QI21_AB_ARMS");
  if (dir == nullptr || *dir == '\0' || spec == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  struct Arm {
    std::string name;
    std::vector<std::pair<std::string, std::string>> env;
    std::vector<double> ms;
  };
  std::vector<Arm> arms;
  {
    std::string s(spec);
    std::size_t p = 0;
    while (p <= s.size()) {
      const std::size_t q = std::min(s.find(';', p), s.size());
      const std::string a = s.substr(p, q - p);
      if (!a.empty()) {
        Arm arm;
        const std::size_t c = a.find(':');
        arm.name = a.substr(0, c);
        if (c != std::string::npos) {
          std::string kv = a.substr(c + 1);
          std::size_t r = 0;
          while (r < kv.size()) {
            const std::size_t e = std::min(kv.find(',', r), kv.size());
            const std::string one = kv.substr(r, e - r);
            const std::size_t eq = one.find('=');
            if (eq != std::string::npos) {
              arm.env.emplace_back(one.substr(0, eq), one.substr(eq + 1));
            }
            r = e + 1;
          }
        }
        arms.push_back(std::move(arm));
      }
      p = q + 1;
    }
  }
  int rounds = 2, side = 1024, fwd = 3;
  if (const char* e = std::getenv("VPIPE_QI21_AB_ROUNDS")) {
    rounds = std::max(1, std::atoi(e));
  }
  if (const char* e = std::getenv("VPIPE_QI21_AB_SIDE")) {
    side = std::max(256, std::atoi(e));
  }
  if (const char* e = std::getenv("VPIPE_QI21_AB_FORWARDS")) {
    fwd = std::max(1, std::atoi(e));
  }
  const bool cached = std::getenv("VPIPE_QI21_AB_CACHED") != nullptr;
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  // One slot per kImgTokensPerSlot (4) image tokens.
  std::vector<std::uint8_t> slot(42, 0);
  slot.resize(42 + (std::size_t)(side / 16) * (side / 16) / 4, 1);
  const std::vector<qi21::ImgBlock> blocks = {{1, side / 16, side / 16}};
  qi21::Layout lay;
  std::string lerr;
  Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
  // Every arm's last output against the FIRST arm's, as rel-L2 -- an arm
  // that changes the arithmetic says by how much.
  std::vector<float> ref_out;
  auto bf = [](std::uint16_t v) {
    const std::uint32_t u = (std::uint32_t)v << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  for (int r = 0; r < rounds; ++r) {
    for (std::size_t k = 0; k < arms.size(); ++k) {
      Arm& a = arms[(r % 2 == 0) ? k : arms.size() - 1 - k];
      for (const auto& kv : a.env) {
        ::setenv(kv.first.c_str(), kv.second.c_str(), 1);
      }
      MetalQwenImage21Transformer::Config c = cfg;
      c.i8_gemm = true;
      // A w4 pack's int8 layout held (VPIPE_I8_W4_HOLD=1) or converted
      // per GEMM (the model's default), as generate-image decides it.
      if (const char* h = std::getenv("VPIPE_I8_W4_HOLD")) {
        c.i8_hold_w4 = std::atoi(h) != 0;
      }
      // VPIPE_QI21_AB_STREAM=1 streams every block (no residency: each
      // forward reads the whole stack through the slot pair);
      // =retain streams, THEN RETAINS -- generate-image's default on a
      // box with room: a 20-step schedule, so residency keeps every block
      // its first pass reads (the wired pool on).
      const char* sm = std::getenv("VPIPE_QI21_AB_STREAM");
      const bool streamed = sm != nullptr;
      const bool retain = streamed && std::string(sm) == "retain";
      const auto tl0 = std::chrono::steady_clock::now();
      auto m = MetalQwenImage21Transformer::load(dir, mc, c, streamed);
      Q21D_REQUIRE(m != nullptr);
      if (streamed) { m->set_residency_reserve(0); }
      if (retain) { m->set_residency_schedule(20); }
      const double load_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - tl0)
                                 .count();
      // As generate-image drives it (VPIPE_QI21_AB_STOP=1): a stop check
      // installed, so a held stack drains every few blocks.
      if (std::getenv("VPIPE_QI21_AB_STOP") != nullptr) {
        m->set_stream_stop([] { return false; });
      }
      metal_compute::SharedBuffer lat = mc->make_shared_buffer(
          (std::size_t)lay.image_len * cfg.in_channels * 2);
      metal_compute::SharedBuffer txt = mc->make_shared_buffer(
          slot.size() * (std::size_t)cfg.txt_dim * 2);
      Q21D_REQUIRE(!lat.empty() && !txt.empty());
      auto* lp = static_cast<std::uint16_t*>(lat.contents());
      auto* tp = static_cast<std::uint16_t*>(txt.contents());
      std::uint32_t h = 12345u;
      auto rnd = [&]() {
        h = h * 1664525u + 1013904223u;
        const float f = ((float)(h >> 8) / 16777216.0f) * 2.0f - 1.0f;
        std::uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        return (std::uint16_t)(u >> 16);
      };
      for (std::size_t i = 0; i < lat.byte_size() / 2; ++i) { lp[i] = rnd(); }
      for (std::size_t i = 0; i < txt.byte_size() / 2; ++i) { tp[i] = rnd(); }
      MetalQwenImage21Transformer::KvCache kv;
      MetalQwenImage21Transformer::Request req;
      req.latents = &lat;
      req.txt = &txt;
      req.layout = &lay;
      req.timestep = 0.75f;
      std::string ferr;
      // The FIRST forward on its own: on a streamed model it is the one
      // that reads the stack (and, retaining, keeps it).
      double first_ms = 0.0;
      {
        const auto tf0 = std::chrono::steady_clock::now();
        if (cached) {
          req.kv = &kv;
          req.kv_mode = MetalQwenImage21Transformer::KvMode::kFill;
        }
        Q21D_REQUIRE(!m->forward(req, &ferr).empty());
        first_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - tf0)
                       .count();
        if (cached) {
          req.kv_mode = MetalQwenImage21Transformer::KvMode::kCached;
        }
      }
      for (int w = 0; w < 2; ++w) {
        Q21D_REQUIRE(!m->forward(req, &ferr).empty());
      }
      std::vector<double> t;
      metal_compute::SharedBuffer out;
      for (int i = 0; i < fwd; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        out = m->forward(req, &ferr);
        Q21D_REQUIRE(!out.empty());
        t.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(t.begin(), t.end());
      a.ms.push_back(t[t.size() / 2]);
      const auto* op = static_cast<const std::uint16_t*>(out.contents());
      const std::size_t on = out.byte_size() / 2;
      double rel = 0.0;
      if (ref_out.empty()) {
        ref_out.resize(on);
        for (std::size_t i = 0; i < on; ++i) { ref_out[i] = bf(op[i]); }
      } else if (ref_out.size() == on) {
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < on; ++i) {
          const double d = (double)bf(op[i]) - ref_out[i];
          num += d * d;
          den += (double)ref_out[i] * ref_out[i];
        }
        rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
      }
      std::printf("[qi21_ab] round %d %-14s %8.1f ms/forward (rows %d%s) "
                  "rel-L2 %.3g vs first; load %.0f ms, first forward %.0f "
                  "ms, %zu MB resident\n",
                  r, a.name.c_str(), t[t.size() / 2],
                  cached ? lay.target_len : lay.joint_len,
                  m->signed_layout() ? ", signed" : "", rel, load_ms,
                  first_ms, (std::size_t)(m->resident_bytes() >> 20));
      m.reset();
      for (const auto& kv2 : a.env) { ::unsetenv(kv2.first.c_str()); }
    }
  }
  const double base = [&] {
    double s = 0.0;
    for (double v : arms[0].ms) { s += v; }
    return s / (double)arms[0].ms.size();
  }();
  for (const Arm& a : arms) {
    double s = 0.0;
    for (double v : a.ms) { s += v; }
    s /= (double)a.ms.size();
    std::printf("[qi21_ab]   %-14s %8.1f ms (mean of %zu)  %.3fx %s\n",
                a.name.c_str(), s, a.ms.size(), base / s,
                arms[0].name.c_str());
  }
}

// DOES A REGULAR IMAGE FILL THE GPU? Two measurements on the cached step
// a generation's steps 2.. run. (1) Throughput against the picture: the
// forward's arithmetic (GEMMs 2*rows*(4H^2 + 3H*FF) a block, attention
// 4*rows*joint*H) over its time, per side in VPIPE_QI21_FILL_SIDES
// ("512,768,1024,1536,2048") -- a rate that keeps climbing with the side
// is a GPU the smaller pictures leave partly idle. (2) Spare capacity:
// VPIPE_QI21_FILL_STREAMS (2) model instances each running forwards on
// its own thread and queue at once, against one alone -- aggregate
// throughput above one stream's is capacity a single forward leaves on
// the table. VPIPE_QI21_FILL names the DiT.
TEST(qwen_image_21_dit, fill_study)
{
  const char* dir = std::getenv("VPIPE_QI21_FILL");
  if (dir == nullptr || *dir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::vector<int> sides = {512, 768, 1024, 1536, 2048};
  if (const char* e = std::getenv("VPIPE_QI21_FILL_SIDES")) {
    sides.clear();
    for (const char* p = e; *p != '\0';) {
      sides.push_back(std::atoi(p));
      while (*p != '\0' && *p != ',') { ++p; }
      if (*p == ',') { ++p; }
    }
  }
  int streams = 2;
  if (const char* e = std::getenv("VPIPE_QI21_FILL_STREAMS")) {
    streams = std::max(1, std::atoi(e));
  }
  MetalQwenImage21Transformer::Config cfg;
  Q21D_REQUIRE(MetalQwenImage21Transformer::Config::read_dims(dir, &cfg));
  cfg.i8_gemm = true;
  std::vector<std::unique_ptr<MetalQwenImage21Transformer>> ms;
  for (int i = 0; i < streams; ++i) {
    ms.push_back(MetalQwenImage21Transformer::load(dir, mc, cfg, false));
    Q21D_REQUIRE(ms.back() != nullptr);
  }
  const double H = cfg.hidden, FF = (double)cfg.hidden * cfg.mlp_ratio;
  for (int side : sides) {
    // One slot per kImgTokensPerSlot (4) image tokens.
    std::vector<std::uint8_t> slot(42, 0);
    slot.resize(42 + (std::size_t)(side / 16) * (side / 16) / 4, 1);
    const std::vector<qi21::ImgBlock> blocks = {{1, side / 16, side / 16}};
    qi21::Layout lay;
    std::string lerr;
    Q21D_REQUIRE(qi21::build_layout(slot, {}, blocks, &lay, &lerr));
    struct Run {
      metal_compute::SharedBuffer lat, txt;
      MetalQwenImage21Transformer::KvCache kv;
      MetalQwenImage21Transformer::Request req;
    };
    std::vector<Run> runs((std::size_t)streams);
    for (int i = 0; i < streams; ++i) {
      Run& r = runs[(std::size_t)i];
      r.lat = mc->make_shared_buffer((std::size_t)lay.image_len *
                                     cfg.in_channels * 2);
      r.txt = mc->make_shared_buffer(slot.size() * (std::size_t)cfg.txt_dim *
                                     2);
      Q21D_REQUIRE(!r.lat.empty() && !r.txt.empty());
      auto* lp = static_cast<std::uint16_t*>(r.lat.contents());
      auto* tp = static_cast<std::uint16_t*>(r.txt.contents());
      for (std::size_t k = 0; k < r.lat.byte_size() / 2; ++k) {
        lp[k] = (std::uint16_t)(0x3c00u + (k * 2654435761u >> 24) % 64u);
      }
      for (std::size_t k = 0; k < r.txt.byte_size() / 2; ++k) {
        tp[k] = (std::uint16_t)(0x3c00u + (k * 2246822519u >> 24) % 64u);
      }
      r.req.latents = &r.lat;
      r.req.txt = &r.txt;
      r.req.layout = &lay;
      r.req.timestep = 0.75f;
      r.req.kv = &r.kv;
      r.req.kv_mode = MetalQwenImage21Transformer::KvMode::kFill;
      std::string ferr;
      Q21D_REQUIRE(!ms[(std::size_t)i]->forward(r.req, &ferr).empty());
      r.req.kv_mode = MetalQwenImage21Transformer::KvMode::kCached;
      Q21D_REQUIRE(!ms[(std::size_t)i]->forward(r.req, &ferr).empty());
    }
    const double rows = lay.target_len, joint = lay.joint_len;
    const double flop = (double)cfg.n_layers *
        (2.0 * rows * (4.0 * H * H + 3.0 * H * FF) + 4.0 * rows * joint * H);
    const int fwd = side >= 1536 ? 3 : 6;
    // One stream alone.
    std::vector<double> t1;
    for (int k = 0; k < fwd; ++k) {
      std::string ferr;
      const auto t0 = std::chrono::steady_clock::now();
      Q21D_REQUIRE(!ms[0]->forward(runs[0].req, &ferr).empty());
      t1.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t1.begin(), t1.end());
    const double one = t1[t1.size() / 2];
    // All streams at once, each `fwd` forwards on its own thread.
    double all = 0.0;
    if (streams > 1) {
      std::vector<std::thread> th;
      std::atomic<int> bad{0};
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < streams; ++i) {
        th.emplace_back([&, i]() {
          for (int k = 0; k < fwd; ++k) {
            std::string ferr;
            if (ms[(std::size_t)i]->forward(runs[(std::size_t)i].req, &ferr)
                    .empty()) {
              ++bad;
            }
          }
        });
      }
      for (auto& t : th) { t.join(); }
      all = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
      Q21D_REQUIRE(bad.load() == 0);
    }
    std::printf("[qi21_fill] side %4d rows %5d joint %5d: one stream %8.1f "
                "ms = %6.1f TOP/s", side, lay.target_len, lay.joint_len, one,
                flop / one * 1e-9);
    if (streams > 1) {
      const double per = all / fwd;   // wall per round of `streams` forwards
      std::printf("; %d streams %8.1f ms a round = %6.1f TOP/s (%.2fx one)",
                  streams, per, flop * streams / per * 1e-9,
                  (one * streams) / per);
    }
    std::printf("\n");
  }
}
