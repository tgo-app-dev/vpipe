// Z-Image's DiT: the layout, the packing and the forward.
//
// Z-Image is a NextDiT / Lumina-Image-2.0 descendant -- the same
// lineage as Boogu-Image -- and six things about it are SILENT when
// wrong, which is what these pin:
//
//   * the sandwich norm (attention_norm2 / ffn_norm2 on the sublayer's
//     OUTPUT, inside the residual and under the gate),
//   * a block's adaLN being a BARE Linear while the final layer's has
//     a SiLU in front of it,
//   * the timestep running BACKWARDS (the model sees 1 - sigma),
//   * two learned pad tokens whose rotary POSITIONS pad differently,
//   * the sequence being cat([image, caption]) -- image FIRST,
//   * and the patch row's (pF, pH, pW, C) ordering.
//
// Env: VPIPE_Z_IMAGE_GOLDEN (tools/dump_z_image_golden.py) for the
// numeric tests; VPIPE_Z_IMAGE_TEST_MODEL_PATH (a real repo root) for
// the loader ones. Each tier skips vacuously without its variable, so
// check that the tests actually RAN.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "generative-models/z-image/metal-z-image-transformer.h"
#include "generative-models/z-image/z-image-layout.h"
#include "stages/model-quantize-stage.h"

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <atomic>
#include <map>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

// minitest's ASSERT_TRUE does NOT abort, so every follow-on step has to
// be guarded by hand.
// `cond` is evaluated ONCE. An ASSERT_TRUE(cond) followed by
// if (!(cond)) evaluates it twice, which is invisible for a
// comparison and doubles the work for a call -- a model loaded, or a
// forward run, twice per use.
#define ZI_REQUIRE(cond)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
golden_dir_()
{
  const char* g = std::getenv("VPIPE_Z_IMAGE_GOLDEN");
  return (g == nullptr || *g == '\0') ? std::string() : std::string(g);
}

std::vector<float>
read_f32_(const std::filesystem::path& p)
{
  std::vector<float> v;
  std::ifstream in(p, std::ios::binary);
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  v.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

float
bf16_(std::uint16_t h)
{
  std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& want)
{
  double num = 0.0, den = 0.0;
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

// ---- the layout, which needs no checkpoint --------------------------

TEST(z_image_layout, pads_each_half_to_32_and_places_the_positions)
{
  zimage::Layout L;
  std::string err;
  // 10 x 10 = 100 image tokens and 20 caption tokens: both pad.
  ZI_REQUIRE(zimage::build_layout(10, 10, 20, &L, &err));
  EXPECT_TRUE(L.img_tokens == 100);
  EXPECT_TRUE(L.img_total == 128);
  EXPECT_TRUE(L.cap_len == 20);
  EXPECT_TRUE(L.cap_total == 32);
  EXPECT_TRUE(L.joint_len == 160);
  EXPECT_TRUE(L.img_off() == 0 && L.cap_off() == 128);
  EXPECT_TRUE(L.img_pad() == 28 && L.cap_pad() == 12);

  // THE IMAGE IS FIRST. Row 0 is a picture row, not a caption row.
  EXPECT_TRUE(L.pos_t[0] == L.cap_total + 1);
  EXPECT_TRUE(L.pos_h[0] == 0 && L.pos_w[0] == 0);
  // ... and every real image row shares that one axis-0 coordinate,
  // with (h, w) walking the grid row-major.
  EXPECT_TRUE(L.pos_t[99] == L.cap_total + 1);
  EXPECT_TRUE(L.pos_h[99] == 9 && L.pos_w[99] == 9);
  EXPECT_TRUE(L.pos_h[10] == 1 && L.pos_w[10] == 0);

  // THE TWO HALVES PAD THEIR POSITIONS DIFFERENTLY, which is the whole
  // point of this assertion: image padding is pinned to the ORIGIN and
  // caption padding CONTINUES the caption's run.
  for (int r = 100; r < 128; ++r) {
    EXPECT_TRUE(L.pos_t[(std::size_t)r] == 0);
    EXPECT_TRUE(L.pos_h[(std::size_t)r] == 0);
    EXPECT_TRUE(L.pos_w[(std::size_t)r] == 0);
  }
  EXPECT_TRUE(L.pos_t[128] == 1);
  EXPECT_TRUE(L.pos_t[128 + 19] == 20);
  EXPECT_TRUE(L.pos_t[128 + 20] == 21);   // the first PAD row
  EXPECT_TRUE(L.pos_t[159] == 32);        // the last one
  for (int r = 128; r < 160; ++r) {
    EXPECT_TRUE(L.pos_h[(std::size_t)r] == 0);
    EXPECT_TRUE(L.pos_w[(std::size_t)r] == 0);
  }

  // An exact multiple pads to nothing.
  zimage::Layout E;
  ZI_REQUIRE(zimage::build_layout(8, 8, 32, &E, &err));
  EXPECT_TRUE(E.img_tokens == 64 && E.img_total == 64);
  EXPECT_TRUE(E.cap_total == 32 && E.joint_len == 96);

  // And the refusals, which exist so a caller sees which invariant it
  // broke rather than a plausible picture.
  zimage::Layout X;
  EXPECT_FALSE(zimage::build_layout(0, 8, 20, &X, &err));
  EXPECT_FALSE(zimage::build_layout(8, 8, 0, &X, &err));
  EXPECT_FALSE(zimage::build_layout(8, 8, 513, &X, &err));
}

TEST(z_image_layout, prompt_is_a_bare_user_turn_with_no_system_message)
{
  // Z-Image is the only image family here with NO system prompt: the
  // reference renders Qwen3's chat template over a single user message
  // with a generation turn and thinking ENABLED, which for Qwen3 adds
  // no forced `<think>` block.
  const std::string p = zimage::prompt_template("a cat");
  EXPECT_TRUE(p == "<|im_start|>user\na cat<|im_end|>\n"
                   "<|im_start|>assistant\n");
  EXPECT_TRUE(p.find("system") == std::string::npos);
  // The tap is the SECOND-to-last layer, expressed the way
  // forward_embeddings_taps indexes (capture AFTER layer L).
  EXPECT_TRUE(zimage::tap_layer(36) == 34);
}

// ---- patch packing --------------------------------------------------

TEST(z_image_layout, patch_packing_matches_the_reference)
{
  const std::string g = golden_dir_();
  if (g.empty()) { return; }
  const std::filesystem::path gd(g);
  std::ifstream mf(gd / "dit_manifest.json");
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
  const int C = geti("in_channels", 16), P = geti("patch", 2);
  const int lh = geti("lat_h", 20), lw = geti("lat_w", 20);

  const std::vector<float> latent = read_f32_(gd / "dit_latent.f32");
  const std::vector<float> want = read_f32_(gd / "dit_patches.f32");
  ZI_REQUIRE(!latent.empty() && !want.empty());
  ZI_REQUIRE(latent.size() == (std::size_t)C * lh * lw);

  std::vector<float> got(want.size(), 0.0f);
  zimage::pack_latents(latent.data(), C, lh, lw, P, got.data());
  // EXACT: this is a permutation, so anything but 0 is a wrong one.
  EXPECT_TRUE(rel_l2_(got, want) == 0.0);

  // And the inverse really is one.
  std::vector<float> back(latent.size(), 0.0f);
  zimage::unpack_latents(want.data(), C, lh, lw, P, back.data());
  EXPECT_TRUE(rel_l2_(back, latent) == 0.0);
}

// ---- the forward against a small reference model --------------------

namespace {

// Everything the golden tests need, loaded once per test.
struct Golden {
  bool ok = false;
  std::string tag;
  std::filesystem::path dir;
  int C = 16, P = 2, lh = 0, lw = 0, grid = 0, cap_len = 0, cap_dim = 0;
  int layers = 0, refiner = 0, heads = 0, dim = 0;
  float sigma = 0.75f;
  std::vector<float> latent, patches, cap, out;
  zimage::Layout lay;
};

// `tag` names BOTH the checkpoint directory and every data file, so a
// second dump into the same directory cannot overwrite the first's
// numbers while leaving its manifest standing. "dit" is the small
// model; "wide" is the one whose geometry clears the accel tiers'
// floors (i8 declines under M = 1024; Sol under 1024 rows or 16 key
// blocks), because a tier tested below its floor declines and the test
// passes having exercised nothing.
Golden
load_golden_(const char* tag = "dit")
{
  Golden G;
  G.tag = tag;
  const std::string g = golden_dir_();
  if (g.empty()) { return G; }
  G.dir = std::filesystem::path(g);
  std::ifstream mf(G.dir / (G.tag + "_manifest.json"));
  if (!mf) { return G; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return G;
  }
  if (!man.is_object()) { return G; }
  auto o = man.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  G.C = geti("in_channels", 16);
  G.P = geti("patch", 2);
  G.lh = geti("lat_h", 0);
  G.lw = geti("lat_w", 0);
  G.grid = geti("grid", 0);
  G.cap_len = geti("cap_len", 0);
  G.cap_dim = geti("cap_dim", 0);
  G.layers = geti("layers", 0);
  G.refiner = geti("refiner", 0);
  G.heads = geti("heads", 0);
  G.dim = geti("dim", 0);
  G.sigma = (float)(o.contains("sigma") ? o.at("sigma").as_real(0.75) : 0.75);
  G.latent = read_f32_(G.dir / (G.tag + "_latent.f32"));
  G.patches = read_f32_(G.dir / (G.tag + "_patches.f32"));
  G.cap = read_f32_(G.dir / (G.tag + "_cap.f32"));
  G.out = read_f32_(G.dir / (G.tag + "_out.f32"));
  std::string err;
  if (!zimage::build_layout(G.lh / G.P, G.lw / G.P, G.cap_len, &G.lay,
                            &err)) {
    return G;
  }
  G.ok = !G.latent.empty() && !G.patches.empty() && !G.cap.empty() &&
         !G.out.empty();
  return G;
}

metal_compute::SharedBuffer
to_bf16_(MetalCompute* mc, const std::vector<float>& v)
{
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
}

// Run the golden's forward. Returns the UNPATCHIFIED [C, lh, lw]
// velocity, which is the shape the reference writes -- so this also
// exercises unpack_latents.
std::vector<float>
run_golden_(MetalZImageTransformer* m, MetalCompute* mc, const Golden& G,
            std::string* err)
{
  std::vector<float> out;
  metal_compute::SharedBuffer lat = to_bf16_(mc, G.patches);
  metal_compute::SharedBuffer cap = to_bf16_(mc, G.cap);
  if (lat.empty() || cap.empty()) { return out; }
  MetalZImageTransformer::Request req;
  req.latents = &lat;
  req.cap = &cap;
  req.layout = &G.lay;
  req.timestep = G.sigma;
  metal_compute::SharedBuffer v = m->forward(req, err);
  if (v.empty()) { return out; }
  const int PD = G.P * G.P * G.C;
  const std::size_t rows = (std::size_t)G.lay.img_tokens;
  if (v.byte_size() < rows * (std::size_t)PD * 2) { return out; }
  std::vector<float> flat(rows * (std::size_t)PD);
  const auto* d = static_cast<const std::uint16_t*>(v.contents());
  for (std::size_t i = 0; i < flat.size(); ++i) { flat[i] = bf16_(d[i]); }
  out.assign((std::size_t)G.C * G.lh * G.lw, 0.0f);
  zimage::unpack_latents(flat.data(), G.C, G.lh, G.lw, G.P, out.data());
  return out;
}

}  // namespace

TEST(z_image_dit, read_dims_reads_the_generated_config)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));
  EXPECT_TRUE(cfg.n_layers == G.layers);
  EXPECT_TRUE(cfg.n_refiner == G.refiner);
  EXPECT_TRUE(cfg.n_heads == G.heads);
  EXPECT_TRUE(cfg.hidden == G.dim);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.cap_dim == G.cap_dim);
  EXPECT_TRUE(cfg.patch == G.P);
  // The SwiGLU width is not in config.json at all: the reference
  // computes int(dim / 3 * 8), and a checkpoint whose feed_forward
  // disagrees has to fail at load rather than at the first GEMM.
  EXPECT_TRUE(cfg.ffn_inner == (int)((double)G.dim / 3.0 * 8.0));
  // The reference's own assert.
  EXPECT_TRUE(cfg.axes[0] + cfg.axes[1] + cfg.axes[2] == cfg.head_dim);
  EXPECT_TRUE(cfg.rope_theta == 256.0f);   // NOT 10000
  EXPECT_TRUE(cfg.norm_eps == 1e-5f);      // NOT 1e-6
}

// The configs this forward must REFUSE rather than run wrongly.
TEST(z_image_dit, read_dims_refuses_what_it_cannot_run)
{
  namespace fs = std::filesystem;
  const fs::path d = fs::temp_directory_path() / "vpipe_zimage_cfg_test";
  std::error_code ec;
  fs::create_directories(d, ec);
  auto put = [&](const char* json) {
    std::ofstream o(d / "config.json");
    o << json;
  };
  MetalZImageTransformer::Config cfg;
  // A different class entirely.
  put(R"({"_class_name":"BooguImageTransformer2DModel","dim":3840})");
  EXPECT_FALSE(MetalZImageTransformer::Config::read_dims(d.string(), &cfg));
  // GQA, which this forward does not implement.
  put(R"({"_class_name":"ZImageTransformer2DModel","dim":3840,
          "n_heads":30,"n_kv_heads":6})");
  EXPECT_FALSE(MetalZImageTransformer::Config::read_dims(d.string(), &cfg));
  // The unreleased Omni variant's SigLIP tower, which also changes the
  // token order and adds a per-token modulation.
  put(R"({"_class_name":"ZImageTransformer2DModel","dim":3840,
          "n_heads":30,"siglip_feat_dim":1152})");
  EXPECT_FALSE(MetalZImageTransformer::Config::read_dims(d.string(), &cfg));
  // More than one patch embedder: the tensor names pick one and this
  // would silently take the first.
  put(R"({"_class_name":"ZImageTransformer2DModel","dim":3840,
          "n_heads":30,"all_patch_size":[2,4]})");
  EXPECT_FALSE(MetalZImageTransformer::Config::read_dims(d.string(), &cfg));
  // Rotary axes that do not sum to the head dim.
  put(R"({"_class_name":"ZImageTransformer2DModel","dim":3840,
          "n_heads":30,"axes_dims":[32,32,32]})");
  EXPECT_FALSE(MetalZImageTransformer::Config::read_dims(d.string(), &cfg));
  fs::remove_all(d, ec);
}

TEST(z_image_dit, matches_reference_on_a_small_model)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(m != nullptr);

  // ENGAGEMENT BEFORE ACCURACY. A golden small enough to miss the fast
  // path verifies the arm nobody takes: gemm_mma_ declines under 64
  // rows, and the refiner stack runs at img_total while the main stack
  // runs at joint_len, so BOTH have to clear it.
  if (mc->supports_matrix_cores()) {
    EXPECT_TRUE(m->uses_mma2());
    EXPECT_TRUE(G.lay.img_total >= 64);
    EXPECT_TRUE(G.lay.joint_len >= 64);
  }

  std::string ferr;
  const std::vector<float> got = run_golden_(m.get(), mc, G, &ferr);
  if (got.empty()) {
    std::printf("[z_image_dit] forward failed: %s\n", ferr.c_str());
  }
  ZI_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, G.out);
  std::printf("[z_image_dit] velocity rel-L2 = %.6f over %zu (%d+%d+%d "
              "blocks, joint %d)\n", rel, G.out.size(), G.refiner,
              G.refiner, G.layers, G.lay.joint_len);
  // bf16 weights and activations against an fp32 reference: the floor
  // is ~1e-2, not 1e-3.
  EXPECT_TRUE(rel < 0.05);
}

// A STREAMED stack must be bit-identical to a preloaded one. Every one
// of these ports rebuilds something after the read, and a wrong
// permutation is silent.
TEST(z_image_dit, streamed_matches_preloaded)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));

  auto pre = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(pre != nullptr);
  std::string e1;
  const std::vector<float> a = run_golden_(pre.get(), mc, G, &e1);
  pre.reset();

  auto str = MetalZImageTransformer::load(ckpt, mc, cfg, true);
  ZI_REQUIRE(str != nullptr);
  EXPECT_TRUE(str->streaming());
  std::string e2;
  const std::vector<float> b = run_golden_(str.get(), mc, G, &e2);
  ZI_REQUIRE(!a.empty() && !b.empty() && a.size() == b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
  }
  std::printf("[z_image_dit] streamed vs preloaded: worst |diff| = %g\n",
              worst);
  EXPECT_TRUE(worst == 0.0);
}

// The PREFETCH-WITHOUT-A-FREE-DESTINATION bug surfaces only with the
// slot pair off, and as a GPU fault rather than a wrong answer. Every
// streaming port runs this A/B.
TEST(z_image_dit, streamed_without_slots_is_still_right)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));
  auto pre = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(pre != nullptr);
  std::string e1;
  const std::vector<float> a = run_golden_(pre.get(), mc, G, &e1);
  pre.reset();

  setenv("VPIPE_Z_IMAGE_NO_SLOTS", "1", 1);
  auto str = MetalZImageTransformer::load(ckpt, mc, cfg, true);
  std::vector<float> b;
  if (str != nullptr) {
    std::string e2;
    b = run_golden_(str.get(), mc, G, &e2);
    str.reset();
  }
  unsetenv("VPIPE_Z_IMAGE_NO_SLOTS");
  ZI_REQUIRE(!a.empty() && !b.empty() && a.size() == b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
  }
  EXPECT_TRUE(worst == 0.0);
}

// On a box WITHOUT matrix cores every GEMM takes the dense tiled path,
// and "not scalar" is not the same as "the right tile". Emulated by
// turning BOTH knobs off: setting only NO_MMA2 would leave the NAX
// attention on, which is a machine that does not exist.
TEST(z_image_dit, no_matrix_cores_stays_on_wide_tiles)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));

  auto ref = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(ref != nullptr);
  std::string e1;
  const std::vector<float> want = run_golden_(ref.get(), mc, G, &e1);
  ref.reset();

  setenv("VPIPE_Z_IMAGE_NO_MMA2", "1", 1);
  setenv("VPIPE_Z_IMAGE_NO_NAX_ATTN", "1", 1);
  auto m4 = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  std::vector<float> got;
  bool wide = false, no_mma = false, no_nax = false;
  if (m4 != nullptr) {
    no_mma = !m4->uses_mma2();
    no_nax = !m4->uses_attn_nax();
    wide = m4->gemm_tile() == 2;
    std::string e2;
    got = run_golden_(m4.get(), mc, G, &e2);
    m4.reset();
  }
  unsetenv("VPIPE_Z_IMAGE_NO_MMA2");
  unsetenv("VPIPE_Z_IMAGE_NO_NAX_ATTN");
  EXPECT_TRUE(no_mma);
  EXPECT_TRUE(no_nax);
  // The WIDEST tile that loaded -- the point of the test. A kernel can
  // be tiled and still be the weakest one available.
  EXPECT_TRUE(wide);
  ZI_REQUIRE(!want.empty() && !got.empty());
  const double rel = rel_l2_(got, want);
  std::printf("[z_image_dit] no-matrix-cores rel-L2 vs mma = %.6f\n", rel);
  EXPECT_TRUE(rel < 0.02);
}

// ---- the real checkpoint --------------------------------------------

TEST(z_image_dit, reads_the_published_config)
{
  const char* r = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH");
  if (r == nullptr || *r == '\0') { return; }
  const std::string dir =
      (std::filesystem::path(r) / "transformer").string();
  MetalZImageTransformer::Config cfg;
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(dir, &cfg));
  EXPECT_TRUE(cfg.hidden == 3840);
  EXPECT_TRUE(cfg.n_heads == 30);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.n_layers == 30);
  EXPECT_TRUE(cfg.n_refiner == 2);
  EXPECT_TRUE(cfg.n_blocks() == 34);
  EXPECT_TRUE(cfg.n_mod_blocks() == 32);
  EXPECT_TRUE(cfg.in_channels == 16);
  EXPECT_TRUE(cfg.patch == 2);
  EXPECT_TRUE(cfg.patch_dim() == 64);
  EXPECT_TRUE(cfg.ffn_inner == 10240);
  EXPECT_TRUE(cfg.cap_dim == 2560);
  EXPECT_TRUE(cfg.rope_theta == 256.0f);
  EXPECT_TRUE(cfg.axes[0] == 32 && cfg.axes[1] == 48 && cfg.axes[2] == 48);
}

// Every tensor this forward reads, by NAME, off the real checkpoint.
// Cheap even on a small box: streaming mode reads only the model-level
// weights rather than the stack.
TEST(z_image_dit, finds_every_tensor_in_the_published_checkpoint)
{
  const char* r = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH");
  if (r == nullptr || *r == '\0') { return; }
  const std::string dir =
      (std::filesystem::path(r) / "transformer").string();
  MetalZImageTransformer::Config cfg;
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(dir, &cfg));
  auto ws = WeightSet::open(dir, nullptr);
  ZI_REQUIRE(ws != nullptr);
  const MetalLlamaWeights& w = ws->src();
  auto has = [&](const std::string& nm) {
    const bool ok = w.info(nm) != nullptr;
    if (!ok) { std::printf("[z_image_dit] MISSING %s\n", nm.c_str()); }
    return ok;
  };
  // The patch key is part of the tensor name (a ModuleDict), not a
  // config lookup.
  const std::string pk = std::to_string(cfg.patch) + "-" +
                         std::to_string(cfg.f_patch);
  EXPECT_TRUE(has("all_x_embedder." + pk + ".weight"));
  EXPECT_TRUE(has("all_x_embedder." + pk + ".bias"));
  EXPECT_TRUE(has("cap_embedder.0.weight"));
  EXPECT_TRUE(has("cap_embedder.1.weight"));
  EXPECT_TRUE(has("cap_embedder.1.bias"));
  EXPECT_TRUE(has("t_embedder.mlp.0.weight"));
  EXPECT_TRUE(has("t_embedder.mlp.2.weight"));
  EXPECT_TRUE(has("x_pad_token"));
  EXPECT_TRUE(has("cap_pad_token"));
  // The final layer's adaLN is `.1` -- there IS a SiLU in front of it.
  EXPECT_TRUE(has("all_final_layer." + pk + ".adaLN_modulation.1.weight"));
  EXPECT_TRUE(has("all_final_layer." + pk + ".linear.weight"));
  for (int i = 0; i < cfg.n_refiner; ++i) {
    const std::string c = "context_refiner." + std::to_string(i) + ".";
    const std::string n = "noise_refiner." + std::to_string(i) + ".";
    EXPECT_TRUE(has(c + "attention.to_q.weight"));
    EXPECT_TRUE(has(c + "attention_norm2.weight"));
    EXPECT_TRUE(has(c + "ffn_norm2.weight"));
    EXPECT_TRUE(has(c + "feed_forward.w1.weight"));
    // THE CONTEXT REFINER HAS NO MODULATION and the noise refiner does.
    // That asymmetry is why the streaming stack is the modulated blocks
    // alone -- a slot has to be refillable by every index that uses it.
    EXPECT_FALSE(w.info(c + "adaLN_modulation.0.weight") != nullptr);
    EXPECT_TRUE(has(n + "adaLN_modulation.0.weight"));
    EXPECT_TRUE(has(n + "adaLN_modulation.0.bias"));
  }
  for (int i = 0; i < cfg.n_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    EXPECT_TRUE(has(p + "attention.to_q.weight"));
    EXPECT_TRUE(has(p + "attention.to_out.0.weight"));
    EXPECT_TRUE(has(p + "attention.norm_q.weight"));
    EXPECT_TRUE(has(p + "feed_forward.w1.weight"));
    EXPECT_TRUE(has(p + "feed_forward.w2.weight"));
    EXPECT_TRUE(has(p + "feed_forward.w3.weight"));
    EXPECT_TRUE(has(p + "attention_norm1.weight"));
    EXPECT_TRUE(has(p + "ffn_norm1.weight"));
    // A block's adaLN is `.0` -- a BARE Linear.
    EXPECT_TRUE(has(p + "adaLN_modulation.0.weight"));
  }
}

// ---- quantize -------------------------------------------------------
//
// What this pins is the LEAF SET and the exclude list, which are the
// two halves of one decision and both silent when wrong: a leaf that
// matches nothing quantizes nothing, and a leaf that matches too much
// takes a precision-sensitive head with it. Run against the small
// golden model, so it costs a second rather than a 24 GB pass.
TEST(z_image_dit, quantize_takes_the_blocks_and_leaves_the_heads)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  const std::string src = (G.dir / G.tag).string();
  std::error_code ec;
  if (!std::filesystem::exists(src + "/config.json", ec)) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::filesystem::path out =
      std::filesystem::temp_directory_path() /
      ("vpipe-zimage-q-" + std::to_string(::getpid()));
  std::filesystem::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(8));
    o.insert("skip_existing", FlexData::make_bool(false));
  }
  vpipe::ModelQuantizeStage st(&sess, "qzi", std::vector<InEdge>{},
                               std::move(cfg));
  ZI_REQUIRE(st.config_error().empty());
  if (!st.quantize_once()) {
    std::printf("[z_image_dit] quantize did not run\n");
    std::filesystem::remove_all(out, ec);
  }
  ZI_REQUIRE(std::filesystem::exists(out / "config.json", ec));

  auto wts = vpipe::genai::MetalLlamaWeights::open_model(out.string());
  ZI_REQUIRE(wts.has_value());
  int blk_q = 0, head_q = 0;
  bool saw_attn = false, saw_ffn = false;
  for (const auto& n : wts->tensor_names()) {
    // The codes keep the `.weight` name and the scales hang off the
    // base WITHOUT it, so the sibling of `X.weight` is `X.scales`.
    if (n.size() <= 7 || n.compare(n.size() - 7, 7, ".scales") != 0) {
      continue;
    }
    const std::string base = n.substr(0, n.size() - 7);
    const bool in_block = base.rfind("layers.", 0) == 0 ||
                          base.rfind("noise_refiner.", 0) == 0 ||
                          base.rfind("context_refiner.", 0) == 0;
    if (in_block) {
      ++blk_q;
      if (base.find("attention.to_") != std::string::npos) { saw_attn = true; }
      if (base.find("feed_forward.w") != std::string::npos) { saw_ffn = true; }
    } else {
      ++head_q;
      std::printf("[z_image_dit] UNEXPECTED quantized head: %s\n",
                  base.c_str());
    }
  }
  std::printf("[z_image_dit] quantized %d block tensors, %d model-level\n",
              blk_q, head_q);
  // ENGAGEMENT: a leaf set that matches nothing is the failure this
  // exists to catch, and it passes every other bar.
  EXPECT_TRUE(blk_q > 0);
  EXPECT_TRUE(saw_attn);
  EXPECT_TRUE(saw_ffn);
  // NOTHING model-level. `t_embedder.mlp.0` is the one at real risk:
  // its leaf is "0", the same leaf attention.to_out.0 carries, and it
  // is a 2-D fp matrix whose K divides the group -- so only the
  // exclude list keeps it dense.
  EXPECT_TRUE(head_q == 0);
  std::filesystem::remove_all(out, ec);
}

// ---- the declared streaming floor -----------------------------------
//
// A DECLARED FLOOR IS A PROMISE, and the three ways this one goes wrong
// are all silent. It read 0 -- "cannot be reduced" -- until this family
// answered for itself, because the shared stem list has no Z-Image
// spelling; that declared a 24.6 GB DiT at full size and put the plan's
// peak 8 GB over a 16 GB box. Booking a slot pair per PREFIX instead of
// per stack over-states by two blocks. And forgetting that an F32
// checkpoint lands as bf16 over-states by exactly 2x.
//
// Asserted as a SHAPE against the checkpoint's own tensor sizes rather
// than a constant, so it survives a re-published checkpoint.
TEST(z_image_dit, streaming_floor_is_the_trunk_plus_one_slot_pair)
{
  const char* r = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH");
  if (r == nullptr || *r == '\0') { return; }
  const std::string dir =
      (std::filesystem::path(r) / "transformer").string();
  auto wts = vpipe::genai::MetalLlamaWeights::open_model(dir);
  ZI_REQUIRE(wts.has_value());

  std::size_t total = 0, widest = 0, ctx = 0;
  bool f32 = false;
  std::map<std::string, std::size_t> blocks;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    const std::size_t nb = (std::size_t)ti->nbytes;
    total += nb;
    if (ti->dtype == "F32") { f32 = true; }
    if (nm.rfind("context_refiner.", 0) == 0) { ctx += nb; continue; }
    const char* pre = nm.rfind("noise_refiner.", 0) == 0 ? "noise_refiner."
                      : nm.rfind("layers.", 0) == 0     ? "layers."
                                                        : nullptr;
    if (pre == nullptr) { continue; }
    const std::size_t i0 = std::strlen(pre);
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { continue; }
    blocks[std::string(pre) + nm.substr(i0, dot - i0)] += nb;
  }
  for (const auto& b : blocks) { widest = std::max(widest, b.second); }
  ZI_REQUIRE(widest > 0 && total > 0);
  // The stack is ONE stack: 2 noise-refiner blocks plus every main
  // layer, under two prefixes.
  EXPECT_TRUE(blocks.size() == 32);

  const std::size_t got =
      MetalZImageTransformer::streaming_floor_bytes(dir);
  // The unit a floor is built from, in the bytes the model HOLDS.
  const std::size_t held_block = f32 ? widest / 2 : widest;
  std::printf("[z_image_dit] floor %zu MB of %zu MB total (%zu blocks, "
              "held block %zu MB, ctx %zu MB, %s)\n", got >> 20,
              total >> 20, blocks.size(), held_block >> 20, ctx >> 20,
              f32 ? "F32 -> bf16" : "bf16");

  // 1. A STEM MATCHED. Zero here is the failure that started this.
  EXPECT_TRUE(got > 0);
  // 2. THE F32 NARROWING IS COUNTED: what is held cannot exceed half
  //    the on-disk bytes when the checkpoint is F32.
  if (f32) { EXPECT_TRUE(got < total / 2); }
  // 3. ONE slot pair, not two. The floor is the trunk -- which is the
  //    two HELD context-refiner blocks plus the small model-level
  //    weights -- plus two slots, so four blocks' worth and change.
  //    Five would mean a pair per prefix; three would mean the context
  //    refiner was treated as streamable when the model holds it.
  EXPECT_TRUE(got > (std::size_t)3 * held_block);
  EXPECT_TRUE(got < (std::size_t)5 * held_block);
}

// ---- the accelerated tiers ------------------------------------------
//
// ENGAGEMENT BEFORE ACCURACY. An unwired tier costs nothing, does
// nothing, and passes every numeric bar -- so each of these asserts the
// tier turned ITSELF on at a geometry that clears its floor before it
// asks whether the answer is still right. The "wide" golden exists for
// exactly that: i8_gemm declines under M = 1024 and Sol under 1024 rows
// or 16 key blocks, and the small golden's joint sequence is 160.
namespace {

// Run both arms of an accel A/B over the wide golden and report the
// error against the SAME model with the tier off -- not against the
// fp32 reference, because what is being measured is what the tier
// changes, and the bf16 baseline's own 6e-3 would swamp it.
struct AccelAb {
  bool ran = false;
  bool engaged = false;
  double rel_vs_off = 0.0;
  double rel_vs_ref = 0.0;
};

}  // namespace

TEST(z_image_dit, i8_gemm_engages_and_stays_accurate)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));
  // THE FLOORS THE TIER DECLINES UNDER, stated so a golden that stops
  // clearing one fails here rather than passing vacuously. i8 wants
  // M >= its min (1024 by default) and, less obviously, K >= 1024 --
  // which is a statement about the model's WIDTH, not its sequence, so
  // the first wide golden (hidden 256) could never have engaged it
  // however long its sequence was.
  EXPECT_TRUE(G.lay.joint_len >= 1024);
  EXPECT_TRUE(cfg.hidden >= 1024);

  auto base = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(base != nullptr);
  std::string e1;
  const std::vector<float> off = run_golden_(base.get(), mc, G, &e1);
  base.reset();

  cfg.i8_gemm = true;
  auto m = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(m != nullptr);
  const bool engaged = m->uses_i8_gemm();
  std::string e2;
  const std::vector<float> on = run_golden_(m.get(), mc, G, &e2);
  m.reset();
  if (!engaged) {
    std::printf("[z_image_dit] i8_gemm unavailable on this box; skipped\n");
    return;
  }
  ZI_REQUIRE(!off.empty() && !on.empty() && off.size() == on.size());
  const double d = rel_l2_(on, off);
  const double r = rel_l2_(on, G.out);
  std::printf("[z_image_dit] i8: rel-L2 vs bf16 %.6f, vs reference %.6f "
              "(bf16 alone %.6f)\n", d, r, rel_l2_(off, G.out));
  // It has regressed to NaN on a bf16 constructor before, which is why
  // the bar is two-sided rather than just "small".
  EXPECT_TRUE(std::isfinite(d) && std::isfinite(r));
  EXPECT_TRUE(d > 0.0);          // it actually changed the arithmetic
  EXPECT_TRUE(r < 0.12);         // and did not wreck it
}

TEST(z_image_dit, sol_attention_engages_and_stays_close)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));
  // Sol's own floors: 1024 query rows and 16 key blocks. EVERY
  // attention in this model is square and non-causal, so both halves of
  // the stack qualify at the same length -- unlike the block-causal
  // families, which have text runs Sol must decline.
  EXPECT_TRUE(G.lay.img_total >= 1024 && G.lay.joint_len >= 1024);

  auto base = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(base != nullptr);
  std::string e1;
  const std::vector<float> off = run_golden_(base.get(), mc, G, &e1);
  base.reset();

  cfg.sol.enabled = true;
  auto m = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(m != nullptr);
  const bool engaged = m->uses_sol_attn();
  std::string e2;
  const std::vector<float> on = run_golden_(m.get(), mc, G, &e2);
  m.reset();
  if (!engaged) {
    std::printf("[z_image_dit] sol_attn unavailable here; skipped\n");
    return;
  }
  ZI_REQUIRE(!off.empty() && !on.empty() && off.size() == on.size());
  const double d = rel_l2_(on, off);
  std::printf("[z_image_dit] sol (tau %.2f): rel-L2 vs dense %.6f\n",
              (double)cfg.sol.tau, d);
  // APPROXIMATE by construction -- it skips key blocks -- so this is a
  // sanity bound, not an equality. Held to a picture, not a norm.
  EXPECT_TRUE(std::isfinite(d));
  EXPECT_TRUE(d > 0.0);
  EXPECT_TRUE(d < 0.6);
}

// ---- runtime LoRA ----------------------------------------------------
//
// Checked against the reference's OWN dense arithmetic: the dumper
// writes a synthetic adapter AND the same model with those factors
// fused into the base weights, then runs the reference on the fused
// one. So this compares `base + runtime adapter` against `(base +
// B@A) run densely in fp32` -- not against a re-derivation of the same
// idea in this file.
//
// What it does NOT establish is that a PUBLISHED Z-Image adapter will
// bind: no such file exists yet, so the module names here are a
// reading of the diffusers-export convention (the base tensor's path
// without ".weight"). When a real one binds 0 modules the fix is a
// rename row in shared/lora-names.h, not this list.
TEST(z_image_dit, runtime_lora_matches_a_fused_reference)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  const std::filesystem::path lp =
      G.dir / (G.tag + "_lora.safetensors");
  const std::vector<float> want =
      read_f32_(G.dir / (G.tag + "_lora_out.f32"));
  std::error_code ec;
  if (!std::filesystem::exists(lp, ec) || want.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));

  // The UNADAPTED run in this model's own precision. Comparing the
  // scale-0 arm against the fp32 reference instead would measure the
  // bf16 floor (~7e-3) and say nothing about the adapter.
  auto plain = MetalZImageTransformer::load(ckpt, mc, cfg, false);
  ZI_REQUIRE(plain != nullptr);
  std::string eb;
  const std::vector<float> base = run_golden_(plain.get(), mc, G, &eb);
  plain.reset();
  ZI_REQUIRE(!base.empty());

  std::vector<MetalZImageTransformer::LoraSpec> specs;
  specs.push_back({lp.string(), 1.0f});
  auto m = MetalZImageTransformer::load(ckpt, mc, cfg, false, specs);
  ZI_REQUIRE(m != nullptr);
  // ENGAGEMENT. An adapter for another model binds nothing and
  // otherwise reports success by saying nothing at all, so this is the
  // assertion that has to come first.
  const int bound = m->lora_modules(0);
  std::printf("[z_image_dit] lora bound %d modules\n", bound);
  ZI_REQUIRE(bound > 0);

  std::string ferr;
  const std::vector<float> got = run_golden_(m.get(), mc, G, &ferr);
  ZI_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, want);
  // And the adapter must actually MOVE the model: equal to the fused
  // reference is only meaningful if the fused reference differs from
  // the base one.
  const double moved = rel_l2_(base, want);
  std::printf("[z_image_dit] lora: rel-L2 vs fused reference %.6f "
              "(the adapter moves the base by %.6f)\n", rel, moved);
  EXPECT_TRUE(moved > 0.01);
  EXPECT_TRUE(rel < 0.05);

  // THE STRENGTH IS LIVE and exactly 0 encodes nothing, so it must
  // reproduce the UNADAPTED model rather than something that merely
  // rounds to it.
  m->set_lora_scale(0, 0.0f);
  std::string e0;
  const std::vector<float> off = run_golden_(m.get(), mc, G, &e0);
  ZI_REQUIRE(!off.empty());
  const double rel_off = rel_l2_(off, base);
  std::printf("[z_image_dit] lora at scale 0: rel-L2 vs base %.6f\n",
              rel_off);
  EXPECT_TRUE(rel_off < 1e-6);
}

// ---- the cooperative stop -------------------------------------------
//
// A stop must land within about one BLOCK, not one forward, and it must
// come back as a stop rather than a failure: empty with NO reason, which
// is the only thing telling the stage whether to log "stopped" or "DiT
// forward failed".
//
// BOTH ARMS ARE MEASURED because they commit differently and only one
// of them is bounded by a block. A streamed block commits and WAITS, so
// its stop sits out whatever is on the GPU -- one block. A preloaded run
// encodes every block into one stream and commits ONCE at the end, so
// the block loop races through in milliseconds and the GPU work all
// happens inside a single wait no check can interrupt. The bound there
// is one FORWARD, and the ratio below is what says so out loud rather
// than leaving it to be discovered.
namespace {

struct StopRun {
  double whole_ms = 0.0;
  double latency_ms = 0.0;
  bool empty_result = false;
  bool no_reason = false;
  bool tripped = false;
};

StopRun
measure_stop_(MetalZImageTransformer* m, MetalCompute* mc, const Golden& G,
              int trip_at)
{
  StopRun r;
  std::string e0;
  const auto w0 = std::chrono::steady_clock::now();
  const std::vector<float> full = run_golden_(m, mc, G, &e0);
  r.whole_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - w0).count();
  if (full.empty()) { return r; }

  // The flag flips on the Nth CHECK rather than on a timer, so what is
  // measured is "how long after the stop became visible did forward
  // return" with no sleep to confuse it.
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
  // DEFAULT-CONSTRUCTED, exactly as the stage declares it: a stop must
  // not write a reason into it, and "stayed empty" is the assertion.
  std::string serr;
  metal_compute::SharedBuffer lat = to_bf16_(mc, G.patches);
  metal_compute::SharedBuffer cap = to_bf16_(mc, G.cap);
  MetalZImageTransformer::Request req;
  req.latents = &lat;
  req.cap = &cap;
  req.layout = &G.lay;
  req.timestep = G.sigma;
  metal_compute::SharedBuffer out = m->forward(req, &serr);
  const auto s1 = std::chrono::steady_clock::now();
  m->set_stream_stop({});

  r.tripped = tripped.time_since_epoch().count() != 0;
  if (r.tripped) {
    r.latency_ms =
        std::chrono::duration<double, std::milli>(s1 - tripped).count();
  }
  r.empty_result = out.empty();
  r.no_reason = serr.empty();
  return r;
}

}  // namespace

TEST(z_image_dit, a_stop_is_not_a_failure_and_lands_inside_a_forward)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(ckpt, &cfg));
  // 34 blocks of checks: the two context refiners, then the modulated
  // stack. Tripping midway leaves most of the stack unrun.
  const int total = cfg.n_blocks();
  const int trip_at = total / 2;

  for (int arm = 0; arm < 2; ++arm) {
    const bool stream = arm == 1;
    auto m = MetalZImageTransformer::load(ckpt, mc, cfg, stream);
    ZI_REQUIRE(m != nullptr);
    if (stream) {
      // Residency off, so every block really streams -- a promoted
      // block is on the preloaded path and would not measure this arm.
      m->set_residency_reserve(0);
      m->set_residency_schedule(2);
    }
    const StopRun r = measure_stop_(m.get(), mc, G, trip_at);
    m.reset();
    ZI_REQUIRE(r.tripped);
    std::printf("[z_image_dit] stop (%s) at check %d/%d: returned %.1f ms "
                "after the flag; a whole forward is %.1f ms (%.0f%% of "
                "one)\n", stream ? "streamed" : "preloaded", trip_at,
                total, r.latency_ms, r.whole_ms,
                r.whole_ms > 0.0 ? 100.0 * r.latency_ms / r.whole_ms : 0.0);
    // A STOP IS NOT A FAILURE. This is what the stage reads to decide
    // between "stopped" and "DiT forward failed", so it matters more
    // than the timing.
    EXPECT_TRUE(r.empty_result);
    EXPECT_TRUE(r.no_reason);
    // And it must not have had to run the rest of the stack. Half the
    // blocks remained, so anything near a whole forward means the
    // check is not where it needs to be.
    EXPECT_TRUE(r.latency_ms < r.whole_ms);
  }
}

// The same question against the PUBLISHED checkpoint, where a block is
// real work. Streaming, which is what any 16 GB box does with this
// family and the arm whose latency is bounded by a block: a streamed
// block commits and waits, so a stop has to sit out only what is
// already on the GPU.
TEST(z_image_dit, stop_on_the_real_checkpoint_lands_within_a_block)
{
  const char* r = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH");
  if (r == nullptr || *r == '\0') { return; }
  const std::string dir =
      (std::filesystem::path(r) / "transformer").string();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(dir, &cfg));
  auto m = MetalZImageTransformer::load(dir, mc, cfg, /*stream=*/true);
  ZI_REQUIRE(m != nullptr);
  // Residency OFF so every block really streams; a promoted block is
  // on the preloaded path and would not measure this arm.
  m->set_residency_reserve(0);
  m->set_residency_schedule(2);

  // 512x512: a 32x32 patch grid, 1024 image rows -- enough that a
  // block is real work and a whole forward is clearly longer than the
  // bound below.
  zimage::Layout lay;
  std::string lerr;
  ZI_REQUIRE(zimage::build_layout(32, 32, 64, &lay, &lerr));
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
      rnd((std::size_t)lay.img_tokens * cfg.patch_dim(), 11u);
  metal_compute::SharedBuffer cap =
      rnd((std::size_t)lay.cap_len * cfg.cap_dim, 12u);
  ZI_REQUIRE(!lat.empty() && !cap.empty());
  MetalZImageTransformer::Request req;
  req.latents = &lat;
  req.cap = &cap;
  req.layout = &lay;
  req.timestep = 0.75f;

  std::string ferr;
  const auto w0 = std::chrono::steady_clock::now();
  metal_compute::SharedBuffer full = m->forward(req, &ferr);
  const double whole_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - w0).count();
  ZI_REQUIRE(!full.empty());
  m->release_resident_blocks((std::size_t)-1);

  const int total = cfg.n_blocks();
  const int trip_at = total / 2;
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
  std::string serr;
  metal_compute::SharedBuffer stopped = m->forward(req, &serr);
  const auto s1 = std::chrono::steady_clock::now();
  m->set_stream_stop({});
  ZI_REQUIRE(tripped.time_since_epoch().count() != 0);
  const double latency_ms =
      std::chrono::duration<double, std::milli>(s1 - tripped).count();
  const double per_block = whole_ms / (double)total;
  std::printf("[z_image_dit] REAL stop at block %d/%d: %.1f ms after the "
              "flag; a whole forward is %.0f ms (%.0f ms/block), so %.1f "
              "blocks\n", trip_at, total, latency_ms, whole_ms, per_block,
              per_block > 0.0 ? latency_ms / per_block : 0.0);
  EXPECT_TRUE(stopped.empty());
  EXPECT_TRUE(serr.empty());
  // WITHIN ABOUT ONE BLOCK. Generous by 3x because the block a stop
  // lands in may be one whose read was not prefetched, but nowhere
  // near the half-stack that remained.
  EXPECT_TRUE(latency_ms < 3.0 * per_block);

  // ---- and the arm a REAL run is mostly on --------------------------
  //
  // `streaming` is per BLOCK: `_stream_blocks && !held`, so a block
  // residency has PROMOTED takes the preloaded path -- no per-block
  // commit. A warm run on this box promotes most of the stack, so most
  // of a real forward is encoded into one stream and committed once,
  // and the block loop races through it in milliseconds while every
  // bit of GPU work happens inside a single wait no check can
  // interrupt. Measured rather than assumed, because the difference
  // between the two arms is the difference between bounding a stop by
  // a block and bounding it by a forward.
  {
    std::string we;
    metal_compute::SharedBuffer warm = m->forward(req, &we);   // promote
    (void)warm;
    const int held = m->resident_block_count();
    int seen2 = 0;
    std::chrono::steady_clock::time_point trip2{};
    m->set_stream_stop([&]() {
      ++seen2;
      if (seen2 < trip_at) { return false; }
      if (trip2.time_since_epoch().count() == 0) {
        trip2 = std::chrono::steady_clock::now();
      }
      return true;
    });
    std::string se2;
    metal_compute::SharedBuffer st2 = m->forward(req, &se2);
    const auto t1 = std::chrono::steady_clock::now();
    m->set_stream_stop({});
    if (trip2.time_since_epoch().count() != 0) {
      const double lat2 =
          std::chrono::duration<double, std::milli>(t1 - trip2).count();
      std::printf("[z_image_dit] REAL stop with %d/%d blocks PROMOTED: "
                  "%.1f ms after the flag (%.1f blocks at the cold "
                  "rate)\n", held, cfg.n_mod_blocks(), lat2,
                  per_block > 0.0 ? lat2 / per_block : 0.0);
    }
    EXPECT_TRUE(st2.empty());
    EXPECT_TRUE(se2.empty());
  }

  // ---- the case a block check CANNOT see ----------------------------
  //
  // Both measurements above trip on the Nth CHECK, so the flag is
  // always already set when a check runs and the answer is always
  // "immediately". A REAL stop arrives at an arbitrary wall-clock
  // moment, and nearly all of a forward's wall time is spent inside a
  // commit-and-wait that no check interrupts. So the question worth
  // asking is what happens when the request lands THERE, and the only
  // way to ask it is a flag set from outside on a timer.
  {
    std::atomic<bool> flag{false};
    std::atomic<long long> at_ns{0};
    // EARLY, at a sixth of the COLD forward. The forwards above have
    // warmed residency so this one is faster, and firing at half would
    // sometimes land past the LAST block's check -- after which there
    // is nothing left to stop and the forward correctly runs to its
    // end. That is real behaviour, not a bug, but it makes a flaky
    // assertion; firing early leaves most of the stack ahead of the
    // flag and the answer deterministic.
    const auto delay = std::chrono::milliseconds((long)(whole_ms / 6.0));
    std::thread t([&]() {
      std::this_thread::sleep_for(delay);
      at_ns.store(std::chrono::steady_clock::now()
                      .time_since_epoch().count());
      flag.store(true);
    });
    m->set_stream_stop([&]() { return flag.load(); });
    std::string se3;
    const auto f0 = std::chrono::steady_clock::now();
    metal_compute::SharedBuffer st3 = m->forward(req, &se3);
    const auto f1 = std::chrono::steady_clock::now();
    t.join();
    m->set_stream_stop({});
    const auto asked =
        std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(at_ns.load()));
    const double lat3 =
        std::chrono::duration<double, std::milli>(f1 - asked).count();
    const double ran =
        std::chrono::duration<double, std::milli>(f1 - f0).count();
    std::printf("[z_image_dit] REAL stop asked mid-forward (on a timer, "
                "not on a check): honoured %.0f ms later, %.0f ms into a "
                "forward that cold takes %.0f ms -- %s\n", lat3, ran,
                whole_ms, st3.empty() ? "STOPPED" : "ran to completion");
    // A stop asked with most of the stack still ahead of it must be
    // HONOURED, and must still not read as a failure.
    EXPECT_TRUE(st3.empty());
    EXPECT_TRUE(se3.empty());
    // And honoured within a couple of blocks of being asked, not at
    // the end of the forward.
    EXPECT_TRUE(lat3 < 4.0 * per_block);
  }
}

// THE ANE TIERS: that their rows are right, that a runtime adapter
// reaches them, and -- opt-in -- what they save.
//
// A SwiGLU feed-forward is row-independent, and so is a projection, so
// the ANE takes a contiguous band of a block's rows and the GPU the
// rest, concurrently, with no seam to blend: the feed-forward tier
// (ane_ffn) and the q/k/v tier (ane_qkv) beside it. The rows come back
// fp16 where the GPU's are bf16 (more mantissa, not less), so the bar is
// a rounding difference through the stack, not bit-identity.
//
// ENGAGEMENT FIRST. The modules' chunk is 2048 rows and a geometry that
// gives the ANE less than one declines the tier outright, which passes
// while testing nothing -- so 1024^2 (4096 image rows) is the floor, and
// the test asserts each tier ARMED before anything about the numbers.
//
// THE ADAPTER. The ANE is fed each block's weights as fp16 inputs, so a
// runtime LoRA has to be MERGED into what it stages; a tier that stages
// the bare weights runs the base model on its rows and the adapted one
// on the GPU's -- a seam that looks like the adapter at half strength.
// A synthetic adapter on q/k/v and the feed-forward of every main block,
// strong enough to move the output well past the tiers' own rounding,
// has to agree ANE-vs-GPU as closely as the base model does.
//
// SPEED is a property of the machine, not of the tier, so it is not
// asserted: VPIPE_Z_IMAGE_ANE_BENCH=<reps> holds every arm, warms them,
// and times their forwards with the order ROTATING per rep, so a clock
// that drifts charges every arm alike. It is meant for the M4 series (no
// matrix cores, so the GPU's GEMMs run the tiled path and the ANE has
// proportionally most to take); a fanless M5 shares one power budget
// between the engines and reads slower. VPIPE_Z_IMAGE_ANE_PX sets the
// square size (default 1024), VPIPE_Z_IMAGE_ANE_STREAM=0 holds the
// blocks (default: stream, which a 16 GB box must).
namespace {

// A bf16 safetensors file, for a synthetic adapter.
bool
write_bf16_st_(const std::filesystem::path& p,
               const std::vector<std::pair<std::string,
                                           std::vector<std::int64_t>>>& ts,
               const std::vector<std::vector<std::uint16_t>>& data)
{
  std::string hdr = "{";
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const std::uint64_t nb = data[i].size() * 2;
    hdr += (i ? "," : "") + std::string("\"") + ts[i].first +
           "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t d = 0; d < ts[i].second.size(); ++d) {
      hdr += (d ? "," : "") + std::to_string(ts[i].second[d]);
    }
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + nb) + "]}";
    off += nb;
  }
  hdr += "}";
  while ((hdr.size() + 8) % 16 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t n = hdr.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const auto& d : data) {
    f.write(reinterpret_cast<const char*>(d.data()),
            (std::streamsize)(d.size() * 2));
  }
  return (bool)f;
}

}  // namespace

TEST(z_image_dit, ane_tiers_rows_are_right)
{
  const char* r = std::getenv("VPIPE_Z_IMAGE_TEST_MODEL_PATH");
  if (r == nullptr || *r == '\0') { return; }
  const std::string dir =
      (std::filesystem::path(r) / "transformer").string();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalZImageTransformer::Config cfg;
  ZI_REQUIRE(MetalZImageTransformer::Config::read_dims(dir, &cfg));
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int px = envi("VPIPE_Z_IMAGE_ANE_PX", 1024);
  const bool stream = envi("VPIPE_Z_IMAGE_ANE_STREAM", 1) != 0;
  const int reps = envi("VPIPE_Z_IMAGE_ANE_BENCH", 0);
  // 8x VAE, 2x patch: a px^2 image is a (px/16)^2 patch grid.
  const int grid = px / 16;
  zimage::Layout lay;
  std::string lerr;
  ZI_REQUIRE(zimage::build_layout(grid, grid, 64, &lay, &lerr));
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
      rnd((std::size_t)lay.img_tokens * cfg.patch_dim(), 21u);
  metal_compute::SharedBuffer cap =
      rnd((std::size_t)lay.cap_len * cfg.cap_dim, 22u);
  ZI_REQUIRE(!lat.empty() && !cap.empty());
  MetalZImageTransformer::Request req;
  req.latents = &lat;
  req.cap = &cap;
  req.layout = &lay;
  req.timestep = 0.75f;

  enum Arm { kGpu, kFf, kQkv, kArms };
  const char* const kName[kArms] = {"GPU-only", "ANE ff", "ANE ff+qkv"};
  auto make = [&](int arm,
                  const std::vector<MetalZImageTransformer::LoraSpec>& lo) {
    MetalZImageTransformer::Config c = cfg;
    c.ane_ffn = arm != kGpu;
    c.ane_qkv = arm == kQkv;
    auto m = MetalZImageTransformer::load(dir, mc, c, stream, lo);
    // A streamed stack admits nothing until a reserve is DECLARED; 0 is
    // the answer that keeps every block on the read path, in every arm.
    if (m != nullptr && stream) {
      m->set_residency_reserve(0);
      m->set_residency_schedule(reps + 2);
    }
    return m;
  };
  const std::size_t n = (std::size_t)lay.img_tokens * cfg.patch_dim();
  auto fwd = [&](MetalZImageTransformer* m, std::vector<float>* out) {
    std::string ferr;
    metal_compute::SharedBuffer o = m->forward(req, &ferr);
    if (o.empty() || o.byte_size() < n * 2) {
      std::printf("[z_image_dit] forward failed: %s\n", ferr.c_str());
      return false;
    }
    if (out != nullptr) {
      const auto* d = static_cast<const std::uint16_t*>(o.contents());
      out->resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t u = (std::uint32_t)d[i] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        (*out)[i] = f;
      }
    }
    return true;
  };
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      const double e = (double)a[i] - (double)b[i];
      num += e * e;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  std::vector<float> out[kArms];
  {
    std::unique_ptr<MetalZImageTransformer> m[kArms];
    for (int a = 0; a < kArms; ++a) {
      m[a] = make(a, {});
      ZI_REQUIRE(m[a] != nullptr);
      ZI_REQUIRE(fwd(m[a].get(), &out[a]));
    }
    // Asked AFTER the forward: the modules are created on the first one,
    // when the row count is finally known.
    EXPECT_TRUE(!m[kGpu]->ane_armed() && !m[kGpu]->ane_qkv_armed());
    EXPECT_TRUE(!m[kFf]->ane_qkv_armed());
    if (!m[kFf]->ane_armed()) {
      // No ANE on this box, or the module would not compile: the tier
      // declining is a valid outcome, and not one to measure.
      std::printf("[z_image_dit] ANE did not arm at %dx%d -- SKIPPED\n", px,
                  px);
      return;
    }
    EXPECT_TRUE(m[kQkv]->ane_armed() && m[kQkv]->ane_qkv_armed());
    const double r_ff = rel(out[kFf], out[kGpu]);
    const double r_qkv = rel(out[kQkv], out[kGpu]);
    std::printf("[z_image_dit] ANE split vs GPU-only at %dx%d (%d image "
                "rows, %d blocks%s): rel-L2 %.4e ff, %.4e ff+qkv\n", px, px,
                lay.img_tokens, cfg.n_blocks(), stream ? ", streamed" : "",
                r_ff, r_qkv);
    EXPECT_TRUE(r_ff < 0.05);
    EXPECT_TRUE(r_qkv < 0.05);

    if (reps > 0) {
      double t[kArms] = {};
      std::vector<double> per[kArms];
      for (int k = 0; k < reps; ++k) {
        for (int j = 0; j < kArms; ++j) {
          const int a = (j + k) % kArms;
          const auto t0 = std::chrono::steady_clock::now();
          ZI_REQUIRE(fwd(m[a].get(), nullptr));
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
      std::printf("[z_image_dit] ANE bench %dx%d%s, %d reps rotated: %s "
                  "%.0f, %s %.0f (%.3fx), %s %.0f ms/fwd (%.3fx; %.3fx over "
                  "ff) | %s| %s| %s\n", px, px, stream ? " streamed" : "",
                  reps, kName[kGpu], t[kGpu] / reps, kName[kFf],
                  t[kFf] / reps, t[kGpu] / t[kFf], kName[kQkv],
                  t[kQkv] / reps, t[kGpu] / t[kQkv], t[kFf] / t[kQkv],
                  list(per[kGpu]).c_str(), list(per[kFf]).c_str(),
                  list(per[kQkv]).c_str());
    }
  }

  // ---- the adapter reaches the ANE's rows ---------------------------
  const int H = cfg.hidden, FFI = cfg.ffn_inner, R = 8;
  std::vector<std::pair<std::string, std::vector<std::int64_t>>> ts;
  std::vector<std::vector<std::uint16_t>> data;
  unsigned st = 0x5eedu;
  auto fill = [&](std::size_t cnt, float amp) {
    std::vector<std::uint16_t> v(cnt);
    for (auto& e : v) {
      st = st * 1664525u + 1013904223u;
      const float f = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f) * amp;
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      e = (std::uint16_t)(u >> 16);
    }
    return v;
  };
  // The effect scales with amp^2: 0.03 moved the output only 1.1e-2,
  // under the tiers' own ~4e-2 rounding, where a missing adapter and a
  // present one read alike.
  const float kAmp = 0.13f;
  for (int i = 0; i < cfg.n_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    struct M { const char* name; int n, k; };
    const M mods[] = {{"attention.to_q", H, H}, {"attention.to_k", H, H},
                      {"attention.to_v", H, H}, {"feed_forward.w1", FFI, H},
                      {"feed_forward.w3", FFI, H}, {"feed_forward.w2", H, FFI}};
    for (const M& mo : mods) {
      ts.push_back({p + mo.name + ".lora_A.weight", {R, mo.k}});
      data.push_back(fill((std::size_t)R * mo.k, kAmp));
      ts.push_back({p + mo.name + ".lora_B.weight", {mo.n, R}});
      data.push_back(fill((std::size_t)mo.n * R, kAmp));
    }
  }
  const std::filesystem::path lp =
      std::filesystem::temp_directory_path() / "z_image_ane_lora.safetensors";
  ZI_REQUIRE(write_bf16_st_(lp, ts, data));
  std::vector<MetalZImageTransformer::LoraSpec> lo = {{lp.string(), 1.0f}};
  std::vector<float> gl, al;
  int bound = 0;
  {
    auto m = make(kGpu, lo);
    ZI_REQUIRE(m != nullptr);
    bound = m->lora_modules(0);
    ZI_REQUIRE(fwd(m.get(), &gl));
  }
  {
    auto m = make(kQkv, lo);
    ZI_REQUIRE(m != nullptr);
    ZI_REQUIRE(fwd(m.get(), &al));
    EXPECT_TRUE(m->ane_armed() && m->ane_qkv_armed());
  }
  std::error_code ec;
  std::filesystem::remove(lp, ec);
  const double moved = rel(gl, out[kGpu]);
  const double r_lora = rel(al, gl);
  std::printf("[z_image_dit] adapter on q/k/v + ff of %d blocks (%d modules "
              "bound): moves the output %.4e; ANE ff+qkv vs GPU-only with "
              "it %.4e (without it %.4e)\n", cfg.n_layers, bound, moved,
              r_lora, rel(out[kQkv], out[kGpu]));
  EXPECT_TRUE(bound == 6 * cfg.n_layers);
  // Moved well past the tiers' own rounding, so an adapter missing from
  // half the rows could not hide inside it...
  EXPECT_TRUE(moved > 0.12);
  // ...and yet the ANE agrees with the GPU as closely as it does on the
  // base model.
  EXPECT_TRUE(r_lora < 0.06);
}
