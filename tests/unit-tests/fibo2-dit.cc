// Fibo 2.0's DiT: the layout, the packing, the edit prompt and the
// forward.
//
// The block is Z-Image's; what is new is each a SILENT failure when
// wrong -- a plausible picture, not a crash -- and these pin them:
//
//   * per-token modulation (the image rows at the timestep, the context
//     and gist rows at t = 0),
//   * the timestep as sigma * t_scale (Z-Image runs it backwards),
//   * the gist tokens: a Perceiver over text bundle 0, as many as the
//     text's length says,
//   * the injection blocks' gated cross-attention to their own bundle,
//   * the rotary planes (gist k at 1 + k, the image at max_gist + 1, the
//     context images on the planes after it),
//   * and the patch row's (channel, dy, dx) order.
//
// Env: VPIPE_FIBO2_GOLDEN -> the `small` dir tools/dump_fibo2_golden.py
// writes. Without it the numeric tests skip vacuously, so check that
// they RAN.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/fibo2/fibo2-layout.h"
#include "generative-models/fibo2/metal-fibo2-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/weight-set.h"
#include "stages/model-quantize-stage.h"
#include "residency-reserve.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

namespace fs = std::filesystem;

// minitest's ASSERT_TRUE does NOT abort, so every follow-on step has to
// be guarded by hand. `cond` is evaluated once.
#define FB_REQUIRE(cond)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
golden_dir_(const char* env = "VPIPE_FIBO2_GOLDEN")
{
  const char* g = std::getenv(env);
  return (g == nullptr || *g == '\0') ? std::string() : std::string(g);
}

std::vector<float>
read_f32_(const fs::path& p)
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
  if (got.size() != want.size()) { return 1e9; }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// One of the golden's scenarios, its inputs as the forward takes them.
struct Scenario {
  std::string name;
  int lh = 0, lw = 0, text_len = 0, gist = 0;
  std::vector<std::pair<int, int>> ctx;          // latent (h, w)
  std::vector<float> x, enc, out, out_bf16, gist_rows;
  std::vector<std::vector<float>> ctx_lat;
  double ref_bf16 = 0.0;
};

struct Golden {
  bool ok = false;
  fs::path dir;
  std::vector<Scenario> sc;
  // The wide golden's synthetic adapter: its fused model's fp32 output
  // on the first scenario, and how many modules it names.
  std::vector<float> lora_out;
  int lora_modules = 0;
};

// VPIPE_FIBO2_GOLDEN: the `small` golden; VPIPE_FIBO2_WIDE_GOLDEN: the
// `wide` one.
Golden
load_golden_(const char* env = "VPIPE_FIBO2_GOLDEN")
{
  Golden G;
  const std::string g = golden_dir_(env);
  if (g.empty()) { return G; }
  G.dir = fs::path(g);
  std::ifstream mf(G.dir / "golden.json");
  if (!mf) { return G; }
  FlexData man;
  try {
    man = FlexData::from_json(mf);
  } catch (...) {
    return G;
  }
  if (!man.is_object()) { return G; }
  if (man.as_object().contains("lora_modules")) {
    G.lora_modules = (int)man.as_object().at("lora_modules").as_int(0);
    G.lora_out = read_f32_(G.dir / "lora_out.f32");
  }
  const FlexData scs = man.as_object().at("scenarios");
  if (!scs.is_array()) { return G; }
  auto arr = scs.as_array();
  for (std::size_t i = 0; i < arr.size(); ++i) {
    const FlexData e = arr.at(i);
    auto o = e.as_object();
    Scenario s;
    s.name = std::string(o.at("name").as_string(""));
    s.text_len = (int)o.at("text_len").as_int(0);
    s.gist = (int)o.at("gist_count").as_int(0);
    s.ref_bf16 = o.at("ref_bf16_rel").as_real(0.0);
    const FlexData lat = o.at("latent");
    s.lh = (int)lat.as_array().at(0).as_int(0);
    s.lw = (int)lat.as_array().at(1).as_int(0);
    const FlexData cx = o.at("context");
    auto ca = cx.as_array();
    const fs::path d = G.dir / s.name;
    for (std::size_t k = 0; k < ca.size(); ++k) {
      const FlexData p = ca.at(k);
      s.ctx.emplace_back((int)p.as_array().at(0).as_int(0),
                         (int)p.as_array().at(1).as_int(0));
      s.ctx_lat.push_back(
          read_f32_(d / ("ctx" + std::to_string(k) + ".f32")));
    }
    s.x = read_f32_(d / "x.f32");
    s.enc = read_f32_(d / "enc.f32");
    s.out = read_f32_(d / "out.f32");
    s.out_bf16 = read_f32_(d / "out_bf16.f32");
    s.gist_rows = read_f32_(d / "gist.f32");
    if (s.x.empty() || s.enc.empty() || s.out.empty()) { return G; }
    G.sc.push_back(std::move(s));
  }
  G.ok = !G.sc.empty();
  return G;
}

std::string
ckpt_(const Golden& G)
{
  return (G.dir / "transformer").string();
}

// The image's patch rows, then every context image's, as one [RT, PD]
// bf16 buffer -- and the layout.
bool
inputs_(MetalCompute* mc, const MetalFibo2Transformer::Config& c,
        const Scenario& s, int gist, fibo2::Layout* lay,
        metal_compute::SharedBuffer* lat, std::string* err)
{
  const int P = c.patch, C = c.in_channels, PD = c.patch_dim();
  std::vector<std::pair<int, int>> grids;
  for (const auto& hw : s.ctx) {
    grids.emplace_back(hw.first / P, hw.second / P);
  }
  if (!fibo2::build_layout(s.lh / P, s.lw / P, grids, gist, c.max_gist,
                           lay, err)) {
    return false;
  }
  std::vector<float> rows((std::size_t)lay->refiner_len() * PD);
  fibo2::pack_latents(s.x.data(), C, s.lh, s.lw, P, rows.data());
  std::size_t off = (std::size_t)lay->img_tokens * PD;
  for (std::size_t k = 0; k < s.ctx.size(); ++k) {
    fibo2::pack_latents(s.ctx_lat[k].data(), C, s.ctx[k].first,
                        s.ctx[k].second, P, rows.data() + off);
    off += (std::size_t)(s.ctx[k].first / P) * (s.ctx[k].second / P) * PD;
  }
  *lat = to_bf16_(mc, rows);
  return !lat->empty();
}

// set_text + one forward at sigma 0.65; the unpatchified [C, lh, lw]
// velocity, which is the shape the reference writes.
std::vector<float>
run_(MetalFibo2Transformer* m, MetalCompute* mc, const Scenario& s,
     std::string* err, bool set_text = true)
{
  const MetalFibo2Transformer::Config& c = m->config();
  std::vector<float> out;
  if (set_text) {
    metal_compute::SharedBuffer txt = to_bf16_(mc, s.enc);
    if (txt.empty() || !m->set_text(txt, s.text_len, err)) { return out; }
  }
  fibo2::Layout lay;
  metal_compute::SharedBuffer lat;
  if (!inputs_(mc, c, s, m->gist_count(), &lay, &lat, err)) { return out; }
  MetalFibo2Transformer::Request req;
  req.latents = &lat;
  req.layout = &lay;
  req.timestep = 0.65f;
  metal_compute::SharedBuffer v = m->forward(req, err);
  if (v.empty()) { return out; }
  const int PD = c.patch_dim();
  const std::size_t rows = (std::size_t)lay.img_tokens;
  if (v.byte_size() < rows * (std::size_t)PD * 2) { return out; }
  std::vector<float> flat(rows * (std::size_t)PD);
  const auto* d = static_cast<const std::uint16_t*>(v.contents());
  for (std::size_t i = 0; i < flat.size(); ++i) { flat[i] = bf16_(d[i]); }
  out.assign((std::size_t)c.in_channels * s.lh * s.lw, 0.0f);
  fibo2::unpack_latents(flat.data(), c.in_channels, s.lh, s.lw, c.patch,
                        out.data());
  return out;
}

double
worst_diff_(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size()) { return 1e9; }
  double w = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    w = std::max(w, (double)std::fabs(a[i] - b[i]));
  }
  return w;
}

}  // namespace

// ---- the layout, which needs no checkpoint --------------------------

TEST(fibo2_layout, rows_and_planes)
{
  fibo2::Layout L;
  std::string err;
  // A 4 x 5 image, contexts 2 x 3 and 3 x 2, 64 gist tokens; the model's
  // max_gist 512 places the image on plane 513.
  FB_REQUIRE(fibo2::build_layout(4, 5, {{2, 3}, {3, 2}}, 64, 512, &L,
                                 &err));
  EXPECT_TRUE(L.img_tokens == 20);
  EXPECT_TRUE(L.ctx_tokens == 12);
  EXPECT_TRUE(L.refiner_len() == 32);
  EXPECT_TRUE(L.joint_len() == 96);
  EXPECT_TRUE(L.gist_off() == 32);
  // The image first, on plane max_gist + 1, row-major.
  EXPECT_TRUE(L.pos_t[0] == 513 && L.pos_h[0] == 0 && L.pos_w[0] == 0);
  EXPECT_TRUE(L.pos_t[19] == 513 && L.pos_h[19] == 3 && L.pos_w[19] == 4);
  EXPECT_TRUE(L.pos_h[5] == 1 && L.pos_w[5] == 0);
  // Context k on plane max_gist + 2 + k, its own rows and columns.
  EXPECT_TRUE(L.pos_t[20] == 514 && L.pos_h[20] == 0 && L.pos_w[20] == 0);
  EXPECT_TRUE(L.pos_t[25] == 514 && L.pos_h[25] == 1 && L.pos_w[25] == 2);
  EXPECT_TRUE(L.pos_t[26] == 515 && L.pos_h[26] == 0 && L.pos_w[26] == 0);
  EXPECT_TRUE(L.pos_t[31] == 515 && L.pos_h[31] == 2 && L.pos_w[31] == 1);
  // Gist k at (1 + k, 0, 0).
  EXPECT_TRUE(L.pos_t[32] == 1 && L.pos_h[32] == 0 && L.pos_w[32] == 0);
  EXPECT_TRUE(L.pos_t[95] == 64 && L.pos_h[95] == 0 && L.pos_w[95] == 0);
}

TEST(fibo2_layout, gist_count_follows_the_text_length)
{
  // The released model's rule: 256..2048 tokens onto 256..512 gists,
  // rounded UP to a multiple of 64.
  auto gc = [](int n) {
    return fibo2::gist_count(n, 256, 512, 64, 256, 2048);
  };
  EXPECT_TRUE(gc(1) == 256);
  EXPECT_TRUE(gc(256) == 256);
  EXPECT_TRUE(gc(257) == 320);     // 256.14 -> up to 320
  EXPECT_TRUE(gc(704) == 320);     // exactly 320
  EXPECT_TRUE(gc(705) == 384);
  EXPECT_TRUE(gc(2048) == 512);
  EXPECT_TRUE(gc(100000) == 512);
}

TEST(fibo2_layout, patch_rows_are_channel_then_dy_then_dx)
{
  // A [2, 4, 4] latent, patch 2: row (py, px) feature c*4 + dy*2 + dx.
  std::vector<float> z(2 * 4 * 4);
  for (std::size_t i = 0; i < z.size(); ++i) { z[i] = (float)i; }
  std::vector<float> rows(4 * 8);
  fibo2::pack_latents(z.data(), 2, 4, 4, 2, rows.data());
  // Row 1 = patch (0, 1): c 0 dy 0 dx 0 is z[0][0][2] = 2.
  EXPECT_TRUE(rows[8 + 0] == 2.0f);
  EXPECT_TRUE(rows[8 + 1] == 3.0f);      // dx 1
  EXPECT_TRUE(rows[8 + 2] == 6.0f);      // dy 1: z[0][1][2]
  EXPECT_TRUE(rows[8 + 4] == 18.0f);     // c 1: z[1][0][2]
  std::vector<float> back(z.size());
  fibo2::unpack_latents(rows.data(), 2, 4, 4, 2, back.data());
  EXPECT_TRUE(back == z);
}

TEST(fibo2_layout, image_sizes_are_the_trained_buckets)
{
  // The reference's get_image_size for each aspect, verbatim: the least
  // crop, then the area closest to 1024^2, then the wider.
  struct Case { double ar; int w, h; };
  const Case cases[] = {
      {1.0, 1024, 1024},        {4.0 / 3.0, 1152, 864},
      {3.0 / 4.0, 864, 1152},   {1.5, 1248, 832},
      {2.0 / 3.0, 832, 1248},   {16.0 / 9.0, 1424, 800},
      {9.0 / 16.0, 800, 1424},  {21.0 / 9.0, 1568, 672}};
  for (const Case& c : cases) {
    const auto s = fibo2::image_size(c.ar);
    if (s.first != c.w || s.second != c.h) {
      std::printf("[fibo2_layout] aspect %.4f: %dx%d, the reference %dx%d\n",
                  c.ar, s.first, s.second, c.w, c.h);
    }
    EXPECT_TRUE(s.first == c.w && s.second == c.h);
  }
}

TEST(fibo2_layout, edit_prompt_is_the_reference_json)
{
  // Each expectation is the reference's _get_edit_prompt output for the
  // same input, verbatim (compact separators, insertion order).
  std::string out, err;
  FB_REQUIRE(fibo2::edit_prompt("make it blue", 1, false, &out, &err));
  EXPECT_TRUE(out ==
              "{\"image\":\"<|vision_start|><|image_pad|><|vision_end|>\","
              "\"edit_instruction\":\"make it blue\"}");
  FB_REQUIRE(fibo2::edit_prompt("swap them", 2, false, &out, &err));
  EXPECT_TRUE(out ==
              "{\"image_1\":\"<|vision_start|><|image_pad|><|vision_end|>\","
              "\"image_2\":\"<|vision_start|><|image_pad|><|vision_end|>\","
              "\"edit_instruction\":\"swap them\"}");
  // A JSON prompt: the caption cleaned (empty values and scores gone).
  FB_REQUIRE(fibo2::edit_prompt(
      "{\"edit_instruction\": \"add a hat\", \"structured_caption\": "
      "{\"short_description\": \"a cat\", \"aesthetic_score\": 7, "
      "\"style\": \"\"}}",
      1, false, &out, &err));
  EXPECT_TRUE(out ==
              "{\"image\":\"<|vision_start|><|image_pad|><|vision_end|>\","
              "\"structured_caption\":{\"short_description\":\"a cat\"},"
              "\"edit_instruction\":\"add a hat\"}");
}

// ---- the forward against the reference ------------------------------

TEST(fibo2_dit, read_dims_reads_the_generated_config)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  EXPECT_TRUE(cfg.hidden == 256 && cfg.n_heads == 2);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.n_layers == 6 && cfg.n_refiner == 2);
  EXPECT_TRUE(cfg.cap_dim == 384 && cfg.n_bundles() == 3);
  EXPECT_TRUE(cfg.injection_ids.size() == 2 &&
              cfg.injection_ids[0] == 1 && cfg.injection_ids[1] == 4);
  EXPECT_TRUE(cfg.bundle_of(1) == 1 && cfg.bundle_of(4) == 2 &&
              cfg.bundle_of(0) == -1);
  // (dim * 8 // 3) // 64 * 64, not Z-Image's int(dim / 3 * 8).
  EXPECT_TRUE(cfg.ffn_inner == 640);
  EXPECT_TRUE(cfg.min_gist == 32 && cfg.max_gist == 64);
  EXPECT_TRUE(cfg.axes_lens[0] == 256);
}

TEST(fibo2_dit, read_dims_refuses_another_class)
{
  const fs::path d = fs::temp_directory_path() / "vpipe_fibo2_cfg_test";
  std::error_code ec;
  fs::create_directories(d, ec);
  auto put = [&](const char* json) {
    std::ofstream o(d / "config.json");
    o << json;
  };
  MetalFibo2Transformer::Config cfg;
  put(R"({"_class_name":"ZImageTransformer2DModel","dim":3840})");
  EXPECT_FALSE(MetalFibo2Transformer::Config::read_dims(d.string(), &cfg));
  // An injection block past the stack.
  put(R"({"_class_name":"BriaFibo2Transformer2DModel",
          "injection_layer_ids":[6,44]})");
  EXPECT_FALSE(MetalFibo2Transformer::Config::read_dims(d.string(), &cfg));
  // The image plane past the rotary table.
  put(R"({"_class_name":"BriaFibo2Transformer2DModel",
          "axes_lens":[256,512,512]})");
  EXPECT_FALSE(MetalFibo2Transformer::Config::read_dims(d.string(), &cfg));
  fs::remove_all(d, ec);
}

// THE THREE SCENARIOS, each against the reference's fp32 forward, with
// the reference's OWN bf16 distance as the bar: an edit with one context
// image, a generation with the shortest text (the fewest gists), and two
// context images of different sizes. And the gist rows themselves.
TEST(fibo2_dit, matches_reference_on_a_small_model)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr);
  for (const Scenario& s : G.sc) {
    std::string err;
    const std::vector<float> got = run_(m.get(), mc, s, &err);
    if (got.empty()) {
      std::printf("[fibo2_dit] %s: forward failed: %s\n", s.name.c_str(),
                  err.c_str());
    }
    FB_REQUIRE(!got.empty());
    EXPECT_TRUE(m->gist_count() == s.gist);
    const double rel = rel_l2_(got, s.out);
    // The gist rows, as the Perceiver left them.
    std::vector<float> gist;
    const metal_compute::SharedBuffer& gb = m->gist();
    const std::size_t gn = (std::size_t)s.gist * cfg.hidden;
    if (gb.byte_size() >= gn * 2) {
      const auto* d = static_cast<const std::uint16_t*>(gb.contents());
      for (std::size_t i = 0; i < gn; ++i) { gist.push_back(bf16_(d[i])); }
    }
    const double grel = rel_l2_(gist, s.gist_rows);
    std::printf("[fibo2_dit] %s: velocity rel-L2 %.5f (reference bf16 "
                "%.5f), gist %.5f, %d gists, %zu context images\n",
                s.name.c_str(), rel, s.ref_bf16, grel, s.gist,
                s.ctx.size());
    EXPECT_TRUE(rel < 1.5 * s.ref_bf16);
    EXPECT_TRUE(grel < 0.02);
  }
}

// THE TEXT'S PASS RUNS ONCE: a second forward reads what the first
// left -- byte-identical -- and a new text takes its place.
TEST(fibo2_dit, the_text_pass_is_kept_between_forwards)
{
  const Golden G = load_golden_();
  if (!G.ok || G.sc.size() < 2) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr);
  std::string err;
  const std::vector<float> a = run_(m.get(), mc, G.sc[0], &err);
  const std::vector<float> b =
      run_(m.get(), mc, G.sc[0], &err, /*set_text=*/false);
  FB_REQUIRE(!a.empty() && !b.empty());
  EXPECT_TRUE(worst_diff_(a, b) == 0.0);
  // Another text, then the first again: the first's answer again.
  const std::vector<float> c = run_(m.get(), mc, G.sc[1], &err);
  FB_REQUIRE(!c.empty());
  const std::vector<float> d = run_(m.get(), mc, G.sc[0], &err);
  EXPECT_TRUE(worst_diff_(a, d) == 0.0);
  // A layout whose gist count is not the text's is refused by name.
  {
    fibo2::Layout lay;
    metal_compute::SharedBuffer lat;
    FB_REQUIRE(inputs_(mc, cfg, G.sc[0], m->gist_count() + 16, &lay, &lat,
                       &err));
    MetalFibo2Transformer::Request req;
    req.latents = &lat;
    req.layout = &lay;
    req.timestep = 0.5f;
    std::string ferr;
    EXPECT_TRUE(m->forward(req, &ferr).empty());
    EXPECT_TRUE(ferr.find("gist") != std::string::npos);
  }
}

// A STREAMED stack must be BIT-IDENTICAL to a preloaded one, with the
// slot pair and without it (the prefetch-without-a-destination A/B).
TEST(fibo2_dit, streamed_matches_preloaded)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  const Scenario& s = G.sc.back();      // two context images
  auto pre = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(pre != nullptr);
  std::string e;
  const std::vector<float> a = run_(pre.get(), mc, s, &e);
  pre.reset();
  for (int no_slots = 0; no_slots < 2; ++no_slots) {
    if (no_slots) { setenv("VPIPE_FIBO2_NO_SLOTS", "1", 1); }
    auto str = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, true);
    std::vector<float> b;
    if (str != nullptr) {
      EXPECT_TRUE(str->streaming());
      b = run_(str.get(), mc, s, &e);
      str.reset();
    }
    if (no_slots) { unsetenv("VPIPE_FIBO2_NO_SLOTS"); }
    FB_REQUIRE(!a.empty() && !b.empty());
    const double w = worst_diff_(a, b);
    std::printf("[fibo2_dit] streamed%s vs preloaded: worst |diff| %g\n",
                no_slots ? " (no slots)" : "", w);
    EXPECT_TRUE(w == 0.0);
  }
}

// The injection blocks' trunk must not size the slot: the streaming
// floor counts their cross-attention once, in the trunk.
TEST(fibo2_dit, streaming_floor_holds_the_cross_attention_in_the_trunk)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  const std::size_t f =
      MetalFibo2Transformer::streaming_floor_bytes(ckpt_(G));
  FB_REQUIRE(f > 0);
  // The small model: H 256, F 640, all bf16. A plain block is q/k/v/o
  // (4 H^2), the SwiGLU (3 H F), the adaLN (4H x 256 + 4H) and six norms.
  const std::size_t H = 256, F = 640;
  const std::size_t block = (4 * H * H + 3 * H * F + 4 * H * 256 + 4 * H +
                             4 * H + 2 * 128) * 2;
  std::size_t total = 0;
  for (const auto& e : fs::directory_iterator(G.dir / "transformer")) {
    if (e.path().extension() == ".safetensors") {
      total += (std::size_t)e.file_size();
    }
  }
  // Trunk + two blocks: everything less six of the eight blocks.
  const std::size_t want = total - 6 * block;
  std::printf("[fibo2_dit] floor %zu, a block %zu, file %zu\n", f, block,
              total);
  // The file carries a header; allow it.
  EXPECT_TRUE(f <= want && want - f < 64 * 1024);
}

// ---- the wide golden: every tier's floor cleared -------------------------
//
// VPIPE_FIBO2_WIDE_GOLDEN (tools/dump_fibo2_golden.py `wide`): dim 1024
// (int8's K floor), 1024 image + 1024 context rows (int8's row floor,
// Sol's 1024 and its 16 key blocks, Sage's 512), a 2176-row joint stack
// (one 2048-row ANE chunk). The small golden clears none of them, so a
// tier tested there passes having run nothing.

namespace {

const char* kWide = "VPIPE_FIBO2_WIDE_GOLDEN";

// The floors, stated so a golden that stops clearing one fails (each
// caller REQUIREs it) rather than passing vacuously.
bool
wide_floors_(const Golden& G, const MetalFibo2Transformer::Config& c)
{
  if (G.sc.empty()) { return false; }
  const Scenario& s = G.sc[0];
  const int img = (s.lh / c.patch) * (s.lw / c.patch);
  int ctx = 0;
  for (const auto& hw : s.ctx) {
    ctx += (hw.first / c.patch) * (hw.second / c.patch);
  }
  return c.hidden >= 1024 && img >= 1024 && ctx >= 1024 &&
         img + ctx + s.gist >= 2048;
}

}  // namespace

TEST(fibo2_dit, wide_matches_reference)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  FB_REQUIRE(wide_floors_(G, cfg));
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr);
  // ENGAGEMENT BEFORE ACCURACY: on matrix cores the GEMMs are matmul2d
  // and the attention NAX; elsewhere the widest steel tile.
  if (mc->supports_matrix_cores()) {
    EXPECT_TRUE(m->uses_mma2());
    EXPECT_TRUE(m->uses_attn_nax());
  } else {
    EXPECT_TRUE(m->gemm_tile() == 2);
  }
  std::string err;
  const std::vector<float> got = run_(m.get(), mc, G.sc[0], &err);
  FB_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, G.sc[0].out);
  std::printf("[fibo2_dit] wide (%s): velocity rel-L2 %.5f (the reference's "
              "bf16 %.5f), %d gists\n",
              mc->supports_matrix_cores() ? "matmul2d + NAX" : "steel",
              rel, G.sc[0].ref_bf16, m->gist_count());
  EXPECT_TRUE(rel < 1.5 * G.sc[0].ref_bf16);
}

// THE INT8 TIERS (an M5): accurate, fast and token each ENGAGE and stay
// within a gross bar of the reference -- a random-input forward is
// chaotic, so two equally accurate routes rank either way; the accuracy
// claim is the GEMM oracle's (gemm_i8.*).
TEST(fibo2_dit, i8_tiers_engage_and_stay_accurate)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  FB_REQUIRE(wide_floors_(G, cfg));
  std::string err;
  std::vector<float> off;
  {
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
    FB_REQUIRE(m != nullptr);
    off = run_(m.get(), mc, G.sc[0], &err);
  }
  FB_REQUIRE(!off.empty());
  const char* const kTier[] = {"accurate", "fast", "token", "token_plain"};
  for (int tier = 0; tier < 4; ++tier) {
    MetalFibo2Transformer::Config c = cfg;
    c.i8_gemm = true;
    c.i8_tier = tier;
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, c, false);
    FB_REQUIRE(m != nullptr);
    const bool engaged =
        m->uses_i8_gemm() && (tier == 0 || m->fast_tier());
    const std::vector<float> on = run_(m.get(), mc, G.sc[0], &err);
    FB_REQUIRE(!on.empty());
    const double d = rel_l2_(on, off);
    const double r = rel_l2_(on, G.sc[0].out);
    std::printf("[fibo2_dit] i8 %s%s: rel-L2 vs bf16 %.5f, vs reference "
                "%.5f (bf16 alone %.5f)\n", kTier[tier],
                engaged ? "" : " NOT ENGAGED", d, r,
                rel_l2_(off, G.sc[0].out));
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(std::isfinite(d) && std::isfinite(r));
    EXPECT_TRUE(d > 0.0);
    EXPECT_TRUE(r < 0.12);
  }
}

// SOL, on either arm (the matrix cores' or the ALU one an M4 runs).
// APPROXIMATE by construction, so a sanity bound, not an equality.
TEST(fibo2_dit, sol_attention_engages_and_stays_close)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  FB_REQUIRE(wide_floors_(G, cfg));
  std::string err;
  std::vector<float> off;
  {
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
    FB_REQUIRE(m != nullptr);
    off = run_(m.get(), mc, G.sc[0], &err);
  }
  cfg.sol.enabled = true;
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr);
  FB_REQUIRE(m->uses_sol_attn());
  const std::vector<float> on = run_(m.get(), mc, G.sc[0], &err);
  FB_REQUIRE(!off.empty() && on.size() == off.size());
  const double d = rel_l2_(on, off);
  std::printf("[fibo2_dit] sol (tau %.2f, %s arm): rel-L2 vs dense %.5f, "
              "vs reference %.5f\n", (double)cfg.sol.tau,
              mc->supports_matrix_cores() ? "matrix-core" : "ALU", d,
              rel_l2_(on, G.sc[0].out));
  EXPECT_TRUE(std::isfinite(d));
  EXPECT_TRUE(d > 0.0);
  EXPECT_TRUE(d < 0.6);
}

// SAGE (the NAX attention, an M5), its scratch built ONCE: the refiner
// and the main stack attend at different row counts and share the
// reservation, and what is held is what the plan books.
TEST(fibo2_dit, sage_attention_engages_and_builds_its_scratch_once)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  FB_REQUIRE(wide_floors_(G, cfg));
  std::string err;
  std::vector<float> off;
  {
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
    FB_REQUIRE(m != nullptr);
    off = run_(m.get(), mc, G.sc[0], &err);
  }
  cfg.sage.enabled = true;
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr);
  if (!m->uses_sage_attn()) {
    std::printf("[fibo2_dit] sage_attn needs the NAX attention (an M5); "
                "skipped\n");
    return;
  }
  const std::vector<float> on = run_(m.get(), mc, G.sc[0], &err);
  const std::vector<float> on2 =
      run_(m.get(), mc, G.sc[0], &err, /*set_text=*/false);
  const int builds = m->sage_scratch_builds();
  const std::size_t held = m->sage_scratch_held();
  FB_REQUIRE(!off.empty() && on.size() == off.size() &&
             on2.size() == on.size());
  const double d = rel_l2_(on, off);
  const double r = rel_l2_(on, G.sc[0].out);
  const bool same2 =
      std::memcmp(on.data(), on2.data(), on.size() * sizeof(float)) == 0;
  MetalFibo2Transformer::Config dense = cfg;
  dense.sage.enabled = false;
  const Scenario& s = G.sc[0];
  int rt = (s.lh / 2) * (s.lw / 2);
  for (const auto& hw : s.ctx) { rt += (hw.first / 2) * (hw.second / 2); }
  const std::size_t booked =
      MetalFibo2Transformer::forward_scratch_bytes(cfg, rt, s.text_len,
                                                   false, true) -
      MetalFibo2Transformer::forward_scratch_bytes(dense, rt, s.text_len,
                                                   false, true);
  std::printf("[fibo2_dit] sage: rel-L2 vs bf16 %.5f, vs reference %.5f; "
              "scratch %d build(s) over two forwards, held %zu KB, booked "
              "%zu KB; second forward %s\n", d, r, builds, held >> 10,
              booked >> 10, same2 ? "bit-identical" : "DIFFERS");
  // Against the reference's OWN bf16 distance: a Sage arm reading its
  // operands in the wrong layout lands ~0.08, under any fixed loose bar.
  EXPECT_TRUE(std::isfinite(d) && d > 0.0 &&
              r < 2.0 * G.sc[0].ref_bf16);
  EXPECT_TRUE(builds == 1);
  EXPECT_TRUE(same2);
  EXPECT_TRUE(held > 0 && held == booked);
}

// SAGE'S INT8 P*V (`sage_pv8`, an M5): the int8 QK run again with P*V in
// int8 too. Asserted ENGAGED first -- the flash function reads the int8
// operands only where the prologue made them (pv8_for), so a host without
// the kernels runs plain Sage and would pass every numeric bar -- then
// that it moved the plain Sage output, stays at the reference's bar, is
// the same on a second forward, and holds what the plan books.
TEST(fibo2_dit, sage_int8_pv_engages_and_stays_close)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  FB_REQUIRE(wide_floors_(G, cfg));
  cfg.sage.enabled = true;
  std::string err;
  std::vector<float> qk;
  {
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
    FB_REQUIRE(m != nullptr);
    if (!m->uses_sage_attn()) {
      std::printf("[fibo2_dit] sage_attn needs the NAX attention (an M5); "
                  "skipped\n");
      return;
    }
    qk = run_(m.get(), mc, G.sc[0], &err);
    EXPECT_TRUE(!m->sage_pv8_ready());
  }
  cfg.sage.pv8 = true;
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr && m->uses_sage_attn());
  const std::vector<float> on = run_(m.get(), mc, G.sc[0], &err);
  const bool engaged = m->sage_pv8_ready();
  const std::vector<float> on2 =
      run_(m.get(), mc, G.sc[0], &err, /*set_text=*/false);
  const std::size_t held = m->sage_scratch_held();
  FB_REQUIRE(!qk.empty() && on.size() == qk.size() &&
             on2.size() == on.size());
  const double d = rel_l2_(on, qk);
  const double r = rel_l2_(on, G.sc[0].out);
  const bool same2 =
      std::memcmp(on.data(), on2.data(), on.size() * sizeof(float)) == 0;
  MetalFibo2Transformer::Config dense = cfg;
  dense.sage.enabled = false;
  const Scenario& s = G.sc[0];
  int rt = (s.lh / 2) * (s.lw / 2);
  for (const auto& hw : s.ctx) { rt += (hw.first / 2) * (hw.second / 2); }
  const std::size_t booked =
      MetalFibo2Transformer::forward_scratch_bytes(cfg, rt, s.text_len,
                                                   false, true) -
      MetalFibo2Transformer::forward_scratch_bytes(dense, rt, s.text_len,
                                                   false, true);
  std::printf("[fibo2_dit] sage pv8: %s; rel-L2 vs int8 QK alone %.5f, vs "
              "reference %.5f; held %zu KB, booked %zu KB; second forward "
              "%s\n", engaged ? "ENGAGED" : "NOT engaged", d, r,
              held >> 10, booked >> 10, same2 ? "bit-identical" : "DIFFERS");
  EXPECT_TRUE(engaged);
  // MUTATION: the flash function built without the int8 P*V's constants
  // over a prologue that made its operands read 0.077 from the reference
  // (this arm 0.013) -- inside a fixed 0.12, so the bar is the
  // reference's own bf16 distance.
  EXPECT_TRUE(std::isfinite(d) && d > 0.0 &&
              r < 2.0 * G.sc[0].ref_bf16);
  EXPECT_TRUE(same2);
  EXPECT_TRUE(held > 0 && held == booked);
}

// RUNTIME LoRA against the reference's OWN dense arithmetic: the dumper
// writes a synthetic adapter AND the model with those factors fused, so
// `base + runtime adapter` is held to `(base + B@A)` run in fp32. Three
// things asserted: every named module BOUND, the adapter MOVES the base
// well past the bf16 floor, and scale 0 is the base EXACTLY.
TEST(fibo2_dit, runtime_lora_matches_a_fused_reference)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok || G.lora_out.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  std::string err;
  std::vector<float> base;
  {
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
    FB_REQUIRE(m != nullptr);
    base = run_(m.get(), mc, G.sc[0], &err);
  }
  FB_REQUIRE(!base.empty());
  std::vector<MetalFibo2Transformer::LoraSpec> specs;
  specs.push_back({(G.dir / "lora.safetensors").string(), 1.0f});
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false, specs);
  FB_REQUIRE(m != nullptr);
  const int bound = m->lora_modules(0);
  std::printf("[fibo2_dit] lora bound %d of %d modules\n", bound,
              G.lora_modules);
  FB_REQUIRE(bound == G.lora_modules);
  const std::vector<float> got = run_(m.get(), mc, G.sc[0], &err);
  FB_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, G.lora_out);
  const double moved = rel_l2_(base, G.lora_out);
  std::printf("[fibo2_dit] lora: rel-L2 vs the fused reference %.5f (the "
              "adapter moves the base by %.5f)\n", rel, moved);
  EXPECT_TRUE(moved > 0.05);
  EXPECT_TRUE(rel < 1.5 * G.sc[0].ref_bf16);
  m->set_lora_scale(0, 0.0f);
  const std::vector<float> zero =
      run_(m.get(), mc, G.sc[0], &err, /*set_text=*/false);
  FB_REQUIRE(zero.size() == base.size());
  EXPECT_TRUE(worst_diff_(zero, base) == 0.0);
}

// A STOP IS NOT A FAILURE, and it lands inside a forward -- on a TIMER
// from another thread, which is what a stop is: a flag tripped on the
// Nth check is set before any check runs, and cannot see a held stack
// that encodes every block and waits out the forward in one commit.
TEST(fibo2_dit, a_timed_stop_lands_inside_a_forward)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  for (int arm = 0; arm < 2; ++arm) {
    const bool stream = arm == 1;
    auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, stream);
    FB_REQUIRE(m != nullptr);
    if (stream) {
      // Residency off, so every block really streams.
      m->set_residency_reserve(0);
      m->set_residency_schedule(4);
    }
    std::string err;
    // A warm forward first, for its time.
    (void)run_(m.get(), mc, G.sc[0], &err);
    const auto w0 = std::chrono::steady_clock::now();
    const std::vector<float> full =
        run_(m.get(), mc, G.sc[0], &err, /*set_text=*/false);
    const double whole = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - w0).count();
    FB_REQUIRE(!full.empty());
    std::atomic<bool> flag{false};
    std::atomic<long long> set_at{0};
    m->set_stream_stop([&]() { return flag.load(); });
    const auto t0 = std::chrono::steady_clock::now();
    std::thread timer([&]() {
      std::this_thread::sleep_for(std::chrono::microseconds(
          (long long)(whole * 1000.0 / 3.0)));
      set_at = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - t0).count();
      flag = true;
    });
    fibo2::Layout lay;
    metal_compute::SharedBuffer lat;
    FB_REQUIRE(inputs_(mc, cfg, G.sc[0], m->gist_count(), &lay, &lat,
                       &err));
    MetalFibo2Transformer::Request req;
    req.latents = &lat;
    req.layout = &lay;
    req.timestep = 0.65f;
    std::string serr;
    metal_compute::SharedBuffer out = m->forward(req, &serr);
    const double done = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    timer.join();
    m->set_stream_stop({});
    const double latency = done - (double)set_at.load() / 1000.0;
    std::printf("[fibo2_dit] timed stop (%s): set at %.1f ms of a %.1f ms "
                "forward, returned %.1f ms later\n",
                stream ? "streamed" : "held", set_at.load() / 1000.0, whole,
                latency);
    EXPECT_TRUE(out.empty() && serr.empty());
    EXPECT_TRUE(done < 0.9 * whole);
  }
}

// THE FLOP/s METER: what the forward counts at encode is EXACTLY a count
// derived here from the structure -- the text pass on the first forward,
// the two timestep rows, each block's two modulation rows, the main
// stack's injection cross-attention -- and all of it credited as the
// buffers retire.
TEST(fibo2_dit, forward_credits_its_flops)
{
  namespace pf = vpipe::perf_flops;
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  auto m = MetalFibo2Transformer::load(ckpt_(G), mc, cfg, false);
  FB_REQUIRE(m != nullptr);
  const Scenario& s = G.sc[0];
  const long long H = cfg.hidden, F = cfg.ffn_inner, CAP = cfg.cap_dim;
  const long long T = s.text_len, Gn = s.gist, NH = cfg.n_heads,
                  Hd = cfg.head_dim;
  const long long IMG = (s.lh / 2) * (s.lw / 2);
  long long RT = IMG;
  for (const auto& hw : s.ctx) { RT += (hw.first / 2) * (hw.second / 2); }
  const long long JT = RT + Gn;
  auto block = [&](long long rows) {
    return 4 * gemm_flops(rows, H, H) + 2 * gemm_flops(rows, F, H) +
           gemm_flops(rows, H, F) + attention_flops(NH, rows * rows, Hd) +
           gemm_flops(2, 4 * H, cfg.adaln_dim);
  };
  std::uint64_t text = 0;
  for (int l = 0; l < cfg.perceiver_layers; ++l) {
    text += gemm_flops(Gn, H, H) + 2 * gemm_flops(T, H, CAP) +
            attention_flops(NH, Gn * T, Hd) + gemm_flops(Gn, H, H) +
            4 * gemm_flops(Gn, H, H) + attention_flops(NH, Gn * Gn, Hd) +
            2 * gemm_flops(Gn, F, H) + gemm_flops(Gn, H, F);
  }
  const long long NI = (long long)cfg.injection_ids.size();
  text += (std::uint64_t)NI * 2 * gemm_flops(T, H, CAP);
  const std::uint64_t forward =
      gemm_flops(2, cfg.time_mid, cfg.time_freq) +
      gemm_flops(2, cfg.adaln_dim, cfg.time_mid) +
      gemm_flops(RT, H, cfg.patch_dim()) +
      (std::uint64_t)cfg.n_refiner * block(RT) +
      (std::uint64_t)cfg.n_layers * block(JT) +
      (std::uint64_t)NI * (2 * gemm_flops(JT, H, H) +
                           attention_flops(NH, JT * T, Hd)) +
      gemm_flops(1, H, cfg.adaln_dim) + gemm_flops(IMG, cfg.patch_dim(), H);
  for (int pass = 0; pass < 2; ++pass) {
    const pf::Totals g0 = pf::totals(pf::kGpu);
    // The tally's total runs over the model's life: this forward's
    // share is the difference.
    const std::uint64_t f0 = m->gpu_flops().flushed();
    std::string err;
    const std::vector<float> v =
        run_(m.get(), mc, s, &err, /*set_text=*/pass == 0);
    FB_REQUIRE(!v.empty());
    const std::uint64_t flushed = m->gpu_flops().flushed() - f0;
    for (int i = 0; i < 400 &&
                    pf::totals(pf::kGpu).flops - g0.flops < flushed; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const std::uint64_t credited = pf::totals(pf::kGpu).flops - g0.flops;
    const std::uint64_t want = forward + (pass == 0 ? text : 0);
    std::printf("[fibo2_dit] forward %d (%s): counted %.4f TFLOP, derived "
                "%.4f, credited %.4f\n", pass + 1,
                pass == 0 ? "with the text pass" : "the text kept",
                flushed / 1e12, want / 1e12, credited / 1e12);
    EXPECT_TRUE(flushed == want);
    EXPECT_TRUE(credited == flushed);
  }
  // THE ANE SPLIT MOVES NO FLOPS (opt-in with the ANE test): the rows
  // the GPU gave up are credited to the ANE, in whole 2048-row chunks of
  // the SwiGLU's 6 H F a row.
  const char* ane = std::getenv("VPIPE_FIBO2_ANE_TEST");
  if (ane == nullptr || std::atoi(ane) == 0) { return; }
  MetalFibo2Transformer::Config c = cfg;
  c.ane_ffn = true;
  c.ane_rows = 1;
  auto a = MetalFibo2Transformer::load(ckpt_(G), mc, c, false);
  FB_REQUIRE(a != nullptr);
  const pf::Totals g0 = pf::totals(pf::kGpu);
  const pf::Totals a0 = pf::totals(pf::kAne);
  std::string err;
  FB_REQUIRE(!run_(a.get(), mc, s, &err).empty());
  const std::uint64_t flushed = a->gpu_flops().flushed();
  for (int i = 0; i < 400 &&
                  pf::totals(pf::kGpu).flops - g0.flops < flushed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!a->ane_armed() || a->ane_gpu_only()) {
    std::printf("[fibo2_dit] no ANE split here; the conservation half is "
                "skipped\n");
    return;
  }
  const std::uint64_t gpu = pf::totals(pf::kGpu).flops - g0.flops;
  const std::uint64_t on_ane = pf::totals(pf::kAne).flops - a0.flops;
  const std::uint64_t moved = forward + text - gpu;
  const std::uint64_t chunk = gemm_flops(2048, 3 * F, H);
  std::printf("[fibo2_dit] ANE split: GPU %.4f + ANE %.4f TFLOP (moved %.4f, "
              "%llu chunks)\n", gpu / 1e12, on_ane / 1e12, moved / 1e12,
              (unsigned long long)(moved / chunk));
  EXPECT_TRUE(gpu == flushed);
  EXPECT_TRUE(moved > 0 && moved % chunk == 0);
  EXPECT_TRUE(on_ane >= moved && (on_ane - moved) % chunk == 0);
}

// THE ANE TIERS' ROWS (opt-in, VPIPE_FIBO2_ANE_TEST=1; the M4 series is
// where the tier is for). The feed-forward split, and q/k/v beside it,
// against the GPU-only forward -- and with an adapter, which the stagers
// fold into the module rather than adding a path after it. One chunk
// pinned to the ANE (2048 of the 2176 main-stack rows).
TEST(fibo2_dit, ane_tiers_rows_are_right)
{
  const char* on = std::getenv("VPIPE_FIBO2_ANE_TEST");
  if (on == nullptr || std::atoi(on) == 0) { return; }
  const Golden G = load_golden_(kWide);
  if (!G.ok || G.lora_out.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(ckpt_(G), &cfg));
  FB_REQUIRE(wide_floors_(G, cfg));
  const std::string lp = (G.dir / "lora.safetensors").string();
  enum Arm { kGpu, kFf, kQkv, kArms };
  const char* const kName[kArms] = {"GPU-only", "ANE ff", "ANE ff+qkv"};
  for (int with_lora = 0; with_lora < 2; ++with_lora) {
    std::vector<MetalFibo2Transformer::LoraSpec> specs;
    if (with_lora) { specs.push_back({lp, 1.0f}); }
    std::vector<float> out[kArms];
    for (int a = 0; a < kArms; ++a) {
      MetalFibo2Transformer::Config c = cfg;
      c.ane_ffn = a != kGpu;
      c.ane_qkv = a == kQkv;
      c.ane_rows = a != kGpu ? 1 : 0;
      auto m = MetalFibo2Transformer::load(ckpt_(G), mc, c, false, specs);
      FB_REQUIRE(m != nullptr);
      std::string err;
      out[a] = run_(m.get(), mc, G.sc[0], &err);
      FB_REQUIRE(!out[a].empty());
      if (a == kFf && !m->ane_armed()) {
        std::printf("[fibo2_dit] the ANE did not arm here -- SKIPPED\n");
        return;
      }
      if (a != kGpu) {
        EXPECT_TRUE(m->ane_armed() && !m->ane_gpu_only());
        EXPECT_TRUE((a == kQkv) == m->ane_qkv_armed());
      }
    }
    const std::vector<float>& want =
        with_lora ? G.lora_out : G.sc[0].out;
    for (int a = 0; a < kArms; ++a) {
      const double r = rel_l2_(out[a], want);
      const double g = rel_l2_(out[a], out[kGpu]);
      std::printf("[fibo2_dit] %s%s: rel-L2 vs the reference %.5f, vs "
                  "GPU-only %.5f\n", kName[a], with_lora ? " + LoRA" : "",
                  r, g);
      EXPECT_TRUE(g < 0.05);
      EXPECT_TRUE(r < 3.0 * G.sc[0].ref_bf16);
    }
  }
}

// ---- quantized packs ----------------------------------------------------
//
// model-quantize over the wide golden's DiT: which tensors a pack takes,
// what it costs against the reference, and -- a MIXED pack, w8 and w4
// blocks in one stack -- that every hold of it is the same bytes.

namespace {

// A model-quantize'd copy of the wide golden's DiT in a temp dir (removed
// by the caller); empty on a failure.
fs::path
quantize_wide_(Session& sess, const Golden& G, int bits, bool mixed)
{
  const fs::path out =
      fs::temp_directory_path() /
      ("vpipe-fibo2-q" + std::to_string(bits) + (mixed ? "m-" : "-") +
       std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(out, ec);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(ckpt_(G)));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(bits));
    o.insert("skip_existing", FlexData::make_bool(false));
    if (mixed) { o.insert("mixed", FlexData::make_bool(true)); }
  }
  vpipe::ModelQuantizeStage st(&sess, "qfibo2", std::vector<InEdge>{},
                               std::move(cfg));
  if (!st.config_error().empty() || !st.quantize_once() ||
      !fs::exists(out / "config.json", ec)) {
    fs::remove_all(out, ec);
    return {};
  }
  return out;
}

}  // namespace

// A w8 pack takes every BLOCK projection -- the seven Linears and the
// adaLN, as Z-Image's does -- and NOTHING else: not the embedders or the
// final layer, and not the text path (the Perceiver, the injection
// blocks' cross-attention), whose leaves are the same words. Then the
// pack runs, near the reference, streamed byte-identical to preloaded.
TEST(fibo2_dit, quantize_takes_the_blocks_and_leaves_the_text_path)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const fs::path q = quantize_wide_(sess, G, 8, false);
  FB_REQUIRE(!q.empty());
  int blk_q = 0, other_q = 0;
  bool saw_attn = false, saw_ffn = false, saw_ada = false;
  {
    auto wts = MetalLlamaWeights::open_model(q.string());
    FB_REQUIRE(wts.has_value());
    for (const auto& n : wts->tensor_names()) {
      if (n.size() <= 7 || n.compare(n.size() - 7, 7, ".scales") != 0) {
        continue;
      }
      const std::string base = n.substr(0, n.size() - 7);
      const bool in_block = (base.rfind("layers.", 0) == 0 ||
                             base.rfind("noise_refiner.", 0) == 0) &&
                            base.find(".cross_attention.") ==
                                std::string::npos;
      if (!in_block) {
        ++other_q;
        std::printf("[fibo2_dit] UNEXPECTED quantized tensor: %s\n",
                    base.c_str());
        continue;
      }
      ++blk_q;
      saw_attn |= base.find("attention.to_") != std::string::npos;
      saw_ffn |= base.find("feed_forward.w") != std::string::npos;
      saw_ada |= base.find("adaLN_modulation.0") != std::string::npos;
    }
  }
  std::printf("[fibo2_dit] w8 pack: %d block tensors quantized, %d other\n",
              blk_q, other_q);
  EXPECT_TRUE(blk_q > 0 && saw_attn && saw_ffn && saw_ada);
  EXPECT_TRUE(other_q == 0);
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(q.string(), &cfg));
  std::vector<float> pre, str;
  {
    auto m = MetalFibo2Transformer::load(q.string(), mc, cfg, false);
    FB_REQUIRE(m != nullptr && m->quant_bits() == 8);
    std::string err;
    pre = run_(m.get(), mc, G.sc[0], &err);
  }
  {
    auto m = MetalFibo2Transformer::load(q.string(), mc, cfg, true);
    FB_REQUIRE(m != nullptr);
    std::string err;
    str = run_(m.get(), mc, G.sc[0], &err);
  }
  std::error_code ec;
  fs::remove_all(q, ec);
  FB_REQUIRE(!pre.empty() && !str.empty());
  const double r = rel_l2_(pre, G.sc[0].out);
  std::printf("[fibo2_dit] w8 pack: rel-L2 vs the reference %.5f (bf16 "
              "reference's own %.5f); streamed vs preloaded worst |diff| "
              "%g\n", r, G.sc[0].ref_bf16, worst_diff_(pre, str));
  EXPECT_TRUE(r < 3.0 * G.sc[0].ref_bf16);
  EXPECT_TRUE(worst_diff_(pre, str) == 0.0);
}

// A MIXED pack (model-quantize `mixed`: whole main blocks w8, the rest
// w4) crosses widths inside one slot pair, so a slot is REBUILT for a
// block stored otherwise (BlockSlots::Ops::same_layout). Every hold --
// preloaded, streamed, streamed with no slots, promoted -- the same
// bytes; on matrix cores under the int8 tiers too.
TEST(fibo2_dit, a_mixed_pack_holds_agree)
{
  const Golden G = load_golden_(kWide);
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const fs::path q = quantize_wide_(sess, G, 4, true);
  FB_REQUIRE(!q.empty());
  MetalFibo2Transformer::Config cfg;
  FB_REQUIRE(MetalFibo2Transformer::Config::read_dims(q.string(), &cfg));
  int n8 = 0, n4 = 0;
  {
    auto ws = WeightSet::open(q.string(), nullptr);
    FB_REQUIRE(ws != nullptr);
    for (int L = 0; L < cfg.n_layers; ++L) {
      const int b = stored_affine_bits(
          *ws, "layers." + std::to_string(L) + ".attention.to_q", 64);
      n8 += b == 8;
      n4 += b == 4;
    }
  }
  std::printf("[fibo2_dit] mixed pack: %d main blocks w8, %d w4\n", n8, n4);
  FB_REQUIRE(n8 > 0 && n4 > 0);
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  int relaid = 0, promoted = 0, f8 = 0, f4 = 0;
  bool fast_on = false;
  auto run = [&](int tier, bool stream, bool slots, bool grow,
                 bool hold = false) {
    std::unique_ptr<Env> ns;
    if (!slots) { ns = std::make_unique<Env>("VPIPE_FIBO2_NO_SLOTS", "1"); }
    MetalFibo2Transformer::Config c = cfg;
    c.i8_gemm = tier >= 0;
    c.i8_tier = tier < 0 ? 0 : tier;
    c.i8_hold_w4 = hold;
    std::vector<float> out;
    auto m = MetalFibo2Transformer::load(q.string(), mc, c, stream);
    if (m == nullptr) { return out; }
    if (tier > 0 && !stream) {
      fast_on = m->fast_tier();
      f8 = m->fast_weights(8);
      f4 = m->fast_weights(4);
    }
    m->set_residency_reserve(grow ? test_residency_reserve(8ull << 30) : 0);
    m->set_residency_schedule(4);
    std::string err;
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = run_(m.get(), mc, G.sc[0], &err);
    }
    if (grow) { promoted = m->resident_block_count(); }
    if (stream && slots && !grow) { relaid = m->slot_relayouts(); }
    return out;
  };
  std::vector<int> tiers = {-1};
  if (mc->supports_matrix_cores()) { tiers.insert(tiers.end(), {0, 1, 2}); }
  for (int tier : tiers) {
    const auto pre = run(tier, false, true, false);
    const auto str = run(tier, true, true, false);
    const auto flat = run(tier, true, false, false);
    const auto grown = run(tier, true, true, true);
    FB_REQUIRE(!pre.empty() && !str.empty() && !flat.empty() &&
               !grown.empty());
    if (tier > 0) {
      // BOTH WIDTHS take the fast / token layout -- a w8 block converted
      // once, a w4 one per GEMM -- and the w4 layout HELD instead is the
      // same values, so the same bytes.
      const int pf8 = f8, pf4 = f4;
      const auto held = run(tier, false, true, false, /*hold=*/true);
      std::printf("[fibo2_dit] mixed pack, tier %d: %d w8 and %d w4 "
                  "projections in its layout; held vs per call %g\n", tier,
                  pf8, pf4, worst_diff_(pre, held));
      EXPECT_TRUE(fast_on && pf8 > 0 && pf4 > 0);
      EXPECT_TRUE(!held.empty() && worst_diff_(pre, held) == 0.0);
    }
    const double r = rel_l2_(pre, G.sc[0].out);
    std::printf("[fibo2_dit] mixed pack, i8 tier %d: rel-L2 vs the "
                "reference %.5f; streamed %g, no slots %g, promoted (%d "
                "blocks) %g; %d slot rebuilds\n", tier, r,
                worst_diff_(pre, str), worst_diff_(pre, flat), promoted,
                worst_diff_(pre, grown), relaid);
    EXPECT_TRUE(r < 0.3);
    EXPECT_TRUE(worst_diff_(pre, str) == 0.0);
    EXPECT_TRUE(worst_diff_(pre, flat) == 0.0);
    EXPECT_TRUE(worst_diff_(pre, grown) == 0.0);
    EXPECT_TRUE(relaid > 0);
    EXPECT_TRUE(promoted > 0);
  }
  std::error_code ec;
  fs::remove_all(q, ec);
}
