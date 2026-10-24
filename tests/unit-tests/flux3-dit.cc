// FLUX.3 Image Klein's DiT and its sampler arithmetic.
//
// Things about this model that are SILENT when wrong, and so are pinned
// against the reference rather than shape-checked:
//
//   * the bf16 TIMESTEP: t rounded to bf16, then 1000 t rounded again
//     -- 0.75 reaches the sinusoid as 752 (a random model moves ~0.1
//     on the difference);
//   * three token MODES with their own 5-block stacks, attending only
//     within themselves, before 28 joint blocks;
//   * modulation SHARED per mode across a whole stack;
//   * a PARALLEL block (one norm, one gate over attn + mlp);
//   * the flow map's target-time embedding added to every mode;
//   * fractional RoPE (image h, w at half their grid index; reference n
//     at l = -(1 + n)).
//
// Env: VPIPE_FLUX3_GOLDEN (tools/dump_flux3_golden.py) for the numeric
// tiers; VPIPE_FLUX3_TEST_MODEL_PATH (the repo root) for the real
// checkpoint. Each tier skips vacuously without its variable -- check
// that the tests actually RAN (they print what they measured).
// VPIPE_FLUX3_STREAM=1 streams the real DiT's blocks (needed on 16 GB).

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/flux3/metal-flux3-transformer.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/weight-set.h"

#include <algorithm>
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
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

// minitest's ASSERT_TRUE does NOT abort; `cond` is evaluated ONCE.
#define F3_REQUIRE(cond)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
env_(const char* k)
{
  const char* g = std::getenv(k);
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

bool
read_json_(const fs::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  try {
    *out = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  return out->is_object();
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

SharedBuffer
to_bf16_(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
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

std::vector<float>
from_bf16_(const SharedBuffer& b, std::size_t n)
{
  std::vector<float> v;
  if (b.byte_size() < n * 2) { return v; }
  v.resize(n);
  const auto* d = static_cast<const std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { v[i] = bf16_(d[i]); }
  return v;
}

}  // namespace

// ---- the sampler arithmetic, which needs no checkpoint ---------------

TEST(flux3_sampler, the_timestep_is_quantized_twice)
{
  // 0.75 is exact in bf16, and 750 is not: between 512 and 1024 bf16
  // steps by 4, and 750 ties between 748 and 752 -- round-half-even
  // takes 752. The reference's sinusoid sees 752.
  EXPECT_TRUE(flux3::quantized_time(0.75f) == 752.0f);
  EXPECT_TRUE(flux3::quantized_time(0.0f) == 0.0f);
  EXPECT_TRUE(flux3::quantized_time(1.0f) == 1000.0f);
  // 0.6 is NOT exact in bf16: 0.6015625, then 601.5625 -> 600.
  EXPECT_TRUE(flux3::quantized_time(0.6f) == 600.0f);
  // The embedding is [cos | sin] of that, each rounded to bf16.
  const std::vector<float> e0 = flux3::time_embedding(0.0f);
  EXPECT_TRUE(e0.size() == 256);
  EXPECT_TRUE(e0[0] == 1.0f && e0[127] == 1.0f);
  EXPECT_TRUE(e0[128] == 0.0f && e0[255] == 0.0f);
  const std::vector<float> e = flux3::time_embedding(0.75f);
  // cos(752), not cos(750): within bf16's rounding of the first and far
  // outside it of the second.
  EXPECT_TRUE(std::fabs(e[0] - std::cos(752.0f)) < 4e-3f);
  EXPECT_TRUE(std::fabs(e[0] - std::cos(750.0f)) > 0.1f);
}

TEST(flux3_sampler, the_schedule_is_the_reference_one)
{
  const std::vector<float> s = flux3::schedule(16, 6.93f);
  F3_REQUIRE(s.size() == 17);
  EXPECT_TRUE(s.front() == 1.0f && s.back() == 0.0f);
  for (std::size_t i = 1; i < s.size(); ++i) {
    EXPECT_TRUE(s[i] < s[i - 1]);
  }
  // t = a s / (1 + (a - 1) s) at s = 0.5: 6.93 / 7.93.
  EXPECT_TRUE(std::fabs(s[8] - 6.93f * 0.5f / (1.0f + 5.93f * 0.5f)) <
              1e-6f);
  // Against the dumped reference, to the bit.
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (g.empty()) { return; }
  FlexData man;
  if (!read_json_(fs::path(g) / "e2e.json", &man)) { return; }
  const FlexData sv = man.as_object().at("schedule");
  F3_REQUIRE(sv.is_array());
  auto arr = sv.as_array();
  F3_REQUIRE(arr.size() == s.size());
  int exact = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if ((float)arr.at(i).as_real(-1.0) == s[i]) { ++exact; }
  }
  std::printf("[flux3_sampler] schedule nodes bit-identical to the "
              "reference: %d / %zu\n", exact, s.size());
  EXPECT_TRUE(exact == (int)s.size());
}

// ---- the small model -------------------------------------------------

namespace {

struct Small {
  bool ok = false;
  fs::path dir;
  std::string tag;
  int gh = 0, gw = 0, txt_rows = 0, txt_w = 0, hidden = 0, heads = 0;
  int depth = 0, single = 0;
  float t = 0.0f, t_next = 0.0f;
  std::vector<std::pair<int, int>> refs;
  std::vector<float> img, cond, txt, out_f32, out_bf16;
  double bar = 0.0;
};

Small
load_small_(const char* tag = "small")
{
  Small S;
  S.tag = tag;
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (g.empty()) { return S; }
  S.dir = fs::path(g);
  FlexData man;
  if (!read_json_(S.dir / (S.tag + ".json"), &man)) { return S; }
  auto o = man.as_object();
  auto geti = [&](const char* k) {
    return o.contains(k) ? (int)o.at(k).as_int(0) : 0;
  };
  S.hidden = geti("hidden");
  S.heads = geti("heads");
  S.depth = geti("depth");
  S.single = geti("single");
  S.txt_rows = geti("text_rows");
  S.txt_w = geti("text_width");
  S.t = (float)o.at("t").as_real(0.0);
  S.t_next = (float)o.at("t_next").as_real(0.0);
  S.bar = o.at("bf16_vs_f32").as_real(0.0);
  {
    const FlexData gr = o.at("grid");
    auto a = gr.as_array();
    S.gh = (int)a.at(0).as_int(0);
    S.gw = (int)a.at(1).as_int(0);
  }
  {
    const FlexData rf = o.at("refs");
    auto a = rf.as_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
      const FlexData one = a.at(i);
      auto p = one.as_array();
      S.refs.emplace_back((int)p.at(0).as_int(0), (int)p.at(1).as_int(0));
    }
  }
  S.img = read_f32_(S.dir / (S.tag + "_img.f32"));
  S.cond = read_f32_(S.dir / (S.tag + "_cond.f32"));
  S.txt = read_f32_(S.dir / (S.tag + "_txt.f32"));
  S.out_f32 = read_f32_(S.dir / (S.tag + "_out_f32.f32"));
  S.out_bf16 = read_f32_(S.dir / (S.tag + "_out_bf16.f32"));
  S.ok = !S.img.empty() && !S.txt.empty() && !S.out_f32.empty();
  return S;
}

std::vector<float>
run_small_(MetalFlux3Transformer* m, MetalCompute* mc, const Small& S,
           bool with_refs, std::string* err)
{
  SharedBuffer lat = to_bf16_(mc, S.img);
  SharedBuffer txt = to_bf16_(mc, S.txt);
  std::vector<SharedBuffer> rb;
  MetalFlux3Transformer::Request req;
  req.latents = &lat;
  req.grid_h = S.gh;
  req.grid_w = S.gw;
  req.text = &txt;
  req.text_rows = S.txt_rows;
  req.t = S.t;
  req.t_next = S.t_next;
  if (with_refs) {
    std::size_t at = 0;
    rb.reserve(S.refs.size());
    for (const auto& r : S.refs) {
      const std::size_t n = (std::size_t)r.first * r.second * 128;
      rb.push_back(to_bf16_(mc, std::vector<float>(
          S.cond.begin() + (long)at, S.cond.begin() + (long)(at + n))));
      at += n;
    }
    for (std::size_t i = 0; i < S.refs.size(); ++i) {
      req.refs.push_back({&rb[i], S.refs[i].first, S.refs[i].second});
    }
  }
  SharedBuffer v = m->forward(req, err);
  if (v.empty()) { return {}; }
  return from_bf16_(v, (std::size_t)S.gh * S.gw * 128);
}

}  // namespace

TEST(flux3_dit, read_dims_reads_the_generated_config)
{
  const Small S = load_small_();
  if (!S.ok) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(
      (S.dir / S.tag).string(), &cfg));
  EXPECT_TRUE(cfg.hidden == S.hidden);
  EXPECT_TRUE(cfg.n_heads == S.heads);
  EXPECT_TRUE(cfg.n_mode == S.depth);
  EXPECT_TRUE(cfg.n_single == S.single);
  EXPECT_TRUE(cfg.joint_dim == S.txt_w);
  EXPECT_TRUE(cfg.mlp_hidden == 3 * S.hidden);
  EXPECT_TRUE(cfg.flow_map);
  EXPECT_TRUE(cfg.rope_theta == 10000.0f);
}

TEST(flux3_dit, read_dims_refuses_what_it_cannot_run)
{
  const fs::path d = fs::temp_directory_path() / "vpipe_flux3_cfg_test";
  std::error_code ec;
  fs::create_directories(d, ec);
  auto put = [&](const char* json) {
    std::ofstream o(d / "config.json");
    o << json;
  };
  MetalFlux3Transformer::Config cfg;
  put(R"({"_class_name":"Flux2Transformer2DModel"})");
  EXPECT_FALSE(MetalFlux3Transformer::Config::read_dims(d.string(), &cfg));
  // Three rotary axes: a different position scheme.
  put(R"({"_class_name":"Flux3Transformer2DModel",
          "axes_dims_rope":[32,48,48]})");
  EXPECT_FALSE(MetalFlux3Transformer::Config::read_dims(d.string(), &cfg));
  // A head dim the bd128 attention cannot take.
  put(R"({"_class_name":"Flux3Transformer2DModel","attention_head_dim":64,
          "num_attention_heads":24,"axes_dims_rope":[16,16,16,16]})");
  EXPECT_FALSE(MetalFlux3Transformer::Config::read_dims(d.string(), &cfg));
  // And the published one reads.
  put(R"({"_class_name":"Flux3Transformer2DModel","attention_head_dim":128,
          "axes_dims_rope":[32,32,32,32],"cond_in_channels":128,
          "eps":1e-06,"flow_map":true,"in_channels":128,
          "joint_attention_dim":20480,"mlp_ratio":3.0,
          "num_attention_heads":24,"num_layers":5,"num_single_layers":28,
          "rope_theta":10000,"timestep_channels":256})");
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(d.string(), &cfg));
  EXPECT_TRUE(cfg.hidden == 3072 && cfg.mlp_hidden == 9216);
  EXPECT_TRUE(cfg.n_mode == 5 && cfg.n_single == 28);
  fs::remove_all(d, ec);
}

TEST(flux3_dit, matches_reference_on_a_small_model)
{
  const Small S = load_small_();
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(m != nullptr);
  // ENGAGEMENT BEFORE ACCURACY: gemm_mma_ declines under 64 rows, and
  // the image mode's own blocks run at grid^2 rows.
  if (mc->supports_matrix_cores()) {
    EXPECT_TRUE(m->uses_mma2());
    EXPECT_TRUE(m->uses_attn_nax());
    EXPECT_TRUE(S.gh * S.gw >= 64);
  }
  std::string ferr;
  const std::vector<float> got = run_small_(m.get(), mc, S, true, &ferr);
  if (got.empty()) {
    std::printf("[flux3_dit] forward failed: %s\n", ferr.c_str());
  }
  F3_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, S.out_f32);
  std::printf("[flux3_dit] small model: velocity rel-L2 vs f32 = %.6f "
              "(reference bf16 %.6f), %d+%d+%d per-mode + %d joint, "
              "%d text + %d image + %zu refs\n",
              rel, S.bar, S.depth, S.depth, S.depth, S.single, S.txt_rows,
              S.gh * S.gw, S.refs.size());
  // At or below twice the reference's own bf16 error: this is a random
  // model with no structure to average errors over.
  EXPECT_TRUE(rel < 2.0 * S.bar + 1e-3);
}

// A STREAMED stack must be bit-identical to a preloaded one, with and
// without the slot pair: a prefetch with no free destination surfaces
// only there, and as a GPU fault rather than a wrong answer.
TEST(flux3_dit, streamed_matches_preloaded)
{
  const Small S = load_small_();
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  auto pre = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(pre != nullptr);
  std::string e;
  const std::vector<float> a = run_small_(pre.get(), mc, S, true, &e);
  // Without references the reference-mode blocks are never read: the
  // stack must still run, and its answer must differ.
  const std::vector<float> a0 = run_small_(pre.get(), mc, S, false, &e);
  pre.reset();
  F3_REQUIRE(!a.empty() && !a0.empty());
  EXPECT_TRUE(rel_l2_(a0, a) > 1e-3);

  for (int slots = 1; slots >= 0; --slots) {
    if (slots == 0) { setenv("VPIPE_FLUX3_NO_SLOTS", "1", 1); }
    auto str = MetalFlux3Transformer::load(ckpt, mc, cfg, true);
    std::vector<float> b, b0;
    if (str != nullptr) {
      EXPECT_TRUE(str->streaming());
      b = run_small_(str.get(), mc, S, true, &e);
      b0 = run_small_(str.get(), mc, S, false, &e);
    }
    str.reset();
    if (slots == 0) { unsetenv("VPIPE_FLUX3_NO_SLOTS"); }
    F3_REQUIRE(b.size() == a.size() && b0.size() == a0.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
      worst = std::max(worst, (double)std::fabs(a0[i] - b0[i]));
    }
    std::printf("[flux3_dit] streamed (%s) vs preloaded: worst |diff| = "
                "%g\n", slots ? "slots" : "NO slots", worst);
    EXPECT_TRUE(worst == 0.0);
  }
}

// ---- the published checkpoint -----------------------------------------

namespace {

std::string
dit_dir_()
{
  const std::string m = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  if (m.empty()) { return m; }
  return (fs::path(m) / "transformer").string();
}

std::vector<float>
golden_text_(const fs::path& g, const std::string& key, int* rows)
{
  FlexData man;
  if (!read_json_(g / "text.json", &man)) { return {}; }
  const FlexData p = man.as_object().at("prompts");
  if (!p.as_object().contains(key)) { return {}; }
  *rows = (int)p.as_object().at(key).as_object().at("padded").as_int(0);
  return read_f32_(g / ("text_" + key + "_bf16.f32"));
}

}  // namespace

// A MIXED PACK (VPIPE_FLUX3_MIXED_DIT = the transformer/ of a
// model-quantize bits 4 + mixed run): whole joint blocks at w8, the rest
// at w4 -- and one slot pair streams the per-mode and joint blocks alike
// -- so a slot built for one width is asked to hold the other
// (BlockSlots::Ops::same_layout). The pack must really mix widths, and
// its four holds -- preloaded, streamed, no slots, promoted -- must be
// BYTE-IDENTICAL with i8 off and, on a matrix-core box, under the signed
// layout. Before the slots rebuilt such a block, preloaded and no-slots
// agreed and the two slot holds did not.
TEST(flux3_dit, a_mixed_pack_holds_agree)
{
  const std::string qd = env_("VPIPE_FLUX3_MIXED_DIT");
  if (qd.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(qd, &cfg));
  int n8 = 0, n4 = 0;
  {
    auto ws = genai::WeightSet::open(qd, nullptr);
    F3_REQUIRE(ws != nullptr);
    for (int L = 0; L < cfg.n_single; ++L) {
      const int b = genai::stored_affine_bits(
          *ws, "single_transformer_blocks." + std::to_string(L) +
                   ".attn.to_q", 64);
      n8 += b == 8;
      n4 += b == 4;
    }
  }
  std::printf("[flux3_dit] mixed pack: %d joint blocks w8, %d w4\n", n8,
              n4);
  F3_REQUIRE(n8 > 0 && n4 > 0);
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, TS = 80;
  std::vector<float> img((std::size_t)gh * gw * cfg.in_channels);
  std::vector<float> txt((std::size_t)TS * cfg.joint_dim);
  unsigned st = 20261018u;
  auto rnd = [&]() {
    st = st * 1664525u + 1013904223u;
    return ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f);
  };
  for (float& v : img) { v = rnd(); }
  for (float& v : txt) { v = rnd(); }
  bool engaged = true;
  int promoted = 0, relaid = 0;
  auto run = [&](bool i8, bool stream, bool slots, bool grow) {
    std::vector<float> out;
    std::unique_ptr<Env> ns;
    if (!slots) { ns = std::make_unique<Env>("VPIPE_FLUX3_NO_SLOTS", "1"); }
    MetalFlux3Transformer::Config c = cfg;
    c.i8_gemm = i8;
    auto m = MetalFlux3Transformer::load(qd, mc, c, stream);
    if (m == nullptr) { return out; }
    if (m->signed_layout() != i8) { engaged = false; }
    m->set_residency_reserve(grow ? (8ull << 30) : 0);
    if (grow) { m->set_residency_schedule(4); }
    SharedBuffer lat = to_bf16_(mc, img);
    SharedBuffer tb = to_bf16_(mc, txt);
    MetalFlux3Transformer::Request req;
    req.latents = &lat;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &tb;
    req.text_rows = TS;
    req.t = 0.75f;
    req.t_next = 0.5f;
    std::string e;
    SharedBuffer v;
    for (int it = 0; it < (grow ? 2 : 1); ++it) { v = m->forward(req, &e); }
    if (grow) { promoted = m->resident_block_count(); }
    if (stream && slots && !grow) { relaid = m->slot_relayouts(); }
    const std::size_t n = (std::size_t)gh * gw * cfg.in_channels;
    if (v.empty() || v.byte_size() < n * 2) { return out; }
    const auto* p = static_cast<const std::uint16_t*>(v.contents());
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)p[i] << 16;
      std::memcpy(&out[i], &u, 4);
    }
    return out;
  };
  std::vector<bool> arms = {false};
  if (mc->supports_matrix_cores()) { arms.push_back(true); }
  for (bool i8 : arms) {
    engaged = true;
    const auto pre = run(i8, false, true, false);
    const auto str = run(i8, true, true, false);
    const auto flat = run(i8, true, false, false);
    const auto grown = run(i8, true, true, true);
    F3_REQUIRE(!pre.empty());
    auto same = [&](const std::vector<float>& a) {
      return a.size() == pre.size() &&
             std::memcmp(a.data(), pre.data(), a.size() * 4) == 0;
    };
    std::printf("[flux3_dit] mixed pack, %s: streamed %s, no slots %s, "
                "promoted (%d blocks) %s%s\n",
                i8 ? "signed i8" : "no i8",
                same(str) ? "IDENTICAL" : "DIFFERENT",
                same(flat) ? "IDENTICAL" : "DIFFERENT", promoted,
                same(grown) ? "IDENTICAL" : "DIFFERENT",
                engaged ? "" : " -- NOT ENGAGED");
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(same(str));
    EXPECT_TRUE(same(flat));
    EXPECT_TRUE(promoted > 0 && same(grown));
    // The slots crossed widths and REBUILT. A slot that refused such a
    // block instead turns itself off, and its holds still agree.
    std::printf("[flux3_dit] ... %d streamed blocks built into their "
                "slot\n", relaid);
    EXPECT_TRUE(relaid > 0);
  }
}

// THE SIGNED INT8 LAYOUT on a real quantized pack (VPIPE_FLUX3_QUANT_DIT at
// a model-quantize'd DiT; i8_gemm on; M5 only), as flux2_smoke.signed_
// layout_matches_native: every arm -- preloaded, streamed through the
// slots, streamed without them, promoted -- BYTE-IDENTICAL, and within a
// gross-error bar of the route this family took before (dequant +
// requant, its groups unrotated;
// VPIPE_I8_SIGNED=0 VPIPE_I8_ACC_ROT=0) in distance from the dense (i8 off)
// forward. 1280 image rows; an
// 80-row text stream (the signed dequant into matmul2d in its own mode
// blocks) and an 8-row one (the steel tiles through the signed dequant).
TEST(flux3_dit, signed_layout_matches_native)
{
  const std::string qd = env_("VPIPE_FLUX3_QUANT_DIT");
  if (qd.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(qd, &cfg));
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40;
  for (int TS : {80, 8}) {
    std::vector<float> img((std::size_t)gh * gw * cfg.in_channels);
    std::vector<float> txt((std::size_t)TS * cfg.joint_dim);
    unsigned st = 20261016u;
    auto rnd = [&]() {
      st = st * 1664525u + 1013904223u;
      return ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f);
    };
    for (float& v : img) { v = rnd(); }
    for (float& v : txt) { v = rnd(); }
    bool engaged = true;
    auto run = [&](bool sgn, bool i8, bool stream, bool grow, bool slots) {
      std::vector<float> out;
      std::unique_ptr<Env> e1, e2, e3;
      if (!sgn) {
        // The route it replaced: dequant + requant, its groups unrotated.
        e1 = std::make_unique<Env>("VPIPE_I8_SIGNED", "0");
        e3 = std::make_unique<Env>("VPIPE_I8_ACC_ROT", "0");
      }
      if (!slots) { e2 = std::make_unique<Env>("VPIPE_FLUX3_NO_SLOTS", "1"); }
      MetalFlux3Transformer::Config c = cfg;
      c.i8_gemm = i8;
      auto m = MetalFlux3Transformer::load(qd, mc, c, stream);
      if (m == nullptr) { return out; }
      if (m->signed_layout() != (sgn && i8)) { engaged = false; }
      m->set_residency_reserve(grow ? (4ull << 30) : 0);
      if (grow) { m->set_residency_schedule(4); }
      SharedBuffer lat = to_bf16_(mc, img);
      SharedBuffer tb = to_bf16_(mc, txt);
      MetalFlux3Transformer::Request req;
      req.latents = &lat;
      req.grid_h = gh;
      req.grid_w = gw;
      req.text = &tb;
      req.text_rows = TS;
      req.t = 0.75f;
      req.t_next = 0.5f;
      std::string e;
      SharedBuffer v;
      for (int it = 0; it < (grow ? 2 : 1); ++it) { v = m->forward(req, &e); }
      if (grow && m->resident_block_count() == 0) { engaged = false; }
      const std::size_t n = (std::size_t)gh * gw * cfg.in_channels;
      if (v.empty() || v.byte_size() < n * 2) { return out; }
      const auto* p = static_cast<const std::uint16_t*>(v.contents());
      out.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = (std::uint32_t)p[i] << 16;
        std::memcpy(&out[i], &u, 4);
      }
      return out;
    };
    const auto dense = run(false, false, false, false, true);
    const auto nat = run(false, true, false, false, true);
    const auto pre = run(true, true, false, false, true);
    const auto str = run(true, true, true, false, true);
    const auto flat = run(true, true, true, false, false);
    const auto grown = run(true, true, true, true, true);
    EXPECT_TRUE(engaged);
    F3_REQUIRE(!pre.empty() && nat.size() == pre.size() &&
               dense.size() == pre.size());
    auto same = [&](const std::vector<float>& a) {
      return a.size() == pre.size() &&
             std::memcmp(a.data(), pre.data(), a.size() * 4) == 0;
    };
    const double rs = rel_l2_(pre, dense), rn = rel_l2_(nat, dense);
    std::printf("[flux3_dit] signed layout, txt %d: vs the dense forward "
                "signed %.3e, unsigned %.3e; streamed %s, no slots %s, "
                "promoted %s\n", TS, rs, rn,
                same(str) ? "IDENTICAL" : "DIFFERENT",
                same(flat) ? "IDENTICAL" : "DIFFERENT",
                same(grown) ? "IDENTICAL" : "DIFFERENT");
    EXPECT_TRUE(same(str));
    EXPECT_TRUE(same(flat));
    EXPECT_TRUE(same(grown));
    // A GROSS-ERROR guard, not a ranking: on random inputs this forward
    // is chaotic -- QI21's dense forward moved 4.8e-2 for a per-weight
    // change of 5e-5 (gemm_i8.signed_dequant_on_a_real_pack: both
    // dequants 7.3e-3 from the bf16 weights, equal to four digits) -- so
    // two equally accurate routes land a few percent apart either way.
    // gemm_i8.signed_layout_matches_oracle is the accuracy statement; a
    // misread layout is O(1) here.
    EXPECT_TRUE(rs <= 1.25 * rn);
  }
}

// THE SIGNED LAYOUT'S WORTH in the FLUX.3 forward, i8_gemm on, at 1024^2
// (4096 image rows + 80 text), preloaded. Arms: bf16
// (VPIPE_FLUX3_TEST_MODEL_PATH's transformer/), the quantized pack
// (VPIPE_FLUX3_QUANT_DIT) signed, and the pack on the route it took before
// (dequant + requant, VPIPE_I8_SIGNED=0). Each arm warms two forwards and
// keeps the median of three; five rounds, reversed every other one.
// VPIPE_FLUX3_SIGNED_BENCH=1 runs it.
TEST(flux3_dit, signed_layout_forward)
{
  const std::string qd = env_("VPIPE_FLUX3_QUANT_DIT");
  const std::string bdir = dit_dir_();
  if (qd.empty() || bdir.empty() || env_("VPIPE_FLUX3_SIGNED_BENCH").empty()) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(qd, &cfg));
  const int gh = 64, gw = 64, TS = 80, rounds = 5;
  const std::vector<float> img((std::size_t)gh * gw * cfg.in_channels,
                               0.25f);
  const std::vector<float> txt((std::size_t)TS * cfg.joint_dim, 0.25f);
  struct Arm { const char* name; std::string dir; bool sgn; };
  const Arm arms[] = {{"bf16", bdir, true},
                      {"w8 signed", qd, true},
                      {"w8 requant", qd, false}};
  double mean[3] = {0, 0, 0};
  for (int r = 0; r < rounds; ++r) {
    for (int k = 0; k < 3; ++k) {
      const int a = (r % 2 == 0) ? k : 2 - k;
      if (!arms[a].sgn) { ::setenv("VPIPE_I8_SIGNED", "0", 1); }
      MetalFlux3Transformer::Config c = cfg;
      c.i8_gemm = true;
      auto m = MetalFlux3Transformer::load(arms[a].dir, mc, c, false);
      ::unsetenv("VPIPE_I8_SIGNED");
      F3_REQUIRE(m != nullptr);
      SharedBuffer lat = to_bf16_(mc, img);
      SharedBuffer tb = to_bf16_(mc, txt);
      MetalFlux3Transformer::Request req;
      req.latents = &lat;
      req.grid_h = gh;
      req.grid_w = gw;
      req.text = &tb;
      req.text_rows = TS;
      req.t = 0.75f;
      req.t_next = 0.5f;
      std::string e;
      for (int w = 0; w < 2; ++w) { F3_REQUIRE(!m->forward(req, &e).empty()); }
      std::vector<double> t;
      for (int i = 0; i < 3; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        F3_REQUIRE(!m->forward(req, &e).empty());
        t.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(t.begin(), t.end());
      mean[a] += t[1] / rounds;
      std::printf("[flux3_dit] round %d %-10s %7.1f ms/forward (signed "
                  "layout %s)\n", r, arms[a].name, t[1],
                  m->signed_layout() ? "on" : "off");
    }
  }
  for (int a = 0; a < 3; ++a) {
    std::printf("[flux3_dit]   %-10s %7.1f ms (mean of %d)  %.3fx bf16\n",
                arms[a].name, mean[a], rounds, mean[0] / mean[a]);
  }
}

TEST(flux3_dit, real_forward_matches_reference)
{
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  const std::string dir = dit_dir_();
  if (g.empty() || dir.empty()) { return; }
  const fs::path gd(g);
  FlexData man;
  if (!read_json_(gd / "dit.json", &man)) { return; }
  auto o = man.as_object();
  int rows = 0;
  const std::vector<float> txt = golden_text_(
      gd, std::string(o.at("prompt").as_string("")), &rows);
  F3_REQUIRE(!txt.empty() && rows > 0);
  auto grid = [&](const char* k, int i) {
    const FlexData a = o.at(k);
    return (int)a.as_array().at((std::size_t)i).as_int(0);
  };
  const int gh = grid("grid", 0), gw = grid("grid", 1);
  const int rh = grid("ref_grid", 0), rw = grid("ref_grid", 1);
  const std::vector<float> img = read_f32_(gd / "dit_img.f32");
  const std::vector<float> cond = read_f32_(gd / "dit_cond.f32");
  const std::vector<float> want = read_f32_(gd / "dit_out_f32.f32");
  const std::vector<float> ref16 = read_f32_(gd / "dit_out_bf16.f32");
  F3_REQUIRE(!img.empty() && !cond.empty() && !want.empty());

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(dir, &cfg));
  const bool stream = !env_("VPIPE_FLUX3_STREAM").empty();
  auto m = MetalFlux3Transformer::load(dir, mc, cfg, stream);
  F3_REQUIRE(m != nullptr);
  SharedBuffer lat = to_bf16_(mc, img);
  SharedBuffer tb = to_bf16_(mc, txt);
  SharedBuffer cb = to_bf16_(mc, cond);
  MetalFlux3Transformer::Request req;
  req.latents = &lat;
  req.grid_h = gh;
  req.grid_w = gw;
  req.text = &tb;
  req.text_rows = rows;
  req.refs.push_back({&cb, rh, rw});
  req.t = (float)o.at("t").as_real(0.0);
  req.t_next = (float)o.at("t_next").as_real(0.0);
  std::string err;
  SharedBuffer v = m->forward(req, &err);
  if (v.empty()) { std::printf("[flux3_dit] %s\n", err.c_str()); }
  F3_REQUIRE(!v.empty());
  const std::vector<float> got = from_bf16_(v, (std::size_t)gh * gw * 128);
  const double rel = rel_l2_(got, want);
  const double bar = rel_l2_(ref16, want);
  const int qb = m->quant_bits();
  std::printf("[flux3_dit] real DiT (%s%s): velocity rel-L2 vs f32 = %.6f, "
              "the reference's own bf16 %.6f (%d text + %d image + %d ref "
              "rows, t %.4f -> %.4f)\n", stream ? "streamed" : "preloaded",
              qb > 0 ? (", w" + std::to_string(qb)).c_str() : "",
              rel, bar, rows, gh * gw, rh * rw, (double)req.t,
              (double)req.t_next);
  // ONE forward is a noisy measure on this model: a few late joint
  // blocks attend sharply, and there two bf16 implementations of the same
  // arithmetic land 0.7-1.6x apart from each other, step by step (see the
  // teacher-forced test below, which is the test of the forward that
  // averages that out). So this is a bound on gross error, not the bar.
  // A QUANTIZED pack (model-quantize) adds its own error on top, and is
  // held to tracking the reference rather than to the bf16 bar.
  // w4 is VISIBLY lossy on this model (0.11 on this forward; its 1024^2
  // picture 29.4 dB from the dense one, the composition kept), so its
  // bound only catches a broken pack.
  EXPECT_TRUE(rel <= bar * (qb == 4 ? 5.0 : qb == 8 ? 1.5 : 1.25));
}

// The 16-step flow map end to end from the reference's own noise and
// text. The truth keeps an f32 model and an f32 state; vpipe keeps an
// f32 state between bf16 forwards, which is the comparison that means
// something (the reference's bf16 run rounds its state every step).
namespace {

std::string
fmt_step_(int k, double rel, const std::vector<double>& bar, int i)
{
  char b[64];
  std::snprintf(b, sizeof(b), " %d:%.4f/%.4f", k, rel,
                i < (int)bar.size() ? bar[(std::size_t)i] : 0.0);
  return b;
}

// What one end-to-end run measured. `ran` false is a vacuous skip (a
// variable unset); `why` non-empty is a failure the test reports.
struct E2e {
  bool ran = false;
  std::string why;
  double rel = 0.0, bar = 0.0;
  // Teacher-forced: the mean over steps of vpipe's error and of the
  // reference's own bf16 error, and the worst step's ratio of the two.
  double mean = 0.0, bar_mean = 0.0, worst_ratio = 0.0;
  // The checkpoint's affine bits (0 = dense). A quantized pack is held to
  // tracking the reference, not to the bf16 bar.
  int quant_bits = 0;
};

// `teacher`: every step starts from the REFERENCE's f32 state and the
// velocity is compared with the one that state implies, (x_{k+1} -
// x_k) / dt -- which isolates the forward at each timestep from the
// trajectory's amplification of whatever came before.
E2e
run_e2e_(const char* tag, bool teacher = false)
{
  E2e r;
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  const std::string dir = dit_dir_();
  if (g.empty() || dir.empty()) { return r; }
  const fs::path gd(g);
  FlexData man;
  if (!read_json_(gd / (std::string(tag) + ".json"), &man)) { return r; }
  r.ran = true;
  auto o = man.as_object();
  int rows = 0;
  const std::vector<float> txt = golden_text_(
      gd, std::string(o.at("prompt").as_string("")), &rows);
  if (txt.empty()) { r.why = "no golden text"; return r; }
  const FlexData lat_a = o.at("latent");
  const int gh = (int)lat_a.as_array().at(0).as_int(0);
  const int gw = (int)lat_a.as_array().at(1).as_int(0);
  const int steps = (int)o.at("steps").as_int(16);
  const float alpha = (float)o.at("alpha").as_real(6.93);
  std::vector<float> x = read_f32_(gd / (std::string(tag) + "_noise.f32"));
  const std::vector<float> want =
      read_f32_(gd / (std::string(tag) + "_out_f32.f32"));
  const std::vector<float> ref16 =
      read_f32_(gd / (std::string(tag) + "_out_bf16.f32"));
  if (x.size() != (std::size_t)gh * gw * 128 || want.empty()) {
    r.why = "golden shapes";
    return r;
  }
  std::vector<float> rtok;
  int rh = 0, rw = 0;
  if (o.contains("ref_size")) {
    const FlexData rs = o.at("ref_size");
    rw = (int)rs.as_array().at(0).as_int(0) / 16;
    rh = (int)rs.as_array().at(1).as_int(0) / 16;
    rtok = read_f32_(gd / (std::string(tag) + "_ref_tokens.f32"));
    if (rtok.size() != (std::size_t)rh * rw * 128) {
      r.why = "reference token shape";
      return r;
    }
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { r.ran = false; return r; }
  MetalFlux3Transformer::Config cfg;
  if (!MetalFlux3Transformer::Config::read_dims(dir, &cfg)) {
    r.why = "read_dims";
    return r;
  }
  // VPIPE_FLUX3_STREAM streams the blocks, as a 16 GB box must: an
  // 11 GB preloaded DiT swaps there and the run never finishes.
  const bool stream = !env_("VPIPE_FLUX3_STREAM").empty();
  auto m = MetalFlux3Transformer::load(dir, mc, cfg, stream);
  if (m == nullptr) { r.why = "load"; return r; }
  r.quant_bits = m->quant_bits();
  SharedBuffer tb = to_bf16_(mc, txt);
  SharedBuffer rb;
  if (!rtok.empty()) { rb = to_bf16_(mc, rtok); }
  const std::vector<float> s = flux3::schedule(steps, alpha);
  const std::vector<float> all_f32 =
      read_f32_(gd / (std::string(tag) + "_steps_f32.f32"));
  std::vector<double> per_bar;
  if (o.contains("bf16_vs_f32_per_step")) {
    const FlexData pb = o.at("bf16_vs_f32_per_step");
    auto arr = pb.as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      per_bar.push_back(arr.at(i).as_real(0.0));
    }
  }
  std::string per_step;
  if (teacher && all_f32.size() < (std::size_t)steps * x.size()) {
    r.why = "teacher forcing needs the dumped steps";
    return r;
  }
  const std::vector<float> noise = x;
  auto truth = [&](int k) {
    // The reference's state BEFORE step k.
    if (k == 0) { return noise; }
    return std::vector<float>(
        all_f32.begin() + (long)((std::size_t)(k - 1) * x.size()),
        all_f32.begin() + (long)((std::size_t)k * x.size()));
  };
  double worst_tf = 0.0;
  std::vector<double> tbar;
  if (o.contains("bf16_teacher_velocity_per_step")) {
    const FlexData tb2 = o.at("bf16_teacher_velocity_per_step");
    auto arr = tb2.as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      tbar.push_back(arr.at(i).as_real(0.0));
    }
  }
  for (int k = 0; k < steps; ++k) {
    if (teacher) { x = truth(k); }
    SharedBuffer lat = to_bf16_(mc, x);
    MetalFlux3Transformer::Request req;
    req.latents = &lat;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &tb;
    req.text_rows = rows;
    if (!rtok.empty()) { req.refs.push_back({&rb, rh, rw}); }
    req.t = s[(std::size_t)k];
    req.t_next = s[(std::size_t)k + 1];
    std::string err;
    SharedBuffer v = m->forward(req, &err);
    if (v.empty()) { r.why = "forward: " + err; return r; }
    const std::vector<float> vel = from_bf16_(v, x.size());
    const float dt = s[(std::size_t)k + 1] - s[(std::size_t)k];
    if (teacher) {
      const std::vector<float> a = truth(k), b = truth(k + 1);
      std::vector<float> vt(a.size());
      for (std::size_t i = 0; i < a.size(); ++i) {
        vt[i] = (b[i] - a[i]) / dt;
      }
      const double e = rel_l2_(vel, vt);
      worst_tf = std::max(worst_tf, e);
      const double bk = k < (int)tbar.size() ? tbar[(std::size_t)k] : 0.0;
      r.mean += e / steps;
      r.bar_mean += bk / steps;
      if (bk > 0.0) { r.worst_ratio = std::max(r.worst_ratio, e / bk); }
      char cb[48];
      std::snprintf(cb, sizeof(cb), " %d:%.4f/%.4f", k, e, bk);
      per_step += cb;
      continue;
    }
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] += dt * vel[i]; }
    // EVERY step against the reference's, beside the reference's own
    // bf16 error at that step: a gap at step 0 is the forward, one that
    // only opens later is the trajectory.
    if (all_f32.size() >= (std::size_t)(k + 1) * x.size()) {
      const std::vector<float> want_k(
          all_f32.begin() + (long)((std::size_t)k * x.size()),
          all_f32.begin() + (long)((std::size_t)(k + 1) * x.size()));
      per_step += fmt_step_(k, rel_l2_(x, want_k), per_bar, k);
    }
  }
  if (teacher) {
    std::printf("[flux3_dit] %s teacher-forced velocity rel-L2 per step "
                "(vpipe / reference bf16):%s\n  mean %.4f / %.4f, worst "
                "step %.2fx its bar\n", tag, per_step.c_str(), r.mean,
                r.bar_mean, r.worst_ratio);
    r.rel = worst_tf;
    return r;
  }
  if (!per_step.empty()) {
    std::printf("[flux3_dit] %s per step (vpipe / reference bf16):%s\n",
                tag, per_step.c_str());
  }
  r.rel = rel_l2_(x, want);
  r.bar = rel_l2_(ref16, want);
  std::printf("[flux3_dit] %s: %d steps at %dx%d latent%s, rel-L2 vs the "
              "f32 reference = %.6f (the reference's own bf16 %.6f)\n", tag,
              steps, gh, gw, rtok.empty() ? "" : " + 1 reference", r.rel,
              r.bar);
  return r;
}

}  // namespace

// A QUANTIZED pack's trajectory leaves the reference's further -- a
// 16-step trajectory amplifies the w8 GEMMs' error 8x at 256^2, where the
// same pack's 1024^2 picture sits 48.7 dB from the dense one -- so it is
// held to staying on the reference's trajectory, not to the bf16 bar.
//
// The bf16 bar is WIDE because the endpoint is one draw of a chaotic
// sum. Kernel arms that are equally accurate PER STEP (teacher-forced
// means 0.0139-0.0143 t2i / 0.0156-0.0162 edit, the reference's own
// 0.0146 / 0.0165; matmul2d's tiles are correctly rounded more often
// than steel's) end 16 steps at 0.92-1.17x (t2i) and 0.79-1.26x (edit)
// of the reference's own bf16 error: M4 steel, M5 steel + NAX, M5
// matmul2d + steel attention, M5 matmul2d + NAX. 1.35x still fails the
// double-rounded QK norm + rope this test caught (edit 1.56x); the
// per-step test below is the tight one.
double
e2e_bound_(const E2e& r, double edit_factor)
{
  if (r.quant_bits == 0) { return r.bar * 1.35; }
  // w4's 256^2 edit trajectory ends 0.74 from the reference's: a sanity
  // bound (a broken pack is noise, at ~1.4), not a quality claim.
  if (r.quant_bits == 4) { return 1.0; }
  return r.bar * 4.0 * edit_factor;
}

TEST(flux3_dit, sixteen_steps_match_reference)
{
  const E2e r = run_e2e_("e2e");
  if (!r.ran) { return; }
  if (!r.why.empty()) { std::printf("[flux3_dit] %s\n", r.why.c_str()); }
  EXPECT_TRUE(r.why.empty());
  EXPECT_TRUE(r.rel <= e2e_bound_(r, 1.0));
}

TEST(flux3_dit, sixteen_step_edit_matches_reference)
{
  const E2e r = run_e2e_("e2e_edit");
  if (!r.ran) { return; }
  if (!r.why.empty()) { std::printf("[flux3_dit] %s\n", r.why.c_str()); }
  EXPECT_TRUE(r.why.empty());
  EXPECT_TRUE(r.rel <= e2e_bound_(r, 1.0));
}

// The forward at EVERY timestep the sampler visits, each from the
// reference's own state: the trajectory test above cannot tell a
// forward that is wrong at one t from a trajectory that amplifies.
TEST(flux3_dit, every_step_is_accurate_teacher_forced)
{
  for (const char* tag : {"e2e", "e2e_edit"}) {
    const E2e r = run_e2e_(tag, /*teacher=*/true);
    if (!r.ran) { continue; }
    if (!r.why.empty()) { std::printf("[flux3_dit] %s\n", r.why.c_str()); }
    EXPECT_TRUE(r.why.empty());
    // ON AVERAGE at the reference's own bf16 error, and no step wildly
    // off it. Single steps scatter 0.7-1.6x around the reference's (the
    // edit's step 3 is the worst, in every arithmetic variant tried):
    // that is the noise between two bf16 implementations at a timestep
    // where a few late blocks attend sharply, and a FORWARD bug would
    // show at every step.
    if (r.bar_mean > 0.0) {
      // A quantized pack's GEMMs add their own error at every step.
      const double k = r.quant_bits == 4 ? 10.0 : r.quant_bits == 8 ? 1.5
                                                                     : 1.1;
      EXPECT_TRUE(r.mean <= r.bar_mean * k);
      EXPECT_TRUE(r.worst_ratio <= 2.0 * k);
    }
  }
}

// Opt-in: write flux3::time_embedding for every node of the 16-step
// schedule to VPIPE_FLUX3_DUMP_TEMB, for a bitwise comparison against
// torch's timestep_embedding.
TEST(flux3_sampler, dump_time_embeddings)
{
  const std::string out = env_("VPIPE_FLUX3_DUMP_TEMB");
  if (out.empty()) { return; }
  const std::vector<float> s = flux3::schedule(16, 6.93f);
  std::ofstream o(out, std::ios::binary);
  for (float t : s) {
    const std::vector<float> e = flux3::time_embedding(t);
    o.write(reinterpret_cast<const char*>(e.data()),
            (std::streamsize)(e.size() * sizeof(float)));
  }
}

// Opt-in: ONE teacher-forced forward of an e2e golden's step
// (VPIPE_FLUX3_TRACE_OF, VPIPE_FLUX3_TRACE_STEP), for a per-block trace
// with VPIPE_FLUX3_TRACE set -- the comparison against the reference's
// own hooks is tools/dump_flux3_golden.py --stage trace.
TEST(flux3_dit, trace_one_step)
{
  const std::string of = env_("VPIPE_FLUX3_TRACE_OF");
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  const std::string dir = dit_dir_();
  if (of.empty() || g.empty() || dir.empty()) { return; }
  const int k = std::atoi(env_("VPIPE_FLUX3_TRACE_STEP").c_str());
  const fs::path gd(g);
  FlexData man;
  F3_REQUIRE(read_json_(gd / (of + ".json"), &man));
  auto o = man.as_object();
  int rows = 0;
  const std::vector<float> txt = golden_text_(
      gd, std::string(o.at("prompt").as_string("")), &rows);
  const FlexData la = o.at("latent");
  const int gh = (int)la.as_array().at(0).as_int(0);
  const int gw = (int)la.as_array().at(1).as_int(0);
  const std::size_t N = (std::size_t)gh * gw * 128;
  std::vector<float> x = read_f32_(gd / (of + "_noise.f32"));
  if (k > 0) {
    const std::vector<float> all = read_f32_(gd / (of + "_steps_f32.f32"));
    F3_REQUIRE(all.size() >= (std::size_t)k * N);
    x.assign(all.begin() + (long)((std::size_t)(k - 1) * N),
             all.begin() + (long)((std::size_t)k * N));
  }
  std::vector<float> rtok;
  int rh = 0, rw = 0;
  if (o.contains("ref_size")) {
    const FlexData rs = o.at("ref_size");
    rw = (int)rs.as_array().at(0).as_int(0) / 16;
    rh = (int)rs.as_array().at(1).as_int(0) / 16;
    rtok = read_f32_(gd / (of + "_ref_tokens.f32"));
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(dir, &cfg));
  // VPIPE_FLUX3_STREAM streams the blocks, as a 16 GB box must: an
  // 11 GB preloaded DiT swaps there and the run never finishes.
  const bool stream = !env_("VPIPE_FLUX3_STREAM").empty();
  auto m = MetalFlux3Transformer::load(dir, mc, cfg, stream);
  F3_REQUIRE(m != nullptr);
  SharedBuffer tb = to_bf16_(mc, txt);
  SharedBuffer rb = rtok.empty() ? SharedBuffer{} : to_bf16_(mc, rtok);
  SharedBuffer lat = to_bf16_(mc, x);
  const std::vector<float> s = flux3::schedule(16, 6.93f);
  MetalFlux3Transformer::Request req;
  req.latents = &lat;
  req.grid_h = gh;
  req.grid_w = gw;
  req.text = &tb;
  req.text_rows = rows;
  if (!rtok.empty()) { req.refs.push_back({&rb, rh, rw}); }
  req.t = s[(std::size_t)k];
  req.t_next = s[(std::size_t)k + 1];
  std::string err;
  SharedBuffer v = m->forward(req, &err);
  F3_REQUIRE(!v.empty());
  const std::string tdir = env_("VPIPE_FLUX3_TRACE");
  if (!tdir.empty()) {
    const std::vector<float> vel = from_bf16_(v, N);
    std::ofstream out(fs::path(tdir) / "vel.f32", std::ios::binary);
    out.write(reinterpret_cast<const char*>(vel.data()),
              (std::streamsize)(vel.size() * sizeof(float)));
  }
}

// ---- the accel tiers, the adapter, the stop and the meter -------------
//
// On the WIDE golden (tools/dump_flux3_golden.py --stage small --tag wide
// --heads 8 --grid 32 --depth 2 --single 3 --lora-rank 8): hidden 1024
// clears i8's K >= 1024 floor, and 1024 image rows clear i8's M floor
// and Sol's 1024 rows / 16 key blocks -- a tier tested below its floor
// declines and the test passes having exercised nothing.

TEST(flux3_dit, wide_model_matches_reference)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  EXPECT_TRUE(cfg.hidden >= 1024 && S.gh * S.gw >= 1024);
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(m != nullptr);
  std::string e;
  const std::vector<float> got = run_small_(m.get(), mc, S, true, &e);
  F3_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, S.out_f32);
  std::printf("[flux3_dit] wide model: rel-L2 vs f32 %.6f (reference bf16 "
              "%.6f)\n", rel, S.bar);
  EXPECT_TRUE(rel < 1.5 * S.bar);
}

TEST(flux3_dit, i8_gemm_engages_and_stays_accurate)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  // THE FLOORS i8_gemm's accepts() holds a GEMM to -- K >= 1024 (the
  // model's WIDTH) and M >= 1024 rows -- or the test passes having
  // routed nothing. The wide golden clears both: hidden 1024, and its
  // image segment alone is 32 x 32 rows.
  F3_REQUIRE(cfg.hidden >= 1024);
  F3_REQUIRE(S.gh * S.gw >= 1024);
  auto base = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(base != nullptr);
  std::string e;
  const std::vector<float> off = run_small_(base.get(), mc, S, true, &e);
  base.reset();
  cfg.i8_gemm = true;
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(m != nullptr);
  const bool engaged = m->uses_i8_gemm();
  const std::vector<float> on = run_small_(m.get(), mc, S, true, &e);
  m.reset();
  if (!engaged) {
    std::printf("[flux3_dit] i8_gemm unavailable on this box; skipped\n");
    return;
  }
  F3_REQUIRE(!off.empty() && on.size() == off.size());
  const double d = rel_l2_(on, off), r = rel_l2_(on, S.out_f32);
  std::printf("[flux3_dit] i8: rel-L2 vs bf16 %.6f, vs reference %.6f\n",
              d, r);
  EXPECT_TRUE(std::isfinite(d) && std::isfinite(r));
  EXPECT_TRUE(d > 0.0);
  EXPECT_TRUE(r < 0.12);
}

TEST(flux3_dit, sol_attention_engages_and_stays_close)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  auto base = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(base != nullptr);
  // Sol's own floor is 16 KEY BLOCKS (64 keys each by default): the
  // wide golden's 1024 image rows alone are 16, before the text and the
  // reference join them in the joint blocks Sol routes.
  F3_REQUIRE(S.gh * S.gw >= 16 * 64);
  std::string e;
  const std::vector<float> off = run_small_(base.get(), mc, S, true, &e);
  base.reset();
  cfg.sol.enabled = true;
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(m != nullptr);
  const bool engaged = m->uses_sol_attn();
  const std::vector<float> on = run_small_(m.get(), mc, S, true, &e);
  const std::uint64_t routed = m->gpu_flops().flushed();
  m.reset();
  if (!engaged) {
    std::printf("[flux3_dit] sol_attn did not load here; skipped\n");
    return;
  }
  F3_REQUIRE(!off.empty() && on.size() == off.size());
  const double d = rel_l2_(on, off);
  std::printf("[flux3_dit] sol (tau %.2f): rel-L2 vs dense %.6f, %llu "
              "FLOP credited\n", (double)cfg.sol.tau, d,
              (unsigned long long)routed);
  EXPECT_TRUE(std::isfinite(d));
  EXPECT_TRUE(d > 0.0);
  EXPECT_TRUE(d < 0.6);
}

// SAGE rides the NAX attention only (it quantizes q/k to int8 inside the
// matrix-core kernel), so it engages on an M5 and declines elsewhere --
// said, and skipped. A segment must reach kSageMinSegment rows; the wide
// golden's image segment is 1024.
//
// AND ITS SCRATCH IS BUILT ONCE. The image mode's stack, the
// references', then the joint stack attend at different row counts, and
// each change used to rebuild Sage's scratch (a private free and
// allocation, two or three a forward). Each forward now reserves the
// joint's: two forwards build it once, and the answer is the same bits
// as without the reservation (VPIPE_FLUX3_SAGE_NO_RESERVE), which still
// rebuilds per row count.
TEST(flux3_dit, sage_attention_engages_and_stays_close)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  F3_REQUIRE(S.gh * S.gw >= 512);   // kSageMinSegment
  auto base = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(base != nullptr);
  std::string e;
  const std::vector<float> off = run_small_(base.get(), mc, S, true, &e);
  base.reset();
  cfg.sage.enabled = true;
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(m != nullptr);
  const bool engaged = m->uses_sage_attn();
  const std::vector<float> on = run_small_(m.get(), mc, S, true, &e);
  const std::vector<float> on2 = run_small_(m.get(), mc, S, true, &e);
  const int builds = m->sage_scratch_builds();
  m.reset();
  if (!engaged) {
    std::printf("[flux3_dit] sage_attn needs the NAX attention (an M5); "
                "skipped\n");
    return;
  }
  ::setenv("VPIPE_FLUX3_SAGE_NO_RESERVE", "1", 1);
  auto u = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  ::unsetenv("VPIPE_FLUX3_SAGE_NO_RESERVE");
  F3_REQUIRE(u != nullptr);
  const std::vector<float> un = run_small_(u.get(), mc, S, true, &e);
  const int ubuilds = u->sage_scratch_builds();
  u.reset();
  F3_REQUIRE(on2.size() == on.size() && un.size() == on.size());
  const bool same2 =
      std::memcmp(on.data(), on2.data(), on.size() * sizeof(float)) == 0;
  const bool same_u =
      std::memcmp(on.data(), un.data(), on.size() * sizeof(float)) == 0;
  std::printf("[flux3_dit] sage scratch: %d build(s) over two forwards "
              "reserved, %d in one without; second forward %s, without "
              "the reservation %s\n", builds, ubuilds,
              same2 ? "bit-identical" : "DIFFERS",
              same_u ? "bit-identical" : "DIFFERS");
  EXPECT_TRUE(builds == 1);
  EXPECT_TRUE(ubuilds >= 2);
  EXPECT_TRUE(same2 && same_u);
  F3_REQUIRE(!off.empty() && on.size() == off.size());
  const double d = rel_l2_(on, off), r = rel_l2_(on, S.out_f32);
  std::printf("[flux3_dit] sage: rel-L2 vs dense %.6f, vs reference "
              "%.6f (dense vs reference %.6f)\n", d, r,
              rel_l2_(off, S.out_f32));
  EXPECT_TRUE(std::isfinite(d) && std::isfinite(r));
  // Engaged means MOVED: an unwired tier is bit-identical to dense.
  EXPECT_TRUE(d > 0.0);
  EXPECT_TRUE(r < 3.0 * S.bar);
}

// Checked against the reference's OWN dense arithmetic: the same model
// with the adapter's factors fused into its weights, run by BFL's DiT.
// What it does NOT establish is that a PUBLISHED FLUX.3 adapter binds --
// none exists yet, so the module names are a reading of the diffusers
// export convention (the base tensor's path without ".weight").
TEST(flux3_dit, runtime_lora_matches_a_fused_reference)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  const fs::path lp = S.dir / (S.tag + "_lora.safetensors");
  const std::vector<float> want = read_f32_(S.dir / (S.tag + "_lora_out.f32"));
  std::error_code ec;
  if (!fs::exists(lp, ec) || want.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  auto plain = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(plain != nullptr);
  std::string e;
  const std::vector<float> base = run_small_(plain.get(), mc, S, true, &e);
  plain.reset();
  F3_REQUIRE(!base.empty());

  std::vector<MetalFlux3Transformer::LoraSpec> specs;
  specs.push_back({lp.string(), 1.0f});
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false, specs);
  F3_REQUIRE(m != nullptr);
  // ENGAGEMENT FIRST: 6 projections per block, every block.
  const int bound = m->lora_modules(0);
  const int expect = 6 * (3 * cfg.n_mode + cfg.n_single);
  std::printf("[flux3_dit] lora bound %d modules (%d expected)\n", bound,
              expect);
  F3_REQUIRE(bound == expect);
  const std::vector<float> got = run_small_(m.get(), mc, S, true, &e);
  F3_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, want);
  const double moved = rel_l2_(base, want);
  std::printf("[flux3_dit] lora: rel-L2 vs the fused reference %.6f (the "
              "adapter moves the base by %.6f)\n", rel, moved);
  EXPECT_TRUE(moved > 0.05);
  EXPECT_TRUE(rel < 3.0 * S.bar);
  // THE STRENGTH IS LIVE, and exactly 0 encodes nothing.
  m->set_lora_scale(0, 0.0f);
  const std::vector<float> off = run_small_(m.get(), mc, S, true, &e);
  F3_REQUIRE(!off.empty());
  const double rel_off = rel_l2_(off, base);
  std::printf("[flux3_dit] lora at scale 0: rel-L2 vs base %.6g\n",
              rel_off);
  EXPECT_TRUE(rel_off == 0.0);
}

// A stop must come back as a STOP -- empty with no reason -- and before
// the rest of the stack has run. Both arms, because they commit
// differently: a streamed block waits on its own, a held stack drains
// every few blocks only because a stop callback is installed.
//
// This one trips on the Nth CHECK, which proves the checks exist and
// what a stop returns -- not that a held stack SEES one: its checks
// all run while it encodes, in microseconds, before the GPU starts.
// The timed test below is the one that can tell.
TEST(flux3_dit, a_stop_is_not_a_failure_and_lands_inside_a_forward)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  for (int arm = 0; arm < 2; ++arm) {
    const bool stream = arm == 1;
    auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, stream);
    F3_REQUIRE(m != nullptr);
    if (stream) {
      m->set_residency_reserve(0);
      m->set_residency_schedule(2);
    }
    std::string e;
    const std::vector<float> full = run_small_(m.get(), mc, S, true, &e);
    F3_REQUIRE(!full.empty());
    const int total = MetalFlux3Transformer::blocks_run(cfg, true);
    int seen = 0;
    m->set_stream_stop([&]() { return ++seen > total / 2; });
    std::string serr;
    const std::vector<float> stopped =
        run_small_(m.get(), mc, S, true, &serr);
    m->set_stream_stop({});
    std::printf("[flux3_dit] stop (%s): %d of %d checks before it was "
                "honoured, result %s, reason '%s'\n",
                stream ? "streamed" : "preloaded", seen, total,
                stopped.empty() ? "empty" : "NOT empty", serr.c_str());
    EXPECT_TRUE(stopped.empty());
    EXPECT_TRUE(serr.empty());
    EXPECT_TRUE(seen <= total / 2 + 1);
  }
}

// What a forward encodes, COUNTED at encode and CREDITED at retire, must
// equal what its shapes say -- every GEMM and every attention, no more.
// A STOP IS A TIMER, NOT A COUNT. Set from another thread partway
// through a measured forward, it must still land INSIDE that forward:
// a held stack encodes every block before the GPU runs any, so without
// its drains every kStopDrainBlocks the flag would only be read once
// the whole forward was done -- and the result would come back whole.
// The STREAMED arm needs the same drains: by the timed forward its
// residency has promoted every block, so it runs held too. MEASURED
// with the drains removed: both arms came back whole.
TEST(flux3_dit, a_timed_stop_lands_inside_a_forward)
{
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  for (int arm = 0; arm < 2; ++arm) {
    const bool stream = arm == 1;
    auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, stream);
    F3_REQUIRE(m != nullptr);
    if (stream) {
      m->set_residency_reserve(0);
      m->set_residency_schedule(2);
    }
    // The forward's own length, warm and WITH the callback installed
    // (the held stack's drains are part of what a stoppable run costs).
    std::atomic<bool> flag{false};
    m->set_stream_stop([&]() { return flag.load(); });
    std::string e;
    double full_ms = 1e30;
    for (int i = 0; i < 4; ++i) {
      const auto a = clk::now();
      F3_REQUIRE(!run_small_(m.get(), mc, S, true, &e).empty());
      if (i > 0) { full_ms = std::min(full_ms, ms(a, clk::now())); }
    }
    // A third of the way in: past the encode, well before the last
    // drain a held stack takes.
    const double at_ms = full_ms / 3.0;
    clk::time_point set_at{};
    const auto t0 = clk::now();
    std::thread timer([&]() {
      std::this_thread::sleep_for(
          std::chrono::microseconds((long long)(at_ms * 1000.0)));
      set_at = clk::now();
      flag.store(true);
    });
    std::string serr;
    const std::vector<float> stopped =
        run_small_(m.get(), mc, S, true, &serr);
    const auto t1 = clk::now();
    timer.join();
    m->set_stream_stop({});
    std::printf("[flux3_dit] timed stop (%s): forward %.2f ms, flag set at "
                "%.2f ms, returned %.2f ms after it, result %s, reason "
                "'%s'\n", stream ? "streamed" : "preloaded", full_ms,
                ms(t0, set_at), ms(set_at, t1),
                stopped.empty() ? "empty" : "NOT empty", serr.c_str());
    EXPECT_TRUE(stopped.empty());
    EXPECT_TRUE(serr.empty());
  }
}

TEST(flux3_dit, forward_credits_its_flops)
{
  namespace pf = vpipe::perf_flops;
  const Small S = load_small_("wide");
  if (!S.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  const std::string ckpt = (S.dir / S.tag).string();
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalFlux3Transformer::load(ckpt, mc, cfg, false);
  F3_REQUIRE(m != nullptr);
  const long long H = cfg.hidden, MLP = cfg.mlp_hidden;
  const long long TXT = S.txt_rows, IMG = (long long)S.gh * S.gw;
  long long COND = 0;
  for (const auto& r : S.refs) { COND += (long long)r.first * r.second; }
  const long long JT = TXT + IMG + COND;
  auto block = [&](long long R) -> std::uint64_t {
    return 4 * gemm_flops(R, H, H) + gemm_flops(R, 2 * MLP, H) +
           gemm_flops(R, H, MLP) +
           attention_flops(cfg.n_heads, R * R, cfg.head_dim);
  };
  const std::uint64_t time1 =
      gemm_flops(1, H, cfg.time_freq) + gemm_flops(1, H, H);
  const std::uint64_t want =
      3 * time1 + 6 * gemm_flops(1, 3 * H, H) + gemm_flops(1, 2 * H, H) +
      gemm_flops(TXT, H, cfg.joint_dim) + gemm_flops(IMG, H, 128) +
      gemm_flops(COND, H, 128) +
      (std::uint64_t)cfg.n_mode * (block(IMG) + block(TXT) + block(COND)) +
      (std::uint64_t)cfg.n_single * block(JT) + gemm_flops(IMG, 128, H);
  const pf::Totals g0 = pf::totals(pf::kGpu);
  std::string e;
  const std::vector<float> out = run_small_(m.get(), mc, S, true, &e);
  F3_REQUIRE(!out.empty());
  const std::uint64_t flushed = m->gpu_flops().flushed();
  for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0.flops < flushed;
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const std::uint64_t got = pf::totals(pf::kGpu).flops - g0.flops;
  std::printf("[flux3_dit] counted %.6f GFLOP, derived %.6f, credited "
              "%.6f\n", flushed / 1e9, want / 1e9, got / 1e9);
  EXPECT_TRUE(flushed == want);
  EXPECT_TRUE(got == flushed);
}

// The ANE takes the tail rows of the joint blocks' feed-forward, under
// the GPU's attention. Needs a joint sequence past one 2048-row chunk
// (below it the tier declines), so this runs the PUBLISHED weights with
// the joint stack cut to 3 blocks, at 1024x1024 (4096 image rows).
TEST(flux3_dit, ane_feed_forward_rows_are_right)
{
  const std::string dir = dit_dir_();
  if (dir.empty() || env_("VPIPE_FLUX3_ANE_TEST").empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux3Transformer::Config cfg;
  F3_REQUIRE(MetalFlux3Transformer::Config::read_dims(dir, &cfg));
  cfg.n_single = 3;
  const int gh = 64, gw = 64, txt = 80;
  auto rnd = [&](std::size_t n, unsigned seed) {
    std::vector<float> v(n);
    unsigned st = seed;
    for (auto& x : v) {
      st = st * 1664525u + 1013904223u;
      x = ((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f);
    }
    return v;
  };
  const std::vector<float> img = rnd((std::size_t)gh * gw * 128, 3u);
  std::vector<float> text = rnd((std::size_t)txt * cfg.joint_dim, 4u);
  for (auto& x : text) { x *= 4.0f; }
  using Specs = std::vector<MetalFlux3Transformer::LoraSpec>;
  int bound = -1;
  auto run = [&](bool ane, bool* armed,
                 const Specs& loras = {}) -> std::vector<float> {
    MetalFlux3Transformer::Config c = cfg;
    c.ane_ffn = ane;
    auto m = MetalFlux3Transformer::load(dir, mc, c, false, loras);
    if (m == nullptr) { return {}; }
    if (!loras.empty()) { bound = m->lora_modules(0); }
    SharedBuffer lb = to_bf16_(mc, img), tb = to_bf16_(mc, text);
    MetalFlux3Transformer::Request req;
    req.latents = &lb;
    req.grid_h = gh;
    req.grid_w = gw;
    req.text = &tb;
    req.text_rows = txt;
    req.t = 0.9f;
    req.t_next = 0.85f;
    std::string e;
    // Twice: the first forward probes, the second splits.
    SharedBuffer v = m->forward(req, &e);
    v = m->forward(req, &e);
    if (armed != nullptr) {
      *armed = m->ane_armed() && !m->ane_gpu_only();
    }
    return from_bf16_(v, img.size());
  };
  bool armed = false;
  const std::vector<float> gpu = run(false, nullptr);
  const std::vector<float> ane = run(true, &armed);
  F3_REQUIRE(!gpu.empty() && ane.size() == gpu.size());
  if (!armed) {
    std::printf("[flux3_dit] the ANE tier did not arm here; skipped\n");
    return;
  }
  const double d = rel_l2_(ane, gpu);
  std::printf("[flux3_dit] ANE feed-forward on 3 joint blocks at %d rows: "
              "rel-L2 vs the GPU %.6f\n", txt + gh * gw, d);
  EXPECT_TRUE(d > 0.0 && d < 0.02);

  // AND UNDER A LoRA. The ANE stages the adapter beside the weight, and
  // linear_in is [gate | up] in one matrix, so the up half's B rows are
  // taken from row mlp_hidden on -- an offset no other family needs, and
  // one that would be silent: a wrong one adds the WRONG rows of B on the
  // ANE's rows only. A synthetic rank-8 adapter on both projections of
  // the three joint blocks, sized to move the output far past the ANE's
  // own 0.02 (factors of 0.05 moved it only 0.0033: three blocks' FF
  // behind a modulation gate is a small share of the residual stream).
  const int R = 8, H = cfg.hidden, MLP = cfg.mlp_hidden;
  auto bf16 = [](const std::vector<float>& v) {
    std::vector<std::uint8_t> out(v.size() * 2);
    for (std::size_t i = 0; i < v.size(); ++i) {
      const std::uint16_t h = genai::train::f32_to_bf16(v[i]);
      std::memcpy(out.data() + 2 * i, &h, 2);
    }
    return out;
  };
  auto scaled = [&](std::size_t n, unsigned seed) {
    std::vector<float> v = rnd(n, seed);
    for (auto& x : v) { x *= 0.35f; }
    return v;
  };
  std::vector<genai::train::OutTensor> ts;
  for (int j = 0; j < cfg.n_single; ++j) {
    const std::string p =
        "single_transformer_blocks." + std::to_string(j) + ".ff.";
    const struct { const char* leaf; int n, k; } mods[] = {
        {"linear_in", 2 * MLP, H}, {"linear_out", H, MLP}};
    for (const auto& md : mods) {
      const unsigned sd = 100u * (unsigned)j + (unsigned)md.k;
      ts.push_back({p + md.leaf + ".lora_A.weight", "BF16", {R, md.k},
                    bf16(scaled((std::size_t)R * md.k, sd))});
      ts.push_back({p + md.leaf + ".lora_B.weight", "BF16", {md.n, R},
                    bf16(scaled((std::size_t)md.n * R, sd + 7u))});
    }
  }
  const fs::path lp = fs::temp_directory_path() / "vpipe-flux3-ane-lora" /
                      "adapter.safetensors";
  std::error_code ec;
  fs::create_directories(lp.parent_path(), ec);
  std::string werr;
  F3_REQUIRE(genai::train::write_safetensors(lp.string(), ts, {}, &werr));
  const Specs specs{{lp.string(), 1.0f}};
  bool armed_l = false;
  const std::vector<float> gpu_l = run(false, nullptr, specs);
  F3_REQUIRE(bound == 2 * cfg.n_single);
  const std::vector<float> ane_l = run(true, &armed_l, specs);
  fs::remove_all(lp.parent_path(), ec);
  F3_REQUIRE(!gpu_l.empty() && ane_l.size() == gpu_l.size());
  F3_REQUIRE(armed_l);
  const double moved = rel_l2_(gpu_l, gpu);
  const double dl = rel_l2_(ane_l, gpu_l);
  std::printf("[flux3_dit] ... under a LoRA (%d modules, moving the GPU "
              "output %.4f): ANE vs GPU %.6f\n", bound, moved, dl);
  EXPECT_TRUE(moved > 0.1);
  EXPECT_TRUE(dl > 0.0 && dl < 0.02);
}

// BFL's own single-file layout has no transformer/ to name its family,
// so unrecognised it fell through to the "krea2" guess and the
// conditioner encoded the prompt as Krea-2's. It is refused by name.
TEST(flux3_sampler, bfl_root_layout_is_recognised_and_diffusers_is_not)
{
  const fs::path d = fs::temp_directory_path() / "vpipe_flux3_layout_test";
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d, ec);
  auto put = [&](const fs::path& p, const char* json) {
    fs::create_directories(p.parent_path(), ec);
    std::ofstream o(p);
    o << json;
  };
  // The BFL root: config.json names the model.
  put(d / "config.json", R"({"format_version":1,"model":"flux-3-image-klein",
                              "dit":{"hidden":3072}})");
  EXPECT_TRUE(flux3::is_bfl_root_layout(d.string()));
  // A variants/ folder of it says so in its manifest instead.
  put(d / "v" / "manifest.json", R"({"kind":"f3_dit_export"})");
  EXPECT_TRUE(flux3::is_bfl_root_layout((d / "v").string()));
  // The published repo ships BOTH: transformer/ wins, and it is read.
  put(d / "transformer" / "config.json",
      R"({"_class_name":"Flux3Transformer2DModel"})");
  EXPECT_FALSE(flux3::is_bfl_root_layout(d.string()));
  // And someone else's root config is not mistaken for it.
  put(d / "other" / "config.json", R"({"model":"flux-2-klein"})");
  EXPECT_FALSE(flux3::is_bfl_root_layout((d / "other").string()));
  EXPECT_FALSE(flux3::is_bfl_root_layout((d / "missing").string()));
  fs::remove_all(d, ec);
}
