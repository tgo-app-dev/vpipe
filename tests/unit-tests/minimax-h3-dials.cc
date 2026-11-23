// DIALS on MiniMax H3 (generative-models/minimax-h3/minimax-h3-dials.h):
// one steering vector added to every prompt row -- after the token
// refiner (the post seam) or into condition_proj's bias (the pre seam) --
// and a dial on a TIME CURVE, the DiT run again with the dial for the
// first steps and the two velocities blended per latent frame.
//
// Self-contained: the GPU cases write their own small H3 checkpoint (as
// minimax_h3_whole does) and their own dial files.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-dials.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "stages/minimax-h3-model-config-stage.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <malloc/malloc.h>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;
namespace h3 = vpipe::genai::minimax_h3;
using H3 = MetalMiniMaxH3Transformer;

struct T {
  std::string               name;
  std::string               dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

bool
write_st_(const fs::path& p, const std::vector<T>& ts,
          const std::string& meta = {})
{
  std::string hdr = "{";
  if (!meta.empty()) { hdr += "\"__metadata__\":" + meta + ","; }
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string shp;
    for (std::size_t d = 0; d < ts[i].shape.size(); ++d) {
      shp += (d ? "," : "") + std::to_string(ts[i].shape[d]);
    }
    hdr += (i ? "," : "") + std::string("\"") + ts[i].name +
           "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[" + shp +
           "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + ts[i].bytes.size()) + "]}";
    off += ts[i].bytes.size();
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  if (!f) { return false; }
  const std::uint64_t n = hdr.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(hdr.data(), (std::streamsize)hdr.size());
  for (const T& t : ts) {
    f.write(reinterpret_cast<const char*>(t.bytes.data()),
            (std::streamsize)t.bytes.size());
  }
  return f.good();
}

std::uint16_t
rne_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16f_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

T
f32_(const std::string& nm, std::vector<std::int64_t> shape,
     const std::vector<float>& v)
{
  T t{nm, "F32", std::move(shape), std::vector<std::uint8_t>(v.size() * 4)};
  std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
  return t;
}

T
bf16_(const std::string& nm, std::vector<std::int64_t> shape,
      const std::vector<float>& v)
{
  T t{nm, "BF16", std::move(shape), std::vector<std::uint8_t>(v.size() * 2)};
  for (std::size_t i = 0; i < v.size(); ++i) {
    const std::uint16_t b = rne_(v[i]);
    std::memcpy(&t.bytes[i * 2], &b, 2);
  }
  return t;
}

fs::path
tmp_(const std::string& name)
{
  return fs::temp_directory_path() /
         ("vpipe-h3-dials-" + std::to_string(::getpid()) + "-" + name);
}

struct Env {
  const char* k;
  Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
  ~Env() { ::unsetenv(k); }
};

constexpr int kH = 256;

// A released-layout H3 checkpoint, hidden 256 (minimax_h3_whole's), two
// main blocks and one refiner block. `bias_delta`, when given, is added
// to condition_proj's bias -- the stored value bf16(bf16(b) + delta),
// what a pre dial patches the bias to.
bool
write_model_(const fs::path& p, const std::vector<float>* bias_delta)
{
  const int H = kH, A = 256, TXT = 256, FREQ = 256, TH = 64, M = 64, F = 512;
  const int layers = 2;
  const std::string P = "model.diffusion_model.";
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto rnd = [&](std::size_t n, float sd, float mean = 0.0f) {
    std::vector<float> v(n);
    for (float& x : v) { x = mean + sd * nd(rng); }
    return v;
  };
  std::vector<T> ts;
  auto add = [&](T t) {
    t.name = P + t.name;
    ts.push_back(std::move(t));
  };
  add(f32_("video_patch_proj.weight", {H, 96}, rnd(H * 96, 0.05f)));
  add(f32_("video_patch_proj.bias", {H}, rnd(H, 0.01f)));
  add(f32_("audio_patch_proj.weight", {H, 32}, rnd(H * 32, 0.05f)));
  add(f32_("audio_patch_proj.bias", {H}, rnd(H, 0.01f)));
  add(bf16_("condition_proj.weight", {H, TXT},
            rnd((std::size_t)H * TXT, 0.03f)));
  std::vector<float> cb = rnd(H, 0.01f);
  if (bias_delta != nullptr) {
    for (int i = 0; i < H; ++i) {
      cb[(std::size_t)i] =
          bf16f_(rne_(cb[(std::size_t)i])) + (*bias_delta)[(std::size_t)i];
    }
  }
  add(bf16_("condition_proj.bias", {H}, cb));
  std::vector<float> inv(16);
  for (int i = 0; i < 16; ++i) {
    inv[(std::size_t)i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
  }
  add(f32_("rope.inv_freq", {16}, inv));
  for (int L = 0; L < layers + 1; ++L) {
    const std::string b = L < layers
        ? "blocks." + std::to_string(L) + "."
        : std::string("token_refiner.blocks.0.");
    add(bf16_(b + "norm1.weight", {H}, rnd(H, 0.02f, 1.0f)));
    add(bf16_(b + "norm2.weight", {H}, rnd(H, 0.02f, 1.0f)));
    add(bf16_(b + "attn.q_norm.weight", {128}, rnd(128, 0.02f, 1.0f)));
    add(bf16_(b + "attn.k_norm.weight", {128}, rnd(128, 0.02f, 1.0f)));
    add(bf16_(b + "attn.qkv_proj.weight", {3 * A, H},
              rnd((std::size_t)3 * A * H, 0.04f)));
    add(bf16_(b + "attn.out_proj.weight", {H, A},
              rnd((std::size_t)H * A, 0.04f)));
    add(bf16_(b + "mlp.fc1.weight", {2 * F, H},
              rnd((std::size_t)2 * F * H, 0.04f)));
    add(bf16_(b + "mlp.fc2.weight", {H, F},
              rnd((std::size_t)H * F, 0.01f)));
    if (L < layers) {
      add(bf16_(b + "adaln_proj.linear.weight", {18 * H, M},
                rnd((std::size_t)18 * H * M, 0.08f)));
      add(bf16_(b + "adaln_proj.linear.bias", {18 * H},
                rnd((std::size_t)18 * H, 0.05f)));
    }
  }
  add(bf16_("token_refiner.final_norm.weight", {H}, rnd(H, 0.02f, 1.0f)));
  add(bf16_("final_layer.norm.weight", {H}, rnd(H, 0.02f, 1.0f)));
  add(bf16_("final_layer.adaln_proj.linear.weight", {2 * H, M},
            rnd((std::size_t)2 * H * M, 0.08f)));
  add(bf16_("final_layer.adaln_proj.linear.bias", {2 * H},
            rnd((std::size_t)2 * H, 0.05f)));
  add(f32_("final_layer.video_out.weight", {96, H}, rnd(96 * H, 0.05f)));
  add(f32_("final_layer.video_out.bias", {96}, rnd(96, 0.01f)));
  add(f32_("final_layer.audio_out.weight", {32, H}, rnd(32 * H, 0.05f)));
  add(f32_("final_layer.audio_out.bias", {32}, rnd(32, 0.01f)));
  add(f32_("time_embedder.proj_in.weight", {TH, FREQ},
           rnd((std::size_t)TH * FREQ, 0.15f)));
  add(f32_("time_embedder.proj_in.bias", {TH}, rnd(TH, 0.2f)));
  add(f32_("time_embedder.proj_out.weight", {M, TH},
           rnd((std::size_t)M * TH, 0.3f)));
  add(f32_("time_embedder.proj_out.bias", {M}, rnd(M, 0.2f)));
  return write_st_(p, ts);
}

std::unique_ptr<H3>
load_(const fs::path& file, metal_compute::MetalCompute* mc, H3::Config* c)
{
  std::string err;
  if (!H3::config_from_json(file.string(), *c, &err)) {
    std::printf("  config: %s\n", err.c_str());
    return nullptr;
  }
  std::unique_ptr<H3> m;
  try {
    m = H3::load(file.string(), mc, *c, false, {});
  } catch (const std::exception& e) {
    std::printf("  refused: %s\n", e.what());
  }
  if (m != nullptr) { m->set_gemm_route(H3::GemmRoute::kSteelBm32); }
  return m;
}

// A dial direction, N(0, 1) scaled -- a magnitude a refined prompt row
// (RMS-normed, ~1) notices.
std::vector<float>
direction_(int seed, float scale)
{
  std::mt19937 rng((unsigned)seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> v(kH);
  for (float& x : v) { x = scale * nd(rng); }
  return v;
}

// A dial file as ComfyUI-Dials writes one: one tensor, string metadata.
bool
write_dial_(const fs::path& p, const std::string& key,
            const std::vector<float>& v, const std::string& meta,
            const char* dtype = "BF16")
{
  T t = std::string(dtype) == "F32" ? f32_(key, {(std::int64_t)v.size()}, v)
                                    : bf16_(key, {(std::int64_t)v.size()}, v);
  return write_st_(p, {t}, meta);
}

// The packed sequence one step of the synthetic model runs: text, a
// t2va layout of `latf` latent frames (no anchors), its audio.
struct Clip {
  h3::PackedLayout L;
  metal_compute::SharedBuffer text;
  std::vector<float> v0, a0;   // the starting state
  int latf = 2, alat = 8;
  int lath = 8, latw = 8;      // the latent grid
};

bool
clip_(metal_compute::MetalCompute* mc, const H3::Config& c, Clip* k)
{
  const int ntext = 12;
  const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
  if (!h3::build_packed_sequence(tags, k->latf, k->lath, k->latw, k->alat,
                                 c.patch_h, c.patch_w, h3::kAudioChannels,
                                 {}, &k->L)) {
    return false;
  }
  k->text = mc->make_shared_buffer((std::size_t)ntext * c.text_dim * 2);
  if (k->text.empty()) { return false; }
  auto* p = static_cast<std::uint16_t*>(k->text.contents());
  for (std::size_t i = 0; i < (std::size_t)ntext * c.text_dim; ++i) {
    p[i] = rne_(std::sin((float)(i % 100003) * 0.005f) * 0.5f);
  }
  std::mt19937_64 g(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  k->v0.resize(k->L.video_indices.size() * (std::size_t)c.video_patch_elems());
  k->a0.resize(k->L.audio_indices.size() * (std::size_t)c.audio_channels);
  for (float& x : k->v0) { x = nd(g); }
  for (float& x : k->a0) { x = nd(g); }
  return true;
}

struct Out {
  std::vector<float> v, a;
  DenoiseReport rep;
  std::vector<std::pair<int, int>> units;
  bool ok = false;
};

Out
denoise_(H3& m, const Clip& k, int steps,
         std::vector<DialBranch> branches = {})
{
  Out o;
  o.v = k.v0;
  o.a = k.a0;
  DenoiseRequest q;
  q.dit = &m;
  q.layout = &k.L;
  q.text = &k.text;
  q.video = o.v.data();
  q.audio = o.a.data();
  q.num_steps = steps;
  q.report = &o.rep;
  q.dial_branches = std::move(branches);
  q.forward_progress = [&](int done, int total) {
    o.units.emplace_back(done, total);
    return true;
  };
  std::string err;
  o.ok = denoise(q, &err);
  if (!o.ok) { std::printf("  denoise: %s\n", err.c_str()); }
  return o;
}

// The generated video rows of latent frame `f` (rows per frame `per`),
// `PE` wide, from a state that leads with no conditioning rows.
std::vector<float>
frame_rows_(const std::vector<float>& v, int f, int per, int PE)
{
  return std::vector<float>(v.begin() + (std::ptrdiff_t)f * per * PE,
                            v.begin() + (std::ptrdiff_t)(f + 1) * per * PE);
}

double
rel_(const std::vector<float>& x, const std::vector<float>& y)
{
  if (x.empty() || x.size() != y.size()) { return 1e9; }
  double d = 0.0, n = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    d += ((double)x[i] - y[i]) * ((double)x[i] - y[i]);
    n += (double)y[i] * y[i];
  }
  return std::sqrt(d / std::max(n, 1e-30));
}

h3::DialCurve
keys_(std::vector<std::pair<double, double>> pts)
{
  h3::DialCurve c;
  c.points = std::move(pts);
  return c;
}

}  // namespace

// The rack's rows, read as the Dial Rack writes them -- and every row it
// can hold that this cannot run, refused by name.
TEST(minimax_h3_dials, rack_rows_read_and_refuse)
{
  const std::string rack =
      R"([{"dial": "dials/h3_motion.safetensors", "strength": 1.5,)"
      R"( "on": true, "color": "#f00"},)"
      R"( {"dial": "dials/h3_impact_frame.safetensors", "strength": 4.5,)"
      R"( "time": {"type": "keys", "points": [[0, 0], [0.4, 0], [0.45, 1],)"
      R"( [0.55, 1], [0.6, 0]], "steps": 6, "audio": "protect"}},)"
      R"( {"dial": "off.safetensors", "strength": 2, "on": false},)"
      R"( {"dial": "zero.safetensors", "strength": 0}])";
  std::vector<h3::DialRow> rows;
  std::string err;
  ASSERT_TRUE(h3::parse_dial_rows(FlexData::make_string(rack), &rows, &err));
  ASSERT_TRUE(rows.size() == 4);
  if (rows.size() != 4) { return; }
  EXPECT_TRUE(!rows[0].timed && rows[0].strength == 1.5);
  EXPECT_TRUE(rows[1].timed && rows[1].time.steps == 6 &&
              rows[1].time.protect_audio && !rows[1].time.seconds &&
              !rows[1].time.step && rows[1].time.points.size() == 5);
  // On, not zero; and a soloed row alone where one is soloed.
  EXPECT_TRUE(h3::active_dial_rows(rows).size() == 2);
  rows[1].solo = true;
  const auto solo = h3::active_dial_rows(rows);
  EXPECT_TRUE(solo.size() == 1 && solo[0].dial.find("impact") !=
                                      std::string::npos);
  rows[1].solo = false;
  // Out and back: the same rows.
  std::vector<h3::DialRow> back;
  ASSERT_TRUE(h3::parse_dial_rows(h3::dial_rows_to_flex(rows), &back, &err));
  EXPECT_TRUE(back.size() == 4 && back[1].time.key() == rows[1].time.key() &&
              !back[2].on && back[0].strength == 1.5);
  // An empty value is no rows; a Python-falsy `time` no curve.
  EXPECT_TRUE(h3::parse_dial_rows(FlexData::make_string(""), &rows, &err) &&
              rows.empty());
  ASSERT_TRUE(h3::parse_dial_rows(
      FlexData::from_json(R"({"dial": "x", "strength": 1, "time": null})"),
      &rows, &err));
  EXPECT_TRUE(rows.size() == 1 && !rows[0].timed);
  // Refused, each by name.
  const std::vector<std::pair<const char*, const char*>> bad = {
      {R"([{"dial": "x"}])", "strength"},
      {R"([{"dial": "x", "strength": 1, "hold": true}])", "hold"},
      {R"([{"dial": "x", "strength": 1, "envelope": "early"}])", "envelope"},
      {R"([{"dial": "x", "strength": 1, "blocks": [1, 0]}])", "blocks"},
      {R"([{"input": "dial_0", "strength": 1}])", "input"},
      {R"([{"dial": "x", "strength": 1, "time": {"type": "audio"}}])",
       "audio-driven"},
      {R"([{"dial": "x", "strength": 1, "time": {"type": "keys",)"
       R"( "points": [[0.5, 0], [0.5, 1]]}}])", "increasing"},
      {R"([{"dial": "x", "strength": 1, "time": {"type": "keys",)"
       R"( "points": [[0, 1]], "smooth": 2}}])", "smooth"},
      {R"([{"dial": "x", "strength": 1, "time": {"type": "keys",)"
       R"( "points": [[0, 1]], "steps": 0}}])", "steps"},
      {R"([{"dial": "x", "strength": 1, "time": {"type": "keys",)"
       R"( "points": [[0, 1]], "audio": "mute"}}])", "audio"},
  };
  for (const auto& [json, word] : bad) {
    err.clear();
    const bool ok =
        h3::parse_dial_rows(FlexData::from_json(json), &rows, &err);
    std::printf("  refused: %s\n", err.c_str());
    EXPECT_TRUE(!ok && err.find(word) != std::string::npos);
  }
}

// MAINodes' curves: linear keys held flat outside, step keys, and rate
// smoothing that still meets every key, never overshoots, and has no
// corner at a key.
TEST(minimax_h3_dials, curves_meet_their_keys)
{
  h3::DialCurve c = keys_({{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0}});
  c.seconds = true;
  EXPECT_TRUE(c.at(-1.0, 9.0) == 0.0 && c.at(0.5, 9.0) == 1.0 &&
              c.at(2.0, 9.0) == 1.5 && c.at(4.0, 9.0) == 1.0);
  // Normalized: times scale by the clip.
  h3::DialCurve n = keys_({{0.0, 0.0}, {0.5, 1.0}});
  EXPECT_TRUE(n.at(1.0, 4.0) == 0.5 && n.at(3.0, 4.0) == 1.0);
  h3::DialCurve s = c;
  s.step = true;
  EXPECT_TRUE(s.at(-1.0, 9.0) == 0.0 && s.at(0.99, 9.0) == 0.0 &&
              s.at(1.0, 9.0) == 2.0 && s.at(2.9, 9.0) == 2.0 &&
              s.at(3.0, 9.0) == 1.0);
  // Smoothing: a rise, a peak, a fall.
  h3::DialCurve m = keys_({{0.0, 0.0}, {1.0, 1.0}, {2.0, 3.0},
                           {3.0, 0.5}});
  m.seconds = true;
  for (double sm : {0.25, 0.5, 1.0}) {
    m.smooth = sm;
    bool through = true, bounded = true, smooth = true;
    for (std::size_t i = 0; i < m.points.size(); ++i) {
      through = through &&
                std::fabs(m.at(m.points[i].first, 0.0) -
                          m.points[i].second) < 1e-12;
    }
    for (std::size_t i = 0; i + 1 < m.points.size(); ++i) {
      const double y0 = m.points[i].second, y1 = m.points[i + 1].second;
      for (int k = 0; k <= 100; ++k) {
        const double t = m.points[i].first + k / 100.0;
        const double y = m.at(t, 0.0);
        bounded = bounded && y >= std::min(y0, y1) - 1e-12 &&
                  y <= std::max(y0, y1) + 1e-12;
      }
    }
    // C1 at the inner keys: the slope on either side agrees.
    for (double x : {1.0, 2.0}) {
      const double h = 1e-6;
      const double l = (m.at(x, 0.0) - m.at(x - h, 0.0)) / h;
      const double r = (m.at(x + h, 0.0) - m.at(x, 0.0)) / h;
      smooth = smooth && std::fabs(l - r) < 1e-4;
    }
    // A peak (secants change sign) is flat there.
    const double hp = 1e-6;
    const double slope2 = (m.at(2.0 + hp, 0.0) - m.at(2.0 - hp, 0.0)) / 2e-6;
    std::printf("  smooth %.2f: through keys %d, no overshoot %d, C1 %d, "
                "slope at the peak %.2e\n", sm, through, bounded, smooth,
                slope2);
    EXPECT_TRUE(through && bounded && smooth && std::fabs(slope2) < 1e-4);
  }
  m.smooth = 0.0;
  EXPECT_TRUE(std::fabs(m.at(1.5, 0.0) - 2.0) < 1e-15);
}

// The two clocks: a video latent frame weighs the curve's mean over its
// pixel frames (1, 4, 4, 4, 4 a group of five, at the clip's fps), an
// audio latent the curve at its start (40 a second); protect silences
// the audio's.
TEST(minimax_h3_dials, curves_land_on_latent_frames)
{
  // 7 latent frames: pixel frames 0 | 1-4 | 5-8 | 9-12 | 13-16 | 17 |
  // 18-21, 22 in all; a step from 0 to 1 at pixel frame 10 (seconds).
  h3::DialCurve c = keys_({{0.0, 0.0}, {10.0 / 24.0, 1.0}});
  c.seconds = true;
  c.step = true;
  std::vector<float> v, a;
  h3::dial_frame_weights(c, 7, 40, 24.0, &v, &a);
  ASSERT_TRUE(v.size() == 7 && a.size() == 40);
  const std::vector<float> want = {0.0f, 0.0f, 0.0f, 0.75f, 1.0f, 1.0f, 1.0f};
  EXPECT_TRUE(v == want);
  // Audio latent j at j / 40 s: 10 / 24 s is 16.67 latents in.
  EXPECT_TRUE(a[16] == 0.0f && a[17] == 1.0f && a[39] == 1.0f);
  // Normalized over the clip's 22 frames: 0.5 is frame 11.
  h3::DialCurve n = keys_({{0.0, 0.0}, {0.5, 1.0}});
  n.step = true;
  h3::dial_frame_weights(n, 7, 40, 24.0, &v, &a);
  EXPECT_TRUE(v[2] == 0.0f && v[3] == 0.5f && v[4] == 1.0f);
  n.protect_audio = true;
  h3::dial_frame_weights(n, 7, 40, 24.0, &v, &a);
  bool silent = true;
  for (float w : a) { silent = silent && w == 0.0f; }
  EXPECT_TRUE(silent && v[4] == 1.0f);
}

// Dial files: both seams and spellings read, metadata named; another
// family's, another width, two tensors and a non-finite value refused.
// And the plan sums rows as the rack does.
TEST(minimax_h3_dials, files_and_the_plan)
{
  const std::vector<float> d1 = direction_(1, 0.5f), d2 = direction_(2, 0.5f);
  const fs::path post = tmp_("post.safetensors"), pre = tmp_("pre.safetensors");
  const fs::path moss = tmp_("moss.safetensors"), two = tmp_("two.safetensors");
  ASSERT_TRUE(write_dial_(post, "diffusion_model.token_refiner.dials_shift",
                          d1, R"({"name": "h3_test_post",)"
                              R"( "format": "ComfyUI-Dials v1"})"));
  ASSERT_TRUE(write_dial_(pre, "condition_proj.diff_b", d2, "", "F32"));
  ASSERT_TRUE(write_dial_(moss, "diffusion_model.token_refiner.dials_shift",
                          d1, R"({"model_family": "MOSS-Image"})"));
  ASSERT_TRUE(write_st_(two, {bf16_("diffusion_model.token_refiner."
                                    "dials_shift", {kH}, d1),
                              bf16_("diffusion_model.x", {kH}, d2)}));
  h3::Dial a, b, x;
  std::string err;
  ASSERT_TRUE(h3::read_dial(post.string(), kH, &a, &err));
  ASSERT_TRUE(h3::read_dial(pre.string(), kH, &b, &err));
  EXPECT_TRUE(a.seam == h3::DialSeam::kPost && a.name == "h3_test_post" &&
              b.seam == h3::DialSeam::kPre && b.name.find("pre") !=
                                                  std::string::npos);
  // bf16 as stored, f32 exactly.
  EXPECT_TRUE(a.vec[3] == bf16f_(rne_(d1[3])) && b.vec == d2);
  for (const fs::path& p : {moss, two}) {
    err.clear();
    EXPECT_TRUE(!h3::read_dial(p.string(), kH, &x, &err));
    std::printf("  refused: %s\n", err.c_str());
  }
  err.clear();
  EXPECT_TRUE(!h3::read_dial(post.string(), 5376, &x, &err) &&
              err.find("5376") != std::string::npos);

  // One post row: vec * strength in f32; two: summed in double.
  h3::DialRow r1, r2;
  r1.dial = "a"; r1.strength = 1.5;
  r2.dial = "b"; r2.strength = -0.75;
  h3::Dial a2 = a;
  a2.vec = d2;
  a2.name = "other";
  h3::DialPlan plan;
  ASSERT_TRUE(h3::compile_dial_plan({r1}, {a}, &plan, &err));
  EXPECT_TRUE(plan.pre.empty() && plan.groups.empty() &&
              plan.post[5] == a.vec[5] * 1.5f);
  ASSERT_TRUE(h3::compile_dial_plan({r1, r2}, {a, a2}, &plan, &err));
  EXPECT_TRUE(plan.post[5] ==
              (float)(1.5 * (double)a.vec[5] - 0.75 * (double)d2[5]));
  // A pre row on its own seam.
  ASSERT_TRUE(h3::compile_dial_plan({r1, r2}, {a, b}, &plan, &err));
  EXPECT_TRUE(plan.post[5] == a.vec[5] * 1.5f &&
              plan.pre[5] == b.vec[5] * -0.75f);
  // Curves: rows on one curve share a branch, another curve is another.
  h3::DialRow t1 = r1, t2 = r2, t3 = r1;
  t1.timed = t2.timed = t3.timed = true;
  t1.time = t2.time = keys_({{0.0, 0.0}, {1.0, 1.0}});
  t3.time = keys_({{0.0, 1.0}});
  ASSERT_TRUE(h3::compile_dial_plan({t1, t2, t3}, {a, a2, a}, &plan, &err));
  EXPECT_TRUE(plan.post.empty() && plan.groups.size() == 2 &&
              plan.groups[0].label == "h3_test_post + other" &&
              plan.groups[0].shift[5] ==
                  (float)(1.5 * (double)a.vec[5] - 0.75 * (double)d2[5]));
  // A pre dial cannot follow a curve.
  err.clear();
  EXPECT_TRUE(!h3::compile_dial_plan({t1}, {b}, &plan, &err) &&
              err.find("pre") != std::string::npos);
  for (const fs::path& p : {post, pre, moss, two}) { fs::remove(p); }
}

// The blend: torch.lerp, exact at both ends, per group of rows; the
// several-branch form adds a difference from the one base.
TEST(minimax_h3_dials, the_blend_is_a_lerp_by_frame)
{
  const std::size_t rows = 6, W = 3;
  std::vector<float> base(rows * W), out;
  std::vector<std::uint16_t> d(rows * W);
  for (std::size_t i = 0; i < rows * W; ++i) {
    base[i] = 0.1f * (float)i - 0.3f;
    d[i] = rne_(1.0f - 0.07f * (float)i);
  }
  // Three frames of two rows: 0, 0.25, 1.
  out = base;
  blend_dial_velocity(out.data(), nullptr, d.data(), rows, W,
                      {0.0f, 0.25f, 1.0f}, 1);
  bool ok = true;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t k = 0; k < W; ++k) {
      const std::size_t i = r * W + k;
      const float b = base[i], e = bf16f_(d[i]);
      const float want = r < 2 ? b : r < 4 ? b + 0.25f * (e - b) : e;
      ok = ok && out[i] == want;
    }
  }
  EXPECT_TRUE(ok);
  // Audio: three latents, the stereo rows cycling twice over them.
  out = base;
  blend_dial_velocity(out.data(), nullptr, d.data(), rows, W,
                      {1.0f, 0.0f, 1.0f}, 2);
  EXPECT_TRUE(out[0] == bf16f_(d[0]) && out[W] == base[W] &&
              out[3 * W] == bf16f_(d[3 * W]) && out[4 * W] == base[4 * W]);
  // Several: out += m (d - base).
  out = base;
  std::vector<float> acc = base;
  blend_dial_velocity(out.data(), acc.data(), d.data(), rows, W,
                      {0.5f, 0.5f, 0.5f}, 1);
  EXPECT_TRUE(out[7] == base[7] + 0.5f * (bf16f_(d[7]) - base[7]));
}

// THE POST SEAM, read off the rows the main blocks see: every prompt row
// is the undialled one plus the shift, rounded to bf16 -- a constant
// dial's (set_dials), a forward's (Step::text_shift), and both one after
// the other.
TEST(minimax_h3_dials, a_post_dial_lands_on_every_prompt_row)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const fs::path file = tmp_("model.safetensors");
  ASSERT_TRUE(write_model_(file, nullptr));
  H3::Config c;
  std::unique_ptr<H3> m = load_(file, mc, &c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { fs::remove(file); return; }
  Clip k;
  ASSERT_TRUE(clip_(mc, c, &k));
  m->capture_text_rows(true);
  const std::vector<float> s1 = direction_(1, 0.7f), s2 = direction_(2, 0.4f);
  // One step, and the prompt rows that step's forward read.
  auto run = [&]() {
    Out o = denoise_(*m, k, 1);
    return std::make_pair(m->captured_text_rows(), o);
  };
  const auto [r0, o0] = run();
  ASSERT_TRUE(o0.ok && r0.size() == (std::size_t)12 * kH);
  ASSERT_TRUE(m->set_dials({}, s1));
  const auto [r1, o1] = run();
  bool exact = r1.size() == r0.size();
  for (std::size_t i = 0; exact && i < r0.size(); ++i) {
    exact = r1[i] == rne_(bf16f_(r0[i]) + bf16f_(rne_(s1[i % kH])));
  }
  std::printf("  constant post dial: every prompt row the undialled one + "
              "the shift in bf16: %s; video moved %.3g\n",
              exact ? "EXACT" : "NO", rel_(o1.v, o0.v));
  EXPECT_TRUE(exact && rel_(o1.v, o0.v) > 1e-3);
  // The same shift as a forward's own (a curve's branch at M 1 alone):
  // the same rows, the same velocity.
  ASSERT_TRUE(m->set_dials({}, {}));
  EXPECT_TRUE(!m->has_dials());
  const metal_compute::SharedBuffer b1 = m->text_shift_buffer(s1);
  DialBranch br;
  br.label = "s1";
  br.shift = &b1;
  br.steps = 1;
  br.video.assign((std::size_t)k.latf, 1.0f);
  br.audio.assign((std::size_t)k.alat, 1.0f);
  const Out oa = denoise_(*m, k, 1, {br});
  EXPECT_TRUE(oa.ok && m->captured_text_rows() == r1 && oa.v == o1.v &&
              oa.a == o1.a && oa.rep.dial_forwards == 0 &&
              oa.rep.forwards == 1);
  // Both: the constant shift, then the forward's.
  ASSERT_TRUE(m->set_dials({}, s2));
  (void)denoise_(*m, k, 1, {br});
  const std::vector<std::uint16_t>& r2 = m->captured_text_rows();
  exact = r2.size() == r0.size();
  for (std::size_t i = 0; exact && i < r0.size(); ++i) {
    const std::uint16_t once =
        rne_(bf16f_(r0[i]) + bf16f_(rne_(s2[i % kH])));
    exact = r2[i] == rne_(bf16f_(once) + bf16f_(rne_(s1[i % kH])));
  }
  EXPECT_TRUE(exact);
  m.reset();
  fs::remove(file);
}

// THE PRE SEAM is condition_proj's bias patched: a dial on one model
// runs exactly as a model whose file holds bf16(b + delta).
TEST(minimax_h3_dials, a_pre_dial_is_the_patched_bias)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const std::vector<float> d = direction_(5, 0.3f);
  const fs::path fa = tmp_("a.safetensors"), fb = tmp_("b.safetensors");
  ASSERT_TRUE(write_model_(fa, nullptr));
  ASSERT_TRUE(write_model_(fb, &d));
  H3::Config ca, cb;
  std::unique_ptr<H3> ma = load_(fa, mc, &ca), mb = load_(fb, mc, &cb);
  ASSERT_TRUE(ma != nullptr && mb != nullptr);
  if (ma == nullptr || mb == nullptr) {
    fs::remove(fa);
    fs::remove(fb);
    return;
  }
  Clip k;
  ASSERT_TRUE(clip_(mc, ca, &k));
  const Out plain = denoise_(*ma, k, 2);
  ASSERT_TRUE(ma->set_dials(d, {}));
  const Out dialled = denoise_(*ma, k, 2);
  const Out patched = denoise_(*mb, k, 2);
  std::printf("  pre dial against the patched file: %s; moved the clip "
              "%.3g\n", dialled.v == patched.v && dialled.a == patched.a
                            ? "IDENTICAL" : "DIFFERENT",
              rel_(dialled.v, plain.v));
  EXPECT_TRUE(dialled.ok && patched.ok && dialled.v == patched.v &&
              dialled.a == patched.a && rel_(dialled.v, plain.v) > 1e-3);
  ma.reset();
  mb.reset();
  fs::remove(fa);
  fs::remove(fb);
}

// A TIME CURVE through the loop, one step so every comparison is exact:
// M 0 is the undialled clip at no cost; a frame at M 1 is the constant
// dial's frame and one at M 0 the undialled one's; `protect` leaves the
// soundtrack the undialled one's; two curves add; and after `steps` the
// branch is off, counted in forwards.
TEST(minimax_h3_dials, a_time_curve_blends_by_latent_frame)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const fs::path file = tmp_("model.safetensors");
  ASSERT_TRUE(write_model_(file, nullptr));
  H3::Config c;
  std::unique_ptr<H3> m = load_(file, mc, &c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { fs::remove(file); return; }
  Clip k;
  k.latf = 4;
  ASSERT_TRUE(clip_(mc, c, &k));
  const int PE = c.video_patch_elems();
  const int per = k.L.num_video_rows / k.latf;
  const std::vector<float> s = direction_(7, 0.7f), s2 = direction_(8, 0.5f);
  const Out base = denoise_(*m, k, 1);
  ASSERT_TRUE(m->set_dials({}, s));
  const Out full = denoise_(*m, k, 1);
  ASSERT_TRUE(m->set_dials({}, s2));
  const Out full2 = denoise_(*m, k, 1);
  ASSERT_TRUE(m->set_dials({}, {}));
  ASSERT_TRUE(base.ok && full.ok && full2.ok);
  const metal_compute::SharedBuffer b = m->text_shift_buffer(s);
  const metal_compute::SharedBuffer b2 = m->text_shift_buffer(s2);
  auto branch = [&](const metal_compute::SharedBuffer& sh,
                    std::vector<float> video, float audio, int steps) {
    DialBranch br;
    br.label = "curve";
    br.shift = &sh;
    br.steps = steps;
    br.video = std::move(video);
    br.audio.assign((std::size_t)k.alat, audio);
    return br;
  };
  // M 0 everywhere: nothing runs.
  const Out zero = denoise_(*m, k, 1, {branch(b, {0, 0, 0, 0}, 0.0f, 1)});
  EXPECT_TRUE(zero.v == base.v && zero.a == base.a &&
              zero.rep.dial_forwards == 0);
  // Frames 0-1 undialled, 2-3 dialled, the soundtrack protected.
  const Out mixed = denoise_(*m, k, 1, {branch(b, {0, 0, 1, 1}, 0.0f, 1)});
  bool frames = mixed.ok && mixed.rep.dial_forwards == 1;
  for (int f = 0; f < k.latf; ++f) {
    const auto got = frame_rows_(mixed.v, f, per, PE);
    frames = frames && got == frame_rows_(f < 2 ? base.v : full.v, f, per,
                                          PE);
  }
  std::printf("  frames at M 0 / M 1 the undialled / dialled clip's: %s; "
              "protected soundtrack the undialled one's: %s\n",
              frames ? "EXACT" : "NO", mixed.a == base.a ? "EXACT" : "NO");
  EXPECT_TRUE(frames && mixed.a == base.a);
  // Following, the soundtrack is the dialled one's where M is 1.
  const Out follow = denoise_(*m, k, 1, {branch(b, {0, 0, 1, 1}, 1.0f, 1)});
  EXPECT_TRUE(follow.a == full.a && follow.rep.dial_forwards == 1);
  // A half: between the two, near their mean (one Euler step is linear
  // in the velocity).
  const Out half =
      denoise_(*m, k, 1, {branch(b, {0.5f, 0.5f, 0.5f, 0.5f}, 0.0f, 1)});
  std::vector<float> mean(base.v.size());
  for (std::size_t i = 0; i < mean.size(); ++i) {
    mean[i] = 0.5f * (base.v[i] + full.v[i]);
  }
  std::printf("  M 0.5: %.3g from the mean of the two clips (they are %.3g "
              "apart)\n", rel_(half.v, mean), rel_(full.v, base.v));
  EXPECT_TRUE(rel_(half.v, mean) < 1e-5 && rel_(full.v, base.v) > 1e-3);
  // Two curves on two halves of the clip: each half its own dial's.
  const Out two = denoise_(*m, k, 1, {branch(b, {1, 1, 0, 0}, 0.0f, 1),
                                      branch(b2, {0, 0, 1, 1}, 0.0f, 1)});
  const std::vector<float> d0 = frame_rows_(two.v, 0, per, PE);
  const std::vector<float> d3 = frame_rows_(two.v, 3, per, PE);
  std::printf("  two curves: frame 0 %.3g from dial 1's, frame 3 %.3g from "
              "dial 2's, %d extra forwards\n",
              rel_(d0, frame_rows_(full.v, 0, per, PE)),
              rel_(d3, frame_rows_(full2.v, 3, per, PE)),
              two.rep.dial_forwards);
  EXPECT_TRUE(two.rep.dial_forwards == 2 &&
              rel_(d0, frame_rows_(full.v, 0, per, PE)) < 1e-6 &&
              rel_(d3, frame_rows_(full2.v, 3, per, PE)) < 1e-6 &&
              two.a == base.a);
  // Blend steps: 3 steps, the curve on the first; 3 + 1 forwards, the
  // bar told so from the first.
  const Out k1 = denoise_(*m, k, 3, {branch(b, {0, 0, 1, 1}, 0.0f, 1)});
  bool units = k1.units.size() == 4;
  for (std::size_t i = 0; units && i < k1.units.size(); ++i) {
    units = k1.units[i].first == (int)i + 1 && k1.units[i].second == 4;
  }
  std::printf("  3 steps, blended on the first: %d forwards + %d, "
              "progress %s\n", k1.rep.forwards, k1.rep.dial_forwards,
              units ? "1..4 of 4" : "WRONG");
  EXPECT_TRUE(k1.ok && k1.rep.forwards == 3 && k1.rep.dial_forwards == 1 &&
              units);
  m.reset();
  fs::remove(file);
}

// MEMORY. A dial branch's forward makes its own velocity buffer, and the
// undialled step's is given back before it: a dialled step's device peak
// is an undialled step's. On the host the undialled velocity is kept
// beside the step's only where two branches share a step, and
// denoise_host_bytes books what the loop holds either way.
TEST(minimax_h3_dials, a_dialled_step_holds_what_it_books)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const fs::path file = tmp_("model.safetensors");
  ASSERT_TRUE(write_model_(file, nullptr));
  H3::Config c;
  std::unique_ptr<H3> m = load_(file, mc, &c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { fs::remove(file); return; }
  // ~18k video rows: a velocity buffer of 3.5 MB, a host state of 7.
  Clip k;
  k.latf = 8;
  k.lath = 96;
  k.latw = 96;
  ASSERT_TRUE(clip_(mc, c, &k));
  const metal_compute::SharedBuffer b = m->text_shift_buffer(
      direction_(7, 0.7f));
  const metal_compute::SharedBuffer b2 = m->text_shift_buffer(
      direction_(8, 0.5f));
  auto branch = [&](const metal_compute::SharedBuffer& sh,
                    std::vector<float> video) {
    DialBranch br;
    br.label = "curve";
    br.shift = &sh;
    br.steps = 1;
    br.video = std::move(video);
    br.audio.assign((std::size_t)k.alat, 0.0f);
    return br;
  };
  const std::vector<float> lo = {1, 1, 1, 1, 0, 0, 0, 0};
  const std::vector<float> hi = {0, 0, 0, 0, 1, 1, 1, 1};
  const std::vector<DialBranch> one = {branch(b, lo)};
  const std::vector<DialBranch> two = {branch(b, lo), branch(b2, hi)};
  std::vector<float> v(k.v0.size()), a(k.a0.size());
  struct Peak {
    double host = 0.0, device = 0.0;
    int dial_forwards = 0;
    bool ok = false;
  };
  auto run = [&](const std::vector<DialBranch>& br) {
    v = k.v0;
    a = k.a0;
    DenoiseRequest q;
    q.dit = m.get();
    q.layout = &k.L;
    q.text = &k.text;
    q.video = v.data();
    q.audio = a.data();
    // Two steps, the branches on the first: the re-noising update draws
    // its fresh noise there (the last step takes x0 and draws none), so
    // denoise_host_bytes' second state is real and the bound is tight.
    q.num_steps = 2;
    q.seed = 7;
    q.update = StepUpdate::kRenoise;
    DenoiseReport rep;
    q.report = &rep;
    q.dial_branches = br;
    auto heap = []() {
      malloc_statistics_t st{};
      malloc_zone_statistics(nullptr, &st);
      return (std::int64_t)st.size_in_use;
    };
    const std::int64_t h0 = heap();
    const std::int64_t d0 = (std::int64_t)mc->memory_budget().allocated;
    std::atomic<bool> done{false};
    std::atomic<std::int64_t> hp{h0}, dp{d0};
    std::thread sampler([&]() {
      while (!done.load()) {
        const std::int64_t h = heap();
        const std::int64_t d = (std::int64_t)mc->memory_budget().allocated;
        if (h > hp.load()) { hp.store(h); }
        if (d > dp.load()) { dp.store(d); }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
    std::string err;
    Peak p;
    p.ok = denoise(q, &err);
    done.store(true);
    sampler.join();
    if (!p.ok) { std::printf("  denoise: %s\n", err.c_str()); }
    p.host = (double)(hp.load() - h0);
    p.device = (double)(dp.load() - d0);
    p.dial_forwards = rep.dial_forwards;
    return p;
  };
  (void)run({});   // the scratch built, the routes settled
  const Peak none = run({});
  const Peak p1 = run(one);
  const Peak p2 = run(two);
  ASSERT_TRUE(none.ok && p1.ok && p2.ok);
  EXPECT_TRUE(p1.dial_forwards == 1 && p2.dial_forwards == 2);
  const double vel =
      (double)k.L.num_video_rows * c.video_patch_elems() * 2 +
      (double)k.L.num_audio_rows * c.audio_channels * 2;
  const double MB = 1048576.0;
  std::printf("  %d video rows (a velocity buffer %.2f MB): device peak "
              "growth undialled %.2f MB, one branch %.2f, two %.2f\n",
              k.L.num_video_rows, vel / MB, none.device / MB,
              p1.device / MB, p2.device / MB);
  EXPECT_TRUE(k.L.num_video_rows >= 18000);
  EXPECT_TRUE(p1.device <= none.device + vel / 2);
  EXPECT_TRUE(p2.device <= none.device + vel / 2);
  const double s0 = (double)denoise_host_bytes(k.L, c, false, false);
  const double s1 = (double)denoise_host_bytes(k.L, c, false, false, &one);
  const double s2 = (double)denoise_host_bytes(k.L, c, false, false, &two);
  std::printf("  host heap growth / booked: one branch %.2f / %.2f MB, two "
              "%.2f / %.2f MB\n", p1.host / MB, s1 / MB, p2.host / MB,
              s2 / MB);
  EXPECT_TRUE(!dial_branches_share_a_step(one) &&
              dial_branches_share_a_step(two) && s1 == s0 && s2 > s1);
  // Covered, and not by a wide margin.
  const std::pair<double, double> arms[] = {{p1.host, s1}, {p2.host, s2}};
  for (const auto& [grew, said] : arms) {
    EXPECT_TRUE(grew <= said * 1.05 + MB);
    EXPECT_TRUE(grew >= said * 0.75);
  }
  m.reset();
  fs::remove(file);
}

// minimax-h3-model-config's `dials`: the rows reach the beat (a pasted
// rack string too), and a row this cannot run fails the config.
TEST(minimax_h3_dials, the_model_config_carries_the_rows)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "dials", FlexData::make_string(
                   R"([{"dial": "dials/h3_aura.safetensors", "strength": 2,)"
                   R"( "time": {"type": "keys", "points": [[0, 0], [1, 1]],)"
                   R"( "audio": "protect", "steps": 6}}])"));
  cfg.as_object().insert_or_assign("dial_dir",
                                   FlexData::make_string("/tmp/dials"));
  MiniMaxH3ModelConfigStage st(&sess, "cfg", {}, cfg);
  const FlexData beat = st.resolved_config();
  ASSERT_TRUE(beat.is_object() && beat.as_object().contains("dials"));
  std::vector<h3::DialRow> rows;
  std::string err;
  ASSERT_TRUE(h3::parse_dial_rows(beat.as_object().at("dials"), &rows, &err));
  EXPECT_TRUE(rows.size() == 1 && rows[0].timed && rows[0].time.steps == 6 &&
              rows[0].time.protect_audio &&
              std::string(beat.as_object().at("dial_dir").as_string()) ==
                  "/tmp/dials");
  FlexData bad = FlexData::make_object();
  bad.as_object().insert_or_assign(
      "dials", FlexData::from_json(R"([{"dial": "x", "strength": 1,)"
                                   R"( "hold": true}])"));
  MiniMaxH3ModelConfigStage sb(&sess, "bad", {}, bad);
  EXPECT_TRUE(!sb.config_error().empty());
  // Unset: nothing in the beat.
  MiniMaxH3ModelConfigStage none(&sess, "none", {}, FlexData::make_object());
  EXPECT_TRUE(!none.resolved_config().as_object().contains("dials"));
}
