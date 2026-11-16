// MiniMax-H3's WHOLE-SEQUENCE block over long sequences: the path a
// forward takes when its blocks cannot run in row bands (the Neural
// Engine's share, VDN's branch, streaming attention, a rank told no
// prompt bands), held to the banded block it must equal.
//
// Its thread-an-element passes count their elements in an int, and the
// feed-forward's SwiGLU ran as one grid of rows x ffn threads: past 2^31
// outputs (from row 149,796 at H3's 14,336) it read and wrote other rows, and
// past 2^32 the grid could not hold it at all. They run in row bands now
// (MetalMiniMaxH3Transformer::elt_rows).
//
// Self-contained: writes its own checkpoint -- H3's block at a narrow
// hidden width and H3's own ffn, the one width the line is drawn at. Its
// ffn is wider than its q|k|v (14,336 against 768), which the released
// model's is not: the scratch the SwiGLU writes into was sized for the
// q|k|v alone, and both paths wrote past it until it held the wider.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <algorithm>
#include <climits>
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
#include <unistd.h>
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
write_st_(const fs::path& p, const std::vector<T>& ts)
{
  std::string hdr = "{";
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
         ("vpipe-h3-whole-" + std::to_string(::getpid()) + "-" + name);
}

struct Env {
  const char* k;
  Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
  ~Env() { ::unsetenv(k); }
};

// A released-layout H3 checkpoint: hidden 256, two heads of 128, ffn
// `F`, text 256, `layers` main blocks and one refiner block.
bool
write_model_(const fs::path& p, int F, int layers)
{
  const int H = 256, A = 256, TXT = 256, FREQ = 256, TH = 64, M = 64;
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
  add(bf16_("condition_proj.bias", {H}, rnd(H, 0.01f)));
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

// One forward of `m` over a packed (text | audio | video) sequence of
// `latf` latent frames of lath x latw; the video velocity, or empty.
struct Geometry {
  int latf = 2, lath = 8, latw = 8, naud = 8, ntext = 8;
};

struct Forward {
  h3::PackedLayout lay;
  metal_compute::SharedBuffer video, audio, text;
  std::vector<float> uniq;
  std::vector<int> ridx;
  int seq = 0;
};

bool
prepare_(metal_compute::MetalCompute* mc, const H3::Config& c,
         const Geometry& g, Forward* f)
{
  const std::vector<int> tags((std::size_t)g.ntext, h3::kTextTag);
  if (!h3::build_packed_sequence(tags, g.latf, g.lath, g.latw, g.naud,
                                 c.patch_h, c.patch_w, h3::kAudioChannels,
                                 {h3::Anchor::kFirst}, &f->lay)) {
    return false;
  }
  auto input = [&](std::size_t n, float k) {
    metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = rne_(std::sin((float)(i % 100003) * k) * 0.5f);
    }
    return b;
  };
  f->video = input(f->lay.video_indices.size() *
                       (std::size_t)c.video_patch_elems(), 0.017f);
  f->audio = input((std::size_t)f->lay.num_audio_rows *
                       (std::size_t)c.audio_channels, 0.031f);
  f->text = input((std::size_t)g.ntext * c.text_dim, 0.005f);
  h3::build_row_timesteps(f->lay, 0.85f, 0.75f, 1.0f, &f->uniq, &f->ridx);
  f->seq = (int)f->ridx.size();
  return !f->video.empty() && !f->audio.empty() && !f->text.empty();
}

std::vector<std::uint16_t>
forward_(H3& m, const Forward& f)
{
  H3::Step st;
  st.video = &f.video;
  st.audio = &f.audio;
  st.text = &f.text;
  st.layout = &f.lay;
  st.timesteps = &f.uniq;
  st.row_timestep_index = &f.ridx;
  std::string err;
  const H3::Velocity out = m.forward(st, &err);
  if (out.empty()) {
    std::printf("  forward: %s\n", err.c_str());
    return {};
  }
  const auto* p = static_cast<const std::uint16_t*>(out.video.contents());
  return std::vector<std::uint16_t>(p, p + out.video.byte_size() / 2);
}

double
rel_(const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& y)
{
  if (x.empty() || x.size() != y.size()) { return 1e9; }
  double d = 0.0, n = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double a = bf16f_(x[i]), b = bf16f_(y[i]);
    d += (a - b) * (a - b);
    n += b * b;
  }
  return std::sqrt(d / std::max(n, 1e-30));
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

}  // namespace

// The whole block's line, and the row bands its elementwise passes take,
// at H3's own widths (model-free).
TEST(minimax_h3_whole, its_line_is_drawn_by_what_still_counts_in_32_bits)
{
  H3::Config c;
  c.hidden = 5376;
  c.n_heads = 56;
  c.head_dim = 128;
  c.ffn = 14336;
  // The residual stream's 2^31 elements: past them no whole block ran.
  EXPECT_TRUE(H3::whole_rows_max(c, false) == 399457);
  // VDN's branch reads the fused q|k|v (21,504 a row) unsigned.
  EXPECT_TRUE(H3::whole_rows_max(c, true) == 199728);
  // A pass over the feed-forward's [rows, 14336] in bands under 2^31.
  EXPECT_TRUE(H3::elt_rows(14336) == 149796);
  EXPECT_TRUE(H3::elt_rows(5376) == 399457);
  {
    Env e("VPIPE_H3_ELT_ROWS", "37");
    EXPECT_TRUE(H3::elt_rows(14336) == 37);
  }
  // The report's draft: a 15 s video reference at 1280 x 720 behind a
  // 12,010-row prompt, 218,406 rows. It ran whole (the prompt passed a
  // band); one Mac bands it now, whatever the prompt's length.
  EXPECT_TRUE(H3::band_for(c, 218406, 12010) > 0);
  EXPECT_TRUE(H3::text_band_rows(c, 12010, H3::band_for(c, 218406, 12010))
              > 0);
}

// The row bands' offsets: a whole forward with every elementwise pass cut
// into bands of 37 rows -- and of 1 -- is the one-grid forward, bit for
// bit (the modulation's and gate's row indices, the residual's and the
// SwiGLU's rows each start at their band).
TEST(minimax_h3_whole, elementwise_row_bands_are_one_grid)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  Env whole("VPIPE_H3_BAND_ROWS", "0");
  const fs::path file = tmp_("small.safetensors");
  ASSERT_TRUE(write_model_(file, 512, 2));
  H3::Config c;
  std::unique_ptr<H3> m = load_(file, mc, &c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { fs::remove(file); return; }
  Geometry g;
  g.latf = 3; g.lath = 16; g.latw = 16; g.naud = 16; g.ntext = 24;
  Forward f;
  ASSERT_TRUE(prepare_(mc, c, g, &f));
  const std::vector<std::uint16_t> one = forward_(*m, f);
  const bool was_whole = m->last_band_rows() == 0;
  std::vector<std::uint16_t> b37, b1;
  {
    Env e("VPIPE_H3_ELT_ROWS", "37");
    b37 = forward_(*m, f);
  }
  {
    Env e("VPIPE_H3_ELT_ROWS", "1");
    b1 = forward_(*m, f);
  }
  std::printf("  %d rows: bands of 37 %s (rel %.3g), of 1 %s (rel %.3g)\n",
              f.seq, b37 == one ? "IDENTICAL" : "DIFFERENT", rel_(b37, one),
              b1 == one ? "IDENTICAL" : "DIFFERENT", rel_(b1, one));
  EXPECT_TRUE(was_whole && !one.empty());
  EXPECT_TRUE(b37 == one && b1 == one);
  m.reset();
  fs::remove(file);
}

// A WHOLE-SEQUENCE forward past both lines -- 2^31 SwiGLU outputs (row
// 149,796) and 2^32 (row 299,593, past which one grid could not hold the
// pass) -- equal to the banded forward. ~27 GB of activations whole.
TEST(minimax_h3_whole, a_long_whole_sequence_is_the_banded_one)
{
  if (std::getenv("VPIPE_H3_BIG_SEQ") == nullptr) {
    std::printf("[ SKIP     ] set VPIPE_H3_BIG_SEQ=1 (allocates ~28 GB)\n");
    return;
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { EXPECT_TRUE(false); return; }
  Env noqmm("VPIPE_H3_NO_QMM_AUTOTUNE", "1");
  Env notile("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1");
  const int F = 14336;                       // H3's ffn
  const fs::path file = tmp_("long.safetensors");
  ASSERT_TRUE(write_model_(file, F, 1));
  H3::Config c;
  std::unique_ptr<H3> m = load_(file, mc, &c);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { fs::remove(file); return; }
  // 37 latent frames of 64 x 128 tokens: 303,104 video rows.
  // VPIPE_H3_WHOLE_FRAMES / _BAND run a shorter clip in smaller bands
  // (the comparison without the length).
  Geometry g;
  g.latf = 37; g.lath = 128; g.latw = 256; g.naud = 64; g.ntext = 8;
  const char* fr = std::getenv("VPIPE_H3_WHOLE_FRAMES");
  const char* bd = std::getenv("VPIPE_H3_WHOLE_BAND");
  if (fr != nullptr) { g.latf = std::max(1, std::atoi(fr)); }
  Forward f;
  ASSERT_TRUE(prepare_(mc, c, g, &f));
  const long long past31 = (1LL << 31) / F, past32 = (1LL << 32) / F;
  std::printf("  %d rows (the SwiGLU's one grid passes 2^31 in row %lld, "
              "2^32 in row %lld)\n", f.seq, past31, past32);
  if (fr == nullptr) { ASSERT_TRUE(f.seq > past32); }
  std::vector<std::uint16_t> whole, banded;
  int whole_band = -1, band = -1;
  {
    Env w("VPIPE_H3_BAND_ROWS", "0");
    whole = forward_(*m, f);
    whole_band = m->last_band_rows();
  }
  (void)m->release_forward_scratch();
  {
    Env b("VPIPE_H3_BAND_ROWS", bd != nullptr ? bd : "8192");
    banded = forward_(*m, f);
    band = m->last_band_rows();
  }
  std::size_t first = whole.size();
  for (std::size_t i = 0; i < std::min(whole.size(), banded.size()); ++i) {
    if (whole[i] != banded[i]) { first = i; break; }
  }
  std::printf("  whole (band %d) vs banded (band %d): %s, rel %.3g, first "
              "difference at video row %lld\n", whole_band, band,
              whole == banded ? "IDENTICAL" : "DIFFERENT",
              rel_(whole, banded),
              first < whole.size()
                  ? (long long)(first / (std::size_t)c.video_patch_elems())
                  : -1LL);
  EXPECT_TRUE(whole_band == 0 && band > 0);
  EXPECT_TRUE(!whole.empty() && whole == banded);
  m.reset();
  fs::remove(file);
}
