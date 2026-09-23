// Qwen-Image-2.1's AutoencoderKLQwenImage21, which runs on the same
// MetalKrea2Vae as Krea-2 / Qwen-Image-Edit after the generalization that
// gave it a variable level count, asymmetric encoder/decoder widths, RGBA
// channels and the `is_residual` shortcuts.
//
// Three tiers, and they check different things on purpose:
//
//  * config   the real vae/config.json parses into the Config the rest of
//             this file assumes. Cheap, no GPU, no weights -- and it is
//             the check that matters most, because a half-read config
//             (four levels instead of five, base_dim standing in for
//             decoder_base_dim) asks for tensors that EXIST at widths
//             that do not, and the failure is a plausible picture.
//  * kernel   the two parameter-free shortcut kernels against a CPU
//             oracle, at small shapes, with no model on disk. NOTE WHAT
//             THIS DOES AND DOES NOT PROVE: the oracle re-states the same
//             closed-form index mapping the kernel implements, so it
//             catches kernel bugs -- bounds, the ulong index discipline,
//             the accumulate-into-out contract, a wrong grid -- and it
//             CANNOT catch a closed form that disagrees with diffusers.
//             That is the golden's job below.
//  * model    the real checkpoint loads, and a round trip through
//             encode+decode comes back as the same picture.
//
// Env: VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH = model root (uses <root>/vae),
// VPIPE_QWEN_IMAGE21_GOLDEN = golden dir. Each tier skips its own vars.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// minitest's ASSERT_TRUE records a failure and CONTINUES, so anything
// guarding a buffer read needs its own return.
// `cond` is evaluated ONCE. An ASSERT_TRUE(cond) followed by
// if (!(cond)) evaluates it twice, which is invisible for a
// comparison and doubles the work for a call -- a model loaded, or a
// forward run, twice per use.
#define Q21_REQUIRE(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
model_root_()
{
  const char* r = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH");
  return (r != nullptr && *r != '\0') ? std::string(r) : std::string();
}

std::string
golden_dir_()
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  return (g != nullptr && *g != '\0') ? std::string(g) : std::string();
}

std::vector<float>
read_f32_(const std::string& path)
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

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Write vpipe's own result beside the golden when VPIPE_QWEN_IMAGE21_DUMP
// names a directory. This exists for BISECTING a numeric gap: with the
// output on disk it can be compared against reference variants (one
// component disabled at a time) without rebuilding anything.
void
dump_(const char* name, const std::vector<float>& v)
{
  const char* d = std::getenv("VPIPE_QWEN_IMAGE21_DUMP");
  if (d == nullptr || *d == '\0') { return; }
  std::error_code ec;
  std::filesystem::create_directories(d, ec);
  std::ofstream out(std::filesystem::path(d) / (std::string(name) + ".f32"),
                    std::ios::binary);
  if (out) {
    out.write(reinterpret_cast<const char*>(v.data()),
              (std::streamsize)(v.size() * sizeof(float)));
  }
}

// The checkpoint's own config, or a Null FlexData when it is not here.
FlexData
vae_config_json_()
{
  const std::string root = model_root_();
  if (root.empty()) { return {}; }
  std::ifstream in(std::filesystem::path(root) / "vae" / "config.json");
  if (!in) { return {}; }
  try {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) { return fd; }
  } catch (...) {
  }
  return {};
}

}  // namespace

// ---- config ---------------------------------------------------------

TEST(qwen_image_21_vae, config_from_json_reads_every_new_key)
{
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }          // model not on this box

  MetalKrea2Vae::Config cfg;                // Krea-2 defaults
  MetalKrea2Vae::config_from_json(fd, &cfg);

  // FIVE levels, and the spatial factor follows from the LENGTH rather
  // than from a separate key -- the default {1,2,4,4} must have been
  // replaced wholesale, not merged into.
  EXPECT_TRUE(cfg.levels() == 5);
  EXPECT_TRUE(cfg.spatial_factor() == 16);
  EXPECT_TRUE(cfg.dim_mult.size() == 5);
  if (cfg.dim_mult.size() == 5) {
    EXPECT_TRUE(cfg.dim_mult[0] == 1 && cfg.dim_mult[1] == 2 &&
                cfg.dim_mult[2] == 4 && cfg.dim_mult[3] == 8 &&
                cfg.dim_mult[4] == 8);
  }
  EXPECT_TRUE(cfg.z_dim == 64);
  // ASYMMETRIC: the encoder is 96 wide and the decoder 144. Reading the
  // decoder at base_dim is the bug that under-books every byte estimate
  // by a third.
  EXPECT_TRUE(cfg.base_dim == 96);
  EXPECT_TRUE(cfg.decoder_base_dim == 144);
  EXPECT_TRUE(cfg.dec_base() == 144);
  // RGBA on both sides.
  EXPECT_TRUE(cfg.in_channels == 4);
  EXPECT_TRUE(cfg.out_channels == 4);
  EXPECT_TRUE(cfg.is_residual);
  EXPECT_TRUE(cfg.num_res_blocks == 2);
  // The temporal flags survive on a still because the shortcut's channel
  // mapping needs them (see the kernels).
  EXPECT_TRUE(cfg.temperal_downsample.size() == 4);
  if (cfg.temperal_downsample.size() == 4) {
    EXPECT_TRUE(cfg.temperal_downsample[0] == 0);
    EXPECT_TRUE(cfg.temperal_downsample[1] == 1);
    EXPECT_TRUE(cfg.temperal_downsample[2] == 1);
    EXPECT_TRUE(cfg.temperal_downsample[3] == 1);
  }
  EXPECT_TRUE(cfg.latents_mean.size() == 64);
  EXPECT_TRUE(cfg.latents_std.size() == 64);
}

// A Krea-2 config must still read as Krea-2 through the same function --
// the generalization's whole risk is that it moved four shipping models.
TEST(qwen_image_21_vae, config_defaults_are_unchanged_for_four_levels)
{
  MetalKrea2Vae::Config cfg;
  EXPECT_TRUE(cfg.levels() == 4);
  EXPECT_TRUE(cfg.spatial_factor() == 8);
  EXPECT_TRUE(cfg.dec_base() == 96);        // sentinel resolves to base_dim
  EXPECT_TRUE(cfg.in_channels == 3 && cfg.out_channels == 3);
  EXPECT_FALSE(cfg.is_residual);

  // An object carrying only the keys the old reader knew leaves the rest
  // alone, so an existing checkpoint cannot acquire five levels or RGBA
  // by accident.
  FlexData fd = FlexData::make_object();
  {
    auto o = fd.as_object();
    o.insert_or_assign("z_dim", FlexData::make_int(16));
    o.insert_or_assign("base_dim", FlexData::make_int(96));
    o.insert_or_assign("num_res_blocks", FlexData::make_int(2));
  }
  MetalKrea2Vae::Config got;
  MetalKrea2Vae::config_from_json(fd, &got);
  EXPECT_TRUE(got.levels() == 4);
  EXPECT_TRUE(got.spatial_factor() == 8);
  EXPECT_TRUE(got.dec_base() == 96);
  EXPECT_FALSE(got.is_residual);
  EXPECT_TRUE(got.out_channels == 3);
}

// ---- the shortcut kernels, against a CPU oracle ---------------------

namespace {

// Dispatch one of the two shortcut kernels over host vectors and return
// `out` (which the kernel ACCUMULATES into, so it is seeded non-zero
// here -- a kernel that assigned instead of adding would pass a
// zero-seeded test and drop the main path's result in the real decode).
bool
run_kernel_(MetalCompute* mc, const char* fn, const std::vector<float>& in,
            std::vector<float>& out, const std::vector<int>& consts)
{
  metal_compute::ComputeLibrary lib = mc->load_library("llm_elementwise");
  metal_compute::ComputeFunction f = lib.function(fn);
  if (!f.valid()) { return false; }
  SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
  SharedBuffer ob = mc->make_shared_buffer(out.size() * 2);
  if (ib.empty() || ob.empty()) { return false; }
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < in.size(); ++i) { d[i] = (_Float16)in[i]; }
    auto* o = static_cast<_Float16*>(ob.contents());
    for (std::size_t i = 0; i < out.size(); ++i) { o[i] = (_Float16)out[i]; }
  }
  metal_compute::CommandStream stream = mc->make_command_stream();
  {
    metal_compute::ComputeEncoder enc = stream.begin_compute();
    if (!enc.valid()) { return false; }
    enc.set_function(f);
    enc.set_buffer(0, ib);
    enc.set_buffer(1, ob);
    for (std::size_t i = 0; i < consts.size(); ++i) {
      enc.set_constant((int)i + 2, consts[i]);
    }
    const int Cout = consts[3];
    const unsigned rows = (unsigned)(out.size() / (std::size_t)Cout);
    enc.dispatch({(unsigned)Cout, rows, 1}, {256, 1, 1});
  }
  metal_compute::CommandStream::Fence fence = stream.commit();
  if (!fence.valid()) { return false; }
  fence.wait();
  const auto* op = static_cast<const _Float16*>(ob.contents());
  for (std::size_t i = 0; i < out.size(); ++i) { out[i] = (float)op[i]; }
  return true;
}

}  // namespace

TEST(qwen_image_21_vae, dup_up_shortcut_matches_cpu)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // The three decoder shapes the real checkpoint uses, by (Cin, Cout, ft):
  // level 0/1 keep the width with a temporal factor, level 2 halves it
  // with one, level 3 halves it without. Each picks a different channel
  // out of the source, which is the whole risk.
  struct Case { int Cin, Cout, ft; };
  const Case cases[] = {{16, 16, 2}, {16, 8, 2}, {8, 4, 1}};
  const int H = 5, W = 3;                   // deliberately not square/even

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);

  for (const Case& c : cases) {
    std::vector<float> in((std::size_t)H * W * c.Cin);
    for (float& v : in) { v = u(rng); }
    const std::size_t orows = (std::size_t)(2 * H) * (2 * W);
    std::vector<float> seed(orows * (std::size_t)c.Cout);
    for (float& v : seed) { v = u(rng); }

    // CPU oracle, seeded with the same values so the ACCUMULATE is part
    // of what is compared.
    std::vector<float> want = seed;
    const int repeats = (c.Cout * c.ft * 4) / c.Cin;
    for (int oy = 0; oy < 2 * H; ++oy) {
      for (int ox = 0; ox < 2 * W; ++ox) {
        for (int o = 0; o < c.Cout; ++o) {
          const int j = ((o * c.ft + (c.ft - 1)) * 2 + (oy & 1)) * 2 + (ox & 1);
          const int sc = j / repeats;
          const std::size_t si =
              ((std::size_t)(oy / 2) * W + (ox / 2)) * c.Cin + sc;
          const std::size_t oi =
              ((std::size_t)oy * (2 * W) + ox) * c.Cout + o;
          want[oi] += in[si];
        }
      }
    }

    std::vector<float> got = seed;
    const bool ok = run_kernel_(mc, "vae_res_dup_up_f16", in, got,
                                {H, W, c.Cin, c.Cout, c.ft});
    Q21_REQUIRE(ok);
    const double r = rel_l2_(got.data(), want.data(), want.size());
    std::printf("[qwen_image_21_vae] dup_up Cin=%d Cout=%d ft=%d rel-L2 %.3e\n",
                c.Cin, c.Cout, c.ft, r);
    EXPECT_TRUE(r < 2e-3);                  // f16 round trip only
  }
}

TEST(qwen_image_21_vae, avg_down_shortcut_matches_cpu)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // The encoder's shapes: level 0 pools without a temporal factor, the
  // middle levels widen WITH one (so half their output channels come out
  // zero -- the padded frame), and the last level neither pools nor
  // widens, collapsing to a plain residual add.
  struct Case { int Cin, Cout, ft, fs; };
  const Case cases[] = {{8, 8, 1, 2}, {8, 16, 2, 2}, {16, 16, 1, 1}};
  const int H = 8, W = 6;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);

  for (const Case& c : cases) {
    std::vector<float> in((std::size_t)H * W * c.Cin);
    for (float& v : in) { v = u(rng); }
    const int OH = H / c.fs, OW = W / c.fs;
    std::vector<float> seed((std::size_t)OH * OW * c.Cout);
    for (float& v : seed) { v = u(rng); }

    std::vector<float> want = seed;
    const int factor = c.ft * c.fs * c.fs;
    const int G = (c.Cin * factor) / c.Cout;
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        for (int o = 0; o < c.Cout; ++o) {
          float acc = 0.0f;
          for (int g = 0; g < G; ++g) {
            const int n = o * G + g;
            const int sc = n / factor;
            const int rem = n % factor;
            const int t = rem / (c.fs * c.fs);
            if (t != c.ft - 1) { continue; }       // zero-padded frame
            const int sy = (rem / c.fs) % c.fs;
            const int sx = rem % c.fs;
            const std::size_t si =
                ((std::size_t)(oy * c.fs + sy) * W + (ox * c.fs + sx)) *
                    c.Cin + sc;
            acc += in[si];
          }
          want[((std::size_t)oy * OW + ox) * c.Cout + o] += acc / (float)G;
        }
      }
    }

    std::vector<float> got = seed;
    const bool ok = run_kernel_(mc, "vae_res_avg_down_f16", in, got,
                                {H, W, c.Cin, c.Cout, c.ft, c.fs});
    Q21_REQUIRE(ok);
    const double r = rel_l2_(got.data(), want.data(), want.size());
    std::printf(
        "[qwen_image_21_vae] avg_down Cin=%d Cout=%d ft=%d fs=%d rel-L2 %.3e\n",
        c.Cin, c.Cout, c.ft, c.fs, r);
    EXPECT_TRUE(r < 2e-3);
  }
}

// ---- the real checkpoint --------------------------------------------

TEST(qwen_image_21_vae, loads_and_round_trips)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/true);
  Q21_REQUIRE(m != nullptr);
  Q21_REQUIRE(m->has_encoder());
  EXPECT_TRUE(m->config().spatial_factor() == 16);

  // A picture the encoder can actually round-trip: smooth, so the test
  // is about the architecture rather than about high-frequency loss.
  // 64x64 is four latent cells a side at /16.
  const int H = 64, W = 64;
  const int IC = cfg.in_channels;
  std::vector<float> img((std::size_t)IC * H * W);
  for (int c = 0; c < IC; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float fx = (float)x / (float)(W - 1);
        const float fy = (float)y / (float)(H - 1);
        img[((std::size_t)c * H + y) * W + x] =
            0.6f * std::sin(6.2831853f * (fx + 0.13f * c)) * std::cos(
                3.1415927f * fy);
      }
    }
  }
  SharedBuffer ib = mc->make_shared_buffer(img.size() * 2);
  Q21_REQUIRE(!ib.empty());
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < img.size(); ++i) { d[i] = (_Float16)img[i]; }
  }

  SharedBuffer lat = m->encode(ib, H, W);
  Q21_REQUIRE(!lat.empty());
  const int hL = H / 16, wL = W / 16;
  // z_dim channels at the /16 grid, and nothing else would be this size.
  Q21_REQUIRE(lat.byte_size() >=
              (std::size_t)cfg.z_dim * hL * wL * 2);

  // encode() whitens; decode() wants the raw latent back.
  SharedBuffer raw = m->unwhiten(lat, hL, wL);
  Q21_REQUIRE(!raw.empty());
  std::string err;
  SharedBuffer rgb = m->decode(raw, hL, wL, &err);
  if (rgb.empty()) {
    std::printf("[qwen_image_21_vae] decode failed: %s\n", err.c_str());
  }
  Q21_REQUIRE(!rgb.empty());
  Q21_REQUIRE(rgb.byte_size() >= (std::size_t)cfg.out_channels * H * W * 2);

  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  std::vector<float> got((std::size_t)cfg.out_channels * H * W);
  bool finite = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    got[i] = (float)rp[i];
    if (!std::isfinite(got[i])) { finite = false; }
  }
  EXPECT_TRUE(finite);

  // PSNR over the channels the input actually carried.
  double se = 0.0;
  const std::size_t n = (std::size_t)IC * H * W;
  for (std::size_t i = 0; i < n && i < got.size(); ++i) {
    const double d = (double)got[i] - (double)img[i];
    se += d * d;
  }
  const double mse = se / (double)n;
  const double psnr = mse > 0.0 ? 10.0 * std::log10(4.0 / mse) : 99.0;
  std::printf("[qwen_image_21_vae] round trip %dx%d -> [%d,%d,%d] PSNR %.2f dB\n",
              H, W, cfg.z_dim, hL, wL, psnr);
  // A loose floor on purpose: this says the architecture is wired, not
  // that it is numerically right. The golden below is what pins that.
  EXPECT_TRUE(psnr > 18.0);
}

TEST(qwen_image_21_vae, decode_matches_golden)
{
  const std::string root = model_root_();
  const std::string gdir = golden_dir_();
  if (root.empty() || gdir.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> lat = read_f32_(gdir + "/g_vae_declat.f32");
  const std::vector<float> want = read_f32_(gdir + "/g_vae_dec.f32");
  if (lat.empty() || want.empty()) { return; }

  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  Q21_REQUIRE(lat.size() % (std::size_t)cfg.z_dim == 0);
  const std::size_t cells = lat.size() / (std::size_t)cfg.z_dim;
  int hL = 1;
  while ((std::size_t)hL * hL < cells) { ++hL; }
  Q21_REQUIRE((std::size_t)hL * hL == cells);

  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/false);
  Q21_REQUIRE(m != nullptr);
  SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
  Q21_REQUIRE(!z.empty());
  {
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
  }
  std::string err;
  SharedBuffer rgb = m->decode(z, hL, hL, &err);
  Q21_REQUIRE(!rgb.empty());
  Q21_REQUIRE(rgb.byte_size() >= want.size() * 2);
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  std::vector<float> got(want.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)rp[i]; }

  dump_("vpipe_vae_dec", got);
  const double r = rel_l2_(got.data(), want.data(), got.size());
  std::printf("[qwen_image_21_vae] decode rel-L2 = %.6f ([%d,%d,%d] -> %dx%d)\n",
              r, cfg.z_dim, hL, hL, hL * 16, hL * 16);
  EXPECT_TRUE(r < 0.05);
}

TEST(qwen_image_21_vae, encode_matches_golden)
{
  const std::string root = model_root_();
  const std::string gdir = golden_dir_();
  if (root.empty() || gdir.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> in   = read_f32_(gdir + "/g_vae_input.f32");
  const std::vector<float> want = read_f32_(gdir + "/g_vae_enc.f32");
  if (in.empty() || want.empty()) { return; }

  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  Q21_REQUIRE(in.size() % (std::size_t)cfg.in_channels == 0);
  const std::size_t hw = in.size() / (std::size_t)cfg.in_channels;
  int H = 1;
  while ((std::size_t)H * H < hw) { ++H; }
  Q21_REQUIRE((std::size_t)H * H == hw);

  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/true);
  Q21_REQUIRE(m != nullptr);
  Q21_REQUIRE(m->has_encoder());
  SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
  Q21_REQUIRE(!ib.empty());
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < in.size(); ++i) { d[i] = (_Float16)in[i]; }
  }
  SharedBuffer lat = m->encode(ib, H, H);
  Q21_REQUIRE(!lat.empty());
  Q21_REQUIRE(lat.byte_size() >= want.size() * 2);
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  std::vector<float> got(want.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)lp[i]; }

  dump_("vpipe_vae_enc", got);
  const double r = rel_l2_(got.data(), want.data(), got.size());
  std::printf("[qwen_image_21_vae] encode rel-L2 = %.6f (%dx%d -> [%d,%d,%d])\n",
              r, H, H, cfg.z_dim, H / 16, H / 16);
  EXPECT_TRUE(r < 0.05);
}

// A TIMED DECODE AT A REAL OUTPUT SIZE.
//
// The golden above decodes whatever the dumper wrote, which is small.
// This one takes the size from VPIPE_QWEN_IMAGE21_VAE_BENCH (pixels,
// e.g. 1024) and times a decode at it, so the conv path can be A/B'd
// where it actually costs something. Random latents: the arithmetic
// does not care and this is a timing instrument, not a correctness
// one -- `decode_matches_golden` is that.
//
// Env-gated and off by default: a 1024 decode is seconds, and the
// suite should not pay for it.
TEST(qwen_image_21_vae, decode_bench)
{
  const char* bench = std::getenv("VPIPE_QWEN_IMAGE21_VAE_BENCH");
  if (bench == nullptr || *bench == '\0') { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);

  const int px = std::atoi(bench) > 0 ? std::atoi(bench) : 1024;
  const int hL = px / cfg.spatial_factor();
  Q21_REQUIRE(hL > 0);
  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/false);
  Q21_REQUIRE(m != nullptr);

  const std::size_t n = (std::size_t)cfg.z_dim * hL * hL;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  Q21_REQUIRE(!z.empty());
  {
    auto* d = static_cast<_Float16*>(z.contents());
    unsigned st = 7u;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f));
    }
  }
  // One warm decode, then the timed pair: the first pays the conv and
  // attention autotunes, which are a property of the box and not of the
  // path under test.
  std::string err;
  SharedBuffer warm = m->decode(z, hL, hL, &err);
  Q21_REQUIRE(!warm.empty());
  warm = SharedBuffer{};

  double best = 0.0;
  for (int i = 0; i < 2; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer rgb = m->decode(z, hL, hL, &err);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    Q21_REQUIRE(!rgb.empty());
    if (best == 0.0 || ms < best) { best = ms; }
  }
  std::printf("[qwen_image_21_vae] decode %dx%d (latent %dx%d, /%d): "
              "%.0f ms\n", px, px, hL, hL, cfg.spatial_factor(), best);
}
