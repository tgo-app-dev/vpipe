// VOSR's colour fix: the release's script corrects what it restored
// against the upscaled picture it restored (`--align_method`, `wavelet`
// by default -- a Gaussian blur's low frequencies, not a wavelet stack).
//
//   color_fix.*       the fix itself, against its definition restated
//                     here in double (no model).
//   vosr_color_fix.*  vae-decode running the method a latent announces,
//                     one picture per latent in order, on the real VAE;
//                     and the restore graph end to end, generate-image
//                     announcing the default and vosr-model-config's
//                     choice. Needs VPIPE_VOSR_TEST_MODEL_PATH,
//                     VPIPE_DINOV2_TEST_MODEL_PATH and VPIPE_VOSR_GOLDEN
//                     (tools/dump_vosr_golden.py), else skips.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/color-fix.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"
#include "stages/vosr-model-config-stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;
namespace fs = std::filesystem;

namespace {

// ---- the fix's definition, in double -----------------------------------

int
refl_(int p, int n)
{
  if (n == 1) { return 0; }
  while (p < 0 || p >= n) { p = p < 0 ? -p : 2 * (n - 1) - p; }
  return p;
}

// cv2.GaussianBlur(img, (0, 0), sigma) on a float plane, written out:
// k[i] k[j] p[refl(y + i - r)][refl(x + j - r)] summed in double.
std::vector<double>
blur_(const std::vector<double>& p, int H, int W, double sigma)
{
  const int n = (int)std::nearbyint(sigma * 8.0 + 1.0) | 1;
  const int r = n / 2;
  std::vector<double> k((std::size_t)n);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = i - r;
    k[(std::size_t)i] = std::exp(-d * d / (2.0 * sigma * sigma));
    s += k[(std::size_t)i];
  }
  for (double& v : k) { v /= s; }
  std::vector<double> t((std::size_t)H * W), o((std::size_t)H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double a = 0.0;
      for (int j = 0; j < n; ++j) {
        a += k[(std::size_t)j] * p[(std::size_t)y * W + refl_(x + j - r, W)];
      }
      t[(std::size_t)y * W + x] = a;
    }
  }
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      double a = 0.0;
      for (int i = 0; i < n; ++i) {
        a += k[(std::size_t)i] * t[(std::size_t)refl_(y + i - r, H) * W + x];
      }
      o[(std::size_t)y * W + x] = a;
    }
  }
  return o;
}

// VOSR's wavelet_color_fix on one plane: clip(low(ref) + x - low(x)).
std::vector<double>
gaussian_def_(const std::vector<double>& x, const std::vector<double>& ref,
              int H, int W, double sigma)
{
  const auto lx = blur_(x, H, W, sigma), lr = blur_(ref, H, W, sigma);
  std::vector<double> o(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    o[i] = std::clamp(lr[i] + x[i] - lx[i], 0.0, 1.0);
  }
  return o;
}

// VOSR's adain_color_fix on one plane.
std::vector<double>
adain_def_(const std::vector<double>& x, const std::vector<double>& ref)
{
  auto st = [](const std::vector<double>& p, double* m, double* s) {
    double a = 0.0;
    for (double v : p) { a += v; }
    *m = a / (double)p.size();
    double q = 0.0;
    for (double v : p) { q += (v - *m) * (v - *m); }
    *s = std::sqrt(q / (double)(p.size() - 1)) + 1e-5;
  };
  double mx, sx, mr, sr;
  st(x, &mx, &sx);
  st(ref, &mr, &sr);
  std::vector<double> o(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    o[i] = std::clamp((x[i] - mx) / sx * sr + mr, 0.0, 1.0);
  }
  return o;
}

// A plane with structure at every scale the blur separates: a slow
// gradient, a mid-frequency ripple and per-pixel grain, in 0..1.
std::vector<double>
plane_(int H, int W, int seed)
{
  std::vector<double> p((std::size_t)H * W);
  std::uint32_t s = 2463534242u + (std::uint32_t)seed * 7919u;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      const double g = (double)(s & 0xffff) / 65535.0 - 0.5;
      const double v = 0.45 + 0.3 * (double)(x + 2 * y) / (W + 2 * H) +
                       0.12 * std::sin(0.37 * x + seed) *
                           std::cos(0.23 * y) +
                       0.08 * g;
      p[(std::size_t)y * W + x] = std::clamp(v, 0.0, 1.0);
    }
  }
  return p;
}

std::vector<float>
to_f_(const std::vector<double>& v)
{
  return std::vector<float>(v.begin(), v.end());
}

double
max_abs_(const std::vector<float>& a, const std::vector<double>& b)
{
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs((double)a[i] - b[i]));
  }
  return m;
}

}  // namespace

// cv2's kernel for a float image at ksize 0: cvRound(8 sigma + 1) | 1
// taps (41 at VOSR's sigma 5), the Gaussian normalised.
TEST(color_fix, the_taps_are_opencvs)
{
  struct Case { double sigma; int n; };
  for (const Case c : {Case{5.0, 41}, Case{2.0, 17}, Case{1.5, 13},
                       Case{0.8, 7}, Case{0.3, 3}}) {
    const auto t = color_fix::gaussian_taps(c.sigma);
    EXPECT_TRUE((int)t.size() == c.n);
    if ((int)t.size() != c.n) {
      std::fprintf(stderr, "[color_fix] sigma %.2f: %zu taps, want %d\n",
                   c.sigma, t.size(), c.n);
      continue;
    }
    const int r = c.n / 2;
    double sum = 0.0, z = 0.0;
    for (int i = 0; i < c.n; ++i) {
      z += std::exp(-(double)(i - r) * (i - r) / (2 * c.sigma * c.sigma));
    }
    double worst = 0.0;
    for (int i = 0; i < c.n; ++i) {
      sum += t[(std::size_t)i];
      const double want =
          std::exp(-(double)(i - r) * (i - r) / (2 * c.sigma * c.sigma)) / z;
      worst = std::max(worst, std::fabs(t[(std::size_t)i] - want));
      EXPECT_TRUE(t[(std::size_t)i] == t[(std::size_t)(c.n - 1 - i)]);
    }
    EXPECT_TRUE(std::fabs(sum - 1.0) < 1e-6);
    // Stored as float, as OpenCV stores them: within half an ulp of the
    // largest tap (~0.9 at sigma 0.3).
    EXPECT_TRUE(worst < 1e-7);
  }
}

// The blur and the fix against the definition, on planes larger and
// smaller than the 41-tap reach -- a 9-pixel side folds the reflection
// several times -- so the edges are what is being checked.
TEST(color_fix, gaussian_is_vosrs_formula)
{
  struct Geo { int H, W; };
  for (const Geo g : {Geo{37, 53}, Geo{17, 9}, Geo{1, 30}, Geo{64, 64}}) {
    const auto x = plane_(g.H, g.W, 1), ref = plane_(g.H, g.W, 2);
    // The blur alone.
    {
      const auto want = blur_(x, g.H, g.W, 5.0);
      std::vector<float> src = to_f_(x), dst(src.size()), tmp(src.size());
      color_fix::gaussian_blur(src.data(), dst.data(), tmp.data(), g.H, g.W,
                               color_fix::gaussian_taps(5.0));
      const double e = max_abs_(dst, want);
      EXPECT_TRUE(e < 2e-6);
      if (!(e < 2e-6)) {
        std::fprintf(stderr, "[color_fix] blur %dx%d off by %.3g\n", g.W,
                     g.H, e);
      }
    }
    // The fix: low(ref) + x - low(x), done as x + low(ref - x).
    for (double sigma : {5.0, 1.5}) {
      const auto want = gaussian_def_(x, ref, g.H, g.W, sigma);
      std::vector<float> got = to_f_(x), r = to_f_(ref), tmp(got.size());
      color_fix::gaussian(got.data(), r.data(), tmp.data(), g.H, g.W,
                          color_fix::gaussian_taps(sigma));
      const double e = max_abs_(got, want);
      EXPECT_TRUE(e < 2e-6);
      if (!(e < 2e-6)) {
        std::fprintf(stderr, "[color_fix] fix %dx%d sigma %.1f off by "
                     "%.3g\n", g.W, g.H, sigma, e);
      }
    }
  }
}

TEST(color_fix, adain_is_vosrs_formula)
{
  const auto x = plane_(40, 70, 3);
  auto ref = plane_(40, 70, 4);
  for (double& v : ref) { v = std::clamp(v * 0.7 + 0.2, 0.0, 1.0); }
  const auto want = adain_def_(x, ref);
  std::vector<float> got = to_f_(x), r = to_f_(ref);
  color_fix::adain(got.data(), r.data(), got.size());
  EXPECT_TRUE(max_abs_(got, want) < 1e-6);
  // A flat plane has no spread to scale: it moves to the reference's
  // mean, and nothing divides by zero.
  std::vector<float> flat(100, 0.5f), fr(100);
  for (int i = 0; i < 100; ++i) { fr[(std::size_t)i] = 0.01f * i; }
  color_fix::adain(flat.data(), fr.data(), flat.size());
  bool finite = true;
  for (float v : flat) { finite = finite && std::isfinite(v); }
  EXPECT_TRUE(finite);
  EXPECT_TRUE(std::fabs(flat[0] - 0.495f) < 1e-4);
}

// ---- through the stages --------------------------------------------------

namespace {

std::vector<std::uint8_t>
read_u8_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<std::uint8_t> out((std::size_t)n);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::vector<float>
read_f32_(const fs::path& p)
{
  const auto b = read_u8_(p);
  std::vector<float> out(b.size() / 4);
  std::memcpy(out.data(), b.data(), out.size() * 4);
  return out;
}

FlexData
obj_(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData o = FlexData::make_object();
  for (const auto& [k, v] : kv) { o.as_object().insert_or_assign(k, v); }
  return o;
}

// The paths and the golden's geometry, or ok == false (skip).
struct Fx {
  std::string root, dino, golden;
  int H = 0, W = 0, lh = 0, lw = 0;
  bool ok = false;
  Fx()
  {
    const char* r = std::getenv("VPIPE_VOSR_TEST_MODEL_PATH");
    const char* d = std::getenv("VPIPE_DINOV2_TEST_MODEL_PATH");
    const char* g = std::getenv("VPIPE_VOSR_GOLDEN");
    if (r == nullptr || d == nullptr || g == nullptr || *r == '\0' ||
        *d == '\0' || *g == '\0') {
      return;
    }
    root = r; dino = d; golden = g;
    std::ifstream in(fs::path(golden) / "manifest.json");
    if (!in) { return; }
    FlexData fd = FlexData::from_json(in);
    if (!fd.is_object()) { return; }
    auto o = fd.as_object();
    auto dims = [&](const char* k, int* a, int* b) {
      if (!o.contains(k)) { return false; }
      FlexData v = o.at(k);
      auto arr = v.as_array();
      if (arr.size() < 3) { return false; }
      *a = (int)arr[1].as_real(0.0);
      *b = (int)arr[2].as_real(0.0);
      return true;
    };
    ok = dims("resized_u8", &H, &W) && dims("lq_latent", &lh, &lw) &&
         H > 0 && W > 0 && lh > 0 && lw > 0;
  }
};

// Emits its beats in order, then ends.
class Beats : public TypedStage<Beats> {
public:
  static constexpr const char* kTypeName = "ut-vosr-color-fix-beats";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<TensorBeatPayload>> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    co_await ctx.write(0, std::move(beats[next++]));
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {{.name = "out", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

// Keeps pictures (U8) and latents (F32, with their sideband).
class Keep : public TypedStage<Keep> {
public:
  static constexpr const char* kTypeName = "ut-vosr-color-fix-keep";
  using TypedStage::TypedStage;
  std::vector<std::vector<std::uint8_t>> pics;
  std::vector<std::vector<float>> latents;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<FlexData> sb;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (tb == nullptr) { co_return; }
    const auto px = tb->materialize_contiguous();
    if (tb->dtype == TensorBeat::DType::U8) {
      pics.emplace_back(px.begin(), px.end());
    } else if (tb->dtype == TensorBeat::DType::F32) {
      std::vector<float> v(px.size() / 4);
      std::memcpy(v.data(), px.data(), v.size() * 4);
      latents.push_back(std::move(v));
      shapes.emplace_back(tb->shape.begin(), tb->shape.end());
      sb.push_back(tb->sideband);
    }
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {{.name = "in", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

std::unique_ptr<TensorBeatPayload>
picture_(const std::vector<std::uint8_t>& px, int H, int W)
{
  auto b = std::make_unique<TensorBeatPayload>();
  b->dtype = TensorBeat::DType::U8;
  b->shape = {3, H, W};
  b->resize_contiguous(px.size());
  std::memcpy(b->as_u8(), px.data(), px.size());
  return b;
}

// A latent, announcing `method` ("" announces nothing).
std::unique_ptr<TensorBeatPayload>
latent_(const std::vector<float>& z, const std::vector<std::int64_t>& shape,
        const std::string& method)
{
  auto b = std::make_unique<TensorBeatPayload>();
  b->dtype = TensorBeat::DType::F32;
  b->shape = shape;
  b->resize_contiguous(z.size());
  std::memcpy(b->as_f32(), z.data(), z.size() * 4);
  if (!method.empty()) {
    b->sideband = obj_(
        {{sideband::kColorRefFrames.data(), FlexData::make_int(1)},
         {sideband::kColorRefMethod.data(), FlexData::make_string(method)},
         {sideband::kColorRefSigma.data(), FlexData::make_real(5.0)}});
  }
  return b;
}

// vae-decode alone: `lats` in, `pics` on its color_reference input when
// given; the pictures it emits.
std::vector<std::vector<std::uint8_t>>
decode_(const std::string& root,
        std::vector<std::unique_ptr<TensorBeatPayload>> lats,
        std::vector<std::unique_ptr<TensorBeatPayload>> pics)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto ls = std::make_unique<Beats>(&sess, "lat", std::vector<InEdge>{},
                                    FlexData::make_object());
  ls->beats = std::move(lats);
  ls->allocate_oports(1);
  Stage* s_lat = pl->insert_stage(std::move(ls));
  Stage* s_pic = nullptr;
  if (!pics.empty()) {
    auto ps = std::make_unique<Beats>(&sess, "pic", std::vector<InEdge>{},
                                      FlexData::make_object());
    ps->beats = std::move(pics);
    ps->allocate_oports(1);
    s_pic = pl->insert_stage(std::move(ps));
  }
  const FlexData cfg = obj_({{"hf_dir", FlexData::make_string(root)},
                             {"autotune", FlexData::make_bool(false)}});
  std::vector<InEdge> in{{s_lat, 0}, {nullptr, 0}, {nullptr, 0}};
  if (s_pic != nullptr) { in[2] = InEdge{s_pic, 0}; }
  Stage* s_vae = pl->insert_stage(
      std::make_unique<VaeDecodeStage>(&sess, "vae", std::move(in), cfg));
  auto* keep = static_cast<Keep*>(pl->insert_stage(std::make_unique<Keep>(
      &sess, "keep", std::vector<InEdge>{{s_vae, 0}},
      FlexData::make_object())));
  {
    PipelineRuntime rt(pl.get(), &sess);
    if (rt.launch()) {
      rt.wait_idle();
      rt.stop();
    }
  }
  return std::move(keep->pics);
}

// The fix restated from an 8-bit decode, rounded as vae-decode rounds.
std::vector<std::uint8_t>
host_fix_(const std::vector<std::uint8_t>& dec,
          const std::vector<std::uint8_t>& pic, int H, int W,
          const char* method)
{
  const std::size_t plane = (std::size_t)H * W;
  std::vector<std::uint8_t> out(dec.size());
  std::vector<float> x(plane), r(plane), tmp(plane);
  const auto taps = color_fix::gaussian_taps(5.0);
  for (int c = 0; c < 3; ++c) {
    const std::size_t at = (std::size_t)c * plane;
    for (std::size_t i = 0; i < plane; ++i) {
      x[i] = dec[at + i] / 255.0f;
      r[i] = pic[at + i] / 255.0f;
    }
    if (std::strcmp(method, "gaussian") == 0) {
      color_fix::gaussian(x.data(), r.data(), tmp.data(), H, W, taps);
    } else {
      color_fix::adain(x.data(), r.data(), plane);
    }
    for (std::size_t i = 0; i < plane; ++i) {
      out[at + i] = (std::uint8_t)std::lround(x[i] * 255.0f);
    }
  }
  return out;
}

struct Diff { int max = 0; double mean = 0.0; };

Diff
diff_(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
{
  Diff d;
  if (a.size() != b.size() || a.empty()) { d.max = 256; return d; }
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int e = std::abs((int)a[i] - (int)b[i]);
    d.max = std::max(d.max, e);
    s += e;
  }
  d.mean = s / (double)a.size();
  return d;
}

// The golden's picture with a colour cast: each channel scaled and
// offset, so a fix against it has somewhere to go.
std::vector<std::uint8_t>
cast_(const std::vector<std::uint8_t>& px, int H, int W, float r, float g,
      float b)
{
  std::vector<std::uint8_t> out(px.size());
  const std::size_t plane = (std::size_t)H * W;
  const float off[3] = {r, g, b};
  for (int c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < plane; ++i) {
      const std::size_t k = (std::size_t)c * plane + i;
      out[k] = (std::uint8_t)std::clamp(
          std::lround(px[k] * 0.85f + off[c]), 0L, 255L);
    }
  }
  return out;
}

}  // namespace

// vae-decode runs what each latent asks for, against ITS picture: one
// picture is read per announcing latent, `none` included, so three
// latents in a row each pair with their own.
TEST(vosr_color_fix, vae_decode_runs_the_method_its_latent_asks)
{
  Fx f;
  if (!f.ok) { return; }
  const auto z = read_f32_(fs::path(f.golden) / "lq_latent.bin");
  const auto px = read_u8_(fs::path(f.golden) / "resized_u8.bin");
  const std::vector<std::int64_t> shape = {16, f.lh, f.lw};
  ASSERT_TRUE(z.size() == (std::size_t)16 * f.lh * f.lw);
  ASSERT_TRUE(px.size() == (std::size_t)3 * f.H * f.W);
  if (z.empty() || px.empty()) { return; }
  const int H = f.lh * 8, W = f.lw * 8;
  ASSERT_TRUE(H == f.H && W == f.W);
  if (H != f.H || W != f.W) { return; }

  // The decode as it is.
  std::vector<std::unique_ptr<TensorBeatPayload>> l0;
  l0.push_back(latent_(z, shape, ""));
  const auto plain = decode_(f.root, std::move(l0), {});
  ASSERT_TRUE(plain.size() == 1);
  if (plain.size() != 1) { return; }
  const auto& U = plain[0];

  // Three latents, three pictures, three casts.
  const auto p0 = cast_(px, H, W, 30, 0, -30);
  const auto p1 = cast_(px, H, W, -25, 25, 0);
  const auto p2 = cast_(px, H, W, 0, -20, 35);
  std::vector<std::unique_ptr<TensorBeatPayload>> ls, ps;
  ls.push_back(latent_(z, shape, "none"));
  ls.push_back(latent_(z, shape, "gaussian"));
  ls.push_back(latent_(z, shape, "adain"));
  ps.push_back(picture_(p0, H, W));
  ps.push_back(picture_(p1, H, W));
  ps.push_back(picture_(p2, H, W));
  const auto got = decode_(f.root, std::move(ls), std::move(ps));
  ASSERT_TRUE(got.size() == 3);
  if (got.size() != 3) { return; }

  // `none`: the decode, to the byte.
  EXPECT_TRUE(got[0] == U);
  // `gaussian` against picture 1, as the release's formula from the
  // 8-bit decode (vae-decode fixes the decode before rounding it, so a
  // level apart at most).
  const auto want1 = host_fix_(U, p1, H, W, "gaussian");
  const auto want2 = host_fix_(U, p2, H, W, "adain");
  const Diff d1 = diff_(got[1], want1), d2 = diff_(got[2], want2);
  // ...and NOT against the picture before it.
  const Diff d1x = diff_(got[1], host_fix_(U, p0, H, W, "gaussian"));
  const Diff moved = diff_(got[1], U);
  std::printf("[vosr] gaussian fix vs restated max %d mean %.4f; adain "
              "max %d mean %.4f; vs the wrong picture mean %.2f; moved "
              "the decode %.2f levels\n", d1.max, d1.mean, d2.max, d2.mean,
              d1x.mean, moved.mean);
  EXPECT_TRUE(d1.max <= 2 && d1.mean < 0.3);
  EXPECT_TRUE(d2.max <= 2 && d2.mean < 0.3);
  EXPECT_TRUE(d1x.mean > 5.0);
  EXPECT_TRUE(moved.mean > 5.0);

  // Asked for with nothing wired: the decode as it is, and no stall.
  std::vector<std::unique_ptr<TensorBeatPayload>> lu;
  lu.push_back(latent_(z, shape, "gaussian"));
  const auto unwired = decode_(f.root, std::move(lu), {});
  ASSERT_TRUE(unwired.size() == 1);
  if (unwired.size() == 1) { EXPECT_TRUE(unwired[0] == U); }
}

namespace {

// The shipped graph's shape: the picture to the conditioner, the
// encoder and vae-decode's color_reference; `color_fix` on a
// vosr-model-config when not empty. Returns the decode and the latent.
struct Restored {
  std::vector<std::uint8_t> pic;
  std::vector<float> latent;
  std::vector<std::int64_t> shape;
  FlexData sb;
};

Restored
restore_(const Fx& f, const std::vector<std::uint8_t>& px,
         const std::string& color_fix)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src = std::make_unique<Beats>(&sess, "src", std::vector<InEdge>{},
                                     FlexData::make_object());
  src->beats.push_back(picture_(px, f.H, f.W));
  src->allocate_oports(1);
  Stage* s_src = pl->insert_stage(std::move(src));
  const FlexData no_tune = FlexData::make_bool(false);

  std::vector<InEdge> cin(7, InEdge{nullptr, 0});
  cin[3] = InEdge{s_src, 0};                         // ref_image
  Stage* s_cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::move(cin),
      obj_({{"hf_dir", FlexData::make_string(f.root)},
            {"encoder_dir", FlexData::make_string(f.dino)},
            {"autotune", no_tune}})));
  Stage* s_enc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
      &sess, "enc", std::vector<InEdge>{{s_src, 0}, {nullptr, 0},
                                        {nullptr, 0}},
      obj_({{"hf_dir", FlexData::make_string(f.root)},
            {"autotune", no_tune}})));
  Stage* s_mc = nullptr;
  if (!color_fix.empty()) {
    s_mc = pl->insert_stage(std::make_unique<VosrModelConfigStage>(
        &sess, "vosr", std::vector<InEdge>{{nullptr, 0}},
        obj_({{"color_fix", FlexData::make_string(color_fix)}})));
  }
  std::vector<InEdge> gin(9, InEdge{nullptr, 0});
  gin[0] = InEdge{s_cond, 0};                        // conditioning
  gin[5] = InEdge{s_enc, 0};                         // ref_latent0
  if (s_mc != nullptr) { gin[7] = InEdge{s_mc, 0}; } // model_config
  Stage* s_gen = pl->insert_stage(std::make_unique<GenerateImageStage>(
      &sess, "gen", std::move(gin),
      obj_({{"hf_dir", FlexData::make_string(f.root)},
            {"seed", FlexData::make_int(42)},
            {"autotune", no_tune}})));
  Stage* s_vae = pl->insert_stage(std::make_unique<VaeDecodeStage>(
      &sess, "vae", std::vector<InEdge>{{s_gen, 0}, {nullptr, 0},
                                        {s_src, 0}},
      obj_({{"hf_dir", FlexData::make_string(f.root)},
            {"autotune", no_tune}})));
  auto* pics = static_cast<Keep*>(pl->insert_stage(std::make_unique<Keep>(
      &sess, "pics", std::vector<InEdge>{{s_vae, 0}},
      FlexData::make_object())));
  auto* lats = static_cast<Keep*>(pl->insert_stage(std::make_unique<Keep>(
      &sess, "lats", std::vector<InEdge>{{s_gen, 0}},
      FlexData::make_object())));
  {
    PipelineRuntime rt(pl.get(), &sess);
    if (rt.launch()) {
      rt.wait_idle();
      rt.stop();
    }
  }
  Restored r;
  if (pics->pics.size() == 1) { r.pic = std::move(pics->pics[0]); }
  if (lats->latents.size() == 1) {
    r.latent = std::move(lats->latents[0]);
    r.shape = std::move(lats->shapes[0]);
    r.sb = lats->sb[0];
  }
  return r;
}

std::string
sb_str_(const FlexData& sb, std::string_view k)
{
  if (!sb.is_object()) { return {}; }
  FlexData o = sb;
  auto m = o.as_object();
  return m.contains(k) ? std::string(m.at(k).as_string("")) : std::string();
}

}  // namespace

// The restore graph as shipped: unset, generate-image asks for the
// release's default (its wavelet fix: a Gaussian at sigma 5) and the
// picture that comes out is that fix of the decode; vosr-model-config's
// `color_fix` reaches the latent as asked.
TEST(vosr_color_fix, the_restore_graph_takes_the_releases_fix)
{
  Fx f;
  if (!f.ok) { return; }
  const auto px = read_u8_(fs::path(f.golden) / "resized_u8.bin");
  if (px.size() != (std::size_t)3 * f.H * f.W) { return; }

  const Restored def = restore_(f, px, "");
  ASSERT_TRUE(!def.pic.empty() && !def.latent.empty());
  if (def.pic.empty() || def.latent.empty()) { return; }
  EXPECT_TRUE(sb_str_(def.sb, sideband::kColorRefMethod) == "gaussian");
  {
    FlexData o = def.sb;
    auto m = o.as_object();
    EXPECT_TRUE(m.contains(sideband::kColorRefSigma) &&
                m.at(sideband::kColorRefSigma).as_real(0.0) == 5.0);
    EXPECT_TRUE(m.contains(sideband::kColorRefFrames) &&
                m.at(sideband::kColorRefFrames).as_int(0) == 1);
  }
  // The same latent decoded plain, then fixed on the host.
  std::vector<std::unique_ptr<TensorBeatPayload>> l;
  l.push_back(latent_(def.latent, def.shape, ""));
  const auto plain = decode_(f.root, std::move(l), {});
  ASSERT_TRUE(plain.size() == 1);
  if (plain.size() != 1) { return; }
  const Diff d = diff_(def.pic, host_fix_(plain[0], px, f.H, f.W,
                                          "gaussian"));
  const Diff moved = diff_(def.pic, plain[0]);
  std::printf("[vosr] restore graph: fix vs restated max %d mean %.4f; "
              "moved the decode %.3f levels (max %d)\n", d.max, d.mean,
              moved.mean, moved.max);
  EXPECT_TRUE(d.max <= 2 && d.mean < 0.3);
  EXPECT_TRUE(moved.max > 0);

  // vosr-model-config's choice reaches the latent.
  const Restored ad = restore_(f, px, "adain");
  EXPECT_TRUE(sb_str_(ad.sb, sideband::kColorRefMethod) == "adain");
}
