// apple-silicon/media: the Apple-native picture and video path. Each test
// is a ROUND TRIP through a real file -- write with the module, read back
// with the module -- and checks what the FFmpeg path cannot keep: more
// than 8 bits, float above 1, alpha, and the colour the file declares.

#include "minitest.h"

#include "apple-silicon/media/color-tags.h"
#include "apple-silicon/media/image-io.h"
#include "apple-silicon/media/video-io.h"
#include "apple-silicon/tensor-beat.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::apple_media;

#define REQUIRE_(cond) do { if (!(cond)) { EXPECT_TRUE(cond); return; } } \
                       while (0)

namespace {

using half = _Float16;

std::string
tmp_(const std::string& name)
{
  return "/tmp/vpipe-apple-media-" + std::to_string(getpid()) + "-" + name;
}

// A planar F16 picture whose samples are fn(c, y, x).
template <class Fn>
TensorBeat
picture_(int C, int H, int W, Fn fn)
{
  TensorBeat t;
  t.dtype = TensorBeat::DType::F16;
  t.shape = {C, H, W};
  t.data.assign((std::size_t)C * H * W * 2, 0);
  auto* p = reinterpret_cast<half*>(t.data.data());
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        p[((std::size_t)c * H + y) * W + x] = (half)fn(c, y, x);
      }
    }
  }
  return t;
}

float
at_(const TensorBeat& t, int c, int y, int x)
{
  const int H = (int)t.shape[1], W = (int)t.shape[2];
  const std::size_t i = ((std::size_t)c * H + y) * W + x;
  if (t.dtype == TensorBeat::DType::U8) { return t.data.data()[i] / 255.f; }
  return (float)reinterpret_cast<const half*>(t.data.data())[i];
}

bool
write_file_(const std::string& path, const std::vector<std::uint8_t>& b)
{
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
  return (bool)f;
}

}  // namespace

// 16-bit PNG: deeper than 8 bits, colour space kept (Display P3).
TEST(apple_media, png16_keeps_depth_and_colour)
{
  const int H = 8, W = 64;
  auto pic = picture_(3, H, W, [](int c, int, int x) {
    return (x + 1) / 65.0f * (c == 1 ? 0.5f : 1.0f);
  });
  ColorTags tags;
  tags.primaries = 12;
  tags.transfer = 13;
  StillWriteOptions opt;
  opt.format = "png";
  opt.bits = 16;
  std::vector<std::uint8_t> bytes;
  std::string err;
  REQUIRE_(encode_still(pic, tags, opt, &bytes, &err));
  const std::string path = tmp_("deep.png");
  REQUIRE_(write_file_(path, bytes));

  StillInfo info;
  REQUIRE_(probe_still(path, &info, &err));
  EXPECT_TRUE(info.bits == 16);
  EXPECT_TRUE(prefers_imageio(info));
  ColorTags back;
  auto got = read_still(path, TensorBeat::DType::F16, false, &back, &err);
  REQUIRE_(got != nullptr);
  EXPECT_TRUE(got->shape == (std::vector<std::int64_t>{3, H, W}));
  EXPECT_TRUE(back.primaries == 12 && back.transfer == 13);
  EXPECT_TRUE(back.source_bits == 16);
  float worst = 0;
  for (int c = 0; c < 3; ++c) {
    for (int x = 0; x < W; ++x) {
      worst = std::max(worst, std::fabs(at_(*got, c, 3, x) -
                                        at_(pic, c, 3, x)));
    }
  }
  // F16 near 1 resolves 2^-11: finer than 10 bits, coarser than 16.
  std::printf("[apple_media] png16 worst error %.6f\n", worst);
  EXPECT_TRUE(worst < 1.0f / 1024);
  std::remove(path.c_str());
}

// OpenEXR: half float, values above 1, alpha -- exactly, in linear
// BT.709 (the primaries ImageIO's OpenEXR stores).
TEST(apple_media, exr_keeps_float_above_one_and_alpha)
{
  const int H = 4, W = 16;
  auto pic = picture_(4, H, W, [](int c, int, int x) {
    if (c == 3) { return (x + 1) / 16.0f; }
    return 0.25f * (x + 1) * (c + 1);  // up to 12.0
  });
  ColorTags tags;
  tags.primaries = 1;
  tags.transfer = 8;
  StillWriteOptions opt;
  opt.format = "exr";
  std::vector<std::uint8_t> bytes;
  std::string err;
  REQUIRE_(encode_still(pic, tags, opt, &bytes, &err));
  const std::string path = tmp_("lin.exr");
  REQUIRE_(write_file_(path, bytes));

  StillInfo info;
  REQUIRE_(probe_still(path, &info, &err));
  EXPECT_TRUE(info.is_float);
  ColorTags back;
  auto got = read_still(path, TensorBeat::DType::F16, true, &back, &err);
  REQUIRE_(got != nullptr);
  REQUIRE_(got->shape[0] == 4);
  std::printf("[apple_media] exr tags %d/%d, (2,1,15)=%.3f a=%.3f\n",
              back.primaries, back.transfer, at_(*got, 2, 1, 15),
              at_(*got, 3, 1, 15));
  EXPECT_TRUE(back.primaries == 1 && back.transfer == 8);
  float worst = 0;
  for (int c = 0; c < 4; ++c) {
    for (int x = 0; x < W; ++x) {
      const float want = at_(pic, c, 1, x);
      worst = std::max(worst, std::fabs(at_(*got, c, 1, x) - want) /
                                  std::max(1.0f, want));
    }
  }
  EXPECT_TRUE(worst < 0.01f);
  std::remove(path.c_str());
}

// Wider primaries go into OpenEXR converted, not clipped: a saturated
// BT.2020 green is outside BT.709, so it reads back with a negative red.
TEST(apple_media, exr_converts_wide_primaries_unclamped)
{
  // Pixel (0,0) BT.2020 green, (1,0) white at 2.0 -- a highlight a linear
  // RAW development keeps -- which must come back 2.0, not 1.
  auto pic = picture_(3, 2, 2, [](int c, int, int x) {
    return x == 1 ? 2.0f : c == 1 ? 1.0f : 0.0f;
  });
  ColorTags tags;
  tags.primaries = 9;
  tags.transfer = 8;
  StillWriteOptions opt;
  opt.format = "exr";
  std::vector<std::uint8_t> bytes;
  std::string err;
  REQUIRE_(encode_still(pic, tags, opt, &bytes, &err));
  const std::string path = tmp_("wide.exr");
  REQUIRE_(write_file_(path, bytes));
  ColorTags back;
  auto got = read_still(path, TensorBeat::DType::F16, false, &back, &err);
  REQUIRE_(got != nullptr);
  std::printf("[apple_media] 2020 green -> exr (%.3f, %.3f, %.3f) %d/%d\n",
              at_(*got, 0, 0, 0), at_(*got, 1, 0, 0), at_(*got, 2, 0, 0),
              back.primaries, back.transfer);
  EXPECT_TRUE(back.transfer == 8);
  EXPECT_TRUE(at_(*got, 0, 0, 0) < -0.1f);  // out of BT.709, kept
  EXPECT_TRUE(at_(*got, 1, 0, 0) > 1.0f);
  std::printf("[apple_media] 2020 white 2.0 -> exr (%.3f, %.3f, %.3f)\n",
              at_(*got, 0, 0, 1), at_(*got, 1, 0, 1), at_(*got, 2, 0, 1));
  for (int c = 0; c < 3; ++c) {
    EXPECT_TRUE(std::fabs(at_(*got, c, 0, 1) - 2.0f) < 0.01f);
  }
  std::remove(path.c_str());
}

// An sRGB picture into OpenEXR is linearized: 0.5 sRGB is ~0.214 linear.
TEST(apple_media, exr_linearizes_an_srgb_picture)
{
  auto pic = picture_(3, 2, 2, [](int, int, int) { return 0.5f; });
  StillWriteOptions opt;
  opt.format = "exr";
  std::vector<std::uint8_t> bytes;
  std::string err;
  REQUIRE_(encode_still(pic, ColorTags::srgb(), opt, &bytes, &err));
  const std::string path = tmp_("srgb.exr");
  REQUIRE_(write_file_(path, bytes));
  ColorTags back;
  auto got = read_still(path, TensorBeat::DType::F16, false, &back, &err);
  REQUIRE_(got != nullptr);
  std::printf("[apple_media] sRGB 0.5 -> exr %.4f (tags %d/%d)\n",
              at_(*got, 0, 0, 0), back.primaries, back.transfer);
  EXPECT_TRUE(std::fabs(at_(*got, 0, 0, 0) - 0.214f) < 0.01f);
  std::remove(path.c_str());
}

// ProRes 4444 keeps alpha and more than 8 bits.
TEST(apple_media, prores4444_keeps_alpha)
{
  const int H = 96, W = 128, N = 6;
  const std::string path = tmp_("alpha.mov");
  VideoWriteOptions wo;
  wo.codec = "prores4444";
  wo.width = W;
  wo.height = H;
  wo.fps_num = 24;
  wo.alpha = true;
  wo.color.primaries = 1;
  wo.color.transfer = 1;
  wo.color.matrix = 1;
  VideoWriter w;
  std::string err;
  REQUIRE_(w.open(path, wo, &err));
  for (int i = 0; i < N; ++i) {
    auto f = picture_(4, H, W, [i](int c, int, int x) {
      if (c == 3) { return x < W / 2 ? 1.0f : 0.25f; }
      return 0.1f + 0.1f * c + 0.05f * i;
    });
    REQUIRE_(w.write(f, i, &err));
  }
  REQUIRE_(w.finish(&err));

  VideoReader r;
  REQUIRE_(r.open(path, VideoReadOptions{}, &err));
  const VideoInfo& vi = r.info();
  std::printf("[apple_media] prores: %s %dx%d %d/%d fps, %lld frames, "
              "alpha %d, bits %d\n", vi.codec.c_str(), vi.width, vi.height,
              vi.fps_num, vi.fps_den, (long long)vi.frames, vi.has_alpha,
              vi.color.source_bits);
  EXPECT_TRUE(vi.codec == "ap4h");
  EXPECT_TRUE(vi.has_alpha);
  EXPECT_TRUE(vi.fps_num == 24 && vi.fps_den == 1);
  TensorBeat f;
  std::int64_t pts = 0;
  int n = 0;
  float worst = 0;
  while (r.read(&f, &pts, &err)) {
    REQUIRE_(f.shape == (std::vector<std::int64_t>{4, H, W}));
    worst = std::max(worst, std::fabs(at_(f, 3, 10, W - 5) - 0.25f));
    worst = std::max(worst, std::fabs(at_(f, 0, 10, 5) -
                                      (0.1f + 0.05f * n)));
    ++n;
  }
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(n == N);
  std::printf("[apple_media] prores worst error %.4f\n", worst);
  EXPECT_TRUE(worst < 0.02f);
  std::remove(path.c_str());
}

// HEVC Main10 tagged PQ, with HDR10 metadata: the file declares BT.2020
// PQ, 10 bits, and the mastering display it was given.
TEST(apple_media, hevc_main10_pq_keeps_hdr_tags)
{
  const int H = 128, W = 128, N = 4;
  const std::string path = tmp_("pq.mov");
  VideoWriteOptions wo;
  wo.codec = "hevc";
  wo.width = W;
  wo.height = H;
  wo.fps_num = 30000;
  wo.fps_den = 1001;
  wo.color.primaries = 9;
  wo.color.transfer = 16;
  wo.color.matrix = 9;
  wo.color.mastering = {35400, 14600, 8500, 39850, 6550, 2300, 15635,
                        16450, 10000000, 50};
  wo.color.content_light = {1000, 400};
  VideoWriter w;
  std::string err;
  REQUIRE_(w.open(path, wo, &err));
  for (int i = 0; i < N; ++i) {
    auto f = picture_(3, H, W, [](int c, int y, int) {
      return 0.2f + 0.5f * y / 128.0f + 0.01f * c;
    });
    REQUIRE_(w.write(f, i, &err));
  }
  REQUIRE_(w.finish(&err));

  VideoReader r;
  REQUIRE_(r.open(path, VideoReadOptions{}, &err));
  const VideoInfo& vi = r.info();
  std::printf("[apple_media] hevc: %s %d/%d/%d bits %d, %d/%d fps, "
              "mastering %zu, cll %zu\n", vi.codec.c_str(),
              vi.color.primaries, vi.color.transfer, vi.color.matrix,
              vi.color.source_bits, vi.fps_num, vi.fps_den,
              vi.color.mastering.size(), vi.color.content_light.size());
  EXPECT_TRUE(vi.codec == "hvc1");
  EXPECT_TRUE(vi.color.primaries == 9 && vi.color.transfer == 16);
  EXPECT_TRUE(vi.color.source_bits == 10);
  EXPECT_TRUE(vi.fps_num == 30000 && vi.fps_den == 1001);
  TensorBeat f;
  std::int64_t pts = 0;
  int n = 0;
  float worst = 0;
  while (r.read(&f, &pts, &err)) {
    worst = std::max(worst, std::fabs(at_(f, 1, 64, 64) -
                                      (0.2f + 0.25f + 0.01f)));
    ++n;
  }
  EXPECT_TRUE(n == N);
  std::printf("[apple_media] hevc worst error %.4f\n", worst);
  EXPECT_TRUE(worst < 0.02f);
  if (vi.color.mastering.size() == 10) {
    EXPECT_TRUE(std::fabs(vi.color.mastering[8] - 10000000) < 1);
  }
  std::remove(path.c_str());
}

// ---- camera RAW --------------------------------------------------------
//
// A RAW is DEVELOPED, and comes out upright. Gated on VPIPE_TEST_RAW
// (a camera RAW file; none ships with the tree) -- a SKIP, said so.

namespace {

struct Stats { float lo = 1e9f, hi = -1e9f; double mean = 0; };

Stats
stats_(const TensorBeat& t)
{
  Stats s;
  const std::size_t n = (std::size_t)t.shape[0] * t.shape[1] * t.shape[2];
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float v = t.dtype == TensorBeat::DType::U8
        ? t.data.data()[i] / 255.f
        : (float)reinterpret_cast<const half*>(t.data.data())[i];
    s.lo = std::min(s.lo, v);
    s.hi = std::max(s.hi, v);
    sum += v;
  }
  s.mean = sum / (double)n;
  return s;
}

}  // namespace

TEST(apple_media, camera_raw_develops_upright_in_both_looks)
{
  const char* path = std::getenv("VPIPE_TEST_RAW");
  if (path == nullptr) {
    std::printf("[apple_media] SKIP: set VPIPE_TEST_RAW to a camera RAW\n");
    return;
  }
  StillInfo info;
  std::string err;
  REQUIRE_(probe_still(path, &info, &err));
  std::printf("[apple_media] raw %s %dx%d (upright) orientation %d, %d "
              "bits\n", info.uti.c_str(), info.width, info.height,
              info.orientation, info.bits);
  EXPECT_TRUE(info.raw);
  EXPECT_TRUE(prefers_imageio(info));

  // Rendered: upright, sRGB, 0..1.
  ColorTags rt;
  auto r = read_still(path, TensorBeat::DType::F16, false, &rt, &err);
  REQUIRE_(r != nullptr);
  REQUIRE_(r->shape == (std::vector<std::int64_t>{3, info.height,
                                                  info.width}));
  const Stats rs = stats_(*r);
  std::printf("[apple_media] rendered [%.3f, %.3f] mean %.3f, tags %d/%d\n",
              rs.lo, rs.hi, rs.mean, rt.primaries, rt.transfer);
  EXPECT_TRUE(rt.primaries == 1 && rt.transfer == 13);
  EXPECT_TRUE(rs.lo >= 0.0f && rs.hi <= 1.0f);
  EXPECT_TRUE(rs.mean > 0.05 && rs.mean < 0.95);

  // The same development in 8 bits.
  auto r8 = read_still(path, TensorBeat::DType::U8, false, nullptr, &err);
  REQUIRE_(r8 != nullptr && r8->shape == r->shape);
  const Stats r8s = stats_(*r8);
  EXPECT_TRUE(std::fabs(r8s.mean - rs.mean) < 0.005);

  // Linear: BT.2020 linear, no tone curve -- darker on average than the
  // rendered look -- and an EV doubles it.
  RawOptions lin;
  lin.look = RawOptions::Look::Linear;
  ColorTags lt;
  auto l = read_still(path, TensorBeat::DType::F16, false, &lt, &err, lin);
  REQUIRE_(l != nullptr && l->shape == r->shape);
  const Stats ls = stats_(*l);
  lin.exposure = 1.0;
  auto l1 = read_still(path, TensorBeat::DType::F16, false, nullptr, &err,
                       lin);
  REQUIRE_(l1 != nullptr);
  const Stats l1s = stats_(*l1);
  std::printf("[apple_media] linear [%.3f, %.3f] mean %.3f, +1 EV mean "
              "%.3f (x%.2f), tags %d/%d\n", ls.lo, ls.hi, ls.mean,
              l1s.mean, l1s.mean / ls.mean, lt.primaries, lt.transfer);
  EXPECT_TRUE(lt.primaries == 9 && lt.transfer == 8);
  EXPECT_TRUE(ls.mean < rs.mean);
  EXPECT_TRUE(l1s.mean / ls.mean > 1.8 && l1s.mean / ls.mean < 2.1);
}
