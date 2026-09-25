// MetalTaeDecoder: the tiny-autoencoder preview decoder.
//
// Two tiers. The frame-count rule needs nothing and always runs. The
// numeric tier compares against goldens written by
// tools/dump_tae_golden.py -- the PUBLISHED decoders (madebyollin's
// taehv.py, KJNodes' 2D loader) run over latents their own encoder made --
// and skips vacuously unless VPIPE_TAE_GOLDEN names that directory. Each
// golden case carries its checkpoint's absolute path, so the one variable
// is enough.
#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/shared/metal-tae-decoder.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using Trim = MetalTaeDecoder::Trim;

namespace {

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

struct Case {
  std::string checkpoint;
  std::string trim;
  int C = 0, T = 0, h = 0, w = 0, F = 0, H = 0, W = 0;
  int IC = 3;   // image channels: 4 for an RGBA TAE
  std::vector<float> latent;
  std::vector<float> frames;
};

// One golden case, or false when the directory does not hold it -- a
// dumper run without that checkpoint writes no case, and that is a skip.
bool
load_case_(const std::string& dir, Case* c)
{
  std::ifstream in(dir + "/meta.json");
  if (!in) { return false; }
  std::stringstream ss;
  ss << in.rdbuf();
  FlexData meta;
  try {
    meta = FlexData::from_json(ss.str());
  } catch (...) {
    return false;
  }
  if (!meta.is_object()) { return false; }
  auto o = meta.as_object();
  auto i = [&](const char* k) { return (int)o.at(k).as_int(0); };
  c->checkpoint = std::string(o.at("checkpoint").as_string(""));
  c->trim = std::string(o.at("trim").as_string(""));
  c->C = i("C"); c->T = i("T"); c->h = i("h"); c->w = i("w");
  c->F = i("F"); c->H = i("H"); c->W = i("W");
  if (o.contains("IC")) { c->IC = i("IC"); }
  c->latent = read_f32_(dir + "/latent.f32");
  c->frames = read_f32_(dir + "/frames.f32");
  return !c->checkpoint.empty() &&
         c->latent.size() == (std::size_t)c->C * c->T * c->h * c->w &&
         c->frames.size() == (std::size_t)c->F * c->IC * c->H * c->W;
}

std::string
golden_(const char* name)
{
  const char* gd = std::getenv("VPIPE_TAE_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return {}; }
  return std::string(gd) + "/" + name;
}

// rel-L2 of the U8 decode against the f32 reference. The U8 rounding
// alone is ~1/255/sqrt(12) per pixel, well under the bars used here.
double
rel_l2_(const std::vector<std::uint8_t>& got, const std::vector<float>& ref)
{
  double num = 0.0, den = 0.0;
  for (std::size_t k = 0; k < ref.size(); ++k) {
    const double d = (double)got[k] / 255.0 - (double)ref[k];
    num += d * d;
    den += (double)ref[k] * (double)ref[k];
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

// One golden case, decoded. minitest's assertions only exist inside a
// TEST, so this reports and CHECK_CASE_ judges.
struct Outcome {
  bool ran = false;       // the case and a GPU were present
  bool loaded = false;
  bool decoded = false;
  int  frames_rule = 0;   // frames_for() on the golden's latent count
  Case c;
  MetalTaeDecoder::Info info;
  MetalTaeDecoder::Frames got;
  double rel = 0.0;
  std::string err;
};

Outcome
decode_case_(const char* name, int max_frames = 0, int shrink = 1)
{
  Outcome o;
  const std::string dir = golden_(name);
  if (dir.empty() || !load_case_(dir, &o.c)) { return o; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return o; }
  o.ran = true;
  auto ws = WeightSet::open(o.c.checkpoint, nullptr);
  if (ws == nullptr) { o.err = "checkpoint did not open"; return o; }
  auto dec = MetalTaeDecoder::load(ws, mc, &o.err);
  if (dec == nullptr) { return o; }
  o.loaded = true;
  o.info = dec->info();
  const Trim trim = o.c.trim == "h3" ? Trim::kH3Chunks : Trim::kLeading;
  o.frames_rule = dec->frames_for(o.c.T, trim);
  o.decoded = dec->decode(o.c.latent.data(), o.c.C, o.c.T, o.c.h, o.c.w,
                          trim, max_frames, shrink, &o.got, &o.err);
  if (o.decoded && o.got.frames == o.c.F && o.got.height == o.c.H &&
      o.got.width == o.c.W) {
    o.rel = rel_l2_(o.got.rgb, o.c.frames);
  }
  std::printf("[tae] %s: [%d, %d, %d, %d] -> %d frames at %dx%d, "
              "rel-L2 %.5f, %zu params%s%s\n", name, o.c.C, o.c.T, o.c.h,
              o.c.w, o.got.frames, o.got.width, o.got.height, o.rel,
              o.info.params, o.err.empty() ? "" : ", ", o.err.c_str());
  return o;
}

}  // namespace

// What the layout READ AS comes before any number: a decoder that parsed
// into the wrong network can still produce plausible pixels.
#define CHECK_CASE_(o, want, bar)                                            \
  do {                                                                       \
    ASSERT_TRUE((o).loaded);                                                 \
    ASSERT_TRUE((o).decoded);                                                \
    if (!(o).loaded || !(o).decoded) { break; }                              \
    EXPECT_TRUE((o).info.layout == (want).layout);                           \
    EXPECT_TRUE((o).info.latent_channels == (want).latent_channels);         \
    EXPECT_TRUE((o).info.spatial == (want).spatial);                         \
    EXPECT_TRUE((o).info.patch == (want).patch);                             \
    EXPECT_TRUE((o).info.time_upscale == (want).time_upscale);               \
    EXPECT_TRUE((o).info.temporal == (want).temporal);                       \
    EXPECT_TRUE((o).frames_rule == (o).c.F);                                 \
    EXPECT_TRUE((o).got.frames == (o).c.F);                                  \
    EXPECT_TRUE((o).got.height == (o).c.H);                                  \
    EXPECT_TRUE((o).got.width == (o).c.W);                                   \
    EXPECT_TRUE((o).rel > 0.0 && (o).rel < (bar));                           \
  } while (0)

// The rule that decides which raw frames are real, in isolation. H3's
// VAE opens every five-latent chunk with a frame covering ONE pixel
// frame, so 5n + 2 latents are 17n + 5 frames -- the counts
// generate-video aligns to. The leading rule is every other TAEHV's
// 4T - 3; a 2D decoder keeps one frame per latent frame either way.
TEST(tae_decoder, the_frame_rule_matches_each_family)
{
  EXPECT_TRUE(MetalTaeDecoder::frames_for(2, 4, Trim::kH3Chunks) == 5);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(7, 4, Trim::kH3Chunks) == 22);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(12, 4, Trim::kH3Chunks) == 39);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(17, 4, Trim::kH3Chunks) == 56);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(27, 4, Trim::kH3Chunks) == 90);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(6, 4, Trim::kLeading) == 21);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(21, 4, Trim::kLeading) == 81);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(7, 1, Trim::kH3Chunks) == 7);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(7, 1, Trim::kLeading) == 7);
  EXPECT_TRUE(MetalTaeDecoder::frames_for(0, 4, Trim::kH3Chunks) == 0);
}

// madebyollin's taeh3 across ONE chunk boundary and then two, so a
// latent frame that opens a chunk (and keeps only its last sub-frame) is
// exercised mid-clip, not just at the start.
TEST(tae_decoder, taeh3_matches_the_reference)
{
  MetalTaeDecoder::Info want;
  want.layout = "taehv";
  want.latent_channels = 24;
  want.spatial = 16;
  want.patch = 2;
  want.time_upscale = 4;
  want.temporal = true;
  for (const char* name : {"taeh3_22", "taeh3_39"}) {
    const Outcome o = decode_case_(name);
    if (!o.ran) { continue; }
    CHECK_CASE_(o, want, 0.02);
  }
}

// A prefix decode is the same frames as the whole one, EXACTLY: the
// memory only chains forward. And it stops early, which is what makes a
// frame cap worth having.
TEST(tae_decoder, a_frame_cap_decodes_an_exact_prefix)
{
  const Outcome all = decode_case_("taeh3_39");
  if (!all.ran) { return; }
  const Outcome part = decode_case_("taeh3_39", 10);
  ASSERT_TRUE(all.decoded && part.decoded);
  if (!all.decoded || !part.decoded) { return; }
  EXPECT_TRUE(part.got.frames == 10);
  EXPECT_TRUE(all.got.frames == all.c.F);
  const std::size_t n = (std::size_t)10 * 3 * all.c.H * all.c.W;
  EXPECT_TRUE(part.got.rgb.size() == n);
  bool same = part.got.rgb.size() == n && all.got.rgb.size() >= n;
  for (std::size_t k = 0; same && k < n; ++k) {
    same = part.got.rgb[k] == all.got.rgb[k];
  }
  EXPECT_TRUE(same);
}

// The shrink is a box filter over the DECODED picture, so it has to equal
// the reference frames averaged 2x2 -- to the U8 floor, not merely look
// similar.
TEST(tae_decoder, a_shrink_is_the_decoded_picture_box_filtered)
{
  const Outcome o = decode_case_("taeh3_22", 0, 2);
  if (!o.ran) { return; }
  ASSERT_TRUE(o.decoded);
  if (!o.decoded) { return; }
  const int H = o.c.H / 2, W = o.c.W / 2;
  EXPECT_TRUE(o.got.height == H && o.got.width == W);
  EXPECT_TRUE(o.got.frames == o.c.F);
  if (o.got.height != H || o.got.width != W || o.got.frames != o.c.F) {
    return;
  }
  std::vector<float> ref((std::size_t)o.c.F * 3 * H * W);
  for (int f = 0; f < o.c.F; ++f) {
    for (int c = 0; c < 3; ++c) {
      const float* sp =
          o.c.frames.data() + ((std::size_t)f * 3 + c) * o.c.H * o.c.W;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const float a = sp[(std::size_t)(2 * y) * o.c.W + 2 * x] +
                          sp[(std::size_t)(2 * y) * o.c.W + 2 * x + 1] +
                          sp[(std::size_t)(2 * y + 1) * o.c.W + 2 * x] +
                          sp[(std::size_t)(2 * y + 1) * o.c.W + 2 * x + 1];
          ref[(((std::size_t)f * 3 + c) * H + y) * W + x] = a * 0.25f;
        }
      }
    }
  }
  const double r = rel_l2_(o.got.rgb, ref);
  std::printf("[tae] shrink 2: rel-L2 %.5f against the box-filtered "
              "reference\n", r);
  EXPECT_TRUE(r > 0.0 && r < 0.02);
}

// Kijai's decoder has no temporal structure and no prefix on its names:
// the reader has to find a 2D stack of Blocks, one of them with a skip.
TEST(tae_decoder, kijai_2d_matches_the_reference)
{
  MetalTaeDecoder::Info want;
  want.layout = "2d";
  want.latent_channels = 24;
  want.spatial = 16;
  want.patch = 1;
  want.time_upscale = 1;
  want.temporal = false;
  const Outcome o = decode_case_("kijai2d");
  if (!o.ran) { return; }
  CHECK_CASE_(o, want, 0.02);
}

// The Wan 2.1 TAEHV: 16 channels, 8x, no pixel shuffle, and the leading
// trim rather than H3's chunked one.
TEST(tae_decoder, taew2_1_matches_the_reference)
{
  MetalTaeDecoder::Info want;
  want.layout = "taehv";
  want.latent_channels = 16;
  want.spatial = 8;
  want.patch = 1;
  want.time_upscale = 4;
  want.temporal = true;
  const Outcome o = decode_case_("taew2_1");
  if (!o.ran) { return; }
  CHECK_CASE_(o, want, 0.02);
}

// TAESD for the FLUX.1 latent space (Z-Image), in diffusers' naming:
// `decoder.layers.<i>`, every index one lower than taesd.py's because the
// input Clamp is implicit there. A still decoder -- no memory, no time.
TEST(tae_decoder, taef1_matches_the_reference)
{
  MetalTaeDecoder::Info want;
  want.layout = "diffusers";
  want.latent_channels = 16;
  want.spatial = 8;
  want.patch = 1;
  want.time_upscale = 1;
  want.temporal = false;
  const Outcome o = decode_case_("taef1");
  if (!o.ran) { return; }
  CHECK_CASE_(o, want, 0.02);
}

// TAESD for FLUX.2: its first three blocks carry the mid-block POOL,
// whose GroupNorm reduces over the whole picture. Dropping that branch
// would still decode -- to the wrong picture -- so this is the case that
// says the reader runs it.
TEST(tae_decoder, taef2_matches_the_reference)
{
  MetalTaeDecoder::Info want;
  want.layout = "diffusers";
  want.latent_channels = 32;
  want.spatial = 8;
  want.patch = 1;
  want.time_upscale = 1;
  want.temporal = false;
  const Outcome o = decode_case_("taef2");
  if (!o.ran) { return; }
  CHECK_CASE_(o, want, 0.02);
}

// TAESD for Qwen-Image-2.1: 16x through a 2x2 pixel-shuffled head, RGBA
// out with STRAIGHT alpha, read from torch .pth -- three things no other
// TAE here has at once. The golden's input carries opaque, clear and
// partial-alpha regions, so the alpha plane is compared, not assumed.
TEST(tae_decoder, taeqi2_1_matches_the_reference)
{
  MetalTaeDecoder::Info want;
  want.layout = "2d";
  want.latent_channels = 64;
  want.spatial = 16;
  want.patch = 2;
  want.time_upscale = 1;
  want.temporal = false;
  const Outcome o = decode_case_("taeqi2_1");
  if (!o.ran) { return; }
  CHECK_CASE_(o, want, 0.02);
  EXPECT_TRUE(o.info.image_channels == 4);
  EXPECT_TRUE(o.got.channels == 4);
}

// An RGBA shrink is PREMULTIPLIED. Judged the way a viewer sees it --
// colour TIMES alpha, since colour under a clear pixel is arbitrary and
// dividing by a near-zero alpha only amplifies rounding. The golden's
// input puts BLACK under its clear pixels right against opaque colour,
// on an odd column so 2x2 boxes straddle the edge, which is where a
// straight average fringes dark. As the Qwen-Image-2.1 port found, such
// a test passes vacuously without partial alpha, so it asserts the
// partial band exists and that the two averages differ.
TEST(tae_decoder, an_rgba_shrink_is_premultiplied)
{
  const Outcome o = decode_case_("taeqi2_1", 0, 2);
  if (!o.ran) { return; }
  ASSERT_TRUE(o.decoded && o.got.channels == 4);
  if (!o.decoded || o.got.channels != 4) { return; }
  const int H = o.c.H / 2, W = o.c.W / 2;
  ASSERT_TRUE(o.got.height == H && o.got.width == W);
  if (o.got.height != H || o.got.width != W) { return; }
  const std::size_t plane = (std::size_t)o.c.H * o.c.W;
  const std::size_t hw = (std::size_t)H * W;
  const float* ref = o.c.frames.data();
  // Colour x alpha, and alpha: [4, H, W], for each of the three.
  std::vector<float> got(4 * hw), pre(4 * hw), flat(4 * hw);
  int partial = 0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float cp[3] = {0, 0, 0}, cf[3] = {0, 0, 0}, a = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          const std::size_t k =
              (std::size_t)(2 * y + dy) * o.c.W + (2 * x + dx);
          const float ak = ref[3 * plane + k];
          for (int c = 0; c < 3; ++c) {
            cp[c] += ref[c * plane + k] * ak;
            cf[c] += ref[c * plane + k];
          }
          a += ak;
        }
      }
      a *= 0.25f;
      const std::size_t q = (std::size_t)y * W + x;
      const float ga = o.got.rgb[3 * hw + q] / 255.0f;
      for (int c = 0; c < 3; ++c) {
        pre[c * hw + q] = cp[c] * 0.25f;           // mean of c*a
        flat[c * hw + q] = cf[c] * 0.25f * a;      // mean c, times mean a
        got[c * hw + q] = o.got.rgb[c * hw + q] / 255.0f * ga;
      }
      pre[3 * hw + q] = flat[3 * hw + q] = a;
      got[3 * hw + q] = ga;
      if (a > 0.05f && a < 0.95f) { ++partial; }
    }
  }
  auto rel = [](const std::vector<float>& u, const std::vector<float>& v) {
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < u.size(); ++k) {
      num += (double)(u[k] - v[k]) * (u[k] - v[k]);
      den += (double)v[k] * v[k];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
  };
  const double r_pre = rel(got, pre), r_flat = rel(got, flat);
  std::printf("[tae] rgba shrink: %d partial-alpha pixels; premultiplied "
              "rel-L2 %.5f against the premultiplied box, %.5f against the "
              "straight one\n", partial, r_pre, r_flat);
  EXPECT_TRUE(partial > 100);
  EXPECT_TRUE(r_pre < 0.02);
  EXPECT_TRUE(r_flat > 2 * r_pre);
}

// The plan a stage books before anything loads is the decoder's OWN
// sizing, read off the headers: after a real decode of the same clip,
// what the decoder holds is exactly what was planned -- for the temporal
// 16x TAE, the 2D one and the 8x one alike, whose per-pixel costs differ
// by several times.
TEST(tae_decoder, the_plan_is_what_a_decode_allocates)
{
  for (const char* name : {"taeh3_22", "kijai2d", "taew2_1", "taef1",
                           "taef2", "taeqi2_1"}) {
    const std::string dir = golden_(name);
    Case c;
    if (dir.empty() || !load_case_(dir, &c)) { continue; }
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr) { return; }
    const Trim trim = c.trim == "h3" ? Trim::kH3Chunks : Trim::kLeading;
    MetalTaeDecoder::Plan plan;
    std::string err;
    ASSERT_TRUE(MetalTaeDecoder::plan(c.checkpoint, c.T, c.h, c.w, trim,
                                      &plan, &err));
    auto ws = WeightSet::open(c.checkpoint, nullptr);
    auto dec = MetalTaeDecoder::load(ws, mc, &err);
    ASSERT_TRUE(dec != nullptr);
    if (dec == nullptr) { continue; }
    MetalTaeDecoder::Frames f;
    ASSERT_TRUE(dec->decode(c.latent.data(), c.C, c.T, c.h, c.w, trim, 0, 1,
                            &f, &err));
    const std::size_t planned = plan.weight_bytes + plan.scratch_bytes;
    std::printf("[tae] %s: planned %zu bytes (%zu weights), holds %zu\n",
                name, planned, plan.weight_bytes, dec->resident_bytes());
    EXPECT_TRUE(plan.info.layout == dec->info().layout);
    EXPECT_TRUE(plan.info.params == dec->info().params);
    EXPECT_TRUE(planned == dec->resident_bytes());
  }
}

// What a preview costs at a real geometry, not a check. Opt-in
// (VPIPE_TAE_BENCH=1) and reads the checkpoint from the taeh3 golden.
// 960x576 at 90 frames is 27 latent frames of 36x60; the half-size row
// is the same clip with the latent pooled 2x, which is how a preview
// reaches a smaller picture.
TEST(tae_decoder, preview_cost)
{
  if (std::getenv("VPIPE_TAE_BENCH") == nullptr) { return; }
  const std::string dir = golden_("taeh3_22");
  Case c;
  if (dir.empty() || !load_case_(dir, &c)) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = WeightSet::open(c.checkpoint, nullptr);
  std::string err;
  auto dec = MetalTaeDecoder::load(ws, mc, &err);
  ASSERT_TRUE(dec != nullptr);
  if (dec == nullptr) { return; }
  for (const int div : {1, 2}) {
    const int T = 27, h = 36 / div, w = 60 / div;
    std::vector<float> lat((std::size_t)24 * T * h * w);
    for (std::size_t k = 0; k < lat.size(); ++k) {
      lat[k] = std::sin(0.37f * (float)k) * 1.5f;
    }
    MetalTaeDecoder::Frames f;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = dec->decode(lat.data(), 24, T, h, w, Trim::kH3Chunks,
                                  0, 1, &f, &err);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      ASSERT_TRUE(ok);
      std::printf("[tae-bench] %dx%d, %d frames: %.1f ms (%.2f ms/frame), "
                  "%.1f MB resident\n", f.width, f.height, f.frames, ms,
                  ms / std::max(1, f.frames),
                  (double)dec->resident_bytes() / 1e6);
    }
  }
}
