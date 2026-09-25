// content-aware-continuation: the step a continuation takes at its seam,
// measured and taken out.
//
// The continuation here is SYNTHESIZED from the reference with a known
// step -- a 1% vertical contraction, a sub-pixel shift, and a tone curve
// that darkens the shadows most, the two things measured at the seams of
// the MiniMax-H3 extend chain -- so the estimate can be held to the
// numbers that made it. Then the whole stage runs in a graph, where the
// window of frames it holds back has to come out, in order, corrected.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/content-aware-continuation-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using Cac = ContentAwareContinuationStage;

namespace {

constexpr int kW = 480, kH = 288;

// A textured frame with real shadows: structure at several scales, a
// vertical falloff into near-black, and fixed-seed grain. Planar RGB on
// a 0..255 scale.
std::vector<float>
texture_(unsigned seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<float> grain(0.0f, 3.0f);
  const float p0 = (float)(seed % 7), p1 = (float)(seed % 11) * 0.3f;
  std::vector<float> f((size_t)3 * kW * kH);
  for (int y = 0; y < kH; ++y) {
    const float fall = 0.15f + 0.85f * (float)y / (float)kH;
    for (int x = 0; x < kW; ++x) {
      const float s = 0.5f + 0.25f * std::sin(0.11f * x + p0) *
                                 std::cos(0.07f * y + p1) +
                      0.15f * std::sin(0.031f * (x + 2 * y) + p1) +
                      0.10f * std::cos(0.19f * (x - y) + p0);
      for (int c = 0; c < 3; ++c) {
        const float v = 255.0f * fall * s * (0.9f + 0.05f * c) + grain(rng);
        f[(size_t)c * kW * kH + (size_t)y * kW + x] =
            std::max(0.0f, std::min(255.0f, v));
      }
    }
  }
  return f;
}

// The step: content at reference position g shows up in the
// continuation at S (g - c0) + c0 + t, and every level is pulled down,
// most in the shadows.
constexpr double kSy = 0.99, kSx = 1.0, kTx = 1.5, kTy = -0.7;

float
darken_(float v)
{
  const float k = 1.0f - v / 255.0f;
  return std::max(0.0f, v - 7.0f * k * k);
}

std::vector<float>
continuation_of_(const std::vector<float>& ref)
{
  // cont(x) = ref(S^-1 (x - c0 - t) + c0): the inverse of what the stage
  // estimates, so its forward warp recovers `ref`.
  Cac::Geometry inv;
  inv.ok = true;
  inv.sx = 1.0 / kSx;
  inv.sy = 1.0 / kSy;
  inv.tx = -kTx / kSx;
  inv.ty = -kTy / kSy;
  std::vector<float> c;
  Cac::warp(ref, kW, kH, inv, &c);
  for (float& v : c) { v = darken_(v); }
  return c;
}

// PSNR over the interior, away from the border a warp clamps.
double
psnr_(const std::vector<float>& a, const std::vector<float>& b)
{
  double se = 0;
  size_t n = 0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 12; y < kH - 12; ++y) {
      for (int x = 12; x < kW - 12; ++x) {
        const size_t i = (size_t)c * kW * kH + (size_t)y * kW + x;
        const double d = (double)a[i] - (double)b[i];
        se += d * d;
        ++n;
      }
    }
  }
  return 10.0 * std::log10(255.0 * 255.0 / (se / (double)n + 1e-12));
}

std::vector<float>
correct_(const std::vector<float>& cont, const Cac::Geometry& g,
         const Cac::Tone& t)
{
  std::vector<float> out;
  Cac::warp(cont, kW, kH, g, &out);
  for (int c = 0; c < 3; ++c) {
    for (size_t i = 0; i < (size_t)kW * kH; ++i) {
      float& v = out[(size_t)c * kW * kH + i];
      const int k = (int)std::lround(std::max(0.0f, std::min(255.0f, v)));
      v = t.lut[(size_t)c][(size_t)k];
    }
  }
  return out;
}

// ---- a graph around the stage ------------------------------------------

std::unique_ptr<TensorBeatPayload>
beat_u8_(const std::vector<float>& planar, int frames)
{
  auto b = std::make_unique<TensorBeatPayload>();
  b->dtype = TensorBeat::DType::U8;
  if (frames > 0) {
    b->shape = {frames, 3, kH, kW};
  } else {
    b->shape = {3, kH, kW};
  }
  const size_t one = planar.size();
  b->resize_contiguous(one * (size_t)std::max(frames, 1));
  std::uint8_t* d = b->as_u8();
  for (int f = 0; f < std::max(frames, 1); ++f) {
    for (size_t i = 0; i < one; ++i) {
      d[(size_t)f * one + i] = (std::uint8_t)std::lround(
          std::max(0.0f, std::min(255.0f, planar[i])));
    }
  }
  return b;
}

class RefSource : public TypedStage<RefSource> {
public:
  static constexpr const char* kTypeName = "ut-cac-ref";
  using TypedStage::TypedStage;
  std::vector<float> frame;
  bool sent = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (sent) { ctx.signal_done(); co_return; }
    sent = true;
    co_await ctx.write(0, beat_u8_(frame, 6));   // a stacked guide
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "clip", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-cac-ref", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class FrameSource : public TypedStage<FrameSource> {
public:
  static constexpr const char* kTypeName = "ut-cac-frames";
  using TypedStage::TypedStage;
  std::vector<float> frame;
  int n = 10, emitted = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (emitted >= n) { ctx.signal_done(); co_return; }
    auto b = beat_u8_(frame, 0);
    // Frame number on the sideband, so ORDER is checkable downstream.
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("i", FlexData::make_int(emitted));
    b->sideband = std::move(sb);
    ++emitted;
    co_await ctx.write(0, std::move(b));
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "frames", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-cac-frames", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class FrameSink : public TypedStage<FrameSink> {
public:
  static constexpr const char* kTypeName = "ut-cac-sink";
  using TypedStage::TypedStage;
  std::vector<std::vector<float>> frames;
  std::vector<int> order;
  bool shape_ok = true;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    const auto* t = dynamic_cast<const TensorBeatPayload*>(in.get());
    if (t == nullptr || t->dtype != TensorBeat::DType::U8 ||
        t->shape.size() != 3 || t->shape[1] != kH || t->shape[2] != kW) {
      shape_ok = false;
      co_return;
    }
    std::vector<float> f(t->element_count());
    for (size_t i = 0; i < f.size(); ++i) { f[i] = (float)t->as_u8()[i]; }
    frames.push_back(std::move(f));
    int idx = -1;
    if (t->sideband.is_object()) {
      FlexData sb = t->sideband;
      auto o = sb.as_object();
      if (o.contains("i")) { idx = (int)o.at("i").as_int(-1); }
    }
    order.push_back(idx);
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "frames", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-cac-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

struct Run {
  std::vector<std::vector<float>> frames;
  std::vector<int> order;
  bool shape_ok = false;
};

Run
run_(const std::vector<float>& ref, const std::vector<float>& cont,
     int n_frames, FlexData cfg)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto fs_u = std::make_unique<FrameSource>(&sess, "frames",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  fs_u->frame = cont;
  fs_u->n = n_frames;
  fs_u->allocate_oports(1);
  auto* fs = static_cast<FrameSource*>(pl->insert_stage(std::move(fs_u)));
  auto rs_u = std::make_unique<RefSource>(&sess, "ref",
                                          std::vector<InEdge>{},
                                          FlexData::make_object());
  rs_u->frame = ref;
  rs_u->allocate_oports(1);
  auto* rs = static_cast<RefSource*>(pl->insert_stage(std::move(rs_u)));
  auto st_u = std::make_unique<Cac>(
      &sess, "cac", std::vector<InEdge>{{fs, 0}, {rs, 0}}, std::move(cfg));
  auto* st = static_cast<Cac*>(pl->insert_stage(std::move(st_u)));
  auto sk_u = std::make_unique<FrameSink>(&sess, "sink",
                                          std::vector<InEdge>{{st, 0}},
                                          FlexData::make_object());
  auto* sk = static_cast<FrameSink*>(pl->insert_stage(std::move(sk_u)));
  PipelineRuntime rt(pl.get(), &sess);
  Run r;
  if (!rt.launch()) { return r; }
  rt.wait_idle();
  rt.stop();
  r.frames = sk->frames;
  r.order = sk->order;
  r.shape_ok = sk->shape_ok;
  return r;
}

}  // namespace

TEST(content_aware_continuation, estimates_the_step_that_made_the_clip)
{
  const std::vector<float> ref = texture_(1);
  const std::vector<float> cont = continuation_of_(ref);

  const Cac::Geometry g = Cac::estimate_geometry(ref, cont, kW, kH);
  std::printf("[content_aware_continuation] geometry: sx %.5f sy %.5f "
              "tx %+.3f ty %+.3f from %d of %d blocks\n", g.sx, g.sy, g.tx,
              g.ty, g.blocks_used, g.blocks_total);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  EXPECT_TRUE(std::fabs(g.sy - kSy) < 0.0015);
  EXPECT_TRUE(std::fabs(g.sx - kSx) < 0.0015);
  EXPECT_TRUE(std::fabs(g.tx - kTx) < 0.25);
  EXPECT_TRUE(std::fabs(g.ty - kTy) < 0.25);

  const Cac::Tone t = Cac::estimate_tone(ref, cont, kW, kH, g);
  ASSERT_TRUE(t.ok);
  // The curve undoes the darkening where there is data: a level the
  // continuation shows at darken_(v) maps back to about v.
  for (int v : {24, 64, 128, 200}) {
    const int d = (int)std::lround(darken_((float)v));
    const float back = t.lut[1][(size_t)d];
    std::printf("[content_aware_continuation] tone: %3d -> %3d -> %6.2f\n", v,
                d, back);
    EXPECT_TRUE(std::fabs(back - (float)v) < 1.5f);
  }

  // And the two together put the reference back.
  const double before = psnr_(cont, ref);
  const double after = psnr_(correct_(cont, g, t), ref);
  std::printf("[content_aware_continuation] PSNR vs reference: %.2f dB "
              "before, %.2f dB after\n", before, after);
  EXPECT_TRUE(after > before + 8.0);
  EXPECT_TRUE(after > 33.0);
}

TEST(content_aware_continuation, the_same_frame_needs_no_correction)
{
  const std::vector<float> ref = texture_(2);
  const Cac::Geometry g = Cac::estimate_geometry(ref, ref, kW, kH);
  ASSERT_TRUE(g.ok);
  EXPECT_TRUE(g.identity());
  const Cac::Tone t = Cac::estimate_tone(ref, ref, kW, kH, g);
  ASSERT_TRUE(t.ok);
  EXPECT_TRUE(t.identity());
}

TEST(content_aware_continuation, an_unrelated_clip_is_not_matched)
{
  // A different picture is a scene change, not a drift: bending it
  // toward the reference would be the error.
  const std::vector<float> ref = texture_(3);
  const std::vector<float> other = texture_(11);
  const Cac::Geometry g = Cac::estimate_geometry(ref, other, kW, kH);
  std::printf("[content_aware_continuation] unrelated: %d of %d blocks, "
              "ok=%d\n", g.blocks_used, g.blocks_total, (int)g.ok);
  EXPECT_FALSE(g.ok);
}

TEST(content_aware_continuation, every_frame_comes_out_corrected_in_order)
{
  const std::vector<float> ref = texture_(4);
  const std::vector<float> cont = continuation_of_(ref);
  const Run r = run_(ref, cont, 10, FlexData::make_object());
  ASSERT_TRUE(r.shape_ok);
  ASSERT_TRUE(r.frames.size() == 10);
  if (r.frames.size() != 10) { return; }
  bool in_order = true;
  for (int i = 0; i < 10; ++i) {
    in_order = in_order && r.order[(size_t)i] == i;
  }
  EXPECT_TRUE(in_order);
  // The held window and the streamed tail alike.
  const double before = psnr_(cont, ref);
  double worst = 1e9;
  for (const auto& f : r.frames) { worst = std::min(worst, psnr_(f, ref)); }
  std::printf("[content_aware_continuation] graph: %.2f dB before, worst "
              "frame %.2f dB after\n", before, worst);
  EXPECT_TRUE(worst > before + 8.0);
}

TEST(content_aware_continuation, turned_off_it_is_a_copy)
{
  const std::vector<float> ref = texture_(5);
  const std::vector<float> cont = continuation_of_(ref);
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("tone", FlexData::make_bool(false));
  cfg.as_object().insert_or_assign("geometry", FlexData::make_bool(false));
  const Run r = run_(ref, cont, 6, std::move(cfg));
  ASSERT_TRUE(r.frames.size() == 6);
  bool same = !r.frames.empty();
  for (const auto& f : r.frames) {
    for (size_t i = 0; i < f.size() && same; ++i) {
      same = f[i] == (float)(std::uint8_t)std::lround(
                         std::max(0.0f, std::min(255.0f, cont[i])));
    }
  }
  EXPECT_TRUE(same);
}

TEST(content_aware_continuation, fade_frames_hands_back_to_the_model)
{
  const std::vector<float> ref = texture_(6);
  const std::vector<float> cont = continuation_of_(ref);
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("fade_frames", FlexData::make_int(4));
  const Run r = run_(ref, cont, 8, std::move(cfg));
  ASSERT_TRUE(r.frames.size() == 8);
  if (r.frames.size() != 8) { return; }
  // Full at the seam, none from frame 4 on.
  const double first = psnr_(r.frames[0], ref);
  std::printf("[content_aware_continuation] fade: frame 0 vs reference "
              "%.2f dB\n", first);
  EXPECT_TRUE(first > psnr_(cont, ref) + 8.0);
  // From frame 4 on the beat is forwarded untouched: the u8 clip exactly.
  bool untouched = true;
  for (size_t i = 0; i < cont.size() && untouched; ++i) {
    untouched = r.frames[6][i] == (float)(std::uint8_t)std::lround(
                                      std::max(0.0f,
                                               std::min(255.0f, cont[i])));
  }
  EXPECT_TRUE(untouched);
}
