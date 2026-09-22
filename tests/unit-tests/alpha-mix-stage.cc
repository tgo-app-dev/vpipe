// `alpha-mix`: Porter-Duff `over`, the optional mask port, and the two
// output modes.
//
// Every case here is driven through a real PipelineRuntime rather than
// a helper, because three of the four things that can go wrong are
// wiring rather than arithmetic: the optional third iport, which beat
// the mask is read against, and whether the output keeps four channels.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/alpha-mix-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace vpipe;
using std::string;
using std::vector;

namespace {

// One planar frame with per-channel constants. `ch` is 3 or 4 for an
// image and 1 for a mask; `vals` holds one 0..1 value per channel.
class ConstFrameSource : public TypedStage<ConstFrameSource> {
public:
  static constexpr const char* kTypeName = "ut-alpha-mix-src";
  using TypedStage::TypedStage;
  int H = 2, W = 2, ch = 3;
  bool drop_ch_dim = false;      // emit [H,W] instead of [1,H,W]
  TensorBeat::DType dt = TensorBeat::DType::F32;
  vector<float> vals{ 1.0f, 1.0f, 1.0f };
  // Optional: override the right half of each row with these values, so
  // a frame can carry two different pixels without a second source.
  bool          split = false;
  vector<float> rvals{};

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = dt;
    if (ch == 1 && drop_ch_dim) { tb.shape = { H, W }; }
    else                        { tb.shape = { ch, H, W }; }
    const size_t plane = static_cast<size_t>(H) * W;
    tb.resize_contiguous(static_cast<size_t>(ch) * plane);
    for (int c = 0; c < ch; ++c) {
      for (size_t i = 0; i < plane; ++i) {
        const bool right = split
            && static_cast<int>(i % static_cast<size_t>(W)) >= W / 2;
        const float v = right ? rvals[static_cast<size_t>(c)]
                              : vals[static_cast<size_t>(c)];
        const size_t o = static_cast<size_t>(c) * plane + i;
        if (dt == TensorBeat::DType::U8) {
          tb.as_u8()[o] =
              static_cast<std::uint8_t>(std::lrintf(v * 255.0f));
        } else {
          tb.as_f32()[o] = v;
        }
      }
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  bool _sent = false;
};

class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-alpha-mix-sink";
  using TypedStage::TypedStage;
  vector<TensorBeat>& out() { return _out; }
  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* p = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      std::lock_guard<std::mutex> g(_mu);
      _out.push_back(static_cast<const TensorBeat&>(*p));
    }
  }
private:
  std::mutex         _mu;
  vector<TensorBeat> _out;
};

struct Src {
  int H = 2, W = 2, ch = 3;
  bool drop_ch_dim = false;
  TensorBeat::DType dt = TensorBeat::DType::F32;
  vector<float> vals{};
  bool split = false;
  vector<float> rvals{};
};

// Drive `front` over `back` (+ optional mask) through the stage.
vector<TensorBeat>
run_mix(Session& sess, const Src& front, const Src& back,
        const Src* mask, FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto mk = [&](const Src& d, const char* id) -> Stage* {
    auto s = std::make_unique<ConstFrameSource>(
        &sess, id, vector<InEdge>{}, FlexData::make_object());
    s->H = d.H; s->W = d.W; s->ch = d.ch; s->dt = d.dt;
    s->vals = d.vals; s->drop_ch_dim = d.drop_ch_dim;
    s->split = d.split; s->rvals = d.rvals;
    s->allocate_oports(1);
    return pl->insert_stage(std::move(s));
  };
  Stage* f = mk(front, "front");
  Stage* b = mk(back, "back");
  vector<InEdge> edges{ { f, 0 }, { b, 0 } };
  if (mask != nullptr) { edges.push_back({ mk(*mask, "mask"), 0 }); }
  auto am = std::make_unique<AlphaMixStage>(
      &sess, "am", std::move(edges), std::move(cfg));
  auto* amp = static_cast<AlphaMixStage*>(pl->insert_stage(std::move(am)));
  auto sk = std::make_unique<Sink>(
      &sess, "sink", vector<InEdge>{ { amp, 0 } }, FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(sk)));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return sink->out();
}

FlexData
cfg_(const char* output = nullptr, const char* bg = nullptr)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  if (output) { o.insert("output", FlexData::make_string(output)); }
  if (bg)     { o.insert("background", FlexData::make_string(bg)); }
  return c;
}

// Channel `c` of pixel `i`, as 0..1.
float
at_(const TensorBeat& tb, int c, size_t i)
{
  const size_t plane =
      static_cast<size_t>(tb.shape[1]) * static_cast<size_t>(tb.shape[2]);
  const size_t o = static_cast<size_t>(c) * plane + i;
  if (tb.dtype == TensorBeat::DType::U8) {
    return static_cast<float>(tb.as_u8()[o]) / 255.0f;
  }
  return tb.as_f32()[o];
}

bool
near_(float a, float b, float tol = 2e-3f)
{
  return std::fabs(a - b) <= tol;
}

Src rgb_(float r, float g, float b)
{
  Src s; s.ch = 3; s.vals = { r, g, b }; return s;
}
Src rgba_(float r, float g, float b, float a)
{
  Src s; s.ch = 4; s.vals = { r, g, b, a }; return s;
}

}  // namespace

TEST(alpha_mix_stage, type_is_registered)
{
  EXPECT_TRUE(string(AlphaMixStage::kTypeName) == "alpha-mix");
}

TEST(alpha_mix_stage, spec_has_two_images_an_optional_mask_and_one_oport)
{
  Session sess;
  AlphaMixStage s(&sess, "am", vector<InEdge>{}, cfg_());
  const StageSpec& sp = s.spec();
  EXPECT_TRUE(sp.category == StageCategory::Visual);
  EXPECT_TRUE(sp.iports.size() == 3);
  EXPECT_TRUE(sp.oports.size() == 1);
  EXPECT_TRUE(string(sp.iports[0].name) == "front");
  EXPECT_TRUE(string(sp.iports[1].name) == "back");
  EXPECT_TRUE(string(sp.iports[2].name) == "alpha");
  // All three inputs and the output are one clock domain: a composite
  // is 1:1:1 -> 1, so a beat on any of them is a beat on all.
  EXPECT_TRUE(sp.iports[0].clock_group == sp.oports[0].clock_group);
  EXPECT_TRUE(sp.iports[1].clock_group == sp.oports[0].clock_group);
  EXPECT_TRUE(sp.iports[2].clock_group == sp.oports[0].clock_group);
  EXPECT_TRUE(s.config_error().empty());
}

TEST(alpha_mix_stage, a_bad_output_mode_is_refused_at_config)
{
  Session sess;
  AlphaMixStage bad(&sess, "am", vector<InEdge>{}, cfg_("rgbx"));
  EXPECT_FALSE(bad.config_error().empty());
  AlphaMixStage ok(&sess, "am", vector<InEdge>{}, cfg_("rgba"));
  EXPECT_TRUE(ok.config_error().empty());
}

// The base case: an opaque front hides the back entirely, and the
// default output is three channels.
TEST(alpha_mix_stage, an_opaque_front_replaces_the_back)
{
  Session sess;
  auto out = run_mix(sess, rgb_(1.0f, 0.0f, 0.0f), rgb_(0.0f, 0.0f, 1.0f),
                     nullptr, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(out[0].shape.size() == 3 && out[0].shape[0] == 3);
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 1.0f));
  EXPECT_TRUE(near_(at_(out[0], 2, 0), 0.0f));
}

// Half-transparent white over opaque black is the midpoint, and NOT
// the front -- which is what says the blend ran at all.
TEST(alpha_mix_stage, a_half_transparent_front_is_the_midpoint)
{
  Session sess;
  auto out = run_mix(sess, rgba_(1.0f, 1.0f, 1.0f, 0.5f),
                     rgb_(0.0f, 0.0f, 0.0f), nullptr, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  for (int c = 0; c < 3; ++c) {
    EXPECT_TRUE(near_(at_(out[0], c, 0), 0.5f));
  }
  std::printf("[alpha_mix] 0.5 white over black -> %.4f\n",
              at_(out[0], 0, 0));
}

// THE TERM THAT GETS DROPPED. With BOTH layers half-transparent the
// back's weight is aB*(1-aF) = 0.25, not (1-aF) = 0.5, so the result is
// 0.5/0.75 = 0.667 white and 0.75 alpha. Weighting the back by (1-aF)
// alone -- the usual slip, invisible against an opaque back -- gives
// 0.5 and 0.5 instead, which both bars below reject.
TEST(alpha_mix_stage, the_back_alpha_weights_the_back)
{
  Session sess;
  auto out = run_mix(sess, rgba_(1.0f, 1.0f, 1.0f, 0.5f),
                     rgba_(0.0f, 0.0f, 0.0f, 0.5f), nullptr, cfg_("rgba"));
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(out[0].shape[0] == 4);
  EXPECT_TRUE(near_(at_(out[0], 3, 0), 0.75f));
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 2.0f / 3.0f));
  EXPECT_FALSE(near_(at_(out[0], 0, 0), 0.5f));
  std::printf("[alpha_mix] 0.5 over 0.5: a=%.4f rgb=%.4f\n",
              at_(out[0], 3, 0), at_(out[0], 0, 0));
}

// rgb output flattens what is left onto the background colour, and
// `background` is what decides it. Two transparent layers leave alpha
// 0.75 above, so a quarter of the background shows through.
TEST(alpha_mix_stage, rgb_output_flattens_onto_the_background_colour)
{
  Session sess;
  auto white = run_mix(sess, rgba_(0.0f, 0.0f, 0.0f, 0.5f),
                       rgba_(0.0f, 0.0f, 0.0f, 0.5f), nullptr,
                       cfg_("rgb", "#FFFFFF"));
  ASSERT_TRUE(white.size() == 1);
  auto black = run_mix(sess, rgba_(0.0f, 0.0f, 0.0f, 0.5f),
                       rgba_(0.0f, 0.0f, 0.0f, 0.5f), nullptr,
                       cfg_("rgb", "#000000"));
  ASSERT_TRUE(black.size() == 1);
  if (white.size() != 1 || black.size() != 1) { return; }
  EXPECT_TRUE(white[0].shape[0] == 3);
  EXPECT_TRUE(near_(at_(white[0], 0, 0), 0.25f));   // 1 - 0.75
  EXPECT_TRUE(near_(at_(black[0], 0, 0), 0.0f));
  // A colour, not just the two extremes: #FF0000 shows only in red.
  auto red = run_mix(sess, rgba_(0.0f, 0.0f, 0.0f, 0.5f),
                     rgba_(0.0f, 0.0f, 0.0f, 0.5f), nullptr,
                     cfg_("rgb", "#FF0000"));
  ASSERT_TRUE(red.size() == 1);
  if (red.size() != 1) { return; }
  EXPECT_TRUE(near_(at_(red[0], 0, 0), 0.25f));
  EXPECT_TRUE(near_(at_(red[0], 1, 0), 0.0f));
}

// The mask port against an RGB front: with no alpha of its own the
// front's alpha IS the mask, so this is the plain stencil case. Split
// frame -- left masked in, right masked out -- so one run shows both.
TEST(alpha_mix_stage, the_mask_is_the_alpha_of_an_rgb_front)
{
  Session sess;
  Src m; m.ch = 1; m.dt = TensorBeat::DType::U8; m.vals = { 1.0f };
  m.split = true; m.rvals = { 0.0f };
  m.W = 2; m.H = 2;
  auto out = run_mix(sess, rgb_(1.0f, 1.0f, 1.0f), rgb_(0.0f, 0.0f, 0.0f),
                     &m, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 1.0f));   // left: front
  EXPECT_TRUE(near_(at_(out[0], 0, 1), 0.0f));   // right: back
}

// [H,W] is the same mask written shorter, and create-mask's `in-mask`
// port accepts both -- so this one does too.
TEST(alpha_mix_stage, a_two_dimensional_mask_is_accepted)
{
  Session sess;
  Src m; m.ch = 1; m.drop_ch_dim = true; m.dt = TensorBeat::DType::F32;
  m.vals = { 0.5f };
  auto out = run_mix(sess, rgb_(1.0f, 1.0f, 1.0f), rgb_(0.0f, 0.0f, 0.0f),
                     &m, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 0.5f));
}

// MULTIPLIES, it does not replace. A 0.5 front under a 0.5 mask is
// 0.25 -- a mask that REPLACED the alpha would give 0.5, and a mask
// that was ignored would give 0.5 as well, so one bar rejects both.
TEST(alpha_mix_stage, the_mask_multiplies_an_rgba_fronts_own_alpha)
{
  Session sess;
  Src m; m.ch = 1; m.dt = TensorBeat::DType::F32; m.vals = { 0.5f };
  auto out = run_mix(sess, rgba_(1.0f, 1.0f, 1.0f, 0.5f),
                     rgb_(0.0f, 0.0f, 0.0f), &m, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 0.25f));
  EXPECT_FALSE(near_(at_(out[0], 0, 0), 0.5f));
}

// An f32 mask is COVERAGE in [0,1] whatever the images around it are,
// so `input_normalized: false` -- an 0..255 f32 image graph -- must not
// scale it. Scaling it would give alpha 0.002 and a front that never
// appears.
TEST(alpha_mix_stage, input_normalized_does_not_scale_an_f32_mask)
{
  Session sess;
  FlexData c = cfg_();
  c.as_object().insert("input_normalized", FlexData::make_bool(false));
  Src f = rgb_(255.0f, 255.0f, 255.0f);
  Src b = rgb_(0.0f, 0.0f, 0.0f);
  Src m; m.ch = 1; m.dt = TensorBeat::DType::F32; m.vals = { 1.0f };
  auto out = run_mix(sess, f, b, &m, std::move(c));
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(near_(out[0].as_f32()[0], 255.0f, 0.6f));
}

// The output dtype follows the FRONT, so a U8 graph stays U8 rather
// than silently widening.
TEST(alpha_mix_stage, the_output_dtype_follows_the_front)
{
  Session sess;
  Src f = rgba_(1.0f, 1.0f, 1.0f, 0.5f);
  f.dt = TensorBeat::DType::U8;
  Src b = rgb_(0.0f, 0.0f, 0.0f);
  b.dt = TensorBeat::DType::U8;
  auto out = run_mix(sess, f, b, nullptr, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(out[0].dtype == TensorBeat::DType::U8);
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 0.5f, 3e-3f));
}

// Mismatched sizes are REFUSED rather than fitted: a composite needs
// one coordinate space and image-resample is where that is said.
TEST(alpha_mix_stage, mismatched_sizes_drop_the_beat)
{
  Session sess;
  Src f = rgb_(1.0f, 1.0f, 1.0f);
  Src b = rgb_(0.0f, 0.0f, 0.0f);
  b.W = 4; b.H = 4;
  auto out = run_mix(sess, f, b, nullptr, cfg_());
  EXPECT_TRUE(out.empty());
}

// A mask at the wrong size is a warning, not a dropped frame: the
// composite still has a well-defined answer (the front's own alpha),
// and losing the picture entirely would be the worse failure.
TEST(alpha_mix_stage, a_wrong_sized_mask_falls_back_to_the_front_alpha)
{
  Session sess;
  Src m; m.ch = 1; m.W = 4; m.H = 4; m.dt = TensorBeat::DType::F32;
  m.vals = { 0.0f };
  auto out = run_mix(sess, rgba_(1.0f, 1.0f, 1.0f, 1.0f),
                     rgb_(0.0f, 0.0f, 0.0f), &m, cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 1.0f));
}

// ---- against the real create-mask stage ------------------------------
//
// Everything above feeds a mask this file SYNTHESISED in what it
// believes create-mask's format to be, which is exactly the shape of
// assumption that goes stale silently. So the last two cases wire an
// actual `create-mask` in, headless (`interactive: false`, where the
// mask comes from `in-mask` and is emitted as it arrives), once in each
// of its two output dtypes. If create-mask ever changes what it emits,
// these fail and the synthetic cases above do not.

namespace {

vector<TensorBeat>
run_mix_with_create_mask(Session& sess, const Src& front, const Src& back,
                         const Src& in_mask, const char* mask_dtype,
                         FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto mk = [&](const Src& d, const char* id) -> Stage* {
    auto s = std::make_unique<ConstFrameSource>(
        &sess, id, vector<InEdge>{}, FlexData::make_object());
    s->H = d.H; s->W = d.W; s->ch = d.ch; s->dt = d.dt;
    s->vals = d.vals; s->drop_ch_dim = d.drop_ch_dim;
    s->split = d.split; s->rvals = d.rvals;
    s->allocate_oports(1);
    return pl->insert_stage(std::move(s));
  };
  Stage* f  = mk(front, "front");
  Stage* b  = mk(back,  "back");
  Stage* im = mk(in_mask, "in-mask");

  FlexData mcfg = FlexData::make_object();
  {
    auto o = mcfg.as_object();
    o.insert("interactive",   FlexData::make_bool(false));
    o.insert("mask_mode",     FlexData::make_string("alpha"));
    o.insert("output",        FlexData::make_string("mask"));
    o.insert("output_dtype",  FlexData::make_string(mask_dtype));
    o.insert("width",         FlexData::make_int(in_mask.W));
    o.insert("height",        FlexData::make_int(in_mask.H));
  }
  auto cm = StageRegistry::get().create(
      "create-mask", &sess, "cm",
      vector<InEdge>{ { nullptr, 0 }, { im, 0 } }, std::move(mcfg));
  if (!cm) { return {}; }
  Stage* cms = pl->insert_stage(std::move(cm));

  auto am = std::make_unique<AlphaMixStage>(
      &sess, "am", vector<InEdge>{ { f, 0 }, { b, 0 }, { cms, 0 } },
      std::move(cfg));
  auto* amp = static_cast<AlphaMixStage*>(pl->insert_stage(std::move(am)));
  auto sk = std::make_unique<Sink>(
      &sess, "sink", vector<InEdge>{ { amp, 0 } }, FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(sk)));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return sink->out();
}

}  // namespace

TEST(alpha_mix_stage, a_headless_create_mask_drives_the_alpha_port_u8)
{
  Session sess;
  Src m; m.ch = 1; m.dt = TensorBeat::DType::U8; m.vals = { 1.0f };
  m.split = true; m.rvals = { 0.0f };
  auto out = run_mix_with_create_mask(
      sess, rgb_(1.0f, 1.0f, 1.0f), rgb_(0.0f, 0.0f, 0.0f), m, "u8",
      cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 1.0f));   // left: front
  EXPECT_TRUE(near_(at_(out[0], 0, 1), 0.0f));   // right: back
}

// create-mask's f32 mode emits coverage in [0,1], and a mid-grey u8
// mask comes out as ~0.5 -- so the composite is the midpoint. A 0..255
// reading of that would be a fully opaque front instead.
TEST(alpha_mix_stage, a_headless_create_mask_drives_the_alpha_port_f32)
{
  Session sess;
  Src m; m.ch = 1; m.dt = TensorBeat::DType::U8;
  m.vals = { 128.0f / 255.0f };
  auto out = run_mix_with_create_mask(
      sess, rgb_(1.0f, 1.0f, 1.0f), rgb_(0.0f, 0.0f, 0.0f), m, "f32",
      cfg_());
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  std::printf("[alpha_mix] create-mask f32 128/255 -> %.4f\n",
              at_(out[0], 0, 0));
  EXPECT_TRUE(near_(at_(out[0], 0, 0), 128.0f / 255.0f, 6e-3f));
}
