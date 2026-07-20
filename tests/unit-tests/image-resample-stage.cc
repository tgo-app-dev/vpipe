#include "minitest.h"
#include "stages/audio-video/image-resample-stage.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace vpipe;
using std::string;
using std::vector;

namespace {

// Emit one planar RGB [3,H,W] frame filled with a constant value.
class OneFrameSource : public TypedStage<OneFrameSource> {
public:
  static constexpr const char* kTypeName = "ut-resample-src";
  using TypedStage::TypedStage;
  int H = 0, W = 0;
  TensorBeat::DType dt = TensorBeat::DType::U8;
  float fill = 200.0f;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = dt;
    tb.shape = { 3, H, W };
    tb.resize_contiguous(static_cast<size_t>(3) * H * W);
    const size_t n = static_cast<size_t>(3) * H * W;
    if (dt == TensorBeat::DType::U8) {
      std::memset(tb.as_u8(), static_cast<int>(fill), n);
    } else {
      float* f = tb.as_f32();
      for (size_t i = 0; i < n; ++i) { f[i] = fill; }
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  bool _sent = false;
};

// Emit one planar RGB [3,H,W] frame with a PADDED row pitch (strides
// set, pitch P > W), mimicking FFmpeg's 32-byte-aligned GBRP linesize
// that load-image forwards. The valid [0,W) columns hold `fill`; the
// pad columns [W,P) hold a distinct `sentinel`. A consumer that ignores
// strides and reads the buffer as tight (pitch W) drifts into the pad /
// wrong plane and leaks the sentinel -- the shear this regresses.
class StridedFrameSource : public TypedStage<StridedFrameSource> {
public:
  static constexpr const char* kTypeName = "ut-resample-src-strided";
  using TypedStage::TypedStage;
  int H = 0, W = 0;
  TensorBeat::DType dt = TensorBeat::DType::U8;
  float fill = 200.0f, sentinel = 17.0f;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    const int64_t P = (W + 31) / 32 * 32;   // 32-byte aligned pitch
    TensorBeat tb;
    tb.dtype   = dt;
    tb.shape   = { 3, H, W };
    tb.strides = { static_cast<int64_t>(H) * P, P, 1 };
    const size_t elems = static_cast<size_t>(3) * H * P;
    const size_t esz   = TensorBeat::byte_size_of(dt);
    tb.data.assign(elems * esz, 0);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int64_t x = 0; x < P; ++x) {
          const size_t i =
              static_cast<size_t>(c) * H * P + static_cast<size_t>(y) * P + x;
          const float v = (x < W) ? fill : sentinel;
          if (dt == TensorBeat::DType::U8) {
            tb.as_u8()[i] = static_cast<uint8_t>(v);
          } else {
            tb.as_f32()[i] = v;
          }
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
  static constexpr const char* kTypeName = "ut-resample-sink";
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

// Drive one frame through an image-resample stage; return the outputs.
vector<TensorBeat> run_resample(Session& sess, int H, int W, TensorBeat::DType dt,
                       float fill, FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto s = std::make_unique<OneFrameSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  s->H = H; s->W = W; s->dt = dt; s->fill = fill;
  s->allocate_oports(1);
  auto* src = static_cast<OneFrameSource*>(pl->insert_stage(std::move(s)));
  auto rs = std::make_unique<ImageResampleStage>(
      &sess, "rs", vector<InEdge>{ { src, 0 } }, std::move(cfg));
  auto* rst = static_cast<ImageResampleStage*>(
      pl->insert_stage(std::move(rs)));
  auto sk = std::make_unique<Sink>(
      &sess, "sink", vector<InEdge>{ { rst, 0 } }, FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(sk)));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return sink->out();
}

// Like run_resample but the source emits a padded (strided) frame.
vector<TensorBeat> run_resample_strided(Session& sess, int H, int W,
                       TensorBeat::DType dt, float fill, float sentinel,
                       FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto s = std::make_unique<StridedFrameSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  s->H = H; s->W = W; s->dt = dt; s->fill = fill; s->sentinel = sentinel;
  s->allocate_oports(1);
  auto* src = static_cast<StridedFrameSource*>(pl->insert_stage(std::move(s)));
  auto rs = std::make_unique<ImageResampleStage>(
      &sess, "rs", vector<InEdge>{ { src, 0 } }, std::move(cfg));
  auto* rst = static_cast<ImageResampleStage*>(
      pl->insert_stage(std::move(rs)));
  auto sk = std::make_unique<Sink>(
      &sess, "sink", vector<InEdge>{ { rst, 0 } }, FlexData::make_object());
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::move(sk)));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return sink->out();
}

FlexData mkcfg(int w, int h, const char* fit, const char* pad = nullptr)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert("width",  FlexData::make_int(w));
  o.insert("height", FlexData::make_int(h));
  o.insert("fit",    FlexData::make_string(fit));
  if (pad) { o.insert("pad_color", FlexData::make_string(pad)); }
  return c;
}

// Config that omits an axis (pass <= 0) so it is inferred from the other.
FlexData mkcfg_wh(int w, int h)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  if (w > 0) { o.insert("width",  FlexData::make_int(w)); }
  if (h > 0) { o.insert("height", FlexData::make_int(h)); }
  return c;
}

}  // namespace

TEST(image_resample_stage, type_is_registered) {
  EXPECT_TRUE(string(ImageResampleStage::kTypeName) == "image-resample");
}

TEST(image_resample_stage, spec_ports_same_clock_domain) {
  Session sess;
  ImageResampleStage s(&sess, "rs", vector<InEdge>{}, mkcfg(64, 64, "pad"));
  const StageSpec& sp = s.spec();
  EXPECT_TRUE(sp.category == StageCategory::Visual);
  EXPECT_TRUE(sp.iports.size() == 1);
  EXPECT_TRUE(sp.oports.size() == 1);
  // iport and oport in the SAME clock domain (1:1).
  EXPECT_TRUE(sp.iports[0].clock_group == sp.oports[0].clock_group);
  EXPECT_TRUE(string(sp.iports[0].tags).find("rgb-frames") != string::npos);
  EXPECT_TRUE(string(sp.oports[0].tags).find("rgb-frames") != string::npos);
}

TEST(image_resample_stage, config_parse_and_validation) {
  Session sess;
  {
    ImageResampleStage s(&sess, "rs", vector<InEdge>{},
                         mkcfg(120, 80, "crop"));
    EXPECT_TRUE(s.config_error().empty());
    EXPECT_TRUE(s.out_width() == 120 && s.out_height() == 80);
    EXPECT_TRUE(s.fit_mode() == 1);   // crop
  }
  {  // missing width -> ok (inferred from height); see also
     // config_infer_dimension for the missing-both error case.
    FlexData c = FlexData::make_object();
    c.as_object().insert("height", FlexData::make_int(64));
    ImageResampleStage s(&sess, "rs", vector<InEdge>{}, std::move(c));
    EXPECT_TRUE(s.config_error().empty());
  }
  {  // bad algorithm -> deferred config error
    ImageResampleStage s(&sess, "rs", vector<InEdge>{},
        FlexData::from_json(R"({"width":8,"height":8,"algorithm":"lanczos"})"));
    EXPECT_FALSE(s.config_error().empty());
  }
}

TEST(image_resample_stage, config_infer_dimension) {
  Session sess;
  {  // width only -> ok, height inferred (0 = auto)
    ImageResampleStage s(&sess, "rs", vector<InEdge>{}, mkcfg_wh(1280, 0));
    EXPECT_TRUE(s.config_error().empty());
    EXPECT_TRUE(s.out_width() == 1280 && s.out_height() == 0);
  }
  {  // height only -> ok, width inferred
    ImageResampleStage s(&sess, "rs", vector<InEdge>{}, mkcfg_wh(0, 720));
    EXPECT_TRUE(s.config_error().empty());
    EXPECT_TRUE(s.out_width() == 0 && s.out_height() == 720);
  }
  {  // neither -> config error
    ImageResampleStage s(&sess, "rs", vector<InEdge>{}, mkcfg_wh(0, 0));
    EXPECT_FALSE(s.config_error().empty());
  }
}

// ---- functional (needs a metal-compute backend) --------------------
static uint8_t px_u8(const TensorBeat& t, int c, int r, int col) {
  const int W = static_cast<int>(t.shape[2]);
  const int H = static_cast<int>(t.shape[1]);
  return t.as_u8()[c * W * H + r * W + col];
}

TEST(image_resample_stage, metal_pad_letterbox) {
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) { return; }
  // 4x2 (W=4,H=2) solid 200 -> 8x8 pad. scale=min(2,4)=2 => new 8x4,
  // pad_y=2: rows 0-1 + 6-7 are grey 114 pad, rows 2-5 are the image (200).
  auto got = run_resample(sess, 2, 4, TensorBeat::DType::U8, 200.0f,
                 mkcfg(8, 8, "pad"));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape.size() == 3 && o.shape[0] == 3
              && o.shape[1] == 8 && o.shape[2] == 8);
  EXPECT_TRUE(o.dtype == TensorBeat::DType::U8);
  EXPECT_TRUE(px_u8(o, 0, 0, 0) == 114);   // pad row
  EXPECT_TRUE(px_u8(o, 0, 3, 0) == 200);   // image row
  EXPECT_TRUE(px_u8(o, 0, 7, 0) == 114);   // pad row
}

TEST(image_resample_stage, metal_stretch_fills) {
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) { return; }
  auto got = run_resample(sess, 2, 4, TensorBeat::DType::U8, 200.0f,
                 mkcfg(8, 8, "stretch"));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape[1] == 8 && o.shape[2] == 8);
  // No padding: corners are the (solid) image.
  EXPECT_TRUE(px_u8(o, 0, 0, 0) == 200);
  EXPECT_TRUE(px_u8(o, 0, 7, 7) == 200);
}

TEST(image_resample_stage, metal_pad_color) {
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) { return; }
  auto got = run_resample(sess, 2, 4, TensorBeat::DType::U8, 200.0f,
                 mkcfg(8, 8, "pad", "#FF0000"));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(px_u8(o, 0, 0, 0) == 255);   // R pad
  EXPECT_TRUE(px_u8(o, 1, 0, 0) == 0);     // G pad
  EXPECT_TRUE(px_u8(o, 2, 0, 0) == 0);     // B pad
}

// Regression: a padded (strided) source must NOT shear. W=40 -> pitch
// 64, so a tight-pitch read drifts 24 px/row into the pad/next plane and
// leaks the sentinel. Stretch 1:1 keeps the solid fill everywhere; any
// sentinel value in the output means strides were ignored.
TEST(image_resample_stage, metal_strided_input_no_shear) {
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) { return; }
  auto got = run_resample_strided(sess, 8, 40, TensorBeat::DType::U8,
                 200.0f, 17.0f, mkcfg(40, 8, "stretch"));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape[1] == 8 && o.shape[2] == 40);
  bool all_fill = true;
  for (int c = 0; c < 3 && all_fill; ++c) {
    for (int r = 0; r < 8 && all_fill; ++r) {
      for (int col = 0; col < 40; ++col) {
        if (px_u8(o, c, r, col) != 200) { all_fill = false; break; }
      }
    }
  }
  EXPECT_TRUE(all_fill);
}

TEST(image_resample_stage, cpu_strided_input_no_shear) {
  Session sess;
  // f32 exercises the CPU fallback, which must also honour strides.
  auto got = run_resample_strided(sess, 8, 40, TensorBeat::DType::F32,
                 0.5f, -1.0f, mkcfg(40, 8, "stretch"));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape[1] == 8 && o.shape[2] == 40);
  const float* f = o.as_f32();
  const size_t n = static_cast<size_t>(3) * 8 * 40;
  bool all_fill = true;
  for (size_t i = 0; i < n; ++i) {
    if (f[i] < 0.49f || f[i] > 0.51f) { all_fill = false; break; }
  }
  EXPECT_TRUE(all_fill);
}

// Inference: a 16x8 (W:H = 2:1) source with only width=8 given must
// yield an 8x4 output (height inferred, aspect preserved).
TEST(image_resample_stage, metal_infer_height_from_width) {
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) { return; }
  auto got = run_resample(sess, 8, 16, TensorBeat::DType::U8, 200.0f,
                 mkcfg_wh(8, 0));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape[0] == 3 && o.shape[1] == 4 && o.shape[2] == 8);
  EXPECT_TRUE(px_u8(o, 0, 0, 0) == 200);   // AR match -> no pad, solid fill
  EXPECT_TRUE(px_u8(o, 2, 3, 7) == 200);
}

// Same source, only height=4 given -> width inferred to 8. f32 exercises
// the CPU path (always runs).
TEST(image_resample_stage, cpu_infer_width_from_height) {
  Session sess;
  auto got = run_resample(sess, 8, 16, TensorBeat::DType::F32, 0.5f,
                 mkcfg_wh(0, 4));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape[0] == 3 && o.shape[1] == 4 && o.shape[2] == 8);
  const float* f = o.as_f32();
  EXPECT_TRUE(f[0] > 0.49f && f[0] < 0.51f);
}

TEST(image_resample_stage, cpu_f32_resample) {
  Session sess;
  // f32 always uses the CPU path (runs with or without metal).
  auto got = run_resample(sess, 2, 4, TensorBeat::DType::F32, 0.5f,
                 mkcfg(8, 8, "stretch"));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.dtype == TensorBeat::DType::F32);
  EXPECT_TRUE(o.shape[1] == 8 && o.shape[2] == 8);
  const float* f = o.as_f32();
  EXPECT_TRUE(f[0] > 0.49f && f[0] < 0.51f);   // solid 0.5 preserved
}
