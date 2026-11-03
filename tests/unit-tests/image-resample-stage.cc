#include "minitest.h"
#include "stages/audio-video/image-resample-stage.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
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
  // A vertical STEP EDGE instead of a solid fill: `fill` on the left
  // half, 0 on the right. A solid image is a fixed point of every
  // sensible resampling filter, so nothing uniform can say WHICH filter
  // ran -- which is why every other test here passes under either.
  bool edge = false;

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
          float v = (x < W) ? fill : sentinel;
          if (edge && x < W && x >= W / 2) { v = 0.0f; }
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
  // Whether each beat arrived in GPU (Shared Metal) storage -- said here
  // because the copy in out() does not keep it.
  vector<bool>& gpu_backed() { return _gpu; }
  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* p = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      std::lock_guard<std::mutex> g(_mu);
      _gpu.push_back(p->external != nullptr);
      _out.push_back(static_cast<const TensorBeat&>(*p));
    }
  }
private:
  std::mutex         _mu;
  vector<TensorBeat> _out;
  vector<bool>       _gpu;
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
                       bool edge,
                       FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto s = std::make_unique<StridedFrameSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  s->H = H; s->W = W; s->dt = dt; s->fill = fill; s->sentinel = sentinel;
  s->edge = edge;
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

// The same config, pinned to the CPU path: f32 frames have GPU kernels
// now, so a test about the CPU code has to say so.
FlexData on_cpu(FlexData c)
{
  c.as_object().insert_or_assign("gpu", FlexData::make_bool(false));
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
  {  // lanczos -> valid algorithm
    ImageResampleStage s(&sess, "rs", vector<InEdge>{},
        FlexData::from_json(R"({"width":8,"height":8,"algorithm":"lanczos"})"));
    EXPECT_TRUE(s.config_error().empty());
  }
  {  // unknown algorithm -> deferred config error
    ImageResampleStage s(&sess, "rs", vector<InEdge>{},
        FlexData::from_json(R"({"width":8,"height":8,"algorithm":"cubic"})"));
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
                 200.0f, 17.0f, /*edge=*/false, mkcfg(40, 8, "stretch"));
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
  // The CPU path must also honour strides.
  auto got = run_resample_strided(sess, 8, 40, TensorBeat::DType::F32,
                 0.5f, -1.0f, /*edge=*/false,
                 on_cpu(mkcfg(40, 8, "stretch")));
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

// Same source, only height=4 given -> width inferred to 8, on the CPU
// path (runs with or without a GPU).
TEST(image_resample_stage, cpu_infer_width_from_height) {
  Session sess;
  auto got = run_resample(sess, 8, 16, TensorBeat::DType::F32, 0.5f,
                 on_cpu(mkcfg_wh(0, 4)));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.shape[0] == 3 && o.shape[1] == 4 && o.shape[2] == 8);
  const float* f = o.as_f32();
  EXPECT_TRUE(f[0] > 0.49f && f[0] < 0.51f);
}

TEST(image_resample_stage, cpu_f32_resample) {
  Session sess;
  // Pinned to the CPU path (runs with or without metal).
  auto got = run_resample(sess, 2, 4, TensorBeat::DType::F32, 0.5f,
                 on_cpu(mkcfg(8, 8, "stretch")));
  ASSERT_TRUE(got.size() == 1);
  const TensorBeat& o = got[0];
  EXPECT_TRUE(o.dtype == TensorBeat::DType::F32);
  EXPECT_TRUE(o.shape[1] == 8 && o.shape[2] == 8);
  const float* f = o.as_f32();
  EXPECT_TRUE(f[0] > 0.49f && f[0] < 0.51f);   // solid 0.5 preserved
}


// WHICH FILTER the default selects -- the one thing the tests above
// cannot say.
//
// Every other case here resamples a SOLID fill, and a constant image is
// a fixed point of any sensible resampling filter, so all of them pass
// under either. This one downscales a STEP EDGE, where the two filters
// genuinely disagree, and pins the default by identity: the unset key
// must produce exactly what "lanczos" produces, and must NOT produce
// what "bilinear" produces.
//
// Stated that way rather than by asserting pixel values, so it keeps
// meaning the same thing if either kernel is ever retuned.
namespace {

FlexData mkcfg_alg(int w, int h, const char* alg)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert("width",  FlexData::make_int(w));
  o.insert("height", FlexData::make_int(h));
  o.insert("fit",    FlexData::make_string("stretch"));
  if (alg != nullptr) { o.insert("algorithm", FlexData::make_string(alg)); }
  return c;
}

// f32 so the comparison is not flattened by u8 quantisation. Every run
// takes the same path (the GPU when there is one), so the identity below
// is about which filter the default selects and nothing else.
vector<float> edge_resample(Session& sess, const char* alg)
{
  auto got = run_resample_strided(sess, 8, 64, TensorBeat::DType::F32,
                 1.0f, -1.0f, /*edge=*/true, mkcfg_alg(16, 8, alg));
  if (got.size() != 1) { return {}; }
  const TensorBeat& o = got[0];
  const float* f = o.as_f32();
  return vector<float>(f, f + static_cast<size_t>(3) * 8 * 16);
}

}  // namespace

TEST(image_resample_stage, default_algorithm_is_lanczos) {
  Session sess;
  const vector<float> dflt = edge_resample(sess, nullptr);
  const vector<float> lanc = edge_resample(sess, "lanczos");
  const vector<float> bili = edge_resample(sess, "bilinear");
  ASSERT_TRUE(!dflt.empty() && dflt.size() == lanc.size());
  ASSERT_TRUE(dflt.size() == bili.size());

  // The two filters must actually differ on this input, or the identity
  // below would hold for either default and prove nothing.
  bool filters_differ = false;
  for (size_t i = 0; i < lanc.size(); ++i) {
    if (lanc[i] != bili[i]) { filters_differ = true; break; }
  }
  EXPECT_TRUE(filters_differ);

  bool same_as_lanczos = true;
  for (size_t i = 0; i < dflt.size(); ++i) {
    if (dflt[i] != lanc[i]) { same_as_lanczos = false; break; }
  }
  EXPECT_TRUE(same_as_lanczos);

  bool same_as_bilinear = true;
  for (size_t i = 0; i < dflt.size(); ++i) {
    if (dflt[i] != bili[i]) { same_as_bilinear = false; break; }
  }
  EXPECT_FALSE(same_as_bilinear);
}

// ---- RGBA ------------------------------------------------------------
//
// A four-channel beat resizes to a four-channel beat, and -- the part
// worth a test -- it resamples PREMULTIPLIED.
//
// THE TWO ANSWERS DIFFER BY A FACTOR OF TWO at a hard alpha edge, so
// this cannot pass by accident. The source is opaque WHITE on the left
// half and fully transparent BLACK on the right. Downscaling 2:1 makes
// each output pixel the average of one source pixel from each side at
// the boundary, and:
//
//   not premultiplied   rgb = (255 + 0)/2 = 127, a = 127
//                       -> a mid-grey edge that is not in the picture
//   premultiplied       rgb = (255*1 + 0*0)/2 = 127 over a = 127,
//                       divided back out -> 255, still white
//
// The transparent half's black is exactly the case a generated RGBA
// image has, which is why the halo is the normal failure and not an
// exotic one.
namespace {

class RgbaEdgeSource : public TypedStage<RgbaEdgeSource> {
public:
  static constexpr const char* kTypeName = "ut-resample-rgba-src";
  using TypedStage::TypedStage;
  int H = 4, W = 8;
  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = { 4, H, W };
    tb.resize_contiguous(static_cast<size_t>(4) * H * W);
    uint8_t* d = tb.bytes_();
    const size_t plane = static_cast<size_t>(H) * W;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t i = static_cast<size_t>(y) * W + x;
        const bool left = x < W / 2;
        for (int c = 0; c < 3; ++c) {
          d[c * plane + i] = left ? 255 : 0;
        }
        d[3 * plane + i] = left ? 255 : 0;
      }
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }
private:
  bool _sent = false;
};

vector<TensorBeat> run_rgba(Session& sess, int H, int W, FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto s = std::make_unique<RgbaEdgeSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  s->H = H; s->W = W;
  s->allocate_oports(1);
  auto* src = static_cast<RgbaEdgeSource*>(pl->insert_stage(std::move(s)));
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

}  // namespace

TEST(image_resample_stage, rgba_keeps_four_channels_and_premultiplies) {
  Session sess;
  const int H = 8, W = 16;
  // THE DEFAULT KERNEL (lanczos, support 3) ON PURPOSE. A bilinear
  // 2:1 downscale samples sources {2x, 2x+1}, so with the edge on an
  // even column NO output pixel ever straddles it, every output alpha
  // is 0 or 255, and the assertion below has nothing to bite on -- the
  // first version of this test was exactly that, and passed with the
  // premultiply deliberately disabled. A wide kernel guarantees the
  // partial-alpha band the test is about, and `partial` below refuses
  // to let it silently go away again.
  //
  // With premultiplication the expected value is EXACT and kernel-
  // independent: premultiplied rgb equals alpha everywhere here (both
  // are 255 on the opaque side and 0 on the other), so any kernel
  // gives resampled_rgb == resampled_alpha, and dividing back out is
  // 255 wherever alpha survives.
  const auto got = run_rgba(sess, H, W, mkcfg(W / 2, H / 2, "stretch"));

  ASSERT_TRUE(got.size() == 1);
  if (got.empty()) { return; }
  const TensorBeat& tb = got[0];
  EXPECT_TRUE(tb.shape.size() == 3);
  EXPECT_TRUE(tb.shape[0] == 4);
  if (tb.shape[0] != 4) { return; }
  EXPECT_TRUE(tb.shape[1] == H / 2 && tb.shape[2] == W / 2);

  const auto bytes = tb.materialize_contiguous();
  const size_t plane = static_cast<size_t>(H / 2) * (W / 2);
  EXPECT_TRUE(bytes.size() == 4 * plane);
  if (bytes.size() != 4 * plane) { return; }

  // Deep in the opaque half: white and opaque.
  const size_t li = static_cast<size_t>(1) * (W / 2) + 0;
  EXPECT_TRUE(bytes[li] > 250);
  EXPECT_TRUE(bytes[3 * plane + li] > 250);
  // Deep in the transparent half: alpha gone.
  const size_t ri = static_cast<size_t>(1) * (W / 2) + (W / 2 - 1);
  EXPECT_TRUE(bytes[3 * plane + ri] < 5);
  // EVERY pixel that retains ANY alpha must still be white -- and the
  // ones that matter are the PARTIALLY transparent ones, because a
  // fully opaque pixel is white either way. A non-premultiplied
  // resample leaves mid-grey exactly there, which is the halo.
  int partial = 0;
  for (size_t i = 0; i < plane; ++i) {
    const int a = bytes[3 * plane + i];
    if (a < 8) { continue; }                      // nothing to say
    if (a <= 247) { ++partial; }
    EXPECT_TRUE(bytes[i] > 250);
    EXPECT_TRUE(bytes[plane + i] > 250);
    EXPECT_TRUE(bytes[2 * plane + i] > 250);
  }
  // THE GUARD THAT KEEPS THIS HONEST: without a partial-alpha band
  // there is no halo to find and the loop above asserts nothing.
  EXPECT_TRUE(partial > 0);
}

// ---- f16 / f32 on the GPU ----------------------------------------------
//
// The float kernels (resample_planar_float) against the CPU path they
// replace, for EVERY mode: f16 and f32, RGB and RGBA, the three filters,
// the four fits, a downscale and an upscale. `gpu: false` pins the CPU
// path, so the two are the same stage with one key apart.
//
// THE PICTURE IS BUILT TO DISAGREE WHEREVER THE TWO COULD: a gradient
// (where the filters differ), a hard vertical edge (where Lanczos and
// bicubic ring below 0 and above 1, which neither path may clamp), a
// block at 1.5 (a linear picture's highlight, ditto), and for RGBA an
// alpha ramp with a fully transparent BLACK band -- the case
// premultiplication exists for, so the kernel's in-place premultiply is
// checked against the CPU's explicit one exactly where a mistake shows.
namespace {

class BeatSource : public TypedStage<BeatSource> {
public:
  static constexpr const char* kTypeName = "ut-resample-beat-src";
  using TypedStage::TypedStage;
  TensorBeat beat;
  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(beat));
  }
private:
  bool _sent = false;
};

float
pattern_(int c, int y, int x, int H, int W)
{
  if (c == 3) {                                  // alpha
    if (x >= W * 3 / 4) { return 0.0f; }         // transparent band
    return 0.25f + 0.75f * float(y) / float(H - 1);
  }
  if (c < 3 && x >= W * 3 / 4) { return 0.0f; }  // ... and it is black
  if (x >= W / 4 && x < W / 4 + 4 && y < H / 2) { return 1.5f; }
  const float edge = x < W / 2 ? 0.9f : 0.1f;
  return 0.5f * edge + 0.5f * (float(x + c * 7) / float(W)) *
                              (float(y) / float(H));
}

TensorBeat
picture_(TensorBeat::DType dt, int C, int H, int W)
{
  TensorBeat t;
  t.dtype = dt;
  t.shape = {C, H, W};
  const size_t n = static_cast<size_t>(C) * H * W;
  t.resize_contiguous(n);
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t i = (static_cast<size_t>(c) * H + y) * W + x;
        const float v = pattern_(c, y, x, H, W);
        if (dt == TensorBeat::DType::F16) {
          reinterpret_cast<_Float16*>(t.bytes_())[i] = (_Float16)v;
        } else {
          t.as_f32()[i] = v;
        }
      }
    }
  }
  return t;
}

vector<float>
floats_(const TensorBeat& t)
{
  const auto bytes = t.materialize_contiguous();
  vector<float> out;
  if (t.dtype == TensorBeat::DType::F16) {
    const auto* h = reinterpret_cast<const _Float16*>(bytes.data());
    for (size_t i = 0; i < bytes.size() / 2; ++i) { out.push_back(h[i]); }
  } else {
    const auto* f = reinterpret_cast<const float*>(bytes.data());
    out.assign(f, f + bytes.size() / 4);
  }
  return out;
}

vector<TensorBeat>
run_beat(Session& sess, const TensorBeat& in, FlexData cfg,
         bool* gpu_backed = nullptr)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto s = std::make_unique<BeatSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  s->beat = in;
  s->allocate_oports(1);
  auto* src = static_cast<BeatSource*>(pl->insert_stage(std::move(s)));
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
  if (gpu_backed != nullptr) {
    *gpu_backed = sink->gpu_backed().size() == 1 && sink->gpu_backed()[0];
  }
  return sink->out();
}

FlexData
mode_cfg_(int w, int h, const char* fit, const char* alg, bool gpu)
{
  FlexData c = FlexData::make_object();
  auto o = c.as_object();
  o.insert("width",     FlexData::make_int(w));
  o.insert("height",    FlexData::make_int(h));
  o.insert("fit",       FlexData::make_string(fit));
  o.insert("algorithm", FlexData::make_string(alg));
  o.insert("pad_color", FlexData::make_string("#204060"));
  o.insert("src_x",     FlexData::make_int(5));
  o.insert("src_y",     FlexData::make_int(3));
  o.insert("scale",     FlexData::make_real(0.7));
  o.insert("gpu",       FlexData::make_bool(gpu));
  return c;
}

}  // namespace

TEST(image_resample_stage, float_gpu_matches_cpu_in_every_mode) {
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) {
    std::printf("[image_resample] SKIP: no Metal device\n");
    return;
  }
  const int H = 23, W = 37;
  struct Size { int w, h; };
  const Size sizes[] = { {16, 20}, {80, 50} };      // down + portrait, up
  const char* fits[] = { "pad", "crop", "stretch", "manual" };
  const char* algs[] = { "bilinear", "lanczos", "bicubic" };
  int cases = 0, failed = 0;
  float worst_of[2] = {0.0f, 0.0f};            // f16, f32
  for (TensorBeat::DType dt : { TensorBeat::DType::F16,
                                TensorBeat::DType::F32 }) {
    const bool f16 = dt == TensorBeat::DType::F16;
    // Both paths compute in float from the same samples; f16 then rounds
    // what it stores, where one ulp near 1.5 is 2^-10.
    const float tol = f16 ? 2e-3f : 2e-5f;
    for (int C : { 3, 4 }) {
      const TensorBeat in = picture_(dt, C, H, W);
      for (const Size& sz : sizes) {
        for (const char* fit : fits) {
          for (const char* alg : algs) {
            bool on_gpu = false;
            auto g = run_beat(sess, in, mode_cfg_(sz.w, sz.h, fit, alg,
                                                  true), &on_gpu);
            auto c = run_beat(sess, in, mode_cfg_(sz.w, sz.h, fit, alg,
                                                  false));
            ++cases;
            bool ok = g.size() == 1 && c.size() == 1;
            float worst = 0.0f, lo = 0.0f, hi = 0.0f;
            if (ok) {
              ok = g[0].dtype == dt && c[0].dtype == dt
                && g[0].shape == (std::vector<int64_t>{C, sz.h, sz.w})
                && g[0].shape == c[0].shape
                // The GPU result IS a GPU buffer -- not a CPU fallback
                // that happens to agree.
                && on_gpu;
            }
            if (ok) {
              const auto a = floats_(g[0]), b = floats_(c[0]);
              ok = a.size() == b.size();
              for (size_t i = 0; ok && i < a.size(); ++i) {
                worst = std::max(worst, std::fabs(a[i] - b[i]));
                lo = std::min(lo, a[i]);
                hi = std::max(hi, a[i]);
              }
              ok = ok && worst <= tol;
              float& w_of = worst_of[f16 ? 0 : 1];
              w_of = std::max(w_of, worst);
            }
            if (!ok) {
              ++failed;
              std::printf("[image_resample] %s C=%d %dx%d %-7s %-8s: "
                          "worst %.3g (tol %.1g) range [%.3f, %.3f] "
                          "outputs %zu/%zu\n", f16 ? "f16" : "f32", C,
                          sz.w, sz.h, fit, alg, worst, tol, lo, hi,
                          g.size(), c.size());
            }
          }
        }
      }
    }
  }
  std::printf("[image_resample] float GPU vs CPU: %d cases, %d differ; "
              "worst |gpu - cpu| f16 %.3g, f32 %.3g\n", cases, failed,
              worst_of[0], worst_of[1]);
  EXPECT_TRUE(cases == 2 * 2 * 2 * 4 * 3);
  EXPECT_TRUE(failed == 0);
}

// Nothing is clamped on the way, on either path: the 1.5 block survives
// a bilinear upscale and Lanczos rings below 0 at the hard edge.
TEST(image_resample_stage, float_samples_are_not_clamped) {
  Session sess;
  const TensorBeat in = picture_(TensorBeat::DType::F32, 3, 23, 37);
  for (bool gpu : { true, false }) {
    auto b = run_beat(sess, in, mode_cfg_(80, 50, "stretch", "bilinear",
                                          gpu));
    auto l = run_beat(sess, in, mode_cfg_(16, 10, "stretch", "lanczos",
                                          gpu));
    ASSERT_TRUE(b.size() == 1 && l.size() == 1);
    if (b.size() != 1 || l.size() != 1) { return; }
    const auto bv = floats_(b[0]), lv = floats_(l[0]);
    const float bmax = *std::max_element(bv.begin(), bv.end());
    const float lmin = *std::min_element(lv.begin(), lv.end());
    std::printf("[image_resample] %s: bilinear max %.3f, lanczos min %.3f\n",
                gpu ? "gpu" : "cpu", bmax, lmin);
    EXPECT_TRUE(bmax > 1.4f);
    EXPECT_TRUE(lmin < 0.0f);
  }
}

// ---- pad_to_grid / crop_to_grid: FLUX.3's reference fits ----------------

namespace {

// One planar [3,H,W] u8 frame that says where each sample came from:
// plane 0 holds the row, plane 1 the column, plane 2 a constant -- so a
// padded or cropped output can be checked pixel by pixel against the
// source position numpy's np.pad would have taken it from.
class WhereSource : public TypedStage<WhereSource> {
public:
  static constexpr const char* kTypeName = "ut-resample-src-where";
  using TypedStage::TypedStage;
  int H = 0, W = 0;
  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = { 3, H, W };
    tb.resize_contiguous(static_cast<size_t>(3) * H * W);
    uint8_t* d = tb.as_u8();
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t i = static_cast<size_t>(y) * W + x;
        d[i] = static_cast<uint8_t>(y);
        d[static_cast<size_t>(H) * W + i] = static_cast<uint8_t>(x);
        d[static_cast<size_t>(2) * H * W + i] = 7;
      }
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }
private:
  bool _sent = false;
};

vector<TensorBeat> run_where(Session& sess, int H, int W, FlexData cfg)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto s = std::make_unique<WhereSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  s->H = H; s->W = W;
  s->allocate_oports(1);
  auto* src = static_cast<WhereSource*>(pl->insert_stage(std::move(s)));
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

FlexData grid_cfg(const char* fit, int max_pixels = 0)
{
  FlexData c = FlexData::make_object();
  c.as_object().insert("fit", FlexData::make_string(fit));
  if (max_pixels > 0) {
    c.as_object().insert("max_pixels", FlexData::make_int(max_pixels));
  }
  return c;
}

std::vector<int64_t> crop_box_of(const TensorBeat& t)
{
  std::vector<int64_t> v;
  if (!t.sideband.is_object()) { return v; }
  FlexData sb = t.sideband;                  // as_object() is a view
  auto o = sb.as_object();
  if (!o.contains(sideband::kCropBox)) { return v; }
  FlexData box = o.at(sideband::kCropBox);
  for (const FlexData& e : box.as_array()) { v.push_back(e.as_int(-1)); }
  return v;
}

}  // namespace

// The geometry, against BFL's own reference_canvas and
// prepare_multi_reference (flux3/pipeline.py) for the same sizes --
// including the 0.995 step-down (4000x3000), the side clamp (5000x100),
// the 64-pixel floor (30x20) and Python's round-half-to-even.
TEST(image_resample_stage, grid_fits_match_bfl)
{
  Session sess;
  ImageResampleStage one(&sess, "one", {}, grid_cfg("pad_to_grid"));
  ImageResampleStage many(&sess, "many", {},
                          grid_cfg("crop_to_grid", 1024 * 1024));
  ASSERT_TRUE(one.fit_mode() == 4 && many.fit_mode() == 5);
  struct Want { int w, h; int fw, fh, cw, ch, x0, y0;
                int mw, mh, mcw, mch, mx, my; };
  const Want cases[] = {
    {1000, 750, 1000, 750, 1008, 752, 4, 1, 1000, 750, 992, 736, 4, 7},
    {4000, 3000, 2341, 1756, 2352, 1760, 5, 2, 1182, 886, 1168, 880, 7, 3},
    {6000, 4000, 2496, 1664, 2496, 1664, 0, 0, 1254, 836, 1248, 832, 3, 2},
    {1023, 767, 1023, 767, 1024, 768, 0, 0, 1023, 767, 1008, 752, 7, 7},
    {30, 20, 30, 20, 64, 64, 17, 22, 30, 20, 16, 16, 7, 2},
    {5000, 100, 4096, 82, 4096, 96, 0, 7, 5000, 100, 4992, 96, 4, 2},
    {2049, 2047, 2039, 2037, 2048, 2048, 4, 5, 1024, 1023, 1024, 1008, 0, 7},
    {3000, 2001, 2495, 1664, 2496, 1664, 0, 0, 1253, 836, 1248, 832, 2, 2},
    {17, 40, 17, 40, 64, 64, 23, 12, 17, 40, 16, 32, 0, 4},
  };
  for (const Want& c : cases) {
    const auto g = one.grid_fit(c.w, c.h);
    const auto m = many.grid_fit(c.w, c.h);
    const bool ok1 = g.fit_w == c.fw && g.fit_h == c.fh && g.out_w == c.cw
                     && g.out_h == c.ch && g.x0 == c.x0 && g.y0 == c.y0;
    const bool ok2 = m.fit_w == c.mw && m.fit_h == c.mh && m.out_w == c.mcw
                     && m.out_h == c.mch && m.x0 == c.mx && m.y0 == c.my;
    if (!ok1 || !ok2) {
      std::printf("[grid] %dx%d: pad %dx%d on %dx%d at (%d,%d); crop %dx%d "
                  "-> %dx%d at (%d,%d)\n", c.w, c.h, g.fit_w, g.fit_h,
                  g.out_w, g.out_h, g.x0, g.y0, m.fit_w, m.fit_h, m.out_w,
                  m.out_h, m.x0, m.y0);
    }
    EXPECT_TRUE(ok1);
    EXPECT_TRUE(ok2);
  }
  // Smaller than a grid cell: nothing to crop to.
  EXPECT_TRUE(many.grid_fit(4096, 1).out_w == 0);
}

// The pad itself, against np.pad: `reflect` when the picture's shorter
// side is longer than the grid (a 30x20 picture on its 64x64 canvas,
// where the 22-row pads are longer than the 20 rows and keep mirroring),
// `symmetric` otherwise (12x40). Positions are numpy's.
TEST(image_resample_stage, pad_to_grid_mirrors_as_numpy)
{
  Session sess;
  struct At { int y, x, sy, sx; };
  struct Case { int H, W; std::vector<int64_t> box; std::vector<At> at; };
  const Case cases[] = {
    {20, 30, {17, 22, 47, 42},
     {{0, 0, 16, 17}, {0, 63, 16, 12}, {63, 0, 3, 17}, {63, 63, 3, 12},
      {21, 16, 1, 1}, {22, 17, 0, 0}, {41, 46, 19, 29}, {42, 47, 18, 28},
      {5, 50, 17, 25}, {60, 10, 0, 7}}},
    {12, 40, {12, 26, 52, 38},
     {{0, 0, 1, 11}, {63, 63, 10, 28}, {25, 11, 0, 0}, {26, 12, 0, 0},
      {37, 51, 11, 39}, {38, 52, 11, 39}}},
  };
  for (const Case& c : cases) {
    auto out = run_where(sess, c.H, c.W, grid_cfg("pad_to_grid"));
    ASSERT_TRUE(out.size() == 1);
    if (out.size() != 1) { continue; }
    const TensorBeat& t = out[0];
    ASSERT_TRUE(t.shape == (std::vector<int64_t>{ 3, 64, 64 }));
    EXPECT_TRUE(crop_box_of(t) == c.box);
    const auto px = t.materialize_contiguous();
    int bad = 0;
    for (const At& a : c.at) {
      const size_t i = static_cast<size_t>(a.y) * 64 + a.x;
      if (px[i] != a.sy || px[4096 + i] != a.sx || px[8192 + i] != 7) {
        std::printf("[pad] %dx%d at (%d,%d): got (%d,%d) want (%d,%d)\n",
                    c.W, c.H, a.y, a.x, px[i], px[4096 + i], a.sy, a.sx);
        ++bad;
      }
    }
    EXPECT_TRUE(bad == 0);
  }
}

// crop_to_grid cuts the centre out unscaled when the picture is under the
// cap, and publishes no box (there is nothing to cut back). A picture that
// already fits pad_to_grid's canvas exactly is passed through, also boxless.
TEST(image_resample_stage, crop_to_grid_cuts_the_centre)
{
  Session sess;
  auto out = run_where(sess, 37, 40, grid_cfg("crop_to_grid", 1 << 20));
  ASSERT_TRUE(out.size() == 1);
  if (out.size() == 1) {
    const TensorBeat& t = out[0];
    ASSERT_TRUE(t.shape == (std::vector<int64_t>{ 3, 32, 32 }));
    const auto px = t.materialize_contiguous();
    // (40 - 32) / 2 = 4 columns and (37 - 32) / 2 = 2 rows in.
    EXPECT_TRUE(px[0] == 2 && px[1024] == 4);
    EXPECT_TRUE(px[31 * 32 + 31] == 33 && px[1024 + 31 * 32 + 31] == 35);
    EXPECT_TRUE(crop_box_of(t).empty());
  }
  auto same = run_where(sess, 64, 80, grid_cfg("pad_to_grid"));
  ASSERT_TRUE(same.size() == 1);
  if (same.size() == 1) {
    EXPECT_TRUE(same[0].shape == (std::vector<int64_t>{ 3, 64, 80 }));
    EXPECT_TRUE(crop_box_of(same[0]).empty());
  }
}

// A picture past the cap is RESAMPLED to the fit first (Lanczos, as BFL's
// PIL resize), then padded: 4000x3000 -> 2341x1756 on 2352x1760.
TEST(image_resample_stage, pad_to_grid_resamples_past_the_cap)
{
  Session sess;
  auto out = run_resample(sess, 3000, 4000, TensorBeat::DType::U8, 90.0f,
                          grid_cfg("pad_to_grid"));
  ASSERT_TRUE(out.size() == 1);
  if (out.size() != 1) { return; }
  ASSERT_TRUE(out[0].shape == (std::vector<int64_t>{ 3, 1760, 2352 }));
  EXPECT_TRUE(crop_box_of(out[0]) ==
              (std::vector<int64_t>{ 5, 2, 2346, 1758 }));
  // A flat picture stays flat through the resample and the mirror.
  const auto px = out[0].materialize_contiguous();
  int off = 0;
  for (size_t i = 0; i < px.size(); i += 997) { off += px[i] != 90; }
  EXPECT_TRUE(off == 0);
}

// crop_to_area: Bria's size for the picture's own aspect about an area
// (common/size-about-area.h, Fibo 2.0's get_image_size) -- at 1 megapixel
// the trained table, at 4 Bria's 4 MP sizes -- and then exactly `crop`
// at that size, up or down. An area no size on the grid comes near is
// refused at construction.
TEST(image_resample_stage, crop_to_area_is_crop_at_bria_s_size)
{
  Session sess;
  auto area = [](double mp) {
    FlexData c = FlexData::make_object();
    c.as_object().insert("fit", FlexData::make_string("crop_to_area"));
    c.as_object().insert("megapixels", FlexData::make_real(mp));
    return c;
  };
  ImageResampleStage one(&sess, "one", {}, area(1.0));
  ImageResampleStage four(&sess, "four", {}, area(4.0));
  ASSERT_TRUE(one.fit_mode() == 6 && one.config_error().empty());
  ASSERT_TRUE(four.config_error().empty());
  struct Want { int w, h, w1, h1, w4, h4; };
  const Want cases[] = {
    {1000, 1000, 1024, 1024, 2048, 2048},
    {4000, 3000, 1152, 864, 2368, 1776},
    {3000, 2000, 1248, 832, 2496, 1664},
    {1920, 1080, 1424, 800, 2816, 1584},
    {1080, 1920, 800, 1424, 1584, 2816},
    {2100, 900, 1568, 672, 3136, 1344},
  };
  int bad = 0;
  for (const Want& c : cases) {
    const auto a = one.area_fit(c.w, c.h);
    const auto b = four.area_fit(c.w, c.h);
    if (a != std::make_pair(c.w1, c.h1) || b != std::make_pair(c.w4, c.h4)) {
      std::printf("[area] %dx%d: 1 MP %dx%d, 4 MP %dx%d\n", c.w, c.h,
                  a.first, a.second, b.first, b.second);
      ++bad;
    }
  }
  EXPECT_TRUE(bad == 0);
  // The pixels: a 300x400 picture to a quarter megapixel is `crop` at
  // the size the rule picks.
  ImageResampleStage quarter(&sess, "q", {}, area(0.25));
  const auto [qw, qh] = quarter.area_fit(400, 300);
  auto got = run_where(sess, 300, 400, area(0.25));
  FlexData crop = FlexData::make_object();
  crop.as_object().insert("fit", FlexData::make_string("crop"));
  crop.as_object().insert("width", FlexData::make_int(qw));
  crop.as_object().insert("height", FlexData::make_int(qh));
  auto want = run_where(sess, 300, 400, std::move(crop));
  ASSERT_TRUE(got.size() == 1 && want.size() == 1);
  if (got.size() == 1 && want.size() == 1) {
    EXPECT_TRUE(got[0].shape == (std::vector<int64_t>{ 3, qh, qw }));
    EXPECT_TRUE(got[0].materialize_contiguous() ==
                want[0].materialize_contiguous());
    std::printf("[area] 400x300 at 0.25 MP -> %dx%d, %s fit:crop\n", qw,
                qh,
                got[0].materialize_contiguous() ==
                        want[0].materialize_contiguous()
                    ? "identical to" : "DIFFERENT from");
  }
  ImageResampleStage none(&sess, "none", {}, area(0.0));
  EXPECT_FALSE(none.config_error().empty());
}
