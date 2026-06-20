#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/vision/detection-overlay-stage.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

// Emits one preconfigured TensorBeat then signals done.
class OneTensorSource : public TypedStage<OneTensorSource> {
public:
  static constexpr const char* kTypeName = "ut-one-tensor-source";
  using TypedStage::TypedStage;

  TensorBeat tb;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) {
      ctx.signal_done();
      co_return;
    }
    _sent = true;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  bool _sent = false;
};

// Emits one preconfigured FlexData then signals done.
class OneFlexDataSource : public TypedStage<OneFlexDataSource> {
public:
  static constexpr const char* kTypeName = "ut-one-flexdata-source";
  using TypedStage::TypedStage;

  FlexData fd = FlexData::make_object();

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) {
      ctx.signal_done();
      co_return;
    }
    _sent = true;
    co_await ctx.write(0, make_payload<FlexDataPayload>(fd));
  }

private:
  bool _sent = false;
};

// Closes its oport immediately (never writes a beat). Used to test
// that the overlay stage signals done when an iport hits EOS.
class ClosedSource : public TypedStage<ClosedSource> {
public:
  static constexpr const char* kTypeName = "ut-closed-source";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

class TensorBeatSink : public TypedStage<TensorBeatSink> {
public:
  static constexpr const char* kTypeName = "ut-overlay-tensor-sink";
  using TypedStage::TypedStage;

  vector<TensorBeat>& collected() { return _collected; }

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    const TensorBeatPayload* p =
        dynamic_cast<const TensorBeatPayload*>(in.get());
    if (p) {
      lock_guard<mutex> g(_mu);
      _collected.push_back(static_cast<const TensorBeat&>(*p));
    }
  }

private:
  mutex              _mu;
  vector<TensorBeat> _collected;
};

// Construct a contiguous TensorBeat of shape [3, H, W] filled with v.
TensorBeat
make_tensor_(int H, int W, float v = 0.0f)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = { 3, H, W };
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  float* p = tb.as_f32();
  for (size_t i = 0; i < static_cast<size_t>(3) * H * W; ++i) {
    p[i] = v;
  }
  return tb;
}

TensorBeat
make_u8_tensor_(int H, int W, uint8_t v = 0)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = { 3, H, W };
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  std::memset(tb.as_u8(), v, static_cast<size_t>(3) * H * W);
  return tb;
}

// Build a FlexData detection-set blob matching yolo-detection-stage's
// schema. Each detection is provided as (cls, x1, y1, x2, y2, score,
// optional class_name).
struct DetSpec {
  int    cls;
  double x1, y1, x2, y2;
  double score;
  const char* class_name = nullptr;
};

FlexData
make_detections_(int frame_w, int frame_h,
                 const vector<DetSpec>& dets)
{
  FlexData out = FlexData::make_object();
  auto root = out.as_object();
  root.insert("frame_width",  FlexData::make_int(frame_w));
  root.insert("frame_height", FlexData::make_int(frame_h));
  FlexData arr = FlexData::make_array();
  auto v = arr.as_array();
  for (const auto& d : dets) {
    FlexData entry = FlexData::make_object();
    auto e = entry.as_object();
    e.insert("class_id", FlexData::make_int(d.cls));
    if (d.class_name) {
      e.insert("class_name", FlexData::make_string(d.class_name));
    }
    e.insert("score", FlexData::make_real(d.score));
    e.insert("x1", FlexData::make_real(d.x1));
    e.insert("y1", FlexData::make_real(d.y1));
    e.insert("x2", FlexData::make_real(d.x2));
    e.insert("y2", FlexData::make_real(d.y2));
    v.push_back(std::move(entry));
  }
  root.insert("detections", std::move(arr));
  return out;
}

// Pixel accessor for a contiguous [3, H, W] TensorBeat.
float
at_(const TensorBeat& tb, int c, int y, int x)
{
  const int H = static_cast<int>(tb.shape[1]);
  const int W = static_cast<int>(tb.shape[2]);
  const size_t off = static_cast<size_t>(c) * H * W
                   + static_cast<size_t>(y) * W + x;
  if (tb.dtype == TensorBeat::DType::U8) {
    return static_cast<float>(tb.as_u8()[off]);
  }
  return tb.as_f32()[off];
}

// Returns true if pixel (y, x) has any non-zero plane value.
bool
is_painted_(const TensorBeat& tb, int y, int x)
{
  return at_(tb, 0, y, x) != 0.0f
      || at_(tb, 1, y, x) != 0.0f
      || at_(tb, 2, y, x) != 0.0f;
}

struct OverlayHarness {
  Session                            sess;
  unique_ptr<Pipeline>               pl;
  OneTensorSource*                   tsrc   = nullptr;
  OneFlexDataSource*                 fsrc   = nullptr;
  DetectionOverlayStage*             ovr    = nullptr;
  TensorBeatSink*                    sink   = nullptr;
};

unique_ptr<OverlayHarness>
build_(TensorBeat tb, FlexData fd, FlexData cfg)
{
  auto h = make_unique<OverlayHarness>();
  h->pl = make_unique<Pipeline>("p", &h->sess);

  auto tsrc_u = make_unique<OneTensorSource>(
      &h->sess, "tsrc", vector<InEdge>{}, FlexData::make_object());
  tsrc_u->tb = std::move(tb);
  tsrc_u->allocate_oports(1);
  h->tsrc = static_cast<OneTensorSource*>(
      h->pl->insert_stage(std::move(tsrc_u)));

  auto fsrc_u = make_unique<OneFlexDataSource>(
      &h->sess, "fsrc", vector<InEdge>{}, FlexData::make_object());
  fsrc_u->fd = std::move(fd);
  fsrc_u->allocate_oports(1);
  h->fsrc = static_cast<OneFlexDataSource*>(
      h->pl->insert_stage(std::move(fsrc_u)));

  auto ovr_u = make_unique<DetectionOverlayStage>(
      &h->sess, "ovr",
      vector<InEdge>{{h->tsrc, 0}, {h->fsrc, 0}},
      std::move(cfg));
  h->ovr = static_cast<DetectionOverlayStage*>(
      h->pl->insert_stage(std::move(ovr_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &h->sess, "sink", vector<InEdge>{{h->ovr, 0}},
      FlexData::make_object());
  h->sink = static_cast<TensorBeatSink*>(
      h->pl->insert_stage(std::move(sink_u)));

  return h;
}

bool
run_(OverlayHarness& h)
{
  PipelineRuntime rt(h.pl.get(), &h.sess);
  if (!rt.launch()) {
    return false;
  }
  rt.wait_idle();
  rt.stop();
  return true;
}

}  // namespace

TEST(detection_overlay_stage, shape_preserved_and_contiguous)
{
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 80, 0.0f);
  FlexData   fd  = make_detections_(80, 64, {
    { /*cls*/ 0, /*x1*/ 10, /*y1*/ 12, /*x2*/ 30, /*y2*/ 32,
      /*score*/ 0.9 }
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label", FlexData::make_bool(false));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  EXPECT_TRUE(h->sink->collected().size() == 1u);
  const TensorBeat& out = h->sink->collected()[0];
  EXPECT_TRUE(out.shape.size() == 3u);
  EXPECT_TRUE(out.shape[0] == 3);
  EXPECT_TRUE(out.shape[1] == 64);
  EXPECT_TRUE(out.shape[2] == 80);
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_TRUE(out.strides.empty());
  EXPECT_TRUE(out.data.size()
              == static_cast<size_t>(3) * 64 * 80
                                       * out.element_byte_size());
}

TEST(detection_overlay_stage, class_names_config_supplies_label)
{
  // class_names config takes effect when the per-detection FlexData
  // entry does NOT carry a `class_name` field. We need a way to
  // observe which label was actually rendered; since the stage paints
  // glyphs directly into the tensor with a deterministic font, we
  // can't easily read the text back. Instead, exercise the
  // precedence rule and rely on the no-crash + bbox-painted side
  // effect to confirm the path was taken. (A label-render unit test
  // would require font-aware OCR — out of scope.)
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 80, 0.0f);
  // Detection without `class_name`; expect "person" to be picked
  // from class_names[0] rather than the "cls0" fallback.
  FlexData fd = make_detections_(80, 64, {
    { /*cls*/ 0, 10, 10, 40, 30, /*score*/ 0.9 }
  });
  FlexData cfg = FlexData::make_object();
  FlexData names_arr = FlexData::make_array();
  auto names_v = names_arr.as_array();
  names_v.push_back(FlexData::make_string("person"));
  names_v.push_back(FlexData::make_string("bicycle"));
  cfg.as_object().insert("class_names", std::move(names_arr));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  EXPECT_TRUE(h->sink->collected().size() == 1u);
  const TensorBeat& out = h->sink->collected()[0];
  // Bbox should still be painted -- regression guard that supplying
  // class_names didn't break the drawing path.
  EXPECT_TRUE(is_painted_(out, 10, 25));
}

TEST(detection_overlay_stage, per_detection_class_name_overrides_config)
{
  // When the FlexData entry carries a `class_name`, it wins over the
  // stage's class_names config. We exercise both paths in one frame:
  // detection 0 has class_name set; detection 1 relies on config.
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 80, 0.0f);
  FlexData fd = make_detections_(80, 64, {
    { /*cls*/ 0, 10, 10, 30, 30, /*score*/ 0.9, /*class_name*/ "cat" },
    { /*cls*/ 1, 40, 10, 60, 30, /*score*/ 0.9 }
  });
  FlexData cfg = FlexData::make_object();
  FlexData names_arr = FlexData::make_array();
  auto names_v = names_arr.as_array();
  names_v.push_back(FlexData::make_string("ignored0"));
  names_v.push_back(FlexData::make_string("bicycle"));
  cfg.as_object().insert("class_names", std::move(names_arr));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];
  // Both bboxes painted; no crash on the two distinct lookup paths.
  EXPECT_TRUE(is_painted_(out, 10, 20));
  EXPECT_TRUE(is_painted_(out, 10, 50));
}

TEST(detection_overlay_stage, out_of_range_class_id_falls_back_to_cls_n)
{
  // class_id is beyond the class_names array -- stage must not
  // dereference past the end, and must not crash. Use a short
  // names array (1 entry) and a detection with class_id=7.
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 80, 0.0f);
  FlexData fd = make_detections_(80, 64, {
    { /*cls*/ 7, 10, 10, 40, 30, /*score*/ 0.9 }
  });
  FlexData cfg = FlexData::make_object();
  FlexData names_arr = FlexData::make_array();
  names_arr.as_array().push_back(FlexData::make_string("only-zero"));
  cfg.as_object().insert("class_names", std::move(names_arr));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];
  EXPECT_TRUE(is_painted_(out, 10, 25));
}

TEST(detection_overlay_stage, u8_passthrough_dtype_and_paints_bbox)
{
  // U8 TensorBeat input must produce a U8 TensorBeat output, with
  // the bbox painted in the per-class colour (mapped to [0, 255]).
  CerrSilencer hush;

  TensorBeat tin = make_u8_tensor_(64, 80, 0);
  FlexData fd = make_detections_(80, 64, {
    { /*cls*/ 5, 10, 10, 40, 30, /*score*/ 0.95 }
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label", FlexData::make_bool(false));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  EXPECT_TRUE(h->sink->collected().size() == 1u);
  const TensorBeat& out = h->sink->collected()[0];
  EXPECT_TRUE(out.dtype == TensorBeat::DType::U8);
  EXPECT_TRUE(out.element_byte_size() == 1u);
  // A pixel inside the bbox border (top-left corner) should now be
  // painted -- non-zero on at least one plane.
  EXPECT_TRUE(is_painted_(out, 10, 10));
  // A pixel far from the bbox should still be zero.
  EXPECT_TRUE(!is_painted_(out, 50, 60));
}

TEST(detection_overlay_stage, bbox_pixels_match_class_color)
{
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 80, 0.0f);
  // Bbox (10, 10) to (40, 30). Thickness 2 will paint rows 10, 11
  // and 29, 30, plus cols 10, 11 and 39, 40.
  FlexData fd = make_detections_(80, 64, {
    { /*cls*/ 5, 10, 10, 40, 30, /*score*/ 0.95 }
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label",   FlexData::make_bool(false));
  cfg.as_object().insert("box_thickness", FlexData::make_int(2));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  EXPECT_TRUE(h->sink->collected().size() == 1u);
  const TensorBeat& out = h->sink->collected()[0];

  // Top-edge pixel inside the bar (row 10, mid-column) must be
  // non-zero.
  EXPECT_TRUE(is_painted_(out, 10, 25));
  // Left-edge pixel (mid-row, column 10) must be non-zero.
  EXPECT_TRUE(is_painted_(out, 20, 10));
  // Bottom-edge pixel (row 30, mid-column).
  EXPECT_TRUE(is_painted_(out, 30, 25));
  // Right-edge pixel.
  EXPECT_TRUE(is_painted_(out, 20, 40));

  // Interior pixel (well inside the bbox, off the border bars) must
  // still be untouched (zero).
  EXPECT_TRUE(!is_painted_(out, 20, 25));

  // Pixel outside the bbox stays zero.
  EXPECT_TRUE(!is_painted_(out, 50, 50));
}

TEST(detection_overlay_stage, top_n_keeps_highest_scores)
{
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 96, 0.0f);
  // Three disjoint bboxes, distinct scores.
  FlexData fd = make_detections_(96, 64, {
    { 0,  5,  5, 25, 25, /*score*/ 0.9 },
    { 1, 30,  5, 50, 25, /*score*/ 0.5 },
    { 2, 60,  5, 80, 25, /*score*/ 0.1 },
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label", FlexData::make_bool(false));
  cfg.as_object().insert("top_n",      FlexData::make_int(1));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];

  // First bbox (highest score 0.9) should be painted; the other
  // two should be entirely untouched. Check a corner pixel of each.
  EXPECT_TRUE(is_painted_(out,  5,  5));   // first bbox top-left
  EXPECT_TRUE(!is_painted_(out, 5, 30));   // second bbox top-left
  EXPECT_TRUE(!is_painted_(out, 5, 60));   // third bbox top-left
}

TEST(detection_overlay_stage, score_threshold_drops_low)
{
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(64, 96, 0.0f);
  FlexData fd = make_detections_(96, 64, {
    { 0,  5,  5, 25, 25, 0.9 },
    { 1, 30,  5, 50, 25, 0.5 },
    { 2, 60,  5, 80, 25, 0.1 },
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label",      FlexData::make_bool(false));
  cfg.as_object().insert("score_threshold", FlexData::make_real(0.6));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];

  EXPECT_TRUE(is_painted_(out, 5,  5));
  EXPECT_TRUE(!is_painted_(out, 5, 30));
  EXPECT_TRUE(!is_painted_(out, 5, 60));
}

TEST(detection_overlay_stage, frame_size_mismatch_rescales)
{
  CerrSilencer hush;

  // Tensor is 200 wide x 100 tall; FlexData reports 50 x 100, so
  // x-coords should be scaled by 4 (200/50) and y by 1 (100/100).
  TensorBeat tin = make_tensor_(100, 200, 0.0f);
  FlexData fd = make_detections_(50, 100, {
    { 0, /*x1*/ 10, /*y1*/ 10, /*x2*/ 20, /*y2*/ 20, /*score*/ 0.9 }
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label",    FlexData::make_bool(false));
  cfg.as_object().insert("box_thickness", FlexData::make_int(2));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];

  // Scaled bbox should be (40, 10) to (80, 20).
  EXPECT_TRUE(is_painted_(out, 10, 60));   // top edge, mid-x
  EXPECT_TRUE(is_painted_(out, 15, 40));   // left edge, mid-y
  EXPECT_TRUE(is_painted_(out, 20, 60));   // bottom edge
  EXPECT_TRUE(is_painted_(out, 15, 80));   // right edge

  // Original-frame coords would have painted around (10..20, 10..20)
  // — confirm that region was NOT painted (the rescale fired).
  EXPECT_TRUE(!is_painted_(out, 15, 15));
}

TEST(detection_overlay_stage, strided_input_materialized)
{
  CerrSilencer hush;

  // Build a pitch-padded TensorBeat: shape [3, 8, 5], pitch P=8.
  // Fill the logical [..., :5] columns with a recognisable value.
  const int H = 8, W = 5, P = 8;
  TensorBeat tin;
  tin.dtype   = TensorBeat::DType::F32;
  tin.shape   = { 3, H, W };
  tin.strides = { static_cast<int64_t>(H) * P, P, 1 };
  tin.data.assign(static_cast<size_t>(3) * H * P * sizeof(float), 0);
  float* in_f = tin.as_f32();
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        in_f[static_cast<size_t>(c) * H * P + y * P + x] =
            0.1f * (c + 1) + 0.01f * y + 0.001f * x;
      }
    }
  }

  // Empty detections so the stage just materializes + emits.
  FlexData fd = make_detections_(W, H, {});
  FlexData cfg = FlexData::make_object();

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];

  EXPECT_TRUE(out.is_contiguous());
  EXPECT_TRUE(out.strides.empty());
  EXPECT_TRUE(out.data.size()
              == static_cast<size_t>(3) * H * W * sizeof(float));

  // Each output pixel should match the strided source.
  const float* out_f = out.as_f32();
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float want = 0.1f * (c + 1) + 0.01f * y + 0.001f * x;
        float got  = out_f[static_cast<size_t>(c) * H * W
                           + y * W + x];
        EXPECT_TRUE(fabs(got - want) < 1e-6f);
      }
    }
  }
}

TEST(detection_overlay_stage, draw_timestamp_paints_bottom_right_when_sideband_set)
{
  // draw_timestamp=true + sideband.timestamp_us present ⇒ pixels in
  // the bottom-right corner are mutated (black bg + white glyphs),
  // and pixels in the top-left corner remain at the input fill (no
  // detections drawn). draw_timestamp=false ⇒ both regions untouched.
  CerrSilencer hush;

  // Use a uniform mid-grey input so any pixel that differs from
  // 0.5f after the run must come from the timestamp overlay.
  auto build_with_ts = [](bool draw_ts) {
    TensorBeat tin = make_tensor_(64, 256, 0.5f);
    FlexData sb = FlexData::make_object();
    // 2026-05-21 12:34:56.789 UTC.
    const uint64_t kTs = 1747830896789000ull;
    sb.as_object().insert_or_assign("timestamp_us",
                                    FlexData::make_uint(kTs));
    tin.sideband = std::move(sb);

    FlexData fd  = make_detections_(256, 64, {});
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("draw_timestamp",
                           FlexData::make_bool(draw_ts));
    return build_(std::move(tin), std::move(fd), std::move(cfg));
  };

  // With draw_timestamp = true: bottom-right has overlay pixels;
  // sideband is propagated to the output.
  {
    auto h = build_with_ts(true);
    EXPECT_TRUE(run_(*h));
    const TensorBeat& out = h->sink->collected()[0];
    EXPECT_TRUE(out.sideband.is_object());
    EXPECT_TRUE(out.sideband.as_object()
                  .at("timestamp_us").get_uint()
                == 1747830896789000ull);

    // Sample the band a few pixels in from the bottom-right edge —
    // the inset is 4 and label_padding (default 2) wraps the glyphs.
    // At (H-8, W-8) we should be inside the label background fill,
    // which is black ⇒ value < 0.5f. At (H-50, 5) (top-left) the
    // input should be untouched.
    EXPECT_TRUE(at_(out, 0, 64 - 8, 256 - 8) < 0.1f);
    EXPECT_TRUE(at_(out, 1, 64 - 8, 256 - 8) < 0.1f);
    EXPECT_TRUE(at_(out, 2, 64 - 8, 256 - 8) < 0.1f);
    EXPECT_TRUE(fabs(at_(out, 0, 5, 5) - 0.5f) < 1e-6f);

    // Somewhere inside the timestamp label region a white glyph
    // pixel must exist (close to 1.0f on all channels). Scan the
    // bottom-right band.
    bool found_white = false;
    for (int y = 64 - 30; y < 64 - 4 && !found_white; ++y) {
      for (int x = 256 - 200; x < 256 - 4; ++x) {
        if (at_(out, 0, y, x) > 0.9f
            && at_(out, 1, y, x) > 0.9f
            && at_(out, 2, y, x) > 0.9f) {
          found_white = true;
          break;
        }
      }
    }
    EXPECT_TRUE(found_white);
  }

  // With draw_timestamp = false: nothing in the bottom-right is
  // mutated.
  {
    auto h = build_with_ts(false);
    EXPECT_TRUE(run_(*h));
    const TensorBeat& out = h->sink->collected()[0];
    EXPECT_TRUE(fabs(at_(out, 0, 64 - 8, 256 - 8) - 0.5f) < 1e-6f);
    EXPECT_TRUE(fabs(at_(out, 1, 64 - 8, 256 - 8) - 0.5f) < 1e-6f);
    EXPECT_TRUE(fabs(at_(out, 2, 64 - 8, 256 - 8) - 0.5f) < 1e-6f);
  }
}

TEST(detection_overlay_stage, draw_timestamp_no_op_without_sideband)
{
  CerrSilencer hush;
  // Input has no sideband; draw_timestamp on must be a quiet no-op.
  TensorBeat tin = make_tensor_(32, 64, 0.5f);
  EXPECT_TRUE(tin.sideband.is_null());
  FlexData fd  = make_detections_(64, 32, {});
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_timestamp", FlexData::make_bool(true));
  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];
  const float* p = out.as_f32();
  const size_t n = static_cast<size_t>(3) * 32 * 64;
  for (size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(fabs(p[i] - 0.5f) < 1e-6f);
  }
}

TEST(detection_overlay_stage, empty_detections_passthrough)
{
  CerrSilencer hush;

  TensorBeat tin = make_tensor_(32, 32, 0.25f);
  FlexData fd = make_detections_(32, 32, {});
  FlexData cfg = FlexData::make_object();

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));
  const TensorBeat& out = h->sink->collected()[0];

  EXPECT_TRUE(out.data.size()
              == static_cast<size_t>(3) * 32 * 32 * sizeof(float));
  const float* p = out.as_f32();
  const size_t n = static_cast<size_t>(3) * 32 * 32;
  for (size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(fabs(p[i] - 0.25f) < 1e-6f);
  }
}

// Build a FlexData detection blob that carries a per-detection
// `track_id` field. Mirrors make_detections_ but with one more knob.
// Used only by the tracker-aware overlay tests below.
struct TrackedDetSpec {
  int    cls;
  double x1, y1, x2, y2;
  double score;
  int    track_id;     // -1 for "no tracker upstream"
};

namespace {

FlexData
make_tracked_detections_(int frame_w, int frame_h,
                         const vector<TrackedDetSpec>& dets)
{
  FlexData out = FlexData::make_object();
  auto root = out.as_object();
  root.insert("frame_width",  FlexData::make_int(frame_w));
  root.insert("frame_height", FlexData::make_int(frame_h));
  FlexData arr = FlexData::make_array();
  auto v = arr.as_array();
  for (const auto& d : dets) {
    FlexData entry = FlexData::make_object();
    auto e = entry.as_object();
    e.insert("class_id", FlexData::make_int(d.cls));
    e.insert("score",    FlexData::make_real(d.score));
    e.insert("x1",       FlexData::make_real(d.x1));
    e.insert("y1",       FlexData::make_real(d.y1));
    e.insert("x2",       FlexData::make_real(d.x2));
    e.insert("y2",       FlexData::make_real(d.y2));
    if (d.track_id >= 0) {
      e.insert("track_id", FlexData::make_int(d.track_id));
    }
    v.push_back(std::move(entry));
  }
  root.insert("detections", std::move(arr));
  return out;
}

// Read the RGB triple at pixel (y, x) of a contiguous [3, H, W]
// TensorBeat. Float values are returned as-is; uint8 values are
// promoted to float.
array<float, 3>
rgb_at_(const TensorBeat& tb, int y, int x)
{
  return { at_(tb, 0, y, x), at_(tb, 1, y, x), at_(tb, 2, y, x) };
}

}  // namespace

TEST(detection_overlay_stage, track_id_keeps_color_across_classes)
{
  // Two detections in the same frame share track_id=5 but have
  // different class_ids. They must render in the same colour
  // (because colour is keyed on track_id when present), even
  // though class-coloured rendering would give them distinct hues.
  CerrSilencer hush;

  TensorBeat tin = make_u8_tensor_(64, 128, 0);
  FlexData fd = make_tracked_detections_(128, 64, {
    { /*cls*/ 0,  10, 10, 40, 30, 0.95, /*track_id*/ 5 },
    { /*cls*/ 7, 70, 10, 100, 30, 0.95, /*track_id*/ 5 },
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label",    FlexData::make_bool(false));
  cfg.as_object().insert("box_thickness", FlexData::make_int(2));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  const TensorBeat& out = h->sink->collected()[0];
  // Pixel inside each bbox's top edge bar (row 10, mid-column).
  auto a = rgb_at_(out, 10, 25);    // bbox A: x in [10, 40]
  auto b = rgb_at_(out, 10, 85);    // bbox B: x in [70, 100]
  EXPECT_TRUE(a[0] != 0.0f || a[1] != 0.0f || a[2] != 0.0f);
  EXPECT_TRUE(a == b);
}

TEST(detection_overlay_stage, distinct_track_ids_get_distinct_colors)
{
  // Same class_id, different track_ids -- bboxes must render in
  // different colours. This is the case where class-only colouring
  // would have produced identical hues and the user couldn't tell
  // the objects apart.
  CerrSilencer hush;

  TensorBeat tin = make_u8_tensor_(64, 128, 0);
  FlexData fd = make_tracked_detections_(128, 64, {
    { /*cls*/ 0,  10, 10,  40, 30, 0.95, /*track_id*/ 3 },
    { /*cls*/ 0, 70, 10, 100, 30, 0.95, /*track_id*/ 8 },
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label",    FlexData::make_bool(false));
  cfg.as_object().insert("box_thickness", FlexData::make_int(2));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  const TensorBeat& out = h->sink->collected()[0];
  auto a = rgb_at_(out, 10, 25);
  auto b = rgb_at_(out, 10, 85);
  EXPECT_TRUE(a[0] != 0.0f || a[1] != 0.0f || a[2] != 0.0f);
  EXPECT_TRUE(b[0] != 0.0f || b[1] != 0.0f || b[2] != 0.0f);
  EXPECT_TRUE(a != b);
}

TEST(detection_overlay_stage, no_track_id_falls_back_to_class_color)
{
  // When track_id is absent, the overlay must still colour by
  // class_id (the legacy behaviour). Two detections without
  // track_id but the same class must paint the same colour;
  // different classes must paint different colours.
  CerrSilencer hush;

  TensorBeat tin = make_u8_tensor_(64, 128, 0);
  FlexData fd = make_detections_(128, 64, {
    { /*cls*/ 4,  10, 10,  40, 30, 0.95 },
    { /*cls*/ 4, 70, 10, 100, 30, 0.95 },
    { /*cls*/ 9, 10, 40,  40, 60, 0.95 },
  });
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("draw_label",    FlexData::make_bool(false));
  cfg.as_object().insert("box_thickness", FlexData::make_int(2));

  auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
  EXPECT_TRUE(run_(*h));

  const TensorBeat& out = h->sink->collected()[0];
  auto same_cls_a = rgb_at_(out, 10, 25);
  auto same_cls_b = rgb_at_(out, 10, 85);
  auto diff_cls   = rgb_at_(out, 40, 25);
  EXPECT_TRUE(same_cls_a == same_cls_b);
  EXPECT_TRUE(same_cls_a != diff_cls);
}

TEST(detection_overlay_stage, closes_on_iport0_eos)
{
  CerrSilencer hush;

  // Replace the tensor source with a closed source: the overlay
  // stage should signal done without emitting any output beat.
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto t_closed_u = make_unique<ClosedSource>(
      &sess, "tsrc", vector<InEdge>{}, FlexData::make_object());
  t_closed_u->allocate_oports(1);
  auto* t_closed = static_cast<ClosedSource*>(
      pl->insert_stage(std::move(t_closed_u)));

  auto fsrc_u = make_unique<OneFlexDataSource>(
      &sess, "fsrc", vector<InEdge>{}, FlexData::make_object());
  fsrc_u->fd = make_detections_(32, 32, {});
  fsrc_u->allocate_oports(1);
  auto* fsrc = static_cast<OneFlexDataSource*>(
      pl->insert_stage(std::move(fsrc_u)));

  auto ovr_u = make_unique<DetectionOverlayStage>(
      &sess, "ovr",
      vector<InEdge>{{t_closed, 0}, {fsrc, 0}},
      FlexData::make_object());
  auto* ovr = static_cast<DetectionOverlayStage*>(
      pl->insert_stage(std::move(ovr_u)));

  auto sink_u = make_unique<TensorBeatSink>(
      &sess, "sink", vector<InEdge>{{ovr, 0}}, FlexData::make_object());
  auto* sink = static_cast<TensorBeatSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(sink->collected().empty());
}

TEST(detection_overlay_stage, bench_fill_rect_throughput)
{
  // Gated on env so the unit suite stays fast. Builds a 1080p u8
  // TensorBeat, places 10 medium-sized bboxes + labels, runs the
  // stage N times, prints draw throughput at INFO level. Useful as
  // a "did we actually go faster" sanity check after touching the
  // drawing path; not a correctness gate.
  if (!std::getenv("VPIPE_BENCH")) {
    return;
  }
  CerrSilencer hush;

  const int H = 1080, W = 1920;
  const int N = 100;

  // Build a representative detection set: 10 bboxes scattered across
  // the frame, each ~300×200 px with a label.
  std::vector<DetSpec> dets;
  dets.reserve(10);
  for (int i = 0; i < 10; ++i) {
    DetSpec d{};
    d.cls   = i;
    d.x1    = 50.0  + (i % 5) * 360.0;
    d.y1    = 50.0  + (i / 5) * 500.0;
    d.x2    = d.x1 + 300.0;
    d.y2    = d.y1 + 200.0;
    d.score = 0.95 - 0.01 * i;
    dets.push_back(d);
  }
  FlexData fd_template = make_detections_(W, H, dets);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) {
    TensorBeat tin = make_u8_tensor_(H, W, 0);
    FlexData fd   = fd_template;
    FlexData cfg  = FlexData::make_object();
    auto h = build_(std::move(tin), std::move(fd), std::move(cfg));
    EXPECT_TRUE(run_(*h));
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[bench] detection-overlay 1080p u8 x 10 dets: "
            << ms << " ms total, " << (ms / N) << " ms/frame, "
            << (1000.0 * N / ms) << " fps\n";
}
