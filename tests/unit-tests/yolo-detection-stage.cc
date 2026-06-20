// YoloDetectionStage tests.
//
// End-to-end paths require a real YOLO mlpackage on disk. Supply one
// via VPIPE_TEST_YOLOV6_MODEL=/path/to/yolov6X.mlpackage.
//
// The stage reads its input size and (optionally) num_classes from
// the model description directly — no test env var sets either. The
// model must use a TensorType input (MLMultiArray) of shape
// [1, 3, S, S] in float32 [0, 1]; ImageType-input models trip the
// stage's clear error and are not exercised here.

#include "minitest.h"

#include "stages/vision/yolo-detection-stage.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/typed-stage.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

const char*
env_or_null_(const char* name)
{
  const char* v = std::getenv(name);
  return (v && *v) ? v : nullptr;
}

// Source: emits one TensorBeat of solid grey then signals done.
class OneTensorSource : public TypedStage<OneTensorSource> {
public:
  static constexpr const char* kTypeName = "ut-one-tensor-source";
  using TypedStage::TypedStage;

  int W = 640;
  int H = 480;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) {
      ctx.signal_done();
      co_return;
    }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {3, H, W};
    tb.resize_contiguous(static_cast<size_t>(3) * H * W);
    float* p = tb.as_f32();
    for (size_t i = 0; i < static_cast<size_t>(3) * H * W; ++i) {
      p[i] = 0.5f;
    }
    co_await ctx.write(0,
        make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  bool _sent = false;
};

// Sink: collects every FlexData beat it sees and exits on EOS.
class FlexDataSink : public TypedStage<FlexDataSink> {
public:
  static constexpr const char* kTypeName = "ut-flex-data-sink";
  using TypedStage::TypedStage;

  vector<FlexData>& collected() { return _collected; }

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    if (const FlexDataPayload* fdp =
            dynamic_cast<const FlexDataPayload*>(in.get())) {
      std::lock_guard<std::mutex> g(_mu);
      _collected.push_back(fdp->data);
    }
  }

private:
  std::mutex       _mu;
  vector<FlexData> _collected;
};

}

TEST(yolo_detection_stage, type_is_registered)
{
  auto& reg = StageRegistry::get();
  EXPECT_TRUE(reg.find_id("yolo-detection") != StageTypeId::unknown);
}

// Construction succeeds with any config; a missing/invalid field is
// recorded in config_error() and deferred to launch.
TEST(yolo_detection_stage, missing_model_path_deferred)
{
  Session sess;
  YoloDetectionStage s(&sess, "y", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(yolo_detection_stage, class_names_size_mismatch_deferred)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("model_path",
                         FlexData::make_string("/tmp/x.mlpackage"));
  cfg.as_object().insert("num_classes", FlexData::make_int(3));
  FlexData names = FlexData::make_array();
  names.as_array().push_back(std::string_view("a"));
  names.as_array().push_back(std::string_view("b"));
  cfg.as_object().insert("class_names", std::move(names));

  YoloDetectionStage s(&sess, "y", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(yolo_detection_stage, model_drives_input_size_and_num_classes)
{
  const char* model = env_or_null_("VPIPE_TEST_YOLOV6_MODEL");
  if (!model) {
    return;
  }
  // Use mismatched src dims to exercise the letterbox path; the
  // model's declared input size drives the resample. No input_size
  // or num_classes set in the config.
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto src_u = make_unique<OneTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->W = 640;
  src_u->H = 480;
  src_u->allocate_oports(1);
  auto* src = static_cast<OneTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("model_path", FlexData::make_string(model));
  // No input_size / num_classes set — both must come from the model.
  cfg.as_object().insert("confidence_threshold",
                         FlexData::make_real(0.0));
  auto yolo_u = make_unique<YoloDetectionStage>(
      &sess, "yolo", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* yolo = static_cast<YoloDetectionStage*>(
      pl->insert_stage(std::move(yolo_u)));

  auto sink_u = make_unique<FlexDataSink>(
      &sess, "sink", vector<InEdge>{{yolo, 0}},
      FlexData::make_object());
  auto* sink = static_cast<FlexDataSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  EXPECT_TRUE(got.size() == 1u);
  if (got.empty()) {
    return;
  }
  const FlexData& fd = got[0];
  EXPECT_TRUE(fd.is_object());
  auto root = fd.as_object();
  EXPECT_TRUE(root.contains("frame_width"));
  EXPECT_TRUE(root.contains("frame_height"));
  EXPECT_TRUE(root.contains("detections"));
  EXPECT_TRUE(root.at("frame_width").as_int(0) == 640);
  EXPECT_TRUE(root.at("frame_height").as_int(0) == 480);
  FlexData dets = root.at("detections");
  EXPECT_TRUE(dets.is_array());
  auto av = dets.as_array();
  // With threshold 0 there should be at least one detection.
  EXPECT_TRUE(av.size() > 0u);
  if (av.size() == 0) {
    return;
  }
  FlexData first = av.at(0);
  EXPECT_TRUE(first.is_object());
  auto eo = first.as_object();
  EXPECT_TRUE(eo.contains("class_id"));
  EXPECT_TRUE(eo.contains("score"));
  EXPECT_TRUE(eo.contains("x1"));
  EXPECT_TRUE(eo.contains("y1"));
  EXPECT_TRUE(eo.contains("x2"));
  EXPECT_TRUE(eo.contains("y2"));
  const float x1 = static_cast<float>(eo.at("x1").as_real(-1.0));
  const float x2 = static_cast<float>(eo.at("x2").as_real(-1.0));
  EXPECT_TRUE(x1 >= 0.0f && x2 <= 640.0f && x1 <= x2);
}
