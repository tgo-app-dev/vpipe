// video-capture: the config surface of the camera source stage.
//
// Mirrors audio-capture-stage.cc (same device_id / device_name exclusivity
// rules), plus the knobs that are video-only: width/height must be set as a
// pair because avfoundation takes ONE "WxH" video_size string, and
// output_dtype selects the emitted TensorBeat element type.
//
// Opening a real camera is deliberately NOT exercised here: it needs a
// physical device and a granted TCC permission, and on a denied/undetermined
// system avfoundation can block. The startup checks report that state
// separately (apps/web-ui/startup-checks.cc).

#include "minitest.h"
#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/stage-registry.h"
#include "stages/audio-video/video-capture-stage.h"

#include <string>
#include <utility>

using namespace std;
using namespace vpipe;

namespace {

FlexData
cfg_with_device_()
{
  FlexData c = FlexData::make_object();
  c.as_object().insert_or_assign("device_id", FlexData::make_uint(0));
  return c;
}

}  // namespace

TEST(video_capture_stage, type_is_registered) {
  EXPECT_TRUE(StageRegistry::get().find_id("video-capture") !=
              StageTypeId::unknown);
}

TEST(video_capture_stage, missing_device_deferred) {
  Session sess;
  VideoCaptureStage s(&sess, "vc", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(video_capture_stage, both_device_id_and_name_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("device_id", FlexData::make_uint(0));
  cfg.as_object().insert_or_assign("device_name",
                                   FlexData::make_string("FaceTime"));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(video_capture_stage, device_id_accepted) {
  Session sess;
  VideoCaptureStage s(&sess, "vc", {}, cfg_with_device_());
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.has_device_id());
  EXPECT_TRUE(s.device_id() == 0u);
  EXPECT_TRUE(s.device_name().empty());
}

TEST(video_capture_stage, device_name_accepted) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("device_name",
                                   FlexData::make_string("MacBook Air Camera"));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_FALSE(s.has_device_id());
  EXPECT_TRUE(s.device_name() == "MacBook Air Camera");
}

TEST(video_capture_stage, negative_device_id_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("device_id", FlexData::make_int(-1));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(video_capture_stage, non_object_config_deferred) {
  Session sess;
  VideoCaptureStage s(&sess, "vc", {}, FlexData::make_string("nope"));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(video_capture_stage, oport_count_is_one) {
  Session sess;
  VideoCaptureStage s(&sess, "vc", {}, cfg_with_device_());
  EXPECT_TRUE(s.num_oports() == 1u);
}

// avfoundation's video_size is one "WxH" string, so half a resolution cannot
// be honoured; it must be rejected rather than silently completed from the
// device default.
TEST(video_capture_stage, half_specified_resolution_deferred) {
  for (const char* key : {"width", "height"}) {
    Session sess;
    FlexData cfg = cfg_with_device_();
    cfg.as_object().insert_or_assign(key, FlexData::make_uint(1280));
    VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
    EXPECT_FALSE(s.config_error().empty());
  }
}

TEST(video_capture_stage, resolution_and_framerate_accepted) {
  Session sess;
  FlexData cfg = cfg_with_device_();
  cfg.as_object().insert_or_assign("width", FlexData::make_uint(1280));
  cfg.as_object().insert_or_assign("height", FlexData::make_uint(720));
  cfg.as_object().insert_or_assign("framerate", FlexData::make_real(30.0));
  cfg.as_object().insert_or_assign("pixel_format",
                                   FlexData::make_string("nv12"));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.req_width() == 1280u);
  EXPECT_TRUE(s.req_height() == 720u);
  EXPECT_TRUE(s.req_framerate() == 30.0);
  EXPECT_TRUE(s.pixel_format() == "nv12");
}

TEST(video_capture_stage, negative_framerate_deferred) {
  Session sess;
  FlexData cfg = cfg_with_device_();
  cfg.as_object().insert_or_assign("framerate", FlexData::make_real(-1.0));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// Unset resolution / framerate means "device default", NOT 0x0 at 0 fps --
// the stage must simply omit the avfoundation option in that case.
TEST(video_capture_stage, unset_resolution_means_device_default) {
  Session sess;
  VideoCaptureStage s(&sess, "vc", {}, cfg_with_device_());
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.req_width() == 0u);
  EXPECT_TRUE(s.req_height() == 0u);
  EXPECT_TRUE(s.req_framerate() == 0.0);
  EXPECT_TRUE(s.pixel_format().empty());
}

TEST(video_capture_stage, output_dtype_defaults_to_u8) {
  Session sess;
  VideoCaptureStage s(&sess, "vc", {}, cfg_with_device_());
  EXPECT_TRUE(s.output_dtype() == TensorBeat::DType::U8);
}

TEST(video_capture_stage, output_dtype_f32_accepted) {
  Session sess;
  FlexData cfg = cfg_with_device_();
  cfg.as_object().insert_or_assign("output_dtype",
                                   FlexData::make_string("f32"));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.output_dtype() == TensorBeat::DType::F32);
}

TEST(video_capture_stage, bad_output_dtype_deferred) {
  Session sess;
  FlexData cfg = cfg_with_device_();
  cfg.as_object().insert_or_assign("output_dtype",
                                   FlexData::make_string("rgb24"));
  VideoCaptureStage s(&sess, "vc", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// The oport must carry the same payload type video-to-rgb emits, so a camera
// is a drop-in replacement for a decoded file/RTSP source.
TEST(video_capture_stage, oport_payload_matches_video_to_rgb) {
  const StageSpec* vc = StageRegistry::get().spec("video-capture");
  const StageSpec* vr = StageRegistry::get().spec("video-to-rgb");
  ASSERT_TRUE(vc != nullptr && vr != nullptr);
  ASSERT_TRUE(vc->oports.size() == 1 && vr->oports.size() == 1);
  EXPECT_TRUE(vc->oports[0].type == vr->oports[0].type);
  EXPECT_TRUE(vc->iports.empty());
}
