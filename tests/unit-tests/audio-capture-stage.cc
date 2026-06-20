#include "minitest.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/stage-registry.h"
#include "stages/audio-video/audio-capture-stage.h"

#include <string>
#include <utility>

using namespace std;
using namespace vpipe;

TEST(audio_capture_stage, type_is_registered) {
  EXPECT_TRUE(StageRegistry::get().find_id("audio-capture") !=
              StageTypeId::unknown);
}

TEST(audio_capture_stage, missing_device_deferred) {
  Session sess;
  AudioCaptureStage s(&sess, "ac", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_capture_stage, both_device_id_and_name_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "device_id",   FlexData::make_uint(0));
  cfg.as_object().insert_or_assign(
      "device_name", FlexData::make_string("Built-in"));
  AudioCaptureStage s(&sess, "ac", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_capture_stage, device_id_accepted) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "device_id", FlexData::make_uint(0));
  AudioCaptureStage s(&sess, "ac", {}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.has_device_id());
  EXPECT_TRUE(s.device_id() == 0u);
  EXPECT_TRUE(s.device_name().empty());
}

TEST(audio_capture_stage, device_name_accepted) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "device_name", FlexData::make_string("Built-in Microphone"));
  AudioCaptureStage s(&sess, "ac", {}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_FALSE(s.has_device_id());
  EXPECT_TRUE(s.device_name() == "Built-in Microphone");
}

TEST(audio_capture_stage, negative_device_id_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "device_id", FlexData::make_int(-1));
  AudioCaptureStage s(&sess, "ac", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_capture_stage, non_object_config_deferred) {
  Session sess;
  AudioCaptureStage s(&sess, "ac", {}, FlexData::make_string("nope"));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_capture_stage, oport_count_is_one) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "device_id", FlexData::make_uint(0));
  AudioCaptureStage s(&sess, "ac", {}, std::move(cfg));
  EXPECT_TRUE(s.num_oports() == 1u);
}
