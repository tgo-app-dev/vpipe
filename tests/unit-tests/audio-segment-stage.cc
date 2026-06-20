// AudioSegmentStage tests.
//
// The config-parsing tests verify the ctor + config plumbing (they do
// not load the model -- that happens in initialize()). The stage emits
// its segment markers as FlexDataPayload {start_us,end_us,index,is_partial}.

#include "minitest.h"

#include "stages/audio-segment-stage.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/stage-registry.h"

#include <iostream>
#include <streambuf>
#include <string>
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

FlexData
cfg_with_model_(const char* path)
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("model_path",
                                   FlexData::make_string(path));
  return cfg;
}

}  // namespace

TEST(audio_segment_stage, type_is_registered) {
  EXPECT_TRUE(string_view(AudioSegmentStage::kTypeName)
              == "audio-segment");
  auto& reg = StageRegistry::get();
  EXPECT_TRUE(reg.find_id("audio-segment") != StageTypeId::unknown);
}

// The web-ui composer reads StageRegistry::spec() to show a type's
// category + ports. VPIPE_REGISTER_STAGE alone lists the type but a
// MISSING VPIPE_REGISTER_SPEC leaves spec() null -> the editor falls back
// to a generic entry with 0 iports / 0 oports (the bug this guards).
TEST(audio_segment_stage, spec_is_registered) {
  const StageSpec* sp = StageRegistry::get().spec("audio-segment");
  EXPECT_TRUE(sp != nullptr);
  if (!sp) { return; }
  EXPECT_TRUE(sp->category == StageCategory::Audio);
  EXPECT_TRUE(sp->iports.size() == 1u);   // "audio"
  EXPECT_TRUE(sp->oports.size() == 1u);   // "segments"
}

TEST(audio_segment_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  AudioSegmentStage s(&sess, "seg", vector<InEdge>{},
                      cfg_with_model_("/tmp/silero-fake.mlpackage"));
  EXPECT_TRUE(s.model_path() == "/tmp/silero-fake.mlpackage");
  EXPECT_TRUE(s.sample_rate()    == 16000);
  EXPECT_TRUE(s.window_samples() == 576);   // unified-v6 default
  EXPECT_TRUE(s.hop_samples()    == 512);   // hop = 32 ms of new audio
  EXPECT_TRUE(s.speech_threshold()  > 0.49 && s.speech_threshold()  < 0.51);
  EXPECT_TRUE(s.silence_threshold() > 0.34 && s.silence_threshold() < 0.36);
  EXPECT_TRUE(s.min_speech_ms()  == 250);
  EXPECT_TRUE(s.min_silence_ms() == 400);
  EXPECT_TRUE(s.max_segment_s()  > 11.9 && s.max_segment_s() < 12.1);
  EXPECT_TRUE(s.num_oports()     == 1u);
  EXPECT_TRUE(s.segments_emitted() == 0u);
  EXPECT_TRUE(s.frames_run()       == 0u);
}

TEST(audio_segment_stage, config_overrides_pure_window) {
  // For a "pure 512-sample window" Silero export the natural default
  // is hop = window: the user only sets window_samples, the stage
  // mirrors hop to match.
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p",
          "window_samples":512,
          "input_feature_name":"input",
          "prob_feature_name":"output",
          "state_h_in_name":"h",
          "state_c_in_name":"c",
          "state_h_out_name":"hn",
          "state_c_out_name":"cn",
          "sr_feature_name":"sr",
          "state_h_shape":[2,1,64],
          "state_c_shape":[2,1,64]})");
  AudioSegmentStage s(&sess, "seg", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.window_samples() == 512);
  EXPECT_TRUE(s.hop_samples()    == 512);   // mirrored when window-only
  EXPECT_TRUE(s.config_error().empty());
}

TEST(audio_segment_stage, missing_model_path_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::make_object();
  AudioSegmentStage s(&sess, "seg", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_segment_stage, hop_exceeds_window_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","window_samples":512,"hop_samples":1024})");
  AudioSegmentStage s(&sess, "seg", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_segment_stage, silence_above_speech_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p",
          "speech_threshold":0.4,
          "silence_threshold":0.5})");
  AudioSegmentStage s(&sess, "seg", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_segment_stage, bad_max_segment_s_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","max_segment_s":0})");
  AudioSegmentStage s(&sess, "seg", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}
