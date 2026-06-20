// AudioTranscribeStage config-parsing tests. The full prefill/decode
// path is exercised by llm-audio-encoder.cc's real_weights_* tests;
// here we just verify the stage's ctor + config plumbing match the
// docs.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-transcribe-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

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
basic_cfg_()
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir",
                         FlexData::make_string("/tmp/asr-fake-model"));
  return cfg;
}

std::string
make_tempdir_()
{
  auto base = std::filesystem::temp_directory_path()
            / "vpipe-asr-models-XXXXXX";
  std::string tmpl = base.string();
  if (!::mkdtemp(tmpl.data())) {
    throw std::runtime_error("mkdtemp failed");
  }
  return tmpl;
}

// Register a models-DB record {local_path: <local>} under `key`.
void
register_model_(Session& sess, const std::string& key,
                const std::string& local)
{
  LmdbEnv* env = sess.lmdb_env();
  LmdbDb   db(*env, "models");
  LmdbTxn  txn(*env, LmdbTxn::Mode::ReadWrite);
  FlexData rec = FlexData::make_object();
  rec.as_object().insert_or_assign("local_path",
                                   FlexData::make_string(local));
  const std::string bytes = rec.to_binary();
  db.put(txn, key, bytes);
  txn.commit();
}

}

TEST(audio_transcribe_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.hf_dir() == "/tmp/asr-fake-model");
  EXPECT_TRUE(s.max_new_tokens() == 256);
  EXPECT_TRUE(s.sample_rate()    == 16000);
  EXPECT_TRUE(s.language_hint().empty());
  EXPECT_TRUE(s.num_oports() == 0);    // sink stage
  EXPECT_TRUE(s.clips_processed() == 0u);
  // Single iport => block mode by default.
  EXPECT_FALSE(s.streaming());
  EXPECT_TRUE(s.pcm_buffer_s() > 29.9 && s.pcm_buffer_s() < 30.1);
  EXPECT_TRUE(s.late_marker_skip());
  EXPECT_TRUE(s.segments_seen() == 0u);
  EXPECT_TRUE(s.segments_dropped_late() == 0u);
}

// hf_dir may name a models-DB key; resolve_model_dir maps it to the
// record's local_path. A non-key value (a path) is returned unchanged,
// and a DB key wins even when an identically-named directory exists.
TEST(audio_transcribe_stage, hf_dir_resolves_via_models_db) {
  const std::string tmp = make_tempdir_();
  CerrSilencer hush;

  FlexData db = FlexData::make_object();
  db.as_object().insert("path", FlexData::make_string(tmp));
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("db", std::move(db));
  Session sess(cfg.to_json());

  register_model_(sess, "mlx-community/Qwen3-ASR-0.6B-8bit",
                  "/models/mlx-community/Qwen3-ASR-0.6B-8bit");
  // Ambiguous: a key that is ALSO a real directory on disk.
  register_model_(sess, tmp, "/db/wins/over/path");

  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{},
      FlexData::from_json(R"({"hf_dir":"x"})"));

  // Key -> the registered local_path.
  EXPECT_TRUE(s.resolve_model_dir("mlx-community/Qwen3-ASR-0.6B-8bit")
              == "/models/mlx-community/Qwen3-ASR-0.6B-8bit");
  // Unknown value -> returned verbatim (treated as a path).
  EXPECT_TRUE(s.resolve_model_dir("/no/such/key") == "/no/such/key");
  // Ambiguity: DB entry beats the same-named real directory.
  EXPECT_TRUE(s.resolve_model_dir(tmp) == "/db/wins/over/path");

  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

TEST(audio_transcribe_stage, streaming_off_when_single_iport) {
  // A single InEdge keeps the stage in block mode no matter what the
  // streaming knobs say.
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","pcm_buffer_s":15.0})");
  AudioTranscribeStage s(&sess, "asr",
                         vector<InEdge>{ { nullptr, 0 } },
                         std::move(cfg));
  EXPECT_FALSE(s.streaming());
  EXPECT_TRUE(s.pcm_buffer_s() > 14.9 && s.pcm_buffer_s() < 15.1);
}

TEST(audio_transcribe_stage, streaming_on_when_two_iports) {
  // Wiring iport1 latches streaming mode at construction.
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","pcm_buffer_s":60.0,"late_marker_skip":false})");
  AudioTranscribeStage s(
      &sess, "asr",
      vector<InEdge>{ { nullptr, 0 }, { nullptr, 0 } },
      std::move(cfg));
  EXPECT_TRUE(s.streaming());
  EXPECT_TRUE(s.pcm_buffer_s() > 59.9 && s.pcm_buffer_s() < 60.1);
  EXPECT_FALSE(s.late_marker_skip());
}

TEST(audio_transcribe_stage, bad_pcm_buffer_s_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","pcm_buffer_s":0})");
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_transcribe_stage, config_overrides) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p",
          "compute_dtype":"f16",
          "page_tokens":32,
          "max_pages":128,
          "max_new_tokens":64,
          "sample_rate":48000,
          "language_hint":"English"})");
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.hf_dir() == "/p");
  EXPECT_TRUE(s.max_new_tokens() == 64);
  EXPECT_TRUE(s.sample_rate()    == 48000);
  EXPECT_TRUE(s.language_hint()  == "English");
}

// Construction must succeed for any config (so a graph can be built/
// edited before required fields are supplied); the missing/empty/bad
// field is recorded in config_error() and deferred to launch.
TEST(audio_transcribe_stage, missing_hf_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::make_object();
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_transcribe_stage, empty_hf_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"hf_dir":""})");
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_transcribe_stage, bad_max_pages_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","max_pages":0})");
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_transcribe_stage, bad_sample_rate_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","sample_rate":0})");
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_transcribe_stage, type_is_registered) {
  // Same pattern as visual-qa-stage.type_is_registered: the registry
  // entry for "audio-transcribe" exists with the right factory.
  EXPECT_TRUE(string_view(AudioTranscribeStage::kTypeName)
              == "audio-transcribe");
}

// The StageSpec must declare BOTH iports so the web-ui composer offers the
// optional iport1 (segment markers) -- streaming mode (audio-segment ->
// audio-transcribe) is impossible to wire in the editor otherwise.
TEST(audio_transcribe_stage, spec_declares_both_iports) {
  Session sess;
  CerrSilencer hush;
  AudioTranscribeStage s(&sess, "asr", vector<InEdge>{ { nullptr, 0 } },
                         FlexData::from_json(R"({"hf_dir":"/p"})"));
  const StageSpec& sp = s.spec();
  EXPECT_TRUE(sp.iports.size() == 2u);   // "audio" + "segments"
  if (sp.iports.size() == 2u) {
    EXPECT_TRUE(sp.iports[0].name == "audio");
    EXPECT_TRUE(sp.iports[1].name == "segments");
  }
  EXPECT_TRUE(sp.oports.size() == 0u);   // sink
}

// ---- End-to-end runtime smoke (env-gated; real Qwen3-ASR) ----------
//
// Targets the metal/no-MLX transcription path, which only exists in the
// no-MLX build. audio-transcribe is a sink (the transcript goes to
// session()->info), so the assertion is that the metal path (audio
// encode -> mROPE-free multimodal prefill -> decode) runs to completion
// without crashing. Token/word-exact ASR accuracy is covered at the LM
// level by llm-audio-encoder's real_weights tests. Env:
// VPIPE_QWEN3_ASR_TEST_MODEL_PATH.
#if defined(VPIPE_BUILD_APPLE_SILICON)

namespace {

// Emits one f32 mono PCM clip (a 2 s sine sweep) as a [1, N] TensorBeat,
// then signals done. A synthetic clip exercises the full encode/prefill/
// decode wiring without needing a speech fixture.
class OnePcmSource : public TypedStage<OnePcmSource> {
public:
  static constexpr const char* kTypeName = "ut-asr-pcm-source";
  using TypedStage::TypedStage;

  int sr = 16000;
  int n  = 32000;   // 2 s

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = { 1, n };
    tb.resize_contiguous(static_cast<size_t>(n));
    float* p = tb.as_f32();
    const float two_pi = 2.0f * static_cast<float>(std::numbers::pi);
    for (int i = 0; i < n; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(sr);
      const float freq = 200.0f + 80.0f * t;   // slow sweep
      p[i] = 0.3f * std::sin(two_pi * freq * t);
    }
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert("sample_rate",
                                   FlexData::make_int(sr));
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  bool _sent = false;
};

}  // namespace

TEST(audio_transcribe_stage, metal_transcribe_smoke) {
  const char* path = std::getenv("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("asr-smoke", &sess);

  auto src = make_unique<OnePcmSource>(
      &sess, "pcm", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* pcmsrc = static_cast<OnePcmSource*>(pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(24));
    o.insert("max_pages", FlexData::make_int(16));
  }
  auto asr = make_unique<AudioTranscribeStage>(
      &sess, "asr", vector<InEdge>{ { pcmsrc, 0 } }, std::move(cfg));
  auto* asr_stage = static_cast<AudioTranscribeStage*>(
      pl.insert_stage(std::move(asr)));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();
  // Reaching here means the metal ASR path (WhisperFeatureExtractor +
  // MetalAudioEncoder + multimodal prefill + decode) ran end-to-end
  // without crashing. The transcript is surfaced via session()->info.
  EXPECT_TRUE(asr_stage->clips_processed() >= 1u);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
