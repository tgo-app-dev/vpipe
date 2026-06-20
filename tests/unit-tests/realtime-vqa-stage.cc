// realtime-vqa-stage.cc -- tests for the streaming video-QA stage. The
// config/registration cases are MLX-free and run in BOTH the MLX and
// no-MLX builds (the stage now builds on the VPIPE_BUILD_APPLE_SILICON
// axis, carrying a self-contained metal/no-MLX generation path).
//
// The end-to-end runtime smoke is env-gated on VPIPE_METAL_VQA_SMOKE_MODEL
// (a metal-supported Qwen3-VL dir, same var as metal_lm_smoke.image_vqa):
// it drives synthetic RGB frames + chrono ticks through a real mini-
// pipeline and asserts a scene answer beat is emitted. Under no-MLX this
// exercises the metal path; under MLX-ON it exercises the MLX path.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/realtime-vqa-stage.h"
#include "stages/trigger-beat.h"

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>
#include <string_view>
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
basic_cfg_()
{
  FlexData cfg = FlexData::make_object();
  auto o = cfg.as_object();
  o.insert("hf_dir", FlexData::make_string("/tmp/rvqa-fake-model"));
  o.insert("questions", FlexData::make_string("What is happening?"));
  return cfg;
}

}  // namespace

// ---- Config / construction (no model; both builds) -----------------

TEST(realtime_vqa_stage, type_is_registered) {
  EXPECT_TRUE(string_view(RealtimeVqaStage::kTypeName) == "realtime-vqa");
}

TEST(realtime_vqa_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.hf_dir() == "/tmp/rvqa-fake-model");
  EXPECT_TRUE(s.questions().size() == 1);
  EXPECT_TRUE(s.questions()[0] == "What is happening?");
  EXPECT_TRUE(s.max_frame_gap_ms() == 10000);
  EXPECT_TRUE(s.idle_ticks_to_end() == 2);
  EXPECT_TRUE(s.scenes_closed() == 0u);
}

TEST(realtime_vqa_stage, config_questions_array) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","questions":["q1","q2","q3"]})");
  RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.questions().size() == 3);
  EXPECT_TRUE(s.questions()[0] == "q1");
  EXPECT_TRUE(s.questions()[2] == "q3");
}

// The per-question preamble defaults to a yes/no/unknown answer-format
// steer, is overridable, and an empty string disables it. It is prepended
// (with a blank line) to every per-question user turn -- see the close_scene
// render sites -- so all branches inherit the same instruction.
TEST(realtime_vqa_stage, question_preamble_default_and_override) {
  Session sess;
  CerrSilencer hush;
  {
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, basic_cfg_());
    EXPECT_FALSE(s.question_preamble().empty());          // default on
    EXPECT_TRUE(s.question_preamble().find("yes/no/unknown")
                != std::string::npos);
  }
  {
    FlexData cfg = FlexData::from_json(
        R"({"hf_dir":"/p","questions":"q","question_preamble":"Be terse."})");
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
    EXPECT_TRUE(s.question_preamble() == "Be terse.");
  }
  {
    FlexData cfg = FlexData::from_json(
        R"({"hf_dir":"/p","questions":"q","question_preamble":""})");
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
    EXPECT_TRUE(s.question_preamble().empty());            // disabled
  }
}

// Localization: the "language" config selects the built-in scene prompts.
// The per-question preamble's locale default is the observable proxy for
// the (otherwise model-internal) prompt selection -- it routes through the
// same effective_language_() + locale table as the describe / preamble /
// audio-interpret prompts.
TEST(realtime_vqa_stage, language_config_selects_localized_prompts) {
  Session sess;                       // default session locale en-us
  CerrSilencer hush;
  // Explicit zh-cn pins the locale and swaps in the Simplified-Chinese
  // question-preamble default (no English "yes/no/unknown"; carries CJK).
  {
    FlexData cfg = FlexData::from_json(
        R"({"hf_dir":"/p","questions":"q","language":"zh-cn"})");
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
    EXPECT_TRUE(s.language() == "zh-cn");
    EXPECT_FALSE(s.question_preamble().empty());
    EXPECT_TRUE(s.question_preamble().find("yes/no/unknown")
                == std::string::npos);
    EXPECT_TRUE(s.question_preamble().find("\xE5\x9B\x9E\xE7\xAD\x94")
                != std::string::npos);   // "回答" (answer)
  }
  // A tag variant normalizes (zh-Hant -> zh-tw).
  {
    FlexData cfg = FlexData::from_json(
        R"({"hf_dir":"/p","questions":"q","language":"zh-Hant"})");
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
    EXPECT_TRUE(s.language() == "zh-tw");
    EXPECT_TRUE(s.question_preamble().find("yes/no/unknown")
                == std::string::npos);
  }
  // An unsupported explicit tag normalizes to "" (inherit), so the
  // English (default session) prompt is used.
  {
    FlexData cfg = FlexData::from_json(
        R"({"hf_dir":"/p","questions":"q","language":"fr"})");
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
    EXPECT_TRUE(s.language().empty());
    EXPECT_TRUE(s.question_preamble().find("yes/no/unknown")
                != std::string::npos);
  }
}

// With no "language" config the stage inherits the session locale, so the
// prompts follow a session configured (or later set) to another language.
TEST(realtime_vqa_stage, language_inherits_session_locale) {
  Session sess(R"({"language":"zh-tw"})");
  CerrSilencer hush;
  RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.language().empty());          // unset -> inherit
  EXPECT_FALSE(s.question_preamble().empty());
  EXPECT_TRUE(s.question_preamble().find("yes/no/unknown")
              == std::string::npos);          // not the English default
}

// The prior-scene recap defaults ON (carried only across temporally-
// continuous scenes -- see prev_recap_) and is disabled by setting
// prev_scene_recap=false.
TEST(realtime_vqa_stage, prev_scene_recap_default_and_override) {
  Session sess;
  CerrSilencer hush;
  {
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, basic_cfg_());
    EXPECT_TRUE(s.prev_scene_recap());                     // default on
  }
  {
    FlexData cfg = FlexData::from_json(
        R"({"hf_dir":"/p","questions":"q","prev_scene_recap":false})");
    RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
    EXPECT_FALSE(s.prev_scene_recap());                    // disabled
  }
}

// Construction must succeed for any config (graph editing before all
// required fields are present); the problem is deferred to launch.
TEST(realtime_vqa_stage, missing_hf_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"questions":"q"})");
  RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(realtime_vqa_stage, missing_questions_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"hf_dir":"/p"})");
  RealtimeVqaStage s(&sess, "rvqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// ---- End-to-end runtime smoke (env-gated; real VLM) ----------------
//
// Targets the self-contained metal/no-MLX generation path, which only
// exists in the no-MLX build (the MLX build drives the stage through the
// MLX path instead). Compiled only there.
#if defined(VPIPE_BUILD_APPLE_SILICON)

namespace {

// Single combined driver feeding the realtime-vqa stage. oport0 carries
// synthetic u8 [3,H,W] RGB frames; oport1 carries chrono-style ticks.
// Emitting both from ONE coroutine in order (all frames, THEN ticks)
// makes the test deterministic: by the time the stage reads its first
// tick on iport1, every frame is already sitting in iport0's backlog.
// (Two independent sources race -- the ticks can drain before any frame
// lands, and the scene never becomes active.)
//
// The frames share one scene (timestamps a fixed sub-gap step apart);
// `idle_ticks` after the frames close the scene via the idle-tick rule.
class VqaDriverSource : public TypedStage<VqaDriverSource> {
public:
  static constexpr const char* kTypeName = "ut-vqa-driver-source";
  using TypedStage::TypedStage;

  int           n_frames = 2;
  int           H        = 128;
  int           W        = 128;
  int           n_ticks  = 8;
  std::uint64_t base_us  = 1'000'000;
  std::uint64_t step_us  = 100'000;   // 0.1 s -> same scene
  // Number of distinct scenes to emit: each is `n_frames` frames step_us
  // apart, with a `scene_gap_us` jump between scenes so the stage's
  // max_frame_gap_ms rule closes the prior scene. 1 = single scene.
  int           n_scenes    = 1;
  std::uint64_t scene_gap_us = 5'000'000;   // 5 s >> default 1 s gap
  // Optional iport2 audio-tagging beats (oport2). Each pair is
  // {timestamp_us, label}; the stage drains those with ts < the latest
  // frame ts ("prior to its image data") and queues the rest.
  bool          emit_audio = false;
  std::vector<std::pair<std::uint64_t, std::string>> audio;
  // Optional iport2 PCM TensorBeats (a pcm stage): each pair is
  // {timestamp_us, n_samples}; the stage senses the F32 mono PCM, drains
  // those before the frame boundary, and (with an audio-capable LM) splices
  // them as soft tokens at scene close.
  bool          emit_pcm = false;
  std::vector<std::pair<std::uint64_t, std::size_t>> pcm;
  std::string   camera_name;   // when set, attached to each frame sideband

  Job process(RuntimeContext& ctx) override
  {
    if (_done) { ctx.signal_done(); co_return; }
    _done = true;
    // 1) frames on oport0 (default ring depth easily holds the frames).
    //    n_scenes batches, each `n_frames` frames step_us apart, with a
    //    scene_gap_us jump between batches so the stage splits scenes.
    for (int s = 0; s < std::max(1, n_scenes); ++s) {
      const std::uint64_t scene_base =
          base_us + static_cast<std::uint64_t>(s) * scene_gap_us;
      for (int idx = 0; idx < n_frames; ++idx) {
        TensorBeat tb;
        tb.dtype = TensorBeat::DType::U8;
        tb.shape = { 3, H, W };
        tb.resize_contiguous(static_cast<size_t>(3) * H * W);
        std::uint8_t* p = tb.as_u8();
        for (int c = 0; c < 3; ++c) {
          for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
              p[(static_cast<size_t>(c) * H + y) * W + x] =
                  static_cast<std::uint8_t>(
                      (x + 2 * y + 40 * c + 7 * idx + 13 * s) & 0xFF);
            }
          }
        }
        tb.sideband = FlexData::make_object();
        tb.sideband.as_object().insert(
            "timestamp_us",
            FlexData::make_uint(scene_base + static_cast<std::uint64_t>(idx)
                                                 * step_us));
        if (!camera_name.empty()) {
          tb.sideband.as_object().insert(
              "camera_name", FlexData::make_string(camera_name));
        }
        co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
      }
    }
    // 2) audio-tagging beats on oport2 (-> stage iport2), written BEFORE
    //    the ticks so they're already queued when the first tick drains.
    //    Mirrors the audio-tagging stage's FlexData: { timestamp_us,
    //    tags:[{label, score}, ...] } sorted by descending score.
    if (emit_audio) {
      for (const auto& ev : audio) {
        FlexData fd = FlexData::make_object();
        auto o = fd.as_object();
        o.insert("timestamp_us", FlexData::make_uint(ev.first));
        FlexData tags = FlexData::make_array();
        {
          auto ta = tags.as_array();
          FlexData t0 = FlexData::make_object();
          t0.as_object().insert("label", FlexData::make_string(ev.second));
          t0.as_object().insert("score", FlexData::make_real(0.9));
          ta.push_back(std::move(t0));
        }
        o.insert("tags", std::move(tags));
        co_await ctx.write(2, make_payload<FlexDataPayload>(std::move(fd)));
      }
    }
    // 2b) PCM TensorBeats on oport2 (mono f32 @ 16 kHz + sideband ts).
    if (emit_pcm) {
      for (const auto& seg : pcm) {
        TensorBeat tb;
        tb.dtype = TensorBeat::DType::F32;
        tb.shape = { static_cast<std::int64_t>(seg.second) };
        tb.resize_contiguous(seg.second);
        float* s = tb.as_f32();
        for (std::size_t i = 0; i < seg.second; ++i) {
          s[i] = 0.1f * std::sin(2.0f * 3.14159265f * 440.0f
                                 * static_cast<float>(i) / 16000.0f);
        }
        tb.sideband = FlexData::make_object();
        tb.sideband.as_object().insert("timestamp_us",
                                       FlexData::make_uint(seg.first));
        tb.sideband.as_object().insert("sample_rate",
                                       FlexData::make_uint(16000));
        co_await ctx.write(2, make_payload<TensorBeatPayload>(std::move(tb)));
      }
    }
    // 3) ticks on oport1. The realtime-vqa stage only treats iport1
    //    beats as a clock pulse (never inspects the payload), so a bare
    //    TriggerPayload is a valid tick.
    for (int t = 0; t < n_ticks; ++t) {
      co_await ctx.write(1, make_payload<TriggerPayload>());
    }
    // Next process() call signals done (EOS flushes any open scene).
  }

private:
  bool _done = false;
};

// Collects every FlexData scene beat the stage emits on oport0.
class VqaSceneSink : public TypedStage<VqaSceneSink> {
public:
  static constexpr const char* kTypeName = "ut-vqa-scene-sink";
  using TypedStage::TypedStage;

  std::vector<FlexData> take()
  {
    lock_guard<mutex> g(_mu);
    return _collected;
  }

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    const auto* p = dynamic_cast<const FlexDataPayload*>(in.get());
    if (p) {
      lock_guard<mutex> g(_mu);
      _collected.push_back(p->data);
    }
  }

private:
  mutex                 _mu;
  std::vector<FlexData> _collected;
};

// Throwaway LMDB directory for the persistence test.
struct TempDir {
  std::string path;
  TempDir()
  {
    auto base = std::filesystem::temp_directory_path() / "rvqa-db-XXXXXX";
    std::string tmpl = base.string();
    if (!mkdtemp(tmpl.data())) { throw std::runtime_error("mkdtemp failed"); }
    path = tmpl;
  }
  ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

// Count records in an LMDB sub-db (0 if the db doesn't exist / is empty).
std::size_t
count_db_(LmdbEnv& env, const std::string& db_name)
{
  try {
    LmdbDb     db(env, db_name);
    LmdbTxn    txn(env, LmdbTxn::Mode::ReadOnly);
    LmdbCursor cur(txn, db);
    std::size_t n = 0;
    std::string_view k, v;
    if (cur.first(k, v)) { do { ++n; } while (cur.next(k, v)); }
    return n;
  } catch (...) {
    return 0;
  }
}

}  // namespace

TEST(realtime_vqa_stage, metal_pipeline_smoke) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("rvqa-smoke", &sess);

  auto drv = make_unique<VqaDriverSource>(
      &sess, "driver", vector<InEdge>{}, FlexData::make_object());
  drv->allocate_oports(2);   // oport0 = frames, oport1 = ticks
  auto* driver = static_cast<VqaDriverSource*>(
      pl.insert_stage(std::move(drv)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    // Two questions -> exercises the batched (pipelined) multi-branch
    // decode path end-to-end, not just single-branch.
    FlexData qs = FlexData::make_array();
    {
      auto v = qs.as_array();
      v.push_back(FlexData::make_string("What color dominates the scene?"));
      v.push_back(FlexData::make_string("Is anything moving?"));
    }
    o.insert("questions", std::move(qs));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(24));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("idle_ticks_to_end", FlexData::make_int(2));
    // Optional: exercise the CoreML vision tower on the metal path when
    // VPIPE_METAL_VQA_COREML_PATH points at a .mlpackage.
    if (const char* cmlp = std::getenv("VPIPE_METAL_VQA_COREML_PATH");
        cmlp && *cmlp) {
      o.insert("coreml_vision_path", FlexData::make_string(cmlp));
    }
  }
  auto rv = make_unique<RealtimeVqaStage>(
      &sess, "rvqa",
      vector<InEdge>{ { driver, 0 }, { driver, 1 } }, std::move(cfg));
  auto* rvqa = static_cast<RealtimeVqaStage*>(
      pl.insert_stage(std::move(rv)));

  auto sk = make_unique<VqaSceneSink>(
      &sess, "sink", vector<InEdge>{ { rvqa, 0 } },
      FlexData::make_object());
  auto* sink = static_cast<VqaSceneSink*>(
      pl.insert_stage(std::move(sk)));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  // Metal compute or the model may be unavailable on this host; a failed
  // launch (e.g. config deferred because the model path is bad) is not a
  // test failure for an env-gated smoke -- but the env was set, so we do
  // expect a launch when the box is configured.
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();

  auto scenes = sink->take();
  std::printf("[realtime_vqa_stage.metal_smoke] scenes=%zu closed=%llu\n",
              scenes.size(),
              static_cast<unsigned long long>(rvqa->scenes_closed()));
  ASSERT_TRUE(!scenes.empty());

  // At least one closed scene must carry the captured frames, the
  // configured question, and a non-empty description + answer produced
  // by the LM.
  bool found = false;
  for (const auto& fd : scenes) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (!root.contains("scene_description")) { continue; }
    const std::string desc(
        root.at("scene_description").as_string(""));
    const std::uint64_t n_frames = root.contains("n_frames")
        ? root.at("n_frames").as_uint(0) : 0;
    // Two questions -> the batched (pipelined) decode must fill BOTH
    // answers non-empty.
    std::string answer;
    bool both_answers = false;
    if (root.contains("answers")) {
      FlexData as = root.at("answers");
      if (as.is_array()) {
        auto v = as.as_array();
        if (!v.empty()) { answer = std::string(v.at(0).as_string("")); }
        both_answers = v.size() == 2
            && !std::string(v.at(0).as_string("")).empty()
            && !std::string(v.at(1).as_string("")).empty();
      }
    }
    std::printf("  scene n_frames=%llu desc='%s' answer='%s' both=%d\n",
                static_cast<unsigned long long>(n_frames),
                desc.c_str(), answer.c_str(), both_answers ? 1 : 0);
    if (n_frames >= 1 && !desc.empty() && both_answers) {
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

// Reusable branch pool: drive TWO scenes through one stage instance and
// assert (a) both scenes produce coherent per-question answers (the pooled
// branch contexts are rebranched off each scene's base WITHOUT stale KV from
// the prior scene), and (b) the pool is reserved ONCE and stays sized to the
// question count across scenes (reused, not churned). Gated on the same
// VPIPE_METAL_VQA_SMOKE_MODEL env as the smoke test.
TEST(realtime_vqa_stage, metal_branch_pool_reuse) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("rvqa-pool", &sess);

  auto drv = make_unique<VqaDriverSource>(
      &sess, "driver", vector<InEdge>{}, FlexData::make_object());
  drv->allocate_oports(2);
  auto* driver = static_cast<VqaDriverSource*>(
      pl.insert_stage(std::move(drv)));
  driver->n_scenes = 2;        // two frame batches, 5 s gap -> two scenes
  driver->n_ticks  = 12;       // enough idle ticks to close the 2nd scene

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    FlexData qs = FlexData::make_array();
    {
      auto v = qs.as_array();
      v.push_back(FlexData::make_string("What color dominates the scene?"));
      v.push_back(FlexData::make_string("Is anything moving?"));
    }
    o.insert("questions", std::move(qs));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(24));
    o.insert("max_pages", FlexData::make_int(32));
    o.insert("idle_ticks_to_end", FlexData::make_int(2));
    if (const char* cmlp = std::getenv("VPIPE_METAL_VQA_COREML_PATH");
        cmlp && *cmlp) {
      o.insert("coreml_vision_path", FlexData::make_string(cmlp));
    }
  }
  auto rv = make_unique<RealtimeVqaStage>(
      &sess, "rvqa",
      vector<InEdge>{ { driver, 0 }, { driver, 1 } }, std::move(cfg));
  auto* rvqa = static_cast<RealtimeVqaStage*>(
      pl.insert_stage(std::move(rv)));

  auto sk = make_unique<VqaSceneSink>(
      &sess, "sink", vector<InEdge>{ { rvqa, 0 } }, FlexData::make_object());
  auto* sink = static_cast<VqaSceneSink*>(pl.insert_stage(std::move(sk)));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();

  auto scenes = sink->take();
  std::printf("[realtime_vqa_stage.pool_reuse] scenes=%zu closed=%llu "
              "pool=%zu\n",
              scenes.size(),
              static_cast<unsigned long long>(rvqa->scenes_closed()),
              rvqa->branch_pool_size());
  // Two scenes must have closed and been emitted.
  EXPECT_TRUE(rvqa->scenes_closed() >= 2u);
  // The pool was reserved once and reused -- sized to the question count,
  // not grown per scene.
  EXPECT_TRUE(rvqa->branch_pool_size() == static_cast<std::size_t>(2));

  // Every emitted scene that carries answers must have BOTH non-empty (a
  // rebranch that leaked the prior scene's KV / dropped a branch would show
  // up as an empty / missing answer on the 2nd scene).
  int coherent = 0;
  for (const auto& fd : scenes) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (!root.contains("scene_description")) { continue; }
    const std::string desc(root.at("scene_description").as_string(""));
    bool both = false;
    if (root.contains("answers")) {
      FlexData as = root.at("answers");   // bind: as_array() views into `as`
      if (as.is_array()) {
        auto v = as.as_array();
        both = v.size() == 2
            && !std::string(v.at(0).as_string("")).empty()
            && !std::string(v.at(1).as_string("")).empty();
      }
    }
    if (!desc.empty() && both) { ++coherent; }
  }
  std::printf("  coherent scenes=%d\n", coherent);
  EXPECT_TRUE(coherent >= 2);
}

// Regression: the metal (no-MLX) path must DRAIN iport2 audio-tagging
// beats up to the latest video frame timestamp -- before the fix it never
// read iport2, so beats backed up all the way to rtsp-capture. Two frames
// put the drain boundary at base+step; we feed three classifications
// before it (one of them "Silence") and one after. The three before must
// be released (Silence included -- it's consumed, just not woven into the
// prompt); the one after stays queued. Gated on the same model env.
TEST(realtime_vqa_stage, metal_audio_drain) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("rvqa-audio", &sess);

  auto drv = make_unique<VqaDriverSource>(
      &sess, "driver", vector<InEdge>{}, FlexData::make_object());
  drv->allocate_oports(3);   // oport0 frames, oport1 ticks, oport2 audio
  drv->emit_audio = true;
  // base_us=1'000'000, step_us=100'000, 2 frames -> boundary = 1'100'000.
  drv->audio = {
      {  999'999, "Speech"},    // < boundary -> released + woven
      {1'050'000, "Silence"},   // < boundary -> released, NOT woven
      {1'099'999, "Dog"},       // < boundary -> released + woven
      {1'200'000, "Music"},     // >= boundary -> queued, NOT released
  };
  auto* driver = static_cast<VqaDriverSource*>(
      pl.insert_stage(std::move(drv)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("questions",
             FlexData::make_string("What is happening?"));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(8));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("idle_ticks_to_end", FlexData::make_int(2));
    if (const char* cmlp = std::getenv("VPIPE_METAL_VQA_COREML_PATH");
        cmlp && *cmlp) {
      o.insert("coreml_vision_path", FlexData::make_string(cmlp));
    }
  }
  auto rv = make_unique<RealtimeVqaStage>(
      &sess, "rvqa",
      vector<InEdge>{ { driver, 0 }, { driver, 1 }, { driver, 2 } },
      std::move(cfg));
  auto* rvqa = static_cast<RealtimeVqaStage*>(
      pl.insert_stage(std::move(rv)));

  auto sk = make_unique<VqaSceneSink>(
      &sess, "sink", vector<InEdge>{ { rvqa, 0 } },
      FlexData::make_object());
  pl.insert_stage(std::move(sk));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();

  std::printf("[realtime_vqa_stage.metal_audio_drain] released=%llu "
              "closed=%llu\n",
              static_cast<unsigned long long>(rvqa->audio_beats_released()),
              static_cast<unsigned long long>(rvqa->scenes_closed()));
  // The three classifications before the frame boundary were released
  // (relieving the backpressure); the one after stayed queued.
  EXPECT_TRUE(rvqa->audio_beats_released() == 3);
  EXPECT_TRUE(rvqa->scenes_closed() >= 1);
}

// PCM audio path: iport2 carries F32 mono PCM TensorBeats (a pcm stage)
// instead of audio-tag FlexData. The stage must type-sense them, drain those
// before the frame boundary, accumulate the scene's audio, and (with an
// audio-capable LM) encode + splice them as soft tokens in the describe-
// prefix. Works for both the 12B unified embedder and the e4b USM Conformer.
// Gated on VPIPE_METAL_VQA_AUDIO_MODEL (an audio-capable VQA model dir --
// gemma-4-e4b-it-4bit or the 12B unified GGUF), else VPIPE_GEMMA12B_GGUF_PATH.
// Both PCM beats precede the boundary so both must be released + a scene
// must close.
TEST(realtime_vqa_stage, metal_pcm_audio_scene) {
  const char* path = std::getenv("VPIPE_METAL_VQA_AUDIO_MODEL");
  if (!path || !*path) { path = std::getenv("VPIPE_GEMMA12B_GGUF_PATH"); }
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("rvqa-pcm", &sess);

  auto drv = make_unique<VqaDriverSource>(
      &sess, "driver", vector<InEdge>{}, FlexData::make_object());
  drv->allocate_oports(3);
  drv->emit_pcm = true;
  // ONE multi-frame scene: 4 frames 20 ms apart (1'000'000..1'060'000), well
  // within max_frame_gap. Audio is only collected for the scene's [first,
  // last) span, so the PCM beats sit strictly inside (a single-frame scene
  // would have an empty window). Three 0.4 s PCM beats (6400 samples @16k).
  drv->n_frames = 4;
  drv->step_us  = 20'000;
  drv->pcm = { {1'010'000, 6400}, {1'030'000, 6400}, {1'050'000, 6400} };
  auto* driver = static_cast<VqaDriverSource*>(
      pl.insert_stage(std::move(drv)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("questions", FlexData::make_string("What is happening?"));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(8));
    o.insert("max_pages", FlexData::make_int(8));
    // High idle threshold so the scene closes on EOS (not idle), keeping all
    // four frames -- and the PCM between them -- in ONE scene.
    o.insert("idle_ticks_to_end", FlexData::make_int(1000));
  }
  auto rv = make_unique<RealtimeVqaStage>(
      &sess, "rvqa",
      vector<InEdge>{ { driver, 0 }, { driver, 1 }, { driver, 2 } },
      std::move(cfg));
  auto* rvqa = static_cast<RealtimeVqaStage*>(
      pl.insert_stage(std::move(rv)));

  auto sk = make_unique<VqaSceneSink>(
      &sess, "sink", vector<InEdge>{ { rvqa, 0 } },
      FlexData::make_object());
  auto* sink = static_cast<VqaSceneSink*>(pl.insert_stage(std::move(sk)));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();

  const auto scenes = sink->take();
  std::printf("[realtime_vqa_stage.metal_pcm_audio_scene] released=%llu "
              "closed=%llu spliced=%llu scenes_emitted=%zu\n",
              static_cast<unsigned long long>(rvqa->audio_beats_released()),
              static_cast<unsigned long long>(rvqa->scenes_closed()),
              static_cast<unsigned long long>(rvqa->audio_tokens_spliced()),
              scenes.size());
  // All three PCM beats were type-sensed + drained (relieving backpressure),
  // a scene closed, AND the scene's audio was actually encoded + spliced as
  // soft tokens (the real proof the PCM path runs through the audio encoder).
  EXPECT_TRUE(rvqa->audio_beats_released() == 3);
  EXPECT_TRUE(rvqa->scenes_closed() >= 1);
  EXPECT_TRUE(rvqa->audio_tokens_spliced() > 0);
}

// Regression: the metal (no-MLX) path must PERSIST closed scenes to LMDB
// (<camera>-video-qa) and seed the questions epoch (<camera>-video-questions)
// -- this was MLX-only before the MLX/metal unification, so the metal build
// silently dropped both. Run with a temp LMDB env + a camera_name in the
// frame sideband, then read the sub-dbs back. Gated on the same model env.
TEST(realtime_vqa_stage, metal_persists_scene_to_lmdb) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  TempDir tdir;
  const std::string sess_cfg =
      R"({"db":{"path":")" + tdir.path + R"("}})";
  Session  sess(sess_cfg);
  Pipeline pl("rvqa-persist", &sess);

  auto drv = make_unique<VqaDriverSource>(
      &sess, "driver", vector<InEdge>{}, FlexData::make_object());
  drv->allocate_oports(2);
  drv->camera_name = "cam0";
  auto* driver = static_cast<VqaDriverSource*>(
      pl.insert_stage(std::move(drv)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("questions", FlexData::make_string("What is happening?"));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(8));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("idle_ticks_to_end", FlexData::make_int(2));
    if (const char* cmlp = std::getenv("VPIPE_METAL_VQA_COREML_PATH");
        cmlp && *cmlp) {
      o.insert("coreml_vision_path", FlexData::make_string(cmlp));
    }
  }
  auto rv = make_unique<RealtimeVqaStage>(
      &sess, "rvqa",
      vector<InEdge>{ { driver, 0 }, { driver, 1 } }, std::move(cfg));
  auto* rvqa = static_cast<RealtimeVqaStage*>(
      pl.insert_stage(std::move(rv)));

  auto sk = make_unique<VqaSceneSink>(
      &sess, "sink", vector<InEdge>{ { rvqa, 0 } }, FlexData::make_object());
  pl.insert_stage(std::move(sk));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  ::unsetenv("VPIPE_LLM_BACKEND");
  rt.wait_idle();
  rt.stop();

  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  const std::size_t n_qa  = count_db_(*env, "cam0-video-qa");
  const std::size_t n_q   = count_db_(*env, "cam0-video-questions");
  std::printf("[realtime_vqa_stage.metal_persist] closed=%llu video-qa=%zu "
              "video-questions=%zu\n",
              static_cast<unsigned long long>(rvqa->scenes_closed()),
              n_qa, n_q);
  EXPECT_TRUE(rvqa->scenes_closed() >= 1);
  EXPECT_TRUE(n_qa >= 1);   // the bug: was 0 on the metal path
  EXPECT_TRUE(n_q  >= 1);   // questions epoch seeded
}

#endif  // VPIPE_BUILD_APPLE_SILICON
