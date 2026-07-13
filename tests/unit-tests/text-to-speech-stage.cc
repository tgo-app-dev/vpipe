// TextToSpeechStage tests. Config-parsing tests verify the ctor + config
// plumbing match the docs; an env-gated end-to-end smoke (real MOSS-TTS
// LM + codec) drives a text beat through the metal synthesis path and
// asserts the emitted PCM TensorBeat is non-trivial.

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
#include "stages/text-to-speech-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
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
  o.insert("hf_dir", FlexData::make_string("/tmp/moss-tts-fake"));
  o.insert("codec_dir", FlexData::make_string("/tmp/moss-codec-fake"));
  return cfg;
}

}  // namespace

TEST(text_to_speech_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.hf_dir() == "/tmp/moss-tts-fake");
  EXPECT_TRUE(s.codec_dir() == "/tmp/moss-codec-fake");
  EXPECT_TRUE(s.max_new_tokens() == 1024);
  EXPECT_TRUE(s.max_frames() == 1000);  // v1.5-only default
  EXPECT_TRUE(s.num_oports() == 1);     // PCM oport
  EXPECT_TRUE(s.clips_emitted() == 0u);
  EXPECT_TRUE(s.interrupt_on_new_text());   // barge-in default ON
  EXPECT_TRUE(s.config_error().empty());
}

// interrupt_on_new_text parses and can be disabled.
TEST(text_to_speech_stage, interrupt_on_new_text_config) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/a","codec_dir":"/b","interrupt_on_new_text":false})");
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.interrupt_on_new_text());
  EXPECT_TRUE(s.config_error().empty());
}

TEST(text_to_speech_stage, config_overrides) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/a","codec_dir":"/b","max_new_tokens":512})");
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.hf_dir() == "/a");
  EXPECT_TRUE(s.codec_dir() == "/b");
  EXPECT_TRUE(s.max_new_tokens() == 512);
}

// Construction must succeed for any config (so a graph can be built/
// edited before required fields are supplied); the missing/empty/bad
// field is recorded in config_error() and deferred to launch.
TEST(text_to_speech_stage, missing_hf_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"codec_dir":"/b"})");
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(text_to_speech_stage, missing_codec_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"hf_dir":"/a"})");
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(text_to_speech_stage, bad_max_new_tokens_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/a","codec_dir":"/b","max_new_tokens":0})");
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(text_to_speech_stage, type_is_registered) {
  EXPECT_TRUE(string_view(TextToSpeechStage::kTypeName)
              == "text-to-speech");
}

// The StageSpec declares a text iport, an optional PCM reference iport, and
// one PCM oport.
TEST(text_to_speech_stage, spec_ports) {
  Session sess;
  CerrSilencer hush;
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, basic_cfg_());
  const StageSpec& sp = s.spec();
  EXPECT_TRUE(sp.iports.size() == 2u);
  EXPECT_TRUE(sp.oports.size() == 1u);
  if (sp.iports.size() == 2u) {
    EXPECT_TRUE(sp.iports[0].name == "text");
    EXPECT_TRUE(sp.iports[1].name == "audio-ref");
    // audio-ref is its own clock domain (sticky, arrives independently of the
    // text stream + PCM output, which share group 0).
    EXPECT_TRUE(sp.iports[0].clock_group == sp.oports[0].clock_group);
    EXPECT_TRUE(sp.iports[1].clock_group != sp.iports[0].clock_group);
  }
  if (sp.oports.size() == 1u) {
    EXPECT_TRUE(sp.oports[0].name == "pcm");
  }
}

// voice_lock + voice_ref_seconds parse and default sanely; an audio-ref edge
// is optional (construction succeeds with or without it).
TEST(text_to_speech_stage, voice_clone_config) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/a","codec_dir":"/b","voice_lock":true,
          "voice_ref_seconds":5.0})");
  TextToSpeechStage s(&sess, "tts", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
}

// ---- End-to-end runtime smoke (env-gated; real MOSS-TTS) -----------
//
// Targets the metal/no-MLX synthesis path (MOSS models are metal-only,
// so this only exists in the no-MLX build). Drives one text beat through
// text -> delay-pattern code generation -> de-delay -> codec -> 24 kHz
// PCM and asserts the emitted TensorBeat carries a real waveform.
//
// Env: VPIPE_MOSS_TTS_MODEL (LM dir) + VPIPE_MOSS_CODEC_MODEL (codec
// dir). Both unset => skip. Loading the 8B LM (~11 GB) + codec (~1.6 GB)
// takes ~20 s and needs ~13 GB RAM, so this test is slow by design.
#if defined(VPIPE_BUILD_APPLE_SILICON)

namespace {

// Emits one text beat, then signals done.
class OneTextSource : public TypedStage<OneTextSource> {
public:
  static constexpr const char* kTypeName = "ut-tts-text-source";
  using TypedStage::TypedStage;

  std::string text = "Hello, this is a speech synthesis test.";

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    co_await ctx.write(0, make_payload<FlexDataPayload>(
        FlexData::make_string(text)));
  }

private:
  bool _sent = false;
};

// Captures the first emitted PCM TensorBeat (sample count + peak abs).
class PcmCollectorSink : public TypedStage<PcmCollectorSink> {
public:
  static constexpr const char* kTypeName = "ut-tts-pcm-sink";
  using TypedStage::TypedStage;

  std::size_t n_samples = 0;
  double      peak      = 0.0;
  int         sample_rate = 0;

  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (tbp && tbp->dtype == TensorBeat::DType::F32) {
      n_samples = static_cast<std::size_t>(tbp->element_count());
      const float* pcm = tbp->as_f32();
      for (std::size_t i = 0; i < n_samples; ++i) {
        const double a = std::fabs(static_cast<double>(pcm[i]));
        if (a > peak) { peak = a; }
      }
      if (tbp->sideband.is_object()) {
        auto sb = tbp->sideband.as_object();
        if (sb.contains("sample_rate")) {
          sample_rate =
              static_cast<int>(sb.at("sample_rate").as_int(0));
        }
      }
    }
    co_return;
  }
};

}  // namespace

TEST(text_to_speech_stage, metal_synthesis_smoke) {
  const char* lm = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* cc = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (!lm || !*lm || !cc || !*cc) {
    return;
  }

  Session  sess;
  sess.enable_profiling(8192);   // capture the TTS LLM-lane perf blocks
  EXPECT_TRUE(sess.profiling_enabled());
  Pipeline pl("tts-smoke", &sess);

  auto src = make_unique<OneTextSource>(
      &sess, "text", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* txtsrc = static_cast<OneTextSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(lm));
    o.insert("codec_dir", FlexData::make_string(cc));
    o.insert("max_new_tokens", FlexData::make_int(512));
  }
  auto tts = make_unique<TextToSpeechStage>(
      &sess, "tts", vector<InEdge>{ { txtsrc, 0 } }, std::move(cfg));
  auto* tts_stage = static_cast<TextToSpeechStage*>(
      pl.insert_stage(std::move(tts)));

  auto sink = make_unique<PcmCollectorSink>(
      &sess, "sink", vector<InEdge>{ { tts_stage, 0 } },
      FlexData::make_object());
  auto* sink_stage = static_cast<PcmCollectorSink*>(
      pl.insert_stage(std::move(sink)));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(tts_stage->clips_emitted() >= 1u);
  // A real waveform: well over 1000 samples and a non-trivial peak.
  std::printf("[tts-smoke] samples=%zu peak=%.4f sample_rate=%d\n",
              sink_stage->n_samples, sink_stage->peak,
              sink_stage->sample_rate);
  EXPECT_TRUE(sink_stage->n_samples > 1000u);
  EXPECT_TRUE(sink_stage->peak > 0.01);
  EXPECT_TRUE(sink_stage->sample_rate == 24000);

  // Profiling events: the synthesis must have recorded the LLM-lane blocks
  // for the MOSS LM (text-prefill + text-decode) and the codec (audio-codec).
  // dump_profiling emits a synthetic stage entry per aux activity that had
  // events, so the labels appear in the JSON iff the brackets fired.
  const char* td = std::getenv("TMPDIR");
  const std::string prof =
      std::string(td && *td ? td : "/tmp") + "/vpipe_tts_prof.json";
  sess.dump_profiling(prof);
  std::ifstream pf(prof);
  const std::string pj((std::istreambuf_iterator<char>(pf)),
                       std::istreambuf_iterator<char>());
  std::printf("[tts-smoke] profiling dump = %zu bytes -> %s\n",
              pj.size(), prof.c_str());
  EXPECT_TRUE(pj.find("text-prefill") != std::string::npos);
  EXPECT_TRUE(pj.find("text-decode")  != std::string::npos);
  EXPECT_TRUE(pj.find("audio-codec")  != std::string::npos);
}

namespace {

// Emits a fixed list of text beats (one per process tick), then done.
class MultiTextSource : public TypedStage<MultiTextSource> {
public:
  static constexpr const char* kTypeName = "ut-tts-multitext-source";
  using TypedStage::TypedStage;

  std::vector<std::string> texts;

  Job process(RuntimeContext& ctx) override
  {
    if (_i >= texts.size()) { ctx.signal_done(); co_return; }
    const std::string t = texts[_i++];
    co_await ctx.write(0, make_payload<FlexDataPayload>(
        FlexData::make_string(t)));
  }

private:
  std::size_t _i = 0;
};

// Emits a reference PCM beat on oport0, then a text beat on oport1 (same
// tick, reference first), then done. Drives external voice cloning: the TTS
// stage drains its reference iport before synthesising the text.
class RefThenTextSource : public TypedStage<RefThenTextSource> {
public:
  static constexpr const char* kTypeName = "ut-tts-ref-text-source";
  using TypedStage::TypedStage;

  std::string          text = "This is a cloned voice test.";
  std::vector<float>   ref_pcm;
  int                  ref_sr = 16000;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = { static_cast<std::int64_t>(ref_pcm.size()) };
    tb.resize_contiguous(ref_pcm.size());
    std::memcpy(tb.as_f32(), ref_pcm.data(), ref_pcm.size() * sizeof(float));
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert("sample_rate",
                                   FlexData::make_int(ref_sr));
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
    co_await ctx.write(1, make_payload<FlexDataPayload>(
        FlexData::make_string(text)));
  }

private:
  bool _sent = false;
};

}  // namespace

// voice_lock (design-once): the second text beat must reuse the first beat's
// voice as a reference. Exercises the reference-splice prompt path end-to-end
// with real generated codes. Both beats must yield a real waveform.
TEST(text_to_speech_stage, metal_voice_lock_smoke) {
  const char* lm = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* cc = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (!lm || !*lm || !cc || !*cc) { return; }

  Session  sess;
  Pipeline pl("tts-voicelock", &sess);

  auto src = make_unique<MultiTextSource>(
      &sess, "text", vector<InEdge>{}, FlexData::make_object());
  src->texts = { "First sentence picks the voice.",
                 "Second sentence keeps the same voice." };
  src->allocate_oports(1);
  auto* txtsrc = static_cast<MultiTextSource*>(pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(lm));
    o.insert("codec_dir", FlexData::make_string(cc));
    o.insert("max_new_tokens", FlexData::make_int(512));
    o.insert("voice_lock", FlexData::make_bool(true));
    // voice_lock spans multiple beats, so both must generate fully -- opt out
    // of the default barge-in (which would abort the first beat the instant the
    // second is queued, leaving nothing to design the locked voice from).
    o.insert("interrupt_on_new_text", FlexData::make_bool(false));
  }
  auto tts = make_unique<TextToSpeechStage>(
      &sess, "tts", vector<InEdge>{ { txtsrc, 0 } }, std::move(cfg));
  auto* tts_stage = static_cast<TextToSpeechStage*>(
      pl.insert_stage(std::move(tts)));

  auto sink = make_unique<PcmCollectorSink>(
      &sess, "sink", vector<InEdge>{ { tts_stage, 0 } },
      FlexData::make_object());
  auto* sink_stage = static_cast<PcmCollectorSink*>(
      pl.insert_stage(std::move(sink)));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[tts-voicelock] clips=%llu last samples=%zu peak=%.4f\n",
              (unsigned long long)tts_stage->clips_emitted(),
              sink_stage->n_samples, sink_stage->peak);
  EXPECT_TRUE(tts_stage->clips_emitted() == 2u);   // both beats synthesized
  EXPECT_TRUE(sink_stage->n_samples > 1000u);      // 2nd (cloned) is real
  EXPECT_TRUE(sink_stage->peak > 0.01);
}

// External voice cloning: a PCM reference on iport1 is resampled (16k -> 24k),
// encoded to RVQ codes, and spliced into the prompt. The text beat then
// synthesizes with that reference. Verifies the iport -> resample -> encode ->
// splice -> generate chain produces a real waveform.
TEST(text_to_speech_stage, metal_voice_clone_smoke) {
  const char* lm = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* cc = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (!lm || !*lm || !cc || !*cc) { return; }

  Session  sess;
  Pipeline pl("tts-voiceclone", &sess);

  // ~1.2 s of a vaguely voiced reference at 16 kHz (fundamental + harmonics
  // with a slow amplitude envelope) so the encoder has real structure to
  // analyse and the resampler runs (16k -> the codec's 24k).
  const int rsr = 16000;
  const int rn  = rsr * 6 / 5;
  std::vector<float> ref(static_cast<std::size_t>(rn));
  for (int i = 0; i < rn; ++i) {
    const double t = static_cast<double>(i) / rsr;
    const double env = 0.5 * (1.0 - std::cos(2.0 * M_PI * t / 1.2));
    ref[(std::size_t)i] = static_cast<float>(
        env * (0.6 * std::sin(2.0 * M_PI * 130.0 * t) +
               0.3 * std::sin(2.0 * M_PI * 260.0 * t) +
               0.1 * std::sin(2.0 * M_PI * 390.0 * t)));
  }

  auto src = make_unique<RefThenTextSource>(
      &sess, "ref-text", vector<InEdge>{}, FlexData::make_object());
  src->ref_pcm = std::move(ref);
  src->ref_sr  = rsr;
  src->allocate_oports(2);   // oport0 = ref PCM, oport1 = text
  auto* rtsrc = static_cast<RefThenTextSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(lm));
    o.insert("codec_dir", FlexData::make_string(cc));
    o.insert("max_new_tokens", FlexData::make_int(512));
  }
  // iport0 = text (rtsrc oport1), iport1 = audio-ref (rtsrc oport0).
  auto tts = make_unique<TextToSpeechStage>(
      &sess, "tts",
      vector<InEdge>{ { rtsrc, 1 }, { rtsrc, 0 } }, std::move(cfg));
  auto* tts_stage = static_cast<TextToSpeechStage*>(
      pl.insert_stage(std::move(tts)));

  auto sink = make_unique<PcmCollectorSink>(
      &sess, "sink", vector<InEdge>{ { tts_stage, 0 } },
      FlexData::make_object());
  auto* sink_stage = static_cast<PcmCollectorSink*>(
      pl.insert_stage(std::move(sink)));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[tts-voiceclone] clips=%llu samples=%zu peak=%.4f sr=%d\n",
              (unsigned long long)tts_stage->clips_emitted(),
              sink_stage->n_samples, sink_stage->peak,
              sink_stage->sample_rate);
  EXPECT_TRUE(tts_stage->clips_emitted() >= 1u);
  EXPECT_TRUE(sink_stage->n_samples > 1000u);
  EXPECT_TRUE(sink_stage->peak > 0.01);
  EXPECT_TRUE(sink_stage->sample_rate == 24000);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
