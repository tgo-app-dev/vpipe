// AudioTaggingStage tests.
//
// The config-parsing tests verify the ctor + config plumbing (they do
// not load the model -- that happens in initialize()). The end-to-end
// model I/O path is exercised by an env-gated smoke test:
//
//     VPIPE_TEST_CED_MODEL=/path/to/ced-base-5s-fp16.mlpackage
//
// When the env var is unset the smoke test passes trivially.

#include "minitest.h"

#include "stages/audio-tagging-stage.h"
#include "stages/beats-audioset-labels.h"
#include "stages/ced-audioset-labels.h"
#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "pipeline/stage-registry.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
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

const char* env_or_null_(const char* name)
{
  const char* v = std::getenv(name);
  return (v && *v) ? v : nullptr;
}

FlexData
cfg_with_model_(const char* path)
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("model_path",
                                   FlexData::make_string(path));
  return cfg;
}

}

TEST(audio_tagging_stage, type_is_registered) {
  EXPECT_TRUE(string_view(AudioTaggingStage::kTypeName)
              == "audio-tagging");
  auto& reg = StageRegistry::get();
  EXPECT_TRUE(reg.find_id("audio-tagging") != StageTypeId::unknown);
}

// Defaults: BEATs is the default family -> 10 s / 8 s (2 s overlap)
// cadence. Only model_path is required. The input/output feature names
// are derived from the model at initialize(), so they are empty here
// (no model loaded) -- not asserted.
TEST(audio_tagging_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{},
                      cfg_with_model_("/tmp/tagger.mlpackage"));
  EXPECT_TRUE(s.model_path() == "/tmp/tagger.mlpackage");
  EXPECT_TRUE(s.model_kind()  == "beats");    // default family
  EXPECT_TRUE(s.sample_rate()    == 16000);
  EXPECT_TRUE(s.window_samples() == 160000);  // 10 s @ 16 kHz
  EXPECT_TRUE(s.hop_samples()    == 128000);  //  8 s @ 16 kHz
  EXPECT_TRUE(s.top_k()          == 5);
  EXPECT_TRUE(s.label_count()    == 527);
  EXPECT_TRUE(s.score_threshold() == 0.0);
  EXPECT_TRUE(s.num_oports()     == 1u);
  EXPECT_TRUE(s.windows_emitted() == 0u);
}

// model_kind="ced" selects the legacy 5 s / 4 s (1 s overlap) cadence.
TEST(audio_tagging_stage, ced_config_defaults) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","model_kind":"ced"})");
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.model_kind()  == "ced");
  EXPECT_TRUE(s.window_samples() == 80000);   // 5 s @ 16 kHz
  EXPECT_TRUE(s.hop_samples()    == 64000);   // 4 s @ 16 kHz
  EXPECT_TRUE(s.label_count()    == 527);
}

// Explicit shape overrides still win over the model_kind profile.
TEST(audio_tagging_stage, beats_config_overrides) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","model_kind":"beats",
          "window_seconds":10.0,"hop_seconds":5.0})");
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.window_samples() == 160000);  // 10 s @ 16 kHz
  EXPECT_TRUE(s.hop_samples()    == 80000);   //  5 s @ 16 kHz (override)
}

// An unrecognised model_kind is a deferred config error.
TEST(audio_tagging_stage, unknown_model_kind_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","model_kind":"wav2vec"})");
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// The model_path field is associated with the "audio-tagging" model
// registry type, so an editor suggests model-fetch'd taggers (the BEATs
// supplement model) and a configured models-DB key resolves to it.
TEST(audio_tagging_stage, model_field_associated_with_registry_type) {
  Session sess;
  CerrSilencer hush;
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{},
                      cfg_with_model_("/p"));
  const ConfigKey* mp = nullptr;
  for (const ConfigKey& k : s.spec().attrs) {
    if (k.key == "model_path") { mp = &k; break; }
  }
  ASSERT_TRUE(mp != nullptr);
  EXPECT_TRUE(mp->suggest_db == "models");
  EXPECT_TRUE(mp->suggest_db_type == "audio-tagging");
}

TEST(audio_tagging_stage, config_overrides) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p",
          "compute_units":1,
          "sample_rate":16000,
          "window_seconds":5.0,
          "hop_seconds":2.5,
          "top_k":10,
          "score_threshold":0.3})");
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.model_path() == "/p");
  EXPECT_TRUE(s.window_samples() == 80000);
  EXPECT_TRUE(s.hop_samples()    == 40000);   // 2.5 s @ 16 kHz
  EXPECT_TRUE(s.top_k()          == 10);
  EXPECT_TRUE(std::abs(s.score_threshold() - 0.3) < 1e-9);
}

// Construction succeeds with any config; a missing/invalid field is
// recorded in config_error() and deferred to launch.
TEST(audio_tagging_stage, missing_model_path_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::make_object();
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_tagging_stage, hop_exceeds_window_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","window_seconds":5.0,"hop_seconds":6.0})");
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_tagging_stage, bad_sample_rate_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"model_path":"/p","sample_rate":100})");
  AudioTaggingStage s(&sess, "tag", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(audio_tagging_stage, embedded_labels_present) {
  // The embedded AudioSet table must hold all 527 names with the
  // known anchors at both ends.
  EXPECT_TRUE(kCedAudiosetLabelCount == 527);
  EXPECT_TRUE(string_view(kCedAudiosetLabels[0]) == "Speech");
  EXPECT_TRUE(string_view(kCedAudiosetLabels[526]) == "Field recording");
  // Comma-containing names must be intact (sourced from
  // class_labels_indices.csv, not the comma-truncated config.json).
  EXPECT_TRUE(string_view(kCedAudiosetLabels[1])
              == "Male speech, man speaking");
  EXPECT_TRUE(string_view(kCedAudiosetLabels[109])
              == "Roaring cats (lions, tigers)");
}

TEST(audio_tagging_stage, beats_embedded_labels_present) {
  // BEATs ships its own 527-name table in the model's (permuted)
  // output order -- NOT the AudioSet CSV order CED uses. Index 0 is
  // "Snake", not "Speech"; verify the anchors that prove the ordering
  // is the BEATs label_dict composition, not a copy of the CED table.
  EXPECT_TRUE(kBeatsAudiosetLabelCount == 527);
  EXPECT_TRUE(string_view(kBeatsAudiosetLabels[0])   == "Snake");
  EXPECT_TRUE(string_view(kBeatsAudiosetLabels[2])   == "Music");
  EXPECT_TRUE(string_view(kBeatsAudiosetLabels[526]) == "Sniff");
  // The two tables must genuinely differ at index 0 (else a wrong
  // pairing would silently mislabel every BEATs tag).
  EXPECT_FALSE(string_view(kBeatsAudiosetLabels[0])
               == string_view(kCedAudiosetLabels[0]));
}

// Env-gated: load the real CED-base model through the session-shared
// CoreMLModelManager and run one 80000-sample window, exercising the
// exact input-binding + f16 output-decode path the stage uses. Asserts
// the model returns 527 finite probabilities in [0, 1] and logs the
// top-5 tags. Passes trivially when VPIPE_TEST_CED_MODEL is unset.
TEST(audio_tagging_stage, real_model_smoke) {
  const char* model = env_or_null_("VPIPE_TEST_CED_MODEL");
  if (!model) { return; }

  Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto loaded = mgr->load(model, /* compute_units */ 2);
  ASSERT_TRUE(loaded != nullptr);

  constexpr int kSamples = 80000;
  std::vector<float> wav(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    wav[i] = 0.2f * std::sin(2.0 * M_PI * 440.0 * i / 16000.0);
  }

  std::vector<float> probs;
  {
    std::lock_guard<std::mutex> lk(loaded->predict_mutex());
    auto* pool = NS::AutoreleasePool::alloc()->init();
    NS::Error* err = nullptr;

    const NS::Object* dims[2] = {
        NS::Number::number(static_cast<long long>(1)),
        NS::Number::number(static_cast<long long>(kSamples)) };
    NS::Array* shape = NS::Array::array(dims, 2);
    auto* in = CML::MultiArray::alloc()->initWithShape(
        shape, CML::MultiArrayDataTypeFloat32, &err);
    ASSERT_TRUE(!err && in != nullptr);
    std::memcpy(in->dataPointer(), wav.data(),
                kSamples * sizeof(float));

    auto* fv = CML::FeatureValue::featureValueWithMultiArray(in);
    auto* key = NS::String::string("waveform", NS::UTF8StringEncoding);
    const NS::Object* objs[1] = { fv };
    const NS::Object* keys[1] = { key };
    NS::Dictionary* dict = NS::Dictionary::dictionary(objs, keys, 1);
    auto* dfp = CML::DictionaryFeatureProvider::alloc()
                    ->initWithDictionary(dict, &err);
    ASSERT_TRUE(!err && dfp != nullptr);

    auto* result =
        loaded->model()->predictionFromFeatures(dfp, nullptr, &err);
    ASSERT_TRUE(!err && result != nullptr);
    auto* out = result->featureValueForName(
        NS::String::string("probabilities", NS::UTF8StringEncoding));
    ASSERT_TRUE(out != nullptr);
    auto* arr = out->multiArrayValue();
    ASSERT_TRUE(arr != nullptr);

    NS::Integer n = 1;
    for (NS::UInteger d = 0; d < arr->shape()->count(); ++d) {
      n *= arr->shape()->object<NS::Number>(d)->longLongValue();
    }
    probs.resize(static_cast<std::size_t>(n));
    for (NS::Integer i = 0; i < n; ++i) {
      probs[static_cast<std::size_t>(i)] =
          arr->objectAtIndexedSubscript(i)->floatValue();
    }
    dfp->release();
    in->release();
    pool->release();
  }

  EXPECT_TRUE(probs.size() == 527u);
  int best = 0;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    EXPECT_TRUE(std::isfinite(probs[i]));
    EXPECT_TRUE(probs[i] >= -1e-3f && probs[i] <= 1.0f + 1e-3f);
    if (probs[i] > probs[static_cast<std::size_t>(best)]) {
      best = static_cast<int>(i);
    }
  }
  sess.info(vpipe::fmt(
      "audio_tagging_stage.real_model_smoke: top class [{}] '{}' "
      "score={:.3f}", best,
      best < kCedAudiosetLabelCount ? kCedAudiosetLabels[best]
                                    : "unknown",
      probs[static_cast<std::size_t>(best)]));
}

// Env-gated: load the real BEATs model and run one 160000-sample
// (10 s @ 16 kHz) window through the "waveform" input, decoding the
// "probs" output. Asserts 527 finite probabilities in [0, 1] and logs
// the top class via the BEATs label table. A 440 Hz sine should land
// on a tonal class. Passes trivially when VPIPE_TEST_BEATS_MODEL is
// unset.
TEST(audio_tagging_stage, beats_real_model_smoke) {
  const char* model = env_or_null_("VPIPE_TEST_BEATS_MODEL");
  if (!model) { return; }

  Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto loaded = mgr->load(model, /* compute_units */ 2);
  ASSERT_TRUE(loaded != nullptr);

  // The stage derives feature names from these accessors; BEATs is a
  // single-input/single-output model named waveform -> probs.
  ASSERT_TRUE(loaded->input_names().size()  == 1u);
  ASSERT_TRUE(loaded->output_names().size() == 1u);
  EXPECT_TRUE(loaded->input_names()[0]  == "waveform");
  EXPECT_TRUE(loaded->output_names()[0] == "probs");

  constexpr int kSamples = 160000;
  std::vector<float> wav(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    wav[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / 16000.0);
  }

  std::vector<float> probs;
  {
    std::lock_guard<std::mutex> lk(loaded->predict_mutex());
    auto* pool = NS::AutoreleasePool::alloc()->init();
    NS::Error* err = nullptr;

    const NS::Object* dims[2] = {
        NS::Number::number(static_cast<long long>(1)),
        NS::Number::number(static_cast<long long>(kSamples)) };
    NS::Array* shape = NS::Array::array(dims, 2);
    auto* in = CML::MultiArray::alloc()->initWithShape(
        shape, CML::MultiArrayDataTypeFloat32, &err);
    ASSERT_TRUE(!err && in != nullptr);
    std::memcpy(in->dataPointer(), wav.data(),
                kSamples * sizeof(float));

    auto* fv = CML::FeatureValue::featureValueWithMultiArray(in);
    auto* key = NS::String::string("waveform", NS::UTF8StringEncoding);
    const NS::Object* objs[1] = { fv };
    const NS::Object* keys[1] = { key };
    NS::Dictionary* dict = NS::Dictionary::dictionary(objs, keys, 1);
    auto* dfp = CML::DictionaryFeatureProvider::alloc()
                    ->initWithDictionary(dict, &err);
    ASSERT_TRUE(!err && dfp != nullptr);

    auto* result =
        loaded->model()->predictionFromFeatures(dfp, nullptr, &err);
    ASSERT_TRUE(!err && result != nullptr);
    auto* out = result->featureValueForName(
        NS::String::string("probs", NS::UTF8StringEncoding));
    ASSERT_TRUE(out != nullptr);
    auto* arr = out->multiArrayValue();
    ASSERT_TRUE(arr != nullptr);

    NS::Integer n = 1;
    for (NS::UInteger d = 0; d < arr->shape()->count(); ++d) {
      n *= arr->shape()->object<NS::Number>(d)->longLongValue();
    }
    probs.resize(static_cast<std::size_t>(n));
    for (NS::Integer i = 0; i < n; ++i) {
      probs[static_cast<std::size_t>(i)] =
          arr->objectAtIndexedSubscript(i)->floatValue();
    }
    dfp->release();
    in->release();
    pool->release();
  }

  EXPECT_TRUE(probs.size() == 527u);
  int best = 0;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    EXPECT_TRUE(std::isfinite(probs[i]));
    EXPECT_TRUE(probs[i] >= -1e-3f && probs[i] <= 1.0f + 1e-3f);
    if (probs[i] > probs[static_cast<std::size_t>(best)]) {
      best = static_cast<int>(i);
    }
  }
  sess.info(vpipe::fmt(
      "audio_tagging_stage.beats_real_model_smoke: top class [{}] '{}' "
      "score={:.3f}", best,
      best < kBeatsAudiosetLabelCount ? kBeatsAudiosetLabels[best]
                                      : "unknown",
      probs[static_cast<std::size_t>(best)]));
}
