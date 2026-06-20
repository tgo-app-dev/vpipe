#include "stages/audio-segment-stage.h"

#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

NS::String*
ns_str_(string_view s)
{
  return NS::String::string(string(s).c_str(),
                            NS::UTF8StringEncoding);
}

NS::Array*
shape_array_(const vector<int64_t>& shape)
{
  vector<const NS::Object*> nums;
  nums.reserve(shape.size());
  for (auto d : shape) {
    nums.push_back(static_cast<const NS::Object*>(
        NS::Number::number(static_cast<long long>(d))));
  }
  return NS::Array::array(nums.data(),
                          static_cast<NS::UInteger>(nums.size()));
}

vector<int64_t>
read_shape_(const NS::Array* arr)
{
  vector<int64_t> out;
  if (!arr) { return out; }
  NS::UInteger n = arr->count();
  out.reserve(n);
  for (NS::UInteger i = 0; i < n; ++i) {
    auto* num = arr->object<NS::Number>(i);
    out.push_back(num ? num->longLongValue() : 0);
  }
  return out;
}

// Read a config array of integers into a vector<int64_t>. Returns true
// when the FlexData node is a non-empty array of integer-like values.
bool
read_int_array_(const FlexData& node, vector<int64_t>* out)
{
  if (!node.is_array()) { return false; }
  auto a = node.as_array();
  if (a.size() == 0) { return false; }
  out->clear();
  out->reserve(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out->push_back(a.at(i).as_int(0));
  }
  return true;
}

size_t
product_(const vector<int64_t>& shape)
{
  size_t n = 1;
  for (auto d : shape) { n *= static_cast<size_t>(std::max<int64_t>(d, 0)); }
  return n;
}

// IEEE-754 binary16 -> binary32 (some Silero CoreML exports emit prob
// as Float16). Same helper as audio-tagging-stage.cc.
float
half_to_float_(std::uint16_t h)
{
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t       exp  = (h >> 10) & 0x1Fu;
  std::uint32_t       mant = h & 0x3FFu;
  std::uint32_t       f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

}  // namespace

AudioSegmentStage::AudioSegmentStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<AudioSegmentStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  allocate_oports(1);

  // Scalar/string attributes: defaults live in kConfigKeys (kSpec.attrs);
  // attr_* resolves the configured value or that default. Clamps / cross-
  // checks reject out-of-range overrides via fail_config -- deferred-
  // validated, so the ctor never throws and the runtime skips a failed
  // stage at launch.
  _model_path    = attr_str("model_path");
  _models_db     = attr_str("models_db");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  _compute_units = static_cast<int>(attr_int("compute_units"));
  {
    const std::int64_t v = attr_int("sample_rate");
    if (v < 8000 || v > 48000) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): sample_rate {} outside [8000, 48000]",
          this->id(), v));
    }
    _sample_rate = static_cast<int>(v);
  }
  // window/hop are interdependent: a "pure window" export has no carryover,
  // so when the user sets a custom window_samples WITHOUT an explicit
  // hop_samples we mirror hop <- window (overlap 0). The unified-v6 default
  // (576/512, overlap 64) applies only when neither is set. attr_* can't tell
  // a configured value from the spec default, so probe the raw config object
  // for these two keys (bind the owning config before as_object() -- a view).
  bool window_set = false, hop_set = false;
  {
    const FlexData& cfg = this->config();
    if (cfg.is_object()) {
      auto root = cfg.as_object();
      window_set = root.contains("window_samples");
      hop_set    = root.contains("hop_samples");
    }
  }
  {
    const std::int64_t v = attr_int("window_samples");
    if (v < 32 || v > (1 << 20)) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): window_samples {} outside "
          "[32, 1048576]", this->id(), v));
    }
    _window_samples = static_cast<int>(v);
  }
  {
    std::int64_t v = attr_int("hop_samples");
    if (window_set && !hop_set) {
      v = _window_samples;   // pure-window export -> no overlap
    }
    if (v < 1 || v > _window_samples) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): hop_samples {} outside "
          "[1, window_samples={}]", this->id(), v, _window_samples));
    }
    _hop_samples = static_cast<int>(v);
  }
  {
    const double v = attr_real("speech_threshold");
    if (v <= 0.0 || v >= 1.0) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): speech_threshold {} outside (0, 1)",
          this->id(), v));
    }
    _speech_threshold = v;
  }
  {
    const double v = attr_real("silence_threshold");
    if (v <= 0.0 || v >= 1.0) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): silence_threshold {} outside (0, 1)",
          this->id(), v));
    }
    _silence_threshold = v;
  }
  if (_silence_threshold >= _speech_threshold) {
    fail_config(fmt(
        "AudioSegmentStage('{}'): silence_threshold ({}) must be < "
        "speech_threshold ({})",
        this->id(), _silence_threshold, _speech_threshold));
  }
  {
    const std::int64_t v = attr_int("min_speech_ms");
    if (v < 0) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): min_speech_ms {} < 0", this->id(), v));
    }
    _min_speech_ms = static_cast<int>(v);
  }
  {
    const std::int64_t v = attr_int("min_silence_ms");
    if (v < 0) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): min_silence_ms {} < 0", this->id(), v));
    }
    _min_silence_ms = static_cast<int>(v);
  }
  {
    const double v = attr_real("max_segment_s");
    if (v <= 0.0 || v > 120.0) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): max_segment_s {} outside (0, 120]",
          this->id(), v));
    }
    _max_segment_s = v;
  }
  {
    const std::int64_t v = attr_int("pre_pad_ms");
    if (v < 0 || v > 5000) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): pre_pad_ms {} outside [0, 5000]",
          this->id(), v));
    }
    _pre_pad_ms = static_cast<int>(v);
  }
  {
    const std::int64_t v = attr_int("post_pad_ms");
    if (v < 0 || v > 5000) {
      fail_config(fmt(
          "AudioSegmentStage('{}'): post_pad_ms {} outside [0, 5000]",
          this->id(), v));
    }
    _post_pad_ms = static_cast<int>(v);
  }
  _input_feature_name = attr_str("input_feature_name");
  _prob_feature_name  = attr_str("prob_feature_name");
  _state_h_in_name    = attr_str("state_h_in_name");
  _state_c_in_name    = attr_str("state_c_in_name");
  _state_h_out_name   = attr_str("state_h_out_name");
  _state_c_out_name   = attr_str("state_c_out_name");
  _sr_feature_name    = attr_str("sr_feature_name");

  // The LSTM state shapes are arrays (no flat attr_* accessor): read them
  // off the config object when present, else the unified-v6 member defaults
  // stand. (as_object() returns a view -- bind the owning config first.)
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("state_h_shape")) {
      vector<int64_t> tmp;
      if (read_int_array_(root.at("state_h_shape"), &tmp)) {
        _state_h_shape = std::move(tmp);
      }
    }
    if (root.contains("state_c_shape")) {
      vector<int64_t> tmp;
      if (read_int_array_(root.at("state_c_shape"), &tmp)) {
        _state_c_shape = std::move(tmp);
      }
    }
  }

  if (_model_path.empty()) {
    fail_config(fmt(
        "AudioSegmentStage('{}'): config.model_path is required",
        this->id()));
  }
  _context_overlap_samples = _window_samples - _hop_samples;
  // Reserve room for a window + one hop's worth of look-ahead.
  _buf.reserve(static_cast<size_t>(_window_samples + _hop_samples));
}

namespace {
constexpr ConfigKey kConfigKeys[] = {
  {.key = "model_path", .type = ConfigType::String, .required = true,
   .doc = "Silero VAD model: a models-DB key (registered by model-fetch) "
          "or a .mlpackage / .mlmodelc dir; a DB key wins over a same-named "
          "path",
   .suggest_db = "models", .suggest_db_type = "silero-vad"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "compute_units", .type = ConfigType::Int,
   .doc = "0=CPUOnly 1=CPU+GPU 2=All 3=CPU+ANE", .def_int = 2},
  {.key = "sample_rate", .type = ConfigType::Int,
   .doc = "expected input rate Hz, [8000,48000]", .def_int = 16000},
  {.key = "window_samples", .type = ConfigType::Int,
   .doc = "per-inference input size, [32,1048576]", .def_int = 576},
  {.key = "hop_samples", .type = ConfigType::Int,
   .doc = "advance per inference, [1,window_samples]", .def_int = 512},
  {.key = "speech_threshold", .type = ConfigType::Real,
   .doc = "VAD prob >= this enters candidate-speech", .def_real = 0.5},
  {.key = "silence_threshold", .type = ConfigType::Real,
   .doc = "VAD prob < this enters candidate-silence (< speech)",
   .def_real = 0.35},
  {.key = "min_speech_ms", .type = ConfigType::Int,
   .doc = "candidate-speech run before opening a segment, ms",
   .def_int = 250},
  {.key = "min_silence_ms", .type = ConfigType::Int,
   .doc = "candidate-silence run before closing a segment, ms",
   .def_int = 400},
  {.key = "max_segment_s", .type = ConfigType::Real,
   .doc = "force-close (partial) above this duration, (0,120]",
   .def_real = 12.0},
  {.key = "pre_pad_ms", .type = ConfigType::Int,
   .doc = "pull the segment start back by this many ms",
   .def_int = 100},
  {.key = "post_pad_ms", .type = ConfigType::Int,
   .doc = "push the segment end forward by this many ms",
   .def_int = 200},
  {.key = "input_feature_name", .type = ConfigType::String,
   .doc = "PCM input feature name", .def_str = "audio_input"},
  {.key = "prob_feature_name", .type = ConfigType::String,
   .doc = "VAD probability output feature name",
   .def_str = "vad_output"},
  {.key = "state_h_in_name", .type = ConfigType::String,
   .doc = "LSTM hidden-state input name", .def_str = "hidden_state"},
  {.key = "state_c_in_name", .type = ConfigType::String,
   .doc = "LSTM cell-state input name", .def_str = "cell_state"},
  {.key = "state_h_out_name", .type = ConfigType::String,
   .doc = "next-step LSTM hidden output", .def_str = "new_hidden_state"},
  {.key = "state_c_out_name", .type = ConfigType::String,
   .doc = "next-step LSTM cell output", .def_str = "new_cell_state"},
  {.key = "sr_feature_name", .type = ConfigType::String,
   .doc = "int32 sample-rate input name; empty disables",
   .def_str = ""},
  {.key = "state_h_shape", .type = ConfigType::Array,
   .doc = "LSTM hidden tensor shape (e.g. [1,128] or [2,1,64])"},
  {.key = "state_c_shape", .type = ConfigType::Array,
   .doc = "LSTM cell tensor shape (e.g. [1,128] or [2,1,64])"},
};
const PortSpec kIports[] = {
  {.name = "audio", .doc = "mono F32 PCM TensorBeat [N] or [1,N]; "
                           "sideband.timestamp_us drives segment times",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "segments", .doc = "FlexData per closed utterance: "
                              "{start_us, end_us, index, is_partial}; "
                              "timestamp-only marker, no PCM",
   .type = &typeid(FlexDataPayload), .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "audio-segment",
  .doc       = "Silero-VAD speech segmenter: window/hop VAD over a mono PCM "
               "stream + a hysteresis FSM, emitting [start_us,end_us) "
               "utterance markers for a downstream slicer (e.g. streaming "
               "audio-transcribe). No PCM flows on the oport.",
  .display_name = "Audio Segment (VAD)",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kConfigKeys,
};
}  // namespace

const StageSpec&
AudioSegmentStage::spec() const noexcept
{
  return kSpec;
}

AudioSegmentStage::~AudioSegmentStage()
{
  if (_opts) {
    _opts->release();
    _opts = nullptr;
  }
}

void
AudioSegmentStage::reset_lstm_state_()
{
  _state_h.assign(product_(_state_h_shape), 0.0f);
  _state_c.assign(product_(_state_c_shape), 0.0f);
}

Job
AudioSegmentStage::initialize(RuntimeContext& /*ctx*/)
{
  CoreMLModelManager* mgr =
      session() ? session()->coreml_model_manager() : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "AudioSegmentStage('{}'): session has no CoreML model manager "
        "(build without VPIPE_BUILD_APPLE_SILICON?)", this->id()));
    co_return;
  }
  // A models-DB key (e.g. a model-fetch'd silero-vad model) resolves to
  // its unpacked .mlpackage; a plain path passes through unchanged.
  _model_path = resolve_model_dir(session(), _models_db, _model_path);
  _loaded = mgr->load(_model_path, _compute_units);
  if (!_loaded) {
    session()->error(fmt(
        "AudioSegmentStage('{}'): model load failed for '{}'",
        this->id(), _model_path));
    co_return;
  }

  reset_lstm_state_();

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto* opts = CML::PredictionOptions::alloc()->init();
  opts->setUsesCPUOnly(false);
  _opts = opts;
  pool->release();

  session()->info(fmt(
      "AudioSegmentStage('{}'): model ready ({}); window={} samples "
      "({:.1f} ms) hop={} samples ({:.1f} ms) overlap={} samples, "
      "sr={} Hz, speech/silence={}/{}, min_speech={}ms min_silence={}ms",
      this->id(), _model_path, _window_samples,
      1000.0 * _window_samples / _sample_rate,
      _hop_samples,
      1000.0 * _hop_samples / _sample_rate,
      _context_overlap_samples,
      _sample_rate, _speech_threshold, _silence_threshold,
      _min_speech_ms, _min_silence_ms));
  co_return;
}

Job
AudioSegmentStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    ctx.signal_done();
    co_return;
  }
  if (!_loaded) {
    co_return;  // initialize() errored.
  }

  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (!tbp) {
    session()->warn(fmt(
        "AudioSegmentStage('{}'): expected TensorBeatPayload on iport "
        "0, got {}; dropping beat", this->id(), p->describe()));
    co_return;
  }
  if (tbp->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "AudioSegmentStage('{}'): expected f32 PCM, got {}; dropping "
        "beat", this->id(), TensorBeat::name_of(tbp->dtype)));
    co_return;
  }
  // Accept rank-1 [N] or rank-2 [1, N].
  size_t n_samples = 0;
  if (tbp->shape.size() == 1) {
    n_samples = static_cast<size_t>(tbp->shape[0]);
  } else if (tbp->shape.size() == 2 && tbp->shape[0] == 1) {
    n_samples = static_cast<size_t>(tbp->shape[1]);
  } else {
    session()->warn(fmt(
        "AudioSegmentStage('{}'): expected shape [N] or [1,N], got "
        "rank={}; dropping beat",
        this->id(), static_cast<int>(tbp->shape.size())));
    co_return;
  }
  if (n_samples == 0) {
    co_return;
  }

  int beat_sr = _sample_rate;
  std::uint64_t beat_ts_us = 0;
  bool beat_has_ts = false;
  if (tbp->sideband.is_object()) {
    auto sb = tbp->sideband.as_object();
    if (sb.contains("sample_rate")) {
      beat_sr = static_cast<int>(
          sb.at("sample_rate").as_int(_sample_rate));
    }
    if (sb.contains("timestamp_us")) {
      beat_ts_us = sb.at("timestamp_us").as_uint(0);
      beat_has_ts = true;
    }
  }
  if (beat_sr != _sample_rate) {
    if (!_sr_warned) {
      session()->warn(fmt(
          "AudioSegmentStage('{}'): input sample_rate {} != expected "
          "{}; the model is fixed-rate, dropping mismatched beats",
          this->id(), beat_sr, _sample_rate));
      _sr_warned = true;
    }
    co_return;
  }

  // Start-of-stream prep. When _context_overlap_samples > 0, prepend
  // that many zeros so the first inference's "new audio" frame starts
  // at the true first sample.
  if (!_stream_started) {
    const std::uint64_t overlap_us =
        static_cast<std::uint64_t>(_context_overlap_samples)
        * 1'000'000ULL / static_cast<std::uint64_t>(_sample_rate);
    _have_base_ts = beat_has_ts;
    if (beat_has_ts) {
      _buf_base_ts_us =
          beat_ts_us > overlap_us ? beat_ts_us - overlap_us : 0;
    } else {
      _buf_base_ts_us = 0;
    }
    if (_context_overlap_samples > 0) {
      _buf.assign(static_cast<size_t>(_context_overlap_samples), 0.0f);
    } else {
      _buf.clear();
    }
    _stream_started = true;
  }

  // Append the new samples.
  {
    AlignedVector<float> tmp = tbp->materialize_contiguous_as<float>();
    _buf.insert(_buf.end(), tmp.data(), tmp.data() + n_samples);
  }

  // Slide the window: run while we have a full window. Advance by hop
  // after each inference.
  while (static_cast<int>(_buf.size()) >= _window_samples) {
    float prob = 0.0f;
    const bool ok = run_window_(_buf.data(), prob);
    if (ok) {
      ++_frames_run;
      const std::uint64_t overlap_us =
          static_cast<std::uint64_t>(_context_overlap_samples)
          * 1'000'000ULL / static_cast<std::uint64_t>(_sample_rate);
      const std::uint64_t window_us =
          static_cast<std::uint64_t>(_window_samples) * 1'000'000ULL
          / static_cast<std::uint64_t>(_sample_rate);
      const std::uint64_t frame_start =
          _buf_base_ts_us + overlap_us;
      const std::uint64_t frame_end =
          _buf_base_ts_us + window_us;
      _last_frame_end_us = frame_end;
      co_await advance_fsm_(ctx, prob, frame_start, frame_end);
    }
    const int adv = std::min(_hop_samples, static_cast<int>(_buf.size()));
    _buf.erase(_buf.begin(), _buf.begin() + adv);
    _buf_base_ts_us += static_cast<std::uint64_t>(adv) * 1'000'000ULL
                       / static_cast<std::uint64_t>(_sample_rate);
  }
  co_return;
}

Job
AudioSegmentStage::drain(RuntimeContext& ctx)
{
  if (_state == State::Speech && _last_frame_end_us >= _segment_start_us) {
    co_await emit_segment_(ctx, _segment_start_us,
                           _last_frame_end_us, /*is_partial=*/true);
    _state = State::Silence;
    _candidate_run_us = 0;
    _candidate_start_us = 0;
  }
  co_return;
}

bool
AudioSegmentStage::run_window_(const float* samples, float& prob_out)
{
  if (!_loaded || !_opts) { return false; }
  std::lock_guard<std::mutex> lk(_loaded->predict_mutex());

  bool ok = false;
  auto* pool = NS::AutoreleasePool::alloc()->init();
  NS::Error* err = nullptr;

  const vector<int64_t> in_shape = { 1, _window_samples };
  NS::Array* sh = shape_array_(in_shape);

  // 1) Input PCM MultiArray (zero-copy; alloc+memcpy fallback).
  auto* in_multi = CML::MultiArray::alloc()->initWithDataPointer(
      const_cast<float*>(samples), sh, CML::MultiArrayDataTypeFloat32,
      /*strides*/ nullptr, /*deallocator*/ nullptr, &err);
  if (err || !in_multi) {
    err = nullptr;
    in_multi = CML::MultiArray::alloc()->initWithShape(
        sh, CML::MultiArrayDataTypeFloat32, &err);
    if (!err && in_multi) {
      std::memcpy(in_multi->dataPointer(), samples,
                  static_cast<size_t>(_window_samples) * sizeof(float));
    }
  }

  // 2) LSTM state inputs (always alloc+memcpy from _state_h/_state_c).
  CML::MultiArray* h_multi = nullptr;
  CML::MultiArray* c_multi = nullptr;
  if (!err && in_multi) {
    NS::Array* hsh = shape_array_(_state_h_shape);
    h_multi = CML::MultiArray::alloc()->initWithShape(
        hsh, CML::MultiArrayDataTypeFloat32, &err);
    if (!err && h_multi) {
      std::memcpy(h_multi->dataPointer(), _state_h.data(),
                  _state_h.size() * sizeof(float));
    }
  }
  if (!err && h_multi) {
    NS::Array* csh = shape_array_(_state_c_shape);
    c_multi = CML::MultiArray::alloc()->initWithShape(
        csh, CML::MultiArrayDataTypeFloat32, &err);
    if (!err && c_multi) {
      std::memcpy(c_multi->dataPointer(), _state_c.data(),
                  _state_c.size() * sizeof(float));
    }
  }

  // 3) Optional int32 sample-rate input.
  CML::MultiArray* sr_multi = nullptr;
  if (!err && c_multi && !_sr_feature_name.empty()) {
    const vector<int64_t> sr_shape = { 1 };
    NS::Array* ssh = shape_array_(sr_shape);
    sr_multi = CML::MultiArray::alloc()->initWithShape(
        ssh, CML::MultiArrayDataTypeInt32, &err);
    if (!err && sr_multi) {
      std::int32_t v = static_cast<std::int32_t>(_sample_rate);
      std::memcpy(sr_multi->dataPointer(), &v, sizeof(v));
    }
  }

  // 4) Build the feature provider and run.
  if (!err && c_multi) {
    vector<const NS::Object*> objs;
    vector<const NS::Object*> keys;
    objs.reserve(4);
    keys.reserve(4);
    auto add_feat = [&](const string& name, CML::MultiArray* arr) {
      auto* fv  = CML::FeatureValue::featureValueWithMultiArray(arr);
      auto* key = ns_str_(name);
      objs.push_back(fv);
      keys.push_back(key);
    };
    add_feat(_input_feature_name, in_multi);
    add_feat(_state_h_in_name,    h_multi);
    add_feat(_state_c_in_name,    c_multi);
    if (sr_multi) { add_feat(_sr_feature_name, sr_multi); }

    NS::Dictionary* dict = NS::Dictionary::dictionary(
        objs.data(), keys.data(),
        static_cast<NS::UInteger>(objs.size()));
    auto* dfp = CML::DictionaryFeatureProvider::alloc()
                    ->initWithDictionary(dict, &err);
    if (!err && dfp) {
      // ANE timeline: the per-window Silero VAD CoreML predict is the
      // Apple-Neural-Engine job (recorded under this stage's gvid so the
      // block is named/colored as audio-segment in the profiler).
      record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
      auto* result =
          _loaded->model()->predictionFromFeatures(dfp, _opts, &err);
      record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
      if (!err && result) {
        // Prob output (accept f16 or f32; flatten to first element).
        auto* nm  = ns_str_(_prob_feature_name);
        auto* val = result->featureValueForName(nm);
        auto* arr = val ? val->multiArrayValue() : nullptr;
        if (arr && arr->dataPointer()) {
          const CML::MultiArrayDataType dt = arr->dataType();
          if (dt == CML::MultiArrayDataTypeFloat32) {
            prob_out = *static_cast<const float*>(arr->dataPointer());
            ok = true;
          } else if (dt == CML::MultiArrayDataTypeFloat16) {
            prob_out = half_to_float_(
                *static_cast<const std::uint16_t*>(arr->dataPointer()));
            ok = true;
          } else {
            session()->warn(fmt(
                "AudioSegmentStage('{}'): unsupported prob dtype {}; "
                "dropping window",
                this->id(), static_cast<long>(dt)));
          }
        } else {
          session()->warn(fmt(
              "AudioSegmentStage('{}'): prob output '{}' missing",
              this->id(), _prob_feature_name));
        }
        // Updated LSTM state outputs.
        if (ok) {
          auto* hnm = ns_str_(_state_h_out_name);
          auto* hv  = result->featureValueForName(hnm);
          auto* harr = hv ? hv->multiArrayValue() : nullptr;
          if (harr && harr->dataPointer()
              && harr->dataType() == CML::MultiArrayDataTypeFloat32) {
            const auto n = std::min(_state_h.size(),
                                    product_(read_shape_(harr->shape())));
            std::memcpy(_state_h.data(), harr->dataPointer(),
                        n * sizeof(float));
          }
          auto* cnm = ns_str_(_state_c_out_name);
          auto* cv  = result->featureValueForName(cnm);
          auto* carr = cv ? cv->multiArrayValue() : nullptr;
          if (carr && carr->dataPointer()
              && carr->dataType() == CML::MultiArrayDataTypeFloat32) {
            const auto n = std::min(_state_c.size(),
                                    product_(read_shape_(carr->shape())));
            std::memcpy(_state_c.data(), carr->dataPointer(),
                        n * sizeof(float));
          }
        }
      } else {
        session()->warn(fmt(
            "AudioSegmentStage('{}'): predictionFromFeatures failed",
            this->id()));
      }
      dfp->release();
    } else {
      session()->warn(fmt(
          "AudioSegmentStage('{}'): feature provider init failed",
          this->id()));
    }
  } else {
    session()->warn(fmt(
        "AudioSegmentStage('{}'): input/state MultiArray init failed",
        this->id()));
  }

  if (sr_multi) { sr_multi->release(); }
  if (c_multi)  { c_multi->release(); }
  if (h_multi)  { h_multi->release(); }
  if (in_multi) { in_multi->release(); }
  pool->release();
  return ok;
}

Job
AudioSegmentStage::emit_segment_(RuntimeContext& ctx,
                                 std::uint64_t   start_us,
                                 std::uint64_t   end_us,
                                 bool            is_partial)
{
  // Timestamp-only speech-segment marker as FlexData -- consistent with the
  // sibling audio-tagging stage's {timestamp_us, tags} output. No PCM flows
  // here; downstream slices its own synchronous PCM stream using [start_us,
  // end_us). Keys: start_us/end_us (UTC us), index (monotonic, for diag/dedup),
  // is_partial (segment force-closed at max_segment_s or flushed at EOS).
  FlexData fd = FlexData::make_object();
  auto obj = fd.as_object();
  obj.insert_or_assign("start_us",   FlexData::make_uint(start_us));
  obj.insert_or_assign("end_us",     FlexData::make_uint(end_us));
  obj.insert_or_assign("index",      FlexData::make_uint(_segments_emitted));
  obj.insert_or_assign("is_partial", FlexData::make_bool(is_partial));
  session()->log_debug(fmt(
      "AudioSegmentStage('{}'): segment #{} [{}..{}] dur={:.2f}s{}",
      this->id(), _segments_emitted, start_us, end_us,
      (end_us - start_us) / 1.0e6, is_partial ? " (partial)" : ""));
  co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  ++_segments_emitted;
}

Job
AudioSegmentStage::advance_fsm_(RuntimeContext& ctx,
                                float           prob,
                                std::uint64_t   frame_start_us,
                                std::uint64_t   frame_end_us)
{
  const std::uint64_t hop_us =
      static_cast<std::uint64_t>(_hop_samples) * 1'000'000ULL
      / static_cast<std::uint64_t>(_sample_rate);
  const std::uint64_t min_speech_us =
      static_cast<std::uint64_t>(_min_speech_ms) * 1000ULL;
  const std::uint64_t min_silence_us =
      static_cast<std::uint64_t>(_min_silence_ms) * 1000ULL;
  const std::uint64_t pre_pad_us =
      static_cast<std::uint64_t>(_pre_pad_ms) * 1000ULL;
  const std::uint64_t post_pad_us =
      static_cast<std::uint64_t>(_post_pad_ms) * 1000ULL;
  const std::uint64_t max_seg_us =
      static_cast<std::uint64_t>(_max_segment_s * 1.0e6);

  if (_state == State::Silence) {
    if (prob >= static_cast<float>(_speech_threshold)) {
      if (_candidate_run_us == 0) {
        _candidate_start_us = frame_start_us;
      }
      _candidate_run_us += hop_us;
      if (_candidate_run_us >= min_speech_us) {
        _segment_start_us =
            _candidate_start_us > pre_pad_us
              ? _candidate_start_us - pre_pad_us
              : 0;
        _state              = State::Speech;
        _candidate_run_us   = 0;
        _candidate_start_us = 0;
      }
    } else {
      _candidate_run_us   = 0;
      _candidate_start_us = 0;
    }
  } else {  // State::Speech
    // Always check max-segment force-close first.
    if (frame_end_us > _segment_start_us
        && frame_end_us - _segment_start_us >= max_seg_us) {
      co_await emit_segment_(ctx, _segment_start_us, frame_end_us,
                             /*is_partial=*/true);
      _state              = State::Silence;
      _candidate_run_us   = 0;
      _candidate_start_us = 0;
      co_return;
    }
    if (prob < static_cast<float>(_silence_threshold)) {
      if (_candidate_run_us == 0) {
        _candidate_start_us = frame_start_us;
      }
      _candidate_run_us += hop_us;
      if (_candidate_run_us >= min_silence_us) {
        const std::uint64_t end =
            _candidate_start_us + post_pad_us;
        co_await emit_segment_(ctx, _segment_start_us, end,
                               /*is_partial=*/false);
        _state              = State::Silence;
        _candidate_run_us   = 0;
        _candidate_start_us = 0;
      }
    } else {
      _candidate_run_us   = 0;
      _candidate_start_us = 0;
    }
  }
}

VPIPE_REGISTER_STAGE(AudioSegmentStage)
// Without this the type registers (so the editor LISTS it) but its spec
// does not, so StageRegistry::spec("audio-segment") is null and the web-ui
// composer falls back to a generic entry: category "generic", 0 iports,
// 0 oports. Mirrors every other spec'd stage (audio-transcribe, etc.).
VPIPE_REGISTER_SPEC(AudioSegmentStage, kSpec)

}
