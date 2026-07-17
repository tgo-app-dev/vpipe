#include "stages/audio-segment-stage.h"

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
// (The f16 prob output is decoded to f32 inside
// CoreMLLoadedModel::predict, so the old host half-to-float helper and
// the NS/CoreML feature marshaling are gone.)

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
  // _loaded drops via shared_ptr; predict() owns all CoreML lifetime.
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
  if (!_loaded) { return false; }

  // Sample-rate feature value must outlive the predict() call.
  const std::int32_t sr_val = static_cast<std::int32_t>(_sample_rate);

  // Inputs: PCM (zero-copy), the two LSTM states, and (optionally) the
  // int32 sample rate. predict() binds each as a model input feature.
  std::vector<CoreMLPredictInput> ins;
  ins.reserve(4);
  auto add_marray = [&](const std::string& name, const void* data,
                        CoreMLDType dtype, std::vector<int64_t> shape) {
    CoreMLPredictInput a;
    a.name  = name;
    a.data  = data;
    a.dtype = dtype;
    a.shape = std::move(shape);
    ins.push_back(std::move(a));
  };
  add_marray(_input_feature_name, samples, CoreMLDType::F32,
             { 1, static_cast<int64_t>(_window_samples) });
  add_marray(_state_h_in_name, _state_h.data(), CoreMLDType::F32,
             _state_h_shape);
  add_marray(_state_c_in_name, _state_c.data(), CoreMLDType::F32,
             _state_c_shape);
  if (!_sr_feature_name.empty()) {
    add_marray(_sr_feature_name, &sr_val, CoreMLDType::I32, { 1 });
  }

  // Outputs: the speech probability plus the updated LSTM states (all
  // delivered as f32; a model that emits f16 prob is decoded inside
  // predict()).
  std::vector<CoreMLPredictOutput> outs;
  outs.reserve(3);
  auto add_out = [&](const std::string& name) {
    CoreMLPredictOutput o;
    o.name = name;
    o.want = CoreMLDType::F32;
    outs.push_back(std::move(o));
  };
  add_out(_prob_feature_name);
  add_out(_state_h_out_name);
  add_out(_state_c_out_name);

  // ANE timeline: the per-window Silero VAD CoreML predict is the
  // Apple-Neural-Engine job (recorded under this stage's gvid).
  record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
  const bool ok = _loaded->predict(ins, outs);
  record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
  if (!ok) {
    session()->warn(fmt(
        "AudioSegmentStage('{}'): prediction failed", this->id()));
    return false;
  }

  const float* pf = static_cast<const float*>(outs[0].data);
  if (!pf) {
    session()->warn(fmt(
        "AudioSegmentStage('{}'): prob output '{}' missing",
        this->id(), _prob_feature_name));
    return false;
  }
  prob_out = pf[0];

  // Feed the updated LSTM states back, clamped to our buffer sizes.
  auto copy_state = [](std::vector<float>&        dst,
                       const CoreMLPredictOutput& o) {
    const auto* f = static_cast<const float*>(o.data);
    if (!f) { return; }
    const size_t n = std::min(dst.size(), product_(o.shape));
    std::memcpy(dst.data(), f, n * sizeof(float));
  };
  copy_state(_state_h, outs[1]);
  copy_state(_state_c, outs[2]);
  return true;
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
