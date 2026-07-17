#include "stages/audio-tagging-stage.h"

#include "stages/beats-audioset-labels.h"
#include "stages/ced-audioset-labels.h"
#include "stages/model-registry.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// (CoreML feature marshaling now lives in CoreMLLoadedModel::predict;
// this stage builds neutral CoreMLPredictInput/Output only. The f16
// output is decoded to f32 inside predict(), so the old host
// half-to-float helper is gone.)

// Per-model-kind window/hop defaults + label table. `model_kind`
// selects one; the window/hop are overridable by explicit config. Both
// tables hold 527 AudioSet names but in DIFFERENT orders (the BEATs
// output order is permuted vs the CSV/CED order), so the table must
// match the model -- a wrong pairing silently mislabels every tag. The
// input/output feature names are NOT here: they are read from the
// CoreML model at load time (single-I/O models, no ambiguity).
struct ModelProfile {
  const char*        kind;
  double             window_seconds;
  double             hop_seconds;
  const char* const* labels;
  int                label_count;
};
const ModelProfile kBeatsProfile = {
    "beats", 10.0, 8.0, kBeatsAudiosetLabels, kBeatsAudiosetLabelCount };
const ModelProfile kCedProfile = {
    "ced", 5.0, 4.0, kCedAudiosetLabels, kCedAudiosetLabelCount };

// Resolve a model_kind string to its profile; nullptr if unknown.
const ModelProfile*
profile_for_(string_view kind)
{
  if (kind == "beats") { return &kBeatsProfile; }
  if (kind == "ced")   { return &kCedProfile; }
  return nullptr;
}

}

AudioTaggingStage::AudioTaggingStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<AudioTaggingStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  // Declare the single output port before anything reads num_oports()
  // (see AudioToPcmStage's note: a 0-oport write dereferences past an
  // empty _out_bufs and crashes inside the WriteAwaiter setup).
  allocate_oports(spec().oports.size());

  // Validation is deferred to launch (see Stage::fail_config): a stage
  // must construct for any config so a graph can be built/edited first.
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  //
  // Resolve the model family first: it selects the label table and the
  // window/hop defaults used below when the config leaves them unset.
  _model_kind = attr_str("model_kind");
  const ModelProfile* prof = profile_for_(_model_kind);
  if (!prof) {
    fail_config(fmt(
        "AudioTaggingStage('{}'): model_kind '{}' unknown (expected "
        "\"beats\" or \"ced\")", this->id(), _model_kind));
    prof = &kBeatsProfile;  // keep a valid table for the rest of the ctor
  }
  _labels      = prof->labels;
  _label_count = prof->label_count;

  // True iff `key` was explicitly provided in this stage's config (vs.
  // defaulted). The window/hop fall back to the model_kind profile --
  // not the generic schema default -- when left unset.
  const FlexData& cfg = this->config();
  auto explicit_ = [&](const char* key) {
    return cfg.is_object() && cfg.as_object().contains(key);
  };

  _model_path    = attr_str("model_path");
  _models_db     = attr_str("models_db");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  _compute_units = static_cast<int>(attr_int("compute_units"));
  {
    int64_t v = attr_int("sample_rate");
    if (v < 8000 || v > 48000) {
      fail_config(fmt(
          "AudioTaggingStage('{}'): sample_rate {} outside "
          "[8000, 48000]", this->id(), v));
    }
    _sample_rate = static_cast<int>(v);
  }
  {
    double v = explicit_("window_seconds") ? attr_real("window_seconds")
                                           : prof->window_seconds;
    if (v <= 0.0 || v > 60.0) {
      fail_config(fmt(
          "AudioTaggingStage('{}'): window_seconds {} outside "
          "(0, 60]", this->id(), v));
    }
    _window_seconds = v;
  }
  {
    double v = explicit_("hop_seconds") ? attr_real("hop_seconds")
                                        : prof->hop_seconds;
    if (v <= 0.0 || v > _window_seconds) {
      fail_config(fmt(
          "AudioTaggingStage('{}'): hop_seconds {} outside "
          "(0, window_seconds={}]", this->id(), v, _window_seconds));
    }
    _hop_seconds = v;
  }
  {
    int64_t v = attr_int("top_k");
    if (v < 1) {
      fail_config(fmt(
          "AudioTaggingStage('{}'): top_k must be >= 1 (got {})",
          this->id(), v));
    }
    _top_k = static_cast<int>(std::min<int64_t>(v, _label_count));
  }
  {
    double v = attr_real("score_threshold");
    if (v < 0.0 || v > 1.0) {
      fail_config(fmt(
          "AudioTaggingStage('{}'): score_threshold {} outside "
          "[0, 1]", this->id(), v));
    }
    _score_threshold = v;
  }

  if (_model_path.empty()) {
    fail_config(fmt(
        "AudioTaggingStage('{}'): config.model_path is required "
        "(path to the AudioSet tagger .mlpackage / .mlmodelc)",
        this->id()));
  }

  _window_samples = static_cast<int>(
      std::lround(_window_seconds * _sample_rate));
  _hop_samples = static_cast<int>(
      std::lround(_hop_seconds * _sample_rate));
  if (_window_samples < 1 || _hop_samples < 1) {
    fail_config(fmt(
        "AudioTaggingStage('{}'): degenerate window/hop sample counts "
        "(window={}, hop={})",
        this->id(), _window_samples, _hop_samples));
  }
  // Reserve room for one window plus a hop's worth of look-ahead so a
  // burst of input doesn't reallocate mid-stream.
  _buf.reserve(static_cast<std::size_t>(_window_samples + _hop_samples));
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "model_path", .type = ConfigType::String, .required = true,
   .doc = "AudioSet tagger: a models-DB key (registered by model-fetch, "
          "e.g. the BEATs supplement model) or a .mlpackage / .mlmodelc "
          "dir; a DB key wins over a same-named path",
   .suggest_db = "models", .suggest_db_type = "audio-tagging"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "model_kind", .type = ConfigType::String,
   .doc = "model family: \"beats\" | \"ced\" (sets label table + "
          "shape defaults)", .def_str = "beats"},
  {.key = "compute_units", .type = ConfigType::Int,
   .doc = "0=CPUOnly 1=CPU+GPU 2=All 3=CPU+ANE", .def_int = 2},
  {.key = "sample_rate", .type = ConfigType::Int,
   .doc = "expected input rate Hz, [8000,48000]", .def_int = 16000},
  {.key = "window_seconds", .type = ConfigType::Real,
   .doc = "window length s, (0,60] (beats:10 ced:5)", .def_real = 10.0},
  {.key = "hop_seconds", .type = ConfigType::Real,
   .doc = "advance per run s, (0,window] (beats:8 ced:4)",
   .def_real = 8.0},
  {.key = "top_k", .type = ConfigType::Int,
   .doc = "tags per window, >= 1", .def_int = 5},
  {.key = "score_threshold", .type = ConfigType::Real,
   .doc = "drop tags below this prob, [0,1]", .def_real = 0.0},
};
// iport and oport share the stage's clock group (0): the cross-cadence
// (many input beats per emitted window) is absorbed by the internal
// sliding buffer, not declared as a clock-domain split.
const PortSpec kIports[] = {
  {.name = "pcm", .doc = "mono F32 16 kHz PCM TensorBeat [N] or [1,N]; "
                         "sideband timestamp_us/sample_rate honoured",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "tags", .doc = "FlexData per window: top_k AudioSet tags "
                          "{label,index,score} + window metadata",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "audio-tagging",
  .doc       = "Runs an AudioSet tagger (CoreML; CED-base or BEATs, per "
               "model_kind) over a sliding PCM window and emits top-k "
               "class tags per window.",
  .display_name = "Audio Tagging",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
AudioTaggingStage::spec() const noexcept
{
  return kSpec;
}

AudioTaggingStage::~AudioTaggingStage()
{
  // _loaded drops via shared_ptr; the CoreMLLoadedModel releases its
  // retained model when the last consumer goes away.
}

Job
AudioTaggingStage::initialize(RuntimeContext& /*ctx*/)
{
  CoreMLModelManager* mgr =
      session() ? session()->coreml_model_manager() : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "AudioTaggingStage('{}'): session has no CoreML model manager "
        "(build without VPIPE_BUILD_APPLE_SILICON?)", this->id()));
  }
  // A models-DB key (e.g. the model-fetch'd BEATs supplement model)
  // resolves to its unpacked .mlpackage; a plain path passes through.
  _model_path = resolve_model_dir(session(), _models_db, _model_path);
  _loaded = mgr->load(_model_path, _compute_units);
  if (!_loaded) {
    session()->error(fmt(
        "AudioTaggingStage('{}'): model load failed for '{}' (see "
        "prior log entries)", this->id(), _model_path));
  }

  // Derive the input/output feature names from the model itself: these
  // taggers have exactly one input and one output, so there is no
  // ambiguity and nothing to configure.
  const auto& in_names  = _loaded->input_names();
  const auto& out_names = _loaded->output_names();
  if (in_names.size() != 1 || out_names.size() != 1) {
    session()->error(fmt(
        "AudioTaggingStage('{}'): expected exactly one input and one "
        "output feature, but model '{}' has {} input(s) / {} output(s)",
        this->id(), _model_path, in_names.size(), out_names.size()));
  }
  _input_feature_name  = in_names.empty()  ? std::string() : in_names[0];
  _output_feature_name = out_names.empty() ? std::string() : out_names[0];

  // Number of output classes: from the model's fixed output shape when
  // available, else the embedded label count. Mismatch is non-fatal --
  // label lookup guards out-of-range indices.
  _n_classes = _label_count;
  auto it = _loaded->output_descs().find(_output_feature_name);
  if (it != _loaded->output_descs().end() && it->second.fixed) {
    int64_t prod = 1;
    for (auto d : it->second.shape) {
      prod *= d;
    }
    if (prod > 0) {
      _n_classes = static_cast<int>(prod);
    }
  }
  if (_n_classes != _label_count) {
    session()->warn(fmt(
        "AudioTaggingStage('{}'): model reports {} output classes but "
        "{} labels are embedded ({}); indices >= {} will be unnamed",
        this->id(), _n_classes, _label_count, _model_kind,
        _label_count));
  }

  session()->info(fmt(
      "AudioTaggingStage('{}'): model ready (kind={}, {}); in='{}' "
      "out='{}' window={} samples ({:.1f}s) hop={} samples ({:.1f}s) "
      "overlap={:.1f}s, sr={} Hz, top_k={}, classes={}",
      this->id(), _model_kind, _model_path, _input_feature_name,
      _output_feature_name, _window_samples, _window_seconds,
      _hop_samples, _hop_seconds,
      _window_seconds - _hop_seconds, _sample_rate, _top_k,
      _n_classes));
  co_return;
}

Job
AudioTaggingStage::process(RuntimeContext& ctx)
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
        "AudioTaggingStage('{}'): expected TensorBeatPayload on iport "
        "0, got {}; dropping beat", this->id(), p->describe()));
    co_return;
  }
  if (tbp->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "AudioTaggingStage('{}'): expected f32 PCM, got {}; dropping "
        "beat", this->id(), TensorBeat::name_of(tbp->dtype)));
    co_return;
  }
  const std::size_t n_samples = tbp->element_count();
  if (n_samples == 0) {
    co_return;
  }

  // Rate must match the model. Sideband override wins; otherwise the
  // configured default applies.
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
          "AudioTaggingStage('{}'): input sample_rate {} != expected "
          "{}; the model is fixed-rate, dropping mismatched beats",
          this->id(), beat_sr, _sample_rate));
      _sr_warned = true;
    }
    co_return;
  }

  // Append. If the buffer was empty, this beat's first sample becomes
  // _buf[0]; capture its timestamp as the window-time base.
  if (_buf.empty()) {
    _buf_base_ts_us = beat_has_ts ? beat_ts_us : 0;
    _have_base_ts   = beat_has_ts;
  }
  {
    AlignedVector<float> tmp = tbp->materialize_contiguous_as<float>();
    _buf.insert(_buf.end(), tmp.data(), tmp.data() + n_samples);
  }

  // Slide the window: run while a full window is buffered, advancing
  // by hop each time (window - hop samples overlap into the next run).
  while (static_cast<int>(_buf.size()) >= _window_samples) {
    std::vector<float> probs;
    if (run_window_(_buf.data(), probs)) {
      // Per-window log of the dominant class so the audio path is
      // visible at the standard "info" level.
      int best = 0;
      for (std::size_t c = 1; c < probs.size(); ++c) {
        if (probs[c] > probs[static_cast<std::size_t>(best)]) {
          best = static_cast<int>(c);
        }
      }
      const char* best_label =
          (best >= 0 && best < _label_count) ? _labels[best] : "?";
      session()->info(fmt(
          "AudioTaggingStage('{}'): window {} ts_us={} -> '{}' "
          "({:.2f})",
          this->id(), _windows_emitted, _buf_base_ts_us, best_label,
          probs.empty() ? 0.0f : probs[static_cast<std::size_t>(best)]));
      FlexData fd = build_tags_(probs, _buf_base_ts_us);
      auto payload = make_payload<FlexDataPayload>(std::move(fd));
      co_await ctx.write(0, std::move(payload));
      ++_windows_emitted;
    }
    int adv = std::min(_hop_samples, static_cast<int>(_buf.size()));
    _buf.erase(_buf.begin(), _buf.begin() + adv);
    _buf_base_ts_us += static_cast<std::uint64_t>(adv) * 1'000'000ULL
                       / static_cast<std::uint64_t>(_sample_rate);
  }
  co_return;
}

Job
AudioTaggingStage::drain(RuntimeContext& /*ctx*/)
{
  // The model needs a full window; a sub-window tail at EOS can't be
  // run, so there's nothing to flush.
  co_return;
}

bool
AudioTaggingStage::run_window_(const float*        samples,
                               std::vector<float>& probs_out)
{
  if (!_loaded) {
    return false;
  }

  // Zero-copy f32 input [1, window]; the model's f16 output (CED) is
  // decoded to f32 inside predict().
  CoreMLPredictInput in;
  in.name  = _input_feature_name;
  in.data  = samples;
  in.dtype = CoreMLDType::F32;
  in.shape = { 1, static_cast<int64_t>(_window_samples) };

  CoreMLPredictOutput out;
  out.name = _output_feature_name;
  out.want = CoreMLDType::F32;

  const CoreMLPredictInput ins[1]  = { std::move(in) };
  CoreMLPredictOutput      outs[1] = { std::move(out) };

  // ANE timeline: the CoreML audio-tagging inference is the Apple-
  // Neural-Engine job (recorded under this stage's gvid).
  record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
  const bool ok = _loaded->predict(ins, outs);
  record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
  if (!ok) {
    session()->warn(fmt(
        "AudioTaggingStage('{}'): prediction failed", this->id()));
    return false;
  }

  const auto* f = static_cast<const float*>(outs[0].data);
  if (!f) {
    return false;
  }
  std::size_t n = 1;
  for (auto d : outs[0].shape) {
    n *= static_cast<std::size_t>(d);
  }
  probs_out.assign(f, f + n);
  return true;
}

FlexData
AudioTaggingStage::build_tags_(const std::vector<float>& probs,
                               std::uint64_t             ts_us) const
{
  FlexData root = FlexData::make_object();
  auto obj = root.as_object();
  obj.insert_or_assign("timestamp_us", FlexData::make_uint(ts_us));
  const std::uint64_t dur_us =
      static_cast<std::uint64_t>(_window_samples) * 1'000'000ULL
      / static_cast<std::uint64_t>(_sample_rate);
  obj.insert_or_assign("duration_us", FlexData::make_uint(dur_us));
  obj.insert_or_assign("sample_rate",
                       FlexData::make_int(_sample_rate));
  obj.insert_or_assign("window_index",
                       FlexData::make_uint(_windows_emitted));

  const int n = static_cast<int>(probs.size());
  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  const int k = std::min(_top_k, n);
  if (k > 0) {
    std::partial_sort(
        order.begin(), order.begin() + k, order.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });
  }

  FlexData tags = FlexData::make_array();
  auto arr = tags.as_array();
  for (int r = 0; r < k; ++r) {
    const int idx = order[static_cast<std::size_t>(r)];
    if (probs[idx] < _score_threshold) {
      break;  // sorted desc -- the rest are below threshold too
    }
    FlexData t = FlexData::make_object();
    auto to = t.as_object();
    const char* name = (idx >= 0 && idx < _label_count)
                           ? _labels[idx]
                           : "unknown";
    to.insert_or_assign("label", FlexData::make_string(name));
    to.insert_or_assign("index", FlexData::make_int(idx));
    to.insert_or_assign("score",
                        FlexData::make_real(probs[idx]));
    arr.push_back(std::move(t));
  }
  obj.insert_or_assign("tags", std::move(tags));
  return root;
}

VPIPE_REGISTER_STAGE(AudioTaggingStage)
VPIPE_REGISTER_SPEC(AudioTaggingStage, kSpec)

}
