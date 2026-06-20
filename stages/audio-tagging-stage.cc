#include "stages/audio-tagging-stage.h"

#include "stages/ced-audioset-labels.h"
#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
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

// NS::String from UTF-8; autoreleased (factory convention).
NS::String*
ns_str_(string_view s)
{
  return NS::String::string(string(s).c_str(),
                            NS::UTF8StringEncoding);
}

// NS::Array<NSNumber*> from a shape vector; autoreleased.
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

// NS::Array<NSNumber*> shape -> vector<int64_t>.
vector<int64_t>
read_shape_(const NS::Array* arr)
{
  vector<int64_t> out;
  if (!arr) {
    return out;
  }
  NS::UInteger n = arr->count();
  out.reserve(n);
  for (NS::UInteger i = 0; i < n; ++i) {
    auto* num = arr->object<NS::Number>(i);
    out.push_back(num ? num->longLongValue() : 0);
  }
  return out;
}

// IEEE-754 binary16 -> binary32. The CED-base export emits its 527
// probabilities as FLOAT16; CoreML hands them back as raw half bytes.
float
half_to_float_(std::uint16_t h)
{
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t       exp  = (h >> 10) & 0x1Fu;
  std::uint32_t       mant = h & 0x3FFu;
  std::uint32_t       f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;                       // +/- zero
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {  // normalise the subnormal
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (mant << 13);   // inf / nan
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
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
  _model_path          = attr_str("model_path");
  _input_feature_name  = attr_str("input_feature_name");
  _output_feature_name = attr_str("output_feature_name");
  _compute_units       = static_cast<int>(attr_int("compute_units"));
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
    double v = attr_real("window_seconds");
    if (v <= 0.0 || v > 60.0) {
      fail_config(fmt(
          "AudioTaggingStage('{}'): window_seconds {} outside "
          "(0, 60]", this->id(), v));
    }
    _window_seconds = v;
  }
  {
    double v = attr_real("hop_seconds");
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
    _top_k = static_cast<int>(std::min<int64_t>(
        v, kCedAudiosetLabelCount));
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
        "(path to the CED-base .mlpackage / .mlmodelc)",
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
   .doc = "CED-base .mlpackage / .mlmodelc dir"},
  {.key = "input_feature_name", .type = ConfigType::String,
   .doc = "model input feature name", .def_str = "waveform"},
  {.key = "output_feature_name", .type = ConfigType::String,
   .doc = "model output feature name", .def_str = "probabilities"},
  {.key = "compute_units", .type = ConfigType::Int,
   .doc = "0=CPUOnly 1=CPU+GPU 2=All 3=CPU+ANE", .def_int = 2},
  {.key = "sample_rate", .type = ConfigType::Int,
   .doc = "expected input rate Hz, [8000,48000]", .def_int = 16000},
  {.key = "window_seconds", .type = ConfigType::Real,
   .doc = "window length s, (0,60]", .def_real = 5.0},
  {.key = "hop_seconds", .type = ConfigType::Real,
   .doc = "advance per run s, (0,window_seconds]", .def_real = 4.0},
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
  .doc       = "Runs the CED-base AudioSet tagger (CoreML) over a sliding "
               "PCM window and emits top-k class tags per window.",
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
  if (_opts) {
    _opts->release();
    _opts = nullptr;
  }
  // _loaded drops via shared_ptr; the CoreMLLoadedModel releases its
  // retained CML::Model when the last consumer goes away.
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
  _loaded = mgr->load(_model_path, _compute_units);
  if (!_loaded) {
    session()->error(fmt(
        "AudioTaggingStage('{}'): model load failed for '{}' (see "
        "prior log entries)", this->id(), _model_path));
  }

  // Number of output classes: from the model's fixed output shape when
  // available, else the embedded label count. Mismatch is non-fatal --
  // label lookup guards out-of-range indices.
  _n_classes = kCedAudiosetLabelCount;
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
  if (_n_classes != kCedAudiosetLabelCount) {
    session()->warn(fmt(
        "AudioTaggingStage('{}'): model reports {} output classes but "
        "{} labels are embedded; indices >= {} will be unnamed",
        this->id(), _n_classes, kCedAudiosetLabelCount,
        kCedAudiosetLabelCount));
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto* opts = CML::PredictionOptions::alloc()->init();
  opts->setUsesCPUOnly(false);
  _opts = opts;  // alloc/init returns +1; no extra retain.
  pool->release();

  session()->info(fmt(
      "AudioTaggingStage('{}'): model ready ({}); window={} samples "
      "({:.1f}s) hop={} samples ({:.1f}s) overlap={:.1f}s, sr={} Hz, "
      "top_k={}, classes={}",
      this->id(), _model_path, _window_samples, _window_seconds,
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
          (best >= 0 && best < kCedAudiosetLabelCount)
              ? kCedAudiosetLabels[best]
              : "?";
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
  if (!_loaded || !_opts) {
    return false;
  }
  // Serialise per-model (CoreML serialises internally anyway; a cheap
  // observable mutex beats opaque blocking inside the framework).
  std::lock_guard<std::mutex> lk(_loaded->predict_mutex());

  bool ok = false;
  auto* pool = NS::AutoreleasePool::alloc()->init();
  NS::Error* err = nullptr;

  const vector<int64_t> in_shape = { 1, _window_samples };
  NS::Array* sh = shape_array_(in_shape);

  // Zero-copy input; fall back to alloc + memcpy if the framework
  // rejects the borrowed pointer.
  auto* in_multi = CML::MultiArray::alloc()->initWithDataPointer(
      const_cast<float*>(samples), sh, CML::MultiArrayDataTypeFloat32,
      /* strides */ nullptr, /* deallocator */ nullptr, &err);
  if (err || !in_multi) {
    err = nullptr;
    in_multi = CML::MultiArray::alloc()->initWithShape(
        sh, CML::MultiArrayDataTypeFloat32, &err);
    if (!err && in_multi) {
      std::memcpy(in_multi->dataPointer(), samples,
                  static_cast<std::size_t>(_window_samples)
                      * sizeof(float));
    }
  }

  if (!err && in_multi) {
    auto* fv  = CML::FeatureValue::featureValueWithMultiArray(in_multi);
    auto* key = ns_str_(_input_feature_name);
    const NS::Object* objs[1] = { fv };
    const NS::Object* keys[1] = { key };
    NS::Dictionary* dict = NS::Dictionary::dictionary(objs, keys, 1);
    auto* dfp = CML::DictionaryFeatureProvider::alloc()
                    ->initWithDictionary(dict, &err);
    if (!err && dfp) {
      // ANE timeline: the CoreML audio-tagging inference is the
      // Apple-Neural-Engine job (recorded under this stage's gvid).
      record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
      auto* result =
          _loaded->model()->predictionFromFeatures(dfp, _opts, &err);
      record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
      if (!err && result) {
        auto* nm  = ns_str_(_output_feature_name);
        auto* val = result->featureValueForName(nm);
        auto* arr = val ? val->multiArrayValue() : nullptr;
        if (arr && arr->dataPointer()) {
          std::size_t n = 1;
          for (auto d : read_shape_(arr->shape())) {
            n *= static_cast<std::size_t>(d);
          }
          probs_out.resize(n);
          const CML::MultiArrayDataType dt = arr->dataType();
          if (dt == CML::MultiArrayDataTypeFloat16) {
            const auto* h =
                static_cast<const std::uint16_t*>(arr->dataPointer());
            for (std::size_t i = 0; i < n; ++i) {
              probs_out[i] = half_to_float_(h[i]);
            }
            ok = true;
          } else if (dt == CML::MultiArrayDataTypeFloat32) {
            const auto* f =
                static_cast<const float*>(arr->dataPointer());
            std::copy(f, f + n, probs_out.begin());
            ok = true;
          } else {
            session()->warn(fmt(
                "AudioTaggingStage('{}'): unsupported output dtype "
                "{}; dropping window",
                this->id(), static_cast<long>(dt)));
          }
        } else {
          session()->warn(fmt(
              "AudioTaggingStage('{}'): output '{}' missing from "
              "prediction", this->id(), _output_feature_name));
        }
      } else {
        session()->warn(fmt(
            "AudioTaggingStage('{}'): predictionFromFeatures failed",
            this->id()));
      }
      dfp->release();
    } else {
      session()->warn(fmt(
          "AudioTaggingStage('{}'): feature provider init failed",
          this->id()));
    }
    in_multi->release();
  } else {
    session()->warn(fmt(
        "AudioTaggingStage('{}'): input MLMultiArray init failed",
        this->id()));
  }

  pool->release();
  return ok;
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
    const char* name = (idx >= 0 && idx < kCedAudiosetLabelCount)
                           ? kCedAudiosetLabels[idx]
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
