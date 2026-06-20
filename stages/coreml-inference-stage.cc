#include "stages/coreml-inference-stage.h"
#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/perf-event.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Build NS::String from std::string (UTF-8). Returned object is
// autoreleased (factory convention); caller does not own +1.
NS::String*
ns_str_(string_view s)
{
  return NS::String::string(string(s).c_str(),
                            NS::UTF8StringEncoding);
}

// Build NS::Array<NSNumber*> from a shape vector. Each element is
// boxed as NS::Number*. Returned array is autoreleased.
NS::Array*
shape_array_(const vector<int64_t>& shape)
{
  vector<const NS::Object*> nums;
  nums.reserve(shape.size());
  for (auto d : shape) {
    nums.push_back(
        static_cast<const NS::Object*>(NS::Number::number(
            static_cast<long long>(d))));
  }
  return NS::Array::array(nums.data(),
                          static_cast<NS::UInteger>(nums.size()));
}

// Copy an NS::Array<NSNumber*> shape into a vector<int64_t>.
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

}

CoreMLInferenceStage::CoreMLInferenceStage(const SessionContextIntf* s,
                                           string                    id,
                                           vector<InEdge>            iports,
                                           FlexData                  config)
  : TypedStage<CoreMLInferenceStage>(s, std::move(id),
                                     std::move(iports),
                                     std::move(config))
{
  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _model_path         = attr_str("model_path");
  _input_feature_name = attr_str("input_feature_name");
  _compute_units      = static_cast<int>(attr_int("compute_units"));
  _uses_cpu_only      = attr_bool("uses_cpu_only");

  // output_feature_names is a composite array (one oport per entry);
  // parsed from the config directly.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("output_feature_names")) {
      auto arr = root.at("output_feature_names");
      if (arr.is_array()) {
        auto av = arr.as_array();
        for (size_t i = 0, n = av.size(); i < n; ++i) {
          _output_feature_names.emplace_back(av.at(i).as_string(""));
        }
      }
    }
  }

  // Validation is deferred to launch (see Stage::fail_config): a stage
  // must construct for any config so a graph can be built/edited first.
  if (_model_path.empty()) {
    fail_config(fmt(
        "CoreMLInferenceStage('{}'): model_path is required",
        this->id()));
  }
  if (_input_feature_name.empty()) {
    fail_config(fmt(
        "CoreMLInferenceStage('{}'): input_feature_name is required",
        this->id()));
  }
  if (_output_feature_names.empty()) {
    fail_config(fmt(
        "CoreMLInferenceStage('{}'): output_feature_names must be a "
        "non-empty array of strings",
        this->id()));
  }

  // One oport per output feature. Previously the stage relied on the
  // oports being allocated externally; declaring them here lets it be
  // used as an edge source through the spec/handle layer.
  allocate_oports(_output_feature_names.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "model_path", .type = ConfigType::String, .required = true,
   .doc = ".mlmodelc dir or .mlmodel file"},
  {.key = "input_feature_name", .type = ConfigType::String,
   .required = true, .doc = "model input feature name"},
  {.key = "output_feature_names", .type = ConfigType::Array,
   .required = true, .doc = "non-empty array of output feature names"},
  {.key = "compute_units", .type = ConfigType::Int,
   .doc = "0=CPUOnly 1=CPU+GPU 2=All 3=CPU+ANE", .def_int = 2},
  {.key = "uses_cpu_only", .type = ConfigType::Bool,
   .doc = "inference-time override of compute_units", .def_bool = false},
};
// One oport per output feature, so the live oport count is dynamic
// (assigned in the ctor from output_feature_names); the spec declares
// no oports. Both ports carry TensorBeats.
const PortSpec kIports[] = {
  {.name = "input", .doc = "model input feature as a TensorBeat",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "coreml-inference",
  .doc       = "Runs a generic CoreML model on a TensorBeat input and "
               "emits one TensorBeat per output feature "
               "(output_feature_names).",
  .display_name = "CoreML Model",
  .category  = StageCategory::Generic,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
CoreMLInferenceStage::spec() const noexcept
{
  return kSpec;
}

CoreMLInferenceStage::~CoreMLInferenceStage()
{
  if (_opts) {
    _opts->release();
    _opts = nullptr;
  }
  // _loaded drops via shared_ptr; the underlying CoreMLLoadedModel
  // releases its retained CML::Model when the last consumer goes
  // away.
}

Job
CoreMLInferenceStage::initialize(RuntimeContext& /*ctx*/)
{
  // Resolve the session-shared CoreML model manager. Failure here
  // means the build is missing CoreML support entirely (non-Apple
  // or VPIPE_BUILD_APPLE_SILICON=OFF) -- not recoverable for an
  // inference stage that was nonetheless instantiated.
  CoreMLModelManager* mgr =
      session() ? session()->coreml_model_manager() : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "CoreMLInferenceStage('{}'): session has no CoreML model "
        "manager (build without VPIPE_BUILD_APPLE_SILICON?)",
        this->id()));
  }

  _loaded = mgr->load(_model_path, _compute_units);
  if (!_loaded) {
    session()->error(fmt(
        "CoreMLInferenceStage('{}'): model load failed for '{}' "
        "(see prior log entries for the underlying error)",
        this->id(), _model_path));
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto* opts = CML::PredictionOptions::alloc()->init();
  opts->setUsesCPUOnly(_uses_cpu_only);
  _opts = opts;  // alloc/init returns +1; no extra retain.
  pool->release();
  co_return;
}

Job
CoreMLInferenceStage::process(RuntimeContext& ctx)
{
  // Read one input tensor, run inference, fan out outputs.
  auto in = co_await ctx.read(0);
  if (!in) {
    ctx.signal_done();
    co_return;
  }

  const TensorBeatPayload* tin =
      dynamic_cast<const TensorBeatPayload*>(in.get());
  if (!tin) {
    session()->error(fmt(
        "coreml-inference: input token is not a TensorBeat"));
    ctx.signal_done();
    co_return;
  }

  // Decide up-front whether every requested output has a fully-
  // specified shape. If yes we take the zero-copy outputBackings
  // path; otherwise we drop into the legacy memcpy path for that
  // call so flexible-shape models still work.
  vector<TensorBeat> outputs;
  bool all_fixed = true;
  for (const auto& name : _output_feature_names) {
    auto it = _loaded->output_descs().find(name);
    if (it == _loaded->output_descs().end() || !it->second.fixed) {
      all_fixed = false;
      break;
    }
  }
  if (all_fixed) {
    outputs.reserve(_output_feature_names.size());
    for (const auto& name : _output_feature_names) {
      const auto& d = _loaded->output_descs().at(name);
      TensorBeat tb;
      tb.shape = d.shape;
      tb.dtype = TensorBeat::DType::F32;  // CoreML output is f32
      size_t n = 1;
      for (auto x : d.shape) {
        n *= static_cast<size_t>(x);
      }
      // tb.data is raw bytes; CoreML writes f32 into this buffer so
      // we size it in bytes = elements * sizeof(float). Previously
      // this was sized to `n` bytes (element-count, not byte-count),
      // which the initWithDataPointer binding would overrun by 3*n
      // bytes -- a latent heap-corruption bug that landed quietly in
      // the malloc bucket's tail padding for small test tensors.
      tb.data.resize(n * sizeof(float));
      outputs.push_back(std::move(tb));
    }
  }

  // Serialise per-model so multiple worker threads can't all stall
  // inside CoreML's framework mutex; observable contention here is
  // strictly cheaper than opaque blocking inside Apple's runtime.
  std::lock_guard<std::mutex> lk(_loaded->predict_mutex());

  // Build the source pointer + strides to hand to MLMultiArray.
  // TensorBeat's PyTorch-style element strides are directly
  // compatible with MLMultiArray (same row-major orientation,
  // larger-than-shape outer strides allowed for pitch padding) so
  // we can publish the buffer zero-copy when the innermost stride
  // is 1. Non-unit innermost stride (a v2+ TensorBeat feature) and
  // any failure path fall back to materialize_contiguous().
  if (tin->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "coreml-inference: unsupported input dtype '{}' "
        "(only f32 is supported) — dropping beat",
        tin->dtype_name()));
    co_return;
  }
  const bool tb_strided = !tin->strides.empty();
  const bool stride_ok  = !tb_strided || tin->strides.back() == 1;
  AlignedVector<float> tmp_in;
  const float*         in_src = tin->as_f32();
  const size_t         in_n   = tin->element_count();
  std::vector<int64_t> in_strides;
  if (stride_ok) {
    in_strides = tb_strided ? tin->strides
                            : tin->contiguous_strides();
  } else {
    tmp_in     = tin->materialize_contiguous_as<float>();
    in_src     = tmp_in.data();
    in_strides = tin->contiguous_strides();
  }

  bool ok = false;
  {
    auto* pool = NS::AutoreleasePool::alloc()->init();

    // ---- Input MultiArray, zero-copy via initWithDataPointer.
    NS::Array* in_shape       = shape_array_(tin->shape);
    NS::Array* in_strides_arr = shape_array_(in_strides);
    NS::Error* err            = nullptr;
    auto* in_multi = CML::MultiArray::alloc()->initWithDataPointer(
        const_cast<float*>(in_src),
        in_shape, CML::MultiArrayDataTypeFloat32,
        in_strides_arr,
        /* deallocator */ nullptr,
        &err);
    if (err || !in_multi) {
      // Surface the NSError, then fall back to alloc + memcpy.
      string desc = "(no NSError details)";
      if (err && err->localizedDescription()) {
        if (const char* utf8 =
                err->localizedDescription()->utf8String()) {
          desc = utf8;
        }
      }
      session()->warn(fmt(
          "coreml-inference: initWithDataPointer failed ({}); "
          "retrying via initWithShape+memcpy", desc));
      err = nullptr;
      const float* contig = in_src;
      if (stride_ok && tb_strided) {
        // The strided source isn't safe for a flat memcpy; build a
        // contiguous copy for the fallback path.
        tmp_in = tin->materialize_contiguous_as<float>();
        contig = tmp_in.data();
      }
      in_multi = CML::MultiArray::alloc()->initWithShape(
          in_shape, CML::MultiArrayDataTypeFloat32, &err);
      if (!err && in_multi) {
        std::memcpy(in_multi->dataPointer(), contig,
                    in_n * sizeof(float));
      }
    }

    if (err || !in_multi) {
      session()->error(fmt(
          "coreml-inference: MLMultiArray init failed"));
    } else {
      auto* fv  = CML::FeatureValue::featureValueWithMultiArray(in_multi);
      auto* key = ns_str_(_input_feature_name);
      const NS::Object* objs[1] = { fv };
      const NS::Object* keys[1] = { key };
      NS::Dictionary* dict = NS::Dictionary::dictionary(objs, keys, 1);

      auto* dfp = CML::DictionaryFeatureProvider::alloc()
                      ->initWithDictionary(dict, &err);
      if (err || !dfp) {
        session()->error(fmt(
            "coreml-inference: feature provider init failed"));
      } else {
        // ---- Output MultiArrays, zero-copy via setOutputBackings
        //      when every requested output has a fixed shape.
        if (all_fixed) {
          vector<const NS::Object*> back_objs;
          vector<const NS::Object*> back_keys;
          back_objs.reserve(outputs.size());
          back_keys.reserve(outputs.size());
          bool back_ok = true;
          for (size_t i = 0; i < outputs.size(); ++i) {
            NS::Array* sh = shape_array_(outputs[i].shape);
            auto* om = CML::MultiArray::alloc()->initWithDataPointer(
                outputs[i].data.data(),
                sh, CML::MultiArrayDataTypeFloat32,
                /* strides */ nullptr,
                /* deallocator */ nullptr,
                &err);
            if (err || !om) {
              back_ok = false;
              break;
            }
            back_objs.push_back(om);
            back_keys.push_back(ns_str_(_output_feature_names[i]));
          }
          if (back_ok) {
            NS::Dictionary* backings = NS::Dictionary::dictionary(
                back_objs.data(), back_keys.data(),
                static_cast<NS::UInteger>(back_objs.size()));
            _opts->setOutputBackings(backings);
          } else {
            // Couldn't bind output backings; fall back to the copy
            // path on this call.
            all_fixed = false;
            outputs.clear();
          }
          // Note: the MultiArray instances created here are
          // autoreleased into `pool`; CoreML retains the entries it
          // pulls out of the dictionary across the predict call.
        }

        // ANE timeline: the CoreML prediction is the Apple-Neural-
        // Engine job. Bracket just the predict call; it records under
        // this stage's gvid so the block is named/colored by the stage.
        record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
        auto* result = _loaded->model()->predictionFromFeatures(
            dfp, _opts, &err);
        record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
        if (err || !result) {
          session()->error(fmt(
              "coreml-inference: predictionFromFeatures failed"));
        } else if (all_fixed) {
          // Outputs are already populated in our TensorBeats.
          ok = true;
        } else {
          // Legacy memcpy path: build outputs from the result
          // FeatureProvider's MultiArrays.
          ok = true;
          outputs.clear();
          for (const auto& name : _output_feature_names) {
            auto* nm  = ns_str_(name);
            auto* val = result->featureValueForName(nm);
            if (!val) {
              ok = false;
              break;
            }
            auto* arr = val->multiArrayValue();
            if (!arr) {
              outputs.emplace_back();
              continue;
            }
            TensorBeat tout;
            tout.shape = read_shape_(arr->shape());
            tout.dtype = TensorBeat::DType::F32;
            size_t n = 1;
            for (auto d : tout.shape) {
              n *= static_cast<size_t>(d);
            }
            // tout.data is raw bytes; CoreML output is float32 so we
            // size in bytes accordingly. Pre-refactor this used a
            // uint8_t vector sized to `n` and the memcpy below wrote
            // 4*n bytes, a quiet overrun that became fatal once
            // TensorStorage shifted the heap layout.
            tout.data.resize(n * sizeof(float));
            std::memcpy(tout.data.data(), arr->dataPointer(),
                        n * sizeof(float));
            outputs.push_back(std::move(tout));
          }
        }
        dfp->release();
      }

      in_multi->release();
    }

    pool->release();
  }

  if (!ok) {
    ctx.signal_done();
    co_return;
  }

  // Fan out: one TensorBeat per output port. If outputs.size() <
  // num_oports we just stop -- alignment is configured statically.
  for (size_t i = 0; i < outputs.size() && i < ctx.num_oports(); ++i) {
    co_await ctx.write(static_cast<unsigned>(i),
        make_payload<TensorBeatPayload>(std::move(outputs[i])));
  }
}

Job
CoreMLInferenceStage::drain(RuntimeContext& /*ctx*/)
{
  co_return;
}

VPIPE_REGISTER_STAGE(CoreMLInferenceStage)
VPIPE_REGISTER_SPEC(CoreMLInferenceStage, kSpec)

}
