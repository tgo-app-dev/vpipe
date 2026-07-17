#include "stages/coreml-inference-stage.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/perf-event.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

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
  // _loaded drops via shared_ptr; the underlying CoreMLLoadedModel
  // releases its retained model when the last consumer goes away.
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

  if (tin->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "coreml-inference: unsupported input dtype '{}' "
        "(only f32 is supported) — dropping beat",
        tin->dtype_name()));
    co_return;
  }

  // Build the input pointer + strides for the model input feature.
  // TensorBeat's PyTorch-style element strides are directly compatible
  // with the MLMultiArray predict() builds (same row-major orientation,
  // larger-than-shape outer strides allowed) so the buffer is published
  // zero-copy when the innermost stride is 1; a non-unit innermost
  // stride is materialized contiguous first.
  const bool tb_strided = !tin->strides.empty();
  const bool stride_ok  = !tb_strided || tin->strides.back() == 1;
  AlignedVector<float> tmp_in;
  const float*         in_src = tin->as_f32();
  std::vector<int64_t> in_strides;
  if (stride_ok) {
    in_strides = tb_strided ? tin->strides : tin->contiguous_strides();
  } else {
    tmp_in     = tin->materialize_contiguous_as<float>();
    in_src     = tmp_in.data();
    in_strides = tin->contiguous_strides();
  }

  CoreMLPredictInput cin;
  cin.name    = _input_feature_name;
  cin.data    = in_src;
  cin.dtype   = CoreMLDType::F32;
  cin.shape   = tin->shape;
  cin.strides = in_strides;

  // One output per configured feature. A fixed-shape output's
  // TensorBeat is pre-allocated and handed to predict() as a zero-copy
  // backing; flexible-shape outputs are filled from predict()'s owned
  // buffer afterwards.
  const size_t                     no = _output_feature_names.size();
  std::vector<TensorBeat>          obeats(no);
  std::vector<CoreMLPredictOutput> couts(no);
  for (size_t i = 0; i < no; ++i) {
    couts[i].name = _output_feature_names[i];
    couts[i].want = CoreMLDType::F32;   // CoreML output decoded to f32
    auto it = _loaded->output_descs().find(_output_feature_names[i]);
    if (it != _loaded->output_descs().end() && it->second.fixed) {
      size_t n = 1;
      for (auto d : it->second.shape) {
        n *= static_cast<size_t>(d);
      }
      obeats[i].dtype = TensorBeat::DType::F32;
      obeats[i].shape = it->second.shape;
      obeats[i].resize_contiguous(n);
      couts[i].backing       = obeats[i].as_f32();
      couts[i].backing_elems = n;
    }
  }

  const CoreMLPredictInput cins[1] = { std::move(cin) };
  // ANE timeline: the CoreML prediction is the Apple-Neural-Engine job.
  record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
  const bool ok = _loaded->predict(cins, couts, _uses_cpu_only);
  record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
  if (!ok) {
    session()->error(fmt("coreml-inference: prediction failed"));
    ctx.signal_done();
    co_return;
  }

  // Fixed outputs are filled in place (backing); flexible ones are
  // copied from predict()'s owned buffer.
  for (size_t i = 0; i < no; ++i) {
    if (couts[i].backing) {
      continue;   // obeats[i] already holds the result
    }
    size_t n = 1;
    for (auto d : couts[i].shape) {
      n *= static_cast<size_t>(d);
    }
    obeats[i].dtype = TensorBeat::DType::F32;
    obeats[i].shape = couts[i].shape;
    obeats[i].resize_contiguous(n);
    if (couts[i].data && n) {
      std::memcpy(obeats[i].as_f32(), couts[i].data, n * sizeof(float));
    }
  }

  // Fan out: one TensorBeat per output port. If no < num_oports we just
  // stop -- alignment is configured statically.
  for (size_t i = 0; i < no && i < ctx.num_oports(); ++i) {
    co_await ctx.write(static_cast<unsigned>(i),
        make_payload<TensorBeatPayload>(std::move(obeats[i])));
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
