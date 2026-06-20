#ifndef VPIPE_APPLE_SILICON_COREML_INFERENCE_STAGE_H
#define VPIPE_APPLE_SILICON_COREML_INFERENCE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <memory>
#include <string>
#include <vector>

namespace CML {
class PredictionOptions;
}

namespace vpipe {

class CoreMLLoadedModel;

// Runs CoreML inference on TensorBeat inputs (apple-silicon/
// tensor-beat.h) and emits TensorBeat outputs.
//
// One input port (the input tensor); one output port per output
// feature in `output_feature_names`.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   model_path             (string, required)  -- .mlmodelc dir or
//                                                 .mlmodel file. The
//                                                 latter is compiled
//                                                 to a tmp .mlmodelc
//                                                 in initialize().
//   input_feature_name     (string, required)
//   output_feature_names   (array<string>, required, non-empty)
//   compute_units          (int, default 2)
//                                       0=CPUOnly, 1=CPU+GPU,
//                                       2=All, 3=CPU+NeuralEngine
//   uses_cpu_only          (bool, default false)
//                                       inference-time override of
//                                       compute_units.
class CoreMLInferenceStage final
  : public TypedStage<CoreMLInferenceStage>
{
public:
  static constexpr const char* kTypeName = "coreml-inference";

  CoreMLInferenceStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);

  ~CoreMLInferenceStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;
  Job drain     (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string              _model_path;
  std::string              _input_feature_name;
  std::vector<std::string> _output_feature_names;
  int                      _compute_units{};   // ComputeUnitsAll == 2
  bool                     _uses_cpu_only{};

  // The model is borrowed from the session-shared
  // CoreMLModelManager via a shared_ptr; multiple stages requesting
  // the same (path, compute_units) pair share one underlying model
  // and thus one per-model prediction mutex.
  std::shared_ptr<CoreMLLoadedModel> _loaded;

  // PredictionOptions is per-stage (the uses_cpu_only override is a
  // per-stage knob) and owned via raw retain/release.
  CML::PredictionOptions* _opts  = nullptr;
};

}

#endif
