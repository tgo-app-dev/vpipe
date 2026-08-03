#ifndef STAGES_MODEL_REGISTER_STAGE_H
#define STAGES_MODEL_REGISTER_STAGE_H

#include "stages/model-detect.h"

#include "pipeline/typed-stage.h"

namespace vpipe {

// One-shot stage that REGISTERS a model directory already on disk into
// the model registry -- model-fetch without the download, for a model
// built locally (model-quantize / lora-fuse output kept outside a
// recipe), copied from another machine, or downloaded by hand.
//
// What it registers is DETECTED, not typed: from the directory alone
// (detect_model_dir, model-detect.h) it derives the runtime hint
// `model_type` and the input / output modalities that the stage model
// pickers and the web-ui model browser filter on, plus best-effort
// family / version / param_class / variant labels. So the usual config is
// one line -- the path -- and the registered record behaves like a
// fetched one.
//
// Detection is deliberately conservative: model_type is set only when the
// directory matches a catalogue entry or carries a config it recognizes,
// and is left EMPTY otherwise rather than guessed, because a wrong
// runtime hint offers a model to a stage that cannot load it. `model_type`
// (config) overrides the detection when the user knows better.
//
// Like the other preparation stages it exposes 1 optional trigger iport
// (any beat type) + 1 FlexData "summary" oport, so it chains into a
// recipe (e.g. quantize -> register -> benchmark) and/or fans a report
// out to save-text. The summary is emitted only on success, so a failed
// registration halts the cascade.
//
// Config (FlexData object):
//   model_dir    (string, required) -- the model directory on disk (or a
//                 bare .mlpackage / .gguf file). Must exist.
//   key          (string, optional) -- registry key to register under;
//                 empty => "<owner>/<repo>" from the two trailing path
//                 components (the layout model-fetch writes), else the
//                 directory name.
//   model_type   (string, optional) -- override the detected runtime hint.
//   overwrite_existing (bool, default false) -- let the key be taken
//                 over from a DIFFERENT directory. Off by default:
//                 silently rewriting another model's registration is
//                 worse than refusing. Re-registering the SAME directory
//                 needs no opt-in -- it is idempotent, so a relaunched
//                 recipe does not halt on its second run.
class ModelRegisterStage final : public TypedStage<ModelRegisterStage>
{
public:
  static constexpr const char* kTypeName = "model-register";

  ModelRegisterStage(const SessionContextIntf* s,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Outcome of one registration (the test seam below returns it;
  // process() turns it into the summary beat).
  struct RegisterResult {
    bool          ok        = false;  // registered -> emit summary
    bool          replaced  = false;  // an existing record was overwritten
    std::string   key;                // the key written
    std::string   local_path;         // the absolute dir registered
    DetectedModel detected;           // what the directory turned out to be
  };

  // Test seam: detect + write the record once, logging through the
  // session. Never throws; `ok` drives whether process() emits a summary.
  RegisterResult register_once();

  // Test-only inspectors.
  const std::string& model_dir() const noexcept { return _model_dir; }
  const std::string& key() const noexcept { return _key; }
  const std::string& model_type() const noexcept { return _model_type; }
  bool overwrite_existing() const noexcept { return _overwrite_existing; }

private:
  std::string _model_dir;
  std::string _key;             // empty => derived from the path
  std::string _model_type;      // empty => detected
  bool        _overwrite_existing{};
};

}

#endif
