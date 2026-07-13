#ifndef STAGES_MODEL_FETCH_STAGE_H
#define STAGES_MODEL_FETCH_STAGE_H

#include "pipeline/typed-stage.h"

namespace vpipe {

// Interactive one-shot stage that obtains a model from HuggingFace:
//
//   1) identify a model -- either browse the internal catalogue
//      (model-catalog.{h,cc}) by drilling down family -> version ->
//      parameter class -> variant, or type a full path / URL directly;
//   2) download every file in the repo to a base path (default
//      ./models), under <base>/<owner>/<repo>;
//   3) for Qwen3-ASR, prepare the tokenizer.json our runtime needs by
//      synthesizing it natively from the repo's byte-level-BPE files
//      (vocab.json + merges.txt + tokenizer_config.json) -- no
//      transformers/Python; a no-op when the repo already ships one;
//   4) register the model in the active LMDB env under sub-db "models",
//      keyed by the path after huggingface.co, with a FlexData record of
//      the key facts;
//   5) signal_done and terminate.
//
// 1 optional trigger iport (any beat) + 1 FlexData "summary" oport, so
// these preparation stages can be cascaded into a recipe and/or dumped to
// a save-text report. All prompts route through the session UI delegate
// (stdin by default, or the browser under the web-ui delegate); the
// optional HuggingFace token prompt uses getpasswd so it is masked.
class ModelFetchStage final : public TypedStage<ModelFetchStage>
{
public:
  static constexpr const char* kTypeName = "model-fetch";

  ModelFetchStage(const SessionContextIntf* s,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*.
  std::string   _base_path;          // download root ("" -> ./models)
  std::string   _db_name;            // LMDB sub-db for the registry
  std::string   _model_path;         // non-interactive: a direct hf path
  std::string   _hf_token;           // optional auth ("" -> $HF_TOKEN)
  bool          _overwrite_existing{};
  bool          _prepare_tokenizer{};
  bool          _skip_existing_files{};
  bool          _verify_tls{};
  unsigned      _timeout_seconds{};
};

}

#endif
