#ifndef STAGES_MODEL_FETCH_STAGE_H
#define STAGES_MODEL_FETCH_STAGE_H

#include "pipeline/typed-stage.h"

namespace vpipe {

// Interactive one-shot stage that obtains a model from HuggingFace or a
// mirror of it:
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
//   4) register the model in the active LMDB env under the model
//      registry sub-db (model-registry.h),
//      keyed by the path after huggingface.co, with a FlexData record of
//      the key facts;
//   5) signal_done and terminate.
//
// WHERE the bytes come from is a separate axis from WHICH model is meant
// (model-source.h). The catalogue speaks HuggingFace throughout, and the
// registry key and on-disk directory are derived from the HuggingFace
// path whichever source served the download -- so a model fetched from
// modelscope.cn (the mirror reachable from mainland China) is the same
// model, in the same place, under the same key. Pick the source with the
// `source` attribute, or once per machine with $VPIPE_MODEL_SOURCE.
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
  std::string   _model_path;         // non-interactive: a direct hf path
  // Which catalogue entry, when several are published from one repo.
  std::string   _model_variant;
  // Registration key override, so two such models can share a directory
  // on disk and still be distinct records in the models DB.
  std::string   _model_key;
  std::string   _hf_token;           // optional auth ("" -> $HF_TOKEN)
  // Where to fetch FROM: a model-source.h name ("" -> $VPIPE_MODEL_SOURCE,
  // else huggingface), and the branch to read ("" -> that source's own
  // default, which is NOT the same string on both).
  std::string   _source;
  std::string   _source_revision;
  bool          _overwrite_existing{};
  bool          _prepare_tokenizer{};
  bool          _skip_existing_files{};
  bool          _verify_tls{};
  bool          _verify_checksums{};
  unsigned      _timeout_seconds{};
  unsigned      _stall_seconds{};
  unsigned      _download_retries{};
  unsigned      _xet_streams{};
};

}

#endif
