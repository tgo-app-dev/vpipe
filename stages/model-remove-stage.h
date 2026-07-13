#ifndef STAGES_MODEL_REMOVE_STAGE_H
#define STAGES_MODEL_REMOVE_STAGE_H

#include "pipeline/typed-stage.h"

namespace vpipe {

// One-shot stage that REMOVES a model from the registry -- the inverse of
// model-fetch. Given a models-DB key (the key model-fetch / model-quantize
// register under), it deletes that record from the LMDB sub-db and, when
// `delete_files` is set, also removes the recorded on-disk model directory.
//
// Like the other preparation stages it exposes 1 optional trigger iport
// (any beat type) + 1 FlexData "summary" oport, so it chains into a
// preparation recipe (e.g. fetch -> quantize -> benchmark -> remove the
// bf16 source) and/or fans a report out to save-text.
//
// The summary is emitted ONLY on success, so a failure HALTS the cascade
// (mirrors model-quantize / -benchmark / -eval). A model that is already
// absent is a success when `missing_ok` is set (the desired end-state --
// "not registered" -- already holds), so a cleanup recipe keeps flowing.
//
// Config (FlexData object):
//   model        (string, required) -- the models-DB key to remove.
//   models_db    (string, default "models") -- LMDB sub-db to remove from.
//   delete_files (bool, default false) -- also delete the model's recorded
//                 `local_path` directory from disk (opt-in; destructive).
//   missing_ok   (bool, default true) -- treat a not-registered model as a
//                 success (already removed) instead of halting the cascade.
class ModelRemoveStage final : public TypedStage<ModelRemoveStage>
{
public:
  static constexpr const char* kTypeName = "model-remove";

  ModelRemoveStage(const SessionContextIntf* s,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Outcome of one removal (the test seam below returns it; process()
  // turns it into the summary beat).
  struct RemoveResult {
    bool        present       = false;  // was the model registered?
    bool        removed       = false;  // did we delete the record?
    bool        files_deleted = false;  // did we rm the local_path dir?
    bool        ok            = false;  // overall success -> emit summary
    std::string local_path;             // recorded local_path (for report)
    std::string delete_note;            // human note on file deletion
  };

  // Test seam: perform the removal once (read record, optional file
  // delete, delete record) and log through the session. Never throws; the
  // returned `ok` drives whether process() emits a summary. `ok` is true
  // when the record was removed OR the model was already absent and
  // missing_ok is set.
  RemoveResult remove_once();

  // Test-only inspectors.
  const std::string& model() const noexcept { return _model; }
  const std::string& models_db() const noexcept { return _models_db; }
  bool delete_files() const noexcept { return _delete_files; }
  bool missing_ok() const noexcept { return _missing_ok; }

private:
  std::string _model;        // registry key to remove
  std::string _models_db;    // LMDB sub-db ("models")
  bool        _delete_files{};
  bool        _missing_ok{};
};

}

#endif
