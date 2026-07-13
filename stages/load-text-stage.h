#ifndef VPIPE_STAGES_LOAD_TEXT_STAGE_H
#define VPIPE_STAGES_LOAD_TEXT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vpipe {

// Source-ish stage: 0 or 1 inputs, 1 output. Reads text files from the local
// filesystem and emits each file's contents as a plain FlexData string beat on
// out-port 0 -- the text-input format, consumed by text-to-image / text-chat.
// Mirrors load-image / load-audio (a file-driven source), for text.
//
// Pacing: when an iport is wired (typically a `chrono` stage), each upstream
// beat triggers exactly one read+emit. With no iport the stage emits every
// file back-to-back and signals done.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   path  (string or array<string>, required) -- one file path or a list.
class LoadTextStage final : public TypedStage<LoadTextStage> {
public:
  static constexpr const char* kTypeName = "load-text";

  LoadTextStage(const SessionContextIntf* session,
                std::string               id,
                std::vector<InEdge>       iports,
                FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessor.
  const std::vector<std::string>& paths() const noexcept { return _paths; }

private:
  std::vector<std::string> _paths;
  std::size_t              _next = 0;
};

}

#endif
