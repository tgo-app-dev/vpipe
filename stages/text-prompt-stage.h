#ifndef VPIPE_STAGES_TEXT_PROMPT_STAGE_H
#define VPIPE_STAGES_TEXT_PROMPT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: 0-1 inputs, 1 output. Emits the configured `text` as a plain
// FlexData string beat on out-port 0 -- the same format text-input produces and
// text-to-image / text-chat consume (a bare string, no media/attachment
// markers). A config-driven prompt source that fits the image-generation
// workflow (feed a fixed prompt to text-to-image without a stdin reader).
//
// With no iport it emits once and signals done. With a trigger iport wired,
// each inbound beat re-emits the text (EOS on the iport ends the stage) -- e.g.
// to re-run generation on the same prompt paced by a chrono/feedback source.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   text  (string, required) -- the prompt text to emit.
class TextPromptStage final : public TypedStage<TextPromptStage> {
public:
  static constexpr const char* kTypeName = "text-prompt";

  TextPromptStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessor.
  const std::string& text() const noexcept { return _text; }

private:
  std::string _text;
  bool        _done = false;   // no-iport one-shot guard
};

}

#endif
