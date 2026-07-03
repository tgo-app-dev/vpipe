#ifndef VPIPE_STAGES_TEXT_INPUT_STAGE_H
#define VPIPE_STAGES_TEXT_INPUT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <string>
#include <vector>

namespace vpipe {

// Source stage: 0-1 inputs, 1 output. Prints `prompt` to stdout, reads
// one line from stdin, wraps it as a FlexData string payload, emits
// it on out-port 0, and signals done. One-shot by default.
//
// When wired with 1 iport, the stage uses inbound beats as triggers
// for the next prompt/stdin round: each beat (after the optional
// first round; see `present_first_without_beat`) gates a fresh
// prompt/read cycle. EOS on the iport ends the stage. The intended
// upstream is a feedback-tx that relays a feedback-rx attached to a
// downstream text-chat oport, forming a chat loop.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   prompt  (string, optional, default ">> ") -- written to stdout
//                                                before the read.
//   count   (int, optional, default 1)        -- number of input
//                                                lines to read. 0
//                                                means "read until
//                                                EOF on stdin".
//   present_first_without_beat
//           (bool, optional, default true)    -- only meaningful when
//                                                an iport is wired.
//                                                When true, the very
//                                                first prompt is
//                                                presented without
//                                                waiting for an
//                                                iport beat (breaks
//                                                the feedback-loop
//                                                deadlock at startup);
//                                                subsequent rounds
//                                                wait for one beat
//                                                each. When false,
//                                                every round (incl.
//                                                the first) waits
//                                                for an iport beat.
//
// Reading stdin from a pipeline worker blocks that worker, which is
// fine for an interactive CLI driver and what every other stdio-
// driven stage does (e.g. shell-stage's std::system call).
class TextInputStage final : public TypedStage<TextInputStage> {
public:
  static constexpr const char* kTypeName = "text-input";

  TextInputStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);

  // Per-launch state reset (the stage object survives stop/relaunch).
  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& prompt() const noexcept { return _prompt; }
  uint64_t           count()  const noexcept { return _count; }
  bool present_first_without_beat() const noexcept
  { return _present_first_without_beat; }
  bool media() const noexcept { return _media; }

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string _prompt;
  uint64_t    _count   = 0;   // 0 == until EOF
  uint64_t    _emitted = 0;
  bool        _present_first_without_beat{};
  bool        _media{};   // read via getmedialine (attachment markers)
  bool        _first_round_seen           = false;
};

}

#endif
