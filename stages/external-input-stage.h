#ifndef VPIPE_STAGES_EXTERNAL_INPUT_STAGE_H
#define VPIPE_STAGES_EXTERNAL_INPUT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source: turns data from OUTSIDE the pipeline -- the C++ API, Python,
// any caller of a stage command -- into TensorBeats on its oport. It
// has no inputs and emits nothing on its own; each command is one beat.
//
// Commands (see docs/STAGE-COMMANDS.md):
//   push    in `data`: any tensor the caller owns. Copied into a beat
//           and emitted; the reply comes once the oport has taken it,
//           so a caller that waits on each push is paced by the
//           pipeline rather than filling memory ahead of it. `sideband`
//           (object) rides on the beat.
//   lease   args `type`, `shape`: the stage allocates a beat of that
//           geometry and replies with it as a WRITABLE buffer. The
//           caller fills it in place and closes the command, and the
//           beat is emitted -- no copy anywhere when the caller let go
//           of its view, one when it kept it (so the view stays its
//           own). Commands queued behind a lease wait for its close.
//   finish  end of stream: the stage emits nothing more and its
//           consumers see EOS.
//
// oport 0: tensor -- one TensorBeat per push / closed lease.
class ExternalInputStage final : public TypedStage<ExternalInputStage> {
public:
  static constexpr const char* kTypeName = "external-input";

  ExternalInputStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  void reset_run_state() override { _emitted = 0; }

private:
  Job push_(RuntimeContext& ctx, std::shared_ptr<StageCommand> cmd);
  Job lease_(RuntimeContext& ctx, std::shared_ptr<StageCommand> cmd);

  std::uint64_t _emitted = 0;
};

}

#endif
