#ifndef VPIPE_STAGES_EXTERNAL_TAP_STAGE_H
#define VPIPE_STAGES_EXTERNAL_TAP_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Passthrough that lets code OUTSIDE the pipeline read -- and edit --
// the beats flowing through it, in place. Every beat goes out exactly
// as it came in unless a caller changed it.
//
// Command (see docs/STAGE-COMMANDS.md):
//   read  the next TensorBeat to arrive is exposed to the caller as a
//         WRITABLE buffer, zero-copy, and the tap HOLDS it -- emitting
//         nothing, so everything downstream waits and everything
//         upstream backs up -- until the caller closes the command.
//         Edits made before the close travel downstream with the beat.
//         The result carries the beat's index, type, shape and
//         sideband. A caller that keeps a view past close() keeps its
//         memory: the tap then forwards a copy.
//
// Two modes (`mode`):
//   pass  (default) beats flow freely; a queued `read` takes the next
//         one to arrive. Reading costs the pipeline nothing between
//         reads.
//   gate  every TensorBeat waits for a `read` before it moves on, so
//         the pipeline runs exactly as fast as the caller consumes it.
//
// A beat that is not a TensorBeat passes through untouched in both
// modes and does not use up a read.
//
// iport 0: in  -- any beat.
// oport 0: out -- the same beats, in order.
class ExternalTapStage final : public TypedStage<ExternalTapStage> {
public:
  static constexpr const char* kTypeName = "external-tap";

  ExternalTapStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  void reset_run_state() override { _seq = 0; }

private:
  bool          _gate = false;
  std::uint64_t _seq  = 0;
};

}

#endif
