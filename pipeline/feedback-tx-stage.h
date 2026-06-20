#ifndef PIPELINE_FEEDBACK_TX_STAGE_H
#define PIPELINE_FEEDBACK_TX_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

class FeedbackRxStage;

// Source half of the single-clock-domain feedback pair. Each call to
// process() waits for the named feedback-rx stage to receive a new
// beat, then re-emits a clone of that beat on out-port 0. There is no
// data-flow edge between this stage and the named rx; the wiring is
// by name (config.from), and the pipeline runtime validates that
// they end up in the same clock domain via the rest of the graph.
//
// Because tx always blocks until rx has received a new beat,
// downstream of tx naturally observes the previous round's beat
// relative to where rx is wired -- the "one-iteration delay" register
// of a synchronous-dataflow feedback edge.
//
// Configuration:
//   from  (string, required) -- id of the feedback-rx stage in the
//                               same pipeline.
//
// Ports: 0 iports, 1 oport (payload type matches whatever the rx side
//        receives; this stage is type-erased).
class FeedbackTxStage final : public TypedStage<FeedbackTxStage> {
public:
  static constexpr const char* kTypeName = "feedback-tx";

  FeedbackTxStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Public accessors used by the pipeline runtime's clock-domain
  // verification pass.
  const std::string& from_id() const noexcept { return _from_id; }
  FeedbackRxStage*   rx()      const noexcept { return _rx;      }

private:
  std::string      _from_id;
  FeedbackRxStage* _rx = nullptr;
  std::uint64_t    _last_seen = 0;
};

}

#endif
