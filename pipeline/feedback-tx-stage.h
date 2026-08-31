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
// ---- PRIMING: THE FIRST ROUND HAS NOTHING TO RELAY ----
//
// A feedback edge is a one-iteration delay, so on iteration ONE there
// is nothing behind it. tx blocks until rx has received a beat, and in
// a loop where rx's beat is CAUSED by what tx emits, that is a
// deadlock: nobody can go first.
//
// `prime` breaks it by emitting one beat before any feedback exists.
// The loop then runs: round 1 consumes the primed beat, produces the
// output that reaches rx, and round 2 relays it for real.
//
// The primed beat is EMPTY, which is not a placeholder but this tree's
// declared way to say "nothing this time" -- `video-ref-encoder` and
// `generate-video` both read an empty tensor as an absent reference
// (see the loop in VideoRefEncoderStage::process), precisely so a
// wired port can beat every request without always having something to
// say. A port that could fall SILENT instead would renumber every
// reference after it, and the numbering is the request.
//
// So the consumer needs no special case for round one; it needs the
// case it already has. What `prime` names is the beat's TYPE, which is
// why it is not a bool: this stage is type-erased and cannot know what
// its consumer reads, and an empty tensor is not an empty anything
// else.
//
// It primes ONCE per run. After that the stage waits for real feedback
// exactly as before, so a graph cannot silently keep running on
// nothing.
//
// Configuration:
//   from  (string, required) -- id of the feedback-rx stage in the
//                               same pipeline.
//   prime (string, default "none") -- "none" waits for feedback, as
//                               before. "empty-tensor" emits one empty
//                               TensorBeatPayload first, so a loop
//                               whose first round has no history can
//                               start.
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

  // What to emit before any feedback has arrived. The beat's TYPE is
  // the whole content of the setting, so it names a kind rather than
  // being a bool -- see PRIMING above.
  enum class Prime { kNone, kEmptyTensor };

private:
  std::string      _from_id;
  FeedbackRxStage* _rx = nullptr;
  Prime            _prime = Prime::kNone;
  // Per-RUN. Stages outlive a stop/relaunch while the runtime does not,
  // so a second launch must prime again -- otherwise the loop that
  // needed a first beat deadlocks on every run but the first.
  bool             _primed = false;
  std::uint64_t    _last_seen = 0;
};

}

#endif
