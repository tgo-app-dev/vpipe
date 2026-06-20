#ifndef PIPELINE_FEEDBACK_RX_STAGE_H
#define PIPELINE_FEEDBACK_RX_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <coroutine>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace vpipe {

class ThreadPool;

// Half of the {feedback-tx, feedback-rx} pair that wires a single-
// clock-domain feedback edge into a pipeline. The pair is a logical
// register / one-beat delay: the rx stage caches the most recent beat
// it has received on its iport, and the tx stage re-emits that cached
// beat on its own oport, so a stage downstream of tx observes the
// beats that arrived at rx with a one-iteration delay.
//
// There is no edge in the graph between rx and tx. The wiring is by
// name: feedback-tx's `from` config string names this rx stage's id.
// The pipeline runtime verifies (post clock-domain analysis) that the
// rx's iport and the matching tx's oport end up in the same clock
// domain via the rest of the graph -- a feedback edge that spans
// clock domains is rejected.
//
// Configuration: none. The stage has exactly 1 iport, 0 oports.
class FeedbackRxStage final : public TypedStage<FeedbackRxStage> {
public:
  static constexpr const char* kTypeName = "feedback-rx";

  FeedbackRxStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Awaitable: suspends the caller until the rx has received a beat
  // with sequence strictly greater than `last_seen_seq`, OR until
  // upstream has closed (EOS). On resume, callers should re-check
  // current_seq() and eos() to distinguish.
  class WaitAwaiter {
  public:
    WaitAwaiter(FeedbackRxStage* rx, std::uint64_t last_seen) noexcept
      : _rx(rx)
      , _last_seen(last_seen)
    {}

    bool await_ready() noexcept;
    bool await_suspend(std::coroutine_handle<> h);
    void await_resume() const noexcept {}

  private:
    FeedbackRxStage* _rx;
    std::uint64_t    _last_seen;
  };

  WaitAwaiter wait_new_beat(std::uint64_t last_seen) noexcept
  {
    return WaitAwaiter{this, last_seen};
  }

  // Snapshot the latest cached beat. Returns nullptr if no beat has
  // arrived yet. Caller may clone() for fanout-safe consumption.
  std::shared_ptr<const BeatPayloadIntf> snapshot() const;

  // Monotonic sequence number; incremented on every beat received.
  std::uint64_t current_seq() const noexcept;

  // True once upstream has closed (EOS observed on the iport).
  bool eos() const noexcept;

private:
  ThreadPool* _pool = nullptr;

  mutable std::mutex                            _mu;
  std::shared_ptr<const BeatPayloadIntf>        _last;
  std::uint64_t                                 _seq = 0;
  bool                                          _eos = false;
  std::vector<std::coroutine_handle<>>          _waiters;
};

}

#endif
