#include "pipeline/feedback-rx-stage.h"

#include "common/thread-pool.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <utility>

using namespace std;

namespace vpipe {

FeedbackRxStage::FeedbackRxStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<FeedbackRxStage>(s, std::move(id), std::move(iports),
                                std::move(config))
{
  // No oports. iport count is determined by the iports vector passed
  // in by the pipeline-spec layer; the runtime validates that exactly
  // one is wired.
}

namespace {
const PortSpec kIports[] = {
  {.name = "in", .doc = "any beat; the latest is cached for the tx side",
   .type = nullptr, .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "feedback-rx",
  .doc       = "Receive half of a single-clock-domain feedback pair: "
               "caches the most recent beat on its iport for the named "
               "feedback-tx to re-emit (a one-iteration delay register).",
  .display_name = "Feedback In",
  .category  = StageCategory::Control,
  .iports    = kIports,
  .oports    = {},
  .attrs     = {},
};
}  // namespace

const StageSpec&
FeedbackRxStage::spec() const noexcept
{
  return kSpec;
}

Job
FeedbackRxStage::initialize(RuntimeContext&)
{
  _pool = session() ? session()->thread_pool() : nullptr;
  if (!_pool) {
    session()->error(fmt(
        "FeedbackRxStage('{}'): no ThreadPool on session",
        this->id()));
  }
  // Per-launch reset. Stage objects outlive a stop (the handle only
  // destroys the runtime), so a stopped run leaves _eos=true (its read
  // returned null when the buffers closed) plus the last run's beat +
  // seq behind. Without this reset a RELAUNCH is dead on arrival: the
  // paired tx's wait_new_beat resolves instantly on the stale _eos,
  // sees no new beat, and signals done -- closing the loop's trigger
  // edge before the first round can run. _waiters is necessarily empty
  // here (the prior run's waiters were woken by the EOS path before
  // its runtime was destroyed); cleared for the record.
  {
    lock_guard<mutex> lk(_mu);
    _eos = false;
    _seq = 0;
    _last.reset();
    _waiters.clear();
  }
  co_return;
}

Job
FeedbackRxStage::process(RuntimeContext& ctx)
{
  auto t = co_await ctx.read(0);
  vector<coroutine_handle<>> wake;
  if (!t) {
    {
      lock_guard<mutex> lk(_mu);
      _eos = true;
      wake.swap(_waiters);
    }
    for (auto h : wake) {
      if (_pool) { _pool->schedule(h); }
    }
    ctx.signal_done();
    co_return;
  }
  // Take a snapshot the tx side can hand out to its downstream:
  // clone() owns its own storage so the read cursor in our iport's
  // EdgeReader can release the slot immediately when this process()
  // iteration returns.
  shared_ptr<const BeatPayloadIntf> snap = t->clone();
  {
    lock_guard<mutex> lk(_mu);
    _last = std::move(snap);
    ++_seq;
    wake.swap(_waiters);
  }
  for (auto h : wake) {
    if (_pool) { _pool->schedule(h); }
  }
}

shared_ptr<const BeatPayloadIntf>
FeedbackRxStage::snapshot() const
{
  lock_guard<mutex> lk(_mu);
  return _last;
}

uint64_t
FeedbackRxStage::current_seq() const noexcept
{
  lock_guard<mutex> lk(_mu);
  return _seq;
}

bool
FeedbackRxStage::eos() const noexcept
{
  lock_guard<mutex> lk(_mu);
  return _eos;
}

bool
FeedbackRxStage::WaitAwaiter::await_ready() noexcept
{
  lock_guard<mutex> lk(_rx->_mu);
  return _rx->_seq > _last_seen || _rx->_eos;
}

bool
FeedbackRxStage::WaitAwaiter::await_suspend(coroutine_handle<> h)
{
  lock_guard<mutex> lk(_rx->_mu);
  if (_rx->_seq > _last_seen || _rx->_eos) {
    return false;
  }
  _rx->_waiters.push_back(h);
  return true;
}

VPIPE_REGISTER_STAGE(FeedbackRxStage)
VPIPE_REGISTER_SPEC(FeedbackRxStage, kSpec)

}
