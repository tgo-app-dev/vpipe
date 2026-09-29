#ifndef VPIPE_APPLE_SILICON_COREML_ANE_WORKER_H
#define VPIPE_APPLE_SILICON_COREML_ANE_WORKER_H

#include <cstddef>
#include <functional>
#include <memory>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

// Threads for Apple Neural Engine dispatch, created ONCE with the
// session's CoreML manager and reused for its lifetime.
//
// WHY THIS EXISTS. A caller that wants the ANE and the GPU working at
// the same time has to start the ANE WITHOUT blocking, and CoreML's
// predict() is synchronous -- so the dispatch has to go to another
// thread. Doing that with a fresh std::thread per unit of work puts a
// creation and a join on the critical path every time; a DiT running
// its feed-forward here would pay ~28 of each per denoise step, for
// work that is the same shape every time.
//
// ONE DISPATCH THREAD, because the ANE is a single device and CoreML
// serialises predicts on a model anyway -- a second would only queue
// behind the first while costing another stack.
//
// THE CONVERT POOL IS SEPARATE from the dispatch thread, deliberately:
// a dispatched job typically has to convert its inputs (an fp16 engine
// fed from a bf16 pipeline, say), and it must be able to call
// parallel_for() from INSIDE that job without deadlocking against
// itself.
//
// This lives beside the CoreML wrapper rather than with any one model
// family because it is a property of the DEVICE, not of a model: the
// vision towers, the inference stage and a generative family all
// dispatch to the same ANE and should share the same thread.
class AneWorker {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
 public:
  AneWorker();
  ~AneWorker();

  AneWorker(const AneWorker&)            = delete;
  AneWorker& operator=(const AneWorker&) = delete;

  // Run `fn` on the dispatch thread and return immediately. At most one
  // outstanding at a time; a second call waits for the first to finish.
  void dispatch(std::function<void()> fn);

  // Block until the outstanding dispatch (if any) has finished.
  void join();

  // Split [0, count) across the convert threads, running `body(lo, hi)`
  // on each slice and returning when all are done. Runs INLINE below a
  // threshold, where the handoff would cost more than the work, and
  // whenever the pool could not start -- so a caller never has to
  // decide.
  //
  // Safe to call from inside a dispatch()ed job (that is the point of
  // the pools being separate); NOT safe to call re-entrantly from
  // inside another parallel_for.
  void parallel_for(std::size_t count,
                    const std::function<void(std::size_t, std::size_t)>& body);

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace vpipe

VPIPE_API_END

#endif
