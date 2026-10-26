#ifndef VPIPE_PIPELINE_RESOURCE_CLAIM_DELIVERY_H
#define VPIPE_PIPELINE_RESOURCE_CLAIM_DELIVERY_H

// The claim the runtime is delivering to a planner on this thread, while
// it delivers it: what a HOST planner reads a claim's `extra` through.
// ResourcePlanner::claim() (pipeline/resource-plan.h) is in plugin ABI 8
// and takes the claim's fields, not the claim, so a key in its bag
// reaches a planner only this way. The runtime opens a scope around each
// claim it delivers; a test driving a planner by hand opens its own.
//
// INTERNAL: not installed and not exported.

#include "pipeline/resource-plan.h"

namespace vpipe {

class DeliveringResourceClaim {
public:
  explicit DeliveringResourceClaim(const ResourceClaim& c) noexcept
    : _outer(slot_())
  {
    slot_() = &c;
  }
  ~DeliveringResourceClaim() { slot_() = _outer; }
  DeliveringResourceClaim(const DeliveringResourceClaim&)            = delete;
  DeliveringResourceClaim& operator=(const DeliveringResourceClaim&) =
      delete;

  // The claim being delivered on this thread; null outside a delivery.
  static const ResourceClaim* current() noexcept { return slot_(); }

private:
  static const ResourceClaim*& slot_() noexcept
  {
    thread_local const ResourceClaim* c = nullptr;
    return c;
  }

  const ResourceClaim* _outer;
};

}  // namespace vpipe

#endif
