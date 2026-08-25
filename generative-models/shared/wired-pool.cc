#include "generative-models/shared/wired-pool.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <cstdlib>

namespace vpipe::genai {
namespace {

GenerativeModelManager*
manager_(metal_compute::MetalCompute* mc)
{
  const auto* sess = mc != nullptr ? mc->session() : nullptr;
  if (sess == nullptr || sess->services() == nullptr) { return nullptr; }
  return sess->services()->generative_model_manager();
}

}  // namespace

void
WiredPool::open(metal_compute::MetalCompute* mc)
{
  auto* mgr = manager_(mc);
  if (mgr != nullptr) {
    const std::size_t lim = mgr->wired_pool_limit();
    if (lim > 0) {
      const std::size_t used = mgr->wired_pool_used();
      _on     = true;
      _budget = lim > used ? lim - used : 0;
    }
  }
  if (const char* e = std::getenv("VPIPE_WIRE_RESIDENT")) {
    _on = std::atoi(e) != 0;
  }
}

std::size_t
WiredPool::wire_one(metal_compute::MetalCompute* mc,
                    metal_compute::SharedBuffer& b, bool on)
{
  if (b.byte_size() == 0 || b.is_wired() == on) { return 0; }
  auto* mgr = manager_(mc);
  if (mgr == nullptr) { return 0; }
  if (!on) {
    const std::size_t n = b.byte_size();
    mgr->unwire_from_pool(b);
    return n;
  }
  return mgr->wire_into_pool(b);
}

void
WiredPool::note_wired(metal_compute::MetalCompute* mc, std::size_t got,
                      std::size_t want)
{
  _wired += got;
  if (got >= want || !_on) { return; }
  _budget   = _wired;
  _retry    = true;
  _retry_at = mc != nullptr ? mc->memory_budget().available_physical : 0;
  if (mc != nullptr && mc->session() != nullptr) {
    mc->session()->log_debug(fmt(
        "wired pool: the box granted {} MB and refused more; holding there "
        "for this forward and retrying on the next", _wired >> 20));
  }
}

bool
WiredPool::retry(metal_compute::MetalCompute* mc, std::size_t block_bytes)
{
  if (!_on || !_retry || mc == nullptr) { return false; }
  const std::size_t now = mc->memory_budget().available_physical;
  if (now <= _retry_at + block_bytes) { return false; }
  auto* mgr = manager_(mc);
  if (mgr == nullptr) { return false; }
  mgr->reopen_wired_pool();
  const std::size_t lim  = mgr->wired_pool_limit();
  const std::size_t used = mgr->wired_pool_used();
  const std::size_t room = lim > used ? lim - used : 0;
  // Only ever RAISED here. The budget also bounds what this model has
  // already wired, and lowering it below `_wired` would read as an
  // over-spend that nothing can give back.
  //
  // CLAMPED TO THE POOL, belt and braces. The arithmetic already gives
  // `_wired + (lim - used) <= lim` because this model's wired bytes are
  // part of the manager's `used` -- but that holds only while the two
  // counters agree, and the failure mode if they ever drift is not a slow
  // run. Wired memory is the one allocation the kernel cannot reclaim, so
  // an over-budget here panics the box rather than degrading it. One
  // min() is a cheap way to never find out.
  std::size_t want = _wired + room;
  if (want > lim) { want = lim; }
  if (want <= _budget) {
    // The ceiling did not move after all -- the pool is full rather than
    // the box being busy. Re-arm against the CURRENT reading so the next
    // look asks about a fresh block's worth.
    _retry_at = now;
    return false;
  }
  if (mc->session() != nullptr) {
    mc->session()->log_debug(fmt(
        "wired pool: retrying -- budget {} -> {} MB", _budget >> 20,
        want >> 20));
  }
  _budget = want;
  _retry  = false;
  return true;
}

}  // namespace vpipe::genai
