#include "common/perf-buffer.h"

using namespace std;

namespace vpipe {

PerfBuffer::PerfBuffer(unsigned capacity,
                       chrono::steady_clock::time_point anchor) noexcept
  : _anchor(anchor)
  , _events(capacity)
{
}

void
PerfBuffer::record(uint32_t stage_gvid,
                   uint32_t type,
                   uint64_t value) noexcept
{
  // Always claim a slot index via fetch_add so dropped() can be
  // computed as (total claims - capacity). On x86 a relaxed
  // fetch_add is one xadd instruction (~10ns under contention),
  // not noticeably slower than a plain load. uint64 makes
  // overflow practically impossible (584 years at 1ns/record).
  uint64_t idx = _next.fetch_add(1, memory_order_relaxed);
  const size_t cap = _events.size();
  if (idx >= cap) {
    return;
  }

  // Per-event clock read happens AFTER the slot has been claimed
  // so wall-clock order tracks slot order even under contention.
  auto now = chrono::steady_clock::now();
  uint64_t ns = static_cast<uint64_t>(
      chrono::duration_cast<chrono::nanoseconds>(now - _anchor).count());

  PerfEvent& slot = _events[static_cast<size_t>(idx)];
  slot.ns         = ns;
  slot.value      = value;
  slot.stage_gvid = stage_gvid;
  slot.type       = type;
}

size_t
PerfBuffer::size() const noexcept
{
  uint64_t n = _next.load(memory_order_relaxed);
  size_t cap = _events.size();
  return n < cap ? static_cast<size_t>(n) : cap;
}

size_t
PerfBuffer::dropped() const noexcept
{
  uint64_t n = _next.load(memory_order_relaxed);
  size_t cap = _events.size();
  return n > cap ? static_cast<size_t>(n - cap) : 0;
}

void
PerfBuffer::clear() noexcept
{
  _next.store(0, memory_order_relaxed);
}

}
