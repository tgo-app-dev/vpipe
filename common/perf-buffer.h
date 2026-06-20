#ifndef PERF_BUFFER_H
#define PERF_BUFFER_H

#include "common/perf-event.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpipe {

// Bounded, append-only event buffer for one worker thread.
//
// Capacity is fixed at construction. A producer calling record()
// claims the next slot via a relaxed atomic fetch_add and writes
// the slot directly. When the buffer is full, record() drops the
// new event but still increments the counter so dropped() reflects
// the true overflow.
//
// Lock-free on the producer side. With per-worker buffers the
// common case is single-producer (the worker thread that owns the
// buffer index), but the overflow buffer used for non-worker
// callers may see multiple producers; the atomic primitives keep
// both cases correct.
//
// Reads (size(), data(), at()) are intended to be called only when
// no producers are active -- typically from Session::dump_profiling
// after stages have stopped, or under a higher-level protocol that
// guarantees quiescence.
class PerfBuffer {
public:
  // capacity == 0 still constructs but record() is a no-op (the
  // first relaxed load returns 0, then the bound check rejects).
  PerfBuffer(unsigned capacity,
             std::chrono::steady_clock::time_point anchor) noexcept;

  PerfBuffer(const PerfBuffer&)            = delete;
  PerfBuffer& operator=(const PerfBuffer&) = delete;

  // Producer hot path. Reads steady_clock, computes ns since the
  // anchor, claims a slot, writes 24 bytes. Drops if full.
  void record(uint32_t stage_gvid,
              uint32_t type,
              uint64_t value) noexcept;

  // Reader-side -- not safe to call concurrently with record().
  std::size_t      capacity() const noexcept { return _events.size(); }
  std::size_t      size()     const noexcept;
  const PerfEvent* data()     const noexcept { return _events.data(); }
  const PerfEvent& at(std::size_t i) const   { return _events[i]; }

  // Number of record() calls that hit the cap and were dropped.
  // (Equal to total record-attempts - min(size, capacity).) Used by
  // dump_profiling so users can see whether the cap was too small.
  std::size_t dropped() const noexcept;

  // Reset the buffer to empty. Caller must guarantee no concurrent
  // producers. Used by enable_profiling on a re-enable.
  void clear() noexcept;

  std::chrono::steady_clock::time_point
  anchor() const noexcept { return _anchor; }

private:
  std::chrono::steady_clock::time_point _anchor;
  std::vector<PerfEvent>                _events;  // size == capacity
  // _next is a write-attempt counter. It can grow past capacity --
  // the slot store is gated by a separate bound check. uint64 to
  // never overflow. Relaxed memory order is enough: writes go to
  // disjoint slots, and dump-time reads are sequenced after all
  // producers via a higher-level happens-before edge (driver
  // teardown).
  std::atomic<std::uint64_t>            _next{0};
};

}

#endif
