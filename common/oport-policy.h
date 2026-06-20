#ifndef OPORT_POLICY_H
#define OPORT_POLICY_H

#include <cstdint>

namespace vpipe {

// Producer-side knobs for an edge buffer. A Vertex / Stage owns one
// OportPolicy per out-port (see Vertex::set_oport_policy) and the
// PipelineRuntime forwards them into the OportBuffer it constructs
// for each connected consumer.
//
// Defined here, not in pipeline/oport-buffer.h, so that common/vertex
// (which holds the per-port policy vector) does not have to depend
// on pipeline/.
enum class OverrunPolicy {
  Backpressure,   // default: a writer that finds the buffer full suspends
  DropOldest,     // writer never blocks; the front-of-queue beat is
                  // evicted to make room for the new one
};

// Per-oport policy. `capacity == 0` selects "default mode": the
// buffer's ring is sized at next_pow2(soft_error) and the soft
// thresholds are armed -- a WARN is logged (one-shot) when the
// active-Beat count crosses soft_warn, and the write throws
// std::runtime_error when it crosses soft_error.
//
// When `capacity != 0` the user has chosen a fixed depth: the value
// is clamped to [1, kMaxCapacity], the ring is sized accordingly,
// soft thresholds are disabled, and overflow triggers the configured
// `mode` (Backpressure or DropOldest).
struct OportPolicy {
  unsigned      capacity   = 0;
  OverrunPolicy mode       = OverrunPolicy::Backpressure;
  std::uint32_t soft_warn  = 1024;
  std::uint32_t soft_error = 2048;

  // Hard upper bound for user-specified depth. (wp - rp) is also
  // bounded by this value, so per-cursor 32-bit sequence numbers
  // with subtract-wraparound remain unambiguous.
  static constexpr std::uint32_t kMaxCapacity = (1u << 20);
};

}

#endif
