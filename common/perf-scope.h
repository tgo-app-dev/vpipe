#ifndef PERF_SCOPE_H
#define PERF_SCOPE_H

#include "common/perf-event.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>

namespace vpipe {

// RAII bracket for an auxiliary-lane begin/end block (see perf-event.h
// PerfAuxLane). Records the begin on construction and the matching end
// on destruction, into the given aux lane. Profiling-enabled is sampled
// ONCE at construction and cached: if profiling was off then, the whole
// scope is a no-op (and stays so even if profiling flips on mid-scope,
// which would otherwise leave an unmatched end). The producer side is
// the lock-free PerfBuffer, so this is safe to use from any thread
// (the LLM MLX-runtime worker, a CoreML callback thread, a pool
// worker). When profiling is off the cost is one atomic load + branch.
//
//   {
//     PerfAuxScope p(session(), kPerfLaneLLM, kGvidLlmVision,
//                    kPerfLlmVisionBegin, n_frames);
//     enc = vision->encode(...);          // timed
//   }
class PerfAuxScope {
public:
  PerfAuxScope(const SessionContextIntf* sess,
               unsigned                  lane,
               std::uint32_t             gvid,
               std::uint32_t             begin_type,
               std::uint64_t             value = 0) noexcept
    : _sess((sess && sess->profiling_enabled()) ? sess : nullptr),
      _lane(lane),
      _gvid(gvid),
      _begin(begin_type),
      _value(value)
  {
    if (_sess) {
      _sess->record_perf_event_aux(_lane, _gvid, _begin, _value);
    }
  }

  // Override the payload recorded on the END event -- e.g. a token
  // count that is only known once the work finishes. Both the begin and
  // end carry a value so a viewer can derive a rate (tokens / duration)
  // from either side.
  void set_value(std::uint64_t v) noexcept { _value = v; }

  ~PerfAuxScope()
  {
    if (_sess) {
      _sess->record_perf_event_aux(_lane, _gvid, _begin + 1u, _value);
    }
  }

  PerfAuxScope(const PerfAuxScope&)            = delete;
  PerfAuxScope& operator=(const PerfAuxScope&) = delete;

private:
  const SessionContextIntf* _sess;   // null => profiling was off; no-op
  unsigned                  _lane;
  std::uint32_t             _gvid;
  std::uint32_t             _begin;
  std::uint64_t             _value;  // recorded on begin AND end
};

}  // namespace vpipe

#endif
