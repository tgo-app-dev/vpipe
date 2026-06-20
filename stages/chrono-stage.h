#ifndef VPIPE_STAGES_CHRONO_STAGE_H
#define VPIPE_STAGES_CHRONO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: 0 inputs, 1 output. Periodically emits a TriggerBeat
// on out-port 0.
//
// Configuration (FlexData object on the 4th constructor parameter).
// Specify the period exactly once via either:
//   frequency_hz       (real)  -- preferred for sub-second rates
// or any non-zero combination of:
//   period_seconds     (real)
//   period_minutes     (real)
//   period_hours       (real)
//   period_days        (real)
// (period_* fields stack additively, so {hours: 1, minutes: 30} is
// 1.5h.) Mixing frequency_hz with any period_* field is rejected.
//
// Optional:
//   count   (int, default 0) -- 0 means "tick forever until the
//                               pipeline is stopped"; N > 0 means
//                               "tick N times then signal done".
//
// The first tick fires *after* one period elapses, not immediately.
// Stop is honoured promptly: the wait is broken into 50ms chunks
// that poll stop_requested().
class ChronoStage final : public TypedStage<ChronoStage> {
public:
  static constexpr const char* kTypeName = "chrono";

  ChronoStage(const SessionContextIntf* session,
              std::string               id,
              std::vector<InEdge>       iports,
              FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  std::chrono::nanoseconds period() const noexcept { return _period; }
  uint64_t                 count()  const noexcept { return _count; }

private:
  std::chrono::nanoseconds _period{};
  uint64_t                 _count = 0;       // 0 == indefinite
  uint64_t                 _emitted = 0;
};

}

#endif
