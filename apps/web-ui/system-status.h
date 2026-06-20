#ifndef WEBUI_SYSTEM_STATUS_H
#define WEBUI_SYSTEM_STATUS_H

#include "common/flex-data.h"

#include <chrono>
#include <memory>
#include <mutex>

namespace vpipe::webui {

// Stateful system-status sampler for the bottom status bar.
//
// GPU utilisation + GPU memory come from the same IOKit `IOAccelerator`
// PerformanceStatistics dictionary nvtop's Apple-Silicon backend (and
// asitop / mactop) read. ANE utilisation comes from the same IOReport
// private framework macmon uses: an "Energy Model" subscription gives
// per-sample energy deltas for ANE / GPU / CPU clusters, and we
// estimate ANE utilisation as
//
//     ane_util_pct = 100 * ane_power_W / ane_max_W
//
// where `ane_max_W` is a per-chip ceiling (8.0 W for M1/M2, 8.5 W for
// M3 — same table macmon ships with). Numbers are clipped to [0,100].
//
// State (held across queries):
//   * an IOReport subscription on the "Energy Model" channels
//   * the previous sample, used to compute the per-second energy delta
//     (and hence power) on the next query
//
// Thread-safe: a single SystemStatusPoller may be queried from any
// thread; an internal mutex serialises sample-delta computation.
class SystemStatusPoller {
public:
  SystemStatusPoller();
  ~SystemStatusPoller();

  SystemStatusPoller(const SystemStatusPoller&)            = delete;
  SystemStatusPoller& operator=(const SystemStatusPoller&) = delete;

  // Snapshot. Returned FlexData object includes the following keys
  // (all optional — absent when the relevant source is unavailable):
  //
  //   "gpu_util_pct"        real -- IOAccelerator GPU util [0,100]
  //   "gpu_renderer_pct"    real
  //   "gpu_tiler_pct"       real
  //   "gpu_in_use_bytes"    uint -- IOAccelerator "In use system memory"
  //   "gpu_alloc_bytes"     uint -- IOAccelerator "Alloc system memory"
  //   "gpu_model"           string -- GPU/chip model, e.g. "Apple M4"
  //                                   (falls back to the CPU brand string)
  //   "gpu_cores"           uint -- GPU core count ("gpu-core-count")
  //   "ane_util_pct"        real -- estimated ANE util [0,100]
  //   "ane_power_w"         real -- instantaneous ANE power, watts
  //   "ane_max_w"           real -- per-chip ANE TDP ceiling
  //   "phys_footprint_bytes" uint -- task_vm_info.phys_footprint, the
  //                                  number Activity Monitor's
  //                                  "Memory" column reports for
  //                                  this process
  //   "phys_total_bytes"    uint -- host hw.memsize
  FlexData query();

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}

#endif
