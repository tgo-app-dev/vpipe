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
// asitop / mactop) read. ANE utilisation comes from the IOReport private
// framework macmon uses, read two ways because on a given machine one
// of them may be dead (common/soc-activity.h has the measurements):
//
//   * power: an "Energy Model" delta over a per-chip ceiling (8.0 W for
//     M1/M2/M4, 8.5 W for M3 -- macmon's table), the way macmon does
//     it. Reported only while that counter is live; on macOS 27 it
//     publishes minutes of energy at once and is silent in between.
//   * activity: the share of the interval the ANE's fabric floor (group
//     "PMP"/"PMP<n>", subgroup "SOC Floor") sat above VMIN, which the
//     power manager updates every second.
//
//     ane_util_pct = max(100 * ane_power_W / ane_max_W, ane_active_pct)
//
// clipped to [0,100]. A dead source contributes nothing, so the max is
// whichever one is live.
//
// State (held across queries):
//   * one IOReport subscription: the "Energy Model" group plus the PMP
//     "SOC Floor" and "AF BW" channels
//   * the previous sample, used to compute the per-second delta on the
//     next query
//   * whether the ANE's own energy channels are live (SocEnergyGate)
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
  //   "ane_util_pct"        real -- estimated ANE util [0,100]: the
  //                                 larger of the two readings below
  //   "ane_power_w"         real -- ANE power, watts; ONLY while the
  //                                 Energy Model counter is live
  //   "ane_active_pct"      real -- share of the interval the ANE's
  //                                 fabric floor sat above VMIN [0,100]
  //   "ane_bw_gbps"         real -- the ANE's fabric bandwidth, GB/s
  //                                 (absent while it is idle)
  //   "ane_units"           int  -- ANE energy channels seen
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
