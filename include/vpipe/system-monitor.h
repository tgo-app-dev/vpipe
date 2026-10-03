// vpipe::SystemMonitor -- the machine's live load, and whether it is being
// held back, as vpipe's own front ends show them: the web UI's status bar
// (ANE, GPU and memory) and the macOS app's thermal row.
//
// Exported so a host that embeds vpipe shows the same numbers, read the
// same way, rather than keeping a second copy of the IOKit / IOReport /
// SMC readers and the reasoning about which of their counters are live on
// which release (stages/system-status.h, stages/gpu-telemetry.h and
// stages/gpu-thermal.h have the measurements).
//
// Session-free: IOKit, IOReport and SMC reads only -- no GPU work, no
// model loading -- so it can run beside a generation without perturbing
// the thing it describes.
//
// Apple Silicon builds. Elsewhere status() is an empty object and
// gpu_thermal() says "unknown".
//
// The documents are FlexData (common/flex-data.h, from the SDK).

#ifndef VPIPE_SYSTEM_MONITOR_H
#define VPIPE_SYSTEM_MONITOR_H

#include <memory>

#include "vpipe/export.h"

VPIPE_API_BEGIN

namespace vpipe {

class FlexData;

class SystemMonitor {
public:
  SystemMonitor();
  ~SystemMonitor();

  SystemMonitor(const SystemMonitor&)            = delete;
  SystemMonitor& operator=(const SystemMonitor&) = delete;

  // One reading -- the web UI's /api/system/status. Every key is optional
  // (absent when its source is unavailable):
  //   "gpu_util_pct"          real    IOAccelerator device utilisation
  //   "gpu_renderer_pct", "gpu_tiler_pct"   real
  //   "gpu_in_use_bytes", "gpu_alloc_bytes" uint  its system memory
  //   "gpu_model"             string  e.g. "Apple M5 Pro"
  //   "gpu_cores"             uint
  //   "ane_util_pct"          real    the ANE's estimated utilisation:
  //                                   the larger of its power over a
  //                                   per-chip ceiling and the share of
  //                                   time its fabric floor sat above
  //                                   idle (whichever is live here)
  //   "ane_power_w", "ane_active_pct", "ane_bw_gbps", "ane_units",
  //   "ane_max_w"                     what that estimate came from
  //   "phys_footprint_bytes"  uint    this process's physical footprint
  //                                   (Activity Monitor's "Memory")
  //   "phys_total_bytes"      uint    the machine's memory
  // The ANE's figures are deltas against the previous call (the first
  // against construction): call it at a steady pace -- the web UI polls
  // every second. Thread-safe.
  FlexData status();

  // Whether the GPU is being held back right now -- `vpipe
  // --gpu-thermal`. Samples for `window_ms` (and blocks for it): the
  // clock comes from a pstate residency histogram, so it only exists as
  // a delta over a window. Keys:
  //   "verdict"          string  unknown | idle | normal | warm |
  //                              throttled (busy AND clamped below 85%
  //                              of the rated clock)
  //   "window_ms"        int
  //   "clock_mhz", "ceiling_mhz"  real  the best active clock seen, and
  //                                     the top of the DVFS table
  //   "gpu_active_pct"   real    share of the window out of idle
  //   "temp_c"           real    GPU die temperature
  //   "gpu_power_w"      real    only while the energy counter is live
  //   "os_thermal_state" string  NSProcessInfo's, for comparison
  static FlexData gpu_thermal(int window_ms = 600);

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}

VPIPE_API_END

#endif
