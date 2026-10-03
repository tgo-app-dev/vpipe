#include "vpipe/system-monitor.h"

#include "common/flex-data.h"

#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "stages/gpu-telemetry.h"
#include "stages/gpu-thermal.h"
#include "stages/system-status.h"
#endif

namespace vpipe {

#if defined(VPIPE_BUILD_APPLE_SILICON)

struct SystemMonitor::Impl {
  SystemStatusPoller poller;
};

SystemMonitor::SystemMonitor() : _impl(std::make_unique<Impl>()) {}

SystemMonitor::~SystemMonitor() = default;

FlexData
SystemMonitor::status()
{
  return _impl->poller.query();
}

FlexData
SystemMonitor::gpu_thermal(int window_ms)
{
  const GpuTelemetry g = GpuTelemetrySampler::sample_once(window_ms);

  ThermalSample s;
  // active_pct, not util_pct: see gpu-telemetry.h. The IOAccelerator
  // utilisation key still reports, it just reports zero.
  s.have_util = g.active_pct.ok;
  s.util_pct  = g.active_pct.avg;
  // The BEST clock seen in the window, not the average. The question is
  // what ceiling the OS allowed, and a run that dipped between kernels
  // still proves the top was reachable; averaging that dip in would
  // report a throttle that is really just a gap in work.
  s.have_clock  = g.freq_mhz.ok && g.ceiling_mhz > 0.0;
  s.clock_mhz   = g.freq_mhz.max;
  s.ceiling_mhz = g.ceiling_mhz;
  s.have_temp   = g.temp_c.ok;
  s.temp_c      = g.temp_c.avg;
  const ThermalReading r = classify_thermal(s);

  FlexData o  = FlexData::make_object();
  auto     oo = o.as_object();
  oo.insert_or_assign(
    "verdict", FlexData::make_string(thermal_verdict_name(r.verdict)));
  oo.insert_or_assign("window_ms", FlexData::make_int(window_ms));
  if (r.have_clock) {
    oo.insert_or_assign("clock_mhz", FlexData::make_real(r.clock_mhz));
    oo.insert_or_assign("ceiling_mhz", FlexData::make_real(r.ceiling_mhz));
  }
  if (g.active_pct.ok) {
    oo.insert_or_assign("gpu_active_pct", FlexData::make_real(r.util_pct));
  }
  if (r.have_temp) {
    oo.insert_or_assign("temp_c", FlexData::make_real(r.temp_c));
  }
  // GPU power, for a person reading the document -- the verdict does not
  // use it. Present only while the Energy Model counter is live: on macOS
  // 27 it is silent for minutes and then publishes everything at once,
  // and the reader drops that lump rather than report it.
  if (g.power_w.ok) {
    oo.insert_or_assign("gpu_power_w", FlexData::make_real(g.power_w.avg));
  }
  // The OS's own answer, alongside ours. Keeping both makes the
  // disagreement visible in a bug report rather than arguable.
  oo.insert_or_assign("os_thermal_state",
                      FlexData::make_string(g.thermal_state));
  return o;
}

#else

struct SystemMonitor::Impl {};

SystemMonitor::SystemMonitor() : _impl(std::make_unique<Impl>()) {}

SystemMonitor::~SystemMonitor() = default;

FlexData
SystemMonitor::status()
{
  return FlexData::make_object();
}

FlexData
SystemMonitor::gpu_thermal(int window_ms)
{
  FlexData o  = FlexData::make_object();
  auto     oo = o.as_object();
  oo.insert_or_assign("verdict", FlexData::make_string("unknown"));
  oo.insert_or_assign("window_ms", FlexData::make_int(window_ms));
  oo.insert_or_assign("note",
                      FlexData::make_string("not an Apple Silicon build"));
  return o;
}

#endif

}
