#ifndef VPIPE_STAGES_GPU_TELEMETRY_H
#define VPIPE_STAGES_GPU_TELEMETRY_H

#include <cstdint>
#include <memory>
#include <string>

namespace vpipe {

// One metric's aggregate over a sampling window.
struct MetricAgg {
  bool   ok  = false;   // any valid sample collected?
  double min = 0.0;
  double avg = 0.0;
  double max = 0.0;
  int    n   = 0;
};

// GPU telemetry aggregated over a benchmark region.
struct GpuTelemetry {
  MetricAgg   freq_mhz;        // GPU clock (residency-weighted active)
  MetricAgg   temp_c;          // GPU die temperature
  MetricAgg   util_pct;        // GPU utilization
  MetricAgg   power_w;         // GPU power
  MetricAgg   footprint_mb;    // process phys-footprint (RAM), in MB
  std::string thermal_state;   // coarse OS thermal-pressure label
};

// Samples GPU metrics on a background thread while a region runs, then
// aggregates min/avg/max. Construction resolves the IOReport / IOKit /
// SMC handles once and never throws -- any unavailable source degrades
// that metric to ok=false. Sources (Apple Silicon):
//   power -- IOReport "Energy Model", channels prefixed "GPU"
//   util  -- IOAccelerator PerformanceStatistics "Device Utilization %"
//   freq  -- IOReport "GPU Stats" pstate residency, weighted by the
//            "voltage-states9" MHz table read from the pmgr IORegistry node
//   temp  -- AppleSMC GPU sensor keys ("Tg..")
//   footprint -- task_info phys_footprint (whole-process RAM; on the UMA
//            this includes the wired model weights + KV + activations)
//   thermal_state -- NSProcessInfo.thermalState (always available)
class GpuTelemetrySampler {
public:
  GpuTelemetrySampler();
  ~GpuTelemetrySampler();

  GpuTelemetrySampler(const GpuTelemetrySampler&)            = delete;
  GpuTelemetrySampler& operator=(const GpuTelemetrySampler&) = delete;

  // Spawn the sampling thread and reset aggregates.
  void start();
  // Join the thread and return the aggregated metrics.
  GpuTelemetry stop();

  // Static descriptors (from the IOAccelerator registry entry).
  static std::string   gpu_model();        // e.g. "Apple M4 Pro"
  static std::uint64_t gpu_core_count();   // e.g. 20

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace vpipe

#endif
