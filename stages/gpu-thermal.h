#ifndef VPIPE_STAGES_GPU_THERMAL_H
#define VPIPE_STAGES_GPU_THERMAL_H

#include <string>

namespace vpipe {

// Is this machine being thermally held back RIGHT NOW?
//
// WHY NOT ProcessInfo.thermalState. That is the only thermal signal macOS
// offers an app, and on the machine this matters most for -- a fanless
// MacBook Air under a long generation -- it reads `nominal` while the GPU
// clock is pinned at roughly two thirds of the chip's ceiling. It is a
// coarse advisory about what an APP should voluntarily stop doing, not a
// report of what the hardware is doing, and it moves late or not at all.
// Someone watching a run get slower gets no explanation from it.
//
// WHAT THIS USES INSTEAD. The GPU publishes a DVFS table (the pstate
// frequency ladder) and a residency histogram over those states. The
// residency-weighted clock ACROSS ACTIVE STATES ONLY -- the idle state
// excluded -- is the speed the GPU ran at while it was running, and the
// top of the table is the speed the silicon is rated for. On M4/M5 that
// ceiling is ~1578 MHz; a heat-soaked Air holds ~1000.
//
// THE GATE IS UTILISATION, and it is the whole reason this is not a
// simple clock threshold. An idle GPU sits at a low pstate because there
// is nothing to do, which is DVFS working, not throttling. Only a clock
// held down WHILE WORK IS QUEUED means the machine could go faster and is
// not being allowed to.
enum class ThermalVerdict {
  Unknown,      // no usable telemetry
  Idle,         // GPU not loaded -- a low clock here says nothing
  Normal,       // loaded, running at or near the ceiling
  Warm,         // hot, but not yet clamped
  Throttled,    // loaded AND clamped: the finding this exists for
};

// Inputs, all optional so a missing source degrades rather than lies.
struct ThermalSample {
  bool   have_util = false;
  double util_pct  = 0.0;    // GPU utilisation over the window, [0,100]

  bool   have_clock = false;
  double clock_mhz  = 0.0;   // best active clock seen in the window
  double ceiling_mhz = 0.0;  // top of the DVFS table, 0 if unknown

  bool   have_temp = false;
  double temp_c    = 0.0;    // GPU die temperature
};

// Tunables, named so the reasoning is reviewable rather than buried.
struct ThermalThresholds {
  // Utilisation at or above which a low clock is evidence rather than
  // noise. Deliberately not 95%: a streaming model alternates between
  // reading weights and computing, so a genuinely busy generation
  // averages well under full occupancy over a multi-second window.
  double busy_util_pct = 70.0;

  // Clock fraction of the ceiling below which a busy GPU counts as
  // clamped. The gap this has to straddle is wide -- a heat-soaked Air
  // holds ~0.63 of the ceiling -- and the risk on the other side is
  // calling a memory-bound kernel "throttled" because it never asked
  // for the top pstate. 0.85 sits well clear of both.
  double clamp_fraction = 0.85;

  // Hot enough to report even when the clock has not moved yet. The die
  // tops out around 101 C on the machine this targets.
  double warm_temp_c = 95.0;
};

// The verdict, plus what it was based on -- callers show the numbers,
// because "Throttled" alone is the kind of claim a user should be able
// to check.
struct ThermalReading {
  ThermalVerdict verdict = ThermalVerdict::Unknown;
  double clock_mhz   = 0.0;
  double ceiling_mhz = 0.0;
  double util_pct    = 0.0;
  double temp_c      = 0.0;
  bool   have_clock  = false;
  bool   have_temp   = false;
};

// Pure: no I/O, no platform calls, so the interesting cases can be
// exercised on a machine that cannot produce them. The box this was
// written on has a fan and never throttles; every threshold below was
// checked against synthesised samples instead.
inline ThermalReading
classify_thermal(const ThermalSample&      s,
                 const ThermalThresholds&  t = ThermalThresholds{})
{
  ThermalReading r;
  r.clock_mhz   = s.clock_mhz;
  r.ceiling_mhz = s.ceiling_mhz;
  r.util_pct    = s.util_pct;
  r.temp_c      = s.temp_c;
  r.have_clock  = s.have_clock && s.ceiling_mhz > 0.0;
  r.have_temp   = s.have_temp;

  const bool hot = s.have_temp && s.temp_c >= t.warm_temp_c;

  // No utilisation reading means no gate, and without the gate a low
  // clock is indistinguishable from an idle one. Report the temperature
  // if there is one and stop there -- a guess would be worse than a gap.
  if (!s.have_util) {
    r.verdict = hot ? ThermalVerdict::Warm : ThermalVerdict::Unknown;
    return r;
  }

  if (s.util_pct < t.busy_util_pct) {
    // Hot while idle still deserves saying: it is what the minute after
    // a long run looks like, and it predicts the next run being slower.
    r.verdict = hot ? ThermalVerdict::Warm : ThermalVerdict::Idle;
    return r;
  }

  if (!r.have_clock) {
    r.verdict = hot ? ThermalVerdict::Warm : ThermalVerdict::Normal;
    return r;
  }

  if (s.clock_mhz <= t.clamp_fraction * s.ceiling_mhz) {
    r.verdict = ThermalVerdict::Throttled;
    return r;
  }
  r.verdict = hot ? ThermalVerdict::Warm : ThermalVerdict::Normal;
  return r;
}

inline const char*
thermal_verdict_name(ThermalVerdict v)
{
  switch (v) {
    case ThermalVerdict::Unknown:   return "unknown";
    case ThermalVerdict::Idle:      return "idle";
    case ThermalVerdict::Normal:    return "normal";
    case ThermalVerdict::Warm:      return "warm";
    case ThermalVerdict::Throttled: return "throttled";
  }
  return "unknown";
}

}  // namespace vpipe

#endif
