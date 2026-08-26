// The throttle verdict (stages/gpu-thermal.h).
//
// This file exists because the interesting case cannot be produced on the
// machine that develops it. A fan-cooled Mac mini does not hold its GPU
// at two thirds of the ceiling for minutes; a fanless MacBook Air under a
// long generation does. So the classifier is pure and the samples are
// synthesised, with the numbers taken from what the Air actually reports:
// a 1578 MHz ceiling held at ~1000, and a die around 101 C.
//
// The negative cases carry as much weight as the positive one. An
// indicator that cries throttle at an idle GPU -- which sits at a low
// pstate because there is nothing to do -- is worse than no indicator,
// because the first time it is wrong is the last time anyone reads it.

#include "minitest.h"
#include "stages/gpu-thermal.h"

using namespace vpipe;

namespace {

// The M4/M5 ladder, and the figures reported from a heat-soaked Air.
constexpr double kCeiling   = 1578.0;
constexpr double kThrottled = 1000.0;
constexpr double kHotDie    = 101.0;

ThermalSample loaded_at(double mhz, double util = 95.0)
{
  ThermalSample s;
  s.have_util   = true;
  s.util_pct    = util;
  s.have_clock  = true;
  s.clock_mhz   = mhz;
  s.ceiling_mhz = kCeiling;
  return s;
}

}  // namespace

// THE CASE THIS EXISTS FOR: busy, and the clock is capped well under what
// the silicon is rated for. 1000/1578 is 0.634.
TEST(gpu_thermal, busy_and_clamped_is_throttled)
{
  const auto r = classify_thermal(loaded_at(kThrottled));
  EXPECT_TRUE(r.verdict == ThermalVerdict::Throttled);
  EXPECT_TRUE(r.have_clock);
}

// Hot AND clamped is still Throttled, not merely Warm -- the clamp is the
// more actionable fact, and temperature is corroboration.
TEST(gpu_thermal, hot_and_clamped_still_reads_throttled)
{
  auto s = loaded_at(kThrottled);
  s.have_temp = true;
  s.temp_c    = kHotDie;
  EXPECT_TRUE(classify_thermal(s).verdict == ThermalVerdict::Throttled);
}

// THE FALSE POSITIVE THAT MATTERS. An idle GPU is at a low pstate for the
// ordinary reason. Same clock as the throttled case, opposite verdict --
// which is what makes utilisation the gate rather than a refinement.
TEST(gpu_thermal, idle_at_the_same_low_clock_is_not_throttled)
{
  const auto r = classify_thermal(loaded_at(kThrottled, /*util=*/3.0));
  EXPECT_TRUE(r.verdict == ThermalVerdict::Idle);
  EXPECT_FALSE(r.verdict == ThermalVerdict::Throttled);
}

// A run at the ceiling is Normal however long it has been going.
TEST(gpu_thermal, busy_at_the_ceiling_is_normal)
{
  EXPECT_TRUE(classify_thermal(loaded_at(kCeiling)).verdict
              == ThermalVerdict::Normal);
}

// One pstate down under load is NOT throttling. This is the other false
// positive: a memory-bound kernel need never ask for the top state.
TEST(gpu_thermal, one_step_below_the_ceiling_is_normal)
{
  EXPECT_TRUE(classify_thermal(loaded_at(1450.0)).verdict
              == ThermalVerdict::Normal);
}

// Straddling the threshold from both sides, so the constant is pinned
// rather than merely present. 0.85 * 1578 = 1341.3.
TEST(gpu_thermal, the_clamp_threshold_is_where_it_claims_to_be)
{
  EXPECT_TRUE(classify_thermal(loaded_at(1345.0)).verdict
              == ThermalVerdict::Normal);
  EXPECT_TRUE(classify_thermal(loaded_at(1335.0)).verdict
              == ThermalVerdict::Throttled);
}

// Hot but keeping its clock: worth saying, not worth calling throttled.
TEST(gpu_thermal, hot_but_unclamped_is_warm)
{
  auto s = loaded_at(kCeiling);
  s.have_temp = true;
  s.temp_c    = kHotDie;
  EXPECT_TRUE(classify_thermal(s).verdict == ThermalVerdict::Warm);
}

// Hot while idle is Warm too -- that is the minute after a long run, and
// it predicts the next one starting slow.
TEST(gpu_thermal, hot_while_idle_is_warm)
{
  auto s = loaded_at(kCeiling, /*util=*/2.0);
  s.have_temp = true;
  s.temp_c    = kHotDie;
  EXPECT_TRUE(classify_thermal(s).verdict == ThermalVerdict::Warm);
}

// Degrade, do not guess. Without utilisation there is no gate, and a low
// clock could be either case.
TEST(gpu_thermal, no_utilisation_reading_refuses_to_decide)
{
  ThermalSample s;
  s.have_clock  = true;
  s.clock_mhz   = kThrottled;
  s.ceiling_mhz = kCeiling;
  EXPECT_TRUE(classify_thermal(s).verdict == ThermalVerdict::Unknown);
}

// A busy GPU with no clock source is Normal, not Throttled: absence of
// evidence must not read as evidence.
TEST(gpu_thermal, no_clock_source_does_not_manufacture_a_throttle)
{
  ThermalSample s;
  s.have_util = true;
  s.util_pct  = 99.0;
  EXPECT_TRUE(classify_thermal(s).verdict == ThermalVerdict::Normal);
}

// An unreadable DVFS table leaves the ceiling at 0, and a fraction of
// zero would make every clock "at or below" it -- i.e. always throttled.
TEST(gpu_thermal, an_unknown_ceiling_is_not_a_zero_ceiling)
{
  ThermalSample s;
  s.have_util   = true;
  s.util_pct    = 99.0;
  s.have_clock  = true;
  s.clock_mhz   = 1578.0;
  s.ceiling_mhz = 0.0;
  const auto r = classify_thermal(s);
  EXPECT_TRUE(r.verdict == ThermalVerdict::Normal);
  EXPECT_FALSE(r.have_clock);
}

// The reading carries its inputs, because the UI shows them and a claim
// the user cannot check is one they cannot act on.
TEST(gpu_thermal, the_reading_reports_what_it_decided_from)
{
  auto s = loaded_at(kThrottled, /*util=*/88.0);
  s.have_temp = true;
  s.temp_c    = kHotDie;
  const auto r = classify_thermal(s);
  EXPECT_TRUE(r.clock_mhz == kThrottled);
  EXPECT_TRUE(r.ceiling_mhz == kCeiling);
  EXPECT_TRUE(r.util_pct == 88.0);
  EXPECT_TRUE(r.temp_c == kHotDie);
}

// The busy gate is a threshold too, and a streaming model alternating
// between disk and compute sits below full occupancy over a window.
TEST(gpu_thermal, the_busy_gate_is_where_it_claims_to_be)
{
  EXPECT_TRUE(classify_thermal(loaded_at(kThrottled, 71.0)).verdict
              == ThermalVerdict::Throttled);
  EXPECT_TRUE(classify_thermal(loaded_at(kThrottled, 69.0)).verdict
              == ThermalVerdict::Idle);
}
