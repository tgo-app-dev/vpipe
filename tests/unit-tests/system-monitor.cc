// vpipe/system-monitor.h: the status bar's readings and the GPU thermal
// verdict, as a host embedding vpipe reads them (no session needed).

#include "minitest.h"
#include "common/flex-data.h"
#include "vpipe/system-monitor.h"

#include <string>

using namespace vpipe;

TEST(system_monitor, status_reads_the_machine)
{
  SystemMonitor m;
  const FlexData s = m.status();
  ASSERT_TRUE(s.is_object());
#if defined(VPIPE_BUILD_APPLE_SILICON)
  if (s.is_object()) {
    const auto o = s.as_object();
    // The machine's memory and this process's footprint are always
    // there; the GPU's and the ANE's depend on what the OS publishes.
    EXPECT_TRUE(o.contains("phys_total_bytes") &&
                o.at("phys_total_bytes").as_uint() > 0);
    EXPECT_TRUE(o.contains("phys_footprint_bytes") &&
                o.at("phys_footprint_bytes").as_uint() > 0);
    if (o.contains("gpu_util_pct")) {
      const double g = o.at("gpu_util_pct").as_real();
      EXPECT_TRUE(g >= 0.0 && g <= 100.0);
    }
  }
  // A second reading is a delta against the first.
  EXPECT_TRUE(m.status().is_object());
#endif
}

TEST(system_monitor, gpu_thermal_says_a_verdict)
{
  const FlexData t = SystemMonitor::gpu_thermal(200);
  ASSERT_TRUE(t.is_object());
  if (t.is_object()) {
    const auto o = t.as_object();
    ASSERT_TRUE(o.contains("verdict") && o.at("verdict").is_string());
    const std::string v(o.at("verdict").as_string());
    EXPECT_TRUE(v == "unknown" || v == "idle" || v == "normal" ||
                v == "warm" || v == "throttled");
    EXPECT_TRUE(o.contains("window_ms") &&
                o.at("window_ms").as_int() == 200);
  }
}
