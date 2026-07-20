// Qwen-Image-Edit-2511 scheduler: the FlowMatchEulerDiscreteScheduler dynamic-
// shifting path added to FlowSchedulerSpec. Unlike the Krea-2 static-shift
// schedule (linspace(1, 1/steps) + a constant mu), Qwen-Image computes mu
// PER-IMAGE from the packed image-token count and stretches the schedule so
// its last nonzero sigma lands on shift_terminal.
//
//  * dynamic sigmas reproduce diffusers set_timesteps(use_dynamic_shifting)
//    for two image-seq lengths (reference computed from the documented
//    algorithm; the golden dumper cross-checks against real diffusers).
//  * schedule invariants: sigma[0] = 1, terminal 0, last nonzero =
//    shift_terminal, strictly decreasing; larger img_seq => larger mu => a
//    schedule that stays higher longer.
//  * the new dynamic-shift fields round-trip through FlexData.
//
// CPU-only -- no model or metal needed.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/krea2/flow-sampler.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vpipe;
using genai::FlowSchedulerSpec;

namespace {

// A dynamic-shift spec matching Qwen-Image-Edit-2511's scheduler_config.json.
FlowSchedulerSpec
qwen_image_sched()
{
  FlowSchedulerSpec s;
  s.dynamic_shift  = true;
  s.steps          = 8;
  s.shift_type     = "exponential";
  s.base_shift     = 0.5;
  s.max_shift      = 0.9;
  s.shift_terminal = 0.02;
  s.base_seq       = 256;
  s.max_seq        = 8192;
  s.num_train      = 1000;
  return s;
}

double
max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
{
  double m = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

}  // namespace

TEST(qwen_image_edit_sched, dynamic_sigmas_match_diffusers)
{
  // Reference sigmas dumped from the REAL diffusers
  // FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(1,1/8,8),
  // mu=calculate_shift(img_seq)) with the model's scheduler_config.json --
  // exactly as QwenImageEditPlusPipeline calls it. (The last nonzero sigma is
  // 0.02 up to diffusers' own float32 drift ~4e-8; our double path is exact.)
  const std::vector<double> ref_1024 = {
      1.00000000, 0.90613425, 0.80135882, 0.68365431, 0.55047023,
      0.39853805, 0.22359914, 0.02000004, 0.00000000};
  const std::vector<double> ref_4096 = {
      1.00000000, 0.91602397, 0.82004577, 0.70929456, 0.58007485,
      0.42734694, 0.24405390, 0.01999998, 0.00000000};

  const FlowSchedulerSpec s = qwen_image_sched();
  const std::vector<double> got_1024 = s.sigmas(1024);
  const std::vector<double> got_4096 = s.sigmas(4096);
  ASSERT_TRUE(got_1024.size() == ref_1024.size());
  ASSERT_TRUE(got_4096.size() == ref_4096.size());

  const double d1 = max_abs_diff(got_1024, ref_1024);
  const double d2 = max_abs_diff(got_4096, ref_4096);
  std::printf("[qwen_image_edit_sched] max|d| vs diffusers: img1024=%.2e "
              "img4096=%.2e\n", d1, d2);
  EXPECT_TRUE(d1 < 1e-5);
  EXPECT_TRUE(d2 < 1e-5);

  // The img_seq_len runtime binding is bit-identical to the explicit overload.
  FlowSchedulerSpec sb = s;
  sb.img_seq_len = 1024;
  EXPECT_TRUE(max_abs_diff(sb.sigmas(), s.sigmas(1024)) == 0.0);
}

TEST(qwen_image_edit_sched, schedule_invariants)
{
  const FlowSchedulerSpec s = qwen_image_sched();
  for (int iseq : {256, 1024, 4096, 8192}) {
    const std::vector<double> sig = s.sigmas(iseq);
    ASSERT_TRUE((int)sig.size() == s.steps + 1);
    EXPECT_TRUE(std::abs(sig[0] - 1.0) < 1e-9);               // sigma[0] = 1
    EXPECT_TRUE(sig[sig.size() - 1] == 0.0);                  // terminal 0
    // Last nonzero sigma stretched exactly onto shift_terminal.
    EXPECT_TRUE(std::abs(sig[sig.size() - 2] - s.shift_terminal) < 1e-9);
    bool dec = true;
    for (std::size_t i = 1; i + 1 < sig.size(); ++i) {
      if (sig[i] >= sig[i - 1]) { dec = false; }
    }
    EXPECT_TRUE(dec);
  }
  // Larger img_seq -> larger mu -> schedule holds higher sigma at mid-step.
  const std::vector<double> lo = s.sigmas(1024), hi = s.sigmas(4096);
  EXPECT_TRUE(hi[4] > lo[4]);
}

TEST(qwen_image_edit_sched, dynamic_fields_roundtrip)
{
  const FlowSchedulerSpec s = qwen_image_sched();
  const FlowSchedulerSpec r = FlowSchedulerSpec::from_flex(s.to_flex());
  EXPECT_TRUE(r == s);
  EXPECT_TRUE(r.dynamic_shift);
  EXPECT_TRUE(std::abs(r.shift_terminal - 0.02) < 1e-12);
  EXPECT_TRUE(r.max_seq == 8192);

  // A non-dynamic spec does not emit the dynamic keys but still round-trips.
  FlowSchedulerSpec stat;   // default: simple, static
  EXPECT_TRUE(FlowSchedulerSpec::from_flex(stat.to_flex()) == stat);
  EXPECT_TRUE(!stat.dynamic_shift);
}
