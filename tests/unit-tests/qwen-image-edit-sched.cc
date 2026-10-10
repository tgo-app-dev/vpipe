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

TEST(qwen_image_edit_sched, raw_sigmas_match_diffusers)
{
  // A few-step adapter ships its schedule as the pipeline's `sigmas=`
  // argument (Viggle's Qwen-Image-2.1 turbo: six nodes, no terminal
  // stretch). Reference from the REAL diffusers
  // FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=..., mu=...),
  // with shift_terminal None, then 0.02.
  const std::vector<double> nodes = {1.0, 0.9375, 0.875, 0.75, 0.5, 0.25};
  const std::vector<double> ref_1024 = {
      1.00000000, 0.96255648, 0.92305654, 0.83717018, 0.63151217,
      0.36356997, 0.00000000};
  const std::vector<double> ref_4096 = {
      1.00000000, 0.96775448, 0.93335831, 0.85719192, 0.66675580,
      0.40009630, 0.00000000};
  const std::vector<double> ref_term = {
      1.00000000, 0.94732386, 0.89113444, 0.76670933, 0.45561373,
      0.01999998, 0.00000000};

  FlowSchedulerSpec s = qwen_image_sched();
  s.steps = (int)nodes.size();
  s.base_sigmas = nodes;
  s.shift_terminal = 0.0;
  const double d1 = max_abs_diff(s.sigmas(1024), ref_1024);
  const double d2 = max_abs_diff(s.sigmas(4096), ref_4096);
  ASSERT_TRUE(s.sigmas(1024).size() == ref_1024.size());
  s.shift_terminal = 0.02;
  const double d3 = max_abs_diff(s.sigmas(4096), ref_term);
  std::printf("[qwen_image_edit_sched] raw nodes max|d| vs diffusers: "
              "img1024=%.2e img4096=%.2e terminal=%.2e\n", d1, d2, d3);
  EXPECT_TRUE(d1 < 1e-6);
  EXPECT_TRUE(d2 < 1e-6);
  EXPECT_TRUE(d3 < 1e-6);

  // Nodes whose count is not the step count are not half-applied: the
  // default grid runs, exactly as with none.
  FlowSchedulerSpec wrong = qwen_image_sched();
  FlowSchedulerSpec none = wrong;
  wrong.base_sigmas = nodes;   // 6 nodes, 8 steps
  EXPECT_TRUE(max_abs_diff(wrong.sigmas(4096), none.sigmas(4096)) == 0.0);
  // And they are part of the spec's identity.
  EXPECT_TRUE(!(wrong == none));
}

// The same nodes with dynamic shifting OFF -- what a distilled checkpoint
// ships (Qwen-Image-2.1-Turbo: scheduler_config use_dynamic_shifting
// false, shift 1.0, shift_terminal null, and its nodes in model_index.json
// `sample_sigmas`). diffusers then applies the STATIC shift
// s' = shift*s / (1 + (shift-1)*s) -- the "linear" curve here -- and the
// terminal stretch if one is configured. References from the REAL diffusers
// 0.40 FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=..., mu=...),
// whose mu the static branch ignores.
TEST(qwen_image_edit_sched, raw_sigmas_static_match_diffusers)
{
  const std::vector<double> turbo = {1.0,      0.978453, 0.95418,
                                     0.926626, 0.89508,  0.845148,
                                     0.704534, 0.414568};
  FlowSchedulerSpec s;
  s.dynamic_shift = false;
  s.shift_type = "linear";
  s.shift = 1.0;
  s.shift_terminal = 0.0;
  s.steps = (int)turbo.size();
  s.base_sigmas = turbo;
  // Shift 1, no stretch: the nodes AS WRITTEN, then the terminal 0 --
  // at any image size, since no mu is involved.
  std::vector<double> want = turbo;
  want.push_back(0.0);
  EXPECT_TRUE(max_abs_diff(s.sigmas(1024), want) == 0.0);
  EXPECT_TRUE(max_abs_diff(s.sigmas(16384), want) == 0.0);

  const std::vector<double> nodes = {1.0, 0.9375, 0.875, 0.75, 0.5, 0.25};
  const std::vector<double> ref_shift3 = {
      1.0, 0.97826087, 0.95454544, 0.89999998, 0.75, 0.5, 0.0};
  const std::vector<double> ref_shift3_term = {
      1.0, 0.95739132, 0.91090906, 0.80399996, 0.50999999, 0.01999998, 0.0};
  FlowSchedulerSpec t = s;
  t.shift = 3.0;
  t.steps = (int)nodes.size();
  t.base_sigmas = nodes;
  const double d1 = max_abs_diff(t.sigmas(4096), ref_shift3);
  t.shift_terminal = 0.02;
  const double d2 = max_abs_diff(t.sigmas(4096), ref_shift3_term);
  std::printf("[qwen_image_edit_sched] static raw nodes max|d| vs diffusers: "
              "shift3=%.2e shift3+terminal=%.2e\n", d1, d2);
  EXPECT_TRUE(d1 < 1e-6);
  EXPECT_TRUE(d2 < 1e-6);

  // Without nodes and without a terminal, the static path is what it was:
  // the simple grid, shifted.
  FlowSchedulerSpec plain;
  plain.steps = 8;
  FlowSchedulerSpec with_term = plain;
  with_term.shift_terminal = 0.0;
  EXPECT_TRUE(max_abs_diff(plain.sigmas(4096), with_term.sigmas(4096)) ==
              0.0);
}
