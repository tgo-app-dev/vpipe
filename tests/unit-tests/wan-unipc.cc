// The UniPC multistep sampler, against the diffusers scheduler.
//
// Verified with a SYNTHETIC velocity rather than a DiT: what is under test
// is the integrator, and a synthetic model makes the trajectory exactly
// reproducible on both sides while removing every other source of
// disagreement. The velocity used is nonlinear AND sigma-dependent
// (0.35x + 0.2 tanh(x) + 0.15 sigma) on purpose -- with a linear model the
// predictor and the corrector can agree by accident, which would let a
// broken corrector pass.
//
// Comparison is per STEP, not just at the end: UniPC carries multistep
// history, so an error in the warmup order or in lower_order_final shows
// up mid-trajectory and can partly cancel by the last step.
//
// Env: VPIPE_WAN_UNIPC_GOLDEN = the golden dir. Skips if unset.

#include "minitest.h"

#include "generative-models/krea2/flow-sampler.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

// The schedule on its own. It is worth separating: a wrong sigma grid and a
// wrong update both show as a wrong trajectory, and only one of them is
// visible here.
TEST(wan_unipc, flow_sigma_schedule)
{
  const char* gd = std::getenv("VPIPE_WAN_UNIPC_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::vector<float> ref = read_f32_(std::string(gd) + "/sigmas.f32");
  if (ref.empty()) { return; }

  FlowSchedulerSpec sc;
  sc.type = "unipc_flow";
  sc.steps = (int)ref.size() - 1;
  sc.shift = 3.0;                     // Wan2.2-I2V flow_shift
  sc.num_train = 1000;
  const std::vector<double> got = sc.sigmas();
  ASSERT_TRUE(got.size() == ref.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::fabs(got[i] - (double)ref[i]));
  }
  std::printf("[wan_unipc] sigma schedule max abs diff = %.3e (%d steps)\n",
              worst, sc.steps);
  // Pure double arithmetic on both sides; the golden is stored f32, so the
  // bar is that storage and nothing more.
  EXPECT_TRUE(worst < 1e-6);
  // The terminal sigma is exactly zero (final_sigmas_type "zero"), and the
  // first is nudged OFF one so log(1 - sigma) stays finite.
  EXPECT_TRUE(got.back() == 0.0);
  EXPECT_TRUE(got.front() < 1.0);
}

TEST(wan_unipc, trajectory_matches_golden)
{
  const char* gd = std::getenv("VPIPE_WAN_UNIPC_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string dir = gd;
  const std::vector<float> x0   = read_f32_(dir + "/x0.f32");
  const std::vector<float> traj = read_f32_(dir + "/traj.f32");
  const std::vector<float> sig  = read_f32_(dir + "/sigmas.f32");
  if (x0.empty() || traj.empty() || sig.empty()) { return; }
  const std::size_t n = x0.size();
  const int steps = (int)sig.size() - 1;
  ASSERT_TRUE(traj.size() == n * (std::size_t)steps);

  FlowSchedulerSpec sc;
  sc.type = "unipc_flow";
  sc.steps = steps;
  sc.shift = 3.0;
  sc.num_train = 1000;
  FlowSamplerSpec sp;
  sp.method = "unipc";
  sp.order = 2;
  sp.solver_bh2 = true;

  FlowSampler s(sp, sc);
  s.reset();
  std::vector<float> x = x0;
  // The same synthetic velocity as the golden.
  auto denoise = [](const std::vector<float>& xx, double sigma) {
    std::vector<float> v(xx.size());
    for (std::size_t k = 0; k < xx.size(); ++k) {
      v[k] = (float)(0.35 * (double)xx[k] + 0.2 * std::tanh((double)xx[k]) +
                     0.15 * sigma);
    }
    return v;
  };

  double worst = 0.0;
  int worst_step = -1;
  for (int i = 0; i < steps; ++i) {
    s.step(i, x, denoise);
    ASSERT_TRUE(x.size() == n);
    const double r = rel_l2_(x.data(), traj.data() + (std::size_t)i * n, n);
    if (r > worst) { worst = r; worst_step = i; }
  }
  std::printf("[wan_unipc] trajectory worst per-step rel-L2 = %.3e (step %d "
              "of %d)\n", worst, worst_step, steps);
  EXPECT_TRUE(worst < 1e-5);
}
