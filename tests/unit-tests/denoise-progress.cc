// The denoise progress mapping: DiT block callbacks -> one bar over the
// whole loop.
//
// Pure arithmetic, no model -- which is the point. Wiring it into a stage
// is three lines the compiler checks; what is easy to get WRONG is the
// accounting, and it goes wrong in ways that are hard to describe from
// watching a bar: it creeps past 100%, it stalls at a step edge, or it
// stops at 87% just as the user is waiting for the result. Each of those
// is one assertion here.
//
// The block count is not known until the first callback (it comes from
// the model, not the config), so every case has to survive learning it
// late -- including never learning it at all.

#include "minitest.h"

#include "interfaces/ui-delegate-intf.h"
#include "stages/denoise-progress.h"

#include <cstdint>
#include <memory>
#include <vector>

using namespace vpipe;

namespace {

// The bar's current fraction, or -1 when it is indeterminate/absent.
double
frac_(const UiProgressRegistry& reg, std::uint64_t id)
{
  for (const auto& it : reg.snapshot()) {
    if (it.id == id) {
      return it.total == 0 ? -1.0 : (double)it.done / (double)it.total;
    }
  }
  return -1.0;
}

// Drive `prog` through a whole loop -- `fwds` forwards per step, `blocks`
// blocks per forward -- sampling the bar after every call, which is what
// gives us the history the real registry does not keep.
struct Run {
  std::vector<double> seen;
  bool monotonic = true;
  bool bounded   = true;
  double last    = -1.0;

  void sample(const UiProgressRegistry& reg, std::uint64_t id)
  {
    const double f = frac_(reg, id);
    if (f < 0.0) { return; }
    if (!seen.empty() && f < seen.back() - 1e-12) { monotonic = false; }
    if (f > 1.0 + 1e-12) { bounded = false; }
    seen.push_back(f);
    last = f;
  }
};

void
run_loop_(DenoiseProgress& prog, const UiProgressRegistry& reg,
          std::uint64_t id, Run& r, int steps, int fwds, int blocks)
{
  for (int s = 0; s < steps; ++s) {
    for (int f = 0; f < fwds; ++f) {
      for (int b = 0; b < blocks; ++b) {
        prog.block_fn()(b, blocks);
        r.sample(reg, id);
      }
      prog.end_forward();
    }
    prog.end_step(s);
    r.sample(reg, id);
  }
}

}  // namespace

TEST(denoise_progress, ends_at_exactly_one)
{
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, /*steps=*/8, /*forwards_per_step=*/1);
  Run r;
  run_loop_(prog, *reg, id, r, 8, 1, 30);
  EXPECT_TRUE(!r.seen.empty());
  // Stopping at 29/30 of the last step reads as a hang at the exact
  // moment the user is waiting for the output.
  EXPECT_TRUE(r.last == 1.0);
}

// The configured step count is not always the count that RUNS.
// MiniMax-H3's `steps` is a sigma grid including the terminal zero, so it
// drives one fewer evaluation, and the shift can collapse duplicates on
// top of that. The sampler reports what it settled on; without adopting
// it the bar divides by the wrong denominator and stops short -- MEASURED
// at 74% on a 4-step H3 run, right where the user is waiting for output.
TEST(denoise_progress, adopts_the_samplers_real_step_count)
{
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  // Told 4; the scheduler runs 3.
  DenoiseProgress prog(&bar, /*steps=*/4, /*forwards_per_step=*/1);
  Run r;
  prog.set_steps(3);
  run_loop_(prog, *reg, id, r, 3, 1, 50);
  EXPECT_TRUE(r.monotonic);
  EXPECT_TRUE(r.bounded);
  EXPECT_TRUE(r.last == 1.0);

  // And adopting it MID-RUN still lands on 1.0: the totals are recomputed
  // per update, so a callback that only learns the real count on its
  // first fire is not too late.
  auto reg2 = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id2 = reg2->open("denoise");
  UiProgress bar2(reg2, id2);
  DenoiseProgress prog2(&bar2, /*steps=*/40, /*forwards_per_step=*/1);
  Run r2;
  prog2.block_fn()(0, 50);            // one block tick at the wrong total
  prog2.set_steps(3);
  run_loop_(prog2, *reg2, id2, r2, 3, 1, 50);
  EXPECT_TRUE(r2.bounded);
  EXPECT_TRUE(r2.last == 1.0);
}

TEST(denoise_progress, never_exceeds_one_or_goes_backwards)
{
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, /*steps=*/4, /*forwards_per_step=*/2);
  Run r;
  run_loop_(prog, *reg, id, r, 4, 2, 20);
  EXPECT_TRUE(r.monotonic);
  EXPECT_TRUE(r.bounded);
  EXPECT_TRUE(r.last == 1.0);
}

TEST(denoise_progress, an_extra_forward_is_clamped_then_resynced)
{
  // A sampler that calls the denoise function more often than the stage
  // sized for. Without the clamp the bar runs past 100% mid-step, which
  // reads as a bug even though the step boundary corrects it.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, /*steps=*/3, /*forwards_per_step=*/1);
  Run r;
  run_loop_(prog, *reg, id, r, 3, /*fwds=*/2, 10);   // TWO, sized for one
  EXPECT_TRUE(r.bounded);
  EXPECT_TRUE(r.last == 1.0);
}

TEST(denoise_progress, a_step_boundary_is_exact)
{
  // Between boundaries the bar interpolates over blocks; AT a boundary it
  // must land on the true fraction whatever the forwards did. That is
  // what lets a guidance-distilled model (one forward per step, H3) and a
  // guided one (two, Wan) share this mapping.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  const int steps = 5;
  DenoiseProgress prog(&bar, steps, 1);
  bool exact = true;
  for (int s = 0; s < steps; ++s) {
    for (int b = 0; b < 16; ++b) { prog.block_fn()(b, 16); }
    prog.end_forward();
    prog.end_step(s);
    const double want = (double)(s + 1) / (double)steps;
    const double got  = frac_(*reg, id);
    if (got < want - 1e-9 || got > want + 1e-9) { exact = false; }
  }
  EXPECT_TRUE(exact);
}

TEST(denoise_progress, reports_steps_when_no_block_callback_ever_fires)
{
  // A model with no block hook, or one whose forward failed before the
  // first block. The bar must still move: silence here is the exact
  // failure this mapping exists to remove.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, /*steps=*/4, /*forwards_per_step=*/1);
  for (int s = 0; s < 4; ++s) { prog.end_step(s); }
  EXPECT_TRUE(frac_(*reg, id) == 1.0);
}

// The hook must be CLEARED when the scope ends. It captures a pointer to
// a stack local while the model is a stage member that outlives it, so a
// hook left installed is called by the next generation with a dangling
// DenoiseProgress -- and the denoise paths have several early returns
// where clearing it by hand is what gets missed.
namespace {
struct FakeDit {
  genai::DitBlockProgressFn fn;
  void set_block_progress(genai::DitBlockProgressFn f) { fn = std::move(f); }
};
}  // namespace

TEST(denoise_progress, scoped_hook_installs_and_clears)
{
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, 2, 1);
  FakeDit dit;
  {
    ScopedBlockProgress<FakeDit> hook(&dit, prog);
    EXPECT_TRUE((bool)dit.fn);
  }
  EXPECT_TRUE(!dit.fn);
}

TEST(denoise_progress, rearm_moves_the_hook_and_clears_the_old_model)
{
  // Wan swaps noise experts mid-loop: crossing the boundary destroys one
  // transformer and builds the other. Without re-arming, every block
  // after the crossing is silent -- the bar freezes for the whole second
  // half of the denoise, which is precisely when the user is watching.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, 2, 1);
  FakeDit high, low;
  {
    ScopedBlockProgress<FakeDit> hook(&high, prog);
    EXPECT_TRUE((bool)high.fn);
    hook.rearm(&low);
    EXPECT_TRUE(!high.fn);          // the old expert let go of it
    EXPECT_TRUE((bool)low.fn);      // the new one has it
    hook.rearm(&low);               // idempotent
    EXPECT_TRUE((bool)low.fn);
  }
  EXPECT_TRUE(!low.fn);             // and the scope still clears
}
