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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
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

// Stage and publish in one go -- what report_block() does when there is
// no command buffer to defer the publish onto, and what a plugin's own
// loop does. The two-phase cases below drive the halves separately.
void
fire_(DenoiseProgress& prog, int done, int total)
{
  if (auto publish = prog.block_fn()(done, total)) { publish(); }
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
        fire_(prog, b, blocks);
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
  fire_(prog2, 0, 50);                // one block tick at the wrong total
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
    for (int b = 0; b < 16; ++b) { fire_(prog, b, 16); }
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

// ---- reporting on the GPU's clock ------------------------------------
//
// The publish half is delivered from a Metal command-buffer completion
// handler (generative-models/shared/dit-gpu-progress.h), which changes
// two things this mapping never had to survive: it arrives on a foreign
// thread, and it can arrive LATE -- after the step it belongs to has been
// re-synced past, or after the stack local that owns the bar has gone.
//
// Neither is reachable from a stage, which is why they are asserted here:
// the shapes below are what a completion handler does, written out by
// hand -- stage inside the forward, publish whenever.

TEST(denoise_progress, a_late_publish_lands_where_it_was_staged)
{
  // The point of staging on the forward's thread. Hold the publishes for
  // a whole forward, cross the step boundary, and only then let them go:
  // they must report the positions they were STAGED at, which the bar has
  // already passed, and so change nothing. Resolving at publish time
  // instead would read the new step's base and jump the bar most of a
  // step ahead of the truth.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  DenoiseProgress prog(&bar, /*steps=*/4, /*forwards_per_step=*/1);
  auto fn = prog.block_fn();

  std::vector<std::function<void()>> held;
  for (int b = 0; b < 10; ++b) { held.push_back(fn(b, 10)); }
  prog.end_forward();
  prog.end_step(0);
  const double after_step = frac_(*reg, id);
  EXPECT_TRUE(after_step == 0.25);

  for (auto& pub : held) { if (pub) { pub(); } }
  EXPECT_TRUE(frac_(*reg, id) == after_step);

  // A block staged in the NEW forward still moves it: 10 + 5 of 40.
  if (auto pub = fn(5, 10)) { pub(); }
  EXPECT_TRUE(frac_(*reg, id) == 0.375);
}

TEST(denoise_progress, a_publish_after_the_reporter_dies_is_inert)
{
  // The forward's last commit is waited for, but the completion handler
  // runs on a Metal thread and is not ordered against that wait -- so the
  // final block can publish after the DenoiseProgress (a stack local) has
  // gone. The closure holds the state by shared_ptr and the destructor
  // cuts the bar loose, so this is a no-op rather than a write through a
  // dangling pointer.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  std::function<void()> orphan;
  {
    UiProgress bar(reg, id);
    DenoiseProgress prog(&bar, /*steps=*/2, /*forwards_per_step=*/1);
    auto fn = prog.block_fn();
    if (auto pub = fn(4, 8)) { pub(); }
    EXPECT_TRUE(frac_(*reg, id) == 0.25);
    orphan = fn(7, 8);              // staged, never published
  }
  // Both the bar and the progress are gone; the closure is not.
  if (orphan) { orphan(); }
  EXPECT_TRUE(true);                // reaching here without a fault is it
}

TEST(denoise_progress, concurrent_publishes_stay_monotonic)
{
  // Publishes for different command buffers are delivered by Metal, not
  // by us, so nothing in this process serialises them against each other
  // or against the denoise loop's own end_step(). The bar must come out
  // monotonic and in range regardless of how they interleave.
  auto reg = std::make_shared<UiProgressRegistry>();
  const std::uint64_t id = reg->open("denoise");
  UiProgress bar(reg, id);
  const int kSteps = 8, kBlocks = 28;
  DenoiseProgress prog(&bar, kSteps, /*forwards_per_step=*/1);
  auto fn = prog.block_fn();

  std::atomic<bool> bad{false};
  double last = 0.0;
  for (int s = 0; s < kSteps; ++s) {
    // Staged in order on this thread, as a forward does...
    std::vector<std::function<void()>> pubs;
    for (int b = 0; b < kBlocks; ++b) { pubs.push_back(fn(b, kBlocks)); }
    // ...and published from two threads at once, as two buffers would.
    auto drain = [&pubs](int lo, int hi) {
      for (int i = lo; i < hi; ++i) {
        auto& pub = pubs[(std::size_t)i];
        if (pub) { pub(); }
      }
    };
    std::thread a([&]() { drain(0, kBlocks / 2); });
    std::thread b([&]() { drain(kBlocks / 2, kBlocks); });
    a.join();
    b.join();
    prog.end_forward();
    prog.end_step(s);
    const double f = frac_(*reg, id);
    if (f < last - 1e-12 || f > 1.0 + 1e-12) { bad = true; }
    last = f;
  }
  EXPECT_TRUE(!bad);
  EXPECT_TRUE(last == 1.0);
}
