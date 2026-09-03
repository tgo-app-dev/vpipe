#ifndef VPIPE_STAGES_DENOISE_PROGRESS_H
#define VPIPE_STAGES_DENOISE_PROGRESS_H

// Per-block denoise progress, shared by the image and video DiT stages.
//
// Both drive a stack of transformer blocks inside a step loop and want the
// same report out of it, so this lives here rather than once per stage. A
// second copy would drift on the one part that is easy to get wrong: how
// an extra forward per step (guidance) is accounted for.

#include "generative-models/shared/dit-block-progress.h"
#include "interfaces/ui-delegate-intf.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

namespace vpipe {

// Turns a DiT's per-block callbacks into one smooth progress report over
// the whole denoise loop.
//
// A step-granular bar is too coarse to be useful: at 1024 a single FLUX.2
// step is ~2 s and a 20B Qwen-Image step ~12 s, so the report sits still
// for the entire time anything is actually happening. The DiT already
// walks its blocks one at a time, so counting those gives 25-60x the
// resolution for a callback that costs a lock and a small closure.
//
// ON THE GPU'S CLOCK, not the encode thread's, which is what the two
// phases below are for -- see DitBlockProgressFn and
// shared/dit-gpu-progress.h. A resident DiT encodes its whole stack in
// under a millisecond and then runs it for tens of seconds, so a bar
// driven from the block loop leaps a full step and freezes, which is
// exactly the "sometimes fast, sometimes stuck" a user reports.
//
// Guidance runs the DiT TWICE per step, and the sampler may call the
// denoise function more than once, so the block count alone cannot say
// where in the loop we are. end_step() therefore RE-SYNCS to the exact
// step boundary: within a step the bar interpolates, and at every step
// edge it is right regardless of how many forwards actually ran.
class DenoiseProgress {
public:
  DenoiseProgress(UiProgress* bar, int steps, int forwards_per_step)
      : _s(std::make_shared<State>())
  {
    _s->bar   = bar;
    _s->steps = steps;
    _s->fwds  = forwards_per_step < 1 ? 1 : forwards_per_step;
  }

  // CUT THE BAR LOOSE, do not assume nobody is still holding the
  // callback.
  //
  // The report is delivered from a Metal command-buffer completion
  // handler (see shared/dit-gpu-progress.h), which runs on a Metal
  // thread and is NOT ordered against the wait_ok() that saw the same
  // buffer finish -- so the last block of the last forward can be
  // reported after this stack local has gone. The state outlives us by
  // shared_ptr and a null bar makes every late report a no-op.
  ~DenoiseProgress()
  {
    std::lock_guard<std::mutex> lk(_s->mu);
    _s->bar = nullptr;
  }

  DenoiseProgress(const DenoiseProgress&)            = delete;
  DenoiseProgress& operator=(const DenoiseProgress&) = delete;

  // Hand this to MetalXTransformer::set_block_progress.
  //
  // TWO PHASES (see DitBlockProgressFn). The outer call runs on the
  // forward's thread and resolves the block to an ABSOLUTE position in
  // the denoise loop; the closure it returns publishes that position and
  // may run much later, on a Metal thread. Resolving early is what makes
  // a late publish harmless -- the number was already fixed against the
  // right forward.
  //
  // Captures the STATE, never `this`: the DiT is a stage member that
  // outlives this object, and a closure already handed to a command
  // buffer cannot be recalled.
  genai::DitBlockProgressFn block_fn()
  {
    return [s = _s](int done, int total) -> std::function<void()> {
      const long long at = stage_(s, done, total);
      if (at < 0) { return {}; }
      return [s, at]() { publish_(s, at); };
    };
  }

  // Call where a forward RETURNS: the callback fires on block ENTRY, so
  // the last block's work is only accounted for here.
  void end_forward()
  {
    std::lock_guard<std::mutex> lk(_s->mu);
    if (_s->blocks > 0) { _s->base += _s->blocks; }
  }

  // Adopt the step count the SAMPLER is actually running.
  //
  // A stage's configured `steps` is not always that number. MiniMax-H3's
  // is a count of sigma GRID POINTS including the terminal zero, so it
  // drives one fewer evaluation -- and the shift collapses duplicate
  // sigmas, which can drop more still. The stage cannot know the
  // difference before the scheduler is built; the per-step callback
  // carries it, and a bar told 4 while 3 run stops at 75% and reads as a
  // hang at the end of every generation.
  //
  // Safe mid-run: the totals are recomputed from steps on each update.
  void set_steps(int n)
  {
    if (n <= 0) { return; }
    std::lock_guard<std::mutex> lk(_s->mu);
    _s->steps = n;
  }

  // Call at the end of denoise step `i` (0-based).
  void end_step(int i)
  {
    std::lock_guard<std::mutex> lk(_s->mu);
    if (_s->bar == nullptr) { return; }
    if (_s->blocks <= 0) {
      _s->bar->update((std::uint64_t)(i + 1), (std::uint64_t)_s->steps);
      return;
    }
    _s->base = (long long)(i + 1) * _s->fwds * _s->blocks;
    _s->publish(_s->base);
  }

private:
  // Shared with every in-flight completion handler, so it must outlive
  // the object that made it and be safe on any thread.
  struct State {
    std::mutex  mu;
    UiProgress* bar    = nullptr;
    int         steps  = 0;
    int         fwds   = 1;
    int         blocks = 0;      // learned from the first callback
    long long   base   = 0;      // blocks completed before this forward
    long long   high   = 0;      // the furthest the bar has been told

    long long total() const { return (long long)steps * fwds * blocks; }

    // MONOTONIC, and that is the point rather than a nicety. Handlers
    // for several blocks ride one command buffer and fire together, and
    // one for an early block can land after end_step() has re-synced
    // past it; publishing that would walk the bar backwards. Keeping the
    // furthest makes a buffer's worth of reports collapse to the last
    // block it actually contained.
    //
    // Called under `mu`.
    void publish(long long d)
    {
      if (bar == nullptr) { return; }
      const long long t = total();
      if (t <= 0) { return; }
      if (d > t) { d = t; }       // an extra forward: clamp, do not wrap
      if (d < high) { return; }
      high = d;
      bar->update((std::uint64_t)d, (std::uint64_t)t);
    }
  };

  // PHASE 1, on the forward's thread: what absolute position is this
  // block? Negative means there is nothing to report.
  static long long stage_(const std::shared_ptr<State>& s, int done,
                          int total)
  {
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->bar == nullptr || total <= 0) { return -1; }
    // Re-baseline if the stack size is not what it was: `high` counts
    // blocks, so a different stack makes every earlier reading
    // incomparable.
    if (s->blocks != total) {
      s->blocks = total;
      s->high   = 0;
    }
    return s->base + done;
  }

  // PHASE 2, on whatever thread the GPU's completion handler runs.
  static void publish_(const std::shared_ptr<State>& s, long long at)
  {
    std::lock_guard<std::mutex> lk(s->mu);
    s->publish(at);
  }

  std::shared_ptr<State> _s;
};

// Installs the block hook for a scope and CLEARS it on the way out.
//
// Not optional bookkeeping: the callback captures a pointer to a stack
// local, while the DiT is a stage member that outlives this function. A
// hook left installed would be called by the next generation with a
// dangling DenoiseProgress, and the denoise paths have several early
// returns (a failed forward, a pipeline stop) where remembering to clear
// it by hand is exactly the kind of thing that gets missed.
template <class Dit>
class ScopedBlockProgress {
public:
  ScopedBlockProgress(Dit* dit, DenoiseProgress& prog)
      : _dit(dit), _prog(&prog)
  {
    if (_dit != nullptr) { _dit->set_block_progress(prog.block_fn()); }
  }
  ~ScopedBlockProgress()
  {
    if (_dit != nullptr) { _dit->set_block_progress(nullptr); }
  }
  ScopedBlockProgress(const ScopedBlockProgress&)            = delete;
  ScopedBlockProgress& operator=(const ScopedBlockProgress&) = delete;

  // Point the hook at a DIFFERENT model mid-scope, clearing the old one.
  //
  // For a stack that is swapped inside the denoise loop rather than held
  // across it: Wan's two noise experts are two checkpoints, and crossing
  // the boundary destroys one transformer and builds the other, taking
  // its hook with it. Re-arming keeps the RAII guarantee -- whatever is
  // installed when the scope ends is what gets cleared -- which the
  // early-return paths need, since `prog` is a stack local and the model
  // is a stage member that outlives it.
  void rearm(Dit* dit)
  {
    if (dit == _dit) { return; }
    if (_dit != nullptr) { _dit->set_block_progress(nullptr); }
    _dit = dit;
    if (_dit != nullptr) { _dit->set_block_progress(_prog->block_fn()); }
  }

private:
  Dit*             _dit  = nullptr;
  DenoiseProgress* _prog = nullptr;
};

}  // namespace vpipe

#endif
