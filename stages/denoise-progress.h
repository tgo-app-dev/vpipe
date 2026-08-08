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

namespace vpipe {

// Turns a DiT's per-block callbacks into one smooth progress report over
// the whole denoise loop.
//
// A step-granular bar is too coarse to be useful: at 1024 a single FLUX.2
// step is ~2 s and a 20B Qwen-Image step ~12 s, so the report sits still
// for the entire time anything is actually happening. The DiT already
// walks its blocks one at a time, so counting those gives 25-60x the
// resolution for a callback that costs a compare.
//
// Guidance runs the DiT TWICE per step, and the sampler may call the
// denoise function more than once, so the block count alone cannot say
// where in the loop we are. end_step() therefore RE-SYNCS to the exact
// step boundary: within a step the bar interpolates, and at every step
// edge it is right regardless of how many forwards actually ran.
class DenoiseProgress {
public:
  DenoiseProgress(UiProgress* bar, int steps, int forwards_per_step)
      : _bar(bar), _steps(steps), _fwds(forwards_per_step < 1
                                            ? 1 : forwards_per_step)
  {
  }

  // Hand this to MetalXTransformer::set_block_progress.
  genai::DitBlockProgressFn block_fn()
  {
    return [this](int done, int total) { on_block_(done, total); };
  }

  // Call where a forward RETURNS: the callback fires on block ENTRY, so
  // the last block's work is only accounted for here.
  void end_forward()
  {
    if (_blocks > 0) { _base += _blocks; }
  }

  // Call at the end of denoise step `i` (0-based).
  void end_step(int i)
  {
    if (_bar == nullptr || _blocks <= 0) {
      if (_bar != nullptr) { _bar->update((std::uint64_t)(i + 1),
                                          (std::uint64_t)_steps); }
      return;
    }
    _base = (long long)(i + 1) * _fwds * _blocks;
    _bar->update((std::uint64_t)_base, (std::uint64_t)total_());
  }

private:
  void on_block_(int done, int total)
  {
    if (_bar == nullptr || total <= 0) { return; }
    _blocks = total;
    // Clamp: an extra forward (a sampler that evaluates more than _fwds
    // times) would otherwise push the bar past 100%, which reads as a bug
    // even though the step boundary re-syncs it a moment later.
    const long long t = total_();
    long long d = _base + done;
    if (d > t) { d = t; }
    _bar->update((std::uint64_t)d, (std::uint64_t)t);
  }

  long long total_() const
  {
    return (long long)_steps * _fwds * _blocks;
  }

  UiProgress* _bar    = nullptr;
  int         _steps  = 0;
  int         _fwds   = 1;
  int         _blocks = 0;      // learned from the first callback
  long long   _base   = 0;      // blocks completed before the current forward
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
