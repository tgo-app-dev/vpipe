#ifndef VPIPE_GENERATIVE_MODELS_SHARED_DIT_GPU_PROGRESS_H
#define VPIPE_GENERATIVE_MODELS_SHARED_DIT_GPU_PROGRESS_H

// Report a DiT block on the GPU's clock rather than the encode thread's.
//
// THE PROBLEM. A DiT forward is one deferred command stream: the block
// loop ENCODES all N blocks and commits once at the end, which is what
// lets the CPU run ahead of the GPU instead of taking turns with it.
// `_block_progress` fired from that loop therefore measures encoding,
// and encoding a whole stack takes about as long as nothing.
//
// MEASURED on an M4 Pro 64 GB, Krea-2 bf16 at 1024x1024 -- 28 blocks,
// ~18 s per step once the stack is resident:
//
//   forward 0 (streaming)   28 callbacks over 28.5 s, ~1.0 s apart
//   forwards 1-7 (resident) 28 callbacks inside ONE MILLISECOND
//
// So the bar leapt a whole step's worth in under a millisecond and then
// could not move for the ~18 s the GPU actually took. Reported as "the
// progress is not uniform -- sometimes it runs fast and sometimes it
// sticks for a few seconds", which is exactly what it was.
//
// The inversion is the tell: the STREAMED path looks smooth and the
// resident path looks wedged. Not a coincidence -- a streamed block must
// commit and wait before its weights are freed, so its callbacks are
// paced against real work by accident. A stack that needs no such
// barrier gets no such pacing.
//
// THE FIX, and why it is free. The report is attached to the command
// buffer open at that point in the encode, and fires when the GPU
// RETIRES it. No barrier, no wait, nothing added to the critical path --
// the stream is already split into fire-and-forget buffers every
// VPIPE_MC_CMDBUF_SPLIT dispatches (50 by default), each of which
// already carries a completion handler for the failure latch, so the
// checkpoints exist whether or not anybody reports from them.
//
// Several blocks' handlers can therefore land on ONE buffer and fire
// together; DenoiseProgress keeps the highest and ignores the rest, so
// the bar advances once per buffer, to the last block that buffer
// contained. That is a report that lags the truth slightly and never
// leads it, which is the right direction for a progress bar.
//
// STAGED HERE, PUBLISHED THERE. `fn` is called on THIS thread, inside the
// forward, and resolves which forward's block this is while that is still
// unambiguous; only the closure it returns is deferred. Doing it the
// other way round -- deferring the whole callback and letting it work out
// its position when it fires -- reads the step boundary at the wrong
// moment: a handler that lands just after the loop crossed into the next
// step would resolve "block 3" against the NEW step and report a position
// most of a step ahead of the truth, after which the bar sits still until
// the real work catches up to the lie. Which is the bug this file exists
// to remove, moved rather than fixed.
//
// THE PUBLISH RUNS ON A METAL THREAD, not the caller's, and may run after
// the forward has returned -- a buffer completes asynchronously and
// nothing orders the last handler against the wait_ok() that saw the same
// buffer finish. DenoiseProgress meets both rules: it holds its state by
// shared_ptr and cuts the bar loose in its destructor. A caller passing
// something else has to meet them itself.
//
// WHERE THIS IS NOT NEEDED: a stack that already commits and waits per
// block (FLUX.2's single stack, Boogu, and MiniMax-H3 before it stopped
// paying for a barrier it no longer needs) reports on the GPU's clock by
// construction. Routing those through here too costs nothing and is what
// keeps the property from quietly lapsing the next time a per-block
// fence is removed for speed.

#include "apple-silicon/metal-compute/command-stream.h"
#include "generative-models/shared/dit-block-progress.h"

#include <functional>
#include <utility>

namespace vpipe::genai {

// Report `done` of `total` blocks when `stream`'s current command buffer
// completes. Falls back to reporting inline when there is no stream to
// hang the publish on, which is no worse than not reporting at all.
inline void
report_block(metal_compute::CommandStream& stream,
             const DitBlockProgressFn& fn, int done, int total)
{
  if (!fn) { return; }
  std::function<void()> publish = fn(done, total);
  if (!publish) { return; }
  if (!stream.valid()) {
    publish();
    return;
  }
  stream.on_completion(std::move(publish));
}

}  // namespace vpipe::genai

#endif
