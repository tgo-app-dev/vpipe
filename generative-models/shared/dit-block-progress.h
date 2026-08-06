#ifndef DIT_BLOCK_PROGRESS_H
#define DIT_BLOCK_PROGRESS_H

#include <functional>

namespace vpipe::genai {

// Per-block progress out of a DiT forward pass.
//
// A denoise STEP is the natural unit to report, but at high resolution one
// step is seconds to tens of seconds of silence -- long enough that a bar
// updated per step reads as stalled. Every DiT here is a stack of
// transformer blocks it already walks one at a time (the same loop that
// polls the streaming stop), so the block is a free, much finer tick.
//
// Fired as each block is ENTERED, with `done` = blocks completed before it
// (so 0 on the first, total-1 on the last) and `total` = blocks in the
// stack. Entry rather than exit because the DiTs' block loops are long and
// have several early-out paths; the one point every block passes through is
// the top, next to the streaming-stop poll.
//
// It therefore never reports `total`, and a caller that needs the forward's
// completion should mark it where the forward RETURNS. Reporting position
// within one forward is all this does: mapping it onto the whole denoise
// loop -- which knows the step count and whether guidance runs a second
// forward per step -- belongs to the caller.
//
// A stack that streams its blocks from disk ticks at wildly uneven
// intervals; that is honest, not a defect. Runs on the forward's own
// thread, between blocks, so it must be cheap and must not re-enter the
// model.
using DitBlockProgressFn = std::function<void(int done, int total)>;

}

#endif
