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

// The same shape, out of a video VAE's decode.
//
// Separate name, identical signature: a VAE decode is not a stack of
// blocks and its unit is the TILE (a clip is chunked in time and each
// chunk tiled in space), so a caller reading `DitBlockProgressFn` on a
// VAE would be told the wrong thing about what it counts. The H3 decoder
// is a 2.4B ViT run once per tile, so at 960x544 a clip is ~15 tiles per
// chunk and a minute of otherwise silent work.
//
// Fired on BOTH edges of each tile: once on entry with the tiles finished
// before it, once on exit with that count incremented. So `done` starts at
// 0 and does reach `total`, and the caller needs no completion call of its
// own -- which matters because a caller closes its bar the instant the
// decode returns, and a completion it sets a microsecond earlier is never
// repainted. (The DiT's above cannot do this: its block loop has early
// exits, so entry is the one point every block passes through.)
using VaeTileProgressFn = std::function<void(int done, int total)>;

}

#endif
