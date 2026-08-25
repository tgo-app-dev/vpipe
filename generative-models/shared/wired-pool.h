#ifndef VPIPE_GENERATIVE_MODELS_SHARED_WIRED_POOL_H
#define VPIPE_GENERATIVE_MODELS_SHARED_WIRED_POOL_H

// A streamed DiT's side of the manager's WIRED POOL: how much of what it
// decided to keep resident it is allowed to mlock, what happens when the
// box refuses, and when to ask again.
//
// WHY A KEPT BLOCK WANTS TO BE WIRED. BlockResidency grows a resident set
// into free RAM and then MEASURES whether those pages are still there,
// shedding one block and ratcheting whenever they are not. That
// measurement is honest but it is also purely reactive: a resident block
// is the coldest memory in the process -- read once per step, never
// written -- so it is the first thing the compressor takes, and the run
// then spends the rest of the schedule in a cycle of admit, compress,
// measure, shed, re-admit. Wiring is what breaks the cycle: mlock'd pages
// cannot be compressed or swapped, so a block that was admitted stays
// worth its RAM.
//
// WHY THERE IS A BUDGET AT ALL, rather than each model wiring what it
// likes. Wired memory is the one allocation the kernel cannot reclaim, so
// over-committing it does not degrade the box, it PANICS it. The manager
// owns one process-wide pool with a ceiling derived from the device's
// recommendedMaxWorkingSetSize, and this class is a model's window onto
// it -- it never wires anything itself, it only decides whether to ask
// and books what the manager granted.
//
// THE ORDER THAT MATTERS. Ask for the FIXED costs first -- any persistent
// activation scratch, then the trunk (the non-block weights every block of
// every forward reads) -- and the shed-able blocks last. A pool that runs
// out then runs out on the half a forward can proceed without. MEASURED on
// MiniMax-H3 when this was inverted and the blocks were wired at load:
// 4315 MB of blocks wired, the pool collapsed on the first refusal, and
// NOTHING was left for the trunk or the scratch -- against ~17 GB wired
// and both protected when the forward does it in this order.
//
// PER-FORWARD scratch is not wired and cannot usefully be. The image DiTs
// allocate their activations as locals inside the forward, so wiring them
// would be an mlock and a munlock per buffer per step against memory the
// allocator is about to recycle anyway -- and a buffer replaced while
// wired leaks its bytes from the pool's counter, since only
// unwire_from_pool() decrements it. Wire what the model HOLDS across
// forwards; a buffer that is reallocated must be unwired BEFORE the
// assignment that drops it.
//
// USAGE, per model:
//   at load           _wire.open(mc);
//                     ... and read weights Copied when _wire.on(), since
//                     mapped pages cannot be wired (see kept_residency).
//   top of forward    if (_wire.on()) {
//                       if (_wire.retry(mc, block_bytes)) {
//                         _resid.note_landscape_changed();
//                       }
//                       wire_fixed_(true);        // scratch, then trunk
//                     }
//   per block         if (_wire.wirable(nb) && _resid.admit(mc, nb)) {
//                       ... keep it, finish every write to it ...
//                       _wire.note_wired(mc, wire_block_(b, true), nb);
//                       _resid.note_admitted(nb);
//                     }
//   on eviction       _wire.note_unwired(wire_block_(b, false));
//   on destruction    unwire everything -- freeing a wired buffer unwires
//                     it in the kernel but does NOT decrement the pool's
//                     counter, so a DiT destroyed per clip would leak its
//                     whole share of the budget per clip.
//
// WHERE THIS CAME FROM, and the one place that does not use it yet. The
// policy was written inline in MetalMiniMaxH3Transformer, which is where
// every measurement quoted here was taken, and extracted when the image
// DiTs (FLUX.2, Krea-2, Qwen-Image-Edit / Mage-Flow, Boogu-Image) needed
// the same thing. H3 still carries its own copy: it is the version that
// has run in anger, and rewriting it was not worth the risk in the same
// change that gave five other models the behaviour for the first time.
// A change to the policy has to touch both until that is folded in.

#include <cstddef>

namespace vpipe::metal_compute {
class MetalCompute;
class SharedBuffer;
}  // namespace vpipe::metal_compute

namespace vpipe::genai {

// WHETHER A KEPT WEIGHT MAY BE A MAPPED VIEW.
//
// Mapped is zero-copy and reads as the obvious win, and it is the wrong
// answer for anything this class touches. A mapped tensor aliases the
// weight set's shard mmap, which means it can be neither WIRED (mlock on
// file-backed pages is refused well before the pool's ceiling -- MEASURED
// at ~4 GB on MiniMax-H3) nor PARKED (mark_inactive refuses on a handle
// that does not own its allocation). Both of those are the whole point of
// keeping the block.
//
// So: Copied whenever the model streams, or whenever the pool is on --
// which is what a FALSE from this says.
//
// MEASURED, MiniMax-H3 at 960x544x21 / 4 steps on the M4 Pro with arms
// INTERLEAVED (copied, mapped, copied, mapped) so thermal drift could not
// be read as a result: 186 / 174 s wired against 198 / 238 s mapped --
// 1.21x, and the mapped arm's 40 s spread against the wired arm's 12 s is
// the page cache being unpredictable in exactly the way wiring removes.
// The box got HEALTHIER, not tighter: compression fell from 2060 to 991
// MB across the run and swap fell with it, while 35 GB sat wired.
//
// A shard the file cannot map falls back to a copy regardless, so this is
// never worse than Copied -- but the LOG, not this rule, is what says
// whether a preload actually mapped.
inline bool
weights_may_be_mapped(bool stream_blocks, bool wire_resident)
{
  return !stream_blocks && !wire_resident;
}

class WiredPool {
 public:
  // Ask the manager whether this model wires at all, and how much of the
  // pool is still unspent. Call once, at load.
  //
  // From the MANAGER rather than from a per-model percentage. The old
  // form asked what share of RAM one DiT could wire, which is the
  // per-model budget the pool exists to replace: a ceiling each model
  // guesses separately never adds up to the box.
  //
  // `_budget` is what is still UNUSED, because every question asked of
  // this class is "can ONE MORE block be wired", not "how large is the
  // pool".
  void open(metal_compute::MetalCompute* mc);

  bool        on() const { return _on; }
  std::size_t wired_bytes() const { return _wired; }
  std::size_t budget() const { return _budget; }

  // May a block of `nb` bytes be KEPT?
  //
  // Past the budget there is nothing to gain: the block would be held
  // unprotected, the compressor would take it (it is the coldest memory
  // in the process), and the next residency walk would shed a block and
  // ratchet the ceiling over the whole resident set. Better not to hold
  // it at all -- which is why this gates ADMISSION and not just wiring.
  bool wirable(std::size_t nb) const
  {
    return !_on || _wired + nb <= _budget;
  }

  // Wire or unwire ONE buffer through the pool. Returns the bytes the
  // pool actually took (wiring) or gave back (unwiring); 0 when the
  // buffer was empty, already in that state, or the pool refused.
  std::size_t wire_one(metal_compute::MetalCompute* mc,
                       metal_compute::SharedBuffer& b, bool on);

  // Book what a whole block's wiring returned against what was asked.
  //
  // A partial grant means the box has said no. The percentage was an
  // UP-TO, not a reservation: another process holds wired memory, or the
  // system limit is nearer than the pool implied. So the budget becomes
  // what was actually granted and growth stops here rather than
  // continuing to admit blocks nothing can protect.
  //
  // HELD ONLY FOR THIS FORWARD. The refusal may have been another
  // process spiking, and a run that never asks again holds a small
  // resident set for the whole schedule on the strength of one syscall.
  // MEASURED: a 9 GB pool that granted 5857 MB sat at 16 of 50 blocks
  // for the rest of the run. retry() is what asks again.
  void note_wired(metal_compute::MetalCompute* mc, std::size_t got,
                  std::size_t want);

  // Give bytes back. Called where a block is dropped, so the counter
  // stays honest without having to infer it from destructors.
  void note_unwired(std::size_t n)
  {
    _wired -= (n > _wired) ? _wired : n;
  }

  // Top-of-forward retry. Returns TRUE when the budget actually rose, in
  // which case the caller must tell BlockResidency the landscape changed
  // -- it stopped growing when the budget ran out and cannot see that the
  // budget moved.
  //
  // GATED on the box having demonstrably freed a block's worth since the
  // refusal, so a genuinely full box is never asked: reopening the
  // ceiling makes the pool's own check pass, and the mlock behind it
  // would then fail and leave that block resident but UNWIRED -- one per
  // forward, exactly the state wirable() exists to avoid. A peer holding
  // wired memory shows up in this reading, since its pages are
  // unavailable while it holds them and return when it lets go.
  //
  // A forward is the granularity because it is where the resident set is
  // reconsidered anyway, and the check costs one budget read when the
  // flag is clear.
  bool retry(metal_compute::MetalCompute* mc, std::size_t block_bytes);

 private:
  bool        _on       = false;
  std::size_t _wired    = 0;
  std::size_t _budget   = 0;
  // Set when mlock refused, cleared when a retry actually raises the
  // budget -- so a box busy at forward 2 is still asked again at forward
  // 5. A single attempt would have spent itself against the same spike
  // that caused the refusal.
  bool        _retry    = false;
  // The available-physical reading at the refusal. The retry is gated on
  // the box having freed a block's worth SINCE, not on time passing.
  std::size_t _retry_at = 0;
};

}  // namespace vpipe::genai

#endif
