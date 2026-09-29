#ifndef VPIPE_GENERATIVE_MODELS_SHARED_WIRED_POOL_H
#define VPIPE_GENERATIVE_MODELS_SHARED_WIRED_POOL_H

// A model's side of the manager's WIRED POOL: how much of what it decided
// to keep resident it is allowed to mlock, what happens when the box
// refuses, and when to ask again. In-tree DiTs and plugins use the same
// class.
//
// WHAT IS FROZEN -- changing any of it is a VPIPE_PLUGIN_ABI_VERSION bump:
//   * WiredPool's layout, which is ONE POINTER: every method is out of
//     line, so nothing about the policy's state is compiled into a caller;
//   * the signatures of the methods below.
// WHAT GROWS FREELY -- never a bump:
//   * open() option keys and info() fields. Added, never renamed; an
//     absent key is its documented default;
//   * new methods, which a plugin built against an older header never
//     calls (one built against a newer header fails to LOAD on an older
//     host, by symbol, rather than misbehaving);
//   * the policy itself: when to refuse, when to reopen, what to log.
//
// WHY A KEPT BLOCK WANTS TO BE WIRED. BlockResidency grows a resident set
// into free RAM and then MEASURES whether those pages are still there,
// shedding one block and ratcheting whenever they are not. That
// measurement is honest but purely reactive: a resident block is the
// coldest memory in the process -- read once per step, never written --
// so it is the first thing the compressor takes, and the run then spends
// the schedule in a cycle of admit, compress, measure, shed, re-admit.
// mlock'd pages cannot be compressed or swapped, so a block that was
// admitted stays worth its RAM.
//
// WHY THERE IS A BUDGET AT ALL. Wired memory is the one allocation the
// kernel cannot reclaim, so over-committing it does not degrade the box,
// it PANICS it. The manager owns one process-wide pool with a ceiling
// derived from the device's recommendedMaxWorkingSetSize, and this class
// is a model's window onto it. The pool is READ LIVE, never snapshotted:
// it is shared by every model and ANE tier in the process, so a figure
// taken at load is stale the moment a peer loads, frees or meets a
// refusal.
//
// THE ORDER THAT MATTERS. Wire the FIXED costs first -- any persistent
// activation scratch, then the trunk -- and the shed-able blocks last. A
// pool that runs out then runs out on the half a forward can proceed
// without. MEASURED on MiniMax-H3 when this was inverted: 4315 MB of
// blocks wired, the pool collapsed on the first refusal, and NOTHING was
// left for the trunk or the scratch -- against ~17 GB wired and both
// protected in this order.
//
// A buffer that is REPLACED must be unwired before the assignment that
// drops it: freeing a wired buffer unwires its pages in the kernel but
// only the pool decrements its counter, so the bytes would stay charged
// for the rest of the process.
//
// USAGE:
//   at load           _wire.open(mc);
//                     ... and read weights Copied when _wire.on(), since
//                     mapped pages cannot be wired (weights_may_be_mapped).
//   per run           _wire.new_run();          (where the schedule is set)
//   top of forward    if (_wire.on()) {
//                       if (_wire.retry(mc, block_bytes)) {
//                         _resid.note_landscape_changed();
//                       }
//                       _wire.wire_set(fixed_buffers, true);
//                     }
//   per block         if (_wire.wirable(nb) && _resid.admit(mc, nb)) {
//                       ... keep it, finish every write to it ...
//                       _wire.note_wired(mc, _wire.wire_set(bufs, true), nb);
//                       _resid.note_admitted(nb);
//                     }
//   on eviction       _wire.note_unwired(_wire.wire_set(bufs, false));
//   on destruction    unwire everything the model wired.

#include "common/flex-data.h"

#include <cstddef>
#include <span>
#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::metal_compute {
class MetalCompute;
class SharedBuffer;
}  // namespace vpipe::metal_compute

namespace vpipe::genai {

// ---- the vocabulary: open() options and info() fields -----------------
//
// Options (all optional):
//   "tag"          text  -- names this model in the pool's log lines.
//   "enabled"      flag  -- false keeps the pool closed for this model
//                           even when the manager's pool is on. Default:
//                           on whenever the manager's pool is.
namespace wired_pool {
inline constexpr std::string_view kTag     = "tag";
inline constexpr std::string_view kEnabled = "enabled";

// info() fields:
inline constexpr std::string_view kInfoOn          = "on";
inline constexpr std::string_view kInfoWiredBytes  = "wired_bytes";
inline constexpr std::string_view kInfoBudget      = "budget";
inline constexpr std::string_view kInfoPoolUsed    = "pool_used";
inline constexpr std::string_view kInfoPoolLimit   = "pool_limit";
inline constexpr std::string_view kInfoPoolAsk     = "pool_ask";
inline constexpr std::string_view kInfoRetryArmed  = "retry_armed";
inline constexpr std::string_view kInfoRefusals    = "refusals";
inline constexpr std::string_view kInfoReopens     = "reopens";
// Bytes the last wire_set(..., true) left unwired because the pool
// refused, including the buffer it stopped on.
inline constexpr std::string_view kInfoLastRefused = "last_refused_bytes";
}  // namespace wired_pool

// WHETHER A KEPT WEIGHT MAY BE A MAPPED VIEW.
//
// Mapped is zero-copy and reads as the obvious win, and it is the wrong
// answer for anything this class touches. A mapped tensor aliases the
// weight set's shard mmap, which means it can be neither WIRED (mlock on
// file-backed pages is refused well before the pool's ceiling -- MEASURED
// at ~4 GB on MiniMax-H3) nor PARKED (mark_inactive refuses on a handle
// that does not own its allocation).
//
// So: Copied whenever the model streams, or whenever the pool is on.
//
// MEASURED, MiniMax-H3 at 960x544x21 / 4 steps on the M4 Pro with arms
// INTERLEAVED: 186 / 174 s wired against 198 / 238 s mapped -- 1.21x, and
// the mapped arm's 40 s spread against the wired arm's 12 s is the page
// cache being unpredictable in exactly the way wiring removes.
inline bool
weights_may_be_mapped(bool stream_blocks, bool wire_resident)
{
  return !stream_blocks && !wire_resident;
}

class WiredPool {
 public:
  WiredPool();
  ~WiredPool();
  WiredPool(WiredPool&& o) noexcept;
  WiredPool& operator=(WiredPool&& o) noexcept;
  WiredPool(const WiredPool&)            = delete;
  WiredPool& operator=(const WiredPool&) = delete;

  // Ask the manager whether this model wires at all. Call at load; calling
  // again (a new run, a new geometry) re-reads the switch and keeps what
  // is already booked.
  void open(metal_compute::MetalCompute* mc);
  void open(metal_compute::MetalCompute* mc, const FlexData& options);

  bool        on() const;
  // What this model has booked as wired through note_wired().
  std::size_t wired_bytes() const;
  // Its booked bytes plus the pool's unused room, now. For reporting;
  // wirable() is the gate.
  std::size_t budget() const;

  // May a block of `nb` bytes be KEPT? Asked of the pool AS IT IS NOW.
  // Always true with the pool closed. Past the pool there is nothing to
  // gain: the block would be held unprotected and the compressor would
  // take it -- which is why this gates ADMISSION and not just wiring.
  bool wirable(std::size_t nb) const;

  // Wire or unwire ONE buffer. Returns the bytes the pool took (wiring) or
  // gave back (unwiring); 0 when the buffer was empty, already in that
  // state, below the pool's minimum, or refused.
  std::size_t wire_one(metal_compute::MetalCompute* mc,
                       metal_compute::SharedBuffer& b, bool on);

  // After wire_one() returned 0 for `b`: did the POOL say no -- full, or
  // capped by a real shortage? False for a buffer too small to be a pool
  // question or one that would not wire for a reason of its own.
  bool refused(metal_compute::MetalCompute* mc,
               const metal_compute::SharedBuffer& b) const;

  // Wire or unwire a SET of buffers -- a block, a trunk, a scratch -- in
  // the order given. Null and empty entries are skipped. Wiring STOPS at
  // the first buffer the pool refuses, keeping what is already wired (a
  // partly protected block beats an unprotected one, and giving it back
  // means competing for it again against a pool that just said no), and
  // arms the retry. Buffers that fail for a reason of their own are
  // skipped. Returns the bytes wired or unwired; does NOT book them --
  // note_wired / note_unwired decide whether they are this model's blocks.
  std::size_t wire_set(std::span<metal_compute::SharedBuffer* const> bufs,
                       bool on);

  // Book what a whole block's wiring returned against what was asked. A
  // SHORTFALL IS NOT ALWAYS A REFUSAL: buffers below the pool's minimum are
  // never wired and every block carries some. Only a pool that cannot take
  // the missing bytes has said no, and then the retry is armed.
  void note_wired(metal_compute::MetalCompute* mc, std::size_t got,
                  std::size_t want);
  // Give booked bytes back where a block is dropped. Never underflows.
  void note_unwired(std::size_t n);

  // A new run is starting: a collapsed ceiling this model did not cause
  // may be asked about again, once.
  void new_run();

  // Top-of-forward retry of a COLLAPSED pool ceiling. TRUE when the
  // ceiling rose, in which case the caller tells BlockResidency the
  // landscape changed and re-wires what it holds.
  //
  // After THIS model's refusal: once the box has freed `block_bytes` since
  // -- a genuinely full box is never asked, because reopening would let
  // the mlock behind it fail and leave a block held but unwired. A ceiling
  // SOMEONE ELSE collapsed: once per run, since this model has no reading
  // from that refusal and the box may well have room; a box that really
  // is full refuses again, which makes it this model's own refusal.
  bool retry(metal_compute::MetalCompute* mc, std::size_t block_bytes);

  // What the pool and this model's window onto it look like now. Fields
  // in wired_pool:: above.
  FlexData info() const;

 private:
  struct Impl;
  Impl* _p = nullptr;
};

}  // namespace vpipe::genai

VPIPE_API_END

#endif
