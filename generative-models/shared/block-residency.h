#ifndef VPIPE_GENERATIVE_MODELS_SHARED_BLOCK_RESIDENCY_H
#define VPIPE_GENERATIVE_MODELS_SHARED_BLOCK_RESIDENCY_H

// Grow a streamed DiT's RESIDENT block set into free RAM, and give it
// back when something else needs the room.
//
// Streaming keeps ~one block live so a model far larger than the box can
// run at all. The cost is that it re-reads the whole checkpoint every
// forward, and on a machine with headroom to spare that is throughput
// thrown away -- a 4-bit MiniMax-H3 streams ~4.5 GB per forward while
// 10+ GB of a 16 GB box sits idle. This keeps blocks after they are used,
// as free memory allows.
//
// WHY THIS IS NOT A CACHE, and the one thing to understand before
// changing it. The access pattern is a repeating sequential scan: blocks
// 0..N-1, then the next step scans 0..N-1 again. For a cyclic scan with
// capacity C < N, LRU has a ZERO percent hit rate -- the block evicted to
// make room is always the one wanted next time round. Recency is exactly
// the wrong signal. What works for a looping scan is a FIXED subset held
// permanently, which gives C/N. So this grows a resident set and then
// leaves it alone: no recency bookkeeping exists here, and adding some
// would make it worse, not better.
//
// The four anti-thrash properties, in the order the failure modes appear:
//
//   1. Admission spends only genuinely FREE headroom, never another
//      block's seat. Nothing is ever evicted in order to admit.
//   2. It keeps `reserve` clear for the rest of the forward -- the
//      activations, the dequant scratch, whatever the caller knows it
//      still needs. Admitting into that room only pushes it back out.
//   3. Hysteresis: admission asks for the block, plus the reserve, plus
//      ONE MORE BLOCK of slack. Without that last term a budget sitting
//      on the line admits, overshoots, evicts, and repeats.
//   4. A RATCHET: after any eviction the ceiling drops to what is left,
//      so a set that was cut back cannot climb straight back into the
//      pressure it was just asked to relieve. It DECAYS rather than
//      latching -- see note_healthy_forward().
//   5. It CHECKS THAT THE BLOCKS IT KEPT ARE STILL IN RAM, and gives one
//      back whenever they are not.
//
// Property 5 is the one that was missing, and free-memory arithmetic is
// not a substitute for it. The old gate asked whether `available_physical`
// -- free + purgeable + FILE-BACKED pages -- covered the next block. On a
// streaming model the file cache is mostly the checkpoint that model just
// re-read, so the signal is self-generated: the more it streams, the more
// room it believes it has. MEASURED on a 64 GB box running an unquantized
// DiT: growth ran to 50 of 50 blocks (61566 MB) against a reading of
// ~18.5 GB available, 18.54 GB of which was file cache, on a machine
// holding 33 GB compressed and 28.7 GB of swap. It ended SLOWER than a
// 24 GB box that streamed.
//
// Tightening the arithmetic to idle pages only does not fix it either,
// and would break the healthy case: macOS keeps `free` small by design,
// so a box whose cache is merely cold reads as having no room. The
// arithmetic cannot distinguish a cache about to be dropped from one in
// use, which is exactly why it cannot find the limit.
//
// So the limit is MEASURED instead. A resident block is worth its RAM
// only while it is IN RAM -- once the OS has compressed or swapped its
// pages it costs a fault or a decompress on every step, which is what a
// streamed block costs anyway, plus the memory it still nominally holds.
// note_weight_residency() is told how much of the resident set is
// actually resident, and any of it having left is a pin that failed:
// give a block back and ratchet, which converges on what the box will
// really hold without anyone predicting it.
//
// This class is policy ONLY. It never owns or frees a block: the models
// differ in how they hold them (one list for Krea-2 / Qwen-Image /
// MiniMax-H3, two for FLUX.2 / Boogu, which have separate double- and
// single-stream stacks), so the caller keeps its containers and this
// answers admit/evict questions about them.
//
// Usage, per forward:
//     _resid.begin_forward(_mc, [&]{ return evict_one_block(); });
//     ... per block, AFTER the commit that retires its GPU work ...
//     if (_resid.admit(_mc, nbytes)) { keep the block; _resid.note_admitted(nbytes); }
// and on demand from a peer that needs room:
//     _resid.release(bytes, [&]{ return evict_one_block(); });

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstddef>
#include <functional>

namespace vpipe::genai {

class BlockResidency {
 public:
  // How much must stay free for the rest of the forward, for the peers
  // that have NOT allocated yet (see note_reserve_allocated). Growth
  // stays off until a caller sets one at least once, so a model that has
  // not been taught what its activations cost keeps exactly the
  // behaviour it had rather than growing against a guess.
  //
  // Setting it to ZERO is a different statement from never setting it:
  // it says the question was asked and the answer is nothing. That is
  // the honest answer for a caller that FREES this model before the next
  // peer runs -- reserving room for a coexistence that does not happen
  // buys nothing and costs the whole denoise. Growth is then bounded by
  // admit()'s own hysteresis (a block either side of the one asked for)
  // instead of by a figure protecting nobody.
  void set_reserve(std::size_t bytes)
  {
    _reserve = bytes;
    _reserve_declared = true;
  }
  std::size_t reserve() const { return _reserve; }

  // How much of that reserve the caller has ALREADY allocated, and which
  // the budget it passes to admit() has therefore already subtracted.
  //
  // Reserving it again is the difference between asking for headroom and
  // asking for it TWICE. MEASURED on the M5 16 GB box: MiniMax-H3
  // allocates its ~4.0 GB of activation scratch at the top of the
  // forward and only then reads the budget, so a 5086 MB reserve against
  // 6021 MB available refused a 206 MB block by 27 MB -- while the 4 GB
  // it was protecting was sitting in its own hands, counted once by the
  // OS and once more here. The run streamed 50 blocks to keep room that
  // already existed.
  //
  // What is left after subtracting it is the part that genuinely has not
  // been allocated yet: the peers that run after the denoise, above all
  // the VAE decode.
  void note_reserve_allocated(std::size_t bytes)
  {
    _reserve_allocated = bytes;
  }

  // The reserve still to be found. Never negative, and never more than
  // what was asked for.
  std::size_t reserve_now() const
  {
    return _reserve > _reserve_allocated ? _reserve - _reserve_allocated : 0;
  }

  std::size_t bytes() const { return _bytes; }
  int count() const { return _count; }

  // Re-arm for a new forward. Growth is re-probed once per forward rather
  // than once per block; the RATCHET deliberately survives, so a set that
  // was cut back does not simply refill on the next step.
  void begin_forward()
  {
    _growing = true;
    _logged_stop = false;
    _admitted_this_forward = 0;
    _shed_this_forward = false;
  }

  // The polling form. Re-arms growth, and gives a block back when the
  // machine is in outright distress -- the coarse backstop for pressure
  // this process did not cause and cannot see in its own page tables.
  //
  // ONE block, as note_weight_residency() does, and for the same reason
  // with an extra one on top: `paging()` is system-wide, so it cannot
  // say how much of that pressure is ours, and shedding proportionally
  // to a number that is mostly another process's would collapse the
  // resident set for someone else's allocation. Distress that persists
  // sheds again next forward, which converges without ever taking a
  // decision it cannot walk back.
  //
  // Once a forward is the right cadence: a step is seconds, shedding is
  // pure bookkeeping plus a free, and the alternative (a memory-pressure
  // dispatch source) would land on an arbitrary thread in the middle of
  // an encode. What matters is STOPPING, which the admit path does
  // immediately.
  void begin_forward(const metal_compute::MetalCompute::MemoryBudget& mb,
                     const std::function<std::size_t()>& evict_one)
  {
    begin_forward();
    if (!mb.paging() || _count <= 0) { return; }
    release(1, evict_one);              // ratchets
    _paged = true;
  }

  // Whether growth was ever stopped or cut back by system paging. For
  // reporting -- a run that quietly streams more than the box should need
  // is worth a line in the log.
  bool paged() const { return _paged; }

  // May a block of `block_bytes` stay resident? Policy only -- the caller
  // keeps the block and calls note_admitted() if this says yes.
  bool admit(metal_compute::MetalCompute* mc, std::size_t block_bytes)
  {
    if (mc == nullptr) { return false; }
    const auto mb = mc->memory_budget();
    const bool ok = admit(mb, block_bytes);
    if (!ok && _log_stop && mc->session() != nullptr) {
      _log_stop = false;
      mc->session()->log_debug(fmt(
          "block residency: stopped growing at {} blocks ({} MB) -- {} MB "
          "available ({} MB of it idle), {} MB compressed of ours, {} MB "
          "swap, against a {} MB reserve still to find (of {} MB)", _count,
          _bytes >> 20,
          mb.available_physical >> 20, mb.free_physical >> 20,
          mb.self_compressed >> 20, mb.swap_used >> 20,
          reserve_now() >> 20, _reserve >> 20));
    }
    return ok;
  }

  // The policy itself, over a budget SNAPSHOT rather than a device -- so
  // it can be exercised at the shapes that matter (a 64 GB box deep into
  // swap) on a machine that is not one.
  bool admit(const metal_compute::MetalCompute::MemoryBudget& mb,
             std::size_t block_bytes)
  {
    if (!_growing || !_reserve_declared || block_bytes == 0) {
      return false;
    }
    if (_bytes + block_bytes > _ceiling) {
      _growing = false;                 // ratcheted by an earlier release
      return false;
    }
    if (mb.recommended == 0) { return false; }
    // Bounded growth per forward. The gate below cannot see the limit
    // (it is arithmetic over a cache it cannot read the future of), so
    // the measurement in note_weight_residency() has to be given a
    // chance to react before the whole checkpoint has been admitted --
    // uncapped, the first forward takes everything and the first
    // evidence arrives after the damage.
    if (_admitted_this_forward >= _per_forward_cap) { return false; }
    // block + reserve + one more block: the trailing term is the
    // hysteresis gap. See property 3 above.
    if (!mb.fits_growth(block_bytes + reserve_now() + block_bytes)) {
      // Logged even from zero blocks. A refusal at _count == 0 used to be
      // silent, which is exactly the case that leaves you unable to tell a
      // reserve that is too big from a box that is already paging -- and
      // guessing between those two is how a reserve gets blamed for a
      // refusal it had no part in. Once per forward, at debug.
      if (!_logged_stop) {
        _logged_stop = true;
        _log_stop = true;
      }
      if (mb.paging()) { _paged = true; }
      _growing = false;   // stop asking for the rest of this forward
      return false;
    }
    return true;
  }

  // Book a block the caller has decided to keep.
  void note_admitted(std::size_t block_bytes)
  {
    _bytes += block_bytes;
    ++_count;
    ++_admitted_this_forward;
    // The unit the ratchet moves in. Largest seen rather than the
    // latest: blocks differ (a first or last block carries extra), and
    // a ceiling that lifts by the smallest of them would admit nothing.
    if (block_bytes > _block_hint) { _block_hint = block_bytes; }
  }

  // How many blocks a single forward may add. Default 8: enough that a
  // short schedule still reaches a useful resident set within a few
  // steps, small enough that an overshoot is a few blocks rather than a
  // whole checkpoint.
  void set_per_forward_cap(int blocks)
  {
    _per_forward_cap = blocks > 0 ? blocks : 1;
  }

  // Has THIS PROCESS's compressed footprint grown since the last
  // forward? One task_info read by the caller, and the gate on the page
  // walk behind it: when nothing of ours is being compressed there is
  // nothing for the walk to find, and the walk is not cheap -- MEASURED
  // at 57 ms per 4.3 GB examined, which on a 62 GB resident set would be
  // ~800 ms of every step spent asking a question whose answer is
  // already known to be no.
  //
  // `noise` ignores the small, ordinary movement every process shows.
  bool self_compression_grew(std::size_t now,
                             std::size_t noise = 64ull << 20)
  {
    const std::size_t was = _self_compressed;
    _self_compressed = now;
    return now > was && now - was > noise;
  }

  // THE measurement. `examined`/`incore` are pages of the RESIDENT SET
  // -- what the caller kept, not what it streamed. Any shortfall is a
  // block that has been compressed or swapped and is now costing a fault
  // per step on top of the RAM it still holds, so one is handed back and
  // the ceiling ratchets.
  //
  // One block at a time on purpose: the shortfall does not say how much
  // is too much, and shedding to the measurement would give back
  // everything the moment a single page moved. A step is seconds, so
  // converging over a few of them is free.
  std::size_t note_weight_residency(
      std::size_t examined, std::size_t incore,
      const std::function<std::size_t()>& evict_one)
  {
    if (examined == 0 || incore >= examined || _count <= 0) { return 0; }
    _evicted_from_ram = true;
    _growing = false;
    const std::size_t freed = release(1, evict_one);   // ratchets
    return freed;
  }

  // Whether any of the resident set was ever found outside RAM.
  bool weights_were_evicted() const { return _evicted_from_ram; }

  // A forward that found nothing wrong: either the resident set was
  // walked and every page was in RAM, or there was no reason to walk it
  // (nothing of ours is being compressed). After enough of them in a
  // row the ratchet lifts by ONE block.
  //
  // WHY THE RATCHET CANNOT SIMPLY LATCH. A ceiling that only ever falls
  // is a one-way door, and the first shed decides the entire run no
  // matter what the box does afterwards. MEASURED on the M5 16 GB box:
  // a MiniMax-H3 run admitted one 206 MB block, measured it out of RAM
  // on the next step -- during the AdaLN bake, the single tightest
  // moment of the run, which then RETIRED 13.2 GB of per-step
  // projections -- and shed it. That took the resident set to zero, so
  // release() ratcheted the ceiling to zero, and `_bytes + block > 0`
  // is true for every block forever after. The remaining ~29 steps
  // streamed the whole checkpoint with RAM half free, on the strength
  // of one sample taken at the worst instant available.
  //
  // Additive increase against the one-block decrease: recovery is
  // slower than retreat, so a box that really is full sheds faster than
  // it climbs and settles, while a box that was only momentarily tight
  // gets its room back. If the block is squeezed out again the next
  // measurement sheds it again -- which costs the walk and a block the
  // model was going to stream anyway, not correctness.
  void note_healthy_forward()
  {
    if (_shed_this_forward) { return; }
    if (_ceiling == kNoCeiling || _block_hint == 0) { return; }
    if (++_quiet < _quiet_forwards) { return; }
    _quiet = 0;
    // Saturating: the ceiling is a byte count, and lifting it past the
    // sentinel would read as "never ratcheted".
    if (_ceiling > kNoCeiling - _block_hint) { _ceiling = kNoCeiling - 1; }
    else { _ceiling += _block_hint; }
  }

  // How many quiet forwards buy one block back. Default 3: long enough
  // that a shed is not undone by the next step, short enough that a
  // 30-step schedule can still recover a useful set.
  void set_quiet_forwards(int n) { _quiet_forwards = n > 0 ? n : 1; }

  // The ground moved in a way no measurement of ours could have
  // predicted, so everything the ratchet concluded is stale: drop it.
  //
  // The case this exists for is MiniMax-H3's AdaLN bake, which retires
  // 13.2 GB of per-step projections partway through the first step. A
  // shortfall measured before that is a fact about a box carrying 13 GB
  // this model is about to stop carrying, and the slow decay above --
  // one block per three quiet forwards -- cannot undo it inside a
  // 6-step turbo schedule. Recovery by decay is for pressure whose
  // cause is invisible; this is for the case where the model KNOWS.
  //
  // Deliberately not automatic. A caller that resets on nothing in
  // particular has simply turned the ratchet off.
  void note_landscape_changed()
  {
    _ceiling = kNoCeiling;
    _quiet   = 0;
    _growing = true;
  }

  // Give back at least `want` bytes. `evict_one` frees ONE block and
  // returns its size, or 0 when there is nothing left to give. Returns
  // the total freed and ratchets the ceiling down.
  std::size_t release(std::size_t want,
                      const std::function<std::size_t()>& evict_one)
  {
    if (want == 0 || _count <= 0) { return 0; }
    std::size_t freed = 0;
    while (freed < want && _count > 0) {
      const std::size_t got = evict_one();
      if (got == 0) { break; }
      freed += got;
      _bytes -= (got > _bytes) ? _bytes : got;
      --_count;
      if (got > _block_hint) { _block_hint = got; }
    }
    if (freed > 0) {
      // Whatever level we were at was too high for the box as it is now.
      // Floored at ONE block: a ceiling of zero is a claim that this box
      // cannot hold a single block, which no measurement taken while
      // something else was spiking has earned. It also cannot be climbed
      // out of by an increase that is itself one block wide.
      _ceiling = _bytes > _block_hint ? _bytes : _block_hint;
      _growing = false;
      _shed_this_forward = true;
      _quiet = 0;
    }
    return freed;
  }

 private:
  static constexpr std::size_t kNoCeiling = (std::size_t)-1;

  std::size_t _reserve = 0;
  std::size_t _reserve_allocated = 0;
  bool        _reserve_declared = false;
  std::size_t _bytes   = 0;
  std::size_t _ceiling = kNoCeiling;
  int         _count   = 0;
  bool        _growing = true;
  bool        _logged_stop = false;
  bool        _log_stop = false;
  bool        _paged   = false;
  bool        _evicted_from_ram = false;
  int         _admitted_this_forward = 0;
  int         _per_forward_cap = 8;
  std::size_t _self_compressed = 0;
  std::size_t _block_hint = 0;
  bool        _shed_this_forward = false;
  int         _quiet = 0;
  int         _quiet_forwards = 3;
};

}  // namespace vpipe::genai

#endif
