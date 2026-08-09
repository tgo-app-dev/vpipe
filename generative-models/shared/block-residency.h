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
//      pressure it was just asked to relieve.
//
// This class is policy ONLY. It never owns or frees a block: the models
// differ in how they hold them (one list for Krea-2 / Qwen-Image /
// MiniMax-H3, two for FLUX.2 / Boogu, which have separate double- and
// single-stream stacks), so the caller keeps its containers and this
// answers admit/evict questions about them.
//
// Usage, per forward:
//     _resid.begin_forward();                       // re-arm growth
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
  // How much must stay free for the rest of the forward. 0 (the default)
  // disables growth entirely, which is the old pure-streaming behaviour --
  // so a model that has not been taught what its activations cost keeps
  // exactly the behaviour it had, rather than growing against a guess.
  void set_reserve(std::size_t bytes) { _reserve = bytes; }
  std::size_t reserve() const { return _reserve; }

  std::size_t bytes() const { return _bytes; }
  int count() const { return _count; }

  // Re-arm for a new forward. Growth is re-probed once per forward rather
  // than once per block; the RATCHET deliberately survives, so a set that
  // was cut back does not simply refill on the next step.
  void begin_forward()
  {
    _growing = true;
    _logged_stop = false;
  }

  // May a block of `block_bytes` stay resident? Policy only -- the caller
  // keeps the block and calls note_admitted() if this says yes.
  bool admit(metal_compute::MetalCompute* mc, std::size_t block_bytes)
  {
    if (!_growing || _reserve == 0 || block_bytes == 0 || mc == nullptr) {
      return false;
    }
    if (_bytes + block_bytes > _ceiling) {
      _growing = false;                 // ratcheted by an earlier release
      return false;
    }
    const auto mb = mc->memory_budget();
    if (mb.recommended == 0) { return false; }
    // block + reserve + one more block: the trailing term is the
    // hysteresis gap. See property 3 above.
    if (!mb.fits_physical(block_bytes + _reserve + block_bytes)) {
      if (!_logged_stop && _count > 0 && mc->session() != nullptr) {
        _logged_stop = true;
        mc->session()->log_debug(fmt(
            "block residency: stopped growing at {} blocks ({} MB) -- {} MB "
            "reclaimable against a {} MB reserve", _count, _bytes >> 20,
            mb.available_physical >> 20, _reserve >> 20));
      }
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
    }
    if (freed > 0) {
      // Whatever level we were at was too high for the box as it is now.
      _ceiling = _bytes;
      _growing = false;
    }
    return freed;
  }

 private:
  std::size_t _reserve = 0;
  std::size_t _bytes   = 0;
  std::size_t _ceiling = (std::size_t)-1;
  int         _count   = 0;
  bool        _growing = true;
  bool        _logged_stop = false;
};

}  // namespace vpipe::genai

#endif
