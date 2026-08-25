#ifndef VPIPE_GENERATIVE_MODELS_SHARED_BLOCK_SLOTS_H
#define VPIPE_GENERATIVE_MODELS_SHARED_BLOCK_SLOTS_H

// TWO REUSABLE DESTINATIONS for a block-streamed stack, refilled in
// place with pread(2), with the next block's read issued under the
// current block's GPU work.
//
// What this replaces is a block's worth of SharedBuffers allocated and
// freed on every block of every forward. That costs the allocation
// itself and, far more, the demand-faulted memcpy out of the shard
// mapping that fills them. MEASURED on an M4 Pro, cold, over each
// family's own checkpoint on the internal SSD -- four arms, interleaved,
// byte-balanced groups, arm-to-group assignment rotated:
//
//   mmap + memcpy      3.1-4.0 GB/s      what a per-block allocation does
//   pread into a       3.4-4.9 GB/s      F_NOCACHE, no readahead
//     buffer that      4.5-6.3 GB/s      cached
//     already exists
//   mmap wrap + fault  0.7-1.0 GB/s      no copy at all -- page faults
//
// So the read is worth up to 1.3-1.6x -- and NOT always: measured end to
// end over four stacks, one gained 1.47x, one 1.31x, one 1.19-1.38x,
// and one tied. A checkpoint small enough to sit in the page cache
// leaves the mapped arm reading warm while pread's F_NOCACHE goes to
// the device, which is most of the difference; the 1.47x is a 17 GB
// checkpoint that fits no cache it would ever stream on. The zero-copy
// wrap is worth avoiding outright: it copies nothing and is still 4x
// slower than copying, because the kernel tracks residency at 4 KB
// granularity over a file of tens of gigabytes. On a SLOW device none
// of this shows -- an external Thunderbolt SSD at 0.84 GB/s makes every
// arm a tie -- so this is a win on the drive a checkpoint should be on,
// and no loss anywhere.
//
// WHY TWO AND NOT MORE. The depth that matters is one outstanding read,
// which is what the prefetch issues. A deeper ring would buy nothing
// (the GPU takes blocks strictly in order) and would cost a block of
// live memory per extra slot on a box that is streaming precisely
// because it has none to spare.
//
// WHAT THIS DOES NOT CHANGE. A promoted block is COPIED out of its slot,
// because the slot has to stay put for the next read -- so the resident
// set, its growth and its ratchet all work exactly as they did. The copy
// is not the waste it looks like: the destination has to be allocated
// either way, so what it adds is one memcpy, against a read that is
// 1.3-1.6x faster for every block that is NOT promoted.
//
// Usage, per forward:
//
//     slots.begin_forward();
//     for (int i = 0; i < n; ++i) {
//       if (resident(i)) { ... ; continue; }
//       const Block* b = slots.acquire(i);
//       if (b == nullptr) { slots.join(); return fail(...); }
//       ... encode ...
//       Fence f = stream.commit();
//       slots.prefetch(next);                 // under the GPU's work
//       f.wait();
//       if (admit) { slots.promote_into(resident[i]); }
//     }
//     slots.join();
//
// JOIN BEFORE EVERY EARLY RETURN. A read may be in flight into a slot,
// and destroying this object -- or freeing the slots -- while a reader
// thread is writing into them is a use-after-free. `join()` is
// idempotent and cheap when nothing is outstanding, and the destructor
// calls it, but a caller that returns while holding a fence must still
// order the two correctly.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/vpipe-format.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <future>
#include <string>

namespace vpipe::genai {

// HOW A DESTINATION RELATES TO THE CHECKPOINT, which the model must
// state per tensor because a wrong answer here is SILENT -- the buffer
// ends up the right size and full of plausible numbers.
enum class Placement {
  // The checkpoint's own words, placed untouched: a quantized pack's
  // u32 codes, or a bf16 tensor a bf16 forward reads as it sits.
  kRaw,
  // Read as bf16. Places u32 and bf16 untouched and converts f16 IN
  // PLACE, which is legitimate only because they are the same width.
  // A model that converts f16 by some other route must NOT say this,
  // or the bytes are converted twice.
  kBf16,
  // NOT addressable by a raw read at all, so ALWAYS rebuilt: a norm
  // folded to (1 + w), a narrowing to f16, a permutation. Enumerating
  // it is still required -- a slot whose derived tensors were skipped
  // would keep the FIRST block's values for the whole run, which is
  // the quietest possible way to be wrong.
  kDerived,
};

template <class Block>
class BlockSlots {
 public:
  using TensorFn = std::function<void(const std::string& name,
                                      metal_compute::SharedBuffer& dst,
                                      Placement how)>;

  // Everything the policy cannot know about a model's blocks. All five
  // are required; a BlockSlots without them stays permanently off,
  // which is the same behaviour the model had before it was given any.
  struct Ops {
    // Visit every (name, destination, raw) of block `index`. The model
    // builds the checkpoint names from the index; this class never
    // knows what a block is called.
    std::function<void(int index, Block&, const TensorFn&)> each;
    // Rebuild ONE tensor a raw read could not place. Returns empty on
    // failure.
    std::function<metal_compute::SharedBuffer(const std::string& name,
                                              Placement how)>
        rebuild_one;
    // Allocate and fill a block from scratch: how a slot is first built
    // and what the whole mechanism falls back to.
    std::function<bool(int index, Block&)> build;
    // Allocate `dst` with `src`'s shapes and flags, optionally copying
    // the bytes. One function for two uses because a promotion and a
    // second slot differ only in whether the contents come along.
    std::function<bool(const Block& src, Block& dst, bool copy)> clone;
    std::function<std::size_t(const Block&)> bytes;
    std::function<bool(const Block&)> empty;
    // OPTIONAL. Fill an EXISTING destination for a tensor the raw read
    // could not place -- an f32 source, whose bytes are twice the width
    // of a bf16 destination and so have nowhere to go without a second
    // buffer. A model that preads into a scratch and converts out of it
    // keeps even that half off the mapped path; returning false falls
    // back to `rebuild_one`, which allocates.
    //
    // Worth having only where a meaningful share of a block is f32. A
    // checkpoint that is bf16 and u32 throughout never reaches this.
    std::function<bool(const std::string& name,
                       const metal_compute::SharedBuffer& dst)>
        fill_unservable;

    // OPTIONAL. Rebuild the block's DERIVED tensors -- the ones with no
    // single checkpoint name, because they are a transform of several
    // (an interleaved gate|up, a permuted projection). Runs after every
    // refill, and is the reason a model that fuses at READ time can
    // keep doing so: moving the fuse to promotion instead would change
    // which kernel a streamed block runs through, and that is a
    // numerics change wearing a performance change's clothes.
    //
    // Empty means the block has no derived tensors, which is the
    // simplest and fastest case.
    std::function<bool(int index, Block&)> post_refill;
  };

  BlockSlots() = default;
  ~BlockSlots() { join(); }
  BlockSlots(const BlockSlots&) = delete;
  BlockSlots& operator=(const BlockSlots&) = delete;

  // `label` names the model in the log; `env_off` is a kill switch read
  // ONCE, on the first streamed block, so a run that never streams
  // never reads it.
  void configure(metal_compute::MetalCompute* mc, Ops ops,
                 std::string label, const char* env_off)
  {
    _mc      = mc;
    _ops     = std::move(ops);
    _label   = std::move(label);
    _env_off = env_off;
  }

  // Turn the mechanism off for this object's life, for an A/B against
  // the per-block-allocation path it replaces. Distinct from the env
  // kill switch only in who decides.
  void disable() { _off = true; _first = false; }

  bool usable() const
  {
    return _mc != nullptr && _ops.each && _ops.rebuild_one && _ops.build &&
           _ops.clone && _ops.bytes && _ops.empty;
  }

  // Re-arm for a forward. Does NOT drop the slots: they are the whole
  // point of surviving between forwards.
  void begin_forward() { _hit = 0; _started = 0; }

  int prefetch_hits() const { return _hit; }
  int prefetch_started() const { return _started; }
  bool on() const { return _ready && !_off; }
  bool paired() const { return _pair; }

  // Throw the slots away, e.g. when the stack's shape changes under
  // them. Joins first.
  void reset()
  {
    join();
    _slot[0] = Block{};
    _slot[1] = Block{};
    _fallback = Block{};
    _ready = false;
    _pair  = false;
    _cur   = 0;
  }

  // The block to run for `index`.
  // Uses a matching outstanding prefetch when there is one, and
  // otherwise reads now. Returns nullptr only when the block could not
  // be produced at all.
  const Block* acquire(int index)
  {
    if (!usable()) { return read_fallback_(index); }
    if (_first) {
      _first = false;
      _off = _env_off != nullptr && std::getenv(_env_off) != nullptr;
    }
    if (_off) { return read_fallback_(index); }

    if (!_ready && !allocate_(index)) {
      // The FIRST block could not even be built. That is not a slot
      // problem, so it is not sticky -- fall through to the ordinary
      // path and let it report.
      return read_fallback_(index);
    }

    // A read ahead for some OTHER block cannot serve this one, and
    // leaving its future outstanding would refuse every later prefetch
    // for the rest of the run (`_pf_fut.valid()` is the gate). Join and
    // discard. Unreachable while a caller prefetches exactly the block
    // it acquires next, which is the shape every caller has -- but the
    // cost of being wrong about that is silent serialisation, so it is
    // handled rather than assumed.
    if (_pf_index >= 0 && _pf_index != index) { join(); }

    int use = -1;
    if (_pf_index == index && _pf_slot >= 0 && _pf_fut.valid()) {
      // This block's read was issued under the PREVIOUS block's GPU
      // work, so waiting costs only the part that did not fit under
      // that window.
      const bool ok = _pf_fut.get();
      use = _pf_slot;
      _pf_index = -1;
      _pf_slot  = -1;
      if (!ok) { give_up_("prefetch of block " + std::to_string(index)); }
      else { ++_hit; }
    } else if (!_off) {
      // Never the slot an outstanding read is writing into.
      use = _pf_slot >= 0 ? (_pf_slot ^ 1) : _cur;
      if (!refill_(index, _slot[(std::size_t)use])) { give_up_(""); }
    }
    if (!_off && use >= 0) {
      _cur = use;
      return &_slot[(std::size_t)use];
    }
    return read_fallback_(index);
  }

  // Issue a read of block `index` into the free slot, if nothing is
  // already outstanding. Called between a commit and its wait, so the
  // read runs under the GPU work of the block just encoded.
  //
  // ONLY EVER INTO A SLOT PAIR, and that is a correctness rule rather
  // than a policy. A read ahead has to land somewhere the CURRENT block
  // is not: the GPU is still reading that one -- being under its work is
  // the entire point -- and a build assigns fresh handles over the old
  // ones, freeing them underneath it. With two slots the free
  // destination is `_cur ^ 1`. With one slot, or none, there is no free
  // destination, and the per-block fallback is the block in use. So a
  // model that cannot take a second slot, or whose checkpoint drove this
  // into the fallback, simply reads serially -- which is exactly the
  // behaviour it had before it was given any slots at all.
  //
  // (An earlier version did issue a fallback read, into the one
  // `_fallback` the current block was living in. It went unseen because
  // the slot route is what every supported checkpoint takes; the
  // fallback A/B is what surfaced it, as a GPU fault mid-stack.)
  //
  // Asking per block and never queueing means the moment it stops being
  // affordable the next iteration is serial again.
  void prefetch(int index)
  {
    if (index < 0 || _pf_index >= 0 || _pf_fut.valid()) { return; }
    // NOT YET. The slots are allocated by the first acquire, so a
    // prefetch before that has nowhere to go. MEASURED as "1 issued, 0
    // landed" on a model whose first layers are pinned, so the first
    // prefetch fires before any block has been acquired.
    if (!on() || !_pair || !_ready) { return; }
    _pf_index = index;
    _pf_slot  = _cur ^ 1;
    ++_started;
    const int dst = _pf_slot;
    _pf_fut = std::async(std::launch::async, [this, index, dst]() {
      return refill_(index, _slot[(std::size_t)dst]);
    });
  }

  // Join any outstanding read. Idempotent, and the destructor calls it
  // -- but a caller returning early must still call it before anything
  // that could free a slot.
  void join()
  {
    if (_pf_fut.valid()) { (void)_pf_fut.get(); }
    _pf_index = -1;
    _pf_slot  = -1;
  }

  // Put the block just run into `dst`, ready to be kept resident.
  //
  // A COPY when it came from a slot, because the slot has to stay put
  // for the next read; a MOVE when it came from the per-block fallback,
  // where nothing else refers to it. Callers do not have to know which,
  // which is the point -- getting it wrong is either a use-after-free
  // or a slot that silently stops being reused.
  bool promote_into(Block& dst)
  {
    if (_last_from_slot) {
      return _ops.clone(_slot[(std::size_t)_cur], dst, /*copy=*/true);
    }
    dst = std::move(_fallback);
    _fallback = Block{};
    return true;
  }

  // The bytes of the block just run, for the residency policy.
  std::size_t last_bytes() const
  {
    if (!_ops.bytes) { return 0; }
    return _ops.bytes(_last_from_slot ? _slot[(std::size_t)_cur]
                                      : _fallback);
  }

 private:
  const Block* read_fallback_(int index)
  {
    _last_from_slot = false;
    if (!_ops.build) { return nullptr; }
    // Serial by construction -- see prefetch(): nothing is ever read
    // ahead into this, because this is the block the GPU is using.
    if (!_ops.build(index, _fallback)) { return nullptr; }
    return &_fallback;
  }

  bool allocate_(int index)
  {
    if (!_ops.build(index, _slot[0])) { return false; }
    // The SECOND slot is refused when the box will not take a durable
    // block-sized allocation, and the run is then single-slot: no
    // prefetch, but the read shape and the absence of per-block
    // allocation -- the larger half of the win -- are unaffected.
    const auto mb = _mc->memory_budget();
    _pair = mb.recommended != 0 &&
            mb.fits_growth(_ops.bytes(_slot[0])) &&
            _ops.clone(_slot[0], _slot[1], /*copy=*/false);
    if (!_pair) { _slot[1] = Block{}; }
    _ready = true;
    _cur   = 0;
    _last_from_slot = true;
    if (_mc->session() != nullptr) {
      _mc->session()->log_debug(fmt(
          "{}: block slots ready ({} x {} MB{})", _label, _pair ? 2 : 1,
          _ops.bytes(_slot[0]) >> 20,
          _pair ? "" : ", single -- prefetch off"));
    }
    return true;
  }

  // PER TENSOR, not per block. A tensor a raw read cannot place is
  // rebuilt the way the model builds it, into the slot, and everything
  // around it still takes the fast path. Answering for the whole block
  // would give up a stack's worth of fast reads to avoid a handful of
  // awkward tensors -- an old quantized pack whose blocks are not
  // byte-identical to block 0 is exactly that checkpoint, and it does
  // not deserve the whole mechanism.
  //
  // `kUnservable` and `kFailed` are repaired the SAME way, which reads
  // odd until you notice what the repair is: it does not top the buffer
  // up, it builds a new one and REPLACES it. That is the right response
  // to a partly-written destination as well as to a dtype no raw read
  // can place -- and it is what lets a slot whose size no longer fits
  // recover, since the replacement is sized from the checkpoint.
  bool refill_(int index, Block& b)
  {
    bool ok = true;
    _ops.each(index, b, [&](const std::string& nm,
                            metal_compute::SharedBuffer& dst, Placement how) {
      if (!ok || dst.empty()) { return; }
      // A DERIVED tensor has no raw form to place, so it is rebuilt
      // every time rather than asked about.
      if (how == Placement::kDerived) {
        metal_compute::SharedBuffer rb = _ops.rebuild_one(nm, how);
        if (rb.empty()) { ok = false; return; }
        dst = std::move(rb);
        return;
      }
      // A destination that does not OWN its allocation is a subview of a
      // shard mapping, and those are mapped read-only -- writing into
      // one is a SIGBUS, not a wrong answer. A model whose streamed
      // reads are Copied never gets here; one that mixes the two would,
      // exactly once, at a point with no other symptom.
      if (!dst.is_owned()) {
        metal_compute::SharedBuffer rb = _ops.rebuild_one(nm, how);
        if (rb.empty()) { ok = false; return; }
        dst = std::move(rb);
        return;
      }
      const Refill r = refill_streamed_tensor(
          *_ws, nm, dst,
          how == Placement::kRaw ? RefillDst::kRaw : RefillDst::kBf16);
      if (r == Refill::kFilled) { return; }
      // kUnservable only: a kFailed destination is PARTLY WRITTEN, so
      // filling it in place would leave whatever the short read left
      // behind. That one has to be replaced.
      if (r == Refill::kUnservable && _ops.fill_unservable &&
          _ops.fill_unservable(nm, dst)) {
        return;
      }
      metal_compute::SharedBuffer rebuilt = _ops.rebuild_one(nm, how);
      if (rebuilt.empty()) { ok = false; return; }
      dst = std::move(rebuilt);
    });
    if (ok && _ops.post_refill) { ok = _ops.post_refill(index, b); }
    if (ok) { _last_from_slot = true; }
    return ok;
  }

  // Sticky, and said ONCE: a fallback that came and went would read as
  // an unexplained slowdown rather than a property of the checkpoint.
  void give_up_(const std::string& where)
  {
    _off = true;
    if (_mc != nullptr && _mc->session() != nullptr) {
      _mc->session()->log_debug(fmt(
          "{}: block slots cannot serve this checkpoint{} -- streaming "
          "per-block allocations, which re-read through the shard "
          "mapping rather than uncached", _label,
          where.empty() ? "" : " (" + where + ")"));
    }
    // Hand the slots back. They are a block each and nothing will read
    // them again this run.
    //
    // JOIN FIRST. A prefetch may be in flight into one of these, and
    // freeing a buffer a reader thread is writing into is the one way
    // this fallback could turn a recoverable refusal into a crash.
    join();
    _slot[0] = Block{};
    _slot[1] = Block{};
    _pair    = false;
    _ready   = false;
  }

 public:
  // The weight set the refill reads from. Set alongside configure();
  // separate because a model opens its set at load and may configure
  // the slots earlier.
  void set_weight_set(WeightSet* ws) { _ws = ws; }

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  WeightSet*                   _ws = nullptr;
  Ops                          _ops;
  std::string                  _label;
  const char*                  _env_off = nullptr;

  Block _slot[2];
  Block _fallback;
  int   _cur = 0;
  bool  _ready = false;
  bool  _pair  = false;
  bool  _off   = false;
  bool  _first = true;
  bool  _last_from_slot = false;

  int               _pf_index = -1;
  int               _pf_slot  = -1;
  std::future<bool> _pf_fut;
  int               _hit = 0;
  int               _started = 0;
};

}  // namespace vpipe::genai

#endif
