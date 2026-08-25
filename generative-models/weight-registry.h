#ifndef VPIPE_GENERATIVE_MODELS_WEIGHT_REGISTRY_H
#define VPIPE_GENERATIVE_MODELS_WEIGHT_REGISTRY_H

// WeightRegistry -- session-level residency policy over every loaded
// model's weights.
//
// The problem it solves. Model weights are the largest, longest-lived
// allocations in the process (a 24 GB DiT, a 10 GB TTS pair), they are
// IMMUTABLE once loaded, and they are idle most of the time -- a
// stopped pipeline's weights are pure occupancy. Until now the only
// ways to reclaim that were to destroy the model (paying a full reload
// on the next use, tens of seconds) or to keep it resident and starve
// everything else. Neither is right when the answer usually wanted is
// "let go of it IF something else needs the RAM, otherwise keep it".
//
// That is exactly what Metal's purgeable-resource state provides, and
// SharedBuffer::mark_inactive() / reactivate() wrap it: parked pages
// are reclaimable but survive when nothing is competing, and
// reactivate() REPORTS whether they survived so a purge is never
// silent. This class applies that across models and adds the policy:
// who is parked, when, and what happens on the way back.
//
// WHO DECIDES. This class parks whatever it is handed and asks no
// questions -- the caller owns the judgement. There is one caller,
// GenerativeModelManager::park_if_unborrowed_(), and one rule: a
// checkpoint anything is still BORROWING is never parked, because the
// borrower reads aliases of these very buffers in a forward pass that
// never asks the set for anything and so never reactivates them. An
// owner that wants to be reclaimable while it is loaded has to say so
// itself, through a hook it calls at a forward boundary; none does
// today, and until one does, "idle" means "nobody is holding it".
//
// Registration, not ownership. A model keeps owning its buffers and
// implements for_each_weight(); the registry stores no buffer pointers
// and calls the enumerator each time instead, so nothing dangles when a
// model's containers reallocate and the forward path is untouched.
//
// NOT the registry's business:
//   * mmap-backed weights (MetalLlamaWeights::load_mapped). Clean
//     file-backed pages already evict and re-fault under pressure --
//     the OS does this better than we can, for free. mark_inactive()
//     declines them anyway (they are subviews of a shard wrap).
//   * per-model block streaming (shared/stream-sizing.h). Its pinned
//     prefix is greedy over per-block byte sizes in stream order, which
//     needs the model's own layout knowledge; the registry has none.

#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::genai {

// Implemented by anything holding model weights.
class WeightOwner {
public:
  virtual ~WeightOwner() = default;

  // Call `cb` for every LARGE, IMMUTABLE-after-load weight buffer.
  //
  // Do NOT list scratch, activation or K/V buffers: those are written
  // during a forward pass, so parking them loses live state rather than
  // a re-readable copy. Listing an mmap-backed buffer is harmless (it
  // declines to park) but pointless.
  virtual void for_each_weight(
      const std::function<void(metal_compute::SharedBuffer&)>& cb) = 0;

  // Re-read the weights from disk. Called ONLY when reactivation finds
  // the kernel discarded the pages, and only between begin_restore()
  // and end_restore(). Returning false means the owner could not be
  // restored and must be treated as unusable -- the caller's fallback
  // is to drop and rebuild the model.
  //
  // This is the ONE place a loaded weight buffer is written after load,
  // and the registry is the only caller, so an owner that wants to
  // enforce immutability can treat everything outside this bracket as
  // read-only without any cooperation from client code.
  virtual bool reload_weights() { return false; }

  // Bracket the registry's reactivation of this owner.
  //
  // Why it exists: between the buffers being taken back and
  // reload_weights() re-filling them, their contents are UNDEFINED.
  // Without a bracket the owner's own lock is dropped in that gap, so a
  // concurrent reader can be handed a buffer full of discarded pages.
  // begin_restore() lets the owner hold whatever lock guards its
  // readers across the whole span; end_restore() releases it.
  //
  // Always called in pairs, begin before the buffers are touched and
  // end after the reload (or immediately, when nothing was discarded
  // and no reload was needed). `ok` is false only when a reload was
  // attempted and failed, which leaves the owner unusable.
  //
  // The default pair is a no-op: an owner with no concurrent readers,
  // or one that reloads atomically, needs nothing here.
  virtual void begin_restore() {}
  virtual void end_restore(bool ok) { (void)ok; }

  // The registry has just handed this owner's pages to the kernel
  // (`parked` true) or taken them back (false).
  //
  // WHY THIS IS A HOOK AND NOT THE CALLER'S JOB. The kernel's purgeable
  // state and the owner's record of it are two halves of one fact, and
  // an owner that reactivates itself on next use keys that off its own
  // half. Set only the first and the pages are volatile with nothing
  // left that will ever take them back: the next reader is handed
  // buffers the kernel may have emptied and gets ZEROS, silently,
  // because the warning for exactly that case keys off the flag that
  // was never set.
  //
  // MEASURED: three of the four call sites that parked a WeightSet
  // remembered to set it and the fourth -- the removable pool -- did
  // not, so any checkpoint that had been pooled once served emptied
  // buffers from that point on. Driving both halves from park() and
  // reactivate() removes the opportunity to set one without the other.
  //
  // Called with the registry's lock HELD, so that the two halves flip
  // together: do not re-enter the registry from here, and keep it to
  // bookkeeping. Default no-op -- an owner with no reactivate-on-use
  // path of its own needs nothing.
  virtual void note_parked(bool parked) { (void)parked; }

  // Short name for logs ("krea2 DiT /path/to/transformer").
  virtual std::string weight_label() const { return {}; }
};

class WeightRegistry {
public:
  explicit WeightRegistry(const SessionContextIntf* session)
    : _session(session) {}

  WeightRegistry(const WeightRegistry&)            = delete;
  WeightRegistry& operator=(const WeightRegistry&) = delete;

  // RAII registration. Destroying it removes the owner, so a model that
  // dies while parked (or mid-reactivate on another thread) cannot be
  // walked afterwards -- remove() waits for any in-flight policy pass.
  class Registration {
  public:
    Registration() = default;
    Registration(WeightRegistry* reg, std::uint64_t id) noexcept
      : _reg(reg), _id(id) {}
    ~Registration() { reset(); }

    Registration(const Registration&)            = delete;
    Registration& operator=(const Registration&) = delete;
    Registration(Registration&& o) noexcept
      : _reg(o._reg), _id(o._id) { o._reg = nullptr; o._id = 0; }
    Registration& operator=(Registration&& o) noexcept
    {
      if (this != &o) {
        reset();
        _reg = o._reg; _id = o._id;
        o._reg = nullptr; o._id = 0;
      }
      return *this;
    }

    void reset() noexcept;
    explicit operator bool() const noexcept { return _id != 0; }

  private:
    WeightRegistry* _reg = nullptr;
    std::uint64_t   _id  = 0;
  };

  // Register `owner`. It must outlive the returned Registration.
  Registration add(WeightOwner* owner);

  // Park one owner's weights: they stay allocated and addressable but
  // become reclaimable. Returns the bytes handed over (0 if the owner
  // is unknown, already parked, or holds nothing parkable).
  std::size_t park(WeightOwner* owner);

  // Take the weights back. Returns true when every buffer's contents
  // SURVIVED -- the fast path, a state flip with no I/O.
  //
  // False means at least one buffer was discarded. In that case the
  // owner's reload_weights() has already been called; `*reloaded` (when
  // given) says whether that succeeded. A false return with
  // *reloaded == false leaves the owner unusable.
  bool reactivate(WeightOwner* owner, bool* reloaded = nullptr);

  // Park every registered owner. The memory-pressure response: each one
  // reactivates on its next use, reloading only if its pages were taken.
  std::size_t park_all();

  struct Stats {
    std::size_t owners        = 0;
    std::size_t parked_owners = 0;
    std::size_t parked_bytes  = 0;
    std::size_t total_bytes   = 0;
  };
  Stats stats() const;

private:
  friend class Registration;

  struct Entry {
    std::uint64_t id     = 0;
    WeightOwner*  owner  = nullptr;
    bool          parked = false;
    std::size_t   bytes  = 0;   // last measured parkable bytes
  };

  void remove_(std::uint64_t id) noexcept;
  Entry* find_(WeightOwner* owner);   // caller holds _mu

  const SessionContextIntf* _session = nullptr;
  mutable std::mutex        _mu;
  std::vector<Entry>        _entries;
  std::uint64_t             _next_id = 1;
};

}  // namespace vpipe::genai

#endif
