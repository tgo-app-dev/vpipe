#ifndef VPIPE_GENERATIVE_MODELS_WEIGHT_SET_H
#define VPIPE_GENERATIVE_MODELS_WEIGHT_SET_H

// WeightSet -- one checkpoint's weights, owned by the model manager
// rather than by whichever model happened to load them first.
//
// The problem it solves. Until now every model opened its OWN
// MetalLlamaWeights mmap over the checkpoint directory and materialised
// its OWN SharedBuffers from it. Two stages naming one model directory
// therefore paid for two full copies of the weights, and the manager --
// the only thing with a session-wide view of memory -- had no idea
// either copy existed. Three consequences, all of which this fixes:
//
//   * DEDUP. The tensors are cached HERE, keyed by name, and handed out
//     as refcount-sharing aliases (SharedBuffer::subview over the whole
//     buffer). Two models built from one checkpoint share the bytes,
//     whatever order they were built in and whichever subset each of
//     them asks for. That last part matters: the old model-level dedup
//     had to guess which stage would build the SUPERSET model first,
//     which made it a race. Sharing per TENSOR has no such ordering.
//   * load_mapped EVERYWHERE. A mapped tensor aliases the mmap, so it
//     is only valid while the mmap lives -- which is why models that
//     dropped their MetalLlamaWeights after loading could not use it.
//     A WeightSet holds the mmap for as long as any holder holds the
//     WeightSet, so `Residency::Mapped` is always safe here.
//   * VISIBILITY. Bytes, residency and parts are queryable, so the
//     manager can report and act on what is actually resident.
//
// LIFETIME (the contract callers must honour). A `Mapped` tensor points
// into this object's mmap and a cached tensor's alias is refcounted
// against a buffer this object owns. **A model built from a WeightSet
// must hold the shared_ptr for its own lifetime.** That is also what
// makes the reference counting real: the checkpoint is unmapped when
// the last model using it goes away, not when the first one does.
//
// PARTS. A part is a named subset of the tensors -- "encoder" for a
// VAE's encoder half -- loaded through ensure_part() on first use and
// releasable on its own. It lets a graph skip weights it will not touch
// (no reference image => no VAE encoder) instead of loading everything
// the class can do.
//
// THREADING. Stage initialize() bodies run concurrently, so every entry
// point is serialised on one recursive mutex (recursive because a
// derived() builder normally calls tensor()). CACHED loads run UNDER
// the lock: two callers racing on one WeightSet are by definition
// loading the same tensor, so serialising them replaces duplicated work
// rather than parallel work.
//
// The exception is the uncached read() of a plain safetensors tensor --
// the block-streaming path. That is a fresh allocation plus a memcpy
// out of a read-only mmap, touching nothing this object owns, so it
// runs OUTSIDE the lock. It has to: streaming implies Copied, a
// streaming model re-reads blocks every forward pass, and holding the
// lock across those memcpys made two pipelines sharing one checkpoint
// take turns instead of overlapping (measured 1.10x vs ~1.5x for two
// threads; see weight_set.streaming_reads_run_concurrently). Mapped
// reads and GGUF reads stay locked -- both mutate shared state.

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::metal_compute { class MetalCompute; }

namespace vpipe::genai {

class WeightSet : public WeightOwner,
                  public std::enable_shared_from_this<WeightSet> {
public:
  // How a tensor is materialised.
  //
  //   Mapped -- zero-copy view into the mmap. The pages are clean and
  //     file-backed, so the OS reclaims and re-faults them by itself:
  //     no parking, no reload path, nothing to get wrong. The default,
  //     and the right answer whenever the bytes are used as they sit on
  //     disk.
  //   Copied -- an owned UMA buffer. Necessary when the caller writes
  //     to the tensor or when the on-disk offset is not GPU-bindable
  //     (load_mapped falls back to this by itself). Owned bytes are
  //     what the residency registry can park, so these are the tensors
  //     park()/reactivate() act on.
  enum class Residency : std::uint8_t { Mapped, Copied };

  // Open `dir` (a HF-style model directory; see
  // MetalLlamaWeights::open_model for the layouts recognised). Returns
  // null when the checkpoint cannot be opened. Prefer
  // GenerativeModelManager::weight_set(), which dedups and registers
  // the result -- this is the un-cached back door for tests.
  static std::shared_ptr<WeightSet> open(const std::string&        dir,
                                         const SessionContextIntf* session);

  ~WeightSet() override;

  WeightSet(const WeightSet&)            = delete;
  WeightSet& operator=(const WeightSet&) = delete;

  const std::string& dir() const noexcept { return _dir; }

  // The underlying checkpoint, for callers that need info() /
  // tensor_names() / their own read strategy (the streaming DiTs re-read
  // blocks per forward and deliberately do not cache).
  const MetalLlamaWeights& src() const noexcept { return *_wts; }

  bool has(const std::string& name) const;

  // A cached tensor, as a refcount-sharing alias. Repeat calls with the
  // same name return aliases of ONE buffer -- that is the dedup. Empty
  // if the tensor is missing or cannot be materialised.
  //
  // `part` attributes the bytes to a named part so release_part() can
  // drop them later; "" means the always-resident trunk.
  metal_compute::SharedBuffer
  tensor(const std::string&           name,
         metal_compute::MetalCompute* mc,
         Residency                    res  = Residency::Mapped,
         const std::string&           part = {});

  // A plain UNCACHED read: the bytes come back and the CALLER owns them.
  //
  // Use this for a tensor the model transforms and then drops (the
  // per-linear pieces an LM row-concatenates into one fused matrix), or
  // one it retains in its own struct and nothing else will want. Caching
  // those would keep a second, redundant copy alive next to the product
  // -- the cache is only free when the entry costs no owned bytes, which
  // is true of Mapped views and not of Copied ones.
  metal_compute::SharedBuffer
  read(const std::string&           name,
       metal_compute::MetalCompute* mc,
       Residency                    res = Residency::Mapped);

  // An UNCACHED read of the same tensor: the bytes come back and are
  // NOT retained, so the buffer dies with the caller's handle.
  //
  // This is the block-streaming path. A memory-bounded DiT re-reads
  // every block from the checkpoint on each forward pass precisely so
  // the weight set does NOT hold them -- caching one would defeat the
  // whole point and put the entire model back in RAM. Going through the
  // set anyway (rather than the model keeping a private mmap) is what
  // lets the manager see that streaming is happening and how much it is
  // moving; stats() counts these separately from resident bytes.
  //
  // Same mechanics as read(); the difference is that these are COUNTED
  // as streaming throughput. Keep them distinct -- a one-time load
  // reported as streaming traffic would misread as a bounded model
  // thrashing.
  //
  // There is no `part` here on purpose: nothing is retained, so there
  // would be nothing for a part to own or release.
  metal_compute::SharedBuffer
  stream_tensor(const std::string&           name,
                metal_compute::MetalCompute* mc,
                Residency                    res = Residency::Mapped);

  // Refill a buffer the CALLER already owns with a streamed tensor's raw
  // bytes. Counted as streaming throughput exactly like stream_tensor(),
  // and the only read path here that allocates nothing.
  //
  // This is the one WeightSet read whose result may legitimately be
  // WRITTEN through, because the memory is not this set's: `dst` came
  // from the caller and no cache entry aliases it. Everything the set
  // hands out is shared and therefore immutable (see the integrity check
  // in the class comment); this hands out nothing.
  //
  // It exists so a block-streamed model can keep TWO destinations for
  // the whole run and alternate them, instead of allocating and freeing
  // a block's worth of buffers on every block of every forward. The read
  // itself is also several times faster -- see
  // MetalLlamaWeights::pread_into, which this forwards to.
  //
  // NO CONVERSION. The bytes arrive as the checkpoint stores them, so a
  // caller whose compute dtype differs has to convert afterwards, which
  // is only possible in place when the widths match (F16 -> BF16 does,
  // F32 -> BF16 does not). Returns false -- having possibly written part
  // of `dst` -- when the tensor is missing, `cap` is too small, the
  // checkpoint is GGUF (converted, not copied), or the read failed. A
  // false means the caller must rewrite the whole buffer by another
  // route, not top this one up.
  bool stream_into(const std::string& name, void* dst, std::size_t cap);

  // Same, for a tensor the caller must TRANSFORM before use. `build`
  // runs every time -- there is no cache to hit -- and its result is
  // counted as streamed. Exists so a streaming model reaches for one
  // obviously-uncached call rather than remembering to bypass
  // derived(), which would silently pin the block it just streamed.
  metal_compute::SharedBuffer
  stream_derived(
      const std::function<metal_compute::SharedBuffer()>& build);

  // A cached DERIVED tensor: one the model builds by transforming the
  // checkpoint's bytes (dequantised, transposed, padded, an HWIO twin).
  // `key` must name the transform AND every config input that changes
  // the resulting bytes, since that key is all the cache compares --
  // two models whose configs disagree must not pick each other's up.
  // `build` runs only on a miss, and may call tensor().
  //
  // Derived entries are NOT parked: the registry's contract is that a
  // reclaimed buffer can be re-read, and the transform that produced
  // these is not retained. They are still shared and still released
  // with their part.
  metal_compute::SharedBuffer
  derived(const std::string&                                  key,
          const std::function<metal_compute::SharedBuffer()>& build,
          const std::string&                                  part = {});

  // ---- parts -------------------------------------------------------

  // True once ensure_part(part, ...) has run its loader successfully.
  bool part_ready(const std::string& part) const;

  // Run `load` once for `part` and remember the outcome, so callers can
  // ask for a part on every use without re-loading it. A part that
  // failed is not retried (the failure is a bad checkpoint, not a
  // transient), and the previous result is returned instead.
  bool ensure_part(const std::string&           part,
                   const std::function<bool()>& load);

  // Drop every cached tensor attributed to `part` and forget that it
  // loaded, so a later ensure_part() rebuilds it. Returns the bytes
  // released (a lower bound: aliases held by live models keep their
  // buffers alive until they too go away). Names the trunk when passed
  // "", which is almost never what a caller wants.
  std::size_t release_part(const std::string& part);

  // ---- WeightOwner (residency) -------------------------------------

  void for_each_weight(
      const std::function<void(metal_compute::SharedBuffer&)>& cb) override;
  bool reload_weights() override;
  std::string weight_label() const override;

  // Hold _mu across the registry's whole reactivate-and-reload span.
  //
  // The gap this closes: the registry drops its own lock between taking
  // the buffers back and calling reload_weights(), and in that gap the
  // entries are no longer marked parked but their contents were
  // discarded. A concurrent tensor() would hit the cache and hand out
  // an alias to garbage. Holding _mu makes such a caller WAIT for the
  // reload instead -- which is the behaviour it wants anyway.
  void begin_restore() override;
  void end_restore(bool ok) override;

  // ---- integrity (opt-in) ------------------------------------------
  //
  // Cached tensors are immutable after load: two models share one
  // buffer, so a write through either would corrupt the other. Nothing
  // in the type system enforces that -- SharedBuffer::contents() hands
  // out a mutable pointer -- and the mapped path gets it for free
  // (PROT_READ) while the copied path, which is most of the stack, does
  // not. This turns the convention into something the suite can check.
  //
  // Enabled by VPIPE_WEIGHT_INTEGRITY=1, read once at open: hashing
  // every tensor at load costs a full pass over the weights, which is
  // seconds on a 20 GB model and pointless in production. Off, the
  // hashing is skipped entirely and verify_integrity() reports nothing.
  bool integrity_enabled() const noexcept { return _integrity; }

  // Re-hash every cached tensor and return how many no longer match
  // what they hashed to at load. Zero is the expected answer, always.
  // Also zero when integrity checking is off -- ask integrity_enabled()
  // to tell "clean" from "not checked".
  std::size_t verify_integrity() const;

  struct Stats {
    std::size_t entries      = 0;
    std::size_t bytes        = 0;   // mapped + copied + derived
    std::size_t mapped_bytes = 0;   // OS-reclaimable, no reload path
    std::size_t copied_bytes = 0;   // owned; parkable + reloadable
    std::size_t parts        = 0;   // parts that loaded successfully
    // Block streaming, CUMULATIVE since this set was opened. Nothing
    // here is resident -- it is throughput, not occupancy -- but it is
    // the cost a memory-bounded model is paying to stay small, which is
    // exactly what someone reading a memory report needs to see.
    std::size_t streamed_reads = 0;
    std::size_t streamed_bytes = 0;
    // Where that streaming time went, split at the one seam that is
    // real: allocating the destination buffer, versus the memcpy that
    // fills it. See MetalLlamaWeights::LoadCost -- the fetch half is a
    // disk read or a page-cache copy depending on the source, and the
    // achieved rate is what tells them apart.
    double      streamed_alloc_ms = 0.0;
    double      streamed_fetch_ms = 0.0;
  };
  Stats stats() const;

  // Set by GenerativeModelManager so this set's owned tensors take part
  // in session-wide residency policy. `reg` keeps the registration
  // alive; `registry` is what lets this set REACTIVATE itself on the
  // next access after the policy parked it. Not for general use.
  void set_registration(WeightRegistry::Registration reg,
                        WeightRegistry*              registry = nullptr);

  // ---- residency policy hooks --------------------------------------

  // Recorded by the REGISTRY, through WeightOwner::note_parked, when it
  // hands these pages to the kernel or takes them back. NOT settable by
  // a caller, and that is the point: this flag is the only thing that
  // will ever take the pages back, so a park that set the kernel's half
  // and forgot this one left every later read serving buffers the
  // kernel was free to empty.
  //
  // A parked set is still fully usable: the next access reactivates it
  // first (see ensure_active_), reloading from disk only if the kernel
  // actually took the pages.
  void note_parked(bool on) override;
  bool parked() const noexcept;

  // ---- may this outlive the launch that opened it? ---------------------
  //
  // The removable pool keeps a released checkpoint alive across launches
  // so a relaunch over the same model pays no reload. That is only sound
  // for a set that is still a plain checkpoint. A set SPECIALISED to a
  // run's parameters is not: handing it to a launch that does not share
  // them gives that launch weights which are silently wrong for it.
  //
  // Said by whoever does the specialising, because nothing else can know
  // -- by the time the pool sees the set, the only thing left is a byte
  // count.
  //
  // Default TRUE, which is the safe default HERE and not the dangerous
  // one: a set nobody specialised is a plain checkpoint. A caller that
  // specialises one and forgets to say so is a thing to catch in review,
  // not to guess at by refusing to pool anything.
  bool recyclable() const noexcept;
  // `why` is recorded so a set that will not be pooled can say what
  // stopped it -- a checkpoint quietly reloading every launch otherwise
  // looks exactly like one the pool never saw.
  void set_not_recyclable(std::string why);
  const std::string& unrecyclable_reason() const noexcept;

  // Monotonic tick of the last time a tensor was asked for. The cap
  // policy parks least-recently-used first, so that a set nothing has
  // touched for a while goes before one in active use.
  std::uint64_t last_use() const noexcept;

private:
  WeightSet(std::string dir, MetalLlamaWeights wts,
            const SessionContextIntf* session);

  struct Entry {
    metal_compute::SharedBuffer buf;          // the OWNING handle
    std::string                 src_name;     // "" => derived
    Residency                   res = Residency::Mapped;
    std::string                 part;
    // Contents hash at load, and again after each reload. Only
    // meaningful when _integrity is on; 0 otherwise.
    std::uint64_t               hash = 0;
  };

  // Caller holds _mu.
  metal_compute::SharedBuffer alias_(const Entry& e) const;
  // Bump _last_use and, if the residency policy parked this set, take
  // it back before handing out any bytes. Called at the top of every
  // accessor. Cheap when not parked: one relaxed atomic store.
  void ensure_active_();
  // 0 when integrity checking is off, so callers need no branch.
  std::uint64_t hash_(const metal_compute::SharedBuffer& b) const;

  const SessionContextIntf*            _session = nullptr;
  bool                                 _integrity = false;
  // True between begin_restore() and end_restore(), during which this
  // object's _mu is held by the restoring thread.
  std::atomic<bool>                    _restoring{false};
  std::atomic<bool>                    _parked{false};
  // See recyclable(). Default true: a set nobody specialised may be
  // handed to the next launch.
  std::atomic<bool>                    _recyclable{true};
  std::string                          _unrecyclable_why;
  std::atomic<std::uint64_t>           _last_use{0};
  WeightRegistry*                      _registry = nullptr;
  // Atomic, not _mu-guarded: these are pure bookkeeping on the
  // streaming hot path, and taking the set's lock to add two numbers
  // would serialise concurrent streamers on the counters alone.
  std::atomic<std::size_t>             _streamed_reads{0};
  std::atomic<std::size_t>             _streamed_bytes{0};
  // Microseconds, so the accumulator stays integral and lock-free.
  std::atomic<std::uint64_t>           _streamed_alloc_us{0};
  std::atomic<std::uint64_t>           _streamed_fetch_us{0};
  std::string                          _dir;
  mutable std::recursive_mutex         _mu;
  std::optional<MetalLlamaWeights>     _wts;
  std::unordered_map<std::string, Entry> _cache;
  std::unordered_map<std::string, bool>  _parts;
  // Remembered from the first materialisation so reload_weights() can
  // re-read without the caller passing a backend back in.
  metal_compute::MetalCompute*         _mc = nullptr;
  // Declared LAST: dropping the registration must happen before the
  // cache it walks is destroyed.
  WeightRegistry::Registration         _reg;
};

// How a model should get its weights.
//
// Goes through the session's model manager, so the checkpoint is
// opened once no matter how many models name it, and the manager can
// see and report what is resident. Falls back to a private,
// unregistered WeightSet::open() when there is no session or no
// manager -- unit tests and the offline tools (quantize, calibration,
// lora-fuse) run outside a pipeline and legitimately want their own.
//
// `variant` separates sets that must NOT be shared even though they
// name one directory (a differently-transformed view of the same
// files). Empty for the ordinary case.
//
// Callers must KEEP the returned shared_ptr for as long as any tensor
// taken from it is in use; see the lifetime note on WeightSet.
std::shared_ptr<WeightSet>
open_weight_set(const std::string&        dir,
                const SessionContextIntf* session,
                const std::string&        variant = {});

}  // namespace vpipe::genai

#endif
