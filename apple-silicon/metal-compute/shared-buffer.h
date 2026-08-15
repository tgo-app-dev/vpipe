#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_SHARED_BUFFER_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_SHARED_BUFFER_H

#include "apple-silicon/metal-compute/buffer-view.h"

#include <cstddef>

namespace MTL { class Buffer; }

namespace vpipe::metal_compute {

class MetalCompute;

// Whether Metal's implicit hazard-tracking machinery covers a
// resource. Tracked (the default) lets Metal insert barriers
// automatically; Untracked hands sync responsibility to the caller
// via Fence / Event. Untracked is the right choice when the
// framework knows the dependency DAG (e.g. you call update_fence
// and wait_for_fence explicitly) -- it removes per-dispatch
// hazard-tracking cost.
enum class HazardTracking : std::uint8_t {
  Tracked   = 0,
  Untracked = 1,
};

// Owning handle for an MTL::Buffer in Shared storage mode (CPU and
// GPU share the same UMA address). Move-only; default-constructed
// instances are empty and safe to destroy.
//
// Allocated only through MetalCompute::make_shared_buffer(); the
// public constructor is the default constructor for moved-from /
// not-yet-assigned slots.
class SharedBuffer {
public:
  SharedBuffer() noexcept = default;
  SharedBuffer(SharedBuffer&&) noexcept;
  SharedBuffer& operator=(SharedBuffer&&) noexcept;
  SharedBuffer(const SharedBuffer&)            = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;
  ~SharedBuffer();

  // True for default-constructed and moved-from instances.
  bool         empty()      const noexcept { return _buf == nullptr; }

  // UMA pointer the CPU can read/write directly. Stable until the
  // buffer is destroyed. Null on an empty SharedBuffer.
  void*        contents()   const noexcept { return _contents; }

  // Byte capacity as requested at allocation. The underlying MTL
  // allocation is page-rounded under the hood but the user view is
  // the requested size.
  std::size_t  byte_size()  const noexcept { return _byte_size; }

  // Underlying MTL::Buffer*. Stable across the SharedBuffer's life
  // (refcount not touched on each call). For low-level encoder code
  // that needs to call setBuffer:offset:atIndex: directly.
  MTL::Buffer* mtl_buffer() const noexcept { return _buf; }

  // Baked-in GPU byte offset into mtl_buffer() (0 for a normal buffer;
  // nonzero only for subview() handles). set_buffer() adds it to the
  // per-call byte_offset so a subview reads the right slice transparently.
  std::size_t  byte_offset() const noexcept { return _base_off; }

  // A non-owning-data, refcount-sharing window [offset, offset+size) into
  // this buffer's memory. The returned handle retains the SAME MTL::Buffer
  // (refcount +1) and carries byte_offset()+offset, so binding it via
  // set_buffer() addresses the slice on the GPU and contents() points at it
  // on the CPU. Lets one big allocation back several logical buffers without
  // copying (e.g. q|k|v sub-weights of a fused concat). Not wired.

  // Optional shape/strides/dtype metadata. Default-constructed view
  // (rank == 0) means "untyped raw buffer". set_view does NOT touch
  // bytes.
  const BufferView& view() const noexcept       { return _view; }
  void              set_view(BufferView v) noexcept { _view = v; }

  // Wired memory toggle. Idempotent.
  //   set_wired(true)  -> mlock(page-rounded contents range) +
  //                       setPurgeableState:NonVolatile.
  //   set_wired(false) -> munlock + setPurgeableState:Volatile.
  // Returns false (with errno preserved) if mlock() failed; the
  // buffer is still usable, just not wired. The most common failure
  // is RLIMIT_MEMLOCK exhaustion -- macOS defaults to 64KB for
  // non-root processes, so wiring larger buffers requires the
  // process to have raised the limit first (e.g. via
  // mlx::core::set_wired_limit's path).
  bool set_wired(bool on) noexcept;
  bool is_wired() const noexcept { return _wired; }

  // ---- is this buffer actually still in RAM? -----------------------
  //
  // A weight buffer kept resident so it need not be re-read is only
  // worth its RAM while it IS in RAM. Once the OS has compressed or
  // swapped its pages the buffer costs everything a streamed block
  // costs -- a fault, a decompress or a disk read -- plus the memory it
  // is still nominally occupying. That is the failure this reports, and
  // it reports it about THIS buffer rather than about the machine, so a
  // policy does not have to infer it from free-memory arithmetic that
  // cannot tell a hot cache from a cold one.
  //
  // `stride_pages` samples: 1 reads every page, N reads every Nth. A
  // pin either holds or it does not, so a sample finds it -- and the
  // vector mincore() needs is one byte per page examined, which at
  // stride 1 over a 1.2 GB block is 75 KB.
  struct PageResidency {
    std::size_t examined  = 0;   // pages the query looked at
    std::size_t incore    = 0;   // of those, still in RAM
    std::size_t paged_out = 0;   // of those, known to have been evicted
    bool        valid     = false;
    // Nothing of this buffer has left RAM.
    bool fully_resident() const noexcept
    {
      return !valid || (examined > 0 && incore == examined);
    }
    double resident_fraction() const noexcept
    {
      return examined > 0 ? (double)incore / (double)examined : 1.0;
    }
  };
  PageResidency page_residency(std::size_t stride_pages = 1) const noexcept;

  // ---- residency: parking an INACTIVE weight buffer ----------------
  //
  // The pair behind "release under memory pressure, reactivate without
  // reloading". mark_inactive() hands the pages to the kernel as
  // reclaimable while keeping the allocation and its GPU address; the
  // contents survive as long as nothing else needs the RAM.
  // reactivate() takes them back and REPORTS whether they survived --
  // that report is the whole point, since a purge is silent otherwise
  // and the buffer would read as garbage.
  //
  // Only meaningful for a buffer this handle OWNS and that was
  // allocated standalone. A subview shares another handle's allocation
  // (parking it would evict memory it does not own) and a heap
  // sub-allocation's residency belongs to the heap, so both refuse and
  // stay fully resident. An explicitly wired buffer also refuses --
  // wiring is a deliberate "never evict this".
  //
  //   mark_inactive() -> true  the buffer is now evictable.
  //                   -> false it stays resident (see above).
  //   reactivate()    -> true  contents INTACT, use as-is (the fast
  //                            path: no I/O, just a state flip).
  //                   -> false contents were DISCARDED; the caller must
  //                            reload the tensor before using it.
  //
  // reactivate() on a buffer that was never parked returns true: there
  // is nothing to have lost.
  bool mark_inactive() noexcept;
  bool reactivate()    noexcept;
  bool is_inactive() const noexcept { return _inactive; }

  // True when THIS handle is the one that put these bytes on the books
  // -- i.e. it owns a standalone allocation rather than aliasing one
  // (subview) or wrapping foreign memory. It is exactly the set of
  // handles mark_inactive() will act on, so callers deciding whether a
  // tensor is parkable can ask up front instead of inferring it from a
  // state-changing call.
  bool is_owned() const noexcept { return _accounted; }

  // Wrap an externally-allocated MTL::Buffer (e.g. one carried by a
  // TensorBeat's ExternalStorageHandle) without a copy. Caller
  // transfers ONE refcount on `buf` to the returned SharedBuffer.
  // `contents` may be null, in which case the SharedBuffer queries
  // `buf->contents()` itself. Mainly used by the tensor-beat
  // bridge; ordinary code goes through MetalCompute::make_shared_buffer.
  static SharedBuffer wrap(MTL::Buffer* buf, void* contents,
                           std::size_t byte_size) noexcept;

  // See the comment block above byte_offset().
  SharedBuffer subview(std::size_t offset, std::size_t size) const noexcept;

private:
  friend class MetalCompute;

  // MetalCompute-only ctor: takes ownership of a freshly-allocated
  // MTL::Buffer at refcount +1.
  SharedBuffer(MTL::Buffer* buf, void* contents,
               std::size_t byte_size) noexcept;

  // Tear down (release buffer + unwire pages if still wired). Used
  // by both the destructor and move-assignment.
  void teardown_() noexcept;

  MTL::Buffer* _buf       = nullptr;
  void*        _contents  = nullptr;
  std::size_t  _byte_size = 0;
  std::size_t  _base_off  = 0;     // GPU byte offset for subview() handles
  BufferView   _view{};
  bool         _wired     = false;
  // True only for the handle make_shared_buffer/wrap_no_copy minted, i.e. the
  // one that put these bytes on the books. subview() handles share the same
  // MTL::Buffer by refcount and must NOT be counted again, or a model that
  // subviews heavily would report several times its real footprint.
  bool         _accounted = false;
  bool         _inactive  = false;   // parked via mark_inactive
};

// ---- Process-wide SharedBuffer accounting -------------------------------
// Device allocations are the bulk of a model's footprint but are invisible to
// max-RSS style tools (and phys_footprint only gives one aggregate number),
// so there was no way to ask "which buffer grew?" when a footprint moved.
// These counters answer that: cumulative and concurrent bytes, plus an
// env-gated per-allocation log for attribution.
//
// `live_bytes` counts buffers whose owning handle is still alive; it drops
// when that handle is destroyed even if a subview outlives it (the memory is
// still held in that case, so treat live_bytes as a close lower bound rather
// than an exact residency figure).
struct MemoryStats {
  std::size_t live_bytes  = 0;   // currently held by live owning handles
  std::size_t peak_bytes  = 0;   // high-water mark of live_bytes
  std::size_t total_bytes = 0;   // cumulative, never decremented
  std::size_t live_count  = 0;
  std::size_t total_count = 0;
};

// Snapshot the counters. Cheap (relaxed atomic loads); safe from any thread.
MemoryStats shared_buffer_memory_stats() noexcept;

// Internal: called by MetalCompute when it mints an owning handle, and by
// SharedBuffer::teardown_ when that handle dies. Not for general use.
void account_alloc_(std::size_t bytes) noexcept;
void account_free_(std::size_t bytes) noexcept;

// Re-arm peak_bytes to the current live_bytes -- call before a phase you want
// a peak for (e.g. after load, to get the prefill-only high-water mark).
void shared_buffer_reset_peak() noexcept;

}  // namespace vpipe::metal_compute

#endif
