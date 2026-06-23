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
};

}  // namespace vpipe::metal_compute

#endif
