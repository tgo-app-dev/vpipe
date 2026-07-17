#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_COMPUTE_ENCODER_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_COMPUTE_ENCODER_H

#include <cstddef>
#include <type_traits>

namespace MTL {
class Buffer;
class CommandBuffer;
class ComputeCommandEncoder;
}

namespace vpipe::metal_compute {

class CommandStream;
class ComputeFunction;
class Fence;
class SharedBuffer;
class Texture;

// CUDA-grid-shape launch dimensions. x/y/z are total thread counts
// (or total threadgroup counts) in each dimension; one-dimensional
// launches just leave y and z at 1. Matches MTLSize semantics.
struct LaunchDims {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;
};

// How the encoder orders successive dispatches:
//   Serial      -- Metal may insert implicit memory barriers
//                  between dispatches based on resource hazards.
//                  Safe default; matches the framework's
//                  Tracked-hazard buffers.
//   Concurrent  -- no implicit barriers; subsequent dispatches
//                  may run in parallel with prior ones and see
//                  stale writes. Pair with Untracked buffers
//                  and explicit memory_barrier() calls. Required
//                  to get the perf win of explicit hazard
//                  management.
enum class DispatchType : std::uint8_t {
  Serial     = 0,
  Concurrent = 1,
};

// Which resource class an explicit memory_barrier() applies to.
// Bitwise OR-able. BarrierScope::Both is the common pair for
// kernels that mix buffer + texture writes.
enum class BarrierScope : std::uint8_t {
  Buffers  = 1,
  Textures = 2,
  Both     = 3,
};

// RAII handle around a MTL::ComputeCommandEncoder. Returned by
// CommandStream::begin_compute(). The destructor calls endEncoding()
// if end() wasn't called; the encoder must be ended before its
// CommandStream commits, otherwise Metal will fault on commit.
//
// Move-only. Default-constructed instances are empty no-ops.
class ComputeEncoder {
public:
  ComputeEncoder() noexcept = default;
  ComputeEncoder(ComputeEncoder&&) noexcept;
  ComputeEncoder& operator=(ComputeEncoder&&) noexcept;
  ComputeEncoder(const ComputeEncoder&)            = delete;
  ComputeEncoder& operator=(const ComputeEncoder&) = delete;
  ~ComputeEncoder();

  bool valid() const noexcept { return _enc != nullptr; }

  // Bind the function whose PSO defines the next dispatch. Must be
  // called before any dispatch().
  void set_function(const ComputeFunction& fn);

  // Bind `buf` at MSL `buffer(index)`. `byte_offset` advances the
  // GPU-visible base. No-op if `buf` is empty.
  void set_buffer(unsigned index, const SharedBuffer& buf,
                  std::size_t byte_offset = 0);

  // Bind a raw MTL::Buffer*. Mainly for legacy code holding a
  // buffer through an external handle (e.g. TensorBeat's
  // ExternalStorageHandle); ordinary code goes through
  // set_buffer(SharedBuffer&). No-op on null.
  void set_mtl_buffer(unsigned index, MTL::Buffer* buf,
                      std::size_t byte_offset = 0);

  // Bind `buf` at MSL `buffer(buf_index)` AND pack its BufferView
  // (rank, dtype, shape[8], strides[8], offset) as a constant
  // struct at MSL `buffer(meta_index)`. Kernels declare the
  // metadata slot as
  //   `constant vpipe_tensor_view& view [[buffer(meta_index)]]`
  // (see msl/vpipe_tensor_view.metal) and use vpipe_tv_index() to
  // compute element offsets. The data buffer is bound at byte 0;
  // the view's offset is applied inside the kernel via the
  // metadata, not at bind time.
  void set_buffer_view(unsigned buf_index,
                       const SharedBuffer& buf,
                       unsigned meta_index);

  // Bind raw constant bytes via MTL setBytes:length:atIndex:. Use
  // the typed `set_constant<T>` wrapper for plain values; this
  // form is for variable-length constants the caller assembles.
  void set_constant_bytes(unsigned index,
                          const void* bytes,
                          std::size_t size);

  // Typed scalar constant. T must be trivially copyable.
  template <class T>
  void set_constant(unsigned index, const T& value)
  {
    static_assert(std::is_trivially_copyable_v<T>,
                  "set_constant requires a trivially-copyable type");
    set_constant_bytes(index, &value, sizeof(T));
  }

  // Declare a dynamic threadgroup-memory buffer at MSL slot
  // `index`. Counted in bytes; combine with `threadgroup` storage-
  // class arguments in the kernel signature.
  void set_threadgroup_memory_length(unsigned index,
                                     std::size_t byte_size);

  // Bind `tex` at MSL `texture(index)`. No-op if `tex` is empty.
  void set_texture(unsigned index, const Texture& tex);

  // Signal `fence` after this encoder's commands retire on the GPU.
  // A subsequent encoder in the same command buffer can call
  // wait_for_fence(fence) to park its commands until the signal
  // arrives. Required for Untracked-hazard resources where Metal's
  // implicit hazard tracking is disabled.
  void update_fence(const Fence& fence);

  // Park this encoder's commands until `fence` is signalled by a
  // prior encoder in the same command buffer.
  void wait_for_fence(const Fence& fence);

  // Insert an explicit memory barrier across the encoder's
  // dispatches. Required between dependent dispatches when the
  // encoder was opened with DispatchType::Concurrent (or when the
  // bound buffers are Untracked-hazard). No-op on a Serial encoder
  // with Tracked buffers -- Metal already barriers there.
  void memory_barrier(BarrierScope scope = BarrierScope::Buffers);

  // Resource-scoped barrier: order prior writes to `buf` before subsequent
  // dispatches' accesses to `buf`, leaving every OTHER buffer unordered. On a
  // Concurrent encoder this lets the next kernel's independent DRAM traffic
  // (e.g. its weights) issue during this kernel's drain, while still making a
  // dependent activation write visible. Finer + cheaper than the scope-wide
  // memory_barrier(Buffers), which drains ALL buffers. No-op on empty buf.
  void memory_barrier_buffer(const SharedBuffer& buf);

  // RAII: for the duration of the scope, swap this encoder's underlying MTL
  // encoder from Serial to a CONCURRENT one (in the SAME command buffer), so
  // the dispatches inside run concurrently -- for a group of provably
  // INDEPENDENT dispatches (unfused q|k|v, gate|up, GDN in_proj qkv|z|a|b). On
  // scope exit it reopens a Serial encoder so the dependent chain keeps Metal's
  // cheap native ordering. The encoder boundaries carry cross-group ordering
  // via Metal's automatic hazard tracking (Tracked buffers). No explicit
  // memoryBarrier -> avoids the full-flush barrier tax. `active == false` is a
  // no-op (stays Serial). Requires the encoder to know its command buffer
  // (set by CommandStream::begin_compute). Move-only.
  class ConcurrentScope {
   public:
    ConcurrentScope(ConcurrentScope&&) noexcept;
    ConcurrentScope& operator=(ConcurrentScope&&) = delete;
    ConcurrentScope(const ConcurrentScope&)       = delete;
    ~ConcurrentScope();

   private:
    friend class ComputeEncoder;
    ConcurrentScope(ComputeEncoder* e, bool active) noexcept;
    ComputeEncoder* _e = nullptr;   // null when inactive -> dtor no-op
  };
  ConcurrentScope concurrent_scope(bool active = true)
  {
    return ConcurrentScope(this, active);
  }

  // Dispatch using Metal's automatic threadgroup-grid calculation.
  // `threads_per_grid` is the *total* number of threads (CUDA-
  // shape); the driver partitions into threadgroups of size
  // `threads_per_threadgroup`. The threadgroup size should respect
  // ComputeFunction::max_total_threads_per_threadgroup().
  void dispatch(LaunchDims threads_per_grid,
                LaunchDims threads_per_threadgroup);

  // End encoding early. The destructor calls this if the user
  // didn't; calling end() twice is harmless.
  void end();

  MTL::ComputeCommandEncoder* mtl_encoder() const noexcept
  {
    return _enc;
  }

  // Number of dispatch() calls so far on this encoder (profiling: decode is a
  // chain of short dependent dispatches, and the per-launch GPU idle between
  // them is the gap vs a fused graph).
  long dispatch_count() const noexcept { return _n_dispatch; }

private:
  friend class CommandStream;
  friend class ConcurrentScope;
  ComputeEncoder(MTL::ComputeCommandEncoder* enc,
                 MTL::CommandBuffer* cb) noexcept;
  // End the current MTL encoder and open a fresh one on _cb with the given
  // dispatch type; used by ConcurrentScope to flip Serial<->Concurrent in
  // place. No-op if _cb is unknown.
  void reencode_(DispatchType dt);

  MTL::ComputeCommandEncoder* _enc = nullptr;
  MTL::CommandBuffer* _cb = nullptr;   // owning command buffer (for reencode_)
  long _n_dispatch = 0;
  // Auto command-buffer split (lifted from the per-model decode split): when
  // _split_every>0, dispatch() commits _cb + reopens a fresh one every
  // _split_every dispatches (outside any concurrent_scope), so a long stream's
  // CPU-encode pipelines against GPU-exec. Set by CommandStream::begin_compute;
  // the rollover is done by _stream so the stream's _cb stays coherent. Tracked
  // buffers auto-fence deps across the split, so it is correctness-safe.
  CommandStream* _stream = nullptr;
  int _split_every = 0;
  int _since_split = 0;
  int _scope_depth = 0;   // >0 inside a concurrent_scope -> suppress split
  DispatchType _dt = DispatchType::Serial;
};

}  // namespace vpipe::metal_compute

#endif
