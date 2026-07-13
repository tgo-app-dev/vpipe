#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_METAL_COMPUTE_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_METAL_COMPUTE_H

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/fence.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/texture.h"
#include "common/session-member.h"

#include <cstddef>
#include <memory>
#include <string_view>

namespace MTL {
class ComputePipelineState;
class Device;
class Function;
}
namespace NS { class Error; }

namespace vpipe::metal_compute {

// Session-shared Metal compute kernel framework. Provides a CUDA-shape
// surface: load_library / get function / make shared buffer / encode
// dispatch on a command stream / wait. Sits alongside (not in place of)
// the existing MetalRuntime: MetalRuntime owns the five legacy inline
// kernels; MetalCompute is the new home for kernels declared via
// add_vpipe_metal_kernel() in CMake.
//
// Construction acquires the system default MTL::Device. Failure to do
// so is reported via valid() == false; callers must check before
// using any other accessor. All other resource-creation methods are
// added in subsequent steps; step 1 only exposes the bare facade so
// Session can hand out a non-null pointer.
class MetalCompute final : public SessionMember {
public:
  explicit MetalCompute(const SessionContextIntf* session);
  ~MetalCompute() override;

  MetalCompute(const MetalCompute&)            = delete;
  MetalCompute& operator=(const MetalCompute&) = delete;

  bool valid() const noexcept;

  // Direct access to the underlying MTL::Device. Stable across the
  // life of the MetalCompute instance. Returns nullptr when
  // valid() == false.
  MTL::Device* device() const noexcept;

  // True when the GPU has hardware matrix units (Apple M5 / GPUFamily
  // Apple10 and newer). The Metal 4 MetalPerformancePrimitives matmul2d
  // path runs on those units and beats the steel simdgroup_matrix GEMM
  // ~2.5-3x; on older GPUs (no matrix cores) it would emulate, so callers
  // gate the matrix-core kernels on this and keep the steel path
  // otherwise. Cheap (cached MTL::Device family query). false when
  // valid() == false.
  bool supports_matrix_cores() const noexcept;

  // Cumulative allocator counters. Useful for tests / telemetry to
  // verify the heap-vs-direct path is being chosen as expected.
  // Monotonically increasing across the life of the MetalCompute.
  struct AllocStats {
    std::size_t buffers_from_heap   = 0;
    std::size_t buffers_from_device = 0;
  };
  AllocStats alloc_stats() const noexcept;

  // Bind an on-disk MTL::BinaryArchive to the PSO build path.
  // Subsequent compute pipeline builds register their compiled
  // binaries with the archive; if `path` already exists, the
  // archive's prior contents are loaded and Metal can resolve
  // matching PSOs without re-compiling.
  //
  // Pass an empty path to disable. Returns true if the archive was
  // created (regardless of whether `path` already existed). The
  // archive is not auto-saved -- call save_binary_archive() to
  // flush.
  bool set_binary_archive_path(std::string_view path) const;

  // Serialize the in-memory archive to the path configured by
  // set_binary_archive_path(). Returns true on success, false on
  // disk I/O failure or if no path is configured.
  bool save_binary_archive() const;

  // Counters for the PSO archive path. `added_to_archive` is the
  // number of PSOs whose binaries were registered with the
  // archive (a non-fatal failure to register still increments
  // `compiled_with_archive_set`). Monotonic.
  struct PsoArchiveStats {
    std::size_t compiled_with_archive_set = 0;
    std::size_t added_to_archive          = 0;
  };
  PsoArchiveStats pso_archive_stats() const noexcept;

  // Working-set residency tracking (Metal 3+). Add buffers /
  // textures to the residency set so they stay in VRAM under
  // memory pressure; commit() to apply pending edits;
  // request_residency()/end_residency() to bring the tracked set
  // into / out of resident memory.
  //
  // Returns false if the device doesn't support MTL::ResidencySet
  // (older OS / pre-Metal3 GPU) or on Metal-side allocation
  // failure.
  bool residency_set_supported() const noexcept;
  bool residency_add(const SharedBuffer&) const;
  bool residency_add(const Texture&) const;
  bool residency_remove(const SharedBuffer&) const;
  bool residency_remove(const Texture&) const;
  bool residency_commit() const;
  bool residency_request() const;
  bool residency_end() const;

  // Telemetry. `current` is the MTL-side allocation count (may
  // lag local add/remove until commit() runs).
  struct ResidencyStats {
    std::size_t add_calls    = 0;
    std::size_t remove_calls = 0;
    std::size_t current      = 0;
  };
  ResidencyStats residency_stats() const noexcept;

  // GPU memory budget snapshot, for over-commit detection. All bytes.
  //   recommended -- device()->recommendedMaxWorkingSetSize(): the amount
  //                  of memory the GPU can keep resident without paging /
  //                  perf penalties (on UMA Apple Silicon this tracks the
  //                  usable slice of system RAM, and follows an
  //                  iogpu.wired_limit_mb override).
  //   allocated   -- device()->currentAllocatedSize(): total resource
  //                  memory this device currently has allocated.
  //   headroom    -- recommended - allocated, clamped at 0: how much a new
  //                  allocation can safely take before over-committing.
  // All zero when valid() == false. Cheap (two device property reads);
  // note `allocated` is a live figure that moves as buffers come and go.
  struct MemoryBudget {
    std::size_t recommended = 0;
    std::size_t allocated   = 0;
    std::size_t headroom    = 0;
    // Reclaimable physical memory available to the process right now: free +
    // purgeable + speculative + file-backed(external) pages. Unlike `headroom`
    // (a Metal working-set figure), this reflects TRUE physical pressure and
    // COUNTS clean/mmap'd weight pages as available (the OS can evict them), so
    // it catches an over-commit the working-set figure misses -- e.g. a big VAE
    // decode running alongside mmap'd DiT weights. 0 when the query is
    // unavailable; callers should skip the check then.
    std::size_t available_physical = 0;
    // True when `need` bytes fit in the current headroom (with an optional
    // safety margin, default 5%). A quick preflight before a big alloc. Does
    // NOT consult available_physical -- check that separately where relevant so
    // this stays a pure GPU-working-set test for existing callers.
    bool fits(std::size_t need, double margin = 0.05) const noexcept
    {
      const double avail = (double)headroom * (1.0 - margin);
      return (double)need <= avail;
    }
    // True when `need` fits the reclaimable physical budget (margin default
    // 10%). Vacuously true when available_physical == 0 (query unavailable).
    bool fits_physical(std::size_t need, double margin = 0.10) const noexcept
    {
      if (available_physical == 0) { return true; }
      return (double)need <= (double)available_physical * (1.0 - margin);
    }
  };
  MemoryBudget memory_budget() const noexcept;

  // Allocate a `byte_size`-byte MTL::Buffer in Shared storage mode
  // (CPU and GPU share the same UMA address). Returns an empty
  // SharedBuffer if the runtime is not valid() or if Metal failed
  // the allocation. `alignment` is the minimum byte alignment the
  // caller needs from contents(); MTL Shared buffers are page-
  // aligned, so any alignment <= page size is satisfied trivially.
  // The parameter is retained for forward compatibility with a
  // future sub-allocating pool.
  SharedBuffer make_shared_buffer(
      std::size_t byte_size,
      std::size_t alignment = 64,
      HazardTracking hazard_tracking = HazardTracking::Tracked) const;

  // Wrap already-mapped host memory as a Shared MTL::Buffer WITHOUT copying
  // (newBufferWithBytesNoCopy). `ptr` must be page-aligned (mmap / vm_allocate
  // memory is); the mapping length is page-rounded for Metal but the returned
  // SharedBuffer reports `byte_size`. Metal does NOT own the memory (null
  // deallocator) -- the CALLER must keep it mapped for the lifetime of the
  // returned buffer and every subview() of it, and must not let the GPU touch
  // it after unmapping. Intended for read-only, file-backed model weights:
  // clean file pages the OS can reclaim under memory pressure and re-fault
  // from disk, instead of dirty anonymous copies that must be swapped.
  // Returns empty if the runtime is invalid, `ptr` is null / not page-aligned,
  // or Metal rejects the wrap.
  SharedBuffer make_no_copy_buffer(void* ptr, std::size_t byte_size) const;

  // Load a `.metallib` whose bytes were embedded into libvpipe by an
  // add_vpipe_metal_kernel(KERNEL_NAME ...) call in CMake. `name`
  // must match the KERNEL_NAME at registration. Returns an invalid
  // ComputeLibrary if the runtime is not valid(), the name is not
  // in the embedded registry, or Metal failed to parse the bytes.
  //
  // Cached: repeated load_library() calls with the same name share
  // the same MTL::Library* under the hood (each returned
  // ComputeLibrary independently retains).
  ComputeLibrary load_library(std::string_view name) const;

  // Register a `.metallib`'s bytes under `name` at RUNTIME (e.g. from a
  // plugin whose kernels were compiled offline). The bytes are COPIED into
  // a process-owned store, so the caller's buffer need not outlive the
  // call; `name` then resolves through load_library(name) exactly like a
  // build-embedded metallib. First-wins: returns false (a warning is
  // logged) if `name` is already registered -- built-in or runtime -- so a
  // plugin can't shadow a built-in kernel, or if the bytes are empty.
  bool register_metal_library(std::string_view     name,
                              const unsigned char* bytes,
                              std::size_t          n) const;

  // Convenience: read `path` off disk and register its bytes under `name`
  // (see register_metal_library). Returns false + warns on a read error.
  bool register_metal_library_file(std::string_view name,
                                   std::string_view path) const;

  // Resolve a kernel entry point on `lib` to a (MTL::Function,
  // MTL::ComputePipelineState) pair. The PSO is cached on
  // (lib_name, fn_name); repeated calls return ComputeFunctions
  // sharing the same PSO. Mainly the back-end of
  // ComputeLibrary::function(); exposed publicly because
  // ComputeLibrary delegates here.
  ComputeFunction function(const ComputeLibrary& lib,
                           std::string_view fn_name) const;

  // Same shape as the unspecialized variant, but additionally
  // pins the MSL `[[function_constant(N)]]` slots in `constants`
  // at JIT-compile time. The PSO cache key extends to include
  // the constants signature, so unique constant tuples get unique
  // PSOs while identical tuples share.
  ComputeFunction function(const ComputeLibrary& lib,
                           std::string_view fn_name,
                           const FunctionConstants& constants) const;

  // Allocate a fresh, independently-ordered command stream
  // (== MTL::CommandQueue). Streams synchronize via shared Events.
  // Returns an empty CommandStream on a non-valid runtime or
  // allocation failure.
  CommandStream make_command_stream() const;

  // Allocate a fresh shared event for cross-stream / CPU<->GPU
  // synchronization. Initial counter is 0.
  Event make_event() const;

  // Allocate a fresh MTL::Fence for intra-command-buffer encoder
  // ordering. Use with `Untracked`-hazard buffers to enforce
  // dependencies between encoders that the framework knows about.
  Fence make_fence() const;

  // Allocate a fresh 2D MTL::Texture matching `desc`. Returns an
  // invalid Texture on allocation failure or if the format is
  // PixelFormat::Unknown.
  Texture make_texture(const TextureDesc& desc) const;

  // Bridge a CVPixelBuffer into a Metal texture, zero-copy when
  // possible (the texture aliases the same IOSurface; subsequent
  // CVPixelBuffer mutations are visible to the kernel). The
  // CVPixelBuffer must be in a format the bridge recognizes
  // (BGRA, single-plane). For NV12 use
  // nv12_textures_from_cv_pixel_buffer instead. Returns an
  // invalid Texture on format mismatch or CVMetalTextureCache
  // failure. The opaque pointer is `CVPixelBufferRef`, cast to
  // void* so this header doesn't drag in CoreVideo.
  Texture texture_from_cv_pixel_buffer(void* cv_pixel_buffer,
                                       PixelFormat format) const;

  // Bridge a bi-planar YUV CVPixelBuffer (NV12: 420YpCbCr8Biplanar
  // Video / Full range) into a pair of Metal textures (Y plane
  // R8Unorm, CbCr plane RG8Unorm). Zero-copy; both textures
  // alias the same IOSurface as the CVPixelBuffer. Returns
  // empty textures on format mismatch (pix-fmt not biplanar,
  // CVPixelBuffer doesn't have 2 planes) or cache failure.
  YuvBiplanarTextures
  nv12_textures_from_cv_pixel_buffer(void* cv_pixel_buffer) const;

private:
  // PSO build helper that routes through the binary archive when
  // one is configured. Returns nullptr on compile failure (err is
  // populated by Metal).
  MTL::ComputePipelineState*
  build_pso_(MTL::Function* fn, NS::Error** out_err) const;

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace vpipe::metal_compute

#endif
