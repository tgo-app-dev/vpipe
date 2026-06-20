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
