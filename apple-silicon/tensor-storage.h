#ifndef VPIPE_APPLE_SILICON_TENSOR_STORAGE_H
#define VPIPE_APPLE_SILICON_TENSOR_STORAGE_H

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vpipe {

// Where a TensorBeat's backing bytes physically live, and what kinds
// of zero-copy access are guaranteed.
//
//   CpuCached: bytes live in TensorBeat::data (an AlignedVector<uint8_t>,
//              64-byte aligned via posix_memalign). On Apple-Silicon
//              UMA they can still be wrapped by MLMultiArray
//              initWithDataPointer for CoreML, but a Metal compute
//              kernel needs a newBufferWithBytes copy and ANE may
//              stage internally. Default and only option on
//              non-Apple builds.
//   Shared:    Apple-only. Bytes live inside an MTL::Buffer with
//              MTLResourceStorageModeShared. The MTL::Buffer handle
//              (accessible via TensorBeat::mtl_buffer()) can be
//              bound directly to a compute kernel -- no staging
//              copy. CoreML / ANE can also ingest the same buffer
//              contents zero-copy.
//
// Stages allocate Shared storage when they know the data will flow
// into a Metal compute kernel or a CoreML model next. The flag is
// observational: a downstream consumer can branch on storage_class()
// to pick a zero-copy fast path (Metal kernel bind, MLMultiArray
// initWithDataPointer on the contents pointer) or fall back to the
// generic CPU path.
enum class TensorStorageClass : std::uint8_t {
  CpuCached = 0,
  Shared    = 1,
};

// Opaque holder for a Shared MTL::Buffer-backed byte region.
//
//   * `mtl_buffer` is the underlying MTL::Buffer* (opaque -- callers
//     `static_cast<MTL::Buffer*>(p)` to bind to a compute kernel).
//   * `contents` is `mtl_buffer->contents()` cached so hot reads
//     don't go through metal-cpp's selector dispatch. CPU and GPU
//     see the same address.
//   * `byte_size` is the buffer's length in bytes.
//   * `deleter` is invoked from the handle's destructor; typically
//     releases the MTL::Buffer at +1.
//
// Held by a TensorBeat through `std::unique_ptr<ExternalStorageHandle>`:
// each TensorBeat owns its bytes exclusively, and copying a Shared
// TensorBeat materialises the bytes into a CpuCached `data` (the
// copy is independent of the source).
struct ExternalStorageHandle {
  using Deleter = void (*)(void* mtl_buffer);

  void*         mtl_buffer = nullptr;
  std::uint8_t* contents   = nullptr;
  std::size_t   byte_size  = 0;
  Deleter       deleter    = nullptr;

  ~ExternalStorageHandle()
  {
    if (mtl_buffer && deleter) { deleter(mtl_buffer); }
  }

  ExternalStorageHandle() = default;
  ExternalStorageHandle(const ExternalStorageHandle&)            = delete;
  ExternalStorageHandle& operator=(const ExternalStorageHandle&) = delete;
};

}

#endif
