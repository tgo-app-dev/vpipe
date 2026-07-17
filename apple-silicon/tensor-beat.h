#ifndef VPIPE_APPLE_SILICON_TENSOR_BEAT_H
#define VPIPE_APPLE_SILICON_TENSOR_BEAT_H

#include "apple-silicon/aligned-allocator.h"
#include "apple-silicon/tensor-storage.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// N-D tensor exchanged between stages. Row-major, PyTorch-style
// strided storage.
//
//   * `data` is the 1-D byte storage, allocated via AlignedAllocator
//     so the buffer is 64-byte aligned regardless of dtype. Typed
//     views (as_f32 / as_u8 / ...) inherit that alignment.
//   * `dtype` names the element type that the bytes encode. The
//     element width follows from dtype; bf16 is 2 bytes (stored as
//     uint16_t in raw form), i8/u8 are 1 byte, f32 is 4.
//   * `shape` is the logical shape; `element_count() == product(shape)`.
//   * `strides`, when non-empty, are *element* strides (NOT bytes) —
//     one per shape dim. Innermost is always 1 in v1; producers use
//     this for pitch padding (outer-dim row pitch wider than the
//     logical inner-dim length). Broadcast (stride 0), transpose
//     (permuted strides), and arbitrary slicing (non-zero offset)
//     are reserved.
//   * `storage_offset` is in elements (NOT bytes), measured from
//     the start of `data`.
//
// Migration note: prior versions stored f32 directly in
// `std::vector<float> data`. Callers that used `tb.data.data()`
// (giving a `float*`) should now use `tb.as_f32()`. Bulk byte access
// is still `tb.data.data()`, but the type is `uint8_t*`.
struct TensorBeat {
  enum class DType : uint8_t {
    U8   = 0,
    I8   = 1,
    Bf16 = 2,
    F32  = 3,
    F16  = 4,   // IEEE half. Appended (ordinal 4) so the ordinals stay
                // locked to metal_compute::DType (tensor-beat-bridge.cc).
  };

  DType                  dtype = DType::F32;
  std::vector<int64_t>   shape;
  std::vector<int64_t>   strides;          // empty = row-major contiguous
  int64_t                storage_offset = 0;
  AlignedVector<uint8_t> data;

  // Optional Shared (Metal MTL::Buffer-backed) storage. When set,
  // `data` is empty and the logical bytes live inside the buffer
  // pointed to by `external->contents`. Allocated by
  // MetalRuntime::make_shared_storage(); a downstream Metal kernel
  // can bind external->mtl_buffer directly without a host roundtrip
  // and CoreML/ANE can ingest external->contents zero-copy via
  // MLMultiArray initWithDataPointer. Use `storage_class()` to
  // branch in consumer code. Default is empty (CpuCached). Held
  // through a unique_ptr because each TensorBeat owns its bytes
  // exclusively -- a copy of a Shared TensorBeat materialises
  // its content into `data` (CpuCached) so the two copies are
  // independent.
  std::unique_ptr<ExternalStorageHandle> external;

  // Optional per-beat sideband metadata, kept alongside the tensor
  // bytes so producers can attach small structured payloads (timing,
  // source identifiers, sequence numbers, ...) without inventing a
  // parallel oport. Default is FlexData::Null. Producers use it via
  // `tb.sideband = FlexData::make_object()`-style assignments;
  // consumers introspect with the standard FlexData API. Deep-copied
  // by the copy constructor (FlexData has value semantics) so two
  // clones from a fanout edge see independent objects.
  //
  // Conventions for well-known keys (object form):
  //   * timestamp_us : uint64  -- microseconds-since-UTC-epoch the
  //                               frame represents (video-to-rgb
  //                               sets this from its source AU's
  //                               capture time).
  //   * camera_name  : string  -- source camera id (video-to-rgb).
  //   * fps_num,
  //     fps_den      : uint64  -- source frame rate as a rational
  //                               (fps = fps_num / fps_den), forwarded
  //                               by video-to-rgb from the capture
  //                               stream. Present only when known;
  //                               a sink (hls-broadcast) adopts it as
  //                               the default encode cadence.
  FlexData sideband;

  TensorBeat() = default;
  ~TensorBeat() = default;

  // Moves transfer the unique_ptr cheaply.
  TensorBeat(TensorBeat&&) noexcept            = default;
  TensorBeat& operator=(TensorBeat&&) noexcept = default;

  // Copies are deep: when the source has Shared storage we cannot
  // duplicate the underlying MTL::Buffer from inside TensorBeat
  // (we have no MetalRuntime here), so we materialise its bytes
  // into a CpuCached `data` instead. The copy is therefore
  // independent of the source -- mutating one does not affect
  // the other -- but its storage_class() degrades from Shared to
  // CpuCached. Callers that need to keep the result on a Shared
  // buffer should re-allocate via MetalRuntime::make_shared_storage()
  // and re-emit through a kernel rather than copy.
  TensorBeat(const TensorBeat& o)
    : dtype(o.dtype)
    , shape(o.shape)
    , strides(o.strides)
    , storage_offset(o.storage_offset)
    , sideband(o.sideband)
  {
    if (o.external) {
      const std::size_t n = o.external->byte_size;
      data.assign(n, static_cast<uint8_t>(0));
      if (n > 0) {
        std::memcpy(data.data(), o.external->contents, n);
      }
    } else {
      data = o.data;
    }
  }

  TensorBeat&
  operator=(const TensorBeat& o)
  {
    if (this != &o) {
      TensorBeat tmp(o);
      *this = std::move(tmp);
    }
    return *this;
  }

  // Storage-class flag (CpuCached / Shared). See TensorStorageClass
  // for the contract.
  TensorStorageClass
  storage_class() const noexcept
  {
    return external ? TensorStorageClass::Shared
                    : TensorStorageClass::CpuCached;
  }

  // For Shared storage: returns the underlying MTL::Buffer pointer
  // (opaque). Returns nullptr for CpuCached; consumers branch on
  // this to pick a Metal-bind fast path versus the CPU fallback.
  void*
  mtl_buffer() const noexcept
  {
    return external ? external->mtl_buffer : nullptr;
  }

  // Internal: return the underlying byte pointer regardless of
  // storage class. Typed accessors (as_f32 / as_u8 / ...) and the
  // materializer go through this so the storage_class flag is
  // transparent to the rest of the codebase. On Apple-Silicon UMA
  // the Shared `contents` pointer is just as host-readable as a
  // CpuCached `data.data()`.
  std::uint8_t*
  bytes_() noexcept
  {
    return external ? external->contents : data.data();
  }
  const std::uint8_t*
  bytes_() const noexcept
  {
    return external ? external->contents : data.data();
  }

  // Byte size of the underlying storage regardless of class.
  // Consumers that compared `tb.data.size()` against an expected
  // byte count should switch to this -- with Shared storage the
  // legacy `data` AlignedVector is empty and the answer comes from
  // the external handle instead.
  std::size_t
  byte_size() const noexcept
  {
    return external ? external->byte_size : data.size();
  }

  // ------------------------------------------------------------------
  // Element-size + dtype helpers
  // ------------------------------------------------------------------

  static constexpr std::size_t
  byte_size_of(DType d) noexcept
  {
    switch (d) {
      case DType::U8:   return 1;
      case DType::I8:   return 1;
      case DType::Bf16: return 2;
      case DType::F16:  return 2;
      case DType::F32:  return 4;
    }
    return 0;
  }

  static constexpr const char*
  name_of(DType d) noexcept
  {
    switch (d) {
      case DType::U8:   return "u8";
      case DType::I8:   return "i8";
      case DType::Bf16: return "bf16";
      case DType::F16:  return "f16";
      case DType::F32:  return "f32";
    }
    return "?";
  }

  std::size_t
  element_byte_size() const noexcept
  {
    return byte_size_of(dtype);
  }

  const char*
  dtype_name() const noexcept
  {
    return name_of(dtype);
  }

  // ------------------------------------------------------------------
  // Shape helpers (dtype-independent)
  // ------------------------------------------------------------------

  // Logical element count: product(shape). NOT the storage footprint;
  // for strided tensors the data buffer can be larger.
  std::size_t
  element_count() const noexcept
  {
    std::size_t n = 1;
    for (auto d : shape) {
      n *= static_cast<std::size_t>(d);
    }
    return n;
  }

  // The strides a row-major contiguous tensor with `shape` would
  // have. Innermost is 1; outer dims are running products. Elements,
  // not bytes.
  std::vector<int64_t>
  contiguous_strides() const
  {
    std::vector<int64_t> s(shape.size());
    int64_t running = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
      s[i] = running;
      running *= shape[i];
    }
    return s;
  }

  // True iff the tensor is row-major contiguous in `data` starting
  // at element 0.
  bool
  is_contiguous() const noexcept
  {
    if (storage_offset != 0) {
      return false;
    }
    if (strides.empty()) {
      return true;
    }
    if (strides.size() != shape.size()) {
      return false;
    }
    int64_t running = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
      if (strides[i] != running) {
        return false;
      }
      running *= shape[i];
    }
    return true;
  }

  // ------------------------------------------------------------------
  // Storage allocation
  // ------------------------------------------------------------------

  // Resize `data` to hold `n_elements` of the current dtype, with no
  // strides (row-major contiguous). Bytes are zero-initialised. Use
  // this when building a fresh tensor whose dtype + shape is already
  // set.
  void
  resize_contiguous(std::size_t n_elements)
  {
    data.assign(n_elements * element_byte_size(),
                static_cast<uint8_t>(0));
    strides.clear();
    storage_offset = 0;
  }

  // ------------------------------------------------------------------
  // Typed accessors. Caller-provided dtype must match `this->dtype`
  // (asserted in debug). Pointer is at `data + storage_offset` in
  // element units.
  // ------------------------------------------------------------------

  const float*
  as_f32() const noexcept
  {
    assert(dtype == DType::F32);
    return reinterpret_cast<const float*>(bytes_())
         + storage_offset;
  }
  float*
  as_f32() noexcept
  {
    assert(dtype == DType::F32);
    return reinterpret_cast<float*>(bytes_()) + storage_offset;
  }

  const uint8_t*
  as_u8() const noexcept
  {
    assert(dtype == DType::U8);
    return bytes_() + storage_offset;
  }
  uint8_t*
  as_u8() noexcept
  {
    assert(dtype == DType::U8);
    return bytes_() + storage_offset;
  }

  const int8_t*
  as_i8() const noexcept
  {
    assert(dtype == DType::I8);
    return reinterpret_cast<const int8_t*>(bytes_())
         + storage_offset;
  }
  int8_t*
  as_i8() noexcept
  {
    assert(dtype == DType::I8);
    return reinterpret_cast<int8_t*>(bytes_()) + storage_offset;
  }

  // bf16 has no portable C++ type. Expose the bit pattern as
  // uint16_t; consumers reinterpret as needed (e.g. __bf16 on Clang).
  const uint16_t*
  as_bf16() const noexcept
  {
    assert(dtype == DType::Bf16);
    return reinterpret_cast<const uint16_t*>(bytes_())
         + storage_offset;
  }
  uint16_t*
  as_bf16() noexcept
  {
    assert(dtype == DType::Bf16);
    return reinterpret_cast<uint16_t*>(bytes_()) + storage_offset;
  }

  // IEEE half (f16). Like bf16, exposed as the raw uint16_t bit
  // pattern; consumers reinterpret as needed (e.g. _Float16 on Clang,
  // or CoreML's half output). Distinct from bf16 (different exponent
  // width) despite the shared 2-byte width.
  const uint16_t*
  as_f16() const noexcept
  {
    assert(dtype == DType::F16);
    return reinterpret_cast<const uint16_t*>(bytes_())
         + storage_offset;
  }
  uint16_t*
  as_f16() noexcept
  {
    assert(dtype == DType::F16);
    return reinterpret_cast<uint16_t*>(bytes_()) + storage_offset;
  }

  // ------------------------------------------------------------------
  // Materialisation: emit a fresh row-major contiguous copy of the
  // logical tensor. Returns the byte storage; consumers that want a
  // typed view call materialize_contiguous_as<T>() instead.
  //
  // v1 assumes strides.back() == 1; non-unit innermost stride is out
  // of scope.
  // ------------------------------------------------------------------

  AlignedVector<uint8_t>
  materialize_contiguous() const
  {
    const std::size_t esz = element_byte_size();
    const std::size_t n   = element_count();
    AlignedVector<uint8_t> out;
    out.assign(n * esz, 0);
    if (n == 0) {
      return out;
    }
    if (is_contiguous()) {
      std::memcpy(out.data(),
                  bytes_()
                      + static_cast<std::size_t>(storage_offset) * esz,
                  n * esz);
      return out;
    }
    if (shape.empty()) {
      return out;
    }
    const std::size_t inner      = static_cast<std::size_t>(shape.back());
    const std::size_t outer_dims = shape.size() - 1;
    std::vector<int64_t> idx(outer_dims, 0);
    std::size_t out_pos = 0;
    while (true) {
      int64_t src = storage_offset;
      for (std::size_t d = 0; d < outer_dims; ++d) {
        src += idx[d] * strides[d];
      }
      if (inner > 0) {
        std::memcpy(out.data() + out_pos * esz,
                    bytes_() + static_cast<std::size_t>(src) * esz,
                    inner * esz);
      }
      out_pos += inner;
      if (outer_dims == 0) {
        break;
      }
      std::size_t d = outer_dims;
      while (d-- > 0) {
        if (++idx[d] < shape[d]) {
          break;
        }
        idx[d] = 0;
        if (d == 0) {
          return out;
        }
      }
    }
    return out;
  }

  // Typed convenience: materialize as a contiguous AlignedVector<T>.
  // T must match the tensor's dtype.
  template <typename T>
  AlignedVector<T>
  materialize_contiguous_as() const
  {
    assert(byte_size_of(dtype) == sizeof(T));
    auto bytes = materialize_contiguous();
    AlignedVector<T> out;
    const std::size_t n = bytes.size() / sizeof(T);
    out.resize(n);
    if (n > 0) {
      std::memcpy(out.data(), bytes.data(), n * sizeof(T));
    }
    return out;
  }
};

// Pipeline-edge payload form of TensorBeat. Holds the data verbatim
// (inherits from TensorBeat) and adds the BeatPayloadIntf virtuals.
// clone() is a deep copy today; consumers in production are
// single-cursor so it is never actually invoked on the hot path.
class TensorBeatPayload : public TensorBeat, public BeatPayloadIntf {
public:
  TensorBeatPayload() = default;
  explicit
  TensorBeatPayload(const TensorBeat& tb)
    : TensorBeat(tb)
  {}
  explicit
  TensorBeatPayload(TensorBeat&& tb) noexcept
    : TensorBeat(std::move(tb))
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    // TensorBeat's copy constructor already materialises Shared
    // storage into a CpuCached `data` so two clones never share
    // an MTL::Buffer -- the slow-cursor fanout path can freely
    // mutate its copy (e.g. detection-overlay drawing on the
    // bytes) without affecting any sibling cursor.
    return std::make_unique<TensorBeatPayload>(
        static_cast<const TensorBeat&>(*this));
  }

  std::string
  describe() const override
  {
    std::string s = "TensorBeat ";
    s += dtype_name();
    s += " [";
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i) { s += ","; }
      s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
  }
};

}

#endif
