#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_TENSOR_BEAT_BRIDGE_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_TENSOR_BEAT_BRIDGE_H

#include "apple-silicon/metal-compute/shared-buffer.h"

namespace vpipe {
struct TensorBeat;
}

namespace vpipe::metal_compute {

class MetalCompute;

// Convert a vpipe::TensorBeat into a SharedBuffer with its
// BufferView pre-populated (rank/shape/strides/dtype/offset).
//
// Zero-copy when `tb.storage_class() == Shared`: the underlying
// MTL::Buffer is retained once for the returned SharedBuffer; the
// TensorBeat continues to own its original refcount, so the bytes
// outlive whichever handle is released first.
//
// Allocate-and-copy when `tb.storage_class() == CpuCached`: a
// fresh Shared buffer is allocated through `mc` and the tensor's
// bytes are memcpy'd in verbatim. Returned BufferView still
// carries the original storage_offset; kernels apply it via
// vpipe_tv_index().
//
// Returns an empty SharedBuffer if `tb.byte_size() == 0` or `mc`
// is not valid().
SharedBuffer
from_tensor_beat(MetalCompute& mc, const TensorBeat& tb);

// Convert a SharedBuffer into a TensorBeat. Always zero-copy: the
// MTL::Buffer ownership transfers into the returned TensorBeat's
// ExternalStorageHandle. The input SharedBuffer is left empty.
// `BufferView` metadata (rank/shape/strides/dtype/offset) is
// materialised into the TensorBeat's shape/strides/dtype/
// storage_offset fields.
//
// Returns a default TensorBeat (CpuCached, empty data) if the
// input SharedBuffer is empty.
//
// Note: if the SharedBuffer was wired, it is unwired as part of
// the move (SharedBuffer's destructor handles the munlock).
// TensorBeat has no equivalent wired-memory concept.
TensorBeat
to_tensor_beat(SharedBuffer&& sb);

}  // namespace vpipe::metal_compute

#endif
