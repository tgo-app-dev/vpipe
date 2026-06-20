#include "apple-silicon/metal-compute/tensor-beat-bridge.h"

#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "apple-silicon/tensor-storage.h"

#include <Metal/Metal.hpp>

#include <cstring>
#include <memory>
#include <utility>

namespace vpipe::metal_compute {

namespace {

// Deleter stored in the ExternalStorageHandle the bridge produces.
// Just releases one refcount on the MTL::Buffer; the buffer was
// allocated through MetalCompute::make_shared_buffer or carried in
// from another TensorBeat, so the refcount semantics are uniform.
void
release_mtl_buffer_(void* mtl_buffer)
{
  if (mtl_buffer != nullptr) {
    static_cast<MTL::Buffer*>(mtl_buffer)->release();
  }
}

// TensorBeat::DType uses the same ordinals as metal_compute::DType
// (U8=0, I8=1, Bf16=2, F32=3) -- the matching order is intentional
// so the conversion is mechanical. The explicit switch documents
// the contract and would catch reordering.
DType
to_view_dtype_(TensorBeat::DType d) noexcept
{
  switch (d) {
    case TensorBeat::DType::U8:   return DType::U8;
    case TensorBeat::DType::I8:   return DType::I8;
    case TensorBeat::DType::Bf16: return DType::Bf16;
    case TensorBeat::DType::F32:  return DType::F32;
  }
  return DType::F32;
}

TensorBeat::DType
to_beat_dtype_(DType d) noexcept
{
  switch (d) {
    case DType::U8:   return TensorBeat::DType::U8;
    case DType::I8:   return TensorBeat::DType::I8;
    case DType::Bf16: return TensorBeat::DType::Bf16;
    case DType::F32:  return TensorBeat::DType::F32;
  }
  return TensorBeat::DType::F32;
}

// Build a BufferView from a TensorBeat's shape/strides/dtype/
// storage_offset. If the TensorBeat is row-major contiguous (empty
// strides), compute the implied strides. Ranks beyond
// BufferView::kMaxRank are truncated.
BufferView
view_from_tensor_beat_(const TensorBeat& tb)
{
  BufferView v{};
  v.dtype  = to_view_dtype_(tb.dtype);
  v.offset = tb.storage_offset;
  const std::size_t r = std::min<std::size_t>(
      tb.shape.size(),
      static_cast<std::size_t>(BufferView::kMaxRank));
  v.rank = static_cast<std::uint8_t>(r);

  const bool contig = tb.strides.empty();
  std::int64_t running = 1;
  if (contig) {
    // contiguous: innermost stride = 1, outer = running product
    for (std::size_t i = r; i-- > 0;) {
      v.shape[i]   = tb.shape[i];
      v.strides[i] = running;
      running *= tb.shape[i];
    }
  } else {
    for (std::size_t i = 0; i < r; ++i) {
      v.shape[i]   = tb.shape[i];
      v.strides[i] = tb.strides[i];
    }
  }
  return v;
}

}  // namespace

SharedBuffer
from_tensor_beat(MetalCompute& mc, const TensorBeat& tb)
{
  const std::size_t n = tb.byte_size();
  if (n == 0 || !mc.valid()) {
    return SharedBuffer{};
  }

  if (tb.storage_class() == TensorStorageClass::Shared) {
    // Zero-copy: take an extra refcount on the same MTL::Buffer.
    // The original TensorBeat keeps its refcount via its
    // ExternalStorageHandle; the bridge's returned SharedBuffer
    // owns its own. Both can outlive the other.
    auto* buf = static_cast<MTL::Buffer*>(tb.external->mtl_buffer);
    if (buf == nullptr) {
      return SharedBuffer{};
    }
    buf->retain();
    SharedBuffer out = SharedBuffer::wrap(
        buf, tb.external->contents, tb.external->byte_size);
    out.set_view(view_from_tensor_beat_(tb));
    return out;
  }

  // CpuCached path: allocate fresh + bytewise copy.
  SharedBuffer out = mc.make_shared_buffer(n);
  if (out.empty()) {
    return SharedBuffer{};
  }
  std::memcpy(out.contents(), tb.data.data(), n);
  out.set_view(view_from_tensor_beat_(tb));
  return out;
}

TensorBeat
to_tensor_beat(SharedBuffer&& sb_in)
{
  // Move the rvalue ref into a local so the caller-side SharedBuffer
  // is actually emptied (binding `sb_in` directly leaves the
  // caller's object intact until its own scope ends -- which would
  // double-release the MTL::Buffer the handle now owns).
  SharedBuffer sb(std::move(sb_in));

  if (sb.empty()) {
    return TensorBeat{};
  }

  TensorBeat tb;
  const BufferView& v = sb.view();
  tb.dtype          = to_beat_dtype_(v.dtype);
  tb.storage_offset = v.offset;

  if (v.rank > 0) {
    tb.shape.assign(v.shape, v.shape + v.rank);
    tb.strides.assign(v.strides, v.strides + v.rank);
  }

  // Transfer MTL::Buffer ownership. The local SharedBuffer's
  // destructor will release one refcount at function return;
  // pre-retain so the ExternalStorageHandle owns the surviving
  // refcount.
  auto* buf = sb.mtl_buffer();
  buf->retain();

  auto handle = std::make_unique<ExternalStorageHandle>();
  handle->mtl_buffer = buf;
  handle->contents   = static_cast<std::uint8_t*>(sb.contents());
  handle->byte_size  = sb.byte_size();
  handle->deleter    = &release_mtl_buffer_;
  tb.external        = std::move(handle);

  // Local `sb`'s dtor runs at return: it releases the pre-bumped
  // refcount (net effect: zero change to buf's refcount across
  // the bridge call) and the new TensorBeat owns the buffer.
  return tb;
}

}  // namespace vpipe::metal_compute
