#include "stages/tensor-buffer.h"

#include <cstring>
#include <vector>

namespace vpipe {

ElementType
element_type_of(TensorBeat::DType d) noexcept
{
  switch (d) {
    case TensorBeat::DType::U8:   return ElementType::U8;
    case TensorBeat::DType::I8:   return ElementType::I8;
    case TensorBeat::DType::Bf16: return ElementType::BF16;
    case TensorBeat::DType::F16:  return ElementType::F16;
    case TensorBeat::DType::F32:  return ElementType::F32;
  }
  return ElementType::U8;
}

bool
tensor_dtype_of(ElementType t, TensorBeat::DType* out) noexcept
{
  TensorBeat::DType d{};
  switch (t) {
    case ElementType::U8:    d = TensorBeat::DType::U8;   break;
    case ElementType::Bytes: d = TensorBeat::DType::U8;   break;
    case ElementType::I8:    d = TensorBeat::DType::I8;   break;
    case ElementType::BF16:  d = TensorBeat::DType::Bf16; break;
    case ElementType::F16:   d = TensorBeat::DType::F16;  break;
    case ElementType::F32:   d = TensorBeat::DType::F32;  break;
    default: return false;
  }
  if (out != nullptr) { *out = d; }
  return true;
}

DataBuffer
tensor_view(const TensorBeat& tb, std::string name,
            std::shared_ptr<void> owner, bool writable)
{
  const std::size_t es = tb.element_byte_size();
  DataBuffer b;
  b.name        = std::move(name);
  b.layout.type = element_type_of(tb.dtype);
  b.layout.shape.assign(tb.shape.begin(), tb.shape.end());
  // TensorBeat strides are in ELEMENTS; a DataBuffer's are in bytes.
  if (!tb.strides.empty()) {
    b.layout.strides.reserve(tb.strides.size());
    for (std::int64_t s : tb.strides) {
      b.layout.strides.push_back(s * static_cast<std::int64_t>(es));
    }
  }
  const std::size_t off = static_cast<std::size_t>(tb.storage_offset) * es;
  const std::size_t total = tb.byte_size();
  b.data     = const_cast<std::uint8_t*>(tb.bytes_()) + off;
  b.size     = total > off ? total - off : 0;
  b.writable = writable;
  b.owner    = std::move(owner);
  return b;
}

bool
tensor_from_buffer(const DataBuffer& b, TensorBeat* out, std::string* err)
{
  auto bad = [&](const char* why) {
    if (err != nullptr) { *err = why; }
    return false;
  };
  TensorBeat::DType d{};
  if (!tensor_dtype_of(b.layout.type, &d)) {
    return bad("the element type has no TensorBeat dtype");
  }
  if (!b.valid()) { return bad("the layout does not fit the buffer"); }

  TensorBeat tb;
  tb.dtype = d;
  tb.shape.assign(b.layout.shape.begin(), b.layout.shape.end());
  const std::size_t n  = static_cast<std::size_t>(b.layout.element_count());
  const std::size_t es = element_size(b.layout.type);
  tb.resize_contiguous(n);
  if (n > 0) {
    const auto* src = static_cast<const std::uint8_t*>(b.data);
    if (b.layout.is_contiguous()) {
      std::memcpy(tb.data.data(), src, n * es);
    } else {
      // Gather row by row along the innermost dimension, which is one
      // memcpy when that dimension is packed and one per element when
      // it is not.
      const auto& shape = b.layout.shape;
      const auto& st    = b.layout.strides;
      const std::size_t rank  = shape.size();
      const std::size_t inner = static_cast<std::size_t>(shape[rank - 1]);
      const bool packed = st[rank - 1] == static_cast<std::int64_t>(es);
      std::vector<std::int64_t> idx(rank, 0);
      std::uint8_t* dst = tb.data.data();
      for (std::size_t row = 0; row < n / inner; ++row) {
        std::int64_t at = 0;
        for (std::size_t k = 0; k + 1 < rank; ++k) { at += idx[k] * st[k]; }
        if (packed) {
          std::memcpy(dst, src + at, inner * es);
          dst += inner * es;
        } else {
          for (std::size_t j = 0; j < inner; ++j) {
            std::memcpy(dst, src + at + j * st[rank - 1], es);
            dst += es;
          }
        }
        // Advance the outer index, last dimension fastest.
        for (std::size_t k = rank - 1; k-- > 0;) {
          if (++idx[k] < shape[k]) { break; }
          idx[k] = 0;
        }
      }
    }
  }
  *out = std::move(tb);
  return true;
}

}
