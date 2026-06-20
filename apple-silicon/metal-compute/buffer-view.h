#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_BUFFER_VIEW_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_BUFFER_VIEW_H

#include <cstddef>
#include <cstdint>

namespace vpipe::metal_compute {

// Element type held by a buffer. Mirrors vpipe::TensorBeat::DType so
// the SharedBuffer <-> TensorBeat bridge is mechanical; widen this
// enum (and element_size below) when TensorBeat adds new dtypes.
enum class DType : std::uint8_t {
  U8,
  I8,
  Bf16,
  F32
};

// Optional shape/stride/dtype metadata attached to a SharedBuffer.
// Strides and offset are in ELEMENTS (matching TensorBeat), not
// bytes. Maximum rank is fixed at 8 so the view is a POD value type
// that kernel launch code can write into an MSL constant buffer
// without an allocation.
struct BufferView {
  static constexpr int kMaxRank = 8;

  DType         dtype                = DType::F32;
  std::uint8_t  rank                 = 0;
  std::int64_t  shape  [kMaxRank]    = {0};
  std::int64_t  strides[kMaxRank]    = {0};
  std::int64_t  offset               = 0;
};

// Byte width of one element of `t`.
inline constexpr std::size_t
element_size(DType t) noexcept
{
  switch (t) {
    case DType::U8:   return 1;
    case DType::I8:   return 1;
    case DType::Bf16: return 2;
    case DType::F32:  return 4;
  }
  return 0;
}

}  // namespace vpipe::metal_compute

#endif
