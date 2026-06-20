#ifndef VPIPE_APPLE_SILICON_ALIGNED_ALLOCATOR_H
#define VPIPE_APPLE_SILICON_ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

namespace vpipe {

// std::allocator-conformant allocator that hands back memory aligned
// to `Alignment` bytes. We use this for TensorBeat's byte storage so
// every typed view (f32, bf16, u8) starts on a SIMD-friendly boundary
// regardless of dtype.
//
// Default alignment is 64 bytes: matches the AVX-512 / Apple-Silicon
// L1 cache line and is a superset of every SIMD ISA we care about
// (NEON 16, SSE 16, AVX 32, AVX-512 / cache line 64).
//
// Backed by posix_memalign; the size passed to allocate() is *not*
// rounded up. posix_memalign requires alignment to be a power of two
// >= sizeof(void*); the default 64 satisfies that on every Apple
// target.
template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T*
  allocate(std::size_t n)
  {
    if (n == 0) {
      return nullptr;
    }
    void* p = nullptr;
    if (::posix_memalign(&p, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(p);
  }

  void
  deallocate(T* p, std::size_t /*n*/) noexcept
  {
    std::free(p);
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t A>
bool
operator==(const AlignedAllocator<T, A>&,
           const AlignedAllocator<U, A>&) noexcept
{
  return true;
}

template <typename T, typename U, std::size_t A>
bool
operator!=(const AlignedAllocator<T, A>&,
           const AlignedAllocator<U, A>&) noexcept
{
  return false;
}

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

}

#endif
