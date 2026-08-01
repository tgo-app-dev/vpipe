#include "apple-silicon/metal-compute/shared-buffer.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace vpipe::metal_compute {

namespace {

// Round `n` up to the next multiple of `page_size`. `page_size` is
// always a power of two on Darwin/Linux, but stay safe with the
// generic ceil-div form.
inline std::size_t
round_up_to_page_(std::size_t n, std::size_t page_size) noexcept
{
  return ((n + page_size - 1) / page_size) * page_size;
}

}  // namespace

// ---- accounting counters (see MemoryStats) ------------------------------
namespace {
std::atomic<std::size_t> g_live_bytes{0};
std::atomic<std::size_t> g_peak_bytes{0};
std::atomic<std::size_t> g_total_bytes{0};
std::atomic<std::size_t> g_live_count{0};
std::atomic<std::size_t> g_total_count{0};

// VPIPE_MC_ALLOC_LOG=<MB>: log every allocation at or above this size with the
// running live total. Set it to attribute a footprint to specific buffers --
// a size is usually enough to identify the caller (e.g. n*vocab*2).
std::size_t alloc_log_threshold_()
{
  static const std::size_t v = [] {
    const char* e = std::getenv("VPIPE_MC_ALLOC_LOG");
    if (e == nullptr) { return (std::size_t)0; }
    const double mb = std::atof(e);
    return mb > 0.0 ? (std::size_t)(mb * 1024.0 * 1024.0) : (std::size_t)0;
  }();
  return v;
}
}  // namespace

void
account_alloc_(std::size_t bytes) noexcept
{
  if (bytes == 0) { return; }
  const std::size_t live =
      g_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  g_total_bytes.fetch_add(bytes, std::memory_order_relaxed);
  g_live_count.fetch_add(1, std::memory_order_relaxed);
  g_total_count.fetch_add(1, std::memory_order_relaxed);
  // Monotonic max without a lock.
  std::size_t prev = g_peak_bytes.load(std::memory_order_relaxed);
  while (live > prev
         && !g_peak_bytes.compare_exchange_weak(prev, live,
                                                std::memory_order_relaxed)) {
  }
  const std::size_t thr = alloc_log_threshold_();
  if (thr != 0 && bytes >= thr) {
    std::fprintf(stderr, "[mc-alloc] %8.1f MB  live=%8.1f MB  peak=%8.1f MB\n",
                 (double)bytes / (1024.0 * 1024.0),
                 (double)live / (1024.0 * 1024.0),
                 (double)g_peak_bytes.load(std::memory_order_relaxed)
                     / (1024.0 * 1024.0));
  }
}

void
account_free_(std::size_t bytes) noexcept
{
  if (bytes == 0) { return; }
  g_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  g_live_count.fetch_sub(1, std::memory_order_relaxed);
}

MemoryStats
shared_buffer_memory_stats() noexcept
{
  MemoryStats m;
  m.live_bytes  = g_live_bytes.load(std::memory_order_relaxed);
  m.peak_bytes  = g_peak_bytes.load(std::memory_order_relaxed);
  m.total_bytes = g_total_bytes.load(std::memory_order_relaxed);
  m.live_count  = g_live_count.load(std::memory_order_relaxed);
  m.total_count = g_total_count.load(std::memory_order_relaxed);
  return m;
}

void
shared_buffer_reset_peak() noexcept
{
  g_peak_bytes.store(g_live_bytes.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
}

SharedBuffer::SharedBuffer(MTL::Buffer* buf, void* contents,
                           std::size_t byte_size) noexcept
  : _buf(buf),
    _contents(contents),
    _byte_size(byte_size)
{
}

SharedBuffer
SharedBuffer::wrap(MTL::Buffer* buf, void* contents,
                   std::size_t byte_size) noexcept
{
  if (buf == nullptr || byte_size == 0) {
    return SharedBuffer{};
  }
  void* c = contents != nullptr ? contents : buf->contents();
  return SharedBuffer{buf, c, byte_size};
}

SharedBuffer
SharedBuffer::subview(std::size_t offset, std::size_t size) const noexcept
{
  if (_buf == nullptr || size == 0) {
    return SharedBuffer{};
  }
  _buf->retain();                          // the subview holds its own ref
  SharedBuffer sv{_buf,
                  static_cast<void*>(
                      static_cast<std::uint8_t*>(_contents) + offset),
                  size};
  sv._base_off = _base_off + offset;       // composes with this view's offset
  return sv;
}

SharedBuffer::SharedBuffer(SharedBuffer&& o) noexcept
  : _buf(std::exchange(o._buf, nullptr)),
    _contents(std::exchange(o._contents, nullptr)),
    _byte_size(std::exchange(o._byte_size, 0)),
    _base_off(std::exchange(o._base_off, 0)),
    _view(o._view),
    _wired(std::exchange(o._wired, false)),
    // The accounting ownership moves WITH the buffer -- if it did not, the
    // destination would never decrement on destruction and live_bytes would
    // only ever climb (nearly every buffer is move-assigned into place).
    _accounted(std::exchange(o._accounted, false)),
    // Same reasoning for the parked flag: lose it and reactivate() on
    // the destination reports "intact" for a buffer the kernel may
    // already have discarded, which reads as silent garbage.
    _inactive(std::exchange(o._inactive, false))
{
  o._view = {};
}

SharedBuffer&
SharedBuffer::operator=(SharedBuffer&& o) noexcept
{
  if (this == &o) {
    return *this;
  }
  teardown_();
  _buf       = std::exchange(o._buf, nullptr);
  _contents  = std::exchange(o._contents, nullptr);
  _byte_size = std::exchange(o._byte_size, 0);
  _base_off  = std::exchange(o._base_off, 0);
  _view      = o._view;
  _wired     = std::exchange(o._wired, false);
  _accounted = std::exchange(o._accounted, false);
  _inactive  = std::exchange(o._inactive, false);
  o._view    = {};
  return *this;
}

SharedBuffer::~SharedBuffer()
{
  teardown_();
}

void
SharedBuffer::teardown_() noexcept
{
  if (_wired && _contents != nullptr && _byte_size != 0) {
    // mlock() rounded the address down and the length up to page
    // boundaries; munlock() must call with the same bounds. The
    // contents pointer of an MTL Shared buffer is already page-
    // aligned on Darwin, so just round the length up.
    const std::size_t page_size =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t unlock_size =
        round_up_to_page_(_byte_size, page_size);
    ::munlock(_contents, unlock_size);
    _wired = false;
  }
  if (_buf != nullptr) {
    if (_accounted) {
      account_free_(_byte_size);
      _accounted = false;
    }
    _buf->release();
    _buf = nullptr;
  }
  _contents  = nullptr;
  _byte_size = 0;
  _base_off  = 0;
}

bool
SharedBuffer::set_wired(bool on) noexcept
{
  if (_buf == nullptr || _contents == nullptr || _byte_size == 0) {
    // Nothing to wire; treat as a successful no-op so callers don't
    // have to special-case empty buffers.
    return true;
  }
  if (on == _wired) {
    return true;
  }
  const std::size_t page_size =
      static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  const std::size_t lock_size =
      round_up_to_page_(_byte_size, page_size);

  if (on) {
    if (::mlock(_contents, lock_size) != 0) {
      // errno preserved by mlock(); caller can inspect.
      return false;
    }
    _buf->setPurgeableState(MTL::PurgeableStateNonVolatile);
    _wired = true;
  } else {
    // Best-effort unlock. A munlock failure is non-fatal -- the OS
    // may unwire automatically when the address is released. We
    // flip the flag regardless so a subsequent set_wired(true)
    // re-attempts the lock.
    ::munlock(_contents, lock_size);
    _buf->setPurgeableState(MTL::PurgeableStateVolatile);
    _wired = false;
  }
  return true;
}

bool
SharedBuffer::mark_inactive() noexcept
{
  if (_buf == nullptr || _byte_size == 0) { return false; }
  // Not ours to park: a subview would evict another handle's memory,
  // and a heap sub-allocation's residency is the heap's to manage.
  if (!_accounted || _buf->heap() != nullptr) { return false; }
  if (_wired) { return false; }        // deliberate "never evict this"
  if (_inactive) { return true; }
  _buf->setPurgeableState(MTL::PurgeableStateVolatile);
  _inactive = true;
  return true;
}

bool
SharedBuffer::reactivate() noexcept
{
  if (!_inactive) { return true; }     // never parked -> nothing lost
  // setPurgeableState returns the PRIOR state. Empty means the kernel
  // reclaimed the pages while they were volatile, so whatever is there
  // now is undefined and the caller has to reload.
  const MTL::PurgeableState prev =
      _buf->setPurgeableState(MTL::PurgeableStateNonVolatile);
  _inactive = false;
  return prev != MTL::PurgeableStateEmpty;
}

}  // namespace vpipe::metal_compute
