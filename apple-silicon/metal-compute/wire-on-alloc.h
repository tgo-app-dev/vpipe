#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_WIRE_ON_ALLOC_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_WIRE_ON_ALLOC_H

// Buffers WIRED AS THEY ARE MADE, on this thread, while a scope is open:
// MetalCompute::make_shared_buffer mlock's every buffer of at least 64 KB it
// allocates from the device (not the small-allocation heap), before anything
// is written to it.
//
// WHY. A model whose weights are read into fresh anonymous memory on a box
// without that much free has them COMPRESSED as they land: they are the
// largest cold anonymous pages on the system, and the compressor takes them
// while the read is still going. The first command buffer then has to make
// them resident, and the box thrashes between decompressing them and
// compressing them again. MEASURED on a 24 GB M5 Pro, Qwen3.8-27B 4-bit
// (16 GB) with 2.5 GB held by another process: the compressor grew to
// 12.5 GB in two seconds, and the first forward took 32 s of ~35K
// decompressions, ~30K compressions and ~17K swap-ins a second; wired as
// read, 1.8 s. The GPU wires every one of these pages at first use anyway,
// so wiring them earlier costs no memory the model would not hold -- the
// kernel reclaims file cache and other processes' cold pages instead of the
// model's.
//
// A refused mlock (the box at its user-wire limit) leaves that buffer as it
// was: unwired, exactly as without a scope. Freeing a wired buffer unwires
// it in the kernel, so a temporary made inside the scope costs nothing past
// its life. Whoever opens a scope accounts for what it wired (the
// generative model manager charges it to the wired pool).
//
// INTERNAL: not installed and not exported (libvpipe is hidden by default).
// It is a host's load-time policy, not something a plugin calls.

#include <cstddef>

namespace vpipe::metal_compute {

class WireOnAlloc {
public:
  WireOnAlloc() noexcept;
  ~WireOnAlloc();
  WireOnAlloc(const WireOnAlloc&)            = delete;
  WireOnAlloc& operator=(const WireOnAlloc&) = delete;

  // A scope is open on this thread (they nest).
  static bool active() noexcept;
  // make_shared_buffer's report of a buffer it wired.
  static void note_wired(std::size_t bytes) noexcept;
  // What this scope wired, freed buffers included.
  std::size_t wired_bytes() const noexcept;

  // The smallest buffer worth a syscall: the wired pool's own minimum.
  static constexpr std::size_t kMinBytes = 64 * 1024;

private:
  std::size_t _start;
};

}  // namespace vpipe::metal_compute

#endif
