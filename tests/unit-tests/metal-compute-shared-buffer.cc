#include "minitest.h"
#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <Metal/Metal.hpp>

#include <cstdint>
#include <cstring>
#include <utility>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

// Helper: get a valid() MetalCompute on this host, or nullptr if
// Metal is unavailable (in which case the test self-skips).
MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

}  // namespace

TEST(metal_compute_shared_buffer, alloc_and_release) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  {
    SharedBuffer buf = mc->make_shared_buffer(4096);
    EXPECT_FALSE(buf.empty());
    EXPECT_TRUE(buf.contents() != nullptr);
    EXPECT_TRUE(buf.byte_size() == 4096u);
    EXPECT_TRUE(buf.mtl_buffer() != nullptr);
  }
  // Destruction completes without crashing.
  EXPECT_TRUE(true);
}

TEST(metal_compute_shared_buffer, contents_is_64_byte_aligned) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(1024);
  const auto ptr = reinterpret_cast<std::uintptr_t>(buf.contents());
  EXPECT_TRUE((ptr & 63u) == 0u);
}

TEST(metal_compute_shared_buffer, zero_size_is_empty) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(0);
  EXPECT_TRUE(buf.empty());
  EXPECT_TRUE(buf.contents() == nullptr);
  EXPECT_TRUE(buf.byte_size() == 0u);
}

TEST(metal_compute_shared_buffer, contents_round_trips_through_cpu) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(256);
  auto* p = static_cast<std::uint8_t*>(buf.contents());
  for (std::size_t i = 0; i < buf.byte_size(); ++i) {
    p[i] = static_cast<std::uint8_t>(i ^ 0xA5);
  }
  for (std::size_t i = 0; i < buf.byte_size(); ++i) {
    EXPECT_TRUE(p[i] == static_cast<std::uint8_t>(i ^ 0xA5));
  }
}

TEST(metal_compute_shared_buffer, view_setter_round_trips) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer buf = mc->make_shared_buffer(64);
  BufferView v{};
  v.dtype      = DType::F32;
  v.rank       = 2;
  v.shape[0]   = 4;
  v.shape[1]   = 4;
  v.strides[0] = 4;
  v.strides[1] = 1;
  v.offset     = 0;
  buf.set_view(v);
  EXPECT_TRUE(buf.view().dtype == DType::F32);
  EXPECT_TRUE(buf.view().rank == 2);
  EXPECT_TRUE(buf.view().shape[0] == 4);
  EXPECT_TRUE(buf.view().shape[1] == 4);
  EXPECT_TRUE(buf.view().strides[0] == 4);
  EXPECT_TRUE(buf.view().strides[1] == 1);
}

TEST(metal_compute_shared_buffer, move_ctor_transfers_ownership) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer src = mc->make_shared_buffer(128);
  EXPECT_FALSE(src.empty());
  void*       contents_before = src.contents();
  MTL::Buffer* mtl_before     = src.mtl_buffer();

  SharedBuffer dst(std::move(src));
  EXPECT_TRUE(src.empty());
  EXPECT_TRUE(src.contents() == nullptr);
  EXPECT_TRUE(src.byte_size() == 0u);
  EXPECT_TRUE(dst.contents() == contents_before);
  EXPECT_TRUE(dst.mtl_buffer() == mtl_before);
  EXPECT_TRUE(dst.byte_size() == 128u);
}

TEST(metal_compute_shared_buffer, move_assignment_releases_old) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer a = mc->make_shared_buffer(256);
  SharedBuffer b = mc->make_shared_buffer(512);
  a = std::move(b);
  EXPECT_TRUE(b.empty());
  EXPECT_FALSE(a.empty());
  EXPECT_TRUE(a.byte_size() == 512u);
}

TEST(metal_compute_shared_buffer, set_wired_round_trip) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Use one page so the lock fits under the default RLIMIT_MEMLOCK
  // (64KB) on a non-root macOS process.
  SharedBuffer buf = mc->make_shared_buffer(4096);
  void* contents_before = buf.contents();

  // mlock can fail on hosts with tight RLIMIT_MEMLOCK or in
  // sandboxes; if so the test self-skips the wired assertions.
  if (!buf.set_wired(true)) {
    return;
  }
  EXPECT_TRUE(buf.is_wired());
  EXPECT_TRUE(buf.contents() == contents_before);

  // Idempotent: second set_wired(true) is a no-op success.
  EXPECT_TRUE(buf.set_wired(true));
  EXPECT_TRUE(buf.is_wired());

  EXPECT_TRUE(buf.set_wired(false));
  EXPECT_FALSE(buf.is_wired());
  EXPECT_TRUE(buf.contents() == contents_before);

  // Idempotent: second set_wired(false) is a no-op success.
  EXPECT_TRUE(buf.set_wired(false));
  EXPECT_FALSE(buf.is_wired());
}

// UNWIRING MUST NOT MAKE THE BUFFER DISCARDABLE.
//
// set_wired(false) used to also flip the buffer to purgeable VOLATILE,
// which is a state its owner never asked for and which nothing tracked
// -- mark_inactive()/reactivate() keep their own `_inactive` flag, so a
// buffer volatile'd this way would never be restored and a reclaim would
// yield garbage instead of a reload.
//
// It was a crash, not a theory. Unwiring a streamed transformer block on
// the way to destroying it discarded pages of a shard mapping its
// neighbours were still reading; the block's own destructor took SIGBUS.
// The buffer must be as usable after an unwire as before the wire.
TEST(metal_compute_shared_buffer, unwiring_leaves_the_buffer_usable) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  SharedBuffer buf = mc->make_shared_buffer(4096);
  auto* p = static_cast<unsigned char*>(buf.contents());
  for (int i = 0; i < 4096; ++i) { p[i] = (unsigned char)(i & 0xff); }

  if (!buf.set_wired(true)) { return; }   // RLIMIT_MEMLOCK: self-skip
  EXPECT_TRUE(buf.set_wired(false));
  EXPECT_FALSE(buf.is_wired());

  // Readable, and still holding what was written. A volatile buffer the
  // kernel had reclaimed would not be.
  bool intact = true;
  for (int i = 0; i < 4096; ++i) {
    if (p[i] != (unsigned char)(i & 0xff)) { intact = false; break; }
  }
  EXPECT_TRUE(intact);

  // And parking still works afterwards -- the flag it keeps was never
  // touched by the wire/unwire round trip, so this is a real park and
  // not a no-op on a buffer that was already volatile.
  if (buf.mark_inactive()) {
    EXPECT_TRUE(buf.reactivate());
  }
}

// A SUBVIEW's contents() need not be page-aligned -- 8 bytes in is
// exactly the shape a safetensors tensor takes inside a mapped shard,
// and it is the shape a streaming transformer wires per block.
//
// A SMOKE TEST, and it is worth being clear about what it does not do:
// mlock's extent is not observable from here, so this cannot pin the
// locked range. It pins that the operation succeeds and leaves the view
// readable at both ends, which is what a caller depends on.
TEST(metal_compute_shared_buffer, wiring_an_unaligned_subview_is_safe) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  SharedBuffer base = mc->make_shared_buffer(64 * 1024);
  auto* bp = static_cast<unsigned char*>(base.contents());
  for (int i = 0; i < 64 * 1024; ++i) { bp[i] = (unsigned char)(i & 0x7f); }

  // 8 bytes in: never page-aligned, which is exactly the shape a
  // safetensors tensor takes inside a mapped shard.
  SharedBuffer view = base.subview(8, 4096);
  if (view.empty()) { return; }           // no subview support: skip
  auto* vp = static_cast<const unsigned char*>(view.contents());
  if (!view.set_wired(true)) { return; }
  EXPECT_TRUE(view.is_wired());
  EXPECT_TRUE(vp[0] == (unsigned char)(8 & 0x7f));
  // The TAIL is the part the old length left out.
  EXPECT_TRUE(vp[4095] == (unsigned char)((8 + 4095) & 0x7f));
  EXPECT_TRUE(view.set_wired(false));
  EXPECT_TRUE(vp[4095] == (unsigned char)((8 + 4095) & 0x7f));
}

TEST(metal_compute_shared_buffer, destructor_unwires_automatically) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  {
    SharedBuffer buf = mc->make_shared_buffer(4096);
    if (!buf.set_wired(true)) {
      return;
    }
    EXPECT_TRUE(buf.is_wired());
    // Destructor runs at end of scope and should munlock + release
    // without crashing.
  }
  EXPECT_TRUE(true);
}

TEST(metal_compute_shared_buffer, set_wired_on_empty_is_noop) {
  SharedBuffer empty;
  EXPECT_TRUE(empty.set_wired(true));
  EXPECT_FALSE(empty.is_wired());
  EXPECT_TRUE(empty.set_wired(false));
}

TEST(metal_compute_shared_buffer, default_constructed_is_empty) {
  SharedBuffer buf;
  EXPECT_TRUE(buf.empty());
  EXPECT_TRUE(buf.contents() == nullptr);
  EXPECT_TRUE(buf.byte_size() == 0u);
  EXPECT_TRUE(buf.mtl_buffer() == nullptr);
  EXPECT_FALSE(buf.is_wired());
}

// ---- residency: park an inactive weight buffer, take it back --------

// The common case: park, reactivate, contents intact. Under no memory
// pressure the kernel keeps volatile pages, so this is the fast path a
// model reload is supposed to avoid -- a state flip, no I/O.
TEST(metal_compute_shared_buffer, reactivate_reports_intact_contents) {
  Session s;
  MetalCompute* mc = get_mc_(s);
  if (mc == nullptr) { return; }

  SharedBuffer b = mc->make_shared_buffer(1 << 20);   // 1 MiB, device path
  ASSERT_TRUE(!b.empty());
  auto* p = static_cast<std::uint8_t*>(b.contents());
  for (int i = 0; i < 256; ++i) { p[i] = static_cast<std::uint8_t>(i); }

  EXPECT_TRUE(b.mark_inactive());
  EXPECT_TRUE(b.is_inactive());
  // The allocation and its CPU address survive parking -- only the
  // pages become reclaimable.
  EXPECT_TRUE(b.contents() == static_cast<void*>(p));
  EXPECT_TRUE(b.byte_size() == (1u << 20));

  EXPECT_TRUE(b.reactivate());          // nothing else wanted the RAM
  EXPECT_FALSE(b.is_inactive());
  bool same = true;
  for (int i = 0; i < 256; ++i) {
    if (p[i] != static_cast<std::uint8_t>(i)) { same = false; break; }
  }
  EXPECT_TRUE(same);
}

// reactivate() on a buffer that was never parked must report intact --
// otherwise every caller would reload weights it still has.
TEST(metal_compute_shared_buffer, reactivate_without_parking_is_intact) {
  Session s;
  MetalCompute* mc = get_mc_(s);
  if (mc == nullptr) { return; }
  SharedBuffer b = mc->make_shared_buffer(4096);
  ASSERT_TRUE(!b.empty());
  EXPECT_FALSE(b.is_inactive());
  EXPECT_TRUE(b.reactivate());
}

// A SUBVIEW shares another handle's allocation, so parking it would
// evict memory it does not own. It must refuse and stay resident.
TEST(metal_compute_shared_buffer, subview_refuses_to_park) {
  Session s;
  MetalCompute* mc = get_mc_(s);
  if (mc == nullptr) { return; }
  SharedBuffer b = mc->make_shared_buffer(1 << 16);
  ASSERT_TRUE(!b.empty());
  SharedBuffer sv = b.subview(0, 4096);
  ASSERT_TRUE(!sv.empty());
  EXPECT_FALSE(sv.mark_inactive());
  EXPECT_FALSE(sv.is_inactive());
  EXPECT_TRUE(sv.reactivate());      // never parked -> intact
}

// An explicitly WIRED buffer is a deliberate "never evict this".
TEST(metal_compute_shared_buffer, wired_buffer_refuses_to_park) {
  Session s;
  MetalCompute* mc = get_mc_(s);
  if (mc == nullptr) { return; }
  SharedBuffer b = mc->make_shared_buffer(1 << 16);
  ASSERT_TRUE(!b.empty());
  if (!b.set_wired(true)) { return; }   // mlock limit -- nothing to test
  EXPECT_FALSE(b.mark_inactive());
  EXPECT_FALSE(b.is_inactive());
  b.set_wired(false);
}

// The parked flag must MOVE with the buffer. If it were dropped, the
// destination's reactivate() would report "intact" for pages the kernel
// may already have discarded -- silent garbage, the same class of bug
// the accounting flag hit.
TEST(metal_compute_shared_buffer, parked_state_survives_a_move) {
  Session s;
  MetalCompute* mc = get_mc_(s);
  if (mc == nullptr) { return; }
  SharedBuffer b = mc->make_shared_buffer(1 << 20);
  ASSERT_TRUE(!b.empty());
  EXPECT_TRUE(b.mark_inactive());

  SharedBuffer moved = std::move(b);
  EXPECT_TRUE(moved.is_inactive());
  EXPECT_FALSE(b.is_inactive());        // moved-from is inert

  SharedBuffer assigned;
  assigned = std::move(moved);
  EXPECT_TRUE(assigned.is_inactive());
  EXPECT_TRUE(assigned.reactivate());
  EXPECT_FALSE(assigned.is_inactive());
}
