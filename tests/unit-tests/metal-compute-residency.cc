#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/metal-compute/texture.h"
#include "common/session.h"
#include "generative-models/weight-set.h"

#include <Metal/Metal.hpp>

#include <cstdlib>
#include <cstring>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

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

TEST(metal_compute_residency, support_probe_returns_bool) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  // Just check it doesn't crash; older hosts return false.
  const bool supported = mc->residency_set_supported();
  (void)supported;
  EXPECT_TRUE(true);
}

TEST(metal_compute_residency, add_buffer_increments_counter) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(4096);
  if (b.empty()) {
    return;
  }
  const auto before = mc->residency_stats();
  EXPECT_TRUE(mc->residency_add(b));
  EXPECT_TRUE(mc->residency_commit());
  const auto after = mc->residency_stats();
  EXPECT_TRUE(after.add_calls == before.add_calls + 1);
  EXPECT_TRUE(after.current >= before.current + 1);
}

TEST(metal_compute_residency, remove_buffer_decrements) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  SharedBuffer b = mc->make_shared_buffer(4096);
  if (b.empty()) {
    return;
  }
  mc->residency_add(b);
  mc->residency_commit();
  const auto mid = mc->residency_stats();

  EXPECT_TRUE(mc->residency_remove(b));
  EXPECT_TRUE(mc->residency_commit());
  const auto after = mc->residency_stats();
  EXPECT_TRUE(after.remove_calls == mid.remove_calls + 1);
  EXPECT_TRUE(after.current <= mid.current);
}

TEST(metal_compute_residency, add_texture_works) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  TextureDesc d{};
  d.format = PixelFormat::RGBA8Unorm;
  d.width  = 16;
  d.height = 16;
  Texture t = mc->make_texture(d);
  if (!t.valid()) {
    return;
  }
  EXPECT_TRUE(mc->residency_add(t));
  EXPECT_TRUE(mc->residency_commit());
  mc->residency_remove(t);
  mc->residency_commit();
}

TEST(metal_compute_residency, request_and_end_are_no_op_safe) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  if (!mc->residency_set_supported()) {
    return;
  }
  // Even with no allocations these should not crash.
  EXPECT_TRUE(mc->residency_request());
  EXPECT_TRUE(mc->residency_end());
}

TEST(metal_compute_residency, add_empty_buffer_is_rejected) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  SharedBuffer empty;
  EXPECT_FALSE(mc->residency_add(empty));
}

// Is a buffer we are holding still IN RAM?
//
// The question a streamed DiT's residency policy has to answer, and the
// only one that distinguishes a pin that is paying for its memory from
// one that is being compressed behind our back. Free-memory arithmetic
// cannot answer it -- on the box this fixes, `available_physical` read
// 18.5 GB while the machine held 28.7 GB of swap, because the figure
// counts file cache and a streaming model's cache is its own re-reads.
//
// VERIFIED against real pressure while this was written: a 2 GB
// anonymous buffer on a 16 GB box went from 2048/2048 sampled pages
// resident to 1636/2048 with 412 PAGED_OUT the moment 8 GB of ballast
// engaged the compressor -- before any swap, which is what makes it an
// early enough signal to act on.
TEST(metal_compute_residency, page_residency_sees_a_live_buffer)
{
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  SharedBuffer b = mc->make_shared_buffer(64ull << 20);
  if (b.empty()) { EXPECT_TRUE(false); return; }
  // Touch it, so the pages exist rather than being lazily unbacked.
  std::memset(b.contents(), 1, b.byte_size());

  const auto r = b.page_residency();
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.examined > 0);
  EXPECT_TRUE(r.fully_resident());
  EXPECT_TRUE(r.paged_out == 0);
  EXPECT_TRUE(r.resident_fraction() > 0.999);

  // Sampling looks at fewer pages and must agree about the verdict.
  const auto s = b.page_residency(64);
  EXPECT_TRUE(s.valid);
  EXPECT_TRUE(s.examined > 0);
  EXPECT_TRUE(s.examined < r.examined);
  EXPECT_TRUE(s.fully_resident());

  // An empty handle answers without claiming anything.
  SharedBuffer none;
  const auto e = none.page_residency();
  EXPECT_TRUE(!e.valid);
  EXPECT_TRUE(e.fully_resident());     // vacuous: nothing has left RAM
}

// The process-scoped figures the policy polls each forward. Unlike the
// system-wide compressor count these say whose memory it is, which is
// what makes "our own weights are being squeezed" a statement we can
// make at all.
TEST(metal_compute_residency, budget_reports_this_process)
{
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const auto mb = mc->memory_budget();
  EXPECT_TRUE(mb.total_physical >= (4ull << 30));
  EXPECT_TRUE(mb.self_footprint > 0);
  EXPECT_TRUE(mb.self_footprint <= mb.total_physical);
  // Idle memory is a subset of what the cache-inclusive figure reports.
  EXPECT_TRUE(mb.free_physical <= mb.available_physical);
}

// ---------------------------------------------------------------------
// Does a block read 100% in RAM the moment it is kept?
//
// The residency policy sheds a block whenever ANY of the resident set
// has left RAM, so the whole thing rests on a freshly-kept block
// measuring as fully resident. If it does not -- if a buffer reads
// partly out-of-core the instant it is written -- then the signal is
// not eviction at all, and the policy would shed on its own arrival and
// converge to keeping nothing. That is a silent failure: the run still
// produces correct output, just at streaming speed forever.
TEST(metal_compute_residency, fresh_buffer_is_fully_incore)
{
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  // Block-sized, so this asks the question at the shape that matters --
  // a few pages could hide behind any allocator rounding.
  SharedBuffer b = mc->make_shared_buffer(256ull << 20);
  if (b.empty()) { return; }
  std::memset(b.contents(), 0xA5, b.byte_size());
  const auto r = b.page_residency(64);
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.examined > 0);
  EXPECT_TRUE(r.fully_resident());
  EXPECT_TRUE(r.paged_out == 0);
}

// The same question one layer up, on the path a promoted block actually
// takes: WeightSet::stream_tensor(Copied) -- allocate, memcpy from the
// mapped shard, hand the bytes to the caller. Copied is the point;
// Mapped would leave the tensor on clean file-backed pages, which the
// OS reclaims freely and which would therefore read out-of-core under
// no memory pressure at all.
TEST(metal_compute_residency, streamed_copy_is_fully_incore)
{
  const char* dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (dir == nullptr || *dir == '\0') { return; }
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto ws = genai::WeightSet::open(dir, nullptr);
  if (!ws) { return; }
  // The largest tensor in the checkpoint: big enough that the sampled
  // walk examines a real number of pages.
  std::string best;
  std::size_t best_bytes = 0;
  for (const std::string& nm : ws->src().tensor_names()) {
    const auto* i = ws->src().info(nm);
    if (i == nullptr) { continue; }
    std::size_t n = 1;
    for (long d : i->shape) { n *= (std::size_t)(d > 0 ? d : 0); }
    if (n > best_bytes) { best_bytes = n; best = nm; }
  }
  if (best.empty()) { return; }
  SharedBuffer b =
      ws->stream_tensor(best, mc, genai::WeightSet::Residency::Copied);
  if (b.empty()) { return; }
  const auto r = b.page_residency(64);
  EXPECT_TRUE(r.valid);
  EXPECT_TRUE(r.examined > 0);
  EXPECT_TRUE(r.fully_resident());
}
