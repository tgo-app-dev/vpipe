// block-residency.cc -- the grow-into-free-RAM policy behind every
// streamed DiT (generative-models/shared/block-residency.h).
//
// This policy decides how much of a checkpoint to keep resident, and it
// gets to decide it on a box whose memory it does not own. It shipped
// untested, and the failure that followed is the one this file exists to
// pin down: on a 64 GB Mac running an unquantized MiniMax-H3, growth ran
// all the way to 50 of 50 blocks (61566 MB) and the machine ended at
// 33 GB compressed with 28.7 GB of swap -- SLOWER than a 24 GB box that
// streamed, because the weights it insisted on holding could only leave
// through the compressor.
//
// It grew that far because it asked free-memory arithmetic:
// `available_physical` counts file-backed pages, and on a streaming
// model the file cache is mostly the checkpoint that model just re-read,
// so the more it streamed the more room it believed it had. At the
// moment of the report it read ~18.5 GB available, 18.54 GB of it cache.
//
// Tightening that arithmetic is not the fix and these tests are written
// to say so. It cannot tell a cache about to be dropped from one in use,
// and spending only idle pages breaks the healthy case instead, since
// macOS keeps `free` small on purpose. So the limit is MEASURED: the
// blocks that were kept are checked for still being in RAM, and a pin
// that failed costs a block. See weights_leaving_ram_is_what_stops_growth,
// which is the one that matters.
//
// Everything here runs on budget SNAPSHOTS, so the shapes that matter
// (a full 64 GB box deep into swap) can be exercised on a machine that
// is not one, and none of it needs a GPU.

#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/shared/block-residency.h"

#include <cstddef>
#include <string>
#include <vector>

using namespace vpipe;
using vpipe::genai::BlockResidency;
using Budget = metal_compute::MetalCompute::MemoryBudget;

namespace {

constexpr std::size_t kGB = 1ull << 30;

// A box with `free` genuinely idle, `cache` in file-backed pages, and
// `comp`/`swap` already paged out. `recommended` is only a validity flag
// for the policy, so any non-zero will do.
Budget box_(std::size_t total, std::size_t free, std::size_t cache,
            std::size_t comp = 0, std::size_t swap = 0)
{
  Budget b;
  b.recommended        = total;
  b.total_physical     = total;
  b.free_physical      = free;
  b.available_physical = free + cache;
  b.compressed         = comp;
  b.swap_used          = swap;
  return b;
}

// Grow until the policy says stop, and report how much was admitted.
std::size_t grow_(BlockResidency& r, const Budget& b, std::size_t block,
                  int limit)
{
  for (int i = 0; i < limit; ++i) {
    if (!r.admit(b, block)) { break; }
    r.note_admitted(block);
  }
  return r.bytes();
}

}  // namespace

// The reported box, once it is already paging. Free-memory arithmetic
// still reads ~18.5 GB available -- all of it file cache -- and would
// admit; the paging check is what refuses.
TEST(block_residency, a_paging_box_does_not_grow)
{
  const Budget b = box_(64 * kGB, /*free=*/0, /*cache=*/18 * kGB + kGB / 2,
                        /*comp=*/33 * kGB, /*swap=*/28 * kGB);
  // What the arithmetic says, and still says for the transient callers
  // (a VAE decode preflight) that this figure exists for.
  EXPECT_TRUE(b.fits_physical(6 * kGB));
  EXPECT_TRUE(b.paging());
  EXPECT_TRUE(!b.fits_growth(6 * kGB));

  BlockResidency r;
  r.set_reserve(3 * kGB);
  EXPECT_TRUE(grow_(r, b, 1231ull << 20, 50) == 0);
}

// THE test: the box looks fine by every arithmetic measure -- 18.5 GB
// "available", nothing compressed yet -- so growth proceeds, and the
// only thing that reveals the truth is that the blocks already kept are
// no longer all in RAM. One block goes back, and the ceiling ratchets so
// it cannot climb back into the same wall.
TEST(block_residency, weights_leaving_ram_is_what_stops_growth)
{
  const std::size_t block = 1231ull << 20;
  const Budget healthy = box_(64 * kGB, /*free=*/2 * kGB,
                              /*cache=*/18 * kGB);
  EXPECT_TRUE(healthy.fits_growth(6 * kGB));   // arithmetic sees no problem

  BlockResidency r;
  r.set_reserve(3 * kGB);
  r.set_per_forward_cap(50);
  grow_(r, healthy, block, 20);
  EXPECT_TRUE(r.count() == 20);
  EXPECT_TRUE(!r.weights_were_evicted());

  // 4 of 4000 sampled pages of the RESIDENT SET are no longer in RAM.
  // A shortfall that small is still a failed pin.
  int evictions = 0;
  const std::size_t freed = r.note_weight_residency(
      4000, 3996, [&]() -> std::size_t { ++evictions; return block; });
  EXPECT_TRUE(freed == block);
  EXPECT_TRUE(evictions == 1);
  EXPECT_TRUE(r.count() == 19);
  EXPECT_TRUE(r.weights_were_evicted());

  // And the ratchet holds on a later forward that still looks healthy.
  r.begin_forward();
  EXPECT_TRUE(grow_(r, healthy, block, 50) == 19 * block);
}

// One block per report, not "shed down to the measurement". A shortfall
// says a pin failed, not how much is too much, and giving back the whole
// set the moment one page moved would undo the entire point.
TEST(block_residency, a_shortfall_gives_back_one_block_at_a_time)
{
  const std::size_t block = kGB;
  BlockResidency r;
  r.set_reserve(kGB);
  r.set_per_forward_cap(50);
  grow_(r, box_(64 * kGB, 40 * kGB, 0), block, 30);
  EXPECT_TRUE(r.count() == 30);
  // Half the resident set is out of RAM -- still one block.
  r.note_weight_residency(1000, 500,
                          [&]() -> std::size_t { return block; });
  EXPECT_TRUE(r.count() == 29);
}

// Fully resident is not a shortfall, and must not cost a block.
TEST(block_residency, a_fully_resident_set_is_left_alone)
{
  BlockResidency r;
  r.set_reserve(kGB);
  r.set_per_forward_cap(50);
  grow_(r, box_(64 * kGB, 40 * kGB, 0), kGB, 10);
  int evictions = 0;
  const std::size_t freed = r.note_weight_residency(
      5000, 5000, [&]() -> std::size_t { ++evictions; return kGB; });
  EXPECT_TRUE(freed == 0);
  EXPECT_TRUE(evictions == 0);
  EXPECT_TRUE(r.count() == 10);
}

// Growth is capped per forward so the measurement gets a chance to speak
// before the whole checkpoint has been admitted. Uncapped, the first
// forward takes everything and the first evidence arrives too late.
TEST(block_residency, growth_is_bounded_per_forward)
{
  BlockResidency r;
  r.set_reserve(kGB);
  r.set_per_forward_cap(8);
  const Budget roomy = box_(64 * kGB, 40 * kGB, 0);
  grow_(r, roomy, kGB, 50);
  EXPECT_TRUE(r.count() == 8);
  r.begin_forward();
  grow_(r, roomy, kGB, 50);
  EXPECT_TRUE(r.count() == 16);
}

// The same box before it went under: plenty genuinely free, nothing
// paged. Growth is the whole point, so it has to still happen.
TEST(block_residency, idle_memory_is_room_to_grow)
{
  const Budget b = box_(64 * kGB, /*free=*/50 * kGB, /*cache=*/4 * kGB);
  EXPECT_TRUE(b.fits_growth(6 * kGB));
  EXPECT_TRUE(!b.paging());

  BlockResidency r;
  r.set_reserve(3 * kGB);
  r.set_per_forward_cap(50);
  const std::size_t block = 1231ull << 20;
  const std::size_t got = grow_(r, b, block, 50);
  EXPECT_TRUE(got == 50 * block);      // a static budget admits the lot
}

// Growth still stops where the arithmetic runs out -- that gate did not
// go away, it merely stopped being the only one. Driven by shrinking the
// budget as blocks are taken, which is what a real box does.
TEST(block_residency, growth_stops_where_available_memory_does)
{
  const std::size_t block = 2 * kGB;
  BlockResidency r;
  r.set_reserve(3 * kGB);
  r.set_per_forward_cap(50);
  std::size_t avail = 12 * kGB;
  int admitted = 0;
  for (int i = 0; i < 50; ++i) {
    const Budget b = box_(64 * kGB, kGB, avail - kGB);
    if (!r.admit(b, block)) { break; }
    r.note_admitted(block);
    avail -= block;
    ++admitted;
  }
  // It stops, with room left over rather than in swap.
  EXPECT_TRUE(admitted > 0);
  EXPECT_TRUE(admitted <= 4);
  EXPECT_TRUE(avail >= 3 * kGB);
}

// The feedback loop that did not exist: once the OS is paging, a forward
// gives blocks back. Nothing inside the process asks for this -- the
// swapping is the only notification there is.
TEST(block_residency, a_paging_box_takes_blocks_back)
{
  BlockResidency r;
  r.set_reserve(3 * kGB);
  r.set_per_forward_cap(50);
  const std::size_t block = 2 * kGB;
  grow_(r, box_(64 * kGB, 50 * kGB, 4 * kGB), block, 20);
  const int grown = r.count();
  EXPECT_TRUE(grown == 20);

  // Now the box is squeezed: 20 GB compressed on a 64 GB machine is
  // past the quarter line, so the next forward sheds -- ONE block, since
  // a system-wide figure cannot say how much of it is ours.
  int evictions = 0;
  std::size_t held = r.bytes();
  r.begin_forward(box_(64 * kGB, 0, 2 * kGB, 20 * kGB, kGB),
                  [&]() -> std::size_t {
                    if (held == 0) { return 0; }
                    held -= block;
                    ++evictions;
                    return block;
                  });
  EXPECT_TRUE(evictions == 1);
  EXPECT_TRUE(r.count() == grown - 1);
  EXPECT_TRUE(r.paged());
}

// And having shed, it must not climb straight back the moment the
// compressor drains -- that is the ratchet, and without it the box
// oscillates between thrashing and recovering.
TEST(block_residency, shedding_ratchets_the_ceiling)
{
  BlockResidency r;
  r.set_reserve(3 * kGB);
  r.set_per_forward_cap(50);
  const std::size_t block = 2 * kGB;
  const Budget roomy = box_(64 * kGB, 50 * kGB, 4 * kGB);
  grow_(r, roomy, block, 20);

  std::size_t held = r.bytes();
  r.begin_forward(box_(64 * kGB, 0, 2 * kGB, 20 * kGB, kGB),
                  [&]() -> std::size_t {
                    if (held == 0) { return 0; }
                    held -= block;
                    return block;
                  });
  const int after_shed = r.count();

  // A forward on a box that once again looks empty.
  r.begin_forward();
  const std::size_t regrown = grow_(r, roomy, block, 20);
  EXPECT_TRUE(r.count() == after_shed);
  EXPECT_TRUE(regrown == (std::size_t)after_shed * block);
}

// A model that never declared what its activations cost keeps the old
// pure-streaming behaviour rather than growing against a guess.
TEST(block_residency, no_reserve_means_no_growth)
{
  BlockResidency r;                    // reserve left at 0
  EXPECT_TRUE(!r.admit(box_(64 * kGB, 50 * kGB, 0), kGB));
  EXPECT_TRUE(r.count() == 0);
}

// ...but a reserve explicitly set to zero is a caller saying "nothing
// runs after me that I do not free first", which is not the same as one
// that never answered. FLUX.2 says it on the path where the DiT is
// dropped before the VAE decode. Growth then rides admit()'s own
// hysteresis instead of a figure that protects nobody.
TEST(block_residency, an_explicit_zero_reserve_still_grows)
{
  BlockResidency r;
  r.set_reserve(0);
  EXPECT_TRUE(r.admit(box_(64 * kGB, 50 * kGB, 0), kGB));
  r.note_admitted(kGB);
  EXPECT_TRUE(r.count() == 1);
  // The hysteresis gap is still the brake: a block either side of the
  // one asked for, so a box with no room refuses even at zero reserve.
  BlockResidency t;
  t.set_reserve(0);
  EXPECT_TRUE(!t.admit(box_(64 * kGB, /*free=*/kGB / 2, /*cache=*/0), kGB));
}

// Swap in use is enough on its own: a box can be swapping without the
// compressor being over the line, and growing then is what turned a
// 64 GB machine into a slower one than a 24 GB machine.
TEST(block_residency, swap_alone_stops_growth)
{
  const Budget b = box_(64 * kGB, /*free=*/20 * kGB, /*cache=*/4 * kGB,
                        /*comp=*/kGB, /*swap=*/12 * kGB);
  EXPECT_TRUE(b.paging());
  EXPECT_TRUE(!b.fits_growth(kGB));
  BlockResidency r;
  r.set_reserve(kGB);
  EXPECT_TRUE(!r.admit(b, kGB));
}

// The query being unavailable must not become a licence to grow without
// limit, nor a refusal to grow at all: with no numbers at all the policy
// keeps its previous vacuous-true behaviour.
TEST(block_residency, an_unavailable_query_is_not_a_signal)
{
  Budget b;
  b.recommended = 64 * kGB;            // everything else left at 0
  EXPECT_TRUE(b.fits_growth(kGB));
  EXPECT_TRUE(!b.paging());
}

// The page walk is expensive -- MEASURED at 57 ms per 4.3 GB examined,
// so ~800 ms of every step on a 62 GB resident set. It is gated on this:
// a process whose own compressed footprint has not moved has nothing for
// the walk to find.
TEST(block_residency, the_page_walk_is_gated_on_a_cheap_signal)
{
  BlockResidency r;
  // First call establishes the baseline rather than firing on it.
  EXPECT_TRUE(!r.self_compression_grew(0));
  EXPECT_TRUE(!r.self_compression_grew(0));
  // Ordinary small movement is noise, not a signal.
  EXPECT_TRUE(!r.self_compression_grew(32ull << 20));
  // A real jump is.
  EXPECT_TRUE(r.self_compression_grew(kGB));
  // Holding steady at the new level is not a fresh signal.
  EXPECT_TRUE(!r.self_compression_grew(kGB));
  // Nor is it going back down.
  EXPECT_TRUE(!r.self_compression_grew(0));
}

// A healthy machine uses the compressor constantly, and the coarse
// system-wide check must not read that as distress. MEASURED on an idle
// 16 GB M5: 2.36 GB compressed, 0.97 GB swap, nothing running. A gate
// that fired there would refuse to grow on every box in the fleet.
TEST(block_residency, an_ordinary_busy_box_is_not_distress)
{
  const std::size_t k16 = 16 * kGB;
  const Budget idle = box_(k16, /*free=*/kGB / 8, /*cache=*/2 * kGB,
                           /*comp=*/2360ull << 20, /*swap=*/970ull << 20);
  EXPECT_TRUE(!idle.paging());
  EXPECT_TRUE(idle.fits_growth(kGB / 2));

  // The reported 64 GB box, by contrast, is unambiguous.
  const Budget bad = box_(64 * kGB, 0, 18 * kGB, 33 * kGB, 28 * kGB);
  EXPECT_TRUE(bad.paging());
}

// The ratchet must not be a one-way door.
//
// This is the M5 16 GB MiniMax-H3 run reproduced in miniature: admit one
// block, measure it out of RAM on the next forward, shed it. That takes
// the resident set to zero, which used to take the CEILING to zero --
// and `_bytes + block > 0` is true for every block, so growth was dead
// for the remaining 29 steps of a 30-step schedule while RAM sat half
// free. One sample, taken during the AdaLN bake (the tightest instant of
// the run, immediately before it hands back 13 GB), decided everything.
TEST(block_residency, shed_to_zero_does_not_latch_growth_off)
{
  BlockResidency r;
  r.set_reserve(1ull << 30);
  const std::size_t blk = 206ull << 20;
  const Budget healthy = box_(64 * kGB, /*free=*/32 * kGB, /*cache=*/8 * kGB);

  r.begin_forward();
  EXPECT_TRUE(r.admit(healthy, blk));
  r.note_admitted(blk);
  EXPECT_TRUE(r.count() == 1);

  // The measurement: none of it is in RAM any more.
  int evicted = 0;
  const std::size_t freed = r.note_weight_residency(
      1000, 0, [&]() -> std::size_t { ++evicted; return blk; });
  EXPECT_TRUE(freed == blk);
  EXPECT_TRUE(evicted == 1);
  EXPECT_TRUE(r.count() == 0);

  // The ceiling is floored at ONE block, never zero -- so the next
  // forward may try again. Zero is unrecoverable: `_bytes + block > 0`
  // is true for every block, and an increase one block wide cannot lift
  // a ceiling of zero off the floor either. That is the bug this test
  // exists for -- 29 of a 30-step schedule streamed with RAM half free
  // because one sample landed during the AdaLN bake.
  r.begin_forward();
  EXPECT_TRUE(r.admit(healthy, blk));
  r.note_admitted(blk);
  EXPECT_TRUE(r.count() == 1);
}

// The ratchet is still a real brake: what it forbids is climbing back to
// the level that just failed, not holding anything at all.
TEST(block_residency, the_ratchet_still_forbids_the_failed_level)
{
  BlockResidency r;
  r.set_reserve(1ull << 30);
  const std::size_t blk = 64ull << 20;
  const Budget healthy = box_(64 * kGB, /*free=*/32 * kGB, /*cache=*/8 * kGB);

  r.begin_forward();
  grow_(r, healthy, blk, 4);
  const int grown = r.count();
  EXPECT_TRUE(grown >= 3);

  r.note_weight_residency(1000, 900,
                          [&]() -> std::size_t { return blk; });
  EXPECT_TRUE(r.count() == grown - 1);

  // Back to `grown` is exactly what was just measured as too much.
  r.begin_forward();
  EXPECT_TRUE(!r.admit(healthy, blk));

  // Quiet forwards buy it back, one block per three.
  for (int i = 0; i < 3; ++i) {
    r.begin_forward();
    r.note_healthy_forward();
  }
  r.begin_forward();
  EXPECT_TRUE(r.admit(healthy, blk));
}

// Recovery is strictly slower than retreat, so a box that is genuinely
// full settles instead of oscillating every step: a shed inside the
// quiet run resets the count, and one healthy forward does not undo it.
TEST(block_residency, recovery_is_slower_than_retreat)
{
  BlockResidency r;
  r.set_reserve(1ull << 30);
  const std::size_t blk = 64ull << 20;
  const Budget healthy = box_(64 * kGB, /*free=*/32 * kGB, /*cache=*/8 * kGB);

  for (int i = 0; i < 4; ++i) {
    r.begin_forward();
    if (r.admit(healthy, blk)) { r.note_admitted(blk); }
  }
  const int grown = r.count();
  EXPECT_TRUE(grown >= 2);

  r.note_weight_residency(1000, 900,
                          [&]() -> std::size_t { return blk; });
  EXPECT_TRUE(r.count() == grown - 1);

  // One quiet forward is not enough to climb back.
  r.begin_forward();
  r.note_healthy_forward();
  r.begin_forward();
  EXPECT_TRUE(!r.admit(healthy, blk));

  // A shed part-way through the quiet run restarts the clock.
  r.begin_forward();
  r.note_healthy_forward();
  r.note_weight_residency(1000, 900,
                          [&]() -> std::size_t { return blk; });
  r.begin_forward();
  r.note_healthy_forward();
  r.begin_forward();
  EXPECT_TRUE(!r.admit(healthy, blk));
}

// A shed and a recovery in the SAME forward is not a quiet forward.
// Without this, begin_forward(paging) sheds a block and the healthy call
// at the bottom of the same step starts crediting it back.
TEST(block_residency, shed_forward_earns_no_credit)
{
  BlockResidency r;
  r.set_reserve(1ull << 30);
  const std::size_t blk = 32ull << 20;
  const Budget healthy = box_(64 * kGB, /*free=*/32 * kGB, /*cache=*/8 * kGB);

  r.begin_forward();
  grow_(r, healthy, blk, 4);
  const int grown = r.count();
  EXPECT_TRUE(grown >= 3);

  // Every forward sheds one and then reports itself healthy. The credit
  // must never accrue, so the ceiling never climbs back and each step
  // ends one block lower than it started.
  for (int i = 0; i < 3; ++i) {
    r.begin_forward();
    r.release(1, [&]() -> std::size_t { return blk; });
    r.note_healthy_forward();          // same forward: must not count
    EXPECT_TRUE(!r.admit(healthy, blk));
  }
}

// A model that KNOWS the ground moved gets to say so. MiniMax-H3's
// AdaLN bake retires 13.2 GB of per-step projections partway through
// the first step, which is exactly when the run measured a shortfall
// and shed -- a fact about a box carrying 13 GB it was about to stop
// carrying. Decay alone cannot undo that inside a 6-step schedule.
TEST(block_residency, a_landscape_change_clears_the_ratchet)
{
  BlockResidency r;
  r.set_reserve(1ull << 30);
  const std::size_t blk = 64ull << 20;
  const Budget healthy = box_(64 * kGB, /*free=*/32 * kGB, /*cache=*/8 * kGB);

  r.begin_forward();
  grow_(r, healthy, blk, 4);
  const int grown = r.count();
  r.note_weight_residency(1000, 900,
                          [&]() -> std::size_t { return blk; });
  r.begin_forward();
  EXPECT_TRUE(!r.admit(healthy, blk));   // ratcheted

  r.note_landscape_changed();
  EXPECT_TRUE(r.admit(healthy, blk));
  r.note_admitted(blk);
  EXPECT_TRUE(r.count() == grown);
}
