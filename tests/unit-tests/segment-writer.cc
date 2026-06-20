#include "minitest.h"
#include "common/segment-writer.h"

using namespace vpipe;

// SegmentWriter's heavy lifting needs a live FFmpeg context, which
// is out of scope for unit tests. The rollover predicate is the
// only piece of pure logic, so we exhaustively test it here.

TEST(segment_writer, decide_rollover_requires_keyframe)
{
  // Far past the target but not a keyframe -- must not roll.
  EXPECT_FALSE(decide_rollover(false,  /*elapsed*/ 60'000,
                               /*target*/ 30'000));
  // Keyframe but not yet at target -- must not roll.
  EXPECT_FALSE(decide_rollover(true, /*elapsed*/ 10'000,
                               /*target*/ 30'000));
}

TEST(segment_writer, decide_rollover_fires_at_or_past_target)
{
  // Exact equality counts as past the target.
  EXPECT_TRUE(decide_rollover(true, 30'000, 30'000));
  EXPECT_TRUE(decide_rollover(true, 30'001, 30'000));
  EXPECT_TRUE(decide_rollover(true, 9'999'999, 30'000));
}

TEST(segment_writer, decide_rollover_handles_zero_target)
{
  // Edge case: target_ms = 0 should roll on every keyframe (the
  // SegmentSpec ctor in the stage clamps zero away, but the pure
  // predicate must still behave sanely).
  EXPECT_TRUE(decide_rollover(true,  0, 0));
  EXPECT_FALSE(decide_rollover(false, 0, 0));
}

TEST(segment_writer, decide_rollover_negative_elapsed_is_pre_target)
{
  // steady_clock should never go backwards, but if a buggy caller
  // hands us a negative elapsed value we must not roll prematurely.
  EXPECT_FALSE(decide_rollover(true,  -5, 30'000));
}
