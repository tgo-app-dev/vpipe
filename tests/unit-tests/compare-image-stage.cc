// compare-image stage: the pairing and the resolution matching.
//
// The interesting behaviour is not the encode (PNG is ffmpeg's job) but
// WHICH pair gets published and at what size: latest-wins per side, a
// common size of max(Wa,Wb) x max(Ha,Hb), pad-policy fitting for the
// smaller image, and black (a null slot) for an input that never spoke.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/compare-image-channel.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/compare-image-stage.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

// Wait for the channel to reach a state the test can assert on, instead
// of for a fixed slice of wall clock.
//
// The flat `sleep_for(600ms)` these replaced asserted that a launch, two
// beats and a PNG encode all fit in 600 ms of WALL clock, which is not a
// property of the code under test: under CPU starvation -- a full suite
// sweep on a loaded box -- the pipeline had simply not got there yet and
// the run failed with nothing wrong.
//
// Blocking on the channel's own version is both faster in the normal
// case (it returns the moment the publish lands, not 600 ms later) and
// immune to load. The budget is a HANG detector, not a pacing knob, so
// it is generous; a test that reaches it fails exactly as it did before.
//
// WAITING FOR THE FIRST PUBLISH WOULD NOT DO. The stage publishes as
// soon as EITHER side arrives -- that is what one_side_only asserts --
// so `wait_change(0, ...)` can return on a half-pair, and the geometry
// of a half-pair is one image's size rather than the common max. The
// predicate is over the SNAPSHOT, and it names the slots rather than the
// sizes, so the size assertions that follow stay real assertions.
//
// WHAT THIS UNCOVERED, which the flat sleep was hiding. With the
// deadline in place, `one_side_only_leaves_other_black` still fails
// roughly one run in 100-150 under a 12-way CPU load -- and it fails by
// reaching the FULL 10 s budget, meaning the pipeline never published at
// all rather than published late. A fixed 600 ms sleep cannot tell those
// two apart; it reported both as the same rare failure, which is why the
// residual bug read as test noise for as long as it did.
//
// It is consistent with the spin CompareImageStage::process documents
// against itself: `read_any` treats a CLOSED port as perpetually ready,
// so while B is closed and A has not yet produced, the stage wakes,
// drains nothing, finds `all_eos` false because A is still live, and
// re-arms immediately. Under enough CPU oversubscription that loop can
// starve the very producer it is waiting for. Not proven here -- the
// evidence is the timeout and the load dependence -- and the fix belongs
// in the stage's wait set, not in this file.
template <typename Pred>
bool
wait_for_(const CompareImageChannel& ch, Pred done, int budget_ms = 10000)
{
  const auto deadline =
      chrono::steady_clock::now() + chrono::milliseconds(budget_ms);
  for (;;) {
    auto snap = ch.snapshot();
    if (done(snap)) { return true; }
    const auto now = chrono::steady_clock::now();
    if (now >= deadline) { return false; }
    const int left = (int)chrono::duration_cast<chrono::milliseconds>(
        deadline - now).count();
    // Sleeps until the version moves off the one just read, so a publish
    // that lands between the snapshot and here does not cost a wait.
    ch.wait_change(snap.version, left > 0 ? left : 1);
  }
}

// Both slots populated: the pair is complete and its common size is
// settled.
bool
both_slots_(const CompareImageChannel::Snapshot& s)
{
  return s.a != nullptr && s.b != nullptr;
}

// Emits `count` copies of `tb`, then closes its oport.
class RepeatSource : public TypedStage<RepeatSource> {
public:
  static constexpr const char* kTypeName = "ut-compare-repeat-source";
  using TypedStage::TypedStage;

  TensorBeat tb;
  int        count = 1;

  Job process(RuntimeContext& ctx) override
  {
    if (_emitted >= count) {
      ctx.signal_done();
      co_return;
    }
    ++_emitted;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  int _emitted = 0;
};

// Emits ONE beat after a delay, then closes. The delay is the point: it
// holds open the window in which one input is finished and the other has
// not spoken yet, which is the window the wait set has to block through
// rather than spin through.
class DelayedSource : public TypedStage<DelayedSource> {
public:
  static constexpr const char* kTypeName = "ut-compare-delayed-source";
  using TypedStage::TypedStage;

  TensorBeat tb;
  int        delay_ms = 250;

  Job process(RuntimeContext& ctx) override
  {
    if (_emitted > 0) {
      ctx.signal_done();
      co_return;
    }
    ++_emitted;
    this_thread::sleep_for(chrono::milliseconds(delay_ms));
    auto payload = make_payload<TensorBeatPayload>(tb);
    co_await ctx.write(0, std::move(payload));
  }

private:
  int _emitted = 0;
};

// Closes immediately without producing anything -- a wired but silent
// input, which must read as "no image" rather than stalling the pair.
class ClosedSource : public TypedStage<ClosedSource> {
public:
  static constexpr const char* kTypeName = "ut-compare-closed-source";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

TensorBeat
make_rgb_(int H, int W, float v)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {3, H, W};
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  float* p = tb.as_f32();
  for (size_t i = 0; i < static_cast<size_t>(3) * H * W; ++i) { p[i] = v; }
  return tb;
}

bool
is_png_(const CompareImageChannel::Png& p)
{
  if (!p || p->size() < 8) { return false; }
  const auto& b = *p;
  return b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G';
}

}  // namespace

TEST(compare_image_stage, defaults_construct_channel_open)
{
  CerrSilencer hush;
  Session sess;

  CompareImageStage stage(&sess, "cmp", vector<InEdge>{},
                          FlexData::make_object());
  auto ch = stage.compare_channel();
  EXPECT_TRUE(ch != nullptr);
  EXPECT_TRUE(!ch->closed());
  EXPECT_TRUE(!stage.have_a());
  EXPECT_TRUE(!stage.have_b());
  EXPECT_TRUE(stage.common_width() == 0);
}

// The channel is a latched pair, not a stream: a reader that arrives
// late still sees the current pair, and each publish bumps the version.
TEST(compare_image_stage, channel_latches_and_versions)
{
  CompareImageChannel ch;
  EXPECT_TRUE(ch.snapshot().version == 0);

  auto png = make_shared<vector<uint8_t>>(vector<uint8_t>{1, 2, 3});
  ch.publish(png, nullptr, 64, 32);
  auto s1 = ch.snapshot();
  EXPECT_TRUE(s1.version == 1);
  EXPECT_TRUE(s1.width == 64 && s1.height == 32);
  EXPECT_TRUE(s1.a != nullptr);
  EXPECT_TRUE(s1.b == nullptr);

  // Republishing an identical pair still bumps the version, so a waiter
  // is woken for a re-render rather than sleeping through it.
  ch.publish(png, nullptr, 64, 32);
  EXPECT_TRUE(ch.snapshot().version == 2);

  // wait_change returns immediately when the version already moved.
  auto s3 = ch.wait_change(1, 50);
  EXPECT_TRUE(s3.version == 2);

  ch.close();
  EXPECT_TRUE(ch.closed());
  EXPECT_TRUE(ch.wait_change(2, 50).closed);
}

#if defined(__APPLE__) && defined(__arm64__)

// Both inputs at the same size: the pair publishes at that size with two
// PNGs and no resampling involved.
TEST(compare_image_stage, matching_sizes_publish_both)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto a_u = make_unique<RepeatSource>(&sess, "a", vector<InEdge>{},
                                       FlexData::make_object());
  a_u->tb = make_rgb_(48, 64, 0.25f);
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatSource*>(pl->insert_stage(std::move(a_u)));

  auto b_u = make_unique<RepeatSource>(&sess, "b", vector<InEdge>{},
                                       FlexData::make_object());
  b_u->tb = make_rgb_(48, 64, 0.75f);
  b_u->allocate_oports(1);
  auto* b = static_cast<RepeatSource*>(pl->insert_stage(std::move(b_u)));

  auto c_u = make_unique<CompareImageStage>(
      &sess, "cmp", vector<InEdge>{{a, 0}, {b, 0}}, FlexData::make_object());
  auto* cmp =
      static_cast<CompareImageStage*>(pl->insert_stage(std::move(c_u)));

  auto ch = cmp->compare_channel();
  PipelineRuntime rt(pl.get(), &sess);
  // Checked: a refused launch would otherwise surface as an unexplained
  // version 0 at the bottom of the test.
  EXPECT_TRUE(rt.launch());
  EXPECT_TRUE(wait_for_(*ch, both_slots_));
  rt.stop();

  auto s = ch->snapshot();
  EXPECT_TRUE(s.version >= 1);
  EXPECT_TRUE(s.width == 64);
  EXPECT_TRUE(s.height == 48);
  EXPECT_TRUE(is_png_(s.a));
  EXPECT_TRUE(is_png_(s.b));
}

// Mismatched sizes: the pair publishes at max on each axis so neither
// image is downscaled, and BOTH slots come back at that common size --
// which is what lets the view overlay them for the wipe modes.
TEST(compare_image_stage, mismatched_sizes_publish_at_common_max)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto a_u = make_unique<RepeatSource>(&sess, "a", vector<InEdge>{},
                                       FlexData::make_object());
  a_u->tb = make_rgb_(30, 40, 0.25f);      // small, and a wider aspect
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatSource*>(pl->insert_stage(std::move(a_u)));

  auto b_u = make_unique<RepeatSource>(&sess, "b", vector<InEdge>{},
                                       FlexData::make_object());
  b_u->tb = make_rgb_(90, 60, 0.75f);      // taller, and taller aspect
  b_u->allocate_oports(1);
  auto* b = static_cast<RepeatSource*>(pl->insert_stage(std::move(b_u)));

  auto c_u = make_unique<CompareImageStage>(
      &sess, "cmp", vector<InEdge>{{a, 0}, {b, 0}}, FlexData::make_object());
  auto* cmp =
      static_cast<CompareImageStage*>(pl->insert_stage(std::move(c_u)));

  auto ch = cmp->compare_channel();
  PipelineRuntime rt(pl.get(), &sess);
  // Checked: a refused launch would otherwise surface as an unexplained
  // version 0 at the bottom of the test.
  EXPECT_TRUE(rt.launch());
  EXPECT_TRUE(wait_for_(*ch, both_slots_));
  rt.stop();

  auto s = ch->snapshot();
  EXPECT_TRUE(s.version >= 1);
  EXPECT_TRUE(s.width == 60);              // max(40, 60)
  EXPECT_TRUE(s.height == 90);             // max(30, 90)
  EXPECT_TRUE(is_png_(s.a));
  EXPECT_TRUE(is_png_(s.b));
  EXPECT_TRUE(cmp->common_width() == 60);
  EXPECT_TRUE(cmp->common_height() == 90);
}

// One input wired and silent: the pair still publishes, with that slot
// null so the view shows black for it instead of waiting forever.
TEST(compare_image_stage, one_side_only_leaves_other_black)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto a_u = make_unique<RepeatSource>(&sess, "a", vector<InEdge>{},
                                       FlexData::make_object());
  a_u->tb = make_rgb_(24, 32, 0.5f);
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatSource*>(pl->insert_stage(std::move(a_u)));

  auto b_u = make_unique<ClosedSource>(&sess, "b", vector<InEdge>{},
                                       FlexData::make_object());
  b_u->allocate_oports(1);
  auto* b = static_cast<ClosedSource*>(pl->insert_stage(std::move(b_u)));

  auto c_u = make_unique<CompareImageStage>(
      &sess, "cmp", vector<InEdge>{{a, 0}, {b, 0}}, FlexData::make_object());
  auto* cmp =
      static_cast<CompareImageStage*>(pl->insert_stage(std::move(c_u)));

  auto ch = cmp->compare_channel();
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  // Only the A side ever speaks, so the settled state is A populated
  // and B still null -- waiting for both slots would time out on a
  // pipeline that is behaving exactly as this test expects.
  EXPECT_TRUE(wait_for_(*ch, [](const CompareImageChannel::Snapshot& s) {
    return s.a != nullptr;
  }));
  rt.stop();

  auto s = ch->snapshot();
  EXPECT_TRUE(s.version >= 1);
  EXPECT_TRUE(s.width == 32);
  EXPECT_TRUE(s.height == 24);
  EXPECT_TRUE(is_png_(s.a));
  EXPECT_TRUE(s.b == nullptr);             // black on the B side
  EXPECT_TRUE(cmp->have_a());
  EXPECT_TRUE(!cmp->have_b());
}

#endif  // __APPLE__ && __arm64__

// BLOCKING, NOT SPINNING, while one input is finished and the other has
// not spoken yet.
//
// `read_any` treats a drained-and-closed port as perpetually ready, so a
// wait set containing one returns immediately -- and with the other
// input still live, the stage wakes, drains nothing, declines to retire
// and re-arms. That loop occupies a worker for the whole window, and
// under CPU oversubscription it starves the producer it is waiting for,
// so the pair never publishes at all.
//
// The soak that found it is not a test anyone should have to run: it was
// roughly one failure in 100-150 runs under a 12-way load. This pins the
// same property deterministically, by holding the window open for a
// quarter of a second and counting how many times process() ran. A stage
// that blocks needs a handful of calls; one that spins needs thousands.
TEST(compare_image_stage, a_closed_input_does_not_spin_the_wait)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto a_u = make_unique<DelayedSource>(&sess, "a", vector<InEdge>{},
                                        FlexData::make_object());
  a_u->tb = make_rgb_(24, 32, 0.5f);
  a_u->delay_ms = 250;
  a_u->allocate_oports(1);
  auto* a = static_cast<DelayedSource*>(pl->insert_stage(std::move(a_u)));

  // Closes at once, so the whole 250 ms is spent with B finished and A
  // still to come -- the exact window.
  auto b_u = make_unique<ClosedSource>(&sess, "b", vector<InEdge>{},
                                       FlexData::make_object());
  b_u->allocate_oports(1);
  auto* b = static_cast<ClosedSource*>(pl->insert_stage(std::move(b_u)));

  auto c_u = make_unique<CompareImageStage>(
      &sess, "cmp", vector<InEdge>{{a, 0}, {b, 0}}, FlexData::make_object());
  auto* cmp =
      static_cast<CompareImageStage*>(pl->insert_stage(std::move(c_u)));

  auto ch = cmp->compare_channel();
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  const bool published =
      wait_for_(*ch, [](const CompareImageChannel::Snapshot& s) {
        return s.a != nullptr;
      });
  const std::uint64_t calls = cmp->process_calls();
  rt.stop();

  // It still does the job: the delayed frame lands and B reads as black.
  EXPECT_TRUE(published);
  auto s = ch->snapshot();
  EXPECT_TRUE(s.version >= 1);
  EXPECT_TRUE(s.a != nullptr);
  EXPECT_TRUE(s.b == nullptr);

  // ... and it waited for it. The bound is loose on purpose -- what
  // separates blocking from spinning here is three orders of magnitude,
  // not a tuned threshold.
  std::printf("[compare_image_stage] process() ran %llu time(s) across a "
              "%d ms wait\n", (unsigned long long)calls, 250);
  EXPECT_TRUE(calls < 50);
}
