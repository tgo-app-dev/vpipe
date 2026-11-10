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
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/compare-image-stage.h"

#include <malloc/malloc.h>

#include <atomic>
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

// RGBA, with a per-pixel alpha ramp so the checkerboard shows through
// at some pixels and not others.
TensorBeat
make_rgba_(int H, int W, float v, float a)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {4, H, W};
  tb.resize_contiguous(static_cast<size_t>(4) * H * W);
  float* p = tb.as_f32();
  const size_t plane = static_cast<size_t>(H) * W;
  for (size_t i = 0; i < 3 * plane; ++i) { p[i] = v; }
  for (size_t i = 0; i < plane; ++i) { p[3 * plane + i] = a; }
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

// ---- RGBA ------------------------------------------------------------
//
// A four-channel beat is accepted, and it is shown OVER A CHECKERBOARD
// rather than flattened onto a colour. That is not decoration: this is
// a comparison view, and on flat white two images that differ ONLY in
// their alpha would be published as identical -- the one thing a
// comparison must never do.
//
// Asserted through the published PNGs rather than by reaching into the
// stage: two images with the same RGB and DIFFERENT alpha must encode
// differently.
TEST(compare_image_stage, rgba_is_accepted_and_alpha_is_visible)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  // Same colour on both sides; only the alpha differs.
  auto a_u = make_unique<RepeatSource>(&sess, "a", vector<InEdge>{},
                                       FlexData::make_object());
  a_u->tb = make_rgba_(48, 64, 0.25f, 1.0f);    // opaque
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatSource*>(pl->insert_stage(std::move(a_u)));

  auto b_u = make_unique<RepeatSource>(&sess, "b", vector<InEdge>{},
                                       FlexData::make_object());
  b_u->tb = make_rgba_(48, 64, 0.25f, 0.0f);    // fully transparent
  b_u->allocate_oports(1);
  auto* b = static_cast<RepeatSource*>(pl->insert_stage(std::move(b_u)));

  auto c_u = make_unique<CompareImageStage>(
      &sess, "cmp", vector<InEdge>{{a, 0}, {b, 0}}, FlexData::make_object());
  auto* cmp =
      static_cast<CompareImageStage*>(pl->insert_stage(std::move(c_u)));

  auto ch = cmp->compare_channel();
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  EXPECT_TRUE(wait_for_(*ch, both_slots_));
  rt.stop();

  auto s = ch->snapshot();
  EXPECT_TRUE(s.version >= 1);
  EXPECT_TRUE(s.width == 64 && s.height == 48);
  // Accepted at all -- a four-channel beat used to be dropped by
  // unpack_ and the slot would simply never fill.
  EXPECT_TRUE(is_png_(s.a));
  EXPECT_TRUE(is_png_(s.b));
  if (!is_png_(s.a) || !is_png_(s.b)) { return; }
  // AND DISTINGUISHABLE. Same RGB, different alpha: over a flat colour
  // these would be one flat field each and could easily encode to the
  // same bytes; over the checkerboard the transparent one carries the
  // pattern and the opaque one does not.
  EXPECT_TRUE(*s.a != *s.b);
}

// NOTHING IS ENCODED UNTIL A READER ASKS. A publish owes the pair's
// PNGs; the first snapshot renders each side once, at the common size;
// a republish keeps a side whose picture and common size are the last
// publish's, and owes afresh a new picture or a new size. Before, the
// stage encoded both sides at full size on the pipeline's thread at
// every publish -- and again for the side that had not changed --
// whether or not any view was watching, and kept the encoder, its frame
// and its packet for the run (250 MiB at 42 MP).
TEST(compare_image_stage, a_pair_is_encoded_when_a_reader_asks)
{
  CompareImageChannel ch;
  int renders = 0;
  auto render = make_shared<const CompareImageChannel::Render>(
      [&renders](const CompareImageChannel::Picture& p, int w, int h) {
        ++renders;
        return make_shared<const vector<uint8_t>>(vector<uint8_t>{
            (uint8_t)p.width, (uint8_t)w, (uint8_t)h});
      });
  auto pic = [](int w, int h) {
    return CompareImageChannel::Picture{
        make_shared<const vector<uint8_t>>((size_t)w * h * 3, 7), w, h};
  };
  const auto a = pic(8, 6);
  ch.publish_pictures(a, pic(4, 3), 8, 6, render);
  EXPECT_TRUE(renders == 0);
  auto s = ch.snapshot();
  EXPECT_TRUE(renders == 2 && s.version == 1 && s.a && s.b);
  if (!s.a || !s.b) { return; }
  // Each side rendered at the COMMON size, from its own picture.
  EXPECT_TRUE((*s.a)[0] == 8 && (*s.b)[0] == 4 && (*s.b)[1] == 8 &&
              (*s.b)[2] == 6);
  ch.snapshot();
  EXPECT_TRUE(renders == 2);                   // once, not per reader
  // B changes, A does not: A keeps its PNG.
  ch.publish_pictures(a, pic(4, 3), 8, 6, render);
  s = ch.snapshot();
  EXPECT_TRUE(renders == 3 && s.version == 2 && s.a && s.b);
  // A new common size owes both.
  const auto a2 = pic(10, 6);
  const auto b2 = pic(4, 3);
  ch.publish_pictures(a2, b2, 10, 6, render);
  s = ch.snapshot();
  EXPECT_TRUE(renders == 5 && s.version == 3);
  // wait_change renders what it returns, as snapshot does.
  ch.publish_pictures(a2, pic(4, 3), 10, 6, render);
  s = ch.wait_change(3, 1000);
  EXPECT_TRUE(renders == 6 && s.version == 4 && s.b);
}

namespace {

// u8 noise: a picture PNG cannot compress, so a render's packet and PNG
// are as large as they can be.
TensorBeat
make_noise_(int H, int W, uint32_t seed)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, H, W};
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  uint8_t* p = tb.as_u8();
  uint32_t x = seed;
  for (size_t i = 0; i < static_cast<size_t>(3) * H * W; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p[i] = (uint8_t)x;
  }
  return tb;
}

size_t
heap_in_use_()
{
  malloc_statistics_t st{};
  malloc_zone_statistics(nullptr, &st);
  return st.size_in_use;
}

}  // namespace

// THE PLAN HOLDS WHAT THE STAGE AND A VIEWER DO. The pair's pictures
// (packed RGB24, each at its own size) are booked before they are
// unpacked, and a viewer's render besides, at pair_bytes -- since no
// signal says whether one will mount. A render of two incompressible
// pictures, one padded to the other's size, grows the heap by no more
// than that render share.
TEST(compare_image_stage, the_plan_books_the_pair_and_a_render)
{
  // A small pair and a large one: FFmpeg's own share grows with the
  // picture (0.5 MB at 1024 x 768, 3 MB at 4096 x 3072).
  const int geo[][4] = {{1024, 768, 512, 384}, {4096, 3072, 2048, 1536}};
  for (const auto& g : geo) {
  const int WA = g[0], HA = g[1], WB = g[2], HB = g[3];
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);
  auto a_u = make_unique<RepeatSource>(&sess, "a", vector<InEdge>{},
                                       FlexData::make_object());
  a_u->tb = make_noise_(HA, WA, 1);
  a_u->allocate_oports(1);
  auto* a = pl->insert_stage(std::move(a_u));
  auto b_u = make_unique<RepeatSource>(&sess, "b", vector<InEdge>{},
                                       FlexData::make_object());
  b_u->tb = make_noise_(HB, WB, 2);
  b_u->allocate_oports(1);
  auto* b = pl->insert_stage(std::move(b_u));
  auto* cmp = static_cast<CompareImageStage*>(
      pl->insert_stage(make_unique<CompareImageStage>(
          &sess, "cmp", vector<InEdge>{{a, 0}, {b, 0}},
          FlexData::make_object())));
  auto ch = cmp->compare_channel();
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();          // both published; nothing has asked to see them
  EXPECT_TRUE(cmp->have_a() && cmp->have_b());

  const size_t booked = sess.services()->generative_model_manager()
                            ->declared_scratch("compare-image:cmp");
  const size_t sides = (size_t)3 * ((size_t)WA * HA + (size_t)WB * HB);
  EXPECT_TRUE(booked ==
              CompareImageStage::pair_bytes(WA, HA, WB, HB, WA, HA));

  // A viewer asks: both sides render, B padded to A's size.
  const size_t base = heap_in_use_();
  atomic<bool> run{true};
  atomic<size_t> peak{base};
  thread sampler([&] {
    while (run.load()) {
      const size_t now = heap_in_use_();
      if (now > peak.load()) { peak.store(now); }
      this_thread::sleep_for(chrono::microseconds(20));
    }
  });
  const auto s = ch->snapshot();
  run.store(false);
  sampler.join();
  const size_t grew = peak.load() - base;
  printf("[compare_image_stage] booked %zu B (pictures %zu, render %zu); "
         "a render grew the heap %zu B, PNGs %zu + %zu B\n", booked, sides,
         booked - sides, grew, s.a ? s.a->size() : 0,
         s.b ? s.b->size() : 0);
  EXPECT_TRUE(is_png_(s.a) && is_png_(s.b));
  EXPECT_TRUE(grew >= (size_t)3 * WA * HA);        // the frame, at least
  EXPECT_TRUE(grew <= booked - sides);
  rt.stop();
  }
}

