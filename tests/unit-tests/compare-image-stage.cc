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
  rt.launch();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
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
  rt.launch();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
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
  rt.launch();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
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
