// create-mask stage: the mask itself -- how it is unpacked, resampled,
// sized, painted onto a reference image, and handed back.
//
// The interesting behaviour is not the PNG (ffmpeg's job) but the mask
// SEMANTICS: that a two-state mask stays two-state across a resample, a
// class index survives as an index rather than being averaged into a
// neighbouring class, that the canvas geometry falls back the way it
// says it does, and that one commit is exactly one output beat.
//
// The commit round trip feeds the stage's OWN published mask PNG back
// in as a commit, which exercises the encode and the decode against
// each other -- the two halves that have to agree for the editor to
// mean anything.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "common/mask-editor-channel.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-command.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/create-mask-stage.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
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
  static constexpr const char* kTypeName = "ut-mask-repeat-source";
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

// Latches every beat it is handed, so the test can look at what the
// stage actually emitted after the runtime has stopped.
class Collector : public TypedStage<Collector> {
public:
  static constexpr const char* kTypeName = "ut-mask-collector";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    auto beat = co_await ctx.read(0);
    if (!beat) {
      ctx.signal_done();
      co_return;
    }
    if (const auto* tb =
            dynamic_cast<const TensorBeatPayload*>(beat.get())) {
      lock_guard<mutex> lk(_mu);
      _beats.push_back(*tb);
    }
  }

  size_t count() const
  {
    lock_guard<mutex> lk(_mu);
    return _beats.size();
  }

  TensorBeat last() const
  {
    lock_guard<mutex> lk(_mu);
    return _beats.empty() ? TensorBeat{} : _beats.back();
  }

private:
  mutable mutex      _mu;
  vector<TensorBeat> _beats;
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

// A u8 mask [1,H,W] whose LEFT half carries `v` and right half 0.
TensorBeat
make_mask_(int H, int W, uint8_t v)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {1, H, W};
  tb.resize_contiguous(static_cast<size_t>(H) * W);
  uint8_t* p = tb.as_u8();
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      p[static_cast<size_t>(y) * W + x] = (x < W / 2) ? v : 0;
    }
  }
  return tb;
}

FlexData
cfg_(std::initializer_list<pair<const char*, FlexData>> kv)
{
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  for (auto& [k, v] : kv) { oo.insert(k, v); }
  return o;
}

}  // namespace

TEST(create_mask_stage, defaults_construct_channel_open)
{
  CerrSilencer hush;
  Session sess;

  CreateMaskStage stage(&sess, "mk", vector<InEdge>{},
                        FlexData::make_object());
  auto ch = stage.mask_channel();
  EXPECT_TRUE(ch != nullptr);
  EXPECT_TRUE(!ch->closed());
  EXPECT_TRUE(stage.mode() == MaskEditorChannel::Mode::Binary);
  EXPECT_TRUE(!stage.have_ref());
  EXPECT_TRUE(stage.mask_width() == 0);
}

// A bad enum is a CONFIG error, not a throw: the ctor must survive it
// and let the runtime skip the stage at launch.
TEST(create_mask_stage, bad_mask_mode_fails_config)
{
  CerrSilencer hush;
  Session sess;

  CreateMaskStage stage(&sess, "mk", vector<InEdge>{},
                        cfg_({{"mask_mode",
                               FlexData::make_string("tri-state")}}));
  EXPECT_TRUE(!stage.config_error().empty());
}

// The canvas falls back the way the doc comment says: both set wins,
// one set infers the other from the source aspect ratio, neither
// matches the source.
TEST(create_mask_stage, canvas_geometry_resolution)
{
  CerrSilencer hush;
  Session sess;
  int w = 0, h = 0;

  CreateMaskStage both(&sess, "a", vector<InEdge>{},
                       cfg_({{"width", FlexData::make_int(64)},
                             {"height", FlexData::make_int(32)}}));
  both.resolve_canvas_(200, 100, &w, &h);
  EXPECT_TRUE(w == 64 && h == 32);

  CreateMaskStage wide(&sess, "b", vector<InEdge>{},
                       cfg_({{"width", FlexData::make_int(100)}}));
  wide.resolve_canvas_(200, 100, &w, &h);
  EXPECT_TRUE(w == 100 && h == 50);

  CreateMaskStage tall(&sess, "c", vector<InEdge>{},
                       cfg_({{"height", FlexData::make_int(50)}}));
  tall.resolve_canvas_(200, 100, &w, &h);
  EXPECT_TRUE(w == 100 && h == 50);

  CreateMaskStage none(&sess, "d", vector<InEdge>{},
                       FlexData::make_object());
  none.resolve_canvas_(200, 100, &w, &h);
  EXPECT_TRUE(w == 200 && h == 100);

  // No source to infer an aspect ratio from: a square, rather than an
  // invented ratio.
  none.resolve_canvas_(0, 0, &w, &h);
  EXPECT_TRUE(w == 0 && h == 0);
  wide.resolve_canvas_(0, 0, &w, &h);
  EXPECT_TRUE(w == 100 && h == 100);
}

// The channel carries frames DOWN only -- latest-wins behind a version.
// The commit goes up as the stage's `commit` command (see below).
TEST(create_mask_stage, channel_latches_frames)
{
  MaskEditorChannel ch;
  EXPECT_TRUE(ch.snapshot().version == 0);

  MaskEditorChannel::Frame f;
  f.width  = 64;
  f.height = 32;
  f.mask   = make_shared<const vector<uint8_t>>(vector<uint8_t>{1, 2, 3});
  ch.publish(f);
  auto s1 = ch.snapshot();
  EXPECT_TRUE(s1.version == 1);
  EXPECT_TRUE(s1.width == 64 && s1.height == 32);
  EXPECT_TRUE(s1.mask != nullptr);
  EXPECT_TRUE(s1.background == nullptr);

  // Republishing still bumps the version, so a waiter is woken for a
  // re-render rather than sleeping through it.
  ch.publish(f);
  EXPECT_TRUE(ch.snapshot().version == 2);
  EXPECT_TRUE(ch.wait_change(1, 50).version == 2);

  ch.close();
  EXPECT_TRUE(ch.closed());
  EXPECT_TRUE(ch.wait_change(2, 50).closed);
}

#if defined(__APPLE__) && defined(__arm64__)

// Headless: the mask arrives, is converted, and leaves as [1,H,W] with
// its two states intact.
TEST(create_mask_stage, headless_mask_passthrough)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto m_u = make_unique<RepeatSource>(&sess, "m", vector<InEdge>{},
                                       FlexData::make_object());
  m_u->tb = make_mask_(24, 32, 255);
  m_u->allocate_oports(1);
  auto* m = static_cast<RepeatSource*>(pl->insert_stage(std::move(m_u)));

  auto k_u = make_unique<CreateMaskStage>(
      &sess, "mk", vector<InEdge>{{nullptr, 0}, {m, 0}},
      cfg_({{"interactive", FlexData::make_bool(false)}}));
  auto* mk = static_cast<CreateMaskStage*>(pl->insert_stage(std::move(k_u)));

  auto c_u = make_unique<Collector>(&sess, "c", vector<InEdge>{{mk, 0}},
                                    FlexData::make_object());
  auto* col = static_cast<Collector*>(pl->insert_stage(std::move(c_u)));

  PipelineRuntime rt(pl.get(), &sess);
  rt.launch();
  this_thread::sleep_for(chrono::milliseconds(600));
  rt.stop();

  ASSERT_TRUE(col->count() >= 1);
  const TensorBeat out = col->last();
  ASSERT_TRUE(out.shape.size() == 3);
  EXPECT_TRUE(out.shape[0] == 1 && out.shape[1] == 24 && out.shape[2] == 32);
  EXPECT_TRUE(out.dtype == TensorBeat::DType::U8);
  const uint8_t* p = out.as_u8();
  EXPECT_TRUE(p[0] == 255);        // left half set
  EXPECT_TRUE(p[31] == 0);         // right half clear
  EXPECT_TRUE(mk->mask_width() == 32 && mk->mask_height() == 24);
}

// Headless overlay: the output is the REFERENCE image's resolution with
// the mask painted on it, not the canvas resolution -- and the unmasked
// half is left exactly as it came in.
TEST(create_mask_stage, headless_overlay_paints_reference)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto r_u = make_unique<RepeatSource>(&sess, "r", vector<InEdge>{},
                                       FlexData::make_object());
  r_u->tb = make_rgb_(24, 32, 0.0f);      // black reference
  r_u->count = 4;
  r_u->allocate_oports(1);
  auto* r = static_cast<RepeatSource*>(pl->insert_stage(std::move(r_u)));

  auto m_u = make_unique<RepeatSource>(&sess, "m", vector<InEdge>{},
                                       FlexData::make_object());
  m_u->tb = make_mask_(24, 32, 255);
  m_u->count = 4;
  m_u->allocate_oports(1);
  auto* m = static_cast<RepeatSource*>(pl->insert_stage(std::move(m_u)));

  auto k_u = make_unique<CreateMaskStage>(
      &sess, "mk", vector<InEdge>{{r, 0}, {m, 0}},
      cfg_({{"interactive", FlexData::make_bool(false)},
            {"output", FlexData::make_string("overlay")},
            {"overlay_color", FlexData::make_string("#ff0000")},
            {"overlay_opacity", FlexData::make_real(1.0)}}));
  auto* mk = static_cast<CreateMaskStage*>(pl->insert_stage(std::move(k_u)));

  auto c_u = make_unique<Collector>(&sess, "c", vector<InEdge>{{mk, 0}},
                                    FlexData::make_object());
  auto* col = static_cast<Collector*>(pl->insert_stage(std::move(c_u)));

  PipelineRuntime rt(pl.get(), &sess);
  rt.launch();
  this_thread::sleep_for(chrono::milliseconds(800));
  rt.stop();

  ASSERT_TRUE(col->count() >= 1);
  const TensorBeat out = col->last();
  ASSERT_TRUE(out.shape.size() == 3);
  EXPECT_TRUE(out.shape[0] == 3 && out.shape[1] == 24 && out.shape[2] == 32);
  const uint8_t* p = out.as_u8();
  const size_t   plane = 24 * 32;
  // Fully opaque red where the mask is set; untouched black where it is
  // not. Planar, so the three channels are plane-strided.
  EXPECT_TRUE(p[0] == 255 && p[plane] == 0 && p[2 * plane] == 0);
  EXPECT_TRUE(p[31] == 0 && p[plane + 31] == 0 && p[2 * plane + 31] == 0);
}

// A class map is an INDEX map: it must come back with the indices it
// went in with, never averaged, and never scaled by input_normalized.
TEST(create_mask_stage, class_indices_survive_unscaled)
{
  Session sess;
  auto pl = make_unique<Pipeline>("p", &sess);

  auto m_u = make_unique<RepeatSource>(&sess, "m", vector<InEdge>{},
                                       FlexData::make_object());
  m_u->tb = make_mask_(16, 16, 3);        // class 3 on the left half
  m_u->allocate_oports(1);
  auto* m = static_cast<RepeatSource*>(pl->insert_stage(std::move(m_u)));

  auto k_u = make_unique<CreateMaskStage>(
      &sess, "mk", vector<InEdge>{{nullptr, 0}, {m, 0}},
      cfg_({{"interactive", FlexData::make_bool(false)},
            {"mask_mode", FlexData::make_string("class")},
            {"classes", FlexData::make_int(5)}}));
  auto* mk = static_cast<CreateMaskStage*>(pl->insert_stage(std::move(k_u)));

  auto c_u = make_unique<Collector>(&sess, "c", vector<InEdge>{{mk, 0}},
                                    FlexData::make_object());
  auto* col = static_cast<Collector*>(pl->insert_stage(std::move(c_u)));

  PipelineRuntime rt(pl.get(), &sess);
  rt.launch();
  this_thread::sleep_for(chrono::milliseconds(600));
  rt.stop();

  ASSERT_TRUE(col->count() >= 1);
  const TensorBeat out = col->last();
  const uint8_t* p = out.as_u8();
  EXPECT_TRUE(p[0] == 3);
  EXPECT_TRUE(p[15] == 0);
}

// The editor round trip. The stage publishes its mask as a PNG; feeding
// that same PNG back as a commit must decode to the mask it encoded and
// produce exactly ONE beat -- which is the whole contract between the
// panel's commit button and the oport.
// Builds m (mask source) -> mk (create-mask, interactive) -> c
// (collector), launched. The stage sits waiting for commands.
struct MaskRig {
  Session                   sess;
  unique_ptr<Pipeline>      pl;
  CreateMaskStage*          mk  = nullptr;
  Collector*                col = nullptr;
  unique_ptr<PipelineRuntime> rt;

  MaskRig(int H, int W, uint8_t v)
  {
    pl = make_unique<Pipeline>("p", &sess);
    auto m_u = make_unique<RepeatSource>(&sess, "m", vector<InEdge>{},
                                         FlexData::make_object());
    m_u->tb = make_mask_(H, W, v);
    m_u->allocate_oports(1);
    auto* m = static_cast<RepeatSource*>(pl->insert_stage(std::move(m_u)));
    auto k_u = make_unique<CreateMaskStage>(
        &sess, "mk", vector<InEdge>{{nullptr, 0}, {m, 0}},
        FlexData::make_object());      // interactive by default
    mk = static_cast<CreateMaskStage*>(pl->insert_stage(std::move(k_u)));
    auto c_u = make_unique<Collector>(&sess, "c", vector<InEdge>{{mk, 0}},
                                      FlexData::make_object());
    col = static_cast<Collector*>(pl->insert_stage(std::move(c_u)));
    rt = make_unique<PipelineRuntime>(pl.get(), &sess);
    rt->launch();
  }
  ~MaskRig() { rt->stop(); }

  // Wait until the stage has published its first frame (it has read the
  // seed mask), or give up.
  MaskEditorChannel::Frame
  settled()
  {
    auto ch = mk->mask_channel();
    return ch->wait_change(0, 3000);
  }
};

DataBuffer
bytes_buffer_(const char* name, vector<uint8_t> bytes)
{
  auto own = make_shared<vector<uint8_t>>(std::move(bytes));
  return DataBuffer::view(
      name, own->data(),
      BufferLayout::contiguous(ElementType::Bytes,
                               {static_cast<int64_t>(own->size())}),
      own);
}

uint64_t
seq_of_(const shared_ptr<StageCommand>& c)
{
  const FlexData r = c->result();
  return r.is_object() && r.as_object().contains("seq")
             ? r.as_object().at("seq").as_uint(0) : 0;
}

TEST(create_mask_stage, commit_round_trip_emits_one_beat)
{
  MaskRig rig(24, 32, 255);
  const auto snap = rig.settled();
  ASSERT_TRUE(snap.version >= 1);
  ASSERT_TRUE(snap.mask != nullptr && !snap.mask->empty());
  EXPECT_TRUE(snap.width == 32 && snap.height == 24);

  // Waiting on commands, so nothing has been emitted yet.
  EXPECT_TRUE(rig.col->count() == 0);

  // The editor's path: its own PNG, as the `commit` command.
  auto cmd = open_stage_command(
      *rig.mk, "commit", FlexData::make_object(),
      {bytes_buffer_("png", *snap.mask)});
  // The reply means the beat is on the oport.
  EXPECT_TRUE(cmd->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(seq_of_(cmd) == 1);
  rig.rt->stop();

  ASSERT_TRUE(rig.col->count() == 1);
  const TensorBeat out = rig.col->last();
  ASSERT_TRUE(out.shape.size() == 3);
  EXPECT_TRUE(out.shape[0] == 1 && out.shape[1] == 24 && out.shape[2] == 32);
  const uint8_t* p = out.as_u8();
  EXPECT_TRUE(p[0] == 255);
  EXPECT_TRUE(p[31] == 0);
}

// A commit the decoder cannot make sense of is refused, not fatal, and
// costs no beat -- the panel is a network peer and may send anything.
// The refusal now reaches the sender as the command's error.
TEST(create_mask_stage, undecodable_commit_is_refused)
{
  CerrSilencer hush;
  MaskRig rig(16, 16, 255);
  rig.settled();
  auto cmd = open_stage_command(
      *rig.mk, "commit", FlexData::make_object(),
      {bytes_buffer_("png", {'n', 'o', 't', 'a', 'p', 'n', 'g'})});
  EXPECT_TRUE(cmd->wait(3000) == CommandState::Failed);
  EXPECT_TRUE(cmd->error().find("PNG") != string::npos);
  rig.rt->stop();
  EXPECT_TRUE(rig.col->count() == 0);
}

// THE REGRESSION the command queue exists for. The old channel was one
// latched commit slot, so two commits landing while the stage was busy
// became ONE beat carrying the second mask. Queued commands are two
// beats, in order -- here posted back to back without waiting.
TEST(create_mask_stage, back_to_back_commits_are_two_beats)
{
  MaskRig rig(8, 8, 255);
  rig.settled();
  auto sample = [](uint8_t v) {
    auto own = make_shared<vector<uint8_t>>(64, v);
    return DataBuffer::view(
        "mask", own->data(),
        BufferLayout::contiguous(ElementType::U8, {8, 8}), own);
  };
  auto a = open_stage_command(*rig.mk, "commit", FlexData::make_object(),
                              {sample(255)});
  auto b = open_stage_command(*rig.mk, "commit", FlexData::make_object(),
                              {sample(0)});
  EXPECT_TRUE(a->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(b->wait(3000) == CommandState::Replied);
  EXPECT_TRUE(seq_of_(a) == 1 && seq_of_(b) == 2);
  rig.rt->stop();
  ASSERT_TRUE(rig.col->count() == 2);
  EXPECT_TRUE(rig.col->last().as_u8()[0] == 0);   // the second one, last
}

// `read` lends the mask OUT in place, and the stage does nothing else
// until it is closed: a commit sent meanwhile waits in the queue. After
// the close the stage has its mask back and serves the commit.
TEST(create_mask_stage, read_holds_the_stage_until_closed)
{
  MaskRig rig(6, 10, 255);
  rig.settled();
  auto rd = open_stage_command(*rig.mk, "read", FlexData::make_object(), {});
  ASSERT_TRUE(rd->wait(3000) == CommandState::Replied);
  const auto bufs = rd->outputs();
  ASSERT_TRUE(bufs.size() == 1);              // no reference image wired
  const DataBuffer& m = bufs[0];
  EXPECT_TRUE(m.name == "mask" && !m.writable);
  EXPECT_TRUE(m.layout.shape == (vector<int64_t>{1, 6, 10}));
  const auto* px = static_cast<const uint8_t*>(m.data);
  EXPECT_TRUE(px[0] == 255 && px[9] == 0);    // left half set

  auto zero = make_shared<vector<uint8_t>>(60, 0);
  auto cm = open_stage_command(
      *rig.mk, "commit", FlexData::make_object(),
      {DataBuffer::view("mask", zero->data(),
                        BufferLayout::contiguous(ElementType::U8, {6, 10}),
                        zero)});
  // Held: the commit cannot be served while the read is open.
  EXPECT_TRUE(cm->wait(300) == CommandState::Pending);
  EXPECT_TRUE(rig.col->count() == 0);

  rd->close();
  EXPECT_TRUE(cm->wait(3000) == CommandState::Replied);
  rig.rt->stop();
  ASSERT_TRUE(rig.col->count() == 1);
  EXPECT_TRUE(rig.col->last().as_u8()[0] == 0);
}

// The host refuses what the spec rules out before the stage sees it;
// the stage refuses what only it can judge.
TEST(create_mask_stage, malformed_commands_are_refused)
{
  MaskRig rig(4, 4, 255);
  rig.settled();
  // Unknown command: named, with what the stage does accept.
  auto u = open_stage_command(*rig.mk, "paint", FlexData::make_object(), {});
  EXPECT_TRUE(u->state() == CommandState::Failed);
  EXPECT_TRUE(u->error().find("commit") != string::npos);
  // A 3-D mask with 2 channels matches neither [1,H,W] nor [H,W].
  auto own = make_shared<vector<uint8_t>>(32, 1);
  auto bad = open_stage_command(
      *rig.mk, "commit", FlexData::make_object(),
      {DataBuffer::view("mask", own->data(),
                        BufferLayout::contiguous(ElementType::U8, {2, 4, 4}),
                        own)});
  EXPECT_TRUE(bad->state() == CommandState::Failed);
  EXPECT_TRUE(bad->error().find("shape") != string::npos);
  // Neither buffer: legal to the spec (both optional), refused by the
  // stage, which is the only one that knows one of them is needed.
  auto none = open_stage_command(*rig.mk, "commit", FlexData::make_object(),
                                 {});
  EXPECT_TRUE(none->wait(3000) == CommandState::Failed);
  rig.rt->stop();
  EXPECT_TRUE(rig.col->count() == 0);
}

#endif  // __APPLE__ && __arm64__
