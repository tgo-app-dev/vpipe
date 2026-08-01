#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "common/preview-channel.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/preview-stage.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// Emits `count` deep copies of `tb`, optionally paced, then closes.
class RepeatSource : public TypedStage<RepeatSource> {
public:
  static constexpr const char* kTypeName = "ut-preview-repeat-source";
  using TypedStage::TypedStage;

  TensorBeat tb;
  int        count       = 1;
  int        per_beat_us = 0;

  Job process(RuntimeContext& ctx) override
  {
    if (_emitted >= count) {
      ctx.signal_done();
      co_return;
    }
    ++_emitted;
    if (per_beat_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(per_beat_us));
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  int _emitted = 0;
};

// Closes its oport immediately without producing anything.
class ClosedSource : public TypedStage<ClosedSource> {
public:
  static constexpr const char* kTypeName = "ut-preview-closed-source";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

TensorBeat
make_tensor_(int H, int W, float v = 0.5f)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {3, H, W};
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  float* p = tb.as_f32();
  for (size_t i = 0; i < static_cast<size_t>(3) * H * W; ++i) { p[i] = v; }
  return tb;
}

// A video-type frame: carries an fps sideband (as video-to-rgb tags its
// output). The preview stage treats such a source as video, never image.
TensorBeat
make_tensor_fps_(int H, int W, int fps, float v = 0.5f)
{
  TensorBeat tb = make_tensor_(H, W, v);
  FlexData sb = FlexData::make_object();
  sb.as_object().insert("fps_num", FlexData::make_uint(fps));
  sb.as_object().insert("fps_den", FlexData::make_uint(1));
  tb.sideband = std::move(sb);
  return tb;
}

TensorBeat
make_pcm_tensor_(int n, int sample_rate)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {static_cast<int64_t>(n)};
  tb.resize_contiguous(static_cast<size_t>(n));
  float* p = tb.as_f32();
  for (int i = 0; i < n; ++i) {
    p[i] = 0.05f * std::sin(2.0f * 3.14159265f * 440.0f
                            * static_cast<float>(i)
                            / static_cast<float>(sample_rate));
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert("sample_rate", FlexData::make_int(sample_rate));
  tb.sideband = std::move(sb);
  return tb;
}

// True if the 4-char box tag `s` appears anywhere in `v`.
bool
has_box_(const vector<uint8_t>& v, const char* s)
{
  const size_t n = std::strlen(s);
  if (v.size() < n) { return false; }
  for (size_t i = 0; i + n <= v.size(); ++i) {
    if (std::memcmp(v.data() + i, s, n) == 0) { return true; }
  }
  return false;
}

// Tally of the WebSocket messages a subscriber received off the channel.
struct Collected {
  int  config    = 0;
  int  init      = 0;
  int  fragment  = 0;
  int  audio     = 0;
  int  image     = 0;
  bool cfg_video = false;
  bool cfg_audio = false;
  vector<uint8_t> last_init;
  vector<uint8_t> last_fragment;
  vector<uint8_t> last_image;
};

Collected
drain_(const std::shared_ptr<PreviewChannel>& ch,
       const std::shared_ptr<PreviewChannel::Subscriber>& sub)
{
  Collected c;
  for (int guard = 0; guard < 2'000'000; ++guard) {
    auto blob = ch->wait_frame(sub, 200);
    if (!blob) {
      if (ch->closed()) { break; }
      continue;
    }
    const auto& b = *blob;
    if (b.empty()) { continue; }
    const uint8_t type = b[0];
    const vector<uint8_t> payload(b.begin() + 1, b.end());
    if (type == PreviewChannel::kMsgConfig) {
      ++c.config;
      const std::string j(payload.begin(), payload.end());
      if (j.find("\"video\"") != std::string::npos) { c.cfg_video = true; }
      if (j.find("\"audio\"") != std::string::npos) { c.cfg_audio = true; }
    } else if (type == PreviewChannel::kMsgInit) {
      ++c.init;
      c.last_init = payload;
    } else if (type == PreviewChannel::kMsgFragment) {
      ++c.fragment;
      c.last_fragment = payload;
    } else if (type == PreviewChannel::kMsgAudio) {
      ++c.audio;
    } else if (type == PreviewChannel::kMsgImage) {
      ++c.image;
      c.last_image = payload;
    }
  }
  ch->unsubscribe(sub);
  return c;
}

}  // namespace

TEST(preview_stage, defaults_construct_channel_open)
{
  CerrSilencer hush;
  Session sess;

  PreviewStage stage(&sess, "pv", vector<InEdge>{}, FlexData::make_object());
  EXPECT_TRUE(!stage.encoder_initialized());
  EXPECT_TRUE(stage.cadence_fps() == 25);
  auto ch = stage.preview_channel();
  EXPECT_TRUE(ch != nullptr);
  EXPECT_TRUE(!ch->closed());
}

#if defined(__APPLE__) && defined(__arm64__)

// Run a preview pipeline for `run_ms` then stop it (the stage is self-
// clocked and never signals done on its own), draining what a subscriber
// attached before launch received.
static Collected
run_preview_(Session& sess, Pipeline& pl, PreviewStage* pv, int run_ms)
{
  auto ch  = pv->preview_channel();
  auto sub = ch->subscribe();
  PipelineRuntime rt(&pl, &sess);
  rt.launch();
  std::this_thread::sleep_for(std::chrono::milliseconds(run_ms));
  rt.stop();
  return drain_(ch, sub);
}

// A RELAUNCH must produce a live stream again. teardown_() closes the
// channel when the pipeline stops, and the web-ui view backend reports
// "waiting" for exactly `!pc || pc->closed()` -- so before
// reset_run_state() re-armed it, every launch after the first left the
// viewer waiting forever while the pipeline ran happily.
//
// The channel is reopened IN PLACE, not replaced: this test holds the
// pointer across both runs, exactly as the view backend and a live
// subscriber do.
TEST(preview_stage, relaunch_reopens_the_channel_and_streams_again)
{
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<ClosedSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->allocate_oports(1);
  auto* src = static_cast<ClosedSource*>(
      pl->insert_stage(std::move(src_u)));
  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  // Captured ONCE, before any launch -- the pointer must stay valid.
  auto ch = pv->preview_channel();
  ASSERT_TRUE(ch != nullptr);

  for (int run = 0; run < 2; ++run) {
    auto sub = ch->subscribe();
    PipelineRuntime rt(pl.get(), &sess);
    EXPECT_TRUE(rt.launch());
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    // Live mid-run: a closed channel here is the "stuck waiting" bug.
    EXPECT_FALSE(ch->closed());
    EXPECT_TRUE(pv->preview_channel() == ch);   // same object, reopened
    rt.stop();
    const Collected c = drain_(ch, sub);
    // Each run is a complete stream of its own: init segment + media.
    EXPECT_TRUE(c.init > 0);
    EXPECT_TRUE(c.fragment > 0);
  }
}

TEST(preview_stage, black_before_input_produces_fmp4)
{
  // No video frames ever arrive: the stage still self-clocks a black
  // stream -- an fMP4 init segment + media fragments.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<ClosedSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->allocate_oports(1);
  auto* src = static_cast<ClosedSource*>(
      pl->insert_stage(std::move(src_u)));

  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  // ~2 s run: the cadence must flush multiple fragments (>= 2), not just
  // the single one the teardown produces -- otherwise a broken live flush
  // would slip past.
  Collected c = run_preview_(sess, *pl, pv, 2000);
  EXPECT_TRUE(c.config >= 1);
  EXPECT_TRUE(c.cfg_video);
  EXPECT_TRUE(c.init >= 1);
  EXPECT_TRUE(has_box_(c.last_init, "ftyp"));
  EXPECT_TRUE(has_box_(c.last_init, "moov"));
  EXPECT_TRUE(c.fragment >= 2);
  EXPECT_TRUE(has_box_(c.last_fragment, "moof"));
  EXPECT_TRUE(has_box_(c.last_fragment, "mdat"));
}

TEST(preview_stage, still_image_switches_to_image_mode)
{
  // A single still image (one beat then EOS, no fps sideband) is an
  // image-type source arriving slower than 1 fps: after ~1 s the stage
  // stops encoding video and sends the picture as a PNG still (type 5).
  // The output still adopts the image's native resolution.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_tensor_(240, 320, 0.5f);   // [3,240,320]
  src_u->count = 1;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatSource*>(pl->insert_stage(std::move(src_u)));

  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  Collected c = run_preview_(sess, *pl, pv, 2500);
  EXPECT_TRUE(c.cfg_video);
  EXPECT_TRUE(c.init >= 1);
  EXPECT_TRUE(has_box_(c.last_init, "moov"));
  EXPECT_TRUE(c.image >= 1);            // flipped to still-image mode
  EXPECT_TRUE(pv->image_mode_active());
  // The still is a PNG (0x89 'P' 'N' 'G' signature).
  EXPECT_TRUE(c.last_image.size() > 8);
  EXPECT_TRUE(c.last_image[0] == 0x89 && c.last_image[1] == 'P'
              && c.last_image[2] == 'N' && c.last_image[3] == 'G');
  // Native resolution adopted from the frame.
  EXPECT_TRUE(pv->output_width() == 320);
  EXPECT_TRUE(pv->output_height() == 240);
}

TEST(preview_stage, image_mode_disabled_stays_video)
{
  // image_mode=false forces the legacy behavior: a still image repeats as
  // continuous video (fragments keep flowing), no type-5 stills.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_tensor_(240, 320, 0.5f);
  src_u->count = 1;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatSource*>(pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("image_mode", FlexData::make_bool(false));
  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  Collected c = run_preview_(sess, *pl, pv, 2500);
  EXPECT_TRUE(c.image == 0);           // never enters image mode
  EXPECT_TRUE(!pv->image_mode_active());
  EXPECT_TRUE(c.fragment >= 2);        // still flowing as video
  EXPECT_TRUE(has_box_(c.last_fragment, "moof"));
}

TEST(preview_stage, fast_image_source_stays_video)
{
  // An image-type source (no fps sideband) that arrives faster than 1 fps
  // is NOT downgraded to stills -- it stays video.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb          = make_tensor_(240, 320, 0.5f);   // no fps sideband
  src_u->count       = 1000;
  src_u->per_beat_us = 100'000;                        // ~10 fps
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatSource*>(pl->insert_stage(std::move(src_u)));

  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  Collected c = run_preview_(sess, *pl, pv, 2500);
  EXPECT_TRUE(c.image == 0);           // fast input -> stays video
  EXPECT_TRUE(!pv->image_mode_active());
  EXPECT_TRUE(c.fragment >= 2);
}

TEST(preview_stage, slow_video_source_stays_video)
{
  // A slow VIDEO-type source (single frame, but tagged with an fps
  // sideband like a camera via video-to-rgb) stays video however slow it
  // runs -- image mode is only for image-type sources.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_tensor_fps_(240, 320, 30, 0.5f);   // fps sideband
  src_u->count = 1;                                       // then idle
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatSource*>(pl->insert_stage(std::move(src_u)));

  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  Collected c = run_preview_(sess, *pl, pv, 2500);
  EXPECT_TRUE(c.image == 0);           // video-type: never image mode
  EXPECT_TRUE(!pv->image_mode_active());
  EXPECT_TRUE(c.fragment >= 2);        // repeats as video
}

TEST(preview_stage, adopts_native_resolution_reinits)
{
  // The pre-input black frame uses the default size; the first real frame
  // re-initializes the stream to the native size -> a SECOND init segment
  // reaches the subscriber.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb          = make_tensor_(240, 320, 0.5f);
  src_u->count       = 100;
  src_u->per_beat_us = 33'000;                  // ~30 fps input
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatSource*>(pl->insert_stage(std::move(src_u)));

  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{src, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  Collected c = run_preview_(sess, *pl, pv, 1200);
  EXPECT_TRUE(c.init >= 2);            // black init + native re-init
  EXPECT_TRUE(pv->output_width() == 320);
  EXPECT_TRUE(pv->output_height() == 240);
}

TEST(preview_stage, video_plus_audio)
{
  // Video on iport 0 + audio on iport 1 -> the config declares both, and
  // audio (type 4) messages flow alongside the fMP4 fragments.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto v_u = make_unique<RepeatSource>(
      &sess, "vsrc", vector<InEdge>{}, FlexData::make_object());
  v_u->tb          = make_tensor_(240, 320, 0.5f);
  v_u->count       = 100;
  v_u->per_beat_us = 33'000;
  v_u->allocate_oports(1);
  auto* v = static_cast<RepeatSource*>(pl->insert_stage(std::move(v_u)));

  auto a_u = make_unique<RepeatSource>(
      &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  a_u->tb          = make_pcm_tensor_(4800, 48000);   // 0.1 s mono @ 48k
  a_u->count       = 20;
  a_u->per_beat_us = 100'000;
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatSource*>(pl->insert_stage(std::move(a_u)));

  auto pv_u = make_unique<PreviewStage>(
      &sess, "pv", vector<InEdge>{{v, 0}, {a, 0}}, FlexData::make_object());
  auto* pv = static_cast<PreviewStage*>(pl->insert_stage(std::move(pv_u)));

  Collected c = run_preview_(sess, *pl, pv, 2000);
  EXPECT_TRUE(c.cfg_video);
  EXPECT_TRUE(c.cfg_audio);
  EXPECT_TRUE(c.fragment >= 2);
  EXPECT_TRUE(c.audio >= 2);
}

#endif  // __APPLE__ && __arm64__
