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
  bool cfg_video = false;
  bool cfg_audio = false;
  vector<uint8_t> last_init;
  vector<uint8_t> last_fragment;
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

TEST(preview_stage, still_image_repeats_at_cadence)
{
  // A single still image (one beat then EOS) must play as continuous video:
  // the cadence repeats the last frame, so fragments keep flowing. The
  // output adopts the image's native resolution.
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

  Collected c = run_preview_(sess, *pl, pv, 2000);
  EXPECT_TRUE(c.cfg_video);
  EXPECT_TRUE(c.init >= 1);
  EXPECT_TRUE(has_box_(c.last_init, "moov"));
  EXPECT_TRUE(c.fragment >= 2);        // still image keeps flowing live
  EXPECT_TRUE(has_box_(c.last_fragment, "moof"));
  // Native resolution adopted from the frame.
  EXPECT_TRUE(pv->output_width() == 320);
  EXPECT_TRUE(pv->output_height() == 240);
  EXPECT_TRUE(pv->codec_string().rfind("avc1.", 0) == 0);
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
