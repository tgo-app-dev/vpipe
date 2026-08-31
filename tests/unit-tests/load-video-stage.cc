// `load-video`: the START AND DURATION window.
//
// The stage demuxes a container into encoded packets and this file
// pins what `start_s` / `duration_s` do to that stream. It is driven by
// a WAV the test WRITES ITSELF, read with `enable_video: false`: a WAV
// is a container FFmpeg demuxes like any other, so it exercises the
// window arithmetic -- the seek, the per-stream end, and the rule that
// the RUN ends only when every enabled stream has passed the end --
// with no checked-in asset and no codec to depend on.
//
// What it deliberately does NOT cover is the keyframe lead-in on a
// video stream, which needs a real inter-coded fixture. That path is
// the one documented at length in the header, and it is the reason
// this stage cannot drop what precedes the window the way load-audio
// does.

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/encoded-segment.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/load-video-stage.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace vpipe;

namespace {

// A mono 16-bit PCM WAV of `n` samples at `rate`. Same shape as the one
// in load-audio-stage.cc, kept local so the two files do not have to
// share a fixture helper across a header.
std::string
write_wav_(int rate, int n, const char* name)
{
  namespace fs = std::filesystem;
  const fs::path p = fs::temp_directory_path() / name;
  std::ofstream f(p, std::ios::binary);
  if (!f) { return {}; }
  auto u32 = [&](std::uint32_t v) { f.write((const char*)&v, 4); };
  auto u16 = [&](std::uint16_t v) { f.write((const char*)&v, 2); };
  const std::uint32_t data_bytes = (std::uint32_t)n * 2;
  f.write("RIFF", 4);
  u32(36 + data_bytes);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  u32(16);
  u16(1);
  u16(1);
  u32((std::uint32_t)rate);
  u32((std::uint32_t)rate * 2);
  u16(2);
  u16(16);
  f.write("data", 4);
  u32(data_bytes);
  for (int i = 0; i < n; ++i) {
    const double t = (double)i / rate;
    const auto s = (std::int16_t)(12000.0 * std::sin(2.0 * M_PI * 440.0 * t));
    u16((std::uint16_t)s);
  }
  f.close();
  return p.string();
}

class SegSink : public TypedStage<SegSink> {
public:
  static constexpr const char* kTypeName = "ut-lv-seg-sink";
  using TypedStage::TypedStage;

  int beats = 0;
  std::size_t total_bytes = 0;
  std::int64_t first_us = -1, last_us = -1;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* s = dynamic_cast<const EncodedSegmentPayload*>(in.get())) {
      ++beats;
      total_bytes += s->data.size();
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
          s->start_utc.time_since_epoch()).count();
      if (first_us < 0) { first_us = us; }
      last_us = us;
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "seg", .doc = "", .type = &typeid(EncodedSegmentPayload)}};
    static const StageSpec s = {.type_name = "ut-lv-seg-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

}  // namespace

TEST(load_video, start_s_and_duration_s_take_a_window)
{
  const int kRate = 16000;
  const int kSeconds = 4;
  const std::string path =
      write_wav_(kRate, kRate * kSeconds, "vpipe-ut-load-video-window.wav");
  if (path.empty()) { return; }

  auto run = [&](double start_s, double duration_s) {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("input_url", FlexData::make_string(path));
    cfg.as_object().insert_or_assign("enable_video",
                                     FlexData::make_bool(false));
    cfg.as_object().insert_or_assign("enable_audio",
                                     FlexData::make_bool(true));
    if (start_s > 0.0) {
      cfg.as_object().insert_or_assign("start_s",
                                       FlexData::make_real(start_s));
    }
    if (duration_s > 0.0) {
      cfg.as_object().insert_or_assign("duration_s",
                                       FlexData::make_real(duration_s));
    }
    auto lv_u = std::make_unique<LoadVideoStage>(
        &sess, "lv", std::vector<InEdge>{}, cfg);
    auto* lv = static_cast<LoadVideoStage*>(pl->insert_stage(std::move(lv_u)));
    auto sink_u = std::make_unique<SegSink>(&sess, "sink",
                                            std::vector<InEdge>{{lv, 0}},
                                            FlexData::make_object());
    auto* sink = static_cast<SegSink*>(pl->insert_stage(std::move(sink_u)));
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) {
      return std::make_tuple((std::size_t)0, (std::int64_t)-1,
                             (std::int64_t)-1);
    }
    rt.wait_idle();
    rt.stop();
    return std::make_tuple(sink->total_bytes, sink->first_us, sink->last_us);
  };

  // The whole file, for the comparison the window is measured against.
  const auto [all_bytes, all_first, all_last] = run(0.0, 0.0);
  const std::size_t want_all = (std::size_t)kRate * kSeconds * 2;
  EXPECT_TRUE(all_bytes == want_all);
  EXPECT_TRUE(all_first == 0);

  const auto [win_bytes, win_first, win_last] = run(1.0, 2.0);
  std::printf("[load_video] window [1,3): %zu B (whole file %zu B), "
              "first_us %lld, last_us %lld\n", win_bytes, all_bytes,
              (long long)win_first, (long long)win_last);

  // A genuine slice: not the file, not empty, and about the right size.
  // Wider tolerance at the START than load-audio's, on purpose -- this
  // stage does NOT drop what precedes the window, so the seek's landing
  // point is part of the answer.
  EXPECT_TRUE(win_bytes > 0 && win_bytes < all_bytes);
  const std::size_t want_win = (std::size_t)kRate * 2 * 2;
  EXPECT_TRUE(win_bytes < want_win * 2);

  // Media time is NOT rebased: a window starting at 1 s stamps ~1 s.
  EXPECT_TRUE(win_first >= 0 && win_first <= 1100000);
  // ... and it stops rather than running to the end of the file.
  EXPECT_TRUE(win_last < 3200000);
  EXPECT_TRUE(win_last < all_last);

  std::filesystem::remove(path);
}

// A negative window is refused at CONFIG, where a graph can still be
// fixed, rather than clamped to something the author did not ask for.
TEST(load_video, a_negative_window_is_refused)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("input_url",
                                   FlexData::make_string("/tmp/x.mp4"));
  cfg.as_object().insert_or_assign("start_s", FlexData::make_real(-1.0));
  auto s = std::make_unique<LoadVideoStage>(
      &sess, "lv", std::vector<InEdge>{}, cfg);
  EXPECT_FALSE(s->config_error().empty());

  auto cfg2 = FlexData::make_object();
  cfg2.as_object().insert_or_assign("input_url",
                                    FlexData::make_string("/tmp/x.mp4"));
  cfg2.as_object().insert_or_assign("duration_s", FlexData::make_real(-2.0));
  auto s2 = std::make_unique<LoadVideoStage>(
      &sess, "lv2", std::vector<InEdge>{}, cfg2);
  EXPECT_FALSE(s2->config_error().empty());
}
