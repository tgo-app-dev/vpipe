// TemporalDecimationStage tests.
//
// The CPU-only suite (decide_keep_for_test) exercises the token-bucket
// drop policy, the max-consecutive-drops cap, and the focus-class
// motion boost without touching the GPU. The Metal-runtime smoke test
// runs the full process() path through a small pipeline (synthetic
// 64x64 frames) and asserts the kept-frame count, sideband telemetry,
// and motion-detection wiring all behave end-to-end.
//
// Run all:
//   vpipe_test --filter '*temporal_decimation*'

#include "minitest.h"

#include "stages/audio-video/temporal-decimation-stage.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

FlexData
make_cfg_(double max_avg_fps,
          unsigned max_consec_drops,
          double motion_threshold,
          double focus_motion_gain)
{
  FlexData cfg = FlexData::make_object();
  auto root = cfg.as_object();
  root.insert("max_avg_fps", FlexData::make_real(max_avg_fps));
  root.insert("max_consecutive_drops",
              FlexData::make_uint(max_consec_drops));
  root.insert("motion_threshold",
              FlexData::make_real(motion_threshold));
  root.insert("focus_motion_gain",
              FlexData::make_real(focus_motion_gain));
  return cfg;
}

unique_ptr<TemporalDecimationStage>
make_stage_(const SessionContextIntf* sess,
            unsigned                  num_iports,
            FlexData                  cfg)
{
  // We're testing the stage as a pure object here (no pipeline), so
  // the iports vector doesn't need real producers — empty entries
  // with the right size are enough to seed _has_iport1 once we feed
  // it through initialize via a stub RuntimeContext below. Stages
  // outside a pipeline can be safely constructed with empty InEdges.
  vector<InEdge> iports(num_iports);
  return make_unique<TemporalDecimationStage>(
      sess, "td", std::move(iports), std::move(cfg));
}

}  // namespace

TEST(temporal_decimation_stage, idle_below_threshold_only_force_keeps)
{
  // Below-threshold motion never triggers a motion-driven keep, so
  // the long-run rate is governed purely by max_consecutive_drops.
  // max_avg_fps is a CEILING, not a target — staying lower is
  // expected. With max_drops=5, the policy force-keeps every 6th
  // frame at most.
  Session sess;
  auto td = make_stage_(&sess, /*num_iports=*/1,
                        make_cfg_(/*max_avg_fps=*/30.0,
                                  /*max_drops=*/5, 0.02, 4.0));

  uint64_t kept = 0;
  uint64_t dt_us = static_cast<uint64_t>(1e6 / 60.0);
  uint64_t ts = 0;
  const int N = 400;
  for (int i = 0; i < N; ++i) {
    bool k = td->decide_keep_for_test(/*total_motion=*/0.0,
                                      /*focus_share=*/0.0, ts);
    if (k) { ++kept; }
    ts += dt_us;
  }
  // 400 / (max_drops + 1) = 400 / 6 ≈ 66.6 keeps; allow ±10 either
  // side.
  EXPECT_TRUE(kept >= 55);
  EXPECT_TRUE(kept <= 75);
}

TEST(temporal_decimation_stage, max_consecutive_drops_caps_drops)
{
  // Zero motion forever; target_fps << source_fps so the policy
  // would otherwise want to drop almost everything. The
  // max-consecutive cap must force-keep at least once every
  // (max_drops + 1) frames.
  Session sess;
  const unsigned MAX_DROPS = 3;
  auto td = make_stage_(&sess, /*num_iports=*/1,
                        make_cfg_(1.0, MAX_DROPS, 0.5, 4.0));

  uint64_t dt_us = static_cast<uint64_t>(1e6 / 60.0);   // 60 fps src
  uint64_t ts = 0;
  unsigned cur_run = 0;
  unsigned worst_run = 0;
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    bool k = td->decide_keep_for_test(0.0, 0.0, ts);
    if (k) {
      cur_run = 0;
    } else {
      ++cur_run;
      if (cur_run > worst_run) { worst_run = cur_run; }
    }
    ts += dt_us;
  }
  EXPECT_TRUE(worst_run <= MAX_DROPS);
}

TEST(temporal_decimation_stage, high_motion_capped_at_max_avg_fps)
{
  // Motion well above threshold every frame: the bucket gates the
  // keep rate to max_avg_fps. For source=60, max_avg_fps=30, N=200
  // frames over (200/60) s = 3.33 s, the cap allows ≤ 100 keeps
  // plus a small bucket-warmup burst (≤ bucket_capacity).
  Session sess;
  auto td = make_stage_(&sess, /*num_iports=*/1,
                        make_cfg_(/*max_avg_fps=*/30.0,
                                  /*max_drops=*/5, 0.02, 4.0));

  uint64_t dt_us = static_cast<uint64_t>(1e6 / 60.0);
  uint64_t ts = 0;
  uint64_t kept = 0;
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    bool k = td->decide_keep_for_test(0.10, 0.0, ts);
    if (k) { ++kept; }
    ts += dt_us;
  }
  // Cap ≈ 100; allow [95, 105] for floating-point refill rounding.
  EXPECT_TRUE(kept <= 105);
  EXPECT_TRUE(kept >= 95);
}

TEST(temporal_decimation_stage, focus_boost_lifts_low_motion_frame)
{
  // total_motion alone is below threshold; focus_motion_share is
  // high. With focus_motion_gain=4 the effective priority crosses
  // the threshold and the frame is kept.
  //
  // Use target_fps < source_fps so the bucket has meaningful
  // headroom — at target_fps == source_fps the bucket is always
  // full enough to auto-keep regardless of motion, which would
  // mask the policy under test.
  Session sess;
  auto td = make_stage_(&sess, /*num_iports=*/2,
                        make_cfg_(30.0, /*max_drops=*/5, 0.10, 4.0));

  uint64_t dt = static_cast<uint64_t>(1e6 / 60.0);  // 60 fps source

  // Warm up the bucket by accepting a few frames at the target rate
  // so we have spare tokens to spend on the boost test.
  uint64_t ts = 0;
  // 5 warmup calls so the post-warmup bucket state has just-enough
  // tokens (≈ 0.5) for the next refill to cross 1.0. This matters
  // because dt=16666us isn't an exact 1/60s, so the per-frame refill
  // rounds to 0.49998 and the bucket only accumulates to >= 1.0
  // every third call.
  for (int i = 0; i < 5; ++i) {
    (void)td->decide_keep_for_test(0.20, 0.0, ts);
    ts += dt;
  }

  // Without focus boost, this frame is below threshold (0.05 < 0.10)
  // and should drop.
  Session sess2;
  auto td_no_boost = make_stage_(&sess2, /*num_iports=*/1,
                                 make_cfg_(30.0, 5, 0.10, 4.0));
  uint64_t ts2 = 0;
  for (int i = 0; i < 5; ++i) {
    (void)td_no_boost->decide_keep_for_test(0.20, 0.0, ts2);
    ts2 += dt;
  }
  const bool drop_when_blind =
      !td_no_boost->decide_keep_for_test(0.05, 0.0, ts2);
  EXPECT_TRUE(drop_when_blind);

  // With focus_motion_share=0.5, priority becomes 0.05 + 4*0.5 = 2.05,
  // well above threshold -> keep.
  const bool keep_when_focused =
      td->decide_keep_for_test(0.05, 0.5, ts);
  EXPECT_TRUE(keep_when_focused);
}

TEST(temporal_decimation_stage, sub_unity_max_avg_fps_supported)
{
  // max_avg_fps < 1 is legal. Drive the stage with steady above-
  // threshold motion at 60 fps source and a 0.5 fps cap; expect
  // ≤ 0.5 × (N/source) ≈ 5 keeps in 600 frames, plus the bucket
  // warmup burst.
  Session sess;
  auto td = make_stage_(&sess, /*num_iports=*/1,
                        make_cfg_(/*max_avg_fps=*/0.5,
                                  /*max_drops=*/1000,
                                  /*thresh=*/0.05,
                                  /*gain=*/4.0));
  uint64_t dt = static_cast<uint64_t>(1e6 / 60.0);
  uint64_t ts = 0;
  uint64_t kept = 0;
  const int N = 600;
  for (int i = 0; i < N; ++i) {
    if (td->decide_keep_for_test(0.20, 0.0, ts)) { ++kept; }
    ts += dt;
  }
  // 0.5 × (600/60) = 5 keeps; plus initial bucket burst (cap=2).
  EXPECT_TRUE(kept >= 4);
  EXPECT_TRUE(kept <= 8);
}

TEST(temporal_decimation_stage, window_avg_respects_cap)
{
  // The user contract: the 32-frame rolling-window keep rate must
  // not exceed max_avg_fps. Allow a small overshoot for the bucket
  // burst — over a 32-source-frame window the cap is
  //   max_avg_fps × (32 / source_fps) + bucket_capacity.
  Session sess;
  auto td = make_stage_(&sess, /*num_iports=*/1,
                        make_cfg_(/*max_avg_fps=*/20.0,
                                  /*max_drops=*/8,
                                  /*thresh=*/0.05,
                                  /*gain=*/4.0));
  uint64_t dt = static_cast<uint64_t>(1e6 / 60.0);
  uint64_t ts = 0;
  std::vector<bool> kept_flags;
  kept_flags.reserve(400);
  for (int i = 0; i < 400; ++i) {
    kept_flags.push_back(td->decide_keep_for_test(0.20, 0.0, ts));
    ts += dt;
  }
  // Rolling 32-frame window cap = 20 × (32/60) + 2 = ~12.66 keeps.
  // Round up + 1 of slack for the bucket warmup transient.
  unsigned worst_window = 0;
  unsigned running = 0;
  for (int i = 0; i < 32; ++i) {
    if (kept_flags[i]) { ++running; }
  }
  worst_window = running;
  for (size_t i = 32; i < kept_flags.size(); ++i) {
    if (kept_flags[i]) { ++running; }
    if (kept_flags[i - 32]) { --running; }
    if (running > worst_window) { worst_window = running; }
  }
  EXPECT_TRUE(worst_window <= 14);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

// Helper for the integration test: emit N planar-u8 TensorBeats.
// Each frame's bytes are produced by a caller-provided callback so
// the test can drive arbitrary motion-signal patterns.
class FrameSequenceSource : public TypedStage<FrameSequenceSource> {
public:
  static constexpr const char* kTypeName = "ut-frame-sequence-source";
  using TypedStage::TypedStage;

  struct Spec {
    int H = 0;
    int W = 0;
    int N = 0;
    uint64_t start_ts_us = 0;
    uint64_t dt_us       = 0;
    // Per-frame fill byte. The whole frame is set to this value, so
    // a sequence like {0, 0, 200, 0, 200, 0} alternates "still" and
    // "big-change" frames in the eyes of the motion kernel.
    vector<uint8_t> fills;
  };

  Spec spec;

  Job process(RuntimeContext& ctx) override
  {
    if (_i >= spec.N) {
      ctx.signal_done();
      co_return;
    }
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = { 3, spec.H, spec.W };
    tb.resize_contiguous(
        static_cast<size_t>(3) * spec.H * spec.W);
    uint8_t v = (_i < static_cast<int>(spec.fills.size()))
                ? spec.fills[_i]
                : spec.fills.back();
    std::memset(tb.as_u8(), v,
                static_cast<size_t>(3) * spec.H * spec.W);
    uint64_t ts = spec.start_ts_us + _i * spec.dt_us;
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert(
        "timestamp_us", FlexData::make_uint(ts));
    ++_i;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  int _i = 0;
};

class TensorSink : public TypedStage<TensorSink> {
public:
  static constexpr const char* kTypeName = "ut-temporal-decim-sink";
  using TypedStage::TypedStage;

  vector<TensorBeat>& collected() { return _collected; }

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    if (const auto* p = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      lock_guard<mutex> g(_mu);
      _collected.push_back(static_cast<const TensorBeat&>(*p));
    }
  }

private:
  mutex              _mu;
  vector<TensorBeat> _collected;
};

}  // namespace

TEST(temporal_decimation_stage, pipeline_drops_static_frames)
{
  // Without a real MetalRuntime we can't run the kernels; skip if
  // the test host doesn't have one.
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);

  // Source: 12 frames at 60 fps source rate. Frames 0 + every frame
  // afterwards share the same fill byte → zero motion. The stage
  // must keep the first frame (no prior signature) but then drop
  // up to max_consecutive_drops in a row.
  auto src_u = make_unique<FrameSequenceSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->spec.H = 32;
  src_u->spec.W = 48;
  src_u->spec.N = 12;
  src_u->spec.start_ts_us = 0;
  src_u->spec.dt_us = static_cast<uint64_t>(1e6 / 60.0);
  src_u->spec.fills = {
      100, 100, 100, 100, 100, 100,
      100, 100, 100, 100, 100, 100
  };
  src_u->allocate_oports(1);
  auto* src = static_cast<FrameSequenceSource*>(
      pl->insert_stage(std::move(src_u)));

  // max_avg_fps = 20 (the ceiling), max_drops = 5 (the floor), big
  // motion threshold so motion-driven keeps never fire — the only
  // keeps we'll see are the unavoidable first frame plus force-keeps
  // every 6th frame after that.
  FlexData cfg = FlexData::make_object();
  {
    auto r = cfg.as_object();
    r.insert("max_avg_fps",           FlexData::make_real(20.0));
    r.insert("max_consecutive_drops", FlexData::make_uint(5));
    r.insert("motion_threshold",      FlexData::make_real(0.9));
    r.insert("focus_motion_gain",     FlexData::make_real(4.0));
    r.insert("tile_w",                FlexData::make_int(16));
    r.insert("tile_h",                FlexData::make_int(9));
  }
  auto td_u = make_unique<TemporalDecimationStage>(
      &sess, "td", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* td = static_cast<TemporalDecimationStage*>(
      pl->insert_stage(std::move(td_u)));

  auto sink_u = make_unique<TensorSink>(
      &sess, "sink", vector<InEdge>{{td, 0}},
      FlexData::make_object());
  auto* sink = static_cast<TensorSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // First frame always kept (no prior signature -> force-keep).
  // After that, only the max_consecutive_drops=5 force-keep fires:
  // for N=12 frames that's ceil((N-1)/6) = 2 additional keeps. Allow
  // a ±1 envelope around the expected 3 keeps.
  const auto& got = sink->collected();
  EXPECT_TRUE(got.size() >= 2);
  EXPECT_TRUE(got.size() <= 4);
  EXPECT_TRUE(td->kept_count() == got.size());

  // First kept frame must carry decim_kept=true in its sideband.
  ASSERT_TRUE(!got.empty());
  const TensorBeat& f0 = got.front();
  ASSERT_TRUE(f0.sideband.is_object());
  auto sb0 = f0.sideband.as_object();
  EXPECT_TRUE(sb0.contains("decim_kept"));
  EXPECT_TRUE(sb0.at("decim_kept").as_bool(false));
}

TEST(temporal_decimation_stage, pipeline_keeps_high_motion_frames)
{
  Session sess;
  if (!sess.metal_compute() || !sess.metal_compute()->valid()) {
    return;
  }

  auto pl = make_unique<Pipeline>("p", &sess);

  // 10 frames alternating fill {0, 255, 0, 255, ...} → every frame
  // (after the first) is maximally different from its predecessor.
  // With motion_threshold low and max_avg_fps == source_fps the
  // policy keeps every frame that has motion above the threshold.
  auto src_u = make_unique<FrameSequenceSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->spec.H = 32;
  src_u->spec.W = 48;
  src_u->spec.N = 10;
  src_u->spec.start_ts_us = 0;
  src_u->spec.dt_us = static_cast<uint64_t>(1e6 / 60.0);
  src_u->spec.fills = { 0, 255, 0, 255, 0, 255, 0, 255, 0, 255 };
  src_u->allocate_oports(1);
  auto* src = static_cast<FrameSequenceSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  {
    auto r = cfg.as_object();
    r.insert("max_avg_fps",           FlexData::make_real(60.0));
    r.insert("max_consecutive_drops", FlexData::make_uint(5));
    r.insert("motion_threshold",      FlexData::make_real(0.05));
    r.insert("focus_motion_gain",     FlexData::make_real(4.0));
    r.insert("tile_w",                FlexData::make_int(16));
    r.insert("tile_h",                FlexData::make_int(9));
  }
  auto td_u = make_unique<TemporalDecimationStage>(
      &sess, "td", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* td = static_cast<TemporalDecimationStage*>(
      pl->insert_stage(std::move(td_u)));

  auto sink_u = make_unique<TensorSink>(
      &sess, "sink", vector<InEdge>{{td, 0}},
      FlexData::make_object());
  auto* sink = static_cast<TensorSink*>(
      pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  const auto& got = sink->collected();
  // At max_avg_fps == source_fps with high motion, expect ~10 kept.
  EXPECT_TRUE(got.size() >= 9);
  EXPECT_TRUE(td->kept_count() == got.size());

  // The second kept frame should have a meaningfully large
  // decim_motion_score (alternating 0/255 patches → ~1.0).
  if (got.size() >= 2) {
    const TensorBeat& f1 = got[1];
    ASSERT_TRUE(f1.sideband.is_object());
    auto sb1 = f1.sideband.as_object();
    ASSERT_TRUE(sb1.contains("decim_motion_score"));
    const double sc = sb1.at("decim_motion_score").as_real(-1.0);
    EXPECT_TRUE(sc > 0.5);
  }
}

#endif  // VPIPE_BUILD_APPLE_SILICON
