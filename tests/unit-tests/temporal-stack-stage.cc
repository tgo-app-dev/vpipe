// temporal-stack: many time-slice beats -> one tensor.
//
// The axis is the whole test. Both wrong answers produce a valid tensor
// of the right rank -- planar stereo concatenated on axis 0 is
// [2+2, N], video stacked on the last axis is [3, H, W*T] -- so nothing
// downstream would fault on either. Only the numbers say which happened,
// which is why these tests read the payload back and not just the shape.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/temporal-stack-stage.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// Emits a fixed list of beats, then ends.
class BeatSource : public TypedStage<BeatSource> {
public:
  static constexpr const char* kTypeName = "ut-ts-source";
  using TypedStage::TypedStage;

  std::vector<TensorBeat> beats;
  std::size_t next = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    TensorBeat tb = beats[next++];
    auto payload = make_payload<TensorBeatPayload>(std::move(tb));
    co_await ctx.write(0, std::move(payload));
    co_return;
  }
};

class StackSink : public TypedStage<StackSink> {
public:
  static constexpr const char* kTypeName = "ut-ts-sink";
  using TypedStage::TypedStage;

  int groups = 0;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
  FlexData sideband;
  // Every group, not just the last: an overlap is a claim about what
  // CONSECUTIVE groups share, so one group's contents cannot show it.
  // `fills` is the leading byte of each frame in a video group, which
  // frame_() sets to a per-frame marker.
  std::vector<std::vector<std::int64_t>> all_shapes;
  std::vector<std::vector<std::uint8_t>> fills;
  std::vector<std::uint64_t> group_ts;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      ++groups;
      shape = tb->shape;
      bytes.assign(tb->data.begin(), tb->data.end());
      sideband = tb->sideband;
      all_shapes.push_back(tb->shape);
      std::vector<std::uint8_t> f;
      if (tb->shape.size() == 4 && !tb->data.empty()) {
        const std::size_t per = (std::size_t)tb->shape[1] * tb->shape[2]
                                * tb->shape[3];
        for (std::int64_t k = 0; k < tb->shape[0]; ++k) {
          const std::size_t off = (std::size_t)k * per;
          if (off < tb->data.size()) { f.push_back(tb->data[off]); }
        }
      }
      fills.push_back(std::move(f));
      std::uint64_t ts = 0;
      if (tb->sideband.is_object()) {
        FlexData sb = tb->sideband;
        auto o = sb.as_object();
        if (o.contains("timestamp_us")) {
          ts = (std::uint64_t)o.at("timestamp_us").as_uint(0);
        }
      }
      group_ts.push_back(ts);
    }
    co_return;
  }
};

TensorBeat
frame_(int h, int w, std::uint8_t fill, std::uint64_t ts_us,
       unsigned fps_num = 0, unsigned fps_den = 0)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, h, w};
  tb.data.resize((std::size_t)3 * h * w);
  std::memset(tb.data.data(), fill, tb.data.size());
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("timestamp_us", FlexData::make_uint(ts_us));
  if (fps_num > 0 && fps_den > 0) {
    o.as_object().insert_or_assign("fps_num", FlexData::make_uint(fps_num));
    o.as_object().insert_or_assign("fps_den", FlexData::make_uint(fps_den));
  }
  tb.sideband = std::move(o);
  return tb;
}

TensorBeat
pcm_(int channels, int n, float base, int sr)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = channels == 1 ? std::vector<std::int64_t>{n}
                           : std::vector<std::int64_t>{channels, n};
  tb.data.resize((std::size_t)channels * n * sizeof(float));
  auto* d = reinterpret_cast<float*>(tb.data.data());
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < n; ++i) { d[c * n + i] = base + c * 100.0f + i; }
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("sample_rate", FlexData::make_int(sr));
  o.as_object().insert_or_assign("timestamp_us", FlexData::make_uint(0));
  tb.sideband = std::move(o);
  return tb;
}

// Run source -> temporal-stack -> sink and hand back the sink.
struct Run {
  Session sess;
  std::unique_ptr<Pipeline> pl;
  StackSink* sink = nullptr;
};

// Returns false when the LAUNCH was refused -- the edges are checked by
// payload type and tag, so a mismatch fails here and not at a beat.
bool
drive_(Run& r, std::vector<TensorBeat> beats, FlexData cfg)
{
  r.pl = std::make_unique<Pipeline>("p", &r.sess);
  auto src_u = std::make_unique<BeatSource>(&r.sess, "src",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  src_u->beats = std::move(beats);
  auto* src = static_cast<BeatSource*>(r.pl->insert_stage(std::move(src_u)));
  // A writing stage has to be given its oports; TypedStage does not
  // infer them for a test stage with no spec of its own.
  src->allocate_oports(1);
  auto ts_u = std::make_unique<TemporalStackStage>(
      &r.sess, "ts", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* ts =
      static_cast<TemporalStackStage*>(r.pl->insert_stage(std::move(ts_u)));
  auto sink_u = std::make_unique<StackSink>(&r.sess, "sink",
                                            std::vector<InEdge>{{ts, 0}},
                                            FlexData::make_object());
  r.sink = static_cast<StackSink*>(r.pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(r.pl.get(), &r.sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  return true;
}

double
sb_real_(const FlexData& sb, const char* k)
{
  if (!sb.is_object()) { return 0.0; }
  FlexData s = sb;
  auto o = s.as_object();
  if (!o.contains(k)) { return 0.0; }
  FlexData v = o.at(k);
  if (v.is_real()) { return v.as_real(0.0); }
  if (v.is_int())  { return (double)v.as_int(0); }
  if (v.is_uint()) { return (double)v.as_uint(0); }
  return 0.0;
}

}  // namespace

// Rank, dtype and a stated sample rate, in that order of authority.
TEST(temporal_stack, sensing_reads_the_modality_off_the_beat)
{
  using Mode = TemporalStackStage::Mode;
  EXPECT_TRUE(TemporalStackStage::sense(frame_(4, 4, 0, 0)) == Mode::kVideo);
  EXPECT_TRUE(TemporalStackStage::sense(pcm_(1, 8, 0, 16000)) == Mode::kAudio);
  EXPECT_TRUE(TemporalStackStage::sense(pcm_(2, 8, 0, 16000)) == Mode::kAudio);
  // A stated rate outranks the shape: audio-to-pcm always states one, and
  // a producer saying what a beat IS beats any guess from its extents.
  {
    TensorBeat tb = frame_(4, 4, 0, 0);
    tb.sideband.as_object().insert_or_assign("sample_rate",
                                             FlexData::make_int(48000));
    EXPECT_TRUE(TemporalStackStage::sense(tb) == Mode::kAudio);
  }
  // Neither shape: generic, which stacks on a new axis without pretending
  // to know what the axis means.
  {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {2, 3, 4};
    tb.data.resize(2 * 3 * 4 * sizeof(float));
    EXPECT_TRUE(TemporalStackStage::sense(tb) == Mode::kGeneric);
  }
  std::printf("[temporal_stack] sensing: rank + dtype, and a stated rate "
              "wins\n");
}

// Video: a NEW leading axis, frames intact and in order.
TEST(temporal_stack, video_stacks_on_a_new_leading_axis)
{
  Run r;
  std::vector<TensorBeat> beats;
  for (int i = 0; i < 5; ++i) {
    // 24000/1001 = 23.976: a rate that a "looks like 24" guess gets wrong.
    beats.push_back(frame_(2, 3, (std::uint8_t)(10 + i),
                           (std::uint64_t)i * 41708, 24000, 1001));
  }
  ASSERT_TRUE(drive_(r, std::move(beats), FlexData::make_object()));
  ASSERT_TRUE(r.sink != nullptr);
  EXPECT_TRUE(r.sink->groups == 1);
  ASSERT_TRUE(r.sink->shape.size() == 4);
  EXPECT_TRUE(r.sink->shape[0] == 5);
  EXPECT_TRUE(r.sink->shape[1] == 3 && r.sink->shape[2] == 2 &&
              r.sink->shape[3] == 3);
  // Frame k is wholly present at offset k -- the check that separates a
  // real stack from a [3, H, W*T] wide frame, which has the same bytes in
  // a different order.
  const std::size_t plane = (std::size_t)3 * 2 * 3;
  ASSERT_TRUE(r.sink->bytes.size() == plane * 5);
  bool ok = true;
  for (int i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < plane; ++j) {
      if (r.sink->bytes[(std::size_t)i * plane + j] !=
          (std::uint8_t)(10 + i)) {
        ok = false;
      }
    }
  }
  EXPECT_TRUE(ok);
  // The rate the frames declared, not one inferred from their spacing.
  const double fps = sb_real_(r.sink->sideband, "fps");
  std::printf("[temporal_stack] video -> [%lld,%lld,%lld,%lld] @ %.4f fps\n",
              (long long)r.sink->shape[0], (long long)r.sink->shape[1],
              (long long)r.sink->shape[2], (long long)r.sink->shape[3], fps);
  EXPECT_TRUE(fps > 23.97 && fps < 23.98);
}

// No declared rate: derived from the span of the timestamps rather than
// left blank or guessed at 24.
TEST(temporal_stack, an_undeclared_rate_is_derived_from_timestamps)
{
  Run r;
  std::vector<TensorBeat> beats;
  for (int i = 0; i < 5; ++i) {          // 20 ms apart == 50 fps
    beats.push_back(frame_(2, 3, 1, (std::uint64_t)i * 20000));
  }
  ASSERT_TRUE(drive_(r, std::move(beats), FlexData::make_object()));
  const double fps = sb_real_(r.sink->sideband, "fps");
  std::printf("[temporal_stack] undeclared rate -> %.3f fps from "
              "timestamps\n", fps);
  EXPECT_TRUE(fps > 49.9 && fps < 50.1);
}

// Audio mono: concatenated on the existing axis, samples in order.
TEST(temporal_stack, audio_concatenates_on_the_last_axis)
{
  Run r;
  std::vector<TensorBeat> beats;
  for (int k = 0; k < 3; ++k) { beats.push_back(pcm_(1, 4, k * 10.0f, 8000)); }
  ASSERT_TRUE(drive_(r, std::move(beats), FlexData::make_object()));
  ASSERT_TRUE(r.sink->shape.size() == 1);
  EXPECT_TRUE(r.sink->shape[0] == 12);
  const auto* d = reinterpret_cast<const float*>(r.sink->bytes.data());
  bool ok = true;
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 4; ++i) {
      if (d[k * 4 + i] != k * 10.0f + i) { ok = false; }
    }
  }
  EXPECT_TRUE(ok);
  EXPECT_TRUE(sb_real_(r.sink->sideband, "sample_rate") == 8000.0);
  // 12 samples at 8 kHz is 1500 us, recomputed for the GROUP rather than
  // carried from the first chunk.
  EXPECT_TRUE(sb_real_(r.sink->sideband, "duration_us") == 1500.0);
  std::printf("[temporal_stack] audio mono -> [12], 1500 us\n");
}

// Planar stereo: [2, n] x k -> [2, sum n], each channel WHOLE.
//
// The chunks arrive channel-major within a chunk and chunk-major
// overall, so a plain byte-append gives L0 R0 L1 R1... under a [2, sum n]
// header -- half of channel 0 would be channel 1's first chunk. Only
// reading the samples back catches it.
TEST(temporal_stack, planar_stereo_keeps_each_channel_whole)
{
  Run r;
  std::vector<TensorBeat> beats;
  for (int k = 0; k < 3; ++k) { beats.push_back(pcm_(2, 4, k * 10.0f, 8000)); }
  ASSERT_TRUE(drive_(r, std::move(beats), FlexData::make_object()));
  ASSERT_TRUE(r.sink->shape.size() == 2);
  EXPECT_TRUE(r.sink->shape[0] == 2 && r.sink->shape[1] == 12);
  const auto* d = reinterpret_cast<const float*>(r.sink->bytes.data());
  bool ok = true;
  for (int c = 0; c < 2; ++c) {
    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 4; ++i) {
        const float want = k * 10.0f + c * 100.0f + i;
        if (d[c * 12 + k * 4 + i] != want) { ok = false; }
      }
    }
  }
  std::printf("[temporal_stack] planar stereo -> [2,12], channels whole: "
              "%s\n", ok ? "yes" : "NO");
  EXPECT_TRUE(ok);
}

// group_size emits mid-stream; the remainder still comes out at EOS.
TEST(temporal_stack, group_size_emits_without_waiting_for_the_end)
{
  Run r;
  std::vector<TensorBeat> beats;
  for (int i = 0; i < 5; ++i) { beats.push_back(frame_(2, 2, 1, 0)); }
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("group_size", FlexData::make_int(2));
  ASSERT_TRUE(drive_(r, std::move(beats), std::move(cfg)));
  // 2 + 2 mid-stream, then the odd one at EOS: a partial group is still a
  // group, and dropping it would silently shorten the stream.
  std::printf("[temporal_stack] group_size 2 over 5 beats -> %d group(s), "
              "last %lld\n", r.sink->groups,
              r.sink->shape.empty() ? -1 : (long long)r.sink->shape[0]);
  EXPECT_TRUE(r.sink->groups == 3);
  EXPECT_TRUE(!r.sink->shape.empty() && r.sink->shape[0] == 1);
}

// The sideband merge: something about the GROUP that no producer of its
// beats could state.
//
// The motivating case is MiniMax-H3's reference ports, which read
// `attach: true` on an audio beat to fold it onto the preceding
// reference as its soundtrack. audio-to-pcm has no idea that concept
// exists, and neither should it.
TEST(temporal_stack, config_sideband_is_merged_last)
{
  Run r;
  std::vector<TensorBeat> beats;
  for (int k = 0; k < 2; ++k) { beats.push_back(pcm_(1, 4, 0, 8000)); }
  auto cfg = FlexData::make_object();
  auto sb = FlexData::make_object();
  sb.as_object().insert_or_assign("attach", FlexData::make_bool(true));
  // Deliberately collides with a computed key: config WINS, because a
  // user pinning a value is correcting this stage.
  sb.as_object().insert_or_assign("sample_rate", FlexData::make_int(32000));
  cfg.as_object().insert_or_assign("sideband", std::move(sb));
  ASSERT_TRUE(drive_(r, std::move(beats), std::move(cfg)));
  ASSERT_TRUE(r.sink->groups == 1);
  FlexData out = r.sink->sideband;
  ASSERT_TRUE(out.is_object());
  auto o = out.as_object();
  EXPECT_TRUE(o.contains("attach") && o.at("attach").as_bool(false));
  EXPECT_TRUE(sb_real_(r.sink->sideband, "sample_rate") == 32000.0);
  // ...and the computed keys it did not name are still there.
  EXPECT_TRUE(o.contains("duration_us"));
  std::printf("[temporal_stack] sideband merge: attach set, sample_rate "
              "overridden, duration_us kept\n");
}

// The ceiling emits what fits and stops, rather than growing without
// bound on a long source.
TEST(temporal_stack, the_ceiling_emits_and_ends)
{
  Run r;
  std::vector<TensorBeat> beats;
  // 1 MB cap, ~48 KB a frame: 21 fit, and 40 are offered.
  for (int i = 0; i < 40; ++i) { beats.push_back(frame_(128, 128, 1, 0)); }
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("max_mb", FlexData::make_int(1));
  ASSERT_TRUE(drive_(r, std::move(beats), std::move(cfg)));
  ASSERT_TRUE(r.sink->groups == 1);
  ASSERT_TRUE(r.sink->shape.size() == 4);
  const std::int64_t kept = r.sink->shape[0];
  std::printf("[temporal_stack] 1 MB ceiling over 40 frames -> kept %lld\n",
              (long long)kept);
  EXPECT_TRUE(kept > 0 && kept < 40);
  EXPECT_TRUE(r.sink->bytes.size() <= (std::size_t)1 << 20);
}

// ---- overlap: what CONSECUTIVE groups share -------------------------
//
// The point of the feature is that group k+1 opens with the tail of
// group k, so the assertion has to be about two groups at once. Every
// frame carries a distinct fill byte, so a group's contents name
// themselves.
TEST(temporal_stack, overlap_carries_beats_into_the_next_group)
{
  std::vector<TensorBeat> beats;
  for (int i = 1; i <= 6; ++i) {
    beats.push_back(frame_(2, 2, (std::uint8_t)i, (std::uint64_t)i * 1000));
  }
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("mode", FlexData::make_string("video"));
  cfg.as_object().insert_or_assign("group_size", FlexData::make_int(3));
  cfg.as_object().insert_or_assign("overlap", FlexData::make_int(1));
  cfg.as_object().insert_or_assign("fps", FlexData::make_real(24.0));

  Run r;
  ASSERT_TRUE(drive_(r, std::move(beats), std::move(cfg)));
  if (r.sink == nullptr) { return; }

  // Groups advance by (group_size - overlap) = 2, and each opens with
  // the last frame of the one before: {1,2,3} {3,4,5} {5,6}.
  EXPECT_TRUE(r.sink->groups == 3);
  if (r.sink->fills.size() != 3) {
    std::printf("[temporal_stack] overlap: %zu group(s)\n",
                r.sink->fills.size());
    return;
  }
  const std::vector<std::vector<std::uint8_t>> want = {
      {1, 2, 3}, {3, 4, 5}, {5, 6}};
  for (std::size_t g = 0; g < want.size(); ++g) {
    EXPECT_TRUE(r.sink->fills[g] == want[g]);
    if (r.sink->fills[g] != want[g]) {
      std::printf("[temporal_stack] group %zu:", g);
      for (std::uint8_t v : r.sink->fills[g]) { std::printf(" %u", v); }
      std::printf("\n");
    }
  }
  // The retained frame's TIMESTAMP comes with it. Carrying the emitted
  // group's would date each clip to a frame it no longer starts on --
  // group 1 starts on frame 3, at 3000 us.
  EXPECT_TRUE(r.sink->group_ts.size() == 3);
  if (r.sink->group_ts.size() == 3) {
    EXPECT_TRUE(r.sink->group_ts[0] == 1000);
    EXPECT_TRUE(r.sink->group_ts[1] == 3000);
    EXPECT_TRUE(r.sink->group_ts[2] == 5000);
  }
}

// An overlap that leaves no room to advance would reopen the group
// already full and emit for ever. Refused at CONFIG, where a graph can
// still be fixed, rather than clamped to something the author did not
// ask for.
TEST(temporal_stack, an_overlap_that_cannot_advance_is_refused)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("group_size", FlexData::make_int(3));
  cfg.as_object().insert_or_assign("overlap", FlexData::make_int(3));
  auto s = std::make_unique<TemporalStackStage>(
      &sess, "ts", std::vector<InEdge>{}, cfg);
  EXPECT_FALSE(s->config_error().empty());

  // One less, and it is a legal hop of 1.
  auto ok = FlexData::make_object();
  ok.as_object().insert_or_assign("group_size", FlexData::make_int(3));
  ok.as_object().insert_or_assign("overlap", FlexData::make_int(2));
  auto s2 = std::make_unique<TemporalStackStage>(
      &sess, "ts2", std::vector<InEdge>{}, ok);
  EXPECT_TRUE(s2->config_error().empty());
}

// Audio beats are variable-length on the time axis, so "the last N
// beats" is not a duration. The stage says so and stacks without an
// overlap rather than retaining a length nobody asked for.
TEST(temporal_stack, overlap_does_not_apply_to_audio)
{
  std::vector<TensorBeat> beats;
  beats.push_back(pcm_(1, 4, 0.0f, 16000));
  beats.push_back(pcm_(1, 4, 100.0f, 16000));
  beats.push_back(pcm_(1, 4, 200.0f, 16000));
  beats.push_back(pcm_(1, 4, 300.0f, 16000));
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("mode", FlexData::make_string("audio"));
  cfg.as_object().insert_or_assign("group_size", FlexData::make_int(2));
  cfg.as_object().insert_or_assign("overlap", FlexData::make_int(1));

  Run r;
  ASSERT_TRUE(drive_(r, std::move(beats), std::move(cfg)));
  if (r.sink == nullptr) { return; }

  // Two groups of two chunks each, 8 samples apiece -- NOT the 12 that
  // a retained chunk would add.
  EXPECT_TRUE(r.sink->groups == 2);
  EXPECT_TRUE(r.sink->all_shapes.size() == 2);
  for (const auto& sh : r.sink->all_shapes) {
    EXPECT_TRUE(sh.size() == 1 && sh[0] == 8);
    if (!(sh.size() == 1 && sh[0] == 8)) {
      std::printf("[temporal_stack] audio group shape %s\n",
                  sh.empty() ? "(empty)" : std::to_string(sh[0]).c_str());
    }
  }
}
