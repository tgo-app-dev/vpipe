// temporal-resample: frames in at one rate, frames out at another.
//
// Two things here are not observable from a beat count, and both get
// their own assertion.
//
// The first is ALIASING, which is the reason the stage exists. Every
// `method` produces the right NUMBER of frames; what separates them is
// what the discarded frames did to the ones that were kept, so the
// end-to-end tests feed an alternating black/white source -- a signal at
// exactly the Nyquist rate of the decimation -- and read the pixels
// back. `nearest` keeps one phase and reports pure black or pure white;
// `average` reports the mean. That difference IS the anti-aliasing.
//
// The second is the CLOCK DOMAIN, which is decided at launch from a
// config key and is invisible at runtime until a feedback loop refuses
// to close.
#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/clock-domain.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/temporal-resample-stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using Method = TemporalResampleStage::Method;

namespace {

bool
have_avfilter_(Session& s)
{
  try {
    const FFmpegLibraries* l = s.ffmpeg_libraries();
    return l != nullptr && l->valid() && l->avfilter().valid();
  } catch (...) {
    return false;
  }
}

class BeatSource : public TypedStage<BeatSource> {
public:
  static constexpr const char* kTypeName = "ut-tr-source";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    auto p = make_payload<TensorBeatPayload>(TensorBeat(beats[next++]));
    co_await ctx.write(0, std::move(p));
    co_return;
  }
};

class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-tr-sink";
  using TypedStage::TypedStage;
  int beats = 0;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<std::uint8_t> first_px;      // one leading byte per beat
  std::vector<std::uint8_t> bytes;         // the LAST beat's payload
  FlexData last_sb;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      ++beats;
      shapes.push_back(tb->shape);
      last_sb = tb->sideband;
      bytes.assign(tb->data.begin(), tb->data.end());
      if (!tb->data.empty()) { first_px.push_back(tb->data[0]); }
    }
    co_return;
  }
};

TensorBeat
frame_(int h, int w, std::uint8_t fill, unsigned fps_num, unsigned fps_den,
       std::uint64_t ts_us)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, h, w};
  tb.data.resize((std::size_t)3 * h * w);
  std::memset(tb.data.data(), fill, tb.data.size());
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("fps_num", FlexData::make_uint(fps_num));
  o.as_object().insert_or_assign("fps_den", FlexData::make_uint(fps_den));
  o.as_object().insert_or_assign("timestamp_us", FlexData::make_uint(ts_us));
  tb.sideband = std::move(o);
  return tb;
}

// A [T, 3, H, W] clip whose frame t is filled with fill(t).
TensorBeat
clip_(int t, int h, int w, double fps, std::uint8_t (*fill)(int))
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {t, 3, h, w};
  const std::size_t per = (std::size_t)3 * h * w;
  tb.data.resize((std::size_t)t * per);
  for (int i = 0; i < t; ++i) {
    std::memset(tb.data.data() + (std::size_t)i * per, fill(i), per);
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("fps", FlexData::make_real(fps));
  o.as_object().insert_or_assign("stacked", FlexData::make_int(t));
  o.as_object().insert_or_assign("timestamp_us", FlexData::make_uint(0));
  tb.sideband = std::move(o);
  return tb;
}

struct Run {
  Session sess;
  std::unique_ptr<Pipeline> pl;
  Sink* sink = nullptr;
  TemporalResampleStage* stage = nullptr;
};

bool
drive_(Run& r, std::vector<TensorBeat> beats, FlexData cfg)
{
  r.pl = std::make_unique<Pipeline>("p", &r.sess);
  auto src_u = std::make_unique<BeatSource>(&r.sess, "src",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  src_u->beats = std::move(beats);
  auto* src = static_cast<BeatSource*>(r.pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  auto st_u = std::make_unique<TemporalResampleStage>(
      &r.sess, "tr", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  r.stage = static_cast<TemporalResampleStage*>(
      r.pl->insert_stage(std::move(st_u)));
  auto sink_u = std::make_unique<Sink>(&r.sess, "sink",
                                       std::vector<InEdge>{{r.stage, 0}},
                                       FlexData::make_object());
  r.sink = static_cast<Sink*>(r.pl->insert_stage(std::move(sink_u)));
  PipelineRuntime rt(r.pl.get(), &r.sess);
  if (!rt.launch()) { return false; }
  rt.wait_idle();
  rt.stop();
  return true;
}

FlexData
cfg_(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData o = FlexData::make_object();
  for (auto& p : kv) { o.as_object().insert_or_assign(p.first, p.second); }
  return o;
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

// The chain is the whole behaviour, and it is a string: it can be
// asserted without FFmpeg, a runtime or a frame.
TEST(temporal_resample, the_chain_says_which_filter_does_the_work)
{
  using S = std::string;
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kNearest, 30, 15, 0)
              == S("fps=15/1"));
  // 29.97 must reach the chain as 30000/1001, not as a decimal: `fps`
  // divides by it.
  EXPECT_TRUE(TemporalResampleStage::chain_for(
                  Method::kNearest, 60, 30000.0 / 1001.0, 0)
              == S("fps=30000/1001"));
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kBlend, 30, 24, 0)
              == S("framerate=fps=24/1"));
  // `average` sizes its box from the RATIO -- the interval one output
  // frame covers -- so 60 -> 15 averages 4 source frames.
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kAverage, 60, 15, 0)
              == S("tmix=frames=4,fps=15/1"));
  // ...and an explicit window overrides that, which is a shutter angle.
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kAverage, 60, 15, 2)
              == S("tmix=frames=2,fps=15/1"));
  // Below two frames there is nothing to average and tmix would be an
  // identity that still costs a pass over every pixel.
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kAverage, 30, 24, 0)
              == S("fps=24/1"));
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kMotion, 24, 60, 0)
                  .rfind("minterpolate=fps=60/1", 0) == 0);
  // A no-op `nearest` is EMPTY, which is how the stage knows it may
  // forward the beat rather than pay for a graph.
  EXPECT_TRUE(TemporalResampleStage::chain_for(Method::kNearest, 24, 24, 0)
              .empty());
  // ...but the filtering methods still have work at equal rates. They
  // are filters, not just resamplers.
  EXPECT_FALSE(TemporalResampleStage::chain_for(Method::kAverage, 48, 24, 0)
               .empty());
  std::printf("[temporal_resample] average 60->15: %s\n",
              TemporalResampleStage::chain_for(Method::kAverage, 60, 15, 0)
                  .c_str());
}

TEST(temporal_resample, the_surface_and_what_it_refuses)
{
  Session s;
  auto make = [&](FlexData c) {
    return std::string(
        TemporalResampleStage(&s, "x", std::vector<InEdge>{}, std::move(c))
            .config_error());
  };
  // output_fps is the whole request; without it the stage would be a
  // pass-through wearing a resampler's name.
  EXPECT_FALSE(make(cfg_({})).empty());
  EXPECT_FALSE(make(cfg_({{"output_fps", FlexData::make_real(0.0)}})).empty());
  EXPECT_FALSE(make(cfg_({{"output_fps", FlexData::make_real(-1.0)}})).empty());
  EXPECT_TRUE(make(cfg_({{"output_fps", FlexData::make_real(24.0)}})).empty());
  // A misspelt method must not silently become the default: the whole
  // point of the key is which filter runs.
  EXPECT_FALSE(make(cfg_({{"output_fps", FlexData::make_real(24.0)},
                          {"method", FlexData::make_string("bilinear")}}))
                   .empty());
  for (const char* m : {"nearest", "blend", "average", "motion"}) {
    EXPECT_TRUE(make(cfg_({{"output_fps", FlexData::make_real(24.0)},
                           {"method", FlexData::make_string(m)}}))
                    .empty());
  }
  EXPECT_FALSE(make(cfg_({{"output_fps", FlexData::make_real(24.0)},
                          {"average_frames", FlexData::make_int(-1)}}))
                   .empty());

  TemporalResampleStage st(&s, "y", std::vector<InEdge>{},
                           cfg_({{"output_fps", FlexData::make_real(24.0)}}));
  const StageSpec& sp = st.spec();
  EXPECT_TRUE(std::string(sp.type_name) == "temporal-resample");
  EXPECT_TRUE(sp.iports.size() == 1 && sp.oports.size() == 1);
  EXPECT_TRUE(std::string(sp.iports[0].name) == "frames");
  EXPECT_TRUE(std::string(sp.oports[0].name) == "frames");
  std::printf("[temporal_resample] %zu iports / %zu oports\n",
              sp.iports.size(), sp.oports.size());
}

// The clock-domain rule, read off the ANALYSIS rather than off the
// stage: a launched graph is what the answer has to be right in.
TEST(temporal_resample, stacked_decides_whether_a_domain_is_crossed)
{
  auto domains = [](bool stacked) {
    Session sess;
    Pipeline pl("p", &sess);
    auto src_u = std::make_unique<BeatSource>(&sess, "src",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
    auto* src = static_cast<BeatSource*>(pl.insert_stage(std::move(src_u)));
    src->allocate_oports(1);
    auto st_u = std::make_unique<TemporalResampleStage>(
        &sess, "tr", std::vector<InEdge>{{src, 0}},
        cfg_({{"output_fps", FlexData::make_real(15.0)},
              {"stacked", FlexData::make_bool(stacked)}}));
    auto* st = static_cast<TemporalResampleStage*>(
        pl.insert_stage(std::move(st_u)));
    auto sink_u = std::make_unique<Sink>(&sess, "sink",
                                         std::vector<InEdge>{{st, 0}},
                                         FlexData::make_object());
    auto* sink = static_cast<Sink*>(pl.insert_stage(std::move(sink_u)));
    const std::vector<Stage*> stages = {src, st, sink};
    const auto a = compute_clock_domains(stages);
    const unsigned din =
        a.port_domain.at(PortKey{st, PortKey::Kind::In, 0});
    const unsigned dout =
        a.port_domain.at(PortKey{st, PortKey::Kind::Out, 0});
    return std::pair<unsigned, unsigned>{din, dout};
  };
  // A frame stream changes the BEAT rate, so the output is on a clock
  // of its own and the analysis has to see two domains.
  const auto stream = domains(false);
  EXPECT_TRUE(stream.first != stream.second);
  // One clip in, one clip out: nothing is crossed, so a feedback loop
  // can close through it.
  const auto stacked = domains(true);
  EXPECT_TRUE(stacked.first == stacked.second);
  std::printf("[temporal_resample] stream domains %u/%u, stacked %u/%u\n",
              stream.first, stream.second, stacked.first, stacked.second);
}

TEST(temporal_resample, a_frame_stream_halves_and_restamps)
{
  Run r;
  if (!have_avfilter_(r.sess)) { return; }
  std::vector<TensorBeat> beats;
  for (int i = 0; i < 30; ++i) {
    beats.push_back(frame_(8, 16, (std::uint8_t)(i * 8), 30, 1,
                           1000000ULL + (std::uint64_t)i * 33333));
  }
  ASSERT_TRUE(drive_(r, std::move(beats),
                     cfg_({{"output_fps", FlexData::make_real(15.0)}})));
  EXPECT_TRUE(r.sink->beats >= 14 && r.sink->beats <= 16);
  ASSERT_TRUE(!r.sink->shapes.empty());
  EXPECT_TRUE(r.sink->shapes[0].size() == 3);
  EXPECT_TRUE(r.sink->shapes[0][0] == 3 && r.sink->shapes[0][1] == 8 &&
              r.sink->shapes[0][2] == 16);
  // The sideband states the rate the beats ACTUALLY have now. A sink
  // adopting the source's 30 would encode a clip at double speed.
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "fps_num") == 15.0);
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "fps_den") == 1.0);
  // ...and `fps`/`stacked` must be GONE: a frame is not a clip.
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "fps") == 0.0);
  std::printf("[temporal_resample] 30 frames @30 -> %d @15, last ts %.0f\n",
              r.sink->beats, sb_real_(r.sink->last_sb, "timestamp_us"));
}

TEST(temporal_resample, a_stacked_clip_is_one_beat_in_one_beat_out)
{
  Run r;
  if (!have_avfilter_(r.sess)) { return; }
  std::vector<TensorBeat> beats;
  beats.push_back(clip_(24, 8, 16, 24.0,
                        [](int i) { return (std::uint8_t)(i * 4); }));
  ASSERT_TRUE(drive_(r, std::move(beats),
                     cfg_({{"output_fps", FlexData::make_real(12.0)},
                           {"stacked", FlexData::make_bool(true)}})));
  EXPECT_TRUE(r.sink->beats == 1);
  ASSERT_TRUE(r.sink->shapes.size() == 1);
  ASSERT_TRUE(r.sink->shapes[0].size() == 4);
  // One second at 24 in, one second at 12 out.
  EXPECT_TRUE(r.sink->shapes[0][0] >= 11 && r.sink->shapes[0][0] <= 13);
  EXPECT_TRUE(r.sink->shapes[0][1] == 3 && r.sink->shapes[0][2] == 8 &&
              r.sink->shapes[0][3] == 16);
  // A clip states `fps`, not the frame pair, and republishes `stacked`.
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "fps") == 12.0);
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "stacked") ==
              (double)r.sink->shapes[0][0]);
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "fps_num") == 0.0);
  std::printf("[temporal_resample] clip 24 -> %lld frames\n",
              (long long)r.sink->shapes[0][0]);
}

// f32 is `video-to-rgb`'s DEFAULT output, so a stage that took only u8
// would refuse the one producer it exists downstream of -- which is how
// this was found: the first real pipeline said "frames must be planar
// u8 RGB, got f32" and forwarded nothing. GBRPF32 is the same three
// planes in float, so the difference is the element size and nothing
// else, and the values have to survive it exactly.
TEST(temporal_resample, f32_frames_are_filtered_as_float_not_reread_as_bytes)
{
  Run r;
  if (!have_avfilter_(r.sess)) { return; }
  // A [0,1] normalised clip -- what video-to-rgb emits with its default
  // `normalize`. Frame t is the constant t/32.
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {32, 3, 4, 8};
  const std::size_t per = (std::size_t)3 * 4 * 8;
  tb.resize_contiguous(32 * per);
  float* d = tb.as_f32();
  for (int i = 0; i < 32; ++i) {
    for (std::size_t k = 0; k < per; ++k) {
      d[(std::size_t)i * per + k] = (float)i / 32.0f;
    }
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("fps", FlexData::make_real(32.0));
  o.as_object().insert_or_assign("timestamp_us", FlexData::make_uint(0));
  tb.sideband = std::move(o);

  ASSERT_TRUE(drive_(r, {tb},
                     cfg_({{"output_fps", FlexData::make_real(16.0)},
                           {"stacked", FlexData::make_bool(true)}})));
  ASSERT_TRUE(r.sink->beats == 1 && r.sink->shapes.size() == 1);
  ASSERT_TRUE(r.sink->shapes[0].size() == 4);
  // The dtype is CARRIED, not converted: a u8 answer here would be a
  // 4x smaller beat that still looked like a clip.
  EXPECT_TRUE(r.sink->bytes.size() ==
              (std::size_t)r.sink->shapes[0][0] * per * sizeof(float));
  const std::int64_t t = r.sink->shapes[0][0];
  EXPECT_TRUE(t >= 15 && t <= 17);
  // Every kept frame is one of the source's own constants, in [0, 1).
  // Reading float bytes as u8 would land far outside it.
  const float* out = reinterpret_cast<const float*>(r.sink->bytes.data());
  double lo = 1e9, hi = -1e9;
  for (std::int64_t i = 0; i < t; ++i) {
    const double v = (double)out[(std::size_t)i * per];
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  EXPECT_TRUE(lo >= 0.0 && hi < 1.0);
  // ...and the range spans the clip rather than collapsing to one
  // value, so the decimation kept different frames and not one.
  EXPECT_TRUE(hi - lo > 0.8);
  std::printf("[temporal_resample] f32 clip 32 -> %lld frames, values "
              "%.4f..%.4f\n", (long long)t, lo, hi);
}

// The point of the stage. The source alternates black and white every
// frame -- a signal exactly at the Nyquist rate of a 2:1 decimation --
// so `nearest` locks onto one phase and every output frame is an
// EXTREME, while `average` covers the pair and reports their mean. Any
// correct anti-aliasing collapses this signal; no correct nearest-
// neighbour does. It is the one assertion a beat count cannot make.
TEST(temporal_resample, averaging_is_visible_in_the_pixels)
{
  {
    Session probe;
    if (!have_avfilter_(probe)) { return; }
  }
  // How many output frames are still at an EXTREME, and their mean.
  // One 32-frame 48 fps clip of alternating black and white, decimated
  // to 24.
  auto measure = [](const char* method, double* mean, int* extremes,
                    int* total) {
    Run r;
    std::vector<TensorBeat> beats;
    beats.push_back(clip_(32, 8, 16, 48.0, [](int i) {
      return (std::uint8_t)(i % 2 == 0 ? 0 : 255);
    }));
    if (!drive_(r, std::move(beats),
                cfg_({{"output_fps", FlexData::make_real(24.0)},
                      {"method", FlexData::make_string(method)},
                      {"stacked", FlexData::make_bool(true)}}))) {
      return false;
    }
    if (r.sink->beats != 1 || r.sink->shapes.size() != 1 ||
        r.sink->shapes[0].size() != 4) {
      return false;
    }
    const std::int64_t t = r.sink->shapes[0][0];
    const std::size_t per = (std::size_t)3 * r.sink->shapes[0][2] *
                            r.sink->shapes[0][3];
    if (t <= 0 || r.sink->bytes.size() < (std::size_t)t * per) {
      return false;
    }
    double sum = 0.0;
    int n = 0;
    for (std::int64_t i = 0; i < t; ++i) {
      const double v = (double)r.sink->bytes[(std::size_t)i * per];
      sum += v;
      if (std::fabs(v - 127.5) > 64.0) { ++n; }
    }
    *mean = sum / (double)t;
    *extremes = n;
    *total = (int)t;
    return true;
  };

  double nm = 0.0, am = 0.0;
  int nx = 0, ax = 0, nt = 0, at = 0;
  ASSERT_TRUE(measure("nearest", &nm, &nx, &nt));
  ASSERT_TRUE(measure("average", &am, &ax, &at));
  ASSERT_TRUE(nt > 4 && at > 4);
  // Nearest locks onto one phase of the flicker: EVERY frame is at an
  // extreme, and the mean is that phase rather than the picture's.
  EXPECT_TRUE(nx == nt);
  EXPECT_TRUE(nm < 32.0 || nm > 223.0);
  // Averaging covers the pair, so the frames sit on the mean. The one
  // exception is the FIRST: tmix's window is not full yet, so its first
  // output is the first source frame alone. That is the filter's
  // start-up and not a miss -- the assertion allows exactly one.
  EXPECT_TRUE(ax <= 1);
  EXPECT_TRUE(am > 96.0 && am < 160.0);
  std::printf("[temporal_resample] flicker at Nyquist: nearest %d/%d "
              "frames extreme (mean %.1f), average %d/%d (mean %.1f)\n",
              nx, nt, nm, ax, at, am);
}
