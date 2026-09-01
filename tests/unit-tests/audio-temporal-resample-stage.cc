// audio-temporal-resample: rate, duration and pitch, which are three
// knobs and not one.
//
// Every wrong composition of them produces a waveform of a plausible
// length. Speeding up by resampling and speeding up by time-stretching
// give byte-for-byte different sound and IDENTICAL sample counts; a
// pitch shift that forgot to put the duration back is a shorter clip
// that still contains the right notes. So these tests measure the PITCH
// -- from zero crossings, which is exact for a pure tone -- next to the
// duration, and the two together pin which chain ran.
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
#include "stages/audio-video/audio-temporal-resample-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using Pitch = AudioTemporalResampleStage::Pitch;

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
  static constexpr const char* kTypeName = "ut-atr-source";
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
  static constexpr const char* kTypeName = "ut-atr-sink";
  using TypedStage::TypedStage;
  int beats = 0;
  int channels = 0;
  std::vector<float> left;         // channel 0, concatenated
  FlexData last_sb;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      ++beats;
      last_sb = tb->sideband;
      const int rank = (int)tb->shape.size();
      if (rank < 1 || rank > 2) { co_return; }
      channels = rank == 2 ? (int)tb->shape[0] : 1;
      const int n = rank == 2 ? (int)tb->shape[1] : (int)tb->shape[0];
      const float* d = tb->as_f32();
      left.insert(left.end(), d, d + n);
    }
    co_return;
  }
};

TensorBeat
tone_(int channels, int n, int sr, double hz)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = channels == 1 ? std::vector<std::int64_t>{n}
                           : std::vector<std::int64_t>{channels, n};
  tb.resize_contiguous((std::size_t)channels * n);
  float* d = tb.as_f32();
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < n; ++i) {
      d[(std::size_t)c * n + i] =
          0.5f * (float)std::sin(2.0 * M_PI * hz * i / sr);
    }
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign("sample_rate", FlexData::make_int(sr));
  o.as_object().insert_or_assign("timestamp_us", FlexData::make_uint(0));
  tb.sideband = std::move(o);
  return tb;
}

// A pure tone's frequency, from zero crossings. The edges are skipped:
// a WSOLA stretch and a resample both leave a partial period there.
double
hz_(const std::vector<float>& x, int rate)
{
  if ((int)x.size() < rate / 8) { return 0.0; }
  const std::size_t lo = x.size() / 8, hi = x.size() - x.size() / 8;
  int cross = 0;
  for (std::size_t i = lo + 1; i < hi; ++i) {
    if ((x[i - 1] < 0.0f) != (x[i] < 0.0f)) { ++cross; }
  }
  return 0.5 * (double)cross * (double)rate / (double)(hi - lo);
}

struct Run {
  Session sess;
  std::unique_ptr<Pipeline> pl;
  Sink* sink = nullptr;
  AudioTemporalResampleStage* stage = nullptr;
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
  auto st_u = std::make_unique<AudioTemporalResampleStage>(
      &r.sess, "atr", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  r.stage = static_cast<AudioTemporalResampleStage*>(
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

// tempo = speed / pitch is the identity the three knobs compose
// through, and it is the part that is easy to get subtly wrong.
TEST(audio_temporal_resample, the_three_knobs_compose_into_one_chain)
{
  using S = AudioTemporalResampleStage;
  using Str = std::string;
  // `maintain` never touches the pitch factor, so the tempo IS the
  // speed: a time-stretch and nothing else.
  EXPECT_TRUE(S::pitch_factor(Pitch::kMaintain, 7.0, 2.0) == 1.0);
  // `follow` sets the pitch factor TO the speed, which makes the tempo
  // 1 -- i.e. no time-stretch at all, which is what "let it act like
  // tape" means.
  EXPECT_TRUE(S::pitch_factor(Pitch::kFollow, 0.0, 2.0) == 2.0);
  EXPECT_TRUE(std::fabs(S::pitch_factor(Pitch::kRaise, 12.0, 1.0) - 2.0)
              < 1e-9);
  EXPECT_TRUE(std::fabs(S::pitch_factor(Pitch::kLower, 12.0, 1.0) - 0.5)
              < 1e-9);

  // maintain, 2x: one atempo and no rate change.
  EXPECT_TRUE(S::chain_for(32000, 32000, 2.0, 1.0) == Str("atempo=2"));
  // follow, 2x: asetrate carries BOTH the pitch and the duration, so
  // there must be no atempo left to undo the duration half of it.
  EXPECT_TRUE(S::chain_for(32000, 32000, 2.0, 2.0) ==
              Str("asetrate=64000,aresample=32000"));
  // raise an octave at normal speed: the duration asetrate took away
  // has to come back, so tempo = 1/2.
  EXPECT_TRUE(S::chain_for(32000, 32000, 1.0, 2.0) ==
              Str("asetrate=64000,aresample=32000,atempo=0.5"));
  // A rate change alone is one aresample.
  EXPECT_TRUE(S::chain_for(44100, 32000, 1.0, 1.0) ==
              Str("aresample=32000"));
  // ...and a request that changes nothing still emits one, so the
  // graph is never empty and the stage has one code path.
  EXPECT_TRUE(S::chain_for(32000, 32000, 1.0, 1.0) ==
              Str("aresample=32000"));
  // atempo is capped at 2 on older builds, so an extreme factor has to
  // decompose into a chain that multiplies to it.
  const Str fast = S::chain_for(32000, 32000, 8.0, 1.0);
  EXPECT_TRUE(fast == Str("atempo=2,atempo=2,atempo=2"));
  const Str slow = S::chain_for(32000, 32000, 0.25, 1.0);
  EXPECT_TRUE(slow == Str("atempo=0.5,atempo=0.5"));
  std::printf("[audio_temporal_resample] raise an octave: %s\n",
              S::chain_for(32000, 32000, 1.0, 2.0).c_str());
}

TEST(audio_temporal_resample, the_surface_and_what_it_refuses)
{
  Session s;
  auto make = [&](FlexData c) {
    return std::string(AudioTemporalResampleStage(
                           &s, "x", std::vector<InEdge>{}, std::move(c))
                           .config_error());
  };
  // Every knob is optional: an unconfigured stage is a pass-through,
  // which is the harmless thing for one to be.
  EXPECT_TRUE(make(cfg_({})).empty());
  EXPECT_FALSE(make(cfg_({{"speed", FlexData::make_real(0.0)}})).empty());
  EXPECT_FALSE(make(cfg_({{"speed", FlexData::make_real(-1.0)}})).empty());
  EXPECT_FALSE(make(cfg_({{"output_sample_rate", FlexData::make_int(10)}}))
                   .empty());
  EXPECT_TRUE(make(cfg_({{"output_sample_rate", FlexData::make_int(32000)}}))
                  .empty());
  EXPECT_FALSE(make(cfg_({{"pitch", FlexData::make_string("higher")}}))
                   .empty());
  // raise / lower without an amount is a request to shift by nothing,
  // and silently doing nothing is the failure this stage is about.
  EXPECT_FALSE(make(cfg_({{"pitch", FlexData::make_string("raise")}}))
                   .empty());
  EXPECT_FALSE(make(cfg_({{"pitch", FlexData::make_string("lower")}}))
                   .empty());
  EXPECT_TRUE(make(cfg_({{"pitch", FlexData::make_string("raise")},
                         {"pitch_semitones", FlexData::make_real(3.0)}}))
                  .empty());
  // The direction lives in `pitch`, so a negative amount would be a
  // second way to spell it and a way to cancel it by accident.
  EXPECT_FALSE(make(cfg_({{"pitch", FlexData::make_string("raise")},
                          {"pitch_semitones", FlexData::make_real(-3.0)}}))
                   .empty());
  EXPECT_TRUE(make(cfg_({{"pitch", FlexData::make_string("follow")},
                         {"speed", FlexData::make_real(1.5)}}))
                  .empty());

  AudioTemporalResampleStage st(&s, "y", std::vector<InEdge>{}, cfg_({}));
  const StageSpec& sp = st.spec();
  EXPECT_TRUE(std::string(sp.type_name) == "audio-temporal-resample");
  EXPECT_TRUE(sp.iports.size() == 1 && sp.oports.size() == 1);
  EXPECT_TRUE(std::string(sp.iports[0].name) == "pcm");
  std::printf("[audio_temporal_resample] %zu iports / %zu oports\n",
              sp.iports.size(), sp.oports.size());
}

TEST(audio_temporal_resample, stacked_decides_whether_a_domain_is_crossed)
{
  auto domains = [](bool stacked) {
    Session sess;
    Pipeline pl("p", &sess);
    auto src_u = std::make_unique<BeatSource>(&sess, "src",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
    auto* src = static_cast<BeatSource*>(pl.insert_stage(std::move(src_u)));
    src->allocate_oports(1);
    auto st_u = std::make_unique<AudioTemporalResampleStage>(
        &sess, "atr", std::vector<InEdge>{{src, 0}},
        cfg_({{"speed", FlexData::make_real(0.5)},
              {"stacked", FlexData::make_bool(stacked)}}));
    auto* st = static_cast<AudioTemporalResampleStage*>(
        pl.insert_stage(std::move(st_u)));
    auto sink_u = std::make_unique<Sink>(&sess, "sink",
                                         std::vector<InEdge>{{st, 0}},
                                         FlexData::make_object());
    auto* sink = static_cast<Sink*>(pl.insert_stage(std::move(sink_u)));
    const std::vector<Stage*> stages = {src, st, sink};
    const auto a = compute_clock_domains(stages);
    return std::pair<unsigned, unsigned>{
        a.port_domain.at(PortKey{st, PortKey::Kind::In, 0}),
        a.port_domain.at(PortKey{st, PortKey::Kind::Out, 0})};
  };
  const auto stream = domains(false);
  EXPECT_TRUE(stream.first != stream.second);
  const auto stacked = domains(true);
  EXPECT_TRUE(stacked.first == stacked.second);
  std::printf("[audio_temporal_resample] stream %u/%u, stacked %u/%u\n",
              stream.first, stream.second, stacked.first, stacked.second);
}

// The four pitch policies, each measured by what it did to the TONE and
// to the LENGTH. Both numbers are needed: three of the four produce the
// same length as one of the others.
TEST(audio_temporal_resample, pitch_and_duration_are_independent)
{
  {
    Session probe;
    if (!have_avfilter_(probe)) { return; }
  }
  constexpr int SR = 32000, N = SR;         // one second
  struct Case { const char* pitch; double speed; double st;
                double want_hz; double want_secs; };
  const Case cases[] = {
    // Hold the pitch while halving the duration -- the default, and
    // what keeps a voice recognisable.
    {"maintain", 2.0, 0.0, 440.0, 0.5},
    // Let it act like tape: half as long AND an octave up, together.
    {"follow",   2.0, 0.0, 880.0, 0.5},
    // Shift up an octave at the original length...
    {"raise",    1.0, 12.0, 880.0, 1.0},
    // ...and down.
    {"lower",    1.0, 12.0, 220.0, 1.0},
    // Both at once: the knobs compose rather than fight.
    {"raise",    2.0, 12.0, 880.0, 0.5},
  };
  for (const Case& c : cases) {
    Run r;
    std::vector<TensorBeat> beats;
    beats.push_back(tone_(1, N, SR, 440.0));
    ASSERT_TRUE(drive_(r, std::move(beats),
                       cfg_({{"speed", FlexData::make_real(c.speed)},
                             {"pitch", FlexData::make_string(c.pitch)},
                             {"pitch_semitones", FlexData::make_real(c.st)},
                             {"stacked", FlexData::make_bool(true)}})));
    EXPECT_TRUE(r.sink->beats == 1);
    const double secs = (double)r.sink->left.size() / SR;
    const double f = hz_(r.sink->left, SR);
    // 3% on the length: atempo's window and aresample's tail both
    // round, and neither is a defect.
    EXPECT_TRUE(std::fabs(secs - c.want_secs) < 0.03 * c.want_secs + 0.02);
    // 2% on the pitch, which separates 440 from 880 and from 220 by a
    // factor of two either way.
    EXPECT_TRUE(std::fabs(f - c.want_hz) < 0.02 * c.want_hz);
    EXPECT_TRUE(sb_real_(r.sink->last_sb, "sample_rate") == (double)SR);
    std::printf("[audio_temporal_resample] %-8s speed %.1f st %.0f -> "
                "%.3f s, %.1f Hz (want %.2f s, %.0f Hz)\n",
                c.pitch, c.speed, c.st, secs, f, c.want_secs, c.want_hz);
  }
}

// The RESOLUTION knob on its own: same seconds, same tone, different
// number of samples. It is the one a model forces.
TEST(audio_temporal_resample, a_rate_change_alone_moves_neither)
{
  {
    Session probe;
    if (!have_avfilter_(probe)) { return; }
  }
  constexpr int SR = 32000;
  Run r;
  std::vector<TensorBeat> beats;
  beats.push_back(tone_(2, SR, SR, 440.0));
  ASSERT_TRUE(drive_(
      r, std::move(beats),
      cfg_({{"output_sample_rate", FlexData::make_int(16000)},
            {"stacked", FlexData::make_bool(true)}})));
  EXPECT_TRUE(r.sink->beats == 1);
  EXPECT_TRUE(r.sink->channels == 2);
  // Half the samples...
  EXPECT_TRUE(r.sink->left.size() > 15500 && r.sink->left.size() < 16500);
  // ...for the same second of the same tone.
  EXPECT_TRUE(std::fabs(hz_(r.sink->left, 16000) - 440.0) < 8.0);
  EXPECT_TRUE(sb_real_(r.sink->last_sb, "sample_rate") == 16000.0);
  EXPECT_TRUE(std::fabs(sb_real_(r.sink->last_sb, "duration_us") - 1e6)
              < 5e4);
  std::printf("[audio_temporal_resample] 32000 -> 16000: %zu samples/ch, "
              "%.1f Hz\n", r.sink->left.size(),
              hz_(r.sink->left, 16000));
}

// Streaming: the chunk boundaries are the filter's, not the input's, so
// the assertion is on the TOTAL -- and on the tail, which only comes out
// because the stage flushes the graph at EOS. Without that flush the
// waveform is short by the filter's latency and nothing says so.
TEST(audio_temporal_resample, a_chunk_stream_keeps_every_sample)
{
  {
    Session probe;
    if (!have_avfilter_(probe)) { return; }
  }
  constexpr int SR = 32000, CH = 3200;      // 100 ms chunks
  Run r;
  std::vector<TensorBeat> beats;
  for (int i = 0; i < 10; ++i) {
    beats.push_back(tone_(1, CH, SR, 440.0));
  }
  ASSERT_TRUE(drive_(r, std::move(beats),
                     cfg_({{"speed", FlexData::make_real(2.0)},
                           {"pitch", FlexData::make_string("maintain")}})));
  EXPECT_TRUE(r.sink->beats >= 1);
  // One second in at 2x is half a second out, within the stretcher's
  // window. A missing flush shows up here as a systematic shortfall.
  EXPECT_TRUE(r.sink->left.size() > 15000 &&
              r.sink->left.size() < 17500);
  EXPECT_TRUE(std::fabs(hz_(r.sink->left, SR) - 440.0) < 20.0);
  std::printf("[audio_temporal_resample] 10 chunks -> %d beats, %zu "
              "samples (want ~16000)\n", r.sink->beats,
              r.sink->left.size());
}
