// `load-audio`: an audio file in, encoded packets out.
//
// Driven by a WAV this test WRITES ITSELF, so it needs no asset and no
// model: 16-bit PCM is 44 bytes of header and a sine, and FFmpeg demuxes
// it like anything else. That is enough to pin what the stage owes its
// only consumer -- `audio-to-pcm` opens a decoder from `codec_id`,
// `sample_rate`, `channels` and `extradata`, and feeds `data` as ONE
// AVPacket, so a beat holding a whole file would be one malformed
// packet.
//
// The last case runs the two stages TOGETHER, because "load-audio emits
// packets" and "audio-to-pcm can decode them" are different claims and
// only the second one is the point of the stage.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/encoded-segment.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/audio-to-pcm-stage.h"
#include "stages/load-audio-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// A mono 16-bit PCM WAV of `n` samples at `rate`, one second of a sine.
// Written rather than checked in: an asset in the repo is a thing to
// keep in sync, and this one is fifteen lines.
std::string
write_wav_(int rate, int n, const char* name = "vpipe-ut-load-audio.wav")
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
  u16(1);                                   // PCM
  u16(1);                                   // mono
  u32((std::uint32_t)rate);
  u32((std::uint32_t)rate * 2);             // byte rate
  u16(2);                                   // block align
  u16(16);                                  // bits
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
  static constexpr const char* kTypeName = "ut-la-seg-sink";
  using TypedStage::TypedStage;

  int beats = 0;
  unsigned codec_id = 0, sample_rate = 0, channels = 0;
  std::size_t total_bytes = 0;
  bool all_audio = true, all_nonempty = true;
  std::int64_t first_us = -1, last_us = -1;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* s = dynamic_cast<const EncodedSegmentPayload*>(in.get())) {
      ++beats;
      codec_id = s->codec_id;
      sample_rate = s->sample_rate;
      channels = s->channels;
      total_bytes += s->data.size();
      if (s->kind != EncodedSegment::Kind::Audio) { all_audio = false; }
      if (s->data.empty()) { all_nonempty = false; }
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
      {.name = "audio", .doc = "", .type = &typeid(EncodedSegmentPayload)}};
    static const StageSpec s = {.type_name = "ut-la-seg-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

class PcmSink : public TypedStage<PcmSink> {
public:
  static constexpr const char* kTypeName = "ut-la-pcm-sink";
  using TypedStage::TypedStage;

  int beats = 0, rate = 0;
  std::size_t samples = 0;
  double peak = 0.0;
  // Per-chunk, for the overlap tests: an overlap is a claim about how
  // far CONSECUTIVE chunks advance, which one chunk cannot show.
  std::vector<std::size_t>   chunk_samples;
  std::vector<std::uint64_t> chunk_ts;
  // The samples themselves, so an overlap can be checked as an identity
  // between two chunks rather than inferred from their sizes.
  std::vector<std::vector<float>> chunk_data;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      ++beats;
      std::size_t n = 1;
      for (std::int64_t d : tb->shape) { n *= (std::size_t)d; }
      samples += n;
      const float* p = tb->as_f32();
      for (std::size_t i = 0; i < n && p != nullptr; ++i) {
        peak = std::max(peak, (double)std::fabs(p[i]));
      }
      chunk_samples.push_back(n);
      if (p != nullptr) { chunk_data.emplace_back(p, p + n); }
      else { chunk_data.emplace_back(); }
      if (tb->sideband.is_object()) {
        FlexData sb = tb->sideband;
        auto o = sb.as_object();
        if (o.contains("sample_rate")) {
          rate = (int)o.at("sample_rate").as_int(0);
        }
        chunk_ts.push_back(o.contains("timestamp_us")
                               ? (std::uint64_t)o.at("timestamp_us").as_uint(0)
                               : 0);
      }
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "pcm", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-la-pcm-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

}  // namespace

TEST(load_audio, stage_surface_and_deferred_validation)
{
  Session sess;
  // No input_url: the ctor must NOT throw -- a graph is edited before it
  // is complete -- but the stage must refuse at launch.
  auto s = std::make_unique<LoadAudioStage>(
      &sess, "la", std::vector<InEdge>{}, FlexData::make_object());
  EXPECT_FALSE(s->config_error().empty());
  const StageSpec& sp = s->spec();
  EXPECT_TRUE(sp.iports.empty());          // a SOURCE
  EXPECT_TRUE(sp.oports.size() == 1);
  EXPECT_TRUE(std::string(sp.oports[0].name) == "audio");
  // The oport type is what audio-to-pcm's iport takes; anything else and
  // the edge is refused at launch rather than failing at runtime.
  EXPECT_TRUE(sp.oports[0].type != nullptr &&
              *sp.oports[0].type == typeid(EncodedSegmentPayload));
}

TEST(load_audio, emits_one_beat_per_packet_with_the_streams_parameters)
{
  const int kRate = 16000, kSamples = 16000;   // one second
  const std::string path = write_wav_(kRate, kSamples);
  if (path.empty()) { return; }

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("input_url", FlexData::make_string(path));
  auto la_u = std::make_unique<LoadAudioStage>(
      &sess, "la", std::vector<InEdge>{}, cfg);
  auto* la = static_cast<LoadAudioStage*>(pl->insert_stage(std::move(la_u)));
  auto sink_u = std::make_unique<SegSink>(&sess, "sink",
                                          std::vector<InEdge>{{la, 0}},
                                          FlexData::make_object());
  auto* sink = static_cast<SegSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[load_audio] %d beat(s), codec_id %u, %u Hz, %u ch, %zu B\n",
              sink->beats, sink->codec_id, sink->sample_rate, sink->channels,
              sink->total_bytes);
  EXPECT_TRUE(sink->beats > 0);
  EXPECT_TRUE(sink->all_audio);
  EXPECT_TRUE(sink->all_nonempty);
  // The parameters audio-to-pcm opens its decoder from.
  EXPECT_TRUE(sink->sample_rate == (unsigned)kRate);
  EXPECT_TRUE(sink->channels == 1);
  EXPECT_TRUE(sink->codec_id != 0);
  // Every sample reaches the consumer: 16-bit mono, so 2 bytes each.
  EXPECT_TRUE(sink->total_bytes == (std::size_t)kSamples * 2);
  // MEDIA time, from the start of the file -- the first packet at zero
  // and time moving forward. A stage that stamped wall-clock "now"
  // would put first_us in the billions.
  EXPECT_TRUE(sink->first_us == 0);
  EXPECT_TRUE(sink->last_us > 0);
  EXPECT_TRUE(sink->last_us < 1000000);
  EXPECT_TRUE(la->packets_emitted() == (std::uint64_t)sink->beats);

  std::filesystem::remove(path);
}

TEST(load_audio, drives_audio_to_pcm)
{
  // The claim that matters: the packets this stage emits are ones
  // audio-to-pcm can actually decode. Different assertion from "beats
  // came out", and the only reason the stage exists.
  const int kRate = 16000, kSamples = 16000;
  const std::string path = write_wav_(kRate, kSamples);
  if (path.empty()) { return; }

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto lcfg = FlexData::make_object();
  lcfg.as_object().insert_or_assign("input_url", FlexData::make_string(path));
  auto la_u = std::make_unique<LoadAudioStage>(
      &sess, "la", std::vector<InEdge>{}, lcfg);
  auto* la = static_cast<LoadAudioStage*>(pl->insert_stage(std::move(la_u)));

  auto pcfg = FlexData::make_object();
  pcfg.as_object().insert_or_assign("output_sample_rate",
                                    FlexData::make_int(16000));
  pcfg.as_object().insert_or_assign("chunk_duration_s",
                                    FlexData::make_real(0.1));
  auto tp_u = std::make_unique<AudioToPcmStage>(
      &sess, "to-pcm", std::vector<InEdge>{{la, 0}}, pcfg);
  auto* tp = static_cast<AudioToPcmStage*>(pl->insert_stage(std::move(tp_u)));

  auto sink_u = std::make_unique<PcmSink>(&sess, "sink",
                                          std::vector<InEdge>{{tp, 0}},
                                          FlexData::make_object());
  auto* sink = static_cast<PcmSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  // A REFUSED LAUNCH is the failure mode to watch for: the edge is
  // checked by payload type AND tag, so a tag this stage does not
  // publish would fail here rather than at the first beat.
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[load_audio] -> audio-to-pcm: %d chunk(s), %zu samples at "
              "%d Hz, peak %.3f\n", sink->beats, sink->samples, sink->rate,
              sink->peak);
  EXPECT_TRUE(sink->beats > 0);
  EXPECT_TRUE(sink->rate == 16000);
  // Same rate in and out, so the sample count survives the decode within
  // a chunk's rounding.
  EXPECT_TRUE(sink->samples > (std::size_t)(kSamples * 0.9));
  // A 12000/32767 sine: decoded, not silence, and not clipped.
  EXPECT_TRUE(sink->peak > 0.2 && sink->peak <= 1.0);

  std::filesystem::remove(path);
}

// ---- start_s / duration_s: a WINDOW of a long file ------------------
//
// The claim is that the stage reads a SLICE and not the file, so the
// test measures the bytes that come out against the bytes in the
// window, and pins the start against a full read of the same file. A
// window that produced the whole file would pass any assertion written
// about the window alone.
TEST(load_audio, start_s_and_duration_s_take_a_window)
{
  const int kRate = 16000;
  const int kSeconds = 4;
  const std::string path =
      write_wav_(kRate, kRate * kSeconds, "vpipe-ut-load-audio-window.wav");
  if (path.empty()) { return; }

  auto run = [&](double start_s, double duration_s) {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("input_url", FlexData::make_string(path));
    if (start_s > 0.0) {
      cfg.as_object().insert_or_assign("start_s",
                                       FlexData::make_real(start_s));
    }
    if (duration_s > 0.0) {
      cfg.as_object().insert_or_assign("duration_s",
                                       FlexData::make_real(duration_s));
    }
    auto la_u = std::make_unique<LoadAudioStage>(
        &sess, "la", std::vector<InEdge>{}, cfg);
    auto* la = static_cast<LoadAudioStage*>(pl->insert_stage(std::move(la_u)));
    auto sink_u = std::make_unique<SegSink>(&sess, "sink",
                                            std::vector<InEdge>{{la, 0}},
                                            FlexData::make_object());
    auto* sink = static_cast<SegSink*>(pl->insert_stage(std::move(sink_u)));
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) { return std::make_tuple((std::size_t)0,
                                               (std::int64_t)-1,
                                               (std::int64_t)-1); }
    rt.wait_idle();
    rt.stop();
    return std::make_tuple(sink->total_bytes, sink->first_us, sink->last_us);
  };

  // The whole file, for the comparison the window is measured against.
  const auto [all_bytes, all_first, all_last] = run(0.0, 0.0);
  const std::size_t want_all = (std::size_t)kRate * kSeconds * 2;
  EXPECT_TRUE(all_bytes == want_all);
  EXPECT_TRUE(all_first == 0);

  // 1 s in, 2 s long: a quarter of the file skipped and half of it read.
  const auto [win_bytes, win_first, win_last] = run(1.0, 2.0);
  const std::size_t want_win = (std::size_t)kRate * 2 * 2;
  std::printf("[load_audio] window [1,3): %zu B (whole file %zu B), "
              "first_us %lld, last_us %lld\n", win_bytes, all_bytes,
              (long long)win_first, (long long)win_last);

  // Packet-accurate, so allow one packet at each end. A WAV packet here
  // is well under a tenth of a second, and the window is two seconds.
  const std::size_t slack = (std::size_t)kRate * 2 / 5;   // 0.2 s of bytes
  EXPECT_TRUE(win_bytes + slack > want_win && win_bytes < want_win + slack);
  // The window is genuinely a slice: not the whole file, and not empty.
  EXPECT_TRUE(win_bytes > 0 && win_bytes < all_bytes);

  // MEDIA time is NOT rebased -- a window starting at 1 s stamps 1 s,
  // so the same samples carry the same times whatever window read them.
  EXPECT_TRUE(win_first > 800000 && win_first <= 1000000);
  EXPECT_TRUE(win_last < 3100000);

  // duration_s alone reads from the beginning and stops.
  const auto [head_bytes, head_first, head_last] = run(0.0, 1.0);
  EXPECT_TRUE(head_first == 0);
  EXPECT_TRUE(head_bytes < all_bytes / 2 + (std::size_t)kRate / 2);
  EXPECT_TRUE(head_bytes > 0);

  std::filesystem::remove(path);
}

// A negative window is refused at CONFIG, where the graph can still be
// fixed, rather than clamped to something the author did not ask for.
TEST(load_audio, a_negative_window_is_refused)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("input_url",
                                   FlexData::make_string("/tmp/x.wav"));
  cfg.as_object().insert_or_assign("start_s", FlexData::make_real(-1.0));
  auto s = std::make_unique<LoadAudioStage>(
      &sess, "la", std::vector<InEdge>{}, cfg);
  EXPECT_FALSE(s->config_error().empty());

  auto cfg2 = FlexData::make_object();
  cfg2.as_object().insert_or_assign("input_url",
                                    FlexData::make_string("/tmp/x.wav"));
  cfg2.as_object().insert_or_assign("duration_s", FlexData::make_real(-2.0));
  auto s2 = std::make_unique<LoadAudioStage>(
      &sess, "la2", std::vector<InEdge>{}, cfg2);
  EXPECT_FALSE(s2->config_error().empty());
}

// ---- chunk_overlap_s: what CONSECUTIVE chunks share -----------------
//
// The mechanism is that the stage KEEPS the tail of the chunk it just
// emitted, so the next one opens with it. That shows up two ways at
// once, and the test pins both: chunks stay chunk_duration_s long, and
// their timestamps advance by only (chunk_duration_s - overlap).
TEST(audio_to_pcm, chunk_overlap_shares_a_tail_with_the_next_chunk)
{
  const int kRate = 16000;
  const int kSeconds = 2;
  const std::string path =
      write_wav_(kRate, kRate * kSeconds, "vpipe-ut-a2p-overlap.wav");
  if (path.empty()) { return; }

  // A stage cannot be copied out of the pipeline, so the lambda hands
  // back just the numbers the claim is about.
  struct Got {
    int beats = 0;
    std::size_t samples = 0;
    std::vector<std::size_t>   chunk_samples;
    std::vector<std::uint64_t> chunk_ts;
    std::vector<std::vector<float>> chunk_data;
  };
  auto run = [&](double overlap_s) {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto lcfg = FlexData::make_object();
    lcfg.as_object().insert_or_assign("input_url",
                                      FlexData::make_string(path));
    auto la_u = std::make_unique<LoadAudioStage>(
        &sess, "la", std::vector<InEdge>{}, lcfg);
    auto* la = static_cast<LoadAudioStage*>(pl->insert_stage(std::move(la_u)));
    auto pcfg = FlexData::make_object();
    pcfg.as_object().insert_or_assign("output_sample_rate",
                                      FlexData::make_int(kRate));
    pcfg.as_object().insert_or_assign("chunk_duration_s",
                                      FlexData::make_real(0.5));
    pcfg.as_object().insert_or_assign("max_chunk_duration_s",
                                      FlexData::make_real(0.5));
    pcfg.as_object().insert_or_assign("flush_on_eos",
                                      FlexData::make_bool(false));
    if (overlap_s > 0.0) {
      pcfg.as_object().insert_or_assign("chunk_overlap_s",
                                        FlexData::make_real(overlap_s));
    }
    auto tp_u = std::make_unique<AudioToPcmStage>(
        &sess, "to-pcm", std::vector<InEdge>{{la, 0}}, pcfg);
    auto* tp = static_cast<AudioToPcmStage*>(pl->insert_stage(std::move(tp_u)));
    auto sink_u = std::make_unique<PcmSink>(&sess, "sink",
                                            std::vector<InEdge>{{tp, 0}},
                                            FlexData::make_object());
    auto* sink = static_cast<PcmSink*>(pl->insert_stage(std::move(sink_u)));
    PipelineRuntime rt(pl.get(), &sess);
    Got g;
    if (!rt.launch()) { return g; }
    rt.wait_idle();
    rt.stop();
    g.beats         = sink->beats;
    g.samples       = sink->samples;
    g.chunk_samples = sink->chunk_samples;
    g.chunk_ts      = sink->chunk_ts;
    g.chunk_data    = sink->chunk_data;
    return g;
  };

  // No overlap: chunks land on PACKET boundaries at or past
  // chunk_duration_s (this WAV demuxes into 2048-sample packets, so a
  // 0.5 s chunk is really 8192 samples). The exact size is the
  // demuxer's business; what matters is that the chunks TILE the
  // stream -- consecutive ones share nothing.
  const Got plain = run(0.0);
  std::printf("[audio_to_pcm] no overlap: %d chunk(s), %zu samples\n",
              plain.beats, plain.samples);
  ASSERT_TRUE(plain.beats >= 2);
  EXPECT_TRUE(plain.samples <= (std::size_t)kRate * kSeconds);
  if (plain.chunk_ts.size() >= 2 && !plain.chunk_samples.empty()) {
    // A tiling stream advances by a WHOLE chunk each time.
    const std::uint64_t hop = plain.chunk_ts[1] - plain.chunk_ts[0];
    EXPECT_TRUE(hop == (std::uint64_t)plain.chunk_samples[0]
                           * 1000000ULL / (std::uint64_t)kRate);
  }

  // 0.25 s of overlap = 4000 samples kept from each chunk.
  const std::size_t kKeep = (std::size_t)(0.25 * kRate);
  const Got ov = run(0.25);
  std::printf("[audio_to_pcm] 0.25s overlap: %d chunk(s), %zu samples "
              "(file holds %d)\n", ov.beats, ov.samples, kRate * kSeconds);

  // More chunks over the same file, and MORE samples than the file
  // holds -- the retained tail goes out twice, which is the point.
  EXPECT_TRUE(ov.beats > plain.beats);
  EXPECT_TRUE(ov.samples > (std::size_t)kRate * kSeconds);

  // THE CLAIM ITSELF: chunk k's last 4000 samples ARE chunk k+1's first
  // 4000. Checked as an identity on the samples, not inferred from
  // sizes -- a stage that retained the wrong end, or re-decoded the
  // region instead of keeping it, would size the same and match here
  // only by accident.
  ASSERT_TRUE(ov.chunk_data.size() >= 3);
  int shared_pairs = 0;
  for (std::size_t i = 1; i < ov.chunk_data.size(); ++i) {
    const auto& prev = ov.chunk_data[i - 1];
    const auto& cur  = ov.chunk_data[i];
    if (prev.size() < kKeep || cur.size() < kKeep) { continue; }
    bool same = true;
    for (std::size_t k = 0; k < kKeep; ++k) {
      if (prev[prev.size() - kKeep + k] != cur[k]) { same = false; break; }
    }
    EXPECT_TRUE(same);
    if (same) { ++shared_pairs; }
    else {
      std::printf("[audio_to_pcm] chunks %zu/%zu do not share their "
                  "%zu-sample seam\n", i - 1, i, kKeep);
    }
  }
  EXPECT_TRUE(shared_pairs == (int)ov.chunk_data.size() - 1);

  // And the stream advances by (chunk - overlap), so the timestamps
  // agree with the samples about where each chunk starts.
  for (std::size_t i = 1; i < ov.chunk_ts.size(); ++i) {
    const std::uint64_t hop = ov.chunk_ts[i] - ov.chunk_ts[i - 1];
    const std::uint64_t want = (std::uint64_t)(ov.chunk_samples[i - 1] - kKeep)
                               * 1000000ULL / (std::uint64_t)kRate;
    EXPECT_TRUE(hop == want);
    if (hop != want) {
      std::printf("[audio_to_pcm] chunk %zu hop %llu us, expected %llu\n",
                  i, (unsigned long long)hop, (unsigned long long)want);
    }
  }

  std::filesystem::remove(path);
}

// An overlap that leaves no room for new samples would emit the same
// window for ever. Warned and dropped to 0, so the run makes progress.
TEST(audio_to_pcm, an_overlap_that_cannot_advance_is_dropped)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("chunk_duration_s",
                                   FlexData::make_real(0.5));
  cfg.as_object().insert_or_assign("chunk_overlap_s",
                                   FlexData::make_real(0.5));
  auto s = std::make_unique<AudioToPcmStage>(
      &sess, "a2p", std::vector<InEdge>{}, cfg);
  // Not a config REFUSAL -- the stage still runs, without the overlap.
  EXPECT_TRUE(s->config_error().empty());
}
