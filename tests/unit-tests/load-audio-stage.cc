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
write_wav_(int rate, int n)
{
  namespace fs = std::filesystem;
  const fs::path p = fs::temp_directory_path() / "vpipe-ut-load-audio.wav";
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
      if (tb->sideband.is_object()) {
        FlexData sb = tb->sideband;
        auto o = sb.as_object();
        if (o.contains("sample_rate")) {
          rate = (int)o.at("sample_rate").as_int(0);
        }
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
