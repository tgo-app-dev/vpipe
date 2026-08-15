// `audio-vae-encode`: the stage that turns PCM into a reference
// soundtrack.
//
// Driven by a STUB family here rather than a model. What the stage owns
// is the accumulation, the rate contract and the drain, and all three
// are wrong in ways a model would hide:
//
//   * it must buffer EVERY beat and encode ONCE, at EOS. An audio VAE
//     is causal and compresses time, so per-beat encoding and
//     concatenation is a different tensor -- the stage exists partly to
//     make that impossible to ask for;
//   * a rate mismatch must REFUSE, not resample. Encoding 48 kHz
//     samples as 16 kHz gives a reference of the right length at the
//     wrong pitch, which nothing downstream can see;
//   * it must publish the ENCODER's shape, not one it predicted.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "generative-models/vae-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/typed-stage.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/runtime-context.h"
#include "stages/audio-vae-encode-stage.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// A family whose encoder records what it was handed. It claims one
// distinctive root so it cannot shadow anything else in the process --
// the registry is a singleton and other tests register into it too.
struct Seen {
  int channels = 0, n_samples = 0, sample_rate = 0;
  int calls = 0;
  float first = 0.0f, last = 0.0f;
};
Seen g_seen;

class StubAudioEncoder : public genai::AudioVaeEncoder {
public:
  int sample_rate() const override { return 16000; }
  int channels() const override { return 2; }

  bool encode(const genai::AudioVaeEncodeRequest& req,
              std::vector<float>* out, std::vector<int>* shape,
              std::string* err) override
  {
    (void)err;
    ++g_seen.calls;
    g_seen.channels    = req.channels;
    g_seen.n_samples   = req.n_samples;
    g_seen.sample_rate = req.sample_rate;
    if (req.n_samples > 0) {
      g_seen.first = req.pcm[0];
      g_seen.last  = req.pcm[(std::size_t)req.channels * req.n_samples - 1];
    }
    // A shape that is NOT derivable from the request: the stage must
    // publish what it is told, not what it worked out.
    const int rows = req.n_samples / 640;
    *shape = {rows, 128};
    out->assign((std::size_t)rows * 128, 0.25f);
    return true;
  }
};

class StubAudioFamily : public genai::VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-aenc"; }

  bool claims(const std::string& root, const std::string&,
              const std::string&) const override
  {
    return root == "/ut/aenc-model";
  }

  std::unique_ptr<genai::VaeDecoder>
  load_decoder(const genai::VaeModelCreateArgs&) override { return nullptr; }

  std::unique_ptr<genai::AudioVaeEncoder>
  load_audio_encoder(const genai::VaeModelCreateArgs&) override
  {
    return std::make_unique<StubAudioEncoder>();
  }
};

// Emits `n_beats` PCM beats and then closes, so the stage sees a real
// EOS rather than being stopped.
class PcmSource : public TypedStage<PcmSource> {
public:
  static constexpr const char* kTypeName = "ut-aenc-pcm-source";
  using TypedStage::TypedStage;

  int channels = 2, per_beat = 1600, n_beats = 3, rate = 16000;
  int emitted = 0;
  // When set, the SECOND beat claims this rate instead -- the
  // mid-stream change the stage must refuse rather than splice.
  int second_rate = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    if (emitted >= n_beats) { ctx.signal_done(); co_return; }
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::F32;
    b->shape = {(std::int64_t)channels, (std::int64_t)per_beat};
    b->resize_contiguous((std::size_t)channels * per_beat);
    float* p = b->as_f32();
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < per_beat; ++i) {
        // A ramp that is unique per (beat, channel, sample), so an
        // append in the wrong order shows as the wrong endpoints.
        p[(std::size_t)c * per_beat + i] =
            (float)(emitted * 1000 + c * 100) + (float)i / per_beat;
      }
    }
    FlexData sb = FlexData::make_object();
    const int r =
        (emitted == 1 && second_rate > 0) ? second_rate : rate;
    sb.as_object().insert_or_assign("sample_rate",
                                    FlexData::make_int((std::int64_t)r));
    b->sideband = std::move(sb);
    ++emitted;
    co_await ctx.write(0, std::move(b));
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "pcm", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-aenc-pcm-source", .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class RowSink : public TypedStage<RowSink> {
public:
  static constexpr const char* kTypeName = "ut-aenc-row-sink";
  using TypedStage::TypedStage;

  int beats = 0, rows = 0, dim = 0, rate = 0;
  double seconds = 0.0;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      ++beats;
      if (tb->shape.size() == 2) {
        rows = (int)tb->shape[0];
        dim  = (int)tb->shape[1];
      }
      if (tb->sideband.is_object()) {
        FlexData sb = tb->sideband;
        auto o = sb.as_object();
        if (o.contains("sample_rate")) {
          rate = (int)o.at("sample_rate").as_int(0);
        }
        if (o.contains("seconds")) {
          seconds = o.at("seconds").as_real(0.0);
        }
      }
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "latent", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-aenc-row-sink", .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// Registered once; the registry is first-wins on tag and outlives every
// test, so a second registration would be silently ignored.
bool
register_stub_()
{
  static const bool once =
      genai::VaeModelRegistry::get().add(std::make_unique<StubAudioFamily>());
  return once;
}

struct Result {
  int beats = 0, rows = 0, dim = 0, rate = 0;
  double seconds = 0.0;
};

Result
run_(int n_beats, int per_beat, int src_rate, int second_rate,
     const char* hf_dir)
{
  g_seen = Seen{};
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<PcmSource>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  src_u->n_beats  = n_beats;
  src_u->per_beat = per_beat;
  src_u->rate     = src_rate;
  src_u->second_rate = second_rate;
  src_u->allocate_oports(1);
  auto* src = static_cast<PcmSource*>(pl->insert_stage(std::move(src_u)));

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(hf_dir));
  auto enc_u = std::make_unique<AudioVaeEncodeStage>(
      &sess, "aenc", std::vector<InEdge>{{src, 0}}, cfg);
  auto* enc =
      static_cast<AudioVaeEncodeStage*>(pl->insert_stage(std::move(enc_u)));

  auto sink_u = std::make_unique<RowSink>(&sess, "sink",
                                          std::vector<InEdge>{{enc, 0}},
                                          FlexData::make_object());
  auto* sink = static_cast<RowSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  return {sink->beats, sink->rows, sink->dim, sink->rate, sink->seconds};
}

}  // namespace

TEST(audio_vae_encode, stage_surface)
{
  Session sess;
  auto s = std::make_unique<AudioVaeEncodeStage>(
      &sess, "aenc", std::vector<InEdge>{}, FlexData::make_object());
  // Deferred validation: an incomplete config CONSTRUCTS, so a graph can
  // be edited before the model exists.
  EXPECT_TRUE(s->config_error().empty());
  const StageSpec& sp = s->spec();
  EXPECT_TRUE(sp.iports.size() == 2 && sp.oports.size() == 1);
  EXPECT_TRUE(std::string(sp.iports[0].name) == "audio");
  EXPECT_TRUE(std::string(sp.oports[0].name) == "latent");
  // The iport takes what audio-vae-decode and audio-to-pcm emit; the
  // oport is what generate-video's ref_audio_rows takes.
  EXPECT_TRUE(sp.iports[0].type != nullptr &&
              *sp.iports[0].type == typeid(TensorBeatPayload));
  EXPECT_TRUE(sp.oports[0].type != nullptr &&
              *sp.oports[0].type == typeid(TensorBeatPayload));
}

TEST(audio_vae_encode, accumulates_every_beat_and_encodes_once_at_eos)
{
  EXPECT_TRUE(register_stub_());
  const Result r = run_(/*n_beats=*/3, /*per_beat=*/1600, 16000, 0,
                        "/ut/aenc-model");
  // ONE encode, over ALL the samples -- not one per beat.
  EXPECT_TRUE(g_seen.calls == 1);
  EXPECT_TRUE(g_seen.n_samples == 3 * 1600);
  EXPECT_TRUE(g_seen.channels == 2);
  EXPECT_TRUE(g_seen.sample_rate == 16000);
  // The endpoints pin the ORDER: the first sample of beat 0 channel 0,
  // and the last of beat 2 channel 1. A per-channel append that
  // interleaved instead would still have the right count.
  EXPECT_TRUE(g_seen.first == 0.0f);
  EXPECT_TRUE(g_seen.last > 2100.0f && g_seen.last < 2101.0f);
  // ONE beat out, carrying the ENCODER's shape and the stream's rate.
  EXPECT_TRUE(r.beats == 1);
  EXPECT_TRUE(r.rows == (3 * 1600) / 640);
  EXPECT_TRUE(r.dim == 128);
  EXPECT_TRUE(r.rate == 16000);
  EXPECT_TRUE(r.seconds > 0.29 && r.seconds < 0.31);
}

TEST(audio_vae_encode, a_rate_the_encoder_does_not_want_is_refused)
{
  EXPECT_TRUE(register_stub_());
  // 48 kHz into a 16 kHz encoder. Resampling silently would give a
  // reference of the right length at the wrong pitch.
  const Result r = run_(2, 1600, 48000, 0, "/ut/aenc-model");
  EXPECT_TRUE(g_seen.calls == 0);
  EXPECT_TRUE(r.beats == 0);
}

TEST(audio_vae_encode, a_mid_stream_rate_change_drops_the_beat)
{
  EXPECT_TRUE(register_stub_());
  // Beat 1 claims a different rate. It is dropped; the rest still
  // encode, because a graph that hiccuped once should not lose the
  // whole reference.
  const Result r = run_(3, 1600, 16000, 22050, "/ut/aenc-model");
  EXPECT_TRUE(g_seen.calls == 1);
  EXPECT_TRUE(g_seen.n_samples == 2 * 1600);
  EXPECT_TRUE(r.beats == 1);
}

TEST(audio_vae_encode, no_family_means_inert_not_a_wrong_encoder)
{
  EXPECT_TRUE(register_stub_());
  // A root nothing claims. There is no built-in audio encoder to fall
  // back to, and falling back to one would encode at the wrong latent
  // geometry -- so the stage produces nothing.
  const Result r = run_(2, 1600, 16000, 0, "/ut/aenc-nobody-claims-this");
  EXPECT_TRUE(g_seen.calls == 0);
  EXPECT_TRUE(r.beats == 0);
}
