// vae-encode over an OUT-OF-TREE VAE family, across more than one beat.
//
// The regression this pins: with its idle policy at "unload", the stage
// used to call the family encoder's release_idle() after each beat and
// never mark itself unloaded. LTX-2.5 implements that hook as a drop it
// cannot come back from, so the stage held a hollow encoder and every
// beat after the first failed with "the encoder was released" -- a clip
// stacked into groups encoded its first group and nothing more. One beat
// per run, which is what every graph before it sent, never reaches the
// second encode, so nothing noticed.
//
// The stub here behaves as LTX-2.5's does: release_idle() hollows the
// encoder for good. The stage has to DESTROY it and load a fresh one.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/vae-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/vae-encode-stage.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

struct Counts {
  int loads = 0, encodes = 0, refused = 0, released = 0;
};
Counts g_counts;

// A causal video encoder in shape only: 32x space, 8x time, and a
// release_idle() it cannot come back from.
class HollowingEncoder : public genai::VaeEncoder {
public:
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 32; }
  int temporal_compression() const override { return 8; }

  bool encode(const genai::VaeEncodeRequest& req, std::vector<float>* out,
              std::vector<int>* shape, std::string* err) override
  {
    if (_released) {
      ++g_counts.refused;
      if (err != nullptr) { *err = "the encoder was released"; }
      return false;
    }
    ++g_counts.encodes;
    const int t = 1 + (req.frames - 1) / 8;
    *shape = {4, t, req.height / 32, req.width / 32};
    out->assign((std::size_t)4 * t * (req.height / 32) * (req.width / 32),
                0.5f);
    return true;
  }

  void release_idle() override
  {
    ++g_counts.released;
    _released = true;
  }

private:
  bool _released = false;
};

class HollowingFamily : public genai::VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-venc"; }

  bool claims(const std::string& root, const std::string&,
              const std::string&) const override
  {
    return root == "/ut/venc-model";
  }

  std::unique_ptr<genai::VaeDecoder>
  load_decoder(const genai::VaeModelCreateArgs&) override { return nullptr; }

  std::unique_ptr<genai::VaeEncoder>
  load_encoder(const genai::VaeModelCreateArgs&) override
  {
    ++g_counts.loads;
    return std::make_unique<HollowingEncoder>();
  }
};

// `n_clips` stacked u8 clips of 9 frames, [9, 3, 64, 64] -- the shape
// temporal-stack emits.
class ClipSource : public TypedStage<ClipSource> {
public:
  static constexpr const char* kTypeName = "ut-venc-clip-source";
  using TypedStage::TypedStage;

  int n_clips = 3;
  int emitted = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    if (emitted >= n_clips) { ctx.signal_done(); co_return; }
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::U8;
    b->shape = {9, 3, 64, 64};
    b->resize_contiguous((std::size_t)9 * 3 * 64 * 64);
    std::memset(b->as_u8(), 40 + 10 * emitted, (std::size_t)9 * 3 * 64 * 64);
    ++emitted;
    co_await ctx.write(0, std::move(b));
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "clip", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-venc-clip-source",
                                .doc = "", .display_name = "",
                                .oports = op};
    return s;
  }
};

class LatentSink : public TypedStage<LatentSink> {
public:
  static constexpr const char* kTypeName = "ut-venc-latent-sink";
  using TypedStage::TypedStage;

  int beats = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (dynamic_cast<const TensorBeatPayload*>(in.get()) != nullptr) {
      ++beats;
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "latent", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = "ut-venc-latent-sink",
                                .doc = "", .display_name = "",
                                .iports = ip};
    return s;
  }
};

// Registered once: the registry is first-wins on tag and outlives every
// test.
bool
register_stub_()
{
  static const bool once =
      genai::VaeModelRegistry::get().add(std::make_unique<HollowingFamily>());
  return once;
}

int
run_(int n_clips, const char* unload_when_idle)
{
  g_counts = Counts{};
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<ClipSource>(&sess, "src",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  src_u->n_clips = n_clips;
  src_u->allocate_oports(1);
  auto* src = static_cast<ClipSource*>(pl->insert_stage(std::move(src_u)));

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir",
                                   FlexData::make_string("/ut/venc-model"));
  cfg.as_object().insert_or_assign("unload_when_idle",
                                   FlexData::make_string(unload_when_idle));
  auto enc_u = std::make_unique<VaeEncodeStage>(
      &sess, "venc", std::vector<InEdge>{{src, 0}}, cfg);
  auto* enc = static_cast<VaeEncodeStage*>(pl->insert_stage(std::move(enc_u)));

  auto sink_u = std::make_unique<LatentSink>(&sess, "sink",
                                             std::vector<InEdge>{{enc, 0}},
                                             FlexData::make_object());
  auto* sink = static_cast<LatentSink*>(pl->insert_stage(std::move(sink_u)));

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return -1; }
  rt.wait_idle();
  rt.stop();
  return sink->beats;
}

}  // namespace

TEST(vae_encode_plugin, an_unloading_stage_encodes_every_clip)
{
  EXPECT_TRUE(register_stub_());
  const int beats = run_(3, "destroy");
  std::printf("[vae_encode_plugin] destroy: %d latents, %d loads, %d "
              "encodes, %d refused, %d release_idle calls\n", beats,
              g_counts.loads, g_counts.encodes, g_counts.refused,
              g_counts.released);
  // Every clip encoded -- not the first one and then refusals.
  EXPECT_TRUE(beats == 3);
  EXPECT_TRUE(g_counts.encodes == 3);
  EXPECT_TRUE(g_counts.refused == 0);
  // Unloading means a FRESH encoder per clip, not the family's soft
  // release on the one it already has.
  EXPECT_TRUE(g_counts.loads == 3);
  EXPECT_TRUE(g_counts.released == 0);
}

TEST(vae_encode_plugin, a_keeping_stage_loads_once)
{
  EXPECT_TRUE(register_stub_());
  const int beats = run_(3, "keep");
  std::printf("[vae_encode_plugin] keep: %d latents, %d loads, %d "
              "encodes\n", beats, g_counts.loads, g_counts.encodes);
  EXPECT_TRUE(beats == 3);
  EXPECT_TRUE(g_counts.encodes == 3);
  EXPECT_TRUE(g_counts.loads == 1);
}
