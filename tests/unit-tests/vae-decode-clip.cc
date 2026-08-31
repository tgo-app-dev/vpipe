// `vae-decode`'s CLIP oport: the same pixels as oport 0, in one beat.
//
// Two claims, tested at two costs.
//
// The SURFACE is free and is what a graph is authored against: two
// oports, the clip typed and tagged, and -- the part that matters for
// feedback -- both in the same clock group as the latent iport, since a
// feedback pair must stay inside one clock domain.
//
// The PIXELS need a real VAE, so that test is gated on
// VPIPE_MINIMAX_H3_TEST_MODEL_PATH and skips vacuously without it. It
// is the only assertion that can catch the failure that matters: a clip
// whose frames are out of order, or off by one, or written at the wrong
// stride. Nothing about the shape would look wrong.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/vae-decode-stage.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// Emits one latent beat, then ends.
class LatentSource : public TypedStage<LatentSource> {
public:
  static constexpr const char* kTypeName = "ut-vdc-latent-source";
  using TypedStage::TypedStage;

  TensorBeat beat;
  bool sent = false;

  Job
  process(RuntimeContext& ctx) override
  {
    if (sent) { ctx.signal_done(); co_return; }
    sent = true;
    TensorBeat tb = beat;
    auto payload = make_payload<TensorBeatPayload>(std::move(tb));
    co_await ctx.write(0, std::move(payload));
    co_return;
  }
};

// Collects whole beats, so a frame stream and a clip can be compared.
class BeatCollector : public TypedStage<BeatCollector> {
public:
  static constexpr const char* kTypeName = "ut-vdc-collector";
  using TypedStage::TypedStage;

  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<std::vector<std::uint8_t>> bytes;
  std::vector<FlexData> sidebands;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      shapes.push_back(tb->shape);
      const std::uint8_t* p = tb->bytes_();
      bytes.emplace_back(p, p + tb->byte_size());
      sidebands.push_back(tb->sideband);
    }
    co_return;
  }
};

}  // namespace

// The surface a graph is authored against.
TEST(vae_decode_clip, the_clip_oport_shares_the_latents_clock_group)
{
  Session sess;
  auto s = std::make_unique<VaeDecodeStage>(
      &sess, "vd", std::vector<InEdge>{}, FlexData::make_object());
  const StageSpec& sp = s->spec();

  ASSERT_TRUE(sp.oports.size() == 2);
  if (sp.oports.size() != 2) { return; }
  EXPECT_TRUE(std::string(sp.oports[0].name) == "image");
  EXPECT_TRUE(std::string(sp.oports[1].name) == "clip");
  // Same payload type as every other tensor port, so it wires to a
  // temporal-slice or a feedback-rx without an adapter.
  EXPECT_TRUE(sp.oports[1].type != nullptr &&
              *sp.oports[1].type == typeid(TensorBeatPayload));
  EXPECT_TRUE(std::string(sp.oports[1].tags) == "rgb-clip");

  // THE POINT: one beat per latent, so the clip oport is in the
  // latent's clock group. A feedback pair must stay inside one clock
  // domain, and this is what lets the loop close through a decode.
  EXPECT_TRUE(s->oport_clock_group(1) == s->iport_clock_group(0));
  EXPECT_TRUE(sp.iports[0].clock_group == sp.oports[1].clock_group);
}

// The pixels. Gated on a real MiniMax-H3 checkpoint; skips otherwise.
//
// The assertion is an IDENTITY between the two oports rather than a
// property of either: frame f of the clip must be byte-for-byte the
// f-th beat of the frame stream. That is what makes a transposed,
// off-by-one or wrongly strided clip fail, and none of those change a
// shape.
TEST(vae_decode_clip, the_clip_is_the_frame_stream_packed)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') {
    std::printf("[vae_decode_clip] VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset; "
                "skipping the pixel test\n");
    return;
  }

  // The smallest video latent the H3 VAE will take: [24, T, h, w]. T is
  // 8 because that is one temporal CHUNK -- fewer is refused outright --
  // and the canvas is tiny so the 2.4B ViT decoder has little to do.
  const int Cz = 24, LT = 8, lh = 4, lw = 4;
  TensorBeat lat;
  lat.dtype = TensorBeat::DType::F32;
  lat.shape = {Cz, LT, lh, lw};
  lat.resize_contiguous((std::size_t)Cz * LT * lh * lw * sizeof(float));
  float* z = reinterpret_cast<float*>(lat.data.data());
  for (std::size_t i = 0; i < (std::size_t)Cz * LT * lh * lw; ++i) {
    z[i] = 0.01f * (float)(i % 17) - 0.05f;
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
  lat.sideband = std::move(sb);

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<LatentSource>(&sess, "src",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
  src_u->beat = lat;
  auto* src = static_cast<LatentSource*>(pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  auto vd_u = std::make_unique<VaeDecodeStage>(
      &sess, "vd", std::vector<InEdge>{{src, 0}}, cfg);
  auto* vd = static_cast<VaeDecodeStage*>(pl->insert_stage(std::move(vd_u)));

  auto frames_u = std::make_unique<BeatCollector>(
      &sess, "frames", std::vector<InEdge>{{vd, 0}}, FlexData::make_object());
  auto* frames =
      static_cast<BeatCollector*>(pl->insert_stage(std::move(frames_u)));
  auto clip_u = std::make_unique<BeatCollector>(
      &sess, "clip", std::vector<InEdge>{{vd, 1}}, FlexData::make_object());
  auto* clip =
      static_cast<BeatCollector*>(pl->insert_stage(std::move(clip_u)));

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[vae_decode_clip] %zu frame beat(s), %zu clip beat(s)\n",
              frames->shapes.size(), clip->shapes.size());
  ASSERT_TRUE(!frames->shapes.empty());
  ASSERT_TRUE(clip->shapes.size() == 1);
  if (frames->shapes.empty() || clip->shapes.size() != 1) { return; }

  // ONE beat per latent on the clip port, whatever F turned out to be.
  const std::size_t F = frames->shapes.size();
  const auto& cs = clip->shapes[0];
  ASSERT_TRUE(cs.size() == 4);
  if (cs.size() != 4) { return; }
  EXPECT_TRUE((std::size_t)cs[0] == F);
  EXPECT_TRUE(cs[1] == 3);
  EXPECT_TRUE(cs[2] == frames->shapes[0][1]);
  EXPECT_TRUE(cs[3] == frames->shapes[0][2]);

  // The identity.
  const std::size_t per = (std::size_t)3 * cs[2] * cs[3];
  ASSERT_TRUE(clip->bytes[0].size() == F * per);
  if (clip->bytes[0].size() != F * per) { return; }
  std::size_t bad = 0;
  for (std::size_t f = 0; f < F; ++f) {
    if (frames->bytes[f].size() != per) { ++bad; continue; }
    if (std::memcmp(clip->bytes[0].data() + f * per,
                    frames->bytes[f].data(), per) != 0) {
      ++bad;
    }
  }
  std::printf("[vae_decode_clip] %zu of %zu frames differ between the "
              "clip and the frame stream\n", bad, F);
  EXPECT_TRUE(bad == 0);

  // The sideband a clip consumer reads.
  FlexData csb = clip->sidebands[0];
  ASSERT_TRUE(csb.is_object());
  if (csb.is_object()) {
    auto o = csb.as_object();
    EXPECT_TRUE(o.contains("frames")
                && (std::size_t)o.at("frames").as_int(0) == F);
    EXPECT_TRUE(o.contains("fps") && o.at("fps").as_real(0.0) > 0.0);
  }
}

// oport0's rate is a property of the CHECKPOINT, not of the stage: one
// beat per latent for an image family, one per FRAME for a video one.
// Only the second is a rate change, and only the second should keep a
// feedback loop from closing through it.
TEST(vae_decode_clip, the_frame_oport_is_a_rate_change_only_for_video)
{
  Session sess;

  // Nothing to probe: the fallback is the IMAGE answer, which is what
  // this stage reported for every checkpoint before it could tell them
  // apart. A probe that comes up empty must not split a domain.
  auto blank = std::make_unique<VaeDecodeStage>(
      &sess, "vd", std::vector<InEdge>{}, FlexData::make_object());
  EXPECT_TRUE(blank->oport_clock_group(0) == blank->iport_clock_group(0));
  EXPECT_TRUE(blank->oport_clock_group(1) == blank->iport_clock_group(0));

  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') {
    std::printf("[vae_decode_clip] VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset; "
                "skipping the video-checkpoint half\n");
    return;
  }

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(root));
  auto vid = std::make_unique<VaeDecodeStage>(
      &sess, "vd-video", std::vector<InEdge>{}, cfg);

  // A VIDEO checkpoint: the frame stream is a rate change and says so,
  // while the clip stays in the latent's clock.
  std::printf("[vae_decode_clip] video checkpoint: iport0=%u oport0=%u "
              "oport1=%u\n", vid->iport_clock_group(0),
              vid->oport_clock_group(0), vid->oport_clock_group(1));
  EXPECT_TRUE(vid->oport_clock_group(0) != vid->iport_clock_group(0));
  EXPECT_TRUE(vid->oport_clock_group(1) == vid->iport_clock_group(0));
}
