// content-refusal.cc -- what happens DOWNSTREAM of a content screen.
//
// A family's content screen (generative-models/content-screen.h) runs in
// diffusion-conditioner; a refused prompt leaves it as a one-row
// conditioning beat tagged `content_blocked`. The rest of the mechanism
// is the host's and is pinned here, with stub families so no model is
// needed:
//
//   generate-image  skips the denoise and emits a 1x1x1 marker carrying
//                   only the flag and the size the picture would have had
//   vae-decode      paints the blank (white) refusal image at that size,
//                   without decoding anything
//
// Each stub COUNTS its work: a refusal that ran a generation or a decode
// would still produce the right shapes, and the counters are what say it
// did not.
//
//   vpipe_test --filter 'content_refusal*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/image-model-registry.h"
#include "generative-models/vae-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/vae-decode-stage.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

int g_generations = 0;
int g_decodes = 0;

const char* const kMarker = "ut-content-refusal.marker";

bool
marked_(const std::string& root)
{
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::path(root) / kMarker, ec);
}

class CountingGenerator : public genai::ImageGenerator {
public:
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 8; }
  bool
  generate(const genai::ImageGenRequest& req,
           genai::ImageGenResult* out) override
  {
    ++g_generations;
    const int h = req.height / 8, w = req.width / 8;
    out->latent.assign((std::size_t)4 * h * w, 0.0f);
    out->shape = {4, h, w};
    return true;
  }
};

class StubImageFamily : public genai::ImageModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-refusal"; }
  bool
  claims(const std::string& root, const std::string&) const override
  {
    return marked_(root);
  }
  std::unique_ptr<genai::ImageGenerator>
  load(const genai::ImageModelCreateArgs&) override
  {
    return std::make_unique<CountingGenerator>();
  }
};

class CountingDecoder : public genai::VaeDecoder {
public:
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 8; }
  bool
  decode(const genai::VaeDecodeRequest&, const genai::VaeFrameSink&,
         std::string* err) override
  {
    ++g_decodes;
    if (err != nullptr) { *err = "the stub decodes nothing"; }
    return false;
  }
};

class StubVaeFamily : public genai::VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-refusal"; }
  bool
  claims(const std::string& root, const std::string&,
         const std::string&) const override
  {
    return marked_(root);
  }
  std::unique_ptr<genai::VaeDecoder>
  load_decoder(const genai::VaeModelCreateArgs&) override
  {
    return std::make_unique<CountingDecoder>();
  }
};

void
register_()
{
  static const bool once = [] {
    genai::ImageModelRegistry::get().add(std::make_unique<StubImageFamily>());
    genai::VaeModelRegistry::get().add(std::make_unique<StubVaeFamily>());
    return true;
  }();
  (void)once;
}

std::filesystem::path
root_()
{
  const auto root =
      std::filesystem::temp_directory_path() / "vpipe-ut-content-refusal";
  std::filesystem::create_directories(root);
  std::ofstream(root / kMarker) << "x";
  return root;
}

// Sends one beat, then ends its stream.
class OneBeat : public TypedStage<OneBeat> {
public:
  static constexpr const char* kTypeName = "ut-content-refusal-src";
  using TypedStage::TypedStage;
  std::unique_ptr<BeatPayloadIntf> beat;
  Job
  process(RuntimeContext& ctx) override
  {
    if (beat) {
      co_await ctx.write(0, std::move(beat));
      co_return;
    }
    ctx.signal_done();
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {{.name = "out", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-content-refusal-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {{.name = "in", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// One stage between a source and a sink, run to the end.
std::vector<std::unique_ptr<BeatPayloadIntf>>
run_(Session& sess, std::unique_ptr<BeatPayloadIntf> beat,
     const std::function<std::unique_ptr<Stage>(std::vector<InEdge>)>& mk)
{
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto su = std::make_unique<OneBeat>(&sess, "src", std::vector<InEdge>{},
                                      FlexData::make_object());
  su->beat = std::move(beat);
  su->allocate_oports(1);
  auto* src = pl->insert_stage(std::move(su));
  auto* mid = pl->insert_stage(mk({{src, 0}}));
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::make_unique<Sink>(
      &sess, "sink", std::vector<InEdge>{{mid, 0}}, FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  if (rt.launch()) {
    rt.wait_idle();
    rt.stop();
  }
  return std::move(sink->captured);
}

}  // namespace

// generate-image turns a tagged CONDITIONING beat into the refusal marker
// without running the family's generation.
TEST(content_refusal, generate_image_skips_the_denoise)
{
  register_();
  g_generations = 0;
  const auto root = root_();
  Session sess;

  auto cond = std::make_unique<TensorBeatPayload>();
  cond->dtype = TensorBeat::DType::Bf16;
  cond->shape = {1, 8};
  cond->resize_contiguous(8);
  std::memset(cond->as_u8(), 0, 16);
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("content_blocked",
                                  FlexData::make_bool(true));
  cond->sideband = std::move(sb);

  auto got = run_(sess, std::move(cond), [&](std::vector<InEdge> in) {
    in.resize(9, InEdge{nullptr, 0});
    FlexData c = FlexData::make_object();
    auto o = c.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root.string()));
    o.insert_or_assign("width", FlexData::make_int(64));
    o.insert_or_assign("height", FlexData::make_int(48));
    o.insert_or_assign("steps", FlexData::make_int(2));
    return std::make_unique<GenerateImageStage>(&sess, "gi", std::move(in),
                                                std::move(c));
  });
  std::filesystem::remove_all(root);

  EXPECT_TRUE(g_generations == 0);
  ASSERT_TRUE(got.size() == 1);
  if (got.size() != 1) { return; }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(got[0].get());
  ASSERT_TRUE(tb != nullptr);
  if (tb == nullptr) { return; }
  EXPECT_TRUE(tb->shape == std::vector<std::int64_t>({1, 1, 1}));
  ASSERT_TRUE(tb->sideband.is_object());
  FlexData osb = tb->sideband;             // as_object() is a view: keep it
  auto o = osb.as_object();
  EXPECT_TRUE(o.contains("content_blocked") &&
              o.at("content_blocked").as_bool(false));
  EXPECT_TRUE(o.contains("refusal_height") &&
              o.at("refusal_height").as_int(0) == 48);
  EXPECT_TRUE(o.contains("refusal_width") &&
              o.at("refusal_width").as_int(0) == 64);
}

// vae-decode paints the blank refusal image at the size the marker
// carries -- non-square, so a transposed h/w would fail -- and decodes
// nothing.
TEST(content_refusal, vae_decode_paints_the_blank_refusal)
{
  register_();
  g_decodes = 0;
  const auto root = root_();
  Session sess;

  auto marker = std::make_unique<TensorBeatPayload>();
  marker->dtype = TensorBeat::DType::F32;
  marker->shape = {1, 1, 1};
  marker->resize_contiguous(1);
  marker->as_f32()[0] = 0.0f;
  FlexData sb = FlexData::make_object();
  {
    auto o = sb.as_object();
    o.insert_or_assign("content_blocked", FlexData::make_bool(true));
    o.insert_or_assign("refusal_height", FlexData::make_int(48));
    o.insert_or_assign("refusal_width", FlexData::make_int(64));
  }
  marker->sideband = std::move(sb);

  auto got = run_(sess, std::move(marker), [&](std::vector<InEdge> in) {
    FlexData c = FlexData::make_object();
    c.as_object().insert_or_assign("hf_dir",
                                   FlexData::make_string(root.string()));
    return std::make_unique<VaeDecodeStage>(&sess, "vae", std::move(in),
                                            std::move(c));
  });
  std::filesystem::remove_all(root);

  EXPECT_TRUE(g_decodes == 0);
  ASSERT_TRUE(got.size() == 1);
  if (got.size() != 1) { return; }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(got[0].get());
  ASSERT_TRUE(tb != nullptr);
  if (tb == nullptr) { return; }
  ASSERT_TRUE(tb->dtype == TensorBeat::DType::U8);
  EXPECT_TRUE(tb->shape == std::vector<std::int64_t>({3, 48, 64}));
  const auto px = tb->materialize_contiguous();
  ASSERT_TRUE(px.size() == (std::size_t)3 * 48 * 64);
  bool all_white = !px.empty();
  for (const auto b : px) {
    if (b != 0xff) { all_white = false; break; }
  }
  EXPECT_TRUE(all_white);
}
