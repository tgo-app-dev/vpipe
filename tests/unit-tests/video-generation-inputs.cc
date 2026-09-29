// video-generation-inputs.cc -- how generate-video pairs its inputs across
// a run of several clips, and the seed each clip gets.
//
// The rule is generate-image's (stages/generation-input.h): one beat and
// the end of its stream serves every clip, a stream that keeps sending is
// consumed a beat per clip, and the run is as long as the shortest
// consumed stream. It is pinned through what a registered family is handed
// call by call -- the seam every out-of-tree video family meets it at.
//
// Before the rule a first-frame latent was read once per conditioning, so
// the SECOND prompt of a run met the end of its stream and ran as
// text-to-video without a word.
//
//   vpipe_test --filter 'video_generation_inputs*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/video-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/generate-video-stage.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

struct Seen {
  int calls = 0;
  std::vector<std::uint64_t> seeds;
  std::vector<float>         cond;   // the conditioning's first value
  std::vector<float>         ref;    // ref_latent0's, -1 when absent
  std::vector<float>         last;   // ref_latent1's, -1 when absent
};
Seen g_seen;

class RecordingGenerator : public genai::VideoGenerator {
public:
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 8; }
  bool
  generate(const genai::VideoGenRequest& req,
           genai::VideoGenResult* out) override
  {
    ++g_seen.calls;
    g_seen.seeds.push_back(req.seed);
    g_seen.cond.push_back(req.cond != nullptr
                              ? static_cast<const float*>(req.cond)[0]
                              : -1.0f);
    g_seen.ref.push_back(req.ref != nullptr ? req.ref[0] : -1.0f);
    g_seen.last.push_back(req.ref_last != nullptr ? req.ref_last[0]
                                                  : -1.0f);
    out->video.assign(4 * 1 * 2 * 2, 0.0f);
    out->video_shape = {4, 1, 2, 2};
    return true;
  }
};

class RecordingFamily : public genai::VideoModelFamily {
public:
  std::string_view tag() const noexcept override
  {
    return "ut-video-gen-inputs";
  }
  bool
  claims(const std::string& root, const std::string&) const override
  {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(root) / "ut-video-gen-inputs.marker", ec);
  }
  std::unique_ptr<genai::VideoGenerator>
  load(const genai::VideoModelCreateArgs&) override
  {
    return std::make_unique<RecordingGenerator>();
  }
};

bool
registered_()
{
  static const bool once = genai::VideoModelRegistry::get().add(
      std::make_unique<RecordingFamily>());
  return once ||
         genai::VideoModelRegistry::get().find("ut-video-gen-inputs") !=
             nullptr;
}

// A source that sends its beats, one per call, and then ends its stream.
class Beats : public TypedStage<Beats> {
public:
  static constexpr const char* kTypeName = "ut-video-gen-inputs-beats";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next < beats.size()) {
      co_await ctx.write(0, std::move(beats[next++]));
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

class Drain : public TypedStage<Drain> {
public:
  static constexpr const char* kTypeName = "ut-video-gen-inputs-drain";
  using TypedStage::TypedStage;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); }
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

// An f32 tensor of `shape` whose every value is `fill`, so the family
// can tell which beat it was handed.
std::unique_ptr<BeatPayloadIntf>
filled_(std::vector<std::int64_t> shape, float fill)
{
  auto tb = std::make_unique<TensorBeatPayload>();
  tb->dtype = TensorBeat::DType::F32;
  std::size_t n = 1;
  for (std::int64_t d : shape) { n *= (std::size_t)d; }
  tb->shape = std::move(shape);
  tb->resize_contiguous(n);
  for (std::size_t i = 0; i < n; ++i) { tb->as_f32()[i] = fill; }
  return tb;
}

// What to wire around generate-video. Each vector is one source's beats,
// by fill value; an empty one leaves that input unwired.
struct Spec {
  std::vector<float> conds = {0.0f};
  std::vector<float> ref0;              // first-frame latent
  std::vector<float> ref1;              // last-frame latent
  std::vector<std::pair<std::string, FlexData>> cfg;
};

bool
generate_(Spec spec)
{
  g_seen = Seen{};
  const auto root =
      std::filesystem::temp_directory_path() / "vpipe-ut-video-gen-inputs";
  std::filesystem::create_directories(root);
  std::ofstream(root / "ut-video-gen-inputs.marker") << "x";

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto source = [&](const char* id, const std::vector<float>& fills,
                    const std::vector<std::int64_t>& shape) {
    auto u = std::make_unique<Beats>(&sess, id, std::vector<InEdge>{},
                                     FlexData::make_object());
    for (float f : fills) { u->beats.push_back(filled_(shape, f)); }
    u->allocate_oports(1);
    return static_cast<Beats*>(pl->insert_stage(std::move(u)));
  };
  const InEdge none{nullptr, 0};
  std::vector<InEdge> in(11, none);
  in[0] = {source("cond", spec.conds, {2, 8}), 0};
  if (!spec.ref0.empty()) {
    in[5] = {source("ref0", spec.ref0, {4, 1, 2, 2}), 0};
  }
  if (!spec.ref1.empty()) {
    in[6] = {source("ref1", spec.ref1, {4, 1, 2, 2}), 0};
  }

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root.string()));
    o.insert_or_assign("width", FlexData::make_int(64));
    o.insert_or_assign("height", FlexData::make_int(64));
    o.insert_or_assign("frames", FlexData::make_int(9));
    o.insert_or_assign("steps", FlexData::make_int(2));
    for (auto& kv : spec.cfg) { o.insert_or_assign(kv.first, kv.second); }
  }
  auto gv_u = std::make_unique<GenerateVideoStage>(&sess, "gv", in, cfg);
  if (!gv_u->config_error().empty()) { return false; }
  auto* gv = pl->insert_stage(std::move(gv_u));
  pl->insert_stage(std::make_unique<Drain>(
      &sess, "lat", std::vector<InEdge>{{gv, 0}}, FlexData::make_object()));

  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  if (launched) {
    rt.wait_idle();
    rt.stop();
  }
  std::filesystem::remove_all(root);
  return launched;
}

}  // namespace

// Many prompts, one first frame: every clip is image-to-video.
TEST(video_generation_inputs, a_single_first_frame_serves_every_prompt)
{
  ASSERT_TRUE(registered_());
  Spec spec;
  spec.conds = {10.0f, 11.0f, 12.0f};
  spec.ref0  = {1.0f};
  ASSERT_TRUE(generate_(std::move(spec)));
  ASSERT_TRUE(g_seen.calls == 3);
  EXPECT_TRUE((g_seen.cond == std::vector<float>{10.0f, 11.0f, 12.0f}));
  EXPECT_TRUE((g_seen.ref == std::vector<float>{1.0f, 1.0f, 1.0f}));
  EXPECT_TRUE((g_seen.last == std::vector<float>{-1.0f, -1.0f, -1.0f}));
}

// One prompt, several first-and-last pairs: one clip per pair, the
// prompt held.
TEST(video_generation_inputs, a_single_prompt_serves_every_keyframe_pair)
{
  ASSERT_TRUE(registered_());
  Spec spec;
  spec.conds = {10.0f};
  spec.ref0  = {1.0f, 2.0f};
  spec.ref1  = {5.0f, 6.0f};
  ASSERT_TRUE(generate_(std::move(spec)));
  ASSERT_TRUE(g_seen.calls == 2);
  EXPECT_TRUE((g_seen.cond == std::vector<float>{10.0f, 10.0f}));
  EXPECT_TRUE((g_seen.ref == std::vector<float>{1.0f, 2.0f}));
  EXPECT_TRUE((g_seen.last == std::vector<float>{5.0f, 6.0f}));
}

// Two consumed streams of different lengths: the shorter ends the run.
// And a run of single beats is exactly one clip.
TEST(video_generation_inputs, the_shorter_stream_ends_and_singles_make_one)
{
  ASSERT_TRUE(registered_());
  {
    Spec spec;
    spec.conds = {10.0f, 11.0f, 12.0f};
    spec.ref0  = {1.0f, 2.0f};
    ASSERT_TRUE(generate_(std::move(spec)));
    EXPECT_TRUE(g_seen.calls == 2);
    EXPECT_TRUE((g_seen.cond == std::vector<float>{10.0f, 11.0f}));
  }
  {
    Spec spec;
    spec.ref0 = {1.0f};
    spec.ref1 = {5.0f};
    ASSERT_TRUE(generate_(std::move(spec)));
    EXPECT_TRUE(g_seen.calls == 1);
  }
}

// seed_sequence, by the seed each clip is handed.
TEST(video_generation_inputs, seed_sequence_increment_keep_randomize)
{
  ASSERT_TRUE(registered_());
  auto run_seq = [](const char* seq) {
    Spec spec;
    spec.conds = {10.0f, 11.0f, 12.0f};
    spec.cfg.push_back({"seed", FlexData::make_int(42)});
    if (seq != nullptr) {
      spec.cfg.push_back({"seed_sequence", FlexData::make_string(seq)});
    }
    return generate_(std::move(spec)) && g_seen.calls == 3;
  };
  ASSERT_TRUE(run_seq(nullptr));             // the default is increment
  EXPECT_TRUE((g_seen.seeds == std::vector<std::uint64_t>{42, 43, 44}));
  ASSERT_TRUE(run_seq("keep"));
  EXPECT_TRUE((g_seen.seeds == std::vector<std::uint64_t>{42, 42, 42}));
  ASSERT_TRUE(run_seq("randomize"));
  if (g_seen.seeds.size() == 3) {
    EXPECT_TRUE(g_seen.seeds[0] != g_seen.seeds[1]);
    EXPECT_TRUE(g_seen.seeds[1] != g_seen.seeds[2]);
    for (std::uint64_t v : g_seen.seeds) {
      EXPECT_TRUE(v < (1ull << 53));
    }
  }
}
