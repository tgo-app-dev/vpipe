// image-references.cc -- what a model family receives as its REFERENCES,
// and how generate-image pairs its inputs across a run of several
// generations.
//
// ImageGenRequest::references is a list rather than two fixed slots, so
// a family that takes several references reads them all and the next
// one that takes more costs no field. What is pinned here is the
// contract as a family meets it through generate-image: every wired
// reference arrives, in port order, packed (a graph that wires only the
// second port hands over ONE), each at its own size and labelled f32.
//
// The GENERATION RULE (stages/generation-input.h) is pinned the same way,
// through what the family is handed call by call: one beat and the end
// of its stream serves every generation, a stream that keeps sending is
// consumed a beat per generation, and the run is as long as the shortest
// consumed stream. And `seed_sequence` by the seed each call gets.
//
//   vpipe_test --filter 'image_references*:generation_input*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/image-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/generation-input.h"

#include <algorithm>
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
  std::vector<std::vector<int>> shapes;
  std::vector<std::string>      dtypes;
  std::vector<float>            first;   // each reference's first value
  // Per CALL, i.e. per generation:
  std::vector<std::uint64_t>      seeds;
  std::vector<float>              cond;  // the conditioning's first value
  std::vector<std::vector<float>> refs;  // its references' first values
};
Seen g_seen;

class RecordingGenerator : public genai::ImageGenerator {
public:
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 8; }
  bool
  generate(const genai::ImageGenRequest& req,
           genai::ImageGenResult* out) override
  {
    ++g_seen.calls;
    g_seen.seeds.push_back(req.seed);
    g_seen.cond.push_back(req.cond != nullptr && req.cond_elem_size == 4
                              ? static_cast<const float*>(req.cond)[0]
                              : -1.0f);
    g_seen.refs.emplace_back();
    for (const genai::NamedTensor& r : req.references) {
      g_seen.shapes.push_back(r.shape);
      g_seen.dtypes.push_back(r.dtype());
      g_seen.first.push_back(r.data != nullptr
                                 ? static_cast<const float*>(r.data)[0]
                                 : -1.0f);
      g_seen.refs.back().push_back(g_seen.first.back());
    }
    const int h = req.height / 8, w = req.width / 8;
    out->latent.assign((std::size_t)4 * h * w, 0.0f);
    out->shape = {4, h, w};
    return true;
  }
};

class RecordingFamily : public genai::ImageModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-image-refs"; }
  bool
  claims(const std::string& root, const std::string&) const override
  {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(root) / "ut-image-refs.marker", ec);
  }
  std::unique_ptr<genai::ImageGenerator>
  load(const genai::ImageModelCreateArgs&) override
  {
    return std::make_unique<RecordingGenerator>();
  }
};

bool
register_()
{
  static const bool once = genai::ImageModelRegistry::get().add(
      std::make_unique<RecordingFamily>());
  return once;
}

// A source that sends its beats, one per call, and then ends its stream.
class Beats : public TypedStage<Beats> {
public:
  static constexpr const char* kTypeName = "ut-image-refs-beats";
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
  static constexpr const char* kTypeName = "ut-image-refs-drain";
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

// A [c, h, w] f32 latent whose every value is `fill`, so the family can
// tell which reference is which.
std::unique_ptr<BeatPayloadIntf>
latent_(int c, int h, int w, float fill)
{
  auto tb = std::make_unique<TensorBeatPayload>();
  tb->dtype = TensorBeat::DType::F32;
  tb->shape = {c, h, w};
  tb->resize_contiguous((std::size_t)c * h * w);
  float* p = tb->as_f32();
  for (std::size_t i = 0; i < (std::size_t)c * h * w; ++i) { p[i] = fill; }
  return tb;
}

// A list beat of [c, h, w] latents, item i filled with fills[i].
std::unique_ptr<BeatPayloadIntf>
latent_list_(const std::vector<std::vector<int>>& shapes,
             const std::vector<float>& fills)
{
  auto lp = std::make_unique<TensorListPayload>();
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    auto one = latent_(shapes[i][0], shapes[i][1], shapes[i][2], fills[i]);
    lp->items.push_back(
        std::move(static_cast<TensorBeat&>(
            *static_cast<TensorBeatPayload*>(one.get()))));
  }
  return lp;
}

// An f32 conditioning [2, 8] whose every value is `fill`, so the family
// can tell which beat it was handed.
std::unique_ptr<BeatPayloadIntf>
cond_(float fill)
{
  auto tb = std::make_unique<TensorBeatPayload>();
  tb->dtype = TensorBeat::DType::F32;
  tb->shape = {2, 8};
  tb->resize_contiguous(16);
  for (int i = 0; i < 16; ++i) { tb->as_f32()[i] = fill; }
  return tb;
}

// What to wire around generate-image. Each vector is one source's beats,
// by fill value; an empty one leaves that input unwired.
struct Spec {
  std::vector<float> conds = {0.0f};
  std::vector<float> ref0;              // ref_latent0, [4, 6, 10] each
  std::vector<float> ref1;              // ref_latent1, [4, 12, 8] each
  std::unique_ptr<BeatPayloadIntf> list;
  // Extra config keys: seed, seed_sequence, reference_mode.
  std::vector<std::pair<std::string, FlexData>> cfg;
};

// generate-image over the recording family. False if the graph did not
// even launch.
bool
generate_(Spec spec)
{
  g_seen = Seen{};
  const auto root =
      std::filesystem::temp_directory_path() / "vpipe-ut-image-refs";
  std::filesystem::create_directories(root);
  std::ofstream(root / "ut-image-refs.marker") << "x";

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto source = [&](const char* id,
                    std::vector<std::unique_ptr<BeatPayloadIntf>> bs) {
    auto u = std::make_unique<Beats>(&sess, id, std::vector<InEdge>{},
                                     FlexData::make_object());
    u->beats = std::move(bs);
    u->allocate_oports(1);
    return static_cast<Beats*>(pl->insert_stage(std::move(u)));
  };
  auto latents = [&](const std::vector<float>& fills, int c, int h, int w) {
    std::vector<std::unique_ptr<BeatPayloadIntf>> bs;
    for (float f : fills) { bs.push_back(latent_(c, h, w, f)); }
    return bs;
  };
  std::vector<std::unique_ptr<BeatPayloadIntf>> cbs;
  for (float f : spec.conds) { cbs.push_back(cond_(f)); }

  const InEdge none{nullptr, 0};
  std::vector<InEdge> in = {{source("cond", std::move(cbs)), 0}, none,
                            none, none, none, none, none, none, none};
  // Different sizes on purpose: references are not batched, each keeps
  // its own geometry.
  if (!spec.ref0.empty()) {
    in[5] = {source("ref0", latents(spec.ref0, 4, 6, 10)), 0};
  }
  if (!spec.ref1.empty()) {
    in[6] = {source("ref1", latents(spec.ref1, 4, 12, 8)), 0};
  }
  if (spec.list) {
    std::vector<std::unique_ptr<BeatPayloadIntf>> lb;
    lb.push_back(std::move(spec.list));
    in[8] = {source("refs", std::move(lb)), 0};
  }

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root.string()));
    o.insert_or_assign("height", FlexData::make_int(64));
    o.insert_or_assign("width", FlexData::make_int(64));
    o.insert_or_assign("steps", FlexData::make_int(2));
    for (auto& kv : spec.cfg) { o.insert_or_assign(kv.first, kv.second); }
  }
  auto gi_u = std::make_unique<GenerateImageStage>(&sess, "gi", in, cfg);
  if (!gi_u->config_error().empty()) { return false; }
  auto* gi = pl->insert_stage(std::move(gi_u));
  pl->insert_stage(std::make_unique<Drain>(
      &sess, "lat", std::vector<InEdge>{{gi, 0}}, FlexData::make_object()));

  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  if (launched) {
    rt.wait_idle();
    rt.stop();
  }
  std::filesystem::remove_all(root);
  return launched;
}

// The original shape: one conditioning, and one latent on each of the
// reference ports `wire` names (0 and/or 1), plus `list` on ref_latents.
bool
run_(const std::vector<int>& wire,
     std::unique_ptr<BeatPayloadIntf> list = nullptr)
{
  Spec spec;
  if (std::find(wire.begin(), wire.end(), 0) != wire.end()) {
    spec.ref0 = {1.0f};
  }
  if (std::find(wire.begin(), wire.end(), 1) != wire.end()) {
    spec.ref1 = {2.0f};
  }
  spec.list = std::move(list);
  return generate_(std::move(spec));
}

bool
registered_()
{
  return register_() ||
         genai::ImageModelRegistry::get().find("ut-image-refs") != nullptr;
}

}  // namespace

TEST(image_references, every_wired_reference_arrives_in_port_order)
{
  ASSERT_TRUE(register_() ||
              genai::ImageModelRegistry::get().find("ut-image-refs"));
  ASSERT_TRUE(run_({0, 1}));
  ASSERT_TRUE(g_seen.calls == 1);
  ASSERT_TRUE(g_seen.shapes.size() == 2);
  if (g_seen.shapes.size() != 2) { return; }
  EXPECT_TRUE((g_seen.shapes[0] == std::vector<int>{4, 6, 10}));
  EXPECT_TRUE((g_seen.shapes[1] == std::vector<int>{4, 12, 8}));
  EXPECT_TRUE(g_seen.first[0] == 1.0f);
  EXPECT_TRUE(g_seen.first[1] == 2.0f);
  EXPECT_TRUE(g_seen.dtypes[0] == "f32");
  EXPECT_TRUE(g_seen.dtypes[1] == "f32");
}

TEST(image_references, the_list_is_packed_and_empty_means_none)
{
  ASSERT_TRUE(register_() ||
              genai::ImageModelRegistry::get().find("ut-image-refs"));
  // Only the SECOND port: one reference, and it is that one.
  ASSERT_TRUE(run_({1}));
  ASSERT_TRUE(g_seen.calls == 1);
  ASSERT_TRUE(g_seen.shapes.size() == 1);
  if (g_seen.shapes.size() == 1) {
    EXPECT_TRUE((g_seen.shapes[0] == std::vector<int>{4, 12, 8}));
    EXPECT_TRUE(g_seen.first[0] == 2.0f);
  }
  // None wired: an empty list, not a null pointer to guard.
  ASSERT_TRUE(run_({}));
  EXPECT_TRUE(g_seen.calls == 1);
  EXPECT_TRUE(g_seen.shapes.empty());
}

// THE LIST PORT: ref_latents' items follow whatever is on the single
// ports, in list order, and a graph with only the list hands the family
// only the list -- as many as it holds.
TEST(image_references, the_list_follows_the_single_ports)
{
  ASSERT_TRUE(register_() ||
              genai::ImageModelRegistry::get().find("ut-image-refs"));
  ASSERT_TRUE(run_({0}, latent_list_({{4, 5, 5}, {4, 7, 9}, {4, 2, 3}},
                                     {3.0f, 4.0f, 5.0f})));
  ASSERT_TRUE(g_seen.calls == 1);
  ASSERT_TRUE(g_seen.shapes.size() == 4);
  if (g_seen.shapes.size() == 4) {
    EXPECT_TRUE((g_seen.first == std::vector<float>{1.0f, 3.0f, 4.0f,
                                                     5.0f}));
    EXPECT_TRUE((g_seen.shapes[2] == std::vector<int>{4, 7, 9}));
  }

  ASSERT_TRUE(run_({}, latent_list_({{4, 5, 5}, {4, 7, 9}},
                                    {6.0f, 7.0f})));
  ASSERT_TRUE(g_seen.shapes.size() == 2);
  if (g_seen.shapes.size() == 2) {
    EXPECT_TRUE((g_seen.first == std::vector<float>{6.0f, 7.0f}));
  }
}

// ---- the generation rule --------------------------------------------------

// The helper on its own: what each answer is, in each mode.
TEST(generation_input, one_beat_broadcasts_several_are_consumed)
{
  using In = GenerationInput;
  using Took = In::Took;
  // One beat, then the end: held for good, and never read again.
  In one;
  EXPECT_TRUE(one.wants());
  EXPECT_TRUE(one.took(true) == Took::kFresh);
  EXPECT_TRUE(one.wants());
  EXPECT_TRUE(one.took(false) == Took::kHold);
  EXPECT_FALSE(one.wants());
  // Several: each one fresh, and the end ENDS the run.
  In many;
  EXPECT_TRUE(many.took(true) == Took::kFresh);
  EXPECT_TRUE(many.took(true) == Took::kFresh);
  EXPECT_TRUE(many.took(false) == Took::kEnded);
  // Nothing at all.
  In empty;
  EXPECT_TRUE(empty.took(false) == Took::kNone);
  EXPECT_FALSE(empty.wants());
  // Latch: the first beat, whatever follows -- no second read.
  In latch(In::Mode::kLatch);
  EXPECT_TRUE(latch.took(true) == Took::kFresh);
  EXPECT_FALSE(latch.wants());
  // Fresh: a single beat is not broadcast.
  In fresh(In::Mode::kFresh);
  EXPECT_TRUE(fresh.took(true) == Took::kFresh);
  EXPECT_TRUE(fresh.took(false) == Took::kEnded);
  // A relaunch starts over.
  one.reset();
  EXPECT_TRUE(one.wants());
  EXPECT_TRUE(one.beats() == 0);

  // A round of broadcasts is idle; one fresh input is not; an ended
  // input stops it.
  GenerationRound held;
  EXPECT_FALSE(held.note(Took::kHold));
  EXPECT_TRUE(held.idle());
  GenerationRound live;
  EXPECT_FALSE(live.note(Took::kHold));
  EXPECT_FALSE(live.note(Took::kFresh));
  EXPECT_FALSE(live.idle());
  EXPECT_TRUE(live.note(Took::kEnded));
}

// Many prompts, one source picture: every generation gets the picture.
TEST(image_references, a_single_reference_serves_every_prompt)
{
  ASSERT_TRUE(registered_());
  Spec spec;
  spec.conds = {10.0f, 11.0f, 12.0f};
  spec.ref0  = {1.0f};
  ASSERT_TRUE(generate_(std::move(spec)));
  ASSERT_TRUE(g_seen.calls == 3);
  EXPECT_TRUE((g_seen.cond == std::vector<float>{10.0f, 11.0f, 12.0f}));
  ASSERT_TRUE(g_seen.refs.size() == 3);
  if (g_seen.refs.size() != 3) { return; }
  for (const auto& r : g_seen.refs) {
    EXPECT_TRUE((r == std::vector<float>{1.0f}));
  }
}

// One prompt, a folder of pictures: the prompt serves every picture, and
// each picture is used ONCE. Before the rule the reference latched and
// the prompt drove, so this made one image out of the first picture.
TEST(image_references, a_single_prompt_serves_every_reference)
{
  ASSERT_TRUE(registered_());
  Spec spec;
  spec.conds = {10.0f};
  spec.ref0  = {1.0f, 2.0f, 3.0f};
  spec.ref1  = {7.0f};                   // a second, single, reference
  ASSERT_TRUE(generate_(std::move(spec)));
  ASSERT_TRUE(g_seen.calls == 3);
  EXPECT_TRUE((g_seen.cond == std::vector<float>{10.0f, 10.0f, 10.0f}));
  ASSERT_TRUE(g_seen.refs.size() == 3);
  if (g_seen.refs.size() != 3) { return; }
  EXPECT_TRUE((g_seen.refs[0] == std::vector<float>{1.0f, 7.0f}));
  EXPECT_TRUE((g_seen.refs[1] == std::vector<float>{2.0f, 7.0f}));
  EXPECT_TRUE((g_seen.refs[2] == std::vector<float>{3.0f, 7.0f}));
}

// Two consumed streams of different lengths: as many generations as the
// SHORTER, each pair from the same position -- never a prompt repeated
// against a picture it was not sent with.
TEST(image_references, the_shorter_consumed_input_ends_the_run)
{
  ASSERT_TRUE(registered_());
  Spec spec;
  spec.conds = {10.0f, 11.0f, 12.0f};
  spec.ref0  = {1.0f, 2.0f};
  ASSERT_TRUE(generate_(std::move(spec)));
  ASSERT_TRUE(g_seen.calls == 2);
  EXPECT_TRUE((g_seen.cond == std::vector<float>{10.0f, 11.0f}));
  ASSERT_TRUE(g_seen.refs.size() == 2);
  if (g_seen.refs.size() != 2) { return; }
  EXPECT_TRUE((g_seen.refs[1] == std::vector<float>{2.0f}));
}

// Every input a broadcast: exactly one generation, not one per input.
TEST(image_references, all_single_beats_make_one_generation)
{
  ASSERT_TRUE(registered_());
  Spec spec;
  spec.ref0 = {1.0f};
  spec.ref1 = {2.0f};
  ASSERT_TRUE(generate_(std::move(spec)));
  EXPECT_TRUE(g_seen.calls == 1);
}

// `reference_mode` still overrides: latch holds the first of several
// pictures, per_beat refuses to stretch a single one.
TEST(image_references, reference_mode_latch_and_per_beat)
{
  ASSERT_TRUE(registered_());
  {
    Spec spec;
    spec.conds = {10.0f, 11.0f, 12.0f};
    spec.ref0  = {1.0f, 2.0f, 3.0f};
    spec.cfg.push_back({"reference_mode", FlexData::make_string("latch")});
    ASSERT_TRUE(generate_(std::move(spec)));
    ASSERT_TRUE(g_seen.calls == 3);
    ASSERT_TRUE(g_seen.refs.size() == 3);
    if (g_seen.refs.size() == 3) {
      EXPECT_TRUE((g_seen.refs[2] == std::vector<float>{1.0f}));
    }
  }
  {
    Spec spec;
    spec.conds = {10.0f, 11.0f, 12.0f};
    spec.ref0  = {1.0f};
    spec.cfg.push_back({"reference_mode", FlexData::make_string("per_beat")});
    ASSERT_TRUE(generate_(std::move(spec)));
    EXPECT_TRUE(g_seen.calls == 1);
  }
}

// seed_sequence, by the seed each generation is handed.
TEST(image_references, seed_sequence_increment_keep_randomize)
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
  // The default is increment.
  ASSERT_TRUE(run_seq(nullptr));
  EXPECT_TRUE((g_seen.seeds == std::vector<std::uint64_t>{42, 43, 44}));
  ASSERT_TRUE(run_seq("increment"));
  EXPECT_TRUE((g_seen.seeds == std::vector<std::uint64_t>{42, 43, 44}));
  ASSERT_TRUE(run_seq("keep"));
  EXPECT_TRUE((g_seen.seeds == std::vector<std::uint64_t>{42, 42, 42}));
  ASSERT_TRUE(run_seq("randomize"));
  if (g_seen.seeds.size() == 3) {
    // Distinct, and small enough to type back into `seed` through an
    // editor that holds numbers as doubles.
    EXPECT_TRUE(g_seen.seeds[0] != g_seen.seeds[1]);
    EXPECT_TRUE(g_seen.seeds[1] != g_seen.seeds[2]);
    for (std::uint64_t v : g_seen.seeds) {
      EXPECT_TRUE(v < (1ull << 53));
    }
  }
  // A broadcast run is still numbered by generation: one prompt, three
  // pictures, three seeds.
  Spec spec;
  spec.conds = {10.0f};
  spec.ref0  = {1.0f, 2.0f, 3.0f};
  spec.cfg.push_back({"seed", FlexData::make_int(7)});
  ASSERT_TRUE(generate_(std::move(spec)));
  EXPECT_TRUE((g_seen.seeds == std::vector<std::uint64_t>{7, 8, 9}));
}
