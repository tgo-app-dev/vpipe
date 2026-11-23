// train-lora-stage.cc -- the LoRA training stages as a graph runs them.
//
//   training_dataset  the folder scan: repeats, caption rules, the trigger
//                     word, buckets, and the three streams in the order
//                     the manifest says they come in.
//   train_lora        the whole stage on the small Qwen-Image-2.1 golden,
//                     fed synthetic conditioning and latents in the
//                     manifest's order: it trains, checkpoints, previews
//                     and reports. VPIPE_QWEN_IMAGE21_TRAIN_GOLDEN=<dir>.
//
//   vpipe_test --filter 'training_dataset*:train_lora*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-command.h"
#include "pipeline/typed-stage.h"
#include "stages/optimizer-select-stage.h"
#include "stages/train-lora-stage.h"
#include "stages/training-dataset-stage.h"
#include "stages/video-ref-encoder-stage.h"
#include "generative-models/train/sample-cache.h"
#include "generative-models/generative-model-manager.h"
#include "common/flex-bag.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-trainer.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory-host.h"
#include "minimax-h3-synthetic.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace vpipe;
namespace fs = std::filesystem;

namespace {

// A 24-bit BMP of w x h, a gradient keyed by `seed`.
void
write_bmp_(const fs::path& p, int w, int h, int seed)
{
  const int row = (w * 3 + 3) & ~3;
  const std::uint32_t data = (std::uint32_t)(row * h);
  std::vector<std::uint8_t> f(54 + data, 0);
  auto u32 = [&](int at, std::uint32_t v) { std::memcpy(&f[at], &v, 4); };
  auto u16 = [&](int at, std::uint16_t v) { std::memcpy(&f[at], &v, 2); };
  f[0] = 'B'; f[1] = 'M';
  u32(2, 54 + data); u32(10, 54); u32(14, 40);
  u32(18, (std::uint32_t)w); u32(22, (std::uint32_t)h);
  u16(26, 1); u16(28, 24); u32(34, data);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      std::uint8_t* px = &f[54 + (std::size_t)y * row + (std::size_t)x * 3];
      px[0] = (std::uint8_t)((x * 7 + seed * 31) & 255);
      px[1] = (std::uint8_t)((y * 5 + seed * 17) & 255);
      px[2] = (std::uint8_t)((x + y + seed * 13) & 255);
    }
  }
  std::ofstream(p, std::ios::binary)
      .write(reinterpret_cast<const char*>(f.data()),
             (std::streamsize)f.size());
}

// A source of prepared beats on one port, then end of stream.
class Beats : public TypedStage<Beats> {
public:
  static constexpr const char* kTypeName = "ut-train-beats";
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

// Records every beat of one input.
class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-train-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> got;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    got.push_back(std::move(in));
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

std::string
text_of_(const BeatPayloadIntf* b)
{
  const auto* f = dynamic_cast<const FlexDataPayload*>(b);
  if (f == nullptr) { return "<not text>"; }
  if (f->data.is_string()) { return std::string(f->data.as_string("")); }
  FlexData d = f->data;
  auto o = d.as_object();
  return o.contains(beat::kText) ? std::string(o.at(beat::kText).as_string(""))
                                 : std::string();
}

}  // namespace

// ---- training-dataset -------------------------------------------------------

TEST(training_dataset, trigger_rules_and_buckets)
{
  using S = TrainingDatasetStage;
  EXPECT_TRUE(S::apply_trigger("a cat", "ohwx", "{trigger}, {caption}") ==
              "ohwx, a cat");
  // Already named, as a word and in any case: untouched.
  EXPECT_TRUE(S::apply_trigger("Photo of OHWX on grass", "ohwx",
                               "{trigger}, {caption}") ==
              "Photo of OHWX on grass");
  // A longer word containing it is NOT the trigger.
  EXPECT_TRUE(S::apply_trigger("ohwxyz style", "ohwx",
                               "{trigger}, {caption}") ==
              "ohwx, ohwxyz style");
  // The placeholder, in place.
  EXPECT_TRUE(S::apply_trigger("a [trigger] dog", "sks",
                               "{trigger}, {caption}") == "a sks dog");
  EXPECT_TRUE(S::apply_trigger("", "sks", "{trigger}, {caption}") == "sks");
  EXPECT_TRUE(S::apply_trigger("plain", "", "{trigger}, {caption}") ==
              "plain");

  int w = 0, h = 0;
  S::bucket(1000, 1000, 1024, 32, 2.0, &w, &h);
  EXPECT_TRUE(w == 1024 && h == 1024);
  S::bucket(1600, 900, 1024, 32, 2.0, &w, &h);     // 16:9
  EXPECT_TRUE(w % 32 == 0 && h % 32 == 0 && w > h);
  EXPECT_TRUE(std::abs(w * h - 1024 * 1024) < 1024 * 1024 / 10);
  S::bucket(4000, 500, 512, 32, 2.0, &w, &h);      // clamped to 2:1
  EXPECT_TRUE(w == 2 * h || std::abs(w - 2 * h) <= 32);
}

TEST(training_dataset, scans_a_folder_into_the_three_streams)
{
  const fs::path root =
      fs::temp_directory_path() / "vpipe-ut-training-dataset";
  fs::remove_all(root);
  fs::create_directories(root / "3_subject");
  write_bmp_(root / "3_subject" / "a.bmp", 96, 64, 1);
  std::ofstream(root / "3_subject" / "a.txt") << "a red car\nthe car, side\n";
  write_bmp_(root / "3_subject" / "b.bmp", 64, 64, 2);
  std::ofstream(root / "3_subject" / "b.txt") << "ohwx parked outside\n";
  write_bmp_(root / "c.bmp", 64, 96, 3);           // no caption

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("dir", FlexData::make_string(root.string()));
    o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
    o.insert_or_assign("resolution", FlexData::make_int(256));
    FlexData pv = FlexData::make_array();
    pv.as_array().push_back(FlexData::make_string("{trigger} at night"));
    o.insert_or_assign("preview_prompts", std::move(pv));
  }
  auto ds_u = std::make_unique<TrainingDatasetStage>(
      &sess, "ds", std::vector<InEdge>{}, cfg);
  ASSERT_TRUE(ds_u->config_error().empty());
  auto* ds = pl->insert_stage(std::move(ds_u));
  Sink* sinks[3];
  for (int p = 0; p < 3; ++p) {
    auto s = std::make_unique<Sink>(&sess, "sink" + std::to_string(p),
                                    std::vector<InEdge>{{ds, (unsigned)p}},
                                    FlexData::make_object());
    sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(s)));
  }
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(sinks[2]->got.size() == 1);
  const auto* mf = dynamic_cast<const FlexDataPayload*>(sinks[2]->got[0].get());
  ASSERT_TRUE(mf != nullptr);
  FlexData man = mf->data;
  auto mo = man.as_object();
  EXPECT_TRUE(mo.at("captions_total").as_int(0) == 4);
  EXPECT_TRUE(mo.at("pictures_total").as_int(0) == 3);
  const FlexData samples = mo.at("samples");
  auto sa = samples.as_array();
  ASSERT_TRUE(sa.size() == 3);
  // Sorted: 3_subject/a, 3_subject/b, c.
  {
    const FlexData a = sa.at(0);
    auto ao = a.as_object();
    EXPECT_TRUE(ao.at("repeats").as_int(0) == 3);
    const FlexData caps = ao.at("captions");
    EXPECT_TRUE(caps.as_array().size() == 2);
    EXPECT_TRUE(std::string(caps.as_array().at(0).as_string("")) ==
                "ohwx, a red car");
    const FlexData pics = ao.at("pictures");
    auto p0 = pics.as_array().at(0);
    // 96x64 at 256^2 area: wider than tall, multiples of 32.
    EXPECT_TRUE(p0.as_object().at("w").as_int(0) >
                p0.as_object().at("h").as_int(0));
  }
  {
    const FlexData c = sa.at(2);
    auto co = c.as_object();
    EXPECT_TRUE(co.at("repeats").as_int(0) == 1);
    const FlexData caps = co.at("captions");
    EXPECT_TRUE(std::string(caps.as_array().at(0).as_string("")) == "ohwx");
  }
  // Captions: the four variants, the empty prompt, one preview.
  ASSERT_TRUE(sinks[1]->got.size() == 6);
  EXPECT_TRUE(text_of_(sinks[1]->got[2].get()) == "ohwx parked outside");
  EXPECT_TRUE(text_of_(sinks[1]->got[4].get()).empty());
  EXPECT_TRUE(text_of_(sinks[1]->got[5].get()) == "ohwx at night");
  // Pictures: one per sample at its bucket, marked as a batch.
  ASSERT_TRUE(sinks[0]->got.size() == 3);
  for (const auto& g : sinks[0]->got) {
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(g.get());
    ASSERT_TRUE(tb != nullptr);
    if (tb == nullptr) { continue; }
    EXPECT_TRUE(tb->dtype == TensorBeat::DType::U8 && tb->shape.size() == 3 &&
                tb->shape[0] == 3 && tb->shape[1] % 32 == 0 &&
                tb->shape[2] % 32 == 0);
    FlexData sb = tb->sideband;
    EXPECT_TRUE(sb.is_object() &&
                sb.as_object().at(beat::kBatch).as_bool(false));
  }
  fs::remove_all(root);
}

// ---- train-lora -------------------------------------------------------------

TEST(train_lora, trains_checkpoints_and_previews_on_the_small_golden)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const fs::path gd(g);
  if (!fs::is_directory(gd / "train")) { return; }
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-lora";
  fs::remove_all(root);
  fs::create_directories(root / "model");
  fs::create_directory_symlink(gd / "train", root / "model" / "transformer");
  const fs::path out = root / "out";

  // Three samples of one caption and one 8x8-latent picture each.
  const int N = 3, TXT = 11, D = 256, C = 64, G = 8;
  FlexData man = FlexData::make_object();
  {
    auto o = man.as_object();
    o.insert_or_assign("kind", FlexData::make_string("training-manifest"));
    o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
    FlexData arr = FlexData::make_array();
    for (int i = 0; i < N; ++i) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert_or_assign("file", FlexData::make_string("s" +
                                                        std::to_string(i)));
      eo.insert_or_assign("repeats", FlexData::make_int(1));
      FlexData caps = FlexData::make_array();
      caps.as_array().push_back(FlexData::make_string("c"));
      eo.insert_or_assign("captions", std::move(caps));
      FlexData pics = FlexData::make_array();
      FlexData po = FlexData::make_object();
      po.as_object().insert_or_assign("w", FlexData::make_int(G * 16));
      po.as_object().insert_or_assign("h", FlexData::make_int(G * 16));
      pics.as_array().push_back(std::move(po));
      eo.insert_or_assign("pictures", std::move(pics));
      arr.as_array().push_back(std::move(e));
    }
    o.insert_or_assign("samples", std::move(arr));
    FlexData pv = FlexData::make_array();
    pv.as_array().push_back(FlexData::make_string("ohwx"));
    o.insert_or_assign("preview_prompts", std::move(pv));
  }
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto cond = [&]() {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {TXT, D};
    tb->resize_contiguous((std::size_t)TXT * D);
    auto* d = reinterpret_cast<std::uint16_t*>(tb->data.data());
    for (int i = 0; i < TXT * D; ++i) {
      float f = nd(rng);
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      d[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return tb;
  };
  auto latent = [&]() {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::F32;
    tb->shape = {C, G, G};
    tb->resize_contiguous((std::size_t)C * G * G);
    for (std::size_t i = 0; i < (std::size_t)C * G * G; ++i) {
      tb->as_f32()[i] = nd(rng);
    }
    return tb;
  };

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src = [&](const char* id) {
    auto u = std::make_unique<Beats>(&sess, id, std::vector<InEdge>{},
                                     FlexData::make_object());
    u->allocate_oports(1);
    return static_cast<Beats*>(pl->insert_stage(std::move(u)));
  };
  Beats* conds = src("conds");
  Beats* lats = src("lats");
  Beats* mans = src("man");
  for (int i = 0; i < N; ++i) {
    conds->beats.push_back(cond());
    lats->beats.push_back(latent());
  }
  conds->beats.push_back(cond());   // the empty prompt
  conds->beats.push_back(cond());   // the preview prompt
  mans->beats.push_back(make_payload<FlexDataPayload>(man));

  FlexData ocfg = FlexData::make_object();
  ocfg.as_object().insert_or_assign("learning_rate",
                                    FlexData::make_real(1e-2));
  ocfg.as_object().insert_or_assign("warmup_steps", FlexData::make_int(0));
  auto opt = pl->insert_stage(std::make_unique<OptimizerSelectStage>(
      &sess, "opt", std::vector<InEdge>{}, ocfg));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string((root / "model")
                                                           .string()));
    o.insert_or_assign("output_dir", FlexData::make_string(out.string()));
    o.insert_or_assign("name", FlexData::make_string("ut-lora"));
    o.insert_or_assign("rank", FlexData::make_int(4));
    o.insert_or_assign("steps", FlexData::make_int(20));
    o.insert_or_assign("save_every", FlexData::make_int(10));
    o.insert_or_assign("preview_steps", FlexData::make_int(3));
    o.insert_or_assign("caption_dropout", FlexData::make_real(0.0));
    o.insert_or_assign("register", FlexData::make_bool(false));
    o.insert_or_assign("resume", FlexData::make_string("never"));
  }
  const InEdge none{nullptr, 0};
  std::vector<InEdge> in = {{conds, 0}, {lats, 0}, {mans, 0}, none, {opt, 0}};
  auto tr_u = std::make_unique<TrainLoraStage>(&sess, "train", in, cfg);
  ASSERT_TRUE(tr_u->config_error().empty());
  if (!tr_u->config_error().empty()) {
    std::printf("[train_lora] %s\n", tr_u->config_error().c_str());
    return;
  }
  auto* tr = pl->insert_stage(std::move(tr_u));
  Sink* sinks[3];
  for (int p = 0; p < 3; ++p) {
    auto s = std::make_unique<Sink>(&sess, "tsink" + std::to_string(p),
                                    std::vector<InEdge>{{tr, (unsigned)p}},
                                    FlexData::make_object());
    sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(s)));
  }
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Twenty metrics beats, and the loss went down.
  ASSERT_TRUE(sinks[2]->got.size() == 20);
  auto loss_at = [&](std::size_t i) {
    const auto* f =
        dynamic_cast<const FlexDataPayload*>(sinks[2]->got[i].get());
    FlexData d = f->data;
    return d.as_object().at("loss").as_real(0.0);
  };
  double first = 0.0, last = 0.0;
  for (std::size_t i = 0; i < 3; ++i) { first += loss_at(i); }
  for (std::size_t i = 17; i < 20; ++i) { last += loss_at(i); }
  std::printf("[train_lora] loss %.4f -> %.4f (3-step means)\n", first / 3,
              last / 3);
  EXPECT_TRUE(last < first);
  // A checkpoint at 10 and the final adapter; their beats.
  EXPECT_TRUE(fs::exists(out / "ut-lora-000010.safetensors"));
  EXPECT_TRUE(fs::exists(out / "ut-lora.safetensors"));
  EXPECT_TRUE(!fs::exists(out / "ut-lora.resume"));
  EXPECT_TRUE(sinks[1]->got.size() == 2);
  // A preview at 10 and at 20, [64, 8, 8] each.
  ASSERT_TRUE(sinks[0]->got.size() == 2);
  for (const auto& p : sinks[0]->got) {
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    ASSERT_TRUE(tb != nullptr);
    if (tb == nullptr) { continue; }
    EXPECT_TRUE(tb->shape.size() == 3 && tb->shape[0] == C &&
                tb->shape[1] == G && tb->shape[2] == G);
    bool finite = true;
    for (std::size_t i = 0; i < (std::size_t)C * G * G; ++i) {
      finite = finite && std::isfinite(tb->as_f32()[i]);
    }
    EXPECT_TRUE(finite);
  }
  fs::remove_all(root);
}

// THE CACHE, end to end on the small golden. Run A encodes everything and
// trains from RAM, writing the cache as the beats arrive; run B is the
// same manifest with every entry marked cached -- no conditioning, no
// latents arrive -- training from DISK; run C, from the cache into RAM.
// All three must be the SAME adapter, bit for bit: the latent is rounded
// to the storage type before it is cached, so the cache changes nothing.
// One sample is held out: it never trains, and its loss is reported at
// every checkpoint, identically in all three runs.
TEST(train_lora, a_cached_rerun_from_disk_trains_the_same_adapter)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const fs::path gd(g);
  if (!fs::is_directory(gd / "train")) { return; }
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-cache";
  fs::remove_all(root);
  fs::create_directories(root / "model");
  fs::create_directory_symlink(gd / "train", root / "model" / "transformer");
  const fs::path out = root / "out";
  const fs::path cache = root / "cache";

  const int N = 4, TXT = 11, D = 256, C = 64, G = 8;
  auto manifest = [&](bool cached) {
    FlexData man = FlexData::make_object();
    auto o = man.as_object();
    o.insert_or_assign("kind", FlexData::make_string("training-manifest"));
    o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
    o.insert_or_assign("cache_dir", FlexData::make_string(cache.string()));
    FlexData arr = FlexData::make_array();
    for (int i = 0; i < N; ++i) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert_or_assign("file", FlexData::make_string("s" +
                                                        std::to_string(i)));
      eo.insert_or_assign("repeats", FlexData::make_int(1));
      if (i == N - 1) {
        eo.insert_or_assign("validation", FlexData::make_bool(true));
      }
      FlexData caps = FlexData::make_array(), ck = FlexData::make_array(),
               cc = FlexData::make_array();
      caps.as_array().push_back(FlexData::make_string("c"));
      ck.as_array().push_back(FlexData::make_string("t" + std::to_string(i)));
      cc.as_array().push_back(FlexData::make_bool(cached));
      eo.insert_or_assign("captions", std::move(caps));
      eo.insert_or_assign("caption_keys", std::move(ck));
      eo.insert_or_assign("caption_cached", std::move(cc));
      FlexData pics = FlexData::make_array();
      FlexData po = FlexData::make_object();
      po.as_object().insert_or_assign("w", FlexData::make_int(G * 16));
      po.as_object().insert_or_assign("h", FlexData::make_int(G * 16));
      po.as_object().insert_or_assign(
          "key", FlexData::make_string("l" + std::to_string(i)));
      po.as_object().insert_or_assign("cached", FlexData::make_bool(cached));
      pics.as_array().push_back(std::move(po));
      eo.insert_or_assign("pictures", std::move(pics));
      arr.as_array().push_back(std::move(e));
    }
    o.insert_or_assign("samples", std::move(arr));
    FlexData pv = FlexData::make_array(), pk = FlexData::make_array(),
             pc = FlexData::make_array();
    pv.as_array().push_back(FlexData::make_string("ohwx"));
    pk.as_array().push_back(FlexData::make_string("pv"));
    pc.as_array().push_back(FlexData::make_bool(cached));
    o.insert_or_assign("preview_prompts", std::move(pv));
    o.insert_or_assign("preview_keys", std::move(pk));
    o.insert_or_assign("preview_cached", std::move(pc));
    o.insert_or_assign("null_key", FlexData::make_string("null"));
    o.insert_or_assign("null_cached", FlexData::make_bool(cached));
    return man;
  };
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto cond = [&]() {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {TXT, D};
    tb->resize_contiguous((std::size_t)TXT * D);
    auto* d = reinterpret_cast<std::uint16_t*>(tb->data.data());
    for (int i = 0; i < TXT * D; ++i) {
      float f = nd(rng);
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      d[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return tb;
  };
  auto latent = [&]() {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::F32;
    tb->shape = {C, G, G};
    tb->resize_contiguous((std::size_t)C * G * G);
    for (std::size_t i = 0; i < (std::size_t)C * G * G; ++i) {
      tb->as_f32()[i] = nd(rng);
    }
    return tb;
  };

  // One run: the stage fed `man`, and the beats when `feed`.
  auto run = [&](const std::string& name, bool cached, const char* mode,
                 std::vector<double>* val) {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto src = [&](const char* id) {
      auto u = std::make_unique<Beats>(&sess, id, std::vector<InEdge>{},
                                       FlexData::make_object());
      u->allocate_oports(1);
      return static_cast<Beats*>(pl->insert_stage(std::move(u)));
    };
    Beats* conds = src("conds");
    Beats* lats = src("lats");
    Beats* mans = src("man");
    if (!cached) {
      for (int i = 0; i < N; ++i) {
        conds->beats.push_back(cond());
        lats->beats.push_back(latent());
      }
      conds->beats.push_back(cond());   // the empty prompt
      conds->beats.push_back(cond());   // the preview prompt
    }
    mans->beats.push_back(make_payload<FlexDataPayload>(manifest(cached)));
    FlexData ocfg = FlexData::make_object();
    ocfg.as_object().insert_or_assign("learning_rate",
                                      FlexData::make_real(1e-2));
    ocfg.as_object().insert_or_assign("warmup_steps", FlexData::make_int(0));
    auto opt = pl->insert_stage(std::make_unique<OptimizerSelectStage>(
        &sess, "opt", std::vector<InEdge>{}, ocfg));
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert_or_assign("hf_dir", FlexData::make_string((root / "model")
                                                             .string()));
      o.insert_or_assign("output_dir", FlexData::make_string(out.string()));
      o.insert_or_assign("name", FlexData::make_string(name));
      o.insert_or_assign("rank", FlexData::make_int(4));
      o.insert_or_assign("steps", FlexData::make_int(12));
      o.insert_or_assign("save_every", FlexData::make_int(6));
      o.insert_or_assign("preview_steps", FlexData::make_int(2));
      o.insert_or_assign("caption_dropout", FlexData::make_real(0.25));
      o.insert_or_assign("register", FlexData::make_bool(false));
      o.insert_or_assign("resume", FlexData::make_string("never"));
      o.insert_or_assign("cache", FlexData::make_string(mode));
    }
    const InEdge none{nullptr, 0};
    std::vector<InEdge> in = {{conds, 0}, {lats, 0}, {mans, 0}, none,
                              {opt, 0}};
    auto tr_u = std::make_unique<TrainLoraStage>(&sess, "train", in, cfg);
    if (!tr_u->config_error().empty()) { return false; }
    auto* tr = pl->insert_stage(std::move(tr_u));
    auto ck = std::make_unique<Sink>(&sess, "ck",
                                     std::vector<InEdge>{{tr, 1u}},
                                     FlexData::make_object());
    Sink* cks = static_cast<Sink*>(pl->insert_stage(std::move(ck)));
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) { return false; }
    rt.wait_idle();
    rt.stop();
    for (const auto& b : cks->got) {
      const auto* f = dynamic_cast<const FlexDataPayload*>(b.get());
      if (f == nullptr) { continue; }
      FlexData d = f->data;
      auto o = d.as_object();
      if (o.contains("val_loss")) {
        val->push_back(o.at("val_loss").as_real(-1.0));
      }
    }
    return fs::exists(out / (name + ".safetensors"));
  };
  auto bytes_of = [](const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  };

  std::vector<double> va, vb, vc;
  ASSERT_TRUE(run("a", false, "ram", &va));
  // The cache holds every text (four captions, the empty prompt, the
  // preview) and every latent.
  EXPECT_TRUE(genai::train::SampleCache::keys_in(cache.string()).size() ==
              (std::size_t)(2 * N + 2));
  ASSERT_TRUE(run("b", true, "disk", &vb));
  ASSERT_TRUE(run("c", true, "ram", &vc));
  const auto fa = bytes_of(out / "a.safetensors");
  const auto fb = bytes_of(out / "b.safetensors");
  const auto fc = bytes_of(out / "c.safetensors");
  std::printf("[train_lora] adapters %zu / %zu / %zu bytes, held-out %zu "
              "scores\n", fa.size(), fb.size(), fc.size(), va.size());
  EXPECT_TRUE(!fa.empty() && fa == fb && fa == fc);
  // A checkpoint at 6 (the final beat carries one too).
  ASSERT_TRUE(va.size() == 2);
  EXPECT_TRUE(va == vb && va == vc);
  EXPECT_TRUE(std::isfinite(va[0]) && va[0] > 0.0);
  fs::remove_all(root);
}

TEST(training_dataset, keys_hold_out_and_the_validation_count)
{
  using S = TrainingDatasetStage;
  EXPECT_TRUE(S::validation_count(-1, 100) == 0);
  EXPECT_TRUE(S::validation_count(-1, 500) == 5);
  EXPECT_TRUE(S::validation_count(-1, 20000) == 64);
  EXPECT_TRUE(S::validation_count(3, 10) == 3);
  EXPECT_TRUE(S::validation_count(50, 10) == 9);   // one always trains
  std::vector<S::Sample> v(10);
  for (int i = 0; i < 10; ++i) {
    v[(std::size_t)i].rel = "p" + std::to_string(i) + ".png";
    v[(std::size_t)i].captions = {"ohwx, a cat", "ohwx"};
    v[(std::size_t)i].pictures = {S::Picture{64, 64, false}};
  }
  // Held out: the same ones every time for one seed.
  S::hold_out(&v, 3, 7);
  std::vector<bool> first;
  for (const auto& s : v) { first.push_back(s.validation); }
  S::hold_out(&v, 3, 7);
  int n = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    EXPECT_TRUE(v[i].validation == first[i]);
    n += v[i].validation ? 1 : 0;
  }
  EXPECT_TRUE(n == 3);
  // Keys: a caption's is its text's, so equal captions share one; a
  // picture's names its file, bucket and flip.
  S::key_samples(&v, false, {});
  EXPECT_TRUE(v[0].caption_keys[0] == v[1].caption_keys[0]);
  EXPECT_TRUE(v[0].caption_keys[0] != v[0].caption_keys[1]);
  EXPECT_TRUE(v[0].pictures[0].key != v[1].pictures[0].key);
  const std::string k0 = v[0].pictures[0].key;
  v[0].pictures[0].flip = true;
  S::key_samples(&v, false, {k0});
  EXPECT_TRUE(v[0].pictures[0].key != k0 && !v[0].pictures[0].cached);
  v[0].pictures[0].flip = false;
  S::key_samples(&v, false, {k0, v[1].caption_keys[1]});
  EXPECT_TRUE(v[0].pictures[0].cached);
  EXPECT_TRUE(!v[0].caption_cached[0] && v[0].caption_cached[1]);
  // RGBA is a different encode.
  S::key_samples(&v, true, {k0});
  EXPECT_TRUE(!v[0].pictures[0].cached);
}

// A RE-RUN EMITS ONLY WHAT IS NOT CACHED. The stage wired to a model and a
// cache: the first run emits everything and keys it; with some of those
// keys then in the cache, the second run marks them cached in its
// manifest and does not emit them -- the conditioner and the VAE are never
// asked for them.
TEST(training_dataset, a_rerun_emits_only_what_is_not_cached)
{
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-dataset-cache";
  fs::remove_all(root);
  fs::create_directories(root / "pics");
  fs::create_directories(root / "model");
  write_bmp_(root / "pics" / "a.bmp", 64, 64, 1);
  std::ofstream(root / "pics" / "a.txt") << "a red car\n";
  write_bmp_(root / "pics" / "b.bmp", 64, 64, 2);
  std::ofstream(root / "pics" / "b.txt") << "a blue car\n";

  // One run: the manifest, and the texts and picture count emitted.
  auto run = [&](FlexData* man, std::vector<std::string>* texts,
                 std::size_t* pictures) {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto mu = std::make_unique<Beats>(&sess, "model", std::vector<InEdge>{},
                                      FlexData::make_object());
    mu->allocate_oports(1);
    mu->beats.push_back(make_payload<FlexDataPayload>(
        FlexData::make_string((root / "model").string())));
    auto* model = pl->insert_stage(std::move(mu));
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert_or_assign("dir", FlexData::make_string((root / "pics")
                                                          .string()));
      o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
      o.insert_or_assign("resolution", FlexData::make_int(128));
      o.insert_or_assign("cache_dir",
                         FlexData::make_string((root / "cache").string()));
    }
    auto ds = pl->insert_stage(std::make_unique<TrainingDatasetStage>(
        &sess, "ds", std::vector<InEdge>{{model, 0}}, cfg));
    Sink* sinks[3];
    for (int p = 0; p < 3; ++p) {
      auto s = std::make_unique<Sink>(&sess, "sink" + std::to_string(p),
                                      std::vector<InEdge>{{ds, (unsigned)p}},
                                      FlexData::make_object());
      sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(s)));
    }
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) { return false; }
    rt.wait_idle();
    rt.stop();
    if (sinks[2]->got.size() != 1) { return false; }
    *man = dynamic_cast<const FlexDataPayload*>(sinks[2]->got[0].get())->data;
    texts->clear();
    for (const auto& b : sinks[1]->got) { texts->push_back(text_of_(b.get())); }
    *pictures = sinks[0]->got.size();
    return true;
  };

  FlexData m1, m2;
  std::vector<std::string> t1, t2;
  std::size_t p1 = 0, p2 = 0;
  ASSERT_TRUE(run(&m1, &t1, &p1));
  // Everything: two captions, the empty prompt, the preview; two pictures.
  EXPECT_TRUE(t1.size() == 4 && p1 == 2);
  auto mo = m1.as_object();
  ASSERT_TRUE(mo.contains("cache_dir"));
  const std::string cdir(mo.at("cache_dir").as_string(""));
  // The cache names the model: it sits one level under cache_dir.
  EXPECT_TRUE(fs::path(cdir).parent_path() == root / "cache");
  const FlexData samples = mo.at("samples");
  const FlexData s0 = samples.as_array().at(0);
  const std::string cap0(
      s0.as_object().at("caption_keys").as_array().at(0).as_string(""));
  const std::string pic0(s0.as_object()
                             .at("pictures")
                             .as_array()
                             .at(0)
                             .as_object()
                             .at("key")
                             .as_string(""));
  const std::string null_key(mo.at("null_key").as_string(""));
  {
    // What a train-lora run would have left: sample 0 and the empty
    // prompt.
    genai::train::SampleCache c;
    std::string err;
    ASSERT_TRUE(c.open(cdir, &err));
    c.put(cap0, "BF16", {1, 2}, std::vector<std::uint8_t>(4, 0), "");
    c.put(pic0, "BF16", {1, 1, 1}, std::vector<std::uint8_t>(2, 0), "");
    c.put(null_key, "BF16", {1, 2}, std::vector<std::uint8_t>(4, 0), "");
    ASSERT_TRUE(c.flush(&err));
  }
  ASSERT_TRUE(run(&m2, &t2, &p2));
  // Sample 1's caption and the preview; sample 1's picture.
  EXPECT_TRUE(t2.size() == 2 && p2 == 1);
  if (t2.size() == 2) {
    EXPECT_TRUE(t2[0] == "ohwx, a blue car");
    EXPECT_TRUE(t2[1] == "ohwx");
  }
  auto m2o = m2.as_object();
  const FlexData s20 = m2o.at("samples").as_array().at(0);
  EXPECT_TRUE(s20.as_object().at("caption_cached").as_array().at(0)
                  .as_bool(false));
  EXPECT_TRUE(s20.as_object().at("pictures").as_array().at(0).as_object()
                  .at("cached").as_bool(false));
  EXPECT_TRUE(m2o.at("null_cached").as_bool(false));
  fs::remove_all(root);
}

// LIVE COMMANDS. A run takes `save` and `preview` between its steps: a
// save before training starts is refused with the reason, then writes a
// checkpoint mid-run and answers with it; a preview of one prompt emits
// exactly one latent; a prompt that does not exist is refused.
TEST(train_lora, save_and_preview_commands_serve_between_steps)
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_TRAIN_GOLDEN");
  if (g == nullptr || *g == '\0') { return; }
  const fs::path gd(g);
  if (!fs::is_directory(gd / "train")) { return; }
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-cmds";
  fs::remove_all(root);
  fs::create_directories(root / "model");
  fs::create_directory_symlink(gd / "train", root / "model" / "transformer");
  const fs::path out = root / "out";

  const int N = 2, TXT = 11, D = 256, C = 64, G = 8, STEPS = 400;
  FlexData man = FlexData::make_object();
  {
    auto o = man.as_object();
    o.insert_or_assign("kind", FlexData::make_string("training-manifest"));
    o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
    FlexData arr = FlexData::make_array();
    for (int i = 0; i < N; ++i) {
      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert_or_assign("file", FlexData::make_string("s" +
                                                        std::to_string(i)));
      eo.insert_or_assign("repeats", FlexData::make_int(1));
      FlexData caps = FlexData::make_array();
      caps.as_array().push_back(FlexData::make_string("c"));
      eo.insert_or_assign("captions", std::move(caps));
      FlexData pics = FlexData::make_array();
      FlexData po = FlexData::make_object();
      po.as_object().insert_or_assign("w", FlexData::make_int(G * 16));
      po.as_object().insert_or_assign("h", FlexData::make_int(G * 16));
      pics.as_array().push_back(std::move(po));
      eo.insert_or_assign("pictures", std::move(pics));
      arr.as_array().push_back(std::move(e));
    }
    o.insert_or_assign("samples", std::move(arr));
    FlexData pv = FlexData::make_array();
    pv.as_array().push_back(FlexData::make_string("ohwx"));
    pv.as_array().push_back(FlexData::make_string("ohwx at night"));
    o.insert_or_assign("preview_prompts", std::move(pv));
  }
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto cond = [&]() {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {TXT, D};
    tb->resize_contiguous((std::size_t)TXT * D);
    auto* d = reinterpret_cast<std::uint16_t*>(tb->data.data());
    for (int i = 0; i < TXT * D; ++i) {
      float f = nd(rng);
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      d[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return tb;
  };
  auto latent = [&]() {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::F32;
    tb->shape = {C, G, G};
    tb->resize_contiguous((std::size_t)C * G * G);
    for (std::size_t i = 0; i < (std::size_t)C * G * G; ++i) {
      tb->as_f32()[i] = nd(rng);
    }
    return tb;
  };

  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src = [&](const char* id) {
    auto u = std::make_unique<Beats>(&sess, id, std::vector<InEdge>{},
                                     FlexData::make_object());
    u->allocate_oports(1);
    return static_cast<Beats*>(pl->insert_stage(std::move(u)));
  };
  Beats* conds = src("conds");
  Beats* lats = src("lats");
  Beats* mans = src("man");
  for (int i = 0; i < N; ++i) {
    conds->beats.push_back(cond());
    lats->beats.push_back(latent());
  }
  conds->beats.push_back(cond());   // the empty prompt
  conds->beats.push_back(cond());   // the two preview prompts
  conds->beats.push_back(cond());
  mans->beats.push_back(make_payload<FlexDataPayload>(man));
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string((root / "model")
                                                           .string()));
    o.insert_or_assign("output_dir", FlexData::make_string(out.string()));
    o.insert_or_assign("name", FlexData::make_string("ut-cmd"));
    o.insert_or_assign("rank", FlexData::make_int(4));
    o.insert_or_assign("steps", FlexData::make_int(STEPS));
    // No scheduled checkpoint before the end, and so no scheduled
    // preview: what the previews port carries early is the command's.
    o.insert_or_assign("save_every", FlexData::make_int(100000));
    o.insert_or_assign("preview_steps", FlexData::make_int(2));
    o.insert_or_assign("register", FlexData::make_bool(false));
    o.insert_or_assign("resume", FlexData::make_string("never"));
  }
  const InEdge none{nullptr, 0};
  std::vector<InEdge> in = {{conds, 0}, {lats, 0}, {mans, 0}, none, none};
  auto tr_u = std::make_unique<TrainLoraStage>(&sess, "train", in, cfg);
  ASSERT_TRUE(tr_u->config_error().empty());
  if (!tr_u->config_error().empty()) { return; }
  auto* tr = pl->insert_stage(std::move(tr_u));
  Sink* sinks[2];
  for (int p = 0; p < 2; ++p) {
    auto s = std::make_unique<Sink>(&sess, "tsink" + std::to_string(p),
                                    std::vector<InEdge>{{tr, (unsigned)p}},
                                    FlexData::make_object());
    sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(s)));
  }
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());

  // `save` until training has started: refused, with the reason, before.
  FlexData saved;
  int refused = 0;
  for (int i = 0; i < 2000; ++i) {
    auto c = open_stage_command(*tr, "save", FlexData::make_object(), {});
    const CommandState st = c->wait(10000);
    if (st == CommandState::Replied) {
      saved = c->result();
      break;
    }
    if (st == CommandState::Failed &&
        c->error().find("not started") != std::string::npos) {
      ++refused;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    std::printf("[train_lora] save: %s\n", c->error().c_str());
    break;
  }
  ASSERT_TRUE(saved.is_object());
  FlexData pr, bad;
  {
    FlexData a = FlexData::make_object();
    a.as_object().insert_or_assign("prompt", FlexData::make_int(1));
    auto c = open_stage_command(*tr, "preview", std::move(a), {});
    if (c->wait(30000) == CommandState::Replied) { pr = c->result(); }
    FlexData b = FlexData::make_object();
    b.as_object().insert_or_assign("prompt", FlexData::make_int(5));
    auto c2 = open_stage_command(*tr, "preview", std::move(b), {});
    EXPECT_TRUE(c2->wait(30000) == CommandState::Failed);
  }
  rt.wait_idle();
  rt.stop();

  if (saved.is_object()) {
    auto so = saved.as_object();
    const int step = (int)so.at("step").as_int(-1);
    const std::string path(so.at("path").as_string(""));
    std::printf("[train_lora] save at step %d of %d ('%s'), %d refusals "
                "before training\n", step, STEPS,
                fs::path(path).filename().c_str(), refused);
    EXPECT_TRUE(step >= 0 && step < STEPS);
    EXPECT_TRUE(fs::exists(path));
  }
  ASSERT_TRUE(pr.is_object());
  if (pr.is_object()) {
    EXPECT_TRUE(pr.as_object().at("previews").as_int(0) == 1);
  }
  // The command's checkpoint and the final adapter; the command's
  // preview (prompt 1 alone) and the final two.
  EXPECT_TRUE(sinks[1]->got.size() == 2);
  ASSERT_TRUE(sinks[0]->got.size() == 3);
  {
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sinks[0]->got[0].get());
    ASSERT_TRUE(tb != nullptr);
    FlexData sb = tb->sideband;
    EXPECT_TRUE(std::string(sb.as_object().at("prompt").as_string("")) ==
                "ohwx at night");
  }
  EXPECT_TRUE(fs::exists(out / "ut-cmd.safetensors"));
  fs::remove_all(root);
}

// ---- MiniMax-H3 ----------------------------------------------------------
//
// The whole stage on a synthetic H3 (minimax-h3-synthetic.h: the released
// layout at hidden 256, two blocks and a refiner), fed what a still
// dataset's graph delivers -- the conditioner's bf16 [rows, 256] and
// vae-encode's still latent, f32 [24, 1, h, w] -- in the manifest's order.
// No model env: it runs everywhere.

namespace {

// Records every beat of one input and hands it to a callback first.
class CbSink : public TypedStage<CbSink> {
public:
  static constexpr const char* kTypeName = "ut-train-cb-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> got;
  std::function<void(const BeatPayloadIntf*)> on;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (on) { on(in.get()); }
    got.push_back(std::move(in));
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

constexpr int kH3N = 3, kH3Txt = 8, kH3D = 256, kH3Z = 24, kH3G = 8;

// One H3 training run through the stage. The beats are a pure function of
// the seed, so two runs see the same dataset.
struct H3Run {
  fs::path model, out;
  int steps = 12, save_every = 6, preview_every = 0;
  std::string resume = "never";
  // CLIPS: each sample a window of `clip_frames` pixel frames (0: a
  // still) -- its latent [24, T, h, w] -- with, when `audio`, its
  // soundtrack latent [2, 32, n] on the audio_latents port.
  int clip_frames = 0;
  bool audio = false;
  // KEYFRAMES (bit 0 first, bit 1 last): each window followed on the
  // latents port by its anchors' stills, and its caption presented with
  // them -- a conditioning of its own whose first 4 rows are a vision
  // block, as the conditioner tags them.
  int anchors = 0;
  // REFERENCES (Ref2VA): every caption encoded with a picture (4 x 6
  // latents) and a soundtrack of 7 latents, as video-ref-encoder emits it
  // -- the plan on its sideband, its rows on two ports beside it (empty
  // for the empty and the preview prompts).
  bool refs = false;
  std::string recompute;   // the stage's `recompute`, when set
  std::function<void(PipelineRuntime*, const BeatPayloadIntf*)> on_ckpt;
  std::vector<std::unique_ptr<BeatPayloadIntf>> previews, ckpts, metrics,
      preview_audio;
  std::string config_error;
  // What the plan counted for the DiT's weights at the first checkpoint,
  // while the model was loaded.
  std::size_t planned = 0;

  void
  run()
  {
    FlexData man = FlexData::make_object();
    {
      auto o = man.as_object();
      o.insert_or_assign("kind", FlexData::make_string("training-manifest"));
      o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
      FlexData arr = FlexData::make_array();
      for (int i = 0; i < kH3N; ++i) {
        FlexData e = FlexData::make_object();
        auto eo = e.as_object();
        eo.insert_or_assign("file",
                            FlexData::make_string("s" + std::to_string(i)));
        eo.insert_or_assign("repeats", FlexData::make_int(1));
        FlexData caps = FlexData::make_array();
        caps.as_array().push_back(FlexData::make_string("c"));
        eo.insert_or_assign("captions", std::move(caps));
        FlexData pics = FlexData::make_array();
        FlexData po = FlexData::make_object();
        po.as_object().insert_or_assign("w", FlexData::make_int(kH3G * 16));
        po.as_object().insert_or_assign("h", FlexData::make_int(kH3G * 16));
        if (clip_frames > 0) {
          po.as_object().insert_or_assign("frames",
                                          FlexData::make_int(clip_frames));
          po.as_object().insert_or_assign("start", FlexData::make_int(0));
          po.as_object().insert_or_assign("audio", FlexData::make_bool(audio));
        }
        if (anchors != 0) {
          FlexData an = FlexData::make_array();
          if (anchors & 1) {
            an.as_array().push_back(FlexData::make_string("first"));
          }
          if (anchors & 2) {
            an.as_array().push_back(FlexData::make_string("last"));
          }
          po.as_object().insert_or_assign("anchors", std::move(an));
          FlexData pc = FlexData::make_array();
          pc.as_array().push_back(FlexData::make_string("c, shown"));
          po.as_object().insert_or_assign("captions", std::move(pc));
        }
        pics.as_array().push_back(std::move(po));
        eo.insert_or_assign("pictures", std::move(pics));
        arr.as_array().push_back(std::move(e));
      }
      o.insert_or_assign("samples", std::move(arr));
      FlexData pv = FlexData::make_array();
      pv.as_array().push_back(FlexData::make_string("ohwx"));
      o.insert_or_assign("preview_prompts", std::move(pv));
    }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto cond = [&]() {
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::Bf16;
      tb->shape = {kH3Txt, kH3D};
      tb->resize_contiguous((std::size_t)kH3Txt * kH3D);
      auto* d = reinterpret_cast<std::uint16_t*>(tb->data.data());
      for (int i = 0; i < kH3Txt * kH3D; ++i) {
        float f = 0.6f * nd(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        d[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
      }
      return tb;
    };
    // A still is one latent frame; a clip of 17n + 5 frames 5n + 2.
    const int lt = clip_frames > 0 ? 5 * ((clip_frames - 5) / 17) + 2 : 1;
    const int alat = (int)std::llround(clip_frames / 24.0 * 40.0);
    auto latent = [&]() {
      // As vae-encode's H3 branch emits it.
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {kH3Z, lt, kH3G, kH3G};
      const std::size_t n = (std::size_t)kH3Z * lt * kH3G * kH3G;
      tb->resize_contiguous(n);
      for (std::size_t i = 0; i < n; ++i) { tb->as_f32()[i] = nd(rng); }
      return tb;
    };
    auto soundtrack = [&]() {
      // As audio-vae-encode's H3 encoder emits it: [2, 32, n], one latent
      // short of the layout's count (the codec's hop rounds down).
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {2, 32, alat - 1};
      const std::size_t n = (std::size_t)2 * 32 * (alat - 1);
      tb->resize_contiguous(n);
      for (std::size_t i = 0; i < n; ++i) { tb->as_f32()[i] = nd(rng); }
      return tb;
    };
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto src = [&](const char* id) {
      auto u = std::make_unique<Beats>(&sess, id, std::vector<InEdge>{},
                                       FlexData::make_object());
      u->allocate_oports(1);
      return static_cast<Beats*>(pl->insert_stage(std::move(u)));
    };
    Beats* conds = src("conds");
    Beats* lats = src("lats");
    Beats* mans = src("man");
    Beats* auds = audio ? src("auds") : nullptr;
    Beats* rvs = refs ? src("rvs") : nullptr;
    Beats* ras = refs ? src("ras") : nullptr;
    auto rows = [&](int n, int w) {
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {n, n > 0 ? w : 1};
      tb->resize_contiguous((std::size_t)n * w);
      for (int i = 0; i < n * w; ++i) { tb->as_f32()[i] = nd(rng); }
      return tb;
    };
    auto with_refs = [&](std::unique_ptr<TensorBeatPayload> c) {
      FlexData sb = FlexData::make_object();
      auto o = sb.as_object();
      FlexData tags = FlexData::make_array();
      for (int r = 0; r < kH3Txt; ++r) {
        tags.as_array().push_back(FlexData::make_int(r >= 1 && r < 4 ? 0 : 1));
      }
      o.insert_or_assign("token_tags", std::move(tags));
      FlexData rl = FlexData::make_array();
      FlexData im = FlexData::make_object();
      im.as_object().insert_or_assign("kind", FlexData::make_string("image"));
      im.as_object().insert_or_assign("latent_frames", FlexData::make_int(1));
      im.as_object().insert_or_assign("latent_height", FlexData::make_int(4));
      im.as_object().insert_or_assign("latent_width", FlexData::make_int(6));
      rl.as_array().push_back(std::move(im));
      FlexData au = FlexData::make_object();
      au.as_object().insert_or_assign("kind", FlexData::make_string("audio"));
      au.as_object().insert_or_assign("audio_latents", FlexData::make_int(7));
      rl.as_array().push_back(std::move(au));
      o.insert_or_assign("references", std::move(rl));
      c->sideband = std::move(sb);
      rvs->beats.push_back(rows(2 * 3, 96));
      ras->beats.push_back(rows(2 * 7, 32));
      return c;
    };
    auto no_refs = [&]() {
      if (rvs == nullptr) { return; }
      rvs->beats.push_back(rows(0, 96));
      ras->beats.push_back(rows(0, 32));
    };
    auto still = [&]() {
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {kH3Z, 1, kH3G, kH3G};
      const std::size_t n = (std::size_t)kH3Z * kH3G * kH3G;
      tb->resize_contiguous(n);
      for (std::size_t i = 0; i < n; ++i) { tb->as_f32()[i] = nd(rng); }
      return tb;
    };
    for (int i = 0; i < kH3N; ++i) {
      conds->beats.push_back(refs ? with_refs(cond()) : cond());
      lats->beats.push_back(latent());
      for (int b = 0; b < 2; ++b) {
        if (anchors & (1 << b)) { lats->beats.push_back(still()); }
      }
      if (anchors != 0) {
        // The presented caption: its first 4 rows the vision block.
        auto c = cond();
        FlexData tags = FlexData::make_array();
        for (int r = 0; r < kH3Txt; ++r) {
          tags.as_array().push_back(FlexData::make_int(r < 4 ? 0 : 1));
        }
        FlexData sb = FlexData::make_object();
        sb.as_object().insert_or_assign("token_tags", std::move(tags));
        c->sideband = std::move(sb);
        conds->beats.push_back(std::move(c));
      }
      if (auds != nullptr) { auds->beats.push_back(soundtrack()); }
    }
    {
      // The empty prompt, as the conditioner encodes it for H3: no rows.
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::Bf16;
      tb->shape = {0, kH3D};
      conds->beats.push_back(std::move(tb));
      no_refs();
    }
    conds->beats.push_back(cond());   // the preview prompt
    no_refs();
    mans->beats.push_back(make_payload<FlexDataPayload>(man));
    FlexData ocfg = FlexData::make_object();
    ocfg.as_object().insert_or_assign("learning_rate",
                                      FlexData::make_real(1e-2));
    ocfg.as_object().insert_or_assign("warmup_steps", FlexData::make_int(0));
    auto opt = pl->insert_stage(std::make_unique<OptimizerSelectStage>(
        &sess, "opt", std::vector<InEdge>{}, ocfg));
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert_or_assign("hf_dir", FlexData::make_string(model.string()));
      o.insert_or_assign("output_dir", FlexData::make_string(out.string()));
      o.insert_or_assign("name", FlexData::make_string("ut-h3"));
      o.insert_or_assign("rank", FlexData::make_int(4));
      o.insert_or_assign("steps", FlexData::make_int(steps));
      o.insert_or_assign("save_every", FlexData::make_int(save_every));
      o.insert_or_assign("preview_every", FlexData::make_int(preview_every));
      o.insert_or_assign("preview_steps", FlexData::make_int(2));
      o.insert_or_assign("caption_dropout", FlexData::make_real(0.2));
      o.insert_or_assign("register", FlexData::make_bool(false));
      o.insert_or_assign("resume", FlexData::make_string(resume));
      if (!recompute.empty()) {
        o.insert_or_assign("recompute", FlexData::make_string(recompute));
      }
    }
    const InEdge none{nullptr, 0};
    std::vector<InEdge> in = {{conds, 0}, {lats, 0}, {mans, 0}, none,
                              {opt, 0}};
    if (auds != nullptr || rvs != nullptr) {
      in.push_back(auds != nullptr ? InEdge{auds, 0} : none);
    }
    if (rvs != nullptr) {
      in.push_back({rvs, 0});
      in.push_back({ras, 0});
    }
    auto tr_u = std::make_unique<TrainLoraStage>(&sess, "train", in, cfg);
    config_error = tr_u->config_error();
    if (!config_error.empty()) { return; }
    auto* tr = pl->insert_stage(std::move(tr_u));
    CbSink* sinks[4];
    for (int p = 0; p < 4; ++p) {
      auto s = std::make_unique<CbSink>(
          &sess, "tsink" + std::to_string(p),
          std::vector<InEdge>{{tr, (unsigned)p}}, FlexData::make_object());
      sinks[p] = static_cast<CbSink*>(pl->insert_stage(std::move(s)));
    }
    PipelineRuntime rt(pl.get(), &sess);
    sinks[1]->on = [&](const BeatPayloadIntf* b) {
      if (planned == 0) {
        if (auto* mgr = sess.services()->generative_model_manager()) {
          planned = mgr->planned_weight_bytes(model.string());
        }
      }
      if (on_ckpt) { on_ckpt(&rt, b); }
    };
    if (!rt.launch()) { config_error = "launch failed"; return; }
    rt.wait_idle();
    rt.stop();
    previews = std::move(sinks[0]->got);
    ckpts = std::move(sinks[1]->got);
    metrics = std::move(sinks[2]->got);
    preview_audio = std::move(sinks[3]->got);
  }
};

FlexData
flex_of_(const BeatPayloadIntf* b)
{
  const auto* f = dynamic_cast<const FlexDataPayload*>(b);
  return f != nullptr ? f->data : FlexData();
}

std::vector<char>
bytes_of_(const fs::path& p)
{
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
}

// The engine's GEMM and attention tunes time candidates, and two loads
// may vote differently: off, so two runs are the same arithmetic.
void
pin_h3_tuners_()
{
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  ::setenv("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE", "1", 1);
}

}  // namespace

// A still dataset trains an H3 adapter end to end: the stage picks the H3
// family off the checkpoint, takes the empty prompt as the conditioner now
// encodes it for H3 ([0, 256]: no rows) with caption dropout off (H3 is
// guidance-distilled), packs [24, 1, h, w] latents as one-frame
// clips, trains on the video's loss (the audio rows laid out as inference
// does, out of the loss), checkpoints and previews ([24, 1, h, w] clip
// latents for vae-decode), and writes an adapter in the diffusers
// spelling that the model's own binder takes on every module.
TEST(train_lora, minimax_h3_trains_on_stills_and_previews)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3";
  fs::remove_all(root);
  fs::create_directories(root);
  H3Run r;
  r.model = root / "h3-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(r.model));
  r.out = root / "out";
  r.steps = 30;
  r.save_every = 15;
  r.run();
  ASSERT_TRUE(r.config_error.empty());
  ASSERT_TRUE(r.metrics.size() == 30);
  if (r.metrics.size() != 30) { return; }
  double first = 0.0, last = 0.0;
  bool parts = true;
  for (std::size_t i = 0; i < r.metrics.size(); ++i) {
    FlexData m = flex_of_(r.metrics[i].get());
    auto o = m.as_object();
    const double l = o.at("loss").as_real(0.0);
    parts = parts && o.contains("loss_video") && !o.contains("loss_audio") &&
            std::fabs(o.at("loss_video").as_real(-1.0) - l) < 1e-6 * (1 + l);
    if (i < 5) { first += l; }
    if (i >= 25) { last += l; }
  }
  std::printf("[train_lora] minimax-h3 loss %.4f -> %.4f (5-step means); "
              "the loss is the video's alone: %s\n", first / 5, last / 5,
              parts ? "yes" : "NO");
  EXPECT_TRUE(last < 0.9 * first);
  // No soundtrack: the loss is the video's mean, and no audio loss is
  // reported for it.
  EXPECT_TRUE(parts);
  EXPECT_TRUE(fs::exists(r.out / "ut-h3-000015.safetensors"));
  EXPECT_TRUE(fs::exists(r.out / "ut-h3.safetensors"));
  EXPECT_TRUE(r.ckpts.size() == 2);
  ASSERT_TRUE(r.previews.size() == 2);
  for (const auto& p : r.previews) {
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    ASSERT_TRUE(tb != nullptr);
    if (tb == nullptr) { continue; }
    EXPECT_TRUE(tb->shape.size() == 4 && tb->shape[0] == kH3Z &&
                tb->shape[1] == 1 && tb->shape[2] == kH3G &&
                tb->shape[3] == kH3G);
    bool finite = true;
    for (std::size_t i = 0; i < tb->element_count(); ++i) {
      finite = finite && std::isfinite(tb->as_f32()[i]);
    }
    EXPECT_TRUE(finite);
  }
  // The adapter binds on the model by its own binder: every projection of
  // the two blocks and the refiner block -- q|k|v (one fused projection,
  // its three factors banded), out, fc1 and fc2.
  {
    Session sess;
    genai::MetalMiniMaxH3Transformer::Config cfg;
    std::string err;
    ASSERT_TRUE(genai::MetalMiniMaxH3Transformer::config_from_json(
        r.model.string(), cfg, &err));
    genai::MetalMiniMaxH3Transformer::LoraSpec sp;
    sp.path = (r.out / "ut-h3.safetensors").string();
    auto m = genai::MetalMiniMaxH3Transformer::load(
        r.model.string(), sess.metal_compute(), cfg, false, {sp});
    ASSERT_TRUE(m != nullptr);
    if (m != nullptr) {
      std::printf("[train_lora] minimax-h3 adapter: %d modules bound, %d "
                  "banded\n", m->lora_modules(0), m->lora_banded(0));
      EXPECT_TRUE(m->lora_modules(0) == 3 * 4);
      EXPECT_TRUE(m->lora_banded(0) == 3);
    }
  }
  // Its metadata names the family and the AdaLN it was trained against.
  {
    FlexData meta;
    ASSERT_TRUE(genai::comfy::read_metadata(
        (r.out / "ut-h3.safetensors").string(), meta));
    const std::string fam =
        meta.is_object() && meta.as_object().contains("vpipe.family")
            ? std::string(meta.as_object().at("vpipe.family").as_string(""))
            : std::string();
    const std::string ad =
        meta.is_object() && meta.as_object().contains("vpipe.adaln")
            ? std::string(meta.as_object().at("vpipe.adaln").as_string(""))
            : std::string();
    std::printf("[train_lora] minimax-h3 adapter: family '%s', adaln '%s'\n",
                fam.c_str(), ad.c_str());
    EXPECT_TRUE(fam == "minimax-h3");
    EXPECT_TRUE(!ad.empty());
  }
  fs::remove_all(root);
}

// A run STOPPED part way and resumed is the run that was not stopped: the
// same adapter, byte for byte. Sample order, sigma, noise and the missing
// soundtrack's noise are all functions of (seed, sample number), and the
// resume state carries the optimizer's whole state.
TEST(train_lora, minimax_h3_a_stopped_run_resumes_the_same_adapter)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3-res";
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path model = root / "h3-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(model));
  const int kSteps = 40;
  H3Run a;
  a.model = model;
  a.out = root / "a";
  a.steps = kSteps;
  a.save_every = 8;
  a.preview_every = -1;
  a.run();
  ASSERT_TRUE(a.config_error.empty());
  // B: stopped once its first checkpoint is out, then run again to the
  // end with resume on.
  H3Run b;
  b.model = model;
  b.out = root / "b";
  b.steps = kSteps;
  b.save_every = 8;
  b.preview_every = -1;
  b.resume = "auto";
  b.on_ckpt = [](PipelineRuntime* rt, const BeatPayloadIntf*) {
    rt->pause();
  };
  b.run();
  ASSERT_TRUE(b.config_error.empty());
  const bool stopped = fs::exists(b.out / "ut-h3.resume" / "run.json") &&
                       !fs::exists(b.out / "ut-h3.safetensors");
  int at = -1;
  if (stopped) {
    std::ifstream in(b.out / "ut-h3.resume" / "run.json");
    FlexData run = FlexData::from_json(in);
    at = (int)run.as_object().at("step").as_int(-1);
  }
  std::printf("[train_lora] minimax-h3: B stopped at step %d of %d (%zu "
              "metrics)\n", at, kSteps, b.metrics.size());
  ASSERT_TRUE(stopped && at > 0 && at < kSteps);
  H3Run c;
  c.model = model;
  c.out = b.out;
  c.steps = kSteps;
  c.save_every = 8;
  c.preview_every = -1;
  c.resume = "auto";
  c.run();
  ASSERT_TRUE(c.config_error.empty());
  std::printf("[train_lora] minimax-h3: resumed run took %zu steps\n",
              c.metrics.size());
  EXPECT_TRUE((int)c.metrics.size() == kSteps - at);
  const std::vector<char> fa = bytes_of_(a.out / "ut-h3.safetensors");
  const std::vector<char> fc = bytes_of_(c.out / "ut-h3.safetensors");
  std::printf("[train_lora] minimax-h3: uninterrupted %zu bytes, resumed "
              "%zu bytes, %s\n", fa.size(), fc.size(),
              fa == fc ? "IDENTICAL" : "DIFFERENT");
  EXPECT_TRUE(!fa.empty() && fa == fc);
  fs::remove_all(root);
}

// THE PLAN FOLLOWS THE AdaLN. Pruned at load (VPIPE_H3_TRAIN_ADALN=prune,
// what a box too small for the released DiT gets), the model holds its
// AdaLN as [D, 8] -- and once loaded the plan counts what it holds, not
// the checkpoint's size the launch claimed. Kept, the two agree.
TEST(train_lora, minimax_h3_the_plan_counts_the_adaln_it_holds)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3-pl";
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path model = root / "h3-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(model));
  std::size_t planned[2] = {0, 0};
  for (int pr = 0; pr < 2; ++pr) {
    ::setenv("VPIPE_H3_TRAIN_ADALN", pr ? "prune" : "keep", 1);
    H3Run r;
    r.model = model;
    r.out = root / (pr ? "pruned" : "kept");
    r.steps = 4;
    r.save_every = 2;
    r.preview_every = -1;
    r.run();
    ASSERT_TRUE(r.config_error.empty());
    planned[pr] = r.planned;
  }
  ::unsetenv("VPIPE_H3_TRAIN_ADALN");
  genai::MetalMiniMaxH3Transformer::Config cfg;
  ASSERT_TRUE(genai::MetalMiniMaxH3Transformer::config_from_json(
      model.string(), cfg));
  const std::size_t adaln =
      genai::MetalMiniMaxH3Transformer::adaln_retired_bytes(model.string());
  std::printf("[train_lora] minimax-h3: the plan counts %.2f MB kept, %.2f "
              "MB pruned at load (the checkpoint's AdaLN %.2f MB)\n",
              planned[0] / 1048576.0, planned[1] / 1048576.0,
              adaln / 1048576.0);
  EXPECT_TRUE(planned[0] > 0 && planned[1] > 0);
  EXPECT_TRUE(planned[0] >= planned[1] + adaln * 3 / 4);
  fs::remove_all(root);
}

// CLIPS WITH THEIR SOUNDTRACKS. Each sample a 22-frame window -- its
// latent [24, 7, h, w] -- and its soundtrack latent on the audio_latents
// port, one latent short of the 37 the layout counts (the codec's hop),
// which the stage holds from its last. The loss is then the two
// modalities' means and both are reported; the previews are clips
// [24, 7, h, w] with their soundtracks [2, 32, 37] on preview_audio.
TEST(train_lora, minimax_h3_trains_on_clips_with_their_soundtracks)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3-av";
  fs::remove_all(root);
  fs::create_directories(root);
  H3Run r;
  r.model = root / "h3-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(r.model));
  r.out = root / "out";
  r.steps = 30;
  r.save_every = 15;
  r.clip_frames = 22;
  r.audio = true;
  r.run();
  ASSERT_TRUE(r.config_error.empty());
  ASSERT_TRUE(r.metrics.size() == 30);
  if (r.metrics.size() != 30) { return; }
  double first = 0.0, last = 0.0, fa = 0.0, la = 0.0;
  bool both = true;
  for (std::size_t i = 0; i < r.metrics.size(); ++i) {
    FlexData m = flex_of_(r.metrics[i].get());
    auto o = m.as_object();
    both = both && o.contains("loss_video") && o.contains("loss_audio");
    const double l = o.at("loss").as_real(0.0);
    const double a = o.contains("loss_audio")
        ? o.at("loss_audio").as_real(0.0) : 0.0;
    // loss = video + audio at weight 1, each its own mean.
    if (o.contains("loss_video")) {
      both = both && std::fabs(o.at("loss_video").as_real(0.0) + a - l) <
                         1e-5 * (1 + l);
    }
    if (i < 5) { first += l; fa += a; }
    if (i >= 25) { last += l; la += a; }
  }
  std::printf("[train_lora] minimax-h3 clips: loss %.4f -> %.4f, its audio "
              "%.4f -> %.4f (5-step means); video + audio = loss: %s\n",
              first / 5, last / 5, fa / 5, la / 5, both ? "yes" : "NO");
  EXPECT_TRUE(both);
  EXPECT_TRUE(last < 0.9 * first);
  EXPECT_TRUE(la < fa);
  ASSERT_TRUE(r.previews.size() == 2 && r.preview_audio.size() == 2);
  for (std::size_t k = 0; k < r.previews.size(); ++k) {
    const auto* v = dynamic_cast<const TensorBeatPayload*>(r.previews[k].get());
    const auto* a =
        dynamic_cast<const TensorBeatPayload*>(r.preview_audio[k].get());
    ASSERT_TRUE(v != nullptr && a != nullptr);
    if (v == nullptr || a == nullptr) { continue; }
    std::printf("[train_lora] minimax-h3 clip preview: [%lld, %lld, %lld, "
                "%lld] + soundtrack [%lld, %lld, %lld]\n",
                (long long)v->shape[0], (long long)v->shape[1],
                (long long)v->shape[2], (long long)v->shape[3],
                (long long)a->shape[0], (long long)a->shape[1],
                (long long)a->shape[2]);
    EXPECT_TRUE(v->shape.size() == 4 && v->shape[1] == 7);
    EXPECT_TRUE(a->shape.size() == 3 && a->shape[0] == 2 &&
                a->shape[1] == 32 && a->shape[2] == 37);
  }
  fs::remove_all(root);
}

// ---- training-dataset: clips ------------------------------------------------

// The clip rules, file-free: a clip takes the longest window bucket it
// fills; its windows spread from the first frame to the last window's;
// its frames at the training rate are what ffmpeg's fps filter makes of
// the container's; and every window keys apart -- frames, start, rate --
// with its soundtrack keyed apart again, while a STILL's key is exactly
// the one it always had (a cache filled before clips existed stays good).
TEST(training_dataset, clip_windows_buckets_and_keys)
{
  using S = TrainingDatasetStage;
  EXPECT_TRUE(S::clip_bucket(50, {22, 39, 73}) == 39);
  EXPECT_TRUE(S::clip_bucket(73, {22, 39, 73}) == 73);
  EXPECT_TRUE(S::clip_bucket(21, {22, 39}) == 0);
  EXPECT_TRUE((S::window_starts(100, 39, 3) == std::vector<int>{0, 31, 61}));
  EXPECT_TRUE((S::window_starts(39, 39, 3) == std::vector<int>{0}));
  EXPECT_TRUE((S::window_starts(100, 39, 1) == std::vector<int>{0}));
  EXPECT_TRUE(S::window_starts(20, 39, 1).empty());
  EXPECT_TRUE(S::frames_at_fps(48, 24.0, 2.0, 24.0) == 48);   // as it is
  EXPECT_TRUE(S::frames_at_fps(60, 30.0, 2.0, 24.0) == 48);   // drops
  EXPECT_TRUE(S::frames_at_fps(0, 0.0, 2.0, 24.0) == 47);     // no count

  std::vector<S::Sample> v(2);
  v[0].rel = "a.mp4"; v[0].size = 10; v[0].mtime = 5;
  v[1].rel = "b.png"; v[1].size = 10; v[1].mtime = 5;
  for (int st : {0, 31}) {
    S::Picture p;
    p.w = 256; p.h = 160; p.frames = 39; p.start = st; p.audio = true;
    v[0].pictures.push_back(p);
  }
  S::Picture still;
  still.w = 256; still.h = 160;
  v[1].pictures.push_back(still);
  S::key_samples(&v, false, {}, 24.0, 32000);
  const auto& a = v[0].pictures;
  EXPECT_TRUE(!a[0].key.empty() && a[0].key != a[1].key);
  EXPECT_TRUE(!a[0].audio_key.empty() && a[0].audio_key != a[1].audio_key);
  EXPECT_TRUE(a[0].audio_key != a[0].key);
  EXPECT_TRUE(v[1].pictures[0].audio_key.empty());
  EXPECT_TRUE(v[1].pictures[0].key ==
              genai::train::cache_key({"latent", "b.png", "10", "5", "256",
                                       "160", "", "rgb"}));
  // Another rate is another window.
  std::vector<S::Sample> w = v;
  S::key_samples(&w, false, {}, 30.0, 32000);
  EXPECT_TRUE(w[0].pictures[0].key != a[0].key);
  EXPECT_TRUE(w[1].pictures[0].key == v[1].pictures[0].key);
}

// A REAL CLIP through the dataset: VPIPE_TRAIN_CLIP=<a video file with a
// soundtrack, 2 s or more>. Two 22-frame windows at 24 fps, fitted to a
// 256^2-area bucket: the images port carries each as [22, 3, H, W] U8,
// the audio port its own 22 / 24 s of the soundtrack at 32 kHz stereo,
// and the manifest says so; the two windows are different frames.
TEST(training_dataset, a_clip_yields_windows_and_their_soundtracks)
{
  const char* clip = std::getenv("VPIPE_TRAIN_CLIP");
  if (clip == nullptr || *clip == '\0') { return; }
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-dataset-clip";
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path cp = root / ("c" + fs::path(clip).extension().string());
  fs::copy_file(clip, cp);
  std::ofstream(root / "c.txt") << "a clip\n";
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("dir", FlexData::make_string(root.string()));
    o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
    o.insert_or_assign("resolution", FlexData::make_int(256));
    o.insert_or_assign("frames", FlexData::make_int(22));
    o.insert_or_assign("windows", FlexData::make_int(2));
  }
  auto ds_u = std::make_unique<TrainingDatasetStage>(
      &sess, "ds", std::vector<InEdge>{}, cfg);
  ASSERT_TRUE(ds_u->config_error().empty());
  auto* ds = pl->insert_stage(std::move(ds_u));
  Sink* sinks[4];
  for (int p = 0; p < 4; ++p) {
    auto s = std::make_unique<Sink>(&sess, "sink" + std::to_string(p),
                                    std::vector<InEdge>{{ds, (unsigned)p}},
                                    FlexData::make_object());
    sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(s)));
  }
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  ASSERT_TRUE(sinks[2]->got.size() == 1);
  FlexData man = dynamic_cast<const FlexDataPayload*>(
                     sinks[2]->got[0].get())->data;
  const FlexData pics =
      man.as_object().at("samples").as_array().at(0).as_object().at(
          "pictures");
  ASSERT_TRUE(pics.as_array().size() == 2);
  const int bw = (int)pics.as_array().at(0).as_object().at("w").as_int(0);
  const int bh = (int)pics.as_array().at(0).as_object().at("h").as_int(0);
  const int st1 =
      (int)pics.as_array().at(1).as_object().at("start").as_int(0);
  std::printf("[training_dataset] clip: windows at 0 and %d, bucket %dx%d, "
              "%zu images, %zu soundtracks\n", st1, bw, bh,
              sinks[0]->got.size(), sinks[3]->got.size());
  EXPECT_TRUE(st1 > 0);
  ASSERT_TRUE(sinks[0]->got.size() == 2 && sinks[3]->got.size() == 2);
  const std::int64_t n = std::llround(22 / 24.0 * 32000);
  std::vector<std::vector<std::uint8_t>> frames;
  for (int k = 0; k < 2; ++k) {
    const auto* v =
        dynamic_cast<const TensorBeatPayload*>(sinks[0]->got[k].get());
    const auto* a =
        dynamic_cast<const TensorBeatPayload*>(sinks[3]->got[k].get());
    ASSERT_TRUE(v != nullptr && a != nullptr);
    if (v == nullptr || a == nullptr) { return; }
    EXPECT_TRUE(v->dtype == TensorBeat::DType::U8 && v->shape.size() == 4 &&
                v->shape[0] == 22 && v->shape[1] == 3 &&
                v->shape[2] == bh && v->shape[3] == bw);
    EXPECT_TRUE(a->dtype == TensorBeat::DType::F32 && a->shape.size() == 2 &&
                a->shape[0] == 2 && a->shape[1] == n);
    FlexData sb = a->sideband;
    EXPECT_TRUE(sb.as_object().at("sample_rate").as_int(0) == 32000 &&
                sb.as_object().at(beat::kBatch).as_bool(false));
    double rms = 0.0;
    for (std::size_t i = 0; i < a->element_count(); ++i) {
      rms += (double)a->as_f32()[i] * a->as_f32()[i];
    }
    // FNV-1a of the frames and the samples: the same on every machine
    // whose FFmpeg decodes the clip the same.
    std::uint64_t hv = 1469598103934665603ull, ha = hv;
    for (std::size_t i = 0; i < v->element_count(); ++i) {
      hv = (hv ^ v->as_u8()[i]) * 1099511628211ull;
    }
    const auto* ab = reinterpret_cast<const std::uint8_t*>(a->as_f32());
    for (std::size_t i = 0; i < a->element_count() * 4; ++i) {
      ha = (ha ^ ab[i]) * 1099511628211ull;
    }
    std::printf("[training_dataset] window %d: [%lld, 3, %lld, %lld] "
                "%016llx, soundtrack [2, %lld] rms %.4f %016llx\n", k,
                (long long)v->shape[0], (long long)v->shape[2],
                (long long)v->shape[3], (unsigned long long)hv,
                (long long)a->shape[1],
                std::sqrt(rms / std::max<std::size_t>(a->element_count(), 1)),
                (unsigned long long)ha);
    frames.emplace_back(v->as_u8(), v->as_u8() + v->element_count());
  }
  EXPECT_TRUE(frames[0] != frames[1]);
  fs::remove_all(root);
}

// KEYFRAME TASKS from a real clip (VPIPE_TRAIN_CLIP): task_mix fl2va, the
// keyframes port wired. Each window is followed on the images port by
// its two anchors -- stills [3, H, W], exactly its first and last frames
// -- and its caption is encoded again, presented with them: the captions
// port carries the plain caption, one presented caption a window, the
// empty prompt and the preview, and the keyframes port a list beside
// each -- empty, the window's two frames, the other's, empty, empty.
TEST(training_dataset, a_keyframe_task_anchors_and_presents_its_frames)
{
  const char* clip = std::getenv("VPIPE_TRAIN_CLIP");
  if (clip == nullptr || *clip == '\0') { return; }
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-dataset-kf";
  fs::remove_all(root);
  fs::create_directories(root);
  fs::copy_file(clip, root / ("c" + fs::path(clip).extension().string()));
  std::ofstream(root / "c.txt") << "a clip\n";
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("dir", FlexData::make_string(root.string()));
    o.insert_or_assign("trigger_word", FlexData::make_string("ohwx"));
    o.insert_or_assign("resolution", FlexData::make_int(256));
    o.insert_or_assign("frames", FlexData::make_int(22));
    o.insert_or_assign("windows", FlexData::make_int(2));
    o.insert_or_assign("audio", FlexData::make_string("none"));
    FlexData tm = FlexData::make_object();
    tm.as_object().insert_or_assign("fl2va", FlexData::make_real(1.0));
    o.insert_or_assign("task_mix", std::move(tm));
  }
  auto ds_u = std::make_unique<TrainingDatasetStage>(
      &sess, "ds", std::vector<InEdge>{}, cfg);
  ASSERT_TRUE(ds_u->config_error().empty());
  auto* ds = pl->insert_stage(std::move(ds_u));
  Sink* sinks[5];
  for (int p = 0; p < 5; ++p) {
    auto s = std::make_unique<Sink>(&sess, "sink" + std::to_string(p),
                                    std::vector<InEdge>{{ds, (unsigned)p}},
                                    FlexData::make_object());
    sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(s)));
  }
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  ASSERT_TRUE(sinks[2]->got.size() == 1);
  FlexData man = dynamic_cast<const FlexDataPayload*>(
                     sinks[2]->got[0].get())->data;
  const FlexData pics =
      man.as_object().at("samples").as_array().at(0).as_object().at(
          "pictures");
  ASSERT_TRUE(pics.as_array().size() == 2);
  const FlexData an = pics.as_array().at(0).as_object().at("anchors");
  EXPECT_TRUE(an.as_array().size() == 2);
  EXPECT_TRUE(pics.as_array().at(0).as_object().contains("captions"));
  // Images: window, first, last; window, first, last.
  ASSERT_TRUE(sinks[0]->got.size() == 6);
  bool anchors_are_frames = true;
  for (int w = 0; w < 2; ++w) {
    const auto* v = dynamic_cast<const TensorBeatPayload*>(
        sinks[0]->got[(std::size_t)w * 3].get());
    ASSERT_TRUE(v != nullptr && v->shape.size() == 4);
    if (v == nullptr) { return; }
    const std::size_t fsz = (std::size_t)3 * v->shape[2] * v->shape[3];
    for (int k = 0; k < 2; ++k) {
      const auto* a = dynamic_cast<const TensorBeatPayload*>(
          sinks[0]->got[(std::size_t)w * 3 + 1 + k].get());
      ASSERT_TRUE(a != nullptr && a->shape.size() == 3 &&
                  (std::size_t)a->element_count() == fsz);
      if (a == nullptr) { return; }
      const std::uint8_t* fr =
          v->as_u8() + (std::size_t)(k == 0 ? 0 : v->shape[0] - 1) * fsz;
      anchors_are_frames = anchors_are_frames &&
                           std::memcmp(a->as_u8(), fr, fsz) == 0;
    }
  }
  EXPECT_TRUE(anchors_are_frames);
  // Captions and the lists beside them.
  ASSERT_TRUE(sinks[1]->got.size() == 5 && sinks[4]->got.size() == 5);
  std::vector<std::size_t> sizes;
  for (const auto& b : sinks[4]->got) {
    const auto* l = dynamic_cast<const TensorListPayload*>(b.get());
    sizes.push_back(l != nullptr ? l->items.size() : 99);
  }
  std::printf("[training_dataset] fl2va: %zu images, anchors are the "
              "window's frames: %s; caption lists %zu %zu %zu %zu %zu\n",
              sinks[0]->got.size(), anchors_are_frames ? "yes" : "NO",
              sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
  EXPECT_TRUE((sizes == std::vector<std::size_t>{0, 2, 2, 0, 0}));
  EXPECT_TRUE(text_of_(sinks[1]->got[1].get()) == "ohwx, a clip");
  fs::remove_all(root);
}

// FL2VA'S KEYFRAME TASKS through the stage: every window pins its first
// and last frames -- their stills after it on the latents port -- and
// trains on its caption PRESENTED with them (a conditioning of its own,
// vision rows tagged), while the sample's plain caption is never used for
// it. The streams pair, the loss falls, and the adapter saves.
TEST(train_lora, minimax_h3_trains_fl2va_keyframe_tasks)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3-kf";
  fs::remove_all(root);
  fs::create_directories(root);
  H3Run r;
  r.model = root / "h3-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(r.model));
  r.out = root / "out";
  r.steps = 30;
  r.save_every = 15;
  r.clip_frames = 22;
  r.anchors = 3;
  r.run();
  ASSERT_TRUE(r.config_error.empty());
  ASSERT_TRUE(r.metrics.size() == 30);
  if (r.metrics.size() != 30) { return; }
  double first = 0.0, last = 0.0;
  for (std::size_t i = 0; i < r.metrics.size(); ++i) {
    FlexData m = flex_of_(r.metrics[i].get());
    const double l = m.as_object().at("loss").as_real(0.0);
    if (i < 5) { first += l; }
    if (i >= 25) { last += l; }
  }
  std::printf("[train_lora] minimax-h3 fl2va: loss %.4f -> %.4f (5-step "
              "means)\n", first / 5, last / 5);
  EXPECT_TRUE(last < 0.9 * first);
  EXPECT_TRUE(fs::exists(r.out / "ut-h3.safetensors"));
  fs::remove_all(root);
}

// REF2VA end to end on the stage: every caption arrives encoded with a
// picture and a soundtrack as video-ref-encoder emits them (the plan on
// the conditioning's sideband, the rows on ports 6 and 7, empty for the
// empty and preview prompts); the synthetic checkpoint says Ref2VA in its
// name. The stage holds the rows with their captions, the plan books
// them, every step packs the reference layout, and the loss falls.
TEST(train_lora, minimax_h3_trains_ref2va_with_references)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3-r2v";
  fs::remove_all(root);
  fs::create_directories(root);
  H3Run r;
  r.model = root / "h3-ref2va-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(r.model));
  r.out = root / "out";
  r.steps = 30;
  r.save_every = 15;
  r.refs = true;
  r.run();
  ASSERT_TRUE(r.config_error.empty());
  ASSERT_TRUE(r.metrics.size() == 30);
  if (r.metrics.size() != 30) { return; }
  double first = 0.0, last = 0.0;
  for (std::size_t i = 0; i < r.metrics.size(); ++i) {
    FlexData m = flex_of_(r.metrics[i].get());
    const double l = m.as_object().at("loss").as_real(0.0);
    if (i < 5) { first += l; }
    if (i >= 25) { last += l; }
  }
  std::printf("[train_lora] minimax-h3 ref2va: loss %.4f -> %.4f (5-step "
              "means)\n", first / 5, last / 5);
  EXPECT_TRUE(last < 0.9 * first);
  FlexData md;
  ASSERT_TRUE(genai::comfy::read_metadata(
      (r.out / "ut-h3.safetensors").string(), md, nullptr));
  const std::string part(
      md.as_object().contains("vpipe.partition")
          ? md.as_object().at("vpipe.partition").as_string("")
          : "");
  std::printf("[train_lora] minimax-h3 ref2va: adapter partition '%s'\n",
              part.c_str());
  EXPECT_TRUE(part == "ref2va");
  fs::remove_all(root);
}

// REFERENCES (Ref2VA): a `<stem>.refs` sidecar names the files a sample is
// conditioned on (one a line, beside the sample or under the folder); a
// `references` folder holds them and is never scanned for samples. Each
// of the sample's captions goes out with its references and its duration
// (1 for a picture) -- video-ref-encoder's request -- keyed by both, so a
// changed reference re-encodes; a caption without references keeps its
// plain key, and a sidecar naming a missing file is refused by name.
TEST(training_dataset, a_samples_references_ride_its_captions)
{
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-ds-refs";
  fs::remove_all(root);
  fs::create_directories(root / "references");
  write_bmp_(root / "a.bmp", 64, 64, 1);
  std::ofstream(root / "a.txt") << "ohwx waves\n";
  std::ofstream(root / "a.refs") << "# the character sheet\n"
                                    "references/char.bmp\n";
  write_bmp_(root / "b.bmp", 64, 64, 2);
  std::ofstream(root / "b.txt") << "ohwx waves\n";
  write_bmp_(root / "references" / "char.bmp", 32, 32, 3);
  auto run = [&](std::vector<std::unique_ptr<BeatPayloadIntf>>* caps,
                 FlexData* man) {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("dir",
                                     FlexData::make_string(root.string()));
    cfg.as_object().insert_or_assign("resolution", FlexData::make_int(64));
    auto ds_u = std::make_unique<TrainingDatasetStage>(
        &sess, "ds", std::vector<InEdge>{}, cfg);
    if (!ds_u->config_error().empty()) { return false; }
    auto* ds = pl->insert_stage(std::move(ds_u));
    Sink* sinks[3];
    for (int p = 0; p < 3; ++p) {
      auto k = std::make_unique<Sink>(&sess, "sink" + std::to_string(p),
                                      std::vector<InEdge>{{ds, (unsigned)p}},
                                      FlexData::make_object());
      sinks[p] = static_cast<Sink*>(pl->insert_stage(std::move(k)));
    }
    PipelineRuntime rt(pl.get(), &sess);
    if (!rt.launch()) { return false; }
    rt.wait_idle();
    rt.stop();
    *caps = std::move(sinks[1]->got);
    if (sinks[2]->got.size() != 1) { return false; }
    *man = dynamic_cast<const FlexDataPayload*>(sinks[2]->got[0].get())->data;
    return true;
  };
  std::vector<std::unique_ptr<BeatPayloadIntf>> caps;
  FlexData man;
  ASSERT_TRUE(run(&caps, &man));
  // Two samples (the references folder is not one), two captions, the
  // empty prompt.
  auto sa = man.as_object().at("samples");
  ASSERT_TRUE(sa.as_array().size() == 2);
  ASSERT_TRUE(caps.size() == 3);
  if (caps.size() != 3) { return; }
  auto beat_of = [&](int i) {
    return dynamic_cast<const FlexDataPayload*>(caps[(std::size_t)i].get())
        ->data;
  };
  const FlexData a = beat_of(0), b = beat_of(1);
  const auto rq = VideoRefEncoderStage::read_beat_request(a);
  const auto rb = VideoRefEncoderStage::read_beat_request(b);
  std::printf("[training_dataset] a's caption: %zu reference(s), frames %d, "
              "batch %d; b's: %s\n", rq.references.size(), rq.frames,
              rq.batch ? 1 : 0, rb.has_references ? "references" : "none");
  ASSERT_TRUE(rq.error.empty() && rq.references.size() == 1);
  if (rq.references.size() != 1) { return; }
  EXPECT_TRUE(fs::path(rq.references[0]).filename() == "char.bmp" &&
              fs::path(rq.references[0]).is_absolute());
  EXPECT_TRUE(rq.frames == 1 && rq.batch);
  EXPECT_TRUE(!rb.has_references && rb.frames == 0);
  // Keyed apart (keys are made where a cache is): the same caption, with
  // and without references; and a changed reference is another request.
  EXPECT_TRUE(sa.as_array().at(0).as_object().contains("references"));
  auto keys = [&]() {
    Session ks;
    FlexData kc = FlexData::make_object();
    kc.as_object().insert_or_assign("dir",
                                    FlexData::make_string(root.string()));
    TrainingDatasetStage d(&ks, "ds", std::vector<InEdge>{}, kc);
    std::vector<TrainingDatasetStage::Sample> v;
    std::string e;
    if (!d.scan(&v, &e) || v.size() != 2) { return std::vector<std::string>{}; }
    TrainingDatasetStage::key_samples(&v, false, {});
    return std::vector<std::string>{v[0].caption_keys[0],
                                    v[1].caption_keys[0]};
  };
  const auto k1 = keys();
  ASSERT_TRUE(k1.size() == 2);
  if (k1.size() != 2) { return; }
  EXPECT_TRUE(k1[0] != k1[1]);
  write_bmp_(root / "references" / "char.bmp", 48, 32, 4);
  const auto k2 = keys();
  EXPECT_TRUE(k2.size() == 2 && k2[0] != k1[0] && k2[1] == k1[1]);
  // A missing reference is refused, naming it.
  std::ofstream(root / "b.refs") << "references/gone.bmp\n";
  std::vector<TrainingDatasetStage::Sample> v;
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("dir", FlexData::make_string(root.string()));
  TrainingDatasetStage ds(&sess, "ds", std::vector<InEdge>{}, cfg);
  std::string err;
  EXPECT_TRUE(!ds.scan(&v, &err) && err.find("gone.bmp") != std::string::npos);
  std::printf("[training_dataset] %s\n", err.c_str());
  // The beat request reader refuses a malformed field.
  FlexData bad = FlexData::make_object();
  bad.as_object().insert_or_assign("frames", FlexData::make_int(0));
  EXPECT_TRUE(!VideoRefEncoderStage::read_beat_request(bad).error.empty());
  fs::remove_all(root);
}

// RECOMPUTE: NEVER through the stage: the same run with every block's
// activations kept and with every block recomputed reports the same
// losses, step for step, and writes the same adapter -- the kept forward
// is the recomputed one to the bit.
TEST(train_lora, minimax_h3_keeps_every_block_where_asked)
{
  pin_h3_tuners_();
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-h3-keep";
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path model = root / "h3-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(model));
  std::vector<double> loss[2];
  std::vector<char> file[2];
  for (int k = 0; k < 2; ++k) {
    H3Run r;
    r.model = model;
    r.out = root / (k ? "never" : "always");
    r.steps = 10;
    r.save_every = 10;
    r.clip_frames = 22;
    r.recompute = k ? "never" : "always";
    r.run();
    ASSERT_TRUE(r.config_error.empty());
    for (const auto& b : r.metrics) {
      FlexData m = flex_of_(b.get());
      loss[k].push_back(m.as_object().at("loss").as_real(0.0));
    }
    file[k] = bytes_of_(r.out / "ut-h3.safetensors");
  }
  std::printf("[train_lora] minimax-h3 recompute always / never: %zu / %zu "
              "steps, first loss %.6f / %.6f, adapters %s\n", loss[0].size(),
              loss[1].size(), loss[0].empty() ? 0.0 : loss[0][0],
              loss[1].empty() ? 0.0 : loss[1][0],
              file[0] == file[1] ? "byte-identical" : "DIFFERENT");
  EXPECT_TRUE(loss[0].size() == 10 && loss[0] == loss[1]);
  EXPECT_TRUE(!file[0].empty() && file[0] == file[1]);
  fs::remove_all(root);
}

// WHAT A RUN HOLDS BESIDE ITS DiT is claimed at launch. A P-H3-Image run
// (a fake root naming FL2VA's DiT apart, `dit_dir`, and a frozen
// `base_lora` adapter): the launch claims the DiT at its streaming floor,
// the adapter at every factor the binder holds (lora_footprint's whole,
// its streamed set the floor) and the step's arena -- the model's own
// figure leaves the adapter out, so without its claim the revision after
// load would drop it from every ledger.
TEST(train_lora, claims_what_a_run_holds_beside_its_dit)
{
  const fs::path root = fs::temp_directory_path() / "vpipe-ut-train-claims";
  fs::remove_all(root);
  fs::create_directories(root / "ph3");
  const fs::path model = root / "h3-fl2va-synthetic.safetensors";
  ASSERT_TRUE(h3_synthetic::write_synthetic_(model));
  // A base adapter: a few modules' factors in diffusers' spelling.
  const fs::path base = root / "pruna.safetensors";
  {
    Session sess;
    genai::MetalMiniMaxH3Transformer::Config cfg;
    std::string err;
    ASSERT_TRUE(genai::MetalMiniMaxH3Transformer::config_from_json(
        model.string(), cfg, &err));
    auto m = genai::MetalMiniMaxH3Transformer::load(
        genai::WeightSet::open(model.string(), nullptr),
        sess.metal_compute(), cfg, false);
    ASSERT_TRUE(m != nullptr);
    auto t = genai::MiniMaxH3Trainer::create(m.get(), &err);
    ASSERT_TRUE(t != nullptr);
    genai::train::LoraParams p;
    ASSERT_TRUE(p.create(sess.metal_compute(), t->modules("attn_ff", true),
                         4, 4.0f, &err));
    p.init_random(3);
    ASSERT_TRUE(t->bind(&p, &err) > 0);
    ASSERT_TRUE(p.save(base.string(), {}, &err));
  }
  std::ofstream(root / "ph3" / "modular_model_index.json")
      << "{\"_class_name\": \"PH3ImageModularPipeline\"}";
  Session sess;
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir",
                       FlexData::make_string((root / "ph3").string()));
    o.insert_or_assign("dit_dir", FlexData::make_string(model.string()));
    o.insert_or_assign("base_lora", FlexData::make_string(base.string()));
    o.insert_or_assign("output_dir",
                       FlexData::make_string((root / "out").string()));
    o.insert_or_assign("name", FlexData::make_string("ut"));
  }
  TrainLoraStage st(&sess, "train", std::vector<InEdge>{}, cfg);
  ASSERT_TRUE(st.config_error().empty());
  const auto claims = st.declare_resources();
  const auto fp = genai::MetalMiniMaxH3Transformer::lora_footprint(
      base.string(), 2);
  bool dit = false, adapter = false, arena = false;
  std::int64_t held = -1;
  for (const ResourceClaim& c : claims) {
    if (c.kind == model_memory::kWeightsKind && c.key == model.string()) {
      dit = true;
    }
    if (c.kind == model_memory::kWeightsKind && c.key == base.string()) {
      adapter = true;
      held = bag::integer(c.extra, model_memory::kClaimHeldBytes, -1);
    }
    if (c.kind == model_memory::kScratchKind &&
        c.key.find("train-lora-arena") != std::string::npos) {
      arena = true;
    }
  }
  std::printf("[train_lora] claims: DiT %s, base adapter %s (held %lld of "
              "the footprint's %zu, streamed %zu), arena %s; %zu claims\n",
              dit ? "yes" : "NO", adapter ? "yes" : "NO", (long long)held,
              fp.whole, fp.streamed, arena ? "yes" : "NO", claims.size());
  EXPECT_TRUE(dit && adapter && arena);
  EXPECT_TRUE(fp.whole > 0 && held == (std::int64_t)fp.whole);
  // MiniMax-H3-Image (VPIPE_MINIMAX_H3_IMAGE_PATH): its per-step heads, a
  // weight set the run opens beside the DiT.
  if (const char* h3i = std::getenv("VPIPE_MINIMAX_H3_IMAGE_PATH")) {
    FlexData c2 = FlexData::make_object();
    c2.as_object().insert_or_assign("hf_dir", FlexData::make_string(h3i));
    c2.as_object().insert_or_assign(
        "output_dir", FlexData::make_string((root / "out").string()));
    c2.as_object().insert_or_assign("name", FlexData::make_string("ut"));
    TrainLoraStage st2(&sess, "train2", std::vector<InEdge>{}, c2);
    bool heads = false;
    for (const ResourceClaim& c : st2.declare_resources()) {
      heads = heads || (c.kind == model_memory::kWeightsKind &&
                        c.key.find("pdd_fused_heads") != std::string::npos);
    }
    std::printf("[train_lora] MiniMax-H3-Image's heads claimed: %s\n",
                heads ? "yes" : "NO");
    EXPECT_TRUE(heads);
  }
  fs::remove_all(root);
}
