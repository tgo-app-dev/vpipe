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
#include "generative-models/train/sample-cache.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
