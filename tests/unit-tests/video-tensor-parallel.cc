// video-tensor-parallel.cc -- generate-video's `tensor_parallel` on a
// REGISTERED family (video-host-ext.h's VideoTensorParallel and
// VideoTpClip): the stage asks the family for its widths, loads a rank
// with its share, books the share in its plan, and opens one channel a
// clip around generate(), let go however the clip ends. Pinned through a
// stub family that exchanges through the channel it is handed, two
// generate-video stages -- ranks 0 and 1, each in a session of its own,
// as two hosts are -- over an in-process transport.
//
//   vpipe_test --filter 'video_tensor_parallel*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/tensor-parallel.h"
#include "generative-models/video-host-ext.h"
#include "generative-models/video-model-registry.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/generate-video-stage.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

// ---- an in-process transport: a clip's two channels paired by order ----

struct Link {
  std::mutex mu;
  std::condition_variable cv;
  std::map<std::uint64_t, std::vector<std::uint8_t>> part[2];
};

struct Channel final : genai::TpChannel {
  std::shared_ptr<Link> link;
  int me = 0;
  std::atomic<int> exchanges{0};
  bool exchange(const void* partial, void* others, std::size_t bytes,
                genai::TpDType, std::uint64_t seq, std::string* err) override
  {
    const auto* p = static_cast<const std::uint8_t*>(partial);
    std::unique_lock<std::mutex> lk(link->mu);
    link->part[me][seq].assign(p, p + bytes);
    link->cv.notify_all();
    if (!link->cv.wait_for(lk, std::chrono::seconds(30), [&]() {
          return link->part[1 - me].count(seq) != 0;
        })) {
      if (err != nullptr) { *err = "the peer never came"; }
      return false;
    }
    const auto& o = link->part[1 - me][seq];
    if (o.size() != bytes) {
      if (err != nullptr) { *err = "the ranks are out of step"; }
      return false;
    }
    std::memcpy(others, o.data(), bytes);
    ++exchanges;
    return true;
  }
};

struct Transport final : genai::TpTransport {
  std::mutex mu;
  std::shared_ptr<Link> link;
  int opened = 0;
  std::vector<genai::TpShare> shares;
  std::string_view name() const override { return "ut-video-tp-pair"; }
  std::shared_ptr<genai::TpChannel> open(const FlexData&,
                                         const genai::TpShare& share,
                                         std::string*) override
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!link || opened == 2) {
      link = std::make_shared<Link>();
      opened = 0;
    }
    ++opened;
    shares.push_back(share);
    auto ch = std::make_shared<Channel>();
    ch->link = link;
    ch->me = share.rank;
    return ch;
  }
};

Transport*
transport_()
{
  static Transport* t = []() {
    auto u = std::make_unique<Transport>();
    Transport* p = u.get();
    genai::register_tp_transport(std::move(u));
    return p;
  }();
  return t;
}

// ---- the stub family: 8 heads, 16 feed-forward rows ----------------------

constexpr int kHeads = 8, kFfn = 16, kRounds = 3;

struct Seen {
  std::mutex mu;
  int whole_loads = 0;
  std::vector<genai::TpShare> rank_loads;
  std::vector<bool> planned_tp;        // config_resources saw a share
  int begins = 0, ends = 0, clips = 0;
  std::vector<float> values;           // each clip's latent value
};
Seen g_seen;

class SplitGenerator final : public genai::VideoGenerator,
                             public genai::VideoTpClip {
public:
  explicit SplitGenerator(genai::TpShare s) : _share(s) {}
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 8; }
  bool generate(const genai::VideoGenRequest&,
                genai::VideoGenResult* out) override
  {
    // Each round, rank r hands r + 1 and takes the peer's: the sum is
    // what both ranks' latents hold (1 + 2 = 3), or 0 where no channel.
    float v = 0.0f;
    if (_ch) {
      for (int i = 0; i < kRounds; ++i) {
        std::int32_t mine = _share.rank + 1, theirs = 0;
        std::string err;
        if (!_ch->exchange(&mine, &theirs, 4, genai::TpDType::kF32,
                           (std::uint64_t)i, &err)) {
          return false;
        }
        v = (float)(mine + theirs);
      }
    }
    {
      std::lock_guard<std::mutex> lk(g_seen.mu);
      ++g_seen.clips;
      g_seen.values.push_back(v);
    }
    out->video.assign(4 * 1 * 2 * 2, v);
    out->video_shape = {4, 1, 2, 2};
    return true;
  }
  void* query_extension(std::string_view id) noexcept override
  {
    return id == genai::kVideoTpClipExt ? static_cast<VideoTpClip*>(this)
                                        : nullptr;
  }
  bool begin_tp_generation(std::shared_ptr<genai::TpChannel> ch,
                           std::function<bool()>, std::string*) override
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    ++g_seen.begins;
    _ch = std::move(ch);
    return _ch != nullptr;
  }
  void end_tp_generation() override
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    ++g_seen.ends;
    _ch.reset();
  }

private:
  genai::TpShare _share;
  std::shared_ptr<genai::TpChannel> _ch;
};

class SplitFamily final : public genai::VideoModelFamily,
                          public genai::VideoTensorParallel,
                          public genai::VideoConfigPlan {
public:
  std::string_view tag() const noexcept override { return "ut-video-tp"; }
  bool claims(const std::string& root, const std::string&) const override
  {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(root) / "ut-video-tp.marker", ec);
  }
  std::unique_ptr<genai::VideoGenerator>
  load(const genai::VideoModelCreateArgs&) override
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    ++g_seen.whole_loads;
    return std::make_unique<SplitGenerator>(genai::TpShare{});
  }
  void* query_extension(std::string_view id) noexcept override
  {
    if (id == genai::kVideoTensorParallelExt) {
      return static_cast<VideoTensorParallel*>(this);
    }
    if (id == genai::kVideoConfigPlanExt) {
      return static_cast<VideoConfigPlan*>(this);
    }
    return nullptr;
  }
  // VideoTensorParallel
  bool tp_widths(const std::string&, int* n_heads, int* n_ffn,
                 std::string*) const override
  {
    *n_heads = kHeads;
    *n_ffn = kFfn;
    return true;
  }
  std::unique_ptr<genai::VideoGenerator>
  load_rank(const genai::VideoModelCreateArgs&,
            const genai::TpShare& share) override
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    g_seen.rank_loads.push_back(share);
    return std::make_unique<SplitGenerator>(share);
  }
  // VideoConfigPlan: what the plan asks with
  std::vector<ResourceClaim>
  config_resources(const std::string&,
                   const genai::VideoPlanContext& ctx) const override
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    g_seen.planned_tp.push_back(ctx.tp != nullptr && ctx.tp->active() &&
                                ctx.tp->n_heads == kHeads &&
                                ctx.tp->n_ffn == kFfn);
    return {};
  }
  std::vector<StageHolding>
  config_holdings(const std::string&,
                  const genai::VideoPlanContext&) const override
  {
    return {};
  }
  std::size_t config_denoise_scratch_bytes(
      const std::string&, int, int, int,
      const genai::VideoPlanContext&) const override
  {
    return 0;
  }
  bool output_clip(const std::string&, const FlexData*, int, int, int, int*,
                   int*, int*, double*) const override
  {
    return false;
  }
  std::size_t decode_working_bytes(const std::string&, int, int,
                                   int) const override
  {
    return 0;
  }
};

// A family that does NOT split: tensor_parallel is refused by name.
class WholeFamily final : public genai::VideoModelFamily {
public:
  std::string_view tag() const noexcept override
  {
    return "ut-video-tp-whole";
  }
  bool claims(const std::string& root, const std::string&) const override
  {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(root) / "ut-video-tp-whole.marker", ec);
  }
  std::unique_ptr<genai::VideoGenerator>
  load(const genai::VideoModelCreateArgs&) override
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    ++g_seen.whole_loads;
    return std::make_unique<SplitGenerator>(genai::TpShare{});
  }
};

void
registered_()
{
  static const bool once = [] {
    genai::VideoModelRegistry::get().add(std::make_unique<SplitFamily>());
    genai::VideoModelRegistry::get().add(std::make_unique<WholeFamily>());
    return true;
  }();
  (void)once;
  transport_();
}

// The conditioning: `n` beats, then the end of the stream.
class Conds : public TypedStage<Conds> {
public:
  static constexpr const char* kTypeName = "ut-video-tp-conds";
  using TypedStage::TypedStage;
  int n = 1, sent = 0;
  Job process(RuntimeContext& ctx) override
  {
    if (sent < n) {
      ++sent;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {2, 8};
      tb->resize_contiguous(16);
      for (int i = 0; i < 16; ++i) { tb->as_f32()[i] = 0.5f; }
      co_await ctx.write(0, std::move(tb));
      co_return;
    }
    ctx.signal_done();
  }
  const StageSpec& spec() const noexcept override
  {
    static const PortSpec op[] = {{.name = "out", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class Drain : public TypedStage<Drain> {
public:
  static constexpr const char* kTypeName = "ut-video-tp-drain";
  using TypedStage::TypedStage;
  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); }
  }
  const StageSpec& spec() const noexcept override
  {
    static const PortSpec ip[] = {{.name = "in", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// One rank's run, in a session of its own: `clips` prompts through a
// generate-video stage over `root`, `tp` its tensor_parallel bag (null:
// none). False when it did not launch.
bool
run_rank_(const std::filesystem::path& root, int clips, const FlexData& tp)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto cu = std::make_unique<Conds>(&sess, "cond", std::vector<InEdge>{},
                                    FlexData::make_object());
  cu->n = clips;
  cu->allocate_oports(1);
  auto* cond = pl->insert_stage(std::move(cu));
  std::vector<InEdge> in(11, InEdge{nullptr, 0});
  in[0] = {cond, 0};
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("hf_dir", FlexData::make_string(root.string()));
    o.insert_or_assign("width", FlexData::make_int(64));
    o.insert_or_assign("height", FlexData::make_int(64));
    o.insert_or_assign("frames", FlexData::make_int(9));
    o.insert_or_assign("steps", FlexData::make_int(2));
    if (tp.is_object()) { o.insert_or_assign("tensor_parallel", tp); }
  }
  auto gu = std::make_unique<GenerateVideoStage>(&sess, "gv", in, cfg);
  if (!gu->config_error().empty()) { return false; }
  auto* gv = pl->insert_stage(std::move(gu));
  pl->insert_stage(std::make_unique<Drain>(
      &sess, "lat", std::vector<InEdge>{{gv, 0}}, FlexData::make_object()));
  PipelineRuntime rt(pl.get(), &sess);
  const bool launched = rt.launch();
  if (launched) {
    rt.wait_idle();
    rt.stop();
  }
  return launched;
}

FlexData
tp_bag_(int rank)
{
  FlexData b = FlexData::make_object();
  auto o = b.as_object();
  o.insert_or_assign("rank", FlexData::make_int(rank));
  o.insert_or_assign("ranks", FlexData::make_int(2));
  o.insert_or_assign("transport", FlexData::make_string("ut-video-tp-pair"));
  return b;
}

std::filesystem::path
root_(const char* marker)
{
  const auto root = std::filesystem::temp_directory_path() /
                    (std::string("vpipe-ut-video-tp-") + marker);
  std::filesystem::create_directories(root);
  std::ofstream(root / (std::string(marker) + ".marker")) << "x";
  return root;
}

}  // namespace

// TWO RANKS OF A REGISTERED FAMILY: each stage loads ITS SHARE through
// load_rank (an even split of the family's widths, unsaid in the bag),
// books it in its plan (the plan context carries the share), and opens
// one channel a clip, begun before generate() and ended after it -- the
// clips' exchanges paired, both ranks' latents the sum of the two.
TEST(video_tensor_parallel, two_ranks_of_a_registered_family)
{
  registered_();
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    g_seen.whole_loads = 0;
    g_seen.rank_loads.clear();
    g_seen.planned_tp.clear();
    g_seen.begins = g_seen.ends = g_seen.clips = 0;
    g_seen.values.clear();
  }
  const auto root = root_("ut-video-tp");
  bool l1 = false;
  std::thread t([&]() { l1 = run_rank_(root, 2, tp_bag_(1)); });
  const bool l0 = run_rank_(root, 2, tp_bag_(0));
  t.join();
  std::filesystem::remove_all(root);
  EXPECT_TRUE(l0 && l1);
  std::lock_guard<std::mutex> lk(g_seen.mu);
  // Every load a rank's, at its share (a stage may let its generator go
  // between clips -- the default idle policy -- and load it again).
  bool shares_ok = g_seen.rank_loads.size() >= 2;
  bool rank_seen[2] = {false, false};
  for (const genai::TpShare& s : g_seen.rank_loads) {
    const int r = s.rank;
    shares_ok = shares_ok && (r == 0 || r == 1) && s.ranks == 2 &&
                s.n_heads == kHeads && s.n_ffn == kFfn &&
                s.head0 == r * kHeads / 2 &&
                s.head1 == (r + 1) * kHeads / 2 && s.ffn0 == r * kFfn / 2 &&
                s.ffn1 == (r + 1) * kFfn / 2;
    if (r == 0 || r == 1) { rank_seen[r] = true; }
  }
  shares_ok = shares_ok && rank_seen[0] && rank_seen[1];
  bool planned = !g_seen.planned_tp.empty();
  for (bool b : g_seen.planned_tp) { planned = planned && b; }
  bool values = g_seen.values.size() == 4;
  for (float v : g_seen.values) { values = values && v == 3.0f; }
  std::printf("[video-tp] whole loads %d, rank loads %zu (shares %s), "
              "plans with the share %zu of %zu, channels begun %d / ended "
              "%d, clips %d, values %s; channels opened %zu\n",
              g_seen.whole_loads, g_seen.rank_loads.size(),
              shares_ok ? "even" : "WRONG",
              (std::size_t)std::count(g_seen.planned_tp.begin(),
                                      g_seen.planned_tp.end(), true),
              g_seen.planned_tp.size(), g_seen.begins, g_seen.ends,
              g_seen.clips, values ? "3 each" : "WRONG",
              transport_()->shares.size());
  EXPECT_TRUE(g_seen.whole_loads == 0);
  EXPECT_TRUE(shares_ok);
  EXPECT_TRUE(planned);
  EXPECT_TRUE(g_seen.begins == 4 && g_seen.ends == 4);
  EXPECT_TRUE(g_seen.clips == 4 && values);
}

// A FAMILY THAT DOES NOT SPLIT, given a share: refused by name at load --
// nothing loaded, no clip made.
TEST(video_tensor_parallel, a_family_that_does_not_split_is_refused)
{
  registered_();
  {
    std::lock_guard<std::mutex> lk(g_seen.mu);
    g_seen.whole_loads = 0;
    g_seen.clips = 0;
  }
  const auto root = root_("ut-video-tp-whole");
  run_rank_(root, 1, tp_bag_(0));
  std::filesystem::remove_all(root);
  std::lock_guard<std::mutex> lk(g_seen.mu);
  std::printf("[video-tp] a family without the split: %d loads, %d clips\n",
              g_seen.whole_loads, g_seen.clips);
  EXPECT_TRUE(g_seen.whole_loads == 0 && g_seen.clips == 0);
}
