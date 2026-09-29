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
#include "pipeline/resource-plan.h"
#include "pipeline/typed-stage.h"
#include "stages/model-detect.h"
#include "stages/vae-encode-stage.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

// ---- THE CLAIM NAMES THE FAMILY'S FILE ----------------------------------
//
// A Comfy-style pack keeps no vae/config.json, so resolve_vae_dir() hands
// back the ROOT, and this stage's claim used to name it -- attributing
// the whole repository to the encoder. MEASURED on an LTX-2.5 pack
// holding all its variants: 154820 MB declared for a 1.4 GB VAE, and the
// DiT's streaming decision sized against 149 GB of "peers". The release
// already asked the family where its file is; the claim now asks the
// same thing.

namespace {

std::string g_claim_root;   // the one root ClaimFamily answers for

class ClaimFamily : public genai::VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-venc-claim"; }

  bool claims(const std::string& root, const std::string&,
              const std::string&) const override
  {
    return !g_claim_root.empty() && root == g_claim_root;
  }

  std::string vae_path(const std::string& root,
                       std::string_view role) const override
  {
    if (role != kRoleVideo) { return {}; }
    return (std::filesystem::path(root) / "vae" / "video-vae.safetensors")
        .string();
  }

  std::unique_ptr<genai::VaeDecoder>
  load_decoder(const genai::VaeModelCreateArgs&) override { return nullptr; }
};

void
write_bytes_(const std::filesystem::path& p, std::size_t floats)
{
  std::filesystem::create_directories(p.parent_path());
  std::string hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[" +
                    std::to_string(floats) + "],\"data_offsets\":[0," +
                    std::to_string(floats * 4) + "]}}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  const std::uint64_t len = hdr.size();
  f.write(reinterpret_cast<const char*>(&len), sizeof(len));
  f.write(hdr.data(), (std::streamsize)hdr.size());
  const std::vector<float> z(floats, 0.0f);
  f.write(reinterpret_cast<const char*>(z.data()),
          (std::streamsize)(z.size() * sizeof(float)));
}

}  // namespace

TEST(vae_encode_plugin, the_claim_names_the_familys_file_not_the_root)
{
  namespace fs = std::filesystem;
  static const bool once = genai::VaeModelRegistry::get().add(
      std::make_unique<ClaimFamily>());
  EXPECT_TRUE(once);
  const fs::path root =
      fs::canonical(fs::temp_directory_path()) / "vpipe-venc-claim";
  std::error_code ec;
  fs::remove_all(root, ec);
  const fs::path vae = root / "vae" / "video-vae.safetensors";
  write_bytes_(vae, 16);
  // Two other VAE-shaped files beside it, as LTX-2.5 ships its audio VAE
  // there, and a much larger DiT the root claim used to sweep up.
  write_bytes_(root / "vae" / "audio-vae.safetensors", 8);
  write_bytes_(root / "diffusion_models" / "dit.safetensors", 4096);
  g_claim_root = root.string();
  // Not vacuous: the host resolver alone names the root here.
  EXPECT_TRUE(resolve_vae_dir(root.string()) == root.string());

  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir",
                                   FlexData::make_string(root.string()));
  VaeEncodeStage stage(&sess, "venc", std::vector<InEdge>{}, cfg);
  const std::vector<ResourceClaim> claims = stage.declare_resources();
  bool file = false, whole = false;
  for (const ResourceClaim& c : claims) {
    std::printf("[vae_encode_plugin] claim %s\n", c.key.c_str());
    if (c.key == vae.string()) { file = true; }
    if (c.key == root.string()) { whole = true; }
  }
  EXPECT_TRUE(file);
  EXPECT_FALSE(whole);

  g_claim_root.clear();
  fs::remove_all(root, ec);
}

// ---- THE `images` LIST -> `latents` -----------------------------------
//
// A multi-reference graph hands vae-encode every picture at once. What it
// must get back is one list with a latent per picture, IN ORDER, each at
// its own size -- the order is what pairs latent i with the picture the
// conditioner's vision tower saw as reference i.

namespace {

class ListSource : public TypedStage<ListSource> {
public:
  static constexpr const char* kTypeName = "ut-venc-list-source";
  using TypedStage::TypedStage;
  std::vector<std::pair<int, int>> sizes;   // {h, w} per picture
  bool emitted = false;

  Job
  process(RuntimeContext& ctx) override
  {
    if (emitted) { ctx.signal_done(); co_return; }
    emitted = true;
    auto list = std::make_unique<TensorListPayload>();
    for (const auto& [h, w] : sizes) {
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::U8;
      tb.shape = {3, h, w};
      tb.resize_contiguous((std::size_t)3 * h * w);
      list->items.push_back(std::move(tb));
    }
    co_await ctx.write(0, std::move(list));
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "images", .doc = "", .type = &typeid(TensorListPayload)}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class ListSink : public TypedStage<ListSink> {
public:
  static constexpr const char* kTypeName = "ut-venc-list-sink";
  using TypedStage::TypedStage;
  std::vector<std::vector<std::int64_t>> shapes;
  int lists = 0;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* lp = dynamic_cast<const TensorListPayload*>(in.get())) {
      ++lists;
      for (const TensorBeat& t : lp->items) { shapes.push_back(t.shape); }
    }
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "latents", .doc = "", .type = &typeid(TensorListPayload)}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

}  // namespace

TEST(vae_encode_plugin, a_list_of_pictures_encodes_to_a_list_of_latents)
{
  EXPECT_TRUE(register_stub_());
  g_counts = Counts{};
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<ListSource>(&sess, "src",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  src_u->sizes = {{64, 64}, {32, 96}};
  src_u->allocate_oports(1);
  auto* src = pl->insert_stage(std::move(src_u));

  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir",
                                   FlexData::make_string("/ut/venc-model"));
  cfg.as_object().insert_or_assign("unload_when_idle",
                                   FlexData::make_string("keep"));
  const InEdge none{nullptr, 0};
  auto* enc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
      &sess, "venc", std::vector<InEdge>{none, none, {src, 0}}, cfg));
  auto* sink = static_cast<ListSink*>(pl->insert_stage(
      std::make_unique<ListSink>(&sess, "sink",
                                 std::vector<InEdge>{{enc, 1}},
                                 FlexData::make_object())));

  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::printf("[vae_encode_plugin] list: %d list(s), %zu latents, %d "
              "encodes, %d loads\n", sink->lists, sink->shapes.size(),
              g_counts.encodes, g_counts.loads);
  EXPECT_TRUE(sink->lists == 1);
  ASSERT_TRUE(sink->shapes.size() == 2);
  if (sink->shapes.size() != 2) { return; }
  // The stub is 32x in space and 8x in time: one frame, each at its own
  // size, in the order they went in.
  EXPECT_TRUE((sink->shapes[0] == std::vector<std::int64_t>{4, 1, 2, 2}));
  EXPECT_TRUE((sink->shapes[1] == std::vector<std::int64_t>{4, 1, 1, 3}));
  EXPECT_TRUE(g_counts.encodes == 2);
  EXPECT_TRUE(g_counts.loads == 1);
}
