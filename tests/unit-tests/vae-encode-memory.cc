// vae-encode reads a picture WHERE IT LIES, and books what it holds.
//
// Every family used to copy the beat (materialize_contiguous), build the
// picture as an f32 vector (4 bytes a sample), convert that into the
// buffer the VAE encodes from, and hold all three -- and the beat -- for
// the whole encode. MEASURED on a 42 MP still through MiniMax-H3's VAE:
// 965 MiB beside a 241 MiB upload, for 3.5 minutes, and macOS kept the
// freed host part in the footprint ~77 s more, on top of the decoder.
// Now the upload is written from the beat's own samples, the beat goes
// once it is, and a float vector exists only where one has to: a
// letterbox resample, an RGBA picture composited for an RGB VAE, a
// registered family's float API. VPIPE_VAE_ENCODE_VIA_FLOATS=1 is the
// old route, and the A/B these tests hold the new one to.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/vae-model-registry.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/vae-encode-stage.h"

#include <malloc/malloc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;

namespace {

// What a registered family was handed, for the model-free test.
std::vector<float> g_seen;
int g_seen_frames = 0;

class RecordingEncoder : public genai::VaeEncoder {
public:
  int latent_channels() const override { return 4; }
  int spatial_compression() const override { return 8; }
  int temporal_compression() const override { return 4; }

  bool encode(const genai::VaeEncodeRequest& req, std::vector<float>* out,
              std::vector<int>* shape, std::string*) override
  {
    g_seen_frames = req.frames;
    g_seen.assign(req.pixels, req.pixels + (std::size_t)req.frames * 3 *
                                               req.height * req.width);
    const int t = 1 + (req.frames - 1) / 4;
    *shape = {4, t, req.height / 8, req.width / 8};
    out->assign((std::size_t)4 * t * (req.height / 8) * (req.width / 8),
                0.0f);
    return true;
  }
};

class RecordingFamily : public genai::VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "ut-venc-mem"; }
  bool claims(const std::string& root, const std::string&,
              const std::string&) const override
  {
    return root == "/ut/venc-mem-model";
  }
  std::unique_ptr<genai::VaeDecoder>
  load_decoder(const genai::VaeModelCreateArgs&) override { return nullptr; }
  std::unique_ptr<genai::VaeEncoder>
  load_encoder(const genai::VaeModelCreateArgs&) override
  {
    return std::make_unique<RecordingEncoder>();
  }
};

bool
register_family_()
{
  static const bool once = genai::VaeModelRegistry::get().add(
      std::make_unique<RecordingFamily>());
  return once;
}

// The measurement's clock: the sink says a warm-up latent is in, the
// sampler takes its baseline and says so, and only then does the source
// let the measured picture go -- so the VAE's load (its weight
// conversions pass through the heap) is behind the baseline.
std::atomic<int> g_latents{0};
std::atomic<bool> g_arm{false}, g_armed{false};

// Emits a prebuilt warm-up beat (when there is one), waits for its
// latent and the sampler, then the prebuilt beat -- each moved, not
// copied, so the beat exists before the baseline and the encode's
// allocations are what the heap gains.
class OneBeat : public TypedStage<OneBeat> {
public:
  static constexpr const char* kTypeName = "ut-venc-mem-source";
  using TypedStage::TypedStage;
  std::unique_ptr<TensorBeatPayload> warm;
  std::unique_ptr<TensorBeatPayload> beat;
  bool warmed = false;

  Job
  process(RuntimeContext& ctx) override
  {
    if (warm) {
      warmed = true;
      co_await ctx.write(0, std::move(warm));
      co_return;
    }
    if (!beat) { ctx.signal_done(); co_return; }
    if (warmed) {
      while (g_latents.load() < 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      g_arm.store(true);
      while (!g_armed.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    co_await ctx.write(0, std::move(beat));
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {
      {.name = "image", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class Keep : public TypedStage<Keep> {
public:
  static constexpr const char* kTypeName = "ut-venc-mem-sink";
  using TypedStage::TypedStage;
  std::vector<std::uint8_t> latent;

  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    if (const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get())) {
      const auto bytes = tb->materialize_contiguous();
      latent.assign(bytes.begin(), bytes.end());
    }
    g_latents.fetch_add(1);
  }

  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {
      {.name = "latent", .doc = "", .type = &typeid(TensorBeatPayload)}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

// The host heap in use (every malloc zone): what a float picture or a
// copy of the beat takes. Metal's buffers -- the upload, the weights --
// are not in it.
std::size_t
heap_in_use_()
{
  malloc_statistics_t st{};
  malloc_zone_statistics(nullptr, &st);
  return st.size_in_use;
}

struct Run {
  std::vector<std::uint8_t> latent;
  std::size_t booked = 0;        // the stage's scratch, as revised
  std::size_t heap_growth = 0;   // the heap's peak over the baseline
  std::size_t heap_dip = 0;      // and its lowest under it
};

// One beat through vae-encode over `model`, the old route or the new.
Run
encode_(const std::string& model, std::unique_ptr<TensorBeatPayload> beat,
        bool via_floats, FlexData cfg = FlexData::make_object(),
        std::unique_ptr<TensorBeatPayload> warm = nullptr)
{
  if (via_floats) { setenv("VPIPE_VAE_ENCODE_VIA_FLOATS", "1", 1); }
  else            { unsetenv("VPIPE_VAE_ENCODE_VIA_FLOATS"); }
  g_latents.store(0);
  g_arm.store(warm == nullptr);
  g_armed.store(false);
  Run r;
  {
    Session sess;
    auto pl = std::make_unique<Pipeline>("p", &sess);
    auto src_u = std::make_unique<OneBeat>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
    src_u->warm = std::move(warm);
    src_u->beat = std::move(beat);
    src_u->allocate_oports(1);
    auto* src = pl->insert_stage(std::move(src_u));
    cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(model));
    // Kernel routes fixed, so the two routes run the same kernels.
    cfg.as_object().insert_or_assign("autotune", FlexData::make_bool(false));
    // Held through the warm-up, so the VAE stays loaded for the beat
    // that is measured.
    cfg.as_object().insert_or_assign("unload_when_idle",
                                     FlexData::make_string("never"));
    auto* enc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
        &sess, "venc", std::vector<InEdge>{{src, 0}}, cfg));
    auto* keep = static_cast<Keep*>(pl->insert_stage(
        std::make_unique<Keep>(&sess, "keep",
                               std::vector<InEdge>{{enc, 0}},
                               FlexData::make_object())));
    std::atomic<std::size_t> base{heap_in_use_()};
    std::atomic<bool> run{true};
    std::atomic<std::size_t> peak{base.load()}, low{base.load()};
    std::thread sampler([&] {
      while (run.load()) {
        if (g_arm.load() && !g_armed.load()) {
          base.store(heap_in_use_());
          peak.store(base.load());
          low.store(base.load());
          g_armed.store(true);
        }
        const std::size_t now = heap_in_use_();
        if (g_armed.load() && now > peak.load()) { peak.store(now); }
        if (g_armed.load() && now < low.load()) { low.store(now); }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
    PipelineRuntime rt(pl.get(), &sess);
    const bool ok = rt.launch();
    if (ok) { rt.wait_idle(); }
    run.store(false);
    sampler.join();
    r.heap_growth = peak.load() > base.load() ? peak.load() - base.load()
                                              : 0;
    r.heap_dip = base.load() > low.load() ? base.load() - low.load() : 0;
    if (auto* mgr = sess.services()->generative_model_manager()) {
      r.booked = mgr->declared_scratch("vae-encode:venc");
    }
    if (ok) { rt.stop(); }
    r.latent = keep->latent;
  }
  unsetenv("VPIPE_VAE_ENCODE_VIA_FLOATS");
  return r;
}

// A u8 picture [3, h, w], or a clip [frames, 3, h, w], whose every
// sample differs from its neighbours in frame, channel and place.
std::unique_ptr<TensorBeatPayload>
picture_(int frames, int c, int h, int w, bool clip)
{
  auto b = std::make_unique<TensorBeatPayload>();
  b->dtype = TensorBeat::DType::U8;
  if (clip) { b->shape = {frames, c, h, w}; }
  else      { b->shape = {c, h, w}; }
  const std::size_t n = (std::size_t)frames * c * h * w;
  b->resize_contiguous(n);
  std::uint8_t* p = b->as_u8();
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = (std::uint8_t)((i * 2654435761u >> 13) + i / 4093);
  }
  return b;
}

}  // namespace

// The model-free half: a registered family, whose API takes floats,
// is handed the beat's samples normalized and laid out [3][frames][H][W]
// -- the transpose a stacked clip needs -- straight from the beat. And
// the stage states what it holds for it before it builds it.
TEST(vae_encode_memory, a_family_is_handed_the_clip_read_in_place)
{
  EXPECT_TRUE(register_family_());
  constexpr int F = 5, H = 24, W = 40;
  auto beat = picture_(F, 3, H, W, true);
  std::vector<std::uint8_t> src(beat->as_u8(),
                                beat->as_u8() + (std::size_t)F * 3 * H * W);
  g_seen.clear();
  const Run r = encode_("/ut/venc-mem-model", std::move(beat), false);
  ASSERT_TRUE(g_seen_frames == F &&
              g_seen.size() == (std::size_t)3 * F * H * W);
  if (g_seen.size() != (std::size_t)3 * F * H * W) { return; }
  std::size_t off = 0;
  const std::size_t px = (std::size_t)H * W;
  for (int c = 0; c < 3; ++c) {
    for (int f = 0; f < F; ++f) {
      for (std::size_t i = 0; i < px; ++i) {
        const float raw = (float)src[((std::size_t)f * 3 + c) * px + i];
        const float want = raw * (2.0f / 255.0f) + -1.0f;
        if (g_seen[((std::size_t)c * F + f) * px + i] != want) { ++off; }
      }
    }
  }
  std::printf("[vae_encode_memory] family: %zu of %zu samples off, booked "
              "%zu B\n", off, g_seen.size(), r.booked);
  EXPECT_TRUE(off == 0);
  // The floats the family takes, and a clip's frame in turn.
  EXPECT_TRUE(r.booked == (std::size_t)3 * F * px * 4 + 3 * px * 4);
}

// The real codecs: the upload written from the beat is the one the old
// route built from its float vector, latent for latent, on every path a
// picture takes -- a still and a clip through MiniMax-H3's VAE, an RGB
// picture into Qwen-Image-2.1's RGBA VAE, and Z-Image's AutoencoderKL
// at the picture's size and letterboxed (the resample keeps its vector,
// and must be what it was).
TEST(vae_encode_memory, the_upload_is_the_float_routes_to_the_bit)
{
  struct Case {
    const char* env;
    const char* name;
    int frames, c, h, w;
    bool clip;
    int tw = 0, th = 0;
  };
  const Case cases[] = {
    {"VPIPE_MINIMAX_H3_TEST_MODEL_PATH", "h3 still", 1, 3, 96, 128, false},
    {"VPIPE_MINIMAX_H3_TEST_MODEL_PATH", "h3 clip", 5, 3, 64, 96, true},
    {"VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH", "qi21 rgb->rgba", 1, 3, 96, 128,
     false},
    {"VPIPE_Z_IMAGE_TEST_MODEL_PATH", "z-image", 1, 3, 96, 128, false},
    {"VPIPE_Z_IMAGE_TEST_MODEL_PATH", "z-image letterbox", 1, 3, 90, 120,
     false, 128, 96},
  };
  int ran = 0;
  for (const Case& k : cases) {
    const char* model = std::getenv(k.env);
    if (model == nullptr || *model == '\0') { continue; }
    auto cfg = [&] {
      FlexData c = FlexData::make_object();
      if (k.tw > 0) {
        c.as_object().insert_or_assign("target_width",
                                       FlexData::make_int(k.tw));
        c.as_object().insert_or_assign("target_height",
                                       FlexData::make_int(k.th));
      }
      return c;
    };
    const Run a = encode_(model, picture_(k.frames, k.c, k.h, k.w, k.clip),
                          false, cfg());
    const Run b = encode_(model, picture_(k.frames, k.c, k.h, k.w, k.clip),
                          true, cfg());
    std::printf("[vae_encode_memory] %s: latent %zu B, %s\n", k.name,
                a.latent.size(),
                a.latent == b.latent ? "identical" : "DIFFERENT");
    EXPECT_TRUE(!a.latent.empty());
    EXPECT_TRUE(a.latent == b.latent);
    ++ran;
  }
  if (ran == 0) {
    std::printf("[vae_encode_memory] SKIP: none of the VAE model paths "
                "set\n");
  }
}

// What the route holds: a 2048 x 1536 still through MiniMax-H3's VAE,
// measured once a 64^2 warm-up has loaded it. The old route's float
// picture (12 bytes a pixel) is the heap the measurement has to see --
// else a pass would prove nothing -- and the new one builds none: what
// remains is the latent's f32 beat. The picture goes once uploaded. The
// upload is Metal's (not in the heap); the plan books it with the
// latent, and the old route's floats besides.
TEST(vae_encode_memory, a_picture_is_read_where_it_lies)
{
  const char* model = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (model == nullptr || *model == '\0') {
    std::printf("[vae_encode_memory] SKIP: "
                "VPIPE_MINIMAX_H3_TEST_MODEL_PATH unset\n");
    return;
  }
  constexpr int H = 1536, W = 2048;
  const std::size_t px = (std::size_t)H * W;
  // A 64^2 warm-up first: the measurement starts once the VAE is in.
  const Run now = encode_(model, picture_(1, 3, H, W, false), false,
                          FlexData::make_object(),
                          picture_(1, 3, 64, 64, false));
  const Run old = encode_(model, picture_(1, 3, H, W, false), true,
                          FlexData::make_object(),
                          picture_(1, 3, 64, 64, false));
  std::printf("[vae_encode_memory] heap growth %.2f B/px (old route "
              "%.2f), the picture given back %.2f B/px, booked %.2f B/px "
              "(old route %.2f)\n",
              (double)now.heap_growth / px, (double)old.heap_growth / px,
              (double)now.heap_dip / px, (double)now.booked / px,
              (double)old.booked / px);
  EXPECT_TRUE(!now.latent.empty() && now.latent == old.latent);
  EXPECT_TRUE(old.heap_growth >= px * 12);
  EXPECT_TRUE(now.heap_growth < px);
  // The picture itself (3 bytes a pixel) goes once it is uploaded: the
  // heap falls under the baseline it was part of.
  EXPECT_TRUE(now.heap_dip >= px * 5 / 2);
  // The upload (bf16) + the moments and the f32 latent; the old route
  // its float picture besides.
  const std::size_t vox = (std::size_t)(H / 16) * (W / 16);
  EXPECT_TRUE(now.booked == px * 6 + (std::size_t)24 * vox * 8);
  EXPECT_TRUE(old.booked == now.booked + px * 12);
  EXPECT_TRUE(now.heap_growth <= now.booked);
}
