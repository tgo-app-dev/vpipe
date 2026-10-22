// Fibo 2.0 Edit Turbo against the released checkpoint: the conditioner,
// the DiT step by step, and an edit through the stages.
//
// The golden is tools/dump_fibo2_golden.py `real` -- the reference's own
// preprocessing of one picture (its 512^2 crop), its conditioning in fp32
// and bf16, every step of its fp32 4-step edit and the bf16 DiT at the
// same points. Each check is held to the reference's OWN bf16 distance
// from its fp32 run, the bar a bf16 port can be asked to meet.
//
// Env: VPIPE_FIBO2_TEST_MODEL_PATH -> the repo ROOT (briaai/Fibo-2.0-
// edit-turbo); VPIPE_FIBO2_REAL_GOLDEN -> a `real` dir of one picture,
// VPIPE_FIBO2_REAL2_GOLDEN -> one of two (`--image2`); the end-to-end
// edits also want VPIPE_FIBO2_E2E=1. Unset, every test skips vacuously --
// check that they RAN.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/fibo2/fibo2-layout.h"
#include "generative-models/fibo2/metal-fibo2-transformer.h"
#include "generative-models/generative-model-manager.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/model-memory.h"
#include "stages/vae-encode-stage.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;

namespace {

namespace fs = std::filesystem;

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v == nullptr || *v == '\0') ? std::string() : std::string(v);
}

bool
read_json_(const fs::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  try {
    *out = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  return out->is_object();
}

std::vector<float>
read_f32_(const fs::path& p)
{
  std::vector<float> v;
  std::ifstream in(p, std::ios::binary);
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  v.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

double
rel_l2_(const float* got, const float* want, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::uint16_t
f2bf_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  return (std::uint16_t)((u + 0x7fffu + r) >> 16);
}

float
bf2f_(std::uint16_t h)
{
  std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// [32, H8, W8] -> FLUX.2's patchified [128, H8/2, W8/2] (c*4 + dy*2 + dx),
// the layout vae-encode emits and generate-image returns.
std::vector<float>
patchify_(const std::vector<float>& z, int H8, int W8)
{
  const int gh = H8 / 2, gw = W8 / 2;
  std::vector<float> rows((std::size_t)gh * gw * 128);
  genai::fibo2::pack_latents(z.data(), 32, H8, W8, 2, rows.data());
  std::vector<float> out(rows.size());
  for (int t = 0; t < gh * gw; ++t) {
    for (int c = 0; c < 128; ++c) {
      out[(std::size_t)c * gh * gw + t] = rows[(std::size_t)t * 128 + c];
    }
  }
  return out;
}

// What a shared helper checked, for its TEST to assert: minitest's
// macros only work inside a test body.
struct Checks {
  bool ran = false;
  std::vector<std::pair<bool, std::string>> items;
  bool
  add(bool ok, const char* what)
  {
    items.emplace_back(ok, what);
    if (!ok) { std::printf("[fibo2] check failed: %s\n", what); }
    return ok;
  }
};
// Checks `cond`; on failure returns `c` from the helper.
#define FB_CHECK_OR_RETURN(c, cond)                                        \
  do {                                                                     \
    if (!(c).add((cond), #cond)) { return (c); }                           \
  } while (0)
#define FB_CHECK(c, cond) (void)(c).add((cond), #cond)

struct Golden {
  bool ok = false;
  fs::path dir;
  std::string root, prompt;
  int size = 0, steps = 0, seed = 0, text_len = 0;
  std::vector<std::pair<int, int>> images;   // each crop's (w, h)
  std::vector<double> sigmas, vel_bar;
  double emb_bar = 0.0, final_bar = 0.0;
};

// `env`: VPIPE_FIBO2_REAL_GOLDEN (one picture) or VPIPE_FIBO2_REAL2_GOLDEN
// (two).
Golden
load_(const char* env)
{
  Golden G;
  G.root = env_("VPIPE_FIBO2_TEST_MODEL_PATH");
  const std::string g = env_(env);
  if (G.root.empty() || g.empty()) { return G; }
  G.dir = fs::path(g);
  FlexData man;
  if (!read_json_(G.dir / "golden.json", &man)) { return G; }
  auto o = man.as_object();
  G.prompt = std::string(o.at("prompt").as_string(""));
  G.size = (int)o.at("size").as_int(0);
  G.steps = (int)o.at("steps").as_int(0);
  G.seed = (int)o.at("seed").as_int(0);
  G.text_len = (int)o.at("text_len").as_int(0);
  G.emb_bar = o.at("embeds_ref_bf16_rel").as_real(0.0);
  G.final_bar = o.at("final_ref_bf16_rel").as_real(0.0);
  const FlexData sg = o.at("sigmas");
  for (std::size_t i = 0; i < sg.as_array().size(); ++i) {
    G.sigmas.push_back(sg.as_array().at(i).as_real(0.0));
  }
  const FlexData vb = o.at("velocity_ref_bf16_rel");
  for (std::size_t i = 0; i < vb.as_array().size(); ++i) {
    G.vel_bar.push_back(vb.as_array().at(i).as_real(0.0));
  }
  const FlexData im = o.at("images");
  for (std::size_t i = 0; i < im.as_array().size(); ++i) {
    const FlexData wh = im.as_array().at(i);
    G.images.emplace_back((int)wh.as_array().at(0).as_int(0),
                          (int)wh.as_array().at(1).as_int(0));
  }
  G.ok = G.size > 0 && G.steps > 0 && G.text_len > 0 &&
         !G.images.empty() && (int)G.sigmas.size() == G.steps + 1 &&
         (int)G.vel_bar.size() == G.steps;
  return G;
}

// Picture k as the reference cropped it, U8 [3, H, W] -- from its PNG's
// raw RGB twin (crop<k>_rgb.u8, HWC).
std::vector<std::uint8_t>
crop_chw_(const Golden& G, int k)
{
  const int W = G.images[(std::size_t)k].first;
  const int H = G.images[(std::size_t)k].second;
  std::vector<std::uint8_t> hwc((std::size_t)W * H * 3);
  std::ifstream in(G.dir / ("crop" + std::to_string(k) + "_rgb.u8"),
                   std::ios::binary);
  if (!in) { return {}; }
  in.read(reinterpret_cast<char*>(hwc.data()), (std::streamsize)hwc.size());
  if (!in) { return {}; }
  std::vector<std::uint8_t> chw(hwc.size());
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        chw[((std::size_t)c * H + y) * W + x] =
            hwc[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }
  return chw;
}

class SourceText : public TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-source-text-fibo2";
  SourceText(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::string> prompts;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next < prompts.size()) {
      co_await ctx.write(0, std::make_unique<FlexDataPayload>(
                                FlexData::make_string(prompts[next++])));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

// Emits one U8 [3, H, W] picture, then ends.
class SourcePicture : public TypedStage<SourcePicture> {
public:
  static constexpr const char* kTypeName = "ut-source-picture-fibo2";
  SourcePicture(const SessionContextIntf* s, std::string id,
                std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::uint8_t> chw;
  int H = 0, W = 0;
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::U8;
      tb->shape = {3, H, W};
      tb->resize_contiguous(chw.size());
      std::memcpy(tb->as_u8(), chw.data(), chw.size());
      co_await ctx.write(0, std::move(tb));
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-fibo2";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// One source stage per picture, in order; empty on a failure.
std::vector<Stage*>
pictures_(Pipeline* pl, Session& sess, const Golden& G)
{
  std::vector<Stage*> out;
  for (std::size_t k = 0; k < G.images.size(); ++k) {
    auto pu = std::make_unique<SourcePicture>(
        &sess, "pic" + std::to_string(k), std::vector<InEdge>{},
        FlexData::make_object());
    pu->chw = crop_chw_(G, (int)k);
    if (pu->chw.empty()) { return {}; }
    pu->W = G.images[k].first;
    pu->H = G.images[k].second;
    out.push_back(pl->insert_stage(std::move(pu)));
  }
  return out;
}

// The conditioner over `prompts`, its pictures on ref_image / ref_image2.
Stage*
conditioner_(Pipeline* pl, Session& sess, const Golden& G, Stage* src,
             const std::vector<Stage*>& pics)
{
  std::vector<InEdge> in{{src, 0}, InEdge{nullptr, 0}, InEdge{nullptr, 0}};
  for (Stage* p : pics) { in.push_back({p, 0}); }
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(G.root));
  return pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::move(in), std::move(c)));
}

Checks
conditioning_test_(const char* env)
{
  Checks c;
  const Golden G = load_(env);
  if (!G.ok) { return c; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return c; }
  c.ran = true;
  auto pl = std::make_unique<Pipeline>("fibo2-cond", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = {G.prompt};
  Stage* src = pl->insert_stage(std::move(srcu));
  const std::vector<Stage*> pics = pictures_(pl.get(), sess, G);
  FB_CHECK_OR_RETURN(c, pics.size() == G.images.size());
  Stage* cond = conditioner_(pl.get(), sess, G, src, pics);
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{cond, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  FB_CHECK(c, rt.launch());
  rt.wait_idle();
  rt.stop();
  FB_CHECK_OR_RETURN(c, sink->captured.size() == 1);
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  FB_CHECK_OR_RETURN(c, tb != nullptr);
  FB_CHECK(c, tb->dtype == TensorBeat::DType::Bf16);
  const bool shape_ok = tb->shape.size() == 3 && tb->shape[0] == 6 &&
                        tb->shape[1] == G.text_len && tb->shape[2] == 7680;
  if (!shape_ok && tb->shape.size() == 3) {
    std::printf("[fibo2_cond] beat [%lld, %lld, %lld], the reference [6, "
                "%d, 7680]\n", (long long)tb->shape[0],
                (long long)tb->shape[1], (long long)tb->shape[2],
                G.text_len);
  }
  FB_CHECK_OR_RETURN(c, shape_ok);
  const std::size_t n = (std::size_t)6 * G.text_len * 7680;
  std::vector<float> got(n);
  const auto* u =
      reinterpret_cast<const std::uint16_t*>((const void*)tb->as_u8());
  for (std::size_t k = 0; k < n; ++k) { got[k] = bf2f_(u[k]); }
  const std::vector<float> want = read_f32_(G.dir / "embeds.f32");
  FB_CHECK_OR_RETURN(c, want.size() == n);
  const double rel = rel_l2_(got.data(), want.data(), n);
  // Per bundle, to name the one that is off when one is.
  std::printf("[fibo2_cond] %zu picture(s), [6, %d, 7680]: rel-L2 %.5f (the "
              "reference's own bf16 %.5f); per bundle", G.images.size(),
              G.text_len, rel, G.emb_bar);
  const std::size_t per = (std::size_t)G.text_len * 7680;
  for (int b = 0; b < 6; ++b) {
    std::printf(" %.4f", rel_l2_(got.data() + b * per,
                                 want.data() + b * per, per));
  }
  std::printf("\n");
  FB_CHECK(c, rel <= 1.25 * G.emb_bar);
  return c;
}

// What ONE forward added to the device's allocation AT ITS PEAK, sampled
// from another thread while it ran (its planes are made per forward and
// let go when it returns), beside what the plan books for it
// (forward_scratch_bytes): the gist and the injection keys / values the
// text pass leaves, and the activations.
// With `streamed`, the WEIGHTS instead: what a streamed DiT holds after
// a forward with residency promoting nothing (no reserve declared) --
// the trunk and its slot pair, less the prompt's products, which the
// arena above books -- beside the floor the plan declares for it.
struct ScratchMeasure {
  bool streamed = false;
  std::size_t grew = 0;
  std::size_t booked = 0;
  std::size_t held = 0;
  std::size_t floor = 0;
};

// `sm`: measure one forward instead of walking the steps -- preloaded for
// the activations (a streamed one would count its slots and promoted
// blocks too), streamed for the floor.
Checks
steps_test_(const char* env, ScratchMeasure* sm = nullptr)
{
  Checks c;
  const Golden G = load_(env);
  if (!G.ok) { return c; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return c; }
  c.ran = true;
  const std::string dit = (fs::path(G.root) / "transformer").string();
  genai::MetalFibo2Transformer::Config cfg;
  FB_CHECK_OR_RETURN(
      c, genai::MetalFibo2Transformer::Config::read_dims(dit, &cfg));
  const bool stream =
      sm != nullptr ? sm->streamed
                    : !env_("VPIPE_FIBO2_STREAM").empty() &&
                          std::atoi(env_("VPIPE_FIBO2_STREAM").c_str()) != 0;
  auto m = genai::MetalFibo2Transformer::load(dit, mc, cfg, stream);
  FB_CHECK_OR_RETURN(c, m != nullptr);
  const std::vector<float> emb = read_f32_(G.dir / "embeds.f32");
  const int H8 = G.size / 8, gh = H8 / 2;
  const int PD = cfg.patch_dim();
  FB_CHECK_OR_RETURN(c, emb.size() == (std::size_t)6 * G.text_len * 7680);
  metal_compute::SharedBuffer tb = mc->make_shared_buffer(emb.size() * 2);
  {
    auto* d = static_cast<std::uint16_t*>(tb.contents());
    for (std::size_t i = 0; i < emb.size(); ++i) { d[i] = f2bf_(emb[i]); }
  }
  std::string err;
  // Before the text: the text pass's products are the forward's to book.
  const std::size_t alloc0 =
      sm != nullptr ? mc->memory_budget_settled().allocated : 0;
  FB_CHECK_OR_RETURN(c, m->set_text(tb, G.text_len, &err));
  // Each picture's latent, in vae-encode's [128, h, w], is its patch rows.
  std::vector<std::pair<int, int>> grids;
  std::vector<std::vector<float>> ctx;
  for (std::size_t k = 0; k < G.images.size(); ++k) {
    const int ch = G.images[k].second / 16, cw = G.images[k].first / 16;
    ctx.push_back(read_f32_(G.dir / ("ctx" + std::to_string(k) + ".f32")));
    FB_CHECK_OR_RETURN(c, ctx.back().size() == (std::size_t)PD * ch * cw);
    grids.emplace_back(ch, cw);
  }
  genai::fibo2::Layout lay;
  FB_CHECK_OR_RETURN(
      c, genai::fibo2::build_layout(gh, gh, grids, m->gist_count(),
                                    cfg.max_gist, &lay, &err));
  const std::size_t img = (std::size_t)gh * gh;
  metal_compute::SharedBuffer lat =
      mc->make_shared_buffer((std::size_t)lay.refiner_len() * PD * 2);
  FB_CHECK_OR_RETURN(c, !lat.empty());
  auto* lb = static_cast<std::uint16_t*>(lat.contents());
  {
    std::size_t row = img;
    for (std::size_t k = 0; k < ctx.size(); ++k) {
      const std::size_t n =
          (std::size_t)grids[k].first * (std::size_t)grids[k].second;
      for (std::size_t t = 0; t < n; ++t) {
        for (int ch = 0; ch < PD; ++ch) {
          lb[(row + t) * PD + ch] = f2bf_(ctx[k][(std::size_t)ch * n + t]);
        }
      }
      row += n;
    }
  }
  for (int k = 0; k < G.steps; ++k) {
    const std::vector<float> x =
        read_f32_(G.dir / ("x" + std::to_string(k) + ".f32"));
    const std::vector<float> want =
        read_f32_(G.dir / ("v" + std::to_string(k) + ".f32"));
    FB_CHECK_OR_RETURN(c, x.size() == (std::size_t)32 * H8 * H8 &&
                              want.size() == x.size());
    std::vector<float> rows(img * PD);
    genai::fibo2::pack_latents(x.data(), 32, H8, H8, 2, rows.data());
    for (std::size_t i = 0; i < rows.size(); ++i) { lb[i] = f2bf_(rows[i]); }
    genai::MetalFibo2Transformer::Request req;
    req.latents = &lat;
    req.layout = &lay;
    const float tt = (float)G.sigmas[(std::size_t)k] * 1000.0f;
    req.timestep = tt / 1000.0f;
    std::atomic<bool> done{false};
    std::atomic<std::size_t> peak{0};
    std::thread sampler;
    if (sm != nullptr) {
      sampler = std::thread([&] {
        while (!done.load()) {
          const std::size_t a = mc->memory_budget().allocated;
          if (a > peak.load()) { peak.store(a); }
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
      });
    }
    metal_compute::SharedBuffer v = m->forward(req, &err);
    if (sm != nullptr) {
      done.store(true);
      sampler.join();
    }
    if (v.empty()) {
      std::printf("[fibo2_dit] step %d failed: %s\n", k, err.c_str());
    }
    FB_CHECK_OR_RETURN(c, !v.empty());
    std::vector<float> vr(img * PD);
    const auto* vp = static_cast<const std::uint16_t*>(v.contents());
    for (std::size_t i = 0; i < vr.size(); ++i) { vr[i] = bf2f_(vp[i]); }
    std::vector<float> got((std::size_t)32 * H8 * H8);
    genai::fibo2::unpack_latents(vr.data(), 32, H8, H8, 2, got.data());
    const double rel = rel_l2_(got.data(), want.data(), got.size());
    std::printf("[fibo2_dit] %zu picture(s), step %d (sigma %.4f, %d gists, "
                "joint %d): velocity rel-L2 %.5f (the reference's bf16 "
                "%.5f)\n", G.images.size(), k, G.sigmas[(std::size_t)k],
                m->gist_count(), lay.joint_len(), rel,
                G.vel_bar[(std::size_t)k]);
    FB_CHECK(c, rel <= 1.25 * G.vel_bar[(std::size_t)k]);
    if (sm != nullptr && stream) {
      const std::size_t H = (std::size_t)cfg.hidden;
      const std::size_t prompt =
          (std::size_t)m->gist_count() * H * 2 +
          2 * cfg.injection_ids.size() * (std::size_t)G.text_len * H * 2;
      sm->held = (std::size_t)m->resident_bytes() - prompt;
      sm->floor = genai::MetalFibo2Transformer::streaming_floor_bytes(dit);
      break;
    }
    if (sm != nullptr) {
      // The latent buffer is the test's own, made after the first read.
      const std::size_t top =
          std::max(peak.load(), mc->memory_budget().allocated);
      sm->grew = top - alloc0 - lat.byte_size();
      sm->booked = genai::MetalFibo2Transformer::forward_scratch_bytes(
          cfg, lay.refiner_len(), G.text_len, false,
          mc->supports_matrix_cores());
      break;
    }
  }
  return c;
}

// The pictures and `prompts` through the stages: diffusion-conditioner
// and vae-encode, then generate-image at the golden's seed (every
// generation at it) -- one latent per prompt (the pictures serve them
// all), empty on a failure. `size_from_reference` leaves generate-image
// without a width and height, as the shipped graphs do (the plan then
// cannot size the DiT's activations); `denoise_scratch` receives what the
// plan's denoise phase holds once the generations ran.
std::vector<std::vector<float>>
run_edits_(const Golden& G, const std::vector<std::string>& prompts,
           bool size_from_reference = false,
           std::size_t* denoise_scratch = nullptr)
{
  std::vector<std::vector<float>> out;
  Session sess;
  if (sess.metal_compute() == nullptr) { return out; }
  auto pl = std::make_unique<Pipeline>("fibo2-stages", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = prompts;
  Stage* src = pl->insert_stage(std::move(srcu));
  const std::vector<Stage*> pics = pictures_(pl.get(), sess, G);
  if (pics.size() != G.images.size() || pics.size() > 2) { return out; }
  Stage* cond = conditioner_(pl.get(), sess, G, src, pics);
  std::vector<InEdge> gin{{cond, 0}, InEdge{nullptr, 0},
                          InEdge{nullptr, 0}, InEdge{nullptr, 0},
                          InEdge{nullptr, 0}};
  for (std::size_t k = 0; k < pics.size(); ++k) {
    FlexData vc = FlexData::make_object();
    vc.as_object().insert("hf_dir", FlexData::make_string(G.root));
    Stage* venc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
        &sess, "venc" + std::to_string(k), std::vector<InEdge>{{pics[k], 0}},
        std::move(vc)));
    gin.push_back({venc, 0});       // ref_latent0, ref_latent1
  }
  FlexData gc = FlexData::make_object();
  gc.as_object().insert("hf_dir", FlexData::make_string(G.root));
  gc.as_object().insert("seed", FlexData::make_int(G.seed));
  // Every generation at the golden's seed (the default increments it).
  gc.as_object().insert("seed_sequence", FlexData::make_string("keep"));
  // The output takes the FIRST picture's size, as the reference does.
  if (!size_from_reference) {
    gc.as_object().insert("width", FlexData::make_int(G.size));
    gc.as_object().insert("height", FlexData::make_int(G.size));
  }
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "gen", std::move(gin),
                                           std::move(gc))));
  if (!gen->config_error().empty()) { return out; }
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return out; }
  rt.wait_idle();
  if (denoise_scratch != nullptr) {
    auto* mgr = sess.generative_model_manager();
    *denoise_scratch =
        mgr != nullptr
            ? mgr->scratch_bytes(std::string(model_memory::kPhaseDenoise))
            : 0;
  }
  rt.stop();
  for (const auto& p : sink->captured) {
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (tb == nullptr || tb->dtype != TensorBeat::DType::F32) { return {}; }
    const float* f = tb->as_f32();
    out.emplace_back(f, f + tb->element_count());
  }
  return out;
}

Checks
edit_test_(const char* env)
{
  Checks c;
  const Golden G = load_(env);
  if (!G.ok || env_("VPIPE_FIBO2_E2E").empty()) { return c; }
  c.ran = true;
  const auto got = run_edits_(G, {G.prompt});
  FB_CHECK_OR_RETURN(c, got.size() == 1);
  const int H8 = G.size / 8;
  const std::vector<float> want =
      patchify_(read_f32_(G.dir / "final.f32"), H8, H8);
  const std::vector<float> ref16 =
      patchify_(read_f32_(G.dir / "final_bf16.f32"), H8, H8);
  FB_CHECK_OR_RETURN(c, got[0].size() == want.size());
  const double rel = rel_l2_(got[0].data(), want.data(), want.size());
  const double bar = rel_l2_(ref16.data(), want.data(), want.size());
  std::printf("[fibo2_stage] %zu picture(s) through the stages (%dx%d, %d "
              "steps, seed %d): latent rel-L2 %.5f (the reference's own "
              "bf16 %.5f)\n", G.images.size(), G.size, G.size, G.steps,
              G.seed, rel, bar);
  FB_CHECK(c, rel < 2.0 * bar);
  return c;
}

}  // namespace

// The stage, end to end: an instruction and the pictures to edit in, the
// six bundles [6, rows, 7680] bf16 out -- every row the reference's,
// held to its own bf16 distance. A wrong JSON, template, splice, mROPE
// band, tap or normalization is a different tensor, not a few percent.
TEST(fibo2_cond, conditioning_matches_the_reference)
{
  const Checks c = conditioning_test_("VPIPE_FIBO2_REAL_GOLDEN");
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
}

// Two pictures: "image_1" / "image_2" markers, each picture its own
// tower run and mROPE band, spliced in order.
TEST(fibo2_cond, two_pictures_match_the_reference)
{
  const Checks c = conditioning_test_("VPIPE_FIBO2_REAL2_GOLDEN");
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
}

// THE DiT, TEACHER-FORCED: at every step's fp32 point, from the fp32
// conditioning and the pictures' fp32 latents, the velocity against the
// reference's fp32 one -- each held to its bf16 DiT's distance at the
// same point (which ran on the reference's own bf16 conditioning, so the
// bar is generous at the first step).
TEST(fibo2_dit, real_steps_match_the_reference)
{
  const Checks c = steps_test_("VPIPE_FIBO2_REAL_GOLDEN");
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
}

// THE DECLARED FLOOR, against what a streamed DiT holds: the trunk (the
// Perceiver and the injection blocks' cross-attention included) and the
// slot pair, with residency promoting nothing. The floor is what the
// claim tells the plan the model can shrink to.
TEST(fibo2_dit, a_streamed_dit_holds_its_declared_floor)
{
  ScratchMeasure sm;
  sm.streamed = true;
  const Checks c = steps_test_("VPIPE_FIBO2_REAL_GOLDEN", &sm);
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
  if (!c.ran || sm.floor == 0) { return; }
  std::printf("[fibo2_dit] streamed: holds %zu bytes of weights; the "
              "floor declares %zu (%+lld)\n", sm.held, sm.floor,
              (long long)sm.held - (long long)sm.floor);
  // To the byte. (The injection gates, a scalar each in the file and a
  // broadcast row in the model, were once 35,830 bytes short of it.)
  EXPECT_TRUE(sm.held == sm.floor);
}

// THE PLAN'S FIGURE FOR A FORWARD, against the device. generate-image
// books forward_scratch_bytes in the denoise phase (and revises it to the
// edit's real rows); this holds that number to what one preloaded forward
// of the real checkpoint actually allocates, at the golden's geometry.
TEST(fibo2_dit, a_forward_allocates_what_the_plan_books)
{
  for (const char* env :
       {"VPIPE_FIBO2_REAL_GOLDEN", "VPIPE_FIBO2_REAL2_GOLDEN"}) {
    ScratchMeasure sm;
    const Checks c = steps_test_(env, &sm);
    for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
    if (!c.ran || sm.booked == 0) { continue; }
    std::printf("[fibo2_dit] %s: one forward's peak allocation %.1f MB; "
                "the plan books %.1f MB (%.3fx)\n", env,
                sm.grew / 1048576.0, sm.booked / 1048576.0,
                sm.grew > 0 ? (double)sm.booked / (double)sm.grew : 0.0);
    // The sampled peak can only under-read the true one; 2 MB covers the
    // per-forward parameter blocks the formula leaves out.
    EXPECT_TRUE(sm.grew > 0);
    EXPECT_TRUE(sm.booked + (std::size_t(2) << 20) >= sm.grew);
    EXPECT_TRUE((double)sm.booked <= 1.25 * (double)sm.grew);
  }
}

TEST(fibo2_dit, real_steps_with_two_pictures_match_the_reference)
{
  const Checks c = steps_test_("VPIPE_FIBO2_REAL2_GOLDEN");
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
}

// The edit through the stages (opt-in, VPIPE_FIBO2_E2E): the reference's
// crops through diffusion-conditioner and vae-encode, then generate-image
// at the golden's seed -- the endpoint latent against the reference's
// fp32 one. The stages add their own conditioning and VAE encode to the
// DiT's error where the reference's bf16 run takes its own, so the bound
// is the bar's ORDER: a fault in the noise, schedule, packing or layout
// is a different latent, near 1.
TEST(fibo2_stage, edit_matches_reference)
{
  const Checks c = edit_test_("VPIPE_FIBO2_REAL_GOLDEN");
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
}

TEST(fibo2_stage, a_two_picture_edit_matches_reference)
{
  const Checks c = edit_test_("VPIPE_FIBO2_REAL2_GOLDEN");
  for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
}

// SECOND GENERATION IS THE TEST: two prompts in one process -- the
// conditioner latching the one picture, generate-image broadcasting its
// latent, the DiT's text pass redone for the second text and its
// residency grown by the first -- give the same latent, BYTE for byte,
// with a different prompt between them proving the text really changed.
TEST(fibo2_stage, a_second_edit_reproduces_the_first)
{
  const Golden G = load_("VPIPE_FIBO2_REAL_GOLDEN");
  if (!G.ok || env_("VPIPE_FIBO2_E2E").empty()) { return; }
  // Sized from the reference, as the shipped graphs run: the plan books
  // a 1024^2 edit's activations, and each generation must revise them to
  // its own -- here a 512^2 edit, the last prompt the golden's (its text
  // rows known), so the ledger is that figure and not the default.
  std::size_t denoise = 0;
  const auto got =
      run_edits_(G, {G.prompt, "make the background bright blue",
                     G.prompt}, true, &denoise);
  ASSERT_TRUE(got.size() == 3);
  if (got.size() != 3) { return; }
  genai::MetalFibo2Transformer::Config cfg;
  ASSERT_TRUE(genai::MetalFibo2Transformer::Config::read_dims(
      (fs::path(G.root) / "transformer").string(), &cfg));
  const int grid = (G.size / 16) * (G.size / 16);
  Session probe;
  auto* pmc = probe.metal_compute();
  const std::size_t want =
      genai::MetalFibo2Transformer::forward_scratch_bytes(
          cfg, 2 * grid, G.text_len, false,
          pmc != nullptr && pmc->supports_matrix_cores());
  std::printf("[fibo2_stage] the denoise phase holds %.1f MB of scratch "
              "(this edit's activations %.1f MB)\n", denoise / 1048576.0,
              want / 1048576.0);
  EXPECT_TRUE(denoise >= want && denoise < want + (std::size_t(64) << 20));
  double worst = 0.0;
  for (std::size_t i = 0; i < got[0].size(); ++i) {
    worst = std::max(worst, (double)std::fabs(got[0][i] - got[2][i]));
  }
  const double moved =
      rel_l2_(got[1].data(), got[0].data(), got[0].size());
  std::printf("[fibo2_stage] third edit vs first: worst |diff| %g; the "
              "second prompt moved the latent %.4f\n", worst, moved);
  EXPECT_TRUE(worst == 0.0);
  EXPECT_TRUE(moved > 0.05);
}
