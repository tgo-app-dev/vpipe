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
#include "stages/fibo2-model-config-stage.h"
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
  // > 0: a GUIDED step -- the prompt's forward, then a negative of this
  // many rows (the prompt's rows tiled) in the second text slot, both
  // under the one measurement.
  int neg_rows = 0;
  std::size_t grew = 0;
  std::size_t booked = 0;
  std::size_t held = 0;
  std::size_t floor = 0;
  int gist = 0, neg_gist = 0;
};

// `sm`: measure one forward (or a guided step's two) instead of walking
// the steps -- preloaded for the activations (a streamed one would count
// its slots and promoted blocks too), streamed for the floor.
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
  // A guided step's negative, made before the baseline (it is the test's
  // own buffer): each bundle's rows tiled from the prompt's -- what is
  // allocated depends on the row count alone.
  const int TN = (sm != nullptr && !stream) ? sm->neg_rows : 0;
  metal_compute::SharedBuffer nbuf;
  if (TN > 0) {
    const std::size_t row = 7680, tn = (std::size_t)TN;
    const std::size_t tl = (std::size_t)G.text_len;
    nbuf = mc->make_shared_buffer(6 * tn * row * 2);
    FB_CHECK_OR_RETURN(c, !nbuf.empty());
    const auto* src = static_cast<const std::uint16_t*>(tb.contents());
    auto* dst = static_cast<std::uint16_t*>(nbuf.contents());
    for (std::size_t b = 0; b < 6; ++b) {
      for (std::size_t t = 0; t < tn; ++t) {
        std::memcpy(dst + (b * tn + t) * row, src + (b * tl + t % tl) * row,
                    row * 2);
      }
    }
  }
  // Before the text: the text pass's products are the forward's to book.
  const std::size_t alloc0 =
      sm != nullptr ? mc->memory_budget_settled().allocated : 0;
  FB_CHECK_OR_RETURN(c, m->set_text(tb, G.text_len, &err));
  if (TN > 0) { FB_CHECK_OR_RETURN(c, m->set_text(nbuf, TN, &err, 1)); }
  // Each picture's latent, in vae-encode's [128, h, w], is its patch rows.
  std::vector<std::pair<int, int>> grids;
  std::vector<std::vector<float>> ctx;
  for (std::size_t k = 0; k < G.images.size(); ++k) {
    const int ch = G.images[k].second / 16, cw = G.images[k].first / 16;
    ctx.push_back(read_f32_(G.dir / ("ctx" + std::to_string(k) + ".f32")));
    FB_CHECK_OR_RETURN(c, ctx.back().size() == (std::size_t)PD * ch * cw);
    grids.emplace_back(ch, cw);
  }
  genai::fibo2::Layout lay, lay_neg;
  FB_CHECK_OR_RETURN(
      c, genai::fibo2::build_layout(gh, gh, grids, m->gist_count(),
                                    cfg.max_gist, &lay, &err));
  if (TN > 0) {
    FB_CHECK_OR_RETURN(
        c, genai::fibo2::build_layout(gh, gh, grids, m->gist_count(1),
                                      cfg.max_gist, &lay_neg, &err));
  }
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
    // The negative's pass, as generate_fibo2_ runs it: the same latent
    // rows on its own layout, reading the second slot.
    bool neg_ok = true;
    if (TN > 0 && !v.empty()) {
      genai::MetalFibo2Transformer::Request rn = req;
      rn.layout = &lay_neg;
      rn.text_slot = 1;
      std::string en;
      neg_ok = !m->forward(rn, &en).empty();
      if (!neg_ok) {
        std::printf("[fibo2_dit] the negative's pass failed: %s\n",
                    en.c_str());
      }
    }
    if (sm != nullptr) {
      done.store(true);
      sampler.join();
    }
    FB_CHECK_OR_RETURN(c, neg_ok);
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
          mc->supports_matrix_cores(), TN > 0 ? TN : -1);
      sm->gist = m->gist_count();
      sm->neg_gist = TN > 0 ? m->gist_count(1) : 0;
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

// A GUIDED step holds two texts and runs two joint lengths: the plan
// books the negative's gist and injection keys and values beside the
// prompt's, and the planes at the longer sequence. A negative SHORTER
// than the prompt (the null caption's 46 rows: a smaller gist) and one
// LONGER (1100 rows: a larger gist, a longer text pass), each against
// one measured prompt-then-negative step.
TEST(fibo2_dit, a_guided_step_allocates_what_the_plan_books)
{
  for (int tn : {46, 1100}) {
    ScratchMeasure sm;
    sm.neg_rows = tn;
    const Checks c = steps_test_("VPIPE_FIBO2_REAL_GOLDEN", &sm);
    for (const auto& [ok, what] : c.items) { EXPECT_TRUE(ok); }
    if (!c.ran || sm.booked == 0) { return; }
    std::printf("[fibo2_dit] guided, %d negative rows (gist %d; the "
                "prompt's %d): peak allocation %.1f MB; the plan books "
                "%.1f MB (%.3fx)\n", tn, sm.neg_gist, sm.gist,
                sm.grew / 1048576.0, sm.booked / 1048576.0,
                sm.grew > 0 ? (double)sm.booked / (double)sm.grew : 0.0);
    // Each arm on its side of the prompt's gist, or it tests one case.
    EXPECT_TRUE(tn < 100 ? sm.neg_gist < sm.gist : sm.neg_gist > sm.gist);
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

// ---- Fibo 2.0 Dev: the guided sampler --------------------------------
//
// tools/dump_fibo2_golden.py `guided`: the SMALL random model in a repo
// root generate-image loads, and the reference pipeline's own __call__
// (output_type "latent") at guidance 5 against a negative of another
// length -- its own gist count -- for a text-to-image and an edit, in
// fp32 and in bf16 (the bar), and the same sampler unguided. Env:
// VPIPE_FIBO2_GUIDED_GOLDEN. No checkpoint needed.

namespace {

// Emits the negative on oport1, THEN the prompt on oport0, as the
// conditioner does (generate-image reads a positive's negative only if it
// is already queued); each [bundles, rows, cap] bf16.
class SourceCond : public TypedStage<SourceCond> {
public:
  static constexpr const char* kTypeName = "ut-source-cond-fibo2";
  SourceCond(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(2); }
  std::vector<float> pos, neg;
  int pos_rows = 0, neg_rows = 0, cap = 0, bundles = 6;
  bool done = false;
  std::unique_ptr<TensorBeatPayload>
  beat_(const std::vector<float>& v, int rows, int cap) const
  {
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::Bf16;
    tb->shape = {bundles, rows, cap};
    tb->resize_contiguous(v.size());
    auto* u = reinterpret_cast<std::uint16_t*>((void*)tb->as_u8());
    for (std::size_t i = 0; i < v.size(); ++i) { u[i] = f2bf_(v[i]); }
    return tb;
  }
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      if (neg_rows > 0) { co_await ctx.write(1, beat_(neg, neg_rows, cap)); }
      co_await ctx.write(0, beat_(pos, pos_rows, cap));
    }
    ctx.signal_done();
    co_return;
  }
};

// One f32 [C, H, W] beat (a vae-encode reference latent), then ends.
class SourceLatent : public TypedStage<SourceLatent> {
public:
  static constexpr const char* kTypeName = "ut-source-latent-fibo2";
  SourceLatent(const SessionContextIntf* s, std::string id,
               std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<float> chw;
  int C = 0, H = 0, W = 0;
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::F32;
      tb->shape = {C, H, W};
      tb->resize_contiguous(chw.size());
      std::memcpy(tb->as_f32(), chw.data(), chw.size() * sizeof(float));
      co_await ctx.write(0, std::move(tb));
    }
    ctx.signal_done();
    co_return;
  }
};

struct GuidedRun {
  double guidance = 1.0;
  bool via_model_config = true;   // fibo2-model-config, else the stage key
  bool wire_negative = true;
};

// One scenario of the guided golden through generate-image; empty on a
// failure.
std::vector<float>
run_guided_(const fs::path& root, const FlexData& sc, int steps,
            const GuidedRun& r)
{
  Session sess;
  if (sess.metal_compute() == nullptr) { return {}; }
  const fs::path d = root / sc.as_object().at("name").as_string();
  const int H = (int)sc.as_object().at("height").as_int(0);
  const int W = (int)sc.as_object().at("width").as_int(0);
  auto pl = std::make_unique<Pipeline>("fibo2-guided", &sess);
  auto cu = std::make_unique<SourceCond>(&sess, "cond", std::vector<InEdge>{},
                                         FlexData::make_object());
  // The golden's [1, bundles, rows, cap].
  const FlexData es = sc.as_object().at("files").as_object().at("enc");
  cu->bundles = (int)es.as_array().at(1).as_int(0);
  cu->cap = (int)es.as_array().at(3).as_int(0);
  cu->pos = read_f32_(d / "enc.f32");
  cu->pos_rows = (int)sc.as_object().at("text_len").as_int(0);
  if (r.wire_negative) {
    cu->neg = read_f32_(d / "neg.f32");
    cu->neg_rows = (int)sc.as_object().at("neg_len").as_int(0);
  }
  const std::size_t row = (std::size_t)cu->bundles * cu->cap;
  if (row == 0 || cu->pos.size() != row * cu->pos_rows ||
      cu->neg.size() != row * cu->neg_rows) {
    std::printf("[fibo2_stage] the golden's conditioning is not [%d, rows, "
                "%d]\n", cu->bundles, cu->cap);
    return {};
  }
  Stage* cond = pl->insert_stage(std::move(cu));
  std::vector<InEdge> gin{{cond, 0}, {cond, 1}, InEdge{nullptr, 0},
                          InEdge{nullptr, 0}, InEdge{nullptr, 0}};
  const FlexData ctx = sc.as_object().at("context");
  for (std::size_t k = 0; k < ctx.as_array().size(); ++k) {
    const FlexData hw = ctx.as_array().at(k);
    auto lu = std::make_unique<SourceLatent>(
        &sess, "lat" + std::to_string(k), std::vector<InEdge>{},
        FlexData::make_object());
    lu->C = 128;
    lu->H = (int)hw.as_array().at(0).as_int(0) / 2;
    lu->W = (int)hw.as_array().at(1).as_int(0) / 2;
    lu->chw = read_f32_(d / ("ctx" + std::to_string(k) + ".f32"));
    if (lu->chw.size() != (std::size_t)128 * lu->H * lu->W) { return {}; }
    gin.push_back({pl->insert_stage(std::move(lu)), 0});
  }
  while (gin.size() < 7) { gin.push_back(InEdge{nullptr, 0}); }
  FlexData gc = FlexData::make_object();
  if (r.via_model_config) {
    FlexData mc = FlexData::make_object();
    mc.as_object().insert("guidance_scale",
                          FlexData::make_real(r.guidance));
    gin.push_back({pl->insert_stage(std::make_unique<Fibo2ModelConfigStage>(
                       &sess, "fibo2cfg", std::vector<InEdge>{},
                       std::move(mc))),
                   0});
  } else {
    gc.as_object().insert("guidance_scale",
                          FlexData::make_real(r.guidance));
  }
  gc.as_object().insert("hf_dir", FlexData::make_string(root.string()));
  gc.as_object().insert(
      "seed", FlexData::make_int(sc.as_object().at("seed").as_int(0)));
  gc.as_object().insert("steps", FlexData::make_int(steps));
  gc.as_object().insert("width", FlexData::make_int(W));
  gc.as_object().insert("height", FlexData::make_int(H));
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "gen", std::move(gin),
                                           std::move(gc))));
  if (!gen->config_error().empty()) {
    std::printf("[fibo2_stage] generate-image: %s\n",
                gen->config_error().c_str());
    return {};
  }
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) {
    std::printf("[fibo2_stage] the graph did not launch\n");
    return {};
  }
  rt.wait_idle();
  rt.stop();
  if (sink->captured.size() != 1) {
    std::printf("[fibo2_stage] %zu latents out\n", sink->captured.size());
    return {};
  }
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  if (tb == nullptr || tb->dtype != TensorBeat::DType::F32) { return {}; }
  return std::vector<float>(tb->as_f32(), tb->as_f32() + tb->element_count());
}

}  // namespace

// The guided sampler against the reference's: the negative's own
// forward a step on its own gist count, uncond + g (cond - uncond), the
// schedule and the noise -- for a text-to-image (guidance from
// fibo2-model-config) and an edit (generate-image's own guidance_scale).
// Held to 2x the reference's own bf16 distance over 6 steps at g 5, and
// to a distance from the UNGUIDED reference that guidance must open.
TEST(fibo2_stage, guided_sampler_matches_reference)
{
  const std::string gd = env_("VPIPE_FIBO2_GUIDED_GOLDEN");
  if (gd.empty()) { return; }
  FlexData man;
  ASSERT_TRUE(read_json_(fs::path(gd) / "golden.json", &man));
  if (!man.is_object()) { return; }
  const int steps = (int)man.as_object().at("steps").as_int(0);
  const double g = man.as_object().at("guidance").as_real(0.0);
  const FlexData scs = man.as_object().at("scenarios");
  ASSERT_TRUE(scs.as_array().size() == 2);
  for (std::size_t i = 0; i < scs.as_array().size(); ++i) {
    const FlexData sc = scs.as_array().at(i);
    const std::string name(sc.as_object().at("name").as_string());
    const fs::path d = fs::path(gd) / name;
    const int H8 = (int)sc.as_object().at("height").as_int(0) / 8;
    const int W8 = (int)sc.as_object().at("width").as_int(0) / 8;
    GuidedRun r;
    r.guidance = g;
    r.via_model_config = (i == 0);
    const std::vector<float> got = run_guided_(fs::path(gd), sc, steps, r);
    const std::vector<float> want =
        patchify_(read_f32_(d / "final.f32"), H8, W8);
    const std::vector<float> ref16 =
        patchify_(read_f32_(d / "final_bf16.f32"), H8, W8);
    const std::vector<float> plain =
        patchify_(read_f32_(d / "final_unguided.f32"), H8, W8);
    ASSERT_TRUE(got.size() == want.size() && !want.empty());
    if (got.size() != want.size() || want.empty()) { continue; }
    const double rel = rel_l2_(got.data(), want.data(), want.size());
    const double bar = rel_l2_(ref16.data(), want.data(), want.size());
    const double away = rel_l2_(got.data(), plain.data(), plain.size());
    std::printf("[fibo2_stage] guided %s (%d steps, g %g, guidance from %s): "
                "latent rel-L2 %.5f (the reference's own bf16 %.5f); the "
                "unguided reference %.4f away\n", name.c_str(), steps, g,
                r.via_model_config ? "fibo2-model-config" : "the stage key",
                rel, bar, away);
    EXPECT_TRUE(rel <= 2.0 * bar);
    EXPECT_TRUE(away > 10.0 * bar);
  }
}

// Guidance 1 runs ONE forward a step, whatever is wired: a negative on
// iport1 changes nothing (Edit Turbo's recipe), and the result is the
// reference's unguided sampler.
TEST(fibo2_stage, guidance_one_ignores_a_wired_negative)
{
  const std::string gd = env_("VPIPE_FIBO2_GUIDED_GOLDEN");
  if (gd.empty()) { return; }
  FlexData man;
  ASSERT_TRUE(read_json_(fs::path(gd) / "golden.json", &man));
  if (!man.is_object()) { return; }
  const int steps = (int)man.as_object().at("steps").as_int(0);
  const FlexData sc = man.as_object().at("scenarios").as_array().at(0);
  const fs::path d = fs::path(gd) / sc.as_object().at("name").as_string();
  const int H8 = (int)sc.as_object().at("height").as_int(0) / 8;
  const int W8 = (int)sc.as_object().at("width").as_int(0) / 8;
  GuidedRun with, without;
  with.guidance = without.guidance = 1.0;
  without.wire_negative = false;
  const std::vector<float> a = run_guided_(fs::path(gd), sc, steps, with);
  const std::vector<float> b = run_guided_(fs::path(gd), sc, steps, without);
  const std::vector<float> plain =
      patchify_(read_f32_(d / "final_unguided.f32"), H8, W8);
  ASSERT_TRUE(a.size() == plain.size() && b.size() == plain.size());
  if (a.size() != plain.size() || b.size() != plain.size()) { return; }
  const double ab = rel_l2_(a.data(), b.data(), b.size());
  const double rel = rel_l2_(a.data(), plain.data(), plain.size());
  std::printf("[fibo2_stage] guidance 1: with a negative wired vs without "
              "%g; vs the reference's unguided sampler %.5f\n", ab, rel);
  EXPECT_TRUE(ab == 0.0);
  EXPECT_TRUE(rel < 0.03);
}

// THE LAUNCH PLAN BOOKS A GUIDED NEGATIVE. declare_resources runs before
// any beat, so the guidance it can see is the model-config CONSTANT
// (apply_constant) or generate-image's own key: guided, the DiT's
// activation claim holds a negative's text beside the prompt's (at the
// prompt's planning length); unguided -- Edit Turbo's graphs -- it does
// not; and a relaunch without the source (reset_run_state, which runs
// after the plan) plans without it. The live preview the constant names
// is booked the same way (it had been read from the runtime latch,
// empty at a first launch's plan). Loads nothing. Env:
// VPIPE_FIBO2_GUIDED_GOLDEN (its small repo root); the preview arm,
// VPIPE_FIBO2_PREVIEW_VAE (a TAE directory, e.g. madebyollin/taef2).
TEST(fibo2_stage, the_launch_plan_books_a_guided_negative)
{
  const std::string gd = env_("VPIPE_FIBO2_GUIDED_GOLDEN");
  if (gd.empty()) { return; }
  Session sess;
  const fs::path root(gd);
  genai::MetalFibo2Transformer::Config cfg;
  ASSERT_TRUE(genai::MetalFibo2Transformer::Config::read_dims(
      (root / "transformer").string(), &cfg));
  auto* mc = sess.metal_compute();
  const bool mx = mc != nullptr && mc->supports_matrix_cores();
  // The claim's geometry: a 1024^2 edit (no width or height) beside a
  // 1024-row prompt.
  const int rows = 2 * (1024 / 16) * (1024 / 16);
  const std::size_t one = genai::MetalFibo2Transformer::forward_scratch_bytes(
      cfg, rows, 1024, false, mx);
  const std::size_t two = genai::MetalFibo2Transformer::forward_scratch_bytes(
      cfg, rows, 1024, false, mx, 1024);
  ASSERT_TRUE(two > one);
  // A scratch claim's key is "label|bytes".
  auto booked = [](const GenerateImageStage& st) -> std::size_t {
    for (const ResourceClaim& c : st.declare_resources()) {
      if (c.key.rfind("fibo2-dit:", 0) == 0) {
        return (std::size_t)std::stoull(c.key.substr(c.key.rfind('|') + 1));
      }
    }
    return 0;
  };
  auto make = [&](double own) {
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(root.string()));
    if (own > 0.0) {
      c.as_object().insert("guidance_scale", FlexData::make_real(own));
    }
    return std::make_unique<GenerateImageStage>(
        &sess, "gen", std::vector<InEdge>{}, std::move(c));
  };
  auto model_cfg = [](double g) {
    FlexData m = FlexData::make_object();
    m.as_object().insert("model_family", FlexData::make_string("fibo2"));
    m.as_object().insert("guidance_scale", FlexData::make_real(g));
    return m;
  };
  constexpr unsigned kModelCfgPort = 7;   // generate-image's model_config
  {
    auto st = make(0.0);
    ASSERT_TRUE(st->config_error().empty());
    const std::size_t plain = booked(*st);
    st->apply_constant(kModelCfgPort, model_cfg(5.0));
    const std::size_t guided = booked(*st);
    st->reset_run_state();
    const std::size_t relaunched = booked(*st);
    std::printf("[fibo2_stage] launch plan: unguided %.1f MB, guided by "
                "the model config %.1f MB, relaunched without it %.1f MB\n",
                plain / 1048576.0, guided / 1048576.0,
                relaunched / 1048576.0);
    EXPECT_TRUE(plain == one);
    EXPECT_TRUE(guided == two);
    EXPECT_TRUE(relaunched == one);
  }
  {
    // The stage's own guidance books it; a model config saying 1 wins.
    auto st = make(5.0);
    EXPECT_TRUE(booked(*st) == two);
    st->apply_constant(kModelCfgPort, model_cfg(1.0));
    EXPECT_TRUE(booked(*st) == one);
  }
  const std::string tae = env_("VPIPE_FIBO2_PREVIEW_VAE");
  if (tae.empty()) { return; }
  {
    // A weights claim on the TAE's file, under its directory.
    auto previews = [&](const GenerateImageStage& st) {
      int n = 0;
      for (const ResourceClaim& c : st.declare_resources()) {
        if (c.kind == model_memory::kWeightsKind &&
            c.key.rfind(tae, 0) == 0) {
          ++n;
        }
      }
      return n;
    };
    auto st = make(0.0);
    const int before = previews(*st);
    FlexData m = model_cfg(1.0);
    m.as_object().insert("preview_vae", FlexData::make_string(tae));
    st->apply_constant(kModelCfgPort, m);
    const int with = previews(*st);
    st->reset_run_state();
    const int after = previews(*st);
    std::printf("[fibo2_stage] the preview's TAE claimed: before the "
                "constant %d, with it %d, relaunched without it %d\n",
                before, with, after);
    EXPECT_TRUE(before == 0);
    EXPECT_TRUE(with == 1);
    EXPECT_TRUE(after == 0);
  }
}

// THE NEGATIVE IS THE NULL CAPTION, AS WRITTEN, AND TEXT ALONE: the
// conditioner's oport1, wired, against the reference's
// encode_prompt(DEFAULT_NEGATIVE_PROMPT) -- uncleaned (cleaned, it would
// be "{}"), and shown none of the pictures the prompt edits. Env:
// VPIPE_FIBO2_TEST_MODEL_PATH (any Fibo 2.0 repo root) +
// VPIPE_FIBO2_NEG_GOLDEN (`negative`, from the same text encoder);
// VPIPE_FIBO2_REAL_GOLDEN adds an edit's picture, which must not reach
// the negative.
TEST(fibo2_cond, the_negative_is_the_null_caption_as_written)
{
  const std::string root = env_("VPIPE_FIBO2_TEST_MODEL_PATH");
  const std::string nd = env_("VPIPE_FIBO2_NEG_GOLDEN");
  if (root.empty() || nd.empty()) { return; }
  FlexData man;
  ASSERT_TRUE(read_json_(fs::path(nd) / "golden.json", &man));
  if (!man.is_object()) { return; }
  EXPECT_TRUE(man.as_object().at("negative").as_string() ==
              std::string(genai::fibo2::kNullCaption));
  const int T = (int)man.as_object().at("text_len").as_int(0);
  const double bar = man.as_object().at("ref_bf16_rel").as_real(0.0);
  const std::vector<float> want = read_f32_(fs::path(nd) / "neg.f32");
  ASSERT_TRUE(T > 0 && want.size() == (std::size_t)6 * T * 7680);
  if (want.size() != (std::size_t)6 * T * 7680) { return; }
  Golden G = load_("VPIPE_FIBO2_REAL_GOLDEN");
  for (const bool edit : {false, true}) {
    if (edit && !G.ok) { continue; }
    Session sess;
    if (sess.metal_compute() == nullptr) { return; }
    auto pl = std::make_unique<Pipeline>("fibo2-neg", &sess);
    auto srcu = std::make_unique<SourceText>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
    srcu->prompts = {edit ? G.prompt : std::string("a red bicycle")};
    Stage* src = pl->insert_stage(std::move(srcu));
    Golden R = G;
    R.root = root;
    const std::vector<Stage*> pics =
        edit ? pictures_(pl.get(), sess, G) : std::vector<Stage*>{};
    Stage* cond = conditioner_(pl.get(), sess, R, src, pics);
    auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
        std::make_unique<SinkCapture>(&sess, "sink",
                                      std::vector<InEdge>{{cond, 0}},
                                      FlexData::make_object())));
    auto* nsink = static_cast<SinkCapture*>(pl->insert_stage(
        std::make_unique<SinkCapture>(&sess, "nsink",
                                      std::vector<InEdge>{{cond, 1}},
                                      FlexData::make_object())));
    PipelineRuntime rt(pl.get(), &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    ASSERT_TRUE(sink->captured.size() == 1 && nsink->captured.size() == 1);
    if (nsink->captured.size() != 1) { continue; }
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(nsink->captured[0].get());
    ASSERT_TRUE(tb != nullptr && tb->shape.size() == 3);
    if (tb == nullptr || tb->shape.size() != 3) { continue; }
    std::printf("[fibo2_cond] %s: the negative [%lld, %lld, %lld] (the "
                "reference's [6, %d, 7680])\n", edit ? "edit" : "text",
                (long long)tb->shape[0], (long long)tb->shape[1],
                (long long)tb->shape[2], T);
    ASSERT_TRUE(tb->shape[0] == 6 && tb->shape[1] == T &&
                tb->shape[2] == 7680);
    if (tb->shape[1] != T) { continue; }
    std::vector<float> got(want.size());
    const auto* u =
        reinterpret_cast<const std::uint16_t*>((const void*)tb->as_u8());
    for (std::size_t k = 0; k < got.size(); ++k) { got[k] = bf2f_(u[k]); }
    const double rel = rel_l2_(got.data(), want.data(), want.size());
    std::printf("[fibo2_cond] %s: the negative's rel-L2 %.5f (the "
                "reference's own bf16 %.5f)\n", edit ? "edit" : "text", rel,
                bar);
    EXPECT_TRUE(rel <= 1.25 * bar);
  }
}
