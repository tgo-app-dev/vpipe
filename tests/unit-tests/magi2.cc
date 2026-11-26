// MAGI-2-preview (sand-ai/MAGI-2-preview) -- the preview DiT against
// tools/dump_magi2_golden.py's restatement of the release's
// inference/model/magi2_preview.py.
//
//   VPIPE_MAGI2_GOLDEN   a `small` golden (its model.safetensors and the
//                        t2v / i2v / text1 cases): a synthetic DiT of the
//                        release's structure (mm, MoE, MoE, mm)
//   VPIPE_MAGI2_REAL_GOLDEN + VPIPE_MAGI2_TEST_MODEL_PATH
//                        a `real` golden over the release's preview/ (a
//                        subset of its layers) and that directory

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "generative-models/magi2/magi2-text.h"
#include "generative-models/magi2/metal-magi2-dit.h"
#include "generative-models/magi2/metal-magi2-refiner.h"
#include "generative-models/magi2/magi2-refine.h"
#include "generative-models/magi2/metal-magi2-turbo.h"
#include "generative-models/magi2/magi2-audio.h"
#include "generative-models/magi2/magi2-schedule.h"
#include "generative-models/magi2/magi2-preview.h"
#include "generative-models/magi2/magi2-family.h"
#include "generative-models/magi2/magi2-accel.h"
#include "generative-models/magi2/magi2-image-encoder.h"
#include "generative-models/vae-host-ext.h"
#include "generative-models/video-host-ext.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/magi2/magi2-layout.h"
#include "generative-models/video-model-registry.h"
#include "generative-models/vae-model-registry.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "generative-models/context-manager.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tensor-parallel.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-plan.h"
#include "stages/generate-video-stage.h"
#include "stages/magi2-model-config-stage.h"
#include "stages/model-memory.h"
#include "stages/model-memory-host.h"

#include <algorithm>
#include <malloc/malloc.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <cmath>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
namespace fs = std::filesystem;
using vpipe::metal_compute::SharedBuffer;

namespace {

std::vector<float>
read_f32(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  std::vector<float> v;
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const std::size_t n = (std::size_t)in.tellg() / 4;
  in.seekg(0);
  v.resize(n);
  in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * 4));
  return v;
}

FlexData
read_json(const fs::path& p)
{
  std::ifstream in(p);
  return in ? FlexData::from_json(in) : FlexData();
}

double
rel(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1e30; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

struct Case {
  FlexData meta;
  std::vector<float> video, audio, text;
  std::vector<std::vector<float>> images, specials;
  std::vector<float> vv32, va32, vvbf, vabf;
};

bool
load_case(const fs::path& d, Case* c)
{
  c->meta = read_json(d / "meta.json");
  if (!c->meta.is_object()) { return false; }
  c->video = read_f32(d / "video.bin");
  c->audio = read_f32(d / "audio.bin");
  c->text = read_f32(d / "text.bin");
  c->vv32 = read_f32(d / "vel_video_fp32.bin");
  c->va32 = read_f32(d / "vel_audio_fp32.bin");
  c->vvbf = read_f32(d / "vel_video_bf16.bin");
  c->vabf = read_f32(d / "vel_audio_bf16.bin");
  const FlexData imgs = c->meta.as_object().at("images");
  auto ia = imgs.as_array();
  for (std::size_t k = 0; k < ia.size(); ++k) {
    c->images.push_back(read_f32(d / fmt("image{}.bin", k)()));
    c->specials.push_back(read_f32(d / fmt("special{}.bin", k)()));
  }
  return true;
}

genai::MetalMagi2Dit::Sample
sample_of(const Case& c, const genai::MetalMagi2Dit::Config& cfg)
{
  auto o = c.meta.as_object();
  genai::MetalMagi2Dit::Sample s;
  s.T = (int)o.at("T").as_int(0);
  s.H = (int)o.at("H").as_int(0);
  s.W = (int)o.at("W").as_int(0);
  s.video = c.video.data();
  s.audio_len = (int)o.at("audio").as_int(0);
  s.audio = c.audio.data();
  s.text_len = (int)o.at("text").as_int(0);
  s.text = c.text.data();
  s.t = (float)o.at("t").as_real(0.0);
  const FlexData imgs_fd = o.at("images");
  auto imgs = imgs_fd.as_array();
  for (std::size_t k = 0; k < imgs.size(); ++k) {
    genai::MetalMagi2Dit::Image im;
    im.latent = c.images[k].data();
    const FlexData one = imgs.at(k);
    auto oo = one.as_object();
    im.h = (int)oo.at("H").as_int(0);
    im.w = (int)oo.at("W").as_int(0);
    im.special = c.specials[k].data();
    s.images.push_back(im);
  }
  (void)cfg;
  return s;
}

// Run every case of a golden dir through `dit`; each forward held to
// within `factor` x the reference's own bf16 distance from its fp32.
// False on a failed forward or a case over its bar.
// This port's row of each of the release's (its order: video, audio,
// text, then per image its special token and latent tokens).
std::vector<int>
release_rows(const genai::MetalMagi2Dit::Sample& s)
{
  const int nvg = s.T * s.H * s.W;
  int nimg = 0;
  for (const auto& im : s.images) { nimg += im.h * im.w; }
  const int n = nvg + nimg + s.audio_len + s.text_len + (int)s.images.size();
  std::vector<int> ours_of_ref(n);
  int rr = 0;
  for (int i = 0; i < nvg; ++i) { ours_of_ref[rr++] = i; }
  const int a0 = nvg + nimg;
  for (int i = 0; i < s.audio_len + s.text_len; ++i) {
    ours_of_ref[rr++] = a0 + i;
  }
  int img = nvg;
  const int sp0 = a0 + s.audio_len + s.text_len;
  for (std::size_t k = 0; k < s.images.size(); ++k) {
    ours_of_ref[rr++] = sp0 + (int)k;
    for (int i = 0; i < s.images[k].h * s.images[k].w; ++i) {
      ours_of_ref[rr++] = img++;
    }
  }
  return ours_of_ref;
}

// The golden's fp32 picks [heads, tokens, k] in the release's order, as
// [tokens, heads, k] in this port's.
void
force_golden_routes(genai::MetalMagi2Dit& dit, const fs::path& d,
                    const genai::MetalMagi2Dit::Sample& s)
{
  const FlexData meta = read_json(d / "meta.json");
  const FlexData ml = meta.as_object().at("moe_layers");
  auto la = ml.as_array();
  const auto& c = dit.config();
  const std::vector<int> ours = release_rows(s);
  const int n = (int)ours.size();
  for (std::size_t i = 0; i < la.size(); ++i) {
    const int li = (int)la.at(i).as_int(0);
    std::ifstream in(d / fmt("routes_{}.bin", li)(), std::ios::binary);
    std::vector<int> ref((std::size_t)c.moe_heads * n * c.top_k);
    in.read(reinterpret_cast<char*>(ref.data()),
            (std::streamsize)(ref.size() * 4));
    std::vector<int> picks(ref.size());
    for (int h = 0; h < c.moe_heads; ++h) {
      for (int r = 0; r < n; ++r) {
        for (int j = 0; j < c.top_k; ++j) {
          picks[((std::size_t)ours[r] * c.moe_heads + h) * c.top_k + j] =
              ref[((std::size_t)h * n + r) * c.top_k + j];
        }
      }
    }
    dit.force_routes(li, std::move(picks));
  }
}

bool
check_cases(genai::MetalMagi2Dit& dit, const fs::path& root,
            const std::vector<std::string>& names, double factor,
            bool forced = true)
{
  bool ok = true;
  for (const std::string& nm : names) {
    Case c;
    if (!load_case(root / nm, &c)) { continue; }
    const genai::MetalMagi2Dit::Sample s = sample_of(c, dit.config());
    if (forced) { force_golden_routes(dit, root / nm, s); }
    std::vector<float> vv, va;
    std::string err;
    if (!dit.forward(s, &vv, &va, &err)) {
      std::printf("[magi2] %s: %s\n", nm.c_str(), err.c_str());
      ok = false;
      continue;
    }
    const double dv = rel(vv, c.vv32), da = rel(va, c.va32);
    const double bv = rel(c.vvbf, c.vv32), ba = rel(c.vabf, c.va32);
    std::printf("[magi2] %s: video %.5f (ref bf16 %.5f)  audio %.5f "
                "(ref bf16 %.5f)\n", nm.c_str(), dv, bv, da, ba);
    ok = ok && dv <= factor * bv && da <= factor * ba;
  }
  return ok;
}

}  // namespace

// THE SEQUENCE: the host's layout -- coordinates and the rope table --
// against the release's data proxy (the golden's coords / rope_sin /
// rope_cos, in ITS token order: video, audio, text, then per image its
// special token and its latent tokens) mapped into this port's order.
// The device's peak allocation over its settled level while `fn` runs,
// sampled from a thread.
template <class Fn>
std::size_t
device_peak_of_(metal_compute::MetalCompute* mc, Fn&& fn)
{
  const std::size_t base = mc->memory_budget_settled().allocated;
  std::atomic<std::size_t> peak{base};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  fn();
  done.store(true);
  sampler.join();
  const std::size_t top =
      std::max(peak.load(), mc->memory_budget().allocated);
  return top > base ? top - base : 0;
}

// The host heap's peak growth over `fn` (malloc's bytes in use), sampled
// from a thread.
template <class Fn>
std::size_t
host_peak_of_(Fn&& fn)
{
  auto in_use = []() {
    malloc_statistics_t st{};
    malloc_zone_statistics(nullptr, &st);
    return (std::size_t)st.size_in_use;
  };
  const std::size_t base = in_use();
  std::atomic<std::size_t> peak{base};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t u = in_use();
      if (u > peak.load()) { peak.store(u); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  fn();
  done.store(true);
  sampler.join();
  return peak.load() > base ? peak.load() - base : 0;
}

TEST(magi2, the_sequence_is_the_release_s)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  const auto bands_buf =
      ws->read("pre_adapter.rope.bands", mc,
               genai::WeightSet::Residency::Copied);
  std::vector<float> bands(bands_buf.byte_size() / 4);
  std::memcpy(bands.data(), bands_buf.contents(), bands.size() * 4);
  for (const char* nm : {"t2v", "i2v", "text1"}) {
    Case c;
    ASSERT_TRUE(load_case(fs::path(g) / nm, &c));
    const auto s = sample_of(c, cfg);
    const auto L = genai::MetalMagi2Dit::layout(s, bands);
    const std::vector<float> rc = read_f32(fs::path(g) / nm / "coords.bin");
    const std::vector<float> rsn = read_f32(fs::path(g) / nm / "rope_sin.bin");
    ASSERT_TRUE((int)rc.size() == L.n * 9);
    // The release's row of each of ours.
    std::vector<int> ref_row(L.n);
    const int nvg = s.T * s.H * s.W;
    int r = 0, rr = 0;
    for (int i = 0; i < nvg; ++i) { ref_row[r++] = rr++; }
    const int ra = rr;                       // audio, then text
    rr += s.audio_len + s.text_len;
    std::vector<int> spec_ref;
    for (const auto& im : s.images) {
      spec_ref.push_back(rr++);
      for (int i = 0; i < im.h * im.w; ++i) { ref_row[r++] = rr++; }
    }
    for (int i = 0; i < s.audio_len + s.text_len; ++i) {
      ref_row[r++] = ra + i;
    }
    for (int sref : spec_ref) { ref_row[r++] = sref; }
    ASSERT_TRUE(r == L.n);
    const int R = (int)L.sin.size() / L.n;
    int bad = 0;
    for (int i = 0; i < L.n; ++i) {
      for (int j = 0; j < 9; ++j) {
        bad += L.coords[(std::size_t)i * 9 + j] !=
               rc[(std::size_t)ref_row[i] * 9 + j];
      }
      for (int j = 0; j < R; ++j) {
        bad += std::fabs(L.sin[(std::size_t)i * R + j] -
                         rsn[(std::size_t)ref_row[i] * R + j]) > 2e-6f;
      }
    }
    std::printf("[magi2] %s: %d tokens, %d coordinate / rope entries off\n",
                nm, L.n, bad);
    EXPECT_TRUE(bad == 0);
  }
}

// THE PREVIEW DiT on the small golden: every case within 1.5x the
// release's own bf16 distance from its fp32 (the same routes).
TEST(magi2, preview_dit_matches_reference)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  if (!err.empty()) { std::printf("[magi2] %s\n", err.c_str()); }
  auto dit = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  if (!dit) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dit != nullptr);
  EXPECT_TRUE(check_cases(*dit, g, {"t2v", "i2v", "text1"}, 1.5));
}

// THE RELEASE'S WEIGHTS, a subset of the layers (the golden's).
TEST(magi2, preview_dit_matches_reference_on_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_REAL_GOLDEN");
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (g == nullptr || m == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const FlexData meta = read_json(fs::path(g) / "t2v" / "meta.json");
  ASSERT_TRUE(meta.is_object());
  std::vector<int> layers;
  const FlexData layers_fd = meta.as_object().at("layers");
  auto la = layers_fd.as_array();
  for (std::size_t i = 0; i < la.size(); ++i) {
    layers.push_back((int)la.at(i).as_int(0));
  }
  auto ws = genai::WeightSet::open(m, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(
      genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err, layers));
  auto dit = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  if (!dit) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dit != nullptr);
  EXPECT_TRUE(check_cases(*dit, g, {"t2v", "i2v"}, 1.5));
}

// STREAMED EXPERTS are the held ones: every MoE layer's experts read per
// forward into the two slots (resident_moe 0) against all held -- the
// same velocity, byte for byte, on every case.
TEST(magi2, streamed_experts_are_the_held_ones)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  auto held = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  cfg.resident_moe = 0;
  auto streamed = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  ASSERT_TRUE(held != nullptr && streamed != nullptr);
  EXPECT_TRUE(streamed->moe_streamed() == 2 && streamed->moe_held() == 0);
  for (const char* nm : {"t2v", "i2v", "text1"}) {
    Case c;
    ASSERT_TRUE(load_case(fs::path(g) / nm, &c));
    const auto s = sample_of(c, cfg);
    std::vector<float> v1, a1, v2, a2;
    ASSERT_TRUE(held->forward(s, &v1, &a1, &err));
    ASSERT_TRUE(streamed->forward(s, &v2, &a2, &err));
    const bool same = v1 == v2 && a1 == a2;
    std::printf("[magi2] %s: streamed %s held\n", nm,
                same ? "==" : "!=");
    EXPECT_TRUE(same);
  }
}

// THE RETAINED EXPERTS ARE ADMITTED THROUGH THE WIRED POOL. With the
// session's pool sized for one MoE layer's experts beyond what is wired
// already, the load holds that one layer and streams the rest -- however
// many the count asks for -- and the velocity is the all-held one to the
// bit; dropping the model gives the pool its bytes back.
TEST(magi2, held_experts_stop_at_the_wired_pool)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto* mgr = sess.services()->generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  const std::size_t before = mgr->wired_pool_used();
  auto all = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  ASSERT_TRUE(all != nullptr);
  const std::size_t ex = all->expert_layer_bytes();
  const std::size_t all_wired = mgr->wired_pool_used() - before;
  const int moe = all->moe_held();
  Case c;
  ASSERT_TRUE(load_case(fs::path(g) / "t2v", &c));
  const auto smp = sample_of(c, cfg);
  std::vector<float> v1, a1;
  ASSERT_TRUE(all->forward(smp, &v1, &a1, &err));
  all.reset();
  const std::size_t after_all = mgr->wired_pool_used();
  // Room for one layer and a half: the second does not fit.
  const std::size_t was = mgr->wired_pool_bytes();
  mgr->set_wired_pool_bytes(mgr->wired_pool_used() + ex + ex / 2);
  auto one = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  ASSERT_TRUE(one != nullptr);
  std::vector<float> v2, a2;
  ASSERT_TRUE(one->forward(smp, &v2, &a2, &err));
  const int held1 = one->moe_held(), streamed1 = one->moe_streamed();
  one.reset();
  const std::size_t after_one = mgr->wired_pool_used();
  mgr->set_wired_pool_bytes(was);
  const bool same = v1 == v2 && a1 == a2;
  std::printf("[magi2] the wired pool: %d MoE layers held (%zu B wired, a "
              "layer %zu B); with room for one, %d held and %d streamed, "
              "the velocity %s; the pool back to %zu / %zu B of %zu\n", moe,
              all_wired, ex, held1, streamed1, same ? "the same" : "MOVED",
              after_all, after_one, before);
  EXPECT_TRUE(moe >= 2 && all_wired >= (std::size_t)moe * ex);
  EXPECT_TRUE(held1 == 1 && streamed1 == moe - 1 && same);
  EXPECT_TRUE(after_all == before && after_one == before);
}

// THE RETAINED EXPERTS' ROOM: the run's widest phase beside them, less
// this DiT's own floor (the plan counts it in every phase), never under
// the request's own forward -- and NOT the DiT's declared whole, which is
// what the first version subtracted (it held no expert on any box).
TEST(magi2, the_retained_experts_room_leaves_the_dit_out)
{
  genai::magi2::DitPlan p;
  for (std::size_t& t : p.trunk_at) { t = (std::size_t)10 << 30; }
  p.expert_layer = (std::size_t)6 << 30;
  p.moe_layers = 36;
  const std::size_t G = (std::size_t)1 << 30;
  const std::size_t floor = genai::magi2::dit_floor_bytes(p);
  const std::size_t ram = 128 * G;
  const std::vector<std::pair<std::string, std::size_t>> peaks = {
      {"condition", floor + 4 * G},
      {"denoise", floor + 30 * G},
      {"decode", floor + 8 * G}};
  const std::size_t r1 = genai::magi2::retention_room(p, -1, ram, peaks,
                                                      25 * G);
  const std::size_t r2 = genai::magi2::retention_room(p, -1, ram, {}, 25 * G);
  const std::size_t r3 = genai::magi2::retention_room(p, -1, ram, peaks,
                                                      40 * G);
  // The denoise phase, its scratch, refiner and the rest: 30 GB beside.
  const std::size_t want1 = ram - model_memory::kStreamHeadroom - 30 * G -
                            p.trunk();
  std::printf("[magi2] retained room: %.1f GB beside a 30 GB phase (%d "
              "layers), %.1f GB with no plan (%d), %.1f GB under a 40 GB "
              "request (%d)\n", r1 / (double)G,
              genai::magi2::resident_moe_for(p, r1), r2 / (double)G,
              genai::magi2::resident_moe_for(p, r2), r3 / (double)G,
              genai::magi2::resident_moe_for(p, r3));
  EXPECT_TRUE(r1 == want1);
  EXPECT_TRUE(r2 == ram - model_memory::kStreamHeadroom - 25 * G -
                        p.trunk());
  EXPECT_TRUE(r3 == ram - model_memory::kStreamHeadroom - 40 * G -
                        p.trunk());
  EXPECT_TRUE(genai::magi2::resident_moe_for(p, r1) ==
              (int)((r1 - 2 * p.expert_layer) / p.expert_layer));
  // A box that cannot hold a layer beside the slots holds none.
  EXPECT_TRUE(genai::magi2::retention_room(p, -1, 40 * G, peaks, 0) == 0);
}

// A BATCH IS ITS SAMPLES. A CFG step runs its two passes as one batch,
// layer by layer, so a streamed layer's experts are read once for both;
// each sample keeps its own residual stream and rope table. The batch's
// velocities must be each sample's alone, to the bit -- held and
// streamed, three samples of different lengths, one with an image.
TEST(magi2, a_batch_is_its_samples)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  const char* names[3] = {"t2v", "i2v", "text1"};
  Case cs[3];
  std::vector<genai::MetalMagi2Dit::Sample> smp;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(load_case(fs::path(g) / names[i], &cs[i]));
    smp.push_back(sample_of(cs[i], cfg));
  }
  for (int resident : {-1, 0}) {
    cfg.resident_moe = resident;
    auto m = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
    ASSERT_TRUE(m != nullptr);
    std::vector<std::vector<float>> bv, ba;
    ASSERT_TRUE(m->forward(
        std::vector<const genai::MetalMagi2Dit::Sample*>{&smp[0], &smp[1],
                                                         &smp[2]},
        &bv, &ba, &err));
    int same = 0;
    for (int i = 0; i < 3; ++i) {
      std::vector<float> v, a;
      ASSERT_TRUE(m->forward(smp[(std::size_t)i], &v, &a, &err));
      same += (v == bv[(std::size_t)i] && a == ba[(std::size_t)i]) ? 1 : 0;
    }
    std::printf("[magi2] batch of 3, experts %s: %d of 3 samples "
                "identical alone\n", resident == 0 ? "streamed" : "held",
                same);
    EXPECT_TRUE(same == 3);
  }
}

// FREE ROUTING. With no picks given, the share of (token, head) rows whose
// six experts are fp32's -- held to the release's own bf16 run's share
// less 3 points (its routes split at the same near ties). False on a
// failure; `model` "" reads the weights from the golden dir.
bool
free_routes_agree(const char* g, const char* model,
                  const std::vector<std::string>& names)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return true; }
  const FlexData meta0 = read_json(fs::path(g) / names[0] / "meta.json");
  std::vector<int> layers;
  if (model[0] != 0) {
    const FlexData lf = meta0.as_object().at("layers");
    auto la = lf.as_array();
    for (std::size_t i = 0; i < la.size(); ++i) {
      layers.push_back((int)la.at(i).as_int(0));
    }
  }
  auto ws = genai::WeightSet::open(model[0] != 0 ? model : g, nullptr);
  if (ws == nullptr) { return false; }
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  if (!genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err, layers)) {
    return false;
  }
  auto dit = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  if (dit == nullptr) { return false; }
  dit->keep_routes(true);
  bool ok = true;
  for (const std::string& nm : names) {
    Case c;
    if (!load_case(fs::path(g) / nm, &c)) { return false; }
    const auto s = sample_of(c, cfg);
    std::vector<float> vv, va;
    if (!dit->forward(s, &vv, &va, &err)) { return false; }
    const std::vector<int> ours = release_rows(s);
    const int n = (int)ours.size();
    const FlexData ml = c.meta.as_object().at("moe_layers");
    auto la = ml.as_array();
    double worst = 1.0;
    for (std::size_t i = 0; i < la.size(); ++i) {
      const int li = (int)la.at(i).as_int(0);
      const std::vector<int> mine = dit->routes(li);
      std::ifstream in(fs::path(g) / nm / fmt("routes_{}.bin", li)(),
                       std::ios::binary);
      std::vector<int> ref((std::size_t)cfg.moe_heads * n * cfg.top_k);
      in.read(reinterpret_cast<char*>(ref.data()),
              (std::streamsize)(ref.size() * 4));
      if (mine.size() != ref.size()) { return false; }
      int same = 0;
      for (int h = 0; h < cfg.moe_heads; ++h) {
        for (int r = 0; r < n; ++r) {
          std::vector<int> a, b;
          for (int j = 0; j < cfg.top_k; ++j) {
            a.push_back(ref[((std::size_t)h * n + r) * cfg.top_k + j]);
            b.push_back(mine[((std::size_t)ours[r] * cfg.moe_heads + h) *
                             cfg.top_k + j]);
          }
          std::sort(a.begin(), a.end());
          std::sort(b.begin(), b.end());
          same += a == b;
        }
      }
      worst = std::min(worst, (double)same / (cfg.moe_heads * n));
    }
    const double base =
        c.meta.as_object().at("bf16_route_agree").as_real(1.0);
    std::printf("[magi2] %s: free picks as fp32's %.4f (ref bf16 %.4f)\n",
                nm.c_str(), worst, base);
    ok = ok && worst >= base - 0.03;
  }
  return ok;
}

TEST(magi2, free_routes_agree_with_the_reference)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  EXPECT_TRUE(free_routes_agree(g, "", {"t2v", "i2v", "text1"}));
}

TEST(magi2, free_routes_agree_with_the_reference_on_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_REAL_GOLDEN");
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (g == nullptr || m == nullptr) { return; }
  EXPECT_TRUE(free_routes_agree(g, m, {"t2v", "i2v"}));
}

// THE TEXT: the release's prompt handling and tokenizer, against
// tools/dump_magi2_golden.py `text` (VPIPE_MAGI2_TEXT_GOLDEN, with
// VPIPE_MAGI2_TEXT_ENCODER the release's text_encoder/): the JSON caption
// as compact markdown, the figure reference, every prompt's ids (CJK
// characters isolated) and the "<Figure 1>" span.
TEST(magi2, prompts_tokenize_as_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_TEXT_GOLDEN");
  const char* te = std::getenv("VPIPE_MAGI2_TEXT_ENCODER");
  if (g == nullptr || te == nullptr) { return; }
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_json(
      (fs::path(te) / "tokenizer.json").string(), &sess);
  ASSERT_TRUE(tok != nullptr);
  const FlexData meta = read_json(fs::path(g) / "meta.json");
  ASSERT_TRUE(meta.is_object());
  auto mo = meta.as_object();
  int bad = 0;
  for (const char* nm : {"fox", "cjk", "neg", "figure", "json", "json_fig"}) {
    if (!mo.contains(nm)) { continue; }
    const FlexData e = mo.at(nm);
    auto eo = e.as_object();
    const std::string prompt(eo.at("prompt").as_string(""));
    const std::string want_text(eo.at("text").as_string(""));
    const std::string text = genai::magi2::normalize_prompt(prompt);
    const bool text_ok = text == want_text;
    const FlexData idf = eo.at("ids");
    auto ia = idf.as_array();
    std::vector<std::int32_t> want;
    for (std::size_t i = 0; i < ia.size(); ++i) {
      want.push_back((std::int32_t)ia.at(i).as_int(0));
    }
    const std::vector<std::int32_t> ids = genai::magi2::encode(*tok, text);
    const FlexData spf = eo.at("span");
    auto sa = spf.as_array();
    std::vector<int> want_span;
    for (std::size_t i = 0; i < sa.size(); ++i) {
      want_span.push_back((int)sa.at(i).as_int(0));
    }
    const std::vector<int> span = genai::magi2::target_span(
        *tok, ids, text, genai::magi2::kFigureToken);
    int first = -1;
    for (std::size_t i = 0; i < std::min(ids.size(), want.size()); ++i) {
      if (ids[i] != want[i]) { first = (int)i; break; }
    }
    std::printf("[magi2] %s: text %s, %zu ids (ref %zu), first off %d, "
                "span %s\n", nm, text_ok ? "==" : "!=", ids.size(),
                want.size(), first, span == want_span ? "==" : "!=");
    if (first >= 0 && std::getenv("VPIPE_MAGI2_TOKDBG") != nullptr) {
      for (int i = std::max(0, first - 2);
           i < std::min<int>(first + 4, (int)ids.size()); ++i) {
        const std::int32_t a = ids[(std::size_t)i];
        const std::int32_t b = i < (int)want.size() ? want[(std::size_t)i] : -1;
        std::printf("    %d: ours %d '%s'  ref %d '%s'\n", i, a,
                    tok->decode(std::vector<std::int32_t>{a}).c_str(), b,
                    tok->decode(std::vector<std::int32_t>{b}).c_str());
      }
    }
    bad += !text_ok || ids != want || span != want_span;
  }
  // The figure reference itself: a plain prompt and a JSON caption.
  {
    const FlexData e = mo.at("figure");
    auto eo = e.as_object();
    const std::string p =
        genai::magi2::with_figure_reference("a cat on a sofa, the camera pans");
    bad += p != std::string(eo.at("prompt").as_string(""));
  }
  if (mo.contains("json") && mo.contains("json_fig")) {
    const FlexData j = mo.at("json");
    const FlexData jf = mo.at("json_fig");
    const std::string p = genai::magi2::with_figure_reference(
        std::string(j.as_object().at("prompt").as_string("")));
    const bool same =
        p == std::string(jf.as_object().at("prompt").as_string(""));
    std::printf("[magi2] json figure reference re-serialized %s\n",
                same ? "==" : "!=");
    bad += !same;
  }
  EXPECT_TRUE(bad == 0);
}

// THE CONDITION: Qwen3.5-27B through vpipe's streamed hybrid, tapped at the
// golden's layer (its first N layers' hidden_states[-3]); every prompt's
// rows within 1.5x the reference's own bf16 distance from its fp32.
TEST(magi2, text_condition_matches_reference)
{
  const char* g = std::getenv("VPIPE_MAGI2_TEXT_GOLDEN");
  const char* te = std::getenv("VPIPE_MAGI2_TEXT_ENCODER");
  if (g == nullptr || te == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const FlexData meta = read_json(fs::path(g) / "meta.json");
  auto mo = meta.as_object();
  const int tap = (int)mo.at("tap_layer").as_int(0);
  genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(te);
  ASSERT_TRUE(mcfg.has_value());
  genai::MetalQwenModel::Config c = genai::MetalQwenModel::config_from(*mcfg);
  c.use_bf16 = true;
  c.backbone_only = true;
  c.weight_prefix = "model.language_model.";
  c.model_seg = "";
  c.zero_centered_norm = true;
  c.max_seq = genai::magi2::kMaxTextTokens + 256;
  c.page_tokens = 256;
  c.stream_layers = true;
  c.stream_decode = true;
  auto ws = genai::WeightSet::open(te, nullptr);
  ASSERT_TRUE(ws != nullptr);
  auto m = genai::MetalQwenModel::load(ws, mc, c);
  ASSERT_TRUE(m != nullptr);
  std::printf("[magi2] encoder: %d layers, hidden %d, streams %d\n",
              m->config().n_layers, m->config().hidden,
              m->streams_decode() ? 1 : 0);
  const int H = m->config().hidden;
  const auto table = ws->tensor("model.language_model.embed_tokens.weight",
                                mc, genai::WeightSet::Residency::Mapped);
  ASSERT_TRUE(!table.empty());
  int bad = 0;
  for (const char* nm : {"fox", "cjk", "figure"}) {
    const FlexData e = mo.at(nm);
    auto eo = e.as_object();
    const FlexData idf = eo.at("ids");
    auto ia = idf.as_array();
    const int n = (int)ia.size();
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
    for (int i = 0; i < n; ++i) {
      std::memcpy((char*)x.contents() + (std::size_t)i * H * 2,
                  (const char*)table.contents() +
                      (std::size_t)ia.at((std::size_t)i).as_int(0) * H * 2,
                  (std::size_t)H * 2);
    }
    genai::ContextManager* cm = m->context_manager();
    const genai::ContextId cid = cm->acquire_root();
    SharedBuffer t = m->forward_embeddings_taps(cid, x, n, {tap});
    cm->release(cid);
    ASSERT_TRUE(t.byte_size() == (std::size_t)n * H * 2);
    std::vector<float> got((std::size_t)n * H);
    const auto* p = (const std::uint16_t*)t.contents();
    for (std::size_t i = 0; i < got.size(); ++i) {
      const std::uint32_t u = (std::uint32_t)p[i] << 16;
      std::memcpy(&got[i], &u, 4);
    }
    const auto f32 = read_f32(fs::path(g) / (std::string(nm) + "_fp32.bin"));
    const auto b16 = read_f32(fs::path(g) / (std::string(nm) + "_bf16.bin"));
    const double d = rel(got, f32), bar = rel(b16, f32);
    std::printf("[magi2] %s: %d rows at layer %d: %.5f (ref bf16 %.5f)\n",
                nm, n, tap, d, bar);
    bad += !(d <= 1.5 * bar);
  }
  EXPECT_TRUE(bad == 0);
}

// THE TURBO DECODER against tools/dump_magi2_golden.py `turbo`
// (VPIPE_MAGI2_TURBO_GOLDEN, with VPIPE_MAGI2_TURBO_DIR the release's
// turbo_vae/): a latent long enough for two windows and the padding,
// within 1.5x the reference's own bf16 distance from its fp32.
TEST(magi2, turbo_decoder_matches_reference)
{
  const char* g = std::getenv("VPIPE_MAGI2_TURBO_GOLDEN");
  const char* d = std::getenv("VPIPE_MAGI2_TURBO_DIR");
  if (g == nullptr || d == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto dec = genai::MetalMagi2Turbo::load(d, mc, &err);
  if (!dec) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dec != nullptr);
  const FlexData meta = read_json(fs::path(g) / "meta.json");
  auto mo = meta.as_object();
  const int T = (int)mo.at("T").as_int(0), H = (int)mo.at("H").as_int(0),
            W = (int)mo.at("W").as_int(0);
  const auto z = read_f32(fs::path(g) / "latent.bin");
  std::vector<float> out;
  bool ok = false;
  const std::size_t grew = device_peak_of_(mc, [&] {
    ok = dec->decode(z.data(), T, H, W, &out, &err);
  });
  ASSERT_TRUE(ok);
  // What the decode allocated against what it says it will -- and the
  // plan, off the checkpoint's shapes alone, the same figures.
  const std::size_t booked = dec->scratch_bytes(T, H, W);
  std::size_t pw = 0, ps = 0;
  ASSERT_TRUE(genai::MetalMagi2Turbo::plan(d, T, H, W, &pw, &ps));
  EXPECT_TRUE(pw == dec->weight_bytes() && ps == booked);
  std::printf("[magi2] turbo decode allocated %.1f MB, booked %.1f MB\n",
              grew / 1048576.0, booked / 1048576.0);
  EXPECT_TRUE(grew <= booked);
  const auto f32 = read_f32(fs::path(g) / "video_fp32.bin");
  const auto b16 = read_f32(fs::path(g) / "video_bf16.bin");
  const double dv = rel(out, f32), bar = rel(b16, f32);
  std::printf("[magi2] turbo %dx%dx%d -> %d frames: %.5f (ref bf16 %.5f), "
              "weights %llu MB\n", T, H, W,
              genai::MetalMagi2Turbo::frames_of(T), dv, bar,
              (unsigned long long)(dec->weight_bytes() >> 20));
  EXPECT_TRUE(dv <= 1.5 * bar);
}

// THE SOUNDTRACK against tools/dump_magi2_golden.py `audio`
// (VPIPE_MAGI2_AUDIO_GOLDEN, with VPIPE_MAGI2_AUDIO_DIR the release's
// stable-audio-open-1.0/): Stable Audio Open's decoder read by the Oobleck
// class (its model_config.json, the "pretransform.model." names), decoded
// whole as the release does, then stretched 441/512 by the FFT resample
// (held alone, on the reference's own decode, to f32 rounding).
// The reference decodes in fp32 and the port in f16. A random latent's
// decode is ILL-CONDITIONED near its zero-padded start: the golden's
// lands 0.0017 .. 0.0102 from fp32 under equally precise f16 roundings
// (MEASURED: the same content 8 frames later decodes at 0.0013). So the
// whole clip is held to that spread, and the median 128-sample window --
// where a fault (a wrong weight, a shifted tap) shows -- to 2x the plain
// f16 emulation's.
TEST(magi2, the_soundtrack_decodes_as_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_AUDIO_GOLDEN");
  const char* d = std::getenv("VPIPE_MAGI2_AUDIO_DIR");
  if (g == nullptr || d == nullptr) { return; }
  const FlexData meta = read_json(fs::path(g) / "meta.json");
  auto mo = meta.as_object();
  const int L = (int)mo.at("frames").as_int(0);
  const long raw_n = (long)mo.at("raw_samples").as_int(0);
  const long num = (long)mo.at("samples").as_int(0);
  const auto z = read_f32(fs::path(g) / "latent.bin");
  const auto raw = read_f32(fs::path(g) / "pcm_raw.bin");
  const auto ref = read_f32(fs::path(g) / "pcm.bin");
  ASSERT_TRUE(genai::magi2::stretched_length(raw_n) == num);

  // The resample alone, on the reference's own decode.
  const auto res = genai::magi2::fft_resample(raw.data(), 2, raw_n, num);
  const double dr = rel(res, ref);
  std::printf("[magi2] resample %ld -> %ld: %.3g from the reference's\n",
              raw_n, num, dr);
  EXPECT_TRUE(dr < 1e-6);

  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  genai::MetalOobleckDecoder::Config cfg;
  ASSERT_TRUE(genai::MetalOobleckDecoder::config_from_dir(d, &cfg, &err));
  auto aws = genai::WeightSet::open(d, nullptr);
  ASSERT_TRUE(aws != nullptr);
  auto dec = genai::MetalOobleckDecoder::load(aws, mc, cfg, &err);
  if (!dec) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dec != nullptr);
  // What the plan books for it (off the headers) is what its set caches:
  // the file is the whole of Stable Audio Open.
  {
    const std::size_t held =
        genai::MetalOobleckDecoder::held_bytes(aws->src(), cfg);
    const std::size_t set = aws->stats().bytes;
    std::printf("[magi2] the audio decoder holds %zu B, booked %zu B (its "
                "file %.2f GB)\n", set, held,
                fs::file_size(fs::path(d) / "model.safetensors") / 1e9);
    EXPECT_TRUE(held > 0 && held == set);
  }
  EXPECT_TRUE(dec->config().hop() == 2048 &&
              dec->config().sample_rate == 44100 &&
              dec->natural_length(L) == raw_n);
  dec->set_core_frames(L);
  std::vector<float> pcm;
  ASSERT_TRUE(dec->decode(z.data(), L, &pcm, /*clamp=*/false, {}, &err));
  const double dd = rel(pcm, raw);
  if (const char* dp = std::getenv("VPIPE_MAGI2_AUDIO_DUMP")) {
    std::FILE* f = std::fopen(dp, "wb");
    if (f != nullptr) {
      std::fwrite(pcm.data(), 4, pcm.size(), f);
      std::fclose(f);
    }
  }
  const auto st = genai::magi2::fft_resample(pcm.data(), 2, raw_n, num);
  const double ds = rel(st, ref);
  // The median over 128-sample windows of each window's largest error.
  std::vector<float> win;
  for (long i0 = 0; i0 + 128 <= raw_n; i0 += 128) {
    float m = 0.0f;
    for (int c = 0; c < 2; ++c) {
      for (long i = i0; i < i0 + 128; ++i) {
        const std::size_t k = (std::size_t)c * raw_n + i;
        m = std::max(m, std::fabs(pcm[k] - raw[k]));
      }
    }
    win.push_back(m);
  }
  std::nth_element(win.begin(), win.begin() + win.size() / 2, win.end());
  const double wmed = win[win.size() / 2];
  const double spread = mo.at("bar_spread").as_real(0);
  const double wbar = mo.at("window_median_f16").as_real(0);
  std::printf("[magi2] SAO decode %d latents: %.5f from fp32 (f16 %.5f, "
              "spread %.5f), stretched %.5f; window median %.6f (f16 "
              "%.6f)\n", L, dd, mo.at("bar_f16").as_real(0), spread, ds,
              wmed, wbar);
  EXPECT_TRUE(dd <= spread && ds <= spread && wmed <= 2.0 * wbar);
}

// THE SAMPLER against tools/dump_magi2_golden.py `sched`
// (VPIPE_MAGI2_SCHED_GOLDEN): the grid (np.linspace(float32(0.999), 0,
// n + 1)[:-1], shifted) to the bit, the DiT's integer timesteps exactly,
// and FlowSampler's UniPC (bh2, order 2) on a toy velocity field of the
// timestep against the release's FlowUniPCMultistepScheduler restated --
// 100, 12 and 5 steps. The reference keeps its sigmas in float32 and
// FlowSampler in double, hence a tolerance on the trajectory alone.
TEST(magi2, the_sampler_steps_as_the_release)
{
  EXPECT_TRUE(genai::magi2::video_latents(10.0) == 32 &&
              genai::magi2::audio_latents(10.0) == 250);
  const char* g = std::getenv("VPIPE_MAGI2_SCHED_GOLDEN");
  if (g == nullptr) { return; }
  const FlexData meta = read_json(fs::path(g) / "meta.json");
  const FlexData cases = meta.as_object().at("cases");
  for (const FlexData& cs : cases.as_array()) {
    auto co = cs.as_object();
    const std::string tag(co.at("tag").as_string());
    const int steps = (int)co.at("steps").as_int(0);
    const double shift = co.at("shift").as_real(0);
    std::vector<double> ref_grid((std::size_t)steps);
    std::vector<std::int64_t> ref_t((std::size_t)steps);
    {
      std::ifstream f(fs::path(g) / (tag + "_grid.bin"), std::ios::binary);
      f.read((char*)ref_grid.data(), (std::streamsize)steps * 8);
      std::ifstream ft(fs::path(g) / (tag + "_t.bin"), std::ios::binary);
      ft.read((char*)ref_t.data(), (std::streamsize)steps * 8);
    }
    const auto grid = genai::magi2::sigma_grid(steps, shift);
    int grid_off = 0, t_off = 0;
    for (int i = 0; i < steps; ++i) {
      grid_off += grid[(std::size_t)i] != ref_grid[(std::size_t)i] ? 1 : 0;
      t_off += genai::magi2::model_timestep(grid[(std::size_t)i]) !=
                       ref_t[(std::size_t)i]
                   ? 1 : 0;
    }
    std::vector<float> x = read_f32(fs::path(g) / (tag + "_x0.bin"));
    const auto ref = read_f32(fs::path(g) / (tag + "_x.bin"));
    genai::FlowSampler fs_(genai::magi2::sampler_spec(),
                           genai::magi2::scheduler_spec(steps, shift));
    fs_.reset();
    int sig_off = 0;
    for (int i = 0; i < steps; ++i) {
      fs_.step(i, x, [&](const std::vector<float>& xi, double sigma) {
        sig_off += sigma != grid[(std::size_t)i] ? 1 : 0;
        const float t = genai::magi2::model_t(sigma);
        std::vector<float> v(xi.size());
        for (std::size_t j = 0; j < xi.size(); ++j) {
          v[j] = xi[j] * (0.6f + 0.4f * t) - 0.25f * std::tanh(xi[j]) +
                 0.05f * std::cos(3.0f * xi[j] + t);
        }
        return v;
      });
    }
    const double d = rel(x, ref);
    std::printf("[magi2] sampler %s: grid %d off, timesteps %d off, "
                "sigmas handed %d off, trajectory %.2e\n", tag.c_str(),
                grid_off, t_off, sig_off, d);
    EXPECT_TRUE(grid_off == 0 && t_off == 0 && sig_off == 0 && d < 1e-5);
  }
}

// THE PREVIEW LOOP against tools/dump_magi2_golden.py `loop`
// (VPIPE_MAGI2_LOOP_GOLDEN), over a TOY velocity field in place of the
// DiT -- a random DiT's trajectory is chaotic, so the loop's plumbing is
// held exactly apart from the DiT's arithmetic: the noise drawn video
// then audio from torch's CPU generator at the seed, the conditional and
// the negative prompt's passes in one batch (the picture on both), uncond
// + g (cond - uncond) at video 5 / audio 7, a UniPC each over the shift-7
// grid, the field told float32(int64(sigma * 1000)) / 1000. The toy reads
// the latent, the timestep, the text rows' mean and the picture's.
TEST(magi2, the_preview_loop_runs_as_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_LOOP_GOLDEN");
  if (g == nullptr) { return; }
  using Sample = genai::MetalMagi2Dit::Sample;
  for (const char* nm : {"t2v", "i2v"}) {
    const fs::path d = fs::path(g) / nm;
    const FlexData meta = read_json(d / "meta.json");
    auto mo = meta.as_object();
    const int text_c = (int)mo.at("text_c").as_int(0);
    const auto cond = read_f32(d / "cond.bin");
    const auto neg = read_f32(d / "neg.bin");
    std::vector<float> img, spec;
    genai::magi2::PreviewRequest r;
    r.T = (int)mo.at("T").as_int(0);
    r.h = (int)mo.at("H").as_int(0);
    r.w = (int)mo.at("W").as_int(0);
    r.audio_len = (int)mo.at("audio").as_int(0);
    r.steps = (int)mo.at("steps").as_int(0);
    r.shift = mo.at("shift").as_real(0);
    r.video_guidance = mo.at("gv").as_real(0);
    r.audio_guidance = mo.at("ga").as_real(0);
    r.seed = (std::uint64_t)mo.at("seed").as_int(0);
    r.cond = cond.data();
    r.cond_rows = (int)(cond.size() / (std::size_t)text_c);
    r.neg = neg.data();
    r.neg_rows = (int)(neg.size() / (std::size_t)text_c);
    if (mo.at("image").as_bool(false)) {
      img = read_f32(d / "image0.bin");
      spec = read_f32(d / "special0.bin");
      genai::MetalMagi2Dit::Image im;
      im.latent = img.data();
      im.h = 4;
      im.w = 6;
      im.special = spec.data();
      r.images.push_back(im);
    }
    int batches = 0, pairs = 0;
    auto toy = [&](const std::vector<const Sample*>& b,
                   std::vector<std::vector<float>>* vv,
                   std::vector<std::vector<float>>* va, std::string*) {
      ++batches;
      pairs += b.size() == 2 ? 1 : 0;
      vv->assign(b.size(), {});
      va->assign(b.size(), {});
      for (std::size_t k = 0; k < b.size(); ++k) {
        const Sample& sm = *b[k];
        double tm = 0.0;
        const std::size_t nt = (std::size_t)sm.text_len * text_c;
        for (std::size_t i = 0; i < nt; ++i) { tm += sm.text[i]; }
        const float m = (float)(tm / (double)nt);
        float im = 0.0f;
        if (!sm.images.empty()) {
          double s2 = 0.0;
          const std::size_t ni = (std::size_t)48 * sm.images[0].h *
                                 sm.images[0].w;
          for (std::size_t i = 0; i < ni; ++i) {
            s2 += sm.images[0].latent[i];
          }
          im = (float)(s2 / (double)ni);
        }
        auto field = [&](const float* x, std::size_t n, bool video) {
          std::vector<float> v(n);
          for (std::size_t i = 0; i < n; ++i) {
            float y = x[i] * (0.6f + 0.4f * sm.t) - 0.25f * std::tanh(x[i]) +
                      0.05f * std::cos(3.0f * x[i] + sm.t);
            y = y + 0.1f * m;
            if (video && !sm.images.empty()) { y = y + 0.02f * im; }
            v[i] = y;
          }
          return v;
        };
        (*vv)[k] = field(sm.video, (std::size_t)48 * sm.T * sm.H * sm.W,
                         true);
        (*va)[k] = field(sm.audio, (std::size_t)sm.audio_len * 64, false);
      }
      return true;
    };
    std::vector<float> v, a;
    std::string err;
    ASSERT_TRUE(genai::magi2::run_preview(toy, 48, 64, r, &v, &a, &err));
    const double dv = rel(v, read_f32(d / "video.bin"));
    const double da = rel(a, read_f32(d / "audio.bin"));
    std::printf("[magi2] loop %s: %d steps, %d batches (%d with the "
                "negative), video %.2e audio %.2e\n", nm, r.steps, batches,
                pairs, dv, da);
    EXPECT_TRUE(batches == r.steps && pairs == r.steps && dv < 1e-5 &&
                da < 1e-5);
  }
}

// THE PLAN BOOKS WHAT THE DiT HOLDS. resident_bytes() -- every weight
// buffer -- equals trunk_bytes() (asked before a load, off the headers)
// plus the held experts plus the two slots; and a batch's forward
// allocates no more than scratch_bytes() books for it. The small golden,
// experts held and streamed.
TEST(magi2, the_plan_books_what_the_dit_holds)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  const std::size_t trunk = genai::MetalMagi2Dit::trunk_bytes(*ws, cfg);
  Case cs[2];
  ASSERT_TRUE(load_case(fs::path(g) / "t2v", &cs[0]));
  ASSERT_TRUE(load_case(fs::path(g) / "i2v", &cs[1]));
  std::vector<genai::MetalMagi2Dit::Sample> smp{sample_of(cs[0], cfg),
                                                sample_of(cs[1], cfg)};
  for (int resident : {-1, 0}) {
    cfg.resident_moe = resident;
    auto m = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
    ASSERT_TRUE(m != nullptr);
    const std::size_t ex = m->expert_layer_bytes();
    const std::size_t want =
        trunk + (std::size_t)m->moe_held() * ex +
        (m->moe_streamed() > 0 ? 2 * ex : 0);
    const std::size_t held = m->resident_bytes();
    // The scratch for the batch: its longest sample's tokens and its
    // largest input.
    int n = 0, inputs = 0;
    for (const auto& sm : smp) {
      const auto L = genai::MetalMagi2Dit::layout(sm, std::vector<float>(
          (std::size_t)cfg.head_dim / 8, 1.0f));
      n = std::max(n, L.n);
      int in = cfg.video_c * sm.T * sm.H * sm.W +
               cfg.audio_c * sm.audio_len +
               cfg.text_c * (sm.text_len + (int)sm.images.size());
      for (const auto& im : sm.images) { in += cfg.video_c * im.h * im.w; }
      inputs = std::max(inputs, in);
    }
    const std::size_t booked =
        genai::MetalMagi2Dit::scratch_bytes(cfg, n, 2, inputs);
    m->release_scratch();
    std::vector<std::vector<float>> vv, va;
    bool ok = false;
    const std::size_t grew = device_peak_of_(mc, [&] {
      ok = m->forward(
          std::vector<const genai::MetalMagi2Dit::Sample*>{&smp[0], &smp[1]},
          &vv, &va, &err);
    });
    ASSERT_TRUE(ok);
    std::printf("[magi2] experts %s: held %zu B against %zu planned; a "
                "batch of 2 (%d tokens) allocated %.2f MB, booked %.2f MB\n",
                resident == 0 ? "streamed" : "held", held, want, n,
                grew / 1048576.0, booked / 1048576.0);
    EXPECT_TRUE(held <= want && want - held <= want / 50);
    EXPECT_TRUE(grew <= booked);
  }
}

// THE RELEASE'S ROOT IS MAGI-2's, to every registry that is asked: the
// video family (generate-video), the VAE family (vae-decode and
// audio-vae-decode, each at its own path), and nothing else claims it.
// The plan's figures off the headers: the DiT's floor is its trunk and
// two slots; the turbo decoder's claim books what it holds.
TEST(magi2, the_release_root_is_claimed_as_magi2)
{
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (m == nullptr) { return; }
  const std::string root = fs::path(m).parent_path().string();
  genai::magi2::Layout l;
  ASSERT_TRUE(genai::magi2::resolve_layout(root, &l));
  EXPECT_TRUE(!l.refiner.empty() && !l.wan_vae.empty());
  auto* vf = genai::VideoModelRegistry::get().claim_for(nullptr, root, "");
  ASSERT_TRUE(vf != nullptr);
  EXPECT_TRUE(vf->tag() == "magi2");
  EXPECT_TRUE(vf->align_frames(root, 121) == 121 &&
              vf->align_frames(root, 122) == 125);
  auto* af = genai::VaeModelRegistry::get().claim_for(nullptr, root, root,
                                                      "");
  ASSERT_TRUE(af != nullptr);
  EXPECT_TRUE(af->tag() == "magi2");
  // The turbo decoder by the FILE it opens -- its claim, its weight set
  // and its release one name.
  const std::string turbo_file =
      (fs::path(l.turbo) / "checkpoint.ckpt").string();
  EXPECT_TRUE(af->vae_path(root, genai::VaeModelFamily::kRoleVideo) ==
                  turbo_file &&
              af->vae_path(root, genai::VaeModelFamily::kRoleAudio) ==
                  l.audio);
  // Its parts are not roots.
  EXPECT_TRUE(!genai::magi2::is_root(l.preview) &&
              !genai::magi2::is_root(l.text_encoder));
  genai::magi2::DitPlan p;
  ASSERT_TRUE(genai::magi2::dit_plan(l.preview, &p));
  const auto h = vf->declare_holdings(root);
  ASSERT_TRUE(h.size() == 1);
  std::printf("[magi2] DiT: trunk %.2f GB, %d MoE layers of %.2f GB; "
              "floor %.2f GB, whole %.2f GB; 540p scratch %.2f GB\n",
              p.trunk() / 1e9, p.moe_layers, p.expert_layer / 1e9,
              h[0].floor / 1e9, h[0].preload / 1e9,
              genai::magi2::denoise_scratch_bytes(p, 896, 512, 125) / 1e9);
  EXPECT_TRUE(p.moe_layers == 36 &&
              h[0].floor == p.trunk() + 2 * p.expert_layer);
  std::size_t tw = 0, ts = 0;
  ASSERT_TRUE(genai::MetalMagi2Turbo::plan(l.turbo, 32, 32, 56, &tw, &ts));
  std::printf("[magi2] turbo decoder: %.1f MB held, a 540p clip's decode "
              "%.2f GB\n", tw / 1048576.0, ts / 1e9);
  // Each codec held at what it keeps, under the name it opens -- and the
  // claims say so: the decoder's file, the audio decoder's directory (a
  // 4.85 GB file of the whole Stable Audio Open), the Wan2.2 encoder half.
  auto one = [&](std::string_view role) {
    const auto hs = af->declare_holdings(root, role);
    return hs.size() == 1 ? hs[0] : StageHolding{};
  };
  const StageHolding hv = one(genai::VaeModelFamily::kRoleVideo);
  const StageHolding ha = one(genai::VaeModelFamily::kRoleAudio);
  const StageHolding he = one(genai::VaeModelFamily::kRoleVideoEncode);
  std::printf("[magi2] holdings: turbo %.1f MB, audio %.1f MB (of %.2f GB "
              "on disk), picture encoder %.1f MB (of %.2f GB)\n",
              hv.preload / 1048576.0, ha.preload / 1048576.0,
              model_memory::dir_weights_bytes(l.audio) / 1e9,
              he.preload / 1048576.0,
              model_memory::dir_weights_bytes(l.wan_vae) / 1e9);
  EXPECT_TRUE(hv.source == turbo_file && hv.preload == tw);
  EXPECT_TRUE(ha.source == l.audio && ha.preload > 0 &&
              ha.preload * 4 < model_memory::dir_weights_bytes(l.audio));
  EXPECT_TRUE(he.source == l.wan_vae && he.preload > 0 &&
              he.preload < model_memory::dir_weights_bytes(l.wan_vae));
  const auto vc = af->declare_resources(root, root);
  ASSERT_TRUE(vc.size() == 1);
  EXPECT_TRUE(vc[0].key == turbo_file &&
              bag::integer(vc[0].extra, model_memory::kClaimHeldBytes, 0) ==
                  (std::int64_t)tw);
  // ...and an audio claim carries its holding through the shared rule.
  std::vector<ResourceClaim> ac = model_memory::weight_claims({l.audio});
  model_memory::claims_at_holdings(&ac, {ha});
  EXPECT_TRUE(ac.size() == 1 &&
              bag::integer(ac[0].extra, model_memory::kClaimHeldBytes, 0) ==
                  (std::int64_t)ha.preload);
}

// IMAGE-TO-VIDEO'S PICTURE against tools/dump_magi2_golden.py `wan22`
// (VPIPE_MAGI2_WAN22_GOLDEN, with VPIPE_MAGI2_TEST_MODEL_PATH): the Wan2.2
// VAE's encoder on one frame, here as a 2D network on the host, against
// the release's 3D path restated (causal 3x3x3 convs over a padded frame,
// AvgDown3D's zero frame, the first chunk's rules) -- a picture on the
// 32-pixel grid, and one off it floored by PIL's LANCZOS. Both in f32, so
// the bar is f32 summation order.
TEST(magi2, the_picture_encodes_as_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_WAN22_GOLDEN");
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (g == nullptr || m == nullptr) { return; }
  const std::string root = fs::path(m).parent_path().string();
  genai::magi2::Layout l;
  ASSERT_TRUE(genai::magi2::resolve_layout(root, &l));
  auto* af = genai::VaeModelRegistry::get().claim_for(nullptr, root, root,
                                                      "");
  ASSERT_TRUE(af != nullptr);
  EXPECT_TRUE(af->vae_path(root,
                           genai::VaeModelFamily::kRoleVideoEncode) ==
              l.wan_vae);
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto pws = genai::WeightSet::open(l.wan_vae, nullptr);
  ASSERT_TRUE(pws != nullptr);
  auto enc = genai::Magi2ImageEncoder::load(pws, mc, &err);
  if (!enc) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(enc != nullptr);
  // The plan's figure for it, off the headers, is what it holds.
  const std::size_t held = genai::Magi2ImageEncoder::held_bytes(pws->src());
  std::printf("[magi2] the picture encoder holds %zu B, booked %zu B\n",
              enc->weight_bytes(), held);
  EXPECT_TRUE(held == enc->weight_bytes());
  const FlexData meta = read_json(fs::path(g) / "meta.json");
  const FlexData cases = meta.as_object().at("cases");
  for (const FlexData& cs : cases.as_array()) {
    auto co = cs.as_object();
    const std::string nm(co.at("name").as_string());
    const int H = (int)co.at("H").as_int(0), W = (int)co.at("W").as_int(0);
    const auto in = read_f32(fs::path(g) / nm / "input.bin");
    const auto ref = read_f32(fs::path(g) / nm / "latent.bin");
    std::vector<float> z;
    int h = 0, w = 0;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = false;
    const std::size_t grew = host_peak_of_([&] {
      ok = enc->encode(in.data(), H, W, &z, &h, &w, &err);
    });
    ASSERT_TRUE(ok);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    const double d = rel(z, ref);
    const std::size_t booked = genai::Magi2ImageEncoder::working_bytes(H, W);
    std::printf("[magi2] picture %s %dx%d -> [48, %d, %d]: %.2e from the "
                "release's (%.0f ms; weights %.1f MB; the host's heap grew "
                "%.1f MB, booked %.1f MB)\n", nm.c_str(), H, W, h, w, d, ms,
                enc->weight_bytes() / 1048576.0, grew / 1048576.0,
                booked / 1048576.0);
    EXPECT_TRUE(h == (int)co.at("h").as_int(0) &&
                w == (int)co.at("w").as_int(0) && d < 1e-4);
    EXPECT_TRUE(grew <= booked);
  }
}

// ---- the refiner ------------------------------------------------------------

namespace {

// Each case of a refiner golden: the velocity against the reference's
// fp32 within `slack` of its own bf16's distance. And THE WINDOW MATTERS
// there: the same model with a radius covering the whole grid (dense
// attention) misses the reference by more than the bar.
bool
check_refiner_cases(genai::MetalMagi2Refiner::Config cfg,
                    const std::shared_ptr<genai::WeightSet>& ws,
                    metal_compute::MetalCompute* mc, const fs::path& g,
                    const std::vector<std::string>& names, double slack)
{
  bool ok = true;
  for (const std::string& nm : names) {
    const fs::path d = g / nm;
    const FlexData meta = read_json(d / "meta.json");
    if (!meta.is_object()) {
      std::printf("[magi2] refiner %s: no meta.json\n", nm.c_str());
      return false;
    }
    auto o = meta.as_object();
    const FlexData blk = o.at("block");
    const FlexData rad = o.at("radius");
    for (int a = 0; a < 3; ++a) {
      cfg.block[a] = (int)blk.as_array()[(std::size_t)a].as_int(0);
      cfg.radius[a] = (int)rad.as_array()[(std::size_t)a].as_int(0);
    }
    const std::vector<float> video = read_f32(d / "video.bin");
    const std::vector<float> text = read_f32(d / "text.bin");
    const std::vector<float> v32 = read_f32(d / "vel_fp32.bin");
    genai::MetalMagi2Refiner::Sample s;
    s.T = (int)o.at("T").as_int(0);
    s.H = (int)o.at("H").as_int(0);
    s.W = (int)o.at("W").as_int(0);
    s.video = video.data();
    s.text_len = (int)o.at("text").as_int(0);
    s.text = text.data();
    const double bar = o.at("bar").as_real(0.0);
    double r[2] = {0.0, 0.0};
    for (int dense = 0; dense < 2; ++dense) {
      genai::MetalMagi2Refiner::Config c2 = cfg;
      if (dense != 0) {
        c2.radius[0] = c2.radius[1] = c2.radius[2] = 1 << 20;
      }
      std::string err;
      auto m = genai::MetalMagi2Refiner::load(ws, mc, c2, &err);
      if (!m) {
        std::printf("[magi2] refiner: %s\n", err.c_str());
        return false;
      }
      if (dense == 0) {
        // THE M5 ROUTES ENGAGE where the GPU has matrix cores, unless an
        // A/B knob turned them off -- else the golden holds the steel ones.
        auto knob = [](const char* nm) {
          const char* e = std::getenv(nm);
          return e != nullptr && e[0] != '\0' && e[0] != '0';
        };
        const bool mc5 = mc->supports_matrix_cores();
        std::printf("[magi2] refiner %s on %s GEMMs, %s window attention\n",
                    nm.c_str(), m->uses_mma() ? "matmul2d" : "steel",
                    m->uses_nax_attention() ? "matrix-core" : "steel");
        if (m->uses_mma() !=
                (mc5 && !knob("VPIPE_MAGI2_NO_MMA2")) ||
            m->uses_nax_attention() !=
                (mc5 && !knob("VPIPE_MAGI2_NO_NAX_ATTN"))) {
          std::printf("[magi2] refiner: the routes are not the GPU's\n");
          return false;
        }
      }
      std::vector<float> v;
      if (!m->forward(s, &v, &err)) {
        std::printf("[magi2] refiner %s: %s\n", nm.c_str(), err.c_str());
        return false;
      }
      r[dense] = rel(v, v32);
    }
    const bool pass = r[0] <= slack * bar && r[1] > slack * bar;
    std::printf("[magi2] refiner %s: %.5f from fp32 (ref bf16 %.5f), "
                "dense %.5f %s\n",
                nm.c_str(), r[0], bar, r[1], pass ? "ok" : "FAIL");
    ok = ok && pass;
  }
  return ok;
}

}  // namespace

TEST(magi2, refiner_matches_reference)
{
  const char* g = std::getenv("VPIPE_MAGI2_REFINER_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Refiner::Config cfg;
  std::string err;
  const bool cfg_ok =
      genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg, &err);
  if (!cfg_ok) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(cfg_ok);
  EXPECT_TRUE(check_refiner_cases(cfg, ws, mc, g,
                                  {"smallblocks", "release_blocks"}, 1.5));
}

// THE RELEASE'S REFINER, a subset of its layers (the golden's).
TEST(magi2, refiner_matches_reference_on_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_REFINER_REAL_GOLDEN");
  const char* dir = std::getenv("VPIPE_MAGI2_REFINER_PATH");
  if (g == nullptr || dir == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const FlexData meta = read_json(fs::path(g) / "real" / "meta.json");
  ASSERT_TRUE(meta.is_object());
  std::vector<int> layers;
  const FlexData lf = meta.as_object().at("layers");
  for (const FlexData& v : lf.as_array()) {
    layers.push_back((int)v.as_int(0));
  }
  auto ws = genai::WeightSet::open(dir, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Refiner::Config cfg;
  std::string err;
  const bool cfg_ok =
      genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg, &err, layers);
  if (!cfg_ok) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(cfg_ok);
  EXPECT_TRUE(cfg.n_q == 32 && cfg.n_kv == 8 && cfg.mlp_inter == 10920);
  EXPECT_TRUE(check_refiner_cases(cfg, ws, mc, g, {"real"}, 1.5));
}

// THE REFINER'S SAMPLING, over a toy field: the ZeroSNR signal scale to
// the bit, torch's trilinear upsample, the re-noise from the seed's draw,
// and UniPC over 5 steps at shift 5 with guidance 2 (cond then the
// negative, a forward each).
TEST(magi2, the_refine_loop_runs_as_the_release)
{
  const char* g = std::getenv("VPIPE_MAGI2_REFINE_GOLDEN");
  if (g == nullptr) { return; }
  const fs::path d(g);
  const FlexData meta = read_json(d / "meta.json");
  ASSERT_TRUE(meta.is_object());
  auto mo = meta.as_object();
  const float s = genai::magi2::refine_signal_scale();
  std::uint32_t bits = 0;
  std::memcpy(&bits, &s, 4);
  std::printf("[magi2] refine signal scale %.9f (bits %u, release's %lld)\n",
              s, bits, (long long)mo.at("signal_scale_bits").as_int(0));
  EXPECT_TRUE((long long)bits == mo.at("signal_scale_bits").as_int(-1));

  const int T = (int)mo.at("T").as_int(0), h = (int)mo.at("h").as_int(0);
  const int w = (int)mo.at("w").as_int(0), H = (int)mo.at("H").as_int(0);
  const int W = (int)mo.at("W").as_int(0);
  const std::vector<float> lat = read_f32(d / "latent.bin");
  const std::vector<float> up = genai::magi2::upsample_trilinear(
      lat.data(), 48, T, h, w, 2 * T - 1, H, W);
  const double du = rel(up, read_f32(d / "up.bin"));
  std::printf("[magi2] refine upsample %.2e from torch's\n", du);
  EXPECT_TRUE(du < 1e-6);

  const std::vector<float> cond = read_f32(d / "cond.bin");
  const std::vector<float> neg = read_f32(d / "neg.bin");
  const int text_c = (int)mo.at("text_c").as_int(0);
  genai::magi2::RefineRequest r;
  r.T = T; r.h = h; r.w = w; r.latent = lat.data(); r.H = H; r.W = W;
  r.steps = (int)mo.at("steps").as_int(0);
  r.shift = mo.at("shift").as_real(0.0);
  r.guidance = mo.at("guidance").as_real(0.0);
  r.seed = (std::uint64_t)mo.at("seed").as_int(0);
  r.cond = cond.data();
  r.cond_rows = (int)mo.at("cond").as_int(0);
  r.neg = neg.data();
  r.neg_rows = (int)mo.at("neg").as_int(0);
  int calls = 0;
  auto toy = [&](const genai::MetalMagi2Refiner::Sample& sm,
                 std::vector<float>* v, std::string*) {
    ++calls;
    double tm = 0.0;
    const std::size_t nt = (std::size_t)sm.text_len * text_c;
    for (std::size_t i = 0; i < nt; ++i) { tm += sm.text[i]; }
    const float m = (float)(tm / (double)nt);
    const std::size_t n = (std::size_t)48 * sm.T * sm.H * sm.W;
    v->assign(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
      const float x = sm.video[i];
      float y = x * 0.6f - 0.25f * std::tanh(x) + 0.05f * std::cos(3.0f * x);
      (*v)[i] = y + 0.1f * m;
    }
    return true;
  };
  std::vector<float> v;
  std::string err;
  ASSERT_TRUE(genai::magi2::run_refine(toy, 48, r, &v, &err));
  const double dv = rel(v, read_f32(d / "video.bin"));
  std::printf("[magi2] refine loop: %d forwards, video %.2e\n", calls, dv);
  EXPECT_TRUE(calls == 2 * r.steps && dv < 1e-5);
}

// THE PLAN BOOKS WHAT THE REFINER HOLDS: resident_bytes() equals
// weight_bytes() (off the headers, before a load), and a forward
// allocates no more than scratch_bytes() books -- at the golden's
// geometry, on whichever attention this box runs (the plan's figure, at
// the matrix-core tile, is the larger).
TEST(magi2, the_plan_books_what_the_refiner_holds)
{
  const char* g = std::getenv("VPIPE_MAGI2_REFINER_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Refiner::Config cfg;
  ASSERT_TRUE(genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg,
                                                            nullptr));
  const std::size_t planned = genai::MetalMagi2Refiner::weight_bytes(*ws,
                                                                     cfg);
  std::string err;
  auto m = genai::MetalMagi2Refiner::load(ws, mc, cfg, &err);
  ASSERT_TRUE(m != nullptr);
  std::printf("[magi2] refiner holds %zu bytes, planned %zu\n",
              (std::size_t)m->resident_bytes(), planned);
  EXPECT_TRUE(m->resident_bytes() == planned);

  const fs::path d = fs::path(g) / "release_blocks";
  const FlexData meta = read_json(d / "meta.json");
  ASSERT_TRUE(meta.is_object());
  auto o = meta.as_object();
  const std::vector<float> video = read_f32(d / "video.bin");
  const std::vector<float> text = read_f32(d / "text.bin");
  genai::MetalMagi2Refiner::Sample s;
  s.T = (int)o.at("T").as_int(0);
  s.H = (int)o.at("H").as_int(0);
  s.W = (int)o.at("W").as_int(0);
  s.video = video.data();
  s.text_len = (int)o.at("text").as_int(0);
  s.text = text.data();
  std::vector<float> v;
  bool ok = false;
  const std::size_t grew = device_peak_of_(mc, [&] {
    ok = m->forward(s, &v, &err);
  });
  ASSERT_TRUE(ok);
  const std::size_t booked = genai::MetalMagi2Refiner::scratch_bytes(
      cfg, s.T, s.H, s.W, s.text_len, m->uses_nax_attention());
  const std::size_t plan = genai::MetalMagi2Refiner::scratch_bytes(
      cfg, s.T, s.H, s.W, s.text_len, true);
  std::printf("[magi2] refiner forward allocated %.2f MB, booked %.2f MB "
              "(the plan's %.2f)\n", grew / 1048576.0, booked / 1048576.0,
              plan / 1048576.0);
  EXPECT_TRUE(grew <= booked && booked <= plan);
  m->release_scratch();
}

// The turbo decoder's working set at the release's two clip geometries --
// the 540p preview and the 1080p refine -- off its config alone.
TEST(magi2, the_turbo_decoder_plans_both_presets)
{
  const char* dir = std::getenv("VPIPE_MAGI2_TURBO_DIR");
  if (dir == nullptr) { return; }
  struct G { int T, h, w; const char* name; };
  for (const G& g : {G{32, 32, 56, "540p preview"},
                     G{63, 68, 120, "1080p refine"}}) {
    std::size_t wb = 0, sb = 0;
    ASSERT_TRUE(genai::MetalMagi2Turbo::plan(dir, g.T, g.h, g.w, &wb, &sb));
    std::printf("[magi2] turbo decode of the %s (%d x %d x %d latents): "
                "weights %.1f MB, working set %.1f MB\n", g.name, g.T, g.h,
                g.w, wb / 1048576.0, sb / 1048576.0);
    EXPECT_TRUE(sb > 0);
  }
}

// THE TILED DECODE IS THE WHOLE ONE. Every op of the turbo decoder is
// local, so tiles with a halo of its receptive field (halo_cells(), off
// its structure) decode the whole plane's pixels bit for bit -- and two
// cells short of it they do not, so the figure is the field's. One window
// in time (7 latent frames), 2 x 2 tiles of a 32 x 34 plane -- each tile
// cut on its inner sides (core 16 x 17 + the halo inside the plane).
TEST(magi2, the_tiled_decode_is_the_whole_one)
{
  const char* dir = std::getenv("VPIPE_MAGI2_TURBO_DIR");
  if (dir == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto dec = genai::MetalMagi2Turbo::load(dir, mc, &err);
  if (!dec) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dec != nullptr);
  const int T = 7, h = 32, w = 34;
  std::vector<float> z((std::size_t)48 * T * h * w);
  std::uint32_t st = 12345;
  for (float& v : z) {
    st = st * 1664525u + 1013904223u;
    v = ((float)(st >> 8) / 16777216.0f - 0.5f) * 3.0f;
  }
  const int halo = dec->halo_cells();
  std::printf("[magi2] turbo receptive field %.4f latent cells: halo %d\n",
              dec->receptive_cells(), halo);
  ASSERT_TRUE(halo >= 2 && 16 + halo < h && 17 + halo < w);
  std::vector<float> whole;
  dec->set_tiles(0, 0, -1);
  ASSERT_TRUE(dec->tiling(T, h, w).ny == 1);
  const auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(dec->decode(z.data(), T, h, w, &whole, &err));
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::printf("[magi2] turbo whole decode %d x %d x %d latents: %.2f s\n", T,
              h, w, secs);
  for (int hl : {halo - 2, halo}) {
    std::vector<float> tiled;
    dec->set_tiles(16, 17, hl);
    const auto t = dec->tiling(T, h, w);
    ASSERT_TRUE(t.ny == 2 && t.nx == 2);
    ASSERT_TRUE(dec->decode(z.data(), T, h, w, &tiled, &err));
    std::size_t diff = 0;
    for (std::size_t i = 0; i < whole.size(); ++i) {
      diff += whole[i] != tiled[i] ? 1 : 0;
    }
    std::printf("[magi2] turbo tiles 2x2, halo %d: %zu of %zu values "
                "differ (rel %.2e)\n", hl, diff, whole.size(),
                rel(tiled, whole));
    EXPECT_TRUE(hl == halo ? diff == 0 : diff > 0);
  }
  dec->set_tiles(0, 0, -1);
}

// THE REFINE IS IN THE PLAN, off the configs alone: generate-video asked
// for the release's 1080p preset (a 896 x 512 x 125 preview refined to
// 1920 x 1088) claims the refiner's weights, books the latent and the
// decode at the refined clip -- 2T - 1 latents, 249 frames on the larger
// grid, the turbo decoder's tiled working set -- and the denoise at the
// larger of the two stages; asked for none, it claims none of it.
TEST(magi2, the_refine_is_in_the_plan)
{
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (m == nullptr) { return; }
  const std::string root = fs::path(m).parent_path().string();
  genai::magi2::Layout l;
  ASSERT_TRUE(genai::magi2::resolve_layout(root, &l));
  constexpr unsigned kModelCfgPort = 9;   // generate-video's iport 9
  auto bytes_of = [](const std::vector<ResourceClaim>& cs,
                     const std::string& label) {
    std::size_t b = 0;
    for (const ResourceClaim& c : cs) {
      if (c.kind != model_memory::kScratchKind ||
          c.key.compare(0, label.size() + 1, label + "|") != 0) {
        continue;
      }
      b += (std::size_t)std::stoull(c.key.substr(label.size() + 1));
    }
    return b;
  };
  struct Got {
    bool refiner = false;
    std::size_t latent = 0, decode = 0, denoise = 0;
  };
  auto plan = [&](bool refine) {
    Got g;
    Session sess;
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(root));
    c.as_object().insert("width", FlexData::make_int(896));
    c.as_object().insert("height", FlexData::make_int(512));
    c.as_object().insert("frames", FlexData::make_int(125));
    c.as_object().insert("fps", FlexData::make_real(12.5));
    GenerateVideoStage gen(&sess, "gen", std::vector<InEdge>{}, c);
    FlexData mc = FlexData::make_object();
    mc.as_object().insert("model_family", FlexData::make_string("magi2"));
    if (refine) {
      mc.as_object().insert("refine_width", FlexData::make_int(1920));
      mc.as_object().insert("refine_height", FlexData::make_int(1088));
    }
    gen.apply_constant(kModelCfgPort, mc);
    const auto cs = gen.declare_resources();
    for (const ResourceClaim& r : cs) {
      if (r.kind == model_memory::kWeightsKind && r.key == l.refiner) {
        g.refiner = true;
      }
    }
    g.latent = bytes_of(cs, "gen-latent");
    g.decode = bytes_of(cs, "vae-decode");
    g.denoise = bytes_of(cs, "gen-denoise");
    return g;
  };
  const Got pv = plan(false), rf = plan(true);
  // The refined clip: 63 latents on 68 x 120 cells, 249 frames.
  const std::size_t latent = (std::size_t)48 * 63 * 68 * 120 * 4;
  std::size_t turbo = 0;
  ASSERT_TRUE(genai::MetalMagi2Turbo::plan(l.turbo, 63, 68, 120, nullptr,
                                           &turbo));
  const std::size_t decode =
      model_memory::video_decode_scratch_bytes(1920, 1088, 249) + turbo;
  genai::magi2::RefinerPlan rp;
  ASSERT_TRUE(genai::magi2::refiner_plan(l.refiner, &rp));
  genai::magi2::RefineAsk ra;
  ra.width = 1920;
  ra.height = 1088;
  ra.steps = 5;
  const std::size_t refine = genai::magi2::refine_scratch_bytes(rp, ra, 125);
  std::printf("[magi2] plan, preview: latent %.1f MB, decode %.1f MB, "
              "denoise %.1f MB, refiner %s\n", pv.latent / 1048576.0,
              pv.decode / 1048576.0, pv.denoise / 1048576.0,
              pv.refiner ? "claimed" : "not claimed");
  std::printf("[magi2] plan, refine to 1920 x 1088: latent %.1f MB, decode "
              "%.1f MB (turbo %.1f), denoise %.1f MB (refine %.1f), refiner "
              "%s (%.1f MB held)\n", rf.latent / 1048576.0,
              rf.decode / 1048576.0, turbo / 1048576.0,
              rf.denoise / 1048576.0, refine / 1048576.0,
              rf.refiner ? "claimed" : "not claimed",
              rp.weights() / 1048576.0);
  EXPECT_TRUE(!pv.refiner && rf.refiner);
  EXPECT_TRUE(rf.latent == latent && pv.latent < latent);
  EXPECT_TRUE(rf.decode == decode && pv.decode < decode);
  EXPECT_TRUE(rf.denoise >= refine && rf.denoise > pv.denoise);
}

// BENCH (VPIPE_MAGI2_REFINER_BENCH = a refiner's dir -- the release's, or
// `dump_magi2_golden.py refiner --hidden 4096` for its widths on random
// weights; VPIPE_MAGI2_REFINER_BENCH_GRID = "T,H,W" latent cells, the
// 1080p refine's 63,68,120 by default): a warm forward's time, its
// projections' and window attention's FLOPs off the sequence, and the
// figure for the release's 30 layers.
namespace {

// A bench's acceleration arms, VPIPE_MAGI2_BENCH_ARMS: a comma list of
// "<tier>[+sage|+pv8]", the tier off / accurate / fast / token /
// token_plain ("off" alone by default) -- run in order, then in reverse,
// each arm's best forward of the two rounds kept.
struct BenchArm {
  std::string name;
  int tier = -1;
  bool sage = false, pv8 = false;
  double best = 1e30;
};

std::vector<BenchArm>
bench_arms_()
{
  std::string list = "off";
  if (const char* e = std::getenv("VPIPE_MAGI2_BENCH_ARMS")) { list = e; }
  std::vector<BenchArm> out;
  std::size_t p = 0;
  while (p <= list.size()) {
    const std::size_t q = std::min(list.find(',', p), list.size());
    const std::string a = list.substr(p, q - p);
    p = q + 1;
    if (a.empty()) { continue; }
    BenchArm b;
    b.name = a;
    const std::size_t plus = a.find('+');
    const std::string t = a.substr(0, plus);
    b.tier = t == "accurate" ? 0 : t == "fast" ? 1 : t == "token" ? 2
             : t == "token_plain" ? 3 : -1;
    const std::string sg = plus == std::string::npos ? "" : a.substr(plus + 1);
    b.sage = sg == "sage" || sg == "pv8";
    b.pv8 = sg == "pv8";
    out.push_back(b);
  }
  return out;
}

// Each arm in order then in reverse, `run` timing one forward of a model
// loaded for the arm (`load` returns false when it cannot).
template <class Load, class Run>
void
bench_rounds_(std::vector<BenchArm>& arms, Load&& load, Run&& run)
{
  for (int round = 0; round < 2; ++round) {
    for (std::size_t i = 0; i < arms.size(); ++i) {
      BenchArm& a = arms[round == 0 ? i : arms.size() - 1 - i];
      if (!load(a)) { continue; }
      for (int r = 0; r < 2; ++r) { a.best = std::min(a.best, run()); }
    }
  }
}

}  // namespace

TEST(magi2, refiner_bench)
{
  const char* dir = std::getenv("VPIPE_MAGI2_REFINER_BENCH");
  if (dir == nullptr) { return; }
  int T = 63, H = 68, W = 120;
  if (const char* g = std::getenv("VPIPE_MAGI2_REFINER_BENCH_GRID")) {
    std::sscanf(g, "%d,%d,%d", &T, &H, &W);
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(dir, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Refiner::Config cfg;
  ASSERT_TRUE(genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg,
                                                            nullptr));
  std::string err;
  const int text = 300;
  std::vector<float> video((std::size_t)cfg.video_c * T * H * W);
  std::vector<float> rows((std::size_t)text * cfg.text_c);
  std::uint32_t st = 7;
  for (std::vector<float>* v : {&video, &rows}) {
    for (float& x : *v) {
      st = st * 1664525u + 1013904223u;
      x = ((float)(st >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
  }
  genai::MetalMagi2Refiner::Sample s;
  s.T = T; s.H = H; s.W = W;
  s.video = video.data();
  s.text_len = text;
  s.text = rows.data();
  std::unique_ptr<genai::MetalMagi2Refiner> m;
  std::vector<BenchArm> arms = bench_arms_();
  bench_rounds_(
      arms,
      [&](const BenchArm& a) {
        m.reset();
        genai::MetalMagi2Refiner::Config c = cfg;
        c.i8_tier = genai::magi2::effective_i8_tier(
            a.tier, mc->supports_matrix_cores());
        c.sage.enabled = a.sage;
        c.sage.pv8 = a.pv8;
        m = genai::MetalMagi2Refiner::load(ws, mc, c, &err);
        return m != nullptr;
      },
      [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> v;
        if (!m->forward(s, &v, &err)) { return 1e30; }
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
      });
  ASSERT_TRUE(m != nullptr);
  const auto q = genai::MetalMagi2Refiner::sequence(
      cfg, T, H, W, text, m->uses_nax_attention() ? 64 : 32,
      m->uses_nax_attention() ? 32 : 16);
  double pairs = 0.0;
  for (std::size_t t = 0; t + 1 < q.qb_off.size(); ++t) {
    const int rows_here = std::min(q.bq, q.n - (int)t * q.bq);
    pairs += (double)(q.qb_off[t + 1] - q.qb_off[t]) * q.bk * rows_here;
  }
  const double per_tok =
      2.0 * ((double)(cfg.n_q + 2 * cfg.n_kv) * cfg.head_dim * cfg.hidden +
             (double)cfg.n_q * cfg.hidden +
             (double)cfg.n_q * cfg.head_dim * cfg.hidden +
             3.0 * cfg.mlp_inter * cfg.hidden);
  const double layers = (double)cfg.layers.size();
  const double gemm = per_tok * q.n * layers;
  const double attn = 4.0 * cfg.n_q * cfg.head_dim * pairs * layers;
  for (const BenchArm& a : arms) {
    std::printf("[magi2] refiner bench %d x %d x %d latents + %d text (%d "
                "tokens laid out; %.0f keys a query), %d layers, %s GEMMs, "
                "%s attention, arm %s: %.2f s a forward, %.2f TFLOP (%.3f "
                "window attention) at %.1f TFLOP/s; 30 layers ~%.1f s a "
                "forward (%.2fx the first arm)\n",
                T, H, W, text, q.n, pairs / q.n, (int)layers,
                m->uses_mma() ? "matmul2d" : "steel",
                m->uses_nax_attention() ? "matrix-core" : "steel",
                a.name.c_str(), a.best, (gemm + attn) * 1e-12, attn * 1e-12,
                (gemm + attn) / a.best * 1e-12, a.best / layers * 30.0,
                arms[0].best / a.best);
  }
  m->release_scratch();
}

// BENCH (VPIPE_MAGI2_PREVIEW_BENCH = a preview DiT's dir --
// `dump_magi2_golden.py wide --release-widths` writes one mm layer and one
// MoE layer at the release's widths on random weights; _GRID = "T,H,W"
// latent cells, the 540p preview's 32,32,56 by default): a guided step's
// batch of two (300 prompt rows, 200 negative), every arm of
// VPIPE_MAGI2_BENCH_ARMS, the experts held.
TEST(magi2, preview_bench)
{
  const char* dir = std::getenv("VPIPE_MAGI2_PREVIEW_BENCH");
  if (dir == nullptr) { return; }
  int T = 32, H = 32, W = 56;
  if (const char* g = std::getenv("VPIPE_MAGI2_PREVIEW_BENCH_GRID")) {
    std::sscanf(g, "%d,%d,%d", &T, &H, &W);
  }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(dir, nullptr);
  ASSERT_TRUE(ws != nullptr);
  // VPIPE_MAGI2_PREVIEW_BENCH_LAYERS = "0" or "1": one layer alone (the
  // mm layer or the MoE layer), to split a forward's time by kind.
  std::vector<int> subset;
  if (const char* e = std::getenv("VPIPE_MAGI2_PREVIEW_BENCH_LAYERS")) {
    subset.push_back(std::atoi(e));
  }
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err,
                                                        subset));
  const int L = genai::magi2::audio_latents_of(4 * (T - 1) + 1);
  std::vector<float> video((std::size_t)cfg.video_c * T * H * W);
  std::vector<float> audio((std::size_t)L * cfg.audio_c);
  std::vector<float> rows((std::size_t)300 * cfg.text_c);
  std::uint32_t st = 11;
  for (std::vector<float>* v : {&video, &audio, &rows}) {
    for (float& x : *v) {
      st = st * 1664525u + 1013904223u;
      x = ((float)(st >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
  }
  genai::MetalMagi2Dit::Sample cs, us;
  for (auto* p : {&cs, &us}) {
    p->T = T; p->H = H; p->W = W;
    p->video = video.data();
    p->audio_len = L;
    p->audio = audio.data();
    p->text = rows.data();
    p->t = 0.5f;
  }
  cs.text_len = 300;
  us.text_len = 200;
  std::unique_ptr<genai::MetalMagi2Dit> m;
  std::vector<BenchArm> arms = bench_arms_();
  bench_rounds_(
      arms,
      [&](const BenchArm& a) {
        m.reset();
        genai::MetalMagi2Dit::Config c = cfg;
        c.i8_tier = genai::magi2::effective_i8_tier(
            a.tier, mc->supports_matrix_cores());
        c.sage.enabled = a.sage;
        c.sage.pv8 = a.pv8;
        m = genai::MetalMagi2Dit::load(ws, mc, c, &err);
        return m != nullptr;
      },
      [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> vv, va;
        if (!m->forward(std::vector<const genai::MetalMagi2Dit::Sample*>{
                            &cs, &us},
                        &vv, &va, &err)) {
          return 1e30;
        }
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
      });
  ASSERT_TRUE(m != nullptr);
  const int n = T * H * W + L + 300;
  const double attn = 2.0 * 4.0 * (double)n * n * cfg.n_q * cfg.head_dim *
                      (double)cfg.layers.size();
  for (const BenchArm& a : arms) {
    std::printf("[magi2] preview bench %d x %d x %d latents + %d audio + "
                "300 / 200 text (a batch of 2), %zu layers, %s GEMMs, %s "
                "attention, arm %s: %.2f s a forward (%.1f TFLOP of "
                "attention); %.2fx the first arm\n",
                T, H, W, L, cfg.layers.size(),
                m->uses_mma() ? "matmul2d" : "steel",
                m->uses_nax_attention() ? "matrix-core" : "steel",
                a.name.c_str(), a.best, attn * 1e-12, arms[0].best / a.best);
  }
  m->release_scratch();
}

// magi2-model-config hands the refine keys on when set, none when not,
// and refuses a refine given half (width without height) or too small.
TEST(magi2, the_model_config_carries_the_refine)
{
  Session sess;
  auto make = [](std::vector<std::pair<std::string, FlexData>> kv) {
    FlexData c = FlexData::make_object();
    for (auto& [k, v] : kv) { c.as_object().insert(k, std::move(v)); }
    return c;
  };
  {
    Magi2ModelConfigStage s(&sess, "a", std::vector<InEdge>{},
                            make({{"refine_width", FlexData::make_int(1920)},
                                  {"refine_height", FlexData::make_int(1088)},
                                  {"refine_steps", FlexData::make_int(5)}}));
    EXPECT_TRUE(s.config_error().empty());
    FlexData out = s.resolved_config();
    const genai::magi2::RefineAsk ra = genai::magi2::refine_ask(&out);
    EXPECT_TRUE(ra.on() && ra.grid_w() == 120 && ra.grid_h() == 68 &&
                ra.steps == 5 && ra.shift == 5.0 && ra.guidance == 2.0);
  }
  {
    Magi2ModelConfigStage s(&sess, "b", std::vector<InEdge>{},
                            make({{"steps", FlexData::make_int(50)}}));
    FlexData out = s.resolved_config();
    EXPECT_TRUE(s.config_error().empty() &&
                !genai::magi2::refine_ask(&out).on());
  }
  {
    Magi2ModelConfigStage s(&sess, "c", std::vector<InEdge>{},
                            make({{"refine_width", FlexData::make_int(1920)}}));
    EXPECT_TRUE(!s.config_error().empty());
  }
  {
    Magi2ModelConfigStage s(&sess, "d", std::vector<InEdge>{},
                            make({{"refine_width", FlexData::make_int(8)},
                                  {"refine_height", FlexData::make_int(8)}}));
    EXPECT_TRUE(!s.config_error().empty());
  }
  // The release floors pixels to whole cells: 1080 is 67 cells, 1072 px.
  FlexData f = FlexData::make_object();
  f.as_object().insert("refine_width", FlexData::make_int(1920));
  f.as_object().insert("refine_height", FlexData::make_int(1080));
  EXPECT_TRUE(genai::magi2::refine_ask(&f).grid_h() == 67);
}

// ---- ACCELERATION: the int8 GEMM tiers and SageAttention ----------------
//
// Against the WIDE goldens -- `dump_magi2_golden.py wide` (the preview DiT
// at 1024 wide, a 1152-row video span) and `refiner --hidden 1024 --cases`
// (5616 video tokens) -- which clear the int8 tiers' floors: every block
// projection 1024 deep or more (the accurate tier's K) and over 1024 rows
// (its row floor); the text and audio spans stay under it, so the dense
// fall-back runs beside the int8 route. Below those floors a golden
// exercises nothing.
namespace {

// The preview's wide cases under `dit` (fp32's MoE picks forced, so a near
// tie does not decide it): the worst of each case's video and audio
// distance from fp32 as a multiple of the reference's own bf16 distance.
double
preview_worst_ratio(genai::MetalMagi2Dit& dit, const fs::path& g)
{
  double worst = 0.0;
  for (const char* nm : {"t2v", "i2v"}) {
    Case c;
    if (!load_case(g / nm, &c)) { return 1e30; }
    const genai::MetalMagi2Dit::Sample s = sample_of(c, dit.config());
    force_golden_routes(dit, g / nm, s);
    std::vector<float> vv, va;
    std::string err;
    if (!dit.forward(s, &vv, &va, &err)) {
      std::printf("[magi2] %s: %s\n", nm, err.c_str());
      return 1e30;
    }
    worst = std::max({worst, rel(vv, c.vv32) / rel(c.vvbf, c.vv32),
                      rel(va, c.va32) / rel(c.vabf, c.va32)});
  }
  return worst;
}

struct RefinerCase {
  genai::MetalMagi2Refiner::Sample s;
  std::vector<float> video, text, v32;
  double bar = 0.0;
};

bool
load_refiner_case(const fs::path& d, genai::MetalMagi2Refiner::Config* cfg,
                  RefinerCase* rc)
{
  const FlexData meta = read_json(d / "meta.json");
  if (!meta.is_object()) { return false; }
  auto o = meta.as_object();
  const FlexData blk = o.at("block");
  const FlexData rad = o.at("radius");
  for (int a = 0; a < 3; ++a) {
    cfg->block[a] = (int)blk.as_array()[(std::size_t)a].as_int(0);
    cfg->radius[a] = (int)rad.as_array()[(std::size_t)a].as_int(0);
  }
  rc->video = read_f32(d / "video.bin");
  rc->text = read_f32(d / "text.bin");
  rc->v32 = read_f32(d / "vel_fp32.bin");
  rc->s.T = (int)o.at("T").as_int(0);
  rc->s.H = (int)o.at("H").as_int(0);
  rc->s.W = (int)o.at("W").as_int(0);
  rc->s.video = rc->video.data();
  rc->s.text_len = (int)o.at("text").as_int(0);
  rc->s.text = rc->text.data();
  rc->bar = o.at("bar").as_real(0.0);
  return !rc->v32.empty() && rc->bar > 0.0;
}

// The batch of two the preview runs a guided step as: the wide golden's
// cases, their tokens and inputs (what scratch_bytes is asked for).
struct PreviewBatch {
  Case cs[2];
  std::vector<genai::MetalMagi2Dit::Sample> smp;
  int n = 0, inputs = 0;
};

bool
load_preview_batch(const fs::path& g, const genai::MetalMagi2Dit::Config& c,
                   PreviewBatch* b)
{
  if (!load_case(g / "t2v", &b->cs[0]) || !load_case(g / "i2v", &b->cs[1])) {
    return false;
  }
  b->smp = {sample_of(b->cs[0], c), sample_of(b->cs[1], c)};
  for (const auto& sm : b->smp) {
    const auto L = genai::MetalMagi2Dit::layout(
        sm, std::vector<float>((std::size_t)c.head_dim / 8, 1.0f));
    b->n = std::max(b->n, L.n);
    int in = c.video_c * sm.T * sm.H * sm.W + c.audio_c * sm.audio_len +
             c.text_c * (sm.text_len + (int)sm.images.size());
    for (const auto& im : sm.images) { in += c.video_c * im.h * im.w; }
    b->inputs = std::max(b->inputs, in);
  }
  return true;
}

// One acceleration arm of both DiTs: the int8 tier (-1 none) and Sage.
struct Arm {
  const char* name;
  int tier;
  bool sage, pv8;
  double bar;     // the worst distance from fp32 allowed, x ref bf16's
};

// The arm's preview: engaged as asked, held == planned, within its bar,
// and two guided steps' device growth within the plan's scratch.
bool
check_preview_arm(metal_compute::MetalCompute* mc,
                  const std::shared_ptr<genai::WeightSet>& ws,
                  const genai::MetalMagi2Dit::Config& base,
                  const fs::path& g, const Arm& a)
{
  genai::MetalMagi2Dit::Config cfg = base;
  cfg.i8_tier = genai::magi2::effective_i8_tier(a.tier, true);
  cfg.sage.enabled = a.sage;
  cfg.sage.pv8 = a.pv8;
  std::string err;
  auto dit = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
  if (!dit) {
    std::printf("[magi2] preview %s: %s\n", a.name, err.c_str());
    return false;
  }
  const std::size_t want =
      genai::MetalMagi2Dit::trunk_bytes(*ws, cfg) +
      (std::size_t)dit->moe_held() * dit->expert_layer_bytes();
  const std::size_t held = dit->resident_bytes();
  const double r = preview_worst_ratio(*dit, g);
  PreviewBatch pb;
  if (!load_preview_batch(g, cfg, &pb)) { return false; }
  const std::size_t booked =
      genai::MetalMagi2Dit::scratch_bytes(cfg, pb.n, 2, pb.inputs, true);
  dit->release_scratch();
  bool ok = true;
  // Two guided steps: the int8 tier's tuner runs the shapes the first
  // met at the top of the second.
  const std::size_t grew = device_peak_of_(mc, [&] {
    for (int step = 0; step < 2 && ok; ++step) {
      for (auto& sm : pb.smp) { sm.step = step; }
      std::vector<std::vector<float>> vv, va;
      ok = dit->forward(std::vector<const genai::MetalMagi2Dit::Sample*>{
                            &pb.smp[0], &pb.smp[1]},
                        &vv, &va, &err);
    }
  });
  const bool engaged =
      dit->i8_tier() == cfg.i8_tier && (a.tier < 0 || dit->i8_gemms() > 0) &&
      dit->sage_on() == a.sage && (!a.sage || dit->sage_calls() > 0) &&
      (dit->sage_pv8_calls() > 0) == (a.sage && a.pv8);
  const bool pass = ok && engaged && held <= want &&
                    want - held <= want / 50 && r <= a.bar &&
                    grew <= booked;
  std::printf("[magi2] preview %-16s tier %-11s %3ld int8 / %3ld dense "
              "GEMMs, sage %ld (pv8 %ld): %.2fx ref bf16 (bar %.1f); held "
              "%.1f MB (planned %.1f); %.2f MB grew, %.2f booked %s\n",
              a.name, genai::magi2::tier_name(dit->i8_tier()),
              dit->i8_gemms(), dit->dense_gemms(), dit->sage_calls(),
              dit->sage_pv8_calls(), r, a.bar, held / 1048576.0,
              want / 1048576.0, grew / 1048576.0, booked / 1048576.0,
              pass ? "ok" : "FAIL");
  return pass;
}

// The arm's refiner, likewise.
bool
check_refiner_arm(metal_compute::MetalCompute* mc,
                  const std::shared_ptr<genai::WeightSet>& ws,
                  const genai::MetalMagi2Refiner::Config& base,
                  const fs::path& g, const Arm& a)
{
  genai::MetalMagi2Refiner::Config cfg = base;
  RefinerCase rc;
  if (!load_refiner_case(g / "release_blocks", &cfg, &rc)) { return false; }
  cfg.i8_tier = genai::magi2::effective_i8_tier(a.tier, true);
  cfg.sage.enabled = a.sage;
  cfg.sage.pv8 = a.pv8;
  std::string err;
  auto m = genai::MetalMagi2Refiner::load(ws, mc, cfg, &err);
  if (!m) {
    std::printf("[magi2] refiner %s: %s\n", a.name, err.c_str());
    return false;
  }
  const std::size_t want = genai::MetalMagi2Refiner::weight_bytes(*ws, cfg);
  const std::size_t held = m->resident_bytes();
  std::vector<float> v;
  bool ok = m->forward(rc.s, &v, &err);
  const double r = ok ? rel(v, rc.v32) / rc.bar : 1e30;
  const std::size_t booked = genai::MetalMagi2Refiner::scratch_bytes(
      cfg, rc.s.T, rc.s.H, rc.s.W, rc.s.text_len, m->uses_nax_attention());
  m->release_scratch();
  const std::size_t grew = device_peak_of_(mc, [&] {
    for (int step = 0; step < 2 && ok; ++step) {
      rc.s.step = step;
      ok = m->forward(rc.s, &v, &err);
    }
  });
  const bool engaged =
      m->i8_tier() == cfg.i8_tier && (a.tier < 0 || m->i8_gemms() > 0) &&
      m->sage_on() == a.sage && (!a.sage || m->sage_calls() > 0) &&
      (m->sage_pv8_calls() > 0) == (a.sage && a.pv8);
  const bool pass =
      ok && engaged && held == want && r <= a.bar && grew <= booked;
  std::printf("[magi2] refiner %-16s tier %-11s %3ld int8 / %3ld dense "
              "GEMMs, sage %ld (pv8 %ld): %.2fx ref bf16 (bar %.1f); held "
              "%.1f MB (planned %.1f); %.2f MB grew, %.2f booked %s\n",
              a.name, genai::magi2::tier_name(m->i8_tier()), m->i8_gemms(),
              m->dense_gemms(), m->sage_calls(), m->sage_pv8_calls(), r,
              a.bar, held / 1048576.0, want / 1048576.0, grew / 1048576.0,
              booked / 1048576.0, pass ? "ok" : "FAIL");
  return pass;
}

}  // namespace

// THE INT8 TIERS AND SAGEATTENTION on both DiTs (matrix cores): each arm
// ENGAGED as asked (int8 GEMMs counted; Sage's calls, and its P*V's only
// where asked), its weights held at what the plan books (a holding tier's
// layout in place of the bf16), the velocity within its bar of the fp32
// reference -- a multiple of the reference's own bf16 distance -- and two
// steps' device growth within the plan's scratch (the int8 context, its
// tuner and Sage's prologue in it). Without matrix cores every tier
// declines: the engine says -1, and the models load at -1.
TEST(magi2, the_acceleration_tiers_hold_their_bars)
{
  const char* gp = std::getenv("VPIPE_MAGI2_WIDE_GOLDEN");
  const char* gr = std::getenv("VPIPE_MAGI2_REFINER_WIDE_GOLDEN");
  if (gp == nullptr && gr == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    EXPECT_TRUE(genai::magi2::effective_i8_tier(2, false) == -1);
    genai::sage::Config sc;
    sc.enabled = true;
    EXPECT_TRUE(!genai::magi2::sage_runs(sc, false));
    std::printf("[magi2] no matrix cores: the tiers decline -- SKIPPED\n");
    return;
  }
  // The bars: the bf16 kernels at 1.5x the reference's own bf16 distance
  // (as the plain goldens); SageAttention 1.5x too, its int8 QK and P*V
  // being far under the projections' rounding; the int8 tiers a GROSS bar
  // of 5x -- on these random weights each int8 GEMM's ~1e-2 against bf16's
  // ~4e-3 sets their distance at 2.7-3.9x, and a broken route (a layout
  // read at the wrong group, a slab off by a modality) is 10x and more.
  const Arm arms[] = {
      {"bf16", -1, false, false, 1.5},
      {"accurate", 0, false, false, 5.0},
      {"fast", 1, false, false, 5.0},
      {"token", 2, false, false, 5.0},
      {"token_plain", 3, false, false, 5.0},
      {"sage", -1, true, false, 1.5},
      {"sage+pv8", -1, true, true, 1.5},
      {"token+sage+pv8", 2, true, true, 5.0},
  };
  if (gp != nullptr) {
    auto ws = genai::WeightSet::open(gp, nullptr);
    ASSERT_TRUE(ws != nullptr);
    genai::MetalMagi2Dit::Config cfg;
    std::string err;
    ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
    for (const Arm& a : arms) {
      EXPECT_TRUE(check_preview_arm(mc, ws, cfg, gp, a));
    }
  }
  if (gr != nullptr) {
    auto ws = genai::WeightSet::open(gr, nullptr);
    ASSERT_TRUE(ws != nullptr);
    genai::MetalMagi2Refiner::Config cfg;
    std::string err;
    ASSERT_TRUE(
        genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg, &err));
    for (const Arm& a : arms) {
      EXPECT_TRUE(check_refiner_arm(mc, ws, cfg, gr, a));
    }
  }
}

// THE PLAN AT EACH TIER, off the release's headers alone: the family's
// plan extension, told the acceleration bag and whether the box has
// matrix cores, books both DiTs at what a load at that tier holds (a
// holding tier's claims carry the smaller figure) and adds the int8 and
// Sage scratch to the denoise; told no matrix cores, it books bf16.
TEST(magi2, the_acceleration_is_in_the_plan)
{
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (m == nullptr) { return; }
  const std::string root = fs::path(m).parent_path().string();
  genai::magi2::Layout l;
  ASSERT_TRUE(genai::magi2::resolve_layout(root, &l));
  auto* vf = genai::VideoModelRegistry::get().claim_for(nullptr, root, "");
  ASSERT_TRUE(vf != nullptr);
  auto* plan = static_cast<genai::VideoConfigPlan*>(
      vf->query_extension(genai::kVideoConfigPlanExt));
  ASSERT_TRUE(plan != nullptr);
  FlexData mcfg = FlexData::make_object();
  mcfg.as_object().insert("refine_width", FlexData::make_int(1920));
  mcfg.as_object().insert("refine_height", FlexData::make_int(1088));
  auto bag = [](bool i8, const char* tier, bool sage, bool pv8) {
    FlexData b = FlexData::make_object();
    genai::accel::set_flag(&b, genai::accel::kI8Gemm, i8);
    genai::accel::set_text(&b, genai::accel::kI8GemmTier, tier);
    genai::accel::set_flag(&b, genai::accel::kSageAttn, sage);
    genai::accel::set_flag(&b, genai::accel::kSagePv8, pv8);
    return b;
  };
  struct Got {
    std::size_t dit_held = 0, refiner_held = 0, denoise = 0;
  };
  auto ask = [&](const FlexData& b, bool mx) {
    genai::VideoPlanContext ctx;
    ctx.model_config = &mcfg;
    ctx.accel = &b;
    ctx.matrix_cores = mx;
    Got got;
    for (const auto& h : plan->config_holdings(root, ctx)) {
      (h.source == l.refiner ? got.refiner_held : got.dit_held) = h.preload;
    }
    got.denoise =
        plan->config_denoise_scratch_bytes(root, 896, 512, 125, ctx);
    return got;
  };
  genai::magi2::DitPlan p;
  genai::magi2::RefinerPlan rp;
  ASSERT_TRUE(genai::magi2::dit_plan(l.preview, &p) &&
              genai::magi2::refiner_plan(l.refiner, &rp));
  const Got off = ask(bag(false, "accurate", false, false), true);
  const Got acc = ask(bag(true, "accurate", false, false), true);
  const Got tok = ask(bag(true, "token", true, true), true);
  const Got m4 = ask(bag(true, "token", true, true), false);
  std::printf("[magi2] plan, bf16: DiT %.2f GB, refiner %.2f GB, denoise "
              "%.2f GB\n", off.dit_held / 1e9, off.refiner_held / 1e9,
              off.denoise / 1e9);
  std::printf("[magi2] plan, accurate: DiT %.2f GB, refiner %.2f GB, "
              "denoise %.2f GB\n", acc.dit_held / 1e9, acc.refiner_held / 1e9,
              acc.denoise / 1e9);
  std::printf("[magi2] plan, token + sage_pv8: DiT %.2f GB (trunk %.2f of "
              "%.2f), refiner %.2f GB, denoise %.2f GB; told no matrix "
              "cores: %.2f / %.2f / %.2f\n", tok.dit_held / 1e9,
              p.trunk(2) / 1e9, p.trunk() / 1e9, tok.refiner_held / 1e9,
              tok.denoise / 1e9, m4.dit_held / 1e9, m4.refiner_held / 1e9,
              m4.denoise / 1e9);
  EXPECT_TRUE(off.dit_held == genai::magi2::dit_preload_bytes(p, -1) &&
              off.refiner_held == rp.weights(-1));
  EXPECT_TRUE(acc.dit_held == off.dit_held &&
              acc.refiner_held == off.refiner_held &&
              acc.denoise > off.denoise);
  EXPECT_TRUE(tok.dit_held == genai::magi2::dit_preload_bytes(p, 2) &&
              tok.dit_held < off.dit_held &&
              tok.refiner_held == rp.weights(2) &&
              tok.refiner_held < off.refiner_held &&
              tok.denoise > off.denoise);
  EXPECT_TRUE(m4.dit_held == off.dit_held &&
              m4.refiner_held == off.refiner_held &&
              m4.denoise == off.denoise);
  // The claims carry the held figure where a tier holds less.
  genai::VideoPlanContext ctx;
  ctx.model_config = &mcfg;
  const FlexData tb = bag(true, "token", false, false);
  ctx.accel = &tb;
  ctx.matrix_cores = true;
  std::size_t claimed_dit = 0, claimed_ref = 0;
  for (const ResourceClaim& c : plan->config_resources(root, ctx)) {
    if (c.kind != model_memory::kWeightsKind) { continue; }
    const auto hb = c.extra.is_object() && c.extra.as_object().contains(
                                               "held_bytes")
                        ? (std::size_t)c.extra.as_object()
                              .at("held_bytes").as_int(0)
                        : 0;
    (c.key == l.refiner ? claimed_ref : claimed_dit) = hb;
  }
  EXPECT_TRUE(claimed_dit == genai::magi2::dit_preload_bytes(p, 2) &&
              claimed_ref == rp.weights(2));
}

// THE ROUTED EXPERTS ON THE MATRIX CORES (magi2_nax.metal; M5): the
// preview's wide cases with fp32's picks forced, the matrix-core expert
// kernel against the simdgroup one (VPIPE_MAGI2_NO_NAX_MOE=1) and both
// against the fp32 reference. The two compute the same products in other
// orders, each rounding h to bf16 at the same point: they agree far inside
// the reference's own bf16 distance, and each holds the bar.
TEST(magi2, the_experts_on_the_matrix_cores_are_the_simdgroup_kernels)
{
  const char* g = std::getenv("VPIPE_MAGI2_WIDE_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[magi2] no matrix cores -- SKIPPED\n");
    return;
  }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  struct Run {
    std::vector<std::vector<float>> vv, va;
    bool nax = false;
    long dispatches = 0;
  };
  auto run = [&](bool nax) {
    Run r;
    if (!nax) { setenv("VPIPE_MAGI2_NO_NAX_MOE", "1", 1); }
    auto dit = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
    unsetenv("VPIPE_MAGI2_NO_NAX_MOE");
    if (!dit) { return r; }
    r.nax = dit->uses_nax_experts();
    for (const char* nm : {"t2v", "i2v"}) {
      Case c;
      if (!load_case(fs::path(g) / nm, &c)) { return r; }
      const auto s = sample_of(c, dit->config());
      force_golden_routes(*dit, fs::path(g) / nm, s);
      std::vector<float> vv, va;
      if (!dit->forward(s, &vv, &va, &err)) { return r; }
      r.vv.push_back(std::move(vv));
      r.va.push_back(std::move(va));
    }
    r.dispatches = dit->nax_expert_dispatches();
    return r;
  };
  const Run nax = run(true), sg = run(false);
  ASSERT_TRUE(nax.vv.size() == 2 && sg.vv.size() == 2);
  EXPECT_TRUE(nax.nax && nax.dispatches > 0 && !sg.nax &&
              sg.dispatches == 0);
  int i = 0;
  for (const char* nm : {"t2v", "i2v"}) {
    Case c;
    ASSERT_TRUE(load_case(fs::path(g) / nm, &c));
    const double bv = rel(c.vvbf, c.vv32), ba = rel(c.vabf, c.va32);
    const double nv = rel(nax.vv[(std::size_t)i], c.vv32) / bv;
    const double na = rel(nax.va[(std::size_t)i], c.va32) / ba;
    const double sv = rel(sg.vv[(std::size_t)i], c.vv32) / bv;
    const double sa = rel(sg.va[(std::size_t)i], c.va32) / ba;
    const double dv = rel(nax.vv[(std::size_t)i], sg.vv[(std::size_t)i]);
    const double da = rel(nax.va[(std::size_t)i], sg.va[(std::size_t)i]);
    std::printf("[magi2] %s experts: matrix-core %.2f / %.2f x ref bf16, "
                "simdgroup %.2f / %.2f (video / audio); the two %.2e / "
                "%.2e apart (ref bf16 %.2e / %.2e)\n", nm, nv, na, sv, sa,
                dv, da, bv, ba);
    EXPECT_TRUE(nv <= 1.5 && na <= 1.5 && sv <= 1.5 && sa <= 1.5);
    EXPECT_TRUE(dv < 0.5 * bv && da < 0.5 * ba);
    ++i;
  }
}

// THE STAGED EXPERT KERNEL (magi2_moe_expert_tg: the route a GPU without
// matrix cores takes) against the first simdgroup kernel
// (VPIPE_MAGI2_MOE_SG=1) and both against the fp32 reference, on the
// preview's wide cases with fp32's picks forced. On an M5 the matrix-core
// route is turned off for both. The same products in another order, h
// rounded to bf16 at the same point: each holds the bar, and they agree
// far inside the reference's own bf16 distance.
TEST(magi2, the_staged_experts_are_the_simdgroup_kernels)
{
  const char* g = std::getenv("VPIPE_MAGI2_WIDE_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  struct Run {
    std::vector<std::vector<float>> vv, va;
    bool staged = false;
  };
  auto run = [&](bool staged) {
    Run r;
    setenv("VPIPE_MAGI2_NO_NAX_MOE", "1", 1);
    if (!staged) { setenv("VPIPE_MAGI2_MOE_SG", "1", 1); }
    auto dit = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
    unsetenv("VPIPE_MAGI2_NO_NAX_MOE");
    unsetenv("VPIPE_MAGI2_MOE_SG");
    if (!dit) { return r; }
    r.staged = dit->uses_staged_experts() && !dit->uses_nax_experts();
    for (const char* nm : {"t2v", "i2v"}) {
      Case c;
      if (!load_case(fs::path(g) / nm, &c)) { return r; }
      const auto s = sample_of(c, dit->config());
      force_golden_routes(*dit, fs::path(g) / nm, s);
      std::vector<float> vv, va;
      if (!dit->forward(s, &vv, &va, &err)) { return r; }
      r.vv.push_back(std::move(vv));
      r.va.push_back(std::move(va));
    }
    return r;
  };
  const Run tg = run(true), sg = run(false);
  ASSERT_TRUE(tg.vv.size() == 2 && sg.vv.size() == 2);
  EXPECT_TRUE(tg.staged && !sg.staged);
  int i = 0;
  for (const char* nm : {"t2v", "i2v"}) {
    Case c;
    ASSERT_TRUE(load_case(fs::path(g) / nm, &c));
    const double bv = rel(c.vvbf, c.vv32), ba = rel(c.vabf, c.va32);
    const double tv = rel(tg.vv[(std::size_t)i], c.vv32) / bv;
    const double ta = rel(tg.va[(std::size_t)i], c.va32) / ba;
    const double sv = rel(sg.vv[(std::size_t)i], c.vv32) / bv;
    const double sa = rel(sg.va[(std::size_t)i], c.va32) / ba;
    const double dv = rel(tg.vv[(std::size_t)i], sg.vv[(std::size_t)i]);
    const double da = rel(tg.va[(std::size_t)i], sg.va[(std::size_t)i]);
    std::printf("[magi2] %s experts: staged %.2f / %.2f x ref bf16, first "
                "simdgroup %.2f / %.2f (video / audio); the two %.2e / "
                "%.2e apart (ref bf16 %.2e / %.2e)\n", nm, tv, ta, sv, sa,
                dv, da, bv, ba);
    EXPECT_TRUE(tv <= 1.5 && ta <= 1.5 && sv <= 1.5 && sa <= 1.5);
    EXPECT_TRUE(dv < 0.5 * bv && da < 0.5 * ba);
    ++i;
  }
}

namespace {

// TWO RANKS IN ONE PROCESS (tensor-parallel.h): each exchange leaves this
// rank's partial under its `seq` and takes the peer's of the same `seq`.
// A peer that never comes fails the exchange after a minute.
struct Magi2PairLink {
  std::mutex              mu;
  std::condition_variable cv;
  struct Entry {
    std::vector<std::uint8_t> part[2];
    bool has[2] = {false, false};
    int  taken = 0;
  };
  std::map<std::uint64_t, Entry> at;
  std::atomic<int>               mismatched{0};
};

struct Magi2PairChannel final : genai::TpChannel {
  std::shared_ptr<Magi2PairLink> link;
  int                            me = 0;
  std::atomic<std::uint64_t>     next{0};
  std::atomic<int>               out_of_step{0};
  std::atomic<std::uint64_t>     exchanges{0};
  Magi2PairChannel(std::shared_ptr<Magi2PairLink> l, int r)
      : link(std::move(l)), me(r)
  {
  }
  bool exchange(const void* partial, void* others, std::size_t bytes,
                genai::TpDType, std::uint64_t seq, std::string* err) override
  {
    if (seq != next.fetch_add(1)) { out_of_step.fetch_add(1); }
    const auto* p = static_cast<const std::uint8_t*>(partial);
    std::unique_lock<std::mutex> lk(link->mu);
    Magi2PairLink::Entry& e = link->at[seq];
    e.part[me].assign(p, p + bytes);
    e.has[me] = true;
    link->cv.notify_all();
    if (!link->cv.wait_for(lk, std::chrono::seconds(60),
                           [&]() { return e.has[1 - me]; })) {
      if (err != nullptr) { *err = "the peer never came"; }
      return false;
    }
    if (e.part[1 - me].size() != bytes) {
      link->mismatched.fetch_add(1);
      if (err != nullptr) { *err = "the ranks are out of step"; }
      return false;
    }
    std::memcpy(others, e.part[1 - me].data(), bytes);
    exchanges.fetch_add(1);
    if (++e.taken == 2) { link->at.erase(seq); }
    return true;
  }
};

// Rank r of two: heads [0, h) / [h, n_q), experts [0, f) / [f, E).
genai::MetalMagi2Dit::Config
magi2_rank_(const genai::MetalMagi2Dit::Config& base, int r, int h, int f)
{
  genai::MetalMagi2Dit::Config c = base;
  c.tp = genai::TpShare{};
  c.tp.rank = r;
  c.tp.ranks = 2;
  c.tp.n_heads = base.n_q;
  c.tp.n_ffn = base.moe_experts;
  c.tp.head0 = r == 0 ? 0 : h;
  c.tp.head1 = r == 0 ? h : base.n_q;
  c.tp.ffn0 = r == 0 ? 0 : f;
  c.tp.ffn1 = r == 0 ? f : base.moe_experts;
  return c;
}

// Each sample's velocities, video and audio.
struct Magi2Out {
  std::vector<std::vector<float>> v, a;
};

// One forward of `batch` through `m`: with `forced` (a case dir), every
// MoE layer's picks are that case's fp32 ones -- a batch of copies of
// one case -- else free.
bool
magi2_batch_(genai::MetalMagi2Dit& m,
             const std::vector<const genai::MetalMagi2Dit::Sample*>& batch,
             const fs::path& forced, Magi2Out* out, std::string* err)
{
  for (int idx : m.config().layers) { m.force_routes(idx, {}); }
  if (!forced.empty()) { force_golden_routes(m, forced, *batch[0]); }
  return m.forward(batch, &out->v, &out->a, err);
}

}  // namespace

// A RANK PAIR (tensor parallel) IS THE WHOLE MODEL: two ranks of the small
// golden in one process, over an in-process channel, their forwards run at
// once on two threads -- an even share (half the heads, half of each MoE
// head's experts and of every dense width) and an uneven one (a quarter),
// each with its experts held and streamed, each with a batch's exchanges
// overlapped across its samples and in line (VPIPE_MAGI2_TP_LAG=0). Every
// pair: its ranks' velocities BIT-IDENTICAL; with the golden's fp32 picks
// forced (a near tie between bf16 sums would otherwise decide it), each
// case alone and a guided pair of one case, within the unsplit test's bar
// of the fp32 reference (1.5x the release's own bf16 distance) -- the
// partials rounded to bf16 before they are added; a free batch of two
// cases the same bits overlapped and in line; the exchanges two a layer a
// sample, in step; each rank holding what the plan books for its share.
namespace {

// The split checks of a preview DiT over golden `g` (its weights in `ws`):
// `release` -- the release's weights, its layer subset -- one even share
// with its experts streamed, the holding unchecked (trunk_bytes books the
// whole checkpoint, which a subset does not hold); else both shares, held
// and streamed, and the acceleration arm on matrix cores.
bool
check_preview_split_(metal_compute::MetalCompute* mc,
                     const std::shared_ptr<genai::WeightSet>& ws,
                     const char* g, const genai::MetalMagi2Dit::Config& cfg,
                     bool release)
{
  std::string err;
  bool pass = true;
  auto check = [&](bool c, const char* what) {
    if (!c) {
      std::printf("[magi2] split: FAILED %s\n", what);
      pass = false;
    }
    return c;
  };
  const std::vector<std::string> names = {"t2v", "i2v"};
  Case cs[2];
  std::vector<genai::MetalMagi2Dit::Sample> smp;
  for (int i = 0; i < 2; ++i) {
    if (!check(load_case(fs::path(g) / names[(std::size_t)i], &cs[i]),
               "the cases")) {
      return false;
    }
    smp.push_back(sample_of(cs[i], cfg));
  }
  // The batches: each case alone (its picks forced), a guided pair of the
  // first case (forced), and the two cases together (free).
  struct Run {
    std::vector<const genai::MetalMagi2Dit::Sample*> batch;
    int forced = -1;           // the case whose picks are forced
  };
  const std::vector<Run> runs = {{{&smp[0]}, 0},
                                 {{&smp[1]}, 1},
                                 {{&smp[0], &smp[0]}, 0},
                                 {{&smp[0], &smp[1]}, -1}};
  // How far a velocity is from its case's fp32, in units of the release's
  // own bf16 distance (video and audio, the worse).
  auto ratio = [&](const Magi2Out& o, std::size_t i, int c) {
    const double bv = rel(cs[c].vvbf, cs[c].vv32);
    const double ba = rel(cs[c].vabf, cs[c].va32);
    return std::max(rel(o.v[i], cs[c].vv32) / bv,
                    rel(o.a[i], cs[c].va32) / ba);
  };
  auto run_all = [&](genai::MetalMagi2Dit& m, std::vector<Magi2Out>* outs,
                     std::string* e) {
    outs->assign(runs.size(), {});
    for (std::size_t r = 0; r < runs.size(); ++r) {
      const fs::path f = runs[r].forced >= 0
                             ? fs::path(g) / names[(std::size_t)runs[r].forced]
                             : fs::path();
      if (!magi2_batch_(m, runs[r].batch, f, &(*outs)[r], e)) {
        return false;
      }
    }
    return true;
  };
  // The unsplit model's worst forced ratio: what the bar is beside.
  double whole_worst = 0.0;
  {
    auto m = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
    if (!check(m != nullptr, "the unsplit load")) { return false; }
    std::vector<Magi2Out> w;
    if (!check(run_all(*m, &w, &err), "the unsplit forwards")) {
      return false;
    }
    for (std::size_t r = 0; r + 1 < runs.size(); ++r) {
      for (std::size_t i = 0; i < runs[r].batch.size(); ++i) {
        whole_worst = std::max(whole_worst, ratio(w[r], i, runs[r].forced));
      }
    }
  }
  // ...and on matrix cores the acceleration on a rank's cut widths: the
  // int8 token tier and SageAttention with its int8 P*V (the tiers' own
  // bar: random weights put int8 at its floor).
  struct ShareArm {
    const char* name;
    int h, f;
    bool accel = false;
  };
  std::vector<ShareArm> shares = {{"even", cfg.n_q / 2, cfg.moe_experts / 2},
                                  {"uneven", cfg.n_q / 4,
                                   cfg.moe_experts / 4}};
  if (release) {
    shares.resize(1);
  } else if (mc->supports_matrix_cores()) {
    shares.push_back({"accel", cfg.n_q / 2, cfg.moe_experts / 2, true});
  }
  for (const ShareArm& sa : shares) {
    for (int resident : {-1, 0}) {
      if (release && resident != 0) { continue; }
      std::vector<Magi2Out> lagged;
      for (int lag : {1, 0}) {
        if (lag == 0) {
          ::setenv("VPIPE_MAGI2_TP_LAG", "0", 1);
        } else {
          ::unsetenv("VPIPE_MAGI2_TP_LAG");
        }
        std::unique_ptr<genai::MetalMagi2Dit> m[2];
        bool loaded = true;
        for (int r = 0; r < 2 && loaded; ++r) {
          genai::MetalMagi2Dit::Config c = magi2_rank_(cfg, r, sa.h, sa.f);
          c.resident_moe = resident;
          if (sa.accel) {
            c.i8_tier = genai::magi2::effective_i8_tier(2, true);
            c.sage.enabled = true;
            c.sage.pv8 = true;
          }
          m[r] = genai::MetalMagi2Dit::load(ws, mc, c, &err);
          if (m[r] == nullptr) {
            std::printf("[magi2] rank %d: %s\n", r, err.c_str());
            loaded = false;
          }
        }
        if (!check(loaded, "the ranks' loads")) { return false; }
        if (!loaded) { return false; }
        // What each rank holds against what the plan books for its share.
        bool booked = true;
        for (int r = 0; r < 2 && !release; ++r) {
          const std::size_t xb = m[r]->expert_layer_bytes();
          const std::size_t want =
              genai::MetalMagi2Dit::trunk_bytes(*ws, m[r]->config()) +
              (std::size_t)m[r]->moe_held() * xb +
              (m[r]->moe_streamed() > 0 ? 2 * xb : 0);
          const std::size_t held = m[r]->resident_bytes();
          // ...and the family's plan of the share says the same, at the
          // tier the rank holds.
          genai::magi2::DitPlan dp;
          const bool planned =
              genai::magi2::dit_plan(g, m[r]->config().tp, &dp) &&
              dp.trunk(m[r]->config().i8_tier) ==
                  genai::MetalMagi2Dit::trunk_bytes(*ws, m[r]->config()) &&
              dp.expert_layer == m[r]->expert_layer_bytes();
          const bool ok = held <= want && want - held <= want / 50 &&
                          planned;
          if (!ok) {
            std::printf("[magi2]   rank %d held %zu, booked %zu, the share "
                        "plan %s\n", r, held, want,
                        planned ? "agrees" : "DIFFERS");
          }
          booked = booked && ok;
        }
        auto link = std::make_shared<Magi2PairLink>();
        std::shared_ptr<Magi2PairChannel> ch[2];
        for (int r = 0; r < 2; ++r) {
          ch[r] = std::make_shared<Magi2PairChannel>(link, r);
          std::string be;
          if (!check(m[r]->begin_tp_generation(ch[r], {}, &be), "begin")) {
            return false;
          }
        }
        std::vector<Magi2Out> o[2];
        std::string e1;
        bool ok1 = false;
        std::thread t([&]() { ok1 = run_all(*m[1], &o[1], &e1); });
        std::string e0;
        const bool ok0 = run_all(*m[0], &o[0], &e0);
        t.join();
        for (int r = 0; r < 2; ++r) { m[r]->end_tp_generation(); }
        if (!ok0 || !ok1) {
          std::printf("[magi2] split forward: %s | %s\n", e0.c_str(),
                      e1.c_str());
        }
        if (!check(ok0 && ok1, "the split forwards")) { return false; }
        if (!ok0 || !ok1) { return false; }
        bool same = true;
        double worst = 0.0;
        for (std::size_t r = 0; r < runs.size(); ++r) {
          for (std::size_t i = 0; i < runs[r].batch.size(); ++i) {
            same = same && o[0][r].v[i] == o[1][r].v[i] &&
                   o[0][r].a[i] == o[1][r].a[i];
            if (runs[r].forced >= 0) {
              worst = std::max(worst, ratio(o[0][r], i, runs[r].forced));
            }
          }
        }
        // Overlapped and in line, the same arithmetic.
        bool lag_same = true;
        if (lag == 1) {
          lagged = o[0];
        } else {
          for (std::size_t r = 0; r < runs.size(); ++r) {
            lag_same = lag_same && lagged[r].v == o[0][r].v &&
                       lagged[r].a == o[0][r].a;
          }
        }
        std::uint64_t samples = 0;
        for (const Run& rr : runs) { samples += rr.batch.size(); }
        const std::uint64_t want_x =
            (std::uint64_t)(2 * cfg.layers.size()) * samples;
        const std::uint64_t got_x = ch[0]->exchanges.load();
        const int oos = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
        std::printf("[magi2] split %-6s experts %-8s %-8s: heads %d/%d, "
                    "experts %d/%d a head; ranks %s; worst %.2fx ref bf16 "
                    "from fp32 (bar %.1f; unsplit %.2fx)%s; %llu of %llu "
                    "exchanges, out of step %d, mismatched %d; overlapped "
                    "%d; held %s\n",
                    sa.name, resident == 0 ? "streamed" : "held",
                    lag ? "lagged" : "in line", sa.h, cfg.n_q - sa.h, sa.f,
                    cfg.moe_experts - sa.f, same ? "IDENTICAL" : "DIFFER",
                    worst, sa.accel ? 5.0 : 1.5, whole_worst,
                    lag == 0 ? (lag_same ? "; = overlapped" : "; != OVERLAPPED")
                             : "",
                    (unsigned long long)got_x, (unsigned long long)want_x,
                    oos, link->mismatched.load(),
                    (int)m[0]->tp_overlapped(),
                    release ? "(a subset: unchecked)"
                    : booked ? "as booked"
                             : "NOT as booked");
        const bool engaged =
            !sa.accel || (m[0]->i8_gemms() > 0 && m[0]->sage_calls() > 0 &&
                          m[0]->sage_pv8_calls() > 0);
        if (sa.accel) {
          std::printf("[magi2]   accel: %ld int8 GEMMs, %ld Sage calls "
                      "(%ld with the int8 P*V) on rank 0\n",
                      m[0]->i8_gemms(), m[0]->sage_calls(),
                      m[0]->sage_pv8_calls());
        }
        check(same, "ranks identical");
        check(engaged, "the tiers engaged");
        check(worst <= (sa.accel ? 5.0 : 1.5), "within the bar");
        check(lag_same, "overlapped == in line");
        check(got_x == want_x, "the exchanges");
        check(oos == 0 && link->mismatched.load() == 0, "in step");
        check(m[0]->tp_overlapped() == (lag == 1), "overlap as asked");
        check(booked, "held as booked");
      }
    }
  }
  ::unsetenv("VPIPE_MAGI2_TP_LAG");
  return pass;
}

}  // namespace

TEST(magi2, a_split_forward_is_the_unsplit_one)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  EXPECT_TRUE(check_preview_split_(mc, ws, g, cfg, false));
}

// ...ON THE RELEASE'S WEIGHTS: its layer subset (the golden's: dense and
// MoE layers), one even share, its experts streamed -- the same checks.
TEST(magi2, a_split_release_is_the_unsplit_one)
{
  const char* g = std::getenv("VPIPE_MAGI2_REAL_GOLDEN");
  const char* m = std::getenv("VPIPE_MAGI2_TEST_MODEL_PATH");
  if (g == nullptr || m == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const FlexData meta = read_json(fs::path(g) / "t2v" / "meta.json");
  ASSERT_TRUE(meta.is_object());
  std::vector<int> layers;
  const FlexData layers_fd = meta.as_object().at("layers");
  auto la = layers_fd.as_array();
  for (std::size_t i = 0; i < la.size(); ++i) {
    layers.push_back((int)la.at(i).as_int(0));
  }
  auto ws = genai::WeightSet::open(m, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(
      genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err, layers));
  EXPECT_TRUE(check_preview_split_(mc, ws, g, cfg, true));
}

// THE REFINER SPLIT (tensor parallel), as the preview's: two ranks of the
// wide refiner golden (two kv groups: heads 4 / 4) over an in-process
// channel, its MLP rows even and uneven (a quarter), run at once on two
// threads in bands of 1024 rows so a layer has several -- overlapped
// (each band's exchanges under the next band's work) and in line
// (VPIPE_MAGI2_TP_LAG=0). Every pair: its ranks BIT-IDENTICAL, overlapped
// and in line the same bits, within the unsplit test's bar of the fp32
// reference (1.5x the release's own bf16 distance), every exchange in
// step, each rank holding what weight_bytes books for its share.
TEST(magi2, a_split_refiner_is_the_unsplit_one)
{
  const char* g = std::getenv("VPIPE_MAGI2_REFINER_WIDE_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Refiner::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg, &err));
  RefinerCase rc;
  ASSERT_TRUE(load_refiner_case(fs::path(g) / "release_blocks", &cfg, &rc));
  cfg.mlp_rows = 1024;
  double whole = 0.0;
  {
    auto m = genai::MetalMagi2Refiner::load(ws, mc, cfg, &err);
    ASSERT_TRUE(m != nullptr);
    std::vector<float> v;
    ASSERT_TRUE(m->forward(rc.s, &v, &err));
    whole = rel(v, rc.v32) / rc.bar;
  }
  ASSERT_TRUE(cfg.n_kv >= 2);
  const int h = cfg.n_q / 2;
  // ...and on matrix cores the int8 token tier and SageAttention with its
  // int8 P*V on the rank's widths (the tiers' own bar).
  struct RowArm {
    const char* name;
    int f;
    bool accel = false;
  };
  std::vector<RowArm> arms = {{"even", cfg.mlp_inter / 2},
                              {"uneven", cfg.mlp_inter / 4}};
  if (mc->supports_matrix_cores()) {
    arms.push_back({"accel", cfg.mlp_inter / 2, true});
  }
  for (const RowArm& ra : arms) {
    std::vector<float> lagged;
    for (int lag : {1, 0}) {
      if (lag == 0) {
        ::setenv("VPIPE_MAGI2_TP_LAG", "0", 1);
      } else {
        ::unsetenv("VPIPE_MAGI2_TP_LAG");
      }
      std::unique_ptr<genai::MetalMagi2Refiner> m[2];
      bool booked = true;
      for (int r = 0; r < 2; ++r) {
        genai::MetalMagi2Refiner::Config c = cfg;
        c.tp = genai::TpShare{};
        c.tp.rank = r;
        c.tp.ranks = 2;
        c.tp.n_heads = cfg.n_q;
        c.tp.n_ffn = cfg.mlp_inter;
        c.tp.head0 = r == 0 ? 0 : h;
        c.tp.head1 = r == 0 ? h : cfg.n_q;
        c.tp.ffn0 = r == 0 ? 0 : ra.f;
        c.tp.ffn1 = r == 0 ? ra.f : cfg.mlp_inter;
        if (ra.accel) {
          c.i8_tier = genai::magi2::effective_i8_tier(2, true);
          c.sage.enabled = true;
          c.sage.pv8 = true;
        }
        m[r] = genai::MetalMagi2Refiner::load(ws, mc, c, &err);
        if (m[r] == nullptr) {
          std::printf("[magi2] refiner rank %d: %s\n", r, err.c_str());
        }
        ASSERT_TRUE(m[r] != nullptr);
        if (m[r] == nullptr) { return; }
        booked = booked && m[r]->resident_bytes() ==
                               genai::MetalMagi2Refiner::weight_bytes(*ws, c);
      }
      auto link = std::make_shared<Magi2PairLink>();
      std::shared_ptr<Magi2PairChannel> ch[2];
      for (int r = 0; r < 2; ++r) {
        ch[r] = std::make_shared<Magi2PairChannel>(link, r);
        std::string be;
        ASSERT_TRUE(m[r]->begin_tp_generation(ch[r], {}, 0, &be));
      }
      std::vector<float> v[2];
      std::string e1;
      bool ok1 = false;
      std::thread t([&]() { ok1 = m[1]->forward(rc.s, &v[1], &e1); });
      std::string e0;
      const bool ok0 = m[0]->forward(rc.s, &v[0], &e0);
      t.join();
      for (int r = 0; r < 2; ++r) { m[r]->end_tp_generation(); }
      if (!ok0 || !ok1) {
        std::printf("[magi2] split refiner: %s | %s\n", e0.c_str(),
                    e1.c_str());
      }
      ASSERT_TRUE(ok0 && ok1);
      if (!ok0 || !ok1) { return; }
      const double r = rel(v[0], rc.v32) / rc.bar;
      bool lag_same = true;
      if (lag == 1) {
        lagged = v[0];
      } else {
        lag_same = lagged == v[0];
      }
      const int oos = ch[0]->out_of_step.load() + ch[1]->out_of_step.load();
      std::printf("[magi2] split refiner %-6s %-8s: heads %d/%d, MLP rows "
                  "%d/%d; ranks %s; %.2fx ref bf16 from fp32 (bar %.1f; "
                  "unsplit %.2fx)%s; %llu exchanges, out of step %d, "
                  "mismatched %d; overlapped %d; held %s\n",
                  ra.name, lag ? "lagged" : "in line", h, cfg.n_q - h, ra.f,
                  cfg.mlp_inter - ra.f, v[0] == v[1] ? "IDENTICAL" : "DIFFER",
                  r, ra.accel ? 5.0 : 1.5, whole,
                  lag == 0 ? (lag_same ? "; = overlapped" : "; != OVERLAPPED")
                           : "",
                  (unsigned long long)ch[0]->exchanges.load(), oos,
                  link->mismatched.load(), (int)m[0]->tp_overlapped(),
                  booked ? "as booked" : "NOT as booked");
      if (ra.accel) {
        std::printf("[magi2]   accel: %ld int8 GEMMs, %ld Sage calls (%ld "
                    "with the int8 P*V) on rank 0\n", m[0]->i8_gemms(),
                    m[0]->sage_calls(), m[0]->sage_pv8_calls());
        EXPECT_TRUE(m[0]->i8_gemms() > 0 && m[0]->sage_pv8_calls() > 0);
      }
      EXPECT_TRUE(v[0] == v[1]);
      EXPECT_TRUE(r <= (ra.accel ? 5.0 : 1.5));
      EXPECT_TRUE(lag_same);
      EXPECT_TRUE(ch[0]->exchanges.load() > 2 * cfg.layers.size());
      EXPECT_TRUE(oos == 0 && link->mismatched.load() == 0);
      EXPECT_TRUE(m[0]->tp_overlapped() == (lag == 1));
      EXPECT_TRUE(booked);
    }
  }
  ::unsetenv("VPIPE_MAGI2_TP_LAG");
}

// THE REFINER'S SHARE AT THE PREVIEW'S FRACTIONS (a stage splits by the
// preview DiT's widths): the release's shapes -- 24 query heads and 256
// experts a MoE head, against the refiner's 32 heads in 8 kv groups and
// 10920 MLP rows -- each boundary scaled on its own, the heads' to whole
// kv groups, so two ranks' shares tile the refiner as theirs tile the
// preview; an inactive share stays inactive.
TEST(magi2, a_refiners_share_follows_the_previews)
{
  genai::MetalMagi2Refiner::Config rc;
  rc.n_q = 32;
  rc.n_kv = 8;
  rc.mlp_inter = 10920;
  auto preview = [](int r, int h, int f) {
    genai::TpShare t;
    t.rank = r;
    t.ranks = 2;
    t.n_heads = 24;
    t.n_ffn = 256;
    t.head0 = r == 0 ? 0 : h;
    t.head1 = r == 0 ? h : 24;
    t.ffn0 = r == 0 ? 0 : f;
    t.ffn1 = r == 0 ? f : 256;
    return t;
  };
  struct Cut {
    int h, f;
    int rh, rf;               // the refiner's boundary
  };
  const Cut cuts[] = {{12, 128, 16, 5460}, {6, 64, 8, 2730},
                      {7, 100, 8, 4265}};
  bool ok = true;
  for (const Cut& ct : cuts) {
    const genai::TpShare a =
        genai::MetalMagi2Refiner::share_at(rc, preview(0, ct.h, ct.f));
    const genai::TpShare b =
        genai::MetalMagi2Refiner::share_at(rc, preview(1, ct.h, ct.f));
    genai::MetalMagi2Refiner::Config ca = rc, cb = rc;
    ca.tp = a;
    cb.tp = b;
    genai::MetalMagi2Refiner::Share sa, sb;
    const bool fit = genai::MetalMagi2Refiner::share_of(ca, &sa, nullptr) &&
                     genai::MetalMagi2Refiner::share_of(cb, &sb, nullptr);
    const bool tile = a.head0 == 0 && a.head1 == ct.rh && b.head0 == ct.rh &&
                      b.head1 == 32 && a.ffn0 == 0 && a.ffn1 == ct.rf &&
                      b.ffn0 == ct.rf && b.ffn1 == 10920;
    std::printf("[magi2] preview heads %d / experts %d -> refiner heads %d "
                "/ rows %d (%s, %s)\n", ct.h, ct.f, a.head1, a.ffn1,
                fit ? "fits" : "DOES NOT FIT", tile ? "tiles" : "WRONG");
    ok = ok && fit && tile;
  }
  EXPECT_TRUE(ok);
  EXPECT_TRUE(!genai::MetalMagi2Refiner::share_at(rc, genai::TpShare{})
                   .active());
}

// ---- the achieved-FLOP/s meter ----------------------------------------------
//
// Every GPU part of MAGI-2 credits common/perf-flops.h: counted as it is
// encoded, credited as its command buffers retire. Two things are held:
// EVERYTHING COUNTED IS FLUSHED AND CREDITED -- a run leaves nothing
// pending (a commit without a flush ahead of it would) and the meter's GPU
// total grows by exactly what the model flushed -- and for the two DiTs
// THE COUNT IS THE MODEL'S: it equals the arithmetic derived here from the
// config, 2MNK a projection, 4 x pairs x d an attention head, and a routed
// expert pick's three [256 x expert_inter] products.

namespace {

namespace pf = vpipe::perf_flops;

// What a run flushed and the meter was credited since construction; waits
// for the last buffer's handler, which may run after the wait that saw it.
struct MeterSince {
  std::uint64_t g0 = 0, f0 = 0;
  explicit MeterSince(const genai::GpuFlopsTally& t)
    : g0(pf::totals(pf::kGpu).flops), f0(t.flushed())
  {
  }
  std::pair<std::uint64_t, std::uint64_t>
  take(const genai::GpuFlopsTally& t) const
  {
    const std::uint64_t f = t.flushed() - f0;
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0 < f; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return {f, pf::totals(pf::kGpu).flops - g0};
  }
};

// One preview forward of `s`, off the config alone.
std::uint64_t
preview_flops_(const genai::MetalMagi2Dit::Config& c,
               const genai::MetalMagi2Dit::Sample& s)
{
  using genai::gemm_flops;
  const long long H = c.hidden, W4 = H * c.streams;
  const long long nh = 2LL * c.streams + (long long)c.streams * c.streams;
  const long long D = c.head_dim, qs = (long long)c.n_q * D;
  const long long ldqkv = qs + 2LL * c.n_kv * D;
  const long long nvg = (long long)s.T * s.H * s.W;
  const long long nsp = (long long)s.images.size();
  // The embedders: video, each image, audio, text, the special tokens.
  std::uint64_t f = gemm_flops(nvg, W4, c.video_c) +
                    gemm_flops(s.audio_len, W4, c.audio_c) +
                    gemm_flops(s.text_len, W4, c.text_c) +
                    gemm_flops(nsp, W4, c.text_c);
  long long nimg = 0;
  for (const auto& im : s.images) {
    nimg += (long long)im.h * im.w;
    f += gemm_flops((long long)im.h * im.w, W4, c.video_c);
  }
  const long long n = nvg + nimg + s.audio_len + s.text_len + nsp;
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    // Two hyper-connections' dots; the attention: q|k|v, the gate, the
    // heads' n x n pairs, the output projection.
    f += 2 * gemm_flops(n, nh, W4) + gemm_flops(n, ldqkv, H) +
         gemm_flops(n, c.n_q, H) + genai::attention_flops(c.n_q, n * n, D) +
         gemm_flops(n, H, qs);
    if (c.moe[i] != 0) {
      // Split and merge, each head's router, every pick, the shared and
      // modality experts.
      f += 2 * gemm_flops(n, H, H) +
           (std::uint64_t)c.moe_heads * gemm_flops(n, c.moe_experts, 256) +
           (std::uint64_t)(n * c.moe_heads * c.top_k) * 3 *
               gemm_flops(1, c.expert_inter, 256) +
           gemm_flops(n, 2LL * c.shared_inter, H) +
           gemm_flops(n, H, c.shared_inter) +
           gemm_flops(n, 2LL * c.ms_inter, H) +
           gemm_flops(n, H, c.ms_inter);
    } else {
      f += gemm_flops(n, 2LL * c.mlp_inter, H) +
           gemm_flops(n, H, c.mlp_inter);
    }
  }
  // The heads.
  return f + gemm_flops(nvg, c.video_c, W4) +
         gemm_flops(s.audio_len, c.audio_c, W4);
}

}  // namespace

// THE PREVIEW DiT on the small golden: a batch of two (held experts) and
// single forwards with every MoE layer streamed (a commit a layer) --
// counted == derived == credited, nothing pending.
TEST(magi2, the_preview_dit_counts_its_flops)
{
  const char* g = std::getenv("VPIPE_MAGI2_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Dit::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Dit::config_from_weights(*ws, &cfg, &err));
  Case cs[2];
  std::vector<genai::MetalMagi2Dit::Sample> smp;
  const char* names[2] = {"t2v", "i2v"};
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(load_case(fs::path(g) / names[i], &cs[i]));
    smp.push_back(sample_of(cs[i], cfg));
  }
  ASSERT_TRUE(!smp[1].images.empty());
  const std::uint64_t want[2] = {preview_flops_(cfg, smp[0]),
                                 preview_flops_(cfg, smp[1])};
  for (int resident : {-1, 0}) {
    cfg.resident_moe = resident;
    auto m = genai::MetalMagi2Dit::load(ws, mc, cfg, &err);
    if (!m) { std::printf("[magi2] %s\n", err.c_str()); }
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::uint64_t f = 0, gc = 0, w = 0;
    if (resident < 0) {
      const MeterSince ms(m->gpu_flops());
      std::vector<std::vector<float>> bv, ba;
      ASSERT_TRUE(m->forward(
          std::vector<const genai::MetalMagi2Dit::Sample*>{&smp[0],
                                                           &smp[1]},
          &bv, &ba, &err));
      std::tie(f, gc) = ms.take(m->gpu_flops());
      w = want[0] + want[1];
    } else {
      for (int i = 0; i < 2; ++i) {
        const MeterSince ms(m->gpu_flops());
        std::vector<float> v, a;
        ASSERT_TRUE(m->forward(smp[(std::size_t)i], &v, &a, &err));
        const auto [fi, gi] = ms.take(m->gpu_flops());
        f += fi;
        gc += gi;
        w += want[i];
      }
    }
    std::printf("[magi2] preview DiT, experts %s: counted %.6f GFLOP, "
                "derived %.6f, credited %.6f\n",
                resident < 0 ? "held (a batch of 2)" : "streamed", f / 1e9,
                w / 1e9, gc / 1e9);
    EXPECT_TRUE(f == w);
    EXPECT_TRUE(gc == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
}

// THE REFINER on the small golden: counted == derived from the config and
// the window's tiles (each query tile's rows times its key blocks' keys,
// as the kernel visits them) == credited. Seq::pairs is that sum, and with
// a radius past the grid -- every tile every block -- the whole n x n.
TEST(magi2, the_refiner_counts_its_flops)
{
  const char* g = std::getenv("VPIPE_MAGI2_REFINER_GOLDEN");
  if (g == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto ws = genai::WeightSet::open(g, nullptr);
  ASSERT_TRUE(ws != nullptr);
  genai::MetalMagi2Refiner::Config cfg;
  std::string err;
  ASSERT_TRUE(genai::MetalMagi2Refiner::config_from_weights(*ws, &cfg, &err));
  const fs::path d = fs::path(g) / "smallblocks";
  const FlexData meta = read_json(d / "meta.json");
  ASSERT_TRUE(meta.is_object());
  if (!meta.is_object()) { return; }
  auto o = meta.as_object();
  const FlexData blk = o.at("block");
  const FlexData rad = o.at("radius");
  for (int a = 0; a < 3; ++a) {
    cfg.block[a] = (int)blk.as_array()[(std::size_t)a].as_int(0);
    cfg.radius[a] = (int)rad.as_array()[(std::size_t)a].as_int(0);
  }
  const std::vector<float> video = read_f32(d / "video.bin");
  const std::vector<float> text = read_f32(d / "text.bin");
  genai::MetalMagi2Refiner::Sample s;
  s.T = (int)o.at("T").as_int(0);
  s.H = (int)o.at("H").as_int(0);
  s.W = (int)o.at("W").as_int(0);
  s.video = video.data();
  s.text_len = (int)o.at("text").as_int(0);
  s.text = text.data();
  auto m = genai::MetalMagi2Refiner::load(ws, mc, cfg, &err);
  if (!m) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const bool nax = m->uses_nax_attention();
  const int bq = nax ? 64 : 32, bk = nax ? 32 : 16;
  const auto q = genai::MetalMagi2Refiner::sequence(cfg, s.T, s.H, s.W,
                                                    s.text_len, bq, bk);
  long long pairs = 0;
  for (std::size_t t = 0; t + 1 < q.qb_off.size(); ++t) {
    const long long rows = std::min<long long>(bq, q.n - (long long)t * bq);
    for (int j = q.qb_off[t]; j < q.qb_off[t + 1]; ++j) {
      pairs += rows * std::min<long long>(
                          bk, q.n - (long long)q.qb_blocks[(std::size_t)j] *
                                        bk);
    }
  }
  EXPECT_TRUE(pairs == q.pairs && pairs < (long long)q.n * q.n);
  {
    genai::MetalMagi2Refiner::Config dc = cfg;
    dc.radius[0] = dc.radius[1] = dc.radius[2] = 1 << 20;
    const auto dq = genai::MetalMagi2Refiner::sequence(dc, s.T, s.H, s.W,
                                                       s.text_len, bq, bk);
    EXPECT_TRUE(dq.pairs == (long long)dq.n * dq.n);
  }
  using genai::gemm_flops;
  const long long n = q.n, H = cfg.hidden, D = cfg.head_dim;
  const long long qs = (long long)cfg.n_q * D;
  const long long ldqkv = qs + 2LL * cfg.n_kv * D;
  const long long I = cfg.mlp_inter;
  const std::uint64_t layer =
      gemm_flops(n, ldqkv, H) + gemm_flops(n, cfg.n_q, H) +
      genai::attention_flops(cfg.n_q, pairs, D) + gemm_flops(n, H, qs) +
      gemm_flops(n, 2 * I, H) + gemm_flops(n, H, I);
  const std::uint64_t want =
      gemm_flops(q.nv_pad, H, cfg.video_c) +
      gemm_flops(n - q.nv_pad, H, cfg.text_c) +
      (std::uint64_t)cfg.layers.size() * layer +
      gemm_flops(q.nv_pad, cfg.video_c, H);
  const MeterSince ms(m->gpu_flops());
  std::vector<float> v;
  ASSERT_TRUE(m->forward(s, &v, &err));
  const auto [f, gc] = ms.take(m->gpu_flops());
  std::printf("[magi2] refiner %d x %d x %d + %d text, %lld slots (%s "
              "tiles): window %.1f%% of n x n; counted %.6f GFLOP, derived "
              "%.6f, credited %.6f\n", s.T, s.H, s.W, s.text_len, n,
              nax ? "64/32" : "32/16", 100.0 * (double)pairs / (n * n),
              f / 1e9, want / 1e9, gc / 1e9);
  EXPECT_TRUE(f == want);
  EXPECT_TRUE(gc == f);
  EXPECT_TRUE(m->gpu_flops().pending() == 0);
}

// THE TURBO DECODER credits what it counts, over two windows in time; and
// it counts what it RUNS: 2 x 2 tiles of an 8 x 8 plane, each clipped to
// the whole plane by its halo, decode it four times and count four times.
TEST(magi2, the_turbo_decoder_credits_what_it_counts)
{
  const char* dir = std::getenv("VPIPE_MAGI2_TURBO_DIR");
  if (dir == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto dec = genai::MetalMagi2Turbo::load(dir, mc, &err);
  if (!dec) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dec != nullptr);
  if (dec == nullptr) { return; }
  const int T = 9, h = 8, w = 8;
  const std::vector<float> z((std::size_t)48 * T * h * w, 0.0f);
  std::uint64_t whole = 0;
  for (int tiles : {1, 2}) {
    dec->set_tiles(tiles == 1 ? 0 : 4, tiles == 1 ? 0 : 4, -1);
    const auto t = dec->tiling(T, h, w);
    ASSERT_TRUE(t.ny == tiles && t.nx == tiles);
    const MeterSince ms(dec->gpu_flops());
    std::vector<float> out;
    ASSERT_TRUE(dec->decode(z.data(), T, h, w, &out, &err));
    const auto [f, gc] = ms.take(dec->gpu_flops());
    if (tiles == 1) { whole = f; }
    std::printf("[magi2] turbo decode %d x %d x %d, %d x %d tiles: counted "
                "%.4f GFLOP, credited %.4f\n", T, h, w, tiles, tiles,
                f / 1e9, gc / 1e9);
    EXPECT_TRUE(f > 0 && gc == f);
    EXPECT_TRUE(dec->gpu_flops().pending() == 0);
    if (tiles == 2) { EXPECT_TRUE(f == 4 * whole); }
  }
  dec->set_tiles(0, 0, -1);
}

// THE SOUNDTRACK'S DECODER (Stable Audio Open on the Oobleck class, as
// MAGI-2 loads it) credits what it counts.
TEST(magi2, the_soundtrack_decoder_credits_what_it_counts)
{
  const char* d = std::getenv("VPIPE_MAGI2_AUDIO_DIR");
  if (d == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  genai::MetalOobleckDecoder::Config cfg;
  ASSERT_TRUE(genai::MetalOobleckDecoder::config_from_dir(d, &cfg, &err));
  auto dec = genai::MetalOobleckDecoder::load(
      genai::WeightSet::open(d, nullptr), mc, cfg, &err);
  if (!dec) { std::printf("[magi2] %s\n", err.c_str()); }
  ASSERT_TRUE(dec != nullptr);
  if (dec == nullptr) { return; }
  // Longer than a tile, so more than one commit carries the count.
  const int L = dec->core_frames() + 8;
  const std::vector<float> z((std::size_t)cfg.latent_dim * L, 0.0f);
  const MeterSince ms(dec->gpu_flops());
  std::vector<float> pcm;
  ASSERT_TRUE(dec->decode(z.data(), L, &pcm, /*clamp=*/false, {}, &err));
  const auto [f, gc] = ms.take(dec->gpu_flops());
  std::printf("[magi2] SAO decode %d latents: counted %.4f GFLOP, credited "
              "%.4f\n", L, f / 1e9, gc / 1e9);
  EXPECT_TRUE(f > 0 && gc == f);
  EXPECT_TRUE(dec->gpu_flops().pending() == 0);
}
