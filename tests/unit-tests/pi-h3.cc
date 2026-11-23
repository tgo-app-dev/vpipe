// pi-h3.cc -- pi-H3 (Lakonik/pi-H3) on MiniMax-H3: its DWD pixel head
// against the restatement golden (tools/dump_pi_h3_golden.py), on the real
// adapter's weights.
//
//   VPIPE_PI_H3_DWD_ADAPTER  a DWD adapter's directory or file
//                            (pi_h3_4step_dwd_adapter/)
//   VPIPE_PI_H3_GOLDEN       the dumper's `dwd` output
//
// The overwrites, the LoRA and the sampling contract are held against a
// patched DiT in minimax_h3_pruned.pi_flow_overwrites_are_the_patched_file
// (self-contained).

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/minimax-h3/metal-minimax-h3-fast-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/weight-set.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-claim-delivery.h"
#include "pipeline/resource-plan.h"
#include "stages/audio-vae-decode-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/model-memory.h"
#include "stages/model-select-stage.h"
#include "stages/vae-decode-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
namespace fs = std::filesystem;

namespace {

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr ? std::string(v) : std::string();
}

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return {}; }
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> v((std::size_t)n / 4);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

FlexData
read_json_(const std::string& path)
{
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return FlexData::from_json(ss.str());
}

std::uint16_t
bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::string
adapter_file_()
{
  std::string p = env_("VPIPE_PI_H3_DWD_ADAPTER");
  std::error_code ec;
  if (!p.empty() && fs::is_directory(p, ec)) {
    p = (fs::path(p) / "diffusion_pytorch_model.safetensors").string();
  }
  return p;
}

}  // namespace

// The head on the golden's features, at two geometries: 7 latents (a
// native block and a 2-latent one, so a second first-frame head) at 6 x 9
// DiT tokens, and 5 (one whole block) at 5 x 7. The bar is the
// restatement's own bf16. And the same features after three rows of
// conditioning, decoded from row 3: identical.
TEST(pi_h3, dwd_head_matches_reference)
{
  const std::string file = adapter_file_();
  const std::string gold = env_("VPIPE_PI_H3_GOLDEN");
  if (file.empty() || gold.empty()) {
    std::printf("  SKIP: VPIPE_PI_H3_DWD_ADAPTER / VPIPE_PI_H3_GOLDEN "
                "unset\n");
    return;
  }
  const FlexData meta = read_json_(gold + "/dwd.json");
  if (!meta.is_object()) {
    std::printf("  SKIP: no dwd.json (run tools/dump_pi_h3_golden.py dwd)\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  auto ws = WeightSet::open(file, nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  MetalMiniMaxH3FastVae::Config cfg;
  std::string err;
  const bool cfg_ok = MetalMiniMaxH3FastVae::config_from_dwd(*ws, &cfg, &err);
  ASSERT_TRUE(cfg_ok);
  if (!cfg_ok) { std::printf("  config: %s\n", err.c_str()); return; }
  EXPECT_TRUE(cfg.dwd() && cfg.n_layers == 8 && cfg.dim == 2048 &&
              cfg.ffn_inner == 8192 && cfg.feature_dim == 5376);
  auto head = MetalMiniMaxH3FastVae::load(ws, mc, cfg, &err);
  ASSERT_TRUE(head != nullptr);
  if (!head) { std::printf("  load: %s\n", err.c_str()); return; }
  // A latent is not what it decodes.
  {
    std::vector<float> z(24 * 1 * 2 * 2, 0.0f);
    std::string zerr;
    EXPECT_TRUE(head->decode(z.data(), 1, 2, 2, &zerr).empty() &&
                !zerr.empty());
  }
  FlexData cs = meta.as_object().at("cases");
  int ran = 0;
  for (const FlexData& c : cs.as_array()) {
    FlexData cc = c;
    auto o = cc.as_object();
    const std::string name(o.at("name").as_string(""));
    const int T = (int)o.at("T").as_int(0);
    const int h2 = (int)o.at("h2").as_int(0);
    const int w2 = (int)o.at("w2").as_int(0);
    const int frames = (int)o.at("frames").as_int(0);
    const double bar = o.at("bf16_rel_l2").as_real(0.0);
    const std::vector<float> x = read_f32_(gold + "/dwd_" + name +
                                           "_feats.f32");
    const std::vector<float> ref = read_f32_(gold + "/dwd_" + name +
                                             "_out.f32");
    const std::size_t Fd = (std::size_t)cfg.feature_dim;
    const std::size_t nd = (std::size_t)T * h2 * w2;
    ASSERT_TRUE(x.size() == nd * Fd);
    if (x.size() != nd * Fd) { continue; }
    // As the DiT hands them over, and again behind 3 conditioning rows.
    const std::size_t lead = 3;
    SharedBuffer fb = mc->make_shared_buffer(nd * Fd * 2);
    SharedBuffer fl = mc->make_shared_buffer((nd + lead) * Fd * 2);
    auto* p = static_cast<std::uint16_t*>(fb.contents());
    auto* q = static_cast<std::uint16_t*>(fl.contents());
    for (std::size_t i = 0; i < lead * Fd; ++i) { q[i] = bf16_(7.0f); }
    for (std::size_t i = 0; i < x.size(); ++i) {
      p[i] = bf16_(x[i]);
      q[lead * Fd + i] = p[i];
    }
    const SharedBuffer out = head->decode_features(fb, 0, T, h2, w2, &err);
    const SharedBuffer outl =
        head->decode_features(fl, lead, T, h2, w2, &err);
    ASSERT_TRUE(!out.empty() && !outl.empty());
    if (out.empty() || outl.empty()) {
      std::printf("  decode: %s\n", err.c_str());
      continue;
    }
    EXPECT_TRUE(head->decoded_frames(T) == frames);
    const std::size_t n = out.byte_size() / 2;
    ASSERT_TRUE(n == ref.size());
    if (n != ref.size()) { continue; }
    const auto* g = static_cast<const std::uint16_t*>(out.contents());
    double d = 0.0, s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double e = (double)f32_(g[i]) - ref[i];
      d += e * e;
      s += (double)ref[i] * ref[i];
    }
    const double rel = std::sqrt(d / std::max(s, 1e-30));
    // The first-frame head's frames alone: every native block's first.
    double d1 = 0.0, s1 = 0.0;
    const std::size_t plane = (std::size_t)h2 * 32 * w2 * 32;
    for (int t = 0; t < T; t += cfg.block) {
      const int fo = t * cfg.patch_t + cfg.frames_to_trim -
                     cfg.frames_to_trim * (t / cfg.block + 1);
      for (int ch = 0; ch < 3; ++ch) {
        const std::size_t base =
            ((std::size_t)ch * frames + (std::size_t)fo) * plane;
        for (std::size_t i = 0; i < plane; ++i) {
          const double e = (double)f32_(g[base + i]) - ref[base + i];
          d1 += e * e;
          s1 += (double)ref[base + i] * ref[base + i];
        }
      }
    }
    const double rel1 = std::sqrt(d1 / std::max(s1, 1e-30));
    const bool same = std::memcmp(out.contents(), outl.contents(),
                                  out.byte_size()) == 0;
    std::printf("[pi_h3] DWD head '%s' [%d x %d x %d] -> %d frames: rel-L2 "
                "%.5f from fp32 (the reference's bf16 %.5f), first frames "
                "%.5f; from row %zu %s\n", name.c_str(), T, h2, w2, frames,
                rel, bar, rel1, lead, same ? "IDENTICAL" : "DIFFERENT");
    EXPECT_TRUE(rel < 1.25 * bar && rel1 < 2.0 * bar);
    EXPECT_TRUE(same);
    ++ran;
  }
  EXPECT_TRUE(ran == 2);
}

// A DWD GRAPH'S PLAN. Its clip is decoded by the adapter's pixel head in
// vae-decode, and the video VAE is never loaded -- so the plan claims the
// head (its weights, vae-decode's decode-phase arena), books its working
// set and the clip as the decode arena, and the DiT's features as the
// payload the latent was; and claims no VAE. The same FL2VA graph with
// the plain 4-step adapter books the VAE's decoder half, its working
// set and the latent, and the decode phase moves by exactly the terms
// swapped (the DiT and the adapters are the denoise's under `always`).
// Configs and headers only; nothing loads.
//   VPIPE_MINIMAX_H3_TEST_MODEL_PATH  an FL2VA root (encoder, VAEs)
//   VPIPE_PI_H3_BASE                  the pruned DiT file
//   VPIPE_PI_H3_DWD_ADAPTER, VPIPE_PI_H3_ADAPTER (the plain 4-step one)
TEST(pi_h3, a_dwd_graph_books_its_head_not_the_vae)
{
  const std::string root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  const std::string base = env_("VPIPE_PI_H3_BASE");
  const std::string dwd = adapter_file_();
  std::string plain = env_("VPIPE_PI_H3_ADAPTER");
  std::error_code ec;
  if (!plain.empty() && fs::is_directory(plain, ec)) {
    plain = (fs::path(plain) / "diffusion_pytorch_model.safetensors")
                .string();
  }
  if (root.empty() || base.empty() || dwd.empty() || plain.empty()) {
    std::printf("  SKIP: VPIPE_MINIMAX_H3_TEST_MODEL_PATH / VPIPE_PI_H3_BASE "
                "/ VPIPE_PI_H3_DWD_ADAPTER / VPIPE_PI_H3_ADAPTER unset\n");
    return;
  }
  constexpr unsigned kModelCfgPort = 9;   // generate-video's iport 9
  const int W = 832, Hpx = 480, FR = 73;
  // A scratch / payload claim's bytes (key `label|bytes`), 0 for none.
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
  struct Plan {
    std::size_t dec = 0, head = 0, arena = 0, payload = 0;
    std::vector<std::string> vae_weights;   // vae-decode's weight claims
    std::string asked;                      // what vae-decode was told
    // The declared (report) ledger's holdings, which merge by source.
    std::vector<std::string> gen_src, dec_src;
  };
  auto plan = [&](const std::string& adapter) {
    Plan out;
    Session sess;
    auto* mgr = sess.generative_model_manager();
    if (mgr == nullptr) { return out; }
    auto cfg = [&](std::vector<std::pair<std::string, FlexData>> kv) {
      FlexData c = FlexData::make_object();
      c.as_object().insert("hf_dir", FlexData::make_string(root));
      for (auto& [k, v] : kv) { c.as_object().insert(k, std::move(v)); }
      return c;
    };
    ModelSelectStage sel(&sess, "sel", std::vector<InEdge>{}, cfg({}));
    GenerateVideoStage gen(
        &sess, "gen", std::vector<InEdge>{},
        cfg({{"width", FlexData::make_int(W)},
             {"height", FlexData::make_int(Hpx)},
             {"frames", FlexData::make_int(FR)},
             {"fps", FlexData::make_real(24.0)},
             {"dit_dir", FlexData::make_string(base)},
             {"unload_when_idle", FlexData::make_string("always")}}));
    FlexData mc = FlexData::make_object();
    mc.as_object().insert("lora", FlexData::make_string(adapter));
    mc.as_object().insert("lora_scale", FlexData::make_real(1.0));
    gen.apply_constant(kModelCfgPort, mc);
    VaeDecodeStage dec(&sess, "dec",
                       std::vector<InEdge>{{&gen, 0}, {&sel, 0}}, cfg({}));
    AudioVaeDecodeStage adec(&sess, "adec", std::vector<InEdge>{}, cfg({}));
    const std::vector<Stage*> stages = {&sel, &gen, &dec, &adec};
    const auto planners = ResourcePlannerRegistry::get().all();
    for (ResourcePlanner* p : planners) { p->begin_plan(&sess); }
    for (Stage* st : stages) {
      for (const ResourceClaim& c : st->declare_resources()) {
        ResourcePlanner* p = ResourcePlannerRegistry::get().find(c.kind);
        if (c.key.empty() || p == nullptr) { continue; }
        DeliveringResourceClaim delivering(c);
        p->claim(&sess, c.key, c.phase, c.last_phase, c.floor_bytes);
      }
    }
    for (Stage* st : stages) {
      for (const ResourceClaim& c : st->decide_resources()) {
        ResourcePlanner* p = ResourcePlannerRegistry::get().find(c.kind);
        if (c.key.empty() || p == nullptr) { continue; }
        p->decide(&sess, c.key, c.phase);
      }
    }
    for (ResourcePlanner* p : planners) { p->end_plan(&sess); }
    std::vector<std::pair<std::string, std::size_t>> by;
    mgr->phase_peak(&by);
    for (const auto& [ph, b] : by) {
      if (ph == model_memory::kPhaseDecode) { out.dec = b; }
    }
    const auto dc = dec.declare_resources();
    out.head = bytes_of(dc, "dwd-head:dec");
    for (const ResourceClaim& c : dc) {
      if (c.kind == model_memory::kWeightsKind) {
        out.vae_weights.push_back(c.key);
      }
    }
    const auto gc = gen.declare_resources();
    out.arena = bytes_of(gc, "vae-decode");
    out.payload = bytes_of(gc, "gen-latent");
    out.asked = gen.planned_dwd_head();
    for (const StageHolding& h :
         static_cast<Stage&>(gen).declare_memory().holdings) {
      out.gen_src.push_back(h.source);
    }
    for (const StageHolding& h :
         static_cast<Stage&>(dec).declare_memory().holdings) {
      out.dec_src.push_back(h.source);
    }
    return out;
  };
  const Plan d = plan(dwd);
  const Plan v = plan(plain);

  // What each is, from the headers.
  MetalMiniMaxH3FastVae::Config hc;
  MetalMiniMaxH3VideoVae::Config vc;
  MetalMiniMaxH3Transformer::Config tc;
  ASSERT_TRUE(MetalMiniMaxH3FastVae::config_from_dwd(dwd, &hc));
  const std::string vdir = MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(vdir, vc, nullptr));
  ASSERT_TRUE(MetalMiniMaxH3Transformer::config_from_json(base, tc, nullptr));
  const int T = MetalMiniMaxH3VideoVae::video_latent_frames_for(vc, FR);
  const int lh = Hpx / vc.patch, lw = W / vc.patch;
  const std::size_t head = MetalMiniMaxH3FastVae::weight_bytes(dwd, -1);
  const std::size_t arena =
      MetalMiniMaxH3FastVae::scratch_bytes(hc, T, lh, lw) +
      (std::size_t)MetalMiniMaxH3FastVae::decoded_frames(hc, T) * W * Hpx *
          3;
  const std::size_t feats = ((std::size_t)T + 2) *
                            (std::size_t)(lh / tc.patch_h) *
                            (std::size_t)(lw / tc.patch_w) *
                            (std::size_t)tc.hidden * 2;
  const std::size_t vae = MetalMiniMaxH3VideoVae::held_bytes(
      vdir, MetalMiniMaxH3VideoVae::Half::kDecoder);
  std::printf("[pi_h3] DWD plan: head %zu MB (expected %zu), arena %zu MB "
              "(%zu), features %zu MB (%zu), %zu VAE claims; plain: arena "
              "%zu MB, latent %zu MB, %zu VAE claims (decoder half %zu MB); "
              "decode phase %zu MB against %zu\n", d.head >> 20,
              head >> 20, d.arena >> 20, arena >> 20, d.payload >> 20,
              feats >> 20, d.vae_weights.size(), v.arena >> 20,
              v.payload >> 20, v.vae_weights.size(), vae >> 20,
              d.dec >> 20, v.dec >> 20);
  EXPECT_TRUE(!d.asked.empty() && v.asked.empty());
  // The head is not the DiT's adapter, though one file holds both: the
  // report ledger merges holdings of one source at the larger, so the
  // head is named apart -- else its bytes vanish inside the adapter's.
  bool apart = d.dec_src.size() == 1;
  for (const std::string& g : d.gen_src) {
    apart = apart && !d.dec_src.empty() && g != d.dec_src[0];
  }
  bool holds_adapter = false;
  for (const std::string& g : d.gen_src) { holds_adapter |= g == dwd; }
  EXPECT_TRUE(apart && holds_adapter);
  EXPECT_TRUE(head > 0 && d.head == head && v.head == 0);
  EXPECT_TRUE(d.vae_weights.empty() && v.vae_weights.size() == 1);
  EXPECT_TRUE(d.arena == arena && d.payload == feats);
  EXPECT_TRUE(v.arena > 0 && v.payload > 0 && v.payload != d.payload);
  // The decode phase: exactly the swapped terms.
  EXPECT_TRUE(vae > 0 &&
              d.dec + vae + v.arena + v.payload ==
                  v.dec + head + d.arena + d.payload);
}

// WHAT THE HEAD HOLDS AND WHAT A DECODE ALLOCATES, against what the plan
// books for them. Loaded through the manager, as vae-decode loads it:
// its set (the adapter FILE's, whose key is the DiT's adapter claim)
// caches nothing, its held bytes are weight_bytes(file) to the byte, a
// decode's device peak is within scratch_bytes -- the arena the plan
// books, the bf16 output in it -- and release_scratch() gives all of it
// back (vae-decode calls it after each decode: the plan books the
// working set in the decode alone). At the shipped graphs' geometries,
// 640 x 352 x 73 (22 latents) and 1344 x 768 x 124 (37), from row 3
// (anchor rows before); the second decode at each again after a release,
// to say what giving the working set back costs a clip.
//   VPIPE_PI_H3_DWD_ADAPTER  as above
TEST(pi_h3, the_head_allocates_what_its_plan_books)
{
  const std::string file = adapter_file_();
  if (file.empty()) {
    std::printf("  SKIP: VPIPE_PI_H3_DWD_ADAPTER unset\n");
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto ws = open_weight_set(file, &sess);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  MetalMiniMaxH3FastVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3FastVae::config_from_dwd(*ws, &cfg, &err));
  auto head = MetalMiniMaxH3FastVae::load(ws, mc, cfg, &err);
  ASSERT_TRUE(head != nullptr);
  if (!head) { std::printf("  load: %s\n", err.c_str()); return; }
  const std::size_t booked_w = MetalMiniMaxH3FastVae::weight_bytes(file, -1);
  std::printf("  held %zu B, booked %zu B; the set caches %zu B\n",
              head->weight_bytes(), booked_w, ws->stats().bytes);
  EXPECT_TRUE(head->weight_bytes() == booked_w);
  EXPECT_TRUE(ws->stats().bytes == 0);

  const int FD = cfg.feature_dim;
  const struct { int T, h2, w2; } geos[] = {{22, 11, 20}, {37, 24, 42}};
  const std::size_t base = mc->memory_budget_settled().allocated;
  for (const auto& g : geos) {
    const std::size_t row0 = 3;
    const std::size_t rows = row0 + (std::size_t)g.T * g.h2 * g.w2;
    SharedBuffer feats = mc->make_shared_buffer(rows * FD * 2);
    ASSERT_TRUE(!feats.empty());
    if (feats.empty()) { return; }
    auto* f = static_cast<std::uint16_t*>(feats.contents());
    for (std::size_t i = 0; i < rows * FD; ++i) {
      f[i] = bf16_((float)std::sin(0.61 * (double)i) * 1.3f);
    }
    MetalMiniMaxH3FastVae::Config eff = head->config();
    eff.i8_tier = head->i8_tier();
    const std::size_t booked =
        MetalMiniMaxH3FastVae::scratch_bytes(eff, g.T, 2 * g.h2, 2 * g.w2);
    double secs[3] = {0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {
      // rep 0 and 1 from a released head (rep 1 the same geometry again:
      // it makes its working set anew), rep 2 with it kept.
      if (rep < 2) { head->release_scratch(); }
      const std::size_t a0 = mc->memory_budget_settled().allocated;
      std::atomic<std::size_t> peak{a0};
      std::atomic<bool> done{false};
      std::thread sampler([&]() {
        while (!done.load()) {
          const std::size_t a = mc->memory_budget().allocated;
          if (a > peak.load()) { peak.store(a); }
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
      });
      const auto t0 = std::chrono::steady_clock::now();
      SharedBuffer px = head->decode_features(feats, row0, g.T, g.h2, g.w2,
                                              &err);
      secs[rep] = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
      done.store(true);
      sampler.join();
      ASSERT_TRUE(!px.empty());
      if (px.empty()) { std::printf("  decode: %s\n", err.c_str()); return; }
      if (rep == 0) {
        const std::size_t grew = peak.load() - a0;
        std::printf("  [%d x %d x %d] tokens -> %d frames %dx%d: peak "
                    "allocation %zu MB, booked %zu MB\n", g.T, g.h2, g.w2,
                    head->decoded_frames(g.T), g.w2 * 32, g.h2 * 32,
                    grew >> 20, booked >> 20);
        // Released before it: nothing beside the weights but the
        // features this test made.
        EXPECT_TRUE(a0 <= base + feats.byte_size() + (std::size_t(1) << 20));
        EXPECT_TRUE(booked + (std::size_t(8) << 20) >= grew);
      }
    }
    std::printf("  a decode %.2f s released / %.2f s again released / "
                "%.2f s kept\n", secs[0], secs[1], secs[2]);
    // The set still caches nothing after a decode.
    EXPECT_TRUE(ws->stats().bytes == 0);
  }
  // Released, the head holds its weights alone.
  head->release_scratch();
  const std::size_t after = mc->memory_budget_settled().allocated;
  std::printf("  released: %zu MB over the start\n",
              (after > base ? after - base : 0) >> 20);
  EXPECT_TRUE(after <= base + (std::size_t(1) << 20));
}
