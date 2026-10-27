// lm-run-memory.cc -- what a language model RUNS in beside its weights
// (genai::lm_run_bytes, the figure the LM stages book), against the device.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "common/session.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/lm-run-memory.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "pipeline/resource-claim-delivery.h"
#include "pipeline/resource-plan.h"
#include "stages/model-memory-host.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return v != nullptr ? std::string(v) : std::string();
}

}  // namespace

// THE PLAN'S FIGURE FOR A LANGUAGE MODEL'S RUN, against what it allocates:
// each family here loaded through the manager into a 17-page pool (8704
// tokens), a prompt that fills one 8192-row prefill piece, then 8 decode
// steps -- the peak device allocation over all of it, sampled from a
// thread, beside lm_run_bytes for that pool and one context. The tuners
// are pinned (their probes are transient buffers of their own). The
// load's warm-up builds the persistent scratch -- the decode buffers,
// Gemma's materialized global scores (170 MB here) -- before the
// baseline, so the figure sits above the measurement by those. MEASURED
// on the M4 Pro: Qwen3.5-4B 1781 MB against 1895 booked, Qwen3.6-35B-A3B
// 1949 / 2072, Ministral-3 14B 2964 / 3250, Gemma-4 e4b 1837 / 2275; on
// a 16 GB base M5, the 35B's experts cached, 1997 / 2089.
// Env: VPIPE_QWEN35_TEST_MODEL_PATH (hybrid), VPIPE_QWEN35_MOE_TEST_MODEL_
// PATH (MoE), VPIPE_MINISTRAL3_TEST_MODEL_PATH (dense, head dim 128),
// VPIPE_GEMMA4_TEST_MODEL_PATH (one-pass prefill, sliding rings),
// VPIPE_QWEN4_EXP_TEST_MODEL_PATH (hyper-connections; its pieces as wide
// as the box allows, hc_piece_rows).
TEST(lm_run_memory, a_prefill_allocates_what_the_plan_books)
{
  const char* envs[] = {"VPIPE_QWEN35_TEST_MODEL_PATH",
                        "VPIPE_QWEN35_MOE_TEST_MODEL_PATH",
                        "VPIPE_MINISTRAL3_TEST_MODEL_PATH",
                        "VPIPE_GEMMA4_TEST_MODEL_PATH",
                        "VPIPE_QWEN4_EXP_TEST_MODEL_PATH"};
  ::setenv("VPIPE_QWEN_GQA_AUTOTUNE", "0", 1);
  ::setenv("VPIPE_QWEN_PREFILL_AUTOTUNE", "0", 1);
  ::setenv("VPIPE_GEMMA_ATTN_AUTOTUNE", "0", 1);
  const int kPrompt = 8192, kSteps = 8;
  for (const char* e : envs) {
    const std::string dir = env_(e);
    if (dir.empty()) { continue; }
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    ASSERT_TRUE(mc != nullptr && mgr != nullptr);
    if (mc == nullptr || mgr == nullptr) { return; }
    (void)mc->memory_budget_settled(15000);
    LoadSpec spec;
    spec.hf_dir = dir;
    spec.page_tokens = 512;
    spec.max_pages = 17;
    auto lm = mgr->load(spec);
    ASSERT_TRUE(lm && lm->valid());
    if (!lm || !lm->valid()) { continue; }
    LmRunShape shape;
    shape.page_tokens = spec.page_tokens;
    shape.max_pages = spec.max_pages;
    shape.kv_tokens = (std::int64_t)spec.page_tokens * spec.max_pages;
    const std::size_t booked = lm_run_bytes(&sess, dir, shape);
    std::vector<std::int32_t> ids((std::size_t)kPrompt);
    for (int i = 0; i < kPrompt; ++i) {
      ids[(std::size_t)i] = 1000 + (std::int32_t)((i * 7919) % 20000);
    }
    const std::size_t alloc0 = mc->memory_budget().allocated;
    std::atomic<std::size_t> peak{alloc0};
    std::atomic<bool> done{false};
    std::thread sampler([&] {
      while (!done.load()) {
        const std::size_t a = mc->memory_budget().allocated;
        if (a > peak.load()) { peak.store(a); }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
    auto ctx = lm->make_context();
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < kSteps && t >= 0; ++i) {
      t = lm->next_token_greedy(ctx);
    }
    done.store(true);
    sampler.join();
    ASSERT_TRUE(t >= 0);
    const std::size_t top =
        std::max(peak.load(), mc->memory_budget().allocated);
    const std::size_t grew = top > alloc0 ? top - alloc0 : 0;
    std::printf("[lm-run] %s: %d-token prefill + %d steps peaked %.1f MB "
                "past the load; lm_run_bytes books %.1f MB (%.3fx)\n",
                dir.c_str(), kPrompt, kSteps, grew / 1048576.0,
                booked / 1048576.0,
                grew > 0 ? (double)booked / (double)grew : 0.0);
    // The sampled peak can only under-read the true one; 2 MB covers
    // the per-step parameter blocks the figure leaves out.
    EXPECT_TRUE(grew > 0);
    EXPECT_TRUE(booked + ((std::size_t)2 << 20) >= grew);
    EXPECT_TRUE((double)booked <= 1.3 * (double)grew + (32 << 20));
  }
  ::unsetenv("VPIPE_QWEN_GQA_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_AUTOTUNE");
  ::unsetenv("VPIPE_GEMMA_ATTN_AUTOTUNE");
}

// A Mixture-of-Experts model sizes its routed-expert cache beside the plan
// -- every peer's weights and arenas -- and beside its own run, which the
// LM stage's plan books under lm_run_label: that claim must not be counted
// a second time as a peer's, and a peer's must be. Env:
// VPIPE_QWEN35_MOE_TEST_MODEL_PATH (its config and shard headers only;
// nothing loads), under a 24 GB box so the experts are cached.
TEST(lm_run_memory, the_expert_cache_leaves_its_own_run_out)
{
  const std::string dir = env_("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (dir.empty()) { return; }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (mgr == nullptr) { return; }
  ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(dir);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  ASSERT_TRUE(c.is_moe());
  LmRunShape shape;
  shape.page_tokens = 512;
  shape.max_pages = 64;
  shape.kv_tokens = 512 * 64;
  c.page_tokens = shape.page_tokens;
  c.max_pages = (int)shape.max_pages;
  const std::size_t run = lm_run_bytes(&sess, dir, shape);
  ::setenv("VPIPE_RAM_LIMIT_MB", "24576", 1);
  const std::size_t alone =
      MetalQwenModel::lm_expert_cache_bytes(&sess, dir, c, run);
  mgr->declare_scratch(lm_run_label(dir), run, std::string());
  const std::size_t own =
      MetalQwenModel::lm_expert_cache_bytes(&sess, dir, c, run);
  const std::size_t peer = alone / 4;
  mgr->declare_scratch("peer", peer, std::string());
  const std::size_t beside =
      MetalQwenModel::lm_expert_cache_bytes(&sess, dir, c, run);
  ::unsetenv("VPIPE_RAM_LIMIT_MB");
  std::printf("[lm-run] runs in %zu MB; cache alone %zu MB, with its own "
              "run booked %zu MB, beside a peer's %zu MB arena %zu MB\n",
              run >> 20, alone >> 20, own >> 20, peer >> 20, beside >> 20);
  EXPECT_TRUE(alone > 0);
  EXPECT_TRUE(own == alone);
  EXPECT_TRUE(beside + peer == alone);
}

// The host's limit (load_spec::kMemoryLimitMb) reaches the PLAN its peers
// size against: the stage's claim books the checkpoint at what its cache,
// sized within the limit, leaves it holding -- the limit less the run,
// never under the floor -- the weights planner honours it as the runtime
// delivers the claim, and the cache sized beside that plan leaves out
// exactly its own share: beside a peer's weights on a box that binds
// before the limit, the session-free figure on that box less the peer.
// Env: VPIPE_QWEN35_MOE_TEST_MODEL_PATH (config and shard headers only).
TEST(lm_run_memory, a_hosts_limit_reaches_the_plan)
{
  const std::string dir = env_("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (dir.empty()) { return; }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (mgr == nullptr) { return; }
  ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(dir);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  ASSERT_TRUE(c.is_moe());
  auto w = MetalLlamaWeights::open_model(dir);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  const std::size_t ex = MetalQwenModel::routed_expert_bytes(*w, c);
  LmRunShape shape;
  shape.page_tokens = 512;
  shape.max_pages = 64;
  shape.kv_tokens = 512 * 64;
  c.page_tokens = shape.page_tokens;
  c.max_pages = (int)shape.max_pages;
  const std::size_t run = lm_run_bytes(&sess, dir, shape);
  const auto plain = model_memory::lm_claims(&sess, dir, "chat", shape, -1);
  ASSERT_TRUE(!plain.empty() && plain[0].floor_bytes > 0);
  if (plain.empty()) { return; }
  const std::size_t floor = plain[0].floor_bytes;
  EXPECT_TRUE(bag::integer(plain[0].extra, model_memory::kClaimHeldBytes,
                           0) == 0);
  // a limit that leaves the model its floor and a quarter of the experts
  const std::size_t held = floor + ex / 4;
  const std::size_t limit = run + held;
  const auto claims =
      model_memory::lm_claims(&sess, dir, "chat", shape, -1, 0, limit);
  EXPECT_TRUE(bag::integer(claims[0].extra, model_memory::kClaimHeldBytes,
                           0) == (std::int64_t)held);
  const std::size_t preload = model_memory::dir_weights_bytes(dir);
  ASSERT_TRUE(held < preload);
  auto& reg = ResourcePlannerRegistry::get();
  for (std::string_view k :
       {model_memory::kWeightsKind, model_memory::kScratchKind}) {
    if (ResourcePlanner* p = reg.find(k)) { p->begin_plan(&sess); }
  }
  for (const ResourceClaim& cl : claims) {
    if (ResourcePlanner* p = reg.find(cl.kind)) {
      DeliveringResourceClaim d(cl);
      p->claim(&sess, cl.key, cl.phase, cl.last_phase, cl.floor_bytes);
    }
  }
  EXPECT_TRUE(mgr->planned_weight_bytes(dir) == held);
  EXPECT_TRUE(mgr->phase_footprint(std::string()) == held);
  // beside a peer's weights, the box binding at an eighth of the experts
  const std::size_t peer = (std::size_t)4 << 30;
  mgr->declare_weights(dir + "/a-peer", peer);
  const std::size_t box =
      (((std::size_t)8 << 30) + run + floor + ex / 8) >> 20;
  ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(box + (peer >> 20)).c_str(),
           1);
  const std::size_t got =
      MetalQwenModel::lm_expert_cache_bytes(&sess, dir, c, run, limit);
  ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(box).c_str(), 1);
  const std::size_t alone =
      MetalQwenModel::lm_expert_cache_bytes(nullptr, dir, c, run, limit);
  ::unsetenv("VPIPE_RAM_LIMIT_MB");
  std::printf("[lm-run] a %zu MB limit: held %zu MB (floor %zu, on disk "
              "%zu); beside the plan and a %zu MB peer a %zu MB cache, "
              "alone on the box less the peer %zu MB\n", limit >> 20,
              held >> 20, floor >> 20, preload >> 20, peer >> 20, got >> 20,
              alone >> 20);
  EXPECT_TRUE(alone > 0 && alone < ex / 4);
  EXPECT_TRUE(got == alone);
}

// The cache the host ASKS for (load_spec::kExpertCacheMb) on a Qwen MoE:
// at most that, within the limit and the box -- every expert resident
// where it holds them all, never under a layer's experts -- and the claim
// booking the rest of the model and that cache (every expert at most),
// the smaller under a limit, which the weights planner honours.
// Env: VPIPE_QWEN35_MOE_TEST_MODEL_PATH (config and shard headers only;
// a box that holds it whole -- 64 GB for the 35B-A3B 4-bit).
TEST(lm_run_memory, the_cache_is_at_most_what_the_host_asks)
{
  const std::string dir = env_("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (dir.empty()) { return; }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (mgr == nullptr) { return; }
  ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(dir);
  ASSERT_TRUE(mcfg.has_value());
  if (!mcfg.has_value()) { return; }
  MetalQwenModel::Config c = MetalQwenModel::config_from(*mcfg);
  ASSERT_TRUE(c.is_moe());
  auto w = MetalLlamaWeights::open_model(dir);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  const std::size_t ex = MetalQwenModel::routed_expert_bytes(*w, c);
  const std::size_t layer = ex / (std::size_t)std::max(1, c.n_layers);
  LmRunShape shape;
  shape.page_tokens = 512;
  shape.max_pages = 64;
  shape.kv_tokens = 512 * 64;
  c.page_tokens = shape.page_tokens;
  c.max_pages = (int)shape.max_pages;
  const std::size_t run = lm_run_bytes(&sess, dir, shape);
  auto cache = [&](std::size_t limit, std::size_t asked) {
    return MetalQwenModel::lm_expert_cache_bytes(nullptr, dir, c, run,
                                                 limit, asked);
  };
  const auto plain = model_memory::lm_claims(&sess, dir, "chat", shape, -1);
  ASSERT_TRUE(!plain.empty() && plain[0].floor_bytes > layer);
  if (plain.empty()) { return; }
  const std::size_t floor = plain[0].floor_bytes;
  const std::size_t rest = floor - layer;
  if (cache(0, 0) != 0) {
    std::printf("[lm-run] this box does not hold the model whole: the "
                "loader's arms are skipped\n");
  } else {
    // what is asked, at most
    EXPECT_TRUE(cache(0, ex / 2) == ex / 2);
    EXPECT_TRUE(cache(0, ex / 4) == ex / 4);
    // every expert, or more: all of them resident
    EXPECT_TRUE(cache(0, ex) == 0);
    EXPECT_TRUE(cache(0, 2 * ex) == 0);
    // less than a layer's experts: a layer's
    EXPECT_TRUE(cache(0, 1) == layer);
    // within a limit leaving about a quarter of the experts: the smaller
    // of the two
    const std::size_t lim = run + floor + ex / 4;
    const std::size_t tight = cache(lim, 0);
    ASSERT_TRUE(tight > layer && tight < ex / 2);
    EXPECT_TRUE(cache(lim, ex / 2) == tight);
    EXPECT_TRUE(cache(lim, tight / 2) == tight / 2);
  }
  // the plan: the rest of the model and the cache asked for
  auto booked = [&](std::size_t limit, std::size_t asked) {
    return bag::integer(model_memory::lm_claims(&sess, dir, "chat", shape,
                                                -1, 0, limit,
                                                asked)[0].extra,
                        model_memory::kClaimHeldBytes, -1);
  };
  EXPECT_TRUE(booked(0, 0) == -1);
  EXPECT_TRUE(booked(0, ex / 2) == (std::int64_t)(rest + ex / 2));
  EXPECT_TRUE(booked(0, 1) == (std::int64_t)floor);
  EXPECT_TRUE(booked(0, 2 * ex) == (std::int64_t)(rest + ex));
  // under a limit, the smaller
  EXPECT_TRUE(booked(run + rest + ex * 3 / 4, ex / 2) ==
              (std::int64_t)(rest + ex / 2));
  EXPECT_TRUE(booked(run + rest + ex / 2, ex * 3 / 4) ==
              (std::int64_t)(rest + ex / 2));
  // delivered as the runtime does
  const auto claims =
      model_memory::lm_claims(&sess, dir, "chat", shape, -1, 0, 0, ex / 2);
  auto& reg = ResourcePlannerRegistry::get();
  for (std::string_view k :
       {model_memory::kWeightsKind, model_memory::kScratchKind}) {
    if (ResourcePlanner* p = reg.find(k)) { p->begin_plan(&sess); }
  }
  for (const ResourceClaim& cl : claims) {
    if (ResourcePlanner* p = reg.find(cl.kind)) {
      DeliveringResourceClaim d(cl);
      p->claim(&sess, cl.key, cl.phase, cl.last_phase, cl.floor_bytes);
    }
  }
  std::printf("[lm-run] half the experts asked: booked %zu MB (floor %zu, "
              "experts %zu)\n", mgr->planned_weight_bytes(dir) >> 20,
              floor >> 20, ex >> 20);
  EXPECT_TRUE(mgr->planned_weight_bytes(dir) == rest + ex / 2);
  mgr->clear_declarations();
}
