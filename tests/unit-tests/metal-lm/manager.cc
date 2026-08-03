// manager.cc -- Model-manager / accounting level checks: the plain text-decode
// smoke, the KV budget vs max_pages, weight-set sharing between an LM and a
// vision tower, the manager's view of KV growth, and the ASR conformer QKV
// fusion.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

TEST(metal_lm_smoke, text_decode) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto ids = lm->tokenizer().encode("The capital of France is");
  ASSERT_TRUE(!ids.empty());
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());

  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 8; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke] %zu tokens | gen='%s'\n", gen.size(),
              text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Regression for "chat truncates ~2048 tokens regardless of max_pages": the
// metal Gemma contiguous KV used to hardcode max_seq=2048, ignoring the
// configured budget. It must now follow page_tokens * max_pages. Load with a
// SMALL budget (128 * 3 = 384) and force-decode past it: the decode must hit
// its cap at ~384 -- NOT run on toward the old 2048 -- proving the budget is
// config-derived. (Build-agnostic: the MLX paged pool caps at the same
// 384-token budget.) Gated on the Gemma checkpoint.
TEST(metal_lm_smoke, gemma_kv_budget_follows_max_pages) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // The contiguous-KV cap is a METAL-backend property; pin metal so the test
  // exercises the metal exec in BOTH builds. (In the MLX build an unset
  // backend loads the MLX paged Gemma path, whose KV grows differently and
  // does not hard-stop at page_tokens*max_pages -- so the cap assertion only
  // makes sense against the metal exec.)
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  if (sess.metal_compute() == nullptr || !sess.metal_compute()->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  const int kPageTokens = 128, kMaxPages = 3;
  const int kBudget = kPageTokens * kMaxPages;     // 384
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = kPageTokens;
  spec.max_pages     = kMaxPages;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto ids = lm->tokenizer().encode("Count upwards:");
  ASSERT_TRUE(!ids.empty());
  ASSERT_TRUE((int)ids.size() < kBudget);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  int32_t forced = lm->prefill(ctx, ids);
  ASSERT_TRUE(forced >= 0);

  // Force-decode (ignore stop tokens) until the KV cap returns -1.
  int produced = 0;
  const int kLimit = 1500;     // > budget(384), << old hardcoded 2048
  for (int i = 0; i < kLimit; ++i) {
    const int32_t n = lm->next_token(ctx, forced);
    if (n < 0) { break; }
    forced = n;
    ++produced;
  }
  const int final_pos = (int)ids.size() + produced;
  std::printf("[metal_lm_smoke.gemma_kv_budget] budget=%d final_pos=%d "
              "produced=%d\n", kBudget, final_pos, produced);
  // Hit a cap well before the loop limit (and far below the old 2048),
  // landing at the configured budget.
  EXPECT_TRUE(produced < kLimit);
  EXPECT_TRUE(final_pos <= kBudget + kPageTokens);   // ~384, NOT ~2048
  EXPECT_TRUE(final_pos >= kBudget - kPageTokens);
}

// The dedup the refactor exists for, asserted on a real model rather than
// a fixture: an LM and the vision tower it feeds live in ONE checkpoint
// directory, and used to open it twice over -- the LM's own mmap plus a
// private one inside the tower. The manager should now report exactly one
// weight set for that directory, with BOTH of them holding it.
//
// This is the property, not a proxy for it: a footprint delta would be
// swamped by the model itself, whereas the holder count says directly
// whether the tower joined the LM's checkpoint or opened its own.
TEST(metal_lm_smoke, lm_and_vision_tower_share_one_weight_set) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.page_tokens = 256;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }
  if (!lm->config().vision.present) { return; }   // not a VL checkpoint

  const auto rep = mgr->weight_report();
  ASSERT_TRUE(rep.size() == 1u);   // ONE checkpoint, not one per consumer
  std::printf("[ws_share] %s: %zu tensors, %.2f GB, %ld holders\n",
              rep[0].dir.c_str(), rep[0].tensors,
              (double)rep[0].bytes / (1024.0 * 1024.0 * 1024.0),
              (long)rep[0].holders);
  // The LM's exec and the tower both hold it, so dropping one must not
  // close the checkpoint out from under the other.
  EXPECT_TRUE(rep[0].holders >= 2);
  EXPECT_TRUE(rep[0].tensors > 0u);
}

// The Qwen3-ASR Conformer fuses each block's q|k|v into ONE matrix so a
// single GEMM replaces three. That fusion was rewritten to build inside
// the weight set's derived() cache (the three pieces are read uncached
// and dropped; only the fused product is kept), and a mistake there --
// wrong order, wrong widths -- would not crash: it would quietly produce
// a plausible-looking wrong transcript. So check the bytes directly.
//
// Loads through the manager, so what is verified is the tensor the
// PRODUCTION path actually built, then re-derives the expected
// concatenation independently from the same checkpoint.
TEST(metal_lm_smoke, asr_conformer_qkv_fusion_matches_pieces) {
  const char* path = std::getenv("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.page_tokens = 256;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }
  if (!lm->config().audio.present) { return; }

  auto ws = genai::open_weight_set(path, &sess);
  ASSERT_TRUE(ws != nullptr);

  // Same element conversion the encoder's to_f16 does.
  auto to_f16 = [&](const std::string& nm) -> std::vector<_Float16> {
    const auto* info = ws->src().info(nm);
    std::vector<_Float16> out;
    if (info == nullptr) { return out; }
    std::size_t n = 1;
    for (auto d : info->shape) { n *= (std::size_t)d; }
    auto raw = ws->read(nm, mc, genai::WeightSet::Residency::Copied);
    if (raw.empty()) { return out; }
    out.resize(n);
    if (info->dtype == "F16") {
      std::memcpy(out.data(), raw.contents(), n * 2);
    } else if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bits = (std::uint32_t)s[i] << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        out[i] = (_Float16)f;
      }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { out[i] = (_Float16)s[i]; }
    } else {
      out.clear();
    }
    return out;
  };

  int checked = 0;
  for (int b = 0; b < lm->config().audio.encoder_layers; ++b) {
    const std::string p =
        "audio_tower.layers." + std::to_string(b) + ".";
    for (const char* suffix : {".weight", ".bias"}) {
      const std::string key = std::string("qwen3asr-conformer/") + p +
                              "qkv" + suffix;
      // A cache HIT never runs the builder, so this returns the tensor
      // the encoder built. An empty result means the key was NOT cached
      // -- which would itself be the bug.
      auto fused = ws->derived(
          key, []() { return vpipe::metal_compute::SharedBuffer{}; });
      ASSERT_TRUE(!fused.empty());

      std::vector<_Float16> want;
      for (const char* proj : {"self_attn.q_proj", "self_attn.k_proj",
                               "self_attn.v_proj"}) {
        auto piece = to_f16(p + proj + suffix);
        ASSERT_TRUE(!piece.empty());
        want.insert(want.end(), piece.begin(), piece.end());
      }
      ASSERT_TRUE(fused.byte_size() == want.size() * 2);
      EXPECT_TRUE(std::memcmp(fused.contents(), want.data(),
                              want.size() * 2) == 0);
      ++checked;
    }
  }
  std::printf("[asr_qkv] %d fused q|k|v tensors match their pieces\n",
              checked);
  EXPECT_TRUE(checked > 0);
}

// KV is the one large allocation that grows DURING a run rather than at
// load, so a weights-only accounting reads as healthy right up to the
// point a long context exhausts the box. Assert the manager can actually
// see it, and that it grows with the conversation -- plumbing that is
// never exercised is plumbing that silently returns zero.
TEST(metal_lm_smoke, manager_sees_kv_grow_with_the_context) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.page_tokens = 256;
  spec.max_pages = 32;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  const std::size_t weights = mgr->resident_weight_bytes();
  const std::size_t kv0     = mgr->resident_kv_bytes();
  EXPECT_TRUE(mgr->resident_bytes() == weights + kv0);

  // Prefill something long enough to claim real pages.
  std::string prompt;
  for (int i = 0; i < 40; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening. ";
  }
  auto ctx = lm->make_context();
  auto ids = lm->tokenizer().encode(prompt);
  if (ids.empty()) { return; }
  lm->prefill(ctx, ids);
  for (int i = 0; i < 8; ++i) { (void)lm->next_token(ctx); }

  const std::size_t kv1 = mgr->resident_kv_bytes();
  std::printf("[kv] weights %.2f GB, kv %zu -> %zu KB after %zu tokens\n",
              (double)weights / (1024.0 * 1024.0 * 1024.0),
              kv0 >> 10, kv1 >> 10, ids.size());
  // The pools start at one page and grow on demand, so a real prefill
  // must move the number. This is the whole point of reporting what is
  // HELD rather than max_pages * page_tokens, which would have been a
  // large constant from the start and told us nothing.
  EXPECT_TRUE(kv1 > kv0);
  // And weights must NOT have moved -- the two are separate terms.
  EXPECT_TRUE(mgr->resident_weight_bytes() == weights);
  EXPECT_TRUE(mgr->resident_bytes() == weights + kv1);
}
