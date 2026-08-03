// qwen-text.cc -- Qwen (and Llama) text generation: safetensors / GGUF / OptiQ-
// mixed / MoE chat smokes, the flash + MMA prefill token-exact checks and the
// GGUF byte-level tokenizer round trip.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// Matrix-core (M5+) prefill GEMM must be greedy token-exact with the steel
// quantized GEMM. Loads the SAME Qwen3.5 checkpoint twice -- once with the
// matrix-core path forced off (VPIPE_QWEN_NO_MMA=1, the steel reference)
// and once with it on -- prefills a prompt long enough to exercise the
// prefill projections (VPIPE_QWEN_MMA_MIN_M lowered so even a short prompt
// routes through it) and greedy-decodes; the two token streams must match.
// On a GPU without matrix cores both loads are steel and the test is a
// trivial (still valid) pass. Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, mma_prefill_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool use_mma) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_QWEN_MMA_MIN_M", "4", 1);   // exercise mma on short prompts
    ::setenv("VPIPE_QWEN_MMA_ATTN_MIN_N", "8", 1);  // and mma flash attention
    if (use_mma) { ::unsetenv("VPIPE_QWEN_NO_MMA"); }
    else         { ::setenv("VPIPE_QWEN_NO_MMA", "1", 1); }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(
        "In a distant kingdom by the northern sea there lived a clever "
        "young clockmaker who dreamed of building a machine that could "
        "tell not only the hour but the weather of tomorrow.");
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // steel
  const auto got = run(true);    // matrix-core (M5) or steel (older)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_QWEN_NO_MMA");
  ::unsetenv("VPIPE_QWEN_MMA_MIN_M");
  ::unsetenv("VPIPE_QWEN_MMA_ATTN_MIN_N");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.mma_prefill_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 full-attention PREFILL key-split flash kernel (sdpa_paged_flash_f16)
// must be greedy token-exact with the scalar query-tiled reference
// (sdpa_paged_qtile) over a prompt long enough that the flash path fires
// (n >= 384). flash is an online-softmax fp-approximation (not bit-identical,
// like the Gemma flash), so this gates that it doesn't flip a greedy argmax vs
// the established M4 path. _flash_attn is read at LOAD, so each variant uses a
// fresh Session (the LM manager caches by spec). Gated on
// VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_flash_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // ~600-token prompt so the full-attention prefill flash path fires.
  std::string big;
  for (int i = 0; i < 60; ++i) {
    big += "The history of computing is long and storied. ";
  }
  big += "Summarize the key milestones.";

  auto gen = [&](bool no_flash, bool force_mixed = false) {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (no_flash) { ::setenv("VPIPE_QWEN_NO_FLASH", "1", 1); }
    else          { ::unsetenv("VPIPE_QWEN_NO_FLASH"); }
    if (force_mixed) { ::setenv("VPIPE_QWEN_FORCE_MIXED", "1", 1); }
    else             { ::unsetenv("VPIPE_QWEN_FORCE_MIXED"); }
    std::vector<std::int32_t> out;
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    if (mc != nullptr && mc->valid() && mgr != nullptr) {
      genai::LoadSpec spec;
      spec.hf_dir = path;
      spec.compute_dtype = "f16";
      spec.page_tokens = 512;
      spec.max_pages = 16;
      auto lm = mgr->load(spec);
      if (lm && lm->valid()) {
        auto ids = lm->tokenizer().encode(big);
        if (ids.size() >= 384) {
          auto ctx = lm->make_context();
          std::int32_t t = lm->prefill(ctx, ids);
          out.push_back(t);
          for (int i = 1; i < 32 && t >= 0; ++i) {
            t = lm->next_token(ctx);
            out.push_back(t);
          }
        }
      }
    }
    ::unsetenv("VPIPE_LLM_BACKEND");
    ::unsetenv("VPIPE_QWEN_NO_FLASH");
    ::unsetenv("VPIPE_QWEN_FORCE_MIXED");
    return out;
  };

  const auto qtile = gen(true);    // scalar query-tiled reference
  const auto flash = gen(false);   // key-split flash (default)
  ASSERT_TRUE(!qtile.empty());
  ASSERT_TRUE(qtile.size() == flash.size());
  std::size_t mism = 0; int first_div = -1;
  for (std::size_t i = 0; i < flash.size(); ++i) {
    if (flash[i] != qtile[i]) {
      ++mism;
      if (first_div < 0) { first_div = (int)i; }
    }
  }
  std::printf("[qwen_flash_tokexact] N=%zu | flash-vs-qtile mism=%zu "
              "(first_div=%d)\n", flash.size(), mism, first_div);
  EXPECT_TRUE(mism == 0);

  // Force the mixed/de-fused paths on this (uniform-affine) model: a uniform
  // layer takes the qkv_fused / mlp_fused re-fusion, the GDN in_proj de-fuses,
  // etc. -- all must reproduce the fused output exactly. On a genuinely-mixed
  // model (OptiQ) FORCE_MIXED is a no-op, so this arm is a strict superset check
  // of the mixed machinery's numeric equivalence to the fused path.
  const auto forced = gen(false, /*force_mixed=*/true);
  if (!forced.empty()) {
    ASSERT_TRUE(forced.size() == flash.size());
    std::size_t fmism = 0; int fdiv = -1;
    for (std::size_t i = 0; i < flash.size(); ++i) {
      if (forced[i] != flash[i]) { ++fmism; if (fdiv < 0) { fdiv = (int)i; } }
    }
    std::printf("[qwen_flash_tokexact] force-mixed-vs-fused mism=%zu "
                "(first_div=%d)\n", fmism, fdiv);
    EXPECT_TRUE(fmism == 0);
  }
}

// End-to-end coherence: render a real chat prompt, prefill + greedy-decode,
// and require a factually correct answer ("Paris"). Unlike the token-exact
// tests above (which only check fast-vs-slow self-consistency and would pass
// even on a mis-decoded checkpoint), this is a cross-reference correctness
// gate -- it fails if the quantized weights are read at the wrong width or
// the dequant is wrong. Works on any Qwen3.5 checkpoint regardless of quant
// (4-bit / 8-bit): selects the w4g64 / w8g64 kernels from config. The
// generation budget is generous enough to span a thinking block before the
// answer. Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_text_chat) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "f16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 96; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen_text_chat] %zu tok | gen='%s'\n", gen.size(),
              text.c_str());
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) {
    c = (char)std::tolower((unsigned char)c);
  }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);
}

// Native k-quant (GGUF Q4_K_M) Qwen3.5-2B end-to-end coherence + perf.
// Loads the model from the .gguf via the metal k-quant path (no requant);
// the tokenizer + chat template come from a safetensors Qwen3.5 dir (the
// GGUF ships none -- the family shares one tokenizer). Renders a factual
// prompt, prefills + greedy-decodes through MetalQwenModel directly, decodes
// the answer, and asserts it names the city -- a real cross-reference (the
// per-tensor q4_K/q5_K/q6_K dispatch + the A_log/conv transforms are all
// exercised; mis-loaded weights produce garbage, not "paris"). Prints
// prefill/decode tok/s for the llama.cpp comparison. Gated on
// VPIPE_QWEN_GGUF_TEST_MODEL_PATH (.gguf) + VPIPE_QWEN35_TEST_MODEL_PATH
// (safetensors dir, for the tokenizer).
TEST(metal_lm_smoke, qwen_gguf_text_chat) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  const char* tok_dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!gguf || !*gguf || !tok_dir || !*tok_dir) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  // The GGUF model (metal k-quant path).
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  ASSERT_TRUE(cfg.has_value());
  EXPECT_TRUE(cfg->architecture == "Qwen3_5ForConditionalGeneration");
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);

  // Tokenizer + chat template from a safetensors Qwen3.5 dir (manager load).
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = tok_dir;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  // A long-ish prompt so the prefill number is comparable to llama.cpp's
  // pp512 (a 24-token prompt is fixed-overhead-bound); the France question
  // at the end still drives the coherence assert.
  std::string prompt;
  for (int i = 0; i < 36; ++i) {
    prompt += "The following is background context for a geography quiz. ";
  }
  prompt += "What is the capital of France? Reply with the city name only.";
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(prompt, /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> lg = model->prefill(ids);
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);

  // Branch a child off the prefilled prefix BEFORE decoding so the pipelined
  // run starts from the same state as the synchronous run (for token-exact
  // A/B). branch() refcount-shares KV pages + deep-copies the GDN conv/ssm.
  const genai::ContextId child = model->context_manager()->branch(
      model->root_context());
  ASSERT_TRUE(child.valid());

  // Decode via the in-stream greedy path (forward_argmax -> decode_step_fast:
  // q6_K embed gather + on-GPU argmax folded into the decode command buffer,
  // the production next_token_greedy path -- no per-token embed round-trip).
  const int kGen = 200;   // thinking-style model: room to reach the answer
  std::vector<std::int32_t> gen;
  gen.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(gen.back());
    if (t < 0) { break; }
    gen.push_back(t);
  }
  const auto t2 = std::chrono::steady_clock::now();

  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  const double pf_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double dc_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::printf("[qwen_gguf_text_chat] prefill %zu tok in %.1f ms (%.0f tok/s) "
              "| decode(sync) %zu tok in %.1f ms (%.1f tok/s)\n",
              ids.size(), pf_ms, (double)ids.size() / (pf_ms / 1000.0),
              gen.size(), dc_ms, (double)gen.size() / (dc_ms / 1000.0));

  // A/B the GPU-resident pipelined path (event-chained no-wait command
  // buffers overlap the host's per-token work with the GPU's next forward) on
  // the child branch from the same prefix -- must be token-exact vs sync AND
  // is the production decode path (so its tok/s is the headline).
  {
    std::vector<std::int32_t> pids;
    const auto p0 = std::chrono::steady_clock::now();
    const bool ok = model->decode_pipelined(child, first, kGen, pids);
    const auto p1 = std::chrono::steady_clock::now();
    const double pp_ms =
        std::chrono::duration<double, std::milli>(p1 - p0).count();
    EXPECT_TRUE(ok);
    std::printf("[qwen_gguf_text_chat] decode(pipelined) %zu tok in "
                "%.1f ms (%.1f tok/s)\n", pids.size(), pp_ms,
                !pids.empty() ? (double)pids.size() / (pp_ms / 1000.0) : 0.0);
    // Token-exact vs the synchronous greedy stream (gen[0]==first).
    int mism = 0;
    for (std::size_t i = 0; i < pids.size() && i + 1 < gen.size(); ++i) {
      if (pids[i] != gen[i + 1]) { ++mism; }
    }
    std::printf("[qwen_gguf_text_chat] pipelined vs sync mismatches=%d/%zu\n",
                mism, pids.size());
    EXPECT_TRUE(mism == 0);
  }
  std::printf("[qwen_gguf_text_chat] gen='%s'\n", text.c_str());
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) { c = (char)std::tolower((unsigned char)c); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);
}

// Mixed-precision affine (mlx-optiq) end-to-end: a Qwen3.5 OptiQ checkpoint
// mixes 4-bit and 8-bit affine linears in one model (per-tensor sensitivity
// quant). The metal path must DE-FUSE q|k|v / in_proj / gate|up (they no
// longer share a bit width) and dispatch each projection at its own width --
// if any bit/offset is wrong the forward produces garbage. Asserts the mixed
// path is engaged (path-selection guard, not timing) AND the model stays
// coherent through prefill + the in-stream greedy decode (reaches the answer),
// plus a sync-vs-pipelined token-exact A/B. Gated on
// VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH (the OptiQ safetensors dir, which also
// carries its own tokenizer.json + chat_template.jinja).
TEST(metal_lm_smoke, qwen_optiq_mixed_precision_text_chat) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  // Path-selection guard: the de-fused per-tensor mixed-affine path engaged.
  // (A silent fall-back to a uniform width would mis-stride the 8-bit tensors
  // -> garbage, but a uniform-bits checkpoint here would pass vacuously.)
  EXPECT_TRUE(model->uses_mixed_precision());

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);

  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid());

  const int kGen = 200;   // thinking-style model: room to reach the answer
  std::vector<std::int32_t> gen;
  gen.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(gen.back());
    if (t < 0) { break; }
    gen.push_back(t);
  }
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen_optiq] mixed=%d gen='%s'\n",
              (int)model->uses_mixed_precision(), text.c_str());
  // Greedy token-exact vs mlx-lm (stock omlx, same affine quant) was verified
  // out-of-band on these exact prompt ids: vpipe's first 24 generated ids
  // match mlx-lm's bit-for-bit (90700,8340,25,271,16,13,220,2972,2014,53983,
  // 279,5952,64700,198,262,348,256,15380,25,328,3710,369,279,6511). The
  // coherence assert below is the in-suite proxy (a mis-strided 8-bit tensor
  // would derail it).
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) { c = (char)std::tolower((unsigned char)c); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);

  // Sync vs GPU-resident pipelined decode must be token-exact (same de-fused
  // mixed-affine forward; the pipelined path is the production decode).
  std::vector<std::int32_t> pids;
  const bool ok = model->decode_pipelined(child, first, kGen, pids);
  EXPECT_TRUE(ok);
  int mism = 0;
  for (std::size_t i = 0; i < pids.size() && i + 1 < gen.size(); ++i) {
    if (pids[i] != gen[i + 1]) { ++mism; }
  }
  std::printf("[qwen_optiq] pipelined vs sync mismatches=%d/%zu\n",
              mism, pids.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5-MoE (35B-A3B) text-chat bring-up: brings up the Mixture-of-Experts
// MLP (256-expert top-8 router + shared expert) on the metal backend, on top
// of the shared hybrid GDN+full-attn backbone. Prefill (on-GPU routed, pair-
// batched expert GEMVs) -> greedy decode (gathered expert matvecs). Dumps the
// first generated ids for out-of-band token-exact comparison vs omlx
// qwen3_5_moe; the in-suite proxy is a coherent "Paris" answer (a mis-routed
// expert or mis-strided 3D expert slab would derail it). Also checks the
// pipelined GPU-resident decode is token-exact with the serial path. Gated on
// VPIPE_QWEN35_MOE_TEST_MODEL_PATH (a ~18-20 GB model -> 64 GB box only).
TEST(metal_lm_smoke, qwen35_moe_text_chat) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  // Guard: the config parsed as MoE (else the test passes vacuously on a dense
  // model and never exercises the expert path).
  ASSERT_TRUE(mcfg.is_moe());
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);

  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid());

  const int kGen = 200;
  std::vector<std::int32_t> gen;
  gen.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(gen.back());
    if (t < 0) { break; }
    gen.push_back(t);
  }
  std::printf("[qwen35_moe] first 16 ids:");
  for (int i = 0; i < 16 && i < (int)gen.size(); ++i) {
    std::printf(" %d", gen[i]);
  }
  std::printf("\n");
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen35_moe] gen='%s'\n", text.c_str());
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) { c = (char)std::tolower((unsigned char)c); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);

  // Pipelined GPU-resident decode must be token-exact with the serial path.
  std::vector<std::int32_t> pids;
  const bool ok = model->decode_pipelined(child, first, kGen, pids);
  EXPECT_TRUE(ok);
  int mism = 0;
  for (std::size_t i = 0; i < pids.size() && i + 1 < gen.size(); ++i) {
    if (pids[i] != gen[i + 1]) { ++mism; }
  }
  std::printf("[qwen35_moe] pipelined vs sync mismatches=%d/%zu\n",
              mism, pids.size());
  EXPECT_TRUE(mism == 0);
}

// GGUF tokenizer scheme regression. Tokenizer::from_gguf must pick byte-level
// (gpt2) vs metaspace (llama) from tokenizer.ggml.model. A Qwen3.5 GGUF is
// "gpt2" byte-level -- the bug forced every GGUF to metaspace, so the raw
// byte-level alphabet (Ġ = space U+0120, Ċ = newline U+010A) leaked into the
// detokenized chat text. Round-trip a string full of spaces + newlines and
// assert no alphabet chars survive. Gated on VPIPE_QWEN_GGUF_MTP_TEST_MODEL_
// PATH (the .gguf file); when VPIPE_QWEN35_TEST_MODEL_PATH (a sibling HF
// tokenizer.json dir) is set too, also cross-checks that the GGUF tokenizer
// ENCODES token-exactly with the HF one (validates the byte-level
// pre-tokenizer, not just decode). Builds in both MLX and no-MLX.
TEST(metal_lm_smoke, qwen_gguf_tokenizer_byte_level_round_trip) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_MTP_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  Session sess;
  auto g = genai::GgufFile::open(gguf);
  ASSERT_TRUE(g.has_value());
  auto tok = genai::Tokenizer::from_gguf(*g, &sess);
  ASSERT_TRUE(tok != nullptr);

  // Spaces AND newlines -- exactly the chars the byte-level alphabet maps to
  // Ġ / Ċ. A correct detokenizer round-trips them verbatim.
  const std::string text =
      "This code defines a GenerativeModelManager class.\n\n"
      "It loads the model and runs inference.";
  auto ids = tok->encode(text);
  ASSERT_TRUE(!ids.empty());
  const std::string back = tok->decode(ids);
  EXPECT_TRUE(back == text);
  EXPECT_TRUE(back.find("\xC4\xA0") == std::string::npos);   // Ġ (U+0120)
  EXPECT_TRUE(back.find("\xC4\x8A") == std::string::npos);   // Ċ (U+010A)

  const char* hf_dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (hf_dir && *hf_dir) {
    auto hf = genai::Tokenizer::from_huggingface_json(
        std::string(hf_dir) + "/tokenizer.json", &sess);
    if (hf != nullptr) {
      EXPECT_TRUE(tok->encode(text) == hf->encode(text));
      EXPECT_TRUE(hf->decode(hf->encode(text)) == text);
    }
  }
}
