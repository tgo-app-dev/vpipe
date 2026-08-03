// attention.cc -- Decode-time attention kernel selection: the GQA flash-decode
// paths (Qwen / Llama / Gemma), the Gemma two-stage argmax, materialized-vs-
// flash and gtile variants, the shared-prefix attention logits check, and the
// M5 fast-path engagement guards.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// PERF-PATH GUARD (not a timing test). Locks in that the Qwen3.5 metal model
// SELECTS its M5 matrix-core fast paths, so a future change -- e.g. an M4-side
// edit (matrix cores absent there) that widens a gate, renames a kernel, or
// breaks a load -- cannot silently drop M5 onto the ~2-2.5x slower steel
// prefill / scalar-attention path. Such a fallback stays TOKEN-EXACT (so the
// token-exact tests miss it) and a perf-floor assertion would FLAKE on the
// M5's thermal throttling (cold ~1240 tok/s prefill vs ~520 hot -- the "qwen
// regression" that turned out to be thermal, not code). So we assert the path
// is engaged, independent of timing. On M4 (no matrix cores) the matmul2d
// assertions are correctly skipped; the GQA flash decode is checked on both.
// Gated on VPIPE_QWEN35_TEST_MODEL_PATH; skips under the A/B disables so they
// don't false-fail.
TEST(metal_lm_smoke, qwen_m5_fastpath_engaged_guard) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // The A/B / safety env switches legitimately force the slow paths -- don't
  // guard when one is set (it isn't the "accidental" disable we're catching).
  if (std::getenv("VPIPE_QWEN_NO_MMA") || std::getenv("VPIPE_QWEN_NO_FLASH") ||
      (std::getenv("VPIPE_QWEN_GQA_ATTN") &&
       std::atoi(std::getenv("VPIPE_QWEN_GQA_ATTN")) == 0)) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  const auto& mcfg_loaded = model->config();

  // GQA flash decode -- the decode fast path (each kv-head read once for all G
  // query heads). Default on for full-attn models, M4 + M5. Off => decode
  // re-reads each kv-head per query head (the ~2x decode regression GQA fixed).
  EXPECT_TRUE(model->gqa_flash_decode());

  if (!mc->supports_matrix_cores()) {
    // M4 / older GPUs: the matmul2d path is correctly absent (steel prefill).
    std::printf("[qwen_m5_fastpath_guard] no matrix cores (M4 path); "
                "gqa_flash_decode=%d\n", model->gqa_flash_decode());
    return;
  }
  // M5+: a 4-bit Qwen prefill MUST engage the matmul2d GEMM (dequant +
  // dense_gemm_mma); only 8-bit weights stay on steel by design. A false here
  // means the matrix-core prefill silently fell back to the ~2-2.5x slower
  // steel quantized GEMM.
  if (mcfg_loaded.quant_bits != 8) {
    EXPECT_TRUE(model->uses_matrix_cores());
  }
  // head_dim-256 prefill flash attention via matmul2d (Qwen3.5). Off => the
  // scalar query-tiled attention (much slower at long prefill).
  if (mcfg_loaded.head_dim == 256 && model->uses_matrix_cores()) {
    EXPECT_TRUE(model->mma_flash_attn());
  }
  std::printf("[qwen_m5_fastpath_guard] matrix cores ON | use_mma=%d "
              "mma_flash_attn=%d gqa_flash_decode=%d\n",
              model->uses_matrix_cores(), model->mma_flash_attn(),
              model->gqa_flash_decode());
}

// PERF-PATH GUARD for the native k-quant (GGUF) path -- the same intent as
// qwen_m5_fastpath_engaged_guard, but for the Q4_K_M GGUF model. On M5 a
// k-quant prefill MUST route dense_gemm_ through the matmul2d matrix units
// (uses_matrix_cores == _use_mma; dequant -> dense_gemm_mma); a regression
// that left it on the steel dequant+dense_gemm_t GEMM stays token-exact
// (~2-2.5x slower) so the token-exact tests miss it, and a perf-floor assert
// would flake on the M5's thermal throttle -- hence a path-engaged assert.
// Loads the model directly (the GGUF ships no tokenizer); the config is read
// from the .gguf. Gated on VPIPE_QWEN_GGUF_TEST_MODEL_PATH; skips under the
// VPIPE_QWEN_NO_MMA A/B disable so it doesn't false-fail.
TEST(metal_lm_smoke, qwen_gguf_m5_fastpath_engaged_guard) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  if (std::getenv("VPIPE_QWEN_NO_MMA")) { return; }   // legit A/B disable
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  const auto& mcfg_loaded = model->config();

  // GQA flash decode -- the decode fast path on both M4 and M5.
  EXPECT_TRUE(model->gqa_flash_decode());

  if (!mc->supports_matrix_cores()) {
    std::printf("[qwen_gguf_fastpath_guard] no matrix cores (M4 path); "
                "gqa_flash_decode=%d\n", model->gqa_flash_decode());
    return;
  }
  // M5+: the k-quant prefill must engage the matmul2d GEMM. A false here means
  // dense_gemm_ silently fell back to the steel quantized-dequant GEMM.
  EXPECT_TRUE(model->uses_matrix_cores());
  if (mcfg_loaded.head_dim == 256 && model->uses_matrix_cores()) {
    EXPECT_TRUE(model->mma_flash_attn());
  }
  std::printf("[qwen_gguf_fastpath_guard] matrix cores ON | use_mma=%d "
              "mma_flash_attn=%d gqa_flash_decode=%d\n",
              model->uses_matrix_cores(), model->mma_flash_attn(),
              model->gqa_flash_decode());
}

// Flash-decode-GQA serial attention (sdpa_paged_gqa_mb256 + sdpa_gqa_merge,
// head_dim 256) must be greedy token-exact with the mb256 per-q-head path.
// Loads the SAME Qwen3.5 model twice -- GQA on (VPIPE_QWEN_GQA_ATTN=1) and
// off (=0, mb256) -- prefills a >128-token prompt (so decode runs the
// long-ctx attention path, _sdpa_mb_min) and greedy-decodes; the token
// streams must match. On a model whose shape the GQA path can't handle
// (head_dim != 256 or Hq/Hkv > 4) both loads fall back to mb256 (still a
// valid, trivial pass). Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_gqa_attn_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  // ~200-token prompt so the decode position exceeds _sdpa_mb_min (128) and
  // the long-context (mb256 / GQA) attention path is exercised.
  std::string prompt;
  for (int i = 0; i < 16; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening, noting "
              "the color of the sky and the names of passing ships. ";
  }

  auto run = [&](int gqa, int no_vec) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_QWEN_GQA_ATTN", gqa ? "1" : "0", 1);
    ::setenv("VPIPE_GQA_NO_VEC", no_vec ? "1" : "0", 1);
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
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref  = run(0, 0);   // mb256 (GQA off)
  const auto allg = run(1, 1);   // flash-decode-GQA all-G (sdpa_paged_gqa)
  const auto vec  = run(1, 0);   // flash-decode-GQA vec (sdpa_paged_gqa_vec)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_QWEN_GQA_ATTN");
  ::unsetenv("VPIPE_GQA_NO_VEC");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == allg.size() && ref.size() == vec.size());
  std::size_t mism_allg = 0, mism_vec = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != allg[i]) { ++mism_allg; }
    if (ref[i] != vec[i]) { ++mism_vec; }
  }
  std::printf("[metal_lm_smoke.qwen_gqa_attn_token_exact] %zu tokens, "
              "allg_mism=%zu vec_mism=%zu\n",
              ref.size(), mism_allg, mism_vec);
  EXPECT_TRUE(mism_allg == 0);
  EXPECT_TRUE(mism_vec == 0);
}

// Flash-decode-GQA on the head_dim-128 decoders must be greedy token-exact
// with the sdpa_paged_mb path. Loads the model twice -- GQA on
// (VPIPE_GQA_ATTN=1) and off (=0) -- prefills a >128-token text prompt (so
// the long-ctx attention path runs) and greedy-decodes; the streams must
// match. Two routings share the same sdpa_paged_gqa kernel at D=128:
//   - VPIPE_QWEN3_ASR_TEST_MODEL_PATH: the dense Qwen3-ASR decoder routes
//     through MetalQwenModel (GQA 16/8=2) -- exercised on this box.
//   - VPIPE_LLM_TEST_MODEL_PATH: a dense LlamaForCausalLM routes through
//     MetalLlamaModel (GQA 32/8=4) -- exercised where a Llama model exists.
// A model whose shape the GQA path can't handle falls back (trivial pass).
TEST(metal_lm_smoke, llama_gqa_attn_token_exact) {
  const char* path = std::getenv("VPIPE_LLM_TEST_MODEL_PATH");
  if (!path || !*path) {
    path = std::getenv("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  }
  if (!path || !*path) { return; }

  std::string prompt;
  for (int i = 0; i < 20; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening, noting "
              "the color of the sky and the names of passing ships. ";
  }

  auto run = [&](int gqa) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GQA_ATTN", gqa ? "1" : "0", 1);
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
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(0);   // sdpa_paged_mb
  const auto got = run(1);   // flash-decode-GQA
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GQA_ATTN");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.llama_gqa_attn_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// The two-stage parallel argmax (argmax_partial -> argmax_combine, the default
// greedy decode path) must be GREEDY TOKEN-EXACT with the single-tg argmax
// (VPIPE_GEMMA_ARGMAX1=1) on REAL model logits -- the direct cross-kernel gate
// for the argmax change. Decodes the same prompt 64 tokens with each kernel
// (next_token_greedy -> decode_step_fast -> encode_argmax_) and requires the
// streams to match bit-for-bit. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_two_stage_argmax_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  auto run = [&](bool force_single) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (force_single) { ::setenv("VPIPE_GEMMA_ARGMAX1", "1", 1); }
    else { ::unsetenv("VPIPE_GEMMA_ARGMAX1"); }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 8;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    std::vector<std::int32_t> ids;
    lm->chat_template()->render_user_turn(
        "List the planets of the solar system in order from the sun.",
        /*is_first_turn=*/true, &ids);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 64 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };
  const auto ref = run(true);     // single-tg argmax
  const auto got = run(false);    // two-stage argmax (default)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_ARGMAX1");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_two_stage_argmax_token_exact] %zu tokens, "
              "%zu mismatches (single-tg vs two-stage)\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Flash-decode-GQA on Gemma-4 (CONTIGUOUS KV + sliding window + ring,
// sdpa_causal_gqa) must be greedy token-exact with the per-q-head sdpa_mb
// path. The prompt exceeds the sliding window (512) so the windowed-range
// scan (first = q_pos-window+1) is exercised on the sliding layers. Loads
// twice -- GQA on (VPIPE_GQA_ATTN=1) and off (=0) -- prefills + greedy-
// decodes; the streams must match. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_gqa_attn_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  std::string prompt;   // ~640 tokens > the 512 sliding window
  for (int i = 0; i < 32; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening, noting "
              "the color of the sky and the names of passing ships. ";
  }

  auto run = [&](int gqa) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GQA_ATTN", gqa ? "1" : "0", 1);
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
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(0);   // sdpa_mb (per-q-head)
  const auto got = run(1);   // flash-decode-GQA (contiguous)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GQA_ATTN");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_gqa_attn_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Materialized paged decode for the global (head_dim 512) layers
// (VPIPE_GEMMA_MAT_DECODE=1) must be greedy token-exact with the base flash
// decode (VECN/MAT unset). This is the omlx/MLX head_dim-512 fallback
// structure: QK GEMV -> parallel softmax -> PV GEMV, replacing the fused
// online-softmax flash. Same softmax math, fp reassociation only, so the
// argmax must not move. Prompt is long enough (>2 global pages @ 512) that the
// QK/PV key scan splits (sp>1) and the rowstat reduction spans a real range.
// Greedy-decodes 48 tokens each and requires the streams to match. Gated on
// VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_mat_decode_matches_flash_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  std::string prompt;   // ~1300 tokens -> 3 global pages @ page_tokens=512
  for (int i = 0; i < 64; ++i) {
    prompt += "The cartographer unrolled the brittle map across the table, "
              "tracing rivers and mountain passes by candlelight. ";
  }

  auto run = [&](bool mat) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (mat) { ::setenv("VPIPE_GEMMA_MAT_DECODE", "1", 1); }
    else     { ::unsetenv("VPIPE_GEMMA_MAT_DECODE"); }
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
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 48 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // base flash decode
  const auto mat = run(true);    // materialized decode
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_MAT_DECODE");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == mat.size());
  std::size_t mm = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != mat[i]) { ++mm; }
  }
  std::printf("[metal_lm_smoke.gemma_mat_decode_matches_flash_token_exact] "
              "%zu tokens, %zu mismatches\n", ref.size(), mm);
  EXPECT_TRUE(mm == 0);
}

// Threadgroup-staged flash-decode for the Gemma-4 GLOBAL layers
// (sdpa_causal_gqa_tile, default ON) must be greedy token-exact with the
// per-q-head sdpa_mb path. Loads the model twice -- gtile ON
// (VPIPE_GEMMA_GTILE_ATTN=1) and OFF (=0, sdpa_mb) -- prefills a prompt long
// enough that the global layers scan a multi-chunk context, and greedy-decodes
// 32 tokens; the streams must match. Engagement is independently confirmed by
// the decode A/B (gtile is measurably faster), so a silent fallback can't fake
// this pass. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_gtile_attn_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  std::string prompt;   // ~640 tokens; global layers scan the full context
  for (int i = 0; i < 32; ++i) {
    prompt += "The cartographer unrolled the chart across the table, tracing "
              "rivers and marking the depths of each harbor in faded ink. ";
  }

  auto run = [&](int gtile) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_GTILE_ATTN", gtile ? "1" : "0", 1);
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
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(0);   // sdpa_mb (per-q-head)
  const auto got = run(1);   // threadgroup-staged flash-decode
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_GTILE_ATTN");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_gtile_attn_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Shared-prefix batched decode attention (phase A reads the N branches' shared
// prefix once, phase B merges each branch's private pages) must match the
// per-branch SDPA. Uses a MULTI-PAGE shared prefix (small page_tokens) so
// shared_pages>=2, N=4 branches at distinct positions, and toggles
// set_shared_attn ON vs OFF in-process to isolate the shared split.
// VPIPE_SDPA_MB_MIN=0 forces the OFF path onto the mb256 kernel too, so the
// only difference is shared-vs-strided splitting of the same online softmax.
//
// Compares LOGITS within a tolerance, not tokens. The two splits sum the same
// online softmax in a different order, so they agree to a few f16 ULP but not
// bit-exactly -- and a token comparison turns that into a coin flip. It really
// is a coin flip: the OFF path once put tokens 0 and 11 at EXACTLY 21.000000
// (top-2 gap 0.000000) where ON separated them by one ULP, so the argmax
// tie-break went different ways and the branch's whole tail diverged (6/64
// positions from a single flip at step 10). Both arms are teacher-forced on
// the OFF arm's tokens so their KV states stay identical the whole way and
// every step's logits are directly comparable -- one flip can no longer
// cascade, and what is left measures the kernels rather than the tie-break.
TEST(metal_lm_smoke, qwen_shared_attn_logits_match) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  if (mcfg.head_dim != 256) { return; }   // shared-attn path is D=256 only
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 32;   // small -> a longish prefix spans several pages
  mcfg.max_pages = 64;
  ::setenv("VPIPE_SDPA_MB_MIN", "0", 1);  // OFF path also uses mb256
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ::unsetenv("VPIPE_SDPA_MB_MIN");
  if (!model) { return; }
  if (!model->shared_attn()) { return; }  // kernels unavailable -> skip
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }

  // A multi-page shared prefix (>= 2 shared pages at page_tokens=32).
  auto prompt = tok->encode(
      "The quick brown fox jumps over the lazy dog near the river bank while "
      "the sun sets slowly behind the distant rolling hills and the wind "
      "carries the scent of pine across the quiet valley below the ridge, and "
      "far away a train whistle echoes through the cool evening air softly.");
  // Need > page_tokens (32) tokens so the shared prefix spans >= 2 pages.
  if ((int)prompt.size() < 34) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid() || model->prefill(root, prompt).empty()) { return; }

  const int N = 4;
  const int n_steps = 16;
  const int V = model->config().vocab;
  auto argmax_of = [&](const float* lg, int n) {
    std::int32_t best = 0; float bv = lg[0];
    for (int v = 1; v < n; ++v) { if (lg[v] > bv) { bv = lg[v]; best = v; } }
    return best;
  };
  // Branch N off the shared prefix (distinct-length suffixes -> distinct
  // positions). `drive` teacher-forces the per-step input tokens; when null
  // the arm picks its own argmax and records what it chose. Returns the
  // [step][branch * V] logits so the two arms can be compared row by row.
  std::vector<std::vector<std::int32_t>> drive_toks;
  auto run = [&](bool on, bool teacher_force) {
    model->set_shared_attn(on);
    auto br = ctxm->branch(root, N);
    std::vector<std::int32_t> cur((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                       (std::int32_t)(100 + i));
      auto l = model->prefill(br[(std::size_t)i], suffix);
      if (l.empty()) { return std::vector<std::vector<float>>{}; }
      cur[(std::size_t)i] = teacher_force
                                ? drive_toks[0][(std::size_t)i]
                                : argmax_of(l.data(), (int)l.size());
    }
    if (!teacher_force) {
      drive_toks.assign(1, cur);   // row 0 = the prefill-chosen input tokens
    }
    std::vector<std::vector<float>> steps;
    std::vector<float> bl;
    for (int st = 0; st < n_steps; ++st) {
      if (!model->decode_batched_step(
              std::span<const genai::ContextId>(br.data(), br.size()),
              std::span<const std::int32_t>(cur.data(), cur.size()),
              std::span<const std::int32_t>(), bl)) {
        break;
      }
      steps.push_back(bl);
      std::vector<std::int32_t> nxt((std::size_t)N);
      for (int i = 0; i < N; ++i) {
        nxt[(std::size_t)i] =
            teacher_force ? drive_toks[(std::size_t)st + 1][(std::size_t)i]
                          : argmax_of(bl.data() + (std::size_t)i * V, V);
      }
      if (!teacher_force) { drive_toks.push_back(nxt); }
      cur = nxt;
    }
    for (auto id : br) { ctxm->release(id); }
    return steps;
  };
  const auto off = run(false, /*teacher_force=*/false);
  const auto on  = run(true,  /*teacher_force=*/true);
  ASSERT_TRUE((int)off.size() == n_steps);
  ASSERT_TRUE(off.size() == on.size());

  // Worst per-element gap and worst per-row relative L2 over every step and
  // branch. Both arms saw the same inputs at every position, so any drift is
  // the attention split alone.
  double max_abs = 0.0, max_rel = 0.0;
  int max_step = -1, max_branch = -1;
  for (std::size_t st = 0; st < off.size(); ++st) {
    for (int i = 0; i < N; ++i) {
      const float* a = on[st].data() + (std::size_t)i * V;
      const float* b = off[st].data() + (std::size_t)i * V;
      double num = 0.0, den = 0.0;
      for (int v = 0; v < V; ++v) {
        const double d = (double)a[v] - (double)b[v];
        if (std::fabs(d) > max_abs) {
          max_abs = std::fabs(d);
          max_step = (int)st;
          max_branch = i;
        }
        num += d * d;
        den += (double)b[v] * (double)b[v];
      }
      const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
      if (rel > max_rel) { max_rel = rel; }
    }
  }
  std::printf("[metal_lm_smoke.qwen_shared_attn] N=%d steps=%d "
              "max|dlogit|=%.6f (step %d branch %d) max rel-L2=%.6g\n",
              N, n_steps, max_abs, max_step, max_branch, max_rel);
  // f16 logits at these magnitudes step by 0.015625, so a few ULP is the
  // floor. Measured over 4 branches x 16 steps on M5: max|dlogit| 0.082-0.090
  // (5-6 ULP) and rel-L2 6.3e-3-6.4e-3, varying a little run to run. The
  // bounds sit ~4x above that -- room for another machine's reduction order,
  // still far under a real divergence, which moves whole logits by O(1)
  // against a signal of O(20-30).
  EXPECT_TRUE(max_abs <= 0.4);
  EXPECT_TRUE(max_rel <= 2.5e-2);
}
