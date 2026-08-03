// gemma-gguf.cc -- The GGUF-loaded Gemma-4 path: text chat, multimodal-leak
// guard, decode benches and category profiles, the strip-down ablations, RMS
// accuracy, pdecode-vs-sync, the SDPA kernel matrix, decode-vs-prefill and the
// sliding-window ring wrap.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// Regression for Gemma-4 text chat (the user-reported "no chat template
// for architecture 'Gemma4ForConditionalGeneration'" + no-MLX load
// failure). Two guarantees, both build-agnostic:
//   1. Loading a metal-supported model does NOT require the caller to set
//      VPIPE_LLM_BACKEND -- a build without MLX defaults to the metal
//      backend (it is the only one), so we UNSET the var first and the
//      load must still succeed. (In the MLX build this loads via MLX.)
//   2. chat_template() is non-null for Gemma-4 (the arm is always-built
//      in make_chat_template), and render_user_turn + prefill produce a
//      valid first token -- i.e. text chat is functional end to end.
// Gated on the Gemma checkpoint (VPIPE_GEMMA4_TEST_MODEL_PATH).
// GGUF q4_0 gemma4 bring-up: load a pure-.gguf dir (config + weights +
// tokenizer all from the GGUF), render a chat turn through the template
// (which prepends <bos> -- Gemma is incoherent without it), greedy-decode,
// and require coherent text. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_text_chat) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  EXPECT_TRUE(tpl->family_name() == "gemma");

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Answer in one word.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  EXPECT_TRUE(ids.front() == 2);          // <bos>

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 24; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[gguf_gemma_text_chat] %zu tok | gen='%s'\n",
              gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Regression: the Gemma-4 12B (qat-q4_0 GGUF) text decoder must NEVER emit
// the multimodal end-of-image / end-of-audio control tokens (<image|>
// 258882 / <audio|> 258883) in text output. On visually-themed text the
// QAT-4bit lm_head assigns <image|> the TOP logit (observed leak:
// "...sketching the memory <image|>topology"); LoadedLanguageModel bakes a
// PERMANENT suppression of exactly these two tokens into the model at load
// -- matching the llama.cpp reference, which masks these two (and only
// these two) to -inf.
//
// This teacher-forces the exact prompt+prefix that triggered the leak and
// asserts the predicted next token is the sensible word (' layout', the
// llama.cpp golden argmax at this position), NOT the control token, and
// that both control ids sit below the winner. Deterministic (greedy).
// Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_no_multimodal_leak) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 1024;
  spec.max_pages     = 4;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto& tok = lm->tokenizer();

  // Tokenizer sanity: the control ids map to the expected literals (a
  // mis-map would print <image|> for a NORMAL id).
  std::int32_t eoi = 258882, eoa = 258883;
  EXPECT_TRUE(tok.decode(std::span<const std::int32_t>(&eoi, 1)) == "<image|>");
  EXPECT_TRUE(tok.decode(std::span<const std::int32_t>(&eoa, 1)) == "<audio|>");

  // The permanent base suppression must be populated (both eoi + eoa) --
  // realtime-vqa folds this into its host-side batched mask, so a regression
  // that empties it would silently re-open the leak on the batched path.
  {
    auto base = lm->base_suppressed_tokens();
    bool has_eoi = false, has_eoa = false;
    for (std::int32_t id : base) {
      if (id == 258882) { has_eoi = true; }
      if (id == 258883) { has_eoa = true; }
    }
    EXPECT_TRUE(has_eoi);
    EXPECT_TRUE(has_eoa);
  }

  // Prompt ("Write a short story about Yi...") + the greedy prefix up to
  // (not including) the leaked <image|>. Position 213 predicts the word
  // after "...sketching the memory".
  static const std::int32_t kPrefix[] = {
      2, 105, 2364, 107, 6974, 496, 2822, 3925, 1003, 54984, 236764, 496,
      21042, 47133, 19042, 236764, 532, 1116, 9338, 61232, 496, 3909, 6347, 529,
      3393, 236761, 11968, 236743, 236778, 236771, 236771, 4171, 236761, 106, 107,
      105, 4368, 107, 100, 45518, 107, 101, 818, 147024, 19462, 529, 1806, 37845,
      53522, 54984, 236858, 236751, 3392, 236764, 30439, 1440, 37676, 3418, 506,
      173152, 15348, 236761, 1701, 1806, 5695, 236764, 1304, 1053, 1010, 79582,
      684, 496, 25556, 528, 506, 5464, 236787, 496, 6571, 20651, 528, 506, 861,
      1494, 236772, 32677, 47133, 600, 1186, 9177, 1208, 13610, 121160, 236761,
      108, 2021, 6533, 1663, 236764, 506, 14510, 3938, 691, 496, 32585, 76692,
      236761, 2282, 54984, 236764, 625, 691, 496, 122400, 607, 886, 199010, 5433,
      236761, 2625, 3782, 236789, 236745, 1164, 1676, 506, 3393, 236793, 1304,
      6345, 1061, 18479, 236761, 2625, 50070, 506, 6818, 75043, 236764, 10685,
      1217, 506, 20974, 93905, 607, 506, 1262, 236761, 236743, 108, 236775, 1509,
      236858, 236751, 711, 496, 20651, 2098, 1304, 71787, 236764, 1116, 6114,
      188312, 699, 78370, 236761, 623, 1509, 236858, 236751, 496, 53970, 92560,
      1781, 108, 5778, 532, 506, 47133, 964, 13710, 1024, 506, 1638, 15612, 1757,
      236764, 496, 47617, 34847, 236772, 1340, 236772, 8281, 14260, 506, 1458,
      531, 162911, 236761, 54984, 16630, 872, 496, 11580, 23037, 13039, 532,
      6074, 130257, 506, 6571};
  std::vector<std::int32_t> ids(kPrefix,
      kPrefix + sizeof(kPrefix) / sizeof(kPrefix[0]));
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t next = lm->prefill(ctx, ids);
  std::int32_t nn = next;
  std::printf("[no_leak] argmax=%d '%s' (golden=11273 ' layout')\n", next,
      tok.decode(std::span<const std::int32_t>(&nn, 1)).c_str());
  // The fix: greedy must not pick either suppressed control token.
  EXPECT_TRUE(next != 258882);
  EXPECT_TRUE(next != 258883);
  // And it matches the llama.cpp golden argmax (' layout', 11273).
  EXPECT_TRUE(next == 11273);
  // Both suppressed ids must sit below the winner (masked to the sentinel).
  const auto& lg = lm->last_logits_host();
  if (!lg.empty() && (int)lg.size() > 258883) {
    EXPECT_TRUE(lg[258882] < lg[11273]);
    EXPECT_TRUE(lg[258883] < lg[11273]);
  }
}

// Opt-in bench: native Q6_K tied embed/lm_head vs the affine8 requant.
// Prints resident memory after load + decode tok/s. Run twice to A/B:
//   VPIPE_GGUF_TEST_MODEL_PATH=<dir> vpipe_test --filter '*q6k_decode_bench'
//   VPIPE_GEMMA_NO_Q6K=1 ... (forces the affine8 path)
// The Q6_K table is lossless and ~25% smaller (vocab*H*6.5625 vs *8 bits).
TEST(metal_lm_smoke, gguf_gemma_q6k_decode_bench) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // phys_footprint (not resident_size): includes IOKit/Metal wired buffers
  // where the weights live, so it reflects the embed-table size delta.
  auto footprint_mb = []() -> double {
    task_vm_info_data_t info{};
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &cnt) != KERN_SUCCESS) {
      return 0.0;
    }
    return static_cast<double>(info.phys_footprint) / (1024.0 * 1024.0);
  };
  const double rss = footprint_mb();

  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(
      "What is the capital of France? Answer in one word.", true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  ASSERT_TRUE(lm->prefill(ctx, ids) >= 0);
  (void)lm->next_token(ctx);              // warm one decode step
  const int N = 48;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) { (void)lm->next_token(ctx); }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  const char* nq = std::getenv("VPIPE_GEMMA_NO_Q6K");
  const bool q6k = !(nq && std::atoi(nq) != 0);
  std::printf("[gguf_q6k_bench] embed=%s rss=%.0f MB decode=%.1f tok/s "
              "(%d steps %.3fs)\n", q6k ? "Q6K" : "affine8", rss,
              static_cast<double>(N) / secs, N, secs);
  EXPECT_TRUE(secs > 0.0);
}

// Parameterized prefill+decode bench mirroring llama.cpp's llama-bench so a
// GGUF model can be A/B'd 1:1 against it:
//   pp@L  -> process L prompt tokens from empty;  tok/s = L / t
//   tg@L  -> prefill L tokens (untimed), then generate G tokens timed
//            via BOTH the synchronous next_token loop AND the pipelined
//            pdecode_* path (vpipe's production decode); tok/s = G / t
// Context sizes from VPIPE_GGUF_BENCH_CTX (default "512,1024,2048,4096"),
// generated-token count from VPIPE_GGUF_BENCH_GEN (default 64). Gated on
// VPIPE_GGUF_TEST_MODEL_PATH. Run:
//   VPIPE_GGUF_TEST_MODEL_PATH=<dir> vpipe_test --filter '*gguf_gemma_pp_tg*'
TEST(metal_lm_smoke, gguf_gemma_pp_tg_bench) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  // Parse the comma-separated context-size list + decode count FIRST: the KV
  // cache capacity (page_tokens * max_pages) must fit the largest ctx + gen,
  // else prefill silently caps and the reported tok/s is garbage.
  std::vector<int> ctxs;
  {
    const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX");
    std::string s = (cs && *cs) ? cs : "512,1024,2048,4096";
    std::size_t i = 0;
    while (i < s.size()) {
      std::size_t j = s.find(',', i);
      if (j == std::string::npos) { j = s.size(); }
      const int v = std::atoi(s.substr(i, j - i).c_str());
      if (v > 0) { ctxs.push_back(v); }
      i = j + 1;
    }
  }
  ASSERT_TRUE(!ctxs.empty());
  const char* gs = std::getenv("VPIPE_GGUF_BENCH_GEN");
  const int G = (gs && *gs) ? std::max(1, std::atoi(gs)) : 64;

  int max_ctx = 0;
  for (const int v : ctxs) { max_ctx = std::max(max_ctx, v); }
  const int page_tokens = 512;
  // room for the deepest prompt + decode budget (+ slack for warmup steps).
  const int need_seq = max_ctx + G + 64;
  int max_pages = (need_seq + page_tokens - 1) / page_tokens;
  if (max_pages < 16) { max_pages = 16; }

  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = page_tokens;
  spec.max_pages     = max_pages;        // sized to max(ctx)+gen above
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Seed token stream: a real rendered turn (keeps <bos> first), then pad
  // with a benign repeated token. Coherence is irrelevant for timing.
  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn(
      "Benchmark.", /*is_first_turn=*/true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  auto make_ids = [&](int L) {
    std::vector<std::int32_t> ids;
    ids.reserve(L);
    ids.push_back(bos);
    for (int k = 1; k < L; ++k) { ids.push_back(fill); }
    return ids;
  };

  // Warm the GPU (first prefill/CB is cold; clock spins up).
  {
    auto wc = lm->make_context();
    ASSERT_TRUE(wc.valid());
    auto wid = make_ids(64);
    ASSERT_TRUE(lm->prefill(wc, wid) >= 0);
    for (int k = 0; k < 4; ++k) { (void)lm->next_token(wc); }
  }

  std::printf("[gguf_pp_tg] gemma4_unified gguf bf16 gen=%d\n", G);
  for (const int L : ctxs) {
    const auto ids = make_ids(L);
    const std::span<const std::int32_t> prompt(ids.data(), ids.size());

    // ---- prefill (pp@L): process L tokens from empty ----
    auto cp = lm->make_context();
    ASSERT_TRUE(cp.valid());
    const auto p0 = std::chrono::steady_clock::now();
    const std::int32_t pf = lm->prefill(cp, ids);
    const double psecs =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - p0).count();
    ASSERT_TRUE(pf >= 0);
    const double pp_tps = (psecs > 0.0) ? (double)L / psecs : 0.0;
    // Early prefill-only print (BEFORE the decode blocks) so SKIP_ATTN probes,
    // which leave logits garbage and can crash pdecode, still report prefill.
    std::printf("[gguf_pp_only] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)\n",
                L, pp_tps, psecs);
    std::fflush(stdout);

    // ---- synchronous decode (tg@L): next_token = host [vocab] readback +
    //      host argmax (the production fallback path when pdecode is off) ----
    double tg_sync = 0.0;
    {
      auto cd = lm->make_context();
      ASSERT_TRUE(cd.valid());
      ASSERT_TRUE(lm->prefill(cd, ids) >= 0);
      (void)lm->next_token(cd);          // warm one step at depth L
      const auto d0 = std::chrono::steady_clock::now();
      for (int k = 0; k < G; ++k) { (void)lm->next_token(cd); }
      const double dsecs =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - d0).count();
      tg_sync = (dsecs > 0.0) ? (double)G / dsecs : 0.0;
    }

    // ---- synchronous GPU-argmax decode (tg@L): next_token_greedy = on-GPU
    //      argmax, no host logit pull, but a fresh command buffer per step ----
    double tg_greedy = 0.0;
    {
      auto cd = lm->make_context();
      ASSERT_TRUE(cd.valid());
      ASSERT_TRUE(lm->prefill(cd, ids) >= 0);
      (void)lm->next_token_greedy(cd);   // warm one step at depth L
      const auto d0 = std::chrono::steady_clock::now();
      for (int k = 0; k < G; ++k) { (void)lm->next_token_greedy(cd); }
      const double dsecs =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - d0).count();
      tg_greedy = (dsecs > 0.0) ? (double)G / dsecs : 0.0;
    }

    // ---- pipelined decode (tg@L, vpipe production path) ----
    double tg_pipe = 0.0;
    {
      auto cd = lm->make_context();
      ASSERT_TRUE(cd.valid());
      const std::int32_t first = lm->prefill(cd, ids);
      ASSERT_TRUE(first >= 0);
      genai::SamplerParams gsp;            // defaults -> argmax-equivalent
      const int budget = G + 8;
      if (lm->pdecode_begin(cd, first, prompt, gsp, budget)) {
        // Run-ahead: PRIME the pipeline to pd.depth (commit until the ring is
        // full), then steady-state refill ONE-ahead right after each wait, so
        // each commit encodes step N+1 WHILE step N runs on the GPU --
        // overlapping dispatch-encoding with GPU execution. The old
        // commit-then-next loop oscillated the ring 0<->1 and never overlapped.
        while (lm->pdecode_commit(cd)) {}     // prime to depth
        for (int k = 0; k < 4; ++k) {         // warm steady-state
          if (lm->pdecode_next(cd) < 0) { break; }
          lm->pdecode_commit(cd);
        }
        int produced = 0;
        const auto d0 = std::chrono::steady_clock::now();
        for (int k = 0; k < G; ++k) {
          if (lm->pdecode_next(cd) < 0) { break; }
          ++produced;
          lm->pdecode_commit(cd);           // refill one ahead (overlaps GPU)
        }
        const double dsecs =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - d0).count();
        tg_pipe = (dsecs > 0.0) ? (double)produced / dsecs : 0.0;
        lm->pdecode_end(cd);
      }
    }

    std::printf("[gguf_pp_tg] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode: sync=%5.2f  greedy=%5.2f  pipe=%5.2f tok/s\n",
                L, pp_tps, psecs, tg_sync, tg_greedy, tg_pipe);
  }
  EXPECT_TRUE(true);
}

// Decode CATEGORY profiler (GGUF gemma4_unified) -- decompose the per-token
// decode GPU cost to chase the llama.cpp decode gap. Loads once with
// VPIPE_GEMMA_CATPROF, then for each DUP category (proj/ffn/lmhead/attn/norm/
// misc, + attn_global/attn_slide) duplicates that category's GPU work and
// diffs decode time vs the `none` baseline -> the delta is the category's
// whole-step cost. Profiles at a low + deep context (the decode attention cost
// grows with KV depth; llama.cpp's flash-decode stays flat -- this isolates how
// much of vpipe's depth-degradation is attention). Gated on
// VPIPE_GGUF_TEST_MODEL_PATH + VPIPE_GEMMA_CATPROF (depths: VPIPE_GGUF_BENCH_CTX
// or default 512,4096).
TEST(metal_lm_smoke, gguf_gemma_decode_catprof) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_CATPROF") == nullptr) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  // Parse depths FIRST so the KV cache (page_tokens*max_pages) fits the
  // deepest profile depth + decode slack (else prefill caps -> degenerate).
  std::vector<int> depths{512, 4096};
  if (const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX")) {
    std::vector<int> v; const char* p = cs;
    while (*p) {
      int x = std::atoi(p);
      if (x > 0) { v.push_back(x); }
      while (*p && *p != ',') { ++p; }
      if (*p == ',') { ++p; }
    }
    if (!v.empty()) { depths = v; }
  }
  int max_depth = 0;
  for (const int v : depths) { max_depth = std::max(max_depth, v); }

  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = std::max(16, (max_depth + 128 + 511) / 512);
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  auto make_ids = [&](int L) {
    std::vector<std::int32_t> ids; ids.reserve(L);
    ids.push_back(bos);
    for (int k = 1; k < L; ++k) { ids.push_back(fill); }
    return ids;
  };

  const int N = 48;
  auto decode_ms = [&](const std::vector<std::int32_t>& ids) -> double {
    auto ctx = lm->make_context();
    if (!ctx.valid() || lm->prefill(ctx, ids) < 0) { return -1.0; }
    (void)lm->next_token_greedy(ctx);          // warm one step at depth
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (lm->next_token_greedy(ctx) < 0) { break; }
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };

  const char* cats[] = {"none", "proj", "ffn", "lmhead", "attn", "norm",
                        "misc", "ple", "attn_global", "attn_slide"};
  const int NC = 10;
  for (int depth : depths) {
    const auto ids = make_ids(depth);
    for (int k = 0; k < 2; ++k) { (void)decode_ms(ids); }   // warm GPU clock
    double best[NC];
    for (int c = 0; c < NC; ++c) {
      ::setenv("VPIPE_GEMMA_DUP_CAT", cats[c], 1);
      double m = 1e18;
      for (int r = 0; r < 3; ++r) { m = std::min(m, decode_ms(ids)); }
      best[c] = m;
    }
    ::unsetenv("VPIPE_GEMMA_DUP_CAT");
    const double T0 = best[0];
    std::printf("[gemma_catprof depth=%-4d] baseline %.1f ms (%.3f ms/tok = "
                "%.2f tok/s); delta = category whole-step GPU cost\n",
                depth, T0, T0 / N, N * 1000.0 / T0);
    for (int c = 1; c < NC; ++c) {
      const double d = best[c] - T0;
      std::printf("[gemma_catprof depth=%-4d] %-12s delta %+7.2f ms "
                  "(%.3f ms/tok) | %5.1f%%\n",
                  depth, cats[c], d, d / N, 100.0 * d / T0);
    }
  }
  EXPECT_TRUE(true);
}

// SINGLE-PREFILL strip ablation (decode). Same categories as gguf_gemma_decode_
// strip, but prefills ONCE per sweep and toggles VPIPE_GEMMA_SKIP_LAYER (+the
// composable SKIP_* knobs, all re-read every decode step) BETWEEN short decode
// windows on the same pdecode session -> ~3 prefills total instead of ~60. The
// re-prefill design wedges the GPU under sustained long-context (8k/16k) prefill
// load; this variant decouples the measurement from prefill so it runs clean at
// depth. Gated on VPIPE_GEMMA_STRIP_ONCE. Intra-sweep depth drift is tiny
// (nconfigs*W steps << depth) and identical for every config (same session).
TEST(metal_lm_smoke, gguf_gemma_decode_strip_once) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_STRIP_ONCE") == nullptr) { return; }
  ::setenv("VPIPE_GEMMA_CATPROF", "1", 1);   // gate the skip path
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  int depth = 8192;
  if (const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX")) {
    const int x = std::atoi(cs);
    if (x > 0) { depth = x; }
  }
  const int W = 24;                          // timed decode tokens per window
  const int SETTLE = 3;                      // untimed steps after each toggle
  const int NSWEEP = 4;
  // All distinct configs measured per sweep (label + SKIP env settings). The
  // empty string for SKIP_LAYER means "no whole-layer strip".
  struct Cfg { const char* label; const char* layer; bool ple, ffn, attn,
               proj, lmhead, embed; };
  const Cfg cfgs[] = {
    {"none",            "",        0,0,0,0,0,0},
    {"strip_sliding",   "sliding", 0,0,0,0,0,0},  // -> all-global
    {"strip_global",    "global",  0,0,0,0,0,0},  // -> all-sliding
    {"glob_noPLE",      "sliding", 1,0,0,0,0,0},
    {"glob_noPLE_noFFN","sliding", 1,1,0,0,0,0},  // 7 global: attn+proj+norm
    {"glob_..noATTN",   "sliding", 1,1,1,0,0,0},  //          proj+norm
    {"glob_..noPROJ",   "sliding", 1,1,1,1,0,0},  //          norm/rope/kv
    {"slid_noPLE_noFFN","global",  1,1,0,0,0,0},  // 35 sliding: attn+proj+norm
    {"slid_..noATTN",   "global",  1,1,1,0,0,0},  //          proj+norm
    {"slid_..noPROJ",   "global",  1,1,1,1,0,0},  //          norm/rope/kv
    {"fixed_only",      "all",     1,0,0,0,0,0},  // embed+lmhead+sample+handoff
    {"fixed_noLM",      "all",     1,0,0,0,1,0},  //  - lm_head
    {"fixed_noLM_noEmb","all",     1,0,0,0,1,1},  //  - embed too
  };
  const int NC = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
  auto apply = [&](const Cfg& c) {
    if (*c.label && *c.layer) { ::setenv("VPIPE_GEMMA_SKIP_LAYER", c.layer, 1); }
    else { ::unsetenv("VPIPE_GEMMA_SKIP_LAYER"); }
    auto tog = [&](const char* e, bool on) {
      if (on) { ::setenv(e, "1", 1); } else { ::unsetenv(e); } };
    tog("VPIPE_GEMMA_SKIP_PLE",    c.ple);
    tog("VPIPE_GEMMA_SKIP_FFN",    c.ffn);
    tog("VPIPE_GEMMA_SKIP_ATTN",   c.attn);
    tog("VPIPE_GEMMA_SKIP_PROJ",   c.proj);
    tog("VPIPE_GEMMA_SKIP_LMHEAD", c.lmhead);
    tog("VPIPE_GEMMA_SKIP_EMBED",  c.embed);
  };
  auto clear_all = [&] {
    ::unsetenv("VPIPE_GEMMA_SKIP_LAYER"); ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
    ::unsetenv("VPIPE_GEMMA_SKIP_FFN"); ::unsetenv("VPIPE_GEMMA_SKIP_ATTN");
    ::unsetenv("VPIPE_GEMMA_SKIP_PROJ"); ::unsetenv("VPIPE_GEMMA_SKIP_LMHEAD");
    ::unsetenv("VPIPE_GEMMA_SKIP_EMBED");
  };

  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  const int sweep_steps = NC * (W + SETTLE) + 32;
  spec.max_pages = std::max(16, (depth + sweep_steps + 511) / 512);
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seedv;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seedv);
  ASSERT_TRUE(!seedv.empty());
  const std::int32_t bos  = seedv.front();
  const std::int32_t fill = seedv.size() > 1 ? seedv[1] : bos;
  std::vector<std::int32_t> ids; ids.reserve(depth);
  ids.push_back(bos);
  for (int k = 1; k < depth; ++k) { ids.push_back(fill); }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());

  // min-of-NSWEEP wall ms/tok per config. Each sweep: 1 prefill, warm, then
  // every config's window back-to-back on the same pdecode session.
  std::vector<double> best(NC, 1e18);
  genai::SamplerParams gsp;                   // defaults -> argmax-equivalent
  for (int sw = 0; sw < NSWEEP + 1; ++sw) {    // +1 warm sweep (discarded)
    clear_all();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, sweep_steps)) {
      lm->pdecode_end(ctx); continue;
    }
    for (int k = 0; k < 4; ++k) {              // warm the pipeline
      if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
    }
    for (int ci = 0; ci < NC; ++ci) {
      apply(cfgs[ci]);
      for (int k = 0; k < SETTLE; ++k) {       // let the toggle take effect
        if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
      }
      const auto t0 = std::chrono::steady_clock::now();
      int got = 0;
      for (int k = 0; k < W; ++k) {
        if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
        ++got;
      }
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      const double mt = got > 0 ? ms / got : 1e18;
      if (sw > 0 && mt < best[ci]) { best[ci] = mt; }
    }
    clear_all();
    lm->pdecode_end(ctx);
  }

  auto B = [&](const char* lbl) -> double {
    for (int i = 0; i < NC; ++i) {
      if (std::string(cfgs[i].label) == lbl) { return best[i]; }
    }
    return 0.0;
  };
  const double base = B("none");
  std::printf("[strip_once depth=%d W=%d sweeps=%d] min wall ms/tok\n",
              depth, W, NSWEEP);
  for (int i = 0; i < NC; ++i) {
    if (i == 0) {
      std::printf("[strip_once] %-18s %.3f ms/tok (baseline)\n",
                  cfgs[i].label, best[i]);
    } else {
      std::printf("[strip_once] %-18s %.3f ms/tok  (-%.3f, %4.1f%%)\n",
                  cfgs[i].label, best[i], base - best[i],
                  100.0 * (base - best[i]) / base);
    }
  }
  // Derived per-category DECODE costs (ms/tok) for the head-to-head with omlx.
  std::printf("[strip_once] === per-category decode cost (ms/tok) ===\n");
  std::printf("[strip_once] sliding TOTAL (35 lyr) %.3f | global TOTAL (7 lyr) "
              "%.3f\n", base - B("strip_sliding"), base - B("strip_global"));
  std::printf("[strip_once] global: SDPA %.3f  QKV+O proj %.3f  norm/rope/kv "
              "%.3f  FFN %.3f  PLE %.3f\n",
              B("glob_noPLE_noFFN") - B("glob_..noATTN"),
              B("glob_..noATTN")   - B("glob_..noPROJ"),
              B("glob_..noPROJ")   - B("fixed_only"),
              B("glob_noPLE")      - B("glob_noPLE_noFFN"),
              B("strip_sliding")   - B("glob_noPLE"));
  std::printf("[strip_once] sliding: SDPA %.3f  QKV+O proj %.3f  norm/rope/kv+"
              "fixed %.3f\n",
              B("slid_noPLE_noFFN") - B("slid_..noATTN"),
              B("slid_..noATTN")    - B("slid_..noPROJ"),
              B("slid_..noPROJ"));
  std::printf("[strip_once] fixed: lm_head %.3f  embed %.3f  norm/argmax/sampler/"
              "handoff %.3f\n",
              B("fixed_only") - B("fixed_noLM"),
              B("fixed_noLM") - B("fixed_noLM_noEmb"),
              B("fixed_noLM_noEmb"));

  // FULL-LOAD DUP sweep (VPIPE_GEMMA_STRIP_DUP): double a category INSIDE the
  // full 42-layer pass (VPIPE_GEMMA_DUP_CAT, read per-step) -> delta vs none =
  // that category's true full-occupancy GPU cost (no isolation clock artifact).
  // norm = all RMSNorm/rope/KV-write; attn_slide/attn_global isolate by type.
  if (std::getenv("VPIPE_GEMMA_STRIP_DUP") != nullptr) {
    const char* dcats[] = {"none", "norm", "attn_slide", "attn_global",
                           "proj", "ffn"};
    const int ND = 6;
    std::vector<double> dbest(ND, 1e18);
    for (int sw = 0; sw < NSWEEP + 1; ++sw) {
      clear_all();
      ::unsetenv("VPIPE_GEMMA_DUP_CAT");
      auto ctx = lm->make_context();
      if (!ctx.valid()) { continue; }
      const std::int32_t first = lm->prefill(ctx, ids);
      if (first < 0) { lm->pdecode_end(ctx); continue; }
      if (!lm->pdecode_begin(ctx, first, prompt, gsp, sweep_steps)) {
        lm->pdecode_end(ctx); continue;
      }
      for (int k = 0; k < 4; ++k) {
        if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
      }
      for (int ci = 0; ci < ND; ++ci) {
        if (ci == 0) { ::unsetenv("VPIPE_GEMMA_DUP_CAT"); }
        else { ::setenv("VPIPE_GEMMA_DUP_CAT", dcats[ci], 1); }
        for (int k = 0; k < SETTLE; ++k) {
          if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
        }
        const auto t0 = std::chrono::steady_clock::now();
        int got = 0;
        for (int k = 0; k < W; ++k) {
          if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
          ++got;
        }
        const double mt = got > 0 ? std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / got : 1e18;
        if (sw > 0 && mt < dbest[ci]) { dbest[ci] = mt; }
      }
      ::unsetenv("VPIPE_GEMMA_DUP_CAT");
      lm->pdecode_end(ctx);
    }
    std::printf("[strip_once] === full-load DUP (delta vs none = cat cost) ===\n");
    for (int ci = 1; ci < ND; ++ci) {
      std::printf("[strip_once] dup %-12s %.3f ms/tok  (+%.3f vs none %.3f)\n",
                  dcats[ci], dbest[ci], dbest[ci] - dbest[0], dbest[0]);
    }
  }
  EXPECT_TRUE(true);
}

// Whole-CATEGORY strip ablation (decode). Removes a layer category's entire
// body (VPIPE_GEMMA_SKIP_LAYER) and reports wall ms/tok, so baseline-minus-
// stripped = that category's wall cost. Mirror of the omlx passthrough-strip
// bench (~/dump/omlx_strip_bench.py) for a head-to-head per-category gap.
// Categories: sliding|global (attn type), shared_kv|own_kv (KV ownership),
// and the 4 individual buckets. Gated on VPIPE_GEMMA_STRIP.
TEST(metal_lm_smoke, gguf_gemma_decode_strip) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_STRIP") == nullptr) { return; }
  ::setenv("VPIPE_GEMMA_CATPROF", "1", 1);   // gate the skip path
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  // Parse depth FIRST so the KV cache (page_tokens*max_pages) fits it + the
  // N decoded tokens of slack; else prefill caps at 8192 -> degenerate at 16k.
  int depth = 2048;
  if (const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX")) {
    const int x = std::atoi(cs);
    if (x > 0) { depth = x; }
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = std::max(16, (depth + 128 + 511) / 512);
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  auto make_ids = [&](int L) {
    std::vector<std::int32_t> ids; ids.reserve(L);
    ids.push_back(bos);
    for (int k = 1; k < L; ++k) { ids.push_back(fill); }
    return ids;
  };

  const int N = 48;
  const auto ids = make_ids(depth);
  // Use the PIPELINED decode path (pdecode_*, vpipe's production decode) so the
  // per-token CPU<->GPU handoff is overlapped -- apples-to-apples with omlx's
  // async_eval, removing the ~1.5 ms sync-handoff confound from every row.
  // Returns total wall ms normalised to N tokens (caller divides by N).
  auto decode_ms = [&](const std::vector<std::int32_t>& seq) -> double {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    const std::span<const std::int32_t> prompt(seq.data(), seq.size());
    const std::int32_t first = lm->prefill(ctx, seq);
    if (first < 0) { return -1.0; }
    genai::SamplerParams gsp;                  // defaults -> argmax-equivalent
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, N + 8)) { return -1.0; }
    for (int k = 0; k < 4; ++k) {              // warm the pipeline
      if (!lm->pdecode_commit(ctx)) { break; }
      if (lm->pdecode_next(ctx) < 0) { break; }
    }
    int produced = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      if (lm->pdecode_next(ctx) < 0) { break; }
      ++produced;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    lm->pdecode_end(ctx);
    return produced > 0 ? ms * (double)N / produced : -1.0;
  };

  const char* strips[] = {"", "sliding", "global", "shared_kv", "own_kv",
                          "slide_own", "global_own", "slide_skip",
                          "global_skip"};
  const int NS = 9;
  for (int k = 0; k < 2; ++k) { (void)decode_ms(ids); }    // warm GPU clock
  double base = 0.0;
  std::printf("[gemma_strip depth=%d N=%d] min-of-5 wall ms/tok\n", depth, N);
  for (int s = 0; s < NS; ++s) {
    if (*strips[s]) { ::setenv("VPIPE_GEMMA_SKIP_LAYER", strips[s], 1); }
    else            { ::unsetenv("VPIPE_GEMMA_SKIP_LAYER"); }
    double m = 1e18;
    for (int r = 0; r < 5; ++r) { m = std::min(m, decode_ms(ids)); }
    const double mt = m / N;
    if (s == 0) {
      base = mt;
      std::printf("[gemma_strip] %-12s %.3f ms/tok (baseline)\n",
                  "none", mt);
    } else {
      std::printf("[gemma_strip] strip %-12s %.3f ms/tok  (-%.3f, %4.1f%%)\n",
                  strips[s], mt, base - mt, 100.0 * (base - mt) / base);
    }
  }
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");

  // Deeper ablation: all-global run (strip sliding -> only the 7 global layers),
  // then toggle PLE. Delta = PLE's wall cost in the all-global config, to
  // head-to-head omlx's PLE handling free of sliding-layer noise.
  auto minms = [&](int reps) {
    double m = 1e18; for (int r = 0; r < reps; ++r) { m = std::min(m, decode_ms(ids)); }
    return m / N;
  };
  ::setenv("VPIPE_GEMMA_SKIP_LAYER", "sliding", 1);
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  (void)decode_ms(ids);
  const double g_ple = minms(5);
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  (void)decode_ms(ids);
  const double g_nople = minms(5);
  // ... and additionally remove the FFN (gate/up + down GEMVs) -> only the
  // attention path + norms remain in the 7 global layers.
  ::setenv("VPIPE_GEMMA_SKIP_FFN", "1", 1);
  (void)decode_ms(ids);
  const double g_noff = minms(5);
  ::unsetenv("VPIPE_GEMMA_SKIP_FFN");
  // Fixed-only floor: strip ALL layers + PLE -> embed + final-norm + lm_head +
  // argmax + the per-token CPU<->GPU handoff. Splits the residual gap into the
  // global-attention path vs the fixed per-token overhead.
  ::setenv("VPIPE_GEMMA_SKIP_LAYER", "all", 1);
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  (void)decode_ms(ids);
  const double fixed_only = minms(5);
  // Fixed-tail split: peel lm_head (262144-vocab GEMV) then embed gather.
  ::setenv("VPIPE_GEMMA_SKIP_LMHEAD", "1", 1);
  (void)decode_ms(ids);
  const double fixed_nolm = minms(5);
  ::setenv("VPIPE_GEMMA_SKIP_EMBED", "1", 1);
  (void)decode_ms(ids);
  const double fixed_nolm_noemb = minms(5);
  ::unsetenv("VPIPE_GEMMA_SKIP_LMHEAD");
  ::unsetenv("VPIPE_GEMMA_SKIP_EMBED");
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");
  std::printf("[gemma_strip] fixed-tail: lm_head %.3f  embed %.3f  "
              "argmax/norm/sampler %.3f ms/tok\n",
              fixed_only - fixed_nolm, fixed_nolm - fixed_nolm_noemb,
              fixed_nolm_noemb);
  std::printf("[gemma_strip] all-global  with-PLE %.3f  no-PLE %.3f  "
              "PLE cost %.3f ms/tok\n", g_ple, g_nople, g_ple - g_nople);
  std::printf("[gemma_strip] all-global  no-PLE %.3f  no-PLE-no-FFN %.3f  "
              "FFN cost %.3f ms/tok (7 global layers)\n",
              g_nople, g_noff, g_nople - g_noff);
  std::printf("[gemma_strip] fixed-only (embed+lm_head+sample+handoff) %.3f | "
              "7 global attn+proj+norm = %.3f ms/tok\n",
              fixed_only, g_noff - fixed_only);
  // Also PLE cost in the FULL run (all 42 layers), for reference.
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  (void)decode_ms(ids);
  const double full_nople = minms(5);
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  std::printf("[gemma_strip] full        with-PLE %.3f  no-PLE %.3f  "
              "PLE cost %.3f ms/tok\n", base, full_nople, base - full_nople);

  // Sliding-attention split: all-SLIDING run (strip global -> 35 sliding
  // layers), no-PLE no-FFN, then peel the SDPA core (SKIP_ATTN) and the QKV+O
  // projections (SKIP_PROJ). Localises the +10%/sliding-layer deficit to the
  // attention kernel vs the projection GEMVs vs the norm/rope/KV-write rest.
  ::setenv("VPIPE_GEMMA_SKIP_LAYER", "global", 1);
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  ::setenv("VPIPE_GEMMA_SKIP_FFN", "1", 1);
  (void)decode_ms(ids);
  const double s_base = minms(5);                 // norm+QKV+SDPA+O (35 sliding)
  ::setenv("VPIPE_GEMMA_SKIP_ATTN", "1", 1);
  (void)decode_ms(ids);
  const double s_noattn = minms(5);               // norm+QKV+O (SDPA removed)
  ::setenv("VPIPE_GEMMA_SKIP_PROJ", "1", 1);
  (void)decode_ms(ids);
  const double s_noproj = minms(5);               // norm+rope+kvwrite (no GEMV)
  ::unsetenv("VPIPE_GEMMA_SKIP_ATTN");
  ::unsetenv("VPIPE_GEMMA_SKIP_PROJ");
  ::unsetenv("VPIPE_GEMMA_SKIP_FFN");
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");
  std::printf("[gemma_strip] all-sliding(35) no-PLE-no-FFN %.3f | "
              "SDPA %.3f  QKV+O proj %.3f  norm/rope/kv+fixed %.3f ms/tok\n",
              s_base, s_base - s_noattn, s_noattn - s_noproj, s_noproj);
  EXPECT_TRUE(true);
}

// RMS-kernel accuracy probe: greedy-decode N tokens via the host-[vocab]-argmax
// path and, at each step, record the token + the top1-top2 logit gap. Lets an
// A/B of two RMS kernels (e.g. RMS_TG 256 vs 512) quantify HOW divergent they
// are: where the streams first differ, how many tokens flip, and -- crucially --
// the top1-top2 gap AT each flip (a tiny gap => a near-tie => benign f32-rounding
// reorder, not a quality regression). Prints tokens= / gaps= CSV + FNV for
// offline diffing. Gated on VPIPE_GEMMA_RMS_ACC (+ VPIPE_GGUF_TEST_MODEL_PATH).
TEST(metal_lm_smoke, gguf_gemma_rms_accuracy) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_RMS_ACC") == nullptr) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = path; spec.compute_dtype = "bf16";
  spec.page_tokens = 512; spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(
      "List the first ten prime numbers.", true, &ids);
  ASSERT_TRUE(!ids.empty());
  int N = 96;
  if (const char* e = std::getenv("VPIPE_GEMMA_RMS_ACC_N")) {
    const int v = std::atoi(e); if (v > 0) { N = v; }
  }
  auto top1_top2 = [&](double& gap) -> std::int32_t {
    const std::vector<float>& lg = lm->last_logits_host();
    if (lg.empty()) { gap = -1.0; return -1; }
    int a1 = 0; float v1 = lg[0];
    for (int i = 1; i < (int)lg.size(); ++i) {
      if (lg[i] > v1) { v1 = lg[i]; a1 = i; }
    }
    float v2 = -1e30f;
    for (int i = 0; i < (int)lg.size(); ++i) {
      if (i != a1 && lg[i] > v2) { v2 = lg[i]; }
    }
    gap = (double)v1 - (double)v2;
    return a1;
  };
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> tok;
  std::vector<double> gaps;
  std::int32_t t = lm->prefill(ctx, ids);
  ASSERT_TRUE(t >= 0);
  { double g; (void)top1_top2(g); tok.push_back(t); gaps.push_back(g); }
  for (int i = 1; i < N; ++i) {
    t = lm->next_token(ctx, t);
    if (t < 0) { break; }
    double g; (void)top1_top2(g);
    tok.push_back(t); gaps.push_back(g);
  }
  std::uint64_t h = 1469598103934665603ULL;
  for (auto x : tok) {
    h ^= (std::uint64_t)(std::uint32_t)x; h *= 1099511628211ULL;
  }
  double mingap = 1e30; int mingi = -1;
  for (int i = 0; i < (int)gaps.size(); ++i) {
    if (gaps[i] >= 0 && gaps[i] < mingap) { mingap = gaps[i]; mingi = i; }
  }
  std::printf("[rms_acc] N=%zu fnv=%016llx mingap=%.4f@%d\n",
              tok.size(), (unsigned long long)h, mingap, mingi);
  std::printf("[rms_acc] tokens=");
  for (auto x : tok) { std::printf("%d,", x); }
  std::printf("\n[rms_acc] gaps=");
  for (auto g : gaps) { std::printf("%.3f,", g); }
  std::printf("\n");
  EXPECT_TRUE(true);
}

// Pipelined-decode token-exactness (GGUF gemma4_unified). pdecode_* greedy
// must produce the SAME token stream as the synchronous next_token loop (host
// [vocab] argmax = ground truth) AND next_token_greedy (on-GPU argmax). A
// mismatch localises a bug to the in-stream embed gather, the GPU argmax, or
// the event-chain KV ordering. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_pdecode_matches_sync) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(
      "List the first ten prime numbers.", true, &ids);
  ASSERT_TRUE(!ids.empty());
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int N = 48;

  // Reference: synchronous next_token (host [vocab] argmax).
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    ref.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
  }

  // On-GPU argmax (next_token_greedy) must match the host argmax.
  std::vector<std::int32_t> grd;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    grd.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      grd.push_back(t);
    }
  }

  // Pipelined greedy (default SamplerParams == argmax-equivalent).
  std::vector<std::int32_t> pipe;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    pipe.push_back(first);
    genai::SamplerParams gsp;             // defaults -> argmax
    ASSERT_TRUE(lm->pdecode_begin(ctx, first, prompt, gsp, N));
    for (int i = 1; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      pipe.push_back(n);
    }
    lm->pdecode_end(ctx);
  }

  std::size_t mism_g = 0, mism_p = 0;
  const std::size_t ng = std::min(ref.size(), grd.size());
  for (std::size_t i = 0; i < ng; ++i) {
    if (ref[i] != grd[i]) { ++mism_g; }
  }
  const std::size_t np = std::min(ref.size(), pipe.size());
  for (std::size_t i = 0; i < np; ++i) {
    if (ref[i] != pipe[i]) { ++mism_p; }
  }
  // FNV-1a over the ref token stream + first 8 ids: cross-config token-exact
  // check (run with/without VPIPE_GEMMA_NO_QKV_FUSE and compare the hash).
  std::uint64_t h = 1469598103934665603ULL;
  for (auto t : ref) { h ^= (std::uint64_t)(std::uint32_t)t; h *= 1099511628211ULL; }
  std::printf("[gguf_pdecode] ref=%zu greedy=%zu pipe=%zu  "
              "greedy_mism=%zu pipe_mism=%zu ref_fnv=%016llx first=",
              ref.size(), grd.size(), pipe.size(), mism_g, mism_p,
              (unsigned long long)h);
  for (std::size_t i = 0; i < ref.size() && i < 8; ++i) {
    std::printf("%d,", ref[i]);
  }
  std::printf("\n");
  ASSERT_TRUE(pipe.size() == ref.size());
  ASSERT_TRUE(grd.size() == ref.size());
  EXPECT_TRUE(mism_g == 0);
  EXPECT_TRUE(mism_p == 0);
}

// Global-layer prefill SDPA kernels (VPIPE_GEMMA_SDPA = flash|dev|staged).
//   * dev must be BIT-IDENTICAL to staged (same BK=8 softmax blocking) ->
//     greedy token-IDENTICAL: a hard gate.
//   * flash (llama.cpp-style Q8/C64, the default) is a different but equally
//     accurate online-softmax; it is NOT bit-identical, so we check it is no
//     WORSE than staged against the scalar serial path (the project's serial
//     reference) and that it stays coherent. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_sdpa_kernels) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_NO_SDPA_DEV");
  ::unsetenv("VPIPE_GEMMA_SDPA");
  ::unsetenv("VPIPE_GEMMA_SCALAR_ATTN");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 16;          // max_seq 8192 -> flash slack at n~600
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // ~600-token prompt so the global/full layers' O(n^2) attention runs.
  std::string para = "The history of computing is long and storied. ";
  std::string big;
  for (int i = 0; i < 40; ++i) { big += para; }
  big += "Summarize the key milestones.";
  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(big, true, &ids);
  ASSERT_TRUE(ids.size() > 256);
  const int N = 48;

  auto gen = [&](const char* sdpa, bool scalar) {
    if (scalar) { ::setenv("VPIPE_GEMMA_SCALAR_ATTN", "1", 1); }
    if (sdpa)   { ::setenv("VPIPE_GEMMA_SDPA", sdpa, 1); }
    std::vector<std::int32_t> out;
    auto ctx = lm->make_context();
    std::int32_t t = lm->prefill(ctx, ids);
    out.push_back(t);
    for (int i = 1; i < N && t >= 0; ++i) {
      t = lm->next_token(ctx, t);
      out.push_back(t);
    }
    ::unsetenv("VPIPE_GEMMA_SCALAR_ATTN");
    ::unsetenv("VPIPE_GEMMA_SDPA");
    return out;
  };
  auto cmp = [](const std::vector<std::int32_t>& a,
                const std::vector<std::int32_t>& b) {
    std::size_t mm = 0;
    const std::size_t m = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < m; ++i) { if (a[i] != b[i]) { ++mm; } }
    return mm;
  };
  // First index where two token streams diverge (-1 == identical over min len).
  auto first_div = [](const std::vector<std::int32_t>& a,
                      const std::vector<std::int32_t>& b) {
    const std::size_t m = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < m; ++i) {
      if (a[i] != b[i]) { return (int)i; }
    }
    return -1;
  };

  const std::vector<std::int32_t> staged = gen("staged", false);
  const std::vector<std::int32_t> dev    = gen("dev", false);
  const std::vector<std::int32_t> flash  = gen("flash", false);
  const std::vector<std::int32_t> scalar = gen(nullptr, true);

  const std::size_t dev_vs_staged = cmp(dev, staged);
  const std::size_t flash_vs_scalar = cmp(flash, scalar);
  const std::size_t staged_vs_scalar = cmp(staged, scalar);
  const std::size_t flash_vs_dev = cmp(flash, dev);
  // The ~600-token prompt is < the sliding ring_cap (window 1024 + chunk
  // 2048 = 3072), so the SLIDING layers run no-wrap -> they take the flash
  // path too. So `flash` here exercises BOTH the global and sliding flash
  // kernels; flash-vs-scalar tracks the whole prefill SDPA, not just global.
  std::printf("[gguf_sdpa] prompt=%zu N=%d | dev-vs-staged=%zu | "
              "flash-vs-scalar=%zu (first_div=%d) | staged-vs-scalar=%zu | "
              "flash-vs-dev=%zu (first_div=%d)\n",
              ids.size(), N, dev_vs_staged, flash_vs_scalar,
              first_div(flash, scalar), staged_vs_scalar, flash_vs_dev,
              first_div(flash, dev));
  ASSERT_TRUE(dev.size() == staged.size() && flash.size() == staged.size());
  // dev is bit-identical to staged -> token-identical (hard gate).
  EXPECT_TRUE(dev_vs_staged == 0);
  // flash uses an fp32 O accumulator + fp32 QK scores, so it is a VALID (not
  // bit-identical) approximation of the serial reference. It must track scalar
  // for a meaningful PREFIX; a broken flash prefill mispredicts from token ~0.
  // It is NOT robust to require flash-vs-scalar==0: a token-16 fp near-tie
  // between the flash/staged/scalar prefills is tipped by the *decode* kernel's
  // rounding, so EXACTLY one of {flash,staged} tie-matches scalar past token 16
  // and the other diverges (~32 tokens) -- and which one flips between the
  // gtile (12B global, fp32) and sdpa_mb decode kernels. That symmetry (same
  // count, same first_div, opposite label) is expected fp noise, not a flash
  // bug. So gate on an EARLY divergence instead. The final correctness gate is
  // decode_matches_prefill (argmax-exact) + metal-flash vs the Python/MLX
  // oracle on the safetensors models (64 GB box, gemma4-12b-bench-results.md).
  const int flash_fd = first_div(flash, scalar);
  EXPECT_TRUE(flash_fd < 0 || flash_fd >= 8);
}

// Metal decode self-consistency A/B (GGUF gemma4_unified). The decode
// forward (qmv / sdpa_mb / qmv_geglu g32 kernels) must produce the SAME
// next-token logits as the prefill forward (steel qmm kernels) at the SAME
// position -- that is the project's token-exact bar. We take the prompt +
// its own prefill argmax token `tok0`, then compute the logits that predict
// the token AFTER tok0 two ways:
//   * decode path:  prefill(prompt) then ONE forced decode step of tok0;
//   * prefill path: prefill(prompt ++ [tok0]) in a single pass (reference).
// They must agree. Since MLX with the identical converted weights is
// coherent, a mismatch localises the bug to a decode-only g32 kernel (this
// is the metal-vs-MLX decode-divergence seen at bring-up, isolated without
// MLX). Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_decode_matches_prefill) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = std::getenv("VPIPE_GGUF_AB_F16") ? "f16" : "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Answer in one word.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  // Decode path: prefill the prompt, then ONE forced decode step of tok0.
  auto cd = lm->make_context();
  ASSERT_TRUE(cd.valid());
  const std::int32_t tok0 = lm->prefill(cd, ids);
  ASSERT_TRUE(tok0 >= 0);
  const std::int32_t dec_next = lm->next_token(cd, tok0);
  ASSERT_TRUE(dec_next >= 0);
  const std::vector<float> Ldec = lm->last_logits_host();  // after tok0

  // Prefill path (reference): prompt ++ [tok0] in one pass.
  std::vector<std::int32_t> ids2 = ids;
  ids2.push_back(tok0);
  auto cp = lm->make_context();
  ASSERT_TRUE(cp.valid());
  const std::int32_t pre_next = lm->prefill(cp, ids2);
  ASSERT_TRUE(pre_next >= 0);
  const std::vector<float> Lpre = lm->last_logits_host();  // after tok0

  // Logit-vector comparison (diagnostic), then the token-exact assertion.
  if (!Lpre.empty() && Ldec.size() == Lpre.size()) {
    double max_abs = 0.0, sum_sq = 0.0, sum_ref = 0.0;
    int amax_d = 0, amax_p = 0;
    for (std::size_t i = 0; i < Lpre.size(); ++i) {
      const double d = std::fabs((double)Ldec[i] - (double)Lpre[i]);
      max_abs = std::fmax(max_abs, d);
      sum_sq += d * d;
      sum_ref += (double)Lpre[i] * (double)Lpre[i];
      if (Ldec[i] > Ldec[amax_d]) { amax_d = (int)i; }
      if (Lpre[i] > Lpre[amax_p]) { amax_p = (int)i; }
    }
    const double rel_l2 = std::sqrt(sum_sq / (sum_ref + 1e-9));
    std::printf("[gguf_decode_ab] tok0=%d prefill_argmax=%d decode_argmax=%d "
                "max_abs=%g rel_l2=%g\n",
                (int)tok0, amax_p, amax_d, max_abs, rel_l2);
  }
  std::printf("[gguf_decode_ab] prefill_next=%d decode_next=%d\n",
              (int)pre_next, (int)dec_next);
  EXPECT_TRUE(dec_next == pre_next);   // decode must match prefill
}

// Sliding-layer ring-wrap decode exactness (GGUF gemma4_unified). The global
// decode attn uses the gtile vec kernel; extending it to the SLIDING (windowed,
// ring-buffered) layers needs a prompt LONG enough to (a) trigger the window
// cutoff (L > window) AND (b) wrap the sliding KV ring (L > window +
// sliding_chunk == ring_cap, 3072 on the 12B) -- the short prompts in
// decode_matches_prefill never exercise either, so a sliding-decode-kernel bug
// in the ring/window math hides there. Same decode-vs-prefill bar: a forced
// decode step at a ring-wrapped position must match the single-pass prefill of
// prompt ++ [tok0] (the flash prefill is the validated sliding reference). Run
// it with the gtile vec sliding kernel (default) AND with VPIPE_GEMMA_GTILE_ATTN
// =0 (sdpa_mb) -- both must pass. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_decode_sliding_ringwrap) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  // Force a bounded sliding ring AND disable the lazy single-pass grow so this
  // test exercises the ACTUAL ring-wrap + staged-sliding path. Pin chunk=2048
  // (ring_cap = window 1024 + 2048 = 3072 < L) and VPIPE_GEMMA_NO_SLIDING_GROW
  // (else a fresh one-shot prefill grows the ring to L and never wraps). Both
  // are read at load; unset after. The wrap path stays reachable in production
  // for incremental (kv_off>0) prefill and when the grow is disabled.
  ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", "2048", 1);
  ::setenv("VPIPE_GEMMA_NO_SLIDING_GROW", "1", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) {
    ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
    ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 16;               // max_seq 8192 (room for L + ring)
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Long prompt: a real rendered turn (<bos> first) padded with a benign token
  // to L. L=4096 > ring_cap(3072) so the sliding layers run BOTH a window
  // cutoff and a ring wrap at the decode position.
  const int L = 4096;
  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  std::vector<std::int32_t> ids;
  ids.reserve(L);
  ids.push_back(bos);
  for (int k = 1; k < L; ++k) { ids.push_back(fill); }

  // Decode path: prefill(prompt), then ONE forced decode of tok0 (sliding
  // kernel at a ring-wrapped position).
  auto cd = lm->make_context();
  ASSERT_TRUE(cd.valid());
  const std::int32_t tok0 = lm->prefill(cd, ids);
  ASSERT_TRUE(tok0 >= 0);
  const std::int32_t dec_next = lm->next_token(cd, tok0);
  ASSERT_TRUE(dec_next >= 0);
  const std::vector<float> Ldec = lm->last_logits_host();

  // Prefill reference: prompt ++ [tok0] in one pass.
  std::vector<std::int32_t> ids2 = ids;
  ids2.push_back(tok0);
  auto cp = lm->make_context();
  ASSERT_TRUE(cp.valid());
  const std::int32_t pre_next = lm->prefill(cp, ids2);
  ASSERT_TRUE(pre_next >= 0);
  const std::vector<float> Lpre = lm->last_logits_host();

  if (!Lpre.empty() && Ldec.size() == Lpre.size()) {
    double max_abs = 0.0, sum_sq = 0.0, sum_ref = 0.0;
    int amax_d = 0, amax_p = 0;
    for (std::size_t i = 0; i < Lpre.size(); ++i) {
      const double d = std::fabs((double)Ldec[i] - (double)Lpre[i]);
      max_abs = std::fmax(max_abs, d);
      sum_sq += d * d;
      sum_ref += (double)Lpre[i] * (double)Lpre[i];
      if (Ldec[i] > Ldec[amax_d]) { amax_d = (int)i; }
      if (Lpre[i] > Lpre[amax_p]) { amax_p = (int)i; }
    }
    std::printf("[gguf_sliding_ringwrap] L=%d tok0=%d prefill_argmax=%d "
                "decode_argmax=%d max_abs=%g rel_l2=%g\n",
                L, (int)tok0, amax_p, amax_d, max_abs,
                std::sqrt(sum_sq / (sum_ref + 1e-9)));
  }
  std::printf("[gguf_sliding_ringwrap] prefill_next=%d decode_next=%d\n",
              (int)pre_next, (int)dec_next);
  EXPECT_TRUE(dec_next == pre_next);   // sliding decode must match prefill
}
