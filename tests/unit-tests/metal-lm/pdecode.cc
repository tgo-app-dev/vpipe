// pdecode.cc -- Pipelined decode (pdecode_*): greedy-matches-sync, rollback
// correctness, sampling self-consistency and the pipeline benches, on Gemma-4
// e4b and Qwen.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// Per-token GPU-resident pipelined decode (LoadedLanguageModel::pdecode_*):
// greedy output must be token-identical to the synchronous next_token loop,
// and the sampled path must be deterministic given a seed + coherent.
TEST(metal_pdecode, greedy_matches_sync) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  if (sess.metal_compute() == nullptr || !sess.metal_compute()->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto ids = lm->tokenizer().encode(
      "Tell me a short story about a curious robot.");
  ASSERT_TRUE(!ids.empty());
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int N = 48;

  // Reference: synchronous greedy next_token loop.
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    ref.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
  }

  // Pipelined greedy (default SamplerParams is argmax-equivalent).
  std::vector<std::int32_t> pipe;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    pipe.push_back(first);
    genai::SamplerParams gsp;   // temperature 1 + all-default -> argmax
    ASSERT_TRUE(lm->pdecode_begin(ctx, first, prompt, gsp, N));
    for (int i = 1; i < N; ++i) {
      ASSERT_TRUE(lm->pdecode_commit(ctx));
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      pipe.push_back(n);
    }
    lm->pdecode_end(ctx);
  }

  ASSERT_TRUE(ref.size() == pipe.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != pipe[i]) { ++mism; }
  }
  std::printf("[metal_pdecode] greedy: %zu tokens, %zu mismatches vs sync\n",
              pipe.size(), mism);
  EXPECT_TRUE(mism == 0u);

  // Sampled path: deterministic given a fixed seed; two runs must match,
  // and the GPU sampler must honour penalties/top-k without crashing.
  auto run_sampled = [&](std::uint64_t seed) {
    std::vector<std::int32_t> out;
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    const std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return out; }
    out.push_back(first);
    genai::SamplerParams sp;
    sp.temperature = 0.8f;
    sp.top_p = 0.95f;
    sp.top_k = 40;
    sp.repetition_penalty = 1.1f;
    sp.seed = seed;
    if (!lm->pdecode_begin(ctx, first, prompt, sp, N)) { return out; }
    for (int i = 1; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      out.push_back(n);
    }
    lm->pdecode_end(ctx);
    return out;
  };
  const auto s1 = run_sampled(12345u);
  const auto s2 = run_sampled(12345u);
  ASSERT_TRUE(s1.size() >= 2u);
  ASSERT_TRUE(s1.size() == s2.size());
  for (std::size_t i = 0; i < s1.size(); ++i) { EXPECT_TRUE(s1[i] == s2[i]); }
  const auto txt = lm->tokenizer().decode(
      std::span<const std::int32_t>(s1.data(), s1.size()));
  std::printf("[metal_pdecode] sampled(seed=12345): %zu tok | '%s'\n",
              s1.size(), txt.c_str());
  EXPECT_TRUE(!txt.empty());
}

// Lever #2 A/B: pipelined run-ahead decode vs the synchronous next_token
// loop. Measures whether overlapping the CPU command-buffer encode of token
// N+1 with the GPU execution of token N (pdecode depth>=2) recovers the
// per-token CPU-encode bubble. Token-exact gate: depth-1 AND depth-2 pdecode
// must reproduce the synchronous next_token_greedy stream exactly.
// Gated on VPIPE_GEMMA4_TEST_MODEL_PATH + VPIPE_GEMMA_PDECODE_BENCH; context
// length via VPIPE_GEMMA_BATCH_PREFIX_LEN (default short).
TEST(metal_lm_smoke, gemma_e4b_pdecode_pipeline_bench) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_GEMMA_PDECODE_BENCH")) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  std::string ptext = "The weather today is";
  if (const char* e = std::getenv("VPIPE_GEMMA_BATCH_PREFIX_LEN")) {
    const int want = std::atoi(e);
    ptext.clear();
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
  }
  auto ids = lm->tokenizer().encode(ptext);
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int K = 64;
  using clock = std::chrono::steady_clock;

  // Reference + timing: synchronous on-GPU-argmax greedy (decode_step_fast).
  std::vector<std::int32_t> ref;
  double sync_s = 0.0;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int i = 0; i < 4; ++i) {          // warm
      t = lm->next_token_greedy(ctx, t);
      ref.push_back(t);
    }
    const auto t0 = clock::now();
    for (int i = 0; i < K; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
    sync_s = std::chrono::duration<double>(clock::now() - t0).count();
  }

  // Pipelined depth d: prefill, then run-ahead commit/next for warm+K tokens.
  auto run_pipe = [&](int depth, std::vector<std::int32_t>& out) -> double {
    ::setenv("VPIPE_GEMMA_PDECODE_DEPTH", depth >= 2 ? "2" : "1", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return -1.0; }
    out.push_back(first);
    genai::SamplerParams gsp;                  // defaults -> argmax
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, K + 8)) { return -1.0; }
    const int warm = 4;
    int committed = 0, emitted = 0;
    auto pump = [&](int target) {
      while (emitted < target) {
        while (committed < K + 4 && lm->pdecode_commit(ctx)) { ++committed; }
        const std::int32_t n = lm->pdecode_next(ctx);
        if (n < 0) { break; }
        out.push_back(n); ++emitted;
      }
    };
    pump(warm);                              // fill + warm
    const auto t0 = clock::now();
    pump(warm + K);
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    lm->pdecode_end(ctx);
    ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH");
    return dt;
  };

  std::vector<std::int32_t> p1, p2;
  const double d1_s = run_pipe(1, p1);
  const double d2_s = run_pipe(2, p2);

  auto mism = [&](const std::vector<std::int32_t>& a) {
    std::size_t m = 0;
    const std::size_t n = std::min(ref.size(), a.size());
    for (std::size_t i = 0; i < n; ++i) { if (ref[i] != a[i]) { ++m; } }
    return m;
  };
  const std::size_t m1 = mism(p1), m2 = mism(p2);
  // FNV-1a fingerprint of the greedy ref tokens -- lets an external A/B confirm
  // token-exactness across a toggle (e.g. VPIPE_GEMMA_PLE_CONCURRENT on vs off).
  std::uint64_t fp = 1469598103934665603ull;
  for (auto t : ref) {
    fp = (fp ^ (std::uint64_t)(std::uint32_t)t) * 1099511628211ull;
  }
  std::printf("[pdecode-ab] ctx=%zu K=%d | sync %.1f tok/s | pipe d1 %.1f "
              "tok/s | pipe d2 %.1f tok/s | d1_mism=%zu d2_mism=%zu ref_fp=%016llx\n",
              ids.size(), K, sync_s > 0 ? K / sync_s : 0.0,
              d1_s > 0 ? K / d1_s : 0.0, d2_s > 0 ? K / d2_s : 0.0, m1, m2,
              (unsigned long long)fp);
  EXPECT_TRUE(m1 == 0);
  EXPECT_TRUE(m2 == 0);
}

// GPU SAMPLING decode (temp>0) self-consistency + coherence + tok/s A/B for
// the histogram sampler. The GPU sampler is deterministic given a fixed base
// seed (per-step seed = base + golden*(step+1)), so a depth-1 and a depth-2
// pdecode run with the SAME params MUST be token-identical -- the self-
// consistency gate. Then it greedily-checks the sampled text is non-degenerate
// (not all one token) and prints sampling-decode tok/s. Run twice -- default
// (histogram) and VPIPE_GEMMA_SAMPLE1=1 (old single-tg) -- to A/B the
// end-to-end sampling-decode win; the test itself only gates self-consistency.
// Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_e4b_sampling_pdecode_selfconsistent) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  if (const char* e = std::getenv("VPIPE_GEMMA_SAMPLE_PREFIX_LEN")) {
    // Room for the long prefix + K decode tokens (page_tokens 512).
    spec.max_pages = (std::atoi(e) + 256) / 512 + 2;
  }
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm && lm->valid());

  std::vector<std::int32_t> ids;
  // Long-context A/B: VPIPE_GEMMA_SAMPLE_PREFIX_LEN pads the prompt to ~N
  // tokens (e.g. 8192) so the sampling-decode tok/s is measured at depth in the
  // KV (the sampler cost is context-independent, but decode wall is not).
  if (const char* e = std::getenv("VPIPE_GEMMA_SAMPLE_PREFIX_LEN")) {
    const int want = std::atoi(e);
    std::string ptext;
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
    ids = lm->tokenizer().encode(ptext);
  } else {
    lm->chat_template()->render_user_turn(
        "Write two sentences about the ocean.", true, &ids);
  }
  ASSERT_TRUE(!ids.empty());
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int K = 64;

  genai::SamplerParams sp;            // temp>0 + top_p<1 -> GPU sampler path
  sp.temperature = 0.7f;
  sp.top_p       = 0.9f;
  sp.top_k       = 40;
  sp.min_p       = 0.0f;
  sp.seed        = 0xBEEFCAFEull;     // fixed base -> deterministic

  using clock = std::chrono::steady_clock;
  auto run = [&](int depth, std::vector<std::int32_t>& out) -> double {
    ::setenv("VPIPE_GEMMA_PDECODE_DEPTH", depth >= 2 ? "2" : "1", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return -1.0; }
    out.push_back(first);
    if (!lm->pdecode_begin(ctx, first, prompt, sp, K + 8)) { return -1.0; }
    int committed = 0, emitted = 0;
    const auto t0 = clock::now();
    while (emitted < K) {
      while (committed < K + 4 && lm->pdecode_commit(ctx)) { ++committed; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      out.push_back(n); ++emitted;
    }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    lm->pdecode_end(ctx);
    ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH");
    return dt;
  };

  std::vector<std::int32_t> a, b;
  const double sa = run(1, a);
  const double sb = run(2, b);

  // Self-consistency: deterministic sampler -> depth-1 == depth-2, 0 mismatch.
  std::size_t mism = 0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) { if (a[i] != b[i]) { ++mism; } }
  // Non-degenerate: more than 4 distinct ids in the sampled stream.
  std::vector<std::int32_t> uniq(a.begin(), a.end());
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::string txt = lm->tokenizer().decode(
      std::span<const std::int32_t>(a.data(), a.size()));
  const bool sample1 = std::getenv("VPIPE_GEMMA_SAMPLE1") != nullptr;
  std::printf("[gemma-sample] %s | d1 %.1f tok/s | d2 %.1f tok/s | "
              "selfconsist_mism=%zu | distinct=%zu/%zu\n",
              sample1 ? "OLD(sample1)" : "NEW(hist)",
              sa > 0 ? K / sa : 0.0, sb > 0 ? K / sb : 0.0,
              mism, uniq.size(), a.size());
  std::printf("[gemma-sample] text: %.200s\n", txt.c_str());
  ASSERT_TRUE(a.size() == b.size());
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(uniq.size() > 4);
}

// Depth-2 run-ahead KV-ROLLBACK correctness. Depth>=2 speculatively commits
// (and KV-appends) the forward for token i+1 before the host has confirmed
// token i isn't a stop -- so on stop, pdecode_end must roll the KV back to
// the last produced token, matching the synchronous loop (where a stop
// token's KV is never appended). The decisive check is NOT just the token
// stream but CONTINUING decode on the same context afterward: if rollback
// left seq_len too high or a stale slot in-window, the continuation diverges.
// Gated on VPIPE_GEMMA4_TEST_MODEL_PATH (a correctness gate, not a bench).
TEST(metal_lm_smoke, gemma_e4b_pdecode_rollback_correct) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  auto ids = lm->tokenizer().encode("Tell me a short story about a robot.");
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int S = 8;      // tokens produced before the (simulated) stop
  const int K2 = 24;    // continuation tokens decoded on the same context

  // Reference: synchronous. Produce t1..tS (KV up to t_{S-1}; tS's KV not
  // appended -- the stop), then continue K2 more from tS.
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int k = 1; k < S; ++k) { t = lm->next_token_greedy(ctx, t);
                                  ref.push_back(t); }
    std::int32_t c = ref.back();           // tS, the stop token
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   ref.push_back(c); }
  }

  // Speculative: depth-2 run-ahead produces t1..tS leaving one speculative
  // commit (KV-appended) in flight; pdecode_end rolls it back. Then continue
  // K2 from tS on the SAME context via the synchronous path.
  std::vector<std::int32_t> gen;
  {
    ::setenv("VPIPE_GEMMA_PDECODE_DEPTH", "2", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH"); return; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH"); return; }
    gen.push_back(first);
    genai::SamplerParams gsp;
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, S + 4)) {
      ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH"); return;
    }
    const int target = S - 1;              // emit t2..tS
    int committed = 0, drained = 0;
    auto can_commit = [&]() { return committed < target + 1; };  // +1 spec
    while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    while (drained < target) {
      const std::int32_t nx = lm->pdecode_next(ctx);
      if (nx < 0) { break; }
      gen.push_back(nx); ++drained;
      while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    }
    lm->pdecode_end(ctx);                  // rolls back the speculative tail
    ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH");
    std::int32_t c = gen.back();           // tS
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   gen.push_back(c); }
  }

  std::size_t mism = 0;
  const std::size_t n = std::min(ref.size(), gen.size());
  for (std::size_t i = 0; i < n; ++i) { if (ref[i] != gen[i]) { ++mism; } }
  std::printf("[pdecode-rollback] ref=%zu gen=%zu (S=%d K2=%d) mism=%zu\n",
              ref.size(), gen.size(), S, K2, mism);
  ASSERT_TRUE(gen.size() == ref.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 single-stream pdecode (depth-D ring) must match the synchronous
// on-GPU-argmax greedy stream token-for-token. Gates the depth-ring refactor
// at depth-1 and is the correctness gate for the GDN ssm-ring run-ahead.
// Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_pdecode_matches_sync) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  auto ids = lm->tokenizer().encode("List the first ten prime numbers.");
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int N = 48;

  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
  }

  std::vector<std::int32_t> pipe;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return; }
    pipe.push_back(first);
    genai::SamplerParams gsp;                 // defaults -> argmax
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, N)) { return; }
    for (int i = 1; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      pipe.push_back(n);
    }
    lm->pdecode_end(ctx);
  }

  std::size_t mism = 0;
  const std::size_t n = std::min(ref.size(), pipe.size());
  for (std::size_t i = 0; i < n; ++i) { if (ref[i] != pipe[i]) { ++mism; } }
  std::printf("[qwen_pdecode] ref=%zu pipe=%zu mism=%zu\n",
              ref.size(), pipe.size(), mism);
  ASSERT_TRUE(pipe.size() == ref.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 depth-2 run-ahead rollback correctness -- the GATE for the GDN
// ssm/conv recurrent-state ring. A depth-2 pdecode produces S tokens leaving
// ONE speculative commit (its KV appended AND its GDN ssm/conv state
// advanced) in flight; pdecode_end must roll BOTH back. Then we CONTINUE
// decoding K2 tokens on the SAME context via the synchronous path -- the
// continuation must match a fully-synchronous run token-for-token. Unlike the
// dense gemma rollback test, this also exercises the recurrent ssm/conv ring:
// without restoring the GDN state the continuation diverges immediately (the
// recurrence carries the discarded token's effect forward). Gated on
// VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_pdecode_rollback_correct) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  auto ids = lm->tokenizer().encode("List the first ten prime numbers.");
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int S = 8;      // tokens produced before the (simulated) stop
  const int K2 = 24;    // continuation tokens decoded on the same context

  // Reference: synchronous. Produce t1..tS (KV/GDN up to t_{S-1}; tS's forward
  // not committed -- the stop), then continue K2 more from tS.
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int k = 1; k < S; ++k) { t = lm->next_token_greedy(ctx, t);
                                  ref.push_back(t); }
    std::int32_t c = ref.back();           // tS, the stop token
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   ref.push_back(c); }
  }

  // Speculative: depth-2 run-ahead produces t1..tS leaving one speculative
  // commit (KV-appended + GDN-advanced) in flight; pdecode_end rolls back the
  // paged KV AND the GDN ssm/conv ring. Then continue K2 from tS on the SAME
  // context via the synchronous path.
  std::vector<std::int32_t> gen;
  {
    ::setenv("VPIPE_QWEN_PDECODE_DEPTH", "2", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH"); return; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH"); return; }
    gen.push_back(first);
    genai::SamplerParams gsp;
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, S + 4)) {
      ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH"); return;
    }
    const int target = S - 1;              // emit t2..tS
    int committed = 0, drained = 0;
    auto can_commit = [&]() { return committed < target + 1; };  // +1 spec
    while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    while (drained < target) {
      const std::int32_t nx = lm->pdecode_next(ctx);
      if (nx < 0) { break; }
      gen.push_back(nx); ++drained;
      while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    }
    lm->pdecode_end(ctx);                  // rolls back KV + GDN ring
    ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH");
    std::int32_t c = gen.back();           // tS
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   gen.push_back(c); }
  }

  std::size_t mism = 0;
  const std::size_t n = std::min(ref.size(), gen.size());
  for (std::size_t i = 0; i < n; ++i) { if (ref[i] != gen[i]) { ++mism; } }
  std::printf("[qwen-rollback] ref=%zu gen=%zu (S=%d K2=%d) mism=%zu\n",
              ref.size(), gen.size(), S, K2, mism);
  ASSERT_TRUE(gen.size() == ref.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 pdecode A/B: synchronous vs run-ahead depth-1 vs depth-2 (the GDN
// ssm/conv ring). Measures the run-ahead win and reconfirms token-exactness
// at each depth. Gated on VPIPE_QWEN_PDECODE_BENCH (+ the model path); set
// VPIPE_QWEN_PDECODE_PREFIX_LEN to sweep prefill length. Mirrors
// gemma_e4b_pdecode_pipeline_bench.
TEST(metal_lm_smoke, qwen_pdecode_pipeline_bench) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_QWEN_PDECODE_BENCH")) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 128;   // 128*512 = 64k ctx headroom for long-context sweeps
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  std::string ptext = "The weather today is";
  if (const char* e = std::getenv("VPIPE_QWEN_PDECODE_PREFIX_LEN")) {
    const int want = std::atoi(e);
    ptext.clear();
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
  }
  auto ids = lm->tokenizer().encode(ptext);
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int K = 64;
  using clock = std::chrono::steady_clock;

  // Reference + timing: synchronous on-GPU-argmax greedy.
  std::vector<std::int32_t> ref;
  double sync_s = 0.0;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int i = 0; i < 4; ++i) { t = lm->next_token_greedy(ctx, t);
                                  ref.push_back(t); }
    const auto t0 = clock::now();
    for (int i = 0; i < K; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
    sync_s = std::chrono::duration<double>(clock::now() - t0).count();
  }

  auto run_pipe = [&](int depth, std::vector<std::int32_t>& out) -> double {
    ::setenv("VPIPE_QWEN_PDECODE_DEPTH", depth >= 2 ? "2" : "1", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return -1.0; }
    out.push_back(first);
    genai::SamplerParams gsp;                  // defaults -> argmax
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, K + 8)) { return -1.0; }
    const int warm = 4;
    int committed = 0, emitted = 0;
    auto pump = [&](int target) {
      while (emitted < target) {
        while (committed < K + 4 && lm->pdecode_commit(ctx)) { ++committed; }
        const std::int32_t n = lm->pdecode_next(ctx);
        if (n < 0) { break; }
        out.push_back(n); ++emitted;
      }
    };
    pump(warm);
    const auto t0 = clock::now();
    pump(warm + K);
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    lm->pdecode_end(ctx);
    ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH");
    return dt;
  };

  std::vector<std::int32_t> p1, p2;
  const double d1_s = run_pipe(1, p1);
  const double d2_s = run_pipe(2, p2);

  auto mism = [&](const std::vector<std::int32_t>& a) {
    std::size_t m = 0;
    const std::size_t n = std::min(ref.size(), a.size());
    for (std::size_t i = 0; i < n; ++i) { if (ref[i] != a[i]) { ++m; } }
    return m;
  };
  const std::size_t m1 = mism(p1), m2 = mism(p2);
  std::printf("[qwen-pdecode-ab] ctx=%zu K=%d | sync %.1f tok/s | pipe d1 %.1f "
              "tok/s | pipe d2 %.1f tok/s | d1_mism=%zu d2_mism=%zu\n",
              ids.size(), K, sync_s > 0 ? K / sync_s : 0.0,
              d1_s > 0 ? K / d1_s : 0.0, d2_s > 0 ? K / d2_s : 0.0, m1, m2);
  EXPECT_TRUE(m1 == 0);
  EXPECT_TRUE(m2 == 0);
}
