// bench.cc -- The metal_lm_bench suite: prefill + decode throughput under
// greedy and top-p sampling, the context sweeps (Qwen safetensors / GGUF /
// MoE), the MoE ablation and category profiles, framework dispatch costs and
// encoder model teardown. All env-gated.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// ---- Decode/prefill throughput bench (env-gated) -----------------
//
// Times prefill + an N-token decode loop on the metal backend under two
// sampling modes that MUST mirror the omlx (MLX server) side exactly:
//   greedy : argmax (temperature 0)
//   top_p  : temperature + nucleus, seeded.
// Sampling is host-side off last_logits_host(); the top_p path uses a
// top-K nucleus (nth_element, K=256) so the host sort cost stays small
// and representative rather than a naive full-vocab O(V log V) sort.
//
// Env:
//   VPIPE_METAL_LM_SMOKE_MODEL  model dir (reuses the smoke var)
//   VPIPE_METAL_BENCH_TOKENS    decode tokens (default 128)
//   VPIPE_METAL_BENCH_PROMPT    prompt text (default builtin)
//   VPIPE_METAL_BENCH_TEMP      top_p temperature (default 0.7)
//   VPIPE_METAL_BENCH_TOP_P     nucleus p (default 0.9)
//   VPIPE_METAL_BENCH_SEED      rng seed (default 1234)
namespace {

std::int32_t
bench_sample_top_p(const std::vector<float>& logits, std::vector<int>& idx,
                   float temp, float top_p, std::mt19937& rng)
{
  const int V = static_cast<int>(logits.size());
  const int K = std::min(V, 256);
  idx.resize(V);
  for (int i = 0; i < V; ++i) { idx[i] = i; }
  std::nth_element(
      idx.begin(), idx.begin() + (K - 1), idx.end(),
      [&](int a, int b) { return logits[a] > logits[b]; });
  idx.resize(K);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
  const float inv_t = 1.0f / (temp > 0.0f ? temp : 1.0f);
  const float maxl = logits[idx[0]];
  std::vector<double> p(K);
  double sum = 0.0;
  for (int i = 0; i < K; ++i) {
    const double e = std::exp(
        static_cast<double>(logits[idx[i]] - maxl) * inv_t);
    p[i] = e;
    sum += e;
  }
  double cum = 0.0;
  int cut = K;
  for (int i = 0; i < K; ++i) {
    cum += p[i] / sum;
    if (cum >= static_cast<double>(top_p)) { cut = i + 1; break; }
  }
  double kept = 0.0;
  for (int i = 0; i < cut; ++i) { kept += p[i]; }
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng) * kept;
  double acc = 0.0;
  for (int i = 0; i < cut; ++i) {
    acc += p[i];
    if (r <= acc) { return idx[i]; }
  }
  return idx[cut - 1];
}

}  // namespace

TEST(metal_lm_bench, decode) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  const int   n_decode = std::getenv("VPIPE_METAL_BENCH_TOKENS")
      ? std::atoi(std::getenv("VPIPE_METAL_BENCH_TOKENS")) : 128;
  const char* p_env = std::getenv("VPIPE_METAL_BENCH_PROMPT");
  const std::string prompt = (p_env && *p_env) ? p_env
      : "Once upon a time, in a small village nestled between tall "
        "mountains, there lived a curious young inventor named Mira. "
        "Every morning she woke before dawn to tinker in her workshop, "
        "and every evening she filled her notebooks with new ideas. One "
        "autumn day, a stranger arrived at the village gate carrying a "
        "broken machine, and Mira's life changed forever. This is the "
        "story of what happened next, told in full detail:";
  const float temp = std::getenv("VPIPE_METAL_BENCH_TEMP")
      ? (float)std::atof(std::getenv("VPIPE_METAL_BENCH_TEMP")) : 0.7f;
  const float top_p = std::getenv("VPIPE_METAL_BENCH_TOP_P")
      ? (float)std::atof(std::getenv("VPIPE_METAL_BENCH_TOP_P")) : 0.9f;
  const unsigned seed = std::getenv("VPIPE_METAL_BENCH_SEED")
      ? (unsigned)std::strtoul(std::getenv("VPIPE_METAL_BENCH_SEED"),
                               nullptr, 10) : 1234u;

  // Bench backend defaults to metal; override with VPIPE_METAL_BENCH_BACKEND
  // (e.g. "mlx" in the MLX build) to A/B the same harness across paths.
  const char* be = std::getenv("VPIPE_METAL_BENCH_BACKEND");
  const std::string backend = (be && *be) ? be : "metal";
  ::setenv("VPIPE_LLM_BACKEND", backend.c_str(), 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
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

  const auto ids = lm->tokenizer().encode(prompt);
  ASSERT_TRUE(!ids.empty());

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };

  // Warmup: full prefill + a short decode so weights/kernels are hot
  // and wired-resident before any timed run.
  {
    auto wired = lm->wired_scope();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 8 && t >= 0; ++i) { t = lm->next_token(ctx, t); }
  }

  std::vector<int> idx_scratch;
  for (int mode = 0; mode < 2; ++mode) {  // 0 = greedy, 1 = top_p
    const bool greedy = (mode == 0);
    std::mt19937 rng(seed);
    auto wired = lm->wired_scope();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());

    const auto t0 = clk::now();
    std::int32_t pred = lm->prefill(ctx, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(pred >= 0);

    std::int32_t t = greedy ? pred
        : bench_sample_top_p(lm->last_logits_host(), idx_scratch,
                             temp, top_p, rng);
    int produced = 0;
    const auto d0 = clk::now();
    for (int i = 0; i < n_decode; ++i) {
      const std::int32_t nx = lm->next_token(ctx, t);
      if (nx < 0) { break; }
      ++produced;
      t = greedy ? nx
          : bench_sample_top_p(lm->last_logits_host(), idx_scratch,
                               temp, top_p, rng);
    }
    const auto d1 = clk::now();

    const double prefill_s = secs(t1 - t0);
    const double decode_s = secs(d1 - d0);
    std::printf(
        "[BENCH] backend=%s model=%s mode=%s prompt_tok=%zu "
        "prefill_s=%.4f prefill_tps=%.1f decode_n=%d decode_s=%.4f "
        "decode_tps=%.2f temp=%.2f top_p=%.2f\n",
        backend.c_str(), path, greedy ? "greedy" : "top_p", ids.size(),
        prefill_s,
        ids.size() / prefill_s, produced, decode_s,
        produced / decode_s, greedy ? 0.0f : temp,
        greedy ? 1.0f : top_p);
    EXPECT_TRUE(produced >= 1);
  }
}

// Qwen3.5 context-length sweep bench, apples-to-apples with the omlx
// reference (omlx_qwen_ctx_bench.py): the SAME synthetic ids
// ((i*131+7)%2000+10) at ctx {1024,2048,4096}, greedy. Per ctx: a warmup
// context, then a FRESH context timed for prefill + a DEC-step greedy decode
// loop, so metal-compute decode tok/s can be compared head-to-head with
// mlx_lm. Backend defaults to metal (VPIPE_METAL_BENCH_BACKEND overrides for
// an A/B). ctx list via VPIPE_QWEN_CTX_LIST (comma-separated), decode steps
// via VPIPE_QWEN_CTX_DEC. Gated on VPIPE_QWEN_CTX_BENCH_MODEL /
// VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_bench, qwen_ctx_sweep) {
  const char* path = std::getenv("VPIPE_QWEN_CTX_BENCH_MODEL");
  if (!path || !*path) { path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH"); }
  if (!path || !*path) { return; }

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN_CTX_LIST")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {1024, 2048, 4096}; }
  const int dec = std::getenv("VPIPE_QWEN_CTX_DEC")
      ? std::atoi(std::getenv("VPIPE_QWEN_CTX_DEC")) : 64;
  // Default to the greedy fast path (next_token_greedy: on-GPU argmax, no
  // full-vocab host logit pull) -- the apples-to-apples match for omlx, which
  // keeps argmax on device. Set VPIPE_QWEN_CTX_GREEDY=0 to A/B the slow
  // next_token (host logit readback + host argmax) path.
  const bool greedy_fast = !(std::getenv("VPIPE_QWEN_CTX_GREEDY")
      && std::atoi(std::getenv("VPIPE_QWEN_CTX_GREEDY")) == 0);
  // VPIPE_QWEN_CTX_MTP=1: ALSO time greedy MTP speculative decode at each ctx
  // (only when the model carries an MTP head, e.g. Qwen3.5-OptiQ) and report
  // its decode tok/s + tok/round + the speedup over the baseline greedy decode.
  const bool mtp_en = std::getenv("VPIPE_QWEN_CTX_MTP")
      && std::atoi(std::getenv("VPIPE_QWEN_CTX_MTP")) != 0;
  // VPIPE_QWEN_CTX_PIPE (default ON): ALSO time the production PIPELINED decode
  // (pdecode_* event-chain CPU/GPU overlap) per ctx and add decode_pipe_tps to
  // the line -- the apples-to-apples match for omlx's async-pipelined loop
  // (mx.async_eval). Set 0 to skip. The sync greedy stays the baseline column.
  const bool pipe_en = !(std::getenv("VPIPE_QWEN_CTX_PIPE")
      && std::atoi(std::getenv("VPIPE_QWEN_CTX_PIPE")) == 0);

  const char* be = std::getenv("VPIPE_METAL_BENCH_BACKEND");
  const std::string backend = (be && *be) ? be : "metal";
  ::setenv("VPIPE_LLM_BACKEND", backend.c_str(), 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  // The pipelined-decode path keeps the timed context AND the pdecode context
  // live at once, so the KV pool must fit 2 * (max ctx + decode) + slack. Size
  // it off the ctx list (was a flat 32 -> decode_pipe_tps returned 0 past 4k).
  {
    int max_ctx = 0;
    for (const int N : ctxs) { max_ctx = std::max(max_ctx, N); }
    // Each live context rounds its token span UP to whole pages, so size per
    // context then double (timed + pdecode contexts coexist), + slack.
    const int per_ctx_pages =
        (max_ctx + dec + 64 + spec.page_tokens - 1) / spec.page_tokens;
    spec.max_pages = std::max(32, 2 * per_ctx_pages + 4);
  }
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };

  for (int N : ctxs) {
    const auto ids = synth(N);
    auto step = [&](auto& c) -> std::int32_t {
      return greedy_fast ? lm->next_token_greedy(c) : lm->next_token(c);
    };
    {                                       // warmup (separate context)
      auto wired = lm->wired_scope();
      auto ctx = lm->make_context();
      ASSERT_TRUE(ctx.valid());
      std::int32_t t = lm->prefill(ctx, ids);
      (void)t;
      for (int i = 0; i < 4; ++i) { if (step(ctx) < 0) { break; } }
    }
    auto wired = lm->wired_scope();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const auto t0 = clk::now();
    const std::int32_t pred = lm->prefill(ctx, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(pred >= 0);
    int produced = 0;
    const auto d0 = clk::now();
    for (int i = 0; i < dec; ++i) {
      const std::int32_t nx = step(ctx);
      if (nx < 0) { break; }
      ++produced;
    }
    const auto d1 = clk::now();
    const double ps = secs(t1 - t0);
    const double ds = secs(d1 - d0);

    // Pipelined decode (vpipe production pdecode_* path): same depth N, warmed
    // pipeline, event-chained commit/next overlap -- the fair match for omlx's
    // async_eval loop. 0 when unavailable / disabled.
    double pipe_tps = 0.0;
    if (pipe_en) {
      auto cd = lm->make_context();
      if (cd.valid()) {
        const std::int32_t pf = lm->prefill(cd, ids);
        const std::span<const std::int32_t> prompt(ids.data(), ids.size());
        genai::SamplerParams gsp;            // defaults -> argmax-equivalent
        if (pf >= 0 && lm->pdecode_begin(cd, pf, prompt, gsp, dec + 8)) {
          for (int k = 0; k < 4; ++k) {      // warm the pipeline
            if (!lm->pdecode_commit(cd)) { break; }
            if (lm->pdecode_next(cd) < 0) { break; }
          }
          int pp = 0;
          const auto p0 = clk::now();
          for (int k = 0; k < dec; ++k) {
            if (!lm->pdecode_commit(cd)) { break; }
            if (lm->pdecode_next(cd) < 0) { break; }
            ++pp;
          }
          const double pds = secs(clk::now() - p0);
          pipe_tps = (pds > 0.0) ? (double)pp / pds : 0.0;
          lm->pdecode_end(cd);
        }
      }
    }

    std::printf(
        "[BENCH-CTX] backend=%s greedy_fast=%d ctx=%d prefill_s=%.4f "
        "prefill_tps=%.1f decode_n=%d decode_s=%.4f decode_tps=%.2f "
        "decode_pipe_tps=%.2f\n",
        backend.c_str(), greedy_fast ? 1 : 0, N, ps, N / ps, produced, ds,
        produced / ds, pipe_tps);
    EXPECT_TRUE(produced >= 1);

    // Greedy MTP speculative decode at the SAME depth N (OptiQ + MTP head).
    // A fresh context prefilled with the same ids so the spec decode starts at
    // depth N; report decode tok/s + tok/round + the speedup vs the baseline.
    if (mtp_en && lm->mtp_available()) {
      // Seed the drafter's KV with the prefix (the text-chat shipping default)
      // so the MTP number reflects the production config; set before prefill.
      lm->set_mtp_prefix_seed(std::getenv("VPIPE_MTP_NO_SEED") == nullptr);
      auto count_round =
          [](int* prod, int* rounds) {
            return [prod, rounds](std::span<const std::int32_t> t) -> bool {
              *prod += (int)t.size();
              *rounds += 1;
              return true;
            };
          };
      const std::function<bool(std::int32_t)> no_stop;
      {                                       // warm the MTP-fused kernels
        auto wctx = lm->make_context();
        if (wctx.valid()) {
          std::int32_t wf = lm->prefill(wctx, ids);
          int wp = 0, wr = 0;
          if (wf >= 0) {
            lm->mtp_generate(wctx, wf, 8, genai::SamplerParams{}, no_stop,
                             count_round(&wp, &wr));
          }
        }
      }
      auto mwired = lm->wired_scope();
      auto mctx = lm->make_context();
      ASSERT_TRUE(mctx.valid());
      const std::int32_t mf = lm->prefill(mctx, ids);
      ASSERT_TRUE(mf >= 0);
      int mprod = 0, mrounds = 0;
      const auto m0 = clk::now();
      const bool mok = lm->mtp_generate(mctx, mf, dec, genai::SamplerParams{},
                                        no_stop, count_round(&mprod, &mrounds));
      const auto m1 = clk::now();
      const double mds = secs(m1 - m0);
      if (mok && mprod > 0 && mds > 0.0) {
        std::printf(
            "[BENCH-CTX-MTP] backend=%s ctx=%d mtp_decode_n=%d "
            "mtp_decode_s=%.4f mtp_decode_tps=%.2f tok_per_round=%.2f "
            "speedup=%.2f\n",
            backend.c_str(), N, mprod, mds, mprod / mds,
            mrounds > 0 ? (double)mprod / (double)mrounds : 0.0,
            (ds > 0.0) ? (mprod / mds) / (produced / ds) : 0.0);
      }
    }
  }
}

// Qwen3.5 GGUF (Q4_K_M) context-length sweep -- the native k-quant counterpart
// of qwen_ctx_sweep, for the head-to-head against llama.cpp's llama-bench
// (-p L -n G -d L) on the SAME .gguf. The Qwen GGUF ships no tokenizer, so this
// drives MetalQwenModel directly with synthetic in-vocab ids (timing only).
// Per ctx L: a fresh branch-from-(empty-)root context, prefill L timed (pp@L),
// then a pipelined-decode run of G tokens timed (tg@L, vpipe's production decode
// path -- proven token-exact in qwen_gguf_text_chat). On M5 the k-quant prefill
// rides the matmul2d matrix units (dequant -> dense_gemm_mma); A/B the steel
// fallback with VPIPE_QWEN_NO_MMA=1. ctx list VPIPE_QWEN_GGUF_BENCH_CTX
// (default 1024,2048,4096), decode steps VPIPE_QWEN_GGUF_BENCH_GEN (default 64).
// Gated on VPIPE_QWEN_GGUF_TEST_MODEL_PATH. Run:
//   VPIPE_QWEN_GGUF_TEST_MODEL_PATH=<.gguf> \
//     vpipe_test --filter '*qwen_gguf_ctx_sweep' --color off
TEST(metal_lm_bench, qwen_gguf_ctx_sweep) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Size the page pool for the largest requested ctx + decode (the bench may
  // sweep long contexts); 32 (16k) is the default for the smoke ctxs.
  int max_ctx = 4096;
  if (const char* e = std::getenv("VPIPE_QWEN_GGUF_BENCH_CTX")) {
    for (const char* p = e; *p;) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > max_ctx) { max_ctx = (int)v; }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  const int gen_hint = std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN")
      ? std::max(1, std::atoi(std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN"))) : 64;
  mcfg.max_pages = std::max(32, (max_ctx + gen_hint + 511) / 512 + 2);
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN_GGUF_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {1024, 2048, 4096}; }
  const int G = std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN")
      ? std::max(1, std::atoi(std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN"))) : 64;

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  // Same synthetic ids as qwen_ctx_sweep / omlx_qwen_ctx_bench (in-vocab).
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  // Warm the GPU (cold first command buffer; clock spins up).
  {
    const genai::ContextId w = cm->branch(model->root_context());
    ASSERT_TRUE(w.valid());
    const std::int32_t f = argmax(model->prefill(w, synth(64)));
    std::vector<std::int32_t> tmp;
    model->decode_pipelined(w, f, 4, tmp);
    cm->release(w);
  }

  std::printf("[qwen_gguf_ctx] Qwen3.5 gguf q4_K_M use_mma=%d gen=%d\n",
              model->uses_matrix_cores(), G);
  for (const int N : ctxs) {
    const auto ids = synth(N);

    // ---- prefill (pp@N): process N tokens from empty ----
    const genai::ContextId cp = cm->branch(model->root_context());
    ASSERT_TRUE(cp.valid());
    const auto t0 = clk::now();
    const std::vector<float> lg = model->prefill(cp, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(!lg.empty());
    cm->release(cp);
    const double ps = secs(t1 - t0);
    const double pp_tps = ps > 0.0 ? (double)N / ps : 0.0;

    // ---- pipelined decode (tg@N): prefill (untimed) then time G tokens ----
    double tg_pipe = 0.0;
    {
      const genai::ContextId cd = cm->branch(model->root_context());
      ASSERT_TRUE(cd.valid());
      const std::int32_t f = argmax(model->prefill(cd, ids));
      std::vector<std::int32_t> out;             // tok1.. (== decode_pipelined)
      // Prime the pdecode ring + warm 4 steps, THEN time G steady-state steps.
      // Excludes the single-call decode_pipelined path's ~1-tok in-call pipeline
      // fill/drain, so tg_pipe matches qwen_ctx_sweep / gguf_gemma_pp_tg's warmed
      // steady-state accounting (was ~1-1.5% conservative at G=64). Collecting
      // every pdecode_next keeps `out` == decode_pipelined's [tok1..] fingerprint.
      const std::span<const std::int32_t> prompt(ids.data(), ids.size());
      genai::GpuSamplerParams gsp{};             // greedy default -> token-exact
      if (f >= 0 && model->pdecode_begin(cd, f, prompt, gsp, G + 8)) {
        while (model->pdecode_commit(cd)) {}     // prime ring to depth
        for (int k = 0; k < 4; ++k) {            // warm steady-state
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          model->pdecode_commit(cd);
        }
        int produced = 0;
        const auto d0 = clk::now();
        for (int k = 0; k < G; ++k) {
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          ++produced;
          model->pdecode_commit(cd);
        }
        const double ds = secs(clk::now() - d0);
        tg_pipe = (ds > 0.0) ? (double)produced / ds : 0.0;
        model->pdecode_end(cd);
      }
      // Greedy id fingerprint: first 12 decoded ids. Deterministic across
      // attention-kernel variants -> diff new all-G path vs VPIPE_GQA_ATTN=0
      // (old mb256) to confirm token-exactness for G=6.
      std::printf("[qwen_gguf_ctx] ctx=%-5d ids:", N);
      for (std::size_t i = 0; i < out.size() && i < 12; ++i) {
        std::printf(" %d", (int)out[i]);
      }
      std::printf("\n");
      cm->release(cd);
    }

    std::printf("[qwen_gguf_ctx] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode(pipe)=%5.2f tok/s\n", N, pp_tps, ps, tg_pipe);
  }
  EXPECT_TRUE(true);
}

// Qwen3.5-MoE (35B-A3B) prefill/decode context sweep, mirroring
// qwen_gguf_ctx_sweep for the head-to-head against the omlx/mlx_lm server.
// Synthetic in-vocab ids; prints pp@N tok/s + tg@N tok/s for each ctx. Grouped
// prefill is default-on at these n (also lifts the pair-path grid.z cap). Gated
// on VPIPE_QWEN35_MOE_TEST_MODEL_PATH; ctxs via VPIPE_QWEN35_MOE_BENCH_CTX
// (default 1024,2048,4096,8192), gen via VPIPE_QWEN35_MOE_BENCH_GEN (default 64).
TEST(metal_lm_bench, qwen35_moe_ctx_sweep) {
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
  mcfg.max_pages = 48;            // ~24k token KV: holds 8k ctx + decode
  ASSERT_TRUE(mcfg.is_moe());
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN35_MOE_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {1024, 2048, 4096, 8192}; }
  const int G = std::getenv("VPIPE_QWEN35_MOE_BENCH_GEN")
      ? std::max(1, std::atoi(std::getenv("VPIPE_QWEN35_MOE_BENCH_GEN"))) : 64;

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  {   // warm the GPU (cold first command buffer)
    const genai::ContextId w = cm->branch(model->root_context());
    ASSERT_TRUE(w.valid());
    const std::int32_t f = argmax(model->prefill(w, synth(64)));
    std::vector<std::int32_t> tmp;
    model->decode_pipelined(w, f, 4, tmp);
    cm->release(w);
  }

  std::printf("[qwen35_moe_ctx] Qwen3.5-MoE 35B-A3B 4bit gen=%d\n", G);
  for (const int N : ctxs) {
    const auto ids = synth(N);
    // prefill (pp@N)
    const genai::ContextId cp = cm->branch(model->root_context());
    ASSERT_TRUE(cp.valid());
    const auto t0 = clk::now();
    const std::vector<float> lg = model->prefill(cp, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(!lg.empty());
    cm->release(cp);
    const double ps = secs(t1 - t0);
    const double pp_tps = ps > 0.0 ? (double)N / ps : 0.0;
    // pipelined decode (tg@N): prefill (untimed) then time G tokens
    double tg_pipe = 0.0;
    {
      const genai::ContextId cd = cm->branch(model->root_context());
      ASSERT_TRUE(cd.valid());
      const std::int32_t f = argmax(model->prefill(cd, ids));
      std::vector<std::int32_t> out;             // tok1.. (== decode_pipelined)
      // Prime the pdecode ring + warm 4 steps, THEN time G steady-state steps.
      // Excludes the single-call decode_pipelined path's ~1-tok in-call pipeline
      // fill/drain, so tg_pipe matches qwen_ctx_sweep / gguf_gemma_pp_tg's warmed
      // steady-state accounting (was ~1-1.5% conservative at G=64). Collecting
      // every pdecode_next keeps `out` == decode_pipelined's [tok1..] fingerprint.
      const std::span<const std::int32_t> prompt(ids.data(), ids.size());
      genai::GpuSamplerParams gsp{};             // greedy default -> token-exact
      if (f >= 0 && model->pdecode_begin(cd, f, prompt, gsp, G + 8)) {
        while (model->pdecode_commit(cd)) {}     // prime ring to depth
        for (int k = 0; k < 4; ++k) {            // warm steady-state
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          model->pdecode_commit(cd);
        }
        int produced = 0;
        const auto d0 = clk::now();
        for (int k = 0; k < G; ++k) {
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          ++produced;
          model->pdecode_commit(cd);
        }
        const double ds = secs(clk::now() - d0);
        tg_pipe = (ds > 0.0) ? (double)produced / ds : 0.0;
        model->pdecode_end(cd);
      }
      // Greedy id fingerprint (first 12): deterministic across attention-kernel
      // variants -> cross-check the new MMA flash (VPIPE_QWEN_SDPA_PMMA=1) vs
      // the key-split flash (=0) produce identical tokens.
      std::printf("[qwen35_moe_ctx] ctx=%-5d ids:", N);
      for (std::size_t i = 0; i < out.size() && i < 12; ++i) {
        std::printf(" %d", (int)out[i]);
      }
      std::printf("\n");
      cm->release(cd);
    }
    std::printf("[qwen35_moe_ctx] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode(pipe)=%5.2f tok/s\n", N, pp_tps, ps, tg_pipe);
  }

  // Mid-context (q_offset>0) cross-check: a 2-chunk prefill (chunk 2 runs at
  // q_offset=split -> the paged steel kernel) must yield the SAME final argmax
  // as a 1-shot prefill of the concatenation (the last token sees the same
  // causal context). Validates the paged steel kernel's q_offset>0 path; the
  // printed id also cross-checks vs the flash when run with STEEL_ATTN=0.
  {
    // q_offset (existing ctx) + chunk2 size via env; default 1k-existing + 2k.
    const int split = std::getenv("VPIPE_QWEN_MIDCTX_QOFF")
        ? std::atoi(std::getenv("VPIPE_QWEN_MIDCTX_QOFF")) : 1024;
    const int n2 = std::getenv("VPIPE_QWEN_MIDCTX_N")
        ? std::atoi(std::getenv("VPIPE_QWEN_MIDCTX_N")) : 2048;
    const int Nc = split + n2;             // chunk2 n2 >= steel min (2048)
    const auto ids = synth(Nc);
    const std::vector<std::int32_t> a(ids.begin(), ids.begin() + split);
    const std::vector<std::int32_t> b(ids.begin() + split, ids.end());
    const genai::ContextId c1 = cm->branch(model->root_context());
    const std::int32_t one = argmax(model->prefill(c1, ids));
    cm->release(c1);
    const genai::ContextId c2 = cm->branch(model->root_context());
    model->prefill(c2, a);                 // chunk 1 (q_offset=0)
    const auto m0 = clk::now();
    const std::int32_t two = argmax(model->prefill(c2, b));  // chunk 2 (q_off=split)
    const double mms = secs(clk::now() - m0);
    cm->release(c2);
    const double m_tps = mms > 0.0 ? (double)(Nc - split) / mms : 0.0;
    std::printf("[qwen35_moe_midctx] 1shot=%d  2chunk(qoff=%d,n=%d)=%d  %s  "
                "chunk2 prefill=%7.1f tok/s (%.3fs)\n",
                one, split, Nc - split, two, one == two ? "MATCH" : "MISMATCH",
                m_tps, mms);
    EXPECT_TRUE(one == two);
  }
  EXPECT_TRUE(true);
}

// Qwen3.5-MoE prefill ABLATION: load once, time the prefill with each component
// removed (timing-only -- output is garbage), to attribute the prefill cost and
// the gap vs omlx. Toggles drive the in-model VPIPE_MOE_ABL / VPIPE_QWEN_SKIP_
// ATTN env (read per-prefill). delta(base - toggle) = that component's cost.
// Gated on VPIPE_QWEN35_MOE_TEST_MODEL_PATH; ctxs via VPIPE_QWEN35_MOE_BENCH_CTX
// (default 2048,8192 -- the steel grouped tier).
TEST(metal_lm_bench, qwen35_moe_ablation) {
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
  mcfg.max_pages = 48;
  ASSERT_TRUE(mcfg.is_moe());
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN35_MOE_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr; const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {2048, 8192}; }

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  // (label, MOE_ABL value, SKIP_ATTN value). "" leaves the env unset.
  struct Tog { const char* name; const char* abl; const char* attn; };
  const Tog togs[] = {
    {"base",    "",       ""},
    {"-shared", "shared", ""},
    {"-gemm",   "gemm",   ""},   // skip expert GEMM (clean baseline B)
    // Backbone isolated with the expert GEMM ALSO skipped, so the routing
    // change from skipping attention can't perturb the (skipped) experts.
    {"-gemm-fattn", "gemm", "1"},  // B - full-attention layers
    {"-gemm-gdn",   "gemm", "2"},  // B - gated-DeltaNet layers
  };
  auto set_env = [&](const Tog& t) {
    if (*t.abl) { ::setenv("VPIPE_MOE_ABL", t.abl, 1); }
    else { ::unsetenv("VPIPE_MOE_ABL"); }
    if (*t.attn) { ::setenv("VPIPE_QWEN_SKIP_ATTN", t.attn, 1); }
    else { ::unsetenv("VPIPE_QWEN_SKIP_ATTN"); }
  };

  // warm
  { const genai::ContextId w = cm->branch(model->root_context());
    model->prefill(w, synth(64)); cm->release(w); }

  std::printf("[qwen35_moe_abl] Qwen3.5-MoE 35B-A3B (prefill ablation, ms)\n");
  for (const int N : ctxs) {
    const auto ids = synth(N);
    double base_s = 0.0;
    for (const Tog& t : togs) {
      set_env(t);
      // 2 runs, take the min (steady state) -- the env is read per prefill.
      double best = 1e9;
      for (int r = 0; r < 2; ++r) {
        const genai::ContextId cp = cm->branch(model->root_context());
        ASSERT_TRUE(cp.valid());
        const auto t0 = clk::now();
        const auto lg = model->prefill(cp, ids);
        const double s = secs(clk::now() - t0);
        cm->release(cp);
        ASSERT_TRUE(!lg.empty());
        best = std::min(best, s);
      }
      if (std::string(t.name) == "base") { base_s = best; }
      const double delta = base_s - best;   // component cost (>=0 for skips)
      std::printf("[qwen35_moe_abl] ctx=%-5d %-9s prefill=%7.1f ms"
                  "  delta_vs_base=%+7.1f ms (%5.1f tok/s)\n",
                  N, t.name, best * 1e3, delta * 1e3, (double)N / best);
    }
    ::unsetenv("VPIPE_MOE_ABL");
    ::unsetenv("VPIPE_QWEN_SKIP_ATTN");
  }
  EXPECT_TRUE(true);
}

// Qwen3.5-MoE DECODE category profiler AT DEPTH. Within one process (steady GPU
// clock), duplicate one decode category's GPU work per step (VPIPE_QWEN_DUP_CAT)
// -> the whole-step delta vs baseline is that category's cost. Run at two depths
// to separate the context-INDEPENDENT costs (ffn=MoE experts, gdn, proj,
// lmhead) from the context-SCALING one (attn). Finds the MoE decode bottleneck.
// Gated on VPIPE_QWEN35_MOE_TEST_MODEL_PATH + VPIPE_QWEN_CATPROF (the latter
// also enables the in-model DUP path). Depths via VPIPE_QWEN35_MOE_BENCH_CTX.
TEST(metal_lm_bench, qwen35_moe_decode_catprof) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_QWEN_CATPROF") == nullptr) { return; }
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
  mcfg.max_pages = 48;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> depths;
  if (const char* e = std::getenv("VPIPE_QWEN35_MOE_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr; const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { depths.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (depths.empty()) { depths = {2048, 8192}; }

  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t b = 0; float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; b = (std::int32_t)i; }
    }
    return b;
  };
  using clk = std::chrono::steady_clock;
  const int N = 32;
  // Fresh branch + prefill each measurement so the depth is steady (no drift).
  auto decode_ms = [&](const std::vector<std::int32_t>& ids) -> double {
    const genai::ContextId c = cm->branch(model->root_context());
    if (!c.valid()) { return -1.0; }
    const std::int32_t f = argmax(model->prefill(c, ids));
    std::vector<std::int32_t> w;
    model->decode_pipelined(c, f, 2, w);              // warm at depth
    std::vector<std::int32_t> out;
    const std::int32_t seed = w.empty() ? f : w.back();
    const auto t0 = clk::now();
    const bool ok = model->decode_pipelined(c, seed, N, out);
    const double ms =
        std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    cm->release(c);
    return ok ? ms : -1.0;
  };

  const char* cats[] = {"none", "proj", "ffn", "lmhead", "attn", "norm",
                        "rope", "misc", "gdn", "gdn_rec"};
  const int NC = 10;
  for (const int depth : depths) {
    const auto ids = synth(depth);
    for (int k = 0; k < 2; ++k) { (void)decode_ms(ids); }   // warm GPU clock
    double best[NC];
    for (int c = 0; c < NC; ++c) {
      ::setenv("VPIPE_QWEN_DUP_CAT", cats[c], 1);
      double m = 1e18;
      for (int r = 0; r < 2; ++r) {
        const double t = decode_ms(ids); if (t > 0) { m = std::min(m, t); }
      }
      best[c] = m;
    }
    ::unsetenv("VPIPE_QWEN_DUP_CAT");
    const double T0 = best[0];
    std::printf("[moe_catprof depth=%-4d] baseline %.1f ms (%.3f ms/tok = "
                "%.2f tok/s); delta = category whole-step GPU cost\n",
                depth, T0, T0 / N, N * 1000.0 / T0);
    for (int c = 1; c < NC; ++c) {
      const double d = best[c] - T0;
      std::printf("[moe_catprof depth=%-4d] %-8s delta %+7.2f ms (%.3f ms/tok)"
                  " | %5.1f%%\n", depth, cats[c], d, d / N, 100.0 * d / T0);
    }
  }
  EXPECT_TRUE(true);
}

// Per-category decode GPU-cost profiler for the Qwen hybrid metal model.
// Loads with VPIPE_QWEN_CATPROF, then for each DUP category re-runs that
// category's ops ONE extra time per step; the whole-step delta vs baseline is
// that category's GPU cost (compute + its hazard-barrier drains). GDN (the
// 48/64 linear layers) is DC_GDN. The residual (baseline - sum of deltas)
// after wrapping GDN is embed + argmax + any inter-dispatch GPU idle -- a big
// residual means dispatch/barrier gaps (fusion helps); a small one means
// decode is compute/bandwidth bound. Gated on VPIPE_QWEN_CATPROF +
// VPIPE_METAL_LM_SMOKE_MODEL.
TEST(metal_lm_bench, qwen_decode_catprof) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_QWEN_CATPROF") == nullptr) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());

  const int N = 48;
  auto decode_ms = [&]() -> double {
    auto ctx = lm->make_context();
    if (!ctx.valid() || lm->prefill(ctx, seed) < 0) { return -1.0; }
    (void)lm->next_token_greedy(ctx);              // warm one step at depth
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (lm->next_token_greedy(ctx) < 0) { break; }
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };

  const char* cats[] = {"none", "ffn", "gdn", "gdn_rec", "proj", "attn",
                        "lmhead", "norm", "rope", "misc"};
  const int NC = 10;
  for (int k = 0; k < 2; ++k) { (void)decode_ms(); }   // warm GPU clock
  double best[NC];
  for (int c = 0; c < NC; ++c) {
    ::setenv("VPIPE_QWEN_DUP_CAT", cats[c], 1);
    double m = 1e18;
    for (int r = 0; r < 3; ++r) { m = std::fmin(m, decode_ms()); }
    best[c] = m;
  }
  ::unsetenv("VPIPE_QWEN_DUP_CAT");
  const double T0 = best[0];
  std::printf("[qwen_catprof] baseline %.1f ms (%.3f ms/tok = %.2f tok/s); "
              "delta = category whole-step GPU cost (compute + barriers)\n",
              T0, T0 / N, N * 1000.0 / T0);
  double sum = 0.0;
  for (int c = 1; c < NC; ++c) {
    const double d = best[c] - T0;
    sum += d;
    std::printf("[qwen_catprof] %-7s delta %+7.2f ms (%.3f ms/tok) | %5.1f%%\n",
                cats[c], d, d / N, 100.0 * d / T0);
  }
  std::printf("[qwen_catprof] sum-of-deltas %.1f ms (%.1f%%); residual %.1f ms "
              "= embed+argmax+GPU idle gaps\n", sum, 100.0 * sum / T0, T0 - sum);
  EXPECT_TRUE(true);
}

// ============================================================================
// Framework-primitive cost study + MLX command-buffer trace replay.
//
// PURE SYSTEMS investigation (not production perf): quantify what each
// metal-compute command primitive costs, and whether MLX's decode recipe --
// ONE MTLDispatchTypeConcurrent encoder per command buffer + a memoryBarrier
// only at true hazards (a captured Qwen3.5-4B OptiQ decode step measured 815
// dispatches, 523 barriers, 24 concurrent blocks, 20 command buffers; ~36% of
// dispatch boundaries are barrier-free/overlap-eligible) -- beats vpipe's
// recipe (a Serial encoder with cheap implicit ordering + a concurrent SUB-
// encoder, i.e. endEncoding+new-encoder, per parallel block).
//
//   Part 1: isolate each primitive's cost with a near-zero dummy kernel.
//   Part 2: replay the captured trace under both models (VPIPE_MLX_TRACE=file).
// Gated on VPIPE_FW_COSTS.
TEST(metal_lm_bench, framework_dispatch_costs) {
  if (std::getenv("VPIPE_FW_COSTS") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  namespace mcpt = metal_compute;
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dummy_disp_f16");
  if (!fn.valid()) { std::printf("[fw] dummy_disp_f16 missing\n"); return; }

  std::vector<mcpt::SharedBuffer> bufs;
  for (int i = 0; i < 9; ++i) { bufs.push_back(mc->make_shared_buffer(256)); }
  auto bind = [&](mcpt::ComputeEncoder& e) {
    e.set_function(fn);
    for (int i = 0; i < 8; ++i) { e.set_buffer(i, bufs[i]); }
    e.set_buffer(8, bufs[0]);            // out == b0 -> RAW chain (serializes)
    e.set_constant(9, 1); e.set_constant(10, 2);
    e.set_constant(11, 3); e.set_constant(12, 4);
  };
  const mcpt::LaunchDims grid{32, 1, 1}, tg{32, 1, 1};   // near-zero compute
  using Clock = std::chrono::steady_clock;
  auto ms = [](auto d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  auto best = [&](auto&& body) -> double {
    double b = 1e18;
    for (int r = 0; r < 6; ++r) {
      const auto t0 = Clock::now();
      body();
      b = std::min(b, ms(Clock::now() - t0));
    }
    return b;
  };
  const int M = 4000;
  const double tA = best([&] {                 // Serial, implicit ordering
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Serial);
      for (int i = 0; i < M; ++i) { bind(e); e.dispatch(grid, tg); } }
    st.commit().wait();
  });
  const double tC = best([&] {                 // Concurrent, no barrier
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
      for (int i = 0; i < M; ++i) { bind(e); e.dispatch(grid, tg); } }
    st.commit().wait();
  });
  const double tB = best([&] {                 // Concurrent + barrier each
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
      for (int i = 0; i < M; ++i) {
        bind(e); e.dispatch(grid, tg);
        e.memory_barrier(mcpt::BarrierScope::Buffers);
      } }
    st.commit().wait();
  });
  const double tD = best([&] {                 // Serial + concurrent sub-scope
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Serial);
      for (int i = 0; i < M; ++i) {
        auto s = e.concurrent_scope(true);
        bind(e); e.dispatch(grid, tg);
      } }
    st.commit().wait();
  });
  auto split = [&](int K) -> double {          // M dispatches in K cmd buffers
    return best([&] {
      auto st = mc->make_command_stream();
      const int per = M / K;
      for (int k = 0; k < K; ++k) {
        { auto e = st.begin_compute(mcpt::DispatchType::Serial);
          for (int i = 0; i < per; ++i) { bind(e); e.dispatch(grid, tg); } }
        auto f = st.commit();
        if (k == K - 1) { f.wait(); }
      }
    });
  };
  const double t1buf = split(1), t20buf = split(20);
  std::printf("[fw] === per-primitive cost (us/disp), near-zero kernel M=%d ===\n",
              M);
  std::printf("[fw]  serial no-barrier      %.3f\n", tA / M * 1e3);
  std::printf("[fw]  concurrent no-barrier  %.3f   (pure launch+overlap)\n",
              tC / M * 1e3);
  std::printf("[fw]  concurrent + barrier   %.3f   -> memoryBarrier ~ %.3f us\n",
              tB / M * 1e3, (tB - tC) / M * 1e3);
  std::printf("[fw]  serial + enc-boundary  %.3f   -> enc-boundary(x2) ~ %.3f "
              "us\n", tD / M * 1e3, (tD - tA) / M * 1e3);
  std::printf("[fw]  commit: 1 buf %.3f ms, 20 bufs %.3f ms -> per-commit ~ "
              "%.3f us\n", t1buf, t20buf, (t20buf - t1buf) / 19.0 * 1e3);

  // Part 2: replay the captured MLX trace under both models.
  const char* tracef = std::getenv("VPIPE_MLX_TRACE");
  if (tracef == nullptr || *tracef == '\0') {
    std::printf("[fw] (set VPIPE_MLX_TRACE=<file> for trace replay)\n");
    return;
  }
  std::vector<char> ev;
  { std::ifstream f(tracef); std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) { continue; }
      const char c = line[0];
      if (c == 'd' || c == 't') { ev.push_back('d'); }
      else if (c == 'B' || c == 'C' || c == '[' || c == ']') { ev.push_back(c); }
    } }
  std::size_t nd = 0, nb = 0, nc = 0, nk = 0;
  for (char c : ev) { nd += (c == 'd'); nb += (c == 'B'); nk += (c == 'C');
                      nc += (c == '['); }
  // MLX model: one Concurrent encoder per command buffer; memoryBarrier at each
  // B; new buffer at each C. (The [ ] blocks are already reflected in where B
  // does/doesn't appear, so they need no separate handling here.)
  const double tMlx = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == 'B') { e.memory_barrier(mcpt::BarrierScope::Buffers); }
      else if (c == 'C') { e.end(); st.commit();
                           e = st.begin_compute(mcpt::DispatchType::Concurrent); }
    }
    e.end(); st.commit().wait();
  });
  // vpipe model: Serial encoder (implicit ordering, NO explicit barrier) + a
  // concurrent sub-encoder scope per [ ] block; new buffer at each C.
  const double tVp = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Serial);
    std::optional<mcpt::ComputeEncoder::ConcurrentScope> cs;
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == '[') { cs.emplace(e.concurrent_scope(true)); }
      else if (c == ']') { cs.reset(); }
      else if (c == 'C') { cs.reset(); e.end(); st.commit();
                           e = st.begin_compute(mcpt::DispatchType::Serial); }
    }
    cs.reset(); e.end(); st.commit().wait();
  });
  // MLX model but a SINGLE command buffer (ignore C) -> isolates how much of
  // MLX's edge is the concurrent-encoder overlap vs the ~20-way cmdbuf split.
  const double tMlx1 = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == 'B') { e.memory_barrier(mcpt::BarrierScope::Buffers); }
    }
    e.end(); st.commit().wait();
  });
  // "vpipe adopts MLX's recipe": concurrent encoder, barrier only at the 523
  // hazards, but keep vpipe's single command buffer per step (no split).
  const double tVpConc = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == 'B') { e.memory_barrier(mcpt::BarrierScope::Buffers); }
    }
    e.end(); st.commit().wait();
  });
  std::printf("[fw] === trace replay (framework overhead only, near-zero "
              "kernel) ===\n");
  std::printf("[fw]  trace: %zu dispatch, %zu barrier, %zu concblk, %zu "
              "cmdbuf\n", nd, nb, nc, nk);
  std::printf("[fw]  MLX model  (concurrent + barriers, %zu bufs) : %.3f ms\n",
              nk, tMlx);
  std::printf("[fw]  MLX model  (concurrent + barriers, 1 buf)    : %.3f ms  "
              "(split gain = %.1f%%)\n", tMlx1, 100.0 * (tMlx1 - tMlx) / tMlx1);
  std::printf("[fw]  vpipe now  (serial + conc sub-enc, %zu bufs) : %.3f ms  "
              "(+%.1f%% vs MLX)\n", nk, tVp, 100.0 * (tVp - tMlx) / tMlx);
  std::printf("[fw]  vpipe-conc (concurrent + barriers, 1 buf)    : %.3f ms\n",
              tVpConc);
  EXPECT_TRUE(true);
}

// ============================================================================
// Encoder-model + hazard-mode + command-buffer-split teardown (systems study).
//
// Resolves the contradiction between "concurrent+barrier is slower than serial
// for vpipe's mostly-serial decode" (prior result) and "MLX's concurrent model
// is faster" (framework study). Root cause: MLX allocates ALL buffers Untracked
// (manual hazards), vpipe defaults to Tracked (driver auto-inserts barriers on a
// Concurrent encoder). So vpipe + concurrent + manual barrier = DOUBLE barriers.
//
// Distinct buffers per dispatch (no artificial sharing): a dependent RAW chain
// (dispatch i reads pool[i], writes pool[i+1]) models the serial decode; an
// independent variant (all read pool[0..7], write pool[i+8]) models parallel
// siblings. Sweeps command-buffer split size to answer commands-vs-buffers.
// Gated on VPIPE_ENC_MODEL.
TEST(metal_lm_bench, encoder_model_teardown) {
  if (std::getenv("VPIPE_ENC_MODEL") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  namespace mcpt = metal_compute;
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dummy_disp_f16");
  if (!fn.valid()) { std::printf("[enc] dummy_disp_f16 missing\n"); return; }

  const int MAXM = 832;
  auto mkpool = [&](mcpt::HazardTracking ht) {
    std::vector<mcpt::SharedBuffer> p;
    for (int i = 0; i < MAXM + 16; ++i) {
      p.push_back(mc->make_shared_buffer(131072, 64, ht));   // 128KB: >64KB heap
                                                             // thr -> both non-heap
                                                             // (isolate tracking)
    }
    return p;
  };
  auto tracked = mkpool(mcpt::HazardTracking::Tracked);
  auto untrack = mkpool(mcpt::HazardTracking::Untracked);
  using Clock = std::chrono::steady_clock;
  auto msf = [](auto d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  auto best = [&](auto&& body) -> double {
    double b = 1e18;
    for (int r = 0; r < 6; ++r) {
      const auto t0 = Clock::now();
      body();
      b = std::min(b, msf(Clock::now() - t0));
    }
    return b;
  };
  const mcpt::LaunchDims g{32, 1, 1}, tgd{32, 1, 1};
  // bmode: 0=no manual barrier, 1=manual barrier before every dispatch.
  auto run = [&](std::vector<mcpt::SharedBuffer>& pool, int M, int S,
                 mcpt::DispatchType dt, int bmode, bool indep) -> double {
    return best([&] {
      auto st = mc->make_command_stream();
      int i = 0;
      while (i < M) {
        const int end = std::min(i + S, M);
        { auto e = st.begin_compute(dt);
          for (int j = i; j < end; ++j) {
            if (bmode == 1 && j > i) {
              e.memory_barrier(mcpt::BarrierScope::Buffers);
            }
            e.set_function(fn);
            if (indep) {
              for (int k = 0; k < 8; ++k) { e.set_buffer(k, pool[k]); }
              e.set_buffer(8, pool[j + 8]);
            } else {
              e.set_buffer(0, pool[j]);
              for (int k = 1; k < 8; ++k) { e.set_buffer(k, pool[0]); }
              e.set_buffer(8, pool[j + 1]);
            }
            e.set_constant(9, 1); e.set_constant(10, 2);
            e.set_constant(11, 3); e.set_constant(12, 4);
            e.dispatch(g, tgd);
          } }
        auto f = st.commit();
        if (end >= M) { f.wait(); }
        i = end;
      }
    });
  };
  const auto SER = mcpt::DispatchType::Serial;
  const auto CON = mcpt::DispatchType::Concurrent;

  std::printf("[enc] === A. cmd-buffer split sweep (Tracked serial chain) ===\n");
  for (int M : {400, 800}) {
    for (int S : {M, 200, 100, 50, 25, 10}) {
      const double t = run(tracked, M, S, SER, 0, false);
      std::printf("[enc]  M=%d  S=%3d cmds/buf (%2d bufs): %6.2f ms  %.3f "
                  "us/disp\n", M, S, (M + S - 1) / S, t, t / M * 1e3);
    }
  }
  const int M = 512, S = 50;
  std::printf("[enc] === B. encoder x hazard model, DEPENDENT chain "
              "(M=%d,S=%d) ===\n", M, S);
  std::printf("[enc]  Tracked  serial  no-barrier   : %6.2f ms\n",
              run(tracked, M, S, SER, 0, false));
  std::printf("[enc]  Tracked  concur  no-barrier(driver auto): %6.2f ms\n",
              run(tracked, M, S, CON, 0, false));
  std::printf("[enc]  Tracked  concur  manual-every(DOUBLE)   : %6.2f ms\n",
              run(tracked, M, S, CON, 1, false));
  std::printf("[enc]  Untrack  serial  no-barrier   : %6.2f ms\n",
              run(untrack, M, S, SER, 0, false));
  std::printf("[enc]  Untrack  concur  manual-every(MLX chain): %6.2f ms\n",
              run(untrack, M, S, CON, 1, false));
  std::printf("[enc] === C. encoder x hazard model, INDEPENDENT (M=%d,S=%d) "
              "===\n", M, S);
  std::printf("[enc]  Tracked  serial  (serializes indep)     : %6.2f ms\n",
              run(tracked, M, S, SER, 0, true));
  std::printf("[enc]  Tracked  concur  no-barrier(driver->overlap): %6.2f ms\n",
              run(tracked, M, S, CON, 0, true));
  std::printf("[enc]  Untrack  concur  no-barrier(overlap ceiling): %6.2f ms\n",
              run(untrack, M, S, CON, 0, true));

  // D. Is the Untracked edge CPU-encode (hidden by pdecode in production) or
  // GPU-side? Repeat the serial chain with a COSTED kernel (residual_add over N
  // elts = real DRAM traffic per dispatch). If the Tracked-vs-Untracked gap
  // shrinks toward 0 as the kernel does real work, the edge is CPU-encode only.
  auto radd = lib.function("residual_add_f16");
  if (radd.valid()) {
    auto runc = [&](std::vector<mcpt::SharedBuffer>& pool, int N) -> double {
      return best([&] {
        auto st = mc->make_command_stream();
        { auto e = st.begin_compute(SER);
          for (int j = 0; j < M; ++j) {
            e.set_function(radd);
            e.set_buffer(0, pool[j]); e.set_buffer(1, pool[0]);
            e.set_buffer(2, pool[j + 1]); e.set_constant(3, N);
            e.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
          } }
        st.commit().wait();
      });
    };
    std::printf("[enc] === D. Tracked vs Untracked SERIAL, costed kernel "
                "(residual_add) ===\n");
    for (int N : {64, 1024, 8192}) {
      const double tt = runc(tracked, N), tu = runc(untrack, N);
      std::printf("[enc]  N=%5d (%3dKB/disp): Tracked %6.2f  Untracked %6.2f  "
                  "gap %+.1f%%\n", N, N * 2 / 1024, tt, tu,
                  100.0 * (tt - tu) / tt);
    }
  }
  EXPECT_TRUE(true);
}
