// qwen-text.cc -- Qwen (and Llama) text generation: safetensors / GGUF / OptiQ-
// mixed / MoE chat smokes, the flash + MMA prefill token-exact checks and the
// GGUF byte-level tokenizer round trip.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"
#include "common/flex-bag.h"
#include "pipeline/resource-plan.h"
#include "stages/model-memory.h"

#include <atomic>
#include <filesystem>
#include <thread>

#include <sys/sysctl.h>
#include <unistd.h>

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

// The three MoE prefill tiers compute one function: a ~350-token turn of
// real text (past the steel tier's density line of ~4 routed rows an
// expert, short of the old 1024-row one) prefilled with each forced --
// per-pair gather, the grouped GEMV, the steel grouped GEMM. The grouped
// GEMV reads the same weights in the same order as the pair gather, so it
// is BIT-IDENTICAL to it. The steel tile stages its dequantized weights
// in f16 and accumulates in another order, so it lands at the model's own
// rounding floor -- MEASURED 0.057 on the last row's logits, against
// 0.065 between the pair gather in f16 and in bf16 -- and is held to that
// floor, measured here, with the same pick everywhere. Those three run on
// the steel kernels (VPIPE_QWEN_MOE_NAX=0); a fourth runs the steel tier
// as a GPU with matrix cores does (the routed experts' NAX twins, the
// shared expert on matmul2d): ENGAGED there and held to the same floor,
// bit-identical to the steel arm everywhere else. On a box that cannot
// hold the experts they live in the expert cache (bit-identical to
// resident). VPIPE_QWEN35_MOE_TEST_MODEL_PATH (Qwen3.6-35B-A3B-4bit).
TEST(metal_lm_smoke, qwen35_moe_prefill_tiers_agree) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  if (!cfg.has_value()) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  ASSERT_TRUE(tok != nullptr);
  if (!tok) { return; }
  auto tpl = genai::make_chat_template(*cfg, *tok);
  ASSERT_TRUE(tpl != nullptr);
  if (!tpl) { return; }
  std::string text =
      "Summarize the following notes on memory planning in two sentences. ";
  const char* para =
      "A graph routinely asks one machine to hold several large models at "
      "once: a diffusion transformer, its text encoder, a VAE, sometimes a "
      "language model beside them. Weights are what a model holds; scratch "
      "is what it allocates to run, and for a decode the second can dwarf "
      "the first. Every stage declares what it will hold before anything "
      "loads, so the plan is the same whichever stage loads first. ";
  for (int i = 0; i < 4; ++i) { text += para; }
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(text, /*is_first_turn=*/true, &ids);
  const int n = (int)ids.size();
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  ASSERT_TRUE(mcfg.is_moe());
  ASSERT_TRUE(n >= 256 && n < 1024 && (long long)n * mcfg.top_k >=
                                          4LL * mcfg.n_experts);
  // A box under 48 GB keeps the experts in a 4 GB cache (a 16 GB M5).
  std::uint64_t ram = 0;
  std::size_t ram_len = sizeof(ram);
  sysctlbyname("hw.memsize", &ram, &ram_len, nullptr, 0);
  if (ram != 0 && ram < (48ull << 30)) {
    mcfg.expert_cache_bytes = (std::size_t)4 << 30;
  }
  struct Arm {
    const char* name;
    const char* grouped;
    const char* steel;
    const char* nax;
    bool        bf16;
  };
  // The f16 arms share one load; the bf16 pair gather, the floor, loads
  // after it is released.
  const Arm arms[] = {{"pair", "0", "0", "0", false},
                      {"gemv", "1", "0", "0", false},
                      {"steel", "1", "1", "0", false},
                      {"nax", "1", "1", "1", false},
                      {"pair bf16", "0", "0", "0", true}};
  std::vector<std::vector<float>> out;
  std::unique_ptr<genai::MetalQwenModel> model;
  bool model_bf16 = false, matrix_cores = false;
  std::uint64_t nax_gemms = 0, shared_mma = 0;
  for (const Arm& a : arms) {
    if (!model || model_bf16 != a.bf16) {
      model.reset();
      mc->memory_budget_settled();
      mcfg.use_bf16 = a.bf16;
      model = genai::MetalQwenModel::load(path, mc, mcfg);
      model_bf16 = a.bf16;
    }
    ASSERT_TRUE(model != nullptr);
    if (!model) { return; }
    ::setenv("VPIPE_QWEN_MOE_GROUPED", a.grouped, 1);
    ::setenv("VPIPE_QWEN_MOE_STEEL", a.steel, 1);
    ::setenv("VPIPE_QWEN_MOE_NAX", a.nax, 1);
    const std::uint64_t g0 = model->moe_nax_gemms();
    const std::uint64_t s0 = model->moe_shared_mma_gemms();
    const auto cid = model->context_manager()->acquire_root();
    out.push_back(model->prefill(cid, ids));
    model->context_manager()->release(cid);
    if (std::string(a.name) == "nax") {
      matrix_cores = model->uses_matrix_cores();
      nax_gemms = model->moe_nax_gemms() - g0;
      shared_mma = model->moe_shared_mma_gemms() - s0;
    } else {
      EXPECT_TRUE(model->moe_nax_gemms() == g0 &&
                  model->moe_shared_mma_gemms() == s0);
    }
  }
  ::unsetenv("VPIPE_QWEN_MOE_GROUPED");
  ::unsetenv("VPIPE_QWEN_MOE_STEEL");
  ::unsetenv("VPIPE_QWEN_MOE_NAX");
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, r = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      d += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
      r += (double)a[i] * a[i];
    }
    return r > 0.0 ? std::sqrt(d / r) : 0.0;
  };
  auto argmax = [](const std::vector<float>& v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
  };
  for (const auto& o : out) {
    ASSERT_TRUE(!o.empty() && o.size() == out[0].size());
    if (o.empty() || o.size() != out[0].size()) { return; }
  }
  const double gp = rel(out[0], out[1]), sp = rel(out[0], out[2]);
  const double np_ = rel(out[0], out[3]), ns = rel(out[2], out[3]);
  const double floor = rel(out[0], out[4]);
  std::printf("[qwen35_moe] %d-token prefill%s, last row from the per-pair "
              "gather: grouped GEMV %.5f, steel %.5f, %s %.5f (from steel "
              "%.5f; %llu NAX expert GEMMs, %llu shared on matmul2d); f16 vs "
              "bf16 %.5f; picks %d %d %d %d %d\n", n,
              mcfg.expert_cache_bytes ? " (experts cached)" : "", gp, sp,
              matrix_cores ? "matrix cores" : "steel again", np_, ns,
              (unsigned long long)nax_gemms, (unsigned long long)shared_mma,
              floor, argmax(out[0]), argmax(out[1]), argmax(out[2]),
              argmax(out[3]), argmax(out[4]));
  EXPECT_TRUE(gp == 0.0);
  EXPECT_TRUE(sp > 0.0);   // the steel tier engaged
  EXPECT_TRUE(floor > 0.0 && sp < 1.25 * floor);
  if (matrix_cores) {
    // Every layer's two expert GEMMs and two shared ones, at least.
    EXPECT_TRUE(nax_gemms >= 2ull * mcfg.n_layers);
    EXPECT_TRUE(shared_mma >= 2ull * mcfg.n_layers);
    EXPECT_TRUE(np_ > 0.0 && np_ < 1.25 * floor);
  } else {
    EXPECT_TRUE(ns == 0.0 && nax_gemms == 0 && shared_mma == 0);
  }
  for (int i = 1; i < 5; ++i) {
    EXPECT_TRUE(argmax(out[0]) == argmax(out[i]));
  }
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

// A language model KEPT WARM (GenerativeModelManager::keep_warm, text-chat's
// `keep_loaded`): once its last user lets it go, the next load of the same
// spec is handed the SAME model -- no read, no warmup; a launch declaring
// other weights releases it before that launch loads; and its time runs
// out on its own. Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, language_model_kept_warm) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;

  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const genai::LoadedLanguageModel* first = lm.get();
  mgr->keep_warm(lm, path, 60.0);
  lm.reset();
  EXPECT_TRUE(mgr->warm_count() == 1);
  EXPECT_TRUE(mgr->cached_count() == 1);   // its last user is gone
  const auto t0 = std::chrono::steady_clock::now();
  auto again = mgr->load(spec);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  EXPECT_TRUE(again.get() == first);
  EXPECT_TRUE(ms < 50.0);
  again.reset();

  // A launch that holds other weights: released before it loads.
  mgr->clear_declarations();
  mgr->declare_weights(path, 1 << 20);
  EXPECT_TRUE(mgr->release_warm_unclaimed() == 0);   // it holds this one
  mgr->clear_declarations();
  EXPECT_TRUE(mgr->release_warm_unclaimed() == 0);   // it holds nothing
  mgr->declare_weights("/nonexistent/another-model", 1 << 20);
  EXPECT_TRUE(mgr->release_warm_unclaimed() == 1);
  mgr->clear_declarations();
  EXPECT_TRUE(mgr->warm_count() == 0);
  EXPECT_TRUE(mgr->cached_count() == 0);             // unloaded

  // Its time runs out on its own.
  auto third = mgr->load(spec);
  ASSERT_TRUE(third != nullptr);
  mgr->keep_warm(third, path, 0.3);
  third.reset();
  EXPECT_TRUE(mgr->warm_count() == 1);
  for (int i = 0; i < 50 && mgr->warm_count() > 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_TRUE(mgr->warm_count() == 0);
  EXPECT_TRUE(mgr->cached_count() == 0);
}

// A language model WIRED AS IT LOADS (load_spec::kWireWeights, text-chat's
// `wire_weights`): its buffers mlock'd as they are allocated, so the
// compressor cannot take the weights mid-read; what the load left wired is
// charged to the wired pool while the model lives and given back once it is
// gone. Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, language_model_wired_as_it_loads) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  bag::set_flag(spec.extra, genai::load_spec::kWireWeights, true);
  const std::size_t pool0 = mgr->wired_pool_used();
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const std::size_t charged = mgr->wired_pool_used() - pool0;
  // Most of the checkpoint (the 9B OptiQ: ~7 GB on disk).
  EXPECT_TRUE(charged > (std::size_t)4 << 30);
  // It decodes as ever.
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t ids[4] = {9707, 11, 1879, 0};
    EXPECT_TRUE(lm->prefill(ctx, std::span<const std::int32_t>(ids, 4)) >= 0);
  }
  lm.reset();
  // Gone: the next planning pass (or load) gives its bytes back.
  mgr->release_warm_unclaimed();
  mgr->clear_declarations();
  mgr->declare_weights("/nonexistent/another-model", 1 << 20);
  mgr->release_warm_unclaimed();
  mgr->clear_declarations();
  EXPECT_TRUE(mgr->wired_pool_used() == pool0);
}

// A language model KEPT WARM is REPORTED (warm_report,
// SessionIntf::memory_report): `kind` "language-model", its canonical
// directory, the bytes its load left wired, its K/V and the time it has
// left. And a report polled while the hold expires -- it reads K/V from a
// copy of the model, so it can be the one that lets it go -- leaves the
// pool where it was without a planning pass. (The test cannot force that
// interleaving; it checks the pool settles whichever thread frees the
// model.) Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, a_warm_language_model_is_reported) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  bag::set_flag(spec.extra, genai::load_spec::kWireWeights, true);
  const std::size_t pool0 = mgr->wired_pool_used();
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  const std::size_t charged = mgr->wired_pool_used() - pool0;
  EXPECT_TRUE(charged > (std::size_t)4 << 30);
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t ids[4] = {9707, 11, 1879, 0};
    EXPECT_TRUE(lm->prefill(ctx, std::span<const std::int32_t>(ids, 4)) >= 0);
  }
  const std::size_t kv = lm->kv_bytes();
  EXPECT_TRUE(kv > 0);

  mgr->keep_warm(lm, path, 60.0);
  auto r = mgr->warm_report();
  ASSERT_TRUE(r.size() == 1);
  if (r.size() != 1) { return; }
  EXPECT_TRUE(r[0].kind == "language-model");
  EXPECT_TRUE(r[0].dir == std::filesystem::canonical(path).string());
  EXPECT_TRUE(r[0].bytes == charged);
  EXPECT_TRUE(r[0].kv_bytes == kv);
  EXPECT_TRUE(r[0].seconds > 50.0 && r[0].seconds <= 60.0);
  std::printf("[warm_report] %s: %zu MB wired, %zu KB of K/V, %.1f s\n",
              r[0].dir.c_str(), r[0].bytes >> 20, r[0].kv_bytes >> 10,
              r[0].seconds);
  const vpipe::SessionIntf& host = sess;
  const std::string json = host.memory_report().to_json();
  EXPECT_TRUE(json.find("\"language-model\"") != std::string::npos);

  // A short hold (renewing only lengthens one, so it is dropped first),
  // the last user gone, a report polling as it expires.
  mgr->keep_warm(lm, path, 0);
  mgr->keep_warm(lm, path, 0.5);
  lm.reset();
  std::atomic<bool> stop{false};
  std::atomic<int> polls{0};
  std::thread poller([&] {
    while (!stop.load()) {
      (void)mgr->warm_report();
      polls.fetch_add(1);
    }
  });
  bool settled = false;
  for (int i = 0; i < 100 && !settled; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    settled = mgr->warm_count() == 0 && mgr->wired_pool_used() == pool0;
  }
  stop.store(true);
  poller.join();
  std::printf("[warm_report] %d reports while the hold expired\n",
              polls.load());
  EXPECT_TRUE(polls.load() > 0);
  EXPECT_TRUE(settled);
  EXPECT_TRUE(mgr->cached_count() == 0);
}

// A CONTEXT SAVED to disk and read back decodes as the one it was saved
// from, token for token (greedy, the same kernels either way); and a
// conversation PARKED with a warm model is taken back by its key, or
// saved to its file when the hold goes. Gated on
// VPIPE_QWEN35_TEST_MODEL_PATH (Qwen3.5: paged K/V and GDN state both).
TEST(metal_lm_smoke, a_context_saved_reads_back_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 16;
  spec.max_pages = 256;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  const auto prompt = lm->tokenizer().encode(
      "Once upon a time, in a village between two rivers, there lived an "
      "old clockmaker who kept every clock in the village a minute fast, "
      "so that nobody would ever be late. One morning");
  ASSERT_TRUE(prompt.size() > 32);   // more than two pages
  auto decode = [&](genai::LoadedLanguageModel::Context& c,
                    std::int32_t t) {
    std::vector<std::int32_t> out{t};
    for (int i = 0; i < 12; ++i) {
      t = lm->next_token_greedy(c, t);
      out.push_back(t);
    }
    return out;
  };
  // As decoded straight on.
  std::vector<std::int32_t> want;
  {
    auto c = lm->make_context();
    const std::int32_t t = lm->prefill(c, prompt);
    ASSERT_TRUE(t >= 0);
    want = decode(c, t);
  }
  // Saved after the prefill, read back, decoded.
  const auto dir = std::filesystem::temp_directory_path()
      / ("vpipe-ctx-" + std::to_string(std::chrono::steady_clock::now()
                                             .time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  const std::string file = (dir / "a.vctx").string();
  {
    auto c = lm->make_context();
    const std::int32_t t = lm->prefill(c, prompt);
    ASSERT_TRUE(t >= 0);
    std::string err;
    ASSERT_TRUE(lm->save_context(c, file, "meta-bytes", &err));
    if (!err.empty()) { std::printf("save: %s\n", err.c_str()); }
  }
  {
    std::string meta, err;
    auto back = lm->load_context(file, &meta, &err);
    if (!back.valid()) { std::printf("load: %s\n", err.c_str()); }
    ASSERT_TRUE(back.valid());
    if (!back.valid()) { return; }
    EXPECT_TRUE(meta == "meta-bytes");
    EXPECT_TRUE(back.seq_len() == (int)prompt.size());
    EXPECT_TRUE(back.last_predicted_id() == want[0]);
    const auto got = decode(back, want[0]);
    EXPECT_TRUE(got == want);
  }
  // A file cut short is refused, and makes nothing.
  {
    std::filesystem::resize_file(file, std::filesystem::file_size(file) / 2);
    std::string meta, err;
    auto back = lm->load_context(file, &meta, &err);
    EXPECT_TRUE(!back.valid());
    EXPECT_TRUE(!err.empty());
  }

  // Parked with the model kept warm: taken back by its key, as it was.
  mgr->keep_warm(lm, path, 60.0);
  {
    auto c = lm->make_context();
    ASSERT_TRUE(lm->prefill(c, prompt) == want[0]);
    const std::string spill = (dir / "kept.vctx").string();
    mgr->park_context("conv-a", lm, std::move(c), "held", spill);
    genai::LoadedLanguageModel::Context none;
    std::string meta;
    EXPECT_TRUE(!mgr->take_context("conv-b", lm, &none, &meta));
    genai::LoadedLanguageModel::Context taken;
    ASSERT_TRUE(mgr->take_context("conv-a", lm, &taken, &meta));
    EXPECT_TRUE(meta == "held");
    EXPECT_TRUE(decode(taken, want[0]) == want);
    // Parked again; the hold ends: saved to its file first.
    auto c2 = lm->make_context();
    ASSERT_TRUE(lm->prefill(c2, prompt) == want[0]);
    mgr->park_context("conv-a", lm, std::move(c2), "held", spill);
    EXPECT_TRUE(!std::filesystem::exists(spill));
    mgr->keep_warm(lm, path, 0);
    EXPECT_TRUE(std::filesystem::exists(spill));
    std::string m2, err;
    auto back = lm->load_context(spill, &m2, &err);
    ASSERT_TRUE(back.valid());
    if (back.valid()) { EXPECT_TRUE(decode(back, want[0]) == want); }
    // Dropped: nothing to take.
    mgr->keep_warm(lm, path, 60.0);
    auto c3 = lm->make_context();
    mgr->park_context("conv-c", lm, std::move(c3), "", "");
    EXPECT_TRUE(mgr->drop_context("conv-c"));
    EXPECT_TRUE(!mgr->drop_context("conv-c"));
    mgr->keep_warm(lm, path, 0);
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// The model manager saves and releases PARKED contexts on threads of its
// own -- the warm hold's reaper, a launch's planner, a host ending a
// conversation -- while another context of the same model decodes.
// Here a second thread saves and releases contexts, and drops parked
// ones, for the whole of a decode that also makes and frees contexts
// of its own: the decode's greedy tokens are those it makes alone, and
// every context saved reads back. Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, contexts_go_on_another_thread_while_it_decodes) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 16;
  spec.max_pages = 1024;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  using Ctx = genai::LoadedLanguageModel::Context;
  const auto prompt = lm->tokenizer().encode(
      "The lighthouse keeper wrote one line in the log every night, and "
      "on the last night of the year he wrote");
  const auto line = lm->tokenizer().encode("A short line, held.");
  constexpr int kTokens = 40;
  auto decode = [&](Ctx& c, std::int32_t t, const std::function<void()>&
                                                 between) {
    std::vector<std::int32_t> out{t};
    for (int i = 1; i < kTokens; ++i) {
      t = lm->next_token_greedy(c, t);
      out.push_back(t);
      between();
    }
    return out;
  };
  std::vector<std::int32_t> want;
  {
    Ctx c = lm->make_context();
    const std::int32_t t = lm->prefill(c, prompt);
    ASSERT_TRUE(t >= 0);
    want = decode(c, t, [] {});
  }

  // Theirs: contexts the other thread saves (every second one) and
  // releases; and contexts parked with the warm model under keys, which
  // it drops, then lets the hold go (saving those with a file).
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path()
      / ("vpipe-ctx-mt-" + std::to_string(::getpid()));
  fs::create_directories(dir);
  constexpr int kTheirs = 48, kParked = 16;
  std::vector<Ctx> theirs;
  for (int i = 0; i < kTheirs; ++i) {
    theirs.push_back(lm->make_context());
    ASSERT_TRUE(lm->prefill(theirs.back(), line) >= 0);
  }
  mgr->keep_warm(lm, path, 600.0);
  auto file = [&](const char* tag, int i) {
    return (dir / (std::string(tag) + std::to_string(i) + ".vctx"))
        .string();
  };
  for (int i = 0; i < kParked; ++i) {
    Ctx c = lm->make_context();
    ASSERT_TRUE(lm->prefill(c, line) >= 0);
    mgr->park_context("k" + std::to_string(i), lm, std::move(c), "m",
                      i % 2 ? file("p", i) : std::string());
  }

  std::atomic<bool> go{false}, done{false};
  std::atomic<int>  saved{0};
  std::thread other([&] {
    while (!go.load()) { std::this_thread::yield(); }
    for (int i = 0; i < kTheirs; ++i) {
      if (i % 2 && lm->save_context(theirs[(size_t)i], file("t", i), "")) {
        ++saved;
      }
      theirs[(size_t)i] = Ctx{};
      if (i % 4 == 0 && i / 4 < kParked / 2) {
        (void)mgr->drop_context("k" + std::to_string(2 * (i / 4)));
      }
      if (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    mgr->keep_warm(lm, path, 0);   // the hold ends: the rest saved
  });
  std::vector<std::int32_t> got;
  {
    Ctx c = lm->make_context();
    const std::int32_t t = lm->prefill(c, prompt);
    go.store(true);
    // A context of its own made and freed at every token: the map grows
    // under the other thread's erases.
    got = decode(c, t, [&] {
      Ctx extra = lm->make_context();
      (void)lm->prefill(extra, line);
    });
    done.store(true);
  }
  other.join();
  EXPECT_TRUE(got == want);
  EXPECT_TRUE(saved.load() == kTheirs / 2);
  int read_back = 0;
  for (int i = 1; i < kTheirs; i += 2) {
    std::string meta, err;
    Ctx back = lm->load_context(file("t", i), &meta, &err);
    if (back.valid() && back.seq_len() == (int)line.size()) { ++read_back; }
  }
  for (int i = 1; i < kParked; i += 2) {
    std::string meta, err;
    Ctx back = lm->load_context(file("p", i), &meta, &err);
    if (back.valid() && back.seq_len() == (int)line.size()
        && meta == "m") {
      ++read_back;
    }
  }
  std::printf("[contexts on another thread] decode %s; %d of %d saved "
              "contexts read back\n", got == want ? "exact" : "DIFFERS",
              read_back, kTheirs / 2 + kParked / 2);
  EXPECT_TRUE(read_back == kTheirs / 2 + kParked / 2);
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// A model kept warm that a launch does not claim STAYS when the launch
// fits beside it and GOES when it does not -- decided by the weights
// planner as a launch's claims reach it. A launch claiming the model
// keeps it whatever the box. Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, a_warm_model_stays_beside_a_launch_that_fits) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (!lm || !lm->valid()) { return; }
  mgr->keep_warm(lm, path, 600.0);
  lm.reset();
  ASSERT_TRUE(mgr->warm_count() == 1);
  ResourcePlanner* pl =
      ResourcePlannerRegistry::get().find(model_memory::kWeightsKind);
  ASSERT_TRUE(pl != nullptr);
  if (pl == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path other = fs::temp_directory_path()
      / ("vpipe-warm-fit-" + std::to_string(::getpid()));
  fs::create_directories(other);
  {
    std::ofstream f(other / "model.safetensors", std::ios::binary);
    const std::string blob(1 << 20, 'x');
    f.write(blob.data(), (std::streamsize)blob.size());
  }
  auto launch = [&](const char* ram_mb, const std::string& dir) {
    ::setenv("VPIPE_RAM_LIMIT_MB", ram_mb, 1);
    pl->begin_plan(&sess);
    pl->claim(&sess, dir, "", "", 0);
    pl->end_plan(&sess);
    ::unsetenv("VPIPE_RAM_LIMIT_MB");
    return mgr->warm_count();
  };
  // A box with room for both: kept.
  EXPECT_TRUE(launch("1048576", other.string()) == 1);
  // The launch claims the warm model itself: kept, however small the box.
  EXPECT_TRUE(launch("1024", path) == 1);
  // A box without room for the model beside the launch: released.
  EXPECT_TRUE(launch("1024", other.string()) == 0);
  mgr->clear_declarations();
  std::error_code ec;
  fs::remove_all(other, ec);
}

// A long prefill onto a BRANCHED context answers as the same tokens fed to
// an unbranched one. branch() freezes the parent's partial tail page, so
// the child's next page starts mid-sequence; the steel prefill attention
// (chunks from 2048 rows) took a key's page to be position / page_tokens
// and read the wrong keys -- 1 of 12 greedy tokens matched. A 1500-token
// chunk (the flash member, which always walked the table) is the control.
// Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, a_long_prefill_onto_a_branch_matches_unbranched) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (mgr == nullptr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 64;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm && lm->valid());
  if (!lm || !lm->valid()) { return; }
  std::string text;
  for (int i = 0; i < 400; ++i) {
    text += "Entry " + std::to_string(i) + ": the courier left station " +
            std::to_string((i * 37) % 101) + " carrying parcel " +
            std::to_string((i * 53) % 997) + ". ";
  }
  const auto all = lm->tokenizer().encode(text);
  ASSERT_TRUE(all.size() > 2400);
  if (all.size() <= 2400) { return; }
  // 100 tokens leave the parent's only page partial when it branches.
  const std::span<const std::int32_t> head(all.data(), 100);
  auto greedy = [&](genai::LoadedLanguageModel::Context& c,
                    std::span<const std::int32_t> tail) {
    std::vector<std::int32_t> out;
    std::int32_t t = lm->prefill(c, tail);
    for (int i = 0; i < 12 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(c);
    }
    return out;
  };
  for (const int n : {1500, 2200}) {
    const std::span<const std::int32_t> tail(all.data() + 100, (size_t)n);
    auto plain = lm->make_context();
    ASSERT_TRUE(lm->prefill(plain, head) >= 0);
    const auto want = greedy(plain, tail);
    auto parent = lm->make_context();
    ASSERT_TRUE(lm->prefill(parent, head) >= 0);
    auto child = lm->branch(parent);
    ASSERT_TRUE(child.valid());
    const auto got = greedy(child, tail);
    int same = 0;
    for (size_t i = 0; i < want.size() && i < got.size(); ++i) {
      same += want[i] == got[i] ? 1 : 0;
    }
    std::printf("[branch prefill] %d-token chunk onto a branch: %d of %zu "
                "greedy tokens as unbranched\n", n, same, want.size());
    EXPECT_TRUE(want.size() == 12 && got == want);
  }
  ::unsetenv("VPIPE_LLM_BACKEND");
}
