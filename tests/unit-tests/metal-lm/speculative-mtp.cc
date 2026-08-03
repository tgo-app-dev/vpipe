// speculative-mtp.cc -- Multi-token-prediction (MTP) speculative decode: the
// draft-head paths on OptiQ / dense / GGUF checkpoints, token-exactness vs
// plain decode, the real-text accept-rate probe, Leviathan sampling and the
// stream / stop token behaviour.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// MTP speculative decode: the bundled mtp.safetensors head drafts tokens, the
// main model verifies them, the longest greedy-matching prefix is accepted,
// and the rejected speculative tail is rolled back (paged KV via kv_rollback +
// GDN recurrent ring via gdn_ring_rollback -- the depth>1 pdecode machinery).
// GREEDY spec decode MUST be token-exact vs a serial forward_argmax loop
// (verification makes the drafter affect only speed, never the tokens). Also
// reports the mean accepted tokens/round (the acceptance the drafter buys).
// Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_optiq_mtp_speculative_token_exact) {
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
  // Gen length env-tunable (VPIPE_MTP_GEN_TOKENS) to measure at the long
  // contexts where MTP's win erodes; size the page pool for the root + the two
  // depth children all reaching ~kGen, + margin.
  int kGen = 96;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  mcfg.max_pages = std::max(8, (3 * kGen) / 512 + 8);
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());   // the mtp.safetensors head loaded

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

  // Prefill on the root context (also stashes the last hidden the MTP drafter
  // consumes). Branch a child off the prefilled prefix BEFORE decoding so the
  // MTP run starts from the exact same state as the serial reference.
  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  // One child branch per MTP depth, both off the prefilled prefix.
  const genai::ContextId child1 =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child1.valid() && child2.valid());

  // Serial greedy reference on the root context (kGen set above; the speed
  // baseline).
  std::vector<std::int32_t> ref;
  ref.push_back(first);
  const auto s0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  const auto s1 = std::chrono::steady_clock::now();
  const double serial_ms =
      std::chrono::duration<double, std::milli>(s1 - s0).count();
  const double serial_tps = (double)(ref.size() - 1) / (serial_ms / 1000.0);

  // MTP speculative decode at depth-1 and depth-2 (token-exact, back-to-back
  // so the thermal state is comparable). draft_len 1 = depth-1, >=2 = depth-2.
  auto run_mtp = [&](genai::ContextId cid, int draft_len, const char* tag) {
    std::vector<std::int32_t> got;
    long accepted = 0, rounds = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = model->mtp_decode(cid, first, (int)ref.size(), got,
                                      draft_len, &accepted, &rounds);
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(ok);
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int mism = 0;
    const std::size_t nn = std::min(ref.size(), got.size());
    for (std::size_t i = 0; i < nn; ++i) {
      if (ref[i] != got[i]) { ++mism; }
    }
    std::printf("[qwen_optiq_mtp] %s: %zu tok in %.0f ms (%.1f tok/s) | "
                "speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
                tag, got.size(), ms, (double)got.size() / (ms / 1000.0),
                ((double)got.size() / (ms / 1000.0)) / serial_tps, rounds,
                rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
    EXPECT_TRUE(mism == 0);
    EXPECT_TRUE(got.size() + 1 >= ref.size());
  };
  std::printf("[qwen_optiq_mtp] serial baseline %.1f tok/s\n", serial_tps);
  run_mtp(child1, /*draft_len=*/1, "depth-1");
  run_mtp(child2, /*draft_len=*/2, "depth-2");
}

// Dense (raw-HF bf16/f16) MTP token-exact verification, SINGLE model load.
// The optiq MTP test above double-loads (direct + via the model manager for a
// chat template), which on a 64 GB box exceeds RAM for the 54 GB bf16 27B. This
// loads the dense model ONCE, tokenizes the prompt from tokenizer.json, and
// checks mtp_decode (depth-1 + depth-2) is token-exact vs plain serial decode.
// Env: VPIPE_DENSE_MTP_MODEL = the raw-HF dense model dir.
TEST(metal_lm_smoke, dense_mtp_speculative_token_exact) {
  const char* path = std::getenv("VPIPE_DENSE_MTP_MODEL");
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
  int kGen = 32;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  mcfg.max_pages = std::max(8, (3 * kGen) / 512 + 8);
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());   // the raw-HF dense MTP head loaded

  // Tokenize the prompt directly from tokenizer.json (no second model load).
  const std::string tk = std::string(path) + "/tokenizer.json";
  auto tok = genai::Tokenizer::from_huggingface_json(tk, &sess);
  ASSERT_TRUE(tok != nullptr);
  std::vector<std::int32_t> ids =
      tok->encode("The capital of France is");
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
  const genai::ContextId child1 =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child1.valid() && child2.valid());

  // Serial greedy reference on the root context.
  std::vector<std::int32_t> ref;
  ref.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  std::printf("[dense_mtp] serial ref %zu tok, first=%d\n", ref.size(), first);

  auto run_mtp = [&](genai::ContextId cid, int draft_len, const char* tag) {
    std::vector<std::int32_t> got;
    long accepted = 0, rounds = 0;
    const bool ok = model->mtp_decode(cid, first, (int)ref.size(), got,
                                      draft_len, &accepted, &rounds);
    EXPECT_TRUE(ok);
    int mism = 0;
    const std::size_t nn = std::min(ref.size(), got.size());
    for (std::size_t i = 0; i < nn; ++i) {
      if (ref[i] != got[i]) { ++mism; }
    }
    std::printf("[dense_mtp] %s: %zu tok rounds=%ld tok/round=%.2f mism=%d\n",
                tag, got.size(), rounds,
                rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
    EXPECT_TRUE(mism == 0);
    EXPECT_TRUE(got.size() + 1 >= ref.size());
  };
  run_mtp(child1, /*draft_len=*/1, "depth-1");
  run_mtp(child2, /*draft_len=*/2, "depth-2");
}

// MTP acceptance/speedup on LONG, COHERENT text -- the regime the "<1.0x at
// long context" complaint is really about. The token-exact test above decodes
// 1000+ tokens out of a trivial prompt, which DEGENERATES (the model's own
// rambling becomes high-entropy and the drafter's acceptance collapses ~ for
// ANY drafter). Here we prefill a long, information-dense passage and decode a
// GROUNDED continuation: the model stays coherent/confident, so acceptance
// reflects real long-context usage (RAG / documents / code), not degeneracy.
TEST(metal_lm_smoke, qwen_optiq_mtp_realtext_accept) {
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
  int kGen = 256;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  // Long enough page pool for the long prefix + root + two depth children.
  mcfg.max_pages = std::max(16, (4 * (2048 + kGen)) / 512 + 8);
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());
  // Prefix-seed is opt-in (off by default); enable it so this bench/teacher-
  // force exercises the seeded ("full history") drafter. VPIPE_MTP_NO_SEED
  // still hard-disables it for the decode-only A/B.
  model->set_mtp_prefix_seed(true);

  // A long, coherent, information-dense passage (no repetition). The decode
  // continues it in the same encyclopedic register -> grounded, low-entropy.
  const char* passage =
      "The deep ocean, comprising all marine waters below roughly one thousand "
      "meters, is the largest continuous habitat on Earth and also the least "
      "explored. Sunlight is fully extinguished within the first few hundred "
      "meters, so the vast volume beneath -- the aphotic zone -- exists in "
      "perpetual darkness. With increasing depth the water grows colder and the "
      "pressure climbs by about one atmosphere for every ten meters of descent, "
      "reaching more than a thousand atmospheres in the deepest trenches. These "
      "conditions long convinced naturalists that the abyss must be lifeless, a "
      "barren desert of cold and crushing weight. The reality proved far "
      "stranger. Life is distributed throughout the water column and across the "
      "seafloor, sustained by a thin, ceaseless rain of organic particles "
      "descending from the sunlit surface, a flux that oceanographers call "
      "marine snow. This detritus -- dead plankton, fecal pellets, and "
      "aggregated mucus -- is the primary food supply for most deep dwellers, "
      "and its slow settling couples the productive surface to the dark interior "
      "over timescales of weeks. A second, wholly independent foundation for "
      "life was discovered in 1977 near the Galapagos Rift, where submersibles "
      "found hot springs venting mineral-rich fluid through the seafloor. "
      "Around these hydrothermal vents thrive dense communities of tube worms, "
      "clams, and shrimp that depend not on sunlight but on chemosynthesis: "
      "specialized bacteria oxidize hydrogen sulfide and methane from the vent "
      "fluid to fix carbon, forming the base of a food web powered by the "
      "planet's internal heat rather than the sun. Many deep-sea animals "
      "generate their own light through bioluminescence, a chemical reaction "
      "between a substrate called luciferin and an enzyme called luciferase. "
      "Light is used to lure prey, to startle predators, to find mates in the "
      "dark, and as counter-illumination camouflage that erases an animal's "
      "silhouette against the faint glow filtering down from above. The "
      "anglerfish dangles a luminous lure before its jaws; the cookie-cutter "
      "shark glows except for a dark collar that mimics a small fish to draw in "
      "larger hunters. Bodies are adapted to scarcity and pressure: metabolism "
      "is slowed, skeletons and muscles are reduced, and proteins are stabilized "
      "by molecules that counteract the deforming effect of extreme pressure. "
      "Growth is unhurried and lifespans are often long; some deep corals live "
      "for thousands of years, recording ocean chemistry in their skeletons like "
      "tree rings. The deep sea also governs the planet's climate over the long "
      "term. Cold, dense water sinking at high latitudes drives the global "
      "overturning circulation that redistributes heat, and the biological pump "
      "that carries carbon downward as marine snow sequesters it in deep waters "
      "and sediments for centuries. Exploration remains difficult and expensive. "
      "Crewed submersibles, remotely operated vehicles, and autonomous gliders "
      "have mapped only a small fraction of the seafloor in detail, and new "
      "species are described on nearly every expedition.";

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
  std::string ask = std::string(passage) +
      "\n\nContinue this encyclopedic article for several more paragraphs in "
      "the same factual, expository style.";
  // Prompt override (VPIPE_MTP_PROMPT_FILE): swap the whole user turn for a
  // prompt read from a file, so acceptance can be A/B'd across prompt domains
  // (e.g. our prose vs MTPLX's coding prompt) on the same head.
  if (const char* pf = std::getenv("VPIPE_MTP_PROMPT_FILE")) {
    std::ifstream pin(pf);
    if (pin) {
      std::stringstream ss;
      ss << pin.rdbuf();
      ask = ss.str();
      std::printf("[mtp_realtext] prompt override from %s (%zu chars)\n",
                  pf, ask.size());
    }
  }
  tpl->render_user_turn(ask, /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  // VPIPE_MTP_NO_THINK: the VL template opens a thinking block (`<think>\n`);
  // close it immediately (`\n</think>\n\n`) so the model answers directly,
  // matching a thinking-OFF runtime (e.g. MTPLX) for a fair acceptance A/B.
  if (std::getenv("VPIPE_MTP_NO_THINK")) {
    const auto& tk = lm->tokenizer();
    auto nl = tk.encode("\n");
    ids.insert(ids.end(), nl.begin(), nl.end());
    const std::int32_t tc = tk.special_token_id("</think>");
    if (tc >= 0) { ids.push_back(tc); }
    auto nl2 = tk.encode("\n\n");
    ids.insert(ids.end(), nl2.begin(), nl2.end());
    std::printf("[mtp_realtext] thinking DISABLED (closed think block, "
                "</think>=%d)\n", tc);
  }

  // VPIPE_MTP_PREFIX_IDS: replace the rendered prefix with an explicit
  // comma-separated token-id list, so an EXACT prompt (e.g. another runtime's
  // rendered tokens) can be fed verbatim to isolate kernel/fp divergence from
  // chat-template divergence.
  if (const char* pe = std::getenv("VPIPE_MTP_PREFIX_IDS")) {
    std::vector<std::int32_t> ov;
    const char* p = pe;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { ++p; continue; }
      ov.push_back((std::int32_t)v);
      p = end;
    }
    if (!ov.empty()) {
      ids = ov;
      std::printf("[mtp_realtext] prefix OVERRIDE from VPIPE_MTP_PREFIX_IDS "
                  "(%zu tok)\n", ids.size());
    }
  }

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
  std::printf("[mtp_realtext] prefix=%zu tok, decode at ctx %zu..%zu\n",
              ids.size(), ids.size(), ids.size() + (std::size_t)kGen);
  // VPIPE_MTP_DUMP_IDS: print the prefix + greedy-continuation token ids as
  // comma-separated lists for cross-runtime divergence analysis.
  if (std::getenv("VPIPE_MTP_DUMP_IDS")) {
    std::printf("[mtp_realtext] prefix_ids=");
    for (std::size_t i = 0; i < ids.size(); ++i) {
      std::printf("%s%d", i ? "," : "", ids[i]);
    }
    std::printf("\n[mtp_realtext] first_pred=%d\n", first);
  }
  // VPIPE_MTP_TF: teacher-forced MTP draft-accuracy on a FIXED continuation
  // (comma-separated ids in VPIPE_MTP_CONT_FILE) walked verbatim through the
  // model from this prefill -- measures depth-1 draft quality WITHOUT the
  // free-run stream-divergence confound (the head is conditioned on the TRUE
  // next token). VPIPE_MTP_TF_CHUNK sets the per-verify width (default 2, the
  // depth-1 decode's MTP attention window).
  if (std::getenv("VPIPE_MTP_TF")) {
    const char* cf = std::getenv("VPIPE_MTP_CONT_FILE");
    std::vector<std::int32_t> cont;
    if (cf) {
      std::ifstream cin2(cf);
      std::string s;
      std::getline(cin2, s, '\0');
      const char* p = s.c_str();
      while (*p) {
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        cont.push_back((std::int32_t)v);
        p = e;
      }
    }
    ASSERT_TRUE(!cont.empty());
    int chunk = 2;
    if (const char* ce = std::getenv("VPIPE_MTP_TF_CHUNK")) {
      const int v = std::atoi(ce);
      if (v >= 1) { chunk = v; }
    }
    const genai::ContextId tfc =
        model->context_manager()->branch(model->root_context());
    long hits = 0, tot = 0;
    const bool ok = model->mtp_teacher_force(tfc, cont, chunk, &hits, &tot);
    EXPECT_TRUE(ok);
    std::printf("[mtp_realtext] teacher-force cont=%zu tok\n", cont.size());
    return;
  }

  const genai::ContextId child1 =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child1.valid() && child2.valid());

  std::vector<std::int32_t> ref;
  ref.push_back(first);
  const auto s0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  const auto s1 = std::chrono::steady_clock::now();
  const double serial_ms =
      std::chrono::duration<double, std::milli>(s1 - s0).count();
  const double serial_tps = (double)(ref.size() - 1) / (serial_ms / 1000.0);
  std::printf("[mtp_realtext] serial baseline %.1f tok/s\n", serial_tps);
  {
    const std::string gen = lm->tokenizer().decode(ref);
    std::printf("[mtp_realtext] gen: %.280s\n", gen.c_str());
  }
  if (std::getenv("VPIPE_MTP_DUMP_IDS")) {
    std::printf("[mtp_realtext] gen_ids=");
    for (std::size_t i = 0; i < ref.size(); ++i) {
      std::printf("%s%d", i ? "," : "", ref[i]);
    }
    std::printf("\n");
  }

  // Optional sampler override (acceptance A/B vs an external runtime in the
  // SAME mode): VPIPE_MTP_TEMP>0 switches to sampling at that temperature with
  // VPIPE_MTP_TOP_P / VPIPE_MTP_TOP_K; VPIPE_MTP_LEVIATHAN=1 uses L-C accept.
  // Default (unset) stays greedy/exact-match.
  auto run_mtp = [&](genai::ContextId cid, int draft_len, const char* tag) {
    std::vector<std::int32_t> got;
    long accepted = 0, rounds = 0;
    genai::MtpDecodeCtl ctl;
    bool sampling = false;
    if (const char* te = std::getenv("VPIPE_MTP_TEMP")) {
      const float t = (float)std::atof(te);
      if (t > 0.0f) {
        sampling = true;
        ctl.sampler.greedy = false;
        ctl.sampler.temperature = t;
        const char* tp = std::getenv("VPIPE_MTP_TOP_P");
        const char* tk = std::getenv("VPIPE_MTP_TOP_K");
        ctl.sampler.top_p = tp ? (float)std::atof(tp) : 1.0f;
        ctl.sampler.top_k = tk ? std::atoi(tk) : 0;
        ctl.sampler.seed = 777ull;
        ctl.sampler.n_iter = 16;
        const char* lv = std::getenv("VPIPE_MTP_LEVIATHAN");
        model->set_leviathan(lv && std::atoi(lv) != 0);
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = sampling
        ? model->mtp_decode(cid, first, (int)ref.size(), got, draft_len,
                            &accepted, &rounds, ctl)
        : model->mtp_decode(cid, first, (int)ref.size(), got, draft_len,
                            &accepted, &rounds);
    model->set_leviathan(false);
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(ok);
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    int mism = 0;
    const std::size_t nn = std::min(ref.size(), got.size());
    for (std::size_t i = 0; i < nn; ++i) {
      if (ref[i] != got[i]) { ++mism; }
    }
    std::printf("[mtp_realtext] %s: %zu tok in %.0f ms (%.1f tok/s) | "
                "speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
                tag, got.size(), ms, (double)got.size() / (ms / 1000.0),
                ((double)got.size() / (ms / 1000.0)) / serial_tps, rounds,
                rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
    // This is a perf/acceptance benchmark, NOT a token-exact check (the
    // controlled-length *_token_exact tests cover that). Over long greedy
    // generation MTP can diverge from the serial reference at a near-tie argmax
    // (a tiny fp difference between the batched-verify and single-decode paths
    // flips the top token, then the sequences cascade apart) -- expected, and
    // independent of the MAXM=4 verify (depth-1 never takes it yet diverges
    // identically). So report mism; don't fail on it.
  };
  run_mtp(child1, /*draft_len=*/1, "depth-1");
  run_mtp(child2, /*draft_len=*/2, "depth-2");
}

// Leviathan-Chen MTP sampling: (A) at temperature it accepts MORE than the
// default exact-match scheme (the whole point -- ours accepts a draft d with
// prob p(d); L-C accepts min(1,p(d)/q(d)) >= p(d)); (B) as temp->0 it must
// converge to the greedy result (a gross-corruption guard on the ratio +
// residual math). Depth-1, pure temperature.
TEST(metal_lm_smoke, qwen_optiq_mtp_leviathan_sampling) {
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
  mcfg.max_pages = 64;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());

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
      "Write a short paragraph explaining why the sky looks blue during the "
      "day and turns red at sunset.", /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t b = 0; float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; b = (std::int32_t)i; }
    }
    return b;
  };
  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  auto* cmgr = model->context_manager();
  const genai::ContextId root = model->root_context();

  const std::uint64_t seed = 777ull;
  // Run mtp_decode and return acceptance (tok/round) for a sampler config.
  auto run = [&](bool lc, float temp, float top_p, int top_k, int draft_len,
                 int n, bool gpu = false) -> double {
    model->set_leviathan(lc);
    model->set_lc_gpu(gpu);
    genai::MtpDecodeCtl ctl;
    ctl.sampler.greedy = false; ctl.sampler.temperature = temp;
    ctl.sampler.top_p = top_p; ctl.sampler.top_k = top_k;
    ctl.sampler.seed = seed; ctl.sampler.n_iter = 16;
    genai::ContextId c = cmgr->branch(root);
    std::vector<std::int32_t> got; long acc = 0, rnds = 0;
    EXPECT_TRUE(model->mtp_decode(c, first, n, got, draft_len, &acc, &rnds,
                                  ctl));
    model->set_leviathan(false);
    model->set_lc_gpu(false);
    return rnds > 0 ? (double)got.size() / (double)rnds : 0.0;
  };
  // Reduce-to-greedy guard: at temp->0 L-C samples ~argmax, so its output must
  // track the (exact-match) GREEDY mtp_decode of the same depth until a rare
  // near-tie deviation cascades. Validates the ratio + residual + nucleus math
  // doesn't corrupt the distribution. Returns the first divergence index.
  auto reduce = [&](float top_p, int top_k, int draft_len,
                    const char* tag, bool gpu = false) -> std::size_t {
    model->set_leviathan(false);
    model->set_lc_gpu(false);
    std::vector<std::int32_t> gref;
    { genai::MtpDecodeCtl gc; genai::ContextId c = cmgr->branch(root);
      long a = 0, r = 0;
      EXPECT_TRUE(model->mtp_decode(c, first, 48, gref, draft_len, &a, &r, gc));
    }
    model->set_leviathan(true);
    model->set_lc_gpu(gpu);
    genai::MtpDecodeCtl ctl;
    ctl.sampler.greedy = false; ctl.sampler.temperature = 0.005f;
    ctl.sampler.top_p = top_p; ctl.sampler.top_k = top_k;
    ctl.sampler.seed = 42ull; ctl.sampler.n_iter = 16;
    genai::ContextId cb = cmgr->branch(root);
    std::vector<std::int32_t> glc; long a2 = 0, r2 = 0;
    EXPECT_TRUE(model->mtp_decode(cb, first, 48, glc, draft_len, &a2, &r2, ctl));
    model->set_leviathan(false);
    model->set_lc_gpu(false);
    const std::size_t nn = std::min(gref.size(), glc.size());
    std::size_t fd = nn;
    for (std::size_t i = 0; i < nn; ++i) {
      if (gref[i] != glc[i]) { fd = i; break; }
    }
    std::printf("[mtp_leviathan] reduce-to-greedy %s: first divergence "
                "@%zu/%zu\n", tag, fd, nn);
    return fd;
  };

  // ---- PART A: depth-1 acceptance benefit grows with temperature. ----
  double tpr_em_hi = 0.0, tpr_lc_hi = 0.0;
  for (float temp : {0.3f, 0.5f, 0.7f, 1.0f, 1.5f}) {
    const double tpr_em = run(false, temp, 1.0f, 0, 1, 200);
    const double tpr_lc = run(true,  temp, 1.0f, 0, 1, 200);
    std::printf("[mtp_leviathan] depth-1 temp=%.2f | exact-match tok/round=%.2f "
                "| leviathan tok/round=%.2f\n", temp, tpr_em, tpr_lc);
    if (temp > 1.4f) { tpr_em_hi = tpr_em; tpr_lc_hi = tpr_lc; }
  }
  EXPECT_TRUE(tpr_lc_hi > tpr_em_hi + 0.1);   // clear high-temp win

  // ---- PART B: depth-2 L-C acceptance at high temp (chained ratio test). ----
  const double d2_em = run(false, 1.5f, 1.0f, 0, 2, 200);
  const double d2_lc = run(true,  1.5f, 1.0f, 0, 2, 200);
  std::printf("[mtp_leviathan] depth-2 temp=1.50 | exact-match tok/round=%.2f "
              "| leviathan tok/round=%.2f\n", d2_em, d2_lc);
  EXPECT_TRUE(d2_lc > d2_em + 0.1);

  // ---- PART C: top_p nucleus acceptance (temperature + nucleus filter). ----
  const double tp_em = run(false, 1.5f, 0.9f, 0, 1, 200);
  const double tp_lc = run(true,  1.5f, 0.9f, 0, 1, 200);
  std::printf("[mtp_leviathan] depth-1 temp=1.50 top_p=0.90 | exact-match "
              "tok/round=%.2f | leviathan tok/round=%.2f\n", tp_em, tp_lc);

  // ---- PART D: reduce-to-greedy across configs (correctness guard). ----
  EXPECT_TRUE(reduce(1.0f, 0,  1, "depth-1 pure-temp") >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  1, "depth-1 top_p=0.9") >= 4);
  EXPECT_TRUE(reduce(1.0f, 40, 1, "depth-1 top_k=40") >= 4);
  EXPECT_TRUE(reduce(1.0f, 0,  2, "depth-2 pure-temp") >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  2, "depth-2 top_p=0.9") >= 4);

  // ---- PART E: on-GPU L-C (lc_sample_f16 / lc_accept_f16). Same nucleus +
  // accept/residual as the host path, done one-threadgroup-per-row on the GPU.
  // (1) reduce-to-greedy: at temp->0 the GPU L-C must track exact-match greedy
  // exactly as the host path does -- the strong correctness anchor that the
  // GPU nucleus + accept + residual + Gumbel-max compose losslessly.
  EXPECT_TRUE(reduce(1.0f, 0,  1, "gpu depth-1 pure-temp", true) >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  1, "gpu depth-1 top_p=0.9", true) >= 4);
  EXPECT_TRUE(reduce(1.0f, 40, 1, "gpu depth-1 top_k=40", true) >= 4);
  EXPECT_TRUE(reduce(1.0f, 0,  2, "gpu depth-2 pure-temp", true) >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  2, "gpu depth-2 top_p=0.9", true) >= 4);
  // (2) acceptance parity host vs GPU (different RNG sub-streams -> the same
  // distribution -> statistically-consistent tok/round, not a drift/bug).
  for (float temp : {0.7f, 1.5f}) {
    const double host = run(true, temp, 0.9f, 0, 1, 200, /*gpu=*/false);
    const double gpu  = run(true, temp, 0.9f, 0, 1, 200, /*gpu=*/true);
    std::printf("[mtp_leviathan] gpu-parity temp=%.2f top_p=0.90 | host "
                "tok/round=%.2f | gpu tok/round=%.2f\n", temp, host, gpu);
    EXPECT_TRUE(std::fabs(host - gpu) < 0.35);
  }
  model->set_leviathan(false);
  model->set_lc_gpu(false);
}

// MTP speculative decode on a NATIVE k-quant GGUF NextN checkpoint (the MTP
// draft block is bundled in the .gguf as blk.{n}.nextn.* + a full attn/ffn
// block). The k-quant verify (kqmm_ dequant+dense matmuls, fused q|k|v, q6_K
// lm_head) MUST be greedy token-exact vs a serial forward_argmax loop -- the
// drafter affects only speed, never the tokens. Reports accepted tokens/round.
// Gated on VPIPE_QWEN_GGUF_MTP_TEST_MODEL_PATH (.gguf) + VPIPE_QWEN35_TEST_
// MODEL_PATH (tokenizer/chat-template dir).
TEST(metal_lm_smoke, qwen_gguf_mtp_speculative_token_exact) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_MTP_TEST_MODEL_PATH");
  const char* tok_dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!gguf || !*gguf || !tok_dir || !*tok_dir) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  ASSERT_TRUE(cfg.has_value());
  // The MTP block is excluded from the main layer count (block_count - nextn).
  EXPECT_TRUE(cfg->num_nextn_layers >= 1);
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Generation length is env-tunable (VPIPE_MTP_GEN_TOKENS) so the speedup can
  // be measured at the longer contexts where chat actually runs (MTP's win
  // erodes as the verify's attention scans a growing KV). Size the KV page pool
  // for the serial ref (root) AND the MTP child both reaching ~kGen, + margin.
  int kGen = 96;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  mcfg.max_pages = std::max(8, (2 * kGen) / 512 + 8);
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());   // the bundled NextN block loaded (k-quant)

  // Tokenizer + chat template from the sibling safetensors Qwen3.5 dir.
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = tok_dir;
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
  // Exact-prefix override (e.g. another runtime's rendered ids) for a long-
  // prefix seed A/B on the GGUF k-quant MTP head.
  if (const char* pe = std::getenv("VPIPE_MTP_PREFIX_IDS")) {
    std::vector<std::int32_t> ov;
    const char* p = pe;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { ++p; continue; }
      ov.push_back((std::int32_t)v);
      p = end;
    }
    if (!ov.empty()) {
      ids = ov;
      std::printf("[qwen_gguf_mtp] prefix OVERRIDE (%zu tok)\n", ids.size());
    }
  }
  // Prefix-seed the MTP drafter (opt-in; VPIPE_MTP_NO_SEED disables) so the
  // GGUF k-quant seed path is exercised.
  model->set_mtp_prefix_seed(std::getenv("VPIPE_MTP_NO_SEED") == nullptr);

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  // Prefill (stashes the last hidden the MTP drafter consumes), then branch a
  // child off the prefilled prefix so MTP starts from the serial ref's state.
  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  // VPIPE_MTP_DUMP_IDS: print this model's own greedy continuation (first +
  // forward_argmax*N) so the TF run below walks an on-distribution sequence.
  if (std::getenv("VPIPE_MTP_DUMP_IDS")) {
    int n = 256;
    if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
      const int v = std::atoi(e);
      if (v > 0) { n = v; }
    }
    std::printf("[qwen_gguf_mtp] cont_ids=%d", first);
    std::int32_t t = first;
    for (int i = 0; i < n; ++i) {
      t = model->forward_argmax(t);
      if (t < 0) { break; }
      std::printf(",%d", t);
    }
    std::printf("\n");
    return;
  }
  // VPIPE_MTP_TF: teacher-forced depth-1 draft accuracy on a fixed continuation
  // (VPIPE_MTP_CONT_FILE) -- the seed A/B for the GGUF k-quant MTP head, exactly
  // as the OptiQ realtext test. Returns before the spec-decode timing below.
  if (std::getenv("VPIPE_MTP_TF")) {
    const char* cf = std::getenv("VPIPE_MTP_CONT_FILE");
    std::vector<std::int32_t> cont;
    if (cf) {
      std::ifstream cin2(cf);
      std::string s;
      std::getline(cin2, s, '\0');
      const char* p = s.c_str();
      while (*p) {
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        cont.push_back((std::int32_t)v);
        p = e;
      }
    }
    ASSERT_TRUE(!cont.empty());
    int chunk = 2;
    if (const char* ce = std::getenv("VPIPE_MTP_TF_CHUNK")) {
      const int v = std::atoi(ce);
      if (v >= 1) { chunk = v; }
    }
    const genai::ContextId tfc =
        model->context_manager()->branch(model->root_context());
    long hits = 0, tot = 0;
    EXPECT_TRUE(model->mtp_teacher_force(tfc, cont, chunk, &hits, &tot));
    return;
  }
  // One child branch per MTP depth, BOTH off the clean prefilled prefix (before
  // the serial ref loop advances the root) so each depth starts from the exact
  // same state as the reference.
  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid() && child2.valid());

  // Serial greedy reference on the root context (kGen set above).
  std::vector<std::int32_t> ref;
  ref.push_back(first);
  const auto sref0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  const auto sref1 = std::chrono::steady_clock::now();
  const double serial_ms =
      std::chrono::duration<double, std::milli>(sref1 - sref0).count();
  const double serial_tps = (double)(ref.size() - 1) / (serial_ms / 1000.0);

  // MTP depth-1 speculative decode on the child branch: token-exact + faster.
  std::vector<std::int32_t> got;
  long accepted = 0, rounds = 0;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = model->mtp_decode(child, first, (int)ref.size(), got,
                                    /*draft_len=*/1, &accepted, &rounds);
  const auto t1 = std::chrono::steady_clock::now();
  EXPECT_TRUE(ok);
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  int mism = 0;
  const std::size_t nn = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < nn; ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  const double mtp_tps = (double)got.size() / (ms / 1000.0);
  std::printf("[qwen_gguf_mtp] %zu tok in %.0f ms (%.1f tok/s) | serial %.1f "
              "tok/s | speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
              got.size(), ms, mtp_tps, serial_tps, mtp_tps / serial_tps, rounds,
              rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(got.size() + 1 >= ref.size());
  EXPECT_TRUE(rounds > 0 && (long)got.size() > rounds);   // some drafts landed

  // Depth-2 (draft_len=2 -> 3 drafts/round, M=3): exercises the k-quant MAXM=4
  // verify-GEMV twins (qmv_q*k_batch4) -- one weight read for the 3-row tile vs
  // the MAXM=2 form's 2 grid.z tiles. Must stay token-exact vs the same serial
  // ref. A/B the MAXM=2-tiled path with VPIPE_MTP_QMV4=0.
  {
    std::vector<std::int32_t> got2;
    long acc2 = 0, rnd2 = 0;
    const auto d0 = std::chrono::steady_clock::now();
    const bool ok2 = model->mtp_decode(child2, first, (int)ref.size(), got2,
                                       /*draft_len=*/2, &acc2, &rnd2);
    const double ms2 =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - d0).count();
    EXPECT_TRUE(ok2);
    int mism2 = 0;
    const std::size_t nn2 = std::min(ref.size(), got2.size());
    for (std::size_t i = 0; i < nn2; ++i) {
      if (ref[i] != got2[i]) { ++mism2; }
    }
    std::printf("[qwen_gguf_mtp] depth-2: %zu tok in %.0f ms (%.1f tok/s) | "
                "speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
                got2.size(), ms2, (double)got2.size() / (ms2 / 1000.0),
                ((double)got2.size() / (ms2 / 1000.0)) / serial_tps, rnd2,
                rnd2 > 0 ? (double)got2.size() / (double)rnd2 : 0.0, mism2);
    EXPECT_TRUE(mism2 == 0);
    EXPECT_TRUE(rnd2 > 0 && (long)got2.size() > rnd2);
  }
}

// MTP speculative SAMPLING (non-greedy): mtp_decode with a sampling verify MUST
// reproduce decode_pipelined (the serial GPU-sampled path) token-for-token --
// SAME temperature/top_p + the SAME per-slot seed (the verify seeds position k
// by its absolute KV slot, byte-identical to decode_pipelined's per-step seed)
// => identical samples; the MTP drafter only changes how many land per round,
// not the tokens. This is the "verification, not prediction" property under
// sampling. Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_optiq_mtp_sampled_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_SAMPLE_ITERS");   // decode_pipelined then uses 16; match below
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
  ASSERT_TRUE(model->has_mtp());

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
  tpl->render_user_turn("Write a short story about a curious robot.",
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
  const std::int32_t first = argmax(lg);   // shared seed token for both paths

  const int kGen = 80;
  const float temp = 0.8f, top_p = 0.95f;
  const std::uint64_t seed = 1234567ull;

  // Two branches off the same prefilled prefix so both paths start identically.
  const genai::ContextId refc =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId mtpc =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(refc.valid() && mtpc.valid());

  // Reference: serial GPU-sampled decode (per-step seed seed+0x9e3779b9*(s+1)).
  std::vector<std::int32_t> ref;
  ASSERT_TRUE(model->decode_pipelined(refc, first, kGen, ref, temp, top_p, seed));

  // MTP speculative sampling, SAME temp/top_p/seed + n_iter=16 (decode_pipelined
  // default). The verify samples each position; the drafter is unchanged.
  genai::MtpDecodeCtl ctl;
  ctl.sampler.greedy      = false;
  ctl.sampler.temperature = temp;
  ctl.sampler.top_p       = top_p;
  ctl.sampler.seed        = seed;
  ctl.sampler.n_iter      = 16;
  std::vector<std::int32_t> got;
  long accepted = 0, rounds = 0;
  ASSERT_TRUE(model->mtp_decode(mtpc, first, kGen + 1, got, /*draft_len=*/1,
                                &accepted, &rounds, ctl));

  // got = [first, s1, s2, ...]; ref = [s1, s2, ...]. Compare the overlap.
  int mism = 0;
  const std::size_t gov = got.empty() ? 0 : got.size() - 1;
  const std::size_t nn = std::min(ref.size(), gov);
  for (std::size_t i = 0; i < nn; ++i) {
    if (ref[i] != got[i + 1]) { ++mism; }
  }
  std::printf("[qwen_optiq_mtp_sampled] ref=%zu got=%zu rounds=%ld "
              "tok/round=%.2f mism=%d\n",
              ref.size(), got.size(), rounds,
              rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
  EXPECT_TRUE(mism == 0);            // sampled output is token-exact vs serial
  EXPECT_TRUE(got.size() == ref.size() + 1);   // mtp includes `first`
  EXPECT_TRUE(rounds < (long)got.size());      // drafts actually landed (speedup)
}

// MTP through the PUBLIC LoadedLanguageModel::mtp_generate path -- the exact
// entry the text-chat / visual-qa / realtime-vqa stages call. The streaming
// (on_tokens) + stop-token (is_stop) decode MUST reproduce the serial
// next_token_greedy loop the stages run WITHOUT MTP, token-for-token, AND stop
// at the same place with the stop token rolled OUT of the context (so the
// subsequent assistant_close commit lands cleanly). Two independent contexts
// off the same prefill so the reference and MTP start identically. Gated on
// VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_optiq_mtp_lm_stream_stop_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  ASSERT_TRUE(lm->mtp_available());   // the LM exposes the MTP fast path
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto is_stop = [tpl](std::int32_t id) { return tpl->is_stop_token(id); };

  const int kBudget = 220;   // thinking model: room to reach a natural stop

  // ---- Reference: the serial greedy loop a stage runs WITHOUT MTP. ----
  // ref = [first, t1, ...] up to (excluding) the stop token, exactly the
  // tokens the stage would emit; ref_hit_stop records the natural turn end.
  std::vector<std::int32_t> ref;
  bool ref_hit_stop = false;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t cur = lm->prefill(ctx, ids);
    ASSERT_TRUE(cur >= 0);
    for (int i = 0; i < kBudget; ++i) {
      if (is_stop(cur)) { ref_hit_stop = true; break; }
      ref.push_back(cur);
      cur = lm->next_token_greedy(ctx, cur);
      if (cur < 0) { break; }
    }
  }
  ASSERT_TRUE(!ref.empty());

  // ---- MTP via the public streaming API (fresh context, same prefill). ----
  std::vector<std::int32_t> got;
  int  produced = 0;
  bool hit_stop = false;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    auto on_toks = [&](std::span<const std::int32_t> toks) -> bool {
      for (std::int32_t id : toks) { got.push_back(id); }
      return true;
    };
    const bool ok = lm->mtp_generate(ctx, first, kBudget,
                                     genai::SamplerParams{}, is_stop, on_toks,
                                     &produced, &hit_stop);
    EXPECT_TRUE(ok);
  }

  int mism = 0;
  const std::size_t nn = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < nn; ++i) { if (ref[i] != got[i]) { ++mism; } }
  std::printf("[qwen_optiq_mtp_lm] ref=%zu got=%zu produced=%d hit_stop=%d "
              "ref_stop=%d mism=%d\n",
              ref.size(), got.size(), produced, (int)hit_stop,
              (int)ref_hit_stop, mism);
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(got.size() == ref.size());     // same length => stop in lockstep
  EXPECT_TRUE((int)got.size() == produced);  // produced == streamed count
  EXPECT_TRUE(hit_stop == ref_hit_stop);     // ended the same way (stop/budget)
}
