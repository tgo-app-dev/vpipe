// gemma-text.cc -- Gemma-4 (safetensors) text generation and prefill: the
// video-describe smokes, thinking-block handling, dense-bf16 / 12B generate,
// and the prefill token-exact family (MMA, materialized-vs-pflash, banded
// sliding, chunked, bounded sub-block, sliding-window grow).

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

namespace {
// Minimal P6 PPM reader -> planar [3,H,W] u8 (same layout as the stage).
bool
read_ppm_planar_(const char* path, std::vector<std::uint8_t>* out,
                 int* H, int* W)
{
  std::FILE* f = std::fopen(path, "rb");
  if (!f) { return false; }
  char magic[3] = {0};
  int w = 0, h = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxv) != 4
      || std::string(magic) != "P6" || w <= 0 || h <= 0 || maxv != 255) {
    std::fclose(f); return false;
  }
  std::fgetc(f);   // single whitespace after header
  std::vector<std::uint8_t> hwc((std::size_t)w * h * 3);
  const std::size_t got = std::fread(hwc.data(), 1, hwc.size(), f);
  std::fclose(f);
  if (got != hwc.size()) { return false; }
  out->resize(hwc.size());
  const std::size_t plane = (std::size_t)w * h;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = (std::size_t)y * w + x;
      (*out)[0 * plane + p] = hwc[p * 3 + 0];
      (*out)[1 * plane + p] = hwc[p * 3 + 1];
      (*out)[2 * plane + p] = hwc[p * 3 + 2];
    }
  }
  *H = h; *W = w;
  return true;
}
}  // namespace

// Reproduces the realtime-vqa Gemma describe on the metal backend in the
// CURRENT build (so it runs in the no-MLX shipping tree, unlike the
// MLX-only llm-gemma4-model-exec tests): encode a real frame, multimodal
// prefill, pdecode describe. Env: VPIPE_METAL_GEMMA_VQA_MODEL (Gemma-4
// dir) + VPIPE_METAL_GEMMA_VQA_FRAME (a P6 PPM, e.g. a vqa-enc dump).
TEST(metal_lm_smoke, gemma_video_describe) {
  const char* path  = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL");
  const char* frame = std::getenv("VPIPE_METAL_GEMMA_VQA_FRAME");
  if (!path || !*path || !frame || !*frame) { return; }
  std::vector<std::uint8_t> rgb; int H = 0, W = 0;
  if (!read_ppm_planar_(frame, &rgb, &H, &W)) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  const char* dt = std::getenv("VPIPE_METAL_GEMMA_VQA_DTYPE");
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = (dt && *dt) ? dt : "bf16";
  spec.page_tokens = 512; spec.max_pages = 32;
  std::printf("[gemma_video_describe] compute_dtype=%s\n",
              spec.compute_dtype.c_str());
  auto lm = mgr->load(spec);
  if (!lm || !lm->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }

  auto menc = genai::MetalGemma4VisionEncoder::load(
      path, mc, genai::MetalGemma4VisionEncoder::config_from(lm->config()));
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!menc) { return; }
  auto ct = genai::make_chat_template(lm->config().architecture,
                                    lm->tokenizer());
  if (!ct) { return; }
  const std::int32_t vpad = ct->video_pad_token_id();

  auto img = menc->encode(rgb.data(), H, W, /*max_soft_tokens=*/280);
  if (img.n_tokens <= 0) { return; }
  std::printf("[gemma_video_describe] frame %dx%d -> %d tok grid %dx%d\n",
              W, H, img.n_tokens, img.grid_h, img.grid_w);

  std::vector<int>   counts{ img.n_tokens };
  std::vector<float> ts{ 0.0f };
  std::vector<std::int32_t> ids;
  ct->render_user_turn_video(
      "Briefly describe what is happening in this video in 2-3 sentences. "
      "Focus on what the people and animals are doing.",
      ts, counts, /*is_first_turn=*/true,
      std::string_view("The current time is 2026-06-14 08:00:00.\n"), &ids);
  std::vector<genai::TokenRef> refs; refs.reserve(ids.size());
  int off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == vpad && off < img.n_tokens) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_buf = &img.embeddings;
      r.image_token_offset = off++;
    } else { r.kind = genai::TokenRef::Kind::Text; r.text_id = id; }
    refs.push_back(r);
  }

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>{});
  ASSERT_TRUE(first >= 0);

  genai::SamplerParams sp;   // greedy
  const int kMax = 80;
  std::vector<std::int32_t> got;
  std::int32_t cur = ctx.last_predicted_id();
  const std::span<const std::int32_t> no_prompt;
  int produced = 0;
  if (lm->pdecode_begin(ctx, cur, no_prompt, sp, kMax)) {
    const bool runahead = lm->pdecode_supports_runahead();
    bool committed = (cur >= 0 && !ct->is_stop_token(cur))
        ? lm->pdecode_commit(ctx) : false;
    if (runahead && committed && kMax > 1) { lm->pdecode_commit(ctx); }
    while (produced < kMax) {
      if (ct->is_stop_token(cur)) { break; }
      got.push_back(cur); ++produced;
      if (produced >= kMax || !committed) { break; }
      cur = lm->pdecode_next(ctx);
      if (cur < 0) { break; }
      const bool cont = (produced + 1 < kMax) && !ct->is_stop_token(cur);
      committed = cont ? lm->pdecode_commit(ctx) : false;
    }
    lm->pdecode_end(ctx);
  } else {
    got.push_back(first);
    for (int s = 0; s < kMax; ++s) {
      const std::int32_t nx = lm->next_token(ctx);
      if (nx < 0 || ct->is_stop_token(nx)) { break; }
      got.push_back(nx);
    }
  }
  const std::string ans = lm->tokenizer().decode(
      std::span<const std::int32_t>(got.data(), got.size()));
  std::printf("[gemma_video_describe] %s\n", ans.c_str());
  EXPECT_TRUE(!ans.empty());
  // Regression guard for the bf16 multimodal splice bug: the spliced image
  // rows must reach the forward in the model's compute dtype, else (bf16)
  // the model "sees no image" and refuses. Frame-independent signature.
  std::string lc = ans;
  for (auto& ch : lc) { ch = (char)std::tolower((unsigned char)ch); }
  EXPECT_TRUE(lc.find("provide the image") == std::string::npos
              && lc.find("provide the video") == std::string::npos
              && lc.find("not provided") == std::string::npos
              && lc.find("no video or any image") == std::string::npos);
}

// A/B channel + parity check for the Gemma-4 CoreML vision tower used by
// realtime-vqa: describe the SAME real frame through the native metal Gemma
// ViT (_mgvis) AND a CoreML soft-token export, and print both. A correct
// CoreML export (right colour layout) yields a description that names the
// same dominant colours as the native tower; an R<->B swap would flip them
// (e.g. an orange frame read as azure). Env: VPIPE_METAL_GEMMA_VQA_MODEL +
// VPIPE_METAL_GEMMA_VQA_FRAME (P6 PPM) + VPIPE_METAL_GEMMA_VQA_COREML
// (.mlpackage). Skips vacuously unless all three are set.
TEST(metal_lm_smoke, gemma_video_describe_coreml) {
  const char* path  = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL");
  const char* frame = std::getenv("VPIPE_METAL_GEMMA_VQA_FRAME");
  const char* cmlp  = std::getenv("VPIPE_METAL_GEMMA_VQA_COREML");
  if (!path || !*path || !frame || !*frame || !cmlp || !*cmlp) { return; }
  std::vector<std::uint8_t> rgb; int H = 0, W = 0;
  if (!read_ppm_planar_(frame, &rgb, &H, &W)) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  const char* dt = std::getenv("VPIPE_METAL_GEMMA_VQA_DTYPE");
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = (dt && *dt) ? dt : "bf16";
  spec.page_tokens = 512; spec.max_pages = 32;
  auto lm = mgr->load(spec);
  if (!lm || !lm->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }

  auto menc = genai::MetalGemma4VisionEncoder::load(
      path, mc, genai::MetalGemma4VisionEncoder::config_from(lm->config()));
  genai::CoreMLVisionEncoder::LoadSpec cm;
  cm.mlpackage_path = cmlp;
  cm.compute_units  = 2;
  cm.patch_size     = lm->config().vision.patch_size;
  cm.spatial_merge_size = lm->config().vision.spatial_merge_size;
  auto cenc = genai::CoreMLVisionEncoder::create(cm, nullptr, &sess);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!menc || !cenc || !cenc->implemented()) { return; }

  auto ct = genai::make_chat_template(lm->config().architecture,
                                      lm->tokenizer());
  if (!ct) { return; }
  const std::int32_t vpad = ct->video_pad_token_id();

  // Shared describe: splice [n_tokens] image rows from `emb` at the video
  // placeholder, multimodal-prefill, greedy-decode a short description.
  auto describe = [&](const metal_compute::SharedBuffer& emb,
                      int n_tokens) -> std::string {
    std::vector<int>   counts{ n_tokens };
    std::vector<float> ts{ 0.0f };
    std::vector<std::int32_t> ids;
    ct->render_user_turn_video(
        "Briefly describe this image in 2-3 sentences. Name the most "
        "prominent colours you see.",
        ts, counts, /*is_first_turn=*/true,
        std::string_view("The current time is 2026-06-14 08:00:00.\n"),
        &ids);
    std::vector<genai::TokenRef> refs; refs.reserve(ids.size());
    int off = 0;
    for (std::int32_t id : ids) {
      genai::TokenRef r;
      if (id == vpad && off < n_tokens) {
        r.kind = genai::TokenRef::Kind::ImageTokens;
        r.embeddings_buf = &emb;
        r.image_token_offset = off++;
      } else {
        r.kind = genai::TokenRef::Kind::Text; r.text_id = id;
      }
      refs.push_back(r);
    }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return {}; }
    const std::int32_t first = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>{});
    if (first < 0) { return {}; }
    genai::SamplerParams sp;   // greedy
    const int kMax = 80;
    std::vector<std::int32_t> got;
    std::int32_t cur = ctx.last_predicted_id();
    int produced = 0;
    const std::span<const std::int32_t> no_prompt;
    if (lm->pdecode_begin(ctx, cur, no_prompt, sp, kMax)) {
      const bool runahead = lm->pdecode_supports_runahead();
      bool committed = (cur >= 0 && !ct->is_stop_token(cur))
          ? lm->pdecode_commit(ctx) : false;
      if (runahead && committed && kMax > 1) { lm->pdecode_commit(ctx); }
      while (produced < kMax) {
        if (ct->is_stop_token(cur)) { break; }
        got.push_back(cur); ++produced;
        if (produced >= kMax || !committed) { break; }
        cur = lm->pdecode_next(ctx);
        if (cur < 0) { break; }
        const bool cont = (produced + 1 < kMax) && !ct->is_stop_token(cur);
        committed = cont ? lm->pdecode_commit(ctx) : false;
      }
      lm->pdecode_end(ctx);
    } else {
      got.push_back(first);
      for (int s = 0; s < kMax; ++s) {
        const std::int32_t nx = lm->next_token(ctx);
        if (nx < 0 || ct->is_stop_token(nx)) { break; }
        got.push_back(nx);
      }
    }
    return lm->tokenizer().decode(
        std::span<const std::int32_t>(got.data(), got.size()));
  };

  auto nimg = menc->encode(rgb.data(), H, W, /*max_soft_tokens=*/280);
  auto cimg = cenc->encode_host(rgb.data(), H, W);
  std::printf("[gemma_coreml] native %d tok grid %dx%d | coreml %d tok "
              "grid %dx%d hidden=%d\n",
              nimg.n_tokens, nimg.grid_h, nimg.grid_w,
              cimg.n_tokens, cimg.grid_h, cimg.grid_w, cimg.out_hidden);
  ASSERT_TRUE(nimg.n_tokens > 0);
  ASSERT_TRUE(cimg.n_tokens > 0);
  // The CoreML soft tokens must match the LM hidden width or the splice
  // reads garbage rows.
  ASSERT_TRUE(cimg.out_hidden == lm->config().hidden);

  const std::string a_native = describe(nimg.embeddings, nimg.n_tokens);
  const std::string a_coreml = describe(cimg.embeddings, cimg.n_tokens);
  std::printf("[gemma_coreml] NATIVE: %s\n", a_native.c_str());
  std::printf("[gemma_coreml] COREML: %s\n", a_coreml.c_str());
  EXPECT_TRUE(!a_coreml.empty());
  // Same "the model actually sees an image" guard as the native test.
  std::string lc = a_coreml;
  for (auto& ch : lc) { ch = (char)std::tolower((unsigned char)ch); }
  EXPECT_TRUE(lc.find("provide the image") == std::string::npos
              && lc.find("provide the video") == std::string::npos
              && lc.find("not provided") == std::string::npos
              && lc.find("no video or any image") == std::string::npos);
}

// Regression: realtime-vqa (and text-chat / visual-qa) disable thinking, but
// Gemma-4 e4b IS a reasoning checkpoint -- it intermittently emits a
// `<|channel>thought ...<channel|>` block (notably on the multi-part audio
// prompt) that leaked into descriptions. We do NOT prefill an empty thought
// channel to suppress it (that makes e4b answer in open meta-reasoning, "The
// user wants me to ..." -- see gemma_video_describe_coreml). Instead the
// GemmaChatTemplate exposes sanitize_output(), mirroring the checkpoint's own
// strip_thinking macro, and realtime-vqa runs every decoded answer through it.
// This test (1) unit-checks sanitize_output on crafted text and (2) decodes a
// reasoning-prone prompt and proves the sanitized output carries no channel
// markers. Env: VPIPE_GEMMA4_TEST_MODEL_PATH (or VPIPE_METAL_GEMMA_VQA_MODEL).
TEST(metal_lm_smoke, gemma_thinking_stripped) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { path = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL"); }
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }
  auto& tok = lm->tokenizer();
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  if (!tpl) { return; }

  // (1) Unit-check the sanitizer directly: a thought block (and its markers)
  // is removed, surrounding answer text is kept; a trailing unclosed thought
  // (truncated decode) is dropped; clean text is untouched.
  EXPECT_TRUE(tpl->sanitize_output(
      "<|channel>thought\nlet me reason<channel|>The answer is 9.")
      == "The answer is 9.");
  EXPECT_TRUE(tpl->sanitize_output("Two boats on a calm lake.")
      == "Two boats on a calm lake.");
  EXPECT_TRUE(tpl->sanitize_output("Answer: 9.<|channel>thought\ncut off")
      == "Answer: 9.");
  // The detokenizer rewrites the channel tokens to the UNIFIED vpipe
  // thinking markers, so streamed text carries the marker form; the
  // sanitizer must strip that form too.
  EXPECT_TRUE(tpl->sanitize_output(
      std::string(vpipe::media_line::kThinkStart)
      + "thought\nlet me reason"
      + std::string(vpipe::media_line::kThinkEnd) + "The answer is 9.")
      == "The answer is 9.");

  // (2) End-to-end: ARM the reasoning channel directly (a `<|think|>` system
  // turn) so the real e4b checkpoint deterministically emits a
  // `<|channel>thought ...<channel|>` block under greedy decode -- exactly
  // the leak realtime-vqa sees intermittently. Then prove sanitize_output
  // strips it from the real model output. Built by hand because the e4b
  // factory path intentionally never arms thinking.
  const std::int32_t bos  = tok.special_token_id("<bos>");
  const std::int32_t sot  = tok.special_token_id("<|turn>");
  const std::int32_t eot  = tok.special_token_id("<turn|>");
  const std::int32_t think = tok.special_token_id("<|think|>");
  if (bos < 0 || sot < 0 || eot < 0 || think < 0) { return; }
  auto append = [&](std::vector<std::int32_t>* d, std::string_view s) {
    auto e = tok.encode(s);
    d->insert(d->end(), e.begin(), e.end());
  };
  std::vector<std::int32_t> ids;
  ids.push_back(bos);
  ids.push_back(sot); append(&ids, "system\n");
  ids.push_back(think); append(&ids, "\n");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids,
      "user\nA farmer has 17 sheep. All but 9 run away. How many sheep are "
      "left?");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids, "model\n");
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> gen;
  for (int i = 0; i < 160 && t >= 0; ++i) {
    if (tpl->is_stop_token(t)) { break; }
    gen.push_back(t);
    t = lm->next_token(ctx);
  }
  const std::string raw =
      tok.decode(std::span<const std::int32_t>(gen.data(), gen.size()));
  const std::string clean = tpl->sanitize_output(raw);
  // The channel token ids now DECODE as the unified vpipe thinking
  // markers (the detokenizer rewrite), so the armed thought block shows
  // up in raw as the marker form, never the literal channel strings.
  const bool raw_had_think =
      raw.find(vpipe::media_line::kThinkStart) != std::string::npos;
  std::printf("[gemma_thinking] raw_had_think=%d\n", raw_had_think ? 1 : 0);
  std::printf("[gemma_thinking] RAW  : %s\n", raw.c_str());
  std::printf("[gemma_thinking] CLEAN: %s\n", clean.c_str());
  EXPECT_TRUE(raw.find("<|channel>") == std::string::npos);

  // The user-facing string must never carry a thought channel, in
  // either the raw or the unified-marker form.
  EXPECT_TRUE(clean.find("<|channel>")  == std::string::npos);
  EXPECT_TRUE(clean.find("<channel|>") == std::string::npos);
  EXPECT_TRUE(clean.find(vpipe::media_line::kThinkStart)
              == std::string::npos);
  EXPECT_TRUE(clean.find(vpipe::media_line::kThinkEnd)
              == std::string::npos);
  EXPECT_TRUE(!clean.empty());
}

// Dense (raw-HF bf16/f16) Gemma-4 coherent-generation smoke. The raw google
// gemma-4 releases (E2B/E4B/12B ...) ship UNQUANTIZED .weight tensors; the
// metal gemma exec detects that and runs the dense GEMM/GEMV path (no affine
// dequant, no scales/biases) instead of the quantized qmv/qmm kernels. A
// deterministic factual prompt under greedy decode must produce the known
// answer -- a broken dense forward (wrong transpose/norm/geglu/double-wide
// ffn) yields word-salad or the wrong token. This is the bf16 bring-up gate
// (token-exact-vs-omlx is a separate, reference-tooling-gated check). Env:
// VPIPE_GEMMA4_DENSE_TEST_MODEL_PATH = a raw bf16 gemma-4 dir (e.g.
// google/gemma-4-E2B-it).
TEST(metal_lm_smoke, gemma_dense_bf16_generates) {
  const char* path = std::getenv("VPIPE_GEMMA4_DENSE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  // minitest's ASSERT_TRUE records the failure but does NOT return, so a
  // failed load has to be short-circuited by hand -- otherwise the null
  // deref below SIGSEGVs and takes the whole test binary down with it.
  ASSERT_TRUE(lm && lm->valid());
  if (!lm || !lm->valid()) { return; }
  auto& tok = lm->tokenizer();
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  ASSERT_TRUE((bool)tpl);

  // Deterministic factual prompt -> greedy -> a competent gemma-4 instruct
  // answers "Paris" (leading token(s) may be whitespace/markdown; substring).
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with only the city name.",
      /*is_first_turn=*/true, &ids);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> gen;
  for (int i = 0; i < 24 && t >= 0; ++i) {
    if (tpl->is_stop_token(t)) { break; }
    gen.push_back(t);
    t = lm->next_token(ctx);
  }
  const std::string out =
      tok.decode(std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[gemma_dense] OUT: %s\n", out.c_str());
  EXPECT_TRUE(!gen.empty());
  EXPECT_TRUE(out.find("Paris") != std::string::npos);
}

// Per-tensor mixed-precision (OptiQ) Gemma-4-12B coherent-generation smoke. The
// mlx-community gemma-4-12B-it-OptiQ-4bit pack quantizes each projection at its
// OWN width (some 4-bit, some 8-bit, varying per layer) -- the metal gemma model
// detects this (_mixed), binds each projection with its own bits and de-fuses
// gate|up / QKV whenever widths differ, then dispatches the width-correct
// w4/w8 kernel per tensor. A wrong per-tensor binding (an 8-bit weight parsed at
// the 4-bit stride) yields word-salad, so the greedy factual answer is the
// bring-up gate (token-exact-vs-omlx is a separate reference-tooling check).
// Env: VPIPE_GEMMA12B_OPTIQ_TEST_MODEL_PATH = the OptiQ model dir.
TEST(metal_lm_smoke, gemma12b_optiq_generates) {
  const char* path = std::getenv("VPIPE_GEMMA12B_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  // minitest's ASSERT_TRUE records the failure but does NOT return, so a
  // failed load has to be short-circuited by hand -- otherwise the null
  // deref below SIGSEGVs and takes the whole test binary down with it.
  ASSERT_TRUE(lm && lm->valid());
  if (!lm || !lm->valid()) { return; }
  auto& tok = lm->tokenizer();
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  ASSERT_TRUE((bool)tpl);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "Name the capital of France and give one sentence about it.",
      /*is_first_turn=*/true, &ids);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> gen;
  for (int i = 0; i < 48 && t >= 0; ++i) {
    if (tpl->is_stop_token(t)) { break; }
    gen.push_back(t);
    t = lm->next_token(ctx);
  }
  const std::string out =
      tok.decode(std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[gemma12b_optiq] OUT: %s\n", out.c_str());
  std::string idline;
  for (std::int32_t g : gen) { idline += std::to_string(g) + " "; }
  std::printf("[gemma12b_optiq] IDS: %s\n", idline.c_str());
  EXPECT_TRUE(gen.size() >= 5);
  EXPECT_TRUE(out.find("Paris") != std::string::npos);
}

// The REALTIME fix: don't just strip the thought block after the fact (that
// still pays its decode cost) -- forbid the reasoning-channel tokens at the
// logit level so the model never GENERATES the block, keeping decode short.
// set_suppressed_tokens masks those logits after softcap, across prefill +
// every decode path. This test ARMS reasoning (a <|think|> system turn) so
// the model WOULD emit `<|channel>thought ...<channel|>` under greedy, then
// shows that with suppression the channel never appears AND the decode is
// strictly shorter (the token-budget win). Env: VPIPE_GEMMA4_TEST_MODEL_PATH
// (or VPIPE_METAL_GEMMA_VQA_MODEL).
TEST(metal_lm_smoke, gemma_thinking_suppressed) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { path = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL"); }
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }
  auto& tok = lm->tokenizer();
  const std::int32_t bos   = tok.special_token_id("<bos>");
  const std::int32_t sot   = tok.special_token_id("<|turn>");
  const std::int32_t eot   = tok.special_token_id("<turn|>");
  const std::int32_t think = tok.special_token_id("<|think|>");
  const std::int32_t chan  = tok.special_token_id("<|channel>");
  if (bos < 0 || sot < 0 || eot < 0 || think < 0 || chan < 0) { return; }
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  if (!tpl) { return; }

  // Hand-built ARMED prompt: a <|think|> system turn forces the reasoning
  // channel on the next model turn.
  auto append = [&](std::vector<std::int32_t>* d, std::string_view s) {
    auto e = tok.encode(s);
    d->insert(d->end(), e.begin(), e.end());
  };
  std::vector<std::int32_t> ids;
  ids.push_back(bos);
  ids.push_back(sot); append(&ids, "system\n");
  ids.push_back(think); append(&ids, "\n");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids,
      "user\nA farmer has 17 sheep. All but 9 run away. How many are left?");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids, "model\n");

  // Greedy decode from a fresh context; returns (generated ids).
  auto decode = [&]() -> std::vector<std::int32_t> {
    auto ctx = lm->make_context();
    std::vector<std::int32_t> gen;
    if (!ctx.valid()) { return gen; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 160 && t >= 0; ++i) {
      if (tpl->is_stop_token(t)) { break; }
      gen.push_back(t);
      t = lm->next_token(ctx);
    }
    return gen;
  };
  auto has_chan = [&](const std::vector<std::int32_t>& v) {
    return std::find(v.begin(), v.end(), chan) != v.end();
  };

  // A) No suppression: the armed model emits the thought channel (long).
  lm->set_suppressed_tokens(std::span<const std::int32_t>{});
  const auto gen_unsup = decode();
  // B) Suppression on: the channel token can never be predicted (short).
  const std::int32_t bans[] = { chan, think };
  lm->set_suppressed_tokens(std::span<const std::int32_t>(bans, 2));
  const auto gen_sup = decode();

  std::printf("[gemma_suppress] unsuppressed: %d tok, channel=%d\n",
              (int)gen_unsup.size(), has_chan(gen_unsup) ? 1 : 0);
  std::printf("[gemma_suppress]   suppressed: %d tok, channel=%d\n",
              (int)gen_sup.size(), has_chan(gen_sup) ? 1 : 0);
  std::printf("[gemma_suppress] answer: %s\n",
              tok.decode(std::span<const std::int32_t>(
                  gen_sup.data(), gen_sup.size())).c_str());

  ASSERT_TRUE(!gen_unsup.empty());
  ASSERT_TRUE(!gen_sup.empty());
  // The armed run must actually reach for the channel (else the test is
  // vacuous); suppression must keep it out of the output entirely.
  EXPECT_TRUE(has_chan(gen_unsup));
  EXPECT_TRUE(!has_chan(gen_sup));
}

// Gemma-4 matrix-core (M5+) prefill GEMM (q/k/v/o, GeGLU gate/up via the
// interleaved-dequant + geglu_interleaved combine, down, PLE gate) must be
// greedy token-exact with the steel quantized GEMM. Loads the SAME e4b
// checkpoint twice -- steel reference (VPIPE_GEMMA_NO_MMA=1) and matrix-core
// -- prefills a prompt and greedy-decodes; the two token streams must match.
// VPIPE_GEMMA_MMA_MIN_M is lowered so a short prompt routes through the dense
// matmul2d path. On a GPU without matrix cores both loads are steel (a
// trivial, still-valid pass). Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_mma_prefill_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool use_mma) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_MMA_MIN_M", "4", 1);  // exercise mma on short prompts
    if (use_mma) { ::unsetenv("VPIPE_GEMMA_NO_MMA"); }
    else         { ::setenv("VPIPE_GEMMA_NO_MMA", "1", 1); }
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
  ::unsetenv("VPIPE_GEMMA_NO_MMA");
  ::unsetenv("VPIPE_GEMMA_MMA_MIN_M");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_mma_prefill_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// The materialized global-attention prefill (default) must be greedy
// token-exact with the pflash reference (materialized OFF). REGRESSION GUARD
// for the steel dense_gemm_t K-tail bug: the materialized PV GEMM has
// contraction K = T_kv, and the kernel read the final K-block unmasked when
// T_kv % 32 != 0, spilling into the next packed row. The prompt below is
// deliberately NOT a multiple of 32 tokens so the tail block exists; a
// power-of-two ctx (the perf bench) never tripped it. The sibling mma/steel
// token-exact tests both run materialized, so they could NOT catch this --
// this one pins materialized against pflash. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_materialized_matches_pflash_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool materialized) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_MATERIALIZED_GLOBAL", materialized ? "1" : "0", 1);
    // Force the DENSE (non-causal) materialized GEMMs: the plain dense_gemm_t
    // is the kernel that had the K-tail bug; the causal variants mostly mask
    // it, so pin the dense path here to keep this a real regression guard.
    ::setenv("VPIPE_GEMMA_MAT_CAUSAL", "0", 1);
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
    // 47 tokens (NOT a multiple of 32) -> the PV contraction has a tail block.
    auto ids = lm->tokenizer().encode(
        "The old lighthouse keeper counted seven ships passing the rocky "
        "point before dawn, each one heavier and slower than the last, and "
        "he wondered which captain would dare the narrow channel.");
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

  const auto ref = run(false);   // pflash reference
  const auto got = run(true);    // materialized, dense GEMMs (K-tail path)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_MATERIALIZED_GLOBAL");
  ::unsetenv("VPIPE_GEMMA_MAT_CAUSAL");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_materialized_matches_pflash_token_exact] "
              "%zu tokens, %zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Banded-materialized SLIDING prefill must be greedy token-exact with the
// simdgroup-flash sliding path. REGRESSION GUARD for the banded GEMM/softmax:
// the prompt is LONGER than the sliding window (512), with VARIED tokens, so
// the trailing-window band actually engages (k0>0 for later rows, and the QK
// below-window tile skips fire) -- every other gemma token-exact prompt is
// <512 tokens, so the band is a no-op there and they cannot catch a banding
// bug. Both arms keep materialized GLOBAL on; only the sliding path differs
// (VPIPE_GEMMA_MAT_SLIDING toggles banded-GEMM vs flash). Gated on
// VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_banded_sliding_matches_flash_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool banded) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_MAT_SLIDING", banded ? "1" : "0", 1);
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
    // 700 VARIED tokens (> sliding_window 512) so the band engages for the
    // later positions. Deterministic pseudo-random ids keep K/V non-uniform
    // (a uniform prompt would hide a window bug -- all keys give the same V).
    std::vector<std::int32_t> ids;
    ids.reserve(700);
    ids.push_back(2);                       // <bos>
    std::uint32_t s = 0x9e3779b9u;
    for (int i = 1; i < 700; ++i) {
      s = s * 1664525u + 1013904223u;       // LCG
      ids.push_back((std::int32_t)(106 + (s >> 9) % 200000));
    }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 16 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // flash sliding
  const auto got = run(true);    // banded-materialized sliding
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_MAT_SLIDING");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_banded_sliding_matches_flash_token_exact] "
              "%zu tokens, %zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Chunked sliding-window prefill (ring wrap) must be greedy token-exact with a
// single-pass prefill. Forces wrapping with a small VPIPE_GEMMA_SLIDING_CHUNK
// on a prompt longer than the ring, vs a chunk large enough to prefill in one
// pass. Validates the ring chunking AND the KV-only intermediate-chunk skip
// (the shared-KV tail + lm_head are dropped for non-final chunks; only the
// own-KV bulk layers run, so the cache they leave must be bit-identical to the
// single-pass cache). Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_chunked_prefill_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  // ~900-token prompt (a sentence repeated) so it exceeds a small ring.
  std::string prompt;
  for (int i = 0; i < 40; ++i) {
    prompt += "The clever clockmaker studied the broken machine carefully, "
              "noting each worn gear and bent spring before she began again. ";
  }

  auto run = [&](int chunk) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", std::to_string(chunk).c_str(), 1);
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
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto chunked = run(128);    // ring=640 < ~900 -> wraps (multi-chunk)
  const auto single  = run(4096);   // ring caps at max_seq -> single pass
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ASSERT_TRUE(!single.empty());
  ASSERT_TRUE(chunked.size() == single.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < single.size(); ++i) {
    if (chunked[i] != single[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_chunked_prefill_token_exact] %zu tokens, "
              "%zu mismatches (chunk=128 vs 4096)\n", single.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Bounded-ring SINGLE-PASS prefill (VPIPE_GEMMA_PREFILL_SUBBLOCK) must agree
// with the GROWN one-pass prefill on a LONG prompt that wraps the bounded ring
// many times. The bounded path runs ONE forward over the whole prompt (large
// proj/FFN/global GEMM batch) and reads the full-batch K/V for the sliding
// attention (ring-independent), writing the bounded ring once; the grown path
// grows the ring to the full prompt. The bounded CHUNKED path (subblock off)
// chunks the whole stack instead. All three must decode the same distribution
// -- which also proves the single full-n ring write leaves the correct
// trailing window resident for decode at depth. ~2400 VARIED tokens (>> ring
// 1024) so the ring wraps repeatedly and a window/decode-state bug shows.
//
// Compares LOGITS within a tolerance, not tokens. The three schedules batch
// their GEMMs differently, so they agree to about one f16 ULP but not bit-
// exactly, and a token comparison escalates that into a whole-tail divergence
// the moment two candidates are within a ULP of each other. That is exactly
// what used to fail here: at step 10 the chunked arm had tokens 1816 and 3303
// at 27.906250/27.859375 where grown had them at 27.906250/27.875000, so the
// top-2 order swapped and the remaining 13 of 24 positions all differed from
// one 2-ULP wobble. The 2400 pseudo-random ids make that likely -- they are
// deliberately off-distribution, which flattens the logits. Every arm is
// teacher-forced on the grown arm's tokens so all three stay on the same
// sequence and each step's logits are directly comparable.
// Gated on the Gemma model.
TEST(metal_lm_smoke, gemma_bounded_subblock_matches_grown_logits) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  // 2400 deterministic pseudo-random ids (non-uniform K/V so a window bug
  // can't hide), >> the bounded ring (window 512 + chunk 512 = 1024).
  std::vector<std::int32_t> ids;
  ids.reserve(2400);
  ids.push_back(2);                         // <bos>
  std::uint32_t s = 0x12345678u;
  for (int i = 1; i < 2400; ++i) {
    s = s * 1664525u + 1013904223u;
    ids.push_back((std::int32_t)(106 + (s >> 9) % 200000));
  }
  const int kGen = 24;

  struct Arm {
    std::vector<std::int32_t>       toks;
    std::vector<std::vector<float>> logits;   // [step][vocab]
  };
  // mode 0 = grown (A); 1 = bounded subblock (C); 2 = bounded chunked (B).
  // `drive` teacher-forces the per-step input token; null = pick its own.
  auto run = [&](int mode, const std::vector<std::int32_t>* drive) {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", "512", 1);
    if (mode == 0) {
      ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");   // grown one-pass (A)
      ::unsetenv("VPIPE_GEMMA_PREFILL_SUBBLOCK");
    } else if (mode == 1) {
      ::setenv("VPIPE_GEMMA_NO_SLIDING_GROW", "1", 1);  // bounded subblock (C)
      ::setenv("VPIPE_GEMMA_PREFILL_SUBBLOCK", "1", 1);
    } else {
      ::setenv("VPIPE_GEMMA_NO_SLIDING_GROW", "1", 1);  // bounded chunked (B)
      ::setenv("VPIPE_GEMMA_PREFILL_SUBBLOCK", "0", 1);
    }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    Arm out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < kGen && t >= 0; ++i) {
      out.logits.push_back(lm->last_logits_host());
      out.toks.push_back(t);
      const bool forced = (drive != nullptr && i < (int)drive->size());
      const std::int32_t feed = forced ? (*drive)[(std::size_t)i] : t;
      t = lm->next_token(ctx, feed);
    }
    return out;
  };

  const Arm A = run(0, nullptr);                 // grown drives the sequence
  const Arm C = run(1, &A.toks);                 // bounded subblock, forced
  const Arm B = run(2, &A.toks);                 // bounded chunked, forced
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");
  ::unsetenv("VPIPE_GEMMA_PREFILL_SUBBLOCK");
  ASSERT_TRUE(!A.logits.empty());
  ASSERT_TRUE(A.logits.size() == C.logits.size());
  ASSERT_TRUE(A.logits.size() == B.logits.size());
  if (A.logits.empty() || A.logits.size() != C.logits.size()
      || A.logits.size() != B.logits.size()) {
    return;
  }
  // Worst per-element gap and worst per-step relative L2 vs the grown arm.
  auto cmp = [&](const Arm& x) {
    double mx = 0.0, mrel = 0.0;
    for (std::size_t st = 0; st < A.logits.size(); ++st) {
      const auto& a = A.logits[st];
      const auto& b = x.logits[st];
      if (a.size() != b.size()) { return std::make_pair(1e30, 1e30); }
      double num = 0.0, den = 0.0;
      for (std::size_t v = 0; v < a.size(); ++v) {
        const double d = (double)b[v] - (double)a[v];
        if (std::fabs(d) > mx) { mx = std::fabs(d); }
        num += d * d;
        den += (double)a[v] * (double)a[v];
      }
      const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
      if (rel > mrel) { mrel = rel; }
    }
    return std::make_pair(mx, mrel);
  };
  const auto ca = cmp(C), ba = cmp(B);
  std::printf("[metal_lm_smoke.gemma_bounded_subblock] %zu steps | "
              "subblock-vs-grown max|dlogit|=%.6f rel-L2=%.6g | "
              "chunked-vs-grown max|dlogit|=%.6f rel-L2=%.6g\n",
              A.logits.size(), ca.first, ca.second, ba.first, ba.second);
  // f16 logits near 28 step by 0.015625. Measured on M5 across runs:
  //   subblock vs grown -- 0.000000 (bit-identical) to 0.191, rel-L2 <= 2.8e-4
  //   chunked  vs grown -- 0.437 to 0.519 (~30 ULP), rel-L2 <= 5.6e-4
  // Neither arm is bit-stable run to run: the subblock arm is often exactly
  // equal to grown but not always, so do NOT tighten this to 0 on the strength
  // of a couple of lucky runs (it was, and it failed on the third). Chunking
  // the stack to <= page tokens reshapes every GEMM batch and reduction order,
  // which is why that arm drifts furthest; a rel-L2 of 5.6e-4 says the
  // distribution is intact either way. One bound for both arms, ~3x above the
  // worst seen and far under a real window/decode-state bug, which puts whole
  // logits O(1) apart on a signal of O(28) and rel-L2 in the 0.1+ range --
  // the failure this test exists to catch.
  EXPECT_TRUE(ca.first <= 1.5);
  EXPECT_TRUE(ca.second <= 5e-3);
  EXPECT_TRUE(ba.first <= 1.5);
  EXPECT_TRUE(ba.second <= 5e-3);
}

// MULTIMODAL prefill must also respect the sliding-window ring. prefill_mm
// (the image/audio splice path) does a single-shot forward; when the prefix
// exceeds the ring it MUST grow the sliding ring first (like prefill()), else
// the single pass wraps the ring and clobbers in-window keys -> corrupted
// sliding/local-attention layers (fluent but ungrounded VQA). A multi-frame
// video scene is always longer than the ring, so this is the realtime-vqa
// path. Build a long multimodal prefix (synthetic audio soft-tokens + a long
// text tail) and require chunk=128 (ring 640 < prefix -> grows) to be greedy
// token-exact with chunk=4096 (ring caps high -> single pass, no wrap). If the
// grow is dropped, chunk=128 wraps and mismatches. Gated on the Gemma model.
TEST(metal_lm_smoke, gemma_mm_prefill_sliding_grow_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  std::string prompt;
  for (int i = 0; i < 40; ++i) {
    prompt += "The clever clockmaker studied the broken machine carefully, "
              "noting each worn gear and bent spring before she began again. ";
  }
  auto run = [&](int chunk) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", std::to_string(chunk).c_str(), 1);
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
    const int hidden = lm->config().hidden;
    if (hidden <= 0) { return out; }
    // A few synthetic audio soft-tokens (deterministic, finite) at the head
    // -> n_mm > 0 routes through prefill_mm; empty image_grids => 1-D RoPE.
    const int n_aud = 16;
    std::vector<float> aud(static_cast<std::size_t>(n_aud) * hidden);
    for (std::size_t i = 0; i < aud.size(); ++i) {
      aud[i] = 0.01f * static_cast<float>((int)(i % 13) - 6);
    }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    std::vector<genai::TokenRef> refs;
    refs.reserve(static_cast<std::size_t>(n_aud) + ids.size());
    for (int k = 0; k < n_aud; ++k) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::AudioTokens;
      r.audio_token_offset = k;
      r.embeddings_host = &aud;
      r.host_hidden = hidden;
      refs.push_back(r);
    }
    for (std::int32_t id : ids) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
      refs.push_back(r);
    }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>());
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };
  const auto grown  = run(128);    // ring 640 < prefix -> prefill_mm grows
  const auto single = run(4096);   // ring caps high -> single pass, no wrap
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ASSERT_TRUE(!single.empty());
  ASSERT_TRUE(grown.size() == single.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < single.size(); ++i) {
    if (grown[i] != single[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_mm_prefill_sliding_grow] %zu tokens, "
              "%zu mismatches (chunk=128 grow vs 4096 single)\n",
              single.size(), mism);
  EXPECT_TRUE(mism == 0);
}

TEST(metal_lm_smoke, gemma_text_chat_default_backend) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // Nothing must force a backend (a prior test may have set it).
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());      // loads with NO env var

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);                    // refutes "no chat template"
  EXPECT_TRUE(tpl->family_name() == "gemma");

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Answer in one word.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  EXPECT_TRUE(ids.front() == 2);                  // <bos>
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill(ctx, ids);
  EXPECT_TRUE(first >= 0);
  std::printf("[metal_lm_smoke.gemma_text_chat_default_backend] first=%d\n",
              (int)first);
}

// gemma4_unified 12B on the metal backend (no-MLX coverage): no PLE, k_eq_v
// full layers (1 K/V head, no v_proj), sliding 8 K/V heads, mixed 4/8-bit
// quant. Feed the raw oracle prompt ids and require the greedy continuation
// to match /tmp/gemma12b_text_oracle.py token-for-token. Gated on
// VPIPE_GEMMA12B_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma12b_unified_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA12B_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");       // no-MLX default == metal
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

  const std::vector<std::int32_t> prompt{
      2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 1567, 1806, 5905,
      7913, 236761, 106, 107, 105, 4368, 107};
  const std::vector<std::int32_t> oracle{
      100, 45518, 107, 818, 2430, 563, 10980, 573, 506, 5618, 529, 1806, 5905,
      7913, 236761};

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> got;
  got.push_back(lm->prefill(ctx, prompt));
  for (std::size_t i = 1; i < oracle.size(); ++i) {
    const std::int32_t nxt = lm->next_token(ctx);
    if (nxt < 0) { break; }
    got.push_back(nxt);
  }
  EXPECT_TRUE(got.size() == oracle.size());
  for (std::size_t i = 0; i < oracle.size() && i < got.size(); ++i) {
    EXPECT_TRUE(got[i] == oracle[i]);
  }
  std::printf("[metal_lm_smoke.gemma12b_unified_token_exact] first=%d\n",
              got.empty() ? -1 : (int)got[0]);
}
