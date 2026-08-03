// multimodal-vqa.cc -- Multimodal (image / audio / video) end-to-end paths:
// image-VQA decode and its MTP token-exact twin, the 12B unified + OptiQ VQA
// and A/V end-to-end smokes, and the GGUF vision-tower load / equivalence
// checks.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

// Metal image-VQA smoke: load a Qwen3-VL model on the metal backend,
// encode a synthetic RGB image with the metal vision tower, splice
// text + image tokens via prefill_multimodal_metal (3-axis mROPE), and
// decode a few tokens. Runs in BOTH builds; proves the no-MLX image-VQA
// path end-to-end. Env: VPIPE_METAL_VQA_SMOKE_MODEL=/path/Qwen3.5-4B-...
TEST(metal_lm_smoke, image_vqa_decode) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
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
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto* vis = lm->metal_vision_encoder();
  ASSERT_TRUE(vis != nullptr);
  const int S = vis->config().spatial_merge;

  // Synthetic 128x128 RGB-planar image.
  const int H = 128, W = 128;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        img[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x + 2 * y + 40 * c) & 0xFF);
      }
    }
  }
  auto enc = vis->encode(img.data(), H, W);
  ASSERT_TRUE(enc.n_tokens > 0 && !enc.embeddings.empty());
  const int n_im = enc.n_tokens;
  const int mh = enc.grid_h / S, mw = enc.grid_w / S;
  ASSERT_TRUE(mh * mw == n_im);

  // Build refs: text prefix + image-token run (referencing the host
  // embeddings) + text suffix.
  auto pre = lm->tokenizer().encode("Describe the image:");
  auto suf = lm->tokenizer().encode("\nAnswer:");
  std::vector<genai::TokenRef> refs;
  refs.reserve(pre.size() + n_im + suf.size());
  for (auto id : pre) {
    genai::TokenRef r;
    r.kind = genai::TokenRef::Kind::Text;
    r.text_id = id;
    refs.push_back(r);
  }
  for (int off = 0; off < n_im; ++off) {
    genai::TokenRef r;
    r.kind = genai::TokenRef::Kind::ImageTokens;
    r.embeddings_buf = &enc.embeddings;
    r.image_token_offset = off;
    refs.push_back(r);
  }
  for (auto id : suf) {
    genai::TokenRef r;
    r.kind = genai::TokenRef::Kind::Text;
    r.text_id = id;
    refs.push_back(r);
  }

  std::pair<int, int> grid{mh, mw};
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>(&grid, 1));
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen = {first};
  for (int i = 0; i < 8; ++i) {
    const std::int32_t nx = lm->next_token(ctx);
    if (nx < 0) { break; }
    gen.push_back(nx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.image_vqa] n_im=%d grid=%dx%d | gen='%s'\n",
              n_im, mh, mw, text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// MULTIMODAL MTP token-exactness: the spec-decode path on a POST-IMAGE context
// (rope_first >= 0, the mROPE-advanced position) MUST reproduce the serial
// greedy loop the stages run without MTP -- this is what visual-qa /
// realtime-vqa scene-describe exercise. Encodes a synthetic image, splices it
// via prefill_multimodal_metal (3-axis mROPE), then compares lm->mtp_generate
// against a next_token_greedy reference from the IDENTICAL post-image state.
// Validates the rope_delta threading (the riskiest part of the stage wiring).
// Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH (the only model here with BOTH a
// vision tower -- via the optiq_vision.safetensors sidecar -- and an MTP head).
TEST(metal_lm_smoke, image_vqa_mtp_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  auto* vis = lm->metal_vision_encoder();
  ASSERT_TRUE(vis != nullptr);          // the OptiQ vision sidecar now loads
  ASSERT_TRUE(lm->mtp_available());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  const int S = vis->config().spatial_merge;

  // Synthetic 128x128 RGB-planar image (same fixture as image_vqa_decode).
  const int H = 128, W = 128;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        img[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x + 2 * y + 40 * c) & 0xFF);
      }
    }
  }
  auto enc = vis->encode(img.data(), H, W);
  ASSERT_TRUE(enc.n_tokens > 0 && !enc.embeddings.empty());
  const int n_im = enc.n_tokens;
  const int mh = enc.grid_h / S, mw = enc.grid_w / S;
  ASSERT_TRUE(mh * mw == n_im);
  std::pair<int, int> grid{mh, mw};

  // Rebuild the text+image+text ref stream per context (refs borrow enc.
  // embeddings, which outlive both prefills).
  auto build_refs = [&]() {
    auto pre = lm->tokenizer().encode("Describe the image:");
    auto suf = lm->tokenizer().encode("\nAnswer:");
    std::vector<genai::TokenRef> refs;
    refs.reserve(pre.size() + (std::size_t)n_im + suf.size());
    for (auto id : pre) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
      refs.push_back(r);
    }
    for (int off = 0; off < n_im; ++off) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_buf = &enc.embeddings;
      r.image_token_offset = off;
      refs.push_back(r);
    }
    for (auto id : suf) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
      refs.push_back(r);
    }
    return refs;
  };
  auto is_stop = [tpl](std::int32_t id) { return tpl->is_stop_token(id); };
  const int kBudget = 64;

  // Reference: the serial greedy loop a stage runs WITHOUT MTP, over the
  // post-image mROPE positions (next_token_greedy reads ctx._rope_next_position).
  std::vector<std::int32_t> ref;
  bool ref_stop = false;
  {
    auto refs = build_refs();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t cur = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>(&grid, 1));
    ASSERT_TRUE(cur >= 0);
    for (int i = 0; i < kBudget; ++i) {
      if (is_stop(cur)) { ref_stop = true; break; }
      ref.push_back(cur);
      cur = lm->next_token_greedy(ctx, cur);
      if (cur < 0) { break; }
    }
  }
  ASSERT_TRUE(!ref.empty());

  // MTP from the IDENTICAL post-image state (fresh context, same prefill).
  std::vector<std::int32_t> got;
  int  produced = 0;
  bool hit_stop = false;
  {
    auto refs = build_refs();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t first = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>(&grid, 1));
    ASSERT_TRUE(first >= 0);
    auto on_toks = [&](std::span<const std::int32_t> toks) -> bool {
      for (std::int32_t id : toks) { got.push_back(id); }
      return true;
    };
    EXPECT_TRUE(lm->mtp_generate(ctx, first, kBudget, genai::SamplerParams{},
                                 is_stop, on_toks, &produced, &hit_stop));
  }

  int mism = 0;
  const std::size_t nn = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < nn; ++i) { if (ref[i] != got[i]) { ++mism; } }
  std::printf("[image_vqa_mtp] n_im=%d ref=%zu got=%zu produced=%d hit=%d "
              "ref_stop=%d mism=%d\n",
              n_im, ref.size(), got.size(), produced, (int)hit_stop,
              (int)ref_stop, mism);
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(got.size() == ref.size());     // stopped in lockstep
  EXPECT_TRUE((int)got.size() == produced);
  EXPECT_TRUE(hit_stop == ref_stop);
}

// Gemma-4-12B "unified" (gemma4_unified, GGUF) END-TO-END vision VQA:
// loader detects the sibling mmproj -> builds Gemma4UnifiedEmbedder ->
// encode_image (host f32) -> Gemma VLM chat render -> owns_kv metal
// multimodal splice (prefill_multimodal_metal) -> greedy decode. Confirms
// the whole wiring at runtime (the embedder graph itself is golden-checked
// in gemma4_unified_embedder.*). Gated on the GGUF dir + a P6 PPM image.
TEST(metal_lm_smoke, gemma12b_unified_vqa_e2e) {
  const char* dir = std::getenv("VPIPE_GEMMA12B_GGUF_PATH");
  const char* imgp = std::getenv("VPIPE_GEMMA12B_TEST_IMAGE");
  if (!dir || !*dir || !imgp || !*imgp) { return; }
  // Read a binary P6 PPM -> planar [3,H,W] u8.
  std::FILE* f = std::fopen(imgp, "rb");
  if (!f) { return; }
  char magic[3] = {0};
  int W = 0, H = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) != 4 ||
      std::string(magic) != "P6" || W <= 0 || H <= 0) {
    std::fclose(f); return;
  }
  std::fgetc(f);   // single whitespace after maxval
  std::vector<std::uint8_t> inter((std::size_t)3 * H * W);
  const std::size_t got = std::fread(inter.data(), 1, inter.size(), f);
  std::fclose(f);
  ASSERT_TRUE(got == inter.size());
  std::vector<std::uint8_t> planar((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        planar[((std::size_t)c * H + y) * W + x] =
            inter[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.compute_dtype = "f16";
  spec.page_tokens = 1024;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Loader plumbing: the unified embedder must be constructed from the
  // sibling mmproj GGUF.
  auto* uni = lm->gemma4_unified_embedder();
  ASSERT_TRUE(uni != nullptr);
  ASSERT_TRUE(uni->has_vision());

  auto enc = uni->encode_image(planar.data(), H, W);
  ASSERT_TRUE(enc.has_value());
  ASSERT_TRUE(enc->n_tokens > 0);
  const int n_im = enc->n_tokens;

  auto tpl = genai::make_chat_template(lm->config().architecture,
                                     lm->tokenizer());
  ASSERT_TRUE(tpl != nullptr);
  const std::int32_t image_pad = tpl->image_pad_token_id();
  std::vector<std::int32_t> ids;
  const int counts[1] = {n_im};
  tpl->render_user_turn_vlm("Describe this image in one sentence.",
                            std::span<const int>(counts, 1),
                            /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int img_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == image_pad && img_off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_host = &enc->rows;
      r.image_token_offset = img_off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  ASSERT_TRUE(img_off == n_im);

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs), {});
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen;
  std::int32_t nx = first;
  for (int i = 0; i < 64 && nx >= 0 && !tpl->is_stop_token(nx); ++i) {
    gen.push_back(nx);
    nx = lm->next_token(ctx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.gemma12b_unified_vqa] %dx%d -> %d img tok | "
              "gen(%zu)='%s'\n", W, H, n_im, gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Gemma-4-12B OptiQ (per-tensor mixed-precision) IMAGE end-to-end: the same
// unified-embedder VQA path, but the model is the mixed 4/8-bit OptiQ pack and
// its vision/audio adaptor ships in the optiq/optiq_vision.safetensors shard
// (dense bf16). Confirms the raw-safetensors unified embedder loads
// (has_unified_safetensors -> load_safetensors), encode_image projects raw
// patches to soft tokens, and the mixed forward consumes text+image tokens.
// Gated on VPIPE_GEMMA12B_OPTIQ_TEST_MODEL_PATH + VPIPE_GEMMA12B_TEST_IMAGE.
TEST(metal_lm_smoke, gemma12b_optiq_vqa_e2e) {
  const char* dir = std::getenv("VPIPE_GEMMA12B_OPTIQ_TEST_MODEL_PATH");
  const char* imgp = std::getenv("VPIPE_GEMMA12B_TEST_IMAGE");
  if (!dir || !*dir || !imgp || !*imgp) { return; }
  std::FILE* f = std::fopen(imgp, "rb");
  if (!f) { return; }
  char magic[3] = {0};
  int W = 0, H = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) != 4 ||
      std::string(magic) != "P6" || W <= 0 || H <= 0) {
    std::fclose(f); return;
  }
  std::fgetc(f);
  std::vector<std::uint8_t> inter((std::size_t)3 * H * W);
  const std::size_t got = std::fread(inter.data(), 1, inter.size(), f);
  std::fclose(f);
  ASSERT_TRUE(got == inter.size());
  std::vector<std::uint8_t> planar((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        planar[((std::size_t)c * H + y) * W + x] =
            inter[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 1024;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // The unified embedder must be built from the optiq_vision safetensors shard.
  auto* uni = lm->gemma4_unified_embedder();
  ASSERT_TRUE(uni != nullptr);
  ASSERT_TRUE(uni->has_vision());

  auto enc = uni->encode_image(planar.data(), H, W);
  ASSERT_TRUE(enc.has_value());
  ASSERT_TRUE(enc->n_tokens > 0);
  const int n_im = enc->n_tokens;

  auto tpl = genai::make_chat_template(lm->config().architecture,
                                     lm->tokenizer());
  ASSERT_TRUE(tpl != nullptr);
  const std::int32_t image_pad = tpl->image_pad_token_id();
  std::vector<std::int32_t> ids;
  const int counts[1] = {n_im};
  tpl->render_user_turn_vlm("Describe this image in one sentence.",
                            std::span<const int>(counts, 1),
                            /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int img_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == image_pad && img_off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_host = &enc->rows;
      r.image_token_offset = img_off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  ASSERT_TRUE(img_off == n_im);

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs), {});
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen;
  std::int32_t nx = first;
  for (int i = 0; i < 64 && nx >= 0 && !tpl->is_stop_token(nx); ++i) {
    gen.push_back(nx);
    nx = lm->next_token(ctx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.gemma12b_optiq_vqa] %dx%d -> %d img tok | "
              "gen(%zu)='%s'\n", W, H, n_im, gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Gemma-4-12B unified AUDIO+VIDEO end-to-end: mirrors the realtime-vqa
// metal path exactly -- render_video_prefix (1 frame) + render_audio_block
// (the new inline audio block) + render_vlm_completion, splicing BOTH image
// and audio soft tokens (TokenRef Image/Audio + embeddings_host) ->
// prefill_multimodal_metal -> greedy decode. Synthetic audio (a tone) so we
// only assert coherent text + that audio_pad slots were all consumed. Gated
// on the GGUF dir (audio needs no external file -- it is synthesised).
TEST(metal_lm_smoke, gemma12b_unified_av_e2e) {
  const char* dir = std::getenv("VPIPE_GEMMA12B_GGUF_PATH");
  const char* imgp = std::getenv("VPIPE_GEMMA12B_TEST_IMAGE");
  if (!dir || !*dir || !imgp || !*imgp) { return; }
  std::FILE* f = std::fopen(imgp, "rb");
  if (!f) { return; }
  char magic[3] = {0};
  int W = 0, H = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) != 4 ||
      std::string(magic) != "P6") { std::fclose(f); return; }
  std::fgetc(f);
  std::vector<std::uint8_t> inter((std::size_t)3 * H * W);
  if (std::fread(inter.data(), 1, inter.size(), f) != inter.size()) {
    std::fclose(f); return;
  }
  std::fclose(f);
  std::vector<std::uint8_t> planar((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        planar[((std::size_t)c * H + y) * W + x] =
            inter[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.compute_dtype = "f16";
  spec.page_tokens = 1024;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  auto* uni = lm->gemma4_unified_embedder();
  ASSERT_TRUE(uni != nullptr && uni->has_vision() && uni->has_audio());

  auto ei = uni->encode_image(planar.data(), H, W);
  ASSERT_TRUE(ei.has_value() && ei->n_tokens > 0);
  const int n_im = ei->n_tokens;
  // 2 s of 440 Hz tone @ 16 kHz -> ceil(32000/640)=50 audio tokens.
  std::vector<float> pcm(32000);
  for (std::size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.2f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f);
  }
  auto ea = uni->encode_audio(pcm.data(), pcm.size());
  ASSERT_TRUE(ea.has_value() && ea->n_tokens > 0);
  const int n_au = ea->n_tokens;

  auto tpl = genai::make_chat_template(lm->config().architecture,
                                     lm->tokenizer());
  ASSERT_TRUE(tpl != nullptr);
  const std::int32_t video_pad = tpl->video_pad_token_id();
  const std::int32_t audio_pad = tpl->audio_pad_token_id();
  ASSERT_TRUE(audio_pad >= 0);

  const float fts[1] = {0.0f};
  const int counts[1] = {n_im};
  std::vector<std::int32_t> ids;
  ASSERT_TRUE(tpl->render_video_prefix(std::span<const float>(fts, 1),
                                       std::span<const int>(counts, 1),
                                       /*is_first_turn=*/true,
                                       std::string_view(), &ids));
  ASSERT_TRUE(tpl->render_audio_block(
      "Audio captured during this scene (<0.0 seconds> to <2.0 seconds>):\n",
      n_au, &ids));
  ASSERT_TRUE(tpl->render_vlm_completion("Describe the scene.", &ids));

  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int img_off = 0, aud_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == video_pad && img_off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_host = &ei->rows;
      r.image_token_offset = img_off++;
    } else if (id == audio_pad && aud_off < n_au) {
      r.kind = genai::TokenRef::Kind::AudioTokens;
      r.embeddings_host = &ea->rows;
      r.audio_token_offset = aud_off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  EXPECT_TRUE(img_off == n_im);
  EXPECT_TRUE(aud_off == n_au);   // all audio_pad slots consumed by the block

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs), {});
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen;
  std::int32_t nx = first;
  for (int i = 0; i < 48 && nx >= 0 && !tpl->is_stop_token(nx); ++i) {
    gen.push_back(nx);
    nx = lm->next_token(ctx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.gemma12b_unified_av] img=%d aud=%d tok | "
              "gen(%zu)='%s'\n", n_im, n_au, gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Gemma-4-12B OptiQ (mixed-precision) AUDIO+VIDEO end-to-end: the same unified
// realtime-vqa path (image + synthetic-tone audio spliced as soft tokens ->
// prefill_multimodal_metal -> greedy), but the model is the mixed 4/8-bit
// OptiQ pack whose vision AND audio adaptors ship in optiq_vision.safetensors.
// Confirms has_vision()+has_audio(), encode_audio, and the mixed forward over a
// text+image+audio token stream. Gated on VPIPE_GEMMA12B_OPTIQ_TEST_MODEL_PATH
// + VPIPE_GEMMA12B_TEST_IMAGE.
TEST(metal_lm_smoke, gemma12b_optiq_av_e2e) {
  const char* dir = std::getenv("VPIPE_GEMMA12B_OPTIQ_TEST_MODEL_PATH");
  const char* imgp = std::getenv("VPIPE_GEMMA12B_TEST_IMAGE");
  if (!dir || !*dir || !imgp || !*imgp) { return; }
  std::FILE* f = std::fopen(imgp, "rb");
  if (!f) { return; }
  char magic[3] = {0};
  int W = 0, H = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) != 4 ||
      std::string(magic) != "P6") { std::fclose(f); return; }
  std::fgetc(f);
  std::vector<std::uint8_t> inter((std::size_t)3 * H * W);
  if (std::fread(inter.data(), 1, inter.size(), f) != inter.size()) {
    std::fclose(f); return;
  }
  std::fclose(f);
  std::vector<std::uint8_t> planar((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        planar[((std::size_t)c * H + y) * W + x] =
            inter[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 1024;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  auto* uni = lm->gemma4_unified_embedder();
  ASSERT_TRUE(uni != nullptr && uni->has_vision() && uni->has_audio());

  auto ei = uni->encode_image(planar.data(), H, W);
  ASSERT_TRUE(ei.has_value() && ei->n_tokens > 0);
  const int n_im = ei->n_tokens;
  std::vector<float> pcm(32000);
  for (std::size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.2f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f);
  }
  auto ea = uni->encode_audio(pcm.data(), pcm.size());
  ASSERT_TRUE(ea.has_value() && ea->n_tokens > 0);
  const int n_au = ea->n_tokens;

  auto tpl = genai::make_chat_template(lm->config().architecture,
                                     lm->tokenizer());
  ASSERT_TRUE(tpl != nullptr);
  const std::int32_t video_pad = tpl->video_pad_token_id();
  const std::int32_t audio_pad = tpl->audio_pad_token_id();
  ASSERT_TRUE(audio_pad >= 0);

  const float fts[1] = {0.0f};
  const int counts[1] = {n_im};
  std::vector<std::int32_t> ids;
  ASSERT_TRUE(tpl->render_video_prefix(std::span<const float>(fts, 1),
                                       std::span<const int>(counts, 1),
                                       /*is_first_turn=*/true,
                                       std::string_view(), &ids));
  ASSERT_TRUE(tpl->render_audio_block(
      "Audio captured during this scene (<0.0 seconds> to <2.0 seconds>):\n",
      n_au, &ids));
  ASSERT_TRUE(tpl->render_vlm_completion("Describe the scene.", &ids));

  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int img_off = 0, aud_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == video_pad && img_off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_host = &ei->rows;
      r.image_token_offset = img_off++;
    } else if (id == audio_pad && aud_off < n_au) {
      r.kind = genai::TokenRef::Kind::AudioTokens;
      r.embeddings_host = &ea->rows;
      r.audio_token_offset = aud_off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  EXPECT_TRUE(img_off == n_im);
  EXPECT_TRUE(aud_off == n_au);

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs), {});
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen;
  std::int32_t nx = first;
  for (int i = 0; i < 48 && nx >= 0 && !tpl->is_stop_token(nx); ++i) {
    gen.push_back(nx);
    nx = lm->next_token(ctx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.gemma12b_optiq_av] img=%d aud=%d tok | "
              "gen(%zu)='%s'\n", n_im, n_au, gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// The Qwen3.5-VL vision tower loaded from mmproj-*.gguf (llama.cpp CLIP layout,
// BF16/F32) must produce the SAME image embeddings as the safetensors tower --
// the mmproj just renames the tensors + splits/transposes the patch-embed
// conv, no requant. Same config + image, only the weight source differs, so
// rel-L2 should be f16-rounding tiny. Gated on VPIPE_QWEN35_TEST_MODEL_PATH
// (safetensors-VL dir: config + reference weights) + VPIPE_QWEN_MMPROJ_TEST_PATH
// (the mmproj-BF16.gguf).
TEST(metal_lm_smoke, qwen_gguf_vision_matches_safetensors) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  const char* mmp = std::getenv("VPIPE_QWEN_MMPROJ_TEST_PATH");
  if (!path || !*path || !mmp || !*mmp) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto mcfg = loader.load_config(path);
  if (!mcfg.has_value() || !mcfg->vision.present) { return; }
  auto cfg = genai::MetalQwenVisionEncoder::config_from(*mcfg);
  if (cfg.depth == 0 || cfg.hidden == 0) { return; }

  const int H = 128, W = 128;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        img[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x * 3 + y * 5 + 17 * c) & 0xFF);
      }
    }
  }
  auto enc = [&](const std::string& gg) -> std::vector<float> {
    auto c = cfg;
    c.gguf_mmproj = gg;          // empty = safetensors, set = mmproj gguf
    auto mv = genai::MetalQwenVisionEncoder::load(path, mc, c);
    if (!mv) { return {}; }
    auto r = mv->encode(img.data(), H, W);
    const auto* p = static_cast<const __fp16*>(r.embeddings.contents());
    std::vector<float> out(r.embeddings.byte_size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = (float)p[i]; }
    return out;
  };
  std::vector<float> st = enc("");
  std::vector<float> gg = enc(mmp);
  ASSERT_TRUE(!st.empty() && st.size() == gg.size());
  double sq = 0.0, df = 0.0;
  float mx = 0.0f;
  for (std::size_t i = 0; i < st.size(); ++i) {
    sq += (double)st[i] * st[i];
    const float d = std::fabs(gg[i] - st[i]);
    df += (double)d * d;
    mx = std::max(mx, d);
  }
  const float rms = (float)std::sqrt(sq / st.size());
  const float drms = (float)std::sqrt(df / st.size());
  std::printf("[gguf-vs-st] tokens=%zu max|d|=%.4f rms=%.4f diff=%.4f (%.3f%%)\n",
              st.size(), mx, rms, drms, 100.0f * drms / rms);
  EXPECT_TRUE(drms < 0.02f * rms);
}

// Loader integration: ModelLoader::load on a Qwen3.5 GGUF (whose dir also holds
// a sibling mmproj-*.gguf, projector_type qwen3vl_merger) must detect it and
// populate VisionConfig from the clip.vision.* metadata (present + dims +
// mmproj_path) so the manager builds the metal vision tower. Gated on
// VPIPE_QWEN_GGUF_TEST_MODEL_PATH (the LM .gguf with a sibling mmproj).
TEST(metal_lm_smoke, qwen_gguf_loader_detects_mmproj_vision) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  Session sess;
  genai::ModelLoader loader(&sess);
  auto w = loader.load(gguf);
  if (!w.has_value()) { return; }   // needs a sibling mmproj to assert vision
  const auto& v = w->config.vision;
  if (!v.present) { return; }        // no mmproj in this dir -> nothing to check
  std::printf("[gguf-dir-vision] present=%d depth=%d hidden=%d heads=%d "
              "merge=%d out=%d numpos=%d mmproj='%s'\n",
              v.present, v.depth, v.hidden_size, v.num_heads,
              v.spatial_merge_size, v.out_hidden_size,
              v.num_position_embeddings,
              v.mmproj_path.empty() ? "" : "set");
  EXPECT_TRUE(!v.mmproj_path.empty());
  EXPECT_TRUE(v.depth > 0 && v.hidden_size > 0 && v.num_heads > 0);
  EXPECT_TRUE(v.out_hidden_size == w->config.hidden);
  EXPECT_TRUE(v.spatial_merge_size == 2 && v.temporal_patch_size == 2);
  EXPECT_TRUE(v.num_position_embeddings > 0);
}
