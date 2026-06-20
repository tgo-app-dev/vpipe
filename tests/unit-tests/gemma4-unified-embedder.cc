// gemma4-unified-embedder.cc -- shape/sanity smoke for the gemma-4-12B
// "unified" shallow multimodal embedder (vision + audio). Runs in both
// builds (MLX-free). Gated on VPIPE_GEMMA12B_MMPROJ (path to the
// mmproj-*.gguf) or VPIPE_GEMMA12B_TEST_MODEL_PATH (dir holding it).
// The op-for-op token-exact check vs llama-mtmd-cli is separate.

#include "minitest.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using vpipe::genai::Gemma4UnifiedEmbedder;

namespace {

std::string mmproj_path_() {
  if (const char* p = std::getenv("VPIPE_GEMMA12B_MMPROJ")) {
    if (*p) { return std::string(p); }
  }
  if (const char* d = std::getenv("VPIPE_GEMMA12B_TEST_MODEL_PATH")) {
    if (*d) { return Gemma4UnifiedEmbedder::find_mmproj(d); }
  }
  return std::string();
}

bool all_finite_(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) { return false; }
  }
  return true;
}

}  // namespace

TEST(gemma4_unified_embedder, smart_resize_multiple_of_48) {
  const std::string mp = mmproj_path_();
  if (mp.empty()) { return; }
  auto emb = Gemma4UnifiedEmbedder::load(mp);
  ASSERT_TRUE(emb != nullptr);
  ASSERT_TRUE(emb->has_vision());
  // 336x288 is already a multiple of 48 and 96768 px is in [92160,645120],
  // so smart-resize is identity.
  int th = 0, tw = 0;
  emb->smart_resize(288, 336, &th, &tw);
  EXPECT_TRUE(th == 288);
  EXPECT_TRUE(tw == 336);
  // A tiny image is grown to >= min pixels; a huge one shrunk to <= max.
  int th2 = 0, tw2 = 0;
  emb->smart_resize(32, 32, &th2, &tw2);
  EXPECT_TRUE(th2 * tw2 >= 40 * 48 * 48);
  EXPECT_TRUE(th2 % 48 == 0 && tw2 % 48 == 0);
  int th3 = 0, tw3 = 0;
  emb->smart_resize(4000, 4000, &th3, &tw3);
  EXPECT_TRUE(th3 * tw3 <= 280 * 48 * 48);
  EXPECT_TRUE(th3 % 48 == 0 && tw3 % 48 == 0);
}

TEST(gemma4_unified_embedder, vision_shapes_and_finite) {
  const std::string mp = mmproj_path_();
  if (mp.empty()) { return; }
  auto emb = Gemma4UnifiedEmbedder::load(mp);
  ASSERT_TRUE(emb != nullptr);
  ASSERT_TRUE(emb->has_vision());
  const int H = 288, W = 336;             // identity resize -> 7x6 = 42 patches
  std::vector<std::uint8_t> rgb((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        rgb[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x * 3 + y * 5 + c * 37) & 0xff);
      }
    }
  }
  auto r = emb->encode_image(rgb.data(), H, W);
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r->grid_w == 7 && r->grid_h == 6);
  EXPECT_TRUE(r->n_tokens == 42);
  EXPECT_TRUE((int)r->rows.size() == r->n_tokens * emb->out_hidden());
  EXPECT_TRUE(all_finite_(r->rows));
  // Not degenerate (the projection produces non-zero rows).
  double mag = 0.0;
  for (float v : r->rows) { mag += (double)v * v; }
  EXPECT_TRUE(mag > 0.0);
  // Deterministic.
  auto r2 = emb->encode_image(rgb.data(), H, W);
  ASSERT_TRUE(r2.has_value());
  bool same = r2->rows.size() == r->rows.size();
  for (std::size_t i = 0; same && i < r->rows.size(); ++i) {
    if (r->rows[i] != r2->rows[i]) { same = false; }
  }
  EXPECT_TRUE(same);
  // Dump token-0 head/tail + stats to compare op-for-op vs the numpy
  // reference (same synthetic pixels + the documented gemma4uv graph).
  const int D = emb->out_hidden();
  double mn = r->rows[0], mx = r->rows[0], sum = 0.0;
  for (float v : r->rows) { mn = std::min(mn, (double)v);
    mx = std::max(mx, (double)v); sum += v; }
  std::printf("[g4u_embed] vision %d tokens (%dx%d)\n",
              r->n_tokens, r->grid_w, r->grid_h);
  std::printf("[g4u_embed] tok0 first8:");
  for (int i = 0; i < 8; ++i) { std::printf(" %.5f", r->rows[i]); }
  std::printf("\n[g4u_embed] tok0 last8: ");
  for (int i = D - 8; i < D; ++i) { std::printf(" %.5f", r->rows[i]); }
  std::printf("\n[g4u_embed] stats sumsq=%.4e mean=%.6f min=%.5f max=%.5f\n",
              mag, sum / r->rows.size(), mn, mx);
  // GOLDEN: op-for-op vs a numpy reference of the documented gemma4uv graph
  // on these exact pixels (see commit msg). Tolerance covers f32 accum order;
  // a wrong graph (norm order / channel order / pos index) is off by O(1)+.
  const float gold0[8] = {1.11591f, 0.06131f, 15.19607f, -0.51836f,
                          0.40112f, -0.09721f, -0.43739f, -0.59640f};
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::abs(r->rows[i] - gold0[i]) < 1e-2f);
  }
  EXPECT_TRUE(std::abs(mag - 8.4542e5) < 5e2);
}

TEST(gemma4_unified_embedder, audio_shapes_and_finite) {
  const std::string mp = mmproj_path_();
  if (mp.empty()) { return; }
  auto emb = Gemma4UnifiedEmbedder::load(mp);
  ASSERT_TRUE(emb != nullptr);
  if (!emb->has_audio()) { return; }
  const std::size_t n = 16000;            // 1 s @16k -> ceil(16000/640)=25 tok
  std::vector<float> pcm(n);
  for (std::size_t i = 0; i < n; ++i) {
    pcm[i] = 0.1f * std::sin(0.02f * (float)i);
  }
  auto r = emb->encode_audio(pcm.data(), n);
  ASSERT_TRUE(r.has_value());
  EXPECT_TRUE(r->n_tokens == 25);
  EXPECT_TRUE((int)r->rows.size() == r->n_tokens * emb->out_hidden());
  EXPECT_TRUE(all_finite_(r->rows));
  double mag = 0.0;
  for (float v : r->rows) { mag += (double)v * v; }
  EXPECT_TRUE(mag > 0.0);
  std::printf("[g4u_embed] audio %d tokens, |x|^2=%.3e tok0:", r->n_tokens,
              mag);
  for (int i = 0; i < 8; ++i) { std::printf(" %.5f", r->rows[i]); }
  std::printf("\n");
  // GOLDEN vs numpy reference of the gemma4ua graph (RMSNorm + projection).
  const float gold0[8] = {-0.03866f, -0.03015f, 3.49732f, 0.04826f,
                          -1.18512f, 0.00429f, 0.06641f, 0.05592f};
  for (int i = 0; i < 8; ++i) {
    EXPECT_TRUE(std::abs(r->rows[i] - gold0[i]) < 1e-2f);
  }
  EXPECT_TRUE(std::abs(mag - 2.1914e5) < 2e2);
}
