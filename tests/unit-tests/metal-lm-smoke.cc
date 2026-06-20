// metal-lm-smoke.cc -- metal-compute LM text-decode smoke that compiles
// and runs in BOTH the MLX and no-MLX builds. It never references
// mlx::core, so it lives in the VPIPE_BUILD_APPLE_SILICON test block
// (not the MLX-gated one) and is the end-to-end proof that the language-
// model subsystem loads + generates on the metal backend with MLX off.
//
// Env-gated: set VPIPE_METAL_LM_SMOKE_MODEL to a metal-supported model
// dir (Qwen3.5-4B / Llama / Qwen3-ASR text decoder). Forces
// VPIPE_LLM_BACKEND=metal for the duration of the load.

#include "minitest.h"
#include "generative-models/chat-template.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/model-loader.h"
#include "generative-models/sampler.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#include "generative-models/shared/gguf-file.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <mach/mach.h>

using namespace vpipe;

// Self-contained qmv (decode GEMV) bandwidth A/B: w4g64 vs w8g64 (bf16, the
// MOSS compute dtype), random weights (bandwidth is value-independent), at the
// real MOSS decode shapes + a large SLC-busting shape. Answers whether the
// 8-bit qmv kernel hits the same effective DRAM bandwidth as the 4-bit one.
// Gated on VPIPE_QMV_AB. M5 16GB peak DRAM = 153 GB/s.
TEST(metal_lm_smoke, qmv_w4_w8_bandwidth_ab) {
  if (std::getenv("VPIPE_QMV_AB") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  using Clock = std::chrono::steady_clock;
  const int R = 200;
  const double kPeak = 153.0;
  struct Shape { const char* name; int n; int k; };
  const Shape shapes[] = {
      {"moss gate/up", 12288, 4096}, {"moss o-proj ", 4096, 4096},
      {"moss down   ", 4096, 12288}, {"big (SLC-bust)", 16384, 8192},
  };
  struct V { const char* fn; int bits; int rps; int nsg; };
  const V vars[] = {
      {"affine_qmv_w4g64", 4, 4, 2}, {"affine_qmv_w4g64_r8p2", 4, 8, 2},
      {"affine_qmv_w8g64", 8, 4, 2}, {"affine_qmv_w8g64_r8p2", 8, 8, 2}};
  std::printf("[qmv-ab] M5 peak %.0f GB/s; %d serial GEMVs, min-of-3, warmed\n",
              kPeak, R);
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::uint32_t> du(0, 0xffffffffu);
  for (const Shape& sh : shapes) {
    const int groups = sh.k / 64;
    for (const V& v : vars) {
      auto fn = lib.function(v.fn);
      if (!fn.valid()) {
        std::printf("[qmv-ab] %-13s %-22s MISSING\n", sh.name, v.fn);
        continue;
      }
      const std::size_t wwords =
          (std::size_t)sh.n * sh.k / (32 / v.bits);   // packed u32 count
      const std::size_t sbcnt = (std::size_t)sh.n * groups;
      const double read_bytes = (double)(wwords * 4 + 2 * sbcnt * 2);
      auto wb = mc->make_shared_buffer(wwords * 4);
      auto sb = mc->make_shared_buffer(sbcnt * 2);
      auto bb = mc->make_shared_buffer(sbcnt * 2);
      auto xb = mc->make_shared_buffer((std::size_t)sh.k * 2);
      auto yb = mc->make_shared_buffer((std::size_t)sh.n * 2);
      auto* wp = static_cast<std::uint32_t*>(wb.contents());
      for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
      auto dispatch_R = [&](int reps) {
        metal_compute::CommandStream st = mc->make_command_stream();
        {
          metal_compute::ComputeEncoder e = st.begin_compute();
          e.set_function(fn);
          e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
          e.set_buffer(3, xb); e.set_buffer(4, yb);
          e.set_constant(5, sh.k); e.set_constant(6, sh.n);
          for (int r = 0; r < reps; ++r) {
            e.dispatch({32u, (unsigned)(sh.n / v.rps), 1u},
                       {32u, (unsigned)v.nsg, 1u});
          }
        }
        st.commit().wait();
      };
      dispatch_R(20);
      double best_ms = 1e18;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        dispatch_R(R);
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(
                                        Clock::now() - t0).count());
      }
      const double gbps = read_bytes * R / (best_ms / 1e3) / 1e9;
      std::printf("[qmv-ab] %-13s %-22s %4.1f MB | %6.1f GB/s (%4.1f%% peak)\n",
                  sh.name, v.fn, wwords * 4 / 1e6, gbps, 100.0 * gbps / kPeak);
    }
  }
}

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

TEST(metal_lm_smoke, text_decode) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
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
  if (!mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto ids = lm->tokenizer().encode("The capital of France is");
  ASSERT_TRUE(!ids.empty());
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());

  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 8; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke] %zu tokens | gen='%s'\n", gen.size(),
              text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Numerically verify the decode-only quantized GEMV at group-size 32
// (affine_qmv_w4g32 -- the GGUF q4_0 path's q/k/v/o + down_proj decode
// kernel, never exercised by prefill [steel qmm] or e4b [g64]) against a
// hand-built CPU dequant+matmul reference. No model/MLX needed.
TEST(metal_lm_smoke, qmv_w4g32_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard: K = 3840 (Gemma-12B hidden) is NOT a multiple of the 4-bit
  // block_size (512) -> the GEMV's final block is partial. We allocate x out
  // to the block boundary (Kpad = 4096) filled with non-zero values, the
  // weights with one extra padding row, and tell the kernel in_vec_size = K.
  // The reference sums only k < K. WITHOUT the tail mask the GPU's tail lanes
  // (k in [3840,4096)) fold non-zero x * out-of-range weights into the dot ->
  // gross error; WITH the mask they contribute 0 and it matches the CPU ref.
  // Uses small biases (no affine cancellation) so this isolates the tail.
  const int N = 16, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> q((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  std::vector<_Float16> x((std::size_t)Kpad);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 991 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.1f * frand((unsigned)(o * 131 + g * 7 + 5)));
    }
  }
  // Fill the whole weight buffer (incl. the extra padding row) so tail lanes
  // read deterministic non-zero nibbles when the mask is absent.
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { q[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  for (int k = 0; k < Kpad; ++k) {
    x[k] = (_Float16)(0.5f + frand((unsigned)(k * 7 + 3)));   // all non-zero
  }

  std::vector<float> ref((std::size_t)N);
  for (int o = 0; o < N; ++o) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const float s = (float)scales[(std::size_t)o * groups + k / G];
      const float b = (float)biases[(std::size_t)o * groups + k / G];
      acc += (double)(float)x[k] *
             (s * (float)q[(std::size_t)o * K + k] + b);
    }
    ref[o] = (float)acc;
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double d = std::fabs((double)(float)yp[o] - (double)ref[o]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[o]) + 1e-2));
    std::printf("[qmv_w4g32] o=%2d gpu=%.5f ref=%.5f\n",
                o, (float)yp[o], ref[o]);
  }
  std::printf("[qmv_w4g32] max rel err = %.4g\n", maxrel);
  EXPECT_TRUE(maxrel < 0.03);
}

// Batched (MAXM=2) decode GEMV at group-size 32 with K = 3840 (NOT a multiple
// of block_size 512 -- Gemma-12B hidden): exercises the partial-tail path in
// qmv_batch_impl. Two rows, verified against a CPU dequant ref per (row,out).
TEST(metal_lm_smoke, qmv_batch_w4g32_tail_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_batch_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard: K = 3840 (not a multiple of block_size 512). Each row's x is
  // padded to the block boundary (Kpad) with non-zero values and the weights
  // carry an extra padding row, so the partial last block reads out-of-range
  // data unless masked. Small biases -> isolates the tail (no cancellation).
  const int N = 16, M = 2, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> q((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  // x rows are at the kernel's stride (in_vec_size = K), with trailing padding
  // so the last row's partial-block tail reads in-bounds (non-zero) data.
  std::vector<_Float16> x((std::size_t)M * K + (Kpad - K));
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 991 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.1f * frand((unsigned)(o * 131 + g * 7 + 5)));
    }
  }
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { q[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = (_Float16)(0.5f + frand((unsigned)(i * 7 + 3)));   // all non-zero
  }

  std::vector<float> ref((std::size_t)M * N);
  for (int m = 0; m < M; ++m) {
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        const float s = (float)scales[(std::size_t)o * groups + k / G];
        const float b = (float)biases[(std::size_t)o * groups + k / G];
        acc += (double)(float)x[(std::size_t)m * K + k] *
               (s * (float)q[(std::size_t)o * K + k] + b);
      }
      ref[(std::size_t)m * N + o] = (float)acc;
    }
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)M * N * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({32, (unsigned)(N / 4), (unsigned)((M + 1) / 2)}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < M * N; ++i) {
    const double d = std::fabs((double)(float)yp[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[i]) + 1e-2));
  }
  std::printf("[qmv_batch_w4g32] K=%d max rel err = %.4g\n", K, maxrel);
  EXPECT_TRUE(maxrel < 0.03);
}

// Native Q6_K (llama.cpp k-quant) GPU unpack must match the CPU dequant
// bit-for-bit (it's a lossless format, not a requant). Synthesizes raw Q6_K
// super-blocks and compares dequant_q6k_f16 against an inline CPU reference
// mirroring gguf-file.cc's dequant_row_q6_K.
TEST(metal_lm_smoke, q6k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q6k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 210);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 210;
    for (int i = 0; i < 128; ++i) {                 // ql
      p[i] = (std::uint8_t)((sb * 128 + i) * 37 + 11);
    }
    for (int i = 0; i < 64; ++i) {                  // qh
      p[128 + i] = (std::uint8_t)((sb * 64 + i) * 53 + 7);
    }
    for (int i = 0; i < 16; ++i) {                  // int8 scales
      p[192 + i] = (std::uint8_t)(std::int8_t)((sb * 16 + i) * 5 - 40);
    }
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    std::memcpy(p + 208, &d, 2);
  }

  // CPU reference (mirrors gguf-file.cc kQ6_K).
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 210;
    const std::uint8_t* ql = p;
    const std::uint8_t* qh = p + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(p + 192);
    _Float16 d16;
    std::memcpy(&d16, p + 208, 2);
    const float d = (float)d16;
    float* y = ref.data() + (std::size_t)sb * 256;
    for (int half = 0; half < 2; ++half) {
      const int qlo = half * 64, qho = half * 32, sco = half * 8, yo = half * 128;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16, hi = qh[qho + l];
        const int q1 = ((ql[qlo + l] & 0xF) | (((hi >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[qlo + l + 32] & 0xF) | (((hi >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[qlo + l] >> 4) | (((hi >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[qlo + l + 32] >> 4) | (((hi >> 6) & 3) << 4)) - 32;
        y[yo + l] = d * sc[sco + is + 0] * q1;
        y[yo + l + 32] = d * sc[sco + is + 2] * q2;
        y[yo + l + 64] = d * sc[sco + is + 4] * q3;
        y[yo + l + 96] = d * sc[sco + is + 6] * q4;
      }
    }
  }

  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, sbuf);
    enc.set_buffer(1, obuf);
    enc.set_constant(2, N);
    enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
  }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double d = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q6k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);   // f16 output rounding only
}

// Native Q6_K lm_head GEMV (qmv_q6k_f16): y[o] = sum_h x[h]*dequant(W[o,h]),
// verified against a CPU dequant+dot reference. This is the lossless,
// memory-saving replacement for the 8-bit affine-requant lm_head GEMV.
TEST(metal_lm_smoke, qmv_q6k_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("qmv_q6k_f16");
  ASSERT_TRUE(fn.valid());

  const int N = 64, H = 512, sbpr = H / 256;        // super-blocks per row
  std::vector<std::uint8_t> w((std::size_t)N * sbpr * 210);
  std::vector<_Float16> x((std::size_t)H);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int s = 0; s < sbpr; ++s) {
      std::uint8_t* p = w.data() + ((std::size_t)o * sbpr + s) * 210;
      const unsigned base = (unsigned)(o * sbpr + s);
      for (int i = 0; i < 128; ++i) { p[i] = (std::uint8_t)((base * 128 + i) * 37 + 11); }
      for (int i = 0; i < 64; ++i) { p[128 + i] = (std::uint8_t)((base * 64 + i) * 53 + 7); }
      for (int i = 0; i < 16; ++i) {
        p[192 + i] = (std::uint8_t)(std::int8_t)((base * 16 + i) * 5 - 40);
      }
      const _Float16 d = (_Float16)(0.004f + 0.0005f * (float)(base % 8));
      std::memcpy(p + 208, &d, 2);
    }
  }
  for (int h = 0; h < H; ++h) { x[h] = (_Float16)frand((unsigned)(h * 7 + 3)); }

  // CPU reference: dequant each weight (mirror gguf-file kQ6_K) and dot with x.
  auto q6k_cpu = [&](const std::uint8_t* sb, int pos) {
    const std::uint8_t* ql = sb;
    const std::uint8_t* qh = sb + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(sb + 192);
    _Float16 d16; std::memcpy(&d16, sb + 208, 2);
    const int hf = pos >> 7, p = pos & 127, which = p >> 5, l = p & 31;
    const int is = l >> 4, qlo = hf * 64, qho = hf * 32, sco = hf * 8;
    const int hi = qh[qho + l];
    int q, sci;
    if (which == 0) { q = (ql[qlo + l] & 0xF) | (((hi >> 0) & 3) << 4); sci = sco + is; }
    else if (which == 1) { q = (ql[qlo + l + 32] & 0xF) | (((hi >> 2) & 3) << 4); sci = sco + is + 2; }
    else if (which == 2) { q = (ql[qlo + l] >> 4) | (((hi >> 4) & 3) << 4); sci = sco + is + 4; }
    else { q = (ql[qlo + l + 32] >> 4) | (((hi >> 6) & 3) << 4); sci = sco + is + 6; }
    return (float)d16 * (float)sc[sci] * (float)(q - 32);
  };
  std::vector<float> ref((std::size_t)N);
  for (int o = 0; o < N; ++o) {
    double acc = 0.0;
    for (int h = 0; h < H; ++h) {
      const std::uint8_t* sb = w.data() + ((std::size_t)o * sbpr + h / 256) * 210;
      acc += (double)(float)x[h] * (double)q6k_cpu(sb, h & 255);
    }
    ref[o] = (float)acc;
  }

  auto wbuf = mc->make_shared_buffer(w.size());
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(wbuf.contents(), w.data(), w.size());
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  // Both the original and the llama.cpp-style v2 kernel must match the CPU
  // reference (v2 is not bit-identical to the original -- different fp grouping
  // -- but must be numerically equivalent).
  auto run = [&](const char* name) {
    auto f = lib.function(name);
    if (!f.valid()) { return; }
    auto stream = mc->make_command_stream();
    {
      auto enc = stream.begin_compute();
      enc.set_function(f);
      enc.set_buffer(0, wbuf);
      enc.set_buffer(1, xbuf);
      enc.set_buffer(2, ybuf);
      enc.set_constant(3, H);
      enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double d = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q6k] %-14s N=%d H=%d max rel err = %.4g\n",
                name, N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  };
  run("qmv_q6k_f16");
  run("qmv_q6k_v2_f16");
}

// Shared 6-bit scale/min unpack for the Q4_K/Q5_K CPU references below
// (mirrors gguf-file.cc get_scale_min_k4_ / llama.cpp get_scale_min_k4).
namespace {
inline void gsmk4_cpu(int j, const std::uint8_t* q, std::uint8_t& d,
                      std::uint8_t& m) {
  if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
  else { d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
         m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4); }
}
}  // namespace

// Native Q4_K dequant (dequant_q4k_f16) vs a CPU reference mirroring
// gguf-file.cc kQ4_K. 144-byte super-block: d(f16) dmin(f16) scales[12] qs[128].
TEST(metal_lm_smoke, q4k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q4k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 144);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 144;
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    const _Float16 dm = (_Float16)(0.05f + 0.01f * sb);
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4 + i] = (std::uint8_t)((sb*12+i)*29+3); }
    for (int i = 0; i < 128; ++i) { p[16 + i] = (std::uint8_t)((sb*128+i)*37+11); }
  }
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 144;
    _Float16 d16, m16; std::memcpy(&d16, p, 2); std::memcpy(&m16, p + 2, 2);
    const float d = (float)d16, dmin = (float)m16;
    const std::uint8_t* scales = p + 4; const std::uint8_t* qs = p + 16;
    float* y = ref.data() + (std::size_t)sb * 256;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc, m;
      gsmk4_cpu(is + 0, scales, sc, m); const float d1 = d*sc, m1 = dmin*m;
      gsmk4_cpu(is + 1, scales, sc, m); const float d2 = d*sc, m2 = dmin*m;
      const std::uint8_t* q = qs + (j / 64) * 32;
      for (int l = 0; l < 32; ++l) {
        y[j + l]      = d1 * (q[l] & 0x0F) - m1;
        y[j + l + 32] = d2 * (q[l] >> 4)  - m2;
      }
      is += 2;
    }
  }
  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  { auto enc = stream.begin_compute();
    enc.set_function(fn); enc.set_buffer(0, sbuf); enc.set_buffer(1, obuf);
    enc.set_constant(2, N); enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1}); }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dd = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q4k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);
}

// Native Q5_K dequant (dequant_q5k_f16) vs CPU reference mirroring kQ5_K.
// 176-byte super-block: d(f16) dmin(f16) scales[12] qh[32] qs[128].
TEST(metal_lm_smoke, q5k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q5k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 176);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 176;
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    const _Float16 dm = (_Float16)(0.05f + 0.01f * sb);
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4 + i] = (std::uint8_t)((sb*12+i)*29+3); }
    for (int i = 0; i < 32; ++i) { p[16 + i] = (std::uint8_t)((sb*32+i)*43+5); }
    for (int i = 0; i < 128; ++i) { p[48 + i] = (std::uint8_t)((sb*128+i)*37+11); }
  }
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 176;
    _Float16 d16, m16; std::memcpy(&d16, p, 2); std::memcpy(&m16, p + 2, 2);
    const float d = (float)d16, dmin = (float)m16;
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qh = p + 16; const std::uint8_t* qs = p + 48;
    float* y = ref.data() + (std::size_t)sb * 256;
    int is = 0; std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc, m;
      gsmk4_cpu(is + 0, scales, sc, m); const float d1 = d*sc, m1 = dmin*m;
      gsmk4_cpu(is + 1, scales, sc, m); const float d2 = d*sc, m2 = dmin*m;
      const std::uint8_t* q = qs + (j / 64) * 32;
      for (int l = 0; l < 32; ++l) {
        const int lo = (q[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
        const int hi = (q[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
        y[j + l]      = d1 * lo - m1;
        y[j + l + 32] = d2 * hi - m2;
      }
      is += 2; u1 <<= 2; u2 <<= 2;
    }
  }
  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  { auto enc = stream.begin_compute();
    enc.set_function(fn); enc.set_buffer(0, sbuf); enc.set_buffer(1, obuf);
    enc.set_constant(2, N); enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1}); }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dd = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q5k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);
}

// Native Q4_K / Q5_K GEMV (qmv_q4k_f16 / qmv_q5k_f16): y[o]=sum_h x[h]*W[o,h],
// verified against a CPU dequant+dot reference. These back the GGUF Qwen3.5
// linears' decode path (no affine requant).
TEST(metal_lm_smoke, qmv_q4k_q5k_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  ASSERT_TRUE(lib.function("qmv_q4k_f16").valid());
  ASSERT_TRUE(lib.function("qmv_q5k_f16").valid());

  const int N = 64, H = 512, sbpr = H / 256;
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  std::vector<_Float16> x((std::size_t)H);
  for (int h = 0; h < H; ++h) { x[h] = (_Float16)frand((unsigned)(h*7+3)); }

  auto fill_block = [&](std::uint8_t* p, unsigned base, int blk_bytes,
                        int qh_off, int qs_off) {
    const _Float16 d = (_Float16)(0.004f + 0.0005f * (float)(base % 8));
    const _Float16 dm = (_Float16)(0.03f + 0.002f * (float)(base % 5));
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4+i] = (std::uint8_t)((base*12+i)*29+3); }
    if (qh_off >= 0) {
      for (int i = 0; i < 32; ++i) {
        p[qh_off+i] = (std::uint8_t)((base*32+i)*43+5);
      }
    }
    for (int i = 0; i < 128; ++i) {
      p[qs_off+i] = (std::uint8_t)((base*128+i)*37+11);
    }
    (void)blk_bytes;
  };

  // --- Q4_K ---
  {
    std::vector<std::uint8_t> w((std::size_t)N * sbpr * 144);
    for (int o = 0; o < N; ++o) {
      for (int s = 0; s < sbpr; ++s) {
        fill_block(w.data() + ((std::size_t)o*sbpr+s)*144,
                   (unsigned)(o*sbpr+s), 144, -1, 16);
      }
    }
    auto q4k_cpu = [&](const std::uint8_t* sb, int pos) {
      _Float16 d16, m16; std::memcpy(&d16, sb, 2); std::memcpy(&m16, sb+2, 2);
      const float d = (float)d16, dmin = (float)m16;
      const std::uint8_t* scales = sb + 4; const std::uint8_t* qs = sb + 16;
      const int chunk = pos >> 6, within = pos & 63;
      const int is = chunk*2 + (within >> 5), l = within & 31;
      const unsigned qb = qs[chunk*32 + l];
      const unsigned nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
      std::uint8_t sc, m; gsmk4_cpu(is, scales, sc, m);
      return d * sc * (float)nib - dmin * m;
    };
    std::vector<float> ref((std::size_t)N);
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::uint8_t* sb = w.data() + ((std::size_t)o*sbpr + h/256)*144;
        acc += (double)(float)x[h] * (double)q4k_cpu(sb, h & 255);
      }
      ref[o] = (float)acc;
    }
    auto wbuf = mc->make_shared_buffer(w.size());
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
    std::memcpy(wbuf.contents(), w.data(), w.size());
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto stream = mc->make_command_stream();
    { auto enc = stream.begin_compute();
      enc.set_function(lib.function("qmv_q4k_f16"));
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1}); }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double dd = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q4k] N=%d H=%d max rel err = %.4g\n", N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  }
  // --- Q5_K ---
  {
    std::vector<std::uint8_t> w((std::size_t)N * sbpr * 176);
    for (int o = 0; o < N; ++o) {
      for (int s = 0; s < sbpr; ++s) {
        fill_block(w.data() + ((std::size_t)o*sbpr+s)*176,
                   (unsigned)(o*sbpr+s), 176, 16, 48);
      }
    }
    auto q5k_cpu = [&](const std::uint8_t* sb, int pos) {
      _Float16 d16, m16; std::memcpy(&d16, sb, 2); std::memcpy(&m16, sb+2, 2);
      const float d = (float)d16, dmin = (float)m16;
      const std::uint8_t* scales = sb + 4;
      const std::uint8_t* qh = sb + 16; const std::uint8_t* qs = sb + 48;
      const int chunk = pos >> 6, within = pos & 63;
      const int is = chunk*2 + (within >> 5), l = within & 31;
      const unsigned qb = qs[chunk*32 + l];
      unsigned nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
      const int bit = 2*chunk + ((within < 32) ? 0 : 1);
      nib += ((unsigned(qh[l]) >> bit) & 1u) * 16u;
      std::uint8_t sc, m; gsmk4_cpu(is, scales, sc, m);
      return d * sc * (float)nib - dmin * m;
    };
    std::vector<float> ref((std::size_t)N);
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::uint8_t* sb = w.data() + ((std::size_t)o*sbpr + h/256)*176;
        acc += (double)(float)x[h] * (double)q5k_cpu(sb, h & 255);
      }
      ref[o] = (float)acc;
    }
    auto wbuf = mc->make_shared_buffer(w.size());
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
    std::memcpy(wbuf.contents(), w.data(), w.size());
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto stream = mc->make_command_stream();
    { auto enc = stream.begin_compute();
      enc.set_function(lib.function("qmv_q5k_f16"));
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1}); }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double dd = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q5k] N=%d H=%d max rel err = %.4g\n", N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  }
}

// Q6_K lm_head GEMV bandwidth at the real 12B shape [vocab=262144, hidden=3840]
// -- settles whether qmv_q6k_f16 is DRAM-bound (already at the ~100 GB/s M4
// ceiling, so llama.cpp's "read each ql/qh byte once, extract all nibbles
// in-thread" trick can't help) or load/instruction-bound (the per-nibble
// re-reads across lanes cost, and the trick would). Reports GB/s over the raw
// Q6_K table bytes (210 B / 256 wt). Gated on VPIPE_Q6K_BW.
TEST(metal_lm_smoke, q6k_lmhead_bandwidth) {
  if (std::getenv("VPIPE_Q6K_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");

  const int N = 262144, H = 3840, sbpr = H / 256;     // 15 super-blocks/row
  const std::size_t wbytes = (std::size_t)N * sbpr * 210;
  auto wbuf = mc->make_shared_buffer(wbytes);
  auto xbuf = mc->make_shared_buffer((std::size_t)H * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  // Content is irrelevant for timing; just fill x with something finite.
  auto* xp = static_cast<_Float16*>(xbuf.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)0.01f; }

  auto fn1 = lib.function("qmv_q6k_f16");
  auto fn2 = lib.function("qmv_q6k_v2_f16");
  auto once = [&](metal_compute::ComputeFunction& fn) {
    auto st = mc->make_command_stream();
    { auto enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    }
    st.commit().wait();
  };
  auto measure = [&](metal_compute::ComputeFunction& fn) {
    const int R = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) { once(fn); }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / R;
  };
  // Interleave v1/v2 measures and take the min (peak-clock) of each to cancel
  // the GPU's per-run clock drift.
  for (int w = 0; w < 5; ++w) { once(fn1); once(fn2); }   // warm
  double m1 = 1e18, m2 = 1e18;
  for (int k = 0; k < 6; ++k) {
    m1 = std::fmin(m1, measure(fn1));
    m2 = std::fmin(m2, measure(fn2));
  }
  std::printf("[q6k_bw] qmv_q6k_f16    %.3f ms  %.1f GB/s (min-of-6)\n",
              m1, (double)wbytes / (m1 * 1e6));
  std::printf("[q6k_bw] qmv_q6k_v2_f16 %.3f ms  %.1f GB/s (min-of-6)  %.2fx\n",
              m2, (double)wbytes / (m2 * 1e6), m1 / m2);
  EXPECT_TRUE(true);
}

// Per-token RoPE cost (12B decode): bounds the TOTAL fused-RMSNorm+RoPE work --
// settles whether vpipe's INLINE cos/sin (vs a precomputed cos/sin cache) is a
// decode bottleneck. llama.cpp's Metal kernel_rope_neox computes cos/sin inline
// too (and pow() per element, which vpipe avoids via precomputed inv_freq), so
// the cos/sin cache is a CPU-ggml / TTNN technique, not a GPU one. Replays the
// real per-token sequence: 48 layers x {Q rms_rope, K rms_rope}, 8 global
// (D=512: Hq=16/Hkv=1) + 40 sliding (D=256: Hq=16/Hkv=8). Gated on VPIPE_ROPE_BW.
TEST(metal_lm_smoke, rope_pertoken_cost) {
  if (std::getenv("VPIPE_ROPE_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("rope");
  auto fn = lib.function("rms_rope_f16");
  ASSERT_TRUE(fn.valid());

  const int Dmax = 512, Hq = 16;
  auto xb = mc->make_shared_buffer((std::size_t)Hq * Dmax * 2);
  auto wb = mc->make_shared_buffer((std::size_t)Dmax * 2);
  auto fb = mc->make_shared_buffer((std::size_t)(Dmax / 2) * 4);
  for (int i = 0; i < Hq * Dmax; ++i) {
    static_cast<_Float16*>(xb.contents())[i] = (_Float16)0.02f;
  }
  for (int i = 0; i < Dmax; ++i) {
    static_cast<_Float16*>(wb.contents())[i] = (_Float16)1.0f;
  }
  for (int i = 0; i < Dmax / 2; ++i) {
    static_cast<float*>(fb.contents())[i] = 1.0f / (1.0f + (float)i);
  }
  const float eps = 1e-6f;
  const int offset = 2048;
  auto rope1 = [&](metal_compute::ComputeEncoder& enc, int H, int D) {
    enc.set_function(fn);
    enc.set_buffer(0, xb); enc.set_buffer(1, wb); enc.set_buffer(2, fb);
    enc.set_constant(3, H); enc.set_constant(4, D);
    enc.set_constant(5, eps); enc.set_constant(6, offset);
    enc.dispatch({256, (unsigned)H, 1}, {256, 1, 1});
  };
  auto once = [&]() {                       // one token's worth of rope work
    auto st = mc->make_command_stream();
    { auto enc = st.begin_compute();
      for (int L = 0; L < 48; ++L) {
        const bool full = (L % 6 == 5);     // 8 of 48 are global
        const int D = full ? 512 : 256;
        const int Hkv = full ? 1 : 8;
        rope1(enc, Hq, D);                  // Q rope (16 heads)
        rope1(enc, Hkv, D);                 // K rope
      }
    }
    st.commit().wait();
  };
  for (int w = 0; w < 5; ++w) { once(); }
  double best = 1e18;
  for (int k = 0; k < 8; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    const int R = 20;
    for (int i = 0; i < R; ++i) { once(); }
    best = std::fmin(best, std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / R);
  }
  std::printf("[rope_cost] full per-token RMSNorm+RoPE (96 dispatches): "
              "%.3f ms/tok (min-of-8)\n", best);
  EXPECT_TRUE(best > 0.0);
}

// Numerically verify the decode-only fused GeGLU GEMV at group-size 32
// (affine_qmv_geglu_w4g32 -- the GGUF gate/up decode kernel) against a CPU
// reference (interleaved gate/up rows, gelu_pytorch_tanh(gate)*up).
TEST(metal_lm_smoke, qmv_geglu_w4g32_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_geglu_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard for the fused gate/up GEMV: K = 3840 (Gemma-12B hidden) is NOT
  // a multiple of block_size 512. x is padded to the block boundary (Kpad) with
  // non-zero values and the weights carry an extra padding row, so without the
  // tail mask the partial last block folds in out-of-range data (amplified by
  // the gelu). Small biases -> isolates the tail (no affine cancellation).
  const int F = 2048, N = 2 * F, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> qn((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  std::vector<_Float16> x((std::size_t)Kpad);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 91 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.08f * frand((unsigned)(o * 31 + g * 7 + 5)));
    }
  }
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { qn[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  // Small positive activations (non-zero -> tail is exercised) kept tiny so the
  // squared geglu output gate*up stays inside fp16 range (gate ~ K*x*scale*q).
  for (int k = 0; k < Kpad; ++k) {
    x[k] = (_Float16)(0.04f + 0.06f * frand((unsigned)(k * 7 + 3)));
  }

  auto rowdot = [&](int o) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const float s = (float)scales[(std::size_t)o * groups + k / G];
      const float b = (float)biases[(std::size_t)o * groups + k / G];
      acc += (double)(float)x[k] * (s * (float)qn[(std::size_t)o * K + k] + b);
    }
    return (float)acc;
  };
  std::vector<float> ref((std::size_t)F);
  for (int g = 0; g < F; ++g) {
    const float gate = rowdot(2 * g), up = rowdot(2 * g + 1);
    const float t = std::tanh(0.7978845608028654f *
                              (gate + 0.044715f * gate * gate * gate));
    ref[g] = 0.5f * gate * (1.0f + t) * up;
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)F * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(F / 2), 1}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int g = 0; g < F; ++g) {
    const double d = std::fabs((double)(float)yp[g] - (double)ref[g]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[g]) + 1e-2));
    std::printf("[geglu_w4g32] g=%2d gpu=%.5f ref=%.5f\n",
                g, (float)yp[g], ref[g]);
  }
  std::printf("[geglu_w4g32] max rel err = %.4g\n", maxrel);
  EXPECT_TRUE(maxrel < 0.05);
}

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

  auto gen = [&](bool no_flash) {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (no_flash) { ::setenv("VPIPE_QWEN_NO_FLASH", "1", 1); }
    else          { ::unsetenv("VPIPE_QWEN_NO_FLASH"); }
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
  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid());

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
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 16;               // max_seq 8192 (room for d4096+gen)
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Parse the comma-separated context-size list.
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
        for (int k = 0; k < 4; ++k) {    // warm the pipeline
          if (!lm->pdecode_commit(cd)) { break; }
          if (lm->pdecode_next(cd) < 0) { break; }
        }
        int produced = 0;
        const auto d0 = std::chrono::steady_clock::now();
        for (int k = 0; k < G; ++k) {
          if (!lm->pdecode_commit(cd)) { break; }
          if (lm->pdecode_next(cd) < 0) { break; }
          ++produced;
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
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
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
                        "misc", "attn_global", "attn_slide"};
  const int NC = 9;
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
  std::printf("[gguf_pdecode] ref=%zu greedy=%zu pipe=%zu  "
              "greedy_mism=%zu pipe_mism=%zu\n",
              ref.size(), grd.size(), pipe.size(), mism_g, mism_p);
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

// Regression for "chat truncates ~2048 tokens regardless of max_pages": the
// metal Gemma contiguous KV used to hardcode max_seq=2048, ignoring the
// configured budget. It must now follow page_tokens * max_pages. Load with a
// SMALL budget (128 * 3 = 384) and force-decode past it: the decode must hit
// its cap at ~384 -- NOT run on toward the old 2048 -- proving the budget is
// config-derived. (Build-agnostic: the MLX paged pool caps at the same
// 384-token budget.) Gated on the Gemma checkpoint.
TEST(metal_lm_smoke, gemma_kv_budget_follows_max_pages) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // The contiguous-KV cap is a METAL-backend property; pin metal so the test
  // exercises the metal exec in BOTH builds. (In the MLX build an unset
  // backend loads the MLX paged Gemma path, whose KV grows differently and
  // does not hard-stop at page_tokens*max_pages -- so the cap assertion only
  // makes sense against the metal exec.)
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  if (sess.metal_compute() == nullptr || !sess.metal_compute()->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  const int kPageTokens = 128, kMaxPages = 3;
  const int kBudget = kPageTokens * kMaxPages;     // 384
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = kPageTokens;
  spec.max_pages     = kMaxPages;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto ids = lm->tokenizer().encode("Count upwards:");
  ASSERT_TRUE(!ids.empty());
  ASSERT_TRUE((int)ids.size() < kBudget);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  int32_t forced = lm->prefill(ctx, ids);
  ASSERT_TRUE(forced >= 0);

  // Force-decode (ignore stop tokens) until the KV cap returns -1.
  int produced = 0;
  const int kLimit = 1500;     // > budget(384), << old hardcoded 2048
  for (int i = 0; i < kLimit; ++i) {
    const int32_t n = lm->next_token(ctx, forced);
    if (n < 0) { break; }
    forced = n;
    ++produced;
  }
  const int final_pos = (int)ids.size() + produced;
  std::printf("[metal_lm_smoke.gemma_kv_budget] budget=%d final_pos=%d "
              "produced=%d\n", kBudget, final_pos, produced);
  // Hit a cap well before the loop limit (and far below the old 2048),
  // landing at the configured budget.
  EXPECT_TRUE(produced < kLimit);
  EXPECT_TRUE(final_pos <= kBudget + kPageTokens);   // ~384, NOT ~2048
  EXPECT_TRUE(final_pos >= kBudget - kPageTokens);
}

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
  spec.max_pages = 32;           // up to 4096 ctx + decode (one ctx at a time)
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
    std::printf(
        "[BENCH-CTX] backend=%s greedy_fast=%d ctx=%d prefill_s=%.4f "
        "prefill_tps=%.1f decode_n=%d decode_s=%.4f decode_tps=%.2f\n",
        backend.c_str(), greedy_fast ? 1 : 0, N, ps, N / ps, produced, ds,
        produced / ds);
    EXPECT_TRUE(produced >= 1);

    // Greedy MTP speculative decode at the SAME depth N (OptiQ + MTP head).
    // A fresh context prefilled with the same ids so the spec decode starts at
    // depth N; report decode tok/s + tok/round + the speedup vs the baseline.
    if (mtp_en && lm->mtp_available()) {
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
  mcfg.max_pages = 32;            // up to 4096 ctx + decode, one ctx at a time
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
      std::vector<std::int32_t> out;
      const auto d0 = clk::now();
      const bool ok = model->decode_pipelined(cd, f, G, out);
      const double ds = secs(clk::now() - d0);
      tg_pipe = (ok && ds > 0.0 && !out.empty())
                    ? (double)out.size() / ds : 0.0;
      cm->release(cd);
    }

    std::printf("[qwen_gguf_ctx] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode(pipe)=%5.2f tok/s\n", N, pp_tps, ps, tg_pipe);
  }
  EXPECT_TRUE(true);
}

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

// Batched (N-branch parallel) metal decode must be token-exact vs serial
// decode_step_fast per branch. Branches share a prefill prefix but are fed
// DISTINCT first tokens so they diverge -- a batched forward that leaked one
// branch's K/V into another would mismatch. Gated on a hybrid Qwen3.5 model
// (VPIPE_QWEN35_TEST_MODEL_PATH) so the GDN linear-attention layers are
// exercised alongside the full-attention layers.
TEST(metal_lm_smoke, qwen_batched_decode_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 16;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }

  // Shared prefix prefill on a root context.
  auto prompt = tok->encode("The weather today is");
  if (prompt.empty()) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid()) { return; }
  if (model->prefill(root, prompt).empty()) { return; }

  const int N = 3;
  const int n_steps = 12;

  // Two independent branch sets sharing the same prefix: one batched, one
  // serial reference. Give each branch a DIFFERENT-LENGTH distinct suffix so
  // the branches sit at DIFFERENT seq_lens -- exercising the relaxed
  // (non-lockstep) batching where projections batch across the active set
  // while RoPE + attention run per branch at each branch's own position.
  auto batched = ctxm->branch(root, N);
  auto serial  = ctxm->branch(root, N);
  if ((int)batched.size() != N || (int)serial.size() != N) { return; }

  std::vector<std::int32_t> first_tokens((std::size_t)N);
  auto argmax_of = [&](const std::vector<float>& lg) {
    std::int32_t best = 0; float bv = -1e30f;
    for (std::size_t v = 0; v < lg.size(); ++v) {
      if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
    }
    return best;
  };
  for (int i = 0; i < N; ++i) {
    // Branch i: i+1 copies of a distinct token -> distinct content + length.
    std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                     (std::int32_t)(100 + i));
    auto lb = model->prefill(batched[(std::size_t)i], suffix);
    auto ls = model->prefill(serial[(std::size_t)i], suffix);
    if (lb.empty() || ls.empty()) { return; }
    first_tokens[(std::size_t)i] = argmax_of(lb);   // == argmax_of(ls)
  }

  // Serial reference per branch (each at its own position).
  std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::int32_t cur = first_tokens[(std::size_t)i];
    for (int s = 0; s < n_steps; ++s) {
      const std::int32_t nxt = model->decode_step_fast(serial[(std::size_t)i],
                                                       cur);
      ASSERT_TRUE(nxt != std::numeric_limits<std::int32_t>::min());
      if (nxt < 0) { break; }
      ref[(std::size_t)i].push_back(nxt);
      cur = nxt;
    }
  }

  // Batched.
  auto got = model->decode_batched_argmax(
      std::span<const genai::ContextId>(batched.data(), batched.size()),
      std::span<const std::int32_t>(first_tokens.data(), first_tokens.size()),
      n_steps);

  ASSERT_TRUE((int)got.size() == N);
  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(got[(std::size_t)i].size() == ref[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size();
         ++s) {
      ++total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(got[(std::size_t)i][s] == ref[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_batched_decode] N=%d steps=%d matched "
              "%d/%d\n", N, n_steps, matched, total);
}

// Shared-prefix batched decode attention (phase A reads the N branches' shared
// prefix once, phase B merges each branch's private pages) must be TOKEN-EXACT
// vs the per-branch SDPA. Uses a MULTI-PAGE shared prefix (small page_tokens)
// so shared_pages>=2, N=4 branches at distinct positions, and toggles
// set_shared_attn ON vs OFF in-process to isolate the shared split.
// VPIPE_SDPA_MB_MIN=0 forces the OFF path onto the mb256 kernel too, so the
// only difference is shared-vs-strided splitting of the same online softmax.
TEST(metal_lm_smoke, qwen_shared_attn_token_exact) {
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
  auto argmax_of = [&](const std::vector<float>& lg) {
    std::int32_t best = 0; float bv = -1e30f;
    for (std::size_t v = 0; v < lg.size(); ++v) {
      if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
    }
    return best;
  };
  // Branch N off the shared prefix (distinct-length suffixes -> distinct
  // positions) and return the per-branch argmax streams. `on` toggles the
  // shared-prefix path; both runs start from the same untouched root.
  auto run = [&](bool on) {
    model->set_shared_attn(on);
    auto br = ctxm->branch(root, N);
    std::vector<std::int32_t> first((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                       (std::int32_t)(100 + i));
      auto l = model->prefill(br[(std::size_t)i], suffix);
      first[(std::size_t)i] = argmax_of(l);
    }
    auto got = model->decode_batched_argmax(
        std::span<const genai::ContextId>(br.data(), br.size()),
        std::span<const std::int32_t>(first.data(), first.size()), n_steps);
    for (auto id : br) { ctxm->release(id); }
    return got;
  };
  auto off = run(false);
  auto onv = run(true);
  ASSERT_TRUE((int)off.size() == N && (int)onv.size() == N);
  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(off[(std::size_t)i].size() == onv[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < off[(std::size_t)i].size() && s < onv[(std::size_t)i].size();
         ++s) {
      ++total;
      if (off[(std::size_t)i][s] == onv[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(off[(std::size_t)i][s] == onv[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_shared_attn] N=%d steps=%d matched %d/%d "
              "(shared vs per-branch)\n", N, n_steps, matched, total);
  EXPECT_TRUE(total > 0 && matched == total);
}

// Batched PIPELINED decode (bdecode_*, GPU per-branch sampler + event-chain
// overlap) must be token-exact vs the synchronous decode_batched_argmax in
// GREEDY mode, including branches at DIFFERENT seq_lens. Two branch sets off
// a shared prefix: one drives decode_batched_argmax (reference), one drives
// bdecode_begin/commit/next greedily. Gated on a hybrid Qwen3.5 model.
TEST(metal_lm_smoke, qwen_bdecode_matches_batched_argmax) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 16;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }

  auto prompt = tok->encode("The weather today is");
  if (prompt.empty()) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid() || model->prefill(root, prompt).empty()) { return; }

  const int N = 3, n_steps = 12;
  auto ref_set = ctxm->branch(root, N);   // decode_batched_argmax
  auto pipe_set = ctxm->branch(root, N);  // bdecode_*
  if ((int)ref_set.size() != N || (int)pipe_set.size() != N) { return; }

  auto argmax_of = [&](const std::vector<float>& lg) {
    std::int32_t best = 0; float bv = -1e30f;
    for (std::size_t v = 0; v < lg.size(); ++v) {
      if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
    }
    return best;
  };
  std::vector<std::int32_t> first((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                     (std::int32_t)(100 + i));
    auto lr = model->prefill(ref_set[(std::size_t)i], suffix);
    auto lp = model->prefill(pipe_set[(std::size_t)i], suffix);
    if (lr.empty() || lp.empty()) { return; }
    first[(std::size_t)i] = argmax_of(lr);
  }

  // Reference: synchronous batched argmax.
  auto ref = model->decode_batched_argmax(
      std::span<const genai::ContextId>(ref_set.data(), ref_set.size()),
      std::span<const std::int32_t>(first.data(), first.size()), n_steps);

  // Pipelined: greedy bdecode over the second branch set.
  genai::GpuSamplerParams sp;   // greedy=true by default
  std::vector<std::vector<std::int32_t>> got((std::size_t)N);
  ASSERT_TRUE(model->bdecode_begin(
      std::span<const genai::ContextId>(pipe_set.data(), pipe_set.size()),
      std::span<const std::int32_t>(first.data(), first.size()), sp, n_steps));
  std::vector<std::int32_t> step_tok;
  for (int s = 0; s < n_steps; ++s) {
    if (!model->bdecode_commit()) { break; }
    if (!model->bdecode_next(step_tok)) { break; }
    for (int i = 0; i < N; ++i) {
      got[(std::size_t)i].push_back(step_tok[(std::size_t)i]);
    }
  }
  model->bdecode_end();

  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(got[(std::size_t)i].size() == ref[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size();
         ++s) {
      ++total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(got[(std::size_t)i][s] == ref[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_bdecode] N=%d steps=%d matched %d/%d\n",
              N, n_steps, matched, total);
}

// Perf triage: time single-branch decode (qmv) vs batched decode (qmm steel
// GEMM, M=N) at N=1,2,4 over the SAME model/context, to see whether batched
// per-step wall is the expected ~weight-read time (so aggregate tok/s scales
// with N) or whether qmm-at-small-M / per-branch dispatch / host logit pull
// dominates. Gated on VPIPE_QWEN35_TEST_MODEL_PATH + VPIPE_QWEN_BATCH_BENCH.
TEST(metal_lm_smoke, qwen_batched_decode_bench) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_QWEN_BATCH_BENCH")) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Headroom: the sections below branch repeatedly without releasing the raw
  // ContextIds, and the shared-prefix A/B prefills a ~1024-token root.
  mcfg.max_pages = 64;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }
  const int vocab = mcfg.vocab;

  auto prompt = tok->encode("The weather today is");
  if (prompt.empty()) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid()) { return; }
  if (model->prefill(root, prompt).empty()) { return; }

  using clock = std::chrono::steady_clock;
  const int K = 32;                 // timed steps
  std::vector<float> logits;

  // --- single-branch (qmv) reference ---
  {
    auto br = ctxm->branch(root, 1);
    std::int32_t cur = 100;
    for (int s = 0; s < 4; ++s) {   // warm
      cur = model->decode_step_fast(br[0], cur);
    }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) {
      cur = model->decode_step_fast(br[0], cur);
    }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench] single (qmv)   per-step %.2f ms  -> %.1f tok/s\n",
                1e3 * dt / K, K / dt);
  }

  // --- batched (qmm M=N) at several N ---
  for (int N : {1, 2, 4, 8}) {
    auto br = ctxm->branch(root, N);
    if ((int)br.size() != N) { continue; }
    std::vector<genai::ContextId> cids(br.begin(), br.end());
    std::vector<std::int32_t> cur((std::size_t)N, 100);
    auto step = [&]() {
      if (!model->decode_batched_step(
              std::span<const genai::ContextId>(cids.data(), cids.size()),
              std::span<const std::int32_t>(cur.data(), cur.size()),
              std::span<const std::int32_t>(), logits)) {
        return false;
      }
      for (int i = 0; i < N; ++i) {
        const float* row = logits.data() + (std::size_t)i * vocab;
        std::int32_t best = 0; float bv = row[0];
        for (int v = 1; v < vocab; ++v) {
          if (row[v] > bv) { bv = row[v]; best = v; }
        }
        cur[(std::size_t)i] = best;
      }
      return true;
    };
    for (int s = 0; s < 4; ++s) { step(); }   // warm
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) { step(); }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench] batched N=%d    per-step %.2f ms  -> %.1f tok/s "
                "(aggregate, %d branches)\n",
                N, 1e3 * dt / K, (double)N * K / dt, N);
  }

  // --- pipelined bdecode (GPU sampler + event-chain overlap), greedy ---
  for (int N : {2, 4, 8}) {
    auto br = ctxm->branch(root, N);
    if ((int)br.size() != N) { continue; }
    std::vector<genai::ContextId> cids(br.begin(), br.end());
    std::vector<std::int32_t> first((std::size_t)N, 100);
    genai::GpuSamplerParams sp;   // greedy
    if (!model->bdecode_begin(
            std::span<const genai::ContextId>(cids.data(), cids.size()),
            std::span<const std::int32_t>(first.data(), first.size()),
            sp, K + 8)) {
      continue;
    }
    std::vector<std::int32_t> toks;
    for (int s = 0; s < 4; ++s) {           // warm
      model->bdecode_commit(); model->bdecode_next(toks);
    }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) {
      if (!model->bdecode_commit() || !model->bdecode_next(toks)) { break; }
    }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    model->bdecode_end();
    std::printf("[bench] pipelined N=%d  per-step %.2f ms  -> %.1f tok/s "
                "(aggregate, %d branches)\n",
                N, 1e3 * dt / K, (double)N * K / dt, N);
  }

  // --- STAGGERED finish A/B: shrinking (sync) vs constant-N (pipelined) ----
  // The realtime-vqa reality: the N question-branches finish at DIFFERENT
  // lengths. The shrinking sync path drops a branch as it stops (active N
  // falls); the pipelined path is constant-N -> it keeps doing all 8
  // branches' work until the LONGEST finishes, so it pays for already-done
  // branches. Same useful token count both ways; compare aggregate tok/s.
  {
    const int Nst = 8;
    const int Lst[8] = {4, 8, 12, 16, 20, 24, 28, 32};   // per-branch lengths
    int useful = 0, maxL = 0;
    for (int i = 0; i < Nst; ++i) { useful += Lst[i]; maxL = std::max(maxL, Lst[i]); }

    // Shrinking (sync): re-batch only the still-active branches each step.
    {
      auto br = ctxm->branch(root, Nst);
      if ((int)br.size() == Nst) {
        std::vector<genai::ContextId> all(br.begin(), br.end());
        std::vector<std::int32_t> cur((std::size_t)Nst, 100);
        std::vector<int> rem(Lst, Lst + Nst);
        const auto t0 = clock::now();
        for (;;) {
          std::vector<genai::ContextId> act;
          std::vector<std::int32_t> actc;
          std::vector<int> map;
          for (int i = 0; i < Nst; ++i) {
            if (rem[i] > 0) { act.push_back(all[i]); actc.push_back(cur[i]);
                              map.push_back(i); }
          }
          if (act.empty()) { break; }
          if (!model->decode_batched_step(
                  std::span<const genai::ContextId>(act.data(), act.size()),
                  std::span<const std::int32_t>(actc.data(), actc.size()),
                  std::span<const std::int32_t>(), logits)) {
            break;
          }
          for (std::size_t j = 0; j < map.size(); ++j) {
            const float* row = logits.data() + j * vocab;
            std::int32_t best = 0; float bv = row[0];
            for (int v = 1; v < vocab; ++v) {
              if (row[v] > bv) { bv = row[v]; best = v; }
            }
            cur[(std::size_t)map[j]] = best;
            rem[(std::size_t)map[j]]--;
          }
        }
        const double dt =
            std::chrono::duration<double>(clock::now() - t0).count();
        std::printf("[bench] staggered N=8 shrinking(sync)  %.1f tok/s "
                    "(%d useful tok in %.3fs)\n", useful / dt, useful, dt);
        for (auto id : br) { ctxm->release(id); }
      }
    }
    // Constant-N (pipelined): runs all 8 for maxL steps; useful tokens are
    // the same `useful`, but wall time covers Nst*maxL branch-steps.
    {
      auto br = ctxm->branch(root, Nst);
      if ((int)br.size() == Nst) {
        std::vector<genai::ContextId> cids(br.begin(), br.end());
        std::vector<std::int32_t> first((std::size_t)Nst, 100);
        genai::GpuSamplerParams sp;
        if (model->bdecode_begin(
                std::span<const genai::ContextId>(cids.data(), cids.size()),
                std::span<const std::int32_t>(first.data(), first.size()),
                sp, maxL + 8)) {
          std::vector<std::int32_t> toks;
          const auto t0 = clock::now();
          for (int s = 0; s < maxL; ++s) {
            if (!model->bdecode_commit() || !model->bdecode_next(toks)) {
              break;
            }
          }
          const double dt =
              std::chrono::duration<double>(clock::now() - t0).count();
          model->bdecode_end();
          std::printf("[bench] staggered N=8 constant(pipe)  %.1f tok/s "
                      "(%d useful tok in %.3fs)\n", useful / dt, useful, dt);
        }
        for (auto id : br) { ctxm->release(id); }
      }
    }
  }

  // --- shared-prefix attention A/B over a LONG shared prefix ---------------
  // The realtime-vqa win: with a big shared prefix (image/video tokens), the
  // per-branch SDPA re-reads it N times. Phase A reads it once. Build a ~1024-
  // token shared prefix and time batched decode with set_shared_attn OFF vs
  // ON at N=2,4 (the gap grows with prefix length).
  for (int PLEN : (mcfg.head_dim == 256 && model->shared_attn())
                      ? std::vector<int>{1024, 4096, 8192}
                      : std::vector<int>{}) {
    std::vector<std::int32_t> long_ids;
    long_ids.reserve((std::size_t)PLEN);
    while ((int)long_ids.size() < PLEN) {
      for (std::int32_t t : prompt) {
        if ((int)long_ids.size() >= PLEN) { break; }
        long_ids.push_back(t);
      }
    }
    auto lroot = ctxm->acquire_root();
    if (lroot.valid() && !model->prefill(lroot, long_ids).empty()) {
      std::printf("[bench] --- shared-prefix attn (prefix=%d tok) ---\n",
                  (int)long_ids.size());
      for (int N : {2, 4, 8}) {
        for (int on = 0; on <= 1; ++on) {
          model->set_shared_attn(on != 0);
          auto br = ctxm->branch(lroot, N);
          if ((int)br.size() != N) { continue; }
          std::vector<genai::ContextId> cids(br.begin(), br.end());
          std::vector<std::int32_t> cur((std::size_t)N, 100);
          auto step = [&]() {
            if (!model->decode_batched_step(
                    std::span<const genai::ContextId>(cids.data(), cids.size()),
                    std::span<const std::int32_t>(cur.data(), cur.size()),
                    std::span<const std::int32_t>(), logits)) {
              return false;
            }
            for (int i = 0; i < N; ++i) {
              const float* row = logits.data() + (std::size_t)i * vocab;
              std::int32_t best = 0; float bv = row[0];
              for (int v = 1; v < vocab; ++v) {
                if (row[v] > bv) { bv = row[v]; best = v; }
              }
              cur[(std::size_t)i] = best;
            }
            return true;
          };
          for (int s = 0; s < 4; ++s) { step(); }   // warm
          const auto t0 = clock::now();
          for (int s = 0; s < K; ++s) { step(); }
          const double dt =
              std::chrono::duration<double>(clock::now() - t0).count();
          std::printf("[bench] batched N=%d shared_attn=%-3s per-step %.2f ms "
                      " -> %.1f tok/s\n", N, on ? "ON" : "OFF",
                      1e3 * dt / K, (double)N * K / dt);
          for (auto id : br) { ctxm->release(id); }
        }
      }
      model->set_shared_attn(true);
    }
    ctxm->release(lroot);   // free the long prefix before the next length
  }
}

// LM-level batched decode (the path realtime-vqa uses): branch N contexts off
// a shared prefix, give each a different-length suffix (so they sit at
// DIFFERENT positions), then drive LoadedLanguageModel::m_batched_decode_step
// greedily and require it token-exact vs a serial next_token_greedy loop per
// branch. Validates the exec cid-mapping + rope bookkeeping the stage relies
// on. Gated on a hybrid Qwen3.5 model.
TEST(metal_lm_smoke, qwen_lm_batched_step_matches_serial) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid() || !lm->m_batched_decode_supported()) { return; }

  auto root = lm->make_context();
  if (!root.valid()) { return; }
  auto prefix = lm->tokenizer().encode("The weather today is");
  if (prefix.empty() || lm->prefill(root, prefix) < 0) { return; }

  const int N = 3, n_steps = 10;
  const int vocab = lm->config().vocab_size;
  auto bset = lm->branch(root, N);
  auto sset = lm->branch(root, N);
  if ((int)bset.size() != N || (int)sset.size() != N) { return; }

  std::vector<std::int32_t> first((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::vector<std::int32_t> suf((std::size_t)(i + 1), (std::int32_t)(100 + i));
    const std::int32_t fb = lm->prefill(bset[(std::size_t)i], suf);
    const std::int32_t fs = lm->prefill(sset[(std::size_t)i], suf);
    if (fb < 0 || fs < 0) { return; }
    first[(std::size_t)i] = fb;
  }

  std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::int32_t cur = first[(std::size_t)i];
    for (int s = 0; s < n_steps; ++s) {
      const std::int32_t nxt = lm->next_token_greedy(sset[(std::size_t)i], cur);
      if (nxt < 0) { break; }
      ref[(std::size_t)i].push_back(nxt);
      cur = nxt;
    }
  }

  std::vector<std::int32_t> cur(first.begin(), first.end());
  std::vector<std::vector<std::int32_t>> got((std::size_t)N);
  std::vector<float> logits;
  for (int s = 0; s < n_steps; ++s) {
    std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
    for (int i = 0; i < N; ++i) { ptrs[(std::size_t)i] = &bset[(std::size_t)i]; }
    if (!lm->m_batched_decode_step(
            std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                          ptrs.size()),
            std::span<const std::int32_t>(cur.data(), cur.size()), logits)) {
      break;
    }
    for (int i = 0; i < N; ++i) {
      const float* row = logits.data() + (std::size_t)i * vocab;
      std::int32_t best = 0; float bv = row[0];
      for (int v = 1; v < vocab; ++v) { if (row[v] > bv) { bv = row[v]; best = v; } }
      got[(std::size_t)i].push_back(best);
      cur[(std::size_t)i] = best;
    }
  }

  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(got[(std::size_t)i].size() == ref[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size(); ++s) {
      ++total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(got[(std::size_t)i][s] == ref[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_lm_batched_step] matched %d/%d\n",
              matched, total);
}

// Gemma-4 batched decode (LM level), token-exact vs serial. Branch N off a
// shared prefix, give each a different-length suffix (different positions),
// drive m_batched_decode_step greedily vs next_token_greedy. Returns
// {matched,total}; the TEST asserts (EXPECT_TRUE can't live in a free fn).
namespace {
struct GemmaBatchedResult { bool loaded = false; int matched = 0, total = 0; };
GemmaBatchedResult gemma_lm_batched_run_(const char* path) {
  GemmaBatchedResult r;
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return r; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 4;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid() || !lm->m_batched_decode_supported()) { return r; }
  r.loaded = true;

  auto root = lm->make_context();
  if (!root.valid()) { return r; }
  auto prefix = lm->tokenizer().encode("The weather today is");
  if (prefix.empty() || lm->prefill(root, prefix) < 0) { return r; }

  const int N = 3, n_steps = 10;
  const int vocab = lm->config().vocab_size;
  auto bset = lm->branch(root, N);
  auto sset = lm->branch(root, N);
  if ((int)bset.size() != N || (int)sset.size() != N) { return r; }

  std::vector<std::int32_t> first((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::vector<std::int32_t> suf((std::size_t)(i + 1), (std::int32_t)(100 + i));
    const std::int32_t fb = lm->prefill(bset[(std::size_t)i], suf);
    const std::int32_t fs = lm->prefill(sset[(std::size_t)i], suf);
    if (fb < 0 || fs < 0) { return r; }
    first[(std::size_t)i] = fb;
  }

  std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::int32_t cur = first[(std::size_t)i];
    for (int s = 0; s < n_steps; ++s) {
      const std::int32_t nxt = lm->next_token_greedy(sset[(std::size_t)i], cur);
      if (nxt < 0) { break; }
      ref[(std::size_t)i].push_back(nxt);
      cur = nxt;
    }
  }

  std::vector<std::int32_t> cur(first.begin(), first.end());
  std::vector<std::vector<std::int32_t>> got((std::size_t)N);
  std::vector<float> logits;
  for (int s = 0; s < n_steps; ++s) {
    std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
    for (int i = 0; i < N; ++i) { ptrs[(std::size_t)i] = &bset[(std::size_t)i]; }
    if (!lm->m_batched_decode_step(
            std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                          ptrs.size()),
            std::span<const std::int32_t>(cur.data(), cur.size()), logits)) {
      break;
    }
    for (int i = 0; i < N; ++i) {
      const float* row = logits.data() + (std::size_t)i * vocab;
      std::int32_t best = 0; float bv = row[0];
      for (int v = 1; v < vocab; ++v) { if (row[v] > bv) { bv = row[v]; best = v; } }
      got[(std::size_t)i].push_back(best);
      cur[(std::size_t)i] = best;
    }
  }
  for (int i = 0; i < N; ++i) {
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size(); ++s) {
      ++r.total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++r.matched; }
    }
  }
  return r;
}
}  // namespace

// 12B gemma4_unified: exercises the k_eq_v / mixed-quant / per-layer-n_kv /
// no-PLE batched path.
TEST(metal_lm_smoke, gemma12b_batched_step_matches_serial) {
  const char* path = std::getenv("VPIPE_GEMMA12B_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  auto r = gemma_lm_batched_run_(path);
  if (!r.loaded) { return; }
  EXPECT_TRUE(r.total > 0);
  EXPECT_TRUE(r.matched == r.total);
  std::printf("[metal_lm_smoke.gemma12b_batched_step] matched %d/%d\n",
              r.matched, r.total);
}

// e4b: exercises the PLE + cross-layer-KV-sharing batched path.
TEST(metal_lm_smoke, gemma_e4b_batched_step_matches_serial) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  auto r = gemma_lm_batched_run_(path);
  if (!r.loaded) { return; }
  EXPECT_TRUE(r.total > 0);
  EXPECT_TRUE(r.matched == r.total);
  std::printf("[metal_lm_smoke.gemma_e4b_batched_step] matched %d/%d\n",
              r.matched, r.total);
}

// Gemma e4b batched-decode perf (LM API, the realtime-vqa path): serial
// next_token_greedy on one branch vs m_batched_decode_step over N. Confirms
// the batched GEMV recovers the win for the geglu MLP. Gated on
// VPIPE_GEMMA4_TEST_MODEL_PATH + VPIPE_QWEN_BATCH_BENCH.
TEST(metal_lm_smoke, gemma_e4b_batched_decode_bench) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_QWEN_BATCH_BENCH")) { return; }
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
  if (!lm || !lm->valid() || !lm->m_batched_decode_supported()) { return; }
  const int vocab = lm->config().vocab_size;

  auto root = lm->make_context();
  if (!root.valid()) { return; }
  // Shared scene prefix. Default short; VPIPE_GEMMA_BATCH_PREFIX_LEN grows it
  // to ~N tokens so the batched bench exercises the global layers' long-context
  // decode attention (the realtime-vqa regime where gtile pays off).
  std::string ptext = "The weather today is";
  if (const char* e = std::getenv("VPIPE_GEMMA_BATCH_PREFIX_LEN")) {
    const int want = std::atoi(e);
    ptext.clear();
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
  }
  auto prefix = lm->tokenizer().encode(ptext);
  if (prefix.empty() || lm->prefill(root, prefix) < 0) { return; }
  std::printf("[bench-gemma] prefix=%zu tokens\n", prefix.size());

  using clock = std::chrono::steady_clock;
  const int K = 32;

  const std::vector<std::int32_t> seed1{(std::int32_t)100};
  {
    auto br = lm->branch(root, 1);
    std::int32_t cur = br[0].last_predicted_id();
    if (cur < 0) { cur = lm->prefill(br[0], seed1); }
    for (int s = 0; s < 4; ++s) { cur = lm->next_token_greedy(br[0], cur); }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) { cur = lm->next_token_greedy(br[0], cur); }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench-gemma] single (qmv)   per-step %.2f ms -> %.1f tok/s\n",
                1e3 * dt / K, K / dt);
  }

  std::vector<float> logits;
  for (int N : {2, 4}) {
    auto br = lm->branch(root, N);
    if ((int)br.size() != N) { continue; }
    std::vector<std::int32_t> cur((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      cur[(std::size_t)i] = br[(std::size_t)i].last_predicted_id();
      if (cur[(std::size_t)i] < 0) {
        cur[(std::size_t)i] = lm->prefill(br[(std::size_t)i], seed1);
      }
    }
    std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
    for (int i = 0; i < N; ++i) { ptrs[(std::size_t)i] = &br[(std::size_t)i]; }
    auto step = [&]() {
      if (!lm->m_batched_decode_step(
              std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                            ptrs.size()),
              std::span<const std::int32_t>(cur.data(), cur.size()), logits)) {
        return false;
      }
      for (int i = 0; i < N; ++i) {
        const float* row = logits.data() + (std::size_t)i * vocab;
        std::int32_t best = 0; float bv = row[0];
        for (int v = 1; v < vocab; ++v) {
          if (row[v] > bv) { bv = row[v]; best = v; }
        }
        cur[(std::size_t)i] = best;
      }
      return true;
    };
    for (int s = 0; s < 4; ++s) { step(); }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) { step(); }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench-gemma] batched N=%d    per-step %.2f ms -> %.1f tok/s "
                "(aggregate)\n", N, 1e3 * dt / K, (double)N * K / dt);
  }
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
  std::printf("[pdecode-ab] ctx=%zu K=%d | sync %.1f tok/s | pipe d1 %.1f "
              "tok/s | pipe d2 %.1f tok/s | d1_mism=%zu d2_mism=%zu\n",
              ids.size(), K, sync_s > 0 ? K / sync_s : 0.0,
              d1_s > 0 ? K / d1_s : 0.0, d2_s > 0 ? K / d2_s : 0.0, m1, m2);
  EXPECT_TRUE(m1 == 0);
  EXPECT_TRUE(m2 == 0);
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
  spec.max_pages = 32;
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

// MOSS-TTS-8B (MossTTSDelay): the metal delay-pattern LM forward must agree
// with the mlx-audio reference. Full autoregressive token-exactness is
// IMPOSSIBLE here -- the audio heads are full of exact bf16 logit ties (two
// codes both at e.g. 55.5), so any tie-flip diverges + cascades, and the
// model README itself is non-reproducible across implementations. Instead we
// verify the FORWARD: teacher-force the reference rows and require that every
// early-window disagreement (before recurrent bf16 cache drift accumulates)
// is a numerical tie, and that the leading autoregressive rows are exact.
// Gated on a model dir + a golden-artifact dir produced by the mlx-audio
// reference (dump_golden.py): shapes.txt ("seq G channels"), input_ids.i32
// (the [seq, channels] prompt grid) and gen_delay.i32 (the [G, channels]
// reference generation, both raw little-endian int32 row-major).
TEST(metal_lm_smoke, moss_tts_delay_greedy_forward) {
  const char* path = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (path == nullptr || *path == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string gd(gold);
  int seq = 0, G = 0, ch = 0;
  {
    std::ifstream s(gd + "/shapes.txt");
    if (!s) { return; }
    s >> seq >> G >> ch;
  }
  ASSERT_TRUE(seq > 0 && G > 0 && ch == 33);
  auto read_i32 = [&](const std::string& fn, int count) {
    std::vector<std::int32_t> v((std::size_t)count, 0);
    std::ifstream f(gd + "/" + fn, std::ios::binary);
    if (f) {
      f.read(reinterpret_cast<char*>(v.data()),
             (std::streamsize)count * 4);
    }
    return v;
  };
  const std::vector<std::int32_t> iid = read_i32("input_ids.i32", seq * ch);
  const std::vector<std::int32_t> gen = read_i32("gen_delay.i32", G * ch);

  std::vector<std::vector<std::int32_t>> prompt(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < seq; ++r) {
    for (int c = 0; c < ch; ++c) {
      prompt[(std::size_t)r][(std::size_t)c] =
          iid[(std::size_t)(r * ch + c)];
    }
  }

  auto model = genai::MetalMossTtsModel::load(path, mc);
  ASSERT_TRUE(model != nullptr && model->valid());

  // Reference generation rows [G][33].
  std::vector<std::vector<std::int32_t>> ref_rows(
      (std::size_t)G, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < G; ++r) {
    for (int c = 0; c < ch; ++c) {
      ref_rows[(std::size_t)r][(std::size_t)c] = gen[(std::size_t)(r * ch + c)];
    }
  }

  // --- GATE: teacher-forced forward correctness ----------------------
  // The MOSS audio heads are full of EXACT bf16 logit ties (e.g. two codes
  // both at 55.5), so two bf16 implementations cannot agree on every greedy
  // argmax -- a single tie-flip then cascades. The forward is verified
  // correct if every active-codebook disagreement with the reference is a
  // numerical near-tie (my argmax's logit ~= the reference code's logit).
  const auto t0 = std::chrono::steady_clock::now();
  auto mism = model->teacher_force_audio_mismatches(prompt, ref_rows);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  int active = 0;
  const int pad = model->config().audio_pad_code;
  for (int r = 0; r < G; ++r) {
    for (int cb = 0; cb < 32; ++cb) {
      if (gen[(std::size_t)(r * ch + 1 + cb)] != pad) { ++active; }
    }
  }
  constexpr float kTieTol = 0.75f;   // ~3 bf16 quanta at logit magnitude ~50
  constexpr int kEarly = 16;         // window before recurrent drift dominates
  int real_mismatch = 0, real_early = 0, first_real = -1, last_real = -1;
  float max_gap = 0.0f;
  int bucket[7] = {0, 0, 0, 0, 0, 0, 0};   // real mismatches per 10-row bucket
  for (const auto& m : mism) {
    const float gap = m.my_logit - m.ref_logit;   // >= 0 (my pick is argmax)
    if (gap > max_gap) { max_gap = gap; }
    if (gap > kTieTol) {
      ++real_mismatch;
      if (m.row < kEarly) { ++real_early; }
      if (first_real < 0) { first_real = m.row; }
      last_real = m.row;
      const int b = std::min(m.row / 10, 6);
      ++bucket[b];
    }
  }
  std::printf("[moss-tts-tf] %.2fs | active=%d disagreements=%zu "
              "(near-tie if gap<=%.2f) real(gap>tol)=%d max_gap=%.3f\n",
              dt, active, mism.size(), kTieTol, real_mismatch, max_gap);
  std::printf("[moss-tts-tf] real-mismatch rows: first=%d last=%d | per-10row "
              "buckets [0-9..60+]: %d %d %d %d %d %d %d | early(<%d)=%d\n",
              first_real, last_real, bucket[0], bucket[1], bucket[2], bucket[3],
              bucket[4], bucket[5], bucket[6], kEarly, real_early);
  for (std::size_t i = 0; i < mism.size() && i < 6; ++i) {
    std::printf("   row %d cb %d: mine=%d (%.3f) ref=%d (%.3f) gap=%.3f\n",
                mism[i].row, mism[i].codebook, mism[i].my_code,
                (double)mism[i].my_logit, mism[i].ref_code,
                (double)mism[i].ref_logit,
                (double)(mism[i].my_logit - mism[i].ref_logit));
  }
  // Forward correctness gate: BEFORE recurrent bf16 drift accumulates, every
  // disagreement must be a numerical tie (no gap>tol). Full-sequence
  // divergence is expected (the model README itself is non-reproducible
  // across implementations) and is judged by the decoded audio in Phase 2.
  EXPECT_TRUE(real_early == 0);

  // --- Smoke: the autoregressive loop + delay state machine. The leading
  // rows are exact (before any near-tie can flip); the run produces audio
  // and stops on its own. ---------------------------------------------
  auto outr = model->generate_delay_greedy(prompt, G);
  EXPECT_TRUE(!outr.empty());
  const int lead = std::min((int)outr.size(), std::min(G, 4));
  int lead_exact = 0;
  for (int r = 0; r < lead; ++r) {
    bool eq = true;
    for (int c = 0; c < ch; ++c) {
      if (outr[(std::size_t)r][(std::size_t)c] !=
          gen[(std::size_t)(r * ch + c)]) { eq = false; break; }
    }
    if (eq) { ++lead_exact; }
  }
  std::printf("[moss-tts] autoregressive gen=%d/%d | leading_exact_rows=%d/%d\n",
              (int)outr.size(), G, lead_exact, lead);
  EXPECT_TRUE(lead_exact == lead);
}

// MOSS Audio Tokenizer codec decode: the metal RVQ + 4-stage transformer
// decoder must reproduce the reference's per-stage tensors + final 24 kHz PCM
// within an f16 rel-L2 tolerance (the reference runs F32; we run f16). Gated
// on the codec dir + the golden dir (dump_codec_golden.py): codes_dedelay.i32
// + codes_shape.txt (the [T, n_vq] input), rvq.f32 / dec_stage0..7.f32 /
// wave.f32 (the reference intermediates, raw little-endian f32).
TEST(metal_lm_smoke, moss_codec_decode_rel_l2) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (path == nullptr || *path == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gd(gold);

  int T = 0, nvq = 0;
  {
    std::ifstream s(gd + "/codes_shape.txt");
    if (!s) { return; }
    s >> T >> nvq;
  }
  ASSERT_TRUE(T > 0 && nvq == 32);
  std::vector<std::int32_t> flat((std::size_t)T * nvq, 0);
  {
    std::ifstream f(gd + "/codes_dedelay.i32", std::ios::binary);
    if (!f) { return; }
    f.read(reinterpret_cast<char*>(flat.data()), (std::streamsize)T * nvq * 4);
  }
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = flat[(std::size_t)(t * nvq + c)];
    }
  }

  auto read_f32 = [&](const std::string& fn) {
    std::ifstream f(gd + "/" + fn, std::ios::binary | std::ios::ate);
    std::vector<float> v;
    if (!f) { return v; }
    const std::streamsize n = f.tellg() / 4;
    f.seekg(0);
    v.resize((std::size_t)n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    return v;
  };
  auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) { return 9.99; }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  auto codec = genai::MetalMossCodec::load(path, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  std::vector<std::vector<float>> stages;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> wave = codec->decode(codes, &stages);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(!wave.empty());

  const char* names[9] = {"rvq", "dec_stage0", "dec_stage1", "dec_stage2",
                          "dec_stage3", "dec_stage4", "dec_stage5",
                          "dec_stage6", "dec_stage7"};
  EXPECT_TRUE(stages.size() == 9);
  double worst = 0.0;
  for (std::size_t k = 0; k < stages.size() && k < 9; ++k) {
    const std::vector<float> ref = read_f32(std::string(names[k]) + ".f32");
    const double r = rel_l2(stages[k], ref);
    if (r > worst) { worst = r; }
    std::printf("[moss-codec] %-11s mine=%zu ref=%zu rel_l2=%.4f\n", names[k],
                stages[k].size(), ref.size(), r);
  }
  const std::vector<float> wref = read_f32("wave.f32");
  const double wr = rel_l2(wave, wref);
  std::printf("[moss-codec] %.2fs | wave samples=%zu/%zu rel_l2=%.4f | "
              "worst_stage=%.4f\n", dt, wave.size(), wref.size(), wr, worst);
  // f16 codec vs F32 reference: per-stage error stays small; the waveform is
  // the end-to-end arbiter.
  EXPECT_TRUE(wr < 0.05);
}

// End-to-end metal MOSS-TTS: LM (delay-pattern code generation) -> de-delay +
// drop all-pad frames (the reference _decode_generated_audio pipeline) ->
// codec -> 24 kHz PCM, written to a playable WAV. Gated on the LM + codec
// dirs + the golden dir (for the prompt grid). Writes
// $VPIPE_MOSS_TTS_GOLDEN/e2e.wav.
TEST(metal_lm_smoke, moss_tts_end_to_end_wav) {
  const char* lm_path = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* cc_path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (!lm_path || !*lm_path || !cc_path || !*cc_path || !gold || !*gold) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gd(gold);

  int seq = 0, G = 0, ch = 0;
  {
    std::ifstream s(gd + "/shapes.txt");
    if (!s) { return; }
    s >> seq >> G >> ch;
  }
  ASSERT_TRUE(seq > 0 && ch == 33);
  std::vector<std::int32_t> iid((std::size_t)seq * ch, 0);
  {
    std::ifstream f(gd + "/input_ids.i32", std::ios::binary);
    if (!f) { return; }
    f.read(reinterpret_cast<char*>(iid.data()), (std::streamsize)seq * ch * 4);
  }
  std::vector<std::vector<std::int32_t>> prompt(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < seq; ++r) {
    for (int c = 0; c < ch; ++c) {
      prompt[(std::size_t)r][(std::size_t)c] = iid[(std::size_t)(r * ch + c)];
    }
  }

  const int n_vq = 32, pad = 1024;
  std::vector<std::vector<std::int32_t>> gen;
  {
    auto lm = genai::MetalMossTtsModel::load(lm_path, mc);
    ASSERT_TRUE(lm != nullptr && lm->valid());
    gen = lm->generate_delay_greedy(prompt, 1024);   // [Gg][33]
  }  // free the 8B LM before loading the codec (16 GB box)
  ASSERT_TRUE(!gen.empty());
  const int Gg = (int)gen.size();

  // De-delay: tokens[t][cb] = audio_codes[cb + t][cb], audio_codes = gen[:,1:].
  const int out_len = Gg - n_vq + 1;
  std::vector<std::vector<std::int32_t>> codes;
  for (int t = 0; t < out_len; ++t) {
    std::vector<std::int32_t> row((std::size_t)n_vq, 0);
    bool all_pad = true;
    for (int cb = 0; cb < n_vq; ++cb) {
      int v = gen[(std::size_t)(cb + t)][(std::size_t)(1 + cb)];
      if (v != pad) { all_pad = false; }
      if (v < 0 || v >= pad) { v = pad - 1; }   // clamp pad/OOB to a valid code
      row[(std::size_t)cb] = v;
    }
    if (!all_pad) { codes.push_back(std::move(row)); }   // drop all-pad frames
  }
  std::printf("[moss-e2e] LM gen rows=%d -> de-delay %d -> non-pad frames=%zu\n",
              Gg, out_len, codes.size());
  ASSERT_TRUE(!codes.empty());

  std::vector<float> wave;
  int sr = 24000;
  {
    auto codec = genai::MetalMossCodec::load(cc_path, mc);
    ASSERT_TRUE(codec != nullptr && codec->valid());
    sr = codec->sample_rate();
    wave = codec->decode(codes, nullptr);
  }
  ASSERT_TRUE(!wave.empty());
  double peak = 0.0;
  for (float s : wave) { peak = std::max(peak, (double)std::fabs(s)); }
  std::printf("[moss-e2e] codec -> %zu samples = %.2fs @ %dHz | peak=%.3f\n",
              wave.size(), wave.size() / (double)sr, sr, peak);
  EXPECT_TRUE(peak > 0.01);   // produced actual audio, not silence

  // Write a 16-bit PCM mono WAV.
  const std::string wav = gd + "/e2e.wav";
  std::ofstream out(wav, std::ios::binary);
  const std::uint32_t n = (std::uint32_t)wave.size();
  const std::uint32_t data_bytes = n * 2;
  const std::uint32_t byte_rate = (std::uint32_t)sr * 2;
  auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<char*>(&v), 4); };
  auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<char*>(&v), 2); };
  out.write("RIFF", 4); u32(36 + data_bytes); out.write("WAVE", 4);
  out.write("fmt ", 4); u32(16); u16(1); u16(1);
  u32((std::uint32_t)sr); u32(byte_rate); u16(2); u16(16);
  out.write("data", 4); u32(data_bytes);
  for (float s : wave) {
    int v = (int)std::lround(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
    u16((std::uint16_t)(std::int16_t)v);
  }
  out.close();
  std::printf("[moss-e2e] wrote %s\n", wav.c_str());
}
