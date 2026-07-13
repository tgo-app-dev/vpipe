// Krea-2-Turbo text-encoder bring-up (M2).
//
// The Krea 2 pipeline conditions its DiT on a stack of hidden states tapped
// from 12 layers of a Qwen3-VL text encoder. That encoder is a standard dense
// Qwen3 decoder (36L / hidden 2560 / 32-q-8-kv / head_dim 128 / ffn 9728,
// q/k-norm, standard RMSNorm, mRoPE theta 5e6), shape-identical to the
// MOSS-TTS-v1.5 backbone -- so we run it on MetalQwenModel (raw bf16, dense)
// and use the new per-layer tap (forward_embeddings_taps) to capture
// hidden_states[{2,5,8,11,14,17,20,23,26,29,32,35}] == the tap after layers
// {1,4,7,10,13,16,19,22,25,28,31,34}.
//
// Verified against the HF diffusers golden (krea2_golden.py): feed the EXACT
// encoder token ids (a1_full_text_ids.i32, the compact [prefix|prompt|suffix]
// sequence), tap, drop the 34 prefix rows, and rel-L2 the (n_real,12,2560)
// stack against a1_prompt_embeds.f32. The golden itself carries ~1% bf16
// reduction-order noise (padded-vs-compact), so the bar is rel-L2 < 0.03.
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH = the Krea-2-Turbo model root (uses
// <root>/text_encoder), VPIPE_KREA2_GOLDEN = the golden dir. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

std::vector<std::int32_t>
read_i32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<std::int32_t> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

MetalQwenModel::Config krea2_encoder_config_()
{
  MetalQwenModel::Config c;
  c.n_layers          = 36;
  c.hidden            = 2560;
  c.n_heads           = 32;
  c.n_kv_heads        = 8;
  c.head_dim          = 128;
  c.ffn_inner         = 9728;
  c.vocab             = 151936;
  c.rope_theta        = 5.0e6f;   // Qwen3-VL text rope_theta
  c.rms_eps           = 1e-6f;
  c.rotary_dim        = 128;      // full rotary
  c.full_attn_interval = 1;
  c.tie_embeddings    = true;
  c.use_bf16          = true;     // raw bf16 dense (auto-detected too)
  c.dense             = true;
  c.zero_centered_norm = false;   // Qwen3-VL uses STANDARD RMSNorm (weight*h)
  c.attn_output_gate  = false;    // q_proj = qd (no gate)
  c.backbone_only     = true;     // we host-gather embeds ourselves
  c.weight_prefix     = "language_model.";
  c.model_seg         = "";       // names are language_model.layers.N
  c.max_seq           = 512;
  c.page_tokens       = 256;
  return c;
}

}  // namespace

TEST(krea2_encoder, layer_tap_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;   // env-gated: skips vacuously
  }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string enc_dir = std::string(root) + "/text_encoder";
  const std::string gdir    = gd;

  // The exact encoder token ids + the golden per-layer stack.
  const std::vector<std::int32_t> ids =
      read_i32_(gdir + "/a1_full_text_ids.i32");
  ASSERT_TRUE(!ids.empty());
  const std::vector<float> golden = read_f32_(gdir + "/a1_prompt_embeds.f32");
  ASSERT_TRUE(!golden.empty());

  const int n = (int)ids.size();               // 44 (compact prefix+prompt+suffix)
  const int H = 2560;
  const int drop = 34;                          // prompt_template_encode_start_idx
  const int n_real = n - drop;                  // 10
  const std::vector<int> select = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};
  const int nL = (int)select.size();            // 12
  ASSERT_TRUE((std::size_t)golden.size() == (std::size_t)n_real * nL * H);
  // HF hidden_states[k] == the residual after layer k-1.
  std::vector<int> tap_layers;
  for (int k : select) { tap_layers.push_back(k - 1); }

  // Load the encoder (raw bf16, dense) and, separately, its embed table (to
  // host-gather the input rows -- backbone_only skips the model's embed muxer).
  auto m = MetalQwenModel::load(enc_dir, mc, krea2_encoder_config_());
  ASSERT_TRUE(m != nullptr);

  auto wts = MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer emb = wts->load("language_model.embed_tokens.weight", mc);
  ASSERT_TRUE(!emb.empty());

  // Gather the n embedding rows by id into a [n*H] bf16 stream.
  SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
  ASSERT_TRUE(!x.empty());
  const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());
  auto* xb = static_cast<std::uint8_t*>(x.contents());
  for (int i = 0; i < n; ++i) {
    const std::size_t row = (std::size_t)ids[i] * H * 2;
    std::memcpy(xb + (std::size_t)i * H * 2, tbl + row, (std::size_t)H * 2);
  }

  // Tap forward.
  ContextManager* cm = m->context_manager();
  const ContextId cid = cm->acquire_root();
  SharedBuffer taps = m->forward_embeddings_taps(cid, x, n, tap_layers);
  cm->release(cid);
  ASSERT_TRUE(!taps.empty());
  ASSERT_TRUE(taps.byte_size() >= (std::size_t)nL * n * H * 2);

  // Reorder the real rows [drop:] into the golden layout (n_real, nL, H)
  // and rel-L2 per layer + overall. taps slot j is [n][H] at j*n*H.
  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  std::vector<float> got((std::size_t)n_real * nL * H);
  for (int p = 0; p < n_real; ++p) {
    for (int j = 0; j < nL; ++j) {
      const std::size_t src = ((std::size_t)j * n + (drop + p)) * H;
      const std::size_t dst = ((std::size_t)p * nL + j) * H;
      for (int h = 0; h < H; ++h) {
        got[dst + h] = bf16_to_f32_(tp[src + h]);
      }
    }
  }

  const double overall =
      rel_l2_(got.data(), golden.data(), got.size());
  std::printf("[krea2_encoder] overall rel-L2 = %.6f\n", overall);
  for (int j = 0; j < nL; ++j) {
    double num = 0.0, den = 0.0;
    for (int p = 0; p < n_real; ++p) {
      const std::size_t o = ((std::size_t)p * nL + j) * H;
      for (int h = 0; h < H; ++h) {
        const double d = (double)got[o + h] - (double)golden[o + h];
        num += d * d;
        den += (double)golden[o + h] * (double)golden[o + h];
      }
    }
    std::printf("  layer %2d: rel-L2 = %.6f\n", select[j],
                den > 0 ? std::sqrt(num / den) : 0.0);
  }

  // Bar: golden carries ~1% bf16 noise; metal vs golden should be < 0.03.
  EXPECT_TRUE(overall < 0.03);
}

// D=128 prefill flash-attention correctness: the key-split simdgroup_matrix
// flash (sdpa_paged_flash_f16) is now enabled for head_dim 128 (it was
// Qwen3.5-D256-gated), so a LONG-prompt encoder pass (n >= 384) runs the flash
// kernel instead of the scalar per-query sdpa_paged. This A/Bs the two on the
// SAME synthetic long input: flash (default) vs scalar (VPIPE_QWEN_NO_FLASH=1),
// tapped hidden states rel-L2'd. The flash does fp32 online softmax (not
// bit-identical to the scalar), so the bar is the flash tolerance, not zero.
// Env: VPIPE_KREA2_TEST_MODEL_PATH (uses <root>/text_encoder). Skips if unset.
TEST(krea2_encoder, flash_d128_matches_scalar)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string enc_dir = std::string(root) + "/text_encoder";

  const int H = 2560;
  const int n = 2048;                           // >= 384 -> flash engages; long
                                                // enough that attention (O(n^2))
                                                // is a visible slice (the D=128
                                                // long-ctx prefill regime, e.g.
                                                // Llama-3, not just the encoder)
  const std::vector<int> tap_layers = {1, 17, 34};   // early / mid / last
  // Local config with a lifted max_seq so the synthetic long input fits (the
  // shared krea2_encoder_config_ caps at 512 for the real ~44-token prompts).
  MetalQwenModel::Config ecfg = krea2_encoder_config_();
  ecfg.max_seq = 4096;

  // Synthetic-but-deterministic token ids (attention correctness is independent
  // of whether the ids are "real" tokens; both paths see the identical input).
  std::vector<std::int32_t> ids((std::size_t)n);
  for (int i = 0; i < n; ++i) {
    ids[(std::size_t)i] = (std::int32_t)(((std::uint32_t)i * 2654435761u) % 151936u);
  }

  // On M5 (matrix cores) the "optimized" run at n>=384 auto-selects the NEW
  // matrix-core sdpa_mma_d128_f16 (use_mma_attn has priority over the flash);
  // on M4 it is the key-split simdgroup flash. Either way it is A/B'd against
  // the scalar sdpa_paged, so this test validates BOTH D=128 paths on their
  // respective hardware. Print which one the box exercises.
  std::printf("[krea2_encoder] D=128 opt path = %s (matrix_cores=%d)\n",
              mc->supports_matrix_cores() ? "matrix-core sdpa_mma_d128"
                                          : "simdgroup key-split flash",
              (int)mc->supports_matrix_cores());

  // Embed table (shared across both loads).
  auto wts = MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  SharedBuffer emb = wts->load("language_model.embed_tokens.weight", mc);
  ASSERT_TRUE(!emb.empty());
  const auto* tbl = static_cast<const std::uint8_t*>(emb.contents());

  auto gather = [&]() {
    SharedBuffer x = mc->make_shared_buffer((std::size_t)n * H * 2);
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      std::memcpy(xb + (std::size_t)i * H * 2,
                  tbl + (std::size_t)ids[(std::size_t)i] * H * 2,
                  (std::size_t)H * 2);
    }
    return x;
  };

  // Run the tap forward under the current flash setting -> host f32 stack.
  // Times the best of 3 forwards (load excluded) so the flash-vs-scalar
  // attention delta is visible (the encoder is otherwise GEMM-bound).
  auto run = [&](const char* label) -> std::vector<float> {
    auto m = MetalQwenModel::load(enc_dir, mc, ecfg);
    if (m == nullptr) { return {}; }
    SharedBuffer x = gather();
    ContextManager* cm = m->context_manager();
    std::vector<float> out;
    double best = 1e30;
    for (int it = 0; it < 3; ++it) {
      const ContextId cid = cm->acquire_root();
      const auto t0 = std::chrono::steady_clock::now();
      SharedBuffer taps = m->forward_embeddings_taps(cid, x, n, tap_layers);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      cm->release(cid);
      if (taps.empty()) { return {}; }
      if (ms < best) { best = ms; }
      if (it == 0) {
        const std::size_t ne = (std::size_t)tap_layers.size() * n * H;
        out.resize(ne);
        const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
        for (std::size_t i = 0; i < ne; ++i) { out[i] = bf16_to_f32_(tp[i]); }
      }
    }
    std::printf("[krea2_encoder] %s forward best %.1f ms (n=%d)\n", label, best,
                n);
    return out;
  };

  // Optimized run: clear both toggles -> the box picks mma (M5) or flash (M4).
  // Scalar reference: force BOTH off -- on M5 the mma attention keys off matrix
  // cores (_use_mma), NOT _flash_attn, so NO_FLASH alone would leave mma active;
  // NO_MMA is needed to actually reach the scalar sdpa_paged (it also drops the
  // GEMM to steel, but the M5 mma-GEMM is separately token-exact-verified, so
  // the residual A/B delta is dominated by the attention under test).
  ::unsetenv("VPIPE_QWEN_NO_FLASH");
  ::unsetenv("VPIPE_QWEN_NO_MMA");
  const std::vector<float> got_opt = run("opt   ");
  ASSERT_TRUE(!got_opt.empty());
  ::setenv("VPIPE_QWEN_NO_FLASH", "1", 1);
  ::setenv("VPIPE_QWEN_NO_MMA", "1", 1);
  const std::vector<float> got_scalar = run("scalar");
  ::unsetenv("VPIPE_QWEN_NO_FLASH");
  ::unsetenv("VPIPE_QWEN_NO_MMA");
  ASSERT_TRUE(!got_scalar.empty());
  ASSERT_TRUE(got_opt.size() == got_scalar.size());

  const double r =
      rel_l2_(got_opt.data(), got_scalar.data(), got_opt.size());
  // Per-tap drift stays BOUNDED across depth (no single layer blows up) -- the
  // signature of a correct kernel with fp32-online-softmax-vs-scalar noise. A
  // real bug (wrong V columns / masking) would give O(0.1-1) rel-L2 on some
  // tap, and pure error accumulation would compound with depth; instead it
  // stays ~0.005 (the residual-stream norm grows, perturbations partly cancel).
  const std::size_t per = (std::size_t)n * H;
  double worst_tap = 0.0;
  for (std::size_t j = 0; j < tap_layers.size(); ++j) {
    const double rj = rel_l2_(got_opt.data() + j * per,
                              got_scalar.data() + j * per, per);
    std::printf("[krea2_encoder] D=128 opt-vs-scalar tap L%-2d rel-L2 = %.6g\n",
                tap_layers[j], rj);
    if (rj > worst_tap) { worst_tap = rj; }
  }
  std::printf("[krea2_encoder] D=128 opt-vs-scalar overall rel-L2 = %.6g "
              "(n=%d)\n", r, n);
  // The optimized attention changes the encoding by ~1% -- far under the
  // metal-vs-golden noise floor (0.03), i.e. correctness-preserving. No tap
  // blows up. (M4 flash ~0.006; M5 mma slightly higher as it also swaps the
  // GEMM to matrix cores, hence the 1.5e-2 / 2.5e-2 headroom.)
  EXPECT_TRUE(worst_tap < 2.5e-2);
  EXPECT_TRUE(r < 1.5e-2);
}
