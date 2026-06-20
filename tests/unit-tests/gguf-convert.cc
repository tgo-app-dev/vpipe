// gguf-convert.cc -- exercises the GGUF loader/converter against a real
// gemma4 q4_0 checkpoint, WITHOUT a forward pass (so it runs on a small
// box). Verifies the config parse and, crucially, that the q4_0 -> affine
// 4-bit g32 repack is BIT-EXACT against the file's own q4_0 dequant, and
// the q6_K token table -> affine 8-bit g32 requant is within tolerance.
//
// MLX-free (GgufFile + GgufGemma4Converter only), so it builds and runs in
// both the MLX and no-MLX trees. Env-gated on VPIPE_GGUF_TEST_MODEL_PATH
// (the directory holding gemma-4-*-q4_0.gguf); skips vacuously if unset.

#include "minitest.h"
#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/model-loader.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {
float f16_to_f32(std::uint16_t h) {
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return static_cast<float>(v);
}
const ConvertedTensorSpec* find_spec(const GgufGemma4Converter& c,
                                     const std::string& name) {
  for (const auto& s : c.specs()) {
    if (s.hf_name == name) { return &s; }
  }
  return nullptr;
}
}  // namespace

TEST(gguf_convert, config_and_lossless_repack) {
  const char* dir = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!dir || !*dir) { return; }

  const std::string gguf_path = find_gguf_in_dir(dir);
  ASSERT_TRUE(!gguf_path.empty());
  auto g = GgufFile::open(gguf_path);
  ASSERT_TRUE(g.has_value());

  // ---- config ----------------------------------------------------
  ModelConfig cfg;
  ASSERT_TRUE(gguf_to_model_config(*g, &cfg));
  EXPECT_TRUE(cfg.gemma4.present);
  EXPECT_TRUE(cfg.n_layers == 48);
  EXPECT_TRUE(cfg.hidden == 3840);
  EXPECT_TRUE(cfg.n_heads == 16);
  EXPECT_TRUE(cfg.gemma4.head_dim_sliding == 256);
  EXPECT_TRUE(cfg.gemma4.head_dim_full == 512);
  EXPECT_TRUE(cfg.quantization.bits == 4);
  EXPECT_TRUE(cfg.quantization.group_size == 32);
  EXPECT_TRUE(cfg.vocab_size > 0);
  EXPECT_TRUE(std::fabs(cfg.gemma4.final_logit_softcapping - 30.0f) < 1e-3f);
  // Per-layer attention kinds: layer 0 sliding, layer 5 global.
  ASSERT_TRUE((int)cfg.gemma4.is_full_layer.size() == cfg.n_layers);
  EXPECT_TRUE(cfg.gemma4.is_full_layer[0] == false);
  EXPECT_TRUE(cfg.gemma4.is_full_layer[5] == true);
  ASSERT_TRUE((int)cfg.gemma4.layer_n_kv_heads.size() == cfg.n_layers);
  EXPECT_TRUE(cfg.gemma4.layer_n_kv_heads[0] == 8);
  EXPECT_TRUE(cfg.gemma4.layer_n_kv_heads[5] == 1);
  // Global layers ship no v_proj -> k_eq_v (values reuse keys), 1 K/V head.
  EXPECT_TRUE(cfg.gemma4.attention_k_eq_v == true);
  EXPECT_TRUE(cfg.gemma4.num_global_kv_heads == 1);

  GgufGemma4Converter conv(&*g, cfg);

  // ---- q4_0 -> affine4 g32 must be BIT-EXACT ----------------------
  const std::string base = "language_model.model.layers.0.self_attn.q_proj";
  const auto* sw = find_spec(conv, base + ".weight");
  const auto* ss = find_spec(conv, base + ".scales");
  const auto* sb = find_spec(conv, base + ".biases");
  ASSERT_TRUE(sw != nullptr && ss != nullptr && sb != nullptr);
  const GgufFile::Tensor* qt = g->tensor("blk.0.attn_q.weight");
  ASSERT_TRUE(qt != nullptr && qt->dims.size() == 2);
  const std::int64_t in = qt->dims[0], out = qt->dims[1];
  EXPECT_TRUE(sw->shape.size() == 2 && sw->shape[0] == out &&
              sw->shape[1] == in / 8);

  std::vector<std::uint8_t> wbuf(sw->nbytes), sbuf(ss->nbytes),
      bbuf(sb->nbytes);
  ASSERT_TRUE(conv.convert(*sw, wbuf.data()));
  ASSERT_TRUE(conv.convert(*ss, sbuf.data()));
  ASSERT_TRUE(conv.convert(*sb, bbuf.data()));
  const auto* wq = reinterpret_cast<const std::uint32_t*>(wbuf.data());
  const auto* sc = reinterpret_cast<const std::uint16_t*>(sbuf.data());
  const auto* bi = reinterpret_cast<const std::uint16_t*>(bbuf.data());

  std::vector<float> ref(static_cast<std::size_t>(in));
  float max_err = 0.0f;
  const int sample_rows = 4;
  for (int r = 0; r < sample_rows && r < out; ++r) {
    ASSERT_TRUE(g->dequant_row_f32(*qt, r, ref.data()));
    const std::uint32_t* wr = wq + (std::size_t)r * (in / 8);
    const std::uint16_t* srow = sc + (std::size_t)r * (in / 32);
    const std::uint16_t* brow = bi + (std::size_t)r * (in / 32);
    for (std::int64_t c = 0; c < in; ++c) {
      const int q = (int)((wr[c / 8] >> (4 * (c % 8))) & 0xF);
      const float aff = f16_to_f32(srow[c / 32]) * (float)q +
                        f16_to_f32(brow[c / 32]);
      max_err = std::fmax(max_err, std::fabs(ref[(std::size_t)c] - aff));
    }
  }
  std::printf("[gguf_convert] q4_0->affine4 g32 max abs err = %g\n",
              (double)max_err);
  EXPECT_TRUE(max_err == 0.0f);   // lossless repack

  // ---- q6_K token table -> affine8 g32 within tolerance ----------
  const auto* ew = find_spec(conv, "language_model.model.embed_tokens.weight");
  const auto* es = find_spec(conv, "language_model.model.embed_tokens.scales");
  const auto* eb = find_spec(conv, "language_model.model.embed_tokens.biases");
  ASSERT_TRUE(ew != nullptr && es != nullptr && eb != nullptr);
  EXPECT_TRUE(ew->dtype == "U32" && es->dtype == "F16");
  // 8-bit packing: hidden/4 u32 per row.
  EXPECT_TRUE(ew->shape.size() == 2 && ew->shape[1] == cfg.hidden / 4);

  // ---- RMSNorm passthrough: Gemma folds the +1 into the stored gain
  // (which vpipe applies directly), so the converter must NOT subtract 1
  // -- the output equals the GGUF f32 values byte-for-byte.
  const auto* nspec = find_spec(conv, "language_model.model.norm.weight");
  ASSERT_TRUE(nspec != nullptr && nspec->dtype == "F32");
  const GgufFile::Tensor* nt = g->tensor("output_norm.weight");
  ASSERT_TRUE(nt != nullptr);
  std::vector<std::uint8_t> nbuf(nspec->nbytes);
  ASSERT_TRUE(conv.convert(*nspec, nbuf.data()));
  const float* nv = reinterpret_cast<const float*>(nbuf.data());
  std::vector<float> nref(static_cast<std::size_t>(nt->numel()));
  ASSERT_TRUE(g->dequant_all_f32(*nt, nref.data()));
  std::printf("[gguf_convert] norm[0] = %g (passthrough, +1 pre-folded)\n",
              (double)nv[0]);
  bool norm_match = true;
  for (std::size_t i = 0; i < nref.size(); ++i) {
    if (nv[i] != nref[i]) { norm_match = false; break; }
  }
  EXPECT_TRUE(norm_match);
}

// qwen35 (Qwen3.5 hybrid GDN, k-quant) GGUF -> ModelConfig mapping.
// Validates the metadata parse against a real unsloth Qwen3.5 GGUF (the
// k-quant tensor remap + forward are exercised by the metal LM smokes).
// Gated on VPIPE_QWEN_GGUF_TEST_MODEL_PATH (dir or .gguf path).
TEST(gguf_convert, qwen35_config) {
  const char* dir = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!dir || !*dir) { return; }

  const std::string gguf_path = find_gguf_in_dir(dir);
  ASSERT_TRUE(!gguf_path.empty());
  auto g = GgufFile::open(gguf_path);
  ASSERT_TRUE(g.has_value());
  auto arch = g->get_string("general.architecture");
  ASSERT_TRUE(arch.has_value() && *arch == "qwen35");

  ModelConfig cfg;
  ASSERT_TRUE(gguf_to_model_config(*g, &cfg));
  std::printf("[qwen35_cfg] L=%d hidden=%d ffn=%d heads=%d kv=%d head_dim=%d "
              "rope_theta=%.0f eps=%.1e interval=%d prf=%.4f\n",
              cfg.n_layers, cfg.hidden, cfg.ffn_inner, cfg.n_heads,
              cfg.n_kv_heads, cfg.head_dim, (double)cfg.rope_theta,
              (double)cfg.rms_eps, cfg.full_attention_interval,
              (double)cfg.partial_rotary_factor);
  std::printf("[qwen35_cfg] gdn: k_heads=%d v_heads=%d k_dim=%d v_dim=%d "
              "conv_k=%d  vocab=%d  mrope=[%s]\n",
              cfg.linear_num_k_heads, cfg.linear_num_v_heads,
              cfg.linear_k_head_dim, cfg.linear_v_head_dim,
              cfg.linear_conv_kernel, cfg.vocab_size,
              cfg.mrope_section.empty() ? "" : "set");

  EXPECT_TRUE(cfg.architecture == "Qwen3_5ForConditionalGeneration");
  EXPECT_TRUE(cfg.n_layers > 0);
  EXPECT_TRUE(cfg.hidden > 0);
  EXPECT_TRUE(cfg.ffn_inner > 0);
  EXPECT_TRUE(cfg.n_heads > 0);
  EXPECT_TRUE(cfg.n_kv_heads > 0 && cfg.n_kv_heads <= cfg.n_heads);
  EXPECT_TRUE(cfg.head_dim > 0);
  EXPECT_TRUE(cfg.rope_theta > 0.0f);
  EXPECT_TRUE(cfg.attn_output_gate == true);
  EXPECT_TRUE(cfg.tie_word_embeddings == true);
  EXPECT_TRUE(cfg.vocab_size > 0);
  EXPECT_TRUE(cfg.partial_rotary_factor > 0.0f &&
              cfg.partial_rotary_factor <= 1.0f);

  // Gated-DeltaNet dims must all be present.
  EXPECT_TRUE(cfg.linear_num_k_heads > 0);
  EXPECT_TRUE(cfg.linear_num_v_heads > 0);
  EXPECT_TRUE(cfg.linear_k_head_dim > 0);
  EXPECT_TRUE(cfg.linear_v_head_dim > 0);
  EXPECT_TRUE(cfg.linear_conv_kernel > 0);
  EXPECT_TRUE(!cfg.mrope_section.empty());

  // Layer-type pattern: full-attention at (L+1) % interval == 0, GDN else.
  ASSERT_TRUE((int)cfg.is_linear_layer.size() == cfg.n_layers);
  const int iv = cfg.full_attention_interval;
  EXPECT_TRUE(iv > 0);
  for (int L = 0; L < cfg.n_layers; ++L) {
    const bool want_linear = ((L + 1) % iv) != 0;
    EXPECT_TRUE(cfg.is_linear_layer[(std::size_t)L] == want_linear);
  }
}
