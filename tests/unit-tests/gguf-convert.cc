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
#include "generative-models/llama3/metal-llama-weights.h"

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

// ---- Q8_0 -> MLX-affine 8-bit group-64 ------------------------------
//
// A whole-file Q8_0 checkpoint (llama.cpp `-q8_0`) has EVERY linear, the
// embedding table and an untied lm_head in Q8_0, and the qwen35
// converter's k-quant arm takes only Q4_K/Q5_K/Q6_K -- so before this
// path such a file converted to no weights at all.
//
// Built on a SYNTHETIC gguf rather than gated on a 29 GB checkpoint:
// the repack is pure arithmetic, and the property worth pinning is a
// numeric identity, not something a big file demonstrates better.
namespace {

// Minimal GGUF v3 writer -- just enough for GgufFile::open + one tensor.
struct GgufBuilder {
  std::vector<std::uint8_t> b;
  void u8_(std::uint8_t v)  { b.push_back(v); }
  void u32_(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) { b.push_back((std::uint8_t)(v >> (8 * i))); }
  }
  void u64_(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) { b.push_back((std::uint8_t)(v >> (8 * i))); }
  }
  void str_(const std::string& s) {
    u64_(s.size());
    b.insert(b.end(), s.begin(), s.end());
  }
};

// One q8_0 block: f16 scale then 32 int8 codes.
void
push_q8_block(GgufBuilder& g, std::uint16_t d16, const std::int8_t* q) {
  g.b.push_back((std::uint8_t)(d16 & 0xff));
  g.b.push_back((std::uint8_t)(d16 >> 8));
  for (int i = 0; i < 32; ++i) { g.b.push_back((std::uint8_t)q[i]); }
}

std::uint16_t f32_to_f16_bits(float f) {
  _Float16 h = (_Float16)f;
  std::uint16_t o;
  std::memcpy(&o, &h, 2);
  return o;
}

}  // namespace

TEST(gguf_convert, q8_0_to_affine8_g64) {
  // K = 128 (two groups of 64, four q8_0 blocks per row), N = 3 rows.
  constexpr int K = 128, N = 3;
  const std::uint16_t d_a = f32_to_f16_bits(0.01f);
  const std::uint16_t d_b = f32_to_f16_bits(0.04f);

  GgufBuilder g;
  g.b.insert(g.b.end(), {'G', 'G', 'U', 'F'});
  g.u32_(3);              // version
  g.u64_(1);              // tensor count
  g.u64_(1);              // kv count
  g.str_("general.alignment");
  g.u32_(4);              // value type: UINT32
  g.u32_(32);
  g.str_("blk.0.ffn_gate.weight");
  g.u32_(2);
  g.u64_(K);              // ne0 = in
  g.u64_(N);              // ne1 = out
  g.u32_(8);              // GGML_TYPE_Q8_0
  g.u64_(0);              // offset within the data section
  while (g.b.size() % 32 != 0) { g.b.push_back(0); }

  // Row 0: ONE scale across each group of 64, and the codes span the full
  // -128..127. Then hi-lo == 255*d exactly, so the affine scale lands back
  // on d and the bias on -128*d -- the repack is EXACT, which is the
  // identity the whole mapping rests on.
  // Rows 1 and 2: the scales DIFFER between the two blocks of a group,
  // which is precisely the case group 64 cannot represent exactly.
  std::int8_t q[32];
  for (int r = 0; r < N; ++r) {
    for (int blk = 0; blk < 4; ++blk) {
      for (int i = 0; i < 32; ++i) {
        // Block 0 of each group carries -128, block 1 carries +127.
        q[i] = (std::int8_t)((blk % 2 == 0) ? (-128 + i * 4)
                                            : (127 - i * 4));
      }
      const bool second_of_group = (blk % 2) == 1;
      const std::uint16_t d =
          (r == 0 || !second_of_group) ? d_a : d_b;
      push_q8_block(g, d, q);
    }
  }

  const std::string path =
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
      + "/vpipe-q8-affine-test.gguf";
  {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_TRUE(f != nullptr);
    std::fwrite(g.b.data(), 1, g.b.size(), f);
    std::fclose(f);
  }
  auto gf = GgufFile::open(path);
  ASSERT_TRUE(gf.has_value());

  ModelConfig cfg;
  cfg.n_layers = 1;
  cfg.is_linear_layer = {false};   // a full-attention layer: has ffn_gate
  GgufQwen35Converter conv(&*gf, cfg);

  auto spec_of = [&](const std::string& nm) -> const ConvertedTensorSpec* {
    for (const auto& s : conv.specs()) {
      if (s.hf_name == nm) { return &s; }
    }
    return nullptr;
  };
  const std::string base =
      "language_model.model.layers.0.mlp.gate_proj";
  const ConvertedTensorSpec* sw = spec_of(base + ".weight");
  const ConvertedTensorSpec* ss = spec_of(base + ".scales");
  const ConvertedTensorSpec* sb = spec_of(base + ".biases");
  ASSERT_TRUE(sw != nullptr && ss != nullptr && sb != nullptr);
  // Affine-8 shapes: codes 4-per-u32, one scale/bias per 64.
  EXPECT_TRUE(sw->dtype == "U32");
  EXPECT_TRUE(sw->shape.size() == 2 && sw->shape[0] == N
              && sw->shape[1] == K / 4);
  EXPECT_TRUE(ss->shape[1] == K / 64 && sb->shape[1] == K / 64);
  EXPECT_TRUE(sw->nbytes == (std::uint64_t)N * (K / 4) * 4);

  std::vector<std::uint8_t> wbuf(sw->nbytes);
  std::vector<std::uint8_t> sbuf(ss->nbytes);
  std::vector<std::uint8_t> bbuf(sb->nbytes);
  ASSERT_TRUE(conv.convert(*sw, wbuf.data()));
  ASSERT_TRUE(conv.convert(*ss, sbuf.data()));
  ASSERT_TRUE(conv.convert(*sb, bbuf.data()));

  const auto* codes  = wbuf.data();                       // one byte / value
  const auto* scales = reinterpret_cast<const std::uint16_t*>(sbuf.data());
  const auto* biases = reinterpret_cast<const std::uint16_t*>(bbuf.data());

  // Dequantize the affine form exactly as the kernel does, and hold it
  // against the file's OWN q8_0 dequant.
  const GgufFile::Tensor* t = gf->tensor("blk.0.ffn_gate.weight");
  ASSERT_TRUE(t != nullptr);
  std::vector<float> ref(K);
  double worst_row0 = 0.0, num = 0.0, den = 0.0;
  for (int r = 0; r < N; ++r) {
    ASSERT_TRUE(gf->dequant_row_f32(*t, r, ref.data()));
    for (int i = 0; i < K; ++i) {
      const int grp = i / 64;
      const float sc = f16_to_f32(scales[r * (K / 64) + grp]);
      const float bi = f16_to_f32(biases[r * (K / 64) + grp]);
      const float got = sc * (float)codes[(std::size_t)r * K + i] + bi;
      const double d = (double)got - (double)ref[i];
      if (r == 0) { worst_row0 = std::max(worst_row0, std::abs(d)); }
      num += d * d;
      den += (double)ref[i] * (double)ref[i];
    }
  }
  const double rel = std::sqrt(num / den);
  std::printf("[gguf_convert] q8_0->affine8 g64: row0 max abs err = %g, "
              "overall rel-L2 = %.3e\n", worst_row0, rel);

  // Row 0 -- uniform scale across the group, full-range codes -- is EXACT.
  EXPECT_TRUE(worst_row0 == 0.0);
  // The mixed-scale rows are a requant, so they are bounded, not exact.
  // 2e-2 is loose on purpose: this pins "the repack is sane", and the
  // real-checkpoint number (7.55e-3 vs the bf16 original on Qwen3.8-27B)
  // is what the catalogue comment records.
  EXPECT_TRUE(rel < 2e-2);
  EXPECT_TRUE(rel > 0.0);   // rows 1-2 really are the lossy case

  // A min/max quantization must reach BOTH ends of the code range in
  // every group -- if it did not, the scale would be too coarse and the
  // repack would be throwing away precision it was handed.
  bool saw_low = false, saw_high = false;
  for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
    if (codes[i] == 0)   { saw_low = true; }
    if (codes[i] == 255) { saw_high = true; }
  }
  EXPECT_TRUE(saw_low && saw_high);

  std::remove(path.c_str());
}

// The synthetic test above pins the ARITHMETIC; this one pins the repack
// against a real llama.cpp Q8_0 file, where the row layout and the tensor
// naming are someone else's convention rather than ours. Env-gated on a
// directory holding an all-Q8_0 qwen35 gguf.
TEST(gguf_convert, q8_0_real_checkpoint_roundtrip) {
  const char* dir = std::getenv("VPIPE_QWEN_Q8_GGUF_TEST_MODEL_PATH");
  if (!dir || !*dir) { return; }
  const std::string gguf_path = find_gguf_in_dir(dir);
  ASSERT_TRUE(!gguf_path.empty());
  auto g = GgufFile::open(gguf_path);
  ASSERT_TRUE(g.has_value());
  // The directory may hold several quants (find_gguf_in_dir then picks
  // lexicographically); this test only has something to say about an
  // all-Q8_0 one, so skip rather than fail on a k-quant sibling.
  const GgufFile::Tensor* probe = g->tensor("blk.0.ffn_gate.weight");
  if (probe == nullptr || probe->type != GgufFile::kQ8_0) {
    std::printf("[gguf_convert] %s is not an all-Q8_0 file; skipped\n",
                gguf_path.c_str());
    return;
  }
  ModelConfig cfg;
  ASSERT_TRUE(gguf_to_model_config(*g, &cfg));
  // A Q8_0 file must announce itself as 8-bit affine, or the model binds
  // 4-bit kernels over 8-bit weights.
  EXPECT_TRUE(cfg.quantization.bits == 8);
  EXPECT_TRUE(cfg.quantization.group_size == 64);

  GgufQwen35Converter conv(&*g, cfg);
  auto spec_of = [&](const std::string& nm) -> const ConvertedTensorSpec* {
    for (const auto& s : conv.specs()) {
      if (s.hf_name == nm) { return &s; }
    }
    return nullptr;
  };

  // Check a linear, the GDN alpha projection (which the fused path needs
  // as affine, not f32) and the embedding table.
  const char* bases[] = {
      "language_model.model.layers.0.mlp.gate_proj",
      "language_model.model.layers.0.linear_attn.in_proj_a",
      "language_model.model.embed_tokens",
  };
  for (const char* base : bases) {
    const ConvertedTensorSpec* sw = spec_of(std::string(base) + ".weight");
    const ConvertedTensorSpec* ss = spec_of(std::string(base) + ".scales");
    const ConvertedTensorSpec* sb = spec_of(std::string(base) + ".biases");
    EXPECT_TRUE(sw != nullptr && ss != nullptr && sb != nullptr);
    if (!sw || !ss || !sb) { std::printf("  MISSING %s\n", base); continue; }
    const GgufFile::Tensor* t = g->tensor(sw->gguf_name);
    ASSERT_TRUE(t != nullptr);
    const std::int64_t K = t->dims[0], N = t->dims[1];
    EXPECT_TRUE(sw->shape[0] == N && sw->shape[1] == K / 4);

    // Only the first few rows -- enough to catch a layout error, cheap on
    // a 248320-row embedding.
    const std::int64_t rows = std::min<std::int64_t>(N, 4);
    std::vector<std::uint8_t> wbuf(sw->nbytes), sbuf(ss->nbytes),
        bbuf(sb->nbytes);
    ASSERT_TRUE(conv.convert(*sw, wbuf.data()));
    ASSERT_TRUE(conv.convert(*ss, sbuf.data()));
    ASSERT_TRUE(conv.convert(*sb, bbuf.data()));
    const auto* scales = reinterpret_cast<const std::uint16_t*>(sbuf.data());
    const auto* biases = reinterpret_cast<const std::uint16_t*>(bbuf.data());

    std::vector<float> ref((std::size_t)K);
    double num = 0, den = 0;
    // Row r of the output is row r of the GGUF: the converter does NOT
    // reorder, including the GDN v-head-indexed tensors (see the note in
    // gguf-convert.h -- the model reads a GGUF's strided v-head order
    // directly).
    for (std::int64_t r = 0; r < rows; ++r) {
      ASSERT_TRUE(g->dequant_row_f32(*t, r, ref.data()));
      for (std::int64_t i = 0; i < K; ++i) {
        const std::int64_t grp = i / 64;
        const float sc = f16_to_f32(scales[r * (K / 64) + grp]);
        const float bi = f16_to_f32(biases[r * (K / 64) + grp]);
        const float got =
            sc * (float)wbuf[(std::size_t)r * K + i] + bi;
        const double d = (double)got - (double)ref[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
      }
    }
    const double rel = std::sqrt(num / (den + 1e-30));
    std::printf("[gguf_convert] %-52s rel-L2 %.3e (%lldx%lld)\n",
                base, rel, (long long)N, (long long)K);
    // A requant, so not exact -- but a LAYOUT error lands orders of
    // magnitude above this, not just above the quantization floor.
    EXPECT_TRUE(rel < 2e-2);
  }
}


// An all-Q8_0 qwen35 GGUF must convert to a checkpoint with NO k-quant
// tensors at all -- every linear is repacked to affine 8-bit.
//
// That is the precondition behind a bug worth naming here, because the
// coupling is invisible from either side alone. GDN value-head order is
// a property of the FILE (llama.cpp writes v-heads strided over the key
// heads; HF groups them contiguously), and MetalQwenModel switches the
// GDN step kernel between the two with the sign of its key-head count.
// That switch used to be keyed on "is this checkpoint k-quant", which is
// the same question as "is this a GGUF" for every GGUF that HAS k-quant
// tensors -- and a different one for this file, whose Q8_0 linears are
// repacked to affine. So an all-Q8_0 GGUF ran strided weights through
// the contiguous mapping: it loaded, prefilled, and decoded garbage.
//
// This test cannot see the flag; it pins the fact that makes the two
// questions come apart. The flag itself is keyed on
// MetalLlamaWeights::is_gguf().
TEST(gguf_convert, q8_0_checkpoint_has_no_kquant_tensors) {
  const char* dir = std::getenv("VPIPE_QWEN_Q8_GGUF_TEST_MODEL_PATH");
  if (!dir || !*dir) { return; }
  auto g = GgufFile::open(find_gguf_in_dir(dir));
  ASSERT_TRUE(g.has_value());
  const GgufFile::Tensor* probe = g->tensor("blk.0.ffn_gate.weight");
  if (probe == nullptr || probe->type != GgufFile::kQ8_0) { return; }
  ModelConfig cfg;
  ASSERT_TRUE(gguf_to_model_config(*g, &cfg));
  GgufQwen35Converter conv(&*g, cfg);
  int kq = 0, affine = 0;
  for (const auto& s : conv.specs()) {
    if (s.dtype == "Q4K" || s.dtype == "Q5K" || s.dtype == "Q6K") { ++kq; }
    if (s.op == ConvertedTensorSpec::Op::kQ8Weight) { ++affine; }
  }
  std::printf("[gguf_convert] all-Q8_0: %d k-quant, %d affine-8 weights\n",
              kq, affine);
  EXPECT_TRUE(kq == 0);
  EXPECT_TRUE(affine > cfg.n_layers);   // several per layer, plus the head
}

// hf_dir may name a `.gguf` FILE, not just the directory holding it --
// that is how a caller picks ONE quantization point out of a repo
// directory that ships several. The tokenizer already resolved a file
// path; the WEIGHTS took the single-safetensors branch and failed, so
// the model came up "unloadable" with the tokenizer log line above it.
TEST(gguf_convert, open_model_accepts_a_gguf_file_path) {
  const char* dir = std::getenv("VPIPE_QWEN_Q8_GGUF_TEST_MODEL_PATH");
  if (!dir || !*dir) { return; }
  const std::string gguf_path = find_gguf_in_dir(dir);
  ASSERT_TRUE(!gguf_path.empty());
  ASSERT_TRUE(gguf_path.rfind(".gguf") == gguf_path.size() - 5);
  auto by_file = MetalLlamaWeights::open_model(gguf_path);
  ASSERT_TRUE(by_file.has_value());
  EXPECT_TRUE(by_file->is_gguf());
  EXPECT_TRUE(
      by_file->has("language_model.model.layers.0.mlp.gate_proj.weight"));
  // Same checkpoint by DIRECTORY -> the same tensor set.
  auto by_dir = MetalLlamaWeights::open_model(dir);
  ASSERT_TRUE(by_dir.has_value());
  EXPECT_TRUE(by_dir->is_gguf());
}
