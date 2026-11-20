// gguf-qwen4exp.cc -- llama.cpp's block formats (ggml-blocks.h) and the
// Qwen3.8-Flash-Next GGUF converter (gguf-qwen4exp.h).
//
//   ggml_blocks.*: model-free. Every block format's affine form at every
//   width it takes is the decoded value, exactly (random blocks).
//
//   qwen4exp_gguf.*: VPIPE_QWEN4EXP_GGUF = a variant's directory (or its
//   first .gguf). The converter's tensors against the file's own decode:
//   the w8 re-encode within its rounding, the v-heads in the release's
//   order, a routed expert's cache layout the decoded rows at the compute
//   dtype's scale rounding. VPIPE_QWEN4EXP_GGUF_DUMP = a directory: the
//   converted tensors dequantized to f32 there (<hf name>.f32 + .shape),
//   for a comparison against another pack of the same release.

#include "minitest.h"
#include "common/session.h"
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/tokenizer.h"
#include "generative-models/shared/ggml-blocks.h"
#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-file.h"
#include "generative-models/shared/gguf-qwen4exp.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/model-loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

void
put_f16_(std::uint8_t* p, float f)
{
  const _Float16 h = (_Float16)f;
  std::memcpy(p, &h, 2);
}

// The f16 fields of a block (scale, min) at their byte offsets.
std::vector<int>
f16_fields_(std::uint32_t t)
{
  switch (t) {
    case ggml::kQ5_1: return {0, 2};
    case ggml::kQ4_K:
    case ggml::kQ5_K: return {0, 2};
    case ggml::kQ3_K: return {108};
    case ggml::kQ6_K: return {208};
    case ggml::kIQ1_M: return {};
    default: return {0};
  }
}

// Code j of a row of `bytes` bytes (the second byte read only inside it).
unsigned
code_at_(const std::uint8_t* row, int bits, int j, std::size_t bytes)
{
  const std::size_t bit = (std::size_t)bits * j;
  unsigned v = row[bit >> 3];
  if ((bit >> 3) + 1 < bytes) { v |= (unsigned)row[(bit >> 3) + 1] << 8; }
  return (v >> (bit & 7)) & ((1u << bits) - 1u);
}

float
bf16_round_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  float r;
  std::memcpy(&r, &u, 4);
  return r;
}

float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A format whose values are its scale times an integer (no free bias).
bool
symmetric_(std::uint32_t t)
{
  return t != ggml::kQ4_K && t != ggml::kQ5_K && t != ggml::kQ5_1;
}

// Where `t` at `bits` sits at bf16's own floor: an exact form whose bias
// is -off * s' with off a power of two. IQ3_S at 4 bits is 2 db (code -
// 7.5): its bias -7.5 s' rounds as a fitted one does (Q4_K's class).
bool
at_floor_(std::uint32_t t, int bits)
{
  return ggml::affine_bits(t) > 0 && bits >= ggml::affine_bits(t) &&
         symmetric_(t) && !(t == ggml::kIQ3_S && bits == 4);
}

std::string
gguf_path_()
{
  const char* e = std::getenv("VPIPE_QWEN4EXP_GGUF");
  return e == nullptr ? std::string() : find_gguf_in_dir(e);
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  double n = 0, d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    n += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    d += (double)b[i] * b[i];
  }
  return d > 0 ? std::sqrt(n / d) : std::sqrt(n);
}

}  // namespace

// Random blocks of every format the converter reads: each format's affine
// form, at each width from its own to 8, is the value ggml decodes.
TEST(ggml_blocks, affine_forms_are_the_decoded_values)
{
  const std::uint32_t types[] = {
      ggml::kQ4_0, ggml::kQ5_0, ggml::kQ5_1, ggml::kQ8_0, ggml::kIQ4_NL,
      ggml::kQ3_K, ggml::kQ4_K, ggml::kQ5_K, ggml::kIQ3_XXS, ggml::kIQ3_S,
      ggml::kIQ4_XS, ggml::kIQ2_XXS};
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> scale(1e-3f, 5e-2f);
  for (const std::uint32_t t : types) {
    const ggml::Geom g = ggml::geom(t);
    const int nb = 64, n = nb * g.elems;
    std::vector<std::uint8_t> blk((std::size_t)nb * g.bytes);
    for (auto& b : blk) { b = (std::uint8_t)(rng() & 0xff); }
    for (int i = 0; i < nb; ++i) {
      for (int at : f16_fields_(t)) {
        // A K-quant's min is subtracted; any sign, as files hold.
        const float v = scale(rng) * ((rng() & 1) && at != 0 ? -1 : 1);
        put_f16_(blk.data() + (std::size_t)i * g.bytes + at, v);
      }
    }
    std::vector<float> y((std::size_t)n);
    ASSERT_TRUE(ggml::decode(t, blk.data(), n, y.data()));
    const int need = ggml::affine_bits(t);
    ASSERT_TRUE(need > 0);
    for (int bits = need; bits <= 8; ++bits) {
      std::vector<std::uint8_t> c((std::size_t)n);
      std::vector<ggml::Affine32> a((std::size_t)n / 32);
      ASSERT_TRUE(ggml::to_affine32(t, blk.data(), n, bits, c.data(),
                                    a.data()));
      std::vector<std::uint8_t> packed((std::size_t)n * bits / 8 + 1);
      ggml::pack_codes(c.data(), n, bits, packed.data());
      double worst = 0;
      int unpacked_bad = 0;
      for (int j = 0; j < n; ++j) {
        const ggml::Affine32& f = a[(std::size_t)j / 32];
        if (code_at_(packed.data(), bits, j, packed.size()) !=
            c[(std::size_t)j]) {
          ++unpacked_bad;
        }
        const float bias = f.off != 0.0f ? -f.off * f.scale : f.bias;
        const float v = f.scale * (float)c[(std::size_t)j] + bias;
        worst = std::max(worst, (double)std::fabs(v - y[(std::size_t)j]));
        if (f.off != 0.0f && std::fabs(bias - f.bias) > 0.0f) {
          worst = std::max(worst, 1.0);
        }
      }
      std::printf("[ggml] %-8s w%d: worst |affine - decoded| %.3g\n",
                  ggml::name(t), bits, worst);
      ASSERT_TRUE(unpacked_bad == 0);
      ASSERT_TRUE(worst == 0.0);
    }
    if (need > 4) {
      std::vector<std::uint8_t> c((std::size_t)n);
      std::vector<ggml::Affine32> a((std::size_t)n / 32);
      ASSERT_FALSE(ggml::to_affine32(t, blk.data(), n, need - 1, c.data(),
                                     a.data()));
    }
  }
}

// A row through row_affine32, as the converters take it: an exact form
// at bf16's floor (a power-of-two offset) or a fitted bias's; a format
// with no group-32 form (a scale a 16 values: IQ2_XS, IQ2_S, IQ1_M, Q6_K)
// rounded to 8 bits over each group's range, and refused narrower.
TEST(ggml_blocks, rows_take_the_exact_form_or_round)
{
  const std::uint32_t types[] = {
      ggml::kIQ2_XXS, ggml::kIQ3_XXS, ggml::kIQ3_S, ggml::kIQ4_XS,
      ggml::kQ5_K, ggml::kQ4_K, ggml::kIQ2_XS, ggml::kIQ2_S,
      ggml::kIQ1_M, ggml::kQ6_K};
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> scale(1e-3f, 5e-2f);
  for (const std::uint32_t t : types) {
    const ggml::Geom g = ggml::geom(t);
    const int nb = 64, n = nb * g.elems;
    std::vector<std::uint8_t> blk((std::size_t)nb * g.bytes);
    for (auto& b : blk) { b = (std::uint8_t)(rng() & 0xff); }
    for (int i = 0; i < nb; ++i) {
      std::uint8_t* bp = blk.data() + (std::size_t)i * g.bytes;
      for (int at : f16_fields_(t)) {
        const float v = scale(rng) * ((rng() & 1) && at != 0 ? -1 : 1);
        put_f16_(bp + at, v);
      }
      if (t == ggml::kIQ1_M) {
        // its f16 scale is the top nibble of each of its four scale words
        std::uint8_t h[2];
        put_f16_(h, scale(rng));
        const unsigned u = (unsigned)h[0] | (unsigned)h[1] << 8;
        for (int w = 0; w < 4; ++w) {
          bp[48 + 2 * w + 1] = (std::uint8_t)((bp[48 + 2 * w + 1] & 0x0f) |
                                              ((u >> (4 * w)) & 0xf) << 4);
        }
      }
    }
    std::vector<float> y((std::size_t)n);
    ASSERT_TRUE(ggml::decode(t, blk.data(), n, y.data()));
    bool finite = true;
    for (float v : y) { finite = finite && std::isfinite(v); }
    ASSERT_TRUE(finite);
    const int w = ggml::affine_width(t);
    std::vector<std::uint8_t> c((std::size_t)n);
    std::vector<std::uint16_t> sc((std::size_t)n / 32), bi(sc.size());
    ASSERT_TRUE(ggml::row_affine32(t, blk.data(), n, w, true, c.data(),
                                   sc.data(), bi.data()));
    std::vector<float> got((std::size_t)n), ref16 = y;
    for (int j = 0; j < n; ++j) {
      got[(std::size_t)j] = bf16_f32_(sc[(std::size_t)j / 32]) *
                                (float)c[(std::size_t)j] +
                            bf16_f32_(bi[(std::size_t)j / 32]);
    }
    for (float& f : ref16) { f = bf16_round_(f); }
    const double e = rel_l2_(got, y), e16 = rel_l2_(ref16, y);
    const bool floor = at_floor_(t, w);
    std::printf("[ggml] %-8s row at w%d: %.2e from the decode (bf16 %.2e)"
                "%s\n", ggml::name(t), w, e, e16,
                ggml::affine_bits(t) > 0 ? "" : " -- rounded, no form");
    ASSERT_TRUE(floor ? e < 1.5 * e16 : e < 8e-3);
    if (ggml::affine_bits(t) == 0) {
      ASSERT_TRUE(w == 8);
      ASSERT_FALSE(ggml::row_affine32(t, blk.data(), n, 6, true, c.data(),
                                      sc.data(), bi.data()));
    }
  }
}

// The converter's tensors against the file's decode.
TEST(qwen4exp_gguf, converted_tensors_are_the_files)
{
  const std::string path = gguf_path_();
  if (path.empty()) {
    std::printf("[skip] VPIPE_QWEN4EXP_GGUF unset\n");
    return;
  }
  auto g = GgufFile::open(path);
  ASSERT_TRUE(g.has_value());
  if (!g) { return; }
  std::printf("[qwen4exp] %s: %zu files, %zu tensors\n", path.c_str(),
              g->file_count(), g->tensors().size());
  ModelConfig cfg;
  ASSERT_TRUE(gguf_to_model_config(*g, &cfg));
  ASSERT_TRUE(cfg.architecture == "Qwen4ExpForConditionalGeneration");
  GgufQwen4ExpConverter conv(&*g);
  std::printf("[qwen4exp] experts: gate|up w%d, down w%d; %zu tensors "
              "presented\n", conv.gate_bits(), conv.down_bits(),
              conv.specs().size());
  ASSERT_TRUE(conv.gate_bits() > 0 && conv.down_bits() > 0);
  const char* dump = std::getenv("VPIPE_QWEN4EXP_GGUF_DUMP");
  auto spec = [&](const std::string& nm) -> const ConvertedTensorSpec* {
    for (const auto& s : conv.specs()) {
      if (s.hf_name == nm) { return &s; }
    }
    return nullptr;
  };
  auto raw = [&](const std::string& nm) {
    const ConvertedTensorSpec* s = spec(nm);
    std::vector<std::uint8_t> b(s != nullptr ? s->nbytes : 0);
    if (s == nullptr || !conv.convert(*s, b.data())) { b.clear(); }
    return b;
  };
  // A w8 projection, dequantized: [N, K].
  auto w8 = [&](const std::string& base, std::int64_t& N, std::int64_t& K) {
    const ConvertedTensorSpec* s = spec(base + ".weight");
    std::vector<float> out;
    if (s == nullptr) { return out; }
    N = s->shape[0];
    K = s->shape[1] * 4;
    const auto w = raw(base + ".weight");
    const auto sc = raw(base + ".scales");
    const auto bi = raw(base + ".biases");
    if (w.empty() || sc.empty() || bi.empty()) { return out; }
    const auto* sp = reinterpret_cast<const std::uint16_t*>(sc.data());
    const auto* bp = reinterpret_cast<const std::uint16_t*>(bi.data());
    out.resize((std::size_t)(N * K));
    for (std::int64_t r = 0; r < N; ++r) {
      for (std::int64_t j = 0; j < K; ++j) {
        const std::size_t gi = (std::size_t)(r * (K / 32) + j / 32);
        out[(std::size_t)(r * K + j)] =
            bf16_f32_(sp[gi]) * (float)w[(std::size_t)(r * K + j)] +
            bf16_f32_(bp[gi]);
      }
    }
    return out;
  };
  auto save = [&](const std::string& nm, const std::vector<float>& v,
                  std::vector<std::int64_t> shape) {
    if (dump == nullptr) { return; }
    std::ofstream(std::string(dump) + "/" + nm + ".f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(v.data()),
               (std::streamsize)(v.size() * 4));
    std::ofstream sh(std::string(dump) + "/" + nm + ".shape");
    for (auto d : shape) { sh << d << " "; }
  };
  // The file's own decode of a 2-D tensor: [N, K].
  auto file_rows = [&](const std::string& gn) {
    const GgufFile::Tensor* t = g->tensor(gn);
    std::vector<float> v;
    if (t == nullptr) { return v; }
    v.resize((std::size_t)t->numel());
    const std::int64_t K = t->dims[0];
    const std::int64_t N = t->dims.size() > 1 ? t->dims[1] : 1;
    for (std::int64_t r = 0; r < N; ++r) {
      if (!g->dequant_row_f32(*t, r, v.data() + r * K)) { v.clear(); break; }
    }
    return v;
  };
  const int nk = 16, nv = 48, r3 = nv / nk;
  auto vsrc = [&](int h) { return (h % r3) * nk + h / r3; };
  const std::string P = "language_model.model.layers.";
  struct Case {
    std::string hf, gguf;
    int vrow0 = -1, vunit = 0;
    bool vcols = false;
  };
  const std::vector<Case> cases = {
      {P + "1.linear_attn.in_proj_qkv", "blk.1.attn_qkv.weight", 4096, 128},
      {P + "1.linear_attn.in_proj_z", "blk.1.attn_gate.weight", 0, 128},
      {P + "1.linear_attn.in_proj_a", "blk.1.ssm_alpha.weight", 0, 1},
      {P + "1.linear_attn.out_proj", "blk.1.ssm_out.weight", -1, 128, true},
      {P + "1.attn_hyper_connection.input_mix_weight_down",
       "blk.1.hc_attn_down.weight"},
      {P + "1.attn_hyper_connection.input_mix_weight_up",
       "blk.1.hc_attn_up.weight"},
      {P + "1.attn_hyper_connection.block_inject_weight",
       "blk.1.hc_attn_inject.weight"},
      {P + "1.mlp.shared_expert.down_proj", "blk.1.ffn_down_shexp.weight"},
      {P + "1.ple.key_proj", "blk.1.ple_key.weight"},
      {P + "3.self_attn.q_proj", "blk.3.attn_q.weight"},
      {P + "3.self_attn.v_proj", "blk.3.attn_v.weight"},
      {P + "3.self_attn.o_proj", "blk.3.attn_output.weight"},
      {"language_model.lm_head", "output.weight"},
  };
  for (const Case& cs : cases) {
    std::int64_t N = 0, K = 0;
    const std::vector<float> got = w8(cs.hf, N, K);
    std::vector<float> ref = file_rows(cs.gguf);
    ASSERT_TRUE(!got.empty() && got.size() == ref.size());
    if (got.empty() || got.size() != ref.size()) { continue; }
    if (cs.vunit > 0) {
      const std::vector<float> src = ref;
      const int U = cs.vunit;
      if (cs.vcols) {
        for (std::int64_t r = 0; r < N; ++r) {
          for (int h = 0; h < nv; ++h) {
            std::memcpy(&ref[(std::size_t)(r * K + h * U)],
                        &src[(std::size_t)(r * K + vsrc(h) * U)],
                        (std::size_t)U * 4);
          }
        }
      } else {
        for (int h = 0; h < nv; ++h) {
          std::memcpy(&ref[(std::size_t)((cs.vrow0 + h * U) * K)],
                      &src[(std::size_t)((cs.vrow0 + vsrc(h) * U) * K)],
                      (std::size_t)U * K * 4);
        }
      }
    }
    const double e = rel_l2_(got, ref);
    std::vector<float> ref16 = ref;
    for (float& f : ref16) { f = bf16_round_(f); }
    const double e16 = rel_l2_(ref16, ref);
    const std::uint32_t ty = g->tensor(cs.gguf)->type;
    std::printf("[qwen4exp] %-58s [%lld, %lld] %s: w8 vs file %.2e "
                "(each value in bf16 %.2e)\n", cs.hf.c_str(), (long long)N,
                (long long)K, ggml::name(ty), e, e16);
    // A symmetric format's exact form moves every value by its group's
    // scale rounding alone: bf16's own error. A free bias (the K-quants,
    // Q5_1) about 15 steps out rounds too; re-encoded (Q6_K, the float
    // sources), 8-bit codes over a 32-value group's range: each ~4-5e-3.
    // (A mapping or ordering error reads ~1.)
    ASSERT_TRUE(at_floor_(ty, 8) ? e < 1.5 * e16 : e < 6e-3);
    save(cs.hf, got, {N, K});
  }
  // Float tensors: as stored, the zero-centred norms less one, the heads
  // in the release's order, ssm_a as A_log.
  for (const std::string& nm :
       {P + "1.linear_attn.dt_bias", P + "1.linear_attn.A_log",
        P + "1.linear_attn.conv1d.weight", P + "1.linear_attn.norm.weight",
        P + "1.attn_hyper_connection.hc_norm.weight",
        P + "3.self_attn.q_norm.weight", P + "1.mlp.gate.weight",
        P + "1.ple.conv1d.weight", P + "1.ple.norm_key.weight",
        P + "3.self_attn.indexer.q_layernorm.weight"}) {
    const auto b = raw(nm);
    ASSERT_TRUE(!b.empty());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
    save(nm, v, spec(nm)->shape);
  }
  {
    std::int64_t N = 0, K = 0;
    save(P + "3.self_attn.indexer.index_qk_proj",
         w8(P + "3.self_attn.indexer.index_qk_proj", N, K), {N, K});
    save("language_model.model.embed_tokens.rows0-4096",
         [&] {
           auto v = w8("language_model.model.embed_tokens", N, K);
           v.resize((std::size_t)(4096 * K));
           return v;
         }(),
         {4096, K});
  }
  // A routed expert into the cache's parts, against the file's rows.
  {
    const int I = cfg.moe_intermediate_size, H = cfg.hidden;
    const int gb = conv.gate_bits(), db = conv.down_bits();
    const int e = 77, L = 1;
    const char* role[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    const GgufFile::Tensor* t[3];
    const std::uint8_t* src[3];
    for (int r = 0; r < 3; ++r) {
      t[r] = g->tensor("blk." + std::to_string(L) + "." + role[r] +
                       ".weight");
      ASSERT_TRUE(t[r] != nullptr);
      if (t[r] == nullptr) { return; }
      const ggml::Geom geo = ggml::geom(t[r]->type);
      const std::size_t slab = (std::size_t)(t[r]->dims[1] *
                                             (t[r]->dims[0] / geo.elems) *
                                             geo.bytes);
      src[r] = t[r]->data + (std::size_t)e * slab;
    }
    std::vector<std::vector<std::uint8_t>> parts = {
        std::vector<std::uint8_t>((std::size_t)2 * I * H * gb / 8),
        std::vector<std::uint8_t>((std::size_t)2 * I * (H / 32) * 2),
        std::vector<std::uint8_t>((std::size_t)2 * I * (H / 32) * 2),
        std::vector<std::uint8_t>((std::size_t)H * I * db / 8),
        std::vector<std::uint8_t>((std::size_t)H * (I / 32) * 2),
        std::vector<std::uint8_t>((std::size_t)H * (I / 32) * 2)};
    std::uint8_t* dst[6];
    for (int i = 0; i < 6; ++i) { dst[i] = parts[(std::size_t)i].data(); }
    ASSERT_TRUE(qwen4exp_expert_to_affine(t[0]->type, src[0], t[1]->type,
                                          src[1], t[2]->type, src[2], I, H,
                                          gb, db, /*bf16=*/true, dst));
    {
      // What a cache miss costs the reader thread past its read.
      const auto c0 = std::chrono::steady_clock::now();
      const int reps = 16;
      for (int k = 0; k < reps; ++k) {
        qwen4exp_expert_to_affine(t[0]->type, src[0], t[1]->type, src[1],
                                  t[2]->type, src[2], I, H, gb, db, true,
                                  dst);
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - c0).count();
      std::printf("[qwen4exp] an expert converted in %.2f ms (one thread)\n",
                  ms / reps);
    }
    // A box with room for every expert loads the stacks whole through the
    // converter: the same bytes, expert e's rows at e.
    {
      const std::string sw = P + std::to_string(L) + ".mlp.switch_mlp.";
      const int part[3][2] = {{0, 0}, {0, 1}, {3, 0}};   // gate, up, down
      const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
      for (int r = 0; r < 3; ++r) {
        const int rows = r < 2 ? I : H, K = r < 2 ? H : I;
        const int bits = r < 2 ? gb : db;
        const std::size_t wrow = (std::size_t)K * bits / 8;
        const std::size_t grow = (std::size_t)K / 32 * 2;
        int bad = 0;
        for (int k = 0; k < 3; ++k) {
          const char* sfx[3] = {".weight", ".scales", ".biases"};
          const auto whole = raw(sw + proj[r] + sfx[k]);
          ASSERT_TRUE(!whole.empty());
          if (whole.empty()) { continue; }
          const std::size_t rb = k == 0 ? wrow : grow;
          for (int i = 0; i < rows; ++i) {
            // The cache's row: gate|up interleaved (row 2i + which).
            const std::size_t at =
                r < 2 ? (2 * (std::size_t)i + part[r][1]) * rb
                      : (std::size_t)i * rb;
            if (std::memcmp(whole.data() + ((std::size_t)e * rows + i) * rb,
                            parts[(std::size_t)(part[r][0] + k)].data() + at,
                            rb) != 0) {
              ++bad;
            }
          }
        }
        std::printf("[qwen4exp] the whole %s stack: expert %d's rows %s\n",
                    proj[r], e, bad == 0 ? "identical" : "DIFFER");
        ASSERT_TRUE(bad == 0);
      }
    }
    auto deq = [&](int w, int s, int b, int rows, int K, int bits,
                   int row0, int stride) {
      std::vector<float> o((std::size_t)rows * K);
      const auto* sp = reinterpret_cast<const std::uint16_t*>(
          parts[(std::size_t)s].data());
      const auto* bp = reinterpret_cast<const std::uint16_t*>(
          parts[(std::size_t)b].data());
      const std::size_t rb = (std::size_t)K * bits / 8;
      for (int r = 0; r < rows; ++r) {
        const std::size_t pr = (std::size_t)row0 + (std::size_t)stride * r;
        const std::uint8_t* row = parts[(std::size_t)w].data() + pr * rb;
        for (int j = 0; j < K; ++j) {
          const std::size_t gi = pr * (std::size_t)(K / 32) + j / 32;
          o[(std::size_t)r * K + j] =
              bf16_f32_(sp[gi]) * (float)code_at_(row, bits, j, rb) +
              bf16_f32_(bp[gi]);
        }
      }
      return o;
    };
    for (int r = 0; r < 3; ++r) {
      const int rows = r < 2 ? I : H, K = r < 2 ? H : I;
      const std::vector<float> got =
          r < 2 ? deq(0, 1, 2, rows, K, gb, r, 2) : deq(3, 4, 5, rows, K, db,
                                                        0, 1);
      std::vector<float> ref((std::size_t)rows * K);
      ASSERT_TRUE(ggml::decode(t[r]->type, src[r], (std::int64_t)rows * K,
                               ref.data()));
      // bf16 of each value: the representation's own floor.
      std::vector<float> ref16 = ref;
      for (float& f : ref16) { f = bf16_round_(f); }
      const double d = rel_l2_(got, ref), d16 = rel_l2_(ref16, ref);
      std::printf("[qwen4exp] expert %d %s (%s -> w%d g32): %.2e from the "
                  "file's decode (each value in bf16: %.2e)\n", e, role[r],
                  ggml::name(t[r]->type), r < 2 ? gb : db, d, d16);
      // As the projections: bf16's own error where the form is exact
      // with a power-of-two offset, else a fitted bias's (or a rounding's).
      ASSERT_TRUE(at_floor_(t[r]->type, r < 2 ? gb : db) ? d < 1.5 * d16
                                                         : d < 6e-3);
      save("expert." + std::to_string(L) + "." + std::to_string(e) + "." +
               role[r],
           got, {rows, K});
    }
  }
}

// Through the model manager, as text-chat loads it: the GGUF's tokenizer
// and chat template, the engine sizing the expert cache. Greedy, thinking
// off: the answer must be Paris.
TEST(qwen4exp_gguf, chats_through_the_manager)
{
  const char* dir = std::getenv("VPIPE_QWEN4EXP_GGUF");
  if (dir == nullptr || *dir == 0) {
    std::printf("[skip] VPIPE_QWEN4EXP_GGUF unset\n");
    return;
  }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.page_tokens = 256;
  spec.max_pages = 8;
  const auto t0 = std::chrono::steady_clock::now();
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (lm == nullptr || !lm->valid()) { return; }
  const double load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
  auto tpl = genai::make_chat_template(lm->config(), lm->tokenizer(), true);
  ASSERT_TRUE(tpl != nullptr);
  if (tpl == nullptr) { return; }
  std::vector<std::int32_t> ids;
  tpl->render_user_turn("What is the capital of France? Reply with the "
                        "city name only.", /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> gen;
  const auto t1 = std::chrono::steady_clock::now();
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  if (first < 0) { return; }
  gen.push_back(first);
  for (int i = 0; i < 64 && !tpl->is_stop_token(gen.back()); ++i) {
    const std::int32_t t = lm->next_token(ctx);
    if (t < 0) { break; }
    gen.push_back(t);
  }
  const double gen_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t1).count();
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen4exp] load %.1f s; %zu prompt ids, %zu tokens in "
              "%.1f s: '%s'\n", load_s, ids.size(), gen.size(), gen_s,
              text.c_str());
  EXPECT_TRUE(text.find("Paris") != std::string::npos);
}

// Decode speed through the manager (VPIPE_QWEN4EXP_GGUF_BENCH=<tokens>):
// a story, greedy, thinking off; ms a token over the second half (the
// expert cache warmer there).
TEST(qwen4exp_gguf, decode_bench)
{
  const char* dir = std::getenv("VPIPE_QWEN4EXP_GGUF");
  const char* nb = std::getenv("VPIPE_QWEN4EXP_GGUF_BENCH");
  if (dir == nullptr || nb == nullptr) { return; }
  const int N = std::max(16, std::atoi(nb));
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.page_tokens = 256;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (lm == nullptr || !lm->valid()) { return; }
  auto tpl = genai::make_chat_template(lm->config(), lm->tokenizer(), true);
  std::vector<std::int32_t> ids;
  tpl->render_user_turn("Write a short story about a lighthouse keeper "
                        "who finds a message in a bottle.", true, &ids);
  auto ctx = lm->make_context();
  std::vector<std::int32_t> gen{lm->prefill(ctx, ids)};
  auto t_half = std::chrono::steady_clock::now();
  const auto t0 = t_half;
  for (int i = 1; i < N; ++i) {
    if (i == N / 2) { t_half = std::chrono::steady_clock::now(); }
    gen.push_back(lm->next_token(ctx));
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double all = std::chrono::duration<double, std::milli>(t1 - t0)
                         .count() / (N - 1);
  const double half = std::chrono::duration<double, std::milli>(t1 - t_half)
                          .count() / (N - N / 2);
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen4exp] decode %d tokens: %.0f ms a token, %.0f over "
              "the second half; '%s'\n", N, all, half,
              text.substr(0, 200).c_str());
}

// A GGUF's routed experts in their own blocks are SIZED as the cache's
// slots hold them -- each layer's gate, up and down slabs as wide as the
// widest layer's, every expert of every layer the file has (its MTP head's
// too) -- not as the converted view's affine copies (the memory plan's
// claim, the cache the loader sizes). A synthetic file: tensor headers and
// zeroed data, no weights.
TEST(qwen4exp_gguf, native_experts_are_sized_as_their_slots_hold_them)
{
  namespace fs = std::filesystem;
  constexpr int H = 256, I = 256, E = 4;
  // layer 0, 1, and an MTP head's layer 2
  struct L3 { std::uint32_t t[3]; };
  auto write = [&](const fs::path& dir, const std::vector<L3>& layers) {
    fs::create_directories(dir);
    std::vector<std::uint8_t> b;
    auto u32 = [&](std::uint32_t v) {
      for (int i = 0; i < 4; ++i) { b.push_back((std::uint8_t)(v >> 8 * i)); }
    };
    auto u64 = [&](std::uint64_t v) {
      for (int i = 0; i < 8; ++i) { b.push_back((std::uint8_t)(v >> 8 * i)); }
    };
    auto str = [&](const std::string& s) {
      u64(s.size());
      b.insert(b.end(), s.begin(), s.end());
    };
    static const char* kRole[3] = {"ffn_gate_exps", "ffn_up_exps",
                                   "ffn_down_exps"};
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(layers.size() * 3);
    u64(2);
    // the key a model's file opens with (find_gguf_in_dir looks for it)
    str("general.architecture");
    u32(8);
    str("qwen4exp");
    str("general.alignment");
    u32(4);
    u32(32);
    std::uint64_t off = 0;
    std::vector<std::uint64_t> sizes;
    for (std::size_t L = 0; L < layers.size(); ++L) {
      for (int r = 0; r < 3; ++r) {
        const std::uint32_t t = layers[L].t[r];
        const ggml::Geom g = ggml::geom(t);
        const std::int64_t K = r < 2 ? H : I, N = r < 2 ? I : H;
        str("blk." + std::to_string(L) + "." + kRole[r] + ".weight");
        u32(3);
        u64((std::uint64_t)K);
        u64((std::uint64_t)N);
        u64((std::uint64_t)E);
        u32(t);
        u64(off);
        const std::uint64_t n = (std::uint64_t)E * N * (K / g.elems) * g.bytes;
        sizes.push_back(n);
        off += (n + 31) / 32 * 32;
      }
    }
    while (b.size() % 32 != 0) { b.push_back(0); }
    b.resize(b.size() + off, 0);
    std::FILE* f = std::fopen((dir / "m.gguf").c_str(), "wb");
    EXPECT_TRUE(f != nullptr);
    if (f == nullptr) { return; }
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
  };
  const fs::path root = fs::temp_directory_path() / "vpipe-native-size-test";
  fs::remove_all(root);
  MetalQwenModel::Config c;
  c.n_layers = 2;
  c.hidden = H;
  c.moe_inner = I;
  c.n_experts = E;
  // Q4_K / Q4_K / Q6_K, IQ4_XS / IQ4_XS / Q5_K, the MTP head's IQ3_XXS /
  // IQ3_S / Q8_0
  write(root / "a", {{{12, 12, 14}}, {{23, 23, 13}}, {{18, 21, 8}}});
  auto slab = [&](std::uint32_t t, int r) {
    const ggml::Geom g = ggml::geom(t);
    const std::int64_t K = r < 2 ? H : I, N = r < 2 ? I : H;
    return (std::size_t)(N * (K / g.elems) * g.bytes);
  };
  const std::size_t gate = std::max({slab(12, 0), slab(23, 0), slab(18, 0)});
  const std::size_t up = std::max({slab(12, 1), slab(23, 1), slab(21, 1)});
  const std::size_t down = std::max({slab(14, 2), slab(13, 2), slab(8, 2)});
  const std::size_t want = (gate + up + down) * E * 3;
  const std::size_t got =
      MetalQwenModel::gguf_native_expert_bytes((root / "a").string(), c);
  std::fprintf(stderr, "  native experts: %zu bytes (want %zu)\n", got, want);
  EXPECT_TRUE(got == want);
  // a format the native kernels do not read (Q2_K): not native, 0 -- the
  // converted view's sizes stand
  write(root / "b", {{{12, 12, 14}}, {{10, 10, 13}}});
  EXPECT_TRUE(MetalQwenModel::gguf_native_expert_bytes((root / "b").string(),
                                                       c) == 0);
  // switched off: 0
  setenv("VPIPE_QWEN_GGUF_NATIVE", "0", 1);
  EXPECT_TRUE(MetalQwenModel::gguf_native_expert_bytes((root / "a").string(),
                                                       c) == 0);
  unsetenv("VPIPE_QWEN_GGUF_NATIVE");
  fs::remove_all(root);
}
