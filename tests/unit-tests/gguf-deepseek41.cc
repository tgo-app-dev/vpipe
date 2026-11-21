// gguf-deepseek41.cc -- DeepSeek-V4.1-Flash as its GGUFs write it
// (gguf-deepseek41.h: ISTA-DASLab's GSQ-RCO builds, AtomicChat's edits).
//
//   VPIPE_DSV41_GGUF       a variant's first .gguf (the other shards and
//                          the n-gram table's file beside it, or the table
//                          in another repo directory of the models root).
//   VPIPE_DSV41_TEST_MODEL_PATH
//                          (optional) a safetensors pack of the release
//                          (its tokenizer.json): the GGUF's tokenizer and
//                          engram token map against it.
//   VPIPE_DSV41_GGUF_DUMP  (optional) a directory: converted tensors,
//                          an expert's rows and n-gram table rows
//                          dequantized there (<name>.f32 + .shape), for a
//                          comparison against another pack of the release.
//   VPIPE_DSV41_GGUF_BENCH=<tokens>  decode_bench.
//   VPIPE_DSV41_GGUF_FAR_STEPS=<tokens>  far_compute_changes_nothing's
//                          decode (64).
//
// config_tokenizer_and_tables and converted_tensors_are_the_files read the
// files on the CPU alone; chats_through_the_manager and decode_bench load
// the model (cap its expert cache: VPIPE_MOE_EXPERT_CACHE_MB).

#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/chat-template.h"
#include "generative-models/deepseek-v41/deepseek-v41-config.h"
#include "generative-models/deepseek-v41/deepseek-v41-engram.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-model-exec.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-model.h"
#include "generative-models/far-memory.h"
#include "generative-models/lm-memory-limit.h"
#include "generative-models/moe-expert-cache.h"
#include "far-test-kernels.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/tokenizer.h"
#include "generative-models/shared/ggml-blocks.h"
#include "generative-models/shared/gguf-convert.h"
#include "generative-models/shared/gguf-deepseek41.h"
#include "generative-models/shared/gguf-file.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <random>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

float
bf16_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
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

unsigned
code_at_(const std::uint8_t* row, int bits, int j, std::size_t bytes)
{
  const std::size_t bit = (std::size_t)bits * j;
  unsigned v = row[bit >> 3];
  if ((bit >> 3) + 1 < bytes) { v |= (unsigned)row[(bit >> 3) + 1] << 8; }
  return (v >> (bit & 7)) & ((1u << bits) - 1u);
}

// Where `t` at `bits` sits at bf16's own floor (gguf-qwen4exp.cc's rule):
// an exact form whose bias is -off * s' with off a power of two.
bool
at_floor_(std::uint32_t t, int bits)
{
  const bool free_bias =
      t == ggml::kQ4_K || t == ggml::kQ5_K || t == ggml::kQ5_1;
  return ggml::affine_bits(t) > 0 && bits >= ggml::affine_bits(t) &&
         !free_bias && !(t == ggml::kIQ3_S && bits == 4);
}

// Affine parts (packed codes, bf16 scales and biases, group 32) as f32.
std::vector<float>
dequant_(const std::uint8_t* w, const std::uint8_t* s, const std::uint8_t* b,
         int rows, int K, int bits)
{
  std::vector<float> o((std::size_t)rows * K);
  const auto* sp = reinterpret_cast<const std::uint16_t*>(s);
  const auto* bp = reinterpret_cast<const std::uint16_t*>(b);
  const std::size_t rb = (std::size_t)K * bits / 8;
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < K; ++j) {
      const std::size_t g = (std::size_t)r * (K / 32) + j / 32;
      o[(std::size_t)r * K + j] =
          bf16_f32_(sp[g]) * (float)code_at_(w + r * rb, bits, j, rb) +
          bf16_f32_(bp[g]);
    }
  }
  return o;
}

// The file's own decode of rows [0, rows) of a tensor of `K` columns.
std::vector<float>
file_rows_(const GgufFile::Tensor& t, std::int64_t rows)
{
  const std::int64_t K = t.dims[0];
  const ggml::Geom g = ggml::geom(t.type);
  const std::int64_t rb =
      t.type == ggml::kF32 ? K * 4
      : (t.type == ggml::kBF16 || t.type == ggml::kF16) ? K * 2
                                                        : K / g.elems * g.bytes;
  std::vector<float> v((std::size_t)(rows * K));
  for (std::int64_t r = 0; r < rows; ++r) {
    if (!ggml::decode(t.type, t.data + r * rb, K, v.data() + r * K)) {
      return {};
    }
  }
  return v;
}

void
save_(const char* dump, const std::string& nm, const std::vector<float>& v,
      std::vector<std::int64_t> shape)
{
  if (dump == nullptr) { return; }
  std::ofstream(std::string(dump) + "/" + nm + ".f32", std::ios::binary)
      .write(reinterpret_cast<const char*>(v.data()),
             (std::streamsize)(v.size() * 4));
  std::ofstream sh(std::string(dump) + "/" + nm + ".shape");
  for (auto d : shape) { sh << d << " "; }
}

std::string
gguf_path_()
{
  const char* e = std::getenv("VPIPE_DSV41_GGUF");
  if (e == nullptr || *e == 0) { return {}; }
  return find_gguf_in_dir(e);
}

}  // namespace

// The config the GGUF's keys stand for against the release's numbers, its
// tokenizer's ids against the release's tokenizer.json, and the engram's
// hash tables (stored in the file) against the ones the release derives.
// A GGUF carries no vision tower: the release's is found beside it (a
// vision/ folder) or where its record lands, <root>/deepseek-ai/
// DeepSeek-V4.1-Flash-1017 -- its three files and a config.json naming
// DeepSeek-V4.1's vision_config. No other folder: an MLX pack holding a
// tower of its own beside the GGUF is not taken for it.
TEST(deepseek41_gguf, its_vision_tower_is_found_beside_it)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() /
      ("dsv41-vision-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::remove_all(root);
  const fs::path g = root / "AtomicChat" / "Some-GGUF";
  fs::create_directories(g);
  {
    // a GGUF's header alone: version 3, no tensors, its architecture
    std::ofstream f(g / "m-00001-of-00004.gguf", std::ios::binary);
    auto u32 = [&](std::uint32_t x) {
      f.write(reinterpret_cast<const char*>(&x), 4);
    };
    auto u64 = [&](std::uint64_t x) {
      f.write(reinterpret_cast<const char*>(&x), 8);
    };
    auto str = [&](const std::string& x) {
      u64(x.size());
      f.write(x.data(), (std::streamsize)x.size());
    };
    f.write("GGUF", 4);
    u32(3);
    u64(0);
    u64(1);
    str("general.architecture");
    u32(8);
    str("deepseek41");
  }
  const std::string first = (g / "m-00001-of-00004.gguf").string();
  using C = genai::DeepseekV41Config;
  EXPECT_TRUE(C::vision_dir(first).empty());
  auto tower = [](const fs::path& d, const char* type, bool whole = true) {
    fs::create_directories(d);
    std::ofstream(d / "config.json")
        << "{\"model_type\": \"" << type << "\", \"vision_config\": "
           "{\"num_hidden_layers\": 32}}";
    std::ofstream(d / "model-00001-of-00048.safetensors") << "x";
    if (whole) {
      std::ofstream(d / "model-00002-of-00048.safetensors") << "x";
    }
  };
  // A pack's tower, beside it in the models root: not its.
  tower(root / "mlx-community" / "DeepSeek-V4.1-Flash-1017-oQ4e-mtp",
        "deepseek_v41");
  EXPECT_TRUE(C::vision_dir(first).empty());
  // Where the record lands, a file short: not yet.
  const fs::path rel = root / "deepseek-ai" / "DeepSeek-V4.1-Flash-1017";
  tower(rel, "deepseek_v41", false);
  EXPECT_TRUE(C::vision_dir(first).empty());
  EXPECT_TRUE(!C::holds_vision(rel.string()));
  // Whole: it.
  tower(rel, "deepseek_v41");
  EXPECT_TRUE(C::holds_vision(rel.string()));
  EXPECT_TRUE(C::vision_dir(first) == rel.string());
  EXPECT_TRUE(C::vision_dir(g.string()) == rel.string());
  // Another model's config there: not it.
  tower(rel, "qwen3_vl");
  EXPECT_TRUE(C::vision_dir(first).empty());
  tower(rel, "deepseek_v41");
  // Beside the shards, it wins.
  tower(g / "vision", "deepseek_v41");
  EXPECT_TRUE(C::vision_dir(first) == (g / "vision").string());
  // Not a GGUF: nothing looked for.
  EXPECT_TRUE(C::vision_dir(rel.string()).empty());
  fs::remove_all(root);
}

TEST(deepseek41_gguf, config_tokenizer_and_tables)
{
  const std::string path = gguf_path_();
  if (path.empty()) {
    std::printf("[skip] VPIPE_DSV41_GGUF unset\n");
    return;
  }
  const auto g = GgufFile::open(path);
  ASSERT_TRUE(g.has_value());
  if (!g) { return; }
  std::string js, err;
  ASSERT_TRUE(gguf_to_deepseek41_json(*g, &js, &err));
  DeepseekV41Config c;
  ASSERT_TRUE(DeepseekV41Config::from_dir(path, &c, &err));
  std::printf("[dsv41-gguf] %zu files; %d layers, dim %d, %d experts top "
              "%d, engram layers %zu (%lld + %lld rows), %s\n",
              g->file_count(), c.n_layers, c.dim, c.n_experts, c.top_k,
              c.engram_layers.size(),
              c.engram_rows.empty() ? 0LL : (long long)c.engram_rows[0],
              c.engram_rows.size() < 2 ? 0LL : (long long)c.engram_rows[1],
              err.c_str());
  EXPECT_TRUE(c.n_layers == 40 && c.dim == 5120 && c.n_experts == 384 &&
              c.top_k == 6 && c.n_mtp == 0 && c.affine);
  EXPECT_TRUE((int)c.compress_ratios.size() == c.n_layers);
  // The engram's tables as the converter presents them.
  GgufDeepseek41Converter conv(&*g);
  std::map<std::string, std::vector<std::uint8_t>> tab;
  for (const auto& s : conv.specs()) {
    if (s.hf_name.rfind("gguf.engram_", 0) != 0) { continue; }
    std::vector<std::uint8_t> b(s.nbytes);
    ASSERT_TRUE(conv.convert(s, b.data()));
    tab[s.hf_name] = std::move(b);
  }
  ASSERT_TRUE(tab.size() == 4);
  if (tab.size() != 4) { return; }
  auto i64 = [&](const char* n) {
    const auto& b = tab[n];
    return std::span<const std::int64_t>(
        reinterpret_cast<const std::int64_t*>(b.data()), b.size() / 8);
  };
  const auto& tm = tab["gguf.engram_token_map"];
  DeepseekV41Engram eg;
  ASSERT_TRUE(eg.init_tables(
      c,
      std::span<const std::int32_t>(
          reinterpret_cast<const std::int32_t*>(tm.data()), tm.size() / 4),
      i64("gguf.engram_multipliers"), i64("gguf.engram_primes"),
      i64("gguf.engram_offsets"), &err));
  std::printf("[dsv41-gguf] engram tables: the release's (%d compressed "
              "ids) %s\n", eg.compressed_vocab(), err.c_str());
  // A mutated table is refused.
  {
    std::vector<std::int64_t> p(i64("gguf.engram_primes").begin(),
                                i64("gguf.engram_primes").end());
    p[3] += 2;
    DeepseekV41Engram bad;
    std::string e2;
    EXPECT_FALSE(bad.init_tables(
        c,
        std::span<const std::int32_t>(
            reinterpret_cast<const std::int32_t*>(tm.data()), tm.size() / 4),
        i64("gguf.engram_multipliers"), p, i64("gguf.engram_offsets"), &e2));
  }
  // Against the release's tokenizer.json, where one is given.
  const char* ref = std::getenv("VPIPE_DSV41_TEST_MODEL_PATH");
  if (ref == nullptr) { return; }
  const std::string tj =
      (std::filesystem::path(ref) / "tokenizer.json").string();
  DeepseekV41Engram er;
  ASSERT_TRUE(er.init(c, tj, &err));
  int map_diff = 0;
  for (std::size_t i = 0; i < er.token_map().size(); ++i) {
    map_diff += er.token_map()[i] != eg.token_map()[i];
  }
  std::printf("[dsv41-gguf] token map: %d of %zu differ from the "
              "tokenizer.json's\n", map_diff, er.token_map().size());
  EXPECT_TRUE(map_diff == 0);
  auto tg = Tokenizer::from_gguf(*g, nullptr);
  auto tr = Tokenizer::from_huggingface_json(tj, nullptr);
  ASSERT_TRUE(tg != nullptr && tr != nullptr);
  if (tg == nullptr || tr == nullptr) { return; }
  const char* cases[] = {
      "Hello, world! What is the capital of France?",
      "  leading spaces and\ttabs\n\nnew lines\r\n",
      "Numbers 1234567 and 3.14159, 2026-11-17.",
      "中文测试：今天天气很好。"
      "日本語のテキスト、カタカナ。",
      "def f(x):\n    return x ** 2  # comment\n",
      "Mixed: naïve café — ‘quotes’ «guillemets» …",
      "!!!???...,,,;;;::: --- ___ +++ ===",
      "emoji 😀🚀 and symbols ∑∫√ ≠ ≤",
      "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"};
  int diff = 0;
  for (const char* s : cases) {
    const auto a = tg->encode(s), b = tr->encode(s);
    if (a != b) {
      ++diff;
      std::printf("[dsv41-gguf] tokenizer differs on '%s' (%zu vs %zu)\n", s,
                  a.size(), b.size());
    }
  }
  std::printf("[dsv41-gguf] tokenizer: %d of %zu cases differ\n", diff,
              std::size(cases));
  EXPECT_TRUE(diff == 0);
}

// The converter's tensors against the file's own decode: a projection's
// affine codes at bf16's floor where the format's form is exact (else a
// fitted bias's), the norms bf16 of the file's values, the floats as
// stored; one routed expert of every format the file's experts take
// (IQ2_XXS / IQ3_XXS exact, IQ1_M / IQ2_XS / IQ2_S rounded to 8 bits),
// and n-gram table rows read from the table's own file.
TEST(deepseek41_gguf, converted_tensors_are_the_files)
{
  const std::string path = gguf_path_();
  if (path.empty()) { return; }
  const char* dump = std::getenv("VPIPE_DSV41_GGUF_DUMP");
  const auto g = GgufFile::open(path);
  ASSERT_TRUE(g.has_value());
  if (!g) { return; }
  GgufDeepseek41Converter conv(&*g);
  std::map<std::string, const ConvertedTensorSpec*> by;
  for (const auto& s : conv.specs()) { by[s.hf_name] = &s; }
  auto raw = [&](const std::string& nm) {
    std::vector<std::uint8_t> b;
    const auto it = by.find(nm);
    if (it == by.end()) { return b; }
    b.resize(it->second->nbytes);
    if (!conv.convert(*it->second, b.data())) { b.clear(); }
    return b;
  };
  const std::string P = "language_model.layers.";
  struct Proj {
    std::string hf, gguf;
  };
  const std::vector<Proj> projs = {
      {P + "2.attn.wq_a", "blk.2.attn_q_a.weight"},
      {P + "2.attn.wq_b", "blk.2.attn_q_b.weight"},
      {P + "2.attn.wkv", "blk.2.attn_kv.weight"},
      {P + "2.attn.wo_a", "blk.2.attn_output_a.weight"},
      {P + "2.attn.wo_b", "blk.2.attn_output_b.weight"},
      {P + "2.ffn.shared_experts.w1", "blk.2.ffn_gate_shexp.weight"},
      {P + "2.ffn.shared_experts.w2", "blk.2.ffn_down_shexp.weight"},
      {P + "1.engram.wkv", "blk.1.engram_wkv.weight"}};
  for (const Proj& p : projs) {
    const GgufFile::Tensor* t = g->tensor(p.gguf);
    ASSERT_TRUE(t != nullptr);
    if (t == nullptr) { continue; }
    const auto w = raw(p.hf + ".weight"), s = raw(p.hf + ".scales"),
               b = raw(p.hf + ".biases");
    ASSERT_TRUE(!w.empty() && !s.empty() && !b.empty());
    if (w.empty() || s.empty() || b.empty()) { continue; }
    const int K = (int)t->dims[0];
    const int rows = (int)std::min<std::int64_t>(t->numel() / K, 2048);
    const int bits = (int)(by[p.hf + ".weight"]->shape[1] * 32 / K);
    const auto got = dequant_(w.data(), s.data(), b.data(), rows, K, bits);
    const auto ref = file_rows_(*t, rows);
    ASSERT_TRUE(ref.size() == got.size());
    std::vector<float> r16 = ref;
    for (float& f : r16) { f = bf16_round_(f); }
    const double e = rel_l2_(got, ref), e16 = rel_l2_(r16, ref);
    std::printf("[dsv41-gguf] %-40s %-6s -> w%d g32: %.2e from the file "
                "(bf16 %.2e)\n", p.hf.substr(P.size()).c_str(),
                ggml::name(t->type), bits, e, e16);
    ASSERT_TRUE(at_floor_(t->type, bits) ? e < 1.5 * e16 : e < 6e-3);
    save_(dump, p.hf, got, {rows, K});
  }
  // Norms (bf16 of the file's f32), floats as stored, the router as is.
  struct Flt {
    std::string hf, gguf;
    bool f32;
  };
  for (const Flt& f : std::vector<Flt>{
           {P + "2.attn_norm.weight", "blk.2.attn_norm.weight", false},
           {P + "2.attn.kv_norm.weight", "blk.2.attn_kv_a_norm.weight", false},
           {P + "2.attn.compressor.norm.weight",
            "blk.2.attn_compressor_norm.weight", false},
           {P + "2.ffn.gate.weight", "blk.2.ffn_gate_inp.weight", false},
           {P + "2.attn.indexer.wq_b.weight", "blk.2.indexer.attn_q_b.weight",
            false},
           {P + "2.hc_attn_fn", "blk.2.hc_attn_fn.weight", true},
           {P + "2.attn.attn_sink", "blk.2.attn_sinks.weight", true},
           {P + "2.ffn.gate.bias", "blk.2.exp_probs_b.bias", true},
           {"language_model.norm.weight", "output_norm.weight", false}}) {
    const GgufFile::Tensor* t = g->tensor(f.gguf);
    const auto b = raw(f.hf);
    ASSERT_TRUE(t != nullptr && !b.empty());
    if (t == nullptr || b.empty()) { continue; }
    const auto ref = file_rows_(*t, t->numel() / t->dims[0]);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      float v;
      if (f.f32) {
        std::memcpy(&v, b.data() + 4 * i, 4);
        bad += v != ref[i];
      } else {
        std::uint16_t h;
        std::memcpy(&h, b.data() + 2 * i, 2);
        bad += bf16_f32_(h) != bf16_round_(ref[i]);
      }
    }
    std::printf("[dsv41-gguf] %-40s %-5s: %zu of %zu values off\n",
                f.hf.substr(f.hf.rfind("layers.") == std::string::npos
                                ? 0 : P.size()).c_str(),
                ggml::name(t->type), bad, ref.size());
    EXPECT_TRUE(bad == 0);
  }
  // A routed expert of every format the file's experts take.
  const int E = 384, I = 2304, H = 5120, ex = 77;
  std::map<std::uint32_t, int> seen;   // format -> layer
  for (int L = 0; L < 40; ++L) {
    for (const char* r : {"ffn_gate_exps", "ffn_down_exps"}) {
      const auto* t = g->tensor(std::format("blk.{}.{}.weight", L, r));
      if (t != nullptr && !seen.count(t->type)) { seen[t->type] = L; }
    }
  }
  std::vector<int> layers;
  for (const auto& [ty, L] : seen) {
    if (std::find(layers.begin(), layers.end(), L) == layers.end()) {
      layers.push_back(L);
    }
  }
  for (const int L : layers) {
    const GgufFile::Tensor* t[3] = {
        g->tensor(std::format("blk.{}.ffn_gate_exps.weight", L)),
        g->tensor(std::format("blk.{}.ffn_up_exps.weight", L)),
        g->tensor(std::format("blk.{}.ffn_down_exps.weight", L))};
    ASSERT_TRUE(t[0] != nullptr && t[1] != nullptr && t[2] != nullptr);
    if (t[0] == nullptr || t[1] == nullptr || t[2] == nullptr) { continue; }
    const int b13 = std::max(deepseek41_expert_bits(t[0]->type),
                             deepseek41_expert_bits(t[1]->type));
    const int b2 = deepseek41_expert_bits(t[2]->type);
    std::vector<std::vector<std::uint8_t>> parts(9);
    std::uint8_t* dst[9];
    const std::size_t c13 = (std::size_t)I * H * b13 / 8;
    const std::size_t c2 = (std::size_t)H * I * b2 / 8;
    const std::size_t s13 = (std::size_t)I * (H / 32) * 2;
    const std::size_t s2 = (std::size_t)H * (I / 32) * 2;
    const std::size_t sz[9] = {c13, s13, s13, c13, s13, s13, c2, s2, s2};
    for (int k = 0; k < 9; ++k) {
      parts[(std::size_t)k].resize(sz[k]);
      dst[k] = parts[(std::size_t)k].data();
    }
    const std::uint8_t* src[3];
    for (int r = 0; r < 3; ++r) {
      const std::int64_t K = t[r]->dims[0], N = t[r]->dims[1];
      const ggml::Geom geo = ggml::geom(t[r]->type);
      src[r] = t[r]->data + (std::size_t)ex * N * (K / geo.elems) * geo.bytes;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(deepseek41_expert_to_affine(t[0]->type, src[0], t[1]->type,
                                            src[1], t[2]->type, src[2], I, H,
                                            b13, b2, dst));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    static const char* kRole[3] = {"w1", "w3", "w2"};
    for (int r = 0; r < 3; ++r) {
      const int N = r < 2 ? I : H, K = r < 2 ? H : I;
      const int bits = r < 2 ? b13 : b2;
      const auto got = dequant_(dst[3 * r], dst[3 * r + 1], dst[3 * r + 2],
                                N, K, bits);
      std::vector<float> ref((std::size_t)N * K);
      ASSERT_TRUE(ggml::decode(t[r]->type, src[r], (std::int64_t)N * K,
                               ref.data()));
      std::vector<float> r16 = ref;
      for (float& f : r16) { f = bf16_round_(f); }
      const double e = rel_l2_(got, ref), e16 = rel_l2_(r16, ref);
      std::printf("[dsv41-gguf] layer %d expert %d %s (%s -> w%d g32): "
                  "%.2e from the file (bf16 %.2e)%s\n", L, ex, kRole[r],
                  ggml::name(t[r]->type), bits, e, e16,
                  ggml::affine_bits(t[r]->type) > 0 ? "" : " -- rounded");
      ASSERT_TRUE(at_floor_(t[r]->type, bits) ? e < 1.5 * e16 : e < 6e-3);
      save_(dump, std::format("expert.{}.{}.{}", L, ex, kRole[r]), got,
            {N, K});
    }
    std::printf("[dsv41-gguf] layer %d: an expert converted in %.1f ms (one "
                "thread)\n", L, ms);
    (void)E;
  }
  // The n-gram table's rows, from its own file.
  const std::string ep = gguf_deepseek41_engram_path(*g, path);
  const auto eg = GgufFile::open_single(ep);
  if (!eg) {
    std::printf("[dsv41-gguf] no n-gram table file at %s\n", ep.c_str());
    return;
  }
  const GgufFile::Tensor* et = eg->tensor("per_layer_token_embd.weight");
  ASSERT_TRUE(et != nullptr && et->dims.size() == 2 && et->dims[0] == 256);
  if (et == nullptr) { return; }
  const ggml::Geom geo = ggml::geom(et->type);
  const std::int64_t rb = 256 / geo.elems * geo.bytes;
  const std::int64_t picks[] = {0, 1, 12345, 16000057, 384006167,
                                384006168, 384006168 + 777};
  std::vector<float> rows;
  for (std::int64_t r : picks) {
    std::vector<float> v(256);
    ASSERT_TRUE(ggml::decode(et->type, et->data + r * rb, 256, v.data()));
    rows.insert(rows.end(), v.begin(), v.end());
  }
  std::printf("[dsv41-gguf] n-gram table %s [256, %lld] read from %s\n",
              ggml::name(et->type), (long long)et->dims[1], ep.c_str());
  save_(dump, "engram.rows", rows, {(std::int64_t)std::size(picks), 256});
}

// The plan against the load: what the model allocates beside its expert
// cache (held at its floor, two layers' experts) against the rest the
// plan books for the checkpoint (expert_bytes), and the claim's floor --
// that rest and two layers of experts at the slot's width -- against all
// the model holds.
TEST(deepseek41_gguf, the_plan_books_what_the_load_holds)
{
  const std::string path = gguf_path_();
  if (path.empty()) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::size_t ex = 0, rest = 0, vis = 0, disk = 0;
  ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(path, &ex, &rest,
                                                      &vis, &disk));
  DeepseekV41Config cfg;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_dir(path, &cfg, &err));
  MetalDeepseekV41Model::Options opt;
  opt.use_bf16 = true;
  opt.expert_cache_bytes = 1;      // the floor: a layer resident, a layer
                                   // transient
  const std::size_t a0 = mc->memory_budget_settled().allocated;
  auto m = MetalDeepseekV41Model::load(WeightSet::open(path, nullptr), mc,
                                       opt, &err);
  ASSERT_TRUE(m != nullptr && m->expert_cache() != nullptr);
  if (m == nullptr || m->expert_cache() == nullptr) {
    std::printf("[dsv41-gguf] %s\n", err.c_str());
    return;
  }
  const std::size_t a1 = mc->memory_budget_settled().allocated;
  const std::size_t cache = m->expert_cache()->bytes();
  const std::size_t held = a1 - a0 - cache;
  const std::size_t floor = rest + 2 * (ex / (std::size_t)cfg.n_layers);
  std::printf("[dsv41-gguf] the plan books %.0f MB beside the experts "
              "(%.0f MB of experts, every one at the slot's width), the "
              "load holds %.0f MB beside its %.0f MB cache; the claim's "
              "floor %.0f MB against %.0f MB held in all\n", rest / 1e6,
              ex / 1e6, held / 1e6, cache / 1e6, floor / 1e6,
              (held + cache) / 1e6);
  EXPECT_TRUE((double)held < 1.15 * (double)rest &&
              (double)held > 0.85 * (double)rest);
  EXPECT_TRUE((double)floor > 0.95 * (double)(held + cache));
}

TEST(deepseek41_gguf, chats_through_the_manager)
{
  const std::string path = gguf_path_();
  if (path.empty()) { return; }
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
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
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> gen;
  const auto t1 = std::chrono::steady_clock::now();
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  if (first < 0) { return; }
  gen.push_back(first);
  for (int i = 0; i < 32 && !tpl->is_stop_token(gen.back()); ++i) {
    const std::int32_t t = lm->next_token(ctx);
    if (t < 0) { break; }
    gen.push_back(t);
  }
  const double gen_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t1).count();
  const std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[dsv41-gguf] load %.1f s; %zu prompt ids, %zu tokens in "
              "%.1f s: '%s'\n", load_s, ids.size(), gen.size(), gen_s,
              text.c_str());
  EXPECT_TRUE(text.find("Paris") != std::string::npos);
}

// Decode speed through the manager (VPIPE_DSV41_GGUF_BENCH=<tokens>).
TEST(deepseek41_gguf, decode_bench)
{
  const std::string path = gguf_path_();
  const char* nb = std::getenv("VPIPE_DSV41_GGUF_BENCH");
  if (path.empty() || nb == nullptr) { return; }
  const int N = std::max(16, std::atoi(nb));
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (mgr == nullptr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
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
  std::printf("[dsv41-gguf] decode %d tokens: %.0f ms a token, %.0f over "
              "the second half; '%s'\n", N, all, half,
              text.substr(0, 200).c_str());
}

// A GGUF'S EXPERTS RUN FAR CHANGE NOTHING (VPIPE_DSV41_GGUF; the model
// itself, ~160 GB of experts -- a large Mac): a greedy decode with the
// expert cache a sixteenth of the native experts (its resident set full
// in a few tokens, so misses are let go to the far tier), then with far
// memory beside
// it and the far side RUNNING the misses it holds in their own blocks
// (kFarExpertsDeepseekGguf: vpipe's iq5 decode kernels on the far bytes,
// the grids read behind their FarGgufTables head), a commit and wait a
// layer, then the same through the HANDSHAKE, the far side replaying the
// rows recorded (far-test-kernels.h) -- the logits the same to the bit.
// VPIPE_DSV41_GGUF_FAR_STEPS: the tokens decoded (64).
TEST(deepseek41_gguf, far_compute_changes_nothing)
{
  using far_test::KernelFar;
  const std::string path = gguf_path_();
  if (path.empty()) { return; }
  const std::string dir = std::filesystem::path(path).parent_path().string();
  const std::size_t ex =
      MetalDeepseekV41Model::gguf_native_expert_bytes(dir);
  if (ex == 0) {
    std::printf("[dsv41-gguf] far: not a GGUF run in its own blocks\n");
    return;
  }
  const char* st = std::getenv("VPIPE_DSV41_GGUF_FAR_STEPS");
  const int steps = st != nullptr ? std::max(4, std::atoi(st)) : 64;
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  ::setenv("VPIPE_MOE_EXPERT_AHEAD", "0", 1);
  auto rec = std::make_shared<far_test::Recording>();
  std::vector<float> out[3];
  int ran = 0, replayed = 0, redos = -1;
  std::uint64_t late = 0;
  bool hs_on = false;
  std::vector<std::int32_t> prompt;
  for (int i = 0; i < 48; ++i) { prompt.push_back(1000 + 37 * i); }
  for (int a = 0; a < 3; ++a) {
    ::setenv("VPIPE_DSV41_HANDSHAKE", a == 2 ? "1" : "0", 1);
    std::shared_ptr<KernelFar> far =
        a == 0 ? nullptr
               : std::make_shared<KernelFar>(
                     mc, ex, a == 1 ? KernelFar::Mode::kCompute
                                    : KernelFar::Mode::kReplay, rec);
    LmFarMemory scope(far);
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.expert_cache_bytes = ex / 16;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(dir, nullptr), mc,
                                         opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) {
      std::printf("[dsv41-gguf] far: %s\n", err.c_str());
      break;
    }
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, prompt, &lg, false, &err));
    out[a] = lg;
    for (int i = 0; i < steps; ++i) {
      const std::int32_t t = (std::int32_t)(std::max_element(lg.begin(),
                                                             lg.end()) -
                                            lg.begin());
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                             &err));
      out[a].insert(out[a].end(), lg.begin(), lg.end());
    }
    if (a == 1) {
      ran = far->ran.load();
      late += m->remote_late();
    }
    if (a == 2) {
      replayed = far->replayed.load();
      hs_on = m->handshake_on();
      redos = m->hs_redos();
      late += m->remote_late();
    }
  }
  ::unsetenv("VPIPE_MOE_EXPERT_AHEAD");
  ::unsetenv("VPIPE_DSV41_HANDSHAKE");
  std::printf("[dsv41-gguf] far: %d run there, %d replayed through the "
              "handshake (%s, %d redone), %llu late\n", ran, replayed,
              hs_on ? "on" : "OFF", redos, (unsigned long long)late);
  ASSERT_TRUE(!out[0].empty());
  for (int a = 1; a < 3; ++a) {
    ASSERT_TRUE(out[a].size() == out[0].size());
    EXPECT_TRUE(std::memcmp(out[0].data(), out[a].data(),
                            out[0].size() * sizeof(float)) == 0);
  }
  EXPECT_TRUE(ran > 0 && replayed > 0 && hs_on && redos == 0);
}

// An expert miss's conversion ON THE GPU (ggml_to_affine.metal) is the
// CPU's (ggml::row_affine32 at 8 bits, bf16), bit for bit: random blocks
// of each format the GGUFs' experts use -- IQ2_XXS exact, IQ2_XS and IQ1_M
// rounded over their groups -- sane steps, a zero one and subnormal ones
// among them; and, with VPIPE_DSV41_GGUF, real experts of every layer.
TEST(deepseek41_gguf, gpu_conversion_is_the_cpus)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("ggml_to_affine");
  auto fn = lib.function("ggml_rows_to_affine8");
  EXPECT_TRUE(fn.valid());
  if (!fn.valid()) { return; }
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables();
  EXPECT_TRUE(tab.size() == 2832);
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  // rows of `type` (raw bytes) against the CPU; how many values differ
  auto check = [&](std::uint32_t type, const std::uint8_t* rows, int N,
                   int K) -> std::size_t {
    const ggml::Geom geo = ggml::geom(type);
    const std::size_t rowb = (std::size_t)(K / geo.elems) * geo.bytes;
    const std::size_t G = (std::size_t)N * (K / 32);
    auto bsrc = mc->make_shared_buffer(rowb * N);
    auto bc = mc->make_shared_buffer((std::size_t)N * K);
    auto bs = mc->make_shared_buffer(G * 2);
    auto bb = mc->make_shared_buffer(G * 2);
    std::memcpy(bsrc.contents(), rows, rowb * N);
    auto st = mc->make_command_stream();
    {
      auto enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, bsrc);
      enc.set_buffer(1, btab);
      enc.set_buffer(2, bc);
      enc.set_buffer(3, bs);
      enc.set_buffer(4, bb);
      enc.set_constant(5, type);
      enc.set_constant(6, (std::uint32_t)K);
      enc.set_constant(7, (std::uint32_t)rowb);
      enc.set_constant(8, (std::uint32_t)G);
      enc.set_constant(9, (std::uint32_t)0);
      enc.dispatch({(unsigned)G, 1, 1}, {256, 1, 1});
    }
    EXPECT_TRUE(st.commit().wait_ok());
    std::vector<std::uint8_t> c((std::size_t)K);
    std::vector<std::uint16_t> s((std::size_t)K / 32), b(s.size());
    std::size_t bad = 0;
    for (int r = 0; r < N; ++r) {
      EXPECT_TRUE(ggml::row_affine32(type, rows + rowb * r, K, 8, true,
                                     c.data(), s.data(), b.data()));
      const auto* gc = static_cast<const std::uint8_t*>(bc.contents()) +
                       (std::size_t)r * K;
      const auto* gs = static_cast<const std::uint16_t*>(bs.contents()) +
                       (std::size_t)r * (K / 32);
      const auto* gb = static_cast<const std::uint16_t*>(bb.contents()) +
                       (std::size_t)r * (K / 32);
      for (int j = 0; j < K; ++j) { bad += gc[j] != c[(std::size_t)j]; }
      for (int g = 0; g < K / 32; ++g) {
        bad += gs[g] != s[(std::size_t)g];
        bad += gb[g] != b[(std::size_t)g];
      }
    }
    return bad;
  };
  std::mt19937 rng(7);
  for (std::uint32_t type : {(std::uint32_t)ggml::kIQ2_XXS,
                             (std::uint32_t)ggml::kIQ2_XS,
                             (std::uint32_t)ggml::kIQ1_M}) {
    EXPECT_TRUE(ggml::gpu_affine8(type));
    const ggml::Geom geo = ggml::geom(type);
    const int N = 96, K = 4096, nb = K / geo.elems;
    std::vector<std::uint8_t> rows((std::size_t)N * nb * geo.bytes);
    for (auto& x : rows) { x = (std::uint8_t)rng(); }
    for (int i = 0; i < N * nb; ++i) {
      std::uint8_t* blk = rows.data() + (std::size_t)i * geo.bytes;
      // the block's step: sane, else zero (1 in 17) or subnormal (1 in 13)
      _Float16 d = (_Float16)(std::ldexp(1.0 + (rng() % 1000) / 1000.0,
                                         -(int)(rng() % 12) - 2));
      if (i % 17 == 0) { d = (_Float16)0.0f; }
      if (i % 13 == 0) { d = (_Float16)std::ldexp(1.0, -20); }
      std::uint16_t h;
      std::memcpy(&h, &d, 2);
      if (type == ggml::kIQ1_M) {
        // its step is the four scales' top nibbles
        for (int k = 0; k < 4; ++k) {
          std::uint16_t sc;
          std::memcpy(&sc, blk + 48 + 2 * k, 2);
          sc = (std::uint16_t)((sc & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12));
          std::memcpy(blk + 48 + 2 * k, &sc, 2);
        }
      } else {
        std::memcpy(blk, &h, 2);
      }
    }
    {
      const std::size_t bad = check(type, rows.data(), N, K);
      if (bad != 0) {
        std::fprintf(stderr, "  type %u: %zu values differ\n", type, bad);
      }
      EXPECT_TRUE(bad == 0);
    }
  }
  // Real experts: one of each layer's three stacks.
  const char* path = std::getenv("VPIPE_DSV41_GGUF");
  if (path == nullptr) { return; }
  auto g = GgufFile::open(path);
  EXPECT_TRUE(g.has_value());
  if (!g) { return; }
  std::size_t slabs = 0;
  for (int L = 0;; ++L) {
    bool any = false;
    for (const char* role : {"ffn_gate_exps", "ffn_up_exps",
                             "ffn_down_exps"}) {
      const GgufFile::Tensor* t =
          g->tensor(std::format("blk.{}.{}.weight", L, role));
      if (t == nullptr) { continue; }
      any = true;
      if (!ggml::gpu_affine8(t->type)) { continue; }
      const int K = (int)t->dims[0], N = (int)t->dims[1];
      const ggml::Geom geo = ggml::geom(t->type);
      const std::size_t bytes = (std::size_t)N * (K / geo.elems) * geo.bytes;
      const int ex = (L * 37 + 11) % (int)t->dims[2];
      std::vector<std::uint8_t> raw(bytes);
      std::FILE* f = std::fopen(g->file_path(t->file).c_str(), "rb");
      EXPECT_TRUE(f != nullptr);
      if (f == nullptr) { return; }
      std::fseek(f, (long)(t->file_offset + (std::uint64_t)ex * bytes),
                 SEEK_SET);
      EXPECT_TRUE(std::fread(raw.data(), 1, bytes, f) == bytes);
      std::fclose(f);
      {
        const std::size_t bad = check(t->type, raw.data(), N, K);
        if (bad != 0) {
          std::fprintf(stderr, "  blk.%d.%s (type %u): %zu differ\n", L,
                       role, t->type, bad);
        }
        EXPECT_TRUE(bad == 0);
      }
      ++slabs;
    }
    if (!any) { break; }
  }
  std::fprintf(stderr, "  %zu real expert slabs converted alike\n", slabs);
}

// Random blocks of `type` with sane steps (a zero one now and then), N
// rows of K values.
static std::vector<std::uint8_t>
random_iq_rows_(std::uint32_t type, int N, int K, std::mt19937& rng)
{
  const ggml::Geom geo = ggml::geom(type);
  const int nb = K / geo.elems;
  std::vector<std::uint8_t> rows((std::size_t)N * nb * geo.bytes);
  for (auto& x : rows) { x = (std::uint8_t)rng(); }
  for (int i = 0; i < N * nb; ++i) {
    std::uint8_t* blk = rows.data() + (std::size_t)i * geo.bytes;
    // IQ3_XXS's grid reaches 62 (IQ2's 43, 8x its step): smaller steps,
    // so a down projection's sums stay inside half's range
    const int e0 = type == ggml::kIQ3_XXS ? 10 : 6;
    _Float16 d = (_Float16)(std::ldexp(1.0 + (rng() % 1000) / 1000.0,
                                       -(int)(rng() % 4) - e0));
    if (i % 17 == 0) { d = (_Float16)0.0f; }
    std::uint16_t h;
    std::memcpy(&h, &d, 2);
    if (type == ggml::kIQ1_M) {
      for (int k = 0; k < 4; ++k) {
        std::uint16_t sc;
        std::memcpy(&sc, blk + 48 + 2 * k, 2);
        sc = (std::uint16_t)((sc & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12));
        std::memcpy(blk + 48 + 2 * k, &sc, 2);
      }
    } else {
      std::memcpy(blk, &h, 2);
    }
  }
  return rows;
}

// A PREFILL'S GROUPED GEMMS OVER THE BLOCKS ON THE MATRIX CORES
// (deepseek_v41_nax's dsv41_moe_{gu,down}_nax_iq), in both builds -- f16,
// and bf16, the one the model binds (it runs bf16) -- against ggml's own
// widening on the CPU, f32 sums: two 32-row tiles of two experts (slots 1
// and 3 of four, a slot 4 KB wider than its expert, as the cache's widest
// layer's), a pad row in each; every type the GGUFs' experts use, and
// gate, up and down of different types in one expert. Judged against
// each output's RMS (bf16 has three fewer mantissa bits; weights and
// activations are staged in the element type, as the affine NAX twins
// stage them): the differences' RMS within 0.1% (f16) / 0.75% (bf16) of
// it, the worst of ~31k within 2% / 12%. Measured on the M5 Max: RMS 0.029-0.047% / 0.23-0.38%, worst
// 0.28-1.02% / 2.3-8.6%. (MUTATION: gate's type read as up's, or the slot
// one stride off, fails the mixed cases.)
TEST(deepseek41_gguf, native_nax_expert_kernels_are_the_cpus)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) {
    std::printf("[dsv41-gguf] nax: no matrix cores here\n");
    return;
  }
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables_native();
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  const int H = 1024, I = 512, R = 64;   // two tiles
  const float lim = 7.0f;
  // the element type's bits, and back
  auto to_elt = [](float v, bool bf) -> std::uint16_t {
    if (!bf) {
      const _Float16 h = (_Float16)v;
      std::uint16_t u;
      std::memcpy(&u, &h, 2);
      return u;
    }
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   // nearest even
    return (std::uint16_t)(u >> 16);
  };
  auto from_elt = [](std::uint16_t u, bool bf) -> float {
    if (!bf) {
      _Float16 h;
      std::memcpy(&h, &u, 2);
      return (float)h;
    }
    const std::uint32_t w = (std::uint32_t)u << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
  };
  using T3 = std::array<std::uint32_t, 3>;   // w1's, w3's, w2's types
  const std::vector<T3> cases = {
      {ggml::kIQ2_XXS, ggml::kIQ2_XXS, ggml::kIQ2_XXS},
      {ggml::kIQ2_XS, ggml::kIQ2_XS, ggml::kIQ2_XS},
      {ggml::kIQ1_M, ggml::kIQ1_M, ggml::kIQ1_M},
      {ggml::kIQ2_S, ggml::kIQ2_S, ggml::kIQ2_S},
      {ggml::kIQ3_XXS, ggml::kIQ3_XXS, ggml::kIQ3_XXS},
      // mixed: gate and up told apart, down its own
      {ggml::kIQ2_XXS, ggml::kIQ2_S, ggml::kIQ2_XS},
      {ggml::kIQ1_M, ggml::kIQ2_XS, ggml::kIQ3_XXS}};
  for (const bool bf : {false, true}) {
    auto lib = mc->load_library(bf ? "deepseek_v41_nax_bf16"
                                   : "deepseek_v41_nax");
    auto f_gu = lib.function("dsv41_moe_gu_nax_iq");
    auto f_dg = lib.function("dsv41_moe_down_nax_iq");
    ASSERT_TRUE(f_gu.valid() && f_dg.valid());
    if (!f_gu.valid() || !f_dg.valid()) { return; }
    // the differences' RMS, and the worst of ~31k, as shares of the
    // output's RMS
    const double rms_bar = bf ? 0.0075 : 0.001;
    const double bar = bf ? 0.12 : 0.02;
    std::mt19937 rng(23);
    for (const T3& ty : cases) {
      auto slab = [&](std::uint32_t t, int N, int K) {
        const ggml::Geom geo = ggml::geom(t);
        return (std::size_t)N * (K / geo.elems) * geo.bytes;
      };
      // a slot as the cache lays it: the widest of the stacks' slabs, and
      // 4 KB more (a wider layer's)
      const std::size_t c13 =
          std::max(slab(ty[0], I, H), slab(ty[1], I, H)) + 4096;
      const std::size_t c2 = slab(ty[2], H, I) + 4096;
      auto p1 = mc->make_shared_buffer(4 * c13);
      auto p3 = mc->make_shared_buffer(4 * c13);
      auto p2 = mc->make_shared_buffer(4 * c2);
      std::vector<float> w1[2], w3[2], w2[2];
      for (int e = 0; e < 2; ++e) {
        const int slot = 1 + 2 * e;
        const auto r1 = random_iq_rows_(ty[0], I, H, rng);
        const auto r3 = random_iq_rows_(ty[1], I, H, rng);
        const auto r2 = random_iq_rows_(ty[2], H, I, rng);
        std::memcpy(static_cast<std::uint8_t*>(p1.contents()) + slot * c13,
                    r1.data(), r1.size());
        std::memcpy(static_cast<std::uint8_t*>(p3.contents()) + slot * c13,
                    r3.data(), r3.size());
        std::memcpy(static_cast<std::uint8_t*>(p2.contents()) + slot * c2,
                    r2.data(), r2.size());
        w1[e].resize((std::size_t)I * H);
        w3[e].resize((std::size_t)I * H);
        w2[e].resize((std::size_t)H * I);
        EXPECT_TRUE(ggml::decode(ty[0], r1.data(), (std::int64_t)I * H,
                                 w1[e].data()));
        EXPECT_TRUE(ggml::decode(ty[1], r3.data(), (std::int64_t)I * H,
                                 w3[e].data()));
        EXPECT_TRUE(ggml::decode(ty[2], r2.data(), (std::int64_t)H * I,
                                 w2[e].data()));
      }
      // R sorted rows: tile 0 expert 0 (slot 1), tile 1 expert 1 (slot 3);
      // row 31 of each a pad (srow / sdst -1)
      auto bx = mc->make_shared_buffer((std::size_t)R * H * 2);
      auto bact = mc->make_shared_buffer((std::size_t)R * I * 2);
      auto bout = mc->make_shared_buffer((std::size_t)R * H * 2);
      auto bsrow = mc->make_shared_buffer((std::size_t)R * 4);
      auto bsdst = mc->make_shared_buffer((std::size_t)R * 4);
      auto bswt = mc->make_shared_buffer((std::size_t)R * 4);
      auto bt2e = mc->make_shared_buffer(8);
      std::vector<float> x((std::size_t)R * H);
      std::uniform_real_distribution<float> U(-1.0f, 1.0f);
      auto* xb = static_cast<std::uint16_t*>(bx.contents());
      for (int k = 0; k < R * H; ++k) {
        xb[k] = to_elt(U(rng), bf);
        x[(std::size_t)k] = from_elt(xb[k], bf);
      }
      auto* srow = static_cast<std::int32_t*>(bsrow.contents());
      auto* sdst = static_cast<std::int32_t*>(bsdst.contents());
      auto* swt = static_cast<float*>(bswt.contents());
      for (int r = 0; r < R; ++r) {
        const bool pad = r % 32 == 31;
        srow[r] = pad ? -1 : r;
        sdst[r] = pad ? -1 : r;
        swt[r] = 0.5f + 0.5f * (float)r / R;
      }
      static_cast<std::int32_t*>(bt2e.contents())[0] = 1;
      static_cast<std::int32_t*>(bt2e.contents())[1] = 3;
      const long cw13 = (long)c13, cw2 = (long)c2;
      auto st = mc->make_command_stream();
      {
        auto enc = st.begin_compute();
        enc.set_function(f_gu);
        enc.set_buffer(0, p1);
        enc.set_buffer(1, p3);
        enc.set_buffer(2, btab);
        enc.set_buffer(3, bx);
        enc.set_buffer(4, bact);
        enc.set_buffer(5, bsrow);
        enc.set_buffer(6, bswt);
        enc.set_buffer(7, bt2e);
        enc.set_constant(8, H);
        enc.set_constant(9, I);
        enc.set_constant(10, lim);
        enc.set_constant(11, ty[0]);
        enc.set_constant(12, ty[1]);
        enc.set_constant(13, cw13);
        enc.dispatch({(unsigned)(I / 64 * 32), 2 * 2, 2}, {32, 2, 2});
        enc.end();
      }
      EXPECT_TRUE(st.commit().wait_ok());
      // the CPU's act (rounded to the element type), then the down GEMM
      // from it; each judged against its output's RMS -- a value near 0 a
      // small difference of large terms
      std::vector<float> act((std::size_t)R * I, 0.0f);
      double e_gu = 0, s_gu = 0, e_dn = 0, s_dn = 0, q_gu = 0, q_dn = 0;
      std::size_t n_gu = 0, n_dn = 0;
      const auto* ga = static_cast<const std::uint16_t*>(bact.contents());
      for (int r = 0; r < R; ++r) {
        if (r % 32 == 31) { continue; }
        const int e = r / 32;
        for (int f = 0; f < I; ++f) {
          double g = 0, u = 0;
          for (int k = 0; k < H; ++k) {
            const double xv = x[(std::size_t)r * H + k];
            g += (double)w1[e][(std::size_t)f * H + k] * xv;
            u += (double)w3[e][(std::size_t)f * H + k] * xv;
          }
          u = std::clamp(u, (double)-lim, (double)lim);
          g = std::min(g, (double)lim);
          const float a = (float)(swt[r] * (g / (1 + std::exp(-g)) * u));
          act[(std::size_t)r * I + f] = from_elt(to_elt(a, bf), bf);
          const double d =
              (double)from_elt(ga[(std::size_t)r * I + f], bf) - a;
          e_gu = std::max(e_gu, std::fabs(d));
          q_gu += d * d;
          s_gu += (double)a * a;
          ++n_gu;
        }
      }
      auto* ab = static_cast<std::uint16_t*>(bact.contents());
      for (int k = 0; k < R * I; ++k) {
        ab[k] = to_elt(act[(std::size_t)k], bf);
      }
      auto st2 = mc->make_command_stream();
      {
        auto enc = st2.begin_compute();
        enc.set_function(f_dg);
        enc.set_buffer(0, p2);
        enc.set_buffer(1, btab);
        enc.set_buffer(2, bact);
        enc.set_buffer(3, bout);
        enc.set_buffer(4, bsdst);
        enc.set_buffer(5, bt2e);
        enc.set_constant(6, I);
        enc.set_constant(7, H);
        enc.set_constant(8, ty[2]);
        enc.set_constant(9, cw2);
        enc.dispatch({(unsigned)(H / 64 * 32), 2 * 2, 2}, {32, 2, 2});
        enc.end();
      }
      EXPECT_TRUE(st2.commit().wait_ok());
      const auto* go = static_cast<const std::uint16_t*>(bout.contents());
      for (int r = 0; r < R; ++r) {
        if (r % 32 == 31) { continue; }
        const int e = r / 32;
        for (int o = 0; o < H; ++o) {
          double sum = 0;
          for (int k = 0; k < I; ++k) {
            sum += (double)w2[e][(std::size_t)o * I + k] *
                   act[(std::size_t)r * I + k];
          }
          const double d =
              (double)from_elt(go[(std::size_t)r * H + o], bf) - sum;
          e_dn = std::max(e_dn, std::fabs(d));
          q_dn += d * d;
          s_dn += sum * sum;
          ++n_dn;
        }
      }
      const double r_gu = std::sqrt(s_gu / std::max<std::size_t>(n_gu, 1));
      const double r_dn = std::sqrt(s_dn / std::max<std::size_t>(n_dn, 1));
      // the differences' RMS (the steady figure) and the worst one
      const double m_gu = std::sqrt(q_gu / std::max<std::size_t>(n_gu, 1));
      const double m_dn = std::sqrt(q_dn / std::max<std::size_t>(n_dn, 1));
      std::printf("[dsv41-gguf] nax %s types %u/%u/%u: gate|up %.3f%% "
                  "(worst %.2f%%) of RMS, down %.3f%% (worst %.2f%%)\n",
                  bf ? "bf16" : "f16", ty[0], ty[1], ty[2],
                  100 * m_gu / r_gu, 100 * e_gu / r_gu, 100 * m_dn / r_dn,
                  100 * e_dn / r_dn);
      EXPECT_TRUE(r_gu > 0 && r_dn > 0);
      EXPECT_TRUE(m_gu <= rms_bar * r_gu && m_dn <= rms_bar * r_dn);
      EXPECT_TRUE(e_gu <= bar * r_gu && e_dn <= bar * r_dn);
    }
  }
}

// The routed experts run in a GGUF's own blocks (dsv41_moe_*_iq*): a
// decode's pair kernels and a prefill's grouped GEMMs against ggml's own
// widening on the CPU, f32 sums, the outputs rounded to half.
TEST(deepseek41_gguf, native_expert_kernels_are_the_cpus)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41");
  // the decode kernels: the model's (iq5_s4_r2, _tg: 4 simdgroups of 2
  // rows), or one named (VPIPE_DSV41_IQ_SW=<suffix>: iq, iq4_tg_r2, ...)
  const char* kSw = std::getenv("VPIPE_DSV41_IQ_SW");
  const std::string suf = kSw != nullptr ? kSw : "iq5_s4_r2";
  const bool v4 = suf != "iq";
  // the grid by the suffix: _sN (simdgroups) _rM (rows each), or iq6's
  // _gG_kK_rM (G row groups of M rows, K slices)
  int S_ = 4, R_ = 2, G_ = 0, K_ = 1;
  for (std::size_t q = 0; q + 2 < suf.size(); ++q) {
    if (suf[q] != '_' || suf[q + 2] < '0' || suf[q + 2] > '9') { continue; }
    const int v = std::atoi(suf.c_str() + q + 2);
    if (suf[q + 1] == 's') { S_ = v; }
    if (suf[q + 1] == 'r') { R_ = v; }
    if (suf[q + 1] == 'g') { G_ = v; }
    if (suf[q + 1] == 'k') { K_ = v; }
  }
  const unsigned rtg = (unsigned)(G_ > 0 ? G_ * R_ : S_ * R_);
  const unsigned sgn = (unsigned)(G_ > 0 ? G_ * K_ : S_);
  auto f_sw = lib.function("dsv41_moe_swiglu_" + suf);
  auto f_dn = lib.function("dsv41_moe_down_" + suf);
  auto f_gu = lib.function("dsv41_moe_gu_iq_grouped");
  auto f_dg = lib.function("dsv41_moe_down_iq_grouped");
  EXPECT_TRUE(f_sw.valid() && f_dn.valid() && f_gu.valid() && f_dg.valid());
  if (!f_sw.valid() || !f_dn.valid() || !f_gu.valid() || !f_dg.valid()) {
    return;
  }
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables_native();
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  std::mt19937 rng(11);
  const int H = 1024, I = 512, R = 32;   // R rows: one grouped tile
  const float lim = 0.0f;
  auto widen = [&](std::uint32_t t, const std::vector<std::uint8_t>& raw,
                  int N, int K) {
    std::vector<float> w((std::size_t)N * K);
    EXPECT_TRUE(ggml::decode(t, raw.data(), (std::int64_t)N * K, w.data()));
    return w;
  };
  auto rel = [](float got, float want) {
    return std::fabs(got - want) / std::max(1e-3f, std::fabs(want));
  };
  for (std::uint32_t t : {(std::uint32_t)ggml::kIQ2_XXS,
                          (std::uint32_t)ggml::kIQ2_XS,
                          (std::uint32_t)ggml::kIQ1_M,
                          (std::uint32_t)ggml::kIQ2_S,
                          (std::uint32_t)ggml::kIQ3_XXS}) {
    // slot 1 of two holds the expert (slot 0 garbage): the stride is used
    const ggml::Geom geo = ggml::geom(t);
    const std::size_t c13 = (std::size_t)I * (H / geo.elems) * geo.bytes;
    const std::size_t c2 = (std::size_t)H * (I / geo.elems) * geo.bytes;
    const auto r1 = random_iq_rows_(t, I, H, rng);
    const auto r3 = random_iq_rows_(t, I, H, rng);
    const auto r2 = random_iq_rows_(t, H, I, rng);
    auto p1 = mc->make_shared_buffer(2 * c13);
    auto p3 = mc->make_shared_buffer(2 * c13);
    auto p2 = mc->make_shared_buffer(2 * c2);
    std::memcpy(static_cast<std::uint8_t*>(p1.contents()) + c13, r1.data(),
                c13);
    std::memcpy(static_cast<std::uint8_t*>(p3.contents()) + c13, r3.data(),
                c13);
    std::memcpy(static_cast<std::uint8_t*>(p2.contents()) + c2, r2.data(),
                c2);
    const auto w1 = widen(t, r1, I, H), w3 = widen(t, r3, I, H);
    const auto w2 = widen(t, r2, H, I);
    // R rows of x (top_k 1: pair p is row p), weights
    auto bx = mc->make_shared_buffer((std::size_t)R * H * 2);
    auto bwt = mc->make_shared_buffer((std::size_t)R * 4);
    auto bslot = mc->make_shared_buffer((std::size_t)R * 4);
    std::vector<float> x((std::size_t)R * H), wt((std::size_t)R);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    for (int i = 0; i < R * H; ++i) {
      x[(std::size_t)i] = (float)(_Float16)U(rng);
      static_cast<_Float16*>(bx.contents())[i] = (_Float16)x[(std::size_t)i];
    }
    for (int p = 0; p < R; ++p) {
      wt[(std::size_t)p] = 0.5f + 0.5f * (float)p / R;
      static_cast<float*>(bwt.contents())[p] = wt[(std::size_t)p];
      static_cast<std::int32_t*>(bslot.contents())[p] = 1;
    }
    // the CPU's: act = wt * silu(w1 x) * (w3 x), rounded; out = w2 act
    std::vector<float> act((std::size_t)R * I), out((std::size_t)R * H);
    for (int p = 0; p < R; ++p) {
      for (int f = 0; f < I; ++f) {
        double g = 0, u = 0;
        for (int k = 0; k < H; ++k) {
          g += (double)w1[(std::size_t)f * H + k] * x[(std::size_t)p * H + k];
          u += (double)w3[(std::size_t)f * H + k] * x[(std::size_t)p * H + k];
        }
        const float a =
            (float)(wt[(std::size_t)p] * (g / (1 + std::exp(-g)) * u));
        act[(std::size_t)p * I + f] = (float)(_Float16)a;
      }
      for (int o = 0; o < H; ++o) {
        double s = 0;
        for (int k = 0; k < I; ++k) {
          s += (double)w2[(std::size_t)o * I + k] * act[(std::size_t)p * I + k];
        }
        out[(std::size_t)p * H + o] = (float)s;
      }
    }
    auto bact = mc->make_shared_buffer((std::size_t)R * I * 2);
    auto bout = mc->make_shared_buffer((std::size_t)R * H * 2);
    auto bactg = mc->make_shared_buffer((std::size_t)R * I * 2);
    auto boutg = mc->make_shared_buffer((std::size_t)R * H * 2);
    auto bsrow = mc->make_shared_buffer((std::size_t)R * 4);
    auto bsdst = mc->make_shared_buffer((std::size_t)R * 4);
    auto bt2e = mc->make_shared_buffer(4);
    for (int p = 0; p < R; ++p) {
      static_cast<std::int32_t*>(bsrow.contents())[p] = p;
      static_cast<std::int32_t*>(bsdst.contents())[p] = p;
    }
    static_cast<std::int32_t*>(bt2e.contents())[0] = 1;
    const long cw13 = (long)c13, cw2 = (long)c2;
    auto st = mc->make_command_stream();
    {
      auto enc = st.begin_compute();
      enc.set_function(f_sw);
      enc.set_buffer(0, p1);
      enc.set_buffer(1, p3);
      enc.set_buffer(2, btab);
      enc.set_buffer(3, bx);
      enc.set_buffer(4, bact);
      enc.set_buffer(5, bslot);
      enc.set_buffer(6, bwt);
      enc.set_constant(7, H);
      enc.set_constant(8, I);
      enc.set_constant(9, 1);
      enc.set_constant(10, lim);
      enc.set_constant(11, t);
      enc.set_constant(12, t);
      enc.set_constant(13, cw13);
      if (v4) {
        enc.dispatch({32, (unsigned)(((I + rtg - 1) / rtg) * sgn),
                      (unsigned)R}, {32, sgn, 1});
      } else {
        enc.dispatch({32, (unsigned)(((I + 7) / 8) * 2), (unsigned)R},
                     {32, 2, 1});
      }
      enc.set_function(f_gu);
      enc.set_buffer(0, p1);
      enc.set_buffer(1, p3);
      enc.set_buffer(2, btab);
      enc.set_buffer(3, bx);
      enc.set_buffer(4, bactg);
      enc.set_buffer(5, bsrow);
      enc.set_buffer(6, bwt);
      enc.set_buffer(7, bt2e);
      enc.set_constant(8, H);
      enc.set_constant(9, I);
      enc.set_constant(10, lim);
      enc.set_constant(11, t);
      enc.set_constant(12, t);
      enc.set_constant(13, cw13);
      enc.dispatch({(unsigned)I, 2, 2}, {32, 2, 2});
    }
    EXPECT_TRUE(st.commit().wait_ok());
    // the down halves from the CPU's act (each half judged alone)
    for (int i = 0; i < R * I; ++i) {
      const _Float16 a = (_Float16)act[(std::size_t)i];
      static_cast<_Float16*>(bact.contents())[i] = a;
    }
    std::size_t bad_sw = 0, bad_gu = 0;
    float worst = 0;
    auto st2 = mc->make_command_stream();
    auto bact2 = mc->make_shared_buffer((std::size_t)R * I * 2);
    {
      auto enc = st2.begin_compute();
      enc.set_function(f_sw);
      enc.set_buffer(0, p1);
      enc.set_buffer(1, p3);
      enc.set_buffer(2, btab);
      enc.set_buffer(3, bx);
      enc.set_buffer(4, bact2);
      enc.set_buffer(5, bslot);
      enc.set_buffer(6, bwt);
      enc.set_constant(7, H);
      enc.set_constant(8, I);
      enc.set_constant(9, 1);
      enc.set_constant(10, lim);
      enc.set_constant(11, t);
      enc.set_constant(12, t);
      enc.set_constant(13, cw13);
      if (v4) {
        enc.dispatch({32, (unsigned)(((I + rtg - 1) / rtg) * sgn),
                      (unsigned)R}, {32, sgn, 1});
      } else {
        enc.dispatch({32, (unsigned)(((I + 7) / 8) * 2), (unsigned)R},
                     {32, 2, 1});
      }
      enc.set_function(f_dn);
      enc.set_buffer(0, p2);
      enc.set_buffer(1, btab);
      enc.set_buffer(2, bact);
      enc.set_buffer(3, bout);
      enc.set_buffer(4, bslot);
      enc.set_constant(5, I);
      enc.set_constant(6, H);
      enc.set_constant(7, t);
      enc.set_constant(8, cw2);
      if (v4) {
        enc.dispatch({32, (unsigned)(((H + rtg - 1) / rtg) * sgn),
                      (unsigned)R}, {32, sgn, 1});
      } else {
        enc.dispatch({32, (unsigned)(((H + 7) / 8) * 2), (unsigned)R},
                     {32, 2, 1});
      }
      enc.set_function(f_dg);
      enc.set_buffer(0, p2);
      enc.set_buffer(1, btab);
      enc.set_buffer(2, bact);
      enc.set_buffer(3, boutg);
      enc.set_buffer(4, bsdst);
      enc.set_buffer(5, bt2e);
      enc.set_constant(6, I);
      enc.set_constant(7, H);
      enc.set_constant(8, t);
      enc.set_constant(9, cw2);
      enc.dispatch({(unsigned)H, 2, 2}, {32, 2, 2});
    }
    EXPECT_TRUE(st2.commit().wait_ok());
    const auto* ga = static_cast<const _Float16*>(bact2.contents());
    const auto* gg = static_cast<const _Float16*>(bactg.contents());
    for (int i = 0; i < R * I; ++i) {
      const float want = act[(std::size_t)i];
      const float e1 = rel((float)ga[i], want), e2 = rel((float)gg[i], want);
      worst = std::max({worst, e1, e2});
      bad_sw += e1 > 1e-2f;
      bad_gu += e2 > 1e-2f;
    }
    std::size_t bad_dn = 0, bad_dg = 0;
    const auto* go = static_cast<const _Float16*>(bout.contents());
    const auto* gq = static_cast<const _Float16*>(boutg.contents());
    for (int i = 0; i < R * H; ++i) {
      const float want = out[(std::size_t)i];
      const float e1 = rel((float)go[i], want), e2 = rel((float)gq[i], want);
      worst = std::max({worst, e1, e2});
      bad_dn += e1 > 1e-2f;
      bad_dg += e2 > 1e-2f;
    }
    // a verify's groups: the pairs grouped by expert on the GPU (all 32
    // share slot 1: four groups of 8), each group's expert read once and
    // every row's sum the decode's
    auto f_grp = lib.function("dsv41_moe_group_pairs");
    auto f_swg = lib.function("dsv41_moe_swiglu_iqg8");
    auto f_dng = lib.function("dsv41_moe_down_iqg8");
    EXPECT_TRUE(f_grp.valid() && f_swg.valid() && f_dng.valid());
    auto bgt = mc->make_shared_buffer((std::size_t)R * 10 * 4);
    auto bact3 = mc->make_shared_buffer((std::size_t)R * I * 2);
    auto bout3 = mc->make_shared_buffer((std::size_t)R * H * 2);
    auto st3 = mc->make_command_stream();
    {
      auto enc = st3.begin_compute();
      enc.set_function(f_grp);
      enc.set_buffer(0, bslot);
      enc.set_buffer(1, bgt);
      enc.set_constant(2, R);
      enc.set_constant(3, 8);
      enc.dispatch({1, 1, 1}, {1, 1, 1});
      enc.set_function(f_swg);
      enc.set_buffer(0, p1);
      enc.set_buffer(1, p3);
      enc.set_buffer(2, btab);
      enc.set_buffer(3, bx);
      enc.set_buffer(4, bact3);
      enc.set_buffer(5, bgt);
      enc.set_buffer(6, bwt);
      enc.set_constant(7, H);
      enc.set_constant(8, I);
      enc.set_constant(9, 1);
      enc.set_constant(10, lim);
      enc.set_constant(11, t);
      enc.set_constant(12, t);
      enc.set_constant(13, cw13);
      enc.dispatch({32, (unsigned)(((I + 7) / 8) * 4), (unsigned)R},
                   {32, 4, 1});
      enc.set_function(f_dng);
      enc.set_buffer(0, p2);
      enc.set_buffer(1, btab);
      enc.set_buffer(2, bact);
      enc.set_buffer(3, bout3);
      enc.set_buffer(4, bgt);
      enc.set_constant(5, I);
      enc.set_constant(6, H);
      enc.set_constant(7, t);
      enc.set_constant(8, cw2);
      enc.dispatch({32, (unsigned)(((H + 7) / 8) * 4), (unsigned)R},
                   {32, 4, 1});
    }
    EXPECT_TRUE(st3.commit().wait_ok());
    // A verify's rows ARE decode rows (what keeps DSpark's tokens the
    // plain decode's): the groups' outputs the decode kernels' bit for
    // bit -- the model's two (iq5_s4_r2, and its _tg twin it takes for
    // IQ1_M / IQ3_XXS), which must agree with each other too.
    if (kSw == nullptr) {
      auto f_swt = lib.function("dsv41_moe_swiglu_iq5_s4_r2_tg");
      auto f_dnt = lib.function("dsv41_moe_down_iq5_s4_r2_tg");
      EXPECT_TRUE(f_swt.valid() && f_dnt.valid());
      auto bact4 = mc->make_shared_buffer((std::size_t)R * I * 2);
      auto bout4 = mc->make_shared_buffer((std::size_t)R * H * 2);
      auto st4 = mc->make_command_stream();
      {
        auto enc = st4.begin_compute();
        enc.set_function(f_swt);
        enc.set_buffer(0, p1);
        enc.set_buffer(1, p3);
        enc.set_buffer(2, btab);
        enc.set_buffer(3, bx);
        enc.set_buffer(4, bact4);
        enc.set_buffer(5, bslot);
        enc.set_buffer(6, bwt);
        enc.set_constant(7, H);
        enc.set_constant(8, I);
        enc.set_constant(9, 1);
        enc.set_constant(10, lim);
        enc.set_constant(11, t);
        enc.set_constant(12, t);
        enc.set_constant(13, cw13);
        enc.dispatch({32, (unsigned)(((I + 7) / 8) * 4), (unsigned)R},
                     {32, 4, 1});
        enc.set_function(f_dnt);
        enc.set_buffer(0, p2);
        enc.set_buffer(1, btab);
        enc.set_buffer(2, bact);
        enc.set_buffer(3, bout4);
        enc.set_buffer(4, bslot);
        enc.set_constant(5, I);
        enc.set_constant(6, H);
        enc.set_constant(7, t);
        enc.set_constant(8, cw2);
        enc.dispatch({32, (unsigned)(((H + 7) / 8) * 4), (unsigned)R},
                     {32, 4, 1});
      }
      EXPECT_TRUE(st4.commit().wait_ok());
      const std::size_t na = (std::size_t)R * I * 2;
      const std::size_t no = (std::size_t)R * H * 2;
      auto same = [](const metal_compute::SharedBuffer& a,
                     const metal_compute::SharedBuffer& b,
                     std::size_t n) {
        return std::memcmp(a.contents(), b.contents(), n) == 0;
      };
      const bool g_act = same(bact3, bact2, na);
      const bool g_out = same(bout3, bout, no);
      const bool t_act = same(bact4, bact2, na);
      const bool t_out = same(bout4, bout, no);
      std::fprintf(stderr, "  type %u: verify groups == decode rows: %s / %s;"
                           " _tg == plain: %s / %s\n", t,
                   g_act ? "same" : "DIFFERENT", g_out ? "same" : "DIFFERENT",
                   t_act ? "same" : "DIFFERENT", t_out ? "same" : "DIFFERENT");
      EXPECT_TRUE(g_act && g_out && t_act && t_out);
    }
    std::size_t bad_swg = 0, bad_dng = 0;
    const auto* g3 = static_cast<const _Float16*>(bact3.contents());
    for (int i = 0; i < R * I; ++i) {
      const float e = rel((float)g3[i], act[(std::size_t)i]);
      worst = std::max(worst, e);
      bad_swg += e > 1e-2f;
    }
    const auto* o3 = static_cast<const _Float16*>(bout3.contents());
    for (int i = 0; i < R * H; ++i) {
      const float e = rel((float)o3[i], out[(std::size_t)i]);
      worst = std::max(worst, e);
      bad_dng += e > 1e-2f;
    }
    EXPECT_TRUE(bad_swg * 1000 < (std::size_t)R * I);
    EXPECT_TRUE(bad_dng * 1000 < (std::size_t)R * H);
    std::fprintf(stderr, "  type %u: off by >1%%: swiglu %zu, gu grouped "
                         "%zu, down %zu, down grouped %zu, verify groups "
                         "%zu / %zu (worst %.2g)\n",
                 t, bad_sw, bad_gu, bad_dn, bad_dg, bad_swg, bad_dng,
                 (double)worst);
    // half rounding of a near-zero value aside, nearly every one agrees
    EXPECT_TRUE(bad_sw * 1000 < (std::size_t)R * I);
    EXPECT_TRUE(bad_gu * 1000 < (std::size_t)R * I);
    EXPECT_TRUE(bad_dn * 1000 < (std::size_t)R * H);
    EXPECT_TRUE(bad_dg * 1000 < (std::size_t)R * H);
  }
}

// VPIPE_DSV41_IQ_BENCH=1: the routed experts' decode kernels at the
// release's shapes, 6 pairs a layer over experts not read since (a pool
// past the caches), their GB/s.
TEST(deepseek41_gguf, native_expert_kernels_bench)
{
  if (std::getenv("VPIPE_DSV41_IQ_BENCH") == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library(std::getenv("VPIPE_DSV41_IQ_BENCH_BF16")
                                  ? "deepseek_v41_bf16" : "deepseek_v41");
  const char* sw_name = std::getenv("VPIPE_DSV41_IQ_BENCH_SW");
  const char* dn_name = std::getenv("VPIPE_DSV41_IQ_BENCH_DN");
  auto f_sw = lib.function(sw_name ? sw_name : "dsv41_moe_swiglu_iq");
  auto f_dn = lib.function(dn_name ? dn_name : "dsv41_moe_down_iq");
  EXPECT_TRUE(f_sw.valid() && f_dn.valid());
  if (!f_sw.valid() || !f_dn.valid()) { return; }
  const bool wide =
      sw_name != nullptr && std::strstr(sw_name, "iq2") != nullptr;
  const bool grouped =
      sw_name != nullptr && std::strstr(sw_name, "iqg") != nullptr;
  const char* rsv = grouped ? "_r2"
                    : sw_name != nullptr ? std::strstr(sw_name, "_r") : nullptr;
  const int RS = rsv != nullptr ? std::atoi(rsv + 2) : 4;
  int NS = 4, RG = 0, KSP = 1;
  for (const char* q = sw_name; q != nullptr && *q != 0; ++q) {
    if (q[0] == '_' && q[1] == 's' && q[2] >= '0' && q[2] <= '9') {
      NS = std::atoi(q + 2);
    }
    if (q[0] == '_' && q[1] == 'g' && q[2] >= '0' && q[2] <= '9') {
      RG = std::atoi(q + 2);
    }
    if (q[0] == '_' && q[1] == 'k' && q[2] >= '0' && q[2] <= '9') {
      KSP = std::atoi(q + 2);
    }
  }
  // iq6: RG row groups of RS rows, KSP slices each
  const int rows_tg = RG > 0 ? RG * RS : NS * RS;
  const int sgs = RG > 0 ? RG * KSP : NS;
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables_native();
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  const bool big = std::getenv("VPIPE_DSV41_IQ_BENCH_BIG") != nullptr;
  const int H = 5120, I = 2304, P = big ? 384 : 6, S = 96, L = big ? 1 : 64;
  std::mt19937 rng(3);
  for (std::uint32_t t : {(std::uint32_t)ggml::kIQ2_XXS,
                          (std::uint32_t)ggml::kIQ2_XS,
                          (std::uint32_t)ggml::kIQ1_M,
                          (std::uint32_t)ggml::kIQ2_S,
                          (std::uint32_t)ggml::kIQ3_XXS}) {
    const ggml::Geom geo = ggml::geom(t);
    const std::size_t c13 = (std::size_t)I * (H / geo.elems) * geo.bytes;
    const std::size_t c2 = (std::size_t)H * (I / geo.elems) * geo.bytes;
    auto p1 = mc->make_shared_buffer(S * c13), p3 = mc->make_shared_buffer(
        S * c13), p2 = mc->make_shared_buffer(S * c2);
    for (auto* b : {&p1, &p3}) {
      for (int s = 0; s < S; ++s) {
        const auto r = random_iq_rows_(t, I, H, rng);
        std::memcpy(static_cast<std::uint8_t*>(b->contents()) + s * c13,
                    r.data(), c13);
      }
    }
    for (int s = 0; s < S; ++s) {
      const auto r = random_iq_rows_(t, H, I, rng);
      std::memcpy(static_cast<std::uint8_t*>(p2.contents()) + s * c2,
                  r.data(), c2);
    }
    auto bx = mc->make_shared_buffer((std::size_t)H * 2);
    for (int i = 0; i < H; ++i) {
      static_cast<_Float16*>(bx.contents())[i] = (_Float16)0.01f;
    }
    auto bact = mc->make_shared_buffer((std::size_t)P * I * 2);
    auto bout = mc->make_shared_buffer((std::size_t)P * H * 2);
    auto bwt = mc->make_shared_buffer((std::size_t)P * 4);
    std::vector<metal_compute::SharedBuffer> slots;
    for (int l = 0; l < L; ++l) {
      slots.push_back(mc->make_shared_buffer((std::size_t)P * 4));
      for (int p = 0; p < P; ++p) {
        static_cast<std::int32_t*>(slots.back().contents())[p] =
            std::getenv("VPIPE_DSV41_IQ_BENCH_HOT") ? p : (l * P + p) % S;
      }
      if (grouped) {
        // a group a pair: {slot, 1, pair}
        auto gb = mc->make_shared_buffer((std::size_t)P * 40);
        auto* g = static_cast<std::int32_t*>(gb.contents());
        const auto* sl = static_cast<const std::int32_t*>(
            slots.back().contents());
        for (int p = 0; p < P; ++p) {
          g[p * 10] = sl[p];
          g[p * 10 + 1] = 1;
          g[p * 10 + 2] = p;
        }
        slots.back() = std::move(gb);
      }
    }
    for (int p = 0; p < P; ++p) {
      static_cast<float*>(bwt.contents())[p] = 1.0f;
    }
    const float lim = 10.0f;
    const long cw13 = (long)c13, cw2 = (long)c2;
    const bool indep = std::getenv("VPIPE_DSV41_IQ_BENCH_INDEP") != nullptr;
    std::vector<metal_compute::SharedBuffer> acts, outs;
    for (int l = 0; indep && l < L; ++l) {
      acts.push_back(mc->make_shared_buffer((std::size_t)P * I * 2));
      outs.push_back(mc->make_shared_buffer((std::size_t)P * H * 2));
    }
    auto run = [&]() {
      auto st = mc->make_command_stream();
      {
        auto enc = st.begin_compute();
        auto cs = enc.concurrent_scope(indep);
        for (int l = 0; l < L; ++l) {
          enc.set_function(f_sw);
          enc.set_buffer(0, p1);
          enc.set_buffer(1, p3);
          enc.set_buffer(2, btab);
          enc.set_buffer(3, bx);
          enc.set_buffer(4, indep ? acts[(std::size_t)l] : bact);
          enc.set_buffer(5, slots[(std::size_t)l]);
          enc.set_buffer(6, bwt);
          enc.set_constant(7, H);
          enc.set_constant(8, I);
          enc.set_constant(9, P);
          enc.set_constant(10, lim);
          enc.set_constant(11, t);
          enc.set_constant(12, t);
          enc.set_constant(13, cw13);
          if (rsv != nullptr) {
            enc.dispatch({32, (unsigned)(((I + rows_tg - 1) / rows_tg) * sgs),
                          (unsigned)P}, {32, (unsigned)sgs, 1});
          } else if (wide) {
            enc.dispatch({32, (unsigned)(((I + 15) / 16) * 4), (unsigned)P},
                         {32, 4, 1});
          } else {
            enc.dispatch({32, (unsigned)(((I + 7) / 8) * 2), (unsigned)P},
                         {32, 2, 1});
          }
          enc.set_function(f_dn);
          enc.set_buffer(0, p2);
          enc.set_buffer(1, btab);
          enc.set_buffer(2, indep ? acts[(std::size_t)l] : bact);
          enc.set_buffer(3, indep ? outs[(std::size_t)l] : bout);
          enc.set_buffer(4, slots[(std::size_t)l]);
          enc.set_constant(5, I);
          enc.set_constant(6, H);
          enc.set_constant(7, t);
          enc.set_constant(8, cw2);
          if (rsv != nullptr) {
            enc.dispatch({32, (unsigned)(((H + rows_tg - 1) / rows_tg) * sgs),
                          (unsigned)P}, {32, (unsigned)sgs, 1});
          } else if (wide) {
            enc.dispatch({32, (unsigned)(((H + 15) / 16) * 4), (unsigned)P},
                         {32, 4, 1});
          } else {
            enc.dispatch({32, (unsigned)(((H + 7) / 8) * 2), (unsigned)P},
                         {32, 2, 1});
          }
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      EXPECT_TRUE(st.commit().wait_ok());
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count();
    };
    run();
    double best = 1e9;
    for (int k = 0; k < 5; ++k) { best = std::min(best, run()); }
    const double bytes = (double)L * P * (2.0 * c13 + c2);
    std::fprintf(stderr, "  type %u: %d layers x %d pairs in %.2f ms = %.3f "
                         "ms a layer, %.0f GB/s\n", t, L, P, best, best / L,
                 bytes / best / 1e6);
  }
}

// VPIPE_DSV41_IQ_BENCH=1: what a dispatch costs by itself -- N tiny
// dispatches in one encoder, each on the same buffer (serial, as a
// forward's are) or each on its own (independent), and in a concurrent
// encoder.
TEST(deepseek41_gguf, dispatch_cost_bench)
{
  if (std::getenv("VPIPE_DSV41_IQ_BENCH") == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41");
  auto f = lib.function("dsv41_fill32");
  const int N = 1280;
  std::vector<metal_compute::SharedBuffer> bs;
  for (int i = 0; i < N; ++i) {
    bs.push_back(mc->make_shared_buffer(4096 * 4));
  }
  for (int tgs : {1, 64}) {
    for (int mode = 0; mode < 3; ++mode) {
      auto run = [&]() {
        auto st = mc->make_command_stream();
        {
          auto enc = st.begin_compute();
          auto cs = enc.concurrent_scope(mode == 2);
          for (int i = 0; i < N; ++i) {
            enc.set_function(f);
            enc.set_buffer(0, bs[(std::size_t)(mode == 0 ? 0 : i)]);
            enc.set_constant(1, i);
            enc.set_constant(2, 64 * tgs);
            enc.dispatch({(unsigned)(64 * tgs), 1, 1}, {64, 1, 1});
          }
        }
        const auto t0 = std::chrono::steady_clock::now();
        EXPECT_TRUE(st.commit().wait_ok());
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
      };
      run();
      double best = 1e9;
      for (int k = 0; k < 5; ++k) { best = std::min(best, run()); }
      static const char* kM[3] = {"same buffer", "own buffers",
                                  "own, concurrent"};
      std::fprintf(stderr, "  %d dispatches of %d threadgroups, %s: %.2f ms "
                           "= %.1f us each\n", N, tgs, kM[mode], best,
                   best * 1000 / N);
    }
  }
}

// VPIPE_DSV41_IQ_BENCH=1: the MLX-style affine expert kernels (an oQ2 /
// oQ4 pack's) at the same shapes, for a reference.
TEST(deepseek41_gguf, affine_expert_kernels_bench)
{
  if (std::getenv("VPIPE_DSV41_IQ_BENCH") == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41");
  const int H = 5120, I = 2304, P = 6, S = 96, L = 64, gs = 64;
  for (int B : {2, 4}) {
    auto f_sw = lib.function("dsv41_moe_swiglu_q" + std::to_string(B));
    auto f_dn = lib.function("dsv41_moe_down_q" + std::to_string(B));
    EXPECT_TRUE(f_sw.valid() && f_dn.valid());
    if (!f_sw.valid() || !f_dn.valid()) { return; }
    const std::size_t cw = (std::size_t)I * H * B / 8;
    const std::size_t cs = (std::size_t)I * (H / gs);   // elements
    std::vector<metal_compute::SharedBuffer> pool;
    for (int k = 0; k < 9; ++k) {
      const bool codes = k % 3 == 0;
      pool.push_back(mc->make_shared_buffer(S * (codes ? cw : cs * 2)));
      std::memset(pool.back().contents(), k % 3 == 0 ? 0x5a : 0x3c,
                  pool.back().byte_size());
    }
    auto bx = mc->make_shared_buffer((std::size_t)H * 2);
    std::memset(bx.contents(), 0, bx.byte_size());
    auto bact = mc->make_shared_buffer((std::size_t)P * I * 2);
    auto bout = mc->make_shared_buffer((std::size_t)P * H * 2);
    auto bwt = mc->make_shared_buffer((std::size_t)P * 4);
    std::vector<metal_compute::SharedBuffer> slots;
    for (int l = 0; l < L; ++l) {
      slots.push_back(mc->make_shared_buffer((std::size_t)P * 4));
      for (int p = 0; p < P; ++p) {
        static_cast<std::int32_t*>(slots.back().contents())[p] =
            (l * P + p) % S;
      }
    }
    const float lim = 10.0f;
    auto run = [&]() {
      auto st = mc->make_command_stream();
      {
        auto enc = st.begin_compute();
        for (int l = 0; l < L; ++l) {
          enc.set_function(f_sw);
          for (int k = 0; k < 6; ++k) {
            enc.set_buffer(k, pool[(std::size_t)k]);
          }
          enc.set_buffer(6, bx);
          enc.set_buffer(7, bact);
          enc.set_buffer(8, slots[(std::size_t)l]);
          enc.set_buffer(9, bwt);
          enc.set_constant(10, H);
          enc.set_constant(11, I);
          enc.set_constant(12, P);
          enc.set_constant(13, lim);
          enc.set_constant(14, gs);
          enc.set_constant(15, (long)cw);
          enc.set_constant(16, (long)cs);
          enc.dispatch({32, (unsigned)(((I + 7) / 8) * 2), (unsigned)P},
                       {32, 2, 1});
          enc.set_function(f_dn);
          for (int k = 0; k < 3; ++k) {
            enc.set_buffer(k, pool[(std::size_t)(6 + k)]);
          }
          enc.set_buffer(3, bact);
          enc.set_buffer(4, bout);
          enc.set_buffer(5, slots[(std::size_t)l]);
          enc.set_constant(6, I);
          enc.set_constant(7, H);
          enc.set_constant(8, gs);
          enc.set_constant(9, (long)cw);
          enc.set_constant(10, (long)cs);
          enc.dispatch({32, (unsigned)(((H + 7) / 8) * 2), (unsigned)P},
                       {32, 2, 1});
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      EXPECT_TRUE(st.commit().wait_ok());
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count();
    };
    run();
    double best = 1e9;
    for (int k = 0; k < 5; ++k) { best = std::min(best, run()); }
    const double bytes = (double)L * P * 3.0 * (cw + 2.0 * cs * 2);
    std::fprintf(stderr, "  affine %d-bit: %d layers x %d pairs in %.2f ms = "
                         "%.3f ms a layer, %.0f GB/s\n", B, L, P, best,
                 best / L, bytes / best / 1e6);
  }
}

// wo_a's groups over affine codes in one dispatch: the per-group GEMV's
// bits, group by group (dsv41_gemv_q*_b against dsv41_gemv_q*).
TEST(deepseek41_gguf, batched_affine_gemv_is_the_per_group_one)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  const int B = 5, gs = 32, K = 4096, rows = 1024, G = 8;
  std::mt19937 rng(5);
  const std::size_t wwords = (std::size_t)G * rows * K * B / 32;
  const std::size_t ns = (std::size_t)G * rows * (K / gs);
  auto bw = mc->make_shared_buffer(wwords * 4);
  auto bs = mc->make_shared_buffer(ns * 2), bb = mc->make_shared_buffer(ns * 2);
  for (std::size_t i = 0; i < wwords; ++i) {
    static_cast<std::uint32_t*>(bw.contents())[i] = (std::uint32_t)rng();
  }
  auto bf = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (std::uint16_t)(u >> 16);
  };
  std::uniform_real_distribution<float> U(-1.0f, 1.0f);
  for (std::size_t i = 0; i < ns; ++i) {
    static_cast<std::uint16_t*>(bs.contents())[i] = bf(U(rng) * 0.01f);
    static_cast<std::uint16_t*>(bb.contents())[i] = bf(U(rng) * 0.1f);
  }
  for (int M : {1, 3}) {
    for (int ks : {1, 4}) {
      const std::string sfx = std::string(M == 1 ? "" : "_rows") +
                              (ks == 4 ? "_k4" : "");
      auto f1 = lib.function("dsv41_gemv_q5" + sfx);
      auto fb = lib.function("dsv41_gemv_q5" + sfx + "_b");
      EXPECT_TRUE(f1.valid() && fb.valid());
      if (!f1.valid() || !fb.valid()) { return; }
      auto bx = mc->make_shared_buffer((std::size_t)G * M * K * 2);
      for (int i = 0; i < G * M * K; ++i) {
        static_cast<std::uint16_t*>(bx.contents())[i] = bf(U(rng));
      }
      auto y1 = mc->make_shared_buffer((std::size_t)G * M * rows * 2);
      auto yb = mc->make_shared_buffer((std::size_t)G * M * rows * 2);
      auto st = mc->make_command_stream();
      {
        auto enc = st.begin_compute();
        for (int g = 0; g < G; ++g) {
          enc.set_function(f1);
          enc.set_buffer(0, bw, (std::size_t)g * rows * K * B / 8);
          enc.set_buffer(1, bs, (std::size_t)g * rows * (K / gs) * 2);
          enc.set_buffer(2, bb, (std::size_t)g * rows * (K / gs) * 2);
          enc.set_buffer(3, bx, (std::size_t)g * M * K * 2);
          enc.set_buffer(4, y1, (std::size_t)g * M * rows * 2);
          enc.set_constant(5, K);
          enc.set_constant(6, rows);
          enc.set_constant(7, M);
          enc.set_constant(8, gs);
          enc.dispatch({32, (unsigned)(((rows + 7) / 8) * 2 * ks), 1},
                       {32, (unsigned)(2 * ks), 1});
        }
        enc.set_function(fb);
        enc.set_buffer(0, bw);
        enc.set_buffer(1, bs);
        enc.set_buffer(2, bb);
        enc.set_buffer(3, bx);
        enc.set_buffer(4, yb);
        enc.set_constant(5, K);
        enc.set_constant(6, rows);
        enc.set_constant(7, M);
        enc.set_constant(8, gs);
        enc.set_constant(9, (long)M * K);
        enc.set_constant(10, (long)rows * K * B / 32);
        enc.set_constant(11, (long)rows * (K / gs));
        enc.set_constant(12, (long)M * rows);
        enc.dispatch({32, (unsigned)(((rows + 7) / 8) * 2 * ks), (unsigned)G},
                     {32, (unsigned)(2 * ks), 1});
      }
      EXPECT_TRUE(st.commit().wait_ok());
      EXPECT_TRUE(std::memcmp(y1.contents(), yb.contents(),
                              y1.byte_size()) == 0);
    }
  }
}

// Random rows of any native format: random bytes, each block's f16 steps
// (scale, min) put in range -- sane, a zero one now and then.
static std::vector<std::uint8_t>
random_ggml_rows_(std::uint32_t type, int N, int K, std::mt19937& rng)
{
  const ggml::Geom geo = ggml::geom(type);
  const int nb = K / geo.elems;
  std::vector<std::uint8_t> rows((std::size_t)N * nb * geo.bytes);
  for (auto& x : rows) { x = (std::uint8_t)rng(); }
  auto put = [&](std::uint8_t* p, int e0) {
    _Float16 d = (_Float16)(std::ldexp(1.0 + (rng() % 1000) / 1000.0,
                                       -(int)(rng() % 4) - e0));
    if (rng() % 17 == 0) { d = (_Float16)0.0f; }
    std::memcpy(p, &d, 2);
  };
  for (int i = 0; i < N * nb; ++i) {
    std::uint8_t* b = rows.data() + (std::size_t)i * geo.bytes;
    switch (type) {
      // steps as far below 1 as each format's codes reach above it, so a
      // down projection's sums stay inside half's range
      case ggml::kQ3_K: put(b + 108, 8); break;
      case ggml::kQ6_K: put(b + 208, 12); break;
      case ggml::kQ4_K: put(b, 8); put(b + 2, 8); break;
      case ggml::kQ5_K: put(b, 10); put(b + 2, 10); break;
      case ggml::kIQ4_XS: put(b, 11); break;
      case ggml::kIQ1_M: {
        std::uint8_t tmp[2];
        put(tmp, 6);
        std::uint16_t h;
        std::memcpy(&h, tmp, 2);
        for (int k = 0; k < 4; ++k) {
          std::uint16_t sc;
          std::memcpy(&sc, b + 48 + 2 * k, 2);
          sc = (std::uint16_t)((sc & 0x0fff) |
                               (((h >> (4 * k)) & 0xf) << 12));
          std::memcpy(b + 48 + 2 * k, &sc, 2);
        }
        break;
      }
      default: put(b, 6); break;
    }
  }
  return rows;
}

// Every native format's GPU reader (ops/ggml_native.h) against the CPU's
// decode (ggml-blocks.cc), value for value.
TEST(ggml_native, every_format_reads_as_the_cpu_decodes)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("ggml_native");
  auto fn = lib.function("ggml_native_dequant");
  auto fn8 = lib.function("ggml_native_dequant8");
  EXPECT_TRUE(fn.valid() && fn8.valid());
  if (!fn.valid() || !fn8.valid()) { return; }
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables_native();
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  std::mt19937 rng(17);
  const int N = 24, K = 1024;
  for (std::uint32_t t : {8u, 20u, 11u, 12u, 13u, 14u, 23u, 18u, 21u, 16u,
                          17u, 22u, 29u}) {
    const auto rows = random_ggml_rows_(t, N, K, rng);
    std::vector<float> want((std::size_t)N * K);
    EXPECT_TRUE(ggml::decode(t, rows.data(), (std::int64_t)N * K,
                             want.data()));
    auto bs = mc->make_shared_buffer(rows.size());
    std::memcpy(bs.contents(), rows.data(), rows.size());
    auto bo = mc->make_shared_buffer((std::size_t)N * K * 4);
    auto bo8 = mc->make_shared_buffer((std::size_t)N * K * 4);
    auto st = mc->make_command_stream();
    {
      auto enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, bs);
      enc.set_buffer(1, btab);
      enc.set_buffer(2, bo);
      enc.set_constant(3, t);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.dispatch({(unsigned)(K / 32), (unsigned)N, 1}, {32, 1, 1});
      enc.set_function(fn8);
      enc.set_buffer(0, bs);
      enc.set_buffer(1, btab);
      enc.set_buffer(2, bo8);
      enc.set_constant(3, t);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {32, 1, 1});
    }
    EXPECT_TRUE(st.commit().wait_ok());
    std::size_t bad = 0, bad8 = 0;
    for (int k = 0; k < 2; ++k) {
      const auto* got = static_cast<const float*>((k == 0 ? bo : bo8)
                                                      .contents());
      for (std::size_t i = 0; i < want.size(); ++i) {
        (k == 0 ? bad : bad8) += got[i] != want[i];
      }
    }
    std::fprintf(stderr, "  %-8s w32 %zu, w8 %zu of %zu values differ\n",
                 ggml::name(t), bad, bad8, want.size());
    EXPECT_TRUE(bad == 0 && bad8 == 0);
  }
}

// A Qwen MoE's routed experts from a GGUF's own blocks
// (qwen_moe_native.metal): the decode's gathers and a prefill's grouped
// GEMMs against ggml's decode on the CPU -- every native format, and a
// layer whose gate, up and down differ.
TEST(ggml_native, qwen_moe_native_kernels_are_the_cpus)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("qwen_moe_native");
  auto f_sw = lib.function("qwen_native_gather_swiglu");
  auto f_dn = lib.function("qwen_native_gather_down");
  auto f_gs = lib.function("qwen_native_grouped_swiglu");
  auto f_gd = lib.function("qwen_native_grouped_down");
  EXPECT_TRUE(f_sw.valid() && f_dn.valid() && f_gs.valid() && f_gd.valid());
  if (!f_sw.valid() || !f_dn.valid() || !f_gs.valid() || !f_gd.valid()) {
    return;
  }
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables_native();
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  std::mt19937 rng(23);
  const int H = 512, I = 256, R = 32;
  auto rel = [](float got, float want) {
    return std::fabs(got - want) / std::max(1e-3f, std::fabs(want));
  };
  std::vector<std::array<std::uint32_t, 3>> sets;
  for (std::uint32_t t : {8u, 20u, 11u, 12u, 13u, 14u, 23u, 18u, 21u, 16u,
                          17u, 22u, 29u}) {
    sets.push_back({t, t, t});
  }
  sets.push_back({21u, 18u, 14u});   // IQ3_S gate, IQ3_XXS up, Q6_K down
  sets.push_back({12u, 12u, 23u});   // Q4_K gate|up, IQ4_XS down
  for (const auto& ts : sets) {
    const std::uint32_t tg = ts[0], tu = ts[1], td = ts[2];
    const auto rg = random_ggml_rows_(tg, I, H, rng);
    const auto ru = random_ggml_rows_(tu, I, H, rng);
    const auto rd = random_ggml_rows_(td, H, I, rng);
    // slot 1 of two holds the expert: the pools' strides are used
    auto pool = [&](const std::vector<std::uint8_t>& r) {
      auto b = mc->make_shared_buffer(2 * r.size());
      std::memset(b.contents(), 0x55, b.byte_size());
      std::memcpy(static_cast<std::uint8_t*>(b.contents()) + r.size(),
                  r.data(), r.size());
      return b;
    };
    auto pg = pool(rg), pu = pool(ru), pd = pool(rd);
    std::vector<float> wg((std::size_t)I * H), wu(wg.size()),
        wd((std::size_t)H * I);
    EXPECT_TRUE(ggml::decode(tg, rg.data(), (std::int64_t)I * H, wg.data()));
    EXPECT_TRUE(ggml::decode(tu, ru.data(), (std::int64_t)I * H, wu.data()));
    EXPECT_TRUE(ggml::decode(td, rd.data(), (std::int64_t)H * I, wd.data()));
    // inputs scaled so the sums stay well inside half's range
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<float> x((std::size_t)R * H);
    auto bx = mc->make_shared_buffer(x.size() * 2);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = (float)(_Float16)(U(rng) * 0.25f);
      static_cast<_Float16*>(bx.contents())[i] = (_Float16)x[i];
    }
    std::vector<float> act((std::size_t)R * I), out((std::size_t)R * H);
    for (int p = 0; p < R; ++p) {
      for (int f = 0; f < I; ++f) {
        double g = 0, u = 0;
        for (int k = 0; k < H; ++k) {
          g += (double)wg[(std::size_t)f * H + k] * x[(std::size_t)p * H + k];
          u += (double)wu[(std::size_t)f * H + k] * x[(std::size_t)p * H + k];
        }
        act[(std::size_t)p * I + f] =
            (float)(_Float16)(float)(g / (1 + std::exp(-g)) * u);
      }
      for (int o = 0; o < H; ++o) {
        double s = 0;
        for (int k = 0; k < I; ++k) {
          s += (double)wd[(std::size_t)o * I + k] * act[(std::size_t)p * I + k];
        }
        out[(std::size_t)p * H + o] = (float)s;
      }
    }
    // the decode: pair p reads row p (top_k 1), slot 1
    auto bslot = mc->make_shared_buffer((std::size_t)R * 4);
    auto bsrow = mc->make_shared_buffer((std::size_t)R * 4);
    auto bt2e = mc->make_shared_buffer(4);
    for (int p = 0; p < R; ++p) {
      static_cast<std::int32_t*>(bslot.contents())[p] = 1;
      static_cast<std::int32_t*>(bsrow.contents())[p] = p;
    }
    static_cast<std::int32_t*>(bt2e.contents())[0] = 1;
    // the down kernels read the CPU's act: each half judged alone
    auto bacpu = mc->make_shared_buffer((std::size_t)R * I * 2);
    for (std::size_t i = 0; i < act.size(); ++i) {
      static_cast<_Float16*>(bacpu.contents())[i] = (_Float16)act[i];
    }
    auto bact = mc->make_shared_buffer((std::size_t)R * I * 2);
    auto bgact = mc->make_shared_buffer((std::size_t)R * I * 2);
    auto bout = mc->make_shared_buffer((std::size_t)R * H * 2);
    auto bgout = mc->make_shared_buffer((std::size_t)R * H * 2);
    const long cwg = (long)rg.size(), cwu = (long)ru.size(),
               cwd = (long)rd.size();
    auto st = mc->make_command_stream();
    {
      auto enc = st.begin_compute();
      enc.set_function(f_sw);
      enc.set_buffer(0, pg); enc.set_buffer(1, pu); enc.set_buffer(2, btab);
      enc.set_buffer(3, bx); enc.set_buffer(4, bact); enc.set_buffer(5, bslot);
      enc.set_constant(6, H); enc.set_constant(7, I); enc.set_constant(8, 1);
      enc.set_constant(9, tg); enc.set_constant(10, tu);
      enc.set_constant(11, cwg); enc.set_constant(12, cwu);
      enc.dispatch({32, (unsigned)(((I + 7) / 8) * 4), (unsigned)R},
                   {32, 4, 1});
      enc.set_function(f_dn);
      enc.set_buffer(0, pd); enc.set_buffer(1, btab); enc.set_buffer(2, bacpu);
      enc.set_buffer(3, bout); enc.set_buffer(4, bslot);
      enc.set_constant(5, I); enc.set_constant(6, H); enc.set_constant(7, td);
      enc.set_constant(8, cwd);
      enc.dispatch({32, (unsigned)(((H + 7) / 8) * 4), (unsigned)R},
                   {32, 4, 1});
      enc.set_function(f_gs);
      enc.set_buffer(0, pg); enc.set_buffer(1, pu); enc.set_buffer(2, btab);
      enc.set_buffer(3, bx); enc.set_buffer(4, bgact); enc.set_buffer(5, bsrow);
      enc.set_buffer(6, bt2e);
      enc.set_constant(7, H); enc.set_constant(8, I);
      enc.set_constant(9, tg); enc.set_constant(10, tu);
      enc.set_constant(11, cwg); enc.set_constant(12, cwu);
      enc.dispatch({(unsigned)I, 2, 2}, {32, 2, 2});
      enc.set_function(f_gd);
      enc.set_buffer(0, pd); enc.set_buffer(1, btab); enc.set_buffer(2, bacpu);
      enc.set_buffer(3, bgout); enc.set_buffer(4, bsrow);
      enc.set_buffer(5, bt2e);
      enc.set_constant(6, I); enc.set_constant(7, H); enc.set_constant(8, td);
      enc.set_constant(9, cwd);
      enc.dispatch({(unsigned)H, 2, 2}, {32, 2, 2});
    }
    EXPECT_TRUE(st.commit().wait_ok());
    std::size_t bad[4] = {0, 0, 0, 0};
    float worst = 0.0f;
    const _Float16* got[4] = {static_cast<const _Float16*>(bact.contents()),
                              static_cast<const _Float16*>(bgact.contents()),
                              static_cast<const _Float16*>(bout.contents()),
                              static_cast<const _Float16*>(bgout.contents())};
    for (int k = 0; k < 4; ++k) {
      const std::vector<float>& want = k < 2 ? act : out;
      for (std::size_t i = 0; i < want.size(); ++i) {
        const float e = rel((float)got[k][i], want[i]);
        worst = std::max(worst, e);
        bad[k] += e > 1e-2f;
      }
    }
    std::fprintf(stderr, "  %s/%s/%s: off by >1%%: gather %zu / %zu, grouped "
                         "%zu / %zu (worst %.2g)\n",
                 ggml::name(tg), ggml::name(tu), ggml::name(td), bad[0],
                 bad[2], bad[1], bad[3], (double)worst);
    for (int k = 0; k < 4; ++k) {
      EXPECT_TRUE(bad[k] * 1000 < (k < 2 ? act : out).size());
    }
  }
}

// VPIPE_DSV41_IQ_BENCH=1: a prefill's grouped GEMMs from a GGUF's own
// blocks, each thread widening its own 8 values (w8) against the whole
// sub-block (w32, four times over): 64 tiles of 32 rows at
// Qwen3.8-Flash-Next-like widths.
TEST(ggml_native, qwen_grouped_prefill_bench)
{
  if (std::getenv("VPIPE_DSV41_IQ_BENCH") == nullptr) { return; }
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("qwen_moe_native_bf16");
  const std::vector<std::uint64_t> tab = ggml::gpu_iq_tables_native();
  auto btab = mc->make_shared_buffer(tab.size() * 8);
  std::memcpy(btab.contents(), tab.data(), tab.size() * 8);
  const int H = 2048, I = 768, T = 64, S = 16;
  std::mt19937 rng(29);
  for (std::uint32_t t : {12u, 14u, 23u, 18u, 21u, 11u}) {
    const ggml::Geom geo = ggml::geom(t);
    const std::size_t cg = (std::size_t)I * (H / geo.elems) * geo.bytes;
    const std::size_t cd = (std::size_t)H * (I / geo.elems) * geo.bytes;
    auto fill = [&](std::size_t n, int N, int K) {
      auto b = mc->make_shared_buffer(n * S);
      for (int s = 0; s < S; ++s) {
        const auto r = random_ggml_rows_(t, N, K, rng);
        std::memcpy(static_cast<std::uint8_t*>(b.contents()) + s * n,
                    r.data(), n);
      }
      return b;
    };
    auto pg = fill(cg, I, H), pu = fill(cg, I, H), pd = fill(cd, H, I);
    auto bx = mc->make_shared_buffer((std::size_t)T * 32 * H * 2);
    std::memset(bx.contents(), 0, bx.byte_size());
    auto bgact = mc->make_shared_buffer((std::size_t)T * 32 * I * 2);
    auto bpart = mc->make_shared_buffer((std::size_t)T * 32 * H * 2);
    auto bsrow = mc->make_shared_buffer((std::size_t)T * 32 * 4);
    auto bt2e = mc->make_shared_buffer((std::size_t)T * 4);
    for (int i = 0; i < T * 32; ++i) {
      static_cast<std::int32_t*>(bsrow.contents())[i] = i;
    }
    for (int i = 0; i < T; ++i) {
      static_cast<std::int32_t*>(bt2e.contents())[i] = i % S;
    }
    double ms[2] = {0, 0};
    for (int v = 0; v < 2; ++v) {
      const std::string suf = v == 0 ? "" : "_w32";
      auto fs = lib.function("qwen_native_grouped_swiglu" + suf);
      auto fd = lib.function("qwen_native_grouped_down" + suf);
      EXPECT_TRUE(fs.valid() && fd.valid());
      if (!fs.valid() || !fd.valid()) { return; }
      auto run = [&]() {
        auto st = mc->make_command_stream();
        {
          auto enc = st.begin_compute();
          enc.set_function(fs);
          enc.set_buffer(0, pg); enc.set_buffer(1, pu);
          enc.set_buffer(2, btab); enc.set_buffer(3, bx);
          enc.set_buffer(4, bgact); enc.set_buffer(5, bsrow);
          enc.set_buffer(6, bt2e);
          enc.set_constant(7, H); enc.set_constant(8, I);
          enc.set_constant(9, t); enc.set_constant(10, t);
          enc.set_constant(11, (long)cg); enc.set_constant(12, (long)cg);
          enc.dispatch({(unsigned)I, (unsigned)T * 2, 2}, {32, 2, 2});
          enc.set_function(fd);
          enc.set_buffer(0, pd); enc.set_buffer(1, btab);
          enc.set_buffer(2, bgact); enc.set_buffer(3, bpart);
          enc.set_buffer(4, bsrow); enc.set_buffer(5, bt2e);
          enc.set_constant(6, I); enc.set_constant(7, H);
          enc.set_constant(8, t); enc.set_constant(9, (long)cd);
          enc.dispatch({(unsigned)H, (unsigned)T * 2, 2}, {32, 2, 2});
        }
        const auto t0 = std::chrono::steady_clock::now();
        EXPECT_TRUE(st.commit().wait_ok());
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
      };
      run();
      std::vector<double> ts;
      for (int k = 0; k < 25; ++k) { ts.push_back(run()); }
      std::sort(ts.begin(), ts.end());
      ms[v] = ts[ts.size() / 2];   // the median: the clocks wander
    }
    std::fprintf(stderr, "  %-8s %d tiles of 32: w8 %.2f ms, w32 %.2f ms "
                         "(%.2fx)\n", ggml::name(t), T, ms[0], ms[1],
                 ms[1] / ms[0]);
  }
}
