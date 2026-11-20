#include "generative-models/shared/gguf-deepseek41.h"

#include "common/flex-data.h"
#include "generative-models/model-loader.h"
#include "generative-models/shared/ggml-blocks.h"
#include "generative-models/shared/gguf-file.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <format>
#include <sstream>

namespace vpipe::genai {

namespace {

constexpr int kGroup = 32;
constexpr std::uint32_t kGgmlI64 = 27;   // ggml_type GGML_TYPE_I64
const std::string kLm = "language_model.";

inline std::uint16_t
bf16_bits_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) { return (std::uint16_t)(u >> 16); }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

// f(lo, hi) over [0, n) in chunks on the global queue.
template <typename F>
void
for_rows_(std::size_t n, F&& f)
{
  if (n < 2) {
    f((std::size_t)0, n);
    return;
  }
  struct Ctx {
    F* f;
    std::size_t n, chunks;
  } ctx{&f, n, std::min<std::size_t>(n, 64)};
  dispatch_apply_f(ctx.chunks,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   &ctx, [](void* p, std::size_t i) {
                     const Ctx& c = *static_cast<const Ctx*>(p);
                     const std::size_t lo = i * c.n / c.chunks;
                     const std::size_t hi = (i + 1) * c.n / c.chunks;
                     if (lo < hi) { (*c.f)(lo, hi); }
                   });
}

std::int64_t
row_bytes_(std::uint32_t type, std::int64_t k)
{
  if (type == ggml::kF32) { return k * 4; }
  if (type == ggml::kF16 || type == ggml::kBF16) { return k * 2; }
  const ggml::Geom g = ggml::geom(type);
  return g.elems == 0 || k % g.elems != 0 ? 0 : k / g.elems * g.bytes;
}

// ggml dims (ne0 fastest) as a row-major shape.
std::vector<std::int64_t>
shape_of_(const GgufFile::Tensor& t)
{
  std::vector<std::int64_t> s(t.dims.rbegin(), t.dims.rend());
  return s;
}

// `K` values of a row into affine codes of `bits` (packed), group 32,
// bf16 scales and biases.
bool
affine_row_(std::uint32_t type, const std::uint8_t* src, int K, int bits,
            std::uint8_t* w, std::uint16_t* s, std::uint16_t* b)
{
  thread_local std::vector<std::uint8_t> c;
  c.resize((std::size_t)K);
  if (!ggml::row_affine32(type, src, K, bits, /*bf16=*/true, c.data(), s,
                          b)) {
    return false;
  }
  ggml::pack_codes(c.data(), K, bits, w);
  return true;
}

// An [N, K] slab of rows into three parts at `bits`.
bool
affine_rows_(std::uint32_t type, const std::uint8_t* src, int N, int K,
             int bits, std::uint8_t* w, std::uint8_t* s, std::uint8_t* b)
{
  const std::int64_t rb = row_bytes_(type, K);
  if (rb == 0) { return false; }
  const std::size_t wr = (std::size_t)K * bits / 8, gr = (std::size_t)K /
                                                         kGroup;
  for (int r = 0; r < N; ++r) {
    if (!affine_row_(type, src + r * rb, K, bits, w + r * wr,
                     reinterpret_cast<std::uint16_t*>(s) + r * gr,
                     reinterpret_cast<std::uint16_t*>(b) + r * gr)) {
      return false;
    }
  }
  return true;
}

std::string
ints_(const std::vector<std::int64_t>& v)
{
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    s += (i ? ", " : "") + std::to_string(v[i]);
  }
  return s + "]";
}

}  // namespace

// ---- config -------------------------------------------------------------

bool
gguf_to_deepseek41_json(const GgufFile& g, std::string* json,
                        std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "deepseek41 GGUF: " + std::move(m); }
    return false;
  };
  if (g.get_string("general.architecture").value_or("") != "deepseek41") {
    return fail("not a deepseek41 GGUF");
  }
  const std::string a = "deepseek41.";
  auto i64 = [&](const std::string& k, std::int64_t d = -1) {
    return g.get_int(a + k).value_or(d);
  };
  auto f64 = [&](const std::string& k, double d) {
    return g.get_float(a + k).value_or(d);
  };
  for (const char* k : {"block_count", "embedding_length",
                        "attention.head_count", "attention.key_length",
                        "expert_count", "expert_used_count",
                        "expert_feed_forward_length"}) {
    if (!g.has(a + k)) { return fail(std::string("no ") + a + k); }
  }
  const std::vector<float> clamp = g.get_float_array(a + "swiglu_clamp_exp");
  const std::vector<float> clamp_sh =
      g.get_float_array(a + "swiglu_clamp_shexp");
  for (const std::vector<float>* v : {&clamp, &clamp_sh}) {
    for (float x : *v) {
      if (x != clamp.front()) {
        return fail("the SwiGLU clamps differ by layer or expert; the port "
                    "runs one");
      }
    }
  }
  if (g.get_string(a + "rope.scaling.type").value_or("yarn") != "yarn") {
    return fail("rope scaling is not yarn");
  }
  const std::int64_t V = (std::int64_t)g.array_len("tokenizer.ggml.tokens");
  std::string t;
  auto kv = [&](const std::string& k, const std::string& v) {
    t += (t.empty() ? "" : ",\n    ") + ("\"" + k + "\": " + v);
  };
  auto num = [](double v) { return std::format("{:.9g}", v); };
  kv("model_type", "\"deepseek_v41\"");
  kv("vocab_size", std::to_string(V));
  kv("num_key_value_heads", "1");
  kv("max_position_embeddings", std::to_string(i64("context_length", 0)));
  kv("hidden_act", "\"silu\"");
  kv("tie_word_embeddings", "false");
  kv("hidden_size", std::to_string(i64("embedding_length")));
  kv("num_hidden_layers", std::to_string(i64("block_count")));
  kv("num_nextn_predict_layers", "0");
  kv("num_attention_heads", std::to_string(i64("attention.head_count")));
  kv("head_dim", std::to_string(i64("attention.key_length")));
  kv("qk_rope_head_dim", std::to_string(i64("rope.dimension_count", 64)));
  kv("q_lora_rank", std::to_string(i64("attention.q_lora_rank")));
  kv("o_lora_rank", std::to_string(i64("attention.output_lora_rank")));
  kv("o_groups", std::to_string(i64("attention.output_group_count")));
  kv("sliding_window", std::to_string(i64("attention.sliding_window")));
  kv("rms_norm_eps", num(f64("attention.layer_norm_rms_epsilon", 1e-20)));
  kv("moe_intermediate_size",
     std::to_string(i64("expert_feed_forward_length")));
  kv("n_routed_experts", std::to_string(i64("expert_count")));
  kv("num_experts_per_tok", std::to_string(i64("expert_used_count")));
  kv("n_shared_experts", std::to_string(i64("expert_shared_count", 1)));
  kv("routed_scaling_factor", num(f64("expert_weights_scale", 1.0)));
  kv("swiglu_limit", num(clamp.empty() ? 0.0 : clamp.front()));
  kv("norm_topk_prob",
     g.get_bool(a + "expert_weights_norm").value_or(true) ? "true"
                                                           : "false");
  kv("scoring_func", "\"" +
                         g.get_string(a + "expert_scoring_func")
                             .value_or("sqrtsoftplus") +
                         "\"");
  kv("rope_theta", num(f64("rope.freq_base", 10000.0)));
  kv("compress_rope_theta", num(f64("attention.compress_rope_freq_base",
                                    160000.0)));
  if (g.has(a + "rope.scaling.factor")) {
    kv("rope_scaling",
       std::format("{{\"rope_type\": \"yarn\", \"factor\": {}, "
                   "\"beta_fast\": {}, \"beta_slow\": {}, "
                   "\"original_max_position_embeddings\": {}}}",
                   num(f64("rope.scaling.factor", 1.0)),
                   num(f64("rope.scaling.yarn_beta_fast", 32.0)),
                   num(f64("rope.scaling.yarn_beta_slow", 1.0)),
                   i64("rope.scaling.original_context_length", 0)));
  }
  // One entry a layer: the file keeps the release's three DSpark stages'
  // too (its draft head is not in it).
  std::vector<std::int64_t> ratios =
      g.get_int_array(a + "attention.compress_ratios");
  if ((std::int64_t)ratios.size() < i64("block_count")) {
    return fail("compress_ratios is shorter than the layers");
  }
  ratios.resize((std::size_t)i64("block_count"));
  kv("compress_ratios", ints_(ratios));
  kv("kv_source_layer_ids", ints_(g.get_int_array(a + "kv_source_layer_ids")));
  kv("index_source_layer_ids",
     ints_(g.get_int_array(a + "index_source_layer_ids")));
  kv("candidate_source_layer_id",
     std::to_string(i64("attention.candidate_source_layer_id")));
  kv("candidate_topk_blocks",
     std::to_string(i64("attention.candidate_topk_blocks", 0)));
  kv("candidate_block_size",
     std::to_string(i64("attention.candidate_block_size", 0)));
  kv("index_n_heads", std::to_string(i64("attention.indexer.head_count")));
  kv("index_head_dim", std::to_string(i64("attention.indexer.key_length")));
  kv("index_topk", std::to_string(i64("attention.indexer.top_k")));
  kv("hc_mult", std::to_string(i64("hyper_connection.count", 4)));
  kv("hc_sinkhorn_iters",
     std::to_string(i64("hyper_connection.sinkhorn_iterations", 20)));
  kv("hc_eps", num(f64("hyper_connection.epsilon", 1e-6)));
  if (g.get_bool(a + "engram.present").value_or(false)) {
    kv("engram_layer_ids", ints_(g.get_int_array(a + "engram.layer_ids")));
    kv("engram_num_embeddings",
       ints_(g.get_int_array(a + "engram.num_embeddings")));
    kv("engram_max_ngram_size", std::to_string(i64("engram_max_ngram_size")));
    kv("engram_vocab_size", std::to_string(i64("engram_vocab_size")));
    kv("engram_n_heads", std::to_string(i64("engram_n_heads")));
    kv("engram_head_dim", std::to_string(i64("engram_head_dim")));
    kv("engram_pad_token_id", std::to_string(i64("engram_pad_token_id")));
    kv("engram_compressed_vocab_size",
       std::to_string(i64("engram_compressed_vocab_size")));
  }
  *json = std::format(
      "{{\n  \"architectures\": [\"DeepseekV41ForCausalLM\"],\n"
      "  \"model_type\": \"deepseek_v41\",\n"
      "  \"bos_token_id\": {},\n  \"eos_token_id\": {},\n"
      "  \"omlx_deepseek_v41\": {{\"quantized_modules\": {{"
      "\"language_model.layers.0.attn.wq_a\": {{\"mode\": \"affine\", "
      "\"bits\": 8, \"group_size\": 32}}}}}},\n"
      "  \"text_config\": {{\n    {}\n  }}\n}}\n",
      g.get_int("tokenizer.ggml.bos_token_id").value_or(0),
      g.get_int("tokenizer.ggml.eos_token_id").value_or(1), t);
  return true;
}

bool
gguf_to_deepseek41_config(const GgufFile& g, ModelConfig* out)
{
  std::string js, err;
  if (!gguf_to_deepseek41_json(g, &js, &err)) { return false; }
  std::istringstream in(js);
  FlexData cfg;
  try {
    cfg = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  auto parsed = ModelLoader::parse_config(cfg);
  if (!parsed) { return false; }
  *out = std::move(*parsed);
  return out->n_layers > 0;
}

std::string
gguf_deepseek41_engram_path(const GgufFile& g, const std::string& first)
{
  namespace fs = std::filesystem;
  const auto f = g.get_string("deepseek41.engram.file");
  if (!f || f->empty() || fs::path(*f).has_parent_path()) { return ""; }
  const fs::path dir = fs::path(first).parent_path();
  std::error_code ec;
  if (fs::is_regular_file(dir / *f, ec)) { return (dir / *f).string(); }
  // Fetched as a record of its own (its publisher's repo), model-fetch
  // puts it at <models>/<org>/<repo>/ beside the shards' <org>/<repo>.
  const fs::path root = dir.parent_path().parent_path();
  for (const auto& org : fs::directory_iterator(root, ec)) {
    if (ec) { break; }
    if (!org.is_directory()) { continue; }
    std::error_code ec2;
    for (const auto& repo : fs::directory_iterator(org.path(), ec2)) {
      if (ec2) { break; }
      if (fs::is_regular_file(repo.path() / *f, ec2)) {
        return (repo.path() / *f).string();
      }
    }
  }
  return (dir / *f).string();
}

// ---- experts ------------------------------------------------------------

bool
deepseek41_rows_to_affine(std::uint32_t type, const std::uint8_t* src,
                          int rows, int K, int bits, std::uint8_t* w,
                          std::uint8_t* s, std::uint8_t* b)
{
  return affine_rows_(type, src, rows, K, bits, w, s, b);
}

int
deepseek41_expert_bits(std::uint32_t type)
{
  return ggml::affine_width(type);
}

bool
deepseek41_expert_to_affine(std::uint32_t w1_type, const std::uint8_t* w1,
                            std::uint32_t w3_type, const std::uint8_t* w3,
                            std::uint32_t w2_type, const std::uint8_t* w2,
                            int I, int H, int b13, int b2,
                            std::uint8_t* const* dst)
{
  return affine_rows_(w1_type, w1, I, H, b13, dst[0], dst[1], dst[2]) &&
         affine_rows_(w3_type, w3, I, H, b13, dst[3], dst[4], dst[5]) &&
         affine_rows_(w2_type, w2, H, I, b2, dst[6], dst[7], dst[8]);
}

// ---- the converter ------------------------------------------------------

GgufDeepseek41Converter::GgufDeepseek41Converter(const GgufFile* gguf)
    : _g(gguf)
{
  build_();
}

void
GgufDeepseek41Converter::add_affine_(const std::string& hf,
                                     const std::string& gguf)
{
  const GgufFile::Tensor* t = _g->tensor(gguf);
  if (t == nullptr || t->dims.empty() || t->dims[0] % kGroup != 0) {
    return;
  }
  const int bits = ggml::affine_width(t->type);
  if (bits == 0) { return; }
  const std::int64_t K = t->dims[0], N = t->numel() / K;
  for (int part = 0; part < 3; ++part) {
    Src s;
    s.kind = Kind::kAffine;
    s.gguf = gguf;
    s.part = part;
    s.bits = bits;
    ConvertedTensorSpec sp;
    sp.gguf_name = gguf;
    if (part == 0) {
      sp.hf_name = hf + ".weight";
      sp.dtype = "U32";
      sp.shape = {N, K * bits / 32};
      sp.nbytes = (std::uint64_t)(N * K * bits / 8);
    } else {
      sp.hf_name = hf + (part == 1 ? ".scales" : ".biases");
      sp.dtype = "BF16";
      sp.shape = {N, K / kGroup};
      sp.nbytes = (std::uint64_t)(N * (K / kGroup) * 2);
    }
    _src.emplace(sp.hf_name, std::move(s));
    _specs.push_back(std::move(sp));
  }
}

void
GgufDeepseek41Converter::add_float_(const std::string& hf,
                                    const std::string& gguf, Kind k)
{
  const GgufFile::Tensor* t = _g->tensor(gguf);
  if (t == nullptr || t->dims.empty()) { return; }
  Src s;
  s.kind = k;
  s.gguf = gguf;
  ConvertedTensorSpec sp;
  sp.hf_name = hf;
  sp.gguf_name = gguf;
  sp.shape = shape_of_(*t);
  const auto n = (std::uint64_t)t->numel();
  if (k == Kind::kRaw) {
    const bool i64 = t->type == kGgmlI64;
    sp.dtype = i64 ? "I64" : "I32";
    sp.nbytes = n * (i64 ? 8 : 4);
  } else {
    sp.dtype = k == Kind::kF32 ? "F32" : "BF16";
    sp.nbytes = n * (k == Kind::kF32 ? 4 : 2);
  }
  _src.emplace(hf, std::move(s));
  _specs.push_back(std::move(sp));
}

void
GgufDeepseek41Converter::add_expert_(const std::string& hf,
                                     const std::string& gguf, int bits)
{
  const GgufFile::Tensor* t = _g->tensor(gguf);
  if (t == nullptr || t->dims.size() != 3 || t->dims[0] % kGroup != 0) {
    return;
  }
  const std::int64_t K = t->dims[0], N = t->dims[1], E = t->dims[2];
  for (int part = 0; part < 3; ++part) {
    Src s;
    s.kind = Kind::kExpert;
    s.gguf = gguf;
    s.part = part;
    s.bits = bits;
    ConvertedTensorSpec sp;
    sp.gguf_name = gguf;
    if (part == 0) {
      sp.hf_name = hf + ".weight";
      sp.dtype = "U32";
      sp.shape = {E, N, K * bits / 32};
      sp.nbytes = (std::uint64_t)(E * N * K * bits / 8);
    } else {
      sp.hf_name = hf + (part == 1 ? ".scales" : ".biases");
      sp.dtype = "BF16";
      sp.shape = {E, N, K / kGroup};
      sp.nbytes = (std::uint64_t)(E * N * (K / kGroup) * 2);
    }
    _src.emplace(sp.hf_name, std::move(s));
    _specs.push_back(std::move(sp));
  }
}

void
GgufDeepseek41Converter::build_()
{
  const int L = (int)_g->get_int("deepseek41.block_count").value_or(0);
  add_float_(kLm + "embed.weight", "token_embd.weight", Kind::kBf16);
  add_float_(kLm + "head.weight", "output.weight", Kind::kBf16);
  add_float_(kLm + "norm.weight", "output_norm.weight", Kind::kBf16);
  for (int l = 0; l < L; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    const std::string p = kLm + "layers." + std::to_string(l) + ".";
    add_float_(p + "attn_norm.weight", b + "attn_norm.weight", Kind::kBf16);
    add_float_(p + "ffn_norm.weight", b + "ffn_norm.weight", Kind::kBf16);
    add_affine_(p + "attn.wq_a", b + "attn_q_a.weight");
    add_float_(p + "attn.q_norm.weight", b + "attn_q_a_norm.weight",
               Kind::kBf16);
    add_affine_(p + "attn.wq_b", b + "attn_q_b.weight");
    add_affine_(p + "attn.wkv", b + "attn_kv.weight");
    add_float_(p + "attn.kv_norm.weight", b + "attn_kv_a_norm.weight",
               Kind::kBf16);
    add_affine_(p + "attn.wo_a", b + "attn_output_a.weight");
    add_affine_(p + "attn.wo_b", b + "attn_output_b.weight");
    add_float_(p + "attn.attn_sink", b + "attn_sinks.weight", Kind::kF32);
    for (const char* h : {"hc_attn_", "hc_ffn_"}) {
      for (const char* x : {"fn", "base", "scale"}) {
        add_float_(p + h + x, b + h + x + ".weight", Kind::kF32);
      }
    }
    add_float_(p + "ffn.gate.weight", b + "ffn_gate_inp.weight",
               Kind::kBf16);
    add_float_(p + "ffn.gate.bias", b + "exp_probs_b.bias", Kind::kF32);
    add_affine_(p + "ffn.shared_experts.w1", b + "ffn_gate_shexp.weight");
    add_affine_(p + "ffn.shared_experts.w3", b + "ffn_up_shexp.weight");
    add_affine_(p + "ffn.shared_experts.w2", b + "ffn_down_shexp.weight");
    add_float_(p + "attn.compressor.wkv.weight",
               b + "attn_compressor_kv.weight", Kind::kBf16);
    add_float_(p + "attn.compressor.wgate.weight",
               b + "attn_compressor_gate.weight", Kind::kBf16);
    add_float_(p + "attn.compressor.norm.weight",
               b + "attn_compressor_norm.weight", Kind::kBf16);
    add_float_(p + "attn.indexer.wq_b.weight", b + "indexer.attn_q_b.weight",
               Kind::kBf16);
    add_float_(p + "attn.indexer.weights_proj.weight",
               b + "indexer.proj.weight", Kind::kBf16);
    add_float_(p + "attn.indexer.wk.weight", b + "indexer.attn_k.weight",
               Kind::kBf16);
    add_float_(p + "attn.indexer.k_norm.weight", b + "indexer.k_norm.weight",
               Kind::kBf16);
    add_float_(p + "engram.q_weight", b + "engram_q.weight", Kind::kBf16);
    add_float_(p + "engram.k_weight", b + "engram_k.weight", Kind::kBf16);
    add_affine_(p + "engram.wkv", b + "engram_wkv.weight");
    // A layer's experts: w1 and w3 at one width (the wider), w2 its own.
    const auto* g1 = _g->tensor(b + "ffn_gate_exps.weight");
    const auto* g3 = _g->tensor(b + "ffn_up_exps.weight");
    const auto* g2 = _g->tensor(b + "ffn_down_exps.weight");
    if (g1 != nullptr && g3 != nullptr && g2 != nullptr) {
      const int b13 = std::max(deepseek41_expert_bits(g1->type),
                               deepseek41_expert_bits(g3->type));
      const int b2 = deepseek41_expert_bits(g2->type);
      if (b13 > 0 && b2 > 0) {
        add_expert_(p + "ffn.experts.w1", b + "ffn_gate_exps.weight", b13);
        add_expert_(p + "ffn.experts.w3", b + "ffn_up_exps.weight", b13);
        add_expert_(p + "ffn.experts.w2", b + "ffn_down_exps.weight", b2);
      }
    }
  }
  add_float_("gguf.engram_token_map", "engram_token_map.weight", Kind::kRaw);
  add_float_("gguf.engram_multipliers", "engram_multipliers.weight",
             Kind::kRaw);
  add_float_("gguf.engram_primes", "engram_primes.weight", Kind::kRaw);
  add_float_("gguf.engram_offsets", "engram_offsets.weight", Kind::kRaw);
}

bool
GgufDeepseek41Converter::build_affine_(const Src& s, Built& o,
                                       std::uint8_t* codes) const
{
  const GgufFile::Tensor* t = _g->tensor(s.gguf);
  if (t == nullptr || t->data == nullptr) { return false; }
  const std::int64_t K = t->dims[0], N = t->numel() / K;
  const std::int64_t rb = row_bytes_(t->type, K);
  if (rb == 0) { return false; }
  const std::size_t wr = (std::size_t)K * s.bits / 8, G = K / kGroup;
  // The codes straight into the caller's buffer (a host vector of them
  // would stay in the footprint once freed: macOS keeps a freed large
  // block dirty), or kept here where a scale part is asked for first.
  if (codes == nullptr) {
    o.codes.resize((std::size_t)N * wr);
    codes = o.codes.data();
  }
  o.scales.resize((std::size_t)N * G);
  o.biases.resize(o.scales.size());
  std::atomic<bool> ok{true};
  for_rows_((std::size_t)N, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t r = lo; r < hi && ok; ++r) {
      if (!affine_row_(t->type, t->data + r * rb, (int)K, s.bits,
                       codes + r * wr, o.scales.data() + r * G,
                       o.biases.data() + r * G)) {
        ok = false;
      }
    }
  });
  return ok;
}

bool
GgufDeepseek41Converter::convert(const ConvertedTensorSpec& spec,
                                 std::uint8_t* dst)
{
  const auto it = _src.find(spec.hf_name);
  if (it == _src.end()) { return false; }
  const Src& s = it->second;
  const GgufFile::Tensor* t = _g->tensor(s.gguf);
  if (t == nullptr || t->data == nullptr) { return false; }
  switch (s.kind) {
    case Kind::kRaw:
      std::memcpy(dst, t->data, (std::size_t)spec.nbytes);
      return true;
    case Kind::kBf16:
    case Kind::kF32: {
      const std::int64_t K = t->dims[0], N = t->numel() / K;
      const std::int64_t rb = row_bytes_(t->type, K);
      if (rb == 0) { return false; }
      const bool f32 = s.kind == Kind::kF32;
      std::atomic<bool> ok{true};
      for_rows_((std::size_t)N, [&](std::size_t lo, std::size_t hi) {
        std::vector<float> x((std::size_t)K);
        for (std::size_t r = lo; r < hi && ok; ++r) {
          float* out = f32 ? reinterpret_cast<float*>(dst) + r * K
                           : x.data();
          if (!ggml::decode(t->type, t->data + r * rb, K, out)) {
            ok = false;
            break;
          }
          if (!f32) {
            auto* d = reinterpret_cast<std::uint16_t*>(dst) + r * K;
            for (std::int64_t j = 0; j < K; ++j) {
              d[j] = bf16_bits_(x[(std::size_t)j]);
            }
          }
        }
      });
      return ok;
    }
    case Kind::kAffine: {
      // The three parts are one build: kept until each is served.
      const std::string base =
          spec.hf_name.substr(0, spec.hf_name.rfind('.'));
      Built& b = _built[base];
      if (s.part == 0 && b.codes.empty()) {
        // The codes straight into `dst`; the groups kept for the others.
        Built fresh;
        if (!build_affine_(s, fresh, dst)) {
          _built.erase(base);
          return false;
        }
        if (b.scales.empty()) {
          b.scales = std::move(fresh.scales);
          b.biases = std::move(fresh.biases);
        }
      } else {
        if (b.scales.empty() && !build_affine_(s, b, nullptr)) {
          _built.erase(base);
          return false;
        }
        const void* p = s.part == 0 ? (const void*)b.codes.data()
                      : s.part == 1 ? (const void*)b.scales.data()
                                    : (const void*)b.biases.data();
        std::memcpy(dst, p, (std::size_t)spec.nbytes);
      }
      if (++b.served == 3) { _built.erase(base); }
      return true;
    }
    case Kind::kExpert: {
      // A whole stack (tests; the model's cache reads an expert at a time).
      const std::int64_t K = t->dims[0], N = t->dims[1], E = t->dims[2];
      const std::int64_t rb = row_bytes_(t->type, K);
      if (rb == 0) { return false; }
      const std::size_t wr = (std::size_t)K * s.bits / 8, G = K / kGroup;
      std::atomic<bool> ok{true};
      for_rows_((std::size_t)E, [&](std::size_t lo, std::size_t hi) {
        std::vector<std::uint8_t> w((std::size_t)N * wr);
        std::vector<std::uint8_t> sc((std::size_t)N * G * 2),
            bi((std::size_t)N * G * 2);
        for (std::size_t e = lo; e < hi && ok; ++e) {
          if (!affine_rows_(t->type, t->data + e * N * rb, (int)N, (int)K,
                            s.bits, w.data(), sc.data(), bi.data())) {
            ok = false;
            break;
          }
          const auto& part = s.part == 0 ? w : s.part == 1 ? sc : bi;
          std::memcpy(dst + e * part.size(), part.data(), part.size());
        }
      });
      return ok;
    }
  }
  return false;
}

}  // namespace vpipe::genai
