#include "generative-models/shared/gguf-qwen4exp.h"

#include "generative-models/model-loader.h"
#include "generative-models/shared/ggml-blocks.h"
#include "generative-models/shared/gguf-file.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <sstream>

namespace vpipe::genai {

namespace {

// The group every non-expert projection is held at: 32, the block of
// the legacy quants and the sub-block of the k-quants and IQ4_XS, where a
// block format's values take 8-bit affine codes EXACTLY (ggml-blocks.h).
constexpr int kW8Group = 32;
constexpr int kXGroup = 32;
const std::string kRoot = "language_model.model.";

// Round-to-nearest-even bf16 / f16 bits of an f32.
inline std::uint16_t
bf16_bits_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) { return (std::uint16_t)(u >> 16); }
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}
inline std::uint16_t
f16_bits_(float f)
{
  const _Float16 h = (_Float16)f;
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
inline std::uint16_t
elt_bits_(float f, bool bf16)
{
  return bf16 ? bf16_bits_(f) : f16_bits_(f);
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
  const ggml::Geom g = ggml::geom(type);
  return g.elems == 0 ? 0 : k / g.elems * g.bytes;
}

// One row of a routed expert into its affine codes (packed) and groups.
bool
expert_row_(std::uint32_t type, const std::uint8_t* src, int K, int bits,
            bool bf16, std::uint8_t* w, std::uint16_t* s, std::uint16_t* b)
{
  thread_local std::vector<std::uint8_t> codes;
  codes.resize((std::size_t)K);
  if (!ggml::row_affine32(type, src, K, bits, bf16, codes.data(), s, b)) {
    return false;
  }
  ggml::pack_codes(codes.data(), K, bits, w);
  return true;
}

}  // namespace

// ---- config ------------------------------------------------------------

bool
gguf_to_qwen4exp_config(const GgufFile& g, ModelConfig* out)
{
  const std::string a = "qwen4exp.";
  auto I = [&](const std::string& k, std::int64_t d) {
    const auto v = g.get_int(a + k);
    return v ? *v : d;
  };
  auto F = [&](const std::string& k, double d) {
    const auto v = g.get_float(a + k);
    return v ? *v : d;
  };
  const std::int64_t L = I("block_count", 0), H = I("embedding_length", 0);
  const std::int64_t D = I("attention.key_length", 256);
  const std::int64_t nv = I("ssm.time_step_rank", 0);
  const std::int64_t interval = I("full_attention_interval", 4);
  if (L <= 0 || H <= 0 || nv <= 0 || interval <= 0) { return false; }
  std::int64_t vocab = (std::int64_t)g.array_len("tokenizer.ggml.tokens");
  if (const auto* t = g.tensor("token_embd.weight");
      t != nullptr && t->dims.size() == 2) {
    vocab = t->dims[1];
  }
  std::ostringstream lt, sec, ple;
  for (std::int64_t i = 0; i < L; ++i) {
    lt << (i ? ", " : "")
       << ((i + 1) % interval == 0 ? "\"full_attention\""
                                   : "\"linear_attention\"");
  }
  {
    const auto s = g.get_int_array(a + "rope.dimension_sections");
    bool first = true;
    for (auto v : s) {
      if (v <= 0) { continue; }
      sec << (first ? "" : ", ") << v;
      first = false;
    }
  }
  {
    // One-indexed in the release's config.
    const auto p = g.get_int_array(a + "ple.layers");
    for (std::size_t i = 0; i < p.size(); ++i) {
      ple << (i ? ", " : "") << p[i] + 1;
    }
  }
  int ratio = 0;
  for (auto r : g.get_int_array(a + "attention.compress_ratios")) {
    if (r > 0) { ratio = (int)r; }
  }
  const std::int64_t rot = I("rope.dimension_count", D / 4);
  const std::int64_t eos = I("ple.eos_token_id", 248044);
  const auto tok_eos = g.get_int("tokenizer.ggml.eos_token_id");
  std::ostringstream js;
  js.precision(17);
  js << "{\"architectures\": [\"Qwen4ExpForConditionalGeneration\"], "
     << "\"model_type\": \"qwen4_exp\", \"tie_word_embeddings\": false, "
     << "\"image_token_id\": " << I("ple.image_token_id", 248056) << ", "
     << "\"eos_token_id\": [" << (tok_eos ? *tok_eos : 248046) << ", " << eos
     << "], "
     // Every non-expert projection is presented as affine w8 at group 32.
     << "\"quantization\": {\"group_size\": 32, \"bits\": 8, "
     << "\"mode\": \"affine\"}, "
     << "\"text_config\": {\"model_type\": \"qwen4_exp_text\", "
     << "\"hidden_size\": " << H << ", \"num_hidden_layers\": " << L << ", "
     << "\"num_attention_heads\": " << I("attention.head_count", 0) << ", "
     << "\"num_key_value_heads\": " << I("attention.head_count_kv", 0)
     << ", \"head_dim\": " << D << ", "
     << "\"rms_norm_eps\": " << F("attention.layer_norm_rms_epsilon", 1e-6)
     << ", \"max_position_embeddings\": " << I("context_length", 262144)
     << ", \"vocab_size\": " << vocab << ", "
     << "\"partial_rotary_factor\": " << (double)rot / (double)D << ", "
     << "\"rope_parameters\": {\"rope_theta\": " << F("rope.freq_base", 1e7)
     << ", \"partial_rotary_factor\": " << (double)rot / (double)D
     << ", \"mrope_interleaved\": true, \"mrope_section\": [" << sec.str()
     << "], \"type\": \"default\"}, "
     << "\"full_attention_interval\": " << interval << ", "
     << "\"layer_types\": [" << lt.str() << "], "
     << "\"linear_conv_kernel_dim\": " << I("ssm.conv_kernel", 4) << ", "
     << "\"linear_key_head_dim\": " << I("ssm.state_size", 128) << ", "
     << "\"linear_num_key_heads\": " << I("ssm.group_count", 16) << ", "
     << "\"linear_num_value_heads\": " << nv << ", "
     << "\"linear_value_head_dim\": " << I("ssm.inner_size", 0) / nv << ", "
     << "\"num_experts\": " << I("expert_count", 0) << ", "
     << "\"num_experts_per_tok\": " << I("expert_used_count", 0) << ", "
     << "\"moe_intermediate_size\": " << I("expert_feed_forward_length", 0)
     << ", \"shared_expert_intermediate_size\": "
     << I("expert_shared_feed_forward_length", 0) << ", "
     << "\"hc_count\": " << I("hyper_connection.count", 4) << ", "
     << "\"hc_lowrank\": " << I("hyper_connection.low_rank", 0) << ", "
     << "\"indexer_n_heads\": " << I("attention.indexer.head_count", 0)
     << ", \"indexer_head_dim\": " << I("attention.indexer.key_length", 0)
     << ", \"indexer_budget\": " << I("attention.indexer.top_k", 0)
     << ", \"indexer_compress_ratio\": " << ratio
     << ", \"indexer_kv_heads\": 1, "
     << "\"ple_layer_ids\": [" << ple.str() << "], "
     << "\"ngram_size\": " << I("ple.ngram_size", 3) << ", "
     << "\"heads_per_ngram\": " << I("ple.heads_per_ngram", 8) << ", "
     << "\"ple_conv_kernel_size\": " << I("ple.conv_kernel", 4) << ", "
     << "\"ple_embed_dim\": " << H << ", "
     << "\"eos_token_id\": " << eos << ", \"bos_token_id\": " << eos << ", "
     << "\"output_gate_type\": \"sigmoid\", \"hidden_act\": \"silu\", "
     << "\"mtp_num_hidden_layers\": 0, \"tie_word_embeddings\": false}}";
  std::istringstream in(js.str());
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

// ---- the converter -------------------------------------------------------

GgufQwen4ExpConverter::GgufQwen4ExpConverter(const GgufFile* gguf) : _g(gguf)
{
  _nk = (int)_g->get_int("qwen4exp.ssm.group_count").value_or(16);
  _nv = (int)_g->get_int("qwen4exp.ssm.time_step_rank").value_or(48);
  build_();
}

int
GgufQwen4ExpConverter::v_src_(int h) const
{
  const int r = _nv / _nk;
  return (h % r) * _nk + h / r;
}

void
GgufQwen4ExpConverter::add_w8_(const std::string& hf,
                               std::vector<std::string> gguf, int vrows,
                               int vunit, bool vcols)
{
  std::int64_t N = 0, K = -1;
  for (const auto& n : gguf) {
    const GgufFile::Tensor* t = _g->tensor(n);
    if (t == nullptr || t->dims.empty() || t->dims.size() > 2) { return; }
    const std::int64_t k = t->dims[0];
    if (K >= 0 && k != K) { return; }
    K = k;
    N += t->dims.size() == 2 ? t->dims[1] : 1;
  }
  if (K <= 0 || K % kW8Group != 0) { return; }
  Src s;
  s.kind = Kind::kW8;
  s.gguf = std::move(gguf);
  if (vunit > 0) {
    s.vunit = vunit;
    if (vcols) {
      s.vcols = 1;
    } else {
      s.vrows = vrows < 0 ? 0 : vrows;
    }
  }
  const auto u = [](std::int64_t v) { return (std::uint64_t)v; };
  for (int part = 0; part < 3; ++part) {
    Src p = s;
    p.part = part;
    ConvertedTensorSpec sp;
    sp.gguf_name = s.gguf.front();
    if (part == 0) {
      sp.hf_name = hf + ".weight";
      sp.dtype = "U32";
      sp.shape = {N, K / 4};
      sp.nbytes = u(N) * u(K);
    } else {
      sp.hf_name = hf + (part == 1 ? ".scales" : ".biases");
      sp.dtype = "BF16";
      sp.shape = {N, K / kW8Group};
      sp.nbytes = u(N) * u(K / kW8Group) * 2;
    }
    _src.emplace(sp.hf_name, std::move(p));
    _specs.push_back(std::move(sp));
  }
}

void
GgufQwen4ExpConverter::add_f32_(const std::string& hf,
                                const std::string& gguf, bool minus_one,
                                int vunit, int vrow0, bool a_log)
{
  const GgufFile::Tensor* t = _g->tensor(gguf);
  if (t == nullptr) { return; }
  Src s;
  s.kind = Kind::kF32;
  s.gguf = {gguf};
  s.minus_one = minus_one;
  s.a_log = a_log;
  s.vunit = vunit;
  s.vrows = vrow0;
  ConvertedTensorSpec sp;
  sp.hf_name = hf;
  sp.gguf_name = gguf;
  sp.dtype = "F32";
  // HF's order: the outer axis first (a conv's [C, K, 1] taps per
  // channel; a 1-D vector as itself).
  sp.shape.assign(t->dims.rbegin(), t->dims.rend());
  if (hf.size() > 14 && hf.compare(hf.size() - 14, 14, "conv1d.weight") ==
                            0 &&
      t->dims.size() == 2) {
    sp.shape = {t->dims[1], t->dims[0], 1};
  }
  sp.nbytes = (std::uint64_t)t->numel() * 4;
  _src.emplace(hf, std::move(s));
  _specs.push_back(std::move(sp));
}

void
GgufQwen4ExpConverter::add_i64_(const std::string& hf, const std::string& key)
{
  const std::size_t n = _g->array_len(key);
  if (n == 0) { return; }
  Src s;
  s.kind = Kind::kI64;
  s.key = key;
  ConvertedTensorSpec sp;
  sp.hf_name = hf;
  sp.dtype = "I64";
  sp.shape = {(std::int64_t)n};
  sp.nbytes = n * 8;
  _src.emplace(hf, std::move(s));
  _specs.push_back(std::move(sp));
}

void
GgufQwen4ExpConverter::add_expert_(const std::string& hf,
                                   const std::string& gguf, int bits)
{
  const GgufFile::Tensor* t = _g->tensor(gguf);
  if (t == nullptr || t->dims.size() != 3 || t->dims[0] % kXGroup != 0) {
    return;
  }
  const std::int64_t K = t->dims[0], N = t->dims[1], E = t->dims[2];
  for (int part = 0; part < 3; ++part) {
    Src s;
    s.kind = Kind::kExpert;
    s.gguf = {gguf};
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
      sp.shape = {E, N, K / kXGroup};
      sp.nbytes = (std::uint64_t)(E * N * (K / kXGroup) * 2);
    }
    _src.emplace(sp.hf_name, std::move(s));
    _specs.push_back(std::move(sp));
  }
}

void
GgufQwen4ExpConverter::build_()
{
  const int L = (int)_g->get_int("qwen4exp.block_count").value_or(0);
  const int interval =
      (int)_g->get_int("qwen4exp.full_attention_interval").value_or(4);
  const int vd = (int)(_g->get_int("qwen4exp.ssm.inner_size").value_or(0) /
                       std::max(_nv, 1));
  const int kd = (int)_g->get_int("qwen4exp.ssm.state_size").value_or(128);
  const int qk_rows = 2 * _nk * kd;    // q | k ahead of v in the conv
  // The experts' widths: the widest exact form any layer's format needs.
  for (int l = 0; l < L; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    for (const char* r : {"ffn_gate_exps.weight", "ffn_up_exps.weight"}) {
      if (const auto* t = _g->tensor(b + r)) {
        _gb = std::max(_gb, ggml::affine_bits(t->type));
      }
    }
    if (const auto* t = _g->tensor(b + "ffn_down_exps.weight")) {
      _db = std::max(_db, ggml::affine_bits(t->type));
    }
  }
  // An expert width the kernels take: 4, 5, 6 or 8.
  auto kernel_bits = [](int b) { return b == 7 ? 8 : b; };
  _gb = kernel_bits(_gb);
  _db = kernel_bits(_db);
  auto hc = [&](const std::string& hf, const std::string& g, bool inject) {
    add_f32_(hf + "hc_norm.weight", g + "_norm.weight", true);
    add_w8_(hf + "input_mix_weight_down", {g + "_down.weight"});
    add_w8_(hf + "input_mix_weight_up", {g + "_up.weight"});
    if (inject) {
      add_w8_(hf + "block_inject_weight", {g + "_inject.weight"});
    }
  };
  for (int l = 0; l < L; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    const std::string p = kRoot + "layers." + std::to_string(l) + ".";
    hc(p + "attn_hyper_connection.", b + "hc_attn", true);
    hc(p + "mlp_hyper_connection.", b + "hc_ffn", true);
    if ((l + 1) % interval == 0) {
      const std::string a = p + "self_attn.";
      add_w8_(a + "q_proj", {b + "attn_q.weight"});
      add_w8_(a + "k_proj", {b + "attn_k.weight"});
      add_w8_(a + "v_proj", {b + "attn_v.weight"});
      add_w8_(a + "o_proj", {b + "attn_output.weight"});
      add_f32_(a + "q_norm.weight", b + "attn_q_norm.weight", true);
      add_f32_(a + "k_norm.weight", b + "attn_k_norm.weight", true);
      add_w8_(a + "indexer.index_qk_proj",
              {b + "indexer.q_proj.weight", b + "indexer.k_proj.weight"});
      add_f32_(a + "indexer.q_layernorm.weight", b + "indexer.q_norm.weight",
               true);
      add_f32_(a + "indexer.k_layernorm.weight", b + "indexer.k_norm.weight",
               true);
    } else {
      const std::string a = p + "linear_attn.";
      add_w8_(a + "in_proj_qkv", {b + "attn_qkv.weight"}, qk_rows, vd);
      add_w8_(a + "in_proj_z", {b + "attn_gate.weight"}, 0, vd);
      add_w8_(a + "in_proj_a", {b + "ssm_alpha.weight"}, 0, 1);
      add_w8_(a + "in_proj_b", {b + "ssm_beta.weight"}, 0, 1);
      add_w8_(a + "out_proj", {b + "ssm_out.weight"}, -1, vd, true);
      add_f32_(a + "conv1d.weight", b + "ssm_conv1d.weight", false, vd,
               qk_rows);
      add_f32_(a + "A_log", b + "ssm_a", false, 1, 0, true);
      add_f32_(a + "dt_bias", b + "ssm_dt.bias", false, 1, 0);
      add_f32_(a + "norm.weight", b + "ssm_norm.weight");
    }
    const std::string m = p + "mlp.";
    add_f32_(m + "gate.weight", b + "ffn_gate_inp.weight");
    add_w8_(m + "shared_expert.gate_proj", {b + "ffn_gate_shexp.weight"});
    add_w8_(m + "shared_expert.up_proj", {b + "ffn_up_shexp.weight"});
    add_w8_(m + "shared_expert.down_proj", {b + "ffn_down_shexp.weight"});
    add_w8_(m + "shared_expert_gate", {b + "ffn_gate_inp_shexp.weight"});
    add_expert_(m + "switch_mlp.gate_proj", b + "ffn_gate_exps.weight", _gb);
    add_expert_(m + "switch_mlp.up_proj", b + "ffn_up_exps.weight", _gb);
    add_expert_(m + "switch_mlp.down_proj", b + "ffn_down_exps.weight", _db);
    if (_g->tensor(b + "ple_key.weight") != nullptr) {
      const std::string q = p + "ple.";
      add_w8_(q + "key_proj", {b + "ple_key.weight"});
      add_w8_(q + "value_proj", {b + "ple_value.weight"});
      add_f32_(q + "conv1d.weight", b + "ple_conv1d.weight");
      add_f32_(q + "norm_conv.weight", b + "ple_norm_conv.weight", true);
      add_f32_(q + "norm_key.weight", b + "ple_norm_key.weight", true);
      add_f32_(q + "norm_query.weight", b + "ple_norm_query.weight", true);
      const std::string e = q + "ple_embedding.";
      add_i64_(e + "layer_multipliers", "qwen4exp.ple.layer_multipliers");
      add_i64_(e + "ngram_heads_vocab_sizes", "qwen4exp.ple.head_vocab_sizes");
      add_i64_(e + "ngram_heads_offsets", "qwen4exp.ple.head_offsets");
    }
  }
  add_w8_(kRoot + "embed_tokens", {"token_embd.weight"});
  add_w8_("language_model.lm_head", {"output.weight"});
  hc(kRoot + "hyper_connection_mixer.", "output_hc", false);
}

bool
GgufQwen4ExpConverter::build_w8_(const Src& s, W8& o,
                                 std::uint8_t* codes) const
{
  // Output row r is source row src_r of the concatenated sources (a
  // v-head row permutation applied); a column permutation moves whole
  // v-heads (a multiple of the group) of codes and groups alike.
  std::vector<const GgufFile::Tensor*> ts;
  std::vector<std::int64_t> row0;
  std::int64_t N = 0, K = 0;
  for (const auto& n : s.gguf) {
    const GgufFile::Tensor* t = _g->tensor(n);
    if (t == nullptr || t->data == nullptr) { return false; }
    ts.push_back(t);
    row0.push_back(N);
    K = t->dims[0];
    N += t->dims.size() == 2 ? t->dims[1] : 1;
  }
  if (s.vcols != 0 && (s.vunit % kW8Group != 0 || K != (std::int64_t)_nv *
                                                         s.vunit)) {
    return false;
  }
  const std::int64_t G = K / kW8Group;
  // The codes straight into the caller's buffer (a host vector the size
  // of the embedding would stay in the footprint once freed), or a
  // scratch when the scales are asked for first.
  std::vector<std::uint8_t> own;
  if (codes == nullptr) {
    own.resize((std::size_t)(N * K));
    codes = own.data();
  }
  o.scales.resize((std::size_t)(N * G));
  o.biases.resize(o.scales.size());
  std::atomic<bool> ok{true};
  for_rows_((std::size_t)N, [&](std::size_t lo, std::size_t hi) {
    std::vector<std::uint8_t> c((std::size_t)K);
    std::vector<std::uint16_t> sc((std::size_t)G), bi((std::size_t)G);
    for (std::size_t r = lo; r < hi && ok; ++r) {
      std::int64_t sr = (std::int64_t)r;
      if (s.vunit > 0 && s.vcols == 0 && sr >= s.vrows &&
          sr < s.vrows + (std::int64_t)_nv * s.vunit) {
        const std::int64_t h = (sr - s.vrows) / s.vunit;
        sr = s.vrows + (std::int64_t)v_src_((int)h) * s.vunit +
             (sr - s.vrows) % s.vunit;
      }
      std::size_t ti = ts.size() - 1;
      while (ti > 0 && row0[ti] > sr) { --ti; }
      const GgufFile::Tensor* t = ts[ti];
      const std::int64_t rb = row_bytes_(t->type, K);
      const std::uint8_t* src = t->data + (sr - row0[ti]) * rb;
      if (rb == 0) {
        ok = false;
        break;
      }
      // Exact where the block format's own grid takes 8 bits (its scale,
      // and a symmetric format's bias from it, in bf16), else rounded.
      if (!ggml::row_affine32(t->type, src, (int)K, 8, /*bf16=*/true,
                              c.data(), sc.data(), bi.data())) {
        ok = false;
        break;
      }
      std::uint8_t* oc = codes + r * K;
      std::uint16_t* os = o.scales.data() + r * G;
      std::uint16_t* ob = o.biases.data() + r * G;
      if (s.vcols != 0) {
        const int U = s.vunit, Ug = U / kW8Group;
        for (int h = 0; h < _nv; ++h) {
          const int f = v_src_(h);
          std::memcpy(oc + (std::size_t)h * U, c.data() + (std::size_t)f * U,
                      (std::size_t)U);
          std::memcpy(os + (std::size_t)h * Ug,
                      sc.data() + (std::size_t)f * Ug, (std::size_t)Ug * 2);
          std::memcpy(ob + (std::size_t)h * Ug,
                      bi.data() + (std::size_t)f * Ug, (std::size_t)Ug * 2);
        }
      } else {
        std::memcpy(oc, c.data(), (std::size_t)K);
        std::memcpy(os, sc.data(), (std::size_t)G * 2);
        std::memcpy(ob, bi.data(), (std::size_t)G * 2);
      }
    }
  });
  return ok;
}

bool
GgufQwen4ExpConverter::convert(const ConvertedTensorSpec& spec,
                               std::uint8_t* dst)
{
  const auto it = _src.find(spec.hf_name);
  if (it == _src.end()) { return false; }
  const Src& s = it->second;
  switch (s.kind) {
    case Kind::kI64: {
      const auto v = _g->get_int_array(s.key);
      if (v.size() * 8 != spec.nbytes) { return false; }
      std::memcpy(dst, v.data(), spec.nbytes);
      return true;
    }
    case Kind::kF32: {
      const GgufFile::Tensor* t = _g->tensor(s.gguf.front());
      if (t == nullptr) { return false; }
      const std::int64_t n = t->numel();
      std::vector<float> v((std::size_t)n);
      if (!_g->dequant_all_f32(*t, v.data())) { return false; }
      for (float& f : v) {
        if (s.minus_one) { f -= 1.0f; }
        if (s.a_log) { f = std::log(-f); }
      }
      auto* o = reinterpret_cast<float*>(dst);
      std::memcpy(o, v.data(), (std::size_t)n * 4);
      if (s.vunit > 0) {
        // Rows of `per` values (a conv channel's taps; 1 for a per-head
        // vector), v-heads of `vunit` rows from row `vrows`.
        const std::int64_t per = t->dims.size() == 2 ? t->dims[0] : 1;
        for (int h = 0; h < _nv; ++h) {
          std::memcpy(o + (s.vrows + (std::int64_t)h * s.vunit) * per,
                      v.data() +
                          (s.vrows + (std::int64_t)v_src_(h) * s.vunit) * per,
                      (std::size_t)(s.vunit * per) * 4);
        }
      }
      return true;
    }
    case Kind::kW8: {
      const std::string base =
          spec.hf_name.substr(0, spec.hf_name.rfind('.'));
      W8& w = _w8[base];
      if (s.part == 0) {
        // The codes, written where they go; the scales and biases kept
        // for their own requests.
        if ((std::uint64_t)spec.shape[0] * spec.shape[1] * 4 !=
                spec.nbytes ||
            !build_w8_(s, w, dst)) {
          _w8.erase(base);
          return false;
        }
      } else {
        if (w.scales.empty() && !build_w8_(s, w, nullptr)) {
          _w8.erase(base);
          return false;
        }
        const auto& v = s.part == 1 ? w.scales : w.biases;
        if (v.size() * 2 != spec.nbytes) { return false; }
        std::memcpy(dst, v.data(), spec.nbytes);
      }
      w.served |= 1 << s.part;
      if (w.served == 7) { _w8.erase(base); }
      return true;
    }
    case Kind::kExpert: {
      const GgufFile::Tensor* t = _g->tensor(s.gguf.front());
      if (t == nullptr || t->data == nullptr) { return false; }
      const std::int64_t K = t->dims[0], N = t->dims[1], E = t->dims[2];
      const std::int64_t rb = row_bytes_(t->type, K);
      const std::int64_t wrow = K * s.bits / 8, grow = K / kXGroup;
      std::atomic<bool> ok{true};
      for_rows_((std::size_t)(E * N), [&](std::size_t lo, std::size_t hi) {
        std::vector<std::uint8_t> w((std::size_t)wrow);
        std::vector<std::uint16_t> sc((std::size_t)grow), bi((std::size_t)grow);
        for (std::size_t r = lo; r < hi; ++r) {
          if (!expert_row_(t->type, t->data + (std::int64_t)r * rb, (int)K,
                           s.bits, /*bf16=*/true, w.data(), sc.data(),
                           bi.data())) {
            ok = false;
            return;
          }
          if (s.part == 0) {
            std::memcpy(dst + r * wrow, w.data(), (std::size_t)wrow);
          } else {
            std::memcpy(dst + r * grow * 2,
                        (s.part == 1 ? sc : bi).data(),
                        (std::size_t)grow * 2);
          }
        }
      });
      return ok;
    }
  }
  return false;
}

// ---- the expert cache's fill ---------------------------------------------

bool
qwen4exp_expert_to_affine(std::uint32_t gate_type, const std::uint8_t* gate,
                          std::uint32_t up_type, const std::uint8_t* up,
                          std::uint32_t down_type, const std::uint8_t* down,
                          int I, int H, int gb, int db, bool bf16,
                          std::uint8_t* const* dst)
{
  const std::int64_t grb = row_bytes_(gate_type, H);
  const std::int64_t urb = row_bytes_(up_type, H);
  const std::int64_t drb = row_bytes_(down_type, I);
  if (grb == 0 || urb == 0 || drb == 0) { return false; }
  const std::size_t gw = (std::size_t)H * gb / 8, gg = (std::size_t)H / 32;
  const std::size_t dw = (std::size_t)I * db / 8, dg = (std::size_t)I / 32;
  auto* guw = dst[0];
  auto* gus = reinterpret_cast<std::uint16_t*>(dst[1]);
  auto* gub = reinterpret_cast<std::uint16_t*>(dst[2]);
  // gate | up interleaved by row: row 2g gate's g, row 2g + 1 up's g.
  for (int g = 0; g < I; ++g) {
    const std::size_t r0 = 2 * (std::size_t)g, r1 = r0 + 1;
    if (!expert_row_(gate_type, gate + g * grb, H, gb, bf16,
                     guw + r0 * gw, gus + r0 * gg, gub + r0 * gg) ||
        !expert_row_(up_type, up + g * urb, H, gb, bf16, guw + r1 * gw,
                     gus + r1 * gg, gub + r1 * gg)) {
      return false;
    }
  }
  auto* d_w = dst[3];
  auto* d_s = reinterpret_cast<std::uint16_t*>(dst[4]);
  auto* d_b = reinterpret_cast<std::uint16_t*>(dst[5]);
  for (int r = 0; r < H; ++r) {
    if (!expert_row_(down_type, down + r * drb, I, db, bf16,
                     d_w + (std::size_t)r * dw, d_s + (std::size_t)r * dg,
                     d_b + (std::size_t)r * dg)) {
      return false;
    }
  }
  return true;
}

bool
qwen4exp_rows(std::uint32_t type, const std::uint8_t* src, std::int64_t n,
              int dim, bool bf16, std::uint16_t* out)
{
  const std::int64_t rb = row_bytes_(type, dim);
  if (rb == 0) { return false; }
  thread_local std::vector<float> row;
  row.resize((std::size_t)dim);
  for (std::int64_t r = 0; r < n; ++r) {
    if (!ggml::decode(type, src + r * rb, dim, row.data())) { return false; }
    for (int j = 0; j < dim; ++j) {
      out[r * dim + j] = elt_bits_(row[(std::size_t)j], bf16);
    }
  }
  return true;
}

}  // namespace vpipe::genai
